/*
 * napi_api.c -- N-API addon wrapping polygrad C core for Node.js.
 *
 * Exposes the public C core as native bindings. Opaque pointers are passed as
 * napi_external values. Int64 arrays are read from JS number[] element-by-
 * element. Shape-returning ops return objects.
 */

#include <node_api.h>
#include <string.h>
#include <stdlib.h>

#include "polygrad.h"
#include "frontend.h"
#include "device.h"
#include "tensor.h"
#include "nn.h"
#include "optim.h"
#include "model.h"
#include "tokenizer.h"
#include "loaders/hf_decode.h"
#include "loaders/gguf_decode.h"
#include "loaders/import_error.h"
#include "bundle.h"
#include "models/mlp.h"
#include "models/compose.h"
#include "models/tabm.h"
#include "models/nam.h"
#include "engine/schedule.h"
#include "engine/realize.h"

/* ── Error-checking macro ──────────────────────────────────────────────── */

#define NAPI_CALL(env, call)                                                                       \
  do {                                                                                             \
    napi_status _s = (call);                                                                       \
    if (_s != napi_ok) {                                                                           \
      const napi_extended_error_info *_ei;                                                         \
      napi_get_last_error_info((env), &_ei);                                                       \
      napi_throw_error((env), NULL, _ei->error_message ? _ei->error_message : "N-API error");      \
      return NULL;                                                                                 \
    }                                                                                              \
  } while (0)

/* ── Helpers ───────────────────────────────────────────────────────────── */

static void *get_external(napi_env env, napi_value val) {
  void *ptr = NULL;
  napi_get_value_external(env, val, &ptr);
  return ptr;
}

static void *get_external_nullable(napi_env env, napi_value val) {
  napi_valuetype ty;
  if (napi_typeof(env, val, &ty) != napi_ok) return NULL;
  if (ty == napi_null || ty == napi_undefined) return NULL;
  return get_external(env, val);
}

static napi_value make_external(napi_env env, void *ptr) {
  napi_value result;
  if (!ptr) {
    NAPI_CALL(env, napi_get_null(env, &result));
    return result;
  }
  NAPI_CALL(env, napi_create_external(env, ptr, NULL, NULL, &result));
  return result;
}

static napi_value make_external_pair(napi_env env, void *a, void *b) {
  napi_value out;
  NAPI_CALL(env, napi_create_array_with_length(env, 2, &out));
  napi_value av = make_external(env, a);
  napi_value bv = make_external(env, b);
  NAPI_CALL(env, napi_set_element(env, out, 0, av));
  NAPI_CALL(env, napi_set_element(env, out, 1, bv));
  return out;
}

static void set_named_size(napi_env env, napi_value obj, const char *name, size_t value) {
  napi_value v;
  napi_create_double(env, (double)value, &v);
  napi_set_named_property(env, obj, name, v);
}

static void set_named_u64(napi_env env, napi_value obj, const char *name, uint64_t value) {
  napi_value v;
  napi_create_double(env, (double)value, &v);
  napi_set_named_property(env, obj, name, v);
}

static void set_named_double(napi_env env, napi_value obj, const char *name, double value) {
  napi_value v;
  napi_create_double(env, value, &v);
  napi_set_named_property(env, obj, name, v);
}

static napi_env g_frontend_buffer_release_env = NULL;
static napi_ref g_frontend_buffer_release_ref = NULL;

static void napi_frontend_buffer_release(uintptr_t buffer_key) {
  if (!g_frontend_buffer_release_env || !g_frontend_buffer_release_ref) return;

  napi_handle_scope scope;
  if (napi_open_handle_scope(g_frontend_buffer_release_env, &scope) != napi_ok) return;

  napi_value fn, global, arg, result;
  if (napi_get_reference_value(g_frontend_buffer_release_env, g_frontend_buffer_release_ref, &fn) !=
      napi_ok)
    goto done;
  if (napi_get_global(g_frontend_buffer_release_env, &global) != napi_ok) goto done;
  if (napi_create_bigint_uint64(g_frontend_buffer_release_env, (uint64_t)buffer_key, &arg) !=
      napi_ok)
    goto done;
  napi_call_function(g_frontend_buffer_release_env, global, fn, 1, &arg, &result);

done:
  napi_close_handle_scope(g_frontend_buffer_release_env, scope);
}

static int read_int64_array(napi_env env, napi_value arr, int64_t *out, int max_len) {
  uint32_t len = 0;
  napi_get_array_length(env, arr, &len);
  if ((int)len > max_len) len = (uint32_t)max_len;
  for (uint32_t i = 0; i < len; i++) {
    napi_value elem;
    napi_get_element(env, arr, i, &elem);
    napi_get_value_int64(env, elem, &out[i]);
  }
  return (int)len;
}

static PolyUOp **read_uop_array(napi_env env, napi_value arr, int *out_n) {
  bool is_arr = false;
  napi_is_array(env, arr, &is_arr);
  if (!is_arr) {
    napi_throw_type_error(env, NULL, "polygrad: expected UOp array");
    return NULL;
  }
  uint32_t n = 0;
  napi_get_array_length(env, arr, &n);
  if (n == 0) {
    if (out_n) *out_n = 0;
    return (PolyUOp **)calloc(1, sizeof(PolyUOp *));
  }
  PolyUOp **out = (PolyUOp **)malloc((size_t)n * sizeof(PolyUOp *));
  if (!out) {
    napi_throw_error(env, NULL, "malloc failed");
    return NULL;
  }
  for (uint32_t i = 0; i < n; i++) {
    napi_value elem;
    napi_get_element(env, arr, i, &elem);
    out[i] = get_external(env, elem);
  }
  if (out_n) *out_n = (int)n;
  return out;
}

static napi_value make_shape_result(napi_env env, void *uop_ptr, int64_t *shape, int ndim) {
  napi_value obj, uop_val, shape_arr;
  NAPI_CALL(env, napi_create_object(env, &obj));
  bool valid = uop_ptr && ndim >= 0 && ndim <= POLY_MAX_DIMS && (ndim == 0 || shape);
  if (valid)
    NAPI_CALL(env, napi_create_external(env, uop_ptr, NULL, NULL, &uop_val));
  else
    NAPI_CALL(env, napi_get_null(env, &uop_val));
  int shape_ndim = valid ? ndim : 0;
  NAPI_CALL(env, napi_create_array_with_length(env, (size_t)shape_ndim, &shape_arr));
  for (int i = 0; i < shape_ndim; i++) {
    napi_value v;
    NAPI_CALL(env, napi_create_int64(env, shape[i], &v));
    NAPI_CALL(env, napi_set_element(env, shape_arr, (uint32_t)i, v));
  }
  NAPI_CALL(env, napi_set_named_property(env, obj, "uop", uop_val));
  NAPI_CALL(env, napi_set_named_property(env, obj, "shape", shape_arr));
  return obj;
}

static char *read_utf8_arg(napi_env env, napi_value val, size_t *out_len) {
  size_t len = 0;
  NAPI_CALL(env, napi_get_value_string_utf8(env, val, NULL, 0, &len));
  char *buf = malloc(len + 1);
  if (!buf) {
    napi_throw_error(env, NULL, "malloc failed");
    return NULL;
  }
  NAPI_CALL(env, napi_get_value_string_utf8(env, val, buf, len + 1, &len));
  if (out_len) *out_len = len;
  return buf;
}

static size_t typedarray_element_size(napi_typedarray_type type) {
  switch (type) {
  case napi_int8_array:
  case napi_uint8_array:
  case napi_uint8_clamped_array:
    return 1;
  case napi_int16_array:
  case napi_uint16_array:
    return 2;
  case napi_int32_array:
  case napi_uint32_array:
  case napi_float32_array:
    return 4;
  case napi_float64_array:
  case napi_bigint64_array:
  case napi_biguint64_array:
    return 8;
  default:
    return 0;
  }
}

static int read_bytes_arg(napi_env env, napi_value val, void **data, size_t *nbytes) {
  bool is_buffer = false;
  if (napi_is_buffer(env, val, &is_buffer) == napi_ok && is_buffer) {
    if (napi_get_buffer_info(env, val, data, nbytes) != napi_ok) {
      napi_throw_error(env, NULL, "polygrad: failed to read Buffer data");
      return 0;
    }
    return 1;
  }

  bool is_typedarray = false;
  if (napi_is_typedarray(env, val, &is_typedarray) == napi_ok && is_typedarray) {
    napi_typedarray_type type;
    size_t length = 0, byte_offset = 0;
    napi_value arraybuffer;
    if (napi_get_typedarray_info(env, val, &type, &length, data, &arraybuffer, &byte_offset) !=
        napi_ok) {
      napi_throw_error(env, NULL, "polygrad: failed to read TypedArray data");
      return 0;
    }
    size_t itemsize = typedarray_element_size(type);
    if (!itemsize) {
      napi_throw_type_error(env, NULL, "polygrad: unsupported TypedArray type");
      return 0;
    }
    *nbytes = length * itemsize;
    return 1;
  }

  bool is_arraybuffer = false;
  if (napi_is_arraybuffer(env, val, &is_arraybuffer) == napi_ok && is_arraybuffer) {
    if (napi_get_arraybuffer_info(env, val, data, nbytes) != napi_ok) {
      napi_throw_error(env, NULL, "polygrad: failed to read ArrayBuffer data");
      return 0;
    }
    return 1;
  }

  napi_throw_type_error(env, NULL, "polygrad: expected Buffer, TypedArray, or ArrayBuffer");
  return 0;
}

static int instance_napi_storage_type(
    int dtype_id,
    napi_typedarray_type *type_out,
    size_t *itemsize_out
) {
  if (!type_out || !itemsize_out) return 0;
#define DTYPE_CASE(name, napi_type, bytes)                                                         \
  if (dtype_id == poly_dtype_id_by_name(name)) {                                                   \
    *type_out = napi_type;                                                                         \
    *itemsize_out = bytes;                                                                         \
    return 1;                                                                                      \
  }
  DTYPE_CASE("bool", napi_uint8_array, 1)
  DTYPE_CASE("int8", napi_int8_array, 1)
  DTYPE_CASE("uint8", napi_uint8_array, 1)
  DTYPE_CASE("int16", napi_int16_array, 2)
  DTYPE_CASE("uint16", napi_uint16_array, 2)
  DTYPE_CASE("int32", napi_int32_array, 4)
  DTYPE_CASE("uint32", napi_uint32_array, 4)
  DTYPE_CASE("int64", napi_bigint64_array, 8)
  DTYPE_CASE("uint64", napi_biguint64_array, 8)
  /* JavaScript has no baseline Float16Array/BFloat16Array. Preserve mutable
   * exact storage bits, as the C API does, instead of mislabelling them F32. */
  DTYPE_CASE("float16", napi_uint16_array, 2)
  DTYPE_CASE("bfloat16", napi_uint16_array, 2)
  DTYPE_CASE("fp8e4m3", napi_uint8_array, 1)
  DTYPE_CASE("fp8e5m2", napi_uint8_array, 1)
  DTYPE_CASE("fp8e4m3fnuz", napi_uint8_array, 1)
  DTYPE_CASE("fp8e5m2fnuz", napi_uint8_array, 1)
  DTYPE_CASE("float32", napi_float32_array, 4)
  DTYPE_CASE("float64", napi_float64_array, 8)
#undef DTYPE_CASE
  return 0;
}

static napi_value read_model_storage_array(napi_env env, PolyModel *model, int index) {
  if (index < 0 || index >= poly_model_buf_count(model)) {
    napi_value result;
    napi_get_null(env, &result);
    return result;
  }
  napi_typedarray_type type;
  size_t itemsize = 0;
  int dtype_id = poly_model_buf_dtype_id(model, index);
  if (!instance_napi_storage_type(dtype_id, &type, &itemsize)) {
    napi_throw_error(env, NULL, "polygrad: unsupported Model storage dtype");
    return NULL;
  }
  size_t nbytes = poly_model_buf_nbytes(model, index);
  size_t len = nbytes / itemsize;
  void *dst = NULL;
  napi_value arraybuf, typed;
  NAPI_CALL(env, napi_create_arraybuffer(env, len * itemsize, &dst, &arraybuf));
  if (poly_model_read_buf(model, index, dst, nbytes) != 0) {
    napi_throw_error(env, NULL, "polygrad: Model buffer read failed");
    return NULL;
  }
  NAPI_CALL(env, napi_create_typedarray(env, type, len, arraybuf, 0, &typed));
  return typed;
}

static napi_value make_uint8_array_copy(napi_env env, const uint8_t *src, size_t len) {
  if (!src) {
    napi_value result;
    napi_get_null(env, &result);
    return result;
  }

  void *dst = NULL;
  napi_value arraybuf, typed;
  NAPI_CALL(env, napi_create_arraybuffer(env, len, &dst, &arraybuf));
  memcpy(dst, src, len);
  NAPI_CALL(env, napi_create_typedarray(env, napi_uint8_array, len, arraybuf, 0, &typed));
  return typed;
}

static int read_io_bindings(
    napi_env env,
    napi_value names_val,
    napi_value arrays_val,
    PolyIOBinding **out_bindings,
    char ***out_names,
    int *out_n
) {
  uint32_t n_names = 0, n_arrays = 0;
  napi_get_array_length(env, names_val, &n_names);
  napi_get_array_length(env, arrays_val, &n_arrays);
  if (n_names != n_arrays) {
    napi_throw_error(env, NULL, "polygrad: binding names/data length mismatch");
    return 0;
  }

  PolyIOBinding *bindings = calloc(n_names ? n_names : 1, sizeof(PolyIOBinding));
  char **names = calloc(n_names ? n_names : 1, sizeof(char *));
  if (!bindings || !names) {
    free(bindings);
    free(names);
    napi_throw_error(env, NULL, "malloc failed");
    return 0;
  }

  for (uint32_t i = 0; i < n_names; i++) {
    napi_value name_val, data_val;
    napi_get_element(env, names_val, i, &name_val);
    napi_get_element(env, arrays_val, i, &data_val);

    names[i] = read_utf8_arg(env, name_val, NULL);
    if (!names[i]) {
      for (uint32_t j = 0; j < i; j++)
        free(names[j]);
      free(names);
      free(bindings);
      return 0;
    }

    napi_typedarray_type type;
    size_t length;
    void *data = NULL;
    size_t byte_offset;
    napi_value arraybuf;
    napi_status status =
        napi_get_typedarray_info(env, data_val, &type, &length, &data, &arraybuf, &byte_offset);
    if (status != napi_ok) {
      for (uint32_t j = 0; j <= i; j++)
        free(names[j]);
      free(names);
      free(bindings);
      {
        const napi_extended_error_info *ei = NULL;
        napi_get_last_error_info(env, &ei);
        napi_throw_error(env, NULL, ei && ei->error_message ? ei->error_message : "N-API error");
      }
      return 0;
    }
    const char *dtype_name = NULL;
    size_t itemsize = 0;
    switch (type) {
    case napi_int8_array:
      dtype_name = "int8";
      itemsize = 1;
      break;
    case napi_uint8_array:
    case napi_uint8_clamped_array:
      dtype_name = "uint8";
      itemsize = 1;
      break;
    case napi_int16_array:
      dtype_name = "int16";
      itemsize = 2;
      break;
    case napi_uint16_array:
      dtype_name = "uint16";
      itemsize = 2;
      break;
    case napi_int32_array:
      dtype_name = "int32";
      itemsize = 4;
      break;
    case napi_uint32_array:
      dtype_name = "uint32";
      itemsize = 4;
      break;
    case napi_bigint64_array:
      dtype_name = "int64";
      itemsize = 8;
      break;
    case napi_biguint64_array:
      dtype_name = "uint64";
      itemsize = 8;
      break;
    case napi_float32_array:
      dtype_name = "float32";
      itemsize = 4;
      break;
    case napi_float64_array:
      dtype_name = "float64";
      itemsize = 8;
      break;
    default:
      break;
    }
    int dtype_id = dtype_name ? poly_dtype_id_by_name(dtype_name) : -1;
    if (dtype_id < 0 || itemsize == 0 || length > SIZE_MAX / itemsize) {
      for (uint32_t j = 0; j <= i; j++)
        free(names[j]);
      free(names);
      free(bindings);
      napi_throw_error(env, NULL, "polygrad: unsupported Model binding TypedArray");
      return 0;
    }

    bindings[i].name = names[i];
    bindings[i].data = data;
    bindings[i].nbytes = length * itemsize;
    bindings[i].dtype_id = dtype_id;
  }

  *out_bindings = bindings;
  *out_names = names;
  *out_n = (int)n_names;
  return 1;
}

static void free_io_bindings(char **names, PolyIOBinding *bindings, int n) {
  if (names) {
    for (int i = 0; i < n; i++)
      free(names[i]);
    free(names);
  }
  free(bindings);
}

/* ── Macros for repetitive wrappers ────────────────────────────────────── */

/* Unary: (ctx, x) -> external */
#define NAPI_UNARY(cname)                                                                          \
  static napi_value napi_##cname(napi_env env, napi_callback_info info) {                          \
    napi_value argv[2];                                                                            \
    size_t argc = 2;                                                                               \
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));                          \
    PolyCtx *ctx = get_external(env, argv[0]);                                                     \
    PolyUOp *x = get_external(env, argv[1]);                                                       \
    PolyUOp *r = cname(ctx, x);                                                                    \
    return make_external(env, r);                                                                  \
  }

/* Binary: (ctx, a, b) -> external */
#define NAPI_BINARY(cname)                                                                         \
  static napi_value napi_##cname(napi_env env, napi_callback_info info) {                          \
    napi_value argv[3];                                                                            \
    size_t argc = 3;                                                                               \
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));                          \
    PolyCtx *ctx = get_external(env, argv[0]);                                                     \
    PolyUOp *a = get_external(env, argv[1]);                                                       \
    PolyUOp *b = get_external(env, argv[2]);                                                       \
    PolyUOp *r = cname(ctx, a, b);                                                                 \
    return make_external(env, r);                                                                  \
  }

/* ── Context ───────────────────────────────────────────────────────────── */

static napi_value napi_poly_ctx_new(napi_env env, napi_callback_info info) {
  (void)info;
  PolyCtx *ctx = poly_ctx_new();
  return make_external(env, ctx);
}

static napi_value napi_poly_ctx_destroy(napi_env env, napi_callback_info info) {
  napi_value argv[1];
  size_t argc = 1;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  poly_ctx_destroy(ctx);
  napi_value undef;
  napi_get_undefined(env, &undef);
  return undef;
}

static napi_value napi_poly_ctx_named_count(napi_env env, napi_callback_info info) {
  napi_value argv[1];
  size_t argc = 1;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  napi_value out;
  NAPI_CALL(env, napi_create_int32(env, poly_ctx_named_count(ctx), &out));
  return out;
}

/* ── Constants ─────────────────────────────────────────────────────────── */

static napi_value napi_poly_const_float(napi_env env, napi_callback_info info) {
  napi_value argv[2];
  size_t argc = 2;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  double val;
  napi_get_value_double(env, argv[1], &val);
  return make_external(env, poly_const_float(ctx, val));
}

static napi_value napi_poly_const_double(napi_env env, napi_callback_info info) {
  napi_value argv[2];
  size_t argc = 2;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  double val;
  napi_get_value_double(env, argv[1], &val);
  return make_external(env, poly_const_double(ctx, val));
}

static napi_value napi_poly_const_int(napi_env env, napi_callback_info info) {
  napi_value argv[2];
  size_t argc = 2;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  int64_t val;
  napi_get_value_int64(env, argv[1], &val);
  return make_external(env, poly_const_int(ctx, val));
}

static napi_value napi_poly_const_float_by_id(napi_env env, napi_callback_info info) {
  napi_value argv[3];
  size_t argc = 3;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  double val;
  int32_t dtype_id;
  napi_get_value_double(env, argv[1], &val);
  napi_get_value_int32(env, argv[2], &dtype_id);
  return make_external(env, poly_const_float_by_id(ctx, val, dtype_id));
}

static napi_value napi_poly_const_int_by_id(napi_env env, napi_callback_info info) {
  napi_value argv[3];
  size_t argc = 3;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  int64_t val;
  int32_t dtype_id;
  napi_get_value_int64(env, argv[1], &val);
  napi_get_value_int32(env, argv[2], &dtype_id);
  return make_external(env, poly_const_int_by_id(ctx, val, dtype_id));
}

/* ── ALU ops ───────────────────────────────────────────────────────────── */

static napi_value napi_poly_contiguous(napi_env env, napi_callback_info info) {
  napi_value argv[2];
  size_t argc = 2;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *x = get_external(env, argv[1]);
  return make_external(env, poly_contiguous(ctx, x));
}

static napi_value napi_poly_alu1(napi_env env, napi_callback_info info) {
  napi_value argv[3];
  size_t argc = 3;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  int32_t op;
  napi_get_value_int32(env, argv[1], &op);
  PolyUOp *src = get_external(env, argv[2]);
  return make_external(env, poly_alu1(ctx, (PolyOps)op, src));
}

static napi_value napi_poly_alu2(napi_env env, napi_callback_info info) {
  napi_value argv[4];
  size_t argc = 4;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  int32_t op;
  napi_get_value_int32(env, argv[1], &op);
  PolyUOp *a = get_external(env, argv[2]);
  PolyUOp *b = get_external(env, argv[3]);
  return make_external(env, poly_alu2(ctx, (PolyOps)op, a, b));
}

static napi_value napi_poly_binop(napi_env env, napi_callback_info info) {
  napi_value argv[4];
  size_t argc = 4;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  int32_t op;
  napi_get_value_int32(env, argv[1], &op);
  PolyUOp *a = get_external(env, argv[2]);
  PolyUOp *b = get_external(env, argv[3]);
  return make_external(env, poly_binop(ctx, (PolyOps)op, a, b));
}

static napi_value napi_poly_alu3(napi_env env, napi_callback_info info) {
  napi_value argv[5];
  size_t argc = 5;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  int32_t op;
  napi_get_value_int32(env, argv[1], &op);
  PolyUOp *a = get_external(env, argv[2]);
  PolyUOp *b = get_external(env, argv[3]);
  PolyUOp *c = get_external(env, argv[4]);
  return make_external(env, poly_alu3(ctx, (PolyOps)op, a, b, c));
}

/* ── Graph construction ────────────────────────────────────────────────── */

static napi_value napi_poly_store_val(napi_env env, napi_callback_info info) {
  napi_value argv[3];
  size_t argc = 3;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *buf = get_external(env, argv[1]);
  PolyUOp *val = get_external(env, argv[2]);
  return make_external(env, poly_store_val(ctx, buf, val));
}

static napi_value napi_poly_sink1(napi_env env, napi_callback_info info) {
  napi_value argv[2];
  size_t argc = 2;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *store = get_external(env, argv[1]);
  return make_external(env, poly_sink1(ctx, store));
}

static napi_value napi_poly_uop_placeholder_like(napi_env env, napi_callback_info info) {
  napi_value argv[3];
  size_t argc = 3;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *like = get_external(env, argv[1]);
  int32_t slot = 0;
  napi_get_value_int32(env, argv[2], &slot);
  return make_external(env, poly_uop_placeholder_like(ctx, like, slot));
}

static napi_value napi_poly_uop_range(napi_env env, napi_callback_info info) {
  napi_value argv[4];
  size_t argc = 4;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  int64_t bound = 0, axis_id = 0;
  int32_t axis_type = 0;
  napi_get_value_int64(env, argv[1], &bound);
  napi_get_value_int64(env, argv[2], &axis_id);
  napi_get_value_int32(env, argv[3], &axis_type);
  return make_external(env, poly_uop_range(ctx, bound, axis_id, axis_type));
}

static napi_value napi_poly_uop_index(napi_env env, napi_callback_info info) {
  napi_value argv[4];
  size_t argc = 4;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *base = get_external(env, argv[1]);
  int n = 0;
  PolyUOp **indices = read_uop_array(env, argv[2], &n);
  if (!indices) return NULL;
  PolyUOp *ret = poly_uop_index(ctx, base, indices, n);
  free(indices);
  return make_external(env, ret);
}

static napi_value napi_poly_uop_load(napi_env env, napi_callback_info info) {
  napi_value argv[2];
  size_t argc = 2;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  return make_external(env, poly_uop_load(get_external(env, argv[0]), get_external(env, argv[1])));
}

static napi_value napi_poly_uop_store(napi_env env, napi_callback_info info) {
  napi_value argv[3];
  size_t argc = 3;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  return make_external(
      env, poly_uop_store(
               get_external(env, argv[0]), get_external(env, argv[1]), get_external(env, argv[2])
           )
  );
}

static napi_value napi_poly_uop_set(napi_env env, napi_callback_info info) {
  napi_value argv[4];
  size_t argc = 4;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *addr = get_external(env, argv[1]);
  PolyUOp *value = get_external(env, argv[2]);
  int n = 0;
  PolyUOp **ranges = read_uop_array(env, argv[3], &n);
  if (!ranges && n != 0) return NULL;
  PolyUOp *ret = poly_uop_set(ctx, addr, value, ranges, n);
  free(ranges);
  return make_external(env, ret);
}

static napi_value napi_poly_uop_group(napi_env env, napi_callback_info info) {
  napi_value argv[2];
  size_t argc = 2;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  int n = 0;
  PolyUOp **srcs = read_uop_array(env, argv[1], &n);
  if (!srcs) return NULL;
  PolyUOp *ret = poly_uop_group(ctx, srcs, n);
  free(srcs);
  return make_external(env, ret);
}

static napi_value napi_poly_uop_end(napi_env env, napi_callback_info info) {
  napi_value argv[3];
  size_t argc = 3;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *body = get_external(env, argv[1]);
  int n = 0;
  PolyUOp **ranges = read_uop_array(env, argv[2], &n);
  if (!ranges) return NULL;
  PolyUOp *ret = poly_uop_end(ctx, body, ranges, n);
  free(ranges);
  return make_external(env, ret);
}

static napi_value napi_poly_uop_sink(napi_env env, napi_callback_info info) {
  napi_value argv[2];
  size_t argc = 2;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  int n = 0;
  PolyUOp **srcs = read_uop_array(env, argv[1], &n);
  if (!srcs) return NULL;
  PolyUOp *ret = poly_uop_sink(ctx, srcs, n);
  free(srcs);
  return make_external(env, ret);
}

static napi_value napi_poly_uop_sink_ex(napi_env env, napi_callback_info info) {
  napi_value argv[4];
  size_t argc = 4;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  int n = 0;
  PolyUOp **srcs = read_uop_array(env, argv[1], &n);
  if (!srcs) return NULL;
  char *name = NULL;
  if (argc > 2) {
    napi_valuetype ty;
    NAPI_CALL(env, napi_typeof(env, argv[2], &ty));
    if (ty == napi_string) name = read_utf8_arg(env, argv[2], NULL);
  }
  bool optimize = true;
  if (argc > 3) napi_get_value_bool(env, argv[3], &optimize);
  PolyUOp *ret = poly_uop_sink_ex(ctx, srcs, n, name, optimize ? 1 : 0);
  free(name);
  free(srcs);
  return make_external(env, ret);
}

static napi_value napi_poly_uop_call(napi_env env, napi_callback_info info) {
  napi_value argv[3];
  size_t argc = 3;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *body = get_external(env, argv[1]);
  int n = 0;
  PolyUOp **args = read_uop_array(env, argv[2], &n);
  if (!args) return NULL;
  PolyUOp *ret = poly_uop_call(ctx, body, args, n);
  free(args);
  return make_external(env, ret);
}

static napi_value napi_poly_uop_after(napi_env env, napi_callback_info info) {
  napi_value argv[3];
  size_t argc = 3;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  return make_external(
      env, poly_uop_after(
               get_external(env, argv[0]), get_external(env, argv[1]), get_external(env, argv[2])
           )
  );
}

static napi_value napi_poly_uop_reduce(napi_env env, napi_callback_info info) {
  napi_value argv[4];
  size_t argc = 4;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  int32_t op = 0;
  napi_get_value_int32(env, argv[1], &op);
  PolyUOp *expr = get_external(env, argv[2]);
  int n = 0;
  PolyUOp **ranges = read_uop_array(env, argv[3], &n);
  if (!ranges) return NULL;
  PolyUOp *ret = poly_uop_reduce(ctx, (PolyOps)op, expr, ranges, n);
  free(ranges);
  return make_external(env, ret);
}

static napi_value napi_poly_uop_flatten(napi_env env, napi_callback_info info) {
  napi_value argv[2];
  size_t argc = 2;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  return make_external(
      env, poly_uop_flatten(get_external(env, argv[0]), get_external(env, argv[1]))
  );
}

static napi_value napi_poly_uop_numel(napi_env env, napi_callback_info info) {
  napi_value argv[2];
  size_t argc = 2;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  int64_t numel = poly_uop_numel(get_external(env, argv[0]), get_external(env, argv[1]));
  napi_value out;
  NAPI_CALL(env, napi_create_int64(env, numel, &out));
  return out;
}

static napi_value napi_poly_register_buffer_by_id(napi_env env, napi_callback_info info) {
  napi_value argv[6];
  size_t argc = 6;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  int32_t role = 0, dtype_id = 0;
  napi_get_value_int32(env, argv[1], &role);
  napi_get_value_int32(env, argv[2], &dtype_id);
  int64_t shape[8];
  int ndim = read_int64_array(env, argv[3], shape, 8);
  char *name = read_utf8_arg(env, argv[4], NULL);
  if (!name) return NULL;
  PolyUOp *ret = poly_register_buffer_by_id(ctx, role, dtype_id, shape, ndim, name);
  free(name);
  return make_external(env, ret);
}

static napi_value napi_poly_register_existing_buffer(napi_env env, napi_callback_info info) {
  napi_value argv[6];
  size_t argc = 6;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  int32_t role = 0;
  napi_get_value_int32(env, argv[1], &role);
  PolyUOp *buffer = get_external(env, argv[2]);
  int64_t shape[8];
  int ndim = read_int64_array(env, argv[3], shape, 8);
  char *name = read_utf8_arg(env, argv[4], NULL);
  bool trainable = false;
  napi_get_value_bool(env, argv[5], &trainable);
  if (!name) return NULL;
  PolyUOp *ret = poly_register_existing_buffer(ctx, role, buffer, shape, ndim, name, trainable);
  free(name);
  return make_external(env, ret);
}

/* ── Buffers ───────────────────────────────────────────────────────────── */

static napi_value napi_poly_buffer_by_id(napi_env env, napi_callback_info info) {
  napi_value argv[3];
  size_t argc = 3;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  int32_t dtype_id;
  int64_t size;
  napi_get_value_int32(env, argv[1], &dtype_id);
  napi_get_value_int64(env, argv[2], &size);
  return make_external(env, poly_buffer_by_id(ctx, dtype_id, size));
}

static napi_value napi_poly_buffer_on_device_by_id(napi_env env, napi_callback_info info) {
  napi_value argv[4];
  size_t argc = 4;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  int32_t dtype_id, device_id;
  int64_t size;
  napi_get_value_int32(env, argv[1], &dtype_id);
  napi_get_value_int64(env, argv[2], &size);
  napi_get_value_int32(env, argv[3], &device_id);
  return make_external(env, poly_buffer_on_device_by_id(ctx, dtype_id, size, device_id));
}

static napi_value napi_poly_buffer_f32(napi_env env, napi_callback_info info) {
  napi_value argv[2];
  size_t argc = 2;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  int64_t size;
  napi_get_value_int64(env, argv[1], &size);
  return make_external(env, poly_buffer_f32(ctx, size));
}

static napi_value napi_poly_buffer_f64(napi_env env, napi_callback_info info) {
  napi_value argv[2];
  size_t argc = 2;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  int64_t size;
  napi_get_value_int64(env, argv[1], &size);
  return make_external(env, poly_buffer_f64(ctx, size));
}

/* ── Host-backed buffers + side table + graph-driven realize ──────────── */

typedef struct {
  void *data;
  size_t nbytes;
  int dtype_id;
  int64_t dims[16];
  int ndim;
} NapiHostArray;

static bool napi_get_host_array(napi_env env, napi_value *argv, NapiHostArray *out) {
  if (!out) return false;
  memset(out, 0, sizeof(*out));
  napi_typedarray_type ta_type;
  size_t ta_len;
  napi_value ab;
  size_t ab_off;
  if (napi_get_typedarray_info(env, argv[1], &ta_type, &ta_len, &out->data, &ab, &ab_off) !=
      napi_ok)
    return false;
  size_t esz = 4;
  switch (ta_type) {
  case napi_int8_array:
  case napi_uint8_array:
  case napi_uint8_clamped_array:
    esz = 1;
    break;
  case napi_int16_array:
  case napi_uint16_array:
    esz = 2;
    break;
  case napi_int32_array:
  case napi_uint32_array:
  case napi_float32_array:
    esz = 4;
    break;
  case napi_float64_array:
  case napi_biguint64_array:
  case napi_bigint64_array:
    esz = 8;
    break;
  default:
    esz = 4;
  }
  out->nbytes = ta_len * esz;
  if (napi_get_value_int32(env, argv[3], &out->dtype_id) != napi_ok) return false;
  napi_valuetype vt;
  if (napi_typeof(env, argv[4], &vt) != napi_ok) return false;
  if (vt == napi_object) {
    bool is_arr;
    if (napi_is_array(env, argv[4], &is_arr) != napi_ok) return false;
    if (is_arr) {
      uint32_t len;
      if (napi_get_array_length(env, argv[4], &len) != napi_ok) return false;
      if (len > 16) len = 16;
      for (uint32_t i = 0; i < len; i++) {
        napi_value el;
        if (napi_get_element(env, argv[4], i, &el) != napi_ok) return false;
        int64_t v;
        if (napi_get_value_int64(env, el, &v) != napi_ok) return false;
        out->dims[i] = v;
      }
      out->ndim = (int)len;
    }
  }
  return true;
}

static napi_value napi_poly_buffer_from_host(napi_env env, napi_callback_info info) {
  napi_value argv[6];
  size_t argc = 6;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  NapiHostArray host;
  if (!napi_get_host_array(env, argv, &host)) return NULL;
  return make_external(
      env, poly_buffer_from_host(ctx, host.data, host.nbytes, host.dtype_id, host.dims, host.ndim)
  );
}

static napi_value napi_poly_tensor_from_host_by_id(napi_env env, napi_callback_info info) {
  napi_value argv[6];
  size_t argc = 6;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  NapiHostArray host;
  if (!napi_get_host_array(env, argv, &host)) return NULL;
  return make_external(
      env,
      poly_tensor_from_host_by_id(ctx, host.data, host.nbytes, host.dtype_id, host.dims, host.ndim)
  );
}

static napi_value napi_poly_tensor_const_int_by_id(napi_env env, napi_callback_info info) {
  napi_value argv[4];
  size_t argc = 4;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  int64_t value;
  int32_t dtype_id, device_id;
  NAPI_CALL(env, napi_get_value_int64(env, argv[1], &value));
  NAPI_CALL(env, napi_get_value_int32(env, argv[2], &dtype_id));
  NAPI_CALL(env, napi_get_value_int32(env, argv[3], &device_id));
  return make_external(env, poly_tensor_const_int_by_id(ctx, value, dtype_id, device_id));
}

static napi_value napi_poly_tensor_const_float_by_id(napi_env env, napi_callback_info info) {
  napi_value argv[4];
  size_t argc = 4;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  double value;
  int32_t dtype_id, device_id;
  NAPI_CALL(env, napi_get_value_double(env, argv[1], &value));
  NAPI_CALL(env, napi_get_value_int32(env, argv[2], &dtype_id));
  NAPI_CALL(env, napi_get_value_int32(env, argv[3], &device_id));
  return make_external(env, poly_tensor_const_float_by_id(ctx, value, dtype_id, device_id));
}

static napi_value napi_poly_tensor_const_like_int(napi_env env, napi_callback_info info) {
  napi_value argv[3];
  size_t argc = 3;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyTensor *ref = get_external(env, argv[1]);
  int64_t value;
  NAPI_CALL(env, napi_get_value_int64(env, argv[2], &value));
  return make_external(env, poly_tensor_const_like_int(ctx, ref, value));
}

static napi_value napi_poly_tensor_const_like_float(napi_env env, napi_callback_info info) {
  napi_value argv[3];
  size_t argc = 3;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyTensor *ref = get_external(env, argv[1]);
  double value;
  NAPI_CALL(env, napi_get_value_double(env, argv[2], &value));
  return make_external(env, poly_tensor_const_like_float(ctx, ref, value));
}

static napi_value napi_poly_tensor_full_int_by_id(napi_env env, napi_callback_info info) {
  napi_value argv[8];
  size_t argc = 8;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  int64_t dims[POLY_MAX_DIMS], value;
  int32_t ndim, dtype_id, device_id;
  bool dtype_explicit, buffer;
  NAPI_CALL(env, napi_get_value_int32(env, argv[2], &ndim));
  NAPI_CALL(env, napi_get_value_int64(env, argv[3], &value));
  NAPI_CALL(env, napi_get_value_int32(env, argv[4], &dtype_id));
  NAPI_CALL(env, napi_get_value_int32(env, argv[5], &device_id));
  NAPI_CALL(env, napi_get_value_bool(env, argv[6], &dtype_explicit));
  NAPI_CALL(env, napi_get_value_bool(env, argv[7], &buffer));
  if (read_int64_array(env, argv[1], dims, POLY_MAX_DIMS) != ndim) {
    napi_throw_range_error(env, NULL, "polygrad: shape length does not match ndim");
    return NULL;
  }
  return make_external(
      env, poly_tensor_full_int_by_id(
               ctx, dims, ndim, value, dtype_id, device_id, dtype_explicit, buffer
           )
  );
}

static napi_value napi_poly_tensor_full_float_by_id(napi_env env, napi_callback_info info) {
  napi_value argv[8];
  size_t argc = 8;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  int64_t dims[POLY_MAX_DIMS];
  double value;
  int32_t ndim, dtype_id, device_id;
  bool dtype_explicit, buffer;
  NAPI_CALL(env, napi_get_value_int32(env, argv[2], &ndim));
  NAPI_CALL(env, napi_get_value_double(env, argv[3], &value));
  NAPI_CALL(env, napi_get_value_int32(env, argv[4], &dtype_id));
  NAPI_CALL(env, napi_get_value_int32(env, argv[5], &device_id));
  NAPI_CALL(env, napi_get_value_bool(env, argv[6], &dtype_explicit));
  NAPI_CALL(env, napi_get_value_bool(env, argv[7], &buffer));
  if (read_int64_array(env, argv[1], dims, POLY_MAX_DIMS) != ndim) {
    napi_throw_range_error(env, NULL, "polygrad: shape length does not match ndim");
    return NULL;
  }
  return make_external(
      env, poly_tensor_full_float_by_id(
               ctx, dims, ndim, value, dtype_id, device_id, dtype_explicit, buffer
           )
  );
}

static napi_value napi_poly_tensor_arange_int_by_id(napi_env env, napi_callback_info info) {
  napi_value argv[6];
  size_t argc = 6;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  int64_t start, stop, step;
  int32_t dtype_id, device_id;
  NAPI_CALL(env, napi_get_value_int64(env, argv[1], &start));
  NAPI_CALL(env, napi_get_value_int64(env, argv[2], &stop));
  NAPI_CALL(env, napi_get_value_int64(env, argv[3], &step));
  NAPI_CALL(env, napi_get_value_int32(env, argv[4], &dtype_id));
  NAPI_CALL(env, napi_get_value_int32(env, argv[5], &device_id));
  return make_external(
      env, poly_tensor_arange_int_by_id(ctx, start, stop, step, dtype_id, device_id)
  );
}

static napi_value napi_poly_tensor_arange_float_by_id(napi_env env, napi_callback_info info) {
  napi_value argv[6];
  size_t argc = 6;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  double start, stop, step;
  int32_t dtype_id, device_id;
  NAPI_CALL(env, napi_get_value_double(env, argv[1], &start));
  NAPI_CALL(env, napi_get_value_double(env, argv[2], &stop));
  NAPI_CALL(env, napi_get_value_double(env, argv[3], &step));
  NAPI_CALL(env, napi_get_value_int32(env, argv[4], &dtype_id));
  NAPI_CALL(env, napi_get_value_int32(env, argv[5], &device_id));
  return make_external(
      env, poly_tensor_arange_float_by_id(ctx, start, stop, step, dtype_id, device_id)
  );
}

static napi_value napi_poly_tensor_manual_seed(napi_env env, napi_callback_info info) {
  napi_value argv[2];
  size_t argc = 2;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  int64_t seed = 0;
  NAPI_CALL(env, napi_get_value_int64(env, argv[1], &seed));
  poly_tensor_manual_seed(ctx, seed);
  napi_value out;
  NAPI_CALL(env, napi_get_undefined(env, &out));
  return out;
}

static napi_value napi_poly_tensor_rand_by_id(napi_env env, napi_callback_info info) {
  napi_value argv[6];
  size_t argc = 6;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  int64_t dims[POLY_MAX_DIMS];
  int32_t ndim = 0, dtype_id = 0, device = 0, contiguous = 1;
  NAPI_CALL(env, napi_get_value_int32(env, argv[2], &ndim));
  NAPI_CALL(env, napi_get_value_int32(env, argv[3], &dtype_id));
  NAPI_CALL(env, napi_get_value_int32(env, argv[4], &device));
  NAPI_CALL(env, napi_get_value_int32(env, argv[5], &contiguous));
  if (read_int64_array(env, argv[1], dims, POLY_MAX_DIMS) != ndim) {
    napi_throw_range_error(env, NULL, "polygrad: shape length does not match ndim");
    return NULL;
  }
  return make_external(
      env, poly_tensor_rand_by_id(ctx, dims, ndim, dtype_id, (PolyDevice)device, contiguous)
  );
}

static napi_value napi_poly_tensor_randn_by_id(napi_env env, napi_callback_info info) {
  napi_value argv[5];
  size_t argc = 5;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  int64_t dims[POLY_MAX_DIMS];
  int32_t ndim = 0, dtype_id = 0, device = 0;
  NAPI_CALL(env, napi_get_value_int32(env, argv[2], &ndim));
  NAPI_CALL(env, napi_get_value_int32(env, argv[3], &dtype_id));
  NAPI_CALL(env, napi_get_value_int32(env, argv[4], &device));
  if (read_int64_array(env, argv[1], dims, POLY_MAX_DIMS) != ndim) {
    napi_throw_range_error(env, NULL, "polygrad: shape length does not match ndim");
    return NULL;
  }
  return make_external(env, poly_tensor_randn_by_id(ctx, dims, ndim, dtype_id, (PolyDevice)device));
}

static napi_value napi_poly_tensor_linspace_by_id(napi_env env, napi_callback_info info) {
  napi_value argv[6];
  size_t argc = 6;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  double start, stop;
  int64_t steps;
  int32_t dtype_id, device_id;
  NAPI_CALL(env, napi_get_value_double(env, argv[1], &start));
  NAPI_CALL(env, napi_get_value_double(env, argv[2], &stop));
  NAPI_CALL(env, napi_get_value_int64(env, argv[3], &steps));
  NAPI_CALL(env, napi_get_value_int32(env, argv[4], &dtype_id));
  NAPI_CALL(env, napi_get_value_int32(env, argv[5], &device_id));
  return make_external(
      env, poly_tensor_linspace_by_id(ctx, start, stop, steps, dtype_id, device_id)
  );
}

static napi_value napi_poly_tensor_eye_by_id(napi_env env, napi_callback_info info) {
  napi_value argv[5];
  size_t argc = 5;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  int64_t n, m;
  int32_t dtype_id, device_id;
  NAPI_CALL(env, napi_get_value_int64(env, argv[1], &n));
  NAPI_CALL(env, napi_get_value_int64(env, argv[2], &m));
  NAPI_CALL(env, napi_get_value_int32(env, argv[3], &dtype_id));
  NAPI_CALL(env, napi_get_value_int32(env, argv[4], &device_id));
  return make_external(env, poly_tensor_eye_by_id(ctx, n, m, dtype_id, device_id));
}

static napi_value napi_poly_uop_has_buffer_identity(napi_env env, napi_callback_info info) {
  napi_value argv[1];
  size_t argc = 1;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyUOp *u = get_external(env, argv[0]);
  napi_value out;
  napi_get_boolean(env, poly_uop_has_buffer_identity(u), &out);
  return out;
}

static napi_value napi_poly_uop_key(napi_env env, napi_callback_info info) {
  napi_value argv[1];
  size_t argc = 1;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyUOp *u = get_external(env, argv[0]);
  napi_value out;
  napi_create_bigint_uint64(env, (uint64_t)(uintptr_t)u, &out);
  return out;
}

static napi_value napi_poly_uop_op(napi_env env, napi_callback_info info) {
  napi_value argv[1];
  size_t argc = 1;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyUOp *u = get_external(env, argv[0]);
  napi_value out;
  NAPI_CALL(env, napi_create_int32(env, poly_uop_op(u), &out));
  return out;
}

static napi_value napi_poly_uop_device(napi_env env, napi_callback_info info) {
  napi_value argv[1];
  size_t argc = 1;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyUOp *u = get_external(env, argv[0]);
  napi_value out;
  NAPI_CALL(env, napi_create_int32(env, poly_uop_device(u), &out));
  return out;
}

static napi_value napi_poly_uop_n_src(napi_env env, napi_callback_info info) {
  napi_value argv[1];
  size_t argc = 1;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyUOp *u = get_external(env, argv[0]);
  napi_value out;
  NAPI_CALL(env, napi_create_int32(env, poly_uop_n_src(u), &out));
  return out;
}

static napi_value napi_poly_uop_src(napi_env env, napi_callback_info info) {
  napi_value argv[2];
  size_t argc = 2;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyUOp *u = get_external(env, argv[0]);
  int32_t idx = 0;
  NAPI_CALL(env, napi_get_value_int32(env, argv[1], &idx));
  return make_external(env, poly_uop_src(u, idx));
}

static napi_value napi_poly_uop_call_grad_fxn_key(napi_env env, napi_callback_info info) {
  napi_value argv[1];
  size_t argc = 1;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyUOp *u = get_external(env, argv[0]);
  napi_value out;
  NAPI_CALL(env, napi_create_uint32(env, poly_uop_call_grad_fxn_key(u), &out));
  return out;
}

static napi_value napi_poly_uop_get_buffer_identity(napi_env env, napi_callback_info info) {
  napi_value argv[1];
  size_t argc = 1;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyUOp *u = get_external(env, argv[0]);
  const PolyUOp *r = poly_uop_get_buffer_identity(u);
  return make_external(env, (void *)r);
}

static napi_value napi_poly_uop_buffer(napi_env env, napi_callback_info info) {
  napi_value argv[2];
  size_t argc = 2;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *u = get_external(env, argv[1]);
  return make_external(env, poly_uop_buffer(ctx, u));
}

static napi_value napi_poly_uop_reachable(napi_env env, napi_callback_info info) {
  napi_value argv[3];
  size_t argc = 3;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *root = get_external(env, argv[1]);
  PolyUOp *target = get_external(env, argv[2]);
  napi_value out;
  /* Frontend parity helper for tinygrad's `target in root.toposort()` check. */
  NAPI_CALL(env, napi_get_boolean(env, poly_uop_reachable(ctx, root, target), &out));
  return out;
}

static napi_value napi_poly_uop_substitute(napi_env env, napi_callback_info info) {
  napi_value argv[4];
  size_t argc = 4;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *root = get_external(env, argv[1]);

  bool from_is_arr = false, to_is_arr = false;
  napi_is_array(env, argv[2], &from_is_arr);
  napi_is_array(env, argv[3], &to_is_arr);
  if (!from_is_arr || !to_is_arr) return make_external(env, root);

  uint32_t n_from = 0, n_to = 0;
  napi_get_array_length(env, argv[2], &n_from);
  napi_get_array_length(env, argv[3], &n_to);
  uint32_t n = n_from < n_to ? n_from : n_to;
  if (n == 0) return make_external(env, root);

  PolyUOp **from = (PolyUOp **)malloc((size_t)n * sizeof(PolyUOp *));
  PolyUOp **to = (PolyUOp **)malloc((size_t)n * sizeof(PolyUOp *));
  if (!from || !to) {
    free(from);
    free(to);
    napi_value null_value;
    napi_get_null(env, &null_value);
    return null_value;
  }

  for (uint32_t i = 0; i < n; i++) {
    napi_value from_el, to_el;
    napi_get_element(env, argv[2], i, &from_el);
    napi_get_element(env, argv[3], i, &to_el);
    from[i] = get_external(env, from_el);
    to[i] = get_external(env, to_el);
  }

  PolyUOp *out_uop = poly_uop_substitute(ctx, root, from, to, (int)n);
  free(from);
  free(to);
  return make_external(env, out_uop);
}

/* poly_realize_uops(ctx, uops[], n, out_uops[]) -- batched raw graph realize.
 * JS signature: realizeUops(ctx, [uop0, uop1, ...]) -> [realizedUop0, ...]. */
static napi_value napi_poly_realize_uops(napi_env env, napi_callback_info info) {
  napi_value argv[2];
  size_t argc = 2;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  bool is_arr;
  napi_is_array(env, argv[1], &is_arr);
  if (!is_arr) {
    napi_value empty;
    napi_create_array_with_length(env, 0, &empty);
    return empty;
  }
  uint32_t n;
  napi_get_array_length(env, argv[1], &n);
  PolyUOp **in_arr = (PolyUOp **)malloc(n * sizeof(PolyUOp *));
  PolyUOp **out_arr = (PolyUOp **)malloc(n * sizeof(PolyUOp *));
  for (uint32_t i = 0; i < n; i++) {
    napi_value el;
    napi_get_element(env, argv[1], i, &el);
    in_arr[i] = get_external(env, el);
    out_arr[i] = NULL;
  }
  int rc = poly_realize_uops(ctx, in_arr, (int)n, out_arr);
  napi_value out_js;
  napi_create_array_with_length(env, n, &out_js);
  if (rc == 0) {
    for (uint32_t i = 0; i < n; i++) {
      napi_value el = make_external(env, out_arr[i]);
      napi_set_element(env, out_js, i, el);
    }
  }
  free(in_arr);
  free(out_arr);
  return out_js;
}

static napi_value napi_poly_tensor_empty_by_id(napi_env env, napi_callback_info info) {
  napi_value argv[5];
  size_t argc = 5;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  int32_t dtype_id = 0;
  int32_t ndim = 0;
  int32_t device = 0;
  napi_get_value_int32(env, argv[1], &dtype_id);
  napi_get_value_int32(env, argv[3], &ndim);
  napi_get_value_int32(env, argv[4], &device);
  int64_t dims[POLY_MAX_DIMS];
  int read_ndim = read_int64_array(env, argv[2], dims, POLY_MAX_DIMS);
  if (ndim != read_ndim) {
    napi_throw_range_error(env, NULL, "polygrad: shape length does not match ndim");
    return NULL;
  }
  PolyTensor *tensor = poly_tensor_empty_by_id(ctx, dtype_id, dims, ndim, device);
  if (!tensor) {
    napi_value null_value;
    napi_get_null(env, &null_value);
    return null_value;
  }
  return make_external(env, tensor);
}

static napi_value napi_poly_tensor_create_with_roots(napi_env env, napi_callback_info info) {
  napi_value argv[5];
  size_t argc = 5;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *uop_logical = get_external(env, argv[1]);
  PolyUOp *uop_physical = get_external(env, argv[2]);
  int32_t role = 0;
  int32_t device = 0;
  napi_get_value_int32(env, argv[3], &role);
  napi_get_value_int32(env, argv[4], &device);
  return make_external(
      env, poly_tensor_create_with_roots(
               ctx, uop_logical, uop_physical, (PolyTensorRole)role, (PolyDevice)device
           )
  );
}

static napi_value napi_poly_tensor_create_result_like(napi_env env, napi_callback_info info) {
  napi_value argv[6];
  size_t argc = 6;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyTensor *input = get_external(env, argv[1]);
  PolyUOp *uop_logical = get_external_nullable(env, argv[2]);
  PolyUOp *uop_physical = get_external(env, argv[3]);
  int32_t role = 0;
  int32_t device = 0;
  napi_get_value_int32(env, argv[4], &role);
  napi_get_value_int32(env, argv[5], &device);
  return make_external(
      env, poly_tensor_create_result_like(
               ctx, input, uop_logical, uop_physical, (PolyTensorRole)role, (PolyDevice)device
           )
  );
}

static napi_value napi_poly_tensor_retain(napi_env env, napi_callback_info info) {
  napi_value argv[1];
  size_t argc = 1;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  return make_external(env, poly_tensor_retain(get_external(env, argv[0])));
}

static napi_value napi_poly_tensor_release(napi_env env, napi_callback_info info) {
  napi_value argv[1], out;
  size_t argc = 1;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  poly_tensor_release(get_external(env, argv[0]));
  NAPI_CALL(env, napi_get_undefined(env, &out));
  return out;
}

static napi_value napi_poly_tensor_replace_roots(napi_env env, napi_callback_info info) {
  napi_value argv[6];
  size_t argc = 6;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyTensor *tensor = get_external(env, argv[1]);
  PolyUOp *uop_logical = get_external(env, argv[2]);
  PolyUOp *uop_physical = get_external(env, argv[3]);
  int32_t role = 0;
  int32_t device = 0;
  napi_get_value_int32(env, argv[4], &role);
  napi_get_value_int32(env, argv[5], &device);
  napi_value out;
  NAPI_CALL(
      env, napi_create_int32(
               env,
               poly_tensor_replace_roots(
                   ctx, tensor, uop_logical, uop_physical, (PolyTensorRole)role, (PolyDevice)device
               ),
               &out
           )
  );
  return out;
}

static napi_value napi_poly_tensor_to_device(napi_env env, napi_callback_info info) {
  napi_value argv[3];
  size_t argc = 3;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyTensor *tensor = get_external(env, argv[1]);
  int32_t device = 0;
  napi_get_value_int32(env, argv[2], &device);
  return make_external(env, poly_tensor_to_device(ctx, tensor, (PolyDevice)device));
}

static napi_value napi_poly_tensor_assign(napi_env env, napi_callback_info info) {
  napi_value argv[3];
  size_t argc = 3;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyTensor *target = get_external(env, argv[1]);
  PolyTensor *value = get_external(env, argv[2]);
  return make_external(env, poly_tensor_assign(ctx, target, value));
}

static napi_value napi_poly_tensor_alu1(napi_env env, napi_callback_info info) {
  napi_value argv[3];
  size_t argc = 3;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  int32_t op = 0;
  napi_get_value_int32(env, argv[1], &op);
  PolyTensor *src = get_external(env, argv[2]);
  return make_external(env, poly_tensor_alu1(ctx, (PolyOps)op, src));
}

static napi_value napi_poly_tensor_alu2(napi_env env, napi_callback_info info) {
  napi_value argv[4];
  size_t argc = 4;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  int32_t op = 0;
  napi_get_value_int32(env, argv[1], &op);
  PolyTensor *a = get_external(env, argv[2]);
  PolyTensor *b = get_external(env, argv[3]);
  return make_external(env, poly_tensor_alu2(ctx, (PolyOps)op, a, b));
}

static napi_value napi_poly_tensor_alu3(napi_env env, napi_callback_info info) {
  napi_value argv[5];
  size_t argc = 5;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  int32_t op = 0;
  napi_get_value_int32(env, argv[1], &op);
  PolyTensor *a = get_external(env, argv[2]);
  PolyTensor *b = get_external(env, argv[3]);
  PolyTensor *c = get_external(env, argv[4]);
  return make_external(env, poly_tensor_alu3(ctx, (PolyOps)op, a, b, c));
}

static napi_value napi_poly_tensor_div(napi_env env, napi_callback_info info) {
  napi_value argv[3];
  size_t argc = 3;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  return make_external(
      env, poly_tensor_div(
               get_external(env, argv[0]), get_external(env, argv[1]), get_external(env, argv[2])
           )
  );
}

static napi_value napi_poly_tensor_exp(napi_env env, napi_callback_info info) {
  napi_value argv[2];
  size_t argc = 2;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  return make_external(
      env, poly_tensor_exp(get_external(env, argv[0]), get_external(env, argv[1]))
  );
}

static napi_value napi_poly_tensor_log(napi_env env, napi_callback_info info) {
  napi_value argv[2];
  size_t argc = 2;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  return make_external(
      env, poly_tensor_log(get_external(env, argv[0]), get_external(env, argv[1]))
  );
}

static napi_value napi_poly_tensor_cos(napi_env env, napi_callback_info info) {
  napi_value argv[2];
  size_t argc = 2;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  return make_external(
      env, poly_tensor_cos(get_external(env, argv[0]), get_external(env, argv[1]))
  );
}

static napi_value napi_poly_tensor_tan(napi_env env, napi_callback_info info) {
  napi_value argv[2];
  size_t argc = 2;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  return make_external(
      env, poly_tensor_tan(get_external(env, argv[0]), get_external(env, argv[1]))
  );
}

static napi_value napi_poly_tensor_log1p(napi_env env, napi_callback_info info) {
  napi_value argv[2];
  size_t argc = 2;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  return make_external(
      env, poly_tensor_log1p(get_external(env, argv[0]), get_external(env, argv[1]))
  );
}

static napi_value napi_poly_tensor_expm1(napi_env env, napi_callback_info info) {
  napi_value argv[2];
  size_t argc = 2;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  return make_external(
      env, poly_tensor_expm1(get_external(env, argv[0]), get_external(env, argv[1]))
  );
}

static napi_value napi_poly_tensor_gelu(napi_env env, napi_callback_info info) {
  napi_value argv[2];
  size_t argc = 2;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  return make_external(
      env, poly_tensor_gelu(get_external(env, argv[0]), get_external(env, argv[1]))
  );
}

static napi_value napi_poly_tensor_quick_gelu(napi_env env, napi_callback_info info) {
  napi_value argv[2];
  size_t argc = 2;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  return make_external(
      env, poly_tensor_quick_gelu(get_external(env, argv[0]), get_external(env, argv[1]))
  );
}

static napi_value napi_poly_tensor_detach(napi_env env, napi_callback_info info) {
  napi_value argv[2];
  size_t argc = 2;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  return make_external(
      env, poly_tensor_detach(get_external(env, argv[0]), get_external(env, argv[1]))
  );
}

static napi_value napi_poly_tensor_contiguous_backward(napi_env env, napi_callback_info info) {
  napi_value argv[2];
  size_t argc = 2;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  return make_external(
      env, poly_tensor_contiguous_backward(get_external(env, argv[0]), get_external(env, argv[1]))
  );
}

static napi_value napi_poly_tensor_sum(napi_env env, napi_callback_info info) {
  napi_value argv[5];
  size_t argc = 5;
  int64_t axes[POLY_MAX_DIMS];
  int32_t n_axes = 0;
  bool keepdim = false;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  read_int64_array(env, argv[2], axes, POLY_MAX_DIMS);
  napi_get_value_int32(env, argv[3], &n_axes);
  napi_get_value_bool(env, argv[4], &keepdim);
  return make_external(
      env,
      poly_tensor_sum(get_external(env, argv[0]), get_external(env, argv[1]), axes, n_axes, keepdim)
  );
}

static napi_value napi_poly_tensor_sum_dtype_by_id(napi_env env, napi_callback_info info) {
  napi_value argv[6];
  size_t argc = 6;
  int64_t axes[POLY_MAX_DIMS];
  int32_t n_axes = 0, dtype_id = -1;
  bool keepdim = false;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  read_int64_array(env, argv[2], axes, POLY_MAX_DIMS);
  napi_get_value_int32(env, argv[3], &n_axes);
  napi_get_value_bool(env, argv[4], &keepdim);
  napi_get_value_int32(env, argv[5], &dtype_id);
  return make_external(
      env,
      poly_tensor_sum_dtype_by_id(
          get_external(env, argv[0]), get_external(env, argv[1]), axes, n_axes, keepdim, dtype_id
      )
  );
}

static napi_value napi_poly_tensor_max(napi_env env, napi_callback_info info) {
  napi_value argv[5];
  size_t argc = 5;
  int64_t axes[POLY_MAX_DIMS];
  int32_t n_axes = 0;
  bool keepdim = false;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  read_int64_array(env, argv[2], axes, POLY_MAX_DIMS);
  napi_get_value_int32(env, argv[3], &n_axes);
  napi_get_value_bool(env, argv[4], &keepdim);
  return make_external(
      env,
      poly_tensor_max(get_external(env, argv[0]), get_external(env, argv[1]), axes, n_axes, keepdim)
  );
}

static napi_value napi_poly_tensor_argmax(napi_env env, napi_callback_info info) {
  napi_value argv[4];
  size_t argc = 4;
  int32_t axis = 0;
  bool keepdim = false;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  napi_get_value_int32(env, argv[2], &axis);
  napi_get_value_bool(env, argv[3], &keepdim);
  return make_external(
      env, poly_tensor_argmax(get_external(env, argv[0]), get_external(env, argv[1]), axis, keepdim)
  );
}

static napi_value napi_poly_tensor_minimum(napi_env env, napi_callback_info info) {
  napi_value argv[3];
  size_t argc = 3;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  return make_external(
      env, poly_tensor_minimum(
               get_external(env, argv[0]), get_external(env, argv[1]), get_external(env, argv[2])
           )
  );
}

static napi_value napi_poly_tensor_dot(napi_env env, napi_callback_info info) {
  napi_value argv[3];
  size_t argc = 3;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  return make_external(
      env, poly_tensor_dot(
               get_external(env, argv[0]), get_external(env, argv[1]), get_external(env, argv[2])
           )
  );
}

static napi_value napi_poly_tensor_dot_dtype_by_id(napi_env env, napi_callback_info info) {
  napi_value argv[4];
  size_t argc = 4;
  int32_t dtype_id = -1;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  napi_get_value_int32(env, argv[3], &dtype_id);
  return make_external(
      env, poly_tensor_dot_dtype_by_id(
               get_external(env, argv[0]), get_external(env, argv[1]), get_external(env, argv[2]),
               dtype_id
           )
  );
}

static napi_value napi_poly_tensor_qr_ex(napi_env env, napi_callback_info info) {
  napi_value argv[3];
  size_t argc = 3;
  int32_t mode = 0;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  napi_get_value_int32(env, argv[2], &mode);
  PolyTensor *q = NULL, *r = NULL;
  if (poly_tensor_qr_ex(get_external(env, argv[0]), get_external(env, argv[1]), mode, &q, &r) !=
      0) {
    napi_throw_error(env, NULL, "polygrad: poly_tensor_qr_ex failed");
    return NULL;
  }
  return make_external_pair(env, q, r);
}

static napi_value napi_poly_tensor_triangular_solve(napi_env env, napi_callback_info info) {
  napi_value argv[6];
  size_t argc = 6;
  int32_t upper = 0, transpose_a = 0, unit_diagonal = 0;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  napi_get_value_int32(env, argv[3], &upper);
  napi_get_value_int32(env, argv[4], &transpose_a);
  napi_get_value_int32(env, argv[5], &unit_diagonal);
  return make_external(
      env, poly_tensor_triangular_solve(
               get_external(env, argv[0]), get_external(env, argv[1]), get_external(env, argv[2]),
               upper, transpose_a, unit_diagonal
           )
  );
}

static napi_value napi_poly_tensor_cholesky(napi_env env, napi_callback_info info) {
  napi_value argv[3];
  size_t argc = 3;
  int32_t upper = 0;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  napi_get_value_int32(env, argv[2], &upper);
  return make_external(
      env, poly_tensor_cholesky(get_external(env, argv[0]), get_external(env, argv[1]), upper)
  );
}

static napi_value napi_poly_tensor_cholesky_solve(napi_env env, napi_callback_info info) {
  napi_value argv[4];
  size_t argc = 4;
  int32_t upper = 0;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  napi_get_value_int32(env, argv[3], &upper);
  return make_external(
      env,
      poly_tensor_cholesky_solve(
          get_external(env, argv[0]), get_external(env, argv[1]), get_external(env, argv[2]), upper
      )
  );
}

static napi_value napi_poly_tensor_solve(napi_env env, napi_callback_info info) {
  napi_value argv[3];
  size_t argc = 3;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  return make_external(
      env, poly_tensor_solve(
               get_external(env, argv[0]), get_external(env, argv[1]), get_external(env, argv[2])
           )
  );
}

static napi_value napi_poly_tensor_lstsq(napi_env env, napi_callback_info info) {
  napi_value argv[3];
  size_t argc = 3;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  return make_external(
      env, poly_tensor_lstsq(
               get_external(env, argv[0]), get_external(env, argv[1]), get_external(env, argv[2])
           )
  );
}

static napi_value napi_poly_tensor_sort(napi_env env, napi_callback_info info) {
  napi_value argv[4];
  size_t argc = 4;
  int32_t dim = 0, descending = 0;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  napi_get_value_int32(env, argv[2], &dim);
  napi_get_value_int32(env, argv[3], &descending);
  PolyTensor *values = NULL, *indices = NULL;
  if (poly_tensor_sort(
          get_external(env, argv[0]), get_external(env, argv[1]), dim, descending, &values, &indices
      ) != 0) {
    napi_throw_error(env, NULL, "polygrad: poly_tensor_sort failed");
    return NULL;
  }
  return make_external_pair(env, values, indices);
}

static napi_value napi_poly_tensor_topk(napi_env env, napi_callback_info info) {
  napi_value argv[6];
  size_t argc = 6;
  int64_t k = 0;
  int32_t dim = 0, largest = 0, sorted = 0;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  napi_get_value_int64(env, argv[2], &k);
  napi_get_value_int32(env, argv[3], &dim);
  napi_get_value_int32(env, argv[4], &largest);
  napi_get_value_int32(env, argv[5], &sorted);
  PolyTensor *values = NULL, *indices = NULL;
  if (poly_tensor_topk(
          get_external(env, argv[0]), get_external(env, argv[1]), k, dim, largest, sorted, &values,
          &indices
      ) != 0) {
    napi_throw_error(env, NULL, "polygrad: poly_tensor_topk failed");
    return NULL;
  }
  return make_external_pair(env, values, indices);
}

static napi_value napi_poly_tensor_softmax(napi_env env, napi_callback_info info) {
  napi_value argv[3];
  size_t argc = 3;
  int32_t axis = 0;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  napi_get_value_int32(env, argv[2], &axis);
  return make_external(
      env, poly_tensor_softmax(get_external(env, argv[0]), get_external(env, argv[1]), axis)
  );
}

static napi_value napi_poly_tensor_log_softmax(napi_env env, napi_callback_info info) {
  napi_value argv[3];
  size_t argc = 3;
  int32_t axis = 0;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  napi_get_value_int32(env, argv[2], &axis);
  return make_external(
      env, poly_tensor_log_softmax(get_external(env, argv[0]), get_external(env, argv[1]), axis)
  );
}

static napi_value napi_poly_tensor_cast_by_id(napi_env env, napi_callback_info info) {
  napi_value argv[3];
  size_t argc = 3;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyTensor *src = get_external(env, argv[1]);
  int32_t dtype_id = 0;
  napi_get_value_int32(env, argv[2], &dtype_id);
  return make_external(env, poly_tensor_cast_by_id(ctx, src, dtype_id));
}

static napi_value napi_poly_tensor_bitcast_by_id(napi_env env, napi_callback_info info) {
  napi_value argv[3];
  size_t argc = 3;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyTensor *src = get_external(env, argv[1]);
  int32_t dtype_id = 0;
  napi_get_value_int32(env, argv[2], &dtype_id);
  return make_external(env, poly_tensor_bitcast_by_id(ctx, src, dtype_id));
}

static napi_value napi_poly_tensor_contiguous(napi_env env, napi_callback_info info) {
  napi_value argv[2];
  size_t argc = 2;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyTensor *tensor = get_external(env, argv[1]);
  return make_external(env, poly_tensor_contiguous(ctx, tensor));
}

static napi_value napi_poly_tensor_reshape(napi_env env, napi_callback_info info) {
  napi_value argv[4];
  size_t argc = 4;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyTensor *tensor = get_external(env, argv[1]);
  int64_t dims[POLY_MAX_DIMS];
  int32_t ndim = 0;
  read_int64_array(env, argv[2], dims, POLY_MAX_DIMS);
  napi_get_value_int32(env, argv[3], &ndim);
  return make_external(env, poly_tensor_reshape(ctx, tensor, dims, ndim));
}

static napi_value napi_poly_tensor_expand(napi_env env, napi_callback_info info) {
  napi_value argv[4];
  size_t argc = 4;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyTensor *tensor = get_external(env, argv[1]);
  int64_t dims[POLY_MAX_DIMS];
  int32_t ndim = 0;
  read_int64_array(env, argv[2], dims, POLY_MAX_DIMS);
  napi_get_value_int32(env, argv[3], &ndim);
  return make_external(env, poly_tensor_expand(ctx, tensor, dims, ndim));
}

static napi_value napi_poly_tensor_permute(napi_env env, napi_callback_info info) {
  napi_value argv[4];
  size_t argc = 4;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyTensor *tensor = get_external(env, argv[1]);
  int64_t order[POLY_MAX_DIMS];
  int32_t ndim = 0;
  read_int64_array(env, argv[2], order, POLY_MAX_DIMS);
  napi_get_value_int32(env, argv[3], &ndim);
  return make_external(env, poly_tensor_permute(ctx, tensor, order, ndim));
}

static napi_value napi_poly_tensor_shrink(napi_env env, napi_callback_info info) {
  napi_value argv[4];
  size_t argc = 4;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyTensor *tensor = get_external(env, argv[1]);
  int64_t flat[POLY_MAX_DIMS * 2];
  int32_t ndim = 0;
  read_int64_array(env, argv[2], flat, POLY_MAX_DIMS * 2);
  napi_get_value_int32(env, argv[3], &ndim);
  return make_external(env, poly_tensor_shrink(ctx, tensor, (int64_t(*)[2])flat, ndim));
}

static napi_value napi_poly_tensor_flip(napi_env env, napi_callback_info info) {
  napi_value argv[4];
  size_t argc = 4;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyTensor *tensor = get_external(env, argv[1]);
  int64_t axes[POLY_MAX_DIMS];
  int32_t n_axes = 0;
  read_int64_array(env, argv[2], axes, POLY_MAX_DIMS);
  napi_get_value_int32(env, argv[3], &n_axes);
  return make_external(env, poly_tensor_flip(ctx, tensor, axes, n_axes));
}

static napi_value napi_poly_tensor_pad_value_bool(napi_env env, napi_callback_info info) {
  napi_value argv[5];
  size_t argc = 5;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyTensor *tensor = get_external(env, argv[1]);
  int64_t flat[POLY_MAX_DIMS * 2];
  int32_t ndim = 0;
  bool value = false;
  read_int64_array(env, argv[2], flat, POLY_MAX_DIMS * 2);
  napi_get_value_int32(env, argv[3], &ndim);
  napi_get_value_bool(env, argv[4], &value);
  return make_external(
      env, poly_tensor_pad_value_bool(ctx, tensor, (int64_t(*)[2])flat, ndim, value)
  );
}

static napi_value napi_poly_tensor_pad_value_int(napi_env env, napi_callback_info info) {
  napi_value argv[5];
  size_t argc = 5;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyTensor *tensor = get_external(env, argv[1]);
  int64_t flat[POLY_MAX_DIMS * 2];
  int32_t ndim = 0;
  int64_t value = 0;
  read_int64_array(env, argv[2], flat, POLY_MAX_DIMS * 2);
  napi_get_value_int32(env, argv[3], &ndim);
  napi_get_value_int64(env, argv[4], &value);
  return make_external(
      env, poly_tensor_pad_value_int(ctx, tensor, (int64_t(*)[2])flat, ndim, value)
  );
}

static napi_value napi_poly_tensor_pad_value_float(napi_env env, napi_callback_info info) {
  napi_value argv[5];
  size_t argc = 5;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyTensor *tensor = get_external(env, argv[1]);
  int64_t flat[POLY_MAX_DIMS * 2];
  int32_t ndim = 0;
  double value = 0.0;
  read_int64_array(env, argv[2], flat, POLY_MAX_DIMS * 2);
  napi_get_value_int32(env, argv[3], &ndim);
  napi_get_value_double(env, argv[4], &value);
  return make_external(
      env, poly_tensor_pad_value_float(ctx, tensor, (int64_t(*)[2])flat, ndim, value)
  );
}

static napi_value napi_poly_tensor_pool(napi_env env, napi_callback_info info) {
  napi_value argv[6];
  size_t argc = 6;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  int64_t kernel[POLY_MAX_DIMS], stride[POLY_MAX_DIMS], dilation[POLY_MAX_DIMS];
  int32_t n_kernel = 0;
  read_int64_array(env, argv[2], kernel, POLY_MAX_DIMS);
  napi_get_value_int32(env, argv[3], &n_kernel);
  read_int64_array(env, argv[4], stride, POLY_MAX_DIMS);
  read_int64_array(env, argv[5], dilation, POLY_MAX_DIMS);
  return make_external(
      env,
      poly_tensor_pool(
          get_external(env, argv[0]), get_external(env, argv[1]), kernel, n_kernel, stride, dilation
      )
  );
}

static napi_value napi_poly_tensor_max_pool2d(napi_env env, napi_callback_info info) {
  napi_value argv[8];
  size_t argc = 8;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  int64_t kernel[POLY_MAX_DIMS], stride[POLY_MAX_DIMS], dilation[POLY_MAX_DIMS];
  int64_t padding[2 * POLY_MAX_DIMS];
  int32_t n_kernel = 0, n_padding = 0;
  read_int64_array(env, argv[2], kernel, POLY_MAX_DIMS);
  napi_get_value_int32(env, argv[3], &n_kernel);
  read_int64_array(env, argv[4], stride, POLY_MAX_DIMS);
  read_int64_array(env, argv[5], dilation, POLY_MAX_DIMS);
  read_int64_array(env, argv[6], padding, 2 * POLY_MAX_DIMS);
  napi_get_value_int32(env, argv[7], &n_padding);
  return make_external(
      env, poly_tensor_max_pool2d(
               get_external(env, argv[0]), get_external(env, argv[1]), kernel, n_kernel, stride,
               dilation, padding, n_padding
           )
  );
}

static napi_value napi_poly_tensor_conv2d(napi_env env, napi_callback_info info) {
  napi_value argv[9];
  size_t argc = 9;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  int64_t stride[POLY_MAX_DIMS], dilation[POLY_MAX_DIMS], padding[2 * POLY_MAX_DIMS];
  int32_t groups = 0, n_padding = 0;
  napi_get_value_int32(env, argv[4], &groups);
  read_int64_array(env, argv[5], stride, POLY_MAX_DIMS);
  read_int64_array(env, argv[6], dilation, POLY_MAX_DIMS);
  read_int64_array(env, argv[7], padding, 2 * POLY_MAX_DIMS);
  napi_get_value_int32(env, argv[8], &n_padding);
  return make_external(
      env, poly_tensor_conv2d(
               get_external(env, argv[0]), get_external(env, argv[1]), get_external(env, argv[2]),
               get_external_nullable(env, argv[3]), groups, stride, dilation, padding, n_padding
           )
  );
}

static napi_value napi_poly_tensor_conv2d_dtype_by_id(napi_env env, napi_callback_info info) {
  napi_value argv[10];
  size_t argc = 10;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  int64_t stride[POLY_MAX_DIMS], dilation[POLY_MAX_DIMS], padding[2 * POLY_MAX_DIMS];
  int32_t groups = 0, n_padding = 0, dtype_id = -1;
  napi_get_value_int32(env, argv[4], &groups);
  read_int64_array(env, argv[5], stride, POLY_MAX_DIMS);
  read_int64_array(env, argv[6], dilation, POLY_MAX_DIMS);
  read_int64_array(env, argv[7], padding, 2 * POLY_MAX_DIMS);
  napi_get_value_int32(env, argv[8], &n_padding);
  napi_get_value_int32(env, argv[9], &dtype_id);
  return make_external(
      env, poly_tensor_conv2d_dtype_by_id(
               get_external(env, argv[0]), get_external(env, argv[1]), get_external(env, argv[2]),
               get_external_nullable(env, argv[3]), groups, stride, dilation, padding, n_padding,
               dtype_id
           )
  );
}

static napi_value napi_poly_tensor_batchnorm(napi_env env, napi_callback_info info) {
  napi_value argv[8];
  size_t argc = 8;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  int64_t axes[POLY_MAX_DIMS];
  int32_t n_axes = 0;
  read_int64_array(env, argv[6], axes, POLY_MAX_DIMS);
  napi_get_value_int32(env, argv[7], &n_axes);
  return make_external(
      env, poly_tensor_batchnorm(
               get_external(env, argv[0]), get_external(env, argv[1]),
               get_external_nullable(env, argv[2]), get_external_nullable(env, argv[3]),
               get_external(env, argv[4]), get_external(env, argv[5]), axes, n_axes
           )
  );
}

static napi_value napi_poly_tensor_one_hot(napi_env env, napi_callback_info info) {
  napi_value argv[3];
  size_t argc = 3;
  int64_t num_classes = 0;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  napi_get_value_int64(env, argv[2], &num_classes);
  return make_external(
      env, poly_tensor_one_hot(get_external(env, argv[0]), get_external(env, argv[1]), num_classes)
  );
}

static napi_value napi_poly_tensor_gather_dim(napi_env env, napi_callback_info info) {
  napi_value argv[4];
  size_t argc = 4;
  int32_t dim = 0;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  napi_get_value_int32(env, argv[2], &dim);
  return make_external(
      env,
      poly_tensor_gather_dim(
          get_external(env, argv[0]), get_external(env, argv[1]), dim, get_external(env, argv[3])
      )
  );
}

static napi_value napi_poly_tensor_index_select(napi_env env, napi_callback_info info) {
  napi_value argv[4];
  size_t argc = 4;
  int32_t dim = 0;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  napi_get_value_int32(env, argv[2], &dim);
  return make_external(
      env,
      poly_tensor_index_select(
          get_external(env, argv[0]), get_external(env, argv[1]), dim, get_external(env, argv[3])
      )
  );
}

static napi_value napi_poly_tensor_clone_into(napi_env env, napi_callback_info info) {
  napi_value argv[3];
  size_t argc = 3;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyTensor *target = get_external(env, argv[1]);
  PolyTensor *source = get_external(env, argv[2]);
  return make_external(env, poly_tensor_clone_into(ctx, target, source));
}

static napi_value napi_poly_tensor_clone(napi_env env, napi_callback_info info) {
  napi_value argv[3];
  size_t argc = 3;
  int32_t device = 0;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  NAPI_CALL(env, napi_get_value_int32(env, argv[2], &device));
  return make_external(
      env,
      poly_tensor_clone(get_external(env, argv[0]), get_external(env, argv[1]), (PolyDevice)device)
  );
}

static napi_value napi_poly_tensor_uop(napi_env env, napi_callback_info info) {
  napi_value argv[1];
  size_t argc = 1;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyTensor *tensor = get_external(env, argv[0]);
  return make_external(env, poly_tensor_uop(tensor));
}

static napi_value napi_poly_uop_retain(napi_env env, napi_callback_info info) {
  napi_value argv[2], out;
  size_t argc = 2;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  int rc = poly_uop_retain(get_external(env, argv[0]), get_external(env, argv[1]));
  NAPI_CALL(env, napi_create_int32(env, rc, &out));
  return out;
}

static napi_value napi_poly_uop_release(napi_env env, napi_callback_info info) {
  napi_value argv[2], out;
  size_t argc = 2;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  poly_uop_release(get_external(env, argv[0]), get_external(env, argv[1]));
  NAPI_CALL(env, napi_get_undefined(env, &out));
  return out;
}

static napi_value napi_poly_tensor_uop_logical(napi_env env, napi_callback_info info) {
  napi_value argv[1];
  size_t argc = 1;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyTensor *tensor = get_external(env, argv[0]);
  return make_external(env, poly_tensor_uop_logical(tensor));
}

static napi_value napi_poly_tensor_uop_physical(napi_env env, napi_callback_info info) {
  napi_value argv[1];
  size_t argc = 1;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyTensor *tensor = get_external(env, argv[0]);
  return make_external(env, poly_tensor_uop_physical(tensor));
}

static napi_value napi_poly_tensor_logical_policy(napi_env env, napi_callback_info info) {
  napi_value argv[1], out;
  size_t argc = 1;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  NAPI_CALL(
      env,
      napi_create_int32(env, (int32_t)poly_tensor_logical_policy(get_external(env, argv[0])), &out)
  );
  return out;
}

static napi_value napi_poly_tensor_logical_state(napi_env env, napi_callback_info info) {
  napi_value argv[1], out;
  size_t argc = 1;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  NAPI_CALL(
      env,
      napi_create_int32(env, (int32_t)poly_tensor_logical_state(get_external(env, argv[0])), &out)
  );
  return out;
}

static napi_value napi_poly_tensor_set_logical_policy(napi_env env, napi_callback_info info) {
  napi_value argv[3], out;
  size_t argc = 3;
  int32_t policy = 0;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  NAPI_CALL(env, napi_get_value_int32(env, argv[2], &policy));
  int rc = poly_tensor_set_logical_policy(
      get_external(env, argv[0]), get_external(env, argv[1]), (PolyLogicalPolicy)policy
  );
  NAPI_CALL(env, napi_create_int32(env, rc, &out));
  return out;
}

static napi_value napi_poly_tensor_device(napi_env env, napi_callback_info info) {
  napi_value argv[1];
  size_t argc = 1;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyTensor *tensor = get_external(env, argv[0]);
  napi_value out;
  NAPI_CALL(env, napi_create_int32(env, (int32_t)poly_tensor_device(tensor), &out));
  return out;
}

static napi_value napi_poly_tensor_custom_kernel(napi_env env, napi_callback_info info) {
  napi_value argv[4];
  size_t argc = 4;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  if (argc < 4) {
    napi_throw_type_error(
        env, NULL, "poly_tensor_custom_kernel expects ctx, body, inputs, grad key"
    );
    return NULL;
  }
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *body = get_external(env, argv[1]);
  bool is_arr = false;
  NAPI_CALL(env, napi_is_array(env, argv[2], &is_arr));
  if (!is_arr) {
    napi_throw_type_error(env, NULL, "poly_tensor_custom_kernel inputs must be an array");
    return NULL;
  }
  uint32_t n = 0;
  NAPI_CALL(env, napi_get_array_length(env, argv[2], &n));
  if (n == 0) {
    napi_throw_range_error(env, NULL, "poly_tensor_custom_kernel requires at least one input");
    return NULL;
  }
  PolyTensor **inputs = calloc(n, sizeof(*inputs));
  PolyTensor **outputs = calloc(n, sizeof(*outputs));
  if (!inputs || !outputs) {
    free(inputs);
    free(outputs);
    napi_throw_error(env, NULL, "poly_tensor_custom_kernel allocation failed");
    return NULL;
  }
  for (uint32_t i = 0; i < n; i++) {
    napi_value el;
    NAPI_CALL(env, napi_get_element(env, argv[2], i, &el));
    inputs[i] = get_external(env, el);
  }
  uint32_t grad_fxn_key = 0;
  NAPI_CALL(env, napi_get_value_uint32(env, argv[3], &grad_fxn_key));
  int rc = poly_tensor_custom_kernel(ctx, body, inputs, (int)n, grad_fxn_key, outputs);
  free(inputs);
  if (rc != 0) {
    free(outputs);
    napi_throw_error(env, NULL, "poly_tensor_custom_kernel failed");
    return NULL;
  }
  napi_value out;
  NAPI_CALL(env, napi_create_array_with_length(env, n, &out));
  for (uint32_t i = 0; i < n; i++) {
    napi_value el = make_external(env, outputs[i]);
    NAPI_CALL(env, napi_set_element(env, out, i, el));
  }
  free(outputs);
  return out;
}

static napi_value napi_poly_realize_tensors(napi_env env, napi_callback_info info) {
  napi_value argv[2];
  size_t argc = 2;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  bool is_arr;
  napi_is_array(env, argv[1], &is_arr);
  if (!is_arr) {
    napi_value empty;
    napi_create_array_with_length(env, 0, &empty);
    return empty;
  }
  uint32_t n;
  napi_get_array_length(env, argv[1], &n);
  PolyTensor **tensors = (PolyTensor **)malloc(n * sizeof(PolyTensor *));
  for (uint32_t i = 0; i < n; i++) {
    napi_value el;
    napi_get_element(env, argv[1], i, &el);
    tensors[i] = get_external(env, el);
  }
  PolyTensor **outputs = (PolyTensor **)calloc(n, sizeof(PolyTensor *));
  if (!outputs) {
    free(tensors);
    napi_value null_value;
    napi_get_null(env, &null_value);
    return null_value;
  }
  int rc = poly_realize_tensors(ctx, tensors, (int)n, outputs);
  free(tensors);
  if (rc != 0) {
    free(outputs);
    napi_value null_value;
    napi_get_null(env, &null_value);
    return null_value;
  }
  napi_value out;
  NAPI_CALL(env, napi_create_array_with_length(env, n, &out));
  for (uint32_t i = 0; i < n; i++) {
    napi_value el = make_external(env, outputs[i]);
    NAPI_CALL(env, napi_set_element(env, out, i, el));
  }
  free(outputs);
  return out;
}

static bool napi_is_nullish(napi_env env, napi_value val) {
  if (!val) return true;
  napi_valuetype type;
  if (napi_typeof(env, val, &type) != napi_ok) return true;
  return type == napi_null || type == napi_undefined;
}

static int32_t napi_read_int_prop(napi_env env, napi_value obj, const char *name, int32_t def) {
  if (napi_is_nullish(env, obj)) return def;
  bool has = false;
  if (napi_has_named_property(env, obj, name, &has) != napi_ok || !has) return def;
  napi_value val;
  if (napi_get_named_property(env, obj, name, &val) != napi_ok || napi_is_nullish(env, val))
    return def;
  int32_t out = def;
  if (napi_get_value_int32(env, val, &out) != napi_ok) return def;
  return out;
}

static double napi_read_double_prop(napi_env env, napi_value obj, const char *name, double def) {
  if (napi_is_nullish(env, obj)) return def;
  bool has = false;
  if (napi_has_named_property(env, obj, name, &has) != napi_ok || !has) return def;
  napi_value val;
  if (napi_get_named_property(env, obj, name, &val) != napi_ok || napi_is_nullish(env, val))
    return def;
  double out = def;
  if (napi_get_value_double(env, val, &out) != napi_ok) return def;
  return out;
}

static bool napi_read_bool_prop(napi_env env, napi_value obj, const char *name, bool def) {
  if (napi_is_nullish(env, obj)) return def;
  bool has = false;
  if (napi_has_named_property(env, obj, name, &has) != napi_ok || !has) return def;
  napi_value val;
  if (napi_get_named_property(env, obj, name, &val) != napi_ok || napi_is_nullish(env, val))
    return def;
  bool out = def;
  if (napi_get_value_bool(env, val, &out) != napi_ok) return def;
  return out;
}

static bool napi_read_tensor_ptr_array(
    napi_env env,
    napi_value val,
    int expected_len,
    bool allow_null,
    PolyTensor ***out
) {
  *out = NULL;
  if (napi_is_nullish(env, val)) return allow_null;

  bool is_arr = false;
  if (napi_is_array(env, val, &is_arr) != napi_ok || !is_arr) {
    napi_throw_error(env, NULL, "polygrad: expected tensor pointer array");
    return false;
  }

  uint32_t n = 0;
  napi_get_array_length(env, val, &n);
  if (expected_len >= 0 && (int)n != expected_len) {
    napi_throw_error(env, NULL, "polygrad: optimizer tensor array length mismatch");
    return false;
  }

  PolyTensor **arr = (PolyTensor **)calloc(n ? n : 1, sizeof(PolyTensor *));
  if (!arr) {
    napi_throw_error(env, NULL, "malloc failed");
    return false;
  }
  for (uint32_t i = 0; i < n; i++) {
    napi_value el;
    napi_get_element(env, val, i, &el);
    arr[i] = get_external(env, el);
  }
  *out = arr;
  return true;
}

static bool napi_read_tensor_ptr_array_len(
    napi_env env,
    napi_value val,
    PolyTensor ***out,
    int *out_n
) {
  *out = NULL;
  *out_n = 0;
  bool is_arr = false;
  if (napi_is_array(env, val, &is_arr) != napi_ok || !is_arr) {
    napi_throw_error(env, NULL, "polygrad: expected tensor pointer array");
    return false;
  }

  uint32_t n = 0;
  napi_get_array_length(env, val, &n);
  if (!napi_read_tensor_ptr_array(env, val, (int)n, false, out)) return false;
  *out_n = (int)n;
  return true;
}

static napi_value napi_poly_jit_new(napi_env env, napi_callback_info info) {
  napi_value argv[1];
  size_t argc = 1;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  return make_external(env, poly_jit_new(ctx));
}

static napi_value napi_poly_jit_free(napi_env env, napi_callback_info info) {
  napi_value argv[1];
  size_t argc = 1;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyJit *jit = get_external(env, argv[0]);
  poly_jit_free(jit);
  napi_value out;
  NAPI_CALL(env, napi_get_undefined(env, &out));
  return out;
}

static napi_value napi_poly_jit_set_prune(napi_env env, napi_callback_info info) {
  napi_value argv[2];
  size_t argc = 2;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyJit *jit = get_external(env, argv[0]);
  bool prune = false;
  napi_get_value_bool(env, argv[1], &prune);
  napi_value out;
  NAPI_CALL(env, napi_create_int32(env, poly_jit_set_prune(jit, prune), &out));
  return out;
}

static napi_value napi_poly_jit_begin_capture(napi_env env, napi_callback_info info) {
  napi_value argv[2];
  size_t argc = 2;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyJit *jit = get_external(env, argv[0]);
  PolyTensor **inputs = NULL;
  int n_inputs = 0;
  if (!napi_read_tensor_ptr_array_len(env, argv[1], &inputs, &n_inputs)) return NULL;
  int rc = poly_jit_begin_capture(jit, inputs, n_inputs);
  free(inputs);
  napi_value out;
  NAPI_CALL(env, napi_create_int32(env, rc, &out));
  return out;
}

static napi_value napi_poly_jit_end_capture(napi_env env, napi_callback_info info) {
  napi_value argv[2];
  size_t argc = 2;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyJit *jit = get_external(env, argv[0]);
  PolyTensor **live_tensors = NULL;
  int n_live_tensors = 0;
  if (!napi_read_tensor_ptr_array_len(env, argv[1], &live_tensors, &n_live_tensors)) return NULL;
  int rc = poly_jit_end_capture(jit, live_tensors, n_live_tensors);
  free(live_tensors);
  napi_value out;
  NAPI_CALL(env, napi_create_int32(env, rc, &out));
  return out;
}

static napi_value napi_poly_jit_cancel_capture(napi_env env, napi_callback_info info) {
  napi_value argv[1];
  size_t argc = 1;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyJit *jit = get_external(env, argv[0]);
  poly_jit_cancel_capture(jit);
  napi_value out;
  NAPI_CALL(env, napi_get_undefined(env, &out));
  return out;
}

static napi_value napi_poly_jit_is_captured(napi_env env, napi_callback_info info) {
  napi_value argv[1];
  size_t argc = 1;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyJit *jit = get_external(env, argv[0]);
  napi_value out;
  NAPI_CALL(env, napi_get_boolean(env, poly_jit_is_captured(jit), &out));
  return out;
}

static napi_value napi_poly_jit_schedule_count(napi_env env, napi_callback_info info) {
  napi_value argv[1];
  size_t argc = 1;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyJit *jit = get_external(env, argv[0]);
  napi_value out;
  NAPI_CALL(env, napi_create_int32(env, poly_jit_schedule_count(jit), &out));
  return out;
}

static napi_value napi_poly_jit_run(napi_env env, napi_callback_info info) {
  napi_value argv[2];
  size_t argc = 2;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyJit *jit = get_external(env, argv[0]);
  PolyTensor **inputs = NULL;
  int n_inputs = 0;
  if (!napi_read_tensor_ptr_array_len(env, argv[1], &inputs, &n_inputs)) return NULL;
  int rc = poly_jit_run(jit, inputs, n_inputs);
  free(inputs);
  napi_value out;
  NAPI_CALL(env, napi_create_int32(env, rc, &out));
  return out;
}

static napi_value napi_poly_optim_build_step(napi_env env, napi_callback_info info) {
  napi_value argv[9];
  size_t argc = 9;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  if (argc < 9) {
    napi_throw_error(env, NULL, "polygrad: poly_optim_build_step expects 9 arguments");
    return NULL;
  }

  PolyCtx *ctx = get_external(env, argv[0]);
  PolyOptimConfig cfg = {
      .kind = napi_read_int_prop(env, argv[1], "kind", POLY_OPTIM_NONE),
      .beta1 = napi_read_double_prop(env, argv[1], "beta1", 0.9),
      .beta2 = napi_read_double_prop(env, argv[1], "beta2", 0.999),
      .eps = napi_read_double_prop(env, argv[1], "eps", 1e-8),
      .weight_decay = napi_read_double_prop(env, argv[1], "weightDecay", 0.0),
      .momentum = napi_read_double_prop(env, argv[1], "momentum", 0.0),
      .nesterov = napi_read_bool_prop(env, argv[1], "nesterov", false),
      .classic = napi_read_bool_prop(env, argv[1], "classic", false),
  };
  PolyTensor *lr = get_external(env, argv[2]);

  bool is_arr = false;
  if (napi_is_array(env, argv[3], &is_arr) != napi_ok || !is_arr) {
    napi_throw_error(env, NULL, "polygrad: optimizer params must be an array");
    return NULL;
  }
  uint32_t n_params_u32 = 0;
  napi_get_array_length(env, argv[3], &n_params_u32);
  int n_params = (int)n_params_u32;

  PolyTensor **params = NULL;
  PolyTensor **grads = NULL;
  PolyTensor **m_tensors = NULL;
  PolyTensor **v_tensors = NULL;
  if (!napi_read_tensor_ptr_array(env, argv[3], n_params, false, &params) ||
      !napi_read_tensor_ptr_array(env, argv[4], n_params, false, &grads) ||
      !napi_read_tensor_ptr_array(env, argv[5], n_params, true, &m_tensors) ||
      !napi_read_tensor_ptr_array(env, argv[6], n_params, true, &v_tensors)) {
    free(params);
    free(grads);
    free(m_tensors);
    free(v_tensors);
    return NULL;
  }
  PolyTensor *bc1 = napi_is_nullish(env, argv[7]) ? NULL : get_external(env, argv[7]);
  PolyTensor *bc2 = napi_is_nullish(env, argv[8]) ? NULL : get_external(env, argv[8]);

  int needed = poly_optim_build_step(
      ctx, &cfg, lr, params, grads, n_params, m_tensors, v_tensors, bc1, bc2, NULL, 0
  );
  if (needed < 0) {
    free(params);
    free(grads);
    free(m_tensors);
    free(v_tensors);
    napi_value null_value;
    napi_get_null(env, &null_value);
    return null_value;
  }

  PolyTensor **outputs = (PolyTensor **)calloc(needed ? needed : 1, sizeof(PolyTensor *));
  if (!outputs) {
    free(params);
    free(grads);
    free(m_tensors);
    free(v_tensors);
    napi_throw_error(env, NULL, "malloc failed");
    return NULL;
  }
  int rc = poly_optim_build_step(
      ctx, &cfg, lr, params, grads, n_params, m_tensors, v_tensors, bc1, bc2, outputs, needed
  );
  free(params);
  free(grads);
  free(m_tensors);
  free(v_tensors);
  if (rc < 0) {
    free(outputs);
    napi_value null_value;
    napi_get_null(env, &null_value);
    return null_value;
  }

  napi_value out;
  NAPI_CALL(env, napi_create_array_with_length(env, (uint32_t)rc, &out));
  for (int i = 0; i < rc; i++) {
    napi_value el = make_external(env, outputs[i]);
    NAPI_CALL(env, napi_set_element(env, out, (uint32_t)i, el));
  }
  free(outputs);
  return out;
}

static napi_value napi_poly_buffer_get_ptr(napi_env env, napi_callback_info info) {
  napi_value argv[2];
  size_t argc = 2;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *buf = get_external(env, argv[1]);
  void *p = poly_buffer_get_ptr(ctx, buf);
  napi_value out;
  napi_create_bigint_uint64(env, (uint64_t)(uintptr_t)p, &out);
  return out;
}

static napi_value napi_poly_buffer_is_allocated(napi_env env, napi_callback_info info) {
  napi_value argv[2];
  size_t argc = 2;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *buf = get_external(env, argv[1]);
  napi_value out;
  NAPI_CALL(env, napi_get_boolean(env, poly_buffer_is_allocated(ctx, buf), &out));
  return out;
}

static napi_value napi_poly_buffer_get_key(napi_env env, napi_callback_info info) {
  napi_value argv[2];
  size_t argc = 2;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *buf = get_external(env, argv[1]);
  napi_value out;
  NAPI_CALL(env, napi_create_bigint_uint64(env, poly_buffer_get_key(ctx, buf), &out));
  return out;
}

static napi_value napi_poly_set_frontend_buffer_release(napi_env env, napi_callback_info info) {
  napi_value argv[1];
  size_t argc = 1;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));

  if (g_frontend_buffer_release_ref) {
    napi_delete_reference(env, g_frontend_buffer_release_ref);
    g_frontend_buffer_release_ref = NULL;
  }
  g_frontend_buffer_release_env = env;
  NAPI_CALL(env, napi_create_reference(env, argv[0], 1, &g_frontend_buffer_release_ref));
  poly_set_frontend_buffer_release(napi_frontend_buffer_release);

  napi_value undef;
  NAPI_CALL(env, napi_get_undefined(env, &undef));
  return undef;
}

static napi_value napi_poly_ctx_set_frontend_buffer_release(napi_env env, napi_callback_info info) {
  napi_value argv[2];
  size_t argc = 2;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);

  if (g_frontend_buffer_release_ref) {
    napi_delete_reference(env, g_frontend_buffer_release_ref);
    g_frontend_buffer_release_ref = NULL;
  }
  g_frontend_buffer_release_env = env;
  NAPI_CALL(env, napi_create_reference(env, argv[1], 1, &g_frontend_buffer_release_ref));
  poly_ctx_set_frontend_buffer_release(ctx, napi_frontend_buffer_release);

  napi_value undef;
  NAPI_CALL(env, napi_get_undefined(env, &undef));
  return undef;
}

/* Copy `nbytes` from the buffer's host pointer into a fresh Node Buffer.
 * Used by JS Tensor.toArray() / item() / tolist() when no JS-side host
 * owner is registered (e.g. realize() allocated the buffer in C). */
static napi_value napi_poly_buffer_read(napi_env env, napi_callback_info info) {
  napi_value argv[3];
  size_t argc = 3;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *buf = get_external(env, argv[1]);
  int64_t nbytes;
  napi_get_value_int64(env, argv[2], &nbytes);
  napi_value out;
  if (nbytes <= 0) {
    napi_create_buffer(env, 0, NULL, &out);
    return out;
  }
  void *dst;
  napi_create_buffer(env, (size_t)nbytes, &dst, &out);
  if (poly_buffer_read(ctx, buf, dst, (size_t)nbytes) != 0) {
    napi_throw_error(env, NULL, "polygrad: poly_buffer_read failed");
    return NULL;
  }
  return out;
}

static napi_value napi_poly_buffer_write(napi_env env, napi_callback_info info) {
  napi_value argv[3];
  size_t argc = 3;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *buf = get_external(env, argv[1]);
  void *src = NULL;
  size_t nbytes = 0;
  if (!read_bytes_arg(env, argv[2], &src, &nbytes)) return NULL;
  if (poly_buffer_write(ctx, buf, src, nbytes) != 0) {
    napi_throw_error(env, NULL, "polygrad: poly_buffer_write failed");
    return NULL;
  }
  napi_value undef;
  NAPI_CALL(env, napi_get_undefined(env, &undef));
  return undef;
}

static napi_value napi_poly_buffer_ensure_device_allocated(napi_env env, napi_callback_info info) {
  napi_value argv[3];
  size_t argc = 3;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *buf = get_external(env, argv[1]);
  int32_t device = 0;
  NAPI_CALL(env, napi_get_value_int32(env, argv[2], &device));
  if (poly_buffer_ensure_device_allocated(ctx, buf, (PolyDevice)device) != 0) {
    napi_throw_error(env, NULL, "polygrad: poly_buffer_ensure_device_allocated failed");
    return NULL;
  }
  napi_value undef;
  NAPI_CALL(env, napi_get_undefined(env, &undef));
  return undef;
}

static napi_value napi_poly_ctx_set_preferred_device(napi_env env, napi_callback_info info) {
  napi_value argv[2];
  size_t argc = 2;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  int32_t device = 0;
  NAPI_CALL(env, napi_get_value_int32(env, argv[1], &device));
  poly_ctx_set_preferred_device(ctx, (PolyDevice)device);
  napi_value undef;
  NAPI_CALL(env, napi_get_undefined(env, &undef));
  return undef;
}

static napi_value napi_poly_ctx_set_logical_policy(napi_env env, napi_callback_info info) {
  napi_value argv[2], out;
  size_t argc = 2;
  int32_t policy = 0;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  NAPI_CALL(env, napi_get_value_int32(env, argv[1], &policy));
  int rc = poly_ctx_set_logical_policy(get_external(env, argv[0]), (PolyLogicalPolicy)policy);
  NAPI_CALL(env, napi_create_int32(env, rc, &out));
  return out;
}

static napi_value napi_poly_ctx_get_logical_policy(napi_env env, napi_callback_info info) {
  napi_value argv[1], out;
  size_t argc = 1;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  NAPI_CALL(
      env,
      napi_create_int32(env, (int32_t)poly_ctx_get_logical_policy(get_external(env, argv[0])), &out)
  );
  return out;
}

static napi_value napi_poly_ctx_stats(napi_env env, napi_callback_info info) {
  napi_value argv[1];
  size_t argc = 1;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyCtxStats s;
  if (poly_ctx_stats(ctx, &s) != 0) {
    napi_throw_error(env, NULL, "polygrad: poly_ctx_stats failed");
    return NULL;
  }
  napi_value out;
  NAPI_CALL(env, napi_create_object(env, &out));
  set_named_size(env, out, "arenaBytes", s.arena_bytes);
  set_named_size(env, out, "arenaHighWater", s.arena_high_water);
  set_named_size(env, out, "scratchBytes", s.scratch_bytes);
  set_named_size(env, out, "scratchHighWater", s.scratch_high_water);
  set_named_size(env, out, "cseEntries", s.cse_entries);
  set_named_size(env, out, "toProgramCacheEntries", s.to_program_cache_entries);
  set_named_size(env, out, "runtimeCacheEntries", s.runtime_cache_entries);
  set_named_size(env, out, "shapeCacheEntries", s.shape_cache_entries);
  set_named_size(env, out, "bufferEntries", s.buffer_entries);
  set_named_size(env, out, "bufferOwnedBytes", s.buffer_owned_bytes);
  set_named_size(env, out, "bufferOwnedCurrentBytes", s.buffer_owned_current_bytes);
  set_named_size(env, out, "bufferOwnedSourceBytes", s.buffer_owned_source_bytes);
  set_named_size(env, out, "tensorRecords", s.tensor_records);
  set_named_size(env, out, "registryEntries", s.registry_entries);
  set_named_size(env, out, "entrypointEntries", s.entrypoint_entries);
  set_named_size(env, out, "compiledArtifactBytes", s.compiled_artifact_bytes);
  set_named_size(env, out, "runtimeArtifactEntries", s.runtime_artifact_entries);
  set_named_size(env, out, "launchCount", s.launch_count);
  set_named_size(env, out, "runtimeCacheHits", s.runtime_cache_hits);
  set_named_size(env, out, "runtimeCacheMisses", s.runtime_cache_misses);
  set_named_size(env, out, "bufferReadCount", s.buffer_read_count);
  set_named_size(env, out, "bufferReadBytes", s.buffer_read_bytes);
  set_named_size(env, out, "bufferWriteCount", s.buffer_write_count);
  set_named_size(env, out, "bufferWriteBytes", s.buffer_write_bytes);
  set_named_size(env, out, "bufferCopyCount", s.buffer_copy_count);
  set_named_size(env, out, "bufferCopyBytes", s.buffer_copy_bytes);
  set_named_u64(env, out, "globalOps", s.global_ops);
  set_named_u64(env, out, "globalMem", s.global_mem);
  set_named_double(env, out, "timeSumS", s.time_sum_s);
  set_named_u64(env, out, "kernelCount", s.kernel_count);
  set_named_u64(env, out, "memUsed", s.mem_used);
  return out;
}

static napi_value napi_poly_ctx_collect(napi_env env, napi_callback_info info) {
  napi_value argv[1], out;
  size_t argc = 1;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  int rc = poly_ctx_collect(get_external(env, argv[0]));
  NAPI_CALL(env, napi_create_int32(env, rc, &out));
  return out;
}

static napi_value napi_poly_ctx_reset_counters(napi_env env, napi_callback_info info) {
  napi_value argv[1];
  size_t argc = 1;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  poly_ctx_reset_counters(ctx);
  napi_value undef;
  NAPI_CALL(env, napi_get_undefined(env, &undef));
  return undef;
}

static napi_value napi_poly_can_run_op(napi_env env, napi_callback_info info) {
  napi_value argv[5];
  size_t argc = 5;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  int32_t device = 0, dtype_id = 0;
  napi_get_value_int32(env, argv[1], &device);
  char *op = read_utf8_arg(env, argv[2], NULL);
  if (!op) return NULL;
  napi_get_value_int32(env, argv[3], &dtype_id);
  int64_t shape[POLY_MAX_DIMS];
  int ndim = read_int64_array(env, argv[4], shape, POLY_MAX_DIMS);
  int rc = poly_can_run_op(ctx, device, op, dtype_id, shape, ndim);
  free(op);
  napi_value out;
  NAPI_CALL(env, napi_create_int32(env, rc, &out));
  return out;
}

/* ── Autograd ──────────────────────────────────────────────────────────── */

static napi_value napi_poly_grad(napi_env env, napi_callback_info info) {
  napi_value argv[3];
  size_t argc = 3;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *loss = get_external(env, argv[1]);
  PolyUOp *wrt = get_external(env, argv[2]);
  return make_external(env, poly_grad(ctx, loss, wrt));
}

static napi_value napi_poly_grad_many(napi_env env, napi_callback_info info) {
  napi_value argv[4];
  size_t argc = 4;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *loss = get_external(env, argv[1]);
  PolyUOp *initial_grad = get_external(env, argv[2]);

  bool is_arr = false;
  napi_is_array(env, argv[3], &is_arr);
  if (!is_arr) {
    napi_value empty;
    NAPI_CALL(env, napi_create_array_with_length(env, 0, &empty));
    return empty;
  }

  uint32_t n = 0;
  napi_get_array_length(env, argv[3], &n);
  PolyUOp **wrts = (PolyUOp **)malloc((size_t)n * sizeof(PolyUOp *));
  PolyUOp **out_grads = (PolyUOp **)calloc((size_t)n, sizeof(PolyUOp *));
  uint8_t *out_present = (uint8_t *)calloc((size_t)n, sizeof(uint8_t));
  if (!wrts || !out_grads || !out_present) {
    free(wrts);
    free(out_grads);
    free(out_present);
    napi_value null_value;
    napi_get_null(env, &null_value);
    return null_value;
  }

  for (uint32_t i = 0; i < n; i++) {
    napi_value el;
    napi_get_element(env, argv[3], i, &el);
    wrts[i] = get_external(env, el);
  }

  /* tinygrad computes gradients for all live targets in one reverse pass.
   * Exposing poly_grad_many keeps JS backward from mutating live tensors
   * between per-target gradient builds. Presence preserves the distinction
   * between an absent/NOOP gradient and a numerical zero. */
  int rc = poly_grad_many_ex(ctx, loss, initial_grad, wrts, (int)n, out_grads, out_present);
  free(wrts);
  if (rc != 0) {
    free(out_grads);
    free(out_present);
    napi_value null_value;
    napi_get_null(env, &null_value);
    return null_value;
  }

  napi_value result, grads_js, present_js;
  NAPI_CALL(env, napi_create_object(env, &result));
  NAPI_CALL(env, napi_create_array_with_length(env, n, &grads_js));
  NAPI_CALL(env, napi_create_array_with_length(env, n, &present_js));
  for (uint32_t i = 0; i < n; i++) {
    napi_value grad_el = make_external(env, out_grads[i]);
    napi_value present_el;
    NAPI_CALL(env, napi_get_boolean(env, out_present[i] != 0, &present_el));
    NAPI_CALL(env, napi_set_element(env, grads_js, i, grad_el));
    NAPI_CALL(env, napi_set_element(env, present_js, i, present_el));
  }
  free(out_grads);
  free(out_present);
  NAPI_CALL(env, napi_set_named_property(env, result, "grads", grads_js));
  NAPI_CALL(env, napi_set_named_property(env, result, "present", present_js));
  return result;
}

/* ── Detach ────────────────────────────────────────────────────────────── */

NAPI_UNARY(poly_detach)

/* ── Cast ──────────────────────────────────────────────────────────────── */

static napi_value napi_poly_cast_by_id(napi_env env, napi_callback_info info) {
  napi_value argv[3];
  size_t argc = 3;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *x = get_external(env, argv[1]);
  int32_t dtype_id;
  napi_get_value_int32(env, argv[2], &dtype_id);
  PolyUOp *r = poly_cast_by_id(ctx, x, dtype_id);
  if (!r) {
    napi_throw_range_error(env, NULL, "poly_cast_by_id failed");
    return NULL;
  }
  return make_external(env, r);
}

static napi_value napi_poly_dtype_id_by_name(napi_env env, napi_callback_info info) {
  napi_value argv[1];
  size_t argc = 1;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  char name[64];
  size_t len = 0;
  NAPI_CALL(env, napi_get_value_string_utf8(env, argv[0], name, sizeof(name), &len));
  napi_value out;
  NAPI_CALL(env, napi_create_int32(env, poly_dtype_id_by_name(name), &out));
  return out;
}

static napi_value napi_poly_uop_dtype_id(napi_env env, napi_callback_info info) {
  napi_value argv[2];
  size_t argc = 2;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *u = get_external(env, argv[1]);
  napi_value out;
  NAPI_CALL(env, napi_create_int32(env, poly_uop_dtype_id(ctx, u), &out));
  return out;
}

static napi_value napi_poly_device_by_name(napi_env env, napi_callback_info info) {
  napi_value argv[1];
  size_t argc = 1;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  char name[64];
  size_t len = 0;
  NAPI_CALL(env, napi_get_value_string_utf8(env, argv[0], name, sizeof(name), &len));
  napi_value out;
  NAPI_CALL(env, napi_create_int32(env, poly_device_by_name(name), &out));
  return out;
}

static napi_value napi_poly_device_name(napi_env env, napi_callback_info info) {
  napi_value argv[1];
  size_t argc = 1;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  int32_t device = 0;
  napi_get_value_int32(env, argv[0], &device);
  napi_value out;
  NAPI_CALL(
      env,
      napi_create_string_utf8(env, poly_device_name((PolyDevice)device), NAPI_AUTO_LENGTH, &out)
  );
  return out;
}

static napi_value napi_poly_device_is_host_addressable(napi_env env, napi_callback_info info) {
  napi_value argv[1];
  size_t argc = 1;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  int32_t device = 0;
  NAPI_CALL(env, napi_get_value_int32(env, argv[0], &device));
  napi_value out;
  NAPI_CALL(env, napi_get_boolean(env, poly_device_is_host_addressable((PolyDevice)device), &out));
  return out;
}

/* ── Shape-taking movement ops ─────────────────────────────────────────── */

#define MAX_DIMS 16

static napi_value napi_poly_reshape(napi_env env, napi_callback_info info) {
  napi_value argv[4];
  size_t argc = 4;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *uop = get_external(env, argv[1]);
  int64_t dims[MAX_DIMS];
  int ndim = read_int64_array(env, argv[2], dims, MAX_DIMS);
  /* argv[3] is ndim from JS but we use the array length */
  (void)ndim;
  int32_t nd;
  napi_get_value_int32(env, argv[3], &nd);
  return make_external(env, poly_reshape(ctx, uop, dims, nd));
}

static napi_value napi_poly_expand(napi_env env, napi_callback_info info) {
  napi_value argv[4];
  size_t argc = 4;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *uop = get_external(env, argv[1]);
  int64_t dims[MAX_DIMS];
  int32_t nd;
  napi_get_value_int32(env, argv[3], &nd);
  read_int64_array(env, argv[2], dims, MAX_DIMS);
  return make_external(env, poly_expand(ctx, uop, dims, nd));
}

static napi_value napi_poly_permute(napi_env env, napi_callback_info info) {
  napi_value argv[4];
  size_t argc = 4;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *uop = get_external(env, argv[1]);
  int64_t order[MAX_DIMS];
  int32_t nd;
  napi_get_value_int32(env, argv[3], &nd);
  read_int64_array(env, argv[2], order, MAX_DIMS);
  return make_external(env, poly_permute(ctx, uop, order, nd));
}

static napi_value napi_poly_flip(napi_env env, napi_callback_info info) {
  napi_value argv[4];
  size_t argc = 4;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *uop = get_external(env, argv[1]);
  int64_t axes[MAX_DIMS];
  int32_t naxes;
  napi_get_value_int32(env, argv[3], &naxes);
  read_int64_array(env, argv[2], axes, MAX_DIMS);
  return make_external(env, poly_flip(ctx, uop, axes, naxes));
}

static napi_value napi_poly_shrink(napi_env env, napi_callback_info info) {
  napi_value argv[4];
  size_t argc = 4;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *uop = get_external(env, argv[1]);
  int64_t flat[MAX_DIMS * 2];
  int32_t npairs;
  napi_get_value_int32(env, argv[3], &npairs);
  read_int64_array(env, argv[2], flat, MAX_DIMS * 2);
  /* poly_shrink expects int64_t (*)[2] -- flat layout is compatible */
  return make_external(env, poly_shrink(ctx, uop, (int64_t(*)[2])flat, npairs));
}

static napi_value napi_poly_shrink_uop(napi_env env, napi_callback_info info) {
  napi_value argv[5];
  size_t argc = 5;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *uop = get_external(env, argv[1]);
  int n_starts = 0, n_sizes = 0;
  PolyUOp **starts = read_uop_array(env, argv[2], &n_starts);
  PolyUOp **sizes = read_uop_array(env, argv[3], &n_sizes);
  int32_t ndim = 0;
  napi_get_value_int32(env, argv[4], &ndim);
  if (!starts || !sizes || n_starts != n_sizes || n_starts != ndim) {
    free(starts);
    free(sizes);
    return NULL;
  }
  PolyUOp *ret = poly_shrink_uop(ctx, uop, starts, sizes, ndim);
  free(starts);
  free(sizes);
  return make_external(env, ret);
}

static napi_value napi_poly_pad(napi_env env, napi_callback_info info) {
  napi_value argv[4];
  size_t argc = 4;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *uop = get_external(env, argv[1]);
  int64_t flat[MAX_DIMS * 2];
  int32_t npairs;
  napi_get_value_int32(env, argv[3], &npairs);
  read_int64_array(env, argv[2], flat, MAX_DIMS * 2);
  return make_external(env, poly_pad(ctx, uop, (int64_t(*)[2])flat, npairs));
}

static napi_value napi_poly_pad_value(napi_env env, napi_callback_info info) {
  napi_value argv[5];
  size_t argc = 5;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *uop = get_external(env, argv[1]);
  int64_t flat[MAX_DIMS * 2];
  int32_t npairs;
  double value;
  read_int64_array(env, argv[2], flat, MAX_DIMS * 2);
  napi_get_value_int32(env, argv[3], &npairs);
  napi_get_value_double(env, argv[4], &value);
  return make_external(env, poly_pad_value(ctx, uop, (int64_t(*)[2])flat, npairs, value));
}

static napi_value napi_poly_pool(napi_env env, napi_callback_info info) {
  napi_value argv[6];
  size_t argc = 6;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *uop = get_external(env, argv[1]);
  int64_t k[MAX_DIMS], stride[MAX_DIMS], dilation[MAX_DIMS];
  int32_t nk;
  read_int64_array(env, argv[2], k, MAX_DIMS);
  napi_get_value_int32(env, argv[3], &nk);
  read_int64_array(env, argv[4], stride, MAX_DIMS);
  read_int64_array(env, argv[5], dilation, MAX_DIMS);
  return make_external(env, poly_pool(ctx, uop, k, nk, stride, dilation));
}

static napi_value napi_poly_max_pool2d(napi_env env, napi_callback_info info) {
  napi_value argv[8];
  size_t argc = 8;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *x = get_external(env, argv[1]);
  int64_t k[MAX_DIMS], stride[MAX_DIMS], dilation[MAX_DIMS], padding[MAX_DIMS * 2];
  int32_t nk, npadding;
  read_int64_array(env, argv[2], k, MAX_DIMS);
  napi_get_value_int32(env, argv[3], &nk);
  read_int64_array(env, argv[4], stride, MAX_DIMS);
  read_int64_array(env, argv[5], dilation, MAX_DIMS);
  read_int64_array(env, argv[6], padding, MAX_DIMS * 2);
  napi_get_value_int32(env, argv[7], &npadding);
  return make_external(env, poly_max_pool2d(ctx, x, k, nk, stride, dilation, padding, npadding));
}

static napi_value napi_poly_conv2d(napi_env env, napi_callback_info info) {
  napi_value argv[9];
  size_t argc = 9;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *x = get_external(env, argv[1]);
  PolyUOp *weight = get_external(env, argv[2]);
  PolyUOp *bias = get_external_nullable(env, argv[3]);
  int32_t groups, npadding;
  int64_t stride[MAX_DIMS], dilation[MAX_DIMS], padding[MAX_DIMS * 2];
  napi_get_value_int32(env, argv[4], &groups);
  read_int64_array(env, argv[5], stride, MAX_DIMS);
  read_int64_array(env, argv[6], dilation, MAX_DIMS);
  read_int64_array(env, argv[7], padding, MAX_DIMS * 2);
  napi_get_value_int32(env, argv[8], &npadding);
  return make_external(
      env, poly_conv2d(ctx, x, weight, bias, groups, stride, dilation, padding, npadding)
  );
}

static napi_value napi_poly_batchnorm(napi_env env, napi_callback_info info) {
  napi_value argv[8];
  size_t argc = 8;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *x = get_external(env, argv[1]);
  PolyUOp *weight = get_external_nullable(env, argv[2]);
  PolyUOp *bias = get_external_nullable(env, argv[3]);
  PolyUOp *mean = get_external(env, argv[4]);
  PolyUOp *invstd = get_external(env, argv[5]);
  int64_t axes[MAX_DIMS];
  int32_t naxes;
  read_int64_array(env, argv[6], axes, MAX_DIMS);
  napi_get_value_int32(env, argv[7], &naxes);
  return make_external(env, poly_batchnorm(ctx, x, weight, bias, mean, invstd, axes, naxes));
}

static napi_value napi_poly_reduce_axis(napi_env env, napi_callback_info info) {
  napi_value argv[5];
  size_t argc = 5;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  int32_t op;
  napi_get_value_int32(env, argv[1], &op);
  PolyUOp *uop = get_external(env, argv[2]);
  int64_t axes[MAX_DIMS];
  int32_t naxes;
  napi_get_value_int32(env, argv[4], &naxes);
  read_int64_array(env, argv[3], axes, MAX_DIMS);
  return make_external(env, poly_reduce_axis(ctx, (PolyOps)op, uop, axes, naxes));
}

/* Old explicit-shape ops removed -- merged into v2 section below */

/* logsumexp: shape read from UOp */
static napi_value napi_poly_logsumexp(napi_env env, napi_callback_info info) {
  napi_value argv[4];
  size_t argc = 4;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *uop = get_external(env, argv[1]);
  int32_t axis, keepdim;
  napi_get_value_int32(env, argv[2], &axis);
  napi_get_value_int32(env, argv[3], &keepdim);
  PolyUOp *r = poly_logsumexp(ctx, uop, axis, keepdim);
  int64_t out_shape[MAX_DIMS];
  int out_ndim = poly_uop_ndim(ctx, (const PolyUOp *)r);
  if (out_ndim > 0) {
    const int64_t *dims = poly_uop_max_shape_dims(ctx, (const PolyUOp *)r);
    if (dims) memcpy(out_shape, dims, out_ndim * sizeof(int64_t));
  }
  return make_shape_result(env, r, out_shape, out_ndim);
}

/* ── Composed elementwise (unary) ──────────────────────────────────────── */

NAPI_UNARY(poly_exp)
NAPI_UNARY(poly_log)
NAPI_UNARY(poly_log1p)
NAPI_UNARY(poly_expm1)
NAPI_UNARY(poly_sin)
NAPI_UNARY(poly_cos)
NAPI_UNARY(poly_tan)
NAPI_UNARY(poly_erf)
NAPI_UNARY(poly_erfc)
NAPI_UNARY(poly_erfinv)
NAPI_UNARY(poly_ndtri)
NAPI_UNARY(poly_digamma)
NAPI_UNARY(poly_lgamma)
NAPI_UNARY(poly_sigmoid)
NAPI_UNARY(poly_tanh_act)
NAPI_UNARY(poly_abs)
NAPI_UNARY(poly_sign)
NAPI_UNARY(poly_square)
NAPI_UNARY(poly_rsqrt)
NAPI_UNARY(poly_ceil)
NAPI_UNARY(poly_floor)
NAPI_UNARY(poly_round_f)
NAPI_UNARY(poly_isinf)
NAPI_UNARY(poly_isnan)
NAPI_UNARY(poly_relu)
NAPI_UNARY(poly_relu6)
NAPI_UNARY(poly_gelu)
NAPI_UNARY(poly_quick_gelu)
NAPI_UNARY(poly_silu)
NAPI_UNARY(poly_mish)
NAPI_UNARY(poly_hardswish)
NAPI_UNARY(poly_hardsigmoid)

/* ── Composed with extra double args ───────────────────────────────────── */

static napi_value napi_poly_leaky_relu(napi_env env, napi_callback_info info) {
  napi_value argv[3];
  size_t argc = 3;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *x = get_external(env, argv[1]);
  double neg_slope;
  napi_get_value_double(env, argv[2], &neg_slope);
  return make_external(env, poly_leaky_relu(ctx, x, neg_slope));
}

static napi_value napi_poly_elu(napi_env env, napi_callback_info info) {
  napi_value argv[3];
  size_t argc = 3;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *x = get_external(env, argv[1]);
  double alpha;
  napi_get_value_double(env, argv[2], &alpha);
  return make_external(env, poly_elu(ctx, x, alpha));
}

static napi_value napi_poly_softplus(napi_env env, napi_callback_info info) {
  napi_value argv[3];
  size_t argc = 3;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *x = get_external(env, argv[1]);
  double beta;
  napi_get_value_double(env, argv[2], &beta);
  return make_external(env, poly_softplus(ctx, x, beta));
}

static napi_value napi_poly_hardtanh(napi_env env, napi_callback_info info) {
  napi_value argv[4];
  size_t argc = 4;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *x = get_external(env, argv[1]);
  double min_val, max_val;
  napi_get_value_double(env, argv[2], &min_val);
  napi_get_value_double(env, argv[3], &max_val);
  return make_external(env, poly_hardtanh(ctx, x, min_val, max_val));
}

static napi_value napi_poly_clamp(napi_env env, napi_callback_info info) {
  napi_value argv[4];
  size_t argc = 4;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *x = get_external(env, argv[1]);
  double lo, hi;
  napi_get_value_double(env, argv[2], &lo);
  napi_get_value_double(env, argv[3], &hi);
  return make_external(env, poly_clamp(ctx, x, lo, hi));
}

/* ── Comparisons (binary) ──────────────────────────────────────────────── */

NAPI_BINARY(poly_eq)
NAPI_BINARY(poly_ne)
NAPI_BINARY(poly_gt)
NAPI_BINARY(poly_ge)
NAPI_BINARY(poly_le)
NAPI_BINARY(poly_maximum)
NAPI_BINARY(poly_minimum)

/* where is ternary: (ctx, cond, x, y) */
static napi_value napi_poly_where_op(napi_env env, napi_callback_info info) {
  napi_value argv[4];
  size_t argc = 4;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *cond = get_external(env, argv[1]);
  PolyUOp *x = get_external(env, argv[2]);
  PolyUOp *y = get_external(env, argv[3]);
  return make_external(env, poly_where_op(ctx, cond, x, y));
}

/* ── Creation (shape-taking) ───────────────────────────────────────────── */

static napi_value napi_poly_rand(napi_env env, napi_callback_info info) {
  napi_value argv[4];
  size_t argc = 4;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  int64_t shape[MAX_DIMS];
  int32_t ndim;
  read_int64_array(env, argv[1], shape, MAX_DIMS);
  napi_get_value_int32(env, argv[2], &ndim);
  int64_t seed;
  napi_get_value_int64(env, argv[3], &seed);
  return make_external(env, poly_rand(ctx, shape, ndim, (uint64_t)seed));
}

static napi_value napi_poly_randn(napi_env env, napi_callback_info info) {
  napi_value argv[4];
  size_t argc = 4;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  int64_t shape[MAX_DIMS];
  int32_t ndim;
  read_int64_array(env, argv[1], shape, MAX_DIMS);
  napi_get_value_int32(env, argv[2], &ndim);
  int64_t seed;
  napi_get_value_int64(env, argv[3], &seed);
  return make_external(env, poly_randn(ctx, shape, ndim, (uint64_t)seed));
}

static napi_value napi_poly_arange(napi_env env, napi_callback_info info) {
  napi_value argv[4];
  size_t argc = 4;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  double start, stop, step;
  napi_get_value_double(env, argv[1], &start);
  napi_get_value_double(env, argv[2], &stop);
  napi_get_value_double(env, argv[3], &step);
  return make_external(env, poly_arange(ctx, start, stop, step));
}

static napi_value napi_poly_eye(napi_env env, napi_callback_info info) {
  napi_value argv[2];
  size_t argc = 2;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  int64_t n;
  napi_get_value_int64(env, argv[1], &n);
  return make_external(env, poly_eye(ctx, n));
}

static napi_value napi_poly_linspace(napi_env env, napi_callback_info info) {
  napi_value argv[4];
  size_t argc = 4;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  double start, stop;
  int64_t steps;
  napi_get_value_double(env, argv[1], &start);
  napi_get_value_double(env, argv[2], &stop);
  napi_get_value_int64(env, argv[3], &steps);
  return make_external(env, poly_linspace(ctx, start, stop, steps));
}

static napi_value napi_poly_full(napi_env env, napi_callback_info info) {
  napi_value argv[4];
  size_t argc = 4;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  int64_t shape[MAX_DIMS];
  int32_t ndim;
  read_int64_array(env, argv[1], shape, MAX_DIMS);
  napi_get_value_int32(env, argv[2], &ndim);
  double fill;
  napi_get_value_double(env, argv[3], &fill);
  return make_external(env, poly_full(ctx, shape, ndim, fill));
}

static napi_value napi_poly_tril(napi_env env, napi_callback_info info) {
  napi_value argv[3];
  size_t argc = 3;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *x = get_external(env, argv[1]);
  int32_t diagonal;
  napi_get_value_int32(env, argv[2], &diagonal);
  return make_external(env, poly_tril(ctx, x, diagonal));
}

static napi_value napi_poly_triu(napi_env env, napi_callback_info info) {
  napi_value argv[3];
  size_t argc = 3;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *x = get_external(env, argv[1]);
  int32_t diagonal;
  napi_get_value_int32(env, argv[2], &diagonal);
  return make_external(env, poly_triu(ctx, x, diagonal));
}

/* ── Metadata ──────────────────────────────────────────────────────────── */

static napi_value napi_poly_op_count(napi_env env, napi_callback_info info) {
  (void)info;
  napi_value result;
  NAPI_CALL(env, napi_create_int32(env, poly_op_count(), &result));
  return result;
}

static napi_value napi_poly_op_name(napi_env env, napi_callback_info info) {
  napi_value argv[1];
  size_t argc = 1;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  int32_t i;
  napi_get_value_int32(env, argv[0], &i);
  const char *name = poly_op_name((PolyOps)i);
  napi_value result;
  if (name) {
    NAPI_CALL(env, napi_create_string_utf8(env, name, strlen(name), &result));
  } else {
    napi_get_null(env, &result);
  }
  return result;
}

static napi_value napi_poly_abi_version(napi_env env, napi_callback_info info) {
  (void)info;
  napi_value result;
  NAPI_CALL(env, napi_create_int32(env, poly_abi_version(), &result));
  return result;
}

/* ── Layernorm ─────────────────────────────────────────────────────────── */

static napi_value napi_poly_layernorm(napi_env env, napi_callback_info info) {
  napi_value argv[6];
  size_t argc = 6;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *uop = get_external(env, argv[1]);
  int32_t axis;
  /* argv[2]/argv[3] are legacy explicit shape inputs. Current core UOps own
   * shape, so this wrapper only preserves the old JS call signature. */
  napi_get_value_int32(env, argv[4], &axis);
  double eps;
  napi_get_value_double(env, argv[5], &eps);
  int64_t out_shape[MAX_DIMS];
  int out_ndim = 0;
  PolyUOp *r = poly_layernorm_apply(ctx, uop, NULL, NULL, axis, eps);
  if (r) {
    out_ndim = poly_uop_ndim(ctx, (const PolyUOp *)r);
    const int64_t *dims = poly_uop_max_shape_dims(ctx, (const PolyUOp *)r);
    if (dims && out_ndim > 0) memcpy(out_shape, dims, out_ndim * sizeof(int64_t));
  }
  return make_shape_result(env, r, out_shape, out_ndim);
}

/* Old gather removed -- merged into v2 section below */

/* ── Linear ────────────────────────────────────────────────────────────── */

static napi_value napi_poly_linear(napi_env env, napi_callback_info info) {
  napi_value argv[10];
  size_t argc = 10;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *x = get_external(env, argv[1]);
  /* argv[2], argv[3]: x_shape, x_ndim (ignored — shape on UOp) */
  PolyUOp *w = get_external(env, argv[4]);
  /* argv[5], argv[6]: w_shape, w_ndim (ignored) */
  napi_valuetype bias_type;
  napi_typeof(env, argv[7], &bias_type);
  PolyUOp *bias = NULL;
  if (bias_type == napi_external) bias = get_external(env, argv[7]);

  /* dot + bias add (poly_add handles broadcasting) */
  PolyUOp *r = poly_dot(ctx, x, w);
  if (r && bias) r = poly_add(ctx, r, bias);

  int64_t out_shape[MAX_DIMS];
  int out_ndim = 0;
  if (r) {
    PolyShape s = poly_uop_max_shape(ctx, r);
    out_ndim = s.ndim;
    if (s.ndim > 0) memcpy(out_shape, s.dims, s.ndim * sizeof(int64_t));
  }
  return make_shape_result(env, r, out_shape, out_ndim);
}

/* ── Causal mask ───────────────────────────────────────────────────────── */

static napi_value napi_poly_causal_mask(napi_env env, napi_callback_info info) {
  napi_value argv[2];
  size_t argc = 2;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  int64_t T;
  napi_get_value_int64(env, argv[1], &T);
  int64_t out_shape[MAX_DIMS];
  int out_ndim = 0;
  PolyUOp *r = poly_causal_mask(ctx, T);
  if (r) {
    out_ndim = poly_uop_ndim(ctx, (const PolyUOp *)r);
    const int64_t *dims = poly_uop_max_shape_dims(ctx, (const PolyUOp *)r);
    if (dims && out_ndim > 0) memcpy(out_shape, dims, out_ndim * sizeof(int64_t));
  }
  return make_shape_result(env, r, out_shape, out_ndim);
}

/* ── Sink-n ────────────────────────────────────────────────────────────── */

static napi_value napi_poly_sink_n(napi_env env, napi_callback_info info) {
  napi_value argv[3];
  size_t argc = 3;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  /* argv[1] is an array of externals, argv[2] is count */
  uint32_t n = 0;
  napi_get_array_length(env, argv[1], &n);
  PolyUOp *stores[64];
  if (n > 64) n = 64;
  for (uint32_t i = 0; i < n; i++) {
    napi_value elem;
    napi_get_element(env, argv[1], i, &elem);
    stores[i] = get_external(env, elem);
  }
  return make_external(env, poly_sink_n(ctx, stores, (int)n));
}

/* ── Cache cleanup ─────────────────────────────────────────────────────── */

static napi_value napi_poly_cpu_cache_flush(napi_env env, napi_callback_info info) {
  (void)info;
  poly_cpu_cache_flush();
  napi_value undef;
  napi_get_undefined(env, &undef);
  return undef;
}

/* ── PolyModel / model runtime ────────────────────────────────────── */

static napi_value napi_poly_model_from_ir(napi_env env, napi_callback_info info) {
  napi_value argv[2];
  size_t argc = 2;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));

  napi_typedarray_type ir_type;
  size_t ir_len = 0;
  void *ir_data = NULL;
  size_t ir_offset;
  napi_value ir_buf;
  NAPI_CALL(
      env, napi_get_typedarray_info(env, argv[0], &ir_type, &ir_len, &ir_data, &ir_buf, &ir_offset)
  );
  (void)ir_type;

  const uint8_t *weights_data = NULL;
  size_t weights_len = 0;
  if (argc > 1) {
    napi_valuetype weights_type;
    napi_typeof(env, argv[1], &weights_type);
    if (weights_type != napi_null && weights_type != napi_undefined) {
      napi_typedarray_type wa_type;
      void *wa_data = NULL;
      size_t wa_offset;
      napi_value wa_buf;
      NAPI_CALL(
          env, napi_get_typedarray_info(
                   env, argv[1], &wa_type, &weights_len, &wa_data, &wa_buf, &wa_offset
               )
      );
      (void)wa_type;
      weights_data = (const uint8_t *)wa_data;
    }
  }

  PolyModel *inst =
      poly_model_from_ir((const uint8_t *)ir_data, (int)ir_len, weights_data, (int)weights_len);

  if (!inst) {
    napi_value result;
    napi_get_null(env, &result);
    return result;
  }
  return make_external(env, inst);
}

static napi_value napi_poly_model_from_program(napi_env env, napi_callback_info info) {
  napi_value argv[2];
  size_t argc = 2;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));

  napi_typedarray_type program_type;
  size_t program_len = 0, program_offset = 0;
  void *program_data = NULL;
  napi_value program_buf;
  NAPI_CALL(
      env,
      napi_get_typedarray_info(
          env, argv[0], &program_type, &program_len, &program_data, &program_buf, &program_offset
      )
  );
  (void)program_type;

  const uint8_t *weights_data = NULL;
  size_t weights_len = 0;
  if (argc > 1) {
    napi_valuetype weights_type;
    napi_typeof(env, argv[1], &weights_type);
    if (weights_type != napi_null && weights_type != napi_undefined) {
      napi_typedarray_type wa_type;
      void *wa_data = NULL;
      size_t wa_offset = 0;
      napi_value wa_buf;
      NAPI_CALL(
          env, napi_get_typedarray_info(
                   env, argv[1], &wa_type, &weights_len, &wa_data, &wa_buf, &wa_offset
               )
      );
      (void)wa_type;
      weights_data = (const uint8_t *)wa_data;
    }
  }
  PolyModel *inst = poly_model_from_program(
      (const uint8_t *)program_data, (int)program_len, weights_data, (int)weights_len
  );
  if (!inst) {
    napi_value result;
    napi_get_null(env, &result);
    return result;
  }
  return make_external(env, inst);
}

static void free_string_array_items(char **items, uint32_t n) {
  if (!items) return;
  for (uint32_t i = 0; i < n; i++)
    free(items[i]);
  free(items);
}

static int read_string_array_alloc(
    napi_env env,
    napi_value arr_val,
    char ***out_items,
    uint32_t *out_n,
    bool allow_null_items
) {
  uint32_t n = 0;
  if (napi_get_array_length(env, arr_val, &n) != napi_ok) {
    napi_throw_error(env, NULL, "polygrad: expected string array");
    return 0;
  }
  char **items = calloc(n ? n : 1, sizeof(char *));
  if (!items) {
    napi_throw_error(env, NULL, "calloc failed");
    return 0;
  }
  for (uint32_t i = 0; i < n; i++) {
    napi_value val;
    if (napi_get_element(env, arr_val, i, &val) != napi_ok) {
      free_string_array_items(items, n);
      napi_throw_error(env, NULL, "polygrad: failed to read string array");
      return 0;
    }
    if (allow_null_items && napi_is_nullish(env, val)) {
      items[i] = NULL;
      continue;
    }
    items[i] = read_utf8_arg(env, val, NULL);
    if (!items[i]) {
      free_string_array_items(items, n);
      return 0;
    }
  }
  *out_items = items;
  *out_n = n;
  return 1;
}

static int read_i32_array_alloc(
    napi_env env,
    napi_value arr_val,
    int **out_items,
    uint32_t *out_n
) {
  uint32_t n = 0;
  if (napi_get_array_length(env, arr_val, &n) != napi_ok) {
    napi_throw_error(env, NULL, "polygrad: expected int array");
    return 0;
  }
  int *items = calloc(n ? n : 1, sizeof(int));
  if (!items) {
    napi_throw_error(env, NULL, "calloc failed");
    return 0;
  }
  for (uint32_t i = 0; i < n; i++) {
    napi_value val;
    int32_t tmp = 0;
    if (napi_get_element(env, arr_val, i, &val) != napi_ok ||
        napi_get_value_int32(env, val, &tmp) != napi_ok) {
      free(items);
      napi_throw_error(env, NULL, "polygrad: failed to read int array");
      return 0;
    }
    items[i] = tmp;
  }
  *out_items = items;
  *out_n = n;
  return 1;
}

static int read_u32_array_alloc(
    napi_env env,
    napi_value arr_val,
    uint32_t **out_items,
    uint32_t *out_n
) {
  uint32_t n = 0;
  if (napi_get_array_length(env, arr_val, &n) != napi_ok) {
    napi_throw_error(env, NULL, "polygrad: expected uint array");
    return 0;
  }
  uint32_t *items = calloc(n ? n : 1, sizeof(uint32_t));
  if (!items) {
    napi_throw_error(env, NULL, "calloc failed");
    return 0;
  }
  for (uint32_t i = 0; i < n; i++) {
    napi_value val;
    uint32_t tmp = 0;
    if (napi_get_element(env, arr_val, i, &val) != napi_ok ||
        napi_get_value_uint32(env, val, &tmp) != napi_ok) {
      free(items);
      napi_throw_error(env, NULL, "polygrad: failed to read uint array");
      return 0;
    }
    items[i] = tmp;
  }
  *out_items = items;
  *out_n = n;
  return 1;
}

static int read_tensor_array_alloc(
    napi_env env,
    napi_value arr_val,
    PolyTensor ***out_items,
    uint32_t *out_n
) {
  uint32_t n = 0;
  if (napi_get_array_length(env, arr_val, &n) != napi_ok) {
    napi_throw_error(env, NULL, "polygrad: expected tensor array");
    return 0;
  }
  PolyTensor **items = calloc(n ? n : 1, sizeof(PolyTensor *));
  if (!items) {
    napi_throw_error(env, NULL, "calloc failed");
    return 0;
  }
  for (uint32_t i = 0; i < n; i++) {
    napi_value val;
    if (napi_get_element(env, arr_val, i, &val) != napi_ok) {
      free(items);
      napi_throw_error(env, NULL, "polygrad: failed to read tensor array");
      return 0;
    }
    items[i] = get_external(env, val);
  }
  *out_items = items;
  *out_n = n;
  return 1;
}

static napi_value napi_poly_model_from_binding_arrays(napi_env env, napi_callback_info info) {
  napi_value argv[12];
  size_t argc = 12;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  if (argc < 12) {
    napi_throw_error(env, NULL, "polygrad: poly_model_from_binding_arrays expects 12 args");
    return NULL;
  }

  PolyCtx *ctx = get_external(env, argv[0]);
  char **binding_names = NULL;
  int *binding_roles = NULL;
  PolyTensor **binding_tensors = NULL;
  uint32_t *binding_flags = NULL;
  char **entry_names = NULL;
  char **entry_inputs = NULL;
  int *entry_input_counts = NULL;
  char **entry_outputs = NULL;
  int *entry_output_counts = NULL;
  char **entry_objectives = NULL;
  uint32_t *entry_flags = NULL;
  uint32_t n_binding_names = 0, n_binding_roles = 0, n_binding_tensors = 0, n_binding_flags = 0;
  uint32_t n_entry_names = 0, n_entry_inputs = 0, n_entry_input_counts = 0;
  uint32_t n_entry_outputs = 0, n_entry_output_counts = 0, n_entry_objectives = 0,
           n_entry_flags = 0;

  if (!read_string_array_alloc(env, argv[1], &binding_names, &n_binding_names, false) ||
      !read_i32_array_alloc(env, argv[2], &binding_roles, &n_binding_roles) ||
      !read_tensor_array_alloc(env, argv[3], &binding_tensors, &n_binding_tensors) ||
      !read_u32_array_alloc(env, argv[4], &binding_flags, &n_binding_flags) ||
      !read_string_array_alloc(env, argv[5], &entry_names, &n_entry_names, false) ||
      !read_string_array_alloc(env, argv[6], &entry_inputs, &n_entry_inputs, false) ||
      !read_i32_array_alloc(env, argv[7], &entry_input_counts, &n_entry_input_counts) ||
      !read_string_array_alloc(env, argv[8], &entry_outputs, &n_entry_outputs, false) ||
      !read_i32_array_alloc(env, argv[9], &entry_output_counts, &n_entry_output_counts) ||
      !read_string_array_alloc(env, argv[10], &entry_objectives, &n_entry_objectives, true) ||
      !read_u32_array_alloc(env, argv[11], &entry_flags, &n_entry_flags)) {
    goto fail;
  }

  if (n_binding_names != n_binding_roles || n_binding_names != n_binding_tensors ||
      n_binding_names != n_binding_flags || n_entry_names != n_entry_input_counts ||
      n_entry_names != n_entry_output_counts || n_entry_names != n_entry_objectives ||
      n_entry_names != n_entry_flags) {
    napi_throw_error(env, NULL, "polygrad: binding/entrypoint array length mismatch");
    goto fail;
  }

  PolyModelError err = {0};
  PolyModel *inst = poly_model_from_binding_arrays(
      ctx, (const char **)binding_names, binding_roles, binding_tensors, binding_flags,
      (int)n_binding_names, (const char **)entry_names, (const char **)entry_inputs,
      entry_input_counts, (const char **)entry_outputs, entry_output_counts,
      (const char **)entry_objectives, entry_flags, (int)n_entry_names, NULL, &err
  );

  free_string_array_items(binding_names, n_binding_names);
  free(binding_roles);
  free(binding_tensors);
  free(binding_flags);
  free_string_array_items(entry_names, n_entry_names);
  free_string_array_items(entry_inputs, n_entry_inputs);
  free(entry_input_counts);
  free_string_array_items(entry_outputs, n_entry_outputs);
  free(entry_output_counts);
  free_string_array_items(entry_objectives, n_entry_objectives);
  free(entry_flags);

  if (!inst) {
    napi_throw_error(env, NULL, err.message[0] ? err.message : "polygrad: instance build failed");
    return NULL;
  }
  return make_external(env, inst);

fail:
  free_string_array_items(binding_names, n_binding_names);
  free(binding_roles);
  free(binding_tensors);
  free(binding_flags);
  free_string_array_items(entry_names, n_entry_names);
  free_string_array_items(entry_inputs, n_entry_inputs);
  free(entry_input_counts);
  free_string_array_items(entry_outputs, n_entry_outputs);
  free(entry_output_counts);
  free_string_array_items(entry_objectives, n_entry_objectives);
  free(entry_flags);
  return NULL;
}

static napi_value napi_poly_model_from_sinks(napi_env env, napi_callback_info info) {
  napi_value argv[3];
  size_t argc = 3;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);

  uint32_t n_names = 0, n_sinks = 0;
  napi_get_array_length(env, argv[1], &n_names);
  napi_get_array_length(env, argv[2], &n_sinks);
  uint32_t n = n_names < n_sinks ? n_names : n_sinks;
  const char **names = calloc(n ? n : 1, sizeof(char *));
  PolyUOp **sinks = calloc(n ? n : 1, sizeof(PolyUOp *));
  if (!names || !sinks) {
    free(names);
    free(sinks);
    napi_throw_error(env, NULL, "calloc failed");
    return NULL;
  }

  for (uint32_t i = 0; i < n; i++) {
    napi_value name_val, sink_val;
    napi_get_element(env, argv[1], i, &name_val);
    napi_get_element(env, argv[2], i, &sink_val);
    names[i] = read_utf8_arg(env, name_val, NULL);
    sinks[i] = get_external(env, sink_val);
    if (!names[i]) {
      for (uint32_t j = 0; j < i; j++)
        free((void *)names[j]);
      free(names);
      free(sinks);
      return NULL;
    }
  }

  PolyModel *inst = poly_model_from_sinks(ctx, names, sinks, (int)n);
  for (uint32_t i = 0; i < n; i++)
    free((void *)names[i]);
  free(names);
  free(sinks);
  return make_external(env, inst);
}

static napi_value napi_poly_model_free(napi_env env, napi_callback_info info) {
  napi_value argv[1];
  size_t argc = 1;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyModel *inst = get_external(env, argv[0]);
  poly_model_free(inst);
  napi_value undef;
  napi_get_undefined(env, &undef);
  return undef;
}

static napi_value napi_poly_model_set_device(napi_env env, napi_callback_info info) {
  napi_value argv[2];
  size_t argc = 2;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyModel *inst = get_external(env, argv[0]);
  int32_t device;
  napi_get_value_int32(env, argv[1], &device);
  napi_value result;
  NAPI_CALL(env, napi_create_int32(env, poly_model_set_device(inst, device), &result));
  return result;
}

static napi_value napi_poly_model_define_module_arrays(napi_env env, napi_callback_info info) {
  napi_value argv[5];
  size_t argc = 5;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  if (argc < 5) {
    napi_throw_error(env, NULL, "polygrad: poly_model_define_module_arrays expects 5 args");
    return NULL;
  }
  PolyModel *inst = get_external(env, argv[0]);
  char **names = NULL;
  PolyTensor **inputs = NULL;
  int *input_counts = NULL;
  PolyTensor **outputs = NULL;
  uint32_t n_names = 0, n_inputs = 0, n_counts = 0, n_outputs = 0;
  if (!read_string_array_alloc(env, argv[1], &names, &n_names, false) ||
      !read_tensor_array_alloc(env, argv[2], &inputs, &n_inputs) ||
      !read_i32_array_alloc(env, argv[3], &input_counts, &n_counts) ||
      !read_tensor_array_alloc(env, argv[4], &outputs, &n_outputs))
    goto fail;
  if (n_names == 0 || n_names != n_counts || n_names != n_outputs) {
    napi_throw_error(env, NULL, "polygrad: module array length mismatch");
    goto fail;
  }
  uint64_t expected_inputs = 0;
  for (uint32_t i = 0; i < n_counts; i++) {
    if (input_counts[i] < 0) {
      napi_throw_error(env, NULL, "polygrad: negative module input count");
      goto fail;
    }
    expected_inputs += (uint32_t)input_counts[i];
  }
  if (expected_inputs != n_inputs) {
    napi_throw_error(env, NULL, "polygrad: flattened module input count mismatch");
    goto fail;
  }

  int rc = poly_model_define_module_arrays(
      inst, (const char **)names, inputs, input_counts, outputs, (int)n_names
  );
  free_string_array_items(names, n_names);
  free(inputs);
  free(input_counts);
  free(outputs);
  napi_value result;
  NAPI_CALL(env, napi_create_int32(env, rc, &result));
  return result;

fail:
  free_string_array_items(names, n_names);
  free(inputs);
  free(input_counts);
  free(outputs);
  return NULL;
}

static napi_value napi_poly_model_set_device_map_arrays(napi_env env, napi_callback_info info) {
  napi_value argv[3];
  size_t argc = 3;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  if (argc < 3) {
    napi_throw_error(env, NULL, "polygrad: poly_model_set_device_map_arrays expects 3 args");
    return NULL;
  }
  PolyModel *inst = get_external(env, argv[0]);
  char **modules = NULL;
  char **devices = NULL;
  uint32_t n_modules = 0, n_devices = 0;
  if (!read_string_array_alloc(env, argv[1], &modules, &n_modules, false) ||
      !read_string_array_alloc(env, argv[2], &devices, &n_devices, false))
    goto fail;
  if (n_modules == 0 || n_modules != n_devices) {
    napi_throw_error(env, NULL, "polygrad: device-map array length mismatch");
    goto fail;
  }
  int rc = poly_model_set_device_map_arrays(
      inst, (const char **)modules, (const char **)devices, (int)n_modules
  );
  free_string_array_items(modules, n_modules);
  free_string_array_items(devices, n_devices);
  napi_value result;
  NAPI_CALL(env, napi_create_int32(env, rc, &result));
  return result;

fail:
  free_string_array_items(modules, n_modules);
  free_string_array_items(devices, n_devices);
  return NULL;
}

static napi_value napi_poly_compose(napi_env env, napi_callback_info info, bool sequential) {
  napi_value argv[2];
  size_t argc = 2;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  if (argc != 2) {
    napi_throw_error(env, NULL, "model factory expects context and JSON");
    return NULL;
  }
  PolyCtx *ctx = get_external(env, argv[0]);
  size_t len = 0;
  char *json = read_utf8_arg(env, argv[1], &len);
  if (!json) return NULL;
  PolyModelError err = {0};
  PolyModel *model = len > 1048576 ? NULL : sequential
      ? poly_sequential_from_json(ctx, json, (int)len, &err)
      : poly_graph_from_json(ctx, json, (int)len, &err);
  free(json);
  if (!model) {
    napi_throw_error(env, NULL, len > 1048576 ? "definition exceeds 1048576 JSON bytes" : err.message);
    return NULL;
  }
  return make_external(env, model);
}

static napi_value napi_poly_sequential_from_json(napi_env env, napi_callback_info info) {
  return napi_poly_compose(env, info, true);
}

static napi_value napi_poly_graph_from_json(napi_env env, napi_callback_info info) {
  return napi_poly_compose(env, info, false);
}

static napi_value napi_poly_mlp_from_json(napi_env env, napi_callback_info info) {
  napi_value argv[2];
  size_t argc = 2;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  size_t spec_len = 0;
  char *spec = read_utf8_arg(env, argv[0], &spec_len);
  if (!spec) return NULL;
  int32_t device;
  NAPI_CALL(env, napi_get_value_int32(env, argv[1], &device));
  PolyModel *inst = poly_mlp_from_json(spec, (int)spec_len, (PolyDevice)device);
  free(spec);
  if (!inst) {
    napi_value result;
    napi_get_null(env, &result);
    return result;
  }
  return make_external(env, inst);
}

static napi_value napi_poly_tabm_from_json(napi_env env, napi_callback_info info) {
  napi_value argv[2];
  size_t argc = 2;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  size_t spec_len = 0;
  char *spec = read_utf8_arg(env, argv[0], &spec_len);
  if (!spec) return NULL;
  int32_t device;
  NAPI_CALL(env, napi_get_value_int32(env, argv[1], &device));
  PolyModel *inst = poly_tabm_from_json(spec, (int)spec_len, (PolyDevice)device);
  free(spec);
  if (!inst) {
    napi_value result;
    napi_get_null(env, &result);
    return result;
  }
  return make_external(env, inst);
}

static napi_value napi_poly_nam_from_json(napi_env env, napi_callback_info info) {
  napi_value argv[2];
  size_t argc = 2;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  size_t spec_len = 0;
  char *spec = read_utf8_arg(env, argv[0], &spec_len);
  if (!spec) return NULL;
  int32_t device;
  NAPI_CALL(env, napi_get_value_int32(env, argv[1], &device));
  PolyModel *inst = poly_nam_from_json(spec, (int)spec_len, (PolyDevice)device);
  free(spec);
  if (!inst) {
    napi_value result;
    napi_get_null(env, &result);
    return result;
  }
  return make_external(env, inst);
}

static napi_value napi_poly_model_param_count(napi_env env, napi_callback_info info) {
  napi_value argv[1];
  size_t argc = 1;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyModel *inst = get_external(env, argv[0]);
  napi_value result;
  NAPI_CALL(env, napi_create_int32(env, poly_model_param_count(inst), &result));
  return result;
}

static napi_value napi_poly_model_param_name(napi_env env, napi_callback_info info) {
  napi_value argv[2];
  size_t argc = 2;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyModel *inst = get_external(env, argv[0]);
  int32_t i;
  napi_get_value_int32(env, argv[1], &i);
  const char *name = poly_model_param_name(inst, i);
  napi_value result;
  if (name) {
    NAPI_CALL(env, napi_create_string_utf8(env, name, strlen(name), &result));
  } else {
    napi_get_null(env, &result);
  }
  return result;
}

static napi_value napi_poly_model_param_shape(napi_env env, napi_callback_info info) {
  napi_value argv[2];
  size_t argc = 2;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyModel *inst = get_external(env, argv[0]);
  int32_t i;
  napi_get_value_int32(env, argv[1], &i);
  int64_t shape[8];
  int ndim = poly_model_param_shape(inst, i, shape, 8);
  napi_value result;
  NAPI_CALL(env, napi_create_array_with_length(env, (size_t)(ndim > 0 ? ndim : 0), &result));
  for (int j = 0; j < ndim; j++) {
    napi_value v;
    NAPI_CALL(env, napi_create_int64(env, shape[j], &v));
    NAPI_CALL(env, napi_set_element(env, result, (uint32_t)j, v));
  }
  return result;
}

static napi_value napi_poly_model_param_data(napi_env env, napi_callback_info info) {
  napi_value argv[2];
  size_t argc = 2;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyModel *inst = get_external(env, argv[0]);
  int32_t i;
  napi_get_value_int32(env, argv[1], &i);
  const char *name = poly_model_param_name(inst, i);
  for (int j = 0; name && j < poly_model_buf_count(inst); j++)
    if (!strcmp(name, poly_model_buf_name(inst, j))) return read_model_storage_array(env, inst, j);
  return read_model_storage_array(env, inst, -1);
}

static napi_value napi_poly_model_param_dtype_id(napi_env env, napi_callback_info info) {
  napi_value argv[2], result;
  size_t argc = 2;
  int32_t i = -1;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  napi_get_value_int32(env, argv[1], &i);
  NAPI_CALL(
      env, napi_create_int32(env, poly_model_param_dtype_id(get_external(env, argv[0]), i), &result)
  );
  return result;
}

static napi_value napi_poly_model_param_trainable(napi_env env, napi_callback_info info) {
  napi_value argv[2];
  size_t argc = 2;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyModel *inst = get_external(env, argv[0]);
  int32_t i;
  napi_get_value_int32(env, argv[1], &i);
  napi_value result;
  NAPI_CALL(env, napi_get_boolean(env, poly_model_param_trainable(inst, i), &result));
  return result;
}

static napi_value napi_poly_model_set_param_trainable(napi_env env, napi_callback_info info) {
  napi_value argv[3];
  size_t argc = 3;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyModel *inst = get_external(env, argv[0]);
  int32_t i;
  bool trainable;
  napi_get_value_int32(env, argv[1], &i);
  napi_get_value_bool(env, argv[2], &trainable);
  napi_value result;
  NAPI_CALL(
      env, napi_create_int32(env, poly_model_set_param_trainable(inst, i, trainable), &result)
  );
  return result;
}

static napi_value napi_poly_model_buf_count(napi_env env, napi_callback_info info) {
  napi_value argv[1];
  size_t argc = 1;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyModel *inst = get_external(env, argv[0]);
  napi_value result;
  NAPI_CALL(env, napi_create_int32(env, poly_model_buf_count(inst), &result));
  return result;
}

static napi_value napi_poly_model_buf_name(napi_env env, napi_callback_info info) {
  napi_value argv[2];
  size_t argc = 2;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyModel *inst = get_external(env, argv[0]);
  int32_t i;
  napi_get_value_int32(env, argv[1], &i);
  const char *name = poly_model_buf_name(inst, i);
  napi_value result;
  if (name) {
    NAPI_CALL(env, napi_create_string_utf8(env, name, strlen(name), &result));
  } else {
    napi_get_null(env, &result);
  }
  return result;
}

static napi_value napi_poly_model_buf_role(napi_env env, napi_callback_info info) {
  napi_value argv[2];
  size_t argc = 2;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyModel *inst = get_external(env, argv[0]);
  int32_t i;
  napi_get_value_int32(env, argv[1], &i);
  napi_value result;
  NAPI_CALL(env, napi_create_int32(env, poly_model_buf_role(inst, i), &result));
  return result;
}

static napi_value napi_poly_model_buf_trainable(napi_env env, napi_callback_info info) {
  napi_value argv[2];
  size_t argc = 2;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyModel *inst = get_external(env, argv[0]);
  int32_t i;
  napi_get_value_int32(env, argv[1], &i);
  napi_value result;
  NAPI_CALL(env, napi_get_boolean(env, poly_model_buf_trainable(inst, i), &result));
  return result;
}

static napi_value napi_poly_model_set_buf_trainable(napi_env env, napi_callback_info info) {
  napi_value argv[3];
  size_t argc = 3;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyModel *inst = get_external(env, argv[0]);
  int32_t i;
  bool trainable;
  napi_get_value_int32(env, argv[1], &i);
  napi_get_value_bool(env, argv[2], &trainable);
  napi_value result;
  NAPI_CALL(env, napi_create_int32(env, poly_model_set_buf_trainable(inst, i, trainable), &result));
  return result;
}

static napi_value napi_poly_model_buf_shape(napi_env env, napi_callback_info info) {
  napi_value argv[2];
  size_t argc = 2;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyModel *inst = get_external(env, argv[0]);
  int32_t i;
  napi_get_value_int32(env, argv[1], &i);
  int64_t shape[8];
  int ndim = poly_model_buf_shape(inst, i, shape, 8);
  napi_value result;
  NAPI_CALL(env, napi_create_array_with_length(env, (size_t)(ndim > 0 ? ndim : 0), &result));
  for (int j = 0; j < ndim; j++) {
    napi_value v;
    NAPI_CALL(env, napi_create_int64(env, shape[j], &v));
    NAPI_CALL(env, napi_set_element(env, result, (uint32_t)j, v));
  }
  return result;
}

static napi_value napi_poly_model_buf_data(napi_env env, napi_callback_info info) {
  napi_value argv[2];
  size_t argc = 2;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyModel *inst = get_external(env, argv[0]);
  int32_t i;
  napi_get_value_int32(env, argv[1], &i);
  return read_model_storage_array(env, inst, i);
}

static napi_value napi_poly_model_write_buf(napi_env env, napi_callback_info info) {
  napi_value argv[3], arraybuf, result;
  size_t argc = 3, len = 0, offset = 0, itemsize = 0;
  void *data = NULL;
  int32_t index = -1;
  napi_typedarray_type actual, expected;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyModel *model = get_external(env, argv[0]);
  NAPI_CALL(env, napi_get_value_int32(env, argv[1], &index));
  NAPI_CALL(env, napi_get_typedarray_info(env, argv[2], &actual, &len, &data, &arraybuf, &offset));
  if (!instance_napi_storage_type(poly_model_buf_dtype_id(model, index), &expected, &itemsize) ||
      actual != expected || len > SIZE_MAX / itemsize ||
      len * itemsize != poly_model_buf_nbytes(model, index)) {
    napi_throw_type_error(
        env, NULL, "polygrad: buffer write requires exact storage dtype and extent"
    );
    return NULL;
  }
  int rc = poly_model_write_buf(model, index, data, len * itemsize);
  NAPI_CALL(env, napi_create_int32(env, rc, &result));
  return result;
}

static napi_value napi_poly_model_buf_dtype_id(napi_env env, napi_callback_info info) {
  napi_value argv[2], result;
  size_t argc = 2;
  int32_t i = -1;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  napi_get_value_int32(env, argv[1], &i);
  NAPI_CALL(
      env, napi_create_int32(env, poly_model_buf_dtype_id(get_external(env, argv[0]), i), &result)
  );
  return result;
}

static napi_value napi_poly_model_export_weights(napi_env env, napi_callback_info info) {
  napi_value argv[2];
  size_t argc = 2;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyModel *inst = get_external(env, argv[0]);
  uint32_t flags = POLY_EXPORT_WEIGHTS_DEFAULT;
  if (argc > 1) napi_get_value_uint32(env, argv[1], &flags);
  int out_len = 0;
  uint8_t *bytes = poly_model_export_weights_ex(inst, &out_len, flags);
  napi_value result = make_uint8_array_copy(env, bytes, (size_t)(out_len > 0 ? out_len : 0));
  free(bytes);
  return result;
}

static napi_value napi_poly_model_import_weights(napi_env env, napi_callback_info info) {
  napi_value argv[2];
  size_t argc = 2;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyModel *inst = get_external(env, argv[0]);
  napi_typedarray_type type;
  size_t len = 0;
  void *data = NULL;
  size_t offset;
  napi_value arraybuf;
  NAPI_CALL(env, napi_get_typedarray_info(env, argv[1], &type, &len, &data, &arraybuf, &offset));
  (void)type;
  napi_value result;
  NAPI_CALL(
      env, napi_create_int32(
               env, poly_model_import_weights(inst, (const uint8_t *)data, (int)len), &result
           )
  );
  return result;
}

static napi_value napi_poly_model_export_ir(napi_env env, napi_callback_info info) {
  napi_value argv[1];
  size_t argc = 1;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyModel *inst = get_external(env, argv[0]);
  int out_len = 0;
  uint8_t *bytes = poly_model_export_ir(inst, &out_len);
  napi_value result = make_uint8_array_copy(env, bytes, (size_t)(out_len > 0 ? out_len : 0));
  free(bytes);
  return result;
}

static napi_value napi_poly_model_export_program(napi_env env, napi_callback_info info) {
  napi_value argv[1];
  size_t argc = 1;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyModel *inst = get_external(env, argv[0]);
  int out_len = 0;
  uint8_t *bytes = poly_model_export_program(inst, &out_len);
  napi_value result = make_uint8_array_copy(env, bytes, (size_t)(out_len > 0 ? out_len : 0));
  free(bytes);
  return result;
}

static napi_value napi_poly_model_save_bundle(napi_env env, napi_callback_info info) {
  napi_value argv[2];
  size_t argc = 2;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyModel *inst = get_external(env, argv[0]);
  uint32_t flags = POLY_EXPORT_WEIGHTS_DEFAULT;
  if (argc > 1) napi_get_value_uint32(env, argv[1], &flags);
  int out_len = 0;
  uint8_t *bytes = poly_model_save_bundle_ex(inst, &out_len, flags);
  napi_value result = make_uint8_array_copy(env, bytes, (size_t)(out_len > 0 ? out_len : 0));
  free(bytes);
  return result;
}

static napi_value napi_poly_model_from_bundle(napi_env env, napi_callback_info info) {
  napi_value argv[1];
  size_t argc = 1;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  uint8_t *data = NULL;
  size_t len = 0;
  napi_typedarray_type type;
  napi_value arraybuf;
  size_t offset;
  NAPI_CALL(
      env, napi_get_typedarray_info(env, argv[0], &type, &len, (void **)&data, &arraybuf, &offset)
  );
  PolyModel *inst = poly_model_from_bundle(data, (int)len);
  if (!inst) {
    napi_value undef;
    napi_get_undefined(env, &undef);
    return undef;
  }
  return make_external(env, inst);
}

static napi_value napi_poly_model_set_optimizer(napi_env env, napi_callback_info info) {
  napi_value argv[10];
  size_t argc = 10;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyModel *inst = get_external(env, argv[0]);
  int32_t kind;
  double lr, beta1, beta2, eps, weight_decay, momentum = 0.0;
  bool nesterov = false, classic = false;
  napi_get_value_int32(env, argv[1], &kind);
  napi_get_value_double(env, argv[2], &lr);
  napi_get_value_double(env, argv[3], &beta1);
  napi_get_value_double(env, argv[4], &beta2);
  napi_get_value_double(env, argv[5], &eps);
  napi_get_value_double(env, argv[6], &weight_decay);
  if (argc > 7) napi_get_value_double(env, argv[7], &momentum);
  if (argc > 8) napi_get_value_bool(env, argv[8], &nesterov);
  if (argc > 9) napi_get_value_bool(env, argv[9], &classic);
  napi_value result;
  NAPI_CALL(
      env, napi_create_int32(
               env,
               poly_model_set_optimizer_ex(
                   inst, kind, (float)lr, (float)beta1, (float)beta2, (float)eps,
                   (float)weight_decay, (float)momentum, nesterov, classic
               ),
               &result
           )
  );
  return result;
}

static napi_value napi_poly_model_forward(napi_env env, napi_callback_info info) {
  napi_value argv[3];
  size_t argc = 3;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyModel *inst = get_external(env, argv[0]);

  PolyIOBinding *bindings = NULL;
  char **names = NULL;
  int n = 0;
  if (!read_io_bindings(env, argv[1], argv[2], &bindings, &names, &n)) {
    return NULL;
  }

  int rc = poly_model_forward(inst, bindings, n);
  free_io_bindings(names, bindings, n);

  napi_value result;
  NAPI_CALL(env, napi_create_int32(env, rc, &result));
  return result;
}

static napi_value napi_poly_model_call(napi_env env, napi_callback_info info) {
  napi_value argv[4];
  size_t argc = 4;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyModel *inst = get_external(env, argv[0]);
  char *entrypoint = read_utf8_arg(env, argv[1], NULL);
  if (!entrypoint) return NULL;

  PolyIOBinding *bindings = NULL;
  char **names = NULL;
  int n = 0;
  if (!read_io_bindings(env, argv[2], argv[3], &bindings, &names, &n)) {
    free(entrypoint);
    return NULL;
  }

  int rc = poly_model_call(inst, entrypoint, bindings, n);
  free_io_bindings(names, bindings, n);
  free(entrypoint);

  napi_value result;
  NAPI_CALL(env, napi_create_int32(env, rc, &result));
  return result;
}

static napi_value napi_poly_model_entrypoints(napi_env env, napi_callback_info info) {
  napi_value argv[1], result;
  size_t argc = 1;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyModel *model = get_external(env, argv[0]);
  int count = poly_model_entrypoint_count(model);
  NAPI_CALL(env, napi_create_array_with_length(env, (size_t)count, &result));
  for (int i = 0; i < count; i++) {
    napi_value row, value;
    const char *name = poly_model_entrypoint_name(model, i);
    const char *objective = poly_model_entrypoint_objective(model, name);
    NAPI_CALL(env, napi_create_object(env, &row));
    NAPI_CALL(env, napi_create_string_utf8(env, name, NAPI_AUTO_LENGTH, &value));
    NAPI_CALL(env, napi_set_named_property(env, row, "name", value));
    if (objective) {
      NAPI_CALL(env, napi_create_string_utf8(env, objective, NAPI_AUTO_LENGTH, &value));
    } else {
      NAPI_CALL(env, napi_get_null(env, &value));
    }
    NAPI_CALL(env, napi_set_named_property(env, row, "objective", value));
    for (int outputs = 0; outputs < 2; outputs++) {
      napi_value names;
      int n = outputs ? poly_model_entrypoint_output_count(model, name)
                      : poly_model_entrypoint_input_count(model, name);
      NAPI_CALL(env, napi_create_array_with_length(env, (size_t)n, &names));
      for (int j = 0; j < n; j++) {
        const char *binding = outputs ? poly_model_entrypoint_output_name(model, name, j)
                                      : poly_model_entrypoint_input_name(model, name, j);
        NAPI_CALL(env, napi_create_string_utf8(env, binding, NAPI_AUTO_LENGTH, &value));
        NAPI_CALL(env, napi_set_element(env, names, (uint32_t)j, value));
      }
      NAPI_CALL(env, napi_set_named_property(env, row, outputs ? "outputs" : "inputs", names));
    }
    NAPI_CALL(env, napi_set_element(env, result, (uint32_t)i, row));
  }
  return result;
}

static napi_value napi_poly_model_entrypoint_output_count(napi_env env, napi_callback_info info) {
  napi_value argv[2];
  size_t argc = 2;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyModel *inst = get_external(env, argv[0]);
  char *entrypoint = read_utf8_arg(env, argv[1], NULL);
  if (!entrypoint) return NULL;
  int count = poly_model_entrypoint_output_count(inst, entrypoint);
  free(entrypoint);
  napi_value result;
  NAPI_CALL(env, napi_create_int32(env, count, &result));
  return result;
}

static napi_value napi_poly_model_entrypoint_output_name(napi_env env, napi_callback_info info) {
  napi_value argv[3];
  size_t argc = 3;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyModel *inst = get_external(env, argv[0]);
  char *entrypoint = read_utf8_arg(env, argv[1], NULL);
  int32_t output_index = -1;
  if (!entrypoint) return NULL;
  napi_get_value_int32(env, argv[2], &output_index);
  const char *name = poly_model_entrypoint_output_name(inst, entrypoint, output_index);
  free(entrypoint);
  if (!name) {
    napi_value result;
    napi_get_null(env, &result);
    return result;
  }
  napi_value result;
  NAPI_CALL(env, napi_create_string_utf8(env, name, NAPI_AUTO_LENGTH, &result));
  return result;
}

static napi_value napi_poly_model_train_step(napi_env env, napi_callback_info info) {
  napi_value argv[4];
  size_t argc = 4;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyModel *inst = get_external(env, argv[0]);

  PolyIOBinding *bindings = NULL;
  char **names = NULL;
  int n = 0;
  if (!read_io_bindings(env, argv[1], argv[2], &bindings, &names, &n)) {
    return NULL;
  }

  float loss = 0.0f;
  char *entrypoint =
      argc > 3 && !napi_is_nullish(env, argv[3]) ? read_utf8_arg(env, argv[3], NULL) : NULL;
  int rc = poly_model_train_step(inst, entrypoint, bindings, n, &loss);
  free(entrypoint);
  free_io_bindings(names, bindings, n);
  if (rc != 0) {
    napi_value result;
    napi_get_null(env, &result);
    return result;
  }

  napi_value result;
  NAPI_CALL(env, napi_create_double(env, (double)loss, &result));
  return result;
}

/* ── Shape-on-UOp accessors ────────────────────────────────────────────── */

static napi_value napi_poly_uop_ndim(napi_env env, napi_callback_info info) {
  napi_value argv[2];
  size_t argc = 2;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *u = get_external(env, argv[1]);
  napi_value result;
  napi_create_int32(env, poly_uop_ndim(ctx, u), &result);
  return result;
}

static napi_value napi_poly_uop_max_shape_dims(napi_env env, napi_callback_info info) {
  napi_value argv[2];
  size_t argc = 2;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *u = get_external(env, argv[1]);
  int ndim = poly_uop_ndim(ctx, u);
  const int64_t *dims = poly_uop_max_shape_dims(ctx, u);
  napi_value arr;
  napi_create_array_with_length(env, ndim > 0 ? ndim : 0, &arr);
  for (int i = 0; i < ndim; i++) {
    napi_value v;
    napi_create_int64(env, dims[i], &v);
    napi_set_element(env, arr, i, v);
  }
  return arr;
}

/* ── v2 composed ops (shape read from UOp) ─────────────────────────────── */

static napi_value napi_poly_softmax(napi_env env, napi_callback_info info) {
  napi_value argv[3];
  size_t argc = 3;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *x = get_external(env, argv[1]);
  int32_t axis;
  napi_get_value_int32(env, argv[2], &axis);
  return make_external(env, poly_softmax(ctx, x, axis));
}

static napi_value napi_poly_log_softmax(napi_env env, napi_callback_info info) {
  napi_value argv[3];
  size_t argc = 3;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *x = get_external(env, argv[1]);
  int32_t axis;
  napi_get_value_int32(env, argv[2], &axis);
  return make_external(env, poly_log_softmax(ctx, x, axis));
}

static napi_value napi_poly_dot(napi_env env, napi_callback_info info) {
  napi_value argv[3];
  size_t argc = 3;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *x = get_external(env, argv[1]);
  PolyUOp *w = get_external(env, argv[2]);
  return make_external(env, poly_dot(ctx, x, w));
}

static napi_value napi_poly_qr(napi_env env, napi_callback_info info) {
  napi_value argv[2];
  size_t argc = 2;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *x = get_external(env, argv[1]);
  PolyUOp *q = NULL, *r = NULL;
  if (poly_qr(ctx, x, &q, &r) != 0) {
    napi_throw_error(env, NULL, "polygrad: poly_qr failed");
    return NULL;
  }
  return make_external_pair(env, q, r);
}

static napi_value napi_poly_qr_ex(napi_env env, napi_callback_info info) {
  napi_value argv[3];
  size_t argc = 3;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *x = get_external(env, argv[1]);
  int32_t mode = 0;
  napi_get_value_int32(env, argv[2], &mode);
  PolyUOp *q = NULL, *r = NULL;
  if (poly_qr_ex(ctx, x, mode, &q, &r) != 0) {
    napi_throw_error(env, NULL, "polygrad: poly_qr_ex failed");
    return NULL;
  }
  return make_external_pair(env, q, r);
}

static napi_value napi_poly_cholesky(napi_env env, napi_callback_info info) {
  napi_value argv[3];
  size_t argc = 3;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *x = get_external(env, argv[1]);
  int32_t upper = 0;
  napi_get_value_int32(env, argv[2], &upper);
  return make_external(env, poly_cholesky(ctx, x, upper));
}

static napi_value napi_poly_cholesky_solve(napi_env env, napi_callback_info info) {
  napi_value argv[4];
  size_t argc = 4;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *chol = get_external(env, argv[1]);
  PolyUOp *b = get_external(env, argv[2]);
  int32_t upper = 0;
  napi_get_value_int32(env, argv[3], &upper);
  return make_external(env, poly_cholesky_solve(ctx, chol, b, upper));
}

static napi_value napi_poly_triangular_solve(napi_env env, napi_callback_info info) {
  napi_value argv[6];
  size_t argc = 6;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *a = get_external(env, argv[1]);
  PolyUOp *b = get_external(env, argv[2]);
  int32_t upper = 0, transpose_a = 0, unit_diagonal = 0;
  napi_get_value_int32(env, argv[3], &upper);
  napi_get_value_int32(env, argv[4], &transpose_a);
  napi_get_value_int32(env, argv[5], &unit_diagonal);
  return make_external(env, poly_triangular_solve(ctx, a, b, upper, transpose_a, unit_diagonal));
}

static napi_value napi_poly_solve(napi_env env, napi_callback_info info) {
  napi_value argv[3];
  size_t argc = 3;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *a = get_external(env, argv[1]);
  PolyUOp *b = get_external(env, argv[2]);
  return make_external(env, poly_solve(ctx, a, b));
}

static napi_value napi_poly_lstsq(napi_env env, napi_callback_info info) {
  napi_value argv[3];
  size_t argc = 3;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *a = get_external(env, argv[1]);
  PolyUOp *b = get_external(env, argv[2]);
  return make_external(env, poly_lstsq(ctx, a, b));
}

static napi_value napi_poly_cross_entropy(napi_env env, napi_callback_info info) {
  napi_value argv[4];
  size_t argc = 4;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *logits = get_external(env, argv[1]);
  PolyUOp *target = get_external(env, argv[2]);
  int32_t axis;
  napi_get_value_int32(env, argv[3], &axis);
  return make_external(env, poly_cross_entropy(ctx, logits, target, axis));
}

static napi_value napi_poly_gather(napi_env env, napi_callback_info info) {
  napi_value argv[3];
  size_t argc = 3;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *table = get_external(env, argv[1]);
  PolyUOp *indices = get_external(env, argv[2]);
  return make_external(env, poly_gather(ctx, table, indices));
}

static napi_value napi_poly_gather_dim(napi_env env, napi_callback_info info) {
  napi_value argv[4];
  size_t argc = 4;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *x = get_external(env, argv[1]);
  int32_t dim;
  napi_get_value_int32(env, argv[2], &dim);
  PolyUOp *index = get_external(env, argv[3]);
  return make_external(env, poly_gather_dim(ctx, x, dim, index));
}

static napi_value napi_poly_scatter(napi_env env, napi_callback_info info) {
  napi_value argv[6];
  size_t argc = 6;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *self = get_external(env, argv[1]);
  int32_t dim;
  napi_get_value_int32(env, argv[2], &dim);
  PolyUOp *index = get_external(env, argv[3]);
  PolyUOp *src = get_external(env, argv[4]);
  char *reduce = read_utf8_arg(env, argv[5], NULL);
  PolyUOp *out = poly_scatter(ctx, self, dim, index, src, reduce);
  free(reduce);
  return make_external(env, out);
}

static napi_value napi_poly_scatter_reduce(napi_env env, napi_callback_info info) {
  napi_value argv[7];
  size_t argc = 7;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *self = get_external(env, argv[1]);
  int32_t dim, include_self;
  napi_get_value_int32(env, argv[2], &dim);
  PolyUOp *index = get_external(env, argv[3]);
  PolyUOp *src = get_external(env, argv[4]);
  char *reduce = read_utf8_arg(env, argv[5], NULL);
  napi_get_value_int32(env, argv[6], &include_self);
  PolyUOp *out = poly_scatter_reduce(ctx, self, dim, index, src, reduce, include_self);
  free(reduce);
  return make_external(env, out);
}

static napi_value napi_poly_tensor_scatter(napi_env env, napi_callback_info info) {
  napi_value argv[6];
  size_t argc = 6;
  int32_t dim = 0;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  napi_get_value_int32(env, argv[2], &dim);
  char *reduce = read_utf8_arg(env, argv[5], NULL);
  PolyTensor *out = poly_tensor_scatter(
      get_external(env, argv[0]), get_external(env, argv[1]), dim, get_external(env, argv[3]),
      get_external(env, argv[4]), reduce
  );
  free(reduce);
  return make_external(env, out);
}

static napi_value napi_poly_tensor_scatter_reduce(napi_env env, napi_callback_info info) {
  napi_value argv[7];
  size_t argc = 7;
  int32_t dim = 0, include_self = 0;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  napi_get_value_int32(env, argv[2], &dim);
  char *reduce = read_utf8_arg(env, argv[5], NULL);
  napi_get_value_int32(env, argv[6], &include_self);
  PolyTensor *out = poly_tensor_scatter_reduce(
      get_external(env, argv[0]), get_external(env, argv[1]), dim, get_external(env, argv[3]),
      get_external(env, argv[4]), reduce, include_self
  );
  free(reduce);
  return make_external(env, out);
}

static napi_value napi_poly_argmax(napi_env env, napi_callback_info info) {
  napi_value argv[4];
  size_t argc = 4;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *x = get_external(env, argv[1]);
  int32_t axis, keepdim;
  napi_get_value_int32(env, argv[2], &axis);
  napi_get_value_int32(env, argv[3], &keepdim);
  return make_external(env, poly_argmax(ctx, x, axis, keepdim));
}

static napi_value napi_poly_sort(napi_env env, napi_callback_info info) {
  napi_value argv[4];
  size_t argc = 4;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *x = get_external(env, argv[1]);
  int32_t dim, descending;
  napi_get_value_int32(env, argv[2], &dim);
  napi_get_value_int32(env, argv[3], &descending);
  PolyUOp *values = NULL, *indices = NULL;
  if (poly_sort(ctx, x, dim, descending, &values, &indices) != 0) {
    napi_throw_error(env, NULL, "polygrad: poly_sort failed");
    return NULL;
  }
  return make_external_pair(env, values, indices);
}

static napi_value napi_poly_argsort(napi_env env, napi_callback_info info) {
  napi_value argv[4];
  size_t argc = 4;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *x = get_external(env, argv[1]);
  int32_t dim, descending;
  napi_get_value_int32(env, argv[2], &dim);
  napi_get_value_int32(env, argv[3], &descending);
  return make_external(env, poly_argsort(ctx, x, dim, descending));
}

static napi_value napi_poly_topk(napi_env env, napi_callback_info info) {
  napi_value argv[6];
  size_t argc = 6;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *x = get_external(env, argv[1]);
  int64_t k;
  int32_t dim, largest, sorted;
  napi_get_value_int64(env, argv[2], &k);
  napi_get_value_int32(env, argv[3], &dim);
  napi_get_value_int32(env, argv[4], &largest);
  napi_get_value_int32(env, argv[5], &sorted);
  PolyUOp *values = NULL, *indices = NULL;
  if (poly_topk(ctx, x, k, dim, largest, sorted, &values, &indices) != 0) {
    napi_throw_error(env, NULL, "polygrad: poly_topk failed");
    return NULL;
  }
  return make_external_pair(env, values, indices);
}

static napi_value napi_poly_sum_reduce(napi_env env, napi_callback_info info) {
  napi_value argv[4];
  size_t argc = 4;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *x = get_external(env, argv[1]);
  int32_t axis, keepdim;
  napi_get_value_int32(env, argv[2], &axis);
  napi_get_value_int32(env, argv[3], &keepdim);
  return make_external(env, poly_sum_reduce(ctx, x, axis, keepdim));
}

static napi_value napi_poly_max_reduce(napi_env env, napi_callback_info info) {
  napi_value argv[4];
  size_t argc = 4;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *x = get_external(env, argv[1]);
  int32_t axis, keepdim;
  napi_get_value_int32(env, argv[2], &axis);
  napi_get_value_int32(env, argv[3], &keepdim);
  return make_external(env, poly_max_reduce(ctx, x, axis, keepdim));
}

static napi_value napi_poly_mean_reduce(napi_env env, napi_callback_info info) {
  napi_value argv[4];
  size_t argc = 4;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *x = get_external(env, argv[1]);
  int32_t axis, keepdim;
  napi_get_value_int32(env, argv[2], &axis);
  napi_get_value_int32(env, argv[3], &keepdim);
  return make_external(env, poly_mean_reduce(ctx, x, axis, keepdim));
}

static napi_value napi_poly_var_reduce(napi_env env, napi_callback_info info) {
  napi_value argv[5];
  size_t argc = 5;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *x = get_external(env, argv[1]);
  int32_t axis, keepdim, correction;
  napi_get_value_int32(env, argv[2], &axis);
  napi_get_value_int32(env, argv[3], &keepdim);
  napi_get_value_int32(env, argv[4], &correction);
  return make_external(env, poly_var_reduce(ctx, x, axis, keepdim, correction));
}

static napi_value napi_poly_one_hot(napi_env env, napi_callback_info info) {
  napi_value argv[3];
  size_t argc = 3;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *x = get_external(env, argv[1]);
  int64_t num_classes;
  napi_get_value_int64(env, argv[2], &num_classes);
  return make_external(env, poly_one_hot(ctx, x, num_classes));
}

static napi_value napi_poly_index_select(napi_env env, napi_callback_info info) {
  napi_value argv[4];
  size_t argc = 4;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *x = get_external(env, argv[1]);
  int32_t dim;
  napi_get_value_int32(env, argv[2], &dim);
  PolyUOp *index = get_external(env, argv[3]);
  return make_external(env, poly_index_select(ctx, x, dim, index));
}

static napi_value napi_poly_einsum(napi_env env, napi_callback_info info) {
  napi_value argv[3];
  size_t argc = 3;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  char *formula = read_utf8_arg(env, argv[1], NULL);
  if (!formula) return NULL;

  uint32_t n = 0;
  if (napi_get_array_length(env, argv[2], &n) != napi_ok) {
    free(formula);
    napi_throw_type_error(env, NULL, "polygrad: einsum operands must be an array");
    return NULL;
  }
  PolyUOp **operands = calloc(n ? n : 1, sizeof(PolyUOp *));
  if (!operands) {
    free(formula);
    napi_throw_error(env, NULL, "calloc failed");
    return NULL;
  }
  for (uint32_t i = 0; i < n; i++) {
    napi_value operand, raw;
    napi_valuetype type;
    bool has_uop = false;
    if (napi_get_element(env, argv[2], i, &operand) != napi_ok ||
        napi_typeof(env, operand, &type) != napi_ok) {
      free(operands);
      free(formula);
      napi_throw_type_error(env, NULL, "polygrad: invalid einsum operand");
      return NULL;
    }
    raw = operand;
    if (type != napi_external) {
      if (type != napi_object ||
          napi_has_named_property(env, operand, "_uop", &has_uop) != napi_ok || !has_uop ||
          napi_get_named_property(env, operand, "_uop", &raw) != napi_ok ||
          napi_typeof(env, raw, &type) != napi_ok || type != napi_external) {
        free(operands);
        free(formula);
        napi_throw_type_error(env, NULL, "polygrad: einsum operand must contain a UOp");
        return NULL;
      }
    }
    operands[i] = get_external(env, raw);
  }

  PolyUOp *result = poly_einsum(ctx, formula, operands, (int)n);
  free(operands);
  free(formula);
  int64_t out_shape[MAX_DIMS];
  int out_ndim = result ? poly_uop_ndim(ctx, result) : 0;
  const int64_t *dims = result ? poly_uop_max_shape_dims(ctx, result) : NULL;
  if (dims && out_ndim > 0) memcpy(out_shape, dims, (size_t)out_ndim * sizeof(int64_t));
  return make_shape_result(env, result, out_shape, out_ndim);
}

static napi_value napi_poly_tensor_einsum(napi_env env, napi_callback_info info) {
  napi_value argv[3];
  size_t argc = 3;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  char *formula = read_utf8_arg(env, argv[1], NULL);
  if (!formula) return NULL;

  uint32_t n = 0;
  if (napi_get_array_length(env, argv[2], &n) != napi_ok) {
    free(formula);
    napi_throw_type_error(env, NULL, "polygrad: Tensor.einsum operands must be an array");
    return NULL;
  }
  PolyTensor **operands = calloc(n ? n : 1, sizeof(*operands));
  if (!operands) {
    free(formula);
    napi_throw_error(env, NULL, "calloc failed");
    return NULL;
  }
  for (uint32_t i = 0; i < n; i++) {
    napi_value operand;
    napi_valuetype type;
    if (napi_get_element(env, argv[2], i, &operand) != napi_ok ||
        napi_typeof(env, operand, &type) != napi_ok || type != napi_external) {
      free(operands);
      free(formula);
      napi_throw_type_error(env, NULL, "polygrad: Tensor.einsum operand must be a Tensor handle");
      return NULL;
    }
    operands[i] = get_external(env, operand);
  }

  PolyTensor *result = poly_tensor_einsum(ctx, formula, operands, (int)n);
  free(operands);
  free(formula);
  return make_external(env, result);
}

static napi_value napi_rearrange_impl(napi_env env, napi_callback_info info, bool tensor_result) {
  napi_value argv[5];
  size_t argc = 5;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  char *formula = read_utf8_arg(env, argv[1], NULL);
  if (!formula) return NULL;
  void *input = get_external(env, argv[2]);

  napi_value keys;
  uint32_t n = 0;
  napi_value kwargs = argv[tensor_result ? 3 : 4];
  if (napi_get_property_names(env, kwargs, &keys) != napi_ok ||
      napi_get_array_length(env, keys, &n) != napi_ok) {
    free(formula);
    napi_throw_type_error(env, NULL, "polygrad: rearrange kwargs must be an object");
    return NULL;
  }
  char **names = calloc(n ? n : 1, sizeof(char *));
  int64_t *values = calloc(n ? n : 1, sizeof(int64_t));
  if (!names || !values) {
    free(names);
    free(values);
    free(formula);
    napi_throw_error(env, NULL, "calloc failed");
    return NULL;
  }

  size_t names_len = 0;
  for (uint32_t i = 0; i < n; i++) {
    napi_value key, value;
    if (napi_get_element(env, keys, i, &key) != napi_ok ||
        napi_get_property(env, kwargs, key, &value) != napi_ok ||
        napi_get_value_int64(env, value, &values[i]) != napi_ok) {
      for (uint32_t j = 0; j < i; j++)
        free(names[j]);
      free(names);
      free(values);
      free(formula);
      napi_throw_type_error(env, NULL, "polygrad: rearrange axis sizes must be integers");
      return NULL;
    }
    names[i] = read_utf8_arg(env, key, NULL);
    if (!names[i]) {
      for (uint32_t j = 0; j < i; j++)
        free(names[j]);
      free(names);
      free(values);
      free(formula);
      return NULL;
    }
    names_len += strlen(names[i]) + (i ? 1 : 0);
  }

  char *axis_names = NULL;
  if (n > 0) {
    axis_names = malloc(names_len + 1);
    if (!axis_names) {
      for (uint32_t i = 0; i < n; i++)
        free(names[i]);
      free(names);
      free(values);
      free(formula);
      napi_throw_error(env, NULL, "malloc failed");
      return NULL;
    }
    size_t offset = 0;
    for (uint32_t i = 0; i < n; i++) {
      if (i) axis_names[offset++] = ' ';
      size_t len = strlen(names[i]);
      memcpy(axis_names + offset, names[i], len);
      offset += len;
    }
    axis_names[offset] = '\0';
  }

  PolyTensor *tensor_out =
      tensor_result ? poly_tensor_rearrange(ctx, formula, input, axis_names, values, (int)n) : NULL;
  PolyUOp *uop_out =
      tensor_result ? NULL : poly_rearrange(ctx, formula, input, axis_names, values, (int)n);
  for (uint32_t i = 0; i < n; i++)
    free(names[i]);
  free(names);
  free(values);
  free(axis_names);
  free(formula);
  if (tensor_result) return make_external(env, tensor_out);
  int64_t out_shape[MAX_DIMS];
  int out_ndim = uop_out ? poly_uop_ndim(ctx, uop_out) : 0;
  const int64_t *dims = uop_out ? poly_uop_max_shape_dims(ctx, uop_out) : NULL;
  if (dims && out_ndim > 0) memcpy(out_shape, dims, (size_t)out_ndim * sizeof(int64_t));
  return make_shape_result(env, uop_out, out_shape, out_ndim);
}

static napi_value napi_poly_rearrange(napi_env env, napi_callback_info info) {
  return napi_rearrange_impl(env, info, false);
}

static napi_value napi_poly_tensor_rearrange(napi_env env, napi_callback_info info) {
  return napi_rearrange_impl(env, info, true);
}

/* ── Module registration ───────────────────────────────────────────────── */

#define DECLARE_NAPI_METHOD(name, fn)                                                              \
  { (name), NULL, (fn), NULL, NULL, NULL, napi_default, NULL }

/* ── Tokenizer N-API wrappers ──────────────────────────────────────────── */

static napi_value napi_poly_tokenizer_from_json(napi_env env, napi_callback_info info) {
  napi_value argv[2];
  size_t argc = 2;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  void *data;
  size_t len;
  NAPI_CALL(env, napi_get_buffer_info(env, argv[0], &data, &len));
  PolyTokenizer *tok = poly_tokenizer_from_json((const char *)data, (int)len);
  if (!tok) {
    napi_value n;
    napi_get_null(env, &n);
    return n;
  }
  return make_external(env, tok);
}

static napi_value napi_poly_tokenize(napi_env env, napi_callback_info info) {
  napi_value argv[2];
  size_t argc = 2;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyTokenizer *tok = get_external(env, argv[0]);
  size_t text_len;
  NAPI_CALL(env, napi_get_value_string_utf8(env, argv[1], NULL, 0, &text_len));
  char *text = malloc(text_len + 1);
  NAPI_CALL(env, napi_get_value_string_utf8(env, argv[1], text, text_len + 1, &text_len));
  int ids[4096];
  int n = poly_tokenize(tok, text, ids, 4096);
  free(text);
  napi_value result;
  NAPI_CALL(env, napi_create_array_with_length(env, (size_t)n, &result));
  for (int i = 0; i < n; i++) {
    napi_value v;
    NAPI_CALL(env, napi_create_int32(env, ids[i], &v));
    NAPI_CALL(env, napi_set_element(env, result, (uint32_t)i, v));
  }
  return result;
}

static napi_value napi_poly_detokenize(napi_env env, napi_callback_info info) {
  napi_value argv[2];
  size_t argc = 2;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyTokenizer *tok = get_external(env, argv[0]);
  uint32_t n;
  NAPI_CALL(env, napi_get_array_length(env, argv[1], &n));
  int *ids = malloc(n * sizeof(int));
  for (uint32_t i = 0; i < n; i++) {
    napi_value el;
    NAPI_CALL(env, napi_get_element(env, argv[1], i, &el));
    NAPI_CALL(env, napi_get_value_int32(env, el, &ids[i]));
  }
  char buf[8192];
  poly_detokenize(tok, ids, (int)n, buf, sizeof(buf));
  free(ids);
  napi_value result;
  NAPI_CALL(env, napi_create_string_utf8(env, buf, NAPI_AUTO_LENGTH, &result));
  return result;
}

static napi_value napi_poly_tokenizer_free(napi_env env, napi_callback_info info) {
  napi_value argv[1];
  size_t argc = 1;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  poly_tokenizer_free(get_external(env, argv[0]));
  napi_value undef;
  napi_get_undefined(env, &undef);
  return undef;
}

static napi_value napi_poly_tokenizer_vocab_size(napi_env env, napi_callback_info info) {
  napi_value argv[1];
  size_t argc = 1;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  napi_value r;
  NAPI_CALL(env, napi_create_int32(env, poly_tokenizer_vocab_size(get_external(env, argv[0])), &r));
  return r;
}

static napi_value napi_poly_tokenizer_bos_id(napi_env env, napi_callback_info info) {
  napi_value argv[1];
  size_t argc = 1;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  napi_value r;
  NAPI_CALL(env, napi_create_int32(env, poly_tokenizer_bos_id(get_external(env, argv[0])), &r));
  return r;
}

static napi_value napi_poly_tokenizer_eos_id(napi_env env, napi_callback_info info) {
  napi_value argv[1];
  size_t argc = 1;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  napi_value r;
  NAPI_CALL(env, napi_create_int32(env, poly_tokenizer_eos_id(get_external(env, argv[0])), &r));
  return r;
}

/* ── HF / GGUF loaders ─────────────────────────────────────────────────── */

extern PolyModel *poly_hf_load(
    const char *config_json,
    int config_len,
    const uint8_t **weight_files,
    const int64_t *weight_lens,
    int n_weight_files,
    int max_batch,
    int max_seq_len,
    PolyDevice device
);

extern PolyModel *poly_gguf_load(
    const uint8_t *data,
    int64_t len,
    int max_batch,
    int max_seq_len,
    PolyDevice device
);

static napi_value napi_poly_hf_load(napi_env env, napi_callback_info info) {
  napi_value argv[5];
  size_t argc = 5;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));

  /* argv[0] = config Buffer, argv[1] = weight Buffers array, argv[2] = maxBatch, argv[3] =
   * maxSeqLen */
  void *cfg_data;
  size_t cfg_len;
  NAPI_CALL(env, napi_get_buffer_info(env, argv[0], &cfg_data, &cfg_len));

  uint32_t n_files;
  NAPI_CALL(env, napi_get_array_length(env, argv[1], &n_files));

  const uint8_t **file_ptrs = malloc(n_files * sizeof(uint8_t *));
  int64_t *file_lens = malloc(n_files * sizeof(int64_t));
  for (uint32_t i = 0; i < n_files; i++) {
    napi_value el;
    NAPI_CALL(env, napi_get_element(env, argv[1], i, &el));
    void *fdata;
    size_t flen;
    NAPI_CALL(env, napi_get_buffer_info(env, el, &fdata, &flen));
    file_ptrs[i] = (const uint8_t *)fdata;
    file_lens[i] = (int64_t)flen;
  }

  int32_t max_batch, max_seq_len, device;
  NAPI_CALL(env, napi_get_value_int32(env, argv[2], &max_batch));
  NAPI_CALL(env, napi_get_value_int32(env, argv[3], &max_seq_len));
  NAPI_CALL(env, napi_get_value_int32(env, argv[4], &device));

  PolyModel *inst = poly_hf_load(
      (const char *)cfg_data, (int)cfg_len, file_ptrs, file_lens, (int)n_files, max_batch,
      max_seq_len, (PolyDevice)device
  );

  free(file_ptrs);
  free(file_lens);

  if (!inst) {
    napi_value n;
    napi_get_null(env, &n);
    return n;
  }
  return make_external(env, inst);
}

static napi_value napi_poly_gguf_load(napi_env env, napi_callback_info info) {
  napi_value argv[4];
  size_t argc = 4;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  void *data;
  size_t len;
  NAPI_CALL(env, napi_get_buffer_info(env, argv[0], &data, &len));
  int32_t max_batch, max_seq_len, device;
  NAPI_CALL(env, napi_get_value_int32(env, argv[1], &max_batch));
  NAPI_CALL(env, napi_get_value_int32(env, argv[2], &max_seq_len));
  NAPI_CALL(env, napi_get_value_int32(env, argv[3], &device));
  PolyModel *inst = poly_gguf_load(
      (const uint8_t *)data, (int64_t)len, max_batch, max_seq_len, (PolyDevice)device
  );
  if (!inst) {
    napi_value n;
    napi_get_null(env, &n);
    return n;
  }
  return make_external(env, inst);
}

static napi_value napi_poly_tokenizer_from_gguf(napi_env env, napi_callback_info info) {
  napi_value argv[1];
  size_t argc = 1;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  void *data;
  size_t len;
  NAPI_CALL(env, napi_get_buffer_info(env, argv[0], &data, &len));
  PolyGgufDecoded *gguf = NULL;
  if (poly_gguf_decode((const uint8_t *)data, (int64_t)len, &gguf) != 0 || !gguf) {
    napi_value n;
    napi_get_null(env, &n);
    return n;
  }
  PolyTokenizer *tok = poly_tokenizer_from_gguf(gguf);
  poly_gguf_decoded_free(gguf);
  if (!tok) {
    napi_value n;
    napi_get_null(env, &n);
    return n;
  }
  return make_external(env, tok);
}

static napi_value napi_poly_import_error_code(napi_env env, napi_callback_info info) {
  (void)info;
  napi_value r;
  NAPI_CALL(env, napi_create_int32(env, (int)poly_import_last_error_code(), &r));
  return r;
}

static napi_value napi_poly_import_error_msg(napi_env env, napi_callback_info info) {
  (void)info;
  const char *msg = poly_import_last_error_message();
  napi_value r;
  NAPI_CALL(env, napi_create_string_utf8(env, msg ? msg : "", NAPI_AUTO_LENGTH, &r));
  return r;
}

NAPI_MODULE_INIT() {
  napi_property_descriptor props[] = {
      /* Context */
      DECLARE_NAPI_METHOD("poly_ctx_new", napi_poly_ctx_new),
      DECLARE_NAPI_METHOD("poly_ctx_destroy", napi_poly_ctx_destroy),
      DECLARE_NAPI_METHOD("poly_ctx_named_count", napi_poly_ctx_named_count),

      /* Constants */
      DECLARE_NAPI_METHOD("poly_const_float", napi_poly_const_float),
      DECLARE_NAPI_METHOD("poly_const_double", napi_poly_const_double),
      DECLARE_NAPI_METHOD("poly_const_int", napi_poly_const_int),
      DECLARE_NAPI_METHOD("poly_const_float_by_id", napi_poly_const_float_by_id),
      DECLARE_NAPI_METHOD("poly_const_int_by_id", napi_poly_const_int_by_id),

      /* ALU */
      DECLARE_NAPI_METHOD("poly_contiguous", napi_poly_contiguous),
      DECLARE_NAPI_METHOD("poly_alu1", napi_poly_alu1),
      DECLARE_NAPI_METHOD("poly_alu2", napi_poly_alu2),
      DECLARE_NAPI_METHOD("poly_binop", napi_poly_binop),
      DECLARE_NAPI_METHOD("poly_alu3", napi_poly_alu3),

      /* Graph construction */
      DECLARE_NAPI_METHOD("poly_store_val", napi_poly_store_val),
      DECLARE_NAPI_METHOD("poly_sink1", napi_poly_sink1),
      DECLARE_NAPI_METHOD("poly_sink_n", napi_poly_sink_n),
      DECLARE_NAPI_METHOD("poly_uop_placeholder_like", napi_poly_uop_placeholder_like),
      DECLARE_NAPI_METHOD("poly_uop_range", napi_poly_uop_range),
      DECLARE_NAPI_METHOD("poly_uop_index", napi_poly_uop_index),
      DECLARE_NAPI_METHOD("poly_uop_load", napi_poly_uop_load),
      DECLARE_NAPI_METHOD("poly_uop_store", napi_poly_uop_store),
      DECLARE_NAPI_METHOD("poly_uop_set", napi_poly_uop_set),
      DECLARE_NAPI_METHOD("poly_uop_group", napi_poly_uop_group),
      DECLARE_NAPI_METHOD("poly_uop_end", napi_poly_uop_end),
      DECLARE_NAPI_METHOD("poly_uop_sink", napi_poly_uop_sink),
      DECLARE_NAPI_METHOD("poly_uop_sink_ex", napi_poly_uop_sink_ex),
      DECLARE_NAPI_METHOD("poly_uop_call", napi_poly_uop_call),
      DECLARE_NAPI_METHOD("poly_uop_after", napi_poly_uop_after),
      DECLARE_NAPI_METHOD("poly_uop_reduce", napi_poly_uop_reduce),
      DECLARE_NAPI_METHOD("poly_uop_flatten", napi_poly_uop_flatten),
      DECLARE_NAPI_METHOD("poly_uop_numel", napi_poly_uop_numel),
      DECLARE_NAPI_METHOD("poly_uop_n_src", napi_poly_uop_n_src),
      DECLARE_NAPI_METHOD("poly_uop_src", napi_poly_uop_src),
      DECLARE_NAPI_METHOD("poly_uop_call_grad_fxn_key", napi_poly_uop_call_grad_fxn_key),
      DECLARE_NAPI_METHOD("poly_register_buffer_by_id", napi_poly_register_buffer_by_id),
      DECLARE_NAPI_METHOD("poly_register_existing_buffer", napi_poly_register_existing_buffer),

      /* Buffers */
      DECLARE_NAPI_METHOD("poly_buffer_by_id", napi_poly_buffer_by_id),
      DECLARE_NAPI_METHOD("poly_buffer_on_device_by_id", napi_poly_buffer_on_device_by_id),
      DECLARE_NAPI_METHOD("poly_buffer_f32", napi_poly_buffer_f32),
      DECLARE_NAPI_METHOD("poly_buffer_f64", napi_poly_buffer_f64),
      DECLARE_NAPI_METHOD("poly_buffer_from_host", napi_poly_buffer_from_host),
      DECLARE_NAPI_METHOD("poly_tensor_from_host_by_id", napi_poly_tensor_from_host_by_id),
      DECLARE_NAPI_METHOD("poly_tensor_const_int_by_id", napi_poly_tensor_const_int_by_id),
      DECLARE_NAPI_METHOD("poly_tensor_const_float_by_id", napi_poly_tensor_const_float_by_id),
      DECLARE_NAPI_METHOD("poly_tensor_const_like_int", napi_poly_tensor_const_like_int),
      DECLARE_NAPI_METHOD("poly_tensor_const_like_float", napi_poly_tensor_const_like_float),
      DECLARE_NAPI_METHOD("poly_tensor_full_int_by_id", napi_poly_tensor_full_int_by_id),
      DECLARE_NAPI_METHOD("poly_tensor_full_float_by_id", napi_poly_tensor_full_float_by_id),
      DECLARE_NAPI_METHOD("poly_tensor_arange_int_by_id", napi_poly_tensor_arange_int_by_id),
      DECLARE_NAPI_METHOD("poly_tensor_arange_float_by_id", napi_poly_tensor_arange_float_by_id),
      DECLARE_NAPI_METHOD("poly_tensor_linspace_by_id", napi_poly_tensor_linspace_by_id),
      DECLARE_NAPI_METHOD("poly_tensor_eye_by_id", napi_poly_tensor_eye_by_id),
      DECLARE_NAPI_METHOD("poly_tensor_manual_seed", napi_poly_tensor_manual_seed),
      DECLARE_NAPI_METHOD("poly_tensor_rand_by_id", napi_poly_tensor_rand_by_id),
      DECLARE_NAPI_METHOD("poly_tensor_randn_by_id", napi_poly_tensor_randn_by_id),
      DECLARE_NAPI_METHOD("poly_buffer_get_ptr", napi_poly_buffer_get_ptr),
      DECLARE_NAPI_METHOD("poly_buffer_is_allocated", napi_poly_buffer_is_allocated),
      DECLARE_NAPI_METHOD("poly_buffer_get_key", napi_poly_buffer_get_key),
      DECLARE_NAPI_METHOD(
          "poly_set_frontend_buffer_release", napi_poly_set_frontend_buffer_release
      ),
      DECLARE_NAPI_METHOD(
          "poly_ctx_set_frontend_buffer_release", napi_poly_ctx_set_frontend_buffer_release
      ),
      DECLARE_NAPI_METHOD("poly_uop_key", napi_poly_uop_key),
      DECLARE_NAPI_METHOD("poly_uop_op", napi_poly_uop_op),
      DECLARE_NAPI_METHOD("poly_uop_device", napi_poly_uop_device),
      DECLARE_NAPI_METHOD("poly_uop_substitute", napi_poly_uop_substitute),
      DECLARE_NAPI_METHOD("poly_uop_has_buffer_identity", napi_poly_uop_has_buffer_identity),
      DECLARE_NAPI_METHOD("poly_uop_get_buffer_identity", napi_poly_uop_get_buffer_identity),
      DECLARE_NAPI_METHOD("poly_uop_buffer", napi_poly_uop_buffer),
      DECLARE_NAPI_METHOD("poly_uop_reachable", napi_poly_uop_reachable),
      DECLARE_NAPI_METHOD("poly_realize_uops", napi_poly_realize_uops),
      DECLARE_NAPI_METHOD("poly_tensor_empty_by_id", napi_poly_tensor_empty_by_id),
      DECLARE_NAPI_METHOD("poly_tensor_create_with_roots", napi_poly_tensor_create_with_roots),
      DECLARE_NAPI_METHOD("poly_tensor_create_result_like", napi_poly_tensor_create_result_like),
      DECLARE_NAPI_METHOD("poly_tensor_replace_roots", napi_poly_tensor_replace_roots),
      DECLARE_NAPI_METHOD("poly_tensor_to_device", napi_poly_tensor_to_device),
      DECLARE_NAPI_METHOD("poly_tensor_assign", napi_poly_tensor_assign),
      DECLARE_NAPI_METHOD("poly_tensor_alu1", napi_poly_tensor_alu1),
      DECLARE_NAPI_METHOD("poly_tensor_alu2", napi_poly_tensor_alu2),
      DECLARE_NAPI_METHOD("poly_tensor_alu3", napi_poly_tensor_alu3),
      DECLARE_NAPI_METHOD("poly_tensor_div", napi_poly_tensor_div),
      DECLARE_NAPI_METHOD("poly_tensor_exp", napi_poly_tensor_exp),
      DECLARE_NAPI_METHOD("poly_tensor_log", napi_poly_tensor_log),
      DECLARE_NAPI_METHOD("poly_tensor_cos", napi_poly_tensor_cos),
      DECLARE_NAPI_METHOD("poly_tensor_tan", napi_poly_tensor_tan),
      DECLARE_NAPI_METHOD("poly_tensor_log1p", napi_poly_tensor_log1p),
      DECLARE_NAPI_METHOD("poly_tensor_expm1", napi_poly_tensor_expm1),
      DECLARE_NAPI_METHOD("poly_tensor_gelu", napi_poly_tensor_gelu),
      DECLARE_NAPI_METHOD("poly_tensor_quick_gelu", napi_poly_tensor_quick_gelu),
      DECLARE_NAPI_METHOD("poly_tensor_detach", napi_poly_tensor_detach),
      DECLARE_NAPI_METHOD("poly_tensor_contiguous_backward", napi_poly_tensor_contiguous_backward),
      DECLARE_NAPI_METHOD("poly_tensor_sum", napi_poly_tensor_sum),
      DECLARE_NAPI_METHOD("poly_tensor_sum_dtype_by_id", napi_poly_tensor_sum_dtype_by_id),
      DECLARE_NAPI_METHOD("poly_tensor_max", napi_poly_tensor_max),
      DECLARE_NAPI_METHOD("poly_tensor_argmax", napi_poly_tensor_argmax),
      DECLARE_NAPI_METHOD("poly_tensor_minimum", napi_poly_tensor_minimum),
      DECLARE_NAPI_METHOD("poly_tensor_dot", napi_poly_tensor_dot),
      DECLARE_NAPI_METHOD("poly_tensor_dot_dtype_by_id", napi_poly_tensor_dot_dtype_by_id),
      DECLARE_NAPI_METHOD("poly_tensor_qr_ex", napi_poly_tensor_qr_ex),
      DECLARE_NAPI_METHOD("poly_tensor_triangular_solve", napi_poly_tensor_triangular_solve),
      DECLARE_NAPI_METHOD("poly_tensor_cholesky", napi_poly_tensor_cholesky),
      DECLARE_NAPI_METHOD("poly_tensor_cholesky_solve", napi_poly_tensor_cholesky_solve),
      DECLARE_NAPI_METHOD("poly_tensor_solve", napi_poly_tensor_solve),
      DECLARE_NAPI_METHOD("poly_tensor_lstsq", napi_poly_tensor_lstsq),
      DECLARE_NAPI_METHOD("poly_tensor_scatter", napi_poly_tensor_scatter),
      DECLARE_NAPI_METHOD("poly_tensor_scatter_reduce", napi_poly_tensor_scatter_reduce),
      DECLARE_NAPI_METHOD("poly_tensor_einsum", napi_poly_tensor_einsum),
      DECLARE_NAPI_METHOD("poly_tensor_rearrange", napi_poly_tensor_rearrange),
      DECLARE_NAPI_METHOD("poly_tensor_sort", napi_poly_tensor_sort),
      DECLARE_NAPI_METHOD("poly_tensor_topk", napi_poly_tensor_topk),
      DECLARE_NAPI_METHOD("poly_tensor_softmax", napi_poly_tensor_softmax),
      DECLARE_NAPI_METHOD("poly_tensor_log_softmax", napi_poly_tensor_log_softmax),
      DECLARE_NAPI_METHOD("poly_tensor_cast_by_id", napi_poly_tensor_cast_by_id),
      DECLARE_NAPI_METHOD("poly_tensor_bitcast_by_id", napi_poly_tensor_bitcast_by_id),
      DECLARE_NAPI_METHOD("poly_tensor_contiguous", napi_poly_tensor_contiguous),
      DECLARE_NAPI_METHOD("poly_tensor_reshape", napi_poly_tensor_reshape),
      DECLARE_NAPI_METHOD("poly_tensor_expand", napi_poly_tensor_expand),
      DECLARE_NAPI_METHOD("poly_tensor_permute", napi_poly_tensor_permute),
      DECLARE_NAPI_METHOD("poly_tensor_shrink", napi_poly_tensor_shrink),
      DECLARE_NAPI_METHOD("poly_tensor_flip", napi_poly_tensor_flip),
      DECLARE_NAPI_METHOD("poly_tensor_pad_value_bool", napi_poly_tensor_pad_value_bool),
      DECLARE_NAPI_METHOD("poly_tensor_pad_value_int", napi_poly_tensor_pad_value_int),
      DECLARE_NAPI_METHOD("poly_tensor_pad_value_float", napi_poly_tensor_pad_value_float),
      DECLARE_NAPI_METHOD("poly_tensor_pool", napi_poly_tensor_pool),
      DECLARE_NAPI_METHOD("poly_tensor_max_pool2d", napi_poly_tensor_max_pool2d),
      DECLARE_NAPI_METHOD("poly_tensor_conv2d", napi_poly_tensor_conv2d),
      DECLARE_NAPI_METHOD("poly_tensor_conv2d_dtype_by_id", napi_poly_tensor_conv2d_dtype_by_id),
      DECLARE_NAPI_METHOD("poly_tensor_batchnorm", napi_poly_tensor_batchnorm),
      DECLARE_NAPI_METHOD("poly_tensor_one_hot", napi_poly_tensor_one_hot),
      DECLARE_NAPI_METHOD("poly_tensor_gather_dim", napi_poly_tensor_gather_dim),
      DECLARE_NAPI_METHOD("poly_tensor_index_select", napi_poly_tensor_index_select),
      DECLARE_NAPI_METHOD("poly_tensor_clone_into", napi_poly_tensor_clone_into),
      DECLARE_NAPI_METHOD("poly_tensor_clone", napi_poly_tensor_clone),
      DECLARE_NAPI_METHOD("poly_tensor_retain", napi_poly_tensor_retain),
      DECLARE_NAPI_METHOD("poly_tensor_release", napi_poly_tensor_release),
      DECLARE_NAPI_METHOD("poly_tensor_uop", napi_poly_tensor_uop),
      DECLARE_NAPI_METHOD("poly_uop_retain", napi_poly_uop_retain),
      DECLARE_NAPI_METHOD("poly_uop_release", napi_poly_uop_release),
      DECLARE_NAPI_METHOD("poly_tensor_uop_logical", napi_poly_tensor_uop_logical),
      DECLARE_NAPI_METHOD("poly_tensor_uop_physical", napi_poly_tensor_uop_physical),
      DECLARE_NAPI_METHOD("poly_tensor_logical_policy", napi_poly_tensor_logical_policy),
      DECLARE_NAPI_METHOD("poly_tensor_logical_state", napi_poly_tensor_logical_state),
      DECLARE_NAPI_METHOD("poly_tensor_set_logical_policy", napi_poly_tensor_set_logical_policy),
      DECLARE_NAPI_METHOD("poly_tensor_device", napi_poly_tensor_device),
      DECLARE_NAPI_METHOD("poly_tensor_custom_kernel", napi_poly_tensor_custom_kernel),
      DECLARE_NAPI_METHOD("poly_realize_tensors", napi_poly_realize_tensors),
      DECLARE_NAPI_METHOD("poly_optim_build_step", napi_poly_optim_build_step),
      DECLARE_NAPI_METHOD("poly_buffer_read", napi_poly_buffer_read),
      DECLARE_NAPI_METHOD("poly_buffer_write", napi_poly_buffer_write),
      DECLARE_NAPI_METHOD(
          "poly_buffer_ensure_device_allocated", napi_poly_buffer_ensure_device_allocated
      ),
      DECLARE_NAPI_METHOD("poly_ctx_set_preferred_device", napi_poly_ctx_set_preferred_device),
      DECLARE_NAPI_METHOD("poly_ctx_set_logical_policy", napi_poly_ctx_set_logical_policy),
      DECLARE_NAPI_METHOD("poly_ctx_get_logical_policy", napi_poly_ctx_get_logical_policy),
      DECLARE_NAPI_METHOD("poly_ctx_collect", napi_poly_ctx_collect),
      DECLARE_NAPI_METHOD("poly_ctx_stats", napi_poly_ctx_stats),
      DECLARE_NAPI_METHOD("poly_ctx_reset_counters", napi_poly_ctx_reset_counters),
      DECLARE_NAPI_METHOD("poly_can_run_op", napi_poly_can_run_op),
      DECLARE_NAPI_METHOD("poly_jit_new", napi_poly_jit_new),
      DECLARE_NAPI_METHOD("poly_jit_free", napi_poly_jit_free),
      DECLARE_NAPI_METHOD("poly_jit_set_prune", napi_poly_jit_set_prune),
      DECLARE_NAPI_METHOD("poly_jit_begin_capture", napi_poly_jit_begin_capture),
      DECLARE_NAPI_METHOD("poly_jit_end_capture", napi_poly_jit_end_capture),
      DECLARE_NAPI_METHOD("poly_jit_cancel_capture", napi_poly_jit_cancel_capture),
      DECLARE_NAPI_METHOD("poly_jit_is_captured", napi_poly_jit_is_captured),
      DECLARE_NAPI_METHOD("poly_jit_schedule_count", napi_poly_jit_schedule_count),
      DECLARE_NAPI_METHOD("poly_jit_run", napi_poly_jit_run),

      /* Autograd */
      DECLARE_NAPI_METHOD("poly_grad", napi_poly_grad),
      DECLARE_NAPI_METHOD("poly_grad_many", napi_poly_grad_many),
      DECLARE_NAPI_METHOD("poly_detach", napi_poly_detach),
      DECLARE_NAPI_METHOD("poly_cast_by_id", napi_poly_cast_by_id),
      DECLARE_NAPI_METHOD("poly_dtype_id_by_name", napi_poly_dtype_id_by_name),
      DECLARE_NAPI_METHOD("poly_uop_dtype_id", napi_poly_uop_dtype_id),
      DECLARE_NAPI_METHOD("poly_device_by_name", napi_poly_device_by_name),
      DECLARE_NAPI_METHOD("poly_device_name", napi_poly_device_name),
      DECLARE_NAPI_METHOD("poly_device_is_host_addressable", napi_poly_device_is_host_addressable),

      /* Movement ops (shape-taking) */
      DECLARE_NAPI_METHOD("poly_reshape", napi_poly_reshape),
      DECLARE_NAPI_METHOD("poly_expand", napi_poly_expand),
      DECLARE_NAPI_METHOD("poly_permute", napi_poly_permute),
      DECLARE_NAPI_METHOD("poly_flip", napi_poly_flip),
      DECLARE_NAPI_METHOD("poly_shrink", napi_poly_shrink),
      DECLARE_NAPI_METHOD("poly_shrink_uop", napi_poly_shrink_uop),
      DECLARE_NAPI_METHOD("poly_pad", napi_poly_pad),
      DECLARE_NAPI_METHOD("poly_reduce_axis", napi_poly_reduce_axis),

      /* Shape-returning ops (kept: layernorm, causal_mask, logsumexp, var_reduce, linear) */
      DECLARE_NAPI_METHOD("poly_logsumexp", napi_poly_logsumexp),
      DECLARE_NAPI_METHOD("poly_layernorm", napi_poly_layernorm),
      DECLARE_NAPI_METHOD("poly_linear", napi_poly_linear),
      DECLARE_NAPI_METHOD("poly_causal_mask", napi_poly_causal_mask),

      /* Composed elementwise (unary) */
      DECLARE_NAPI_METHOD("poly_exp", napi_poly_exp),
      DECLARE_NAPI_METHOD("poly_log", napi_poly_log),
      DECLARE_NAPI_METHOD("poly_log1p", napi_poly_log1p),
      DECLARE_NAPI_METHOD("poly_expm1", napi_poly_expm1),
      DECLARE_NAPI_METHOD("poly_sin", napi_poly_sin),
      DECLARE_NAPI_METHOD("poly_cos", napi_poly_cos),
      DECLARE_NAPI_METHOD("poly_tan", napi_poly_tan),
      DECLARE_NAPI_METHOD("poly_erf", napi_poly_erf),
      DECLARE_NAPI_METHOD("poly_erfc", napi_poly_erfc),
      DECLARE_NAPI_METHOD("poly_erfinv", napi_poly_erfinv),
      DECLARE_NAPI_METHOD("poly_ndtri", napi_poly_ndtri),
      DECLARE_NAPI_METHOD("poly_digamma", napi_poly_digamma),
      DECLARE_NAPI_METHOD("poly_lgamma", napi_poly_lgamma),
      DECLARE_NAPI_METHOD("poly_sigmoid", napi_poly_sigmoid),
      DECLARE_NAPI_METHOD("poly_tanh_act", napi_poly_tanh_act),
      DECLARE_NAPI_METHOD("poly_abs", napi_poly_abs),
      DECLARE_NAPI_METHOD("poly_sign", napi_poly_sign),
      DECLARE_NAPI_METHOD("poly_square", napi_poly_square),
      DECLARE_NAPI_METHOD("poly_rsqrt", napi_poly_rsqrt),
      DECLARE_NAPI_METHOD("poly_ceil", napi_poly_ceil),
      DECLARE_NAPI_METHOD("poly_floor", napi_poly_floor),
      DECLARE_NAPI_METHOD("poly_round_f", napi_poly_round_f),
      DECLARE_NAPI_METHOD("poly_isinf", napi_poly_isinf),
      DECLARE_NAPI_METHOD("poly_isnan", napi_poly_isnan),
      DECLARE_NAPI_METHOD("poly_relu", napi_poly_relu),
      DECLARE_NAPI_METHOD("poly_relu6", napi_poly_relu6),
      DECLARE_NAPI_METHOD("poly_gelu", napi_poly_gelu),
      DECLARE_NAPI_METHOD("poly_quick_gelu", napi_poly_quick_gelu),
      DECLARE_NAPI_METHOD("poly_silu", napi_poly_silu),
      DECLARE_NAPI_METHOD("poly_mish", napi_poly_mish),
      DECLARE_NAPI_METHOD("poly_hardswish", napi_poly_hardswish),
      DECLARE_NAPI_METHOD("poly_hardsigmoid", napi_poly_hardsigmoid),

      /* Composed with extra double args */
      DECLARE_NAPI_METHOD("poly_leaky_relu", napi_poly_leaky_relu),
      DECLARE_NAPI_METHOD("poly_elu", napi_poly_elu),
      DECLARE_NAPI_METHOD("poly_softplus", napi_poly_softplus),
      DECLARE_NAPI_METHOD("poly_hardtanh", napi_poly_hardtanh),
      DECLARE_NAPI_METHOD("poly_clamp", napi_poly_clamp),

      /* Comparisons */
      DECLARE_NAPI_METHOD("poly_eq", napi_poly_eq),
      DECLARE_NAPI_METHOD("poly_ne", napi_poly_ne),
      DECLARE_NAPI_METHOD("poly_gt", napi_poly_gt),
      DECLARE_NAPI_METHOD("poly_ge", napi_poly_ge),
      DECLARE_NAPI_METHOD("poly_le", napi_poly_le),
      DECLARE_NAPI_METHOD("poly_maximum", napi_poly_maximum),
      DECLARE_NAPI_METHOD("poly_minimum", napi_poly_minimum),
      DECLARE_NAPI_METHOD("poly_where_op", napi_poly_where_op),

      /* Creation */
      DECLARE_NAPI_METHOD("poly_rand", napi_poly_rand),
      DECLARE_NAPI_METHOD("poly_randn", napi_poly_randn),
      DECLARE_NAPI_METHOD("poly_arange", napi_poly_arange),
      DECLARE_NAPI_METHOD("poly_eye", napi_poly_eye),
      DECLARE_NAPI_METHOD("poly_linspace", napi_poly_linspace),
      DECLARE_NAPI_METHOD("poly_full", napi_poly_full),
      DECLARE_NAPI_METHOD("poly_tril", napi_poly_tril),
      DECLARE_NAPI_METHOD("poly_triu", napi_poly_triu),

      /* Metadata */
      DECLARE_NAPI_METHOD("poly_op_count", napi_poly_op_count),
      DECLARE_NAPI_METHOD("poly_op_name", napi_poly_op_name),
      DECLARE_NAPI_METHOD("poly_abi_version", napi_poly_abi_version),

      /* Cache cleanup */
      DECLARE_NAPI_METHOD("poly_cpu_cache_flush", napi_poly_cpu_cache_flush),

      /* PolyModel / model runtime */
      DECLARE_NAPI_METHOD("poly_model_from_ir", napi_poly_model_from_ir),
      DECLARE_NAPI_METHOD("poly_model_from_program", napi_poly_model_from_program),
      DECLARE_NAPI_METHOD("poly_model_from_sinks", napi_poly_model_from_sinks),
      DECLARE_NAPI_METHOD("poly_model_from_binding_arrays", napi_poly_model_from_binding_arrays),
      DECLARE_NAPI_METHOD("poly_model_free", napi_poly_model_free),
      DECLARE_NAPI_METHOD("poly_model_set_device", napi_poly_model_set_device),
      DECLARE_NAPI_METHOD("poly_model_define_module_arrays", napi_poly_model_define_module_arrays),
      DECLARE_NAPI_METHOD(
          "poly_model_set_device_map_arrays", napi_poly_model_set_device_map_arrays
      ),
      DECLARE_NAPI_METHOD("poly_sequential_from_json", napi_poly_sequential_from_json),
      DECLARE_NAPI_METHOD("poly_graph_from_json", napi_poly_graph_from_json),
      DECLARE_NAPI_METHOD("poly_mlp_from_json", napi_poly_mlp_from_json),
      DECLARE_NAPI_METHOD("poly_tabm_from_json", napi_poly_tabm_from_json),
      DECLARE_NAPI_METHOD("poly_nam_from_json", napi_poly_nam_from_json),
      DECLARE_NAPI_METHOD("poly_model_param_count", napi_poly_model_param_count),
      DECLARE_NAPI_METHOD("poly_model_param_name", napi_poly_model_param_name),
      DECLARE_NAPI_METHOD("poly_model_param_shape", napi_poly_model_param_shape),
      DECLARE_NAPI_METHOD("poly_model_param_data", napi_poly_model_param_data),
      DECLARE_NAPI_METHOD("poly_model_param_dtype_id", napi_poly_model_param_dtype_id),
      DECLARE_NAPI_METHOD("poly_model_param_trainable", napi_poly_model_param_trainable),
      DECLARE_NAPI_METHOD("poly_model_set_param_trainable", napi_poly_model_set_param_trainable),
      DECLARE_NAPI_METHOD("poly_model_buf_count", napi_poly_model_buf_count),
      DECLARE_NAPI_METHOD("poly_model_entrypoints", napi_poly_model_entrypoints),
      DECLARE_NAPI_METHOD("poly_model_buf_name", napi_poly_model_buf_name),
      DECLARE_NAPI_METHOD("poly_model_buf_role", napi_poly_model_buf_role),
      DECLARE_NAPI_METHOD("poly_model_buf_trainable", napi_poly_model_buf_trainable),
      DECLARE_NAPI_METHOD("poly_model_set_buf_trainable", napi_poly_model_set_buf_trainable),
      DECLARE_NAPI_METHOD("poly_model_buf_shape", napi_poly_model_buf_shape),
      DECLARE_NAPI_METHOD("poly_model_buf_data", napi_poly_model_buf_data),
      DECLARE_NAPI_METHOD("poly_model_write_buf", napi_poly_model_write_buf),
      DECLARE_NAPI_METHOD("poly_model_buf_dtype_id", napi_poly_model_buf_dtype_id),
      DECLARE_NAPI_METHOD("poly_model_export_weights", napi_poly_model_export_weights),
      DECLARE_NAPI_METHOD("poly_model_import_weights", napi_poly_model_import_weights),
      DECLARE_NAPI_METHOD("poly_model_export_ir", napi_poly_model_export_ir),
      DECLARE_NAPI_METHOD("poly_model_export_program", napi_poly_model_export_program),
      DECLARE_NAPI_METHOD("poly_model_save_bundle", napi_poly_model_save_bundle),
      DECLARE_NAPI_METHOD("poly_model_from_bundle", napi_poly_model_from_bundle),
      DECLARE_NAPI_METHOD("poly_model_set_optimizer", napi_poly_model_set_optimizer),
      DECLARE_NAPI_METHOD("poly_model_forward", napi_poly_model_forward),
      DECLARE_NAPI_METHOD("poly_model_call", napi_poly_model_call),
      DECLARE_NAPI_METHOD(
          "poly_model_entrypoint_output_count", napi_poly_model_entrypoint_output_count
      ),
      DECLARE_NAPI_METHOD(
          "poly_model_entrypoint_output_name", napi_poly_model_entrypoint_output_name
      ),
      DECLARE_NAPI_METHOD("poly_model_train_step", napi_poly_model_train_step),
      /* Shape-on-UOp accessors */
      DECLARE_NAPI_METHOD("poly_uop_ndim", napi_poly_uop_ndim),
      DECLARE_NAPI_METHOD("poly_uop_max_shape_dims", napi_poly_uop_max_shape_dims),
      /* v2 composed ops */
      DECLARE_NAPI_METHOD("poly_softmax", napi_poly_softmax),
      DECLARE_NAPI_METHOD("poly_log_softmax", napi_poly_log_softmax),
      DECLARE_NAPI_METHOD("poly_dot", napi_poly_dot),
      DECLARE_NAPI_METHOD("poly_qr", napi_poly_qr),
      DECLARE_NAPI_METHOD("poly_qr_ex", napi_poly_qr_ex),
      DECLARE_NAPI_METHOD("poly_cholesky", napi_poly_cholesky),
      DECLARE_NAPI_METHOD("poly_cholesky_solve", napi_poly_cholesky_solve),
      DECLARE_NAPI_METHOD("poly_triangular_solve", napi_poly_triangular_solve),
      DECLARE_NAPI_METHOD("poly_solve", napi_poly_solve),
      DECLARE_NAPI_METHOD("poly_lstsq", napi_poly_lstsq),
      DECLARE_NAPI_METHOD("poly_cross_entropy", napi_poly_cross_entropy),
      DECLARE_NAPI_METHOD("poly_gather", napi_poly_gather),
      DECLARE_NAPI_METHOD("poly_gather_dim", napi_poly_gather_dim),
      DECLARE_NAPI_METHOD("poly_scatter", napi_poly_scatter),
      DECLARE_NAPI_METHOD("poly_scatter_reduce", napi_poly_scatter_reduce),
      DECLARE_NAPI_METHOD("poly_argmax", napi_poly_argmax),
      DECLARE_NAPI_METHOD("poly_sort", napi_poly_sort),
      DECLARE_NAPI_METHOD("poly_argsort", napi_poly_argsort),
      DECLARE_NAPI_METHOD("poly_topk", napi_poly_topk),
      DECLARE_NAPI_METHOD("poly_einsum", napi_poly_einsum),
      DECLARE_NAPI_METHOD("poly_rearrange", napi_poly_rearrange),
      DECLARE_NAPI_METHOD("poly_sum_reduce", napi_poly_sum_reduce),
      DECLARE_NAPI_METHOD("poly_max_reduce", napi_poly_max_reduce),
      DECLARE_NAPI_METHOD("poly_mean_reduce", napi_poly_mean_reduce),
      DECLARE_NAPI_METHOD("poly_var_reduce", napi_poly_var_reduce),
      DECLARE_NAPI_METHOD("poly_pad_value", napi_poly_pad_value),
      DECLARE_NAPI_METHOD("poly_pool", napi_poly_pool),
      DECLARE_NAPI_METHOD("poly_max_pool2d", napi_poly_max_pool2d),
      DECLARE_NAPI_METHOD("poly_conv2d", napi_poly_conv2d),
      DECLARE_NAPI_METHOD("poly_batchnorm", napi_poly_batchnorm),
      DECLARE_NAPI_METHOD("poly_one_hot", napi_poly_one_hot),
      DECLARE_NAPI_METHOD("poly_index_select", napi_poly_index_select),

      /* Tokenizer */
      DECLARE_NAPI_METHOD("poly_tokenizer_from_json", napi_poly_tokenizer_from_json),
      DECLARE_NAPI_METHOD("poly_tokenize", napi_poly_tokenize),
      DECLARE_NAPI_METHOD("poly_detokenize", napi_poly_detokenize),
      DECLARE_NAPI_METHOD("poly_tokenizer_free", napi_poly_tokenizer_free),
      DECLARE_NAPI_METHOD("poly_tokenizer_vocab_size", napi_poly_tokenizer_vocab_size),
      DECLARE_NAPI_METHOD("poly_tokenizer_bos_id", napi_poly_tokenizer_bos_id),
      DECLARE_NAPI_METHOD("poly_tokenizer_eos_id", napi_poly_tokenizer_eos_id),

      /* Loaders */
      DECLARE_NAPI_METHOD("poly_hf_load", napi_poly_hf_load),
      DECLARE_NAPI_METHOD("poly_gguf_load", napi_poly_gguf_load),
      DECLARE_NAPI_METHOD("poly_tokenizer_from_gguf", napi_poly_tokenizer_from_gguf),

      /* Import error */
      DECLARE_NAPI_METHOD("poly_import_last_error_code", napi_poly_import_error_code),
      DECLARE_NAPI_METHOD("poly_import_last_error_message", napi_poly_import_error_msg),
  };

  NAPI_CALL(env, napi_define_properties(env, exports, sizeof(props) / sizeof(props[0]), props));

  return exports;
}
