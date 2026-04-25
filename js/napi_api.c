/*
 * napi_api.c -- N-API addon wrapping polygrad C core for Node.js.
 *
 * Exposes frontend.h functions as native bindings. Opaque pointers
 * are passed as napi_external values. Int64 arrays are read from
 * JS number[] element-by-element. Shape-returning ops return objects.
 */

#include <node_api.h>
#include <string.h>
#include <stdlib.h>

#include "polygrad.h"
#include "frontend.h"
#include "instance.h"
#include "tokenizer.h"
#include "loaders/hf_decode.h"
#include "loaders/gguf_decode.h"
#include "loaders/import_error.h"
#include "bundle.h"
#include "models/mlp.h"
#include "models/tabm.h"
#include "models/nam.h"
#include "engine/schedule.h"

/* ── Error-checking macro ──────────────────────────────────────────────── */

#define NAPI_CALL(env, call)                                       \
  do {                                                             \
    napi_status _s = (call);                                       \
    if (_s != napi_ok) {                                           \
      const napi_extended_error_info *_ei;                         \
      napi_get_last_error_info((env), &_ei);                       \
      napi_throw_error((env), NULL,                                \
        _ei->error_message ? _ei->error_message : "N-API error");  \
      return NULL;                                                 \
    }                                                              \
  } while (0)

/* ── Helpers ───────────────────────────────────────────────────────────── */

static void *get_external(napi_env env, napi_value val) {
  void *ptr = NULL;
  napi_get_value_external(env, val, &ptr);
  return ptr;
}

static napi_value make_external(napi_env env, void *ptr) {
  napi_value result;
  NAPI_CALL(env, napi_create_external(env, ptr, NULL, NULL, &result));
  return result;
}

static napi_env g_frontend_buffer_release_env = NULL;
static napi_ref g_frontend_buffer_release_ref = NULL;

static void napi_frontend_buffer_release(uintptr_t buffer_key) {
  if (!g_frontend_buffer_release_env || !g_frontend_buffer_release_ref) return;

  napi_handle_scope scope;
  if (napi_open_handle_scope(g_frontend_buffer_release_env, &scope) != napi_ok) return;

  napi_value fn, global, arg, result;
  if (napi_get_reference_value(g_frontend_buffer_release_env, g_frontend_buffer_release_ref, &fn) != napi_ok)
    goto done;
  if (napi_get_global(g_frontend_buffer_release_env, &global) != napi_ok)
    goto done;
  if (napi_create_bigint_uint64(g_frontend_buffer_release_env, (uint64_t)buffer_key, &arg) != napi_ok)
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

static napi_value make_shape_result(napi_env env, void *uop_ptr,
                                    int64_t *shape, int ndim) {
  napi_value obj, uop_val, shape_arr;
  NAPI_CALL(env, napi_create_object(env, &obj));
  NAPI_CALL(env, napi_create_external(env, uop_ptr, NULL, NULL, &uop_val));
  NAPI_CALL(env, napi_create_array_with_length(env, (size_t)ndim, &shape_arr));
  for (int i = 0; i < ndim; i++) {
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

static napi_value make_float32_array_copy(napi_env env, const float *src, size_t len) {
  if (!src) {
    napi_value result;
    napi_get_null(env, &result);
    return result;
  }

  void *dst = NULL;
  napi_value arraybuf, typed;
  NAPI_CALL(env, napi_create_arraybuffer(env, len * sizeof(float), &dst, &arraybuf));
  memcpy(dst, src, len * sizeof(float));
  NAPI_CALL(env, napi_create_typedarray(
    env, napi_float32_array, len, arraybuf, 0, &typed));
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
  NAPI_CALL(env, napi_create_typedarray(
    env, napi_uint8_array, len, arraybuf, 0, &typed));
  return typed;
}

static int read_io_bindings(
    napi_env env,
    napi_value names_val,
    napi_value arrays_val,
    PolyIOBinding **out_bindings,
    char ***out_names,
    int *out_n)
{
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
      for (uint32_t j = 0; j < i; j++) free(names[j]);
      free(names);
      free(bindings);
      return 0;
    }

    napi_typedarray_type type;
    size_t length;
    void *data = NULL;
    size_t byte_offset;
    napi_value arraybuf;
    napi_status status = napi_get_typedarray_info(
      env, data_val, &type, &length, &data, &arraybuf, &byte_offset);
    if (status != napi_ok) {
      for (uint32_t j = 0; j <= i; j++) free(names[j]);
      free(names);
      free(bindings);
      {
        const napi_extended_error_info *ei = NULL;
        napi_get_last_error_info(env, &ei);
        napi_throw_error(env, NULL,
          ei && ei->error_message ? ei->error_message : "N-API error");
      }
      return 0;
    }
    if (type != napi_float32_array) {
      for (uint32_t j = 0; j <= i; j++) free(names[j]);
      free(names);
      free(bindings);
      napi_throw_error(env, NULL, "polygrad: instance bindings must be Float32Array");
      return 0;
    }

    bindings[i].name = names[i];
    bindings[i].data = (float *)data;
  }

  *out_bindings = bindings;
  *out_names = names;
  *out_n = (int)n_names;
  return 1;
}

static void free_io_bindings(char **names, PolyIOBinding *bindings, int n) {
  if (names) {
    for (int i = 0; i < n; i++) free(names[i]);
    free(names);
  }
  free(bindings);
}

/* ── Macros for repetitive wrappers ────────────────────────────────────── */

/* Unary: (ctx, x) -> external */
#define NAPI_UNARY(cname)                                          \
  static napi_value napi_##cname(napi_env env, napi_callback_info info) { \
    napi_value argv[2];                                            \
    size_t argc = 2;                                               \
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL)); \
    PolyCtx *ctx = get_external(env, argv[0]);                     \
    PolyUOp *x = get_external(env, argv[1]);                       \
    PolyUOp *r = cname(ctx, x);                                   \
    return make_external(env, r);                                  \
  }

/* Binary: (ctx, a, b) -> external */
#define NAPI_BINARY(cname)                                         \
  static napi_value napi_##cname(napi_env env, napi_callback_info info) { \
    napi_value argv[3];                                            \
    size_t argc = 3;                                               \
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL)); \
    PolyCtx *ctx = get_external(env, argv[0]);                     \
    PolyUOp *a = get_external(env, argv[1]);                       \
    PolyUOp *b = get_external(env, argv[2]);                       \
    PolyUOp *r = cname(ctx, a, b);                                \
    return make_external(env, r);                                  \
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

/* ── Buffers ───────────────────────────────────────────────────────────── */

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

static napi_value napi_poly_buffer_from_host(napi_env env, napi_callback_info info) {
  napi_value argv[6];
  size_t argc = 6;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  napi_typedarray_type ta_type;
  size_t ta_len;
  void *ta_data;
  napi_value ab;
  size_t ab_off;
  NAPI_CALL(env, napi_get_typedarray_info(env, argv[1], &ta_type, &ta_len, &ta_data, &ab, &ab_off));
  size_t esz = 4;
  switch (ta_type) {
    case napi_int8_array: case napi_uint8_array: case napi_uint8_clamped_array: esz = 1; break;
    case napi_int16_array: case napi_uint16_array: esz = 2; break;
    case napi_int32_array: case napi_uint32_array: case napi_float32_array: esz = 4; break;
    case napi_float64_array: case napi_biguint64_array: case napi_bigint64_array: esz = 8; break;
    default: esz = 4;
  }
  size_t nbytes = ta_len * esz;
  int dtype_id;
  napi_get_value_int32(env, argv[3], &dtype_id);
  int64_t dims[16];
  int ndim = 0;
  napi_valuetype vt;
  napi_typeof(env, argv[4], &vt);
  if (vt == napi_object) {
    bool is_arr;
    napi_is_array(env, argv[4], &is_arr);
    if (is_arr) {
      uint32_t len;
      napi_get_array_length(env, argv[4], &len);
      if (len > 16) len = 16;
      for (uint32_t i = 0; i < len; i++) {
        napi_value el;
        napi_get_element(env, argv[4], i, &el);
        int64_t v;
        napi_get_value_int64(env, el, &v);
        dims[i] = v;
      }
      ndim = (int)len;
    }
  }
  return make_external(env, poly_buffer_from_host(ctx, ta_data, nbytes, dtype_id, dims, ndim));
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

static napi_value napi_poly_uop_get_buffer_identity(napi_env env, napi_callback_info info) {
  napi_value argv[1];
  size_t argc = 1;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyUOp *u = get_external(env, argv[0]);
  const PolyUOp *r = poly_uop_get_buffer_identity(u);
  return make_external(env, (void *)r);
}

/* poly_realize(ctx, uops[], n, out_uops[]) -- batched graph-side realize.
 * JS signature: realize(ctx, [uop0, uop1, ...]) -> [realizedUop0, ...]. */
static napi_value napi_poly_realize(napi_env env, napi_callback_info info) {
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
  int rc = poly_realize(ctx, in_arr, (int)n, out_arr);
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
    const int64_t *dims = poly_uop_dims(ctx, (const PolyUOp *)r);
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
  napi_value argv[3]; size_t argc = 3;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *x = get_external(env, argv[1]);
  int32_t diagonal;
  napi_get_value_int32(env, argv[2], &diagonal);
  return make_external(env, poly_tril(ctx, x, diagonal));
}

static napi_value napi_poly_triu(napi_env env, napi_callback_info info) {
  napi_value argv[3]; size_t argc = 3;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *x = get_external(env, argv[1]);
  int32_t diagonal;
  napi_get_value_int32(env, argv[2], &diagonal);
  return make_external(env, poly_triu(ctx, x, diagonal));
}

/* ── Realize (stateful builder) ────────────────────────────────────────── */

static napi_value napi_poly_realize_begin(napi_env env, napi_callback_info info) {
  napi_value argv[1];
  size_t argc = 1;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  poly_realize_begin(ctx);
  napi_value undef;
  napi_get_undefined(env, &undef);
  return undef;
}

static napi_value napi_poly_realize_bind(napi_env env, napi_callback_info info) {
  napi_value argv[3];
  size_t argc = 3;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *buf = get_external(env, argv[1]);
  /* argv[2] is a Float32Array or Float64Array */
  napi_typedarray_type type;
  size_t length;
  void *data;
  size_t byte_offset;
  napi_value arraybuf;
  NAPI_CALL(env, napi_get_typedarray_info(env, argv[2], &type, &length,
                                           &data, &arraybuf, &byte_offset));
  (void)type;
  poly_realize_bind(ctx, buf, data);
  napi_value undef;
  napi_get_undefined(env, &undef);
  return undef;
}

static napi_value napi_poly_realize_exec(napi_env env, napi_callback_info info) {
  napi_value argv[2];
  size_t argc = 2;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *sink = get_external(env, argv[1]);
  int rc = poly_realize_exec(ctx, sink);
  napi_value result;
  NAPI_CALL(env, napi_create_int32(env, rc, &result));
  return result;
}

/* ── Assign ────────────────────────────────────────────────────────────── */

static napi_value napi_poly_assign(napi_env env, napi_callback_info info) {
  napi_value argv[3];
  size_t argc = 3;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *target = get_external(env, argv[1]);
  PolyUOp *value = get_external(env, argv[2]);
  return make_external(env, poly_assign(ctx, target, value));
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
  int64_t shape[MAX_DIMS];
  int32_t ndim, axis;
  napi_get_value_int32(env, argv[3], &ndim);
  read_int64_array(env, argv[2], shape, MAX_DIMS);
  napi_get_value_int32(env, argv[4], &axis);
  double eps;
  napi_get_value_double(env, argv[5], &eps);
  int64_t out_shape[MAX_DIMS];
  int out_ndim = 0;
  PolyUOp *r = poly_layernorm(ctx, uop, shape, ndim, axis, eps,
                               out_shape, &out_ndim);
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
    PolyShape s = poly_uop_shape(ctx, r);
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
  PolyUOp *r = poly_causal_mask(ctx, T, out_shape, &out_ndim);
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

static napi_value napi_poly_sched_cache_flush(napi_env env, napi_callback_info info) {
  (void)info;
  poly_sched_cache_flush();
  napi_value undef;
  napi_get_undefined(env, &undef);
  return undef;
}

/* ── PolyInstance / model runtime ────────────────────────────────────── */

static napi_value napi_poly_instance_from_ir(napi_env env, napi_callback_info info) {
  napi_value argv[2];
  size_t argc = 2;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));

  napi_typedarray_type ir_type;
  size_t ir_len = 0;
  void *ir_data = NULL;
  size_t ir_offset;
  napi_value ir_buf;
  NAPI_CALL(env, napi_get_typedarray_info(
    env, argv[0], &ir_type, &ir_len, &ir_data, &ir_buf, &ir_offset));
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
      NAPI_CALL(env, napi_get_typedarray_info(
        env, argv[1], &wa_type, &weights_len, &wa_data, &wa_buf, &wa_offset));
      (void)wa_type;
      weights_data = (const uint8_t *)wa_data;
    }
  }

  PolyInstance *inst = poly_instance_from_ir(
    (const uint8_t *)ir_data, (int)ir_len,
    weights_data, (int)weights_len);

  if (!inst) {
    napi_value result;
    napi_get_null(env, &result);
    return result;
  }
  return make_external(env, inst);
}

static napi_value napi_poly_instance_free(napi_env env, napi_callback_info info) {
  napi_value argv[1];
  size_t argc = 1;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyInstance *inst = get_external(env, argv[0]);
  poly_instance_free(inst);
  napi_value undef;
  napi_get_undefined(env, &undef);
  return undef;
}

static napi_value napi_poly_instance_set_device(napi_env env, napi_callback_info info) {
  napi_value argv[2];
  size_t argc = 2;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyInstance *inst = get_external(env, argv[0]);
  int32_t device;
  napi_get_value_int32(env, argv[1], &device);
  napi_value result;
  NAPI_CALL(env, napi_create_int32(env, poly_instance_set_device(inst, device), &result));
  return result;
}

static napi_value napi_poly_mlp_from_json(napi_env env, napi_callback_info info) {
  napi_value argv[1];
  size_t argc = 1;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  size_t spec_len = 0;
  char *spec = read_utf8_arg(env, argv[0], &spec_len);
  if (!spec) return NULL;
  PolyInstance *inst = poly_mlp_from_json(spec, (int)spec_len);
  free(spec);
  if (!inst) {
    napi_value result;
    napi_get_null(env, &result);
    return result;
  }
  return make_external(env, inst);
}

static napi_value napi_poly_tabm_instance(napi_env env, napi_callback_info info) {
  napi_value argv[1];
  size_t argc = 1;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  size_t spec_len = 0;
  char *spec = read_utf8_arg(env, argv[0], &spec_len);
  if (!spec) return NULL;
  PolyInstance *inst = poly_tabm_instance(spec, (int)spec_len);
  free(spec);
  if (!inst) {
    napi_value result;
    napi_get_null(env, &result);
    return result;
  }
  return make_external(env, inst);
}

static napi_value napi_poly_nam_instance(napi_env env, napi_callback_info info) {
  napi_value argv[1];
  size_t argc = 1;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  size_t spec_len = 0;
  char *spec = read_utf8_arg(env, argv[0], &spec_len);
  if (!spec) return NULL;
  PolyInstance *inst = poly_nam_instance(spec, (int)spec_len);
  free(spec);
  if (!inst) {
    napi_value result;
    napi_get_null(env, &result);
    return result;
  }
  return make_external(env, inst);
}

static napi_value napi_poly_instance_param_count(napi_env env, napi_callback_info info) {
  napi_value argv[1];
  size_t argc = 1;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyInstance *inst = get_external(env, argv[0]);
  napi_value result;
  NAPI_CALL(env, napi_create_int32(env, poly_instance_param_count(inst), &result));
  return result;
}

static napi_value napi_poly_instance_param_name(napi_env env, napi_callback_info info) {
  napi_value argv[2];
  size_t argc = 2;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyInstance *inst = get_external(env, argv[0]);
  int32_t i;
  napi_get_value_int32(env, argv[1], &i);
  const char *name = poly_instance_param_name(inst, i);
  napi_value result;
  if (name) {
    NAPI_CALL(env, napi_create_string_utf8(env, name, strlen(name), &result));
  } else {
    napi_get_null(env, &result);
  }
  return result;
}

static napi_value napi_poly_instance_param_shape(napi_env env, napi_callback_info info) {
  napi_value argv[2];
  size_t argc = 2;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyInstance *inst = get_external(env, argv[0]);
  int32_t i;
  napi_get_value_int32(env, argv[1], &i);
  int64_t shape[8];
  int ndim = poly_instance_param_shape(inst, i, shape, 8);
  napi_value result;
  NAPI_CALL(env, napi_create_array_with_length(env, (size_t)(ndim > 0 ? ndim : 0), &result));
  for (int j = 0; j < ndim; j++) {
    napi_value v;
    NAPI_CALL(env, napi_create_int64(env, shape[j], &v));
    NAPI_CALL(env, napi_set_element(env, result, (uint32_t)j, v));
  }
  return result;
}

static napi_value napi_poly_instance_param_data(napi_env env, napi_callback_info info) {
  napi_value argv[2];
  size_t argc = 2;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyInstance *inst = get_external(env, argv[0]);
  int32_t i;
  napi_get_value_int32(env, argv[1], &i);
  int64_t numel = 0;
  float *data = poly_instance_param_data(inst, i, &numel);
  return make_float32_array_copy(env, data, (size_t)(numel > 0 ? numel : 0));
}

static napi_value napi_poly_instance_buf_count(napi_env env, napi_callback_info info) {
  napi_value argv[1];
  size_t argc = 1;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyInstance *inst = get_external(env, argv[0]);
  napi_value result;
  NAPI_CALL(env, napi_create_int32(env, poly_instance_buf_count(inst), &result));
  return result;
}

static napi_value napi_poly_instance_buf_name(napi_env env, napi_callback_info info) {
  napi_value argv[2];
  size_t argc = 2;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyInstance *inst = get_external(env, argv[0]);
  int32_t i;
  napi_get_value_int32(env, argv[1], &i);
  const char *name = poly_instance_buf_name(inst, i);
  napi_value result;
  if (name) {
    NAPI_CALL(env, napi_create_string_utf8(env, name, strlen(name), &result));
  } else {
    napi_get_null(env, &result);
  }
  return result;
}

static napi_value napi_poly_instance_buf_role(napi_env env, napi_callback_info info) {
  napi_value argv[2];
  size_t argc = 2;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyInstance *inst = get_external(env, argv[0]);
  int32_t i;
  napi_get_value_int32(env, argv[1], &i);
  napi_value result;
  NAPI_CALL(env, napi_create_int32(env, poly_instance_buf_role(inst, i), &result));
  return result;
}

static napi_value napi_poly_instance_buf_shape(napi_env env, napi_callback_info info) {
  napi_value argv[2];
  size_t argc = 2;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyInstance *inst = get_external(env, argv[0]);
  int32_t i;
  napi_get_value_int32(env, argv[1], &i);
  int64_t shape[8];
  int ndim = poly_instance_buf_shape(inst, i, shape, 8);
  napi_value result;
  NAPI_CALL(env, napi_create_array_with_length(env, (size_t)(ndim > 0 ? ndim : 0), &result));
  for (int j = 0; j < ndim; j++) {
    napi_value v;
    NAPI_CALL(env, napi_create_int64(env, shape[j], &v));
    NAPI_CALL(env, napi_set_element(env, result, (uint32_t)j, v));
  }
  return result;
}

static napi_value napi_poly_instance_buf_data(napi_env env, napi_callback_info info) {
  napi_value argv[2];
  size_t argc = 2;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyInstance *inst = get_external(env, argv[0]);
  int32_t i;
  napi_get_value_int32(env, argv[1], &i);
  int64_t numel = 0;
  float *data = poly_instance_buf_data(inst, i, &numel);
  return make_float32_array_copy(env, data, (size_t)(numel > 0 ? numel : 0));
}

static napi_value napi_poly_instance_export_weights(napi_env env, napi_callback_info info) {
  napi_value argv[1];
  size_t argc = 1;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyInstance *inst = get_external(env, argv[0]);
  int out_len = 0;
  uint8_t *bytes = poly_instance_export_weights(inst, &out_len);
  napi_value result = make_uint8_array_copy(env, bytes, (size_t)(out_len > 0 ? out_len : 0));
  free(bytes);
  return result;
}

static napi_value napi_poly_instance_import_weights(napi_env env, napi_callback_info info) {
  napi_value argv[2];
  size_t argc = 2;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyInstance *inst = get_external(env, argv[0]);
  napi_typedarray_type type;
  size_t len = 0;
  void *data = NULL;
  size_t offset;
  napi_value arraybuf;
  NAPI_CALL(env, napi_get_typedarray_info(
    env, argv[1], &type, &len, &data, &arraybuf, &offset));
  (void)type;
  napi_value result;
  NAPI_CALL(env, napi_create_int32(
    env,
    poly_instance_import_weights(inst, (const uint8_t *)data, (int)len),
    &result));
  return result;
}

static napi_value napi_poly_instance_export_ir(napi_env env, napi_callback_info info) {
  napi_value argv[1];
  size_t argc = 1;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyInstance *inst = get_external(env, argv[0]);
  int out_len = 0;
  uint8_t *bytes = poly_instance_export_ir(inst, &out_len);
  napi_value result = make_uint8_array_copy(env, bytes, (size_t)(out_len > 0 ? out_len : 0));
  free(bytes);
  return result;
}

static napi_value napi_poly_instance_save_bundle(napi_env env, napi_callback_info info) {
  napi_value argv[1];
  size_t argc = 1;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyInstance *inst = get_external(env, argv[0]);
  int out_len = 0;
  uint8_t *bytes = poly_instance_save_bundle(inst, &out_len);
  napi_value result = make_uint8_array_copy(env, bytes, (size_t)(out_len > 0 ? out_len : 0));
  free(bytes);
  return result;
}

static napi_value napi_poly_instance_from_bundle(napi_env env, napi_callback_info info) {
  napi_value argv[1];
  size_t argc = 1;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  uint8_t *data = NULL;
  size_t len = 0;
  napi_typedarray_type type;
  napi_value arraybuf;
  size_t offset;
  NAPI_CALL(env, napi_get_typedarray_info(env, argv[0], &type, &len, (void **)&data, &arraybuf, &offset));
  PolyInstance *inst = poly_instance_from_bundle(data, (int)len);
  if (!inst) {
    napi_value undef;
    napi_get_undefined(env, &undef);
    return undef;
  }
  return make_external(env, inst);
}

static napi_value napi_poly_instance_set_optimizer(napi_env env, napi_callback_info info) {
  napi_value argv[7];
  size_t argc = 7;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyInstance *inst = get_external(env, argv[0]);
  int32_t kind;
  double lr, beta1, beta2, eps, weight_decay;
  napi_get_value_int32(env, argv[1], &kind);
  napi_get_value_double(env, argv[2], &lr);
  napi_get_value_double(env, argv[3], &beta1);
  napi_get_value_double(env, argv[4], &beta2);
  napi_get_value_double(env, argv[5], &eps);
  napi_get_value_double(env, argv[6], &weight_decay);
  napi_value result;
  NAPI_CALL(env, napi_create_int32(
    env,
    poly_instance_set_optimizer(
      inst, kind, (float)lr, (float)beta1, (float)beta2, (float)eps, (float)weight_decay),
    &result));
  return result;
}

static napi_value napi_poly_instance_forward(napi_env env, napi_callback_info info) {
  napi_value argv[3];
  size_t argc = 3;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyInstance *inst = get_external(env, argv[0]);

  PolyIOBinding *bindings = NULL;
  char **names = NULL;
  int n = 0;
  if (!read_io_bindings(env, argv[1], argv[2], &bindings, &names, &n)) {
    return NULL;
  }

  int rc = poly_instance_forward(inst, bindings, n);
  free_io_bindings(names, bindings, n);

  napi_value result;
  NAPI_CALL(env, napi_create_int32(env, rc, &result));
  return result;
}

static napi_value napi_poly_instance_train_step(napi_env env, napi_callback_info info) {
  napi_value argv[3];
  size_t argc = 3;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyInstance *inst = get_external(env, argv[0]);

  PolyIOBinding *bindings = NULL;
  char **names = NULL;
  int n = 0;
  if (!read_io_bindings(env, argv[1], argv[2], &bindings, &names, &n)) {
    return NULL;
  }

  float loss = 0.0f;
  int rc = poly_instance_train_step(inst, bindings, n, &loss);
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
  napi_value argv[2]; size_t argc = 2;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *u = get_external(env, argv[1]);
  napi_value result;
  napi_create_int32(env, poly_uop_ndim(ctx, u), &result);
  return result;
}

static napi_value napi_poly_uop_dims(napi_env env, napi_callback_info info) {
  napi_value argv[2]; size_t argc = 2;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *u = get_external(env, argv[1]);
  int ndim = poly_uop_ndim(ctx, u);
  const int64_t *dims = poly_uop_dims(ctx, u);
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
  napi_value argv[3]; size_t argc = 3;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *x = get_external(env, argv[1]);
  int32_t axis; napi_get_value_int32(env, argv[2], &axis);
  return make_external(env, poly_softmax(ctx, x, axis));
}

static napi_value napi_poly_log_softmax(napi_env env, napi_callback_info info) {
  napi_value argv[3]; size_t argc = 3;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *x = get_external(env, argv[1]);
  int32_t axis; napi_get_value_int32(env, argv[2], &axis);
  return make_external(env, poly_log_softmax(ctx, x, axis));
}

static napi_value napi_poly_dot(napi_env env, napi_callback_info info) {
  napi_value argv[3]; size_t argc = 3;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *x = get_external(env, argv[1]);
  PolyUOp *w = get_external(env, argv[2]);
  return make_external(env, poly_dot(ctx, x, w));
}

static napi_value napi_poly_layernorm_v2(napi_env env, napi_callback_info info) {
  napi_value argv[4]; size_t argc = 4;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *x = get_external(env, argv[1]);
  int32_t axis; napi_get_value_int32(env, argv[2], &axis);
  double eps; napi_get_value_double(env, argv[3], &eps);
  return make_external(env, poly_layernorm_v2(ctx, x, axis, eps));
}

static napi_value napi_poly_cross_entropy(napi_env env, napi_callback_info info) {
  napi_value argv[4]; size_t argc = 4;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *logits = get_external(env, argv[1]);
  PolyUOp *target = get_external(env, argv[2]);
  int32_t axis; napi_get_value_int32(env, argv[3], &axis);
  return make_external(env, poly_cross_entropy(ctx, logits, target, axis));
}

static napi_value napi_poly_linear_v2(napi_env env, napi_callback_info info) {
  napi_value argv[4]; size_t argc = 4;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *x = get_external(env, argv[1]);
  PolyUOp *w = get_external(env, argv[2]);
  PolyUOp *bias = get_external(env, argv[3]); /* NULL if js passes null/undefined */
  return make_external(env, poly_linear_v2(ctx, x, w, bias));
}

static napi_value napi_poly_gather(napi_env env, napi_callback_info info) {
  napi_value argv[3]; size_t argc = 3;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *table = get_external(env, argv[1]);
  PolyUOp *indices = get_external(env, argv[2]);
  return make_external(env, poly_gather(ctx, table, indices));
}

static napi_value napi_poly_sum_reduce(napi_env env, napi_callback_info info) {
  napi_value argv[4]; size_t argc = 4;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *x = get_external(env, argv[1]);
  int32_t axis, keepdim;
  napi_get_value_int32(env, argv[2], &axis);
  napi_get_value_int32(env, argv[3], &keepdim);
  return make_external(env, poly_sum_reduce(ctx, x, axis, keepdim));
}

static napi_value napi_poly_max_reduce(napi_env env, napi_callback_info info) {
  napi_value argv[4]; size_t argc = 4;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *x = get_external(env, argv[1]);
  int32_t axis, keepdim;
  napi_get_value_int32(env, argv[2], &axis);
  napi_get_value_int32(env, argv[3], &keepdim);
  return make_external(env, poly_max_reduce(ctx, x, axis, keepdim));
}

static napi_value napi_poly_mean_reduce(napi_env env, napi_callback_info info) {
  napi_value argv[4]; size_t argc = 4;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *x = get_external(env, argv[1]);
  int32_t axis, keepdim;
  napi_get_value_int32(env, argv[2], &axis);
  napi_get_value_int32(env, argv[3], &keepdim);
  return make_external(env, poly_mean_reduce(ctx, x, axis, keepdim));
}

static napi_value napi_poly_var_reduce(napi_env env, napi_callback_info info) {
  napi_value argv[5]; size_t argc = 5;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  PolyCtx *ctx = get_external(env, argv[0]);
  PolyUOp *x = get_external(env, argv[1]);
  int32_t axis, keepdim, correction;
  napi_get_value_int32(env, argv[2], &axis);
  napi_get_value_int32(env, argv[3], &keepdim);
  napi_get_value_int32(env, argv[4], &correction);
  return make_external(env, poly_var_reduce(ctx, x, axis, keepdim, correction));
}

/* ── Module registration ───────────────────────────────────────────────── */

#define DECLARE_NAPI_METHOD(name, fn) \
  { (name), NULL, (fn), NULL, NULL, NULL, napi_default, NULL }

/* ── Tokenizer N-API wrappers ──────────────────────────────────────────── */

static napi_value napi_poly_tokenizer_from_json(napi_env env, napi_callback_info info) {
  napi_value argv[2]; size_t argc = 2;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  void *data; size_t len;
  NAPI_CALL(env, napi_get_buffer_info(env, argv[0], &data, &len));
  PolyTokenizer *tok = poly_tokenizer_from_json((const char *)data, (int)len);
  if (!tok) { napi_value n; napi_get_null(env, &n); return n; }
  return make_external(env, tok);
}

static napi_value napi_poly_tokenize(napi_env env, napi_callback_info info) {
  napi_value argv[2]; size_t argc = 2;
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
  napi_value argv[2]; size_t argc = 2;
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
  napi_value argv[1]; size_t argc = 1;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  poly_tokenizer_free(get_external(env, argv[0]));
  napi_value undef;
  napi_get_undefined(env, &undef);
  return undef;
}

static napi_value napi_poly_tokenizer_vocab_size(napi_env env, napi_callback_info info) {
  napi_value argv[1]; size_t argc = 1;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  napi_value r;
  NAPI_CALL(env, napi_create_int32(env, poly_tokenizer_vocab_size(get_external(env, argv[0])), &r));
  return r;
}

static napi_value napi_poly_tokenizer_bos_id(napi_env env, napi_callback_info info) {
  napi_value argv[1]; size_t argc = 1;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  napi_value r;
  NAPI_CALL(env, napi_create_int32(env, poly_tokenizer_bos_id(get_external(env, argv[0])), &r));
  return r;
}

static napi_value napi_poly_tokenizer_eos_id(napi_env env, napi_callback_info info) {
  napi_value argv[1]; size_t argc = 1;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  napi_value r;
  NAPI_CALL(env, napi_create_int32(env, poly_tokenizer_eos_id(get_external(env, argv[0])), &r));
  return r;
}

/* ── HF / GGUF loaders ─────────────────────────────────────────────────── */

extern PolyInstance *poly_hf_load(
    const char *config_json, int config_len,
    const uint8_t **weight_files, const int64_t *weight_lens,
    int n_weight_files, int max_batch, int max_seq_len);

extern PolyInstance *poly_gguf_load(
    const uint8_t *data, int64_t len, int max_batch, int max_seq_len);

static napi_value napi_poly_hf_load(napi_env env, napi_callback_info info) {
  napi_value argv[4]; size_t argc = 4;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));

  /* argv[0] = config Buffer, argv[1] = weight Buffers array, argv[2] = maxBatch, argv[3] = maxSeqLen */
  void *cfg_data; size_t cfg_len;
  NAPI_CALL(env, napi_get_buffer_info(env, argv[0], &cfg_data, &cfg_len));

  uint32_t n_files;
  NAPI_CALL(env, napi_get_array_length(env, argv[1], &n_files));

  const uint8_t **file_ptrs = malloc(n_files * sizeof(uint8_t *));
  int64_t *file_lens = malloc(n_files * sizeof(int64_t));
  for (uint32_t i = 0; i < n_files; i++) {
    napi_value el;
    NAPI_CALL(env, napi_get_element(env, argv[1], i, &el));
    void *fdata; size_t flen;
    NAPI_CALL(env, napi_get_buffer_info(env, el, &fdata, &flen));
    file_ptrs[i] = (const uint8_t *)fdata;
    file_lens[i] = (int64_t)flen;
  }

  int32_t max_batch, max_seq_len;
  NAPI_CALL(env, napi_get_value_int32(env, argv[2], &max_batch));
  NAPI_CALL(env, napi_get_value_int32(env, argv[3], &max_seq_len));

  PolyInstance *inst = poly_hf_load(
      (const char *)cfg_data, (int)cfg_len,
      file_ptrs, file_lens, (int)n_files,
      max_batch, max_seq_len);

  free(file_ptrs);
  free(file_lens);

  if (!inst) { napi_value n; napi_get_null(env, &n); return n; }
  return make_external(env, inst);
}

static napi_value napi_poly_gguf_load(napi_env env, napi_callback_info info) {
  napi_value argv[3]; size_t argc = 3;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  void *data; size_t len;
  NAPI_CALL(env, napi_get_buffer_info(env, argv[0], &data, &len));
  int32_t max_batch, max_seq_len;
  NAPI_CALL(env, napi_get_value_int32(env, argv[1], &max_batch));
  NAPI_CALL(env, napi_get_value_int32(env, argv[2], &max_seq_len));
  PolyInstance *inst = poly_gguf_load((const uint8_t *)data, (int64_t)len,
                                      max_batch, max_seq_len);
  if (!inst) { napi_value n; napi_get_null(env, &n); return n; }
  return make_external(env, inst);
}

static napi_value napi_poly_tokenizer_from_gguf(napi_env env, napi_callback_info info) {
  napi_value argv[1]; size_t argc = 1;
  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  void *data; size_t len;
  NAPI_CALL(env, napi_get_buffer_info(env, argv[0], &data, &len));
  PolyGgufDecoded *gguf = NULL;
  if (poly_gguf_decode((const uint8_t *)data, (int64_t)len, &gguf) != 0 || !gguf) {
    napi_value n; napi_get_null(env, &n); return n;
  }
  PolyTokenizer *tok = poly_tokenizer_from_gguf(gguf);
  poly_gguf_decoded_free(gguf);
  if (!tok) { napi_value n; napi_get_null(env, &n); return n; }
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

    /* Constants */
    DECLARE_NAPI_METHOD("poly_const_float", napi_poly_const_float),
    DECLARE_NAPI_METHOD("poly_const_double", napi_poly_const_double),
    DECLARE_NAPI_METHOD("poly_const_int", napi_poly_const_int),

    /* ALU */
    DECLARE_NAPI_METHOD("poly_contiguous", napi_poly_contiguous),
    DECLARE_NAPI_METHOD("poly_alu1", napi_poly_alu1),
    DECLARE_NAPI_METHOD("poly_alu2", napi_poly_alu2),
    DECLARE_NAPI_METHOD("poly_alu3", napi_poly_alu3),

    /* Graph construction */
    DECLARE_NAPI_METHOD("poly_store_val", napi_poly_store_val),
    DECLARE_NAPI_METHOD("poly_sink1", napi_poly_sink1),
    DECLARE_NAPI_METHOD("poly_sink_n", napi_poly_sink_n),
    DECLARE_NAPI_METHOD("poly_assign", napi_poly_assign),

    /* Buffers */
    DECLARE_NAPI_METHOD("poly_buffer_f32", napi_poly_buffer_f32),
    DECLARE_NAPI_METHOD("poly_buffer_f64", napi_poly_buffer_f64),
    DECLARE_NAPI_METHOD("poly_buffer_from_host", napi_poly_buffer_from_host),
    DECLARE_NAPI_METHOD("poly_buffer_get_ptr", napi_poly_buffer_get_ptr),
    DECLARE_NAPI_METHOD("poly_buffer_get_key", napi_poly_buffer_get_key),
    DECLARE_NAPI_METHOD("poly_set_frontend_buffer_release", napi_poly_set_frontend_buffer_release),
    DECLARE_NAPI_METHOD("poly_uop_has_buffer_identity", napi_poly_uop_has_buffer_identity),
    DECLARE_NAPI_METHOD("poly_uop_get_buffer_identity", napi_poly_uop_get_buffer_identity),
    DECLARE_NAPI_METHOD("poly_realize", napi_poly_realize),
    DECLARE_NAPI_METHOD("poly_buffer_read", napi_poly_buffer_read),
    DECLARE_NAPI_METHOD("poly_ctx_set_preferred_device", napi_poly_ctx_set_preferred_device),

    /* Autograd */
    DECLARE_NAPI_METHOD("poly_grad", napi_poly_grad),
    DECLARE_NAPI_METHOD("poly_detach", napi_poly_detach),
    DECLARE_NAPI_METHOD("poly_cast_by_id", napi_poly_cast_by_id),
    DECLARE_NAPI_METHOD("poly_dtype_id_by_name", napi_poly_dtype_id_by_name),

    /* Movement ops (shape-taking) */
    DECLARE_NAPI_METHOD("poly_reshape", napi_poly_reshape),
    DECLARE_NAPI_METHOD("poly_expand", napi_poly_expand),
    DECLARE_NAPI_METHOD("poly_permute", napi_poly_permute),
    DECLARE_NAPI_METHOD("poly_flip", napi_poly_flip),
    DECLARE_NAPI_METHOD("poly_shrink", napi_poly_shrink),
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

    /* Realize (stateful builder) */
    DECLARE_NAPI_METHOD("poly_realize_begin", napi_poly_realize_begin),
    DECLARE_NAPI_METHOD("poly_realize_bind", napi_poly_realize_bind),
    DECLARE_NAPI_METHOD("poly_realize_exec", napi_poly_realize_exec),

    /* Metadata */
    DECLARE_NAPI_METHOD("poly_op_count", napi_poly_op_count),
    DECLARE_NAPI_METHOD("poly_op_name", napi_poly_op_name),
    DECLARE_NAPI_METHOD("poly_abi_version", napi_poly_abi_version),

    /* Cache cleanup */
    DECLARE_NAPI_METHOD("poly_cpu_cache_flush", napi_poly_cpu_cache_flush),
    DECLARE_NAPI_METHOD("poly_sched_cache_flush", napi_poly_sched_cache_flush),

    /* PolyInstance / model runtime */
    DECLARE_NAPI_METHOD("poly_instance_from_ir", napi_poly_instance_from_ir),
    DECLARE_NAPI_METHOD("poly_instance_free", napi_poly_instance_free),
    DECLARE_NAPI_METHOD("poly_instance_set_device", napi_poly_instance_set_device),
    DECLARE_NAPI_METHOD("poly_mlp_from_json", napi_poly_mlp_from_json),
    DECLARE_NAPI_METHOD("poly_tabm_instance", napi_poly_tabm_instance),
    DECLARE_NAPI_METHOD("poly_nam_instance", napi_poly_nam_instance),
    DECLARE_NAPI_METHOD("poly_instance_param_count", napi_poly_instance_param_count),
    DECLARE_NAPI_METHOD("poly_instance_param_name", napi_poly_instance_param_name),
    DECLARE_NAPI_METHOD("poly_instance_param_shape", napi_poly_instance_param_shape),
    DECLARE_NAPI_METHOD("poly_instance_param_data", napi_poly_instance_param_data),
    DECLARE_NAPI_METHOD("poly_instance_buf_count", napi_poly_instance_buf_count),
    DECLARE_NAPI_METHOD("poly_instance_buf_name", napi_poly_instance_buf_name),
    DECLARE_NAPI_METHOD("poly_instance_buf_role", napi_poly_instance_buf_role),
    DECLARE_NAPI_METHOD("poly_instance_buf_shape", napi_poly_instance_buf_shape),
    DECLARE_NAPI_METHOD("poly_instance_buf_data", napi_poly_instance_buf_data),
    DECLARE_NAPI_METHOD("poly_instance_export_weights", napi_poly_instance_export_weights),
    DECLARE_NAPI_METHOD("poly_instance_import_weights", napi_poly_instance_import_weights),
    DECLARE_NAPI_METHOD("poly_instance_export_ir", napi_poly_instance_export_ir),
    DECLARE_NAPI_METHOD("poly_instance_save_bundle", napi_poly_instance_save_bundle),
    DECLARE_NAPI_METHOD("poly_instance_from_bundle", napi_poly_instance_from_bundle),
    DECLARE_NAPI_METHOD("poly_instance_set_optimizer", napi_poly_instance_set_optimizer),
    DECLARE_NAPI_METHOD("poly_instance_forward", napi_poly_instance_forward),
    DECLARE_NAPI_METHOD("poly_instance_train_step", napi_poly_instance_train_step),
    /* Shape-on-UOp accessors */
    DECLARE_NAPI_METHOD("poly_uop_ndim", napi_poly_uop_ndim),
    DECLARE_NAPI_METHOD("poly_uop_dims", napi_poly_uop_dims),
    /* v2 composed ops */
    DECLARE_NAPI_METHOD("poly_softmax", napi_poly_softmax),
    DECLARE_NAPI_METHOD("poly_log_softmax", napi_poly_log_softmax),
    DECLARE_NAPI_METHOD("poly_dot", napi_poly_dot),
    DECLARE_NAPI_METHOD("poly_layernorm_v2", napi_poly_layernorm_v2),
    DECLARE_NAPI_METHOD("poly_cross_entropy", napi_poly_cross_entropy),
    DECLARE_NAPI_METHOD("poly_linear_v2", napi_poly_linear_v2),
    DECLARE_NAPI_METHOD("poly_gather", napi_poly_gather),
    DECLARE_NAPI_METHOD("poly_sum_reduce", napi_poly_sum_reduce),
    DECLARE_NAPI_METHOD("poly_max_reduce", napi_poly_max_reduce),
    DECLARE_NAPI_METHOD("poly_mean_reduce", napi_poly_mean_reduce),
    DECLARE_NAPI_METHOD("poly_var_reduce", napi_poly_var_reduce),

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

  NAPI_CALL(env, napi_define_properties(
    env, exports, sizeof(props) / sizeof(props[0]), props));

  return exports;
}
