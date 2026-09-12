/*
 * frontend.c — FFI-friendly helpers for language bindings
 *
 * Thin wrappers around the core API that avoid passing PolyArg/PolyDType
 * across FFI boundaries. Execution uses graph/schedule entrypoints backed by
 * ctx->buffers; there is no separate runtime buffer-binding graph.
 */

#define _GNU_SOURCE
#include "frontend.h"
#include "bigint.h"
#include "ctx.h"
#include "engine/realize.h"
#include "engine/schedule.h"
#include "schedule/rangeify.h"
#include "codegen/codegen.h"
#include "tensor.h"
#include "interp.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "utils.h"

/* Current Tinygrad UOp.new_buffer always records a concrete device.  FFI
 * constructors without a device argument use the context/default device. */
static PolyDevice frontend_buffer_device(PolyCtx *ctx) {
  PolyDevice device = poly_ctx_get_preferred_device(ctx);
  return poly_device_can_execute(device) ? device : poly_device_default();
}

/* C argument adaptation for current Tinygrad UOp.new_buffer. */
static PolyUOp *frontend_new_buffer(
    PolyCtx *ctx,
    PolyDType dtype,
    int64_t size,
    PolyDevice device
) {
  if (device == POLY_DEVICE_AUTO) device = frontend_buffer_device(ctx);
  PolyUOp *device_uop = poly_device_uop(ctx, device);
  return device_uop
             ? poly_uop_new_buffer(ctx, device_uop, size, dtype, poly_ctx_next_unique_id(ctx))
             : NULL;
}

PolyUOp *poly_buffer_by_id(PolyCtx *ctx, int dtype_id, int64_t size) {
  PolyDType dt;
  if (!poly_dtype_by_id(dtype_id, &dt)) return NULL;
  return frontend_new_buffer(ctx, dt, size, POLY_DEVICE_AUTO);
}

PolyUOp *poly_buffer_on_device_by_id(PolyCtx *ctx, int dtype_id, int64_t size, int device_id) {
  PolyDType dt;
  if (!poly_dtype_by_id(dtype_id, &dt)) return NULL;
  PolyDevice device = (PolyDevice)device_id;
  if (!poly_device_can_execute(device)) return NULL;
  return frontend_new_buffer(ctx, dt, size, device);
}

PolyUOp *poly_buffer_f32(PolyCtx *ctx, int64_t size) {
  return frontend_new_buffer(ctx, POLY_FLOAT32, size, POLY_DEVICE_AUTO);
}

PolyUOp *poly_buffer_f64(PolyCtx *ctx, int64_t size) {
  return frontend_new_buffer(ctx, POLY_FLOAT64, size, POLY_DEVICE_AUTO);
}

PolyUOp *poly_const_int_decimal(PolyCtx *ctx, const char *value) {
  PolyInt integer = {0};
  if (!ctx || !value || !poly_int_from_decimal(&integer, value)) {
    poly_int_free(&integer);
    return NULL;
  }
  PolyUOp *out = poly_uop(ctx, POLY_OP_CONST, POLY_WEAKINT, NULL, 0, poly_int_as_arg(&integer));
  poly_int_free(&integer);
  return out;
}

PolyUOp *poly_uop_variable_by_id(
    PolyCtx *ctx,
    const char *name,
    PolyUOp *min_val,
    PolyUOp *max_val,
    int dtype_id,
    int64_t multiple_of,
    bool param
) {
  PolyDType dtype;
  if (!min_val || !max_val || min_val->op != POLY_OP_CONST || max_val->op != POLY_OP_CONST ||
      !poly_dtype_by_id(dtype_id, &dtype))
    return NULL;
  return poly_uop_variable(ctx, name, min_val->arg, max_val->arg, dtype, multiple_of, param);
}

PolyTensor *poly_tensor_empty_by_id(
    PolyCtx *ctx,
    int dtype_id,
    const int64_t *dims,
    int ndim,
    int device_id
) {
  PolyDType dt;
  if (!poly_dtype_by_id(dtype_id, &dt)) return NULL;
  return poly_tensor_empty(ctx, dt, dims, ndim, (PolyDevice)device_id);
}

PolyTensor *poly_tensor_empty_uop_by_id(
    PolyCtx *ctx,
    int dtype_id,
    PolyUOp **dims,
    int ndim,
    int device_id
) {
  PolyDType dt;
  if (!poly_dtype_by_id(dtype_id, &dt)) return NULL;
  return poly_tensor_empty_uop(ctx, dt, dims, ndim, (PolyDevice)device_id);
}

PolyTensor *poly_tensor_from_host_by_id(
    PolyCtx *ctx,
    void *ptr,
    size_t nbytes,
    int dtype_id,
    const int64_t *dims,
    int ndim
) {
  PolyDType dt;
  if (!poly_dtype_by_id(dtype_id, &dt)) return NULL;
  return poly_tensor_from_host(ctx, ptr, nbytes, dt, dims, ndim);
}

PolyTensor *poly_tensor_const_int_by_id(PolyCtx *ctx, int64_t value, int dtype_id, int device_id) {
  PolyUOp *value_uop = poly_const_int_by_id(ctx, value, dtype_id);
  if (!value_uop) return NULL;
  return poly_tensor_create_with_roots(
      ctx, value_uop, value_uop, POLY_TENSOR_VALUE, (PolyDevice)device_id
  );
}

PolyTensor *poly_tensor_const_uint_by_id(
    PolyCtx *ctx,
    uint64_t value,
    int dtype_id,
    int device_id
) {
  PolyUOp *uop = poly_const_uint_by_id(ctx, value, dtype_id);
  return uop ? poly_tensor_create_with_roots(
                   ctx, uop, uop, POLY_TENSOR_VALUE, (PolyDevice)device_id
               )
             : NULL;
}

PolyTensor *poly_tensor_const_float_by_id(PolyCtx *ctx, double value, int dtype_id, int device_id) {
  PolyUOp *value_uop = poly_const_float_by_id(ctx, value, dtype_id);
  if (!value_uop) return NULL;
  return poly_tensor_create_with_roots(
      ctx, value_uop, value_uop, POLY_TENSOR_VALUE, (PolyDevice)device_id
  );
}

PolyTensor *poly_tensor_full_int_by_id(
    PolyCtx *ctx,
    const int64_t *dims,
    int ndim,
    int64_t value,
    int dtype_id,
    int device_id,
    bool dtype_explicit,
    bool buffer
) {
  PolyUOp *value_uop = poly_full_int_by_id(ctx, dims, ndim, value, dtype_id);
  return poly_tensor_full_from_value(
      ctx, value_uop, dims, ndim, (PolyDevice)device_id, value_uop ? value_uop->dtype : POLY_VOID,
      dtype_explicit, buffer
  );
}

PolyTensor *poly_tensor_full_uint_by_id(
    PolyCtx *ctx,
    const int64_t *dims,
    int ndim,
    uint64_t value,
    int dtype_id,
    int device_id,
    bool dtype_explicit,
    bool buffer
) {
  PolyUOp *uop = poly_full_uint_by_id(ctx, dims, ndim, value, dtype_id);
  return poly_tensor_full_from_value(
      ctx, uop, dims, ndim, (PolyDevice)device_id, uop ? uop->dtype : POLY_VOID, dtype_explicit,
      buffer
  );
}

PolyTensor *poly_tensor_full_invalid_by_id(
    PolyCtx *ctx,
    const int64_t *dims,
    int ndim,
    int dtype_id,
    int device_id,
    bool buffer
) {
  PolyDType dtype;
  if (!poly_dtype_by_id(dtype_id, &dtype)) return NULL;
  PolyUOp *value = poly_full_invalid_by_id(ctx, dims, ndim, dtype_id);
  return poly_tensor_full_from_value(
      ctx, value, dims, ndim, (PolyDevice)device_id, dtype, true, buffer
  );
}

PolyTensor *poly_tensor_full_float_by_id(
    PolyCtx *ctx,
    const int64_t *dims,
    int ndim,
    double value,
    int dtype_id,
    int device_id,
    bool dtype_explicit,
    bool buffer
) {
  PolyUOp *value_uop = poly_full_float_by_id(ctx, dims, ndim, value, dtype_id);
  return poly_tensor_full_from_value(
      ctx, value_uop, dims, ndim, (PolyDevice)device_id, value_uop ? value_uop->dtype : POLY_VOID,
      dtype_explicit, buffer
  );
}

PolyTensor *poly_tensor_arange_int_by_id(
    PolyCtx *ctx,
    int64_t start,
    int64_t stop,
    int64_t step,
    int dtype_id,
    int device_id
) {
  PolyUOp *value_uop = poly_arange_int_by_id(ctx, start, stop, step, dtype_id);
  if (!value_uop) return NULL;
  return poly_tensor_create_with_roots(
      ctx, value_uop, value_uop, POLY_TENSOR_VALUE, (PolyDevice)device_id
  );
}

PolyTensor *poly_tensor_arange_float_by_id(
    PolyCtx *ctx,
    double start,
    double stop,
    double step,
    int dtype_id,
    int device_id
) {
  PolyUOp *value_uop = poly_arange_float_by_id(ctx, start, stop, step, dtype_id);
  if (!value_uop) return NULL;
  return poly_tensor_create_with_roots(
      ctx, value_uop, value_uop, POLY_TENSOR_VALUE, (PolyDevice)device_id
  );
}

PolyTensor *poly_tensor_linspace_by_id(
    PolyCtx *ctx,
    double start,
    double stop,
    int64_t steps,
    int dtype_id,
    int device_id
) {
  PolyUOp *value_uop = poly_linspace_by_id(ctx, start, stop, steps, dtype_id);
  if (!value_uop) return NULL;
  return poly_tensor_create_with_roots(
      ctx, value_uop, value_uop, POLY_TENSOR_VALUE, (PolyDevice)device_id
  );
}

PolyTensor *poly_tensor_eye_by_id(PolyCtx *ctx, int64_t n, int64_t m, int dtype_id, int device_id) {
  PolyUOp *value_uop = poly_eye_by_id(ctx, n, m, dtype_id);
  if (!value_uop) return NULL;
  return poly_tensor_create_with_roots(
      ctx, value_uop, value_uop, POLY_TENSOR_VALUE, (PolyDevice)device_id
  );
}

int poly_uop_op(PolyUOp *u) {
  return u ? (int)u->op : 0;
}

int poly_uop_dtype_id(PolyCtx *ctx, PolyUOp *u) {
  (void)ctx;
  if (!u) return 0;
  PolyDType sdt = u->dtype;
  for (int i = 0; i < poly_dtype_count(); i++) {
    PolyDType candidate;
    if (poly_dtype_by_id(i, &candidate) && poly_dtype_eq(sdt, candidate)) return i;
  }
  return -1;
}

int poly_uop_n_src(PolyUOp *u) {
  return u ? u->n_src : 0;
}

PolyUOp *poly_uop_src(PolyUOp *u, int idx) {
  if (!u || idx < 0 || idx >= u->n_src) return NULL;
  return u->src[idx];
}

uint32_t poly_uop_call_grad_fxn_key(PolyUOp *u) {
  if (!u || (u->op != POLY_OP_CALL && u->op != POLY_OP_FUNCTION) ||
      u->arg.kind != POLY_ARG_CALL_INFO || !u->arg.call_info)
    return 0;
  return u->arg.call_info->grad_fxn_key;
}

static bool canrun_shape_valid(const int64_t *shape, int ndim) {
  if (ndim < 0 || ndim > POLY_MAX_DIMS) return false;
  if (ndim > 0 && !shape) return false;
  for (int i = 0; i < ndim; i++)
    if (shape[i] < 0) return false;
  return true;
}

static bool canrun_shape_budget_ok(const int64_t *shape, int ndim, int64_t limit) {
  int64_t numel = poly_shape_numel_checked(shape, ndim);
  return numel >= 0 && numel <= limit;
}

static PolyUOp *canrun_buffer(PolyCtx *ctx, PolyDType dt, const int64_t *shape, int ndim) {
  if (!canrun_shape_valid(shape, ndim)) return NULL;
  int64_t numel = poly_shape_numel_checked(shape, ndim);
  if (numel < 0) return NULL;
  /* Tinygrad UOp.new_buffer records the selected device in ParamArg
   * (uop/ops.py:814-817); capability probes must schedule the same graph. */
  PolyUOp *buf = frontend_new_buffer(ctx, dt, numel, POLY_DEVICE_AUTO);
  if (!buf) return NULL;
  return poly_reshape(ctx, buf, (int64_t *)shape, ndim);
}

static PolyUOp *canrun_store_sink(PolyCtx *ctx, PolyUOp *value) {
  if (!ctx || !value) return NULL;
  int ndim = poly_uop_ndim(ctx, value);
  if (ndim < 0 || ndim > POLY_MAX_DIMS) return NULL;
  const int64_t *dims = poly_uop_max_shape_dims(ctx, value);
  if (ndim > 0 && !dims) return NULL;
  int64_t shape[POLY_MAX_DIMS];
  for (int i = 0; i < ndim; i++)
    shape[i] = dims[i];
  PolyUOp *target = canrun_buffer(ctx, value->dtype, shape, ndim);
  PolyUOp *store = target ? poly_store_val(ctx, target, value) : NULL;
  return store ? poly_sink1(ctx, store) : NULL;
}

static PolyUOp *canrun_build_probe_graph(
    PolyCtx *ctx,
    const char *op,
    PolyDType dt,
    const int64_t *shape,
    int ndim
) {
  if (!ctx || !op || !canrun_shape_valid(shape, ndim)) return NULL;

  if (!strcmp(op, "matmul") || !strcmp(op, "dot")) {
    if (ndim != 3 || shape[0] <= 0 || shape[1] <= 0 || shape[2] <= 0) return NULL;
    int64_t a_shape[2] = {shape[0], shape[1]};
    int64_t b_shape[2] = {shape[1], shape[2]};
    if (!canrun_shape_budget_ok(a_shape, 2, 4096) || !canrun_shape_budget_ok(b_shape, 2, 4096))
      return NULL;
    PolyUOp *a = canrun_buffer(ctx, dt, a_shape, 2);
    PolyUOp *b = canrun_buffer(ctx, dt, b_shape, 2);
    return (a && b) ? poly_dot(ctx, a, b) : NULL;
  }

  if (!strcmp(op, "triangular_solve") || !strcmp(op, "triangularSolve")) {
    if ((ndim != 2 && ndim != 3) || shape[0] <= 0 || shape[0] != shape[1] || shape[0] > 16)
      return NULL;
    int64_t a_shape[2] = {shape[0], shape[1]};
    int64_t b_shape[2] = {shape[0], (ndim == 3) ? shape[2] : 1};
    PolyUOp *a = canrun_buffer(ctx, dt, a_shape, 2);
    PolyUOp *b = canrun_buffer(ctx, dt, b_shape, 2);
    return (a && b) ? poly_triangular_solve(ctx, a, b, 0, 0, 0) : NULL;
  }

  if (!strcmp(op, "solve")) {
    if ((ndim != 2 && ndim != 3) || shape[0] <= 0 || shape[0] != shape[1] || shape[0] > 12)
      return NULL;
    int64_t a_shape[2] = {shape[0], shape[1]};
    int64_t b_shape[2] = {shape[0], (ndim == 3) ? shape[2] : 1};
    PolyUOp *a = canrun_buffer(ctx, dt, a_shape, 2);
    PolyUOp *b = canrun_buffer(ctx, dt, b_shape, 2);
    return (a && b) ? poly_solve(ctx, a, b) : NULL;
  }

  if (!strcmp(op, "lstsq")) {
    if ((ndim != 2 && ndim != 3) || shape[0] <= 0 || shape[1] <= 0 || shape[0] > 12 ||
        shape[1] > 12)
      return NULL;
    int64_t a_shape[2] = {shape[0], shape[1]};
    int64_t b_shape[2] = {shape[0], (ndim == 3) ? shape[2] : 1};
    PolyUOp *a = canrun_buffer(ctx, dt, a_shape, 2);
    PolyUOp *b = canrun_buffer(ctx, dt, b_shape, 2);
    return (a && b) ? poly_lstsq(ctx, a, b) : NULL;
  }

  if (!canrun_shape_budget_ok(shape, ndim, 8192)) return NULL;
  PolyUOp *a = canrun_buffer(ctx, dt, shape, ndim);
  if (!a) return NULL;

  if (!strcmp(op, "qr")) {
    if (ndim < 2 || shape[ndim - 2] > 16 || shape[ndim - 1] > 16) return NULL;
    PolyUOp *q = NULL, *r = NULL;
    return poly_qr_ex(ctx, a, POLY_QR_REDUCED, &q, &r) == 0 ? r : NULL;
  }
  if (!strcmp(op, "cholesky")) {
    if (ndim < 2 || shape[ndim - 2] <= 0 || shape[ndim - 2] != shape[ndim - 1] ||
        shape[ndim - 1] > 16)
      return NULL;
    return poly_cholesky(ctx, a, 0);
  }
  if (!strcmp(op, "gather") || !strcmp(op, "gather_dim")) {
    if (ndim < 1) return NULL;
    if (!canrun_shape_budget_ok(shape, ndim, 4096)) return NULL;
    PolyUOp *idx = canrun_buffer(ctx, POLY_INT32, shape, ndim);
    return idx ? poly_gather_dim(ctx, a, ndim - 1, idx) : NULL;
  }
  if (!strcmp(op, "sort")) {
    if (ndim < 1) return NULL;
    PolyUOp *values = NULL, *indices = NULL;
    if (poly_sort(ctx, a, ndim - 1, 0, &values, &indices) != 0) return NULL;
    (void)indices;
    return values;
  }
  if (!strcmp(op, "argsort")) {
    if (ndim < 1) return NULL;
    return poly_argsort(ctx, a, ndim - 1, 0);
  }
  if (!strcmp(op, "topk")) {
    if (ndim < 1 || shape[ndim - 1] <= 0) return NULL;
    int64_t k = shape[ndim - 1] >= 2 ? 2 : 1;
    PolyUOp *values = NULL, *indices = NULL;
    if (poly_topk(ctx, a, k, ndim - 1, 1, 1, &values, &indices) != 0) return NULL;
    (void)indices;
    return values;
  }
  if (!strcmp(op, "sum") || !strcmp(op, "reduce_sum"))
    return poly_sum_reduce(ctx, a, ndim > 0 ? ndim - 1 : 0, 0);
  if (!strcmp(op, "max") || !strcmp(op, "reduce_max"))
    return poly_max_reduce(ctx, a, ndim > 0 ? ndim - 1 : 0, 0);
  if (!strcmp(op, "mean")) return poly_mean_reduce(ctx, a, ndim > 0 ? ndim - 1 : 0, 0);
  if (!strcmp(op, "neg")) return poly_alu1(ctx, POLY_OP_NEG, a);
  if (!strcmp(op, "sqrt")) return poly_alu1(ctx, POLY_OP_SQRT, a);
  if (!strcmp(op, "exp2")) return poly_alu1(ctx, POLY_OP_EXP2, a);
  if (!strcmp(op, "log2")) return poly_alu1(ctx, POLY_OP_LOG2, a);
  if (!strcmp(op, "exp")) return poly_exp(ctx, a);
  if (!strcmp(op, "log")) return poly_log(ctx, a);
  if (!strcmp(op, "relu")) return poly_relu(ctx, a);
  if (!strcmp(op, "sigmoid")) return poly_sigmoid(ctx, a);

  PolyUOp *b = canrun_buffer(ctx, dt, shape, ndim);
  if (!b) return NULL;
  if (!strcmp(op, "add")) return poly_add(ctx, a, b);
  if (!strcmp(op, "sub")) return poly_sub(ctx, a, b);
  if (!strcmp(op, "mul")) return poly_mul(ctx, a, b);
  if (!strcmp(op, "div")) return poly_div(ctx, a, b);
  if (!strcmp(op, "maximum")) return poly_maximum(ctx, a, b);
  if (!strcmp(op, "gt")) return poly_gt(ctx, a, b);
  if (!strcmp(op, "where")) {
    PolyUOp *cond = poly_gt(ctx, a, b);
    return cond ? poly_where_op(ctx, cond, a, b) : NULL;
  }
  return NULL;
}

int poly_can_run_op(
    PolyCtx *ctx,
    int device_id,
    const char *op,
    int dtype_id,
    const int64_t *shape,
    int n_shape
) {
  if (!op || !canrun_shape_valid(shape, n_shape)) return -1;
  if (n_shape > 0 && !canrun_shape_budget_ok(shape, n_shape, 16384)) return -2;

  PolyDType dt;
  if (!poly_dtype_by_id(dtype_id, &dt)) return -1;
  dt = dt;

  PolyDevice device = (PolyDevice)device_id;
  if (device == POLY_DEVICE_AUTO && ctx) device = poly_ctx_get_preferred_device(ctx);
  if (device == POLY_DEVICE_AUTO) device = poly_device_default();
  if (!poly_device_can_execute(device)) return 0;

  int rc = 0;
  PolyCtx *probe = poly_ctx_new();
  PolyVarBinding *var_bindings = NULL;
  int n_var_bindings = 0;
  if (!probe) return -1;
  poly_ctx_set_preferred_device(probe, device);

  PolyUOp *value = canrun_build_probe_graph(probe, op, dt, shape, n_shape);
  PolyUOp *sink = value ? canrun_store_sink(probe, value) : NULL;
  if (!sink) goto cleanup;
  PolyUOp *linear = poly_linear_effect_sink(probe, sink, &var_bindings, &n_var_bindings);
  if (!linear || !poly_compile_linear(probe, linear, -1)) goto cleanup;
  rc = 1;

cleanup:
  free(var_bindings);
  poly_ctx_destroy(probe);
  return rc;
}

void poly_cpu_cache_flush(void) {
  /* Retained for ABI/frontend cleanup paths. CPU program caches are per-context
   * now and are released through poly_ctx_destroy(). */
}

int poly_abi_version(void) {
  return POLYGRAD_ABI_VERSION;
}

/* Debug helper — check opset state */
void poly_debug_opsets(void) {
  fprintf(stderr, "=== OpSet debug ===\n");
  fprintf(
      stderr, "POLY_GROUP_ALU.bits = [%llu, %llu]\n", (unsigned long long)POLY_GROUP_ALU.bits[0],
      (unsigned long long)POLY_GROUP_ALU.bits[1]
  );
  fprintf(
      stderr, "POLY_GROUP_UNARY.bits = [%llu, %llu]\n",
      (unsigned long long)POLY_GROUP_UNARY.bits[0], (unsigned long long)POLY_GROUP_UNARY.bits[1]
  );
  fprintf(
      stderr, "POLY_GROUP_BINARY.bits = [%llu, %llu]\n",
      (unsigned long long)POLY_GROUP_BINARY.bits[0], (unsigned long long)POLY_GROUP_BINARY.bits[1]
  );
  fprintf(
      stderr, "poly_opset_has(ALU, ADD=%d) = %d\n", POLY_OP_ADD,
      poly_opset_has(POLY_GROUP_ALU, POLY_OP_ADD)
  );
  fprintf(
      stderr, "poly_opset_has(BINARY, ADD=%d) = %d\n", POLY_OP_ADD,
      poly_opset_has(POLY_GROUP_BINARY, POLY_OP_ADD)
  );
  fprintf(stderr, "===================\n");
}

/* Debug helper — print UOp info */
void poly_debug_uop(PolyCtx *ctx, PolyUOp *u) {
  if (!u) {
    fprintf(stderr, "poly_debug_uop: NULL\n");
    return;
  }
  fprintf(
      stderr, "UOp@%p: op=%s(%d) n_src=%d arg.kind=%d", (void *)u, poly_op_name(u->op), u->op,
      u->n_src, u->arg.kind
  );
  if (u->arg.kind == POLY_ARG_INT) fprintf(stderr, " arg.i=%lld", (long long)u->arg.i);
  if (u->arg.kind == POLY_ARG_FLOAT) fprintf(stderr, " arg.f=%f", u->arg.f);
  fprintf(stderr, "\n");
  for (int i = 0; i < u->n_src; i++) {
    fprintf(
        stderr, "  src[%d]: @%p op=%s(%d)\n", i, (void *)u->src[i], poly_op_name(u->src[i]->op),
        u->src[i]->op
    );
  }
  /* Try shape */
  PolyShape s = poly_uop_max_shape(ctx, u);
  if (s.ndim >= 0) {
    fprintf(stderr, "  shape: (");
    for (int i = 0; i < s.ndim; i++) {
      if (i) fprintf(stderr, ", ");
      fprintf(stderr, "%lld", (long long)s.dims[i]);
    }
    fprintf(stderr, ")\n");
    if (s.ndim > 0 && s.dims) free(s.dims);
  } else {
    fprintf(stderr, "  shape: NONE\n");
  }
}

PolyUOp *poly_cast_by_id(PolyCtx *ctx, PolyUOp *x, int dtype_id) {
  PolyDType target;
  return poly_dtype_by_id(dtype_id, &target) ? poly_cast(ctx, x, target) : NULL;
}

PolyUOp *poly_bitcast_by_id(PolyCtx *ctx, PolyUOp *x, int dtype_id) {
  PolyDType target;
  if (!ctx || !x || !poly_dtype_by_id(dtype_id, &target)) return NULL;
  return poly_bitcast(ctx, x, target);
}

PolyUOp *poly_uop_range(PolyCtx *ctx, int64_t bound, int64_t axis_id, int axis_type) {
  return axis_type >= POLY_AXIS_DEVICE && axis_type <= POLY_AXIS_LOOP
             ? poly_range(ctx, bound, axis_id, (PolyAxisType)axis_type)
             : NULL;
}
