/* Existing canRun diagnostic policy, separate from FFI argument conversion.
 * This probes compilation, not execution or numerical correctness. */
#include "engine/realize.h"
#include "engine/schedule.h"
#include "ctx.h"
#include "device.h"
#include "tensor.h"
#include <stdlib.h>
#include <string.h>

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

static PolyUOp *canrun_budget_exceeded(int *status) {
  /* Probe limits bound diagnostic cost, not the underlying operation. */
  *status = -2;
  return NULL;
}

static PolyUOp *canrun_buffer(PolyCtx *ctx, PolyDType dt, const int64_t *shape, int ndim) {
  if (!canrun_shape_valid(shape, ndim)) return NULL;
  int64_t numel = poly_shape_numel_checked(shape, ndim);
  if (numel < 0) return NULL;
  /* Tinygrad UOp.new_buffer records the selected device in ParamArg
   * (uop/ops.py:814-817); capability probes must schedule the same graph. */
  PolyUOp *device = poly_device_uop(ctx, poly_ctx_get_preferred_device(ctx));
  PolyUOp *buf =
      device ? poly_uop_new_buffer(ctx, device, numel, dt, poly_ctx_next_unique_id(ctx)) : NULL;
  if (!buf) return NULL;
  return poly_uop_reshape(ctx, buf, (int64_t *)shape, ndim);
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
  PolyUOp *store = target ? poly_uop_store_val(ctx, target, value) : NULL;
  return store ? poly_uop_sink1(ctx, store) : NULL;
}

static PolyUOp *canrun_build_probe_graph(
    PolyCtx *ctx,
    const char *op,
    PolyDType dt,
    const int64_t *shape,
    int ndim,
    int *status
) {
  if (!ctx || !op || !canrun_shape_valid(shape, ndim)) return NULL;

  if (!strcmp(op, "matmul") || !strcmp(op, "dot")) {
    if (ndim != 3 || shape[0] <= 0 || shape[1] <= 0 || shape[2] <= 0) return NULL;
    int64_t a_shape[2] = {shape[0], shape[1]};
    int64_t b_shape[2] = {shape[1], shape[2]};
    if (!canrun_shape_budget_ok(a_shape, 2, 4096) || !canrun_shape_budget_ok(b_shape, 2, 4096))
      return canrun_budget_exceeded(status);
    PolyUOp *a = canrun_buffer(ctx, dt, a_shape, 2);
    PolyUOp *b = canrun_buffer(ctx, dt, b_shape, 2);
    return (a && b) ? poly_uop_dot(ctx, a, b) : NULL;
  }

  if (!strcmp(op, "triangular_solve") || !strcmp(op, "triangularSolve")) {
    if ((ndim != 2 && ndim != 3) || shape[0] <= 0 || shape[0] != shape[1]) return NULL;
    if (shape[0] > 16) return canrun_budget_exceeded(status);
    int64_t a_shape[2] = {shape[0], shape[1]};
    int64_t b_shape[2] = {shape[0], (ndim == 3) ? shape[2] : 1};
    PolyUOp *a = canrun_buffer(ctx, dt, a_shape, 2);
    PolyUOp *b = canrun_buffer(ctx, dt, b_shape, 2);
    return (a && b) ? poly_uop_triangular_solve(ctx, a, b, 0, 0, 0) : NULL;
  }

  if (!strcmp(op, "solve")) {
    if ((ndim != 2 && ndim != 3) || shape[0] <= 0 || shape[0] != shape[1]) return NULL;
    if (shape[0] > 12) return canrun_budget_exceeded(status);
    int64_t a_shape[2] = {shape[0], shape[1]};
    int64_t b_shape[2] = {shape[0], (ndim == 3) ? shape[2] : 1};
    PolyUOp *a = canrun_buffer(ctx, dt, a_shape, 2);
    PolyUOp *b = canrun_buffer(ctx, dt, b_shape, 2);
    return (a && b) ? poly_uop_solve(ctx, a, b) : NULL;
  }

  if (!strcmp(op, "lstsq")) {
    if ((ndim != 2 && ndim != 3) || shape[0] <= 0 || shape[1] <= 0) return NULL;
    if (shape[0] > 12 || shape[1] > 12) return canrun_budget_exceeded(status);
    int64_t a_shape[2] = {shape[0], shape[1]};
    int64_t b_shape[2] = {shape[0], (ndim == 3) ? shape[2] : 1};
    PolyUOp *a = canrun_buffer(ctx, dt, a_shape, 2);
    PolyUOp *b = canrun_buffer(ctx, dt, b_shape, 2);
    return (a && b) ? poly_uop_lstsq(ctx, a, b) : NULL;
  }

  if (!canrun_shape_budget_ok(shape, ndim, 8192)) return canrun_budget_exceeded(status);
  PolyUOp *a = canrun_buffer(ctx, dt, shape, ndim);
  if (!a) return NULL;

  if (!strcmp(op, "qr")) {
    if (ndim < 2) return NULL;
    if (shape[ndim - 2] > 16 || shape[ndim - 1] > 16) return canrun_budget_exceeded(status);
    PolyUOp *q = NULL, *r = NULL;
    return poly_uop_qr_ex(ctx, a, POLY_QR_REDUCED, &q, &r) == 0 ? r : NULL;
  }
  if (!strcmp(op, "cholesky")) {
    if (ndim < 2 || shape[ndim - 2] <= 0 || shape[ndim - 2] != shape[ndim - 1]) return NULL;
    if (shape[ndim - 1] > 16) return canrun_budget_exceeded(status);
    return poly_uop_cholesky(ctx, a, 0);
  }
  if (!strcmp(op, "gather") || !strcmp(op, "gather_dim")) {
    if (ndim < 1) return NULL;
    if (!canrun_shape_budget_ok(shape, ndim, 4096)) return canrun_budget_exceeded(status);
    PolyUOp *idx = canrun_buffer(ctx, POLY_INT32, shape, ndim);
    return idx ? poly_uop_gather_dim(ctx, a, ndim - 1, idx) : NULL;
  }
  if (!strcmp(op, "sort")) {
    if (ndim < 1) return NULL;
    PolyUOp *values = NULL, *indices = NULL;
    if (poly_uop_sort(ctx, a, ndim - 1, 0, &values, &indices) != 0) return NULL;
    (void)indices;
    return values;
  }
  if (!strcmp(op, "argsort")) {
    if (ndim < 1) return NULL;
    return poly_uop_argsort(ctx, a, ndim - 1, 0);
  }
  if (!strcmp(op, "topk")) {
    if (ndim < 1 || shape[ndim - 1] <= 0) return NULL;
    int64_t k = shape[ndim - 1] >= 2 ? 2 : 1;
    PolyUOp *values = NULL, *indices = NULL;
    if (poly_uop_topk(ctx, a, k, ndim - 1, 1, 1, &values, &indices) != 0) return NULL;
    (void)indices;
    return values;
  }
  if (!strcmp(op, "sum") || !strcmp(op, "reduce_sum"))
    return poly_uop_sum_reduce(ctx, a, ndim > 0 ? ndim - 1 : 0, 0);
  if (!strcmp(op, "max") || !strcmp(op, "reduce_max"))
    return poly_uop_max_reduce(ctx, a, ndim > 0 ? ndim - 1 : 0, 0);
  if (!strcmp(op, "mean")) return poly_uop_mean_reduce(ctx, a, ndim > 0 ? ndim - 1 : 0, 0);
  if (!strcmp(op, "neg")) return poly_uop_alu1(ctx, POLY_OP_NEG, a);
  if (!strcmp(op, "sqrt")) return poly_uop_alu1(ctx, POLY_OP_SQRT, a);
  if (!strcmp(op, "exp2")) return poly_uop_alu1(ctx, POLY_OP_EXP2, a);
  if (!strcmp(op, "log2")) return poly_uop_alu1(ctx, POLY_OP_LOG2, a);
  if (!strcmp(op, "exp")) return poly_uop_exp(ctx, a);
  if (!strcmp(op, "log")) return poly_uop_log(ctx, a);
  if (!strcmp(op, "relu")) return poly_uop_relu(ctx, a);
  if (!strcmp(op, "sigmoid")) return poly_uop_sigmoid(ctx, a);

  PolyUOp *b = canrun_buffer(ctx, dt, shape, ndim);
  if (!b) return NULL;
  if (!strcmp(op, "add")) return poly_uop_add(ctx, a, b);
  if (!strcmp(op, "sub")) return poly_uop_sub(ctx, a, b);
  if (!strcmp(op, "mul")) return poly_uop_mul(ctx, a, b);
  if (!strcmp(op, "div")) return poly_uop_div(ctx, a, b);
  if (!strcmp(op, "maximum")) return poly_uop_maximum(ctx, a, b);
  if (!strcmp(op, "gt")) return poly_uop_gt(ctx, a, b);
  if (!strcmp(op, "where")) {
    PolyUOp *cond = poly_uop_gt(ctx, a, b);
    return cond ? poly_uop_where(ctx, cond, a, b) : NULL;
  }
  return NULL;
}

int poly_can_compile_op(
    PolyCtx *ctx,
    PolyDevice device,
    const char *op,
    PolyDType dt,
    const int64_t *shape,
    int n_shape
) {
  if (!op || !canrun_shape_valid(shape, n_shape)) return -1;
  if (n_shape > 0 && !canrun_shape_budget_ok(shape, n_shape, 16384)) return -2;

  if (device == POLY_DEVICE_AUTO && ctx) device = poly_ctx_get_preferred_device(ctx);
  if (device == POLY_DEVICE_AUTO) device = poly_device_default();
  if (!poly_device_can_execute(device)) return 0;

  int rc = 0;
  PolyCtx *probe = poly_ctx_new();
  PolyVarBinding *var_bindings = NULL;
  int n_var_bindings = 0;
  if (!probe) return -1;
  poly_ctx_set_preferred_device(probe, device);

  PolyUOp *value = canrun_build_probe_graph(probe, op, dt, shape, n_shape, &rc);
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
