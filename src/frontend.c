/*
 * frontend.c — FFI-friendly helpers for language bindings
 *
 * Thin wrappers around the core API that avoid passing PolyArg/PolyDType
 * across FFI boundaries. Execution uses graph/schedule entrypoints backed by
 * ctx->buffers; there is no separate runtime buffer-binding graph.
 */

#define _GNU_SOURCE
#include "frontend.h"
#include "frontend_internal.h"
#include "engine/realize.h"
#include "engine/schedule.h"
#include "schedule/rangeify.h"
#include "codegen.h"
#include "tensor.h"
#include "interp.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "utils.h"

/* Dtype table for FFI (shared by buffer/dtype convenience helpers) */

static const PolyDType *_dtype_table_ffi[] = {
    &POLY_VOID,    &POLY_BOOL,     &POLY_INT8,    &POLY_UINT8,   &POLY_INT16,
    &POLY_UINT16,  &POLY_INT32,    &POLY_UINT32,  &POLY_INT64,   &POLY_UINT64,
    &POLY_FLOAT16, &POLY_BFLOAT16, &POLY_FLOAT32, &POLY_FLOAT64,
};
#define N_DTYPE_FFI ((int)(sizeof(_dtype_table_ffi) / sizeof(_dtype_table_ffi[0])))

PolyUOp *poly_buffer_by_id(PolyCtx *ctx, int dtype_id, int64_t size) {
  PolyDType dt;
  if (!poly_dtype_by_id(dtype_id, &dt)) return NULL;
  return poly_buffer(ctx, poly_dtype_scalar(dt), size);
}

PolyUOp *poly_buffer_on_device_by_id(
    PolyCtx *ctx,
    int dtype_id,
    int64_t size,
    int device_id
) {
  PolyDType dt;
  if (!poly_dtype_by_id(dtype_id, &dt)) return NULL;
  PolyDevice device = (PolyDevice)device_id;
  if (device <= POLY_DEVICE_HOST || device > POLY_DEVICE_X86) return NULL;
  return poly_buffer_on_device(ctx, poly_dtype_scalar(dt), size, device);
}

PolyUOp *poly_buffer_var_by_id(
    PolyCtx *ctx,
    int dtype_id,
    PolyUOp *batch_var,
    const int64_t *inner_dims,
    int n_inner,
    int device_id
) {
  if (!ctx || !batch_var || n_inner < 0 || n_inner >= POLY_MAX_DIMS ||
      (n_inner > 0 && !inner_dims))
    return NULL;
  PolyDType dt;
  if (!poly_dtype_by_id(dtype_id, &dt)) return NULL;
  PolyDevice device = (PolyDevice)device_id;
  if (device != POLY_DEVICE_AUTO &&
      (device <= POLY_DEVICE_HOST || device > POLY_DEVICE_X86))
    return NULL;

  PolyUOp *var = poly_uop_unbind_var(batch_var);
  if (!var || var->arg.kind != POLY_ARG_DEFINE_VAR || var->arg.define_var.max_val < 0)
    return NULL;

  int ndim = n_inner + 1;
  int64_t max_shape[POLY_MAX_DIMS];
  int64_t alloc = var->arg.define_var.max_val;
  max_shape[0] = alloc;
  for (int i = 0; i < n_inner; i++) {
    if (inner_dims[i] < 0 ||
        (inner_dims[i] != 0 && alloc > INT64_MAX / inner_dims[i]))
      return NULL;
    alloc *= inner_dims[i];
    max_shape[i + 1] = inner_dims[i];
  }

  PolyUOp *base =
      poly_buffer_on_device(ctx, poly_dtype_scalar(dt), alloc, device);
  PolyUOp *reshaped = poly_reshape(ctx, base, max_shape, ndim);
  if (!base || !reshaped) return NULL;

  PolyUOp *starts[POLY_MAX_DIMS];
  PolyUOp *sizes[POLY_MAX_DIMS];
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(0));
  if (!zero) return NULL;
  starts[0] = zero;
  sizes[0] = batch_var;
  for (int i = 0; i < n_inner; i++) {
    starts[i + 1] = zero;
    sizes[i + 1] =
        poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(inner_dims[i]));
    if (!sizes[i + 1]) return NULL;
  }
  return poly_shrink_uop(ctx, reshaped, starts, sizes, ndim);
}

PolyUOp *poly_buffer_f32(PolyCtx *ctx, int64_t size) {
  return poly_buffer(ctx, POLY_FLOAT32, size);
}

PolyUOp *poly_buffer_f64(PolyCtx *ctx, int64_t size) {
  return poly_buffer(ctx, POLY_FLOAT64, size);
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
  return poly_tensor_empty(ctx, poly_dtype_scalar(dt), dims, ndim, (PolyDevice)device_id);
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
  return poly_tensor_from_host(ctx, ptr, nbytes, poly_dtype_scalar(dt), dims, ndim);
}

PolyTensor *poly_tensor_const_int_by_id(
    PolyCtx *ctx,
    int64_t value,
    int dtype_id,
    int device_id
) {
  PolyUOp *value_uop = poly_const_int_by_id(ctx, value, dtype_id);
  if (!value_uop) return NULL;
  return poly_tensor_create_with_roots(
      ctx, value_uop, value_uop, POLY_TENSOR_VALUE, (PolyDevice)device_id
  );
}

PolyTensor *poly_tensor_const_float_by_id(
    PolyCtx *ctx,
    double value,
    int dtype_id,
    int device_id
) {
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
    int device_id
) {
  PolyUOp *value_uop = poly_full_int_by_id(ctx, dims, ndim, value, dtype_id);
  if (!value_uop) return NULL;
  return poly_tensor_create_with_roots(
      ctx, value_uop, value_uop, POLY_TENSOR_VALUE, (PolyDevice)device_id
  );
}

PolyTensor *poly_tensor_full_float_by_id(
    PolyCtx *ctx,
    const int64_t *dims,
    int ndim,
    double value,
    int dtype_id,
    int device_id
) {
  PolyUOp *value_uop = poly_full_float_by_id(ctx, dims, ndim, value, dtype_id);
  if (!value_uop) return NULL;
  return poly_tensor_create_with_roots(
      ctx, value_uop, value_uop, POLY_TENSOR_VALUE, (PolyDevice)device_id
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

PolyTensor *poly_tensor_eye_by_id(
    PolyCtx *ctx,
    int64_t n,
    int64_t m,
    int dtype_id,
    int device_id
) {
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
  PolyDType sdt = poly_dtype_scalar(u->dtype);
  for (int i = 0; i < N_DTYPE_FFI; i++) {
    if (poly_dtype_eq(sdt, *_dtype_table_ffi[i])) return i;
  }
  return 0;
}

int poly_uop_n_src(PolyUOp *u) {
  return u ? u->n_src : 0;
}

PolyUOp *poly_uop_src(PolyUOp *u, int idx) {
  if (!u || idx < 0 || idx >= u->n_src) return NULL;
  return u->src[idx];
}

static PolyDType frontend_value_dtype(PolyDType dt) {
  if (!dt.is_ptr) return poly_dtype_scalar(dt);
  PolyDType base = dt;
  base.is_ptr = false;
  base.addrspace = POLY_ADDR_GLOBAL;
  base.vcount = 0;
  base.ptr_size = 0;
  return poly_dtype_scalar(base);
}

PolyUOp *poly_uop_placeholder_like(PolyCtx *ctx, PolyUOp *like, int slot) {
  if (!ctx || !like || slot < 0) return NULL;
  int ndim = poly_uop_ndim(ctx, like);
  if (ndim < 0 || ndim > POLY_MAX_DIMS) return NULL;
  const int64_t *dims = poly_uop_max_shape_dims(ctx, like);
  if (ndim > 0 && !dims) return NULL;
  int64_t numel = ndim == 0 ? 1 : poly_shape_numel_checked(dims, ndim);
  if (numel < 0) return NULL;
  PolyDType ptr_dt = poly_dtype_ptr(frontend_value_dtype(like->dtype), numel, POLY_ADDR_GLOBAL);
  PolyUOp *param = poly_uop0(ctx, POLY_OP_PARAM, ptr_dt, poly_arg_int(slot));
  if (!param || ndim <= 1) return param;
  return poly_reshape(ctx, param, (int64_t *)dims, ndim);
}

PolyUOp *poly_uop_range(PolyCtx *ctx, int64_t bound, int64_t axis_id, int axis_type) {
  if (!ctx || bound < 0) return NULL;
  if (axis_type < POLY_AXIS_GLOBAL || axis_type > POLY_AXIS_PLACEHOLDER) return NULL;
  PolyUOp *bound_uop = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(bound));
  return poly_uop1(
      ctx, POLY_OP_RANGE, POLY_INT32, bound_uop,
      poly_arg_range(axis_id, (PolyAxisType)axis_type)
  );
}

PolyUOp *poly_uop_index(
    PolyCtx *ctx,
    PolyUOp *base,
    PolyUOp **indices,
    int n_indices,
    int keep_ptr
) {
  if (!ctx || !base || n_indices < 0 || (n_indices > 0 && !indices)) return NULL;
  if (n_indices > POLY_MAX_DIMS) return NULL;
  PolyUOp *src[POLY_MAX_DIMS + 1];
  src[0] = base;
  for (int i = 0; i < n_indices; i++) {
    if (!indices[i]) return NULL;
    src[1 + i] = indices[i];
  }
  PolyDType out_dt = keep_ptr ? base->dtype : frontend_value_dtype(base->dtype);
  return poly_uop(ctx, POLY_OP_INDEX, out_dt, src, n_indices + 1, poly_arg_none());
}

PolyUOp *poly_uop_load(PolyCtx *ctx, PolyUOp *addr) {
  if (!ctx || !addr) return NULL;
  if (!addr->dtype.is_ptr) return addr;
  return poly_uop1(ctx, POLY_OP_LOAD, frontend_value_dtype(addr->dtype), addr, poly_arg_none());
}

PolyUOp *poly_uop_store(PolyCtx *ctx, PolyUOp *addr, PolyUOp *value) {
  if (!ctx || !addr || !value) return NULL;
  PolyUOp *ptr = addr;
  if (!ptr->dtype.is_ptr && ptr->op == POLY_OP_INDEX && ptr->n_src >= 1 && ptr->src[0] &&
      ptr->src[0]->dtype.is_ptr) {
    PolyUOp *src[POLY_MAX_DIMS + 1];
    if (ptr->n_src > POLY_MAX_DIMS + 1) return NULL;
    for (int i = 0; i < ptr->n_src; i++)
      src[i] = ptr->src[i];
    ptr = poly_uop(ctx, POLY_OP_INDEX, ptr->src[0]->dtype, src, ptr->n_src, ptr->arg);
  }
  return poly_store_val(ctx, ptr, value);
}

PolyUOp *poly_uop_set(PolyCtx *ctx, PolyUOp *addr, PolyUOp *value, PolyUOp **ranges, int n_ranges) {
  if (!ctx || !addr || !value || n_ranges < 0 || (n_ranges > 0 && !ranges)) return NULL;
  if (addr->op != POLY_OP_INDEX || addr->n_src < 1 || !addr->src[0]) return NULL;
  PolyUOp *store = poly_uop_store(ctx, addr, value);
  if (!store) return NULL;
  PolyUOp *effect = store;
  if (n_ranges > 0) {
    effect = poly_uop_end(ctx, store, ranges, n_ranges);
    if (!effect) return NULL;
  }
  return poly_uop_after(ctx, addr->src[0], effect);
}

PolyUOp *poly_uop_group(PolyCtx *ctx, PolyUOp **srcs, int n_src) {
  if (!ctx || n_src <= 0 || !srcs) return NULL;
  return poly_uop(ctx, POLY_OP_GROUP, POLY_VOID, srcs, n_src, poly_arg_none());
}

PolyUOp *poly_uop_end(PolyCtx *ctx, PolyUOp *body, PolyUOp **ranges, int n_ranges) {
  if (!ctx || !body || n_ranges < 0 || (n_ranges > 0 && !ranges)) return NULL;
  PolyUOp **src = malloc((size_t)(n_ranges + 1) * sizeof(PolyUOp *));
  if (!src) return NULL;
  src[0] = body;
  for (int i = 0; i < n_ranges; i++) {
    if (!ranges[i]) {
      free(src);
      return NULL;
    }
    src[1 + i] = ranges[i];
  }
  PolyUOp *ret = poly_uop(ctx, POLY_OP_END, POLY_VOID, src, n_ranges + 1, poly_arg_none());
  free(src);
  return ret;
}

PolyUOp *poly_uop_sink(PolyCtx *ctx, PolyUOp **srcs, int n_src) {
  if (!ctx || n_src < 0 || (n_src > 0 && !srcs)) return NULL;
  return poly_uop(ctx, POLY_OP_SINK, POLY_VOID, srcs, n_src, poly_arg_none());
}

PolyUOp *poly_uop_sink_ex(
    PolyCtx *ctx,
    PolyUOp **srcs,
    int n_src,
    const char *name,
    int optimize
) {
  if (!ctx || n_src < 0 || (n_src > 0 && !srcs)) return NULL;
  PolyArg arg = (name && name[0]) ? poly_arg_str(name) : poly_arg_none();
  if (optimize) return poly_uop(ctx, POLY_OP_SINK, POLY_VOID, srcs, n_src, arg);
  return poly_uop_tagged_arg(
      ctx, POLY_OP_SINK, POLY_VOID, srcs, n_src, arg, 0, poly_arg_bool(false)
  );
}

PolyUOp *poly_uop_call(PolyCtx *ctx, PolyUOp *body, PolyUOp **args, int n_args) {
  if (!ctx || !body || n_args < 0 || (n_args > 0 && !args)) return NULL;
  PolyUOp **src = malloc((size_t)(n_args + 1) * sizeof(PolyUOp *));
  if (!src) return NULL;
  src[0] = body;
  for (int i = 0; i < n_args; i++) {
    if (!args[i]) {
      free(src);
      return NULL;
    }
    src[1 + i] = args[i];
  }
  PolyUOp *ret = poly_uop(ctx, POLY_OP_CALL, POLY_VOID, src, n_args + 1, poly_arg_none());
  free(src);
  return ret;
}

PolyUOp *poly_uop_after(PolyCtx *ctx, PolyUOp *target, PolyUOp *effect) {
  if (!ctx || !target || !effect) return NULL;
  PolyUOp *src[2] = {target, effect};
  return poly_uop(ctx, POLY_OP_AFTER, target->dtype, src, 2, poly_arg_none());
}

PolyUOp *poly_uop_reduce(PolyCtx *ctx, PolyOps reduce_op, PolyUOp *expr, PolyUOp **ranges, int n_ranges) {
  if (!ctx || !expr || n_ranges < 0 || (n_ranges > 0 && !ranges)) return NULL;
  switch (reduce_op) {
    case POLY_OP_ADD:
    case POLY_OP_MUL:
    case POLY_OP_MAX:
    case POLY_OP_AND:
    case POLY_OP_OR:
      break;
    default:
      return NULL;
  }
  if (n_ranges == 0) return expr;
  PolyUOp **src = malloc((size_t)(n_ranges + 1) * sizeof(PolyUOp *));
  if (!src) return NULL;
  PolyUOp **from = malloc((size_t)n_ranges * sizeof(PolyUOp *));
  PolyUOp **to = malloc((size_t)n_ranges * sizeof(PolyUOp *));
  if (!from || !to) {
    free(src);
    free(from);
    free(to);
    return NULL;
  }
  int n_subs = 0;
  src[0] = expr;
  for (int i = 0; i < n_ranges; i++) {
    if (!ranges[i] || ranges[i]->op != POLY_OP_RANGE) {
      free(src);
      free(from);
      free(to);
      return NULL;
    }
    PolyAxisType axis_type = poly_range_axis_type(ranges[i]->arg);
    if (axis_type == POLY_AXIS_LOOP) {
      PolyArg range_arg = ranges[i]->arg;
      PolyArg new_arg = poly_arg_range(
          poly_range_axis_id(range_arg), POLY_AXIS_REDUCE
      );
      if (poly_range_n_extra(range_arg) > 0) {
        new_arg = poly_arg_range_ex(
            poly_range_axis_id(range_arg), POLY_AXIS_REDUCE,
            poly_range_extra(range_arg), poly_range_n_extra(range_arg)
        );
      }
      PolyUOp *rr = poly_uop(ctx, POLY_OP_RANGE, ranges[i]->dtype, ranges[i]->src, ranges[i]->n_src, new_arg);
      if (!rr) {
        free(src);
        free(from);
        free(to);
        return NULL;
      }
      from[n_subs] = ranges[i];
      to[n_subs] = rr;
      n_subs++;
      src[i + 1] = rr;
    } else if (axis_type == POLY_AXIS_REDUCE || axis_type == POLY_AXIS_GROUP_REDUCE ||
               axis_type == POLY_AXIS_UNROLL) {
      src[i + 1] = ranges[i];
    } else {
      free(src);
      free(from);
      free(to);
      return NULL;
    }
  }
  if (n_subs > 0) src[0] = poly_uop_substitute(ctx, expr, from, to, n_subs);
  PolyUOp *ret = poly_uop(ctx, POLY_OP_REDUCE, expr->dtype, src, n_ranges + 1, poly_arg_ops(reduce_op));
  free(src);
  free(from);
  free(to);
  return ret;
}

int64_t poly_uop_numel(PolyCtx *ctx, PolyUOp *u) {
  if (!ctx || !u) return -1;
  int ndim = poly_uop_ndim(ctx, u);
  if (ndim < 0 || ndim > POLY_MAX_DIMS) return -1;
  const int64_t *dims = poly_uop_max_shape_dims(ctx, u);
  if (ndim > 0 && !dims) return -1;
  return ndim == 0 ? 1 : poly_shape_numel_checked(dims, ndim);
}

PolyUOp *poly_uop_flatten(PolyCtx *ctx, PolyUOp *u) {
  if (!ctx || !u) return NULL;
  int64_t numel = poly_uop_numel(ctx, u);
  if (numel < 0) return NULL;
  int ndim = poly_uop_ndim(ctx, u);
  const int64_t *dims = poly_uop_max_shape_dims(ctx, u);
  if (ndim == 1 && dims && dims[0] == numel) return u;
  int64_t shape[1] = {numel};
  return poly_reshape(ctx, u, shape, 1);
}

/* Dynamic shapes (DEFINE_VAR / BIND) */

PolyUOp *poly_define_var(PolyCtx *ctx, const char *name, int64_t min_val, int64_t max_val) {
  /* Name string is copied into arena by poly_uop_create (POLY_ARG_DEFINE_VAR case) */
  return poly_uop0(
      ctx, POLY_OP_DEFINE_VAR, POLY_INT32, poly_arg_define_var(name, min_val, max_val)
  );
}

PolyUOp *poly_bind_var(PolyCtx *ctx, PolyUOp *var, int64_t value) {
  assert(var->op == POLY_OP_DEFINE_VAR);
  PolyUOp *val = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(value));
  return poly_uop2(ctx, POLY_OP_BIND, var->dtype, var, val, poly_arg_none());
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
  PolyUOp *buf = poly_buffer(ctx, poly_dtype_scalar(dt), numel);
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
  PolyUOp *target = canrun_buffer(ctx, poly_dtype_scalar(value->dtype), shape, ndim);
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
  dt = poly_dtype_scalar(dt);

  PolyDevice device = (PolyDevice)device_id;
  if (device == POLY_DEVICE_AUTO && ctx) device = poly_ctx_get_preferred_device(ctx);
  if (device == POLY_DEVICE_AUTO) device = poly_device_default();
  if (device == POLY_DEVICE_HOST || !poly_device_can_execute(device)) return 0;

  int rc = 0;
  PolyCtx *probe = poly_ctx_new();
  PolySchedule *sched = NULL;
  PolyCompiledSchedule *plan = NULL;
  if (!probe) return -1;
  poly_ctx_set_preferred_device(probe, device);

  PolyUOp *value = canrun_build_probe_graph(probe, op, dt, shape, n_shape);
  PolyUOp *sink = value ? canrun_store_sink(probe, value) : NULL;
  if (!sink) goto cleanup;
  sched = poly_schedule_effect_sink(probe, sink);
  if (!sched) goto cleanup;
  plan = poly_lower_schedule(probe, sched, device);
  if (!plan) goto cleanup;
  rc = 1;

cleanup:
  poly_compiled_schedule_free(plan);
  poly_schedule_free(sched);
  poly_ctx_destroy(probe);
  return rc;
}

/* POLY_MAX_REALIZE_BUFS defined in frontend_internal.h */

static bool uop_vec_append(PolyUOp ***items, int *count, int *cap, PolyUOp *u) {
  if (!items || !count || !cap) return false;
  if (*count >= *cap) {
    int new_cap = *cap ? *cap * 2 : 16;
    PolyUOp **tmp = realloc(*items, (size_t)new_cap * sizeof(PolyUOp *));
    if (!tmp) return false;
    *items = tmp;
    *cap = new_cap;
  }
  (*items)[(*count)++] = u;
  return true;
}

static bool uop_vec_contains(PolyUOp **items, int count, PolyUOp *u) {
  for (int i = 0; i < count; i++)
    if (items[i] == u) return true;
  return false;
}

typedef bool (*CollectBufferFn)(PolyUOp *u, void *user_data);

static bool collect_input_buffers_postorder(
    PolyUOp *root,
    CollectBufferFn collect,
    void *user_data
) {
  if (!root || !collect) return false;

  PolyMap *visited = poly_map_new(256);
  if (!visited) return false;

  int stack_cap = 256;
  int stack_top = 0;
  PolyUOp **stack = malloc((size_t)stack_cap * sizeof(PolyUOp *));
  int *state = malloc((size_t)stack_cap * sizeof(int));
  if (!stack || !state) {
    free(stack);
    free(state);
    poly_map_destroy(visited);
    return false;
  }

  stack[stack_top] = root;
  state[stack_top++] = 0;

  while (stack_top > 0) {
    PolyUOp *u = stack[stack_top - 1];
    int s = state[stack_top - 1];
    uint32_t h = poly_ptr_hash(u);

    if (s == 0 && poly_map_get(visited, h, u, poly_ptr_eq) != NULL) {
      stack_top--;
      continue;
    }

    if (s == 0) {
      state[stack_top - 1] = 1;
      /* BUFFER_VIEW is an executable storage identity, not a value
       * computation. Its sources describe alias provenance and must not add
       * the arena/base bookkeeping buffers as separate CALL arguments. */
      if (u->op != POLY_OP_BUFFER && u->op != POLY_OP_BUFFER_VIEW) {
        for (int i = u->n_src - 1; i >= 0; i--) {
          PolyUOp *src = u->src[i];
          if (!src) continue;
          uint32_t sh = poly_ptr_hash(src);
          if (poly_map_get(visited, sh, src, poly_ptr_eq) != NULL) continue;
          if (stack_top >= stack_cap) {
            int new_cap = stack_cap * 2;
            PolyUOp **new_stack = realloc(stack, (size_t)new_cap * sizeof(PolyUOp *));
            int *new_state = realloc(state, (size_t)new_cap * sizeof(int));
            if (!new_stack || !new_state) {
              free(new_stack ? new_stack : stack);
              free(new_state ? new_state : state);
              poly_map_destroy(visited);
              return false;
            }
            stack = new_stack;
            state = new_state;
            stack_cap = new_cap;
          }
          stack[stack_top] = src;
          state[stack_top++] = 0;
        }
      }
      continue;
    }

    stack_top--;
    if (poly_map_get(visited, h, u, poly_ptr_eq) != NULL) continue;
    poly_map_set(visited, h, u, (void *)(uintptr_t)1, poly_ptr_eq);
    if ((u->op == POLY_OP_BUFFER || u->op == POLY_OP_BUFFER_VIEW) &&
        !collect(u, user_data)) {
      free(stack);
      free(state);
      poly_map_destroy(visited);
      return false;
    }
  }

  free(stack);
  free(state);
  poly_map_destroy(visited);
  return true;
}

typedef struct {
  PolyUOp ***ordered;
  int *n;
  int *cap;
} DynamicBufferCollect;

static bool collect_dynamic_buffer(PolyUOp *u, void *user_data) {
  DynamicBufferCollect *c = (DynamicBufferCollect *)user_data;
  if (!c || !c->ordered || !c->n || !c->cap) return false;
  return uop_vec_contains(*c->ordered, *c->n, u) ||
         uop_vec_append(c->ordered, c->n, c->cap, u);
}

typedef struct {
  PolyUOp **ordered;
  int *n;
  int max_bufs;
} FixedBufferCollect;

static bool collect_fixed_buffer(PolyUOp *u, void *user_data) {
  FixedBufferCollect *c = (FixedBufferCollect *)user_data;
  if (!c || !c->ordered || !c->n || c->max_bufs <= 0) return false;
  int stored = *c->n < c->max_bufs ? *c->n : c->max_bufs;
  if (uop_vec_contains(c->ordered, stored, u)) return true;
  if (*c->n < c->max_bufs) c->ordered[*c->n] = u;
  (*c->n)++;
  return true;
}

/* Reconstruct the buffer-to-PARAM ordering used by kernel-graph scheduling:
 * 1. Output buffers (STORE targets in SINK source order)
 * 2. Remaining input buffers (toposort encounter order) */
bool poly_collect_ordered_buffers_alloc(
    PolyCtx *ctx,
    PolyUOp *tensor_sink,
    PolyUOp ***out_ordered,
    int *out_n_ordered
) {
  if (!ctx || !tensor_sink || !out_ordered || !out_n_ordered) return false;
  *out_ordered = NULL;
  *out_n_ordered = 0;

  PolyUOp **ordered = NULL;
  int n = 0, cap = 0;

  /* Output buffers first */
  for (int i = 0; i < tensor_sink->n_src; i++) {
    PolyUOp *store = tensor_sink->src[i];
    if (store && store->op == POLY_OP_STORE && store->n_src >= 1 &&
        (store->src[0]->op == POLY_OP_BUFFER ||
         store->src[0]->op == POLY_OP_BUFFER_VIEW)) {
      PolyUOp *buf = store->src[0];
      if (!uop_vec_contains(ordered, n, buf) && !uop_vec_append(&ordered, &n, &cap, buf)) {
        free(ordered);
        return false;
      }
    }
  }

  /* Input buffers in toposort order. This is a local scan, like tinygrad's
   * temporary UOp.toposort() result, so do not grow the persistent ctx arena. */
  DynamicBufferCollect collect = {.ordered = &ordered, .n = &n, .cap = &cap};
  if (!collect_input_buffers_postorder(tensor_sink, collect_dynamic_buffer, &collect)) {
    free(ordered);
    return false;
  }

  *out_ordered = ordered;
  *out_n_ordered = n;
  return true;
}

int poly_collect_ordered_buffers(
    PolyCtx *ctx,
    PolyUOp *tensor_sink,
    PolyUOp **ordered,
    int max_bufs
) {
  if (!ctx || !tensor_sink || !ordered || max_bufs <= 0) return 0;
  int n = 0;

  /* Output buffers first */
  for (int i = 0; i < tensor_sink->n_src; i++) {
    PolyUOp *store = tensor_sink->src[i];
    if (store && store->op == POLY_OP_STORE && store->n_src >= 1 &&
        (store->src[0]->op == POLY_OP_BUFFER ||
         store->src[0]->op == POLY_OP_BUFFER_VIEW)) {
      PolyUOp *buf = store->src[0];
      if (!uop_vec_contains(ordered, n < max_bufs ? n : max_bufs, buf)) {
        if (n < max_bufs) ordered[n] = buf;
        n++;
      }
    }
  }

  FixedBufferCollect collect = {.ordered = ordered, .n = &n, .max_bufs = max_bufs};
  if (!collect_input_buffers_postorder(tensor_sink, collect_fixed_buffer, &collect)) return 0;
  return n;
}

/* Weak context cleanup hook called from ctx.c when this translation unit is
 * linked. Frontend-global caches were removed; per-context caches are owned by
 * ctx/schedule/program-cache teardown. */
void poly_frontend_ctx_cleanup(PolyCtx *ctx) {
  (void)ctx;
}

/* Structural hash/eq for graph caches.
 * Cached schedules/programs need to match computations that are structurally
 * identical but use different BUFFER UOp instances, such as fresh training
 * step buffers. We hash/compare the computation DAG structure: ops, dtypes,
 * args, and connectivity, treating BUFFER and BUFFER_VIEW storage identities
 * as positional placeholders (first encountered = 0, etc.).
 */

/* POLY_MAX_STRUCT_NODES defined in frontend_internal.h */

typedef struct {
  PolyMap *visited; /* UOp* -> 1-based index into hashes. Dynamic for model-scale DAGs. */
  uint32_t *hashes;
  int n_hashes;
  int cap_hashes;
  PolyUOp **bufs;
  int n_bufs;
  int cap_bufs;
} StructHashCtx;

static uint32_t struct_hash_impl(PolyUOp *u, StructHashCtx *ctx) {
  /* Check if already visited */
  void *memo_val = poly_map_get(ctx->visited, poly_ptr_hash(u), u, poly_ptr_eq);
  if (memo_val) return ctx->hashes[(int)((intptr_t)memo_val - 1)];

  uint32_t h = 0x811c9dc5; /* FNV-1a offset basis */

  if (u->op == POLY_OP_BUFFER || u->op == POLY_OP_BUFFER_VIEW) {
    /* Storage identities: use positional ID instead of pointer identity.
     * BUFFER_VIEW metadata remains in op/dtype/arg; its provenance sources
     * are deliberately outside the executable schedule-cache identity. */
    int buf_id = -1;
    for (int i = 0; i < ctx->n_bufs; i++) {
      if (ctx->bufs[i] == u) {
        buf_id = i;
        break;
      }
    }
    if (buf_id < 0) {
      if (ctx->n_bufs >= ctx->cap_bufs) {
        int new_cap = ctx->cap_bufs ? ctx->cap_bufs * 2 : 64;
        PolyUOp **new_bufs = realloc(ctx->bufs, (size_t)new_cap * sizeof(PolyUOp *));
        if (!new_bufs) return h;
        ctx->bufs = new_bufs;
        ctx->cap_bufs = new_cap;
      }
      buf_id = ctx->n_bufs;
      ctx->bufs[ctx->n_bufs++] = u;
    }
    h ^= (uint32_t)u->op;
    h *= 0x01000193;
    h ^= (uint32_t)u->dtype.priority;
    h *= 0x01000193;
    h ^= (uint32_t)u->dtype.bitsize;
    h *= 0x01000193;
    h ^= (uint32_t)buf_id;
    h *= 0x01000193;
    h ^= poly_arg_hash(u->arg);
    h *= 0x01000193;
  } else {
    h ^= (uint32_t)u->op;
    h *= 0x01000193;
    h ^= (uint32_t)u->dtype.priority;
    h *= 0x01000193;
    h ^= (uint32_t)u->dtype.bitsize;
    h *= 0x01000193;
    for (int i = 0; i < u->n_src; i++) {
      h ^= struct_hash_impl(u->src[i], ctx);
      h *= 0x01000193;
    }
    h ^= poly_arg_hash(u->arg);
    h *= 0x01000193;
  }

  if (ctx->n_hashes >= ctx->cap_hashes) {
    int new_cap = ctx->cap_hashes ? ctx->cap_hashes * 2 : 1024;
    uint32_t *new_hashes = realloc(ctx->hashes, (size_t)new_cap * sizeof(uint32_t));
    if (!new_hashes) return h;
    ctx->hashes = new_hashes;
    ctx->cap_hashes = new_cap;
  }
  int idx = ctx->n_hashes++;
  ctx->hashes[idx] = h;
  poly_map_set(ctx->visited, poly_ptr_hash(u), u, (void *)(intptr_t)(idx + 1), poly_ptr_eq);
  return h;
}

uint32_t poly_structural_hash(PolyUOp *u) {
  if (!u) return 0;
  StructHashCtx ctx;
  memset(&ctx, 0, sizeof(ctx));
  ctx.visited = poly_map_new(1024);
  if (!ctx.visited) return 0;
  uint32_t h = struct_hash_impl(u, &ctx);
  poly_map_destroy(ctx.visited);
  free(ctx.hashes);
  free(ctx.bufs);
  return h;
}

/* Structural equality */

typedef struct {
  PolyMap *a_to_b;
  PolyMap *b_to_a;
} BufPairs;

typedef struct {
  PolyMap *a_to_b;
  PolyMap *b_to_a;
} EqVisited;

static bool struct_eq_impl(PolyUOp *a, PolyUOp *b, BufPairs *bp, EqVisited *ev) {
  if (a == b) return true;
  if (!a || !b) return false;

  /* Check if this pair already visited (DAG sharing) */
  void *seen_b = poly_map_get(ev->a_to_b, poly_ptr_hash(a), a, poly_ptr_eq);
  if (seen_b) return seen_b == b;
  if (poly_map_get(ev->b_to_a, poly_ptr_hash(b), b, poly_ptr_eq)) return false;

  poly_map_set(ev->a_to_b, poly_ptr_hash(a), a, b, poly_ptr_eq);
  poly_map_set(ev->b_to_a, poly_ptr_hash(b), b, a, poly_ptr_eq);

  /* Both executable storage identities? Track positional correspondence. */
  bool a_storage = a->op == POLY_OP_BUFFER || a->op == POLY_OP_BUFFER_VIEW;
  bool b_storage = b->op == POLY_OP_BUFFER || b->op == POLY_OP_BUFFER_VIEW;
  if (a_storage || b_storage) {
    if (!a_storage || !b_storage || a->op != b->op) return false;
    if (!poly_dtype_eq(a->dtype, b->dtype)) return false;
    if (!poly_arg_eq(a->arg, b->arg)) return false;
    /* Check existing mapping */
    void *mapped_b = poly_map_get(bp->a_to_b, poly_ptr_hash(a), a, poly_ptr_eq);
    if (mapped_b) return mapped_b == b;
    if (poly_map_get(bp->b_to_a, poly_ptr_hash(b), b, poly_ptr_eq)) return false;
    poly_map_set(bp->a_to_b, poly_ptr_hash(a), a, b, poly_ptr_eq);
    poly_map_set(bp->b_to_a, poly_ptr_hash(b), b, a, poly_ptr_eq);
    return true;
  }

  /* Same op, dtype, n_src, arg? */
  if (a->op != b->op) return false;
  if (!poly_dtype_eq(a->dtype, b->dtype)) return false;
  if (a->n_src != b->n_src) return false;
  if (!poly_arg_eq(a->arg, b->arg)) return false;

  /* Recursively compare sources */
  for (int i = 0; i < a->n_src; i++) {
    if (!struct_eq_impl(a->src[i], b->src[i], bp, ev)) return false;
  }
  return true;
}

bool poly_structural_eq(const void *a, const void *b) {
  BufPairs bp = {.a_to_b = poly_map_new(64), .b_to_a = poly_map_new(64)};
  EqVisited ev = {.a_to_b = poly_map_new(1024), .b_to_a = poly_map_new(1024)};
  if (!bp.a_to_b || !bp.b_to_a || !ev.a_to_b || !ev.b_to_a) {
    if (bp.a_to_b) poly_map_destroy(bp.a_to_b);
    if (bp.b_to_a) poly_map_destroy(bp.b_to_a);
    if (ev.a_to_b) poly_map_destroy(ev.a_to_b);
    if (ev.b_to_a) poly_map_destroy(ev.b_to_a);
    return false;
  }
  bool ok = struct_eq_impl((PolyUOp *)a, (PolyUOp *)b, &bp, &ev);
  poly_map_destroy(bp.a_to_b);
  poly_map_destroy(bp.b_to_a);
  poly_map_destroy(ev.a_to_b);
  poly_map_destroy(ev.b_to_a);
  return ok;
}

/* DFS to assign positional IDs to BUFFER/BUFFER_VIEW or shaped call PARAM
 * storage identities, matching source traversal order.
 * Children are visited left-to-right, same as struct_hash_impl().
 * n_bufs counts total BUFFERs found (may exceed buf_order capacity).
 * buf_order is only written up to POLY_MAX_REALIZE_BUFS entries.
 * Callers must check *n_bufs <= POLY_MAX_REALIZE_BUFS after the call. */
void poly_collect_buf_order(
    PolyUOp *u,
    PolyUOp **buf_order,
    int *n_bufs,
    PolyUOp **visited,
    int *n_visited
) {
  if (!u || !n_bufs || !n_visited) return;

  /* Model-scale graphs exceed POLY_MAX_STRUCT_NODES. Keep the public scratch
   * arrays for ABI compatibility, but use a dynamic visited map so traversal
   * stays O(nodes) instead of revisiting shared DAG tails after the cap. */
  PolyMap *seen = poly_map_new(1024);
  if (!seen) return;
  int initial_visited = *n_visited;
  for (int i = 0; visited && i < initial_visited && i < POLY_MAX_STRUCT_NODES; i++) {
    if (visited[i])
      poly_map_set(seen, poly_ptr_hash(visited[i]), visited[i], visited[i], poly_ptr_eq);
  }

  int cap = 1024;
  int sp = 0;
  PolyUOp **stack = malloc((size_t)cap * sizeof(PolyUOp *));
  if (!stack) {
    poly_map_destroy(seen);
    return;
  }
  stack[sp++] = u;

  while (sp > 0) {
    PolyUOp *cur = stack[--sp];
    if (!cur) continue;
    if (poly_map_get(seen, poly_ptr_hash(cur), cur, poly_ptr_eq)) continue;
    poly_map_set(seen, poly_ptr_hash(cur), cur, cur, poly_ptr_eq);
    if (visited && *n_visited < POLY_MAX_STRUCT_NODES) visited[*n_visited] = cur;
    (*n_visited)++;

    if (cur->op == POLY_OP_BUFFER || cur->op == POLY_OP_BUFFER_VIEW ||
        poly_uop_is_shaped_value_param(cur)) {
      if (buf_order && *n_bufs < POLY_MAX_REALIZE_BUFS) buf_order[*n_bufs] = cur;
      (*n_bufs)++; /* always count, even past capacity */
      continue;
    }

    if (sp + cur->n_src > cap) {
      int new_cap = cap;
      while (sp + cur->n_src > new_cap)
        new_cap *= 2;
      PolyUOp **new_stack = realloc(stack, (size_t)new_cap * sizeof(PolyUOp *));
      if (!new_stack) break;
      stack = new_stack;
      cap = new_cap;
    }
    for (int i = cur->n_src - 1; i >= 0; i--)
      stack[sp++] = cur->src[i];
  }

  free(stack);
  poly_map_destroy(seen);
}

bool poly_collect_buf_order_alloc(
    PolyUOp *u,
    PolyUOp ***out_buf_order,
    int *out_n_bufs,
    int *out_n_visited
) {
  if (!u || !out_buf_order || !out_n_bufs || !out_n_visited) return false;
  *out_buf_order = NULL;
  *out_n_bufs = 0;
  *out_n_visited = 0;

  PolyMap *seen = poly_map_new(1024);
  if (!seen) return false;

  int stack_cap = 1024;
  int sp = 0;
  PolyUOp **stack = malloc((size_t)stack_cap * sizeof(PolyUOp *));
  if (!stack) {
    poly_map_destroy(seen);
    return false;
  }
  stack[sp++] = u;

  PolyUOp **buf_order = NULL;
  int n_bufs = 0, buf_cap = 0;
  bool ok = true;

  while (ok && sp > 0) {
    PolyUOp *cur = stack[--sp];
    if (!cur) continue;
    if (poly_map_get(seen, poly_ptr_hash(cur), cur, poly_ptr_eq)) continue;
    poly_map_set(seen, poly_ptr_hash(cur), cur, cur, poly_ptr_eq);
    (*out_n_visited)++;

    if (cur->op == POLY_OP_BUFFER || cur->op == POLY_OP_BUFFER_VIEW ||
        poly_uop_is_shaped_value_param(cur)) {
      ok = uop_vec_append(&buf_order, &n_bufs, &buf_cap, cur);
      continue;
    }

    if (sp + cur->n_src > stack_cap) {
      int new_cap = stack_cap;
      while (sp + cur->n_src > new_cap)
        new_cap *= 2;
      PolyUOp **new_stack = realloc(stack, (size_t)new_cap * sizeof(PolyUOp *));
      if (!new_stack) {
        ok = false;
        break;
      }
      stack = new_stack;
      stack_cap = new_cap;
    }
    for (int i = cur->n_src - 1; i >= 0; i--)
      stack[sp++] = cur->src[i];
  }

  free(stack);
  poly_map_destroy(seen);
  if (!ok) {
    free(buf_order);
    return false;
  }
  *out_buf_order = buf_order;
  *out_n_bufs = n_bufs;
  return true;
}

int poly_find_buf_position(PolyUOp *buf, PolyUOp **buf_order, int n_bufs) {
  for (int i = 0; i < n_bufs; i++)
    if (buf_order[i] == buf) return i;
  return -1;
}

/* Helpers shared by all builds (exec_plan + realize) */

/* Structural ownership/source validation plus the INDEX-coordinate predicate
 * from pinned tinygrad/uop/spec.py:77. This is not a complete spec_tensor
 * implementation. */
bool poly_validate_kernel_graph(PolyCtx *ctx, PolyUOp *root) {
  if (!root) return false;
  PolyMap *visited = poly_map_new(256);
  int stack_cap = 1024;
  PolyUOp **stack = malloc((size_t)stack_cap * sizeof(PolyUOp *));
  PolyUOp **parent_stack = malloc((size_t)stack_cap * sizeof(PolyUOp *));
  int *parent_src_idx = malloc((size_t)stack_cap * sizeof(int));
  if (!visited || !stack || !parent_stack || !parent_src_idx) {
    if (visited) poly_map_destroy(visited);
    free(stack);
    free(parent_stack);
    free(parent_src_idx);
    return false;
  }
  int sp = 0;
  stack[sp++] = root;
  parent_stack[0] = NULL;
  parent_src_idx[0] = -1;

  while (sp > 0) {
    sp--;
    PolyUOp *u = stack[sp];
    PolyUOp *parent = parent_stack[sp];
    int src_idx = parent_src_idx[sp];
    if (!u) {
      free(stack);
      free(parent_stack);
      free(parent_src_idx);
      poly_map_destroy(visited);
      return false;
    }
    if (!poly_ctx_owns_ptr(ctx, u)) {
      if (parent) {
        fprintf(
            stderr,
            "polygrad: realize: foreign/stale UOp pointer %p referenced by %s(%p) src[%d]\n",
            (void *)u, poly_op_name(parent->op), (void *)parent, src_idx
        );
        fprintf(
            stderr, "polygrad: realize: parent %s n_src=%d\n", poly_op_name(parent->op),
            parent->n_src
        );
        for (int si = 0; si < parent->n_src; si++) {
          PolyUOp *ps = parent->src[si];
          bool owned = poly_ctx_owns_ptr(ctx, ps);
          fprintf(
              stderr, "  parent.src[%d]=%p %s%s\n", si, (void *)ps, owned ? "" : "[FOREIGN] ",
              (owned && ps) ? poly_op_name(ps->op) : ""
          );
        }
      } else {
        fprintf(
            stderr, "polygrad: realize: foreign/stale root UOp pointer %p in kernel graph\n",
            (void *)u
        );
      }
      free(stack);
      free(parent_stack);
      free(parent_src_idx);
      poly_map_destroy(visited);
      return false;
    }
    if (poly_map_get(visited, poly_ptr_hash(u), u, poly_ptr_eq)) continue;
    poly_map_set(visited, poly_ptr_hash(u), u, u, poly_ptr_eq);

    if (u->n_src > 64) {
      fprintf(
          stderr, "polygrad: realize: invalid n_src=%d on %s(%p)\n", u->n_src, poly_op_name(u->op),
          (void *)u
      );
      free(stack);
      free(parent_stack);
      free(parent_src_idx);
      poly_map_destroy(visited);
      return false;
    }
    if (u->op == POLY_OP_INDEX) {
      for (int i = 1; i < u->n_src; i++) {
        if (u->src[i] && !poly_dtype_is_int(u->src[i]->dtype)) {
          fprintf(
              stderr,
              "polygrad: codegen: INDEX src[%d] must have integer dtype, got %s\n",
              i, u->src[i]->dtype.name ? u->src[i]->dtype.name : "unknown"
          );
          free(stack);
          free(parent_stack);
          free(parent_src_idx);
          poly_map_destroy(visited);
          return false;
        }
      }
    }
    for (int i = 0; i < u->n_src; i++) {
      if (!u->src[i]) {
        fprintf(
            stderr, "polygrad: realize: NULL src[%d] on %s(%p), n_src=%d\n", i, poly_op_name(u->op),
            (void *)u, u->n_src
        );
        free(stack);
        free(parent_stack);
        free(parent_src_idx);
        poly_map_destroy(visited);
        return false;
      }
      if (sp >= stack_cap) {
        int new_cap = stack_cap * 2;
        PolyUOp **new_stack = realloc(stack, (size_t)new_cap * sizeof(PolyUOp *));
        PolyUOp **new_parent_stack = realloc(parent_stack, (size_t)new_cap * sizeof(PolyUOp *));
        int *new_parent_src_idx = realloc(parent_src_idx, (size_t)new_cap * sizeof(int));
        if (!new_stack || !new_parent_stack || !new_parent_src_idx) {
          free(new_stack ? new_stack : stack);
          free(new_parent_stack ? new_parent_stack : parent_stack);
          free(new_parent_src_idx ? new_parent_src_idx : parent_src_idx);
          poly_map_destroy(visited);
          return false;
        }
        stack = new_stack;
        parent_stack = new_parent_stack;
        parent_src_idx = new_parent_src_idx;
        stack_cap = new_cap;
      }
      stack[sp++] = u->src[i];
      parent_stack[sp - 1] = u;
      parent_src_idx[sp - 1] = i;
    }
  }
  free(stack);
  free(parent_stack);
  free(parent_src_idx);
  poly_map_destroy(visited);
  return true;
}

/* POLY_SCHED_CACHE_VERSION defined in frontend_internal.h */

/* CPU realize (not available in Emscripten) */

int poly_collect_output_buffers_in_sink(PolyUOp *tensor_sink, PolyUOp **out, int cap) {
  if (!tensor_sink || tensor_sink->op != POLY_OP_SINK) return 0;
  int n_seen = 0;
  for (int i = 0; i < tensor_sink->n_src; i++) {
    PolyUOp *store = tensor_sink->src[i];
    if (!store || store->op != POLY_OP_STORE || store->n_src < 1) continue;
    PolyUOp *buf = store->src[0];
    if (!buf || (buf->op != POLY_OP_BUFFER && buf->op != POLY_OP_BUFFER_VIEW)) continue;
    bool dup = false;
    for (int j = 0; j < n_seen; j++) {
      if (out[j] == buf) {
        dup = true;
        break;
      }
    }
    if (!dup && n_seen < cap) out[n_seen++] = buf;
  }
  return n_seen;
}

void poly_cpu_cache_flush(void) {
  /* Retained for ABI/frontend cleanup paths. CPU program caches are per-context
   * now and are released through poly_ctx_destroy(). */
}

/* Exec plan functions (poly_complete_create_schedule_with_vars, poly_run_schedule,
 * backend lowering, etc.) live in engine/schedule.c. Frontend execution now
 * reaches them only through graph/tensor realize entrypoints. */

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
