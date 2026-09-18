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
#include "tensor.h"

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

PolyUOp *poly_uop_buffer_by_id(PolyCtx *ctx, int dtype_id, int64_t size) {
  PolyDType dt;
  if (!poly_dtype_by_id(dtype_id, &dt)) return NULL;
  return frontend_new_buffer(ctx, dt, size, POLY_DEVICE_AUTO);
}

PolyUOp *poly_uop_buffer_on_device_by_id(PolyCtx *ctx, int dtype_id, int64_t size, int device_id) {
  PolyDType dt;
  if (!poly_dtype_by_id(dtype_id, &dt)) return NULL;
  PolyDevice device = (PolyDevice)device_id;
  if (!poly_device_can_execute(device)) return NULL;
  return frontend_new_buffer(ctx, dt, size, device);
}

PolyUOp *poly_uop_buffer_f32(PolyCtx *ctx, int64_t size) {
  return frontend_new_buffer(ctx, POLY_FLOAT32, size, POLY_DEVICE_AUTO);
}

PolyUOp *poly_uop_buffer_f64(PolyCtx *ctx, int64_t size) {
  return frontend_new_buffer(ctx, POLY_FLOAT64, size, POLY_DEVICE_AUTO);
}

PolyUOp *poly_uop_const_int_decimal(PolyCtx *ctx, const char *value) {
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

PolyTensor *poly_tensor_empty_uop_name_by_id(
    PolyCtx *ctx,
    int dtype_id,
    PolyUOp **dims,
    int ndim,
    const char *device
) {
  PolyDType dt;
  if (!poly_dtype_by_id(dtype_id, &dt)) return NULL;
  return poly_tensor_empty_uop_name(ctx, dt, dims, ndim, device);
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
  PolyUOp *value_uop = poly_uop_const_int_by_id(ctx, value, dtype_id);
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
  PolyUOp *uop = poly_uop_const_uint_by_id(ctx, value, dtype_id);
  return uop ? poly_tensor_create_with_roots(
                   ctx, uop, uop, POLY_TENSOR_VALUE, (PolyDevice)device_id
               )
             : NULL;
}

PolyTensor *poly_tensor_const_float_by_id(PolyCtx *ctx, double value, int dtype_id, int device_id) {
  PolyUOp *value_uop = poly_uop_const_float_by_id(ctx, value, dtype_id);
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
  PolyUOp *value_uop = poly_uop_full_int_by_id(ctx, dims, ndim, value, dtype_id);
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
  PolyUOp *uop = poly_uop_full_uint_by_id(ctx, dims, ndim, value, dtype_id);
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
  PolyUOp *value = poly_uop_full_invalid_by_id(ctx, dims, ndim, dtype_id);
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
  PolyUOp *value_uop = poly_uop_full_float_by_id(ctx, dims, ndim, value, dtype_id);
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
  PolyUOp *value_uop = poly_uop_arange_int_by_id(ctx, start, stop, step, dtype_id);
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
  PolyUOp *value_uop = poly_uop_arange_float_by_id(ctx, start, stop, step, dtype_id);
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
  PolyUOp *value_uop = poly_uop_linspace_by_id(ctx, start, stop, steps, dtype_id);
  if (!value_uop) return NULL;
  return poly_tensor_create_with_roots(
      ctx, value_uop, value_uop, POLY_TENSOR_VALUE, (PolyDevice)device_id
  );
}

PolyTensor *poly_tensor_eye_by_id(PolyCtx *ctx, int64_t n, int64_t m, int dtype_id, int device_id) {
  PolyUOp *value_uop = poly_uop_eye_by_id(ctx, n, m, dtype_id);
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

PolyUOp *poly_uop_cast_by_id(PolyCtx *ctx, PolyUOp *x, int dtype_id) {
  PolyDType target;
  return poly_dtype_by_id(dtype_id, &target) ? poly_uop_cast(ctx, x, target) : NULL;
}

PolyUOp *poly_uop_bitcast_by_id(PolyCtx *ctx, PolyUOp *x, int dtype_id) {
  PolyDType target;
  if (!ctx || !x || !poly_dtype_by_id(dtype_id, &target)) return NULL;
  return poly_uop_bitcast(ctx, x, target);
}

PolyUOp *poly_uop_range_by_id(PolyCtx *ctx, int64_t bound, int64_t axis_id, int axis_type) {
  return axis_type >= POLY_AXIS_DEVICE && axis_type <= POLY_AXIS_LOOP
             ? poly_uop_range(ctx, bound, axis_id, (PolyAxisType)axis_type)
             : NULL;
}

/* Dtype-ID conversion only; construction policy belongs to tensor.c. */
PolyUOp *poly_uop_const_int_by_id(PolyCtx *ctx, int64_t value, int dtype_id) {
  PolyDType dtype;
  return poly_dtype_by_id(dtype_id, &dtype) ? poly_uop_const_int_dtype(ctx, value, dtype) : NULL;
}

PolyUOp *poly_uop_const_uint_by_id(PolyCtx *ctx, uint64_t value, int dtype_id) {
  PolyDType dtype;
  return poly_dtype_by_id(dtype_id, &dtype) ? poly_uop_const_uint_dtype(ctx, value, dtype) : NULL;
}

PolyUOp *poly_uop_const_float_by_id(PolyCtx *ctx, double value, int dtype_id) {
  PolyDType dtype;
  return poly_dtype_by_id(dtype_id, &dtype) ? poly_uop_const_float_dtype(ctx, value, dtype) : NULL;
}

PolyUOp *poly_uop_full_int_by_id(
    PolyCtx *ctx,
    const int64_t *shape,
    int ndim,
    int64_t fill_value,
    int dtype_id
) {
  PolyDType dtype;
  return poly_dtype_by_id(dtype_id, &dtype)
             ? poly_uop_full_int_dtype(ctx, shape, ndim, fill_value, dtype)
             : NULL;
}

PolyUOp *poly_uop_full_uint_by_id(
    PolyCtx *ctx,
    const int64_t *shape,
    int ndim,
    uint64_t value,
    int dtype_id
) {
  PolyDType dtype;
  return poly_dtype_by_id(dtype_id, &dtype)
             ? poly_uop_full_uint_dtype(ctx, shape, ndim, value, dtype)
             : NULL;
}

PolyUOp *poly_uop_full_float_by_id(
    PolyCtx *ctx,
    const int64_t *shape,
    int ndim,
    double fill_value,
    int dtype_id
) {
  PolyDType dtype;
  return poly_dtype_by_id(dtype_id, &dtype)
             ? poly_uop_full_float_dtype(ctx, shape, ndim, fill_value, dtype)
             : NULL;
}

PolyUOp *poly_uop_full_invalid_by_id(PolyCtx *ctx, const int64_t *shape, int ndim, int dtype_id) {
  PolyDType dtype;
  return poly_dtype_by_id(dtype_id, &dtype) ? poly_uop_full_invalid_dtype(ctx, shape, ndim, dtype)
                                            : NULL;
}

PolyUOp *poly_uop_arange_int_by_id(
    PolyCtx *ctx,
    int64_t start,
    int64_t stop,
    int64_t step,
    int dtype_id
) {
  PolyDType dtype;
  return poly_dtype_by_id(dtype_id, &dtype)
             ? poly_uop_arange_int_dtype(ctx, start, stop, step, dtype)
             : NULL;
}

PolyUOp *poly_uop_arange_float_by_id(
    PolyCtx *ctx,
    double start,
    double stop,
    double step,
    int dtype_id
) {
  PolyDType dtype;
  return poly_dtype_by_id(dtype_id, &dtype)
             ? poly_uop_arange_float_dtype(ctx, start, stop, step, dtype)
             : NULL;
}

PolyUOp *poly_uop_linspace_by_id(
    PolyCtx *ctx,
    double start,
    double stop,
    int64_t steps,
    int dtype_id
) {
  PolyDType dtype;
  return poly_dtype_by_id(dtype_id, &dtype)
             ? poly_uop_linspace_dtype(ctx, start, stop, steps, dtype)
             : NULL;
}

PolyUOp *poly_uop_eye_by_id(PolyCtx *ctx, int64_t n, int64_t m, int dtype_id) {
  PolyDType dtype;
  return poly_dtype_by_id(dtype_id, &dtype) ? poly_uop_eye_dtype(ctx, n, m, dtype) : NULL;
}

PolyUOp *poly_uop_rand_by_id(
    PolyCtx *ctx,
    const int64_t *shape,
    int ndim,
    uint64_t seed,
    int dtype_id
) {
  PolyDType dtype;
  return poly_dtype_by_id(dtype_id, &dtype) ? poly_uop_rand_dtype(ctx, shape, ndim, seed, dtype)
                                            : NULL;
}

PolyUOp *poly_uop_randn_by_id(
    PolyCtx *ctx,
    const int64_t *shape,
    int ndim,
    uint64_t seed,
    int dtype_id
) {
  PolyDType dtype;
  return poly_dtype_by_id(dtype_id, &dtype) ? poly_uop_randn_dtype(ctx, shape, ndim, seed, dtype)
                                            : NULL;
}

int poly_can_run_op(
    PolyCtx *ctx,
    int device_id,
    const char *op,
    int dtype_id,
    const int64_t *shape,
    int n_shape
) {
  PolyDType dtype;
  if (!poly_dtype_by_id(dtype_id, &dtype)) return -1;
  return poly_can_compile_op(ctx, (PolyDevice)device_id, op, dtype, shape, n_shape);
}

PolyTensor *poly_tensor_sum_dtype_by_id(
    PolyCtx *ctx,
    PolyTensor *src,
    int64_t *axes,
    int n_axes,
    bool keepdim,
    int dtype_id
) {
  PolyDType dtype;
  if (!poly_dtype_by_id(dtype_id, &dtype)) return NULL;
  return poly_tensor_sum_dtype(ctx, src, axes, n_axes, keepdim, &dtype);
}

PolyTensor *poly_tensor_dot_dtype_by_id(
    PolyCtx *ctx,
    PolyTensor *src,
    PolyTensor *weight,
    int dtype_id
) {
  PolyDType dtype;
  if (!poly_dtype_by_id(dtype_id, &dtype)) return NULL;
  return poly_tensor_dot_dtype(ctx, src, weight, &dtype);
}

PolyTensor *poly_tensor_conv2d_dtype_by_id(
    PolyCtx *ctx,
    PolyTensor *src,
    PolyTensor *weight,
    PolyTensor *bias,
    int groups,
    const int64_t *stride,
    const int64_t *dilation,
    const int64_t *padding,
    int n_padding,
    int dtype_id
) {
  PolyDType dtype;
  if (!poly_dtype_by_id(dtype_id, &dtype)) return NULL;
  return poly_tensor_conv2d_dtype(
      ctx, src, weight, bias, groups, stride, dilation, padding, n_padding, &dtype
  );
}

PolyTensor *poly_tensor_cast_by_id(PolyCtx *ctx, PolyTensor *src, int dtype_id) {
  PolyDType dtype;
  if (!poly_dtype_by_id(dtype_id, &dtype)) return NULL;
  return poly_tensor_cast(ctx, src, dtype);
}

PolyTensor *poly_tensor_bitcast_by_id(PolyCtx *ctx, PolyTensor *src, int dtype_id) {
  PolyDType dtype;
  if (!poly_dtype_by_id(dtype_id, &dtype)) return NULL;
  return poly_tensor_bitcast(ctx, src, dtype);
}

PolyTensor *poly_tensor_rand_by_id(
    PolyCtx *ctx,
    const int64_t *dims,
    int ndim,
    int dtype_id,
    PolyDevice device,
    int contiguous
) {
  PolyDType dtype;
  if (!poly_dtype_by_id(dtype_id, &dtype)) return NULL;
  return poly_tensor_rand(ctx, dims, ndim, dtype, device, contiguous);
}

PolyTensor *poly_tensor_randn_by_id(
    PolyCtx *ctx,
    const int64_t *dims,
    int ndim,
    int dtype_id,
    PolyDevice device
) {
  PolyDType dtype;
  if (!poly_dtype_by_id(dtype_id, &dtype)) return NULL;
  return poly_tensor_randn(ctx, dims, ndim, dtype, device);
}

PolyUOp *poly_register_buffer_by_id(
    PolyCtx *ctx,
    int role,
    int dtype_id,
    const int64_t *shape,
    int ndim,
    const char *name
) {
  PolyDType dt;
  if (!poly_dtype_by_id(dtype_id, &dt)) return NULL;
  return poly_register_buffer(ctx, role, dt, shape, ndim, name);
}
