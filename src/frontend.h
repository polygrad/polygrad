/* frontend.h -- Thin dtype-ID, flattened-argument and opaque-handle adapters. */
#ifndef POLY_FRONTEND_H
#define POLY_FRONTEND_H

#include "core.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Dtype-ID adapters for bindings that cannot pass PolyDType by value. */
PolyUOp *poly_uop_cast_by_id(PolyCtx *ctx, PolyUOp *x, int dtype_id);
PolyUOp *poly_uop_bitcast_by_id(PolyCtx *ctx, PolyUOp *x, int dtype_id);
PolyUOp *poly_uop_buffer_by_id(PolyCtx *ctx, int dtype_id, int64_t size);
PolyUOp *poly_uop_buffer_on_device_by_id(PolyCtx *ctx, int dtype_id, int64_t size, int device_id);
PolyUOp *poly_uop_buffer_f32(PolyCtx *ctx, int64_t size);
PolyUOp *poly_uop_buffer_f64(PolyCtx *ctx, int64_t size);
/* Lossless Python int/JS BigInt adaptation to the existing weakint CONST. */
PolyUOp *poly_uop_const_int_decimal(PolyCtx *ctx, const char *value);
/* Scalar CONST endpoints avoid exposing PolyArg's tagged union through FFI. */
PolyUOp *poly_uop_variable_by_id(
    PolyCtx *ctx,
    const char *name,
    PolyUOp *min_val,
    PolyUOp *max_val,
    int dtype_id,
    int64_t multiple_of,
    bool param
);
PolyTensor *poly_tensor_empty_by_id(
    PolyCtx *ctx,
    int dtype_id,
    const int64_t *dims,
    int ndim,
    int device_id
);
PolyTensor *poly_tensor_empty_uop_by_id(
    PolyCtx *ctx,
    int dtype_id,
    PolyUOp **dims,
    int ndim,
    int device_id
);
PolyTensor *poly_tensor_empty_uop_name_by_id(
    PolyCtx *ctx,
    int dtype_id,
    PolyUOp **dims,
    int ndim,
    const char *device
);
PolyTensor *poly_tensor_from_host_by_id(
    PolyCtx *ctx,
    void *ptr,
    size_t nbytes,
    int dtype_id,
    const int64_t *dims,
    int ndim
);
PolyTensor *poly_tensor_const_int_by_id(PolyCtx *ctx, int64_t value, int dtype_id, int device_id);
PolyTensor *poly_tensor_const_float_by_id(PolyCtx *ctx, double value, int dtype_id, int device_id);

PolyTensor *poly_tensor_full_invalid_by_id(
    PolyCtx *ctx,
    const int64_t *dims,
    int ndim,
    int dtype_id,
    int device_id,
    bool buffer
);
PolyTensor *poly_tensor_const_uint_by_id(PolyCtx *ctx, uint64_t value, int dtype_id, int device_id);
PolyTensor *poly_tensor_full_uint_by_id(
    PolyCtx *ctx,
    const int64_t *dims,
    int ndim,
    uint64_t value,
    int dtype_id,
    int device_id,
    bool dtype_explicit,
    bool buffer
);
PolyTensor *poly_tensor_full_int_by_id(
    PolyCtx *ctx,
    const int64_t *dims,
    int ndim,
    int64_t value,
    int dtype_id,
    int device_id,
    bool dtype_explicit,
    bool buffer
);
PolyTensor *poly_tensor_full_float_by_id(
    PolyCtx *ctx,
    const int64_t *dims,
    int ndim,
    double value,
    int dtype_id,
    int device_id,
    bool dtype_explicit,
    bool buffer
);
PolyTensor *poly_tensor_arange_int_by_id(
    PolyCtx *ctx,
    int64_t start,
    int64_t stop,
    int64_t step,
    int dtype_id,
    int device_id
);
PolyTensor *poly_tensor_arange_float_by_id(
    PolyCtx *ctx,
    double start,
    double stop,
    double step,
    int dtype_id,
    int device_id
);
PolyTensor *poly_tensor_linspace_by_id(
    PolyCtx *ctx,
    double start,
    double stop,
    int64_t steps,
    int dtype_id,
    int device_id
);
PolyTensor *poly_tensor_eye_by_id(PolyCtx *ctx, int64_t n, int64_t m, int dtype_id, int device_id);
/* Raw creation adapters. Graph semantics live in the typed Tensor core. */
PolyUOp *poly_uop_const_int_by_id(PolyCtx *ctx, int64_t value, int dtype_id);
PolyUOp *poly_uop_const_uint_by_id(PolyCtx *ctx, uint64_t value, int dtype_id);
PolyUOp *poly_uop_const_float_by_id(PolyCtx *ctx, double value, int dtype_id);
PolyUOp *poly_uop_full_int_by_id(
    PolyCtx *ctx,
    const int64_t *shape,
    int ndim,
    int64_t fill_value,
    int dtype_id
);
PolyUOp *poly_uop_full_uint_by_id(
    PolyCtx *ctx,
    const int64_t *shape,
    int ndim,
    uint64_t value,
    int dtype_id
);
PolyUOp *poly_uop_full_float_by_id(
    PolyCtx *ctx,
    const int64_t *shape,
    int ndim,
    double fill_value,
    int dtype_id
);
PolyUOp *poly_uop_full_invalid_by_id(PolyCtx *ctx, const int64_t *shape, int ndim, int dtype_id);
PolyUOp *poly_uop_arange_int_by_id(
    PolyCtx *ctx,
    int64_t start,
    int64_t stop,
    int64_t step,
    int dtype_id
);
PolyUOp *poly_uop_arange_float_by_id(
    PolyCtx *ctx,
    double start,
    double stop,
    double step,
    int dtype_id
);
PolyUOp *poly_uop_linspace_by_id(
    PolyCtx *ctx,
    double start,
    double stop,
    int64_t steps,
    int dtype_id
);
PolyUOp *poly_uop_eye_by_id(PolyCtx *ctx, int64_t n, int64_t m, int dtype_id);
PolyUOp *poly_uop_rand_by_id(
    PolyCtx *ctx,
    const int64_t *shape,
    int ndim,
    uint64_t seed,
    int dtype_id
);
PolyUOp *poly_uop_randn_by_id(
    PolyCtx *ctx,
    const int64_t *shape,
    int ndim,
    uint64_t seed,
    int dtype_id
);

/* UOp field inspection for opaque-handle language bindings. */
int poly_uop_op(PolyUOp *u);
int poly_uop_dtype_id(PolyCtx *ctx, PolyUOp *u);
int poly_uop_n_src(PolyUOp *u);
PolyUOp *poly_uop_src(PolyUOp *u, int idx);
uint32_t poly_uop_call_grad_fxn_key(PolyUOp *u);
PolyUOp *poly_uop_range_by_id(PolyCtx *ctx, int64_t bound, int64_t axis_id, int axis_type);

/* FFI capability probe. Builds a representative tensor graph for `op` and
 * `shape`, then asks the selected backend to lower it. Returns 1 for supported,
 * 0 for unsupported, and a negative value for an invalid/too-large query.
 *
 * Shape conventions:
 * - elementwise/unary/reduce/qr/cholesky: input shape
 * - matmul/dot: [m, k, n] for (m,k) @ (k,n)
 * - triangular_solve/solve: [n, n] for vector RHS or [n, n, nrhs]
 * - lstsq: [m, n] for vector RHS or [m, n, nrhs]
 */
int poly_can_run_op(
    PolyCtx *ctx,
    int device_id,
    const char *op,
    int dtype_id,
    const int64_t *shape,
    int n_shape
);

PolyTensor *poly_tensor_rand_by_id(
    PolyCtx *ctx,
    const int64_t *dims,
    int ndim,
    int dtype_id,
    PolyDevice device,
    int contiguous
);

PolyTensor *poly_tensor_randn_by_id(
    PolyCtx *ctx,
    const int64_t *dims,
    int ndim,
    int dtype_id,
    PolyDevice device
);

PolyTensor *poly_tensor_sum_dtype_by_id(
    PolyCtx *ctx,
    PolyTensor *src,
    int64_t *axes,
    int n_axes,
    bool keepdim,
    int dtype_id
);

PolyTensor *poly_tensor_dot_dtype_by_id(
    PolyCtx *ctx,
    PolyTensor *src,
    PolyTensor *weight,
    int dtype_id
);

PolyTensor *poly_tensor_cast_by_id(PolyCtx *ctx, PolyTensor *src, int dtype_id);

PolyTensor *poly_tensor_bitcast_by_id(PolyCtx *ctx, PolyTensor *src, int dtype_id);

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
);

/* Non-variadic named-buffer registration for FFI/frontends. These mirror
 * poly_param/poly_input/poly_output/poly_target/poly_aux, but accept an exact
 * string and dtype id so Python/JS/WASM do not need to call variadic C APIs. */
PolyUOp *poly_register_buffer_by_id(
    PolyCtx *ctx,
    int role,
    int dtype_id,
    const int64_t *shape,
    int ndim,
    const char *name
) POLY_DEPRECATED("use poly_model_from_binding_arrays");

#ifdef __cplusplus
}
#endif

#endif
