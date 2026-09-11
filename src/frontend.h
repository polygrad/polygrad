/*
 * frontend.h — FFI-friendly helpers for language bindings
 *
 * Provides a simplified C surface that avoids passing PolyArg (tagged union)
 * and PolyDType (struct) across FFI boundaries. All functions take only
 * opaque pointers, integers, and doubles.
 */

#ifndef POLY_FRONTEND_H
#define POLY_FRONTEND_H

#include "polygrad.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Dtype-ID adapters for bindings that cannot pass PolyDType by value. */
PolyUOp *poly_cast_by_id(PolyCtx *ctx, PolyUOp *x, int dtype_id);
PolyUOp *poly_bitcast_by_id(PolyCtx *ctx, PolyUOp *x, int dtype_id);
PolyUOp *poly_buffer_by_id(PolyCtx *ctx, int dtype_id, int64_t size);
PolyUOp *poly_buffer_on_device_by_id(PolyCtx *ctx, int dtype_id, int64_t size, int device_id);
PolyUOp *poly_buffer_f32(PolyCtx *ctx, int64_t size);
PolyUOp *poly_buffer_f64(PolyCtx *ctx, int64_t size);
PolyUOp *poly_uop_variable_by_id(
    PolyCtx *ctx,
    const char *name,
    int64_t min_val,
    int64_t max_val,
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
PolyUOp *poly_full_invalid_by_id(PolyCtx *ctx, const int64_t *dims, int ndim, int dtype_id);
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
int poly_uop_op(PolyUOp *u);
int poly_uop_dtype_id(PolyCtx *ctx, PolyUOp *u);
int poly_uop_n_src(PolyUOp *u);
PolyUOp *poly_uop_src(PolyUOp *u, int idx);
uint32_t poly_uop_call_grad_fxn_key(PolyUOp *u);
PolyUOp *poly_uop_range(PolyCtx *ctx, int64_t bound, int64_t axis_id, int axis_type);

/* ABI version (callers check at load time for compatibility). */
int poly_abi_version(void);

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

/* Debug: print UOp info to stderr */
void poly_debug_uop(PolyCtx *ctx, PolyUOp *u);
void poly_debug_opsets(void);

/* Cache cleanup (for leak-free shutdown) */

/* ABI cleanup hook retained for frontends. CPU program caches are per-context
 * and are released by poly_ctx_destroy(). */
void poly_cpu_cache_flush(void);

#ifdef __cplusplus
}
#endif

#endif /* POLY_FRONTEND_H */
