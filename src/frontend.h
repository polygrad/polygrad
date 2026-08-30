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

/* Public C/frontend ABI version. Bump when exported symbols or public struct
 * layouts used by frontends change. */
#define POLYGRAD_ABI_VERSION 63

/* Current ElementwiseMixin._binop for language UOp operators. Compiler
 * matchers keep using raw poly_alu2. */
PolyUOp *poly_binop(PolyCtx *ctx, PolyOps op, PolyUOp *a, PolyUOp *b);

#ifdef __cplusplus
extern "C" {
#endif

/* FFI buffer constructors for bindings that cannot pass PolyDType by value. */
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
PolyTensor *poly_tensor_const_like_int(PolyCtx *ctx, PolyTensor *ref, int64_t value);
PolyTensor *poly_tensor_const_like_float(PolyCtx *ctx, PolyTensor *ref, double value);
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
void poly_tensor_manual_seed(PolyCtx *ctx, int64_t seed);
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
int poly_uop_op(PolyUOp *u);
int poly_uop_dtype_id(PolyCtx *ctx, PolyUOp *u);
int poly_uop_n_src(PolyUOp *u);
PolyUOp *poly_uop_src(PolyUOp *u, int idx);
/* Pinned tinygrad uop/ops.py `resolve`: simplify a boolean UOp, return its
 * proven value when constant, otherwise the caller-provided default. */
int poly_uop_resolve(PolyCtx *ctx, PolyUOp *u, int default_value);

/* FFI-safe UOp construction helpers used by tinygrad-style custom kernels.
 * These only build UOps; execution still flows through normal CALL scheduling. */
PolyUOp *poly_uop_placeholder_like(PolyCtx *ctx, PolyUOp *like, int slot);
PolyUOp *poly_uop_range(PolyCtx *ctx, int64_t bound, int64_t axis_id, int axis_type);
PolyUOp *poly_uop_index(
    PolyCtx *ctx,
    PolyUOp *base,
    PolyUOp **indices,
    int n_indices
);
PolyUOp *poly_uop_load(PolyCtx *ctx, PolyUOp *addr);
PolyUOp *poly_uop_store(PolyCtx *ctx, PolyUOp *addr, PolyUOp *value);
PolyUOp *poly_uop_set(PolyCtx *ctx, PolyUOp *addr, PolyUOp *value, PolyUOp **ranges, int n_ranges);
PolyUOp *poly_uop_group(PolyCtx *ctx, PolyUOp **srcs, int n_src);
PolyUOp *poly_uop_end(PolyCtx *ctx, PolyUOp *body, PolyUOp **ranges, int n_ranges);
PolyUOp *poly_uop_sink(PolyCtx *ctx, PolyUOp **srcs, int n_src);
PolyUOp *poly_uop_sink_ex(PolyCtx *ctx, PolyUOp **srcs, int n_src, const char *name, int optimize);
PolyUOp *poly_uop_call(PolyCtx *ctx, PolyUOp *body, PolyUOp **args, int n_args);
PolyUOp *poly_uop_after(PolyCtx *ctx, PolyUOp *target, PolyUOp *effect);
PolyUOp *poly_uop_reduce(
    PolyCtx *ctx,
    PolyOps reduce_op,
    PolyUOp *expr,
    PolyUOp **ranges,
    int n_ranges
);
PolyUOp *poly_uop_flatten(PolyCtx *ctx, PolyUOp *u);
int64_t poly_uop_numel(PolyCtx *ctx, PolyUOp *u);

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

/* Free cached schedule results (param-to-binding mappings). */

#ifdef __cplusplus
}
#endif

#endif /* POLY_FRONTEND_H */
