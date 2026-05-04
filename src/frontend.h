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
#include "tensor.h"
#include "engine/schedule.h" /* PolyBuffer, PolyDevice */

/* The graph-side realize ABI now uses the batched Tensor-style poly_realize
 * entrypoint, so bump the public ABI version alongside that refactor. */
#define POLYGRAD_ABI_VERSION 4

#ifdef __cplusplus
extern "C" {
#endif

/* In-place assignment */

/* Create ASSIGN(target, value): in-place write of value into target's buffer.
 * Target must be a BUFFER-rooted UOp (full-buffer ASSIGN only).
 * The ASSIGN is realized as a kernel that writes to the existing buffer
 * (no intermediate allocation). WAR edges ensure readers complete first. */
PolyUOp *poly_assign(PolyCtx *ctx, PolyUOp *target, PolyUOp *value);

/* Buffer shortcuts */

PolyUOp *poly_buffer_f32(PolyCtx *ctx, int64_t size);
PolyUOp *poly_buffer_f64(PolyCtx *ctx, int64_t size);
PolyUOp *poly_buffer_by_id(PolyCtx *ctx, int64_t size, int dtype_id);
int poly_uop_dtype_id(PolyCtx *ctx, PolyUOp *u);

/* Dynamic shapes (DEFINE_VAR / BIND) */

/* Create a symbolic integer variable with bounds [min_val, max_val]. */
PolyUOp *poly_define_var(PolyCtx *ctx, const char *name, int64_t min_val, int64_t max_val);

/* Bind a concrete value to a DEFINE_VAR (creates BIND UOp). */
PolyUOp *poly_bind_var(PolyCtx *ctx, PolyUOp *var, int64_t value);

/* Create a dynamic buffer with variable dim 0 and fixed inner dims.
 * 1D: poly_buffer_var(ctx, dt, var, NULL, 0)  -> shape (max_B,)
 * 2D: poly_buffer_var(ctx, dt, var, &K, 1)    -> shape (max_B, K)
 * Allocation size = max_val * product(inner_dims). */
PolyUOp *poly_buffer_var(
    PolyCtx *ctx,
    PolyDType dt,
    PolyUOp *batch_var,
    const int64_t *inner_dims,
    int n_inner_dims
);

typedef struct PolyVarBinding {
  PolyUOp *var; /* DEFINE_VAR UOp */
  int32_t value; /* concrete runtime value */
} PolyVarBinding;

/* ABI version (callers check at load time for compatibility). */
int poly_abi_version(void);

/* Debug: print UOp info to stderr */
void poly_debug_uop(PolyCtx *ctx, PolyUOp *u);
void poly_debug_opsets(void);

/* Cache cleanup (for leak-free shutdown) */

/* Free all cached compiled CPU programs (dlclose + free). */
void poly_cpu_cache_flush(void);

/* Free cached schedule results (param-to-binding mappings). */
void poly_sched_cache_flush(void);

#ifdef POLY_HAS_CUDA

void poly_cuda_flush_buffers(void);
void poly_cuda_prog_cache_flush(void);

#endif /* POLY_HAS_CUDA */

#ifdef __cplusplus
}
#endif

#endif /* POLY_FRONTEND_H */
