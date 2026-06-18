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

/* The graph-side realize ABI now uses the batched Tensor-style poly_realize
 * entrypoint, so bump the public ABI version alongside that refactor. */
#define POLYGRAD_ABI_VERSION 4

#ifdef __cplusplus
extern "C" {
#endif

/* FFI buffer constructors for bindings that cannot pass PolyDType by value. */
PolyUOp *poly_buffer_by_id(PolyCtx *ctx, int dtype_id, int64_t size);
PolyUOp *poly_buffer_f32(PolyCtx *ctx, int64_t size);
PolyUOp *poly_buffer_f64(PolyCtx *ctx, int64_t size);
int poly_uop_dtype_id(PolyCtx *ctx, PolyUOp *u);

/* Dynamic shapes (DEFINE_VAR / BIND) */

/* Create a symbolic integer variable with bounds [min_val, max_val]. */
PolyUOp *poly_define_var(PolyCtx *ctx, const char *name, int64_t min_val, int64_t max_val);

/* Bind a concrete value to a DEFINE_VAR (creates BIND UOp). */
PolyUOp *poly_bind_var(PolyCtx *ctx, PolyUOp *var, int64_t value);

/* ABI version (callers check at load time for compatibility). */
int poly_abi_version(void);

/* Debug: print UOp info to stderr */
void poly_debug_uop(PolyCtx *ctx, PolyUOp *u);
void poly_debug_opsets(void);

/* Cache cleanup (for leak-free shutdown) */

/* ABI cleanup hook retained for frontends. CPU program caches are per-context
 * and are released by poly_ctx_destroy(). */
void poly_cpu_cache_flush(void);

/* Free cached schedule results (param-to-binding mappings). */
void poly_sched_cache_flush(void);

#ifdef __cplusplus
}
#endif

#endif /* POLY_FRONTEND_H */
