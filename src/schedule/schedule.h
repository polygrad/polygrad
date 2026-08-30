/* Current Tinygrad schedule linearizer. */

#ifndef POLY_SCHEDULE_SCHEDULE_H
#define POLY_SCHEDULE_SCHEDULE_H

#include "polygrad.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Port of tinygrad schedule/__init__.py:create_schedule. */
PolyUOp *poly_create_schedule(PolyCtx *ctx, PolyUOp *kernel_graph);

/* Current tinygrad schedule/__init__.py:create_linear_with_vars. */
PolyUOp *poly_create_linear_with_vars(
    PolyCtx *ctx,
    PolyUOp *big_call,
    PolyVarBinding **var_bindings_out,
    int *n_var_bindings_out
);

/* Current tinygrad schedule/__init__.py:pm_copy_from_store. */
PolyUOp *poly_copy_from_store(PolyCtx *ctx, PolyUOp *linear);

/* C observability for current Tinygrad's module-level schedule_cache. */
size_t poly_schedule_cache_len(PolyCtx *ctx);

#ifdef __cplusplus
}
#endif

#endif /* POLY_SCHEDULE_SCHEDULE_H */
