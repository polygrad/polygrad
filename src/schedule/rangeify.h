/*
 * rangeify.h -- Tinygrad-aligned kernel graph construction.
 *
 * Mirrors tinygrad schedule/rangeify.py.
 */

#ifndef POLY_SCHEDULE_RANGEIFY_H
#define POLY_SCHEDULE_RANGEIFY_H

#include "uop/upat.h"
#include "polygrad.h"
#include "schedule/indexing.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Internal stage helpers kept visible for parity probes and focused tests. */
PolyPatternMatcher *poly_pm_mops(void);
PolyUOp *poly_apply_earliest_rewrites(PolyCtx *ctx, PolyUOp *sink);
PolyPatternMatcher *poly_pm_const_buffer_folding(void);
PolyPatternMatcher *poly_pm_remove_bufferize(void);
PolyPatternMatcher *poly_pm_add_buffers(void);
bool poly_find_bufs(PolyCtx *ctx, PolyUOp *store);

/* Tinygrad schedule/rangeify.py analogue: build the kernel graph for a sink. */
PolyUOp *poly_get_kernel_graph(PolyCtx *ctx, PolyUOp *sink);

#ifdef __cplusplus
}
#endif

#endif /* POLY_SCHEDULE_RANGEIFY_H */
