/*
 * reduce_simplify.h — Phase D port of tinygrad's pm_reduce_simplify pass.
 *
 * Tinygrad source: tinygrad/codegen/simplify.py:73-149
 *   - no_range          (line 75)
 *   - reduce_unparented (line 77-92)
 *   - pm_reduce_collapse rules (line 94-119)
 *   - reduce_collapse driver (line 129-142)
 *   - pm_reduce_simplify combinator (line 147-149)
 *
 * Polygrad runs this pass once between Stage 2 (cleanup_dead_bufferize_axes)
 * and Stage 3 (remove_bufferize) inside src/rangeify.c. The result is that
 * arange-derived REDUCEs whose value depends on a strict subset of the
 * outer ranges are folded to closed-form expressions before the linearizer
 * sees them, matching tinygrad's IR shape.
 */

#ifndef POLY_REDUCE_SIMPLIFY_H
#define POLY_REDUCE_SIMPLIFY_H

#include "polygrad.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Apply pm_reduce_simplify to `sink`. Returns the rewritten sink (or `sink`
 * unchanged if no rule fires). Allocates and destroys a per-pass PolyUOpCache
 * internally; caller does not need to manage one. */
PolyUOp *poly_apply_reduce_simplify(PolyCtx *ctx, PolyUOp *sink);

/* Apply ONLY pm_reduce_unparented (without the symbolic_simple concat that
 * the production poly_apply_reduce_simplify does). This is for parity tests
 * that mirror tinygrad's tg_reduce_unparented_gt.py ground truth, which
 * runs pm_reduce_unparented in isolation. Production callers should use
 * poly_apply_reduce_simplify instead. */
PolyUOp *poly_apply_reduce_unparented_only(PolyCtx *ctx, PolyUOp *sink);

#ifdef __cplusplus
}
#endif

#endif /* POLY_REDUCE_SIMPLIFY_H */
