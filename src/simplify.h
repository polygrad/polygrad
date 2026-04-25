/*
 * simplify.h -- ports of tinygrad/codegen/simplify.py
 *
 * This header exposes the simplify.py stage helpers used by:
 *   - codegen full_rewrite_to_sink (Stage 2-5 parity work)
 *   - rangeify Phase D reduce simplify
 */

#ifndef POLY_SIMPLIFY_H
#define POLY_SIMPLIFY_H

#include "polygrad.h"
#include "pat.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  PolyUOp *r[POLY_MAX_DIMS];
  PolyUOp *c[POLY_MAX_DIMS];
  int n;
} SplitRangeCtx;

/* simplify.py stage helpers used by full_rewrite_to_sink. */
PolyPatternMatcher *poly_pm_flatten_range(void);
PolyPatternMatcher *poly_pm_split_ranges(void);
PolyPatternMatcher *poly_pm_simplify_ranges(void);
PolyPatternMatcher *poly_pm_load_collapse(void);

/* rangeify Phase D entrypoints. */
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

#endif /* POLY_SIMPLIFY_H */
