/*
 * uop/weak.h — weak dtype commit and lowering matchers
 *
 * Mirrors tinygrad/uop/weak.py.  These are UOp rewrite semantics shared by
 * every renderer; codegen only decides where the matchers run.
 */

#ifndef POLY_UOP_WEAK_H
#define POLY_UOP_WEAK_H

#include "uop/upat.h"

PolyPatternMatcher *poly_pm_commit_weak(void);
PolyPatternMatcher *poly_pm_cast_weak(void);
PolyPatternMatcher *poly_pm_lower_weak(void);
PolyPatternMatcher *poly_pm_lower_index_dtype(void);

/* tinygrad/uop/weak.py:commit_weak. */
PolyUOp *poly_commit_weak(PolyCtx *ctx, PolyUOp *u, PolyDType dtype);

/* Callback form used by pm_lower_index_dtype. */
PolyUOp *poly_lower_weak_srcs(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b);

#endif
