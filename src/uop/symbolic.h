#ifndef POLY_UOP_SYMBOLIC_H
#define POLY_UOP_SYMBOLIC_H

#include "uop/upat.h"

/* Tinygrad 2026-08-22/a9069c177a9d uop/symbolic.py:parse_valid. */
bool poly_parse_valid(
    PolyCtx *ctx,
    PolyUOp *clause,
    PolyUOp **expr,
    bool *is_upper,
    int64_t *bound
);

/* Tinygrad 2026-08-22/a9069c177a9d
 * uop/symbolic.py:pm_move_where_on_load. */
PolyPatternMatcher *poly_pm_move_where_on_load(void);

#ifdef POLY_TESTING
void poly_test_exact_int_range_alloc_fail(bool fail);
#endif

#endif
