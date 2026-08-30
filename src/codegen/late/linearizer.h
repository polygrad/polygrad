#ifndef POLY_CODEGEN_LATE_LINEARIZER_H
#define POLY_CODEGEN_LATE_LINEARIZER_H

#include "uop/upat.h"

/* Current Tinygrad 2026-08-22/a9069c177a9d
 * codegen/late/linearizer.py:CFGContext + pm_add_control_flow. */
PolyUOp *poly_apply_control_flow(PolyCtx *ctx, PolyUOp *sink);

/* Current Tinygrad 2026-08-22/a9069c177a9d
 * codegen/late/linearizer.py:pm_split_ends. */
PolyPatternMatcher *poly_pm_split_ends(void);

/* Current Tinygrad 2026-08-22/a9069c177a9d codegen/late/linearizer.py:linearize. */
PolyUOp **poly_linearize(PolyCtx *ctx, PolyUOp *sink, int *n_out);

#endif
