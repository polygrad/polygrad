#ifndef POLY_SCHEDULE_MULTI_H
#define POLY_SCHEDULE_MULTI_H

#include "polygrad.h"

/* C implementation of tinygrad schedule/multi.py:multi_pm. */
PolyUOp *poly_apply_multi_pm(PolyCtx *ctx, PolyUOp *sink);

#endif
