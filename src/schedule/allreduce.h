#ifndef POLY_SCHEDULE_ALLREDUCE_H
#define POLY_SCHEDULE_ALLREDUCE_H

#include "polygrad.h"

/* C implementation of tinygrad schedule/allreduce.py:create_allreduce_function. */
PolyUOp *poly_create_allreduce_function(PolyCtx *ctx, PolyUOp *red);

#endif
