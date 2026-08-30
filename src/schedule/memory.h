/* Current Tinygrad schedule/memory.py memory-plan rewrite. */

#ifndef POLY_SCHEDULE_MEMORY_H
#define POLY_SCHEDULE_MEMORY_H

#include "polygrad.h"

/* Current Tinygrad schedule/memory.py:_collect_bufs. The caller owns *buffers. */
bool poly_collect_bufs(PolyUOp *uop, PolyUOp ***buffers, int *count, int *capacity);

PolyUOp *poly_memory_plan_rewrite(
    PolyCtx *ctx,
    PolyUOp *linear,
    PolyUOp **held_buffers,
    int held_count
);

#endif
