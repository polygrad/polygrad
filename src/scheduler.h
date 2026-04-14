/*
 * scheduler.h — Tensor-to-kernel scheduler API
 *
 * Converts tensor-level UOp graphs (BUFFER + ALU ops with shapes)
 * into kernel-level IR (PARAM/RANGE/INDEX/LOAD/STORE/END/SINK)
 * ready for the existing linearize → render → compile pipeline.
 */

#ifndef POLY_SCHED_H
#define POLY_SCHED_H

#include "polygrad.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Convert a tensor-level SINK to kernel-level SINK.
 *
 * The tensor SINK should have STORE sources, where each STORE writes
 * a computed value to a BUFFER. poly_schedule() produces a new graph with:
 * - PARAM nodes for each buffer
 * - RANGE loops over the output shape
 * - INDEX/LOAD for reading inputs
 * - ALU for computation
 * - STORE/END/SINK for writing output
 *
 * Returns NULL on error.
 */
PolyUOp *poly_schedule(PolyCtx *ctx, PolyUOp *tensor_sink);

#ifdef __cplusplus
}
#endif

#endif /* POLY_SCHED_H */
