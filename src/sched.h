/*
 * sched.h — Tensor-to-kernel scheduler API
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

/* Convenience: create a BUFFER UOp (1D, given element count) */
PolyUOp *poly_buffer(PolyCtx *ctx, PolyDType scalar_dtype, int64_t size);

/* Convenience: reshape a tensor UOp */
PolyUOp *poly_reshape(PolyCtx *ctx, PolyUOp *src, int64_t *dims, int ndim);

/* Convenience: expand (broadcast) a tensor UOp */
PolyUOp *poly_expand(PolyCtx *ctx, PolyUOp *src, int64_t *dims, int ndim);

/* Convenience: reduce a tensor along specified axes */
PolyUOp *poly_reduce_axis(PolyCtx *ctx, PolyOps reduce_op, PolyUOp *src,
                         int64_t *axes, int n_axes);

/* Convenience: permute (transpose) tensor dimensions */
PolyUOp *poly_permute(PolyCtx *ctx, PolyUOp *src, int64_t *perm, int ndim);

/* Convenience: shrink (slice) a tensor — pairs are (start, end) per dim */
PolyUOp *poly_shrink(PolyCtx *ctx, PolyUOp *src, int64_t (*pairs)[2], int ndim);

/* Convenience: flip (reverse) tensor along specified axes */
PolyUOp *poly_flip(PolyCtx *ctx, PolyUOp *src, int64_t *axes, int n_axes);

/* Convenience: pad a tensor — pairs are (before, after) per dim */
PolyUOp *poly_pad(PolyCtx *ctx, PolyUOp *src, int64_t (*pairs)[2], int ndim);

#ifdef __cplusplus
}
#endif

#endif /* POLY_SCHED_H */
