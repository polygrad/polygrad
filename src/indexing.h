/*
 * indexing.h — Movement op index transforms for rangeify
 *
 * Ports tinygrad's indexing.py apply_movement_op() to C11.
 * Transforms output ranges through movement ops to compute input ranges.
 */

#ifndef POLY_INDEXING_H
#define POLY_INDEXING_H

#include "polygrad.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Apply a movement op's index transform to output ranges.
 *
 * Given output ranges (what the consumer sees), compute input ranges
 * (what the movement op's source sees) after applying the op's semantics.
 *
 * op:        the movement op (RESHAPE, EXPAND, PERMUTE, SHRINK, FLIP, PAD)
 * in_shape:  shape of the movement op's source (input)
 * arg:       the movement op's argument
 * out_rngs:  output ranges (from consumer), length n_out
 * n_out:     number of output ranges
 * in_rngs:   [out] input ranges for the source, must hold at least in_shape.ndim entries
 * n_in_out:  [out] number of input ranges written
 *
 * For PAD, also outputs:
 * valid_out: [out] validity predicate UOp (AND of bounds checks), or NULL if no padding
 *
 * Returns true on success, false on error.
 */
bool poly_apply_movement_op(PolyCtx *ctx, PolyOps op,
                            PolyShape in_shape, PolyArg arg,
                            PolyUOp **out_rngs, int n_out,
                            PolyUOp **in_rngs, int *n_in_out,
                            PolyUOp **valid_out);

/* Compute flat index from multi-dimensional ranges and shape strides.
 * Returns a UOp expression: ranges[0]*stride[0] + ranges[1]*stride[1] + ... */
PolyUOp *poly_compute_flat_index(PolyCtx *ctx, PolyUOp **ranges, int ndim,
                                PolyShape shape);

/* Compute reshape index transform: given output ranges for out_shape,
 * compute input ranges for in_shape via flatten + decompose. */
void poly_reshape_indices(PolyCtx *ctx,
                          PolyUOp **out_ranges, int out_ndim, PolyShape out_shape,
                          PolyUOp **in_ranges, int in_ndim, PolyShape in_shape);

/* Compute flat index from multi-dimensional ranges and UOp bounds.
 * Like poly_compute_flat_index, but bounds are UOp* (can be CONST or DEFINE_VAR).
 * When all bounds are CONST, produces the same result as poly_compute_flat_index.
 * When bounds include DEFINE_VAR, strides are UOp MUL chains. */
PolyUOp *poly_compute_flat_index_symbolic(PolyCtx *ctx, PolyUOp **ranges,
                                           PolyUOp **bounds, int ndim);

#ifdef __cplusplus
}
#endif

#endif /* POLY_INDEXING_H */
