/* mixin/movement.h -- Movement compositions on UOp graphs. */
#ifndef POLY_MIXIN_MOVEMENT_H
#define POLY_MIXIN_MOVEMENT_H

#include "../core.h"

#ifdef __cplusplus
extern "C" {
#endif

PolyUOp *poly_uop_reshape(PolyCtx *ctx, PolyUOp *src, int64_t *dims, int ndim);

PolyUOp *poly_uop_reshape_symbolic(PolyCtx *ctx, PolyUOp *src, PolyUOp **dims, int ndim);

PolyUOp *poly_uop_stack_axis(PolyCtx *ctx, PolyUOp **src, int n_src, int dim);

PolyUOp *poly_uop_expand(PolyCtx *ctx, PolyUOp *src, int64_t *dims, int ndim);

PolyUOp *poly_uop_expand_symbolic(PolyCtx *ctx, PolyUOp *src, PolyUOp **dims, int ndim);

PolyUOp *poly_uop_permute(PolyCtx *ctx, PolyUOp *src, int64_t *perm, int ndim);

PolyUOp *poly_uop_shrink(PolyCtx *ctx, PolyUOp *src, int64_t (*pairs)[2], int ndim);

PolyUOp *poly_uop_shrink_symbolic(
    PolyCtx *ctx,
    PolyUOp *src,
    PolyUOp **starts,
    PolyUOp **sizes,
    int ndim
);

PolyUOp *poly_uop_pad_symbolic(
    PolyCtx *ctx,
    PolyUOp *src,
    PolyUOp **offsets,
    PolyUOp **sizes,
    int ndim
);

PolyUOp *poly_uop_flip(PolyCtx *ctx, PolyUOp *src, int64_t *axes, int n_axes);

PolyUOp *poly_uop_pad(PolyCtx *ctx, PolyUOp *src, int64_t (*pairs)[2], int ndim);

/* Movement-op helpers (port of tinygrad mixin/movement.py) */

/* Tensor.repeat -- movement.py:465. n_repeats >= input ndim. */
PolyUOp *poly_uop_repeat(PolyCtx *ctx, PolyUOp *x, const int64_t *repeats, int n_repeats);

/* Tensor.shrink_to -- movement.py:168. ends[i] == -1 means no-op (keep dim). */
PolyUOp *poly_uop_shrink_to(PolyCtx *ctx, PolyUOp *x, const int64_t *ends, int n_ends);

/* Tensor.cat -- tensor.py:1364. Concatenate tensors along `dim`.
 * All tensors must have identical shape except along `dim`. */
PolyUOp *poly_uop_cat(PolyCtx *ctx, PolyUOp **tensors, int n_tensors, int dim);

/* Tensor._pad_circular -- tensor.py:1075. Circular (wrap-around) padding.
 * Negative pads not supported. Each pad must be <= corresponding dim size. */
PolyUOp *poly_uop_pad_circular(PolyCtx *ctx, PolyUOp *x, int64_t (*pads)[2], int ndim);

/* Tensor._pad_reflect_replicate (mode="reflect") -- tensor.py:1081.
 * Reflect padding without repeating the edge. Each pad must be < dim size. */
PolyUOp *poly_uop_pad_reflect(PolyCtx *ctx, PolyUOp *x, int64_t (*pads)[2], int ndim);

/* Tensor._pad_reflect_replicate (mode="replicate") -- tensor.py:1081.
 * Replicate (edge-extend) padding. Repeats the boundary element. */
PolyUOp *poly_uop_pad_replicate(PolyCtx *ctx, PolyUOp *x, int64_t (*pads)[2], int ndim);

/* Tensor._cumalu -- tensor.py:2048. Cumulative reduction along `axis`.
 * Supports POLY_OP_ADD, POLY_OP_MAX, POLY_OP_MUL (uses poly_uop_pad_value with
 * the operator's identity element). include_initial=true uses negative pad
 * on the right (tinygrad parity). */
PolyUOp *poly_uop_cumalu(PolyCtx *ctx, PolyUOp *x, int axis, PolyOps op);

PolyUOp *poly_uop_split_cumalu(PolyCtx *ctx, PolyUOp *x, int axis, PolyOps op);

/* Broadcasting (matches tinygrad's _broadcasted) */

/* Broadcast a UOp to a target shape via reshape + expand.
 * Equivalent to tinygrad's _broadcast_to: left-pad dims with 1, then expand. */
PolyUOp *poly_uop_broadcast_to(PolyCtx *ctx, PolyUOp *x, const int64_t *shape, int ndim);

/* Broadcast two UOps to a common shape (tinygrad's _broadcasted).
 * Returns the broadcast shape via out_shape/out_ndim. Returns false on
 * incompatible shapes. */
bool poly_uop_broadcast_pair(
    PolyCtx *ctx,
    PolyUOp **a,
    PolyUOp **b,
    int64_t *out_shape,
    int *out_ndim
);

#ifdef __cplusplus
}
#endif

#endif
