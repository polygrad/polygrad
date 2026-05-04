/*
 * indexing.c — Movement op index transforms for rangeify
 *
 * Ports tinygrad's indexing.py apply_movement_op() to C11.
 * Refactored from the movement-op cases in the retired single-kernel lowerer.
 *
 * Key difference from that retired path: these functions transform ranges
 * without doing any lowering — they return index UOp expressions
 * that the rangeify pipeline uses for scheduling decisions.
 */

#include "schedule/indexing.h"
#include "pat.h"
#include <stdio.h>
#include <string.h>

static PolyUOp *index_const(PolyCtx *ctx, int64_t value) {
  return poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(value));
}

static PolyUOp *to_index_dtype(PolyCtx *ctx, PolyUOp *u) {
  if (!u || poly_dtype_eq(poly_dtype_scalar(u->dtype), POLY_INDEX)) return u;
  return poly_uop1(ctx, POLY_OP_CAST, POLY_INDEX, u, poly_arg_none());
}

static PolyUOp *index_neg_like_tinygrad(PolyCtx *ctx, PolyUOp *u) {
  /* tinygrad ElementwiseMixin.neg() constructs x * -1, not a NEG UOp.
   * Movement indexes run through symbolic div/mod simplification before late
   * decompositions can introduce NEG, so keep this source shape identical. */
  return poly_uop2(ctx, POLY_OP_MUL, POLY_INDEX, to_index_dtype(ctx, u), index_const(ctx, -1), poly_arg_none());
}

/* Flat index computation */

PolyUOp *poly_compute_flat_index(PolyCtx *ctx, PolyUOp **ranges, int ndim, PolyShape shape) {
  if (ndim == 0) return index_const(ctx, 0);
  if (ndim == 1) return to_index_dtype(ctx, ranges[0]);

  int64_t strides[POLY_MAX_DIMS];
  strides[ndim - 1] = 1;
  for (int i = ndim - 2; i >= 0; i--) {
    if (__builtin_mul_overflow(strides[i + 1], shape.dims[i + 1], &strides[i])) {
      fprintf(
          stderr, "poly_compute_flat_index: stride overflow at dim %d: %lld * %lld\n", i,
          (long long)strides[i + 1], (long long)shape.dims[i + 1]
      );
      for (int j = 0; j < ndim; j++)
        fprintf(stderr, "  shape[%d] = %lld\n", j, (long long)shape.dims[j]);
    }
  }

  PolyUOp *flat = index_const(ctx, 0);
  for (int i = ndim - 1; i >= 0; i--) {
    PolyUOp *term;
    if (strides[i] == 1) {
      term = to_index_dtype(ctx, ranges[i]);
    } else {
      PolyUOp *s = index_const(ctx, strides[i]);
      term = poly_uop2(ctx, POLY_OP_MUL, POLY_INDEX, s, to_index_dtype(ctx, ranges[i]), poly_arg_none());
    }
    flat = poly_uop2(ctx, POLY_OP_ADD, POLY_INDEX, flat, term, poly_arg_none());
  }
  return poly_graph_rewrite(ctx, flat, poly_symbolic());
}

/* Symbolic flat index computation */

PolyUOp *poly_compute_flat_index_symbolic(
    PolyCtx *ctx,
    PolyUOp **ranges,
    PolyUOp **bounds,
    int ndim
) {
  if (ndim == 0) return index_const(ctx, 0);
  if (ndim == 1) return to_index_dtype(ctx, ranges[0]);

  /* Build strides bottom-up as UOp expressions.
   * stride[ndim-1] = 1, stride[i] = stride[i+1] * bounds[i+1] */
  PolyUOp *strides[POLY_MAX_DIMS];
  strides[ndim - 1] = index_const(ctx, 1);
  for (int i = ndim - 2; i >= 0; i--)
    strides[i] =
        poly_uop2(ctx, POLY_OP_MUL, POLY_INDEX, strides[i + 1], to_index_dtype(ctx, bounds[i + 1]), poly_arg_none());

  /* Match poly_compute_flat_index / tinygrad _apply_reshape ordering for
   * symbolic bounds too: innermost dim first, then symbolic canonicalization. */
  PolyUOp *flat = index_const(ctx, 0);
  for (int i = ndim - 1; i >= 0; i--) {
    PolyUOp *term;
    /* Optimize: stride == 1 → skip the MUL */
    if (strides[i]->op == POLY_OP_CONST && strides[i]->arg.i == 1)
      term = to_index_dtype(ctx, ranges[i]);
    else
      term = poly_uop2(ctx, POLY_OP_MUL, POLY_INDEX, strides[i], to_index_dtype(ctx, ranges[i]), poly_arg_none());
    flat = poly_uop2(ctx, POLY_OP_ADD, POLY_INDEX, flat, term, poly_arg_none());
  }
  return poly_graph_rewrite(ctx, flat, poly_symbolic());
}

/* Reshape index transform */

void poly_reshape_indices(
    PolyCtx *ctx,
    PolyUOp **out_ranges,
    int out_ndim,
    PolyShape out_shape,
    PolyUOp **in_ranges,
    int in_ndim,
    PolyShape in_shape
) {
  PolyUOp *combined = poly_compute_flat_index(ctx, out_ranges, out_ndim, out_shape);

  int64_t in_stride = 1;
  for (int j = in_ndim - 1; j >= 0; j--) {
    PolyUOp *dim_val = index_const(ctx, in_shape.dims[j]);
    PolyUOp *shifted;
    if (in_stride == 1) {
      shifted = combined;
    } else {
      PolyUOp *s = index_const(ctx, in_stride);
      shifted = poly_uop2(ctx, POLY_OP_IDIV, POLY_INDEX, combined, s, poly_arg_none());
    }
    in_ranges[j] = poly_uop2(ctx, POLY_OP_MOD, POLY_INDEX, shifted, dim_val, poly_arg_none());
    in_stride *= in_shape.dims[j];
  }
}

/* apply_movement_op */

bool poly_apply_movement_op(
    PolyCtx *ctx,
    PolyOps op,
    PolyShape in_shape,
    PolyArg arg,
    PolyUOp **out_rngs,
    int n_out,
    PolyUOp **in_rngs,
    int *n_in_out,
    PolyUOp **valid_out
) {
  if (valid_out) *valid_out = NULL;

  switch (op) {

  /* EXPAND: zero out expanded dims where in_dim=1 but out_dim>1 */
  case POLY_OP_EXPAND: {
    if (arg.kind != POLY_ARG_INT_TUPLE) return false;
    int n = in_shape.ndim;
    PolyUOp *zero = index_const(ctx, 0);
    for (int i = 0; i < n; i++) {
      if (i >= n_out) {
        in_rngs[i] = zero;
        continue;
      }
      if (in_shape.dims[i] == 1 && arg.int_tuple.vals[i] != 1) {
        in_rngs[i] = zero;
      } else {
        in_rngs[i] = out_rngs[i];
      }
    }
    *n_in_out = n;
    return true;
  }

  /* PERMUTE: reorder ranges by inverse permutation */
  case POLY_OP_PERMUTE: {
    if (arg.kind != POLY_ARG_INT_TUPLE) return false;
    int n = arg.int_tuple.n;
    PolyUOp *zero = index_const(ctx, 0);
    for (int i = 0; i < n; i++)
      in_rngs[i] = zero;
    for (int i = 0; i < n && i < n_out; i++) {
      int p = (int)arg.int_tuple.vals[i];
      if (p >= 0 && p < n) in_rngs[p] = out_rngs[i];
    }
    *n_in_out = n;
    return true;
  }

  /* SHRINK: offset range by start value */
  case POLY_OP_SHRINK: {
    if (arg.kind != POLY_ARG_PAIR_TUPLE) return false;
    int n = arg.pair_tuple.n;
    PolyUOp *zero = index_const(ctx, 0);
    for (int i = 0; i < n; i++) {
      if (i >= n_out) {
        in_rngs[i] = zero;
        continue;
      }
      int64_t start = arg.pair_tuple.pairs[i][0];
      if (start == 0) {
        in_rngs[i] = out_rngs[i];
      } else {
        PolyUOp *off = index_const(ctx, start);
        in_rngs[i] =
            poly_uop2(ctx, POLY_OP_ADD, POLY_INDEX, to_index_dtype(ctx, out_rngs[i]), off, poly_arg_none());
      }
    }
    *n_in_out = in_shape.ndim;
    return true;
  }

  /* FLIP: reverse indices along specified axes */
  case POLY_OP_FLIP: {
    if (arg.kind != POLY_ARG_INT_TUPLE) return false;
    bool flipped[POLY_MAX_DIMS] = {false};
    for (int i = 0; i < arg.int_tuple.n; i++) {
      int ax = (int)arg.int_tuple.vals[i];
      if (ax >= 0 && ax < in_shape.ndim) flipped[ax] = true;
    }
    int n = in_shape.ndim;
    PolyUOp *zero = index_const(ctx, 0);
    for (int i = 0; i < n; i++) {
      if (i >= n_out) {
        in_rngs[i] = zero;
        continue;
      }
      if (flipped[i]) {
        PolyUOp *max_idx = index_const(ctx, in_shape.dims[i] - 1);
        in_rngs[i] = poly_uop2(
            ctx, POLY_OP_ADD, POLY_INDEX, max_idx, index_neg_like_tinygrad(ctx, out_rngs[i]),
            poly_arg_none()
        );
      } else {
        in_rngs[i] = out_rngs[i];
      }
    }
    *n_in_out = n;
    return true;
  }

  /* RESHAPE: flatten + decompose */
  case POLY_OP_RESHAPE: {
    if (arg.kind != POLY_ARG_INT_TUPLE) return false;
    PolyShape out_shape;
    out_shape.dims = arg.int_tuple.vals;
    out_shape.ndim = arg.int_tuple.n;

    /* Use the RESHAPE arg's ndim as the authoritative output ndim,
     * not n_out (which may differ if range propagation assigned
     * a different number of ranges to this node). */
    poly_reshape_indices(
        ctx, out_rngs, out_shape.ndim, out_shape, in_rngs, in_shape.ndim, in_shape
    );
    *n_in_out = in_shape.ndim;
    return true;
  }

  /* PAD: offset + bounds check */
  case POLY_OP_PAD: {
    if (arg.kind != POLY_ARG_PAIR_TUPLE) return false;
    int n = arg.pair_tuple.n;
    PolyUOp *valid = NULL;
    PolyUOp *zero = index_const(ctx, 0);
    PolyUOp *falsev = poly_uop0(ctx, POLY_OP_CONST, POLY_BOOL, poly_arg_bool(false));

    for (int i = 0; i < n; i++) {
      if (i >= n_out) {
        in_rngs[i] = zero;
        valid =
            valid ? poly_uop2(ctx, POLY_OP_AND, POLY_BOOL, valid, falsev, poly_arg_none()) : falsev;
        continue;
      }
      int64_t begin = arg.pair_tuple.pairs[i][0];
      PolyUOp *shifted;
      if (begin == 0) {
        shifted = out_rngs[i];
      } else {
        PolyUOp *off = index_const(ctx, begin);
        shifted = poly_uop2(
            ctx, POLY_OP_ADD, POLY_INDEX, to_index_dtype(ctx, out_rngs[i]),
            index_neg_like_tinygrad(ctx, off), poly_arg_none()
        );
      }
      /* tinygrad schedule/indexing.py::apply_movement_op(PAD) forms the
       * valid mask on the output-space range:
       *
       *   valid = (r >= begin) & (r < in_dim + begin)
       *   index = valid.where(r - begin, Invalid)
       *
       * Building validity from `shifted = r - begin` is equivalent for values,
       * but loses the explicit begin/end constants that tinygrad's symbolic
       * and late-decomp passes preserve in linear IR. Keep the same unshifted
       * bounds here, then use the already-computed shifted index for the
       * data address. */
      int64_t end_val;
      if (__builtin_add_overflow(in_shape.dims[i], begin, &end_val)) return false;
      PolyUOp *begin_c = index_const(ctx, begin);
      PolyUOp *end_c = index_const(ctx, end_val);
      PolyUOp *zero = index_const(ctx, 0);
      PolyUOp *true_const = poly_uop0(ctx, POLY_OP_CONST, POLY_BOOL, poly_arg_bool(true));
      PolyUOp *lt_begin =
          poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, to_index_dtype(ctx, out_rngs[i]), begin_c, poly_arg_none());
      PolyUOp *ge_begin =
          poly_uop2(ctx, POLY_OP_CMPNE, POLY_BOOL, lt_begin, true_const, poly_arg_none());
      PolyUOp *lt_dim =
          poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, to_index_dtype(ctx, out_rngs[i]), end_c, poly_arg_none());
      PolyUOp *dv = poly_uop2(ctx, POLY_OP_AND, POLY_BOOL, ge_begin, lt_dim, poly_arg_none());
      /* Clamp index to valid range: WHERE(valid, shifted, 0).
       * Matches tinygrad indexing.py:137: valid.where(r-s, UOp.invalid()).
       * Prevents negative INDEX offsets that crash non-short-circuiting backends. */
      in_rngs[i] = poly_uop3(ctx, POLY_OP_WHERE, POLY_INDEX, dv, shifted, zero, poly_arg_none());
      valid = valid ? poly_uop2(ctx, POLY_OP_AND, POLY_BOOL, valid, dv, poly_arg_none()) : dv;
    }
    if (valid_out) *valid_out = valid;
    *n_in_out = in_shape.ndim;
    return true;
  }

  default:
    fprintf(stderr, "polygrad: indexing: unsupported movement op %s\n", poly_op_name(op));
    return false;
  }
}
