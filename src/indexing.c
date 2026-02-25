/*
 * indexing.c — Movement op index transforms for rangeify
 *
 * Ports tinygrad's indexing.py apply_movement_op() to C11.
 * Refactored from the movement op cases in sched.c lower_uop().
 *
 * Key difference from sched.c: these functions transform ranges
 * without doing any lowering — they return index UOp expressions
 * that the rangeify pipeline uses for scheduling decisions.
 */

#include "indexing.h"
#include <stdio.h>
#include <string.h>

/* ── Flat index computation ──────────────────────────────────────────── */

PolyUOp *poly_compute_flat_index(PolyCtx *ctx, PolyUOp **ranges, int ndim,
                                PolyShape shape) {
  if (ndim == 0)
    return poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  if (ndim == 1)
    return ranges[0];

  int64_t strides[POLY_MAX_DIMS];
  strides[ndim - 1] = 1;
  for (int i = ndim - 2; i >= 0; i--)
    strides[i] = strides[i + 1] * shape.dims[i + 1];

  PolyUOp *flat = NULL;
  for (int i = 0; i < ndim; i++) {
    PolyUOp *term;
    if (strides[i] == 1) {
      term = ranges[i];
    } else {
      PolyUOp *s = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(strides[i]));
      term = poly_uop2(ctx, POLY_OP_MUL, POLY_INT32, ranges[i], s, poly_arg_none());
    }
    flat = flat ? poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, flat, term, poly_arg_none()) : term;
  }
  return flat;
}

/* ── Symbolic flat index computation ────────────────────────────────── */

PolyUOp *poly_compute_flat_index_symbolic(PolyCtx *ctx, PolyUOp **ranges,
                                           PolyUOp **bounds, int ndim) {
  if (ndim == 0)
    return poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  if (ndim == 1)
    return ranges[0];

  /* Build strides bottom-up as UOp expressions.
   * stride[ndim-1] = 1, stride[i] = stride[i+1] * bounds[i+1] */
  PolyUOp *strides[POLY_MAX_DIMS];
  strides[ndim - 1] = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(1));
  for (int i = ndim - 2; i >= 0; i--)
    strides[i] = poly_uop2(ctx, POLY_OP_MUL, POLY_INT32, strides[i + 1], bounds[i + 1],
                             poly_arg_none());

  /* Sum terms: ranges[i] * strides[i] */
  PolyUOp *flat = NULL;
  for (int i = 0; i < ndim; i++) {
    PolyUOp *term;
    /* Optimize: stride == 1 → skip the MUL */
    if (strides[i]->op == POLY_OP_CONST && strides[i]->arg.i == 1)
      term = ranges[i];
    else
      term = poly_uop2(ctx, POLY_OP_MUL, POLY_INT32, ranges[i], strides[i],
                         poly_arg_none());
    flat = flat ? poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, flat, term, poly_arg_none()) : term;
  }
  return flat;
}

/* ── Reshape index transform ─────────────────────────────────────────── */

void poly_reshape_indices(PolyCtx *ctx,
                          PolyUOp **out_ranges, int out_ndim, PolyShape out_shape,
                          PolyUOp **in_ranges, int in_ndim, PolyShape in_shape) {
  PolyUOp *combined = poly_compute_flat_index(ctx, out_ranges, out_ndim, out_shape);

  int64_t in_stride = 1;
  for (int j = in_ndim - 1; j >= 0; j--) {
    PolyUOp *dim_val = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(in_shape.dims[j]));
    PolyUOp *shifted;
    if (in_stride == 1) {
      shifted = combined;
    } else {
      PolyUOp *s = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(in_stride));
      shifted = poly_uop2(ctx, POLY_OP_IDIV, POLY_INT32, combined, s, poly_arg_none());
    }
    in_ranges[j] = poly_uop2(ctx, POLY_OP_MOD, POLY_INT32, shifted, dim_val, poly_arg_none());
    in_stride *= in_shape.dims[j];
  }
}

/* ── apply_movement_op ───────────────────────────────────────────────── */

bool poly_apply_movement_op(PolyCtx *ctx, PolyOps op,
                            PolyShape in_shape, PolyArg arg,
                            PolyUOp **out_rngs, int n_out,
                            PolyUOp **in_rngs, int *n_in_out,
                            PolyUOp **valid_out) {
  if (valid_out) *valid_out = NULL;

  switch (op) {

  /* EXPAND: zero out expanded dims where in_dim=1 but out_dim>1 */
  case POLY_OP_EXPAND: {
    if (arg.kind != POLY_ARG_INT_TUPLE) return false;
    int n = in_shape.ndim;
    PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
    for (int i = 0; i < n; i++) {
      if (i >= n_out) { in_rngs[i] = zero; continue; }
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
    PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
    for (int i = 0; i < n; i++) in_rngs[i] = zero;
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
    PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
    for (int i = 0; i < n; i++) {
      if (i >= n_out) { in_rngs[i] = zero; continue; }
      int64_t start = arg.pair_tuple.pairs[i][0];
      if (start == 0) {
        in_rngs[i] = out_rngs[i];
      } else {
        PolyUOp *off = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(start));
        in_rngs[i] = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32,
                                out_rngs[i], off, poly_arg_none());
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
    PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
    for (int i = 0; i < n; i++) {
      if (i >= n_out) { in_rngs[i] = zero; continue; }
      if (flipped[i]) {
        PolyUOp *max_idx = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32,
                                     poly_arg_int(in_shape.dims[i] - 1));
        in_rngs[i] = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, max_idx,
            poly_uop1(ctx, POLY_OP_NEG, POLY_INT32, out_rngs[i], poly_arg_none()),
            poly_arg_none());
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
    /* out_shape comes from the arg (the target shape) */
    PolyShape out_shape;
    out_shape.dims = arg.int_tuple.vals;
    out_shape.ndim = arg.int_tuple.n;

    poly_reshape_indices(ctx, out_rngs, n_out, out_shape,
                         in_rngs, in_shape.ndim, in_shape);
    *n_in_out = in_shape.ndim;
    return true;
  }

  /* PAD: offset + bounds check */
  case POLY_OP_PAD: {
    if (arg.kind != POLY_ARG_PAIR_TUPLE) return false;
    int n = arg.pair_tuple.n;
    PolyUOp *valid = NULL;
    PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
    PolyUOp *falsev = poly_uop0(ctx, POLY_OP_CONST, POLY_BOOL, poly_arg_bool(false));

    for (int i = 0; i < n; i++) {
      if (i >= n_out) {
        in_rngs[i] = zero;
        valid = valid ? poly_uop2(ctx, POLY_OP_AND, POLY_BOOL, valid, falsev, poly_arg_none()) : falsev;
        continue;
      }
      int64_t begin = arg.pair_tuple.pairs[i][0];
      if (begin == 0) {
        in_rngs[i] = out_rngs[i];
      } else {
        PolyUOp *off = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(begin));
        in_rngs[i] = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, out_rngs[i],
            poly_uop1(ctx, POLY_OP_NEG, POLY_INT32, off, poly_arg_none()),
            poly_arg_none());
      }
      /* valid_i = (in_idx >= 0) AND (in_idx < in_dim) */
      PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
      PolyUOp *dim = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32,
                               poly_arg_int(in_shape.dims[i]));
      PolyUOp *ge_zero = poly_uop1(ctx, POLY_OP_NEG, POLY_BOOL,
          poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, in_rngs[i], zero,
                    poly_arg_none()), poly_arg_none());
      PolyUOp *lt_dim = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL,
                                  in_rngs[i], dim, poly_arg_none());
      PolyUOp *dv = poly_uop2(ctx, POLY_OP_AND, POLY_BOOL,
                              ge_zero, lt_dim, poly_arg_none());
      valid = valid ? poly_uop2(ctx, POLY_OP_AND, POLY_BOOL,
                                valid, dv, poly_arg_none()) : dv;
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
