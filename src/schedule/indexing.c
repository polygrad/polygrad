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
#include <stdlib.h>
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

static PolyUOp *shape_stack_get(PolyUOp *stack, int idx) {
  if (!stack || stack->op != POLY_OP_STACK || idx < 0 || idx >= stack->n_src) return NULL;
  return stack->src[idx];
}

static _Thread_local PolyPatternMatcher *g_pm_reshape_indexing = NULL;
static _Thread_local PolyPatternMatcher *g_pm_pad_valid = NULL;

static PolyPatternMatcher *poly_pm_reshape_indexing(void) {
  if (g_pm_reshape_indexing) return g_pm_reshape_indexing;
  /* Pinned tinygrad/schedule/indexing.py:125-127. The three matchers run as
   * one fixed-point rewrite over the complete ordered coordinate SINK. */
  PolyPatternMatcher *symbolic_valid =
      poly_pm_concat(poly_symbolic(), poly_pm_simplify_valid());
  g_pm_reshape_indexing =
      poly_pm_thread_cache(poly_pm_concat(symbolic_valid, poly_pm_drop_and_clauses()));
  poly_pm_destroy(symbolic_valid);
  return g_pm_reshape_indexing;
}

static PolyPatternMatcher *poly_pm_pad_valid(void) {
  if (g_pm_pad_valid) return g_pm_pad_valid;
  /* Pinned tinygrad/schedule/indexing.py:136-141 simplifies the newly added
   * PAD validity before wrapping the shifted coordinate in WHERE(...,Invalid). */
  g_pm_pad_valid = poly_pm_thread_cache(
      poly_pm_concat(poly_symbolic(), poly_pm_simplify_valid())
  );
  return g_pm_pad_valid;
}

static bool index_invalid_where(PolyUOp *coord) {
  return coord && coord->op == POLY_OP_WHERE && coord->n_src == 3 && coord->src[2] &&
         coord->src[2]->op == POLY_OP_CONST && coord->src[2]->arg.kind == POLY_ARG_INVALID;
}

/* Pinned tinygrad/uop/ops.py:571-581. These are projections of an existing
 * index coordinate, not rewrite or placement state. */
PolyUOp *poly_index_get_idx(PolyCtx *ctx, PolyUOp *coord) {
  if (!ctx || !coord || !poly_dtype_is_index(poly_dtype_scalar(coord->dtype))) return NULL;
  if (coord->op == POLY_OP_STACK) {
    PolyUOp *stack_src[16];
    PolyUOp **src = coord->n_src <= (int)(sizeof(stack_src) / sizeof(stack_src[0]))
                        ? stack_src
                        : malloc((size_t)coord->n_src * sizeof(*src));
    if (!src) return NULL;
    for (int i = 0; i < coord->n_src; i++) {
      src[i] = poly_index_get_idx(ctx, coord->src[i]);
      if (!src[i]) {
        if (src != stack_src) free(src);
        return NULL;
      }
    }
    PolyUOp *ret = poly_uop(ctx, POLY_OP_STACK, coord->dtype, src, coord->n_src, poly_arg_none());
    if (src != stack_src) free(src);
    return ret;
  }
  return index_invalid_where(coord) ? coord->src[1] : coord;
}

PolyUOp *poly_index_get_valid(PolyCtx *ctx, PolyUOp *coord) {
  if (!ctx || !coord || !poly_dtype_is_index(poly_dtype_scalar(coord->dtype))) return NULL;
  if (coord->op == POLY_OP_STACK) {
    PolyUOp *stack_src[16];
    PolyUOp **src = coord->n_src <= (int)(sizeof(stack_src) / sizeof(stack_src[0]))
                        ? stack_src
                        : malloc((size_t)coord->n_src * sizeof(*src));
    if (!src) return NULL;
    for (int i = 0; i < coord->n_src; i++) {
      src[i] = poly_index_get_valid(ctx, coord->src[i]);
      if (!src[i]) {
        if (src != stack_src) free(src);
        return NULL;
      }
    }
    PolyDType dtype = poly_dtype_vec(POLY_BOOL, coord->dtype.count);
    PolyUOp *ret = poly_uop(ctx, POLY_OP_STACK, dtype, src, coord->n_src, poly_arg_none());
    if (src != stack_src) free(src);
    return ret;
  }
  if (index_invalid_where(coord)) return coord->src[0];
  return poly_uop0(
      ctx, POLY_OP_CONST, POLY_BOOL,
      poly_arg_bool(!(coord->op == POLY_OP_CONST && coord->arg.kind == POLY_ARG_INVALID))
  );
}

/* Pinned schedule/indexing.py:82-86 derives the PAD value wrapper from the
 * get_valid() projection of every transformed coordinate.  In particular, an
 * unchanged coordinate can already carry invalidity from an earlier movement
 * and must remain part of this PAD's wrapper predicate. */
static bool pad_wrapper_valid_from_coords(
    PolyCtx *ctx, PolyUOp **coords, int n_coords, int n_out, PolyUOp **valid_out
) {
  PolyUOp *valid = NULL;
  PolyUOp *falsev = poly_uop0(ctx, POLY_OP_CONST, POLY_BOOL, poly_arg_bool(false));
  for (int i = 0; i < n_coords; i++) {
    PolyUOp *dim_valid = i < n_out ? poly_index_get_valid(ctx, coords[i]) : falsev;
    if (!dim_valid) return false;
    if (dim_valid->op == POLY_OP_CONST && dim_valid->arg.kind == POLY_ARG_BOOL &&
        dim_valid->arg.b)
      continue;
    valid = valid
                ? poly_uop2(ctx, POLY_OP_AND, POLY_BOOL, valid, dim_valid, poly_arg_none())
                : dim_valid;
  }
  *valid_out = valid;
  return true;
}

static bool expand_target_dim_is_one(PolyCtx *ctx, PolyUOp *movement, PolyArg arg, int dim, bool *is_one) {
  if (!is_one) return false;
  *is_one = false;
  if (!movement || arg.kind != POLY_ARG_NONE || movement->n_src != 2) return false;
  /* Pinned indexing consumes EXPAND.marg == src[1].as_shape. Shape inference
   * already preserves that value and its symbolic dimensions. */
  PolyShape target_shape = poly_uop_max_shape_cached(ctx, movement);
  if (target_shape.ndim < 0 || dim < 0 || dim >= target_shape.ndim) return false;
  PolyUOp *target = poly_uop_shape_dim(ctx, movement, dim);
  int64_t v = target_shape.dims[dim];
  if (target) {
    target = poly_graph_rewrite(ctx, target, poly_symbolic_simple());
    if (poly_uop_const_i64(target, &v) != 0) return true;
  }
  *is_one = v == 1;
  return true;
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

bool poly_reshape_indices(
    PolyCtx *ctx,
    PolyUOp *reshape,
    PolyUOp **out_ranges,
    int n_out,
    PolyUOp **in_ranges,
    int *n_in_out
) {
  if (!ctx || !reshape || reshape->op != POLY_OP_RESHAPE ||
      reshape->n_src != 2 || n_out < 0 || (n_out > 0 && !out_ranges) ||
      !in_ranges || !n_in_out)
    return false;
  PolyShape out_shape = poly_uop_max_shape_cached(ctx, reshape);
  PolyShape in_shape = poly_uop_max_shape_cached(ctx, reshape->src[0]);
  if (out_shape.ndim < n_out || in_shape.ndim < 0 ||
      out_shape.ndim > POLY_MAX_DIMS || in_shape.ndim > POLY_MAX_DIMS)
    return false;

  /* Pinned schedule/rangeify.py:63-77 maps a partial RESHAPE INDEX only when
   * the unindexed output suffix is exactly the same as an input suffix. */
  int suffix_ndim = out_shape.ndim - n_out;
  int n_in = in_shape.ndim - suffix_ndim;
  if (n_in < 0) return false;
  for (int i = 0; i < suffix_ndim; i++) {
    PolyUOp *in_dim = poly_uop_shape_dim(ctx, reshape->src[0], n_in + i);
    PolyUOp *out_dim = poly_uop_shape_dim(ctx, reshape, n_out + i);
    in_dim = in_dim ? to_index_dtype(ctx, in_dim)
                    : index_const(ctx, in_shape.dims[n_in + i]);
    out_dim = out_dim ? to_index_dtype(ctx, out_dim)
                      : index_const(ctx, out_shape.dims[n_out + i]);
    in_dim = poly_graph_rewrite(ctx, in_dim, poly_symbolic_simple());
    out_dim = poly_graph_rewrite(ctx, out_dim, poly_symbolic_simple());
    if (!in_dim || !out_dim || in_dim != out_dim) return false;
  }

  /* Pinned indexing.py:142-145 first simplifies the ordered coordinate SINK,
   * replaces its active ranges with PLACEHOLDER ranges, applies the cached
   * reshape, then restores the original ranges. This prevents existing range
   * bounds/validity from changing commutative rewrite order inside reshape. */
  PolyUOp *coordinate_sink = poly_uop(
      ctx, POLY_OP_SINK, POLY_VOID, out_ranges, n_out, poly_arg_none()
  );
  coordinate_sink = coordinate_sink
                        ? poly_graph_rewrite(ctx, coordinate_sink, poly_symbolic())
                        : NULL;
  if (!coordinate_sink || coordinate_sink->op != POLY_OP_SINK ||
      coordinate_sink->n_src != n_out)
    return false;

  int range_cap = 0;
  PolyUOp **coordinate_topo = poly_toposort_alloc(ctx, coordinate_sink, &range_cap);
  if (!coordinate_topo && range_cap != 0) return false;
  poly_toposort_free(coordinate_topo);
  PolyUOp **original_ranges = range_cap > 0
                                  ? malloc((size_t)range_cap * sizeof(*original_ranges))
                                  : NULL;
  PolyUOp **placeholder_ranges = range_cap > 0
                                     ? malloc((size_t)range_cap * sizeof(*placeholder_ranges))
                                     : NULL;
  if (range_cap > 0 && (!original_ranges || !placeholder_ranges)) {
    free(original_ranges);
    free(placeholder_ranges);
    return false;
  }
  int n_replacements = range_cap > 0
                           ? poly_uop_ranges(
                                 ctx, coordinate_sink, original_ranges, range_cap
                             )
                           : 0;
  bool placeholder_ok = true;
  for (int i = 0; i < n_replacements; i++) {
    PolyUOp *range = original_ranges[i];
    if (!range || range->op != POLY_OP_RANGE || range->n_src != 1) {
      placeholder_ok = false;
      break;
    }
    placeholder_ranges[i] = poly_uop1(
        ctx, POLY_OP_RANGE, range->dtype, range->src[0],
        poly_arg_range(i, POLY_AXIS_PLACEHOLDER)
    );
    if (!placeholder_ranges[i]) {
      placeholder_ok = false;
      break;
    }
  }
  if (!placeholder_ok) {
    free(original_ranges);
    free(placeholder_ranges);
    return false;
  }
  PolyUOp *placeholder_sink = n_replacements > 0
                                  ? poly_uop_substitute(
                                        ctx, coordinate_sink, original_ranges,
                                        placeholder_ranges, n_replacements
                                    )
                                  : coordinate_sink;
  if (!placeholder_sink || placeholder_sink->op != POLY_OP_SINK ||
      placeholder_sink->n_src != n_out) {
    free(original_ranges);
    free(placeholder_ranges);
    return false;
  }

  /* Pinned _apply_reshape uses the exact symbolic output shape to flatten,
   * then floor-mod/divides by each exact symbolic input dimension
   * (schedule/indexing.py:113-127). Cached PolyShape dimensions are allocation
   * maxima only, so use them solely for static dimensions. */
  PolyUOp *out_bounds[POLY_MAX_DIMS];
  for (int i = 0; i < n_out; i++) {
    PolyUOp *dim = poly_uop_shape_dim(ctx, reshape, i);
    out_bounds[i] = dim ? to_index_dtype(ctx, dim) : index_const(ctx, out_shape.dims[i]);
  }
  /* Pinned _apply_reshape constructs every acc*src term and all div/mod nodes
   * before the one complete-SINK rewrite. In particular, do not call the
   * shared flat-index helper here: its eager symbolic rewrite combines
   * Invalid guards before reshape validity simplification can reason locally. */
  PolyUOp *acc = index_const(ctx, 1);
  PolyUOp *combined = index_const(ctx, 0);
  for (int i = n_out - 1; i >= 0; i--) {
    PolyUOp *term = poly_uop2(
        ctx, POLY_OP_MUL, POLY_INDEX, acc,
        to_index_dtype(ctx, placeholder_sink->src[i]), poly_arg_none()
    );
    combined = term ? poly_uop2(
                          ctx, POLY_OP_ADD, POLY_INDEX, combined, term,
                          poly_arg_none()
                      )
                    : NULL;
    acc = acc ? poly_uop2(
                    ctx, POLY_OP_MUL, POLY_INDEX, acc,
                    to_index_dtype(ctx, out_bounds[i]), poly_arg_none()
                )
              : NULL;
    if (!combined || !acc) break;
  }
  if (!combined || !acc) {
    free(original_ranges);
    free(placeholder_ranges);
    return false;
  }

  for (int j = n_in - 1; j >= 0; j--) {
    PolyUOp *dim = poly_uop_shape_dim(ctx, reshape->src[0], j);
    PolyUOp *dim_val = dim ? to_index_dtype(ctx, dim) : index_const(ctx, in_shape.dims[j]);
    in_ranges[j] =
        poly_uop2(ctx, POLY_OP_FLOORMOD, POLY_INDEX, combined, dim_val, poly_arg_none());
    combined =
        poly_uop2(ctx, POLY_OP_FLOORDIV, POLY_INDEX, combined, dim_val, poly_arg_none());
    if (!in_ranges[j] || !combined) {
      free(original_ranges);
      free(placeholder_ranges);
      return false;
    }
  }
  if (n_in > 0) {
    PolyUOp *sink = poly_uop(
        ctx, POLY_OP_SINK, POLY_VOID, in_ranges, n_in, poly_arg_none()
    );
    PolyUOp *simplified_placeholder =
        sink ? poly_graph_rewrite(ctx, sink, poly_pm_reshape_indexing()) : NULL;
    PolyUOp *simplified = simplified_placeholder && n_replacements > 0
                              ? poly_uop_substitute(
                                    ctx, simplified_placeholder, placeholder_ranges,
                                    original_ranges, n_replacements
                                )
                              : simplified_placeholder;
    if (!simplified || simplified->op != POLY_OP_SINK || simplified->n_src != n_in) {
      free(original_ranges);
      free(placeholder_ranges);
      return false;
    }
    for (int j = 0; j < n_in; j++) in_ranges[j] = simplified->src[j];
  }
  free(original_ranges);
  free(placeholder_ranges);
  *n_in_out = n_in;
  return true;
}

/* apply_movement_op */

bool poly_apply_movement_op(
    PolyCtx *ctx,
    PolyUOp *movement,
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
    int n = in_shape.ndim;
    PolyUOp *zero = index_const(ctx, 0);
    for (int i = 0; i < n; i++) {
      if (i >= n_out) {
        in_rngs[i] = zero;
        continue;
      }
      bool target_is_one = false;
      if (!expand_target_dim_is_one(ctx, movement, arg, i, &target_is_one)) return false;
      if (in_shape.dims[i] == 1 && !target_is_one) {
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
    if (movement && movement->arg.kind == POLY_ARG_NONE && movement->n_src >= 3 &&
        !movement->src[0]->dtype.is_ptr) {
      int n = in_shape.ndim;
      PolyUOp *zero = index_const(ctx, 0);
      for (int i = 0; i < n; i++) {
        if (i >= n_out) {
          in_rngs[i] = zero;
          continue;
        }
        PolyUOp *off = shape_stack_get(movement->src[1], i);
        if (!off) return false;
        int64_t off_i = 0;
        if (poly_uop_const_i64(off, &off_i) == 0 && off_i == 0) {
          in_rngs[i] = out_rngs[i];
        } else {
          in_rngs[i] =
              poly_uop2(ctx, POLY_OP_ADD, POLY_INDEX, to_index_dtype(ctx, out_rngs[i]),
                        to_index_dtype(ctx, off), poly_arg_none());
        }
      }
      *n_in_out = n;
      return true;
    }
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
    if (arg.int_tuple.n != in_shape.ndim) return false;
    bool flipped[POLY_MAX_DIMS] = {false};
    for (int i = 0; i < arg.int_tuple.n; i++)
      flipped[i] = arg.int_tuple.vals[i] != 0;
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
    if (!movement || movement->arg.kind != POLY_ARG_NONE || movement->n_src != 2)
      return false;
    if (!poly_reshape_indices(ctx, movement, out_rngs, n_out, in_rngs, n_in_out))
      return false;
    return true;
  }

  /* PAD: offset + bounds check */
  case POLY_OP_PAD: {
    if (movement && movement->arg.kind == POLY_ARG_NONE && movement->n_src == 3 &&
        !movement->src[0]->dtype.is_ptr) {
      int n = in_shape.ndim;
      if (movement->src[1]->op != POLY_OP_STACK || movement->src[1]->n_src != n) return false;
      PolyUOp *valid = NULL;
      PolyUOp *zero = index_const(ctx, 0);
      PolyUOp *invalid = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_invalid());
      PolyUOp *truev = poly_uop0(ctx, POLY_OP_CONST, POLY_BOOL, poly_arg_bool(true));

      for (int i = 0; i < n; i++) {
        if (i >= n_out) {
          in_rngs[i] = zero;
          continue;
        }
        PolyUOp *offset = shape_stack_get(movement->src[1], i);
        if (!offset) return false;
        PolyUOp *index = to_index_dtype(ctx, out_rngs[i]);
        PolyUOp *offset_index = to_index_dtype(ctx, offset);
        PolyUOp *input_size = poly_uop_shape_dim(ctx, movement->src[0], i);
        if (!input_size) input_size = index_const(ctx, in_shape.dims[i]);
        PolyUOp *output_size = shape_stack_get(movement->src[2], i);
        int64_t offset_value = 0;
        if (output_size && poly_uop_const_i64(offset, &offset_value) == 0 &&
            offset_value == 0 && output_size == input_size) {
          /* Pinned indexing.py:137 leaves an unchanged PAD dimension alone. */
          in_rngs[i] = out_rngs[i];
          continue;
        }
        PolyUOp *shifted = poly_uop2(
            ctx, POLY_OP_ADD, POLY_INDEX, index,
            index_neg_like_tinygrad(ctx, offset_index), poly_arg_none()
        );
        input_size = to_index_dtype(ctx, input_size);
        PolyUOp *end =
            poly_uop2(ctx, POLY_OP_ADD, POLY_INDEX, input_size, offset_index, poly_arg_none());
        PolyUOp *lt_begin =
            poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, index, offset_index, poly_arg_none());
        PolyUOp *ge_begin =
            poly_uop2(ctx, POLY_OP_CMPNE, POLY_BOOL, lt_begin, truev, poly_arg_none());
        PolyUOp *lt_end = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, index, end, poly_arg_none());
        PolyUOp *dim_valid =
            poly_uop2(ctx, POLY_OP_AND, POLY_BOOL, ge_begin, lt_end, poly_arg_none());
        dim_valid = poly_graph_rewrite(ctx, dim_valid, poly_pm_pad_valid());
        if (!dim_valid) return false;
        /* Pinned indexing.py:137 keeps address and validity separable through
         * UOp.get_idx/get_valid: valid.where(r-off, Invalid). */
        in_rngs[i] = poly_uop3(
            ctx, POLY_OP_WHERE, POLY_INDEX, dim_valid, shifted, invalid, poly_arg_none()
        );
      }
      if (!pad_wrapper_valid_from_coords(ctx, in_rngs, n, n_out, &valid)) return false;
      if (valid_out) *valid_out = valid;
      *n_in_out = n;
      return true;
    }

    /* Legacy imported/raw pair-tuple PAD. */
    if (arg.kind != POLY_ARG_PAIR_TUPLE) return false;
    int n = arg.pair_tuple.n;
    PolyUOp *valid = NULL;
    PolyUOp *zero = index_const(ctx, 0);
    PolyUOp *invalid = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_invalid());

    for (int i = 0; i < n; i++) {
      if (i >= n_out) {
        in_rngs[i] = zero;
        continue;
      }
      int64_t begin = arg.pair_tuple.pairs[i][0];
      if (begin == 0 && arg.pair_tuple.pairs[i][1] == 0) {
        in_rngs[i] = out_rngs[i];
        continue;
      }
      PolyUOp *off = index_const(ctx, begin);
      PolyUOp *shifted = poly_uop2(
          ctx, POLY_OP_ADD, POLY_INDEX, to_index_dtype(ctx, out_rngs[i]),
          index_neg_like_tinygrad(ctx, off), poly_arg_none()
      );
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
      PolyUOp *true_const = poly_uop0(ctx, POLY_OP_CONST, POLY_BOOL, poly_arg_bool(true));
      PolyUOp *lt_begin =
          poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, to_index_dtype(ctx, out_rngs[i]), begin_c, poly_arg_none());
      PolyUOp *ge_begin =
          poly_uop2(ctx, POLY_OP_CMPNE, POLY_BOOL, lt_begin, true_const, poly_arg_none());
      PolyUOp *lt_dim =
          poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, to_index_dtype(ctx, out_rngs[i]), end_c, poly_arg_none());
      PolyUOp *dv = poly_uop2(ctx, POLY_OP_AND, POLY_BOOL, ge_begin, lt_dim, poly_arg_none());
      dv = poly_graph_rewrite(ctx, dv, poly_pm_pad_valid());
      if (!dv) return false;
      /* Pinned indexing.py:137 preserves invalidity in the coordinate. The
       * late gater moves this predicate onto LOAD/STORE before rendering. */
      in_rngs[i] =
          poly_uop3(ctx, POLY_OP_WHERE, POLY_INDEX, dv, shifted, invalid, poly_arg_none());
    }
    if (!pad_wrapper_valid_from_coords(ctx, in_rngs, n, n_out, &valid)) return false;
    if (valid_out) *valid_out = valid;
    *n_in_out = in_shape.ndim;
    return true;
  }

  default:
    fprintf(stderr, "polygrad: indexing: unsupported movement op %s\n", poly_op_name(op));
    return false;
  }
}
