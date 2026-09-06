/* UOp movement cleanup (tinygrad/uop/movement.py). */

#include "uop/movement.h"
#include <stdlib.h>

static PolyUOp *replace_srcs(PolyCtx *ctx, PolyUOp *u, PolyUOp **src, int n_src) {
  return poly_uop_tagged_arg(ctx, u->op, u->dtype, src, n_src, u->arg, u->tag, u->tag_arg);
}

static bool shape_equal(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  int a_ndim = poly_uop_ndim(ctx, a), b_ndim = poly_uop_ndim(ctx, b);
  if (a_ndim < 0 || a_ndim != b_ndim) return false;
  for (int i = 0; i < a_ndim; i++) {
    PolyUOp *adim = poly_uop_shape_dim(ctx, a, i);
    PolyUOp *bdim = poly_uop_shape_dim(ctx, b, i);
    if (adim == bdim) continue;
    int64_t aval = 0, bval = 0;
    if (!adim || !bdim || poly_uop_const_i64(adim, &aval) != 0 ||
        poly_uop_const_i64(bdim, &bval) != 0 || aval != bval)
      return false;
  }
  return true;
}

static PolyUOp *merge_adjacent_reshape(
    PolyCtx *ctx,
    PolyUOp *reshape,
    const PolyBindings *bindings
) {
  (void)bindings;
  if (!reshape || reshape->op != POLY_OP_RESHAPE || reshape->n_src != 2) return NULL;
  PolyUOp *inner = reshape->src[0];
  if (!inner || inner->op != POLY_OP_RESHAPE || inner->n_src != 2) return NULL;
  PolyUOp *src[] = {inner->src[0], reshape->src[1]};
  return replace_srcs(ctx, reshape, src, 2);
}

static PolyUOp *remove_noop_reshape(PolyCtx *ctx, PolyUOp *reshape, const PolyBindings *bindings) {
  (void)bindings;
  return reshape && reshape->op == POLY_OP_RESHAPE && reshape->n_src == 2 &&
                 shape_equal(ctx, reshape->src[0], reshape)
             ? reshape->src[0]
             : NULL;
}

static PolyUOp *merge_permute(PolyCtx *ctx, PolyUOp *permute, const PolyBindings *bindings) {
  (void)bindings;
  if (!permute || permute->op != POLY_OP_PERMUTE || permute->n_src != 1 ||
      permute->arg.kind != POLY_ARG_INT_TUPLE)
    return NULL;
  PolyUOp *inner = permute->src[0];
  if (!inner || inner->op != POLY_OP_PERMUTE || inner->n_src != 1 ||
      inner->arg.kind != POLY_ARG_INT_TUPLE || permute->arg.int_tuple.n != inner->arg.int_tuple.n)
    return NULL;
  int n = permute->arg.int_tuple.n;
  int64_t composed[POLY_MAX_DIMS];
  if (n < 0 || n > POLY_MAX_DIMS) return NULL;
  for (int i = 0; i < n; i++) {
    int64_t axis = permute->arg.int_tuple.vals[i];
    if (axis < 0 || axis >= n) return NULL;
    composed[i] = inner->arg.int_tuple.vals[axis];
  }
  PolyArg arg = {.kind = POLY_ARG_INT_TUPLE, .int_tuple = {.vals = composed, .n = n}};
  return poly_uop_tagged_arg(
      ctx, inner->op, inner->dtype, inner->src, inner->n_src, arg, inner->tag, inner->tag_arg
  );
}

static PolyUOp *remove_noop_permute(PolyCtx *ctx, PolyUOp *permute, const PolyBindings *bindings) {
  (void)ctx;
  (void)bindings;
  if (!permute || permute->op != POLY_OP_PERMUTE || permute->n_src != 1 ||
      permute->arg.kind != POLY_ARG_INT_TUPLE)
    return NULL;
  for (int i = 0; i < permute->arg.int_tuple.n; i++)
    if (permute->arg.int_tuple.vals[i] != i) return NULL;
  return permute->src[0];
}

static PolyUOp *stack_of_ordered_indexes(
    PolyCtx *ctx,
    PolyUOp *stack,
    const PolyBindings *bindings
) {
  (void)bindings;
  if (!stack || stack->op != POLY_OP_STACK || stack->n_src <= 0) return NULL;
  PolyUOp *base = NULL;
  for (int i = 0; i < stack->n_src; i++) {
    PolyUOp *index = stack->src[i];
    int64_t lane = -1;
    if (!index || index->op != POLY_OP_INDEX || index->n_src != 2 || !index->src[1] ||
        poly_uop_const_i64(index->src[1], &lane) != 0 || lane != i)
      return NULL;
    if (i == 0)
      base = index->src[0];
    else if (index->src[0] != base)
      return NULL;
  }
  return base && shape_equal(ctx, stack, base) ? base : NULL;
}

static PolyUOp *const_index_into_stack(PolyCtx *ctx, PolyUOp *index, const PolyBindings *bindings) {
  (void)bindings;
  if (!index || index->op != POLY_OP_INDEX || index->n_src < 2 || !index->src[0] ||
      index->src[0]->op != POLY_OP_STACK)
    return NULL;
  int64_t lane = 0;
  if (poly_uop_const_i64(index->src[1], &lane) != 0) return NULL;
  PolyUOp *stack = index->src[0];
  if (lane < 0) lane += stack->n_src;
  if (lane < 0 || lane >= stack->n_src) return NULL;
  if (index->n_src == 2) return stack->src[lane];
  return poly_uop_index(ctx, stack->src[lane], index->src + 2, index->n_src - 2);
}

static PolyUOp *index_on_index(PolyCtx *ctx, PolyUOp *index, const PolyBindings *bindings) {
  (void)bindings;
  /* Pinned mop_cleanup permits empty coordinate tuples on either INDEX;
   * all-scalar is then vacuously true, but the base source is still required. */
  if (!index || index->op != POLY_OP_INDEX || index->n_src < 1) return NULL;
  PolyUOp *inner = index->src[0];
  if (!inner || inner->op != POLY_OP_INDEX || inner->n_src < 1) return NULL;

  bool all_scalar = true;
  for (int i = 1; i < inner->n_src; i++)
    all_scalar &= poly_uop_ndim(ctx, inner->src[i]) == 0;
  for (int i = 1; i < index->n_src; i++)
    all_scalar &= poly_uop_ndim(ctx, index->src[i]) == 0;
  if (!all_scalar) return NULL;

  int n_coords = inner->n_src + index->n_src - 2;
  if (n_coords > POLY_MAX_DIMS) return NULL;
  PolyUOp *coords[POLY_MAX_DIMS];
  for (int i = 1; i < inner->n_src; i++)
    coords[i - 1] = inner->src[i];
  for (int i = 1; i < index->n_src; i++)
    coords[inner->n_src + i - 2] = index->src[i];
  return poly_uop_index(ctx, inner->src[0], coords, n_coords);
}

static PolyUOp *index_on_shaped_index(PolyCtx *ctx, PolyUOp *index, const PolyBindings *bindings) {
  (void)bindings;
  if (!index || index->op != POLY_OP_INDEX || index->n_src < 2) return NULL;
  PolyUOp *inner = index->src[0];
  if (!inner || inner->op != POLY_OP_INDEX || inner->n_src != 2 ||
      poly_uop_ndim(ctx, inner->src[1]) != index->n_src - 1)
    return NULL;
  PolyUOp *coordinate = poly_uop_index(ctx, inner->src[1], index->src + 1, index->n_src - 1);
  return coordinate ? poly_uop_index(ctx, inner->src[0], &coordinate, 1) : NULL;
}

static _Thread_local PolyPatternMatcher *g_mop_cleanup = NULL;

PolyPatternMatcher *poly_mop_cleanup(void) {
  if (g_mop_cleanup) return g_mop_cleanup;
  PolyNamedRule rules[] = {
      POLY_RULE(poly_upat_op(POLY_OP_RESHAPE, NULL, 0, NULL), merge_adjacent_reshape),
      POLY_RULE(poly_upat_op(POLY_OP_RESHAPE, NULL, 0, NULL), remove_noop_reshape),
      POLY_RULE(poly_upat_op(POLY_OP_PERMUTE, NULL, 0, NULL), merge_permute),
      POLY_RULE(poly_upat_op(POLY_OP_PERMUTE, NULL, 0, NULL), remove_noop_permute),
      POLY_RULE(poly_upat_op(POLY_OP_STACK, NULL, 0, NULL), stack_of_ordered_indexes),
      POLY_RULE(poly_upat_op(POLY_OP_INDEX, NULL, 0, NULL), const_index_into_stack),
      POLY_RULE(poly_upat_op(POLY_OP_INDEX, NULL, 0, NULL), index_on_index),
      POLY_RULE(poly_upat_op(POLY_OP_INDEX, NULL, 0, NULL), index_on_shaped_index),
  };
  g_mop_cleanup =
      poly_pm_thread_cache(poly_pm_new_named(rules, (int)(sizeof(rules) / sizeof(rules[0]))));
  return g_mop_cleanup;
}
