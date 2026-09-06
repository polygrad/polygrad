/*
 * mixin/elementwise.c -- ElementwiseMixin UOp composition
 *
 * Mirrors tinygrad/mixin/elementwise.py. Tensor dual-root construction calls
 * these same promotion rules independently for logical and physical roots.
 */

#include "polygrad.h"

/* C helper for current ElementwiseMixin._broadcasted.promote. It is public
 * inside the core because Polygrad applies one promotion to each retained
 * logical/physical root (mixin/elementwise.py:21-29). */
PolyUOp *poly_elementwise_promote(PolyCtx *ctx, PolyUOp *root, PolyDType common) {
  if (!ctx || !root) return NULL;
  PolyUOp *base = poly_uop_base(root);
  /* Invalid is a sentinel, not a bool value to cast to the common dtype. */
  if (base->op == POLY_OP_CONST && base->arg.kind == POLY_ARG_INVALID) return root;
  if (poly_dtype_is_weak(root->dtype) && base->op == POLY_OP_CONST) {
    return poly_const_like_dtype(ctx, root, base->arg, poly_dtype_weak(common));
  }
  return poly_dtype_eq(root->dtype, common) ? root : poly_cast(ctx, root, common);
}

/* Current ElementwiseMixin._broadcasted applies least_upper_dtype while
 * keeping bare weak CONSTs weak (mixin/elementwise.py:21-29). */
bool poly_broadcasted_pair(PolyCtx *ctx, PolyUOp **a, PolyUOp **b) {
  if (!ctx || !a || !b || !*a || !*b) return false;

  PolyDType common;
  if (!poly_dtype_least_upper((*a)->dtype, (*b)->dtype, &common)) return false;
  *a = poly_elementwise_promote(ctx, *a, common);
  *b = poly_elementwise_promote(ctx, *b, common);
  return *a != NULL && *b != NULL;
}

/* Current ElementwiseMixin._binop promotes once, then constructs the selected
 * ALU UOp (mixin/elementwise.py:32-34). */
PolyUOp *poly_binop(PolyCtx *ctx, PolyOps op, PolyUOp *a, PolyUOp *b) {
  if (!poly_broadcasted_pair(ctx, &a, &b)) return NULL;
  return poly_alu2(ctx, op, a, b);
}
