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

/* Broadcasting binary ops */

PolyUOp *poly_add(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  if (!poly_broadcasted_pair(ctx, &a, &b)) return NULL;
  return poly_alu2(ctx, POLY_OP_ADD, a, b);
}

PolyUOp *poly_sub(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  if (!poly_broadcasted_pair(ctx, &a, &b)) return NULL;

  /* Current ElementwiseMixin.sub is a.alu(ADD, -b) after the one
   * _broadcasted promotion pass. Negation keeps its scalar -1 weak and UOp
   * shape inference owns any shape broadcast (mixin/elementwise.py:104-119). */
  PolyUOp *neg_b = NULL;
  if (poly_dtype_is_bool(b->dtype)) {
    neg_b = poly_logical_not(ctx, b);
  } else {
    PolyUOp *minus_one = poly_const_typed(ctx, poly_dtype_weak(b->dtype), -1.0);
    neg_b = minus_one ? poly_alu2(ctx, POLY_OP_MUL, b, minus_one) : NULL;
  }
  return neg_b ? poly_alu2(ctx, POLY_OP_ADD, a, neg_b) : NULL;
}

PolyUOp *poly_mul(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  if (!poly_broadcasted_pair(ctx, &a, &b)) return NULL;
  return poly_alu2(ctx, POLY_OP_MUL, a, b);
}

/* Comparisons (broadcasting) */

/* Logical NOT — `CMPNE(x, CONST(true))` for
 * bool inputs, matching tinygrad's `logical_not()` after CAST elision
 * (mixin/elementwise.py:39-47 + symbolic.py:93-131). Raw `NEG(bool)` retains
 * arithmetic NEG semantics and is not a second logical-NOT spelling. */
PolyUOp *poly_logical_not(PolyCtx *ctx, PolyUOp *x) {
  PolyUOp *t = poly_const_typed(ctx, POLY_BOOL, 1);
  return poly_alu2(ctx, POLY_OP_CMPNE, x, t);
}

/* All comparison helpers return BOOL, mirroring tinygrad's
 * mixin/elementwise.py:218-247:
 *   eq(a,b) = (a != b).logical_not()
 *   ne(a,b) = CMPNE(a,b)
 *   gt(a,b) = CMPLT(b,a)         (operand swap)
 *   lt(a,b) = CMPLT(a,b)
 *   ge(a,b) = (a < b).logical_not()
 *   le(a,b) = (a > b).logical_not() = (b < a).logical_not()
 *
 * Polygrad previously had ge/le returning float WHERE(0,1); fixed in P5
 * for tinygrad parity and to let Phase D's reduce_collapse Rule 4 match
 * polygrad's tril/triu masks. */
PolyUOp *poly_eq(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  /* Pinned Tensor.eq reaches _binop/_broadcasted, which broadcasts and then
   * promotes both operands with least_upper_dtype before CMPNE
   * (mixin/elementwise.py:324-325, mixin/__init__.py:439-449). */
  if (!poly_broadcasted_pair(ctx, &a, &b)) return NULL;
  PolyUOp *ne = poly_alu2(ctx, POLY_OP_CMPNE, a, b);
  return poly_logical_not(ctx, ne);
}

PolyUOp *poly_where_op(PolyCtx *ctx, PolyUOp *cond, PolyUOp *x, PolyUOp *y) {
  if (!cond || !poly_broadcasted_pair(ctx, &x, &y)) return NULL;
  /* tinygrad Tensor.where casts non-bool conditions to bool before building
   * Ops.WHERE. Keeping that in the core constructor preserves the expected
   * CMPNE(cond, 0) node in helper graphs such as nonzero-value padding. */
  if (!poly_dtype_is_bool(cond->dtype)) cond = poly_cast(ctx, cond, POLY_BOOL);
  return poly_alu3(ctx, POLY_OP_WHERE, cond, x, y);
}
