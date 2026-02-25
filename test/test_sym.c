/*
 * test_sym.c — Tests for symbolic simplification + ALU constant folding
 */

#include "test_harness.h"
#include "../src/pat.h"

/* ── Helper: apply symbolic_simple via graph_rewrite ──────────────────── */

static PolyUOp *simplify(PolyCtx *ctx, PolyUOp *root) {
  return poly_graph_rewrite(ctx, root, poly_symbolic_simple());
}

/* ── ALU constant fold tests ──────────────────────────────────────────── */

TEST(alu, fold_add_int) {
  PolyArg ops[2] = { poly_arg_int(2), poly_arg_int(3) };
  PolyArg r = poly_exec_alu(POLY_OP_ADD, POLY_INT32, ops, 2);
  ASSERT_INT_EQ(r.i, 5);
  PASS();
}

TEST(alu, fold_mul_int) {
  PolyArg ops[2] = { poly_arg_int(4), poly_arg_int(7) };
  PolyArg r = poly_exec_alu(POLY_OP_MUL, POLY_INT32, ops, 2);
  ASSERT_INT_EQ(r.i, 28);
  PASS();
}

TEST(alu, fold_neg_float) {
  PolyArg ops[1] = { poly_arg_float(3.14) };
  PolyArg r = poly_exec_alu(POLY_OP_NEG, POLY_FLOAT32, ops, 1);
  ASSERT_FLOAT_EQ(r.f, -3.14, 1e-6);
  PASS();
}

TEST(alu, fold_add_float) {
  PolyArg ops[2] = { poly_arg_float(1.5), poly_arg_float(2.5) };
  PolyArg r = poly_exec_alu(POLY_OP_ADD, POLY_FLOAT32, ops, 2);
  ASSERT_FLOAT_EQ(r.f, 4.0, 1e-6);
  PASS();
}

TEST(alu, fold_idiv) {
  PolyArg ops[2] = { poly_arg_int(7), poly_arg_int(3) };
  PolyArg r = poly_exec_alu(POLY_OP_IDIV, POLY_INT32, ops, 2);
  ASSERT_INT_EQ(r.i, 2);
  PASS();
}

TEST(alu, fold_mod) {
  PolyArg ops[2] = { poly_arg_int(7), poly_arg_int(3) };
  PolyArg r = poly_exec_alu(POLY_OP_MOD, POLY_INT32, ops, 2);
  ASSERT_INT_EQ(r.i, 1);
  PASS();
}

TEST(alu, fold_cmplt) {
  PolyArg ops[2] = { poly_arg_int(2), poly_arg_int(5) };
  PolyArg r = poly_exec_alu(POLY_OP_CMPLT, POLY_INT32, ops, 2);
  ASSERT_TRUE(r.b == true);
  ops[0] = poly_arg_int(5);
  r = poly_exec_alu(POLY_OP_CMPLT, POLY_INT32, ops, 2);
  ASSERT_TRUE(r.b == false);
  PASS();
}

TEST(alu, fold_where) {
  PolyArg ops[3] = { poly_arg_bool(true), poly_arg_int(10), poly_arg_int(20) };
  PolyArg r = poly_exec_alu(POLY_OP_WHERE, POLY_INT32, ops, 3);
  ASSERT_INT_EQ(r.i, 10);
  ops[0] = poly_arg_bool(false);
  r = poly_exec_alu(POLY_OP_WHERE, POLY_INT32, ops, 3);
  ASSERT_INT_EQ(r.i, 20);
  PASS();
}

/* ── Symbolic simplification tests ────────────────────────────────────── */

TEST(sym, add_zero) {
  /* x + 0 -> x */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(5));
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, x, zero, poly_arg_none());

  PolyUOp *r = simplify(ctx, add);
  ASSERT_PTR_EQ(r, x);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, add_zero_reversed) {
  /* 0 + x -> x (commutative) */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(5));
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, zero, x, poly_arg_none());

  PolyUOp *r = simplify(ctx, add);
  ASSERT_PTR_EQ(r, x);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, mul_one) {
  /* x * 1 -> x */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(42));
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(1));
  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, POLY_INT32, x, one, poly_arg_none());

  PolyUOp *r = simplify(ctx, mul);
  ASSERT_PTR_EQ(r, x);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, mul_zero) {
  /* x * 0 -> 0 */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(42));
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, POLY_INT32, x, zero, poly_arg_none());

  PolyUOp *r = simplify(ctx, mul);
  ASSERT_TRUE(r->op == POLY_OP_CONST);
  ASSERT_INT_EQ(r->arg.i, 0);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, const_fold_add) {
  /* 2 + 3 -> 5 */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(2));
  PolyUOp *b = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(3));
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, a, b, poly_arg_none());

  PolyUOp *r = simplify(ctx, add);
  ASSERT_TRUE(r->op == POLY_OP_CONST);
  ASSERT_INT_EQ(r->arg.i, 5);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, const_fold_neg) {
  /* NEG(3.0) -> -3.0 */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *c = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(3.0));
  PolyUOp *neg = poly_uop1(ctx, POLY_OP_NEG, POLY_FLOAT32, c, poly_arg_none());

  PolyUOp *r = simplify(ctx, neg);
  ASSERT_TRUE(r->op == POLY_OP_CONST);
  ASSERT_FLOAT_EQ(r->arg.f, -3.0, 1e-6);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, double_neg) {
  /* NEG(NEG(x)) -> x  (use non-CONST source to avoid const-fold firing first) */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_uop0(ctx, POLY_OP_DEFINE_VAR, POLY_FLOAT32, poly_arg_str("v"));
  PolyUOp *neg1 = poly_uop1(ctx, POLY_OP_NEG, POLY_FLOAT32, x, poly_arg_none());
  PolyUOp *neg2 = poly_uop1(ctx, POLY_OP_NEG, POLY_FLOAT32, neg1, poly_arg_none());

  PolyUOp *r = simplify(ctx, neg2);
  ASSERT_PTR_EQ(r, x);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, div_self) {
  /* x // x -> 1 */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(7));
  PolyUOp *div = poly_uop2(ctx, POLY_OP_IDIV, POLY_INT32, x, x, poly_arg_none());

  PolyUOp *r = simplify(ctx, div);
  ASSERT_TRUE(r->op == POLY_OP_CONST);
  ASSERT_INT_EQ(r->arg.i, 1);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, mod_self) {
  /* x % x -> 0 */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(5));
  PolyUOp *mod = poly_uop2(ctx, POLY_OP_MOD, POLY_INT32, x, x, poly_arg_none());

  PolyUOp *r = simplify(ctx, mod);
  ASSERT_TRUE(r->op == POLY_OP_CONST);
  ASSERT_INT_EQ(r->arg.i, 0);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, combined_rewrite) {
  /* (a + 0) * 1 -> a  (requires two rules, cascading) */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(42));
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(1));
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, a, zero, poly_arg_none());
  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, POLY_INT32, add, one, poly_arg_none());

  PolyUOp *r = simplify(ctx, mul);
  /* Should simplify to just 'a' */
  ASSERT_PTR_EQ(r, a);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, const_fold_mul_float) {
  /* 2.0 * 3.5 -> 7.0 */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(2.0));
  PolyUOp *b = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(3.5));
  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, a, b, poly_arg_none());

  PolyUOp *r = simplify(ctx, mul);
  ASSERT_TRUE(r->op == POLY_OP_CONST);
  ASSERT_FLOAT_EQ(r->arg.f, 7.0, 1e-6);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, where_same_branches) {
  /* WHERE(cond, v, v) -> v */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *cond = poly_uop0(ctx, POLY_OP_CONST, POLY_BOOL, poly_arg_bool(true));
  PolyUOp *v = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(99));
  PolyUOp *wh = poly_uop3(ctx, POLY_OP_WHERE, POLY_INT32, cond, v, v, poly_arg_none());

  PolyUOp *r = simplify(ctx, wh);
  ASSERT_PTR_EQ(r, v);
  poly_ctx_destroy(ctx);
  PASS();
}
