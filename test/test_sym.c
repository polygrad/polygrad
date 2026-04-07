/*
 * test_sym.c — Tests for symbolic simplification + ALU constant folding
 */

#include "test_harness.h"
#include "../src/pat.h"
#include "../src/tensor.h"

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

/* ── poly_uop_minmax tinygrad parity tests ────────────────────────────
 *
 * Every assertion below corresponds to one row of
 * test/parity_scripts/tg_minmax_gt.py output, captured against
 * tinygrad/uop/ops.py:856-897. To re-verify after a tinygrad bump:
 *
 *   PYTHONPATH=references/tinygrad_latest conda run -n tiny \
 *     python test/parity_scripts/tg_minmax_gt.py
 *
 * Then update the literal values below if tinygrad's semantics changed. */

static PolyUOp *mk_const(PolyCtx *ctx, int64_t v) {
  return poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(v));
}
static PolyUOp *mk_dvar(PolyCtx *ctx, const char *name, int64_t lo, int64_t hi) {
  return poly_uop0(ctx, POLY_OP_DEFINE_VAR, POLY_INT32,
                   poly_arg_define_var(name, lo, hi));
}
static PolyUOp *mk_range(PolyCtx *ctx, int64_t n, int64_t axis_id) {
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(n));
  return poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound,
                   poly_arg_range(axis_id, POLY_AXIS_LOOP));
}
/* check_mm queries minmax and asserts both bounds match the tinygrad
 * ground truth. Implemented as a macro because ASSERT_INT_EQ touches the
 * test-local _passed/_failed counters that only exist inside a TEST. */
#define check_mm(ctx, u, want_lo, want_hi, label) do {                       \
  int64_t _mm_lo, _mm_hi;                                                    \
  poly_uop_minmax((ctx), (u), &_mm_lo, &_mm_hi);                             \
  if (_mm_lo != (want_lo) || _mm_hi != (want_hi)) {                          \
    fprintf(stderr, "%s: got [%lld..%lld], want [%lld..%lld]\n",             \
            (label), (long long)_mm_lo, (long long)_mm_hi,                   \
            (long long)(want_lo), (long long)(want_hi));                     \
  }                                                                          \
  ASSERT_INT_EQ(_mm_lo, (want_lo));                                          \
  ASSERT_INT_EQ(_mm_hi, (want_hi));                                          \
} while (0)

TEST(sym, minmax_const_pos) {
  PolyCtx *ctx = poly_ctx_new();
  check_mm(ctx, mk_const(ctx, 5), 5, 5, "CONST 5");
  poly_ctx_destroy(ctx);
  PASS();
}
TEST(sym, minmax_const_neg) {
  PolyCtx *ctx = poly_ctx_new();
  check_mm(ctx, mk_const(ctx, -2), -2, -2, "CONST -2");
  poly_ctx_destroy(ctx);
  PASS();
}
TEST(sym, minmax_define_var_pos) {
  PolyCtx *ctx = poly_ctx_new();
  check_mm(ctx, mk_dvar(ctx, "x", 2, 7), 2, 7, "DEFINE_VAR[2..7]");
  poly_ctx_destroy(ctx);
  PASS();
}
TEST(sym, minmax_define_var_mixed_sign) {
  PolyCtx *ctx = poly_ctx_new();
  check_mm(ctx, mk_dvar(ctx, "y", -3, 4), -3, 4, "DEFINE_VAR[-3..4]");
  poly_ctx_destroy(ctx);
  PASS();
}
TEST(sym, minmax_range_10) {
  PolyCtx *ctx = poly_ctx_new();
  check_mm(ctx, mk_range(ctx, 10, 0), 0, 9, "RANGE(10)");
  poly_ctx_destroy(ctx);
  PASS();
}
TEST(sym, minmax_range_1_degenerate) {
  PolyCtx *ctx = poly_ctx_new();
  check_mm(ctx, mk_range(ctx, 1, 1), 0, 0, "RANGE(1)");
  poly_ctx_destroy(ctx);
  PASS();
}

/* ── ADD / SUB ── */
TEST(sym, minmax_add_r_const) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *u = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, mk_range(ctx,10,0), mk_const(ctx,3), poly_arg_none());
  check_mm(ctx, u, 3, 12, "r+3");
  poly_ctx_destroy(ctx);
  PASS();
}
TEST(sym, minmax_add_r_dvar) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *u = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, mk_range(ctx,10,0), mk_dvar(ctx,"x",2,7), poly_arg_none());
  check_mm(ctx, u, 2, 16, "r+dv");
  poly_ctx_destroy(ctx);
  PASS();
}
TEST(sym, minmax_sub_r_const) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *u = poly_uop2(ctx, POLY_OP_SUB, POLY_INT32, mk_range(ctx,10,0), mk_const(ctx,3), poly_arg_none());
  check_mm(ctx, u, -3, 6, "r-3");
  poly_ctx_destroy(ctx);
  PASS();
}
TEST(sym, minmax_sub_const_r) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *u = poly_uop2(ctx, POLY_OP_SUB, POLY_INT32, mk_const(ctx,3), mk_range(ctx,10,0), poly_arg_none());
  check_mm(ctx, u, -6, 3, "3-r");
  poly_ctx_destroy(ctx);
  PASS();
}
TEST(sym, minmax_sub_dv_dvn) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *dv  = mk_dvar(ctx,"x",2,7);
  PolyUOp *dvn = mk_dvar(ctx,"y",-3,4);
  PolyUOp *u = poly_uop2(ctx, POLY_OP_SUB, POLY_INT32, dv, dvn, poly_arg_none());
  check_mm(ctx, u, -2, 10, "dv-dvn");
  poly_ctx_destroy(ctx);
  PASS();
}

/* ── MUL (4-corner) ── */
TEST(sym, minmax_mul_r_3) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *u = poly_uop2(ctx, POLY_OP_MUL, POLY_INT32, mk_range(ctx,10,0), mk_const(ctx,3), poly_arg_none());
  check_mm(ctx, u, 0, 27, "r*3");
  poly_ctx_destroy(ctx);
  PASS();
}
TEST(sym, minmax_mul_r_neg2) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *u = poly_uop2(ctx, POLY_OP_MUL, POLY_INT32, mk_range(ctx,10,0), mk_const(ctx,-2), poly_arg_none());
  check_mm(ctx, u, -18, 0, "r*-2");
  poly_ctx_destroy(ctx);
  PASS();
}
TEST(sym, minmax_mul_dv_dvn_mixed_signs) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *dv  = mk_dvar(ctx,"x",2,7);
  PolyUOp *dvn = mk_dvar(ctx,"y",-3,4);
  PolyUOp *u = poly_uop2(ctx, POLY_OP_MUL, POLY_INT32, dv, dvn, poly_arg_none());
  /* corners: 2*-3=-6, 2*4=8, 7*-3=-21, 7*4=28 -> [-21, 28] */
  check_mm(ctx, u, -21, 28, "dv*dvn");
  poly_ctx_destroy(ctx);
  PASS();
}
TEST(sym, minmax_mul_dvn_dvn_squared) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *dvn = mk_dvar(ctx,"y",-3,4);
  PolyUOp *u = poly_uop2(ctx, POLY_OP_MUL, POLY_INT32, dvn, dvn, poly_arg_none());
  /* corners: -3*-3=9, -3*4=-12, 4*-3=-12, 4*4=16 -> [-12, 16] */
  check_mm(ctx, u, -12, 16, "dvn*dvn");
  poly_ctx_destroy(ctx);
  PASS();
}

/* ── IDIV ── */
TEST(sym, minmax_idiv_r_3) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *u = poly_uop2(ctx, POLY_OP_IDIV, POLY_INT32, mk_range(ctx,10,0), mk_const(ctx,3), poly_arg_none());
  check_mm(ctx, u, 0, 3, "r//3");
  poly_ctx_destroy(ctx);
  PASS();
}

/* ── MOD ── */
TEST(sym, minmax_mod_r_3) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *u = poly_uop2(ctx, POLY_OP_MOD, POLY_INT32, mk_range(ctx,10,0), mk_const(ctx,3), poly_arg_none());
  check_mm(ctx, u, 0, 2, "r%3");
  poly_ctx_destroy(ctx);
  PASS();
}
TEST(sym, minmax_mod_dvn_3_mixed) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *dvn = mk_dvar(ctx,"y",-3,4);
  PolyUOp *u = poly_uop2(ctx, POLY_OP_MOD, POLY_INT32, dvn, mk_const(ctx,3), poly_arg_none());
  /* tinygrad: dvn vmin=-3 vmax=4, mod 3 -> middle branch -> [-2, 2] */
  check_mm(ctx, u, -2, 2, "dvn%3");
  poly_ctx_destroy(ctx);
  PASS();
}

/* ── SHL / SHR ── */
TEST(sym, minmax_shl_r_2) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *u = poly_uop2(ctx, POLY_OP_SHL, POLY_INT32, mk_range(ctx,10,0), mk_const(ctx,2), poly_arg_none());
  check_mm(ctx, u, 0, 36, "r<<2");
  poly_ctx_destroy(ctx);
  PASS();
}
TEST(sym, minmax_shr_r_1) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *u = poly_uop2(ctx, POLY_OP_SHR, POLY_INT32, mk_range(ctx,10,0), mk_const(ctx,1), poly_arg_none());
  check_mm(ctx, u, 0, 4, "r>>1");
  poly_ctx_destroy(ctx);
  PASS();
}

/* ── XOR with -1 (bitwise NOT) ── */
TEST(sym, minmax_xor_r_neg1) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *u = poly_uop2(ctx, POLY_OP_XOR, POLY_INT32, mk_range(ctx,10,0), mk_const(ctx,-1), poly_arg_none());
  /* ~r where r in [0,9] -> [~9, ~0] = [-10, -1] */
  check_mm(ctx, u, -10, -1, "r xor -1");
  poly_ctx_destroy(ctx);
  PASS();
}

/* ── AND (int with non-negative const) ── */
TEST(sym, minmax_and_r_5) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *u = poly_uop2(ctx, POLY_OP_AND, POLY_INT32, mk_range(ctx,10,0), mk_const(ctx,5), poly_arg_none());
  /* tinygrad: r vmin=0 (>=0), vmax=min(9,5)=5 -> [0, 5] */
  check_mm(ctx, u, 0, 5, "r & 5");
  poly_ctx_destroy(ctx);
  PASS();
}

/* ── MAX ── */
TEST(sym, minmax_max_r_5) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *u = poly_uop2(ctx, POLY_OP_MAX, POLY_INT32, mk_range(ctx,10,0), mk_const(ctx,5), poly_arg_none());
  check_mm(ctx, u, 5, 9, "max(r,5)");
  poly_ctx_destroy(ctx);
  PASS();
}
TEST(sym, minmax_max_dv_5) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *u = poly_uop2(ctx, POLY_OP_MAX, POLY_INT32, mk_dvar(ctx,"x",2,7), mk_const(ctx,5), poly_arg_none());
  check_mm(ctx, u, 5, 7, "max(dv,5)");
  poly_ctx_destroy(ctx);
  PASS();
}

/* ── CMPLT (bool bounds) ── */
TEST(sym, minmax_cmplt_r_5) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *u = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, mk_range(ctx,10,0), mk_const(ctx,5), poly_arg_none());
  /* (a1<b0) = (9<5)=false=0, (a0<b1) = (0<5)=true=1 -> [0,1] */
  check_mm(ctx, u, 0, 1, "r<5");
  poly_ctx_destroy(ctx);
  PASS();
}
TEST(sym, minmax_cmplt_r_neg1_static_false) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *u = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, mk_range(ctx,10,0), mk_const(ctx,-1), poly_arg_none());
  /* (a1<b0) = (9<-1)=false=0, (a0<b1) = (0<-1)=false=0 -> [0,0] */
  check_mm(ctx, u, 0, 0, "r<-1");
  poly_ctx_destroy(ctx);
  PASS();
}
TEST(sym, minmax_cmplt_r_20_static_true) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *u = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, mk_range(ctx,10,0), mk_const(ctx,20), poly_arg_none());
  /* (a1<b0) = (9<20)=true=1, (a0<b1) = (0<20)=true=1 -> [1,1] */
  check_mm(ctx, u, 1, 1, "r<20");
  poly_ctx_destroy(ctx);
  PASS();
}

/* ── CMPNE ── */
TEST(sym, minmax_cmpne_r_neg1) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *u = poly_uop2(ctx, POLY_OP_CMPNE, POLY_BOOL, mk_range(ctx,10,0), mk_const(ctx,-1), poly_arg_none());
  /* def_ne: (a1<b0)||(b1<a0) = (9<-1)||(-1<0) = false||true = true
   * all_eq: false (a0=0 != a1=9). vmin=1, vmax=1 */
  check_mm(ctx, u, 1, 1, "r!=-1");
  poly_ctx_destroy(ctx);
  PASS();
}

/* ── WHERE (int branches) ── */
TEST(sym, minmax_where_int) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *cond = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, mk_range(ctx,10,0), mk_const(ctx,5), poly_arg_none());
  PolyUOp *u = poly_uop3(ctx, POLY_OP_WHERE, POLY_INT32, cond, mk_const(ctx,3), mk_const(ctx,5), poly_arg_none());
  check_mm(ctx, u, 3, 5, "WHERE(r<5,3,5)");
  poly_ctx_destroy(ctx);
  PASS();
}
TEST(sym, minmax_where_dv_dvn) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *cond = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, mk_range(ctx,10,0), mk_const(ctx,5), poly_arg_none());
  PolyUOp *dv  = mk_dvar(ctx,"x",2,7);
  PolyUOp *dvn = mk_dvar(ctx,"y",-3,4);
  PolyUOp *u = poly_uop3(ctx, POLY_OP_WHERE, POLY_INT32, cond, dv, dvn, poly_arg_none());
  /* min(2,-3)=-3, max(7,4)=7 -> [-3, 7] */
  check_mm(ctx, u, -3, 7, "WHERE(r<5,dv,dvn)");
  poly_ctx_destroy(ctx);
  PASS();
}

/* ── CAST (monotone narrowing to signed) ── */
TEST(sym, minmax_cast_dv_to_int16) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *u = poly_uop1(ctx, POLY_OP_CAST, POLY_INT16, mk_dvar(ctx,"x",2,7), poly_arg_none());
  check_mm(ctx, u, 2, 7, "CAST dv i32->i16");
  poly_ctx_destroy(ctx);
  PASS();
}
TEST(sym, minmax_cast_r_to_int8) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *u = poly_uop1(ctx, POLY_OP_CAST, POLY_INT8, mk_range(ctx,10,0), poly_arg_none());
  check_mm(ctx, u, 0, 9, "CAST r i32->i8");
  poly_ctx_destroy(ctx);
  PASS();
}

/* ── Diamond / chained (memoization stress) ── */
TEST(sym, minmax_diamond_r3_mul2) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *r = mk_range(ctx,10,0);
  PolyUOp *r3 = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, r, mk_const(ctx,3), poly_arg_none());
  PolyUOp *u = poly_uop2(ctx, POLY_OP_MUL, POLY_INT32, r3, mk_const(ctx,2), poly_arg_none());
  /* (r+3)*2 with r in [0,9]: r+3 in [3,12], *2 in [6,24] */
  check_mm(ctx, u, 6, 24, "(r+3)*2");
  poly_ctx_destroy(ctx);
  PASS();
}

/* ── Phase D end-to-end gate ──
 * poly_arange returns a *Tensor-level* UOp graph composed of movement ops
 * (RESHAPE / EXPAND / PAD / PERMUTE / SHRINK) over a CONST + REDUCE_AXIS
 * + ADD. Tinygrad's _min_max has no special case for any of these (they
 * live in GroupOp.Movement, not GroupOp.Binary), so the bounds at this
 * level fall through to the dtype default. Tight bounds emerge only after
 * rangeify replaces movement ops with RANGE/INDEX — which is exactly where
 * Phase D's reduce_collapse driver runs.
 *
 * This test is therefore a smoke check: poly_uop_minmax must not crash
 * on a real arange graph and must respect the dtype-bounds fallback. */
TEST(sym, minmax_arange_0_to_5_smoke) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *ar = poly_arange(ctx, 0.0, 5.0, 1.0);
  ASSERT_NOT_NULL(ar);
  int64_t lo, hi;
  poly_uop_minmax(ctx, ar, &lo, &hi);
  /* Loose bounds: Tensor-level graph hasn't been rangeified yet. */
  ASSERT_TRUE(lo <= 0);
  ASSERT_TRUE(hi >= 4);
  poly_ctx_destroy(ctx);
  PASS();
}

/* ── PolyUOpCache lifetime + reuse ── */
TEST(sym, minmax_cache_reuse_yields_same) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOpCache *c = poly_uop_cache_new();
  ASSERT_NOT_NULL(c);
  PolyUOp *u = poly_uop2(ctx, POLY_OP_MUL, POLY_INT32,
                          mk_dvar(ctx,"x",2,7), mk_dvar(ctx,"y",-3,4),
                          poly_arg_none());
  int64_t lo1, hi1, lo2, hi2;
  poly_uop_minmax_ex(ctx, u, c, &lo1, &hi1);
  poly_uop_minmax_ex(ctx, u, c, &lo2, &hi2);  /* second query should hit cache */
  ASSERT_INT_EQ(lo1, lo2);
  ASSERT_INT_EQ(hi1, hi2);
  ASSERT_INT_EQ(lo1, -21);
  ASSERT_INT_EQ(hi1, 28);
  poly_uop_cache_destroy(c);
  poly_ctx_destroy(ctx);
  PASS();
}
