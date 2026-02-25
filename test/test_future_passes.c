/*
 * test_future_passes.c — Tests for tinygrad passes not yet ported to polygrad.
 *
 * These tests FAIL until each pass is ported. They track progress:
 *   - Missing transcendental decompositions (LOG2, SIN)
 *   - Missing late decompositions (MOD→AND, MULACC→MUL+ADD)
 *   - Missing symbolic rules (divmod cancel, bool algebra, nested WHERE, etc.)
 *   - Expander pass (UNROLL/CONTRACT expansion)
 *   - Devectorizer pass (VECTORIZE const fold)
 *
 * Tests for already-ported passes (EXP2, MAX→WHERE, MUL→SHL, IDIV→SHR,
 * double_mod, LOG2/SIN e2e) serve as regression tests and must always PASS.
 *
 * Run with filter:  build/polygrad_test sym_future
 *                   build/polygrad_test transcendental
 *                   build/polygrad_test decomp
 */

#include "test_harness.h"
#include "../src/codegen.h"

/* ── Helpers ─────────────────────────────────────────────────────────── */

static int count_ops_in(PolyCtx *ctx, PolyUOp *sink, PolyOps op) {
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, sink, &n_topo);
  int count = 0;
  for (int i = 0; i < n_topo; i++)
    if (topo[i]->op == op) count++;
  /* topo is arena-allocated, don't free */
  return count;
}

/* Build a simple unary kernel: out[i] = OP(in[i]) for i in [0, n) */
static PolyUOp *make_unary_kernel(PolyCtx *ctx, PolyOps op, int n) {
  PolyDType ptr_f32 = poly_dtype_ptr(POLY_FLOAT32, -1, POLY_ADDR_GLOBAL);
  PolyUOp *p0 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(0));
  PolyUOp *p1 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(1));
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(n));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));
  PolyUOp *idx0 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p0, range, poly_arg_none());
  PolyUOp *idx1 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p1, range, poly_arg_none());
  PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx0, poly_arg_none());
  PolyUOp *alu = poly_uop1(ctx, op, POLY_FLOAT32, load, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx1, alu, poly_arg_none());
  PolyUOp *end_src[2] = { store, range };
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  return poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());
}

/* Build a binary kernel: out[i] = in0[i] OP in1[i] for i in [0, n) */
static PolyUOp *make_binary_kernel(PolyCtx *ctx, PolyOps op, int n) {
  PolyDType ptr_f32 = poly_dtype_ptr(POLY_FLOAT32, -1, POLY_ADDR_GLOBAL);
  PolyUOp *p0 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(0));
  PolyUOp *p1 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(1));
  PolyUOp *p2 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(2));
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(n));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));
  PolyUOp *idx0 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p0, range, poly_arg_none());
  PolyUOp *idx1 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p1, range, poly_arg_none());
  PolyUOp *idx2 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p2, range, poly_arg_none());
  PolyUOp *ld0 = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx0, poly_arg_none());
  PolyUOp *ld1 = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx1, poly_arg_none());
  PolyUOp *alu = poly_uop2(ctx, op, POLY_FLOAT32, ld0, ld1, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx2, alu, poly_arg_none());
  PolyUOp *end_src[2] = { store, range };
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  return poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());
}

static PolyUOp *simplify(PolyCtx *ctx, PolyUOp *root) {
  return poly_graph_rewrite(ctx, root, poly_symbolic_simple());
}

/* ════════════════════════════════════════════════════════════════════════
 * SECTION 1: Missing transcendental decompositions — MUST FAIL
 * ════════════════════════════════════════════════════════════════════════ */

/*
 * LOG2 decomposition — tinygrad's xlog2:
 *   LOG2(d) → frexp-based polynomial. No LOG2 should remain.
 *   Ref: tinygrad/uop/decompositions.py xlog2()
 */
TEST(transcendental, decomp_log2_ir) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = make_unary_kernel(ctx, POLY_OP_LOG2, 4);
  PolyUOp *rewritten = poly_full_rewrite_to_sink(ctx, sink);
  int n_log2 = count_ops_in(ctx, rewritten, POLY_OP_LOG2);
  poly_ctx_destroy(ctx);
  ASSERT_INT_EQ(n_log2, 0);  /* LOG2 must be fully decomposed */
  PASS();
}

/*
 * SIN decomposition — tinygrad's xsin:
 *   SIN(d) → Payne-Hanek + polynomial. No SIN should remain.
 *   Ref: tinygrad/uop/decompositions.py xsin()
 */
TEST(transcendental, decomp_sin_ir) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = make_unary_kernel(ctx, POLY_OP_SIN, 4);
  PolyUOp *rewritten = poly_full_rewrite_to_sink(ctx, sink);
  int n_sin = count_ops_in(ctx, rewritten, POLY_OP_SIN);
  poly_ctx_destroy(ctx);
  ASSERT_INT_EQ(n_sin, 0);  /* SIN must be fully decomposed */
  PASS();
}

/* ════════════════════════════════════════════════════════════════════════
 * SECTION 2: Missing late decompositions — MUST FAIL
 * ════════════════════════════════════════════════════════════════════════ */

/*
 * MOD → AND: x % (2^n) → x & (2^n - 1) for int
 * Ref: tinygrad/uop/decompositions.py get_late_rewrite_patterns line 445
 */
TEST(decomp, mod_to_and) {
  PolyCtx *ctx = poly_ctx_new();
  PolyDType ptr_i32 = poly_dtype_ptr(POLY_INT32, -1, POLY_ADDR_GLOBAL);
  PolyUOp *p0 = poly_uop0(ctx, POLY_OP_PARAM, ptr_i32, poly_arg_int(0));
  PolyUOp *p1 = poly_uop0(ctx, POLY_OP_PARAM, ptr_i32, poly_arg_int(1));
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(8));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));
  PolyUOp *idx0 = poly_uop2(ctx, POLY_OP_INDEX, ptr_i32, p0, range, poly_arg_none());
  PolyUOp *idx1 = poly_uop2(ctx, POLY_OP_INDEX, ptr_i32, p1, range, poly_arg_none());
  PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, POLY_INT32, idx0, poly_arg_none());
  PolyUOp *four = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(4));
  PolyUOp *mod = poly_uop2(ctx, POLY_OP_MOD, POLY_INT32, load, four, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx1, mod, poly_arg_none());
  PolyUOp *end_src[2] = { store, range };
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());
  PolyUOp *rewritten = poly_full_rewrite_to_sink(ctx, sink);
  int n_mod = count_ops_in(ctx, rewritten, POLY_OP_MOD);
  int n_and = count_ops_in(ctx, rewritten, POLY_OP_AND);
  poly_ctx_destroy(ctx);
  ASSERT_INT_EQ(n_mod, 0);  /* MOD must be eliminated */
  ASSERT_TRUE(n_and > 0);   /* AND must appear */
  PASS();
}

/*
 * MULACC → MUL+ADD: for renderers without native FMA
 * Ref: tinygrad/uop/decompositions.py get_late_rewrite_patterns line 472
 */
TEST(decomp, mulacc_to_mul_add) {
  PolyCtx *ctx = poly_ctx_new();
  PolyDType ptr_f32 = poly_dtype_ptr(POLY_FLOAT32, -1, POLY_ADDR_GLOBAL);
  PolyUOp *p0 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(0));
  PolyUOp *p1 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(1));
  PolyUOp *p2 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(2));
  PolyUOp *p3 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(3));
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(4));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));
  PolyUOp *idx0 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p0, range, poly_arg_none());
  PolyUOp *idx1 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p1, range, poly_arg_none());
  PolyUOp *idx2 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p2, range, poly_arg_none());
  PolyUOp *idx3 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p3, range, poly_arg_none());
  PolyUOp *ld0 = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx0, poly_arg_none());
  PolyUOp *ld1 = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx1, poly_arg_none());
  PolyUOp *ld2 = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx2, poly_arg_none());
  PolyUOp *mulacc_srcs[3] = { ld0, ld1, ld2 };
  PolyUOp *mulacc = poly_uop(ctx, POLY_OP_MULACC, POLY_FLOAT32, mulacc_srcs, 3, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx3, mulacc, poly_arg_none());
  PolyUOp *end_src[2] = { store, range };
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());
  PolyUOp *rewritten = poly_full_rewrite_to_sink(ctx, sink);
  int n_mulacc = count_ops_in(ctx, rewritten, POLY_OP_MULACC);
  poly_ctx_destroy(ctx);
  ASSERT_INT_EQ(n_mulacc, 0);  /* MULACC must be eliminated */
  PASS();
}

/* x * (-1) → NEG(x). Ref: tinygrad get_late_rewrite_patterns */
TEST(decomp, mul_neg1_to_neg) {
  PolyCtx *ctx = poly_ctx_new();
  PolyDType ptr_i32 = poly_dtype_ptr(POLY_INT32, -1, POLY_ADDR_GLOBAL);
  PolyUOp *p0 = poly_uop0(ctx, POLY_OP_PARAM, ptr_i32, poly_arg_int(0));
  PolyUOp *p1 = poly_uop0(ctx, POLY_OP_PARAM, ptr_i32, poly_arg_int(1));
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(4));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));
  PolyUOp *idx0 = poly_uop2(ctx, POLY_OP_INDEX, ptr_i32, p0, range, poly_arg_none());
  PolyUOp *idx1 = poly_uop2(ctx, POLY_OP_INDEX, ptr_i32, p1, range, poly_arg_none());
  PolyUOp *ld0 = poly_uop1(ctx, POLY_OP_LOAD, POLY_INT32, idx0, poly_arg_none());
  PolyUOp *neg1 = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(-1));
  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, POLY_INT32, ld0, neg1, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx1, mul, poly_arg_none());
  PolyUOp *end_src[2] = { store, range };
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());

  PolyUOp *rewritten = poly_full_rewrite_to_sink(ctx, sink);
  int n_mul = count_ops_in(ctx, rewritten, POLY_OP_MUL);
  int n_neg = count_ops_in(ctx, rewritten, POLY_OP_NEG);
  poly_ctx_destroy(ctx);
  ASSERT_INT_EQ(n_mul, 0);
  ASSERT_TRUE(n_neg > 0);
  PASS();
}

/* x + NEG(y) → SUB(x, y). Ref: tinygrad get_late_rewrite_patterns */
TEST(decomp, add_neg_to_sub) {
  PolyCtx *ctx = poly_ctx_new();
  PolyDType ptr_i32 = poly_dtype_ptr(POLY_INT32, -1, POLY_ADDR_GLOBAL);
  PolyUOp *p0 = poly_uop0(ctx, POLY_OP_PARAM, ptr_i32, poly_arg_int(0));
  PolyUOp *p1 = poly_uop0(ctx, POLY_OP_PARAM, ptr_i32, poly_arg_int(1));
  PolyUOp *p2 = poly_uop0(ctx, POLY_OP_PARAM, ptr_i32, poly_arg_int(2));
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(4));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));
  PolyUOp *idx0 = poly_uop2(ctx, POLY_OP_INDEX, ptr_i32, p0, range, poly_arg_none());
  PolyUOp *idx1 = poly_uop2(ctx, POLY_OP_INDEX, ptr_i32, p1, range, poly_arg_none());
  PolyUOp *idx2 = poly_uop2(ctx, POLY_OP_INDEX, ptr_i32, p2, range, poly_arg_none());
  PolyUOp *ld0 = poly_uop1(ctx, POLY_OP_LOAD, POLY_INT32, idx0, poly_arg_none());
  PolyUOp *ld1 = poly_uop1(ctx, POLY_OP_LOAD, POLY_INT32, idx1, poly_arg_none());
  PolyUOp *neg = poly_uop1(ctx, POLY_OP_NEG, POLY_INT32, ld1, poly_arg_none());
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, ld0, neg, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx2, add, poly_arg_none());
  PolyUOp *end_src[2] = { store, range };
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());

  PolyUOp *rewritten = poly_full_rewrite_to_sink(ctx, sink);
  int n_add = count_ops_in(ctx, rewritten, POLY_OP_ADD);
  int n_sub = count_ops_in(ctx, rewritten, POLY_OP_SUB);
  poly_ctx_destroy(ctx);
  ASSERT_INT_EQ(n_add, 0);
  ASSERT_TRUE(n_sub > 0);
  PASS();
}

/* RECIPROCAL(x) → FDIV(1, x). Ref: tinygrad get_late_rewrite_patterns */
TEST(decomp, recip_to_fdiv) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = make_unary_kernel(ctx, POLY_OP_RECIPROCAL, 4);
  PolyUOp *rewritten = poly_full_rewrite_to_sink(ctx, sink);
  int n_recip = count_ops_in(ctx, rewritten, POLY_OP_RECIPROCAL);
  int n_fdiv = count_ops_in(ctx, rewritten, POLY_OP_FDIV);
  poly_ctx_destroy(ctx);
  ASSERT_INT_EQ(n_recip, 0);
  ASSERT_TRUE(n_fdiv > 0);
  PASS();
}

/* a * (1 / b) → a / b. Ref: tinygrad get_late_rewrite_patterns */
TEST(decomp, mul_one_over_b_to_fdiv) {
  PolyCtx *ctx = poly_ctx_new();
  PolyDType ptr_f32 = poly_dtype_ptr(POLY_FLOAT32, -1, POLY_ADDR_GLOBAL);
  PolyUOp *p0 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(0));
  PolyUOp *p1 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(1));
  PolyUOp *p2 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(2));
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(4));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));
  PolyUOp *idx0 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p0, range, poly_arg_none());
  PolyUOp *idx1 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p1, range, poly_arg_none());
  PolyUOp *idx2 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p2, range, poly_arg_none());
  PolyUOp *ld0 = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx0, poly_arg_none());
  PolyUOp *ld1 = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx1, poly_arg_none());
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(1.0));
  PolyUOp *one_over = poly_uop2(ctx, POLY_OP_FDIV, POLY_FLOAT32, one, ld1, poly_arg_none());
  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, ld0, one_over, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx2, mul, poly_arg_none());
  PolyUOp *end_src[2] = { store, range };
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());

  PolyUOp *rewritten = poly_full_rewrite_to_sink(ctx, sink);
  int n_mul = count_ops_in(ctx, rewritten, POLY_OP_MUL);
  int n_fdiv = count_ops_in(ctx, rewritten, POLY_OP_FDIV);
  poly_ctx_destroy(ctx);
  ASSERT_INT_EQ(n_mul, 0);
  ASSERT_TRUE(n_fdiv > 0);
  PASS();
}

/* ════════════════════════════════════════════════════════════════════════
 * SECTION 3: Missing symbolic simplification rules — MUST FAIL
 * ════════════════════════════════════════════════════════════════════════ */

/* (x % c) + (x // c) * c → x.  Ref: tinygrad symbolic_simple line 50 */
TEST(sym_future, divmod_cancel) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(100));
  PolyUOp *x = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));
  PolyUOp *c = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(8));
  PolyUOp *mod = poly_uop2(ctx, POLY_OP_MOD, POLY_INT32, x, c, poly_arg_none());
  PolyUOp *div = poly_uop2(ctx, POLY_OP_IDIV, POLY_INT32, x, c, poly_arg_none());
  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, POLY_INT32, div, c, poly_arg_none());
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, mod, mul, poly_arg_none());
  PolyUOp *r = simplify(ctx, add);
  int matched = (r == x);
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(matched);
  PASS();
}

/* bool * bool → AND.  Ref: tinygrad symbolic_simple line 82 */
TEST(sym_future, bool_mul_is_and) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(10));
  PolyUOp *r0 = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));
  PolyUOp *five = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(5));
  PolyUOp *three = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(3));
  PolyUOp *a = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, r0, five, poly_arg_none());
  PolyUOp *b = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, r0, three, poly_arg_none());
  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, POLY_BOOL, a, b, poly_arg_none());
  PolyUOp *r = simplify(ctx, mul);
  int is_and = (r->op == POLY_OP_AND);
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(is_and);
  PASS();
}

/* bool + bool → OR.  Ref: tinygrad symbolic_simple line 83 */
TEST(sym_future, bool_add_is_or) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(10));
  PolyUOp *r0 = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));
  PolyUOp *five = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(5));
  PolyUOp *three = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(3));
  PolyUOp *a = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, r0, five, poly_arg_none());
  PolyUOp *b = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, r0, three, poly_arg_none());
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_BOOL, a, b, poly_arg_none());
  PolyUOp *r = simplify(ctx, add);
  int is_or = (r->op == POLY_OP_OR);
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(is_or);
  PASS();
}

/* (x * x2) / x2 → x.  Ref: tinygrad symbolic_simple line 90 */
TEST(sym_future, mul_div_cancel) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_uop0(ctx, POLY_OP_DEFINE_VAR, POLY_FLOAT32, poly_arg_str("x"));
  PolyUOp *x2 = poly_uop0(ctx, POLY_OP_DEFINE_VAR, POLY_FLOAT32, poly_arg_str("x2"));
  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, x, x2, poly_arg_none());
  PolyUOp *div = poly_uop2(ctx, POLY_OP_FDIV, POLY_FLOAT32, mul, x2, poly_arg_none());
  PolyUOp *r = simplify(ctx, div);
  int matched = (r == x);
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(matched);
  PASS();
}

/* a.where(b.where(c, d), d) → (a & b).where(c, d).  Ref: tinygrad line 117 */
TEST(sym_future, nested_where) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_uop0(ctx, POLY_OP_DEFINE_VAR, POLY_BOOL, poly_arg_str("a"));
  PolyUOp *b = poly_uop0(ctx, POLY_OP_DEFINE_VAR, POLY_BOOL, poly_arg_str("b"));
  PolyUOp *c = poly_uop0(ctx, POLY_OP_DEFINE_VAR, POLY_FLOAT32, poly_arg_str("c"));
  PolyUOp *d = poly_uop0(ctx, POLY_OP_DEFINE_VAR, POLY_FLOAT32, poly_arg_str("d"));
  PolyUOp *inner = poly_uop3(ctx, POLY_OP_WHERE, POLY_FLOAT32, b, c, d, poly_arg_none());
  PolyUOp *outer = poly_uop3(ctx, POLY_OP_WHERE, POLY_FLOAT32, a, inner, d, poly_arg_none());
  PolyUOp *r = simplify(ctx, outer);
  int is_where = (r->op == POLY_OP_WHERE);
  int cond_is_and = is_where && (r->src[0]->op == POLY_OP_AND);
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(is_where);
  ASSERT_TRUE(cond_is_and);
  PASS();
}

/* x ^ 0 → x.  Ref: tinygrad symbolic_simple line 44 */
TEST(sym_future, xor_zero) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_uop0(ctx, POLY_OP_DEFINE_VAR, POLY_INT32, poly_arg_str("x"));
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *xor = poly_uop2(ctx, POLY_OP_XOR, POLY_INT32, x, zero, poly_arg_none());
  PolyUOp *r = simplify(ctx, xor);
  int matched = (r == x);
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(matched);
  PASS();
}

/* x != x → False (ints only).  Ref: tinygrad symbolic_simple line 72 */
TEST(sym_future, cmpne_self) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_uop0(ctx, POLY_OP_DEFINE_VAR, POLY_INT32, poly_arg_str("x"));
  PolyUOp *cmpne = poly_uop2(ctx, POLY_OP_CMPNE, POLY_BOOL, x, x, poly_arg_none());
  PolyUOp *r = simplify(ctx, cmpne);
  int is_const = (r->op == POLY_OP_CONST);
  int is_false = is_const && (r->arg.b == false);
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(is_const);
  ASSERT_TRUE(is_false);
  PASS();
}

/* bool & True → bool; bool & False → False.  Ref: tinygrad line 61 */
TEST(sym_future, bool_and_const) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_uop0(ctx, POLY_OP_DEFINE_VAR, POLY_BOOL, poly_arg_str("x"));
  PolyUOp *t = poly_uop0(ctx, POLY_OP_CONST, POLY_BOOL, poly_arg_bool(true));
  PolyUOp *f = poly_uop0(ctx, POLY_OP_CONST, POLY_BOOL, poly_arg_bool(false));
  PolyUOp *and_t = poly_uop2(ctx, POLY_OP_AND, POLY_BOOL, x, t, poly_arg_none());
  PolyUOp *and_f = poly_uop2(ctx, POLY_OP_AND, POLY_BOOL, x, f, poly_arg_none());
  PolyUOp *r_t = simplify(ctx, and_t);
  PolyUOp *r_f = simplify(ctx, and_f);
  int rt_is_x = (r_t == x);
  int rf_is_const = (r_f->op == POLY_OP_CONST);
  int rf_is_false = rf_is_const && (r_f->arg.b == false);
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(rt_is_x);
  ASSERT_TRUE(rf_is_const);
  ASSERT_TRUE(rf_is_false);
  PASS();
}

/* bool | True → True; bool | False → bool.  Ref: tinygrad line 62 */
TEST(sym_future, bool_or_const) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_uop0(ctx, POLY_OP_DEFINE_VAR, POLY_BOOL, poly_arg_str("x"));
  PolyUOp *t = poly_uop0(ctx, POLY_OP_CONST, POLY_BOOL, poly_arg_bool(true));
  PolyUOp *f = poly_uop0(ctx, POLY_OP_CONST, POLY_BOOL, poly_arg_bool(false));
  PolyUOp *or_t = poly_uop2(ctx, POLY_OP_OR, POLY_BOOL, x, t, poly_arg_none());
  PolyUOp *or_f = poly_uop2(ctx, POLY_OP_OR, POLY_BOOL, x, f, poly_arg_none());
  PolyUOp *r_t = simplify(ctx, or_t);
  PolyUOp *r_f = simplify(ctx, or_f);
  int rt_is_true = (r_t->op == POLY_OP_CONST && r_t->arg.b == true);
  int rf_is_x = (r_f == x);
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(rt_is_true);
  ASSERT_TRUE(rf_is_x);
  PASS();
}

/* VECTORIZE(CONST, CONST) → CONST(vec).  Ref: tinygrad sym line 390 */
TEST(sym_future, vectorize_const_fold) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *c1 = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(1.0));
  PolyUOp *c2 = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(1.0));
  PolyUOp *vec_srcs[2] = { c1, c2 };
  PolyUOp *vec = poly_uop(ctx, POLY_OP_VECTORIZE, POLY_FLOAT32, vec_srcs, 2, poly_arg_none());
  PolyUOp *r = simplify(ctx, vec);
  int folded = (r->op == POLY_OP_CONST || r->op == POLY_OP_VCONST);
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(folded);  /* must fold */
  PASS();
}

/* ════════════════════════════════════════════════════════════════════════
 * SECTION 4: Expander pass tests — MUST FAIL
 * ════════════════════════════════════════════════════════════════════════ */

/* UNROLL with empty arg → identity.  Ref: tinygrad expander.py line 102 */
TEST(expander, unroll_empty_arg_removed) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *val = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(42.0));
  PolyUOp *unroll = poly_uop1(ctx, POLY_OP_UNROLL, POLY_FLOAT32, val, poly_arg_none());
  /* After expander pass, UNROLL() with no axes should be removed */
  PolyUOp *r = poly_full_rewrite_to_sink(ctx, unroll);
  int not_unroll = (r->op != POLY_OP_UNROLL);
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(not_unroll);  /* UNROLL must be eliminated */
  PASS();
}

/* ════════════════════════════════════════════════════════════════════════
 * SECTION 5: Regression tests — MUST PASS (already ported)
 * ════════════════════════════════════════════════════════════════════════ */

/* EXP2 decomposition: EXP2 must be fully eliminated */
TEST(regression, exp2_decomp) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = make_unary_kernel(ctx, POLY_OP_EXP2, 4);
  PolyUOp *rewritten = poly_full_rewrite_to_sink(ctx, sink);
  ASSERT_INT_EQ(count_ops_in(ctx, rewritten, POLY_OP_EXP2), 0);
  poly_ctx_destroy(ctx);
  PASS();
}

/* EXP2 e2e correctness */
TEST(regression, exp2_e2e) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = make_unary_kernel(ctx, POLY_OP_EXP2, 4);
  int n;
  PolyUOp **lin = poly_linearize(ctx, sink, &n);
  char *src = poly_render_c(lin, n, "exp2_kernel");
  PolyProgram *prog = poly_compile_c(src, "exp2_kernel");
  ASSERT_NOT_NULL(prog);
  float in[4] = { 0.0f, 1.0f, 2.0f, -1.0f };
  float out[4] = { 0 };
  void *args[2] = { in, out };
  poly_program_call(prog, args, 2);
  ASSERT_FLOAT_EQ(out[0], 1.0f, 1e-4);
  ASSERT_FLOAT_EQ(out[1], 2.0f, 1e-4);
  ASSERT_FLOAT_EQ(out[2], 4.0f, 1e-4);
  ASSERT_FLOAT_EQ(out[3], 0.5f, 1e-4);
  poly_program_destroy(prog);
  free(src);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

/* MUL → SHL: x * 4 → SHL(x, 2) */
TEST(regression, mul_to_shl) {
  PolyCtx *ctx = poly_ctx_new();
  PolyDType ptr_i32 = poly_dtype_ptr(POLY_INT32, -1, POLY_ADDR_GLOBAL);
  PolyUOp *p0 = poly_uop0(ctx, POLY_OP_PARAM, ptr_i32, poly_arg_int(0));
  PolyUOp *p1 = poly_uop0(ctx, POLY_OP_PARAM, ptr_i32, poly_arg_int(1));
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(4));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));
  PolyUOp *idx0 = poly_uop2(ctx, POLY_OP_INDEX, ptr_i32, p0, range, poly_arg_none());
  PolyUOp *idx1 = poly_uop2(ctx, POLY_OP_INDEX, ptr_i32, p1, range, poly_arg_none());
  PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, POLY_INT32, idx0, poly_arg_none());
  PolyUOp *four = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(4));
  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, POLY_INT32, load, four, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx1, mul, poly_arg_none());
  PolyUOp *end_src[2] = { store, range };
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());
  PolyUOp *rewritten = poly_full_rewrite_to_sink(ctx, sink);
  ASSERT_INT_EQ(count_ops_in(ctx, rewritten, POLY_OP_MUL), 0);
  ASSERT_TRUE(count_ops_in(ctx, rewritten, POLY_OP_SHL) > 0);
  poly_ctx_destroy(ctx);
  PASS();
}

/* IDIV → SHR: x // 8 → SHR(x + correction, 3) */
TEST(regression, idiv_to_shr) {
  PolyCtx *ctx = poly_ctx_new();
  PolyDType ptr_i32 = poly_dtype_ptr(POLY_INT32, -1, POLY_ADDR_GLOBAL);
  PolyUOp *p0 = poly_uop0(ctx, POLY_OP_PARAM, ptr_i32, poly_arg_int(0));
  PolyUOp *p1 = poly_uop0(ctx, POLY_OP_PARAM, ptr_i32, poly_arg_int(1));
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(4));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));
  PolyUOp *idx0 = poly_uop2(ctx, POLY_OP_INDEX, ptr_i32, p0, range, poly_arg_none());
  PolyUOp *idx1 = poly_uop2(ctx, POLY_OP_INDEX, ptr_i32, p1, range, poly_arg_none());
  PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, POLY_INT32, idx0, poly_arg_none());
  PolyUOp *eight = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(8));
  PolyUOp *div = poly_uop2(ctx, POLY_OP_IDIV, POLY_INT32, load, eight, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx1, div, poly_arg_none());
  PolyUOp *end_src[2] = { store, range };
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());
  PolyUOp *rewritten = poly_full_rewrite_to_sink(ctx, sink);
  ASSERT_INT_EQ(count_ops_in(ctx, rewritten, POLY_OP_IDIV), 0);
  ASSERT_TRUE(count_ops_in(ctx, rewritten, POLY_OP_SHR) > 0);
  poly_ctx_destroy(ctx);
  PASS();
}

/* MAX → WHERE: MAX(a, b) → WHERE(CMPLT(a, b), b, a) */
TEST(regression, max_to_where) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = make_binary_kernel(ctx, POLY_OP_MAX, 4);
  PolyUOp *rewritten = poly_full_rewrite_to_sink(ctx, sink);
  ASSERT_INT_EQ(count_ops_in(ctx, rewritten, POLY_OP_MAX), 0);
  ASSERT_TRUE(count_ops_in(ctx, rewritten, POLY_OP_WHERE) > 0);
  poly_ctx_destroy(ctx);
  PASS();
}

/* (x % y) % y → x % y (idempotent rule, already ported) */
TEST(regression, double_mod) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(100));
  PolyUOp *x = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));
  PolyUOp *y = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(7));
  PolyUOp *mod1 = poly_uop2(ctx, POLY_OP_MOD, POLY_INT32, x, y, poly_arg_none());
  PolyUOp *mod2 = poly_uop2(ctx, POLY_OP_MOD, POLY_INT32, mod1, y, poly_arg_none());
  PolyUOp *r = simplify(ctx, mod2);
  ASSERT_PTR_EQ(r, mod1);
  poly_ctx_destroy(ctx);
  PASS();
}

/* LOG2 e2e: C renderer natively emits log2f */
TEST(regression, log2_e2e) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = make_unary_kernel(ctx, POLY_OP_LOG2, 4);
  int n;
  PolyUOp **lin = poly_linearize(ctx, sink, &n);
  char *src = poly_render_c(lin, n, "log2_kernel");
  PolyProgram *prog = poly_compile_c(src, "log2_kernel");
  ASSERT_NOT_NULL(prog);
  float in[4] = { 1.0f, 2.0f, 4.0f, 8.0f };
  float out[4] = { 0 };
  void *args[2] = { in, out };
  poly_program_call(prog, args, 2);
  ASSERT_FLOAT_EQ(out[0], 0.0f, 1e-4);
  ASSERT_FLOAT_EQ(out[1], 1.0f, 1e-4);
  ASSERT_FLOAT_EQ(out[2], 2.0f, 1e-4);
  ASSERT_FLOAT_EQ(out[3], 3.0f, 1e-4);
  poly_program_destroy(prog);
  free(src);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

/* SIN e2e: C renderer natively emits sinf */
TEST(regression, sin_e2e) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = make_unary_kernel(ctx, POLY_OP_SIN, 4);
  int n;
  PolyUOp **lin = poly_linearize(ctx, sink, &n);
  char *src = poly_render_c(lin, n, "sin_kernel");
  PolyProgram *prog = poly_compile_c(src, "sin_kernel");
  ASSERT_NOT_NULL(prog);
  float in[4] = { 0.0f, 1.5707963f, 3.1415927f, 6.2831853f };
  float out[4] = { 0 };
  void *args[2] = { in, out };
  poly_program_call(prog, args, 2);
  ASSERT_FLOAT_EQ(out[0], 0.0f, 1e-4);
  ASSERT_FLOAT_EQ(out[1], 1.0f, 1e-4);
  ASSERT_FLOAT_EQ(out[2], 0.0f, 1e-3);
  ASSERT_FLOAT_EQ(out[3], 0.0f, 1e-3);
  poly_program_destroy(prog);
  free(src);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

/* SIN decomp e2e (large angles): forces xsin rewrite path, including Payne-Hanek branch */
TEST(regression, sin_decomp_large_e2e) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = make_unary_kernel(ctx, POLY_OP_SIN, 4);
  PolyUOp *rewritten = poly_full_rewrite_to_sink(ctx, sink);
  ASSERT_INT_EQ(count_ops_in(ctx, rewritten, POLY_OP_SIN), 0);

  int n;
  PolyUOp **lin = poly_linearize(ctx, rewritten, &n);
  char *src = poly_render_c(lin, n, "sin_decomp_large_kernel");
  PolyProgram *prog = poly_compile_c(src, "sin_decomp_large_kernel");
  ASSERT_NOT_NULL(prog);

  float in[4] = { 31.0f, 100000.0f, -100000.0f, 1234567.0f };
  float out[4] = { 0 };
  void *args[2] = { in, out };
  poly_program_call(prog, args, 2);

  for (int i = 0; i < 4; i++) {
    ASSERT_FLOAT_EQ(out[i], sinf(in[i]), 2e-3);
  }

  poly_program_destroy(prog);
  free(src);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

/* SQRT e2e: C renderer uses native sqrtf */
TEST(regression, sqrt_e2e) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = make_unary_kernel(ctx, POLY_OP_SQRT, 4);
  int n;
  PolyUOp **lin = poly_linearize(ctx, sink, &n);
  char *src = poly_render_c(lin, n, "sqrt_kernel");
  PolyProgram *prog = poly_compile_c(src, "sqrt_kernel");
  ASSERT_NOT_NULL(prog);
  float in[4] = { 1.0f, 4.0f, 9.0f, 16.0f };
  float out[4] = { 0 };
  void *args[2] = { in, out };
  poly_program_call(prog, args, 2);
  ASSERT_FLOAT_EQ(out[0], 1.0f, 1e-6);
  ASSERT_FLOAT_EQ(out[1], 2.0f, 1e-6);
  ASSERT_FLOAT_EQ(out[2], 3.0f, 1e-6);
  ASSERT_FLOAT_EQ(out[3], 4.0f, 1e-6);
  poly_program_destroy(prog);
  free(src);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

/* RECIPROCAL e2e: C renderer uses (1.0f/x) */
TEST(regression, reciprocal_e2e) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = make_unary_kernel(ctx, POLY_OP_RECIPROCAL, 4);
  int n;
  PolyUOp **lin = poly_linearize(ctx, sink, &n);
  char *src = poly_render_c(lin, n, "recip_kernel");
  PolyProgram *prog = poly_compile_c(src, "recip_kernel");
  ASSERT_NOT_NULL(prog);
  float in[4] = { 1.0f, 2.0f, 4.0f, 0.5f };
  float out[4] = { 0 };
  void *args[2] = { in, out };
  poly_program_call(prog, args, 2);
  ASSERT_FLOAT_EQ(out[0], 1.0f, 1e-6);
  ASSERT_FLOAT_EQ(out[1], 0.5f, 1e-6);
  ASSERT_FLOAT_EQ(out[2], 0.25f, 1e-6);
  ASSERT_FLOAT_EQ(out[3], 2.0f, 1e-6);
  poly_program_destroy(prog);
  free(src);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

/* FDIV e2e: C renderer uses native / */
TEST(regression, fdiv_e2e) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = make_binary_kernel(ctx, POLY_OP_FDIV, 4);
  int n;
  PolyUOp **lin = poly_linearize(ctx, sink, &n);
  char *src = poly_render_c(lin, n, "fdiv_kernel");
  PolyProgram *prog = poly_compile_c(src, "fdiv_kernel");
  ASSERT_NOT_NULL(prog);
  float a[4] = { 10.0f, 7.0f, 1.0f, 0.0f };
  float b[4] = { 2.0f, 3.0f, 3.0f, 1.0f };
  float out[4] = { 0 };
  void *args[3] = { a, b, out };
  poly_program_call(prog, args, 3);
  ASSERT_FLOAT_EQ(out[0], 5.0f, 1e-6);
  ASSERT_FLOAT_EQ(out[1], 7.0f/3.0f, 1e-5);
  ASSERT_FLOAT_EQ(out[2], 1.0f/3.0f, 1e-5);
  ASSERT_FLOAT_EQ(out[3], 0.0f, 1e-6);
  poly_program_destroy(prog);
  free(src);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}
