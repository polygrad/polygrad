/*
 * test_pat.c — Tests for pattern matcher: PolyPat, PatternMatcher, graph_rewrite
 */

#include "test_harness.h"
#include "../src/pat.h"

/* ── Pattern matching tests ───────────────────────────────────────────── */

TEST(pat, match_op_literal) {
  /* Pattern: match CONST op */
  PolyPat *p = poly_pat_op(POLY_OP_CONST, NULL, 0, NULL);
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *c = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(42));
  PolyUOp *a = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(1));
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, c, a, poly_arg_none());

  PolyBindings b = { .n = 0 };
  ASSERT_TRUE(poly_pat_match(p, c, &b));
  b.n = 0;
  ASSERT_FALSE(poly_pat_match(p, add, &b));

  poly_pat_free(p);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pat, match_wildcard_binding) {
  /* Pattern: match any UOp, bind to "x" */
  PolyPat *p = poly_pat_any("x");
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *c = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(5));

  PolyBindings b = { .n = 0 };
  ASSERT_TRUE(poly_pat_match(p, c, &b));
  ASSERT_INT_EQ(b.n, 1);
  ASSERT_PTR_EQ(poly_bind(&b, "x"), c);

  poly_pat_free(p);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pat, match_src_children) {
  /* Pattern: ADD(CONST, CONST) */
  PolyPat *p = poly_pat_op2(POLY_OP_ADD,
    poly_pat_op(POLY_OP_CONST, NULL, 0, NULL),
    poly_pat_op(POLY_OP_CONST, NULL, 0, NULL), NULL);
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(2));
  PolyUOp *b = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(3));
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, a, b, poly_arg_none());

  PolyBindings binds = { .n = 0 };
  ASSERT_TRUE(poly_pat_match(p, add, &binds));

  /* Should fail for ADD(CONST, ADD) */
  PolyUOp *add2 = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, a, add, poly_arg_none());
  binds.n = 0;
  ASSERT_FALSE(poly_pat_match(p, add2, &binds));

  poly_pat_free(p);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pat, match_named_identity) {
  /* Pattern: IDIV(x, x) — same UOp in both positions */
  PolyPat *p = poly_pat_op2(POLY_OP_IDIV,
    poly_pat_any("x"), poly_pat_any("x"), NULL);
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(7));
  PolyUOp *b = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(3));

  /* Same pointer: should match */
  PolyUOp *div1 = poly_uop2(ctx, POLY_OP_IDIV, POLY_INT32, a, a, poly_arg_none());
  PolyBindings binds = { .n = 0 };
  ASSERT_TRUE(poly_pat_match(p, div1, &binds));
  ASSERT_PTR_EQ(poly_bind(&binds, "x"), a);

  /* Different pointers: should fail */
  PolyUOp *div2 = poly_uop2(ctx, POLY_OP_IDIV, POLY_INT32, a, b, poly_arg_none());
  binds.n = 0;
  ASSERT_FALSE(poly_pat_match(p, div2, &binds));

  poly_pat_free(p);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pat, match_commutative) {
  /* Pattern: ADD(var("x"), CONST(0)) with commutative */
  PolyPat *p = poly_pat_op2c(POLY_OP_ADD,
    poly_pat_any("x"), poly_pat_const_val(poly_arg_int(0)), NULL);
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(5));
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));

  /* x + 0: should match */
  PolyUOp *add1 = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, x, zero, poly_arg_none());
  PolyBindings binds = { .n = 0 };
  ASSERT_TRUE(poly_pat_match(p, add1, &binds));
  ASSERT_PTR_EQ(poly_bind(&binds, "x"), x);

  /* 0 + x: should also match (commutative) */
  PolyUOp *add2 = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, zero, x, poly_arg_none());
  binds.n = 0;
  ASSERT_TRUE(poly_pat_match(p, add2, &binds));
  ASSERT_PTR_EQ(poly_bind(&binds, "x"), x);

  poly_pat_free(p);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pat, match_cvar) {
  /* cvar matches CONST and VCONST */
  PolyPat *p = poly_pat_cvar("c");
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *c = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(42));
  PolyUOp *v = poly_uop0(ctx, POLY_OP_VCONST, POLY_INT32, poly_arg_int(7));
  PolyUOp *x = poly_uop0(ctx, POLY_OP_ADD, POLY_INT32, poly_arg_none());

  PolyBindings b = { .n = 0 };
  ASSERT_TRUE(poly_pat_match(p, c, &b));
  b.n = 0;
  ASSERT_TRUE(poly_pat_match(p, v, &b));
  b.n = 0;
  ASSERT_FALSE(poly_pat_match(p, x, &b));

  poly_pat_free(p);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pat, match_const_val) {
  /* Match CONST with specific arg value */
  PolyPat *p = poly_pat_const_val(poly_arg_int(0));
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(1));

  PolyBindings b = { .n = 0 };
  ASSERT_TRUE(poly_pat_match(p, zero, &b));
  b.n = 0;
  ASSERT_FALSE(poly_pat_match(p, one, &b));

  poly_pat_free(p);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pat, match_opset) {
  /* Match any op from GroupOp.Unary */
  PolyPat *p = poly_pat_ops1(POLY_GROUP_UNARY, poly_pat_cvar(NULL), "a");
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *c = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(4.0));
  PolyUOp *neg = poly_uop1(ctx, POLY_OP_NEG, POLY_FLOAT32, c, poly_arg_none());
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, c, c, poly_arg_none());

  PolyBindings b = { .n = 0 };
  ASSERT_TRUE(poly_pat_match(p, neg, &b));
  b.n = 0;
  ASSERT_FALSE(poly_pat_match(p, add, &b));

  poly_pat_free(p);
  poly_ctx_destroy(ctx);
  PASS();
}

/* ── PatternMatcher tests ─────────────────────────────────────────────── */

/* Simple rewrite: return x from x+0 */
static PolyUOp *test_rewrite_identity(PolyCtx *ctx, PolyUOp *root,
                                      const PolyBindings *b) {
  (void)ctx; (void)root;
  return poly_bind(b, "x");
}

TEST(pat, pm_rewrite_basic) {
  /* Single rule: ADD(x, CONST(0)) -> x */
  PolyPat *p = poly_pat_op2c(POLY_OP_ADD,
    poly_pat_any("x"), poly_pat_const_val(poly_arg_int(0)), NULL);
  PolyRule rules[] = {{ p, test_rewrite_identity }};
  PolyPatternMatcher *pm = poly_pm_new(rules, 1);

  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(5));
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, x, zero, poly_arg_none());

  PolyUOp *result = poly_pm_rewrite(pm, ctx, add);
  ASSERT_NOT_NULL(result);
  ASSERT_PTR_EQ(result, x);

  /* MUL should not match */
  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, POLY_INT32, x, zero, poly_arg_none());
  ASSERT_TRUE(poly_pm_rewrite(pm, ctx, mul) == NULL);

  poly_pm_destroy(pm);
  poly_pat_free(p);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pat, pm_early_reject) {
  /* The early_reject optimization should skip patterns quickly */
  PolyPat *p = poly_pat_op2(POLY_OP_ADD,
    poly_pat_op(POLY_OP_MUL, NULL, 0, NULL),
    poly_pat_op(POLY_OP_CONST, NULL, 0, NULL), NULL);
  PolyRule rules[] = {{ p, test_rewrite_identity }};
  PolyPatternMatcher *pm = poly_pm_new(rules, 1);

  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *c1 = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(2));
  PolyUOp *c2 = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(3));
  /* ADD(CONST, CONST) — no MUL in sources, should be rejected early */
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, c1, c2, poly_arg_none());
  ASSERT_TRUE(poly_pm_rewrite(pm, ctx, add) == NULL);

  poly_pm_destroy(pm);
  poly_pat_free(p);
  poly_ctx_destroy(ctx);
  PASS();
}

/* ── graph_rewrite tests ──────────────────────────────────────────────── */

TEST(pat, graph_rewrite_noop) {
  /* No rules match — graph should be unchanged */
  PolyPat *p = poly_pat_op2c(POLY_OP_ADD, poly_pat_any("x"),
    poly_pat_const_val(poly_arg_int(999)), NULL);
  PolyRule rules[] = {{ p, test_rewrite_identity }};
  PolyPatternMatcher *pm = poly_pm_new(rules, 1);

  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(2));
  PolyUOp *b = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(3));
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, a, b, poly_arg_none());

  PolyUOp *result = poly_graph_rewrite(ctx, add, pm);
  ASSERT_PTR_EQ(result, add);

  poly_pm_destroy(pm);
  poly_pat_free(p);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pat, graph_rewrite_simple) {
  /* Rewrite x+0 -> x in a graph */
  PolyPat *p = poly_pat_op2c(POLY_OP_ADD, poly_pat_any("x"),
    poly_pat_const_val(poly_arg_int(0)), NULL);
  PolyRule rules[] = {{ p, test_rewrite_identity }};
  PolyPatternMatcher *pm = poly_pm_new(rules, 1);

  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *five = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(5));
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, five, zero, poly_arg_none());

  PolyUOp *result = poly_graph_rewrite(ctx, add, pm);
  ASSERT_PTR_EQ(result, five);

  poly_pm_destroy(pm);
  poly_pat_free(p);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pat, graph_rewrite_nested) {
  /* Rewrite (a + 0) + 0 -> a (double application) */
  PolyPat *p = poly_pat_op2c(POLY_OP_ADD, poly_pat_any("x"),
    poly_pat_const_val(poly_arg_int(0)), NULL);
  PolyRule rules[] = {{ p, test_rewrite_identity }};
  PolyPatternMatcher *pm = poly_pm_new(rules, 1);

  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(7));
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *inner = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, a, zero, poly_arg_none());
  PolyUOp *outer = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, inner, zero, poly_arg_none());

  PolyUOp *result = poly_graph_rewrite(ctx, outer, pm);
  ASSERT_PTR_EQ(result, a);

  poly_pm_destroy(pm);
  poly_pat_free(p);
  poly_ctx_destroy(ctx);
  PASS();
}

/* ── pm_concat test ───────────────────────────────────────────────────── */

static PolyUOp *test_rewrite_div_self(PolyCtx *ctx, PolyUOp *root,
                                      const PolyBindings *b) {
  return poly_const_like_int(ctx, poly_bind(b, "x"), 1);
}

TEST(pat, pm_concat) {
  PolyPat *p1 = poly_pat_op2c(POLY_OP_ADD, poly_pat_any("x"),
    poly_pat_const_val(poly_arg_int(0)), NULL);
  PolyPat *p2 = poly_pat_op2(POLY_OP_IDIV, poly_pat_any("x"),
    poly_pat_any("x"), NULL);
  PolyRule r1[] = {{ p1, test_rewrite_identity }};
  PolyRule r2[] = {{ p2, test_rewrite_div_self }};

  PolyPatternMatcher *pm1 = poly_pm_new(r1, 1);
  PolyPatternMatcher *pm2 = poly_pm_new(r2, 1);
  PolyPatternMatcher *combined = poly_pm_concat(pm1, pm2);

  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *five = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(5));
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, five, zero, poly_arg_none());
  PolyUOp *div = poly_uop2(ctx, POLY_OP_IDIV, POLY_INT32, five, five, poly_arg_none());

  ASSERT_PTR_EQ(poly_pm_rewrite(combined, ctx, add), five);
  PolyUOp *div_result = poly_pm_rewrite(combined, ctx, div);
  ASSERT_NOT_NULL(div_result);
  ASSERT_INT_EQ(div_result->arg.i, 1);

  poly_pm_destroy(pm1);
  poly_pm_destroy(pm2);
  poly_pm_destroy(combined);
  poly_pat_free(p1);
  poly_pat_free(p2);
  poly_ctx_destroy(ctx);
  PASS();
}
