/*
 * test_pat.c — Tests for pattern matcher: PolyPat, PatternMatcher, graph_rewrite
 */

#include "test_harness.h"
#include "../src/pat.h"

typedef struct {
  const char *key;
  char *value;
  bool had;
} PatEnvSave;

static PatEnvSave pat_save_env(const char *key) {
  const char *v = getenv(key);
  char *copy = NULL;
  if (v) {
    size_t n = strlen(v) + 1;
    copy = malloc(n);
    if (copy) memcpy(copy, v, n);
  }
  return (PatEnvSave){.key = key, .value = copy, .had = (v != NULL)};
}

static void pat_restore_env(PatEnvSave *s) {
  if (!s) return;
  if (s->had) {
    setenv(s->key, s->value ? s->value : "", 1);
  } else {
    unsetenv(s->key);
  }
  free(s->value);
  s->value = NULL;
}

/* Pattern matching tests */

TEST(pat, match_op_literal) {
  /* Pattern: match CONST op */
  PolyPat *p = poly_pat_op(POLY_OP_CONST, NULL, 0, NULL);
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *c = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(42));
  PolyUOp *a = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(1));
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, c, a, poly_arg_none());

  PolyBindings b = {.n = 0};
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

  PolyBindings b = {.n = 0};
  ASSERT_TRUE(poly_pat_match(p, c, &b));
  ASSERT_INT_EQ(b.n, 1);
  ASSERT_PTR_EQ(poly_bind(&b, "x"), c);

  poly_pat_free(p);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pat, match_more_than_inline_bindings) {
  enum { N = 20 };
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *srcs[N];
  PolyPat *pats[N];
  char names[N][8];

  for (int i = 0; i < N; i++) {
    snprintf(names[i], sizeof(names[i]), "x%d", i);
    srcs[i] = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(i));
    pats[i] = poly_pat_any(names[i]);
  }

  PolyPat *p = poly_pat_op(POLY_OP_SINK, pats, N, NULL);
  PolyUOp *sink = poly_uop(ctx, POLY_OP_SINK, POLY_VOID, srcs, N, poly_arg_none());

  PolyBindings b = {.n = 0};
  ASSERT_TRUE(poly_pat_match(p, sink, &b));
  ASSERT_INT_EQ(b.n, N);
  ASSERT_PTR_EQ(poly_bind(&b, "x0"), srcs[0]);
  ASSERT_PTR_EQ(poly_bind(&b, "x19"), srcs[19]);

  poly_bindings_free(&b);
  poly_pat_free(p);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pat, match_src_children) {
  /* Pattern: ADD(CONST, CONST) */
  PolyPat *p = poly_pat_op2(
      POLY_OP_ADD, poly_pat_op(POLY_OP_CONST, NULL, 0, NULL),
      poly_pat_op(POLY_OP_CONST, NULL, 0, NULL), NULL
  );
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(2));
  PolyUOp *b = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(3));
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, a, b, poly_arg_none());

  PolyBindings binds = {.n = 0};
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
  PolyPat *p = poly_pat_op2(POLY_OP_IDIV, poly_pat_any("x"), poly_pat_any("x"), NULL);
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(7));
  PolyUOp *b = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(3));

  /* Same pointer: should match */
  PolyUOp *div1 = poly_uop2(ctx, POLY_OP_IDIV, POLY_INT32, a, a, poly_arg_none());
  PolyBindings binds = {.n = 0};
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
  PolyPat *p =
      poly_pat_op2c(POLY_OP_ADD, poly_pat_any("x"), poly_pat_const_val(poly_arg_int(0)), NULL);
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(5));
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));

  /* x + 0: should match */
  PolyUOp *add1 = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, x, zero, poly_arg_none());
  PolyBindings binds = {.n = 0};
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

  PolyBindings b = {.n = 0};
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

  PolyBindings b = {.n = 0};
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

  PolyBindings b = {.n = 0};
  ASSERT_TRUE(poly_pat_match(p, neg, &b));
  b.n = 0;
  ASSERT_FALSE(poly_pat_match(p, add, &b));

  poly_pat_free(p);
  poly_ctx_destroy(ctx);
  PASS();
}

/* PatternMatcher tests */

/* Simple rewrite: return x from x+0 */
static PolyUOp *test_rewrite_identity(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)ctx;
  (void)root;
  return poly_bind(b, "x");
}

TEST(pat, pm_rewrite_basic) {
  /* Single rule: ADD(x, CONST(0)) -> x */
  PolyPat *p =
      poly_pat_op2c(POLY_OP_ADD, poly_pat_any("x"), poly_pat_const_val(poly_arg_int(0)), NULL);
  PolyRule rules[] = {{p, test_rewrite_identity}};
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
  PolyPat *p = poly_pat_op2(
      POLY_OP_ADD, poly_pat_op(POLY_OP_MUL, NULL, 0, NULL),
      poly_pat_op(POLY_OP_CONST, NULL, 0, NULL), NULL
  );
  PolyRule rules[] = {{p, test_rewrite_identity}};
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

TEST(pat, pm_rule_stats_disabled_by_default) {
  PatEnvSave track = pat_save_env("POLY_TRACK_MATCH_STATS");
  PatEnvSave track_tg = pat_save_env("TRACK_MATCH_STATS");
  PatEnvSave print = pat_save_env("POLY_PRINT_MATCH_STATS");
  PatEnvSave print_tg = pat_save_env("PRINT_MATCH_STATS");
  unsetenv("POLY_TRACK_MATCH_STATS");
  unsetenv("TRACK_MATCH_STATS");
  unsetenv("POLY_PRINT_MATCH_STATS");
  unsetenv("PRINT_MATCH_STATS");

  PolyPat *p =
      poly_pat_op2c(POLY_OP_ADD, poly_pat_any("x"), poly_pat_const_val(poly_arg_int(0)), NULL);
  PolyNamedRule rules[] = {POLY_RULE(p, test_rewrite_identity)};
  PolyPatternMatcher *pm = poly_pm_new_named(rules, 1);

  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(5));
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, x, zero, poly_arg_none());
  ASSERT_PTR_EQ(poly_pm_rewrite(pm, ctx, add), x);

  PolyRuleStats stats = {0};
  ASSERT_INT_EQ(poly_pm_get_rule_stats(pm, 0, &stats), 0);
  ASSERT_STR_EQ(stats.name, "test_rewrite_identity");
  ASSERT_INT_EQ((int)stats.candidates, 0);
  ASSERT_INT_EQ((int)stats.attempts, 0);
  ASSERT_INT_EQ((int)stats.pattern_matches, 0);
  ASSERT_INT_EQ((int)stats.rewrites, 0);

  poly_pm_destroy(pm);
  poly_pat_free(p);
  poly_ctx_destroy(ctx);
  pat_restore_env(&print_tg);
  pat_restore_env(&print);
  pat_restore_env(&track_tg);
  pat_restore_env(&track);
  PASS();
}

TEST(pat, pm_rule_stats_track_named_rewrites) {
  PatEnvSave track = pat_save_env("POLY_TRACK_MATCH_STATS");
  setenv("POLY_TRACK_MATCH_STATS", "1", 1);

  PolyPat *p =
      poly_pat_op2c(POLY_OP_ADD, poly_pat_any("x"), poly_pat_const_val(poly_arg_int(0)), NULL);
  PolyNamedRule rules[] = {POLY_RULE(p, test_rewrite_identity)};
  PolyPatternMatcher *pm = poly_pm_new_named(rules, 1);
  ASSERT_INT_EQ(poly_pm_rule_count(pm), 1);

  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(5));
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, x, zero, poly_arg_none());
  ASSERT_PTR_EQ(poly_pm_rewrite(pm, ctx, add), x);

  PolyRuleStats stats = {0};
  ASSERT_INT_EQ(poly_pm_get_rule_stats(pm, 0, &stats), 0);
  ASSERT_STR_EQ(stats.name, "test_rewrite_identity");
  ASSERT_INT_EQ((int)stats.candidates, 1);
  ASSERT_INT_EQ((int)stats.attempts, 1);
  ASSERT_INT_EQ((int)stats.pattern_matches, 1);
  ASSERT_INT_EQ((int)stats.rewrites, 1);
  ASSERT_TRUE(stats.total_ms >= 0.0);
  ASSERT_TRUE(stats.rewrite_ms >= 0.0);

  poly_pm_reset_rule_stats(pm);
  ASSERT_INT_EQ(poly_pm_get_rule_stats(pm, 0, &stats), 0);
  ASSERT_STR_EQ(stats.name, "test_rewrite_identity");
  ASSERT_INT_EQ((int)stats.candidates, 0);
  ASSERT_INT_EQ((int)stats.rewrites, 0);

  ASSERT_INT_EQ(poly_pm_get_rule_stats(pm, 1, &stats), -1);

  poly_pm_destroy(pm);
  poly_pat_free(p);
  poly_ctx_destroy(ctx);
  pat_restore_env(&track);
  PASS();
}

TEST(pat, pm_rewrite_trace_json_env) {
  const char *path = "temp/pat_rewrite_trace_test.jsonl";
  PatEnvSave trace = pat_save_env("POLY_REWRITE_TRACE_JSON");
  remove(path);
  setenv("POLY_REWRITE_TRACE_JSON", path, 1);

  PolyPat *p =
      poly_pat_op2c(POLY_OP_ADD, poly_pat_any("x"), poly_pat_const_val(poly_arg_int(0)), NULL);
  PolyNamedRule rules[] = {POLY_RULE(p, test_rewrite_identity)};
  PolyPatternMatcher *pm = poly_pm_new_named(rules, 1);

  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(5));
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, x, zero, poly_arg_none());
  ASSERT_PTR_EQ(poly_pm_rewrite(pm, ctx, add), x);

  poly_pm_destroy(pm);
  pm = NULL;

  FILE *fp = fopen(path, "rb");
  ASSERT_NOT_NULL(fp);
  char buf[1024];
  size_t n = fread(buf, 1, sizeof(buf) - 1, fp);
  fclose(fp);
  buf[n] = '\0';
  ASSERT_TRUE(strstr(buf, "\"event\":\"rewrite\"") != NULL);
  ASSERT_TRUE(strstr(buf, "\"rule\":\"test_rewrite_identity\"") != NULL);
  ASSERT_TRUE(strstr(buf, "\"before_op\":\"ADD\"") != NULL);
  ASSERT_TRUE(strstr(buf, "\"after_op\":\"CONST\"") != NULL);

  remove(path);
  poly_pat_free(p);
  poly_ctx_destroy(ctx);
  pat_restore_env(&trace);
  PASS();
}

/* graph_rewrite tests */

TEST(pat, graph_rewrite_noop) {
  /* No rules match — graph should be unchanged */
  PolyPat *p =
      poly_pat_op2c(POLY_OP_ADD, poly_pat_any("x"), poly_pat_const_val(poly_arg_int(999)), NULL);
  PolyRule rules[] = {{p, test_rewrite_identity}};
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
  PolyPat *p =
      poly_pat_op2c(POLY_OP_ADD, poly_pat_any("x"), poly_pat_const_val(poly_arg_int(0)), NULL);
  PolyRule rules[] = {{p, test_rewrite_identity}};
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
  PolyPat *p =
      poly_pat_op2c(POLY_OP_ADD, poly_pat_any("x"), poly_pat_const_val(poly_arg_int(0)), NULL);
  PolyRule rules[] = {{p, test_rewrite_identity}};
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

/* pm_concat test */

static PolyUOp *test_rewrite_div_self(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  return poly_const_like_int(ctx, poly_bind(b, "x"), 1);
}

TEST(pat, pm_concat) {
  PolyPat *p1 =
      poly_pat_op2c(POLY_OP_ADD, poly_pat_any("x"), poly_pat_const_val(poly_arg_int(0)), NULL);
  PolyPat *p2 = poly_pat_op2(POLY_OP_IDIV, poly_pat_any("x"), poly_pat_any("x"), NULL);
  PolyNamedRule r1[] = {POLY_RULE(p1, test_rewrite_identity)};
  PolyNamedRule r2[] = {POLY_RULE(p2, test_rewrite_div_self)};

  PolyPatternMatcher *pm1 = poly_pm_new_named(r1, 1);
  PolyPatternMatcher *pm2 = poly_pm_new_named(r2, 1);
  PolyPatternMatcher *combined = poly_pm_concat(pm1, pm2);
  PolyRuleStats stats = {0};
  ASSERT_INT_EQ(poly_pm_get_rule_stats(combined, 0, &stats), 0);
  ASSERT_STR_EQ(stats.name, "test_rewrite_identity");
  ASSERT_INT_EQ(poly_pm_get_rule_stats(combined, 1, &stats), 0);
  ASSERT_STR_EQ(stats.name, "test_rewrite_div_self");

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

/* CALL gating in graph_rewrite */

static PolyUOp *rewrite_neg_to_zero(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)root;
  (void)b;
  return poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(0.0));
}

TEST(pat, graph_rewrite_skips_call_body) {
  PolyCtx *ctx = poly_ctx_new();

  /* Build: CALL(callee, ADD(NEG(c1), c2))
   * callee = NEG(c3)
   *
   * A rewrite rule NEG->0 should fire on the outer NEG(c1)
   * but NOT on the callee's NEG(c3) when enter_calls=false. */
  PolyUOp *c1 = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(1.0));
  PolyUOp *c2 = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(2.0));
  PolyUOp *c3 = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(3.0));
  PolyUOp *callee_neg = poly_uop1(ctx, POLY_OP_NEG, POLY_FLOAT32, c3, poly_arg_none());
  PolyUOp *outer_neg = poly_uop1(ctx, POLY_OP_NEG, POLY_FLOAT32, c1, poly_arg_none());
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, outer_neg, c2, poly_arg_none());
  PolyUOp *call = poly_uop2(ctx, POLY_OP_CALL, POLY_FLOAT32, callee_neg, add, poly_arg_none());

  /* Pattern: NEG(x) -> CONST(0) */
  PolyPat *neg_pat = poly_pat_op1(POLY_OP_NEG, poly_pat_any("x"), NULL);
  PolyRule rules[] = {{neg_pat, rewrite_neg_to_zero}};
  PolyPatternMatcher *pm = poly_pm_new(rules, 1);

  /* enter_calls=true: both NEGs rewritten */
  PolyUOp *result_enter = poly_graph_rewrite_ctx_ex2(ctx, call, pm, NULL, false, true);
  ASSERT_NOT_NULL(result_enter);
  ASSERT_TRUE(result_enter->op == POLY_OP_CALL);
  ASSERT_TRUE(result_enter->src[0]->op == POLY_OP_CONST); /* callee NEG was rewritten */

  /* enter_calls=false: only outer NEG rewritten, callee NEG preserved */
  PolyUOp *result_skip = poly_graph_rewrite_ctx_ex2(ctx, call, pm, NULL, false, false);
  ASSERT_NOT_NULL(result_skip);
  ASSERT_TRUE(result_skip->op == POLY_OP_CALL);
  ASSERT_TRUE(result_skip->src[0]->op == POLY_OP_NEG); /* callee NEG preserved */
  /* But the outer NEG in ADD should have been rewritten */
  PolyUOp *add_result = result_skip->src[1]; /* the ADD arg */
  ASSERT_TRUE(add_result->op == POLY_OP_ADD);
  ASSERT_TRUE(add_result->src[0]->op == POLY_OP_CONST); /* outer NEG was rewritten to 0 */

  poly_pm_destroy(pm);
  poly_pat_free(neg_pat);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pat, graph_rewrite_call_args_do_not_inherit_callee_body_gating) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *c1 = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(1.0));
  PolyUOp *c2 = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(2.0));
  PolyUOp *shared_neg = poly_uop1(ctx, POLY_OP_NEG, POLY_FLOAT32, c1, poly_arg_none());
  PolyUOp *callee = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, shared_neg, c2, poly_arg_none());
  PolyUOp *call = poly_uop2(ctx, POLY_OP_CALL, POLY_FLOAT32, callee, shared_neg, poly_arg_none());

  PolyPat *neg_pat = poly_pat_op1(POLY_OP_NEG, poly_pat_any("x"), NULL);
  PolyRule rules[] = {{neg_pat, rewrite_neg_to_zero}};
  PolyPatternMatcher *pm = poly_pm_new(rules, 1);

  PolyUOp *result = poly_graph_rewrite_ctx_ex2(ctx, call, pm, NULL, false, false);
  ASSERT_NOT_NULL(result);
  ASSERT_INT_EQ(result->op, POLY_OP_CALL);
  ASSERT_PTR_EQ(result->src[0], callee);
  ASSERT_INT_EQ(result->src[1]->op, POLY_OP_CONST);

  poly_pm_destroy(pm);
  poly_pat_free(neg_pat);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pat, walk_rewrite_call_args_do_not_inherit_callee_body_gating) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *c1 = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(1.0));
  PolyUOp *c2 = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(2.0));
  PolyUOp *shared_neg = poly_uop1(ctx, POLY_OP_NEG, POLY_FLOAT32, c1, poly_arg_none());
  PolyUOp *callee = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, shared_neg, c2, poly_arg_none());
  PolyUOp *call = poly_uop2(ctx, POLY_OP_CALL, POLY_FLOAT32, callee, shared_neg, poly_arg_none());

  PolyPat *neg_pat = poly_pat_op1(POLY_OP_NEG, poly_pat_any("x"), NULL);
  PolyRule rules[] = {{neg_pat, rewrite_neg_to_zero}};
  PolyPatternMatcher *pm = poly_pm_new(rules, 1);

  PolyUOp *result = poly_graph_walk_rewrite(ctx, call, pm, NULL, NULL, false);
  ASSERT_NOT_NULL(result);
  ASSERT_INT_EQ(result->op, POLY_OP_CALL);
  ASSERT_PTR_EQ(result->src[0], callee);
  ASSERT_INT_EQ(result->src[1]->op, POLY_OP_CONST);

  poly_pm_destroy(pm);
  poly_pat_free(neg_pat);
  poly_ctx_destroy(ctx);
  PASS();
}

static PolyUOp *rewrite_add_to_sub(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  return poly_uop(ctx, POLY_OP_SUB, root->dtype, root->src, root->n_src, root->arg);
}

static PolyUOp *rewrite_sub_to_add(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  return poly_uop(ctx, POLY_OP_ADD, root->dtype, root->src, root->n_src, root->arg);
}

TEST(pat, graph_rewrite_bottom_up_cycle_fails_closed) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(1));
  PolyUOp *b = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(2));
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, a, b, poly_arg_none());

  PolyPat *add_pat = poly_pat_op2(POLY_OP_ADD, poly_pat_any("x"), poly_pat_any("y"), NULL);
  PolyPat *sub_pat = poly_pat_op2(POLY_OP_SUB, poly_pat_any("x"), poly_pat_any("y"), NULL);
  PolyRule rules[] = {{add_pat, rewrite_add_to_sub}, {sub_pat, rewrite_sub_to_add}};
  PolyPatternMatcher *pm = poly_pm_new(rules, 2);

  PolyUOp *result = poly_graph_rewrite_ex(ctx, add, pm, true);
  ASSERT_TRUE(result == NULL);

  poly_pm_destroy(pm);
  poly_pat_free(add_pat);
  poly_pat_free(sub_pat);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pat, graph_rewrite_stack_limit_fails_closed) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *src[32];
  for (int i = 0; i < 32; i++)
    src[i] = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(i));
  PolyUOp *sink = poly_uop(ctx, POLY_OP_SINK, POLY_VOID, src, 32, poly_arg_none());

  PolyPat *never_pat = poly_pat_op(POLY_OP_ADD, NULL, 0, NULL);
  PolyRule rules[] = {{never_pat, test_rewrite_identity}};
  PolyPatternMatcher *pm = poly_pm_new(rules, 1);

  setenv("POLY_REWRITE_STACK_LIMIT", "8", 1);
  PolyUOp *result = poly_graph_rewrite(ctx, sink, pm);
  unsetenv("POLY_REWRITE_STACK_LIMIT");
  ASSERT_TRUE(result == NULL);

  poly_pm_destroy(pm);
  poly_pat_free(never_pat);
  poly_ctx_destroy(ctx);
  PASS();
}

/* walk_rewrite tests */

static PolyUOp *rewrite_const5_to_const6(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  if (root->op == POLY_OP_CONST && root->arg.kind == POLY_ARG_INT && root->arg.i == 5)
    return poly_uop0(ctx, POLY_OP_CONST, root->dtype, poly_arg_int(6));
  return NULL;
}

static PolyUOp *rewrite_const6_to_const7(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  if (root->op == POLY_OP_CONST && root->arg.kind == POLY_ARG_INT && root->arg.i == 6)
    return poly_uop0(ctx, POLY_OP_CONST, root->dtype, poly_arg_int(7));
  return NULL;
}

TEST(pat, walk_rewrite_no_retraversal) {
  /* Substitution: CONST(5) -> CONST(6) via bpm.
   * A second rule: CONST(6) -> CONST(7) via pm.
   * With walk_rewrite: bpm fires on CONST(5) -> CONST(6), result stored as-is.
   * The pm rule (6->7) does NOT fire because walk_rewrite doesn't re-traverse
   * into the bpm result.
   * With unified_rewrite: both rules would fire (5->6->7). */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *c5 = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(5));
  PolyUOp *c1 = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(1));
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, c5, c1, poly_arg_none());

  /* bpm: 5 -> 6 */
  PolyPat *p_bpm = poly_pat_cvar("c");
  PolyRule r_bpm[] = {{p_bpm, rewrite_const5_to_const6}};
  PolyPatternMatcher *bpm = poly_pm_new(r_bpm, 1);

  /* pm: 6 -> 7 */
  PolyPat *p_pm = poly_pat_cvar("c");
  PolyRule r_pm[] = {{p_pm, rewrite_const6_to_const7}};
  PolyPatternMatcher *pm = poly_pm_new(r_pm, 1);

  PolyUOp *result = poly_graph_walk_rewrite(ctx, add, pm, bpm, NULL, true);
  ASSERT_NOT_NULL(result);
  /* The 5 was rewritten to 6 by bpm, but pm (6->7) did NOT fire on it
   * because walk_rewrite doesn't re-traverse bpm results. */
  ASSERT_TRUE(result->op == POLY_OP_ADD);
  PolyUOp *left = result->src[0];
  ASSERT_TRUE(left->op == POLY_OP_CONST && left->arg.i == 6); /* 6, not 7 */

  poly_pm_destroy(bpm);
  poly_pm_destroy(pm);
  poly_pat_free(p_bpm);
  poly_pat_free(p_pm);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pat, walk_rewrite_bpm_short_circuits) {
  /* bpm rewrites NEG(x) -> CONST(0). Children of NEG should NOT be visited. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *c1 = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(1.0));
  PolyUOp *neg = poly_uop1(ctx, POLY_OP_NEG, POLY_FLOAT32, c1, poly_arg_none());
  PolyUOp *c2 = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(2.0));
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, neg, c2, poly_arg_none());

  PolyPat *neg_pat = poly_pat_op1(POLY_OP_NEG, poly_pat_any("x"), NULL);
  PolyRule r_bpm[] = {{neg_pat, rewrite_neg_to_zero}};
  PolyPatternMatcher *bpm = poly_pm_new(r_bpm, 1);

  PolyUOp *result = poly_graph_walk_rewrite(ctx, add, NULL, bpm, NULL, true);
  ASSERT_NOT_NULL(result);
  /* NEG was rewritten to CONST(0) by bpm */
  ASSERT_TRUE(result->op == POLY_OP_ADD);
  ASSERT_TRUE(result->src[0]->op == POLY_OP_CONST);
  ASSERT_TRUE(result->src[0]->arg.f == 0.0);

  poly_pm_destroy(bpm);
  poly_pat_free(neg_pat);
  poly_ctx_destroy(ctx);
  PASS();
}
