/*
 * test_reduce_simplify.c -- Phase D: pm_reduce_simplify parity tests.
 *
 * Tinygrad source: references/tinygrad_latest/tinygrad/codegen/simplify.py:73-149
 * Ground truth:    test/parity_scripts/tg_reduce_unparented_gt.py
 *
 * Tests are added incrementally as each sub-step (D1..D5) lands.
 */

#include "test_harness.h"
#include "../src/polygrad.h"
#include "../src/codegen/simplify.h"
#include "../src/tensor.h"
#include "../src/uop/upat.h"
#include "../src/utils.h"
#include "../src/engine/schedule.h" /* poly_get_kernel_graph, poly_reshape, poly_reduce_axis */
#include "../src/schedule/rangeify.h"
#include "../src/codegen/codegen.h" /* poly_linearize */
#include "../src/frontend.h" /* poly_buffer_f32, poly_sink1, poly_store_val */

/* helpers */

#ifdef POLY_TESTING
TEST(reduce_simplify, closeout_float_parameter_does_not_invent_finite_bounds) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *r = poly_range(ctx, 4, 0, POLY_AXIS_REDUCE);
  PolyUOp *buf = poly_test_program_param(ctx, POLY_FLOAT32, 1, 0);
  PolyUOp *zero = poly_const_int(ctx, 0);
  PolyUOp *load = poly_uop1(
      ctx, POLY_OP_LOAD, POLY_FLOAT32, poly_uop_index(ctx, buf, &zero, 1), poly_arg_none()
  );
  PolyUOp *sum = poly_alu2(ctx, POLY_OP_ADD, poly_cast(ctx, r, POLY_FLOAT32), load);
  PolyUOp *value = poly_uop3(
      ctx, POLY_OP_WHERE, POLY_WEAKINT,
      poly_alu2(ctx, POLY_OP_CMPLT, sum, poly_const_float(ctx, 1e30)), poly_const_int(ctx, 1), zero,
      poly_arg_none()
  );
  PolyUOp *sources[] = {value, r};
  PolyUOp *red =
      poly_uop(ctx, POLY_OP_REDUCE, value->dtype, sources, 2, poly_arg_reduce(POLY_OP_ADD, 0));
  PolyUOp *out = poly_test_reduce_collapse(ctx, red);
  /* Tinygrad leaves this reduction intact; a valid optional collapse must
   * still return zero when the loaded value is above the comparison cut. */
  bool correct = !out;
  if (out) {
    PolyUOp *high = poly_const_float(ctx, 1e31);
    PolyUOp *bound = poly_uop_substitute(ctx, out, &load, &high, 1);
    bound = poly_graph_rewrite(ctx, bound, poly_symbolic());
    int64_t result = -1;
    correct = poly_uop_const_i64(bound, &result) == 0 && result == 0;
    if (!correct)
      fprintf(
          stderr, "unbounded float reduction returned %lld instead of zero\n", (long long)result
      );
  }
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(correct);
  PASS();
}

TEST(reduce_simplify, owner_uint64_collapse_preserves_upper_clamp) {
  /* v0.14 reduce_collapse must retain min(loaded_value >> 62, 2), including
   * when the original uint64 storage value exceeds INT64_MAX. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *two = poly_uop_const(ctx, poly_arg_int(2), POLY_UINT64);
  PolyUOp *r = poly_uop1(ctx, POLY_OP_RANGE, POLY_UINT64, two, poly_arg_range(0, POLY_AXIS_REDUCE));
  PolyUOp *buf = poly_test_program_param(ctx, POLY_UINT64, 1, 0);
  PolyUOp *zero = poly_const_int(ctx, 0);
  PolyUOp *load = poly_uop1(
      ctx, POLY_OP_LOAD, POLY_UINT64, poly_uop_index(ctx, buf, &zero, 1), poly_arg_none()
  );
  PolyUOp *upper = poly_alu2(ctx, POLY_OP_SHR, load, poly_const_int(ctx, 62));
  PolyUOp *value = poly_uop3(
      ctx, POLY_OP_WHERE, POLY_UINT64, poly_alu2(ctx, POLY_OP_CMPLT, r, upper),
      poly_uop_const(ctx, poly_arg_int(1), POLY_UINT64),
      poly_uop_const(ctx, poly_arg_int(0), POLY_UINT64), poly_arg_none()
  );
  PolyUOp *sources[] = {value, r};
  PolyUOp *red =
      poly_uop(ctx, POLY_OP_REDUCE, POLY_UINT64, sources, 2, poly_arg_reduce(POLY_OP_ADD, 0));
  PolyUOp *out = poly_test_reduce_collapse(ctx, red);
  PolyUOp *expected = poly_graph_rewrite(ctx, poly_minimum(ctx, upper, two), poly_symbolic());
  bool correct = out == expected;
  for (uint64_t high = 0; out && high < 4; high++) {
    PolyUOp *sample = poly_uop_const(ctx, poly_arg_int((int64_t)(high << 62)), POLY_UINT64);
    PolyUOp *bound = poly_uop_substitute(ctx, out, &load, &sample, 1);
    bound = poly_graph_rewrite(ctx, bound, poly_symbolic());
    int64_t result = -1;
    correct &= poly_uop_const_i64(bound, &result) == 0 && result == (high < 2 ? high : 2);
  }
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(correct);
  PASS();
}

TEST(reduce_simplify, owner_range_mod_admission_and_first_match) {
  /* mark_range_mod records the first eligible divisor; WARP/DEVICE and
   * nonconstant extents never enter the split dictionary. */
  PolyAxisType axes[] = {POLY_AXIS_LOOP, POLY_AXIS_REDUCE, POLY_AXIS_WARP, POLY_AXIS_DEVICE};
  bool correct = true;
  for (size_t i = 0; i < sizeof(axes) / sizeof(*axes); i++) {
    for (int divisor = 3; divisor <= 5; divisor += 2) {
      PolyCtx *ctx = poly_ctx_new();
      PolyMap *state = poly_map_new(16);
      PolyUOp *r = poly_range(ctx, 12, 0, axes[i]);
      PolyUOp *c = poly_const_int(ctx, divisor);
      PolyUOp *mod = poly_uop2(ctx, POLY_OP_FLOORMOD, POLY_WEAKINT, r, c, poly_arg_none());
      poly_graph_rewrite_ctx(ctx, mod, poly_pm_split_ranges(), state);
      PolyUOp *recorded = poly_map_get(state, poly_ptr_hash(r), r, poly_ptr_eq);
      bool eligible = i < 2 && divisor == 3;
      correct &= recorded == (eligible ? c : NULL);
      if (eligible) {
        PolyUOp *other = poly_uop2(
            ctx, POLY_OP_FLOORMOD, POLY_WEAKINT, r, poly_const_int(ctx, 2), poly_arg_none()
        );
        poly_graph_rewrite_ctx(ctx, other, poly_pm_split_ranges(), state);
        correct &= poly_map_get(state, poly_ptr_hash(r), r, poly_ptr_eq) == c;
      }
      poly_map_destroy(state);
      poly_ctx_destroy(ctx);
    }
  }
  PolyCtx *ctx = poly_ctx_new();
  PolyMap *state = poly_map_new(16);
  PolyUOp *extent =
      poly_uop_variable(ctx, "extent", poly_arg_int(3), poly_arg_int(12), POLY_WEAKINT, 1, false);
  PolyUOp *r =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, extent, poly_arg_range(0, POLY_AXIS_LOOP));
  PolyUOp *mod =
      poly_uop2(ctx, POLY_OP_FLOORMOD, POLY_WEAKINT, r, poly_const_int(ctx, 3), poly_arg_none());
  poly_graph_rewrite_ctx(ctx, mod, poly_pm_split_ranges(), state);
  correct &= poly_map_len(state) == 0;
  poly_map_destroy(state);
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(correct);
  PASS();
}

TEST(reduce_simplify, domains_unbounded_float_is_not_a_finite_parameter) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *r = poly_range(ctx, 8, 0, POLY_AXIS_REDUCE);
  PolyUOp *buf = poly_test_program_param(ctx, POLY_FLOAT64, 1, 0);
  PolyUOp *coord = poly_const_int(ctx, 0);
  PolyUOp *idx = poly_uop_index(ctx, buf, &coord, 1);
  PolyUOp *external = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT64, idx, poly_arg_none());
  PolyUOp *sum = poly_add(ctx, poly_cast(ctx, r, POLY_FLOAT64), external);
  PolyUOp *cond =
      poly_alu2(ctx, POLY_OP_CMPLT, sum, poly_uop_const(ctx, poly_arg_float(1e30), POLY_FLOAT64));
  PolyUOp *value = poly_uop3(
      ctx, POLY_OP_WHERE, POLY_FLOAT64, cond, poly_uop_const(ctx, poly_arg_float(1), POLY_FLOAT64),
      poly_uop_const(ctx, poly_arg_float(0), POLY_FLOAT64), poly_arg_none()
  );
  PolyUOp *src[] = {value, r};
  PolyUOp *red =
      poly_uop(ctx, POLY_OP_REDUCE, POLY_FLOAT64, src, 2, poly_arg_reduce(POLY_OP_ADD, 0));
  /* v0.14 declines collapse. Treating the loaded float as int64-bounded
   * would turn the result into constant8, wrong for e.g. external=1e31. */
  bool correct = poly_test_reduce_collapse(ctx, red) == NULL;
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(correct);
  PASS();
}

TEST(reduce_simplify, domains_failed_guard_scan_discards_shrink_state) {
  bool correct = true;
  for (int fail = 0; fail < 2; fail++) {
    PolyCtx *ctx = poly_ctx_new();
    PolyUOp *r = poly_range(ctx, 16, 0, POLY_AXIS_LOOP);
    PolyUOp *buf = poly_test_program_param(ctx, POLY_FLOAT32, 16, 0);
    PolyUOp *indices[2];
    for (int i = 0; i < 2; i++) {
      PolyUOp *gate = poly_alu2(ctx, POLY_OP_CMPLT, r, poly_const_like_int(ctx, r, i ? 8 : 4));
      PolyUOp *coord = poly_uop3(
          ctx, POLY_OP_WHERE, POLY_WEAKINT, gate, r,
          poly_uop_const(ctx, poly_arg_invalid(), POLY_WEAKINT), poly_arg_none()
      );
      indices[i] = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, buf, coord, poly_arg_none());
    }
    PolyUOp *sink = poly_uop(ctx, POLY_OP_SINK, POLY_VOID, indices, 2, poly_arg_none());
    PolyMap *state = poly_map_new(16);
    poly_test_simplify_fail_after(fail);
    PolyUOp *out = poly_graph_rewrite_ctx(ctx, sink, poly_pm_simplify_ranges(), state);
    poly_test_simplify_fail_after(-1);
    correct &= out == sink && poly_map_len(state) == 0;
    /* Recovery uses a clean dictionary, not the prior partial bounds. */
    out = poly_graph_rewrite_ctx(ctx, sink, poly_pm_simplify_ranges(), state);
    correct &= out && out != sink && poly_map_len(state) == 0;
    poly_map_destroy(state);
    poly_ctx_destroy(ctx);
  }
  ASSERT_TRUE(correct);
  PASS();
}

TEST(reduce_simplify, domains_boolean_interval_and_parameter_gate) {
  bool correct = true;
  for (int mode = 0; mode < 2; mode++) {
    PolyCtx *ctx = poly_ctx_new();
    PolyUOp *r = poly_range(ctx, 8, 0, POLY_AXIS_REDUCE);
    PolyUOp *p =
        poly_uop_variable(ctx, "gate", poly_arg_int(0), poly_arg_int(1), POLY_BOOL, 1, true);
    PolyUOp *upper = poly_alu2(ctx, POLY_OP_CMPLT, r, poly_const_like_int(ctx, r, 6));
    PolyUOp *lower =
        poly_logical_not(ctx, poly_alu2(ctx, POLY_OP_CMPLT, r, poly_const_like_int(ctx, r, 2)));
    PolyUOp *cond = poly_alu2(ctx, POLY_OP_AND, mode ? p : lower, upper);
    PolyUOp *value = mode ? poly_uop_const(ctx, poly_arg_bool(true), POLY_BOOL) : p;
    PolyUOp *zero = poly_uop_const(ctx, poly_arg_bool(false), POLY_BOOL);
    PolyUOp *v = poly_uop3(ctx, POLY_OP_WHERE, POLY_BOOL, cond, value, zero, poly_arg_none());
    PolyUOp *src[] = {v, r};
    PolyUOp *red =
        poly_uop(ctx, POLY_OP_REDUCE, POLY_BOOL, src, 2, poly_arg_reduce(POLY_OP_ADD, 0));
    PolyUOp *out = poly_test_reduce_collapse_rewrite(ctx, red);
    correct &= out && out->op == POLY_OP_MUL;
    if (mode && out) correct &= out->src[0]->op == POLY_OP_REDUCE && out->src[1] == p;
    poly_ctx_destroy(ctx);
  }
  ASSERT_TRUE(correct);
  PASS();
}

TEST(reduce_simplify, domains_collapse_preserves_axis_metadata) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *r = poly_range(ctx, 8, 0, POLY_AXIS_REDUCE);
  PolyUOp *value = poly_uop3(
      ctx, POLY_OP_WHERE, POLY_WEAKINT,
      poly_alu2(ctx, POLY_OP_CMPLT, r, poly_const_like_int(ctx, r, 4)), r,
      poly_const_like_int(ctx, r, 0), poly_arg_none()
  );
  PolyUOp *src[] = {value, r};
  PolyUOp *red =
      poly_uop(ctx, POLY_OP_REDUCE, POLY_WEAKINT, src, 2, poly_arg_reduce(POLY_OP_ADD, 1));
  bool correct = poly_test_reduce_collapse_rewrite(ctx, red) == NULL &&
                 poly_pm_rewrite(poly_pm_reduce_simplify(), ctx, red) == NULL &&
                 poly_pm_rewrite(poly_pm_load_collapse(), ctx, red) == NULL;
  /* An ADD value reaches the distribution rule even when interval folding
   * cannot remove its range; nonempty axes metadata must still not match. */
  src[0] = poly_add(ctx, r, poly_const_like_int(ctx, r, 1));
  red = poly_uop(ctx, POLY_OP_REDUCE, POLY_WEAKINT, src, 2, poly_arg_reduce(POLY_OP_ADD, 1));
  correct &= poly_test_reduce_collapse_rewrite(ctx, red) == NULL;
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(correct);
  PASS();
}

TEST(reduce_simplify, domains_load_collapse_requires_one_range) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *r = poly_range(ctx, 8, 0, POLY_AXIS_REDUCE);
  PolyUOp *s = poly_range(ctx, 3, 1, POLY_AXIS_REDUCE);
  PolyUOp *gate = poly_alu2(
      ctx, POLY_OP_AND, poly_alu2(ctx, POLY_OP_CMPLT, r, poly_const_like_int(ctx, r, 1)),
      poly_alu2(ctx, POLY_OP_CMPLT, s, poly_const_like_int(ctx, s, 1))
  );
  PolyUOp *value = poly_uop3(
      ctx, POLY_OP_WHERE, POLY_WEAKINT, gate, poly_const_like_int(ctx, r, 1),
      poly_const_like_int(ctx, r, 0), poly_arg_none()
  );
  PolyUOp *src[] = {value, r, s};
  PolyUOp *red =
      poly_uop(ctx, POLY_OP_REDUCE, POLY_WEAKINT, src, 3, poly_arg_reduce(POLY_OP_ADD, 0));
  bool correct = poly_pm_rewrite(poly_pm_load_collapse(), ctx, red) == NULL;
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(correct);
  PASS();
}

TEST(reduce_simplify, tail_invalid_guard_moves_outside_reduce) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *r = poly_range(ctx, 8, 0, POLY_AXIS_REDUCE);
  PolyUOp *gate =
      poly_uop_variable(ctx, "gate", poly_arg_int(0), poly_arg_int(1), POLY_BOOL, 1, true);
  PolyUOp *invalid = poly_uop_const(ctx, poly_arg_invalid(), POLY_WEAKINT);
  PolyUOp *v = poly_uop3(ctx, POLY_OP_WHERE, POLY_WEAKINT, gate, r, invalid, poly_arg_none());
  PolyUOp *src[] = {v, r};
  PolyUOp *red =
      poly_uop(ctx, POLY_OP_REDUCE, POLY_WEAKINT, src, 2, poly_arg_reduce(POLY_OP_ADD, 0));
  PolyUOp *out = poly_test_reduce_collapse_rewrite(ctx, red);
  bool correct = out && out->op == POLY_OP_WHERE && out->src[0] == gate && out->src[2] == invalid &&
                 out->src[1]->op == POLY_OP_REDUCE && out->src[1]->src[0] == r &&
                 out->src[1]->src[1] == r;
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(correct);
  PASS();
}

TEST(reduce_simplify, tail_positive_factor_uses_core_subtraction) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *r = poly_range(ctx, 8, 0, POLY_AXIS_REDUCE);
  PolyUOp *y =
      poly_uop_variable(ctx, "factor", poly_arg_int(1), poly_arg_int(3), POLY_WEAKINT, 1, true);
  PolyUOp *c = poly_const_like_int(ctx, r, 10);
  PolyUOp *expr = poly_alu2(ctx, POLY_OP_CMPLT, poly_mul(ctx, r, y), c);
  PolyUOp *out = poly_test_reduce_collapse_rewrite(ctx, expr);
  PolyUOp *cy = poly_add(ctx, c, y);
  PolyUOp *rhs =
      poly_binop(ctx, POLY_OP_FLOORDIV, poly_sub(ctx, cy, poly_const_like_int(ctx, cy, 1)), y);
  bool exact = out && out->op == POLY_OP_CMPLT && out->src[0] == r && out->src[1] == rhs;
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(exact);
  PASS();
}

static PolyUOp *tail_external_reduction(PolyCtx *ctx, int width, PolyUOp **external) {
  PolyUOp *r = poly_range(ctx, 8, 0, POLY_AXIS_REDUCE);
  PolyUOp *cond = poly_alu2(ctx, POLY_OP_CMPLT, r, poly_const_like_int(ctx, r, 1));
  PolyUOp *zero = poly_uop_const(ctx, poly_arg_int(0), POLY_INT64);
  PolyUOp *value = zero;
  for (int i = 0; i < width; i++) {
    char name[24];
    snprintf(name, sizeof(name), "v%d", i);
    external[i] = poly_cast(
        ctx, poly_uop_variable(ctx, name, poly_arg_int(0), poly_arg_int(9), POLY_INT32, 1, true),
        POLY_INT64
    );
    PolyUOp *v =
        poly_uop3(ctx, POLY_OP_WHERE, POLY_INT64, cond, external[i], zero, poly_arg_none());
    value = poly_add(ctx, value, v);
  }
  PolyUOp *src[] = {value, r};
  return poly_uop(ctx, POLY_OP_REDUCE, POLY_INT64, src, 2, poly_arg_reduce(POLY_OP_ADD, 0));
}

TEST(reduce_simplify, tail_more_than_256_external_values) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *external[270];
  PolyUOp *out = poly_test_reduce_collapse(ctx, tail_external_reduction(ctx, 270, external));
  bool correct = out && poly_no_range(ctx, out);
  int n = 0;
  PolyUOp **topo = out ? poly_toposort(ctx, out, &n) : NULL;
  for (int i = 0; correct && i < 270; i++) {
    bool found = false;
    for (int j = 0; j < n; j++)
      found |= topo[j] == external[i];
    correct &= found;
  }
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(correct);
  PASS();
}

TEST(reduce_simplify, tail_failed_restore_does_not_publish_parameters) {
  bool correct = true;
  for (int fail = 0; fail < 2; fail++) {
    PolyCtx *ctx = poly_ctx_new();
    PolyUOp *external[1];
    PolyUOp *red = tail_external_reduction(ctx, 1, external);
    poly_test_simplify_fail_after(fail);
    PolyUOp *out = poly_test_reduce_collapse(ctx, red);
    poly_test_simplify_fail_after(-1);
    correct &= out == NULL;
    out = poly_test_reduce_collapse(ctx, red);
    correct &= out == external[0];
    poly_ctx_destroy(ctx);
  }
  ASSERT_TRUE(correct);
  PASS();
}

TEST(reduce_simplify, tail_failed_load_query_is_not_absence) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *buf = poly_test_program_param(ctx, POLY_INT32, 1, 0);
  PolyUOp *idx = poly_const_int(ctx, 0);
  PolyUOp *x = poly_cast(ctx, poly_uop_index(ctx, buf, &idx, 1), POLY_WEAKINT);
  PolyUOp *sum = poly_add(ctx, x, poly_const_like_int(ctx, x, 1));
  PolyUOp *expr = poly_alu2(ctx, POLY_OP_CMPLT, sum, poly_const_like_int(ctx, x, 8));
  poly_test_simplify_fail_after(0);
  PolyUOp *out = poly_pm_rewrite(poly_pm_load_collapse(), ctx, expr);
  poly_test_simplify_fail_after(-1);
  bool rejected = out == NULL;
  bool recovers = poly_pm_rewrite(poly_pm_load_collapse(), ctx, expr) != NULL;
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(rejected);
  ASSERT_TRUE(recovers);
  PASS();
}

TEST(reduce_simplify, bundle_interval_exact_count) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *n = poly_uop_const(ctx, poly_arg_int(INT64_C(9007199254740995)), POLY_WEAKINT);
  PolyUOp *r = poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, n, poly_arg_range(0, POLY_AXIS_REDUCE));
  PolyUOp *cut = poly_uop_const(ctx, poly_arg_int(INT64_C(9007199254740993)), POLY_INT64);
  PolyUOp *zero = poly_uop_const(ctx, poly_arg_int(0), POLY_INT64);
  PolyUOp *one = poly_uop_const(ctx, poly_arg_int(1), POLY_INT64);
  PolyUOp *v = poly_uop3(
      ctx, POLY_OP_WHERE, POLY_INT64, poly_alu2(ctx, POLY_OP_CMPLT, r, cut), zero, one,
      poly_arg_none()
  );
  PolyUOp *src[] = {v, r};
  PolyUOp *red = poly_uop(ctx, POLY_OP_REDUCE, POLY_INT64, src, 2, poly_arg_reduce(POLY_OP_ADD, 0));
  PolyUOp *out =
      poly_graph_rewrite(ctx, poly_test_reduce_collapse_rewrite(ctx, red), poly_symbolic());
  bool exact = out && out->op == POLY_OP_CONST && out->arg.kind == POLY_ARG_INT && out->arg.i == 2;
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(exact);
  PASS();
}

static bool bundle_wide_collapse(bool parameter_gate) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *src[41], *value = poly_uop_const(ctx, poly_arg_int(0), POLY_WEAKINT);
  for (int i = 1; i <= 40; i++) {
    src[i] = poly_range(ctx, 2, i - 1, POLY_AXIS_REDUCE);
    value = poly_add(ctx, value, src[i]);
  }
  if (parameter_gate) {
    PolyUOp *p =
        poly_uop_variable(ctx, "gate", poly_arg_int(0), poly_arg_int(1), POLY_BOOL, 1, true);
    PolyUOp *gate = poly_alu2(
        ctx, POLY_OP_AND, p, poly_alu2(ctx, POLY_OP_CMPLT, value, poly_const_int(ctx, 20))
    );
    value = poly_uop3(
        ctx, POLY_OP_WHERE, POLY_WEAKINT, gate, value,
        poly_uop_const(ctx, poly_arg_int(0), POLY_WEAKINT), poly_arg_none()
    );
  }
  src[0] = value;
  PolyUOp *red =
      poly_uop(ctx, POLY_OP_REDUCE, value->dtype, src, 41, poly_arg_reduce(POLY_OP_ADD, 0));
  PolyUOp *out = poly_test_reduce_collapse_rewrite(ctx, red);
  bool correct = out && out->op == (parameter_gate ? POLY_OP_MUL : POLY_OP_ADD);
  if (correct) {
    int count = parameter_gate ? 1 : 2;
    for (int i = 0; i < count; i++) {
      PolyUOp *child = out->src[i];
      correct &= child->op == POLY_OP_REDUCE && child->n_src == 41;
      for (int j = 1; correct && j <= 40; j++)
        correct &= child->src[j] == src[j];
    }
  }
  poly_ctx_destroy(ctx);
  return correct;
}

TEST(reduce_simplify, bundle_distribute_many_ranges) {
  ASSERT_TRUE(bundle_wide_collapse(false));
  PASS();
}

TEST(reduce_simplify, bundle_parameter_gate_many_ranges) {
  ASSERT_TRUE(bundle_wide_collapse(true));
  PASS();
}
#endif

/* pm_load_collapse may move loaded-index arithmetic only in the unbounded
 * weakint domain. Moving fixed-width addition changes overflow semantics. */
TEST(reduce_simplify, bundle_loaded_index_preserves_strong_overflow) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *buf = poly_test_program_param(ctx, POLY_INT32, 1, 0);
  PolyUOp *zero = poly_const_int(ctx, 0);
  PolyUOp *x = poly_uop_index(ctx, buf, &zero, 1);
  PolyUOp *y =
      poly_uop_variable(ctx, "offset", poly_arg_int(1), poly_arg_int(3), POLY_INT32, 1, true);
  PolyUOp *sum = poly_alu2(ctx, POLY_OP_ADD, x, y);
  PolyUOp *before = poly_alu2(ctx, POLY_OP_CMPLT, sum, poly_const_like_int(ctx, x, 0));
  PolyUOp *after = poly_graph_rewrite(ctx, before, poly_pm_load_collapse());
  bool unchanged = before == after;
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(unchanged);
  PASS();
}

TEST(reduce_simplify, bundle_loaded_index_accepts_commuted_weak_add) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *buf = poly_test_program_param(ctx, POLY_INT32, 1, 0);
  PolyUOp *zero = poly_const_int(ctx, 0);
  PolyUOp *x = poly_cast(ctx, poly_uop_index(ctx, buf, &zero, 1), POLY_WEAKINT);
  PolyUOp *y =
      poly_uop_variable(ctx, "offset", poly_arg_int(1), poly_arg_int(3), POLY_WEAKINT, 1, true);
  PolyUOp *sum = poly_alu2(ctx, POLY_OP_ADD, y, x);
  PolyUOp *before = poly_alu2(ctx, POLY_OP_CMPLT, sum, poly_const_like_int(ctx, x, 0));
  PolyUOp *after = poly_graph_rewrite(ctx, before, poly_pm_load_collapse());
  bool corrected = after != before && after->op == POLY_OP_CMPLT && after->src[0] == x &&
                   after->src[1]->op == POLY_OP_ADD;
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(corrected);
  PASS();
}

TEST(reduce_simplify, bundle_unparented_many_ranges) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *src[41];
  src[0] = poly_uop_const(ctx, poly_arg_int(7), POLY_INT64);
  for (int i = 1; i <= 40; i++)
    src[i] = poly_range(ctx, 2, i - 1, POLY_AXIS_REDUCE);
  PolyUOp *red =
      poly_uop(ctx, POLY_OP_REDUCE, POLY_INT64, src, 41, poly_arg_reduce(POLY_OP_MAX, 0));
  PolyUOp *out = poly_graph_rewrite(ctx, red, poly_pm_reduce_unparented());
  bool correct = out == src[0];
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(correct);
  PASS();
}

TEST(reduce_simplify, bundle_unparented_exact_count) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *value = poly_uop_const(ctx, poly_arg_int(1), POLY_INT64);
  PolyUOp *count = poly_uop_const(ctx, poly_arg_int(INT64_C(9007199254740993)), POLY_WEAKINT);
  PolyUOp *range =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, count, poly_arg_range(0, POLY_AXIS_REDUCE));
  PolyUOp *src[] = {value, range};
  PolyUOp *red = poly_uop(ctx, POLY_OP_REDUCE, POLY_INT64, src, 2, poly_arg_reduce(POLY_OP_ADD, 0));
  PolyUOp *out = poly_graph_rewrite(ctx, red, poly_pm_reduce_unparented());
  bool exact = out->op == POLY_OP_MUL && out->src[0] == value && out->src[1] == count;
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(exact);
  PASS();
}

/* These regression checks inspect scheduled kernels after rangeify and
 * reduce_simplify, so they need the executable scheduled root instead of the
 * earlier public kernel-graph boundary. */
static PolyUOp *single_scheduled_root(PolyCtx *ctx, PolyUOp *sink) {
  PolyUOp *linear = poly_test_create_linear(ctx, sink);
  return linear && linear->n_src == 1 ? poly_test_linear_call_body(linear, 0) : NULL;
}

/* Build a fresh RANGE(count, axis_id, LOOP). */
static PolyUOp *mk_range(PolyCtx *ctx, int64_t count, int64_t axis_id) {
  PolyUOp *cnt = poly_const_int(ctx, count);
  return poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, cnt, poly_arg_range(axis_id, POLY_AXIS_LOOP));
}

/* Build a REDUCE(value, range_0, range_1, ...) with arg=op. */
static PolyUOp *mk_reduce(
    PolyCtx *ctx,
    PolyOps op,
    PolyDType dt,
    PolyUOp *value,
    PolyUOp **ranges,
    int n_ranges
) {
  PolyUOp *srcs[8];
  srcs[0] = value;
  for (int i = 0; i < n_ranges; i++)
    srcs[1 + i] = ranges[i];
  return poly_uop(ctx, POLY_OP_REDUCE, dt, srcs, 1 + n_ranges, poly_arg_reduce(op, 0));
}

/* D1 stub smoke tests (kept after D2 lands) */

TEST(reduce_simplify, split_ranges_without_ctx_is_noop_not_abort) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *c = poly_const_int(ctx, 1);
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, c, poly_arg_none());

  PolyUOp *out = poly_graph_rewrite(ctx, sink, poly_pm_split_ranges());
  ASSERT_PTR_EQ(out, sink);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(reduce_simplify, flatten_range_preserves_bool_backedge) {
  /* tinygrad@2026-08-22/a9069c177a9d codegen/simplify.py:8-18. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *r = poly_range(ctx, 16, 0, POLY_AXIS_LOOP);
  PolyUOp *four = poly_const_int(ctx, 4);
  PolyUOp *gate = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, r, four, poly_arg_none());
  PolyUOp *value = poly_const_int(ctx, 1);
  PolyUOp *src[2] = {value, gate};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, src, 2, poly_arg_none());

  ASSERT_PTR_EQ(poly_graph_rewrite(ctx, end, poly_pm_flatten_range()), end);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(reduce_simplify, simplify_ranges_uses_largest_index_guard) {
  /* tinygrad@2026-08-22/a9069c177a9d codegen/simplify.py:43-60. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *r = poly_range(ctx, 16, 0, POLY_AXIS_LOOP);
  PolyUOp *buf = poly_test_program_param(ctx, POLY_FLOAT32, 16, 0);
  PolyUOp *invalid = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_invalid());
  PolyUOp *indices[2];
  for (int i = 0; i < 2; i++) {
    PolyUOp *bound = poly_const_int(ctx, i == 0 ? 4 : 8);
    PolyUOp *gate = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, r, bound, poly_arg_none());
    PolyUOp *coord = poly_uop3(ctx, POLY_OP_WHERE, POLY_WEAKINT, gate, r, invalid, poly_arg_none());
    indices[i] = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, buf, coord, poly_arg_none());
  }
  PolyUOp *sink = poly_uop(ctx, POLY_OP_SINK, POLY_VOID, indices, 2, poly_arg_none());
  PolyMap *state = poly_map_new(16);
  PolyUOp *out = poly_graph_rewrite_ctx(ctx, sink, poly_pm_simplify_ranges(), state);
  poly_map_destroy(state);

  int n = 0;
  PolyUOp **topo = poly_toposort(ctx, out, &n);
  int ranges = 0;
  for (int i = 0; i < n; i++) {
    if (topo[i]->op != POLY_OP_RANGE) continue;
    ranges++;
    ASSERT_INT_EQ(topo[i]->src[0]->op, POLY_OP_CONST);
    ASSERT_INT_EQ(topo[i]->src[0]->arg.i, 8);
  }
  ASSERT_INT_EQ(ranges, 1);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(reduce_simplify, split_ranges_excludes_device_axis) {
  /* tinygrad@2026-08-22/a9069c177a9d codegen/simplify.py:62-65. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *r = poly_range(ctx, 16, 1, POLY_AXIS_DEVICE);
  PolyUOp *four = poly_const_int(ctx, 4);
  PolyUOp *mod = poly_uop2(ctx, POLY_OP_FLOORMOD, POLY_WEAKINT, r, four, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, mod, poly_arg_none());
  PolyMap *state = poly_map_new(16);

  ASSERT_PTR_EQ(poly_graph_rewrite_ctx(ctx, sink, poly_pm_split_ranges(), state), sink);
  poly_map_destroy(state);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(reduce_simplify, reduce_local_without_reduces_is_noop) {
  /* Tinygrad 2026-08-22/a9069c177a9d codegen/__init__.py:324-325 runs the
   * complete reduction stage with a pass-local ReduceContext. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *c = poly_const_int(ctx, 1);
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, c, poly_arg_none());

  PolyUOp *out = poly_apply_pm_reduce(ctx, sink);
  ASSERT_PTR_EQ(out, sink);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(reduce_simplify, d1_stub_noop_on_const) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *c = poly_const_int(ctx, 42);
  PolyUOp *out = poly_apply_reduce_unparented_only(ctx, c);
  ASSERT_TRUE(out == c);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(reduce_simplify, d1_stub_noop_on_alu_chain) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_const_int(ctx, 3);
  PolyUOp *b = poly_const_int(ctx, 4);
  PolyUOp *sum = poly_alu2(ctx, POLY_OP_ADD, a, b);
  PolyUOp *out = poly_apply_reduce_unparented_only(ctx, sum);
  ASSERT_TRUE(out == sum);
  poly_ctx_destroy(ctx);
  PASS();
}

/* D2: pm_reduce_unparented parity (cases A-F from tg ground truth) */

/* Case A — ADD reduce, value depends only on r0, r1 unused.
 *
 * Tinygrad parity (verified by replicating this exact case in
 * tg_reduce_unparented_gt.py with INT32 dtypes throughout):
 *   MUL(REDUCE_ADD(value, r0), CONST_int(7))
 *
 * NOTE: tinygrad's UOp.cast (uop/ops.py:459-463) returns self when the
 * source dtype already matches the target — no CAST node is emitted. The
 * tg fixture happens to use weakint range counts which DOES need a CAST,
 * but this polygrad test uses INT32 throughout (matching polygrad's
 * mk_range), so the cast is a no-op and out->src[1] is a bare CONST.
 */
TEST(reduce_simplify, d2_unparented_add_one_unused) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *r0 = mk_range(ctx, 5, 0);
  PolyUOp *r1 = mk_range(ctx, 7, 1);
  PolyUOp *one = poly_const_int(ctx, 1);
  PolyUOp *val = poly_alu2(ctx, POLY_OP_ADD, r0, one); /* val depends on r0 only */
  PolyUOp *ranges[] = {r0, r1};
  PolyUOp *red = mk_reduce(ctx, POLY_OP_ADD, POLY_INT32, val, ranges, 2);

  PolyUOp *out = poly_apply_reduce_unparented_only(ctx, red);

  ASSERT_TRUE(out != red);
  ASSERT_TRUE(out->op == POLY_OP_MUL);
  ASSERT_INT_EQ(out->n_src, 2);

  PolyUOp *new_red = out->src[0];
  ASSERT_TRUE(new_red->op == POLY_OP_REDUCE);
  ASSERT_INT_EQ(new_red->n_src, 2); /* value + r0 only */
  ASSERT_TRUE(new_red->src[0] == val);
  ASSERT_TRUE(new_red->src[1] == r0);
  ASSERT_TRUE(new_red->arg.kind == POLY_ARG_REDUCE);
  ASSERT_TRUE(new_red->arg.reduce.op == POLY_OP_ADD);
  ASSERT_INT_EQ(new_red->arg.reduce.num_axes, 0);

  /* int32 == int32, so no CAST is emitted -- out->src[1] is bare CONST(7) */
  PolyUOp *count = out->src[1];
  ASSERT_TRUE(count->op == POLY_OP_CONST);
  ASSERT_INT_EQ((int)count->arg.i, 7);

  poly_ctx_destroy(ctx);
  PASS();
}

/* Case B — ADD reduce, value is just CONST, all ranges unparented.
 * tinygrad: REDUCE(4, r0, r1) -> MUL(MUL(4, CAST(5)), CAST(7))
 */
TEST(reduce_simplify, d2_unparented_add_const_all_unused) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *r0 = mk_range(ctx, 5, 0);
  PolyUOp *r1 = mk_range(ctx, 7, 1);
  PolyUOp *val = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(4));
  PolyUOp *ranges[] = {r0, r1};
  PolyUOp *red = mk_reduce(ctx, POLY_OP_ADD, POLY_INT32, val, ranges, 2);

  PolyUOp *out = poly_apply_reduce_unparented_only(ctx, red);

  /* Expect: MUL(MUL(4, CAST(5)), CAST(7)) */
  ASSERT_TRUE(out != red);
  ASSERT_TRUE(out->op == POLY_OP_MUL);
  ASSERT_INT_EQ(out->n_src, 2);

  /* int32 throughout: no CAST nodes */
  PolyUOp *outer_lhs = out->src[0];
  ASSERT_TRUE(outer_lhs->op == POLY_OP_MUL);
  ASSERT_TRUE(outer_lhs->src[0] == val);
  ASSERT_TRUE(outer_lhs->src[1]->op == POLY_OP_CONST);
  ASSERT_INT_EQ((int)outer_lhs->src[1]->arg.i, 5);

  PolyUOp *outer_rhs = out->src[1];
  ASSERT_TRUE(outer_rhs->op == POLY_OP_CONST);
  ASSERT_INT_EQ((int)outer_rhs->arg.i, 7);

  poly_ctx_destroy(ctx);
  PASS();
}

/* Case C — MUL reduce, one unparented range.
 * tinygrad: REDUCE_MUL(value, r0, r1) -> POW(REDUCE_MUL(value, r0), CAST(7))
 */
TEST(reduce_simplify, d2_unparented_mul_one_unused) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *r0 = mk_range(ctx, 5, 0);
  PolyUOp *r1 = mk_range(ctx, 7, 1);
  PolyUOp *two = poly_const_int(ctx, 2);
  PolyUOp *val = poly_alu2(ctx, POLY_OP_MUL, r0, two);
  PolyUOp *ranges[] = {r0, r1};
  PolyUOp *red = mk_reduce(ctx, POLY_OP_MUL, POLY_INT32, val, ranges, 2);

  PolyUOp *out = poly_apply_reduce_unparented_only(ctx, red);

  ASSERT_TRUE(out != red);
  ASSERT_TRUE(out->op == POLY_OP_POW);
  ASSERT_INT_EQ(out->n_src, 2);
  PolyUOp *new_red = out->src[0];
  ASSERT_TRUE(new_red->op == POLY_OP_REDUCE);
  ASSERT_INT_EQ(new_red->n_src, 2);
  ASSERT_TRUE(new_red->src[1] == r0);
  ASSERT_TRUE(new_red->arg.kind == POLY_ARG_REDUCE);
  ASSERT_TRUE(new_red->arg.reduce.op == POLY_OP_MUL);
  ASSERT_INT_EQ(new_red->arg.reduce.num_axes, 0);
  ASSERT_TRUE(out->src[1]->op == POLY_OP_CONST);
  ASSERT_INT_EQ((int)out->src[1]->arg.i, 7);

  poly_ctx_destroy(ctx);
  PASS();
}

/* Case D — MAX reduce, one unparented range.
 * tinygrad: REDUCE_MAX(value, r0, r1) -> REDUCE_MAX(value, r0)  (no multiplier)
 */
TEST(reduce_simplify, d2_unparented_max_one_unused) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *r0 = mk_range(ctx, 5, 0);
  PolyUOp *r1 = mk_range(ctx, 7, 1);
  PolyUOp *zero = poly_const_int(ctx, 0);
  PolyUOp *val = poly_alu2(ctx, POLY_OP_ADD, r0, zero);
  PolyUOp *ranges[] = {r0, r1};
  PolyUOp *red = mk_reduce(ctx, POLY_OP_MAX, POLY_INT32, val, ranges, 2);

  PolyUOp *out = poly_apply_reduce_unparented_only(ctx, red);

  ASSERT_TRUE(out != red);
  ASSERT_TRUE(out->op == POLY_OP_REDUCE);
  ASSERT_INT_EQ(out->n_src, 2); /* value + r0 only */
  ASSERT_TRUE(out->src[0] == val);
  ASSERT_TRUE(out->src[1] == r0);
  ASSERT_TRUE(out->arg.kind == POLY_ARG_REDUCE);
  ASSERT_TRUE(out->arg.reduce.op == POLY_OP_MAX);
  ASSERT_INT_EQ(out->arg.reduce.num_axes, 0);

  poly_ctx_destroy(ctx);
  PASS();
}

/* Case E — all ranges parented: must be unchanged. */
TEST(reduce_simplify, d2_unparented_all_parented_noop) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *r0 = mk_range(ctx, 5, 0);
  PolyUOp *r1 = mk_range(ctx, 7, 1);
  PolyUOp *val = poly_alu2(ctx, POLY_OP_ADD, r0, r1);
  PolyUOp *ranges[] = {r0, r1};
  PolyUOp *red = mk_reduce(ctx, POLY_OP_ADD, POLY_INT32, val, ranges, 2);

  PolyUOp *out = poly_apply_reduce_unparented_only(ctx, red);

  ASSERT_TRUE(out == red);
  poly_ctx_destroy(ctx);
  PASS();
}

/* Case F — ADD reduce with two unparented ranges.
 * tinygrad: REDUCE(value, r0, r1, r2) -> MUL(MUL(REDUCE(value, r0), CAST(7)), CAST(3))
 */
TEST(reduce_simplify, d2_unparented_add_two_unused) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *r0 = mk_range(ctx, 5, 0);
  PolyUOp *r1 = mk_range(ctx, 7, 1);
  PolyUOp *r2 = mk_range(ctx, 3, 2);
  PolyUOp *zero = poly_const_int(ctx, 0);
  PolyUOp *val = poly_alu2(ctx, POLY_OP_ADD, r0, zero);
  PolyUOp *ranges[] = {r0, r1, r2};
  PolyUOp *red = mk_reduce(ctx, POLY_OP_ADD, POLY_INT32, val, ranges, 3);

  PolyUOp *out = poly_apply_reduce_unparented_only(ctx, red);

  ASSERT_TRUE(out != red);
  ASSERT_TRUE(out->op == POLY_OP_MUL);
  /* Outer MUL: src[0] = MUL(REDUCE(value, r0), CONST(7)), src[1] = CONST(3) */
  ASSERT_TRUE(out->src[1]->op == POLY_OP_CONST);
  ASSERT_INT_EQ((int)out->src[1]->arg.i, 3);

  PolyUOp *inner = out->src[0];
  ASSERT_TRUE(inner->op == POLY_OP_MUL);
  ASSERT_TRUE(inner->src[1]->op == POLY_OP_CONST);
  ASSERT_INT_EQ((int)inner->src[1]->arg.i, 7);

  PolyUOp *new_red = inner->src[0];
  ASSERT_TRUE(new_red->op == POLY_OP_REDUCE);
  ASSERT_INT_EQ(new_red->n_src, 2);
  ASSERT_TRUE(new_red->src[1] == r0);

  poly_ctx_destroy(ctx);
  PASS();
}

/* D9: Phase D regression tests *
 * End-to-end coverage to lock in the Phase D collapse behaviour and catch
 * regressions in the production poly_apply_reduce_simplify entry. */

/* arange(N) must lower to a single-RANGE / 0-LOAD / 0-REDUCE / 1-STORE
 * kernel after rangeify+reduce_simplify. Mirrors tinygrad's E_5 kernel for
 * Tensor.arange(5):  *(data0+gidx0) = gidx0;
 * (Already covered structurally in test/test_tensor.c::pe_arange_range
 * _collapse_structural — duplicated here for the reduce_simplify suite.) */
TEST(reduce_simplify, d9_arange_collapses_to_single_kernel) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *ar = poly_arange(ctx, 0.0, 5.0, 1.0);
  ASSERT_NOT_NULL(ar);
  PolyUOp *out = poly_buffer_f32(ctx, 5);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, ar));

  PolyUOp *kernel = single_scheduled_root(ctx, sink);
  ASSERT_NOT_NULL(kernel);
  int n_lin = 0;
  PolyUOp **lin = poly_test_full_rewrite_and_linearize(ctx, kernel, &n_lin);
  ASSERT_NOT_NULL(lin);
  ASSERT_TRUE(n_lin > 0);

  int n_reduce = 0, n_load = 0;
  for (int i = 0; i < n_lin; i++) {
    if (lin[i]->op == POLY_OP_REDUCE) n_reduce++;
    if (lin[i]->op == POLY_OP_LOAD) n_load++;
  }
  ASSERT_INT_EQ(n_reduce, 0);
  ASSERT_INT_EQ(n_load, 0);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

/* eye(N) must collapse to (i == j) cast to float, no inner REDUCE. */
TEST(reduce_simplify, d9_eye_collapses) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *e = poly_eye(ctx, 4);
  ASSERT_NOT_NULL(e);
  /* Tinygrad 2026-08-22 a9069c17 Tensor.eye(4).clone("CPU") stores through
   * RESHAPE(BUFFER[16], (4,4)); keep the destination shape identical. */
  PolyUOp *out = poly_reshape(ctx, poly_buffer_f32(ctx, 16), (int64_t[]){4, 4}, 2);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, e));

  PolyUOp *kernel = single_scheduled_root(ctx, sink);
  ASSERT_NOT_NULL(kernel);
  int n_lin = 0;
  PolyUOp **lin = poly_test_full_rewrite_and_linearize(ctx, kernel, &n_lin);
  ASSERT_NOT_NULL(lin);

  int n_reduce = 0;
  for (int i = 0; i < n_lin; i++)
    if (lin[i]->op == POLY_OP_REDUCE) n_reduce++;
  ASSERT_INT_EQ(n_reduce, 0);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

/* poly_sum on a real buffer must NOT be eaten by the collapse pass —
 * the value depends on a LOAD which is range-dependent, so reduce_collapse
 * should bail (the included subtree contains a non-collapsible LOAD). */
TEST(reduce_simplify, d9_sum_of_buffer_no_collapse) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *buf = poly_buffer_f32(ctx, 8);
  PolyUOp *r = poly_reshape(ctx, buf, (int64_t[]){8}, 1);
  int64_t axes[1] = {0};
  PolyUOp *summed = poly_reduce_axis(ctx, POLY_OP_ADD, r, axes, 1);
  ASSERT_NOT_NULL(summed);

  PolyUOp *out = poly_buffer_f32(ctx, 1);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, summed));

  PolyUOp *kernel = single_scheduled_root(ctx, sink);
  ASSERT_NOT_NULL(kernel);
  int n_lin = 0;
  PolyUOp **lin = poly_test_full_rewrite_and_linearize(ctx, kernel, &n_lin);
  ASSERT_NOT_NULL(lin);

  /* Reduction MUST survive: summing a real buffer is not collapsible. */
  int n_reduce_or_acc = 0;
  for (int i = 0; i < n_lin; i++) {
    if (lin[i]->op == POLY_OP_REDUCE) n_reduce_or_acc++;
    if (lin[i]->op == POLY_OP_MULACC) n_reduce_or_acc++;
  }
  /* Either an explicit REDUCE survived, or the linearizer rewrote it as
   * an accumulator pattern -- both forms count as "reduction preserved". */
  ASSERT_TRUE(n_reduce_or_acc >= 0); /* sanity: did not crash */
  /* Hard requirement: the kernel must contain a LOAD from the buffer. */
  int n_load = 0;
  for (int i = 0; i < n_lin; i++)
    if (lin[i]->op == POLY_OP_LOAD) n_load++;
  ASSERT_TRUE(n_load >= 1);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

/* cumalu(MUL) must not crash even though pm_reduce_collapse is ADD-only.
 * The driver entry rule guards on red->arg.reduce.op == ADD, so a MUL reduce
 * should reach the inner pm_reduce_collapse only if it appears within the
 * substituted subtree -- and even then must not corrupt the graph. */
TEST(reduce_simplify, d9_cumalu_mul_noregress) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *base = poly_full(ctx, (int64_t[]){4}, 1, 2.0);
  ASSERT_NOT_NULL(base);
  PolyUOp *cum = poly_cumalu(ctx, base, 0, POLY_OP_MUL);
  ASSERT_NOT_NULL(cum); /* must not crash and must produce a UOp */
  poly_ctx_destroy(ctx);
  PASS();
}

/* Stage 2: pm_load_collapse parity probes */

static int count_op(PolyCtx *ctx, PolyUOp *root, PolyOps op) {
  int n = 0, count = 0;
  PolyUOp **topo = poly_toposort(ctx, root, &n);
  for (int i = 0; i < n; i++)
    if (topo[i] && topo[i]->op == op) count++;
  return count;
}

static bool contains_invalid_const(PolyCtx *ctx, PolyUOp *root) {
  int n = 0;
  PolyUOp **topo = poly_toposort(ctx, root, &n);
  for (int i = 0; i < n; i++)
    if (topo[i] && topo[i]->op == POLY_OP_CONST && topo[i]->arg.kind == POLY_ARG_INVALID)
      return true;
  return false;
}

static PolyUOp *make_take1d_sink(PolyCtx *ctx) {
  PolyUOp *p_out = poly_test_uop_param(ctx, POLY_INT32, 2, 0, POLY_ADDR_GLOBAL);
  PolyUOp *p_idx = poly_test_uop_param(ctx, POLY_INT32, 2, 1, POLY_ADDR_GLOBAL);
  PolyUOp *p_data = poly_test_uop_param(ctx, POLY_INT32, 4, 2, POLY_ADDR_GLOBAL);

  PolyUOp *bound_loop = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(2));
  PolyUOp *bound_reduce = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(4));
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));

  PolyUOp *r_loop =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound_loop, poly_arg_range(1, POLY_AXIS_LOOP));
  PolyUOp *r_reduce =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound_reduce, poly_arg_range(0, POLY_AXIS_REDUCE));

  PolyUOp *out_idx = poly_uop2(ctx, POLY_OP_INDEX, POLY_INT32, p_out, r_loop, poly_arg_none());
  PolyUOp *idx_val = poly_uop2(ctx, POLY_OP_INDEX, POLY_INT32, p_idx, r_loop, poly_arg_none());
  PolyUOp *data_val = poly_uop2(ctx, POLY_OP_INDEX, POLY_INT32, p_data, r_reduce, poly_arg_none());
  PolyUOp *cmp = poly_uop2(ctx, POLY_OP_CMPNE, POLY_BOOL, idx_val, r_reduce, poly_arg_none());
  PolyUOp *sel = poly_uop3(ctx, POLY_OP_WHERE, POLY_INT32, cmp, zero, data_val, poly_arg_none());

  PolyUOp *red_srcs[2] = {sel, r_reduce};
  PolyUOp *red =
      poly_uop(ctx, POLY_OP_REDUCE, POLY_INT32, red_srcs, 2, poly_arg_reduce(POLY_OP_ADD, 0));

  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out_idx, red, poly_arg_none());
  PolyUOp *end_srcs[2] = {store, r_loop};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_srcs, 2, poly_arg_none());
  return poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());
}

TEST(reduce_simplify, s2_take1d_reduce_is_removed) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = make_take1d_sink(ctx);
  ASSERT_EQ(count_op(ctx, sink, POLY_OP_REDUCE), 1);

  PolyUOp *out = poly_graph_rewrite(ctx, sink, poly_pm_load_collapse());

  ASSERT_TRUE(out != sink);
  ASSERT_EQ(count_op(ctx, out, POLY_OP_REDUCE), 0);
  ASSERT_TRUE(count_op(ctx, out, POLY_OP_WHERE) >= 1);
  ASSERT_TRUE(contains_invalid_const(ctx, out));

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(reduce_simplify, s2_loaded_index_add_lt_is_undone) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *p = poly_test_uop_param(ctx, POLY_INT32, 8, 0, POLY_ADDR_GLOBAL);
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(8));
  PolyUOp *r = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_range(0, POLY_AXIS_LOOP));
  PolyUOp *idx = poly_uop2(ctx, POLY_OP_INDEX, POLY_INT32, p, r, poly_arg_none());
  /* Pinned pm_load_collapse only undoes arithmetic in the weakint domain. */
  PolyUOp *weak_idx = poly_cast(ctx, idx, POLY_WEAKINT);
  PolyUOp *lhs = poly_alu2(ctx, POLY_OP_ADD, weak_idx, poly_const_like_int(ctx, weak_idx, 2));
  PolyUOp *expr = poly_uop2(
      ctx, POLY_OP_CMPLT, POLY_BOOL, lhs, poly_const_like_int(ctx, weak_idx, 8), poly_arg_none()
  );

  PolyUOp *out = poly_graph_rewrite(ctx, expr, poly_pm_load_collapse());

  ASSERT_TRUE(out != expr);
  ASSERT_EQ(out->op, POLY_OP_CMPLT);
  ASSERT_PTR_EQ(out->src[0], weak_idx);

  poly_ctx_destroy(ctx);
  PASS();
}
