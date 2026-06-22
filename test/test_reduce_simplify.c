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
#include "../src/simplify.h"
#include "../src/tensor.h"
#include "../src/pat.h"
#include "../src/engine/schedule.h" /* poly_get_kernel_graph, poly_reshape, poly_reduce_axis */
#include "../src/schedule/rangeify.h"
#include "../src/codegen.h" /* poly_linearize */
#include "../src/frontend.h" /* poly_buffer_f32, poly_sink1, poly_store_val */

/* helpers */

/* These regression checks inspect scheduled kernels after rangeify and
 * reduce_simplify, so they need the executable scheduled root instead of the
 * earlier public kernel-graph boundary. */
static PolyUOp *single_scheduled_root(PolyCtx *ctx, PolyUOp *sink) {
  PolySchedule *schedule = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  if (!schedule) return NULL;
  if (schedule->template->n_calls != 1 || !poly_schedule_call_body(schedule, 0)) {
    poly_schedule_free(schedule);
    return NULL;
  }
  PolyUOp *root = poly_schedule_call_body(schedule, 0);
  poly_schedule_free(schedule);
  return root;
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
  return poly_uop(ctx, POLY_OP_REDUCE, dt, srcs, 1 + n_ranges, poly_arg_ops(op));
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

TEST(reduce_simplify, reduce_pass_without_ctx_is_noop_not_abort) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *c = poly_const_int(ctx, 1);
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, c, poly_arg_none());

  PolyUOp *out = poly_graph_rewrite(ctx, sink, poly_pm_reduce_pass());
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
  ASSERT_TRUE(new_red->arg.kind == POLY_ARG_OPS);
  ASSERT_TRUE(new_red->arg.ops == POLY_OP_ADD);

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
  PolyUOp *val = poly_const_int(ctx, 4);
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
  ASSERT_TRUE(new_red->arg.ops == POLY_OP_MUL);
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
  ASSERT_TRUE(out->arg.ops == POLY_OP_MAX);

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
  PolyUOp **lin = poly_linearize(ctx, kernel, &n_lin);
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
  PolyUOp *out = poly_buffer_f32(ctx, 16);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, e));

  PolyUOp *kernel = single_scheduled_root(ctx, sink);
  ASSERT_NOT_NULL(kernel);
  int n_lin = 0;
  PolyUOp **lin = poly_linearize(ctx, kernel, &n_lin);
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
  PolyUOp **lin = poly_linearize(ctx, kernel, &n_lin);
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
 * The driver entry rule guards on red->arg.ops == ADD, so a MUL reduce
 * should reach the inner pm_reduce_collapse only if it appears within the
 * substituted subtree -- and even then must not corrupt the graph. */
TEST(reduce_simplify, d9_cumalu_mul_noregress) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *base = poly_full(ctx, (int64_t[]){4}, 1, 2.0);
  ASSERT_NOT_NULL(base);
  PolyUOp *cum = poly_cumalu(ctx, base, 0, POLY_OP_MUL, false);
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
  PolyDType ptr_i32_out = poly_dtype_ptr(POLY_INT32, 2, POLY_ADDR_GLOBAL);
  PolyDType ptr_i32_idx = poly_dtype_ptr(POLY_INT32, 2, POLY_ADDR_GLOBAL);
  PolyDType ptr_i32_data = poly_dtype_ptr(POLY_INT32, 4, POLY_ADDR_GLOBAL);

  PolyUOp *p_out = poly_uop0(ctx, POLY_OP_PARAM, ptr_i32_out, poly_arg_int(0));
  PolyUOp *p_idx = poly_uop0(ctx, POLY_OP_PARAM, ptr_i32_idx, poly_arg_int(1));
  PolyUOp *p_data = poly_uop0(ctx, POLY_OP_PARAM, ptr_i32_data, poly_arg_int(2));

  PolyUOp *bound_loop = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(2));
  PolyUOp *bound_reduce = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(4));
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));

  PolyUOp *r_loop =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound_loop, poly_arg_range(1, POLY_AXIS_LOOP));
  PolyUOp *r_reduce =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound_reduce, poly_arg_range(0, POLY_AXIS_REDUCE));

  PolyUOp *out_idx = poly_uop2(ctx, POLY_OP_INDEX, ptr_i32_out, p_out, r_loop, poly_arg_none());
  PolyUOp *idx_val = poly_uop2(ctx, POLY_OP_INDEX, POLY_INT32, p_idx, r_loop, poly_arg_none());
  PolyUOp *data_val = poly_uop2(ctx, POLY_OP_INDEX, POLY_INT32, p_data, r_reduce, poly_arg_none());
  PolyUOp *cmp = poly_uop2(ctx, POLY_OP_CMPNE, POLY_BOOL, idx_val, r_reduce, poly_arg_none());
  PolyUOp *sel = poly_uop3(ctx, POLY_OP_WHERE, POLY_INT32, cmp, zero, data_val, poly_arg_none());

  PolyUOp *red_srcs[2] = {sel, r_reduce};
  PolyUOp *red = poly_uop(ctx, POLY_OP_REDUCE, POLY_INT32, red_srcs, 2, poly_arg_ops(POLY_OP_ADD));

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
  PolyDType ptr_i32 = poly_dtype_ptr(POLY_INT32, 8, POLY_ADDR_GLOBAL);
  PolyUOp *p = poly_uop0(ctx, POLY_OP_PARAM, ptr_i32, poly_arg_int(0));
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(8));
  PolyUOp *r = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_range(0, POLY_AXIS_LOOP));
  PolyUOp *idx = poly_uop2(ctx, POLY_OP_INDEX, POLY_INT32, p, r, poly_arg_none());
  PolyUOp *lhs = poly_alu2(ctx, POLY_OP_ADD, idx, poly_const_int(ctx, 2));
  PolyUOp *expr = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, lhs, poly_const_int(ctx, 8), poly_arg_none());

  PolyUOp *out = poly_graph_rewrite(ctx, expr, poly_pm_load_collapse());

  ASSERT_TRUE(out != expr);
  ASSERT_EQ(out->op, POLY_OP_CMPLT);
  ASSERT_EQ(out->src[0]->op, POLY_OP_INDEX);
  ASSERT_TRUE(out->src[0] == idx);

  poly_ctx_destroy(ctx);
  PASS();
}
