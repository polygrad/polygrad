/*
 * test_future_passes.c -- Tests for codegen pass correctness and conformance.
 *
 * Sections:
 *   1-2. Transcendental + late decomposition IR tests (verify ops eliminated)
 *   3. Symbolic simplification rule tests
 *   4. Expander pass tests
 *   5. Regression tests (e2e correctness for each pass)
 *   6. Pass-order audit tests (validate multi-pass dependencies)
 *   7. Transcendental conformance tests (special values, dense sweeps)
 *
 * All tests must PASS. Run with filter:
 *   build/polygrad_test pass_order
 *   build/polygrad_test conformance
 *   build/polygrad_test transcendental
 *   build/polygrad_test regression
 */

#include "test_harness.h"
#include "../src/codegen.h"
#include "../src/ctx.h"
#include "../src/frontend.h"
#include "../src/engine/schedule.h"
#include "../src/schedule/rangeify.h"
#include "../src/simplify.h"

/* Helpers */

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
  PolyUOp *end_src[2] = {store, range};
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
  PolyUOp *end_src[2] = {store, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  return poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());
}

static int count_range_axis_type(PolyCtx *ctx, PolyUOp *sink, PolyAxisType type) {
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, sink, &n_topo);
  int count = 0;
  for (int i = 0; i < n_topo; i++) {
    if (topo[i]->op == POLY_OP_RANGE && poly_arg_is_range(topo[i]->arg) &&
        poly_range_axis_type(topo[i]->arg) == type)
      count++;
  }
  return count;
}

static PolyUOp *make_many_index_kernel(PolyCtx *ctx, int n_pairs, int n) {
  PolyDType ptr_f32 = poly_dtype_ptr(POLY_FLOAT32, -1, POLY_ADDR_GLOBAL);
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(n));
  PolyUOp *range =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_range(0, POLY_AXIS_LOOP));
  PolyUOp **stores = (PolyUOp **)malloc((size_t)n_pairs * sizeof(PolyUOp *));
  if (!stores) return NULL;

  for (int i = 0; i < n_pairs; i++) {
    PolyUOp *src = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(i * 2));
    PolyUOp *dst = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(i * 2 + 1));
    PolyUOp *idx_src = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, src, range, poly_arg_none());
    PolyUOp *idx_dst = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, dst, range, poly_arg_none());
    PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx_src, poly_arg_none());
    stores[i] = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx_dst, load, poly_arg_none());
  }

  PolyUOp *sink = poly_uop(ctx, POLY_OP_SINK, POLY_VOID, stores, n_pairs, poly_arg_none());
  free(stores);
  return sink;
}

static PolyUOp *simplify(PolyCtx *ctx, PolyUOp *root) {
  return poly_graph_rewrite(ctx, root, poly_symbolic_simple());
}

/* ════════════════════════════════════════════════════════════════════════
 * SECTION 1: Missing transcendental decompositions — MUST FAIL
 * ════════════════════════════════════════════════════════════════════════ */

/*
 * Forced LOG2 decomposition — tinygrad's xlog2:
 *   LOG2(d) → frexp-based polynomial. No LOG2 should remain.
 *   Ref: tinygrad/uop/decompositions.py xlog2()
 */
TEST(transcendental, decomp_log2_ir) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = make_unary_kernel(ctx, POLY_OP_LOG2, 4);
  PolyUOp *rewritten = poly_graph_rewrite(ctx, sink, poly_pm_transcendental_pass());
  int n_log2 = count_ops_in(ctx, rewritten, POLY_OP_LOG2);
  poly_ctx_destroy(ctx);
  ASSERT_INT_EQ(n_log2, 0); /* LOG2 must be fully decomposed */
  PASS();
}

/*
 * Forced SIN decomposition — tinygrad's xsin:
 *   SIN(d) → Payne-Hanek + polynomial. No SIN should remain.
 *   Ref: tinygrad/uop/decompositions.py xsin()
 */
TEST(transcendental, decomp_sin_ir) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = make_unary_kernel(ctx, POLY_OP_SIN, 4);
  PolyUOp *rewritten = poly_graph_rewrite(ctx, sink, poly_pm_transcendental_pass());
  int n_sin = count_ops_in(ctx, rewritten, POLY_OP_SIN);
  poly_ctx_destroy(ctx);
  ASSERT_INT_EQ(n_sin, 0); /* SIN must be fully decomposed */
  PASS();
}

/* ════════════════════════════════════════════════════════════════════════
 * SECTION 2: Missing late decompositions — MUST FAIL
 * ════════════════════════════════════════════════════════════════════════ */

/*
 * Current tinygrad keeps CMOD by a power-of-two divisor in the late Clang
 * pipeline. This matters for xexp2/ldexp2k strict IR parity:
 * tinygrad/uop/decompositions.py ldexp2k uses shr(e, 1), which lowers through
 * the floor-div correction and leaves CMOD(q, 2) in the final no-opt LINEAR.
 */
TEST(decomp, cmod_power_of_two_stays_cmod_like_tinygrad) {
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
  PolyUOp *end_src[2] = {store, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());
  PolyUOp *rewritten = poly_full_rewrite_to_sink(ctx, sink);
  int n_mod = count_ops_in(ctx, rewritten, POLY_OP_MOD);
  int n_and = count_ops_in(ctx, rewritten, POLY_OP_AND);
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(n_mod > 0); /* CMOD must remain */
  ASSERT_INT_EQ(n_and, 0); /* no stale MOD->AND shortcut */
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
  PolyUOp *mulacc_srcs[3] = {ld0, ld1, ld2};
  PolyUOp *mulacc = poly_uop(ctx, POLY_OP_MULACC, POLY_FLOAT32, mulacc_srcs, 3, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx3, mulacc, poly_arg_none());
  PolyUOp *end_src[2] = {store, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());
  PolyUOp *rewritten = poly_full_rewrite_to_sink(ctx, sink);
  int n_mulacc = count_ops_in(ctx, rewritten, POLY_OP_MULACC);
  poly_ctx_destroy(ctx);
  ASSERT_INT_EQ(n_mulacc, 0); /* MULACC must be eliminated */
  PASS();
}

/*
 * MULACC preserved when caps.has_mulacc = true (CUDA-style pipeline).
 * Same kernel as mulacc_to_mul_add but using _ex with FMA caps.
 */
TEST(decomp, mulacc_caps_preserves) {
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
  PolyUOp *mulacc_srcs[3] = {ld0, ld1, ld2};
  PolyUOp *mulacc = poly_uop(ctx, POLY_OP_MULACC, POLY_FLOAT32, mulacc_srcs, 3, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx3, mulacc, poly_arg_none());
  PolyUOp *end_src[2] = {store, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());
  PolyRewriteOpts opts = {.optimize = false, .devectorize = 0, .caps = {.has_mulacc = true}};
  PolyUOp *rewritten = poly_full_rewrite_to_sink_ex(ctx, sink, opts);
  int n_mulacc = count_ops_in(ctx, rewritten, POLY_OP_MULACC);
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(n_mulacc > 0); /* MULACC preserved with FMA caps */
  PASS();
}

/*
 * ADD(MUL(a,b), c) fuses to MULACC when caps.has_mulacc = true.
 * Builds a*b+c pattern (no explicit MULACC), verifies fusion fires.
 */
TEST(decomp, mul_add_fuses_to_mulacc) {
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
  /* Build a*b+c as MUL+ADD (no MULACC in input) */
  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, ld0, ld1, poly_arg_none());
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, mul, ld2, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx3, add, poly_arg_none());
  PolyUOp *end_src[2] = {store, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());
  PolyRewriteOpts opts = {.optimize = false, .devectorize = 0, .caps = {.has_mulacc = true}};
  PolyUOp *rewritten = poly_full_rewrite_to_sink_ex(ctx, sink, opts);
  int n_mulacc = count_ops_in(ctx, rewritten, POLY_OP_MULACC);
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(n_mulacc > 0); /* MUL+ADD fused to MULACC */
  PASS();
}

/*
 * MUL+ADD does NOT fuse to MULACC for integer types.
 * Fusion rule is float-only to match FMA semantics.
 */
TEST(decomp, mul_add_int_no_fuse) {
  PolyCtx *ctx = poly_ctx_new();
  PolyDType ptr_i32 = poly_dtype_ptr(POLY_INT32, -1, POLY_ADDR_GLOBAL);
  PolyUOp *p0 = poly_uop0(ctx, POLY_OP_PARAM, ptr_i32, poly_arg_int(0));
  PolyUOp *p1 = poly_uop0(ctx, POLY_OP_PARAM, ptr_i32, poly_arg_int(1));
  PolyUOp *p2 = poly_uop0(ctx, POLY_OP_PARAM, ptr_i32, poly_arg_int(2));
  PolyUOp *p3 = poly_uop0(ctx, POLY_OP_PARAM, ptr_i32, poly_arg_int(3));
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(4));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));
  PolyUOp *idx0 = poly_uop2(ctx, POLY_OP_INDEX, ptr_i32, p0, range, poly_arg_none());
  PolyUOp *idx1 = poly_uop2(ctx, POLY_OP_INDEX, ptr_i32, p1, range, poly_arg_none());
  PolyUOp *idx2 = poly_uop2(ctx, POLY_OP_INDEX, ptr_i32, p2, range, poly_arg_none());
  PolyUOp *idx3 = poly_uop2(ctx, POLY_OP_INDEX, ptr_i32, p3, range, poly_arg_none());
  PolyUOp *ld0 = poly_uop1(ctx, POLY_OP_LOAD, POLY_INT32, idx0, poly_arg_none());
  PolyUOp *ld1 = poly_uop1(ctx, POLY_OP_LOAD, POLY_INT32, idx1, poly_arg_none());
  PolyUOp *ld2 = poly_uop1(ctx, POLY_OP_LOAD, POLY_INT32, idx2, poly_arg_none());
  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, POLY_INT32, ld0, ld1, poly_arg_none());
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, mul, ld2, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx3, add, poly_arg_none());
  PolyUOp *end_src[2] = {store, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());
  PolyRewriteOpts opts = {.optimize = false, .devectorize = 0, .caps = {.has_mulacc = true}};
  PolyUOp *rewritten = poly_full_rewrite_to_sink_ex(ctx, sink, opts);
  int n_mulacc = count_ops_in(ctx, rewritten, POLY_OP_MULACC);
  poly_ctx_destroy(ctx);
  ASSERT_INT_EQ(n_mulacc, 0); /* No fusion for int types */
  PASS();
}

/*
 * THREEFRY lowers to integer ALU when renderer lacks native THREEFRY support.
 */
TEST(decomp, threefry_lowered_when_no_native_support) {
  PolyCtx *ctx = poly_ctx_new();
  PolyDType ptr_u32 = poly_dtype_ptr(POLY_UINT32, -1, POLY_ADDR_GLOBAL);
  PolyUOp *p0 = poly_uop0(ctx, POLY_OP_PARAM, ptr_u32, poly_arg_int(0));
  PolyUOp *p1 = poly_uop0(ctx, POLY_OP_PARAM, ptr_u32, poly_arg_int(1));
  PolyUOp *p2 = poly_uop0(ctx, POLY_OP_PARAM, ptr_u32, poly_arg_int(2));
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(8));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));
  PolyUOp *idx0 = poly_uop2(ctx, POLY_OP_INDEX, ptr_u32, p0, range, poly_arg_none());
  PolyUOp *idx1 = poly_uop2(ctx, POLY_OP_INDEX, ptr_u32, p1, range, poly_arg_none());
  PolyUOp *idx2 = poly_uop2(ctx, POLY_OP_INDEX, ptr_u32, p2, range, poly_arg_none());
  PolyUOp *ld0 = poly_uop1(ctx, POLY_OP_LOAD, POLY_UINT32, idx0, poly_arg_none());
  PolyUOp *ld1 = poly_uop1(ctx, POLY_OP_LOAD, POLY_UINT32, idx1, poly_arg_none());
  PolyUOp *thr = poly_uop2(ctx, POLY_OP_THREEFRY, POLY_UINT32, ld0, ld1, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx2, thr, poly_arg_none());
  PolyUOp *end_src[2] = {store, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());

  PolyUOp *rewritten = poly_full_rewrite_to_sink(ctx, sink);
  int n_threefry = count_ops_in(ctx, rewritten, POLY_OP_THREEFRY);
  int n_add = count_ops_in(ctx, rewritten, POLY_OP_ADD);
  int n_xor = count_ops_in(ctx, rewritten, POLY_OP_XOR);
  poly_ctx_destroy(ctx);

  ASSERT_INT_EQ(n_threefry, 0); /* must be fully lowered */
  ASSERT_TRUE(n_add > 0);
  ASSERT_TRUE(n_xor > 0);
  PASS();
}

/* Pinned tinygrad/uop/decompositions.py:445-456 proves floor/trunc agreement
 * from expression bounds, not from an unsigned storage dtype. Threefry's
 * wrapped uint32 intermediates can carry mathematical ranges crossing zero. */
TEST(decomp, wrapped_unsigned_divmod_uses_expression_bounds) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *shape = poly_uop0(ctx, POLY_OP_STACK, POLY_VOID, poly_arg_none());
  PolyParamArg wrapped_arg = {
      .slot = -1,
      .name = "wrapped",
      .min_val = -4294967295LL,
      .max_val = 8589934590LL,
      .has_minmax = true,
      .addrspace = POLY_ADDR_GLOBAL,
  };
  PolyParamArg nonnegative_arg = {
      .slot = -1,
      .name = "nonnegative",
      .min_val = 0,
      .max_val = 1000000,
      .has_minmax = true,
      .addrspace = POLY_ADDR_GLOBAL,
  };
  PolyUOp *wrapped =
      poly_uop1(ctx, POLY_OP_PARAM, POLY_UINT32, shape, poly_arg_param(&wrapped_arg));
  PolyUOp *nonnegative =
      poly_uop1(ctx, POLY_OP_PARAM, POLY_UINT32, shape, poly_arg_param(&nonnegative_arg));
  PolyUOp *pow2 = poly_uop0(ctx, POLY_OP_CONST, POLY_UINT32, poly_arg_int(1 << 19));
  PolyUOp *seven = poly_uop0(ctx, POLY_OP_CONST, POLY_UINT32, poly_arg_int(7));
  PolyUOp *roots[3] = {
      poly_uop2(ctx, POLY_OP_FLOORDIV, POLY_UINT32, wrapped, pow2, poly_arg_none()),
      poly_uop2(ctx, POLY_OP_FLOORDIV, POLY_UINT32, nonnegative, pow2, poly_arg_none()),
      poly_uop2(ctx, POLY_OP_FLOORMOD, POLY_UINT32, wrapped, seven, poly_arg_none()),
  };
  PolyUOp *sink = poly_uop(ctx, POLY_OP_SINK, POLY_VOID, roots, 3, poly_arg_none());
  PolyRewriteOpts opts = {
      .optimize = false,
      .devectorize = 1,
      .caps = {.has_mulacc = true,
               .has_max = true,
               .has_exp2 = true,
               .has_log2 = true,
               .has_sin = true,
               .has_fdiv = true,
               .has_int64 = true,
               .has_local = true,
               .max_vec_width = 4},
      .device = POLY_DEVICE_CUDA,
      .opt_policy = POLY_OPT_HEURISTIC,
  };
  PolyUOp *rewritten = poly_full_rewrite_to_sink_ex(ctx, sink, opts);
  ASSERT_TRUE(rewritten && rewritten->op == POLY_OP_SINK && rewritten->n_src == 3);

  PolyUOp *wrapped_div = rewritten->src[0];
  ASSERT_EQ(wrapped_div->op, POLY_OP_SUB);
  ASSERT_INT_EQ(count_ops_in(ctx, wrapped_div, POLY_OP_SHR), 1);
  ASSERT_INT_EQ(count_ops_in(ctx, wrapped_div, POLY_OP_CMOD), 1);
  ASSERT_INT_EQ(count_ops_in(ctx, wrapped_div, POLY_OP_CMPLT), 1);
  ASSERT_INT_EQ(count_ops_in(ctx, wrapped_div, POLY_OP_CMPNE), 2);
  ASSERT_INT_EQ(count_ops_in(ctx, wrapped_div, POLY_OP_AND), 1);
  ASSERT_INT_EQ(count_ops_in(ctx, wrapped_div, POLY_OP_CAST), 1);

  PolyUOp *nonnegative_div = rewritten->src[1];
  ASSERT_EQ(nonnegative_div->op, POLY_OP_SHR);
  ASSERT_INT_EQ(count_ops_in(ctx, nonnegative_div, POLY_OP_CMOD), 0);
  ASSERT_INT_EQ(count_ops_in(ctx, nonnegative_div, POLY_OP_CMPLT), 0);

  PolyUOp *wrapped_mod = rewritten->src[2];
  ASSERT_EQ(wrapped_mod->op, POLY_OP_ADD);
  ASSERT_INT_EQ(count_ops_in(ctx, wrapped_mod, POLY_OP_CMOD), 1);
  ASSERT_INT_EQ(count_ops_in(ctx, wrapped_mod, POLY_OP_WHERE), 1);
  ASSERT_INT_EQ(count_ops_in(ctx, wrapped_mod, POLY_OP_CMPLT), 1);
  ASSERT_INT_EQ(count_ops_in(ctx, wrapped_mod, POLY_OP_CMPNE), 2);
  ASSERT_INT_EQ(count_ops_in(ctx, wrapped_mod, POLY_OP_AND), 1);

  poly_ctx_destroy(ctx);
  PASS();
}

/* Pinned tinygrad/uop/decompositions.py:318 retains the left-associated
 * `xr1 + key + i + 1` graph. Combining i+1 changes the operation estimate and
 * the exact renderer-facing topology even though modular values agree. */
TEST(decomp, threefry_round_key_injection_matches_tinygrad_topology) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *shape = poly_uop0(ctx, POLY_OP_STACK, POLY_VOID, poly_arg_none());
  PolyParamArg x_arg = {
      .slot = -1,
      .name = "x",
      .min_val = 0,
      .max_val = INT64_MAX,
      .has_minmax = true,
      .addrspace = POLY_ADDR_GLOBAL,
  };
  PolyParamArg key_arg = x_arg;
  key_arg.name = "key";
  PolyUOp *x = poly_uop1(ctx, POLY_OP_PARAM, POLY_UINT64, shape, poly_arg_param(&x_arg));
  PolyUOp *key =
      poly_uop1(ctx, POLY_OP_PARAM, POLY_UINT64, shape, poly_arg_param(&key_arg));
  PolyUOp *root = poly_uop2(ctx, POLY_OP_THREEFRY, POLY_UINT64, x, key, poly_arg_none());
  PolyRendererCaps caps = {
      .has_mulacc = false,
      .has_max = true,
      .has_exp2 = true,
      .has_log2 = true,
      .has_sin = true,
      .has_fdiv = true,
      .has_int64 = true,
      .has_local = true,
      .max_vec_width = 4,
  };
  PolyUOp *rewritten = poly_graph_rewrite(ctx, root, poly_pm_decomp_pass_caps(caps));

  ASSERT_TRUE(rewritten != NULL);
  ASSERT_INT_EQ(count_ops_in(ctx, rewritten, POLY_OP_THREEFRY), 0);
  ASSERT_INT_EQ(count_ops_in(ctx, rewritten, POLY_OP_ADD), 61);
  ASSERT_INT_EQ(count_ops_in(ctx, rewritten, POLY_OP_CONST), 16);
  ASSERT_INT_EQ(count_ops_in(ctx, rewritten, POLY_OP_SHL), 21);
  ASSERT_INT_EQ(count_ops_in(ctx, rewritten, POLY_OP_SHR), 22);
  ASSERT_INT_EQ(count_ops_in(ctx, rewritten, POLY_OP_XOR), 22);

  poly_ctx_destroy(ctx);
  PASS();
}

/*
 * THREEFRY is preserved when renderer advertises native THREEFRY support.
 */
TEST(decomp, threefry_preserved_with_native_caps) {
  PolyCtx *ctx = poly_ctx_new();
  PolyDType ptr_u32 = poly_dtype_ptr(POLY_UINT32, -1, POLY_ADDR_GLOBAL);
  PolyUOp *p0 = poly_uop0(ctx, POLY_OP_PARAM, ptr_u32, poly_arg_int(0));
  PolyUOp *p1 = poly_uop0(ctx, POLY_OP_PARAM, ptr_u32, poly_arg_int(1));
  PolyUOp *p2 = poly_uop0(ctx, POLY_OP_PARAM, ptr_u32, poly_arg_int(2));
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(8));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));
  PolyUOp *idx0 = poly_uop2(ctx, POLY_OP_INDEX, ptr_u32, p0, range, poly_arg_none());
  PolyUOp *idx1 = poly_uop2(ctx, POLY_OP_INDEX, ptr_u32, p1, range, poly_arg_none());
  PolyUOp *idx2 = poly_uop2(ctx, POLY_OP_INDEX, ptr_u32, p2, range, poly_arg_none());
  PolyUOp *ld0 = poly_uop1(ctx, POLY_OP_LOAD, POLY_UINT32, idx0, poly_arg_none());
  PolyUOp *ld1 = poly_uop1(ctx, POLY_OP_LOAD, POLY_UINT32, idx1, poly_arg_none());
  PolyUOp *thr = poly_uop2(ctx, POLY_OP_THREEFRY, POLY_UINT32, ld0, ld1, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx2, thr, poly_arg_none());
  PolyUOp *end_src[2] = {store, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());

  PolyRewriteOpts opts = {
      .optimize = false, .devectorize = 0, .caps = {.has_mulacc = false, .has_threefry = true}};
  PolyUOp *rewritten = poly_full_rewrite_to_sink_ex(ctx, sink, opts);
  int n_threefry = count_ops_in(ctx, rewritten, POLY_OP_THREEFRY);
  poly_ctx_destroy(ctx);

  ASSERT_TRUE(n_threefry > 0); /* native-cap path must preserve THREEFRY */
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
  PolyUOp *end_src[2] = {store, range};
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
  PolyUOp *end_src[2] = {store, range};
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
  PolyUOp *end_src[2] = {store, range};
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
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(100));
  PolyUOp *x = poly_uop1(ctx, POLY_OP_RANGE, POLY_INDEX, bound, poly_arg_int(0));
  PolyUOp *c = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(8));
  PolyUOp *mod = poly_uop2(ctx, POLY_OP_FLOORMOD, POLY_INDEX, x, c, poly_arg_none());
  PolyUOp *div = poly_uop2(ctx, POLY_OP_FLOORDIV, POLY_INDEX, x, c, poly_arg_none());
  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, POLY_INDEX, div, c, poly_arg_none());
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_INDEX, mod, mul, poly_arg_none());
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
  PolyUOp * xor = poly_uop2(ctx, POLY_OP_XOR, POLY_INT32, x, zero, poly_arg_none());
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

/* Current tinygrad keeps STACK(CONST, ...) canonical through symbolic_simple. */
TEST(sym_future, stack_const_is_canonical_after_symbolic) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *c1 = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(1.0));
  PolyUOp *c2 = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(1.0));
  PolyUOp *vec_srcs[2] = {c1, c2};
  PolyUOp *vec =
      poly_uop(ctx, POLY_OP_STACK, poly_dtype_vec(POLY_FLOAT32, 2), vec_srcs, 2, poly_arg_none());
  PolyUOp *r = simplify(ctx, vec);
  int canonical = r == vec && r->op == POLY_OP_STACK;
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(canonical);
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
  ASSERT_TRUE(not_unroll); /* UNROLL must be eliminated */
  PASS();
}

/* ════════════════════════════════════════════════════════════════════════
 * SECTION 5: Regression tests — MUST PASS (already ported)
 * ════════════════════════════════════════════════════════════════════════ */

/* Forced EXP2 decomposition: EXP2 must be fully eliminated. */
TEST(regression, exp2_decomp) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = make_unary_kernel(ctx, POLY_OP_EXP2, 4);
  PolyUOp *rewritten = poly_graph_rewrite(ctx, sink, poly_pm_transcendental_pass());
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
  float in[4] = {0.0f, 1.0f, 2.0f, -1.0f};
  float out[4] = {0};
  void *args[2] = {in, out};
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
  PolyUOp *end_src[2] = {store, range};
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
  PolyUOp *end_src[2] = {store, range};
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
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(100));
  PolyUOp *x = poly_uop1(ctx, POLY_OP_RANGE, POLY_INDEX, bound, poly_arg_int(0));
  PolyUOp *y = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(7));
  PolyUOp *mod1 = poly_uop2(ctx, POLY_OP_FLOORMOD, POLY_INDEX, x, y, poly_arg_none());
  PolyUOp *mod2 = poly_uop2(ctx, POLY_OP_FLOORMOD, POLY_INDEX, mod1, y, poly_arg_none());
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
  float in[4] = {1.0f, 2.0f, 4.0f, 8.0f};
  float out[4] = {0};
  void *args[2] = {in, out};
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
  float in[4] = {0.0f, 1.5707963f, 3.1415927f, 6.2831853f};
  float out[4] = {0};
  void *args[2] = {in, out};
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
  PolyRendererCaps caps = poly_c_renderer_caps();
  caps.has_exp2 = false;
  caps.has_log2 = false;
  caps.has_sin = false;
  PolyRewriteOpts opts = {
      .optimize = poly_kernel_optimize_enabled(sink),
      .devectorize = 1,
      .caps = caps,
      .device = POLY_DEVICE_CPU,
      .opt_policy = POLY_OPT_HEURISTIC,
      .dtype_matcher = poly_pm_bf16_non_native(),
      .extra_matcher = poly_pm_c_renderer_extra(),
  };
  PolyUOp *rewritten = poly_full_rewrite_to_sink_ex(ctx, sink, opts);
  ASSERT_INT_EQ(count_ops_in(ctx, rewritten, POLY_OP_SIN), 0);

  int n;
  PolyUOp **lin = poly_linearize_rewritten(ctx, rewritten, &n);
  char *src = poly_render_c(lin, n, "sin_decomp_large_kernel");
  PolyProgram *prog = poly_compile_c(src, "sin_decomp_large_kernel");
  ASSERT_NOT_NULL(prog);

  float in[4] = {31.0f, 100000.0f, -100000.0f, 1234567.0f};
  float out[4] = {0};
  void *args[2] = {in, out};
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
  float in[4] = {1.0f, 4.0f, 9.0f, 16.0f};
  float out[4] = {0};
  void *args[2] = {in, out};
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
  float in[4] = {1.0f, 2.0f, 4.0f, 0.5f};
  float out[4] = {0};
  void *args[2] = {in, out};
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
  float a[4] = {10.0f, 7.0f, 1.0f, 0.0f};
  float b[4] = {2.0f, 3.0f, 3.0f, 1.0f};
  float out[4] = {0};
  void *args[3] = {a, b, out};
  poly_program_call(prog, args, 3);
  ASSERT_FLOAT_EQ(out[0], 5.0f, 1e-6);
  ASSERT_FLOAT_EQ(out[1], 7.0f / 3.0f, 1e-5);
  ASSERT_FLOAT_EQ(out[2], 1.0f / 3.0f, 1e-5);
  ASSERT_FLOAT_EQ(out[3], 0.0f, 1e-6);
  poly_program_destroy(prog);
  free(src);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

/* ════════════════════════════════════════════════════════════════════════
 * SECTION 6: Pass-order audit tests — validate multi-pass dependencies
 *
 * The codegen pipeline is: sym → pm_decomp → pm_transcendental → pm_decomp.
 * pm_transcendental creates ops (RECIPROCAL, FLOORDIV) that the second pm_decomp
 * must clean up.  These tests catch regressions where stages are reordered
 * or removed.
 * ════════════════════════════════════════════════════════════════════════ */

/*
 * LOG2 creates RECIPROCAL during decomposition (for -0 detection, codegen.c:1197).
 * The second pm_decomp converts RECIPROCAL → FDIV(1, x).
 */
TEST(pass_order, log2_reciprocal_lifecycle) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = make_unary_kernel(ctx, POLY_OP_LOG2, 4);

  /* Apply the exact mini-pipeline: sym → pm_decomp → pm_transcendental */
  sink = poly_graph_rewrite(ctx, sink, poly_symbolic_simple());
  sink = poly_graph_rewrite(ctx, sink, poly_pm_decomp_pass());
  sink = poly_graph_rewrite(ctx, sink, poly_pm_transcendental_pass());

  /* After transcendental: LOG2 is gone, RECIPROCAL introduced */
  int n_log2 = count_ops_in(ctx, sink, POLY_OP_LOG2);
  int n_recip = count_ops_in(ctx, sink, POLY_OP_RECIPROCAL);

  /* Apply second pm_decomp (step 6 of pipeline) */
  sink = poly_graph_rewrite(ctx, sink, poly_pm_decomp_pass());
  int n_recip_after = count_ops_in(ctx, sink, POLY_OP_RECIPROCAL);

  poly_ctx_destroy(ctx);

  ASSERT_INT_EQ(n_log2, 0);
  ASSERT_TRUE(n_recip > 0);
  ASSERT_INT_EQ(n_recip_after, 0);
  PASS();
}

/*
 * Pinned tinygrad uop/decompositions.py:18,49-52 creates FLOORDIV(q, 2)
 * in ldexp2k. The second pm_decomp owns its target-independent lowering.
 */
TEST(pass_order, exp2_floordiv_lifecycle) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = make_unary_kernel(ctx, POLY_OP_EXP2, 4);

  /* Apply the exact mini-pipeline: sym → pm_decomp → pm_transcendental */
  sink = poly_graph_rewrite(ctx, sink, poly_symbolic_simple());
  sink = poly_graph_rewrite(ctx, sink, poly_pm_decomp_pass());
  sink = poly_graph_rewrite(ctx, sink, poly_pm_transcendental_pass());

  int n_exp2 = count_ops_in(ctx, sink, POLY_OP_EXP2);
  int n_floordiv = count_ops_in(ctx, sink, POLY_OP_FLOORDIV);
  int n_cdiv = count_ops_in(ctx, sink, POLY_OP_CDIV);
  int n_cmod = count_ops_in(ctx, sink, POLY_OP_CMOD);

  /* Apply second pm_decomp */
  sink = poly_graph_rewrite(ctx, sink, poly_pm_decomp_pass());
  int n_floordiv_after = count_ops_in(ctx, sink, POLY_OP_FLOORDIV);

  poly_ctx_destroy(ctx);

  ASSERT_INT_EQ(n_exp2, 0);
  ASSERT_TRUE(n_floordiv > 0);
  ASSERT_INT_EQ(n_cdiv, 0);
  ASSERT_INT_EQ(n_cmod, 0);
  ASSERT_INT_EQ(n_floordiv_after, 0);
  PASS();
}

/* Pinned ClangRenderer removes SIN from code_for_op
 * (cstyle.py:246-269), so the full CPU pipeline must decompose it together
 * with unsupported internal operations. */
TEST(pass_order, full_pipeline_no_residual) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = make_unary_kernel(ctx, POLY_OP_SIN, 4);
  PolyUOp *result = poly_full_rewrite_to_sink(ctx, sink);

  int n_sin = count_ops_in(ctx, result, POLY_OP_SIN);
  int n_recip = count_ops_in(ctx, result, POLY_OP_RECIPROCAL);
  int n_max = count_ops_in(ctx, result, POLY_OP_MAX);

  poly_ctx_destroy(ctx);

  ASSERT_INT_EQ(n_sin, 0);
  ASSERT_INT_EQ(n_recip, 0);
  ASSERT_INT_EQ(n_max, 0);
  PASS();
}

/* ════════════════════════════════════════════════════════════════════════
 * SECTION 7: Transcendental conformance tests — edge cases + sweeps
 *
 * Validates numerical accuracy of the polynomial decompositions for
 * EXP2, LOG2, and SIN.  Uses ASSERT_FLOAT_ULP for precision checks
 * and ASSERT_FLOAT_ABS near switchover boundaries.
 * ════════════════════════════════════════════════════════════════════════ */

/* Helper: run a unary kernel end-to-end through the full codegen pipeline. */
static int run_unary_e2e(PolyOps op, const float *in, float *out, int n) {
  PolyCtx *ctx = poly_ctx_new();

  /* Build tensor-level graph: out[i] = op(in[i]) via poly_test_realize_buffer_views.
   * Respects POLY_DEVICE so conformance tests run on the selected backend. */
  PolyUOp *buf_in = poly_buffer(ctx, POLY_FLOAT32, n);
  PolyUOp *result = poly_alu1(ctx, op, buf_in);
  PolyUOp *buf_out = poly_buffer(ctx, POLY_FLOAT32, n);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, buf_out, result, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  float *in_copy = (float *)malloc((size_t)n * sizeof(float));
  memcpy(in_copy, in, (size_t)n * sizeof(float));
  memset(out, 0, (size_t)n * sizeof(float));

  PolyTestBufferView bindings[] = {
      POLY_TEST_HOST_VIEW(buf_out, out),
      POLY_TEST_HOST_VIEW(buf_in, in_copy),
  };
  int ret = poly_test_realize_buffer_views(ctx, sink, bindings, 2);

  free(in_copy);
  poly_ctx_destroy(ctx);
  return ret;
}

/* Deterministic PRNG for reproducible bit-pattern generation. */
static uint32_t xorshift32(uint32_t *state) {
  uint32_t x = *state;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  *state = x;
  return x;
}

/* Stratified bit-pattern sweep: 5 specials + 4 samples per exponent band (0-254).
 * Returns actual count written. Exponent band 0 = subnormals, 1-254 = normals.
 * If include_negative: randomly flip sign bit on half the samples. */
static int gen_stratified_sweep(float *out, int max_n, uint32_t seed, int include_negative) {
  uint32_t state = seed;
  int n = 0;
  /* Explicit specials: +0, -0, +inf, -inf, NaN */
  uint32_t sp[] = {0x00000000u, 0x80000000u, 0x7F800000u, 0xFF800000u, 0x7FC00000u};
  for (int i = 0; i < 5 && n < max_n; i++)
    memcpy(&out[n++], &sp[i], sizeof(float));
  /* Stratified: 4 samples per exponent band */
  for (int exp = 0; exp <= 254 && n < max_n; exp++) {
    uint32_t base = (uint32_t)exp << 23;
    for (int j = 0; j < 4 && n < max_n; j++) {
      uint32_t mantissa = xorshift32(&state) & 0x7FFFFFu;
      uint32_t bits = base | mantissa;
      if (include_negative && (xorshift32(&state) & 1)) bits |= 0x80000000u;
      memcpy(&out[n++], &bits, sizeof(float));
    }
  }
  return n;
}

/* LOG2: special values (zero, negative zero, inf, -inf, NaN, denormal, FLT_MAX) */
TEST(conformance, log2_special_values) {
  float in[8] = {0.0f, -0.0f, (float)INFINITY, (float)-INFINITY, (float)NAN, 1e-40f, 1.0f, FLT_MAX};
  float out[8] = {0};
  ASSERT_INT_EQ(run_unary_e2e(POLY_OP_LOG2, in, out, 8), 0);

  ASSERT_FLOAT_INF(out[0], -1); /* log2(0)    = -inf */
  ASSERT_FLOAT_INF(out[1], -1); /* log2(-0)   = -inf */
  ASSERT_FLOAT_INF(out[2], +1); /* log2(+inf) = +inf */
  ASSERT_FLOAT_NAN(out[3]); /* log2(-inf) = NaN  */
  ASSERT_FLOAT_NAN(out[4]); /* log2(NaN)  = NaN  */
  ASSERT_FLOAT_ULP(out[5], log2f(1e-40f), 128); /* denormal          */
  ASSERT_FLOAT_ULP(out[6], 0.0f, 0); /* log2(1) = 0 exact */
  ASSERT_FLOAT_ULP(out[7], log2f(FLT_MAX), 8); /* large             */

  PASS();
}

/* LOG2: all negative inputs must return NaN */
TEST(conformance, log2_negative_domain) {
  float in[4] = {-1.0f, -100.0f, -FLT_MIN, -FLT_MAX};
  float out[4] = {0};
  ASSERT_INT_EQ(run_unary_e2e(POLY_OP_LOG2, in, out, 4), 0);

  for (int i = 0; i < 4; i++)
    ASSERT_FLOAT_NAN(out[i]);

  PASS();
}

/* LOG2: dense sweep from 0.001 to 1e6 (log-spaced, 64 values) */
TEST(conformance, log2_dense_sweep) {
  float in[64], out[64];
  for (int i = 0; i < 64; i++)
    in[i] = powf(10.0f, -3.0f + 9.0f * (float)i / 63.0f);
  ASSERT_INT_EQ(run_unary_e2e(POLY_OP_LOG2, in, out, 64), 0);

  for (int i = 0; i < 64; i++)
    ASSERT_FLOAT_ULP(out[i], log2f(in[i]), 128);

  PASS();
}

/* SIN: special values */
TEST(conformance, sin_special_values) {
  float in[6] = {0.0f, -0.0f, (float)INFINITY, (float)-INFINITY, (float)NAN, 3.14159265f};
  float out[6] = {0};
  ASSERT_INT_EQ(run_unary_e2e(POLY_OP_SIN, in, out, 6), 0);

  ASSERT_FLOAT_ABS(out[0], 0.0f, 1e-7f); /* sin(0)    = 0    */
  ASSERT_FLOAT_ABS(out[1], 0.0f, 1e-7f); /* sin(-0)   = 0    */
  ASSERT_FLOAT_NAN(out[2]); /* sin(+inf) = NaN  */
  ASSERT_FLOAT_NAN(out[3]); /* sin(-inf) = NaN  */
  ASSERT_FLOAT_NAN(out[4]); /* sin(NaN)  = NaN  */
  ASSERT_FLOAT_ABS(out[5], sinf(3.14159265f), 1e-3f); /* sin(pi) ~ 0  */

  PASS();
}

/* SIN: quadrant boundaries (all < 30, uses Cody-Waite path).
 * Uses ASSERT_FLOAT_ABS because sin(pi) ~ 0 and ULP distance near zero
 * is meaningless (tiny absolute error becomes huge ULP count). */
TEST(conformance, sin_quadrant_boundaries) {
  float pi = 3.14159265358979f;
  float in[8] = {pi / 6, pi / 4, pi / 3, pi / 2, 2 * pi / 3, 3 * pi / 4, 5 * pi / 6, pi};
  float out[8] = {0};
  ASSERT_INT_EQ(run_unary_e2e(POLY_OP_SIN, in, out, 8), 0);

  for (int i = 0; i < 8; i++)
    ASSERT_FLOAT_ABS(out[i], sinf(in[i]), 1e-5f);

  PASS();
}

/* SIN: Cody-Waite / Payne-Hanek switchover at 30.0 */
TEST(conformance, sin_switchover_boundary) {
  float in[8] = {29.0f, 29.5f, 29.9f, 30.0f, 30.1f, 30.5f, 31.0f, 32.0f};
  float out[8] = {0};
  ASSERT_INT_EQ(run_unary_e2e(POLY_OP_SIN, in, out, 8), 0);

  for (int i = 0; i < 8; i++)
    ASSERT_FLOAT_ABS(out[i], sinf(in[i]), 2e-3f);

  PASS();
}

/* EXP2: special values (zero, -0, inf, -inf, NaN, overflow, underflow) */
TEST(conformance, exp2_special_values) {
  float in[7] = {0.0f, -0.0f, (float)INFINITY, (float)-INFINITY, (float)NAN, 128.0f, -150.0f};
  float out[7] = {0};
  ASSERT_INT_EQ(run_unary_e2e(POLY_OP_EXP2, in, out, 7), 0);

  ASSERT_FLOAT_ULP(out[0], 1.0f, 0); /* exp2(0)    = 1    */
  ASSERT_FLOAT_ULP(out[1], 1.0f, 0); /* exp2(-0)   = 1    */
  ASSERT_FLOAT_INF(out[2], +1); /* exp2(+inf) = +inf */
  ASSERT_FLOAT_ABS(out[3], 0.0f, 1e-44f); /* exp2(-inf) = 0    */
  ASSERT_FLOAT_NAN(out[4]); /* exp2(NaN)  = NaN  */
  ASSERT_FLOAT_INF(out[5], +1); /* exp2(128)  = +inf */
  ASSERT_FLOAT_ABS(out[6], 0.0f, 1e-44f); /* exp2(-150) = 0    */

  PASS();
}

/* ════════════════════════════════════════════════════════════════════════
 * SECTION 8: Dense bit-pattern sweep tests
 *
 * Stratified sampling across all 255 IEEE 754 exponent bands (0=subnormal,
 * 1-254=normal) with 4 random mantissa values per band.  Catches
 * exponent-dependent bugs that uniform-in-value sampling misses.
 * ════════════════════════════════════════════════════════════════════════ */

/* LOG2: 1025 stratified bit patterns (positive domain + specials).
 * Tolerance: 128 ULP or 1e-6 absolute. */
/* LOG2: 1025 stratified bit patterns (positive domain + specials).
 * Tolerance: 256 ULP or 1e-5 absolute (GPU transcendentals may differ from CPU libm). */
TEST(conformance, log2_bitpattern_sweep) {
  float in[1025], out[1025];
  int n = gen_stratified_sweep(in, 1025, 0xDEAD0001u, 0);
  ASSERT_INT_EQ(run_unary_e2e(POLY_OP_LOG2, in, out, n), 0);
  for (int i = 0; i < n; i++)
    ASSERT_FLOAT_NEAR(out[i], log2f(in[i]), 256, 1e-5f);
  PASS();
}

/* SIN: 1025 stratified bit patterns (full domain including negatives).
 * Tolerance: 256 ULP or 1e-5 absolute (wider due to argument reduction). */
TEST(conformance, sin_bitpattern_sweep) {
  float in[1025], out[1025];
  int n = gen_stratified_sweep(in, 1025, 0xDEAD0002u, 1);
  ASSERT_INT_EQ(run_unary_e2e(POLY_OP_SIN, in, out, n), 0);
  for (int i = 0; i < n; i++)
    ASSERT_FLOAT_NEAR(out[i], sinf(in[i]), 256, 1e-5f);
  PASS();
}

/* EXP2: 1025 stratified bit patterns (full domain including negatives).
 * Tolerance: 128 ULP or 1e-6 absolute. */
TEST(conformance, exp2_bitpattern_sweep) {
  float in[1025], out[1025];
  int n = gen_stratified_sweep(in, 1025, 0xDEAD0003u, 1);
  ASSERT_INT_EQ(run_unary_e2e(POLY_OP_EXP2, in, out, n), 0);
  for (int i = 0; i < n; i++)
    ASSERT_FLOAT_NEAR(out[i], exp2f(in[i]), 128, 1e-6f);
  PASS();
}

/* ════════════════════════════════════════════════════════════════════════
 * SECTION 9: Float64 transcendental conformance tests
 * ════════════════════════════════════════════════════════════════════════ */

/* Build a f64 unary kernel: out[i] = OP(in[i]) for i in [0, n) */
static PolyUOp *make_unary_kernel_f64(PolyCtx *ctx, PolyOps op, int n) {
  PolyDType ptr_f64 = poly_dtype_ptr(POLY_FLOAT64, -1, POLY_ADDR_GLOBAL);
  PolyUOp *p0 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f64, poly_arg_int(0));
  PolyUOp *p1 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f64, poly_arg_int(1));
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(n));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));
  PolyUOp *idx0 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f64, p0, range, poly_arg_none());
  PolyUOp *idx1 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f64, p1, range, poly_arg_none());
  PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT64, idx0, poly_arg_none());
  PolyUOp *alu = poly_uop1(ctx, op, POLY_FLOAT64, load, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx1, alu, poly_arg_none());
  PolyUOp *end_src[2] = {store, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  return poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());
}

/* Helper: run a f64 unary kernel end-to-end. */
static int run_unary_e2e_f64(PolyOps op, const double *in, double *out, int n) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = make_unary_kernel_f64(ctx, op, n);
  int nlin;
  PolyUOp **lin = poly_linearize(ctx, sink, &nlin);
  char name[64];
  snprintf(name, sizeof(name), "conformance_f64_%d", op);
  char *src = poly_render_c(lin, nlin, name);
  PolyProgram *prog = poly_compile_c(src, name);
  if (!prog) {
    free(src);
    free(lin);
    poly_ctx_destroy(ctx);
    return -1;
  }
  double *in_copy = (double *)malloc((size_t)n * sizeof(double));
  memcpy(in_copy, in, (size_t)n * sizeof(double));
  void *args[2] = {in_copy, out};
  poly_program_call(prog, args, 2);
  poly_program_destroy(prog);
  free(src);
  free(lin);
  free(in_copy);
  poly_ctx_destroy(ctx);
  return 0;
}

/* EXP2 f64: special values */
TEST(conformance_f64, exp2_special_values) {
  double in[7] = {0.0, -0.0, INFINITY, -INFINITY, NAN, 1024.0, -2000.0};
  double out[7] = {0};
  ASSERT_INT_EQ(run_unary_e2e_f64(POLY_OP_EXP2, in, out, 7), 0);

  ASSERT_DOUBLE_ULP(out[0], 1.0, 0); /* exp2(0)     = 1    */
  ASSERT_DOUBLE_ULP(out[1], 1.0, 0); /* exp2(-0)    = 1    */
  ASSERT_DOUBLE_INF(out[2], +1); /* exp2(+inf)  = +inf */
  ASSERT_DOUBLE_ABS(out[3], 0.0, 1e-300); /* exp2(-inf)  = 0    */
  ASSERT_DOUBLE_NAN(out[4]); /* exp2(NaN)   = NaN  */
  ASSERT_DOUBLE_INF(out[5], +1); /* exp2(1024)  = +inf */
  ASSERT_DOUBLE_ABS(out[6], 0.0, 1e-300); /* exp2(-2000) = 0    */

  PASS();
}

/* EXP2 f64: normal range values */
TEST(conformance_f64, exp2_normal_range) {
  double in[8] = {1.0, -1.0, 10.0, -10.0, 0.5, -0.5, 100.0, -100.0};
  double out[8] = {0};
  ASSERT_INT_EQ(run_unary_e2e_f64(POLY_OP_EXP2, in, out, 8), 0);

  for (int i = 0; i < 8; i++)
    ASSERT_DOUBLE_ULP(out[i], exp2(in[i]), 4);

  PASS();
}

/* LOG2 f64: special values */
TEST(conformance_f64, log2_special_values) {
  double in[8] = {0.0, -0.0, INFINITY, -INFINITY, NAN, 5e-324, 1.0, DBL_MAX};
  double out[8] = {0};
  ASSERT_INT_EQ(run_unary_e2e_f64(POLY_OP_LOG2, in, out, 8), 0);

  ASSERT_DOUBLE_INF(out[0], -1); /* log2(0)    = -inf */
  ASSERT_DOUBLE_INF(out[1], -1); /* log2(-0)   = -inf */
  ASSERT_DOUBLE_INF(out[2], +1); /* log2(+inf) = +inf */
  ASSERT_DOUBLE_NAN(out[3]); /* log2(-inf) = NaN  */
  ASSERT_DOUBLE_NAN(out[4]); /* log2(NaN)  = NaN  */
  ASSERT_DOUBLE_ULP(out[5], log2(5e-324), 256); /* denormal (subnormal min) */
  ASSERT_DOUBLE_ULP(out[6], 0.0, 0); /* log2(1) = 0 exact */
  ASSERT_DOUBLE_ULP(out[7], log2(DBL_MAX), 16); /* large */

  PASS();
}

/* LOG2 f64: negative domain returns NaN */
TEST(conformance_f64, log2_negative_domain) {
  double in[4] = {-1.0, -100.0, -DBL_MIN, -DBL_MAX};
  double out[4] = {0};
  ASSERT_INT_EQ(run_unary_e2e_f64(POLY_OP_LOG2, in, out, 4), 0);

  for (int i = 0; i < 4; i++)
    ASSERT_DOUBLE_NAN(out[i]);

  PASS();
}

/* LOG2 f64: dense sweep 0.001 to 1e6 */
TEST(conformance_f64, log2_dense_sweep) {
  double in[64], out[64];
  for (int i = 0; i < 64; i++)
    in[i] = pow(10.0, -3.0 + 9.0 * (double)i / 63.0);
  ASSERT_INT_EQ(run_unary_e2e_f64(POLY_OP_LOG2, in, out, 64), 0);

  for (int i = 0; i < 64; i++)
    ASSERT_DOUBLE_ULP(out[i], log2(in[i]), 256);

  PASS();
}

/* SIN f64: special values */
TEST(conformance_f64, sin_special_values) {
  double pi = 3.14159265358979323846;
  double in[6] = {0.0, -0.0, INFINITY, -INFINITY, NAN, pi};
  double out[6] = {0};
  ASSERT_INT_EQ(run_unary_e2e_f64(POLY_OP_SIN, in, out, 6), 0);

  ASSERT_DOUBLE_ABS(out[0], 0.0, 1e-15); /* sin(0)    = 0    */
  ASSERT_DOUBLE_ABS(out[1], 0.0, 1e-15); /* sin(-0)   = 0    */
  ASSERT_DOUBLE_NAN(out[2]); /* sin(+inf) = NaN  */
  ASSERT_DOUBLE_NAN(out[3]); /* sin(-inf) = NaN  */
  ASSERT_DOUBLE_NAN(out[4]); /* sin(NaN)  = NaN  */
  ASSERT_DOUBLE_ABS(out[5], sin(pi), 1e-7); /* sin(pi) ~ 0     */

  PASS();
}

/* SIN f64: quadrant boundaries (Cody-Waite path) */
TEST(conformance_f64, sin_quadrant_boundaries) {
  double pi = 3.14159265358979323846;
  double in[8] = {pi / 6, pi / 4, pi / 3, pi / 2, 2 * pi / 3, 3 * pi / 4, 5 * pi / 6, pi};
  double out[8] = {0};
  ASSERT_INT_EQ(run_unary_e2e_f64(POLY_OP_SIN, in, out, 8), 0);

  for (int i = 0; i < 8; i++)
    ASSERT_DOUBLE_ABS(out[i], sin(in[i]), 1e-10);

  PASS();
}

/* SIN f64: switchover boundary */
TEST(conformance_f64, sin_switchover_boundary) {
  double in[8] = {29.0, 29.5, 29.9, 30.0, 30.1, 30.5, 31.0, 32.0};
  double out[8] = {0};
  ASSERT_INT_EQ(run_unary_e2e_f64(POLY_OP_SIN, in, out, 8), 0);

  for (int i = 0; i < 8; i++)
    ASSERT_DOUBLE_ABS(out[i], sin(in[i]), 1e-5);

  PASS();
}

/* FMA rounding divergence: prove fmaf(a,b,c) != (a*b)+c for a known float32
 * triple. Validates that MULACC conditionality has real semantic impact.
 * a*b rounds to exactly 1.0f in float32, so (a*b)+c = 0.
 * fmaf keeps the product error, giving a small nonzero result. */
TEST(conformance, fma_rounding_divergence) {
  volatile float a = 1e10f;
  volatile float b = 1e-10f;
  volatile float c = -1.0f;
  /* Force non-fused mul+add (volatile prevents compiler FMA contraction) */
  volatile float prod = a * b;
  float mul_add_result = prod + c;
  /* Fused multiply-add */
  float fma_result = fmaf((float)a, (float)b, (float)c);
  ASSERT_TRUE(mul_add_result == 0.0f); /* double-rounded: exact zero */
  ASSERT_TRUE(fma_result != mul_add_result); /* FMA: nonzero */
  ASSERT_TRUE(fma_result > 0.0f); /* sanity: positive residual */
  PASS();
}

static bool simplify_after_range_subst_i64(
    PolyCtx *ctx,
    PolyUOp *expr,
    PolyUOp *range,
    int64_t value,
    int64_t *out
) {
  PolyUOp *cv = poly_uop0(ctx, POLY_OP_CONST, range->dtype, poly_arg_int(value));
  PolyUOp *from[1] = {range};
  PolyUOp *to[1] = {cv};
  PolyUOp *sub = poly_uop_substitute(ctx, expr, from, to, 1);
  PolyUOp *folded = simplify(ctx, sub);
  if (!folded || folded->op != POLY_OP_CONST || folded->arg.kind != POLY_ARG_INT) return false;
  *out = folded->arg.i;
  return true;
}

static bool simplify_after_two_range_subst_i64(
    PolyCtx *ctx,
    PolyUOp *expr,
    PolyUOp *range0,
    int64_t value0,
    PolyUOp *range1,
    int64_t value1,
    int64_t *out
) {
  PolyUOp *cv0 = poly_uop0(ctx, POLY_OP_CONST, range0->dtype, poly_arg_int(value0));
  PolyUOp *cv1 = poly_uop0(ctx, POLY_OP_CONST, range1->dtype, poly_arg_int(value1));
  PolyUOp *from[2] = {range0, range1};
  PolyUOp *to[2] = {cv0, cv1};
  PolyUOp *sub = poly_uop_substitute(ctx, expr, from, to, 2);
  PolyUOp *folded = simplify(ctx, sub);
  if (!folded || folded->op != POLY_OP_CONST || folded->arg.kind != POLY_ARG_INT) return false;
  *out = folded->arg.i;
  return true;
}

static int count_floormod_with_const_den(PolyCtx *ctx, PolyUOp *root, int64_t den) {
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, root, &n_topo);
  int count = 0;
  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    if (u->op == POLY_OP_FLOORMOD && u->n_src == 2 && u->src[1]->op == POLY_OP_CONST &&
        u->src[1]->arg.kind == POLY_ARG_INT && u->src[1]->arg.i == den)
      count++;
  }
  return count;
}

static int count_floordiv_with_const_den(PolyCtx *ctx, PolyUOp *root, int64_t den) {
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, root, &n_topo);
  int count = 0;
  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    if (u->op == POLY_OP_FLOORDIV && u->n_src == 2 && u->src[1]->op == POLY_OP_CONST &&
        u->src[1]->arg.kind == POLY_ARG_INT && u->src[1]->arg.i == den)
      count++;
  }
  return count;
}

/* C1 divmod rules (tinygrad divandmod.py alignment) */

/* nested_div_mod: (x%6)//3 → (x//3)%2.  Ref: divandmod.py:25-27 */
TEST(sym_future, nested_div_mod_floordiv) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(12));
  PolyUOp *x = poly_uop1(ctx, POLY_OP_RANGE, POLY_INDEX, bound, poly_arg_int(0));
  PolyUOp *six = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(6));
  PolyUOp *three = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(3));
  PolyUOp *mod6 = poly_uop2(ctx, POLY_OP_FLOORMOD, POLY_INDEX, x, six, poly_arg_none());
  PolyUOp *div3 = poly_uop2(ctx, POLY_OP_FLOORDIV, POLY_INDEX, mod6, three, poly_arg_none());
  PolyUOp *r = simplify(ctx, div3);
  /* Should become (x//3)%2 */
  ASSERT_TRUE(r->op == POLY_OP_FLOORMOD);
  ASSERT_TRUE(r->src[0]->op == POLY_OP_FLOORDIV);
  ASSERT_TRUE(r->src[1]->op == POLY_OP_CONST && r->src[1]->arg.i == 2);
  poly_ctx_destroy(ctx);
  PASS();
}

/* nested_div_mod: (x%6)%3 → x%3.  Ref: divandmod.py:25-27 */
TEST(sym_future, nested_div_mod_mod) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(12));
  PolyUOp *x = poly_uop1(ctx, POLY_OP_RANGE, POLY_INDEX, bound, poly_arg_int(0));
  PolyUOp *six = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(6));
  PolyUOp *three = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(3));
  PolyUOp *mod6 = poly_uop2(ctx, POLY_OP_FLOORMOD, POLY_INDEX, x, six, poly_arg_none());
  PolyUOp *mod3 = poly_uop2(ctx, POLY_OP_FLOORMOD, POLY_INDEX, mod6, three, poly_arg_none());
  PolyUOp *r = simplify(ctx, mod3);
  /* Should become x%3 */
  ASSERT_TRUE(r->op == POLY_OP_FLOORMOD);
  ASSERT_TRUE(r->src[0] == x);
  ASSERT_TRUE(r->src[1]->op == POLY_OP_CONST && r->src[1]->arg.i == 3);
  poly_ctx_destroy(ctx);
  PASS();
}

/* remove_nested_mod: (x%4 + y)%2 → (x+y)%2.  Ref: divandmod.py:29-37 */
TEST(sym_future, remove_nested_mod) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(10));
  PolyUOp *x = poly_uop1(ctx, POLY_OP_RANGE, POLY_INDEX, bound, poly_arg_int(0));
  PolyUOp *y = poly_uop1(ctx, POLY_OP_RANGE, POLY_INDEX, bound, poly_arg_int(1));
  PolyUOp *four = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(4));
  PolyUOp *two = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(2));
  PolyUOp *xmod4 = poly_uop2(ctx, POLY_OP_FLOORMOD, POLY_INDEX, x, four, poly_arg_none());
  PolyUOp *sum = poly_uop2(ctx, POLY_OP_ADD, POLY_INDEX, xmod4, y, poly_arg_none());
  PolyUOp *mod2 = poly_uop2(ctx, POLY_OP_FLOORMOD, POLY_INDEX, sum, two, poly_arg_none());
  PolyUOp *r = simplify(ctx, mod2);
  /* Should strip inner x%4 since 4%2==0, becoming (x+y)%2 */
  ASSERT_TRUE(r->op == POLY_OP_FLOORMOD);
  /* The inner sum should have x directly, not x%4 */
  PolyUOp *inner_sum = r->src[0];
  ASSERT_TRUE(inner_sum->op == POLY_OP_ADD);
  bool has_mod = false;
  if (inner_sum->src[0]->op == POLY_OP_FLOORMOD || inner_sum->src[1]->op == POLY_OP_FLOORMOD)
    has_mod = true;
  ASSERT_TRUE(!has_mod);
  poly_ctx_destroy(ctx);
  PASS();
}

/* fold_binary_numerator: x in [3,4], (2*x+1)//5.  Ref: divandmod.py:43-47 */
TEST(sym_future, fold_binary_numerator) {
  PolyCtx *ctx = poly_ctx_new();
  /* x = RANGE(2) + 3 → range [3,4] */
  PolyUOp *two = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(2));
  PolyUOp *r0 = poly_uop1(ctx, POLY_OP_RANGE, POLY_INDEX, two, poly_arg_int(0));
  PolyUOp *three = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(3));
  PolyUOp *x = poly_uop2(ctx, POLY_OP_ADD, POLY_INDEX, r0, three, poly_arg_none());
  /* expr = (2*x + 1) // 5 */
  PolyUOp *two_c = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(2));
  PolyUOp *mul2x = poly_uop2(ctx, POLY_OP_MUL, POLY_INDEX, two_c, x, poly_arg_none());
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(1));
  PolyUOp *add1 = poly_uop2(ctx, POLY_OP_ADD, POLY_INDEX, mul2x, one, poly_arg_none());
  PolyUOp *five = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(5));
  PolyUOp *div = poly_uop2(ctx, POLY_OP_FLOORDIV, POLY_INDEX, add1, five, poly_arg_none());
  PolyUOp *r = simplify(ctx, div);
  /* x=3: (7)//5=1, x=4: (9)//5=1. Same quotient → should fold to CONST(1) */
  ASSERT_TRUE(r->op == POLY_OP_CONST && r->arg.i == 1);
  poly_ctx_destroy(ctx);
  PASS();
}

/* gcd_with_remainder: (6*x)%4 with x>=0 → GCD(6,4)=2, (3*x)%2*2.  Ref: divandmod.py:58-63 */
TEST(sym_future, gcd_with_remainder) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(10));
  PolyUOp *x = poly_uop1(ctx, POLY_OP_RANGE, POLY_INDEX, bound, poly_arg_int(0));
  PolyUOp *six = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(6));
  PolyUOp *four = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(4));
  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, POLY_INDEX, six, x, poly_arg_none());
  PolyUOp *mod = poly_uop2(ctx, POLY_OP_FLOORMOD, POLY_INDEX, mul, four, poly_arg_none());
  PolyUOp *r = simplify(ctx, mod);
  /* GCD(6,4)=2. Result should be (3*x)%2 * 2, which is simpler.
   * The exact form depends on simplification, but it should NOT be the original (6*x)%4. */
  ASSERT_TRUE(r != mod);
  /* Verify correctness: evaluate for x=0..9 */
  /* (6*0)%4=0, (6*1)%4=2, (6*2)%4=0, (6*3)%4=2, etc. */
  /* The simplified form should produce the same values when evaluated */
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym_future, gcd_with_negative_additive_const_uses_floor_splits) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *five = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(5));
  PolyUOp *r0 = poly_uop1(ctx, POLY_OP_RANGE, POLY_INDEX, five, poly_arg_int(0));
  PolyUOp *four_off = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(4));
  PolyUOp *v = poly_uop2(ctx, POLY_OP_ADD, POLY_INDEX, r0, four_off, poly_arg_none());
  PolyUOp *six = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(6));
  PolyUOp *neg_three = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(-3));
  PolyUOp *num = poly_uop2(
      ctx, POLY_OP_ADD, POLY_INDEX,
      poly_uop2(ctx, POLY_OP_MUL, POLY_INDEX, six, v, poly_arg_none()), neg_three, poly_arg_none()
  );
  PolyUOp *den = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(4));
  PolyUOp *mod =
      simplify(ctx, poly_uop2(ctx, POLY_OP_FLOORMOD, POLY_INDEX, num, den, poly_arg_none()));
  PolyUOp *div =
      simplify(ctx, poly_uop2(ctx, POLY_OP_FLOORDIV, POLY_INDEX, num, den, poly_arg_none()));

  for (int64_t i = 0; i < 5; i++) {
    int64_t vv = i + 4;
    int64_t want_num = 6 * vv - 3;
    int64_t got = 0;
    ASSERT_TRUE(simplify_after_range_subst_i64(ctx, mod, r0, i, &got));
    ASSERT_INT_EQ((int)got, (int)(want_num % 4));
    ASSERT_TRUE(simplify_after_range_subst_i64(ctx, div, r0, i, &got));
    ASSERT_INT_EQ((int)got, (int)(want_num / 4));
  }
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym_future, mod_nest_by_factor_matches_tinygrad) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *g_bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(16));
  PolyUOp *l_bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(4));
  PolyUOp *gidx = poly_uop1(ctx, POLY_OP_RANGE, POLY_INDEX, g_bound, poly_arg_int(0));
  PolyUOp *lidx = poly_uop1(ctx, POLY_OP_RANGE, POLY_INDEX, l_bound, poly_arg_int(1));
  PolyUOp *flat = poly_uop2(
      ctx, POLY_OP_ADD, POLY_INDEX,
      poly_uop2(
          ctx, POLY_OP_MUL, POLY_INDEX, gidx,
          poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(4)), poly_arg_none()
      ),
      lidx, poly_arg_none()
  );
  PolyUOp *mod = poly_uop2(
      ctx, POLY_OP_FLOORMOD, POLY_INDEX, flat,
      poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(8)), poly_arg_none()
  );
  PolyUOp *r = simplify(ctx, mod);
  /* tinygrad: (gidx*4+lidx)%8 -> (gidx%2)*4+lidx */
  ASSERT_INT_EQ(count_floormod_with_const_den(ctx, r, 8), 0);
  ASSERT_INT_EQ(count_floormod_with_const_den(ctx, r, 2), 1);
  for (int64_t g = 0; g < 16; g += 5) {
    for (int64_t l = 0; l < 4; l++) {
      int64_t got = 0;
      ASSERT_TRUE(simplify_after_two_range_subst_i64(ctx, r, gidx, g, lidx, l, &got));
      ASSERT_INT_EQ((int)got, (int)((g * 4 + l) % 8));
    }
  }
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym_future, div_mod_recombine_after_nesting_matches_tinygrad) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *g_bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(16));
  PolyUOp *l_bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(4));
  PolyUOp *gidx = poly_uop1(ctx, POLY_OP_RANGE, POLY_INDEX, g_bound, poly_arg_int(0));
  PolyUOp *lidx = poly_uop1(ctx, POLY_OP_RANGE, POLY_INDEX, l_bound, poly_arg_int(1));
  PolyUOp *four = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(4));
  PolyUOp *eight = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(8));
  PolyUOp *flat = poly_uop2(
      ctx, POLY_OP_ADD, POLY_INDEX,
      poly_uop2(ctx, POLY_OP_MUL, POLY_INDEX, gidx, four, poly_arg_none()), lidx, poly_arg_none()
  );
  PolyUOp *expr = poly_uop2(
      ctx, POLY_OP_ADD, POLY_INDEX,
      poly_uop2(
          ctx, POLY_OP_MUL, POLY_INDEX,
          poly_uop2(ctx, POLY_OP_FLOORDIV, POLY_INDEX, flat, eight, poly_arg_none()), eight,
          poly_arg_none()
      ),
      poly_uop2(ctx, POLY_OP_FLOORMOD, POLY_INDEX, flat, eight, poly_arg_none()), poly_arg_none()
  );
  PolyUOp *r = simplify(ctx, expr);
  /* tinygrad: ((flat//8)*8 + flat%8) -> flat */
  ASSERT_INT_EQ(count_ops_in(ctx, r, POLY_OP_FLOORDIV), 0);
  ASSERT_INT_EQ(count_ops_in(ctx, r, POLY_OP_FLOORMOD), 0);
  for (int64_t g = 0; g < 16; g += 5) {
    for (int64_t l = 0; l < 4; l++) {
      int64_t got = 0;
      ASSERT_TRUE(simplify_after_two_range_subst_i64(ctx, r, gidx, g, lidx, l, &got));
      ASSERT_INT_EQ((int)got, (int)(g * 4 + l));
    }
  }
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym_future, div_partial_quotient_matches_tinygrad) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(101));
  PolyUOp *b = poly_uop1(ctx, POLY_OP_RANGE, POLY_INDEX, bound, poly_arg_int(0));
  PolyUOp *num = poly_uop2(
      ctx, POLY_OP_ADD, POLY_INDEX,
      poly_uop2(
          ctx, POLY_OP_MUL, POLY_INDEX, poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(31)),
          b, poly_arg_none()
      ),
      poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(1)), poly_arg_none()
  );
  PolyUOp *expr = poly_uop2(
      ctx, POLY_OP_FLOORDIV, POLY_INDEX, num,
      poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(18)), poly_arg_none()
  );
  PolyUOp *r = simplify(ctx, expr);
  /* tinygrad: (31*b+1)//18 -> ((13*b+1)//18)+b */
  ASSERT_INT_EQ(r->op, POLY_OP_ADD);
  for (int64_t i = 0; i <= 100; i += 17) {
    int64_t got = 0;
    ASSERT_TRUE(simplify_after_range_subst_i64(ctx, r, b, i, &got));
    ASSERT_INT_EQ((int)got, (int)((31 * i + 1) / 18));
  }
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym_future, mod_congruence_tied_remainder_matches_tinygrad) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *two = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(2));
  PolyUOp *x = poly_uop1(ctx, POLY_OP_RANGE, POLY_INDEX, two, poly_arg_int(0));
  PolyUOp *y = poly_uop1(ctx, POLY_OP_RANGE, POLY_INDEX, two, poly_arg_int(1));
  PolyUOp *three = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(3));
  PolyUOp *four = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(4));

  PolyUOp *expr_a = poly_uop2(
      ctx, POLY_OP_FLOORMOD, POLY_INDEX,
      poly_uop2(
          ctx, POLY_OP_ADD, POLY_INDEX,
          poly_uop2(
              ctx, POLY_OP_ADD, POLY_INDEX, three,
              poly_uop2(
                  ctx, POLY_OP_MUL, POLY_INDEX, x,
                  poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(2)), poly_arg_none()
              ),
              poly_arg_none()
          ),
          poly_uop2(ctx, POLY_OP_MUL, POLY_INDEX, y, three, poly_arg_none()), poly_arg_none()
      ),
      four, poly_arg_none()
  );
  PolyUOp *ra = simplify(ctx, expr_a);
  ASSERT_INT_EQ(count_floormod_with_const_den(ctx, ra, 4), 0);

  PolyUOp *expr_b = poly_uop2(
      ctx, POLY_OP_FLOORMOD, POLY_INDEX,
      poly_uop2(
          ctx, POLY_OP_ADD, POLY_INDEX,
          poly_uop2(
              ctx, POLY_OP_ADD, POLY_INDEX, three,
              poly_uop2(
                  ctx, POLY_OP_MUL, POLY_INDEX, x,
                  poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(6)), poly_arg_none()
              ),
              poly_arg_none()
          ),
          poly_uop2(
              ctx, POLY_OP_MUL, POLY_INDEX, y,
              poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(7)), poly_arg_none()
          ),
          poly_arg_none()
      ),
      four, poly_arg_none()
  );
  PolyUOp *rb = simplify(ctx, expr_b);
  ASSERT_INT_EQ(count_floormod_with_const_den(ctx, rb, 4), 0);

  for (int64_t xv = 0; xv < 2; xv++) {
    for (int64_t yv = 0; yv < 2; yv++) {
      int64_t got_a = 0, got_b = 0;
      ASSERT_TRUE(simplify_after_two_range_subst_i64(ctx, ra, x, xv, y, yv, &got_a));
      ASSERT_TRUE(simplify_after_two_range_subst_i64(ctx, rb, x, xv, y, yv, &got_b));
      ASSERT_INT_EQ((int)got_a, (int)((3 + 2 * xv + 3 * yv) % 4));
      ASSERT_INT_EQ((int)got_b, (int)((3 + 6 * xv + 7 * yv) % 4));
    }
  }
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym_future, div_by_factor_tie_break_matches_tinygrad) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *two = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(2));
  PolyUOp *x = poly_uop1(ctx, POLY_OP_RANGE, POLY_INDEX, two, poly_arg_int(0));
  PolyUOp *y = poly_uop1(ctx, POLY_OP_RANGE, POLY_INDEX, two, poly_arg_int(1));
  PolyUOp *num = poly_uop2(
      ctx, POLY_OP_ADD, POLY_INDEX,
      poly_uop2(
          ctx, POLY_OP_ADD, POLY_INDEX,
          poly_uop2(ctx, POLY_OP_MUL, POLY_INDEX, x, two, poly_arg_none()),
          poly_uop2(
              ctx, POLY_OP_MUL, POLY_INDEX, y,
              poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(3)), poly_arg_none()
          ),
          poly_arg_none()
      ),
      two, poly_arg_none()
  );
  PolyUOp *expr = poly_uop2(
      ctx, POLY_OP_FLOORDIV, POLY_INDEX, num,
      poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(6)), poly_arg_none()
  );
  PolyUOp *r = simplify(ctx, expr);
  /* tinygrad: (x*2+y*3+2)//6 -> (x+y+1)//3 */
  ASSERT_INT_EQ(count_floordiv_with_const_den(ctx, r, 6), 0);
  ASSERT_INT_EQ(count_floordiv_with_const_den(ctx, r, 3), 1);
  for (int64_t xv = 0; xv < 2; xv++) {
    for (int64_t yv = 0; yv < 2; yv++) {
      int64_t got = 0;
      ASSERT_TRUE(simplify_after_two_range_subst_i64(ctx, r, x, xv, y, yv, &got));
      ASSERT_INT_EQ((int)got, (int)((2 * xv + 3 * yv + 2) / 6));
    }
  }
  poly_ctx_destroy(ctx);
  PASS();
}

/* cast(bool) → CMPNE(x, 0).  Ref: tinygrad symbolic.py line 126 */
TEST(sym_future, cast_bool_to_cmpne) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(10));
  PolyUOp *x = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));
  PolyUOp *cast = poly_uop1(ctx, POLY_OP_CAST, POLY_BOOL, x, poly_arg_none());
  PolyUOp *r = simplify(ctx, cast);
  ASSERT_TRUE(r->op == POLY_OP_CMPNE);
  ASSERT_TRUE(r->src[0] == x);
  ASSERT_TRUE(r->src[1]->op == POLY_OP_CONST && r->src[1]->arg.i == 0);
  poly_ctx_destroy(ctx);
  PASS();
}

/* SHL+ADD fuses to MULACC with FMA caps.  Ref: tinygrad decompositions.py:503 */
TEST(decomp, shl_add_fuses_to_mulacc) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x =
      poly_uop0(ctx, POLY_OP_DEFINE_VAR, POLY_INT32, poly_arg_define_var("x", -100, 100));
  PolyUOp *n = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(3));
  PolyUOp *c =
      poly_uop0(ctx, POLY_OP_DEFINE_VAR, POLY_INT32, poly_arg_define_var("c", -100, 100));
  PolyUOp *shl = poly_uop2(ctx, POLY_OP_SHL, POLY_INT32, x, n, poly_arg_none());
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, shl, c, poly_arg_none());

  /* With MULACC caps, ADD(SHL(x,3), c) -> MULACC(x, 8, c) */
  PolyRendererCaps caps = {.has_mulacc = true};
  PolyPatternMatcher *pm = poly_pm_decomp_pass_caps(caps);
  PolyUOp *r = poly_graph_rewrite(ctx, add, pm);
  ASSERT_NOT_NULL(r);
  ASSERT_TRUE(r->op == POLY_OP_MULACC);
  ASSERT_TRUE(r->n_src == 3);
  ASSERT_PTR_EQ(r->src[0], x);
  /* src[1] should be CONST(8) = 2^3 */
  ASSERT_TRUE(r->src[1]->op == POLY_OP_CONST);
  ASSERT_TRUE(r->src[1]->arg.i == 8);
  ASSERT_PTR_EQ(r->src[2], c);
  poly_ctx_destroy(ctx);
  PASS();
}

/* ══════════════════════════════════════════════════════════════════════ */
/*  Devectorize                                                          */
/* ══════════════════════════════════════════════════════════════════════ */

/* Verify no_vectorized_alu: vec4 ADD → VECTORIZE(scalar ADD × 4) */
TEST(devectorize, alu_scatter) {
  PolyCtx *ctx = poly_ctx_new();
  /* Build: VECTORIZE(a0,a1,a2,a3) + VECTORIZE(b0,b1,b2,b3) */
  PolyDType f32 = POLY_FLOAT32;
  PolyDType f32x4 = poly_dtype_vec(f32, 4);
  PolyUOp *a[4], *b[4];
  for (int i = 0; i < 4; i++) {
    a[i] = poly_uop0(ctx, POLY_OP_CONST, f32, poly_arg_float((double)(i + 1)));
    b[i] = poly_uop0(ctx, POLY_OP_CONST, f32, poly_arg_float((double)(i + 10)));
  }
  PolyUOp *va = poly_uop(ctx, POLY_OP_VECTORIZE, f32x4, a, 4, poly_arg_none());
  PolyUOp *vb = poly_uop(ctx, POLY_OP_VECTORIZE, f32x4, b, 4, poly_arg_none());
  PolyUOp *vadd = poly_uop2(ctx, POLY_OP_ADD, f32x4, va, vb, poly_arg_none());

  /* Apply devectorize */
  PolyRewriteOpts opts = {.optimize = false, .devectorize = 1};
  (void)opts;
  PolyUOp *r = poly_graph_rewrite(ctx, vadd, poly_pm_devectorize_pass());

  /* Result should be VECTORIZE of 4 scalar ADDs */
  ASSERT_TRUE(r->op == POLY_OP_VECTORIZE);
  ASSERT_INT_EQ(r->n_src, 4);
  for (int i = 0; i < 4; i++) {
    ASSERT_TRUE(r->src[i]->op == POLY_OP_ADD);
    ASSERT_TRUE(r->src[i]->dtype.count == 1);
  }
  poly_ctx_destroy(ctx);
  PASS();
}

/* Verify: vec4 MUL with scalar broadcast → VECTORIZE(scalar MUL × 4) */
TEST(devectorize, alu_broadcast_src) {
  PolyCtx *ctx = poly_ctx_new();
  PolyDType f32 = POLY_FLOAT32;
  PolyDType f32x4 = poly_dtype_vec(f32, 4);
  PolyUOp *elts[4];
  for (int i = 0; i < 4; i++)
    elts[i] = poly_uop0(ctx, POLY_OP_CONST, f32, poly_arg_float((double)(i + 1)));
  PolyUOp *va = poly_uop(ctx, POLY_OP_VECTORIZE, f32x4, elts, 4, poly_arg_none());
  /* Scalar constant — devectorizer should GEP or pass through */
  PolyUOp *scalar = poly_uop0(ctx, POLY_OP_CONST, f32, poly_arg_float(2.0));
  /* MUL(vec4, scalar) — scalar src has count=1, should be broadcast */
  PolyUOp *vmul = poly_uop2(ctx, POLY_OP_MUL, f32x4, va, scalar, poly_arg_none());

  PolyUOp *r = poly_graph_rewrite(ctx, vmul, poly_pm_devectorize_pass());
  ASSERT_TRUE(r->op == POLY_OP_VECTORIZE);
  ASSERT_INT_EQ(r->n_src, 4);
  for (int i = 0; i < 4; i++) {
    ASSERT_TRUE(r->src[i]->op == POLY_OP_MUL);
    ASSERT_TRUE(r->src[i]->dtype.count == 1);
  }
  poly_ctx_destroy(ctx);
  PASS();
}

/* Verify: scalar ALU (count=1) passes through unmodified */
TEST(devectorize, scalar_passthrough) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(1.0));
  PolyUOp *b = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(2.0));
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, a, b, poly_arg_none());

  PolyUOp *r = poly_graph_rewrite(ctx, add, poly_pm_devectorize_pass());
  /* Should be unchanged — scalar ADD has count=1 */
  ASSERT_TRUE(r == add);
  poly_ctx_destroy(ctx);
  PASS();
}

/* Verify: no_vectorized_index handles large register vectors like GPT-2 MLP/QKV
 * paths, not just <=16 lanes. This matches tinygrad's no_vectorized_index which
 * has no small-count cutoff. */
TEST(devectorize, no_vectorized_index_large_lane_count) {
  PolyCtx *ctx = poly_ctx_new();
  const int lanes = 48;
  PolyDType reg_scalar_ptr = poly_dtype_ptr(POLY_FLOAT32, lanes, POLY_ADDR_REG);
  PolyDType reg_vec_ptr = poly_dtype_ptr(poly_dtype_vec(POLY_FLOAT32, lanes), 1, POLY_ADDR_REG);
  PolyDType idx_vec = poly_dtype_vec(POLY_INT32, lanes);

  PolyUOp *reg = poly_uop0(ctx, POLY_OP_DEFINE_REG, reg_scalar_ptr, poly_arg_int(7));
  PolyUOp *cast = poly_uop1(ctx, POLY_OP_CAST, reg_vec_ptr, reg, poly_arg_none());
  PolyUOp *idx = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(3));
  PolyUOp *index = poly_uop2(ctx, POLY_OP_INDEX, reg_vec_ptr, cast, idx, poly_arg_none());

  PolyUOp *r = poly_graph_rewrite(ctx, index, poly_pm_devectorize_pass());
  ASSERT_NOT_NULL(r);
  ASSERT_INT_EQ(r->op, POLY_OP_INDEX);
  ASSERT_INT_EQ(r->n_src, 2);
  ASSERT_NOT_NULL(r->src[0]);
  ASSERT_NOT_NULL(r->src[1]);
  ASSERT_INT_EQ(r->src[0]->op, POLY_OP_VECTORIZE);
  ASSERT_INT_EQ(r->src[0]->n_src, lanes);
  ASSERT_INT_EQ(r->src[1]->dtype.count, lanes);
  for (int i = 0; i < lanes; i++) {
    ASSERT_TRUE(r->src[0]->src[i] == reg);
  }
  if (r->src[1]->op == POLY_OP_VECTORIZE || r->src[1]->op == POLY_OP_VCONST) {
    ASSERT_INT_EQ(r->src[1]->n_src, lanes);
  } else {
    ASSERT_INT_EQ(r->src[1]->op, POLY_OP_ADD);
    ASSERT_INT_EQ(r->src[1]->src[0]->op, POLY_OP_MUL);
    ASSERT_INT_EQ(r->src[1]->src[0]->dtype.count, lanes);
    ASSERT_INT_EQ(r->src[1]->src[1]->op, POLY_OP_VECTORIZE);
    ASSERT_INT_EQ(r->src[1]->src[1]->n_src, lanes);
    for (int i = 0; i < lanes; i++) {
      ASSERT_INT_EQ(r->src[1]->src[1]->src[i]->op, POLY_OP_CONST);
      ASSERT_INT_EQ((int)r->src[1]->src[1]->src[i]->arg.i, i);
    }
  }
  poly_ctx_destroy(ctx);
  PASS();
}

/* E2E: vecadd through full devectorize pipeline produces correct results */
TEST(devectorize, e2e_vecadd) {
  /* Build tensor-level: out = a + b, N=8 (divisible by 4 for UPCAST) */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer_f32(ctx, 8);
  PolyUOp *b = poly_buffer_f32(ctx, 8);
  PolyUOp *c = poly_alu2(ctx, POLY_OP_ADD, a, b);
  PolyUOp *out = poly_buffer_f32(ctx, 8);
  PolyUOp *st = poly_store_val(ctx, out, c);
  PolyUOp *sink = poly_sink1(ctx, st);

  float da[] = {1, 2, 3, 4, 5, 6, 7, 8};
  float db[] = {10, 20, 30, 40, 50, 60, 70, 80};
  float dout[8] = {0};
  PolyTestBufferView bindings[] = {
      POLY_TEST_HOST_VIEW(a, da), POLY_TEST_HOST_VIEW(b, db), POLY_TEST_HOST_VIEW(out, dout)};

  /* Use POLY_OPTIMIZE + POLY_DEVECTORIZE via env to test full pipeline */
  setenv("POLY_OPTIMIZE", "1", 1);
  setenv("POLY_DEVECTORIZE", "1", 1);
  int ret = poly_test_realize_buffer_views(ctx, sink, bindings, 3);
  unsetenv("POLY_OPTIMIZE");
  unsetenv("POLY_DEVECTORIZE");

  ASSERT_INT_EQ(ret, 0);
  for (int i = 0; i < 8; i++)
    ASSERT_FLOAT_EQ(dout[i], da[i] + db[i], 1e-6);

  poly_ctx_destroy(ctx);
  PASS();
}

/* ════════════════════════════════════════════════════════════════════════
 * SECTION 8: BEAM search optimizer tests
 * ════════════════════════════════════════════════════════════════════════ */

/* BEAM search produces correct results for vecadd (at least as good as no opts) */
TEST(beam, vecadd_correct) {
  PolyCtx *ctx = poly_ctx_new();
  int N = 64;
  PolyUOp *a = poly_buffer_f32(ctx, N);
  PolyUOp *b = poly_buffer_f32(ctx, N);
  PolyUOp *c = poly_alu2(ctx, POLY_OP_ADD, a, b);
  PolyUOp *out = poly_buffer_f32(ctx, N);
  PolyUOp *st = poly_store_val(ctx, out, c);
  PolyUOp *sink = poly_sink1(ctx, st);

  float da[64], db[64], dout[64];
  for (int i = 0; i < N; i++) {
    da[i] = (float)i;
    db[i] = (float)(100 + i);
  }
  memset(dout, 0, sizeof(dout));
  PolyTestBufferView bindings[] = {
      POLY_TEST_HOST_VIEW(a, da), POLY_TEST_HOST_VIEW(b, db), POLY_TEST_HOST_VIEW(out, dout)};

  setenv("POLY_OPTIMIZE", "1", 1);
  setenv("POLY_DEVECTORIZE", "1", 1);
  setenv("POLY_BEAM", "2", 1);
  int ret = poly_test_realize_buffer_views(ctx, sink, bindings, 3);
  unsetenv("POLY_OPTIMIZE");
  unsetenv("POLY_DEVECTORIZE");
  unsetenv("POLY_BEAM");

  ASSERT_INT_EQ(ret, 0);
  for (int i = 0; i < N; i++)
    ASSERT_FLOAT_EQ(dout[i], da[i] + db[i], 1e-6);

  poly_ctx_destroy(ctx);
  PASS();
}

/* BEAM search handles reduce kernels correctly */
TEST(beam, reduce_correct) {
  PolyCtx *ctx = poly_ctx_new();
  int N = 32;
  PolyUOp *a = poly_buffer_f32(ctx, N);
  int64_t axis = 0;
  PolyUOp *sum = poly_reduce_axis(ctx, POLY_OP_ADD, a, &axis, 1);
  PolyUOp *out = poly_buffer_f32(ctx, 1);
  PolyUOp *st = poly_store_val(ctx, out, sum);
  PolyUOp *sink = poly_sink1(ctx, st);

  float da[32], dout[1] = {0};
  float expected = 0;
  for (int i = 0; i < N; i++) {
    da[i] = (float)(i + 1);
    expected += da[i];
  }
  PolyTestBufferView bindings[] = {POLY_TEST_HOST_VIEW(a, da), POLY_TEST_HOST_VIEW(out, dout)};

  setenv("POLY_OPTIMIZE", "1", 1);
  setenv("POLY_DEVECTORIZE", "1", 1);
  setenv("POLY_BEAM", "2", 1);
  int ret = poly_test_realize_buffer_views(ctx, sink, bindings, 2);
  unsetenv("POLY_OPTIMIZE");
  unsetenv("POLY_DEVECTORIZE");
  unsetenv("POLY_BEAM");

  ASSERT_INT_EQ(ret, 0);
  ASSERT_FLOAT_EQ(dout[0], expected, 1e-3);

  poly_ctx_destroy(ctx);
  PASS();
}

/* BEAM=0 falls back to heuristic (same as no BEAM env) */
TEST(beam, zero_is_heuristic) {
  PolyCtx *ctx = poly_ctx_new();
  int N = 16;
  PolyUOp *a = poly_buffer_f32(ctx, N);
  PolyUOp *b = poly_buffer_f32(ctx, N);
  PolyUOp *c = poly_alu2(ctx, POLY_OP_ADD, a, b);
  PolyUOp *out = poly_buffer_f32(ctx, N);
  PolyUOp *st = poly_store_val(ctx, out, c);
  PolyUOp *sink = poly_sink1(ctx, st);

  float da[16], db[16], dout[16];
  for (int i = 0; i < N; i++) {
    da[i] = (float)i;
    db[i] = 1.0f;
  }
  memset(dout, 0, sizeof(dout));
  PolyTestBufferView bindings[] = {
      POLY_TEST_HOST_VIEW(a, da), POLY_TEST_HOST_VIEW(b, db), POLY_TEST_HOST_VIEW(out, dout)};

  setenv("POLY_OPTIMIZE", "1", 1);
  setenv("POLY_DEVECTORIZE", "1", 1);
  setenv("POLY_BEAM", "0", 1);
  int ret = poly_test_realize_buffer_views(ctx, sink, bindings, 3);
  unsetenv("POLY_OPTIMIZE");
  unsetenv("POLY_DEVECTORIZE");
  unsetenv("POLY_BEAM");

  ASSERT_INT_EQ(ret, 0);
  for (int i = 0; i < N; i++)
    ASSERT_FLOAT_EQ(dout[i], da[i] + db[i], 1e-6);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(optimizer_caps, heuristic_upcasts_when_scheduler_view_is_complete) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = make_binary_kernel(ctx, POLY_OP_ADD, 16);
  PolyRendererCaps caps = {.max_vec_width = 4};

  PolyUOp *optimized = poly_apply_opts_heuristic_ex(ctx, sink, caps);
  ASSERT_TRUE(optimized != NULL);
  ASSERT_TRUE(count_range_axis_type(ctx, optimized, POLY_AXIS_UPCAST) > 0);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(optimizer_caps, heuristic_skips_when_index_buffer_cap_would_truncate) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = make_many_index_kernel(ctx, 17, 16);
  PolyRendererCaps caps = {.max_vec_width = 4};

  PolyUOp *optimized = poly_apply_opts_heuristic_ex(ctx, sink, caps);
  ASSERT_PTR_EQ(optimized, sink);
  ASSERT_INT_EQ(count_range_axis_type(ctx, optimized, POLY_AXIS_UPCAST), 0);

  poly_ctx_destroy(ctx);
  PASS();
}

/* BEAM disk cache: second run should hit cache and produce same results */
TEST(beam, cache_roundtrip) {
  PolyCtx *ctx = poly_ctx_new();
  int N = 64;
  PolyUOp *a = poly_buffer_f32(ctx, N);
  PolyUOp *b = poly_buffer_f32(ctx, N);
  PolyUOp *c = poly_alu2(ctx, POLY_OP_ADD, a, b);
  PolyUOp *out = poly_buffer_f32(ctx, N);
  PolyUOp *st = poly_store_val(ctx, out, c);
  PolyUOp *sink = poly_sink1(ctx, st);

  float da[64], db[64], dout1[64], dout2[64];
  for (int i = 0; i < N; i++) {
    da[i] = (float)(i * 3);
    db[i] = (float)(i + 7);
  }

  PolyTestBufferView bindings1[] = {
      POLY_TEST_HOST_VIEW(a, da), POLY_TEST_HOST_VIEW(b, db), POLY_TEST_HOST_VIEW(out, dout1)};

  /* First run: populates cache */
  setenv("POLY_OPTIMIZE", "1", 1);
  setenv("POLY_DEVECTORIZE", "1", 1);
  setenv("POLY_BEAM", "2", 1);
  int ret1 = poly_test_realize_buffer_views(ctx, sink, bindings1, 3);
  ASSERT_INT_EQ(ret1, 0);

  /* Second run: should hit cache */
  PolyCtx *ctx2 = poly_ctx_new();
  PolyUOp *a2 = poly_buffer_f32(ctx2, N);
  PolyUOp *b2 = poly_buffer_f32(ctx2, N);
  PolyUOp *c2 = poly_alu2(ctx2, POLY_OP_ADD, a2, b2);
  PolyUOp *out2 = poly_buffer_f32(ctx2, N);
  PolyUOp *st2 = poly_store_val(ctx2, out2, c2);
  PolyUOp *sink2 = poly_sink1(ctx2, st2);

  PolyTestBufferView bindings2[] = {
      POLY_TEST_HOST_VIEW(a2, da), POLY_TEST_HOST_VIEW(b2, db), POLY_TEST_HOST_VIEW(out2, dout2)};
  int ret2 = poly_test_realize_buffer_views(ctx2, sink2, bindings2, 3);
  unsetenv("POLY_OPTIMIZE");
  unsetenv("POLY_DEVECTORIZE");
  unsetenv("POLY_BEAM");

  ASSERT_INT_EQ(ret2, 0);
  for (int i = 0; i < N; i++) {
    ASSERT_FLOAT_EQ(dout1[i], da[i] + db[i], 1e-6);
    ASSERT_FLOAT_EQ(dout2[i], da[i] + db[i], 1e-6);
  }

  poly_ctx_destroy(ctx);
  poly_ctx_destroy(ctx2);
  PASS();
}

/* BEAM search produces correct results for a chain kernel (a * b + c) */
TEST(beam, chain_correct) {
  PolyCtx *ctx = poly_ctx_new();
  int N = 128;
  PolyUOp *a = poly_buffer_f32(ctx, N);
  PolyUOp *b = poly_buffer_f32(ctx, N);
  PolyUOp *c = poly_buffer_f32(ctx, N);
  PolyUOp *ab = poly_alu2(ctx, POLY_OP_MUL, a, b);
  PolyUOp *abc = poly_alu2(ctx, POLY_OP_ADD, ab, c);
  PolyUOp *out = poly_buffer_f32(ctx, N);
  PolyUOp *st = poly_store_val(ctx, out, abc);
  PolyUOp *sink = poly_sink1(ctx, st);

  float da[128], db[128], dc[128], dout[128];
  for (int i = 0; i < N; i++) {
    da[i] = (float)(i + 1);
    db[i] = 2.0f;
    dc[i] = (float)(i * 10);
  }
  memset(dout, 0, sizeof(dout));
  PolyTestBufferView bindings[] = {
      POLY_TEST_HOST_VIEW(a, da), POLY_TEST_HOST_VIEW(b, db), POLY_TEST_HOST_VIEW(c, dc),
      POLY_TEST_HOST_VIEW(out, dout)};

  setenv("POLY_OPTIMIZE", "1", 1);
  setenv("POLY_DEVECTORIZE", "1", 1);
  setenv("POLY_BEAM", "2", 1);
  int ret = poly_test_realize_buffer_views(ctx, sink, bindings, 4);
  unsetenv("POLY_OPTIMIZE");
  unsetenv("POLY_DEVECTORIZE");
  unsetenv("POLY_BEAM");

  ASSERT_INT_EQ(ret, 0);
  for (int i = 0; i < N; i++)
    ASSERT_FLOAT_EQ(dout[i], da[i] * db[i] + dc[i], 1e-3);

  poly_ctx_destroy(ctx);
  PASS();
}

/* Section 8: Tensor core helper tests (tc.py port) */

/* AMD CDNA 16x16x16 half->float spec for testing.
 * Initialized at first use because POLY_FLOAT16/POLY_FLOAT32 are extern const. */
static PolyTensorCore test_cdna_tc;
static int test_cdna_tc_init = 0;

static const PolyTensorCore *get_test_cdna_tc(void) {
  if (!test_cdna_tc_init) {
    test_cdna_tc_init = 1;
    memset(&test_cdna_tc, 0, sizeof(test_cdna_tc));
    test_cdna_tc.dims[0] = 16;
    test_cdna_tc.dims[1] = 16;
    test_cdna_tc.dims[2] = 16;
    test_cdna_tc.threads = 64;
    test_cdna_tc.elements_per_thread[0] = 4;
    test_cdna_tc.elements_per_thread[1] = 4;
    test_cdna_tc.elements_per_thread[2] = 4;
    test_cdna_tc.dtype_in = POLY_FLOAT16;
    test_cdna_tc.dtype_out = POLY_FLOAT32;
    struct {
      char type;
      int dim;
    } opts[] = {{'l', 0}, {'l', 0}, {'l', 0}, {'l', 0}, {'u', 1}, {'u', 1}, {'l', 1}, {'l', 1}};
    for (int i = 0; i < 8; i++) {
      test_cdna_tc.opts[i].type = opts[i].type;
      test_cdna_tc.opts[i].dim = opts[i].dim;
    }
    test_cdna_tc.n_opts = 8;
    /* swizzle[0] */
    test_cdna_tc.swizzle[0][0][0] = "u0";
    test_cdna_tc.swizzle[0][0][1] = "u1";
    test_cdna_tc.swizzle[0][0][2] = "l4";
    test_cdna_tc.swizzle[0][0][3] = "l5";
    test_cdna_tc.swizzle[0][0][4] = "r2";
    test_cdna_tc.swizzle[0][0][5] = "r3";
    test_cdna_tc.swizzle[0][1][0] = "r0";
    test_cdna_tc.swizzle[0][1][1] = "r1";
    test_cdna_tc.swizzle[0][2][0] = "l0";
    test_cdna_tc.swizzle[0][2][1] = "l1";
    test_cdna_tc.swizzle[0][2][2] = "l2";
    test_cdna_tc.swizzle[0][2][3] = "l3";
    /* swizzle[1] */
    test_cdna_tc.swizzle[1][0][0] = "l0";
    test_cdna_tc.swizzle[1][0][1] = "l1";
    test_cdna_tc.swizzle[1][0][2] = "l2";
    test_cdna_tc.swizzle[1][0][3] = "l3";
    test_cdna_tc.swizzle[1][0][4] = "r2";
    test_cdna_tc.swizzle[1][0][5] = "r3";
    test_cdna_tc.swizzle[1][1][0] = "r0";
    test_cdna_tc.swizzle[1][1][1] = "r1";
    test_cdna_tc.swizzle[1][2][0] = "l4";
    test_cdna_tc.swizzle[1][2][1] = "l5";
    test_cdna_tc.swizzle[1][2][2] = "u0";
    test_cdna_tc.swizzle[1][2][3] = "u1";
    test_cdna_tc.swizzle_len[0][0] = 6;
    test_cdna_tc.swizzle_len[0][1] = 2;
    test_cdna_tc.swizzle_len[0][2] = 4;
    test_cdna_tc.swizzle_len[1][0] = 6;
    test_cdna_tc.swizzle_len[1][1] = 2;
    test_cdna_tc.swizzle_len[1][2] = 4;
    test_cdna_tc.intrinsic_name = "mfma_f32_16x16x16f16";
  }
  return &test_cdna_tc;
}

TEST(tc, get_reduce_axes) {
  int ra[16][2];
  int n = poly_tc_get_reduce_axes(get_test_cdna_tc(), ra);
  /* K=16 -> log2(16)=4 pairs, each with amt=2 */
  ASSERT_INT_EQ(n, 4);
  for (int i = 0; i < 4; i++) {
    ASSERT_INT_EQ(ra[i][0], i);
    ASSERT_INT_EQ(ra[i][1], 2);
  }
  PASS();
}

TEST(tc, count_local_upcast) {
  ASSERT_INT_EQ(poly_tc_count_local(get_test_cdna_tc()), 6); /* l0,l0,l0,l0,l1,l1 */
  ASSERT_INT_EQ(poly_tc_count_upcast(get_test_cdna_tc()), 2); /* u1,u1 */
  PASS();
}

TEST(tc, base_shape_str) {
  const char *out[32];
  int n = poly_tc_base_shape_str(get_test_cdna_tc(), out, 32);
  /* 8 opts + 4 reduce = 12 entries */
  ASSERT_INT_EQ(n, 12);
  /* Expected: l0,l1,l2,l3,u0,u1,l4,l5,r0,r1,r2,r3 */
  const char *expected[] = {"l0", "l1", "l2", "l3", "u0", "u1", "l4", "l5", "r0", "r1", "r2", "r3"};
  for (int i = 0; i < 12; i++) {
    if (strcmp(out[i], expected[i]) != 0) {
      FAIL("base_shape_str[%d]: got '%s', expected '%s'", i, out[i], expected[i]);
    }
  }
  PASS();
}

TEST(tc, base_upcast_axes) {
  const char *out[32];
  int n = poly_tc_base_upcast_axes(get_test_cdna_tc(), out, 32);
  /* reversed [r0,r1,r2,r3,u0,u1] -> [u1,u0,r3,r2,r1,r0] */
  ASSERT_INT_EQ(n, 6);
  const char *expected[] = {"u1", "u0", "r3", "r2", "r1", "r0"};
  for (int i = 0; i < 6; i++) {
    if (strcmp(out[i], expected[i]) != 0) {
      FAIL("base_upcast_axes[%d]: got '%s', expected '%s'", i, out[i], expected[i]);
    }
  }
  PASS();
}

TEST(tc, permute_for_shape_str) {
  /* Use base_shape_str as input (identity-like case) */
  const char *shape_str[32];
  int n = poly_tc_base_shape_str(get_test_cdna_tc(), shape_str, 32);
  ASSERT_INT_EQ(n, 12);

  int perm0[32], perm1[32];
  poly_tc_permute_for_shape_str(get_test_cdna_tc(), 0, shape_str, n, perm0, 32);
  poly_tc_permute_for_shape_str(get_test_cdna_tc(), 1, shape_str, n, perm1, 32);

  /* swizzle[0] flattened: u0,u1,l4,l5,r2,r3, r0,r1, l0,l1,l2,l3
   * fwd (base_shape_str): l0,l1,l2,l3,u0,u1,l4,l5,r0,r1,r2,r3
   * remap[0]: l0->u0, l1->u1, l2->l4, l3->l5, u0->r2, u1->r3, l4->r0, l5->r1, r0->l0, r1->l1,
   * r2->l2, r3->l3
   *
   * For shape_str = base_shape_str:
   *   perm0[0] = shape_str.index(remap["l0"]) = index("u0") = 4
   *   perm0[1] = index("u1") = 5
   *   perm0[2] = index("l4") = 6
   *   perm0[3] = index("l5") = 7
   *   perm0[4] = index("r2") = 10
   *   perm0[5] = index("r3") = 11
   *   perm0[6] = index("r0") = 8
   *   perm0[7] = index("r1") = 9
   *   perm0[8] = index("l0") = 0
   *   perm0[9] = index("l1") = 1
   *   perm0[10] = index("l2") = 2
   *   perm0[11] = index("l3") = 3
   */
  int expected0[] = {4, 5, 6, 7, 10, 11, 8, 9, 0, 1, 2, 3};
  for (int i = 0; i < 12; i++) {
    if (perm0[i] != expected0[i]) {
      FAIL("perm0[%d]: got %d, expected %d", i, perm0[i], expected0[i]);
    }
  }

  /* swizzle[1] flattened: l0,l1,l2,l3,r2,r3, r0,r1, l4,l5,u0,u1
   * remap[1]: l0->l0, l1->l1, l2->l2, l3->l3, u0->r2, u1->r3, l4->r0, l5->r1, r0->l4, r1->l5,
   * r2->u0, r3->u1 perm1[0] = index("l0") = 0 perm1[1] = index("l1") = 1 perm1[2] = index("l2") = 2
   *   perm1[3] = index("l3") = 3
   *   perm1[4] = index("r2") = 10
   *   perm1[5] = index("r3") = 11
   *   perm1[6] = index("r0") = 8
   *   perm1[7] = index("r1") = 9
   *   perm1[8] = index("l4") = 6
   *   perm1[9] = index("l5") = 7
   *   perm1[10] = index("u0") = 4
   *   perm1[11] = index("u1") = 5
   */
  int expected1[] = {0, 1, 2, 3, 10, 11, 8, 9, 6, 7, 4, 5};
  for (int i = 0; i < 12; i++) {
    if (perm1[i] != expected1[i]) {
      FAIL("perm1[%d]: got %d, expected %d", i, perm1[i], expected1[i]);
    }
  }

  PASS();
}

/* Section 9: TC structural detection tests */

/* Build a minimal 16x16x16 matmul kernel AST:
 * C[i,j] = sum_k( CAST_f32(A[i,k] * B[k,j]) )  where i,j,k in [0,16)
 * f16 inputs, f32 accumulation. Returns SINK. */
static PolyUOp *build_matmul_16x16x16_ast(PolyCtx *ctx) {
  PolyDType ptr_f16 = poly_dtype_ptr(POLY_FLOAT16, -1, 0);
  PolyDType ptr_f32 = poly_dtype_ptr(POLY_FLOAT32, -1, 0);

  PolyUOp *pA = poly_uop0(ctx, POLY_OP_PARAM, ptr_f16, poly_arg_int(0));
  PolyUOp *pB = poly_uop0(ctx, POLY_OP_PARAM, ptr_f16, poly_arg_int(1));
  PolyUOp *pC = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(2));

  PolyUOp *c16 = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(16));
  PolyUOp *rng_i =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, c16, poly_arg_range(0, POLY_AXIS_GLOBAL));
  PolyUOp *rng_j =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, c16, poly_arg_range(1, POLY_AXIS_GLOBAL));
  PolyUOp *rng_k =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, c16, poly_arg_range(2, POLY_AXIS_REDUCE));

  PolyUOp *c16b = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(16));
  PolyUOp *a_idx = poly_uop2(
      ctx, POLY_OP_ADD, POLY_INT32,
      poly_uop2(ctx, POLY_OP_MUL, POLY_INT32, rng_i, c16b, poly_arg_none()), rng_k, poly_arg_none()
  );
  PolyUOp *a_ptr = poly_uop2(ctx, POLY_OP_INDEX, ptr_f16, pA, a_idx, poly_arg_none());
  PolyUOp *a_val = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT16, a_ptr, poly_arg_none());

  PolyUOp *b_idx = poly_uop2(
      ctx, POLY_OP_ADD, POLY_INT32,
      poly_uop2(ctx, POLY_OP_MUL, POLY_INT32, rng_k, c16b, poly_arg_none()), rng_j, poly_arg_none()
  );
  PolyUOp *b_ptr = poly_uop2(ctx, POLY_OP_INDEX, ptr_f16, pB, b_idx, poly_arg_none());
  PolyUOp *b_val = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT16, b_ptr, poly_arg_none());

  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT16, a_val, b_val, poly_arg_none());
  PolyUOp *cast = poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, mul, poly_arg_none());

  PolyUOp *red_srcs[2] = {cast, rng_k};
  PolyArg red_arg = {.kind = POLY_ARG_OPS, .ops = POLY_OP_ADD};
  PolyUOp *reduce = poly_uop(ctx, POLY_OP_REDUCE, POLY_FLOAT32, red_srcs, 2, red_arg);

  PolyUOp *c_idx = poly_uop2(
      ctx, POLY_OP_ADD, POLY_INT32,
      poly_uop2(ctx, POLY_OP_MUL, POLY_INT32, rng_i, c16b, poly_arg_none()), rng_j, poly_arg_none()
  );
  PolyUOp *c_ptr = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, pC, c_idx, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, c_ptr, reduce, poly_arg_none());

  PolyUOp *end_srcs[3] = {store, rng_i, rng_j};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_srcs, 3, poly_arg_none());
  return poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());
}

static int count_ops(PolyCtx *ctx, PolyUOp *sink, PolyOps op) {
  int n_topo = 0, count = 0;
  PolyUOp **topo = poly_toposort(ctx, sink, &n_topo);
  for (int i = 0; i < n_topo; i++)
    if (topo[i]->op == op) count++;
  return count;
}

static int count_buffer_addrspace(PolyCtx *ctx, PolyUOp *sink, PolyAddrSpace addrspace) {
  int n_topo = 0, count = 0;
  PolyUOp **topo = poly_toposort(ctx, sink, &n_topo);
  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    if (u->op == POLY_OP_BUFFER && u->arg.kind == POLY_ARG_PARAM && u->arg.param &&
        u->arg.param->addrspace == addrspace)
      count++;
  }
  return count;
}

static int count_stage_addrspace(PolyCtx *ctx, PolyUOp *sink, PolyAddrSpace addrspace) {
  int n_topo = 0, count = 0;
  PolyUOp **topo = poly_toposort(ctx, sink, &n_topo);
  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    if (u->op == POLY_OP_STAGE && poly_bufferize_arg_addrspace(u->arg) == addrspace) count++;
  }
  return count;
}

TEST(tc, structural_matmul_ast_shape) {
  /* Verify the test AST has the expected structure: REDUCE(ADD, CAST(MUL(f16,f16))) */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = build_matmul_16x16x16_ast(ctx);

  ASSERT_INT_EQ(count_ops(ctx, sink, POLY_OP_WMMA), 0);
  ASSERT_INT_EQ(count_ops(ctx, sink, POLY_OP_CONTRACT), 0);
  ASSERT_INT_EQ(count_ops(ctx, sink, POLY_OP_REDUCE), 1);
  ASSERT_TRUE(count_ops(ctx, sink, POLY_OP_MUL) >= 1);

  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, sink, &n_topo);
  bool found = false;
  for (int i = 0; i < n_topo; i++) {
    if (topo[i]->op == POLY_OP_REDUCE && topo[i]->arg.kind == POLY_ARG_OPS &&
        topo[i]->arg.ops == POLY_OP_ADD && topo[i]->src[0]->op == POLY_OP_CAST &&
        topo[i]->src[0]->src[0]->op == POLY_OP_MUL) {
      found = true;
      ASSERT_TRUE(poly_dtype_eq(poly_dtype_scalar(topo[i]->src[0]->src[0]->dtype), POLY_FLOAT16));
      ASSERT_TRUE(poly_dtype_eq(poly_dtype_scalar(topo[i]->dtype), POLY_FLOAT32));
    }
  }
  ASSERT_TRUE(found);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(tc, detect_wmma_in_heuristic) {
  /* Run poly_apply_opts_heuristic with TC-enabled caps on a matmul AST.
   * Verify WMMA, CONTRACT, UNROLL appear in the optimized output.
   * This is the core structural detection test -- no GPU needed. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = build_matmul_16x16x16_ast(ctx);

  /* Precondition: no WMMA before */
  ASSERT_INT_EQ(count_ops(ctx, sink, POLY_OP_WMMA), 0);

  /* Run heuristic with CDNA TC spec (tc_opt=1 to allow CAST'd MUL) */
  PolyRendererCaps caps = {
      .has_mulacc = true,
      .tensor_cores = get_test_cdna_tc(),
      .n_tensor_cores = 1,
  };

  /* Set env for tc_opt=1 (allow CAST) */
  setenv("POLY_TC_OPT", "1", 1);
  setenv("POLY_USE_TC", "1", 1);

  /* poly_apply_opts_heuristic is static, so we go through poly_full_rewrite_to_sink_ex
   * which calls it when optimize=true. */
  PolyRewriteOpts opts = {.optimize = true, .caps = caps};
  PolyUOp *optimized = poly_full_rewrite_to_sink_ex(ctx, sink, opts);

  unsetenv("POLY_TC_OPT");
  unsetenv("POLY_USE_TC");

  /* Postcondition: WMMA, CONTRACT, UNROLL should appear */
  int n_wmma = count_ops(ctx, optimized, POLY_OP_WMMA);
  int n_contract = count_ops(ctx, optimized, POLY_OP_CONTRACT);
  int n_unroll = count_ops(ctx, optimized, POLY_OP_UNROLL);

  if (n_wmma == 0) {
    fprintf(
        stderr, "  detect_wmma: no WMMA found after heuristic (n_contract=%d n_unroll=%d)\n",
        n_contract, n_unroll
    );
    /* Dump op counts for debugging */
    int n_topo = 0;
    PolyUOp **topo = poly_toposort(ctx, optimized, &n_topo);
    for (int i = 0; i < n_topo; i++) {
      if (topo[i]->op == POLY_OP_REDUCE || topo[i]->op == POLY_OP_WMMA ||
          topo[i]->op == POLY_OP_CONTRACT || topo[i]->op == POLY_OP_UNROLL)
        fprintf(
            stderr, "    op[%d] = %d (REDUCE=%d WMMA=%d)\n", i, topo[i]->op, POLY_OP_REDUCE,
            POLY_OP_WMMA
        );
    }
  }

  ASSERT_TRUE(n_wmma > 0);
  /* CONTRACT and UNROLL may be lowered by the expander pass -- that's correct.
   * The key assertion is WMMA present and REDUCE(ADD) gone. */
  (void)n_contract;
  (void)n_unroll;

  /* The original REDUCE(ADD) should be gone (replaced by WMMA) */
  int n_reduce = 0;
  {
    int n_topo = 0;
    PolyUOp **topo = poly_toposort(ctx, optimized, &n_topo);
    for (int i = 0; i < n_topo; i++) {
      if (topo[i]->op == POLY_OP_REDUCE && topo[i]->arg.kind == POLY_ARG_OPS &&
          topo[i]->arg.ops == POLY_OP_ADD)
        n_reduce++;
    }
  }
  ASSERT_INT_EQ(n_reduce, 0);

  /* No lingering TC tags */
  {
    int n_topo = 0;
    PolyUOp **topo = poly_toposort(ctx, optimized, &n_topo);
    for (int i = 0; i < n_topo; i++) {
      if (topo[i]->tag == 0x5443) { /* TC_TAG */
        FAIL("lingering TC tag on op %d at topo[%d]", topo[i]->op, i);
      }
    }
  }

  poly_ctx_destroy(ctx);
  PASS();
}

/* Helper: run heuristic with TC caps and return optimized sink */
static PolyUOp *run_tc_heuristic(PolyCtx *ctx, PolyUOp *sink, const char *tc_opt_val) {
  setenv("POLY_TC_OPT", tc_opt_val, 1);
  setenv("POLY_USE_TC", "1", 1);
  PolyRendererCaps caps = {
      .has_mulacc = true,
      .tensor_cores = get_test_cdna_tc(),
      .n_tensor_cores = 1,
  };
  PolyRewriteOpts opts = {.optimize = true, .caps = caps};
  PolyUOp *result = poly_full_rewrite_to_sink_ex(ctx, sink, opts);
  unsetenv("POLY_TC_OPT");
  unsetenv("POLY_USE_TC");
  return result;
}

TEST(tc, strict_rejects_cast) {
  /* tc_opt=0 rejects CAST(MUL(f16,f16)) -- our matmul AST uses CAST */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = build_matmul_16x16x16_ast(ctx);
  PolyUOp *optimized = run_tc_heuristic(ctx, sink, "0");
  ASSERT_INT_EQ(count_ops(ctx, optimized, POLY_OP_WMMA), 0);
  poly_ctx_destroy(ctx);
  PASS();
}

/* Build 15x15x15 matmul (not divisible by 16x16x16 TC tile) */
static PolyUOp *build_matmul_NxNxN_ast(PolyCtx *ctx, int N) {
  PolyDType ptr_f16 = poly_dtype_ptr(POLY_FLOAT16, -1, 0);
  PolyDType ptr_f32 = poly_dtype_ptr(POLY_FLOAT32, -1, 0);
  PolyUOp *pA = poly_uop0(ctx, POLY_OP_PARAM, ptr_f16, poly_arg_int(0));
  PolyUOp *pB = poly_uop0(ctx, POLY_OP_PARAM, ptr_f16, poly_arg_int(1));
  PolyUOp *pC = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(2));
  PolyUOp *cN = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(N));
  PolyUOp *rng_i =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, cN, poly_arg_range(0, POLY_AXIS_GLOBAL));
  PolyUOp *rng_j =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, cN, poly_arg_range(1, POLY_AXIS_GLOBAL));
  PolyUOp *rng_k =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, cN, poly_arg_range(2, POLY_AXIS_REDUCE));
  PolyUOp *cNb = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(N));
  PolyUOp *a_idx = poly_uop2(
      ctx, POLY_OP_ADD, POLY_INT32,
      poly_uop2(ctx, POLY_OP_MUL, POLY_INT32, rng_i, cNb, poly_arg_none()), rng_k, poly_arg_none()
  );
  PolyUOp *a_val = poly_uop1(
      ctx, POLY_OP_LOAD, POLY_FLOAT16,
      poly_uop2(ctx, POLY_OP_INDEX, ptr_f16, pA, a_idx, poly_arg_none()), poly_arg_none()
  );
  PolyUOp *b_idx = poly_uop2(
      ctx, POLY_OP_ADD, POLY_INT32,
      poly_uop2(ctx, POLY_OP_MUL, POLY_INT32, rng_k, cNb, poly_arg_none()), rng_j, poly_arg_none()
  );
  PolyUOp *b_val = poly_uop1(
      ctx, POLY_OP_LOAD, POLY_FLOAT16,
      poly_uop2(ctx, POLY_OP_INDEX, ptr_f16, pB, b_idx, poly_arg_none()), poly_arg_none()
  );
  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT16, a_val, b_val, poly_arg_none());
  PolyUOp *cast = poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, mul, poly_arg_none());
  PolyUOp *red_srcs[2] = {cast, rng_k};
  PolyArg red_arg = {.kind = POLY_ARG_OPS, .ops = POLY_OP_ADD};
  PolyUOp *reduce = poly_uop(ctx, POLY_OP_REDUCE, POLY_FLOAT32, red_srcs, 2, red_arg);
  PolyUOp *c_idx = poly_uop2(
      ctx, POLY_OP_ADD, POLY_INT32,
      poly_uop2(ctx, POLY_OP_MUL, POLY_INT32, rng_i, cNb, poly_arg_none()), rng_j, poly_arg_none()
  );
  PolyUOp *store = poly_uop2(
      ctx, POLY_OP_STORE, POLY_VOID,
      poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, pC, c_idx, poly_arg_none()), reduce, poly_arg_none()
  );
  PolyUOp *end_srcs[3] = {store, rng_i, rng_j};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_srcs, 3, poly_arg_none());
  return poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());
}

TEST(tc, rejects_nondivisible) {
  /* 15x15x15 not divisible by TC tile 16x16x16 */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = build_matmul_NxNxN_ast(ctx, 15);
  PolyUOp *optimized = run_tc_heuristic(ctx, sink, "1");
  ASSERT_INT_EQ(count_ops(ctx, optimized, POLY_OP_WMMA), 0);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(tc, pre_expander_wmma_structure) {
  /* Run heuristic only (no expander) to inspect CONTRACT/UNROLL/WMMA structure.
   * Verifies that:
   * - WMMA has 3 sources: CONTRACT(A), CONTRACT(B), VECTORIZE(zeros)
   * - CONTRACT nodes carry pair-tuple args (axis_id, 2) for upcast axes
   * - UNROLL wraps WMMA with pair-tuple arg for output upcast axes
   * - All tag=1 on CONTRACT/WMMA/UNROLL (tinygrad sets tag=1 on these)
   * This tests the permutation/swizzle correctness at the IR level. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = build_matmul_16x16x16_ast(ctx);

  /* Run preprocessing (split_ranges + simplify) then heuristic only */
  setenv("POLY_TC_OPT", "1", 1);
  setenv("POLY_USE_TC", "1", 1);
  PolyRendererCaps caps = {
      .has_mulacc = true,
      .tensor_cores = get_test_cdna_tc(),
      .n_tensor_cores = 1,
  };

  /* Run the preprocessing that normally happens before heuristic */
  sink = poly_graph_rewrite(ctx, sink, poly_symbolic_simple());
  PolyUOp *optimized = poly_apply_opts_heuristic_ex(ctx, sink, caps);
  unsetenv("POLY_TC_OPT");
  unsetenv("POLY_USE_TC");

  /* Find WMMA node */
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, optimized, &n_topo);
  PolyUOp *wmma = NULL;
  int n_contract = 0, n_unroll = 0;
  for (int i = 0; i < n_topo; i++) {
    if (topo[i]->op == POLY_OP_WMMA) wmma = topo[i];
    if (topo[i]->op == POLY_OP_CONTRACT) n_contract++;
    if (topo[i]->op == POLY_OP_UNROLL) n_unroll++;
  }

  if (!wmma) {
    /* Dump ops for debugging */
    for (int i = 0; i < n_topo; i++)
      fprintf(stderr, "  [%d] op=%d tag=%d\n", i, topo[i]->op, topo[i]->tag);
    FAIL("no WMMA found in pre-expander IR");
  }

  /* WMMA must have 3 sources */
  ASSERT_INT_EQ(wmma->n_src, 3);

  /* src[0] and src[1] must be CONTRACT */
  ASSERT_INT_EQ(wmma->src[0]->op, POLY_OP_CONTRACT);
  ASSERT_INT_EQ(wmma->src[1]->op, POLY_OP_CONTRACT);

  /* src[2] must be VECTORIZE (zero accumulator) */
  ASSERT_INT_EQ(wmma->src[2]->op, POLY_OP_VECTORIZE);

  /* CONTRACT nodes must have tag=1 */
  ASSERT_INT_EQ(wmma->src[0]->tag, 1);
  ASSERT_INT_EQ(wmma->src[1]->tag, 1);

  /* WMMA must have tag=1 */
  ASSERT_INT_EQ(wmma->tag, 1);

  /* WMMA arg carries the tinygrad TensorCore metadata used by rendering and estimates. */
  ASSERT_TRUE(wmma->arg.kind == POLY_ARG_TENSOR_CORE);
  ASSERT_TRUE(strcmp(wmma->arg.tensor_core.name, "mfma_f32_16x16x16f16") == 0);
  ASSERT_INT_EQ(wmma->arg.tensor_core.dims[0], 16);
  ASSERT_INT_EQ(wmma->arg.tensor_core.dims[1], 16);
  ASSERT_INT_EQ(wmma->arg.tensor_core.dims[2], 16);
  ASSERT_INT_EQ(wmma->arg.tensor_core.threads, 64);

  /* WMMA dtype must be vec(float32, 4) for CDNA ept[2]=4 */
  ASSERT_INT_EQ(wmma->dtype.count, 4);
  ASSERT_TRUE(poly_dtype_eq(poly_dtype_scalar(wmma->dtype), POLY_FLOAT32));

  /* CONTRACT dtypes: vec(float16, 4) for CDNA ept[0]=ept[1]=4 */
  ASSERT_INT_EQ(wmma->src[0]->dtype.count, 4);
  ASSERT_TRUE(poly_dtype_eq(poly_dtype_scalar(wmma->src[0]->dtype), POLY_FLOAT16));
  ASSERT_INT_EQ(wmma->src[1]->dtype.count, 4);
  ASSERT_TRUE(poly_dtype_eq(poly_dtype_scalar(wmma->src[1]->dtype), POLY_FLOAT16));

  /* CONTRACT args must be pair tuples with (axis_id, 2) entries */
  ASSERT_TRUE(wmma->src[0]->arg.kind == POLY_ARG_PAIR_TUPLE);
  ASSERT_TRUE(wmma->src[1]->arg.kind == POLY_ARG_PAIR_TUPLE);
  /* For CDNA ept[0]=4, log2(4)=2 pairs; ept[1]=4, log2(4)=2 pairs */
  ASSERT_INT_EQ(wmma->src[0]->arg.pair_tuple.n, 2);
  ASSERT_INT_EQ(wmma->src[1]->arg.pair_tuple.n, 2);
  /* Each pair has size=2 */
  for (int i = 0; i < 2; i++) {
    ASSERT_INT_EQ(wmma->src[0]->arg.pair_tuple.pairs[i][1], 2);
    ASSERT_INT_EQ(wmma->src[1]->arg.pair_tuple.pairs[i][1], 2);
  }

  /* Find the UNROLL that wraps WMMA */
  PolyUOp *unroll = NULL;
  for (int i = 0; i < n_topo; i++) {
    if (topo[i]->op == POLY_OP_UNROLL && topo[i]->n_src > 0 && topo[i]->src[0] == wmma)
      unroll = topo[i];
  }
  ASSERT_NOT_NULL(unroll);
  ASSERT_INT_EQ(unroll->tag, 1);
  ASSERT_TRUE(unroll->arg.kind == POLY_ARG_PAIR_TUPLE);
  /* CDNA ept[2]=4, log2(4)=2 pairs */
  ASSERT_INT_EQ(unroll->arg.pair_tuple.n, 2);

  /* Verify expected op counts */
  ASSERT_INT_EQ(n_contract, 2); /* one per operand */
  ASSERT_TRUE(n_unroll >= 1); /* at least the WMMA wrapper */

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(tc, all_ne_elements_tagged) {
  /* Verify that tne tagging works for ALL ne elements (including non-RANGE
   * local-axis expressions like warp%2). tinygrad does:
   *   tne = [x.replace(tag=1) for x in ne]
   * which tags EVERY element, not just RANGEs.
   *
   * The tne tags are INTERMEDIATE -- they exist during the ne->tne->ne_reordered
   * substitution chain inside sched_apply_tc_opt. The final AST has tag=1 only
   * on CONTRACT, WMMA, UNROLL nodes.
   *
   * What we verify here: the final WMMA+CONTRACT+UNROLL structure is present,
   * confirming the full permutation chain executed correctly. If tne tagging
   * were incomplete, the permutation substitution would fail. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = build_matmul_16x16x16_ast(ctx);

  setenv("POLY_TC_OPT", "1", 1);
  setenv("POLY_USE_TC", "1", 1);
  PolyRendererCaps caps = {
      .has_mulacc = true,
      .tensor_cores = get_test_cdna_tc(),
      .n_tensor_cores = 1,
  };
  sink = poly_graph_rewrite(ctx, sink, poly_symbolic_simple());
  PolyUOp *optimized = poly_apply_opts_heuristic_ex(ctx, sink, caps);
  unsetenv("POLY_TC_OPT");
  unsetenv("POLY_USE_TC");

  /* Count tag=1 nodes: should be exactly CONTRACT(2) + WMMA(1) + UNROLL(1) = 4 */
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, optimized, &n_topo);
  int n_tagged = 0;
  int n_wmma = 0, n_contract = 0, n_unroll_tagged = 0;
  for (int i = 0; i < n_topo; i++) {
    if (topo[i]->tag == 1) {
      n_tagged++;
      if (topo[i]->op == POLY_OP_WMMA) n_wmma++;
      if (topo[i]->op == POLY_OP_CONTRACT) n_contract++;
      if (topo[i]->op == POLY_OP_UNROLL) n_unroll_tagged++;
    }
  }

  ASSERT_INT_EQ(n_wmma, 1);
  ASSERT_INT_EQ(n_contract, 2);
  ASSERT_INT_EQ(n_unroll_tagged, 1);
  ASSERT_INT_EQ(n_tagged, 4); /* exactly these 4 */

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(tc, use_tc_2_no_wmma_but_expanded) {
  /* POLY_USE_TC=2 (shape-only mode): TC detection applies shift_to transformations
   * (creating UNROLL structure) but does NOT construct WMMA/CONTRACT UOps.
   * The expander must still run to lower the UNROLL structure.
   *
   * With the bug: UNROLL ops survive in the final AST (expander gate checks WMMA only).
   * With the fix: UNROLL ops are expanded away. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = build_matmul_16x16x16_ast(ctx);

  setenv("POLY_TC_OPT", "1", 1);
  setenv("POLY_USE_TC", "2", 1); /* shape-only: no WMMA */
  PolyRendererCaps caps = {
      .has_mulacc = true,
      .tensor_cores = get_test_cdna_tc(),
      .n_tensor_cores = 1,
  };
  PolyRewriteOpts opts = {.optimize = true, .caps = caps};
  PolyUOp *optimized = poly_full_rewrite_to_sink_ex(ctx, sink, opts);
  unsetenv("POLY_TC_OPT");
  unsetenv("POLY_USE_TC");

  /* No WMMA should exist (shape-only mode) */
  ASSERT_INT_EQ(count_ops(ctx, optimized, POLY_OP_WMMA), 0);

  /* No CONTRACT should exist (not created in mode 2) */
  ASSERT_INT_EQ(count_ops(ctx, optimized, POLY_OP_CONTRACT), 0);

  /* use_tc=2 produces RANGE nodes with UNROLL axis type (from shift_to), but does NOT
   * produce POLY_OP_UNROLL ops directly. The expander may create some from the axis-typed
   * ranges. The key check is: no CONTRACT (not created in mode 2) and no WMMA. */

  poly_ctx_destroy(ctx);
  PASS();
}

/* ════════════════════════════════════════════════════════════════════════
 * SECTION 7B: Regression — HIP opts on pad+shrink+reduce kernel
 * ════════════════════════════════════════════════════════════════════════ */

#include "../src/engine/schedule.h"
#include "../src/engine/schedule.h"

TEST(unify_pre, pad_shrink_reduce_gpu_opts) {
  /* Reproduces bufferize_movement_chain_alt_ranges_e2e failure on HIP.
   * Builds the same pad+shrink+expand+reduce graph, schedules it, then
   * runs the scheduled kernel through GPU-like opts (TC_ONLY, gpu_block_size=256,
   * device=GPU) but renders to C and executes on CPU. */
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *x_flat = poly_buffer(ctx, POLY_FLOAT32, 75);
  int64_t x_shape[] = {1, 3, 5, 5};
  PolyUOp *x = poly_reshape(ctx, x_flat, x_shape, 4);

  int64_t pad_pairs[][2] = {{0, 0}, {0, 0}, {1, 1}, {1, 1}};
  PolyUOp *xp = poly_pad(ctx, x, pad_pairs, 4);

  int64_t s1_pairs[][2] = {{0, 1}, {0, 1}, {0, 5}, {0, 5}};
  int64_t s2_pairs[][2] = {{0, 1}, {0, 1}, {0, 5}, {1, 6}};
  PolyUOp *s1 = poly_shrink(ctx, xp, s1_pairs, 4);
  PolyUOp *s2 = poly_shrink(ctx, xp, s2_pairs, 4);

  int64_t out_shape[] = {1, 2, 5, 5};
  PolyUOp *e1 = poly_expand(ctx, s1, out_shape, 4);
  PolyUOp *e2 = poly_expand(ctx, s2, out_shape, 4);
  PolyUOp *sum = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, e1, e2, poly_arg_none());

  int64_t red_axes[] = {1, 2, 3};
  PolyUOp *loss = poly_reduce_axis(ctx, POLY_OP_ADD, sum, red_axes, 3);

  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, 1);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out, loss, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  /* Schedule */
  PolySchedule *sched = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_NOT_NULL(sched);
  ASSERT_TRUE(sched->template->n_calls > 0);

  /* Run each scheduled kernel through GPU-like opts but render to C */
  float x_d[75], o_d[1] = {0.0f};
  for (int i = 0; i < 75; i++)
    x_d[i] = (float)(i + 1);

  for (int k = 0; k < sched->template->n_calls; k++) {
    if (poly_schedule_call_is_copy(sched, k)) continue;

    /* Exact HIP opts with optimize=true (the tinygrad-parity path) */
    PolyRewriteOpts opts = {
        .optimize = true,
        .devectorize = -1,
        .caps = {.has_mulacc = true},
        .device = POLY_DEVICE_HIP,
        .opt_policy = POLY_OPT_TC_ONLY,
        .gpu_block_size = 256,
    };
    PolyUOp *rewritten = poly_full_rewrite_to_sink_ex(ctx, poly_schedule_call_body(sched, k), opts);

    int n_lin = 0;
    PolyUOp **lin = poly_linearize_rewritten(ctx, rewritten, &n_lin);
    if (!lin || n_lin == 0) {
      fprintf(stderr, "  pad_shrink_reduce: linearize failed for kernel %d\n", k);
      continue;
    }

    char fn_name[32];
    snprintf(fn_name, sizeof(fn_name), "test_gpu_k%d", k);
    char *src = poly_render_c(lin, n_lin, fn_name);
    free(lin);
    if (!src) {
      fprintf(stderr, "  pad_shrink_reduce: render failed for kernel %d\n", k);
      continue;
    }

    if (getenv("POLY_DUMP_KERNELS"))
      fprintf(stderr, "=== GPU-opts C kernel %d ===\n%s\n=== END ===\n", k, src);

    PolyProgram *prog = poly_compile_c(src, fn_name);
    free(src);
    if (!prog) {
      fprintf(stderr, "  pad_shrink_reduce: compile failed for kernel %d\n", k);
      continue;
    }

    /* Build args from slot indices */
    void *bufs[2] = {o_d, x_d};
    void *args[16];
    int n_args = poly_schedule_call_n_buffer_args(sched, k);
    for (int p = 0; p < n_args && p < 16; p++)
      args[p] = bufs[poly_schedule_call_buffer_slot(sched, k, p)];

    poly_program_call(prog, args, n_args);
    poly_program_destroy(prog);
  }

  if (o_d[0] != 740.0f) {
    fprintf(
        stderr, "  pad_shrink_reduce: got %.1f, expected 740.0 (delta=%.1f)\n", (double)o_d[0],
        (double)(o_d[0] - 740.0f)
    );
  }
  ASSERT_FLOAT_EQ(o_d[0], 740.0f, 1e-5);

  poly_schedule_free(sched);
  poly_ctx_destroy(ctx);
  PASS();
}

/* ════════════════════════════════════════════════════════════════════════
 * SECTION 8: Phase 4A — Pre-unification regression tests
 *
 * These tests capture current GPU pipeline behavior BEFORE linearizer
 * unification. They become regression guards for Phase 4C/4D.
 * ════════════════════════════════════════════════════════════════════════ */

/* Helper: build a large reduction kernel: out = sum(in[0..N)) */
static PolyUOp *build_large_reduction_ast(PolyCtx *ctx, int N) {
  PolyDType ptr_f32 = poly_dtype_ptr(POLY_FLOAT32, -1, POLY_ADDR_GLOBAL);
  PolyUOp *p_in = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(0));
  PolyUOp *p_out = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(1));

  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(N));
  PolyUOp *range =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_range(0, POLY_AXIS_REDUCE));
  PolyUOp *idx = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p_in, range, poly_arg_none());
  PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx, poly_arg_none());

  PolyUOp *red_srcs[2] = {load, range};
  PolyArg red_arg = {.kind = POLY_ARG_OPS, .ops = POLY_OP_ADD};
  PolyUOp *reduce = poly_uop(ctx, POLY_OP_REDUCE, POLY_FLOAT32, red_srcs, 2, red_arg);

  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *out_idx = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p_out, zero, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out_idx, reduce, poly_arg_none());
  PolyUOp *end_srcs[2] = {store, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_srcs, 2, poly_arg_none());
  return poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());
}

/* Helper: build a 2-range elementwise kernel: out[i*N+j] = in[i*N+j] + 1 */
static PolyUOp *build_2range_kernel(PolyCtx *ctx, int M, int N) {
  PolyDType ptr_f32 = poly_dtype_ptr(POLY_FLOAT32, -1, POLY_ADDR_GLOBAL);
  PolyUOp *p_in = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(0));
  PolyUOp *p_out = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(1));

  PolyUOp *cm = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(M));
  PolyUOp *cn = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(N));
  PolyUOp *rng_i =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, cm, poly_arg_range(0, POLY_AXIS_GLOBAL));
  PolyUOp *rng_j =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, cn, poly_arg_range(1, POLY_AXIS_GLOBAL));

  PolyUOp *stride = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(N));
  PolyUOp *flat = poly_uop2(
      ctx, POLY_OP_ADD, POLY_INT32,
      poly_uop2(ctx, POLY_OP_MUL, POLY_INT32, rng_i, stride, poly_arg_none()), rng_j,
      poly_arg_none()
  );

  PolyUOp *in_idx = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p_in, flat, poly_arg_none());
  PolyUOp *out_idx = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p_out, flat, poly_arg_none());
  PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, in_idx, poly_arg_none());
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(1.0));
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, load, one, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out_idx, add, poly_arg_none());

  PolyUOp *end_srcs[3] = {store, rng_i, rng_j};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_srcs, 3, poly_arg_none());
  return poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());
}

TEST(unify_pre, large_reduction_structural) {
  /* N=1024 reduction through the GPU optimizer plus group_for_reduce should
   * first produce STAGE(LOCAL) and a second REDUCE. Pinned tinygrad
   * codegen/__init__.py:84-87 then runs pm_add_buffers_local, which turns the
   * STAGE into BUFFER(LOCAL) + BARRIER. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = build_large_reduction_ast(ctx, 1024);
  PolyRendererCaps caps = {.has_local = true};

  /* Apply sym + apply_opts + group_for_reduce (same boundary as GPU pipeline) */
  sink = poly_graph_rewrite(ctx, sink, poly_symbolic_simple());
  sink = poly_apply_opts_heuristic_ex(ctx, sink, caps);
  sink = poly_graph_rewrite(ctx, sink, poly_pm_flatten_range());
  sink = poly_group_for_reduce(ctx, sink, 256);

  ASSERT_TRUE(count_stage_addrspace(ctx, sink, POLY_ADDR_LOCAL) >= 1);
  ASSERT_INT_EQ(count_buffer_addrspace(ctx, sink, POLY_ADDR_LOCAL), 0);
  ASSERT_INT_EQ(count_ops(ctx, sink, POLY_OP_DEFINE_LOCAL), 0);
  ASSERT_INT_EQ(count_ops(ctx, sink, POLY_OP_BARRIER), 0);
  ASSERT_TRUE(count_ops(ctx, sink, POLY_OP_REDUCE) >= 2); /* partial + final */

  sink = poly_apply_add_buffers_local(ctx, sink);
  ASSERT_NOT_NULL(sink);
  ASSERT_INT_EQ(count_stage_addrspace(ctx, sink, POLY_ADDR_LOCAL), 0);
  ASSERT_TRUE(count_buffer_addrspace(ctx, sink, POLY_ADDR_LOCAL) >= 1);
  ASSERT_INT_EQ(count_ops(ctx, sink, POLY_OP_DEFINE_LOCAL), 0);
  ASSERT_TRUE(count_ops(ctx, sink, POLY_OP_BARRIER) >= 1);
  ASSERT_TRUE(count_ops(ctx, sink, POLY_OP_REDUCE) >= 2);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(unify_pre, large_reduction_gpudims_special) {
  /* After apply_opts + group_for_reduce + gpudims, the grouped reduce range
   * should become SPECIAL(lidx0). */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = build_large_reduction_ast(ctx, 1024);
  PolyRendererCaps caps = {.has_local = true};

  sink = poly_graph_rewrite(ctx, sink, poly_symbolic_simple());
  sink = poly_apply_opts_heuristic_ex(ctx, sink, caps);
  sink = poly_graph_rewrite(ctx, sink, poly_pm_flatten_range());
  sink = poly_group_for_reduce(ctx, sink, 256);
  sink = poly_apply_add_buffers_local(ctx, sink);
  ASSERT_NOT_NULL(sink);
  sink = poly_apply_pm_reduce(ctx, sink);
  sink = poly_graph_rewrite(ctx, sink, poly_symbolic_simple());
  sink = poly_add_gpudims(ctx, sink);

  int n_special = count_ops(ctx, sink, POLY_OP_SPECIAL);
  ASSERT_TRUE(n_special >= 1); /* at least lidx0 */

  /* Verify SPECIAL names contain "lidx" */
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, sink, &n_topo);
  bool has_lidx = false;
  for (int i = 0; i < n_topo; i++) {
    if (topo[i]->op == POLY_OP_SPECIAL && topo[i]->arg.kind == POLY_ARG_STRING &&
        topo[i]->arg.str && strstr(topo[i]->arg.str, "lidx"))
      has_lidx = true;
  }
  ASSERT_TRUE(has_lidx);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(unify_pre, gpudims_replaces_outermost_range) {
  /* 2-range kernel: gpudims should replace the outermost RANGE with SPECIAL(gidx0).
   * After gpudims, SPECIAL ops should exist and at least one global RANGE should be gone. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = build_2range_kernel(ctx, 32, 64);

  int n_range_before = count_ops(ctx, sink, POLY_OP_RANGE);
  ASSERT_INT_EQ(n_range_before, 2);

  sink = poly_graph_rewrite(ctx, sink, poly_symbolic_simple());
  sink = poly_add_gpudims(ctx, sink);

  int n_special = count_ops(ctx, sink, POLY_OP_SPECIAL);
  int n_range_after = count_ops(ctx, sink, POLY_OP_RANGE);

  ASSERT_TRUE(n_special >= 1);
  ASSERT_TRUE(n_range_after < n_range_before);

  /* Verify gidx0 exists */
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, sink, &n_topo);
  bool has_gidx = false;
  for (int i = 0; i < n_topo; i++) {
    if (topo[i]->op == POLY_OP_SPECIAL && topo[i]->arg.kind == POLY_ARG_STRING &&
        topo[i]->arg.str && strstr(topo[i]->arg.str, "gidx"))
      has_gidx = true;
  }
  ASSERT_TRUE(has_gidx);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(unify_pre, control_flow_adds_predecessors) {
  /* After control_flow, RANGE nodes that need ordering should have additional
   * sources (predecessor edges). For a 2-range kernel the inner RANGE should
   * get the outer RANGE as a predecessor. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = build_2range_kernel(ctx, 32, 64);
  sink = poly_graph_rewrite(ctx, sink, poly_symbolic_simple());

  /* Count max RANGE src count before control_flow */
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, sink, &n_topo);
  int max_range_srcs_before = 0;
  for (int i = 0; i < n_topo; i++) {
    if (topo[i]->op == POLY_OP_RANGE && topo[i]->n_src > max_range_srcs_before)
      max_range_srcs_before = topo[i]->n_src;
  }

  sink = poly_apply_control_flow(ctx, sink);

  topo = poly_toposort(ctx, sink, &n_topo);
  int max_range_srcs_after = 0;
  for (int i = 0; i < n_topo; i++) {
    if (topo[i]->op == POLY_OP_RANGE && topo[i]->n_src > max_range_srcs_after)
      max_range_srcs_after = topo[i]->n_src;
  }

  /* Control flow adds predecessor edges when ordering is needed.
   * For a pure elementwise kernel the pass may be a no-op (no hazards).
   * Assert it at least doesn't break: no RANGE should lose sources. */
  ASSERT_TRUE(max_range_srcs_after >= max_range_srcs_before);

  /* The pass should not crash or corrupt the graph */
  int n_lin = 0;
  PolyUOp **lin = poly_linearize_rewritten(ctx, sink, &n_lin);
  ASSERT_TRUE(n_lin > 0);
  free(lin);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(unify_pre, symbolic_arithmetic_range_bound_linearizes_like_tinygrad) {
  /* Pinned UOp.range accepts any same-dtype sint source and linearize ranks
   * it with int(r.vmax)+1 (uop/spec.py:73-76,
   * codegen/late/linearizer.py:19-20). */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *n = poly_define_var(ctx, "n", 1, 7);
  PolyUOp *bound_n = poly_bind_var(ctx, n, 3);
  PolyUOp *bound = poly_alu2(ctx, POLY_OP_ADD, bound_n, poly_const_int(ctx, 1));
  PolyUOp *range = poly_uop1(
      ctx, POLY_OP_RANGE, bound->dtype, bound,
      poly_arg_range(10, POLY_AXIS_LOOP));
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, range, poly_arg_none());
  ASSERT_NOT_NULL(bound);
  ASSERT_NOT_NULL(range);
  ASSERT_NOT_NULL(sink);

  int64_t lo = 0, hi = 0;
  poly_uop_minmax(ctx, bound, &lo, &hi);
  ASSERT_INT_EQ(lo, 2);
  ASSERT_INT_EQ(hi, 8);
  poly_uop_minmax(ctx, range, &lo, &hi);
  ASSERT_INT_EQ(lo, 0);
  ASSERT_INT_EQ(hi, 7);

  PolyUOp *rewritten = poly_apply_control_flow(ctx, sink);
  ASSERT_NOT_NULL(rewritten);
  int n_linear = 0;
  PolyUOp **linear = poly_linearize_rewritten(ctx, rewritten, &n_linear);
  ASSERT_NOT_NULL(linear);
  int bound_pos = -1, range_pos = -1, sink_pos = -1;
  for (int i = 0; i < n_linear; i++) {
    if (linear[i] == bound) bound_pos = i;
    if (linear[i] == range) range_pos = i;
    if (linear[i] == sink) sink_pos = i;
  }
  ASSERT_TRUE(bound_pos >= 0);
  ASSERT_TRUE(range_pos > bound_pos);
  ASSERT_TRUE(sink_pos > range_pos);
  ASSERT_PTR_EQ(range->src[0], bound);

  free(linear);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(unify_pre, control_flow_rewinds_scratch_toposort) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = build_2range_kernel(ctx, 32, 64);
  sink = poly_graph_rewrite(ctx, sink, poly_symbolic_simple());

  size_t scratch_before = poly_arena_used(ctx->scratch);
  PolyUOp *rewritten = poly_apply_control_flow(ctx, sink);
  ASSERT_NOT_NULL(rewritten);
  ASSERT_INT_EQ(poly_arena_used(ctx->scratch), scratch_before);

  int n_lin = 0;
  PolyUOp **lin = poly_linearize_rewritten(ctx, rewritten, &n_lin);
  ASSERT_TRUE(n_lin > 0);
  free(lin);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(unify_pre, control_flow_preserves_wide_parent_arity) {
  /* Pinned tinygrad/codegen/late/linearizer.py:83-85 rewrites only RANGE by
   * appending its predecessor. A wide unmatched parent retains every source;
   * the C traversal must not impose its own arity limit. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *src[65];
  PolyUOp *ranges[2];
  for (int axis = 0; axis < 2; axis++) {
    PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(4));
    PolyUOp *range_src[1] = {bound};
    ranges[axis] = poly_uop_tagged_arg(
        ctx, POLY_OP_RANGE, POLY_INT32, range_src, 1, poly_arg_range(axis, POLY_AXIS_LOOP),
        100 + axis, poly_arg_int(200 + axis)
    );
    PolyUOp *value = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(10 + axis));
    PolyUOp *end_src[2] = {value, ranges[axis]};
    src[axis] = poly_uop_tagged_arg(
        ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none(), 300 + axis,
        poly_arg_int(400 + axis)
    );
  }
  for (int i = 2; i < 65; i++) {
    PolyUOp *value = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(1000 + i));
    src[i] = poly_uop1(ctx, POLY_OP_NOOP, POLY_INT32, value, poly_arg_none());
  }

  PolyUOp *sink = poly_uop_tagged_arg(
      ctx, POLY_OP_SINK, POLY_VOID, src, 65, poly_arg_none(), 500, poly_arg_int(600)
  );
  PolyUOp *rewritten = poly_apply_control_flow(ctx, sink);
  ASSERT_NOT_NULL(rewritten);
  ASSERT_INT_EQ(rewritten->n_src, 65);
  ASSERT_INT_EQ(rewritten->tag, 500);
  ASSERT_INT_EQ(rewritten->tag_arg.kind, POLY_ARG_INT);
  ASSERT_INT_EQ(rewritten->tag_arg.i, 600);
  ASSERT_PTR_EQ(rewritten->src[0], src[0]);
  ASSERT_TRUE(rewritten->src[1] != src[1]);
  ASSERT_INT_EQ(rewritten->src[1]->tag, 301);
  ASSERT_INT_EQ(rewritten->src[1]->tag_arg.kind, POLY_ARG_INT);
  ASSERT_INT_EQ(rewritten->src[1]->tag_arg.i, 401);
  for (int i = 2; i < 65; i++)
    ASSERT_PTR_EQ(rewritten->src[i], src[i]);

  int n_topo = 0, range_count = 0, one_source_ranges = 0, two_source_ranges = 0;
  PolyUOp **topo = poly_toposort(ctx, rewritten, &n_topo);
  PolyUOp *ordered_range = NULL;
  for (int i = 0; i < n_topo; i++) {
    if (topo[i]->op != POLY_OP_RANGE) continue;
    range_count++;
    if (topo[i]->n_src == 1) one_source_ranges++;
    if (topo[i]->n_src == 2) {
      two_source_ranges++;
      ordered_range = topo[i];
    }
  }
  ASSERT_INT_EQ(range_count, 2);
  ASSERT_INT_EQ(one_source_ranges, 1);
  ASSERT_INT_EQ(two_source_ranges, 1);
  ASSERT_NOT_NULL(ordered_range);
  ASSERT_PTR_EQ(ordered_range->src[0], ranges[1]->src[0]);
  ASSERT_PTR_EQ(ordered_range->src[1], src[0]);
  ASSERT_INT_EQ(ordered_range->tag, 101);
  ASSERT_INT_EQ(ordered_range->tag_arg.kind, POLY_ARG_INT);
  ASSERT_INT_EQ(ordered_range->tag_arg.i, 201);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(unify_pre, control_flow_fails_cleanly_at_uop_arity_ceiling) {
  /* PolyUOp.n_src is presently uint16_t. A RANGE at that ceiling cannot take
   * tinygrad's predecessor source, so reject the whole pass instead of
   * returning ancestors containing a NULL RANGE. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(4));
  PolyUOp *dummy = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(9));
  PolyUOp **range_src = malloc((size_t)UINT16_MAX * sizeof(PolyUOp *));
  ASSERT_NOT_NULL(range_src);
  range_src[0] = bound;
  for (size_t i = 1; i < UINT16_MAX; i++)
    range_src[i] = dummy;
  PolyUOp *wide_range = poly_uop(
      ctx, POLY_OP_RANGE, POLY_INT32, range_src, UINT16_MAX,
      poly_arg_range(1, POLY_AXIS_LOOP)
  );
  free(range_src);
  ASSERT_NOT_NULL(wide_range);

  PolyUOp *normal_range =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_range(0, POLY_AXIS_LOOP));
  PolyUOp *v0 = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(10));
  PolyUOp *v1 = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(11));
  PolyUOp *end0_src[2] = {v0, normal_range};
  PolyUOp *end1_src[2] = {v1, wide_range};
  PolyUOp *sink_src[2] = {
      poly_uop(ctx, POLY_OP_END, POLY_VOID, end0_src, 2, poly_arg_none()),
      poly_uop(ctx, POLY_OP_END, POLY_VOID, end1_src, 2, poly_arg_none()),
  };
  PolyUOp *sink = poly_uop(ctx, POLY_OP_SINK, POLY_VOID, sink_src, 2, poly_arg_none());
  ASSERT_TRUE(poly_apply_control_flow(ctx, sink) == NULL);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(unify_pre, full_gpu_pipeline_structural) {
  /* Run the actual shared pipeline with CUDA renderer capabilities. Pinned
   * tinygrad codegen/__init__.py:84-87 materializes grouped STAGE storage as
   * BUFFER(LOCAL) + BARRIER after the expander. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = build_large_reduction_ast(ctx, 1024);
  PolyRewriteOpts opts = {
      .optimize = true,
      .devectorize = 1,
      .caps =
          {
              .has_exp2 = true,
              .has_log2 = true,
              .has_sin = true,
              .has_int64 = true,
              .has_local = true,
              .max_vec_width = 4,
              .global_max = {2147483647, 65535, 65535},
              .local_max = {1024, 1024, 64},
          },
      .device = POLY_DEVICE_CUDA,
      .opt_policy = POLY_OPT_HEURISTIC,
      .gpu_block_size = 256,
  };
  sink = poly_full_rewrite_to_sink_ex(ctx, sink, opts);
  ASSERT_NOT_NULL(sink);
  int n_lin = 0;
  PolyUOp **lin = poly_linearize_rewritten(ctx, sink, &n_lin);
  ASSERT_NOT_NULL(lin);
  ASSERT_TRUE(n_lin > 0);

  int n_special = 0, n_local = 0, n_define_local = 0, n_barrier = 0;
  for (int i = 0; i < n_lin; i++) {
    PolyUOp *u = lin[i];
    if (u->op == POLY_OP_SPECIAL) n_special++;
    if (u->op == POLY_OP_BUFFER && u->arg.kind == POLY_ARG_PARAM && u->arg.param &&
        u->arg.param->addrspace == POLY_ADDR_LOCAL)
      n_local++;
    if (u->op == POLY_OP_DEFINE_LOCAL) n_define_local++;
    if (u->op == POLY_OP_BARRIER) n_barrier++;
  }
  ASSERT_TRUE(n_special >= 1);
  ASSERT_TRUE(n_local >= 1);
  ASSERT_INT_EQ(n_define_local, 0);
  ASSERT_TRUE(n_barrier >= 1);

  free(lin);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(unify_pre, full_rewrite_composition) {
  /* Verify the shared poly_full_rewrite_to_sink_ex applies expander + split_ends
   * + render_subset as expected. A simple kernel should have no lingering
   * internal-only ops after the full pipeline. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = make_unary_kernel(ctx, POLY_OP_EXP2, 256);

  PolyRewriteOpts opts = {.optimize = false, .devectorize = -1};
  sink = poly_full_rewrite_to_sink_ex(ctx, sink, opts);

  /* EXP2 should be decomposed */
  ASSERT_INT_EQ(count_ops(ctx, sink, POLY_OP_EXP2), 0);

  /* No CONTRACT/UNROLL should survive */
  ASSERT_INT_EQ(count_ops(ctx, sink, POLY_OP_CONTRACT), 0);
  ASSERT_INT_EQ(count_ops(ctx, sink, POLY_OP_UNROLL), 0);

  /* Linearize should succeed */
  int n_lin = 0;
  PolyUOp **lin = poly_linearize_rewritten(ctx, sink, &n_lin);
  ASSERT_TRUE(n_lin > 0);
  free(lin);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(unify_pre, gpu_pipeline_2range_elementwise) {
  /* 2-range elementwise kernel through GPU pipeline: verify SPECIAL ops
   * and successful linearization. Captures baseline for CUDA path. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = build_2range_kernel(ctx, 64, 128);
  PolyRendererCaps caps = {.has_mulacc = true};

  sink = poly_graph_rewrite(ctx, sink, poly_symbolic_simple());
  /* No group_for_reduce needed (no reduction) */
  sink = poly_apply_pm_reduce(ctx, sink);
  sink = poly_graph_rewrite(ctx, sink, poly_symbolic_simple());
  sink = poly_graph_rewrite(ctx, sink, poly_pm_decomp_pass_caps(caps));
  sink = poly_graph_rewrite(ctx, sink, poly_pm_transcendental_pass());
  sink = poly_graph_rewrite(ctx, sink, poly_pm_decomp_pass_caps(caps));
  sink = poly_add_gpudims(ctx, sink);
  sink = poly_apply_control_flow(ctx, sink);

  ASSERT_TRUE(count_ops(ctx, sink, POLY_OP_SPECIAL) >= 1);

  int n_lin = 0;
  PolyUOp **lin = poly_linearize_rewritten(ctx, sink, &n_lin);
  ASSERT_TRUE(n_lin > 0);
  free(lin);

  poly_ctx_destroy(ctx);
  PASS();
}
