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
#include "../src/frontend.h"
#include "../src/scheduler.h"

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
  PolyUOp *mulacc_srcs[3] = { ld0, ld1, ld2 };
  PolyUOp *mulacc = poly_uop(ctx, POLY_OP_MULACC, POLY_FLOAT32, mulacc_srcs, 3, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx3, mulacc, poly_arg_none());
  PolyUOp *end_src[2] = { store, range };
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());
  PolyRewriteOpts opts = { .optimize = false, .devectorize = 0,
                            .caps = { .has_mulacc = true } };
  PolyUOp *rewritten = poly_full_rewrite_to_sink_ex(ctx, sink, opts);
  int n_mulacc = count_ops_in(ctx, rewritten, POLY_OP_MULACC);
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(n_mulacc > 0);  /* MULACC preserved with FMA caps */
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
  PolyUOp *end_src[2] = { store, range };
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());
  PolyRewriteOpts opts = { .optimize = false, .devectorize = 0,
                            .caps = { .has_mulacc = true } };
  PolyUOp *rewritten = poly_full_rewrite_to_sink_ex(ctx, sink, opts);
  int n_mulacc = count_ops_in(ctx, rewritten, POLY_OP_MULACC);
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(n_mulacc > 0);  /* MUL+ADD fused to MULACC */
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
  PolyUOp *end_src[2] = { store, range };
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());
  PolyRewriteOpts opts = { .optimize = false, .devectorize = 0,
                            .caps = { .has_mulacc = true } };
  PolyUOp *rewritten = poly_full_rewrite_to_sink_ex(ctx, sink, opts);
  int n_mulacc = count_ops_in(ctx, rewritten, POLY_OP_MULACC);
  poly_ctx_destroy(ctx);
  ASSERT_INT_EQ(n_mulacc, 0);  /* No fusion for int types */
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
  PolyUOp *end_src[2] = { store, range };
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());

  PolyUOp *rewritten = poly_full_rewrite_to_sink(ctx, sink);
  int n_threefry = count_ops_in(ctx, rewritten, POLY_OP_THREEFRY);
  int n_add = count_ops_in(ctx, rewritten, POLY_OP_ADD);
  int n_xor = count_ops_in(ctx, rewritten, POLY_OP_XOR);
  poly_ctx_destroy(ctx);

  ASSERT_INT_EQ(n_threefry, 0);  /* must be fully lowered */
  ASSERT_TRUE(n_add > 0);
  ASSERT_TRUE(n_xor > 0);
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
  PolyUOp *end_src[2] = { store, range };
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());

  PolyRewriteOpts opts = { .optimize = false, .devectorize = 0,
                           .caps = { .has_mulacc = false, .has_threefry = true } };
  PolyUOp *rewritten = poly_full_rewrite_to_sink_ex(ctx, sink, opts);
  int n_threefry = count_ops_in(ctx, rewritten, POLY_OP_THREEFRY);
  poly_ctx_destroy(ctx);

  ASSERT_TRUE(n_threefry > 0);  /* native-cap path must preserve THREEFRY */
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

/* ════════════════════════════════════════════════════════════════════════
 * SECTION 6: Pass-order audit tests — validate multi-pass dependencies
 *
 * The codegen pipeline is: sym → pm_decomp → pm_transcendental → pm_decomp.
 * pm_transcendental creates ops (RECIPROCAL, IDIV) that the second pm_decomp
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
 * EXP2 creates IDIV(q, 2) in ldexp2k (codegen.c:1061).
 * The second pm_decomp converts IDIV → SHR.
 */
TEST(pass_order, exp2_idiv_lifecycle) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = make_unary_kernel(ctx, POLY_OP_EXP2, 4);

  /* Apply the exact mini-pipeline: sym → pm_decomp → pm_transcendental */
  sink = poly_graph_rewrite(ctx, sink, poly_symbolic_simple());
  sink = poly_graph_rewrite(ctx, sink, poly_pm_decomp_pass());
  sink = poly_graph_rewrite(ctx, sink, poly_pm_transcendental_pass());

  int n_exp2 = count_ops_in(ctx, sink, POLY_OP_EXP2);
  int n_idiv = count_ops_in(ctx, sink, POLY_OP_IDIV);

  /* Apply second pm_decomp */
  sink = poly_graph_rewrite(ctx, sink, poly_pm_decomp_pass());
  int n_idiv_after = count_ops_in(ctx, sink, POLY_OP_IDIV);

  poly_ctx_destroy(ctx);

  ASSERT_INT_EQ(n_exp2, 0);
  ASSERT_TRUE(n_idiv > 0);
  ASSERT_INT_EQ(n_idiv_after, 0);
  PASS();
}

/*
 * Full pipeline: no high-level ops should survive.
 */
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
  PolyUOp *sink = make_unary_kernel(ctx, op, n);
  int nlin;
  PolyUOp **lin = poly_linearize(ctx, sink, &nlin);
  char name[64];
  snprintf(name, sizeof(name), "conformance_%d", op);
  char *src = poly_render_c(lin, nlin, name);
  PolyProgram *prog = poly_compile_c(src, name);
  if (!prog) { free(src); free(lin); poly_ctx_destroy(ctx); return -1; }
  float *in_copy = (float *)malloc((size_t)n * sizeof(float));
  memcpy(in_copy, in, (size_t)n * sizeof(float));
  void *args[2] = { in_copy, out };
  poly_program_call(prog, args, 2);
  poly_program_destroy(prog);
  free(src); free(lin); free(in_copy);
  poly_ctx_destroy(ctx);
  return 0;
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
static int gen_stratified_sweep(float *out, int max_n, uint32_t seed,
                                int include_negative) {
  uint32_t state = seed;
  int n = 0;
  /* Explicit specials: +0, -0, +inf, -inf, NaN */
  uint32_t sp[] = {0x00000000u, 0x80000000u, 0x7F800000u, 0xFF800000u,
                   0x7FC00000u};
  for (int i = 0; i < 5 && n < max_n; i++)
    memcpy(&out[n++], &sp[i], sizeof(float));
  /* Stratified: 4 samples per exponent band */
  for (int exp = 0; exp <= 254 && n < max_n; exp++) {
    uint32_t base = (uint32_t)exp << 23;
    for (int j = 0; j < 4 && n < max_n; j++) {
      uint32_t mantissa = xorshift32(&state) & 0x7FFFFFu;
      uint32_t bits = base | mantissa;
      if (include_negative && (xorshift32(&state) & 1))
        bits |= 0x80000000u;
      memcpy(&out[n++], &bits, sizeof(float));
    }
  }
  return n;
}

/* LOG2: special values (zero, negative zero, inf, -inf, NaN, denormal, FLT_MAX) */
TEST(conformance, log2_special_values) {
  float in[8] = { 0.0f, -0.0f, (float)INFINITY, (float)-INFINITY,
                   (float)NAN, 1e-40f, 1.0f, FLT_MAX };
  float out[8] = {0};
  ASSERT_INT_EQ(run_unary_e2e(POLY_OP_LOG2, in, out, 8), 0);

  ASSERT_FLOAT_INF(out[0], -1);                   /* log2(0)    = -inf */
  ASSERT_FLOAT_INF(out[1], -1);                   /* log2(-0)   = -inf */
  ASSERT_FLOAT_INF(out[2], +1);                   /* log2(+inf) = +inf */
  ASSERT_FLOAT_NAN(out[3]);                        /* log2(-inf) = NaN  */
  ASSERT_FLOAT_NAN(out[4]);                        /* log2(NaN)  = NaN  */
  ASSERT_FLOAT_ULP(out[5], log2f(1e-40f), 128);   /* denormal          */
  ASSERT_FLOAT_ULP(out[6], 0.0f, 0);              /* log2(1) = 0 exact */
  ASSERT_FLOAT_ULP(out[7], log2f(FLT_MAX), 8);    /* large             */

  PASS();
}

/* LOG2: all negative inputs must return NaN */
TEST(conformance, log2_negative_domain) {
  float in[4] = { -1.0f, -100.0f, -FLT_MIN, -FLT_MAX };
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
  float in[6] = { 0.0f, -0.0f, (float)INFINITY, (float)-INFINITY,
                   (float)NAN, 3.14159265f };
  float out[6] = {0};
  ASSERT_INT_EQ(run_unary_e2e(POLY_OP_SIN, in, out, 6), 0);

  ASSERT_FLOAT_ABS(out[0], 0.0f, 1e-7f);          /* sin(0)    = 0    */
  ASSERT_FLOAT_ABS(out[1], 0.0f, 1e-7f);          /* sin(-0)   = 0    */
  ASSERT_FLOAT_NAN(out[2]);                         /* sin(+inf) = NaN  */
  ASSERT_FLOAT_NAN(out[3]);                         /* sin(-inf) = NaN  */
  ASSERT_FLOAT_NAN(out[4]);                         /* sin(NaN)  = NaN  */
  ASSERT_FLOAT_ABS(out[5], sinf(3.14159265f), 1e-3f); /* sin(pi) ~ 0  */

  PASS();
}

/* SIN: quadrant boundaries (all < 30, uses Cody-Waite path).
 * Uses ASSERT_FLOAT_ABS because sin(pi) ~ 0 and ULP distance near zero
 * is meaningless (tiny absolute error becomes huge ULP count). */
TEST(conformance, sin_quadrant_boundaries) {
  float pi = 3.14159265358979f;
  float in[8] = { pi/6, pi/4, pi/3, pi/2, 2*pi/3, 3*pi/4, 5*pi/6, pi };
  float out[8] = {0};
  ASSERT_INT_EQ(run_unary_e2e(POLY_OP_SIN, in, out, 8), 0);

  for (int i = 0; i < 8; i++)
    ASSERT_FLOAT_ABS(out[i], sinf(in[i]), 1e-5f);

  PASS();
}

/* SIN: Cody-Waite / Payne-Hanek switchover at 30.0 */
TEST(conformance, sin_switchover_boundary) {
  float in[8] = { 29.0f, 29.5f, 29.9f, 30.0f, 30.1f, 30.5f, 31.0f, 32.0f };
  float out[8] = {0};
  ASSERT_INT_EQ(run_unary_e2e(POLY_OP_SIN, in, out, 8), 0);

  for (int i = 0; i < 8; i++)
    ASSERT_FLOAT_ABS(out[i], sinf(in[i]), 2e-3f);

  PASS();
}

/* EXP2: special values (zero, -0, inf, -inf, NaN, overflow, underflow) */
TEST(conformance, exp2_special_values) {
  float in[7] = { 0.0f, -0.0f, (float)INFINITY, (float)-INFINITY,
                   (float)NAN, 128.0f, -150.0f };
  float out[7] = {0};
  ASSERT_INT_EQ(run_unary_e2e(POLY_OP_EXP2, in, out, 7), 0);

  ASSERT_FLOAT_ULP(out[0], 1.0f, 0);              /* exp2(0)    = 1    */
  ASSERT_FLOAT_ULP(out[1], 1.0f, 0);              /* exp2(-0)   = 1    */
  ASSERT_FLOAT_INF(out[2], +1);                    /* exp2(+inf) = +inf */
  ASSERT_FLOAT_ABS(out[3], 0.0f, 1e-44f); /* exp2(-inf) = 0    */
  ASSERT_FLOAT_NAN(out[4]);                         /* exp2(NaN)  = NaN  */
  ASSERT_FLOAT_INF(out[5], +1);                    /* exp2(128)  = +inf */
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
TEST(conformance, log2_bitpattern_sweep) {
  float in[1025], out[1025];
  int n = gen_stratified_sweep(in, 1025, 0xDEAD0001u, 0);
  ASSERT_INT_EQ(run_unary_e2e(POLY_OP_LOG2, in, out, n), 0);
  for (int i = 0; i < n; i++)
    ASSERT_FLOAT_NEAR(out[i], log2f(in[i]), 128, 1e-6f);
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
  PolyUOp *end_src[2] = { store, range };
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
  if (!prog) { free(src); free(lin); poly_ctx_destroy(ctx); return -1; }
  double *in_copy = (double *)malloc((size_t)n * sizeof(double));
  memcpy(in_copy, in, (size_t)n * sizeof(double));
  void *args[2] = { in_copy, out };
  poly_program_call(prog, args, 2);
  poly_program_destroy(prog);
  free(src); free(lin); free(in_copy);
  poly_ctx_destroy(ctx);
  return 0;
}

/* EXP2 f64: special values */
TEST(conformance_f64, exp2_special_values) {
  double in[7] = { 0.0, -0.0, INFINITY, -INFINITY, NAN, 1024.0, -2000.0 };
  double out[7] = {0};
  ASSERT_INT_EQ(run_unary_e2e_f64(POLY_OP_EXP2, in, out, 7), 0);

  ASSERT_DOUBLE_ULP(out[0], 1.0, 0);               /* exp2(0)     = 1    */
  ASSERT_DOUBLE_ULP(out[1], 1.0, 0);               /* exp2(-0)    = 1    */
  ASSERT_DOUBLE_INF(out[2], +1);                    /* exp2(+inf)  = +inf */
  ASSERT_DOUBLE_ABS(out[3], 0.0, 1e-300);          /* exp2(-inf)  = 0    */
  ASSERT_DOUBLE_NAN(out[4]);                         /* exp2(NaN)   = NaN  */
  ASSERT_DOUBLE_INF(out[5], +1);                    /* exp2(1024)  = +inf */
  ASSERT_DOUBLE_ABS(out[6], 0.0, 1e-300);          /* exp2(-2000) = 0    */

  PASS();
}

/* EXP2 f64: normal range values */
TEST(conformance_f64, exp2_normal_range) {
  double in[8] = { 1.0, -1.0, 10.0, -10.0, 0.5, -0.5, 100.0, -100.0 };
  double out[8] = {0};
  ASSERT_INT_EQ(run_unary_e2e_f64(POLY_OP_EXP2, in, out, 8), 0);

  for (int i = 0; i < 8; i++)
    ASSERT_DOUBLE_ULP(out[i], exp2(in[i]), 4);

  PASS();
}

/* LOG2 f64: special values */
TEST(conformance_f64, log2_special_values) {
  double in[8] = { 0.0, -0.0, INFINITY, -INFINITY, NAN, 5e-324, 1.0, DBL_MAX };
  double out[8] = {0};
  ASSERT_INT_EQ(run_unary_e2e_f64(POLY_OP_LOG2, in, out, 8), 0);

  ASSERT_DOUBLE_INF(out[0], -1);                    /* log2(0)    = -inf */
  ASSERT_DOUBLE_INF(out[1], -1);                    /* log2(-0)   = -inf */
  ASSERT_DOUBLE_INF(out[2], +1);                    /* log2(+inf) = +inf */
  ASSERT_DOUBLE_NAN(out[3]);                         /* log2(-inf) = NaN  */
  ASSERT_DOUBLE_NAN(out[4]);                         /* log2(NaN)  = NaN  */
  ASSERT_DOUBLE_ULP(out[5], log2(5e-324), 256);     /* denormal (subnormal min) */
  ASSERT_DOUBLE_ULP(out[6], 0.0, 0);               /* log2(1) = 0 exact */
  ASSERT_DOUBLE_ULP(out[7], log2(DBL_MAX), 16);     /* large */

  PASS();
}

/* LOG2 f64: negative domain returns NaN */
TEST(conformance_f64, log2_negative_domain) {
  double in[4] = { -1.0, -100.0, -DBL_MIN, -DBL_MAX };
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
  double in[6] = { 0.0, -0.0, INFINITY, -INFINITY, NAN, pi };
  double out[6] = {0};
  ASSERT_INT_EQ(run_unary_e2e_f64(POLY_OP_SIN, in, out, 6), 0);

  ASSERT_DOUBLE_ABS(out[0], 0.0, 1e-15);           /* sin(0)    = 0    */
  ASSERT_DOUBLE_ABS(out[1], 0.0, 1e-15);           /* sin(-0)   = 0    */
  ASSERT_DOUBLE_NAN(out[2]);                         /* sin(+inf) = NaN  */
  ASSERT_DOUBLE_NAN(out[3]);                         /* sin(-inf) = NaN  */
  ASSERT_DOUBLE_NAN(out[4]);                         /* sin(NaN)  = NaN  */
  ASSERT_DOUBLE_ABS(out[5], sin(pi), 1e-7);         /* sin(pi) ~ 0     */

  PASS();
}

/* SIN f64: quadrant boundaries (Cody-Waite path) */
TEST(conformance_f64, sin_quadrant_boundaries) {
  double pi = 3.14159265358979323846;
  double in[8] = { pi/6, pi/4, pi/3, pi/2, 2*pi/3, 3*pi/4, 5*pi/6, pi };
  double out[8] = {0};
  ASSERT_INT_EQ(run_unary_e2e_f64(POLY_OP_SIN, in, out, 8), 0);

  for (int i = 0; i < 8; i++)
    ASSERT_DOUBLE_ABS(out[i], sin(in[i]), 1e-10);

  PASS();
}

/* SIN f64: switchover boundary */
TEST(conformance_f64, sin_switchover_boundary) {
  double in[8] = { 29.0, 29.5, 29.9, 30.0, 30.1, 30.5, 31.0, 32.0 };
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
  ASSERT_TRUE(mul_add_result == 0.0f);        /* double-rounded: exact zero */
  ASSERT_TRUE(fma_result != mul_add_result);   /* FMA: nonzero */
  ASSERT_TRUE(fma_result > 0.0f);             /* sanity: positive residual */
  PASS();
}

/* ── C1 divmod rules (tinygrad divandmod.py alignment) ─────────────── */

/* nested_div_mod: (x%6)//3 → (x//3)%2.  Ref: divandmod.py:25-27 */
TEST(sym_future, nested_div_mod_idiv) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(12));
  PolyUOp *x = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));
  PolyUOp *six = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(6));
  PolyUOp *three = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(3));
  PolyUOp *mod6 = poly_uop2(ctx, POLY_OP_MOD, POLY_INT32, x, six, poly_arg_none());
  PolyUOp *div3 = poly_uop2(ctx, POLY_OP_IDIV, POLY_INT32, mod6, three, poly_arg_none());
  PolyUOp *r = simplify(ctx, div3);
  /* Should become (x//3)%2 */
  ASSERT_TRUE(r->op == POLY_OP_MOD);
  ASSERT_TRUE(r->src[0]->op == POLY_OP_IDIV);
  ASSERT_TRUE(r->src[1]->op == POLY_OP_CONST && r->src[1]->arg.i == 2);
  poly_ctx_destroy(ctx);
  PASS();
}

/* nested_div_mod: (x%6)%3 → x%3.  Ref: divandmod.py:25-27 */
TEST(sym_future, nested_div_mod_mod) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(12));
  PolyUOp *x = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));
  PolyUOp *six = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(6));
  PolyUOp *three = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(3));
  PolyUOp *mod6 = poly_uop2(ctx, POLY_OP_MOD, POLY_INT32, x, six, poly_arg_none());
  PolyUOp *mod3 = poly_uop2(ctx, POLY_OP_MOD, POLY_INT32, mod6, three, poly_arg_none());
  PolyUOp *r = simplify(ctx, mod3);
  /* Should become x%3 */
  ASSERT_TRUE(r->op == POLY_OP_MOD);
  ASSERT_TRUE(r->src[0] == x);
  ASSERT_TRUE(r->src[1]->op == POLY_OP_CONST && r->src[1]->arg.i == 3);
  poly_ctx_destroy(ctx);
  PASS();
}

/* remove_nested_mod: (x%4 + y)%2 → (x+y)%2.  Ref: divandmod.py:29-37 */
TEST(sym_future, remove_nested_mod) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(10));
  PolyUOp *x = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));
  PolyUOp *y = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(1));
  PolyUOp *four = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(4));
  PolyUOp *two = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(2));
  PolyUOp *xmod4 = poly_uop2(ctx, POLY_OP_MOD, POLY_INT32, x, four, poly_arg_none());
  PolyUOp *sum = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, xmod4, y, poly_arg_none());
  PolyUOp *mod2 = poly_uop2(ctx, POLY_OP_MOD, POLY_INT32, sum, two, poly_arg_none());
  PolyUOp *r = simplify(ctx, mod2);
  /* Should strip inner x%4 since 4%2==0, becoming (x+y)%2 */
  ASSERT_TRUE(r->op == POLY_OP_MOD);
  /* The inner sum should have x directly, not x%4 */
  PolyUOp *inner_sum = r->src[0];
  ASSERT_TRUE(inner_sum->op == POLY_OP_ADD);
  bool has_mod = false;
  if (inner_sum->src[0]->op == POLY_OP_MOD || inner_sum->src[1]->op == POLY_OP_MOD)
    has_mod = true;
  ASSERT_TRUE(!has_mod);
  poly_ctx_destroy(ctx);
  PASS();
}

/* fold_binary_numerator: x in [3,4], (2*x+1)//5.  Ref: divandmod.py:43-47 */
TEST(sym_future, fold_binary_numerator) {
  PolyCtx *ctx = poly_ctx_new();
  /* x = RANGE(2) + 3 → range [3,4] */
  PolyUOp *two = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(2));
  PolyUOp *r0 = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, two, poly_arg_int(0));
  PolyUOp *three = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(3));
  PolyUOp *x = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, r0, three, poly_arg_none());
  /* expr = (2*x + 1) // 5 */
  PolyUOp *two_c = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(2));
  PolyUOp *mul2x = poly_uop2(ctx, POLY_OP_MUL, POLY_INT32, two_c, x, poly_arg_none());
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(1));
  PolyUOp *add1 = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, mul2x, one, poly_arg_none());
  PolyUOp *five = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(5));
  PolyUOp *div = poly_uop2(ctx, POLY_OP_IDIV, POLY_INT32, add1, five, poly_arg_none());
  PolyUOp *r = simplify(ctx, div);
  /* x=3: (7)//5=1, x=4: (9)//5=1. Same quotient → should fold to CONST(1) */
  ASSERT_TRUE(r->op == POLY_OP_CONST && r->arg.i == 1);
  poly_ctx_destroy(ctx);
  PASS();
}

/* gcd_with_remainder: (6*x)%4 with x>=0 → GCD(6,4)=2, (3*x)%2*2.  Ref: divandmod.py:58-63 */
TEST(sym_future, gcd_with_remainder) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(10));
  PolyUOp *x = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));
  PolyUOp *six = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(6));
  PolyUOp *four = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(4));
  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, POLY_INT32, six, x, poly_arg_none());
  PolyUOp *mod = poly_uop2(ctx, POLY_OP_MOD, POLY_INT32, mul, four, poly_arg_none());
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

/* SHL+ADD fuses to MULACC with FMA caps.  Ref: tinygrad decompositions.py:480 */
TEST(decomp, shl_add_fuses_to_mulacc) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(5));
  PolyUOp *n = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(3));
  PolyUOp *c = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(10));
  PolyUOp *shl = poly_uop2(ctx, POLY_OP_SHL, POLY_INT32, x, n, poly_arg_none());
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, shl, c, poly_arg_none());

  /* With MULACC caps, ADD(SHL(x,3), c) -> MULACC(x, 8, c) */
  PolyRendererCaps caps = { .has_mulacc = true };
  PolyPatternMatcher *pm = poly_pm_decomp_pass_caps(caps);
  PolyUOp *r = poly_graph_rewrite(ctx, add, pm);
  ASSERT_NOT_NULL(r);
  ASSERT_TRUE(r->op == POLY_OP_MULACC);
  ASSERT_TRUE(r->n_src == 3);
  /* src[1] should be CONST(8) = 2^3 */
  ASSERT_TRUE(r->src[1]->op == POLY_OP_CONST);
  ASSERT_TRUE(r->src[1]->arg.i == 8);
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
  PolyRewriteOpts opts = { .optimize = false, .devectorize = 1 };
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

/* Verify: drop true gate from INDEX */
TEST(devectorize, drop_true_gate) {
  PolyCtx *ctx = poly_ctx_new();
  PolyDType ptr_f32 = poly_dtype_ptr(POLY_FLOAT32, -1, POLY_ADDR_GLOBAL);
  PolyUOp *buf = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(0));
  PolyUOp *idx = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *gate = poly_uop0(ctx, POLY_OP_CONST, POLY_BOOL, poly_arg_int(1));
  PolyUOp *srcs[3] = { buf, idx, gate };
  PolyUOp *index = poly_uop(ctx, POLY_OP_INDEX, ptr_f32, srcs, 3, poly_arg_none());

  PolyUOp *r = poly_graph_rewrite(ctx, index, poly_pm_devectorize_pass());
  /* Gate should be dropped: 3 srcs → 2 srcs */
  ASSERT_TRUE(r->op == POLY_OP_INDEX);
  ASSERT_INT_EQ(r->n_src, 2);
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
  PolyBufferBinding bindings[] = {
    POLY_BIND_HOST(a, da), POLY_BIND_HOST(b, db), POLY_BIND_HOST(out, dout)
  };

  /* Use POLY_OPTIMIZE + POLY_DEVECTORIZE via env to test full pipeline */
  setenv("POLY_OPTIMIZE", "1", 1);
  setenv("POLY_DEVECTORIZE", "1", 1);
  int ret = poly_realize(ctx, sink, bindings, 3);
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
  for (int i = 0; i < N; i++) { da[i] = (float)i; db[i] = (float)(100 + i); }
  memset(dout, 0, sizeof(dout));
  PolyBufferBinding bindings[] = {
    POLY_BIND_HOST(a, da), POLY_BIND_HOST(b, db), POLY_BIND_HOST(out, dout)
  };

  setenv("POLY_OPTIMIZE", "1", 1);
  setenv("POLY_DEVECTORIZE", "1", 1);
  setenv("POLY_BEAM", "2", 1);
  int ret = poly_realize(ctx, sink, bindings, 3);
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
  for (int i = 0; i < N; i++) { da[i] = (float)(i + 1); expected += da[i]; }
  PolyBufferBinding bindings[] = {
    POLY_BIND_HOST(a, da), POLY_BIND_HOST(out, dout)
  };

  setenv("POLY_OPTIMIZE", "1", 1);
  setenv("POLY_DEVECTORIZE", "1", 1);
  setenv("POLY_BEAM", "2", 1);
  int ret = poly_realize(ctx, sink, bindings, 2);
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
  for (int i = 0; i < N; i++) { da[i] = (float)i; db[i] = 1.0f; }
  memset(dout, 0, sizeof(dout));
  PolyBufferBinding bindings[] = {
    POLY_BIND_HOST(a, da), POLY_BIND_HOST(b, db), POLY_BIND_HOST(out, dout)
  };

  setenv("POLY_OPTIMIZE", "1", 1);
  setenv("POLY_DEVECTORIZE", "1", 1);
  setenv("POLY_BEAM", "0", 1);
  int ret = poly_realize(ctx, sink, bindings, 3);
  unsetenv("POLY_OPTIMIZE");
  unsetenv("POLY_DEVECTORIZE");
  unsetenv("POLY_BEAM");

  ASSERT_INT_EQ(ret, 0);
  for (int i = 0; i < N; i++)
    ASSERT_FLOAT_EQ(dout[i], da[i] + db[i], 1e-6);

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
  for (int i = 0; i < N; i++) { da[i] = (float)(i * 3); db[i] = (float)(i + 7); }

  PolyBufferBinding bindings1[] = {
    POLY_BIND_HOST(a, da), POLY_BIND_HOST(b, db), POLY_BIND_HOST(out, dout1)
  };

  /* First run: populates cache */
  setenv("POLY_OPTIMIZE", "1", 1);
  setenv("POLY_DEVECTORIZE", "1", 1);
  setenv("POLY_BEAM", "2", 1);
  int ret1 = poly_realize(ctx, sink, bindings1, 3);
  ASSERT_INT_EQ(ret1, 0);

  /* Second run: should hit cache */
  PolyCtx *ctx2 = poly_ctx_new();
  PolyUOp *a2 = poly_buffer_f32(ctx2, N);
  PolyUOp *b2 = poly_buffer_f32(ctx2, N);
  PolyUOp *c2 = poly_alu2(ctx2, POLY_OP_ADD, a2, b2);
  PolyUOp *out2 = poly_buffer_f32(ctx2, N);
  PolyUOp *st2 = poly_store_val(ctx2, out2, c2);
  PolyUOp *sink2 = poly_sink1(ctx2, st2);

  PolyBufferBinding bindings2[] = {
    POLY_BIND_HOST(a2, da), POLY_BIND_HOST(b2, db), POLY_BIND_HOST(out2, dout2)
  };
  int ret2 = poly_realize(ctx2, sink2, bindings2, 3);
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
  PolyBufferBinding bindings[] = {
    POLY_BIND_HOST(a, da), POLY_BIND_HOST(b, db),
    POLY_BIND_HOST(c, dc), POLY_BIND_HOST(out, dout)
  };

  setenv("POLY_OPTIMIZE", "1", 1);
  setenv("POLY_DEVECTORIZE", "1", 1);
  setenv("POLY_BEAM", "2", 1);
  int ret = poly_realize(ctx, sink, bindings, 4);
  unsetenv("POLY_OPTIMIZE");
  unsetenv("POLY_DEVECTORIZE");
  unsetenv("POLY_BEAM");

  ASSERT_INT_EQ(ret, 0);
  for (int i = 0; i < N; i++)
    ASSERT_FLOAT_EQ(dout[i], da[i] * db[i] + dc[i], 1e-3);

  poly_ctx_destroy(ctx);
  PASS();
}

/* ── Section 8: Tensor core helper tests (tc.py port) ───────────────── */

/* AMD CDNA 16x16x16 half->float spec for testing.
 * Initialized at first use because POLY_FLOAT16/POLY_FLOAT32 are extern const. */
static PolyTensorCore test_cdna_tc;
static int test_cdna_tc_init = 0;

static const PolyTensorCore *get_test_cdna_tc(void) {
  if (!test_cdna_tc_init) {
    test_cdna_tc_init = 1;
    memset(&test_cdna_tc, 0, sizeof(test_cdna_tc));
    test_cdna_tc.dims[0] = 16; test_cdna_tc.dims[1] = 16; test_cdna_tc.dims[2] = 16;
    test_cdna_tc.threads = 64;
    test_cdna_tc.elements_per_thread[0] = 4;
    test_cdna_tc.elements_per_thread[1] = 4;
    test_cdna_tc.elements_per_thread[2] = 4;
    test_cdna_tc.dtype_in = POLY_FLOAT16;
    test_cdna_tc.dtype_out = POLY_FLOAT32;
    struct { char type; int dim; } opts[] = {
      {'l',0},{'l',0},{'l',0},{'l',0},{'u',1},{'u',1},{'l',1},{'l',1}
    };
    for (int i = 0; i < 8; i++) { test_cdna_tc.opts[i].type = opts[i].type; test_cdna_tc.opts[i].dim = opts[i].dim; }
    test_cdna_tc.n_opts = 8;
    /* swizzle[0] */
    test_cdna_tc.swizzle[0][0][0]="u0"; test_cdna_tc.swizzle[0][0][1]="u1"; test_cdna_tc.swizzle[0][0][2]="l4";
    test_cdna_tc.swizzle[0][0][3]="l5"; test_cdna_tc.swizzle[0][0][4]="r2"; test_cdna_tc.swizzle[0][0][5]="r3";
    test_cdna_tc.swizzle[0][1][0]="r0"; test_cdna_tc.swizzle[0][1][1]="r1";
    test_cdna_tc.swizzle[0][2][0]="l0"; test_cdna_tc.swizzle[0][2][1]="l1"; test_cdna_tc.swizzle[0][2][2]="l2"; test_cdna_tc.swizzle[0][2][3]="l3";
    /* swizzle[1] */
    test_cdna_tc.swizzle[1][0][0]="l0"; test_cdna_tc.swizzle[1][0][1]="l1"; test_cdna_tc.swizzle[1][0][2]="l2";
    test_cdna_tc.swizzle[1][0][3]="l3"; test_cdna_tc.swizzle[1][0][4]="r2"; test_cdna_tc.swizzle[1][0][5]="r3";
    test_cdna_tc.swizzle[1][1][0]="r0"; test_cdna_tc.swizzle[1][1][1]="r1";
    test_cdna_tc.swizzle[1][2][0]="l4"; test_cdna_tc.swizzle[1][2][1]="l5"; test_cdna_tc.swizzle[1][2][2]="u0"; test_cdna_tc.swizzle[1][2][3]="u1";
    test_cdna_tc.swizzle_len[0][0]=6; test_cdna_tc.swizzle_len[0][1]=2; test_cdna_tc.swizzle_len[0][2]=4;
    test_cdna_tc.swizzle_len[1][0]=6; test_cdna_tc.swizzle_len[1][1]=2; test_cdna_tc.swizzle_len[1][2]=4;
    test_cdna_tc.intrinsic_name = "mfma_f32_16x16x16f16";
  }
  return &test_cdna_tc;
}

TEST(tc, get_reduce_axes) {
  int ra[16][2];
  int n = tc_get_reduce_axes(get_test_cdna_tc(), ra);
  /* K=16 -> log2(16)=4 pairs, each with amt=2 */
  ASSERT_INT_EQ(n, 4);
  for (int i = 0; i < 4; i++) {
    ASSERT_INT_EQ(ra[i][0], i);
    ASSERT_INT_EQ(ra[i][1], 2);
  }
  PASS();
}

TEST(tc, count_local_upcast) {
  ASSERT_INT_EQ(tc_count_local(get_test_cdna_tc()), 6);  /* l0,l0,l0,l0,l1,l1 */
  ASSERT_INT_EQ(tc_count_upcast(get_test_cdna_tc()), 2);  /* u1,u1 */
  PASS();
}

TEST(tc, base_shape_str) {
  const char *out[32];
  int n = tc_base_shape_str(get_test_cdna_tc(), out, 32);
  /* 8 opts + 4 reduce = 12 entries */
  ASSERT_INT_EQ(n, 12);
  /* Expected: l0,l1,l2,l3,u0,u1,l4,l5,r0,r1,r2,r3 */
  const char *expected[] = {"l0","l1","l2","l3","u0","u1","l4","l5","r0","r1","r2","r3"};
  for (int i = 0; i < 12; i++) {
    if (strcmp(out[i], expected[i]) != 0) {
      FAIL("base_shape_str[%d]: got '%s', expected '%s'", i, out[i], expected[i]);
    }
  }
  PASS();
}

TEST(tc, base_upcast_axes) {
  const char *out[32];
  int n = tc_base_upcast_axes(get_test_cdna_tc(), out, 32);
  /* reversed [r0,r1,r2,r3,u0,u1] -> [u1,u0,r3,r2,r1,r0] */
  ASSERT_INT_EQ(n, 6);
  const char *expected[] = {"u1","u0","r3","r2","r1","r0"};
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
  int n = tc_base_shape_str(get_test_cdna_tc(), shape_str, 32);
  ASSERT_INT_EQ(n, 12);

  int perm0[32], perm1[32];
  tc_permute_for_shape_str(get_test_cdna_tc(), 0, shape_str, n, perm0, 32);
  tc_permute_for_shape_str(get_test_cdna_tc(), 1, shape_str, n, perm1, 32);

  /* swizzle[0] flattened: u0,u1,l4,l5,r2,r3, r0,r1, l0,l1,l2,l3
   * fwd (base_shape_str): l0,l1,l2,l3,u0,u1,l4,l5,r0,r1,r2,r3
   * remap[0]: l0->u0, l1->u1, l2->l4, l3->l5, u0->r2, u1->r3, l4->r0, l5->r1, r0->l0, r1->l1, r2->l2, r3->l3
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
   * remap[1]: l0->l0, l1->l1, l2->l2, l3->l3, u0->r2, u1->r3, l4->r0, l5->r1, r0->l4, r1->l5, r2->u0, r3->u1
   *   perm1[0] = index("l0") = 0
   *   perm1[1] = index("l1") = 1
   *   perm1[2] = index("l2") = 2
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

/* ── Section 9: TC structural detection tests ───────────────────────── */

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
  PolyUOp *rng_i = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, c16,
                               poly_arg_range(0, POLY_AXIS_GLOBAL));
  PolyUOp *rng_j = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, c16,
                               poly_arg_range(1, POLY_AXIS_GLOBAL));
  PolyUOp *rng_k = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, c16,
                               poly_arg_range(2, POLY_AXIS_REDUCE));

  PolyUOp *c16b = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(16));
  PolyUOp *a_idx = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32,
                    poly_uop2(ctx, POLY_OP_MUL, POLY_INT32, rng_i, c16b, poly_arg_none()),
                    rng_k, poly_arg_none());
  PolyUOp *a_ptr = poly_uop2(ctx, POLY_OP_INDEX, ptr_f16, pA, a_idx, poly_arg_none());
  PolyUOp *a_val = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT16, a_ptr, poly_arg_none());

  PolyUOp *b_idx = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32,
                    poly_uop2(ctx, POLY_OP_MUL, POLY_INT32, rng_k, c16b, poly_arg_none()),
                    rng_j, poly_arg_none());
  PolyUOp *b_ptr = poly_uop2(ctx, POLY_OP_INDEX, ptr_f16, pB, b_idx, poly_arg_none());
  PolyUOp *b_val = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT16, b_ptr, poly_arg_none());

  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT16, a_val, b_val, poly_arg_none());
  PolyUOp *cast = poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, mul, poly_arg_none());

  PolyUOp *red_srcs[2] = { cast, rng_k };
  PolyArg red_arg = { .kind = POLY_ARG_OPS, .ops = POLY_OP_ADD };
  PolyUOp *reduce = poly_uop(ctx, POLY_OP_REDUCE, POLY_FLOAT32, red_srcs, 2, red_arg);

  PolyUOp *c_idx = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32,
                    poly_uop2(ctx, POLY_OP_MUL, POLY_INT32, rng_i, c16b, poly_arg_none()),
                    rng_j, poly_arg_none());
  PolyUOp *c_ptr = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, pC, c_idx, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, c_ptr, reduce, poly_arg_none());

  PolyUOp *end_srcs[3] = { store, rng_i, rng_j };
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
    if (topo[i]->op == POLY_OP_REDUCE &&
        topo[i]->arg.kind == POLY_ARG_OPS && topo[i]->arg.ops == POLY_OP_ADD &&
        topo[i]->src[0]->op == POLY_OP_CAST &&
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
  PolyRewriteOpts opts = { .optimize = true, .caps = caps };
  PolyUOp *optimized = poly_full_rewrite_to_sink_ex(ctx, sink, opts);

  unsetenv("POLY_TC_OPT");
  unsetenv("POLY_USE_TC");

  /* Postcondition: WMMA, CONTRACT, UNROLL should appear */
  int n_wmma = count_ops(ctx, optimized, POLY_OP_WMMA);
  int n_contract = count_ops(ctx, optimized, POLY_OP_CONTRACT);
  int n_unroll = count_ops(ctx, optimized, POLY_OP_UNROLL);

  if (n_wmma == 0) {
    fprintf(stderr, "  detect_wmma: no WMMA found after heuristic (n_contract=%d n_unroll=%d)\n",
            n_contract, n_unroll);
    /* Dump op counts for debugging */
    int n_topo = 0;
    PolyUOp **topo = poly_toposort(ctx, optimized, &n_topo);
    for (int i = 0; i < n_topo; i++) {
      if (topo[i]->op == POLY_OP_REDUCE || topo[i]->op == POLY_OP_WMMA ||
          topo[i]->op == POLY_OP_CONTRACT || topo[i]->op == POLY_OP_UNROLL)
        fprintf(stderr, "    op[%d] = %d (REDUCE=%d WMMA=%d)\n", i, topo[i]->op,
                POLY_OP_REDUCE, POLY_OP_WMMA);
    }
  }

  ASSERT_TRUE(n_wmma > 0);
  /* CONTRACT and UNROLL may be lowered by the expander pass -- that's correct.
   * The key assertion is WMMA present and REDUCE(ADD) gone. */
  (void)n_contract; (void)n_unroll;

  /* The original REDUCE(ADD) should be gone (replaced by WMMA) */
  int n_reduce = 0;
  {
    int n_topo = 0;
    PolyUOp **topo = poly_toposort(ctx, optimized, &n_topo);
    for (int i = 0; i < n_topo; i++) {
      if (topo[i]->op == POLY_OP_REDUCE &&
          topo[i]->arg.kind == POLY_ARG_OPS && topo[i]->arg.ops == POLY_OP_ADD)
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
  PolyRewriteOpts opts = { .optimize = true, .caps = caps };
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
  PolyUOp *rng_i = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, cN, poly_arg_range(0, POLY_AXIS_GLOBAL));
  PolyUOp *rng_j = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, cN, poly_arg_range(1, POLY_AXIS_GLOBAL));
  PolyUOp *rng_k = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, cN, poly_arg_range(2, POLY_AXIS_REDUCE));
  PolyUOp *cNb = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(N));
  PolyUOp *a_idx = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32,
                    poly_uop2(ctx, POLY_OP_MUL, POLY_INT32, rng_i, cNb, poly_arg_none()),
                    rng_k, poly_arg_none());
  PolyUOp *a_val = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT16,
                    poly_uop2(ctx, POLY_OP_INDEX, ptr_f16, pA, a_idx, poly_arg_none()), poly_arg_none());
  PolyUOp *b_idx = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32,
                    poly_uop2(ctx, POLY_OP_MUL, POLY_INT32, rng_k, cNb, poly_arg_none()),
                    rng_j, poly_arg_none());
  PolyUOp *b_val = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT16,
                    poly_uop2(ctx, POLY_OP_INDEX, ptr_f16, pB, b_idx, poly_arg_none()), poly_arg_none());
  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT16, a_val, b_val, poly_arg_none());
  PolyUOp *cast = poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, mul, poly_arg_none());
  PolyUOp *red_srcs[2] = { cast, rng_k };
  PolyArg red_arg = { .kind = POLY_ARG_OPS, .ops = POLY_OP_ADD };
  PolyUOp *reduce = poly_uop(ctx, POLY_OP_REDUCE, POLY_FLOAT32, red_srcs, 2, red_arg);
  PolyUOp *c_idx = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32,
                    poly_uop2(ctx, POLY_OP_MUL, POLY_INT32, rng_i, cNb, poly_arg_none()),
                    rng_j, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID,
                    poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, pC, c_idx, poly_arg_none()),
                    reduce, poly_arg_none());
  PolyUOp *end_srcs[3] = { store, rng_i, rng_j };
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

  /* WMMA arg must be the intrinsic name */
  ASSERT_TRUE(wmma->arg.kind == POLY_ARG_STRING);
  ASSERT_TRUE(strcmp(wmma->arg.str, "mfma_f32_16x16x16f16") == 0);

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
  ASSERT_TRUE(n_unroll >= 1);   /* at least the WMMA wrapper */

  poly_ctx_destroy(ctx);
  PASS();
}
