/*
 * test_codegen.c — Tests for linearizer, C renderer, and CPU runtime
 */

#include "test_harness.h"
#include "../src/codegen.h"

/* ── Helper: build c[i] = a[i] OP b[i] kernel IR ────────────────────── */

typedef struct {
  PolyCtx *ctx;
  PolyUOp *sink;
  int n;         /* loop bound */
} VecKernel;

static VecKernel make_vec_binop(PolyOps alu_op, int n) {
  PolyCtx *ctx = poly_ctx_new();
  PolyDType ptr_f32 = poly_dtype_ptr(POLY_FLOAT32, -1, POLY_ADDR_GLOBAL);

  /* PARAM: buffer pointers */
  PolyUOp *p0 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(0));
  PolyUOp *p1 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(1));
  PolyUOp *p2 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(2));

  /* loop: for ridx0 in range(n) */
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(n));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));

  /* index into buffers */
  PolyUOp *idx0 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p0, range, poly_arg_none());
  PolyUOp *idx1 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p1, range, poly_arg_none());
  PolyUOp *idx2 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p2, range, poly_arg_none());

  /* load */
  PolyUOp *load0 = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx0, poly_arg_none());
  PolyUOp *load1 = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx1, poly_arg_none());

  /* ALU */
  PolyUOp *alu = poly_uop2(ctx, alu_op, POLY_FLOAT32, load0, load1, poly_arg_none());

  /* store */
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx2, alu, poly_arg_none());

  /* end loop + sink */
  PolyUOp *end_src[2] = { store, range };
  PolyUOp *end  = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());

  return (VecKernel){ ctx, sink, n };
}

static int count_lin_ops(PolyUOp **lin, int n, PolyOps op) {
  int c = 0;
  for (int i = 0; i < n; i++) if (lin[i]->op == op) c++;
  return c;
}

/* ── Linearizer tests ────────────────────────────────────────────────── */

TEST(codegen, linearize_order) {
  VecKernel k = make_vec_binop(POLY_OP_ADD, 10);
  int n;
  PolyUOp **lin = poly_linearize(k.ctx, k.sink, &n);

  /* PARAMs should come first (priority -20) */
  ASSERT_TRUE(n >= 6);
  ASSERT_TRUE(lin[0]->op == POLY_OP_PARAM);
  ASSERT_TRUE(lin[1]->op == POLY_OP_PARAM);
  ASSERT_TRUE(lin[2]->op == POLY_OP_PARAM);

  /* SINK should be last */
  ASSERT_TRUE(lin[n-1]->op == POLY_OP_SINK);

  /* END should come just before SINK */
  ASSERT_TRUE(lin[n-2]->op == POLY_OP_END);

  free(lin);
  poly_ctx_destroy(k.ctx);
  PASS();
}

TEST(codegen, linearize_deps) {
  /* Verify: every UOp's sources appear before it in the linearized list */
  VecKernel k = make_vec_binop(POLY_OP_ADD, 10);
  int n;
  PolyUOp **lin = poly_linearize(k.ctx, k.sink, &n);

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < lin[i]->n_src; j++) {
      PolyUOp *src = lin[i]->src[j];
      bool found = false;
      for (int k = 0; k < i; k++) {
        if (lin[k] == src) { found = true; break; }
      }
      ASSERT_TRUE(found);
    }
  }

  free(lin);
  poly_ctx_destroy(k.ctx);
  PASS();
}

TEST(codegen, reduce_merge_shared_end) {
  /* Two REDUCE ops over the same RANGE should share one merged END chain. */
  PolyCtx *ctx = poly_ctx_new();
  PolyDType ptr_f32 = poly_dtype_ptr(POLY_FLOAT32, -1, POLY_ADDR_GLOBAL);

  PolyUOp *pin = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(0));
  PolyUOp *pout0 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(1));
  PolyUOp *pout1 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(2));

  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(8));
  PolyUOp *r0 = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));

  PolyUOp *in_idx = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, pin, r0, poly_arg_none());
  PolyUOp *in_ld = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, in_idx, poly_arg_none());

  PolyUOp *red0_srcs[2] = { in_ld, r0 };
  PolyUOp *red1_srcs[2] = { in_ld, r0 };
  PolyUOp *sum = poly_uop(ctx, POLY_OP_REDUCE, POLY_FLOAT32, red0_srcs, 2, poly_arg_ops(POLY_OP_ADD));
  PolyUOp *mx = poly_uop(ctx, POLY_OP_REDUCE, POLY_FLOAT32, red1_srcs, 2, poly_arg_ops(POLY_OP_MAX));

  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *out0_idx = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, pout0, zero, poly_arg_none());
  PolyUOp *out1_idx = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, pout1, zero, poly_arg_none());
  PolyUOp *st0 = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out0_idx, sum, poly_arg_none());
  PolyUOp *st1 = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out1_idx, mx, poly_arg_none());
  PolyUOp *stores[2] = { st0, st1 };
  PolyUOp *sink = poly_uop(ctx, POLY_OP_SINK, POLY_VOID, stores, 2, poly_arg_none());

  int n = 0;
  PolyUOp **lin = poly_linearize(ctx, sink, &n);
  ASSERT_TRUE(n > 0);
  ASSERT_INT_EQ(count_lin_ops(lin, n, POLY_OP_DEFINE_REG), 2);
  ASSERT_INT_EQ(count_lin_ops(lin, n, POLY_OP_END), 1);

  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

/* ── Renderer tests ──────────────────────────────────────────────────── */

TEST(codegen, render_vecadd) {
  VecKernel k = make_vec_binop(POLY_OP_ADD, 10);
  int n;
  PolyUOp **lin = poly_linearize(k.ctx, k.sink, &n);
  char *src = poly_render_c(lin, n, "vecadd");

  /* check key substrings in generated C */
  ASSERT_NOT_NULL(strstr(src, "void vecadd("));
  ASSERT_NOT_NULL(strstr(src, "float* restrict data0"));
  ASSERT_NOT_NULL(strstr(src, "float* restrict data1"));
  ASSERT_NOT_NULL(strstr(src, "float* restrict data2"));
  ASSERT_NOT_NULL(strstr(src, "for (int ridx0 = 0; ridx0 < 10; ridx0++)"));
  ASSERT_NOT_NULL(strstr(src, "float val0"));
  ASSERT_NOT_NULL(strstr(src, "float val1"));
  ASSERT_NOT_NULL(strstr(src, "float alu0"));
  /* wrapper function */
  ASSERT_NOT_NULL(strstr(src, "void vecadd_call(void **args)"));

  free(src);
  free(lin);
  poly_ctx_destroy(k.ctx);
  PASS();
}

TEST(codegen, render_vecmul) {
  VecKernel k = make_vec_binop(POLY_OP_MUL, 8);
  int n;
  PolyUOp **lin = poly_linearize(k.ctx, k.sink, &n);
  char *src = poly_render_c(lin, n, "vecmul");

  /* multiply uses * operator */
  ASSERT_NOT_NULL(strstr(src, "void vecmul("));
  ASSERT_NOT_NULL(strstr(src, "ridx0 < 8"));

  free(src);
  free(lin);
  poly_ctx_destroy(k.ctx);
  PASS();
}

/* ── End-to-end tests ────────────────────────────────────────────────── */

TEST(codegen, e2e_vecadd) {
  /* c[i] = a[i] + b[i] for i in 0..9 */
  VecKernel k = make_vec_binop(POLY_OP_ADD, 10);
  int n;
  PolyUOp **lin = poly_linearize(k.ctx, k.sink, &n);
  char *src = poly_render_c(lin, n, "vecadd");

  PolyProgram *prog = poly_compile_c(src, "vecadd");
  ASSERT_NOT_NULL(prog);

  float a[10], b[10], c[10];
  for (int i = 0; i < 10; i++) {
    a[i] = (float)(i + 1);         /* 1, 2, ..., 10 */
    b[i] = (float)((i + 1) * 10);  /* 10, 20, ..., 100 */
    c[i] = 0.0f;
  }

  void *args[3] = { a, b, c };
  poly_program_call(prog, args, 3);

  for (int i = 0; i < 10; i++) {
    ASSERT_FLOAT_EQ(c[i], a[i] + b[i], 1e-6);
  }

  poly_program_destroy(prog);
  free(src);
  free(lin);
  poly_ctx_destroy(k.ctx);
  PASS();
}

TEST(codegen, e2e_vecmul) {
  /* c[i] = a[i] * b[i] for i in 0..7 */
  VecKernel k = make_vec_binop(POLY_OP_MUL, 8);
  int n;
  PolyUOp **lin = poly_linearize(k.ctx, k.sink, &n);
  char *src = poly_render_c(lin, n, "vecmul");

  PolyProgram *prog = poly_compile_c(src, "vecmul");
  ASSERT_NOT_NULL(prog);

  float a[8], b[8], c[8];
  for (int i = 0; i < 8; i++) {
    a[i] = (float)(i + 1);
    b[i] = 0.5f;
    c[i] = 0.0f;
  }

  void *args[3] = { a, b, c };
  poly_program_call(prog, args, 3);

  for (int i = 0; i < 8; i++) {
    ASSERT_FLOAT_EQ(c[i], a[i] * 0.5f, 1e-6);
  }

  poly_program_destroy(prog);
  free(src);
  free(lin);
  poly_ctx_destroy(k.ctx);
  PASS();
}

TEST(codegen, e2e_vecsub) {
  /* c[i] = a[i] - b[i] for i in 0..3 */
  VecKernel k = make_vec_binop(POLY_OP_SUB, 4);
  int n;
  PolyUOp **lin = poly_linearize(k.ctx, k.sink, &n);
  char *src = poly_render_c(lin, n, "vecsub");

  PolyProgram *prog = poly_compile_c(src, "vecsub");
  ASSERT_NOT_NULL(prog);

  float a[4] = { 10, 20, 30, 40 };
  float b[4] = { 1, 2, 3, 4 };
  float c[4] = { 0 };

  void *args[3] = { a, b, c };
  poly_program_call(prog, args, 3);

  for (int i = 0; i < 4; i++) {
    ASSERT_FLOAT_EQ(c[i], a[i] - b[i], 1e-6);
  }

  poly_program_destroy(prog);
  free(src);
  free(lin);
  poly_ctx_destroy(k.ctx);
  PASS();
}

/* ── WGSL renderer tests ─────────────────────────────────────────────── */

TEST(codegen, render_wgsl_vecadd) {
  VecKernel k = make_vec_binop(POLY_OP_ADD, 10);
  int n;
  PolyUOp **lin = poly_linearize(k.ctx, k.sink, &n);
  char *src = poly_render_wgsl(lin, n, "vecadd");

  /* buffer bindings */
  ASSERT_NOT_NULL(strstr(src, "@group(0) @binding(0)"));
  ASSERT_NOT_NULL(strstr(src, "var<storage,read_write> data0: array<f32>"));
  ASSERT_NOT_NULL(strstr(src, "@group(0) @binding(1)"));
  ASSERT_NOT_NULL(strstr(src, "var<storage,read_write> data1: array<f32>"));
  ASSERT_NOT_NULL(strstr(src, "@group(0) @binding(2)"));
  ASSERT_NOT_NULL(strstr(src, "var<storage,read_write> data2: array<f32>"));

  /* compute shader entry point */
  ASSERT_NOT_NULL(strstr(src, "@compute @workgroup_size(1)"));
  ASSERT_NOT_NULL(strstr(src, "fn vecadd("));
  ASSERT_NOT_NULL(strstr(src, "@builtin(global_invocation_id)"));

  /* loop */
  ASSERT_NOT_NULL(strstr(src, "for (var ridx0: i32 = 0; ridx0 < 10; ridx0++)"));

  /* array indexing (not pointer arithmetic) */
  ASSERT_NOT_NULL(strstr(src, "data0[ridx0]"));
  ASSERT_NOT_NULL(strstr(src, "data1[ridx0]"));
  ASSERT_NOT_NULL(strstr(src, "data2[ridx0]"));

  /* variable declarations with WGSL types */
  ASSERT_NOT_NULL(strstr(src, "var val0: f32"));
  ASSERT_NOT_NULL(strstr(src, "var val1: f32"));
  ASSERT_NOT_NULL(strstr(src, "var alu0: f32"));

  /* no C-style wrapper */
  ASSERT_TRUE(strstr(src, "_call") == NULL);

  free(src);
  free(lin);
  poly_ctx_destroy(k.ctx);
  PASS();
}

TEST(codegen, render_wgsl_vecmul) {
  VecKernel k = make_vec_binop(POLY_OP_MUL, 8);
  int n;
  PolyUOp **lin = poly_linearize(k.ctx, k.sink, &n);
  char *src = poly_render_wgsl(lin, n, "vecmul");

  ASSERT_NOT_NULL(strstr(src, "fn vecmul("));
  ASSERT_NOT_NULL(strstr(src, "ridx0 < 8"));
  ASSERT_NOT_NULL(strstr(src, "*"));  /* multiply operator */

  free(src);
  free(lin);
  poly_ctx_destroy(k.ctx);
  PASS();
}

TEST(codegen, render_wgsl_unary) {
  /* b[i] = -a[i] for i in 0..5 */
  PolyCtx *ctx = poly_ctx_new();
  PolyDType ptr_f32 = poly_dtype_ptr(POLY_FLOAT32, -1, POLY_ADDR_GLOBAL);

  PolyUOp *p0 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(0));
  PolyUOp *p1 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(1));

  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(6));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));

  PolyUOp *idx0 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p0, range, poly_arg_none());
  PolyUOp *idx1 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p1, range, poly_arg_none());

  PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx0, poly_arg_none());
  PolyUOp *neg  = poly_uop1(ctx, POLY_OP_NEG, POLY_FLOAT32, load, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx1, neg, poly_arg_none());

  PolyUOp *end_src[2] = { store, range };
  PolyUOp *end  = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());

  int n;
  PolyUOp **lin = poly_linearize(ctx, sink, &n);
  char *src = poly_render_wgsl(lin, n, "vecneg");

  /* only 2 bindings */
  ASSERT_NOT_NULL(strstr(src, "data0: array<f32>"));
  ASSERT_NOT_NULL(strstr(src, "data1: array<f32>"));
  ASSERT_NOT_NULL(strstr(src, "fn vecneg("));
  /* NEG renders as (-val) */
  ASSERT_NOT_NULL(strstr(src, "(-val0)"));

  free(src);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, render_wgsl_where) {
  /* c[i] = cond ? a[i] : b[i], where cond = (a[i] < 5.0) */
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

  PolyUOp *load0 = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx0, poly_arg_none());
  PolyUOp *load1 = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx1, poly_arg_none());

  PolyUOp *five = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(5.0));
  PolyUOp *cond = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, load0, five, poly_arg_none());

  PolyUOp *where_src[3] = { cond, load0, load1 };
  PolyUOp *where = poly_uop(ctx, POLY_OP_WHERE, POLY_FLOAT32, where_src, 3, poly_arg_none());

  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx2, where, poly_arg_none());

  PolyUOp *end_src[2] = { store, range };
  PolyUOp *end  = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());

  int n;
  PolyUOp **lin = poly_linearize(ctx, sink, &n);
  char *src = poly_render_wgsl(lin, n, "vecwhere");

  /* WHERE maps to select(false_val, true_val, cond) */
  ASSERT_NOT_NULL(strstr(src, "select("));

  free(src);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, render_wgsl_reduce) {
  /* out[0] = sum(a[0..9]) — reduce with accumulator */
  PolyCtx *ctx = poly_ctx_new();
  PolyDType ptr_f32 = poly_dtype_ptr(POLY_FLOAT32, -1, POLY_ADDR_GLOBAL);

  PolyUOp *p0 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(0));
  PolyUOp *p1 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(1));

  /* accumulator */
  PolyUOp *acc = poly_uop0(ctx, POLY_OP_DEFINE_LOCAL, POLY_FLOAT32, poly_arg_float(0.0));

  /* inner loop */
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(10));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));

  PolyUOp *idx0 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p0, range, poly_arg_none());
  PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx0, poly_arg_none());

  /* acc += load */
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, acc, load, poly_arg_none());
  PolyUOp *store_acc = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, acc, add, poly_arg_none());

  PolyUOp *end_src[2] = { store_acc, range };
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());

  /* store result */
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *idx1 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p1, zero, poly_arg_none());
  PolyUOp *store_out_src[3] = { idx1, acc, end };
  PolyUOp *store_out = poly_uop(ctx, POLY_OP_STORE, POLY_VOID, store_out_src, 3, poly_arg_none());

  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store_out, poly_arg_none());

  int n_lin;
  PolyUOp **lin = poly_linearize(ctx, sink, &n_lin);
  char *src = poly_render_wgsl(lin, n_lin, "reduce_sum");

  /* accumulator: var acc0: f32 = 0.0; */
  ASSERT_NOT_NULL(strstr(src, "var acc0: f32 = 0.0"));
  /* loop present */
  ASSERT_NOT_NULL(strstr(src, "for (var ridx0: i32 = 0;"));
  /* accumulator store (not array write) */
  ASSERT_NOT_NULL(strstr(src, "acc0 = "));
  /* output buffer store */
  ASSERT_NOT_NULL(strstr(src, "data1[0]"));

  free(src);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

/* ── Unary op end-to-end ─────────────────────────────────────────────── */

TEST(codegen, e2e_neg) {
  /* b[i] = -a[i] for i in 0..5 */
  PolyCtx *ctx = poly_ctx_new();
  PolyDType ptr_f32 = poly_dtype_ptr(POLY_FLOAT32, -1, POLY_ADDR_GLOBAL);

  PolyUOp *p0 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(0));
  PolyUOp *p1 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(1));

  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(6));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));

  PolyUOp *idx0 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p0, range, poly_arg_none());
  PolyUOp *idx1 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p1, range, poly_arg_none());

  PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx0, poly_arg_none());
  PolyUOp *neg  = poly_uop1(ctx, POLY_OP_NEG, POLY_FLOAT32, load, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx1, neg, poly_arg_none());

  PolyUOp *end_src[2] = { store, range };
  PolyUOp *end  = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());

  int n;
  PolyUOp **lin = poly_linearize(ctx, sink, &n);
  char *src = poly_render_c(lin, n, "vecneg");

  PolyProgram *prog = poly_compile_c(src, "vecneg");
  ASSERT_NOT_NULL(prog);

  float a[6] = { 1.0f, -2.5f, 3.14f, 0.0f, -100.0f, 42.0f };
  float b[6] = { 0 };

  void *args[2] = { a, b };
  poly_program_call(prog, args, 2);

  for (int i = 0; i < 6; i++) {
    ASSERT_FLOAT_EQ(b[i], -a[i], 1e-6);
  }

  poly_program_destroy(prog);
  free(src);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}
