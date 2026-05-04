/*
 * test_codegen.c — Tests for linearizer, C renderer, and CPU runtime
 */

#include "test_harness.h"
#include "../src/codegen.h"
#include "../src/frontend.h"
#include "../src/tensor.h" /* poly_sum_reduce */

/* Helper: build c[i] = a[i] OP b[i] kernel IR */

typedef struct {
  PolyCtx *ctx;
  PolyUOp *sink;
  int n; /* loop bound */
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
  PolyUOp *end_src[2] = {store, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());

  return (VecKernel){ctx, sink, n};
}

static int count_lin_ops(PolyUOp **lin, int n, PolyOps op) {
  int c = 0;
  for (int i = 0; i < n; i++)
    if (lin[i]->op == op) c++;
  return c;
}

static int count_special_named(PolyUOp **lin, int n, const char *name) {
  int c = 0;
  for (int i = 0; i < n; i++) {
    if (lin[i]->op == POLY_OP_SPECIAL && lin[i]->arg.str && strcmp(lin[i]->arg.str, name) == 0) c++;
  }
  return c;
}

static VecKernel make_vec_copy_with_weak_index_expr(int n) {
  PolyCtx *ctx = poly_ctx_new();
  PolyDType ptr_f32 = poly_dtype_ptr(POLY_FLOAT32, -1, POLY_ADDR_GLOBAL);

  PolyUOp *p0 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(0));
  PolyUOp *p1 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(1));

  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(n));
  PolyUOp *range =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_INDEX, bound, poly_arg_range(0, POLY_AXIS_LOOP));

  /* tinygrad carries address expressions as weakint through most of codegen,
   * then pm_lower_index_dtype narrows them to a concrete integer dtype. This
   * test mirrors that path; explicit user casts to int64 are intentionally not
   * stripped by tinygrad and are not part of this invariant. */
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(0));
  PolyUOp *addr = poly_uop2(ctx, POLY_OP_ADD, POLY_INDEX, range, zero, poly_arg_none());

  PolyUOp *idx0 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p0, addr, poly_arg_none());
  PolyUOp *idx1 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p1, range, poly_arg_none());
  PolyUOp *load0 = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx0, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx1, load0, poly_arg_none());

  PolyUOp *end_src[2] = {store, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());
  return (VecKernel){ctx, sink, n};
}

static bool subtree_has_i64_dtype(PolyUOp *u) {
  if (!u) return false;
  PolyDType scalar = poly_dtype_scalar(u->dtype);
  if (poly_dtype_is_int(scalar) && scalar.bitsize == 64) return true;
  for (int i = 0; i < u->n_src; i++)
    if (subtree_has_i64_dtype(u->src[i])) return true;
  return false;
}

static int count_indexes_with_i64_addr(PolyUOp **nodes, int n) {
  int bad = 0;
  for (int i = 0; i < n; i++) {
    if (nodes[i]->op != POLY_OP_INDEX || nodes[i]->n_src < 2) continue;
    if (subtree_has_i64_dtype(nodes[i]->src[1])) bad++;
    if (nodes[i]->n_src >= 3 && subtree_has_i64_dtype(nodes[i]->src[2])) bad++;
  }
  return bad;
}

/* Linearizer tests */

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
  ASSERT_TRUE(lin[n - 1]->op == POLY_OP_SINK);

  /* END should come just before SINK */
  ASSERT_TRUE(lin[n - 2]->op == POLY_OP_END);

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
        if (lin[k] == src) {
          found = true;
          break;
        }
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

  PolyUOp *red0_srcs[2] = {in_ld, r0};
  PolyUOp *red1_srcs[2] = {in_ld, r0};
  PolyUOp *sum =
      poly_uop(ctx, POLY_OP_REDUCE, POLY_FLOAT32, red0_srcs, 2, poly_arg_ops(POLY_OP_ADD));
  PolyUOp *mx =
      poly_uop(ctx, POLY_OP_REDUCE, POLY_FLOAT32, red1_srcs, 2, poly_arg_ops(POLY_OP_MAX));

  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *out0_idx = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, pout0, zero, poly_arg_none());
  PolyUOp *out1_idx = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, pout1, zero, poly_arg_none());
  PolyUOp *st0 = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out0_idx, sum, poly_arg_none());
  PolyUOp *st1 = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out1_idx, mx, poly_arg_none());
  PolyUOp *stores[2] = {st0, st1};
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

/* Renderer tests */

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

  /* Default C path follows tinygrad's ClangRenderer float4 upcast:
   * 8 elements become a 2-iteration loop with vec4 loads/stores. */
  ASSERT_NOT_NULL(strstr(src, "void vecmul("));
  ASSERT_NOT_NULL(strstr(src, "ridx0 < 2"));
  ASSERT_NOT_NULL(strstr(src, "__attribute__((vector_size(16)))"));
  ASSERT_NOT_NULL(strstr(src, "val0[0]"));
  ASSERT_NOT_NULL(strstr(src, "*"));

  free(src);
  free(lin);
  poly_ctx_destroy(k.ctx);
  PASS();
}

/* End-to-end tests */

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
    a[i] = (float)(i + 1); /* 1, 2, ..., 10 */
    b[i] = (float)((i + 1) * 10); /* 10, 20, ..., 100 */
    c[i] = 0.0f;
  }

  void *args[3] = {a, b, c};
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

  void *args[3] = {a, b, c};
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

  float a[4] = {10, 20, 30, 40};
  float b[4] = {1, 2, 3, 4};
  float c[4] = {0};

  void *args[3] = {a, b, c};
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

/* WGSL renderer tests */

TEST(codegen, render_wgsl_vecadd) {
  VecKernel k = make_vec_binop(POLY_OP_ADD, 10);
  int n;
  PolyUOp **lin = poly_linearize(k.ctx, k.sink, &n);
  char *src = poly_render_wgsl(lin, n, "vecadd");

  /* preamble: INFINITY uniform at binding(0) */
  ASSERT_NOT_NULL(strstr(src, "fn nan()"));
  ASSERT_NOT_NULL(strstr(src, "var<uniform> INFINITY"));
  ASSERT_NOT_NULL(strstr(src, "@group(0) @binding(0)"));

  /* buffer bindings: offset by +1 (binding 0 = INFINITY) */
  ASSERT_NOT_NULL(strstr(src, "@group(0) @binding(1)"));
  ASSERT_NOT_NULL(strstr(src, "var<storage,read_write> data0: array<f32>"));
  ASSERT_NOT_NULL(strstr(src, "@group(0) @binding(2)"));
  ASSERT_NOT_NULL(strstr(src, "var<storage,read_write> data1: array<f32>"));
  ASSERT_NOT_NULL(strstr(src, "@group(0) @binding(3)"));
  ASSERT_NOT_NULL(strstr(src, "var<storage,read_write> data2: array<f32>"));

  /* compute shader entry point: workgroup_id + local_invocation_id */
  ASSERT_NOT_NULL(strstr(src, "@compute @workgroup_size(1)"));
  ASSERT_NOT_NULL(strstr(src, "fn vecadd("));
  ASSERT_NOT_NULL(strstr(src, "@builtin(workgroup_id) gindex"));
  ASSERT_NOT_NULL(strstr(src, "@builtin(local_invocation_id) lindex"));

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
  PolyUOp **lin = poly_linearize_webgpu(k.ctx, k.sink, &n);
  char *src = poly_render_wgsl(lin, n, "vecmul");

  ASSERT_NOT_NULL(strstr(src, "fn vecmul("));
  ASSERT_NOT_NULL(strstr(src, "@compute @workgroup_size(2,1,1)"));
  ASSERT_NOT_NULL(strstr(src, "var lidx0: i32 = i32(lindex.x);"));
  /* The tinygrad-style recursive tuplize tiebreak may order vector lanes as
   * offsets first and base lane last. The semantic contract is four scalar
   * stores to the four lanes, not a specific temporary-number pairing. */
  ASSERT_NOT_NULL(strstr(src, "data2[alu0] ="));
  ASSERT_NOT_NULL(strstr(src, "data2[alu1] ="));
  ASSERT_NOT_NULL(strstr(src, "data2[alu2] ="));
  ASSERT_NOT_NULL(strstr(src, "data2[alu3] ="));
  ASSERT_NOT_NULL(strstr(src, "*")); /* multiply operator */
  ASSERT_TRUE(strstr(src, "for (var ridx0") == NULL);
  ASSERT_TRUE(strstr(src, "vec4<f32>") == NULL);

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
  PolyUOp *neg = poly_uop1(ctx, POLY_OP_NEG, POLY_FLOAT32, load, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx1, neg, poly_arg_none());

  PolyUOp *end_src[2] = {store, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
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

  PolyUOp *where_src[3] = {cond, load0, load1};
  PolyUOp *where = poly_uop(ctx, POLY_OP_WHERE, POLY_FLOAT32, where_src, 3, poly_arg_none());

  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx2, where, poly_arg_none());

  PolyUOp *end_src[2] = {store, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
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

TEST(codegen, render_wgsl_uint32_ops) {
  /* out[i] = where(a[i] < b[i], (a[i] >> 1) + (a[i] % 31), a[i] // b[i]) */
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

  PolyUOp *la = poly_uop1(ctx, POLY_OP_LOAD, POLY_UINT32, idx0, poly_arg_none());
  PolyUOp *lb = poly_uop1(ctx, POLY_OP_LOAD, POLY_UINT32, idx1, poly_arg_none());
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_UINT32, poly_arg_int(1));
  PolyUOp *thirty_one = poly_uop0(ctx, POLY_OP_CONST, POLY_UINT32, poly_arg_int(31));

  PolyUOp *cond = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, la, lb, poly_arg_none());
  PolyUOp *rhs = poly_uop2(
      ctx, POLY_OP_ADD, POLY_UINT32,
      poly_uop2(ctx, POLY_OP_SHR, POLY_UINT32, la, one, poly_arg_none()),
      poly_uop2(ctx, POLY_OP_MOD, POLY_UINT32, la, thirty_one, poly_arg_none()), poly_arg_none()
  );
  PolyUOp *lhs = poly_uop2(ctx, POLY_OP_IDIV, POLY_UINT32, la, lb, poly_arg_none());
  PolyUOp *sel_src[3] = {cond, rhs, lhs};
  PolyUOp *sel = poly_uop(ctx, POLY_OP_WHERE, POLY_UINT32, sel_src, 3, poly_arg_none());

  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx2, sel, poly_arg_none());
  PolyUOp *end_src[2] = {store, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());

  int n;
  PolyUOp **lin = poly_linearize(ctx, sink, &n);
  char *src = poly_render_wgsl(lin, n, "vecu32");

  ASSERT_NOT_NULL(strstr(src, "array<u32>"));
  ASSERT_NOT_NULL(strstr(src, "var val0: u32"));
  ASSERT_NOT_NULL(strstr(src, "31u"));
  ASSERT_NOT_NULL(strstr(src, ">>"));
  ASSERT_NOT_NULL(strstr(src, "%"));
  ASSERT_NOT_NULL(strstr(src, "select("));

  free(src);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, render_wgsl_uint8_storage_is_packed_like_tinygrad) {
  /* out[i] = in[i] for uint8. WGSL has no byte-addressable storage buffer,
   * so tinygrad packs byte/short storage into atomic<u32> lanes. */
  PolyCtx *ctx = poly_ctx_new();
  PolyDType ptr_u8 = poly_dtype_ptr(POLY_UINT8, -1, POLY_ADDR_GLOBAL);

  PolyUOp *p0 = poly_uop0(ctx, POLY_OP_PARAM, ptr_u8, poly_arg_int(0));
  PolyUOp *p1 = poly_uop0(ctx, POLY_OP_PARAM, ptr_u8, poly_arg_int(1));
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(5));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));
  PolyUOp *idx0 = poly_uop2(ctx, POLY_OP_INDEX, ptr_u8, p0, range, poly_arg_none());
  PolyUOp *idx1 = poly_uop2(ctx, POLY_OP_INDEX, ptr_u8, p1, range, poly_arg_none());
  PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, POLY_UINT8, idx0, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx1, load, poly_arg_none());
  PolyUOp *end_src[2] = {store, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());

  int n;
  PolyUOp **lin = poly_linearize(ctx, sink, &n);
  char *src = poly_render_wgsl(lin, n, "copy_u8");

  ASSERT_NOT_NULL(strstr(src, "data0: array<atomic<u32>>"));
  ASSERT_NOT_NULL(strstr(src, "data1: array<atomic<u32>>"));
  ASSERT_NOT_NULL(strstr(src, "atomicLoad(&data0[(ridx0/4)]"));
  ASSERT_NOT_NULL(strstr(src, "atomicAnd(&data1[(ridx0/4)]"));
  ASSERT_NOT_NULL(strstr(src, "atomicAdd(&data1[(ridx0/4)]"));
  ASSERT_TRUE(strstr(src, "data1[ridx0] =") == NULL);

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

  PolyUOp *end_src[2] = {store_acc, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());

  /* store result */
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *idx1 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p1, zero, poly_arg_none());
  PolyUOp *store_out_src[3] = {idx1, acc, end};
  PolyUOp *store_out = poly_uop(ctx, POLY_OP_STORE, POLY_VOID, store_out_src, 3, poly_arg_none());

  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store_out, poly_arg_none());

  int n_lin;
  PolyUOp **lin = poly_linearize(ctx, sink, &n_lin);
  char *src = poly_render_wgsl(lin, n_lin, "reduce_sum");

  /* accumulator declaration hoisted to function scope, init in body */
  ASSERT_NOT_NULL(strstr(src, "var acc0: f32;"));
  ASSERT_NOT_NULL(strstr(src, "acc0 = 0.0"));
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

TEST(codegen, render_wgsl_define_var) {
  /* out[i] = data[i] + cast(N) for i in 0..N, with N as DEFINE_VAR.
   * Verified against tinygrad: PARAM gets var<storage,read_write>,
   * DEFINE_VAR gets var<uniform>, sequential binding indices. */
  PolyCtx *ctx = poly_ctx_new();
  PolyDType ptr_f32 = poly_dtype_ptr(POLY_FLOAT32, -1, POLY_ADDR_GLOBAL);

  PolyUOp *p0 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(0));
  PolyUOp *p1 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(1));
  PolyUOp *N = poly_uop0(ctx, POLY_OP_DEFINE_VAR, POLY_INT32, poly_arg_define_var("N", 1, 16));

  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, N, poly_arg_int(0));

  PolyUOp *idx0 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p0, range, poly_arg_none());
  PolyUOp *idx1 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p1, range, poly_arg_none());

  PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx0, poly_arg_none());
  PolyUOp *cast_n = poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, N, poly_arg_none());
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, load, cast_n, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx1, add, poly_arg_none());

  PolyUOp *end_src[2] = {store, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());

  int n_lin;
  PolyUOp **lin = poly_linearize(ctx, sink, &n_lin);
  char *src = poly_render_wgsl(lin, n_lin, "var_kernel");

  /* binding(0) = INFINITY uniform (always) */
  ASSERT_NOT_NULL(strstr(src, "@group(0) @binding(0)\nvar<uniform> INFINITY"));

  /* binding(1) = data0 storage buffer */
  ASSERT_NOT_NULL(strstr(src, "@group(0) @binding(1)\nvar<storage,read_write> data0: array<f32>"));

  /* binding(2) = data1 storage buffer */
  ASSERT_NOT_NULL(strstr(src, "@group(0) @binding(2)\nvar<storage,read_write> data1: array<f32>"));

  /* binding(3) = N scalar uniform (DEFINE_VAR) */
  ASSERT_NOT_NULL(strstr(src, "@group(0) @binding(3)\nvar<uniform> N: i32"));

  /* the variable name N appears in the kernel body */
  ASSERT_NOT_NULL(strstr(src, "f32(N)"));

  free(src);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, render_wgsl_param_bindings_follow_encounter_order) {
  /* Match tinygrad WGSL binding assignment:
   * bindings are sequential in PARAM/DEFINE_VAR encounter order, not PARAM.arg.
   * Sparse PARAM ids used to leak into @binding(N), which mismatched runtime binding order. */
  PolyCtx *ctx = poly_ctx_new();
  PolyDType ptr_f32 = poly_dtype_ptr(POLY_FLOAT32, -1, POLY_ADDR_GLOBAL);

  PolyUOp *p7 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(7));
  PolyUOp *p2 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(2));
  PolyUOp *p9 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(9));

  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(8));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));

  PolyUOp *idx7 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p7, range, poly_arg_none());
  PolyUOp *idx2 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p2, range, poly_arg_none());
  PolyUOp *idx9 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p9, range, poly_arg_none());

  PolyUOp *lhs = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx2, poly_arg_none());
  PolyUOp *rhs = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx9, poly_arg_none());
  PolyUOp *sum = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, lhs, rhs, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx7, sum, poly_arg_none());

  PolyUOp *end_src[2] = {store, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());

  int n_lin = 0;
  PolyUOp **lin = poly_linearize(ctx, sink, &n_lin);
  char *src = poly_render_wgsl(lin, n_lin, "param_order");

  ASSERT_NOT_NULL(strstr(src, "@group(0) @binding(1)\nvar<storage,read_write>"));
  ASSERT_NOT_NULL(strstr(src, "@group(0) @binding(2)\nvar<storage,read_write>"));
  ASSERT_NOT_NULL(strstr(src, "@group(0) @binding(3)\nvar<storage,read_write>"));
  ASSERT_NOT_NULL(strstr(src, "var<storage,read_write> data7: array<f32>;"));
  ASSERT_NOT_NULL(strstr(src, "var<storage,read_write> data2: array<f32>;"));
  ASSERT_NOT_NULL(strstr(src, "var<storage,read_write> data9: array<f32>;"));
  ASSERT_TRUE(strstr(src, "@group(0) @binding(8)\nvar<storage,read_write> data7: array<f32>") == NULL);
  ASSERT_TRUE(strstr(src, "@group(0) @binding(10)\nvar<storage,read_write> data9: array<f32>") == NULL);

  free(src);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

/* WebGPU GPU linearizer output tests */

TEST(codegen, linearize_webgpu_vecadd_emits_gpudims) {
  /* Verify GPU linearizer produces SPECIAL ops and correct WGSL builtins */
  VecKernel k = make_vec_binop(POLY_OP_ADD, 1024);
  int n_lin = 0;
  PolyUOp **lin = poly_linearize_webgpu(k.ctx, k.sink, &n_lin);
  ASSERT_NOT_NULL(lin);
  ASSERT_TRUE(n_lin > 0);

  /* GPU linearizer should produce at least gidx0 SPECIAL op */
  ASSERT_TRUE(count_special_named(lin, n_lin, "gidx0") >= 1);

  /* Render to WGSL and verify GPU-specific patterns */
  char *wgsl = poly_render_wgsl(lin, n_lin, "vecadd_gpu");
  ASSERT_NOT_NULL(wgsl);
  ASSERT_NOT_NULL(strstr(wgsl, "i32(gindex."));
  ASSERT_NOT_NULL(strstr(wgsl, "@compute @workgroup_size("));
  /* Builtins: workgroup_id + local_invocation_id */
  ASSERT_NOT_NULL(strstr(wgsl, "workgroup_id"));
  ASSERT_NOT_NULL(strstr(wgsl, "local_invocation_id"));

  free(wgsl);
  free(lin);
  poly_ctx_destroy(k.ctx);
  PASS();
}

TEST(codegen, linearize_webgpu_reduce_emits_shared_barrier) {
  /* Verify GPU reduction produces shared memory + barrier.
   * 4096 elements > 256*2 threshold triggers group_for_reduce. */
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer_f32(ctx, 4096);
  PolyUOp *out = poly_buffer_f32(ctx, 1);
  PolyUOp *sum = poly_sum_reduce(ctx, a, 0, 0);
  PolyUOp *store = poly_store_val(ctx, out, sum);
  PolyUOp *sink = poly_sink1(ctx, store);

  PolySchedule *sched = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_NOT_NULL(sched);

  bool found = false;
  for (int i = 0; i < sched->n_items; i++) {
    int n_lin = 0;
    PolyUOp **lin = poly_linearize_webgpu(ctx, sched->items[i].root, &n_lin);
    ASSERT_NOT_NULL(lin);
    char *wgsl = poly_render_wgsl(lin, n_lin, "reduce_webgpu");
    ASSERT_NOT_NULL(wgsl);
    if (strstr(wgsl, "var<workgroup>") && strstr(wgsl, "workgroupBarrier();"))
      found = true;
    free(wgsl);
    free(lin);
  }
  ASSERT_TRUE(found);

  poly_schedule_free(sched);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, full_rewrite_post_index_lowering_narrows_i64_addressing) {
  VecKernel k = make_vec_copy_with_weak_index_expr(32);
  PolyRewriteOpts opts = {
      .optimize = false,
      .devectorize = 1,
      .caps = {.max_vec_width = 1},
      .device = POLY_DEVICE_WEBGPU,
  };

  PolyUOp *rewritten = poly_full_rewrite_to_sink_ex(k.ctx, k.sink, opts);
  ASSERT_NOT_NULL(rewritten);

  int n_topo = 0;
  PolyUOp **topo = poly_toposort(k.ctx, rewritten, &n_topo);
  ASSERT_TRUE(n_topo > 0);
  ASSERT_INT_EQ(count_indexes_with_i64_addr(topo, n_topo), 0);

  poly_ctx_destroy(k.ctx);
  PASS();
}

TEST(codegen, linearize_webgpu_narrows_i64_addressing) {
  VecKernel k = make_vec_copy_with_weak_index_expr(32);
  int n_lin = 0;
  PolyUOp **lin = poly_linearize_webgpu(k.ctx, k.sink, &n_lin);
  ASSERT_NOT_NULL(lin);
  ASSERT_TRUE(n_lin > 0);
  ASSERT_INT_EQ(count_indexes_with_i64_addr(lin, n_lin), 0);

  free(lin);
  poly_ctx_destroy(k.ctx);
  PASS();
}

/* Unary op end-to-end */

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
  PolyUOp *neg = poly_uop1(ctx, POLY_OP_NEG, POLY_FLOAT32, load, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx1, neg, poly_arg_none());

  PolyUOp *end_src[2] = {store, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());

  int n;
  PolyUOp **lin = poly_linearize(ctx, sink, &n);
  char *src = poly_render_c(lin, n, "vecneg");

  PolyProgram *prog = poly_compile_c(src, "vecneg");
  ASSERT_NOT_NULL(prog);

  float a[6] = {1.0f, -2.5f, 3.14f, 0.0f, -100.0f, 42.0f};
  float b[6] = {0};

  void *args[2] = {a, b};
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
