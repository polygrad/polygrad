/*
 * test_cuda.c — CUDA renderer, runtime, and end-to-end tests
 *
 * Guarded by POLY_HAS_CUDA (compile-time) and poly_cuda_available() (runtime).
 * Tests that require a GPU are skipped gracefully if CUDA is not available.
 */

#ifdef POLY_HAS_CUDA

#include "test_harness.h"
#include "../src/codegen.h"
#include "../src/frontend.h"
#include "../src/sched.h"
#include "../src/rangeify.h"

/* Skip helper: PASS immediately if no GPU */
#define SKIP_IF_NO_CUDA() do { \
  if (!poly_cuda_available()) { PASS(); } \
} while (0)

/* ── Helper: build vecadd kernel IR (tensor-level) ───────────────────── */

typedef struct {
  PolyCtx *ctx;
  PolyUOp *sink;
  PolyUOp *buf_a, *buf_b, *buf_c;
  int n;
} TensorVecadd;

static TensorVecadd make_tensor_vecadd(int n) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, n);
  PolyUOp *b = poly_buffer(ctx, POLY_FLOAT32, n);
  PolyUOp *c = poly_buffer(ctx, POLY_FLOAT32, n);
  PolyUOp *add = poly_alu2(ctx, POLY_OP_ADD, a, b);
  PolyUOp *store = poly_store_val(ctx, c, add);
  PolyUOp *sink = poly_sink1(ctx, store);
  return (TensorVecadd){ ctx, sink, a, b, c, n };
}

static int count_lin_ops(PolyUOp **lin, int n, PolyOps op) {
  int count = 0;
  for (int i = 0; i < n; i++) if (lin[i]->op == op) count++;
  return count;
}

/* ── Render tests ────────────────────────────────────────────────────── */

TEST(cuda, render_vecadd) {
  /* Test CUDA source generation — no GPU needed */
  PolyCtx *ctx = poly_ctx_new();
  PolyDType ptr_f32 = poly_dtype_ptr(POLY_FLOAT32, -1, POLY_ADDR_GLOBAL);

  PolyUOp *p0 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(0));
  PolyUOp *p1 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(1));
  PolyUOp *p2 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(2));

  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(10));
  PolyUOp *special = poly_uop1(ctx, POLY_OP_SPECIAL, POLY_INT32, bound, poly_arg_str("gidx0"));

  PolyUOp *idx0 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p0, special, poly_arg_none());
  PolyUOp *idx1 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p1, special, poly_arg_none());
  PolyUOp *idx2 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p2, special, poly_arg_none());

  PolyUOp *ld0 = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx0, poly_arg_none());
  PolyUOp *ld1 = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx1, poly_arg_none());
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, ld0, ld1, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx2, add, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  int n_lin;
  PolyUOp **lin = poly_linearize_rewritten(ctx, sink, &n_lin);
  ASSERT_NOT_NULL(lin);

  char *src = poly_render_cuda(lin, n_lin, "test_kernel", 256);
  free(lin);
  ASSERT_NOT_NULL(src);

  /* Check key CUDA features in output */
  ASSERT_TRUE(strstr(src, "__global__") != NULL);
  ASSERT_TRUE(strstr(src, "__launch_bounds__") != NULL);
  ASSERT_TRUE(strstr(src, "blockIdx") != NULL);
  ASSERT_TRUE(strstr(src, "threadIdx") != NULL);
  ASSERT_TRUE(strstr(src, "extern \"C\"") != NULL);
  ASSERT_TRUE(strstr(src, "gidx0") != NULL);
  /* No _call wrapper */
  ASSERT_TRUE(strstr(src, "_call") == NULL);

  free(src);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(cuda, linearize_reduce_merge_shared_end) {
  /* CUDA rewrite path should merge shared reduce END chains exactly once. */
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
  PolyUOp **lin = poly_linearize_cuda(ctx, sink, &n);
  ASSERT_NOT_NULL(lin);
  ASSERT_TRUE(n > 0);
  ASSERT_INT_EQ(count_lin_ops(lin, n, POLY_OP_DEFINE_REG), 2);
  ASSERT_INT_EQ(count_lin_ops(lin, n, POLY_OP_END), 1);

  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

/* ── E2E tests (require GPU) ─────────────────────────────────────────── */

TEST(cuda, e2e_vecadd) {
  SKIP_IF_NO_CUDA();

  int n = 1024;
  TensorVecadd tv = make_tensor_vecadd(n);

  float *a = malloc(n * sizeof(float));
  float *b = malloc(n * sizeof(float));
  float *c_cpu = calloc(n, sizeof(float));
  float *c_gpu = calloc(n, sizeof(float));

  for (int i = 0; i < n; i++) {
    a[i] = (float)i * 0.1f;
    b[i] = (float)(n - i) * 0.05f;
  }

  /* CPU reference */
  PolyBufferBinding cpu_binds[] = {
    { tv.buf_c, c_cpu }, { tv.buf_a, a }, { tv.buf_b, b }
  };
  int ret = poly_realize(tv.ctx, tv.sink, cpu_binds, 3);
  ASSERT_TRUE(ret == 0);

  /* GPU */
  PolyBufferBinding gpu_binds[] = {
    { tv.buf_c, c_gpu }, { tv.buf_a, a }, { tv.buf_b, b }
  };
  ret = poly_realize_cuda(tv.ctx, tv.sink, gpu_binds, 3);
  ASSERT_TRUE(ret == 0);
  poly_cuda_copyback(gpu_binds, 3);

  /* Compare */
  for (int i = 0; i < n; i++) {
    ASSERT_FLOAT_EQ(c_gpu[i], c_cpu[i], 1e-5);
  }

  free(a); free(b); free(c_cpu); free(c_gpu);
  poly_ctx_destroy(tv.ctx);
  PASS();
}

TEST(cuda, e2e_neg) {
  SKIP_IF_NO_CUDA();

  int n = 512;
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *buf_a = poly_buffer(ctx, POLY_FLOAT32, n);
  PolyUOp *buf_c = poly_buffer(ctx, POLY_FLOAT32, n);
  PolyUOp *neg = poly_alu1(ctx, POLY_OP_NEG, buf_a);
  PolyUOp *store = poly_store_val(ctx, buf_c, neg);
  PolyUOp *sink = poly_sink1(ctx, store);

  float *a = malloc(n * sizeof(float));
  float *c_cpu = calloc(n, sizeof(float));
  float *c_gpu = calloc(n, sizeof(float));
  for (int i = 0; i < n; i++) a[i] = (float)i - 256.0f;

  PolyBufferBinding cpu_binds[] = { { buf_c, c_cpu }, { buf_a, a } };
  ASSERT_TRUE(poly_realize(ctx, sink, cpu_binds, 2) == 0);

  PolyBufferBinding gpu_binds[] = { { buf_c, c_gpu }, { buf_a, a } };
  ASSERT_TRUE(poly_realize_cuda(ctx, sink, gpu_binds, 2) == 0);
  poly_cuda_copyback(gpu_binds, 2);

  for (int i = 0; i < n; i++) ASSERT_FLOAT_EQ(c_gpu[i], c_cpu[i], 1e-5);

  free(a); free(c_cpu); free(c_gpu);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(cuda, e2e_chain) {
  SKIP_IF_NO_CUDA();

  int n = 256;
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *buf_a = poly_buffer(ctx, POLY_FLOAT32, n);
  PolyUOp *buf_b = poly_buffer(ctx, POLY_FLOAT32, n);
  PolyUOp *buf_c = poly_buffer(ctx, POLY_FLOAT32, n);
  /* c = (a + b) * a */
  PolyUOp *add = poly_alu2(ctx, POLY_OP_ADD, buf_a, buf_b);
  PolyUOp *mul = poly_alu2(ctx, POLY_OP_MUL, add, buf_a);
  PolyUOp *store = poly_store_val(ctx, buf_c, mul);
  PolyUOp *sink = poly_sink1(ctx, store);

  float *a = malloc(n * sizeof(float));
  float *b = malloc(n * sizeof(float));
  float *c_cpu = calloc(n, sizeof(float));
  float *c_gpu = calloc(n, sizeof(float));
  for (int i = 0; i < n; i++) { a[i] = (float)i * 0.01f; b[i] = 1.0f; }

  PolyBufferBinding cpu_binds[] = {
    { buf_c, c_cpu }, { buf_a, a }, { buf_b, b }
  };
  ASSERT_TRUE(poly_realize(ctx, sink, cpu_binds, 3) == 0);

  PolyBufferBinding gpu_binds[] = {
    { buf_c, c_gpu }, { buf_a, a }, { buf_b, b }
  };
  ASSERT_TRUE(poly_realize_cuda(ctx, sink, gpu_binds, 3) == 0);
  poly_cuda_copyback(gpu_binds, 3);

  for (int i = 0; i < n; i++) ASSERT_FLOAT_EQ(c_gpu[i], c_cpu[i], 1e-5);

  free(a); free(b); free(c_cpu); free(c_gpu);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(cuda, e2e_exp2) {
  SKIP_IF_NO_CUDA();

  int n = 256;
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *buf_a = poly_buffer(ctx, POLY_FLOAT32, n);
  PolyUOp *buf_c = poly_buffer(ctx, POLY_FLOAT32, n);
  PolyUOp *exp = poly_alu1(ctx, POLY_OP_EXP2, buf_a);
  PolyUOp *store = poly_store_val(ctx, buf_c, exp);
  PolyUOp *sink = poly_sink1(ctx, store);

  float *a = malloc(n * sizeof(float));
  float *c_cpu = calloc(n, sizeof(float));
  float *c_gpu = calloc(n, sizeof(float));
  for (int i = 0; i < n; i++) a[i] = (float)i * 0.05f - 6.0f;

  PolyBufferBinding cpu_binds[] = { { buf_c, c_cpu }, { buf_a, a } };
  ASSERT_TRUE(poly_realize(ctx, sink, cpu_binds, 2) == 0);

  PolyBufferBinding gpu_binds[] = { { buf_c, c_gpu }, { buf_a, a } };
  ASSERT_TRUE(poly_realize_cuda(ctx, sink, gpu_binds, 2) == 0);
  poly_cuda_copyback(gpu_binds, 2);

  for (int i = 0; i < n; i++) ASSERT_FLOAT_EQ(c_gpu[i], c_cpu[i], 1e-4);

  free(a); free(c_cpu); free(c_gpu);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(cuda, e2e_reduce_sum) {
  SKIP_IF_NO_CUDA();

  int n = 512;
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *buf_a = poly_buffer(ctx, POLY_FLOAT32, n);
  PolyUOp *buf_c = poly_buffer(ctx, POLY_FLOAT32, 1);
  int64_t axes[] = { 0 };
  PolyUOp *red = poly_reduce_axis(ctx, POLY_OP_ADD, buf_a, axes, 1);
  PolyUOp *store = poly_store_val(ctx, buf_c, red);
  PolyUOp *sink = poly_sink1(ctx, store);

  float *a = malloc(n * sizeof(float));
  float c_cpu = 0, c_gpu = 0;
  for (int i = 0; i < n; i++) a[i] = 1.0f;

  PolyBufferBinding cpu_binds[] = { { buf_c, &c_cpu }, { buf_a, a } };
  ASSERT_TRUE(poly_realize(ctx, sink, cpu_binds, 2) == 0);

  PolyBufferBinding gpu_binds[] = { { buf_c, &c_gpu }, { buf_a, a } };
  ASSERT_TRUE(poly_realize_cuda(ctx, sink, gpu_binds, 2) == 0);
  poly_cuda_copyback(gpu_binds, 2);

  ASSERT_FLOAT_EQ(c_gpu, c_cpu, 1e-2);  /* reduce sums can accumulate fp error */

  free(a);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(cuda, e2e_reduce_sum_parallel) {
  SKIP_IF_NO_CUDA();

  /* Large N triggers parallel reduction (N > block_size * 2 = 512) */
  int n = 10000;
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *buf_a = poly_buffer(ctx, POLY_FLOAT32, n);
  PolyUOp *buf_c = poly_buffer(ctx, POLY_FLOAT32, 1);
  int64_t axes[] = { 0 };
  PolyUOp *red = poly_reduce_axis(ctx, POLY_OP_ADD, buf_a, axes, 1);
  PolyUOp *store = poly_store_val(ctx, buf_c, red);
  PolyUOp *sink = poly_sink1(ctx, store);

  float *a = malloc(n * sizeof(float));
  float c_cpu = 0, c_gpu = 0;
  for (int i = 0; i < n; i++) a[i] = 1.0f;

  PolyBufferBinding cpu_binds[] = { { buf_c, &c_cpu }, { buf_a, a } };
  ASSERT_TRUE(poly_realize(ctx, sink, cpu_binds, 2) == 0);

  PolyBufferBinding gpu_binds[] = { { buf_c, &c_gpu }, { buf_a, a } };
  ASSERT_TRUE(poly_realize_cuda(ctx, sink, gpu_binds, 2) == 0);
  poly_cuda_copyback(gpu_binds, 2);

  /* Parallel reduce may have slightly different FP rounding */
  ASSERT_FLOAT_EQ(c_gpu, c_cpu, 1.0f);  /* expect ~10000, allow 1.0 error */

  free(a);
  poly_ctx_destroy(ctx);
  PASS();
}

#endif /* POLY_HAS_CUDA */
