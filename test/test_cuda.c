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
#include "../src/engine/schedule.h"
#include "../src/schedule/rangeify.h"
#include "../src/nn.h"
#include "../src/optim.h"
#include "../src/tensor.h"
#include <string.h>

/* Skip helper: PASS immediately if no GPU */
#define SKIP_IF_NO_CUDA()                                                                          \
  do {                                                                                             \
    if (!poly_cuda_available()) {                                                                  \
      PASS();                                                                                      \
    }                                                                                              \
  } while (0)

/* Helper: build vecadd kernel IR (tensor-level) */

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
  return (TensorVecadd){ctx, sink, a, b, c, n};
}

static int count_lin_ops(PolyUOp **lin, int n, PolyOps op) {
  int count = 0;
  for (int i = 0; i < n; i++)
    if (lin[i]->op == op) count++;
  return count;
}

static int64_t test_uop_numel(PolyCtx *ctx, PolyUOp *u) {
  PolyShape s = poly_uop_shape(ctx, u);
  if (s.ndim < 0 || !s.dims) return -1;
  int64_t n = 1;
  for (int i = 0; i < s.ndim; i++)
    n *= s.dims[i];
  free(s.dims);
  return n;
}

static const char *cuda_source_illegal_wide_f32_vector(const char *src) {
  if (!src) return NULL;
  const char *bad[] = {
      "float8",       "float16",       "float32",       "float64",
      "float128",     "float256",      "float512",      "make_float8",
      "make_float16", "make_float32",  "make_float64",  "make_float128",
      "make_float256", "make_float512",
  };
  for (int i = 0; i < (int)(sizeof(bad) / sizeof(bad[0])); i++)
    if (strstr(src, bad[i]) != NULL) return bad[i];
  return NULL;
}

static bool cuda_source_has_illegal_wide_f32_vector(const char *src) {
  return cuda_source_illegal_wide_f32_vector(src) != NULL;
}

/* Render tests */

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
  ASSERT_TRUE(strstr(src, "extern \"C\"") != NULL);
  ASSERT_TRUE(strstr(src, "gidx0") != NULL);
  /* No _call wrapper */
  ASSERT_TRUE(strstr(src, "_call") == NULL);

  free(src);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(cuda, render_mulacc_fma) {
  /* CUDA renderer must emit __fmaf_rn for MULACC -- no GPU needed.
   * Build MUL+ADD pattern; CUDA pipeline fuses to MULACC; renderer emits FMA. */
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
  /* Build MUL+ADD (no MULACC in input) -- CUDA pipeline should fuse */
  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, ld0, ld1, poly_arg_none());
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, mul, ld2, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx3, add, poly_arg_none());
  PolyUOp *end_src[2] = {store, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());

  int n_lin;
  PolyUOp **lin = poly_linearize_cuda(ctx, sink, &n_lin);
  ASSERT_NOT_NULL(lin);
  char *src = poly_render_cuda(lin, n_lin, "fma_test", 256);
  free(lin);
  ASSERT_NOT_NULL(src);
  /* Only assert presence -- don't negative-check for decomposed patterns,
   * source will contain * and + for indexing/gpudims/etc. */
  ASSERT_TRUE(strstr(src, "__fmaf_rn(") != NULL);
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

  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(1024));
  PolyUOp *r0 =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_range(0, POLY_AXIS_REDUCE));

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
  PolyUOp *end0_srcs[2] = {st0, r0};
  PolyUOp *end1_srcs[2] = {st1, r0};
  PolyUOp *end0 = poly_uop(ctx, POLY_OP_END, POLY_VOID, end0_srcs, 2, poly_arg_none());
  PolyUOp *end1 = poly_uop(ctx, POLY_OP_END, POLY_VOID, end1_srcs, 2, poly_arg_none());
  PolyUOp *ends[2] = {end0, end1};
  PolyUOp *sink = poly_uop(ctx, POLY_OP_SINK, POLY_VOID, ends, 2, poly_arg_none());

  int n = 0;
  PolyUOp **lin = poly_linearize_cuda(ctx, sink, &n);
  ASSERT_NOT_NULL(lin);
  ASSERT_TRUE(n > 0);
  ASSERT_TRUE(count_lin_ops(lin, n, POLY_OP_DEFINE_LOCAL) >= 2);
  ASSERT_TRUE(count_lin_ops(lin, n, POLY_OP_BARRIER) >= 2);
  ASSERT_TRUE(count_lin_ops(lin, n, POLY_OP_END) >= 1);

  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

/* CUDA binding helpers */

static int build_cuda_bindings(PolyTestBufferView *out, PolyUOp **bufs, float **host_ptrs, int n);
static void free_cuda_bindings(PolyTestBufferView *bindings, int n);
static void readback_cuda_binding(PolyTestBufferView *b, void *host_dst, size_t nbytes);

/* E2E tests (require GPU) */

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
  PolyTestBufferView cpu_binds[] = {
      POLY_TEST_HOST_VIEW(tv.buf_c, c_cpu), POLY_TEST_HOST_VIEW(tv.buf_a, a), POLY_TEST_HOST_VIEW(tv.buf_b, b)
  };
  int ret = poly_test_realize_buffer_views(tv.ctx, tv.sink, cpu_binds, 3);
  ASSERT_TRUE(ret == 0);

  /* GPU via unified poly_test_realize_buffer_views with CUDA-domain bindings */
  PolyUOp *bufs[] = {tv.buf_c, tv.buf_a, tv.buf_b};
  float *ptrs[] = {NULL, a, b};
  PolyTestBufferView cuda_binds[3];
  ASSERT_INT_EQ(build_cuda_bindings(cuda_binds, bufs, ptrs, 3), 0);
  ASSERT_INT_EQ(poly_test_realize_buffer_views(tv.ctx, tv.sink, cuda_binds, 3), 0);
  readback_cuda_binding(&cuda_binds[0], c_gpu, n * sizeof(float));
  free_cuda_bindings(cuda_binds, 3);

  /* Compare */
  for (int i = 0; i < n; i++) {
    ASSERT_FLOAT_EQ(c_gpu[i], c_cpu[i], 1e-5);
  }

  free(a);
  free(b);
  free(c_cpu);
  free(c_gpu);
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
  for (int i = 0; i < n; i++)
    a[i] = (float)i - 256.0f;

  PolyTestBufferView cpu_binds[] = {POLY_TEST_HOST_VIEW(buf_c, c_cpu), POLY_TEST_HOST_VIEW(buf_a, a)};
  ASSERT_TRUE(poly_test_realize_buffer_views(ctx, sink, cpu_binds, 2) == 0);

  PolyUOp *bufs[] = {buf_c, buf_a};
  float *ptrs[] = {NULL, a};
  PolyTestBufferView cuda_binds[2];
  ASSERT_INT_EQ(build_cuda_bindings(cuda_binds, bufs, ptrs, 2), 0);
  ASSERT_INT_EQ(poly_test_realize_buffer_views(ctx, sink, cuda_binds, 2), 0);
  readback_cuda_binding(&cuda_binds[0], c_gpu, n * sizeof(float));
  free_cuda_bindings(cuda_binds, 2);

  for (int i = 0; i < n; i++)
    ASSERT_FLOAT_EQ(c_gpu[i], c_cpu[i], 1e-5);

  free(a);
  free(c_cpu);
  free(c_gpu);
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
  for (int i = 0; i < n; i++) {
    a[i] = (float)i * 0.01f;
    b[i] = 1.0f;
  }

  PolyTestBufferView cpu_binds[] = {
      POLY_TEST_HOST_VIEW(buf_c, c_cpu), POLY_TEST_HOST_VIEW(buf_a, a), POLY_TEST_HOST_VIEW(buf_b, b)
  };
  ASSERT_TRUE(poly_test_realize_buffer_views(ctx, sink, cpu_binds, 3) == 0);

  PolyUOp *bufs[] = {buf_c, buf_a, buf_b};
  float *ptrs[] = {NULL, a, b};
  PolyTestBufferView cuda_binds[3];
  ASSERT_INT_EQ(build_cuda_bindings(cuda_binds, bufs, ptrs, 3), 0);
  ASSERT_INT_EQ(poly_test_realize_buffer_views(ctx, sink, cuda_binds, 3), 0);
  readback_cuda_binding(&cuda_binds[0], c_gpu, n * sizeof(float));
  free_cuda_bindings(cuda_binds, 3);

  for (int i = 0; i < n; i++)
    ASSERT_FLOAT_EQ(c_gpu[i], c_cpu[i], 1e-5);

  free(a);
  free(b);
  free(c_cpu);
  free(c_gpu);
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
  for (int i = 0; i < n; i++)
    a[i] = (float)i * 0.05f - 6.0f;

  PolyTestBufferView cpu_binds[] = {POLY_TEST_HOST_VIEW(buf_c, c_cpu), POLY_TEST_HOST_VIEW(buf_a, a)};
  ASSERT_TRUE(poly_test_realize_buffer_views(ctx, sink, cpu_binds, 2) == 0);

  PolyUOp *bufs[] = {buf_c, buf_a};
  float *ptrs[] = {NULL, a};
  PolyTestBufferView cuda_binds[2];
  ASSERT_INT_EQ(build_cuda_bindings(cuda_binds, bufs, ptrs, 2), 0);
  ASSERT_INT_EQ(poly_test_realize_buffer_views(ctx, sink, cuda_binds, 2), 0);
  readback_cuda_binding(&cuda_binds[0], c_gpu, n * sizeof(float));
  free_cuda_bindings(cuda_binds, 2);

  for (int i = 0; i < n; i++)
    ASSERT_FLOAT_EQ(c_gpu[i], c_cpu[i], 1e-4);

  free(a);
  free(c_cpu);
  free(c_gpu);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(cuda, e2e_reduce_sum) {
  SKIP_IF_NO_CUDA();

  int n = 512;
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *buf_a = poly_buffer(ctx, POLY_FLOAT32, n);
  PolyUOp *buf_c = poly_buffer(ctx, POLY_FLOAT32, 1);
  int64_t axes[] = {0};
  PolyUOp *red = poly_reduce_axis(ctx, POLY_OP_ADD, buf_a, axes, 1);
  PolyUOp *store = poly_store_val(ctx, buf_c, red);
  PolyUOp *sink = poly_sink1(ctx, store);

  float *a = malloc(n * sizeof(float));
  float c_cpu = 0, c_gpu = 0;
  for (int i = 0; i < n; i++)
    a[i] = 1.0f;

  PolyTestBufferView cpu_binds[] = {POLY_TEST_HOST_VIEW(buf_c, &c_cpu), POLY_TEST_HOST_VIEW(buf_a, a)};
  ASSERT_TRUE(poly_test_realize_buffer_views(ctx, sink, cpu_binds, 2) == 0);

  PolyUOp *bufs[] = {buf_c, buf_a};
  float *ptrs[] = {NULL, a};
  PolyTestBufferView cuda_binds[2];
  ASSERT_INT_EQ(build_cuda_bindings(cuda_binds, bufs, ptrs, 2), 0);
  ASSERT_INT_EQ(poly_test_realize_buffer_views(ctx, sink, cuda_binds, 2), 0);
  readback_cuda_binding(&cuda_binds[0], &c_gpu, sizeof(float));
  free_cuda_bindings(cuda_binds, 2);

  ASSERT_FLOAT_EQ(c_gpu, c_cpu, 1e-2); /* reduce sums can accumulate fp error */

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
  int64_t axes[] = {0};
  PolyUOp *red = poly_reduce_axis(ctx, POLY_OP_ADD, buf_a, axes, 1);
  PolyUOp *store = poly_store_val(ctx, buf_c, red);
  PolyUOp *sink = poly_sink1(ctx, store);

  float *a = malloc(n * sizeof(float));
  float c_cpu = 0, c_gpu = 0;
  for (int i = 0; i < n; i++)
    a[i] = 1.0f;

  PolyTestBufferView cpu_binds[] = {POLY_TEST_HOST_VIEW(buf_c, &c_cpu), POLY_TEST_HOST_VIEW(buf_a, a)};
  ASSERT_TRUE(poly_test_realize_buffer_views(ctx, sink, cpu_binds, 2) == 0);

  PolyUOp *bufs[] = {buf_c, buf_a};
  float *ptrs[] = {NULL, a};
  PolyTestBufferView cuda_binds[2];
  ASSERT_INT_EQ(build_cuda_bindings(cuda_binds, bufs, ptrs, 2), 0);
  ASSERT_INT_EQ(poly_test_realize_buffer_views(ctx, sink, cuda_binds, 2), 0);
  readback_cuda_binding(&cuda_binds[0], &c_gpu, sizeof(float));
  free_cuda_bindings(cuda_binds, 2);

  /* Parallel reduce may have slightly different FP rounding */
  ASSERT_FLOAT_EQ(c_gpu, c_cpu, 1.0f); /* expect ~10000, allow 1.0 error */

  free(a);
  poly_ctx_destroy(ctx);
  PASS();
}

/* ══════════════════════════════════════════════════════════════════════ */
/*  Unified poly_test_realize_buffer_views() with CUDA-domain bindings       */
/* ══════════════════════════════════════════════════════════════════════ */

/* Helper: build CUDA-domain bindings from host data */
static int build_cuda_bindings(PolyTestBufferView *out, PolyUOp **bufs, float **host_ptrs, int n) {
  for (int i = 0; i < n; i++) {
    size_t nbytes = (size_t)bufs[i]->arg.i * poly_dtype_itemsize(poly_dtype_scalar(bufs[i]->dtype));
    unsigned long long dptr = poly_cuda_alloc(nbytes);
    if (!dptr) return -1;
    if (host_ptrs[i])
      poly_cuda_copy_htod(dptr, host_ptrs[i], nbytes);
    else
      poly_cuda_memset(dptr, 0, nbytes);
    out[i].buffer = bufs[i];
    out[i].handle = (PolyBuffer){(void *)(uintptr_t)dptr, nbytes, POLY_DEVICE_CUDA, true};
  }
  return 0;
}

static void free_cuda_bindings(PolyTestBufferView *bindings, int n) {
  for (int i = 0; i < n; i++)
    if (bindings[i].handle.owned)
      poly_cuda_free((unsigned long long)(uintptr_t)bindings[i].handle.ptr);
}

static void readback_cuda_binding(PolyTestBufferView *b, void *host_dst, size_t nbytes) {
  poly_cuda_copy_dtoh(host_dst, (unsigned long long)(uintptr_t)b->handle.ptr, nbytes);
}

TEST(cuda, realize_unified_vecadd) {
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

  /* CPU reference via poly_test_realize_buffer_views */
  PolyTestBufferView cpu_b[] = {
      POLY_TEST_HOST_VIEW(tv.buf_c, c_cpu), POLY_TEST_HOST_VIEW(tv.buf_a, a), POLY_TEST_HOST_VIEW(tv.buf_b, b)
  };
  ASSERT_INT_EQ(poly_test_realize_buffer_views(tv.ctx, tv.sink, cpu_b, 3), 0);

  /* CUDA via poly_test_realize_buffer_views (device inferred from CUDA-domain bindings) */
  PolyUOp *bufs[] = {tv.buf_c, tv.buf_a, tv.buf_b};
  float *ptrs[] = {NULL, a, b};
  PolyTestBufferView cuda_b[3];
  ASSERT_INT_EQ(build_cuda_bindings(cuda_b, bufs, ptrs, 3), 0);

  ASSERT_INT_EQ(poly_test_realize_buffer_views(tv.ctx, tv.sink, cuda_b, 3), 0);

  readback_cuda_binding(&cuda_b[0], c_gpu, n * sizeof(float));
  free_cuda_bindings(cuda_b, 3);

  for (int i = 0; i < n; i++)
    ASSERT_FLOAT_EQ(c_gpu[i], c_cpu[i], 1e-5);

  free(a);
  free(b);
  free(c_cpu);
  free(c_gpu);
  poly_ctx_destroy(tv.ctx);
  PASS();
}

TEST(cuda, tensor_place_computed_expression_to_cuda_e2e) {
  SKIP_IF_NO_CUDA();

  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer_f32(ctx, 3);
  float input[3] = {1.0f, 2.0f, 3.0f};
  poly_buffer_set(ctx, a, input, sizeof(input), POLY_DEVICE_CPU);

  PolyTensor *at = poly_tensor_create(ctx, a, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(at);
  PolyUOp *mul = poly_alu2(ctx, POLY_OP_MUL, poly_tensor_uop(at), poly_const_float(ctx, 2.0f));
  ASSERT_NOT_NULL(mul);
  PolyTensor *mt = poly_tensor_create(ctx, mul, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(mt);
  PolyTensor *cuda_t = poly_tensor_to_device(ctx, mt, POLY_DEVICE_CUDA);
  ASSERT_NOT_NULL(cuda_t);

  PolyTensor *out = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &cuda_t, 1, &out), 0);
  ASSERT_PTR_EQ(out, cuda_t);

  float got[3] = {0};
  const PolyUOp *buf = poly_uop_get_buffer_identity(poly_tensor_uop(cuda_t));
  ASSERT_NOT_NULL(buf);
  ASSERT_INT_EQ(poly_buffer_read(ctx, (PolyUOp *)buf, got, sizeof(got)), 0);
  ASSERT_FLOAT_EQ(got[0], 2.0f, 1e-5f);
  ASSERT_FLOAT_EQ(got[1], 4.0f, 1e-5f);
  ASSERT_FLOAT_EQ(got[2], 6.0f, 1e-5f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(cuda, tensor_place_computed_expression_cuda_cpu_roundtrip_e2e) {
  SKIP_IF_NO_CUDA();

  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer_f32(ctx, 3);
  float input[3] = {1.0f, 2.0f, 3.0f};
  poly_buffer_set(ctx, a, input, sizeof(input), POLY_DEVICE_CPU);

  PolyTensor *at = poly_tensor_create(ctx, a, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(at);
  PolyUOp *add = poly_alu2(ctx, POLY_OP_ADD, poly_tensor_uop(at), poly_const_float(ctx, 1.0f));
  ASSERT_NOT_NULL(add);
  PolyTensor *xt = poly_tensor_create(ctx, add, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(xt);
  PolyTensor *cuda_t = poly_tensor_to_device(ctx, xt, POLY_DEVICE_CUDA);
  ASSERT_NOT_NULL(cuda_t);
  PolyTensor *cpu_t = poly_tensor_to_device(ctx, cuda_t, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(cpu_t);

  PolyTensor *out = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &cpu_t, 1, &out), 0);
  ASSERT_PTR_EQ(out, cpu_t);

  float got[3] = {0};
  const PolyUOp *buf = poly_uop_get_buffer_identity(poly_tensor_uop(cpu_t));
  ASSERT_NOT_NULL(buf);
  ASSERT_INT_EQ(poly_buffer_read(ctx, (PolyUOp *)buf, got, sizeof(got)), 0);
  ASSERT_FLOAT_EQ(got[0], 2.0f, 1e-5f);
  ASSERT_FLOAT_EQ(got[1], 3.0f, 1e-5f);
  ASSERT_FLOAT_EQ(got[2], 4.0f, 1e-5f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(cuda, realize_unified_reduce) {
  SKIP_IF_NO_CUDA();

  int n = 256;
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *buf_a = poly_buffer(ctx, POLY_FLOAT32, n);
  PolyUOp *buf_c = poly_buffer(ctx, POLY_FLOAT32, 1);
  int64_t axes[] = {0};
  PolyUOp *s = poly_reduce_axis(ctx, POLY_OP_ADD, buf_a, axes, 1);
  PolyUOp *store = poly_store_val(ctx, buf_c, s);
  PolyUOp *sink = poly_sink1(ctx, store);

  float *a = malloc(n * sizeof(float));
  float c_cpu = 0, c_gpu = 0;
  float expected = 0;
  for (int i = 0; i < n; i++) {
    a[i] = (float)(i + 1);
    expected += a[i];
  }

  /* CPU */
  PolyTestBufferView cpu_b[] = {POLY_TEST_HOST_VIEW(buf_c, &c_cpu), POLY_TEST_HOST_VIEW(buf_a, a)};
  ASSERT_INT_EQ(poly_test_realize_buffer_views(ctx, sink, cpu_b, 2), 0);
  ASSERT_FLOAT_EQ(c_cpu, expected, 1e-2);

  /* CUDA */
  PolyUOp *bufs[] = {buf_c, buf_a};
  float *ptrs[] = {NULL, a};
  PolyTestBufferView cuda_b[2];
  ASSERT_INT_EQ(build_cuda_bindings(cuda_b, bufs, ptrs, 2), 0);
  ASSERT_INT_EQ(poly_test_realize_buffer_views(ctx, sink, cuda_b, 2), 0);
  readback_cuda_binding(&cuda_b[0], &c_gpu, sizeof(float));
  free_cuda_bindings(cuda_b, 2);

  ASSERT_FLOAT_EQ(c_gpu, c_cpu, 1.0f);

  free(a);
  poly_ctx_destroy(ctx);
  PASS();
}

/* ══════════════════════════════════════════════════════════════════════ */
/*  Instance-level CUDA tests                                            */
/* ══════════════════════════════════════════════════════════════════════ */

#include "../src/instance.h"
#include "../src/models/mlp.h"

static PolyInstance *make_test_mlp(int n_in, int n_out) {
  char spec[256];
  snprintf(
      spec, sizeof(spec),
      "{\"layers\":[%d,4,%d],\"activation\":\"relu\",\"bias\":true,"
      "\"loss\":\"mse\",\"batch_size\":1,\"seed\":42}",
      n_in, n_out
  );
  return poly_mlp_from_json(spec, (int)strlen(spec));
}

TEST(cuda, large_mlp_train_cuda_codegen_no_wide_f32_vectors) {
  /* Codegen-only regression for the larger MLP train step. tinygrad lowers
   * this graph without render-visible f32 vector typedefs like float128 or
   * float512; CUDA C has no such vector names, so they must be scalarized or
   * split before rendering. No CUDA device is required for this test. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  int64_t x_shape[2] = {32, 128};
  int64_t y_shape[2] = {32, 64};
  PolyUOp *x_buf = poly_input(ctx, POLY_FLOAT32, x_shape, 2, "x");
  PolyUOp *y_buf = poly_target(ctx, POLY_FLOAT32, y_shape, 2, "y");
  ASSERT_NOT_NULL(x_buf);
  ASSERT_NOT_NULL(y_buf);

  PolyUOp *x = poly_reshape(ctx, x_buf, x_shape, 2);
  x = poly_linear(ctx, "layers.0", x, 128, 256, true);
  ASSERT_NOT_NULL(x);
  x = poly_relu(ctx, x);
  ASSERT_NOT_NULL(x);
  PolyUOp *pred = poly_linear(ctx, "layers.1", x, 256, 64, true);
  ASSERT_NOT_NULL(pred);
  PolyUOp *target = poly_reshape(ctx, y_buf, y_shape, 2);
  PolyUOp *loss = poly_mse_loss(ctx, pred, target);
  ASSERT_NOT_NULL(loss);

  PolyUOp *param_bufs[4] = {
      poly_ctx_get(ctx, "layers.0.weight"),
      poly_ctx_get(ctx, "layers.0.bias"),
      poly_ctx_get(ctx, "layers.1.weight"),
      poly_ctx_get(ctx, "layers.1.bias"),
  };
  int64_t param_shapes[4][2] = {
      {256, 128},
      {256, 0},
      {64, 256},
      {64, 0},
  };
  int param_ndims[4] = {2, 1, 2, 1};
  int64_t param_numels[4] = {256 * 128, 256, 64 * 256, 64};
  PolyUOp *wrts[4];
  for (int i = 0; i < 4; i++) {
    ASSERT_NOT_NULL(param_bufs[i]);
    wrts[i] = poly_reshape(ctx, param_bufs[i], param_shapes[i], param_ndims[i]);
    ASSERT_NOT_NULL(wrts[i]);
  }

  PolyUOp *grads[4] = {0};
  ASSERT_INT_EQ(poly_grad_many(ctx, loss, NULL, wrts, 4, grads), 0);

  PolyUOp *stores[5];
  PolyUOp *loss_out = poly_buffer(ctx, POLY_FLOAT32, 1);
  ASSERT_NOT_NULL(loss_out);
  PolyUOp *loss_flat = loss;
  if (test_uop_numel(ctx, loss) != 1) {
    int64_t one_shape[1] = {1};
    loss_flat = poly_reshape(ctx, loss, one_shape, 1);
  }
  stores[0] = poly_store_val(ctx, loss_out, loss_flat);
  ASSERT_NOT_NULL(stores[0]);

  PolyOptimConfig cfg = {
      .kind = POLY_OPTIM_SGD,
      .lr = 0.01f,
      .momentum = 0.0f,
      .weight_decay = 0.0f,
      .classic = false,
  };
  for (int i = 0; i < 4; i++) {
    ASSERT_NOT_NULL(grads[i]);
    PolyUOp *grad = grads[i];
    PolyShape gs = poly_uop_shape(ctx, grad);
    if (gs.ndim > 1 || (gs.ndim == 1 && gs.dims && gs.dims[0] != param_numels[i])) {
      int64_t flat[1] = {param_numels[i]};
      grad = poly_reshape(ctx, grad, flat, 1);
    }
    if (gs.dims) free(gs.dims);

    PolyOptimUpdate upd;
    ASSERT_INT_EQ(
        poly_optim_build_update(
            ctx, &cfg, param_bufs[i], grad, NULL, NULL, NULL, NULL, param_numels[i], &upd
        ),
        0
    );
    stores[i + 1] = poly_store_buffer_update(ctx, param_bufs[i], upd.param_new);
    ASSERT_NOT_NULL(stores[i + 1]);
  }

  PolyUOp *sink = poly_sink_n(ctx, stores, 5);
  ASSERT_NOT_NULL(sink);
  PolySchedule *sched = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_NOT_NULL(sched);

  int n_rendered = 0;
  for (int i = 0; i < sched->n_items; i++) {
    if (sched->items[i].kind != POLY_EXEC_COMPUTE) continue;
    int n_lin = 0;
    PolyUOp **lin = poly_linearize_cuda(ctx, sched->items[i].root, &n_lin);
    ASSERT_NOT_NULL(lin);
    char name[64];
    snprintf(name, sizeof(name), "large_mlp_train_%d", i);
    char *src = poly_render_cuda(lin, n_lin, name, 256);
    ASSERT_NOT_NULL(src);
    const char *bad_token = cuda_source_illegal_wide_f32_vector(src);
    if (bad_token) {
      for (int j = 0; j < n_lin; j++) {
        if (lin[j] && lin[j]->dtype.count >= 16) {
          fprintf(
              stderr, "    lin[%d] op=%s dtype=%s count=%d nsrc=%d\n", j,
              poly_op_name(lin[j]->op), poly_dtype_name(lin[j]->dtype), lin[j]->dtype.count,
              lin[j]->n_src
          );
          for (int k = 0; k < lin[j]->n_src && k < 4; k++) {
            fprintf(
                stderr, "      src[%d] op=%s dtype=%s count=%d\n", k,
                poly_op_name(lin[j]->src[k]->op), poly_dtype_name(lin[j]->src[k]->dtype),
                lin[j]->src[k]->dtype.count
            );
          }
          for (int p = 0; p < n_lin; p++) {
            if (!lin[p]) continue;
            for (int s = 0; s < lin[p]->n_src; s++) {
              if (lin[p]->src[s] == lin[j]) {
                fprintf(
                    stderr, "      parent lin[%d] op=%s dtype=%s count=%d src_slot=%d\n", p,
                    poly_op_name(lin[p]->op), poly_dtype_name(lin[p]->dtype), lin[p]->dtype.count,
                    s
                );
              }
            }
          }
        }
      }
      const char *where = strstr(src, bad_token);
      if (where) {
        const char *start = where;
        while (start > src && (where - start) < 180) start--;
        fprintf(
            stderr,
            "    CUDA bad token item=%d token=%s context:\n%.*s\n",
            i,
            bad_token,
            360,
            start
        );
      }
      free(lin);
      free(src);
      poly_schedule_free(sched);
      poly_ctx_destroy(ctx);
      FAIL(
          "CUDA renderer emitted illegal wide f32 vector token %s for large MLP train step",
          bad_token
      );
    }
    free(lin);
    free(src);
    n_rendered++;
  }
  ASSERT_TRUE(n_rendered > 0);

  poly_schedule_free(sched);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(cuda, instance_set_device_cuda) {
  SKIP_IF_NO_CUDA();

  PolyInstance *inst = make_test_mlp(2, 3);
  ASSERT_NOT_NULL(inst);

  /* Read initial host data */
  int64_t numel;
  float *cpu_data = poly_instance_buf_data(inst, 0, &numel);
  ASSERT_NOT_NULL(cpu_data);
  float saved = cpu_data[0];

  ASSERT_INT_EQ(poly_instance_set_device(inst, POLY_DEVICE_CUDA), 0);

  /* buf_data auto-readbacks from GPU (tinygrad-style) */
  float *gpu_data = poly_instance_buf_data(inst, 0, &numel);
  ASSERT_NOT_NULL(gpu_data);
  ASSERT_FLOAT_EQ(gpu_data[0], saved, 1e-6);

  /* Switch back to CPU */
  ASSERT_INT_EQ(poly_instance_set_device(inst, POLY_DEVICE_CPU), 0);

  /* buf_data still works on CPU */
  ASSERT_NOT_NULL(poly_instance_buf_data(inst, 0, &numel));

  poly_instance_free(inst);
  PASS();
}

TEST(cuda, instance_cuda_forward_parity) {
  SKIP_IF_NO_CUDA();

  int n_in = 2, n_out = 3;
  PolyInstance *inst = make_test_mlp(n_in, n_out);
  ASSERT_NOT_NULL(inst);

  /* Seed weights deterministically */
  for (int p = 0; p < poly_instance_param_count(inst); p++) {
    int64_t numel;
    float *data = poly_instance_param_data(inst, p, &numel);
    for (int64_t j = 0; j < numel; j++)
      data[j] = (float)(j % 7 - 3) * 0.1f;
  }

  /* Forward on CPU */
  float input[] = {1.0f, 2.0f};
  float out_cpu[3] = {0};
  PolyIOBinding io[] = {{"x", input}};
  ASSERT_INT_EQ(poly_instance_forward(inst, io, 1), 0);

  /* Read CPU output */
  int out_idx = -1;
  for (int i = 0; i < poly_instance_buf_count(inst); i++)
    if (poly_instance_buf_role(inst, i) == POLY_ROLE_OUTPUT) {
      out_idx = i;
      break;
    }
  ASSERT_TRUE(out_idx >= 0);
  {
    int64_t numel;
    float *cpu_out = poly_instance_buf_data(inst, out_idx, &numel);
    ASSERT_NOT_NULL(cpu_out);
    memcpy(out_cpu, cpu_out, n_out * sizeof(float));
  }

  /* Switch to CUDA */
  ASSERT_INT_EQ(poly_instance_set_device(inst, POLY_DEVICE_CUDA), 0);

  /* Forward on CUDA */
  ASSERT_INT_EQ(poly_instance_forward(inst, io, 1), 0);

  /* Readback output */
  float out_gpu[3] = {0};
  poly_instance_readback_buf(inst, out_idx, out_gpu, n_out * sizeof(float));

  /* Compare */
  for (int i = 0; i < n_out; i++)
    ASSERT_FLOAT_EQ(out_gpu[i], out_cpu[i], 1e-4);

  poly_instance_free(inst);
  PASS();
}

TEST(cuda, instance_cuda_roundtrip) {
  SKIP_IF_NO_CUDA();

  PolyInstance *inst = make_test_mlp(2, 1);
  ASSERT_NOT_NULL(inst);

  /* Seed weights */
  for (int p = 0; p < poly_instance_param_count(inst); p++) {
    int64_t numel;
    float *data = poly_instance_param_data(inst, p, &numel);
    for (int64_t j = 0; j < numel; j++)
      data[j] = (float)(j % 5 - 2) * 0.2f;
  }

  float input[] = {1.0f, -1.0f};
  PolyIOBinding io[] = {{"x", input}};
  float results[4];

  /* CPU -> forward */
  ASSERT_INT_EQ(poly_instance_forward(inst, io, 1), 0);
  int out_idx = -1;
  for (int i = 0; i < poly_instance_buf_count(inst); i++)
    if (poly_instance_buf_role(inst, i) == POLY_ROLE_OUTPUT) {
      out_idx = i;
      break;
    }
  {
    int64_t n;
    results[0] = *poly_instance_buf_data(inst, out_idx, &n);
  }

  /* CUDA -> forward */
  ASSERT_INT_EQ(poly_instance_set_device(inst, POLY_DEVICE_CUDA), 0);
  ASSERT_INT_EQ(poly_instance_forward(inst, io, 1), 0);
  poly_instance_readback_buf(inst, out_idx, &results[1], sizeof(float));

  /* CPU -> forward */
  ASSERT_INT_EQ(poly_instance_set_device(inst, POLY_DEVICE_CPU), 0);
  ASSERT_INT_EQ(poly_instance_forward(inst, io, 1), 0);
  {
    int64_t n;
    results[2] = *poly_instance_buf_data(inst, out_idx, &n);
  }

  /* CUDA -> forward */
  ASSERT_INT_EQ(poly_instance_set_device(inst, POLY_DEVICE_CUDA), 0);
  ASSERT_INT_EQ(poly_instance_forward(inst, io, 1), 0);
  poly_instance_readback_buf(inst, out_idx, &results[3], sizeof(float));

  /* All 4 results should match within tolerance */
  for (int i = 1; i < 4; i++)
    ASSERT_FLOAT_EQ(results[i], results[0], 1e-4);

  poly_instance_free(inst);
  PASS();
}

TEST(cuda, instance_cuda_large_mlp_train_no_wide_vector_types) {
  SKIP_IF_NO_CUDA();

  const char *spec = "{\"layers\":[128,256,64],\"activation\":\"relu\",\"bias\":true,"
                     "\"loss\":\"mse\",\"batch_size\":32,\"seed\":42}";
  PolyInstance *inst = poly_mlp_from_json(spec, (int)strlen(spec));
  ASSERT_NOT_NULL(inst);
  ASSERT_INT_EQ(poly_instance_set_device(inst, POLY_DEVICE_CUDA), 0);
  ASSERT_INT_EQ(
      poly_instance_set_optimizer(inst, POLY_OPTIM_SGD, 0.01f, 0.0f, 0.0f, 0.0f, 0.0f), 0
  );

  float x[32 * 128];
  float y[32 * 64];
  for (int i = 0; i < (int)(sizeof(x) / sizeof(x[0])); i++)
    x[i] = (float)((i % 17) - 8) * 0.01f;
  for (int i = 0; i < (int)(sizeof(y) / sizeof(y[0])); i++)
    y[i] = (float)((i % 13) - 6) * 0.01f;

  PolyIOBinding io[] = {
      {"x", x},
      {"y", y},
  };
  float loss = 0.0f;
  ASSERT_INT_EQ(poly_instance_train_step(inst, io, 2, &loss), 0);
  ASSERT_TRUE(isfinite(loss));

  poly_instance_free(inst);
  PASS();
}

#endif /* POLY_HAS_CUDA */
