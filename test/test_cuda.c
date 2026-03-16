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
#include "../src/scheduler.h"
#include "../src/rangeify.h"
#include <string.h>

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
  PolyUOp *end_src[2] = { store, range };
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

/* ── CUDA binding helpers ──────────────────────────────────────────────── */

static int build_cuda_bindings(PolyBufferBinding *out, PolyUOp **bufs,
                                float **host_ptrs, int n);
static void free_cuda_bindings(PolyBufferBinding *bindings, int n);
static void readback_cuda_binding(PolyBufferBinding *b, void *host_dst, size_t nbytes);

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
    POLY_BIND_HOST(tv.buf_c, c_cpu), POLY_BIND_HOST(tv.buf_a, a), POLY_BIND_HOST(tv.buf_b, b)
  };
  int ret = poly_realize(tv.ctx, tv.sink, cpu_binds, 3);
  ASSERT_TRUE(ret == 0);

  /* GPU via unified poly_realize with CUDA-domain bindings */
  PolyUOp *bufs[] = { tv.buf_c, tv.buf_a, tv.buf_b };
  float *ptrs[] = { NULL, a, b };
  PolyBufferBinding cuda_binds[3];
  ASSERT_INT_EQ(build_cuda_bindings(cuda_binds, bufs, ptrs, 3), 0);
  ASSERT_INT_EQ(poly_realize(tv.ctx, tv.sink, cuda_binds, 3), 0);
  readback_cuda_binding(&cuda_binds[0], c_gpu, n * sizeof(float));
  free_cuda_bindings(cuda_binds, 3);

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

  PolyBufferBinding cpu_binds[] = { POLY_BIND_HOST(buf_c, c_cpu), POLY_BIND_HOST(buf_a, a) };
  ASSERT_TRUE(poly_realize(ctx, sink, cpu_binds, 2) == 0);

  PolyUOp *bufs[] = { buf_c, buf_a };
  float *ptrs[] = { NULL, a };
  PolyBufferBinding cuda_binds[2];
  ASSERT_INT_EQ(build_cuda_bindings(cuda_binds, bufs, ptrs, 2), 0);
  ASSERT_INT_EQ(poly_realize(ctx, sink, cuda_binds, 2), 0);
  readback_cuda_binding(&cuda_binds[0], c_gpu, n * sizeof(float));
  free_cuda_bindings(cuda_binds, 2);

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
    POLY_BIND_HOST(buf_c, c_cpu), POLY_BIND_HOST(buf_a, a), POLY_BIND_HOST(buf_b, b)
  };
  ASSERT_TRUE(poly_realize(ctx, sink, cpu_binds, 3) == 0);

  PolyUOp *bufs[] = { buf_c, buf_a, buf_b };
  float *ptrs[] = { NULL, a, b };
  PolyBufferBinding cuda_binds[3];
  ASSERT_INT_EQ(build_cuda_bindings(cuda_binds, bufs, ptrs, 3), 0);
  ASSERT_INT_EQ(poly_realize(ctx, sink, cuda_binds, 3), 0);
  readback_cuda_binding(&cuda_binds[0], c_gpu, n * sizeof(float));
  free_cuda_bindings(cuda_binds, 3);

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

  PolyBufferBinding cpu_binds[] = { POLY_BIND_HOST(buf_c, c_cpu), POLY_BIND_HOST(buf_a, a) };
  ASSERT_TRUE(poly_realize(ctx, sink, cpu_binds, 2) == 0);

  PolyUOp *bufs[] = { buf_c, buf_a };
  float *ptrs[] = { NULL, a };
  PolyBufferBinding cuda_binds[2];
  ASSERT_INT_EQ(build_cuda_bindings(cuda_binds, bufs, ptrs, 2), 0);
  ASSERT_INT_EQ(poly_realize(ctx, sink, cuda_binds, 2), 0);
  readback_cuda_binding(&cuda_binds[0], c_gpu, n * sizeof(float));
  free_cuda_bindings(cuda_binds, 2);

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

  PolyBufferBinding cpu_binds[] = { POLY_BIND_HOST(buf_c, &c_cpu), POLY_BIND_HOST(buf_a, a) };
  ASSERT_TRUE(poly_realize(ctx, sink, cpu_binds, 2) == 0);

  PolyUOp *bufs[] = { buf_c, buf_a };
  float *ptrs[] = { NULL, a };
  PolyBufferBinding cuda_binds[2];
  ASSERT_INT_EQ(build_cuda_bindings(cuda_binds, bufs, ptrs, 2), 0);
  ASSERT_INT_EQ(poly_realize(ctx, sink, cuda_binds, 2), 0);
  readback_cuda_binding(&cuda_binds[0], &c_gpu, sizeof(float));
  free_cuda_bindings(cuda_binds, 2);

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

  PolyBufferBinding cpu_binds[] = { POLY_BIND_HOST(buf_c, &c_cpu), POLY_BIND_HOST(buf_a, a) };
  ASSERT_TRUE(poly_realize(ctx, sink, cpu_binds, 2) == 0);

  PolyUOp *bufs[] = { buf_c, buf_a };
  float *ptrs[] = { NULL, a };
  PolyBufferBinding cuda_binds[2];
  ASSERT_INT_EQ(build_cuda_bindings(cuda_binds, bufs, ptrs, 2), 0);
  ASSERT_INT_EQ(poly_realize(ctx, sink, cuda_binds, 2), 0);
  readback_cuda_binding(&cuda_binds[0], &c_gpu, sizeof(float));
  free_cuda_bindings(cuda_binds, 2);

  /* Parallel reduce may have slightly different FP rounding */
  ASSERT_FLOAT_EQ(c_gpu, c_cpu, 1.0f);  /* expect ~10000, allow 1.0 error */

  free(a);
  poly_ctx_destroy(ctx);
  PASS();
}

/* ══════════════════════════════════════════════════════════════════════ */
/*  Unified poly_realize() with CUDA-domain bindings                     */
/* ══════════════════════════════════════════════════════════════════════ */

/* Helper: build CUDA-domain bindings from host data */
static int build_cuda_bindings(PolyBufferBinding *out, PolyUOp **bufs,
                                float **host_ptrs, int n) {
  for (int i = 0; i < n; i++) {
    size_t nbytes = (size_t)bufs[i]->arg.i * poly_dtype_itemsize(
                      poly_dtype_scalar(bufs[i]->dtype));
    unsigned long long dptr = poly_cuda_alloc(nbytes);
    if (!dptr) return -1;
    if (host_ptrs[i])
      poly_cuda_copy_htod(dptr, host_ptrs[i], nbytes);
    else
      poly_cuda_memset(dptr, 0, nbytes);
    out[i].buffer = bufs[i];
    out[i].handle = (PolyBufferHandle){
      (void *)(uintptr_t)dptr, nbytes, POLY_DEVICE_CUDA, true
    };
  }
  return 0;
}

static void free_cuda_bindings(PolyBufferBinding *bindings, int n) {
  for (int i = 0; i < n; i++)
    if (bindings[i].handle.owned)
      poly_cuda_free((unsigned long long)(uintptr_t)bindings[i].handle.ptr);
}

static void readback_cuda_binding(PolyBufferBinding *b, void *host_dst, size_t nbytes) {
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

  /* CPU reference via poly_realize */
  PolyBufferBinding cpu_b[] = {
    POLY_BIND_HOST(tv.buf_c, c_cpu), POLY_BIND_HOST(tv.buf_a, a), POLY_BIND_HOST(tv.buf_b, b)
  };
  ASSERT_INT_EQ(poly_realize(tv.ctx, tv.sink, cpu_b, 3), 0);

  /* CUDA via poly_realize (device inferred from CUDA-domain bindings) */
  PolyUOp *bufs[] = { tv.buf_c, tv.buf_a, tv.buf_b };
  float *ptrs[] = { NULL, a, b };
  PolyBufferBinding cuda_b[3];
  ASSERT_INT_EQ(build_cuda_bindings(cuda_b, bufs, ptrs, 3), 0);

  ASSERT_INT_EQ(poly_realize(tv.ctx, tv.sink, cuda_b, 3), 0);

  readback_cuda_binding(&cuda_b[0], c_gpu, n * sizeof(float));
  free_cuda_bindings(cuda_b, 3);

  for (int i = 0; i < n; i++)
    ASSERT_FLOAT_EQ(c_gpu[i], c_cpu[i], 1e-5);

  free(a); free(b); free(c_cpu); free(c_gpu);
  poly_ctx_destroy(tv.ctx);
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
  for (int i = 0; i < n; i++) { a[i] = (float)(i + 1); expected += a[i]; }

  /* CPU */
  PolyBufferBinding cpu_b[] = {
    POLY_BIND_HOST(buf_c, &c_cpu), POLY_BIND_HOST(buf_a, a)
  };
  ASSERT_INT_EQ(poly_realize(ctx, sink, cpu_b, 2), 0);
  ASSERT_FLOAT_EQ(c_cpu, expected, 1e-2);

  /* CUDA */
  PolyUOp *bufs[] = { buf_c, buf_a };
  float *ptrs[] = { NULL, a };
  PolyBufferBinding cuda_b[2];
  ASSERT_INT_EQ(build_cuda_bindings(cuda_b, bufs, ptrs, 2), 0);
  ASSERT_INT_EQ(poly_realize(ctx, sink, cuda_b, 2), 0);
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
#include "../src/model_mlp.h"

static PolyInstance *make_test_mlp(int n_in, int n_out) {
  char spec[256];
  snprintf(spec, sizeof(spec),
    "{\"layers\":[%d,4,%d],\"activation\":\"relu\",\"bias\":true,"
    "\"loss\":\"mse\",\"batch_size\":1,\"seed\":42}", n_in, n_out);
  return poly_mlp_instance(spec, (int)strlen(spec));
}

TEST(cuda, instance_set_device_cuda) {
  SKIP_IF_NO_CUDA();

  PolyInstance *inst = make_test_mlp(2, 3);
  ASSERT_NOT_NULL(inst);

  ASSERT_INT_EQ(poly_instance_set_device(inst, POLY_DEVICE_CUDA), 0);

  /* buf_data should return NULL on CUDA device */
  int64_t numel;
  ASSERT_TRUE(poly_instance_buf_data(inst, 0, &numel) == NULL);

  /* Switch back to CPU */
  ASSERT_INT_EQ(poly_instance_set_device(inst, POLY_DEVICE_CPU), 0);

  /* buf_data should work again on CPU */
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
  float input[] = { 1.0f, 2.0f };
  float out_cpu[3] = {0};
  PolyIOBinding io[] = { {"x", input} };
  ASSERT_INT_EQ(poly_instance_forward(inst, io, 1), 0);

  /* Read CPU output */
  int out_idx = -1;
  for (int i = 0; i < poly_instance_buf_count(inst); i++)
    if (poly_instance_buf_role(inst, i) == POLY_ROLE_OUTPUT) { out_idx = i; break; }
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

  float input[] = { 1.0f, -1.0f };
  PolyIOBinding io[] = { {"x", input} };
  float results[4];

  /* CPU -> forward */
  ASSERT_INT_EQ(poly_instance_forward(inst, io, 1), 0);
  int out_idx = -1;
  for (int i = 0; i < poly_instance_buf_count(inst); i++)
    if (poly_instance_buf_role(inst, i) == POLY_ROLE_OUTPUT) { out_idx = i; break; }
  { int64_t n; results[0] = *poly_instance_buf_data(inst, out_idx, &n); }

  /* CUDA -> forward */
  ASSERT_INT_EQ(poly_instance_set_device(inst, POLY_DEVICE_CUDA), 0);
  ASSERT_INT_EQ(poly_instance_forward(inst, io, 1), 0);
  poly_instance_readback_buf(inst, out_idx, &results[1], sizeof(float));

  /* CPU -> forward */
  ASSERT_INT_EQ(poly_instance_set_device(inst, POLY_DEVICE_CPU), 0);
  ASSERT_INT_EQ(poly_instance_forward(inst, io, 1), 0);
  { int64_t n; results[2] = *poly_instance_buf_data(inst, out_idx, &n); }

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

#endif /* POLY_HAS_CUDA */
