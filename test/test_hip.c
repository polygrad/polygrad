/*
 * test_hip.c -- HIP renderer, runtime, and end-to-end tests
 *
 * Guarded by POLY_HAS_HIP (compile-time) and poly_hip_available() (runtime).
 * Tests that require a GPU are skipped gracefully if HIP is not available.
 */

#ifdef POLY_HAS_HIP

#include "test_harness.h"
#include "../src/codegen.h"
#include "../src/frontend.h"
#include "../src/scheduler.h"
#include "../src/rangeify.h"
#include <string.h>

/* Skip helper: PASS immediately if no GPU */
#define SKIP_IF_NO_HIP() do { \
  if (!poly_hip_available()) { PASS(); } \
} while (0)

/* ── Helper: build vecadd kernel IR (tensor-level) ───────────────────── */

typedef struct {
  PolyCtx *ctx;
  PolyUOp *sink;
  PolyUOp *buf_a, *buf_b, *buf_c;
  int n;
} HipTensorVecadd;

static HipTensorVecadd hip_make_tensor_vecadd(int n) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, n);
  PolyUOp *b = poly_buffer(ctx, POLY_FLOAT32, n);
  PolyUOp *c = poly_buffer(ctx, POLY_FLOAT32, n);
  PolyUOp *add = poly_alu2(ctx, POLY_OP_ADD, a, b);
  PolyUOp *store = poly_store_val(ctx, c, add);
  PolyUOp *sink = poly_sink1(ctx, store);
  return (HipTensorVecadd){ ctx, sink, a, b, c, n };
}

/* ── Render tests (no GPU needed) ────────────────────────────────────── */

TEST(hip, render_vecadd) {
  /* Test HIP source generation -- no GPU needed */
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

  char *src = poly_render_hip(lin, n_lin, "test_kernel", 256);
  free(lin);
  ASSERT_NOT_NULL(src);

  /* Check key HIP features in output */
  ASSERT_TRUE(strstr(src, "__attribute__((global))") != NULL);
  ASSERT_TRUE(strstr(src, "amdgpu_flat_work_group_size") != NULL);
  ASSERT_TRUE(strstr(src, "__ockl_get_group_id") != NULL);
  ASSERT_TRUE(strstr(src, "__ockl_get_local_id") != NULL);
  ASSERT_TRUE(strstr(src, "extern \"C\"") != NULL);
  ASSERT_TRUE(strstr(src, "gidx0") != NULL);
  /* Must NOT contain CUDA-specific tokens */
  ASSERT_TRUE(strstr(src, "blockIdx") == NULL);
  ASSERT_TRUE(strstr(src, "threadIdx") == NULL);
  ASSERT_TRUE(strstr(src, "__global__") == NULL);
  ASSERT_TRUE(strstr(src, "__launch_bounds__") == NULL);

  free(src);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(hip, render_mulacc_fma) {
  /* HIP renderer must emit __builtin_fmaf for MULACC -- no GPU needed. */
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
  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, ld0, ld1, poly_arg_none());
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, mul, ld2, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx3, add, poly_arg_none());
  PolyUOp *end_src[2] = { store, range };
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());

  int n_lin;
  PolyUOp **lin = poly_linearize_hip(ctx, sink, &n_lin);
  ASSERT_NOT_NULL(lin);
  char *src = poly_render_hip(lin, n_lin, "fma_test", 256);
  free(lin);
  ASSERT_NOT_NULL(src);
  ASSERT_TRUE(strstr(src, "__builtin_fmaf(") != NULL);
  /* Must NOT contain CUDA FMA */
  ASSERT_TRUE(strstr(src, "__fmaf_rn(") == NULL);
  free(src);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(hip, render_math_intrinsics) {
  /* HIP renderer must emit __ocml_* for transcendentals -- no GPU needed. */
  PolyCtx *ctx = poly_ctx_new();
  PolyDType ptr_f32 = poly_dtype_ptr(POLY_FLOAT32, -1, POLY_ADDR_GLOBAL);
  PolyUOp *p0 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(0));
  PolyUOp *p1 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(1));
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(4));
  PolyUOp *special = poly_uop1(ctx, POLY_OP_SPECIAL, POLY_INT32, bound, poly_arg_str("gidx0"));
  PolyUOp *idx0 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p0, special, poly_arg_none());
  PolyUOp *idx1 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p1, special, poly_arg_none());
  PolyUOp *ld0 = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx0, poly_arg_none());
  PolyUOp *exp2 = poly_uop1(ctx, POLY_OP_EXP2, POLY_FLOAT32, ld0, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx1, exp2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  int n_lin;
  PolyUOp **lin = poly_linearize_rewritten(ctx, sink, &n_lin);
  ASSERT_NOT_NULL(lin);
  char *src = poly_render_hip(lin, n_lin, "math_test", 256);
  free(lin);
  ASSERT_NOT_NULL(src);
  ASSERT_TRUE(strstr(src, "__ocml_exp2_f32") != NULL);
  /* Must NOT contain C math functions that CUDA uses */
  ASSERT_TRUE(strstr(src, "exp2f(") == NULL);
  free(src);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(hip, render_shared_mem) {
  /* HIP renderer must emit __attribute__((shared, aligned(16))) for DEFINE_LOCAL. */
  PolyCtx *ctx = poly_ctx_new();
  PolyDType smem_ptr = poly_dtype_ptr(POLY_FLOAT32, 64, POLY_ADDR_LOCAL);
  PolyUOp *local = poly_uop0(ctx, POLY_OP_DEFINE_LOCAL, smem_ptr, poly_arg_none());
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *idx = poly_uop2(ctx, POLY_OP_INDEX, smem_ptr, local, zero, poly_arg_none());
  PolyUOp *cst = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(1.0));
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx, cst, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  int n_lin;
  PolyUOp **lin = poly_linearize_rewritten(ctx, sink, &n_lin);
  ASSERT_NOT_NULL(lin);
  char *src = poly_render_hip(lin, n_lin, "smem_test", 256);
  free(lin);
  ASSERT_NOT_NULL(src);
  ASSERT_TRUE(strstr(src, "__attribute__((shared, aligned(16)))") != NULL);
  /* Must NOT contain CUDA shared memory syntax */
  ASSERT_TRUE(strstr(src, "__shared__") == NULL);
  free(src);
  poly_ctx_destroy(ctx);
  PASS();
}

/* ── HIP binding helpers ─────────────────────────────────────────────── */

static int build_hip_bindings(PolyBufferBinding *out, PolyUOp **bufs,
                               float **host_ptrs, int n) {
  for (int i = 0; i < n; i++) {
    size_t nbytes = (size_t)bufs[i]->arg.i * poly_dtype_itemsize(
                      poly_dtype_scalar(bufs[i]->dtype));
    void *dptr = poly_hip_alloc(nbytes);
    if (!dptr) return -1;
    if (host_ptrs[i])
      poly_hip_copy_htod(dptr, host_ptrs[i], nbytes);
    else
      poly_hip_memset(dptr, 0, nbytes);
    out[i].buffer = bufs[i];
    out[i].handle = (PolyBufferHandle){ dptr, nbytes, POLY_DEVICE_HIP, true };
  }
  return 0;
}

static void free_hip_bindings(PolyBufferBinding *bindings, int n) {
  for (int i = 0; i < n; i++)
    if (bindings[i].handle.owned)
      poly_hip_free(bindings[i].handle.ptr);
}

static void readback_hip_binding(PolyBufferBinding *b, void *host_dst, size_t nbytes) {
  poly_hip_copy_dtoh(host_dst, b->handle.ptr, nbytes);
}

/* ── E2E tests (require GPU) ─────────────────────────────────────────── */

TEST(hip, e2e_vecadd) {
  SKIP_IF_NO_HIP();

  int n = 1024;
  HipTensorVecadd tv = hip_make_tensor_vecadd(n);

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

  /* GPU via unified poly_realize with HIP-domain bindings */
  PolyUOp *bufs[] = { tv.buf_c, tv.buf_a, tv.buf_b };
  float *ptrs[] = { NULL, a, b };
  PolyBufferBinding hip_binds[3];
  ASSERT_INT_EQ(build_hip_bindings(hip_binds, bufs, ptrs, 3), 0);
  ASSERT_INT_EQ(poly_realize(tv.ctx, tv.sink, hip_binds, 3), 0);
  readback_hip_binding(&hip_binds[0], c_gpu, n * sizeof(float));
  free_hip_bindings(hip_binds, 3);

  /* Compare */
  for (int i = 0; i < n; i++) {
    ASSERT_FLOAT_EQ(c_gpu[i], c_cpu[i], 1e-5);
  }

  free(a); free(b); free(c_cpu); free(c_gpu);
  poly_ctx_destroy(tv.ctx);
  PASS();
}

TEST(hip, e2e_neg) {
  SKIP_IF_NO_HIP();

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
  PolyBufferBinding hip_binds[2];
  ASSERT_INT_EQ(build_hip_bindings(hip_binds, bufs, ptrs, 2), 0);
  ASSERT_INT_EQ(poly_realize(ctx, sink, hip_binds, 2), 0);
  readback_hip_binding(&hip_binds[0], c_gpu, n * sizeof(float));
  free_hip_bindings(hip_binds, 2);

  for (int i = 0; i < n; i++) ASSERT_FLOAT_EQ(c_gpu[i], c_cpu[i], 1e-5);

  free(a); free(c_cpu); free(c_gpu);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(hip, e2e_exp2) {
  SKIP_IF_NO_HIP();

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
  PolyBufferBinding hip_binds[2];
  ASSERT_INT_EQ(build_hip_bindings(hip_binds, bufs, ptrs, 2), 0);
  ASSERT_INT_EQ(poly_realize(ctx, sink, hip_binds, 2), 0);
  readback_hip_binding(&hip_binds[0], c_gpu, n * sizeof(float));
  free_hip_bindings(hip_binds, 2);

  for (int i = 0; i < n; i++) ASSERT_FLOAT_EQ(c_gpu[i], c_cpu[i], 1e-4);

  free(a); free(c_cpu); free(c_gpu);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(hip, e2e_reduce_sum) {
  SKIP_IF_NO_HIP();

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
  PolyBufferBinding hip_binds[2];
  ASSERT_INT_EQ(build_hip_bindings(hip_binds, bufs, ptrs, 2), 0);
  ASSERT_INT_EQ(poly_realize(ctx, sink, hip_binds, 2), 0);
  readback_hip_binding(&hip_binds[0], &c_gpu, sizeof(float));
  free_hip_bindings(hip_binds, 2);

  ASSERT_FLOAT_EQ(c_gpu, c_cpu, 1e-2);

  free(a);
  poly_ctx_destroy(ctx);
  PASS();
}

/* ── Instance-level HIP tests ────────────────────────────────────────── */

#include "../src/instance.h"
#include "../src/model_mlp.h"

static PolyInstance *hip_make_test_mlp(int n_in, int n_out) {
  char spec[256];
  snprintf(spec, sizeof(spec),
    "{\"layers\":[%d,4,%d],\"activation\":\"relu\",\"bias\":true,"
    "\"loss\":\"mse\",\"batch_size\":1,\"seed\":42}", n_in, n_out);
  return poly_mlp_instance(spec, (int)strlen(spec));
}

TEST(hip, instance_set_device_hip) {
  SKIP_IF_NO_HIP();

  PolyInstance *inst = hip_make_test_mlp(2, 3);
  ASSERT_NOT_NULL(inst);

  ASSERT_INT_EQ(poly_instance_set_device(inst, POLY_DEVICE_HIP), 0);

  /* buf_data should return NULL on HIP device */
  int64_t numel;
  ASSERT_TRUE(poly_instance_buf_data(inst, 0, &numel) == NULL);

  /* Switch back to CPU */
  ASSERT_INT_EQ(poly_instance_set_device(inst, POLY_DEVICE_CPU), 0);

  /* buf_data should work again on CPU */
  ASSERT_NOT_NULL(poly_instance_buf_data(inst, 0, &numel));

  poly_instance_free(inst);
  PASS();
}

TEST(hip, instance_hip_forward_parity) {
  SKIP_IF_NO_HIP();

  int n_in = 2, n_out = 3;
  PolyInstance *inst = hip_make_test_mlp(n_in, n_out);
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

  /* Switch to HIP */
  ASSERT_INT_EQ(poly_instance_set_device(inst, POLY_DEVICE_HIP), 0);

  /* Forward on HIP */
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

TEST(hip, instance_hip_roundtrip) {
  SKIP_IF_NO_HIP();

  PolyInstance *inst = hip_make_test_mlp(2, 1);
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

  /* HIP -> forward */
  ASSERT_INT_EQ(poly_instance_set_device(inst, POLY_DEVICE_HIP), 0);
  ASSERT_INT_EQ(poly_instance_forward(inst, io, 1), 0);
  poly_instance_readback_buf(inst, out_idx, &results[1], sizeof(float));

  /* CPU -> forward */
  ASSERT_INT_EQ(poly_instance_set_device(inst, POLY_DEVICE_CPU), 0);
  ASSERT_INT_EQ(poly_instance_forward(inst, io, 1), 0);
  { int64_t n; results[2] = *poly_instance_buf_data(inst, out_idx, &n); }

  /* HIP -> forward */
  ASSERT_INT_EQ(poly_instance_set_device(inst, POLY_DEVICE_HIP), 0);
  ASSERT_INT_EQ(poly_instance_forward(inst, io, 1), 0);
  poly_instance_readback_buf(inst, out_idx, &results[3], sizeof(float));

  /* All 4 results should match within tolerance */
  for (int i = 1; i < 4; i++)
    ASSERT_FLOAT_EQ(results[i], results[0], 1e-4);

  poly_instance_free(inst);
  PASS();
}

#endif /* POLY_HAS_HIP */
