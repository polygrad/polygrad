/*
 * test_hip.c -- HIP renderer, runtime, and end-to-end tests
 *
 * Guarded by POLY_HAS_HIP (compile-time) and poly_hip_available() (runtime).
 * Tests that require a GPU are skipped gracefully if HIP is not available.
 */

#ifdef POLY_HAS_HIP

#include "test_harness.h"
#include "../src/codegen/codegen.h"
#include "../src/frontend.h"
#include "../src/engine/schedule.h"
#include "../src/schedule/rangeify.h"
#include <string.h>

TEST_BACKEND(hip, native_fp8_types_and_casts_match_current_renderer) {
  /* Tinygrad 2026-08-22/a9069c177a9d HIPRenderer.type_map and
   * string_rewrite use byte storage plus AMD FP8 conversion builtins. */
  const PolyDType dtypes[] = {POLY_FP8E4M3, POLY_FP8E5M2};
  const char *types[] = {"hip_fp8", "hip_bf8"};
  const char *builtins[] = {"__builtin_amdgcn_cvt_f32_fp8", "__builtin_amdgcn_cvt_f32_bf8"};
  for (int i = 0; i < 2; i++) {
    PolyCtx *ctx = poly_ctx_new();
    ASSERT_NOT_NULL(ctx);
    PolyUOp *out = poly_test_program_param(ctx, POLY_FLOAT32, 1, 0);
    PolyUOp *in = poly_test_program_param(ctx, dtypes[i], 1, 1);
    PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(0));
    PolyUOp *out_idx = poly_uop_index(ctx, out, &zero, 1);
    PolyUOp *in_idx = poly_uop_index(ctx, in, &zero, 1);
    PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, dtypes[i], in_idx, poly_arg_none());
    PolyUOp *cast = poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, load, poly_arg_none());
    PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out_idx, cast, poly_arg_none());
    PolyUOp *sink = poly_test_kernel_sink(ctx, &store, 1, "fp8_load");
    int n_linear = 0;
    PolyUOp **linear = poly_do_linearize(ctx, sink, &n_linear);
    ASSERT_NOT_NULL(linear);
    char *source = poly_render_hip(ctx, linear, n_linear, "fp8_load", 1, "gfx950");
    free(linear);
    ASSERT_NOT_NULL(source);
    ASSERT_NOT_NULL(strstr(source, "typedef unsigned char hip_bf8;"));
    ASSERT_NOT_NULL(strstr(source, "typedef unsigned char hip_fp8;"));
    char pointer_type[64];
    snprintf(pointer_type, sizeof(pointer_type), "%s*", types[i]);
    ASSERT_NOT_NULL(strstr(source, pointer_type));
    ASSERT_NOT_NULL(strstr(source, builtins[i]));
    free(source);

    PolyUOp *fp8_out = poly_test_program_param(ctx, dtypes[i], 1, 2);
    PolyUOp *f32_in = poly_test_program_param(ctx, POLY_FLOAT32, 1, 3);
    PolyUOp *fp8_out_idx = poly_uop_index(ctx, fp8_out, &zero, 1);
    PolyUOp *f32_in_idx = poly_uop_index(ctx, f32_in, &zero, 1);
    PolyUOp *f32_load =
        poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, f32_in_idx, poly_arg_none());
    PolyUOp *to_fp8 =
        poly_uop1(ctx, POLY_OP_CAST, dtypes[i], f32_load, poly_arg_none());
    PolyUOp *fp8_store =
        poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, fp8_out_idx, to_fp8, poly_arg_none());
    sink = poly_test_kernel_sink(ctx, &fp8_store, 1, "fp8_store");
    linear = poly_do_linearize(ctx, sink, &n_linear);
    ASSERT_NOT_NULL(linear);
    source = poly_render_hip(ctx, linear, n_linear, "fp8_store", 1, "gfx950");
    free(linear);
    ASSERT_NOT_NULL(source);
    char helper_call[64];
    snprintf(helper_call, sizeof(helper_call), "f32_to_fp8(");
    ASSERT_NOT_NULL(strstr(source, helper_call));
    ASSERT_NOT_NULL(strstr(source, "__builtin_amdgcn_cvt_pk_"));
    free(source);
    if (i == 0) {
      PolyUOp *nan =
          poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKFLOAT, poly_arg_float(NAN));
      PolyUOp *fp8_inf =
          poly_uop1(ctx, POLY_OP_CAST, dtypes[i], nan, poly_arg_none());
      fp8_store =
          poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, fp8_out_idx, fp8_inf, poly_arg_none());
      sink = poly_test_kernel_sink(ctx, &fp8_store, 1, "fp8_const_store");
      linear = poly_do_linearize(ctx, sink, &n_linear);
      ASSERT_NOT_NULL(linear);
      source = poly_render_hip(ctx, linear, n_linear, "fp8_const_store", 1, "gfx950");
      free(linear);
      ASSERT_NOT_NULL(source);
      ASSERT_NOT_NULL(strstr(source, "f32_to_fp8(NAN, 0)"));
      free(source);
    }
    poly_ctx_destroy(ctx);
  }
  PASS();
}

/* Skip helper: PASS immediately if no GPU */
#define SKIP_IF_NO_HIP()                                                                           \
  do {                                                                                             \
    if (!poly_hip_available()) {                                                                   \
      PASS();                                                                                      \
    }                                                                                              \
  } while (0)

/* Helper: build vecadd kernel IR (tensor-level) */

typedef struct {
  PolyCtx *ctx;
  PolyUOp *sink;
  PolyUOp *buf_a, *buf_b, *buf_c;
  int n;
} HipTensorVecadd;

static HipTensorVecadd hip_make_tensor_vecadd(int n) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_test_buffer(ctx, POLY_FLOAT32, n);
  PolyUOp *b = poly_test_buffer(ctx, POLY_FLOAT32, n);
  PolyUOp *c = poly_test_buffer(ctx, POLY_FLOAT32, n);
  PolyUOp *add = poly_alu2(ctx, POLY_OP_ADD, a, b);
  PolyUOp *store = poly_store_val(ctx, c, add);
  PolyUOp *sink = poly_sink1(ctx, store);
  return (HipTensorVecadd){ctx, sink, a, b, c, n};
}

/* Render tests (no GPU needed) */

TEST_BACKEND(hip, render_vecadd) {
  /* Test HIP source generation -- no GPU needed */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *p0 = poly_test_uop_param(ctx, POLY_FLOAT32, -1, 0, POLY_ADDR_GLOBAL);
  PolyUOp *p1 = poly_test_uop_param(ctx, POLY_FLOAT32, -1, 1, POLY_ADDR_GLOBAL);
  PolyUOp *p2 = poly_test_uop_param(ctx, POLY_FLOAT32, -1, 2, POLY_ADDR_GLOBAL);

  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(10));
  PolyUOp *special = poly_uop1(ctx, POLY_OP_SPECIAL, POLY_INT32, bound, poly_arg_str("gidx0"));

  PolyUOp *idx0 = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, p0, special, poly_arg_none());
  PolyUOp *idx1 = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, p1, special, poly_arg_none());
  PolyUOp *idx2 = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, p2, special, poly_arg_none());

  PolyUOp *ld0 = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx0, poly_arg_none());
  PolyUOp *ld1 = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx1, poly_arg_none());
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, ld0, ld1, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx2, add, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  int n_lin;
  PolyUOp **lin = poly_do_linearize(ctx, sink, &n_lin);
  ASSERT_NOT_NULL(lin);

  char *src = poly_render_hip(ctx, lin, n_lin, "test_kernel", 256, "gfx1100");
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

TEST_BACKEND(hip, renderer_inlines_current_casted_literals) {
  /* Current tinygrad renderer/cstyle.py:26-47,238-241. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *shape = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(32));
  PolyParamArg arg = {.slot = 0, .addrspace = POLY_ADDR_GLOBAL};
  PolyUOp *out = poly_uop1(ctx, POLY_OP_PARAM, POLY_INT32, shape, poly_arg_param(&arg));
  PolyUOp *weak_index = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(20));
  PolyUOp *index = poly_uop1(ctx, POLY_OP_CAST, POLY_INT32, weak_index, poly_arg_none());
  PolyUOp *address = poly_uop2(ctx, POLY_OP_INDEX, POLY_INT32, out, index, poly_arg_none());
  PolyUOp *weak_value = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(7));
  PolyUOp *value = poly_uop1(ctx, POLY_OP_CAST, POLY_INT32, weak_value, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, address, value, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());
  int n = 0;
  PolyUOp **uops = poly_toposort(ctx, sink, &n);
  ASSERT_NOT_NULL(uops);
  char *source = poly_render_hip(ctx, uops, n, "casted_const", 1, "gfx1100");
  ASSERT_NOT_NULL(source);
  ASSERT_NOT_NULL(strstr(source, "data0+20"));
  ASSERT_NOT_NULL(strstr(source, " = 7;"));
  ASSERT_TRUE(strstr(source, "cast0") == NULL);
  free(source);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(hip, render_muladd_matches_pinned_hipstyle) {
  /* Pinned HIPRenderer does not advertise Ops.MULACC
   * (renderer/cstyle.py:128-136,472-508), so ordinary MUL+ADD must not be
   * fused by the shared late matcher. Explicit WMMA/MULACC rendering remains
  * a separate renderer ability. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *p0 = poly_test_uop_param(ctx, POLY_FLOAT32, -1, 0, POLY_ADDR_GLOBAL);
  PolyUOp *p1 = poly_test_uop_param(ctx, POLY_FLOAT32, -1, 1, POLY_ADDR_GLOBAL);
  PolyUOp *p2 = poly_test_uop_param(ctx, POLY_FLOAT32, -1, 2, POLY_ADDR_GLOBAL);
  PolyUOp *p3 = poly_test_uop_param(ctx, POLY_FLOAT32, -1, 3, POLY_ADDR_GLOBAL);
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(4));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));
  PolyUOp *idx0 = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, p0, range, poly_arg_none());
  PolyUOp *idx1 = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, p1, range, poly_arg_none());
  PolyUOp *idx2 = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, p2, range, poly_arg_none());
  PolyUOp *idx3 = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, p3, range, poly_arg_none());
  PolyUOp *ld0 = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx0, poly_arg_none());
  PolyUOp *ld1 = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx1, poly_arg_none());
  PolyUOp *ld2 = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx2, poly_arg_none());
  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, ld0, ld1, poly_arg_none());
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, mul, ld2, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx3, add, poly_arg_none());
  PolyUOp *end_src[2] = {store, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());

  int n_lin;
  PolyUOp **lin = poly_linearize_hip(ctx, sink, &n_lin);
  ASSERT_NOT_NULL(lin);
  int mulacc_count = 0, mul_count = 0, add_count = 0;
  for (int i = 0; i < n_lin; i++) {
    mulacc_count += lin[i]->op == POLY_OP_MULACC;
    mul_count += lin[i]->op == POLY_OP_MUL;
    add_count += lin[i]->op == POLY_OP_ADD;
  }
  ASSERT_INT_EQ(mulacc_count, 0);
  ASSERT_TRUE(mul_count >= 1);
  ASSERT_TRUE(add_count >= 1);
  char *src = poly_render_hip(ctx, lin, n_lin, "fma_test", 256, "gfx1100");
  free(lin);
  ASSERT_NOT_NULL(src);
  ASSERT_TRUE(strstr(src, "__builtin_fmaf(") == NULL);
  ASSERT_TRUE(strstr(src, "__fmaf_rn(") == NULL);
  free(src);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(hip, render_math_intrinsics) {
  /* HIP renderer must emit __ocml_* for transcendentals -- no GPU needed. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *p0 = poly_test_uop_param(ctx, POLY_FLOAT32, -1, 0, POLY_ADDR_GLOBAL);
  PolyUOp *p1 = poly_test_uop_param(ctx, POLY_FLOAT32, -1, 1, POLY_ADDR_GLOBAL);
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(4));
  PolyUOp *special = poly_uop1(ctx, POLY_OP_SPECIAL, POLY_INT32, bound, poly_arg_str("gidx0"));
  PolyUOp *idx0 = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, p0, special, poly_arg_none());
  PolyUOp *idx1 = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, p1, special, poly_arg_none());
  PolyUOp *ld0 = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx0, poly_arg_none());
  PolyUOp *exp2 = poly_uop1(ctx, POLY_OP_EXP2, POLY_FLOAT32, ld0, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx1, exp2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  int n_lin;
  PolyUOp **lin = poly_do_linearize(ctx, sink, &n_lin);
  ASSERT_NOT_NULL(lin);
  char *src = poly_render_hip(ctx, lin, n_lin, "math_test", 256, "gfx1100");
  free(lin);
  ASSERT_NOT_NULL(src);
  ASSERT_TRUE(strstr(src, "__ocml_exp2_f32") != NULL);
  /* Must NOT contain C math functions that CUDA uses */
  ASSERT_TRUE(strstr(src, "exp2f(") == NULL);
  free(src);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(hip, render_shared_mem) {
  /* Current C-style HIP renders BUFFER(LOCAL) as shared storage. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *size = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(64));
  PolyParamArg local_arg = {.slot = 0, .addrspace = POLY_ADDR_LOCAL};
  PolyUOp *local =
      poly_uop1(ctx, POLY_OP_BUFFER, POLY_FLOAT32, size, poly_arg_param(&local_arg));
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *idx = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, local, zero, poly_arg_none());
  PolyUOp *cst = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(1.0));
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx, cst, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  int n_lin;
  PolyUOp **lin = poly_do_linearize(ctx, sink, &n_lin);
  ASSERT_NOT_NULL(lin);
  char *src = poly_render_hip(ctx, lin, n_lin, "smem_test", 256, "gfx1100");
  free(lin);
  ASSERT_NOT_NULL(src);
  ASSERT_TRUE(strstr(src, "__attribute__((shared, aligned(16)))") != NULL);
  /* Must NOT contain CUDA shared memory syntax */
  ASSERT_TRUE(strstr(src, "__shared__") == NULL);
  free(src);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(hip, vector_local_shrink_load_store_use_typed_lvalue) {
  /* HIPRenderer inherits the same pinned CStyleLanguage.render_access as CUDA
   * (renderer/cstyle.py:47-58,179-184,472-520). */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *size = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(128));
  PolyParamArg local_arg = {.slot = 0, .addrspace = POLY_ADDR_LOCAL};
  PolyParamArg global_arg = {.slot = 0, .addrspace = POLY_ADDR_GLOBAL};
  PolyUOp *local =
      poly_uop1(ctx, POLY_OP_BUFFER, POLY_FLOAT32, size, poly_arg_param(&local_arg));
  PolyUOp *global =
      poly_uop1(ctx, POLY_OP_PARAM, POLY_FLOAT32, size, poly_arg_param(&global_arg));
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(32));
  PolyUOp *idx =
      poly_uop1(ctx, POLY_OP_SPECIAL, POLY_INT32, bound, poly_arg_str("lidx0"));
  PolyUOp *width = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(4));
  PolyUOp *local_srcs[3] = {local, idx, width};
  PolyUOp *local_vec =
      poly_uop(ctx, POLY_OP_SHRINK, POLY_FLOAT32, local_srcs, 3, poly_arg_none());
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *global_srcs[3] = {global, zero, width};
  PolyUOp *global_vec =
      poly_uop(ctx, POLY_OP_SHRINK, POLY_FLOAT32, global_srcs, 3, poly_arg_none());
  PolyUOp *values[4];
  for (int i = 0; i < 4; i++)
    values[i] =
        poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float((double)i + 1.0));
  PolyUOp *value = poly_uop(ctx, POLY_OP_STACK, POLY_FLOAT32, values, 4, poly_arg_none());
  PolyUOp *local_store =
      poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, local_vec, value, poly_arg_none());
  PolyUOp *local_load =
      poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, local_vec, poly_arg_none());
  PolyUOp *global_store =
      poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, global_vec, local_load, poly_arg_none());
  PolyUOp *sink_srcs[2] = {local_store, global_store};
  PolyUOp *sink = poly_uop(ctx, POLY_OP_SINK, POLY_VOID, sink_srcs, 2, poly_arg_none());

  int n_lin = 0;
  PolyUOp **lin = poly_do_linearize(ctx, sink, &n_lin);
  ASSERT_NOT_NULL(lin);
  char *source =
      poly_render_hip(ctx, lin, n_lin, "local_float4_access", 32, "gfx1100");
  free(lin);
  ASSERT_NOT_NULL(source);
  const char *first = strstr(source, "*((float4*)((smem0+");
  ASSERT_NOT_NULL(first);
  ASSERT_NOT_NULL(strstr(first + 1, "*((float4*)((smem0+"));
  free(source);
  poly_ctx_destroy(ctx);
  PASS();
}

/* HIP binding helpers */

static int build_hip_bindings(PolyTestBufferView *out, PolyUOp **bufs, float **host_ptrs, int n) {
  for (int i = 0; i < n; i++) {
    size_t nbytes = (size_t)bufs[i]->arg.i * poly_dtype_itemsize(bufs[i]->dtype);
    void *dptr = poly_hip_alloc(nbytes);
    if (!dptr) return -1;
    if (host_ptrs[i])
      poly_hip_copy_htod(dptr, host_ptrs[i], nbytes);
    else
      poly_hip_memset(dptr, 0, nbytes);
    out[i].buffer = bufs[i];
    out[i].handle = (PolyBuffer){
        .ptr = dptr,
        .nbytes = nbytes,
        .device = POLY_DEVICE_HIP,
        .owned = true,
    };
  }
  return 0;
}

static void free_hip_bindings(PolyTestBufferView *bindings, int n) {
  for (int i = 0; i < n; i++)
    if (bindings[i].handle.owned) poly_hip_free(bindings[i].handle.ptr);
}

static void readback_hip_binding(PolyTestBufferView *b, void *host_dst, size_t nbytes) {
  poly_hip_copy_dtoh(host_dst, b->handle.ptr, nbytes);
}

/* E2E tests (require GPU) */

TEST_BACKEND(hip, e2e_vecadd) {
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
  PolyTestBufferView cpu_binds[] = {
      POLY_TEST_HOST_VIEW(tv.buf_c, c_cpu), POLY_TEST_HOST_VIEW(tv.buf_a, a), POLY_TEST_HOST_VIEW(tv.buf_b, b)
  };
  int ret = poly_test_realize_buffer_views(tv.ctx, tv.sink, cpu_binds, 3);
  ASSERT_TRUE(ret == 0);

  /* GPU via unified poly_test_realize_buffer_views with HIP-domain bindings */
  PolyUOp *bufs[] = {tv.buf_c, tv.buf_a, tv.buf_b};
  float *ptrs[] = {NULL, a, b};
  PolyTestBufferView hip_binds[3];
  ASSERT_INT_EQ(build_hip_bindings(hip_binds, bufs, ptrs, 3), 0);
  ASSERT_INT_EQ(poly_test_realize_buffer_views(tv.ctx, tv.sink, hip_binds, 3), 0);
  readback_hip_binding(&hip_binds[0], c_gpu, n * sizeof(float));
  free_hip_bindings(hip_binds, 3);

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

TEST_BACKEND(hip, e2e_neg) {
  SKIP_IF_NO_HIP();

  int n = 512;
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *buf_a = poly_test_buffer(ctx, POLY_FLOAT32, n);
  PolyUOp *buf_c = poly_test_buffer(ctx, POLY_FLOAT32, n);
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
  PolyTestBufferView hip_binds[2];
  ASSERT_INT_EQ(build_hip_bindings(hip_binds, bufs, ptrs, 2), 0);
  ASSERT_INT_EQ(poly_test_realize_buffer_views(ctx, sink, hip_binds, 2), 0);
  readback_hip_binding(&hip_binds[0], c_gpu, n * sizeof(float));
  free_hip_bindings(hip_binds, 2);

  for (int i = 0; i < n; i++)
    ASSERT_FLOAT_EQ(c_gpu[i], c_cpu[i], 1e-5);

  free(a);
  free(c_cpu);
  free(c_gpu);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(hip, e2e_exp2) {
  SKIP_IF_NO_HIP();

  int n = 256;
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *buf_a = poly_test_buffer(ctx, POLY_FLOAT32, n);
  PolyUOp *buf_c = poly_test_buffer(ctx, POLY_FLOAT32, n);
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
  PolyTestBufferView hip_binds[2];
  ASSERT_INT_EQ(build_hip_bindings(hip_binds, bufs, ptrs, 2), 0);
  ASSERT_INT_EQ(poly_test_realize_buffer_views(ctx, sink, hip_binds, 2), 0);
  readback_hip_binding(&hip_binds[0], c_gpu, n * sizeof(float));
  free_hip_bindings(hip_binds, 2);

  for (int i = 0; i < n; i++)
    ASSERT_FLOAT_EQ(c_gpu[i], c_cpu[i], 1e-4);

  free(a);
  free(c_cpu);
  free(c_gpu);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(hip, e2e_reduce_sum) {
  SKIP_IF_NO_HIP();

  int n = 512;
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *buf_a = poly_test_buffer(ctx, POLY_FLOAT32, n);
  PolyUOp *buf_c = poly_test_buffer(ctx, POLY_FLOAT32, 1);
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
  PolyTestBufferView hip_binds[2];
  ASSERT_INT_EQ(build_hip_bindings(hip_binds, bufs, ptrs, 2), 0);
  ASSERT_INT_EQ(poly_test_realize_buffer_views(ctx, sink, hip_binds, 2), 0);
  readback_hip_binding(&hip_binds[0], &c_gpu, sizeof(float));
  free_hip_bindings(hip_binds, 2);

  ASSERT_FLOAT_EQ(c_gpu, c_cpu, 1e-2);

  free(a);
  poly_ctx_destroy(ctx);
  PASS();
}

/* Instance-level HIP tests */

#include "../src/instance.h"
#include "../src/models/mlp.h"

static PolyInstance *hip_make_test_mlp(int n_in, int n_out) {
  char spec[256];
  snprintf(
      spec, sizeof(spec),
      "{\"layers\":[%d,4,%d],\"activation\":\"relu\",\"bias\":true,"
      "\"loss\":\"mse\",\"batch_size\":1,\"seed\":42}",
      n_in, n_out
  );
  return poly_mlp_from_json(spec, (int)strlen(spec), POLY_DEVICE_AUTO);
}

TEST_BACKEND(hip, instance_set_device_hip) {
  SKIP_IF_NO_HIP();

  PolyInstance *inst = hip_make_test_mlp(2, 3);
  ASSERT_NOT_NULL(inst);

  /* Read initial host data */
  int64_t numel;
  float *cpu_data = poly_instance_buf_data(inst, 0, &numel);
  ASSERT_NOT_NULL(cpu_data);
  float saved = cpu_data[0];

  ASSERT_INT_EQ(poly_instance_set_device(inst, POLY_DEVICE_HIP), 0);

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

TEST_BACKEND(hip, instance_hip_forward_parity) {
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
  float input[] = {1.0f, 2.0f};
  float out_cpu[3] = {0};
  PolyIOBinding io[] = {POLY_IO_BINDING_ARRAY("x", input, POLY_FLOAT32)};
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

TEST_BACKEND(hip, instance_hip_roundtrip) {
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

  float input[] = {1.0f, -1.0f};
  PolyIOBinding io[] = {POLY_IO_BINDING_ARRAY("x", input, POLY_FLOAT32)};
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

  /* HIP -> forward */
  ASSERT_INT_EQ(poly_instance_set_device(inst, POLY_DEVICE_HIP), 0);
  ASSERT_INT_EQ(poly_instance_forward(inst, io, 1), 0);
  poly_instance_readback_buf(inst, out_idx, &results[1], sizeof(float));

  /* CPU -> forward */
  ASSERT_INT_EQ(poly_instance_set_device(inst, POLY_DEVICE_CPU), 0);
  ASSERT_INT_EQ(poly_instance_forward(inst, io, 1), 0);
  {
    int64_t n;
    results[2] = *poly_instance_buf_data(inst, out_idx, &n);
  }

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

/* Regression: realize_ex with poly_full + an ALU BUFFER shape variable on GPU. */
/* Originally landed (pre-Phase-B) as a guard for the const-registry buffer
 * migration path: poly_full used to malloc a host buffer and stash it via
 * g_const_bindings, which only worked on GPU after the realize-time
 * migrated_consts[64] copy. Phase B (commit 6044282) rewrote poly_full as
 * a pure UOp (CONST -> reshape -> expand), and Phase E deleted the
 * const-registry entirely, so this test now exercises a different code
 * path entirely: the only HIP smoke that runs poly_test_realize_buffer_views_vars with a
 * dynamic ALU BUFFER shape. Renamed accordingly. */
TEST_BACKEND(hip, realize_ex_full_plus_buffer_dyn_shape) {
  SKIP_IF_NO_HIP();
  PolyCtx *ctx = poly_ctx_new();

  /* out[N] = full(3.14)[N] + a[N], with N symbolic. */
  PolyUOp *N = poly_uop_variable(ctx, "N", 1, 16, POLY_WEAKINT, 1, false);
  int64_t shape_max[] = {16};
  PolyUOp *fill = poly_full(ctx, shape_max, 1, 3.14);
  PolyUOp *buf_a = poly_test_buffer_var(ctx, POLY_FLOAT32, N, NULL, 0);
  PolyUOp *buf_out = poly_test_buffer_var(ctx, POLY_FLOAT32, N, NULL, 0);
  PolyUOp *add = poly_alu2(ctx, POLY_OP_ADD, fill, buf_a);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, buf_out, add, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  float a_data[16] = {1, 2, 3, 4};
  float out_data[16] = {0};
  PolyTestBufferView bindings[] = {
      POLY_TEST_HOST_VIEW(buf_a, a_data),
      POLY_TEST_HOST_VIEW(buf_out, out_data),
  };
  PolyVarBinding vars[] = {{.var = N, .value = 4}};

  int ret = poly_test_realize_buffer_views_vars(ctx, sink, bindings, 2, vars, 1);
  poly_ctx_destroy(ctx);

  ASSERT_INT_EQ(ret, 0);
  ASSERT_FLOAT_EQ(out_data[0], 4.14f, 1e-4);
  ASSERT_FLOAT_EQ(out_data[1], 5.14f, 1e-4);
  ASSERT_FLOAT_EQ(out_data[2], 6.14f, 1e-4);
  ASSERT_FLOAT_EQ(out_data[3], 7.14f, 1e-4);
  PASS();
}

static PolyUOp *hip_test_fragment(
    PolyCtx *ctx, PolyDType dtype, int lanes
) {
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, dtype, poly_arg_float(0.0));
  PolyUOp **src = malloc((size_t)lanes * sizeof(*src));
  if (!src) return NULL;
  for (int i = 0; i < lanes; i++) src[i] = zero;
  PolyUOp *fragment =
      poly_uop(ctx, POLY_OP_STACK, dtype, src, lanes, poly_arg_none());
  free(src);
  return fragment;
}

static char *hip_render_test_wmma(
    PolyDType dtype_in,
    PolyDType dtype_out,
    const int dims[3],
    const int lanes[3],
    int threads,
    const char *arch
) {
  PolyCtx *ctx = poly_ctx_new();
  if (!ctx) return NULL;
  PolyUOp *src[3] = {
      hip_test_fragment(ctx, dtype_in, lanes[0]),
      hip_test_fragment(ctx, dtype_in, lanes[1]),
      hip_test_fragment(ctx, dtype_out, lanes[2]),
  };
  PolyUOp *wmma = poly_uop(
      ctx, POLY_OP_WMMA, dtype_out, src, 3,
      poly_arg_tensor_core(dims, dtype_in, "AMD", threads, NULL, NULL, false)
  );
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, wmma, poly_arg_none());
  int n_lin = 0;
  PolyUOp **lin = poly_do_linearize(ctx, sink, &n_lin);
  char *source = lin ? poly_render_hip(ctx, lin, n_lin, "wmma_test", threads, arch) : NULL;
  free(lin);
  poly_ctx_destroy(ctx);
  return source;
}

TEST_BACKEND(hip, render_wmma_matches_current_architecture_matrix) {
  /* Tinygrad 2026-08-22/a9069c177a9d renderer/cstyle.py:528-563 emits one
   * architecture-specific binding for every wmma_args signature. */
  const int k32[] = {16, 16, 32}, k128[] = {16, 16, 128};
  const int k16[] = {16, 16, 16};
  const int cdna32_lanes[] = {8, 8, 4}, cdna128_lanes[] = {32, 32, 4};
  const int rdna3_lanes[] = {16, 16, 8}, rdna4_lanes[] = {8, 8, 8};

  char *gfx942 = hip_render_test_wmma(
      POLY_FP8E4M3, POLY_FLOAT32, k32, cdna32_lanes, 64, "gfx942"
  );
  ASSERT_NOT_NULL(gfx942);
  ASSERT_NOT_NULL(strstr(
      gfx942,
      "#define __WMMA_16_16_32_float8_e4m3_float "
      "__builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8"
  ));
  ASSERT_NOT_NULL(strstr(gfx942, ", 0, 0, 0);"));
  free(gfx942);

  char *gfx950 = hip_render_test_wmma(
      POLY_FP8E5M2, POLY_FLOAT32, k128, cdna128_lanes, 64, "gfx950"
  );
  ASSERT_NOT_NULL(gfx950);
  ASSERT_NOT_NULL(strstr(
      gfx950,
      "#define __WMMA_16_16_128_float8_e5m2_float "
      "__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4"
  ));
  ASSERT_NOT_NULL(strstr(gfx950, ", 1, 1, 0, 0, 0, 0);"));
  free(gfx950);

  char *gfx1100_i8 = hip_render_test_wmma(
      POLY_INT8, POLY_INT32, k16, rdna3_lanes, 32, "gfx1100"
  );
  ASSERT_NOT_NULL(gfx1100_i8);
  ASSERT_NOT_NULL(strstr(gfx1100_i8, "typedef int wmma_int4"));
  ASSERT_NOT_NULL(strstr(
      gfx1100_i8, "__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32"
  ));
  free(gfx1100_i8);

  char *gfx1100_f16 = hip_render_test_wmma(
      POLY_FLOAT16, POLY_FLOAT16, k16, rdna3_lanes, 32, "gfx1100"
  );
  ASSERT_NOT_NULL(gfx1100_f16);
  ASSERT_NOT_NULL(strstr(gfx1100_f16, "half16 c_frag = {};"));
  ASSERT_NOT_NULL(strstr(
      gfx1100_f16, "__builtin_amdgcn_wmma_f16_16x16x16_f16_w32"
  ));
  free(gfx1100_f16);

  char *gfx1200 = hip_render_test_wmma(
      POLY_BFLOAT16, POLY_BFLOAT16, k16, rdna4_lanes, 32, "gfx1200"
  );
  ASSERT_NOT_NULL(gfx1200);
  ASSERT_NOT_NULL(strstr(
      gfx1200,
      "#define __WMMA_16_16_16___bf16___bf16 "
      "__builtin_amdgcn_wmma_bf16_16x16x16_bf16_w32_gfx12"
  ));
  free(gfx1200);
  PASS();
}

TEST_BACKEND(hip, rewrite_bf16_wmma_preserves_native_fragments) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *out = poly_test_uop_param(ctx, POLY_FLOAT32, -1, 0, POLY_ADDR_GLOBAL);
  PolyUOp *bound =
      poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(1));
  PolyUOp *special =
      poly_uop1(ctx, POLY_OP_SPECIAL, POLY_INT32, bound, poly_arg_str("gidx0"));
  PolyUOp *out_idx =
      poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, out, special, poly_arg_none());

  PolyUOp *one =
      poly_uop0(ctx, POLY_OP_CONST, POLY_BFLOAT16, poly_arg_float(1.0));
  PolyUOp *bf16_lanes[] = {one, one, one, one};
  PolyUOp *a =
      poly_uop(ctx, POLY_OP_STACK, POLY_BFLOAT16, bf16_lanes, 4, poly_arg_none());
  PolyUOp *b =
      poly_uop(ctx, POLY_OP_STACK, POLY_BFLOAT16, bf16_lanes, 4, poly_arg_none());
  PolyUOp *zero =
      poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(0.0));
  PolyUOp *f32_lanes[] = {zero, zero, zero, zero};
  PolyUOp *acc =
      poly_uop(ctx, POLY_OP_STACK, POLY_FLOAT32, f32_lanes, 4, poly_arg_none());
  PolyUOp *wmma_src[] = {a, b, acc};
  int dims[] = {16, 16, 16};
  PolyUOp *wmma = poly_uop(
      ctx, POLY_OP_WMMA, POLY_FLOAT32, wmma_src, 3,
      poly_arg_tensor_core(dims, POLY_BFLOAT16, "AMD", 64, NULL, NULL, false)
  );
  PolyUOp *lane_idx = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(0));
  PolyUOp *lane = poly_uop_index(ctx, wmma, &lane_idx, 1);
  PolyUOp *store =
      poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out_idx, lane, poly_arg_none());
  PolyUOp *sink_src[] = {store};
  PolyUOp *sink =
      poly_uop_sink_ex(ctx, sink_src, 1, "bf16_wmma_rewrite", 0);

  PolyUOp *rewritten = poly_rewrite_hip(ctx, sink);
  ASSERT_NOT_NULL(rewritten);
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, rewritten, &n_topo);
  ASSERT_NOT_NULL(topo);
  PolyUOp *rewritten_wmma = NULL;
  int n_wmma = 0;
  for (int i = 0; i < n_topo; i++) {
    if (topo[i]->op != POLY_OP_WMMA) continue;
    rewritten_wmma = topo[i];
    n_wmma++;
  }
  ASSERT_INT_EQ(n_wmma, 1);
  ASSERT_NOT_NULL(rewritten_wmma);
  ASSERT_INT_EQ(rewritten_wmma->n_src, 3);
  ASSERT_TRUE(poly_dtype_eq(rewritten_wmma->src[0]->dtype, POLY_BFLOAT16));
  ASSERT_TRUE(poly_dtype_eq(rewritten_wmma->src[1]->dtype, POLY_BFLOAT16));

  int n_lin = 0;
  PolyUOp **lin = poly_do_linearize(ctx, rewritten, &n_lin);
  ASSERT_NOT_NULL(lin);
  char *source =
      poly_render_hip(ctx, lin, n_lin, "bf16_wmma_rewrite", 64, "gfx942");
  free(lin);
  ASSERT_NOT_NULL(source);
  ASSERT_NOT_NULL(
      strstr(source, "__builtin_amdgcn_mfma_f32_16x16x16bf16_1k")
  );

  if (poly_hip_available()) {
    PolyHipProgram *program =
        poly_compile_hip(source, "bf16_wmma_rewrite");
    ASSERT_NOT_NULL(program);
    poly_hip_program_destroy(program);
  }
  free(source);
  poly_ctx_destroy(ctx);
  PASS();
}

/* E2E MFMA smoke test: all-ones matmul on GPU */
TEST_BACKEND(hip, wmma_mfma_e2e) {
  SKIP_IF_NO_HIP();
  if (poly_hip_wave_size() != 64) {
    PASS();
  } /* CDNA wave64 only */

  /* Build hand-crafted kernel IR:
   *   Each of 64 threads loads half4 from A and B, runs mfma_f32_16x16x16f16,
   *   stores float4 result to C. With all inputs = 1.0h, every output = 16.0f. */
  PolyCtx *ctx = poly_ctx_new();

  PolyDType f16v4 = POLY_FLOAT16;
  PolyDType f32v4 = POLY_FLOAT32;

  /* Kernel params: A (half*), B (half*), C (float*) */
  PolyUOp *pA = poly_test_uop_param(ctx, POLY_FLOAT16, -1, 0, POLY_ADDR_GLOBAL);
  PolyUOp *pB = poly_test_uop_param(ctx, POLY_FLOAT16, -1, 1, POLY_ADDR_GLOBAL);
  PolyUOp *pC = poly_test_uop_param(ctx, POLY_FLOAT32, -1, 2, POLY_ADDR_GLOBAL);

  /* Thread index: lidx0 in [0, 64) */
  PolyUOp *bound64 = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(64));
  PolyUOp *tid = poly_uop1(ctx, POLY_OP_SPECIAL, POLY_INT32, bound64, poly_arg_str("lidx0"));

  /* Per-thread offset: tid * 4 */
  PolyUOp *four = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(4));
  PolyUOp *offset = poly_uop2(ctx, POLY_OP_MUL, POLY_INT32, tid, four, poly_arg_none());

  /* Load A[tid*4 .. tid*4+3] as 4 individual halves, then VECTORIZE */
  PolyUOp *a_elems[4], *b_elems[4];
  for (int i = 0; i < 4; i++) {
    PolyUOp *ci = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(i));
    PolyUOp *idx = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, offset, ci, poly_arg_none());

    PolyUOp *a_ptr = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT16, pA, idx, poly_arg_none());
    a_elems[i] = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT16, a_ptr, poly_arg_none());

    PolyUOp *b_ptr = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT16, pB, idx, poly_arg_none());
    b_elems[i] = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT16, b_ptr, poly_arg_none());
  }
  PolyUOp *a_vec = poly_uop(ctx, POLY_OP_STACK, f16v4, a_elems, 4, poly_arg_none());
  PolyUOp *b_vec = poly_uop(ctx, POLY_OP_STACK, f16v4, b_elems, 4, poly_arg_none());

  /* Zero accumulator (float4) */
  PolyUOp *zero_f32 = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(0.0));
  PolyUOp *c_elems[4] = {zero_f32, zero_f32, zero_f32, zero_f32};
  PolyUOp *c_vec = poly_uop(ctx, POLY_OP_STACK, f32v4, c_elems, 4, poly_arg_none());

  /* WMMA: D = A * B + C */
  PolyUOp *wmma_srcs[3] = {a_vec, b_vec, c_vec};
  int dims[] = {16, 16, 16};
  PolyUOp *d_vec = poly_uop(
      ctx, POLY_OP_WMMA, f32v4, wmma_srcs, 3,
      poly_arg_tensor_core(dims, POLY_FLOAT16, "AMD", 64, NULL, NULL, false)
  );

  /* Store all 4 output lanes */
  PolyUOp *stores[4];
  for (int i = 0; i < 4; i++) {
    PolyUOp *lane_idx = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(i));
    PolyUOp *lane = poly_uop_index(ctx, d_vec, &lane_idx, 1);
    PolyUOp *ci = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(i));
    PolyUOp *idx = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, offset, ci, poly_arg_none());
    PolyUOp *c_ptr = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, pC, idx, poly_arg_none());
    stores[i] = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, c_ptr, lane, poly_arg_none());
  }
  PolyUOp *sink = poly_uop(ctx, POLY_OP_SINK, POLY_VOID, stores, 4, poly_arg_none());

  /* Linearize (raw IR, no optimization passes) */
  int n_lin;
  PolyUOp **lin = poly_do_linearize(ctx, sink, &n_lin);
  ASSERT_NOT_NULL(lin);

  /* Render to HIP source */
  char *src = poly_render_hip(ctx, lin, n_lin, "mfma_e2e", 64, poly_hip_arch());
  free(lin);
  ASSERT_NOT_NULL(src);

  /* Compile to HSACO via comgr */
  PolyHipProgram *prog = poly_compile_hip(src, "mfma_e2e");
  if (!prog) {
    fprintf(stderr, "  mfma_e2e: rendered source:\n%s\n", src);
    free(src);
    poly_ctx_destroy(ctx);
    FAIL("poly_compile_hip failed for MFMA kernel");
  }
  free(src);

  /* Allocate device buffers:
   *   A: 256 halves (64 threads * 4 elements), all = 1.0h
   *   B: 256 halves, all = 1.0h
   *   C: 256 floats, zeroed */
  size_t a_bytes = 256 * sizeof(uint16_t);
  size_t c_bytes = 256 * sizeof(float);

  uint16_t h_a[256], h_b[256];
  for (int i = 0; i < 256; i++) {
    h_a[i] = 0x3C00;
    h_b[i] = 0x3C00;
  } /* 1.0h */

  void *d_a = poly_hip_alloc(a_bytes);
  void *d_b = poly_hip_alloc(a_bytes);
  void *d_c = poly_hip_alloc(c_bytes);
  ASSERT_NOT_NULL(d_a);
  ASSERT_NOT_NULL(d_b);
  ASSERT_NOT_NULL(d_c);

  poly_hip_copy_htod(d_a, h_a, a_bytes);
  poly_hip_copy_htod(d_b, h_b, a_bytes);
  poly_hip_memset(d_c, 0, c_bytes);

  /* Launch: 1 block of 64 threads (one wave) */
  void *args[3] = {&d_a, &d_b, &d_c};
  int rc = poly_hip_launch(prog, args, 3, /*grid*/ 1, 1, 1, /*block*/ 64, 1, 1);
  ASSERT_INT_EQ(rc, 0);
  rc = poly_hip_sync();
  ASSERT_INT_EQ(rc, 0);

  /* Readback and verify all 256 outputs */
  float h_c[256];
  poly_hip_copy_dtoh(h_c, d_c, c_bytes);

  int n_wrong = 0;
  for (int i = 0; i < 256; i++) {
    float diff = h_c[i] - 16.0f;
    if (diff < -0.5f || diff > 0.5f) {
      if (n_wrong < 5) fprintf(stderr, "  mfma_e2e: h_c[%d] = %.4f (expected 16.0)\n", i, h_c[i]);
      n_wrong++;
    }
  }
  if (n_wrong > 0) {
    fprintf(stderr, "  mfma_e2e: %d/256 outputs wrong\n", n_wrong);
  }

  /* Cleanup */
  poly_hip_free(d_a);
  poly_hip_free(d_b);
  poly_hip_free(d_c);
  poly_hip_program_destroy(prog);
  poly_ctx_destroy(ctx);

  ASSERT_INT_EQ(n_wrong, 0);
  PASS();
}

/* f16 helper: convert f32 to IEEE 754 half */
static uint16_t f32_to_f16(float f) {
  uint32_t x;
  memcpy(&x, &f, 4);
  uint32_t sign = (x >> 16) & 0x8000;
  int exp = ((x >> 23) & 0xFF) - 127 + 15;
  uint32_t mant = (x >> 13) & 0x3FF;
  if (exp <= 0) return (uint16_t)sign; /* underflow -> zero */
  if (exp >= 31) return (uint16_t)(sign | 0x7C00); /* overflow -> inf */
  return (uint16_t)(sign | ((uint32_t)exp << 10) | mant);
}

/* Automatic TC E2E: 16x16x16 f16 matmul through poly_test_realize_buffer_views */

TEST_BACKEND(hip, tc_auto_matmul_e2e) {
  SKIP_IF_NO_HIP();
  if (poly_hip_wave_size() != 64) {
    PASS();
  } /* CDNA wave64 only */

  /* Build 16x16x16 matmul: C[i,j] = sum_k(A[i,k] * B[k,j])
   * A, B are f16, accumulation and output C are f32.
   * Pattern: reshape + expand + MUL(f16) + CAST(f32) + REDUCE(ADD) */
  const int M = 16, N = 16, K = 16;

  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *buf_a = poly_test_buffer(ctx, POLY_FLOAT16, M * K); /* f16[256] */
  PolyUOp *buf_b = poly_test_buffer(ctx, POLY_FLOAT16, K * N); /* f16[256] */
  PolyUOp *buf_c = poly_test_buffer(ctx, POLY_FLOAT32, M * N); /* f32[256] */

  /* A: [M*K] -> [M, 1, K] -> expand [M, N, K] */
  int64_t a_3d[] = {M, 1, K};
  PolyUOp *ar = poly_reshape(ctx, buf_a, a_3d, 3);
  int64_t a_exp[] = {M, N, K};
  PolyUOp *ae = poly_expand(ctx, ar, a_exp, 3);

  /* B: [K*N] -> [K, N] -> permute(1,0) -> [N, K] -> [1, N, K] -> expand [M, N, K] */
  int64_t b_2d[] = {K, N};
  PolyUOp *br = poly_reshape(ctx, buf_b, b_2d, 2);
  int64_t b_perm[] = {1, 0};
  PolyUOp *bp = poly_permute(ctx, br, b_perm, 2);
  int64_t b_3d[] = {1, N, K};
  PolyUOp *br2 = poly_reshape(ctx, bp, b_3d, 3);
  int64_t b_exp[] = {M, N, K};
  PolyUOp *be = poly_expand(ctx, br2, b_exp, 3);

  /* MUL in f16, CAST to f32, REDUCE(ADD) on axis 2 (K) */
  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT16, ae, be, poly_arg_none());
  PolyUOp *cast = poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, mul, poly_arg_none());
  int64_t red_axes[] = {2};
  PolyUOp *sum = poly_reduce_axis(ctx, POLY_OP_ADD, cast, red_axes, 1);

  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, buf_c, sum, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  /* Host data: all ones (result = 16.0f for every element) */
  uint16_t h_a[M * K], h_b[K * N];
  for (int i = 0; i < M * K; i++)
    h_a[i] = 0x3C00; /* 1.0h */
  for (int i = 0; i < K * N; i++)
    h_b[i] = 0x3C00;
  float h_c[M * N];
  memset(h_c, 0, sizeof(h_c));

  /* Allocate HIP buffers and copy input data */
  size_t a_bytes = (size_t)(M * K) * 2; /* f16 = 2 bytes */
  size_t b_bytes = (size_t)(K * N) * 2;
  size_t c_bytes = (size_t)(M * N) * 4; /* f32 = 4 bytes */

  void *d_a = poly_hip_alloc(a_bytes);
  void *d_b = poly_hip_alloc(b_bytes);
  void *d_c = poly_hip_alloc(c_bytes);
  ASSERT_NOT_NULL(d_a);
  ASSERT_NOT_NULL(d_b);
  ASSERT_NOT_NULL(d_c);

  poly_hip_copy_htod(d_a, h_a, a_bytes);
  poly_hip_copy_htod(d_b, h_b, b_bytes);
  poly_hip_memset(d_c, 0, c_bytes);

  /* Build HIP bindings manually (f16 buffers need raw void* handling) */
  PolyTestBufferView hip_binds[3] = {
      {.buffer = buf_c, .handle = {d_c, c_bytes, POLY_DEVICE_HIP, true}},
      {.buffer = buf_a, .handle = {d_a, a_bytes, POLY_DEVICE_HIP, true}},
      {.buffer = buf_b, .handle = {d_b, b_bytes, POLY_DEVICE_HIP, true}},
  };

  /* Execute through full poly_test_realize_buffer_views path */
  setenv("TC_OPT", "1", 1);
  setenv("TC", "1", 1);
  int ret = poly_test_realize_buffer_views(ctx, sink, hip_binds, 3);
  unsetenv("TC_OPT");
  unsetenv("TC");

  if (ret != 0) {
    fprintf(stderr, "  tc_auto_matmul: poly_test_realize_buffer_views failed (ret=%d)\n", ret);
    /* Preflight diagnostic: dump the scheduled kernel shape */
    fprintf(stderr, "  (check POLY_DUMP_KERNELS=1 for kernel IR before HIP lowering)\n");
  }
  ASSERT_INT_EQ(ret, 0);

  /* Readback and verify: all 256 outputs should be 16.0f */
  poly_hip_copy_dtoh(h_c, d_c, c_bytes);

  int n_wrong = 0;
  for (int i = 0; i < M * N; i++) {
    float diff = h_c[i] - 16.0f;
    if (diff < -0.5f || diff > 0.5f) {
      if (n_wrong < 5)
        fprintf(stderr, "  tc_auto_matmul: h_c[%d] = %.4f (expected 16.0)\n", i, h_c[i]);
      n_wrong++;
    }
  }
  if (n_wrong > 0) fprintf(stderr, "  tc_auto_matmul: %d/%d outputs wrong\n", n_wrong, M * N);

  poly_hip_free(d_a);
  poly_hip_free(d_b);
  poly_hip_free(d_c);
  poly_ctx_destroy(ctx);
  ASSERT_INT_EQ(n_wrong, 0);
  PASS();
}

TEST_BACKEND(hip, tc_auto_matmul_unique_values) {
  SKIP_IF_NO_HIP();
  if (poly_hip_wave_size() != 64) {
    PASS();
  }

  /* 16x16x16 matmul with unique values to catch swizzle/lane-mapping bugs.
   * A[i][k] = (i*16+k+1) as f16, B[k][j] = (k*16+j+1) as f16.
   * C_ref[i][j] = sum_k(A[i][k] * B[k][j]) computed in f32 on CPU. */
  const int M = 16, N = 16, K = 16;

  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *buf_a = poly_test_buffer(ctx, POLY_FLOAT16, M * K);
  PolyUOp *buf_b = poly_test_buffer(ctx, POLY_FLOAT16, K * N);
  PolyUOp *buf_c = poly_test_buffer(ctx, POLY_FLOAT32, M * N);

  /* Same matmul graph as tc_auto_matmul_e2e */
  int64_t a_3d[] = {M, 1, K};
  PolyUOp *ar = poly_reshape(ctx, buf_a, a_3d, 3);
  int64_t a_exp[] = {M, N, K};
  PolyUOp *ae = poly_expand(ctx, ar, a_exp, 3);

  int64_t b_2d[] = {K, N};
  PolyUOp *br = poly_reshape(ctx, buf_b, b_2d, 2);
  int64_t b_perm[] = {1, 0};
  PolyUOp *bp = poly_permute(ctx, br, b_perm, 2);
  int64_t b_3d[] = {1, N, K};
  PolyUOp *br2 = poly_reshape(ctx, bp, b_3d, 3);
  int64_t b_exp[] = {M, N, K};
  PolyUOp *be = poly_expand(ctx, br2, b_exp, 3);

  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT16, ae, be, poly_arg_none());
  PolyUOp *cast = poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, mul, poly_arg_none());
  int64_t red_axes[] = {2};
  PolyUOp *sum = poly_reduce_axis(ctx, POLY_OP_ADD, cast, red_axes, 1);

  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, buf_c, sum, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  /* Host data: unique per-element values, scaled to 0.01x to keep products in f16 range.
   * A[i][k] = 0.01*(i*16+k+1), B[k][j] = 0.01*(k*16+j+1).
   * Max value: 0.01*256 = 2.56. Max product: ~6.5. Sum of 16: ~50. Safe for f16. */
  uint16_t h_a[M * K], h_b[K * N];
  for (int i = 0; i < M; i++)
    for (int k = 0; k < K; k++)
      h_a[i * K + k] = f32_to_f16(0.01f * (float)(i * K + k + 1));
  for (int k = 0; k < K; k++)
    for (int j = 0; j < N; j++)
      h_b[k * N + j] = f32_to_f16(0.01f * (float)(k * N + j + 1));

  /* CPU reference in f32. Use the same f16-rounded values the GPU sees. */
  float c_ref[M * N];
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      float acc = 0.0f;
      for (int k = 0; k < K; k++) {
        float a_val = 0.01f * (float)(i * K + k + 1);
        float b_val = 0.01f * (float)(k * N + j + 1);
        acc += a_val * b_val;
      }
      c_ref[i * N + j] = acc;
    }
  }

  size_t a_bytes = (size_t)(M * K) * 2;
  size_t b_bytes = (size_t)(K * N) * 2;
  size_t c_bytes = (size_t)(M * N) * 4;

  void *d_a = poly_hip_alloc(a_bytes);
  void *d_b = poly_hip_alloc(b_bytes);
  void *d_c = poly_hip_alloc(c_bytes);
  ASSERT_NOT_NULL(d_a);
  ASSERT_NOT_NULL(d_b);
  ASSERT_NOT_NULL(d_c);

  poly_hip_copy_htod(d_a, h_a, a_bytes);
  poly_hip_copy_htod(d_b, h_b, b_bytes);
  poly_hip_memset(d_c, 0, c_bytes);

  PolyTestBufferView hip_binds[3] = {
      {.buffer = buf_c, .handle = {d_c, c_bytes, POLY_DEVICE_HIP, true}},
      {.buffer = buf_a, .handle = {d_a, a_bytes, POLY_DEVICE_HIP, true}},
      {.buffer = buf_b, .handle = {d_b, b_bytes, POLY_DEVICE_HIP, true}},
  };

  setenv("TC_OPT", "1", 1);
  setenv("TC", "1", 1);
  int ret = poly_test_realize_buffer_views(ctx, sink, hip_binds, 3);
  unsetenv("TC_OPT");
  unsetenv("TC");
  ASSERT_INT_EQ(ret, 0);

  float h_c[M * N];
  poly_hip_copy_dtoh(h_c, d_c, c_bytes);

  /* Compare: relative tolerance for f16 multiply + f32 accumulation.
   * Values are small (max ~2.56) so f16 precision is good. */
  int n_wrong = 0;
  for (int i = 0; i < M * N; i++) {
    float diff = h_c[i] - c_ref[i];
    if (diff < 0) diff = -diff;
    float atol = (c_ref[i] < 0 ? -c_ref[i] : c_ref[i]) * 0.01f; /* 1% relative */
    if (atol < 0.01f) atol = 0.01f;
    if (diff > atol) {
      if (n_wrong < 5)
        fprintf(
            stderr, "  unique_matmul: h_c[%d] = %.2f, expected %.2f (diff=%.2f)\n", i, h_c[i],
            c_ref[i], diff
        );
      n_wrong++;
    }
  }
  if (n_wrong > 0) fprintf(stderr, "  unique_matmul: %d/%d outputs wrong\n", n_wrong, M * N);

  poly_hip_free(d_a);
  poly_hip_free(d_b);
  poly_hip_free(d_c);
  poly_ctx_destroy(ctx);
  ASSERT_INT_EQ(n_wrong, 0);
  PASS();
}

#endif /* POLY_HAS_HIP */
