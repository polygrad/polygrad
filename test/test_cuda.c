/*
 * test_cuda.c — CUDA renderer, runtime, and end-to-end tests
 *
 * Guarded by POLY_HAS_CUDA (compile-time) and poly_cuda_available() (runtime).
 * Tests that require a GPU are skipped gracefully if CUDA is not available.
 */

#ifdef POLY_HAS_CUDA

#include "test_harness.h"
#include "../src/bigint.h"
#include "../src/codegen/codegen.h"
#include "../src/ctx.h"
#include "../src/frontend.h"
#include "../src/device.h"
#include "../src/model.h"
#include "../src/engine/realize.h"
#include "../src/engine/jit.h"
#include "../src/engine/schedule.h"
#include "../src/schedule/rangeify.h"
#include "../src/nn.h"
#include "../src/optim.h"
#include "../src/tensor.h"
#include <string.h>

/* Renderer-only tests remain runnable without hardware; runtime tests skip. */
#define SKIP_IF_NO_CUDA()                                                                          \
  do {                                                                                             \
    if (!poly_cuda_available()) SKIP("CUDA runtime unavailable");                                  \
  } while (0)

/* Helper: build vecadd kernel IR (tensor-level) */

TEST(cuda, beam_compiler_bytes_and_device_timing) {
  SKIP_IF_NO_CUDA();
  const char *source = "extern \"C\" __global__ void test(float *out) { out[0] = 7; }";
  uint8_t *ba = NULL, *bb = NULL;
  int na = 0, nb = 0;
  PolyCudaProgram *a = poly_compile_cuda_with_binary(source, "test", &ba, &na);
  PolyCudaProgram *b = poly_compile_cuda_with_binary(source, "test", &bb, &nb);
  unsigned long long ptr = poly_cuda_alloc(sizeof(float));
  void *args[] = {&ptr};
  double elapsed = NAN;
  int rc = a && ptr ? poly_cuda_launch_timed(a, args, 1, 1, 1, 1, 1, 1, 1, &elapsed) : -1;
  float value = 0;
  if (!rc) rc = poly_cuda_copy_dtoh(&value, ptr, sizeof(value));
  bool equal = a && b && ba && bb && na > 0 && na == nb && !memcmp(ba, bb, (size_t)na);
  free(ba);
  free(bb);
  poly_cuda_program_destroy(a);
  poly_cuda_program_destroy(b);
  if (ptr) poly_cuda_free(ptr);
  ASSERT_TRUE(equal);
  ASSERT_INT_EQ(rc, 0);
  ASSERT_TRUE(isfinite(elapsed) && elapsed >= 0);
  ASSERT_FLOAT_EQ(value, 7, 0);
  PASS();
}

typedef struct {
  PolyCtx *ctx;
  PolyUOp *sink;
  PolyUOp *buf_a, *buf_b, *buf_c;
  int n;
} TensorVecadd;

static TensorVecadd make_tensor_vecadd(int n, PolyDevice device) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_test_buffer_on_device(ctx, POLY_FLOAT32, n, device);
  PolyUOp *b = poly_test_buffer_on_device(ctx, POLY_FLOAT32, n, device);
  PolyUOp *c = poly_test_buffer_on_device(ctx, POLY_FLOAT32, n, device);
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
  PolyShape s = poly_uop_max_shape(ctx, u);
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
      "float8",       "float16",       "float32",       "float64",       "float128",
      "float256",     "float512",      "make_float8",   "make_float16",  "make_float32",
      "make_float64", "make_float128", "make_float256", "make_float512",
  };
  for (int i = 0; i < (int)(sizeof(bad) / sizeof(bad[0])); i++)
    if (strstr(src, bad[i]) != NULL) return bad[i];
  return NULL;
}

static bool cuda_source_has_illegal_wide_f32_vector(const char *src) {
  return cuda_source_illegal_wide_f32_vector(src) != NULL;
}

/* Render tests */

TEST_BACKEND(cuda, native_fp8_types_match_current_renderer) {
  /* Tinygrad 2026-08-22/a9069c177a9d CUDARenderer.type_map and
   * render_kernel use CUDA FP8 types and include cuda_fp8.h. */
  const PolyDType dtypes[] = {POLY_FP8E4M3, POLY_FP8E5M2};
  const char *types[] = {"__nv_fp8_e4m3", "__nv_fp8_e5m2"};
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
    char *source = poly_render_cuda(ctx, linear, n_linear, "fp8_load", 1);
    free(linear);
    ASSERT_NOT_NULL(source);
    ASSERT_NOT_NULL(strstr(source, "#include <cuda_fp8.h>"));
    char pointer_type[64];
    snprintf(pointer_type, sizeof(pointer_type), "%s*", types[i]);
    ASSERT_NOT_NULL(strstr(source, pointer_type));
    free(source);

    PolyUOp *fp8_out = poly_test_program_param(ctx, dtypes[i], 1, 2);
    PolyUOp *f32_in = poly_test_program_param(ctx, POLY_FLOAT32, 1, 3);
    PolyUOp *fp8_out_idx = poly_uop_index(ctx, fp8_out, &zero, 1);
    PolyUOp *f32_in_idx = poly_uop_index(ctx, f32_in, &zero, 1);
    PolyUOp *f32_load = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, f32_in_idx, poly_arg_none());
    PolyUOp *to_fp8 = poly_uop1(ctx, POLY_OP_CAST, dtypes[i], f32_load, poly_arg_none());
    PolyUOp *fp8_store =
        poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, fp8_out_idx, to_fp8, poly_arg_none());
    sink = poly_test_kernel_sink(ctx, &fp8_store, 1, "fp8_store");
    linear = poly_do_linearize(ctx, sink, &n_linear);
    ASSERT_NOT_NULL(linear);
    source = poly_render_cuda(ctx, linear, n_linear, "fp8_store", 1);
    free(linear);
    ASSERT_NOT_NULL(source);
    char cast_expr[96];
    snprintf(cast_expr, sizeof(cast_expr), "(%s)(", types[i]);
    ASSERT_NOT_NULL(strstr(source, cast_expr));
    free(source);
    if (i == 0) {
      PolyUOp *nan = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKFLOAT, poly_arg_float(NAN));
      PolyUOp *fp8_inf = poly_uop1(ctx, POLY_OP_CAST, dtypes[i], nan, poly_arg_none());
      fp8_store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, fp8_out_idx, fp8_inf, poly_arg_none());
      sink = poly_test_kernel_sink(ctx, &fp8_store, 1, "fp8_const_store");
      linear = poly_do_linearize(ctx, sink, &n_linear);
      ASSERT_NOT_NULL(linear);
      source = poly_render_cuda(ctx, linear, n_linear, "fp8_const_store", 1);
      free(linear);
      ASSERT_NOT_NULL(source);
      ASSERT_NOT_NULL(strstr(source, "((__nv_fp8_e4m3)(NAN))"));
      free(source);
    }
    poly_ctx_destroy(ctx);
  }
  PASS();
}

static PolyUOp *cuda_test_fragment(PolyCtx *ctx, PolyDType dtype, int lanes, double value) {
  PolyUOp *scalar = poly_uop0(ctx, POLY_OP_CONST, dtype, poly_arg_float(value));
  PolyUOp **src = malloc((size_t)lanes * sizeof(*src));
  if (!src) return NULL;
  for (int i = 0; i < lanes; i++)
    src[i] = scalar;
  PolyUOp *fragment = poly_uop(ctx, POLY_OP_STACK, dtype, src, lanes, poly_arg_none());
  free(src);
  return fragment;
}

TEST_BACKEND(cuda, render_wmma_matches_current_inline_ptx) {
  /* Tinygrad 2026-08-22/a9069c177a9d renderer/cstyle.py:443-469 emits one
   * inline-PTX helper for every WMMA signature in the linear UOp list. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  int fp8_dims[] = {8, 16, 32};
  PolyUOp *fp8_src[] = {
      cuda_test_fragment(ctx, POLY_FP8E4M3, 16, 0.0),
      cuda_test_fragment(ctx, POLY_FP8E4M3, 8, 0.0),
      cuda_test_fragment(ctx, POLY_FLOAT32, 4, 0.0),
  };
  PolyUOp *fp8 = poly_uop(
      ctx, POLY_OP_WMMA, POLY_FLOAT32, fp8_src, 3,
      poly_arg_tensor_core(fp8_dims, POLY_FP8E4M3, "CUDA", 32, NULL, NULL, false)
  );

  int bf16_dims[] = {8, 16, 16};
  PolyUOp *bf16_src[] = {
      cuda_test_fragment(ctx, POLY_BFLOAT16, 8, 0.0),
      cuda_test_fragment(ctx, POLY_BFLOAT16, 4, 0.0),
      cuda_test_fragment(ctx, POLY_FLOAT32, 4, 0.0),
  };
  PolyUOp *bf16 = poly_uop(
      ctx, POLY_OP_WMMA, POLY_FLOAT32, bf16_src, 3,
      poly_arg_tensor_core(bf16_dims, POLY_BFLOAT16, "CUDA", 32, NULL, NULL, false)
  );

  int half_dims[] = {8, 16, 8};
  PolyUOp *half_src[] = {
      cuda_test_fragment(ctx, POLY_FLOAT16, 4, 0.0),
      cuda_test_fragment(ctx, POLY_FLOAT16, 2, 0.0),
      cuda_test_fragment(ctx, POLY_FLOAT16, 4, 0.0),
  };
  PolyUOp *half = poly_uop(
      ctx, POLY_OP_WMMA, POLY_FLOAT16, half_src, 3,
      poly_arg_tensor_core(half_dims, POLY_FLOAT16, "CUDA", 32, NULL, NULL, false)
  );

  PolyUOp *roots[] = {fp8, bf16, half};
  PolyUOp *sink = poly_uop(ctx, POLY_OP_SINK, POLY_VOID, roots, 3, poly_arg_none());
  int n_linear = 0;
  PolyUOp **linear = poly_do_linearize(ctx, sink, &n_linear);
  ASSERT_NOT_NULL(linear);
  char *source = poly_render_cuda(ctx, linear, n_linear, "wmma_render", 32);
  free(linear);
  ASSERT_NOT_NULL(source);
  ASSERT_NOT_NULL(strstr(source, "__WMMA_8_16_32_float8_e4m3_float"));
  ASSERT_NOT_NULL(strstr(source, "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32"));
  ASSERT_NOT_NULL(strstr(source, "__WMMA_8_16_16___bf16_float"));
  ASSERT_NOT_NULL(strstr(source, "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"));
  ASSERT_NOT_NULL(strstr(source, "__WMMA_8_16_8_half_half"));
  ASSERT_NOT_NULL(strstr(source, "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16"));
  free(source);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(cuda, devectorizer_keeps_aligned_float4_store_like_tinygrad) {
  /* tinygrad Renderer.supports_float4 defaults true and CUDARenderer keeps it;
   * devectorizer.py:140-184 therefore folds this contiguous lane group into
   * one vector STORE instead of GROUP(4 x STORE). */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *buf = poly_test_program_param(ctx, POLY_FLOAT32, 128, 0);
  PolyUOp *c32 = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(32));
  PolyUOp *special = poly_uop1(ctx, POLY_OP_SPECIAL, POLY_WEAKINT, c32, poly_arg_str("lidx0"));
  PolyUOp *c4 = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(4));
  PolyUOp *base = poly_uop2(ctx, POLY_OP_MUL, POLY_WEAKINT, special, c4, poly_arg_none());
  PolyUOp *stores[4];
  for (int i = 0; i < 4; i++) {
    PolyUOp *lane = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(i));
    PolyUOp *coord = poly_uop2(ctx, POLY_OP_ADD, POLY_WEAKINT, base, lane, poly_arg_none());
    PolyUOp *target = poly_uop_index(ctx, buf, &coord, 1);
    PolyUOp *value = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float((double)i));
    stores[i] = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, target, value, poly_arg_none());
  }
  /* tinygrad@2026-08-22/a9069c177a9d codegen/late/coalesce.py:125-160
   * groups adjacent scalar stores into one SHRINK-backed vector store. */
  PolyUOp *sink = poly_test_kernel_sink(ctx, stores, 4, "float4_store");

  PolyUOp *rewritten = poly_rewrite_cuda(ctx, sink);
  ASSERT_NOT_NULL(rewritten);
  int n = 0, n_store = 0, n_group = 0;
  PolyUOp **topo = poly_toposort(ctx, rewritten, &n);
  PolyUOp *final_store = NULL;
  for (int i = 0; i < n; i++) {
    if (topo[i]->op == POLY_OP_STORE) n_store++, final_store = topo[i];
    if (topo[i]->op == POLY_OP_GROUP) n_group++;
  }
  ASSERT_INT_EQ(n_store, 1);
  ASSERT_INT_EQ(n_group, 0);
  ASSERT_NOT_NULL(final_store);

  int n_lin = 0;
  PolyUOp **lin = poly_do_linearize(ctx, rewritten, &n_lin);
  ASSERT_NOT_NULL(lin);
  char *source = poly_render_cuda(ctx, lin, n_lin, "float4_store", 32);
  free(lin);
  ASSERT_NOT_NULL(source);
  /* tinygrad cstyle.py:179-184 casts the indexed scalar PARAM address to the
   * vector access dtype before dereference. */
  ASSERT_TRUE(strstr(source, "*((float4*)") != NULL);
  ASSERT_TRUE(strstr(source, " = make_float4(") != NULL);
  free(source);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(cuda, raw_cross_dtype_store_does_not_invent_late_cast) {
  /* Pinned UOp.store keeps the supplied value dtype (uop/ops.py:531-533),
   * and the late pipeline has no STORE-dtype coercion
   * (codegen/__init__.py:101-137). A malformed float* <- int value therefore
   * stays malformed; codegen must not invent a CAST after devectorization. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *out = poly_test_program_param(ctx, POLY_FLOAT32, 4, 0);
  PolyUOp *a = poly_test_program_param(ctx, POLY_INT32, 4, 1);
  PolyUOp *b = poly_test_program_param(ctx, POLY_INT32, 4, 2);
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(4));
  PolyUOp *range =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, bound, poly_arg_range(0, POLY_AXIS_GLOBAL));
  PolyUOp *out_index = poly_uop_index(ctx, out, &range, 1);
  PolyUOp *a_index = poly_uop_index(ctx, a, &range, 1);
  PolyUOp *b_index = poly_uop_index(ctx, b, &range, 1);
  PolyUOp *a_load = poly_uop1(ctx, POLY_OP_LOAD, POLY_INT32, a_index, poly_arg_none());
  PolyUOp *b_load = poly_uop1(ctx, POLY_OP_LOAD, POLY_INT32, b_index, poly_arg_none());
  PolyUOp *value = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, a_load, b_load, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out_index, value, poly_arg_none());
  PolyUOp *end = poly_uop2(ctx, POLY_OP_END, POLY_VOID, store, range, poly_arg_none());
  PolyUOp *sink = poly_test_kernel_sink(ctx, &end, 1, "cross_dtype_store");

  PolyUOp *rewritten = poly_rewrite_cuda(ctx, sink);
  ASSERT_NOT_NULL(rewritten);
  int n_topo = 0, invented_casts = 0;
  PolyUOp **topo = poly_toposort(ctx, rewritten, &n_topo);
  ASSERT_NOT_NULL(topo);
  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    if (u->op == POLY_OP_CAST && u->n_src == 1 && poly_uop_max_numel(ctx, u) == 4 &&
        poly_dtype_eq(u->dtype, POLY_FLOAT32) && poly_uop_max_numel(ctx, u->src[0]) == 4 &&
        poly_dtype_eq(u->src[0]->dtype, POLY_INT32))
      invented_casts++;
  }
  ASSERT_INT_EQ(invented_casts, 0);

  int n_linear = 0;
  PolyUOp **linear = poly_do_linearize(ctx, rewritten, &n_linear);
  ASSERT_NOT_NULL(linear);
  char *source = poly_render_cuda(ctx, linear, n_linear, "cross_dtype_store", 1);
  free(linear);
  ASSERT_NOT_NULL(source);
  ASSERT_TRUE(strstr(source, "make_int4(") != NULL);
  ASSERT_TRUE(strstr(source, "(float4)(make_int4(") == NULL);
  free(source);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(cuda, vector_local_shrink_load_store_use_typed_lvalue) {
  /* Pinned CStyleLanguage.render_access is shared by LOAD/STORE and every
   * address space (renderer/cstyle.py:47-58,179-184).  The final RMSNorm
   * reduction uses SHRINK<float4>(BUFFER<float@LOCAL>, idx, 4), so both
   * directions must dereference a float4 pointer rather than assign to the
   * scalar shared-memory address expression. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *size = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(128));
  PolyParamArg local_arg = {.slot = 0, .addrspace = POLY_ADDR_LOCAL};
  PolyParamArg global_arg = {.slot = 0, .addrspace = POLY_ADDR_GLOBAL};
  PolyUOp *local = poly_uop1(ctx, POLY_OP_BUFFER, POLY_FLOAT32, size, poly_arg_param(&local_arg));
  PolyUOp *global = poly_uop1(ctx, POLY_OP_PARAM, POLY_FLOAT32, size, poly_arg_param(&global_arg));
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(32));
  PolyUOp *idx = poly_uop1(ctx, POLY_OP_SPECIAL, POLY_INT32, bound, poly_arg_str("lidx0"));
  PolyUOp *width = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(4));
  PolyUOp *local_srcs[3] = {local, idx, width};
  PolyUOp *local_vec = poly_uop(ctx, POLY_OP_SHRINK, POLY_FLOAT32, local_srcs, 3, poly_arg_none());
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *global_srcs[3] = {global, zero, width};
  PolyUOp *global_vec =
      poly_uop(ctx, POLY_OP_SHRINK, POLY_FLOAT32, global_srcs, 3, poly_arg_none());
  PolyUOp *values[4];
  for (int i = 0; i < 4; i++)
    values[i] = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float((double)i + 1.0));
  PolyUOp *value = poly_uop(ctx, POLY_OP_STACK, POLY_FLOAT32, values, 4, poly_arg_none());
  PolyUOp *local_store =
      poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, local_vec, value, poly_arg_none());
  PolyUOp *local_load = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, local_vec, poly_arg_none());
  PolyUOp *global_store =
      poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, global_vec, local_load, poly_arg_none());
  PolyUOp *sink_srcs[2] = {local_store, global_store};
  PolyUOp *sink = poly_uop(ctx, POLY_OP_SINK, POLY_VOID, sink_srcs, 2, poly_arg_none());

  int n_lin = 0;
  PolyUOp **lin = poly_do_linearize(ctx, sink, &n_lin);
  ASSERT_NOT_NULL(lin);
  char *source = poly_render_cuda(ctx, lin, n_lin, "local_float4_access", 32);
  free(lin);
  ASSERT_NOT_NULL(source);
  const char *first = strstr(source, "*((float4*)((smem0+");
  ASSERT_NOT_NULL(first);
  ASSERT_NOT_NULL(strstr(first + 1, "*((float4*)((smem0+"));
  free(source);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(cuda, render_vecadd) {
  /* Test CUDA source generation — no GPU needed */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *p0 = poly_test_program_param(ctx, POLY_FLOAT32, 10, 0);
  PolyUOp *p1 = poly_test_program_param(ctx, POLY_FLOAT32, 10, 1);
  PolyUOp *p2 = poly_test_program_param(ctx, POLY_FLOAT32, 10, 2);

  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(10));
  PolyUOp *special = poly_uop1(ctx, POLY_OP_SPECIAL, POLY_INT32, bound, poly_arg_str("gidx0"));

  PolyUOp *idx0 = poly_uop_index(ctx, p0, &special, 1);
  PolyUOp *idx1 = poly_uop_index(ctx, p1, &special, 1);
  PolyUOp *idx2 = poly_uop_index(ctx, p2, &special, 1);

  PolyUOp *ld0 = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx0, poly_arg_none());
  PolyUOp *ld1 = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx1, poly_arg_none());
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, ld0, ld1, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx2, add, poly_arg_none());
  PolyUOp *sink = poly_test_kernel_sink(ctx, &store, 1, "test_kernel");

  int n_lin;
  PolyUOp **lin = poly_do_linearize(ctx, sink, &n_lin);
  ASSERT_NOT_NULL(lin);

  char *src = poly_render_cuda(ctx, lin, n_lin, "test_kernel", 256);
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

TEST_BACKEND(cuda, uint32_vector_uses_pinned_uint_spelling) {
  /* tinygrad@2026-08-22/a9069c177a9d CUDARenderer.type_map maps uint32 to
   * `uint`; CStyleLanguage._render_dtype appends the vector width. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyDType uint4 = POLY_UINT32;
  PolyUOp *global = poly_test_program_param(ctx, POLY_UINT32, 4, 0);
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *width = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(4));
  PolyUOp *view_src[3] = {global, zero, width};
  PolyUOp *view = poly_uop(ctx, POLY_OP_SHRINK, uint4, view_src, 3, poly_arg_none());
  PolyUOp *lanes[4];
  for (int i = 0; i < 4; i++)
    lanes[i] = poly_uop0(ctx, POLY_OP_CONST, POLY_UINT32, poly_arg_int(i));
  PolyUOp *value = poly_uop(ctx, POLY_OP_STACK, uint4, lanes, 4, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, view, value, poly_arg_none());
  PolyUOp *sink = poly_test_kernel_sink(ctx, &store, 1, "store_uint4");

  int n_lin = 0;
  PolyUOp **lin = poly_do_linearize(ctx, sink, &n_lin);
  ASSERT_NOT_NULL(lin);
  char *src = poly_render_cuda(ctx, lin, n_lin, "store_uint4", 1);
  free(lin);
  ASSERT_NOT_NULL(src);
  ASSERT_NOT_NULL(strstr(src, "typedef unsigned int uint;"));
  ASSERT_NOT_NULL(strstr(src, "uint4"));
  ASSERT_NOT_NULL(strstr(src, "make_uint4"));
  ASSERT_TRUE(strstr(src, "unsigned int4") == NULL);

  free(src);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(cuda, exact_uint64_bigint_const_renders_and_executes_like_tinygrad) {
  /* Pinned CUDARenderer keeps the exact Python integer in the UOp and emits
   * the uint64 literal at the fixed-width render boundary
   * (renderer/cstyle.py:37, dtype.py:92-100). */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyInt value = {0};
  ASSERT_TRUE(poly_int_from_decimal(&value, "18446744073709550593"));
  PolyUOp *constant = poly_uop0(ctx, POLY_OP_CONST, POLY_UINT64, poly_int_as_arg(&value));
  poly_int_free(&value);

  PolyUOp *out = poly_test_program_param(ctx, POLY_UINT64, 1, 0);
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(0));
  PolyUOp *index = poly_uop_index(ctx, out, &zero, 1);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, index, constant, poly_arg_none());
  PolyUOp *sink = poly_test_kernel_sink(ctx, &store, 1, "store_exact_uint64");
  int n_lin = 0;
  PolyUOp **lin = poly_do_linearize(ctx, sink, &n_lin);
  ASSERT_NOT_NULL(lin);
  char *src = poly_render_cuda(ctx, lin, n_lin, "store_exact_uint64", 1);
  ASSERT_NOT_NULL(src);
  ASSERT_NOT_NULL(strstr(src, "18446744073709550593ull"));

  if (poly_cuda_available()) {
    PolyCudaProgram *prog = poly_compile_cuda(src, "store_exact_uint64");
    ASSERT_NOT_NULL(prog);
    unsigned long long device = poly_cuda_alloc(sizeof(uint64_t));
    ASSERT_TRUE(device != 0);
    void *args[1] = {&device};
    ASSERT_INT_EQ(poly_cuda_launch(prog, args, 1, 1, 1, 1, 1, 1, 1), 0);
    ASSERT_INT_EQ(poly_cuda_sync(), 0);
    uint64_t output = 0;
    ASSERT_INT_EQ(poly_cuda_copy_dtoh(&output, device, sizeof(output)), 0);
    ASSERT_TRUE(output == UINT64_C(18446744073709550593));
    poly_cuda_free(device);
    poly_cuda_program_destroy(prog);
  }

  free(src);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(cuda, render_muladd_matches_pinned_cudastyle) {
  /* Pinned CUDARenderer does not advertise Ops.MULACC in code_for_op
   * (renderer/cstyle.py:389-426), so MUL+ADD remains ordered MUL+ADD. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *p0 = poly_test_program_param(ctx, POLY_FLOAT32, 4, 0);
  PolyUOp *p1 = poly_test_program_param(ctx, POLY_FLOAT32, 4, 1);
  PolyUOp *p2 = poly_test_program_param(ctx, POLY_FLOAT32, 4, 2);
  PolyUOp *p3 = poly_test_program_param(ctx, POLY_FLOAT32, 4, 3);
  PolyUOp *bound = poly_const_int(ctx, 4);
  PolyUOp *range =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, bound, poly_arg_range(0, POLY_AXIS_WEAK));
  PolyUOp *idx0 = poly_uop_index(ctx, p0, &range, 1);
  PolyUOp *idx1 = poly_uop_index(ctx, p1, &range, 1);
  PolyUOp *idx2 = poly_uop_index(ctx, p2, &range, 1);
  PolyUOp *idx3 = poly_uop_index(ctx, p3, &range, 1);
  PolyUOp *ld0 = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx0, poly_arg_none());
  PolyUOp *ld1 = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx1, poly_arg_none());
  PolyUOp *ld2 = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx2, poly_arg_none());
  /* Build MUL+ADD (no MULACC in input). */
  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, ld0, ld1, poly_arg_none());
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, mul, ld2, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx3, add, poly_arg_none());
  PolyUOp *end_src[2] = {store, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_test_kernel_sink(ctx, &end, 1, "fma_test");

  int n_lin;
  PolyUOp **lin = poly_linearize_cuda(ctx, sink, &n_lin);
  ASSERT_NOT_NULL(lin);
  ASSERT_INT_EQ(count_lin_ops(lin, n_lin, POLY_OP_MULACC), 0);
  ASSERT_TRUE(count_lin_ops(lin, n_lin, POLY_OP_MUL) >= 1);
  ASSERT_TRUE(count_lin_ops(lin, n_lin, POLY_OP_ADD) >= 1);
  char *src = poly_render_cuda(ctx, lin, n_lin, "fma_test", 256);
  free(lin);
  ASSERT_NOT_NULL(src);
  ASSERT_TRUE(strstr(src, "__fmaf_rn(") == NULL);
  free(src);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(cuda, render_half_reciprocal_and_single_consumer_match_pinned) {
  /* A non-cancelling reciprocal isolates the renderer contract. Pinned CUDA
   * keeps four devectorized float16 RECIPROCAL lanes as hrcp, and
   * CStyleLanguage._render inlines one-consumer non-WHERE ALU unless
   * EXPAND_SSA=1 (renderer/cstyle.py:127-134,232-237,411-426). The separate
   * shared-symbolic regression covers x * reciprocal(x) -> 1. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *pin = poly_test_program_param(ctx, POLY_FLOAT16, 4, 0);
  PolyUOp *pout = poly_test_program_param(ctx, POLY_FLOAT16, 4, 1);
  PolyUOp *bound = poly_const_int(ctx, 4);
  PolyUOp *range =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, bound, poly_arg_range(0, POLY_AXIS_WEAK));
  PolyUOp *in_idx = poly_uop_index(ctx, pin, &range, 1);
  PolyUOp *out_idx = poly_uop_index(ctx, pout, &range, 1);
  PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT16, in_idx, poly_arg_none());
  PolyUOp *recip = poly_uop1(ctx, POLY_OP_RECIPROCAL, POLY_FLOAT16, load, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out_idx, recip, poly_arg_none());
  PolyUOp *end_src[2] = {store, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_test_kernel_sink(ctx, &end, 1, "reciprocal_test");

  /* A renderer unit test needs an explicit Target-equivalent capability set;
   * it must not inherit a prior test's failed CUDA runtime discovery. */
  PolyRewriteOpts opts = {
      .optimize = true,
      .caps =
          {
              .device = "CUDA",
              .has_exp2 = true,
              .has_log2 = true,
              .has_sin = true,
              .supports_float16 = true,
              .supports_bfloat16 = true,
              .has_int64 = true,
              .has_local = true,
              .max_vec_width = 4,
              .global_max = {2147483647, 65535, 65535},
              .local_max = {1024, 1024, 64},
          },
      .device = POLY_DEVICE_CUDA,
      .opt_policy = POLY_OPT_HEURISTIC,
  };
  int n_lin;
  PolyUOp **lin = poly_test_full_rewrite_and_linearize_ex(ctx, sink, opts, &n_lin);
  ASSERT_NOT_NULL(lin);
  ASSERT_INT_EQ(count_lin_ops(lin, n_lin, POLY_OP_RECIPROCAL), 4);
  ASSERT_INT_EQ(count_lin_ops(lin, n_lin, POLY_OP_FDIV), 0);

  const char *old_expand_ssa = getenv("EXPAND_SSA");
  char *saved_expand_ssa = old_expand_ssa ? strdup(old_expand_ssa) : NULL;
  unsetenv("EXPAND_SSA");
  char *src = poly_render_cuda(ctx, lin, n_lin, "reciprocal_test", 256);
  ASSERT_NOT_NULL(src);
  ASSERT_TRUE(strstr(src, "hrcp(") != NULL);
  ASSERT_TRUE(strstr(src, "half alu") == NULL);
  free(src);

  setenv("EXPAND_SSA", "1", 1);
  src = poly_render_cuda(ctx, lin, n_lin, "reciprocal_test", 256);
  ASSERT_NOT_NULL(src);
  ASSERT_TRUE(strstr(src, "hrcp(") != NULL);
  ASSERT_TRUE(strstr(src, "half alu") != NULL);
  free(src);
  if (saved_expand_ssa) {
    setenv("EXPAND_SSA", saved_expand_ssa, 1);
    free(saved_expand_ssa);
  } else {
    unsetenv("EXPAND_SSA");
  }

  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(cuda, linearize_reduce_merge_shared_end) {
  /* CUDA rewrite path should merge shared reduce END chains exactly once. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *pin = poly_test_program_param(ctx, POLY_FLOAT32, 1024, 0);
  PolyUOp *pout0 = poly_test_program_param(ctx, POLY_FLOAT32, 1, 1);
  PolyUOp *pout1 = poly_test_program_param(ctx, POLY_FLOAT32, 1, 2);

  PolyUOp *bound = poly_const_int(ctx, 1024);
  PolyUOp *r0 =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, bound, poly_arg_range(0, POLY_AXIS_REDUCE));

  PolyUOp *in_idx = poly_uop_index(ctx, pin, &r0, 1);
  PolyUOp *in_ld = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, in_idx, poly_arg_none());

  PolyUOp *red0_srcs[2] = {in_ld, r0};
  PolyUOp *red1_srcs[2] = {in_ld, r0};
  PolyUOp *sum =
      poly_uop(ctx, POLY_OP_REDUCE, POLY_FLOAT32, red0_srcs, 2, poly_arg_reduce(POLY_OP_ADD, 0));
  PolyUOp *mx =
      poly_uop(ctx, POLY_OP_REDUCE, POLY_FLOAT32, red1_srcs, 2, poly_arg_reduce(POLY_OP_MAX, 0));

  PolyUOp *zero = poly_const_int(ctx, 0);
  PolyUOp *out0_idx = poly_uop_index(ctx, pout0, &zero, 1);
  PolyUOp *out1_idx = poly_uop_index(ctx, pout1, &zero, 1);
  PolyUOp *st0 = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out0_idx, sum, poly_arg_none());
  PolyUOp *st1 = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out1_idx, mx, poly_arg_none());
  /* REDUCE already ends r0. An outer END(r0) invents another lifetime and
   * pinned CFGContext rejects its cyclic ordering. Lowering generates the
   * shared reduction END chains tested below from these two stores. */
  PolyUOp *stores[2] = {st0, st1};
  PolyUOp *sink = poly_test_kernel_sink(ctx, stores, 2, "shared_reduce");

  int n = 0;
  PolyUOp **lin = poly_linearize_cuda(ctx, sink, &n);
  ASSERT_NOT_NULL(lin);
  ASSERT_TRUE(n > 0);
  int local_buffers = 0;
  for (int i = 0; i < n; i++)
    if (lin[i]->op == POLY_OP_BUFFER && lin[i]->arg.kind == POLY_ARG_PARAM && lin[i]->arg.param &&
        lin[i]->arg.param->addrspace == POLY_ADDR_LOCAL)
      local_buffers++;
  /* Pinned pm_add_buffers_local + pm_remove_vec_dtypes emits one canonical
   * BUFFER(..., ParamArg(addrspace=LOCAL)) per shared reduction. */
  ASSERT_INT_EQ(local_buffers, 2);
  ASSERT_TRUE(count_lin_ops(lin, n, POLY_OP_BARRIER) >= 2);
  ASSERT_TRUE(count_lin_ops(lin, n, POLY_OP_END) >= 1);

  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

/* CUDA binding helpers */

static void select_cuda_for_cuda_phase(PolyCtx *ctx) {
  poly_ctx_set_preferred_device(ctx, POLY_DEVICE_CUDA);
}

/* E2E tests (require GPU) */

TEST_BACKEND(cuda, tensor_realize_cuda_lazy_opens_backend_without_availability_probe) {
  PolyCtx *ctx = poly_ctx_new();
  float input[3] = {1.0f, 2.0f, 3.0f};
  int64_t shape[1] = {3};
  /* Pinned _frompy creates a deviceful source BUFFER followed by explicit
   * target COPYs (uop/ops.py:747-765). Exercise that Tensor construction path
   * instead of wrapping a logical-only raw BUFFER. */
  PolyTensor *host = poly_tensor_from_host(ctx, input, sizeof(input), POLY_FLOAT32, shape, 1);
  PolyTensor *at = poly_tensor_to_device(ctx, host, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(host);
  ASSERT_NOT_NULL(at);
  PolyTensor *cuda_a = poly_tensor_to_device(ctx, at, POLY_DEVICE_CUDA);
  ASSERT_NOT_NULL(cuda_a);
  int f32 = poly_dtype_id_by_name("float32");
  PolyTensor *one = poly_tensor_const_float_by_id(ctx, 1.0f, f32, POLY_DEVICE_CUDA);
  PolyTensor *bt = poly_tensor_alu2(ctx, POLY_OP_ADD, cuda_a, one);
  ASSERT_NOT_NULL(one);
  ASSERT_NOT_NULL(bt);

  PolyTensor *out = NULL;
  select_cuda_for_cuda_phase(ctx);
  int rc = poly_realize_tensors(ctx, &bt, 1, &out);
  if (rc != 0) {
    bool cuda_available_after_failure = poly_cuda_available();
    poly_ctx_destroy(ctx);
    if (!cuda_available_after_failure) PASS();
    FAIL("CUDA tensor realize failed even though CUDA is available");
  }
  ASSERT_PTR_EQ(out, bt);

  float got[3] = {0};
  const PolyUOp *buf = poly_uop_get_buffer_identity(poly_tensor_uop(bt));
  ASSERT_NOT_NULL(buf);
  ASSERT_INT_EQ(poly_buffer_read(ctx, (PolyUOp *)buf, got, sizeof(got)), 0);
  ASSERT_FLOAT_EQ(got[0], 2.0f, 1e-5f);
  ASSERT_FLOAT_EQ(got[1], 3.0f, 1e-5f);
  ASSERT_FLOAT_EQ(got[2], 4.0f, 1e-5f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(cuda, tensor_to_device_copy_uses_dense_call_arguments) {
  SKIP_IF_NO_CUDA();

  /* Pinned exec_copy consumes dense resolved [dest, src] arguments on every
   * backend, unlike exec_kernel's sparse ast.arg.globals selection
   * (engine/realize.py:156-184). A GPU destination does not make COPY a
   * PROGRAM call. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  float input[3] = {1.0f, 2.0f, 3.0f};
  int64_t shape[1] = {3};
  PolyTensor *host = poly_tensor_from_host(ctx, input, sizeof(input), POLY_FLOAT32, shape, 1);
  PolyTensor *cpu = poly_tensor_to_device(ctx, host, POLY_DEVICE_CPU);
  PolyTensor *two =
      poly_tensor_const_float_by_id(ctx, 2.0f, poly_dtype_id_by_name("float32"), POLY_DEVICE_CPU);
  PolyTensor *scaled = poly_tensor_alu2(ctx, POLY_OP_MUL, cpu, two);
  PolyTensor *cuda = poly_tensor_to_device(ctx, scaled, POLY_DEVICE_CUDA);
  PolyTensor *roundtrip = poly_tensor_to_device(ctx, cuda, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(host);
  ASSERT_NOT_NULL(cpu);
  ASSERT_NOT_NULL(two);
  ASSERT_NOT_NULL(scaled);
  ASSERT_NOT_NULL(cuda);
  ASSERT_NOT_NULL(roundtrip);
  ASSERT_INT_EQ(poly_tensor_uop(cuda)->op, POLY_OP_COPY);
  ASSERT_INT_EQ(poly_uop_device(poly_tensor_uop(cuda)), POLY_DEVICE_CUDA);
  ASSERT_INT_EQ(poly_tensor_uop(roundtrip)->op, POLY_OP_COPY);
  ASSERT_INT_EQ(poly_uop_device(poly_tensor_uop(roundtrip)), POLY_DEVICE_CPU);

  PolyTensor *out = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &roundtrip, 1, &out), 0);
  ASSERT_PTR_EQ(out, roundtrip);
  float got[3] = {0};
  const PolyUOp *buffer = poly_uop_get_buffer_identity(poly_tensor_uop(roundtrip));
  ASSERT_NOT_NULL(buffer);
  ASSERT_INT_EQ(poly_buffer_read(ctx, (PolyUOp *)buffer, got, sizeof(got)), 0);
  ASSERT_FLOAT_EQ(got[0], 2.0f, 1e-5f);
  ASSERT_FLOAT_EQ(got[1], 4.0f, 1e-5f);
  ASSERT_FLOAT_EQ(got[2], 6.0f, 1e-5f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(cuda, placed_host_gather_memory_plan_keeps_cuda_staging) {
  SKIP_IF_NO_CUDA();

  PolyCtx *ctx = poly_ctx_new();
  float x_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
  int32_t index_data[] = {0, 0, 1, 0};
  int64_t shape[] = {2, 2};
  PolyTensor *x_host = poly_tensor_from_host(ctx, x_data, sizeof(x_data), POLY_FLOAT32, shape, 2);
  PolyTensor *index_host =
      poly_tensor_from_host(ctx, index_data, sizeof(index_data), POLY_INT32, shape, 2);
  ASSERT_NOT_NULL(x_host);
  ASSERT_NOT_NULL(index_host);
  PolyTensor *x_cpu = poly_tensor_to_device(ctx, x_host, POLY_DEVICE_CPU);
  PolyTensor *index_cpu = poly_tensor_to_device(ctx, index_host, POLY_DEVICE_CPU);
  PolyTensor *x_cuda = poly_tensor_to_device(ctx, x_cpu, POLY_DEVICE_CUDA);
  PolyTensor *index_cuda = poly_tensor_to_device(ctx, index_cpu, POLY_DEVICE_CUDA);
  ASSERT_NOT_NULL(x_cuda);
  ASSERT_NOT_NULL(index_cuda);

  PolyTensor *result = poly_tensor_gather_dim(ctx, x_cuda, 1, index_cuda);
  ASSERT_NOT_NULL(result);

  select_cuda_for_cuda_phase(ctx);
  PolyTensor *out = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &result, 1, &out), 0);
  ASSERT_PTR_EQ(out, result);

  const PolyUOp *out_buf = poly_uop_get_buffer_identity(poly_tensor_uop(result));
  ASSERT_NOT_NULL(out_buf);
  float got[4] = {0};
  ASSERT_INT_EQ(poly_buffer_read(ctx, (PolyUOp *)out_buf, got, sizeof(got)), 0);
  const float expected[] = {1.0f, 1.0f, 4.0f, 3.0f};
  for (int i = 0; i < 4; i++)
    ASSERT_FLOAT_EQ(got[i], expected[i], 1e-5f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(cuda, direct_alloc_lazy_opens_backend_without_availability_probe) {
  unsigned long long dptr = poly_cuda_alloc(sizeof(float));
  if (!dptr) {
    if (!poly_cuda_available()) PASS();
    FAIL("poly_cuda_alloc did not lazy-open an available CUDA backend");
  }
  poly_cuda_free(dptr);
  PASS();
}

TEST_BACKEND(cuda, gated_output_store_preserves_current_residency) {
  SKIP_IF_NO_CUDA();

  /* Tinygrad 2026-08-22 a9069c17 UOp.valid encodes the gate in INDEX as
   * WHERE(gate, index, Invalid); memory coalescing rejects gated STOREs. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *out = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 1, POLY_DEVICE_CUDA);
  ASSERT_NOT_NULL(out);
  float initial = 11.0f;
  poly_buffer_set(ctx, out, &initial, sizeof(initial), POLY_DEVICE_CPU);

  PolyUOp *p0 = poly_test_program_param(ctx, POLY_FLOAT32, 1, 0);
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(0));
  PolyUOp *index = poly_uop_index(ctx, p0, &zero, 1);
  PolyUOp *value = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(42.0));
  PolyUOp *gate = poly_uop0(ctx, POLY_OP_CONST, POLY_BOOL, poly_arg_bool(false));
  PolyUOp *invalid = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_invalid());
  PolyUOp *gated_offset =
      poly_uop3(ctx, POLY_OP_WHERE, POLY_WEAKINT, gate, zero, invalid, poly_arg_none());
  index = poly_uop_index(ctx, p0, &gated_offset, 1);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, index, value, poly_arg_none());
  ASSERT_INT_EQ(store->n_src, 2);
  ASSERT_EQ(store->src[0]->op, POLY_OP_INDEX);
  ASSERT_EQ(store->src[0]->src[1]->op, POLY_OP_WHERE);
  PolyUOp *body = poly_test_kernel_sink(ctx, &store, 1, "gated_store");
  PolyUOp *call_src[] = {body, out};
  PolyUOp *call = poly_uop(ctx, POLY_OP_CALL, POLY_VOID, call_src, 2, poly_arg_none());
  PolyUOp *linear = poly_uop1(ctx, POLY_OP_LINEAR, POLY_VOID, call, poly_arg_none());
  ASSERT_NOT_NULL(linear);

  ASSERT_INT_EQ(poly_run_linear(ctx, linear, NULL, 0, NULL, 0, true, false, false), 0);

  float got = 0.0f;
  ASSERT_INT_EQ(poly_buffer_read(ctx, out, &got, sizeof(got)), 0);
  ASSERT_FLOAT_EQ(got, 11.0f, 0.0f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(cuda, sparse_program_globals_bind_exact_call_slots) {
  SKIP_IF_NO_CUDA();

  /* Pinned ProgramInfo records only reachable PARAM slots, and exec_kernel
   * passes bufs[i] for exactly those globals (uop/ops.py:1138-1158,
   * engine/realize.py:176-184). Direct CUDA formal parameters are compact, so
   * sparse slots 0,3,4 must not receive dense CALL slots 0,1,2. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *out = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 2, POLY_DEVICE_CUDA);
  PolyUOp *unused1 = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 1, POLY_DEVICE_CUDA);
  PolyUOp *unused2 = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 1, POLY_DEVICE_CUDA);
  PolyUOp *src3 = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 1, POLY_DEVICE_CUDA);
  PolyUOp *src4 = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 1, POLY_DEVICE_CUDA);
  ASSERT_NOT_NULL(out);
  ASSERT_NOT_NULL(unused1);
  ASSERT_NOT_NULL(unused2);
  ASSERT_NOT_NULL(src3);
  ASSERT_NOT_NULL(src4);
  float out_init[2] = {0.0f, 0.0f}, u1 = 11.0f, u2 = 22.0f, v3 = 5.0f, v4 = 4.0f;
  poly_buffer_set(ctx, out, out_init, sizeof(out_init), POLY_DEVICE_CPU);
  poly_buffer_set(ctx, unused1, &u1, sizeof(u1), POLY_DEVICE_CPU);
  poly_buffer_set(ctx, unused2, &u2, sizeof(u2), POLY_DEVICE_CPU);
  poly_buffer_set(ctx, src3, &v3, sizeof(v3), POLY_DEVICE_CPU);
  poly_buffer_set(ctx, src4, &v4, sizeof(v4), POLY_DEVICE_CPU);

  PolyUOp *p0 = poly_test_program_param(ctx, POLY_FLOAT32, 2, 0);
  PolyUOp *p3 = poly_test_program_param(ctx, POLY_FLOAT32, 1, 3);
  PolyUOp *p4 = poly_test_program_param(ctx, POLY_FLOAT32, 1, 4);
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(0));
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(1));
  PolyUOp *dst0 = poly_uop_index(ctx, p0, &zero, 1);
  PolyUOp *dst1 = poly_uop_index(ctx, p0, &one, 1);
  PolyUOp *idx3 = poly_uop_index(ctx, p3, &zero, 1);
  PolyUOp *idx4 = poly_uop_index(ctx, p4, &zero, 1);
  PolyUOp *load3 = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx3, poly_arg_none());
  PolyUOp *load4 = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx4, poly_arg_none());
  PolyUOp *store0 = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, dst0, load3, poly_arg_none());
  PolyUOp *store1 = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, dst1, load4, poly_arg_none());
  PolyUOp *body_src[] = {store0, store1};
  PolyUOp *body = poly_test_kernel_sink(ctx, body_src, 2, "sparse_globals");
  PolyUOp *call_src[] = {body, out, unused1, unused2, src3, src4};
  PolyUOp *call = poly_uop(ctx, POLY_OP_CALL, POLY_VOID, call_src, 6, poly_arg_none());
  PolyUOp *linear = poly_uop1(ctx, POLY_OP_LINEAR, POLY_VOID, call, poly_arg_none());
  ASSERT_NOT_NULL(linear);

  PolyUOp *compiled = poly_compile_linear(ctx, linear, -1);
  ASSERT_NOT_NULL(compiled);
  const PolyProgramInfo *info = poly_program_info(ctx, poly_test_linear_call_body(compiled, 0));
  ASSERT_NOT_NULL(info);
  ASSERT_INT_EQ(info->n_globals, 3);
  ASSERT_INT_EQ(info->globals[0], 0);
  ASSERT_INT_EQ(info->globals[1], 3);
  ASSERT_INT_EQ(info->globals[2], 4);
  ASSERT_INT_EQ(poly_run_linear(ctx, compiled, NULL, 0, NULL, 0, true, true, false), 0);

  float got[2] = {0.0f, 0.0f};
  ASSERT_INT_EQ(poly_buffer_read(ctx, out, got, sizeof(got)), 0);
  ASSERT_FLOAT_EQ(got[0], 5.0f, 0.0f);
  ASSERT_FLOAT_EQ(got[1], 4.0f, 0.0f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(cuda, e2e_vecadd) {
  SKIP_IF_NO_CUDA();

  int n = 1024;
  TensorVecadd tv = make_tensor_vecadd(n, POLY_DEVICE_CUDA);

  float *a = malloc(n * sizeof(float));
  float *b = malloc(n * sizeof(float));
  float *c_gpu = calloc(n, sizeof(float));

  for (int i = 0; i < n; i++) {
    a[i] = (float)i * 0.1f;
    b[i] = (float)(n - i) * 0.05f;
  }

  ASSERT_INT_EQ(poly_buffer_write(tv.ctx, tv.buf_a, a, n * sizeof(float)), 0);
  ASSERT_INT_EQ(poly_buffer_write(tv.ctx, tv.buf_b, b, n * sizeof(float)), 0);
  ASSERT_INT_EQ(poly_realize_sink(tv.ctx, tv.sink), 0);
  ASSERT_INT_EQ(poly_buffer_read(tv.ctx, tv.buf_c, c_gpu, n * sizeof(float)), 0);

  for (int i = 0; i < n; i++)
    ASSERT_FLOAT_EQ(c_gpu[i], a[i] + b[i], 1e-5);

  free(a);
  free(b);
  free(c_gpu);
  poly_ctx_destroy(tv.ctx);
  PASS();
}

TEST_BACKEND(cuda, e2e_neg) {
  SKIP_IF_NO_CUDA();

  int n = 512;
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *buf_a = poly_test_buffer_on_device(ctx, POLY_FLOAT32, n, POLY_DEVICE_CUDA);
  PolyUOp *buf_c = poly_test_buffer_on_device(ctx, POLY_FLOAT32, n, POLY_DEVICE_CUDA);
  PolyUOp *neg = poly_alu1(ctx, POLY_OP_NEG, buf_a);
  PolyUOp *store = poly_store_val(ctx, buf_c, neg);
  PolyUOp *sink = poly_sink1(ctx, store);

  float *a = malloc(n * sizeof(float));
  float *c_gpu = calloc(n, sizeof(float));
  for (int i = 0; i < n; i++)
    a[i] = (float)i - 256.0f;

  ASSERT_INT_EQ(poly_buffer_write(ctx, buf_a, a, n * sizeof(float)), 0);
  ASSERT_INT_EQ(poly_realize_sink(ctx, sink), 0);
  ASSERT_INT_EQ(poly_buffer_read(ctx, buf_c, c_gpu, n * sizeof(float)), 0);

  for (int i = 0; i < n; i++)
    ASSERT_FLOAT_EQ(c_gpu[i], -a[i], 1e-5);

  free(a);
  free(c_gpu);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(cuda, e2e_chain) {
  SKIP_IF_NO_CUDA();

  int n = 256;
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *buf_a = poly_test_buffer_on_device(ctx, POLY_FLOAT32, n, POLY_DEVICE_CUDA);
  PolyUOp *buf_b = poly_test_buffer_on_device(ctx, POLY_FLOAT32, n, POLY_DEVICE_CUDA);
  PolyUOp *buf_c = poly_test_buffer_on_device(ctx, POLY_FLOAT32, n, POLY_DEVICE_CUDA);
  /* c = (a + b) * a */
  PolyUOp *add = poly_alu2(ctx, POLY_OP_ADD, buf_a, buf_b);
  PolyUOp *mul = poly_alu2(ctx, POLY_OP_MUL, add, buf_a);
  PolyUOp *store = poly_store_val(ctx, buf_c, mul);
  PolyUOp *sink = poly_sink1(ctx, store);

  float *a = malloc(n * sizeof(float));
  float *b = malloc(n * sizeof(float));
  float *c_gpu = calloc(n, sizeof(float));
  for (int i = 0; i < n; i++) {
    a[i] = (float)i * 0.01f;
    b[i] = 1.0f;
  }

  ASSERT_INT_EQ(poly_buffer_write(ctx, buf_a, a, n * sizeof(float)), 0);
  ASSERT_INT_EQ(poly_buffer_write(ctx, buf_b, b, n * sizeof(float)), 0);
  ASSERT_INT_EQ(poly_realize_sink(ctx, sink), 0);
  ASSERT_INT_EQ(poly_buffer_read(ctx, buf_c, c_gpu, n * sizeof(float)), 0);

  for (int i = 0; i < n; i++)
    ASSERT_FLOAT_EQ(c_gpu[i], (a[i] + b[i]) * a[i], 1e-5);

  free(a);
  free(b);
  free(c_gpu);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(cuda, e2e_exp2) {
  SKIP_IF_NO_CUDA();

  int n = 256;
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *buf_a = poly_test_buffer_on_device(ctx, POLY_FLOAT32, n, POLY_DEVICE_CUDA);
  PolyUOp *buf_c = poly_test_buffer_on_device(ctx, POLY_FLOAT32, n, POLY_DEVICE_CUDA);
  PolyUOp *exp = poly_alu1(ctx, POLY_OP_EXP2, buf_a);
  PolyUOp *store = poly_store_val(ctx, buf_c, exp);
  PolyUOp *sink = poly_sink1(ctx, store);

  float *a = malloc(n * sizeof(float));
  float *c_gpu = calloc(n, sizeof(float));
  for (int i = 0; i < n; i++)
    a[i] = (float)i * 0.05f - 6.0f;

  ASSERT_INT_EQ(poly_buffer_write(ctx, buf_a, a, n * sizeof(float)), 0);
  ASSERT_INT_EQ(poly_realize_sink(ctx, sink), 0);
  ASSERT_INT_EQ(poly_buffer_read(ctx, buf_c, c_gpu, n * sizeof(float)), 0);

  for (int i = 0; i < n; i++)
    ASSERT_FLOAT_EQ(c_gpu[i], exp2f(a[i]), 1e-4);

  free(a);
  free(c_gpu);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(cuda, e2e_reduce_sum) {
  SKIP_IF_NO_CUDA();

  int n = 512;
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *buf_a = poly_test_buffer_on_device(ctx, POLY_FLOAT32, n, POLY_DEVICE_CUDA);
  PolyUOp *buf_c = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 1, POLY_DEVICE_CUDA);
  int64_t axes[] = {0};
  PolyUOp *red = poly_reduce_axis(ctx, POLY_OP_ADD, buf_a, axes, 1);
  PolyUOp *store = poly_store_val(ctx, buf_c, red);
  PolyUOp *sink = poly_sink1(ctx, store);

  float *a = malloc(n * sizeof(float));
  float c_gpu = 0;
  for (int i = 0; i < n; i++)
    a[i] = 1.0f;

  ASSERT_INT_EQ(poly_buffer_write(ctx, buf_a, a, n * sizeof(float)), 0);
  ASSERT_INT_EQ(poly_realize_sink(ctx, sink), 0);
  ASSERT_INT_EQ(poly_buffer_read(ctx, buf_c, &c_gpu, sizeof(float)), 0);

  ASSERT_FLOAT_EQ(c_gpu, 512.0f, 1e-2); /* reduce sums can accumulate fp error */

  free(a);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(cuda, e2e_reduce_sum_parallel) {
  SKIP_IF_NO_CUDA();

  /* Large N triggers parallel reduction (N > block_size * 2 = 512) */
  int n = 10000;
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *buf_a = poly_test_buffer_on_device(ctx, POLY_FLOAT32, n, POLY_DEVICE_CUDA);
  PolyUOp *buf_c = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 1, POLY_DEVICE_CUDA);
  int64_t axes[] = {0};
  PolyUOp *red = poly_reduce_axis(ctx, POLY_OP_ADD, buf_a, axes, 1);
  PolyUOp *store = poly_store_val(ctx, buf_c, red);
  PolyUOp *sink = poly_sink1(ctx, store);

  float *a = malloc(n * sizeof(float));
  float c_gpu = 0;
  for (int i = 0; i < n; i++)
    a[i] = 1.0f;

  ASSERT_INT_EQ(poly_buffer_write(ctx, buf_a, a, n * sizeof(float)), 0);
  ASSERT_INT_EQ(poly_realize_sink(ctx, sink), 0);
  ASSERT_INT_EQ(poly_buffer_read(ctx, buf_c, &c_gpu, sizeof(float)), 0);

  /* Parallel reduce may have slightly different FP rounding */
  ASSERT_FLOAT_EQ(c_gpu, 10000.0f, 1.0f);

  free(a);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(cuda, realize_unified_vecadd) {
  SKIP_IF_NO_CUDA();

  int n = 1024;
  TensorVecadd tv = make_tensor_vecadd(n, POLY_DEVICE_CUDA);

  float *a = malloc(n * sizeof(float));
  float *b = malloc(n * sizeof(float));
  float *c_gpu = calloc(n, sizeof(float));
  for (int i = 0; i < n; i++) {
    a[i] = (float)i * 0.1f;
    b[i] = (float)(n - i) * 0.05f;
  }

  ASSERT_INT_EQ(poly_buffer_write(tv.ctx, tv.buf_a, a, n * sizeof(float)), 0);
  ASSERT_INT_EQ(poly_buffer_write(tv.ctx, tv.buf_b, b, n * sizeof(float)), 0);
  PolyVarBinding *vars = NULL;
  int n_vars = 0;
  PolyUOp *linear = poly_linear_effect_sink(tv.ctx, tv.sink, &vars, &n_vars);
  ASSERT_NOT_NULL(linear);
  ASSERT_INT_EQ(poly_run_linear(tv.ctx, linear, vars, n_vars, NULL, 0, true, false, false), 0);
  ASSERT_INT_EQ(poly_buffer_read(tv.ctx, tv.buf_c, c_gpu, n * sizeof(float)), 0);

  for (int i = 0; i < n; i++)
    ASSERT_FLOAT_EQ(c_gpu[i], a[i] + b[i], 1e-5);

  free(vars);
  free(a);
  free(b);
  free(c_gpu);
  poly_ctx_destroy(tv.ctx);
  PASS();
}

TEST_BACKEND(cuda, tensor_place_computed_expression_to_cuda_e2e) {
  SKIP_IF_NO_CUDA();

  PolyCtx *ctx = poly_ctx_new();
  float input[3] = {1.0f, 2.0f, 3.0f};
  int64_t shape[1] = {3};
  int f32 = poly_dtype_id_by_name("float32");
  PolyTensor *host = poly_tensor_from_host(ctx, input, sizeof(input), POLY_FLOAT32, shape, 1);
  PolyTensor *at = poly_tensor_to_device(ctx, host, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(at);
  PolyTensor *two = poly_tensor_const_float_by_id(ctx, 2.0f, f32, POLY_DEVICE_CPU);
  PolyTensor *mt = poly_tensor_alu2(ctx, POLY_OP_MUL, at, two);
  ASSERT_NOT_NULL(two);
  ASSERT_NOT_NULL(mt);
  PolyTensor *cuda_t = poly_tensor_to_device(ctx, mt, POLY_DEVICE_CUDA);
  ASSERT_NOT_NULL(cuda_t);

  PolyTensor *out = NULL;
  select_cuda_for_cuda_phase(ctx);
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

TEST_BACKEND(cuda, device_less_constant_copy_matches_pinned_single_call_schedule) {
  SKIP_IF_NO_CUDA();

  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  select_cuda_for_cuda_phase(ctx);

  PolyUOp *ones = poly_full(ctx, (int64_t[]){4}, 1, 1.0);
  PolyUOp *contiguous = poly_contiguous(ctx, ones);
  ASSERT_NOT_NULL(ones);
  ASSERT_NOT_NULL(contiguous);

  /* Tinygrad 2026-08-22/a9069c177a9d Tensor.to rejects this route by returning
   * self for device-free values. Direct UOp.copy_to_device still constructs
   * one scheduled SINK call and does not materialize a separate producer. */
  PolyUOp *physical =
      poly_copy_to_device_uop(ctx, contiguous, poly_device_uop(ctx, POLY_DEVICE_CUDA));
  ASSERT_NOT_NULL(physical);
  ASSERT_INT_EQ(physical->op, POLY_OP_COPY);

  PolyUOp *scheduled_out = NULL;
  PolyUOp *schedule = poly_test_linear_values(ctx, &physical, 1, &scheduled_out);
  ASSERT_NOT_NULL(schedule);
  ASSERT_NOT_NULL(scheduled_out);
  ASSERT_INT_EQ(schedule->n_src, 1);
  ASSERT_INT_EQ(poly_test_linear_call_body(schedule, 0)->op, POLY_OP_SINK);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(cuda, tensor_place_computed_expression_cuda_cpu_roundtrip_e2e) {
  SKIP_IF_NO_CUDA();

  PolyCtx *ctx = poly_ctx_new();
  float input[3] = {1.0f, 2.0f, 3.0f};
  int64_t shape[1] = {3};
  int f32 = poly_dtype_id_by_name("float32");
  PolyTensor *host = poly_tensor_from_host(ctx, input, sizeof(input), POLY_FLOAT32, shape, 1);
  PolyTensor *at = poly_tensor_to_device(ctx, host, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(at);
  PolyTensor *one = poly_tensor_const_float_by_id(ctx, 1.0f, f32, POLY_DEVICE_CPU);
  PolyTensor *xt = poly_tensor_alu2(ctx, POLY_OP_ADD, at, one);
  ASSERT_NOT_NULL(one);
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

TEST_BACKEND(cuda, realize_unified_reduce) {
  SKIP_IF_NO_CUDA();

  int n = 256;
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *buf_a = poly_test_buffer_on_device(ctx, POLY_FLOAT32, n, POLY_DEVICE_CUDA);
  PolyUOp *buf_c = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 1, POLY_DEVICE_CUDA);
  int64_t axes[] = {0};
  PolyUOp *s = poly_reduce_axis(ctx, POLY_OP_ADD, buf_a, axes, 1);
  PolyUOp *store = poly_store_val(ctx, buf_c, s);
  PolyUOp *sink = poly_sink1(ctx, store);

  float *a = malloc(n * sizeof(float));
  float c_gpu = 0;
  float expected = 0;
  for (int i = 0; i < n; i++) {
    a[i] = (float)(i + 1);
    expected += a[i];
  }

  ASSERT_INT_EQ(poly_buffer_write(ctx, buf_a, a, n * sizeof(float)), 0);
  PolyVarBinding *vars = NULL;
  int n_vars = 0;
  PolyUOp *linear = poly_linear_effect_sink(ctx, sink, &vars, &n_vars);
  ASSERT_NOT_NULL(linear);
  ASSERT_INT_EQ(poly_run_linear(ctx, linear, vars, n_vars, NULL, 0, true, false, false), 0);
  ASSERT_INT_EQ(poly_buffer_read(ctx, buf_c, &c_gpu, sizeof(float)), 0);

  ASSERT_FLOAT_EQ(c_gpu, expected, 1.0f);

  free(vars);
  free(a);
  poly_ctx_destroy(ctx);
  PASS();
}

/* ══════════════════════════════════════════════════════════════════════ */
/*  Model-level CUDA tests                                            */
/* ══════════════════════════════════════════════════════════════════════ */

#include "../src/model.h"
#include "../src/models/mlp.h"

static PolyModel *make_test_mlp(int n_in, int n_out) {
  char spec[256];
  snprintf(
      spec, sizeof(spec),
      "{\"layers\":[%d,4,%d],\"activation\":\"relu\",\"bias\":true,"
      "\"loss\":\"mse\",\"batch_size\":1,\"seed\":42}",
      n_in, n_out
  );
  return poly_mlp_from_json(spec, (int)strlen(spec), POLY_DEVICE_AUTO);
}

TEST_BACKEND(cuda, large_mlp_train_cuda_codegen_no_wide_f32_vectors) {
  /* Codegen-only regression for the larger MLP train step. tinygrad lowers
   * this graph without render-visible f32 vector typedefs like float128 or
   * float512; CUDA C has no such vector names, so they must be scalarized or
   * split before rendering. No CUDA device is required for this test. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyModel *inst = poly_model_new(ctx, NULL);
  ASSERT_NOT_NULL(inst);

  int64_t x_shape[2] = {32, 128};
  int64_t y_shape[2] = {32, 64};
  PolyTensor *x_tensor = poly_model_input(inst, "x", POLY_FLOAT32, x_shape, 2);
  PolyTensor *y_tensor = poly_model_target(inst, "y", POLY_FLOAT32, y_shape, 2);
  ASSERT_NOT_NULL(x_tensor);
  ASSERT_NOT_NULL(y_tensor);

  PolyTensor *w0_tensor = NULL, *b0_tensor = NULL, *w1_tensor = NULL, *b1_tensor = NULL;
  int64_t w0_shape[2] = {256, 128};
  int64_t b0_shape[1] = {256};
  int64_t w1_shape[2] = {64, 256};
  int64_t b1_shape[1] = {64};

  ASSERT_INT_EQ(poly_model_scope_push(inst, "layers.0"), POLY_STATUS_OK);
  w0_tensor = poly_model_param(inst, "weight", POLY_FLOAT32, w0_shape, 2);
  b0_tensor = poly_model_param(inst, "bias", POLY_FLOAT32, b0_shape, 1);
  ASSERT_INT_EQ(poly_model_scope_pop(inst), POLY_STATUS_OK);
  ASSERT_NOT_NULL(w0_tensor);
  ASSERT_NOT_NULL(b0_tensor);

  ASSERT_INT_EQ(poly_model_scope_push(inst, "layers.1"), POLY_STATUS_OK);
  w1_tensor = poly_model_param(inst, "weight", POLY_FLOAT32, w1_shape, 2);
  b1_tensor = poly_model_param(inst, "bias", POLY_FLOAT32, b1_shape, 1);
  ASSERT_INT_EQ(poly_model_scope_pop(inst), POLY_STATUS_OK);
  ASSERT_NOT_NULL(w1_tensor);
  ASSERT_NOT_NULL(b1_tensor);

  PolyUOp *x = poly_tensor_uop(x_tensor);
  x = poly_linear_apply(ctx, x, poly_tensor_uop(w0_tensor), poly_tensor_uop(b0_tensor));
  ASSERT_NOT_NULL(x);
  x = poly_relu(ctx, x);
  ASSERT_NOT_NULL(x);
  PolyUOp *pred = poly_linear_apply(ctx, x, poly_tensor_uop(w1_tensor), poly_tensor_uop(b1_tensor));
  ASSERT_NOT_NULL(pred);
  PolyUOp *target = poly_tensor_uop(y_tensor);
  PolyUOp *loss = poly_mse_loss(ctx, pred, target);
  ASSERT_NOT_NULL(loss);

  PolyUOp *param_bufs[4] = {
      (PolyUOp *)poly_uop_get_buffer_identity(poly_tensor_uop(w0_tensor)),
      (PolyUOp *)poly_uop_get_buffer_identity(poly_tensor_uop(b0_tensor)),
      (PolyUOp *)poly_uop_get_buffer_identity(poly_tensor_uop(w1_tensor)),
      (PolyUOp *)poly_uop_get_buffer_identity(poly_tensor_uop(b1_tensor)),
  };
  int64_t param_numels[4] = {256 * 128, 256, 64 * 256, 64};
  PolyUOp *wrts[4] = {
      poly_tensor_uop(w0_tensor),
      poly_tensor_uop(b0_tensor),
      poly_tensor_uop(w1_tensor),
      poly_tensor_uop(b1_tensor),
  };
  for (int i = 0; i < 4; i++) {
    ASSERT_NOT_NULL(param_bufs[i]);
    ASSERT_NOT_NULL(wrts[i]);
  }

  PolyUOp *grads[4] = {0};
  uint8_t grad_present[4] = {0};
  ASSERT_INT_EQ(poly_grad_many_ex(ctx, loss, NULL, wrts, 4, grads, grad_present), 0);
  for (int i = 0; i < 4; i++)
    ASSERT_INT_EQ(grad_present[i], 1);

  PolyUOp *stores[5];
  PolyUOp *loss_out = poly_test_buffer(ctx, POLY_FLOAT32, 1);
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
      .momentum = 0.0f,
      .weight_decay = 0.0f,
      .classic = false,
  };
  PolyUOp *lr = poly_const_float(ctx, 0.01);
  ASSERT_NOT_NULL(lr);
  for (int i = 0; i < 4; i++) {
    ASSERT_NOT_NULL(grads[i]);
    PolyUOp *grad = grads[i];
    PolyShape gs = poly_uop_max_shape(ctx, grad);
    if (gs.ndim > 1 || (gs.ndim == 1 && gs.dims && gs.dims[0] != param_numels[i])) {
      int64_t flat[1] = {param_numels[i]};
      grad = poly_reshape(ctx, grad, flat, 1);
    }
    if (gs.dims) free(gs.dims);

    PolyOptimUpdate upd;
    ASSERT_INT_EQ(
        poly_optim_build_update(
            ctx, &cfg, lr, param_bufs[i], grad, NULL, NULL, NULL, NULL, param_numels[i], &upd
        ),
        0
    );
    stores[i + 1] = poly_store_buffer_update(ctx, param_bufs[i], upd.param_new);
    ASSERT_NOT_NULL(stores[i + 1]);
  }

  PolyUOp *sink = poly_sink_n(ctx, stores, 5);
  ASSERT_NOT_NULL(sink);
  PolyUOp *sched = poly_test_create_linear(ctx, sink);
  ASSERT_NOT_NULL(sched);
  ASSERT_INT_EQ(sched->n_src, 12);

  /* Current tinygrad retains the first [32,256] linear output as one kernel:
   * SINK(END(STORE(...), batch_range, output_range)). Assert the exact
   * operation topology so this codegen test cannot silently bypass callify or
   * fuse the first two reductions again. */
  PolyUOp *first_body = poly_test_linear_call_body(sched, 0);
  ASSERT_NOT_NULL(first_body);
  int n_first = 0;
  PolyUOp **first_topo = poly_toposort(ctx, first_body, &n_first);
  ASSERT_NOT_NULL(first_topo);
  ASSERT_INT_EQ(n_first, 29);
  ASSERT_INT_EQ(count_lin_ops(first_topo, n_first, POLY_OP_SINK), 1);
  ASSERT_INT_EQ(count_lin_ops(first_topo, n_first, POLY_OP_END), 1);
  ASSERT_INT_EQ(count_lin_ops(first_topo, n_first, POLY_OP_STORE), 1);
  ASSERT_INT_EQ(count_lin_ops(first_topo, n_first, POLY_OP_REDUCE), 1);
  ASSERT_INT_EQ(count_lin_ops(first_topo, n_first, POLY_OP_RANGE), 3);
  ASSERT_INT_EQ(count_lin_ops(first_topo, n_first, POLY_OP_PARAM), 4);
  ASSERT_INT_EQ(count_lin_ops(first_topo, n_first, POLY_OP_INDEX), 4);
  ASSERT_INT_EQ(count_lin_ops(first_topo, n_first, POLY_OP_ADD), 4);
  ASSERT_INT_EQ(count_lin_ops(first_topo, n_first, POLY_OP_MUL), 4);
  ASSERT_INT_EQ(count_lin_ops(first_topo, n_first, POLY_OP_CONST), 6);
  ASSERT_INT_EQ(first_body->op, POLY_OP_SINK);
  ASSERT_INT_EQ(first_body->n_src, 1);
  PolyUOp *first_end = first_body->src[0];
  ASSERT_NOT_NULL(first_end);
  ASSERT_INT_EQ(first_end->op, POLY_OP_END);
  ASSERT_INT_EQ(first_end->n_src, 3);
  ASSERT_INT_EQ(first_end->src[0]->op, POLY_OP_STORE);
  ASSERT_INT_EQ(first_end->src[1]->op, POLY_OP_RANGE);
  ASSERT_INT_EQ(first_end->src[2]->op, POLY_OP_RANGE);
  ASSERT_INT_EQ(first_end->src[0]->n_src, 2);
  ASSERT_INT_EQ(first_end->src[0]->src[0]->op, POLY_OP_INDEX);
  ASSERT_INT_EQ(first_end->src[0]->src[1]->op, POLY_OP_ADD);

  int n_rendered = 0;
  for (int i = 0; i < sched->n_src; i++) {
    if (poly_test_linear_call_is_copy(sched, i)) continue;
    int n_body = 0;
    PolyUOp **body_topo = poly_toposort(ctx, poly_test_linear_call_body(sched, i), &n_body);
    ASSERT_NOT_NULL(body_topo);
    ASSERT_INT_EQ(count_lin_ops(body_topo, n_body, POLY_OP_BUFFER), 0);
    int n_lin = 0;
    PolyUOp **lin = poly_linearize_cuda(ctx, poly_test_linear_call_body(sched, i), &n_lin);
    ASSERT_NOT_NULL(lin);
    char name[64];
    snprintf(name, sizeof(name), "large_mlp_train_%d", i);
    char *src = poly_render_cuda(ctx, lin, n_lin, name, 256);
    ASSERT_NOT_NULL(src);
    const char *bad_token = cuda_source_illegal_wide_f32_vector(src);
    if (bad_token) {
      for (int j = 0; j < n_lin; j++) {
        if (lin[j] && poly_uop_max_numel(ctx, lin[j]) >= 16) {
          fprintf(
              stderr, "    lin[%d] op=%s dtype=%s lanes=%lld nsrc=%d\n", j,
              poly_op_name(lin[j]->op), poly_dtype_name(lin[j]->dtype),
              (long long)poly_uop_max_numel(ctx, lin[j]), lin[j]->n_src
          );
          for (int k = 0; k < lin[j]->n_src && k < 4; k++) {
            fprintf(
                stderr, "      src[%d] op=%s dtype=%s lanes=%lld\n", k,
                poly_op_name(lin[j]->src[k]->op), poly_dtype_name(lin[j]->src[k]->dtype),
                (long long)poly_uop_max_numel(ctx, lin[j]->src[k])
            );
          }
          for (int p = 0; p < n_lin; p++) {
            if (!lin[p]) continue;
            for (int s = 0; s < lin[p]->n_src; s++) {
              if (lin[p]->src[s] == lin[j]) {
                fprintf(
                    stderr, "      parent lin[%d] op=%s dtype=%s lanes=%lld src_slot=%d\n", p,
                    poly_op_name(lin[p]->op), poly_dtype_name(lin[p]->dtype),
                    (long long)poly_uop_max_numel(ctx, lin[p]), s
                );
              }
            }
          }
        }
      }
      const char *where = strstr(src, bad_token);
      if (where) {
        const char *start = where;
        while (start > src && (where - start) < 180)
          start--;
        fprintf(
            stderr, "    CUDA bad token item=%d token=%s context:\n%.*s\n", i, bad_token, 360, start
        );
      }
      free(lin);
      free(src);
      poly_model_free(inst);
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
  ASSERT_INT_EQ(n_rendered, 12);

  poly_model_free(inst);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(cuda, instance_set_device_cuda) {
  SKIP_IF_NO_CUDA();

  PolyModel *inst = make_test_mlp(2, 3);
  ASSERT_NOT_NULL(inst);

  /* Read initial host data */
  int64_t numel;
  float *cpu_data = poly_model_buf_data(inst, 0, &numel);
  ASSERT_NOT_NULL(cpu_data);
  float saved = cpu_data[0];

  ASSERT_INT_EQ(poly_model_set_device(inst, POLY_DEVICE_CUDA), 0);

  /* buf_data auto-readbacks from GPU (tinygrad-style) */
  float *gpu_data = poly_model_buf_data(inst, 0, &numel);
  ASSERT_NOT_NULL(gpu_data);
  ASSERT_FLOAT_EQ(gpu_data[0], saved, 1e-6);

  /* Switch back to CPU */
  ASSERT_INT_EQ(poly_model_set_device(inst, POLY_DEVICE_CPU), 0);

  /* buf_data still works on CPU */
  ASSERT_NOT_NULL(poly_model_buf_data(inst, 0, &numel));

  poly_model_free(inst);
  PASS();
}

TEST_BACKEND(cuda, instance_cuda_forward_parity) {
  SKIP_IF_NO_CUDA();

  int n_in = 2, n_out = 3;
  PolyModel *inst = make_test_mlp(n_in, n_out);
  ASSERT_NOT_NULL(inst);

  /* Seed weights deterministically */
  for (int p = 0; p < poly_model_param_count(inst); p++) {
    int64_t numel;
    float *data = poly_model_param_data(inst, p, &numel);
    for (int64_t j = 0; j < numel; j++)
      data[j] = (float)(j % 7 - 3) * 0.1f;
  }

  /* Forward on CPU */
  float input[] = {1.0f, 2.0f};
  float out_cpu[3] = {0};
  PolyIOBinding io[] = {POLY_IO_BINDING_ARRAY("x", input, POLY_FLOAT32)};
  ASSERT_INT_EQ(poly_model_forward(inst, io, 1), 0);

  /* Read CPU output */
  int out_idx = -1;
  for (int i = 0; i < poly_model_buf_count(inst); i++)
    if (poly_model_buf_role(inst, i) == POLY_ROLE_OUTPUT) {
      out_idx = i;
      break;
    }
  ASSERT_TRUE(out_idx >= 0);
  {
    int64_t numel;
    float *cpu_out = poly_model_buf_data(inst, out_idx, &numel);
    ASSERT_NOT_NULL(cpu_out);
    memcpy(out_cpu, cpu_out, n_out * sizeof(float));
  }

  /* Switch to CUDA */
  ASSERT_INT_EQ(poly_model_set_device(inst, POLY_DEVICE_CUDA), 0);

  /* Forward on CUDA */
  ASSERT_INT_EQ(poly_model_forward(inst, io, 1), 0);

  /* Readback output */
  float out_gpu[3] = {0};
  poly_model_readback_buf(inst, out_idx, out_gpu, n_out * sizeof(float));

  /* Compare */
  for (int i = 0; i < n_out; i++)
    ASSERT_FLOAT_EQ(out_gpu[i], out_cpu[i], 1e-4);

  poly_model_free(inst);
  PASS();
}

TEST_BACKEND(cuda, instance_cuda_roundtrip) {
  SKIP_IF_NO_CUDA();

  PolyModel *inst = make_test_mlp(2, 1);
  ASSERT_NOT_NULL(inst);

  /* Seed weights */
  for (int p = 0; p < poly_model_param_count(inst); p++) {
    int64_t numel;
    float *data = poly_model_param_data(inst, p, &numel);
    for (int64_t j = 0; j < numel; j++)
      data[j] = (float)(j % 5 - 2) * 0.2f;
  }

  float input[] = {1.0f, -1.0f};
  PolyIOBinding io[] = {POLY_IO_BINDING_ARRAY("x", input, POLY_FLOAT32)};
  float results[4];

  /* CPU -> forward */
  ASSERT_INT_EQ(poly_model_forward(inst, io, 1), 0);
  int out_idx = -1;
  for (int i = 0; i < poly_model_buf_count(inst); i++)
    if (poly_model_buf_role(inst, i) == POLY_ROLE_OUTPUT) {
      out_idx = i;
      break;
    }
  {
    int64_t n;
    results[0] = *poly_model_buf_data(inst, out_idx, &n);
  }

  /* CUDA -> forward */
  ASSERT_INT_EQ(poly_model_set_device(inst, POLY_DEVICE_CUDA), 0);
  ASSERT_INT_EQ(poly_model_forward(inst, io, 1), 0);
  poly_model_readback_buf(inst, out_idx, &results[1], sizeof(float));

  /* CPU -> forward */
  ASSERT_INT_EQ(poly_model_set_device(inst, POLY_DEVICE_CPU), 0);
  ASSERT_INT_EQ(poly_model_forward(inst, io, 1), 0);
  {
    int64_t n;
    results[2] = *poly_model_buf_data(inst, out_idx, &n);
  }

  /* CUDA -> forward */
  ASSERT_INT_EQ(poly_model_set_device(inst, POLY_DEVICE_CUDA), 0);
  ASSERT_INT_EQ(poly_model_forward(inst, io, 1), 0);
  poly_model_readback_buf(inst, out_idx, &results[3], sizeof(float));

  /* All 4 results should match within tolerance */
  for (int i = 1; i < 4; i++)
    ASSERT_FLOAT_EQ(results[i], results[0], 1e-4);

  poly_model_free(inst);
  PASS();
}

TEST_BACKEND(cuda, buffer_residency_keeps_single_host_root) {
  SKIP_IF_NO_CUDA();

  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *buf = poly_test_buffer(ctx, POLY_FLOAT32, 1);
  ASSERT_NOT_NULL(buf);

  PolyBuffer *host = NULL;
  ASSERT_INT_EQ(poly_buffer_alloc_owned_host(ctx, buf, sizeof(float), true, &host), 0);
  ASSERT_NOT_NULL(host);
  ((float *)host->ptr)[0] = 1.0f;
  ASSERT_INT_EQ(poly_buffer_mark_host_written(ctx, buf), 0);

  ASSERT_INT_EQ(poly_buffer_ensure_device_current(ctx, buf, POLY_DEVICE_CUDA), 0);
  PolyBuffer *cur = poly_buffer_get(ctx, buf);
  ASSERT_NOT_NULL(cur);
  ASSERT_NOT_NULL(cur->src);
  ASSERT_TRUE(cur->src->src == NULL);

  ASSERT_INT_EQ(poly_buffer_ensure_host_current(ctx, buf, &host), 0);
  ASSERT_NOT_NULL(host);
  ((float *)host->ptr)[0] = 2.0f;
  ASSERT_INT_EQ(poly_buffer_mark_host_written(ctx, buf), 0);

  ASSERT_INT_EQ(poly_buffer_ensure_device_current(ctx, buf, POLY_DEVICE_CPU), 0);
  cur = poly_buffer_get(ctx, buf);
  ASSERT_NOT_NULL(cur);
  ASSERT_NOT_NULL(cur->src);
  ASSERT_TRUE(poly_device_is_host_addressable(cur->device));
  ASSERT_TRUE(cur->src->src == NULL);
  PolyBuffer *root = cur->src;

  ASSERT_INT_EQ(poly_buffer_ensure_device_allocated(ctx, buf, POLY_DEVICE_CUDA), 0);
  cur = poly_buffer_get(ctx, buf);
  ASSERT_NOT_NULL(cur);
  ASSERT_TRUE(poly_devices_share_storage(cur->device, POLY_DEVICE_CUDA));
  ASSERT_EQ(cur->src, root);
  ASSERT_TRUE(cur->src->src == NULL);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(cuda, buffer_allocate_preserves_valid_cuda_only_residency) {
  SKIP_IF_NO_CUDA();

  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *buf = poly_test_buffer(ctx, POLY_FLOAT32, 2);
  ASSERT_NOT_NULL(buf);

  float src[] = {7.0f, 9.0f};
  unsigned long long dptr = poly_cuda_alloc(sizeof(src));
  ASSERT_TRUE(dptr != 0);
  ASSERT_INT_EQ(poly_cuda_copy_htod(dptr, src, sizeof(src)), 0);

  PolyBuffer dev = {
      .ptr = (void *)(uintptr_t)dptr,
      .nbytes = sizeof(src),
      .device = POLY_DEVICE_CUDA,
      .owned = true,
      .allocator = NULL,
      .src = NULL,
      .valid = true,
  };
  poly_buffer_adopt(ctx, buf, &dev);

  ASSERT_INT_EQ(poly_buffer_allocate(ctx, buf, POLY_DEVICE_CPU), 0);
  float got[2] = {0.0f, 0.0f};
  ASSERT_INT_EQ(poly_buffer_read(ctx, buf, got, sizeof(got)), 0);
  ASSERT_FLOAT_EQ(got[0], 7.0f, 1e-6f);
  ASSERT_FLOAT_EQ(got[1], 9.0f, 1e-6f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(cuda, instance_host_write_after_set_device_reacquire_updates_cuda_input) {
  SKIP_IF_NO_CUDA();

  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  int64_t shape[1] = {4};
  int f32 = poly_dtype_id_by_name("float32");
  PolyTensor *x = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(x);
  PolyTensor *one = poly_tensor_const_float_by_id(ctx, 1.0, f32, POLY_DEVICE_CPU);
  PolyTensor *out = poly_tensor_alu2(ctx, POLY_OP_ADD, x, one);
  ASSERT_NOT_NULL(one);
  ASSERT_NOT_NULL(out);

  const char *binding_names[] = {"x", "output"};
  int binding_roles[] = {POLY_ROLE_INPUT, POLY_ROLE_OUTPUT};
  PolyTensor *binding_tensors[] = {x, out};
  uint32_t binding_flags[] = {0, 0};

  const char *entry_names[] = {"forward"};
  const char *entry_inputs[] = {"x"};
  int entry_input_counts[] = {1};
  const char *entry_outputs[] = {"output"};
  int entry_output_counts[] = {1};
  const char *entry_objectives[] = {NULL};
  uint32_t entry_flags[] = {0};

  PolyModelError err = {0};
  PolyModel *inst = poly_model_from_binding_arrays(
      ctx, binding_names, binding_roles, binding_tensors, binding_flags, 2, entry_names,
      entry_inputs, entry_input_counts, entry_outputs, entry_output_counts, entry_objectives,
      entry_flags, 1, NULL, &err
  );
  ASSERT_NOT_NULL(inst);
  ASSERT_INT_EQ(poly_model_set_device(inst, POLY_DEVICE_CUDA), 0);

  int out_idx = -1;
  for (int i = 0; i < poly_model_buf_count(inst); i++)
    if (strcmp(poly_model_buf_name(inst, i), "output") == 0) out_idx = i;
  ASSERT_TRUE(out_idx >= 0);

  float input_a[] = {1.0f, 2.0f, 3.0f, 4.0f};
  PolyIOBinding input_a_io[] = {
      POLY_IO_BINDING_ARRAY("x", input_a, POLY_FLOAT32),
  };
  ASSERT_INT_EQ(poly_model_forward(inst, input_a_io, 1), 0);
  float got[4] = {0};
  ASSERT_INT_EQ(poly_model_readback_buf(inst, out_idx, got, sizeof(got)), 0);
  for (int i = 0; i < 4; i++)
    ASSERT_FLOAT_EQ(got[i], input_a[i] + 1.0f, 1e-5f);

  float input_b[] = {-5.0f, 0.25f, 7.0f, 11.0f};
  PolyIOBinding input_b_io[] = {
      POLY_IO_BINDING_ARRAY("x", input_b, POLY_FLOAT32),
  };
  ASSERT_INT_EQ(poly_model_forward(inst, input_b_io, 1), 0);
  memset(got, 0, sizeof(got));
  ASSERT_INT_EQ(poly_model_readback_buf(inst, out_idx, got, sizeof(got)), 0);
  for (int i = 0; i < 4; i++)
    ASSERT_FLOAT_EQ(got[i], input_b[i] + 1.0f, 1e-5f);

  poly_model_free(inst);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(cuda, instance_cuda_large_mlp_train_no_wide_vector_types) {
  SKIP_IF_NO_CUDA();

  const char *spec = "{\"layers\":[128,256,64],\"activation\":\"relu\",\"bias\":true,"
                     "\"loss\":\"mse\",\"batch_size\":32,\"seed\":42}";
  PolyModel *inst = poly_mlp_from_json(spec, (int)strlen(spec), POLY_DEVICE_AUTO);
  ASSERT_NOT_NULL(inst);
  ASSERT_INT_EQ(poly_model_set_device(inst, POLY_DEVICE_CUDA), 0);
  ASSERT_INT_EQ(poly_model_set_optimizer(inst, POLY_OPTIM_SGD, 0.01f, 0.0f, 0.0f, 0.0f, 0.0f), 0);

  float x[32 * 128];
  float y[32 * 64];
  for (int i = 0; i < (int)(sizeof(x) / sizeof(x[0])); i++)
    x[i] = (float)((i % 17) - 8) * 0.01f;
  for (int i = 0; i < (int)(sizeof(y) / sizeof(y[0])); i++)
    y[i] = (float)((i % 13) - 6) * 0.01f;

  PolyIOBinding io[] = {
      POLY_IO_BINDING_ARRAY("x", x, POLY_FLOAT32),
      POLY_IO_BINDING_ARRAY("y", y, POLY_FLOAT32),
  };
  float loss = 0.0f;
  ASSERT_INT_EQ(poly_model_train_step(inst, NULL, io, 2, &loss), 0);
  ASSERT_TRUE(isfinite(loss));

  poly_model_free(inst);
  PASS();
}

TEST_BACKEND(cuda, instance_cuda_adam_train_lazily_created_state_stays_on_cuda) {
  SKIP_IF_NO_CUDA();

  const char *spec = "{\"layers\":[2,4,1],\"activation\":\"relu\",\"bias\":true,"
                     "\"loss\":\"mse\",\"batch_size\":1,\"seed\":42}";
  PolyModel *inst = poly_mlp_from_json(spec, (int)strlen(spec), POLY_DEVICE_AUTO);
  ASSERT_NOT_NULL(inst);

  ASSERT_INT_EQ(poly_model_set_device(inst, POLY_DEVICE_CUDA), 0);
  ASSERT_INT_EQ(
      poly_model_set_optimizer(inst, POLY_OPTIM_ADAM, 0.01f, 0.9f, 0.999f, 1e-8f, 0.0f), 0
  );

  float x[] = {1.0f, 2.0f};
  float y[] = {3.0f};
  PolyIOBinding io[] = {
      POLY_IO_BINDING_ARRAY("x", x, POLY_FLOAT32),
      POLY_IO_BINDING_ARRAY("y", y, POLY_FLOAT32),
  };

  float loss = 0.0f;
  ASSERT_INT_EQ(poly_model_train_step(inst, NULL, io, 2, &loss), 0);
  ASSERT_TRUE(isfinite(loss));

  poly_model_free(inst);
  PASS();
}

TEST_BACKEND(cuda, jit_graph_batches_programs_and_replays_inputs) {
  SKIP_IF_NO_CUDA();
  ASSERT_TRUE(poly_cuda_graph_available());

  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  int64_t dims[1] = {4};
  float capture_data[4] = {10.0f, 20.0f, 30.0f, 40.0f};
  float replay_data[4] = {100.0f, 200.0f, 300.0f, 400.0f};
  float ones_data[4] = {1.0f, 1.0f, 1.0f, 1.0f};

  PolyTensor *capture_host =
      poly_tensor_from_host(ctx, capture_data, sizeof(capture_data), POLY_FLOAT32, dims, 1);
  PolyTensor *ones_host =
      poly_tensor_from_host(ctx, ones_data, sizeof(ones_data), POLY_FLOAT32, dims, 1);
  PolyTensor *capture = poly_tensor_to_device(ctx, capture_host, POLY_DEVICE_CUDA);
  PolyTensor *ones = poly_tensor_to_device(ctx, ones_host, POLY_DEVICE_CUDA);
  ASSERT_NOT_NULL(capture);
  ASSERT_NOT_NULL(ones);
  PolyTensor *realized = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &capture, 1, &realized), 0);
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &ones, 1, &realized), 0);

  PolyJit *jit = poly_jit_new(ctx);
  ASSERT_NOT_NULL(jit);
  ASSERT_INT_EQ(poly_jit_begin_capture(jit, &capture, 1), 0);
  PolyTensor *first = poly_tensor_alu2(ctx, POLY_OP_ADD, capture, ones);
  ASSERT_NOT_NULL(first);
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &first, 1, &realized), 0);
  PolyTensor *out = poly_tensor_alu2(ctx, POLY_OP_MUL, first, ones);
  ASSERT_NOT_NULL(out);
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &out, 1, &realized), 0);

  poly_ctx_reset_counters(ctx);
  ASSERT_INT_EQ(poly_jit_end_capture(jit, ctx->tensors, ctx->n_tensors), 0);
  PolyUOp *linear = poly_jit_captured_linear(jit);
  ASSERT_NOT_NULL(linear);
  ASSERT_EQ(linear->op, POLY_OP_LINEAR);
  ASSERT_INT_EQ(linear->n_src, 1);
  PolyUOp *graph_call = linear->src[0];
  ASSERT_NOT_NULL(graph_call);
  ASSERT_EQ(graph_call->op, POLY_OP_CALL);
  ASSERT_INT_EQ(graph_call->n_src, 2);
  ASSERT_EQ(graph_call->src[1]->op, POLY_OP_PARAM);
  PolyUOp *graph_fn = graph_call->src[0];
  ASSERT_EQ(graph_fn->op, POLY_OP_CUSTOM_FUNCTION);
  ASSERT_TRUE(
      graph_fn->arg.kind == POLY_ARG_STRING && graph_fn->arg.str &&
      strcmp(graph_fn->arg.str, "graph") == 0
  );
  ASSERT_INT_EQ(graph_fn->n_src, 1);
  PolyUOp *nested = graph_fn->src[0];
  ASSERT_EQ(nested->op, POLY_OP_LINEAR);
  ASSERT_INT_EQ(nested->n_src, 2);
  ASSERT_EQ(nested->src[0]->src[0]->op, POLY_OP_PROGRAM);
  ASSERT_EQ(nested->src[1]->src[0]->op, POLY_OP_PROGRAM);

  PolyCtxStats capture_stats = {0};
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &capture_stats), 0);
  ASSERT_INT_EQ(capture_stats.kernel_count, 1);
  float got[4] = {0};
  PolyUOp *out_buffer = (PolyUOp *)poly_uop_get_buffer_identity(poly_tensor_uop(out));
  ASSERT_NOT_NULL(out_buffer);
  ASSERT_INT_EQ(poly_buffer_read(ctx, out_buffer, got, sizeof(got)), 0);
  for (int i = 0; i < 4; i++)
    ASSERT_FLOAT_EQ(got[i], capture_data[i] + 1.0f, 1e-5f);

  PolyTensor *replay_host =
      poly_tensor_from_host(ctx, replay_data, sizeof(replay_data), POLY_FLOAT32, dims, 1);
  PolyTensor *replay = poly_tensor_to_device(ctx, replay_host, POLY_DEVICE_CUDA);
  ASSERT_NOT_NULL(replay);
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &replay, 1, &realized), 0);
  poly_ctx_reset_counters(ctx);
  ASSERT_INT_EQ(poly_jit_run(jit, &replay, 1), 0);
  PolyCtxStats replay_stats = {0};
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &replay_stats), 0);
  ASSERT_INT_EQ(replay_stats.kernel_count, 1);
  memset(got, 0, sizeof(got));
  ASSERT_INT_EQ(poly_buffer_read(ctx, out_buffer, got, sizeof(got)), 0);
  for (int i = 0; i < 4; i++)
    ASSERT_FLOAT_EQ(got[i], replay_data[i] + 1.0f, 1e-5f);

  poly_jit_free(jit);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(cuda, jit_graph_batches_memory_planned_reductions) {
  SKIP_IF_NO_CUDA();
  ASSERT_TRUE(poly_cuda_graph_available());

  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  int64_t dims[2] = {2, 4};
  int64_t axes[1] = {1};
  float capture_data[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  float replay_data[8] = {10, 20, 30, 40, 50, 60, 70, 80};
  float ones_data[8] = {1, 1, 1, 1, 1, 1, 1, 1};

  PolyTensor *capture_host =
      poly_tensor_from_host(ctx, capture_data, sizeof(capture_data), POLY_FLOAT32, dims, 2);
  PolyTensor *replay_host =
      poly_tensor_from_host(ctx, replay_data, sizeof(replay_data), POLY_FLOAT32, dims, 2);
  PolyTensor *ones_host =
      poly_tensor_from_host(ctx, ones_data, sizeof(ones_data), POLY_FLOAT32, dims, 2);
  PolyTensor *capture = poly_tensor_to_device(ctx, capture_host, POLY_DEVICE_CUDA);
  PolyTensor *replay = poly_tensor_to_device(ctx, replay_host, POLY_DEVICE_CUDA);
  PolyTensor *ones = poly_tensor_to_device(ctx, ones_host, POLY_DEVICE_CUDA);
  ASSERT_NOT_NULL(capture);
  ASSERT_NOT_NULL(replay);
  ASSERT_NOT_NULL(ones);
  PolyTensor *realized = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &capture, 1, &realized), 0);
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &replay, 1, &realized), 0);
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &ones, 1, &realized), 0);

  PolyJit *jit = poly_jit_new(ctx);
  ASSERT_NOT_NULL(jit);
  ASSERT_INT_EQ(poly_jit_begin_capture(jit, &capture, 1), 0);
  PolyTensor *r0 = poly_tensor_sum(ctx, capture, axes, 1, false);
  PolyTensor *x1 = poly_tensor_alu2(ctx, POLY_OP_ADD, capture, ones);
  PolyTensor *r1 = poly_tensor_sum(ctx, x1, axes, 1, false);
  PolyTensor *x2 = poly_tensor_alu2(ctx, POLY_OP_ADD, x1, ones);
  PolyTensor *r2 = poly_tensor_sum(ctx, x2, axes, 1, false);
  ASSERT_NOT_NULL(r0);
  ASSERT_NOT_NULL(r1);
  ASSERT_NOT_NULL(r2);
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &r0, 1, &realized), 0);
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &r1, 1, &realized), 0);
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &r2, 1, &realized), 0);
  PolyTensor *r01 = poly_tensor_alu2(ctx, POLY_OP_ADD, r0, r1);
  PolyTensor *out = poly_tensor_alu2(ctx, POLY_OP_ADD, r01, r2);
  ASSERT_NOT_NULL(out);
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &out, 1, &realized), 0);

  poly_ctx_reset_counters(ctx);
  ASSERT_INT_EQ(poly_jit_end_capture(jit, ctx->tensors, ctx->n_tensors), 0);
  PolyUOp *linear = poly_jit_captured_linear(jit);
  ASSERT_NOT_NULL(linear);
  ASSERT_EQ(linear->op, POLY_OP_LINEAR);
  ASSERT_INT_EQ(linear->n_src, 1);
  PolyUOp *graph_call = linear->src[0];
  ASSERT_EQ(graph_call->op, POLY_OP_CALL);
  ASSERT_EQ(graph_call->src[0]->op, POLY_OP_CUSTOM_FUNCTION);
  ASSERT_TRUE(
      graph_call->src[0]->arg.kind == POLY_ARG_STRING && graph_call->src[0]->arg.str &&
      strcmp(graph_call->src[0]->arg.str, "graph") == 0
  );
  PolyUOp *nested = graph_call->src[0]->src[0];
  ASSERT_EQ(nested->op, POLY_OP_LINEAR);
  ASSERT_INT_EQ(nested->n_src, 4);
  for (int i = 0; i < nested->n_src; i++)
    ASSERT_EQ(nested->src[i]->src[0]->op, POLY_OP_PROGRAM);

  PolyCtxStats capture_stats = {0};
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &capture_stats), 0);
  ASSERT_INT_EQ(capture_stats.kernel_count, 1);
  float got[2] = {0};
  PolyUOp *out_buffer = (PolyUOp *)poly_uop_get_buffer_identity(poly_tensor_uop(out));
  ASSERT_NOT_NULL(out_buffer);
  ASSERT_INT_EQ(poly_buffer_read(ctx, out_buffer, got, sizeof(got)), 0);
  ASSERT_FLOAT_EQ(got[0], 42.0f, 1e-5f);
  ASSERT_FLOAT_EQ(got[1], 90.0f, 1e-5f);

  poly_ctx_reset_counters(ctx);
  ASSERT_INT_EQ(poly_jit_run(jit, &replay, 1), 0);
  PolyCtxStats replay_stats = {0};
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &replay_stats), 0);
  ASSERT_INT_EQ(replay_stats.kernel_count, 1);
  memset(got, 0, sizeof(got));
  ASSERT_INT_EQ(poly_buffer_read(ctx, out_buffer, got, sizeof(got)), 0);
  ASSERT_FLOAT_EQ(got[0], 312.0f, 1e-5f);
  ASSERT_FLOAT_EQ(got[1], 792.0f, 1e-5f);

  poly_jit_free(jit);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(cuda, compiled_schedule_preserves_mixed_call_devices) {
  SKIP_IF_NO_CUDA();
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  poly_ctx_set_preferred_device(ctx, POLY_DEVICE_CUDA);

  int64_t shape[] = {2};
  PolyTensor *x = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_CUDA);
  PolyTensor *w0 = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_CUDA);
  PolyTensor *w1 = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_CUDA);
  PolyTensor *hidden = poly_tensor_alu2(ctx, POLY_OP_ADD, x, w0);
  PolyTensor *output = poly_tensor_alu2(ctx, POLY_OP_MUL, hidden, w1);
  ASSERT_NOT_NULL(x);
  ASSERT_NOT_NULL(w0);
  ASSERT_NOT_NULL(w1);
  ASSERT_NOT_NULL(hidden);
  ASSERT_NOT_NULL(output);

  float w0_data[] = {3.0f, 4.0f};
  float w1_data[] = {2.0f, 3.0f};
  PolyUOp *w0_buffer = (PolyUOp *)poly_uop_get_buffer_identity(poly_tensor_uop_physical(w0));
  PolyUOp *w1_buffer = (PolyUOp *)poly_uop_get_buffer_identity(poly_tensor_uop_physical(w1));
  ASSERT_NOT_NULL(w0_buffer);
  ASSERT_NOT_NULL(w1_buffer);
  ASSERT_INT_EQ(poly_buffer_write(ctx, w0_buffer, w0_data, sizeof(w0_data)), 0);
  ASSERT_INT_EQ(poly_buffer_write(ctx, w1_buffer, w1_data, sizeof(w1_data)), 0);

  PolyBindingSpec bindings[] = {
      {.name = "x", .role = POLY_ROLE_INPUT, .tensor = x},
      {.name = "layers.0.weight", .role = POLY_ROLE_PARAM, .tensor = w0},
      {.name = "layers.1.weight", .role = POLY_ROLE_PARAM, .tensor = w1},
      {.name = "output", .role = POLY_ROLE_OUTPUT, .tensor = output},
  };
  const char *input_names[] = {"x"};
  const char *output_names[] = {"output"};
  PolyEntrypointSpec entrypoints[] = {{
      .name = "forward",
      .inputs = input_names,
      .n_inputs = 1,
      .outputs = output_names,
      .n_outputs = 1,
  }};
  PolyModel *inst = poly_model_from_bindings(ctx, bindings, 4, entrypoints, 1, NULL, NULL);
  ASSERT_NOT_NULL(inst);

  PolyTensor *module0_inputs[] = {x};
  PolyTensor *module1_inputs[] = {hidden};
  PolyModelModuleSpec modules[] = {
      {.name = "layers.0", .inputs = module0_inputs, .n_inputs = 1, .output = hidden},
      {.name = "layers.1", .inputs = module1_inputs, .n_inputs = 1, .output = output},
  };
  ASSERT_INT_EQ(poly_model_define_modules(inst, modules, 2), 0);
  PolyModelDeviceMapEntry map[] = {
      {.module = "layers.0", .device = "CUDA"},
      {.module = "layers.1", .device = "INTERP"},
  };
  ASSERT_INT_EQ(poly_model_set_device_map(inst, map, 2), 0);

  float x_data[] = {1.0f, 2.0f};
  ASSERT_INT_EQ(
      poly_buffer_write(ctx, poly_model_get_buffer(inst, "x"), x_data, sizeof(x_data)), 0
  );
  PolyVarBinding *vars = NULL;
  int n_vars = 0;
  PolyUOp *schedule =
      poly_linear_effect_sink(ctx, poly_model_get_sink(inst, "forward"), &vars, &n_vars);
  ASSERT_NOT_NULL(schedule);
  ASSERT_INT_EQ(schedule->n_src, 3);
  ASSERT_EQ(poly_test_linear_call_body(schedule, 0)->op, POLY_OP_SINK);
  ASSERT_EQ(poly_test_linear_call_body(schedule, 1)->op, POLY_OP_COPY);
  ASSERT_EQ(poly_test_linear_call_body(schedule, 2)->op, POLY_OP_SINK);

  PolyUOp *compiled = poly_compile_linear(ctx, schedule, -1);
  ASSERT_NOT_NULL(compiled);
  ASSERT_EQ(compiled->src[0]->src[0]->op, POLY_OP_PROGRAM);
  ASSERT_EQ(compiled->src[1]->src[0]->op, POLY_OP_COPY);
  ASSERT_EQ(compiled->src[2]->src[0]->op, POLY_OP_PROGRAM);
  const PolyProgramInfo *first_info = poly_program_info(ctx, compiled->src[0]->src[0]);
  const PolyProgramInfo *third_info = poly_program_info(ctx, compiled->src[2]->src[0]);
  ASSERT_NOT_NULL(first_info);
  ASSERT_NOT_NULL(third_info);
  ASSERT_INT_EQ(poly_device_by_name(first_info->target), POLY_DEVICE_CUDA);
  ASSERT_INT_EQ(poly_device_by_name(third_info->target), POLY_DEVICE_INTERP);

  ASSERT_INT_EQ(poly_run_linear(ctx, compiled, vars, n_vars, NULL, 0, true, true, false), 0);
  float got[2] = {0.0f, 0.0f};
  ASSERT_INT_EQ(poly_buffer_read(ctx, poly_model_get_buffer(inst, "output"), got, sizeof(got)), 0);
  ASSERT_FLOAT_EQ(got[0], 8.0f, 1e-6f);
  ASSERT_FLOAT_EQ(got[1], 18.0f, 1e-6f);

  free(vars);

  PolyModelDeviceMapEntry reverse_map[] = {
      {.module = "layers.0", .device = "INTERP"},
      {.module = "layers.1", .device = "CUDA"},
  };
  ASSERT_INT_EQ(poly_model_set_device_map(inst, reverse_map, 2), 0);
  ASSERT_INT_EQ(
      poly_buffer_write(ctx, poly_model_get_buffer(inst, "x"), x_data, sizeof(x_data)), 0
  );
  vars = NULL;
  n_vars = 0;
  schedule = poly_linear_effect_sink(ctx, poly_model_get_sink(inst, "forward"), &vars, &n_vars);
  ASSERT_NOT_NULL(schedule);
  ASSERT_INT_EQ(schedule->n_src, 3);
  compiled = poly_compile_linear(ctx, schedule, -1);
  ASSERT_NOT_NULL(compiled);
  first_info = poly_program_info(ctx, compiled->src[0]->src[0]);
  third_info = poly_program_info(ctx, compiled->src[2]->src[0]);
  ASSERT_NOT_NULL(first_info);
  ASSERT_NOT_NULL(third_info);
  ASSERT_INT_EQ(poly_device_by_name(first_info->target), POLY_DEVICE_INTERP);
  ASSERT_INT_EQ(
      poly_device_from_device_uop(poly_uop_device_uop_cached(ctx, compiled->src[1]->src[0], NULL)),
      POLY_DEVICE_CUDA
  );
  ASSERT_INT_EQ(poly_device_by_name(third_info->target), POLY_DEVICE_CUDA);
  ASSERT_INT_EQ(poly_run_linear(ctx, compiled, vars, n_vars, NULL, 0, true, true, false), 0);
  memset(got, 0, sizeof(got));
  ASSERT_INT_EQ(poly_buffer_read(ctx, poly_model_get_buffer(inst, "output"), got, sizeof(got)), 0);
  ASSERT_FLOAT_EQ(got[0], 8.0f, 1e-6f);
  ASSERT_FLOAT_EQ(got[1], 18.0f, 1e-6f);

  free(vars);
  poly_model_free(inst);
  poly_ctx_destroy(ctx);
  PASS();
}

#endif /* POLY_HAS_CUDA */
