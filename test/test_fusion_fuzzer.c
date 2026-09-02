/*
 * test_fusion_fuzzer.c -- deterministic differential fusion fuzzer
 *
 * These cases are intentionally small.  A failure should identify the
 * backend, case name, output lane, and a graph family that is already close
 * to minimized; set POLY_DUMP_KERNELS=1 for kernel dumps.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "test_harness.h"
#include "../src/polygrad.h"
#include "../src/frontend.h"
#include "../src/tensor.h"

typedef enum {
  FUSION_CASE_MOVEMENT_REDUCE = 0,
  FUSION_CASE_WHERE_REDUCE,
  FUSION_CASE_NO_BARRIER_REDUCE,
  FUSION_CASE_EXPLICIT_BARRIER_REDUCE,
  FUSION_CASE_QR_RECONSTRUCT,
  FUSION_CASE_QR_BARRIER_RECONSTRUCT,
  FUSION_CASE_CHOLESKY_SOLVE,
  FUSION_CASE_CHOLESKY_BARRIER_SOLVE,
  FUSION_CASE_SOLVE,
  FUSION_CASE_LSTSQ_TALL,
  FUSION_CASE_COUNT,
} FusionCase;

typedef struct {
  const char *name;
  int out_numel;
  float tol;
} FusionCaseInfo;

static const FusionCaseInfo FUSION_CASES[FUSION_CASE_COUNT] = {
    [FUSION_CASE_MOVEMENT_REDUCE] = {"movement_reduce", 3, 2e-4f},
    [FUSION_CASE_WHERE_REDUCE] = {"where_reduce", 4, 2e-4f},
    [FUSION_CASE_NO_BARRIER_REDUCE] = {"no_barrier_reduce", 3, 2e-4f},
    [FUSION_CASE_EXPLICIT_BARRIER_REDUCE] = {"explicit_barrier_reduce", 3, 2e-4f},
    [FUSION_CASE_QR_RECONSTRUCT] = {"qr_reconstruct", 6, 2e-3f},
    [FUSION_CASE_QR_BARRIER_RECONSTRUCT] = {"qr_barrier_reconstruct", 6, 2e-3f},
    [FUSION_CASE_CHOLESKY_SOLVE] = {"cholesky_solve", 4, 3e-4f},
    [FUSION_CASE_CHOLESKY_BARRIER_SOLVE] = {"cholesky_barrier_solve", 4, 3e-4f},
    [FUSION_CASE_SOLVE] = {"solve", 2, 4e-4f},
    [FUSION_CASE_LSTSQ_TALL] = {"lstsq_tall", 2, 6e-4f},
};

static PolyTensor *fusion_host_f32(
    PolyCtx *ctx,
    float *data,
    int64_t *shape,
    int ndim,
    PolyDevice device
) {
  int dtype = poly_dtype_id_by_name("float32");
  if (dtype < 0) return NULL;
  int64_t numel = 1;
  for (int i = 0; i < ndim; i++)
    numel *= shape[i];
  PolyTensor *host =
      poly_tensor_from_host_by_id(ctx, data, (size_t)numel * sizeof(float), dtype, shape, ndim);
  return host ? poly_tensor_to_device(ctx, host, device) : NULL;
}

static PolyTensor *fusion_realize_value(PolyCtx *ctx, PolyTensor *value) {
  PolyTensor *out = NULL;
  if (!value || poly_realize_tensors(ctx, &value, 1, &out) != 0) return NULL;
  return out;
}

static int fusion_read_tensor(PolyCtx *ctx, PolyTensor *tensor, float *out, int n) {
  if (!ctx || !tensor || !out || n < 0) return -1;
  const PolyUOp *buf = poly_uop_get_buffer_identity(poly_tensor_uop_physical(tensor));
  if (!buf) return -1;
  return poly_buffer_read(ctx, (PolyUOp *)buf, out, (size_t)n * sizeof(float));
}

static PolyTensor *fusion_build_case(PolyCtx *ctx, FusionCase which, PolyDevice device) {
  int f32 = poly_dtype_id_by_name("float32");
  if (f32 < 0) return NULL;
  switch (which) {
  case FUSION_CASE_MOVEMENT_REDUCE: {
    static float x_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    int64_t x_shape[] = {3, 4};
    PolyTensor *x = fusion_host_f32(ctx, x_data, x_shape, 2, device);
    if (!x) return NULL;
    PolyTensor *p = poly_tensor_pad_value_float(ctx, x, (int64_t[][2]){{1, 0}, {0, 1}}, 2, 0.0);
    PolyTensor *s = p ? poly_tensor_shrink(ctx, p, (int64_t[][2]){{1, 4}, {1, 5}}, 2) : NULL;
    PolyTensor *scale =
        poly_tensor_full_float_by_id(ctx, x_shape, 2, 0.25, f32, device, true, false);
    PolyTensor *one = poly_tensor_full_float_by_id(ctx, x_shape, 2, 1.0, f32, device, true, false);
    PolyTensor *scaled = (s && scale) ? poly_tensor_alu2(ctx, POLY_OP_MUL, s, scale) : NULL;
    PolyTensor *y = (scaled && one) ? poly_tensor_alu2(ctx, POLY_OP_ADD, scaled, one) : NULL;
    int64_t axis = 1;
    return y ? poly_tensor_sum(ctx, y, &axis, 1, false) : NULL;
  }

  case FUSION_CASE_WHERE_REDUCE: {
    static float x_data[] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    };
    int64_t shape[] = {4, 4};
    PolyTensor *x = fusion_host_f32(ctx, x_data, shape, 2, device);
    PolyTensor *seven = poly_tensor_full_float_by_id(ctx, shape, 2, 7.0, f32, device, true, false);
    PolyTensor *neg = poly_tensor_full_float_by_id(ctx, shape, 2, -2.0, f32, device, true, false);
    PolyTensor *mask = (x && seven) ? poly_tensor_alu2(ctx, POLY_OP_CMPLT, x, seven) : NULL;
    PolyTensor *sel = (mask && neg) ? poly_tensor_alu3(ctx, POLY_OP_WHERE, mask, x, neg) : NULL;
    int64_t axis = 0;
    return sel ? poly_tensor_sum(ctx, sel, &axis, 1, false) : NULL;
  }

  case FUSION_CASE_NO_BARRIER_REDUCE: {
    static float x_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    int64_t shape[] = {3, 3};
    PolyTensor *x = fusion_host_f32(ctx, x_data, shape, 2, device);
    PolyTensor *one = poly_tensor_full_float_by_id(ctx, shape, 2, 1.0, f32, device, true, false);
    PolyTensor *y = (x && one) ? poly_tensor_alu2(ctx, POLY_OP_ADD, x, one) : NULL;
    PolyTensor *square = y ? poly_tensor_alu2(ctx, POLY_OP_MUL, y, y) : NULL;
    PolyTensor *z = square ? poly_tensor_alu2(ctx, POLY_OP_ADD, square, y) : NULL;
    int64_t axis = 0;
    return z ? poly_tensor_sum(ctx, z, &axis, 1, false) : NULL;
  }

  case FUSION_CASE_EXPLICIT_BARRIER_REDUCE: {
    static float x_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    int64_t shape[] = {3, 3};
    PolyTensor *x = fusion_host_f32(ctx, x_data, shape, 2, device);
    PolyTensor *one = poly_tensor_full_float_by_id(ctx, shape, 2, 1.0, f32, device, true, false);
    PolyTensor *mid = (x && one) ? poly_tensor_alu2(ctx, POLY_OP_ADD, x, one) : NULL;
    PolyTensor *mid_t = mid ? fusion_realize_value(ctx, mid) : NULL;
    PolyTensor *square = mid_t ? poly_tensor_alu2(ctx, POLY_OP_MUL, mid_t, mid_t) : NULL;
    PolyTensor *z = square ? poly_tensor_alu2(ctx, POLY_OP_ADD, square, mid_t) : NULL;
    int64_t axis = 0;
    return z ? poly_tensor_sum(ctx, z, &axis, 1, false) : NULL;
  }

  case FUSION_CASE_QR_RECONSTRUCT: {
    static float a_data[] = {1.0f, 2.0f, -1.0f, 3.0f, 0.5f, 4.0f};
    int64_t shape[] = {3, 2};
    PolyTensor *a = fusion_host_f32(ctx, a_data, shape, 2, device);
    PolyTensor *q = NULL, *r = NULL;
    if (!a || poly_tensor_qr_ex(ctx, a, POLY_QR_REDUCED, &q, &r) != 0 || !q || !r) return NULL;
    return poly_tensor_dot(ctx, q, r);
  }

  case FUSION_CASE_QR_BARRIER_RECONSTRUCT: {
    static float a_data[] = {1.0f, 2.0f, -1.0f, 3.0f, 0.5f, 4.0f};
    int64_t shape[] = {3, 2};
    PolyTensor *a = fusion_host_f32(ctx, a_data, shape, 2, device);
    PolyTensor *q = NULL, *r = NULL;
    if (!a || poly_tensor_qr_ex(ctx, a, POLY_QR_REDUCED, &q, &r) != 0 || !q || !r) return NULL;
    PolyTensor *q_t = fusion_realize_value(ctx, q);
    PolyTensor *r_t = fusion_realize_value(ctx, r);
    return (q_t && r_t) ? poly_tensor_dot(ctx, q_t, r_t) : NULL;
  }

  case FUSION_CASE_CHOLESKY_SOLVE: {
    static float a_data[] = {4, 2, 2, 5};
    static float b_data[] = {1, 2, 3, 4};
    int64_t shape[] = {2, 2};
    PolyTensor *a = fusion_host_f32(ctx, a_data, shape, 2, device);
    PolyTensor *b = fusion_host_f32(ctx, b_data, shape, 2, device);
    PolyTensor *l = a ? poly_tensor_cholesky(ctx, a, 0) : NULL;
    return (l && b) ? poly_tensor_cholesky_solve(ctx, l, b, 0) : NULL;
  }

  case FUSION_CASE_CHOLESKY_BARRIER_SOLVE: {
    static float a_data[] = {4, 2, 2, 5};
    static float b_data[] = {1, 2, 3, 4};
    int64_t shape[] = {2, 2};
    PolyTensor *a = fusion_host_f32(ctx, a_data, shape, 2, device);
    PolyTensor *b = fusion_host_f32(ctx, b_data, shape, 2, device);
    PolyTensor *l = a ? poly_tensor_cholesky(ctx, a, 0) : NULL;
    PolyTensor *l_t = l ? fusion_realize_value(ctx, l) : NULL;
    return (l_t && b) ? poly_tensor_cholesky_solve(ctx, l_t, b, 0) : NULL;
  }

  case FUSION_CASE_SOLVE: {
    static float a_data[] = {0, 2, 1, 3};
    static float b_data[] = {4, 5};
    int64_t a_shape[] = {2, 2};
    int64_t b_shape[] = {2};
    PolyTensor *a = fusion_host_f32(ctx, a_data, a_shape, 2, device);
    PolyTensor *b = fusion_host_f32(ctx, b_data, b_shape, 1, device);
    return (a && b) ? poly_tensor_solve(ctx, a, b) : NULL;
  }

  case FUSION_CASE_LSTSQ_TALL: {
    static float a_data[] = {1, 0, 1, 1, 1, 2};
    static float b_data[] = {1, 2, 3};
    int64_t a_shape[] = {3, 2};
    int64_t b_shape[] = {3};
    PolyTensor *a = fusion_host_f32(ctx, a_data, a_shape, 2, device);
    PolyTensor *b = fusion_host_f32(ctx, b_data, b_shape, 1, device);
    return (a && b) ? poly_tensor_lstsq(ctx, a, b) : NULL;
  }

  default:
    return NULL;
  }
}

static int fusion_run_case(FusionCase which, PolyDevice device, float *out, int out_cap) {
  if (which < 0 || which >= FUSION_CASE_COUNT) return -1;
  const FusionCaseInfo *info = &FUSION_CASES[which];
  if (out_cap < info->out_numel) return -1;

  PolyCtx *ctx = poly_ctx_new();
  if (!ctx) return -1;
  poly_ctx_set_preferred_device(ctx, device);

  PolyTensor *value = fusion_build_case(ctx, which, device);
  PolyTensor *tensor = value ? fusion_realize_value(ctx, value) : NULL;
  int rc = tensor ? fusion_read_tensor(ctx, tensor, out, info->out_numel) : -1;
  poly_ctx_destroy(ctx);
  return rc;
}

static int fusion_check_close(
    const char *backend,
    FusionCase which,
    const float *expected,
    const float *got,
    int *bad_lane,
    float *bad_expected,
    float *bad_got
) {
  const FusionCaseInfo *info = &FUSION_CASES[which];
  for (int i = 0; i < info->out_numel; i++) {
    float a = expected[i], b = got[i];
    if ((isnan(a) && isnan(b)) || fabsf(a - b) <= info->tol) continue;
    (void)backend;
    if (bad_lane) *bad_lane = i;
    if (bad_expected) *bad_expected = a;
    if (bad_got) *bad_got = b;
    return -1;
  }
  return 0;
}

TEST(fusion, generated_graphs_cpu_interp_parity) {
  for (int c = 0; c < FUSION_CASE_COUNT; c++) {
    float cpu[16] = {0};
    float interp[16] = {0};
    ASSERT_INT_EQ(fusion_run_case((FusionCase)c, POLY_DEVICE_CPU, cpu, 16), 0);
    ASSERT_INT_EQ(fusion_run_case((FusionCase)c, POLY_DEVICE_INTERP, interp, 16), 0);
    int lane = -1;
    float exp = 0.0f, got = 0.0f;
    if (fusion_check_close("interp", (FusionCase)c, cpu, interp, &lane, &exp, &got) != 0) {
      const FusionCaseInfo *info = &FUSION_CASES[c];
      FAIL(
          "interp %s lane %d got %.8g expected %.8g (abs %.8g tol %.8g)", info->name, lane,
          (double)got, (double)exp, (double)fabsf(got - exp), (double)info->tol
      );
    }
  }
  PASS();
}

TEST_BACKEND(cuda, generated_fusion_graphs_match_cpu) {
#ifndef POLY_HAS_CUDA
  SKIP("CUDA backend not compiled");
#else
  if (poly_selftest_device(POLY_DEVICE_CUDA) != 0) SKIP("CUDA runtime unavailable");

  for (int c = 0; c < FUSION_CASE_COUNT; c++) {
    float cpu[16] = {0};
    float cuda[16] = {0};
    ASSERT_INT_EQ(fusion_run_case((FusionCase)c, POLY_DEVICE_CPU, cpu, 16), 0);
    ASSERT_INT_EQ(fusion_run_case((FusionCase)c, POLY_DEVICE_CUDA, cuda, 16), 0);
    int lane = -1;
    float exp = 0.0f, got = 0.0f;
    if (fusion_check_close("cuda", (FusionCase)c, cpu, cuda, &lane, &exp, &got) != 0) {
      const FusionCaseInfo *info = &FUSION_CASES[c];
      FAIL(
          "cuda %s lane %d got %.8g expected %.8g (abs %.8g tol %.8g)", info->name, lane,
          (double)got, (double)exp, (double)fabsf(got - exp), (double)info->tol
      );
    }
  }
  PASS();
#endif
}
