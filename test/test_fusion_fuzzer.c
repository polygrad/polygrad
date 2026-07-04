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

static PolyUOp *fusion_host_f32(
    PolyCtx *ctx,
    float *data,
    int64_t *shape,
    int ndim
) {
  int dtype = poly_dtype_id_by_name("float32");
  if (dtype < 0) return NULL;
  int64_t numel = 1;
  for (int i = 0; i < ndim; i++) numel *= shape[i];
  return poly_buffer_from_host(ctx, data, (size_t)numel * sizeof(float), dtype, shape, ndim);
}

static PolyTensor *fusion_realize_value(
    PolyCtx *ctx,
    PolyUOp *value,
    PolyDevice device
) {
  PolyTensor *tensor = poly_tensor_create(ctx, value, POLY_TENSOR_VALUE, device);
  if (!tensor) return NULL;
  PolyTensor *out = NULL;
  if (poly_realize_tensors(ctx, &tensor, 1, &out) != 0) return NULL;
  return out;
}

static int fusion_read_tensor(
    PolyCtx *ctx,
    PolyTensor *tensor,
    float *out,
    int n
) {
  if (!ctx || !tensor || !out || n < 0) return -1;
  const PolyUOp *buf = poly_uop_get_buffer_identity(poly_tensor_uop(tensor));
  if (!buf) return -1;
  return poly_buffer_read(ctx, (PolyUOp *)buf, out, (size_t)n * sizeof(float));
}

static PolyUOp *fusion_build_case(
    PolyCtx *ctx,
    FusionCase which,
    PolyDevice device
) {
  switch (which) {
    case FUSION_CASE_MOVEMENT_REDUCE: {
      static float x_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
      int64_t x_shape[] = {3, 4};
      PolyUOp *x = fusion_host_f32(ctx, x_data, x_shape, 2);
      if (!x) return NULL;
      PolyUOp *p = poly_pad(ctx, x, (int64_t[][2]){{1, 0}, {0, 1}}, 2);
      PolyUOp *s = p ? poly_shrink(ctx, p, (int64_t[][2]){{1, 4}, {1, 5}}, 2) : NULL;
      PolyUOp *scale = poly_full(ctx, x_shape, 2, 0.25);
      PolyUOp *one = poly_full(ctx, x_shape, 2, 1.0);
      PolyUOp *y = (s && scale && one) ? poly_add(ctx, poly_mul(ctx, s, scale), one) : NULL;
      return y ? poly_sum_reduce(ctx, y, 1, 0) : NULL;
    }

    case FUSION_CASE_WHERE_REDUCE: {
      static float x_data[] = {
          0, 1, 2, 3,
          4, 5, 6, 7,
          8, 9, 10, 11,
          12, 13, 14, 15,
      };
      int64_t shape[] = {4, 4};
      PolyUOp *x = fusion_host_f32(ctx, x_data, shape, 2);
      PolyUOp *seven = poly_full(ctx, shape, 2, 7.0);
      PolyUOp *neg = poly_full(ctx, shape, 2, -2.0);
      PolyUOp *mask = (x && seven) ? poly_gt(ctx, seven, x) : NULL;
      PolyUOp *sel = (mask && neg) ? poly_where_op(ctx, mask, x, neg) : NULL;
      return sel ? poly_sum_reduce(ctx, sel, 0, 0) : NULL;
    }

    case FUSION_CASE_NO_BARRIER_REDUCE: {
      static float x_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
      int64_t shape[] = {3, 3};
      PolyUOp *x = fusion_host_f32(ctx, x_data, shape, 2);
      PolyUOp *one = poly_full(ctx, shape, 2, 1.0);
      PolyUOp *y = (x && one) ? poly_add(ctx, x, one) : NULL;
      PolyUOp *z = y ? poly_add(ctx, poly_square(ctx, y), y) : NULL;
      return z ? poly_sum_reduce(ctx, z, 0, 0) : NULL;
    }

    case FUSION_CASE_EXPLICIT_BARRIER_REDUCE: {
      static float x_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
      int64_t shape[] = {3, 3};
      PolyUOp *x = fusion_host_f32(ctx, x_data, shape, 2);
      PolyUOp *one = poly_full(ctx, shape, 2, 1.0);
      PolyUOp *mid = (x && one) ? poly_add(ctx, x, one) : NULL;
      PolyTensor *mid_t = mid ? fusion_realize_value(ctx, mid, device) : NULL;
      PolyUOp *mid_uop = mid_t ? poly_tensor_uop(mid_t) : NULL;
      PolyUOp *z = mid_uop ? poly_add(ctx, poly_square(ctx, mid_uop), mid_uop) : NULL;
      return z ? poly_sum_reduce(ctx, z, 0, 0) : NULL;
    }

    case FUSION_CASE_QR_RECONSTRUCT: {
      static float a_data[] = {1.0f, 2.0f, -1.0f, 3.0f, 0.5f, 4.0f};
      int64_t shape[] = {3, 2};
      PolyUOp *a = fusion_host_f32(ctx, a_data, shape, 2);
      PolyUOp *q = NULL, *r = NULL;
      if (!a || poly_qr(ctx, a, &q, &r) != 0 || !q || !r) return NULL;
      return poly_dot(ctx, q, r);
    }

    case FUSION_CASE_QR_BARRIER_RECONSTRUCT: {
      static float a_data[] = {1.0f, 2.0f, -1.0f, 3.0f, 0.5f, 4.0f};
      int64_t shape[] = {3, 2};
      PolyUOp *a = fusion_host_f32(ctx, a_data, shape, 2);
      PolyUOp *q = NULL, *r = NULL;
      if (!a || poly_qr(ctx, a, &q, &r) != 0 || !q || !r) return NULL;
      PolyTensor *q_t = fusion_realize_value(ctx, q, device);
      PolyTensor *r_t = fusion_realize_value(ctx, r, device);
      PolyUOp *q_uop = q_t ? poly_tensor_uop(q_t) : NULL;
      PolyUOp *r_uop = r_t ? poly_tensor_uop(r_t) : NULL;
      return (q_uop && r_uop) ? poly_dot(ctx, q_uop, r_uop) : NULL;
    }

    case FUSION_CASE_CHOLESKY_SOLVE: {
      static float a_data[] = {4, 2, 2, 5};
      static float b_data[] = {1, 2, 3, 4};
      int64_t shape[] = {2, 2};
      PolyUOp *a = fusion_host_f32(ctx, a_data, shape, 2);
      PolyUOp *b = fusion_host_f32(ctx, b_data, shape, 2);
      PolyUOp *l = a ? poly_cholesky(ctx, a, 0) : NULL;
      return (l && b) ? poly_cholesky_solve(ctx, l, b, 0) : NULL;
    }

    case FUSION_CASE_CHOLESKY_BARRIER_SOLVE: {
      static float a_data[] = {4, 2, 2, 5};
      static float b_data[] = {1, 2, 3, 4};
      int64_t shape[] = {2, 2};
      PolyUOp *a = fusion_host_f32(ctx, a_data, shape, 2);
      PolyUOp *b = fusion_host_f32(ctx, b_data, shape, 2);
      PolyUOp *l = a ? poly_cholesky(ctx, a, 0) : NULL;
      PolyTensor *l_t = l ? fusion_realize_value(ctx, l, device) : NULL;
      PolyUOp *l_uop = l_t ? poly_tensor_uop(l_t) : NULL;
      return (l_uop && b) ? poly_cholesky_solve(ctx, l_uop, b, 0) : NULL;
    }

    case FUSION_CASE_SOLVE: {
      static float a_data[] = {0, 2, 1, 3};
      static float b_data[] = {4, 5};
      int64_t a_shape[] = {2, 2};
      int64_t b_shape[] = {2};
      PolyUOp *a = fusion_host_f32(ctx, a_data, a_shape, 2);
      PolyUOp *b = fusion_host_f32(ctx, b_data, b_shape, 1);
      return (a && b) ? poly_solve(ctx, a, b) : NULL;
    }

    case FUSION_CASE_LSTSQ_TALL: {
      static float a_data[] = {1, 0, 1, 1, 1, 2};
      static float b_data[] = {1, 2, 3};
      int64_t a_shape[] = {3, 2};
      int64_t b_shape[] = {3};
      PolyUOp *a = fusion_host_f32(ctx, a_data, a_shape, 2);
      PolyUOp *b = fusion_host_f32(ctx, b_data, b_shape, 1);
      return (a && b) ? poly_lstsq(ctx, a, b) : NULL;
    }

    default:
      return NULL;
  }
}

static int fusion_run_case(
    FusionCase which,
    PolyDevice device,
    float *out,
    int out_cap
) {
  if (which < 0 || which >= FUSION_CASE_COUNT) return -1;
  const FusionCaseInfo *info = &FUSION_CASES[which];
  if (out_cap < info->out_numel) return -1;

  PolyCtx *ctx = poly_ctx_new();
  if (!ctx) return -1;
  poly_ctx_set_preferred_device(ctx, device);

  PolyUOp *value = fusion_build_case(ctx, which, device);
  PolyTensor *tensor = value ? fusion_realize_value(ctx, value, device) : NULL;
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
          "interp %s lane %d got %.8g expected %.8g (abs %.8g tol %.8g)",
          info->name,
          lane,
          (double)got,
          (double)exp,
          (double)fabsf(got - exp),
          (double)info->tol
      );
    }
  }
  PASS();
}

TEST(fusion, generated_graphs_cuda_parity_when_available) {
#ifndef POLY_HAS_CUDA
  SKIP("CUDA backend not compiled");
#else
  if (poly_selftest_device(POLY_DEVICE_CUDA) != 0)
    SKIP("CUDA runtime unavailable");

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
          "cuda %s lane %d got %.8g expected %.8g (abs %.8g tol %.8g)",
          info->name,
          lane,
          (double)got,
          (double)exp,
          (double)fabsf(got - exp),
          (double)info->tol
      );
    }
  }
  PASS();
#endif
}
