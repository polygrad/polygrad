/*
 * test_autograd.c — End-to-end tests for reverse-mode autodiff
 */

#include <math.h>

#include "test_harness.h"
#include "../src/polygrad.h"
#include "../src/sched.h"
#include "../src/codegen.h"
#include "../src/frontend.h"

#define LN2_F 0.69314718055994530942f

#define RUN_GRAD_EXPR(ctx, out_buf, expr, fn_name, args, n_args) do { \
  PolyUOp *__store = poly_uop2((ctx), POLY_OP_STORE, POLY_VOID, (out_buf), (expr), poly_arg_none()); \
  PolyUOp *__sink = poly_uop((ctx), POLY_OP_SINK, POLY_VOID, (PolyUOp *[]){__store}, 1, poly_arg_none()); \
  PolyUOp *__kernel = poly_schedule((ctx), __sink); \
  ASSERT_NOT_NULL(__kernel); \
  int __n_lin = 0; \
  PolyUOp **__lin = poly_linearize((ctx), __kernel, &__n_lin); \
  ASSERT_TRUE(__n_lin > 0); \
  char *__src = poly_render_c(__lin, __n_lin, (fn_name)); \
  ASSERT_NOT_NULL(__src); \
  PolyProgram *__prog = poly_compile_c(__src, (fn_name)); \
  ASSERT_NOT_NULL(__prog); \
  poly_program_call(__prog, (args), (n_args)); \
  poly_program_destroy(__prog); \
  free(__src); \
  free(__lin); \
} while (0)

static int compile_expr_program(PolyCtx *ctx, PolyUOp *expr, const char *fn_name,
                                PolyProgram **prog_out) {
  PolyUOp *out = poly_buffer(ctx, poly_dtype_scalar(expr->dtype), 1);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out, expr, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  int n_lin = 0;
  PolyUOp *kernel = poly_schedule(ctx, sink);
  if (!kernel) return 0;
  PolyUOp **lin = poly_linearize(ctx, kernel, &n_lin);
  if (!lin) return 0;
  char *src = poly_render_c(lin, n_lin, fn_name);
  if (!src) {
    free(lin);
    return 0;
  }

  PolyProgram *prog = poly_compile_c(src, fn_name);
  free(src);
  free(lin);
  if (!prog) return 0;

  *prog_out = prog;
  return 1;
}

/* Central finite-difference check for one target input buffer.
 * Uses a single compiled loss program and replays it with +/- h perturbations. */
static int finite_diff_check(PolyCtx *ctx, PolyUOp *loss,
                              float **inputs, int n_inputs,
                              int target_input, int n,
                              const float *ad_grad,
                              float h, float tol,
                              const char *fn_name,
                              int *bad_i, float *bad_num, float *bad_ad) {
  *bad_i = -1;
  *bad_num = 0.0f;
  *bad_ad = 0.0f;

  PolyProgram *prog = NULL;
  if (!compile_expr_program(ctx, loss, fn_name, &prog)) return 0;

  float out = 0.0f;
  void *args[1 + 8] = {0};
  if (n_inputs > 8) {
    poly_program_destroy(prog);
    return 0;
  }
  args[0] = &out;
  for (int i = 0; i < n_inputs; i++) args[i + 1] = inputs[i];

  float *target = inputs[target_input];
  for (int i = 0; i < n; i++) {
    float orig = target[i];

    target[i] = orig + h;
    poly_program_call(prog, args, 1 + n_inputs);
    float plus = out;

    target[i] = orig - h;
    poly_program_call(prog, args, 1 + n_inputs);
    float minus = out;

    target[i] = orig;

    float num = (plus - minus) / (2.0f * h);
    float ad = ad_grad[i];
    if (fabsf(num - ad) > tol) {
      *bad_i = i;
      *bad_num = num;
      *bad_ad = ad;
      poly_program_destroy(prog);
      return 1;
    }
  }

  poly_program_destroy(prog);
  return 1;
}

TEST(autograd, mul_reduce_sum_1d_e2e) {
  int N = 8;
  float x_d[8], gx_d[8];
  for (int i = 0; i < N; i++) {
    x_d[i] = (float)(i - 3);
    gx_d[i] = 0.0f;
  }

  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, x, x, poly_arg_none());
  int64_t ax[] = {0};
  PolyUOp *loss = poly_reduce_axis(ctx, POLY_OP_ADD, mul, ax, 1);
  PolyUOp *gx = poly_grad(ctx, loss, x);
  ASSERT_NOT_NULL(gx);

  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, N);
  void *args[2] = { gx_d, x_d };
  RUN_GRAD_EXPR(ctx, out, gx, "ad_mul_sum", args, 2);

  for (int i = 0; i < N; i++)
    ASSERT_FLOAT_EQ(gx_d[i], 2.0f * x_d[i], 1e-5);

  int bad_i; float bad_num, bad_ad;
  float *inputs[] = { x_d };
  ASSERT_TRUE(finite_diff_check(ctx, loss, inputs, 1, 0, N, gx_d,
              1e-3f, 2e-3f, "fd_mul_sum", &bad_i, &bad_num, &bad_ad));
  ASSERT_INT_EQ(bad_i, -1);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(autograd, fdiv_const_reduce_sum_1d_e2e) {
  int N = 6;
  float x_d[6], gx_d[6];
  for (int i = 0; i < N; i++) {
    x_d[i] = (float)(i + 2);
    gx_d[i] = 0.0f;
  }

  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *c = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(2.0));
  PolyUOp *q = poly_uop2(ctx, POLY_OP_FDIV, POLY_FLOAT32, x, c, poly_arg_none());
  int64_t ax[] = {0};
  PolyUOp *loss = poly_reduce_axis(ctx, POLY_OP_ADD, q, ax, 1);
  PolyUOp *gx = poly_grad(ctx, loss, x);
  ASSERT_NOT_NULL(gx);

  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, N);
  void *args[2] = { gx_d, x_d };
  RUN_GRAD_EXPR(ctx, out, gx, "ad_fdiv_const", args, 2);

  for (int i = 0; i < N; i++) ASSERT_FLOAT_EQ(gx_d[i], 0.5f, 1e-5);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(autograd, expand_reduce_e2e) {
  float x_d[1] = {3};
  float gx_d[1] = {0};

  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_buffer(ctx, POLY_FLOAT32, 1);
  int64_t eshape[] = {5};
  PolyUOp *xs = poly_reshape(ctx, x, NULL, 0);
  PolyUOp *xe = poly_expand(ctx, xs, eshape, 1);
  int64_t ax[] = {0};
  PolyUOp *loss = poly_reduce_axis(ctx, POLY_OP_ADD, xe, ax, 1);
  PolyUOp *gx = poly_grad(ctx, loss, x);
  ASSERT_NOT_NULL(gx);

  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, 1);
  void *args[2] = { gx_d, x_d };
  RUN_GRAD_EXPR(ctx, out, gx, "ad_expand", args, 2);

  ASSERT_FLOAT_EQ(gx_d[0], 5.0f, 1e-5);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(autograd, permute_reduce_e2e) {
  float x_d[6], gx_d[6];
  for (int i = 0; i < 6; i++) {
    x_d[i] = (float)(i + 1);
    gx_d[i] = 0.0f;
  }

  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_buffer(ctx, POLY_FLOAT32, 6);
  int64_t shape[] = {2, 3};
  int64_t perm[] = {1, 0};
  PolyUOp *xr = poly_reshape(ctx, x, shape, 2);
  PolyUOp *xp = poly_permute(ctx, xr, perm, 2);
  int64_t ax[] = {0, 1};
  PolyUOp *loss = poly_reduce_axis(ctx, POLY_OP_ADD, xp, ax, 2);
  PolyUOp *gx = poly_grad(ctx, loss, x);
  ASSERT_NOT_NULL(gx);

  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, 6);
  void *args[2] = { gx_d, x_d };
  RUN_GRAD_EXPR(ctx, out, gx, "ad_permute", args, 2);

  for (int i = 0; i < 6; i++) ASSERT_FLOAT_EQ(gx_d[i], 1.0f, 1e-5);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(autograd, shrink_reduce_e2e) {
  float x_d[6], gx_d[6];
  for (int i = 0; i < 6; i++) {
    x_d[i] = (float)i;
    gx_d[i] = 0.0f;
  }

  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_buffer(ctx, POLY_FLOAT32, 6);
  int64_t pairs[1][2] = {{1, 5}};
  PolyUOp *xs = poly_shrink(ctx, x, pairs, 1);
  int64_t ax[] = {0};
  PolyUOp *loss = poly_reduce_axis(ctx, POLY_OP_ADD, xs, ax, 1);
  PolyUOp *gx = poly_grad(ctx, loss, x);
  ASSERT_NOT_NULL(gx);

  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, 6);
  void *args[2] = { gx_d, x_d };
  RUN_GRAD_EXPR(ctx, out, gx, "ad_shrink", args, 2);

  float expected[6] = {0, 1, 1, 1, 1, 0};
  for (int i = 0; i < 6; i++) ASSERT_FLOAT_EQ(gx_d[i], expected[i], 1e-5);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(autograd, pad_reduce_e2e) {
  float x_d[4] = {2, 4, 6, 8};
  float gx_d[4] = {0, 0, 0, 0};

  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_buffer(ctx, POLY_FLOAT32, 4);
  int64_t pairs[1][2] = {{1, 2}};
  PolyUOp *xp = poly_pad(ctx, x, pairs, 1);
  int64_t ax[] = {0};
  PolyUOp *loss = poly_reduce_axis(ctx, POLY_OP_ADD, xp, ax, 1);
  PolyUOp *gx = poly_grad(ctx, loss, x);
  ASSERT_NOT_NULL(gx);

  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, 4);
  void *args[2] = { gx_d, x_d };
  RUN_GRAD_EXPR(ctx, out, gx, "ad_pad", args, 2);

  for (int i = 0; i < 4; i++) ASSERT_FLOAT_EQ(gx_d[i], 1.0f, 1e-5);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(autograd, no_path_zero_e2e) {
  int N = 5;
  float a_d[5], b_d[5], gb_d[5];
  for (int i = 0; i < N; i++) {
    a_d[i] = (float)(i + 1);
    b_d[i] = (float)(10 + i);
    gb_d[i] = -1.0f;
  }

  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *b = poly_buffer(ctx, POLY_FLOAT32, N);
  int64_t ax[] = {0};
  PolyUOp *loss = poly_reduce_axis(ctx, POLY_OP_ADD, a, ax, 1);
  PolyUOp *gb = poly_grad(ctx, loss, b);
  ASSERT_NOT_NULL(gb);

  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, N);
  void *args[3] = { gb_d, a_d, b_d };
  RUN_GRAD_EXPR(ctx, out, gb, "ad_zero", args, 3);

  for (int i = 0; i < N; i++) ASSERT_FLOAT_EQ(gb_d[i], 0.0f, 1e-5);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(autograd, neg_reduce_sum_e2e) {
  int N = 7;
  float x_d[7], gx_d[7];
  for (int i = 0; i < N; i++) {
    x_d[i] = (float)(i - 2);
    gx_d[i] = 0.0f;
  }

  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *nx = poly_uop1(ctx, POLY_OP_NEG, POLY_FLOAT32, x, poly_arg_none());
  int64_t ax[] = {0};
  PolyUOp *loss = poly_reduce_axis(ctx, POLY_OP_ADD, nx, ax, 1);
  PolyUOp *gx = poly_grad(ctx, loss, x);
  ASSERT_NOT_NULL(gx);

  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, N);
  void *args[2] = { gx_d, x_d };
  RUN_GRAD_EXPR(ctx, out, gx, "ad_neg", args, 2);

  for (int i = 0; i < N; i++) ASSERT_FLOAT_EQ(gx_d[i], -1.0f, 1e-5);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(autograd, add_reduce_sum_e2e) {
  int N = 6;
  float x_d[6], y_d[6], gx_d[6], gy_d[6];
  for (int i = 0; i < N; i++) {
    x_d[i] = (float)(i + 1);
    y_d[i] = (float)(10 + i);
    gx_d[i] = 0.0f;
    gy_d[i] = 0.0f;
  }

  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *y = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *z = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, x, y, poly_arg_none());
  int64_t ax[] = {0};
  PolyUOp *loss = poly_reduce_axis(ctx, POLY_OP_ADD, z, ax, 1);
  PolyUOp *gx = poly_grad(ctx, loss, x);
  PolyUOp *gy = poly_grad(ctx, loss, y);
  ASSERT_NOT_NULL(gx);
  ASSERT_NOT_NULL(gy);

  PolyUOp *outx = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *outy = poly_buffer(ctx, POLY_FLOAT32, N);
  void *args_x[3] = { gx_d, x_d, y_d };
  void *args_y[3] = { gy_d, x_d, y_d };
  RUN_GRAD_EXPR(ctx, outx, gx, "ad_add_x", args_x, 3);
  RUN_GRAD_EXPR(ctx, outy, gy, "ad_add_y", args_y, 3);

  for (int i = 0; i < N; i++) {
    ASSERT_FLOAT_EQ(gx_d[i], 1.0f, 1e-5);
    ASSERT_FLOAT_EQ(gy_d[i], 1.0f, 1e-5);
  }

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(autograd, sub_reduce_sum_e2e) {
  int N = 6;
  float x_d[6], y_d[6], gx_d[6], gy_d[6];
  for (int i = 0; i < N; i++) {
    x_d[i] = (float)(i + 1);
    y_d[i] = (float)(10 + i);
    gx_d[i] = 0.0f;
    gy_d[i] = 0.0f;
  }

  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *y = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *z = poly_uop2(ctx, POLY_OP_SUB, POLY_FLOAT32, x, y, poly_arg_none());
  int64_t ax[] = {0};
  PolyUOp *loss = poly_reduce_axis(ctx, POLY_OP_ADD, z, ax, 1);
  PolyUOp *gx = poly_grad(ctx, loss, x);
  PolyUOp *gy = poly_grad(ctx, loss, y);
  ASSERT_NOT_NULL(gx);
  ASSERT_NOT_NULL(gy);

  PolyUOp *outx = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *outy = poly_buffer(ctx, POLY_FLOAT32, N);
  void *args_x[3] = { gx_d, x_d, y_d };
  void *args_y[3] = { gy_d, x_d, y_d };
  RUN_GRAD_EXPR(ctx, outx, gx, "ad_sub_x", args_x, 3);
  RUN_GRAD_EXPR(ctx, outy, gy, "ad_sub_y", args_y, 3);

  for (int i = 0; i < N; i++) {
    ASSERT_FLOAT_EQ(gx_d[i], 1.0f, 1e-5);
    ASSERT_FLOAT_EQ(gy_d[i], -1.0f, 1e-5);
  }

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(autograd, exp2_reduce_sum_e2e) {
  int N = 5;
  float x_d[5] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
  float gx_d[5] = {0};

  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *e = poly_uop1(ctx, POLY_OP_EXP2, POLY_FLOAT32, x, poly_arg_none());
  int64_t ax[] = {0};
  PolyUOp *loss = poly_reduce_axis(ctx, POLY_OP_ADD, e, ax, 1);
  PolyUOp *gx = poly_grad(ctx, loss, x);
  ASSERT_NOT_NULL(gx);

  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, N);
  void *args[2] = { gx_d, x_d };
  RUN_GRAD_EXPR(ctx, out, gx, "ad_exp2", args, 2);

  for (int i = 0; i < N; i++) {
    float expected = exp2f(x_d[i]) * LN2_F;
    ASSERT_FLOAT_EQ(gx_d[i], expected, 2e-5);
  }

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(autograd, log2_reduce_sum_e2e) {
  int N = 5;
  float x_d[5] = {1.0f, 2.0f, 4.0f, 8.0f, 16.0f};
  float gx_d[5] = {0};

  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *l = poly_uop1(ctx, POLY_OP_LOG2, POLY_FLOAT32, x, poly_arg_none());
  int64_t ax[] = {0};
  PolyUOp *loss = poly_reduce_axis(ctx, POLY_OP_ADD, l, ax, 1);
  PolyUOp *gx = poly_grad(ctx, loss, x);
  ASSERT_NOT_NULL(gx);

  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, N);
  void *args[2] = { gx_d, x_d };
  RUN_GRAD_EXPR(ctx, out, gx, "ad_log2", args, 2);

  for (int i = 0; i < N; i++) {
    float expected = 1.0f / (x_d[i] * LN2_F);
    ASSERT_FLOAT_EQ(gx_d[i], expected, 5e-5);
  }

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(autograd, sqrt_reduce_sum_e2e) {
  int N = 5;
  float x_d[5] = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f};
  float gx_d[5] = {0};

  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *s = poly_uop1(ctx, POLY_OP_SQRT, POLY_FLOAT32, x, poly_arg_none());
  int64_t ax[] = {0};
  PolyUOp *loss = poly_reduce_axis(ctx, POLY_OP_ADD, s, ax, 1);
  PolyUOp *gx = poly_grad(ctx, loss, x);
  ASSERT_NOT_NULL(gx);

  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, N);
  void *args[2] = { gx_d, x_d };
  RUN_GRAD_EXPR(ctx, out, gx, "ad_sqrt", args, 2);

  for (int i = 0; i < N; i++) {
    float expected = 1.0f / (2.0f * sqrtf(x_d[i]));
    ASSERT_FLOAT_EQ(gx_d[i], expected, 5e-5);
  }

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(autograd, recip_reduce_sum_e2e) {
  int N = 5;
  float x_d[5] = {1.0f, 2.0f, -3.0f, 4.0f, -5.0f};
  float gx_d[5] = {0};

  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *r = poly_uop1(ctx, POLY_OP_RECIPROCAL, POLY_FLOAT32, x, poly_arg_none());
  int64_t ax[] = {0};
  PolyUOp *loss = poly_reduce_axis(ctx, POLY_OP_ADD, r, ax, 1);
  PolyUOp *gx = poly_grad(ctx, loss, x);
  ASSERT_NOT_NULL(gx);

  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, N);
  void *args[2] = { gx_d, x_d };
  RUN_GRAD_EXPR(ctx, out, gx, "ad_recip", args, 2);

  for (int i = 0; i < N; i++) {
    float expected = -1.0f / (x_d[i] * x_d[i]);
    ASSERT_FLOAT_EQ(gx_d[i], expected, 2e-5);
  }

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(autograd, where_reduce_sum_e2e) {
  int N = 8;
  float x_d[8] = {-3.0f, -1.0f, 0.0f, 0.1f, 2.0f, -5.0f, 7.0f, -0.2f};
  float gx_d[8] = {0};

  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(0.0));
  PolyUOp *cond = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, zero, x, poly_arg_none());
  PolyUOp *y = poly_uop3(ctx, POLY_OP_WHERE, POLY_FLOAT32, cond, x, zero, poly_arg_none());
  int64_t ax[] = {0};
  PolyUOp *loss = poly_reduce_axis(ctx, POLY_OP_ADD, y, ax, 1);
  PolyUOp *gx = poly_grad(ctx, loss, x);
  ASSERT_NOT_NULL(gx);

  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, N);
  void *args[2] = { gx_d, x_d };
  RUN_GRAD_EXPR(ctx, out, gx, "ad_where", args, 2);

  for (int i = 0; i < N; i++) {
    float expected = x_d[i] > 0.0f ? 1.0f : 0.0f;
    ASSERT_FLOAT_EQ(gx_d[i], expected, 1e-5);
  }

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(autograd, reshape_reduce_e2e) {
  int N = 6;
  float x_d[6], gx_d[6];
  for (int i = 0; i < N; i++) {
    x_d[i] = (float)(i + 1);
    gx_d[i] = 0.0f;
  }

  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_buffer(ctx, POLY_FLOAT32, N);
  int64_t shape[] = {2, 3};
  PolyUOp *xr = poly_reshape(ctx, x, shape, 2);
  int64_t ax[] = {0, 1};
  PolyUOp *loss = poly_reduce_axis(ctx, POLY_OP_ADD, xr, ax, 2);
  PolyUOp *gx = poly_grad(ctx, loss, x);
  ASSERT_NOT_NULL(gx);

  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, N);
  void *args[2] = { gx_d, x_d };
  RUN_GRAD_EXPR(ctx, out, gx, "ad_reshape", args, 2);

  for (int i = 0; i < N; i++) ASSERT_FLOAT_EQ(gx_d[i], 1.0f, 1e-5);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(autograd, flip_reduce_e2e) {
  int N = 7;
  float x_d[7], gx_d[7];
  for (int i = 0; i < N; i++) {
    x_d[i] = (float)(i + 1);
    gx_d[i] = 0.0f;
  }

  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_buffer(ctx, POLY_FLOAT32, N);
  int64_t axes[] = {0};
  PolyUOp *xf = poly_flip(ctx, x, axes, 1);
  int64_t ax[] = {0};
  PolyUOp *loss = poly_reduce_axis(ctx, POLY_OP_ADD, xf, ax, 1);
  PolyUOp *gx = poly_grad(ctx, loss, x);
  ASSERT_NOT_NULL(gx);

  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, N);
  void *args[2] = { gx_d, x_d };
  RUN_GRAD_EXPR(ctx, out, gx, "ad_flip", args, 2);

  for (int i = 0; i < N; i++) ASSERT_FLOAT_EQ(gx_d[i], 1.0f, 1e-5);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(autograd, chain_mul_exp2_e2e) {
  int N = 6;
  float x_d[6] = {-1.5f, -0.5f, 0.25f, 0.75f, 1.25f, 2.0f};
  float gx_d[6] = {0};

  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *sq = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, x, x, poly_arg_none());
  PolyUOp *e = poly_uop1(ctx, POLY_OP_EXP2, POLY_FLOAT32, sq, poly_arg_none());
  int64_t ax[] = {0};
  PolyUOp *loss = poly_reduce_axis(ctx, POLY_OP_ADD, e, ax, 1);
  PolyUOp *gx = poly_grad(ctx, loss, x);
  ASSERT_NOT_NULL(gx);

  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, N);
  void *args[2] = { gx_d, x_d };
  RUN_GRAD_EXPR(ctx, out, gx, "ad_chain_mul_exp2", args, 2);

  for (int i = 0; i < N; i++) {
    float expected = 2.0f * x_d[i] * exp2f(x_d[i] * x_d[i]) * LN2_F;
    ASSERT_FLOAT_EQ(gx_d[i], expected, 2e-4);
  }

  int bad_i; float bad_num, bad_ad;
  float *inputs[] = { x_d };
  ASSERT_TRUE(finite_diff_check(ctx, loss, inputs, 1, 0, N, gx_d,
              1e-3f, 2e-3f, "fd_chain_mul_exp2", &bad_i, &bad_num, &bad_ad));
  ASSERT_INT_EQ(bad_i, -1);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(autograd, max_reduce_backward_e2e) {
  /* reduce-MAX gradient requires multi-kernel scheduling (CONTIGUOUS barriers
   * create BUFFERIZE intermediates). Must use poly_realize, not single-kernel
   * RUN_GRAD_EXPR. */
  int N = 6;
  float x_d[6] = {1.0f, 3.0f, 2.0f, 4.0f, 5.0f, 6.0f};
  float gx_d[6] = {0};

  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_buffer_f32(ctx, N);
  int64_t shape[] = {2, 3};
  PolyUOp *xr = poly_reshape(ctx, x, shape, 2);
  int64_t ax[] = {1};
  PolyUOp *m = poly_reduce_axis(ctx, POLY_OP_MAX, xr, ax, 1);
  int64_t ax2[] = {0};
  PolyUOp *loss = poly_reduce_axis(ctx, POLY_OP_ADD, m, ax2, 1);
  /* Differentiate wrt the reshaped UOp (logical shape), not the flat BUFFER.
   * This matches the Python frontend which uses leaf._uop (RESHAPE(BUFFER)). */
  PolyUOp *gx = poly_grad(ctx, loss, xr);
  ASSERT_NOT_NULL(gx);

  PolyUOp *out = poly_buffer_f32(ctx, N);
  PolyUOp *store = poly_store_val(ctx, out, gx);
  PolyUOp *sink = poly_sink1(ctx, store);

  PolyBufferBinding bindings[] = { {x, x_d}, {out, gx_d} };
  int ret = poly_realize(ctx, sink, bindings, 2);
  ASSERT_INT_EQ(ret, 0);

  float expected[6] = {0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
  for (int i = 0; i < N; i++) ASSERT_FLOAT_EQ(gx_d[i], expected[i], 1e-5);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(autograd, fdiv_both_e2e) {
  int N = 6;
  float x_d[6] = {1.0f, 2.0f, 3.0f, -1.0f, -2.0f, 4.0f};
  float y_d[6] = {2.0f, 4.0f, -2.0f, 5.0f, -3.0f, 8.0f};
  float gx_d[6] = {0}, gy_d[6] = {0};

  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *y = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *q = poly_uop2(ctx, POLY_OP_FDIV, POLY_FLOAT32, x, y, poly_arg_none());
  int64_t ax[] = {0};
  PolyUOp *loss = poly_reduce_axis(ctx, POLY_OP_ADD, q, ax, 1);
  PolyUOp *gx = poly_grad(ctx, loss, x);
  PolyUOp *gy = poly_grad(ctx, loss, y);
  ASSERT_NOT_NULL(gx);
  ASSERT_NOT_NULL(gy);

  PolyUOp *outx = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *outy = poly_buffer(ctx, POLY_FLOAT32, N);
  void *args_x[2] = { gx_d, y_d };
  void *args_y[3] = { gy_d, x_d, y_d };
  RUN_GRAD_EXPR(ctx, outx, gx, "ad_fdiv_both_x", args_x, 2);
  RUN_GRAD_EXPR(ctx, outy, gy, "ad_fdiv_both_y", args_y, 3);

  for (int i = 0; i < N; i++) {
    float ex = 1.0f / y_d[i];
    float ey = -x_d[i] / (y_d[i] * y_d[i]);
    ASSERT_FLOAT_EQ(gx_d[i], ex, 1e-5);
    ASSERT_FLOAT_EQ(gy_d[i], ey, 1e-5);
  }

  int bad_i; float bad_num, bad_ad;
  float *inputs[] = { x_d, y_d };
  ASSERT_TRUE(finite_diff_check(ctx, loss, inputs, 2, 0, N, gx_d,
              1e-3f, 2e-3f, "fd_fdiv_x", &bad_i, &bad_num, &bad_ad));
  ASSERT_INT_EQ(bad_i, -1);
  ASSERT_TRUE(finite_diff_check(ctx, loss, inputs, 2, 1, N, gy_d,
              1e-3f, 2e-3f, "fd_fdiv_y", &bad_i, &bad_num, &bad_ad));
  ASSERT_INT_EQ(bad_i, -1);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(autograd, multi_wrt_same_loss_e2e) {
  int N = 5;
  float x_d[5], y_d[5], gx_d[5], gy_d[5];
  for (int i = 0; i < N; i++) {
    x_d[i] = (float)(i + 1);
    y_d[i] = (float)(2 * i - 3);
    gx_d[i] = 0.0f;
    gy_d[i] = 0.0f;
  }

  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *y = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *xy = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, x, y, poly_arg_none());
  PolyUOp *xx = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, x, x, poly_arg_none());
  PolyUOp *val = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, xy, xx, poly_arg_none());
  int64_t ax[] = {0};
  PolyUOp *loss = poly_reduce_axis(ctx, POLY_OP_ADD, val, ax, 1);

  /* same loss, two separate wrt traversals */
  PolyUOp *gx = poly_grad(ctx, loss, x);
  PolyUOp *gy = poly_grad(ctx, loss, y);
  ASSERT_NOT_NULL(gx);
  ASSERT_NOT_NULL(gy);

  PolyUOp *outx = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *outy = poly_buffer(ctx, POLY_FLOAT32, N);
  void *args_x[3] = { gx_d, x_d, y_d };
  void *args_y[3] = { gy_d, x_d, y_d };
  RUN_GRAD_EXPR(ctx, outx, gx, "ad_multi_x", args_x, 3);
  RUN_GRAD_EXPR(ctx, outy, gy, "ad_multi_y", args_y, 3);

  for (int i = 0; i < N; i++) {
    ASSERT_FLOAT_EQ(gx_d[i], y_d[i] + 2.0f * x_d[i], 1e-5);
    ASSERT_FLOAT_EQ(gy_d[i], x_d[i], 1e-5);
  }

  poly_ctx_destroy(ctx);
  PASS();
}
