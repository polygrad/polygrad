/*
 * test_autograd.c — End-to-end tests for reverse-mode autodiff
 */

#include <math.h>

#include "test_harness.h"
#include "../src/ctx.h"
#include "../src/polygrad.h"
#include "../src/engine/schedule.h"
#include "../src/schedule/rangeify.h"
#include "../src/codegen.h"
#include "../src/frontend.h"
#include "../src/frontend_internal.h"
#include "../src/tensor.h"

/* Autograd e2e tests compile concrete scheduled kernels, not the earlier
 * pre-codegen kernel graph returned by poly_get_kernel_graph(). */
static PolyUOp *single_scheduled_root(PolyCtx *ctx, PolyUOp *sink) {
  PolySchedule *schedule = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  if (!schedule) return NULL;
  if (schedule->template->n_calls != 1 || !poly_schedule_call_body(schedule, 0)) {
    poly_schedule_free(schedule);
    return NULL;
  }
  PolyUOp *root = poly_schedule_call_body(schedule, 0);
  poly_schedule_free(schedule);
  return root;
}

#define LN2_F 0.69314718055994530942f

static int run_grad_expr(
    PolyCtx *ctx,
    PolyUOp *out_buf,
    PolyUOp *expr,
    const char *fn_name,
    void **args,
    int n_args
) {
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out_buf, expr, poly_arg_none());
  PolyUOp *sink = poly_uop(ctx, POLY_OP_SINK, POLY_VOID, (PolyUOp *[]){store}, 1, poly_arg_none());

  PolySchedule *schedule = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  if (!schedule) return -1;
  if (schedule->template->n_calls != 1 || !poly_schedule_call_body(schedule, 0)) {
    poly_schedule_free(schedule);
    return -1;
  }

  if (poly_schedule_call_is_copy(schedule, 0)) {
    PolyTestBufferView *bindings =
        calloc((size_t)(n_args > 0 ? n_args : 1), sizeof(PolyTestBufferView));
    if (!bindings) {
      poly_schedule_free(schedule);
      return -1;
    }
    int n_bindings = 0;
    for (int i = 0; i < schedule->template->n_buf_slots; i++) {
      if (schedule->template->buf_slots[i].is_intermediate) continue;
      if (n_bindings >= n_args) {
        free(bindings);
        poly_schedule_free(schedule);
        return -1;
      }
      bindings[n_bindings] = POLY_TEST_HOST_VIEW(schedule->template->buf_slots[i].buf_uop, args[n_bindings]);
      n_bindings++;
    }
    poly_schedule_free(schedule);
    int ret = poly_test_realize_buffer_views(ctx, sink, bindings, n_bindings);
    free(bindings);
    return ret;
  }

  PolyUOp *kernel = poly_schedule_call_body(schedule, 0);
  int n_lin = 0;
  PolyUOp **lin = poly_linearize(ctx, kernel, &n_lin);
  if (!lin || n_lin <= 0) {
    poly_schedule_free(schedule);
    free(lin);
    return -1;
  }

  char *src = poly_render_c(lin, n_lin, fn_name);
  free(lin);
  if (!src) {
    poly_schedule_free(schedule);
    return -1;
  }

  PolyProgram *prog = poly_compile_c(src, fn_name);
  free(src);
  if (!prog) {
    poly_schedule_free(schedule);
    return -1;
  }

  poly_program_call(prog, args, n_args);
  poly_program_destroy(prog);
  poly_schedule_free(schedule);
  return 0;
}

#define RUN_GRAD_EXPR(ctx, out_buf, expr, fn_name, args, n_args)                                   \
  do {                                                                                             \
    ASSERT_INT_EQ(run_grad_expr((ctx), (out_buf), (expr), (fn_name), (args), (n_args)), 0);        \
  } while (0)

TEST(autograd, after_store_passes_gradient_to_stored_value) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_buffer(ctx, POLY_FLOAT32, 4);
  PolyUOp *target = poly_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CPU);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, target, x, poly_arg_none());
  PolyUOp *after = poly_uop2(ctx, POLY_OP_AFTER, POLY_FLOAT32, target, store, poly_arg_none());
  PolyUOp *loss = poly_reduce_axis(ctx, POLY_OP_ADD, after, (int64_t[]){0}, 1);
  PolyUOp *gx = poly_grad(ctx, loss, x);
  ASSERT_NOT_NULL(gx);

  float gx_data[4] = {0};
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, 4);
  void *args[1] = {gx_data};
  RUN_GRAD_EXPR(ctx, out, gx, "ad_after_store", args, 1);
  for (int i = 0; i < 4; i++)
    ASSERT_FLOAT_EQ(gx_data[i], 1.0f, 1e-6f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(autograd, after_call_splits_data_and_boundary_gradients) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *data = poly_buffer(ctx, POLY_FLOAT32, 4);
  PolyUOp *body = poly_uop_sink(ctx, &data, 1);
  PolyUOp *args[] = {data};
  PolyUOp *call = poly_uop_call(ctx, body, args, 1);
  PolyUOp *after = poly_uop_after(ctx, data, call);
  PolyUOp *loss = poly_reduce_axis(ctx, POLY_OP_ADD, after, (int64_t[]){0}, 1);
  PolyUOp *gdata = poly_grad(ctx, loss, data);
  PolyUOp *gafter = poly_grad(ctx, loss, after);
  ASSERT_NOT_NULL(gdata);
  ASSERT_NOT_NULL(gafter);

  float gdata_values[4] = {0};
  float gafter_values[4] = {0};
  PolyUOp *data_out = poly_buffer(ctx, POLY_FLOAT32, 4);
  PolyUOp *after_out = poly_buffer(ctx, POLY_FLOAT32, 4);
  void *data_args[] = {gdata_values};
  void *after_args[] = {gafter_values};
  RUN_GRAD_EXPR(ctx, data_out, gdata, "ad_after_call_data", data_args, 1);
  RUN_GRAD_EXPR(ctx, after_out, gafter, "ad_after_call_boundary", after_args, 1);
  for (int i = 0; i < 4; i++) {
    ASSERT_FLOAT_EQ(gdata_values[i], 1.0f, 1e-6f);
    ASSERT_FLOAT_EQ(gafter_values[i], 1.0f, 1e-6f);
  }

  poly_ctx_destroy(ctx);
  PASS();
}

static int compile_expr_program(
    PolyCtx *ctx,
    PolyUOp *expr,
    const char *fn_name,
    PolyProgram **prog_out
) {
  PolyUOp *out = poly_buffer(ctx, poly_dtype_scalar(expr->dtype), 1);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out, expr, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  int n_lin = 0;
  PolyUOp *kernel = single_scheduled_root(ctx, sink);
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
static int finite_diff_check(
    PolyCtx *ctx,
    PolyUOp *loss,
    float **inputs,
    int n_inputs,
    int target_input,
    int n,
    const float *ad_grad,
    float h,
    float tol,
    const char *fn_name,
    int *bad_i,
    float *bad_num,
    float *bad_ad
) {
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
  for (int i = 0; i < n_inputs; i++)
    args[i + 1] = inputs[i];

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
  /* Pinned gradient.py:64 constructs x*ctx for each MUL source, and
   * compute_gradient:124 accumulates the repeated contribution with ADD.
   * Preserve both ordered topology and shared-node identity before lowering. */
  ASSERT_INT_EQ(gx->op, POLY_OP_ADD);
  ASSERT_TRUE(gx->src[0] == gx->src[1]);
  ASSERT_INT_EQ(gx->src[0]->op, POLY_OP_MUL);
  ASSERT_TRUE(gx->src[0]->src[0] == x);

  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, N);
  void *args[2] = {gx_d, x_d};
  RUN_GRAD_EXPR(ctx, out, gx, "ad_mul_sum", args, 2);

  for (int i = 0; i < N; i++)
    ASSERT_FLOAT_EQ(gx_d[i], 2.0f * x_d[i], 1e-5);

  int bad_i;
  float bad_num, bad_ad;
  float *inputs[] = {x_d};
  ASSERT_TRUE(finite_diff_check(
      ctx, loss, inputs, 1, 0, N, gx_d, 1e-3f, 2e-3f, "fd_mul_sum", &bad_i, &bad_num, &bad_ad
  ));
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
  void *args[2] = {gx_d, x_d};
  RUN_GRAD_EXPR(ctx, out, gx, "ad_fdiv_const", args, 2);

  for (int i = 0; i < N; i++)
    ASSERT_FLOAT_EQ(gx_d[i], 0.5f, 1e-5);

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
  int64_t aligned_shape[] = {1};
  PolyUOp *aligned = poly_reshape(ctx, xs, aligned_shape, 1);
  PolyUOp *xe = poly_expand(ctx, aligned, eshape, 1);
  ASSERT_INT_EQ(xe->op, POLY_OP_EXPAND);
  ASSERT_INT_EQ(xe->n_src, 2);
  ASSERT_INT_EQ(xe->arg.kind, POLY_ARG_NONE);
  ASSERT_PTR_EQ(xe->src[0], aligned);
  ASSERT_INT_EQ(aligned->op, POLY_OP_RESHAPE);
  ASSERT_INT_EQ(aligned->n_src, 2);
  int64_t ax[] = {0};
  PolyUOp *loss = poly_reduce_axis(ctx, POLY_OP_ADD, xe, ax, 1);
  PolyUOp *gx = poly_grad(ctx, loss, x);
  ASSERT_NOT_NULL(gx);
  ASSERT_INT_EQ(gx->op, POLY_OP_RESHAPE);
  ASSERT_INT_EQ(gx->n_src, 2);
  ASSERT_INT_EQ(gx->src[0]->op, POLY_OP_RESHAPE);
  ASSERT_INT_EQ(gx->src[0]->n_src, 2);
  ASSERT_INT_EQ(gx->src[0]->src[0]->op, POLY_OP_REDUCE);
  ASSERT_INT_EQ(gx->src[0]->src[0]->arg.kind, POLY_ARG_REDUCE_AXIS);
  ASSERT_INT_EQ(gx->src[0]->src[0]->arg.reduce_axis.op, POLY_OP_ADD);
  ASSERT_INT_EQ(gx->src[0]->src[0]->arg.reduce_axis.n, 1);
  ASSERT_INT_EQ(gx->src[0]->src[0]->arg.reduce_axis.axes[0], 0);
  ASSERT_INT_EQ(gx->src[0]->src[0]->src[0]->op, POLY_OP_EXPAND);
  ASSERT_INT_EQ(gx->src[0]->src[0]->src[0]->n_src, 2);
  ASSERT_INT_EQ(gx->src[0]->src[0]->src[0]->src[0]->op, POLY_OP_RESHAPE);

  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, 1);
  void *args[2] = {gx_d, x_d};
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
  void *args[2] = {gx_d, x_d};
  RUN_GRAD_EXPR(ctx, out, gx, "ad_permute", args, 2);

  for (int i = 0; i < 6; i++)
    ASSERT_FLOAT_EQ(gx_d[i], 1.0f, 1e-5);

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
  void *args[2] = {gx_d, x_d};
  RUN_GRAD_EXPR(ctx, out, gx, "ad_shrink", args, 2);

  float expected[6] = {0, 1, 1, 1, 1, 0};
  for (int i = 0; i < 6; i++)
    ASSERT_FLOAT_EQ(gx_d[i], expected[i], 1e-5);

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
  void *args[2] = {gx_d, x_d};
  RUN_GRAD_EXPR(ctx, out, gx, "ad_pad", args, 2);

  for (int i = 0; i < 4; i++)
    ASSERT_FLOAT_EQ(gx_d[i], 1.0f, 1e-5);

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
  ASSERT_INT_EQ(gb->op, POLY_OP_EXPAND);
  ASSERT_INT_EQ(gb->arg.kind, POLY_ARG_NONE);
  ASSERT_INT_EQ(gb->n_src, 2);
  ASSERT_INT_EQ(gb->src[0]->op, POLY_OP_RESHAPE);
  ASSERT_INT_EQ(gb->src[0]->n_src, 2);
  ASSERT_INT_EQ(gb->src[0]->src[0]->op, POLY_OP_CONST);
  ASSERT_INT_EQ(gb->src[1]->op, POLY_OP_STACK);
  ASSERT_INT_EQ(gb->src[1]->n_src, 1);
  ASSERT_INT_EQ(gb->src[1]->src[0]->op, POLY_OP_CONST);
  ASSERT_INT_EQ(gb->src[1]->src[0]->arg.i, N);

  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, N);
  void *args[3] = {gb_d, a_d, b_d};
  RUN_GRAD_EXPR(ctx, out, gb, "ad_zero", args, 3);

  for (int i = 0; i < N; i++)
    ASSERT_FLOAT_EQ(gb_d[i], 0.0f, 1e-5);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(autograd, no_path_zero_preserves_symbolic_shape_sources) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *n = poly_define_var(ctx, "n", 1, 8);
  int64_t inner[] = {2};
  PolyUOp *target = poly_buffer_var(ctx, POLY_FLOAT32, n, inner, 1);
  PolyUOp *unrelated = poly_buffer_f32(ctx, 1);
  PolyUOp *loss = poly_reduce_axis(ctx, POLY_OP_ADD, unrelated, (int64_t[]){0}, 1);
  PolyUOp *grad = poly_grad(ctx, loss, target);
  ASSERT_NOT_NULL(grad);

  ASSERT_INT_EQ(grad->op, POLY_OP_EXPAND);
  ASSERT_INT_EQ(grad->arg.kind, POLY_ARG_NONE);
  ASSERT_INT_EQ(grad->n_src, 2);
  ASSERT_INT_EQ(grad->src[0]->op, POLY_OP_RESHAPE);
  ASSERT_INT_EQ(grad->src[0]->n_src, 2);
  ASSERT_INT_EQ(grad->src[1]->op, POLY_OP_STACK);
  ASSERT_INT_EQ(grad->src[1]->n_src, 2);
  ASSERT_PTR_EQ(grad->src[1]->src[0], n);
  ASSERT_INT_EQ(grad->src[1]->src[1]->op, POLY_OP_CONST);
  ASSERT_INT_EQ(grad->src[1]->src[1]->arg.kind, POLY_ARG_INT);
  ASSERT_INT_EQ(grad->src[1]->src[1]->arg.i, 2);

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
  void *args[2] = {gx_d, x_d};
  RUN_GRAD_EXPR(ctx, out, gx, "ad_neg", args, 2);

  for (int i = 0; i < N; i++)
    ASSERT_FLOAT_EQ(gx_d[i], -1.0f, 1e-5);

  int bad_i;
  float bad_num, bad_ad;
  float *inputs[] = {x_d};
  ASSERT_TRUE(finite_diff_check(
      ctx, loss, inputs, 1, 0, N, gx_d, 1e-3f, 2e-3f, "fd_neg", &bad_i, &bad_num, &bad_ad
  ));
  ASSERT_INT_EQ(bad_i, -1);

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
  void *args_x[3] = {gx_d, x_d, y_d};
  void *args_y[3] = {gy_d, x_d, y_d};
  RUN_GRAD_EXPR(ctx, outx, gx, "ad_add_x", args_x, 3);
  RUN_GRAD_EXPR(ctx, outy, gy, "ad_add_y", args_y, 3);

  for (int i = 0; i < N; i++) {
    ASSERT_FLOAT_EQ(gx_d[i], 1.0f, 1e-5);
    ASSERT_FLOAT_EQ(gy_d[i], 1.0f, 1e-5);
  }

  int bad_i;
  float bad_num, bad_ad;
  float *inputs[] = {x_d, y_d};
  ASSERT_TRUE(finite_diff_check(
      ctx, loss, inputs, 2, 0, N, gx_d, 1e-3f, 2e-3f, "fd_add_x", &bad_i, &bad_num, &bad_ad
  ));
  ASSERT_INT_EQ(bad_i, -1);
  ASSERT_TRUE(finite_diff_check(
      ctx, loss, inputs, 2, 1, N, gy_d, 1e-3f, 2e-3f, "fd_add_y", &bad_i, &bad_num, &bad_ad
  ));
  ASSERT_INT_EQ(bad_i, -1);

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
  void *args_x[3] = {gx_d, x_d, y_d};
  void *args_y[3] = {gy_d, x_d, y_d};
  RUN_GRAD_EXPR(ctx, outx, gx, "ad_sub_x", args_x, 3);
  RUN_GRAD_EXPR(ctx, outy, gy, "ad_sub_y", args_y, 3);

  for (int i = 0; i < N; i++) {
    ASSERT_FLOAT_EQ(gx_d[i], 1.0f, 1e-5);
    ASSERT_FLOAT_EQ(gy_d[i], -1.0f, 1e-5);
  }

  int bad_i;
  float bad_num, bad_ad;
  float *inputs[] = {x_d, y_d};
  ASSERT_TRUE(finite_diff_check(
      ctx, loss, inputs, 2, 0, N, gx_d, 1e-3f, 2e-3f, "fd_sub_x", &bad_i, &bad_num, &bad_ad
  ));
  ASSERT_INT_EQ(bad_i, -1);
  ASSERT_TRUE(finite_diff_check(
      ctx, loss, inputs, 2, 1, N, gy_d, 1e-3f, 2e-3f, "fd_sub_y", &bad_i, &bad_num, &bad_ad
  ));
  ASSERT_INT_EQ(bad_i, -1);

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
  void *args[2] = {gx_d, x_d};
  RUN_GRAD_EXPR(ctx, out, gx, "ad_exp2", args, 2);

  for (int i = 0; i < N; i++) {
    float expected = exp2f(x_d[i]) * LN2_F;
    ASSERT_FLOAT_EQ(gx_d[i], expected, 2e-5);
  }

  int bad_i;
  float bad_num, bad_ad;
  float *inputs[] = {x_d};
  ASSERT_TRUE(finite_diff_check(
      ctx, loss, inputs, 1, 0, N, gx_d, 1e-3f, 2e-3f, "fd_exp2", &bad_i, &bad_num, &bad_ad
  ));
  ASSERT_INT_EQ(bad_i, -1);

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
  void *args[2] = {gx_d, x_d};
  RUN_GRAD_EXPR(ctx, out, gx, "ad_log2", args, 2);

  for (int i = 0; i < N; i++) {
    float expected = 1.0f / (x_d[i] * LN2_F);
    ASSERT_FLOAT_EQ(gx_d[i], expected, 5e-5);
  }

  int bad_i;
  float bad_num, bad_ad;
  float *inputs[] = {x_d};
  ASSERT_TRUE(finite_diff_check(
      ctx, loss, inputs, 1, 0, N, gx_d, 1e-3f, 2e-3f, "fd_log2", &bad_i, &bad_num, &bad_ad
  ));
  ASSERT_INT_EQ(bad_i, -1);

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
  void *args[2] = {gx_d, x_d};
  RUN_GRAD_EXPR(ctx, out, gx, "ad_sqrt", args, 2);

  for (int i = 0; i < N; i++) {
    float expected = 1.0f / (2.0f * sqrtf(x_d[i]));
    ASSERT_FLOAT_EQ(gx_d[i], expected, 5e-5);
  }

  int bad_i;
  float bad_num, bad_ad;
  float *inputs[] = {x_d};
  ASSERT_TRUE(finite_diff_check(
      ctx, loss, inputs, 1, 0, N, gx_d, 1e-3f, 2e-3f, "fd_sqrt", &bad_i, &bad_num, &bad_ad
  ));
  ASSERT_INT_EQ(bad_i, -1);

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
  void *args[2] = {gx_d, x_d};
  RUN_GRAD_EXPR(ctx, out, gx, "ad_recip", args, 2);

  for (int i = 0; i < N; i++) {
    float expected = -1.0f / (x_d[i] * x_d[i]);
    ASSERT_FLOAT_EQ(gx_d[i], expected, 2e-5);
  }

  int bad_i;
  float bad_num, bad_ad;
  float *inputs[] = {x_d};
  ASSERT_TRUE(finite_diff_check(
      ctx, loss, inputs, 1, 0, N, gx_d, 1e-3f, 2e-3f, "fd_recip", &bad_i, &bad_num, &bad_ad
  ));
  ASSERT_INT_EQ(bad_i, -1);

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
  void *args[2] = {gx_d, x_d};
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
  void *args[2] = {gx_d, x_d};
  RUN_GRAD_EXPR(ctx, out, gx, "ad_reshape", args, 2);

  for (int i = 0; i < N; i++)
    ASSERT_FLOAT_EQ(gx_d[i], 1.0f, 1e-5);

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
  void *args[2] = {gx_d, x_d};
  RUN_GRAD_EXPR(ctx, out, gx, "ad_flip", args, 2);

  for (int i = 0; i < N; i++)
    ASSERT_FLOAT_EQ(gx_d[i], 1.0f, 1e-5);

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
  void *args[2] = {gx_d, x_d};
  RUN_GRAD_EXPR(ctx, out, gx, "ad_chain_mul_exp2", args, 2);

  for (int i = 0; i < N; i++) {
    float expected = 2.0f * x_d[i] * exp2f(x_d[i] * x_d[i]) * LN2_F;
    ASSERT_FLOAT_EQ(gx_d[i], expected, 2e-4);
  }

  int bad_i;
  float bad_num, bad_ad;
  float *inputs[] = {x_d};
  ASSERT_TRUE(finite_diff_check(
      ctx, loss, inputs, 1, 0, N, gx_d, 1e-3f, 2e-3f, "fd_chain_mul_exp2", &bad_i, &bad_num, &bad_ad
  ));
  ASSERT_INT_EQ(bad_i, -1);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(autograd, max_reduce_backward_e2e) {
  /* reduce-MAX gradient requires multi-kernel scheduling (CONTIGUOUS barriers
   * create BUFFERIZE intermediates). Must use poly_test_realize_buffer_views, not single-kernel
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

  PolyTestBufferView bindings[] = {POLY_TEST_HOST_VIEW(x, x_d), POLY_TEST_HOST_VIEW(out, gx_d)};
  int ret = poly_test_realize_buffer_views(ctx, sink, bindings, 2);
  ASSERT_INT_EQ(ret, 0);

  float expected[6] = {0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
  for (int i = 0; i < N; i++)
    ASSERT_FLOAT_EQ(gx_d[i], expected[i], 1e-5);

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
  void *args_x[2] = {gx_d, y_d};
  void *args_y[3] = {gy_d, x_d, y_d};
  RUN_GRAD_EXPR(ctx, outx, gx, "ad_fdiv_both_x", args_x, 2);
  RUN_GRAD_EXPR(ctx, outy, gy, "ad_fdiv_both_y", args_y, 3);

  for (int i = 0; i < N; i++) {
    float ex = 1.0f / y_d[i];
    float ey = -x_d[i] / (y_d[i] * y_d[i]);
    ASSERT_FLOAT_EQ(gx_d[i], ex, 1e-5);
    ASSERT_FLOAT_EQ(gy_d[i], ey, 1e-5);
  }

  int bad_i;
  float bad_num, bad_ad;
  float *inputs[] = {x_d, y_d};
  ASSERT_TRUE(finite_diff_check(
      ctx, loss, inputs, 2, 0, N, gx_d, 1e-3f, 2e-3f, "fd_fdiv_x", &bad_i, &bad_num, &bad_ad
  ));
  ASSERT_INT_EQ(bad_i, -1);
  ASSERT_TRUE(finite_diff_check(
      ctx, loss, inputs, 2, 1, N, gy_d, 1e-3f, 2e-3f, "fd_fdiv_y", &bad_i, &bad_num, &bad_ad
  ));
  ASSERT_INT_EQ(bad_i, -1);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(autograd, substitute_nested_replacements_rewrite_replacement_graph) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_buffer(ctx, POLY_FLOAT32, 1);
  PolyUOp *y = poly_buffer(ctx, POLY_FLOAT32, 1);
  PolyUOp *z = poly_buffer(ctx, POLY_FLOAT32, 1);
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(1.0));
  PolyUOp *two = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(2.0));
  PolyUOp *replacement = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, y, one, poly_arg_none());
  PolyUOp *expr = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, x, two, poly_arg_none());

  PolyUOp *from[2] = {x, y};
  PolyUOp *to[2] = {replacement, z};
  PolyUOp *sub = poly_uop_substitute(ctx, expr, from, to, 2);

  ASSERT_TRUE(sub->op == POLY_OP_MUL);
  ASSERT_TRUE(sub->src[1] == two);
  ASSERT_TRUE(sub->src[0]->op == POLY_OP_ADD);
  ASSERT_TRUE(sub->src[0]->src[0] == z);
  ASSERT_TRUE(sub->src[0]->src[1] == one);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(autograd, substitute_keeps_call_and_function_bodies_opaque) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *old = poly_buffer(ctx, POLY_FLOAT32, 1);
  PolyUOp *new_value = poly_buffer(ctx, POLY_FLOAT32, 1);
  PolyUOp *body = poly_sink1(ctx, old);
  ASSERT_NOT_NULL(ctx);
  ASSERT_NOT_NULL(old);
  ASSERT_NOT_NULL(new_value);
  ASSERT_NOT_NULL(body);

  PolyOps opaque_ops[2] = {POLY_OP_CALL, POLY_OP_FUNCTION};
  for (int i = 0; i < 2; i++) {
    PolyUOp *opaque_src[2] = {body, old};
    PolyUOp *opaque = poly_uop(ctx, opaque_ops[i], POLY_VOID, opaque_src, 2, poly_arg_none());
    PolyUOp *root = poly_sink1(ctx, opaque);
    PolyUOp *from[1] = {old};
    PolyUOp *to[1] = {new_value};
    PolyUOp *rewritten = poly_uop_substitute(ctx, root, from, to, 1);
    ASSERT_NOT_NULL(opaque);
    ASSERT_NOT_NULL(root);
    ASSERT_NOT_NULL(rewritten);
    ASSERT_PTR_NEQ(rewritten, root);
    ASSERT_INT_EQ(rewritten->op, POLY_OP_SINK);
    ASSERT_INT_EQ(rewritten->n_src, 1);

    PolyUOp *rewritten_opaque = rewritten->src[0];
    ASSERT_NOT_NULL(rewritten_opaque);
    ASSERT_INT_EQ(rewritten_opaque->op, opaque_ops[i]);
    ASSERT_INT_EQ(rewritten_opaque->n_src, 2);
    ASSERT_PTR_EQ(rewritten_opaque->src[0], body);
    ASSERT_PTR_EQ(rewritten_opaque->src[0]->src[0], old);
    ASSERT_FALSE(poly_uop_reachable(ctx, rewritten_opaque->src[0], new_value));
    ASSERT_PTR_EQ(rewritten_opaque->src[1], new_value);
  }

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(autograd, substitute_many_opaque_body_is_root_order_independent) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *old = poly_buffer(ctx, POLY_FLOAT32, 1);
  PolyUOp *new_value = poly_buffer(ctx, POLY_FLOAT32, 1);
  PolyUOp *body = poly_sink1(ctx, old);
  ASSERT_NOT_NULL(ctx);
  ASSERT_NOT_NULL(old);
  ASSERT_NOT_NULL(new_value);
  ASSERT_NOT_NULL(body);

  PolyOps opaque_ops[2] = {POLY_OP_CALL, POLY_OP_FUNCTION};
  for (int op_index = 0; op_index < 2; op_index++) {
    PolyUOp *opaque_src[2] = {body, old};
    PolyUOp *opaque =
        poly_uop(ctx, opaque_ops[op_index], POLY_VOID, opaque_src, 2, poly_arg_none());
    ASSERT_NOT_NULL(opaque);

    for (int opaque_first = 0; opaque_first < 2; opaque_first++) {
      PolyUOp *roots[2] = {
          opaque_first ? opaque : body,
          opaque_first ? body : opaque,
      };
      PolyUOp *from[1] = {old};
      PolyUOp *to[1] = {new_value};
      PolyUOp *out[2] = {NULL, NULL};
      ASSERT_INT_EQ(poly_uop_substitute_many(ctx, roots, 2, from, to, 1, out), 0);

      PolyUOp *rewritten_body = out[opaque_first ? 1 : 0];
      PolyUOp *rewritten_opaque = out[opaque_first ? 0 : 1];
      ASSERT_PTR_EQ(rewritten_body, body);
      ASSERT_PTR_EQ(rewritten_body->src[0], old);
      ASSERT_FALSE(poly_uop_reachable(ctx, rewritten_body, new_value));
      ASSERT_NOT_NULL(rewritten_opaque);
      ASSERT_INT_EQ(rewritten_opaque->op, opaque_ops[op_index]);
      ASSERT_INT_EQ(rewritten_opaque->n_src, 2);
      ASSERT_PTR_EQ(rewritten_opaque->src[0], body);
      ASSERT_PTR_EQ(rewritten_opaque->src[1], new_value);
    }
  }

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(autograd, substitute_deep_chain_is_iterative) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_buffer(ctx, POLY_FLOAT32, 1);
  PolyUOp *y = poly_buffer(ctx, POLY_FLOAT32, 1);
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(1.0));
  PolyUOp *expr = x;
  const int depth = 20000;
  for (int i = 0; i < depth; i++) {
    expr = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, expr, one, poly_arg_none());
  }

  PolyUOp *from[1] = {x};
  PolyUOp *to[1] = {y};
  PolyUOp *sub = poly_uop_substitute(ctx, expr, from, to, 1);

  PolyUOp *cur = sub;
  for (int i = 0; i < depth; i++) {
    ASSERT_TRUE(cur->op == POLY_OP_ADD);
    ASSERT_TRUE(cur->src[1] == one);
    cur = cur->src[0];
  }
  ASSERT_TRUE(cur == y);

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
  void *args_x[3] = {gx_d, x_d, y_d};
  void *args_y[3] = {gy_d, x_d, y_d};
  RUN_GRAD_EXPR(ctx, outx, gx, "ad_multi_x", args_x, 3);
  RUN_GRAD_EXPR(ctx, outy, gy, "ad_multi_y", args_y, 3);

  for (int i = 0; i < N; i++) {
    ASSERT_FLOAT_EQ(gx_d[i], y_d[i] + 2.0f * x_d[i], 1e-5);
    ASSERT_FLOAT_EQ(gy_d[i], x_d[i], 1e-5);
  }

  /* Note: finite_diff_check not added here because the loss graph x*y + x*x
   * has x appearing in multiple subexpressions, causing PARAM ordering
   * ambiguity in the single-kernel compilation path. The analytical gradient
   * check above (y + 2x) is sufficient. The fdiv_both_e2e test validates
   * the finite_diff_check path for two-variable graphs. */

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(autograd, grad_reverse_pass_rewinds_scratch_toposort) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *x = poly_buffer(ctx, POLY_FLOAT32, 8);
  PolyUOp *y = poly_alu2(ctx, POLY_OP_MUL, x, x);
  PolyUOp *loss = poly_reduce_axis(ctx, POLY_OP_ADD, y, (int64_t[]){0}, 1);

  size_t scratch_before = poly_arena_used(ctx->scratch);
  PolyUOp *gx = poly_grad(ctx, loss, x);
  ASSERT_NOT_NULL(gx);
  ASSERT_INT_EQ(poly_arena_used(ctx->scratch), scratch_before);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(autograd, grad_many_reverse_pass_rewinds_scratch_toposort) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *x = poly_buffer(ctx, POLY_FLOAT32, 8);
  PolyUOp *y = poly_buffer(ctx, POLY_FLOAT32, 8);
  PolyUOp *xy = poly_alu2(ctx, POLY_OP_MUL, x, y);
  PolyUOp *loss = poly_reduce_axis(ctx, POLY_OP_ADD, xy, (int64_t[]){0}, 1);

  PolyUOp *wrts[] = {x, y};
  PolyUOp *grads[] = {NULL, NULL};
  size_t scratch_before = poly_arena_used(ctx->scratch);
  ASSERT_INT_EQ(poly_grad_many(ctx, loss, NULL, wrts, 2, grads), 0);
  ASSERT_NOT_NULL(grads[0]);
  ASSERT_NOT_NULL(grads[1]);
  ASSERT_INT_EQ(poly_arena_used(ctx->scratch), scratch_before);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(autograd, grad_many_ex_distinguishes_absent_from_numeric_zero) {
  PolyCtx *ctx = poly_ctx_new();
  int64_t shape[] = {4};
  int64_t axis[] = {0};
  PolyUOp *x = poly_buffer(ctx, POLY_FLOAT32, 4);
  PolyUOp *zero = poly_expand(ctx, poly_const_float(ctx, 0.0), shape, 1);
  PolyUOp *wrts[] = {x};
  PolyUOp *grads[] = {NULL};
  uint8_t present[] = {0};

  PolyUOp *numeric_zero = poly_alu2(ctx, POLY_OP_MUL, x, zero);
  PolyUOp *zero_loss = poly_reduce_axis(ctx, POLY_OP_ADD, numeric_zero, axis, 1);
  ASSERT_INT_EQ(poly_grad_many_ex(ctx, zero_loss, NULL, wrts, 1, grads, present), 0);
  ASSERT_NOT_NULL(grads[0]);
  ASSERT_INT_EQ(present[0], 1);

  PolyUOp *detached = poly_detach(ctx, x);
  PolyUOp *detach_loss = poly_reduce_axis(ctx, POLY_OP_ADD, detached, axis, 1);
  present[0] = 1;
  ASSERT_INT_EQ(poly_grad_many_ex(ctx, detach_loss, NULL, wrts, 1, grads, present), 0);
  ASSERT_NOT_NULL(grads[0]);
  ASSERT_INT_EQ(present[0], 0);

  PolyUOp *cmp = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, x, zero, poly_arg_none());
  PolyUOp *cmp_float = poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, cmp, poly_arg_none());
  PolyUOp *cmp_loss = poly_reduce_axis(ctx, POLY_OP_ADD, cmp_float, axis, 1);
  present[0] = 1;
  ASSERT_INT_EQ(poly_grad_many_ex(ctx, cmp_loss, NULL, wrts, 1, grads, present), 0);
  ASSERT_NOT_NULL(grads[0]);
  ASSERT_INT_EQ(present[0], 0);

  poly_ctx_destroy(ctx);
  PASS();
}
