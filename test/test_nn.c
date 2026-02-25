/*
 * test_nn.c — Tests for the Tensor + neural network API
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "test_harness.h"
#include "../src/nn.h"
#include "../src/rangeify.h"
#include "../src/frontend.h"
#include "../src/sched.h"

/* ── M1: Creation + elementwise + movement + realize ─────────────────── */

TEST(nn, create_zeros) {
  int64_t shape[] = {2, 3};
  PolyTensor *t = poly_tensor_zeros(shape, 2);
  ASSERT_NOT_NULL(t);
  ASSERT_INT_EQ(t->ndim, 2);
  ASSERT_INT_EQ(t->shape[0], 2);
  ASSERT_INT_EQ(t->shape[1], 3);
  ASSERT_INT_EQ(t->numel, 6);
  ASSERT_NOT_NULL(t->data);
  for (int i = 0; i < 6; i++)
    ASSERT_FLOAT_EQ(t->data[i], 0.0f, 1e-7);
  poly_tensor_free(t);
  PASS();
}

TEST(nn, realize_add) {
  float a_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
  float b_data[] = {10.0f, 20.0f, 30.0f, 40.0f};

  PolyCtx *ctx = poly_ctx_new();
  PolyTensor *a = poly_tensor_input(ctx, a_data, (int64_t[]){4}, 1);
  PolyTensor *b = poly_tensor_input(ctx, b_data, (int64_t[]){4}, 1);
  PolyTensor *c = poly_tensor_add(a, b);

  int rc = poly_tensor_realize(c);
  ASSERT_INT_EQ(rc, 0);
  ASSERT_NOT_NULL(c->data);
  ASSERT_FLOAT_EQ(c->data[0], 11.0f, 1e-5);
  ASSERT_FLOAT_EQ(c->data[1], 22.0f, 1e-5);
  ASSERT_FLOAT_EQ(c->data[2], 33.0f, 1e-5);
  ASSERT_FLOAT_EQ(c->data[3], 44.0f, 1e-5);

  poly_tensor_free(c);
  poly_tensor_free(a);
  poly_tensor_free(b);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(nn, realize_chain) {
  float a_data[] = {1.0f, 2.0f, 3.0f};
  float b_data[] = {10.0f, 20.0f, 30.0f};
  float c_data[] = {2.0f, 2.0f, 2.0f};

  PolyCtx *ctx = poly_ctx_new();
  PolyTensor *a = poly_tensor_input(ctx, a_data, (int64_t[]){3}, 1);
  PolyTensor *b = poly_tensor_input(ctx, b_data, (int64_t[]){3}, 1);
  PolyTensor *c = poly_tensor_input(ctx, c_data, (int64_t[]){3}, 1);
  PolyTensor *ab = poly_tensor_add(a, b);
  PolyTensor *result = poly_tensor_mul(ab, c);

  int rc = poly_tensor_realize(result);
  ASSERT_INT_EQ(rc, 0);
  /* (1+10)*2=22, (2+20)*2=44, (3+30)*2=66 */
  ASSERT_FLOAT_EQ(result->data[0], 22.0f, 1e-5);
  ASSERT_FLOAT_EQ(result->data[1], 44.0f, 1e-5);
  ASSERT_FLOAT_EQ(result->data[2], 66.0f, 1e-5);

  poly_tensor_free(result);
  poly_tensor_free(ab);
  poly_tensor_free(a);
  poly_tensor_free(b);
  poly_tensor_free(c);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(nn, realize_reshape_2d) {
  float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  PolyCtx *ctx = poly_ctx_new();
  PolyTensor *a = poly_tensor_input(ctx, data, (int64_t[]){6}, 1);
  PolyTensor *r = poly_tensor_reshape(a, (int64_t[]){2, 3}, 2);

  ASSERT_INT_EQ(r->ndim, 2);
  ASSERT_INT_EQ(r->shape[0], 2);
  ASSERT_INT_EQ(r->shape[1], 3);

  int rc = poly_tensor_realize(r);
  ASSERT_INT_EQ(rc, 0);
  for (int i = 0; i < 6; i++)
    ASSERT_FLOAT_EQ(r->data[i], data[i], 1e-7);

  poly_tensor_free(r);
  poly_tensor_free(a);
  poly_ctx_destroy(ctx);
  PASS();
}

/* ── M2: Reductions + matmul + activations ───────────────────────────── */

TEST(nn, sum_axis) {
  /* 2x3 matrix: [[1,2,3],[4,5,6]], sum axis=1 → [6, 15] */
  float data[] = {1, 2, 3, 4, 5, 6};

  PolyCtx *ctx = poly_ctx_new();
  PolyTensor *a = poly_tensor_input(ctx, data, (int64_t[]){2, 3}, 2);
  PolyTensor *s = poly_tensor_sum(a, 1);

  ASSERT_INT_EQ(s->ndim, 1);
  ASSERT_INT_EQ(s->shape[0], 2);

  int rc = poly_tensor_realize(s);
  ASSERT_INT_EQ(rc, 0);
  ASSERT_FLOAT_EQ(s->data[0], 6.0f, 1e-5);
  ASSERT_FLOAT_EQ(s->data[1], 15.0f, 1e-5);

  poly_tensor_free(s);
  poly_tensor_free(a);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(nn, mean_all) {
  float data[] = {2, 4, 6, 8};

  PolyCtx *ctx = poly_ctx_new();
  PolyTensor *a = poly_tensor_input(ctx, data, (int64_t[]){4}, 1);
  PolyTensor *m = poly_tensor_mean(a, -1);

  int rc = poly_tensor_realize(m);
  ASSERT_INT_EQ(rc, 0);
  ASSERT_FLOAT_EQ(m->data[0], 5.0f, 1e-5);

  poly_tensor_free(m);
  poly_tensor_free(a);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(nn, matmul_2x2) {
  /* [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]] */
  float a_data[] = {1, 2, 3, 4};
  float b_data[] = {5, 6, 7, 8};

  PolyCtx *ctx = poly_ctx_new();
  PolyTensor *a = poly_tensor_input(ctx, a_data, (int64_t[]){2, 2}, 2);
  PolyTensor *b = poly_tensor_input(ctx, b_data, (int64_t[]){2, 2}, 2);
  PolyTensor *c = poly_tensor_matmul(a, b);

  ASSERT_INT_EQ(c->ndim, 2);
  ASSERT_INT_EQ(c->shape[0], 2);
  ASSERT_INT_EQ(c->shape[1], 2);

  int rc = poly_tensor_realize(c);
  ASSERT_INT_EQ(rc, 0);
  ASSERT_FLOAT_EQ(c->data[0], 19.0f, 1e-4);
  ASSERT_FLOAT_EQ(c->data[1], 22.0f, 1e-4);
  ASSERT_FLOAT_EQ(c->data[2], 43.0f, 1e-4);
  ASSERT_FLOAT_EQ(c->data[3], 50.0f, 1e-4);

  poly_tensor_free(c);
  poly_tensor_free(a);
  poly_tensor_free(b);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(nn, relu_values) {
  float data[] = {-3.0f, -1.0f, 0.0f, 0.5f, 2.0f};

  PolyCtx *ctx = poly_ctx_new();
  PolyTensor *a = poly_tensor_input(ctx, data, (int64_t[]){5}, 1);
  PolyTensor *r = poly_tensor_relu(a);

  int rc = poly_tensor_realize(r);
  ASSERT_INT_EQ(rc, 0);
  ASSERT_FLOAT_EQ(r->data[0], 0.0f, 1e-6);
  ASSERT_FLOAT_EQ(r->data[1], 0.0f, 1e-6);
  ASSERT_FLOAT_EQ(r->data[2], 0.0f, 1e-6);
  ASSERT_FLOAT_EQ(r->data[3], 0.5f, 1e-6);
  ASSERT_FLOAT_EQ(r->data[4], 2.0f, 1e-6);

  poly_tensor_free(r);
  poly_tensor_free(a);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(nn, sigmoid_values) {
  float data[] = {0.0f, 10.0f, -10.0f};

  PolyCtx *ctx = poly_ctx_new();
  PolyTensor *a = poly_tensor_input(ctx, data, (int64_t[]){3}, 1);
  PolyTensor *s = poly_tensor_sigmoid(a);

  int rc = poly_tensor_realize(s);
  ASSERT_INT_EQ(rc, 0);
  ASSERT_FLOAT_EQ(s->data[0], 0.5f, 1e-5);
  ASSERT_TRUE(s->data[1] > 0.999f);   /* sigmoid(10) ≈ 1 */
  ASSERT_TRUE(s->data[2] < 0.001f);   /* sigmoid(-10) ≈ 0 */

  poly_tensor_free(s);
  poly_tensor_free(a);
  poly_ctx_destroy(ctx);
  PASS();
}

/* ── M3: Autograd backward ───────────────────────────────────────────── */

TEST(nn, backward_mul) {
  /* loss = sum(a * b), d(loss)/da = b */
  float a_data[] = {1, 2, 3, 4};
  float b_data[] = {5, 6, 7, 8};

  PolyCtx *ctx = poly_ctx_new();
  PolyTensor *a = poly_tensor_input(ctx, a_data, (int64_t[]){4}, 1);
  PolyTensor *b = poly_tensor_input(ctx, b_data, (int64_t[]){4}, 1);
  PolyTensor *prod = poly_tensor_mul(a, b);
  PolyTensor *loss = poly_tensor_sum(prod, -1);

  PolyTensor *params[] = {a};
  PolyTensor **grads = poly_tensor_backward(ctx, loss, params, 1);
  ASSERT_NOT_NULL(grads);
  ASSERT_NOT_NULL(grads[0]);

  int rc = poly_tensor_realize(grads[0]);
  ASSERT_INT_EQ(rc, 0);
  for (int i = 0; i < 4; i++)
    ASSERT_FLOAT_EQ(grads[0]->data[i], b_data[i], 1e-5);

  poly_tensor_free(grads[0]);
  free(grads);
  poly_tensor_free(loss);
  poly_tensor_free(prod);
  poly_tensor_free(a);
  poly_tensor_free(b);
  poly_ctx_destroy(ctx);
  PASS();
}

/* ── M4: nn layers + SGD optimizer ───────────────────────────────────── */

TEST(nn, linear_forward) {
  /* Linear(2→3, no bias): out = x @ w^T
   * x = [[1,2]], w = [[1,0],[0,1],[1,1]]
   * out = [[1,2,3]] */
  PolyLinear *l = poly_nn_linear(2, 3, false);
  /* Override weights for deterministic test */
  l->weight->data[0] = 1; l->weight->data[1] = 0;  /* row 0 */
  l->weight->data[2] = 0; l->weight->data[3] = 1;  /* row 1 */
  l->weight->data[4] = 1; l->weight->data[5] = 1;  /* row 2 */

  float x_data[] = {1, 2};
  PolyCtx *ctx = poly_ctx_new();
  PolyTensor *x = poly_tensor_input(ctx, x_data, (int64_t[]){1, 2}, 2);
  PolyTensor *out = poly_nn_linear_forward(l, ctx, x);

  ASSERT_INT_EQ(out->ndim, 2);
  ASSERT_INT_EQ(out->shape[0], 1);
  ASSERT_INT_EQ(out->shape[1], 3);

  int rc = poly_tensor_realize(out);
  ASSERT_INT_EQ(rc, 0);
  ASSERT_FLOAT_EQ(out->data[0], 1.0f, 1e-5);  /* 1*1 + 2*0 */
  ASSERT_FLOAT_EQ(out->data[1], 2.0f, 1e-5);  /* 1*0 + 2*1 */
  ASSERT_FLOAT_EQ(out->data[2], 3.0f, 1e-5);  /* 1*1 + 2*1 */

  poly_tensor_free(out);
  poly_tensor_free(x);
  poly_ctx_destroy(ctx);
  poly_nn_linear_free(l);
  PASS();
}

TEST(nn, sgd_simple_step) {
  /* Simple: param w=[1,1], loss = sum(w*w) = 2, grad = [2,2]
   * After SGD step (lr=0.1): w = [1,1] - 0.1*[2,2] = [0.8, 0.8] */
  PolyTensor *w = poly_tensor_from_data((float[]){1.0f, 1.0f}, (int64_t[]){2}, 1);
  w->requires_grad = true;
  PolyTensor *params[] = {w};
  PolySGD sgd = poly_sgd_new(params, 1, 0.1f);

  PolyCtx *ctx = poly_ctx_new();
  PolyTensor *pw = poly_tensor_wrap(ctx, w);
  PolyTensor *sq = poly_tensor_mul(pw, pw);
  PolyTensor *loss = poly_tensor_sum(sq, -1);

  float loss_val = -1.0f;
  int rc = poly_sgd_step(&sgd, ctx, loss, &loss_val);
  ASSERT_INT_EQ(rc, 0);
  ASSERT_FLOAT_EQ(loss_val, 2.0f, 1e-5);

  /* Check param updated: w = [0.8, 0.8] */
  ASSERT_FLOAT_EQ(w->data[0], 0.8f, 1e-5);
  ASSERT_FLOAT_EQ(w->data[1], 0.8f, 1e-5);

  poly_tensor_free(loss);
  poly_tensor_free(sq);
  poly_tensor_free(pw);
  poly_ctx_destroy(ctx);
  free(sgd.params);
  poly_tensor_free(w);
  PASS();
}

TEST(nn, sgd_loss_decreases) {
  /* Two steps: verify loss decreases */
  PolyTensor *w = poly_tensor_from_data((float[]){2.0f, 3.0f}, (int64_t[]){2}, 1);
  w->requires_grad = true;
  PolyTensor *params[] = {w};
  PolySGD sgd = poly_sgd_new(params, 1, 0.1f);

  float loss1 = 0.0f, loss2 = 0.0f;

  /* Step 1 */
  PolyCtx *ctx1 = poly_ctx_new();
  PolyTensor *pw1 = poly_tensor_wrap(ctx1, w);
  PolyTensor *sq1 = poly_tensor_mul(pw1, pw1);
  PolyTensor *l1 = poly_tensor_sum(sq1, -1);
  ASSERT_INT_EQ(poly_sgd_step(&sgd, ctx1, l1, &loss1), 0);
  poly_tensor_free(l1); poly_tensor_free(sq1); poly_tensor_free(pw1);
  poly_ctx_destroy(ctx1);

  /* Step 2 */
  PolyCtx *ctx2 = poly_ctx_new();
  PolyTensor *pw2 = poly_tensor_wrap(ctx2, w);
  PolyTensor *sq2 = poly_tensor_mul(pw2, pw2);
  PolyTensor *l2 = poly_tensor_sum(sq2, -1);
  ASSERT_INT_EQ(poly_sgd_step(&sgd, ctx2, l2, &loss2), 0);
  poly_tensor_free(l2); poly_tensor_free(sq2); poly_tensor_free(pw2);
  poly_ctx_destroy(ctx2);

  ASSERT_TRUE(loss2 < loss1);

  free(sgd.params);
  poly_tensor_free(w);
  PASS();
}

/* ── M5: XOR training ────────────────────────────────────────────────── */

TEST(nn, xor_training) {
  poly_nn_seed(42);  /* reproducible initialization */
  poly_rangeify_stats_reset();
  float x_data[] = {0, 0, 0, 1, 1, 0, 1, 1};
  float y_data[] = {0, 1, 1, 0};

  /* Model: Linear(2→8) → relu → Linear(8→1) → sigmoid */
  PolyLinear *l1 = poly_nn_linear(2, 8, true);
  PolyLinear *l2 = poly_nn_linear(8, 1, true);

  /* Collect params */
  PolyTensor *params[4];
  int np = 0;
  np += poly_nn_linear_params(l1, params + np, 4 - np);
  np += poly_nn_linear_params(l2, params + np, 4 - np);
  PolySGD sgd = poly_sgd_new(params, np, 0.5f);

  float loss_val = 1.0f;
  for (int step = 0; step < 500 && loss_val > 0.01f; step++) {
    PolyCtx *ctx = poly_ctx_new();
    PolyTensor *x = poly_tensor_input(ctx, x_data, (int64_t[]){4, 2}, 2);
    PolyTensor *y = poly_tensor_input(ctx, y_data, (int64_t[]){4, 1}, 2);

    PolyTensor *h_lin = poly_nn_linear_forward(l1, ctx, x);
    PolyTensor *h = poly_tensor_relu(h_lin);
    PolyTensor *out_lin = poly_nn_linear_forward(l2, ctx, h);
    PolyTensor *out = poly_tensor_sigmoid(out_lin);
    PolyTensor *loss = poly_tensor_mse(out, y);

    int rc = poly_sgd_step(&sgd, ctx, loss, &loss_val);
    if (rc != 0) {
      poly_ctx_destroy(ctx);
      FAIL("sgd_step failed at step %d", step);
    }
    if (step < 10 || step % 50 == 0)
      fprintf(stderr, "  xor step %d: loss=%.6f\n", step, loss_val);

    /* Free intermediate wrappers (UOps are arena-managed, freed by ctx_destroy) */
    poly_tensor_free(loss);
    poly_tensor_free(out);
    poly_tensor_free(out_lin);
    poly_tensor_free(h);
    poly_tensor_free(h_lin);
    poly_tensor_free(x);
    poly_tensor_free(y);
    poly_ctx_destroy(ctx);
  }

  ASSERT_TRUE(loss_val < 0.05f);
  PolyRangeifyStats stats = poly_rangeify_stats_get();
  ASSERT_INT_EQ(stats.remap_unique_bound_matches, 0);
  ASSERT_INT_EQ(stats.remap_bound_matches, 0);

  free(sgd.params);
  poly_nn_linear_free(l1);
  poly_nn_linear_free(l2);
  PASS();
}

/* ── M6: Direct matmul numeric test ─────────────────────────────────── */
/* Regression test for accumulator placement in multi-range kernels.
 * (2,3) @ (3,4) = (2,4) — K=3, M*N=8, so K < M*N.
 * Each K contributes uniquely, so "last-K-only" would be obviously wrong.
 * A = [[1,2,3],[4,5,6]], B = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
 * Expected C = A@B = [[38,44,50,56],[83,98,113,128]] */
TEST(nn, matmul_numeric) {
  float a_data[] = {1, 2, 3, 4, 5, 6};
  float b_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  float expected[] = {38, 44, 50, 56, 83, 98, 113, 128};

  PolyCtx *ctx = poly_ctx_new();
  PolyTensor *a = poly_tensor_input(ctx, a_data, (int64_t[]){2, 3}, 2);
  PolyTensor *b = poly_tensor_input(ctx, b_data, (int64_t[]){3, 4}, 2);
  PolyTensor *c = poly_tensor_matmul(a, b);

  int rc = poly_tensor_realize(c);
  ASSERT_INT_EQ(rc, 0);
  ASSERT_NOT_NULL(c->data);

  for (int i = 0; i < 8; i++)
    ASSERT_FLOAT_EQ(c->data[i], expected[i], 1e-5);

  poly_tensor_free(c);
  poly_tensor_free(a);
  poly_tensor_free(b);
  poly_ctx_destroy(ctx);
  PASS();
}

/* ── Compiled Step tests ─────────────────────────────────────────────── */

TEST(step, compile_step_basic) {
  /* a[8] + 1.0 -> out[8]. Compile once, run twice with different data. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *buf_a   = poly_buffer_f32(ctx, 8);
  PolyUOp *buf_out  = poly_buffer_f32(ctx, 8);
  PolyUOp *one = poly_const_float(ctx, 1.0);
  PolyUOp *add = poly_alu2(ctx, POLY_OP_ADD, buf_a, one);
  PolyUOp *store = poly_store_val(ctx, buf_out, add);
  PolyUOp *sink = poly_sink1(ctx, store);

  PolyStep *step = poly_compile_step(ctx, sink);
  ASSERT_NOT_NULL(step);
  ASSERT_INT_EQ(poly_step_n_kernels(step), 1);
  ASSERT_INT_EQ(poly_step_n_intermediates(step), 0);

  /* Run 1 */
  float a_data[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };
  float out_data[8] = {0};
  PolyBufferBinding bindings[] = {
    { .buffer = buf_a, .data = a_data },
    { .buffer = buf_out, .data = out_data },
  };
  int ret = poly_step_run(step, bindings, 2);
  ASSERT_INT_EQ(ret, 0);
  for (int i = 0; i < 8; i++)
    ASSERT_FLOAT_EQ(out_data[i], (float)(i + 1) + 1.0f, 1e-5);

  /* Run 2: different data, same compiled step */
  for (int i = 0; i < 8; i++) a_data[i] = (float)(i + 10);
  memset(out_data, 0, sizeof(out_data));
  ret = poly_step_run(step, bindings, 2);
  ASSERT_INT_EQ(ret, 0);
  for (int i = 0; i < 8; i++)
    ASSERT_FLOAT_EQ(out_data[i], (float)(i + 10) + 1.0f, 1e-5);

  poly_step_destroy(step);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(step, compile_step_multikernel) {
  /* shared_scalar_reduce_branches: produces intermediates.
   * a[8] -> sum -> reshape(1) -> expand(8) -> {add(c0), mul(e0)} -> out1, out2
   * sum(1..8) = 36, out1 = 36+10 = 46, out2 = 36*2 = 72 */
  int N = 8;
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a   = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *c0  = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *e0  = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *oc  = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *oe  = poly_buffer(ctx, POLY_FLOAT32, N);

  int64_t axes[]   = {0};
  int64_t one_sh[] = {1};
  int64_t exp_sh[] = {N};

  PolyUOp *sum     = poly_reduce_axis(ctx, POLY_OP_ADD, a, axes, 1);
  PolyUOp *sum_1   = poly_reshape(ctx, sum, one_sh, 1);
  PolyUOp *sum_exp = poly_expand(ctx, sum_1, exp_sh, 1);

  PolyUOp *add_op = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, sum_exp, c0, poly_arg_none());
  PolyUOp *mul_op = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, sum_exp, e0, poly_arg_none());
  PolyUOp *sc  = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, oc, add_op, poly_arg_none());
  PolyUOp *se  = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, oe, mul_op, poly_arg_none());
  PolyUOp *sink = poly_uop(ctx, POLY_OP_SINK, POLY_VOID,
                            (PolyUOp *[]){sc, se}, 2, poly_arg_none());

  PolyStep *step = poly_compile_step(ctx, sink);
  ASSERT_NOT_NULL(step);
  ASSERT_TRUE(poly_step_n_kernels(step) >= 2);
  ASSERT_TRUE(poly_step_n_intermediates(step) >= 1);

  float a_d[8], c0_d[8], e0_d[8], oc_d[8], oe_d[8];
  for (int i = 0; i < N; i++) {
    a_d[i]  = (float)(i + 1);
    c0_d[i] = 10.0f;
    e0_d[i] = 2.0f;
    oc_d[i] = 0.0f;
    oe_d[i] = 0.0f;
  }

  PolyBufferBinding bindings[] = {
    { oc, oc_d }, { oe, oe_d },
    { a, a_d }, { c0, c0_d }, { e0, e0_d },
  };
  int ret = poly_step_run(step, bindings, 5);
  ASSERT_INT_EQ(ret, 0);

  for (int i = 0; i < N; i++) {
    ASSERT_FLOAT_EQ(oc_d[i], 46.0f, 1e-5);
    ASSERT_FLOAT_EQ(oe_d[i], 72.0f, 1e-5);
  }

  poly_step_destroy(step);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(step, compile_step_assign) {
  /* ASSIGN(a, a + 1.0). Run twice: a increments by 1 each time. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *buf_a = poly_buffer_f32(ctx, 4);
  PolyUOp *one = poly_const_float(ctx, 1.0);
  PolyUOp *add = poly_alu2(ctx, POLY_OP_ADD, buf_a, one);
  PolyUOp *assign = poly_assign(ctx, buf_a, add);
  PolyUOp *sink = poly_sink1(ctx, assign);

  PolyStep *step = poly_compile_step(ctx, sink);
  ASSERT_NOT_NULL(step);

  float a_data[4] = { 10.0f, 20.0f, 30.0f, 40.0f };
  PolyBufferBinding bindings[] = {
    { .buffer = buf_a, .data = a_data },
  };

  /* Run 1: a becomes a + 1 */
  int ret = poly_step_run(step, bindings, 1);
  ASSERT_INT_EQ(ret, 0);
  ASSERT_FLOAT_EQ(a_data[0], 11.0f, 1e-5);
  ASSERT_FLOAT_EQ(a_data[1], 21.0f, 1e-5);
  ASSERT_FLOAT_EQ(a_data[2], 31.0f, 1e-5);
  ASSERT_FLOAT_EQ(a_data[3], 41.0f, 1e-5);

  /* Run 2: a becomes a + 1 again (accumulates) */
  ret = poly_step_run(step, bindings, 1);
  ASSERT_INT_EQ(ret, 0);
  ASSERT_FLOAT_EQ(a_data[0], 12.0f, 1e-5);
  ASSERT_FLOAT_EQ(a_data[1], 22.0f, 1e-5);
  ASSERT_FLOAT_EQ(a_data[2], 32.0f, 1e-5);
  ASSERT_FLOAT_EQ(a_data[3], 42.0f, 1e-5);

  poly_step_destroy(step);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(step, compile_step_dynamic_default_override) {
  /* a[N] + 1.0 -> out[N] with BIND(N, 4).
   * Compile once. Run without var_bindings (uses default N=4).
   * Then run with var_bindings N=8 (override). */
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *N = poly_define_var(ctx, "N", 1, 16);
  PolyUOp *bind_N = poly_bind_var(ctx, N, 4);

  /* Build buffers with BIND as source (so strip_bind_values sees it) */
  PolyUOp *unique_a = poly_uop0(ctx, POLY_OP_UNIQUE, POLY_VOID, poly_arg_int(3000000));
  PolyUOp *unique_o = poly_uop0(ctx, POLY_OP_UNIQUE, POLY_VOID, poly_arg_int(3000001));
  PolyUOp *src_a[2] = { unique_a, bind_N };
  PolyUOp *src_o[2] = { unique_o, bind_N };
  PolyUOp *buf_a   = poly_uop(ctx, POLY_OP_BUFFER, POLY_FLOAT32, src_a, 2, poly_arg_int(16));
  PolyUOp *buf_out  = poly_uop(ctx, POLY_OP_BUFFER, POLY_FLOAT32, src_o, 2, poly_arg_int(16));

  PolyUOp *one = poly_const_float(ctx, 1.0);
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, buf_a, one, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, buf_out, add, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyStep *step = poly_compile_step(ctx, sink);
  ASSERT_NOT_NULL(step);

  float a_data[16], out_data[16];
  for (int i = 0; i < 16; i++) {
    a_data[i] = (float)(i + 1);
    out_data[i] = -999.0f;
  }

  PolyBufferBinding bindings[] = {
    { .buffer = buf_a, .data = a_data },
    { .buffer = buf_out, .data = out_data },
  };

  /* Run 1: no var_bindings, uses BIND default N=4 */
  int ret = poly_step_run(step, bindings, 2);
  ASSERT_INT_EQ(ret, 0);
  ASSERT_FLOAT_EQ(out_data[0], 2.0f, 1e-5);
  ASSERT_FLOAT_EQ(out_data[1], 3.0f, 1e-5);
  ASSERT_FLOAT_EQ(out_data[2], 4.0f, 1e-5);
  ASSERT_FLOAT_EQ(out_data[3], 5.0f, 1e-5);
  /* Element 4 untouched */
  ASSERT_FLOAT_EQ(out_data[4], -999.0f, 1e-5);

  /* Run 2: override N=8 */
  for (int i = 0; i < 16; i++) out_data[i] = -999.0f;
  PolyVarBinding var_bind = { .var = N, .value = 8 };
  ret = poly_step_run_ex(step, bindings, 2, &var_bind, 1);
  ASSERT_INT_EQ(ret, 0);
  for (int i = 0; i < 8; i++)
    ASSERT_FLOAT_EQ(out_data[i], (float)(i + 1) + 1.0f, 1e-5);
  /* Element 8 untouched */
  ASSERT_FLOAT_EQ(out_data[8], -999.0f, 1e-5);

  poly_step_destroy(step);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(step, compile_step_missing_binding) {
  /* Compile a step, call with incomplete buffer bindings -> returns -1. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *buf_a   = poly_buffer_f32(ctx, 4);
  PolyUOp *buf_out  = poly_buffer_f32(ctx, 4);
  PolyUOp *one = poly_const_float(ctx, 1.0);
  PolyUOp *add = poly_alu2(ctx, POLY_OP_ADD, buf_a, one);
  PolyUOp *store = poly_store_val(ctx, buf_out, add);
  PolyUOp *sink = poly_sink1(ctx, store);

  PolyStep *step = poly_compile_step(ctx, sink);
  ASSERT_NOT_NULL(step);

  /* Only provide buf_out, missing buf_a */
  float out_data[4] = {0};
  PolyBufferBinding bindings[] = {
    { .buffer = buf_out, .data = out_data },
  };
  int ret = poly_step_run(step, bindings, 1);
  ASSERT_INT_EQ(ret, -1);

  poly_step_destroy(step);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(step, compile_step_missing_var) {
  /* DEFINE_VAR without BIND. poly_step_run (no vars) -> -1.
   * poly_step_run_ex with var_bindings -> 0 (success). */
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *N = poly_define_var(ctx, "N", 1, 16);
  PolyUOp *buf_a   = poly_buffer_var(ctx, POLY_FLOAT32, N, NULL, 0);
  PolyUOp *buf_out  = poly_buffer_var(ctx, POLY_FLOAT32, N, NULL, 0);

  PolyUOp *one = poly_const_float(ctx, 1.0);
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, buf_a, one, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, buf_out, add, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyStep *step = poly_compile_step(ctx, sink);
  ASSERT_NOT_NULL(step);

  float a_data[16], out_data[16];
  for (int i = 0; i < 16; i++) {
    a_data[i] = (float)(i + 1);
    out_data[i] = -999.0f;
  }
  PolyBufferBinding bindings[] = {
    { .buffer = buf_a, .data = a_data },
    { .buffer = buf_out, .data = out_data },
  };

  /* Run without var_bindings -> should fail (no BIND default, var needed) */
  int ret = poly_step_run(step, bindings, 2);
  ASSERT_INT_EQ(ret, -1);

  /* Run with explicit var_bindings -> should succeed */
  for (int i = 0; i < 16; i++) out_data[i] = -999.0f;
  PolyVarBinding var_bind = { .var = N, .value = 4 };
  ret = poly_step_run_ex(step, bindings, 2, &var_bind, 1);
  ASSERT_INT_EQ(ret, 0);
  ASSERT_FLOAT_EQ(out_data[0], 2.0f, 1e-5);
  ASSERT_FLOAT_EQ(out_data[1], 3.0f, 1e-5);
  ASSERT_FLOAT_EQ(out_data[2], 4.0f, 1e-5);
  ASSERT_FLOAT_EQ(out_data[3], 5.0f, 1e-5);
  ASSERT_FLOAT_EQ(out_data[4], -999.0f, 1e-5);

  poly_step_destroy(step);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(step, compile_step_metadata) {
  /* Basic metadata queries. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *buf_a   = poly_buffer_f32(ctx, 4);
  PolyUOp *buf_out  = poly_buffer_f32(ctx, 4);
  PolyUOp *add = poly_alu2(ctx, POLY_OP_ADD, buf_a, poly_const_float(ctx, 1.0));
  PolyUOp *store = poly_store_val(ctx, buf_out, add);
  PolyUOp *sink = poly_sink1(ctx, store);

  PolyStep *step = poly_compile_step(ctx, sink);
  ASSERT_NOT_NULL(step);
  ASSERT_TRUE(poly_step_n_kernels(step) >= 1);
  ASSERT_INT_EQ(poly_step_n_intermediates(step), 0);

  /* NULL step returns 0 */
  ASSERT_INT_EQ(poly_step_n_kernels(NULL), 0);
  ASSERT_INT_EQ(poly_step_n_intermediates(NULL), 0);

  poly_step_destroy(step);
  poly_ctx_destroy(ctx);
  PASS();
}
