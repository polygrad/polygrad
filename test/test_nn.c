/*
 * test_nn.c — Tests for the Tensor + neural network API
 */

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "test_harness.h"
#include "../src/nn.h"
#include "../src/rangeify.h"
#include "../src/frontend.h"
#include "../src/codegen.h"
#include "../src/scheduler.h"

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

TEST(nn, matmul_invalid_shape_returns_null) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer_f32(ctx, 4);
  PolyUOp *b = poly_buffer_f32(ctx, 3);
  int64_t out_shape[POLY_MAX_DIMS];
  int out_ndim = -1;

  PolyUOp *r = poly_dot(ctx,
                        a, (int64_t[]){2, 2}, 2,
                        b, (int64_t[]){1, 3}, 2,
                        out_shape, &out_ndim);

  ASSERT_TRUE(r == NULL);
  ASSERT_INT_EQ(out_ndim, 0);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(nn, matmul_broadcast_batch_numeric) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer_f32(ctx, 8);
  PolyUOp *b = poly_buffer_f32(ctx, 4);
  int64_t out_shape[POLY_MAX_DIMS];
  int out_ndim = -1;

  PolyUOp *r = poly_dot(ctx,
                        a, (int64_t[]){2, 2, 2}, 3,
                        b, (int64_t[]){1, 2, 2}, 3,
                        out_shape, &out_ndim);
  ASSERT_NOT_NULL(r);
  ASSERT_INT_EQ(out_ndim, 3);
  ASSERT_INT_EQ(out_shape[0], 2);
  ASSERT_INT_EQ(out_shape[1], 2);
  ASSERT_INT_EQ(out_shape[2], 2);

  PolyUOp *out_buf = poly_buffer_f32(ctx, 8);
  PolyUOp *store = poly_store_val(ctx, out_buf, r);
  PolyUOp *sink = poly_sink1(ctx, store);

  float a_data[] = {1, 2, 3, 4, 5, 6, 7, 8};
  float b_data[] = {1, 10, 100, 1000};
  float out_data[8] = {0};
  float expected[] = {201, 2010, 403, 4030, 605, 6050, 807, 8070};
  PolyBufferBinding bindings[] = {
    POLY_BIND_HOST(a, a_data ),
    POLY_BIND_HOST(b, b_data ),
    POLY_BIND_HOST(out_buf, out_data ),
  };

  int ret = poly_realize(ctx, sink, bindings, 3);
  ASSERT_INT_EQ(ret, 0);
  for (int i = 0; i < 8; i++) ASSERT_FLOAT_EQ(out_data[i], expected[i], 1e-5f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(nn, matmul_invalid_broadcast_returns_null) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer_f32(ctx, 24);
  PolyUOp *b = poly_buffer_f32(ctx, 120);
  int64_t out_shape[POLY_MAX_DIMS];
  int out_ndim = -1;

  PolyUOp *r = poly_dot(ctx,
                        a, (int64_t[]){2, 3, 4}, 3,
                        b, (int64_t[]){5, 4, 6}, 3,
                        out_shape, &out_ndim);

  ASSERT_TRUE(r == NULL);
  ASSERT_INT_EQ(out_ndim, 0);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(nn, cross_entropy_sparse_targets) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *logits = poly_buffer_f32(ctx, 6);
  PolyUOp *target = poly_buffer_f32(ctx, 2);
  int64_t out_shape[POLY_MAX_DIMS];
  int out_ndim = -1;

  PolyUOp *loss = poly_cross_entropy(ctx,
                                     logits, (int64_t[]){2, 3}, 2,
                                     target, (int64_t[]){2}, 1,
                                     1, out_shape, &out_ndim);
  ASSERT_NOT_NULL(loss);
  ASSERT_INT_EQ(out_ndim, 0);

  PolyUOp *out_buf = poly_buffer_f32(ctx, 1);
  PolyUOp *store = poly_store_val(ctx, out_buf, loss);
  PolyUOp *sink = poly_sink1(ctx, store);

  float logits_data[] = {0, 0, 0, 0, 0, 0};
  float target_data[] = {0, 2};
  float out_data[] = {0};
  PolyBufferBinding bindings[] = {
    POLY_BIND_HOST(logits, logits_data ),
    POLY_BIND_HOST(target, target_data ),
    POLY_BIND_HOST(out_buf, out_data ),
  };

  int ret = poly_realize(ctx, sink, bindings, 3);
  ASSERT_INT_EQ(ret, 0);
  ASSERT_FLOAT_EQ(out_data[0], logf(3.0f), 1e-5f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(nn, cross_entropy_dense_targets) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *logits = poly_buffer_f32(ctx, 6);
  PolyUOp *target = poly_buffer_f32(ctx, 6);
  int64_t out_shape[POLY_MAX_DIMS];
  int out_ndim = -1;

  PolyUOp *loss = poly_cross_entropy(ctx,
                                     logits, (int64_t[]){2, 3}, 2,
                                     target, (int64_t[]){2, 3}, 2,
                                     1, out_shape, &out_ndim);
  ASSERT_NOT_NULL(loss);
  ASSERT_INT_EQ(out_ndim, 0);

  PolyUOp *out_buf = poly_buffer_f32(ctx, 1);
  PolyUOp *store = poly_store_val(ctx, out_buf, loss);
  PolyUOp *sink = poly_sink1(ctx, store);

  float logits_data[] = {0, 0, 0, 0, 0, 0};
  float target_data[] = {1, 0, 0, 0, 0, 1};
  float out_data[] = {0};
  PolyBufferBinding bindings[] = {
    POLY_BIND_HOST(logits, logits_data ),
    POLY_BIND_HOST(target, target_data ),
    POLY_BIND_HOST(out_buf, out_data ),
  };

  int ret = poly_realize(ctx, sink, bindings, 3);
  ASSERT_INT_EQ(ret, 0);
  ASSERT_FLOAT_EQ(out_data[0], logf(3.0f), 1e-5f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(nn, cross_entropy_sparse_targets_non_last_axis) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *logits = poly_buffer_f32(ctx, 12);
  PolyUOp *target = poly_buffer_f32(ctx, 4);
  int64_t out_shape[POLY_MAX_DIMS];
  int out_ndim = -1;

  PolyUOp *loss = poly_cross_entropy(ctx,
                                     logits, (int64_t[]){2, 3, 2}, 3,
                                     target, (int64_t[]){2, 2}, 2,
                                     -2, out_shape, &out_ndim);
  ASSERT_NOT_NULL(loss);
  ASSERT_INT_EQ(out_ndim, 0);

  PolyUOp *out_buf = poly_buffer_f32(ctx, 1);
  PolyUOp *store = poly_store_val(ctx, out_buf, loss);
  PolyUOp *sink = poly_sink1(ctx, store);

  float logits_data[] = {
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0
  };
  float target_data[] = {0, 2, 1, 0};
  float out_data[] = {0};
  PolyBufferBinding bindings[] = {
    POLY_BIND_HOST(logits, logits_data ),
    POLY_BIND_HOST(target, target_data ),
    POLY_BIND_HOST(out_buf, out_data ),
  };

  int ret = poly_realize(ctx, sink, bindings, 3);
  ASSERT_INT_EQ(ret, 0);
  ASSERT_FLOAT_EQ(out_data[0], logf(3.0f), 1e-5f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(nn, cross_entropy_invalid_shape_returns_null) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *logits = poly_buffer_f32(ctx, 6);
  PolyUOp *target = poly_buffer_f32(ctx, 4);
  int64_t out_shape[POLY_MAX_DIMS];
  int out_ndim = -1;

  PolyUOp *loss = poly_cross_entropy(ctx,
                                     logits, (int64_t[]){2, 3}, 2,
                                     target, (int64_t[]){2, 2}, 2,
                                     1, out_shape, &out_ndim);

  ASSERT_TRUE(loss == NULL);
  ASSERT_INT_EQ(out_ndim, 0);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(nn, log_softmax_non_last_axis_flat_buffer) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_buffer_f32(ctx, 12);

  PolyUOp *y = poly_log_softmax(ctx, x, (int64_t[]){2, 3, 2}, 3, 1);
  ASSERT_NOT_NULL(y);

  PolyUOp *out_buf = poly_buffer_f32(ctx, 12);
  PolyUOp *store = poly_store_val(ctx, out_buf, y);
  PolyUOp *sink = poly_sink1(ctx, store);

  float x_data[] = {
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0
  };
  float out_data[12] = {0};
  PolyBufferBinding bindings[] = {
    POLY_BIND_HOST(x, x_data ),
    POLY_BIND_HOST(out_buf, out_data ),
  };

  int ret = poly_realize(ctx, sink, bindings, 2);
  ASSERT_INT_EQ(ret, 0);
  for (int i = 0; i < 12; i++) ASSERT_FLOAT_EQ(out_data[i], -logf(3.0f), 1e-5f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(nn, layernorm_non_last_axis_flat_buffer) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_buffer_f32(ctx, 12);
  int64_t out_shape[POLY_MAX_DIMS];
  int out_ndim = -1;

  PolyUOp *y = poly_layernorm(ctx, x, (int64_t[]){2, 3, 2}, 3, 1, 1e-5,
                              out_shape, &out_ndim);
  ASSERT_NOT_NULL(y);
  ASSERT_INT_EQ(out_ndim, 3);
  ASSERT_INT_EQ(out_shape[0], 2);
  ASSERT_INT_EQ(out_shape[1], 3);
  ASSERT_INT_EQ(out_shape[2], 2);

  PolyUOp *out_buf = poly_buffer_f32(ctx, 12);
  PolyUOp *store = poly_store_val(ctx, out_buf, y);
  PolyUOp *sink = poly_sink1(ctx, store);

  float x_data[] = {
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0
  };
  float out_data[12] = {0};
  PolyBufferBinding bindings[] = {
    POLY_BIND_HOST(x, x_data ),
    POLY_BIND_HOST(out_buf, out_data ),
  };

  int ret = poly_realize(ctx, sink, bindings, 2);
  ASSERT_INT_EQ(ret, 0);
  for (int i = 0; i < 12; i++) ASSERT_FLOAT_EQ(out_data[i], 0.0f, 1e-5f);

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
    POLY_BIND_HOST(buf_a, a_data ),
    POLY_BIND_HOST(buf_out, out_data ),
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
    POLY_BIND_HOST(oc, oc_d), POLY_BIND_HOST(oe, oe_d),
    POLY_BIND_HOST(a, a_d), POLY_BIND_HOST(c0, c0_d), POLY_BIND_HOST(e0, e0_d),
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
    POLY_BIND_HOST(buf_a, a_data ),
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
    POLY_BIND_HOST(buf_a, a_data ),
    POLY_BIND_HOST(buf_out, out_data ),
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
    POLY_BIND_HOST(buf_out, out_data ),
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
    POLY_BIND_HOST(buf_a, a_data ),
    POLY_BIND_HOST(buf_out, out_data ),
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

TEST(step, compile_value_and_grad_quadratic) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *p = poly_buffer_f32(ctx, 4);
  PolyUOp *sq = poly_alu2(ctx, POLY_OP_MUL, p, p);
  int64_t axes[] = {0};
  PolyUOp *loss = poly_reduce_axis(ctx, POLY_OP_ADD, sq, axes, 1);

  PolyUOp *params[] = { p };
  int loss_idx = -1;
  int grad_idxs[1] = { -1 };
  PolyStep *step = poly_compile_value_and_grad(ctx, loss, params, 1, &loss_idx, grad_idxs);
  ASSERT_NOT_NULL(step);

  int n_bufs = poly_step_n_buffers(step);
  ASSERT_TRUE(n_bufs >= 3); /* loss output, grad output, param input */
  ASSERT_TRUE(loss_idx >= 0 && loss_idx < n_bufs);
  ASSERT_TRUE(grad_idxs[0] >= 0 && grad_idxs[0] < n_bufs);
  ASSERT_TRUE(loss_idx != grad_idxs[0]);

  PolyStepBufferInfo info;
  ASSERT_INT_EQ(poly_step_buffer_info(step, loss_idx, &info), 0);
  ASSERT_INT_EQ(info.role, POLY_STEP_BUF_OUTPUT);
  ASSERT_INT_EQ(info.numel, 1);
  ASSERT_TRUE(info.nbytes > 0);

  ASSERT_INT_EQ(poly_step_buffer_info(step, grad_idxs[0], &info), 0);
  ASSERT_INT_EQ(info.role, POLY_STEP_BUF_OUTPUT);
  ASSERT_INT_EQ(info.numel, 4);
  ASSERT_TRUE(info.nbytes > 0);

  int param_idx = -1;
  for (int i = 0; i < n_bufs; i++) {
    ASSERT_INT_EQ(poly_step_buffer_info(step, i, &info), 0);
    if (info.role == POLY_STEP_BUF_INPUT && info.numel == 4) {
      param_idx = i;
      break;
    }
  }
  ASSERT_TRUE(param_idx >= 0);
  ASSERT_INT_EQ(poly_step_buffer_info(step, n_bufs, &info), -1);
  ASSERT_INT_EQ(poly_step_buffer_info(NULL, 0, &info), -1);

  float p_data[4] = { 1.0f, 2.0f, -3.0f, 4.0f };
  float loss_out[1] = {0};
  float grad_out[4] = {0};
  void *buf_ptrs[16] = {0};
  ASSERT_TRUE(n_bufs < 16);
  buf_ptrs[loss_idx] = loss_out;
  buf_ptrs[grad_idxs[0]] = grad_out;
  buf_ptrs[param_idx] = p_data;

  int ret = poly_step_run_indexed(step, buf_ptrs, n_bufs);
  ASSERT_INT_EQ(ret, 0);
  ASSERT_FLOAT_EQ(loss_out[0], 30.0f, 1e-5);
  ASSERT_FLOAT_EQ(grad_out[0], 2.0f, 1e-5);
  ASSERT_FLOAT_EQ(grad_out[1], 4.0f, 1e-5);
  ASSERT_FLOAT_EQ(grad_out[2], -6.0f, 1e-5);
  ASSERT_FLOAT_EQ(grad_out[3], 8.0f, 1e-5);

  /* Deterministic replay for fixed input. */
  float loss_out_2[1] = {0};
  float grad_out_2[4] = {0};
  buf_ptrs[loss_idx] = loss_out_2;
  buf_ptrs[grad_idxs[0]] = grad_out_2;
  ret = poly_step_run_indexed(step, buf_ptrs, n_bufs);
  ASSERT_INT_EQ(ret, 0);
  ASSERT_FLOAT_EQ(loss_out_2[0], loss_out[0], 1e-6);
  for (int i = 0; i < 4; i++)
    ASSERT_FLOAT_EQ(grad_out_2[i], grad_out[i], 1e-6);

  poly_step_destroy(step);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(step, constant_buffers_autobind_rand) {
  PolyCtx *ctx = poly_ctx_new();
  int64_t shape[1] = {16};
  PolyUOp *rnd = poly_rand(ctx, shape, 1, 1337u);
  ASSERT_NOT_NULL(rnd);
  PolyUOp *out = poly_buffer_f32(ctx, 16);
  PolyUOp *store = poly_store_val(ctx, out, rnd);
  PolyUOp *sink = poly_sink1(ctx, store);
  PolyStep *step = poly_compile_step(ctx, sink);
  ASSERT_NOT_NULL(step);

  int nbuf = poly_step_n_buffers(step);
  int out_idx = -1;
  int const_count = 0;
  PolyStepBufferInfo bi;
  for (int i = 0; i < nbuf; i++) {
    ASSERT_INT_EQ(poly_step_buffer_info(step, i, &bi), 0);
    if (bi.role == POLY_STEP_BUF_OUTPUT && bi.numel == 16) out_idx = i;
    if (bi.role == POLY_STEP_BUF_CONSTANT) const_count++;
  }
  ASSERT_TRUE(out_idx >= 0);
  ASSERT_TRUE(const_count > 0);

  float out1[16] = {0}, out2[16] = {0};
  void *bufs[64] = {0};
  ASSERT_TRUE(nbuf < 64);
  bufs[out_idx] = out1;
  ASSERT_INT_EQ(poly_step_run_indexed(step, bufs, nbuf), 0);
  bufs[out_idx] = out2;
  ASSERT_INT_EQ(poly_step_run_indexed(step, bufs, nbuf), 0);

  for (int i = 0; i < 16; i++) {
    ASSERT_TRUE(out1[i] >= 0.0f && out1[i] < 1.0f);
    ASSERT_FLOAT_EQ(out1[i], out2[i], 1e-7);
  }

  poly_step_destroy(step);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(nn, threefry_reference_vector_cpu) {
  /* Reference from JAX threefry2x32 primitive:
   * key0=1337, key1=0, x0=counters 0..19, x1=0.
   * This matches tinygrad's uint32 elementwise lowering semantics. */
  const uint32_t ref[20] = {
    2732499619u, 3322027265u, 2482432314u, 3871860445u, 3571867126u,
    3019569655u, 2459680734u, 2731866067u, 986922480u, 1616040745u,
    4238711754u, 3594775990u, 3046419939u, 3519108299u, 586160567u,
    3928687287u, 4074505382u, 4210472430u, 4094881238u, 3306411770u
  };

  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *counter = poly_buffer(ctx, POLY_UINT32, 20);
  PolyUOp *key = poly_buffer(ctx, POLY_UINT32, 20);
  PolyUOp *out = poly_buffer(ctx, POLY_UINT32, 20);
  PolyUOp *thr = poly_uop2(ctx, POLY_OP_THREEFRY, POLY_UINT32, counter, key, poly_arg_none());
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, thr));

  uint32_t counter_data[20], key_data[20], out_data[20];
  for (int i = 0; i < 20; i++) {
    counter_data[i] = (uint32_t)i;
    key_data[i] = 1337u;
    out_data[i] = 0u;
  }
  PolyBufferBinding binds[3] = {
    POLY_BIND_HOST(counter, counter_data), POLY_BIND_HOST(key, key_data), POLY_BIND_HOST(out, out_data)
  };
  ASSERT_INT_EQ(poly_realize(ctx, sink, binds, 3), 0);
  for (int i = 0; i < 20; i++)
    ASSERT_INT_EQ((int)out_data[i], (int)ref[i]);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(nn, threefry_reference_vector_uint64_cpu) {
  /* Reference from JAX threefry2x32 primitive with packed uint64 lanes:
   * x = (x1<<32)|x0 with x0=0..19, x1=0
   * key = (key1<<32)|key0 with key0=1337, key1=0 */
  const uint64_t ref[20] = {
    2091981003842036387ull, 7289724007706926337ull, 5665610599518103866ull,
    1710967476831381213ull, 10223482319493553654ull, 17028600460329286135ull,
    17124603982541209566ull, 3105577853879908307ull, 2086892522412654064ull,
    3335679299718140713ull, 16051258882154404810ull, 6689751810128997814ull,
    14143017499898457571ull, 15433975895207007435ull, 585627789352245687ull,
    6751685475094430391ull, 13535415641566872742ull, 11824248585908041198ull,
    12094218887010381270ull, 17901838715724224250ull
  };

  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *counter = poly_buffer(ctx, POLY_UINT64, 20);
  PolyUOp *key = poly_buffer(ctx, POLY_UINT64, 20);
  PolyUOp *out = poly_buffer(ctx, POLY_UINT64, 20);
  PolyUOp *thr = poly_uop2(ctx, POLY_OP_THREEFRY, POLY_UINT64, counter, key, poly_arg_none());
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, thr));

  uint64_t counter_data[20], key_data[20], out_data[20];
  for (int i = 0; i < 20; i++) {
    counter_data[i] = (uint64_t)i;
    key_data[i] = 1337ull;
    out_data[i] = 0ull;
  }
  PolyBufferBinding binds[3] = {
    POLY_BIND_HOST(counter, counter_data), POLY_BIND_HOST(key, key_data), POLY_BIND_HOST(out, out_data)
  };
  ASSERT_INT_EQ(poly_realize(ctx, sink, binds, 3), 0);
  for (int i = 0; i < 20; i++) {
    if (out_data[i] != ref[i]) {
      FAIL("out_data[%d]=%llu expected %llu", i,
           (unsigned long long)out_data[i], (unsigned long long)ref[i]);
    }
  }

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(nn, frontend_randn_stats_and_determinism) {
  PolyCtx *ctx = poly_ctx_new();
  int64_t shape[1] = {2048};
  PolyUOp *rn = poly_randn(ctx, shape, 1, 2026u);
  ASSERT_NOT_NULL(rn);
  PolyUOp *out = poly_buffer_f32(ctx, 2048);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, rn));

  float *a = calloc(2048, sizeof(float));
  float *b = calloc(2048, sizeof(float));
  ASSERT_NOT_NULL(a);
  ASSERT_NOT_NULL(b);
  PolyBufferBinding bind = POLY_BIND_HOST(out, a);
  ASSERT_INT_EQ(poly_realize(ctx, sink, &bind, 1), 0);
  bind.handle.ptr = b;
  ASSERT_INT_EQ(poly_realize(ctx, sink, &bind, 1), 0);

  double mean = 0.0, var = 0.0;
  for (int i = 0; i < 2048; i++) {
    mean += a[i];
    ASSERT_FLOAT_EQ(a[i], b[i], 1e-7); /* deterministic replay */
  }
  mean /= 2048.0;
  for (int i = 0; i < 2048; i++) {
    double d = (double)a[i] - mean;
    var += d * d;
  }
  var /= 2048.0;
  ASSERT_TRUE(fabs(mean) < 0.1);
  ASSERT_TRUE(fabs(var - 1.0) < 0.2);

  free(a);
  free(b);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(nn, frontend_creation_helpers) {
  PolyCtx *ctx = poly_ctx_new();

  /* arange(0,5,1) sum = 10 */
  PolyUOp *ar = poly_arange(ctx, 0.0, 5.0, 1.0);
  ASSERT_NOT_NULL(ar);
  int64_t a_shape[1] = {5}; int a_ndim = 1;
  PolyUOp *ar_sum = poly_sum_reduce(ctx, ar, a_shape, a_ndim, 0, 0, a_shape, &a_ndim);
  PolyUOp *buf0 = poly_buffer_f32(ctx, 1);

  float ar_out[1] = {0};
  PolyBufferBinding b0 = POLY_BIND_HOST(buf0, ar_out);
  ASSERT_INT_EQ(poly_realize(ctx, poly_sink1(ctx, poly_store_val(ctx, buf0, ar_sum)), &b0, 1), 0);
  ASSERT_FLOAT_EQ(ar_out[0], 10.0f, 1e-5);

  /* eye(3) has trace/sum 3 */
  PolyUOp *eye = poly_eye(ctx, 3);
  ASSERT_NOT_NULL(eye);
  int64_t e_shape[2] = {3, 3};
  int64_t s1[8]; int nd1 = 0;
  PolyUOp *r1 = poly_sum_reduce(ctx, eye, e_shape, 2, 1, 1, s1, &nd1);
  int64_t s2[8]; int nd2 = 0;
  PolyUOp *r2 = poly_sum_reduce(ctx, r1, s1, nd1, 0, 0, s2, &nd2);
  PolyUOp *buf1 = poly_buffer_f32(ctx, 1);
  float eye_out[1] = {0};
  PolyBufferBinding b1 = POLY_BIND_HOST(buf1, eye_out);
  ASSERT_INT_EQ(poly_realize(ctx, poly_sink1(ctx, poly_store_val(ctx, buf1, r2)), &b1, 1), 0);
  ASSERT_FLOAT_EQ(eye_out[0], 3.0f, 1e-5);

  /* tril/triu on ones(3,3): both sums are 6 */
  PolyUOp *ones = poly_full(ctx, e_shape, 2, 1.0);
  ASSERT_NOT_NULL(ones);
  PolyUOp *tl = poly_tril(ctx, ones, e_shape, 2, 0);
  PolyUOp *tu = poly_triu(ctx, ones, e_shape, 2, 0);
  ASSERT_NOT_NULL(tl);
  ASSERT_NOT_NULL(tu);
  int64_t ts1[8], ts2[8];
  int tnd1 = 0, tnd2 = 0;
  PolyUOp *tl_s = poly_sum_reduce(ctx, poly_sum_reduce(ctx, tl, e_shape, 2, 1, 1, ts1, &tnd1), ts1, tnd1, 0, 0, ts2, &tnd2);
  int64_t us1[8], us2[8];
  int und1 = 0, und2 = 0;
  PolyUOp *tu_s = poly_sum_reduce(ctx, poly_sum_reduce(ctx, tu, e_shape, 2, 1, 1, us1, &und1), us1, und1, 0, 0, us2, &und2);
  PolyUOp *buf2 = poly_buffer_f32(ctx, 1);
  PolyUOp *buf3 = poly_buffer_f32(ctx, 1);
  float tri_out[2] = {0, 0};
  PolyUOp *sink = poly_sink_n(ctx, (PolyUOp *[]){
    poly_store_val(ctx, buf2, tl_s),
    poly_store_val(ctx, buf3, tu_s)
  }, 2);
  PolyBufferBinding binds[2] = { POLY_BIND_HOST(buf2, &tri_out[0]), POLY_BIND_HOST(buf3, &tri_out[1]) };
  ASSERT_INT_EQ(poly_realize(ctx, sink, binds, 2), 0);
  ASSERT_FLOAT_EQ(tri_out[0], 6.0f, 1e-5);
  ASSERT_FLOAT_EQ(tri_out[1], 6.0f, 1e-5);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(nn, frontend_math_wrappers_and_lgamma_grad) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_buffer_f32(ctx, 1);

  /* Forward checks: log1p/expm1 */
  PolyUOp *y1 = poly_log1p(ctx, x);
  PolyUOp *y2 = poly_expm1(ctx, x);
  PolyUOp *o1 = poly_buffer_f32(ctx, 1);
  PolyUOp *o2 = poly_buffer_f32(ctx, 1);
  PolyUOp *sink = poly_sink_n(ctx, (PolyUOp *[]){
    poly_store_val(ctx, o1, y1),
    poly_store_val(ctx, o2, y2),
  }, 2);
  float xv[1] = {0.2f}, out1[1] = {0}, out2[1] = {0};
  PolyBufferBinding binds[3] = { POLY_BIND_HOST(x, xv), POLY_BIND_HOST(o1, out1), POLY_BIND_HOST(o2, out2) };
  ASSERT_INT_EQ(poly_realize(ctx, sink, binds, 3), 0);
  ASSERT_FLOAT_EQ(out1[0], log1pf(0.2f), 5e-3f);
  ASSERT_FLOAT_EQ(out2[0], expm1f(0.2f), 5e-3f);

  /* lgamma gradient override check via compile_value_and_grad */
  PolyUOp *lg = poly_lgamma(ctx, x);
  int64_t axes[] = {0};
  PolyUOp *loss = poly_reduce_axis(ctx, POLY_OP_ADD, lg, axes, 1);
  PolyUOp *params[] = { x };
  int loss_idx = -1, grad_idx[1] = {-1};
  PolyStep *step = poly_compile_value_and_grad(ctx, loss, params, 1, &loss_idx, grad_idx);
  ASSERT_NOT_NULL(step);

  int nbuf = poly_step_n_buffers(step);
  int x_idx = -1;
  PolyStepBufferInfo info;
  for (int i = 0; i < nbuf; i++) {
    ASSERT_INT_EQ(poly_step_buffer_info(step, i, &info), 0);
    if (info.role == POLY_STEP_BUF_INPUT && info.numel == 1) x_idx = i;
  }
  ASSERT_TRUE(x_idx >= 0);

  float xin[1] = {3.2f}, lout[1] = {0}, gout[1] = {0};
  void *bufs[16] = {0};
  ASSERT_TRUE(nbuf < 16);
  bufs[x_idx] = xin;
  bufs[loss_idx] = lout;
  bufs[grad_idx[0]] = gout;
  ASSERT_INT_EQ(poly_step_run_indexed(step, bufs, nbuf), 0);
  ASSERT_FLOAT_EQ(lout[0], (float)lgamma(3.2), 5e-3f);
  double h = 1e-4;
  double fd = (lgamma(3.2 + h) - lgamma(3.2 - h)) / (2.0 * h);
  ASSERT_FLOAT_EQ(gout[0], (float)fd, 5e-2f);

  poly_step_destroy(step);
  poly_ctx_destroy(ctx);
  PASS();
}

/* ── Special math correctness tests (baked reference constants) ────────── */

/* Helper: compile a scalar f32 function, evaluate at given input, return output */
static float eval_scalar_f32(PolyUOp *(*fn)(PolyCtx *, PolyUOp *), float xval) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_buffer_f32(ctx, 1);
  PolyUOp *y = fn(ctx, x);
  PolyUOp *out = poly_buffer_f32(ctx, 1);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, y));
  float inv = xval, result = 0;
  PolyBufferBinding binds[] = { POLY_BIND_HOST(x, &inv), POLY_BIND_HOST(out, &result) };
  int rc = poly_realize(ctx, sink, binds, 2);
  poly_ctx_destroy(ctx);
  return (rc == 0) ? result : NAN;
}

/* Helper: compile a scalar f64 function, evaluate at given input, return output */
static double eval_scalar_f64(PolyUOp *(*fn)(PolyCtx *, PolyUOp *), double xval) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_buffer_f64(ctx, 1);
  PolyUOp *y = fn(ctx, x);
  PolyUOp *out = poly_buffer_f64(ctx, 1);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, y));
  double inv = xval, result = 0.0;
  PolyBufferBinding binds[] = { POLY_BIND_HOST(x, &inv), POLY_BIND_HOST(out, &result) };
  int rc = poly_realize(ctx, sink, binds, 2);
  poly_ctx_destroy(ctx);
  return (rc == 0) ? result : (double)NAN;
}

TEST(nn, special_math_erf) {
  /* Reference: A&S Table 7.1, DLMF 7.2.1 */
  struct { float x; float ref; } cases[] = {
    { 0.0f,   0.0f },
    { 0.5f,   0.5204998778f },
    { 1.0f,   0.8427007929f },
    { 2.0f,   0.9953222650f },
    { 3.0f,   0.9999779095f },
    {-1.0f,  -0.8427007929f },
  };
  for (int i = 0; i < 6; i++) {
    float got = eval_scalar_f32(poly_erf, cases[i].x);
    ASSERT_FLOAT_EQ(got, cases[i].ref, 5e-4f);
  }
  PASS();
}

TEST(nn, special_math_erfc) {
  /* Reference: erfc(x) = 1 - erf(x). Tail values from DLMF 7.2.2.
   * Uses A&S tau directly (no 1-erf cancellation). */
  struct { float x; float ref; float tol; } cases[] = {
    { 0.0f,   1.0f,             1e-4f },
    { 1.0f,   0.15729920706f,   1e-4f },
    { 2.0f,   0.00467773498f,   1e-5f },
    { 3.0f,   2.20904970e-5f,   5e-6f },
    { 5.0f,   1.53745979e-12f,  1e-12f },
    /* Negative: erfc(-x) = 2 - erfc(x) */
    {-1.0f,   1.84270079294f,   1e-4f },
  };
  for (int i = 0; i < 6; i++) {
    float got = eval_scalar_f32(poly_erfc, cases[i].x);
    ASSERT_TRUE(got >= 0.0f);
    /* For very small tail values, check relative error instead */
    if (cases[i].ref < 1e-4f && cases[i].ref > 0.0f) {
      float rel_err = fabsf(got - cases[i].ref) / cases[i].ref;
      ASSERT_TRUE(rel_err < 0.5f);  /* within 50% relative error for f32 tails */
    } else {
      ASSERT_FLOAT_EQ(got, cases[i].ref, cases[i].tol);
    }
  }
  PASS();
}

TEST(nn, special_math_erfinv) {
  /* Reference: erfinv values from Wolfram Alpha */
  struct { float x; float ref; } cases[] = {
    {  0.0f,     0.0f },
    {  0.5f,     0.4769362762f },
    { -0.5f,    -0.4769362762f },
    {  0.9f,     1.1630871536f },
    {  0.99f,    1.8213863677f },
    {  0.9999f,  2.7510639058f },
  };
  for (int i = 0; i < 6; i++) {
    float got = eval_scalar_f32(poly_erfinv, cases[i].x);
    /* Winitzki approximation: ~0.35% max relative error */
    float tol = 0.01f + 0.005f * fabsf(cases[i].ref);
    ASSERT_FLOAT_EQ(got, cases[i].ref, tol);
  }

  /* Roundtrip: erfinv(erf(x)) ~= x */
  float roundtrip_x[] = { -2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f };
  for (int i = 0; i < 7; i++) {
    float v = roundtrip_x[i];
    float erf_v = eval_scalar_f32(poly_erf, v);
    float back = eval_scalar_f32(poly_erfinv, erf_v);
    ASSERT_FLOAT_EQ(back, v, 0.05f);
  }
  PASS();
}

TEST(nn, special_math_ndtri) {
  /* Reference: inverse normal CDF from standard tables / Wolfram Alpha */
  struct { float p; float ref; float tol; } cases[] = {
    { 0.5f,    0.0f,          1e-4f },
    { 0.75f,   0.6744897502f, 0.01f },
    { 0.9f,    1.2815515655f, 0.02f },
    { 0.99f,   2.3263478740f, 0.05f },
    /* Tail tests -- PPL-critical */
    { 1e-3f,  -3.0902323062f, 0.1f  },
    { 1e-6f,  -4.7534243060f, 0.2f  },
  };
  for (int i = 0; i < 6; i++) {
    float got = eval_scalar_f32(poly_ndtri, cases[i].p);
    ASSERT_FLOAT_EQ(got, cases[i].ref, cases[i].tol);
  }
  /* Symmetry: ndtri(1-p) = -ndtri(p) */
  float p_sym = 0.9f;
  float v1 = eval_scalar_f32(poly_ndtri, p_sym);
  float v2 = eval_scalar_f32(poly_ndtri, 1.0f - p_sym);
  ASSERT_FLOAT_EQ(v1, -v2, 0.02f);
  PASS();
}

TEST(nn, special_math_digamma) {
  /* Reference: DLMF 5.4.14, A&S Table 6.3 */
  struct { float x; float ref; } cases[] = {
    { 1.0f,  -0.5772156649f },   /* Euler-Mascheroni */
    { 2.0f,   0.4227843351f },
    { 0.5f,  -1.9635100260f },
    { 5.0f,   1.5061176685f },
    { 10.0f,  2.2517525890f },
  };
  for (int i = 0; i < 5; i++) {
    float got = eval_scalar_f32(poly_digamma, cases[i].x);
    ASSERT_FLOAT_EQ(got, cases[i].ref, 5e-3f);
  }
  PASS();
}

TEST(nn, special_math_lgamma) {
  /* Reference: lgamma values from standard tables.
   * All computation is f32 (cf() emits POLY_FLOAT32), so expect ~1e-3 accuracy. */
  struct { float x; float ref; float tol; } cases[] = {
    { 1.0f,   0.0f,          5e-4f },
    { 2.0f,   0.0f,          5e-4f },
    { 0.5f,   0.5723649429f, 1e-3f },   /* ln(sqrt(pi)) */
    { 3.5f,   1.2009736024f, 1e-3f },
    { 5.0f,   3.1780538303f, 5e-3f },
    { 10.0f, 12.8018274801f, 0.02f },
  };
  for (int i = 0; i < 6; i++) {
    float got = eval_scalar_f32(poly_lgamma, cases[i].x);
    ASSERT_FLOAT_EQ(got, cases[i].ref, cases[i].tol);
  }
  PASS();
}

TEST(nn, special_math_log1p_expm1) {
  /* Near-zero tests: the whole point of log1p/expm1 */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_buffer_f32(ctx, 1);
  PolyUOp *y1 = poly_log1p(ctx, x);
  PolyUOp *y2 = poly_expm1(ctx, x);
  PolyUOp *o1 = poly_buffer_f32(ctx, 1);
  PolyUOp *o2 = poly_buffer_f32(ctx, 1);
  PolyUOp *sink = poly_sink_n(ctx, (PolyUOp *[]){
    poly_store_val(ctx, o1, y1),
    poly_store_val(ctx, o2, y2),
  }, 2);

  struct { float x; float ref_log1p; float ref_expm1; } cases[] = {
    { 0.0f,     0.0f,           0.0f },
    { 1.0f,     0.6931471806f,  1.7182818285f },
    { 0.2f,     0.1823215568f,  0.2214027582f },
    { 1e-6f,    1e-6f,          1e-6f },   /* near-zero: should NOT be 0 */
  };
  for (int i = 0; i < 4; i++) {
    float xv = cases[i].x, out1 = 0, out2 = 0;
    PolyBufferBinding binds[] = { POLY_BIND_HOST(x, &xv), POLY_BIND_HOST(o1, &out1), POLY_BIND_HOST(o2, &out2) };
    ASSERT_INT_EQ(poly_realize(ctx, sink, binds, 3), 0);
    ASSERT_FLOAT_EQ(out1, cases[i].ref_log1p, 5e-4f);
    ASSERT_FLOAT_EQ(out2, cases[i].ref_expm1, 5e-4f);
    /* Critical: near-zero must not be zero */
    if (cases[i].x == 1e-6f) {
      ASSERT_TRUE(out1 > 0.0f);
      ASSERT_TRUE(out2 > 0.0f);
    }
  }
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(nn, special_math_logsumexp) {
  /* logsumexp([a, b]) = max(a,b) + log(exp(a-max) + exp(b-max)) */
  PolyCtx *ctx = poly_ctx_new();

  /* Case 1: logsumexp([0, 0]) = ln(2) */
  {
    PolyUOp *x = poly_buffer_f32(ctx, 2);
    int64_t shape[] = {2};
    int64_t out_shape[1]; int out_ndim = 0;
    PolyUOp *y = poly_logsumexp(ctx, x, shape, 1, 0, 0, out_shape, &out_ndim);
    PolyUOp *out = poly_buffer_f32(ctx, 1);
    PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, y));
    float xv[] = {0.0f, 0.0f}, result = 0;
    PolyBufferBinding binds[] = { POLY_BIND_HOST(x, xv), POLY_BIND_HOST(out, &result) };
    ASSERT_INT_EQ(poly_realize(ctx, sink, binds, 2), 0);
    ASSERT_FLOAT_EQ(result, 0.6931471806f, 1e-4f);  /* ln(2) */
  }

  /* Case 2: overflow stability -- logsumexp([1000, 1001]) */
  {
    PolyUOp *x2 = poly_buffer_f32(ctx, 2);
    int64_t shape2[] = {2};
    int64_t out_shape2[1]; int out_ndim2 = 0;
    PolyUOp *y2 = poly_logsumexp(ctx, x2, shape2, 1, 0, 0, out_shape2, &out_ndim2);
    PolyUOp *out2 = poly_buffer_f32(ctx, 1);
    PolyUOp *sink2 = poly_sink1(ctx, poly_store_val(ctx, out2, y2));
    float xv2[] = {1000.0f, 1001.0f}, result2 = 0;
    PolyBufferBinding binds2[] = { POLY_BIND_HOST(x2, xv2), POLY_BIND_HOST(out2, &result2) };
    ASSERT_INT_EQ(poly_realize(ctx, sink2, binds2, 2), 0);
    /* Expected: 1001 + ln(1 + e^-1) = 1001.3133 */
    ASSERT_TRUE(isfinite(result2));
    ASSERT_FLOAT_EQ(result2, 1001.3133f, 0.01f);
  }

  /* Case 3: underflow stability -- logsumexp([-1000, -999]) */
  {
    PolyUOp *x3 = poly_buffer_f32(ctx, 2);
    int64_t shape3[] = {2};
    int64_t out_shape3[1]; int out_ndim3 = 0;
    PolyUOp *y3 = poly_logsumexp(ctx, x3, shape3, 1, 0, 0, out_shape3, &out_ndim3);
    PolyUOp *out3 = poly_buffer_f32(ctx, 1);
    PolyUOp *sink3 = poly_sink1(ctx, poly_store_val(ctx, out3, y3));
    float xv3[] = {-1000.0f, -999.0f}, result3 = 0;
    PolyBufferBinding binds3[] = { POLY_BIND_HOST(x3, xv3), POLY_BIND_HOST(out3, &result3) };
    ASSERT_INT_EQ(poly_realize(ctx, sink3, binds3, 2), 0);
    /* Expected: -999 + ln(1 + e^-1) = -998.6867 */
    ASSERT_TRUE(isfinite(result3));
    ASSERT_FLOAT_EQ(result3, -998.6867f, 0.01f);
  }

  poly_ctx_destroy(ctx);
  PASS();
}

/* ── f64 accuracy tests — verify dtype-correct constants ─────────────── */

TEST(nn, special_math_f64_lgamma) {
  /* lgamma() is ISO C99 <math.h>; use it as reference for f64 accuracy. */
  struct { double x; double tol; } cases[] = {
    { 1.0,  1e-12 },   /* lgamma(1) = 0 exactly */
    { 2.0,  1e-12 },   /* lgamma(2) = 0 exactly */
    { 0.5,  1e-12 },   /* ln(sqrt(pi)) */
    { 1.5,  1e-12 },
    { 10.0, 1e-10 },
  };
  for (int i = 0; i < 5; i++) {
    double got = eval_scalar_f64(poly_lgamma, cases[i].x);
    double ref = lgamma(cases[i].x);
    double err = fabs(got - ref);
    ASSERT_TRUE(err < cases[i].tol);
  }
  PASS();
}

TEST(nn, special_math_f64_digamma) {
  /* digamma() is not in ISO C — use baked high-precision constants.
   * ψ(1) = -γ, ψ(2) = 1 - γ (via ψ(n+1) = ψ(n) + 1/n),
   * ψ(5) = precomputed.
   * Tolerance note: the algorithm uses 6 recurrence steps + 4-term Bernoulli
   * asymptotic at x=6, giving ~2-3e-9 truncation error.  Tolerances are set
   * to 1e-7 to stay well above this while still requiring f64 precision
   * (f32 gives ~1e-5 error due to transcendental polynomial approximation). */
  struct { double x; double ref; double tol; } cases[] = {
    { 1.0, -0.5772156649015328606, 1e-7 },  /* Euler-Mascheroni */
    { 2.0,  0.4227843350984671394, 1e-7 },  /* 1 - γ */
    { 5.0,  1.5061176684318004727, 1e-7 },  /* precomputed */
  };
  for (int i = 0; i < 3; i++) {
    double got = eval_scalar_f64(poly_digamma, cases[i].x);
    double err = fabs(got - cases[i].ref);
    ASSERT_TRUE(err < cases[i].tol);
  }
  /* Recurrence check: ψ(x+1) = ψ(x) + 1/x */
  double psi15 = eval_scalar_f64(poly_digamma, 1.5);
  double psi25 = eval_scalar_f64(poly_digamma, 2.5);
  ASSERT_TRUE(fabs(psi25 - (psi15 + 1.0/1.5)) < 1e-10);
  PASS();
}

TEST(nn, special_math_f64_log1p) {
  /* log1p() is ISO C99 <math.h> */
  struct { double x; double tol; } cases[] = {
    { 1e-10, 1e-18 },   /* tiny x: f32 loses most digits */
    { 1.0,   1e-14 },   /* ln(2) */
    { 0.2,   1e-13 },
  };
  for (int i = 0; i < 3; i++) {
    double got = eval_scalar_f64(poly_log1p, cases[i].x);
    double ref = log1p(cases[i].x);
    double err = fabs(got - ref);
    ASSERT_TRUE(err < cases[i].tol);
  }
  PASS();
}

TEST(nn, special_math_f64_expm1) {
  /* expm1() is ISO C99 <math.h> */
  struct { double x; double tol; } cases[] = {
    { 1e-10, 1e-18 },   /* tiny x: f32 loses most digits */
    { 1.0,   1e-14 },   /* e - 1 */
    { 0.2,   1e-13 },
  };
  for (int i = 0; i < 3; i++) {
    double got = eval_scalar_f64(poly_expm1, cases[i].x);
    double ref = expm1(cases[i].x);
    double err = fabs(got - ref);
    ASSERT_TRUE(err < cases[i].tol);
  }
  PASS();
}

/* ── C5 primitive tests ───────────────────────────────────────────────── */

TEST(nn, c5_detach_stops_grad) {
  /* detach(x) passes forward value but blocks gradient */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_buffer_f32(ctx, 1);
  /* loss = x^2 + detach(x)^2
   * d/dx = 2x + 0 = 2x (detach kills the second term's gradient) */
  PolyUOp *sq = poly_alu2(ctx, POLY_OP_MUL, x, x);
  PolyUOp *dx = poly_detach(ctx, x);
  PolyUOp *sq_det = poly_alu2(ctx, POLY_OP_MUL, dx, dx);
  PolyUOp *sum = poly_alu2(ctx, POLY_OP_ADD, sq, sq_det);
  int64_t axes[] = {0};
  PolyUOp *loss = poly_reduce_axis(ctx, POLY_OP_ADD, sum, axes, 1);

  int loss_idx = -1, grad_idx[1] = {-1};
  PolyStep *step = poly_compile_value_and_grad(ctx, loss, (PolyUOp *[]){x}, 1, &loss_idx, grad_idx);
  ASSERT_NOT_NULL(step);

  int nbuf = poly_step_n_buffers(step);
  int x_idx = -1;
  PolyStepBufferInfo info;
  for (int i = 0; i < nbuf; i++) {
    poly_step_buffer_info(step, i, &info);
    if (info.role == POLY_STEP_BUF_INPUT && info.numel == 1) x_idx = i;
  }
  ASSERT_TRUE(x_idx >= 0);

  float xv = 3.0f, lout = 0, gout = 0;
  void *bufs[16] = {0};
  bufs[x_idx] = &xv;
  bufs[loss_idx] = &lout;
  bufs[grad_idx[0]] = &gout;
  ASSERT_INT_EQ(poly_step_run_indexed(step, bufs, nbuf), 0);
  /* Forward: 3^2 + 3^2 = 18 */
  ASSERT_FLOAT_EQ(lout, 18.0f, 1e-3f);
  /* Gradient: 2*3 = 6 (NOT 2*3 + 2*3 = 12) */
  ASSERT_FLOAT_EQ(gout, 6.0f, 1e-3f);

  poly_step_destroy(step);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(nn, c5_linspace) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *ls = poly_linspace(ctx, 0.0, 1.0, 5);
  ASSERT_NOT_NULL(ls);
  PolyUOp *out = poly_buffer_f32(ctx, 5);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, ls));
  float result[5] = {0};
  PolyBufferBinding binds[] = { POLY_BIND_HOST(out, result) };
  ASSERT_INT_EQ(poly_realize(ctx, sink, binds, 1), 0);
  float expected[] = {0.0f, 0.25f, 0.5f, 0.75f, 1.0f};
  for (int i = 0; i < 5; i++) {
    ASSERT_FLOAT_EQ(result[i], expected[i], 1e-6f);
  }
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(nn, c5_full) {
  PolyCtx *ctx = poly_ctx_new();
  int64_t shape[] = {4};
  PolyUOp *f = poly_full(ctx, shape, 1, 3.14);
  ASSERT_NOT_NULL(f);
  PolyUOp *out = poly_buffer_f32(ctx, 4);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, f));
  float result[4] = {0};
  PolyBufferBinding binds[] = { POLY_BIND_HOST(out, result) };
  ASSERT_INT_EQ(poly_realize(ctx, sink, binds, 1), 0);
  for (int i = 0; i < 4; i++) {
    ASSERT_FLOAT_EQ(result[i], 3.14f, 1e-5f);
  }
  poly_ctx_destroy(ctx);
  PASS();
}

/* ── C2c: RNG determinism + cross-backend parity ─────────────────────── */

TEST(nn, c2c_rand_bitpattern_8) {
  /* Tier 1 bit-exact: full poly_rand pipeline (THREEFRY -> SHR 8 -> CAST f32 -> MUL 2^-24).
   * seed=1337 -> key_lo=1337, key_hi=0, mixed_key=1337.
   * counter=[0..7], THREEFRY outputs match threefry_reference_vector_cpu. */
  const uint32_t thr_ref[8] = {
    2732499619u, 3322027265u, 2482432314u, 3871860445u,
    3571867126u, 3019569655u, 2459680734u, 2731866067u
  };
  float expected[8];
  for (int i = 0; i < 8; i++)
    expected[i] = (float)(thr_ref[i] >> 8) * (1.0f / 16777216.0f);

  PolyCtx *ctx = poly_ctx_new();
  int64_t shape[1] = {8};
  PolyUOp *r = poly_rand(ctx, shape, 1, 1337u);
  ASSERT_NOT_NULL(r);
  PolyUOp *out = poly_buffer_f32(ctx, 8);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, r));
  float result[8] = {0};
  PolyBufferBinding bind = POLY_BIND_HOST(out, result);
  ASSERT_INT_EQ(poly_realize(ctx, sink, &bind, 1), 0);
  /* Bit-exact comparison via memcmp -- true Tier 1 contract */
  if (memcmp(result, expected, sizeof(expected)) != 0) {
    for (int i = 0; i < 8; i++) {
      if (result[i] != expected[i])
        fprintf(stderr, "  [%d] got %.9g expected %.9g\n", i, result[i], expected[i]);
    }
    FAIL("poly_rand bitpattern mismatch (memcmp)");
  }
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(nn, c2c_rand_seed_mixing) {
  /* Verify different seeds produce different outputs, and hi/lo 32-bit
   * differences are not lost by the xor/rotate mixing. */
  int64_t shape[1] = {4};
  float r0[4], r1[4], rhi[4], rboth[4];
  float *all[4] = {r0, r1, rhi, rboth};
  uint64_t seeds[4] = {0, 1, 0x100000000ull, 0x100000001ull};

  PolyCtx *ctx = poly_ctx_new();
  for (int s = 0; s < 4; s++) {
    PolyUOp *t = poly_rand(ctx, shape, 1, seeds[s]);
    ASSERT_NOT_NULL(t);
    PolyUOp *o = poly_buffer_f32(ctx, 4);
    PolyUOp *sk = poly_sink1(ctx, poly_store_val(ctx, o, t));
    PolyBufferBinding b = POLY_BIND_HOST(o, all[s]);
    ASSERT_INT_EQ(poly_realize(ctx, sk, &b, 1), 0);
  }
  /* All 4 seeds must produce pairwise-different outputs */
  for (int i = 0; i < 4; i++) {
    for (int j = i + 1; j < 4; j++) {
      if (memcmp(all[i], all[j], 4 * sizeof(float)) == 0) {
        FAIL("seed %llu and %llu produced identical output",
             (unsigned long long)seeds[i], (unsigned long long)seeds[j]);
      }
    }
  }
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(nn, c2c_rand_range_and_stats) {
  /* Tier 2 statistical: uniform [0,1) with reasonable mean/variance. */
  PolyCtx *ctx = poly_ctx_new();
  int64_t shape[1] = {4096};
  PolyUOp *r = poly_rand(ctx, shape, 1, 42u);
  ASSERT_NOT_NULL(r);
  PolyUOp *out = poly_buffer_f32(ctx, 4096);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, r));
  float *buf = calloc(4096, sizeof(float));
  PolyBufferBinding bind = POLY_BIND_HOST(out, buf);
  ASSERT_INT_EQ(poly_realize(ctx, sink, &bind, 1), 0);

  double sum = 0;
  float vmin = buf[0], vmax = buf[0];
  int count_exact_one = 0;
  for (int i = 0; i < 4096; i++) {
    ASSERT_TRUE(buf[i] >= 0.0f);
    ASSERT_TRUE(buf[i] < 1.0f);
    if (buf[i] == 1.0f) count_exact_one++;
    if (buf[i] < vmin) vmin = buf[i];
    if (buf[i] > vmax) vmax = buf[i];
    sum += buf[i];
  }
  double mean = sum / 4096.0;
  ASSERT_TRUE(fabs(mean - 0.5) < 0.05);
  ASSERT_TRUE(vmin != vmax); /* not degenerate */
  ASSERT_INT_EQ(count_exact_one, 0); /* 1.0f should never appear with top-24-bit mapping */

  double var = 0;
  for (int i = 0; i < 4096; i++) {
    double d = buf[i] - mean;
    var += d * d;
  }
  var /= 4096.0;
  ASSERT_TRUE(fabs(var - 1.0 / 12.0) < 0.02);

  free(buf);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(nn, c2c_randn_tails) {
  /* Tier 2: Gaussian tails, no inf/NaN from Box-Muller. */
  PolyCtx *ctx = poly_ctx_new();
  int64_t shape[1] = {8192};
  PolyUOp *rn = poly_randn(ctx, shape, 1, 99u);
  ASSERT_NOT_NULL(rn);
  PolyUOp *out = poly_buffer_f32(ctx, 8192);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, rn));
  float *buf = calloc(8192, sizeof(float));
  PolyBufferBinding bind = POLY_BIND_HOST(out, buf);
  ASSERT_INT_EQ(poly_realize(ctx, sink, &bind, 1), 0);

  double mean = 0;
  double max_abs = 0;
  for (int i = 0; i < 8192; i++) {
    mean += buf[i];
    double a = fabs((double)buf[i]);
    if (a > max_abs) max_abs = a;
  }
  mean /= 8192.0;

  double var = 0;
  for (int i = 0; i < 8192; i++) {
    double d = buf[i] - mean;
    var += d * d;
  }
  var /= 8192.0;
  double sd = sqrt(var);

  int beyond_3sigma = 0;
  for (int i = 0; i < 8192; i++) {
    if (fabs(buf[i] - mean) > 3.0 * sd) beyond_3sigma++;
  }

  ASSERT_TRUE(fabs(mean) < 0.1);
  ASSERT_TRUE(fabs(var - 1.0) < 0.2);
  ASSERT_TRUE(beyond_3sigma < 80); /* expected ~22 for Gaussian */
  ASSERT_TRUE(max_abs < 8.0); /* catches inf/NaN or repeated-zero bugs */

  free(buf);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(nn, c2c_rand_determinism) {
  /* Same seed must produce identical output: (1) same graph replayed,
   * (2) two separately built graphs with same seed. */
  int64_t shape[1] = {32};

  /* Case 1: same graph, two realize calls */
  PolyCtx *ctx1 = poly_ctx_new();
  PolyUOp *r1 = poly_rand(ctx1, shape, 1, 12345u);
  PolyUOp *o1 = poly_buffer_f32(ctx1, 32);
  PolyUOp *s1 = poly_sink1(ctx1, poly_store_val(ctx1, o1, r1));
  float a[32] = {0}, b[32] = {0};
  PolyBufferBinding bind1 = POLY_BIND_HOST(o1, a);
  ASSERT_INT_EQ(poly_realize(ctx1, s1, &bind1, 1), 0);
  bind1.handle.ptr = b;
  ASSERT_INT_EQ(poly_realize(ctx1, s1, &bind1, 1), 0);
  if (memcmp(a, b, sizeof(a)) != 0)
    FAIL("same graph replay not deterministic");

  /* Case 2: two separately built graphs, different contexts */
  PolyCtx *ctx2 = poly_ctx_new();
  PolyUOp *r2 = poly_rand(ctx2, shape, 1, 12345u);
  PolyUOp *o2 = poly_buffer_f32(ctx2, 32);
  PolyUOp *s2 = poly_sink1(ctx2, poly_store_val(ctx2, o2, r2));
  float c[32] = {0};
  PolyBufferBinding bind2 = POLY_BIND_HOST(o2, c);
  ASSERT_INT_EQ(poly_realize(ctx2, s2, &bind2, 1), 0);
  if (memcmp(a, c, sizeof(a)) != 0)
    FAIL("separate graphs with same seed not deterministic");

  poly_ctx_destroy(ctx1);
  poly_ctx_destroy(ctx2);
  PASS();
}

TEST(nn, c2c_threefry_lowered_in_compiled_kernel) {
  /* Verify THREEFRY is fully decomposed in the real compile path (not just
   * isolated rewrite helpers). Build a THREEFRY kernel, run full_rewrite_to_sink
   * with has_threefry=false, then scan the linearized output for residual ops. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *counter = poly_buffer(ctx, POLY_UINT32, 8);
  PolyUOp *key = poly_buffer(ctx, POLY_UINT32, 8);
  PolyUOp *out = poly_buffer(ctx, POLY_UINT32, 8);
  PolyUOp *thr = poly_uop2(ctx, POLY_OP_THREEFRY, POLY_UINT32, counter, key, poly_arg_none());
  PolyUOp *store = poly_store_val(ctx, out, thr);
  PolyUOp *sink = poly_sink1(ctx, store);

  /* Schedule + rewrite with has_threefry=false */
  PolyScheduleResult sr = poly_schedule_v2(ctx, sink);
  ASSERT_TRUE(sr.n_kernels >= 1);
  PolyRewriteOpts opts = {0};
  opts.caps.has_threefry = false;
  opts.caps.has_mulacc = false;
  PolyUOp *rewritten = poly_full_rewrite_to_sink_ex(ctx, sr.kernels[0], opts);
  ASSERT_NOT_NULL(rewritten);

  /* Linearize and scan for residual THREEFRY ops */
  int n_uops = 0;
  PolyUOp **uops = poly_linearize(ctx, rewritten, &n_uops);
  ASSERT_NOT_NULL(uops);
  ASSERT_TRUE(n_uops > 0);
  int residual_threefry = 0;
  for (int i = 0; i < n_uops; i++) {
    if (uops[i]->op == POLY_OP_THREEFRY) residual_threefry++;
  }
  if (residual_threefry > 0)
    FAIL("found %d residual THREEFRY ops after full_rewrite_to_sink", residual_threefry);

  /* Also verify it actually executes on CPU */
  uint32_t counter_data[8], key_data[8], out_data[8];
  for (int i = 0; i < 8; i++) {
    counter_data[i] = (uint32_t)i;
    key_data[i] = 42u;
    out_data[i] = 0u;
  }
  PolyBufferBinding binds[3] = {
    POLY_BIND_HOST(counter, counter_data), POLY_BIND_HOST(key, key_data), POLY_BIND_HOST(out, out_data)
  };
  ASSERT_INT_EQ(poly_realize(ctx, sink, binds, 3), 0);
  /* Verify we got non-zero output (THREEFRY actually ran) */
  int any_nonzero = 0;
  for (int i = 0; i < 8; i++) {
    if (out_data[i] != 0) any_nonzero = 1;
  }
  ASSERT_TRUE(any_nonzero);

  free(uops);
  poly_schedule_result_free(&sr);
  poly_ctx_destroy(ctx);
  PASS();
}

/* ── C5: Primitive edge-case tests ───────────────────────────────────── */

TEST(nn, c5_arange_negative_step) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *ar = poly_arange(ctx, 5.0, 0.0, -1.0);
  ASSERT_NOT_NULL(ar);
  PolyUOp *out = poly_buffer_f32(ctx, 5);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, ar));
  float result[5] = {0};
  PolyBufferBinding bind = POLY_BIND_HOST(out, result);
  ASSERT_INT_EQ(poly_realize(ctx, sink, &bind, 1), 0);
  float expected[] = {5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
  for (int i = 0; i < 5; i++)
    ASSERT_FLOAT_EQ(result[i], expected[i], 1e-6f);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(nn, c5_arange_empty_and_zero_step) {
  PolyCtx *ctx = poly_ctx_new();
  /* start > stop with positive step -> zero-length buffer, not NULL */
  PolyUOp *ar = poly_arange(ctx, 5.0, 3.0, 1.0);
  ASSERT_NOT_NULL(ar);
  /* step=0 -> NULL (error) */
  PolyUOp *bad = poly_arange(ctx, 0.0, 5.0, 0.0);
  ASSERT_TRUE(bad == NULL);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(nn, c5_arange_fractional_step) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *ar = poly_arange(ctx, 0.0, 1.0, 0.3);
  ASSERT_NOT_NULL(ar);
  /* ceil((1.0-0.0)/0.3 - 1e-12) = ceil(3.333... - eps) = 4 */
  PolyUOp *out = poly_buffer_f32(ctx, 4);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, ar));
  float result[4] = {0};
  PolyBufferBinding bind = POLY_BIND_HOST(out, result);
  ASSERT_INT_EQ(poly_realize(ctx, sink, &bind, 1), 0);
  float expected[] = {0.0f, 0.3f, 0.6f, 0.9f};
  for (int i = 0; i < 4; i++)
    ASSERT_FLOAT_EQ(result[i], expected[i], 1e-6f);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(nn, c5_eye_edge_cases) {
  PolyCtx *ctx = poly_ctx_new();
  /* eye(1) -> [[1.0]] */
  PolyUOp *e1 = poly_eye(ctx, 1);
  ASSERT_NOT_NULL(e1);
  PolyUOp *o1 = poly_buffer_f32(ctx, 1);
  float r1[1] = {0};
  PolyBufferBinding b1 = POLY_BIND_HOST(o1, r1);
  ASSERT_INT_EQ(poly_realize(ctx, poly_sink1(ctx, poly_store_val(ctx, o1, e1)), &b1, 1), 0);
  ASSERT_FLOAT_EQ(r1[0], 1.0f, 1e-7f);
  /* eye(0) -> non-NULL zero-length buffer */
  PolyUOp *e0 = poly_eye(ctx, 0);
  ASSERT_NOT_NULL(e0);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(nn, c5_tril_triu_diagonal_offset) {
  PolyCtx *ctx = poly_ctx_new();
  int64_t shape[2] = {3, 3};
  PolyUOp *ones = poly_full(ctx, shape, 2, 1.0);
  ASSERT_NOT_NULL(ones);

  /* tril(diagonal=1): keeps main diagonal + 1 superdiagonal
   * [[1,1,0],[1,1,1],[1,1,1]] -> sum=8 */
  PolyUOp *tl1 = poly_tril(ctx, ones, shape, 2, 1);
  ASSERT_NOT_NULL(tl1);
  int64_t s1[8], s2[8]; int nd1 = 0, nd2 = 0;
  PolyUOp *sum_tl = poly_sum_reduce(ctx,
    poly_sum_reduce(ctx, tl1, shape, 2, 1, 1, s1, &nd1),
    s1, nd1, 0, 0, s2, &nd2);
  PolyUOp *buf = poly_buffer_f32(ctx, 1);
  float out[1] = {0};
  PolyBufferBinding bind = POLY_BIND_HOST(buf, out);
  ASSERT_INT_EQ(poly_realize(ctx, poly_sink1(ctx, poly_store_val(ctx, buf, sum_tl)), &bind, 1), 0);
  ASSERT_FLOAT_EQ(out[0], 8.0f, 1e-5f);

  /* triu(diagonal=-1): keeps main diagonal + 1 subdiagonal
   * [[1,1,1],[1,1,1],[0,1,1]] -> sum=8 */
  PolyUOp *tu1 = poly_triu(ctx, ones, shape, 2, -1);
  ASSERT_NOT_NULL(tu1);
  int64_t s3[8], s4[8]; int nd3 = 0, nd4 = 0;
  PolyUOp *sum_tu = poly_sum_reduce(ctx,
    poly_sum_reduce(ctx, tu1, shape, 2, 1, 1, s3, &nd3),
    s3, nd3, 0, 0, s4, &nd4);
  PolyUOp *buf2 = poly_buffer_f32(ctx, 1);
  float out2[1] = {0};
  PolyBufferBinding bind2 = POLY_BIND_HOST(buf2, out2);
  ASSERT_INT_EQ(poly_realize(ctx, poly_sink1(ctx, poly_store_val(ctx, buf2, sum_tu)), &bind2, 1), 0);
  ASSERT_FLOAT_EQ(out2[0], 8.0f, 1e-5f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(nn, c5_tril_zeros_check) {
  /* Verify masked-out elements are exactly zero, not just correct sum. */
  PolyCtx *ctx = poly_ctx_new();
  int64_t shape[2] = {3, 3};
  PolyUOp *ones = poly_full(ctx, shape, 2, 1.0);
  PolyUOp *tl = poly_tril(ctx, ones, shape, 2, 0);
  ASSERT_NOT_NULL(tl);
  PolyUOp *out = poly_buffer_f32(ctx, 9);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, tl));
  float result[9] = {0};
  PolyBufferBinding bind = POLY_BIND_HOST(out, result);
  ASSERT_INT_EQ(poly_realize(ctx, sink, &bind, 1), 0);
  /* Row-major: [[1,0,0],[1,1,0],[1,1,1]] */
  float expected[9] = {1,0,0, 1,1,0, 1,1,1};
  for (int i = 0; i < 9; i++)
    ASSERT_FLOAT_EQ(result[i], expected[i], 1e-7f);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(nn, c5_linspace_edge_cases) {
  PolyCtx *ctx = poly_ctx_new();

  /* Single point: linspace(3,7,1) -> {3.0} (start) */
  PolyUOp *ls1 = poly_linspace(ctx, 3.0, 7.0, 1);
  ASSERT_NOT_NULL(ls1);
  PolyUOp *o1 = poly_buffer_f32(ctx, 1);
  float r1[1] = {0};
  PolyBufferBinding b1 = POLY_BIND_HOST(o1, r1);
  ASSERT_INT_EQ(poly_realize(ctx, poly_sink1(ctx, poly_store_val(ctx, o1, ls1)), &b1, 1), 0);
  ASSERT_FLOAT_EQ(r1[0], 3.0f, 1e-7f);

  /* start == stop: linspace(5,5,3) -> {5,5,5} */
  PolyUOp *ls2 = poly_linspace(ctx, 5.0, 5.0, 3);
  ASSERT_NOT_NULL(ls2);
  PolyUOp *o2 = poly_buffer_f32(ctx, 3);
  float r2[3] = {0};
  PolyBufferBinding b2 = POLY_BIND_HOST(o2, r2);
  ASSERT_INT_EQ(poly_realize(ctx, poly_sink1(ctx, poly_store_val(ctx, o2, ls2)), &b2, 1), 0);
  for (int i = 0; i < 3; i++)
    ASSERT_FLOAT_EQ(r2[i], 5.0f, 1e-7f);

  /* steps=0 -> non-NULL zero-length buffer */
  PolyUOp *ls0 = poly_linspace(ctx, 0.0, 1.0, 0);
  ASSERT_NOT_NULL(ls0);

  poly_ctx_destroy(ctx);
  PASS();
}

/* ── ABI stability ───────────────────────────────────────────────────── */

TEST(nn, c5_abi_step_buffer_info_layout) {
  /* Verify sizeof, field offsets, and alignment of PolyStepBufferInfo
   * so FFI consumers (Python ctypes, JS koffi) don't silently read garbage. */
  ASSERT_INT_EQ(POLY_STEP_BUFFER_INFO_VERSION, 1);

  /* version at offset 0, index immediately after */
  ASSERT_INT_EQ((int)offsetof(PolyStepBufferInfo, version), 0);
  ASSERT_INT_EQ((int)offsetof(PolyStepBufferInfo, index), (int)sizeof(int));

  /* Tight sizeof bounds: int*3 + PolyDType + int64_t*2, plus up to 16 bytes padding */
  size_t min_sz = sizeof(int) * 3 + sizeof(PolyDType) + sizeof(int64_t) * 2;
  ASSERT_TRUE(sizeof(PolyStepBufferInfo) >= min_sz);
  ASSERT_TRUE(sizeof(PolyStepBufferInfo) <= min_sz + 16);

  /* Alignment: dtype and numel fields must be naturally aligned */
  ASSERT_INT_EQ((int)(offsetof(PolyStepBufferInfo, dtype) % _Alignof(PolyDType)), 0);
  ASSERT_INT_EQ((int)(offsetof(PolyStepBufferInfo, numel) % _Alignof(int64_t)), 0);

  PASS();
}

TEST(wasm, step_plan_basic) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer_f32(ctx, 8);
  PolyUOp *out = poly_buffer_f32(ctx, 8);
  PolyUOp *v = poly_alu2(ctx, POLY_OP_ADD, a, poly_const_float(ctx, 1.0));
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, v));

  PolyWasmStepPlan *p = poly_render_step_wasm_plan(ctx, sink);
  ASSERT_NOT_NULL(p);
  int nk = poly_wasm_stepplan_n_kernels(p);
  ASSERT_TRUE(nk >= 1);
  int n_total = poly_wasm_stepplan_n_buffers(p);
  int n_bindable = poly_wasm_stepplan_n_bindable_buffers(p);
  ASSERT_TRUE(n_total >= n_bindable);
  ASSERT_TRUE(n_bindable >= 2);
  for (int k = 0; k < nk; k++) {
    int len = 0;
    const uint8_t *bytes = poly_wasm_stepplan_kernel_bytes(p, k, &len);
    ASSERT_NOT_NULL(bytes);
    ASSERT_TRUE(len > 8);
    int np = poly_wasm_stepplan_kernel_n_params(p, k);
    ASSERT_TRUE(np >= 1);
    for (int i = 0; i < np; i++) {
      int bi = poly_wasm_stepplan_kernel_param_buf_index(p, k, i);
      ASSERT_TRUE(bi >= 0 && bi < n_total);
    }
  }
  int n_exec = 0;
  const int *order = poly_wasm_stepplan_exec_order(p, &n_exec);
  ASSERT_NOT_NULL(order);
  ASSERT_INT_EQ(n_exec, nk);

  poly_wasm_stepplan_destroy(p);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(wasm, step_plan_rand_threefry_decomposed) {
  /* Verify poly_rand kernel renders to valid WASM (THREEFRY fully decomposed).
   * WASM renderer has has_threefry=false, so any residual THREEFRY op would
   * cause a render failure. */
  PolyCtx *ctx = poly_ctx_new();
  int64_t shape[] = {16};
  PolyUOp *r = poly_rand(ctx, shape, 1, 42);
  PolyUOp *out = poly_buffer_f32(ctx, 16);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, r));

  PolyWasmStepPlan *p = poly_render_step_wasm_plan(ctx, sink);
  ASSERT_NOT_NULL(p);  /* render success = no residual THREEFRY */
  int nk = poly_wasm_stepplan_n_kernels(p);
  ASSERT_TRUE(nk >= 1);
  /* Verify each kernel has valid WASM (magic header \0asm) */
  for (int k = 0; k < nk; k++) {
    int len = 0;
    const uint8_t *bytes = poly_wasm_stepplan_kernel_bytes(p, k, &len);
    ASSERT_NOT_NULL(bytes);
    ASSERT_TRUE(len > 8);
    ASSERT_INT_EQ(bytes[0], 0x00);  /* \0 */
    ASSERT_INT_EQ(bytes[1], 0x61);  /* a */
    ASSERT_INT_EQ(bytes[2], 0x73);  /* s */
    ASSERT_INT_EQ(bytes[3], 0x6d);  /* m */
  }
  poly_wasm_stepplan_destroy(p);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(wasm, step_plan_buf_nbytes) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer_f32(ctx, 8);
  PolyUOp *out = poly_buffer_f32(ctx, 8);
  PolyUOp *v = poly_alu2(ctx, POLY_OP_ADD, a, poly_const_float(ctx, 1.0));
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, v));

  PolyWasmStepPlan *p = poly_render_step_wasm_plan(ctx, sink);
  ASSERT_NOT_NULL(p);

  /* f32 buffers: 8 elements * 4 bytes = 32 */
  int n_bindable = poly_wasm_stepplan_n_bindable_buffers(p);
  ASSERT_TRUE(n_bindable >= 2);
  for (int i = 0; i < n_bindable; i++) {
    int64_t nbytes = poly_wasm_stepplan_buf_nbytes(p, i);
    int64_t nelems = poly_wasm_stepplan_buf_size(p, i);
    ASSERT_INT_EQ(nbytes, nelems * 4);  /* f32 = 4 bytes */
  }

  /* bindable_buf_index is identity today */
  for (int i = 0; i < n_bindable; i++) {
    ASSERT_INT_EQ(poly_wasm_stepplan_bindable_buf_index(p, i), i);
  }
  /* out of range returns -1 */
  ASSERT_INT_EQ(poly_wasm_stepplan_bindable_buf_index(p, -1), -1);
  ASSERT_INT_EQ(poly_wasm_stepplan_bindable_buf_index(p, n_bindable), -1);

  poly_wasm_stepplan_destroy(p);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(wasm, step_plan_buf_nbytes_f64) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer_f64(ctx, 4);
  PolyUOp *out = poly_buffer_f64(ctx, 4);
  PolyUOp *v = poly_alu2(ctx, POLY_OP_ADD, a, poly_const_double(ctx, 1.0));
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, v));

  PolyWasmStepPlan *p = poly_render_step_wasm_plan(ctx, sink);
  ASSERT_NOT_NULL(p);

  /* f64 buffers: 4 elements * 8 bytes = 32 */
  int n_bindable = poly_wasm_stepplan_n_bindable_buffers(p);
  ASSERT_TRUE(n_bindable >= 2);
  for (int i = 0; i < n_bindable; i++) {
    int64_t nbytes = poly_wasm_stepplan_buf_nbytes(p, i);
    int64_t nelems = poly_wasm_stepplan_buf_size(p, i);
    ASSERT_INT_EQ(nbytes, nelems * 8);  /* f64 = 8 bytes */
  }

  poly_wasm_stepplan_destroy(p);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(wasm, abi_version) {
  ASSERT_INT_EQ(poly_abi_version(), POLYGRAD_ABI_VERSION);
  ASSERT_TRUE(poly_abi_version() >= 1);
  PASS();
}

/* ── C3: Float64 pipeline tests ──────────────────────────────────────── */

TEST(f64, const_typed) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *cf32 = poly_const_typed(ctx, POLY_FLOAT32, 3.14);
  ASSERT_TRUE(poly_dtype_eq(cf32->dtype, POLY_FLOAT32));
  PolyUOp *cf64 = poly_const_typed(ctx, POLY_FLOAT64, 3.14);
  ASSERT_TRUE(poly_dtype_eq(cf64->dtype, POLY_FLOAT64));
  PolyUOp *ci32 = poly_const_typed(ctx, POLY_INT32, 42.0);
  ASSERT_TRUE(poly_dtype_eq(ci32->dtype, POLY_INT32));
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(f64, vecadd_e2e) {
  int N = 8;
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer_f64(ctx, N);
  PolyUOp *b = poly_buffer_f64(ctx, N);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT64, N);
  PolyUOp *add = poly_alu2(ctx, POLY_OP_ADD, a, b);
  PolyUOp *store = poly_store_val(ctx, out, add);
  PolyUOp *sink = poly_sink1(ctx, store);

  double a_d[8], b_d[8], o_d[8];
  for (int i = 0; i < N; i++) {
    a_d[i] = (double)(i + 1) * 1.1;
    b_d[i] = (double)(i + 1) * 2.2;
  }

  PolyBufferBinding bindings[] = {
    POLY_BIND_HOST(out, o_d), POLY_BIND_HOST(a, a_d), POLY_BIND_HOST(b, b_d),
  };
  int ret = poly_realize(ctx, sink, bindings, 3);
  ASSERT_INT_EQ(ret, 0);
  for (int i = 0; i < N; i++)
    ASSERT_FLOAT_EQ(o_d[i], a_d[i] + b_d[i], 1e-14);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(f64, exp_log_roundtrip) {
  /* exp(log(x)) roundtrip with f64 buffers.
   * poly_log/poly_exp use f32 constants internally (1/ln2, ln2 via cf()).
   * This limits precision to ~1e-7 even with f64 buffers. The test verifies
   * that the f64 renderer path works (exp2/log2 instead of exp2f/log2f)
   * and that f64 buffer I/O is correct. Full f64 precision requires
   * dtype-aware composed ops (future work). */
  int N = 4;
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_buffer_f64(ctx, N);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT64, N);
  PolyUOp *lx = poly_log(ctx, x);
  PolyUOp *elx = poly_exp(ctx, lx);
  PolyUOp *store = poly_store_val(ctx, out, elx);
  PolyUOp *sink = poly_sink1(ctx, store);

  double x_d[4] = { 0.5, 1.0, 2.71828, 100.0 };
  double o_d[4] = {0};

  PolyBufferBinding bindings[] = {
    POLY_BIND_HOST(out, o_d), POLY_BIND_HOST(x, x_d),
  };
  int ret = poly_realize(ctx, sink, bindings, 2);
  ASSERT_INT_EQ(ret, 0);
  for (int i = 0; i < N; i++) {
    double rel_err = fabs(o_d[i] - x_d[i]) / fabs(x_d[i]);
    ASSERT_TRUE(rel_err < 1e-5);
  }

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(f64, value_and_grad) {
  /* Quadratic loss with f64 buffers: loss = sum(p*p), grad = 2*p */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *p = poly_buffer_f64(ctx, 4);
  PolyUOp *sq = poly_alu2(ctx, POLY_OP_MUL, p, p);
  int64_t axes[] = {0};
  PolyUOp *loss = poly_reduce_axis(ctx, POLY_OP_ADD, sq, axes, 1);

  PolyUOp *params[] = { p };
  int loss_idx = -1;
  int grad_idxs[1] = { -1 };
  PolyStep *step = poly_compile_value_and_grad(ctx, loss, params, 1,
                                                &loss_idx, grad_idxs);
  ASSERT_NOT_NULL(step);

  int n_bufs = poly_step_n_buffers(step);
  ASSERT_TRUE(loss_idx >= 0);
  ASSERT_TRUE(grad_idxs[0] >= 0);

  /* Find param buffer */
  int param_idx = -1;
  for (int i = 0; i < n_bufs; i++) {
    PolyStepBufferInfo info;
    poly_step_buffer_info(step, i, &info);
    if (info.role == POLY_STEP_BUF_INPUT &&
        poly_dtype_eq(info.dtype, POLY_FLOAT64) &&
        info.numel == 4) {
      param_idx = i;
      break;
    }
  }
  ASSERT_TRUE(param_idx >= 0);

  double p_data[4] = { 1.0, 2.0, -3.0, 4.0 };
  double loss_out[1] = {0};
  double grad_out[4] = {0};
  void *buf_ptrs[64] = {0};
  ASSERT_TRUE(n_bufs <= 64);
  buf_ptrs[loss_idx] = loss_out;
  buf_ptrs[grad_idxs[0]] = grad_out;
  buf_ptrs[param_idx] = p_data;

  int ret = poly_step_run_indexed(step, buf_ptrs, n_bufs);
  ASSERT_INT_EQ(ret, 0);
  /* loss = 1 + 4 + 9 + 16 = 30 */
  ASSERT_FLOAT_EQ(loss_out[0], 30.0, 1e-12);
  /* grad = 2 * p */
  ASSERT_FLOAT_EQ(grad_out[0], 2.0, 1e-12);
  ASSERT_FLOAT_EQ(grad_out[1], 4.0, 1e-12);
  ASSERT_FLOAT_EQ(grad_out[2], -6.0, 1e-12);
  ASSERT_FLOAT_EQ(grad_out[3], 8.0, 1e-12);

  poly_step_destroy(step);
  poly_ctx_destroy(ctx);
  PASS();
}
