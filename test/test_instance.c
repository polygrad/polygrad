/*
 * test_instance.c -- Tests for PolyInstance runtime
 */

#include "test_harness.h"
#include "../src/instance.h"
#include "../src/ir.h"
#include "../src/frontend.h"
#include "../src/sched.h"
#include <string.h>
#include <stdlib.h>
#include <math.h>

/* ── Helper: build IR bytes for a simple add graph ───────────────────── */
/* out = a + b, forward entrypoint */
static uint8_t *make_add_ir(int *out_len) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer_f32(ctx, 4);
  PolyUOp *b = poly_buffer_f32(ctx, 4);
  PolyUOp *sum = poly_alu2(ctx, POLY_OP_ADD, a, b);
  PolyUOp *out_buf = poly_buffer_f32(ctx, 4);
  PolyUOp *store = poly_store_val(ctx, out_buf, sum);
  PolyUOp *sink = poly_sink1(ctx, store);

  PolyIrBufEntry bufs[] = {
    { "a", POLY_IR_ROLE_INPUT, a, { 4 }, 1 },
    { "b", POLY_IR_ROLE_INPUT, b, { 4 }, 1 },
    { "output", POLY_IR_ROLE_OUTPUT, out_buf, { 4 }, 1 },
  };
  PolyIrEntrypoint eps[] = { { "forward", sink } };
  PolyIrSpec spec = { ctx, bufs, 3, eps, 1 };

  uint8_t *bytes = poly_ir_export(&spec, out_len);
  poly_ctx_destroy(ctx);
  return bytes;
}

/* ── Helper: build IR bytes for a trainable model ────────────────────── */
/* out = w * x, loss = sum((out - y)^2) / N */
static uint8_t *make_train_ir(int n, int *out_len) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *w = poly_buffer_f32(ctx, n);
  PolyUOp *x = poly_buffer_f32(ctx, n);
  PolyUOp *y = poly_buffer_f32(ctx, n);
  PolyUOp *out_buf = poly_buffer_f32(ctx, n);
  PolyUOp *loss_buf = poly_buffer_f32(ctx, 1);

  /* Forward: out = w * x */
  PolyUOp *prod = poly_alu2(ctx, POLY_OP_MUL, w, x);
  PolyUOp *fwd_store = poly_store_val(ctx, out_buf, prod);
  PolyUOp *fwd_sink = poly_sink1(ctx, fwd_store);

  /* Loss: sum((prod - y)^2) */
  PolyUOp *diff = poly_alu2(ctx, POLY_OP_ADD, prod,
                              poly_alu1(ctx, POLY_OP_NEG, y));
  PolyUOp *sq = poly_alu2(ctx, POLY_OP_MUL, diff, diff);
  int64_t axes[] = { 0 };
  PolyUOp *loss_val = poly_reduce_axis(ctx, POLY_OP_ADD, sq, axes, 1);

  /* Scale by 1/N */
  PolyUOp *scale = poly_const_float(ctx, 1.0 / (double)n);
  PolyUOp *mse = poly_alu2(ctx, POLY_OP_MUL, loss_val, scale);

  PolyUOp *loss_store = poly_store_val(ctx, loss_buf, mse);
  PolyUOp *loss_sink = poly_sink1(ctx, loss_store);

  PolyIrBufEntry bufs[] = {
    { "w", POLY_IR_ROLE_PARAM, w, { n }, 1 },
    { "x", POLY_IR_ROLE_INPUT, x, { n }, 1 },
    { "y", POLY_IR_ROLE_TARGET, y, { n }, 1 },
    { "output", POLY_IR_ROLE_OUTPUT, out_buf, { n }, 1 },
    { "loss", POLY_IR_ROLE_OUTPUT, loss_buf, { 1 }, 1 },
  };
  PolyIrEntrypoint eps[] = {
    { "forward", fwd_sink },
    { "loss", loss_sink },
  };
  PolyIrSpec spec = { ctx, bufs, 5, eps, 2 };

  uint8_t *bytes = poly_ir_export(&spec, out_len);
  poly_ctx_destroy(ctx);
  return bytes;
}

/* ── Tests ───────────────────────────────────────────────────────────── */

TEST(instance, create_from_ir) {
  int ir_len = 0;
  uint8_t *ir = make_add_ir(&ir_len);
  ASSERT_NOT_NULL(ir);

  PolyInstance *inst = poly_instance_from_ir(ir, ir_len, NULL, 0);
  ASSERT_NOT_NULL(inst);

  ASSERT_INT_EQ(poly_instance_buf_count(inst), 3);
  ASSERT_INT_EQ(poly_instance_param_count(inst), 0);

  ASSERT_STR_EQ(poly_instance_buf_name(inst, 0), "a");
  ASSERT_INT_EQ(poly_instance_buf_role(inst, 0), POLY_ROLE_INPUT);
  ASSERT_STR_EQ(poly_instance_buf_name(inst, 1), "b");
  ASSERT_STR_EQ(poly_instance_buf_name(inst, 2), "output");
  ASSERT_INT_EQ(poly_instance_buf_role(inst, 2), POLY_ROLE_OUTPUT);

  int64_t shape[8];
  int ndim = poly_instance_buf_shape(inst, 2, shape, 8);
  ASSERT_INT_EQ(ndim, 1);
  ASSERT_INT_EQ((int)shape[0], 4);

  poly_instance_free(inst);
  free(ir);
  PASS();
}

TEST(instance, forward_add) {
  int ir_len = 0;
  uint8_t *ir = make_add_ir(&ir_len);
  PolyInstance *inst = poly_instance_from_ir(ir, ir_len, NULL, 0);
  ASSERT_NOT_NULL(inst);

  float a_data[] = { 1.0f, 2.0f, 3.0f, 4.0f };
  float b_data[] = { 10.0f, 20.0f, 30.0f, 40.0f };
  PolyIOBinding inputs[] = {
    { "a", a_data },
    { "b", b_data },
  };

  int ret = poly_instance_forward(inst, inputs, 2);
  ASSERT_INT_EQ(ret, 0);

  /* Read output */
  int64_t numel;
  float *out = poly_instance_buf_data(inst, 2, &numel);
  ASSERT_NOT_NULL(out);
  ASSERT_INT_EQ((int)numel, 4);
  ASSERT_TRUE(fabsf(out[0] - 11.0f) < 1e-5f);
  ASSERT_TRUE(fabsf(out[1] - 22.0f) < 1e-5f);
  ASSERT_TRUE(fabsf(out[2] - 33.0f) < 1e-5f);
  ASSERT_TRUE(fabsf(out[3] - 44.0f) < 1e-5f);

  poly_instance_free(inst);
  free(ir);
  PASS();
}

TEST(instance, param_enumeration) {
  int ir_len = 0;
  uint8_t *ir = make_train_ir(4, &ir_len);
  PolyInstance *inst = poly_instance_from_ir(ir, ir_len, NULL, 0);
  ASSERT_NOT_NULL(inst);

  ASSERT_INT_EQ(poly_instance_param_count(inst), 1);
  ASSERT_STR_EQ(poly_instance_param_name(inst, 0), "w");

  int64_t shape[8];
  int ndim = poly_instance_param_shape(inst, 0, shape, 8);
  ASSERT_INT_EQ(ndim, 1);
  ASSERT_INT_EQ((int)shape[0], 4);

  int64_t numel;
  float *data = poly_instance_param_data(inst, 0, &numel);
  ASSERT_NOT_NULL(data);
  ASSERT_INT_EQ((int)numel, 4);

  /* Params should be zero-initialized */
  for (int i = 0; i < 4; i++)
    ASSERT_TRUE(data[i] == 0.0f);

  poly_instance_free(inst);
  free(ir);
  PASS();
}

TEST(instance, weights_round_trip) {
  int ir_len = 0;
  uint8_t *ir = make_train_ir(4, &ir_len);
  PolyInstance *inst = poly_instance_from_ir(ir, ir_len, NULL, 0);
  ASSERT_NOT_NULL(inst);

  /* Set param values */
  int64_t numel;
  float *w = poly_instance_param_data(inst, 0, &numel);
  w[0] = 1.0f; w[1] = 2.0f; w[2] = 3.0f; w[3] = 4.0f;

  /* Export */
  int st_len = 0;
  uint8_t *st = poly_instance_export_weights(inst, &st_len);
  ASSERT_NOT_NULL(st);
  ASSERT_TRUE(st_len > 0);

  /* Create new instance and import */
  PolyInstance *inst2 = poly_instance_from_ir(ir, ir_len, st, st_len);
  ASSERT_NOT_NULL(inst2);

  float *w2 = poly_instance_param_data(inst2, 0, &numel);
  ASSERT_TRUE(fabsf(w2[0] - 1.0f) < 1e-6f);
  ASSERT_TRUE(fabsf(w2[1] - 2.0f) < 1e-6f);
  ASSERT_TRUE(fabsf(w2[2] - 3.0f) < 1e-6f);
  ASSERT_TRUE(fabsf(w2[3] - 4.0f) < 1e-6f);

  poly_instance_free(inst);
  poly_instance_free(inst2);
  free(ir);
  free(st);
  PASS();
}

TEST(instance, export_ir_round_trip) {
  int ir_len = 0;
  uint8_t *ir = make_add_ir(&ir_len);
  PolyInstance *inst = poly_instance_from_ir(ir, ir_len, NULL, 0);
  ASSERT_NOT_NULL(inst);

  /* Re-export IR */
  int ir2_len = 0;
  uint8_t *ir2 = poly_instance_export_ir(inst, &ir2_len);
  ASSERT_NOT_NULL(ir2);
  ASSERT_TRUE(ir2_len > 0);

  /* Create new instance from re-exported IR */
  PolyInstance *inst2 = poly_instance_from_ir(ir2, ir2_len, NULL, 0);
  ASSERT_NOT_NULL(inst2);
  ASSERT_INT_EQ(poly_instance_buf_count(inst2), 3);
  ASSERT_STR_EQ(poly_instance_buf_name(inst2, 0), "a");

  poly_instance_free(inst);
  poly_instance_free(inst2);
  free(ir);
  free(ir2);
  PASS();
}

TEST(instance, train_step_sgd) {
  int ir_len = 0;
  uint8_t *ir = make_train_ir(4, &ir_len);
  PolyInstance *inst = poly_instance_from_ir(ir, ir_len, NULL, 0);
  ASSERT_NOT_NULL(inst);

  /* Init weights to 1.0 */
  int64_t numel;
  float *w = poly_instance_param_data(inst, 0, &numel);
  for (int i = 0; i < 4; i++) w[i] = 1.0f;

  /* Configure SGD */
  poly_instance_set_optimizer(inst, POLY_OPTIM_SGD,
                               0.05f, 0.0f, 0.0f, 0.0f, 0.0f);

  /* Training data: x=1,1,1,1; y=3,3,3,3 (target: w=3) */
  float x[] = { 1.0f, 1.0f, 1.0f, 1.0f };
  float y[] = { 3.0f, 3.0f, 3.0f, 3.0f };
  PolyIOBinding io[] = {
    { "x", x },
    { "y", y },
  };

  /* Run several train steps and check loss decreases */
  float prev_loss = 1e10f;
  for (int step = 0; step < 50; step++) {
    float loss;
    int ret = poly_instance_train_step(inst, io, 2, &loss);
    ASSERT_INT_EQ(ret, 0);
    if (step > 0)
      ASSERT_TRUE(loss <= prev_loss + 1e-6f);
    prev_loss = loss;
  }

  /* Loss should decrease substantially */
  ASSERT_TRUE(prev_loss < 2.0f);

  poly_instance_free(inst);
  free(ir);
  PASS();
}

TEST(instance, train_step_adam) {
  int ir_len = 0;
  uint8_t *ir = make_train_ir(4, &ir_len);
  PolyInstance *inst = poly_instance_from_ir(ir, ir_len, NULL, 0);
  ASSERT_NOT_NULL(inst);

  /* Init weights */
  int64_t numel;
  float *w = poly_instance_param_data(inst, 0, &numel);
  for (int i = 0; i < 4; i++) w[i] = 0.5f;

  /* Configure Adam */
  poly_instance_set_optimizer(inst, POLY_OPTIM_ADAM,
                               0.05f, 0.9f, 0.999f, 1e-8f, 0.0f);

  float x[] = { 1.0f, 1.0f, 1.0f, 1.0f };
  float y[] = { 3.0f, 3.0f, 3.0f, 3.0f };
  PolyIOBinding io[] = { { "x", x }, { "y", y } };

  float first_loss = -1.0f;
  float prev_loss = 1e10f;
  for (int step = 0; step < 50; step++) {
    float loss;
    int ret = poly_instance_train_step(inst, io, 2, &loss);
    ASSERT_INT_EQ(ret, 0);
    if (step == 0) first_loss = loss;
    prev_loss = loss;
  }

  /* Loss should decrease from initial */
  ASSERT_TRUE(prev_loss < first_loss);

  poly_instance_free(inst);
  free(ir);
  PASS();
}

TEST(instance, null_safety) {
  ASSERT_INT_EQ(poly_instance_buf_count(NULL), 0);
  ASSERT_INT_EQ(poly_instance_param_count(NULL), 0);
  ASSERT_TRUE(poly_instance_buf_name(NULL, 0) == NULL);
  ASSERT_TRUE(poly_instance_param_name(NULL, 0) == NULL);
  ASSERT_TRUE(poly_instance_param_data(NULL, 0, NULL) == NULL);
  poly_instance_free(NULL);  /* should not crash */
  PASS();
}
