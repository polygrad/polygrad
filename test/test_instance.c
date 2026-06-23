/*
 * test_instance.c -- Tests for PolyInstance runtime
 */

#include "test_harness.h"
#include "../src/instance.h"
#include "../src/codegen.h"
#include "../src/ctx.h"
#include "../src/engine/schedule.h"
#include "../src/ir.h"
#include "../src/frontend.h"
#include "../src/engine/schedule.h"
#include "../src/optim.h"
#include "../src/tensor.h"
#include "../src/safetensors.h"
#include <string.h>
#include <stdlib.h>
#include <math.h>

/* Helper: build IR bytes for a simple add graph */
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
      {.name = "a", .role = POLY_IR_ROLE_INPUT, .buffer = a, .shape = {4}, .ndim = 1},
      {.name = "b", .role = POLY_IR_ROLE_INPUT, .buffer = b, .shape = {4}, .ndim = 1},
      {.name = "output", .role = POLY_IR_ROLE_OUTPUT, .buffer = out_buf, .shape = {4}, .ndim = 1},
  };
  PolyIrEntrypoint eps[] = {{.name = "forward", .sink = sink}};
  PolyIrSpec spec = {ctx, bufs, 3, eps, 1};

  uint8_t *bytes = poly_ir_export(&spec, out_len);
  poly_ctx_destroy(ctx);
  return bytes;
}

TEST(optim, build_step_sgd_updates_param_with_after_store) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *p_buf = poly_buffer_f32(ctx, 1);
  PolyUOp *g_buf = poly_buffer_f32(ctx, 1);
  float p_data[] = {1.0f};
  float g_data[] = {2.0f};
  poly_buffer_set(ctx, p_buf, p_data, sizeof(p_data), POLY_DEVICE_CPU);
  poly_buffer_set(ctx, g_buf, g_data, sizeof(g_data), POLY_DEVICE_CPU);

  PolyTensor *param = poly_tensor_create(ctx, p_buf, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  PolyTensor *grad = poly_tensor_create(ctx, g_buf, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(param);
  ASSERT_NOT_NULL(grad);

  PolyOptimConfig cfg = {
      .kind = POLY_OPTIM_SGD,
      .lr = 0.1f,
      .beta1 = 0.0f,
      .beta2 = 0.0f,
      .eps = 0.0f,
      .weight_decay = 0.0f,
      .momentum = 0.0f,
      .nesterov = false,
      .classic = false,
  };
  int need = poly_optim_build_step(ctx, &cfg, &param, &grad, 1, NULL, NULL, NULL, NULL, NULL, 0);
  ASSERT_INT_EQ(need, 1);
  PolyTensor *outs[1] = {NULL};
  ASSERT_INT_EQ(
      poly_optim_build_step(ctx, &cfg, &param, &grad, 1, NULL, NULL, NULL, NULL, outs, 1), 1
  );
  ASSERT_PTR_EQ(outs[0], param);
  ASSERT_INT_EQ(poly_tensor_uop(param)->op, POLY_OP_AFTER);

  PolyTensor *realized = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, outs, 1, &realized), 0);
  ASSERT_PTR_EQ(realized, param);
  PolyBuffer *buf = poly_buffer_get(ctx, p_buf);
  ASSERT_NOT_NULL(buf);
  ASSERT_FLOAT_EQ(((float *)buf->ptr)[0], 0.8f, 1e-5f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(optim, build_step_adam_updates_beta_power_state_in_graph) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *p_buf = poly_buffer_f32(ctx, 1);
  PolyUOp *g_buf = poly_buffer_f32(ctx, 1);
  PolyUOp *m_buf = poly_buffer_f32(ctx, 1);
  PolyUOp *v_buf = poly_buffer_f32(ctx, 1);
  PolyUOp *bc1_buf = poly_buffer_f32(ctx, 1);
  PolyUOp *bc2_buf = poly_buffer_f32(ctx, 1);
  float p_data[] = {1.0f}, g_data[] = {1.0f}, m_data[] = {0.0f}, v_data[] = {0.0f};
  float bc1_data[] = {1.0f}, bc2_data[] = {1.0f};
  poly_buffer_set(ctx, p_buf, p_data, sizeof(p_data), POLY_DEVICE_CPU);
  poly_buffer_set(ctx, g_buf, g_data, sizeof(g_data), POLY_DEVICE_CPU);
  poly_buffer_set(ctx, m_buf, m_data, sizeof(m_data), POLY_DEVICE_CPU);
  poly_buffer_set(ctx, v_buf, v_data, sizeof(v_data), POLY_DEVICE_CPU);
  poly_buffer_set(ctx, bc1_buf, bc1_data, sizeof(bc1_data), POLY_DEVICE_CPU);
  poly_buffer_set(ctx, bc2_buf, bc2_data, sizeof(bc2_data), POLY_DEVICE_CPU);

  PolyTensor *param = poly_tensor_create(ctx, p_buf, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  PolyTensor *grad = poly_tensor_create(ctx, g_buf, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  PolyTensor *m = poly_tensor_create(ctx, m_buf, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  PolyTensor *v = poly_tensor_create(ctx, v_buf, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  PolyTensor *bc1 = poly_tensor_create(ctx, bc1_buf, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  PolyTensor *bc2 = poly_tensor_create(ctx, bc2_buf, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(param);
  ASSERT_NOT_NULL(grad);
  ASSERT_NOT_NULL(m);
  ASSERT_NOT_NULL(v);
  ASSERT_NOT_NULL(bc1);
  ASSERT_NOT_NULL(bc2);

  PolyOptimConfig cfg = {
      .kind = POLY_OPTIM_ADAM,
      .lr = 0.1f,
      .beta1 = 0.9f,
      .beta2 = 0.999f,
      .eps = 1e-8f,
      .weight_decay = 0.0f,
      .momentum = 0.0f,
      .nesterov = false,
      .classic = false,
  };
  PolyTensor *m_arr[] = {m};
  PolyTensor *v_arr[] = {v};
  int need = poly_optim_build_step(ctx, &cfg, &param, &grad, 1, m_arr, v_arr, bc1, bc2, NULL, 0);
  ASSERT_INT_EQ(need, 5);
  PolyTensor *outs[5] = {0};
  ASSERT_INT_EQ(
      poly_optim_build_step(ctx, &cfg, &param, &grad, 1, m_arr, v_arr, bc1, bc2, outs, 5), 5
  );
  PolyTensor *realized[5] = {0};
  ASSERT_INT_EQ(poly_realize_tensors(ctx, outs, 5, realized), 0);

  ASSERT_FLOAT_EQ(((float *)poly_buffer_get(ctx, p_buf)->ptr)[0], 0.9f, 1e-4f);
  ASSERT_FLOAT_EQ(((float *)poly_buffer_get(ctx, m_buf)->ptr)[0], 0.1f, 1e-5f);
  ASSERT_FLOAT_EQ(((float *)poly_buffer_get(ctx, v_buf)->ptr)[0], 0.001f, 1e-6f);
  ASSERT_FLOAT_EQ(((float *)poly_buffer_get(ctx, bc1_buf)->ptr)[0], 0.9f, 1e-6f);
  ASSERT_FLOAT_EQ(((float *)poly_buffer_get(ctx, bc2_buf)->ptr)[0], 0.999f, 1e-6f);

  poly_ctx_destroy(ctx);
  PASS();
}

/* Helper: build IR bytes for a trainable model */
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
  PolyUOp *diff = poly_alu2(ctx, POLY_OP_ADD, prod, poly_alu1(ctx, POLY_OP_NEG, y));
  PolyUOp *sq = poly_alu2(ctx, POLY_OP_MUL, diff, diff);
  int64_t axes[] = {0};
  PolyUOp *loss_val = poly_reduce_axis(ctx, POLY_OP_ADD, sq, axes, 1);

  /* Scale by 1/N */
  PolyUOp *scale = poly_const_float(ctx, 1.0 / (double)n);
  PolyUOp *mse = poly_alu2(ctx, POLY_OP_MUL, loss_val, scale);

  PolyUOp *loss_store = poly_store_val(ctx, loss_buf, mse);
  PolyUOp *loss_sink = poly_sink1(ctx, loss_store);

  PolyIrBufEntry bufs[] = {
      {.name = "w", .role = POLY_IR_ROLE_PARAM, .buffer = w, .shape = {n}, .ndim = 1},
      {.name = "x", .role = POLY_IR_ROLE_INPUT, .buffer = x, .shape = {n}, .ndim = 1},
      {.name = "y", .role = POLY_IR_ROLE_TARGET, .buffer = y, .shape = {n}, .ndim = 1},
      {.name = "output", .role = POLY_IR_ROLE_OUTPUT, .buffer = out_buf, .shape = {n}, .ndim = 1},
      {.name = "loss", .role = POLY_IR_ROLE_OUTPUT, .buffer = loss_buf, .shape = {1}, .ndim = 1},
  };
  PolyIrEntrypoint eps[] = {
      {.name = "forward", .sink = fwd_sink},
      {.name = "loss", .sink = loss_sink},
  };
  PolyIrSpec spec = {ctx, bufs, 5, eps, 2};

  uint8_t *bytes = poly_ir_export(&spec, out_len);
  poly_ctx_destroy(ctx);
  return bytes;
}

/* Tests */

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

  float a_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
  float b_data[] = {10.0f, 20.0f, 30.0f, 40.0f};
  PolyIOBinding inputs[] = {
      {"a", a_data},
      {"b", b_data},
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

TEST(instance, staged_build_forward_e2e) {
  PolyCtx *ctx = poly_ctx_new();
  PolyInstance *inst = poly_instance_new(ctx, NULL);
  ASSERT_NOT_NULL(inst);
  ASSERT_INT_EQ(poly_instance_stage(inst), POLY_INSTANCE_BUILDING);

  int64_t shape[] = {4};
  PolyTensor *x = poly_instance_input(inst, "x", POLY_FLOAT32, shape, 1);
  PolyTensor *w = poly_instance_param(inst, "w", POLY_FLOAT32, shape, 1);
  ASSERT_NOT_NULL(x);
  ASSERT_NOT_NULL(w);

  PolyUOp *sum = poly_alu2(ctx, POLY_OP_ADD, poly_tensor_uop(x), poly_tensor_uop(w));
  PolyTensor *out = poly_tensor_create(ctx, sum, POLY_TENSOR_VALUE, POLY_DEVICE_AUTO);
  ASSERT_NOT_NULL(out);
  ASSERT_INT_EQ(poly_instance_output(inst, "output", out), POLY_STATUS_OK);

  const char *inputs[] = {"x"};
  const char *outputs[] = {"output"};
  ASSERT_INT_EQ(
      poly_instance_entrypoint(inst, "forward", inputs, 1, outputs, 1, NULL), POLY_STATUS_OK
  );
  ASSERT_INT_EQ(poly_instance_build(inst, NULL), POLY_STATUS_OK);
  ASSERT_INT_EQ(poly_instance_stage(inst), POLY_INSTANCE_BUILT);
  ASSERT_INT_EQ(poly_instance_buf_count(inst), 3);
  ASSERT_INT_EQ(poly_instance_param_count(inst), 1);

  int64_t numel = 0;
  float *w_data = poly_instance_param_data(inst, 0, &numel);
  ASSERT_NOT_NULL(w_data);
  ASSERT_INT_EQ((int)numel, 4);
  w_data[0] = 10.0f;
  w_data[1] = 20.0f;
  w_data[2] = 30.0f;
  w_data[3] = 40.0f;

  float x_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
  PolyIOBinding io[] = {{"x", x_data}};
  ASSERT_INT_EQ(poly_instance_forward(inst, io, 1), 0);

  float *y = poly_instance_buf_data_named(inst, "output", &numel);
  ASSERT_NOT_NULL(y);
  ASSERT_INT_EQ((int)numel, 4);
  ASSERT_FLOAT_EQ(y[0], 11.0f, 1e-5f);
  ASSERT_FLOAT_EQ(y[1], 22.0f, 1e-5f);
  ASSERT_FLOAT_EQ(y[2], 33.0f, 1e-5f);
  ASSERT_FLOAT_EQ(y[3], 44.0f, 1e-5f);

  poly_instance_free(inst);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(instance, staged_build_validation_rewinds_scratch_on_success) {
  PolyCtx *ctx = poly_ctx_new();
  PolyInstance *inst = poly_instance_new(ctx, NULL);
  ASSERT_NOT_NULL(inst);

  int64_t shape[] = {4};
  PolyTensor *x = poly_instance_input(inst, "x", POLY_FLOAT32, shape, 1);
  PolyTensor *w = poly_instance_param(inst, "w", POLY_FLOAT32, shape, 1);
  ASSERT_NOT_NULL(x);
  ASSERT_NOT_NULL(w);

  PolyUOp *sum = poly_alu2(ctx, POLY_OP_ADD, poly_tensor_uop(x), poly_tensor_uop(w));
  PolyTensor *out = poly_tensor_create(ctx, sum, POLY_TENSOR_VALUE, POLY_DEVICE_AUTO);
  ASSERT_NOT_NULL(out);
  ASSERT_INT_EQ(poly_instance_output(inst, "output", out), POLY_STATUS_OK);

  const char *inputs[] = {"x"};
  const char *outputs[] = {"output"};
  ASSERT_INT_EQ(
      poly_instance_entrypoint(inst, "forward", inputs, 1, outputs, 1, NULL), POLY_STATUS_OK
  );

  size_t scratch_before = poly_arena_used(ctx->scratch);
  ASSERT_INT_EQ(poly_instance_build(inst, NULL), POLY_STATUS_OK);
  ASSERT_INT_EQ(poly_arena_used(ctx->scratch), scratch_before);

  poly_instance_free(inst);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(instance, staged_stage_guards) {
  PolyCtx *ctx = poly_ctx_new();
  PolyInstance *inst = poly_instance_new(ctx, NULL);
  ASSERT_NOT_NULL(inst);
  ASSERT_INT_EQ(poly_instance_forward(inst, NULL, 0), -1);

  int64_t shape[] = {1};
  PolyTensor *x = poly_instance_input(inst, "x", POLY_FLOAT32, shape, 1);
  ASSERT_NOT_NULL(x);
  ASSERT_INT_EQ(poly_instance_output(inst, "output", x), POLY_STATUS_OK);
  const char *inputs[] = {"x"};
  const char *outputs[] = {"output"};
  ASSERT_INT_EQ(
      poly_instance_entrypoint(inst, "forward", inputs, 1, outputs, 1, NULL), POLY_STATUS_OK
  );
  ASSERT_INT_EQ(poly_instance_build(inst, NULL), POLY_STATUS_OK);
  ASSERT_TRUE(poly_instance_param(inst, "late", POLY_FLOAT32, shape, 1) == NULL);
  const PolyInstanceError *err = poly_instance_last_error(inst);
  ASSERT_NOT_NULL(err);
  ASSERT_INT_EQ(err->code, POLY_STATUS_BAD_STAGE);

  poly_instance_free(inst);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(instance, staged_build_validation_rewinds_scratch_on_unbound_storage) {
  PolyCtx *ctx = poly_ctx_new();
  PolyInstance *inst = poly_instance_new(ctx, NULL);
  ASSERT_NOT_NULL(inst);

  int64_t shape[] = {4};
  PolyTensor *x = poly_instance_input(inst, "x", POLY_FLOAT32, shape, 1);
  ASSERT_NOT_NULL(x);

  PolyUOp *w_buf = poly_buffer_f32(ctx, 4);
  PolyTensor *w = poly_tensor_create(ctx, w_buf, POLY_TENSOR_VALUE, POLY_DEVICE_AUTO);
  ASSERT_NOT_NULL(w);

  PolyUOp *sum = poly_alu2(ctx, POLY_OP_ADD, poly_tensor_uop(x), poly_tensor_uop(w));
  PolyTensor *out = poly_tensor_create(ctx, sum, POLY_TENSOR_VALUE, POLY_DEVICE_AUTO);
  ASSERT_NOT_NULL(out);
  ASSERT_INT_EQ(poly_instance_output(inst, "output", out), POLY_STATUS_OK);

  const char *inputs[] = {"x"};
  const char *outputs[] = {"output"};
  ASSERT_INT_EQ(
      poly_instance_entrypoint(inst, "forward", inputs, 1, outputs, 1, NULL), POLY_STATUS_OK
  );

  size_t scratch_before = poly_arena_used(ctx->scratch);
  ASSERT_INT_EQ(poly_instance_build(inst, NULL), POLY_STATUS_INVALID);
  ASSERT_INT_EQ(poly_arena_used(ctx->scratch), scratch_before);

  poly_instance_free(inst);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(instance, staged_build_rejects_unbound_storage_leaf) {
  PolyCtx *ctx = poly_ctx_new();
  PolyInstance *inst = poly_instance_new(ctx, NULL);
  ASSERT_NOT_NULL(inst);

  int64_t shape[] = {4};
  PolyTensor *x = poly_instance_input(inst, "x", POLY_FLOAT32, shape, 1);
  ASSERT_NOT_NULL(x);

  PolyUOp *w_buf = poly_buffer_f32(ctx, 4);
  PolyTensor *w = poly_tensor_create(ctx, w_buf, POLY_TENSOR_VALUE, POLY_DEVICE_AUTO);
  ASSERT_NOT_NULL(w);

  PolyUOp *sum = poly_alu2(ctx, POLY_OP_ADD, poly_tensor_uop(x), poly_tensor_uop(w));
  PolyTensor *out = poly_tensor_create(ctx, sum, POLY_TENSOR_VALUE, POLY_DEVICE_AUTO);
  ASSERT_NOT_NULL(out);
  ASSERT_INT_EQ(poly_instance_output(inst, "output", out), POLY_STATUS_OK);

  const char *inputs[] = {"x"};
  const char *outputs[] = {"output"};
  ASSERT_INT_EQ(
      poly_instance_entrypoint(inst, "forward", inputs, 1, outputs, 1, NULL), POLY_STATUS_OK
  );
  ASSERT_INT_EQ(poly_instance_build(inst, NULL), POLY_STATUS_INVALID);
  ASSERT_INT_EQ(poly_instance_stage(inst), POLY_INSTANCE_FAILED);
  const PolyInstanceError *err = poly_instance_last_error(inst);
  ASSERT_NOT_NULL(err);
  ASSERT_TRUE(strstr(err->message, "unbound storage") != NULL);

  poly_instance_free(inst);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(instance, staged_bindings_set_tensor_metadata) {
  PolyCtx *ctx = poly_ctx_new();
  PolyInstance *inst = poly_instance_new(ctx, NULL);
  ASSERT_NOT_NULL(inst);

  int64_t shape[] = {4};
  PolyTensor *x = poly_instance_input(inst, "x", POLY_FLOAT32, shape, 1);
  PolyTensor *y = poly_instance_target(inst, "y", POLY_FLOAT32, shape, 1);
  PolyTensor *w = poly_instance_param(inst, "w", POLY_FLOAT32, shape, 1);
  ASSERT_NOT_NULL(x);
  ASSERT_NOT_NULL(y);
  ASSERT_NOT_NULL(w);

  ASSERT_TRUE(!poly_tensor_requires_grad(x));
  ASSERT_TRUE(!poly_tensor_requires_grad(y));
  ASSERT_TRUE(poly_tensor_requires_grad(w));
  ASSERT_INT_EQ(poly_tensor_provenance(x), POLY_TENSOR_PROVENANCE_USER_INPUT);
  ASSERT_INT_EQ(poly_tensor_provenance(y), POLY_TENSOR_PROVENANCE_USER_INPUT);
  ASSERT_INT_EQ(poly_tensor_provenance(w), POLY_TENSOR_PROVENANCE_PARAM_INIT);

  PolyTensor *w_cuda = poly_tensor_to_device(ctx, w, POLY_DEVICE_CUDA);
  ASSERT_NOT_NULL(w_cuda);
  ASSERT_TRUE(poly_tensor_requires_grad(w_cuda));
  ASSERT_INT_EQ(poly_tensor_provenance(w_cuda), POLY_TENSOR_PROVENANCE_PARAM_INIT);

  PolyUOp *state_buf = poly_buffer_f32(ctx, 4);
  PolyTensor *state = poly_tensor_create(ctx, state_buf, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(state);
  ASSERT_INT_EQ(poly_instance_state(inst, "loaded", state, 0), POLY_STATUS_OK);
  ASSERT_TRUE(poly_tensor_requires_grad(state));
  ASSERT_INT_EQ(poly_tensor_provenance(state), POLY_TENSOR_PROVENANCE_STATE_LOADED);

  PolyUOp *aux_buf = poly_buffer_f32(ctx, 4);
  PolyTensor *aux = poly_tensor_create(ctx, aux_buf, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(aux);
  ASSERT_INT_EQ(poly_instance_aux(inst, "aux", aux, 0), POLY_STATUS_OK);
  ASSERT_TRUE(!poly_tensor_requires_grad(aux));
  ASSERT_INT_EQ(poly_tensor_provenance(aux), POLY_TENSOR_PROVENANCE_STATE_LOADED);

  ASSERT_INT_EQ(poly_instance_output(inst, "echo", x), POLY_STATUS_OK);
  ASSERT_INT_EQ(poly_tensor_provenance(x), POLY_TENSOR_PROVENANCE_USER_INPUT);

  PolyUOp *sum = poly_alu2(ctx, POLY_OP_ADD, poly_tensor_uop(x), poly_tensor_uop(w));
  PolyTensor *out = poly_tensor_create(ctx, sum, POLY_TENSOR_VALUE, POLY_DEVICE_AUTO);
  ASSERT_NOT_NULL(out);
  ASSERT_INT_EQ(poly_tensor_provenance(out), POLY_TENSOR_PROVENANCE_UNKNOWN);
  ASSERT_INT_EQ(poly_instance_output(inst, "computed", out), POLY_STATUS_OK);
  ASSERT_INT_EQ(poly_tensor_provenance(out), POLY_TENSOR_PROVENANCE_COMPUTED);

  poly_instance_free(inst);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(instance, staged_build_rejects_unbound_trainable_storage_leaf) {
  PolyCtx *ctx = poly_ctx_new();
  PolyInstance *inst = poly_instance_new(ctx, NULL);
  ASSERT_NOT_NULL(inst);

  int64_t x_shape[] = {2, 2};
  PolyTensor *x = poly_instance_input(inst, "x", POLY_FLOAT32, x_shape, 2);
  ASSERT_NOT_NULL(x);

  PolyUOp *w_buf = poly_buffer_f32(ctx, 4);
  int64_t w_shape[] = {2, 2};
  PolyUOp *w_view = poly_reshape(ctx, w_buf, w_shape, 2);
  PolyTensor *w = poly_tensor_create(ctx, w_view, POLY_TENSOR_VALUE, POLY_DEVICE_AUTO);
  ASSERT_NOT_NULL(w);
  poly_tensor_set_requires_grad(w, true);
  poly_tensor_set_provenance(w, POLY_TENSOR_PROVENANCE_PARAM_INIT);

  PolyTensor *w_alias = poly_tensor_create(ctx, w_view, POLY_TENSOR_VALUE, POLY_DEVICE_AUTO);
  ASSERT_NOT_NULL(w_alias);

  PolyUOp *sum = poly_alu2(ctx, POLY_OP_ADD, poly_tensor_uop(x), poly_tensor_uop(w_alias));
  PolyTensor *out = poly_tensor_create(ctx, sum, POLY_TENSOR_VALUE, POLY_DEVICE_AUTO);
  ASSERT_NOT_NULL(out);
  ASSERT_INT_EQ(poly_instance_output(inst, "output", out), POLY_STATUS_OK);

  const char *inputs[] = {"x"};
  const char *outputs[] = {"output"};
  ASSERT_INT_EQ(
      poly_instance_entrypoint(inst, "forward", inputs, 1, outputs, 1, NULL), POLY_STATUS_OK
  );
  ASSERT_INT_EQ(poly_instance_build(inst, NULL), POLY_STATUS_INVALID);
  const PolyInstanceError *err = poly_instance_last_error(inst);
  ASSERT_NOT_NULL(err);
  ASSERT_TRUE(strstr(err->message, "unbound trainable storage") != NULL);

  poly_instance_free(inst);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(instance, staged_runtime_trainability_uses_tensor_metadata) {
  PolyCtx *ctx = poly_ctx_new();
  PolyInstance *inst = poly_instance_new(ctx, NULL);
  ASSERT_NOT_NULL(inst);

  int64_t x_shape[] = {1, 1};
  PolyTensor *x = poly_instance_input(inst, "x", POLY_FLOAT32, x_shape, 2);
  PolyTensor *w = poly_instance_param(inst, "w", POLY_FLOAT32, x_shape, 2);
  ASSERT_NOT_NULL(x);
  ASSERT_NOT_NULL(w);
  ASSERT_TRUE(poly_tensor_requires_grad(w));

  poly_tensor_set_requires_grad(w, false);

  PolyUOp *prod = poly_alu2(ctx, POLY_OP_MUL, poly_tensor_uop(x), poly_tensor_uop(w));
  PolyTensor *out = poly_tensor_create(ctx, prod, POLY_TENSOR_VALUE, POLY_DEVICE_AUTO);
  ASSERT_NOT_NULL(out);
  ASSERT_INT_EQ(poly_instance_output(inst, "output", out), POLY_STATUS_OK);

  const char *inputs[] = {"x"};
  const char *outputs[] = {"output"};
  ASSERT_INT_EQ(
      poly_instance_entrypoint(inst, "forward", inputs, 1, outputs, 1, NULL), POLY_STATUS_OK
  );
  ASSERT_INT_EQ(poly_instance_build(inst, NULL), POLY_STATUS_OK);
  ASSERT_INT_EQ(poly_instance_param_count(inst), 1);
  ASSERT_TRUE(!poly_instance_param_trainable(inst, 0));

  poly_instance_free(inst);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(instance, staged_objective_accepts_one_element_output) {
  PolyCtx *ctx = poly_ctx_new();
  PolyInstance *inst = poly_instance_new(ctx, NULL);
  ASSERT_NOT_NULL(inst);

  int64_t shape[] = {1};
  PolyTensor *x = poly_instance_input(inst, "x", POLY_FLOAT32, shape, 1);
  ASSERT_NOT_NULL(x);
  ASSERT_INT_EQ(poly_instance_output(inst, "loss", x), POLY_STATUS_OK);

  const char *inputs[] = {"x"};
  const char *outputs[] = {"loss"};
  PolyEntrypointOptions opts = {.objective = "loss"};
  ASSERT_INT_EQ(
      poly_instance_entrypoint(inst, "loss", inputs, 1, outputs, 1, &opts), POLY_STATUS_OK
  );
  ASSERT_INT_EQ(poly_instance_build(inst, NULL), POLY_STATUS_OK);

  poly_instance_free(inst);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(instance, staged_objective_round_trips_through_ir) {
  PolyCtx *ctx = poly_ctx_new();
  PolyInstance *inst = poly_instance_new(ctx, NULL);
  ASSERT_NOT_NULL(inst);

  int64_t x_shape[] = {1};
  PolyTensor *x = poly_instance_input(inst, "x", POLY_FLOAT32, x_shape, 1);
  ASSERT_NOT_NULL(x);
  PolyTensor *y = poly_instance_target(inst, "y", POLY_FLOAT32, x_shape, 1);
  ASSERT_NOT_NULL(y);

  ASSERT_INT_EQ(poly_instance_output(inst, "logits", x), POLY_STATUS_OK);
  ASSERT_INT_EQ(poly_instance_output(inst, "loss", y), POLY_STATUS_OK);

  const char *forward_inputs[] = {"x"};
  const char *forward_outputs[] = {"logits"};
  ASSERT_INT_EQ(
      poly_instance_entrypoint(inst, "forward", forward_inputs, 1, forward_outputs, 1, NULL),
      POLY_STATUS_OK
  );

  const char *loss_inputs[] = {"x", "y"};
  const char *loss_outputs[] = {"loss"};
  PolyEntrypointOptions opts = {.objective = "loss", .flags = 5};
  ASSERT_INT_EQ(
      poly_instance_entrypoint(inst, "loss", loss_inputs, 2, loss_outputs, 1, &opts), POLY_STATUS_OK
  );
  ASSERT_INT_EQ(poly_instance_build(inst, NULL), POLY_STATUS_OK);

  int ir_len = 0;
  uint8_t *ir = poly_instance_export_ir(inst, &ir_len);
  ASSERT_NOT_NULL(ir);

  PolyIrSpec imported;
  ASSERT_INT_EQ(poly_ir_import(ir, ir_len, &imported), 0);
  ASSERT_INT_EQ(imported.n_entrypoints, 2);
  ASSERT_STR_EQ(imported.entrypoints[1].name, "loss");
  ASSERT_INT_EQ(imported.entrypoints[1].n_inputs, 2);
  ASSERT_STR_EQ(imported.entrypoints[1].inputs[0], "x");
  ASSERT_STR_EQ(imported.entrypoints[1].inputs[1], "y");
  ASSERT_INT_EQ(imported.entrypoints[1].n_outputs, 1);
  ASSERT_STR_EQ(imported.entrypoints[1].outputs[0], "loss");
  ASSERT_STR_EQ(imported.entrypoints[1].objective, "loss");
  ASSERT_INT_EQ(imported.entrypoints[1].flags, 5);

  poly_ir_spec_free(&imported);
  poly_ctx_destroy(imported.ctx);
  free(ir);
  poly_instance_free(inst);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(instance, staged_objective_rejects_vector_output) {
  PolyCtx *ctx = poly_ctx_new();
  PolyInstance *inst = poly_instance_new(ctx, NULL);
  ASSERT_NOT_NULL(inst);

  int64_t shape[] = {2};
  PolyTensor *x = poly_instance_input(inst, "x", POLY_FLOAT32, shape, 1);
  ASSERT_NOT_NULL(x);
  ASSERT_INT_EQ(poly_instance_output(inst, "loss", x), POLY_STATUS_OK);

  const char *inputs[] = {"x"};
  const char *outputs[] = {"loss"};
  PolyEntrypointOptions opts = {.objective = "loss"};
  ASSERT_INT_EQ(
      poly_instance_entrypoint(inst, "loss", inputs, 1, outputs, 1, &opts), POLY_STATUS_OK
  );
  ASSERT_INT_EQ(poly_instance_build(inst, NULL), POLY_STATUS_INVALID);
  ASSERT_INT_EQ(poly_instance_stage(inst), POLY_INSTANCE_FAILED);
  const PolyInstanceError *err = poly_instance_last_error(inst);
  ASSERT_NOT_NULL(err);
  ASSERT_TRUE(strstr(err->message, "scalar or one element") != NULL);

  poly_instance_free(inst);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(instance, staged_state_aliases_share_storage) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *buf = poly_buffer_f32(ctx, 2);
  float init[] = {3.0f, 4.0f};
  poly_buffer_set(ctx, buf, init, sizeof(init), POLY_DEVICE_CPU);
  PolyTensor *state = poly_tensor_create(ctx, buf, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(state);

  PolyInstance *inst = poly_instance_new(ctx, NULL);
  ASSERT_NOT_NULL(inst);
  ASSERT_INT_EQ(poly_instance_state(inst, "encoder.weight", state, 0), POLY_STATUS_OK);
  ASSERT_INT_EQ(poly_instance_state(inst, "lm_head.weight", state, 0), POLY_STATUS_OK);
  ASSERT_INT_EQ(poly_instance_output(inst, "output", state), POLY_STATUS_OK);
  const char *outputs[] = {"output"};
  ASSERT_INT_EQ(
      poly_instance_entrypoint(inst, "forward", NULL, 0, outputs, 1, NULL), POLY_STATUS_OK
  );
  ASSERT_INT_EQ(poly_instance_build(inst, NULL), POLY_STATUS_OK);

  int64_t n0 = 0, n1 = 0;
  float *a = poly_instance_buf_data_named(inst, "encoder.weight", &n0);
  float *b = poly_instance_buf_data_named(inst, "lm_head.weight", &n1);
  ASSERT_NOT_NULL(a);
  ASSERT_NOT_NULL(b);
  ASSERT_PTR_EQ(a, b);
  ASSERT_INT_EQ((int)n0, 2);
  ASSERT_INT_EQ((int)n1, 2);
  ASSERT_FLOAT_EQ(a[0], 3.0f, 1e-6f);
  ASSERT_FLOAT_EQ(a[1], 4.0f, 1e-6f);

  poly_instance_free(inst);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(instance, staged_canonical_names_round_trip_through_ir) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyInstance *inst = poly_instance_new(ctx, NULL);
  ASSERT_NOT_NULL(inst);

  int64_t shape[] = {2};
  PolyTensor *x = poly_instance_input(inst, "x", POLY_FLOAT32, shape, 1);
  ASSERT_NOT_NULL(x);
  ASSERT_INT_EQ(poly_instance_scope_push(inst, "layers.%d", 0), POLY_STATUS_OK);
  PolyTensor *w = poly_instance_param(inst, "weight", POLY_FLOAT32, shape, 1);
  ASSERT_NOT_NULL(w);
  ASSERT_INT_EQ(poly_instance_scope_pop(inst), POLY_STATUS_OK);

  PolyUOp *prod = poly_alu2(ctx, POLY_OP_MUL, poly_tensor_uop(x), poly_tensor_uop(w));
  ASSERT_NOT_NULL(prod);
  PolyTensor *logits = poly_tensor_create(ctx, prod, POLY_TENSOR_VALUE, POLY_DEVICE_AUTO);
  ASSERT_NOT_NULL(logits);
  ASSERT_INT_EQ(poly_instance_output(inst, "logits", logits), POLY_STATUS_OK);

  const char *inputs[] = {"x"};
  const char *outputs[] = {"logits"};
  ASSERT_INT_EQ(
      poly_instance_entrypoint(inst, "forward", inputs, 1, outputs, 1, NULL), POLY_STATUS_OK
  );
  ASSERT_INT_EQ(poly_instance_build(inst, NULL), POLY_STATUS_OK);
  ASSERT_INT_EQ(poly_instance_param_count(inst), 1);
  ASSERT_STR_EQ(poly_instance_param_name(inst, 0), "layers.0.weight");

  int ir_len = 0;
  uint8_t *ir = poly_instance_export_ir(inst, &ir_len);
  ASSERT_NOT_NULL(ir);
  ASSERT_TRUE(ir_len > 0);

  PolyInstance *inst2 = poly_instance_from_ir(ir, ir_len, NULL, 0);
  ASSERT_NOT_NULL(inst2);
  ASSERT_INT_EQ(poly_instance_param_count(inst2), 1);
  ASSERT_STR_EQ(poly_instance_param_name(inst2, 0), "layers.0.weight");

  int found_x = 0, found_logits = 0;
  for (int i = 0; i < poly_instance_buf_count(inst2); i++) {
    const char *name = poly_instance_buf_name(inst2, i);
    if (name && strcmp(name, "x") == 0) found_x = 1;
    if (name && strcmp(name, "logits") == 0) found_logits = 1;
  }
  ASSERT_TRUE(found_x);
  ASSERT_TRUE(found_logits);

  poly_instance_free(inst2);
  free(ir);
  poly_instance_free(inst);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(instance, from_binding_arrays_forward_e2e) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *x_buf = poly_buffer_f32(ctx, 2);
  PolyUOp *w_buf = poly_buffer_f32(ctx, 2);
  ASSERT_NOT_NULL(x_buf);
  ASSERT_NOT_NULL(w_buf);

  float w_init[] = {2.0f, 3.0f};
  poly_buffer_set(ctx, w_buf, w_init, sizeof(w_init), POLY_DEVICE_CPU);

  PolyTensor *x = poly_tensor_create(ctx, x_buf, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  PolyTensor *w = poly_tensor_create(ctx, w_buf, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(x);
  ASSERT_NOT_NULL(w);

  PolyUOp *prod = poly_alu2(ctx, POLY_OP_MUL, poly_tensor_uop(x), poly_tensor_uop(w));
  PolyTensor *out = poly_tensor_create(ctx, prod, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(out);

  const char *binding_names[] = {"x", "w", "output"};
  int binding_roles[] = {POLY_ROLE_INPUT, POLY_ROLE_PARAM, POLY_ROLE_OUTPUT};
  PolyTensor *binding_tensors[] = {x, w, out};
  uint32_t binding_flags[] = {0, 0, 0};

  const char *entry_names[] = {"forward"};
  const char *entry_inputs[] = {"x"};
  int entry_input_counts[] = {1};
  const char *entry_outputs[] = {"output"};
  int entry_output_counts[] = {1};
  const char *entry_objectives[] = {NULL};
  uint32_t entry_flags[] = {0};

  PolyInstanceError err = {0};
  PolyInstance *inst = poly_instance_from_binding_arrays(
      ctx, binding_names, binding_roles, binding_tensors, binding_flags, 3, entry_names,
      entry_inputs, entry_input_counts, entry_outputs, entry_output_counts, entry_objectives,
      entry_flags, 1, NULL, &err
  );
  ASSERT_NOT_NULL(inst);
  ASSERT_INT_EQ(poly_ctx_named_count(ctx), 0);
  ASSERT_INT_EQ(poly_instance_param_count(inst), 1);
  ASSERT_STR_EQ(poly_instance_param_name(inst, 0), "w");

  float x_data[] = {10.0f, 20.0f};
  PolyIOBinding io[] = {{"x", x_data}};
  ASSERT_INT_EQ(poly_instance_forward(inst, io, 1), 0);

  int64_t numel = 0;
  float *y = poly_instance_buf_data_named(inst, "output", &numel);
  ASSERT_NOT_NULL(y);
  ASSERT_INT_EQ((int)numel, 2);
  ASSERT_FLOAT_EQ(y[0], 20.0f, 1e-5f);
  ASSERT_FLOAT_EQ(y[1], 60.0f, 1e-5f);

  poly_instance_free(inst);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(instance, from_sinks_wraps_selected_lazy_tensor_graph) {
  PolyCtx *ctx = poly_ctx_new();
  int64_t shape[] = {4};

  PolyUOp *w = poly_buffer_f32(ctx, 4);
  float w_data[] = {2.0f, 3.0f, 4.0f, 5.0f};
  poly_buffer_set(ctx, w, w_data, sizeof(w_data), POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(poly_register_existing_buffer(ctx, POLY_ROLE_PARAM, w, shape, 1, "w", true));

  PolyUOp *x = poly_register_buffer_by_id(
      ctx, POLY_ROLE_INPUT, poly_dtype_id_by_name("float32"), shape, 1, "x"
  );
  PolyUOp *out = poly_register_buffer_by_id(
      ctx, POLY_ROLE_OUTPUT, poly_dtype_id_by_name("float32"), shape, 1, "output"
  );
  ASSERT_NOT_NULL(x);
  ASSERT_NOT_NULL(out);

  /* A stale registered entrypoint should not leak into this package when the
   * frontend asks for a selected sink set. This is what lets multiple lazy
   * models share a ctx but export one ABI at a time. */
  PolyUOp *stale = poly_register_buffer_by_id(
      ctx, POLY_ROLE_OUTPUT, poly_dtype_id_by_name("float32"), shape, 1, "stale"
  );
  poly_register_entrypoint(
      ctx, "stale", poly_sink1(ctx, poly_store_val(ctx, stale, poly_alu2(ctx, POLY_OP_ADD, x, w)))
  );

  PolyUOp *prod = poly_alu2(ctx, POLY_OP_MUL, x, w);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, prod));
  const char *names[] = {"forward"};
  PolyUOp *sinks[] = {sink};
  PolyInstance *inst = poly_instance_from_sinks(ctx, names, sinks, 1);
  ASSERT_NOT_NULL(inst);
  ASSERT_INT_EQ(poly_instance_param_count(inst), 1);
  ASSERT_STR_EQ(poly_instance_param_name(inst, 0), "w");
  ASSERT_INT_EQ(poly_instance_buf_count(inst), 3);
  for (int i = 0; i < poly_instance_buf_count(inst); i++)
    ASSERT_TRUE(strcmp(poly_instance_buf_name(inst, i), "stale") != 0);

  float x_data[] = {10.0f, 10.0f, 10.0f, 10.0f};
  PolyIOBinding io[] = {{"x", x_data}};
  ASSERT_INT_EQ(poly_instance_forward(inst, io, 1), 0);
  int out_idx = -1;
  for (int i = 0; i < poly_instance_buf_count(inst); i++)
    if (strcmp(poly_instance_buf_name(inst, i), "output") == 0) out_idx = i;
  ASSERT_TRUE(out_idx >= 0);
  int64_t numel = 0;
  float *out_data = poly_instance_buf_data(inst, out_idx, &numel);
  ASSERT_INT_EQ((int)numel, 4);
  ASSERT_FLOAT_EQ(out_data[0], 20.0f, 1e-5f);
  ASSERT_FLOAT_EQ(out_data[1], 30.0f, 1e-5f);
  ASSERT_FLOAT_EQ(out_data[2], 40.0f, 1e-5f);
  ASSERT_FLOAT_EQ(out_data[3], 50.0f, 1e-5f);

  poly_instance_free(inst);
  poly_ctx_destroy(ctx);
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
  w[0] = 1.0f;
  w[1] = 2.0f;
  w[2] = 3.0f;
  w[3] = 4.0f;

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
  for (int i = 0; i < 4; i++)
    w[i] = 1.0f;

  /* Configure SGD */
  poly_instance_set_optimizer(inst, POLY_OPTIM_SGD, 0.05f, 0.0f, 0.0f, 0.0f, 0.0f);

  /* Training data: x=1,1,1,1; y=3,3,3,3 (target: w=3) */
  float x[] = {1.0f, 1.0f, 1.0f, 1.0f};
  float y[] = {3.0f, 3.0f, 3.0f, 3.0f};
  PolyIOBinding io[] = {
      {"x", x},
      {"y", y},
  };

  /* Run several train steps and check loss decreases */
  float prev_loss = 1e10f;
  for (int step = 0; step < 50; step++) {
    float loss;
    int ret = poly_instance_train_step(inst, io, 2, &loss);
    ASSERT_INT_EQ(ret, 0);
    if (step > 0) ASSERT_TRUE(loss <= prev_loss + 1e-6f);
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
  for (int i = 0; i < 4; i++)
    w[i] = 0.5f;

  /* Configure Adam */
  poly_instance_set_optimizer(inst, POLY_OPTIM_ADAM, 0.05f, 0.9f, 0.999f, 1e-8f, 0.0f);

  float x[] = {1.0f, 1.0f, 1.0f, 1.0f};
  float y[] = {3.0f, 3.0f, 3.0f, 3.0f};
  PolyIOBinding io[] = {{"x", x}, {"y", y}};

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

static int test_find_instance_buf(PolyInstance *inst, const char *name) {
  for (int i = 0; i < poly_instance_buf_count(inst); i++)
    if (strcmp(poly_instance_buf_name(inst, i), name) == 0) return i;
  return -1;
}

static bool test_safetensors_has_name(PolySafetensorView *views, int n, const char *name) {
  for (int i = 0; i < n; i++)
    if (strcmp(views[i].name, name) == 0) return true;
  return false;
}

static void test_safetensors_free_views(PolySafetensorView *views, int n, char *metadata) {
  if (views) {
    for (int i = 0; i < n; i++)
      free(views[i].name);
  }
  free(views);
  free(metadata);
}

TEST(instance, adam_optimizer_state_is_named_checkpoint_state) {
  int ir_len = 0;
  uint8_t *ir = make_train_ir(4, &ir_len);
  PolyInstance *inst = poly_instance_from_ir(ir, ir_len, NULL, 0);
  ASSERT_NOT_NULL(inst);

  int64_t numel;
  float *w = poly_instance_param_data(inst, 0, &numel);
  ASSERT_INT_EQ(numel, 4);
  for (int i = 0; i < 4; i++)
    w[i] = 0.5f;

  ASSERT_INT_EQ(poly_instance_set_optimizer(inst, POLY_OPTIM_ADAM, 0.05f, 0.9f, 0.999f, 1e-8f, 0.0f), 0);

  float x[] = {1.0f, 1.0f, 1.0f, 1.0f};
  float y[] = {3.0f, 3.0f, 3.0f, 3.0f};
  PolyIOBinding io[] = {{"x", x}, {"y", y}};
  float loss = 0.0f;
  ASSERT_INT_EQ(poly_instance_train_step(inst, io, 2, &loss), 0);

  int b1_idx = test_find_instance_buf(inst, "optim.adam.b1_t");
  int b2_idx = test_find_instance_buf(inst, "optim.adam.b2_t");
  int m_idx = test_find_instance_buf(inst, "optim.adam.m.w");
  int v_idx = test_find_instance_buf(inst, "optim.adam.v.w");
  ASSERT_TRUE(b1_idx >= 0);
  ASSERT_TRUE(b2_idx >= 0);
  ASSERT_TRUE(m_idx >= 0);
  ASSERT_TRUE(v_idx >= 0);
  ASSERT_INT_EQ(poly_instance_buf_role(inst, b1_idx), POLY_ROLE_AUX);
  ASSERT_INT_EQ(poly_instance_buf_role(inst, m_idx), POLY_ROLE_AUX);

  int64_t n_b1 = 0;
  float *b1 = poly_instance_buf_data(inst, b1_idx, &n_b1);
  ASSERT_INT_EQ(n_b1, 1);
  ASSERT_FLOAT_EQ(b1[0], 0.9f, 1e-6f);
  int64_t n_m = 0;
  float *m = poly_instance_buf_data(inst, m_idx, &n_m);
  ASSERT_INT_EQ(n_m, 4);
  ASSERT_TRUE(fabsf(m[0]) > 0.0f);

  int weights_len = 0;
  uint8_t *weights = poly_instance_export_weights(inst, &weights_len);
  ASSERT_NOT_NULL(weights);
  int n_views = 0;
  char *metadata = NULL;
  PolySafetensorView *views = poly_safetensors_decode(weights, weights_len, &n_views, &metadata);
  ASSERT_NOT_NULL(views);
  ASSERT_TRUE(test_safetensors_has_name(views, n_views, "w"));
  ASSERT_TRUE(test_safetensors_has_name(views, n_views, "optim.adam.b1_t"));
  ASSERT_TRUE(test_safetensors_has_name(views, n_views, "optim.adam.b2_t"));
  ASSERT_TRUE(test_safetensors_has_name(views, n_views, "optim.adam.m.w"));
  ASSERT_TRUE(test_safetensors_has_name(views, n_views, "optim.adam.v.w"));
  test_safetensors_free_views(views, n_views, metadata);

  int model_only_len = 0;
  uint8_t *model_only =
      poly_instance_export_weights_ex(inst, &model_only_len, POLY_EXPORT_WEIGHTS_PARAMS);
  ASSERT_NOT_NULL(model_only);
  n_views = 0;
  metadata = NULL;
  views = poly_safetensors_decode(model_only, model_only_len, &n_views, &metadata);
  ASSERT_NOT_NULL(views);
  ASSERT_TRUE(test_safetensors_has_name(views, n_views, "w"));
  ASSERT_FALSE(test_safetensors_has_name(views, n_views, "optim.adam.b1_t"));
  ASSERT_FALSE(test_safetensors_has_name(views, n_views, "optim.adam.m.w"));
  test_safetensors_free_views(views, n_views, metadata);
  free(model_only);

  int ir2_len = 0;
  uint8_t *ir2 = poly_instance_export_ir(inst, &ir2_len);
  ASSERT_NOT_NULL(ir2);
  PolyInstance *restored = poly_instance_from_ir(ir2, ir2_len, weights, weights_len);
  ASSERT_NOT_NULL(restored);
  int rb1_idx = test_find_instance_buf(restored, "optim.adam.b1_t");
  ASSERT_TRUE(rb1_idx >= 0);
  float *rb1 = poly_instance_buf_data(restored, rb1_idx, NULL);
  ASSERT_FLOAT_EQ(rb1[0], 0.9f, 1e-6f);

  ASSERT_INT_EQ(
      poly_instance_set_optimizer(restored, POLY_OPTIM_ADAM, 0.05f, 0.9f, 0.999f, 1e-8f, 0.0f), 0
  );
  ASSERT_INT_EQ(poly_instance_train_step(restored, io, 2, &loss), 0);
  rb1_idx = test_find_instance_buf(restored, "optim.adam.b1_t");
  rb1 = poly_instance_buf_data(restored, rb1_idx, NULL);
  ASSERT_FLOAT_EQ(rb1[0], 0.81f, 1e-5f);

  poly_instance_free(restored);
  poly_instance_free(inst);
  free(ir2);
  free(weights);
  free(ir);
  PASS();
}

TEST(instance, sgd_momentum_state_is_named_checkpoint_state) {
  int ir_len = 0;
  uint8_t *ir = make_train_ir(4, &ir_len);
  PolyInstance *inst = poly_instance_from_ir(ir, ir_len, NULL, 0);
  ASSERT_NOT_NULL(inst);

  int64_t numel;
  float *w = poly_instance_param_data(inst, 0, &numel);
  ASSERT_INT_EQ(numel, 4);
  for (int i = 0; i < 4; i++)
    w[i] = 0.5f;

  ASSERT_INT_EQ(
      poly_instance_set_optimizer_ex(
          inst, POLY_OPTIM_SGD, 0.05f, 0.0f, 0.0f, 0.0f, 0.0f, 0.9f, false, false
      ),
      0
  );

  float x[] = {1.0f, 1.0f, 1.0f, 1.0f};
  float y[] = {3.0f, 3.0f, 3.0f, 3.0f};
  PolyIOBinding io[] = {{"x", x}, {"y", y}};
  float loss = 0.0f;
  ASSERT_INT_EQ(poly_instance_train_step(inst, io, 2, &loss), 0);

  int b_idx = test_find_instance_buf(inst, "optim.sgd.b.w");
  ASSERT_TRUE(b_idx >= 0);
  ASSERT_INT_EQ(poly_instance_buf_role(inst, b_idx), POLY_ROLE_AUX);

  int64_t n_b = 0;
  float *b = poly_instance_buf_data(inst, b_idx, &n_b);
  ASSERT_INT_EQ(n_b, 4);
  ASSERT_TRUE(fabsf(b[0]) > 0.0f);

  /* Make the reuse check strong: imported optimizer state should be consumed
   * as-is, not silently reinitialized to zero on the next train graph build. */
  for (int i = 0; i < 4; i++)
    b[i] = 7.0f;

  int weights_len = 0;
  uint8_t *weights = poly_instance_export_weights(inst, &weights_len);
  ASSERT_NOT_NULL(weights);
  int n_views = 0;
  char *metadata = NULL;
  PolySafetensorView *views = poly_safetensors_decode(weights, weights_len, &n_views, &metadata);
  ASSERT_NOT_NULL(views);
  ASSERT_TRUE(test_safetensors_has_name(views, n_views, "w"));
  ASSERT_TRUE(test_safetensors_has_name(views, n_views, "optim.sgd.b.w"));
  test_safetensors_free_views(views, n_views, metadata);

  int model_only_len = 0;
  uint8_t *model_only =
      poly_instance_export_weights_ex(inst, &model_only_len, POLY_EXPORT_WEIGHTS_PARAMS);
  ASSERT_NOT_NULL(model_only);
  n_views = 0;
  metadata = NULL;
  views = poly_safetensors_decode(model_only, model_only_len, &n_views, &metadata);
  ASSERT_NOT_NULL(views);
  ASSERT_TRUE(test_safetensors_has_name(views, n_views, "w"));
  ASSERT_FALSE(test_safetensors_has_name(views, n_views, "optim.sgd.b.w"));
  test_safetensors_free_views(views, n_views, metadata);
  free(model_only);

  int ir2_len = 0;
  uint8_t *ir2 = poly_instance_export_ir(inst, &ir2_len);
  ASSERT_NOT_NULL(ir2);
  PolyInstance *restored = poly_instance_from_ir(ir2, ir2_len, weights, weights_len);
  ASSERT_NOT_NULL(restored);
  int rb_idx = test_find_instance_buf(restored, "optim.sgd.b.w");
  ASSERT_TRUE(rb_idx >= 0);
  float *rb = poly_instance_buf_data(restored, rb_idx, NULL);
  ASSERT_FLOAT_EQ(rb[0], 7.0f, 1e-6f);

  ASSERT_INT_EQ(
      poly_instance_set_optimizer_ex(
          restored, POLY_OPTIM_SGD, 0.05f, 0.0f, 0.0f, 0.0f, 0.0f, 0.9f, false, false
      ),
      0
  );
  ASSERT_INT_EQ(poly_instance_train_step(restored, io, 2, &loss), 0);
  rb_idx = test_find_instance_buf(restored, "optim.sgd.b.w");
  rb = poly_instance_buf_data(restored, rb_idx, NULL);
  ASSERT_TRUE(rb[0] > 5.0f);

  poly_instance_free(restored);
  poly_instance_free(inst);
  free(ir2);
  free(weights);
  free(ir);
  PASS();
}

TEST(instance, inline_forward_copies_prefixed_weights) {
  int child_ir_len = 0;
  uint8_t *child_ir = make_train_ir(4, &child_ir_len);
  PolyInstance *child = poly_instance_from_ir(child_ir, child_ir_len, NULL, 0);
  ASSERT_NOT_NULL(child);

  int64_t n;
  float *cw = poly_instance_param_data(child, 0, &n);
  ASSERT_INT_EQ(n, 4);
  for (int i = 0; i < 4; i++)
    cw[i] = (float)(i + 2);

  PolyCtx *ctx = poly_ctx_new();
  PolyInstance *parent = poly_instance_new(ctx, NULL);
  ASSERT_NOT_NULL(parent);
  int64_t s[] = {4};
  PolyTensor *x_tensor = poly_instance_input(parent, "x", POLY_FLOAT32, s, 1);
  ASSERT_NOT_NULL(x_tensor);
  PolyUOp *x = poly_tensor_uop(x_tensor);

  PolyInstanceInlineBinding binds[] = {{"x", x}};
  PolyInstanceInlineOutput outs[2];
  int n_out = 0;
  ASSERT_INT_EQ(
      poly_instance_inline_entrypoint(
          ctx, child, "forward", "child.", binds, 1, false, outs, 2, &n_out
      ),
      0
  );
  ASSERT_INT_EQ(n_out, 1);
  ASSERT_STR_EQ(outs[0].name, "output");

  PolyUOp *child_w = poly_ctx_get(ctx, "%s", "child.w");
  ASSERT_NOT_NULL(child_w);
  PolyTensor *child_w_tensor =
      poly_tensor_create(ctx, child_w, POLY_TENSOR_VALUE, POLY_DEVICE_AUTO);
  ASSERT_NOT_NULL(child_w_tensor);
  poly_tensor_set_requires_grad(child_w_tensor, false);
  ASSERT_INT_EQ(poly_instance_state(parent, "child.w", child_w_tensor, 0), POLY_STATUS_OK);

  PolyUOp *one = poly_const_float(ctx, 1.0);
  PolyUOp *y = poly_alu2(ctx, POLY_OP_ADD, outs[0].uop, one);
  PolyTensor *out_tensor = poly_tensor_create(ctx, y, POLY_TENSOR_VALUE, POLY_DEVICE_AUTO);
  ASSERT_NOT_NULL(out_tensor);
  ASSERT_INT_EQ(poly_instance_output(parent, "output", out_tensor), POLY_STATUS_OK);

  const char *inputs[] = {"x"};
  const char *outputs[] = {"output"};
  ASSERT_INT_EQ(
      poly_instance_entrypoint(parent, "forward", inputs, 1, outputs, 1, NULL), POLY_STATUS_OK
  );
  ASSERT_INT_EQ(poly_instance_build(parent, NULL), POLY_STATUS_OK);
  ASSERT_INT_EQ(poly_instance_copy_prefixed_weights(parent, child, "child."), 0);

  float x_data[] = {1, 1, 1, 1};
  PolyIOBinding io[] = {{"x", x_data}};
  ASSERT_INT_EQ(poly_instance_forward(parent, io, 1), 0);

  float *out_data = poly_instance_buf_data_named(parent, "output", NULL);
  ASSERT_NOT_NULL(out_data);
  ASSERT_FLOAT_EQ(out_data[0], 3.0f, 1e-5f);
  ASSERT_FLOAT_EQ(out_data[1], 4.0f, 1e-5f);
  ASSERT_FLOAT_EQ(out_data[2], 5.0f, 1e-5f);
  ASSERT_FLOAT_EQ(out_data[3], 6.0f, 1e-5f);

  int child_w_idx = -1;
  for (int i = 0; i < poly_instance_param_count(parent); i++)
    if (strcmp(poly_instance_param_name(parent, i), "child.w") == 0) child_w_idx = i;
  ASSERT_TRUE(child_w_idx >= 0);
  ASSERT_TRUE(!poly_instance_param_trainable(parent, child_w_idx));

  poly_instance_free(parent);
  poly_ctx_destroy(ctx);
  poly_instance_free(child);
  free(child_ir);
  PASS();
}

TEST(instance, inline_frozen_submodel_not_updated_by_parent_train) {
  int child_ir_len = 0;
  uint8_t *child_ir = make_train_ir(4, &child_ir_len);
  PolyInstance *child = poly_instance_from_ir(child_ir, child_ir_len, NULL, 0);
  ASSERT_NOT_NULL(child);
  float *cw = poly_instance_param_data(child, 0, NULL);
  for (int i = 0; i < 4; i++)
    cw[i] = 2.0f;

  PolyCtx *ctx = poly_ctx_new();
  PolyInstance *parent = poly_instance_new(ctx, NULL);
  ASSERT_NOT_NULL(parent);
  int64_t s[] = {4};
  PolyTensor *x_tensor = poly_instance_input(parent, "x", POLY_FLOAT32, s, 1);
  PolyTensor *target_tensor = poly_instance_target(parent, "y", POLY_FLOAT32, s, 1);
  PolyTensor *head_tensor = poly_instance_param(parent, "head", POLY_FLOAT32, s, 1);
  ASSERT_NOT_NULL(x_tensor);
  ASSERT_NOT_NULL(target_tensor);
  ASSERT_NOT_NULL(head_tensor);
  PolyUOp *x = poly_tensor_uop(x_tensor);
  PolyUOp *target = poly_tensor_uop(target_tensor);
  PolyUOp *head = poly_tensor_uop(head_tensor);

  PolyInstanceInlineBinding binds[] = {{"x", x}};
  PolyInstanceInlineOutput outs[2];
  int n_out = 0;
  ASSERT_INT_EQ(
      poly_instance_inline_entrypoint(
          ctx, child, "forward", "enc.", binds, 1, false, outs, 2, &n_out
      ),
      0
  );
  ASSERT_INT_EQ(n_out, 1);

  PolyUOp *enc_w_uop = poly_ctx_get(ctx, "%s", "enc.w");
  ASSERT_NOT_NULL(enc_w_uop);
  PolyTensor *enc_w_tensor =
      poly_tensor_create(ctx, enc_w_uop, POLY_TENSOR_VALUE, POLY_DEVICE_AUTO);
  ASSERT_NOT_NULL(enc_w_tensor);
  poly_tensor_set_requires_grad(enc_w_tensor, false);
  ASSERT_INT_EQ(poly_instance_state(parent, "enc.w", enc_w_tensor, 0), POLY_STATUS_OK);

  PolyUOp *pred = poly_alu2(ctx, POLY_OP_MUL, outs[0].uop, head);
  PolyTensor *out_tensor = poly_tensor_create(ctx, pred, POLY_TENSOR_VALUE, POLY_DEVICE_AUTO);
  ASSERT_NOT_NULL(out_tensor);
  ASSERT_INT_EQ(poly_instance_output(parent, "output", out_tensor), POLY_STATUS_OK);

  PolyUOp *diff = poly_alu2(ctx, POLY_OP_ADD, pred, poly_alu1(ctx, POLY_OP_NEG, target));
  PolyUOp *sq = poly_alu2(ctx, POLY_OP_MUL, diff, diff);
  int64_t axes[] = {0};
  PolyUOp *loss_val = poly_alu2(
      ctx, POLY_OP_MUL, poly_reduce_axis(ctx, POLY_OP_ADD, sq, axes, 1), poly_const_float(ctx, 0.25)
  );
  PolyTensor *loss_tensor = poly_tensor_create(ctx, loss_val, POLY_TENSOR_VALUE, POLY_DEVICE_AUTO);
  ASSERT_NOT_NULL(loss_tensor);
  ASSERT_INT_EQ(poly_instance_output(parent, "loss", loss_tensor), POLY_STATUS_OK);

  const char *forward_inputs[] = {"x"};
  const char *forward_outputs[] = {"output"};
  ASSERT_INT_EQ(
      poly_instance_entrypoint(
          parent, "forward", forward_inputs, 1, forward_outputs, 1, NULL
      ),
      POLY_STATUS_OK
  );
  const char *loss_inputs[] = {"x", "y"};
  const char *loss_outputs[] = {"loss"};
  PolyEntrypointOptions loss_opts = {.objective = "loss"};
  ASSERT_INT_EQ(
      poly_instance_entrypoint(parent, "loss", loss_inputs, 2, loss_outputs, 1, &loss_opts),
      POLY_STATUS_OK
  );
  ASSERT_INT_EQ(poly_instance_build(parent, NULL), POLY_STATUS_OK);
  ASSERT_INT_EQ(poly_instance_copy_prefixed_weights(parent, child, "enc."), 0);

  for (int i = 0; i < poly_instance_param_count(parent); i++) {
    float *p = poly_instance_param_data(parent, i, NULL);
    if (strcmp(poly_instance_param_name(parent, i), "head") == 0) {
      for (int j = 0; j < 4; j++)
        p[j] = 1.0f;
    }
  }

  float x_data[] = {1, 1, 1, 1};
  float y_data[] = {6, 6, 6, 6};
  PolyIOBinding io[] = {{"x", x_data}, {"y", y_data}};
  ASSERT_INT_EQ(
      poly_instance_set_optimizer(parent, POLY_OPTIM_SGD, 0.05f, 0.0f, 0.0f, 0.0f, 0.0f), 0
  );

  float first_loss = 0.0f, last_loss = 0.0f;
  ASSERT_INT_EQ(poly_instance_train_step(parent, io, 2, &first_loss), 0);
  for (int step = 0; step < 10; step++)
    ASSERT_INT_EQ(poly_instance_train_step(parent, io, 2, &last_loss), 0);
  ASSERT_TRUE(last_loss < first_loss);

  int enc_idx = -1, head_idx = -1;
  for (int i = 0; i < poly_instance_param_count(parent); i++) {
    if (strcmp(poly_instance_param_name(parent, i), "enc.w") == 0) enc_idx = i;
    if (strcmp(poly_instance_param_name(parent, i), "head") == 0) head_idx = i;
  }
  ASSERT_TRUE(enc_idx >= 0);
  ASSERT_TRUE(head_idx >= 0);
  ASSERT_TRUE(!poly_instance_param_trainable(parent, enc_idx));
  ASSERT_TRUE(poly_instance_param_trainable(parent, head_idx));

  float *enc_w = poly_instance_param_data(parent, enc_idx, NULL);
  float *head_w = poly_instance_param_data(parent, head_idx, NULL);
  for (int i = 0; i < 4; i++)
    ASSERT_FLOAT_EQ(enc_w[i], 2.0f, 1e-6f);
  ASSERT_TRUE(head_w[0] > 1.0f);

  poly_instance_free(parent);
  poly_ctx_destroy(ctx);
  poly_instance_free(child);
  free(child_ir);
  PASS();
}

TEST(instance, null_safety) {
  ASSERT_INT_EQ(poly_instance_buf_count(NULL), 0);
  ASSERT_INT_EQ(poly_instance_param_count(NULL), 0);
  ASSERT_TRUE(poly_instance_buf_name(NULL, 0) == NULL);
  ASSERT_TRUE(poly_instance_param_name(NULL, 0) == NULL);
  ASSERT_TRUE(poly_instance_param_data(NULL, 0, NULL) == NULL);
  poly_instance_free(NULL); /* should not crash */
  PASS();
}

/* Phase 5: Backend-aware PolyInstance tests */

TEST(instance, call_basic) {
  int ir_len = 0;
  uint8_t *ir = make_add_ir(&ir_len);
  PolyInstance *inst = poly_instance_from_ir(ir, ir_len, NULL, 0);
  ASSERT_NOT_NULL(inst);

  float a_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
  float b_data[] = {10.0f, 20.0f, 30.0f, 40.0f};
  PolyIOBinding io[] = {{"a", a_data}, {"b", b_data}};

  /* Use poly_instance_call instead of poly_instance_forward */
  int ret = poly_instance_call(inst, "forward", io, 2);
  ASSERT_INT_EQ(ret, 0);

  int64_t numel;
  float *out = poly_instance_buf_data(inst, 2, &numel);
  ASSERT_NOT_NULL(out);
  ASSERT_TRUE(fabsf(out[0] - 11.0f) < 1e-5f);
  ASSERT_TRUE(fabsf(out[1] - 22.0f) < 1e-5f);
  ASSERT_TRUE(fabsf(out[2] - 33.0f) < 1e-5f);
  ASSERT_TRUE(fabsf(out[3] - 44.0f) < 1e-5f);

  poly_instance_free(inst);
  free(ir);
  PASS();
}

TEST(instance, set_device_interp) {
  int ir_len = 0;
  uint8_t *ir = make_add_ir(&ir_len);
  PolyInstance *inst = poly_instance_from_ir(ir, ir_len, NULL, 0);
  ASSERT_NOT_NULL(inst);

  /* Switch to interpreter */
  int ret = poly_instance_set_device(inst, POLY_DEVICE_INTERP);
  ASSERT_INT_EQ(ret, 0);

  float a_data[] = {5.0f, 6.0f, 7.0f, 8.0f};
  float b_data[] = {100.0f, 200.0f, 300.0f, 400.0f};
  PolyIOBinding io[] = {{"a", a_data}, {"b", b_data}};

  ret = poly_instance_forward(inst, io, 2);
  ASSERT_INT_EQ(ret, 0);

  int64_t numel;
  float *out = poly_instance_buf_data(inst, 2, &numel);
  ASSERT_NOT_NULL(out);
  ASSERT_TRUE(fabsf(out[0] - 105.0f) < 1e-5f);
  ASSERT_TRUE(fabsf(out[1] - 206.0f) < 1e-5f);
  ASSERT_TRUE(fabsf(out[2] - 307.0f) < 1e-5f);
  ASSERT_TRUE(fabsf(out[3] - 408.0f) < 1e-5f);

  poly_instance_free(inst);
  free(ir);
  PASS();
}

TEST(instance, cpu_vs_interp_forward) {
  int ir_len = 0;
  uint8_t *ir = make_add_ir(&ir_len);

  /* Run on CPU */
  PolyInstance *inst_cpu = poly_instance_from_ir(ir, ir_len, NULL, 0);
  ASSERT_NOT_NULL(inst_cpu);

  float a_data[] = {1.5f, 2.5f, 3.5f, 4.5f};
  float b_data[] = {0.1f, 0.2f, 0.3f, 0.4f};
  PolyIOBinding io[] = {{"a", a_data}, {"b", b_data}};

  ASSERT_INT_EQ(poly_instance_forward(inst_cpu, io, 2), 0);
  int64_t numel;
  float *cpu_out = poly_instance_buf_data(inst_cpu, 2, &numel);

  /* Run on INTERP */
  PolyInstance *inst_interp = poly_instance_from_ir(ir, ir_len, NULL, 0);
  ASSERT_NOT_NULL(inst_interp);
  ASSERT_INT_EQ(poly_instance_set_device(inst_interp, POLY_DEVICE_INTERP), 0);
  ASSERT_INT_EQ(poly_instance_forward(inst_interp, io, 2), 0);
  float *interp_out = poly_instance_buf_data(inst_interp, 2, &numel);

  /* Compare outputs */
  for (int i = 0; i < 4; i++)
    ASSERT_TRUE(fabsf(cpu_out[i] - interp_out[i]) < 1e-6f);

  poly_instance_free(inst_cpu);
  poly_instance_free(inst_interp);
  free(ir);
  PASS();
}

TEST(instance, cpu_vs_interp_train) {
  int ir_len = 0;
  uint8_t *ir = make_train_ir(4, &ir_len);

  float x[] = {1.0f, 1.0f, 1.0f, 1.0f};
  float y[] = {3.0f, 3.0f, 3.0f, 3.0f};
  PolyIOBinding io[] = {{"x", x}, {"y", y}};

  /* CPU training */
  PolyInstance *inst_cpu = poly_instance_from_ir(ir, ir_len, NULL, 0);
  ASSERT_NOT_NULL(inst_cpu);
  {
    int64_t n;
    float *w = poly_instance_param_data(inst_cpu, 0, &n);
    for (int i = 0; i < 4; i++)
      w[i] = 1.0f;
  }
  poly_instance_set_optimizer(inst_cpu, POLY_OPTIM_SGD, 0.05f, 0.0f, 0.0f, 0.0f, 0.0f);

  float cpu_losses[5];
  for (int s = 0; s < 5; s++) {
    ASSERT_INT_EQ(poly_instance_train_step(inst_cpu, io, 2, &cpu_losses[s]), 0);
  }

  /* INTERP training */
  PolyInstance *inst_interp = poly_instance_from_ir(ir, ir_len, NULL, 0);
  ASSERT_NOT_NULL(inst_interp);
  ASSERT_INT_EQ(poly_instance_set_device(inst_interp, POLY_DEVICE_INTERP), 0);
  {
    int64_t n;
    float *w = poly_instance_param_data(inst_interp, 0, &n);
    for (int i = 0; i < 4; i++)
      w[i] = 1.0f;
  }
  poly_instance_set_optimizer(inst_interp, POLY_OPTIM_SGD, 0.05f, 0.0f, 0.0f, 0.0f, 0.0f);

  float interp_losses[5];
  for (int s = 0; s < 5; s++) {
    ASSERT_INT_EQ(poly_instance_train_step(inst_interp, io, 2, &interp_losses[s]), 0);
  }

  /* Compare loss trajectories */
  for (int s = 0; s < 5; s++)
    ASSERT_TRUE(fabsf(cpu_losses[s] - interp_losses[s]) < 1e-4f);

  /* Both should decrease */
  ASSERT_TRUE(cpu_losses[4] < cpu_losses[0]);

  poly_instance_free(inst_cpu);
  poly_instance_free(inst_interp);
  free(ir);
  PASS();
}

TEST(instance, set_device_roundtrip) {
  int ir_len = 0;
  uint8_t *ir = make_add_ir(&ir_len);
  PolyInstance *inst = poly_instance_from_ir(ir, ir_len, NULL, 0);
  ASSERT_NOT_NULL(inst);

  float a[] = {1.0f, 2.0f, 3.0f, 4.0f};
  float b[] = {10.0f, 20.0f, 30.0f, 40.0f};
  PolyIOBinding io[] = {{"a", a}, {"b", b}};
  float expected[] = {11.0f, 22.0f, 33.0f, 44.0f};
  int64_t numel;

  /* Run on CPU */
  ASSERT_INT_EQ(poly_instance_forward(inst, io, 2), 0);
  float *out = poly_instance_buf_data(inst, 2, &numel);
  for (int i = 0; i < 4; i++)
    ASSERT_TRUE(fabsf(out[i] - expected[i]) < 1e-5f);

  /* Switch to INTERP and run */
  ASSERT_INT_EQ(poly_instance_set_device(inst, POLY_DEVICE_INTERP), 0);
  ASSERT_INT_EQ(poly_instance_forward(inst, io, 2), 0);
  out = poly_instance_buf_data(inst, 2, &numel);
  for (int i = 0; i < 4; i++)
    ASSERT_TRUE(fabsf(out[i] - expected[i]) < 1e-5f);

  /* Switch back to CPU (should hit exec cache) */
  ASSERT_INT_EQ(poly_instance_set_device(inst, POLY_DEVICE_CPU), 0);
  ASSERT_INT_EQ(poly_instance_forward(inst, io, 2), 0);
  out = poly_instance_buf_data(inst, 2, &numel);
  for (int i = 0; i < 4; i++)
    ASSERT_TRUE(fabsf(out[i] - expected[i]) < 1e-5f);

  poly_instance_free(inst);
  free(ir);
  PASS();
}

TEST(instance, set_device_unsupported) {
  int ir_len = 0;
  uint8_t *ir = make_add_ir(&ir_len);
  PolyInstance *inst = poly_instance_from_ir(ir, ir_len, NULL, 0);
  ASSERT_NOT_NULL(inst);

  /* CUDA: fails if not available, succeeds if available -- both are valid */
#ifdef POLY_HAS_CUDA
  if (!poly_cuda_available()) ASSERT_TRUE(poly_instance_set_device(inst, POLY_DEVICE_CUDA) < 0);
#else
  ASSERT_TRUE(poly_instance_set_device(inst, POLY_DEVICE_CUDA) < 0);
#endif

  /* set_device(NULL) */
  ASSERT_TRUE(poly_instance_set_device(NULL, POLY_DEVICE_CPU) < 0);

  poly_instance_free(inst);
  free(ir);
  PASS();
}
