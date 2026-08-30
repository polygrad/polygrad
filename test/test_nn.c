/*
 * test_nn.c -- Tests for nn convenience builders + frontend composed ops
 */

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "test_harness.h"
#include "../src/nn.h"
#include "../src/instance.h"
#include "../src/schedule/rangeify.h"
#include "../src/schedule/rangeify.h"
#include "../src/frontend.h"
#include "../src/codegen/codegen.h"
#include "../src/engine/schedule.h"

static int nn_param_index(PolyInstance *inst, const char *name) {
  int n = poly_instance_param_count(inst);
  for (int i = 0; i < n; i++) {
    const char *got = poly_instance_param_name(inst, i);
    if (got && strcmp(got, name) == 0) return i;
  }
  return -1;
}

/* Convenience builder tests */

TEST(nn, instance_linear_declares_params) {
  PolyCtx *ctx = poly_ctx_new();
  PolyInstance *inst = poly_instance_new(ctx, NULL);
  int64_t xs[] = {2, 4};
  PolyTensor *x_tensor = poly_instance_input(inst, "x", POLY_FLOAT32, xs, 2);
  ASSERT_NOT_NULL(x_tensor);

  PolyTensor *out_tensor = poly_instance_linear(inst, "fc1", x_tensor, 4, 8, true);
  ASSERT_NOT_NULL(out_tensor);
  ASSERT_NOT_NULL(poly_tensor_uop_logical(out_tensor));
  ASSERT_NOT_NULL(poly_tensor_uop_physical(out_tensor));
  ASSERT_INT_EQ(poly_tensor_uop_logical(out_tensor)->op, POLY_OP_ADD);
  ASSERT_INT_EQ(poly_tensor_uop_physical(out_tensor)->op, POLY_OP_ADD);
  ASSERT_INT_EQ(poly_instance_output(inst, "output", out_tensor), POLY_STATUS_OK);
  const char *inputs[] = {"x"};
  const char *outputs[] = {"output"};
  ASSERT_INT_EQ(
      poly_instance_entrypoint(inst, "forward", inputs, 1, outputs, 1, NULL), POLY_STATUS_OK
  );
  ASSERT_NOT_NULL(poly_tensor_uop_physical(out_tensor));
  ASSERT_INT_EQ(poly_instance_build(inst, NULL), POLY_STATUS_OK);

  int wi = nn_param_index(inst, "fc1.weight");
  int bi = nn_param_index(inst, "fc1.bias");
  ASSERT_TRUE(wi >= 0);
  ASSERT_TRUE(bi >= 0);
  int64_t shape[8];
  ASSERT_INT_EQ(poly_instance_param_shape(inst, wi, shape, 8), 2);
  ASSERT_INT_EQ(shape[0], 8);
  ASSERT_INT_EQ(shape[1], 4);
  ASSERT_TRUE(poly_instance_param_trainable(inst, wi));
  ASSERT_INT_EQ(poly_instance_param_shape(inst, bi, shape, 8), 1);
  ASSERT_INT_EQ(shape[0], 8);

  poly_instance_free(inst);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(nn, instance_linear_no_bias) {
  PolyCtx *ctx = poly_ctx_new();
  PolyInstance *inst = poly_instance_new(ctx, NULL);
  int64_t xs[] = {1, 4};
  PolyTensor *x_tensor = poly_instance_input(inst, "x", POLY_FLOAT32, xs, 2);
  ASSERT_NOT_NULL(x_tensor);

  PolyTensor *out_tensor = poly_instance_linear(inst, "fc", x_tensor, 4, 2, false);
  ASSERT_NOT_NULL(out_tensor);
  ASSERT_NOT_NULL(poly_tensor_uop_physical(out_tensor));
  ASSERT_INT_EQ(poly_tensor_uop_physical(out_tensor)->op, POLY_OP_REDUCE);
  ASSERT_INT_EQ(poly_instance_output(inst, "output", out_tensor), POLY_STATUS_OK);
  const char *inputs[] = {"x"};
  const char *outputs[] = {"output"};
  ASSERT_INT_EQ(
      poly_instance_entrypoint(inst, "forward", inputs, 1, outputs, 1, NULL), POLY_STATUS_OK
  );
  ASSERT_INT_EQ(poly_instance_build(inst, NULL), POLY_STATUS_OK);
  ASSERT_TRUE(nn_param_index(inst, "fc.weight") >= 0);
  ASSERT_INT_EQ(nn_param_index(inst, "fc.bias"), -1);

  poly_instance_free(inst);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(nn, instance_linear_duplicate_prefix_rejected) {
  PolyCtx *ctx = poly_ctx_new();
  PolyInstance *inst = poly_instance_new(ctx, NULL);
  int64_t xs[] = {1, 4};
  PolyTensor *x_tensor = poly_instance_input(inst, "x", POLY_FLOAT32, xs, 2);
  ASSERT_NOT_NULL(x_tensor);
  ASSERT_NOT_NULL(poly_instance_linear(inst, "shared", x_tensor, 4, 8, true));
  ASSERT_TRUE(poly_instance_linear(inst, "shared", x_tensor, 4, 8, true) == NULL);
  const PolyInstanceError *err = poly_instance_last_error(inst);
  ASSERT_TRUE(err && strstr(err->message, "duplicate binding") != NULL);

  poly_instance_free(inst);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(nn, nn_linear_e2e) {
  PolyCtx *ctx = poly_ctx_new();
  PolyInstance *inst = poly_instance_new(ctx, NULL);
  int64_t xs[] = {1, 2};
  PolyTensor *x_tensor = poly_instance_input(inst, "x", POLY_FLOAT32, xs, 2);
  ASSERT_NOT_NULL(x_tensor);

  PolyTensor *out_tensor = poly_instance_linear(inst, "fc", x_tensor, 2, 3, true);
  ASSERT_NOT_NULL(out_tensor);
  ASSERT_NOT_NULL(poly_tensor_uop_physical(out_tensor));
  ASSERT_INT_EQ(poly_instance_output(inst, "output", out_tensor), POLY_STATUS_OK);
  const char *inputs[] = {"x"};
  const char *outputs[] = {"output"};
  ASSERT_INT_EQ(
      poly_instance_entrypoint(inst, "forward", inputs, 1, outputs, 1, NULL), POLY_STATUS_OK
  );
  ASSERT_INT_EQ(poly_instance_build(inst, NULL), POLY_STATUS_OK);

  /* Set weights: W = [[1,0],[0,1],[1,1]], b = [0,0,0] */
  float *wd = poly_instance_buf_data_named(inst, "fc.weight", NULL);
  float w_init[] = {1, 0, 0, 1, 1, 1};
  memcpy(wd, w_init, sizeof(w_init));

  float *bd = poly_instance_buf_data_named(inst, "fc.bias", NULL);
  memset(bd, 0, 3 * sizeof(float));

  /* Execute: x = [2, 3], expect [2, 3, 5] */
  float x_data[] = {2.0f, 3.0f};
  PolyIOBinding io[] = {POLY_IO_BINDING_ARRAY("x", x_data, POLY_FLOAT32)};
  ASSERT_INT_EQ(poly_instance_call(inst, "forward", io, 1), 0);

  float *od = poly_instance_buf_data_named(inst, "output", NULL);
  ASSERT_FLOAT_EQ(od[0], 2.0f, 1e-5);
  ASSERT_FLOAT_EQ(od[1], 3.0f, 1e-5);
  ASSERT_FLOAT_EQ(od[2], 5.0f, 1e-5);

  poly_instance_free(inst);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(nn, instance_layernorm_declares_params) {
  PolyCtx *ctx = poly_ctx_new();
  PolyInstance *inst = poly_instance_new(ctx, NULL);
  int64_t xs[] = {2, 4};
  PolyTensor *x_tensor = poly_instance_input(inst, "x", POLY_FLOAT32, xs, 2);
  ASSERT_NOT_NULL(x_tensor);

  PolyTensor *out_tensor = poly_instance_layernorm(inst, "ln", x_tensor, 4, 1e-5);
  ASSERT_NOT_NULL(out_tensor);
  ASSERT_NOT_NULL(poly_tensor_uop_logical(out_tensor));
  ASSERT_NOT_NULL(poly_tensor_uop_physical(out_tensor));
  ASSERT_INT_EQ(poly_tensor_uop_logical(out_tensor)->op, POLY_OP_ADD);
  ASSERT_INT_EQ(poly_tensor_uop_physical(out_tensor)->op, POLY_OP_ADD);
  ASSERT_INT_EQ(poly_instance_output(inst, "output", out_tensor), POLY_STATUS_OK);
  const char *inputs[] = {"x"};
  const char *outputs[] = {"output"};
  ASSERT_INT_EQ(
      poly_instance_entrypoint(inst, "forward", inputs, 1, outputs, 1, NULL), POLY_STATUS_OK
  );
  ASSERT_INT_EQ(poly_instance_build(inst, NULL), POLY_STATUS_OK);
  ASSERT_TRUE(nn_param_index(inst, "ln.weight") >= 0);
  ASSERT_TRUE(nn_param_index(inst, "ln.bias") >= 0);

  poly_instance_free(inst);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(nn, instance_rmsnorm_declares_params) {
  PolyCtx *ctx = poly_ctx_new();
  PolyInstance *inst = poly_instance_new(ctx, NULL);
  int64_t xs[] = {2, 4};
  PolyTensor *x_tensor = poly_instance_input(inst, "x", POLY_FLOAT32, xs, 2);
  ASSERT_NOT_NULL(x_tensor);

  PolyTensor *out_tensor = poly_instance_rmsnorm(inst, "rms", x_tensor, 4, 1e-6);
  ASSERT_NOT_NULL(out_tensor);
  ASSERT_NOT_NULL(poly_tensor_uop_logical(out_tensor));
  ASSERT_NOT_NULL(poly_tensor_uop_physical(out_tensor));
  ASSERT_INT_EQ(poly_tensor_uop_logical(out_tensor)->op, POLY_OP_MUL);
  ASSERT_INT_EQ(poly_tensor_uop_physical(out_tensor)->op, POLY_OP_MUL);
  ASSERT_INT_EQ(poly_instance_output(inst, "output", out_tensor), POLY_STATUS_OK);
  const char *inputs[] = {"x"};
  const char *outputs[] = {"output"};
  ASSERT_INT_EQ(
      poly_instance_entrypoint(inst, "forward", inputs, 1, outputs, 1, NULL), POLY_STATUS_OK
  );
  ASSERT_INT_EQ(poly_instance_build(inst, NULL), POLY_STATUS_OK);
  ASSERT_TRUE(nn_param_index(inst, "rms.weight") >= 0);

  poly_instance_free(inst);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(nn, instance_embedding_declares_params) {
  PolyCtx *ctx = poly_ctx_new();
  PolyInstance *inst = poly_instance_new(ctx, NULL);
  int64_t ts[] = {3};
  PolyTensor *tok_tensor = poly_instance_input(inst, "tokens", POLY_INT32, ts, 1);
  ASSERT_NOT_NULL(tok_tensor);

  PolyTensor *out_tensor = poly_instance_embedding(inst, "emb", tok_tensor, 100, 64);
  ASSERT_NOT_NULL(out_tensor);
  ASSERT_NOT_NULL(poly_tensor_uop_logical(out_tensor));
  ASSERT_NOT_NULL(poly_tensor_uop_physical(out_tensor));
  ASSERT_INT_EQ(poly_tensor_uop_logical(out_tensor)->op, POLY_OP_REDUCE);
  ASSERT_INT_EQ(poly_tensor_uop_physical(out_tensor)->op, POLY_OP_REDUCE);
  ASSERT_INT_EQ(poly_instance_output(inst, "output", out_tensor), POLY_STATUS_OK);
  const char *inputs[] = {"tokens"};
  const char *outputs[] = {"output"};
  ASSERT_INT_EQ(
      poly_instance_entrypoint(inst, "forward", inputs, 1, outputs, 1, NULL), POLY_STATUS_OK
  );
  ASSERT_INT_EQ(poly_instance_build(inst, NULL), POLY_STATUS_OK);

  int wi = nn_param_index(inst, "emb.weight");
  ASSERT_TRUE(wi >= 0);
  int64_t shape[8];
  ASSERT_INT_EQ(poly_instance_param_shape(inst, wi, shape, 8), 2);
  ASSERT_INT_EQ(shape[0], 100);
  ASSERT_INT_EQ(shape[1], 64);

  poly_instance_free(inst);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(nn, embedding_e2e) {
  PolyCtx *ctx = poly_ctx_new();

  /* table: (4, 3) weight matrix */
  PolyUOp *table_buf = poly_buffer_f32(ctx, 12);
  int64_t table_shape[] = {4, 3};
  PolyUOp *table = poly_reshape(ctx, table_buf, table_shape, 2);

  /* indices: (2,) tokens */
  PolyUOp *idx_buf = poly_test_buffer(ctx, POLY_INT32, 2);
  int64_t idx_shape[] = {2};
  PolyUOp *indices = poly_reshape(ctx, idx_buf, idx_shape, 1);

  PolyUOp *result = poly_embedding_apply(ctx, indices, table);
  ASSERT_NOT_NULL(result);

  PolyUOp *out_buf = poly_buffer_f32(ctx, 6);
  PolyUOp *store = poly_test_store_to_buffer(ctx, out_buf, result);
  PolyUOp *sink = poly_sink1(ctx, store);

  float table_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
  int32_t idx_data[] = {0, 2};
  float out_data[6] = {0};

  /* poly_gather internally creates an arange const buffer --
   * const_registry fallback in build_slot_data_from_bindings should bind it */
  PolyTestBufferView bindings[] = {
      POLY_TEST_HOST_VIEW(out_buf, out_data),
      POLY_TEST_HOST_VIEW(idx_buf, idx_data),
      POLY_TEST_HOST_VIEW(table_buf, table_data),
  };
  int ret = poly_test_realize_buffer_views(ctx, sink, bindings, 3);
  ASSERT_INT_EQ(ret, 0);

  fprintf(
      stderr, "  embedding out: [%.1f %.1f %.1f | %.1f %.1f %.1f]\n", out_data[0], out_data[1],
      out_data[2], out_data[3], out_data[4], out_data[5]
  );
  ASSERT_FLOAT_EQ(out_data[0], 1.0f, 1e-4);
  ASSERT_FLOAT_EQ(out_data[1], 2.0f, 1e-4);
  ASSERT_FLOAT_EQ(out_data[2], 3.0f, 1e-4);
  ASSERT_FLOAT_EQ(out_data[3], 7.0f, 1e-4);
  ASSERT_FLOAT_EQ(out_data[4], 8.0f, 1e-4);
  ASSERT_FLOAT_EQ(out_data[5], 9.0f, 1e-4);

  poly_ctx_destroy(ctx);
  PASS();
}

/* Existing composed op tests (kept, no PolyTensor dependency) */

TEST(nn, matmul_invalid_shape_returns_null) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_reshape(ctx, poly_buffer_f32(ctx, 4), (int64_t[]){2, 2}, 2);
  PolyUOp *b = poly_reshape(ctx, poly_buffer_f32(ctx, 3), (int64_t[]){1, 3}, 2);

  PolyUOp *r = poly_dot(ctx, a, b);

  ASSERT_TRUE(r == NULL);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(nn, matmul_broadcast_batch_numeric) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a_buf = poly_buffer_f32(ctx, 8);
  PolyUOp *b_buf = poly_buffer_f32(ctx, 4);
  PolyUOp *a = poly_reshape(ctx, a_buf, (int64_t[]){2, 2, 2}, 3);
  PolyUOp *b = poly_reshape(ctx, b_buf, (int64_t[]){1, 2, 2}, 3);

  PolyUOp *r = poly_dot(ctx, a, b);
  ASSERT_NOT_NULL(r);
  PolyShape s = poly_uop_max_shape(ctx, r);
  ASSERT_INT_EQ(s.ndim, 3);
  ASSERT_INT_EQ(s.dims[0], 2);
  ASSERT_INT_EQ(s.dims[1], 2);
  ASSERT_INT_EQ(s.dims[2], 2);
  if (s.dims) free(s.dims);

  PolyUOp *out_buf = poly_buffer_f32(ctx, 8);
  PolyUOp *store = poly_test_store_to_buffer(ctx, out_buf, r);
  PolyUOp *sink = poly_sink1(ctx, store);

  float a_data[] = {1, 2, 3, 4, 5, 6, 7, 8};
  float b_data[] = {1, 10, 100, 1000};
  float out_data[8] = {0};
  float expected[] = {201, 2010, 403, 4030, 605, 6050, 807, 8070};
  PolyTestBufferView bindings[] = {
      POLY_TEST_HOST_VIEW(a_buf, a_data),
      POLY_TEST_HOST_VIEW(b_buf, b_data),
      POLY_TEST_HOST_VIEW(out_buf, out_data),
  };

  int ret = poly_test_realize_buffer_views(ctx, sink, bindings, 3);
  ASSERT_INT_EQ(ret, 0);
  for (int i = 0; i < 8; i++)
    ASSERT_FLOAT_EQ(out_data[i], expected[i], 1e-5f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(nn, matmul_64x64_upcast_lane_e2e) {
  PolyCtx *ctx = poly_ctx_new();
  const int N = 64;
  PolyUOp *a_buf = poly_buffer_f32(ctx, N * N);
  PolyUOp *b_buf = poly_buffer_f32(ctx, N * N);
  PolyUOp *out_buf = poly_buffer_f32(ctx, N * N);
  PolyUOp *a = poly_reshape(ctx, a_buf, (int64_t[]){N, N}, 2);
  PolyUOp *b = poly_reshape(ctx, b_buf, (int64_t[]){N, N}, 2);

  PolyUOp *r = poly_dot(ctx, a, b);
  ASSERT_NOT_NULL(r);
  PolyUOp *sink = poly_sink1(ctx, poly_test_store_to_buffer(ctx, out_buf, r));

  float *a_data = calloc((size_t)N * (size_t)N, sizeof(float));
  float *b_data = calloc((size_t)N * (size_t)N, sizeof(float));
  float *out_data = calloc((size_t)N * (size_t)N, sizeof(float));
  ASSERT_NOT_NULL(a_data);
  ASSERT_NOT_NULL(b_data);
  ASSERT_NOT_NULL(out_data);

  for (int i = 0; i < N * N; i++) {
    a_data[i] = (float)((i * 13) % 37 - 18) * 0.03125f;
    b_data[i] = (float)((i * 17) % 41 - 20) * 0.015625f;
  }

  PolyTestBufferView bindings[] = {
      POLY_TEST_HOST_VIEW(a_buf, a_data),
      POLY_TEST_HOST_VIEW(b_buf, b_data),
      POLY_TEST_HOST_VIEW(out_buf, out_data),
  };
  int ret = poly_test_realize_buffer_views(ctx, sink, bindings, 3);
  ASSERT_INT_EQ(ret, 0);

  /* 64x64 triggers tinygrad's CPU upcast+unroll matmul path. This catches
   * register-lane aliasing where four accumulators are broadcast across the
   * output vector instead of reading the sixteen distinct accumulator slots. */
  for (int row = 0; row < N; row++) {
    for (int col = 0; col < N; col++) {
      double expected = 0.0;
      for (int k = 0; k < N; k++)
        expected += (double)a_data[row * N + k] * (double)b_data[k * N + col];
      ASSERT_FLOAT_EQ(out_data[row * N + col], (float)expected, 1e-4f);
    }
  }

  free(a_data);
  free(b_data);
  free(out_data);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(nn, matmul_invalid_broadcast_returns_null) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_reshape(ctx, poly_buffer_f32(ctx, 24), (int64_t[]){2, 3, 4}, 3);
  PolyUOp *b = poly_reshape(ctx, poly_buffer_f32(ctx, 120), (int64_t[]){5, 4, 6}, 3);

  PolyUOp *r = poly_dot(ctx, a, b);

  ASSERT_TRUE(r == NULL);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(nn, cross_entropy_sparse_targets) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *logits_buf = poly_buffer_f32(ctx, 6);
  PolyUOp *target_buf = poly_buffer_f32(ctx, 2);
  PolyUOp *logits = poly_reshape(ctx, logits_buf, (int64_t[]){2, 3}, 2);
  PolyUOp *target = poly_reshape(ctx, target_buf, (int64_t[]){2}, 1);

  PolyUOp *loss = poly_cross_entropy(ctx, logits, target, 1);
  ASSERT_NOT_NULL(loss);

  PolyUOp *out_buf = poly_buffer_f32(ctx, 1);
  PolyUOp *store = poly_test_store_to_buffer(ctx, out_buf, loss);
  PolyUOp *sink = poly_sink1(ctx, store);

  float logits_data[] = {0, 0, 0, 0, 0, 0};
  float target_data[] = {0, 2};
  float out_data[] = {0};
  PolyTestBufferView bindings[] = {
      POLY_TEST_HOST_VIEW(logits_buf, logits_data),
      POLY_TEST_HOST_VIEW(target_buf, target_data),
      POLY_TEST_HOST_VIEW(out_buf, out_data),
  };

  int ret = poly_test_realize_buffer_views(ctx, sink, bindings, 3);
  ASSERT_INT_EQ(ret, 0);
  ASSERT_FLOAT_EQ(out_data[0], logf(3.0f), 1e-5f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(nn, cross_entropy_dense_targets) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *logits_buf = poly_buffer_f32(ctx, 6);
  PolyUOp *target_buf = poly_buffer_f32(ctx, 6);
  PolyUOp *logits = poly_reshape(ctx, logits_buf, (int64_t[]){2, 3}, 2);
  PolyUOp *target = poly_reshape(ctx, target_buf, (int64_t[]){2, 3}, 2);

  PolyUOp *loss = poly_cross_entropy(ctx, logits, target, 1);
  ASSERT_NOT_NULL(loss);

  PolyUOp *out_buf = poly_buffer_f32(ctx, 1);
  PolyUOp *store = poly_test_store_to_buffer(ctx, out_buf, loss);
  PolyUOp *sink = poly_sink1(ctx, store);

  float logits_data[] = {0, 0, 0, 0, 0, 0};
  float target_data[] = {1, 0, 0, 0, 0, 1};
  float out_data[] = {0};
  PolyTestBufferView bindings[] = {
      POLY_TEST_HOST_VIEW(logits_buf, logits_data),
      POLY_TEST_HOST_VIEW(target_buf, target_data),
      POLY_TEST_HOST_VIEW(out_buf, out_data),
  };

  int ret = poly_test_realize_buffer_views(ctx, sink, bindings, 3);
  ASSERT_INT_EQ(ret, 0);
  ASSERT_FLOAT_EQ(out_data[0], logf(3.0f), 1e-5f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(nn, cross_entropy_sparse_targets_non_last_axis) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *logits_buf = poly_buffer_f32(ctx, 12);
  PolyUOp *target_buf = poly_buffer_f32(ctx, 4);
  PolyUOp *logits = poly_reshape(ctx, logits_buf, (int64_t[]){2, 3, 2}, 3);
  PolyUOp *target = poly_reshape(ctx, target_buf, (int64_t[]){2, 2}, 2);

  PolyUOp *loss = poly_cross_entropy(ctx, logits, target, -2);
  ASSERT_NOT_NULL(loss);

  PolyUOp *out_buf = poly_buffer_f32(ctx, 1);
  PolyUOp *store = poly_test_store_to_buffer(ctx, out_buf, loss);
  PolyUOp *sink = poly_sink1(ctx, store);

  float logits_data[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  float target_data[] = {0, 2, 1, 0};
  float out_data[] = {0};
  PolyTestBufferView bindings[] = {
      POLY_TEST_HOST_VIEW(logits_buf, logits_data),
      POLY_TEST_HOST_VIEW(target_buf, target_data),
      POLY_TEST_HOST_VIEW(out_buf, out_data),
  };

  int ret = poly_test_realize_buffer_views(ctx, sink, bindings, 3);
  ASSERT_INT_EQ(ret, 0);
  ASSERT_FLOAT_EQ(out_data[0], logf(3.0f), 1e-5f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(nn, cross_entropy_dense_targets_non_last_axis) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *logits_buf = poly_buffer_f32(ctx, 12);
  PolyUOp *target_buf = poly_buffer_f32(ctx, 12);
  PolyUOp *logits = poly_reshape(ctx, logits_buf, (int64_t[]){2, 3, 2}, 3);
  PolyUOp *target = poly_reshape(ctx, target_buf, (int64_t[]){2, 3, 2}, 3);

  PolyUOp *loss = poly_cross_entropy(ctx, logits, target, 1);
  ASSERT_NOT_NULL(loss);

  PolyUOp *out_buf = poly_buffer_f32(ctx, 1);
  PolyUOp *store = poly_test_store_to_buffer(ctx, out_buf, loss);
  PolyUOp *sink = poly_sink1(ctx, store);

  float logits_data[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  float target_data[] = {
      1, 0, 0, 0, 0, 1,
      0, 1, 1, 0, 0, 0,
  };
  float out_data[] = {0};
  PolyTestBufferView bindings[] = {
      POLY_TEST_HOST_VIEW(logits_buf, logits_data),
      POLY_TEST_HOST_VIEW(target_buf, target_data),
      POLY_TEST_HOST_VIEW(out_buf, out_data),
  };

  int ret = poly_test_realize_buffer_views(ctx, sink, bindings, 3);
  ASSERT_INT_EQ(ret, 0);
  ASSERT_FLOAT_EQ(out_data[0], logf(3.0f), 1e-5f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(nn, cross_entropy_invalid_shape_returns_null) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *logits = poly_reshape(ctx, poly_buffer_f32(ctx, 6), (int64_t[]){2, 3}, 2);
  PolyUOp *target = poly_reshape(ctx, poly_buffer_f32(ctx, 4), (int64_t[]){2, 2}, 2);

  PolyUOp *loss = poly_cross_entropy(ctx, logits, target, 1);

  ASSERT_TRUE(loss == NULL);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(nn, log_softmax_non_last_axis_flat_buffer) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x_buf = poly_buffer_f32(ctx, 12);
  PolyUOp *x = poly_reshape(ctx, x_buf, (int64_t[]){2, 3, 2}, 3);

  PolyUOp *y = poly_log_softmax(ctx, x, 1);
  ASSERT_NOT_NULL(y);

  PolyUOp *out_buf = poly_buffer_f32(ctx, 12);
  PolyUOp *store = poly_test_store_to_buffer(ctx, out_buf, y);
  PolyUOp *sink = poly_sink1(ctx, store);

  float x_data[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  float out_data[12] = {0};
  PolyTestBufferView bindings[] = {
      POLY_TEST_HOST_VIEW(x_buf, x_data),
      POLY_TEST_HOST_VIEW(out_buf, out_data),
  };

  int ret = poly_test_realize_buffer_views(ctx, sink, bindings, 2);
  ASSERT_INT_EQ(ret, 0);
  for (int i = 0; i < 12; i++)
    ASSERT_FLOAT_EQ(out_data[i], -logf(3.0f), 1e-5f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(nn, layernorm_non_last_axis) {
  PolyCtx *ctx = poly_ctx_new();
  int64_t xs[] = {2, 3, 2};
  PolyUOp *x_buf = poly_test_buffer(ctx, POLY_FLOAT32, 12);
  PolyUOp *x = poly_reshape(ctx, x_buf, xs, 3);

  PolyUOp *y = poly_layernorm_apply(ctx, x, NULL, NULL, 1, 1e-5);
  ASSERT_NOT_NULL(y);
  PolyShape s = poly_uop_max_shape(ctx, y);
  ASSERT_INT_EQ(s.ndim, 3);
  ASSERT_INT_EQ(s.dims[0], 2);
  ASSERT_INT_EQ(s.dims[1], 3);
  ASSERT_INT_EQ(s.dims[2], 2);
  if (s.dims) free(s.dims);

  PolyUOp *out_buf = poly_test_buffer(ctx, POLY_FLOAT32, 12);
  PolyUOp *store = poly_test_store_to_buffer(ctx, out_buf, y);
  PolyUOp *sink = poly_sink1(ctx, store);

  /* All-zero input → layernorm output is 0/0 → NaN, but 0-0=0 so var=0.
   * Actually: (0-0)/sqrt(0+eps) = 0. Output should be 0. */
  float x_data[12] = {0};
  float out_data[12] = {0};
  PolyTestBufferView bindings[] = {
      POLY_TEST_HOST_VIEW(x_buf, x_data),
      POLY_TEST_HOST_VIEW(out_buf, out_data),
  };
  int ret = poly_test_realize_buffer_views(ctx, sink, bindings, 2);
  ASSERT_INT_EQ(ret, 0);

  for (int i = 0; i < 12; i++)
    ASSERT_FLOAT_EQ(out_data[i], 0.0f, 1e-5f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(nn, threefry_reference_vector_cpu) {
  /* Reference from JAX threefry2x32 primitive:
   * key0=1337, key1=0, x0=counters 0..19, x1=0.
   * This matches tinygrad's uint32 elementwise lowering semantics. */
  const uint32_t ref[20] = {2732499619u, 3322027265u, 2482432314u, 3871860445u, 3571867126u,
                            3019569655u, 2459680734u, 2731866067u, 986922480u,  1616040745u,
                            4238711754u, 3594775990u, 3046419939u, 3519108299u, 586160567u,
                            3928687287u, 4074505382u, 4210472430u, 4094881238u, 3306411770u};

  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *counter = poly_test_buffer(ctx, POLY_UINT32, 20);
  PolyUOp *key = poly_test_buffer(ctx, POLY_UINT32, 20);
  PolyUOp *out = poly_test_buffer(ctx, POLY_UINT32, 20);
  PolyUOp *thr = poly_uop2(ctx, POLY_OP_THREEFRY, POLY_UINT32, counter, key, poly_arg_none());
  PolyUOp *sink = poly_sink1(ctx, poly_test_store_to_buffer(ctx, out, thr));

  uint32_t counter_data[20], key_data[20], out_data[20];
  for (int i = 0; i < 20; i++) {
    counter_data[i] = (uint32_t)i;
    key_data[i] = 1337u;
    out_data[i] = 0u;
  }
  PolyTestBufferView binds[3] = {
      POLY_TEST_HOST_VIEW(counter, counter_data), POLY_TEST_HOST_VIEW(key, key_data),
      POLY_TEST_HOST_VIEW(out, out_data)
  };
  ASSERT_INT_EQ(poly_test_realize_buffer_views(ctx, sink, binds, 3), 0);
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
      2091981003842036387ull,  7289724007706926337ull,  5665610599518103866ull,
      1710967476831381213ull,  10223482319493553654ull, 17028600460329286135ull,
      17124603982541209566ull, 3105577853879908307ull,  2086892522412654064ull,
      3335679299718140713ull,  16051258882154404810ull, 6689751810128997814ull,
      14143017499898457571ull, 15433975895207007435ull, 585627789352245687ull,
      6751685475094430391ull,  13535415641566872742ull, 11824248585908041198ull,
      12094218887010381270ull, 17901838715724224250ull
  };

  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *counter = poly_test_buffer(ctx, POLY_UINT64, 20);
  PolyUOp *key = poly_test_buffer(ctx, POLY_UINT64, 20);
  PolyUOp *out = poly_test_buffer(ctx, POLY_UINT64, 20);
  PolyUOp *thr = poly_uop2(ctx, POLY_OP_THREEFRY, POLY_UINT64, counter, key, poly_arg_none());
  PolyUOp *sink = poly_sink1(ctx, poly_test_store_to_buffer(ctx, out, thr));

  uint64_t counter_data[20], key_data[20], out_data[20];
  for (int i = 0; i < 20; i++) {
    counter_data[i] = (uint64_t)i;
    key_data[i] = 1337ull;
    out_data[i] = 0ull;
  }
  PolyTestBufferView binds[3] = {
      POLY_TEST_HOST_VIEW(counter, counter_data), POLY_TEST_HOST_VIEW(key, key_data),
      POLY_TEST_HOST_VIEW(out, out_data)
  };
  ASSERT_INT_EQ(poly_test_realize_buffer_views(ctx, sink, binds, 3), 0);
  for (int i = 0; i < 20; i++) {
    if (out_data[i] != ref[i]) {
      FAIL(
          "out_data[%d]=%llu expected %llu", i, (unsigned long long)out_data[i],
          (unsigned long long)ref[i]
      );
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
  PolyUOp *sink = poly_sink1(ctx, poly_test_store_to_buffer(ctx, out, rn));

  float *a = calloc(2048, sizeof(float));
  float *b = calloc(2048, sizeof(float));
  ASSERT_NOT_NULL(a);
  ASSERT_NOT_NULL(b);
  PolyTestBufferView bind = POLY_TEST_HOST_VIEW(out, a);
  ASSERT_INT_EQ(poly_test_realize_buffer_views(ctx, sink, &bind, 1), 0);
  bind.handle.ptr = b;
  ASSERT_INT_EQ(poly_test_realize_buffer_views(ctx, sink, &bind, 1), 0);

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
  PolyUOp *ar_sum = poly_sum_reduce(ctx, ar, 0, 0);
  PolyUOp *buf0 = poly_buffer_f32(ctx, 1);

  float ar_out[1] = {0};
  PolyTestBufferView b0 = POLY_TEST_HOST_VIEW(buf0, ar_out);
  ASSERT_INT_EQ(
      poly_test_realize_buffer_views(ctx, poly_sink1(ctx, poly_store_val(ctx, buf0, ar_sum)), &b0, 1),
      0
  );
  ASSERT_FLOAT_EQ(ar_out[0], 10.0f, 1e-5);

  /* eye(3) has trace/sum 3 */
  PolyUOp *eye = poly_eye(ctx, 3);
  ASSERT_NOT_NULL(eye);
  PolyUOp *r1 = poly_sum_reduce(ctx, eye, 1, 1);
  PolyUOp *r2 = poly_sum_reduce(ctx, r1, 0, 0);
  PolyUOp *buf1 = poly_buffer_f32(ctx, 1);
  float eye_out[1] = {0};
  PolyTestBufferView b1 = POLY_TEST_HOST_VIEW(buf1, eye_out);
  ASSERT_INT_EQ(
      poly_test_realize_buffer_views(ctx, poly_sink1(ctx, poly_store_val(ctx, buf1, r2)), &b1, 1), 0
  );
  ASSERT_FLOAT_EQ(eye_out[0], 3.0f, 1e-5);

  /* tril/triu on ones(3,3): both sums are 6 */
  int64_t e_shape[2] = {3, 3};
  PolyUOp *ones = poly_full(ctx, e_shape, 2, 1.0);
  ASSERT_NOT_NULL(ones);
  PolyUOp *tl = poly_tril(ctx, ones, 0);
  PolyUOp *tu = poly_triu(ctx, ones, 0);
  ASSERT_NOT_NULL(tl);
  ASSERT_NOT_NULL(tu);
  PolyUOp *tl_s = poly_sum_reduce(ctx, poly_sum_reduce(ctx, tl, 1, 1), 0, 0);
  PolyUOp *tu_s = poly_sum_reduce(ctx, poly_sum_reduce(ctx, tu, 1, 1), 0, 0);
  PolyUOp *buf2 = poly_buffer_f32(ctx, 1);
  PolyUOp *buf3 = poly_buffer_f32(ctx, 1);
  float tri_out[2] = {0, 0};
  PolyUOp *sink = poly_sink_n(
      ctx, (PolyUOp *[]){poly_store_val(ctx, buf2, tl_s), poly_store_val(ctx, buf3, tu_s)}, 2
  );
  PolyTestBufferView binds[2] = {
      POLY_TEST_HOST_VIEW(buf2, &tri_out[0]), POLY_TEST_HOST_VIEW(buf3, &tri_out[1])
  };
  ASSERT_INT_EQ(poly_test_realize_buffer_views(ctx, sink, binds, 2), 0);
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
  PolyUOp *sink = poly_sink_n(
      ctx,
      (PolyUOp *[]){
          poly_store_val(ctx, o1, y1),
          poly_store_val(ctx, o2, y2),
      },
      2
  );
  float xv[1] = {0.2f}, out1[1] = {0}, out2[1] = {0};
  PolyTestBufferView binds[3] = {
      POLY_TEST_HOST_VIEW(x, xv), POLY_TEST_HOST_VIEW(o1, out1), POLY_TEST_HOST_VIEW(o2, out2)
  };
  ASSERT_INT_EQ(poly_test_realize_buffer_views(ctx, sink, binds, 3), 0);
  ASSERT_FLOAT_EQ(out1[0], log1pf(0.2f), 5e-3f);
  ASSERT_FLOAT_EQ(out2[0], expm1f(0.2f), 5e-3f);

  /* lgamma gradient check through direct autograd + realize. */
  PolyUOp *lg = poly_lgamma(ctx, x);
  int64_t axes[] = {0};
  PolyUOp *loss = poly_reduce_axis(ctx, POLY_OP_ADD, lg, axes, 1);
  PolyUOp *grad = poly_grad(ctx, loss, x);
  ASSERT_NOT_NULL(grad);

  float xin[1] = {3.2f}, lout[1] = {0}, gout[1] = {0};
  PolyUOp *loss_out = poly_buffer_f32(ctx, 1);
  PolyUOp *grad_out = poly_buffer_f32(ctx, 1);
  PolyUOp *grad_sink = poly_sink_n(
      ctx,
      (PolyUOp *[]){
          poly_store_val(ctx, loss_out, loss),
          poly_store_val(ctx, grad_out, grad),
      },
      2
  );
  PolyTestBufferView grad_bindings[] = {
      POLY_TEST_HOST_VIEW(x, xin),
      POLY_TEST_HOST_VIEW(loss_out, lout),
      POLY_TEST_HOST_VIEW(grad_out, gout),
  };
  ASSERT_INT_EQ(poly_test_realize_buffer_views(ctx, grad_sink, grad_bindings, 3), 0);
  ASSERT_FLOAT_EQ(lout[0], (float)lgamma(3.2), 5e-3f);
  double h = 1e-4;
  double fd = (lgamma(3.2 + h) - lgamma(3.2 - h)) / (2.0 * h);
  ASSERT_FLOAT_EQ(gout[0], (float)fd, 5e-2f);

  poly_ctx_destroy(ctx);
  PASS();
}

/* Special math correctness tests (baked reference constants) */

/* Helper: compile a scalar f32 function, evaluate at given input, return output */
static float eval_scalar_f32(PolyUOp *(*fn)(PolyCtx *, PolyUOp *), float xval) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_buffer_f32(ctx, 1);
  PolyUOp *y = fn(ctx, x);
  PolyUOp *out = poly_buffer_f32(ctx, 1);
  PolyUOp *sink = poly_sink1(ctx, poly_test_store_to_buffer(ctx, out, y));
  float inv = xval, result = 0;
  PolyTestBufferView binds[] = {POLY_TEST_HOST_VIEW(x, &inv), POLY_TEST_HOST_VIEW(out, &result)};
  int rc = poly_test_realize_buffer_views(ctx, sink, binds, 2);
  poly_ctx_destroy(ctx);
  return (rc == 0) ? result : NAN;
}

/* Helper: compile a scalar f64 function, evaluate at given input, return output */
static double eval_scalar_f64(PolyUOp *(*fn)(PolyCtx *, PolyUOp *), double xval) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_buffer_f64(ctx, 1);
  PolyUOp *y = fn(ctx, x);
  PolyUOp *out = poly_buffer_f64(ctx, 1);
  PolyUOp *sink = poly_sink1(ctx, poly_test_store_to_buffer(ctx, out, y));
  double inv = xval, result = 0.0;
  PolyTestBufferView binds[] = {POLY_TEST_HOST_VIEW(x, &inv), POLY_TEST_HOST_VIEW(out, &result)};
  int rc = poly_test_realize_buffer_views(ctx, sink, binds, 2);
  poly_ctx_destroy(ctx);
  return (rc == 0) ? result : (double)NAN;
}

TEST(nn, special_math_erf) {
  /* Reference: A&S Table 7.1, DLMF 7.2.1 */
  struct {
    float x;
    float ref;
  } cases[] = {
      {0.0f, 0.0f},          {0.5f, 0.5204998778f}, {1.0f, 0.8427007929f},
      {2.0f, 0.9953222650f}, {3.0f, 0.9999779095f}, {-1.0f, -0.8427007929f},
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
  struct {
    float x;
    float ref;
    float tol;
  } cases[] = {
      {0.0f, 1.0f, 1e-4f},
      {1.0f, 0.15729920706f, 1e-4f},
      {2.0f, 0.00467773498f, 1e-5f},
      {3.0f, 2.20904970e-5f, 5e-6f},
      {5.0f, 1.53745979e-12f, 1e-12f},
      /* Negative: erfc(-x) = 2 - erfc(x) */
      {-1.0f, 1.84270079294f, 1e-4f},
  };
  for (int i = 0; i < 6; i++) {
    float got = eval_scalar_f32(poly_erfc, cases[i].x);
    ASSERT_TRUE(got >= 0.0f);
    /* For very small tail values, check relative error instead */
    if (cases[i].ref < 1e-4f && cases[i].ref > 0.0f) {
      float rel_err = fabsf(got - cases[i].ref) / cases[i].ref;
      ASSERT_TRUE(rel_err < 0.5f); /* within 50% relative error for f32 tails */
    } else {
      ASSERT_FLOAT_EQ(got, cases[i].ref, cases[i].tol);
    }
  }
  PASS();
}

TEST(nn, special_math_erfinv) {
  /* Reference: erfinv values from Wolfram Alpha */
  struct {
    float x;
    float ref;
  } cases[] = {
      {0.0f, 0.0f},          {0.5f, 0.4769362762f},  {-0.5f, -0.4769362762f},
      {0.9f, 1.1630871536f}, {0.99f, 1.8213863677f}, {0.9999f, 2.7510639058f},
  };
  for (int i = 0; i < 6; i++) {
    float got = eval_scalar_f32(poly_erfinv, cases[i].x);
    /* Winitzki approximation: ~0.35% max relative error */
    float tol = 0.01f + 0.005f * fabsf(cases[i].ref);
    ASSERT_FLOAT_EQ(got, cases[i].ref, tol);
  }

  /* Roundtrip: erfinv(erf(x)) ~= x */
  float roundtrip_x[] = {-2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f};
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
  struct {
    float p;
    float ref;
    float tol;
  } cases[] = {
      {0.5f, 0.0f, 1e-4f},
      {0.75f, 0.6744897502f, 0.01f},
      {0.9f, 1.2815515655f, 0.02f},
      {0.99f, 2.3263478740f, 0.05f},
      /* Tail tests -- PPL-critical */
      {1e-3f, -3.0902323062f, 0.1f},
      {1e-6f, -4.7534243060f, 0.2f},
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
  struct {
    float x;
    float ref;
  } cases[] = {
      {1.0f, -0.5772156649f}, /* Euler-Mascheroni */
      {2.0f, 0.4227843351f},  {0.5f, -1.9635100260f}, {5.0f, 1.5061176685f}, {10.0f, 2.2517525890f},
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
  struct {
    float x;
    float ref;
    float tol;
  } cases[] = {
      {1.0f, 0.0f, 5e-4f},
      {2.0f, 0.0f, 5e-4f},
      {0.5f, 0.5723649429f, 1e-3f}, /* ln(sqrt(pi)) */
      {3.5f, 1.2009736024f, 1e-3f},
      {5.0f, 3.1780538303f, 5e-3f},
      {10.0f, 12.8018274801f, 0.02f},
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
  PolyUOp *sink = poly_sink_n(
      ctx,
      (PolyUOp *[]){
          poly_store_val(ctx, o1, y1),
          poly_store_val(ctx, o2, y2),
      },
      2
  );

  struct {
    float x;
    float ref_log1p;
    float ref_expm1;
  } cases[] = {
      {0.0f, 0.0f, 0.0f},
      {1.0f, 0.6931471806f, 1.7182818285f},
      {0.2f, 0.1823215568f, 0.2214027582f},
      {1e-6f, 1e-6f, 1e-6f}, /* near-zero: should NOT be 0 */
  };
  for (int i = 0; i < 4; i++) {
    float xv = cases[i].x, out1 = 0, out2 = 0;
    PolyTestBufferView binds[] = {
        POLY_TEST_HOST_VIEW(x, &xv), POLY_TEST_HOST_VIEW(o1, &out1), POLY_TEST_HOST_VIEW(o2, &out2)
    };
    ASSERT_INT_EQ(poly_test_realize_buffer_views(ctx, sink, binds, 3), 0);
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
    PolyUOp *y = poly_logsumexp(ctx, x, 0, 0);
    PolyUOp *out = poly_buffer_f32(ctx, 1);
    PolyUOp *sink = poly_sink1(ctx, poly_test_store_to_buffer(ctx, out, y));
    float xv[] = {0.0f, 0.0f}, result = 0;
    PolyTestBufferView binds[] = {POLY_TEST_HOST_VIEW(x, xv), POLY_TEST_HOST_VIEW(out, &result)};
    ASSERT_INT_EQ(poly_test_realize_buffer_views(ctx, sink, binds, 2), 0);
    ASSERT_FLOAT_EQ(result, 0.6931471806f, 1e-4f); /* ln(2) */
  }

  /* Case 2: overflow stability -- logsumexp([1000, 1001]) */
  {
    PolyUOp *x2 = poly_buffer_f32(ctx, 2);
    PolyUOp *y2 = poly_logsumexp(ctx, x2, 0, 0);
    PolyUOp *out2 = poly_buffer_f32(ctx, 1);
    PolyUOp *sink2 = poly_sink1(ctx, poly_store_val(ctx, out2, y2));
    float xv2[] = {1000.0f, 1001.0f}, result2 = 0;
    PolyTestBufferView binds2[] = {POLY_TEST_HOST_VIEW(x2, xv2), POLY_TEST_HOST_VIEW(out2, &result2)};
    ASSERT_INT_EQ(poly_test_realize_buffer_views(ctx, sink2, binds2, 2), 0);
    /* Expected: 1001 + ln(1 + e^-1) = 1001.3133 */
    ASSERT_TRUE(isfinite(result2));
    ASSERT_FLOAT_EQ(result2, 1001.3133f, 0.01f);
  }

  /* Case 3: underflow stability -- logsumexp([-1000, -999]) */
  {
    PolyUOp *x3 = poly_buffer_f32(ctx, 2);
    PolyUOp *y3 = poly_logsumexp(ctx, x3, 0, 0);
    PolyUOp *out3 = poly_buffer_f32(ctx, 1);
    PolyUOp *sink3 = poly_sink1(ctx, poly_store_val(ctx, out3, y3));
    float xv3[] = {-1000.0f, -999.0f}, result3 = 0;
    PolyTestBufferView binds3[] = {POLY_TEST_HOST_VIEW(x3, xv3), POLY_TEST_HOST_VIEW(out3, &result3)};
    ASSERT_INT_EQ(poly_test_realize_buffer_views(ctx, sink3, binds3, 2), 0);
    /* Expected: -999 + ln(1 + e^-1) = -998.6867 */
    ASSERT_TRUE(isfinite(result3));
    ASSERT_FLOAT_EQ(result3, -998.6867f, 0.01f);
  }

  poly_ctx_destroy(ctx);
  PASS();
}

/* f64 accuracy tests — verify dtype-correct constants */

TEST(nn, special_math_f64_lgamma) {
  /* lgamma() is ISO C99 <math.h>; use it as reference for f64 accuracy. */
  struct {
    double x;
    double tol;
  } cases[] = {
      {1.0, 1e-12}, /* lgamma(1) = 0 exactly */
      {2.0, 1e-12}, /* lgamma(2) = 0 exactly */
      {0.5, 1e-12}, /* ln(sqrt(pi)) */
      {1.5, 1e-12}, {10.0, 1e-10},
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
  struct {
    double x;
    double ref;
    double tol;
  } cases[] = {
      {1.0, -0.5772156649015328606, 1e-7}, /* Euler-Mascheroni */
      {2.0, 0.4227843350984671394, 1e-7}, /* 1 - γ */
      {5.0, 1.5061176684318004727, 1e-7}, /* precomputed */
  };
  for (int i = 0; i < 3; i++) {
    double got = eval_scalar_f64(poly_digamma, cases[i].x);
    double err = fabs(got - cases[i].ref);
    ASSERT_TRUE(err < cases[i].tol);
  }
  /* Recurrence check: ψ(x+1) = ψ(x) + 1/x */
  double psi15 = eval_scalar_f64(poly_digamma, 1.5);
  double psi25 = eval_scalar_f64(poly_digamma, 2.5);
  ASSERT_TRUE(fabs(psi25 - (psi15 + 1.0 / 1.5)) < 1e-10);
  PASS();
}

TEST(nn, special_math_f64_log1p) {
  /* log1p() is ISO C99 <math.h> */
  struct {
    double x;
    double tol;
  } cases[] = {
      {1e-10, 1e-18}, /* tiny x: f32 loses most digits */
      {1.0, 1e-14}, /* ln(2) */
      {0.2, 1e-13},
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
  struct {
    double x;
    double tol;
  } cases[] = {
      {1e-10, 1e-18}, /* tiny x: f32 loses most digits */
      {1.0, 1e-14}, /* e - 1 */
      {0.2, 1e-13},
  };
  for (int i = 0; i < 3; i++) {
    double got = eval_scalar_f64(poly_expm1, cases[i].x);
    double ref = expm1(cases[i].x);
    double err = fabs(got - ref);
    ASSERT_TRUE(err < cases[i].tol);
  }
  PASS();
}

/* C5 primitive tests */

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

  PolyUOp *grad = poly_grad(ctx, loss, x);
  ASSERT_NOT_NULL(grad);

  float xv = 3.0f, lout = 0, gout = 0;
  PolyUOp *loss_out = poly_buffer_f32(ctx, 1);
  PolyUOp *grad_out = poly_buffer_f32(ctx, 1);
  PolyUOp *grad_sink = poly_sink_n(
      ctx,
      (PolyUOp *[]){
          poly_store_val(ctx, loss_out, loss),
          poly_store_val(ctx, grad_out, grad),
      },
      2
  );
  PolyTestBufferView bindings[] = {
      POLY_TEST_HOST_VIEW(x, &xv),
      POLY_TEST_HOST_VIEW(loss_out, &lout),
      POLY_TEST_HOST_VIEW(grad_out, &gout),
  };
  ASSERT_INT_EQ(poly_test_realize_buffer_views(ctx, grad_sink, bindings, 3), 0);
  /* Forward: 3^2 + 3^2 = 18 */
  ASSERT_FLOAT_EQ(lout, 18.0f, 1e-3f);
  /* Gradient: 2*3 = 6 (NOT 2*3 + 2*3 = 12) */
  ASSERT_FLOAT_EQ(gout, 6.0f, 1e-3f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(nn, c5_linspace) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *ls = poly_linspace(ctx, 0.0, 1.0, 5);
  ASSERT_NOT_NULL(ls);
  PolyUOp *out = poly_buffer_f32(ctx, 5);
  PolyUOp *sink = poly_sink1(ctx, poly_test_store_to_buffer(ctx, out, ls));
  float result[5] = {0};
  PolyTestBufferView binds[] = {POLY_TEST_HOST_VIEW(out, result)};
  ASSERT_INT_EQ(poly_test_realize_buffer_views(ctx, sink, binds, 1), 0);
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
  PolyUOp *sink = poly_sink1(ctx, poly_test_store_to_buffer(ctx, out, f));
  float result[4] = {0};
  PolyTestBufferView binds[] = {POLY_TEST_HOST_VIEW(out, result)};
  ASSERT_INT_EQ(poly_test_realize_buffer_views(ctx, sink, binds, 1), 0);
  for (int i = 0; i < 4; i++) {
    ASSERT_FLOAT_EQ(result[i], 3.14f, 1e-5f);
  }
  poly_ctx_destroy(ctx);
  PASS();
}

/* C2c: RNG determinism + cross-backend parity */

TEST(nn, c2c_rand_bitpattern_8) {
  /* Current Tinygrad 2026-08-22/a9069c177a9d Tensor._next_counter and
   * RandMixin._rand use storage-backed uint32 seed/counter Tensors, update the
   * counter through AFTER/STORE, and emit two packed THREEFRY operations. */
  const uint32_t expected_bits[8] = {
      UINT32_C(0x3efa31a0), UINT32_C(0x3eb22b7c), UINT32_C(0x3f28c97e),
      UINT32_C(0x3f22effe), UINT32_C(0x3ef13c94), UINT32_C(0x3e10dd30),
      UINT32_C(0x3e8e61ec), UINT32_C(0x3d4c9dc0),
  };

  PolyCtx *ctx = poly_ctx_new();
  int64_t shape[1] = {8};
  PolyUOp *r = poly_rand(ctx, shape, 1, 1337u);
  ASSERT_NOT_NULL(r);
  int n_topo = 0, threefry_u64 = 0, after = 0, store = 0, copy = 0;
  PolyUOp **topo = poly_toposort_alloc(ctx, r, &n_topo);
  ASSERT_NOT_NULL(topo);
  for (int i = 0; i < n_topo; i++) {
    if (topo[i]->op == POLY_OP_THREEFRY && poly_dtype_eq(topo[i]->dtype, POLY_UINT64))
      threefry_u64++;
    if (topo[i]->op == POLY_OP_AFTER) after++;
    if (topo[i]->op == POLY_OP_STORE) store++;
    if (topo[i]->op == POLY_OP_COPY) copy++;
  }
  ASSERT_INT_EQ(threefry_u64, 2);
  ASSERT_INT_EQ(after, 1);
  ASSERT_INT_EQ(store, 1);
  ASSERT_INT_EQ(copy, 2);
  poly_toposort_free(topo);
  PolyUOp *out = poly_buffer_f32(ctx, 8);
  PolyUOp *sink = poly_sink1(ctx, poly_test_store_to_buffer(ctx, out, r));
  float result[8] = {0};
  PolyTestBufferView bind = POLY_TEST_HOST_VIEW(out, result);
  ASSERT_INT_EQ(poly_test_realize_buffer_views(ctx, sink, &bind, 1), 0);
  if (memcmp(result, expected_bits, sizeof(expected_bits)) != 0) {
    for (int i = 0; i < 8; i++) {
      uint32_t got = 0;
      memcpy(&got, &result[i], sizeof(got));
      if (got != expected_bits[i])
        fprintf(stderr, "  [%d] got 0x%08x expected 0x%08x\n", i, got, expected_bits[i]);
    }
    FAIL("poly_rand bitpattern mismatch (memcmp)");
  }
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(nn, c2c_rand_optimized_small_bitpattern) {
  /* Current public Tensor.rand seed 0 reference. Storage identity prevents
   * literal THREEFRY folding under the optimized pipeline. */
  const uint32_t expected_bits[4] = {
      UINT32_C(0x3f53fc92), UINT32_C(0x3f1f7328),
      UINT32_C(0x3f0eaa8c), UINT32_C(0x3f165e0a),
  };

  PolyCtx *ctx = poly_ctx_new();
  int64_t shape[1] = {4};
  PolyUOp *r = poly_rand(ctx, shape, 1, 0u);
  ASSERT_NOT_NULL(r);
  PolyUOp *realized = NULL;
  PolyVarBinding *vars = NULL;
  int n_vars = 0;
  /* Tinygrad 2026-08-22/a9069c177a9d tensor.py:405-417 crosses
   * transform_to_call before scheduling a Tensor value. That callify pass
   * materializes this scalar CONTIGUOUS key as the first of two CALLs. */
  PolyUOp *linear =
      poly_linear_with_vars(ctx, &r, 1, &realized, &vars, &n_vars);
  ASSERT_NOT_NULL(linear);
  ASSERT_NOT_NULL(realized);
  float result[4] = {0};
  int run_rc = poly_run_linear(
      ctx, linear, vars, n_vars, NULL, 0, true, false, false
  );
  if (run_rc == 0)
    run_rc = poly_buffer_read(ctx, realized, result, sizeof(result));
  bool exact = memcmp(result, expected_bits, sizeof(expected_bits)) == 0;
  bool in_range = true;
  for (int i = 0; i < 4; i++)
    if (!(result[i] >= 0.0f && result[i] < 1.0f)) in_range = false;
  free(vars);
  poly_ctx_destroy(ctx);

  ASSERT_INT_EQ(run_rc, 0);
  ASSERT_TRUE(exact);
  ASSERT_TRUE(in_range);
  PASS();
}

TEST(nn, c2c_rand_seed_mixing) {
  /* Tensor._next_counter stores the Python seed in a uint32 Tensor. Current
   * Tinygrad therefore truncates explicit seed values to their low word. */
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
    PolyTestBufferView b = POLY_TEST_HOST_VIEW(o, all[s]);
    ASSERT_INT_EQ(poly_test_realize_buffer_views(ctx, sk, &b, 1), 0);
  }
  ASSERT_TRUE(memcmp(r0, r1, sizeof(r0)) != 0);
  ASSERT_TRUE(memcmp(r0, rhi, sizeof(r0)) == 0);
  ASSERT_TRUE(memcmp(r1, rboth, sizeof(r1)) == 0);
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
  PolyUOp *sink = poly_sink1(ctx, poly_test_store_to_buffer(ctx, out, r));
  float *buf = calloc(4096, sizeof(float));
  PolyTestBufferView bind = POLY_TEST_HOST_VIEW(out, buf);
  ASSERT_INT_EQ(poly_test_realize_buffer_views(ctx, sink, &bind, 1), 0);

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
  PolyUOp *sink = poly_sink1(ctx, poly_test_store_to_buffer(ctx, out, rn));
  float *buf = calloc(8192, sizeof(float));
  PolyTestBufferView bind = POLY_TEST_HOST_VIEW(out, buf);
  ASSERT_INT_EQ(poly_test_realize_buffer_views(ctx, sink, &bind, 1), 0);

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
  PolyTestBufferView bind1 = POLY_TEST_HOST_VIEW(o1, a);
  ASSERT_INT_EQ(poly_test_realize_buffer_views(ctx1, s1, &bind1, 1), 0);
  bind1.handle.ptr = b;
  ASSERT_INT_EQ(poly_test_realize_buffer_views(ctx1, s1, &bind1, 1), 0);
  if (memcmp(a, b, sizeof(a)) != 0) FAIL("same graph replay not deterministic");

  /* Case 2: two separately built graphs, different contexts */
  PolyCtx *ctx2 = poly_ctx_new();
  PolyUOp *r2 = poly_rand(ctx2, shape, 1, 12345u);
  PolyUOp *o2 = poly_buffer_f32(ctx2, 32);
  PolyUOp *s2 = poly_sink1(ctx2, poly_store_val(ctx2, o2, r2));
  float c[32] = {0};
  PolyTestBufferView bind2 = POLY_TEST_HOST_VIEW(o2, c);
  ASSERT_INT_EQ(poly_test_realize_buffer_views(ctx2, s2, &bind2, 1), 0);
  if (memcmp(a, c, sizeof(a)) != 0) FAIL("separate graphs with same seed not deterministic");

  poly_ctx_destroy(ctx1);
  poly_ctx_destroy(ctx2);
  PASS();
}

TEST(nn, c2c_threefry_lowered_in_compiled_kernel) {
  /* Verify THREEFRY is fully decomposed in the real compile path (not just
   * isolated rewrite helpers). Build a THREEFRY kernel, run full_rewrite_to_sink
   * with has_threefry=false, then scan the linearized output for residual ops. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *counter = poly_test_buffer(ctx, POLY_UINT32, 8);
  PolyUOp *key = poly_test_buffer(ctx, POLY_UINT32, 8);
  PolyUOp *out = poly_test_buffer(ctx, POLY_UINT32, 8);
  PolyUOp *thr = poly_uop2(ctx, POLY_OP_THREEFRY, POLY_UINT32, counter, key, poly_arg_none());
  PolyUOp *store = poly_test_store_to_buffer(ctx, out, thr);
  PolyUOp *sink = poly_sink1(ctx, store);

  /* Schedule + rewrite with has_threefry=false */
  PolyUOp *linear = poly_test_create_linear(ctx, sink);
  ASSERT_NOT_NULL(linear);
  ASSERT_TRUE(linear->n_src >= 1);
  PolyRewriteOpts opts = {0};
  opts.caps.has_threefry = false;
  opts.caps.has_mulacc = false;
  PolyUOp *rewritten =
      poly_full_rewrite_to_sink_ex(ctx, poly_test_linear_call_body(linear, 0), opts);
  ASSERT_NOT_NULL(rewritten);

  /* Tinygrad linearizes the already rewritten sink at this stage. */
  int n_uops = 0;
  PolyUOp **uops = poly_do_linearize(ctx, rewritten, &n_uops);
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
  PolyTestBufferView binds[3] = {
      POLY_TEST_HOST_VIEW(counter, counter_data), POLY_TEST_HOST_VIEW(key, key_data),
      POLY_TEST_HOST_VIEW(out, out_data)
  };
  ASSERT_INT_EQ(poly_test_realize_buffer_views(ctx, sink, binds, 3), 0);
  /* Verify we got non-zero output (THREEFRY actually ran) */
  int any_nonzero = 0;
  for (int i = 0; i < 8; i++) {
    if (out_data[i] != 0) any_nonzero = 1;
  }
  ASSERT_TRUE(any_nonzero);

  free(uops);
  poly_ctx_destroy(ctx);
  PASS();
}

/* C5: Primitive edge-case tests */

TEST(nn, c5_arange_negative_step) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *ar = poly_arange(ctx, 5.0, 0.0, -1.0);
  ASSERT_NOT_NULL(ar);
  PolyUOp *out = poly_buffer_f32(ctx, 5);
  PolyUOp *sink = poly_sink1(ctx, poly_test_store_to_buffer(ctx, out, ar));
  float result[5] = {0};
  PolyTestBufferView bind = POLY_TEST_HOST_VIEW(out, result);
  ASSERT_INT_EQ(poly_test_realize_buffer_views(ctx, sink, &bind, 1), 0);
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
  PolyUOp *sink = poly_sink1(ctx, poly_test_store_to_buffer(ctx, out, ar));
  float result[4] = {0};
  PolyTestBufferView bind = POLY_TEST_HOST_VIEW(out, result);
  ASSERT_INT_EQ(poly_test_realize_buffer_views(ctx, sink, &bind, 1), 0);
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
  PolyTestBufferView b1 = POLY_TEST_HOST_VIEW(o1, r1);
  ASSERT_INT_EQ(
      poly_test_realize_buffer_views(ctx, poly_sink1(ctx, poly_store_val(ctx, o1, e1)), &b1, 1), 0
  );
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
  PolyUOp *tl1 = poly_tril(ctx, ones, 1);
  ASSERT_NOT_NULL(tl1);
  PolyUOp *sum_tl = poly_sum_reduce(ctx, poly_sum_reduce(ctx, tl1, 1, 1), 0, 0);
  PolyUOp *buf = poly_buffer_f32(ctx, 1);
  float out[1] = {0};
  PolyTestBufferView bind = POLY_TEST_HOST_VIEW(buf, out);
  ASSERT_INT_EQ(
      poly_test_realize_buffer_views(
          ctx, poly_sink1(ctx, poly_store_val(ctx, buf, sum_tl)), &bind, 1
      ),
      0
  );
  ASSERT_FLOAT_EQ(out[0], 8.0f, 1e-5f);

  /* triu(diagonal=-1): keeps main diagonal + 1 subdiagonal
   * [[1,1,1],[1,1,1],[0,1,1]] -> sum=8 */
  PolyUOp *tu1 = poly_triu(ctx, ones, -1);
  ASSERT_NOT_NULL(tu1);
  PolyUOp *sum_tu = poly_sum_reduce(ctx, poly_sum_reduce(ctx, tu1, 1, 1), 0, 0);
  PolyUOp *buf2 = poly_buffer_f32(ctx, 1);
  float out2[1] = {0};
  PolyTestBufferView bind2 = POLY_TEST_HOST_VIEW(buf2, out2);
  ASSERT_INT_EQ(
      poly_test_realize_buffer_views(
          ctx, poly_sink1(ctx, poly_store_val(ctx, buf2, sum_tu)), &bind2, 1
      ),
      0
  );
  ASSERT_FLOAT_EQ(out2[0], 8.0f, 1e-5f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(nn, c5_tril_zeros_check) {
  /* Verify masked-out elements are exactly zero, not just correct sum. */
  PolyCtx *ctx = poly_ctx_new();
  int64_t shape[2] = {3, 3};
  PolyUOp *ones = poly_full(ctx, shape, 2, 1.0);
  PolyUOp *tl = poly_tril(ctx, ones, 0);
  ASSERT_NOT_NULL(tl);
  PolyUOp *out = poly_buffer_f32(ctx, 9);
  PolyUOp *sink = poly_sink1(ctx, poly_test_store_to_buffer(ctx, out, tl));
  float result[9] = {0};
  PolyTestBufferView bind = POLY_TEST_HOST_VIEW(out, result);
  ASSERT_INT_EQ(poly_test_realize_buffer_views(ctx, sink, &bind, 1), 0);
  /* Row-major: [[1,0,0],[1,1,0],[1,1,1]] */
  float expected[9] = {1, 0, 0, 1, 1, 0, 1, 1, 1};
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
  PolyTestBufferView b1 = POLY_TEST_HOST_VIEW(o1, r1);
  ASSERT_INT_EQ(
      poly_test_realize_buffer_views(ctx, poly_sink1(ctx, poly_store_val(ctx, o1, ls1)), &b1, 1), 0
  );
  ASSERT_FLOAT_EQ(r1[0], 3.0f, 1e-7f);

  /* start == stop: linspace(5,5,3) -> {5,5,5} */
  PolyUOp *ls2 = poly_linspace(ctx, 5.0, 5.0, 3);
  ASSERT_NOT_NULL(ls2);
  PolyUOp *o2 = poly_buffer_f32(ctx, 3);
  float r2[3] = {0};
  PolyTestBufferView b2 = POLY_TEST_HOST_VIEW(o2, r2);
  ASSERT_INT_EQ(
      poly_test_realize_buffer_views(ctx, poly_sink1(ctx, poly_store_val(ctx, o2, ls2)), &b2, 1), 0
  );
  for (int i = 0; i < 3; i++)
    ASSERT_FLOAT_EQ(r2[i], 5.0f, 1e-7f);

  /* steps=0 -> non-NULL zero-length buffer */
  PolyUOp *ls0 = poly_linspace(ctx, 0.0, 1.0, 0);
  ASSERT_NOT_NULL(ls0);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(wasm, abi_version) {
  ASSERT_INT_EQ(poly_abi_version(), POLYGRAD_ABI_VERSION);
  ASSERT_TRUE(poly_abi_version() >= 1);
  PASS();
}

/* C3: Float64 pipeline tests */

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
  PolyUOp *out = poly_test_buffer(ctx, POLY_FLOAT64, N);
  PolyUOp *add = poly_alu2(ctx, POLY_OP_ADD, a, b);
  PolyUOp *store = poly_test_store_to_buffer(ctx, out, add);
  PolyUOp *sink = poly_sink1(ctx, store);

  double a_d[8], b_d[8], o_d[8];
  for (int i = 0; i < N; i++) {
    a_d[i] = (double)(i + 1) * 1.1;
    b_d[i] = (double)(i + 1) * 2.2;
  }

  PolyTestBufferView bindings[] = {
      POLY_TEST_HOST_VIEW(out, o_d),
      POLY_TEST_HOST_VIEW(a, a_d),
      POLY_TEST_HOST_VIEW(b, b_d),
  };
  int ret = poly_test_realize_buffer_views(ctx, sink, bindings, 3);
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
  PolyUOp *out = poly_test_buffer(ctx, POLY_FLOAT64, N);
  PolyUOp *lx = poly_log(ctx, x);
  PolyUOp *elx = poly_exp(ctx, lx);
  PolyUOp *store = poly_test_store_to_buffer(ctx, out, elx);
  PolyUOp *sink = poly_sink1(ctx, store);

  double x_d[4] = {0.5, 1.0, 2.71828, 100.0};
  double o_d[4] = {0};

  PolyTestBufferView bindings[] = {
      POLY_TEST_HOST_VIEW(out, o_d),
      POLY_TEST_HOST_VIEW(x, x_d),
  };
  int ret = poly_test_realize_buffer_views(ctx, sink, bindings, 2);
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
  PolyUOp *grad = poly_grad(ctx, loss, p);
  ASSERT_NOT_NULL(grad);

  double p_data[4] = {1.0, 2.0, -3.0, 4.0};
  double loss_out[1] = {0};
  double grad_out[4] = {0};
  PolyUOp *loss_buf = poly_buffer_f64(ctx, 1);
  PolyUOp *grad_buf = poly_buffer_f64(ctx, 4);
  PolyUOp *sink = poly_sink_n(
      ctx,
      (PolyUOp *[]){
          poly_store_val(ctx, loss_buf, loss),
          poly_store_val(ctx, grad_buf, grad),
      },
      2
  );
  PolyTestBufferView bindings[] = {
      POLY_TEST_HOST_VIEW(p, p_data),
      POLY_TEST_HOST_VIEW(loss_buf, loss_out),
      POLY_TEST_HOST_VIEW(grad_buf, grad_out),
  };
  int ret = poly_test_realize_buffer_views(ctx, sink, bindings, 3);
  ASSERT_INT_EQ(ret, 0);
  /* loss = 1 + 4 + 9 + 16 = 30 */
  ASSERT_FLOAT_EQ(loss_out[0], 30.0, 1e-12);
  /* grad = 2 * p */
  ASSERT_FLOAT_EQ(grad_out[0], 2.0, 1e-12);
  ASSERT_FLOAT_EQ(grad_out[1], 4.0, 1e-12);
  ASSERT_FLOAT_EQ(grad_out[2], -6.0, 1e-12);
  ASSERT_FLOAT_EQ(grad_out[3], 8.0, 1e-12);

  poly_ctx_destroy(ctx);
  PASS();
}
