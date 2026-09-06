/*
 * test_nam.c -- Tests for NAM (Neural Additive Model) builder
 */

#include "test_harness.h"
#include "../src/models/nam.h"
#include "../src/model.h"
#include <string.h>
#include <stdlib.h>
#include <math.h>

/* Specs */

static const char *simple_nam_spec =
    "{\"n_features\":2,\"hidden_sizes\":[4],\"activation\":\"relu\","
    "\"n_outputs\":1,\"loss\":\"mse\",\"batch_size\":1,\"seed\":42}";

static const char *exu_nam_spec = "{\"n_features\":2,\"hidden_sizes\":[4],\"activation\":\"exu\","
                                  "\"n_outputs\":1,\"loss\":\"mse\",\"batch_size\":1,\"seed\":42}";

static const char *ce_nam_spec =
    "{\"n_features\":2,\"hidden_sizes\":[4],\"activation\":\"relu\","
    "\"n_outputs\":3,\"loss\":\"cross_entropy\",\"batch_size\":1,\"seed\":42}";

/* Tests */

TEST(nam, create_simple) {
  PolyModel *inst =
      poly_nam_from_json(simple_nam_spec, (int)strlen(simple_nam_spec), POLY_DEVICE_AUTO);
  ASSERT_NOT_NULL(inst);

  /* intercept (1) + 2 features * (weight + bias per layer) = 1 + 2*(2*2) = 9 */
  /* layers: [1, 4, 1] => 2 linear layers per feature
   * Per feature: layer0.weight + layer0.bias + layer1.weight + layer1.bias = 4
   * Total: intercept + 2*4 = 9 */
  ASSERT_INT_EQ(poly_model_param_count(inst), 9);

  /* Check param names */
  ASSERT_STR_EQ(poly_model_param_name(inst, 0), "intercept");
  ASSERT_STR_EQ(poly_model_param_name(inst, 1), "features.0.layers.0.weight");
  ASSERT_STR_EQ(poly_model_param_name(inst, 2), "features.0.layers.0.bias");
  ASSERT_STR_EQ(poly_model_param_name(inst, 3), "features.0.layers.1.weight");
  ASSERT_STR_EQ(poly_model_param_name(inst, 4), "features.0.layers.1.bias");
  ASSERT_STR_EQ(poly_model_param_name(inst, 5), "features.1.layers.0.weight");
  ASSERT_STR_EQ(poly_model_param_name(inst, 6), "features.1.layers.0.bias");
  ASSERT_STR_EQ(poly_model_param_name(inst, 7), "features.1.layers.1.weight");
  ASSERT_STR_EQ(poly_model_param_name(inst, 8), "features.1.layers.1.bias");

  /* Check intercept shape: (1,) */
  int64_t shape[8];
  int ndim = poly_model_param_shape(inst, 0, shape, 8);
  ASSERT_INT_EQ(ndim, 1);
  ASSERT_INT_EQ((int)shape[0], 1);

  /* Check features.0.layers.0.weight shape: (4, 1) */
  ndim = poly_model_param_shape(inst, 1, shape, 8);
  ASSERT_INT_EQ(ndim, 2);
  ASSERT_INT_EQ((int)shape[0], 4);
  ASSERT_INT_EQ((int)shape[1], 1);

  /* Check features.0.layers.0.bias shape: (4,) */
  ndim = poly_model_param_shape(inst, 2, shape, 8);
  ASSERT_INT_EQ(ndim, 1);
  ASSERT_INT_EQ((int)shape[0], 4);

  poly_model_free(inst);
  PASS();
}

TEST(nam, staged_builder_does_not_use_ctx_registry) {
  PolyModel *inst =
      poly_nam_from_json(simple_nam_spec, (int)strlen(simple_nam_spec), POLY_DEVICE_AUTO);
  ASSERT_NOT_NULL(inst);
  ASSERT_INT_EQ(poly_ctx_named_count(poly_model_ctx(inst)), 0);
  poly_model_free(inst);
  PASS();
}

TEST(nam, create_exu) {
  PolyModel *inst = poly_nam_from_json(exu_nam_spec, (int)strlen(exu_nam_spec), POLY_DEVICE_AUTO);
  ASSERT_NOT_NULL(inst);

  /* With ExU, hidden layers get extra exu.weight + exu.bias params.
   * layers: [1, 4, 1] => 2 linear layers per feature.
   * Only first hidden layer (l=0) gets ExU params (l < n_linear-1).
   * Per feature: layer0.weight + layer0.bias + exu.0.weight + exu.0.bias
   *            + layer1.weight + layer1.bias = 6
   * Total: intercept + 2*6 = 13 */
  ASSERT_INT_EQ(poly_model_param_count(inst), 13);

  /* Check ExU param names */
  ASSERT_STR_EQ(poly_model_param_name(inst, 3), "features.0.exu.0.weight");
  ASSERT_STR_EQ(poly_model_param_name(inst, 4), "features.0.exu.0.bias");

  poly_model_free(inst);
  PASS();
}

TEST(nam, init_values) {
  PolyModel *inst =
      poly_nam_from_json(simple_nam_spec, (int)strlen(simple_nam_spec), POLY_DEVICE_AUTO);
  ASSERT_NOT_NULL(inst);

  /* Intercept should be zero */
  int64_t numel;
  float *intercept = poly_model_param_data(inst, 0, &numel);
  ASSERT_NOT_NULL(intercept);
  ASSERT_INT_EQ((int)numel, 1);
  ASSERT_TRUE(intercept[0] == 0.0f);

  /* Bias should be zero */
  float *b = poly_model_param_data(inst, 2, &numel);
  ASSERT_NOT_NULL(b);
  for (int64_t i = 0; i < numel; i++)
    ASSERT_TRUE(b[i] == 0.0f);

  /* Weight should have Kaiming init (bounded) */
  float *w = poly_model_param_data(inst, 1, &numel);
  float bound = sqrtf(6.0f / 1.0f); /* fan_in = 1 */
  for (int64_t i = 0; i < numel; i++)
    ASSERT_TRUE(w[i] >= -bound && w[i] <= bound);

  poly_model_free(inst);
  PASS();
}

TEST(nam, exu_init_values) {
  PolyModel *inst = poly_nam_from_json(exu_nam_spec, (int)strlen(exu_nam_spec), POLY_DEVICE_AUTO);
  ASSERT_NOT_NULL(inst);

  /* ExU weight should be zero (exp(0) = 1) */
  int64_t numel;
  float *exu_w = poly_model_param_data(inst, 3, &numel);
  ASSERT_NOT_NULL(exu_w);
  for (int64_t i = 0; i < numel; i++)
    ASSERT_TRUE(exu_w[i] == 0.0f);

  /* ExU bias should be zero */
  float *exu_b = poly_model_param_data(inst, 4, &numel);
  ASSERT_NOT_NULL(exu_b);
  for (int64_t i = 0; i < numel; i++)
    ASSERT_TRUE(exu_b[i] == 0.0f);

  poly_model_free(inst);
  PASS();
}

TEST(nam, deterministic_init) {
  PolyModel *inst1 =
      poly_nam_from_json(simple_nam_spec, (int)strlen(simple_nam_spec), POLY_DEVICE_AUTO);
  PolyModel *inst2 =
      poly_nam_from_json(simple_nam_spec, (int)strlen(simple_nam_spec), POLY_DEVICE_AUTO);
  ASSERT_NOT_NULL(inst1);
  ASSERT_NOT_NULL(inst2);

  int64_t numel1, numel2;
  float *w1 = poly_model_param_data(inst1, 1, &numel1);
  float *w2 = poly_model_param_data(inst2, 1, &numel2);
  ASSERT_INT_EQ((int)numel1, (int)numel2);

  for (int64_t i = 0; i < numel1; i++)
    ASSERT_TRUE(w1[i] == w2[i]);

  poly_model_free(inst1);
  poly_model_free(inst2);
  PASS();
}

TEST(nam, forward_produces_output) {
  PolyModel *inst =
      poly_nam_from_json(simple_nam_spec, (int)strlen(simple_nam_spec), POLY_DEVICE_AUTO);
  ASSERT_NOT_NULL(inst);

  float x[] = {1.0f, 2.0f};
  PolyIOBinding inputs[] = {POLY_IO_BINDING_ARRAY("x", x, POLY_FLOAT32)};

  int ret = poly_model_forward(inst, inputs, 1);
  ASSERT_INT_EQ(ret, 0);

  /* Find output buffer */
  int n_bufs = poly_model_buf_count(inst);
  int found_output = 0;
  for (int i = 0; i < n_bufs; i++) {
    if (strcmp(poly_model_buf_name(inst, i), "output") == 0) {
      int64_t numel;
      float *out = poly_model_buf_data(inst, i, &numel);
      ASSERT_NOT_NULL(out);
      ASSERT_INT_EQ((int)numel, 1); /* batch=1, n_outputs=1 */
      ASSERT_TRUE(isfinite(out[0]));
      found_output = 1;
      break;
    }
  }
  ASSERT_TRUE(found_output);

  poly_model_free(inst);
  PASS();
}

TEST(nam, forward_deterministic) {
  PolyModel *inst =
      poly_nam_from_json(simple_nam_spec, (int)strlen(simple_nam_spec), POLY_DEVICE_AUTO);
  ASSERT_NOT_NULL(inst);

  float x[] = {1.0f, 2.0f};
  PolyIOBinding inputs[] = {POLY_IO_BINDING_ARRAY("x", x, POLY_FLOAT32)};

  poly_model_forward(inst, inputs, 1);
  float out1 = 0.0f;
  int n_bufs = poly_model_buf_count(inst);
  for (int i = 0; i < n_bufs; i++) {
    if (strcmp(poly_model_buf_name(inst, i), "output") == 0) {
      int64_t numel;
      float *out = poly_model_buf_data(inst, i, &numel);
      out1 = out[0];
      break;
    }
  }

  poly_model_forward(inst, inputs, 1);
  float out2 = 0.0f;
  for (int i = 0; i < n_bufs; i++) {
    if (strcmp(poly_model_buf_name(inst, i), "output") == 0) {
      int64_t numel;
      float *out = poly_model_buf_data(inst, i, &numel);
      out2 = out[0];
      break;
    }
  }

  ASSERT_TRUE(out1 == out2);

  poly_model_free(inst);
  PASS();
}

TEST(nam, train_mse_loss_decreases) {
  PolyModel *inst =
      poly_nam_from_json(simple_nam_spec, (int)strlen(simple_nam_spec), POLY_DEVICE_AUTO);
  ASSERT_NOT_NULL(inst);

  poly_model_set_optimizer(inst, POLY_OPTIM_SGD, 0.01f, 0.0f, 0.0f, 0.0f, 0.0f);

  float x[] = {1.0f, 2.0f};
  float y[] = {5.0f};
  PolyIOBinding io[] = {
      POLY_IO_BINDING_ARRAY("x", x, POLY_FLOAT32),
      POLY_IO_BINDING_ARRAY("y", y, POLY_FLOAT32),
  };

  float first_loss = -1.0f;
  float last_loss = 1e10f;
  for (int step = 0; step < 100; step++) {
    float loss;
    int ret = poly_model_train_step(inst, NULL, io, 2, &loss);
    ASSERT_INT_EQ(ret, 0);
    ASSERT_TRUE(isfinite(loss));
    if (step == 0) first_loss = loss;
    last_loss = loss;
  }

  ASSERT_TRUE(last_loss < first_loss);

  poly_model_free(inst);
  PASS();
}

TEST(nam, train_exu_loss_decreases) {
  PolyModel *inst = poly_nam_from_json(exu_nam_spec, (int)strlen(exu_nam_spec), POLY_DEVICE_AUTO);
  ASSERT_NOT_NULL(inst);

  poly_model_set_optimizer(inst, POLY_OPTIM_SGD, 0.01f, 0.0f, 0.0f, 0.0f, 0.0f);

  float x[] = {1.0f, 2.0f};
  float y[] = {5.0f};
  PolyIOBinding io[] = {
      POLY_IO_BINDING_ARRAY("x", x, POLY_FLOAT32),
      POLY_IO_BINDING_ARRAY("y", y, POLY_FLOAT32),
  };

  float first_loss = -1.0f;
  float last_loss = 1e10f;
  for (int step = 0; step < 100; step++) {
    float loss;
    int ret = poly_model_train_step(inst, NULL, io, 2, &loss);
    ASSERT_INT_EQ(ret, 0);
    ASSERT_TRUE(isfinite(loss));
    if (step == 0) first_loss = loss;
    last_loss = loss;
  }

  ASSERT_TRUE(last_loss < first_loss);

  poly_model_free(inst);
  PASS();
}

TEST(nam, train_cross_entropy_loss_decreases) {
  PolyModel *inst = poly_nam_from_json(ce_nam_spec, (int)strlen(ce_nam_spec), POLY_DEVICE_AUTO);
  ASSERT_NOT_NULL(inst);

  poly_model_set_optimizer(inst, POLY_OPTIM_SGD, 0.01f, 0.0f, 0.0f, 0.0f, 0.0f);

  float x[] = {1.0f, 0.0f};
  float y[] = {0.0f, 1.0f, 0.0f};
  PolyIOBinding io[] = {
      POLY_IO_BINDING_ARRAY("x", x, POLY_FLOAT32),
      POLY_IO_BINDING_ARRAY("y", y, POLY_FLOAT32),
  };

  float first_loss = -1.0f;
  float last_loss = 1e10f;
  for (int step = 0; step < 100; step++) {
    float loss;
    int ret = poly_model_train_step(inst, NULL, io, 2, &loss);
    ASSERT_INT_EQ(ret, 0);
    ASSERT_TRUE(isfinite(loss));
    if (step == 0) first_loss = loss;
    last_loss = loss;
  }

  ASSERT_TRUE(last_loss < first_loss);

  poly_model_free(inst);
  PASS();
}

TEST(nam, save_load_roundtrip) {
  PolyModel *inst =
      poly_nam_from_json(simple_nam_spec, (int)strlen(simple_nam_spec), POLY_DEVICE_AUTO);
  ASSERT_NOT_NULL(inst);

  /* Train a few steps */
  poly_model_set_optimizer(inst, POLY_OPTIM_SGD, 0.01f, 0.0f, 0.0f, 0.0f, 0.0f);
  float x[] = {1.0f, 2.0f};
  float y[] = {5.0f};
  PolyIOBinding io[] = {
      POLY_IO_BINDING_ARRAY("x", x, POLY_FLOAT32), POLY_IO_BINDING_ARRAY("y", y, POLY_FLOAT32)};
  for (int i = 0; i < 10; i++) {
    float loss;
    poly_model_train_step(inst, NULL, io, 2, &loss);
  }

  /* Forward to get prediction before save */
  PolyIOBinding fwd[] = {POLY_IO_BINDING_ARRAY("x", x, POLY_FLOAT32)};
  poly_model_forward(inst, fwd, 1);
  float pred_before = 0.0f;
  int n_bufs = poly_model_buf_count(inst);
  for (int i = 0; i < n_bufs; i++) {
    if (strcmp(poly_model_buf_name(inst, i), "output") == 0) {
      int64_t numel;
      float *out = poly_model_buf_data(inst, i, &numel);
      pred_before = out[0];
      break;
    }
  }

  /* Export IR + weights */
  int ir_len = 0, w_len = 0;
  uint8_t *ir = poly_model_export_ir(inst, &ir_len);
  uint8_t *weights = poly_model_export_weights(inst, &w_len);
  ASSERT_NOT_NULL(ir);
  ASSERT_NOT_NULL(weights);
  ASSERT_TRUE(ir_len > 0);
  ASSERT_TRUE(w_len > 0);

  /* Reload */
  PolyModel *inst2 = poly_model_from_ir(ir, ir_len, weights, w_len);
  ASSERT_NOT_NULL(inst2);

  /* Forward on reloaded */
  poly_model_forward(inst2, fwd, 1);
  float pred_after = 0.0f;
  int n_bufs2 = poly_model_buf_count(inst2);
  for (int i = 0; i < n_bufs2; i++) {
    if (strcmp(poly_model_buf_name(inst2, i), "output") == 0) {
      int64_t numel;
      float *out = poly_model_buf_data(inst2, i, &numel);
      pred_after = out[0];
      break;
    }
  }

  /* Predictions should match */
  float diff = fabsf(pred_before - pred_after);
  ASSERT_TRUE(diff < 1e-6f);

  free(ir);
  free(weights);
  poly_model_free(inst);
  poly_model_free(inst2);
  PASS();
}

TEST(nam, null_and_invalid) {
  ASSERT_TRUE(poly_nam_from_json(NULL, 0, POLY_DEVICE_AUTO) == NULL);
  ASSERT_TRUE(poly_nam_from_json("{}", 2, POLY_DEVICE_AUTO) == NULL);

  const char *no_features = "{\"hidden_sizes\":[4],\"n_outputs\":1}";
  ASSERT_TRUE(poly_nam_from_json(no_features, (int)strlen(no_features), POLY_DEVICE_AUTO) == NULL);

  PASS();
}
