/*
 * test_mlp.c -- Tests for MLP family builder
 */

#include "test_harness.h"
#include "../src/model_mlp.h"
#include "../src/instance.h"
#include <string.h>
#include <stdlib.h>
#include <math.h>

/* ── Helper: build MLP spec JSON ─────────────────────────────────────── */

static const char *simple_mlp_spec =
  "{\"layers\":[2,4,1],\"activation\":\"relu\",\"bias\":true,"
  "\"loss\":\"mse\",\"batch_size\":1,\"seed\":42}";

static const char *no_bias_spec =
  "{\"layers\":[3,2],\"activation\":\"none\",\"bias\":false,"
  "\"loss\":\"none\",\"batch_size\":1,\"seed\":42}";

/* ── Tests ───────────────────────────────────────────────────────────── */

TEST(mlp, create_simple) {
  PolyInstance *inst = poly_mlp_instance(simple_mlp_spec,
                                          (int)strlen(simple_mlp_spec));
  ASSERT_NOT_NULL(inst);

  /* 2 weights + 2 biases = 4 params */
  ASSERT_INT_EQ(poly_instance_param_count(inst), 4);

  /* Check param names */
  ASSERT_STR_EQ(poly_instance_param_name(inst, 0), "layers.0.weight");
  ASSERT_STR_EQ(poly_instance_param_name(inst, 1), "layers.0.bias");
  ASSERT_STR_EQ(poly_instance_param_name(inst, 2), "layers.1.weight");
  ASSERT_STR_EQ(poly_instance_param_name(inst, 3), "layers.1.bias");

  /* Check param shapes */
  int64_t shape[8];
  int ndim;

  ndim = poly_instance_param_shape(inst, 0, shape, 8);
  ASSERT_INT_EQ(ndim, 2);
  ASSERT_INT_EQ((int)shape[0], 4);  /* out_dim */
  ASSERT_INT_EQ((int)shape[1], 2);  /* in_dim */

  ndim = poly_instance_param_shape(inst, 1, shape, 8);
  ASSERT_INT_EQ(ndim, 1);
  ASSERT_INT_EQ((int)shape[0], 4);  /* out_dim */

  ndim = poly_instance_param_shape(inst, 2, shape, 8);
  ASSERT_INT_EQ(ndim, 2);
  ASSERT_INT_EQ((int)shape[0], 1);  /* out_dim */
  ASSERT_INT_EQ((int)shape[1], 4);  /* in_dim */

  ndim = poly_instance_param_shape(inst, 3, shape, 8);
  ASSERT_INT_EQ(ndim, 1);
  ASSERT_INT_EQ((int)shape[0], 1);  /* out_dim */

  poly_instance_free(inst);
  PASS();
}

TEST(mlp, create_no_bias) {
  PolyInstance *inst = poly_mlp_instance(no_bias_spec,
                                          (int)strlen(no_bias_spec));
  ASSERT_NOT_NULL(inst);

  /* 1 weight, no bias */
  ASSERT_INT_EQ(poly_instance_param_count(inst), 1);
  ASSERT_STR_EQ(poly_instance_param_name(inst, 0), "layers.0.weight");

  int64_t shape[8];
  int ndim = poly_instance_param_shape(inst, 0, shape, 8);
  ASSERT_INT_EQ(ndim, 2);
  ASSERT_INT_EQ((int)shape[0], 2);  /* out_dim */
  ASSERT_INT_EQ((int)shape[1], 3);  /* in_dim */

  poly_instance_free(inst);
  PASS();
}

TEST(mlp, deterministic_init) {
  /* Same seed should produce identical weights */
  PolyInstance *inst1 = poly_mlp_instance(simple_mlp_spec,
                                           (int)strlen(simple_mlp_spec));
  PolyInstance *inst2 = poly_mlp_instance(simple_mlp_spec,
                                           (int)strlen(simple_mlp_spec));
  ASSERT_NOT_NULL(inst1);
  ASSERT_NOT_NULL(inst2);

  int64_t numel1, numel2;
  float *w1 = poly_instance_param_data(inst1, 0, &numel1);
  float *w2 = poly_instance_param_data(inst2, 0, &numel2);
  ASSERT_INT_EQ((int)numel1, (int)numel2);

  for (int64_t i = 0; i < numel1; i++)
    ASSERT_TRUE(w1[i] == w2[i]);

  poly_instance_free(inst1);
  poly_instance_free(inst2);
  PASS();
}

TEST(mlp, cross_seed_divergence) {
  const char *spec_seed99 =
    "{\"layers\":[2,4,1],\"activation\":\"relu\",\"bias\":true,"
    "\"loss\":\"mse\",\"batch_size\":1,\"seed\":99}";

  PolyInstance *inst1 = poly_mlp_instance(simple_mlp_spec,
                                           (int)strlen(simple_mlp_spec));
  PolyInstance *inst2 = poly_mlp_instance(spec_seed99,
                                           (int)strlen(spec_seed99));
  ASSERT_NOT_NULL(inst1);
  ASSERT_NOT_NULL(inst2);

  int64_t numel1, numel2;
  float *w1 = poly_instance_param_data(inst1, 0, &numel1);
  float *w2 = poly_instance_param_data(inst2, 0, &numel2);

  /* At least one weight should differ */
  int any_diff = 0;
  for (int64_t i = 0; i < numel1; i++) {
    if (w1[i] != w2[i]) { any_diff = 1; break; }
  }
  ASSERT_TRUE(any_diff);

  poly_instance_free(inst1);
  poly_instance_free(inst2);
  PASS();
}

TEST(mlp, kaiming_bounds) {
  /* Kaiming init: values should be within [-sqrt(6/fan_in), +sqrt(6/fan_in)] */
  PolyInstance *inst = poly_mlp_instance(simple_mlp_spec,
                                          (int)strlen(simple_mlp_spec));
  ASSERT_NOT_NULL(inst);

  /* Layer 0 weight: fan_in = 2, bound = sqrt(6/2) = sqrt(3) ~ 1.732 */
  int64_t numel;
  float *w = poly_instance_param_data(inst, 0, &numel);
  float bound = sqrtf(6.0f / 2.0f);
  for (int64_t i = 0; i < numel; i++) {
    ASSERT_TRUE(w[i] >= -bound && w[i] <= bound);
  }

  /* Layer 0 bias: should be zero-initialized */
  float *b = poly_instance_param_data(inst, 1, &numel);
  for (int64_t i = 0; i < numel; i++) {
    ASSERT_TRUE(b[i] == 0.0f);
  }

  poly_instance_free(inst);
  PASS();
}

TEST(mlp, forward_produces_output) {
  PolyInstance *inst = poly_mlp_instance(simple_mlp_spec,
                                          (int)strlen(simple_mlp_spec));
  ASSERT_NOT_NULL(inst);

  float x[] = { 1.0f, 2.0f };
  PolyIOBinding inputs[] = { { "x", x } };

  int ret = poly_instance_forward(inst, inputs, 1);
  ASSERT_INT_EQ(ret, 0);

  /* Find output buffer and check it has a value */
  int n_bufs = poly_instance_buf_count(inst);
  int found_output = 0;
  for (int i = 0; i < n_bufs; i++) {
    if (strcmp(poly_instance_buf_name(inst, i), "output") == 0) {
      int64_t numel;
      float *out = poly_instance_buf_data(inst, i, &numel);
      ASSERT_NOT_NULL(out);
      ASSERT_INT_EQ((int)numel, 1);  /* batch=1, out_dim=1 */
      /* Output should be finite */
      ASSERT_TRUE(isfinite(out[0]));
      found_output = 1;
      break;
    }
  }
  ASSERT_TRUE(found_output);

  poly_instance_free(inst);
  PASS();
}

TEST(mlp, forward_deterministic) {
  /* Same instance, same input -> same output */
  PolyInstance *inst = poly_mlp_instance(simple_mlp_spec,
                                          (int)strlen(simple_mlp_spec));
  ASSERT_NOT_NULL(inst);

  float x[] = { 1.0f, 2.0f };
  PolyIOBinding inputs[] = { { "x", x } };

  poly_instance_forward(inst, inputs, 1);

  /* Find output */
  int n_bufs = poly_instance_buf_count(inst);
  float out1 = 0.0f;
  for (int i = 0; i < n_bufs; i++) {
    if (strcmp(poly_instance_buf_name(inst, i), "output") == 0) {
      int64_t numel;
      float *out = poly_instance_buf_data(inst, i, &numel);
      out1 = out[0];
      break;
    }
  }

  /* Run again */
  poly_instance_forward(inst, inputs, 1);
  float out2 = 0.0f;
  for (int i = 0; i < n_bufs; i++) {
    if (strcmp(poly_instance_buf_name(inst, i), "output") == 0) {
      int64_t numel;
      float *out = poly_instance_buf_data(inst, i, &numel);
      out2 = out[0];
      break;
    }
  }

  ASSERT_TRUE(out1 == out2);

  poly_instance_free(inst);
  PASS();
}

TEST(mlp, null_and_invalid) {
  /* NULL input */
  ASSERT_TRUE(poly_mlp_instance(NULL, 0) == NULL);

  /* Empty JSON */
  ASSERT_TRUE(poly_mlp_instance("{}", 2) == NULL);

  /* Missing layers */
  const char *no_layers = "{\"activation\":\"relu\"}";
  ASSERT_TRUE(poly_mlp_instance(no_layers, (int)strlen(no_layers)) == NULL);

  /* Too few layers */
  const char *one_layer = "{\"layers\":[4]}";
  ASSERT_TRUE(poly_mlp_instance(one_layer, (int)strlen(one_layer)) == NULL);

  PASS();
}

TEST(mlp, train_single_layer) {
  /* Single-layer MLP: 2 -> 1 with MSE */
  const char *spec =
    "{\"layers\":[2,1],\"activation\":\"none\",\"bias\":true,"
    "\"loss\":\"mse\",\"batch_size\":1,\"seed\":42}";

  PolyInstance *inst = poly_mlp_instance(spec, (int)strlen(spec));
  ASSERT_NOT_NULL(inst);

  /* Configure SGD */
  poly_instance_set_optimizer(inst, POLY_OPTIM_SGD,
                               0.05f, 0.0f, 0.0f, 0.0f, 0.0f);

  /* Training data: x=[1, 2], y=[5] (target: w=[1,2], b=0 gives 5) */
  float x[] = { 1.0f, 2.0f };
  float y[] = { 5.0f };
  PolyIOBinding io[] = {
    { "x", x },
    { "y", y },
  };

  float first_loss = -1.0f;
  float prev_loss = 1e10f;
  for (int step = 0; step < 50; step++) {
    float loss;
    int ret = poly_instance_train_step(inst, io, 2, &loss);
    ASSERT_INT_EQ(ret, 0);
    ASSERT_TRUE(isfinite(loss));
    if (step == 0) first_loss = loss;
    prev_loss = loss;
  }

  /* Loss should decrease from initial */
  ASSERT_TRUE(prev_loss < first_loss);

  poly_instance_free(inst);
  PASS();
}

TEST(mlp, train_multi_layer) {
  /* Multi-layer MLP: 1 -> 4 -> 1 with relu + MSE.
   * Regression test for chained-reduction codegen bug where CONST(0)
   * pseudo-ranges from singleton dims entered REDUCE sources, producing
   * END(CONST) that corrupted scope depth in the C renderer. */
  const char *spec =
    "{\"layers\":[1,4,1],\"activation\":\"relu\",\"bias\":true,"
    "\"loss\":\"mse\",\"batch_size\":1,\"seed\":42}";

  PolyInstance *inst = poly_mlp_instance(spec, (int)strlen(spec));
  ASSERT_NOT_NULL(inst);

  /* Configure SGD */
  poly_instance_set_optimizer(inst, POLY_OPTIM_SGD,
                               0.01f, 0.0f, 0.0f, 0.0f, 0.0f);

  float x[] = { 1.0f };
  float y[] = { 2.0f };
  PolyIOBinding io[] = {
    { "x", x },
    { "y", y },
  };

  float first_loss = -1.0f;
  float prev_loss = 1e10f;
  for (int step = 0; step < 100; step++) {
    float loss;
    int ret = poly_instance_train_step(inst, io, 2, &loss);
    ASSERT_INT_EQ(ret, 0);
    ASSERT_TRUE(isfinite(loss));
    if (step == 0) first_loss = loss;
    prev_loss = loss;
  }

  /* Loss should decrease from initial */
  ASSERT_TRUE(prev_loss < first_loss);

  poly_instance_free(inst);
  PASS();
}

TEST(mlp, train_cross_entropy) {
  /* 3-class classification: 2 -> 4 -> 3 with cross-entropy loss */
  const char *spec =
    "{\"layers\":[2,4,3],\"activation\":\"relu\",\"bias\":true,"
    "\"loss\":\"cross_entropy\",\"batch_size\":1,\"seed\":42}";

  PolyInstance *inst = poly_mlp_instance(spec, (int)strlen(spec));
  ASSERT_NOT_NULL(inst);

  poly_instance_set_optimizer(inst, POLY_OPTIM_SGD,
                               0.01f, 0.0f, 0.0f, 0.0f, 0.0f);

  /* x=[1, 0], target=class 1 (one-hot: [0, 1, 0]) */
  float x[] = { 1.0f, 0.0f };
  float y[] = { 0.0f, 1.0f, 0.0f };
  PolyIOBinding io[] = {
    { "x", x },
    { "y", y },
  };

  float first_loss = -1.0f;
  float prev_loss = 1e10f;
  for (int step = 0; step < 100; step++) {
    float loss;
    int ret = poly_instance_train_step(inst, io, 2, &loss);
    ASSERT_INT_EQ(ret, 0);
    ASSERT_TRUE(isfinite(loss));
    if (step == 0) first_loss = loss;
    prev_loss = loss;
  }

  ASSERT_TRUE(prev_loss < first_loss);

  poly_instance_free(inst);
  PASS();
}
