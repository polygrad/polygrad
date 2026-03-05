/*
 * test_tabm.c -- Tests for TabM (BatchEnsemble MLP) builder
 */

#include "test_harness.h"
#include "../src/model_tabm.h"
#include "../src/instance.h"
#include <string.h>
#include <stdlib.h>
#include <math.h>

/* ── Specs ──────────────────────────────────────────────────────────── */

static const char *simple_tabm_spec =
  "{\"layers\":[2,4,1],\"activation\":\"relu\","
  "\"loss\":\"mse\",\"batch_size\":1,\"seed\":42,\"n_ensemble\":4}";

static const char *ce_tabm_spec =
  "{\"layers\":[2,4,3],\"activation\":\"relu\","
  "\"loss\":\"cross_entropy\",\"batch_size\":1,\"seed\":42,\"n_ensemble\":4}";

/* ── Tests ──────────────────────────────────────────────────────────── */

TEST(tabm, create_simple) {
  PolyInstance *inst = poly_tabm_instance(simple_tabm_spec,
                                           (int)strlen(simple_tabm_spec));
  ASSERT_NOT_NULL(inst);

  /* 2 layers * 4 params = 8 params (weight + r + s + b per layer) */
  ASSERT_INT_EQ(poly_instance_param_count(inst), 8);

  /* Check param names */
  ASSERT_STR_EQ(poly_instance_param_name(inst, 0), "layers.0.weight");
  ASSERT_STR_EQ(poly_instance_param_name(inst, 1), "layers.0.r");
  ASSERT_STR_EQ(poly_instance_param_name(inst, 2), "layers.0.s");
  ASSERT_STR_EQ(poly_instance_param_name(inst, 3), "layers.0.b");
  ASSERT_STR_EQ(poly_instance_param_name(inst, 4), "layers.1.weight");
  ASSERT_STR_EQ(poly_instance_param_name(inst, 5), "layers.1.r");
  ASSERT_STR_EQ(poly_instance_param_name(inst, 6), "layers.1.s");
  ASSERT_STR_EQ(poly_instance_param_name(inst, 7), "layers.1.b");

  /* Check param shapes */
  int64_t shape[8];
  int ndim;

  /* layers.0.weight: (4, 2) */
  ndim = poly_instance_param_shape(inst, 0, shape, 8);
  ASSERT_INT_EQ(ndim, 2);
  ASSERT_INT_EQ((int)shape[0], 4);
  ASSERT_INT_EQ((int)shape[1], 2);

  /* layers.0.r: (4, 2) -- k=4, in_dim=2 */
  ndim = poly_instance_param_shape(inst, 1, shape, 8);
  ASSERT_INT_EQ(ndim, 2);
  ASSERT_INT_EQ((int)shape[0], 4);  /* k */
  ASSERT_INT_EQ((int)shape[1], 2);  /* in_dim */

  /* layers.0.s: (4, 4) -- k=4, out_dim=4 */
  ndim = poly_instance_param_shape(inst, 2, shape, 8);
  ASSERT_INT_EQ(ndim, 2);
  ASSERT_INT_EQ((int)shape[0], 4);  /* k */
  ASSERT_INT_EQ((int)shape[1], 4);  /* out_dim */

  /* layers.0.b: (4, 4) -- k=4, out_dim=4 */
  ndim = poly_instance_param_shape(inst, 3, shape, 8);
  ASSERT_INT_EQ(ndim, 2);
  ASSERT_INT_EQ((int)shape[0], 4);  /* k */
  ASSERT_INT_EQ((int)shape[1], 4);  /* out_dim */

  poly_instance_free(inst);
  PASS();
}

TEST(tabm, init_values) {
  PolyInstance *inst = poly_tabm_instance(simple_tabm_spec,
                                           (int)strlen(simple_tabm_spec));
  ASSERT_NOT_NULL(inst);

  /* r should be initialized to ones */
  int64_t numel;
  float *r = poly_instance_param_data(inst, 1, &numel);
  ASSERT_NOT_NULL(r);
  for (int64_t i = 0; i < numel; i++)
    ASSERT_TRUE(r[i] == 1.0f);

  /* s should be initialized to ones */
  float *s = poly_instance_param_data(inst, 2, &numel);
  ASSERT_NOT_NULL(s);
  for (int64_t i = 0; i < numel; i++)
    ASSERT_TRUE(s[i] == 1.0f);

  /* b should be initialized to zeros */
  float *b = poly_instance_param_data(inst, 3, &numel);
  ASSERT_NOT_NULL(b);
  for (int64_t i = 0; i < numel; i++)
    ASSERT_TRUE(b[i] == 0.0f);

  /* weight should have Kaiming init (bounded) */
  float *w = poly_instance_param_data(inst, 0, &numel);
  float bound = sqrtf(6.0f / 2.0f);  /* fan_in = 2 */
  for (int64_t i = 0; i < numel; i++)
    ASSERT_TRUE(w[i] >= -bound && w[i] <= bound);

  poly_instance_free(inst);
  PASS();
}

TEST(tabm, deterministic_init) {
  PolyInstance *inst1 = poly_tabm_instance(simple_tabm_spec,
                                            (int)strlen(simple_tabm_spec));
  PolyInstance *inst2 = poly_tabm_instance(simple_tabm_spec,
                                            (int)strlen(simple_tabm_spec));
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

TEST(tabm, forward_produces_output) {
  PolyInstance *inst = poly_tabm_instance(simple_tabm_spec,
                                           (int)strlen(simple_tabm_spec));
  ASSERT_NOT_NULL(inst);

  float x[] = { 1.0f, 2.0f };
  PolyIOBinding inputs[] = { { "x", x } };

  int ret = poly_instance_forward(inst, inputs, 1);
  ASSERT_INT_EQ(ret, 0);

  /* Find output buffer */
  int n_bufs = poly_instance_buf_count(inst);
  int found_output = 0;
  for (int i = 0; i < n_bufs; i++) {
    if (strcmp(poly_instance_buf_name(inst, i), "output") == 0) {
      int64_t numel;
      float *out = poly_instance_buf_data(inst, i, &numel);
      ASSERT_NOT_NULL(out);
      ASSERT_INT_EQ((int)numel, 1);  /* batch=1, out_dim=1 */
      ASSERT_TRUE(isfinite(out[0]));
      found_output = 1;
      break;
    }
  }
  ASSERT_TRUE(found_output);

  poly_instance_free(inst);
  PASS();
}

TEST(tabm, forward_deterministic) {
  PolyInstance *inst = poly_tabm_instance(simple_tabm_spec,
                                           (int)strlen(simple_tabm_spec));
  ASSERT_NOT_NULL(inst);

  float x[] = { 1.0f, 2.0f };
  PolyIOBinding inputs[] = { { "x", x } };

  poly_instance_forward(inst, inputs, 1);
  float out1 = 0.0f;
  int n_bufs = poly_instance_buf_count(inst);
  for (int i = 0; i < n_bufs; i++) {
    if (strcmp(poly_instance_buf_name(inst, i), "output") == 0) {
      int64_t numel;
      float *out = poly_instance_buf_data(inst, i, &numel);
      out1 = out[0];
      break;
    }
  }

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

TEST(tabm, train_mse_loss_decreases) {
  PolyInstance *inst = poly_tabm_instance(simple_tabm_spec,
                                           (int)strlen(simple_tabm_spec));
  ASSERT_NOT_NULL(inst);

  poly_instance_set_optimizer(inst, POLY_OPTIM_SGD,
                               0.01f, 0.0f, 0.0f, 0.0f, 0.0f);

  float x[] = { 1.0f, 2.0f };
  float y[] = { 5.0f };
  PolyIOBinding io[] = {
    { "x", x },
    { "y", y },
  };

  float first_loss = -1.0f;
  float last_loss = 1e10f;
  for (int step = 0; step < 100; step++) {
    float loss;
    int ret = poly_instance_train_step(inst, io, 2, &loss);
    ASSERT_INT_EQ(ret, 0);
    ASSERT_TRUE(isfinite(loss));
    if (step == 0) first_loss = loss;
    last_loss = loss;
  }

  ASSERT_TRUE(last_loss < first_loss);

  poly_instance_free(inst);
  PASS();
}

TEST(tabm, train_cross_entropy_loss_decreases) {
  PolyInstance *inst = poly_tabm_instance(ce_tabm_spec,
                                           (int)strlen(ce_tabm_spec));
  ASSERT_NOT_NULL(inst);

  poly_instance_set_optimizer(inst, POLY_OPTIM_SGD,
                               0.01f, 0.0f, 0.0f, 0.0f, 0.0f);

  float x[] = { 1.0f, 0.0f };
  float y[] = { 0.0f, 1.0f, 0.0f };
  PolyIOBinding io[] = {
    { "x", x },
    { "y", y },
  };

  float first_loss = -1.0f;
  float last_loss = 1e10f;
  for (int step = 0; step < 100; step++) {
    float loss;
    int ret = poly_instance_train_step(inst, io, 2, &loss);
    ASSERT_INT_EQ(ret, 0);
    ASSERT_TRUE(isfinite(loss));
    if (step == 0) first_loss = loss;
    last_loss = loss;
  }

  ASSERT_TRUE(last_loss < first_loss);

  poly_instance_free(inst);
  PASS();
}

TEST(tabm, save_load_roundtrip) {
  PolyInstance *inst = poly_tabm_instance(simple_tabm_spec,
                                           (int)strlen(simple_tabm_spec));
  ASSERT_NOT_NULL(inst);

  /* Run a few training steps */
  poly_instance_set_optimizer(inst, POLY_OPTIM_SGD,
                               0.01f, 0.0f, 0.0f, 0.0f, 0.0f);
  float x[] = { 1.0f, 2.0f };
  float y[] = { 5.0f };
  PolyIOBinding io[] = { { "x", x }, { "y", y } };
  for (int i = 0; i < 10; i++) {
    float loss;
    poly_instance_train_step(inst, io, 2, &loss);
  }

  /* Forward to get prediction before save */
  PolyIOBinding fwd[] = { { "x", x } };
  poly_instance_forward(inst, fwd, 1);
  float pred_before = 0.0f;
  int n_bufs = poly_instance_buf_count(inst);
  for (int i = 0; i < n_bufs; i++) {
    if (strcmp(poly_instance_buf_name(inst, i), "output") == 0) {
      int64_t numel;
      float *out = poly_instance_buf_data(inst, i, &numel);
      pred_before = out[0];
      break;
    }
  }

  /* Export IR + weights */
  int ir_len = 0, w_len = 0;
  uint8_t *ir = poly_instance_export_ir(inst, &ir_len);
  uint8_t *weights = poly_instance_export_weights(inst, &w_len);
  ASSERT_NOT_NULL(ir);
  ASSERT_NOT_NULL(weights);
  ASSERT_TRUE(ir_len > 0);
  ASSERT_TRUE(w_len > 0);

  /* Reload */
  PolyInstance *inst2 = poly_instance_from_ir(ir, ir_len, weights, w_len);
  ASSERT_NOT_NULL(inst2);

  /* Forward on reloaded */
  poly_instance_forward(inst2, fwd, 1);
  float pred_after = 0.0f;
  int n_bufs2 = poly_instance_buf_count(inst2);
  for (int i = 0; i < n_bufs2; i++) {
    if (strcmp(poly_instance_buf_name(inst2, i), "output") == 0) {
      int64_t numel;
      float *out = poly_instance_buf_data(inst2, i, &numel);
      pred_after = out[0];
      break;
    }
  }

  /* Predictions should match */
  float diff = fabsf(pred_before - pred_after);
  ASSERT_TRUE(diff < 1e-6f);

  free(ir);
  free(weights);
  poly_instance_free(inst);
  poly_instance_free(inst2);
  PASS();
}

TEST(tabm, null_and_invalid) {
  ASSERT_TRUE(poly_tabm_instance(NULL, 0) == NULL);
  ASSERT_TRUE(poly_tabm_instance("{}", 2) == NULL);

  const char *no_layers = "{\"activation\":\"relu\",\"n_ensemble\":4}";
  ASSERT_TRUE(poly_tabm_instance(no_layers, (int)strlen(no_layers)) == NULL);

  const char *one_layer = "{\"layers\":[4],\"n_ensemble\":4}";
  ASSERT_TRUE(poly_tabm_instance(one_layer, (int)strlen(one_layer)) == NULL);

  PASS();
}
