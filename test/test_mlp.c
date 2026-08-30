/*
 * test_mlp.c -- Tests for MLP family builder
 */

#include "test_harness.h"
#include "../src/models/mlp.h"
#include "../src/instance.h"
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

/* Helper: build MLP spec JSON */

static const char *simple_mlp_spec = "{\"layers\":[2,4,1],\"activation\":\"relu\",\"bias\":true,"
                                     "\"loss\":\"mse\",\"batch_size\":1,\"seed\":42}";

static const char *no_bias_spec = "{\"layers\":[3,2],\"activation\":\"none\",\"bias\":false,"
                                  "\"loss\":\"none\",\"batch_size\":1,\"seed\":42}";

typedef struct {
  const char *key;
  char *value;
  bool had_value;
} MlpEnvSave;

static MlpEnvSave mlp_save_env(const char *key) {
  const char *cur = getenv(key);
  return (MlpEnvSave){
      .key = key,
      .value = cur ? strdup(cur) : NULL,
      .had_value = cur != NULL,
  };
}

static void mlp_restore_env(MlpEnvSave *s) {
  if (!s) return;
  if (s->had_value)
    setenv(s->key, s->value ? s->value : "", 1);
  else
    unsetenv(s->key);
  free(s->value);
  s->value = NULL;
}

/* Tests */

TEST(mlp, create_simple) {
  PolyInstance *inst = poly_mlp_from_json(simple_mlp_spec, (int)strlen(simple_mlp_spec), POLY_DEVICE_AUTO);
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
  ASSERT_INT_EQ((int)shape[0], 4); /* out_dim */
  ASSERT_INT_EQ((int)shape[1], 2); /* in_dim */

  ndim = poly_instance_param_shape(inst, 1, shape, 8);
  ASSERT_INT_EQ(ndim, 1);
  ASSERT_INT_EQ((int)shape[0], 4); /* out_dim */

  ndim = poly_instance_param_shape(inst, 2, shape, 8);
  ASSERT_INT_EQ(ndim, 2);
  ASSERT_INT_EQ((int)shape[0], 1); /* out_dim */
  ASSERT_INT_EQ((int)shape[1], 4); /* in_dim */

  ndim = poly_instance_param_shape(inst, 3, shape, 8);
  ASSERT_INT_EQ(ndim, 1);
  ASSERT_INT_EQ((int)shape[0], 1); /* out_dim */

  poly_instance_free(inst);
  PASS();
}

TEST(mlp, staged_builder_retains_complete_physical_template) {
  PolyInstance *inst = poly_mlp_from_json(
      simple_mlp_spec, (int)strlen(simple_mlp_spec), POLY_DEVICE_INTERP
  );

  ASSERT_NOT_NULL(inst);
  ASSERT_INT_EQ(poly_ctx_get_preferred_device(poly_instance_ctx(inst)), POLY_DEVICE_INTERP);
  PolyUOp *sink = poly_instance_get_sink(inst, "forward");
  ASSERT_NOT_NULL(sink);
  ASSERT_INT_EQ(sink->op, POLY_OP_SINK);
  ASSERT_INT_EQ(sink->n_src, 1);
  ASSERT_INT_EQ(sink->src[0]->op, POLY_OP_STORE);
  ASSERT_INT_EQ(poly_uop_device(sink->src[0]->src[0]), POLY_DEVICE_INTERP);

  int n = 0;
  PolyUOp **topo = poly_toposort_alloc(poly_instance_ctx(inst), sink, &n);
  ASSERT_NOT_NULL(topo);
  for (int i = 0; i < n; i++)
    if (topo[i]->op == POLY_OP_BUFFER)
      ASSERT_INT_EQ(poly_uop_device(topo[i]), POLY_DEVICE_INTERP);
  free(topo);

  poly_instance_free(inst);
  PASS();
}

TEST(mlp, staged_builder_does_not_use_ctx_registry) {
  PolyInstance *inst = poly_mlp_from_json(simple_mlp_spec, (int)strlen(simple_mlp_spec), POLY_DEVICE_AUTO);
  ASSERT_NOT_NULL(inst);
  ASSERT_INT_EQ(poly_ctx_named_count(poly_instance_ctx(inst)), 0);
  poly_instance_free(inst);
  PASS();
}

TEST(mlp, create_no_bias) {
  PolyInstance *inst = poly_mlp_from_json(no_bias_spec, (int)strlen(no_bias_spec), POLY_DEVICE_AUTO);
  ASSERT_NOT_NULL(inst);

  /* 1 weight, no bias */
  ASSERT_INT_EQ(poly_instance_param_count(inst), 1);
  ASSERT_STR_EQ(poly_instance_param_name(inst, 0), "layers.0.weight");

  int64_t shape[8];
  int ndim = poly_instance_param_shape(inst, 0, shape, 8);
  ASSERT_INT_EQ(ndim, 2);
  ASSERT_INT_EQ((int)shape[0], 2); /* out_dim */
  ASSERT_INT_EQ((int)shape[1], 3); /* in_dim */

  poly_instance_free(inst);
  PASS();
}

TEST(mlp, deterministic_init) {
  /* Same seed should produce identical weights */
  PolyInstance *inst1 = poly_mlp_from_json(simple_mlp_spec, (int)strlen(simple_mlp_spec), POLY_DEVICE_AUTO);
  PolyInstance *inst2 = poly_mlp_from_json(simple_mlp_spec, (int)strlen(simple_mlp_spec), POLY_DEVICE_AUTO);
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
  const char *spec_seed99 = "{\"layers\":[2,4,1],\"activation\":\"relu\",\"bias\":true,"
                            "\"loss\":\"mse\",\"batch_size\":1,\"seed\":99}";

  PolyInstance *inst1 = poly_mlp_from_json(simple_mlp_spec, (int)strlen(simple_mlp_spec), POLY_DEVICE_AUTO);
  PolyInstance *inst2 = poly_mlp_from_json(spec_seed99, (int)strlen(spec_seed99), POLY_DEVICE_AUTO);
  ASSERT_NOT_NULL(inst1);
  ASSERT_NOT_NULL(inst2);

  int64_t numel1, numel2;
  float *w1 = poly_instance_param_data(inst1, 0, &numel1);
  float *w2 = poly_instance_param_data(inst2, 0, &numel2);

  /* At least one weight should differ */
  int any_diff = 0;
  for (int64_t i = 0; i < numel1; i++) {
    if (w1[i] != w2[i]) {
      any_diff = 1;
      break;
    }
  }
  ASSERT_TRUE(any_diff);

  poly_instance_free(inst1);
  poly_instance_free(inst2);
  PASS();
}

TEST(mlp, kaiming_bounds) {
  /* Kaiming init: values should be within [-sqrt(6/fan_in), +sqrt(6/fan_in)] */
  PolyInstance *inst = poly_mlp_from_json(simple_mlp_spec, (int)strlen(simple_mlp_spec), POLY_DEVICE_AUTO);
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
  PolyInstance *inst = poly_mlp_from_json(simple_mlp_spec, (int)strlen(simple_mlp_spec), POLY_DEVICE_AUTO);
  ASSERT_NOT_NULL(inst);

  float x[] = {1.0f, 2.0f};
  PolyIOBinding inputs[] = {POLY_IO_BINDING_ARRAY("x", x, POLY_FLOAT32)};

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
      ASSERT_INT_EQ((int)numel, 1); /* batch=1, out_dim=1 */
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
  PolyInstance *inst = poly_mlp_from_json(simple_mlp_spec, (int)strlen(simple_mlp_spec), POLY_DEVICE_AUTO);
  ASSERT_NOT_NULL(inst);

  float x[] = {1.0f, 2.0f};
  PolyIOBinding inputs[] = {POLY_IO_BINDING_ARRAY("x", x, POLY_FLOAT32)};

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

TEST(mlp, forward_and_train_replay_stats_plateau) {
  MlpEnvSave pcache = mlp_save_env("POLY_PCACHE");
  MlpEnvSave scache = mlp_save_env("SCACHE");
  setenv("POLY_PCACHE", "1", 1);
  setenv("SCACHE", "1", 1);

  PolyInstance *inst = poly_mlp_from_json(simple_mlp_spec, (int)strlen(simple_mlp_spec), POLY_DEVICE_AUTO);
  ASSERT_NOT_NULL(inst);
  PolyCtx *ctx = poly_instance_ctx(inst);
  ASSERT_NOT_NULL(ctx);

  float x[] = {1.0f, 2.0f};
  float y[] = {5.0f};
  PolyIOBinding forward_io[] = {POLY_IO_BINDING_ARRAY("x", x, POLY_FLOAT32)};
  PolyIOBinding train_io[] = {POLY_IO_BINDING_ARRAY("x", x, POLY_FLOAT32), POLY_IO_BINDING_ARRAY("y", y, POLY_FLOAT32)};

  ASSERT_INT_EQ(poly_instance_forward(inst, forward_io, 1), 0);
  PolyCtxStats forward_first = {0};
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &forward_first), 0);
  /* Pinned compile_linear sends every CALL(SINK) through to_program, including
   * PythonRenderer, whose cache is keyed on the raw SINK before compilation
   * (engine/realize.py:244-267; codegen/__init__.py:244-250). INTERP is the
   * corresponding Polygrad execution backend and must cache the same boundary. */
  ASSERT_TRUE(forward_first.to_program_cache_entries > 0);
  ASSERT_TRUE(forward_first.runtime_cache_entries > 0);

  for (int iter = 0; iter < 16; iter++)
    ASSERT_INT_EQ(poly_instance_forward(inst, forward_io, 1), 0);

  PolyCtxStats forward_replay = {0};
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &forward_replay), 0);
  ASSERT_INT_EQ(forward_replay.arena_bytes, forward_first.arena_bytes);
  ASSERT_INT_EQ(forward_replay.cse_entries, forward_first.cse_entries);
  ASSERT_INT_EQ(forward_replay.to_program_cache_entries, forward_first.to_program_cache_entries);
  ASSERT_INT_EQ(forward_replay.runtime_cache_entries, forward_first.runtime_cache_entries);
  ASSERT_INT_EQ(forward_replay.shape_cache_entries, forward_first.shape_cache_entries);
  ASSERT_INT_EQ(forward_replay.buffer_entries, forward_first.buffer_entries);
  ASSERT_INT_EQ(forward_replay.compiled_artifact_bytes, forward_first.compiled_artifact_bytes);

  ASSERT_INT_EQ(
      poly_instance_set_optimizer(inst, POLY_OPTIM_ADAM, 0.001f, 0.9f, 0.999f, 1e-8f, 0.0f), 0
  );
  float loss = 0.0f;
  ASSERT_INT_EQ(poly_instance_train_step(inst, train_io, 2, &loss), 0);
  ASSERT_TRUE(isfinite(loss));

  PolyCtxStats train_first = {0};
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &train_first), 0);
  ASSERT_TRUE(train_first.to_program_cache_entries >= forward_first.to_program_cache_entries);
  ASSERT_TRUE(train_first.runtime_cache_entries >= forward_first.runtime_cache_entries);

  for (int iter = 0; iter < 16; iter++) {
    ASSERT_INT_EQ(poly_instance_train_step(inst, train_io, 2, &loss), 0);
    ASSERT_TRUE(isfinite(loss));
  }

  PolyCtxStats train_replay = {0};
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &train_replay), 0);
  ASSERT_INT_EQ(train_replay.arena_bytes, train_first.arena_bytes);
  ASSERT_INT_EQ(train_replay.cse_entries, train_first.cse_entries);
  ASSERT_INT_EQ(train_replay.to_program_cache_entries, train_first.to_program_cache_entries);
  ASSERT_INT_EQ(train_replay.runtime_cache_entries, train_first.runtime_cache_entries);
  ASSERT_INT_EQ(train_replay.shape_cache_entries, train_first.shape_cache_entries);
  ASSERT_INT_EQ(train_replay.buffer_entries, train_first.buffer_entries);
  ASSERT_INT_EQ(train_replay.compiled_artifact_bytes, train_first.compiled_artifact_bytes);

  poly_instance_free(inst);
  mlp_restore_env(&scache);
  mlp_restore_env(&pcache);
  PASS();
}

TEST(mlp, null_and_invalid) {
  /* NULL input */
  ASSERT_TRUE(poly_mlp_from_json(NULL, 0, POLY_DEVICE_AUTO) == NULL);

  /* Empty JSON */
  ASSERT_TRUE(poly_mlp_from_json("{}", 2, POLY_DEVICE_AUTO) == NULL);

  /* Missing layers */
  const char *no_layers = "{\"activation\":\"relu\"}";
  ASSERT_TRUE(poly_mlp_from_json(no_layers, (int)strlen(no_layers), POLY_DEVICE_AUTO) == NULL);

  /* Too few layers */
  const char *one_layer = "{\"layers\":[4]}";
  ASSERT_TRUE(poly_mlp_from_json(one_layer, (int)strlen(one_layer), POLY_DEVICE_AUTO) == NULL);

  PASS();
}

TEST(mlp, train_single_layer) {
  /* Single-layer MLP: 2 -> 1 with MSE */
  const char *spec = "{\"layers\":[2,1],\"activation\":\"none\",\"bias\":true,"
                     "\"loss\":\"mse\",\"batch_size\":1,\"seed\":42}";

  PolyInstance *inst = poly_mlp_from_json(spec, (int)strlen(spec), POLY_DEVICE_AUTO);
  ASSERT_NOT_NULL(inst);

  /* Configure SGD */
  poly_instance_set_optimizer(inst, POLY_OPTIM_SGD, 0.05f, 0.0f, 0.0f, 0.0f, 0.0f);

  /* Training data: x=[1, 2], y=[5] (target: w=[1,2], b=0 gives 5) */
  float x[] = {1.0f, 2.0f};
  float y[] = {5.0f};
  PolyIOBinding io[] = {
      POLY_IO_BINDING_ARRAY("x", x, POLY_FLOAT32),
      POLY_IO_BINDING_ARRAY("y", y, POLY_FLOAT32),
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
  const char *spec = "{\"layers\":[1,4,1],\"activation\":\"relu\",\"bias\":true,"
                     "\"loss\":\"mse\",\"batch_size\":1,\"seed\":42}";

  PolyInstance *inst = poly_mlp_from_json(spec, (int)strlen(spec), POLY_DEVICE_AUTO);
  ASSERT_NOT_NULL(inst);

  /* Configure SGD */
  poly_instance_set_optimizer(inst, POLY_OPTIM_SGD, 0.01f, 0.0f, 0.0f, 0.0f, 0.0f);

  float x[] = {1.0f};
  float y[] = {2.0f};
  PolyIOBinding io[] = {
      POLY_IO_BINDING_ARRAY("x", x, POLY_FLOAT32),
      POLY_IO_BINDING_ARRAY("y", y, POLY_FLOAT32),
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
  const char *spec = "{\"layers\":[2,4,3],\"activation\":\"relu\",\"bias\":true,"
                     "\"loss\":\"cross_entropy\",\"batch_size\":1,\"seed\":42}";

  PolyInstance *inst = poly_mlp_from_json(spec, (int)strlen(spec), POLY_DEVICE_AUTO);
  ASSERT_NOT_NULL(inst);

  poly_instance_set_optimizer(inst, POLY_OPTIM_SGD, 0.01f, 0.0f, 0.0f, 0.0f, 0.0f);

  /* x=[1, 0], target=class 1 (one-hot: [0, 1, 0]) */
  float x[] = {1.0f, 0.0f};
  float y[] = {0.0f, 1.0f, 0.0f};
  PolyIOBinding io[] = {
      POLY_IO_BINDING_ARRAY("x", x, POLY_FLOAT32),
      POLY_IO_BINDING_ARRAY("y", y, POLY_FLOAT32),
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

TEST(mlp, train_batch2_mse) {
  /* batch_size=2: 2 -> 3 -> 2, relu, MSE (P0 regression) */
  const char *spec = "{\"layers\":[2,3,2],\"activation\":\"relu\",\"bias\":false,"
                     "\"loss\":\"mse\",\"batch_size\":2,\"seed\":42}";

  PolyInstance *inst = poly_mlp_from_json(spec, (int)strlen(spec), POLY_DEVICE_AUTO);
  ASSERT_NOT_NULL(inst);
  poly_instance_set_optimizer(inst, POLY_OPTIM_SGD, 0.01f, 0.0f, 0.0f, 0.0f, 0.0f);

  float x[] = {0.5f, 0.5f, 0.5f, 0.5f};
  float y[] = {0.3f, 0.3f, 0.3f, 0.3f};
  PolyIOBinding io[] = {POLY_IO_BINDING_ARRAY("x", x, POLY_FLOAT32), POLY_IO_BINDING_ARRAY("y", y, POLY_FLOAT32)};

  float first_loss = -1.0f, prev_loss = 1e10f;
  for (int step = 0; step < 50; step++) {
    float loss;
    ASSERT_INT_EQ(poly_instance_train_step(inst, io, 2, &loss), 0);
    ASSERT_TRUE(isfinite(loss));
    if (step == 0) first_loss = loss;
    prev_loss = loss;
  }
  ASSERT_TRUE(prev_loss < first_loss);
  poly_instance_free(inst);
  PASS();
}

TEST(mlp, train_batch4_cross_entropy) {
  /* batch_size=4: 4 -> 8 -> 3, relu, cross-entropy (P0 regression) */
  const char *spec = "{\"layers\":[4,8,3],\"activation\":\"relu\",\"bias\":true,"
                     "\"loss\":\"cross_entropy\",\"batch_size\":4,\"seed\":42}";

  PolyInstance *inst = poly_mlp_from_json(spec, (int)strlen(spec), POLY_DEVICE_AUTO);
  ASSERT_NOT_NULL(inst);
  poly_instance_set_optimizer(inst, POLY_OPTIM_SGD, 0.01f, 0.0f, 0.0f, 0.0f, 0.0f);

  float x[4 * 4], y[4 * 3];
  for (int i = 0; i < 16; i++)
    x[i] = (float)(i % 7) * 0.1f;
  memset(y, 0, sizeof(y));
  for (int i = 0; i < 4; i++)
    y[i * 3 + (i % 3)] = 1.0f;
  PolyIOBinding io[] = {POLY_IO_BINDING_ARRAY("x", x, POLY_FLOAT32), POLY_IO_BINDING_ARRAY("y", y, POLY_FLOAT32)};

  float first_loss = -1.0f, prev_loss = 1e10f;
  for (int step = 0; step < 30; step++) {
    float loss;
    ASSERT_INT_EQ(poly_instance_train_step(inst, io, 2, &loss), 0);
    ASSERT_TRUE(isfinite(loss));
    if (step == 0) first_loss = loss;
    prev_loss = loss;
  }
  ASSERT_TRUE(prev_loss < first_loss);
  poly_instance_free(inst);
  PASS();
}
