/*
 * poly_model_mlp.c -- MLP family builder for PolyInstance
 *
 * Two entry points:
 *   poly_mlp(cfg)         -- builds from MLPConfig struct (like GPT-2)
 *   poly_mlp_from_json(json,len) -- FFI wrapper that parses JSON then calls build
 *
 * Deterministic weight init via SplitMix64.
 */

#define _POSIX_C_SOURCE 200809L
#include "mlp.h"
#include "../nn.h"
#include "../tensor.h"
#include "../instance.h"
#include "../../vendor/cjson/cJSON.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/* Stateless PRNG (SplitMix64) */

static uint64_t splitmix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}

static float prng_float(uint64_t seed, uint64_t stream, uint64_t idx) {
  uint64_t r = splitmix64(seed ^ splitmix64(stream) ^ splitmix64(idx));
  return (float)(r >> 40) * 0x1.0p-24f;
}

static uint64_t fnv1a_64(const char *s, size_t len) {
  uint64_t h = 0xcbf29ce484222325ULL;
  for (size_t i = 0; i < len; i++)
    h = (h ^ (uint8_t)s[i]) * 0x100000001b3ULL;
  return h;
}

void poly_init_param_kaiming(
    uint64_t seed,
    const char *name,
    float *data,
    int64_t numel,
    int64_t fan_in
) {
  uint64_t stream = fnv1a_64(name, strlen(name));
  float bound = sqrtf(6.0f / (float)fan_in);
  for (int64_t i = 0; i < numel; i++)
    data[i] = (prng_float(seed, stream, (uint64_t)i) * 2.0f - 1.0f) * bound;
}

/* Activation dispatch */

typedef enum { ACT_NONE, ACT_RELU, ACT_GELU, ACT_SILU, ACT_TANH, ACT_SIGMOID } ActivationKind;

static ActivationKind parse_activation(const char *s) {
  if (!s || strcmp(s, "none") == 0) return ACT_NONE;
  if (strcmp(s, "relu") == 0) return ACT_RELU;
  if (strcmp(s, "gelu") == 0) return ACT_GELU;
  if (strcmp(s, "silu") == 0) return ACT_SILU;
  if (strcmp(s, "tanh") == 0) return ACT_TANH;
  if (strcmp(s, "sigmoid") == 0) return ACT_SIGMOID;
  return ACT_RELU;
}

static PolyUOp *apply_activation(PolyCtx *ctx, PolyUOp *x, ActivationKind act) {
  switch (act) {
  case ACT_RELU:
    return poly_relu(ctx, x);
  case ACT_GELU:
    return poly_gelu(ctx, x);
  case ACT_SILU:
    return poly_silu(ctx, x);
  case ACT_TANH:
    return poly_tanh_act(ctx, x);
  case ACT_SIGMOID:
    return poly_sigmoid(ctx, x);
  case ACT_NONE:
    return x;
  }
  return x;
}

/* MLP Config */

MLPConfig poly_mlp_config_default(void) {
  return (MLPConfig){
      .n_layers = 0,
      .activation = "relu",
      .use_bias = 1,
      .loss = "none",
      .batch_size = 1,
      .seed = 42,
  };
}

/* MLP Builder (config struct) */

PolyInstance *poly_mlp(const MLPConfig *cfg) {
  if (!cfg || cfg->n_layers < 2 || cfg->n_layers > POLY_MLP_MAX_LAYERS) return NULL;

  int n_linear = cfg->n_layers - 1;
  int in_dim = cfg->layers[0];
  int out_dim = cfg->layers[cfg->n_layers - 1];
  int batch_size = cfg->batch_size > 0 ? cfg->batch_size : 1;
  ActivationKind activation = parse_activation(cfg->activation);
  const char *loss_type = cfg->loss ? cfg->loss : "none";

  PolyCtx *ctx = poly_ctx_new();
  if (!ctx) return NULL;
  PolyInstanceOptions opts = {
      .own_ctx_on_success = true,
      .own_ctx_on_failure = true,
  };
  PolyInstance *inst = poly_instance_new(ctx, &opts);
  if (!inst) {
    poly_ctx_destroy(ctx);
    return NULL;
  }

  /* Declare I/O buffers on the staged instance. */
  int64_t x_shape[] = {batch_size, in_dim};
  PolyTensor *x_tensor = poly_instance_input(inst, "x", POLY_FLOAT32, x_shape, 2);
  if (!x_tensor) goto fail_pre_build;

  /* Forward: chain of linear + activation. */
  PolyUOp *x = poly_tensor_uop(x_tensor);
  for (int l = 0; l < n_linear; l++) {
    if (poly_instance_scope_push(inst, "layers.%d", l) != POLY_STATUS_OK) goto fail_pre_build;

    int64_t ws[] = {cfg->layers[l + 1], cfg->layers[l]};
    PolyTensor *w_tensor = poly_instance_param(inst, "weight", POLY_FLOAT32, ws, 2);
    if (!w_tensor) goto fail_pre_build;
    PolyUOp *w = poly_tensor_uop(w_tensor);

    PolyUOp *b = NULL;
    if (cfg->use_bias) {
      int64_t bs[] = {cfg->layers[l + 1]};
      PolyTensor *b_tensor = poly_instance_param(inst, "bias", POLY_FLOAT32, bs, 1);
      if (!b_tensor) goto fail_pre_build;
      b = poly_tensor_uop(b_tensor);
    }

    if (poly_instance_scope_pop(inst) != POLY_STATUS_OK) goto fail_pre_build;

    x = poly_linear_apply(ctx, x, w, b);
    if (!x) goto fail_pre_build;
    if (l < n_linear - 1) x = apply_activation(ctx, x, activation);
    if (!x) goto fail_pre_build;
  }

  PolyTensor *out_tensor = poly_tensor_create(ctx, x, POLY_TENSOR_VALUE, POLY_DEVICE_AUTO);
  if (!out_tensor || poly_instance_output(inst, "output", out_tensor) != POLY_STATUS_OK)
    goto fail_pre_build;
  const char *forward_inputs[] = {"x"};
  const char *forward_outputs[] = {"output"};
  if (poly_instance_entrypoint(inst, "forward", forward_inputs, 1, forward_outputs, 1, NULL) !=
      POLY_STATUS_OK)
    goto fail_pre_build;

  /* Loss graph. */
  int has_loss =
      loss_type && (strcmp(loss_type, "mse") == 0 || strcmp(loss_type, "cross_entropy") == 0);
  if (has_loss) {
    int64_t y_shape[] = {batch_size, out_dim};
    PolyTensor *y_tensor = poly_instance_target(inst, "y", POLY_FLOAT32, y_shape, 2);
    if (!y_tensor) goto fail_pre_build;
    PolyUOp *y = poly_tensor_uop(y_tensor);

    PolyUOp *loss_val = (strcmp(loss_type, "mse") == 0) ? poly_mse_loss(ctx, x, y)
                                                        : poly_cross_entropy(ctx, x, y, -1);
    if (!loss_val) goto fail_pre_build;

    PolyTensor *loss_tensor =
        poly_tensor_create(ctx, loss_val, POLY_TENSOR_VALUE, POLY_DEVICE_AUTO);
    if (!loss_tensor || poly_instance_output(inst, "loss", loss_tensor) != POLY_STATUS_OK)
      goto fail_pre_build;
    const char *loss_inputs[] = {"x", "y"};
    const char *loss_outputs[] = {"loss"};
    PolyEntrypointOptions loss_opts = {.objective = "loss"};
    if (poly_instance_entrypoint(inst, "loss", loss_inputs, 2, loss_outputs, 1, &loss_opts) !=
        POLY_STATUS_OK)
      goto fail_pre_build;
  }

  PolyInstanceError err = {0};
  if (poly_instance_build(inst, &err) != POLY_STATUS_OK) {
    if (err.message[0]) fprintf(stderr, "poly_mlp: build failed: %s\n", err.message);
    poly_instance_free(inst);
    return NULL;
  }

  /* Init weights after build so runtime buffers keep the same legacy data API. */
  for (int l = 0; l < n_linear; l++) {
    char name[128];
    snprintf(name, sizeof(name), "layers.%d.weight", l);
    float *w = poly_instance_buf_data_named(inst, name, NULL);
    if (w)
      poly_init_param_kaiming(
          cfg->seed, name, w, (int64_t)cfg->layers[l + 1] * cfg->layers[l], (int64_t)cfg->layers[l]
      );
  }

  return inst;

fail_pre_build:
  poly_instance_free(inst);
  poly_ctx_destroy(ctx);
  return NULL;
}

/* FFI wrapper (JSON -> config -> build) */

PolyInstance *poly_mlp_from_json(const char *json, int len) {
  if (!json || len <= 0) return NULL;

  cJSON *root = cJSON_ParseWithLength(json, (size_t)len);
  if (!root) return NULL;

  cJSON *layers = cJSON_GetObjectItem(root, "layers");
  if (!layers || !cJSON_IsArray(layers) || cJSON_GetArraySize(layers) < 2) {
    fprintf(stderr, "poly_mlp_from_json: 'layers' must be an array with >= 2 entries\n");
    cJSON_Delete(root);
    return NULL;
  }

  MLPConfig cfg = poly_mlp_config_default();
  cJSON *v;
  if ((v = cJSON_GetObjectItem(root, "activation"))) cfg.activation = v->valuestring;
  if ((v = cJSON_GetObjectItem(root, "bias"))) cfg.use_bias = cJSON_IsTrue(v);
  if ((v = cJSON_GetObjectItem(root, "loss"))) cfg.loss = v->valuestring;
  if ((v = cJSON_GetObjectItem(root, "batch_size"))) cfg.batch_size = v->valueint;
  if ((v = cJSON_GetObjectItem(root, "seed"))) cfg.seed = (uint64_t)v->valuedouble;

  cfg.n_layers = cJSON_GetArraySize(layers);
  if (cfg.n_layers > POLY_MLP_MAX_LAYERS) cfg.n_layers = POLY_MLP_MAX_LAYERS;
  for (int i = 0; i < cfg.n_layers; i++)
    cfg.layers[i] = cJSON_GetArrayItem(layers, i)->valueint;

  PolyInstance *inst = poly_mlp(&cfg);
  cJSON_Delete(root);
  return inst;
}
