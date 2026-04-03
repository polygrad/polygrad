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
#include "../frontend.h"
#include "../scheduler.h"
#include "../../vendor/cjson/cJSON.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/* ── Stateless PRNG (SplitMix64) ────────────────────────────────────── */

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

void poly_init_param_kaiming(uint64_t seed, const char *name,
                              float *data, int64_t numel, int64_t fan_in) {
  uint64_t stream = fnv1a_64(name, strlen(name));
  float bound = sqrtf(6.0f / (float)fan_in);
  for (int64_t i = 0; i < numel; i++)
    data[i] = (prng_float(seed, stream, (uint64_t)i) * 2.0f - 1.0f) * bound;
}

/* ── Activation dispatch ─────────────────────────────────────────────── */

typedef enum {
  ACT_NONE, ACT_RELU, ACT_GELU, ACT_SILU, ACT_TANH, ACT_SIGMOID
} ActivationKind;

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
  case ACT_RELU:    return poly_relu(ctx, x);
  case ACT_GELU:    return poly_gelu(ctx, x);
  case ACT_SILU:    return poly_silu(ctx, x);
  case ACT_TANH:    return poly_tanh_act(ctx, x);
  case ACT_SIGMOID: return poly_sigmoid(ctx, x);
  case ACT_NONE:    return x;
  }
  return x;
}

/* ── MLP Config ──────────────────────────────────────────────────────── */

MLPConfig poly_mlp_config_default(void) {
  return (MLPConfig){
    .n_layers   = 0,
    .activation = "relu",
    .use_bias   = 1,
    .loss       = "none",
    .batch_size = 1,
    .seed       = 42,
  };
}

/* ── MLP Builder (config struct) ─────────────────────────────────────── */

PolyInstance *poly_mlp(const MLPConfig *cfg) {
  if (!cfg || cfg->n_layers < 2 || cfg->n_layers > POLY_MLP_MAX_LAYERS)
    return NULL;

  int n_linear = cfg->n_layers - 1;
  int in_dim = cfg->layers[0];
  int out_dim = cfg->layers[cfg->n_layers - 1];
  int batch_size = cfg->batch_size > 0 ? cfg->batch_size : 1;
  ActivationKind activation = parse_activation(cfg->activation);
  const char *loss_type = cfg->loss ? cfg->loss : "none";

  PolyCtx *ctx = poly_ctx_new();

  /* Register I/O buffers */
  int64_t x_shape[] = { batch_size, in_dim };
  PolyUOp *x_buf = poly_input(ctx, POLY_FLOAT32, x_shape, 2, "x");
  int64_t out_shape[] = { batch_size, out_dim };
  PolyUOp *out_buf = poly_output(ctx, POLY_FLOAT32, out_shape, 2, "output");

  /* Forward: chain of linear + activation */
  PolyUOp *x = poly_reshape(ctx, x_buf, x_shape, 2);
  for (int l = 0; l < n_linear; l++) {
    char prefix[64];
    snprintf(prefix, sizeof(prefix), "layers.%d", l);
    x = poly_linear(ctx, prefix, x,
                     cfg->layers[l], cfg->layers[l + 1], cfg->use_bias);
    if (l < n_linear - 1)
      x = apply_activation(ctx, x, activation);
  }

  poly_register_entrypoint(ctx, "forward",
      poly_sink1(ctx, poly_store_val(ctx, out_buf, x)));

  /* Loss graph */
  int has_loss = loss_type &&
      (strcmp(loss_type, "mse") == 0 || strcmp(loss_type, "cross_entropy") == 0);
  if (has_loss) {
    int64_t y_shape[] = { batch_size, out_dim };
    PolyUOp *y = poly_reshape(ctx,
        poly_target(ctx, POLY_FLOAT32, y_shape, 2, "y"), y_shape, 2);
    PolyUOp *loss_buf = poly_output(ctx, POLY_FLOAT32, (int64_t[]){1}, 1, "loss");

    PolyUOp *loss_val = (strcmp(loss_type, "mse") == 0)
        ? poly_mse_loss(ctx, x, y)
        : poly_cross_entropy(ctx, x, y, -1);

    poly_register_entrypoint(ctx, "loss",
        poly_sink1(ctx, poly_store_val(ctx, loss_buf, loss_val)));
  }

  /* Create instance and init weights */
  PolyInstance *inst = poly_instance_from_ctx(ctx);
  if (inst) {
    for (int l = 0; l < n_linear; l++) {
      char name[128];
      snprintf(name, sizeof(name), "layers.%d.weight", l);
      float *w = poly_instance_buf_data_named(inst, name, NULL);
      if (w)
        poly_init_param_kaiming(cfg->seed, name, w,
            (int64_t)cfg->layers[l + 1] * cfg->layers[l],
            (int64_t)cfg->layers[l]);
    }
    poly_instance_own_ctx(inst);
  }

  return inst;
}

/* ── FFI wrapper (JSON -> config -> build) ───────────────────────────── */

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
  if ((v = cJSON_GetObjectItem(root, "bias")))        cfg.use_bias   = cJSON_IsTrue(v);
  if ((v = cJSON_GetObjectItem(root, "loss")))        cfg.loss       = v->valuestring;
  if ((v = cJSON_GetObjectItem(root, "batch_size")))  cfg.batch_size = v->valueint;
  if ((v = cJSON_GetObjectItem(root, "seed")))        cfg.seed       = (uint64_t)v->valuedouble;

  cfg.n_layers = cJSON_GetArraySize(layers);
  if (cfg.n_layers > POLY_MLP_MAX_LAYERS) cfg.n_layers = POLY_MLP_MAX_LAYERS;
  for (int i = 0; i < cfg.n_layers; i++)
    cfg.layers[i] = cJSON_GetArrayItem(layers, i)->valueint;

  PolyInstance *inst = poly_mlp(&cfg);
  cJSON_Delete(root);
  return inst;
}
