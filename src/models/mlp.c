/*
 * poly_model_mlp.c -- MLP family builder for PolyInstance
 *
 * Builds a tensor-level UOp graph from a JSON spec, exports to IR,
 * then creates a PolyInstance. Deterministic weight init via SplitMix64.
 */

#define _POSIX_C_SOURCE 200809L
#include "mlp.h"
#include "../nn.h"
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
  return (float)(r >> 40) * 0x1.0p-24f;  /* [0, 1) uniform */
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

typedef enum { ACT_NONE, ACT_RELU, ACT_GELU, ACT_SILU, ACT_TANH, ACT_SIGMOID } ActivationKind;

static ActivationKind parse_activation(const char *s) {
  if (!s || strcmp(s, "none") == 0) return ACT_NONE;
  if (strcmp(s, "relu") == 0) return ACT_RELU;
  if (strcmp(s, "gelu") == 0) return ACT_GELU;
  if (strcmp(s, "silu") == 0) return ACT_SILU;
  if (strcmp(s, "tanh") == 0) return ACT_TANH;
  if (strcmp(s, "sigmoid") == 0) return ACT_SIGMOID;
  return ACT_RELU; /* default */
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

/* ── MLP Builder ─────────────────────────────────────────────────────── */

PolyInstance *poly_mlp_instance(const char *spec_json, int spec_len) {
  if (!spec_json || spec_len <= 0) return NULL;

  /* Parse JSON */
  cJSON *root = cJSON_ParseWithLength(spec_json, (size_t)spec_len);
  if (!root) {
    fprintf(stderr, "poly_mlp_instance: JSON parse error\n");
    return NULL;
  }

  /* Extract fields */
  cJSON *layers_arr = cJSON_GetObjectItem(root, "layers");
  cJSON *act_item = cJSON_GetObjectItem(root, "activation");
  cJSON *bias_item = cJSON_GetObjectItem(root, "bias");
  cJSON *loss_item = cJSON_GetObjectItem(root, "loss");
  cJSON *batch_item = cJSON_GetObjectItem(root, "batch_size");
  cJSON *seed_item = cJSON_GetObjectItem(root, "seed");

  if (!layers_arr || !cJSON_IsArray(layers_arr)) {
    fprintf(stderr, "poly_mlp_instance: 'layers' must be an array\n");
    cJSON_Delete(root);
    return NULL;
  }

  int n_layers = cJSON_GetArraySize(layers_arr);
  if (n_layers < 2) {
    fprintf(stderr, "poly_mlp_instance: need at least 2 layers\n");
    cJSON_Delete(root);
    return NULL;
  }

  int *layer_sizes = malloc(n_layers * sizeof(int));
  for (int i = 0; i < n_layers; i++) {
    cJSON *item = cJSON_GetArrayItem(layers_arr, i);
    layer_sizes[i] = item ? item->valueint : 0;
  }

  ActivationKind activation = parse_activation(
      act_item ? act_item->valuestring : "relu");
  int use_bias = bias_item ? cJSON_IsTrue(bias_item) : 1;
  const char *loss_type = loss_item ? loss_item->valuestring : "none";
  int batch_size = batch_item ? batch_item->valueint : 1;
  uint64_t seed = seed_item ? (uint64_t)seed_item->valuedouble : 42;

  if (batch_size < 1) batch_size = 1;

  /* Build graph using named buffer registry */
  PolyCtx *ctx = poly_ctx_new();
  int n_linear = n_layers - 1;
  int in_dim = layer_sizes[0];
  int out_dim = layer_sizes[n_layers - 1];

  /* Register I/O buffers */
  int64_t x_shape[] = { batch_size, in_dim };
  PolyUOp *x_buf = poly_input(ctx, POLY_FLOAT32, x_shape, 2, "x");
  int64_t out_shape[] = { batch_size, out_dim };
  PolyUOp *out_buf = poly_output(ctx, POLY_FLOAT32, out_shape, 2, "output");

  /* Build forward graph: chain of linear + activation using nn builders */
  PolyUOp *x = poly_reshape(ctx, x_buf, x_shape, 2);
  for (int l = 0; l < n_linear; l++) {
    char prefix[64];
    snprintf(prefix, sizeof(prefix), "layers.%d", l);
    x = poly_linear(ctx, prefix, x, layer_sizes[l], layer_sizes[l + 1], use_bias);

    /* Activation (skip on last layer) */
    if (l < n_linear - 1)
      x = apply_activation(ctx, x, activation);
  }

  /* Store forward result */
  PolyUOp *fwd_store = poly_store_val(ctx, out_buf, x);
  PolyUOp *fwd_sink = poly_sink1(ctx, fwd_store);
  poly_register_entrypoint(ctx, "forward", fwd_sink);

  /* Loss graph */
  int has_loss = (loss_type && (strcmp(loss_type, "mse") == 0 ||
                                strcmp(loss_type, "cross_entropy") == 0));
  if (has_loss) {
    int64_t y_shape[] = { batch_size, out_dim };
    PolyUOp *y_buf = poly_target(ctx, POLY_FLOAT32, y_shape, 2, "y");
    PolyUOp *loss_buf = poly_output(ctx, POLY_FLOAT32, (int64_t[]){1}, 1, "loss");

    PolyUOp *y = poly_reshape(ctx, y_buf, y_shape, 2);
    PolyUOp *loss_val;
    if (strcmp(loss_type, "mse") == 0) {
      PolyUOp *diff = poly_alu2(ctx, POLY_OP_ADD, x,
                                  poly_alu1(ctx, POLY_OP_NEG, y));
      PolyUOp *sq = poly_alu2(ctx, POLY_OP_MUL, diff, diff);
      int64_t axes1[] = { 1 };
      PolyUOp *sum0 = poly_reduce_axis(ctx, POLY_OP_ADD, sq, axes1, 1);
      int64_t axes0[] = { 0 };
      PolyUOp *sum1 = poly_reduce_axis(ctx, POLY_OP_ADD, sum0, axes0, 1);
      double scale = 1.0 / ((double)batch_size * out_dim);
      loss_val = poly_alu2(ctx, POLY_OP_MUL, sum1,
                           poly_const_float(ctx, scale));
    } else {
      PolyUOp *log_probs = poly_log_softmax(ctx, x, 1);
      PolyUOp *prod = poly_alu2(ctx, POLY_OP_MUL, y, log_probs);
      int64_t axes1[] = { 1 };
      PolyUOp *sum_class = poly_reduce_axis(ctx, POLY_OP_ADD, prod, axes1, 1);
      int64_t axes0[] = { 0 };
      PolyUOp *sum_batch = poly_reduce_axis(ctx, POLY_OP_ADD, sum_class, axes0, 1);
      double scale = -1.0 / (double)batch_size;
      loss_val = poly_alu2(ctx, POLY_OP_MUL, sum_batch,
                           poly_const_float(ctx, scale));
    }

    PolyUOp *loss_store = poly_store_val(ctx, loss_buf, loss_val);
    PolyUOp *loss_sink = poly_sink1(ctx, loss_store);
    poly_register_entrypoint(ctx, "loss", loss_sink);
  }

  /* Create instance from ctx registry */
  PolyInstance *inst = poly_instance_from_ctx(ctx);

  /* Initialize weights (Kaiming) directly in instance buffers */
  if (inst) {
    for (int l = 0; l < n_linear; l++) {
      int l_in = layer_sizes[l];
      int l_out = layer_sizes[l + 1];
      int64_t w_numel = (int64_t)l_out * l_in;
      char name[128];
      snprintf(name, sizeof(name), "layers.%d.weight", l);
      float *w_data = poly_instance_buf_data_named(inst, name, NULL);
      if (w_data) poly_init_param_kaiming(seed, name, w_data, w_numel, (int64_t)l_in);
      /* Bias stays zero-initialized (calloc in instance_from_spec) */
    }
  }

  /* Transfer ctx ownership to instance (builder-created ctx) */
  if (inst) poly_instance_own_ctx(inst);

  free(layer_sizes);
  cJSON_Delete(root);
  return inst;
}
