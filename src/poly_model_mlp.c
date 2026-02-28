/*
 * poly_model_mlp.c -- MLP family builder for PolyInstance
 *
 * Builds a tensor-level UOp graph from a JSON spec, exports to IR,
 * then creates a PolyInstance. Deterministic weight init via SplitMix64.
 */

#define _POSIX_C_SOURCE 200809L
#include "poly_model_mlp.h"
#include "poly_ir.h"
#include "frontend.h"
#include "sched.h"
#include "poly_safetensors.h"
#include "../vendor/cjson/cJSON.h"
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

  /* Build graph */
  PolyCtx *ctx = poly_ctx_new();
  int n_linear = n_layers - 1;
  int n_params = n_linear * (use_bias ? 2 : 1);

  /* Allocate interface table entries */
  int max_bufs = n_params + 4;  /* params + x + output + y + loss */
  PolyIrBufEntry *bufs = calloc(max_bufs, sizeof(PolyIrBufEntry));
  int n_bufs = 0;

  /* Param init data (temporary, will be passed via weights) */
  float **param_datas = calloc(n_params, sizeof(float *));
  int64_t *param_numels = calloc(n_params, sizeof(int64_t));
  char **param_names = calloc(n_params, sizeof(char *));
  PolyUOp **param_buf_uops = calloc(n_params, sizeof(PolyUOp *));
  int pi = 0;

  /* Create weight and bias buffers */
  for (int l = 0; l < n_linear; l++) {
    int in_dim = layer_sizes[l];
    int out_dim = layer_sizes[l + 1];

    /* Weight: flat (out_dim * in_dim) */
    int64_t w_numel = (int64_t)out_dim * in_dim;
    char w_name[128];
    snprintf(w_name, sizeof(w_name), "layers.%d.weight", l);

    PolyUOp *w_buf = poly_buffer_f32(ctx, w_numel);
    float *w_data = malloc(w_numel * sizeof(float));
    poly_init_param_kaiming(seed, w_name, w_data, w_numel, (int64_t)in_dim);

    param_buf_uops[pi] = w_buf;
    param_datas[pi] = w_data;
    param_numels[pi] = w_numel;
    param_names[pi] = strdup(w_name);

    bufs[n_bufs++] = (PolyIrBufEntry){
      param_names[pi], POLY_IR_ROLE_PARAM, w_buf,
      { out_dim, in_dim }, 2
    };
    pi++;

    /* Bias: flat (out_dim) */
    if (use_bias) {
      int64_t b_numel = (int64_t)out_dim;
      char b_name[128];
      snprintf(b_name, sizeof(b_name), "layers.%d.bias", l);

      PolyUOp *b_buf = poly_buffer_f32(ctx, b_numel);
      float *b_data = calloc(b_numel, sizeof(float)); /* zero init */

      param_buf_uops[pi] = b_buf;
      param_datas[pi] = b_data;
      param_numels[pi] = b_numel;
      param_names[pi] = strdup(b_name);

      bufs[n_bufs++] = (PolyIrBufEntry){
        param_names[pi], POLY_IR_ROLE_PARAM, b_buf,
        { out_dim }, 1
      };
      pi++;
    }
  }

  /* Input buffer */
  int in_dim = layer_sizes[0];
  int out_dim = layer_sizes[n_layers - 1];
  int64_t x_numel = (int64_t)batch_size * in_dim;
  PolyUOp *x_buf = poly_buffer_f32(ctx, x_numel);
  bufs[n_bufs++] = (PolyIrBufEntry){
    "x", POLY_IR_ROLE_INPUT, x_buf, { batch_size, in_dim }, 2
  };

  /* Output buffer */
  int64_t out_numel = (int64_t)batch_size * out_dim;
  PolyUOp *out_buf = poly_buffer_f32(ctx, out_numel);
  bufs[n_bufs++] = (PolyIrBufEntry){
    "output", POLY_IR_ROLE_OUTPUT, out_buf, { batch_size, out_dim }, 2
  };

  /* Build forward graph: chain of linear + activation */
  PolyUOp *x = x_buf;
  pi = 0;
  for (int l = 0; l < n_linear; l++) {
    int l_in = layer_sizes[l];
    int l_out = layer_sizes[l + 1];

    PolyUOp *w = param_buf_uops[pi++];

    /* Reshape weight from flat to 2D: (out, in) */
    int64_t w_2d[] = { l_out, l_in };
    PolyUOp *w_2d_op = poly_reshape(ctx, w, w_2d, 2);

    /* Transpose: (out, in) -> (in, out) */
    int64_t perm[] = { 1, 0 };
    PolyUOp *wt = poly_permute(ctx, w_2d_op, perm, 2);

    /* Matmul: (batch, in) @ (in, out) -> (batch, out) */
    int64_t x_shape[] = { batch_size, l_in };
    int64_t wt_shape[] = { l_in, l_out };
    int64_t dot_shape[8];
    int dot_ndim;
    x = poly_dot(ctx, x, x_shape, 2, wt, wt_shape, 2, dot_shape, &dot_ndim);

    /* Add bias */
    if (use_bias) {
      PolyUOp *b = param_buf_uops[pi++];
      int64_t b_reshape[] = { 1, l_out };
      PolyUOp *br = poly_reshape(ctx, b, b_reshape, 2);
      int64_t b_expand[] = { batch_size, l_out };
      PolyUOp *be = poly_expand(ctx, br, b_expand, 2);
      x = poly_alu2(ctx, POLY_OP_ADD, x, be);
    }

    /* Activation (skip on last layer) */
    if (l < n_linear - 1) {
      x = apply_activation(ctx, x, activation);
    }
  }

  /* Store forward result */
  PolyUOp *fwd_store = poly_store_val(ctx, out_buf, x);
  PolyUOp *fwd_sink = poly_sink1(ctx, fwd_store);

  /* Build entrypoints */
  int max_eps = 2;
  PolyIrEntrypoint eps[2];
  int n_eps = 0;
  eps[n_eps++] = (PolyIrEntrypoint){ "forward", fwd_sink };

  /* Loss graph */
  int has_loss = (loss_type && (strcmp(loss_type, "mse") == 0 ||
                                strcmp(loss_type, "cross_entropy") == 0));
  if (has_loss) {
    /* Target buffer */
    PolyUOp *y_buf = poly_buffer_f32(ctx, out_numel);
    bufs[n_bufs++] = (PolyIrBufEntry){
      "y", POLY_IR_ROLE_TARGET, y_buf, { batch_size, out_dim }, 2
    };

    /* Loss output buffer */
    PolyUOp *loss_buf = poly_buffer_f32(ctx, 1);
    bufs[n_bufs++] = (PolyIrBufEntry){
      "loss", POLY_IR_ROLE_OUTPUT, loss_buf, { 1 }, 1
    };

    PolyUOp *loss_val;
    if (strcmp(loss_type, "mse") == 0) {
      /* MSE = mean((forward_output - y)^2) */
      PolyUOp *diff = poly_alu2(ctx, POLY_OP_ADD, x,
                                  poly_alu1(ctx, POLY_OP_NEG, y_buf));
      PolyUOp *sq = poly_alu2(ctx, POLY_OP_MUL, diff, diff);

      /* Reduce over all dims */
      int64_t axes0[] = { 0 };
      PolyUOp *sum0 = poly_reduce_axis(ctx, POLY_OP_ADD, sq, axes0, 1);
      int64_t axes1[] = { 0 };
      PolyUOp *sum1 = poly_reduce_axis(ctx, POLY_OP_ADD, sum0, axes1, 1);

      /* Scale by 1/(batch_size * out_dim) */
      double scale = 1.0 / ((double)batch_size * out_dim);
      loss_val = poly_alu2(ctx, POLY_OP_MUL, sum1,
                           poly_const_float(ctx, scale));
    } else {
      /* Cross-entropy = -mean(sum(target * log_softmax(x), axis=-1))
       * target is one-hot: (batch_size, out_dim)
       * log_softmax over axis 1 (class axis) */
      int64_t logits_shape[] = { batch_size, out_dim };
      PolyUOp *log_probs = poly_log_softmax(ctx, x, logits_shape, 2, 1);

      /* -(target * log_probs) summed over class axis, then mean over batch */
      PolyUOp *prod = poly_alu2(ctx, POLY_OP_MUL, y_buf, log_probs);
      int64_t axes_class[] = { 1 };
      PolyUOp *sum_class = poly_reduce_axis(ctx, POLY_OP_ADD, prod, axes_class, 1);
      int64_t axes_batch[] = { 0 };
      PolyUOp *sum_batch = poly_reduce_axis(ctx, POLY_OP_ADD, sum_class, axes_batch, 1);

      /* Scale by -1/batch_size */
      double scale = -1.0 / (double)batch_size;
      loss_val = poly_alu2(ctx, POLY_OP_MUL, sum_batch,
                           poly_const_float(ctx, scale));
    }

    PolyUOp *loss_store = poly_store_val(ctx, loss_buf, loss_val);
    PolyUOp *loss_sink = poly_sink1(ctx, loss_store);
    eps[n_eps++] = (PolyIrEntrypoint){ "loss", loss_sink };
  }

  (void)max_eps;

  /* Export to IR */
  PolyIrSpec spec = { ctx, bufs, n_bufs, eps, n_eps };
  int ir_len = 0;
  uint8_t *ir_data = poly_ir_export(&spec, &ir_len);

  /* Encode weights as safetensors */
  PolySafetensorEntry *st_entries = malloc(pi * sizeof(PolySafetensorEntry));
  int n_st = 0;
  for (int i = 0; i < pi; i++) {
    /* Find the buf entry for this param to get its shape */
    for (int j = 0; j < n_bufs; j++) {
      if (strcmp(bufs[j].name, param_names[i]) == 0) {
        st_entries[n_st].name = param_names[i];
        st_entries[n_st].data = param_datas[i];
        st_entries[n_st].shape = bufs[j].shape;
        st_entries[n_st].ndim = bufs[j].ndim;
        n_st++;
        break;
      }
    }
  }
  int st_len = 0;
  uint8_t *st_data = poly_safetensors_encode(st_entries, n_st, NULL, &st_len);
  free(st_entries);

  /* Create instance from IR + weights */
  PolyInstance *inst = NULL;
  if (ir_data && st_data) {
    inst = poly_instance_from_ir(ir_data, ir_len, st_data, st_len);
  }

  /* Cleanup */
  for (int i = 0; i < pi; i++) {
    free(param_datas[i]);
    free(param_names[i]);
  }
  free(param_datas);
  free(param_numels);
  free(param_names);
  free(param_buf_uops);
  free(bufs);
  free(layer_sizes);
  free(ir_data);
  free(st_data);
  poly_ctx_destroy(ctx);
  cJSON_Delete(root);

  return inst;
}
