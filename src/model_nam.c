/*
 * model_nam.c -- NAM (Neural Additive Model) builder for PolyInstance
 *
 * Builds a tensor-level UOp graph from a JSON spec, exports to IR,
 * then creates a PolyInstance. Same pattern as model_mlp.c / model_tabm.c.
 *
 * NAM forward: g(E[y]) = intercept + sum(fk(xk) for k in 0..K-1)
 * Each fk is a small MLP operating on scalar feature xk.
 *
 * With ExU activation (default):
 *   ExU(x) = ReLU(exp(w) * (x - b)) where w, b are learnable per-unit
 *
 * Reference: Agarwal et al. (2021), arXiv:2004.13912 (NeurIPS 2021)
 */

#define _POSIX_C_SOURCE 200809L
#include "model_nam.h"
#include "model_mlp.h"  /* poly_init_param_kaiming */
#include "ir.h"
#include "frontend.h"
#include "scheduler.h"
#include "safetensors.h"
#include "../vendor/cjson/cJSON.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/* ── Activation types ──────────────────────────────────────────────── */

typedef enum { NAM_ACT_RELU, NAM_ACT_GELU, NAM_ACT_SILU, NAM_ACT_EXU } NamActivation;

static NamActivation nam_parse_activation(const char *s) {
  if (!s || strcmp(s, "exu") == 0) return NAM_ACT_EXU;
  if (strcmp(s, "relu") == 0) return NAM_ACT_RELU;
  if (strcmp(s, "gelu") == 0) return NAM_ACT_GELU;
  if (strcmp(s, "silu") == 0) return NAM_ACT_SILU;
  return NAM_ACT_EXU;
}

/* ── Dynamic param list ────────────────────────────────────────────── */

typedef struct {
  float *data;
  int64_t numel;
  char *name;
  PolyUOp *buf;
  int64_t shape[4];
  int ndim;
} NamParam;

typedef struct {
  NamParam *items;
  int count;
  int capacity;
} NamParamList;

static void nam_param_init(NamParamList *list) {
  list->count = 0;
  list->capacity = 64;
  list->items = malloc(list->capacity * sizeof(NamParam));
}

static int nam_param_add(NamParamList *list, PolyCtx *ctx,
                         const char *name, int64_t numel,
                         const int64_t *shape, int ndim,
                         float *init_data) {
  if (list->count >= list->capacity) {
    list->capacity *= 2;
    list->items = realloc(list->items, list->capacity * sizeof(NamParam));
  }
  int idx = list->count++;
  NamParam *p = &list->items[idx];
  p->buf = poly_buffer_f32(ctx, numel);
  p->data = init_data;
  p->numel = numel;
  p->name = strdup(name);
  p->ndim = ndim;
  for (int i = 0; i < ndim && i < 4; i++) p->shape[i] = shape[i];
  return idx;
}

static void nam_param_free(NamParamList *list) {
  for (int i = 0; i < list->count; i++) {
    free(list->items[i].data);
    free(list->items[i].name);
  }
  free(list->items);
}

/* ── NAM Builder ───────────────────────────────────────────────────── */

PolyInstance *poly_nam_instance(const char *spec_json, int spec_len) {
  if (!spec_json || spec_len <= 0) return NULL;

  /* Parse JSON */
  cJSON *root = cJSON_ParseWithLength(spec_json, (size_t)spec_len);
  if (!root) {
    fprintf(stderr, "poly_nam_instance: JSON parse error\n");
    return NULL;
  }

  /* Extract fields */
  cJSON *nf_item = cJSON_GetObjectItem(root, "n_features");
  cJSON *hs_item = cJSON_GetObjectItem(root, "hidden_sizes");
  cJSON *act_item = cJSON_GetObjectItem(root, "activation");
  cJSON *no_item = cJSON_GetObjectItem(root, "n_outputs");
  cJSON *loss_item = cJSON_GetObjectItem(root, "loss");
  cJSON *batch_item = cJSON_GetObjectItem(root, "batch_size");
  cJSON *seed_item = cJSON_GetObjectItem(root, "seed");

  if (!nf_item || !cJSON_IsNumber(nf_item)) {
    fprintf(stderr, "poly_nam_instance: 'n_features' required\n");
    cJSON_Delete(root);
    return NULL;
  }
  int n_features = nf_item->valueint;
  if (n_features < 1) {
    fprintf(stderr, "poly_nam_instance: n_features must be >= 1\n");
    cJSON_Delete(root);
    return NULL;
  }

  /* Parse hidden_sizes array */
  int n_hidden = 1;
  int hidden_sizes[16] = { 64 };
  if (hs_item && cJSON_IsArray(hs_item)) {
    n_hidden = cJSON_GetArraySize(hs_item);
    if (n_hidden > 16) n_hidden = 16;
    for (int i = 0; i < n_hidden; i++) {
      hidden_sizes[i] = cJSON_GetArrayItem(hs_item, i)->valueint;
    }
  }

  NamActivation activation = nam_parse_activation(
      act_item ? act_item->valuestring : NULL);
  int n_outputs = no_item ? no_item->valueint : 1;
  if (n_outputs < 1) n_outputs = 1;
  const char *loss_type = loss_item ? loss_item->valuestring : "none";
  int batch_size = batch_item ? batch_item->valueint : 1;
  uint64_t seed = seed_item ? (uint64_t)seed_item->valuedouble : 42;

  if (batch_size < 1) batch_size = 1;

  /* Build layer sizes for each feature subnet: [1, h1, h2, ..., n_outputs] */
  int n_layers = n_hidden + 2;  /* input(1) + hidden... + output */
  int *subnet_sizes = malloc(n_layers * sizeof(int));
  subnet_sizes[0] = 1;
  for (int i = 0; i < n_hidden; i++) subnet_sizes[i + 1] = hidden_sizes[i];
  subnet_sizes[n_layers - 1] = n_outputs;
  int n_linear = n_layers - 1;  /* number of linear transformations */

  /* ── Create param list ────────────────────────────────────────────── */

  PolyCtx *ctx = poly_ctx_new();
  NamParamList params;
  nam_param_init(&params);

  /* Intercept: (n_outputs,) -- zero init */
  {
    float *data = calloc(n_outputs, sizeof(float));
    int64_t shape[] = { n_outputs };
    nam_param_add(&params, ctx, "intercept", n_outputs, shape, 1, data);
  }

  /* Per-feature subnet params */
  for (int k = 0; k < n_features; k++) {
    for (int l = 0; l < n_linear; l++) {
      int in_dim = subnet_sizes[l];
      int out_dim = subnet_sizes[l + 1];

      /* Weight: (out_dim, in_dim) -- Kaiming init */
      {
        int64_t numel = (int64_t)out_dim * in_dim;
        char name[128];
        snprintf(name, sizeof(name), "features.%d.layers.%d.weight", k, l);
        float *data = malloc(numel * sizeof(float));
        poly_init_param_kaiming(seed, name, data, numel, (int64_t)in_dim);
        int64_t shape[] = { out_dim, in_dim };
        nam_param_add(&params, ctx, name, numel, shape, 2, data);
      }

      /* Bias: (out_dim,) -- zero init */
      {
        int64_t numel = out_dim;
        char name[128];
        snprintf(name, sizeof(name), "features.%d.layers.%d.bias", k, l);
        float *data = calloc(numel, sizeof(float));
        int64_t shape[] = { out_dim };
        nam_param_add(&params, ctx, name, numel, shape, 1, data);
      }

      /* ExU params (only for hidden layers, not final output layer) */
      if (activation == NAM_ACT_EXU && l < n_linear - 1) {
        /* ExU weight: (out_dim,) -- zero init (exp(0)=1) */
        {
          char name[128];
          snprintf(name, sizeof(name), "features.%d.exu.%d.weight", k, l);
          float *data = calloc(out_dim, sizeof(float));
          int64_t shape[] = { out_dim };
          nam_param_add(&params, ctx, name, out_dim, shape, 1, data);
        }
        /* ExU bias: (out_dim,) -- zero init */
        {
          char name[128];
          snprintf(name, sizeof(name), "features.%d.exu.%d.bias", k, l);
          float *data = calloc(out_dim, sizeof(float));
          int64_t shape[] = { out_dim };
          nam_param_add(&params, ctx, name, out_dim, shape, 1, data);
        }
      }
    }
  }

  /* ── Create I/O buffers ───────────────────────────────────────────── */

  /* Max bufs: params + x + output + y + loss */
  int max_bufs = params.count + 4;
  PolyIrBufEntry *bufs = calloc(max_bufs, sizeof(PolyIrBufEntry));
  int n_bufs = 0;

  /* Register all params as IR bufs */
  for (int i = 0; i < params.count; i++) {
    NamParam *p = &params.items[i];
    bufs[n_bufs] = (PolyIrBufEntry){
      p->name, POLY_IR_ROLE_PARAM, p->buf,
      { 0 }, p->ndim
    };
    for (int d = 0; d < p->ndim; d++) bufs[n_bufs].shape[d] = p->shape[d];
    n_bufs++;
  }

  /* Input: (batch_size, n_features) */
  int64_t x_numel = (int64_t)batch_size * n_features;
  PolyUOp *x_buf = poly_buffer_f32(ctx, x_numel);
  bufs[n_bufs++] = (PolyIrBufEntry){
    "x", POLY_IR_ROLE_INPUT, x_buf, { batch_size, n_features }, 2
  };

  /* Output: (batch_size, n_outputs) */
  int64_t out_numel = (int64_t)batch_size * n_outputs;
  PolyUOp *out_buf = poly_buffer_f32(ctx, out_numel);
  bufs[n_bufs++] = (PolyIrBufEntry){
    "output", POLY_IR_ROLE_OUTPUT, out_buf, { batch_size, n_outputs }, 2
  };

  /* ── Build forward graph ──────────────────────────────────────────── */

  /* Start with intercept broadcast to (batch_size, n_outputs) */
  PolyUOp *intercept_buf = params.items[0].buf;
  int64_t intercept_shape[] = { 1, n_outputs };
  PolyUOp *accum = poly_reshape(ctx, intercept_buf, intercept_shape, 2);
  int64_t accum_expanded[] = { batch_size, n_outputs };
  accum = poly_expand(ctx, accum, accum_expanded, 2);

  /* Reshape flat x_buf to 2D for feature extraction via shrink */
  int64_t x_2d_shape[] = { batch_size, n_features };
  PolyUOp *x_2d = poly_reshape(ctx, x_buf, x_2d_shape, 2);

  /* Process each feature subnet */
  int pi = 1;  /* param index (0 = intercept) */
  for (int k = 0; k < n_features; k++) {
    /* Extract feature k: shrink x from (batch, n_features) to (batch, 1) */
    int64_t shrink_pairs[][2] = {
      { 0, batch_size },
      { k, k + 1 }
    };
    PolyUOp *xk = poly_shrink(ctx, x_2d, shrink_pairs, 2);

    /* Run through MLP layers */
    for (int l = 0; l < n_linear; l++) {
      int in_dim = subnet_sizes[l];
      int out_dim = subnet_sizes[l + 1];

      /* Get weight and bias */
      PolyUOp *w = params.items[pi++].buf;
      PolyUOp *bias = params.items[pi++].buf;

      /* Reshape weight to (out_dim, in_dim) */
      int64_t w_shape[] = { out_dim, in_dim };
      PolyUOp *w_2d = poly_reshape(ctx, w, w_shape, 2);

      /* Transpose to (in_dim, out_dim) */
      int64_t perm[] = { 1, 0 };
      PolyUOp *wt = poly_permute(ctx, w_2d, perm, 2);

      /* xk @ W^T: (batch, in_dim) @ (in_dim, out_dim) -> (batch, out_dim) */
      int64_t xk_shape[] = { batch_size, in_dim };
      int64_t wt_shape[] = { in_dim, out_dim };
      int64_t dot_shape[8];
      int dot_ndim;
      xk = poly_dot(ctx, xk, xk_shape, 2, wt, wt_shape, 2, dot_shape, &dot_ndim);

      /* Add bias: reshape bias to (1, out_dim), expand to (batch, out_dim) */
      int64_t b_1d[] = { 1, out_dim };
      PolyUOp *b_2d = poly_reshape(ctx, bias, b_1d, 2);
      int64_t b_exp[] = { batch_size, out_dim };
      b_2d = poly_expand(ctx, b_2d, b_exp, 2);
      xk = poly_alu2(ctx, POLY_OP_ADD, xk, b_2d);

      /* Activation (skip on last layer) */
      if (l < n_linear - 1) {
        if (activation == NAM_ACT_EXU) {
          /* ExU: ReLU(exp(exu_w) * (x - exu_b)) */
          PolyUOp *exu_w = params.items[pi++].buf;
          PolyUOp *exu_b = params.items[pi++].buf;

          /* Reshape exu params to (1, out_dim) and expand */
          int64_t eu_1d[] = { 1, out_dim };
          int64_t eu_exp[] = { batch_size, out_dim };

          PolyUOp *ew = poly_reshape(ctx, exu_w, eu_1d, 2);
          ew = poly_expand(ctx, ew, eu_exp, 2);
          PolyUOp *eb = poly_reshape(ctx, exu_b, eu_1d, 2);
          eb = poly_expand(ctx, eb, eu_exp, 2);

          /* xk = ReLU(exp(ew) * (xk - eb)) */
          xk = poly_alu2(ctx, POLY_OP_ADD, xk, poly_alu1(ctx, POLY_OP_NEG, eb));
          PolyUOp *exp_w = poly_exp(ctx, ew);
          xk = poly_alu2(ctx, POLY_OP_MUL, exp_w, xk);
          xk = poly_relu(ctx, xk);
        } else if (activation == NAM_ACT_RELU) {
          xk = poly_relu(ctx, xk);
        } else if (activation == NAM_ACT_GELU) {
          xk = poly_gelu(ctx, xk);
        } else if (activation == NAM_ACT_SILU) {
          xk = poly_silu(ctx, xk);
        }
      }
    }

    /* xk is now (batch_size, n_outputs). Add to accumulator. */
    accum = poly_alu2(ctx, POLY_OP_ADD, accum, xk);
  }

  /* Store forward result */
  int64_t out_shape[] = { batch_size, n_outputs };
  PolyUOp *fwd_result = poly_reshape(ctx, accum, out_shape, 2);
  PolyUOp *fwd_store = poly_store_val(ctx, out_buf, fwd_result);
  PolyUOp *fwd_sink = poly_sink1(ctx, fwd_store);

  /* Build entrypoints */
  PolyIrEntrypoint eps[2];
  int n_eps = 0;
  eps[n_eps++] = (PolyIrEntrypoint){ "forward", fwd_sink };

  /* ── Loss graph ───────────────────────────────────────────────────── */

  int has_loss = (loss_type && (strcmp(loss_type, "mse") == 0 ||
                                strcmp(loss_type, "cross_entropy") == 0));
  if (has_loss) {
    /* Target buffer */
    PolyUOp *y_buf = poly_buffer_f32(ctx, out_numel);
    bufs[n_bufs++] = (PolyIrBufEntry){
      "y", POLY_IR_ROLE_TARGET, y_buf, { batch_size, n_outputs }, 2
    };

    /* Loss output buffer */
    PolyUOp *loss_buf = poly_buffer_f32(ctx, 1);
    bufs[n_bufs++] = (PolyIrBufEntry){
      "loss", POLY_IR_ROLE_OUTPUT, loss_buf, { 1 }, 1
    };

    PolyUOp *loss_val;
    if (strcmp(loss_type, "mse") == 0) {
      PolyUOp *diff = poly_alu2(ctx, POLY_OP_ADD, fwd_result,
                                  poly_alu1(ctx, POLY_OP_NEG, y_buf));
      PolyUOp *sq = poly_alu2(ctx, POLY_OP_MUL, diff, diff);
      int64_t axes_r0[] = { 0 };
      PolyUOp *sum0 = poly_reduce_axis(ctx, POLY_OP_ADD, sq, axes_r0, 1);
      int64_t axes_r1[] = { 0 };
      PolyUOp *sum1 = poly_reduce_axis(ctx, POLY_OP_ADD, sum0, axes_r1, 1);
      double mse_scale = 1.0 / ((double)batch_size * n_outputs);
      loss_val = poly_alu2(ctx, POLY_OP_MUL, sum1,
                           poly_const_float(ctx, mse_scale));
    } else {
      /* Cross-entropy */
      int64_t logits_shape[] = { batch_size, n_outputs };
      PolyUOp *log_probs = poly_log_softmax(ctx, fwd_result, logits_shape, 2, 1);
      PolyUOp *prod = poly_alu2(ctx, POLY_OP_MUL, y_buf, log_probs);
      int64_t axes_class[] = { 1 };
      PolyUOp *sum_class = poly_reduce_axis(ctx, POLY_OP_ADD, prod, axes_class, 1);
      int64_t axes_batch[] = { 0 };
      PolyUOp *sum_batch = poly_reduce_axis(ctx, POLY_OP_ADD, sum_class, axes_batch, 1);
      double ce_scale = -1.0 / (double)batch_size;
      loss_val = poly_alu2(ctx, POLY_OP_MUL, sum_batch,
                           poly_const_float(ctx, ce_scale));
    }

    PolyUOp *loss_store = poly_store_val(ctx, loss_buf, loss_val);
    PolyUOp *loss_sink = poly_sink1(ctx, loss_store);
    eps[n_eps++] = (PolyIrEntrypoint){ "loss", loss_sink };
  }

  /* ── Export IR ─────────────────────────────────────────────────────── */

  PolyIrSpec spec = { ctx, bufs, n_bufs, eps, n_eps };
  int ir_len = 0;
  uint8_t *ir_data = poly_ir_export(&spec, &ir_len);

  /* Encode weights as safetensors */
  PolySafetensorEntry *st_entries = malloc(params.count * sizeof(PolySafetensorEntry));
  int n_st = 0;
  for (int i = 0; i < params.count; i++) {
    NamParam *p = &params.items[i];
    st_entries[n_st].name = p->name;
    st_entries[n_st].data = p->data;
    st_entries[n_st].shape = p->shape;
    st_entries[n_st].ndim = p->ndim;
    n_st++;
  }
  int st_len = 0;
  uint8_t *st_data = poly_safetensors_encode(st_entries, n_st, NULL, &st_len);
  free(st_entries);

  /* Create instance */
  PolyInstance *inst = NULL;
  if (ir_data && st_data) {
    inst = poly_instance_from_ir(ir_data, ir_len, st_data, st_len);
  }

  /* Cleanup */
  nam_param_free(&params);
  free(bufs);
  free(subnet_sizes);
  free(ir_data);
  free(st_data);
  poly_ctx_destroy(ctx);
  cJSON_Delete(root);

  return inst;
}
