/*
 * model_tabm.c -- TabM (BatchEnsemble MLP) builder for PolyInstance
 *
 * Builds a tensor-level UOp graph from a JSON spec using the named
 * buffer registry (poly_param/poly_input/poly_output/poly_target),
 * then creates a PolyInstance via poly_instance_from_ctx.
 *
 * TabM forward pass per layer:
 *   For each ensemble member i: l_i(x) = s_i * (W @ (r_i * x)) + b_i
 * After final layer: mean over k members.
 */

#define _POSIX_C_SOURCE 200809L
#include "tabm.h"
#include "mlp.h"  /* poly_init_param_kaiming */
#include "../instance.h"
#include "../frontend.h"
#include "../../vendor/cjson/cJSON.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/* Activation dispatch (same as model_mlp.c) */

typedef enum { TABM_ACT_NONE, TABM_ACT_RELU, TABM_ACT_GELU, TABM_ACT_SILU } TabmActivation;

static TabmActivation tabm_parse_activation(const char *s) {
  if (!s || strcmp(s, "none") == 0) return TABM_ACT_NONE;
  if (strcmp(s, "relu") == 0) return TABM_ACT_RELU;
  if (strcmp(s, "gelu") == 0) return TABM_ACT_GELU;
  if (strcmp(s, "silu") == 0) return TABM_ACT_SILU;
  return TABM_ACT_RELU;
}

static PolyUOp *tabm_apply_activation(PolyCtx *ctx, PolyUOp *x, TabmActivation act) {
  switch (act) {
  case TABM_ACT_RELU: return poly_relu(ctx, x);
  case TABM_ACT_GELU: return poly_gelu(ctx, x);
  case TABM_ACT_SILU: return poly_silu(ctx, x);
  case TABM_ACT_NONE: return x;
  }
  return x;
}

/* TabM Builder */

PolyInstance *poly_tabm_instance(const char *spec_json, int spec_len) {
  if (!spec_json || spec_len <= 0) return NULL;

  /* Parse JSON */
  cJSON *root = cJSON_ParseWithLength(spec_json, (size_t)spec_len);
  if (!root) {
    fprintf(stderr, "poly_tabm_instance: JSON parse error\n");
    return NULL;
  }

  /* Extract fields */
  cJSON *layers_arr = cJSON_GetObjectItem(root, "layers");
  cJSON *act_item = cJSON_GetObjectItem(root, "activation");
  cJSON *loss_item = cJSON_GetObjectItem(root, "loss");
  cJSON *batch_item = cJSON_GetObjectItem(root, "batch_size");
  cJSON *seed_item = cJSON_GetObjectItem(root, "seed");
  cJSON *ensemble_item = cJSON_GetObjectItem(root, "n_ensemble");

  if (!layers_arr || !cJSON_IsArray(layers_arr)) {
    fprintf(stderr, "poly_tabm_instance: 'layers' must be an array\n");
    cJSON_Delete(root);
    return NULL;
  }

  int n_layers = cJSON_GetArraySize(layers_arr);
  if (n_layers < 2) {
    fprintf(stderr, "poly_tabm_instance: need at least 2 layers\n");
    cJSON_Delete(root);
    return NULL;
  }

  int *layer_sizes = malloc(n_layers * sizeof(int));
  for (int i = 0; i < n_layers; i++) {
    cJSON *item = cJSON_GetArrayItem(layers_arr, i);
    layer_sizes[i] = item ? item->valueint : 0;
  }

  TabmActivation activation = tabm_parse_activation(
      act_item ? act_item->valuestring : "relu");
  const char *loss_type = loss_item ? loss_item->valuestring : "none";
  int batch_size = batch_item ? batch_item->valueint : 1;
  uint64_t seed = seed_item ? (uint64_t)seed_item->valuedouble : 42;
  int k = ensemble_item ? ensemble_item->valueint : 32;

  if (batch_size < 1) batch_size = 1;
  if (k < 1) k = 1;

  /* Build graph using named buffer registry */
  PolyCtx *ctx = poly_ctx_new();
  int n_linear = n_layers - 1;
  int in_dim = layer_sizes[0];
  int out_dim = layer_sizes[n_layers - 1];

  /* Register I/O buffers */
  int64_t x_io_shape[] = { batch_size, in_dim };
  PolyUOp *x_buf = poly_input(ctx, POLY_FLOAT32, x_io_shape, 2, "x");
  int64_t out_io_shape[] = { batch_size, out_dim };
  PolyUOp *out_buf = poly_output(ctx, POLY_FLOAT32, out_io_shape, 2, "output");

  /* Register param buffers for each layer: weight, r, s, b */
  PolyUOp **param_bufs = calloc(n_linear * 4, sizeof(PolyUOp *));
  int pi = 0;
  for (int l = 0; l < n_linear; l++) {
    int l_in = layer_sizes[l];
    int l_out = layer_sizes[l + 1];

    int64_t ws[] = { l_out, l_in };
    param_bufs[pi++] = poly_param(ctx, POLY_FLOAT32, ws, 2, "layers.%d.weight", l);

    int64_t rs[] = { k, l_in };
    param_bufs[pi++] = poly_param(ctx, POLY_FLOAT32, rs, 2, "layers.%d.r", l);

    int64_t ss[] = { k, l_out };
    param_bufs[pi++] = poly_param(ctx, POLY_FLOAT32, ss, 2, "layers.%d.s", l);

    int64_t bs[] = { k, l_out };
    param_bufs[pi++] = poly_param(ctx, POLY_FLOAT32, bs, 2, "layers.%d.b", l);
  }

  /* Build forward graph */

  /* Expand input from (batch_size, in_dim) to (k, in_dim)
   * For batch_size=1: reshape (1, in_dim) -> (1, in_dim), expand to (k, in_dim)
   * NOTE: batch_size > 1 not supported for TabM yet */
  int64_t x_2d[] = { batch_size, in_dim };
  PolyUOp *x = poly_reshape(ctx, x_buf, x_2d, 2);

  /* Reshape to (1, in_dim) then expand to (k, in_dim) */
  int64_t x_1d[] = { 1, in_dim };
  x = poly_reshape(ctx, x, x_1d, 2);
  int64_t x_expanded[] = { k, in_dim };
  x = poly_expand(ctx, x, x_expanded, 2);

  int param_idx = 0;
  for (int l = 0; l < n_linear; l++) {
    int l_in = layer_sizes[l];
    int l_out = layer_sizes[l + 1];

    /* Get params: weight, r, s, b */
    PolyUOp *w = param_bufs[param_idx++];
    PolyUOp *r = param_bufs[param_idx++];
    PolyUOp *s = param_bufs[param_idx++];
    PolyUOp *b = param_bufs[param_idx++];

    /* Reshape r to (k, l_in) */
    int64_t r_shape[] = { k, l_in };
    PolyUOp *r_2d = poly_reshape(ctx, r, r_shape, 2);

    /* Step 1: Input scaling: r * x -> (k, l_in) */
    x = poly_alu2(ctx, POLY_OP_MUL, r_2d, x);

    /* Step 2: Matmul with shared weight W */
    /* W: (l_out, l_in) -> transpose to (l_in, l_out) */
    int64_t w_2d[] = { l_out, l_in };
    PolyUOp *w_reshaped = poly_reshape(ctx, w, w_2d, 2);
    int64_t perm[] = { 1, 0 };
    PolyUOp *wt = poly_permute(ctx, w_reshaped, perm, 2);

    /* (k, l_in) @ (l_in, l_out) -> (k, l_out) */
    x = poly_dot(ctx, x, wt);

    /* Step 3: Output scaling: s * x -> (k, l_out) */
    int64_t s_shape[] = { k, l_out };
    PolyUOp *s_2d = poly_reshape(ctx, s, s_shape, 2);
    x = poly_alu2(ctx, POLY_OP_MUL, s_2d, x);

    /* Step 4: Per-member bias: x + b -> (k, l_out) */
    int64_t b_shape[] = { k, l_out };
    PolyUOp *b_2d = poly_reshape(ctx, b, b_shape, 2);
    x = poly_alu2(ctx, POLY_OP_ADD, x, b_2d);

    /* Step 5: Activation (skip on last layer) */
    if (l < n_linear - 1) {
      x = tabm_apply_activation(ctx, x, activation);
    }
  }

  /* Mean over ensemble: reduce_axis(ADD, axis=0) then scale by 1/k */
  int64_t axes0[] = { 0 };
  PolyUOp *sum_k = poly_reduce_axis(ctx, POLY_OP_ADD, x, axes0, 1);

  double scale_k = 1.0 / (double)k;
  PolyUOp *mean_k = poly_alu2(ctx, POLY_OP_MUL, sum_k,
                                poly_const_float(ctx, scale_k));

  /* Reshape mean to (batch_size, out_dim) for output */
  int64_t out_shape[] = { batch_size, out_dim };
  PolyUOp *fwd_result = poly_reshape(ctx, mean_k, out_shape, 2);

  /* Store forward result */
  PolyUOp *fwd_store = poly_store_val(ctx, out_buf, fwd_result);
  PolyUOp *fwd_sink = poly_sink1(ctx, fwd_store);
  poly_register_entrypoint(ctx, "forward", fwd_sink);

  /* Loss graph (same as MLP but operates on mean prediction) */
  int has_loss = (loss_type && (strcmp(loss_type, "mse") == 0 ||
                                strcmp(loss_type, "cross_entropy") == 0));
  if (has_loss) {
    int64_t y_shape[] = { batch_size, out_dim };
    PolyUOp *y_buf = poly_target(ctx, POLY_FLOAT32, y_shape, 2, "y");
    PolyUOp *loss_buf = poly_output(ctx, POLY_FLOAT32, (int64_t[]){1}, 1, "loss");

    PolyUOp *y = poly_reshape(ctx, y_buf, y_shape, 2);
    PolyUOp *loss_val;
    if (strcmp(loss_type, "mse") == 0) {
      /* MSE = mean((mean_pred - y)^2) */
      PolyUOp *diff = poly_alu2(ctx, POLY_OP_ADD, fwd_result,
                                  poly_alu1(ctx, POLY_OP_NEG, y));
      PolyUOp *sq = poly_alu2(ctx, POLY_OP_MUL, diff, diff);
      int64_t axes_r0[] = { 1 };
      PolyUOp *sum0 = poly_reduce_axis(ctx, POLY_OP_ADD, sq, axes_r0, 1);
      int64_t axes_r1[] = { 0 };
      PolyUOp *sum1 = poly_reduce_axis(ctx, POLY_OP_ADD, sum0, axes_r1, 1);
      double mse_scale = 1.0 / ((double)batch_size * out_dim);
      loss_val = poly_alu2(ctx, POLY_OP_MUL, sum1,
                           poly_const_float(ctx, mse_scale));
    } else {
      /* Cross-entropy on mean prediction */
      PolyUOp *log_probs = poly_log_softmax(ctx, fwd_result, 1);
      PolyUOp *prod = poly_alu2(ctx, POLY_OP_MUL, y, log_probs);
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
    poly_register_entrypoint(ctx, "loss", loss_sink);
  }

  /* Create instance from ctx registry */
  PolyInstance *inst = poly_instance_from_ctx(ctx);

  /* Initialize weights directly in instance buffers */
  if (inst) {
    for (int l = 0; l < n_linear; l++) {
      int l_in = layer_sizes[l];
      int l_out = layer_sizes[l + 1];
      char name[128];

      /* Weight: Kaiming init */
      snprintf(name, sizeof(name), "layers.%d.weight", l);
      float *w_data = poly_instance_buf_data_named(inst, name, NULL);
      if (w_data) {
        int64_t w_numel = (int64_t)l_out * l_in;
        poly_init_param_kaiming(seed, name, w_data, w_numel, (int64_t)l_in);
      }

      /* r: init ones */
      snprintf(name, sizeof(name), "layers.%d.r", l);
      int64_t r_numel = 0;
      float *r_data = poly_instance_buf_data_named(inst, name, &r_numel);
      if (r_data) {
        for (int64_t i = 0; i < r_numel; i++) r_data[i] = 1.0f;
      }

      /* s: init ones */
      snprintf(name, sizeof(name), "layers.%d.s", l);
      int64_t s_numel = 0;
      float *s_data = poly_instance_buf_data_named(inst, name, &s_numel);
      if (s_data) {
        for (int64_t i = 0; i < s_numel; i++) s_data[i] = 1.0f;
      }

      /* b: stays zero-initialized (calloc in instance_from_ctx) */
    }
  }

  /* Transfer ctx ownership to instance */
  if (inst) poly_instance_own_ctx(inst);

  free(param_bufs);
  free(layer_sizes);
  cJSON_Delete(root);
  return inst;
}
