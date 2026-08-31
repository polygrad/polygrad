/*
 * model_tabm.c -- TabM (BatchEnsemble MLP) builder for PolyInstance
 *
 * Builds a tensor-level UOp graph from a JSON spec using staged
 * PolyInstance bindings and entrypoints.
 *
 * TabM forward pass per layer:
 *   For each ensemble member i: l_i(x) = s_i * (W @ (r_i * x)) + b_i
 * After final layer: mean over k members.
 */

#define _POSIX_C_SOURCE 200809L
#include "tabm.h"
#include "mlp.h" /* poly_init_param_kaiming */
#include "../instance.h"
#include "../tensor.h"
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

static PolyTensor *tabm_apply_activation(PolyCtx *ctx, PolyTensor *x, TabmActivation act) {
  switch (act) {
  case TABM_ACT_RELU:
    return poly_tensor_relu(ctx, x);
  case TABM_ACT_GELU:
    return poly_tensor_gelu(ctx, x);
  case TABM_ACT_SILU:
    return poly_tensor_silu(ctx, x);
  case TABM_ACT_NONE:
    return x;
  }
  return x;
}

static PolyTensor *tabm_float_scalar(PolyCtx *ctx, double value) {
  PolyUOp *constant = poly_const_typed(ctx, POLY_FLOAT32, value);
  if (!constant) return NULL;
  PolyTensor *out =
      poly_tensor_create_with_roots(ctx, constant, constant, POLY_TENSOR_VALUE, POLY_DEVICE_AUTO);
  if (out) {
    poly_tensor_set_requires_grad(out, false);
    poly_tensor_set_provenance(out, POLY_TENSOR_PROVENANCE_CONST_INIT);
  }
  return out;
}

/* TabM Builder */

PolyInstance *poly_tabm_instance(const char *spec_json, int spec_len, PolyDevice device) {
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

  int *layer_sizes = malloc((size_t)n_layers * sizeof(int));
  if (!layer_sizes) {
    cJSON_Delete(root);
    return NULL;
  }
  for (int i = 0; i < n_layers; i++) {
    cJSON *item = cJSON_GetArrayItem(layers_arr, i);
    layer_sizes[i] = item ? item->valueint : 0;
  }

  TabmActivation activation = tabm_parse_activation(act_item ? act_item->valuestring : "relu");
  const char *loss_type = loss_item ? loss_item->valuestring : "none";
  int batch_size = batch_item ? batch_item->valueint : 1;
  uint64_t seed = seed_item ? (uint64_t)seed_item->valuedouble : 42;
  int k = ensemble_item ? ensemble_item->valueint : 32;

  if (batch_size < 1) batch_size = 1;
  if (k < 1) k = 1;

  PolyCtx *ctx = poly_ctx_new();
  if (!ctx) goto fail_no_ctx;
  if (device != POLY_DEVICE_AUTO) poly_ctx_set_preferred_device(ctx, device);
  PolyInstanceOptions opts = {
      .own_ctx_on_success = true,
      .own_ctx_on_failure = true,
  };
  PolyInstance *inst = poly_instance_new(ctx, &opts);
  if (!inst) {
    poly_ctx_destroy(ctx);
    goto fail_no_ctx;
  }

  PolyTensor **param_bufs = NULL;

  int n_linear = n_layers - 1;
  int in_dim = layer_sizes[0];
  int out_dim = layer_sizes[n_layers - 1];

  int64_t x_io_shape[] = {batch_size, in_dim};
  PolyTensor *x_tensor = poly_instance_input(inst, "x", POLY_FLOAT32, x_io_shape, 2);
  if (!x_tensor) goto fail_pre_build;

  param_bufs = calloc((size_t)n_linear * 4, sizeof(PolyTensor *));
  if (!param_bufs) goto fail_pre_build;
  int pi = 0;
  for (int l = 0; l < n_linear; l++) {
    int l_in = layer_sizes[l];
    int l_out = layer_sizes[l + 1];

    if (poly_instance_scope_push(inst, "layers.%d", l) != POLY_STATUS_OK) goto fail_pre_build;

    int64_t ws[] = {l_out, l_in};
    PolyTensor *w_tensor = poly_instance_param(inst, "weight", POLY_FLOAT32, ws, 2);
    if (!w_tensor) goto fail_pre_build;
    param_bufs[pi++] = w_tensor;

    int64_t rs[] = {k, l_in};
    PolyTensor *r_tensor = poly_instance_param(inst, "r", POLY_FLOAT32, rs, 2);
    if (!r_tensor) goto fail_pre_build;
    param_bufs[pi++] = r_tensor;

    int64_t ss[] = {k, l_out};
    PolyTensor *s_tensor = poly_instance_param(inst, "s", POLY_FLOAT32, ss, 2);
    if (!s_tensor) goto fail_pre_build;
    param_bufs[pi++] = s_tensor;

    int64_t bs[] = {k, l_out};
    PolyTensor *b_tensor = poly_instance_param(inst, "b", POLY_FLOAT32, bs, 2);
    if (!b_tensor) goto fail_pre_build;
    param_bufs[pi++] = b_tensor;

    if (poly_instance_scope_pop(inst) != POLY_STATUS_OK) goto fail_pre_build;
  }

  /* Build forward graph. */
  int64_t x_2d[] = {batch_size, in_dim};
  PolyTensor *x = poly_tensor_reshape(ctx, x_tensor, x_2d, 2);

  /* Reshape to (1, in_dim) then expand to (k, in_dim).
   * NOTE: batch_size > 1 not supported for TabM yet, preserving previous behavior. */
  int64_t x_1d[] = {1, in_dim};
  x = poly_tensor_reshape(ctx, x, x_1d, 2);
  int64_t x_expanded[] = {k, in_dim};
  x = poly_tensor_expand(ctx, x, x_expanded, 2);
  if (!x) goto fail_pre_build;

  int param_idx = 0;
  for (int l = 0; l < n_linear; l++) {
    int l_in = layer_sizes[l];
    int l_out = layer_sizes[l + 1];

    PolyTensor *w = param_bufs[param_idx++];
    PolyTensor *r = param_bufs[param_idx++];
    PolyTensor *s = param_bufs[param_idx++];
    PolyTensor *b = param_bufs[param_idx++];

    int64_t r_shape[] = {k, l_in};
    PolyTensor *r_2d = poly_tensor_reshape(ctx, r, r_shape, 2);
    x = poly_tensor_alu2(ctx, POLY_OP_MUL, r_2d, x);

    int64_t w_2d[] = {l_out, l_in};
    PolyTensor *w_reshaped = poly_tensor_reshape(ctx, w, w_2d, 2);
    int64_t perm[] = {1, 0};
    PolyTensor *wt = poly_tensor_permute(ctx, w_reshaped, perm, 2);
    x = poly_tensor_dot(ctx, x, wt);

    int64_t s_shape[] = {k, l_out};
    PolyTensor *s_2d = poly_tensor_reshape(ctx, s, s_shape, 2);
    x = poly_tensor_alu2(ctx, POLY_OP_MUL, s_2d, x);

    int64_t b_shape[] = {k, l_out};
    PolyTensor *b_2d = poly_tensor_reshape(ctx, b, b_shape, 2);
    x = poly_tensor_alu2(ctx, POLY_OP_ADD, x, b_2d);

    if (l < n_linear - 1) x = tabm_apply_activation(ctx, x, activation);
    if (!x) goto fail_pre_build;
  }

  int64_t axes0[] = {0};
  PolyTensor *sum_k = poly_tensor_sum(ctx, x, axes0, 1, false);

  double scale_k = 1.0 / (double)k;
  PolyTensor *scale = tabm_float_scalar(ctx, scale_k);
  PolyTensor *mean_k = scale ? poly_tensor_alu2(ctx, POLY_OP_MUL, sum_k, scale) : NULL;

  int64_t out_shape[] = {batch_size, out_dim};
  PolyTensor *out_tensor = mean_k ? poly_tensor_reshape(ctx, mean_k, out_shape, 2) : NULL;
  if (!out_tensor || poly_instance_output(inst, "output", out_tensor) != POLY_STATUS_OK)
    goto fail_pre_build;
  const char *forward_inputs[] = {"x"};
  const char *forward_outputs[] = {"output"};
  if (poly_instance_entrypoint(inst, "forward", forward_inputs, 1, forward_outputs, 1, NULL) !=
      POLY_STATUS_OK)
    goto fail_pre_build;

  int has_loss =
      (loss_type && (strcmp(loss_type, "mse") == 0 || strcmp(loss_type, "cross_entropy") == 0));
  if (has_loss) {
    int64_t y_shape[] = {batch_size, out_dim};
    PolyTensor *y_tensor = poly_instance_target(inst, "y", POLY_FLOAT32, y_shape, 2);
    if (!y_tensor) goto fail_pre_build;
    PolyTensor *loss_tensor;
    if (strcmp(loss_type, "mse") == 0) {
      PolyTensor *diff = poly_tensor_alu2(ctx, POLY_OP_SUB, out_tensor, y_tensor);
      PolyTensor *sq = diff ? poly_tensor_alu2(ctx, POLY_OP_MUL, diff, diff) : NULL;
      int64_t axes_r0[] = {1};
      PolyTensor *sum0 = sq ? poly_tensor_sum(ctx, sq, axes_r0, 1, false) : NULL;
      int64_t axes_r1[] = {0};
      PolyTensor *sum1 = sum0 ? poly_tensor_sum(ctx, sum0, axes_r1, 1, false) : NULL;
      double mse_scale = 1.0 / ((double)batch_size * out_dim);
      PolyTensor *loss_scale = tabm_float_scalar(ctx, mse_scale);
      loss_tensor =
          sum1 && loss_scale ? poly_tensor_alu2(ctx, POLY_OP_MUL, sum1, loss_scale) : NULL;
    } else {
      PolyTensor *log_probs = poly_tensor_log_softmax(ctx, out_tensor, 1);
      PolyTensor *prod = log_probs ? poly_tensor_alu2(ctx, POLY_OP_MUL, y_tensor, log_probs) : NULL;
      int64_t axes_class[] = {1};
      PolyTensor *sum_class = prod ? poly_tensor_sum(ctx, prod, axes_class, 1, false) : NULL;
      int64_t axes_batch[] = {0};
      PolyTensor *sum_batch =
          sum_class ? poly_tensor_sum(ctx, sum_class, axes_batch, 1, false) : NULL;
      double ce_scale = -1.0 / (double)batch_size;
      PolyTensor *loss_scale = tabm_float_scalar(ctx, ce_scale);
      loss_tensor = sum_batch && loss_scale
                        ? poly_tensor_alu2(ctx, POLY_OP_MUL, sum_batch, loss_scale)
                        : NULL;
    }
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
    if (err.message[0]) fprintf(stderr, "poly_tabm_instance: build failed: %s\n", err.message);
    poly_instance_free(inst);
    free(param_bufs);
    free(layer_sizes);
    cJSON_Delete(root);
    return NULL;
  }

  /* Initialize weights directly in instance buffers. */
  for (int l = 0; l < n_linear; l++) {
    int l_in = layer_sizes[l];
    int l_out = layer_sizes[l + 1];
    char name[128];

    snprintf(name, sizeof(name), "layers.%d.weight", l);
    float *w_data = poly_instance_buf_data_named(inst, name, NULL);
    if (w_data) {
      int64_t w_numel = (int64_t)l_out * l_in;
      poly_init_param_kaiming(seed, name, w_data, w_numel, (int64_t)l_in);
    }

    snprintf(name, sizeof(name), "layers.%d.r", l);
    int64_t r_numel = 0;
    float *r_data = poly_instance_buf_data_named(inst, name, &r_numel);
    if (r_data) {
      for (int64_t i = 0; i < r_numel; i++)
        r_data[i] = 1.0f;
    }

    snprintf(name, sizeof(name), "layers.%d.s", l);
    int64_t s_numel = 0;
    float *s_data = poly_instance_buf_data_named(inst, name, &s_numel);
    if (s_data) {
      for (int64_t i = 0; i < s_numel; i++)
        s_data[i] = 1.0f;
    }
  }

  free(param_bufs);
  free(layer_sizes);
  cJSON_Delete(root);
  return inst;

fail_pre_build:
  free(param_bufs);
  poly_instance_free(inst);
  poly_ctx_destroy(ctx);
fail_no_ctx:
  free(layer_sizes);
  cJSON_Delete(root);
  return NULL;
}
