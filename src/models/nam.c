/*
 * model_nam.c -- NAM (Neural Additive Model) builder for PolyInstance
 *
 * Builds a tensor-level UOp graph from a JSON spec using staged
 * PolyInstance bindings and entrypoints.
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
#include "nam.h"
#include "mlp.h" /* poly_init_param_kaiming */
#include "../instance.h"
#include "../tensor.h"
#include "../../vendor/cjson/cJSON.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/* Activation types */

typedef enum { NAM_ACT_RELU, NAM_ACT_GELU, NAM_ACT_SILU, NAM_ACT_EXU } NamActivation;

static NamActivation nam_parse_activation(const char *s) {
  if (!s || strcmp(s, "exu") == 0) return NAM_ACT_EXU;
  if (strcmp(s, "relu") == 0) return NAM_ACT_RELU;
  if (strcmp(s, "gelu") == 0) return NAM_ACT_GELU;
  if (strcmp(s, "silu") == 0) return NAM_ACT_SILU;
  return NAM_ACT_EXU;
}

/* NAM Builder */

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
  int hidden_sizes[16] = {64};
  if (hs_item && cJSON_IsArray(hs_item)) {
    n_hidden = cJSON_GetArraySize(hs_item);
    if (n_hidden > 16) n_hidden = 16;
    for (int i = 0; i < n_hidden; i++) {
      hidden_sizes[i] = cJSON_GetArrayItem(hs_item, i)->valueint;
    }
  }

  NamActivation activation = nam_parse_activation(act_item ? act_item->valuestring : NULL);
  int n_outputs = no_item ? no_item->valueint : 1;
  if (n_outputs < 1) n_outputs = 1;
  const char *loss_type = loss_item ? loss_item->valuestring : "none";
  int batch_size = batch_item ? batch_item->valueint : 1;
  uint64_t seed = seed_item ? (uint64_t)seed_item->valuedouble : 42;

  if (batch_size < 1) batch_size = 1;

  /* Build layer sizes for each feature subnet: [1, h1, h2, ..., n_outputs] */
  int n_layers = n_hidden + 2; /* input(1) + hidden... + output */
  int *subnet_sizes = malloc((size_t)n_layers * sizeof(int));
  if (!subnet_sizes) {
    cJSON_Delete(root);
    return NULL;
  }
  subnet_sizes[0] = 1;
  for (int i = 0; i < n_hidden; i++)
    subnet_sizes[i + 1] = hidden_sizes[i];
  subnet_sizes[n_layers - 1] = n_outputs;
  int n_linear = n_layers - 1; /* number of linear transformations */

  PolyCtx *ctx = poly_ctx_new();
  if (!ctx) goto fail_no_instance;
  PolyInstanceOptions opts = {
      .own_ctx_on_success = true,
      .own_ctx_on_failure = true,
  };
  PolyInstance *inst = poly_instance_new(ctx, &opts);
  if (!inst) {
    poly_ctx_destroy(ctx);
    goto fail_no_instance;
  }

  int64_t x_shape[] = {batch_size, n_features};
  PolyTensor *x_tensor = poly_instance_input(inst, "x", POLY_FLOAT32, x_shape, 2);
  if (!x_tensor) goto fail_pre_build;

  /* Register intercept param: (n_outputs,) */
  int64_t intercept_shape_1d[] = {n_outputs};
  PolyTensor *intercept_tensor =
      poly_instance_param(inst, "intercept", POLY_FLOAT32, intercept_shape_1d, 1);
  if (!intercept_tensor) goto fail_pre_build;

  /* Start with intercept broadcast to (batch_size, n_outputs). */
  int64_t intercept_shape[] = {1, n_outputs};
  PolyUOp *accum = poly_reshape(ctx, poly_tensor_uop(intercept_tensor), intercept_shape, 2);
  int64_t accum_expanded[] = {batch_size, n_outputs};
  accum = poly_expand(ctx, accum, accum_expanded, 2);
  if (!accum) goto fail_pre_build;

  PolyUOp *x_2d = poly_tensor_uop(x_tensor);

  /* Process each feature subnet. */
  for (int k = 0; k < n_features; k++) {
    int64_t shrink_pairs[][2] = {{0, batch_size}, {k, k + 1}};
    PolyUOp *xk = poly_shrink(ctx, x_2d, shrink_pairs, 2);
    if (!xk) goto fail_pre_build;

    for (int l = 0; l < n_linear; l++) {
      int in_dim = subnet_sizes[l];
      int out_dim = subnet_sizes[l + 1];

      if (poly_instance_scope_push(inst, "features.%d.layers.%d", k, l) != POLY_STATUS_OK)
        goto fail_pre_build;

      int64_t w_shape[] = {out_dim, in_dim};
      PolyTensor *w_tensor = poly_instance_param(inst, "weight", POLY_FLOAT32, w_shape, 2);
      if (!w_tensor) goto fail_pre_build;
      PolyUOp *w = poly_tensor_uop(w_tensor);

      int64_t b_shape[] = {out_dim};
      PolyTensor *bias_tensor = poly_instance_param(inst, "bias", POLY_FLOAT32, b_shape, 1);
      if (!bias_tensor) goto fail_pre_build;
      PolyUOp *bias = poly_tensor_uop(bias_tensor);

      if (poly_instance_scope_pop(inst) != POLY_STATUS_OK) goto fail_pre_build;

      int64_t perm[] = {1, 0};
      PolyUOp *wt = poly_permute(ctx, w, perm, 2);
      xk = poly_dot(ctx, xk, wt);

      int64_t b_1d[] = {1, out_dim};
      PolyUOp *b_2d = poly_reshape(ctx, bias, b_1d, 2);
      int64_t b_exp[] = {batch_size, out_dim};
      b_2d = poly_expand(ctx, b_2d, b_exp, 2);
      xk = poly_alu2(ctx, POLY_OP_ADD, xk, b_2d);
      if (!xk) goto fail_pre_build;

      /* Activation (skip on last layer). */
      if (l < n_linear - 1) {
        if (activation == NAM_ACT_EXU) {
          int64_t eu_shape[] = {out_dim};
          if (poly_instance_scope_push(inst, "features.%d.exu.%d", k, l) != POLY_STATUS_OK)
            goto fail_pre_build;
          PolyTensor *exu_w_tensor = poly_instance_param(inst, "weight", POLY_FLOAT32, eu_shape, 1);
          if (!exu_w_tensor) goto fail_pre_build;
          PolyTensor *exu_b_tensor = poly_instance_param(inst, "bias", POLY_FLOAT32, eu_shape, 1);
          if (!exu_b_tensor) goto fail_pre_build;
          if (poly_instance_scope_pop(inst) != POLY_STATUS_OK) goto fail_pre_build;

          int64_t eu_1d[] = {1, out_dim};
          int64_t eu_exp[] = {batch_size, out_dim};

          PolyUOp *ew = poly_reshape(ctx, poly_tensor_uop(exu_w_tensor), eu_1d, 2);
          ew = poly_expand(ctx, ew, eu_exp, 2);
          PolyUOp *eb = poly_reshape(ctx, poly_tensor_uop(exu_b_tensor), eu_1d, 2);
          eb = poly_expand(ctx, eb, eu_exp, 2);

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
        if (!xk) goto fail_pre_build;
      }
    }

    accum = poly_alu2(ctx, POLY_OP_ADD, accum, xk);
    if (!accum) goto fail_pre_build;
  }

  int64_t out_shape[] = {batch_size, n_outputs};
  PolyUOp *fwd_result = poly_reshape(ctx, accum, out_shape, 2);
  PolyTensor *out_tensor = poly_tensor_create(ctx, fwd_result, POLY_TENSOR_VALUE, POLY_DEVICE_AUTO);
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
    int64_t y_shape[] = {batch_size, n_outputs};
    PolyTensor *y_tensor = poly_instance_target(inst, "y", POLY_FLOAT32, y_shape, 2);
    if (!y_tensor) goto fail_pre_build;
    PolyUOp *y = poly_tensor_uop(y_tensor);
    PolyUOp *loss_val;
    if (strcmp(loss_type, "mse") == 0) {
      PolyUOp *diff = poly_alu2(ctx, POLY_OP_ADD, fwd_result, poly_alu1(ctx, POLY_OP_NEG, y));
      PolyUOp *sq = poly_alu2(ctx, POLY_OP_MUL, diff, diff);
      int64_t axes_r0[] = {0};
      PolyUOp *sum0 = poly_reduce_axis(ctx, POLY_OP_ADD, sq, axes_r0, 1);
      int64_t axes_r1[] = {0};
      PolyUOp *sum1 = poly_reduce_axis(ctx, POLY_OP_ADD, sum0, axes_r1, 1);
      double mse_scale = 1.0 / ((double)batch_size * n_outputs);
      loss_val = poly_alu2(ctx, POLY_OP_MUL, sum1, poly_const_float(ctx, mse_scale));
    } else {
      PolyUOp *log_probs = poly_log_softmax(ctx, fwd_result, 1);
      PolyUOp *prod = poly_alu2(ctx, POLY_OP_MUL, y, log_probs);
      int64_t axes_class[] = {1};
      PolyUOp *sum_class = poly_reduce_axis(ctx, POLY_OP_ADD, prod, axes_class, 1);
      int64_t axes_batch[] = {0};
      PolyUOp *sum_batch = poly_reduce_axis(ctx, POLY_OP_ADD, sum_class, axes_batch, 1);
      double ce_scale = -1.0 / (double)batch_size;
      loss_val = poly_alu2(ctx, POLY_OP_MUL, sum_batch, poly_const_float(ctx, ce_scale));
    }
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
    if (err.message[0]) fprintf(stderr, "poly_nam_instance: build failed: %s\n", err.message);
    poly_instance_free(inst);
    free(subnet_sizes);
    cJSON_Delete(root);
    return NULL;
  }

  /* Initialize weights (Kaiming) directly in instance buffers. */
  for (int k = 0; k < n_features; k++) {
    for (int l = 0; l < n_linear; l++) {
      int in_dim = subnet_sizes[l];
      int out_dim = subnet_sizes[l + 1];
      int64_t w_numel = (int64_t)out_dim * in_dim;
      char name[128];
      snprintf(name, sizeof(name), "features.%d.layers.%d.weight", k, l);
      float *w_data = poly_instance_buf_data_named(inst, name, NULL);
      if (w_data) poly_init_param_kaiming(seed, name, w_data, w_numel, (int64_t)in_dim);
    }
  }

  free(subnet_sizes);
  cJSON_Delete(root);
  return inst;

fail_pre_build:
  poly_instance_free(inst);
  poly_ctx_destroy(ctx);
fail_no_instance:
  free(subnet_sizes);
  cJSON_Delete(root);
  return NULL;
}
