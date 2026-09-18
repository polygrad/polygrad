/*
 * poly_model_mlp.c -- MLP family builder for PolyModel
 *
 * Two entry points:
 * Typed C and registered JSON construction share the same topology builder.
 *
 * Deterministic weight init via SplitMix64.
 */

#define _POSIX_C_SOURCE 200809L
#include "mlp.h"
#include "factory.h"
#include "layers.h"

#include "../nn/nn.h"
#include "../tensor.h"
#include "../model.h"
#include "../../vendor/cjson/cJSON.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

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

static PolyModel *mlp_build(PolyCtx *ctx, const MLPConfig *cfg, PolyModelError *err) {
  if (!cfg || cfg->n_layers < 2 || cfg->n_layers > POLY_MLP_MAX_LAYERS) return NULL;

  int n_linear = cfg->n_layers - 1;
  int in_dim = cfg->layers[0];
  int out_dim = cfg->layers[cfg->n_layers - 1];
  int batch_size = cfg->batch_size > 0 ? cfg->batch_size : 1;
  const char *activation = cfg->activation;
  const char *loss_type = cfg->loss ? cfg->loss : "none";

  PolyModel *inst = poly_model_new(ctx, NULL);
  if (!inst) return NULL;

  /* Declare I/O buffers on the staged model. */
  int64_t x_shape[] = {batch_size, in_dim};
  PolyTensor *x_tensor = poly_model_input(inst, "x", POLY_FLOAT32, x_shape, 2);
  if (!x_tensor) goto fail_pre_build;

  /* Forward: chain of linear + activation. */
  PolyTensor *x = x_tensor;
  for (int l = 0; l < n_linear; l++) {
    char prefix[64];
    snprintf(prefix, sizeof(prefix), "layers.%d", l);
    x = poly_model_linear(inst, prefix, x, cfg->layers[l], cfg->layers[l + 1], cfg->use_bias);
    if (!x) goto fail_pre_build;
    if (l < n_linear - 1) x = model_activation(ctx, x, activation);
    if (!x) goto fail_pre_build;
  }

  if (poly_model_output(inst, "output", x) != POLY_STATUS_OK) goto fail_pre_build;
  const char *forward_inputs[] = {"x"};
  const char *forward_outputs[] = {"output"};
  if (poly_model_entrypoint(inst, "forward", forward_inputs, 1, forward_outputs, 1, NULL) !=
      POLY_STATUS_OK)
    goto fail_pre_build;

  /* Loss graph. */
  int has_loss =
      loss_type && (strcmp(loss_type, "mse") == 0 || strcmp(loss_type, "cross_entropy") == 0);
  if (has_loss) {
    int64_t y_shape[] = {batch_size, out_dim};
    PolyTensor *y_tensor = poly_model_target(inst, "y", POLY_FLOAT32, y_shape, 2);
    if (!y_tensor) goto fail_pre_build;

    PolyTensor *loss_tensor = strcmp(loss_type, "mse") == 0
                                  ? poly_tensor_mse_loss(ctx, x, y_tensor)
                                  : poly_tensor_cross_entropy(ctx, x, y_tensor, 1, 2, 0);
    if (!loss_tensor || poly_model_output(inst, "loss", loss_tensor) != POLY_STATUS_OK)
      goto fail_pre_build;
    const char *loss_inputs[] = {"x", "y"};
    const char *loss_outputs[] = {"loss"};
    PolyEntrypointOptions loss_opts = {.objective = "loss"};
    if (poly_model_entrypoint(inst, "loss", loss_inputs, 2, loss_outputs, 1, &loss_opts) !=
        POLY_STATUS_OK)
      goto fail_pre_build;
  }

  if (poly_model_build(inst, err) != POLY_STATUS_OK) {
    poly_model_free(inst);
    return NULL;
  }

  /* Initialize through the same coherent writes used by checkpoint loading. */
  for (int l = 0; l < n_linear; l++) {
    char name[128];
    snprintf(name, sizeof(name), "layers.%d.weight", l);
    if (model_init_param(inst, name, cfg->seed, cfg->layers[l], 0)) goto fail_pre_build;
  }

  return inst;

fail_pre_build:
  poly_model_free(inst);
  return NULL;
}

/* Standalone C callers own a context through the returned Model; frontends
 * use the context-taking form so Runtime disposal reaches every family. */
PolyModel *poly_mlp_into(PolyCtx *ctx, const MLPConfig *cfg, PolyDevice device) {
  PolyModelFactoryScope scope;
  if (!model_factory_begin(&scope, ctx, device)) return NULL;
  return model_factory_end(&scope, mlp_build(scope.ctx, cfg, NULL));
}

/* FFI wrapper (JSON -> config -> build) */

PolyModel *model_mlp_from_config(PolyCtx *ctx, const cJSON *root, PolyModelError *err) {
  if (!model_config_sizes(root, "layers", 2, POLY_MLP_MAX_LAYERS, true, err) ||
      !model_config_training(root, err) ||
      !model_config_choice(root, "activation", "|relu|gelu|silu|tanh|sigmoid|none|", err))
    return NULL;
  cJSON *bias = cJSON_GetObjectItemCaseSensitive(root, "bias");
  if (bias && !cJSON_IsBool(bias)) {
    model_factory_error(err, "bias", "expected boolean");
    return NULL;
  }
  cJSON *layers = cJSON_GetObjectItemCaseSensitive(root, "layers");
  MLPConfig cfg = poly_mlp_config_default();
  cJSON *v;
  if ((v = cJSON_GetObjectItemCaseSensitive(root, "activation"))) cfg.activation = v->valuestring;
  if (bias) cfg.use_bias = cJSON_IsTrue(bias);
  if ((v = cJSON_GetObjectItemCaseSensitive(root, "loss"))) cfg.loss = v->valuestring;
  if ((v = cJSON_GetObjectItemCaseSensitive(root, "batch_size"))) cfg.batch_size = v->valueint;
  if ((v = cJSON_GetObjectItemCaseSensitive(root, "seed"))) cfg.seed = (uint64_t)v->valuedouble;
  cfg.n_layers = cJSON_GetArraySize(layers);
  for (int i = 0; i < cfg.n_layers; i++)
    cfg.layers[i] = cJSON_GetArrayItem(layers, i)->valueint;
  return mlp_build(ctx, &cfg, err);
}
