/* TabM without feature embeddings: BatchEnsemble backbone, independent heads.
 * Equations/initialization: yandex-research/tabm (TabM, LinearBatchEnsemble,
 * LinearEnsemble). Seed/name-keyed randomness remains Polygrad's initializer. */
#include "factory.h"
#include "layers.h"
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int tabm_init(
    PolyModel *model,
    const char *name,
    uint64_t seed,
    int fan_in,
    int repeat,
    bool signs
) {
  int64_t n = poly_model_buf_numel_named(model, name);
  if (n <= 0 || (uint64_t)n > SIZE_MAX / sizeof(float)) return -1;
  float *values = malloc((size_t)n * sizeof(float));
  if (!values) return -1;
  poly_init_param_kaiming(seed, name, values, repeat ? repeat : n, fan_in);
  /* Reference Linear initializes U[-1/sqrt(fan_in),1/sqrt(fan_in)]. Hidden
   * biases start shared across members; head weights/biases are independent. */
  for (int64_t i = 0; i < n; i++) {
    if (repeat && i >= repeat)
      values[i] = values[i % repeat];
    else
      values[i] = signs ? (values[i] < 0 ? -1 : 1) : values[i] / sqrtf(6);
  }
  int rc = poly_model_write_buf_named(model, name, values, (size_t)n * sizeof(float));
  free(values);
  return rc;
}

PolyModel *model_tabm_from_config(PolyCtx *ctx, const cJSON *root, PolyModelError *err) {
  /* A misspelled or unsupported recipe option must not silently train a
   * different model. Keep the accepted keys beside the fields consumed below. */
  const char *keys[] = {"format", "type",       "layers", "n_ensemble", "batch_size",
                        "seed",   "activation", "loss",   "quantiles"};
  for (const cJSON *field = root->child; field; field = field->next) {
    bool known = false;
    for (size_t i = 0; i < sizeof(keys) / sizeof(keys[0]); i++)
      if (!strcmp(field->string, keys[i])) known = true;
    if (!known) {
      model_factory_error(err, field->string, "unsupported TabM option");
      return NULL;
    }
  }
  if (!model_config_training(root, err) ||
      !model_config_sizes(root, "layers", 2, INT_MAX, true, err) ||
      !model_config_integer(root, "n_ensemble", 1, INT_MAX, false, err) ||
      !model_config_choice(root, "activation", "|relu|gelu|silu|none|", err))
    return NULL;
  const cJSON *layers = cJSON_GetObjectItemCaseSensitive(root, "layers"), *v;
  int n = cJSON_GetArraySize(layers), k = 32, max_batch = 1;
  uint64_t seed = 42;
  const char *activation = "relu", *loss = "none";
  if ((v = cJSON_GetObjectItemCaseSensitive(root, "n_ensemble"))) k = v->valueint;
  if ((v = cJSON_GetObjectItemCaseSensitive(root, "batch_size"))) max_batch = v->valueint;
  if ((v = cJSON_GetObjectItemCaseSensitive(root, "seed"))) seed = (uint64_t)v->valuedouble;
  if ((v = cJSON_GetObjectItemCaseSensitive(root, "activation"))) activation = v->valuestring;
  if ((v = cJSON_GetObjectItemCaseSensitive(root, "loss"))) loss = v->valuestring;
  PolyModel *model = poly_model_new(ctx, NULL);
  if (!model) return NULL;
  PolyUOp *batch = model_batch_dim(ctx, max_batch);
  int in = cJSON_GetArrayItem(layers, 0)->valueint;
  PolyUOp *input_shape[] = {batch, poly_uop_const_int(ctx, in)};
  PolyTensor *x = poly_model_input_uop(model, "x", POLY_FLOAT32, input_shape, 2);
  PolyUOp *view[] = {batch, poly_uop_const_int(ctx, 1), input_shape[1]};
  if (x) x = poly_tensor_reshape_uop(ctx, x, view, 3);
  view[1] = poly_uop_const_int(ctx, k);
  if (x) x = poly_tensor_expand_uop(ctx, x, view, 3);
  if (!x) goto fail;

  for (int l = 0; l < n - 2; l++) {
    int out = cJSON_GetArrayItem(layers, l + 1)->valueint;
    if (poly_model_scope_push(model, "layers.%d", l) != POLY_STATUS_OK) goto fail;
    int64_t ws[] = {out, in}, rs[] = {k, in}, ss[] = {k, out};
    PolyTensor *w = poly_model_param(model, "weight", POLY_FLOAT32, ws, 2);
    PolyTensor *r = poly_model_param(model, "r", POLY_FLOAT32, rs, 2);
    PolyTensor *s = poly_model_param(model, "s", POLY_FLOAT32, ss, 2);
    PolyTensor *b = poly_model_param(model, "b", POLY_FLOAT32, ss, 2);
    if (poly_model_scope_pop(model) != POLY_STATUS_OK || !w || !r || !s || !b) goto fail;
    int64_t transpose[] = {1, 0};
    w = poly_tensor_permute(ctx, w, transpose, 2);
    x = poly_tensor_alu2(ctx, POLY_OP_MUL, x, r);
    x = x && w ? poly_tensor_dot(ctx, x, w) : NULL;
    x = x ? poly_tensor_alu2(ctx, POLY_OP_MUL, x, s) : NULL;
    x = x ? poly_tensor_alu2(ctx, POLY_OP_ADD, x, b) : NULL;
    x = x ? model_activation(ctx, x, activation) : NULL;
    if (!x) goto fail;
    in = out;
  }
  int out = cJSON_GetArrayItem(layers, n - 1)->valueint;
  int64_t ws[] = {k, in, out}, bs[] = {k, out}, transpose[] = {1, 0, 2};
  PolyTensor *w = poly_model_param(model, "output.weight", POLY_FLOAT32, ws, 3);
  PolyTensor *b = poly_model_param(model, "output.bias", POLY_FLOAT32, bs, 2);
  x = poly_tensor_permute(ctx, x, transpose, 3);
  x = x && w ? poly_tensor_dot(ctx, x, w) : NULL;
  x = x ? poly_tensor_permute(ctx, x, transpose, 3) : NULL;
  x = x && b ? poly_tensor_alu2(ctx, POLY_OP_ADD, x, b) : NULL;
  if (!x || poly_model_output(model, "members", x) != POLY_STATUS_OK) goto fail;
  const char *inputs[] = {"x"}, *members[] = {"members"}, *output[] = {"output"};
  if (poly_model_entrypoint(model, "members", inputs, 1, members, 1, NULL) != POLY_STATUS_OK)
    goto fail;
  int64_t axis[] = {1};
  PolyTensor *prediction;
  if (!strcmp(loss, "cross_entropy")) {
    /* Existing consumers apply softmax to output. Log mean probabilities keeps
     * that contract while implementing probability (not logit) averaging. */
    prediction = poly_tensor_log_softmax(ctx, x, -1);
    prediction = prediction ? poly_tensor_logsumexp(ctx, prediction, axis, 1, false) : NULL;
    PolyTensor *log_k = poly_tensor_full_from_value(
        ctx, poly_uop_const_typed(ctx, POLY_FLOAT32, log((double)k)), NULL, 0, POLY_DEVICE_AUTO,
        POLY_FLOAT32, true, false
    );
    prediction = prediction && log_k ? poly_tensor_alu2(ctx, POLY_OP_SUB, prediction, log_k) : NULL;
  } else
    prediction = poly_tensor_mean(ctx, x, axis, 1, false);
  if (!prediction || poly_model_output(model, "output", prediction) != POLY_STATUS_OK ||
      poly_model_entrypoint(model, "forward", inputs, 1, output, 1, NULL) != POLY_STATUS_OK ||
      model_tabular_loss(
          model, x, batch, out, k, loss, cJSON_GetObjectItemCaseSensitive(root, "quantiles"), err
      ) != POLY_STATUS_OK ||
      poly_model_build(model, err) != POLY_STATUS_OK)
    goto fail;

  for (int l = 0; l < n - 2; l++) {
    char name[128];
    int width = cJSON_GetArrayItem(layers, l)->valueint;
    snprintf(name, sizeof(name), "layers.%d.weight", l);
    if (tabm_init(model, name, seed, width, 0, false)) goto fail;
    snprintf(name, sizeof(name), "layers.%d.b", l);
    if (tabm_init(model, name, seed, width, cJSON_GetArrayItem(layers, l + 1)->valueint, false))
      goto fail;
    snprintf(name, sizeof(name), "layers.%d.r", l);
    if (l == 0 ? tabm_init(model, name, seed, width, 0, true)
               : model_init_param(model, name, 0, 0, 1))
      goto fail;
    snprintf(name, sizeof(name), "layers.%d.s", l);
    if (model_init_param(model, name, 0, 0, 1)) goto fail;
  }
  if (tabm_init(model, "output.weight", seed, in, 0, false) ||
      tabm_init(model, "output.bias", seed, in, 0, false))
    goto fail;
  return model;
fail:
  poly_model_free(model);
  return NULL;
}
