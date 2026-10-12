/* TabM: shared numerical embeddings, BatchEnsemble backbone, independent heads.
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
    int sign_chunk
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
    else if (sign_chunk && i % sign_chunk)
      values[i] = values[i - i % sign_chunk];
    else
      values[i] = sign_chunk ? (values[i] < 0 ? -1 : 1) : values[i] / sqrtf(6);
  }
  int rc = poly_model_write_buf_named(model, name, values, (size_t)n * sizeof(float));
  free(values);
  return rc;
}

static int tabm_init_embedding_scaling(PolyModel *model, uint64_t seed, int chunk) {
  const char *name = "layers.0.r";
  int64_t n = poly_model_buf_numel_named(model, name);
  if (n <= 0 || n % chunk || (uint64_t)n > SIZE_MAX / sizeof(float)) return -1;
  float *values = malloc((size_t)n * sizeof(float));
  if (!values) return -1;
  /* TabM.make selects normal start scaling when embeddings are present, with
   * one shared draw per feature/member, not one per embedding coordinate. */
  poly_init_param_normal(seed, name, values, n / chunk);
  for (int64_t i = n; i-- > 0;)
    values[i] = values[i / chunk];
  int rc = poly_model_write_buf_named(model, name, values, (size_t)n * sizeof(float));
  free(values);
  return rc;
}

/* rtdl-num-embeddings 0.0.12: LinearReLUEmbeddings and
 * PiecewiseLinearEmbeddings(version='B'). The fit-time bin edges are inputs to
 * construction, never inferred from inference data. Ragged encodings reserve
 * their last coordinate for the last bin, including one-bin features. */
static PolyTensor *tabm_embeddings(
    PolyModel *model,
    PolyTensor *x,
    PolyUOp *batch,
    int features,
    int width,
    const cJSON *spec,
    PolyModelError *err
) {
  PolyCtx *ctx = poly_model_ctx(model);
  const char *type = cJSON_GetObjectItemCaseSensitive(spec, "type")->valuestring;
  bool piecewise = !strcmp(type, "piecewise_linear");
  bool activation = true;
  const cJSON *a = cJSON_GetObjectItemCaseSensitive(spec, "activation");
  if (piecewise) activation = a && cJSON_IsTrue(a);
  const char *wn = piecewise ? "num_module.linear0.weight" : "num_module.linear.weight";
  const char *bn = piecewise ? "num_module.linear0.bias" : "num_module.linear.bias";
  int64_t linear_shape[] = {features, width};
  PolyTensor *w = poly_model_param(model, wn, POLY_FLOAT32, linear_shape, 2);
  PolyTensor *b = poly_model_param(model, bn, POLY_FLOAT32, linear_shape, 2);
  PolyUOp *view[] = {batch, poly_uop_const_int(ctx, features), poly_uop_const_int(ctx, 1)};
  x = poly_tensor_reshape_uop(ctx, x, view, 3);
  PolyTensor *linear = x && w ? poly_tensor_alu2(ctx, POLY_OP_MUL, x, w) : NULL;
  linear = linear && b ? poly_tensor_alu2(ctx, POLY_OP_ADD, linear, b) : NULL;
  if (!linear) return NULL;
  PolyTensor *output = NULL;
  if (!piecewise)
    output = poly_tensor_relu(ctx, linear);
  else {
    const cJSON *bins = cJSON_GetObjectItemCaseSensitive(spec, "bins");
    if (!cJSON_IsArray(bins) || cJSON_GetArraySize(bins) != features) goto invalid_bins;
    int nb = 0;
    for (int f = 0; f < features; f++) {
      const cJSON *edges = cJSON_GetArrayItem(bins, f);
      int count = cJSON_GetArraySize(edges);
      if (!cJSON_IsArray(edges) || count < 2) goto invalid_bins;
      if (count - 1 > nb) nb = count - 1;
    }
    if ((size_t)features > SIZE_MAX / sizeof(float) / 4 / (size_t)nb) return NULL;
    size_t count = (size_t)features * nb;
    float *values = calloc(count * 4, sizeof(float));
    if (!values) return NULL;
    float *weight = values, *bias = values + count, *lo = bias + count, *hi = lo + count;
    for (int f = 0; f < features; f++) {
      const cJSON *edges = cJSON_GetArrayItem(bins, f);
      int nf = cJSON_GetArraySize(edges) - 1;
      for (int i = 0; i < nb; i++) {
        size_t offset = (size_t)f * nb + i;
        lo[offset] = i == 0 || (i == nb - 1 && nf == 1) ? -INFINITY : 0;
        hi[offset] = i == nb - 1 ? INFINITY : 1;
      }
      for (int i = 0; i < nf; i++) {
        const cJSON *left = cJSON_GetArrayItem(edges, i), *right = cJSON_GetArrayItem(edges, i + 1);
        float l = (float)left->valuedouble, r = (float)right->valuedouble;
        float scale = 1.0f / (r - l), offset = -l / (r - l);
        if (!cJSON_IsNumber(left) || !cJSON_IsNumber(right) || !isfinite(l) || !isfinite(r) ||
            !(r > l) || !isfinite(scale) || !isfinite(offset)) {
          free(values);
          goto invalid_bins;
        }
        size_t index = (size_t)f * nb + (i == nf - 1 ? nb - 1 : i);
        weight[index] = scale;
        bias[index] = offset;
      }
    }
    int64_t shape[] = {features, nb};
    const char *names[] = {
        "num_module.impl.weight", "num_module.impl.bias", "num_module.impl.lower",
        "num_module.impl.upper"};
    PolyTensor *constants[4];
    for (int i = 0; i < 4; i++)
      constants[i] = poly_model_aux_from_host(
          model, names[i], POLY_FLOAT32, shape, 2, values + i * count, count * sizeof(float)
      );
    free(values);
    for (int i = 0; i < 4; i++)
      if (!constants[i]) return NULL;
    PolyTensor *encoded = poly_tensor_alu2(ctx, POLY_OP_MUL, x, constants[0]);
    encoded = encoded ? poly_tensor_alu2(ctx, POLY_OP_ADD, encoded, constants[1]) : NULL;
    encoded = encoded ? poly_tensor_clamp(ctx, encoded, constants[2], constants[3]) : NULL;
    int64_t weights_shape[] = {features, nb, width}, perm[] = {1, 0, 2};
    w = poly_model_param(model, "num_module.linear.weight", POLY_FLOAT32, weights_shape, 3);
    encoded = encoded ? poly_tensor_permute(ctx, encoded, perm, 3) : NULL;
    output = encoded && w ? poly_tensor_dot(ctx, encoded, w) : NULL;
    output = output ? poly_tensor_permute(ctx, output, perm, 3) : NULL;
    if (output && activation) output = poly_tensor_relu(ctx, output);
    output = output ? poly_tensor_alu2(ctx, POLY_OP_ADD, linear, output) : NULL;
  }
  view[1] = poly_uop_const_int(ctx, (int64_t)features * width);
  return output ? poly_tensor_reshape_uop(ctx, output, view, 2) : NULL;
invalid_bins:
  model_factory_error(
      err, "num_embeddings.bins",
      "expected strictly increasing finite float32 edges for every feature"
  );
  return NULL;
}

PolyModel *model_tabm_from_config(PolyCtx *ctx, const cJSON *root, PolyModelError *err) {
  /* A misspelled or unsupported recipe option must not silently train a
   * different model. Keep the accepted keys beside the fields consumed below. */
  const char *keys[] = {"format",     "type", "layers",    "n_ensemble", "batch_size",    "seed",
                        "activation", "loss", "quantiles", "dropout",    "num_embeddings"};
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
  const cJSON *embedding = cJSON_GetObjectItemCaseSensitive(root, "num_embeddings");
  int embedding_width = 1;
  bool piecewise = false;
  if (embedding) {
    const cJSON *type = cJSON_GetObjectItemCaseSensitive(embedding, "type");
    if (!cJSON_IsObject(embedding) || !cJSON_IsString(type) ||
        (strcmp(type->valuestring, "linear_relu") && strcmp(type->valuestring, "piecewise_linear")
        ) ||
        !model_config_integer(embedding, "d_embedding", 1, INT_MAX, true, err)) {
      model_factory_error(
          err, "num_embeddings", "expected linear_relu or piecewise_linear with d_embedding"
      );
      return NULL;
    }
    piecewise = !strcmp(type->valuestring, "piecewise_linear");
    for (const cJSON *f = embedding->child; f; f = f->next) {
      bool valid = !strcmp(f->string, "type") || !strcmp(f->string, "d_embedding") ||
                   (piecewise && !strcmp(f->string, "bins")) ||
                   (piecewise && !strcmp(f->string, "activation") && cJSON_IsBool(f));
      if (!valid) {
        model_factory_error(err, "num_embeddings", "unsupported embedding option");
        return NULL;
      }
    }
    embedding_width = cJSON_GetObjectItemCaseSensitive(embedding, "d_embedding")->valueint;
    if (cJSON_GetArrayItem(layers, 0)->valueint > INT_MAX / embedding_width) {
      model_factory_error(err, "num_embeddings.d_embedding", "embedded input is too wide");
      return NULL;
    }
  }
  double dropout = 0;
  if ((v = cJSON_GetObjectItemCaseSensitive(root, "dropout"))) {
    if (!cJSON_IsNumber(v) || !isfinite(v->valuedouble) || v->valuedouble < 0 ||
        v->valuedouble > 1) {
      model_factory_error(err, "dropout", "expected a probability in [0, 1]");
      return NULL;
    }
    dropout = v->valuedouble;
  }
  PolyModel *model = poly_model_new(ctx, NULL);
  if (!model) return NULL;
  PolyTensorCapture *capture = dropout > 0 && strcmp(loss, "none")
                                   ? poly_tensor_capture_begin_seeded(ctx, (int64_t)seed)
                                   : NULL;
  if (dropout > 0 && strcmp(loss, "none") && !capture) goto fail;
  PolyUOp *batch = model_batch_dim(ctx, max_batch);
  int in = cJSON_GetArrayItem(layers, 0)->valueint;
  PolyUOp *input_shape[] = {batch, poly_uop_const_int(ctx, in)};
  PolyTensor *x = poly_model_input_uop(model, "x", POLY_FLOAT32, input_shape, 2);
  if (x && embedding) {
    x = tabm_embeddings(model, x, batch, in, embedding_width, embedding, err);
    in *= embedding_width;
    input_shape[1] = poly_uop_const_int(ctx, in);
  }
  PolyUOp *view[] = {batch, poly_uop_const_int(ctx, 1), input_shape[1]};
  if (x) x = poly_tensor_reshape_uop(ctx, x, view, 3);
  view[1] = poly_uop_const_int(ctx, k);
  if (x) x = poly_tensor_expand_uop(ctx, x, view, 3);
  if (!x) goto fail;
  PolyTensor *paths[] = {x, x};
  int n_paths = capture ? 2 : 1;

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
    for (int pass = 0; pass < n_paths; pass++) {
      x = poly_tensor_alu2(ctx, POLY_OP_MUL, paths[pass], r);
      x = x && w ? poly_tensor_dot(ctx, x, w) : NULL;
      x = x ? poly_tensor_alu2(ctx, POLY_OP_MUL, x, s) : NULL;
      x = x ? poly_tensor_alu2(ctx, POLY_OP_ADD, x, b) : NULL;
      x = x ? model_activation(ctx, x, activation) : NULL;
      if (x && pass) x = poly_tensor_dropout(ctx, x, dropout, true);
      if (!x) goto fail;
      paths[pass] = x;
    }
    in = out;
  }
  int out = cJSON_GetArrayItem(layers, n - 1)->valueint;
  int64_t ws[] = {k, in, out}, bs[] = {k, out}, transpose[] = {1, 0, 2};
  PolyTensor *w = poly_model_param(model, "output.weight", POLY_FLOAT32, ws, 3);
  PolyTensor *b = poly_model_param(model, "output.bias", POLY_FLOAT32, bs, 2);
  for (int pass = 0; pass < n_paths; pass++) {
    x = poly_tensor_permute(ctx, paths[pass], transpose, 3);
    x = x && w ? poly_tensor_dot(ctx, x, w) : NULL;
    x = x ? poly_tensor_permute(ctx, x, transpose, 3) : NULL;
    x = x && b ? poly_tensor_alu2(ctx, POLY_OP_ADD, x, b) : NULL;
    if (!x) goto fail;
    paths[pass] = x;
  }
  x = paths[0];
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
          model, paths[n_paths - 1], batch, out, k, loss,
          cJSON_GetObjectItemCaseSensitive(root, "quantiles"), err
      ) != POLY_STATUS_OK)
    goto fail;
  if (capture) {
    /* Capture owns the mutable RNG wrappers; the named Model adopts their
     * initial storage while the training graph retains the counter effects. */
    PolyTensor *states[2 * (POLY_DEVICE_DISK + 1)] = {0};
    int mutable[2 * (POLY_DEVICE_DISK + 1)] = {0}, ns = 0;
    for (int i = 0; i <= POLY_DEVICE_DISK; i++) {
      int device = poly_tensor_capture_rng(capture, i, &states[ns], &states[ns + 1]);
      if (device < 0) break;
      mutable[ns] = mutable[ns + 1] = 1;
      ns += 2;
    }
    PolyTensor *wrapped = NULL;
    int rc = poly_tensor_capture_wrap(capture, states, mutable, ns, &paths[1], 1, &wrapped);
    poly_tensor_release(wrapped);
    for (int i = 0; i < ns; i++) {
      char name[64];
      snprintf(
          name, sizeof(name), "__rng.%s.%s", poly_device_name(states[i]->device),
          i % 2 ? "counter" : "seed"
      );
      if (rc == 0 && poly_model_aux(model, name, states[i], 0) != POLY_STATUS_OK) rc = -1;
      poly_tensor_release(states[i]);
    }
    poly_tensor_capture_end(capture);
    capture = NULL;
    if (rc) goto fail;
  }
  if (poly_model_build(model, err) != POLY_STATUS_OK) goto fail;

  for (int l = 0; l < n - 2; l++) {
    char name[128];
    int width = cJSON_GetArrayItem(layers, l)->valueint * (l == 0 ? embedding_width : 1);
    snprintf(name, sizeof(name), "layers.%d.weight", l);
    if (tabm_init(model, name, seed, width, 0, false)) goto fail;
    snprintf(name, sizeof(name), "layers.%d.b", l);
    if (tabm_init(model, name, seed, width, cJSON_GetArrayItem(layers, l + 1)->valueint, false))
      goto fail;
    snprintf(name, sizeof(name), "layers.%d.r", l);
    if (l == 0 ? tabm_init(model, name, seed, width, 0, embedding_width)
               : model_init_param(model, name, 0, 0, 1))
      goto fail;
    snprintf(name, sizeof(name), "layers.%d.s", l);
    if (model_init_param(model, name, 0, 0, 1)) goto fail;
  }
  if (embedding) {
    if (tabm_init(
            model, piecewise ? "num_module.linear0.weight" : "num_module.linear.weight", seed,
            embedding_width, 0, false
        ) ||
        tabm_init(
            model, piecewise ? "num_module.linear0.bias" : "num_module.linear.bias", seed,
            embedding_width, 0, false
        ) ||
        (piecewise && model_init_param(model, "num_module.linear.weight", 0, 0, 0)))
      goto fail;
    if (n > 2 && tabm_init_embedding_scaling(model, seed, embedding_width)) goto fail;
  }
  if (tabm_init(model, "output.weight", seed, in, 0, false) ||
      tabm_init(model, "output.bias", seed, in, 0, false))
    goto fail;
  return model;
fail:
  if (capture) poly_tensor_capture_end(capture);
  poly_model_free(model);
  return NULL;
}
