#include "vision.h"
#include "../loaders/bind.h"
#include "../loaders/import_error.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

bool model_vision_int(
    const cJSON *j,
    const char *key,
    int fallback,
    int min,
    int *out,
    PolyModelError *err
) {
  if (!model_config_integer(j, key, min, 65536, false, err)) return false;
  const cJSON *v = cJSON_GetObjectItemCaseSensitive(j, key);
  *out = v ? v->valueint : fallback;
  return *out >= min || model_factory_error(err, key, "missing positive dimension");
}
bool model_vision_float(
    const cJSON *j,
    const char *key,
    double fallback,
    double *out,
    PolyModelError *err
) {
  const cJSON *v = cJSON_GetObjectItemCaseSensitive(j, key);
  *out = v ? v->valuedouble : fallback;
  return ((!v || cJSON_IsNumber(v)) && isfinite(*out) && *out > 0) ||
         model_factory_error(err, key, "expected finite positive number");
}
bool model_vision_bool(
    const cJSON *j,
    const char *key,
    bool fallback,
    bool *out,
    PolyModelError *err
) {
  const cJSON *v = cJSON_GetObjectItemCaseSensitive(j, key);
  *out = v ? cJSON_IsTrue(v) : fallback;
  return !v || cJSON_IsBool(v) || model_factory_error(err, key, "expected boolean");
}
bool model_vision_config(const cJSON *j, ModelVisionConfig *c, PolyModelError *err) {
  memset(c, 0, sizeof(*c));
  if (!cJSON_IsObject(j)) return model_factory_error(err, "$", "expected encoder config object");
  if (!model_vision_int(j, "hidden_size", 0, 1, &c->dim, err) ||
      !model_vision_int(j, "num_attention_heads", 0, 1, &c->heads, err) ||
      !model_vision_int(j, "num_hidden_layers", 0, 1, &c->layers, err) ||
      !model_vision_int(j, "intermediate_size", c->dim * 4, 1, &c->hidden, err) ||
      !model_vision_int(j, "batch_size", 1, 1, &c->batch, err) ||
      !model_vision_int(j, "image_size", 224, 1, &c->image, err) ||
      !model_vision_int(j, "patch_size", 16, 1, &c->patch, err) ||
      !model_vision_int(j, "num_channels", 3, 1, &c->channels, err) ||
      !model_vision_float(j, "layer_norm_eps", 1e-6, &c->eps, err) ||
      !model_config_choice(j, "hidden_act", "|gelu|gelu_new|quick_gelu|silu|relu|", err))
    return false;
  if (c->dim % c->heads || c->image % c->patch || c->image / c->patch > 256)
    return model_factory_error(
        err, "dimensions",
        "heads must divide width; image must contain a square grid of at most 256x256 patches"
    );
  c->tokens = (c->image / c->patch) * (c->image / c->patch) + 1;
  if (c->layers > 256)
    return model_factory_error(err, "num_hidden_layers", "maximum supported layer count is 256");
  const cJSON *act = cJSON_GetObjectItemCaseSensitive(j, "hidden_act");
  c->activation = act ? act->valuestring : "gelu";
  return true;
}

PolyTensor *model_vision_slice(PolyCtx *ctx, PolyTensor *x, int axis, int64_t start, int64_t end) {
  if (!x) return NULL;
  int64_t pairs[POLY_MAX_DIMS][2];
  PolyUOp *u = poly_tensor_uop_physical(x);
  int n = poly_uop_ndim(ctx, u);
  const int64_t *shape = poly_uop_max_shape_dims(ctx, u);
  if (axis < 0) axis += n;
  if (!shape || n <= 0 || n > POLY_MAX_DIMS || axis < 0 || axis >= n) return NULL;
  for (int i = 0; i < n; i++) {
    pairs[i][0] = 0;
    pairs[i][1] = shape[i];
  }
  pairs[axis][0] = start;
  pairs[axis][1] = end;
  return poly_tensor_shrink(ctx, x, pairs, n);
}

/* tinygrad.nn.Conv2d followed by CLIP/ViT patch flatten/transpose. */
PolyTensor *model_vision_patch(
    PolyModel *m,
    const ModelVisionConfig *c,
    PolyTensor *x,
    const char *prefix,
    bool bias
) {
  PolyCtx *ctx = poly_model_ctx(m);
  char name[192];
  snprintf(name, sizeof(name), "%s.weight", prefix);
  PolyTensor *w = poly_model_param(
      m, name, POLY_FLOAT32, (int64_t[]){c->dim, c->channels, c->patch, c->patch}, 4
  );
  PolyTensor *b = NULL;
  if (bias) {
    snprintf(name, sizeof(name), "%s.bias", prefix);
    b = poly_model_param(m, name, POLY_FLOAT32, (int64_t[]){c->dim}, 1);
  }
  if (!w || (bias && !b)) return NULL;
  x = poly_tensor_conv2d(
      ctx, x, w, b, 1, (int64_t[]){c->patch, c->patch}, (int64_t[]){1, 1}, (int64_t[]){0, 0, 0, 0},
      4
  );
  x = poly_tensor_reshape(
      ctx, x, (int64_t[]){c->batch, c->dim, (c->image / c->patch) * (c->image / c->patch)}, 3
  );
  return poly_tensor_permute(ctx, x, (int64_t[]){0, 2, 1}, 3);
}

/* CLIP/ViT multihead attention. DINOv3 supplies full-sequence rotary AUX with
 * identity entries for prefix tokens, so the existing half-split RoPE applies. */
PolyTensor *model_vision_attention(
    PolyModel *m,
    const ModelVisionConfig *c,
    PolyTensor *x,
    const char *const names[4],
    const bool bias[4],
    bool causal,
    PolyTensor *cos,
    PolyTensor *sin
) {
  PolyCtx *ctx = poly_model_ctx(m);
  PolyTensor *qkv[3];
  for (int i = 0; i < 3; i++) {
    PolyTensor *v = poly_model_linear(m, names[i], x, c->dim, c->dim, bias[i]);
    v = poly_tensor_reshape(
        ctx, v, (int64_t[]){c->batch, c->tokens, c->heads, c->dim / c->heads}, 4
    );
    v = poly_tensor_permute(ctx, v, (int64_t[]){0, 2, 1, 3}, 4);
    qkv[i] = i < 2 && cos ? poly_tensor_rope(ctx, v, cos, sin) : v;
    if (!qkv[i]) return NULL;
  }
  x = poly_tensor_sdpa(ctx, qkv[0], qkv[1], qkv[2], NULL, 0, causal, 0, 0);
  x = poly_tensor_permute(ctx, x, (int64_t[]){0, 2, 1, 3}, 4);
  x = poly_tensor_reshape(ctx, x, (int64_t[]){c->batch, c->tokens, c->dim}, 3);
  return poly_model_linear(m, names[3], x, c->dim, c->dim, bias[3]);
}
PolyTensor *model_vision_activation(PolyCtx *ctx, PolyTensor *x, const char *act) {
  if (!strcmp(act, "gelu")) return poly_tensor_gelu_exact(ctx, x);
  if (!strcmp(act, "gelu_new")) return poly_tensor_gelu(ctx, x);
  if (!strcmp(act, "quick_gelu")) return poly_tensor_quick_gelu(ctx, x);
  return model_activation(ctx, x, act);
}
PolyTensor *model_vision_scale(PolyModel *m, PolyTensor *x, const char *name, int dim) {
  PolyTensor *w = poly_model_param(m, name, POLY_FLOAT32, (int64_t[]){dim}, 1);
  return poly_tensor_alu2(poly_model_ctx(m), POLY_OP_MUL, x, w);
}
PolyModel *model_vision_finish(PolyModel *m, PolyModelError *err) {
  if (m && poly_model_build(m, err) == POLY_STATUS_OK && poly_model_require_weights(m) == 0)
    return m;
  poly_model_free(m);
  return NULL;
}

/* HF names are retained by these four builders. This adapter owns decoding's
 * transaction, not a second name registry; no prefix copying or shape coercion. */
PolyModel *model_vision_import(
    const PolyHfDecoded *hf,
    const PolyGenericImportOpts *opts,
    PolyModel *(*build)(PolyCtx *, const cJSON *, PolyModelError *),
    const char *unused
) {
  if (!hf || !opts) return NULL;
  PolyModelError err = {0};
  cJSON *json = cJSON_Duplicate(hf->config, true);
  if (!json) return NULL;
  if (opts->max_batch > 0) {
    cJSON_DeleteItemFromObject(json, "batch_size");
    cJSON_AddNumberToObject(json, "batch_size", opts->max_batch);
  }
  if (opts->max_seq_len > 0) {
    cJSON_DeleteItemFromObject(json, "max_seq_len");
    cJSON_AddNumberToObject(json, "max_seq_len", opts->max_seq_len);
  }
  PolyModelFactoryScope scope;
  PolyModel *m = NULL;
  if (model_factory_begin(&scope, opts->ctx, opts->device))
    m = model_factory_end(&scope, build(scope.ctx, json, &err));
  cJSON_Delete(json);
  if (!m) {
    poly_import_error_set(
        POLY_IMPORT_ERR_UNSUPPORTED_MODEL, "%s",
        err.message[0] ? err.message : "vision construction failed"
    );
    return NULL;
  }
  int n = poly_model_buf_count(m);
  bool *seen = calloc((size_t)n, sizeof(bool));
  PolyBindIndex *index = poly_bind_index_create(m);
  if (!seen || !index) goto fail;
  for (int i = 0; i < hf->n_tensors; i++) {
    const PolyDecodedTensor *t = &hf->tensors[i];
    /* Mask tokens are unused by the explicitly unmasked inference signature. */
    if (unused && !strcmp(t->name, unused)) continue;
    /* HF serializes an empty register parameter when DINOv3 uses no registers. */
    if (!strcmp(hf->model_type, "dinov3_vit") && !strcmp(t->name, "embeddings.register_tokens") &&
        t->ndim == 3 && t->shape[0] == 1 && t->shape[1] == 0 && t->numel == 0)
      continue;
    int b = 0;
    while (b < n && strcmp(t->name, poly_model_buf_name(m, b)))
      b++;
    if (b == n || poly_model_buf_role(m, b) != POLY_ROLE_PARAM || seen[b] ||
        poly_import_bind_tensor(index, t->name, t, 0, -1) != 1) {
      poly_import_error_set(
          POLY_IMPORT_ERR_WEIGHT_MISMATCH, "invalid, unexpected or duplicate vision weight '%s'",
          t->name
      );
      goto fail;
    }
    seen[b] = true;
  }
  for (int b = 0; b < n; b++)
    if (poly_model_buf_role(m, b) == POLY_ROLE_PARAM && !seen[b]) {
      poly_import_error_set(
          POLY_IMPORT_ERR_WEIGHT_MISMATCH, "missing vision weight '%s'", poly_model_buf_name(m, b)
      );
      goto fail;
    }
  free(seen);
  poly_bind_index_destroy(index);
  return m;
fail:
  free(seen);
  poly_bind_index_destroy(index);
  poly_model_free(m);
  if (poly_import_last_error_code() == POLY_IMPORT_OK)
    poly_import_error_set(POLY_IMPORT_ERR_INTERNAL, "vision checkpoint allocation failed");
  return NULL;
}
