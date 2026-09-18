#include "vision.h"
#include <stdio.h>

/* extra/models/vit.py:ViT and TransformerBlock(prenorm=True), with HF state
 * names, exact GELU, configured epsilon and the ViTModel tanh pooler. */
PolyModel *model_vit_from_config(PolyCtx *ctx, const cJSON *root, PolyModelError *err) {
  ModelVisionConfig c;
  bool bias;
  if (!model_vision_config(root, &c, err) || !model_vision_bool(root, "qkv_bias", true, &bias, err))
    return NULL;
  if (!cJSON_GetObjectItemCaseSensitive(root, "layer_norm_eps")) c.eps = 1e-12;
  PolyModel *m = poly_model_new(ctx, NULL);
  if (!m) return NULL;
  PolyTensor *x = poly_model_input(
      m, "pixel_values", POLY_FLOAT32, (int64_t[]){c.batch, c.channels, c.image, c.image}, 4
  );
  x = model_vision_patch(m, &c, x, "embeddings.patch_embeddings.projection", true);
  PolyTensor *cls =
      poly_model_param(m, "embeddings.cls_token", POLY_FLOAT32, (int64_t[]){1, 1, c.dim}, 3);
  cls = poly_tensor_expand(ctx, cls, (int64_t[]){c.batch, 1, c.dim}, 3);
  x = poly_tensor_cat(ctx, (PolyTensor *[]){cls, x}, 2, 1);
  PolyTensor *pos = poly_model_param(
      m, "embeddings.position_embeddings", POLY_FLOAT32, (int64_t[]){1, c.tokens, c.dim}, 3
  );
  x = poly_tensor_alu2(ctx, POLY_OP_ADD, x, pos);
  for (int i = 0; i < c.layers; i++) {
    char name[160], p[128], proj[4][192];
    snprintf(p, sizeof(p), "encoder.layer.%d", i);
    snprintf(name, sizeof(name), "%s.layernorm_before", p);
    PolyTensor *h = poly_model_layernorm(m, name, x, c.dim, c.eps);
    const char *suffix[] =
        {"attention.attention.query", "attention.attention.key", "attention.attention.value",
         "attention.output.dense"},
               *names[4];
    for (int k = 0; k < 4; k++) {
      snprintf(proj[k], sizeof(proj[k]), "%s.%s", p, suffix[k]);
      names[k] = proj[k];
    }
    h = model_vision_attention(
        m, &c, h, names, (bool[]){bias, bias, bias, true}, false, NULL, NULL
    );
    x = poly_tensor_alu2(ctx, POLY_OP_ADD, x, h);
    snprintf(name, sizeof(name), "%s.layernorm_after", p);
    h = poly_model_layernorm(m, name, x, c.dim, c.eps);
    snprintf(name, sizeof(name), "%s.intermediate.dense", p);
    h = model_vision_activation(
        ctx, poly_model_linear(m, name, h, c.dim, c.hidden, true), c.activation
    );
    snprintf(name, sizeof(name), "%s.output.dense", p);
    h = poly_model_linear(m, name, h, c.hidden, c.dim, true);
    x = poly_tensor_contiguous(ctx, poly_tensor_alu2(ctx, POLY_OP_ADD, x, h));
  }
  x = poly_model_layernorm(m, "layernorm", x, c.dim, c.eps);
  PolyTensor *pool =
      poly_tensor_reshape(ctx, model_vision_slice(ctx, x, 1, 0, 1), (int64_t[]){c.batch, c.dim}, 2);
  pool = poly_tensor_tanh(ctx, poly_model_linear(m, "pooler.dense", pool, c.dim, c.dim, true));
  if (!x || !pool || poly_model_output(m, "last_hidden_state", x) != POLY_STATUS_OK ||
      poly_model_output(m, "pooler_output", pool) != POLY_STATUS_OK ||
      poly_model_entrypoint(
          m, "forward", (const char *[]){"pixel_values"}, 1,
          (const char *[]){"last_hidden_state", "pooler_output"}, 2, NULL
      ) != POLY_STATUS_OK) {
    model_factory_error(err, "ViT", "graph construction failed");
    poly_model_free(m);
    return NULL;
  }
  return model_vision_finish(m, err);
}
PolyModel *model_vit_from_hf_decoded(const PolyHfDecoded *hf, const PolyGenericImportOpts *opts) {
  return model_vision_import(hf, opts, model_vit_from_config, NULL);
}
