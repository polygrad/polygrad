#include "vision.h"
#include <stdio.h>
#include <string.h>

/* Pinned extra/models/clip.py: pre-norm residual encoders and CLIP scoring.
 * HF names/layout and EOS pooling follow Transformers CLIPModel. */
static PolyTensor *clip_encoder(
    PolyModel *m,
    const ModelVisionConfig *c,
    PolyTensor *h,
    const char *tower,
    bool causal
) {
  PolyCtx *ctx = poly_model_ctx(m);
  for (int i = 0; i < c->layers; i++) {
    char prefix[128], name[192], proj[4][192];
    snprintf(prefix, sizeof(prefix), "%s.encoder.layers.%d", tower, i);
    snprintf(name, sizeof(name), "%s.layer_norm1", prefix);
    PolyTensor *x = poly_model_layernorm(m, name, h, c->dim, c->eps);
    const char *suffix[] = {"q_proj", "k_proj", "v_proj", "out_proj"};
    const char *names[4];
    for (int k = 0; k < 4; k++) {
      snprintf(proj[k], sizeof(proj[k]), "%s.self_attn.%s", prefix, suffix[k]);
      names[k] = proj[k];
    }
    x = model_vision_attention(
        m, c, x, names, (bool[]){true, true, true, true}, causal, NULL, NULL
    );
    h = poly_tensor_alu2(ctx, POLY_OP_ADD, h, x);
    snprintf(name, sizeof(name), "%s.layer_norm2", prefix);
    x = poly_model_layernorm(m, name, h, c->dim, c->eps);
    snprintf(name, sizeof(name), "%s.mlp.fc1", prefix);
    x = model_vision_activation(
        ctx, poly_model_linear(m, name, x, c->dim, c->hidden, true), c->activation
    );
    snprintf(name, sizeof(name), "%s.mlp.fc2", prefix);
    x = poly_model_linear(m, name, x, c->hidden, c->dim, true);
    h = poly_tensor_contiguous(ctx, poly_tensor_alu2(ctx, POLY_OP_ADD, h, x));
    if (!h) return NULL;
  }
  return h;
}
static PolyTensor *clip_normalize(PolyCtx *ctx, PolyTensor *x) {
  PolyTensor *norm =
      poly_tensor_sum(ctx, poly_tensor_alu2(ctx, POLY_OP_MUL, x, x), (int64_t[]){1}, 1, true);
  return poly_tensor_div(ctx, x, poly_tensor_alu1(ctx, POLY_OP_SQRT, norm), 0);
}
PolyModel *model_clip_from_config(PolyCtx *ctx, const cJSON *root, PolyModelError *err) {
  ModelVisionConfig v, t;
  const cJSON *vj = cJSON_GetObjectItemCaseSensitive(root, "vision_config");
  const cJSON *tj = cJSON_GetObjectItemCaseSensitive(root, "text_config");
  int batch, projection, length, max_length, vocab, eos;
  if (!model_vision_config(vj, &v, err) || !model_vision_config(tj, &t, err) ||
      !model_vision_int(root, "batch_size", 1, 1, &batch, err) ||
      !model_vision_int(root, "projection_dim", 512, 1, &projection, err) ||
      !model_vision_int(tj, "max_position_embeddings", 77, 1, &max_length, err) ||
      !model_vision_int(root, "max_seq_len", max_length, 1, &length, err) ||
      !model_vision_int(tj, "vocab_size", 49408, 1, &vocab, err) ||
      !model_vision_int(tj, "eos_token_id", 49407, 0, &eos, err))
    return NULL;
  if (length > max_length || eos >= vocab) {
    model_factory_error(err, "text_config", "sequence exceeds positions or EOS exceeds vocabulary");
    return NULL;
  }
  v.batch = t.batch = batch;
  t.tokens = length;
  if (!cJSON_GetObjectItemCaseSensitive(vj, "layer_norm_eps")) v.eps = 1e-5;
  if (!cJSON_GetObjectItemCaseSensitive(tj, "layer_norm_eps")) t.eps = 1e-5;
  if (!cJSON_GetObjectItemCaseSensitive(vj, "hidden_act")) v.activation = "quick_gelu";
  if (!cJSON_GetObjectItemCaseSensitive(tj, "hidden_act")) t.activation = "quick_gelu";
  PolyModel *m = poly_model_new(ctx, NULL);
  if (!m) return NULL;
  PolyTensor *pixels = poly_model_input(
      m, "pixel_values", POLY_FLOAT32, (int64_t[]){batch, v.channels, v.image, v.image}, 4
  );
  PolyTensor *ids = poly_model_input(m, "input_ids", POLY_INT32, (int64_t[]){batch, length}, 2);
  PolyTensor *image =
      model_vision_patch(m, &v, pixels, "vision_model.embeddings.patch_embedding", false);
  PolyTensor *cls = poly_model_param(
      m, "vision_model.embeddings.class_embedding", POLY_FLOAT32, (int64_t[]){v.dim}, 1
  );
  cls = poly_tensor_reshape(ctx, cls, (int64_t[]){1, 1, v.dim}, 3);
  cls = poly_tensor_expand(ctx, cls, (int64_t[]){batch, 1, v.dim}, 3);
  image = poly_tensor_cat(ctx, (PolyTensor *[]){cls, image}, 2, 1);
  PolyTensor *pos = poly_model_param(
      m, "vision_model.embeddings.position_embedding.weight", POLY_FLOAT32,
      (int64_t[]){v.tokens, v.dim}, 2
  );
  image = poly_tensor_alu2(ctx, POLY_OP_ADD, image, pos);
  image = poly_model_layernorm(m, "vision_model.pre_layrnorm", image, v.dim, v.eps);
  image = clip_encoder(m, &v, image, "vision_model", false);
  image = model_vision_slice(ctx, image, 1, 0, 1);
  image = poly_tensor_reshape(ctx, image, (int64_t[]){batch, v.dim}, 2);
  image = poly_model_layernorm(m, "vision_model.post_layernorm", image, v.dim, v.eps);
  image = poly_model_linear(m, "visual_projection", image, v.dim, projection, false);
  PolyTensor *text =
      poly_model_embedding(m, "text_model.embeddings.token_embedding", ids, vocab, t.dim);
  pos = poly_model_param(
      m, "text_model.embeddings.position_embedding.weight", POLY_FLOAT32,
      (int64_t[]){max_length, t.dim}, 2
  );
  pos = model_vision_slice(ctx, pos, 0, 0, length);
  text = poly_tensor_alu2(ctx, POLY_OP_ADD, text, pos);
  text = clip_encoder(m, &t, text, "text_model", true);
  text = poly_model_layernorm(m, "text_model.final_layer_norm", text, t.dim, t.eps);
  /* HF's legacy eos=2 uses max token ID; otherwise pool the first EOS. Like
   * the reference, callers must supply EOS and right-pad the token sequence. */
  PolyTensor *pool_ids =
      eos == 2
          ? ids
          : poly_tensor_alu2(ctx, POLY_OP_CMPEQ, ids, poly_tensor_const_like_int(ctx, ids, eos));
  pool_ids = poly_tensor_argmax(ctx, pool_ids, 1, false);
  pool_ids = poly_tensor_reshape(ctx, pool_ids, (int64_t[]){batch, 1, 1}, 3);
  pool_ids = poly_tensor_expand(ctx, pool_ids, (int64_t[]){batch, 1, t.dim}, 3);
  text = poly_tensor_gather_dim(ctx, text, 1, pool_ids);
  text = poly_tensor_reshape(ctx, text, (int64_t[]){batch, t.dim}, 2);
  text = poly_model_linear(m, "text_projection", text, t.dim, projection, false);
  image = clip_normalize(ctx, image);
  text = clip_normalize(ctx, text);
  PolyTensor *scale =
      poly_tensor_exp(ctx, poly_model_param(m, "logit_scale", POLY_FLOAT32, NULL, 0));
  PolyTensor *scores =
      poly_tensor_dot(ctx, text, poly_tensor_permute(ctx, image, (int64_t[]){1, 0}, 2));
  scores = poly_tensor_alu2(ctx, POLY_OP_MUL, scores, scale);
  PolyTensor *iscores = poly_tensor_permute(ctx, scores, (int64_t[]){1, 0}, 2);
  const char *outputs[] = {"image_embeds", "text_embeds", "logits_per_image", "logits_per_text"};
  PolyTensor *values[] = {image, text, iscores, scores};
  for (int i = 0; i < 4; i++)
    if (!values[i] || poly_model_output(m, outputs[i], values[i]) != POLY_STATUS_OK) goto fail;
  if (poly_model_entrypoint(
          m, "forward", (const char *[]){"pixel_values", "input_ids"}, 2, outputs, 4, NULL
      ) != POLY_STATUS_OK ||
      poly_model_entrypoint(
          m, "encode_image", (const char *[]){"pixel_values"}, 1, outputs, 1, NULL
      ) != POLY_STATUS_OK ||
      poly_model_entrypoint(
          m, "encode_text", (const char *[]){"input_ids"}, 1, outputs + 1, 1, NULL
      ) != POLY_STATUS_OK)
    goto fail;
  return model_vision_finish(m, err);
fail:
  model_factory_error(err, "CLIP", "graph construction failed");
  poly_model_free(m);
  return NULL;
}
PolyModel *model_clip_from_hf_decoded(const PolyHfDecoded *hf, const PolyGenericImportOpts *opts) {
  return model_vision_import(hf, opts, model_clip_from_config, NULL);
}
