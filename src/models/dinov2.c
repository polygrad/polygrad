#include "vision.h"
#include <stdio.h>
#include <math.h>

/* Transformers5.3 Dinov2Model: fixed-resolution, unmasked eval encoder.
 * No counterpart exists in the pinned Tinygrad model catalogue. */
PolyModel *model_dinov2_from_config(PolyCtx *ctx, const cJSON *root, PolyModelError *err) {
  ModelVisionConfig c;
  bool bias, gated;
  double ratio;
  if (!model_vision_config(root, &c, err) ||
      !model_vision_bool(root, "qkv_bias", true, &bias, err) ||
      !model_vision_bool(root, "use_swiglu_ffn", false, &gated, err) ||
      !model_vision_float(root, "mlp_ratio", 4, &ratio, err))
    return NULL;
  if (ratio * c.dim > 65536) {
    model_factory_error(err, "mlp_ratio", "intermediate width exceeds 65536");
    return NULL;
  }
  c.hidden = (int)(ratio * c.dim);
  if (gated) c.hidden = ((int)(c.hidden * 2.0 / 3) + 7) / 8 * 8;
  if (c.hidden < 1) {
    model_factory_error(err, "mlp_ratio", "intermediate width must be positive");
    return NULL;
  }
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
    snprintf(name, sizeof(name), "%s.norm1", p);
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
    snprintf(name, sizeof(name), "%s.layer_scale1.lambda1", p);
    h = model_vision_scale(m, h, name, c.dim);
    x = poly_tensor_alu2(ctx, POLY_OP_ADD, x, h);
    snprintf(name, sizeof(name), "%s.norm2", p);
    h = poly_model_layernorm(m, name, x, c.dim, c.eps);
    snprintf(name, sizeof(name), "%s.mlp.%s", p, gated ? "weights_in" : "fc1");
    h = poly_model_linear(m, name, h, c.dim, c.hidden * (gated ? 2 : 1), true);
    if (gated) {
      PolyTensor *a = poly_tensor_silu(ctx, model_vision_slice(ctx, h, 2, 0, c.hidden));
      h = poly_tensor_alu2(
          ctx, POLY_OP_MUL, a, model_vision_slice(ctx, h, 2, c.hidden, 2 * c.hidden)
      );
    } else
      h = model_vision_activation(ctx, h, c.activation);
    snprintf(name, sizeof(name), "%s.mlp.%s", p, gated ? "weights_out" : "fc2");
    h = poly_model_linear(m, name, h, c.hidden, c.dim, true);
    snprintf(name, sizeof(name), "%s.layer_scale2.lambda1", p);
    h = model_vision_scale(m, h, name, c.dim);
    x = poly_tensor_contiguous(ctx, poly_tensor_alu2(ctx, POLY_OP_ADD, x, h));
  }
  x = poly_model_layernorm(m, "layernorm", x, c.dim, c.eps);
  PolyTensor *pool =
      poly_tensor_reshape(ctx, model_vision_slice(ctx, x, 1, 0, 1), (int64_t[]){c.batch, c.dim}, 2);
  if (!x || !pool || poly_model_output(m, "last_hidden_state", x) != POLY_STATUS_OK ||
      poly_model_output(m, "pooler_output", pool) != POLY_STATUS_OK ||
      poly_model_entrypoint(
          m, "forward", (const char *[]){"pixel_values"}, 1,
          (const char *[]){"last_hidden_state", "pooler_output"}, 2, NULL
      ) != POLY_STATUS_OK) {
    model_factory_error(err, "DINOv2", "graph construction failed");
    poly_model_free(m);
    return NULL;
  }
  return model_vision_finish(m, err);
}
PolyModel *model_dinov2_from_hf_decoded(
    const PolyHfDecoded *hf,
    const PolyGenericImportOpts *opts
) {
  return model_vision_import(hf, opts, model_dinov2_from_config, "embeddings.mask_token");
}
