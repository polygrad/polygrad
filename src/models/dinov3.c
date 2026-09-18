#include "vision.h"
#include "../device.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* Transformers5.3 DINOv3ViTRopePositionEmbedding in eval mode: normalized
 * patch centers, independent Y/X frequencies, no rotation of class/registers.
 * Fixed-shape AUX uses half-width tables expected by Tensor.rope. */
static PolyTensor *dinov3_rotary(
    PolyModel *m,
    const ModelVisionConfig *c,
    int registers,
    double theta,
    bool sine
) {
  int half = c->dim / c->heads / 2, quarter = half / 2, grid = c->image / c->patch,
      prefix = 1 + registers;
  size_t n = (size_t)c->tokens * half;
  float *values = malloc(n * sizeof(float));
  if (!values) return NULL;
  for (int t = 0; t < c->tokens; t++)
    for (int j = 0; j < half; j++) {
      float value = sine ? 0 : 1;
      if (t >= prefix) {
        int p = t - prefix, coordinate = j < quarter ? p / grid : p % grid;
        float coord = 2.0f * ((coordinate + 0.5f) / grid) - 1.0f;
        float freq = 1.0f / powf((float)theta, (float)(j % quarter) / quarter);
        float angle = (6.2831853071795864769f * coord) * freq;
        value = sine ? sinf(angle) : cosf(angle);
      }
      values[(size_t)t * half + j] = value;
    }
  PolyCtx *ctx = poly_model_ctx(m);
  PolyDevice device = poly_ctx_get_preferred_device(ctx);
  PolyTensor *x =
      poly_tensor_empty(ctx, POLY_FLOAT32, (int64_t[]){1, 1, c->tokens, half}, 4, device);
  PolyUOp *buffer = x ? (PolyUOp *)poly_uop_get_buffer_identity(poly_tensor_uop_physical(x)) : NULL;
  bool copied = buffer && poly_buffer_ensure_device_allocated(ctx, buffer, device) == 0 &&
                poly_buffer_write(ctx, buffer, values, n * sizeof(float)) == 0;
  free(values);
  if (!copied || poly_model_aux(m, sine ? "rope_sin" : "rope_cos", x, 0) != POLY_STATUS_OK) {
    poly_tensor_release(x);
    return NULL;
  }
  return x;
}

PolyModel *model_dinov3_from_config(PolyCtx *ctx, const cJSON *root, PolyModelError *err) {
  ModelVisionConfig c;
  int registers;
  bool bias[4], mlp_bias, gated;
  double theta;
  const char *keys[] = {"query_bias", "key_bias", "value_bias", "proj_bias"};
  if (!model_vision_config(root, &c, err) ||
      !model_vision_int(root, "num_register_tokens", 0, 0, &registers, err) ||
      !model_vision_float(root, "rope_theta", 100, &theta, err) ||
      !model_vision_bool(root, "mlp_bias", true, &mlp_bias, err) ||
      !model_vision_bool(root, "use_gated_mlp", false, &gated, err))
    return NULL;
  for (int i = 0; i < 4; i++)
    if (!model_vision_bool(root, keys[i], i != 1, &bias[i], err)) return NULL;
  if (c.dim / c.heads % 4 || theta < 1 || theta > 1e30 || registers > 64) {
    model_factory_error(
        err, "DINOv3",
        "head dimension must divide into four rotary quarters; at most 64 registers; theta in "
        "[1,1e30]"
    );
    return NULL;
  }
  c.tokens += registers;
  if (!cJSON_GetObjectItemCaseSensitive(root, "layer_norm_eps")) c.eps = 1e-5;
  PolyModel *m = poly_model_new(ctx, NULL);
  if (!m) return NULL;
  PolyTensor *cos = dinov3_rotary(m, &c, registers, theta, false),
             *sin = dinov3_rotary(m, &c, registers, theta, true);
  if (!cos || !sin) goto fail;
  PolyTensor *x = poly_model_input(
      m, "pixel_values", POLY_FLOAT32, (int64_t[]){c.batch, c.channels, c.image, c.image}, 4
  );
  x = model_vision_patch(m, &c, x, "embeddings.patch_embeddings", true);
  PolyTensor *cls =
      poly_model_param(m, "embeddings.cls_token", POLY_FLOAT32, (int64_t[]){1, 1, c.dim}, 3);
  cls = poly_tensor_expand(ctx, cls, (int64_t[]){c.batch, 1, c.dim}, 3);
  if (registers) {
    PolyTensor *r = poly_model_param(
        m, "embeddings.register_tokens", POLY_FLOAT32, (int64_t[]){1, registers, c.dim}, 3
    );
    r = poly_tensor_expand(ctx, r, (int64_t[]){c.batch, registers, c.dim}, 3);
    x = poly_tensor_cat(ctx, (PolyTensor *[]){cls, r, x}, 3, 1);
  } else
    x = poly_tensor_cat(ctx, (PolyTensor *[]){cls, x}, 2, 1);
  for (int i = 0; i < c.layers; i++) {
    char name[160], p[128], proj[4][192];
    snprintf(p, sizeof(p), "layer.%d", i);
    snprintf(name, sizeof(name), "%s.norm1", p);
    PolyTensor *h = poly_model_layernorm(m, name, x, c.dim, c.eps);
    const char *suffix[] = {"q_proj", "k_proj", "v_proj", "o_proj"}, *names[4];
    for (int k = 0; k < 4; k++) {
      snprintf(proj[k], sizeof(proj[k]), "%s.attention.%s", p, suffix[k]);
      names[k] = proj[k];
    }
    h = model_vision_attention(m, &c, h, names, bias, false, cos, sin);
    snprintf(name, sizeof(name), "%s.layer_scale1.lambda1", p);
    h = model_vision_scale(m, h, name, c.dim);
    x = poly_tensor_alu2(ctx, POLY_OP_ADD, x, h);
    snprintf(name, sizeof(name), "%s.norm2", p);
    h = poly_model_layernorm(m, name, x, c.dim, c.eps);
    snprintf(name, sizeof(name), "%s.mlp.up_proj", p);
    PolyTensor *up = poly_model_linear(m, name, h, c.dim, c.hidden, mlp_bias);
    if (gated) {
      snprintf(name, sizeof(name), "%s.mlp.gate_proj", p);
      PolyTensor *gate = model_vision_activation(
          ctx, poly_model_linear(m, name, h, c.dim, c.hidden, mlp_bias), c.activation
      );
      h = poly_tensor_alu2(ctx, POLY_OP_MUL, gate, up);
    } else
      h = model_vision_activation(ctx, up, c.activation);
    snprintf(name, sizeof(name), "%s.mlp.down_proj", p);
    h = poly_model_linear(m, name, h, c.hidden, c.dim, mlp_bias);
    snprintf(name, sizeof(name), "%s.layer_scale2.lambda1", p);
    h = model_vision_scale(m, h, name, c.dim);
    x = poly_tensor_contiguous(ctx, poly_tensor_alu2(ctx, POLY_OP_ADD, x, h));
  }
  x = poly_model_layernorm(m, "norm", x, c.dim, c.eps);
  PolyTensor *pool =
      poly_tensor_reshape(ctx, model_vision_slice(ctx, x, 1, 0, 1), (int64_t[]){c.batch, c.dim}, 2);
  if (!x || !pool || poly_model_output(m, "last_hidden_state", x) != POLY_STATUS_OK ||
      poly_model_output(m, "pooler_output", pool) != POLY_STATUS_OK ||
      poly_model_entrypoint(
          m, "forward", (const char *[]){"pixel_values"}, 1,
          (const char *[]){"last_hidden_state", "pooler_output"}, 2, NULL
      ) != POLY_STATUS_OK)
    goto fail;
  return model_vision_finish(m, err);
fail:
  model_factory_error(err, "DINOv3", "graph construction failed");
  poly_model_free(m);
  return NULL;
}
PolyModel *model_dinov3_from_hf_decoded(
    const PolyHfDecoded *hf,
    const PolyGenericImportOpts *opts
) {
  return model_vision_import(hf, opts, model_dinov3_from_config, "embeddings.mask_token");
}
