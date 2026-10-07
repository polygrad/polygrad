/* Gemma4's vision tower, shared by multimodal Gemma-family model builders.
 * Inputs are processor-produced flattened patches and integer (x,y) positions.
 * Keep padded soft tokens plus a mask: the enclosing model inserts valid tokens
 * into text slots without a data-dependent host readback or output allocation. */
#include "gemma.h"
#include "vision.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

PolyTensor *model_gemma4_vision(
    PolyModel *m,
    const cJSON *j,
    int b,
    int n,
    PolyTensor *pixels,
    PolyTensor *positions,
    PolyTensor **valid_out,
    PolyTensor **hidden_states,
    PolyModelError *err
) {
  PolyCtx *ctx = poly_model_ctx(m);
  int d, hidden, heads, kv, hd, layers, patch, pool, table_size;
  double eps, theta;
  bool clipped, standardize, bias;
  const cJSON *rope = cJSON_GetObjectItemCaseSensitive(j, "rope_parameters");
  if (rope && !cJSON_IsObject(rope)) {
    model_factory_error(err, "vision_config.rope_parameters", "expected object");
    return NULL;
  }
  if (!model_vision_int(j, "hidden_size", 768, 1, &d, err) ||
      !model_vision_int(j, "intermediate_size", 3072, 1, &hidden, err) ||
      !model_vision_int(j, "num_attention_heads", 12, 1, &heads, err) ||
      !model_vision_int(j, "num_key_value_heads", heads, 1, &kv, err) ||
      !model_vision_int(j, "head_dim", 64, 1, &hd, err) ||
      !model_vision_int(j, "num_hidden_layers", 16, 1, &layers, err) ||
      !model_vision_int(j, "patch_size", 16, 1, &patch, err) ||
      !model_vision_int(j, "pooling_kernel_size", 3, 2, &pool, err) ||
      !model_vision_int(j, "position_embedding_size", 10240, 1, &table_size, err) ||
      !model_vision_float(j, "rms_norm_eps", 1e-6, &eps, err) ||
      !model_vision_float(rope, "rope_theta", 100, &theta, err) ||
      !model_vision_bool(j, "use_clipped_linears", false, &clipped, err) ||
      !model_vision_bool(j, "standardize", false, &standardize, err) ||
      !model_vision_bool(j, "attention_bias", false, &bias, err) ||
      !model_config_choice(j, "hidden_activation", "|gelu_pytorch_tanh|", err) ||
      !model_config_choice(rope, "rope_type", "|axial|", err))
    return NULL;
  if (hd % 4 || heads % kv || layers > 256 || patch > 256 || pool > 256 || n % (pool * pool) ||
      heads > 256 || hd > 4096 || clipped || bias) {
    model_factory_error(
        err, "vision_config", "invalid dimensions or unsupported clipped/bias projections"
    );
    return NULL;
  }
  int out_n = n / (pool * pool);
  PolyTensor *px = model_vision_slice(ctx, positions, 2, 0, 1);
  PolyTensor *py = model_vision_slice(ctx, positions, 2, 1, 2);
  px = poly_tensor_reshape(ctx, px, (int64_t[]){b, n}, 2);
  py = poly_tensor_reshape(ctx, py, (int64_t[]){b, n}, 2);
  PolyTensor *neg = poly_tensor_const_like_int(ctx, px, -1);
  PolyTensor *pad = poly_tensor_alu2(
      ctx, POLY_OP_AND, poly_tensor_alu2(ctx, POLY_OP_CMPEQ, px, neg),
      poly_tensor_alu2(ctx, POLY_OP_CMPEQ, py, neg)
  );
  PolyTensor *valid =
      poly_tensor_alu2(ctx, POLY_OP_CMPEQ, pad, poly_tensor_const_like_int(ctx, pad, 0));
  PolyTensor *valid3 = poly_tensor_reshape(ctx, valid, (int64_t[]){b, n, 1}, 3);
  pixels = model_gemma_scale(
      ctx,
      poly_tensor_alu2(ctx, POLY_OP_SUB, pixels, poly_tensor_const_like_float(ctx, pixels, .5)), 2
  );
  PolyTensor *x = model_gemma_linear(
      m, "vision_tower.", "patch_embedder.input_proj", pixels, 3 * patch * patch, d
  );
  PolyTensor *table = poly_model_param(
      m, "vision_tower.patch_embedder.position_embedding_table", POLY_FLOAT32,
      (int64_t[]){2, table_size, d}, 3
  );
  PolyTensor *xy[] = {px, py};
  PolyTensor *pos_embed[2];
  for (int axis = 0; axis < 2; axis++) {
    xy[axis] =
        poly_tensor_alu2(ctx, POLY_OP_MAX, xy[axis], poly_tensor_const_like_int(ctx, xy[axis], 0));
    PolyTensor *ids = poly_tensor_reshape(ctx, xy[axis], (int64_t[]){b * n}, 1);
    PolyTensor *t = model_vision_slice(ctx, table, 0, axis, axis + 1);
    t = poly_tensor_reshape(ctx, t, (int64_t[]){table_size, d}, 2);
    pos_embed[axis] =
        poly_tensor_reshape(ctx, poly_tensor_index_select(ctx, t, 0, ids), (int64_t[]){b, n, d}, 3);
  }
  PolyTensor *pos = poly_tensor_alu2(ctx, POLY_OP_ADD, pos_embed[0], pos_embed[1]);
  pos =
      poly_tensor_alu3(ctx, POLY_OP_WHERE, valid3, pos, poly_tensor_const_like_float(ctx, pos, 0));
  x = poly_tensor_alu2(ctx, POLY_OP_ADD, x, pos);
  if (hidden_states) hidden_states[0] = x;
  /* One frequency per quarter head. Tensor.rope rotates each of the two
   * spatial head slices independently after reshaping, matching axial RoPE. */
  float *freq = malloc((size_t)(hd / 4) * sizeof(float));
  if (!freq) return NULL;
  for (int i = 0; i < hd / 4; i++)
    freq[i] = (float)(1.0 / pow(theta, (double)i / (hd / 4)));
  PolyTensor *inv = poly_model_aux_from_host(
      m, "vision_rope.inv_freq", POLY_FLOAT32, (int64_t[]){hd / 4}, 1, freq,
      (size_t)(hd / 4) * sizeof(float)
  );
  free(freq);
  PolyTensor *angle = poly_tensor_cast(ctx, positions, POLY_FLOAT32);
  angle = poly_tensor_reshape(ctx, angle, (int64_t[]){b, 1, n, 2, 1}, 5);
  angle = poly_tensor_alu2(ctx, POLY_OP_MUL, angle, inv);
  PolyTensor *cos = poly_tensor_cos(ctx, angle), *sin = poly_tensor_alu1(ctx, POLY_OP_SIN, angle);
  PolyTensor *mask = poly_tensor_reshape(ctx, valid, (int64_t[]){b, 1, 1, 1, n}, 5);
  PolyTensor *mf = poly_tensor_cast(ctx, mask, POLY_FLOAT32);
  mask = poly_tensor_alu3(
      ctx, POLY_OP_WHERE, mask, poly_tensor_const_like_float(ctx, mf, 0),
      poly_tensor_const_like_float(ctx, mf, -3.4028234663852886e38)
  );
  ModelGemmaAttention attn = {b, n, d, heads, kv, hd, 2, eps, true};
  for (int i = 0; i < layers; i++) {
    char base[128];
    snprintf(base, sizeof(base), "vision_tower.encoder.layers.%d.", i);
    PolyTensor *h = model_gemma_norm(m, base, "input_layernorm", x, d, eps);
    h = model_gemma_attention(m, base, &attn, h, mask, cos, sin);
    x = poly_tensor_alu2(
        ctx, POLY_OP_ADD, x, model_gemma_norm(m, base, "post_attention_layernorm", h, d, eps)
    );
    h = model_gemma_norm(m, base, "pre_feedforward_layernorm", x, d, eps);
    PolyTensor *g =
        poly_tensor_gelu(ctx, model_gemma_linear(m, base, "mlp.gate_proj.linear", h, d, hidden));
    h = poly_tensor_alu2(
        ctx, POLY_OP_MUL, g, model_gemma_linear(m, base, "mlp.up_proj.linear", h, d, hidden)
    );
    h = model_gemma_linear(m, base, "mlp.down_proj.linear", h, hidden, d);
    x = poly_tensor_contiguous(
        ctx,
        poly_tensor_alu2(
            ctx, POLY_OP_ADD, x, model_gemma_norm(m, base, "post_feedforward_layernorm", h, d, eps)
        )
    );
    if (!x) return NULL;
    if (hidden_states) hidden_states[i + 1] = x;
  }
  x = poly_tensor_alu3(ctx, POLY_OP_WHERE, valid3, x, poly_tensor_const_like_float(ctx, x, 0));
  /* Processor positions, rather than storage order, determine pooling bins. */
  PolyTensor *width = poly_tensor_max(ctx, xy[0], (int64_t[]){1}, 1, true);
  width = poly_tensor_alu2(ctx, POLY_OP_ADD, width, poly_tensor_const_like_int(ctx, width, 1));
  width = poly_tensor_alu2(ctx, POLY_OP_IDIV, width, poly_tensor_const_like_int(ctx, width, pool));
  PolyTensor *gx =
      poly_tensor_alu2(ctx, POLY_OP_IDIV, xy[0], poly_tensor_const_like_int(ctx, xy[0], pool));
  PolyTensor *gy =
      poly_tensor_alu2(ctx, POLY_OP_IDIV, xy[1], poly_tensor_const_like_int(ctx, xy[1], pool));
  PolyTensor *bins =
      poly_tensor_alu2(ctx, POLY_OP_ADD, gx, poly_tensor_alu2(ctx, POLY_OP_MUL, gy, width));
  PolyTensor *weights = poly_tensor_cast(ctx, poly_tensor_one_hot(ctx, bins, out_n), POLY_FLOAT32);
  weights =
      poly_tensor_alu2(ctx, POLY_OP_MUL, weights, poly_tensor_cast(ctx, valid3, POLY_FLOAT32));
  PolyTensor *counts = poly_tensor_sum(ctx, weights, (int64_t[]){1}, 1, false);
  *valid_out =
      poly_tensor_alu2(ctx, POLY_OP_CMPLT, poly_tensor_const_like_float(ctx, counts, 0), counts);
  weights = model_gemma_scale(ctx, weights, 1.0 / (pool * pool));
  x = poly_tensor_dot(ctx, poly_tensor_permute(ctx, weights, (int64_t[]){0, 2, 1}, 3), x);
  x = model_gemma_scale(ctx, x, sqrt(d));
  if (standardize) {
    PolyTensor *bias_t =
        poly_model_param(m, "vision_tower.std_bias", POLY_FLOAT32, (int64_t[]){d}, 1);
    PolyTensor *scale_t =
        poly_model_param(m, "vision_tower.std_scale", POLY_FLOAT32, (int64_t[]){d}, 1);
    x = poly_tensor_alu2(ctx, POLY_OP_MUL, poly_tensor_alu2(ctx, POLY_OP_SUB, x, bias_t), scale_t);
  }
  return x;
}
