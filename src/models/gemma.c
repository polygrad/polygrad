#include "gemma.h"
#include <stdio.h>

bool model_gemma_integer(
    const cJSON *j,
    const char *key,
    int fallback,
    int lo,
    int hi,
    int *out,
    PolyModelError *err
) {
  if (!model_config_integer(j, key, lo, hi, false, err)) return false;
  const cJSON *v = cJSON_GetObjectItemCaseSensitive(j, key);
  *out = v ? v->valueint : fallback;
  return true;
}

PolyTensor *model_gemma_scale(PolyCtx *ctx, PolyTensor *x, double factor) {
  return poly_tensor_alu2(ctx, POLY_OP_MUL, x, poly_tensor_const_like_float(ctx, x, factor));
}

PolyTensor *model_gemma_norm(
    PolyModel *m,
    const char *base,
    const char *suffix,
    PolyTensor *x,
    int width,
    double eps
) {
  char name[192];
  snprintf(name, sizeof(name), "%s%s", base, suffix);
  return poly_model_rmsnorm(m, name, x, width, eps);
}

PolyTensor *model_gemma_linear(
    PolyModel *m,
    const char *base,
    const char *suffix,
    PolyTensor *x,
    int in,
    int out
) {
  char name[192];
  snprintf(name, sizeof(name), "%s%s", base, suffix);
  return poly_tensor_contiguous(poly_model_ctx(m), poly_model_linear(m, name, x, in, out, false));
}

PolyTensor *model_gemma_clippable_linear(
    PolyModel *m,
    const char *base,
    const char *suffix,
    PolyTensor *x,
    int in,
    int out,
    bool clipped
) {
  PolyCtx *ctx = poly_model_ctx(m);
  char name[192];
  for (int stage = 0; stage < 2; stage++) {
    if (stage) {
      snprintf(name, sizeof(name), "%s%s.linear", base, suffix);
      x = poly_model_linear(m, name, x, in, out, false);
    }
    if (clipped) {
      /* Checkpoint buffers are scalars, not learned vectors. Keep their rank
       * for strict safetensors binding and portable Model round trips. */
      snprintf(name, sizeof(name), "%s%s.%s_min", base, suffix, stage ? "output" : "input");
      PolyTensor *lo = poly_model_param(m, name, POLY_FLOAT32, NULL, 0);
      snprintf(name, sizeof(name), "%s%s.%s_max", base, suffix, stage ? "output" : "input");
      PolyTensor *hi = poly_model_param(m, name, POLY_FLOAT32, NULL, 0);
      x = poly_tensor_minimum(ctx, poly_tensor_alu2(ctx, POLY_OP_MAX, x, lo), hi);
    }
  }
  return poly_tensor_contiguous(ctx, x);
}

PolyTensor *model_gemma_attention(
    PolyModel *m,
    const char *base,
    const ModelGemmaAttention *c,
    PolyTensor *x,
    PolyTensor *mask,
    PolyTensor *cos,
    PolyTensor *sin
) {
  PolyCtx *ctx = poly_model_ctx(m);
  int b = c->batch, n = c->length, hd = c->head_dim, kv = c->kv_heads, groups = c->heads / kv;
  char name[192];
  const char *proj[] = {"q_proj", "k_proj", "v_proj"};
  PolyTensor *qkv[3];
  for (int k = 0; k < 3; k++) {
    int heads = k ? kv : c->heads;
    snprintf(
        name, sizeof(name), "%sself_attn.%s%s", base, proj[k], c->nested_linear ? ".linear" : ""
    );
    PolyTensor *v =
        poly_tensor_contiguous(ctx, poly_model_linear(m, name, x, c->dim, heads * hd, false));
    v = poly_tensor_reshape(ctx, v, (int64_t[]){b, n, heads, hd}, 4);
    v = poly_tensor_permute(ctx, v, (int64_t[]){0, 2, 1, 3}, 4);
    if (k == 2)
      v = poly_tensor_rmsnorm_apply(ctx, v, NULL, c->eps);
    else {
      snprintf(name, sizeof(name), "%sself_attn.%s_norm", base, k ? "k" : "q");
      v = poly_model_rmsnorm(m, name, v, hd, c->eps);
      if (c->rope_axes > 1)
        v = poly_tensor_reshape(
            ctx, v, (int64_t[]){b, heads, n, c->rope_axes, hd / c->rope_axes}, 5
        );
      v = poly_tensor_rope(ctx, v, cos, sin);
    }
    qkv[k] = poly_tensor_reshape(ctx, v, (int64_t[]){b, kv, k ? 1 : groups, n, hd}, 5);
  }
  /* Gemma uses scale=1 after Q/K normalization. Broadcast KV groups without
   * repeated storage; SDPA's default sqrt(head_dim) scaling is not applicable. */
  PolyTensor *key = poly_tensor_permute(ctx, qkv[1], (int64_t[]){0, 1, 2, 4, 3}, 5);
  PolyTensor *scores = poly_tensor_dot(ctx, qkv[0], key);
  scores = poly_tensor_alu2(ctx, POLY_OP_ADD, scores, mask);
  PolyTensor *h = poly_tensor_dot(ctx, poly_tensor_softmax(ctx, scores, -1), qkv[2]);
  h = poly_tensor_reshape(ctx, h, (int64_t[]){b, c->heads, n, hd}, 4);
  h = poly_tensor_permute(ctx, h, (int64_t[]){0, 2, 1, 3}, 4);
  h = poly_tensor_reshape(ctx, h, (int64_t[]){b, n, c->heads * hd}, 3);
  snprintf(name, sizeof(name), "%sself_attn.o_proj%s", base, c->nested_linear ? ".linear" : "");
  return poly_tensor_contiguous(ctx, poly_model_linear(m, name, h, c->heads * hd, c->dim, false));
}
