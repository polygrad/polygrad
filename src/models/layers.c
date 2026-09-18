/* Model-owned layer construction. Reuses nn programs; owns no runtime.
 * Deprecated ctx-registry constructors remain only for the existing C ABI. */
#include "layers.h"
#include "../device.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "factory.h"

PolyTensor *model_activation(PolyCtx *ctx, PolyTensor *x, const char *name) {
  if (!name || !strcmp(name, "none")) return x;
  if (!strcmp(name, "relu")) return poly_tensor_relu(ctx, x);
  if (!strcmp(name, "gelu")) return poly_tensor_gelu(ctx, x);
  if (!strcmp(name, "silu")) return poly_tensor_silu(ctx, x);
  if (!strcmp(name, "sigmoid")) return poly_tensor_sigmoid(ctx, x);
  if (!strcmp(name, "tanh")) return poly_tensor_tanh(ctx, x);
  return NULL;
}

/* Stateless PRNG (SplitMix64) */

static uint64_t splitmix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}

static float prng_float(uint64_t seed, uint64_t stream, uint64_t idx) {
  uint64_t r = splitmix64(seed ^ splitmix64(stream) ^ splitmix64(idx));
  return (float)(r >> 40) * 0x1.0p-24f;
}

static uint64_t fnv1a_64(const char *s, size_t len) {
  uint64_t h = 0xcbf29ce484222325ULL;
  for (size_t i = 0; i < len; i++)
    h = (h ^ (uint8_t)s[i]) * 0x100000001b3ULL;
  return h;
}

void poly_init_param_kaiming(
    uint64_t seed,
    const char *name,
    float *data,
    int64_t numel,
    int64_t fan_in
) {
  uint64_t stream = fnv1a_64(name, strlen(name));
  float bound = sqrtf(6.0f / (float)fan_in);
  for (int64_t i = 0; i < numel; i++)
    data[i] = (prng_float(seed, stream, (uint64_t)i) * 2.0f - 1.0f) * bound;
}

int model_init_param(
    PolyModel *model,
    const char *name,
    uint64_t seed,
    int64_t fan_in,
    float fill
) {
  int64_t n = poly_model_buf_numel_named(model, name);
  if (n <= 0 || (uint64_t)n > SIZE_MAX / sizeof(float) || fan_in < 0) return -1;
  float *values = malloc((size_t)n * sizeof(float));
  if (!values) return -1;
  if (fan_in)
    poly_init_param_kaiming(seed, name, values, n, fan_in);
  else
    for (int64_t i = 0; i < n; i++)
      values[i] = fill;
  int rc = poly_model_write_buf_named(model, name, values, (size_t)n * sizeof(float));
  free(values);
  return rc;
}

PolyTensor *poly_model_rope_frequencies(
    PolyModel *model,
    const char *name,
    const PolyModelRoPEConfig *c,
    bool sine
) {
  if (!model || !name || !c || c->length <= 0 || c->dim <= 0 || c->dim % 2 || !isfinite(c->theta) ||
      c->theta < 1 || !isfinite(c->factor) || c->factor < 1 ||
      (c->factor != 1 &&
       (!isfinite(c->low_freq) || !isfinite(c->high_freq) || !isfinite(c->original_context) ||
        c->low_freq <= 0 || c->high_freq <= c->low_freq || c->original_context <= 0)))
    return NULL;
  int half = c->dim / 2;
  if ((size_t)c->length > SIZE_MAX / (size_t)half) return NULL;
  size_t count = (size_t)c->length * (size_t)half;
  if (count > SIZE_MAX / sizeof(float)) return NULL;
  float *data = malloc(count * sizeof(float));
  if (!data) return NULL;
  /* extra/models/llama.py:precompute_freqs_cis, fixed positions starting at 0.
   * The owned AUX snapshot makes export independent of this temporary array. */
  for (int t = 0; t < c->length; t++)
    for (int j = 0; j < half; j++) {
      double freq = 1.0 / pow(c->theta, (double)j / half);
      /* Llama 3.1/3.2's wavelength scaling (Meta apply_scaling and HF
       * _compute_llama3_parameters). Model-owned extension: the pinned
       * extra/models/llama.py has only the unscaled frequency constructor. */
      if (c->factor != 1) {
        double wavelength = 6.2831853071795864769 / freq;
        if (wavelength > c->original_context / c->low_freq)
          freq /= c->factor;
        else if (wavelength >= c->original_context / c->high_freq) {
          double smooth =
              (c->original_context / wavelength - c->low_freq) / (c->high_freq - c->low_freq);
          freq *= (1 - smooth) / c->factor + smooth;
        }
      }
      float angle = (float)((double)t * freq);
      data[(size_t)t * half + j] = sine ? sinf(angle) : cosf(angle);
    }
  PolyCtx *ctx = poly_model_ctx(model);
  PolyDevice device = poly_ctx_get_preferred_device(ctx);
  PolyTensor *v =
      poly_tensor_empty(ctx, POLY_FLOAT32, (int64_t[]){1, 1, c->length, half}, 4, device);
  PolyUOp *buffer = v ? (PolyUOp *)poly_uop_get_buffer_identity(poly_tensor_uop_physical(v)) : NULL;
  /* from_host borrows its input. Initialize owned device storage instead so
   * freeing this temporary cannot leave a pending COPY reading released bytes. */
  bool copied = buffer && poly_buffer_ensure_device_allocated(ctx, buffer, device) == 0 &&
                poly_buffer_write(ctx, buffer, data, count * sizeof(float)) == 0;
  free(data);
  if (!copied || poly_model_aux(model, name, v, 0) != POLY_STATUS_OK) {
    poly_tensor_release(v);
    return NULL;
  }
  return v;
}

int poly_model_lstm_cell(
    PolyModel *model,
    const char *prefix,
    PolyTensor *x,
    PolyTensor *h,
    PolyTensor *c,
    int input_size,
    int hidden_size,
    bool bias,
    PolyTensor **new_h,
    PolyTensor **new_c
) {
  if (!new_h || !new_c || new_h == new_c) return -1;
  *new_h = *new_c = NULL;
  if (!model || !x || input_size <= 0 || hidden_size <= 0) return -1;
  PolyCtx *ctx = poly_model_ctx(model);
  bool scoped = prefix && prefix[0];
  if (!ctx || (scoped && poly_model_scope_push(model, "%s", prefix) != POLY_STATUS_OK)) return -1;
  int64_t gates = (int64_t)hidden_size * 4;
  PolyTensor *wi =
      poly_model_param(model, "weight_ih", POLY_FLOAT32, (int64_t[]){gates, input_size}, 2);
  PolyTensor *wh =
      wi ? poly_model_param(model, "weight_hh", POLY_FLOAT32, (int64_t[]){gates, hidden_size}, 2)
         : NULL;
  PolyTensor *bi = bias && wh ? poly_model_param(model, "bias_ih", POLY_FLOAT32, &gates, 1) : NULL;
  PolyTensor *bh = bi ? poly_model_param(model, "bias_hh", POLY_FLOAT32, &gates, 1) : NULL;
  if (scoped && poly_model_scope_pop(model) != POLY_STATUS_OK) return -1;
  if (!wi || !wh || (bias && (!bi || !bh))) return -1;
  return poly_tensor_lstm_cell(ctx, x, h, c, wi, wh, bi, bh, new_h, new_c);
}

PolyUOp *poly_linear(
    PolyCtx *ctx,
    const char *prefix,
    PolyUOp *x,
    int in_features,
    int out_features,
    bool use_bias
) {
  int64_t ws[] = {out_features, in_features};
  PolyUOp *w = poly_param(ctx, POLY_FLOAT32, ws, 2, "%s.weight", prefix);
  if (!w) return NULL;
  w = poly_reshape(ctx, w, ws, 2);

  PolyUOp *b = NULL;
  if (use_bias) {
    int64_t bs[] = {out_features};
    b = poly_param(ctx, POLY_FLOAT32, bs, 1, "%s.bias", prefix);
    if (!b) return NULL;
  }

  return poly_linear_apply(ctx, x, w, b);
}

PolyUOp *poly_layernorm(PolyCtx *ctx, const char *prefix, PolyUOp *x, int dim, double eps) {
  int64_t ds[] = {dim};
  PolyUOp *w = poly_param(ctx, POLY_FLOAT32, ds, 1, "%s.weight", prefix);
  PolyUOp *b = poly_param(ctx, POLY_FLOAT32, ds, 1, "%s.bias", prefix);
  if (!w || !b) return NULL;

  return poly_layernorm_apply(
      ctx, x, poly_reshape(ctx, w, ds, 1), poly_reshape(ctx, b, ds, 1), -1, eps
  );
}

PolyUOp *poly_rmsnorm(PolyCtx *ctx, const char *prefix, PolyUOp *x, int dim, double eps) {
  int64_t ds[] = {dim};
  PolyUOp *w = poly_param(ctx, POLY_FLOAT32, ds, 1, "%s.weight", prefix);
  if (!w) return NULL;
  return poly_rmsnorm_apply(ctx, x, poly_reshape(ctx, w, ds, 1), eps);
}

PolyUOp *poly_embedding(
    PolyCtx *ctx,
    const char *prefix,
    PolyUOp *tokens,
    int vocab_size,
    int embed_dim
) {
  int64_t ws[] = {vocab_size, embed_dim};
  PolyUOp *w = poly_param(ctx, POLY_FLOAT32, ws, 2, "%s.weight", prefix);
  if (!w) return NULL;
  return poly_embedding_apply(ctx, tokens, poly_reshape(ctx, w, ws, 2));
}

static int model_parameters(
    PolyModel *inst,
    const char *prefix,
    const int64_t *shape,
    int ndim,
    int64_t bias_dim,
    PolyTensor **weight,
    PolyTensor **bias
) {
  if (!weight || !bias || weight == bias) return -1;
  *weight = *bias = NULL;
  if (!inst) return -1;
  PolyCtx *ctx = poly_model_ctx(inst);
  if (!ctx) return -1;
  bool scoped = prefix && prefix[0];
  if (scoped && poly_model_scope_push(inst, "%s", prefix) != POLY_STATUS_OK) return -1;

  PolyTensor *w = poly_model_param(inst, "weight", POLY_FLOAT32, shape, ndim);
  PolyTensor *b = NULL;
  if (w && bias_dim) {
    b = poly_model_param(inst, "bias", POLY_FLOAT32, &bias_dim, 1);
  }

  bool popped = !scoped || poly_model_scope_pop(inst) == POLY_STATUS_OK;
  if (!w || (bias_dim && !b) || !popped) {
    poly_tensor_release(w);
    poly_tensor_release(b);
    return -1;
  }
  *weight = w;
  *bias = b;
  return 0;
}

int poly_model_linear_parameters(
    PolyModel *model,
    const char *prefix,
    int in_features,
    int out_features,
    bool use_bias,
    PolyTensor **weight,
    PolyTensor **bias
) {
  if (in_features <= 0 || out_features <= 0) return -1;
  return model_parameters(
      model, prefix, (int64_t[]){out_features, in_features}, 2, use_bias ? out_features : 0, weight,
      bias
  );
}

int poly_model_norm_parameters(
    PolyModel *model,
    const char *prefix,
    int dim,
    bool use_bias,
    PolyTensor **weight,
    PolyTensor **bias
) {
  if (dim <= 0) return -1;
  return model_parameters(model, prefix, (int64_t[]){dim}, 1, use_bias ? dim : 0, weight, bias);
}

PolyTensor *poly_model_embedding_parameters(
    PolyModel *model,
    const char *prefix,
    int vocab_size,
    int embed_dim
) {
  PolyTensor *w = NULL, *b = NULL;
  if (vocab_size <= 0 || embed_dim <= 0 ||
      model_parameters(model, prefix, (int64_t[]){vocab_size, embed_dim}, 2, 0, &w, &b))
    return NULL;
  return w;
}

PolyTensor *poly_model_linear(
    PolyModel *inst,
    const char *prefix,
    PolyTensor *x,
    int in_features,
    int out_features,
    bool use_bias
) {
  PolyTensor *w, *b;
  if (!x || poly_model_linear_parameters(inst, prefix, in_features, out_features, use_bias, &w, &b))
    return NULL;
  PolyTensor *out = poly_tensor_linear_apply(poly_model_ctx(inst), x, w, b);
  poly_tensor_release(w);
  poly_tensor_release(b);
  return out;
}

PolyTensor *poly_model_layernorm(
    PolyModel *inst,
    const char *prefix,
    PolyTensor *x,
    int dim,
    double eps
) {
  PolyTensor *w, *b;
  if (!x || poly_model_norm_parameters(inst, prefix, dim, true, &w, &b)) return NULL;
  PolyTensor *out = poly_tensor_layernorm_apply(poly_model_ctx(inst), x, w, b, -1, eps);
  poly_tensor_release(w);
  poly_tensor_release(b);
  return out;
}

PolyTensor *poly_model_rmsnorm(
    PolyModel *inst,
    const char *prefix,
    PolyTensor *x,
    int dim,
    double eps
) {
  PolyTensor *w, *b;
  if (!x || poly_model_norm_parameters(inst, prefix, dim, false, &w, &b)) return NULL;
  PolyTensor *out = poly_tensor_rmsnorm_apply(poly_model_ctx(inst), x, w, eps);
  poly_tensor_release(w);
  return out;
}

PolyTensor *poly_model_embedding(
    PolyModel *inst,
    const char *prefix,
    PolyTensor *tokens,
    int vocab_size,
    int embed_dim
) {
  PolyTensor *w =
      tokens ? poly_model_embedding_parameters(inst, prefix, vocab_size, embed_dim) : NULL;
  if (!w) return NULL;
  PolyTensor *out = poly_tensor_embedding_apply(poly_model_ctx(inst), tokens, w);
  poly_tensor_release(w);
  return out;
}
