/* Gemma4 Conformer audio tower. The caller supplies processor log-mel features;
 * this graph owns subsampling, blocked attention and causal depthwise convolution.
 * Fixed capacities retain padded tokens plus validity, avoiding host compaction. */
#include "gemma.h"
#include <math.h>
#include <stdio.h>

typedef struct {
  int b, n, d, heads, layers, chunk, left, right, kernel, channels[2], output;
  double eps, clip, residual, cap, invalid;
  bool clipped;
} AudioConfig;

static bool number(
    const cJSON *j,
    const char *key,
    double fallback,
    double lo,
    double hi,
    double *out,
    PolyModelError *err
) {
  const cJSON *v = cJSON_GetObjectItemCaseSensitive(j, key);
  *out = v ? v->valuedouble : fallback;
  return ((!v || cJSON_IsNumber(v)) && isfinite(*out) && *out >= lo && *out <= hi) ||
         model_factory_error(err, key, "invalid audio config value");
}
static PolyTensor *clip(PolyCtx *ctx, PolyTensor *x, double lo, double hi) {
  return poly_tensor_minimum(
      ctx, poly_tensor_alu2(ctx, POLY_OP_MAX, x, poly_tensor_const_like_float(ctx, x, lo)),
      poly_tensor_const_like_float(ctx, x, hi)
  );
}
static PolyTensor *slice(PolyCtx *ctx, PolyTensor *x, int axis, int start, int end) {
  PolyUOp *u = poly_tensor_uop_physical(x);
  int n = poly_uop_ndim(ctx, u);
  const int64_t *dims = poly_uop_max_shape_dims(ctx, u);
  if (!dims || n < 1 || n > POLY_MAX_DIMS) return NULL;
  int64_t ranges[POLY_MAX_DIMS][2];
  for (int i = 0; i < n; i++) {
    ranges[i][0] = 0;
    ranges[i][1] = dims[i];
  }
  ranges[axis][0] = start;
  ranges[axis][1] = end;
  return poly_tensor_shrink(ctx, x, ranges, n);
}
static PolyTensor *feedforward(
    PolyModel *m,
    const AudioConfig *c,
    const char *base,
    PolyTensor *x
) {
  PolyCtx *ctx = poly_model_ctx(m);
  PolyTensor *h = clip(ctx, x, -c->clip, c->clip);
  /* HF's feedforward and attention RMS norms use their default epsilon,
   * whereas the convolution norms use the configurable epsilon. */
  h = model_gemma_norm(m, base, "pre_layer_norm", h, c->d, 1e-6);
  h = poly_tensor_silu(
      ctx, model_gemma_clippable_linear(m, base, "ffw_layer_1", h, c->d, 4 * c->d, c->clipped)
  );
  h = model_gemma_clippable_linear(m, base, "ffw_layer_2", h, 4 * c->d, c->d, c->clipped);
  h = model_gemma_norm(m, base, "post_layer_norm", clip(ctx, h, -c->clip, c->clip), c->d, 1e-6);
  return poly_tensor_alu2(ctx, POLY_OP_ADD, x, model_gemma_scale(ctx, h, c->residual));
}

static PolyTensor *context(PolyCtx *ctx, const AudioConfig *c, PolyTensor *x) {
  int width = c->chunk + c->left + c->right;
  x = poly_tensor_pad_value_float(
      ctx, x, (int64_t[][2]){{0, 0}, {c->left, c->right + c->chunk - 1}, {0, 0}, {0, 0}}, 4, 0
  );
  x = poly_tensor_unfold(ctx, x, 1, width, c->chunk);
  return poly_tensor_contiguous(ctx, poly_tensor_permute(ctx, x, (int64_t[]){0, 2, 1, 4, 3}, 5));
}

static PolyTensor *attention(
    PolyModel *m,
    const AudioConfig *c,
    const char *base,
    PolyTensor *x,
    PolyTensor *pos,
    PolyTensor *mask
) {
  PolyCtx *ctx = poly_model_ctx(m);
  int b = c->b, n = c->n, h = c->heads, hd = c->d / h, k = c->chunk, blocks = (n + k - 1) / k;
  int width = k + c->left + c->right, plen = width / 2 + 1;
  PolyTensor *v[3];
  const char *names[] = {"q_proj", "k_proj", "v_proj"};
  for (int i = 0; i < 3; i++) {
    v[i] = model_gemma_clippable_linear(m, base, names[i], x, c->d, c->d, c->clipped);
    v[i] = poly_tensor_reshape(ctx, v[i], (int64_t[]){b, n, h, hd}, 4);
  }
  char name[192];
  snprintf(name, sizeof(name), "%sper_dim_scale", base);
  PolyTensor *s =
      poly_tensor_softplus(ctx, poly_model_param(m, name, POLY_FLOAT32, (int64_t[]){hd}, 1), 1);
  v[0] = poly_tensor_alu2(ctx, POLY_OP_MUL, model_gemma_scale(ctx, v[0], 1 / sqrt(hd) / log(2)), s);
  v[1] = model_gemma_scale(ctx, v[1], log(1 + exp(1)) / log(2));
  v[0] = poly_tensor_pad_value_float(
      ctx, v[0], (int64_t[][2]){{0, 0}, {0, blocks * k - n}, {0, 0}, {0, 0}}, 4, 0
  );
  v[0] = poly_tensor_reshape(ctx, v[0], (int64_t[]){b, blocks, k, h, hd}, 5);
  PolyTensor *q = poly_tensor_permute(ctx, v[0], (int64_t[]){0, 3, 1, 2, 4}, 5);
  PolyTensor *keys = context(ctx, c, v[1]), *values = context(ctx, c, v[2]);
  PolyTensor *ac =
      poly_tensor_dot(ctx, q, poly_tensor_permute(ctx, keys, (int64_t[]){0, 1, 2, 4, 3}, 5));
  snprintf(name, sizeof(name), "%srelative_k_proj", base);
  PolyTensor *r = poly_model_linear(m, name, pos, c->d, c->d, false);
  r = poly_tensor_reshape(ctx, r, (int64_t[]){plen, h, hd}, 3);
  r = poly_tensor_permute(ctx, r, (int64_t[]){1, 2, 0}, 3);
  PolyTensor *bd =
      poly_tensor_dot(ctx, poly_tensor_reshape(ctx, q, (int64_t[]){b, h, blocks * k, hd}, 4), r);
  bd = poly_tensor_reshape(ctx, bd, (int64_t[]){b, h, blocks, k, plen}, 5);
  /* Relative shift is a reshape of zero-padded rows, not a cyclic roll. */
  bd = poly_tensor_pad_value_float(
      ctx, bd, (int64_t[][2]){{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, width + 1 - plen}}, 5, 0
  );
  bd = poly_tensor_reshape(ctx, bd, (int64_t[]){b, h, blocks, k * (width + 1)}, 4);
  bd = slice(ctx, bd, 3, 0, k * width);
  bd = poly_tensor_reshape(ctx, bd, (int64_t[]){b, h, blocks, k, width}, 5);
  PolyTensor *a = model_gemma_scale(
      ctx,
      poly_tensor_tanh(
          ctx, model_gemma_scale(ctx, poly_tensor_alu2(ctx, POLY_OP_ADD, ac, bd), 1 / c->cap)
      ),
      c->cap
  );
  a = poly_tensor_alu3(
      ctx, POLY_OP_WHERE, mask, a, poly_tensor_const_like_float(ctx, a, c->invalid)
  );
  a = poly_tensor_softmax(ctx, a, -1);
  a = poly_tensor_dot(ctx, a, values);
  a = poly_tensor_permute(ctx, a, (int64_t[]){0, 2, 3, 1, 4}, 5);
  a = poly_tensor_reshape(ctx, a, (int64_t[]){b, blocks * k, c->d}, 3);
  return model_gemma_clippable_linear(
      m, base, "post", slice(ctx, a, 1, 0, n), c->d, c->d, c->clipped
  );
}

PolyTensor *model_gemma4_audio(
    PolyModel *m,
    const cJSON *j,
    int b,
    int frames,
    PolyTensor *features,
    PolyTensor *valid,
    PolyTensor **valid_out,
    PolyTensor **states,
    PolyModelError *err
) {
  PolyCtx *ctx = poly_model_ctx(m);
  AudioConfig c = {.b = b};
  if (!model_config_choice(j, "model_type", "|gemma4_audio|", err) ||
      !model_config_choice(j, "hidden_act", "|silu|", err) ||
      !model_gemma_integer(j, "hidden_size", 1024, 2, 65536, &c.d, err) ||
      !model_gemma_integer(j, "num_hidden_layers", 12, 1, 256, &c.layers, err) ||
      !model_gemma_integer(j, "num_attention_heads", 8, 1, 256, &c.heads, err) ||
      !model_gemma_integer(j, "attention_chunk_size", 12, 1, 8192, &c.chunk, err) ||
      !model_gemma_integer(j, "attention_context_left", 13, 1, 8192, &c.left, err) ||
      !model_gemma_integer(j, "attention_context_right", 0, 0, 8192, &c.right, err) ||
      !model_gemma_integer(j, "conv_kernel_size", 5, 1, 1024, &c.kernel, err) ||
      !model_gemma_integer(j, "output_proj_dims", 1536, 1, 65536, &c.output, err) ||
      !number(j, "rms_norm_eps", 1e-6, 1e-15, 1, &c.eps, err) ||
      !number(j, "gradient_clipping", 1e10, 1e-15, 3.4028234663852886e38, &c.clip, err) ||
      !number(j, "residual_weight", .5, 0, 1e6, &c.residual, err) ||
      !number(j, "attention_logit_cap", 50, 1e-15, 1e6, &c.cap, err) ||
      !number(
          j, "attention_invalid_logits_value", -1e9, -3.4028234663852886e38, -1, &c.invalid, err
      ))
    return NULL;
  const cJSON *clipped = cJSON_GetObjectItemCaseSensitive(j, "use_clipped_linears");
  const cJSON *channels = cJSON_GetObjectItemCaseSensitive(j, "subsampling_conv_channels");
  if ((clipped && !cJSON_IsBool(clipped)) || !cJSON_IsArray(channels) ||
      cJSON_GetArraySize(channels) != 2 || c.d % 2 || c.d % c.heads) {
    model_factory_error(err, "audio_config", "invalid channels, heads or clipping flag");
    return NULL;
  }
  c.clipped = !clipped || cJSON_IsTrue(clipped);
  for (int i = 0; i < 2; i++) {
    const cJSON *v = cJSON_GetArrayItem(channels, i);
    if (!cJSON_IsNumber(v) || v->valuedouble != v->valueint || v->valueint < 1 ||
        v->valueint > 4096) {
      model_factory_error(err, "subsampling_conv_channels", "expected two positive channel counts");
      return NULL;
    }
    c.channels[i] = v->valueint;
  }
  if (c.channels[0] % 4) {
    model_factory_error(err, "subsampling_conv_channels", "first count must be a multiple of four");
    return NULL;
  }
  c.left--;
  int n = frames, freq = c.channels[0], in = 1;
  PolyTensor *x = poly_tensor_reshape(ctx, features, (int64_t[]){b, 1, n, freq}, 4);
  char name[192];
  for (int i = 0; i < 2; i++) {
    x = poly_tensor_alu2(
        ctx, POLY_OP_MUL, x,
        poly_tensor_reshape(
            ctx, poly_tensor_cast(ctx, valid, POLY_FLOAT32), (int64_t[]){b, 1, n, 1}, 4
        )
    );
    snprintf(name, sizeof(name), "audio_tower.subsample_conv_projection.layer%d.conv.weight", i);
    PolyTensor *w =
        poly_model_param(m, name, POLY_FLOAT32, (int64_t[]){c.channels[i], in, 3, 3}, 4);
    x = poly_tensor_conv2d(
        ctx, x, w, NULL, 1, (int64_t[]){2, 2}, (int64_t[]){1, 1}, (int64_t[]){1, 1}, 2
    );
    x = poly_tensor_permute(ctx, x, (int64_t[]){0, 2, 3, 1}, 4);
    snprintf(name, sizeof(name), "audio_tower.subsample_conv_projection.layer%d.norm.weight", i);
    w = poly_model_param(m, name, POLY_FLOAT32, (int64_t[]){c.channels[i]}, 1);
    /* This checkpoint has a LayerNorm scale but no bias. The shared affine
     * LayerNorm API binds both; compose the scale with its non-affine form. */
    x = poly_tensor_relu(
        ctx, poly_tensor_alu2(
                 ctx, POLY_OP_MUL, poly_tensor_layernorm_apply(ctx, x, NULL, NULL, -1, c.eps), w
             )
    );
    x = poly_tensor_contiguous(ctx, poly_tensor_permute(ctx, x, (int64_t[]){0, 3, 1, 2}, 4));
    PolyTensor *idx =
        poly_tensor_arange_int(ctx, 0, n, 2, POLY_INT32, poly_ctx_get_preferred_device(ctx));
    valid = poly_tensor_index_select(ctx, valid, 1, idx);
    n = (n + 1) / 2;
    freq = (freq + 1) / 2;
    in = c.channels[i];
  }
  c.n = n;
  x = poly_tensor_reshape(
      ctx, poly_tensor_permute(ctx, x, (int64_t[]){0, 2, 3, 1}, 4), (int64_t[]){b, n, freq * in}, 3
  );
  x = poly_model_linear(
      m, "audio_tower.subsample_conv_projection.input_proj_linear", x, freq * in, c.d, false
  );
  if (!x) return NULL;
  if (states) states[0] = x;
  *valid_out = poly_tensor_cast(ctx, valid, POLY_BOOL);
  int width = c.chunk + c.left + c.right, blocks = (n + c.chunk - 1) / c.chunk,
      plen = width / 2 + 1;
  PolyTensor *times =
      poly_tensor_arange_int(ctx, 0, c.d / 2, 1, POLY_INT32, poly_ctx_get_preferred_device(ctx));
  times = poly_tensor_exp(
      ctx, model_gemma_scale(
               ctx, poly_tensor_cast(ctx, times, POLY_FLOAT32), -log(10000) / fmax(c.d / 2 - 1, 1)
           )
  );
  PolyTensor *positions =
      poly_tensor_arange_int(ctx, plen - 1, -1, -1, POLY_INT32, poly_ctx_get_preferred_device(ctx));
  positions = poly_tensor_reshape(
      ctx, poly_tensor_cast(ctx, positions, POLY_FLOAT32), (int64_t[]){1, plen, 1}, 3
  );
  positions = poly_tensor_alu2(ctx, POLY_OP_MUL, positions, times);
  PolyTensor *pos = poly_tensor_cat(
      ctx,
      (PolyTensor *[]
      ){poly_tensor_alu1(ctx, POLY_OP_SIN, positions), poly_tensor_cos(ctx, positions)},
      2, -1
  );
  /* Key validity and local horizon are independent of query padding. Padded
   * block queries are masked only to match the reference's 4D-to-5D conversion. */
  PolyTensor *vm = poly_tensor_pad_value_int(
      ctx, valid, (int64_t[][2]){{0, 0}, {c.left, c.right + c.chunk - 1}}, 2, 0
  );
  vm = poly_tensor_unfold(ctx, vm, 1, width, c.chunk);
  vm = poly_tensor_reshape(ctx, vm, (int64_t[]){b, 1, blocks, 1, width}, 5);
  vm = poly_tensor_alu2(ctx, POLY_OP_CMPNE, vm, poly_tensor_const_like_int(ctx, vm, 0));
  PolyTensor *q = poly_tensor_arange_int(
      ctx, 0, blocks * c.chunk, 1, POLY_INT32, poly_ctx_get_preferred_device(ctx)
  );
  q = poly_tensor_reshape(ctx, q, (int64_t[]){1, 1, blocks, c.chunk, 1}, 5);
  PolyTensor *start = poly_tensor_arange_int(
      ctx, 0, blocks * c.chunk, c.chunk, POLY_INT32, poly_ctx_get_preferred_device(ctx)
  );
  start = poly_tensor_reshape(ctx, start, (int64_t[]){1, 1, blocks, 1, 1}, 5);
  PolyTensor *ki = poly_tensor_arange_int(
      ctx, -c.left, c.chunk + c.right, 1, POLY_INT32, poly_ctx_get_preferred_device(ctx)
  );
  ki = poly_tensor_alu2(ctx, POLY_OP_ADD, start, ki);
  PolyTensor *delta = poly_tensor_alu2(ctx, POLY_OP_SUB, ki, q);
  PolyTensor *mask = poly_tensor_alu2(
      ctx, POLY_OP_AND, vm,
      poly_tensor_alu2(ctx, POLY_OP_CMPLT, q, poly_tensor_const_like_int(ctx, q, n))
  );
  PolyTensor *past = poly_tensor_alu2(
      ctx, POLY_OP_AND,
      poly_tensor_alu2(ctx, POLY_OP_CMPLT, poly_tensor_const_like_int(ctx, delta, -c.left), delta),
      poly_tensor_alu2(ctx, POLY_OP_CMPLT, delta, poly_tensor_const_like_int(ctx, delta, 1))
  );
  PolyTensor *future = poly_tensor_alu2(
      ctx, POLY_OP_AND,
      poly_tensor_alu2(ctx, POLY_OP_CMPLT, poly_tensor_const_like_int(ctx, delta, 0), delta),
      poly_tensor_alu2(ctx, POLY_OP_CMPLT, delta, poly_tensor_const_like_int(ctx, delta, c.right))
  );
  mask = poly_tensor_alu2(ctx, POLY_OP_AND, mask, poly_tensor_alu2(ctx, POLY_OP_OR, past, future));
  for (int i = 0; i < c.layers; i++) {
    char base[128], sub[160];
    snprintf(base, sizeof(base), "audio_tower.layers.%d.", i);
    snprintf(sub, sizeof(sub), "%sfeed_forward1.", base);
    x = feedforward(m, &c, sub, x);
    PolyTensor *h =
        model_gemma_norm(m, base, "norm_pre_attn", clip(ctx, x, -c.clip, c.clip), c.d, 1e-6);
    snprintf(sub, sizeof(sub), "%sself_attn.", base);
    h = attention(m, &c, sub, h, pos, mask);
    h = model_gemma_norm(m, base, "norm_post_attn", clip(ctx, h, -c.clip, c.clip), c.d, 1e-6);
    x = poly_tensor_alu2(ctx, POLY_OP_ADD, x, h);
    snprintf(sub, sizeof(sub), "%slconv1d.", base);
    h = model_gemma_norm(m, sub, "pre_layer_norm", x, c.d, c.eps);
    h = model_gemma_clippable_linear(m, sub, "linear_start", h, c.d, 2 * c.d, c.clipped);
    h = poly_tensor_alu2(
        ctx, POLY_OP_MUL, slice(ctx, h, 2, 0, c.d),
        poly_tensor_sigmoid(ctx, slice(ctx, h, 2, c.d, 2 * c.d))
    );
    h = poly_tensor_permute(ctx, h, (int64_t[]){0, 2, 1}, 3);
    h = poly_tensor_pad_value_float(
        ctx, h, (int64_t[][2]){{0, 0}, {0, 0}, {c.kernel - 1, 0}}, 3, 0
    );
    snprintf(name, sizeof(name), "%sdepthwise_conv1d.weight", sub);
    PolyTensor *w = poly_model_param(m, name, POLY_FLOAT32, (int64_t[]){c.d, 1, c.kernel}, 3);
    h = poly_tensor_conv2d(ctx, h, w, NULL, c.d, (int64_t[]){1}, (int64_t[]){1}, (int64_t[]){0}, 1);
    h = poly_tensor_permute(ctx, h, (int64_t[]){0, 2, 1}, 3);
    h = model_gemma_norm(m, sub, "conv_norm", clip(ctx, h, -c.clip, c.clip), c.d, c.eps);
    h = model_gemma_clippable_linear(
        m, sub, "linear_end", poly_tensor_silu(ctx, h), c.d, c.d, c.clipped
    );
    x = poly_tensor_alu2(ctx, POLY_OP_ADD, x, h);
    snprintf(sub, sizeof(sub), "%sfeed_forward2.", base);
    x = feedforward(m, &c, sub, x);
    x = poly_tensor_contiguous(
        ctx, model_gemma_norm(m, base, "norm_out", clip(ctx, x, -c.clip, c.clip), c.d, 1e-6)
    );
    if (!x) return NULL;
    if (states) states[i + 1] = x;
  }
  return poly_model_linear(m, "audio_tower.output_proj", x, c.d, c.output, true);
}
