#include "llama.h"
#include "factory.h"
#include "layers.h"
#include "../nn/nn.h"
#include "../loaders/import_desc.h"
#include "../loaders/import_error.h"
#include "../loaders/bind.h"
#include <limits.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* extra/models/llama.py: Transformer/TransformerBlock, no-cache branch.
 * State uses HF names and HF half-split Q/K layout. Tinygrad's HF converter
 * permutes those weights to interleaved pairs; poly_tensor_rope instead applies
 * the equivalent half-split rotation directly. Never permute HF weights twice. */
typedef struct {
  int dim, hidden_dim, heads, kv_heads, layers, vocab, batch, length;
  double eps, theta;
  double factor, low_freq, high_freq, original_context;
  bool tied;
} LlamaConfig;

static void llama_error(PolyModelError *err, const char *message) {
  if (err) {
    memset(err, 0, sizeof(*err));
    err->code = POLY_STATUS_INVALID;
    snprintf(err->message, sizeof(err->message), "Llama: %s", message);
  }
}

static bool config_int(const cJSON *json, const char *name, int fallback, int *out) {
  const cJSON *v = cJSON_GetObjectItemCaseSensitive(json, name);
  double n = v ? v->valuedouble : fallback;
  if ((v && !cJSON_IsNumber(v)) || !isfinite(n) || n < 1 || n > INT_MAX || trunc(n) != n)
    return false;
  *out = (int)n;
  return true;
}

static bool config_float(const cJSON *json, const char *name, double fallback, double *out) {
  const cJSON *v = cJSON_GetObjectItemCaseSensitive(json, name);
  *out = v ? v->valuedouble : fallback;
  return (!v || cJSON_IsNumber(v)) && isfinite(*out) && *out > 0;
}

static bool llama_config(const cJSON *json, LlamaConfig *c, PolyModelError *err) {
  if (!cJSON_IsObject(json)) goto invalid;
  const cJSON *kind = cJSON_GetObjectItemCaseSensitive(json, "model_type");
  if (kind && (!cJSON_IsString(kind) || strcmp(kind->valuestring, "llama"))) goto invalid;
  const cJSON *scaling = cJSON_GetObjectItemCaseSensitive(json, "rope_scaling");
  c->factor = 1;
  if (scaling && !cJSON_IsNull(scaling)) {
    const cJSON *type = cJSON_GetObjectItemCaseSensitive(scaling, "rope_type");
    if (!type) type = cJSON_GetObjectItemCaseSensitive(scaling, "type");
    if (!cJSON_IsObject(scaling) || !cJSON_IsString(type) || strcmp(type->valuestring, "llama3") ||
        !config_float(scaling, "factor", 0, &c->factor) || c->factor < 1 ||
        !config_float(scaling, "low_freq_factor", 0, &c->low_freq) ||
        !config_float(scaling, "high_freq_factor", 0, &c->high_freq) ||
        c->high_freq <= c->low_freq ||
        !config_float(scaling, "original_max_position_embeddings", 0, &c->original_context)) {
      llama_error(err, "invalid or unsupported rope_scaling (expected llama3)");
      return false;
    }
  }
  const cJSON *tied = cJSON_GetObjectItemCaseSensitive(json, "tie_word_embeddings");
  if (tied && !cJSON_IsBool(tied)) goto invalid;
  c->tied = cJSON_IsTrue(tied);
  const char *flags[] = {"attention_bias", "mlp_bias"};
  for (int i = 0; i < 2; i++) {
    const cJSON *v = cJSON_GetObjectItemCaseSensitive(json, flags[i]);
    if (v && !cJSON_IsFalse(v)) {
      llama_error(err, "unsupported bias");
      if (err) snprintf(err->message, sizeof(err->message), "Llama: %s must be false", flags[i]);
      return false;
    }
  }
  const cJSON *act = cJSON_GetObjectItemCaseSensitive(json, "hidden_act");
  if (act && (!cJSON_IsString(act) || strcmp(act->valuestring, "silu"))) goto invalid;
  if (!config_int(json, "hidden_size", 0, &c->dim) ||
      !config_int(json, "intermediate_size", 0, &c->hidden_dim) ||
      !config_int(json, "num_attention_heads", 0, &c->heads) ||
      !config_int(json, "num_key_value_heads", c->heads, &c->kv_heads) ||
      !config_int(json, "num_hidden_layers", 0, &c->layers) ||
      !config_int(json, "vocab_size", 0, &c->vocab) ||
      !config_int(json, "batch_size", 1, &c->batch) ||
      !config_int(json, "max_seq_len", 1, &c->length) ||
      !config_float(json, "rms_norm_eps", 1e-6, &c->eps) ||
      !config_float(json, "rope_theta", 10000, &c->theta))
    goto invalid;
  if (c->theta < 1 || c->eps > FLT_MAX) goto invalid;
  const cJSON *partial = cJSON_GetObjectItemCaseSensitive(json, "partial_rotary_factor");
  if (partial && (!cJSON_IsNumber(partial) || partial->valuedouble != 1)) goto invalid;
  /* Do not silently use unscaled defaults for a newer nested RoPE schema. */
  if (cJSON_GetObjectItemCaseSensitive(json, "rope_parameters")) {
    llama_error(err, "use rope_theta and rope_scaling; rope_parameters is not supported");
    return false;
  }
  if (c->dim % c->heads || c->heads % c->kv_heads || (c->dim / c->heads) % 2) goto invalid;
  int hd, max_pos;
  if (!config_int(json, "head_dim", c->dim / c->heads, &hd) || hd != c->dim / c->heads ||
      !config_int(json, "max_position_embeddings", c->length, &max_pos) || c->length > max_pos)
    goto invalid;
  return true;
invalid:
  llama_error(err, "invalid dense Llama dimensions, activation or configuration");
  return false;
}

static PolyTensor *precompute_freqs(PolyModel *model, const LlamaConfig *c, bool sine) {
  int half = c->dim / c->heads / 2;
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
  if (!copied) return NULL;
  if (!v || poly_model_aux(model, sine ? "freqs_sin" : "freqs_cos", v, 0) != POLY_STATUS_OK)
    return NULL;
  return v;
}

static PolyTensor *attention(
    PolyModel *model,
    const LlamaConfig *c,
    PolyTensor *x,
    PolyTensor *cos,
    PolyTensor *sin,
    const char *prefix
) {
  PolyCtx *ctx = poly_model_ctx(model);
  int hd = c->dim / c->heads;
  PolyTensor *qkv[3];
  const char *names[] = {"q", "k", "v"};
  for (int i = 0; i < 3; i++) {
    char name[128];
    snprintf(name, sizeof(name), "%s.self_attn.%s_proj", prefix, names[i]);
    int heads = i == 0 ? c->heads : c->kv_heads;
    PolyTensor *v = poly_model_linear(model, name, x, c->dim, heads * hd, false);
    v = poly_tensor_reshape(ctx, v, (int64_t[]){c->batch, c->length, heads, hd}, 4);
    v = poly_tensor_permute(ctx, v, (int64_t[]){0, 2, 1, 3}, 4);
    qkv[i] = i < 2 ? poly_tensor_rope(ctx, v, cos, sin) : v;
    if (!qkv[i]) return NULL;
  }
  PolyTensor *out = poly_tensor_sdpa(ctx, qkv[0], qkv[1], qkv[2], NULL, 0, 1, 1, 0);
  out = poly_tensor_permute(ctx, out, (int64_t[]){0, 2, 1, 3}, 4);
  out = poly_tensor_reshape(ctx, out, (int64_t[]){c->batch, c->length, c->dim}, 3);
  char name[128];
  snprintf(name, sizeof(name), "%s.self_attn.o_proj", prefix);
  return poly_model_linear(model, name, out, c->dim, c->dim, false);
}

static PolyModel *llama_build(PolyCtx *ctx, const LlamaConfig *c, PolyModelError *err) {
  PolyModel *m = poly_model_new(ctx, NULL);
  if (!m) return NULL;
  const char *stage = "embedding and rotary state";
  PolyTensor *tokens =
      poly_model_input(m, "tokens", POLY_INT32, (int64_t[]){c->batch, c->length}, 2);
  PolyTensor *cos = precompute_freqs(m, c, false), *sin = precompute_freqs(m, c, true);
  PolyTensor *embedding = poly_model_param(
      m, "model.embed_tokens.weight", POLY_FLOAT32, (int64_t[]){c->vocab, c->dim}, 2
  );
  if (c->tied &&
      (!embedding || poly_model_state(m, "lm_head.weight", embedding, 0) != POLY_STATUS_OK))
    goto fail;
  PolyTensor *h = poly_tensor_embedding_apply(ctx, tokens, embedding);
  h = poly_tensor_contiguous(ctx, h);
  if (!h || !cos || !sin) goto fail;
  for (int i = 0; i < c->layers; i++) {
    stage = "attention and feed-forward block";
    char prefix[64], name[128];
    snprintf(prefix, sizeof(prefix), "model.layers.%d", i);
    snprintf(name, sizeof(name), "%s.input_layernorm", prefix);
    PolyTensor *norm = poly_model_rmsnorm(m, name, h, c->dim, c->eps);
    PolyTensor *a = attention(m, c, norm, cos, sin, prefix);
    h = poly_tensor_alu2(ctx, POLY_OP_ADD, h, a);
    snprintf(name, sizeof(name), "%s.post_attention_layernorm", prefix);
    norm = poly_model_rmsnorm(m, name, h, c->dim, c->eps);
    snprintf(name, sizeof(name), "%s.mlp.gate_proj", prefix);
    PolyTensor *gate =
        poly_tensor_silu(ctx, poly_model_linear(m, name, norm, c->dim, c->hidden_dim, false));
    snprintf(name, sizeof(name), "%s.mlp.up_proj", prefix);
    PolyTensor *up = poly_model_linear(m, name, norm, c->dim, c->hidden_dim, false);
    PolyTensor *ff = poly_tensor_alu2(ctx, POLY_OP_MUL, gate, up);
    snprintf(name, sizeof(name), "%s.mlp.down_proj", prefix);
    ff = poly_model_linear(m, name, ff, c->hidden_dim, c->dim, false);
    h = poly_tensor_contiguous(ctx, poly_tensor_alu2(ctx, POLY_OP_ADD, h, ff));
    if (!h) goto fail;
  }
  h = poly_model_rmsnorm(m, "model.norm", h, c->dim, c->eps);
  stage = "output projection";
  PolyTensor *logits = c->tied ? poly_tensor_linear_apply(ctx, h, embedding, NULL)
                               : poly_model_linear(m, "lm_head", h, c->dim, c->vocab, false);
  if (!logits || poly_model_output(m, "logits", logits) != POLY_STATUS_OK ||
      poly_model_entrypoint(
          m, "forward", (const char *[]){"tokens"}, 1, (const char *[]){"logits"}, 1, NULL
      ) != POLY_STATUS_OK)
    goto fail;
  if (poly_model_build(m, err) != POLY_STATUS_OK) goto fail;
  return m;
fail:
  if (err && !err->message[0]) {
    const PolyModelError *detail = poly_model_last_error(m);
    if (detail && detail->message[0])
      *err = *detail;
    else
      llama_error(err, stage);
  }
  poly_model_free(m);
  return NULL;
}

static PolyModel *llama_create(
    PolyCtx *ctx,
    const LlamaConfig *c,
    PolyDevice device,
    PolyModelError *err
) {
  PolyModelFactoryScope scope;
  if (!model_factory_begin(&scope, ctx, device)) {
    llama_error(err, "construction requires an idle runtime and executable device");
    return NULL;
  }
  return model_factory_end(&scope, llama_build(scope.ctx, c, err));
}

PolyModel *poly_llama_from_json(PolyCtx *ctx, const char *json, int len, PolyModelError *err) {
  if (err) memset(err, 0, sizeof(*err));
  const char *end = NULL;
  cJSON *root = json && len > 0 && len <= 1048576
                    ? cJSON_ParseWithLengthOpts(json, (size_t)len, &end, 0)
                    : NULL;
  if (root && end)
    while (end < json + len && (*end == ' ' || *end == '\n' || *end == '\r' || *end == '\t'))
      end++;
  LlamaConfig c;
  PolyModel *m = NULL;
  if (!root || !json || end != json + len)
    llama_error(err, "invalid configuration JSON");
  else if (llama_config(root, &c, err))
    m = llama_create(ctx, &c, POLY_DEVICE_AUTO, err);
  cJSON_Delete(root);
  return m;
}

PolyModel *poly_llama_from_hf_decoded_generic(
    const PolyHfDecoded *hf,
    const PolyGenericImportOpts *opts
) {
  PolyModelError err = {0};
  LlamaConfig c;
  if (!hf || !opts || !llama_config(hf->config, &c, &err)) goto invalid;
  if (opts->max_batch > 0) c.batch = opts->max_batch;
  if (opts->max_seq_len > 0) c.length = opts->max_seq_len;
  int max_pos;
  if (!config_int(hf->config, "max_position_embeddings", c.length, &max_pos) || c.length > max_pos)
    goto invalid;
  PolyModel *m = llama_create(opts->ctx, &c, opts->device, &err);
  if (!m) goto invalid;
  int n = poly_model_buf_count(m);
  bool *seen = calloc((size_t)n, sizeof(bool));
  PolyBindIndex *idx = poly_bind_index_create(m);
  if (!seen || !idx) goto fail;
  const PolyDecodedTensor *embed = NULL, *head = NULL;
  for (int i = 0; i < hf->n_tensors; i++) {
    const PolyDecodedTensor *t = &hf->tensors[i];
    if (!strcmp(t->name, "model.embed_tokens.weight")) embed = t;
    if (!strcmp(t->name, "lm_head.weight")) head = t;
    int b = 0;
    while (b < n && strcmp(poly_model_buf_name(m, b), t->name))
      b++;
    if (b == n || poly_model_buf_role(m, b) != POLY_ROLE_PARAM || seen[b]) {
      poly_import_error_set(
          POLY_IMPORT_ERR_WEIGHT_MISMATCH, "Llama: unexpected or duplicate weight '%s'", t->name
      );
      goto fail;
    }
    float *data = poly_decoded_tensor_to_f32(t);
    int rc = data ? poly_import_copy_named_tensor(idx, t->name, data, t->shape, t->ndim, 0) : -1;
    free(data);
    if (rc != 1) {
      poly_import_error_set(POLY_IMPORT_ERR_WEIGHT_MISMATCH, "Llama: invalid weight '%s'", t->name);
      goto fail;
    }
    seen[b] = true;
  }
  for (int b = 0; b < n; b++)
    if (poly_model_buf_role(m, b) == POLY_ROLE_PARAM && !seen[b]) {
      const char *name = poly_model_buf_name(m, b);
      if (c.tied && ((embed && !strcmp(name, "lm_head.weight")) ||
                     (head && !strcmp(name, "model.embed_tokens.weight"))))
        continue;
      poly_import_error_set(
          POLY_IMPORT_ERR_WEIGHT_MISMATCH, "Llama: missing weight '%s'", poly_model_buf_name(m, b)
      );
      goto fail;
    }
  /* HF safetensors may retain either name of the config-declared tied pair.
   * Both names write the same unpublished storage; if both occur, validate
   * equality before returning the Model so file order cannot choose its value. */
  if (c.tied && head && embed) {
    bool equal = head->ndim == embed->ndim && head->numel == embed->numel;
    for (int i = 0; equal && i < head->ndim; i++)
      equal = head->shape[i] == embed->shape[i];
    float *a = equal ? poly_decoded_tensor_to_f32(embed) : NULL;
    float *b = equal ? poly_decoded_tensor_to_f32(head) : NULL;
    equal = a && b && !memcmp(a, b, (size_t)head->numel * sizeof(float));
    free(a);
    free(b);
    if (!equal) {
      poly_import_error_set(
          POLY_IMPORT_ERR_WEIGHT_MISMATCH, "Llama: tied lm_head disagrees with embedding"
      );
      goto fail;
    }
  }
  free(seen);
  poly_bind_index_destroy(idx);
  return m;
fail:
  free(seen);
  poly_bind_index_destroy(idx);
  poly_model_free(m);
  if (poly_import_last_error_code() == POLY_IMPORT_OK)
    poly_import_error_set(POLY_IMPORT_ERR_INTERNAL, "Llama: checkpoint allocation failed");
  return NULL;
invalid:
  poly_import_error_set(
      POLY_IMPORT_ERR_UNSUPPORTED_MODEL, "%s",
      err.message[0] ? err.message : "Llama: invalid configuration"
  );
  return NULL;
}
