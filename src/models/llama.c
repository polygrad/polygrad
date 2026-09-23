#include "../model.h"
#include "../polygrad.h"
#include "../tensor.h"
#include "factory.h"
#include "transformer.h"
#include "../nn/nn.h"
#include "registry.h"
#include "../loaders/import_error.h"
#include "../loaders/bind.h"
#include "../../vendor/cjson/cJSON.h"
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
typedef ModelTransformerConfig LlamaConfig;

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
  memset(c, 0, sizeof(*c));
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
  c->cache_capacity = c->prefill_chunk = 0;
  if (cJSON_GetObjectItemCaseSensitive(json, "cache_capacity")) {
    if (!config_int(json, "cache_capacity", 0, &c->cache_capacity) || c->batch != 1 ||
        !config_int(json, "prefill_chunk_size", 1, &c->prefill_chunk) ||
        c->prefill_chunk > c->cache_capacity)
      goto invalid;
  } else if (cJSON_GetObjectItemCaseSensitive(json, "prefill_chunk_size"))
    goto invalid;
  int hd, max_pos;
  if (!config_int(json, "head_dim", c->dim / c->heads, &hd) || hd != c->dim / c->heads ||
      !config_int(
          json, "max_position_embeddings",
          c->cache_capacity > c->length ? c->cache_capacity : c->length, &max_pos
      ) ||
      c->length > max_pos || c->cache_capacity > max_pos)
    goto invalid;
  c->head_dim = hd;
  return true;
invalid:
  llama_error(err, "invalid dense Llama dimensions, activation or configuration");
  return false;
}

static const ModelTransformerNames llama_names = {
    .label = "Llama",
    .input = "tokens",
    .output = "logits",
    .cos = "freqs_cos",
    .sin = "freqs_sin",
    .embedding = "model.embed_tokens",
    .norm = "model.norm",
    .head = "lm_head",
    .block = "model.layers.%d",
    .attn_norm = "input_layernorm",
    .qkv = {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"},
    .out = "self_attn.o_proj",
    .ffn_norm = "post_attention_layernorm",
    .gate = "mlp.gate_proj",
    .up = "mlp.up_proj",
    .down = "mlp.down_proj"};

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
  return model_factory_end(&scope, model_transformer_build(scope.ctx, c, &llama_names, err));
}

PolyModel *model_llama_from_config(PolyCtx *ctx, const cJSON *root, PolyModelError *err) {
  LlamaConfig c;
  return llama_config(root, &c, err) ? model_transformer_build(ctx, &c, &llama_names, err) : NULL;
}

PolyModel *model_llama_from_hf_decoded(const PolyHfDecoded *hf, const PolyGenericImportOpts *opts) {
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
    int rc = poly_import_bind_tensor(idx, t->name, t, 0, -1);
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
