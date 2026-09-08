/*
 * gpt2.c -- GPT-2 model builder + HF import semantics
 *
 * Uses staged PolyModel bindings with nn.h apply/instance helpers.
 *
 * Weight naming matches HuggingFace GPT-2 (minus "transformer." prefix).
 * Weights stored in PyTorch nn.Linear convention: (out, in).
 *
 * Conv1D transpose during HF import:
 *
 *   HF GPT-2 uses OpenAI's Conv1D class (not nn.Linear). Conv1D stores
 *   weights as (in, out) and computes x @ w. Polygrad stores (out, in)
 *   and computes x @ w.T (standard nn.Linear convention).
 *
 *   All Conv1D weights must be transposed during import. This includes
 *   both rectangular weights (c_attn: 768x2304, c_fc: 768x3072) and
 *   square weights (c_proj: 768x768). Detection is by name: any weight
 *   containing ".c_" is Conv1D.
 *
 *   tinygrad handles this identically -- see gpt2.py lines 133-137
 *   where it transposes the same four weight families by explicit name.
 */

#define _POSIX_C_SOURCE 200809L
#include "gpt2.h"
#include "../nn.h"
#include "../tensor.h"
#include "../model.h"
#include "../../vendor/cjson/cJSON.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/* GPT-2 Config */

GPT2Config poly_gpt2_config_default(void) {
  return (GPT2Config){
      .vocab_size = 50257,
      .n_embd = 768,
      .n_head = 12,
      .n_layer = 12,
      .max_seq_len = 1024,
      .batch_size = 1,
      .norm_eps = 1e-5f,
  };
}

/* GPT-2 Builder */

PolyModel *poly_gpt2(const GPT2Config *cfg, PolyDevice device) {
  if (!cfg || cfg->n_layer < 1 || cfg->n_embd < 1 || cfg->vocab_size < 1) return NULL;

  int V = cfg->vocab_size;
  int D = cfg->n_embd;
  int H = cfg->n_head;
  int L = cfg->n_layer;
  int T = cfg->max_seq_len;
  int B = cfg->batch_size > 0 ? cfg->batch_size : 1;
  int head_dim = D / H;
  double eps = cfg->norm_eps > 0 ? (double)cfg->norm_eps : 1e-5;

  if (D % H != 0) {
    fprintf(stderr, "poly_gpt2: n_embd (%d) not divisible by n_head (%d)\n", D, H);
    return NULL;
  }

  PolyCtx *ctx = poly_ctx_new();
  if (!ctx) return NULL;
  if (device != POLY_DEVICE_AUTO) poly_ctx_set_preferred_device(ctx, device);
  PolyModelOptions opts = {
      .own_ctx_on_success = true,
      .own_ctx_on_failure = true,
  };
  PolyModel *inst = poly_model_new(ctx, &opts);
  if (!inst) {
    poly_ctx_destroy(ctx);
    return NULL;
  }

  int64_t x_shape[] = {B, T};
  /* Pinned tinygrad embedding/gather requires integer token indices
   * (mixin/__init__.py:1088-1093,1106-1124). */
  PolyTensor *x_tensor = poly_model_input(inst, "x", POLY_INT32, x_shape, 2);
  if (!x_tensor) goto fail_pre_build;

  int64_t pos_shape[] = {1, T};
  PolyTensor *pos_tensor = poly_model_input(inst, "positions", POLY_INT32, pos_shape, 2);
  if (!pos_tensor) goto fail_pre_build;

  /* Token + position embeddings. Keep wte table visible for LM-head tying. */
  if (poly_model_scope_push(inst, "wte") != POLY_STATUS_OK) goto fail_pre_build;
  int64_t wte_shape[] = {V, D};
  PolyTensor *wte_tensor = poly_model_param(inst, "weight", POLY_FLOAT32, wte_shape, 2);
  if (!wte_tensor) goto fail_pre_build;
  if (poly_model_scope_pop(inst) != POLY_STATUS_OK) goto fail_pre_build;
  PolyTensor *tok_emb = poly_tensor_embedding_apply(ctx, x_tensor, wte_tensor);
  tok_emb = poly_tensor_contiguous(ctx, tok_emb);

  PolyTensor *pos_emb = poly_model_embedding(inst, "wpe", pos_tensor, T, D);
  pos_emb = poly_tensor_contiguous(ctx, pos_emb);

  int64_t h_shape[] = {B, T, D};
  PolyTensor *pos_exp = poly_tensor_expand(ctx, pos_emb, h_shape, 3);
  PolyTensor *h = poly_tensor_alu2(ctx, POLY_OP_ADD, tok_emb, pos_exp);
  h = poly_tensor_contiguous(ctx, h);
  if (!h) goto fail_pre_build;

  /* Causal mask: (T, T) -> (1, 1, T, T) */
  PolyTensor *mask = poly_tensor_causal_mask(ctx, T);
  mask = poly_tensor_reshape(ctx, mask, (int64_t[]){1, 1, T, T}, 4);
  mask = poly_tensor_contiguous(ctx, mask);
  if (!mask) goto fail_pre_build;

  for (int i = 0; i < L; i++) {
    char prefix[64];

    snprintf(prefix, sizeof(prefix), "h.%d.ln_1", i);
    PolyTensor *ln1 = poly_model_layernorm(inst, prefix, h, D, eps);
    ln1 = poly_tensor_contiguous(ctx, ln1);
    if (!ln1) goto fail_pre_build;

    snprintf(prefix, sizeof(prefix), "h.%d.attn.c_attn", i);
    PolyTensor *qkv = poly_model_linear(inst, prefix, ln1, D, 3 * D, true);
    qkv = poly_tensor_contiguous(ctx, qkv);
    if (!qkv) goto fail_pre_build;

    int64_t shrink_q[][2] = {{0, B}, {0, T}, {0, D}};
    int64_t shrink_k[][2] = {{0, B}, {0, T}, {D, 2 * D}};
    int64_t shrink_v[][2] = {{0, B}, {0, T}, {2 * D, 3 * D}};
    PolyTensor *q = poly_tensor_shrink(ctx, qkv, shrink_q, 3);
    q = poly_tensor_contiguous(ctx, q);
    PolyTensor *k = poly_tensor_shrink(ctx, qkv, shrink_k, 3);
    k = poly_tensor_contiguous(ctx, k);
    PolyTensor *v = poly_tensor_shrink(ctx, qkv, shrink_v, 3);
    v = poly_tensor_contiguous(ctx, v);
    if (!q || !k || !v) goto fail_pre_build;

    int64_t mh[] = {B, T, H, head_dim};
    int64_t perm[] = {0, 2, 1, 3};
    q = poly_tensor_reshape(ctx, q, mh, 4);
    q = poly_tensor_permute(ctx, q, perm, 4);
    k = poly_tensor_reshape(ctx, k, mh, 4);
    k = poly_tensor_permute(ctx, k, perm, 4);
    v = poly_tensor_reshape(ctx, v, mh, 4);
    v = poly_tensor_permute(ctx, v, perm, 4);
    if (!q || !k || !v) goto fail_pre_build;

    PolyTensor *attn_out = poly_tensor_sdpa(ctx, q, k, v, mask, 0);
    attn_out = poly_tensor_contiguous(ctx, attn_out);
    if (!attn_out) goto fail_pre_build;

    attn_out = poly_tensor_permute(ctx, attn_out, (int64_t[]){0, 2, 1, 3}, 4);
    attn_out = poly_tensor_reshape(ctx, attn_out, (int64_t[]){B, T, D}, 3);

    snprintf(prefix, sizeof(prefix), "h.%d.attn.c_proj", i);
    attn_out = poly_model_linear(inst, prefix, attn_out, D, D, true);
    attn_out = poly_tensor_contiguous(ctx, attn_out);
    h = poly_tensor_alu2(ctx, POLY_OP_ADD, h, attn_out);
    h = poly_tensor_contiguous(ctx, h);
    if (!h) goto fail_pre_build;

    snprintf(prefix, sizeof(prefix), "h.%d.ln_2", i);
    PolyTensor *ln2 = poly_model_layernorm(inst, prefix, h, D, eps);
    ln2 = poly_tensor_contiguous(ctx, ln2);
    if (!ln2) goto fail_pre_build;

    snprintf(prefix, sizeof(prefix), "h.%d.mlp.c_fc", i);
    PolyTensor *ffn = poly_model_linear(inst, prefix, ln2, D, 4 * D, true);
    ffn = poly_tensor_contiguous(ctx, ffn);
    ffn = poly_tensor_gelu(ctx, ffn);
    ffn = poly_tensor_contiguous(ctx, ffn);
    if (!ffn) goto fail_pre_build;

    snprintf(prefix, sizeof(prefix), "h.%d.mlp.c_proj", i);
    ffn = poly_model_linear(inst, prefix, ffn, 4 * D, D, true);
    ffn = poly_tensor_contiguous(ctx, ffn);
    h = poly_tensor_alu2(ctx, POLY_OP_ADD, h, ffn);
    h = poly_tensor_contiguous(ctx, h);
    if (!h) goto fail_pre_build;
  }

  h = poly_model_layernorm(inst, "ln_f", h, D, eps);
  h = poly_tensor_contiguous(ctx, h);
  if (!h) goto fail_pre_build;

  PolyTensor *logits = poly_tensor_linear_apply(ctx, h, wte_tensor, NULL);
  if (!logits) goto fail_pre_build;

  if (poly_model_output(inst, "output", logits) != POLY_STATUS_OK) goto fail_pre_build;
  const char *forward_inputs[] = {"x", "positions"};
  const char *forward_outputs[] = {"output"};
  if (poly_model_entrypoint(inst, "forward", forward_inputs, 2, forward_outputs, 1, NULL) !=
      POLY_STATUS_OK)
    goto fail_pre_build;

  PolyTensor *logits_sq = poly_tensor_alu2(ctx, POLY_OP_MUL, logits, logits);
  int64_t reduce_all[] = {0, 1, 2};
  PolyTensor *loss_tensor = poly_tensor_sum(ctx, logits_sq, reduce_all, 3, false);
  loss_tensor = poly_tensor_reshape(ctx, loss_tensor, (int64_t[]){1}, 1);
  if (!loss_tensor || poly_model_output(inst, "loss", loss_tensor) != POLY_STATUS_OK)
    goto fail_pre_build;
  const char *loss_inputs[] = {"x", "positions"};
  const char *loss_outputs[] = {"loss"};
  PolyEntrypointOptions loss_opts = {.objective = "loss"};
  if (poly_model_entrypoint(inst, "loss", loss_inputs, 2, loss_outputs, 1, &loss_opts) !=
      POLY_STATUS_OK)
    goto fail_pre_build;

  PolyModelError err = {0};
  if (poly_model_build(inst, &err) != POLY_STATUS_OK) {
    if (err.message[0]) fprintf(stderr, "poly_gpt2: build failed: %s\n", err.message);
    poly_model_free(inst);
    return NULL;
  }
  return inst;

fail_pre_build:
  poly_model_free(inst);
  poly_ctx_destroy(ctx);
  return NULL;
}

PolyModel *poly_gpt2_from_json(const char *json, int len, PolyDevice device) {
  if (!json || len <= 0) return NULL;

  cJSON *root = cJSON_ParseWithLength(json, (size_t)len);
  if (!root) return NULL;

  GPT2Config cfg = poly_gpt2_config_default();
  cJSON *v;
  if ((v = cJSON_GetObjectItem(root, "vocab_size"))) cfg.vocab_size = v->valueint;
  if ((v = cJSON_GetObjectItem(root, "n_embd"))) cfg.n_embd = v->valueint;
  if ((v = cJSON_GetObjectItem(root, "n_head"))) cfg.n_head = v->valueint;
  if ((v = cJSON_GetObjectItem(root, "n_layer"))) cfg.n_layer = v->valueint;
  if ((v = cJSON_GetObjectItem(root, "n_positions"))) cfg.max_seq_len = v->valueint;
  if ((v = cJSON_GetObjectItem(root, "batch_size"))) cfg.batch_size = v->valueint;
  if ((v = cJSON_GetObjectItem(root, "layer_norm_epsilon"))) cfg.norm_eps = (float)v->valuedouble;

  PolyModel *inst = poly_gpt2(&cfg, device);
  cJSON_Delete(root);
  return inst;
}

/* HF import (model-specific) */

#include "../loaders/hf_decode.h"
#include "../loaders/bind.h"
#include "../loaders/import_desc.h"
#include "../loaders/import_error.h"

static const char *gpt2_strip_prefix(const char *name, const char *prefix) {
  size_t plen = strlen(prefix);
  if (strncmp(name, prefix, plen) == 0) return name + plen;
  return name;
}

static int gpt2_should_skip(const char *name) {
  if (strstr(name, "attn.bias") != NULL && strstr(name, "c_attn") == NULL &&
      strstr(name, "c_proj") == NULL)
    return 1;
  if (strstr(name, "attn.masked_bias") != NULL) return 1;
  if (strcmp(name, "lm_head.weight") == 0) return 1;
  return 0;
}

static int gpt2_needs_transpose(const char *name, int src_ndim, int dst_ndim) {
  /*
   * HF GPT-2 uses Conv1D layers which store weights as (in, out).
   * Polygrad's linear layer stores (out, in) and computes x @ w.T.
   * All Conv1D weights need transposing during import, including
   * square ones (attn.c_proj is 768x768).
   *
   * Conv1D layers: c_attn, c_proj, c_fc (all contain ".c_" in name).
   * Non-Conv1D 2D weights: wte.weight, wpe.weight (embeddings).
   */
  if (src_ndim != 2 || dst_ndim != 2) return 0;
  if (strstr(name, ".c_") != NULL) return 1;
  return 0;
}

PolyModel *poly_gpt2_from_hf_decoded(
    const PolyHfDecoded *hf,
    int max_batch,
    int max_seq_len,
    PolyDevice device
) {
  if (!hf || !hf->config) return NULL;

  GPT2Config cfg = poly_gpt2_config_default();
  cJSON *v;
  if ((v = cJSON_GetObjectItem(hf->config, "vocab_size"))) cfg.vocab_size = v->valueint;
  if ((v = cJSON_GetObjectItem(hf->config, "n_embd"))) cfg.n_embd = v->valueint;
  if ((v = cJSON_GetObjectItem(hf->config, "n_head"))) cfg.n_head = v->valueint;
  if ((v = cJSON_GetObjectItem(hf->config, "n_layer"))) cfg.n_layer = v->valueint;
  if ((v = cJSON_GetObjectItem(hf->config, "n_positions"))) cfg.max_seq_len = v->valueint;
  if ((v = cJSON_GetObjectItem(hf->config, "layer_norm_epsilon")))
    cfg.norm_eps = (float)v->valuedouble;
  if (max_batch > 0) cfg.batch_size = max_batch;
  if (max_seq_len > 0) cfg.max_seq_len = max_seq_len;

  PolyModel *inst = poly_gpt2(&cfg, device);
  if (!inst) return NULL;

  PolyBindIndex *idx = poly_bind_index_create(inst);
  if (!idx) {
    poly_model_free(inst);
    return NULL;
  }
  int loaded = 0, skipped = 0;

  for (int i = 0; i < hf->n_tensors; i++) {
    const PolyDecodedTensor *t = &hf->tensors[i];

    const char *name = gpt2_strip_prefix(t->name, "transformer.");
    name = gpt2_strip_prefix(name, "model.");

    if (gpt2_should_skip(name)) {
      skipped++;
      continue;
    }

    float *f32 = poly_decoded_tensor_to_f32(t);
    if (!f32) {
      poly_import_error_set(
          POLY_IMPORT_ERR_WEIGHT_MISMATCH, "failed to convert weight '%s' (dtype=%d)", t->name,
          t->dtype
      );
      goto fail;
    }

    int64_t dst_shape[8];
    int dst_ndim = poly_bind_index_dst_shape(idx, name, dst_shape, 8);
    int transpose = (dst_ndim > 0) ? gpt2_needs_transpose(name, t->ndim, dst_ndim) : 0;

    int rc = poly_import_copy_named_tensor(idx, name, f32, t->shape, t->ndim, transpose);
    if (rc == 1)
      loaded++;
    else if (rc == 0)
      fprintf(stderr, "poly_gpt2_from_hf: no buffer for '%s'\n", name);

    free(f32);
    /* An ignored source key is distinct from a failed write to named state. */
    if (rc < 0) goto fail;
  }

  poly_bind_index_destroy(idx);
  fprintf(stderr, "poly_gpt2_from_hf: loaded %d parameters, skipped %d\n", loaded, skipped);
  return inst;

fail:
  poly_bind_index_destroy(idx);
  poly_model_free(inst);
  return NULL;
}

PolyModel *poly_gpt2_from_hf(
    const char *config_json,
    int config_len,
    const uint8_t **weight_files,
    const int64_t *weight_lens,
    int n_weight_files,
    int max_batch,
    int max_seq_len,
    PolyDevice device
) {
  PolyHfDecoded *hf = NULL;
  if (poly_hf_decode(config_json, config_len, weight_files, weight_lens, n_weight_files, &hf) !=
          0 ||
      !hf)
    return NULL;
  PolyModel *inst = poly_gpt2_from_hf_decoded(hf, max_batch, max_seq_len, device);
  poly_hf_decoded_free(hf);
  return inst;
}

/* Registry adapter */
PolyModel *poly_gpt2_from_hf_decoded_generic(
    const PolyHfDecoded *hf,
    const PolyGenericImportOpts *opts
) {
  return poly_gpt2_from_hf_decoded(
      hf, opts ? opts->max_batch : 0, opts ? opts->max_seq_len : 0,
      opts ? opts->device : POLY_DEVICE_AUTO
  );
}

/* GGUF import (model-specific) */

#include "../loaders/gguf_decode.h"

/*
 * GGUF name remapping: matches tinygrad gpt2.py _remap_gguf_key().
 * GGUF uses "blk.N.attn_qkv.weight", polygrad uses "h.N.attn.c_attn.weight".
 */
static const char *gpt2_gguf_remap[][2] = {
    {"blk.", "h."},
    {".attn_qkv.bias", ".attn.c_attn.bias"},
    {".attn_qkv.weight", ".attn.c_attn.weight"},
    {".ffn_norm.bias", ".ln_2.bias"},
    {".ffn_norm.weight", ".ln_2.weight"},
    {".attn_norm.bias", ".ln_1.bias"},
    {".attn_norm.weight", ".ln_1.weight"},
    {".attn_output.bias", ".attn.c_proj.bias"},
    {".attn_output.weight", ".attn.c_proj.weight"},
    {".ffn_up.bias", ".mlp.c_fc.bias"},
    {".ffn_up.weight", ".mlp.c_fc.weight"},
    {".ffn_down.bias", ".mlp.c_proj.bias"},
    {".ffn_down.weight", ".mlp.c_proj.weight"},
    {"token_embd.weight", "wte.weight"},
    {"output.weight", "lm_head.weight"},
    {"output_norm.bias", "ln_f.bias"},
    {"output_norm.weight", "ln_f.weight"},
    {"position_embd.weight", "wpe.weight"},
    {NULL, NULL}};

static const char *gpt2_gguf_map_name(const char *name, char *buf, int buf_size) {
  /* Apply all replacements in order */
  strncpy(buf, name, (size_t)(buf_size - 1));
  buf[buf_size - 1] = '\0';

  for (int i = 0; gpt2_gguf_remap[i][0]; i++) {
    const char *old_s = gpt2_gguf_remap[i][0];
    const char *new_s = gpt2_gguf_remap[i][1];
    char *pos = strstr(buf, old_s);
    if (!pos) continue;
    size_t old_len = strlen(old_s);
    size_t new_len = strlen(new_s);
    size_t tail_len = strlen(pos + old_len);
    if ((pos - buf) + new_len + tail_len >= (size_t)(buf_size - 1)) continue;
    memmove(pos + new_len, pos + old_len, tail_len + 1);
    memcpy(pos, new_s, new_len);
  }

  /* Skip lm_head.weight (weight tying with wte) */
  if (strcmp(buf, "lm_head.weight") == 0) return NULL;

  return buf;
}

PolyModel *poly_gpt2_from_gguf_decoded(
    const PolyGgufDecoded *gguf,
    int max_batch,
    int max_seq_len,
    PolyDevice device
) {
  if (!gguf) return NULL;

  /* Extract config from GGUF KV metadata */
  GPT2Config cfg = poly_gpt2_config_default();
  cfg.n_embd = poly_gguf_kv_int(gguf, "gpt2.embedding_length", cfg.n_embd);
  cfg.n_head = poly_gguf_kv_int(gguf, "gpt2.attention.head_count", cfg.n_head);
  cfg.n_layer = poly_gguf_kv_int(gguf, "gpt2.block_count", cfg.n_layer);
  cfg.max_seq_len = poly_gguf_kv_int(gguf, "gpt2.context_length", cfg.max_seq_len);
  cfg.norm_eps =
      (float)poly_gguf_kv_float(gguf, "gpt2.attention.layer_norm_epsilon", (double)cfg.norm_eps);
  /* vocab_size from token_embd.weight shape (not always in KV) */
  for (int i = 0; i < gguf->n_tensors; i++) {
    if (strcmp(gguf->tensors[i].name, "token_embd.weight") == 0 && gguf->tensors[i].ndim == 2) {
      cfg.vocab_size = (int)gguf->tensors[i].shape[0];
      break;
    }
  }

  if (max_batch > 0) cfg.batch_size = max_batch;
  if (max_seq_len > 0) cfg.max_seq_len = max_seq_len;

  PolyModel *inst = poly_gpt2(&cfg, device);
  if (!inst) return NULL;

  PolyBindIndex *idx = poly_bind_index_create(inst);
  if (!idx) {
    poly_model_free(inst);
    return NULL;
  }
  int loaded = 0, skipped = 0;
  char name_buf[256];

  for (int i = 0; i < gguf->n_tensors; i++) {
    const PolyDecodedTensor *t = &gguf->tensors[i];

    /* Remap GGUF name to polygrad internal name */
    const char *name = gpt2_gguf_map_name(t->name, name_buf, sizeof(name_buf));
    if (!name) {
      skipped++;
      continue;
    }

    /* Convert to F32 (dequantize if needed) */
    float *f32 = poly_decoded_tensor_to_f32(t);
    if (!f32) {
      poly_import_error_set(
          POLY_IMPORT_ERR_WEIGHT_MISMATCH, "failed to convert weight '%s' (dtype=%d)", t->name,
          t->dtype
      );
      goto fail;
    }

    /*
     * GGUF weights are stored in the model's native convention
     * (not Conv1D). No transpose needed -- GGUF stores (out, in)
     * which matches polygrad's linear layer convention.
     */
    int rc = poly_import_copy_named_tensor(idx, name, f32, t->shape, t->ndim, 0);
    if (rc == 1)
      loaded++;
    else if (rc == 0)
      fprintf(stderr, "poly_gpt2_from_gguf: no buffer for '%s' (was '%s')\n", name, t->name);

    free(f32);
    if (rc < 0) goto fail;
  }

  poly_bind_index_destroy(idx);
  fprintf(stderr, "poly_gpt2_from_gguf: loaded %d parameters, skipped %d\n", loaded, skipped);
  return inst;

fail:
  poly_bind_index_destroy(idx);
  poly_model_free(inst);
  return NULL;
}

PolyModel *poly_gpt2_from_gguf(
    const uint8_t *data,
    int64_t len,
    int max_batch,
    int max_seq_len,
    PolyDevice device
) {
  PolyGgufDecoded *gguf = NULL;
  if (poly_gguf_decode(data, len, &gguf) != 0 || !gguf) return NULL;
  PolyModel *inst = poly_gpt2_from_gguf_decoded(gguf, max_batch, max_seq_len, device);
  poly_gguf_decoded_free(gguf);
  return inst;
}

/* GGUF registry adapter */
PolyModel *poly_gpt2_from_gguf_decoded_generic(
    const PolyGgufDecoded *gguf,
    const PolyGenericImportOpts *opts
) {
  return poly_gpt2_from_gguf_decoded(
      gguf, opts ? opts->max_batch : 0, opts ? opts->max_seq_len : 0,
      opts ? opts->device : POLY_DEVICE_AUTO
  );
}
