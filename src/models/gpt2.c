/*
 * gpt2.c -- GPT-2 model builder + HF import semantics
 *
 * Uses the nn.h layer API (poly_linear, poly_layernorm, poly_embedding)
 * and the named buffer registry (poly_param/poly_input/poly_output).
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
#include "../instance.h"
#include "../frontend.h"
#include "../scheduler.h"
#include "../../vendor/cjson/cJSON.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>



/* ── GPT-2 Config ───────────────────────────────────────────────── */

GPT2Config poly_gpt2_config_default(void) {
  return (GPT2Config){
    .vocab_size  = 50257,
    .n_embd      = 768,
    .n_head      = 12,
    .n_layer     = 12,
    .max_seq_len = 1024,
    .batch_size  = 1,
    .norm_eps    = 1e-5f,
  };
}

/* ── GPT-2 Builder ───────────────────────────────────────────────── */

PolyInstance *poly_gpt2(const GPT2Config *cfg) {
  if (!cfg || cfg->n_layer < 1 || cfg->n_embd < 1 || cfg->vocab_size < 1)
    return NULL;

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

  /* ── Register I/O buffers ────────────────────────────────────── */

  int64_t x_shape[] = { B, T };
  PolyUOp *x_buf = poly_input(ctx, POLY_FLOAT32, x_shape, 2, "x");

  int64_t out_shape[] = { B, T, V };
  PolyUOp *out_buf = poly_output(ctx, POLY_FLOAT32, out_shape, 3, "output");

  int64_t pos_shape[] = { 1, T };
  PolyUOp *pos_buf = poly_input(ctx, POLY_FLOAT32, pos_shape, 2, "positions");

  /* ── Build forward graph ─────────────────────────────────────── */

  /* Token + position embeddings */
  PolyUOp *x_shaped = poly_reshape(ctx, x_buf, x_shape, 2);
  PolyUOp *tok_emb = poly_embedding(ctx, "wte", x_shaped, V, D);
  tok_emb = poly_contiguous(ctx,tok_emb);

  PolyUOp *pos_shaped = poly_reshape(ctx, pos_buf, pos_shape, 2);
  PolyUOp *pos_emb = poly_embedding(ctx, "wpe", pos_shaped, T, D);
  pos_emb = poly_contiguous(ctx,pos_emb);

  int64_t h_shape[] = { B, T, D };
  PolyUOp *pos_exp = poly_expand(ctx, pos_emb, h_shape, 3);
  PolyUOp *h = poly_alu2(ctx, POLY_OP_ADD, tok_emb, pos_exp);
  h = poly_contiguous(ctx,h);

  /* Causal mask: (T, T) -> (1, 1, T, T) */
  PolyUOp *mask = poly_contiguous(ctx,poly_reshape(ctx, poly_causal_mask(ctx, T),
                                        (int64_t[]){ 1, 1, T, T }, 4));

  /* ── Transformer blocks ──────────────────────────────────────── */

  for (int i = 0; i < L; i++) {
    char prefix[64];

    /* LayerNorm 1 */
    snprintf(prefix, sizeof(prefix), "h.%d.ln_1", i);
    PolyUOp *ln1 = poly_contiguous(ctx,poly_layernorm(ctx, prefix, h, D, eps));

    /* QKV = Linear(D, 3D) */
    snprintf(prefix, sizeof(prefix), "h.%d.attn.c_attn", i);
    PolyUOp *qkv = poly_contiguous(ctx,poly_linear(ctx, prefix, ln1, D, 3 * D, true));

    /* Split Q, K, V via shrink */
    int64_t shrink_q[][2] = { {0, B}, {0, T}, {0, D} };
    int64_t shrink_k[][2] = { {0, B}, {0, T}, {D, 2*D} };
    int64_t shrink_v[][2] = { {0, B}, {0, T}, {2*D, 3*D} };
    PolyUOp *q = poly_contiguous(ctx,poly_shrink(ctx, qkv, shrink_q, 3));
    PolyUOp *k = poly_contiguous(ctx,poly_shrink(ctx, qkv, shrink_k, 3));
    PolyUOp *v = poly_contiguous(ctx,poly_shrink(ctx, qkv, shrink_v, 3));

    /* Multi-head reshape + permute: (B,T,D) -> (B,H,T,hd) */
    int64_t mh[] = { B, T, H, head_dim };
    int64_t perm[] = { 0, 2, 1, 3 };
    q = poly_permute(ctx, poly_reshape(ctx, q, mh, 4), perm, 4);
    k = poly_permute(ctx, poly_reshape(ctx, k, mh, 4), perm, 4);
    v = poly_permute(ctx, poly_reshape(ctx, v, mh, 4), perm, 4);

    /* Scaled dot-product attention with causal mask */
    PolyUOp *attn_out = poly_contiguous(ctx, poly_sdpa(ctx, q, k, v, mask, 0));

    /* Merge heads: (B,H,T,hd) -> (B,T,D) */
    attn_out = poly_reshape(
        ctx,
        poly_permute(ctx, attn_out, (int64_t[]){ 0, 2, 1, 3 }, 4),
        (int64_t[]){ B, T, D }, 3
    );

    /* Output projection + residual */
    snprintf(prefix, sizeof(prefix), "h.%d.attn.c_proj", i);
    attn_out = poly_contiguous(ctx,poly_linear(ctx, prefix, attn_out, D, D, true));
    h = poly_contiguous(ctx,poly_alu2(ctx, POLY_OP_ADD, h, attn_out));

    /* LayerNorm 2 + FFN + residual */
    snprintf(prefix, sizeof(prefix), "h.%d.ln_2", i);
    PolyUOp *ln2 = poly_contiguous(ctx,poly_layernorm(ctx, prefix, h, D, eps));

    snprintf(prefix, sizeof(prefix), "h.%d.mlp.c_fc", i);
    PolyUOp *ffn = poly_contiguous(ctx,poly_linear(ctx, prefix, ln2, D, 4 * D, true));
    ffn = poly_contiguous(ctx,poly_gelu(ctx, ffn));

    snprintf(prefix, sizeof(prefix), "h.%d.mlp.c_proj", i);
    ffn = poly_contiguous(ctx,poly_linear(ctx, prefix, ffn, 4 * D, D, true));
    h = poly_contiguous(ctx,poly_alu2(ctx, POLY_OP_ADD, h, ffn));
  }

  /* Final layernorm */
  h = poly_contiguous(ctx,poly_layernorm(ctx, "ln_f", h, D, eps));

  /* LM head: weight-tied linear (h @ wte.T, no bias) */
  PolyUOp *wte = poly_ctx_get(ctx, "wte.weight");
  int64_t wte_shape[] = { V, D };
  PolyUOp *logits = poly_linear_apply(ctx, h, poly_reshape(ctx, wte, wte_shape, 2), NULL);

  /* Store output */
  PolyUOp *fwd_store = poly_store_val(ctx, out_buf, logits);
  PolyUOp *fwd_sink = poly_sink1(ctx, fwd_store);
  poly_register_entrypoint(ctx, "forward", fwd_sink);

  /* Loss: sum(logits^2) — surrogate for training test */
  PolyUOp *loss_buf = poly_output(ctx, POLY_FLOAT32, (int64_t[]){1}, 1, "loss");
  PolyUOp *logits_sq = poly_alu2(ctx, POLY_OP_MUL, logits, logits);
  int64_t reduce_all[] = { 0, 1, 2 };
  PolyUOp *loss_sum = poly_reduce_axis(ctx, POLY_OP_ADD, logits_sq, reduce_all, 3);
  PolyUOp *loss_val = poly_reshape(ctx, loss_sum, (int64_t[]){1}, 1);
  PolyUOp *loss_store = poly_store_val(ctx, loss_buf, loss_val);
  poly_register_entrypoint(ctx, "loss", poly_sink1(ctx, loss_store));

  /* Create instance */
  PolyInstance *inst = poly_instance_from_ctx(ctx);
  if (inst) poly_instance_own_ctx(inst);

  return inst;
}

PolyInstance *poly_gpt2_from_json(const char *json, int len) {
  if (!json || len <= 0) return NULL;

  cJSON *root = cJSON_ParseWithLength(json, (size_t)len);
  if (!root) return NULL;

  GPT2Config cfg = poly_gpt2_config_default();
  cJSON *v;
  if ((v = cJSON_GetObjectItem(root, "vocab_size")))          cfg.vocab_size  = v->valueint;
  if ((v = cJSON_GetObjectItem(root, "n_embd")))              cfg.n_embd      = v->valueint;
  if ((v = cJSON_GetObjectItem(root, "n_head")))              cfg.n_head      = v->valueint;
  if ((v = cJSON_GetObjectItem(root, "n_layer")))             cfg.n_layer     = v->valueint;
  if ((v = cJSON_GetObjectItem(root, "n_positions")))         cfg.max_seq_len = v->valueint;
  if ((v = cJSON_GetObjectItem(root, "batch_size")))          cfg.batch_size  = v->valueint;
  if ((v = cJSON_GetObjectItem(root, "layer_norm_epsilon")))  cfg.norm_eps    = (float)v->valuedouble;

  PolyInstance *inst = poly_gpt2(&cfg);
  cJSON_Delete(root);
  return inst;
}

/* ── HF import (model-specific) ─────────────────────────────────── */

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
  if (strstr(name, "attn.bias") != NULL &&
      strstr(name, "c_attn") == NULL &&
      strstr(name, "c_proj") == NULL)
    return 1;
  if (strstr(name, "attn.masked_bias") != NULL)
    return 1;
  if (strcmp(name, "lm_head.weight") == 0)
    return 1;
  return 0;
}

static int gpt2_needs_transpose(
    const char *name,
    int src_ndim, int dst_ndim)
{
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

PolyInstance *poly_gpt2_from_hf_decoded(
    const PolyHfDecoded *hf,
    int max_batch, int max_seq_len)
{
  if (!hf || !hf->config) return NULL;

  GPT2Config cfg = poly_gpt2_config_default();
  cJSON *v;
  if ((v = cJSON_GetObjectItem(hf->config, "vocab_size")))
    cfg.vocab_size = v->valueint;
  if ((v = cJSON_GetObjectItem(hf->config, "n_embd")))
    cfg.n_embd = v->valueint;
  if ((v = cJSON_GetObjectItem(hf->config, "n_head")))
    cfg.n_head = v->valueint;
  if ((v = cJSON_GetObjectItem(hf->config, "n_layer")))
    cfg.n_layer = v->valueint;
  if ((v = cJSON_GetObjectItem(hf->config, "n_positions")))
    cfg.max_seq_len = v->valueint;
  if ((v = cJSON_GetObjectItem(hf->config, "layer_norm_epsilon")))
    cfg.norm_eps = (float)v->valuedouble;
  if (max_batch > 0) cfg.batch_size = max_batch;
  if (max_seq_len > 0) cfg.max_seq_len = max_seq_len;

  PolyInstance *inst = poly_gpt2(&cfg);
  if (!inst) return NULL;

  PolyBindIndex *idx = poly_bind_index_create(inst);
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
    if (!f32) continue;

    int64_t dst_shape[8];
    int dst_ndim = poly_bind_index_dst_shape(idx, name, dst_shape, 8);
    int transpose = (dst_ndim > 0)
        ? gpt2_needs_transpose(name, t->ndim, dst_ndim)
        : 0;

    int rc = poly_import_copy_named_tensor(
        idx, name, f32, t->shape, t->ndim, transpose);
    if (rc == 1) loaded++;
    else if (rc == 0)
      fprintf(stderr, "poly_gpt2_from_hf: no buffer for '%s'\n", name);

    free(f32);
  }

  poly_bind_index_destroy(idx);
  fprintf(stderr, "poly_gpt2_from_hf: loaded %d parameters, skipped %d\n",
          loaded, skipped);
  return inst;
}

PolyInstance *poly_gpt2_from_hf(
    const char *config_json, int config_len,
    const uint8_t **weight_files, const int64_t *weight_lens,
    int n_weight_files,
    int max_batch, int max_seq_len)
{
  PolyHfDecoded *hf = NULL;
  if (poly_hf_decode(config_json, config_len,
                     weight_files, weight_lens, n_weight_files,
                     &hf) != 0 || !hf)
    return NULL;
  PolyInstance *inst = poly_gpt2_from_hf_decoded(hf, max_batch, max_seq_len);
  poly_hf_decoded_free(hf);
  return inst;
}

/* Registry adapter */
PolyInstance *poly_gpt2_from_hf_decoded_generic(
    const PolyHfDecoded *hf,
    const PolyGenericImportOpts *opts)
{
  return poly_gpt2_from_hf_decoded(hf,
      opts ? opts->max_batch : 0,
      opts ? opts->max_seq_len : 0);
}
