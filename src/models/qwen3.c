/*
 * qwen3.c -- Qwen3 model builder + GGUF import semantics
 *
 * Qwen3 is a standard LLaMA-family transformer:
 *   - RMSNorm (pre-norm)
 *   - Separate Q/K/V projections (no fused QKV)
 *   - Grouped Query Attention (GQA): n_kv_heads < n_heads
 *   - Per-head Q/K RMSNorm (qk_norm)
 *   - Rotary Position Embeddings (RoPE)
 *   - SwiGLU FFN: down(silu(gate(x)) * up(x))
 *   - No bias in linear layers
 *
 * Weight naming matches GGUF convention (blk.N.attn_q.weight, etc.).
 * GGUF adapter remaps to internal names.
 *
 * Reference: tinygrad/apps/llm.py TransformerBlock
 */

#define _POSIX_C_SOURCE 200809L
#include "qwen3.h"
#include "registry.h"
#include "factory.h"
#include "../utils.h"

#include "transformer.h"
#include "../model.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/* Config */

Qwen3Config poly_qwen3_config_default(void) {
  return (Qwen3Config){
      .vocab_size = 151936,
      .dim = 1024,
      .n_heads = 16,
      .n_kv_heads = 8,
      .n_layers = 28,
      .hidden_dim = 3072,
      .head_dim = 0, /* computed from dim/n_heads if 0 */
      .max_seq_len = 128,
      .batch_size = 1,
      .norm_eps = 1e-6f,
      .rope_theta = 1000000.0f,
      .qk_norm = 0, /* set to head_dim to enable */
  };
}

/* Builder */

static const ModelTransformerNames qwen3_names = {
    .label = "Qwen3",
    .input = "x",
    .output = "output",
    .cos = "rope_cos",
    .sin = "rope_sin",
    .embedding = "token_embd",
    .norm = "output_norm",
    .block = "blk.%d",
    .attn_norm = "attn_norm",
    .qkv = {"attn_q", "attn_k", "attn_v"},
    .qk_norm = {"attn_q_norm", "attn_k_norm"},
    .out = "attn_output",
    .ffn_norm = "ffn_norm",
    .gate = "ffn_gate",
    .up = "ffn_up",
    .down = "ffn_down"};

static PolyModel *qwen3_build(PolyCtx *ctx, const Qwen3Config *cfg) {
  if (!cfg || cfg->n_heads < 1 || cfg->dim < 1 || cfg->dim % cfg->n_heads) return NULL;
  ModelTransformerConfig c = {
      .dim = cfg->dim,
      .hidden_dim = cfg->hidden_dim,
      .heads = cfg->n_heads,
      .kv_heads = cfg->n_kv_heads > 0 ? cfg->n_kv_heads : cfg->n_heads,
      .layers = cfg->n_layers,
      .vocab = cfg->vocab_size,
      .batch = cfg->batch_size > 0 ? cfg->batch_size : 1,
      .length = cfg->max_seq_len,
      .head_dim = cfg->head_dim > 0 ? cfg->head_dim : cfg->dim / cfg->n_heads,
      .qk_norm = cfg->qk_norm,
      .cache_capacity = cfg->cache_capacity,
      .prefill_chunk = cfg->prefill_chunk_size,
      .eps = cfg->norm_eps > 0 ? (double)cfg->norm_eps : 1e-6,
      .theta = cfg->rope_theta,
      .factor = 1,
      .tied = true,
      .materialize_intermediates = true};
  return model_transformer_build(ctx, &c, &qwen3_names, NULL);
}

/* Standalone C callers own a context through the returned Model; frontends
 * use the context-taking form so Runtime disposal reaches every family. */
PolyModel *poly_qwen3_into(PolyCtx *ctx, const Qwen3Config *cfg, PolyDevice device) {
  PolyModelFactoryScope scope;
  if (!model_factory_begin(&scope, ctx, device)) return NULL;
  return model_factory_end(&scope, qwen3_build(scope.ctx, cfg));
}

/* GGUF import */

#include "../loaders/gguf_decode.h"
#include "../loaders/bind.h"
#include "../loaders/import_error.h"

PolyModel *model_qwen3_from_gguf_decoded(
    const PolyGgufDecoded *gguf,
    const PolyGenericImportOpts *opts
) {
  PolyCtx *ctx = opts ? opts->ctx : NULL;
  int max_batch = opts ? opts->max_batch : 0;
  int max_seq_len = opts ? opts->max_seq_len : 0;
  PolyDevice device = opts ? opts->device : POLY_DEVICE_AUTO;
  if (!gguf) return NULL;

  /* Extract config from GGUF KV */
  const char *arch = gguf->arch ? gguf->arch : "qwen3";
  char key[128];

  Qwen3Config cfg = poly_qwen3_config_default();

#define KV_INT(field, kname)                                                                       \
  do {                                                                                             \
    snprintf(key, sizeof(key), "%s.%s", arch, kname);                                              \
    int v = poly_gguf_kv_int(gguf, key, -1);                                                       \
    if (v >= 0) cfg.field = v;                                                                     \
  } while (0)
#define KV_FLOAT(field, kname)                                                                     \
  do {                                                                                             \
    snprintf(key, sizeof(key), "%s.%s", arch, kname);                                              \
    double v = poly_gguf_kv_float(gguf, key, -1.0);                                                \
    if (v > 0) cfg.field = (float)v;                                                               \
  } while (0)

  KV_INT(dim, "embedding_length");
  KV_INT(n_heads, "attention.head_count");
  KV_INT(n_kv_heads, "attention.head_count_kv");
  KV_INT(n_layers, "block_count");
  KV_INT(hidden_dim, "feed_forward_length");
  KV_FLOAT(norm_eps, "attention.layer_norm_rms_epsilon");
  KV_FLOAT(rope_theta, "rope.freq_base");

#undef KV_INT
#undef KV_FLOAT

  /* The adapter derives head widths before the builder can validate config. */
  if (cfg.n_heads < 1) {
    poly_import_error_set(POLY_IMPORT_ERR_PARSE, "qwen3.attention.head_count must be positive");
    return NULL;
  }

  /* vocab_size from token_embd.weight shape */
  for (int i = 0; i < gguf->n_tensors; i++) {
    if (strcmp(gguf->tensors[i].name, "token_embd.weight") == 0 && gguf->tensors[i].ndim == 2) {
      cfg.vocab_size = (int)gguf->tensors[i].shape[0];
      break;
    }
  }

  /* Infer head_dim from attn_q.weight shape: (n_heads*head_dim, dim) */
  cfg.head_dim = cfg.dim / cfg.n_heads; /* default fallback */
  for (int i = 0; i < gguf->n_tensors; i++) {
    if (strcmp(gguf->tensors[i].name, "blk.0.attn_q.weight") == 0 && gguf->tensors[i].ndim == 2) {
      /* GGUF shape is (out_features, in_features) = (n_heads*head_dim, dim) */
      int q_out = (int)gguf->tensors[i].shape[0];
      cfg.head_dim = q_out / cfg.n_heads;
      break;
    }
  }
  /* Check if qk_norm weights exist */
  for (int i = 0; i < gguf->n_tensors; i++) {
    if (strstr(gguf->tensors[i].name, "attn_q_norm.weight")) {
      cfg.qk_norm = cfg.head_dim;
      break;
    }
  }

  if (max_batch > 0) cfg.batch_size = max_batch;
  if (max_seq_len > 0) cfg.max_seq_len = max_seq_len;
  cfg.cache_capacity = opts ? opts->cache_capacity : 0;
  cfg.prefill_chunk_size = opts ? opts->prefill_chunk_size : 0;

  if (poly_debug_at_least(1))
    fprintf(
        stderr,
        "Qwen3 GGUF: V=%d D=%d H=%d KvH=%d L=%d FF=%d hd=%d T=%d "
        "eps=%.1e rope=%.0f qk_norm=%d\n",
        cfg.vocab_size, cfg.dim, cfg.n_heads, cfg.n_kv_heads, cfg.n_layers, cfg.hidden_dim,
        cfg.head_dim, cfg.max_seq_len, cfg.norm_eps, cfg.rope_theta, cfg.qk_norm
    );

  PolyModel *inst = poly_qwen3_into(ctx, &cfg, device);
  if (!inst) return NULL;

  /* Bind GGUF weights -- names already match internal names */
  PolyBindIndex *idx = poly_bind_index_create(inst);
  if (!idx) {
    poly_model_free(inst);
    return NULL;
  }
  int loaded = 0, skipped = 0;

  for (int i = 0; i < gguf->n_tensors; i++) {
    const PolyDecodedTensor *t = &gguf->tensors[i];

    /* Skip output.weight if present (weight tying with token_embd) */
    if (strcmp(t->name, "output.weight") == 0) {
      skipped++;
      continue;
    }

    /*
     * GGUF stores weights in the model linear layer's (out, in) convention.
     * No transpose needed (unlike HF Conv1D).
     */
    int rc = poly_import_bind_tensor(idx, t->name, t, 0, -1);
    if (rc == 1)
      loaded++;
    else if (rc == 0) {
      skipped++;
      if (poly_debug_at_least(1)) fprintf(stderr, "Qwen3 GGUF: ignoring weight '%s'\n", t->name);
    }

    if (rc < 0) goto fail;
  }

  poly_bind_index_destroy(idx);
  if (poly_debug_at_least(1))
    fprintf(stderr, "Qwen3 GGUF: loaded %d, skipped %d\n", loaded, skipped);
  return inst;

fail:
  poly_bind_index_destroy(idx);
  poly_model_free(inst);
  return NULL;
}
