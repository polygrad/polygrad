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
#include "../nn.h"
#include "../tensor.h"
#include "../instance.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/* Config */

Qwen3Config poly_qwen3_config_default(void) {
    return (Qwen3Config){
        .vocab_size  = 151936,
        .dim         = 1024,
        .n_heads     = 16,
        .n_kv_heads  = 8,
        .n_layers    = 28,
        .hidden_dim  = 3072,
        .head_dim    = 0,  /* computed from dim/n_heads if 0 */
        .max_seq_len = 128,
        .batch_size  = 1,
        .norm_eps    = 1e-6f,
        .rope_theta  = 1000000.0f,
        .qk_norm     = 0,  /* set to head_dim to enable */
    };
}

/* Builder */

PolyInstance *poly_qwen3(const Qwen3Config *cfg) {
  if (!cfg || cfg->n_layers < 1 || cfg->dim < 1 || cfg->vocab_size < 1) return NULL;

  int V = cfg->vocab_size;
  int D = cfg->dim;
  int H = cfg->n_heads;
  int KvH = cfg->n_kv_heads > 0 ? cfg->n_kv_heads : H;
  int L = cfg->n_layers;
  int FF = cfg->hidden_dim;
  int T = cfg->max_seq_len;
  int B = cfg->batch_size > 0 ? cfg->batch_size : 1;
  int hd = cfg->head_dim > 0 ? cfg->head_dim : D / H;
  double eps = cfg->norm_eps > 0 ? (double)cfg->norm_eps : 1e-6;
  int qk_norm = cfg->qk_norm;

  if (D % H != 0) {
    fprintf(stderr, "poly_qwen3: dim (%d) not divisible by n_heads (%d)\n", D, H);
    return NULL;
  }

  PolyCtx *ctx = poly_ctx_new();
  if (!ctx) return NULL;
  PolyInstanceOptions opts = {
      .own_ctx_on_success = true,
      .own_ctx_on_failure = true,
  };
  PolyInstance *inst = poly_instance_new(ctx, &opts);
  if (!inst) {
    poly_ctx_destroy(ctx);
    return NULL;
  }

  int64_t x_shape[] = {B, T};
  PolyTensor *x_tensor = poly_instance_input(inst, "x", POLY_FLOAT32, x_shape, 2);
  if (!x_tensor) goto fail_pre_build;

  int half_hd = hd / 2;
  int64_t rope_shape[] = {T, half_hd};
  PolyTensor *rope_cos_tensor = poly_instance_input(inst, "rope_cos", POLY_FLOAT32, rope_shape, 2);
  if (!rope_cos_tensor) goto fail_pre_build;
  PolyTensor *rope_sin_tensor = poly_instance_input(inst, "rope_sin", POLY_FLOAT32, rope_shape, 2);
  if (!rope_sin_tensor) goto fail_pre_build;

  PolyUOp *x = poly_tensor_uop(x_tensor);
  if (poly_instance_scope_push(inst, "token_embd") != POLY_STATUS_OK) goto fail_pre_build;
  int64_t token_shape[] = {V, D};
  PolyTensor *token_tensor = poly_instance_param(inst, "weight", POLY_FLOAT32, token_shape, 2);
  if (!token_tensor) goto fail_pre_build;
  if (poly_instance_scope_pop(inst) != POLY_STATUS_OK) goto fail_pre_build;
  PolyUOp *token_weight = poly_tensor_uop(token_tensor);
  PolyUOp *h = poly_embedding_apply(ctx, x, poly_reshape(ctx, token_weight, token_shape, 2));
  h = poly_contiguous(ctx, h);
  if (!h) goto fail_pre_build;

  PolyUOp *rope_cos =
      poly_reshape(ctx, poly_tensor_uop(rope_cos_tensor), (int64_t[]){1, 1, T, half_hd}, 4);
  PolyUOp *rope_sin =
      poly_reshape(ctx, poly_tensor_uop(rope_sin_tensor), (int64_t[]){1, 1, T, half_hd}, 4);

  PolyUOp *mask =
      poly_contiguous(ctx, poly_reshape(ctx, poly_causal_mask(ctx, T), (int64_t[]){1, 1, T, T}, 4));
  if (!mask) goto fail_pre_build;

  for (int i = 0; i < L; i++) {
    char pf[64];

    snprintf(pf, sizeof(pf), "blk.%d.attn_norm", i);
    PolyUOp *x_norm = poly_contiguous(ctx, poly_instance_rmsnorm(inst, pf, h, D, eps));
    if (!x_norm) goto fail_pre_build;

    snprintf(pf, sizeof(pf), "blk.%d.attn_q", i);
    PolyUOp *q = poly_contiguous(ctx, poly_instance_linear(inst, pf, x_norm, D, H * hd, false));
    snprintf(pf, sizeof(pf), "blk.%d.attn_k", i);
    PolyUOp *k = poly_contiguous(ctx, poly_instance_linear(inst, pf, x_norm, D, KvH * hd, false));
    snprintf(pf, sizeof(pf), "blk.%d.attn_v", i);
    PolyUOp *v = poly_contiguous(ctx, poly_instance_linear(inst, pf, x_norm, D, KvH * hd, false));
    if (!q || !k || !v) goto fail_pre_build;

    q = poly_permute(
        ctx, poly_reshape(ctx, q, (int64_t[]){B, T, H, hd}, 4), (int64_t[]){0, 2, 1, 3}, 4
    );
    k = poly_permute(
        ctx, poly_reshape(ctx, k, (int64_t[]){B, T, KvH, hd}, 4), (int64_t[]){0, 2, 1, 3}, 4
    );
    v = poly_permute(
        ctx, poly_reshape(ctx, v, (int64_t[]){B, T, KvH, hd}, 4), (int64_t[]){0, 2, 1, 3}, 4
    );

    if (qk_norm > 0) {
      snprintf(pf, sizeof(pf), "blk.%d.attn_q_norm", i);
      q = poly_contiguous(ctx, poly_instance_rmsnorm(inst, pf, q, qk_norm, eps));
      snprintf(pf, sizeof(pf), "blk.%d.attn_k_norm", i);
      k = poly_contiguous(ctx, poly_instance_rmsnorm(inst, pf, k, qk_norm, eps));
      if (!q || !k) goto fail_pre_build;
    }

    q = poly_rope(ctx, q, rope_cos, rope_sin);
    k = poly_rope(ctx, k, rope_cos, rope_sin);

    PolyUOp *attn = poly_contiguous(ctx, poly_sdpa(ctx, q, k, v, mask, 0));
    if (!attn) goto fail_pre_build;

    attn = poly_reshape(
        ctx, poly_permute(ctx, attn, (int64_t[]){0, 2, 1, 3}, 4), (int64_t[]){B, T, H * hd}, 3
    );

    snprintf(pf, sizeof(pf), "blk.%d.attn_output", i);
    attn = poly_contiguous(ctx, poly_instance_linear(inst, pf, attn, H * hd, D, false));
    h = poly_contiguous(ctx, poly_add(ctx, h, attn));
    if (!h) goto fail_pre_build;

    snprintf(pf, sizeof(pf), "blk.%d.ffn_norm", i);
    PolyUOp *h_norm = poly_contiguous(ctx, poly_instance_rmsnorm(inst, pf, h, D, eps));
    if (!h_norm) goto fail_pre_build;

    snprintf(pf, sizeof(pf), "blk.%d.ffn_gate", i);
    PolyUOp *gate =
        poly_contiguous(ctx, poly_silu(ctx, poly_instance_linear(inst, pf, h_norm, D, FF, false)));

    snprintf(pf, sizeof(pf), "blk.%d.ffn_up", i);
    PolyUOp *up = poly_contiguous(ctx, poly_instance_linear(inst, pf, h_norm, D, FF, false));
    if (!gate || !up) goto fail_pre_build;

    PolyUOp *gated = poly_contiguous(ctx, poly_alu2(ctx, POLY_OP_MUL, gate, up));

    snprintf(pf, sizeof(pf), "blk.%d.ffn_down", i);
    PolyUOp *ffn_out = poly_contiguous(ctx, poly_instance_linear(inst, pf, gated, FF, D, false));

    h = poly_contiguous(ctx, poly_add(ctx, h, ffn_out));
    if (!h) goto fail_pre_build;
  }

  h = poly_contiguous(ctx, poly_instance_rmsnorm(inst, "output_norm", h, D, eps));
  if (!h) goto fail_pre_build;

  PolyUOp *logits =
      poly_linear_apply(ctx, h, poly_reshape(ctx, token_weight, token_shape, 2), NULL);
  if (!logits) goto fail_pre_build;

  PolyTensor *out_tensor = poly_tensor_create(ctx, logits, POLY_TENSOR_VALUE, POLY_DEVICE_AUTO);
  if (!out_tensor || poly_instance_output(inst, "output", out_tensor) != POLY_STATUS_OK)
    goto fail_pre_build;
  const char *forward_inputs[] = {"x", "rope_cos", "rope_sin"};
  const char *forward_outputs[] = {"output"};
  if (poly_instance_entrypoint(inst, "forward", forward_inputs, 3, forward_outputs, 1, NULL) !=
      POLY_STATUS_OK)
    goto fail_pre_build;

  PolyInstanceError err = {0};
  if (poly_instance_build(inst, &err) != POLY_STATUS_OK) {
    if (err.message[0]) fprintf(stderr, "poly_qwen3: build failed: %s\n", err.message);
    poly_instance_free(inst);
    return NULL;
  }
  return inst;

fail_pre_build:
  poly_instance_free(inst);
  poly_ctx_destroy(ctx);
  return NULL;
}

/* GGUF import */

#include "../loaders/gguf_decode.h"
#include "../loaders/bind.h"
#include "../loaders/import_desc.h"
#include "../loaders/import_error.h"

PolyInstance *poly_qwen3_from_gguf_decoded(
    const PolyGgufDecoded *gguf,
    int max_batch, int max_seq_len)
{
    if (!gguf) return NULL;

    /* Extract config from GGUF KV */
    const char *arch = gguf->arch ? gguf->arch : "qwen3";
    char key[128];

    Qwen3Config cfg = poly_qwen3_config_default();

    #define KV_INT(field, kname) do { \
        snprintf(key, sizeof(key), "%s.%s", arch, kname); \
        int v = poly_gguf_kv_int(gguf, key, -1); \
        if (v >= 0) cfg.field = v; \
    } while(0)
    #define KV_FLOAT(field, kname) do { \
        snprintf(key, sizeof(key), "%s.%s", arch, kname); \
        double v = poly_gguf_kv_float(gguf, key, -1.0); \
        if (v > 0) cfg.field = (float)v; \
    } while(0)

    KV_INT(dim,        "embedding_length");
    KV_INT(n_heads,    "attention.head_count");
    KV_INT(n_kv_heads, "attention.head_count_kv");
    KV_INT(n_layers,   "block_count");
    KV_INT(hidden_dim, "feed_forward_length");
    KV_FLOAT(norm_eps, "attention.layer_norm_rms_epsilon");
    KV_FLOAT(rope_theta, "rope.freq_base");

    #undef KV_INT
    #undef KV_FLOAT

    /* vocab_size from token_embd.weight shape */
    for (int i = 0; i < gguf->n_tensors; i++) {
        if (strcmp(gguf->tensors[i].name, "token_embd.weight") == 0 &&
            gguf->tensors[i].ndim == 2) {
            cfg.vocab_size = (int)gguf->tensors[i].shape[0];
            break;
        }
    }

    /* Infer head_dim from attn_q.weight shape: (n_heads*head_dim, dim) */
    cfg.head_dim = cfg.dim / cfg.n_heads;  /* default fallback */
    for (int i = 0; i < gguf->n_tensors; i++) {
        if (strcmp(gguf->tensors[i].name, "blk.0.attn_q.weight") == 0 &&
            gguf->tensors[i].ndim == 2) {
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

    fprintf(stderr, "poly_qwen3: V=%d D=%d H=%d KvH=%d L=%d FF=%d hd=%d T=%d "
            "eps=%.1e rope=%.0f qk_norm=%d\n",
            cfg.vocab_size, cfg.dim, cfg.n_heads, cfg.n_kv_heads,
            cfg.n_layers, cfg.hidden_dim, cfg.head_dim,
            cfg.max_seq_len, cfg.norm_eps, cfg.rope_theta, cfg.qk_norm);

    PolyInstance *inst = poly_qwen3(&cfg);
    if (!inst) return NULL;

    /* Precompute RoPE frequencies and fill the input buffers */
    {
        int hd = cfg.head_dim;
        int T = cfg.max_seq_len;
        double theta = (double)cfg.rope_theta;

        /* Find rope_cos and rope_sin buffers */
        int nb = poly_instance_buf_count(inst);
        float *cos_data = NULL, *sin_data = NULL;
        int64_t cos_numel = 0, sin_numel = 0;
        for (int b = 0; b < nb; b++) {
            const char *bname = poly_instance_buf_name(inst, b);
            if (strcmp(bname, "rope_cos") == 0)
                cos_data = poly_instance_buf_data(inst, b, &cos_numel);
            else if (strcmp(bname, "rope_sin") == 0)
                sin_data = poly_instance_buf_data(inst, b, &sin_numel);
        }
        if (cos_data && sin_data) {
            int half = hd / 2;
            for (int pos = 0; pos < T; pos++) {
                for (int j = 0; j < half; j++) {
                    double freq = 1.0 / pow(theta, (double)(2 * j) / (double)hd);
                    double angle = (double)pos * freq;
                    cos_data[pos * half + j] = (float)cos(angle);
                    sin_data[pos * half + j] = (float)sin(angle);
                }
            }
        }
    }

    /* Bind GGUF weights -- names already match internal names */
    PolyBindIndex *idx = poly_bind_index_create(inst);
    int loaded = 0, skipped = 0;

    for (int i = 0; i < gguf->n_tensors; i++) {
        const PolyDecodedTensor *t = &gguf->tensors[i];

        /* Skip output.weight if present (weight tying with token_embd) */
        if (strcmp(t->name, "output.weight") == 0) { skipped++; continue; }

        float *f32 = poly_decoded_tensor_to_f32(t);
        if (!f32) {
            fprintf(stderr, "poly_qwen3_from_gguf: failed to convert '%s' (dtype=%d)\n",
                    t->name, t->dtype);
            continue;
        }

        /*
         * GGUF stores weights in (out, in) convention matching poly_linear.
         * No transpose needed (unlike HF Conv1D).
         */
        int rc = poly_import_copy_named_tensor(
            idx, t->name, f32, t->shape, t->ndim, 0);
        if (rc == 1) loaded++;
        else if (rc == 0)
            fprintf(stderr, "poly_qwen3_from_gguf: no buffer for '%s'\n", t->name);

        free(f32);
    }

    poly_bind_index_destroy(idx);
    fprintf(stderr, "poly_qwen3_from_gguf: loaded %d, skipped %d\n",
            loaded, skipped);
    return inst;
}

/* Registry adapter */
PolyInstance *poly_qwen3_from_gguf_decoded_generic(
    const PolyGgufDecoded *gguf,
    const PolyGenericImportOpts *opts)
{
    return poly_qwen3_from_gguf_decoded(gguf,
        opts ? opts->max_batch : 0,
        opts ? opts->max_seq_len : 0);
}
