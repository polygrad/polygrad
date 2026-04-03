/*
 * gpt2.c -- GPT-2 model builder
 *
 * Uses the nn.h layer API (poly_linear, poly_layernorm, poly_embedding)
 * and the named buffer registry (poly_param/poly_input/poly_output).
 *
 * Weight naming matches HuggingFace GPT-2 (minus "transformer." prefix).
 * Weights stored in PyTorch convention (out, in). The HF loader transposes
 * Conv1D weights (in, out) -> (out, in) during import.
 */

#define _POSIX_C_SOURCE 200809L
#include "models.h"
#include "../nn.h"
#include "../tensor.h"
#include "../instance.h"
#include "../frontend.h"
#include "../scheduler.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#define REALIZE(u) poly_uop1(ctx, POLY_OP_CONTIGUOUS, (u)->dtype, (u), poly_arg_none())


/* ── GPT-2 Builder ───────────────────────────────────────────────── */

PolyInstance *poly_gpt2_build(const GPT2Config *cfg, int max_batch) {
  if (!cfg || cfg->n_layer < 1 || cfg->n_embd < 1 || cfg->vocab_size < 1)
    return NULL;

  int V = cfg->vocab_size;
  int D = cfg->n_embd;
  int H = cfg->n_head;
  int L = cfg->n_layer;
  int T = cfg->max_seq_len;
  int B = max_batch > 0 ? max_batch : 1;
  int head_dim = D / H;
  double eps = cfg->norm_eps > 0 ? (double)cfg->norm_eps : 1e-5;

  if (D % H != 0) {
    fprintf(stderr, "poly_gpt2_build: n_embd (%d) not divisible by n_head (%d)\n", D, H);
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
  tok_emb = REALIZE(tok_emb);

  PolyUOp *pos_shaped = poly_reshape(ctx, pos_buf, pos_shape, 2);
  PolyUOp *pos_emb = poly_embedding(ctx, "wpe", pos_shaped, T, D);
  pos_emb = REALIZE(pos_emb);

  int64_t h_shape[] = { B, T, D };
  PolyUOp *pos_exp = poly_expand(ctx, pos_emb, h_shape, 3);
  PolyUOp *h = poly_alu2(ctx, POLY_OP_ADD, tok_emb, pos_exp);
  h = REALIZE(h);

  /* Causal mask: (T, T) -> (1, 1, T, T) */
  PolyUOp *mask = REALIZE(poly_reshape(ctx, poly_causal_mask(ctx, T),
                                        (int64_t[]){ 1, 1, T, T }, 4));

  /* ── Transformer blocks ──────────────────────────────────────── */

  for (int i = 0; i < L; i++) {
    char prefix[64];

    /* LayerNorm 1 */
    snprintf(prefix, sizeof(prefix), "h.%d.ln_1", i);
    PolyUOp *ln1 = REALIZE(poly_layernorm(ctx, prefix, h, D, eps));

    /* QKV = Linear(D, 3D) */
    snprintf(prefix, sizeof(prefix), "h.%d.attn.c_attn", i);
    PolyUOp *qkv = REALIZE(poly_linear(ctx, prefix, ln1, D, 3 * D, true));

    /* Split Q, K, V via shrink */
    int64_t shrink_q[][2] = { {0, B}, {0, T}, {0, D} };
    int64_t shrink_k[][2] = { {0, B}, {0, T}, {D, 2*D} };
    int64_t shrink_v[][2] = { {0, B}, {0, T}, {2*D, 3*D} };
    PolyUOp *q = REALIZE(poly_shrink(ctx, qkv, shrink_q, 3));
    PolyUOp *k = REALIZE(poly_shrink(ctx, qkv, shrink_k, 3));
    PolyUOp *v = REALIZE(poly_shrink(ctx, qkv, shrink_v, 3));

    /* Multi-head reshape + permute: (B,T,D) -> (B,H,T,hd) */
    int64_t mh[] = { B, T, H, head_dim };
    int64_t perm[] = { 0, 2, 1, 3 };
    q = poly_permute(ctx, poly_reshape(ctx, q, mh, 4), perm, 4);
    k = poly_permute(ctx, poly_reshape(ctx, k, mh, 4), perm, 4);
    v = poly_permute(ctx, poly_reshape(ctx, v, mh, 4), perm, 4);

    /* scores = Q @ K.T / sqrt(hd) + mask */
    PolyUOp *kt = poly_permute(ctx, k, (int64_t[]){ 0, 1, 3, 2 }, 4);
    PolyUOp *scores = REALIZE(poly_dot(ctx, q, kt));
    scores = poly_alu2(
        ctx, POLY_OP_MUL, scores,
        poly_const_float(ctx, 1.0 / sqrt((double)head_dim))
    );
    PolyUOp *mask_exp = poly_expand(ctx, mask, (int64_t[]){ B, H, T, T }, 4);
    scores = REALIZE(poly_alu2(ctx, POLY_OP_ADD, scores, mask_exp));

    /* softmax -> attn @ V */
    PolyUOp *attn = REALIZE(poly_softmax(ctx, scores, -1));
    PolyUOp *attn_out = REALIZE(poly_dot(ctx, attn, v));

    /* Merge heads: (B,H,T,hd) -> (B,T,D) */
    attn_out = poly_reshape(
        ctx,
        poly_permute(ctx, attn_out, (int64_t[]){ 0, 2, 1, 3 }, 4),
        (int64_t[]){ B, T, D }, 3
    );

    /* Output projection + residual */
    snprintf(prefix, sizeof(prefix), "h.%d.attn.c_proj", i);
    attn_out = REALIZE(poly_linear(ctx, prefix, attn_out, D, D, true));
    h = REALIZE(poly_alu2(ctx, POLY_OP_ADD, h, attn_out));

    /* LayerNorm 2 + FFN + residual */
    snprintf(prefix, sizeof(prefix), "h.%d.ln_2", i);
    PolyUOp *ln2 = REALIZE(poly_layernorm(ctx, prefix, h, D, eps));

    snprintf(prefix, sizeof(prefix), "h.%d.mlp.c_fc", i);
    PolyUOp *ffn = REALIZE(poly_linear(ctx, prefix, ln2, D, 4 * D, true));
    ffn = REALIZE(poly_gelu(ctx, ffn));

    snprintf(prefix, sizeof(prefix), "h.%d.mlp.c_proj", i);
    ffn = REALIZE(poly_linear(ctx, prefix, ffn, 4 * D, D, true));
    h = REALIZE(poly_alu2(ctx, POLY_OP_ADD, h, ffn));
  }

  /* Final layernorm */
  h = REALIZE(poly_layernorm(ctx, "ln_f", h, D, eps));

  /* LM head: h @ wte.T (weight tying) */
  PolyUOp *wte = poly_ctx_get(ctx, "wte.weight");
  int64_t wte_shape[] = { V, D };
  PolyUOp *wte_2d = poly_reshape(ctx, wte, wte_shape, 2);
  int64_t perm_t[] = { 1, 0 };
  PolyUOp *logits = poly_dot(ctx, h, poly_permute(ctx, wte_2d, perm_t, 2));

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
