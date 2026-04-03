/*
 * nn.c — Neural network layers
 *
 * Two functions per layer:
 *   poly_X_apply(ctx, x, w, b, ...) — the math, takes explicit weight UOps
 *   poly_X(ctx, prefix, x, ...)     — registers named params, calls apply
 *
 * All math uses shape-on-UOp (no explicit shape parameters).
 */

#include "nn.h"
#include "frontend.h"
#include "scheduler.h"  /* poly_reshape, poly_permute, poly_expand */
#include <stdint.h>
#include <stdio.h>

/* ── RNG ────────────────────────────────────────────────────────────── */

static uint32_t nn_rng_state = 12345;

void poly_nn_seed(uint32_t seed) { nn_rng_state = seed; }

/* ── Linear ─────────────────────────────────────────────────────────── */

PolyUOp *poly_linear_apply(PolyCtx *ctx, PolyUOp *x, PolyUOp *w, PolyUOp *b) {
  if (!ctx || !x || !w) return NULL;
  /* w is (out_features, in_features), transpose to (in, out) for dot */
  int64_t perm[] = {1, 0};
  PolyUOp *wt = poly_permute(ctx, w, perm, 2);
  PolyUOp *out = poly_dot_v2(ctx, x, wt);
  if (!out) return NULL;
  if (b) out = poly_alu2(ctx, POLY_OP_ADD, out, b);
  return out;
}

PolyUOp *poly_linear(PolyCtx *ctx, const char *prefix, PolyUOp *x,
                      int in_features, int out_features, bool use_bias) {
  int64_t ws[] = {out_features, in_features};
  PolyUOp *w = poly_param(ctx, POLY_FLOAT32, ws, 2, "%s.weight", prefix);
  if (!w) return NULL;

  PolyUOp *b = NULL;
  if (use_bias) {
    int64_t bs[] = {out_features};
    b = poly_param(ctx, POLY_FLOAT32, bs, 1, "%s.bias", prefix);
    if (!b) return NULL;
    b = poly_reshape(ctx, b, bs, 1);
  }

  return poly_linear_apply(ctx, x, poly_reshape(ctx, w, ws, 2), b);
}

/* ── LayerNorm ──────────────────────────────────────────────────────── */

PolyUOp *poly_layernorm_apply(PolyCtx *ctx, PolyUOp *x,
                               PolyUOp *w, PolyUOp *b,
                               int axis, double eps) {
  if (!ctx || !x) return NULL;

  /* Normalize: (x - mean) / sqrt(var + eps) */
  PolyUOp *mean = poly_mean_reduce_v2(ctx, x, axis, 1);
  PolyUOp *centered = poly_alu2(ctx, POLY_OP_ADD, x,
                                 poly_alu1(ctx, POLY_OP_NEG, mean));
  PolyUOp *sq = poly_alu2(ctx, POLY_OP_MUL, centered, centered);
  PolyUOp *var = poly_mean_reduce_v2(ctx, sq, axis, 1);
  PolyUOp *eps_c = poly_const_float(ctx, eps);
  PolyUOp *denom = poly_alu1(ctx, POLY_OP_SQRT,
                              poly_alu2(ctx, POLY_OP_ADD, var, eps_c));
  PolyUOp *normed = poly_alu2(ctx, POLY_OP_MUL, centered,
                               poly_alu1(ctx, POLY_OP_RECIPROCAL, denom));

  /* Affine: w * normed + b (skip if w is NULL) */
  if (w) normed = poly_alu2(ctx, POLY_OP_MUL, normed, w);
  if (b) normed = poly_alu2(ctx, POLY_OP_ADD, normed, b);

  return normed;
}

PolyUOp *poly_layernorm(PolyCtx *ctx, const char *prefix, PolyUOp *x,
                         int dim, double eps) {
  int64_t ds[] = {dim};
  PolyUOp *w = poly_param(ctx, POLY_FLOAT32, ds, 1, "%s.weight", prefix);
  PolyUOp *b = poly_param(ctx, POLY_FLOAT32, ds, 1, "%s.bias", prefix);
  if (!w || !b) return NULL;

  return poly_layernorm_apply(ctx, x,
                               poly_reshape(ctx, w, ds, 1),
                               poly_reshape(ctx, b, ds, 1),
                               -1, eps);
}

/* ── RMSNorm ────────────────────────────────────────────────────────── */

PolyUOp *poly_rmsnorm_apply(PolyCtx *ctx, PolyUOp *x, PolyUOp *w, double eps) {
  return poly_rmsnorm_v2(ctx, x, w, eps);
}

PolyUOp *poly_rmsnorm(PolyCtx *ctx, const char *prefix, PolyUOp *x,
                       int dim, double eps) {
  int64_t ds[] = {dim};
  PolyUOp *w = poly_param(ctx, POLY_FLOAT32, ds, 1, "%s.weight", prefix);
  if (!w) return NULL;
  return poly_rmsnorm_apply(ctx, x, poly_reshape(ctx, w, ds, 1), eps);
}

/* ── Embedding ──────────────────────────────────────────────────────── */

PolyUOp *poly_embedding_apply(PolyCtx *ctx, PolyUOp *tokens, PolyUOp *table) {
  return poly_gather_v2(ctx, table, tokens);
}

PolyUOp *poly_embedding(PolyCtx *ctx, const char *prefix, PolyUOp *tokens,
                         int vocab_size, int embed_dim) {
  int64_t ws[] = {vocab_size, embed_dim};
  PolyUOp *w = poly_param(ctx, POLY_FLOAT32, ws, 2, "%s.weight", prefix);
  if (!w) return NULL;
  return poly_embedding_apply(ctx, tokens, poly_reshape(ctx, w, ws, 2));
}
