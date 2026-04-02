/*
 * nn.c — Neural network convenience builders
 *
 * Thin wrappers: register named params + call frontend.c v2 composed ops.
 * Matches tinygrad nn/__init__.py: Linear, LayerNorm, RMSNorm, Embedding.
 */

#include "nn.h"
#include "frontend.h"
#include "scheduler.h"  /* poly_reshape */
#include <stdint.h>
#include <string.h>
#include <stdio.h>

/* ── RNG ────────────────────────────────────────────────────────────── */

static uint32_t nn_rng_state = 12345;

void poly_nn_seed(uint32_t seed) { nn_rng_state = seed; }

/* ── Convenience builders ───────────────────────────────────────────── */

PolyUOp *poly_nn_linear(PolyCtx *ctx, const char *prefix, PolyUOp *x,
                         int in_features, int out_features, bool use_bias) {
  int64_t ws[] = {out_features, in_features};
  PolyUOp *w = poly_param(ctx, POLY_FLOAT32, ws, 2, "%s.weight", prefix);
  if (!w) return NULL;
  PolyUOp *w_shaped = poly_reshape(ctx, w, ws, 2);

  PolyUOp *b_shaped = NULL;
  if (use_bias) {
    int64_t bs[] = {out_features};
    PolyUOp *b = poly_param(ctx, POLY_FLOAT32, bs, 1, "%s.bias", prefix);
    if (!b) return NULL;
    b_shaped = poly_reshape(ctx, b, bs, 1);
  }

  return poly_linear_v2(ctx, x, w_shaped, b_shaped);
}

PolyUOp *poly_nn_layernorm(PolyCtx *ctx, const char *prefix, PolyUOp *x,
                            int dim, double eps) {
  /* Normalize */
  PolyUOp *normed = poly_layernorm_v2(ctx, x, -1, eps);
  if (!normed) return NULL;

  /* Affine: weight * normed + bias */
  int64_t ds[] = {dim};
  PolyUOp *w = poly_param(ctx, POLY_FLOAT32, ds, 1, "%s.weight", prefix);
  PolyUOp *b = poly_param(ctx, POLY_FLOAT32, ds, 1, "%s.bias", prefix);
  if (!w || !b) return NULL;

  PolyUOp *w_shaped = poly_reshape(ctx, w, ds, 1);
  PolyUOp *b_shaped = poly_reshape(ctx, b, ds, 1);

  PolyUOp *scaled = poly_alu2(ctx, POLY_OP_MUL, normed, w_shaped);
  return poly_alu2(ctx, POLY_OP_ADD, scaled, b_shaped);
}

PolyUOp *poly_nn_rmsnorm(PolyCtx *ctx, const char *prefix, PolyUOp *x,
                          int dim, double eps) {
  int64_t ds[] = {dim};
  PolyUOp *w = poly_param(ctx, POLY_FLOAT32, ds, 1, "%s.weight", prefix);
  if (!w) return NULL;
  PolyUOp *w_shaped = poly_reshape(ctx, w, ds, 1);

  return poly_rmsnorm_v2(ctx, x, w_shaped, eps);
}

PolyUOp *poly_nn_embedding(PolyCtx *ctx, const char *prefix, PolyUOp *tokens,
                            int vocab_size, int embed_dim) {
  int64_t ws[] = {vocab_size, embed_dim};
  PolyUOp *w = poly_param(ctx, POLY_FLOAT32, ws, 2, "%s.weight", prefix);
  if (!w) return NULL;
  PolyUOp *w_shaped = poly_reshape(ctx, w, ws, 2);

  return poly_gather_v2(ctx, w_shaped, tokens);
}
