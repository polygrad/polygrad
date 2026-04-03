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
#include "tensor.h"     /* poly_mean_reduce */
#include "scheduler.h"  /* poly_reshape, poly_permute, poly_expand */
#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* ── RNG ────────────────────────────────────────────────────────────── */

static uint32_t nn_rng_state = 12345;

void poly_nn_seed(uint32_t seed) { nn_rng_state = seed; }

/* ── Linear ─────────────────────────────────────────────────────────── */

PolyUOp *poly_linear_apply(
    PolyCtx *ctx,
    PolyUOp *x,
    PolyUOp *w,
    PolyUOp *b
) {
  if (!ctx || !x || !w) return NULL;
  int64_t perm[] = {1, 0};
  PolyUOp *out = poly_dot(ctx, x, poly_permute(ctx, w, perm, 2));
  if (!out) return NULL;
  if (b) out = poly_add(ctx, out, b);
  return out;
}

PolyUOp *poly_linear(
    PolyCtx *ctx,
    const char *prefix,
    PolyUOp *x,
    int in_features,
    int out_features,
    bool use_bias
) {
  int64_t ws[] = {out_features, in_features};
  PolyUOp *w = poly_param(ctx, POLY_FLOAT32, ws, 2, "%s.weight", prefix);
  if (!w) return NULL;
  w = poly_reshape(ctx, w, ws, 2);

  PolyUOp *b = NULL;
  if (use_bias) {
    int64_t bs[] = {out_features};
    b = poly_param(ctx, POLY_FLOAT32, bs, 1, "%s.bias", prefix);
    if (!b) return NULL;
  }

  return poly_linear_apply(ctx, x, w, b);
}

/* ── LayerNorm ──────────────────────────────────────────────────────── */

PolyUOp *poly_layernorm_apply(
    PolyCtx *ctx,
    PolyUOp *x,
    PolyUOp *w,
    PolyUOp *b,
    int axis,
    double eps
) {
  if (!ctx || !x) return NULL;

  PolyUOp *mean = poly_mean_reduce(ctx, x, axis, 1);
  PolyUOp *centered = poly_sub(ctx, x, mean);
  PolyUOp *sq = poly_alu2(ctx, POLY_OP_MUL, centered, centered);
  PolyUOp *var = poly_mean_reduce(ctx, sq, axis, 1);
  PolyUOp *denom = poly_alu1(
      ctx, POLY_OP_SQRT,
      poly_add(ctx, var, poly_const_float(ctx, eps))
  );
  PolyUOp *normed = poly_mul(
      ctx, centered,
      poly_alu1(ctx, POLY_OP_RECIPROCAL, denom)
  );

  if (w) normed = poly_mul(ctx, normed, w);
  if (b) normed = poly_add(ctx, normed, b);

  return normed;
}

PolyUOp *poly_layernorm(
    PolyCtx *ctx,
    const char *prefix,
    PolyUOp *x,
    int dim,
    double eps
) {
  int64_t ds[] = {dim};
  PolyUOp *w = poly_param(ctx, POLY_FLOAT32, ds, 1, "%s.weight", prefix);
  PolyUOp *b = poly_param(ctx, POLY_FLOAT32, ds, 1, "%s.bias", prefix);
  if (!w || !b) return NULL;

  return poly_layernorm_apply(
      ctx, x,
      poly_reshape(ctx, w, ds, 1),
      poly_reshape(ctx, b, ds, 1),
      -1, eps
  );
}

/* ── RMSNorm ────────────────────────────────────────────────────────── */

PolyUOp *poly_rmsnorm_apply(
    PolyCtx *ctx,
    PolyUOp *x,
    PolyUOp *w,
    double eps
) {
  if (!ctx || !x) return NULL;
  int ndim = poly_uop_ndim(ctx, x);
  if (ndim < 1) return NULL;
  const int64_t *dims = poly_uop_dims(ctx, x);
  if (!dims) return NULL;
  int64_t shape[POLY_MAX_DIMS];
  memcpy(shape, dims, ndim * sizeof(int64_t));
  int axis = ndim - 1;

  PolyUOp *x2 = poly_alu2(ctx, POLY_OP_MUL, x, x);
  PolyUOp *m = poly_mean_reduce(ctx, x2, axis, 1);
  m = poly_alu2(ctx, POLY_OP_ADD, m, poly_const_float(ctx, eps));
  PolyUOp *rrms = poly_alu1(ctx, POLY_OP_SQRT, m);
  rrms = poly_alu2(ctx, POLY_OP_FDIV, poly_const_float(ctx, 1.0), rrms);
  rrms = poly_expand(ctx, rrms, shape, ndim);
  PolyUOp *normed = poly_alu2(ctx, POLY_OP_MUL, x, rrms);

  if (w) {
    int w_ndim = poly_uop_ndim(ctx, w);
    if (w_ndim == 1) {
      const int64_t *w_dims = poly_uop_dims(ctx, w);
      int64_t bc[POLY_MAX_DIMS];
      for (int i = 0; i < ndim - 1; i++) bc[i] = 1;
      bc[ndim - 1] = w_dims ? w_dims[0] : 0;
      PolyUOp *w_r = poly_reshape(ctx, w, bc, ndim);
      PolyUOp *w_e = poly_expand(ctx, w_r, shape, ndim);
      normed = poly_alu2(ctx, POLY_OP_MUL, normed, w_e);
    }
  }
  return normed;
}

PolyUOp *poly_rmsnorm(
    PolyCtx *ctx,
    const char *prefix,
    PolyUOp *x,
    int dim,
    double eps
) {
  int64_t ds[] = {dim};
  PolyUOp *w = poly_param(ctx, POLY_FLOAT32, ds, 1, "%s.weight", prefix);
  if (!w) return NULL;
  return poly_rmsnorm_apply(ctx, x, poly_reshape(ctx, w, ds, 1), eps);
}

/* ── Embedding ──────────────────────────────────────────────────────── */

PolyUOp *poly_embedding_apply(
    PolyCtx *ctx,
    PolyUOp *tokens,
    PolyUOp *table
) {
  return poly_gather(ctx, table, tokens);
}

PolyUOp *poly_embedding(
    PolyCtx *ctx,
    const char *prefix,
    PolyUOp *tokens,
    int vocab_size,
    int embed_dim
) {
  int64_t ws[] = {vocab_size, embed_dim};
  PolyUOp *w = poly_param(ctx, POLY_FLOAT32, ws, 2, "%s.weight", prefix);
  if (!w) return NULL;
  return poly_embedding_apply(ctx, tokens, poly_reshape(ctx, w, ws, 2));
}

/* ── Causal attention mask ──────────────────────────────────────────── */

PolyUOp *poly_causal_mask(PolyCtx *ctx, int64_t T) {
  if (!ctx || T <= 0) return NULL;

  PolyUOp *arange_buf = poly_arange(ctx, 0.0, (double)T, 1.0);
  int64_t row_shape[] = { T, 1 };
  PolyUOp *row = poly_expand(ctx, poly_reshape(ctx, arange_buf, row_shape, 2),
                              (int64_t[]){ T, T }, 2);
  int64_t col_shape[] = { 1, T };
  PolyUOp *col = poly_expand(ctx, poly_reshape(ctx, arange_buf, col_shape, 2),
                              (int64_t[]){ T, T }, 2);

  PolyUOp *mask = poly_alu2(ctx, POLY_OP_CMPLT, row, col);
  return poly_where_op(
      ctx, mask,
      poly_const_float(ctx, -1e9),
      poly_const_float(ctx, 0.0)
  );
}
