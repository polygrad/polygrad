/*
 * nn.h — Neural network layers
 *
 * Two functions per layer:
 *   poly_X_apply — takes explicit weight UOps, does the math
 *   poly_X       — registers named params on ctx, calls apply
 */

#ifndef POLY_NN_H
#define POLY_NN_H

#include "polygrad.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

void poly_nn_seed(uint32_t seed);

/* Linear: x @ w.T + b */

PolyUOp *poly_linear_apply(PolyCtx *ctx, PolyUOp *x, PolyUOp *w, PolyUOp *b);
PolyUOp *poly_linear(
    PolyCtx *ctx,
    const char *prefix,
    PolyUOp *x,
    int in_features,
    int out_features,
    bool use_bias
);

/* LayerNorm: (x - mean) / sqrt(var + eps), optionally * w + b */

PolyUOp *poly_layernorm_apply(
    PolyCtx *ctx,
    PolyUOp *x,
    PolyUOp *w,
    PolyUOp *b,
    int axis,
    double eps
);
PolyUOp *poly_layernorm(PolyCtx *ctx, const char *prefix, PolyUOp *x, int dim, double eps);

/* RMSNorm: x * rsqrt(mean(x^2) + eps) * w */

PolyUOp *poly_rmsnorm_apply(PolyCtx *ctx, PolyUOp *x, PolyUOp *w, double eps);
PolyUOp *poly_rmsnorm(PolyCtx *ctx, const char *prefix, PolyUOp *x, int dim, double eps);

/* Embedding: gather(table, tokens) */

PolyUOp *poly_embedding_apply(PolyCtx *ctx, PolyUOp *tokens, PolyUOp *table);
PolyUOp *poly_embedding(
    PolyCtx *ctx,
    const char *prefix,
    PolyUOp *tokens,
    int vocab_size,
    int embed_dim
);

/* Transformer building blocks */

/* Causal attention mask: (T, T), 0 where allowed, -1e9 where masked. */
PolyUOp *poly_causal_mask(PolyCtx *ctx, int64_t T);

/* Multi-Head Attention */

/* SDPA: Q @ K.T / sqrt(d) + mask + softmax → @ V. No projections. */
PolyUOp *poly_sdpa(PolyCtx *ctx, PolyUOp *q, PolyUOp *k, PolyUOp *v, PolyUOp *mask, int is_causal);

#ifdef __cplusplus
}
#endif

#endif /* POLY_NN_H */
