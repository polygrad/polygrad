/*
 * nn.h — Neural network layers
 *
 * Layer boundaries exist at two explicit levels:
 *   poly_X_apply         — low-level raw-UOp program
 *   poly_tensor_X_apply  — the same program over paired Tensor roots
 *   poly_instance_X      — declares paired Instance params, returns a Tensor
 *   poly_X               — legacy ctx-global raw-UOp registration
 */

#ifndef POLY_NN_H
#define POLY_NN_H

#include "polygrad.h"
#include <stdbool.h>

typedef struct PolyInstance PolyInstance;

#ifdef __cplusplus
extern "C" {
#endif

void poly_nn_seed(uint32_t seed);

/* Linear: x @ w.T + b */

PolyUOp *poly_linear_apply(PolyCtx *ctx, PolyUOp *x, PolyUOp *w, PolyUOp *b);
PolyTensor *poly_tensor_linear_apply(
    PolyCtx *ctx,
    PolyTensor *x,
    PolyTensor *w,
    PolyTensor *b
);
PolyUOp *poly_linear(
    PolyCtx *ctx,
    const char *prefix,
    PolyUOp *x,
    int in_features,
    int out_features,
    bool use_bias
) POLY_DEPRECATED("use poly_instance_linear or poly_linear_apply with explicit params");
PolyTensor *poly_instance_linear(
    PolyInstance *inst,
    const char *prefix,
    PolyTensor *x,
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
PolyTensor *poly_tensor_layernorm_apply(
    PolyCtx *ctx,
    PolyTensor *x,
    PolyTensor *w,
    PolyTensor *b,
    int axis,
    double eps
);
PolyUOp *poly_layernorm(PolyCtx *ctx, const char *prefix, PolyUOp *x, int dim, double eps)
    POLY_DEPRECATED("use poly_instance_layernorm or poly_layernorm_apply with explicit params");
PolyTensor *poly_instance_layernorm(
    PolyInstance *inst,
    const char *prefix,
    PolyTensor *x,
    int dim,
    double eps
);

/* RMSNorm: x * rsqrt(mean(x^2) + eps) * w */

PolyUOp *poly_rmsnorm_apply(PolyCtx *ctx, PolyUOp *x, PolyUOp *w, double eps);
PolyTensor *poly_tensor_rmsnorm_apply(
    PolyCtx *ctx,
    PolyTensor *x,
    PolyTensor *w,
    double eps
);
PolyUOp *poly_rmsnorm(PolyCtx *ctx, const char *prefix, PolyUOp *x, int dim, double eps)
    POLY_DEPRECATED("use poly_instance_rmsnorm or poly_rmsnorm_apply with explicit params");
PolyTensor *poly_instance_rmsnorm(
    PolyInstance *inst,
    const char *prefix,
    PolyTensor *x,
    int dim,
    double eps
);

/* Embedding: gather(table, tokens) */

PolyUOp *poly_embedding_apply(PolyCtx *ctx, PolyUOp *tokens, PolyUOp *table);
PolyTensor *poly_tensor_embedding_apply(
    PolyCtx *ctx,
    PolyTensor *tokens,
    PolyTensor *table
);
PolyUOp *poly_embedding(
    PolyCtx *ctx,
    const char *prefix,
    PolyUOp *tokens,
    int vocab_size,
    int embed_dim
) POLY_DEPRECATED("use poly_instance_embedding or poly_embedding_apply with explicit params");
PolyTensor *poly_instance_embedding(
    PolyInstance *inst,
    const char *prefix,
    PolyTensor *tokens,
    int vocab_size,
    int embed_dim
);

/* Transformer building blocks */

/* Causal attention mask: (T, T), 0 where allowed, -1e9 where masked. */
PolyUOp *poly_causal_mask(PolyCtx *ctx, int64_t T);
PolyTensor *poly_tensor_causal_mask(PolyCtx *ctx, int64_t T);

/* Multi-Head Attention */

/* SDPA: Q @ K.T / sqrt(d) + mask + softmax → @ V. No projections. */
PolyUOp *poly_sdpa(PolyCtx *ctx, PolyUOp *q, PolyUOp *k, PolyUOp *v, PolyUOp *mask, int is_causal);
PolyTensor *poly_tensor_sdpa(
    PolyCtx *ctx,
    PolyTensor *q,
    PolyTensor *k,
    PolyTensor *v,
    PolyTensor *mask,
    int is_causal
);

#ifdef __cplusplus
}
#endif

#endif /* POLY_NN_H */
