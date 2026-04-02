/*
 * nn.h — Neural network convenience builders for polygrad
 *
 * Thin wrappers that register named parameters on PolyCtx via the
 * buffer registry (poly_param) and build computation graphs using
 * existing frontend.c v2 composed ops.
 *
 * Pattern: poly_nn_linear(ctx, "layers.0", x, 768, 3072)
 *   1. Registers "layers.0.weight" and "layers.0.bias" via poly_param
 *   2. Builds x @ weight.T + bias via poly_linear_v2
 *   3. Returns the output UOp
 *
 * Matches tinygrad nn/__init__.py: Linear, LayerNorm, RMSNorm, Embedding
 * are Python classes that create Tensor params + build graphs.
 * Here they're C functions that do both in one call.
 */

#ifndef POLY_NN_H
#define POLY_NN_H

#include "polygrad.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── RNG seed ───────────────────────────────────────────────────────── */

void poly_nn_seed(uint32_t seed);

/* ── Convenience builders ───────────────────────────────────────────── */

/* Linear: registers {prefix}.weight (out_features, in_features)
 *                   {prefix}.bias   (out_features) if use_bias
 * Returns: x @ weight.T + bias  (poly_linear_v2) */
PolyUOp *poly_nn_linear(PolyCtx *ctx, const char *prefix, PolyUOp *x,
                         int in_features, int out_features, bool use_bias);

/* LayerNorm: registers {prefix}.weight (dim)
 *                      {prefix}.bias   (dim)
 * Returns: layernorm(x) * weight + bias  (poly_layernorm_v2 + alu) */
PolyUOp *poly_nn_layernorm(PolyCtx *ctx, const char *prefix, PolyUOp *x,
                            int dim, double eps);

/* RMSNorm: registers {prefix}.weight (dim)
 * Returns: rmsnorm(x, weight)  (poly_rmsnorm_v2) */
PolyUOp *poly_nn_rmsnorm(PolyCtx *ctx, const char *prefix, PolyUOp *x,
                          int dim, double eps);

/* Embedding: registers {prefix}.weight (vocab_size, embed_dim)
 * Returns: gather(weight, tokens)  (poly_gather_v2) */
PolyUOp *poly_nn_embedding(PolyCtx *ctx, const char *prefix, PolyUOp *tokens,
                            int vocab_size, int embed_dim);

#ifdef __cplusplus
}
#endif

#endif /* POLY_NN_H */
