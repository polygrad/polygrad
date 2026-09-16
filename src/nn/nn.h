/* nn.h — Reusable layer programs with caller-owned parameters.
 * Model parameter construction belongs to models/layers.h. */

#ifndef POLY_NN_H
#define POLY_NN_H

#include "polygrad.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Linear: x @ w.T + b */

PolyUOp *poly_linear_apply(PolyCtx *ctx, PolyUOp *x, PolyUOp *w, PolyUOp *b);
PolyTensor *poly_tensor_linear_apply(PolyCtx *ctx, PolyTensor *x, PolyTensor *w, PolyTensor *b);

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
PolyUOp *poly_layernorm_axes_apply(
    PolyCtx *ctx,
    PolyUOp *x,
    PolyUOp *w,
    PolyUOp *b,
    const int64_t *axes,
    int n_axes,
    double eps
);
PolyTensor *poly_tensor_layernorm_axes_apply(
    PolyCtx *ctx,
    PolyTensor *x,
    PolyTensor *w,
    PolyTensor *b,
    const int64_t *axes,
    int n_axes,
    double eps
);
PolyUOp *poly_groupnorm_apply(
    PolyCtx *ctx,
    PolyUOp *x,
    PolyUOp *w,
    PolyUOp *b,
    int groups,
    double eps
);
PolyTensor *poly_tensor_groupnorm_apply(
    PolyCtx *ctx,
    PolyTensor *x,
    PolyTensor *w,
    PolyTensor *b,
    int groups,
    double eps
);
/* nn.BatchNorm owns no hidden state: running buffers are supplied by the
 * frontend or Model builder. NULL mean/variance select untracked statistics. */
int poly_tensor_batchnorm_stats(
    PolyCtx *ctx,
    PolyTensor *x,
    PolyTensor *running_mean,
    PolyTensor *running_var,
    bool training,
    PolyTensor **mean,
    PolyTensor **var
);
PolyTensor *poly_tensor_batchnorm_apply(
    PolyCtx *ctx,
    PolyTensor *x,
    PolyTensor *w,
    PolyTensor *b,
    PolyTensor *running_mean,
    PolyTensor *running_var,
    PolyTensor *num_batches,
    bool training,
    double eps,
    double momentum
);

/* RMSNorm: x * rsqrt(mean(x^2) + eps) * w */

PolyUOp *poly_rmsnorm_apply(PolyCtx *ctx, PolyUOp *x, PolyUOp *w, double eps);
PolyTensor *poly_tensor_rmsnorm_apply(PolyCtx *ctx, PolyTensor *x, PolyTensor *w, double eps);
PolyUOp *poly_instancenorm_apply(
    PolyCtx *ctx,
    PolyUOp *x,
    PolyUOp *w,
    PolyUOp *b,
    int num_features,
    double eps
);
PolyTensor *poly_tensor_instancenorm_apply(
    PolyCtx *ctx,
    PolyTensor *x,
    PolyTensor *w,
    PolyTensor *b,
    int num_features,
    double eps
);

/* Embedding: gather(table, tokens) */

PolyUOp *poly_embedding_apply(PolyCtx *ctx, PolyUOp *tokens, PolyUOp *table);
PolyTensor *poly_tensor_embedding_apply(PolyCtx *ctx, PolyTensor *tokens, PolyTensor *table);

/* Transformer building blocks */

/* nn.LSTMCell: explicit caller-owned weights/state; both state inputs may be
 * NULL for a zero initial state. Successful outputs are caller-owned handles. */
int poly_lstm_cell(
    PolyCtx *ctx,
    PolyUOp *x,
    PolyUOp *h,
    PolyUOp *c,
    PolyUOp *weight_ih,
    PolyUOp *weight_hh,
    PolyUOp *bias_ih,
    PolyUOp *bias_hh,
    PolyUOp **new_h,
    PolyUOp **new_c
);
int poly_tensor_lstm_cell(
    PolyCtx *ctx,
    PolyTensor *x,
    PolyTensor *h,
    PolyTensor *c,
    PolyTensor *weight_ih,
    PolyTensor *weight_hh,
    PolyTensor *bias_ih,
    PolyTensor *bias_hh,
    PolyTensor **new_h,
    PolyTensor **new_c
);

/* Causal attention mask: (T, T), 0 where allowed, -1e9 where masked. */
PolyUOp *poly_causal_mask(PolyCtx *ctx, int64_t T);
PolyTensor *poly_tensor_causal_mask(PolyCtx *ctx, int64_t T);

/* Multi-Head Attention */

/* RandMixin.dropout. Returns an owned reference, including identity results. */
PolyTensor *poly_tensor_dropout(PolyCtx *ctx, PolyTensor *x, double p, int training);
/* Inference SDPA; Tensor SDPA additionally owns training dropout/RNG. */
PolyUOp *poly_sdpa(
    PolyCtx *ctx,
    PolyUOp *q,
    PolyUOp *k,
    PolyUOp *v,
    PolyUOp *mask,
    int is_causal,
    int enable_gqa
);
PolyTensor *poly_tensor_sdpa(
    PolyCtx *ctx,
    PolyTensor *q,
    PolyTensor *k,
    PolyTensor *v,
    PolyTensor *mask,
    double dropout_p,
    int is_causal,
    int enable_gqa,
    int training
);

#ifdef __cplusplus
}
#endif

#endif /* POLY_NN_H */
