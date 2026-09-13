#ifndef POLY_MODEL_LAYERS_H
#define POLY_MODEL_LAYERS_H
#include "model.h"
#include "nn/nn.h"
#ifdef __cplusplus
extern "C" {
#endif

PolyUOp *poly_linear(
    PolyCtx *ctx,
    const char *prefix,
    PolyUOp *x,
    int in_features,
    int out_features,
    bool use_bias
) POLY_DEPRECATED("use poly_model_linear or poly_linear_apply with explicit params");

PolyUOp *poly_layernorm(PolyCtx *ctx, const char *prefix, PolyUOp *x, int dim, double eps)
    POLY_DEPRECATED("use poly_model_layernorm or poly_layernorm_apply with explicit params");

PolyUOp *poly_rmsnorm(PolyCtx *ctx, const char *prefix, PolyUOp *x, int dim, double eps)
    POLY_DEPRECATED("use poly_model_rmsnorm or poly_rmsnorm_apply with explicit params");

PolyUOp *poly_embedding(
    PolyCtx *ctx,
    const char *prefix,
    PolyUOp *tokens,
    int vocab_size,
    int embed_dim
) POLY_DEPRECATED("use poly_model_embedding or poly_embedding_apply with explicit params");

PolyTensor *poly_model_linear(
    PolyModel *inst,
    const char *prefix,
    PolyTensor *x,
    int in_features,
    int out_features,
    bool use_bias
);

PolyTensor *poly_model_layernorm(
    PolyModel *inst,
    const char *prefix,
    PolyTensor *x,
    int dim,
    double eps
);

PolyTensor *poly_model_rmsnorm(
    PolyModel *inst,
    const char *prefix,
    PolyTensor *x,
    int dim,
    double eps
);

PolyTensor *poly_model_embedding(
    PolyModel *inst,
    const char *prefix,
    PolyTensor *tokens,
    int vocab_size,
    int embed_dim
);

/* Create scoped parameter bindings, then invoke the shared NN cell.
 * As with linear, the Model builder/weight importer owns initialization. */
int poly_model_lstm_cell(
    PolyModel *model,
    const char *prefix,
    PolyTensor *x,
    PolyTensor *h,
    PolyTensor *c,
    int input_size,
    int hidden_size,
    bool bias,
    PolyTensor **new_h,
    PolyTensor **new_c
);

#ifdef __cplusplus
}
#endif
#endif
