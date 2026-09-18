#ifndef POLY_MODEL_LAYERS_H
#define POLY_MODEL_LAYERS_H
#include "model.h"
#include "nn/nn.h"
#ifdef __cplusplus
extern "C" {
#endif

/* Seed/name-keyed SplitMix64 Kaiming uniform, bound = sqrt(6/fan_in). */
void poly_init_param_kaiming(
    uint64_t seed,
    const char *name,
    float *data,
    int64_t numel,
    int64_t fan_in
);

PolyUOp *poly_linear(
    PolyCtx *ctx,
    const char *prefix,
    PolyUOp *x,
    int in_features,
    int out_features,
    bool use_bias
) POLY_DEPRECATED("use poly_model_linear or poly_uop_linear_apply with explicit params");

PolyUOp *poly_layernorm(PolyCtx *ctx, const char *prefix, PolyUOp *x, int dim, double eps)
    POLY_DEPRECATED("use poly_model_layernorm or poly_uop_layernorm_apply with explicit params");

PolyUOp *poly_rmsnorm(PolyCtx *ctx, const char *prefix, PolyUOp *x, int dim, double eps)
    POLY_DEPRECATED("use poly_model_rmsnorm or poly_uop_rmsnorm_apply with explicit params");

PolyUOp *poly_embedding(
    PolyCtx *ctx,
    const char *prefix,
    PolyUOp *tokens,
    int vocab_size,
    int embed_dim
) POLY_DEPRECATED("use poly_model_embedding or poly_uop_embedding_apply with explicit params");

PolyTensor *poly_model_linear(
    PolyModel *inst,
    const char *prefix,
    PolyTensor *x,
    int in_features,
    int out_features,
    bool use_bias
);

/* Declare one Linear's state for callers that reuse it across several calls.
 * The caller owns the returned Tensor handles; initialization stays with the
 * family/checkpoint loader. Same (out,in) convention as tinygrad.nn.Linear. */
int poly_model_linear_parameters(
    PolyModel *model,
    const char *prefix,
    int in_features,
    int out_features,
    bool use_bias,
    PolyTensor **weight,
    PolyTensor **bias
);

int poly_model_norm_parameters(
    PolyModel *model,
    const char *prefix,
    int dim,
    bool use_bias,
    PolyTensor **weight,
    PolyTensor **bias
);
PolyTensor *poly_model_embedding_parameters(
    PolyModel *model,
    const char *prefix,
    int vocab_size,
    int embed_dim
);

/* Fixed sequence table generation shared by named families and JSON RoPE.
 * factor==1 is pinned Llama RoPE; other factors use the existing Llama3 extension. */
typedef struct {
  int length, dim;
  double theta, factor, low_freq, high_freq, original_context;
} PolyModelRoPEConfig;
PolyTensor *poly_model_rope_frequencies(
    PolyModel *model,
    const char *name,
    const PolyModelRoPEConfig *config,
    bool sine
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
