#ifndef POLY_MODELS_TRANSFORMER_H
#define POLY_MODELS_TRANSFORMER_H

#include "../model.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Causal decoder only (tinygrad llm/model.py), not a base for vision encoders.
 * Successful adoption transfers Model ownership. Failure leaves it with caller.
 * The borrowed model accessor permits ordinary poly_model_* operations.
 * Adopt only trusted producers: signature/bounds checks do not prove causality
 * or the append-only cache semantics required by prefix reuse and rewind. */
typedef struct PolyTransformer PolyTransformer;
bool poly_transformer_available(const PolyModel *model);
PolyTransformer *poly_transformer_from_model(PolyModel *model, PolyModelError *error);
PolyModel *poly_transformer_model(const PolyTransformer *transformer);
void poly_transformer_free(PolyTransformer *transformer);
const PolyModelError *poly_transformer_last_error(const PolyTransformer *transformer);
int poly_transformer_vocab(const PolyTransformer *transformer);
int poly_transformer_position(PolyTransformer *transformer);
int poly_transformer_reset(PolyTransformer *transformer);
int poly_transformer_rewind(PolyTransformer *transformer, int position);
int poly_transformer_append(
    PolyTransformer *transformer,
    const int32_t *tokens,
    int count,
    float *logits,
    int n_logits
);
int poly_transformer_prefill(
    PolyTransformer *transformer,
    const int32_t *tokens,
    int count,
    float *logits,
    int n_logits
);
/* start retains device logits; next returns one sampled token without copying
 * the vocabulary to the host. next: 0 token, 1 capacity reached, -1 error. */
int poly_transformer_start(PolyTransformer *transformer, const int32_t *tokens, int count);
int poly_transformer_next(PolyTransformer *transformer, float temperature, int32_t *token);
#ifdef POLY_TESTING
void poly_transformer_test_fail_after(PolyTransformer *transformer, int calls);
#endif
#ifdef __cplusplus
}
#endif
#endif
