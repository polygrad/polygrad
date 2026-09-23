#ifndef POLY_MODELS_TRANSFORMER_H
#define POLY_MODELS_TRANSFORMER_H

#include "../model.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Internal dense Transformer configuration. Checkpoint adapters own defaults,
 * name/layout conversion and validation of their external config dialects. */
typedef struct {
  int dim, hidden_dim, heads, kv_heads, layers, vocab, batch, length;
  int head_dim, qk_norm, cache_capacity, prefill_chunk;
  double eps, theta, factor, low_freq, high_freq, original_context;
  bool tied;
  /* Preserve the existing Qwen execution boundaries during consolidation. */
  bool materialize_intermediates;
} ModelTransformerConfig;

typedef struct {
  const char *label, *input, *output, *cos, *sin;
  const char *embedding, *norm, *head;
  const char *block, *attn_norm, *qkv[3], *qk_norm[2], *out, *ffn_norm;
  const char *gate, *up, *down;
} ModelTransformerNames;

/* Uses the caller's factory scope; returns an ordinary checkpoint-required
 * Model. No new execution path or public C ABI. */
PolyModel *model_transformer_build(
    PolyCtx *ctx,
    const ModelTransformerConfig *config,
    const ModelTransformerNames *names,
    PolyModelError *err
);

/* Causal decoder only (tinygrad llm/model.py), not a base for vision encoders.
 * Successful adoption transfers Model ownership. Failure leaves it with caller.
 * The borrowed model accessor permits ordinary poly_model_* operations. */
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
