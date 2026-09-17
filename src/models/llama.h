/* Dense Llama family. Construction/configuration stays above the Tensor core. */
#ifndef POLY_MODELS_LLAMA_H
#define POLY_MODELS_LLAMA_H

#include "../model.h"

#ifdef __cplusplus
extern "C" {
#endif

/* HF-style Llama configuration, float32, fixed [batch_size,max_seq_len] tokens
 * -> [batch_size,max_seq_len,vocab_size] logits. Separate or tied output weights,
 * unscaled or llama3 RoPE scaling, no KV cache.
 * Unsupported architecture options fail explicitly.
 * Borrows ctx (which must outlive the Model); NULL creates an owned context.
 * Execution/export reject until all PARAM storage has been initialized by
 * successful writes or checkpoint loading. This constructs storage, not
 * pretrained weights. Use poly_hf_load[_into]
 * for a complete config+safetensors checkpoint, or explicit Model writes. */
PolyModel *poly_llama_from_json(PolyCtx *ctx, const char *json, int len, PolyModelError *err);

#ifdef __cplusplus
}
#endif
#endif
