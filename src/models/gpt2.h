#ifndef POLY_MODEL_GPT2_H
#define POLY_MODEL_GPT2_H

#include "../model.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int vocab_size;
  int n_embd;
  int n_head;
  int n_layer;
  int max_seq_len;
  int batch_size;
  float norm_eps;
} GPT2Config;

GPT2Config poly_gpt2_config_default(void);
/* Checkpoint-required: execution/export reject until every PARAM is supplied
 * through Model write/upload or checkpoint loading (raw pointer reads do not count). */
/* Borrows ctx and restores its defaults on every exit. Explicit NULL ctx
 * requests a standalone Model that owns its context. */
PolyModel *poly_gpt2_into(PolyCtx *ctx, const GPT2Config *cfg, PolyDevice device);

#ifdef __cplusplus
}
#endif

#endif /* POLY_MODEL_GPT2_H */
