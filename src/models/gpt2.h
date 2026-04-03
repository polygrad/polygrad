#ifndef POLY_MODEL_GPT2_H
#define POLY_MODEL_GPT2_H

#include "../instance.h"

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
PolyInstance *poly_gpt2(const GPT2Config *cfg);
PolyInstance *poly_gpt2_from_json(const char *json, int len);

#ifdef __cplusplus
}
#endif

#endif /* POLY_MODEL_GPT2_H */
