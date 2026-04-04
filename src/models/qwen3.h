/*
 * qwen3.h -- Qwen3 model builder
 */

#ifndef POLY_MODEL_QWEN3_H
#define POLY_MODEL_QWEN3_H

#include "../instance.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int vocab_size;
    int dim;            /* embedding_length / hidden_size */
    int n_heads;        /* attention.head_count */
    int n_kv_heads;     /* attention.head_count_kv */
    int n_layers;       /* block_count */
    int hidden_dim;     /* feed_forward_length (intermediate_size) */
    int head_dim;       /* dim / n_heads */
    int max_seq_len;
    int batch_size;
    float norm_eps;
    float rope_theta;
    int qk_norm;        /* head_dim if per-head Q/K norm, 0 otherwise */
} Qwen3Config;

Qwen3Config poly_qwen3_config_default(void);
PolyInstance *poly_qwen3(const Qwen3Config *cfg);

#ifdef __cplusplus
}
#endif

#endif /* POLY_MODEL_QWEN3_H */
