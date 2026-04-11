#ifndef POLY_MODEL_HF_LOADER_H
#define POLY_MODEL_HF_LOADER_H

#include "../instance.h"
#include "../loaders/hf_decode.h"
#include "../loaders/import_error.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Model configuration (cJSON wrapper for HF config.json) */

typedef struct PolyModelConfig PolyModelConfig;

PolyModelConfig *poly_model_config_new(void);
PolyModelConfig *poly_model_config_from_json(const char *json, int len);
int poly_model_config_get_int(const PolyModelConfig *cfg, const char *key, int default_val);
float poly_model_config_get_float(const PolyModelConfig *cfg, const char *key, float default_val);
const char *poly_model_config_get_string(const PolyModelConfig *cfg, const char *key, const char *default_val);
void poly_model_config_set_int(PolyModelConfig *config, const char *key, int value);
void poly_model_config_set_float(PolyModelConfig *config, const char *key, float value);
void poly_model_config_free(PolyModelConfig *config);

/* HuggingFace loader (auto-dispatch by model_type) */

PolyInstance *poly_hf_load(
    const char *config_json, int config_len,
    const uint8_t **weight_files, const int64_t *weight_lens,
    int n_weight_files,
    int max_batch, int max_seq_len);

#ifdef __cplusplus
}
#endif

#endif /* POLY_MODEL_HF_LOADER_H */
