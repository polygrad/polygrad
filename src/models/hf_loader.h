#ifndef POLY_MODEL_HF_LOADER_H
#define POLY_MODEL_HF_LOADER_H

#include "../model.h"
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
const char *poly_model_config_get_string(
    const PolyModelConfig *cfg,
    const char *key,
    const char *default_val
);
void poly_model_config_set_int(PolyModelConfig *config, const char *key, int value);
void poly_model_config_set_float(PolyModelConfig *config, const char *key, float value);
void poly_model_config_free(PolyModelConfig *config);

/* HuggingFace loader (auto-dispatch by model_type) */

typedef struct PolyHfImport PolyHfImport;
/* Incremental safetensors import. Add each file's complete length-prefixed header
 * and total length, then build. request returns borrowed JSON {file,offset,length}
 * for one tensor, or NULL when complete/failed. feed supplies exactly that range.
 * At most one tensor (two for tied-alias comparisons) is staged at a time; legacy
 * F32 adapters may additionally need conversion/transpose scratch. No model is
 * published until finish. Free aborts safely after any failed or interrupted step.
 * Source files must be immutable throughout the import. No I/O is performed here. */
PolyHfImport *poly_hf_import_new(
    PolyCtx *ctx,
    const char *config_json,
    int max_batch,
    int max_seq_len,
    PolyDevice device
);
int poly_hf_import_add_file(
    PolyHfImport *session,
    const uint8_t *header,
    int header_len,
    int64_t file_len
);
int poly_hf_import_build(PolyHfImport *session);
const char *poly_hf_import_request(PolyHfImport *session);
int poly_hf_import_feed(PolyHfImport *session, const uint8_t *data, size_t length);
PolyModel *poly_hf_import_finish(PolyHfImport *session);
void poly_hf_import_free(PolyHfImport *session);

/* Runtime-bound imports borrow ctx; the standalone form owns a new context. */
PolyModel *poly_hf_load_into(
    PolyCtx *ctx,
    const char *config_json,
    int config_len,
    const uint8_t **weight_files,
    const int64_t *weight_lens,
    int n_weight_files,
    int max_batch,
    int max_seq_len,
    PolyDevice device
);

PolyModel *poly_hf_load(
    const char *config_json,
    int config_len,
    const uint8_t **weight_files,
    const int64_t *weight_lens,
    int n_weight_files,
    int max_batch,
    int max_seq_len,
    PolyDevice device
);

#ifdef __cplusplus
}
#endif

#endif /* POLY_MODEL_HF_LOADER_H */
