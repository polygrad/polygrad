/*
 * hf_decode.h -- Generic HuggingFace config.json + safetensors decoder
 *
 * Parses HF files into PolyHfDecoded. Contains zero model-specific logic.
 */

#ifndef POLY_HF_DECODE_H
#define POLY_HF_DECODE_H

#include "decoded.h"
#include "../../vendor/cjson/cJSON.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Result of decoding HuggingFace config.json + safetensors files.
 *
 * Created by poly_hf_decode(). Freed by poly_hf_decoded_free().
 * Contains parsed config JSON and all decoded tensor views.
 */
typedef struct {
    cJSON              *config;      /* parsed config.json (owned) */
    const char         *model_type;  /* borrowed from config */
    PolyDecodedTensor  *tensors;     /* array of decoded tensor views */
    int                 n_tensors;
} PolyHfDecoded;

/*
 * Decode HF config.json + safetensors weight files.
 *
 * No model-specific logic. Returns 0 on success, -1 on error.
 * On success, *out is a newly allocated PolyHfDecoded (caller frees).
 * Caller must keep weight_files buffers alive until *out is freed.
 */
int poly_hf_decode(
    const char *config_json, int config_len,
    const uint8_t **weight_files, const int64_t *weight_lens,
    int n_weight_files,
    PolyHfDecoded **out);

void poly_hf_decoded_free(PolyHfDecoded *hf);

#ifdef __cplusplus
}
#endif

#endif /* POLY_HF_DECODE_H */
