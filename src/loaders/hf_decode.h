/*
 * hf_decode.h -- Generic HuggingFace config.json + weight decoder
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
 * Result of decoding config.json with safetensors or restricted PyTorch ZIP files.
 *
 * Created by poly_hf_decode(). Freed by poly_hf_decoded_free().
 * Contains parsed config JSON and all decoded tensor views.
 */
typedef struct {
  cJSON *config; /* parsed config.json (owned) */
  const char *model_type; /* borrowed from config */
  PolyDecodedTensor *tensors; /* array of decoded tensor views */
  int n_tensors;
} PolyHfDecoded;

/*
 * Decode config.json with safetensors or restricted PyTorch ZIP state dictionaries.
 *
 * No model-specific logic. Returns 0 on success, -1 on error.
 * On success, *out is a newly allocated PolyHfDecoded (caller frees).
 * Caller must keep weight_files buffers alive until *out is freed.
 */
int poly_hf_decode(
    const char *config_json,
    int config_len,
    const uint8_t **weight_files,
    const int64_t *weight_lens,
    int n_weight_files,
    PolyHfDecoded **out
);

void poly_hf_decoded_free(PolyHfDecoded *hf);

/* Plan checkpoint files without I/O. files_json is an array of repository-relative
 * names. Prefer safetensors over PyTorch; within a format an index is authoritative.
 * The JSON result has format, index (name or null), and files (sorted unique names).
 * If an index is selected but index_json is NULL, files is empty: fetch the index
 * and call again. Otherwise every selected shard must exist in files_json.
 * Returns required output bytes including NUL, or -1 with poly_import_last_error.
 * A NULL/short output buffer is not written; call again with the returned size. */
int poly_hf_plan(const char *files_json, const char *index_json, char *out, int capacity);

#ifdef __cplusplus
}
#endif

#endif /* POLY_HF_DECODE_H */
