/*
 * hf_loader.c -- HuggingFace model loader (thin auto-dispatch wrapper)
 *
 * Decodes config.json + safetensors, looks up model by model_type,
 * dispatches to model-owned import function. Zero model-specific logic.
 */

#define _POSIX_C_SOURCE 200809L
#include "hf_loader.h"
#include "../loaders/import_desc.h"
#include "../loaders/import_error.h"
#include <stdio.h>

PolyInstance *poly_hf_load(
    const char *config_json, int config_len,
    const uint8_t **weight_files, const int64_t *weight_lens,
    int n_weight_files,
    int max_batch, int max_seq_len)
{
  poly_import_error_clear();

  /* 1. Generic decode */
  PolyHfDecoded *hf = NULL;
  if (poly_hf_decode(config_json, config_len,
                     weight_files, weight_lens, n_weight_files,
                     &hf) != 0 || !hf)
    return NULL;

  /* 2. Lookup model descriptor */
  const PolyImportDesc *desc = poly_import_desc_find(hf->model_type);
  if (!desc || !desc->from_hf_decoded) {
    poly_import_error_set(POLY_IMPORT_ERR_UNSUPPORTED_MODEL,
        "unsupported model_type '%s'", hf->model_type);
    poly_hf_decoded_free(hf);
    return NULL;
  }

  /* 3. Dispatch to model-owned importer */
  PolyGenericImportOpts opts = {
    .max_batch   = max_batch,
    .max_seq_len = max_seq_len,
  };
  PolyInstance *inst = desc->from_hf_decoded(hf, &opts);

  /* 4. Cleanup */
  poly_hf_decoded_free(hf);
  return inst;
}
