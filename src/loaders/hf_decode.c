/*
 * hf_decode.c -- Generic HuggingFace config.json + safetensors decoder
 *
 * Parses HF files into PolyHfDecoded. Zero model-specific logic.
 */

#define _POSIX_C_SOURCE 200809L
#include "hf_decode.h"
#include "import_error.h"
#include "../safetensors.h"
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#ifdef POLY_TESTING
static int test_hf_alloc_fail_after = -1;
void poly_test_hf_alloc_fail_after(int count) {
  test_hf_alloc_fail_after = count;
}
#endif

static int hf_allocation_fails(void) {
#ifdef POLY_TESTING
  if (test_hf_alloc_fail_after == 0) {
    test_hf_alloc_fail_after = -1;
    return 1;
  }
  if (test_hf_alloc_fail_after > 0) test_hf_alloc_fail_after--;
#endif
  return 0;
}

int poly_hf_decode(
    const char *config_json,
    int config_len,
    const uint8_t **weight_files,
    const int64_t *weight_lens,
    int n_weight_files,
    PolyHfDecoded **out
) {
  if (!out) return -1;
  *out = NULL;

  if (!config_json || config_len <= 0 || n_weight_files < 0 ||
      (n_weight_files && (!weight_files || !weight_lens))) {
    poly_import_error_set(POLY_IMPORT_ERR_PARSE, "invalid HF config or shard table");
    return -1;
  }

  cJSON *config = cJSON_ParseWithLength(config_json, (size_t)config_len);
  if (!config || !cJSON_IsObject(config)) {
    cJSON_Delete(config);
    poly_import_error_set(POLY_IMPORT_ERR_PARSE, "failed to parse config.json");
    return -1;
  }

  cJSON *mt = cJSON_GetObjectItemCaseSensitive(config, "model_type");
  const char *model_type = (mt && cJSON_IsString(mt)) ? mt->valuestring : "";

  PolyHfDecoded *hf = hf_allocation_fails() ? NULL : calloc(1, sizeof(PolyHfDecoded));
  if (!hf) {
    cJSON_Delete(config);
    poly_import_error_set(POLY_IMPORT_ERR_INTERNAL, "HF allocation failed");
    return -1;
  }
  hf->config = config;
  hf->model_type = model_type;
  /* Like safe_load's dictionary construction, publish only a complete
   * candidate. Decode each shard once; payload bytes remain caller-owned. */
  for (int f = 0; f < n_weight_files; f++) {
    int n = 0;
    PolySafetensorViewEx *views =
        poly_safetensors_decode_ex(weight_files[f], weight_lens[f], &n, NULL);
    if (!views) {
      poly_import_error_set(POLY_IMPORT_ERR_PARSE, "failed to decode weight file %d", f);
      goto fail;
    }
    if (n > 0) {
      PolyDecodedTensor *next = NULL;
      if (n <= INT_MAX - hf->n_tensors && (size_t)(hf->n_tensors + n) <= SIZE_MAX / sizeof(*next) &&
          !hf_allocation_fails())
        next = realloc(hf->tensors, (size_t)(hf->n_tensors + n) * sizeof(*next));
      if (!next) {
        for (int i = 0; i < n; i++)
          free(views[i].name);
        free(views);
        poly_import_error_set(POLY_IMPORT_ERR_INTERNAL, "HF tensor table allocation failed");
        goto fail;
      }
      hf->tensors = next;
    }
    for (int i = 0; i < n; i++) {
      PolyDecodedTensor *dt = &hf->tensors[hf->n_tensors];
      memset(dt, 0, sizeof(*dt));
      dt->name = views[i].name;
      dt->data = views[i].raw_data;
      dt->ndim = views[i].ndim;
      dt->numel = views[i].numel;
      /* Map PolySafetensorDType to unified POLY_DECODED_* codes.
       * The enum values 0-9 happen to match (both start F32,F16,...). */
      dt->dtype = (int)views[i].dtype;
      for (int d = 0; d < views[i].ndim && d < 8; d++)
        dt->shape[d] = views[i].shape[d];
      hf->n_tensors++;
    }

    free(views);
  }

  *out = hf;
  return 0;

fail:
  poly_hf_decoded_free(hf);
  return -1;
}

void poly_hf_decoded_free(PolyHfDecoded *hf) {
  if (!hf) return;
  for (int i = 0; i < hf->n_tensors; i++)
    free(hf->tensors[i].name);
  free(hf->tensors);
  if (hf->config) cJSON_Delete(hf->config);
  free(hf);
}
