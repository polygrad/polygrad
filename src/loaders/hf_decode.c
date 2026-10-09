/*
 * hf_decode.c -- Generic HuggingFace config.json + weight decoder
 *
 * Parses HF files into PolyHfDecoded. Zero model-specific logic.
 */

#define _POSIX_C_SOURCE 200809L
#include "hf_decode.h"
#include "import_error.h"
#include "torch_decode.h"
#include "../safetensors.h"
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <stdbool.h>

static int hf_name_cmp(const void *a, const void *b) {
  return strcmp(*(const char *const *)a, *(const char *const *)b);
}

static int hf_relative_name(const char *s) {
  if (!s || !*s || *s == '/' || strpbrk(s, "\\:")) return 0;
  for (const char *p = s; *p;) {
    const char *end = strchr(p, '/');
    size_t n = end ? (size_t)(end - p) : strlen(p);
    if (!n || (n == 1 && *p == '.') || (n == 2 && p[0] == '.' && p[1] == '.')) return 0;
    p += n;
    if (*p && !*++p) return 0;
  }
  return 1;
}

/* cJSON strings are NUL-terminated; reject escaped NUL before parsing so a
 * manifest cannot silently select a truncated filename. Ignore escaped slashes. */
static cJSON *hf_manifest_json(const char *text) {
  if (!text) return NULL;
  for (const char *p = text; *p; p++) {
    if (*p == '\\' && p[1]) {
      if (!strncmp(p + 1, "u0000", 5)) return NULL;
      p++;
    }
  }
  return cJSON_ParseWithOpts(text, NULL, 1);
}

int poly_hf_plan(const char *files_json, const char *index_json, char *out, int capacity) {
  cJSON *listing = hf_manifest_json(files_json), *index = NULL, *result = NULL;
  char **names = NULL, **shards = NULL, *encoded = NULL;
  const char *error = "invalid checkpoint file list";
  int rc = -1, count = cJSON_GetArraySize(listing), n_shards = 0;
  poly_import_error_clear();
  if (!cJSON_IsArray(listing) || count <= 0 || count > 100000) goto done;
  names = malloc((size_t)count * sizeof(*names));
  if (!names) goto oom;
  int i = 0;
  cJSON *item;
  cJSON_ArrayForEach(item, listing) {
    if (!cJSON_IsString(item) || !item->valuestring[0]) goto done;
    names[i++] = item->valuestring;
  }
  qsort(names, (size_t)count, sizeof(*names), hf_name_cmp);
  const char *candidates[] = {
      "model.safetensors", "model.safetensors.index.json", "pytorch_model.bin",
      "pytorch_model.bin.index.json"};
  int choice;
  for (choice = 0; choice < 4; choice++)
    if (bsearch(&candidates[choice], names, (size_t)count, sizeof(*names), hf_name_cmp)) break;
  error = "no model.safetensors or pytorch_model.bin checkpoint found";
  if (choice == 4) goto done;
  bool indexed = choice % 2 == 1;
  result = cJSON_CreateObject();
  if (!result ||
      !cJSON_AddStringToObject(result, "format", choice < 2 ? "safetensors" : "pytorch") ||
      !(indexed ? cJSON_AddStringToObject(result, "index", candidates[choice])
                : cJSON_AddNullToObject(result, "index")))
    goto oom;
  cJSON *files = cJSON_AddArrayToObject(result, "files");
  if (!files) goto oom;
  if (!indexed) {
    if (index_json) {
      error = "index supplied for an unsharded checkpoint";
      goto done;
    }
    item = cJSON_CreateString(candidates[choice]);
    if (!item) goto oom;
    cJSON_AddItemToArray(files, item);
  } else if (index_json) {
    index = hf_manifest_json(index_json);
    cJSON *mapping = cJSON_GetObjectItemCaseSensitive(index, "weight_map");
    int n = cJSON_GetArraySize(mapping);
    error = "invalid checkpoint weight_map";
    if (!cJSON_IsObject(mapping) || n <= 0 || n > 100000) goto done;
    shards = malloc((size_t)n * sizeof(*shards));
    if (!shards) goto oom;
    cJSON_ArrayForEach(item, mapping) {
      error = "invalid checkpoint weight_map";
      if (!item->string || !item->string[0] || !cJSON_IsString(item)) goto done;
      error = "invalid checkpoint shard path";
      if (!hf_relative_name(item->valuestring)) goto done;
      error = "checkpoint shard not found in file list";
      if (!bsearch(&item->valuestring, names, (size_t)count, sizeof(*names), hf_name_cmp))
        goto done;
      shards[n_shards++] = item->valuestring;
    }
    qsort(shards, (size_t)n_shards, sizeof(*shards), hf_name_cmp);
    for (i = 0; i < n_shards; i++) {
      if (i && !strcmp(shards[i], shards[i - 1])) continue;
      item = cJSON_CreateString(shards[i]);
      if (!item) goto oom;
      cJSON_AddItemToArray(files, item);
    }
  }
  encoded = cJSON_PrintUnformatted(result);
  if (!encoded || strlen(encoded) >= INT_MAX) goto oom;
  rc = (int)strlen(encoded) + 1;
  if (out && capacity >= rc) memcpy(out, encoded, (size_t)rc);
  goto done;
oom:
  error = "checkpoint planning allocation failed";
done:
  if (rc < 0) poly_import_error_set(POLY_IMPORT_ERR_PARSE, "Hugging Face: %s", error);
  free(encoded);
  free(shards);
  free(names);
  cJSON_Delete(result);
  cJSON_Delete(index);
  cJSON_Delete(listing);
  return rc;
}

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
    if (!weight_files[f] || weight_lens[f] < 0 || (uint64_t)weight_lens[f] > SIZE_MAX) {
      poly_import_error_set(POLY_IMPORT_ERR_PARSE, "invalid weight file %d", f);
      goto fail;
    }
    /* Dispatch by container signature, never retry an invalid safetensors file
     * as pickle. Both decoders publish the same contiguous tensor contract. */
    if (weight_lens[f] >= 4 && !memcmp(weight_files[f], "PK\003\004", 4)) {
      PolyDecodedTensor *decoded = NULL;
      if (poly_torch_decode(weight_files[f], (size_t)weight_lens[f], &decoded, &n)) goto fail;
      PolyDecodedTensor *next = NULL;
      if (n <= INT_MAX - hf->n_tensors && (size_t)(hf->n_tensors + n) <= SIZE_MAX / sizeof(*next) &&
          !hf_allocation_fails())
        next = realloc(
            hf->tensors, (size_t)(hf->n_tensors + n ? hf->n_tensors + n : 1) * sizeof(*next)
        );
      if (!next) {
        for (int i = 0; i < n; i++)
          free(decoded[i].name);
        free(decoded);
        poly_import_error_set(POLY_IMPORT_ERR_INTERNAL, "HF tensor table allocation failed");
        goto fail;
      }
      hf->tensors = next;
      memcpy(hf->tensors + hf->n_tensors, decoded, (size_t)n * sizeof(*decoded));
      hf->n_tensors += n;
      free(decoded);
      continue;
    }
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

  if (decoded_names_unique(hf->tensors, hf->n_tensors) != 0) goto fail;
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
