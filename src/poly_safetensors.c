/*
 * poly_safetensors.c -- Safetensors encode/decode for f32 tensors
 *
 * Safetensors format:
 *   [8 bytes LE] header_size
 *   [header_size bytes] JSON header
 *   [remaining] raw tensor data (concatenated, packed)
 *
 * JSON header maps tensor names to:
 *   { "dtype": "F32", "shape": [...], "data_offsets": [start, end] }
 * Optional: "__metadata__" key with string-valued metadata.
 *
 * Only f32 (float32) tensors supported.
 */

#define _POSIX_C_SOURCE 200809L
#include "poly_safetensors.h"
#include "../vendor/cjson/cJSON.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <inttypes.h>

/* ── Helpers ────────────────────────────────────────────────────────── */

static void write_le64(uint8_t *dst, uint64_t v) {
  for (int i = 0; i < 8; i++)
    dst[i] = (uint8_t)(v >> (i * 8));
}

static uint64_t read_le64(const uint8_t *src) {
  uint64_t v = 0;
  for (int i = 0; i < 8; i++)
    v |= (uint64_t)src[i] << (i * 8);
  return v;
}

static int64_t compute_numel(const int64_t *shape, int ndim) {
  int64_t n = 1;
  for (int i = 0; i < ndim; i++) n *= shape[i];
  return n;
}

/* Compare entries by name for deterministic output */
static int entry_cmp(const void *a, const void *b) {
  const int *ia = (const int *)a;
  const int *ib = (const int *)b;
  /* We'll pass indices via a wrapper; use a global for qsort context */
  return 0; /* placeholder -- replaced by actual sort below */
}

/* ── Encode ─────────────────────────────────────────────────────────── */

uint8_t *poly_safetensors_encode(const PolySafetensorEntry *entries, int n,
                                 const char *metadata_json,
                                 int *out_len) {
  *out_len = 0;
  if (n < 0) return NULL;

  /* Sort indices by name for deterministic output */
  int *order = malloc(n * sizeof(int));
  if (!order) return NULL;
  for (int i = 0; i < n; i++) order[i] = i;

  /* Simple insertion sort (small N expected) */
  for (int i = 1; i < n; i++) {
    int key = order[i];
    int j = i - 1;
    while (j >= 0 && strcmp(entries[order[j]].name, entries[key].name) > 0) {
      order[j + 1] = order[j];
      j--;
    }
    order[j + 1] = key;
  }

  /* Compute data offsets */
  uint64_t *offsets = calloc(n, sizeof(uint64_t));
  uint64_t data_offset = 0;
  for (int i = 0; i < n; i++) {
    int idx = order[i];
    offsets[i] = data_offset;
    int64_t numel = compute_numel(entries[idx].shape, entries[idx].ndim);
    data_offset += (uint64_t)numel * sizeof(float);
  }

  /* Build JSON header */
  cJSON *root = cJSON_CreateObject();
  if (!root) { free(order); free(offsets); return NULL; }

  /* Add __metadata__ if provided */
  if (metadata_json && metadata_json[0] != '\0') {
    cJSON *meta = cJSON_Parse(metadata_json);
    if (meta) {
      cJSON_AddItemToObject(root, "__metadata__", meta);
    }
  }

  for (int i = 0; i < n; i++) {
    int idx = order[i];
    cJSON *tensor = cJSON_CreateObject();
    cJSON_AddStringToObject(tensor, "dtype", "F32");

    cJSON *shape_arr = cJSON_CreateArray();
    for (int d = 0; d < entries[idx].ndim; d++)
      cJSON_AddItemToArray(shape_arr, cJSON_CreateNumber((double)entries[idx].shape[d]));
    cJSON_AddItemToObject(tensor, "shape", shape_arr);

    int64_t numel = compute_numel(entries[idx].shape, entries[idx].ndim);
    uint64_t start = offsets[i];
    uint64_t end = start + (uint64_t)numel * sizeof(float);
    cJSON *data_offs = cJSON_CreateArray();
    cJSON_AddItemToArray(data_offs, cJSON_CreateNumber((double)start));
    cJSON_AddItemToArray(data_offs, cJSON_CreateNumber((double)end));
    cJSON_AddItemToObject(tensor, "data_offsets", data_offs);

    cJSON_AddItemToObject(root, entries[idx].name, tensor);
  }

  char *json_str = cJSON_PrintUnformatted(root);
  cJSON_Delete(root);
  if (!json_str) { free(order); free(offsets); return NULL; }

  uint64_t header_size = (uint64_t)strlen(json_str);
  uint64_t total_size = 8 + header_size + data_offset;

  uint8_t *buf = malloc((size_t)total_size);
  if (!buf) { free(json_str); free(order); free(offsets); return NULL; }

  /* Write header size (8 bytes LE) */
  write_le64(buf, header_size);

  /* Write JSON header */
  memcpy(buf + 8, json_str, (size_t)header_size);
  free(json_str);

  /* Write tensor data in sorted order */
  for (int i = 0; i < n; i++) {
    int idx = order[i];
    int64_t numel = compute_numel(entries[idx].shape, entries[idx].ndim);
    uint64_t byte_offset = 8 + header_size + offsets[i];
    memcpy(buf + byte_offset, entries[idx].data, (size_t)numel * sizeof(float));
  }

  free(order);
  free(offsets);
  *out_len = (int)total_size;
  return buf;
}

/* ── Decode ─────────────────────────────────────────────────────────── */

PolySafetensorView *poly_safetensors_decode(const uint8_t *data, int len,
                                            int *n_out,
                                            char **metadata_out) {
  *n_out = 0;
  if (metadata_out) *metadata_out = NULL;

  if (!data || len < 8) {
    fprintf(stderr, "poly_safetensors_decode: data too short\n");
    return NULL;
  }

  uint64_t header_size = read_le64(data);
  if (8 + header_size > (uint64_t)len) {
    fprintf(stderr, "poly_safetensors_decode: header_size %" PRIu64
            " exceeds data length %d\n", header_size, len);
    return NULL;
  }

  /* Parse JSON header */
  char *json_str = malloc((size_t)header_size + 1);
  if (!json_str) return NULL;
  memcpy(json_str, data + 8, (size_t)header_size);
  json_str[header_size] = '\0';

  cJSON *root = cJSON_Parse(json_str);
  free(json_str);
  if (!root) {
    fprintf(stderr, "poly_safetensors_decode: JSON parse error\n");
    return NULL;
  }

  /* Count tensor entries (skip __metadata__) */
  int count = 0;
  cJSON *item;
  cJSON_ArrayForEach(item, root) {
    if (strcmp(item->string, "__metadata__") != 0) count++;
  }

  PolySafetensorView *views = calloc(count, sizeof(PolySafetensorView));
  if (!views && count > 0) { cJSON_Delete(root); return NULL; }

  const uint8_t *data_region = data + 8 + (size_t)header_size;
  uint64_t data_region_len = (uint64_t)len - 8 - header_size;

  int vi = 0;
  cJSON_ArrayForEach(item, root) {
    if (strcmp(item->string, "__metadata__") == 0) {
      if (metadata_out) {
        char *meta_str = cJSON_PrintUnformatted(item);
        if (meta_str) *metadata_out = meta_str;
      }
      continue;
    }

    /* Validate dtype */
    cJSON *dtype_val = cJSON_GetObjectItemCaseSensitive(item, "dtype");
    if (!dtype_val || !cJSON_IsString(dtype_val)) {
      fprintf(stderr, "poly_safetensors_decode: missing dtype for '%s'\n",
              item->string);
      goto fail;
    }
    if (strcmp(dtype_val->valuestring, "F32") != 0) {
      fprintf(stderr, "poly_safetensors_decode: unsupported dtype '%s' for '%s'\n",
              dtype_val->valuestring, item->string);
      goto fail;
    }

    /* Parse shape */
    cJSON *shape_arr = cJSON_GetObjectItemCaseSensitive(item, "shape");
    if (!shape_arr || !cJSON_IsArray(shape_arr)) {
      fprintf(stderr, "poly_safetensors_decode: missing shape for '%s'\n",
              item->string);
      goto fail;
    }
    int ndim = cJSON_GetArraySize(shape_arr);
    if (ndim > 8) {
      fprintf(stderr, "poly_safetensors_decode: ndim %d > 8 for '%s'\n",
              ndim, item->string);
      goto fail;
    }

    int64_t numel = 1;
    for (int d = 0; d < ndim; d++) {
      cJSON *dim = cJSON_GetArrayItem(shape_arr, d);
      if (!dim || !cJSON_IsNumber(dim)) goto fail;
      views[vi].shape[d] = (int64_t)dim->valuedouble;
      numel *= views[vi].shape[d];
    }
    views[vi].ndim = ndim;
    views[vi].numel = numel;

    /* Parse data_offsets */
    cJSON *offs_arr = cJSON_GetObjectItemCaseSensitive(item, "data_offsets");
    if (!offs_arr || !cJSON_IsArray(offs_arr) || cJSON_GetArraySize(offs_arr) != 2)
      goto fail;
    uint64_t start = (uint64_t)cJSON_GetArrayItem(offs_arr, 0)->valuedouble;
    uint64_t end = (uint64_t)cJSON_GetArrayItem(offs_arr, 1)->valuedouble;

    if (end < start || end > data_region_len) {
      fprintf(stderr, "poly_safetensors_decode: data_offsets out of bounds for '%s'\n",
              item->string);
      goto fail;
    }
    if ((end - start) != (uint64_t)numel * sizeof(float)) {
      fprintf(stderr, "poly_safetensors_decode: data size mismatch for '%s'\n",
              item->string);
      goto fail;
    }

    views[vi].name = strdup(item->string);
    views[vi].data = (const float *)(data_region + start);
    vi++;
  }

  cJSON_Delete(root);
  *n_out = count;
  return views;

fail:
  for (int i = 0; i < vi; i++) free(views[i].name);
  free(views);
  cJSON_Delete(root);
  return NULL;
}
