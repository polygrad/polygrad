/*
 * poly_safetensors.c -- Safetensors encode/decode
 *
 * Safetensors format:
 *   [8 bytes LE] header_size
 *   [header_size bytes] JSON header
 *   [remaining] raw tensor data (concatenated, packed)
 *
 * JSON header maps tensor names to:
 *   { "dtype": "F32"|"F16"|"BF16"|..., "shape": [...], "data_offsets": [start, end] }
 * Optional: "__metadata__" key with string-valued metadata.
 *
 * Two decode APIs:
 *   poly_safetensors_decode()    -- F32-only (original, backward-compatible)
 *   poly_safetensors_decode_ex() -- Multi-dtype (F16, BF16, F32, F64, int types)
 */

#define _POSIX_C_SOURCE 200809L
#include "safetensors.h"
#include "../vendor/cjson/cJSON.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>

/* Helpers */

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

static bool compute_numel_checked(const int64_t *shape, int ndim, int64_t *out) {
  if (!out || ndim < 0 || ndim > 8 || (ndim > 0 && !shape)) return false;
  int64_t n = 1;
  for (int i = 0; i < ndim; i++) {
    if (shape[i] < 0 || (shape[i] != 0 && n > INT64_MAX / shape[i])) return false;
    n *= shape[i];
  }
  *out = n;
  return true;
}

static const char *safetensor_dtype_name(PolySafetensorDType dtype) {
  switch (dtype) {
  case POLY_ST_F32:
    return "F32";
  case POLY_ST_F16:
    return "F16";
  case POLY_ST_BF16:
    return "BF16";
  case POLY_ST_F64:
    return "F64";
  case POLY_ST_I64:
    return "I64";
  case POLY_ST_I32:
    return "I32";
  case POLY_ST_I16:
    return "I16";
  case POLY_ST_I8:
    return "I8";
  case POLY_ST_U8:
    return "U8";
  case POLY_ST_BOOL:
    return "BOOL";
  case POLY_ST_U16:
    return "U16";
  case POLY_ST_U32:
    return "U32";
  case POLY_ST_U64:
    return "U64";
  }
  return NULL;
}

static bool json_nonnegative_i64(const cJSON *value, int64_t *out) {
  /* cJSON stores numbers as double.  Accept only integers in the exactly
   * representable JSON range so the cast below cannot overflow or round. */
  static const double JSON_MAX_EXACT_INTEGER = 9007199254740991.0;
  if (!value || !out || !cJSON_IsNumber(value) || !isfinite(value->valuedouble) ||
      value->valuedouble < 0.0 || value->valuedouble > JSON_MAX_EXACT_INTEGER)
    return false;
  int64_t parsed = (int64_t)value->valuedouble;
  if ((double)parsed != value->valuedouble) return false;
  *out = parsed;
  return true;
}

static bool json_nonnegative_u64(const cJSON *value, uint64_t *out) {
  int64_t parsed = 0;
  if (!json_nonnegative_i64(value, &parsed)) return false;
  *out = (uint64_t)parsed;
  return true;
}

/* Encode */

uint8_t *poly_safetensors_encode(
    const PolySafetensorEntry *entries,
    int n,
    const char *metadata_json,
    int *out_len
) {
  if (out_len) *out_len = 0;
  if (!out_len || n < 0 || (n > 0 && !entries)) return NULL;

  /* Validate public rows before deterministic ordering.  In particular, the
   * insertion sort below compares names and therefore must never see NULL.
   * Keep all shape/dtype/data checks together so malformed encoder input is
   * rejected without partially constructing the archive. */
  for (int i = 0; i < n; i++) {
    int64_t numel = 0;
    int elem_size = poly_safetensor_dtype_size(entries[i].dtype);
    if (!entries[i].name || !*entries[i].name || !safetensor_dtype_name(entries[i].dtype) ||
        elem_size <= 0 || !compute_numel_checked(entries[i].shape, entries[i].ndim, &numel) ||
        (numel > 0 && !entries[i].data))
      return NULL;
  }

  /* Sort indices by name for deterministic output */
  int *order = n > 0 ? malloc((size_t)n * sizeof(int)) : NULL;
  if (n > 0 && !order) return NULL;
  for (int i = 0; i < n; i++)
    order[i] = i;

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
  uint64_t *offsets = calloc((size_t)n, sizeof(uint64_t));
  if (n > 0 && !offsets) {
    free(order);
    return NULL;
  }
  uint64_t data_offset = 0;
  for (int i = 0; i < n; i++) {
    int idx = order[i];
    offsets[i] = data_offset;
    int64_t numel = 0;
    int elem_size = poly_safetensor_dtype_size(entries[idx].dtype);
    if (!compute_numel_checked(entries[idx].shape, entries[idx].ndim, &numel) ||
        (uint64_t)numel > (UINT64_MAX - data_offset) / (uint64_t)elem_size) {
      free(order);
      free(offsets);
      return NULL;
    }
    data_offset += (uint64_t)numel * (uint64_t)elem_size;
  }

  /* Build JSON header */
  cJSON *root = cJSON_CreateObject();
  if (!root) {
    free(order);
    free(offsets);
    return NULL;
  }

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
    cJSON_AddStringToObject(tensor, "dtype", safetensor_dtype_name(entries[idx].dtype));

    cJSON *shape_arr = cJSON_CreateArray();
    for (int d = 0; d < entries[idx].ndim; d++)
      cJSON_AddItemToArray(shape_arr, cJSON_CreateNumber((double)entries[idx].shape[d]));
    cJSON_AddItemToObject(tensor, "shape", shape_arr);

    int64_t numel = 0;
    if (!compute_numel_checked(entries[idx].shape, entries[idx].ndim, &numel)) {
      cJSON_Delete(tensor);
      cJSON_Delete(root);
      free(order);
      free(offsets);
      return NULL;
    }
    int elem_size = poly_safetensor_dtype_size(entries[idx].dtype);
    uint64_t start = offsets[i];
    uint64_t end = start + (uint64_t)numel * (uint64_t)elem_size;
    cJSON *data_offs = cJSON_CreateArray();
    cJSON_AddItemToArray(data_offs, cJSON_CreateNumber((double)start));
    cJSON_AddItemToArray(data_offs, cJSON_CreateNumber((double)end));
    cJSON_AddItemToObject(tensor, "data_offsets", data_offs);

    cJSON_AddItemToObject(root, entries[idx].name, tensor);
  }

  char *json_str = cJSON_PrintUnformatted(root);
  cJSON_Delete(root);
  if (!json_str) {
    free(order);
    free(offsets);
    return NULL;
  }

  uint64_t header_size = (uint64_t)strlen(json_str);
  /* Pad JSON header to 4-byte alignment so float data starts at a 4-byte
   * aligned offset. Total header region = 8 + header_size; since 8%4==0,
   * we need header_size%4==0. Pad with trailing spaces (valid JSON whitespace). */
  uint64_t pad = (4 - header_size % 4) % 4;
  if (pad > 0) {
    char *padded = realloc(json_str, header_size + pad + 1);
    if (!padded) {
      free(json_str);
      free(order);
      free(offsets);
      return NULL;
    }
    json_str = padded;
    for (uint64_t p = 0; p < pad; p++)
      json_str[header_size + p] = ' ';
    json_str[header_size + pad] = '\0';
    header_size += pad;
  }
  if (header_size > UINT64_MAX - 8 || data_offset > UINT64_MAX - 8 - header_size ||
      8 + header_size + data_offset > (uint64_t)INT_MAX ||
      8 + header_size + data_offset > (uint64_t)SIZE_MAX) {
    free(json_str);
    free(order);
    free(offsets);
    return NULL;
  }
  uint64_t total_size = 8 + header_size + data_offset;

  uint8_t *buf = malloc((size_t)total_size);
  if (!buf) {
    free(json_str);
    free(order);
    free(offsets);
    return NULL;
  }

  /* Write header size (8 bytes LE) */
  write_le64(buf, header_size);

  /* Write JSON header */
  memcpy(buf + 8, json_str, (size_t)header_size);
  free(json_str);

  /* Write tensor data in sorted order */
  for (int i = 0; i < n; i++) {
    int idx = order[i];
    int64_t numel = 0;
    (void)compute_numel_checked(entries[idx].shape, entries[idx].ndim, &numel);
    int elem_size = poly_safetensor_dtype_size(entries[idx].dtype);
    uint64_t byte_offset = 8 + header_size + offsets[i];
    if (numel > 0) memcpy(buf + byte_offset, entries[idx].data, (size_t)numel * (size_t)elem_size);
  }

  free(order);
  free(offsets);
  *out_len = (int)total_size;
  return buf;
}

/* Decode */

PolySafetensorView *poly_safetensors_decode(
    const uint8_t *data,
    int len,
    int *n_out,
    char **metadata_out
) {
  if (!n_out) return NULL;
  *n_out = 0;
  if (metadata_out) *metadata_out = NULL;

  if (!data || len < 8) {
    fprintf(stderr, "poly_safetensors_decode: data too short\n");
    return NULL;
  }

  uint64_t header_size = read_le64(data);
  if (header_size > (uint64_t)len - 8) {
    fprintf(
        stderr, "poly_safetensors_decode: header_size %" PRIu64 " exceeds data length %d\n",
        header_size, len
    );
    return NULL;
  }

  /* Parse JSON header */
  char *json_str = malloc((size_t)header_size + 1);
  if (!json_str) return NULL;
  memcpy(json_str, data + 8, (size_t)header_size);
  json_str[header_size] = '\0';

  cJSON *root = cJSON_Parse(json_str);
  free(json_str);
  if (!root || !cJSON_IsObject(root)) {
    fprintf(stderr, "poly_safetensors_decode: JSON parse error\n");
    cJSON_Delete(root);
    return NULL;
  }

  /* Count tensor entries (skip __metadata__) */
  int count = 0;
  cJSON *item;
  cJSON_ArrayForEach(item, root) {
    if (!item->string) goto fail_root;
    if (strcmp(item->string, "__metadata__") != 0) count++;
  }

  /* A valid empty archive still needs a non-NULL success result. */
  PolySafetensorView *views = calloc(count ? (size_t)count : 1, sizeof(PolySafetensorView));
  if (!views) {
    cJSON_Delete(root);
    return NULL;
  }

  const uint8_t *data_region = data + 8 + (size_t)header_size;
  uint64_t data_region_len = (uint64_t)len - 8 - header_size;

  char *metadata = NULL;
  int vi = 0;
  cJSON_ArrayForEach(item, root) {
    if (strcmp(item->string, "__metadata__") == 0) {
      if (metadata_out) {
        char *meta_str = cJSON_PrintUnformatted(item);
        if (!meta_str) goto fail;
        free(metadata);
        metadata = meta_str;
      }
      continue;
    }

    /* Validate dtype */
    cJSON *dtype_val = cJSON_GetObjectItemCaseSensitive(item, "dtype");
    if (!dtype_val || !cJSON_IsString(dtype_val)) {
      fprintf(stderr, "poly_safetensors_decode: missing dtype for '%s'\n", item->string);
      goto fail;
    }
    if (strcmp(dtype_val->valuestring, "F32") != 0) {
      fprintf(
          stderr, "poly_safetensors_decode: unsupported dtype '%s' for '%s'\n",
          dtype_val->valuestring, item->string
      );
      goto fail;
    }

    /* Parse shape */
    cJSON *shape_arr = cJSON_GetObjectItemCaseSensitive(item, "shape");
    if (!shape_arr || !cJSON_IsArray(shape_arr)) {
      fprintf(stderr, "poly_safetensors_decode: missing shape for '%s'\n", item->string);
      goto fail;
    }
    int ndim = cJSON_GetArraySize(shape_arr);
    if (ndim > 8) {
      fprintf(stderr, "poly_safetensors_decode: ndim %d > 8 for '%s'\n", ndim, item->string);
      goto fail;
    }

    int64_t numel = 0;
    for (int d = 0; d < ndim; d++) {
      cJSON *dim = cJSON_GetArrayItem(shape_arr, d);
      if (!json_nonnegative_i64(dim, &views[vi].shape[d])) goto fail;
    }
    if (!compute_numel_checked(views[vi].shape, ndim, &numel)) goto fail;
    views[vi].ndim = ndim;
    views[vi].numel = numel;

    /* Parse data_offsets */
    cJSON *offs_arr = cJSON_GetObjectItemCaseSensitive(item, "data_offsets");
    if (!offs_arr || !cJSON_IsArray(offs_arr) || cJSON_GetArraySize(offs_arr) != 2) goto fail;
    uint64_t start = 0, end = 0;
    if (!json_nonnegative_u64(cJSON_GetArrayItem(offs_arr, 0), &start) ||
        !json_nonnegative_u64(cJSON_GetArrayItem(offs_arr, 1), &end))
      goto fail;

    if (end < start || end > data_region_len) {
      fprintf(
          stderr, "poly_safetensors_decode: data_offsets out of bounds for '%s'\n", item->string
      );
      goto fail;
    }
    if ((uint64_t)numel > UINT64_MAX / sizeof(float) ||
        (end - start) != (uint64_t)numel * sizeof(float)) {
      fprintf(stderr, "poly_safetensors_decode: data size mismatch for '%s'\n", item->string);
      goto fail;
    }

    views[vi].name = strdup(item->string);
    if (!views[vi].name) goto fail;
    views[vi].data = (const float *)(data_region + start);
    vi++;
  }

  cJSON_Delete(root);
  *n_out = count;
  if (metadata_out) *metadata_out = metadata;
  return views;

fail_root:
  cJSON_Delete(root);
  return NULL;

fail:
  free(metadata);
  for (int i = 0; i < vi; i++)
    free(views[i].name);
  free(views);
  cJSON_Delete(root);
  return NULL;
}

/* Multi-dtype support */

int poly_safetensor_dtype_size(PolySafetensorDType dtype) {
  switch (dtype) {
  case POLY_ST_F64:
  case POLY_ST_I64:
    return 8;
  case POLY_ST_F32:
  case POLY_ST_I32:
    return 4;
  case POLY_ST_F16:
  case POLY_ST_BF16:
  case POLY_ST_I16:
    return 2;
  case POLY_ST_I8:
  case POLY_ST_U8:
  case POLY_ST_BOOL:
    return 1;
  case POLY_ST_U16:
    return 2;
  case POLY_ST_U32:
    return 4;
  case POLY_ST_U64:
    return 8;
  }
  return 0;
}

static int parse_safetensor_dtype(const char *s, PolySafetensorDType *out) {
  if (strcmp(s, "F32") == 0) {
    *out = POLY_ST_F32;
    return 0;
  }
  if (strcmp(s, "F16") == 0) {
    *out = POLY_ST_F16;
    return 0;
  }
  if (strcmp(s, "BF16") == 0) {
    *out = POLY_ST_BF16;
    return 0;
  }
  if (strcmp(s, "F64") == 0) {
    *out = POLY_ST_F64;
    return 0;
  }
  if (strcmp(s, "I64") == 0) {
    *out = POLY_ST_I64;
    return 0;
  }
  if (strcmp(s, "I32") == 0) {
    *out = POLY_ST_I32;
    return 0;
  }
  if (strcmp(s, "I16") == 0) {
    *out = POLY_ST_I16;
    return 0;
  }
  if (strcmp(s, "I8") == 0) {
    *out = POLY_ST_I8;
    return 0;
  }
  if (strcmp(s, "U8") == 0) {
    *out = POLY_ST_U8;
    return 0;
  }
  if (strcmp(s, "BOOL") == 0) {
    *out = POLY_ST_BOOL;
    return 0;
  }
  if (strcmp(s, "U16") == 0) {
    *out = POLY_ST_U16;
    return 0;
  }
  if (strcmp(s, "U32") == 0) {
    *out = POLY_ST_U32;
    return 0;
  }
  if (strcmp(s, "U64") == 0) {
    *out = POLY_ST_U64;
    return 0;
  }
  return -1;
}

PolySafetensorViewEx *poly_safetensors_decode_ex(
    const uint8_t *data,
    int64_t len,
    int *n_out,
    char **metadata_out
) {
  if (!n_out) return NULL;
  *n_out = 0;
  if (metadata_out) *metadata_out = NULL;

  if (!data || len < 8) {
    fprintf(stderr, "poly_safetensors_decode_ex: data too short\n");
    return NULL;
  }

  uint64_t header_size = read_le64(data);
  if (header_size > (uint64_t)len - 8 || header_size >= SIZE_MAX) {
    fprintf(
        stderr,
        "poly_safetensors_decode_ex: header_size %" PRIu64 " exceeds data length %" PRId64 "\n",
        header_size, len
    );
    return NULL;
  }

  /* Parse JSON header (strip trailing spaces per safetensors spec) */
  char *json_str = malloc((size_t)header_size + 1);
  if (!json_str) return NULL;
  memcpy(json_str, data + 8, (size_t)header_size);
  json_str[header_size] = '\0';

  /* Trim trailing whitespace from JSON (safetensors spec allows padding) */
  size_t json_len = header_size;
  while (json_len > 0 && (json_str[json_len - 1] == ' ' || json_str[json_len - 1] == '\0'))
    json_str[--json_len] = '\0';

  cJSON *root = cJSON_Parse(json_str);
  free(json_str);
  if (!root || !cJSON_IsObject(root)) {
    fprintf(stderr, "poly_safetensors_decode_ex: JSON parse error\n");
    cJSON_Delete(root);
    return NULL;
  }

  /* Count tensor entries (skip __metadata__) */
  int count = 0;
  cJSON *item;
  cJSON_ArrayForEach(item, root) {
    if (!item->string) goto fail_root_ex;
    if (strcmp(item->string, "__metadata__") != 0) count++;
  }

  PolySafetensorViewEx *views = calloc(count ? (size_t)count : 1, sizeof(PolySafetensorViewEx));
  if (!views) {
    cJSON_Delete(root);
    return NULL;
  }

  const uint8_t *data_region = data + 8 + (size_t)header_size;
  uint64_t data_region_len = (uint64_t)len - 8 - header_size;

  char *metadata = NULL;
  int vi = 0;
  cJSON_ArrayForEach(item, root) {
    if (strcmp(item->string, "__metadata__") == 0) {
      if (metadata_out) {
        char *meta_str = cJSON_PrintUnformatted(item);
        if (!meta_str) goto fail_ex;
        free(metadata);
        metadata = meta_str;
      }
      continue;
    }

    /* Parse dtype */
    cJSON *dtype_val = cJSON_GetObjectItemCaseSensitive(item, "dtype");
    if (!dtype_val || !cJSON_IsString(dtype_val)) {
      fprintf(stderr, "poly_safetensors_decode_ex: missing dtype for '%s'\n", item->string);
      goto fail_ex;
    }

    PolySafetensorDType dtype;
    if (parse_safetensor_dtype(dtype_val->valuestring, &dtype) != 0) {
      fprintf(
          stderr, "poly_safetensors_decode_ex: unsupported dtype '%s' for '%s'\n",
          dtype_val->valuestring, item->string
      );
      goto fail_ex;
    }

    /* Parse shape */
    cJSON *shape_arr = cJSON_GetObjectItemCaseSensitive(item, "shape");
    if (!shape_arr || !cJSON_IsArray(shape_arr)) {
      fprintf(stderr, "poly_safetensors_decode_ex: missing shape for '%s'\n", item->string);
      goto fail_ex;
    }
    int ndim = cJSON_GetArraySize(shape_arr);
    if (ndim > 8) {
      fprintf(stderr, "poly_safetensors_decode_ex: ndim %d > 8 for '%s'\n", ndim, item->string);
      goto fail_ex;
    }

    int64_t numel = 0;
    for (int d = 0; d < ndim; d++) {
      cJSON *dim = cJSON_GetArrayItem(shape_arr, d);
      if (!json_nonnegative_i64(dim, &views[vi].shape[d])) goto fail_ex;
    }
    if (!compute_numel_checked(views[vi].shape, ndim, &numel)) goto fail_ex;
    views[vi].ndim = ndim;
    views[vi].numel = numel;
    views[vi].dtype = dtype;

    /* Parse data_offsets */
    cJSON *offs_arr = cJSON_GetObjectItemCaseSensitive(item, "data_offsets");
    if (!offs_arr || !cJSON_IsArray(offs_arr) || cJSON_GetArraySize(offs_arr) != 2) goto fail_ex;
    uint64_t start = 0, end = 0;
    if (!json_nonnegative_u64(cJSON_GetArrayItem(offs_arr, 0), &start) ||
        !json_nonnegative_u64(cJSON_GetArrayItem(offs_arr, 1), &end))
      goto fail_ex;

    if (end < start || end > data_region_len) {
      fprintf(
          stderr, "poly_safetensors_decode_ex: data_offsets out of bounds for '%s'\n", item->string
      );
      goto fail_ex;
    }
    int elem_size = poly_safetensor_dtype_size(dtype);
    if (elem_size == 0 || (uint64_t)numel > UINT64_MAX / (uint64_t)elem_size ||
        (end - start) != (uint64_t)numel * (uint64_t)elem_size) {
      fprintf(stderr, "poly_safetensors_decode_ex: data size mismatch for '%s'\n", item->string);
      goto fail_ex;
    }

    views[vi].name = strdup(item->string);
    if (!views[vi].name) goto fail_ex;
    views[vi].raw_data = data_region + start;
    vi++;
  }

  cJSON_Delete(root);
  *n_out = count;
  if (metadata_out) *metadata_out = metadata;
  return views;

fail_root_ex:
  cJSON_Delete(root);
  return NULL;

fail_ex:
  free(metadata);
  for (int i = 0; i < vi; i++)
    free(views[i].name);
  free(views);
  cJSON_Delete(root);
  return NULL;
}

/* F16/BF16 to F32 conversion */

static float f16_to_f32(uint16_t h) {
  uint32_t sign = ((uint32_t)h & 0x8000) << 16;
  uint32_t exponent = (h >> 10) & 0x1F;
  uint32_t mantissa = h & 0x03FF;

  if (exponent == 0) {
    if (mantissa == 0) {
      /* Signed zero */
      float f;
      uint32_t bits = sign;
      memcpy(&f, &bits, sizeof(f));
      return f;
    }
    /* Denormalized: convert to normalized f32 */
    exponent = 1;
    while (!(mantissa & 0x0400)) {
      mantissa <<= 1;
      exponent--;
    }
    mantissa &= 0x03FF;
    uint32_t bits = sign | ((uint32_t)(exponent + 127 - 15) << 23) | ((uint32_t)mantissa << 13);
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
  } else if (exponent == 31) {
    /* Inf or NaN */
    uint32_t bits = sign | 0x7F800000 | ((uint32_t)mantissa << 13);
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
  }

  /* Normalized */
  uint32_t bits = sign | ((uint32_t)(exponent + 127 - 15) << 23) | ((uint32_t)mantissa << 13);
  float f;
  memcpy(&f, &bits, sizeof(f));
  return f;
}

static float bf16_to_f32(uint16_t h) {
  uint32_t bits = (uint32_t)h << 16;
  float f;
  memcpy(&f, &bits, sizeof(f));
  return f;
}

float *poly_safetensors_to_f32(const PolySafetensorViewEx *view) {
  if (!view || view->numel <= 0) return NULL;

  float *out = malloc(view->numel * sizeof(float));
  if (!out) return NULL;

  const uint8_t *raw = view->raw_data;

  switch (view->dtype) {
  case POLY_ST_F32:
    memcpy(out, raw, view->numel * sizeof(float));
    break;

  case POLY_ST_F16:
    for (int64_t i = 0; i < view->numel; i++) {
      uint16_t h;
      memcpy(&h, raw + i * 2, 2);
      out[i] = f16_to_f32(h);
    }
    break;

  case POLY_ST_BF16:
    for (int64_t i = 0; i < view->numel; i++) {
      uint16_t h;
      memcpy(&h, raw + i * 2, 2);
      out[i] = bf16_to_f32(h);
    }
    break;

  case POLY_ST_F64:
    for (int64_t i = 0; i < view->numel; i++) {
      double d;
      memcpy(&d, raw + i * 8, 8);
      out[i] = (float)d;
    }
    break;

  case POLY_ST_I64:
    for (int64_t i = 0; i < view->numel; i++) {
      int64_t v;
      memcpy(&v, raw + i * 8, 8);
      out[i] = (float)v;
    }
    break;

  case POLY_ST_I32:
    for (int64_t i = 0; i < view->numel; i++) {
      int32_t v;
      memcpy(&v, raw + i * 4, 4);
      out[i] = (float)v;
    }
    break;

  case POLY_ST_I16:
    for (int64_t i = 0; i < view->numel; i++) {
      int16_t v;
      memcpy(&v, raw + i * 2, 2);
      out[i] = (float)v;
    }
    break;

  case POLY_ST_I8:
    for (int64_t i = 0; i < view->numel; i++)
      out[i] = (float)(int8_t)raw[i];
    break;

  case POLY_ST_U8:
    for (int64_t i = 0; i < view->numel; i++)
      out[i] = (float)raw[i];
    break;

  case POLY_ST_BOOL:
    for (int64_t i = 0; i < view->numel; i++)
      out[i] = raw[i] ? 1.0f : 0.0f;
    break;

  case POLY_ST_U16:
    for (int64_t i = 0; i < view->numel; i++) {
      uint16_t v;
      memcpy(&v, raw + i * 2, 2);
      out[i] = (float)v;
    }
    break;

  case POLY_ST_U32:
    for (int64_t i = 0; i < view->numel; i++) {
      uint32_t v;
      memcpy(&v, raw + i * 4, 4);
      out[i] = (float)v;
    }
    break;

  case POLY_ST_U64:
    for (int64_t i = 0; i < view->numel; i++) {
      uint64_t v;
      memcpy(&v, raw + i * 8, 8);
      out[i] = (float)v;
    }
    break;

  default:
    free(out);
    return NULL;
  }

  return out;
}
