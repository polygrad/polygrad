/*
 * gguf_decode.c -- GGUF binary format decoder
 *
 * Parses GGUF v2/v3 files into PolyGgufDecoded. Zero model-specific logic.
 * Tensor data pointers are zero-copy into the caller's buffer.
 *
 * Format reference: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
 *
 * GGUF binary layout:
 *   [magic:4] [version:u32] [n_tensors:u64] [n_kv:u64]
 *   [kv entries...]
 *   [tensor info entries...]
 *   [alignment padding]
 *   [tensor data (contiguous, aligned)]
 */

#define _POSIX_C_SOURCE 200809L
#include "gguf_decode.h"
#include "import_error.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <limits.h>

/* GGML type codes */

#define GGML_TYPE_F32 0
#define GGML_TYPE_F16 1
#define GGML_TYPE_Q4_0 2
#define GGML_TYPE_Q4_1 3
#define GGML_TYPE_Q8_0 8
#define GGML_TYPE_Q4_K 12
#define GGML_TYPE_Q5_K 13
#define GGML_TYPE_Q6_K 14
/* Pinned llm/gguf.py _GGML_NATIVE;16–18 are quantized types, not integers. */
#define GGML_TYPE_I8 24
#define GGML_TYPE_I16 25
#define GGML_TYPE_I32 26
#define GGML_TYPE_BF16 30

/* GGUF KV value type codes */
#define GGUF_TYPE_UINT8 0
#define GGUF_TYPE_INT8 1
#define GGUF_TYPE_UINT16 2
#define GGUF_TYPE_INT16 3
#define GGUF_TYPE_UINT32 4
#define GGUF_TYPE_INT32 5
#define GGUF_TYPE_FLOAT32 6
#define GGUF_TYPE_BOOL 7
#define GGUF_TYPE_STRING 8
#define GGUF_TYPE_ARRAY 9
#define GGUF_TYPE_UINT64 10
#define GGUF_TYPE_INT64 11
#define GGUF_TYPE_FLOAT64 12

/* Reader helpers */

typedef struct {
  const uint8_t *data;
  int64_t len;
  int64_t pos;
  int failed;
} GgufReader;

static int reader_ok(GgufReader *r, int64_t need) {
  /* Python struct.unpack raises on short reads. Sticky failure prevents a
   * truncated scalar's zero return from becoming valid metadata. */
  if (r->failed || need < 0 || need > r->len - r->pos) {
    r->failed = 1;
    return 0;
  }
  return 1;
}

static uint8_t read_u8(GgufReader *r) {
  if (!reader_ok(r, 1)) return 0;
  return r->data[r->pos++];
}

static uint16_t read_u16(GgufReader *r) {
  if (!reader_ok(r, 2)) return 0;
  uint16_t v;
  memcpy(&v, r->data + r->pos, 2);
  r->pos += 2;
  return v;
}

static uint32_t read_u32(GgufReader *r) {
  if (!reader_ok(r, 4)) return 0;
  uint32_t v;
  memcpy(&v, r->data + r->pos, 4);
  r->pos += 4;
  return v;
}

static int32_t read_i32(GgufReader *r) {
  if (!reader_ok(r, 4)) return 0;
  int32_t v;
  memcpy(&v, r->data + r->pos, 4);
  r->pos += 4;
  return v;
}

static uint64_t read_u64(GgufReader *r) {
  if (!reader_ok(r, 8)) return 0;
  uint64_t v;
  memcpy(&v, r->data + r->pos, 8);
  r->pos += 8;
  return v;
}

static int64_t read_i64(GgufReader *r) {
  if (!reader_ok(r, 8)) return 0;
  int64_t v;
  memcpy(&v, r->data + r->pos, 8);
  r->pos += 8;
  return v;
}

static float read_f32(GgufReader *r) {
  if (!reader_ok(r, 4)) return 0;
  float v;
  memcpy(&v, r->data + r->pos, 4);
  r->pos += 4;
  return v;
}

static double read_f64(GgufReader *r) {
  if (!reader_ok(r, 8)) return 0;
  double v;
  memcpy(&v, r->data + r->pos, 8);
  r->pos += 8;
  return v;
}

/* Read GGUF string: u64 length + bytes (NOT null-terminated) */
static char *read_string(GgufReader *r) {
  uint64_t slen = read_u64(r);
  if (slen > INT_MAX || slen >= SIZE_MAX) {
    r->failed = 1;
    return NULL;
  }
  if (!reader_ok(r, (int64_t)slen)) return NULL;
  char *s = malloc((size_t)slen + 1);
  if (!s) {
    r->failed = 1;
    return NULL;
  }
  memcpy(s, r->data + r->pos, slen);
  s[slen] = '\0';
  r->pos += (int64_t)slen;
  return s;
}

/* KV value reader */

static void free_kv_value(PolyGgufKV *kv) {
  if (kv->type == GGUF_TYPE_STRING) free(kv->val.s.str);
  if (kv->type == GGUF_TYPE_ARRAY && kv->arr_data) {
    if (kv->arr_type == GGUF_TYPE_STRING) {
      char **strings = kv->arr_data;
      for (int i = 0; i < kv->arr_count; i++)
        free(strings[i]);
    }
    free(kv->arr_data);
  }
}

static int read_kv_value(GgufReader *r, PolyGgufKV *kv) {
  switch (kv->type) {
  case GGUF_TYPE_UINT8:
    kv->val.u64 = read_u8(r);
    break;
  case GGUF_TYPE_INT8:
    kv->val.i64 = (int8_t)read_u8(r);
    break;
  case GGUF_TYPE_UINT16:
    kv->val.u64 = read_u16(r);
    break;
  case GGUF_TYPE_INT16:
    kv->val.i64 = (int16_t)read_u16(r);
    break;
  case GGUF_TYPE_UINT32:
    kv->val.u64 = read_u32(r);
    break;
  case GGUF_TYPE_INT32:
    kv->val.i64 = read_i32(r);
    break;
  case GGUF_TYPE_FLOAT32:
    kv->val.f64 = read_f32(r);
    break;
  case GGUF_TYPE_BOOL:
    kv->val.u64 = read_u8(r);
    break;
  case GGUF_TYPE_STRING:
    kv->val.s.str = read_string(r);
    kv->val.s.len = kv->val.s.str ? (int)strlen(kv->val.s.str) : 0;
    break;
  case GGUF_TYPE_UINT64:
    kv->val.u64 = read_u64(r);
    break;
  case GGUF_TYPE_INT64:
    kv->val.i64 = read_i64(r);
    break;
  case GGUF_TYPE_FLOAT64:
    kv->val.f64 = read_f64(r);
    break;
  case GGUF_TYPE_ARRAY: {
    int32_t elem_type = read_i32(r);
    uint64_t count = read_u64(r);
    static const int min_bytes[] = {1, 1, 2, 2, 4, 4, 4, 1, 8, 12, 8, 8, 8};
    if (r->failed || elem_type < 0 || elem_type > GGUF_TYPE_FLOAT64 || count > INT_MAX ||
        count > (uint64_t)(r->len - r->pos) / min_bytes[elem_type])
      return -1;
    kv->arr_type = elem_type;
    kv->arr_count = (int)count;
    if (elem_type == GGUF_TYPE_STRING) {
      /* Store string array */
      char **strs = calloc(count, sizeof(char *));
      if (count && !strs) return -1;
      kv->arr_data = strs;
      for (uint64_t i = 0; i < count && !r->failed; i++)
        strs[i] = read_string(r);
    } else if (elem_type == GGUF_TYPE_INT32 || elem_type == GGUF_TYPE_UINT32) {
      /* Store int32 array */
      int32_t *ints = calloc(count, sizeof(int32_t));
      if (count && !ints) return -1;
      kv->arr_data = ints;
      for (uint64_t i = 0; i < count; i++)
        ints[i] = read_i32(r);
    } else {
      /* Skip other array types */
      for (uint64_t i = 0; i < count; i++) {
        PolyGgufKV tmp = {.type = elem_type};
        int rc = read_kv_value(r, &tmp);
        free_kv_value(&tmp);
        if (rc != 0) return -1;
      }
      kv->val.u64 = count;
    }
    break;
  }
  default:
    poly_import_error_set(POLY_IMPORT_ERR_PARSE, "unknown GGUF KV type %d", kv->type);
    return -1;
  }
  return r->failed ? -1 : 0;
}

/* Bytes per element for GGML types */

/* Returns bytes per block and elements per block for quantized types.
 * For native types, block_size=1. */
static void ggml_type_info(int type, int *block_bytes, int *block_elems) {
  switch (type) {
  case GGML_TYPE_F32:
    *block_bytes = 4;
    *block_elems = 1;
    break;
  case GGML_TYPE_F16:
    *block_bytes = 2;
    *block_elems = 1;
    break;
  case GGML_TYPE_BF16:
    *block_bytes = 2;
    *block_elems = 1;
    break;
  case GGML_TYPE_I8:
    *block_bytes = 1;
    *block_elems = 1;
    break;
  case GGML_TYPE_I16:
    *block_bytes = 2;
    *block_elems = 1;
    break;
  case GGML_TYPE_I32:
    *block_bytes = 4;
    *block_elems = 1;
    break;
  case GGML_TYPE_Q4_0:
    *block_bytes = 18;
    *block_elems = 32;
    break;
  case GGML_TYPE_Q4_1:
    *block_bytes = 20;
    *block_elems = 32;
    break;
  case GGML_TYPE_Q8_0:
    *block_bytes = 34;
    *block_elems = 32;
    break;
  case GGML_TYPE_Q4_K:
    *block_bytes = 144;
    *block_elems = 256;
    break;
  case GGML_TYPE_Q5_K:
    *block_bytes = 176;
    *block_elems = 256;
    break;
  case GGML_TYPE_Q6_K:
    *block_bytes = 210;
    *block_elems = 256;
    break;
  default:
    *block_bytes = 0;
    *block_elems = 0;
    break;
  }
}

/* Map GGML type code to unified POLY_DECODED_* code */
static int ggml_to_decoded_dtype(int ggml_type) {
  switch (ggml_type) {
  case 0:
    return POLY_DECODED_F32;
  case 1:
    return POLY_DECODED_F16;
  case 2:
    return POLY_DECODED_Q4_0;
  case 3:
    return POLY_DECODED_Q4_1;
  case 8:
    return POLY_DECODED_Q8_0;
  case 12:
    return POLY_DECODED_Q4_K;
  case 13:
    return POLY_DECODED_Q5_K;
  case 14:
    return POLY_DECODED_Q6_K;
  case GGML_TYPE_I8:
    return POLY_DECODED_I8;
  case GGML_TYPE_I16:
    return POLY_DECODED_I16;
  case GGML_TYPE_I32:
    return POLY_DECODED_I32;
  case 30:
    return POLY_DECODED_BF16;
  default:
    return -1;
  }
}

/* Main decode */

int poly_gguf_decode(const uint8_t *data, int64_t len, PolyGgufDecoded **out) {
  if (!out) return -1;
  *out = NULL;

  if (!data || len < 24) {
    poly_import_error_set(POLY_IMPORT_ERR_PARSE, "GGUF data too short");
    return -1;
  }

  GgufReader r = {.data = data, .len = len, .pos = 0};

  /* Magic */
  if (data[0] != 'G' || data[1] != 'G' || data[2] != 'U' || data[3] != 'F') {
    poly_import_error_set(POLY_IMPORT_ERR_PARSE, "invalid GGUF magic");
    return -1;
  }
  r.pos = 4;

  uint32_t version = read_u32(&r);
  if (version != 2 && version != 3) {
    poly_import_error_set(POLY_IMPORT_ERR_PARSE, "unsupported GGUF version %u", version);
    return -1;
  }

  uint64_t n_tensors = read_u64(&r);
  uint64_t n_kv = read_u64(&r);
  /* Bound wire counts before narrowing or allocating. Even an empty-name
   * scalar tensor consumes24 bytes; a scalar KV consumes at least13. */
  if (n_tensors > INT_MAX || n_kv > INT_MAX || n_tensors > (uint64_t)(len - 24) / 24 ||
      n_kv > (uint64_t)(len - 24) / 13) {
    poly_import_error_set(POLY_IMPORT_ERR_PARSE, "invalid GGUF table counts");
    return -1;
  }

  typedef struct {
    uint64_t offset;
    int type;
  } TensorInfo;
  TensorInfo *tinfos = NULL;
  PolyGgufDecoded *gguf = calloc(1, sizeof(*gguf));
  if (!gguf) return -1;
  gguf->arch = "";

  /* Parse KV metadata */
  PolyGgufKV *kv = n_kv ? calloc((size_t)n_kv, sizeof(*kv)) : NULL;
  if (n_kv && !kv) goto fail;
  gguf->kv = kv;
  gguf->n_kv = (int)n_kv;

  for (uint64_t i = 0; i < n_kv; i++) {
    kv[i].key = read_string(&r);
    kv[i].type = read_i32(&r);
    if (r.failed || read_kv_value(&r, &kv[i]) != 0) goto fail;
    if (kv[i].key && strcmp(kv[i].key, "general.architecture") == 0 &&
        kv[i].type == GGUF_TYPE_STRING && kv[i].val.s.str)
      gguf->arch = kv[i].val.s.str;
  }

  /* Parse tensor info entries */
  tinfos = n_tensors ? calloc((size_t)n_tensors, sizeof(*tinfos)) : NULL;
  PolyDecodedTensor *tensors = n_tensors ? calloc((size_t)n_tensors, sizeof(*tensors)) : NULL;
  gguf->tensors = tensors;
  if (tensors) gguf->n_tensors = (int)n_tensors;
  if (n_tensors && (!tinfos || !tensors)) goto fail;
  for (uint64_t i = 0; i < n_tensors; i++) {
    tensors[i].name = read_string(&r);
    uint32_t nd = read_u32(&r);
    if (r.failed || nd > 8) goto fail;
    tensors[i].ndim = (int)nd;
    int64_t numel = 1;
    /* GGUF stores dims in reverse order (innermost first) */
    for (uint32_t d = 0; d < nd; d++) {
      uint64_t dim = read_u64(&r);
      if (r.failed || dim > INT64_MAX || (dim && numel > INT64_MAX / (int64_t)dim)) goto fail;
      tensors[i].shape[nd - 1 - d] = (int64_t)dim;
      numel *= (int64_t)dim;
    }
    tensors[i].numel = numel;
    tinfos[i].type = (int)read_u32(&r);
    tinfos[i].offset = read_u64(&r);
    tensors[i].dtype = ggml_to_decoded_dtype(tinfos[i].type);
    if (r.failed || tensors[i].dtype < 0) goto fail;
  }

  /* Compute data section start (aligned) */
  int64_t alignment = 32;
  for (uint64_t i = 0; i < n_kv; i++) {
    if (kv[i].key && strcmp(kv[i].key, "general.alignment") == 0) {
      if (kv[i].type == GGUF_TYPE_UINT32)
        alignment = (int64_t)kv[i].val.u64;
      else if (kv[i].type == GGUF_TYPE_INT32)
        alignment = (int64_t)kv[i].val.i64;
      break;
    }
  }
  if (alignment <= 0) goto fail;
  int64_t padding = r.pos % alignment ? alignment - r.pos % alignment : 0;
  if (n_tensors && !reader_ok(&r, padding)) goto fail;
  int64_t data_start = n_tensors ? r.pos + padding : r.pos;

  /* Build decoded tensors */
  for (uint64_t i = 0; i < n_tensors; i++) {
    uint64_t available = (uint64_t)(len - data_start);
    if (tinfos[i].offset > available) goto fail;
    available -= tinfos[i].offset;
    int block_bytes, block_elems;
    ggml_type_info(tinfos[i].type, &block_bytes, &block_elems);
    if (!block_bytes || !block_elems) goto fail;
    uint64_t numel = (uint64_t)tensors[i].numel;
    /* The supported quantizers consume whole blocks. Pinned GGUF reshape
     * rejects a remainder too; accepting one would leave uninitialized values. */
    if (numel % block_elems) goto fail;
    uint64_t blocks = numel / block_elems;
    if (blocks > available / block_bytes) goto fail;
    /* Publish borrowed bytes only after the complete typed/block span fits. */
    tensors[i].data = data + data_start + tinfos[i].offset;
  }

  free(tinfos);

  *out = gguf;
  return 0;

fail:
  free(tinfos);
  poly_gguf_decoded_free(gguf);
  poly_import_error_set(POLY_IMPORT_ERR_PARSE, "invalid or unallocatable GGUF data");
  return -1;
}

void poly_gguf_decoded_free(PolyGgufDecoded *gguf) {
  if (!gguf) return;
  for (int i = 0; i < gguf->n_kv; i++) {
    free(gguf->kv[i].key);
    free_kv_value(&gguf->kv[i]);
  }
  free(gguf->kv);
  for (int i = 0; i < gguf->n_tensors; i++)
    free(gguf->tensors[i].name);
  free(gguf->tensors);
  free(gguf);
}

/* KV lookup helpers */

int poly_gguf_kv_int(const PolyGgufDecoded *g, const char *key, int def) {
  if (!g) return def;
  for (int i = 0; i < g->n_kv; i++) {
    if (g->kv[i].key && strcmp(g->kv[i].key, key) == 0) {
      switch (g->kv[i].type) {
      case GGUF_TYPE_UINT8:
      case GGUF_TYPE_UINT16:
      case GGUF_TYPE_UINT32:
      case GGUF_TYPE_UINT64:
        return (int)g->kv[i].val.u64;
      case GGUF_TYPE_INT8:
      case GGUF_TYPE_INT16:
      case GGUF_TYPE_INT32:
      case GGUF_TYPE_INT64:
        return (int)g->kv[i].val.i64;
      default:
        return def;
      }
    }
  }
  return def;
}

double poly_gguf_kv_float(const PolyGgufDecoded *g, const char *key, double def) {
  if (!g) return def;
  for (int i = 0; i < g->n_kv; i++) {
    if (g->kv[i].key && strcmp(g->kv[i].key, key) == 0) {
      if (g->kv[i].type == GGUF_TYPE_FLOAT32 || g->kv[i].type == GGUF_TYPE_FLOAT64)
        return g->kv[i].val.f64;
      return def;
    }
  }
  return def;
}

const char **poly_gguf_kv_string_array(const PolyGgufDecoded *g, const char *key, int *count_out) {
  if (!g) {
    if (count_out) *count_out = 0;
    return NULL;
  }
  for (int i = 0; i < g->n_kv; i++) {
    if (g->kv[i].key && strcmp(g->kv[i].key, key) == 0 && g->kv[i].type == GGUF_TYPE_ARRAY &&
        g->kv[i].arr_type == GGUF_TYPE_STRING) {
      if (count_out) *count_out = g->kv[i].arr_count;
      return (const char **)g->kv[i].arr_data;
    }
  }
  if (count_out) *count_out = 0;
  return NULL;
}

const int32_t *poly_gguf_kv_int_array(const PolyGgufDecoded *g, const char *key, int *count_out) {
  if (!g) {
    if (count_out) *count_out = 0;
    return NULL;
  }
  for (int i = 0; i < g->n_kv; i++) {
    if (g->kv[i].key && strcmp(g->kv[i].key, key) == 0 && g->kv[i].type == GGUF_TYPE_ARRAY &&
        (g->kv[i].arr_type == GGUF_TYPE_INT32 || g->kv[i].arr_type == GGUF_TYPE_UINT32)) {
      if (count_out) *count_out = g->kv[i].arr_count;
      return (const int32_t *)g->kv[i].arr_data;
    }
  }
  if (count_out) *count_out = 0;
  return NULL;
}

const char *poly_gguf_kv_string(const PolyGgufDecoded *g, const char *key, const char *def) {
  if (!g) return def;
  for (int i = 0; i < g->n_kv; i++) {
    if (g->kv[i].key && strcmp(g->kv[i].key, key) == 0 && g->kv[i].type == GGUF_TYPE_STRING)
      return g->kv[i].val.s.str;
  }
  return def;
}
