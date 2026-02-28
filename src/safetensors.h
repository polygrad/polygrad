/*
 * poly_safetensors.h -- Safetensors encode/decode
 *
 * Format: 8-byte LE header_size + JSON header + raw tensor data
 * JSON header maps tensor names to {dtype, shape, data_offsets}.
 *
 * Two APIs:
 *   poly_safetensors_decode()    -- F32-only (original, backward-compatible)
 *   poly_safetensors_decode_ex() -- Multi-dtype (F16, BF16, F32, F64, int types)
 */

#ifndef POLY_SAFETENSORS_H
#define POLY_SAFETENSORS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Encode ─────────────────────────────────────────────────────────── */

typedef struct {
  const char *name;
  const float *data;
  const int64_t *shape;
  int ndim;
} PolySafetensorEntry;

/* Encode entries into safetensors binary format.
 * Keys are sorted for determinism. Caller frees returned bytes.
 * metadata_json: optional JSON string stored in __metadata__ (NULL to skip).
 * Returns NULL on error (sets *out_len to 0). */
uint8_t *poly_safetensors_encode(const PolySafetensorEntry *entries, int n,
                                 const char *metadata_json,
                                 int *out_len);

/* ── Decode (F32-only, original API) ───────────────────────────────── */

typedef struct {
  char *name;            /* heap-allocated (caller frees) */
  const float *data;     /* points into input bytes (zero-copy) */
  int64_t shape[8];
  int ndim;
  int64_t numel;
} PolySafetensorView;

/* Decode safetensors bytes into views. Views point into data (zero-copy).
 * Caller frees returned array and each view's name.
 * metadata_out: if non-NULL, receives heap-allocated metadata JSON string
 *               (NULL if no metadata in file; caller frees).
 * Returns NULL on error (sets *n_out to 0). */
PolySafetensorView *poly_safetensors_decode(const uint8_t *data, int len,
                                            int *n_out,
                                            char **metadata_out);

/* ── Multi-dtype decode ────────────────────────────────────────────── */

typedef enum {
  POLY_ST_F32,   POLY_ST_F16,   POLY_ST_BF16,
  POLY_ST_F64,   POLY_ST_I64,   POLY_ST_I32,
  POLY_ST_I16,   POLY_ST_I8,    POLY_ST_U8,
  POLY_ST_BOOL
} PolySafetensorDType;

typedef struct {
  char *name;                  /* heap-allocated (caller frees) */
  const uint8_t *raw_data;    /* zero-copy pointer into input buffer */
  int64_t shape[8];
  int ndim;
  int64_t numel;
  PolySafetensorDType dtype;
} PolySafetensorViewEx;

/* Decode safetensors bytes, accepting any dtype.
 * Returns zero-copy views with dtype tag. Caller frees array + names.
 * len is int64_t to support files > 2GB. */
PolySafetensorViewEx *poly_safetensors_decode_ex(
    const uint8_t *data, int64_t len,
    int *n_out, char **metadata_out);

/* Convert a multi-dtype view to F32 data.
 * Returns malloc'd float array (numel elements). Caller frees.
 * Handles F16, BF16 -> F32 conversion with IEEE 754 correctness.
 * Returns NULL on unsupported dtype. */
float *poly_safetensors_to_f32(const PolySafetensorViewEx *view);

/* Get byte size per element for a safetensor dtype. */
int poly_safetensor_dtype_size(PolySafetensorDType dtype);

#ifdef __cplusplus
}
#endif

#endif /* POLY_SAFETENSORS_H */
