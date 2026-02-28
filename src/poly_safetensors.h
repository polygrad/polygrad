/*
 * poly_safetensors.h -- Safetensors encode/decode for f32 tensors
 *
 * Format: 8-byte LE header_size + JSON header + raw tensor data
 * JSON header maps tensor names to {dtype, shape, data_offsets}.
 * Only f32 tensors supported (IR v1 scope).
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

/* ── Decode ─────────────────────────────────────────────────────────── */

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

#ifdef __cplusplus
}
#endif

#endif /* POLY_SAFETENSORS_H */
