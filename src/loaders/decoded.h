/*
 * decoded.h -- Format-neutral decoded weight tensor type
 *
 * Shared by all format decoders (HF safetensors, GGUF, ONNX).
 * Format-specific decoded result types live in their own headers.
 *
 * LIFETIME: tensor name and data pointers are borrowed from the parent
 * decoded object. The caller must keep source buffers (weight file bytes)
 * alive until the decoded object is freed.
 */

#ifndef POLY_DECODED_H
#define POLY_DECODED_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Unified decoded dtype enum. No collisions between formats.
 * Decoders map format-native codes to these at decode time.
 */
enum {
  /* Native types (shared by HF and GGUF) */
  POLY_DECODED_F32 = 0,
  POLY_DECODED_F16 = 1,
  POLY_DECODED_BF16 = 2,
  POLY_DECODED_F64 = 3,
  POLY_DECODED_I64 = 4,
  POLY_DECODED_I32 = 5,
  POLY_DECODED_I16 = 6,
  POLY_DECODED_I8 = 7,
  POLY_DECODED_U8 = 8,
  POLY_DECODED_BOOL = 9,
  /* GGML quantized types (no HF equivalent) */
  POLY_DECODED_Q4_0 = 100,
  POLY_DECODED_Q4_1 = 101,
  POLY_DECODED_Q8_0 = 102,
  POLY_DECODED_Q4_K = 103,
  POLY_DECODED_Q5_K = 104,
  POLY_DECODED_Q6_K = 105,
};

typedef struct {
  char *name; /* owned string (freed by parent decoded_free) */
  const void *data; /* raw bytes in format-native dtype (zero-copy) */
  int64_t shape[8];
  int ndim;
  int64_t numel;
  int dtype; /* POLY_DECODED_* unified code */
} PolyDecodedTensor;

/*
 * Convert a decoded tensor to a newly allocated F32 buffer.
 *
 * HF: handles F32 (copy), F16 (convert), BF16 (convert), I32 (convert).
 * GGUF: handles Q4_0, Q8_0, etc. (dequantize to F32).
 *
 * Returns malloc'd float array (numel elements). Caller frees.
 * Returns NULL on unsupported dtype or error.
 */
float *poly_decoded_tensor_to_f32(const PolyDecodedTensor *t);

#ifdef __cplusplus
}
#endif

#endif /* POLY_DECODED_H */
