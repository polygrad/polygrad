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
 * Format-neutral view of one decoded weight tensor.
 *
 * dtype uses format-specific type codes:
 *   HF: PolySafetensorDType values (0=F32, 1=F16, 2=BF16, ...)
 *   GGUF: GGML type codes (0=F32, 1=F16, 2=Q4_0, 8=Q8_0, ...)
 *
 * Use poly_decoded_tensor_to_f32() to convert any supported dtype to F32.
 */
typedef struct {
    char       *name;       /* owned string (freed by parent decoded_free) */
    const void *data;       /* raw bytes in format-native dtype (zero-copy) */
    int64_t     shape[8];
    int         ndim;
    int64_t     numel;
    int         dtype;      /* format-specific dtype code */
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
