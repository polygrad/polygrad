/*
 * decoded.c -- poly_decoded_tensor_to_f32 implementation
 *
 * Converts decoded tensors from any supported format dtype to F32.
 *
 * HF dtypes (PolySafetensorDType 0-9): delegate to safetensors codec.
 * GGML types (0=F32, 1=F16, 2=Q4_0, 8=Q8_0, ...): dequantize here.
 *
 * The dtype field is format-specific. HF and GGML both use 0=F32, 1=F16,
 * but diverge after that. We distinguish by checking if the dtype falls
 * in the GGML quantized range (2+) which has no HF equivalent.
 * The caller knows which format their tensor came from.
 */

#include "decoded.h"
#include "../safetensors.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* GGML type codes (matching ggml.h) */
#define GGML_F32 0
#define GGML_F16 1
#define GGML_Q4_0 2
#define GGML_Q4_1 3
#define GGML_Q8_0 8
#define GGML_I8 16
#define GGML_I16 17
#define GGML_I32 18
#define GGML_BF16 30

/* F16/BF16 conversion (shared with safetensors) */

static float f16_to_f32(uint16_t h) {
  uint32_t sign = ((uint32_t)h & 0x8000) << 16;
  uint32_t exponent = (h >> 10) & 0x1F;
  uint32_t mantissa = h & 0x03FF;
  uint32_t result;
  if (exponent == 0) {
    if (mantissa == 0) {
      result = sign;
    } else {
      exponent = 1;
      while (!(mantissa & 0x0400)) {
        mantissa <<= 1;
        exponent--;
      }
      mantissa &= 0x03FF;
      result = sign | ((127 - 15 + exponent) << 23) | (mantissa << 13);
    }
  } else if (exponent == 31) {
    result = sign | 0x7F800000 | (mantissa << 13);
  } else {
    result = sign | ((exponent + 112) << 23) | (mantissa << 13);
  }
  float f;
  memcpy(&f, &result, sizeof(f));
  return f;
}

static float bf16_to_f32(uint16_t h) {
  uint32_t bits = (uint32_t)h << 16;
  float f;
  memcpy(&f, &bits, sizeof(f));
  return f;
}

/* GGML Q4_0 dequantization */
/* Block: 2 bytes f16 scale + 16 bytes (32 x 4-bit unsigned) = 18 bytes */

static void dequant_q4_0(const uint8_t *block, float *out, int64_t n_blocks) {
  for (int64_t b = 0; b < n_blocks; b++) {
    float scale = f16_to_f32(*(const uint16_t *)(block + b * 18));
    const uint8_t *qs = block + b * 18 + 2;
    /* Interleaved: low nibble, high nibble per byte */
    for (int j = 0; j < 16; j++) {
      out[b * 32 + j * 2] = ((int)(qs[j] & 0x0F) - 8) * scale;
      out[b * 32 + j * 2 + 1] = ((int)(qs[j] >> 4) - 8) * scale;
    }
  }
}

/* GGML Q4_1 dequantization */
/* Block: 2 bytes f16 scale + 2 bytes f16 min + 16 bytes = 20 bytes */

static void dequant_q4_1(const uint8_t *block, float *out, int64_t n_blocks) {
  for (int64_t b = 0; b < n_blocks; b++) {
    float d = f16_to_f32(*(const uint16_t *)(block + b * 20));
    float m = f16_to_f32(*(const uint16_t *)(block + b * 20 + 2));
    const uint8_t *qs = block + b * 20 + 4;
    for (int j = 0; j < 16; j++) {
      out[b * 32 + j * 2] = (qs[j] & 0x0F) * d + m;
      out[b * 32 + j * 2 + 1] = (qs[j] >> 4) * d + m;
    }
  }
}

/* GGML Q8_0 dequantization */
/* Block: 2 bytes f16 scale + 32 bytes int8 = 34 bytes */

static void dequant_q8_0(const uint8_t *block, float *out, int64_t n_blocks) {
  for (int64_t b = 0; b < n_blocks; b++) {
    float scale = f16_to_f32(*(const uint16_t *)(block + b * 34));
    const int8_t *qs = (const int8_t *)(block + b * 34 + 2);
    for (int j = 0; j < 32; j++)
      out[b * 32 + j] = qs[j] * scale;
  }
}

/* GGML Q6_K dequantization */
/* Block: 128 bytes ql + 64 bytes qh + 16 bytes scales + 2 bytes d = 210 bytes
 * 256 elements per block.
 * ql: low 4 bits of each element (packed as nibbles, 2 per byte)
 * qh: high 2 bits of each element (packed as 2-bit pairs, 4 per byte)
 * scales: 16 x int8 scales (one per 16-element group)
 * d: f16 super-block scale
 * value = d * scale[group] * ((ql_low4 | qh_high2 << 4) - 32)
 */
static void dequant_q6_k(const uint8_t *block, float *out, int64_t n_blocks) {
  for (int64_t b = 0; b < n_blocks; b++) {
    const uint8_t *ql = block + b * 210; /* 128 bytes */
    const uint8_t *qh = block + b * 210 + 128; /* 64 bytes */
    const int8_t *sc = (const int8_t *)(block + b * 210 + 192); /* 16 bytes */
    float d = f16_to_f32(*(const uint16_t *)(block + b * 210 + 208));

    for (int g = 0; g < 16; g++) {
      float group_scale = d * sc[g];
      for (int j = 0; j < 16; j++) {
        int idx = g * 16 + j;
        /* Extract low 4 bits from ql (nibble-packed) */
        int ql_byte = idx / 2;
        int lo4 = (idx & 1) ? (ql[ql_byte] >> 4) : (ql[ql_byte] & 0x0F);
        /* Extract high 2 bits from qh (2-bit packed, 4 per byte) */
        int qh_byte = idx / 4;
        int qh_shift = (idx % 4) * 2;
        int hi2 = (qh[qh_byte] >> qh_shift) & 0x03;
        /* Combine: 6-bit value (0-63), subtract 32 for signed range */
        int val = (lo4 | (hi2 << 4)) - 32;
        out[b * 256 + idx] = group_scale * val;
      }
    }
  }
}

/* GGML native type conversion */

static float *ggml_to_f32(const void *data, int64_t numel, int ggml_type) {
  float *out = malloc((size_t)numel * sizeof(float));
  if (!out) return NULL;

  switch (ggml_type) {
  case GGML_F32:
    memcpy(out, data, (size_t)numel * sizeof(float));
    break;
  case GGML_F16: {
    const uint16_t *src = (const uint16_t *)data;
    for (int64_t i = 0; i < numel; i++)
      out[i] = f16_to_f32(src[i]);
    break;
  }
  case GGML_BF16: {
    const uint16_t *src = (const uint16_t *)data;
    for (int64_t i = 0; i < numel; i++)
      out[i] = bf16_to_f32(src[i]);
    break;
  }
  case GGML_I8: {
    const int8_t *src = (const int8_t *)data;
    for (int64_t i = 0; i < numel; i++)
      out[i] = (float)src[i];
    break;
  }
  case GGML_I16: {
    const int16_t *src = (const int16_t *)data;
    for (int64_t i = 0; i < numel; i++)
      out[i] = (float)src[i];
    break;
  }
  case GGML_I32: {
    const int32_t *src = (const int32_t *)data;
    for (int64_t i = 0; i < numel; i++)
      out[i] = (float)src[i];
    break;
  }
  case GGML_Q4_0:
    dequant_q4_0((const uint8_t *)data, out, numel / 32);
    break;
  case GGML_Q4_1:
    dequant_q4_1((const uint8_t *)data, out, numel / 32);
    break;
  case GGML_Q8_0:
    dequant_q8_0((const uint8_t *)data, out, numel / 32);
    break;
  case 14: /* Q6_K */
    dequant_q6_k((const uint8_t *)data, out, numel / 256);
    break;
  default:
    free(out);
    return NULL;
  }
  return out;
}

/* Public API */

/*
 * The dtype field interpretation depends on the source format.
 * HF safetensors: dtype = PolySafetensorDType (0=F32, 1=F16, 2=BF16, ...)
 * GGUF: dtype = GGML type code (0=F32, 1=F16, 2=Q4_0, 8=Q8_0, ...)
 *
 * We detect GGUF by checking for quantized type codes (2-14, 30, 39)
 * that have block-based dequantization. Types 0 (F32) and 1 (F16) are
 * the same in both formats. For HF-exclusive types (BF16=2 in safetensors
 * vs Q4_0=2 in GGML), the caller's context determines interpretation.
 *
 * In practice this ambiguity doesn't arise because:
 *   - HF tensors go through poly_hf_decode which sets dtype from safetensors
 *   - GGUF tensors go through poly_gguf_decode which sets dtype from GGML
 *   - poly_decoded_tensor_to_f32 is called by model importers that know
 *     their source format
 *
 * We resolve by trying safetensors first (works for F32/F16/BF16/I32 which
 * are the common HF types), then falling back to GGML dequant.
 */

float *poly_decoded_tensor_to_f32(const PolyDecodedTensor *t) {
  if (!t || !t->data || t->numel <= 0) return NULL;

  /* Unified dispatch on POLY_DECODED_* codes */
  float *out = malloc((size_t)t->numel * sizeof(float));
  if (!out) return NULL;

  switch (t->dtype) {
  case POLY_DECODED_F32:
    memcpy(out, t->data, (size_t)t->numel * sizeof(float));
    return out;
  case POLY_DECODED_F16: {
    const uint16_t *src = (const uint16_t *)t->data;
    for (int64_t i = 0; i < t->numel; i++)
      out[i] = f16_to_f32(src[i]);
    return out;
  }
  case POLY_DECODED_BF16: {
    const uint16_t *src = (const uint16_t *)t->data;
    for (int64_t i = 0; i < t->numel; i++)
      out[i] = bf16_to_f32(src[i]);
    return out;
  }
  case POLY_DECODED_I32: {
    const int32_t *src = (const int32_t *)t->data;
    for (int64_t i = 0; i < t->numel; i++)
      out[i] = (float)src[i];
    return out;
  }
  case POLY_DECODED_I16: {
    const int16_t *src = (const int16_t *)t->data;
    for (int64_t i = 0; i < t->numel; i++)
      out[i] = (float)src[i];
    return out;
  }
  case POLY_DECODED_I8: {
    const int8_t *src = (const int8_t *)t->data;
    for (int64_t i = 0; i < t->numel; i++)
      out[i] = (float)src[i];
    return out;
  }
  case POLY_DECODED_U8: {
    const uint8_t *src = (const uint8_t *)t->data;
    for (int64_t i = 0; i < t->numel; i++)
      out[i] = (float)src[i];
    return out;
  }
  case POLY_DECODED_Q4_0:
    dequant_q4_0((const uint8_t *)t->data, out, t->numel / 32);
    return out;
  case POLY_DECODED_Q4_1:
    dequant_q4_1((const uint8_t *)t->data, out, t->numel / 32);
    return out;
  case POLY_DECODED_Q8_0:
    dequant_q8_0((const uint8_t *)t->data, out, t->numel / 32);
    return out;
  case POLY_DECODED_Q6_K:
    dequant_q6_k((const uint8_t *)t->data, out, t->numel / 256);
    return out;
  default:
    free(out);
    return NULL;
  }
}
