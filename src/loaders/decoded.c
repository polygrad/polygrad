/*
 * decoded.c -- poly_decoded_tensor_to_f32 implementation
 *
 * Converts decoded tensors from any supported format dtype to F32.
 *
 * Decoders map wire dtypes to POLY_DECODED_* before reaching this boundary.
 * Quantized layouts follow tinygrad/llm/gguf.py:ggml_data_to_tensor.
 */

#include "decoded.h"
#include <stdlib.h>
#include <string.h>

/* File payloads are borrowed bytes and need not have native scalar alignment. */
static uint16_t read_u16(const uint8_t *bytes) {
  uint16_t value;
  memcpy(&value, bytes, sizeof(value));
  return value;
}

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
    float scale = f16_to_f32(read_u16(block + b * 18));
    const uint8_t *qs = block + b * 18 + 2;
    /* q_to_uint8 transposes the bit planes before flattening. */
    for (int j = 0; j < 16; j++) {
      out[b * 32 + j] = ((int)(qs[j] & 0x0F) - 8) * scale;
      out[b * 32 + j + 16] = ((int)(qs[j] >> 4) - 8) * scale;
    }
  }
}

/* GGML Q4_1 dequantization */
/* Block: 2 bytes f16 scale + 2 bytes f16 min + 16 bytes = 20 bytes */

static void dequant_q4_1(const uint8_t *block, float *out, int64_t n_blocks) {
  for (int64_t b = 0; b < n_blocks; b++) {
    float d = f16_to_f32(read_u16(block + b * 20));
    float m = f16_to_f32(read_u16(block + b * 20 + 2));
    const uint8_t *qs = block + b * 20 + 4;
    for (int j = 0; j < 16; j++) {
      out[b * 32 + j] = (qs[j] & 0x0F) * d + m;
      out[b * 32 + j + 16] = (qs[j] >> 4) * d + m;
    }
  }
}

/* GGML Q8_0 dequantization */
/* Block: 2 bytes f16 scale + 32 bytes int8 = 34 bytes */

static void dequant_q8_0(const uint8_t *block, float *out, int64_t n_blocks) {
  for (int64_t b = 0; b < n_blocks; b++) {
    float scale = f16_to_f32(read_u16(block + b * 34));
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
    float d = f16_to_f32(read_u16(block + b * 210 + 208));

    for (int g = 0; g < 16; g++) {
      float group_scale = d * sc[g];
      for (int j = 0; j < 16; j++) {
        int idx = g * 16 + j;
        /* Each 128-element half flattens low/high planes independently,
         * matching q_to_uint8 on [2,64] low and [2,32] high byte arrays. */
        int half = idx / 128, pos = idx % 128;
        int lo4 = (ql[half * 64 + pos % 64] >> (4 * (pos / 64))) & 0x0F;
        int hi2 = (qh[half * 32 + pos % 32] >> (2 * (pos / 32))) & 0x03;
        /* Combine: 6-bit value (0-63), subtract 32 for signed range */
        int val = (lo4 | (hi2 << 4)) - 32;
        out[b * 256 + idx] = group_scale * val;
      }
    }
  }
}

float *poly_decoded_tensor_to_f32(const PolyDecodedTensor *t) {
  if (!t || !t->data || t->numel <= 0) return NULL;
  if ((uint64_t)t->numel > SIZE_MAX / sizeof(float)) return NULL;
  int block_elems = 1;
  switch (t->dtype) {
  case POLY_DECODED_Q4_0:
  case POLY_DECODED_Q4_1:
  case POLY_DECODED_Q8_0:
    block_elems = 32;
    break;
  case POLY_DECODED_Q6_K:
    block_elems = 256;
    break;
  case POLY_DECODED_F32:
  case POLY_DECODED_F16:
  case POLY_DECODED_BF16:
  case POLY_DECODED_I32:
  case POLY_DECODED_I16:
  case POLY_DECODED_I8:
  case POLY_DECODED_U8:
    break;
  default:
    return NULL;
  }
  /* Do not publish an unwritten tail, including for callers bypassing GGUF. */
  if (t->numel % block_elems != 0) return NULL;

  /* Unified dispatch on POLY_DECODED_* codes */
  float *out = malloc((size_t)t->numel * sizeof(float));
  if (!out) return NULL;

  switch (t->dtype) {
  case POLY_DECODED_F32:
    memcpy(out, t->data, (size_t)t->numel * sizeof(float));
    return out;
  case POLY_DECODED_F16: {
    const uint8_t *src = t->data;
    for (int64_t i = 0; i < t->numel; i++)
      out[i] = f16_to_f32(read_u16(src + 2 * i));
    return out;
  }
  case POLY_DECODED_BF16: {
    const uint8_t *src = t->data;
    for (int64_t i = 0; i < t->numel; i++)
      out[i] = bf16_to_f32(read_u16(src + 2 * i));
    return out;
  }
  case POLY_DECODED_I32: {
    const uint8_t *src = t->data;
    for (int64_t i = 0; i < t->numel; i++) {
      int32_t value;
      memcpy(&value, src + 4 * i, sizeof(value));
      out[i] = (float)value;
    }
    return out;
  }
  case POLY_DECODED_I16: {
    const uint8_t *src = t->data;
    for (int64_t i = 0; i < t->numel; i++) {
      int16_t value;
      memcpy(&value, src + 2 * i, sizeof(value));
      out[i] = (float)value;
    }
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
