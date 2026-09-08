/*
 * test_hf.c -- Tests for HuggingFace model loading infrastructure
 *
 * Covers: multi-dtype safetensors, PolyModelConfig, GPT-2 builder,
 *         HF loader, poly_gather, poly_layernorm, poly_linear.
 */

#include "test_harness.h"
#include "../src/safetensors.h"
#include "../src/models/models.h"
#include "../src/models/qwen3.h"
#include "../src/nn.h"
#include "../src/frontend.h"
#include "../src/codegen/codegen.h"
#include "../src/engine/schedule.h"
#include "../src/loaders/hf_decode.h"
#include "../src/loaders/gguf_decode.h"
#include "../src/loaders/bind.h"
#include "../src/loaders/import_desc.h"
#include <string.h>
#include <stdlib.h>
#include <math.h>

/* Safetensors multi-dtype */

TEST(hf, decoded_output_byte_count_cannot_wrap) {
  float input = 1.0f;
  PolyDecodedTensor t = {.data = &input, .numel = INT64_MAX / 2 + 1, .dtype = POLY_DECODED_F32};
  float *out = poly_decoded_tensor_to_f32(&t);
  bool rejected = out == NULL;
  free(out);
  ASSERT_TRUE(rejected);
  PASS();
}

TEST(hf, decoded_quantization_rejects_partial_blocks) {
  uint8_t bytes[210] = {0};
  int types[] = {POLY_DECODED_Q4_0, POLY_DECODED_Q4_1, POLY_DECODED_Q8_0, POLY_DECODED_Q6_K};
  bool rejected = true;
  for (int i = 0; i < 4; i++) {
    PolyDecodedTensor t = {.data = bytes, .numel = 1, .dtype = types[i]};
    float *out = poly_decoded_tensor_to_f32(&t);
    rejected = rejected && out == NULL;
    free(out);
  }
  ASSERT_TRUE(rejected);
  PASS();
}

TEST(hf, decoded_native_accepts_unaligned_bytes) {
  uint8_t bytes[5] = {0, 0, 60, 0, 0};
  PolyDecodedTensor t = {.data = bytes + 1, .numel = 1, .dtype = POLY_DECODED_F16};
  float *out = poly_decoded_tensor_to_f32(&t);
  ASSERT_NOT_NULL(out);
  ASSERT_FLOAT_EQ(out[0], 1.0f, 0.0f);
  free(out);
  PASS();
}

TEST(hf, decoded_q4_nibbles_match_pinned_halves) {
  uint8_t bytes[20] = {0, 60, 0, 64};
  for (int kind = 0; kind < 2; kind++) {
    int offset = kind ? 4 : 2;
    for (int j = 0; j < 16; j++)
      bytes[offset + j] = (uint8_t)(j | ((15 - j) << 4));
    PolyDecodedTensor t = {
        .data = bytes, .numel = 32, .dtype = kind ? POLY_DECODED_Q4_1 : POLY_DECODED_Q4_0};
    float *out = poly_decoded_tensor_to_f32(&t);
    ASSERT_NOT_NULL(out);
    bool matches = true;
    for (int j = 0; j < 16; j++) {
      matches = matches && out[j] == (float)(j + (kind ? 2 : -8));
      matches = matches && out[16 + j] == (float)(15 - j + (kind ? 2 : -8));
    }
    free(out);
    ASSERT_TRUE(matches);
    bytes[2] = 0;
    bytes[3] = 64;
  }
  PASS();
}

TEST(hf, decoded_q6_bit_planes_match_pinned_groups) {
  uint8_t bytes[210];
  for (int i = 0; i < 128; i++)
    bytes[i] = (uint8_t)(i * 13 + 7);
  for (int i = 0; i < 64; i++)
    bytes[128 + i] = (uint8_t)(i * 7 + 3);
  for (int i = 0; i < 16; i++)
    bytes[192 + i] = (uint8_t)(i + 1);
  bytes[208] = 0;
  bytes[209] = 60;
  PolyDecodedTensor t = {.data = bytes, .numel = 256, .dtype = POLY_DECODED_Q6_K};
  float *out = poly_decoded_tensor_to_f32(&t);
  ASSERT_NOT_NULL(out);
  bool matches = true;
  for (int i = 0; i < 256; i++) {
    int half = i / 128, pos = i % 128;
    int lo = (bytes[half * 64 + pos % 64] >> (4 * (pos / 64))) & 15;
    int hi = (bytes[128 + half * 32 + pos % 32] >> (2 * (pos / 32))) & 3;
    matches = matches && out[i] == (float)(((lo | (hi << 4)) - 32) * (i / 16 + 1));
  }
  free(out);
  ASSERT_TRUE(matches);
  PASS();
}

TEST(hf, decode_rejects_partial_shards_and_invalid_tables) {
  const uint8_t invalid[] = {1, 0, 0, 0, 0, 0, 0, 0, '{'};
  const uint8_t *files[] = {invalid};
  int64_t lengths[] = {sizeof(invalid)};
  PolyHfDecoded *hf = NULL;
  int rc = poly_hf_decode("{}", 2, files, lengths, 1, &hf);
  bool rejected = rc != 0 && !hf;
  poly_hf_decoded_free(hf);
  ASSERT_TRUE(rejected);
  ASSERT_TRUE(poly_hf_decode("{}", 2, files, NULL, 1, &hf) != 0);
  ASSERT_TRUE(poly_hf_decode("{}", 2, NULL, lengths, 1, &hf) != 0);
  ASSERT_TRUE(poly_hf_decode("[]", 2, NULL, NULL, 0, &hf) != 0);
  ASSERT_TRUE(poly_hf_decode("{}", 2, NULL, NULL, -1, &hf) != 0);
  ASSERT_TRUE(poly_hf_decode("{}", 2, NULL, NULL, 0, NULL) != 0);
  PASS();
}

static void gguf_test_u64(uint8_t *data, int *pos, uint64_t value, int bytes) {
  for (int i = 0; i < bytes; i++)
    data[(*pos)++] = (uint8_t)(value >> (8 * i));
}

#ifdef POLY_TESTING
extern void poly_test_hf_alloc_fail_after(int count);
TEST(hf, decode_allocation_failure_keeps_output_unpublished) {
  float value = 1;
  int64_t shape = 1;
  PolySafetensorEntry entry = {
      .name = "w", .data = &value, .shape = &shape, .ndim = 1, .dtype = POLY_ST_F32};
  int len;
  uint8_t *bytes = poly_safetensors_encode(&entry, 1, NULL, &len);
  ASSERT_NOT_NULL(bytes);
  const uint8_t *files[] = {bytes};
  int64_t lengths[] = {len};
  for (int fail = 0; fail < 2; fail++) {
    PolyHfDecoded *hf = NULL;
    poly_test_hf_alloc_fail_after(fail);
    int rc = poly_hf_decode("{}", 2, files, lengths, 1, &hf);
    poly_test_hf_alloc_fail_after(-1);
    bool rejected = rc != 0 && !hf;
    poly_hf_decoded_free(hf);
    ASSERT_TRUE(rejected);
  }
  free(bytes);
  PASS();
}
#endif

static int gguf_test_header(uint8_t *data, int tensors, int kv) {
  memcpy(data, "GGUF", 4);
  int pos = 4;
  gguf_test_u64(data, &pos, 3, 4);
  gguf_test_u64(data, &pos, tensors, 8);
  gguf_test_u64(data, &pos, kv, 8);
  return pos;
}

TEST(hf, gguf_nested_array_skip_reclaims_storage) {
  uint8_t data[128] = {0};
  int pos = gguf_test_header(data, 0, 1);
  gguf_test_u64(data, &pos, 1, 8);
  data[pos++] = 'x';
  gguf_test_u64(data, &pos, 9, 4); /* ARRAY of one ARRAY of one INT32 */
  gguf_test_u64(data, &pos, 9, 4);
  gguf_test_u64(data, &pos, 1, 8);
  gguf_test_u64(data, &pos, 5, 4);
  gguf_test_u64(data, &pos, 1, 8);
  gguf_test_u64(data, &pos, 7, 4);
  PolyGgufDecoded *g = NULL;
  int rc = poly_gguf_decode(data, (pos + 31) / 32 * 32, &g);
  poly_gguf_decoded_free(g);
  ASSERT_INT_EQ(rc, 0);
  PASS();
}

TEST(hf, gguf_rejects_truncated_metadata) {
  uint8_t data[64] = {0};
  int pos = gguf_test_header(data, 0, 1);
  gguf_test_u64(data, &pos, 1, 8);
  data[pos++] = 'x';
  gguf_test_u64(data, &pos, 9, 4);
  gguf_test_u64(data, &pos, 5, 4);
  gguf_test_u64(data, &pos, 1, 8); /* missing INT32 */
  PolyGgufDecoded *g = NULL;
  int rc = poly_gguf_decode(data, pos, &g);
  bool rejected = rc != 0 && !g;
  poly_gguf_decoded_free(g);
  ASSERT_TRUE(rejected);
  PASS();
}

TEST(hf, gguf_rejects_overflowing_string_length) {
  uint8_t data[32] = {0};
  int pos = gguf_test_header(data, 0, 1);
  gguf_test_u64(data, &pos, UINT64_MAX, 8);
  PolyGgufDecoded *g = NULL;
  int rc = poly_gguf_decode(data, pos, &g);
  poly_gguf_decoded_free(g);
  ASSERT_TRUE(rc != 0);
  PASS();
}

TEST(hf, gguf_rejects_truncated_tensor_payload) {
  uint8_t data[128] = {0};
  int pos = gguf_test_header(data, 1, 0);
  gguf_test_u64(data, &pos, 1, 8);
  data[pos++] = 'x';
  gguf_test_u64(data, &pos, 1, 4);
  gguf_test_u64(data, &pos, 2, 8); /* two float32 elements */
  gguf_test_u64(data, &pos, 0, 4);
  gguf_test_u64(data, &pos, 0, 8);
  PolyGgufDecoded *g = NULL;
  int rc = poly_gguf_decode(data, (pos + 31) / 32 * 32 + 4, &g);
  bool rejected = rc != 0 && !g;
  poly_gguf_decoded_free(g);
  ASSERT_TRUE(rejected);
  PASS();
}

TEST(hf, gguf_layout_bounds_and_truncation) {
  uint8_t data[128] = {0};
  int pos = gguf_test_header(data, 1, 0);
  gguf_test_u64(data, &pos, 1, 8);
  data[pos++] = 'x';
  int rank_pos = pos;
  gguf_test_u64(data, &pos, 1, 4);
  int dim_pos = pos;
  gguf_test_u64(data, &pos, 2, 8);
  gguf_test_u64(data, &pos, 0, 4);
  int offset_pos = pos;
  gguf_test_u64(data, &pos, 0, 8);
  int start = (pos + 31) / 32 * 32;
  float values[] = {1, 2};
  memcpy(data + start, values, sizeof(values));
  int len = start + sizeof(values);
  PolyGgufDecoded *g = NULL;
  ASSERT_INT_EQ(poly_gguf_decode(data, len, &g), 0);
  ASSERT_INT_EQ(g->n_tensors, 1);
  ASSERT_INT_EQ(g->tensors[0].numel, 2);
  ASSERT_PTR_EQ(g->tensors[0].data, data + start);
  ASSERT_INT_EQ(g->tensors[0].shape[0], 2);
  poly_gguf_decoded_free(g);
  for (int size = 0; size < len; size++) {
    g = NULL;
    ASSERT_TRUE(poly_gguf_decode(data, size, &g) != 0);
    ASSERT_TRUE(g == NULL);
  }
  int positions[] = {8, 16, rank_pos, dim_pos, offset_pos};
  for (int i = 0; i < 5; i++) {
    uint8_t malformed[128];
    memcpy(malformed, data, sizeof(data));
    int at = positions[i];
    gguf_test_u64(malformed, &at, UINT64_MAX, i == 2 ? 4 : 8);
    ASSERT_TRUE(poly_gguf_decode(malformed, len, &g) != 0);
    ASSERT_TRUE(g == NULL);
  }
  ASSERT_TRUE(poly_gguf_decode(data, len, NULL) != 0);
  PASS();
}

static int gguf_test_tensor(uint8_t *data, int type, int numel, int nbytes) {
  int pos = gguf_test_header(data, 1, 0);
  gguf_test_u64(data, &pos, 1, 8);
  data[pos++] = 'x';
  gguf_test_u64(data, &pos, 1, 4);
  gguf_test_u64(data, &pos, numel, 8);
  gguf_test_u64(data, &pos, type, 4);
  gguf_test_u64(data, &pos, 0, 8);
  return (pos + 31) / 32 * 32 + nbytes;
}

TEST(hf, gguf_native_type_ids_match_pinned_metadata) {
  int types[] = {24, 25, 26};
  int dtypes[] = {POLY_DECODED_I8, POLY_DECODED_I16, POLY_DECODED_I32};
  for (int i = 0; i < 3; i++) {
    uint8_t data[128] = {0};
    int len = gguf_test_tensor(data, types[i], 1, 1 << i);
    memset(data + len - (1 << i), 255, 1 << i);
    data[len - (1 << i)] = 254;
    PolyGgufDecoded *g = NULL;
    ASSERT_INT_EQ(poly_gguf_decode(data, len, &g), 0);
    ASSERT_INT_EQ(g->tensors[0].dtype, dtypes[i]);
    float *values = poly_decoded_tensor_to_f32(&g->tensors[0]);
    ASSERT_NOT_NULL(values);
    ASSERT_FLOAT_EQ(values[0], -2, 0);
    free(values);
    poly_gguf_decoded_free(g);
    len = gguf_test_tensor(data, 16 + i, 1, 1 << i);
    ASSERT_TRUE(poly_gguf_decode(data, len, &g) != 0);
    ASSERT_TRUE(g == NULL);
  }
  PASS();
}

TEST(hf, gguf_quantization_requires_complete_blocks) {
  uint8_t data[128] = {0};
  PolyGgufDecoded *g = NULL;
  int len = gguf_test_tensor(data, 8, 1, 34);
  int rc = poly_gguf_decode(data, len, &g);
  bool rejected = rc != 0 && !g;
  poly_gguf_decoded_free(g);
  ASSERT_TRUE(rejected);
  len = gguf_test_tensor(data, 8, 32, 34);
  ASSERT_INT_EQ(poly_gguf_decode(data, len, &g), 0);
  ASSERT_INT_EQ(g->tensors[0].dtype, POLY_DECODED_Q8_0);
  float *values = poly_decoded_tensor_to_f32(&g->tensors[0]);
  ASSERT_NOT_NULL(values);
  for (int i = 0; i < 32; i++)
    ASSERT_FLOAT_EQ(values[i], 0, 0);
  free(values);
  poly_gguf_decoded_free(g);
  PASS();
}

TEST(hf, decode_shards_publish_together_and_borrow_bytes) {
  float value = 3;
  int64_t shape = 1;
  PolySafetensorEntry entry = {
      .name = "w", .data = &value, .shape = &shape, .ndim = 1, .dtype = POLY_ST_F32};
  int len, empty_len;
  uint8_t *bytes = poly_safetensors_encode(&entry, 1, NULL, &len);
  uint8_t *empty = poly_safetensors_encode(NULL, 0, NULL, &empty_len);
  const uint8_t invalid[] = {1, 0, 0, 0, 0, 0, 0, 0, '{'};
  const uint8_t *files[] = {empty, bytes, invalid};
  int64_t lengths[] = {empty_len, len, sizeof(invalid)};
  PolyHfDecoded *hf = NULL;
  ASSERT_INT_EQ(poly_hf_decode("{}", 2, files, lengths, 2, &hf), 0);
  ASSERT_INT_EQ(hf->n_tensors, 1);
  ASSERT_PTR_EQ(hf->tensors[0].data, bytes + len - sizeof(float));
  poly_hf_decoded_free(hf);
  ASSERT_TRUE(poly_hf_decode("{}", 2, files, lengths, 3, &hf) != 0);
  ASSERT_TRUE(hf == NULL);
  free(bytes);
  free(empty);
  PASS();
}

/* Helper: create a minimal safetensors file with given dtype string */
static uint8_t *make_st_file(
    const char *name,
    const char *dtype_str,
    const void *data,
    int64_t nbytes,
    const int64_t *shape,
    int ndim,
    int64_t *out_len
) {
  /* Build JSON header manually */
  char header[512];
  char shape_str[128] = "[";
  for (int i = 0; i < ndim; i++) {
    char dim[32];
    snprintf(dim, sizeof(dim), "%s%lld", i > 0 ? "," : "", (long long)shape[i]);
    strcat(shape_str, dim);
  }
  strcat(shape_str, "]");

  snprintf(
      header, sizeof(header), "{\"%s\":{\"dtype\":\"%s\",\"shape\":%s,\"data_offsets\":[0,%lld]}}",
      name, dtype_str, shape_str, (long long)nbytes
  );

  uint64_t header_size = strlen(header);
  uint64_t total = 8 + header_size + (uint64_t)nbytes;

  uint8_t *buf = malloc((size_t)total);
  /* Write header size LE */
  for (int i = 0; i < 8; i++)
    buf[i] = (uint8_t)(header_size >> (i * 8));
  memcpy(buf + 8, header, header_size);
  memcpy(buf + 8 + header_size, data, nbytes);

  *out_len = (int64_t)total;
  return buf;
}

TEST(hf, safetensors_decode_ex_f32) {
  float data[] = {1.0f, 2.0f, 3.0f};
  int64_t shape[] = {3};
  int64_t file_len;
  uint8_t *file = make_st_file("w", "F32", data, sizeof(data), shape, 1, &file_len);

  int n = 0;
  PolySafetensorViewEx *views = poly_safetensors_decode_ex(file, file_len, &n, NULL);
  ASSERT_NOT_NULL(views);
  ASSERT_INT_EQ(n, 1);
  ASSERT_INT_EQ(views[0].dtype, POLY_ST_F32);
  ASSERT_INT_EQ(views[0].numel, 3);

  float *f32 = poly_safetensors_to_f32(&views[0]);
  ASSERT_NOT_NULL(f32);
  ASSERT_FLOAT_EQ(f32[0], 1.0f, 0.0f);
  ASSERT_FLOAT_EQ(f32[1], 2.0f, 0.0f);
  ASSERT_FLOAT_EQ(f32[2], 3.0f, 0.0f);

  free(f32);
  free(views[0].name);
  free(views);
  free(file);
  PASS();
}

TEST(hf, safetensors_decode_ex_f16) {
  /* F16 encoding: 1.0 = 0x3C00, 0.5 = 0x3800, -1.0 = 0xBC00 */
  uint16_t data[] = {0x3C00, 0x3800, 0xBC00};
  int64_t shape[] = {3};
  int64_t file_len;
  uint8_t *file = make_st_file("w", "F16", data, sizeof(data), shape, 1, &file_len);

  int n = 0;
  PolySafetensorViewEx *views = poly_safetensors_decode_ex(file, file_len, &n, NULL);
  ASSERT_NOT_NULL(views);
  ASSERT_INT_EQ(views[0].dtype, POLY_ST_F16);

  float *f32 = poly_safetensors_to_f32(&views[0]);
  ASSERT_NOT_NULL(f32);
  ASSERT_FLOAT_EQ(f32[0], 1.0f, 1e-6f);
  ASSERT_FLOAT_EQ(f32[1], 0.5f, 1e-6f);
  ASSERT_FLOAT_EQ(f32[2], -1.0f, 1e-6f);

  free(f32);
  free(views[0].name);
  free(views);
  free(file);
  PASS();
}

TEST(hf, safetensors_decode_ex_bf16) {
  /* BF16 encoding: upper 16 bits of float32 */
  /* 1.0f = 0x3F800000 -> BF16 = 0x3F80 */
  /* -2.0f = 0xC0000000 -> BF16 = 0xC000 */
  uint16_t data[] = {0x3F80, 0xC000};
  int64_t shape[] = {2};
  int64_t file_len;
  uint8_t *file = make_st_file("w", "BF16", data, sizeof(data), shape, 1, &file_len);

  int n = 0;
  PolySafetensorViewEx *views = poly_safetensors_decode_ex(file, file_len, &n, NULL);
  ASSERT_NOT_NULL(views);
  ASSERT_INT_EQ(views[0].dtype, POLY_ST_BF16);

  float *f32 = poly_safetensors_to_f32(&views[0]);
  ASSERT_NOT_NULL(f32);
  ASSERT_FLOAT_EQ(f32[0], 1.0f, 1e-6f);
  ASSERT_FLOAT_EQ(f32[1], -2.0f, 1e-6f);

  free(f32);
  free(views[0].name);
  free(views);
  free(file);
  PASS();
}

TEST(hf, safetensors_decode_ex_i32) {
  int32_t data[] = {42, -7, 0, 100};
  int64_t shape[] = {4};
  int64_t file_len;
  uint8_t *file = make_st_file("idx", "I32", data, sizeof(data), shape, 1, &file_len);

  int n = 0;
  PolySafetensorViewEx *views = poly_safetensors_decode_ex(file, file_len, &n, NULL);
  ASSERT_NOT_NULL(views);
  ASSERT_INT_EQ(views[0].dtype, POLY_ST_I32);

  float *f32 = poly_safetensors_to_f32(&views[0]);
  ASSERT_NOT_NULL(f32);
  ASSERT_FLOAT_EQ(f32[0], 42.0f, 0.0f);
  ASSERT_FLOAT_EQ(f32[1], -7.0f, 0.0f);
  ASSERT_FLOAT_EQ(f32[2], 0.0f, 0.0f);
  ASSERT_FLOAT_EQ(f32[3], 100.0f, 0.0f);

  free(f32);
  free(views[0].name);
  free(views);
  free(file);
  PASS();
}

TEST(hf, safetensors_header_padding) {
  /* Test that trailing spaces in JSON header are tolerated */
  float data[] = {1.0f};
  /* Manually build with trailing spaces */
  const char *header = "{\"w\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}   ";
  uint64_t header_size = strlen(header);
  uint64_t total = 8 + header_size + 4;
  uint8_t *buf = malloc((size_t)total);
  for (int i = 0; i < 8; i++)
    buf[i] = (uint8_t)(header_size >> (i * 8));
  memcpy(buf + 8, header, header_size);
  memcpy(buf + 8 + header_size, data, 4);

  int n = 0;
  PolySafetensorViewEx *views = poly_safetensors_decode_ex(buf, (int64_t)total, &n, NULL);
  ASSERT_NOT_NULL(views);
  ASSERT_INT_EQ(n, 1);

  float *f32 = poly_safetensors_to_f32(&views[0]);
  ASSERT_FLOAT_EQ(f32[0], 1.0f, 0.0f);

  free(f32);
  free(views[0].name);
  free(views);
  free(buf);
  PASS();
}

/* PolyModelConfig */

TEST(hf, config_parse) {
  const char *json = "{\"model_type\":\"gpt2\",\"vocab_size\":50257,"
                     "\"n_embd\":768,\"n_head\":12,\"n_layer\":12,"
                     "\"n_positions\":1024,\"layer_norm_epsilon\":1e-5}";
  PolyModelConfig *cfg = poly_model_config_from_json(json, (int)strlen(json));
  ASSERT_NOT_NULL(cfg);

  ASSERT_INT_EQ(poly_model_config_get_int(cfg, "vocab_size", 0), 50257);
  ASSERT_INT_EQ(poly_model_config_get_int(cfg, "n_embd", 0), 768);
  ASSERT_INT_EQ(poly_model_config_get_int(cfg, "n_head", 0), 12);
  ASSERT_INT_EQ(poly_model_config_get_int(cfg, "n_layer", 0), 12);
  ASSERT_INT_EQ(poly_model_config_get_int(cfg, "n_positions", 0), 1024);

  const char *mt = poly_model_config_get_string(cfg, "model_type", "");
  ASSERT_STR_EQ(mt, "gpt2");

  /* Default values for missing keys */
  ASSERT_INT_EQ(poly_model_config_get_int(cfg, "missing", 42), 42);

  poly_model_config_free(cfg);
  PASS();
}

/* GPT-2 builder */

TEST(hf, gpt2_build_tiny) {
  GPT2Config cfg = {
      .vocab_size = 32,
      .n_embd = 16,
      .n_head = 2,
      .n_layer = 1,
      .max_seq_len = 8,
      .batch_size = 1,
      .norm_eps = 1e-5f};

  PolyModel *inst = poly_gpt2(&cfg, POLY_DEVICE_AUTO);
  ASSERT_NOT_NULL(inst);

  /* Check param count: wte + wpe + 1 layer (12 params) + ln_f (2) = 16 */
  int n_params = poly_model_param_count(inst);
  ASSERT_INT_EQ(n_params, 16);

  /* Check wte.weight shape */
  int64_t shape[8];
  for (int i = 0; i < n_params; i++) {
    const char *name = poly_model_param_name(inst, i);
    if (strcmp(name, "wte.weight") == 0) {
      int ndim = poly_model_param_shape(inst, i, shape, 8);
      ASSERT_INT_EQ(ndim, 2);
      ASSERT_INT_EQ(shape[0], 32); /* vocab_size */
      ASSERT_INT_EQ(shape[1], 16); /* n_embd */
    }
  }

  /* Check we have the expected buffer names */
  int n_bufs = poly_model_buf_count(inst);
  ASSERT_TRUE(n_bufs >= 16 + 4); /* params + x + output + positions + arange */

  int found_x = 0, found_output = 0;
  for (int i = 0; i < n_bufs; i++) {
    const char *name = poly_model_buf_name(inst, i);
    if (strcmp(name, "x") == 0) found_x = 1;
    if (strcmp(name, "output") == 0) found_output = 1;
  }
  ASSERT_TRUE(found_x);
  ASSERT_TRUE(found_output);

  ASSERT_INT_EQ(poly_ctx_named_count(poly_model_ctx(inst)), 0);

  poly_model_free(inst);
  PASS();
}

TEST(hf, gpt2_build_multi_layer) {
  GPT2Config cfg = {
      .vocab_size = 64,
      .n_embd = 32,
      .n_head = 4,
      .n_layer = 3,
      .max_seq_len = 16,
      .batch_size = 2,
      .norm_eps = 1e-5f};

  PolyModel *inst = poly_gpt2(&cfg, POLY_DEVICE_AUTO);
  ASSERT_NOT_NULL(inst);

  /* 2 (wte+wpe) + 3*12 (layers) + 2 (ln_f) = 40 params */
  ASSERT_INT_EQ(poly_model_param_count(inst), 40);

  /* Verify a deep layer param exists */
  int found = 0;
  int n_params = poly_model_param_count(inst);
  for (int i = 0; i < n_params; i++) {
    if (strcmp(poly_model_param_name(inst, i), "h.2.mlp.c_proj.weight") == 0) found = 1;
  }
  ASSERT_TRUE(found);

  poly_model_free(inst);
  PASS();
}

/* HF loader */

TEST(hf, qwen3_build_tiny_staged) {
  Qwen3Config cfg = poly_qwen3_config_default();
  cfg.vocab_size = 32;
  cfg.dim = 16;
  cfg.n_heads = 2;
  cfg.n_kv_heads = 1;
  cfg.n_layers = 1;
  cfg.hidden_dim = 32;
  cfg.head_dim = 8;
  cfg.max_seq_len = 4;
  cfg.batch_size = 1;
  cfg.qk_norm = 0;

  PolyModel *inst = poly_qwen3(&cfg, POLY_DEVICE_AUTO);
  ASSERT_NOT_NULL(inst);
  ASSERT_INT_EQ(poly_ctx_named_count(poly_model_ctx(inst)), 0);

  ASSERT_INT_EQ(poly_model_param_count(inst), 11);
  ASSERT_STR_EQ(poly_model_param_name(inst, 0), "token_embd.weight");
  ASSERT_STR_EQ(poly_model_param_name(inst, 1), "blk.0.attn_norm.weight");
  ASSERT_STR_EQ(poly_model_param_name(inst, 2), "blk.0.attn_q.weight");

  int64_t numel = 0;
  int x_idx = -1;
  for (int i = 0; i < poly_model_buf_count(inst); i++)
    if (strcmp(poly_model_buf_name(inst, i), "x") == 0) x_idx = i;
  ASSERT_TRUE(x_idx >= 0);
  ASSERT_INT_EQ(poly_model_buf_dtype_id(inst, x_idx), poly_dtype_id_by_name("int32"));
  ASSERT_NOT_NULL(poly_model_buf_data_raw(inst, x_idx, &numel));
  ASSERT_INT_EQ(numel, 4);
  ASSERT_NOT_NULL(poly_model_buf_data_named(inst, "rope_cos", &numel));
  ASSERT_INT_EQ(numel, 16);
  ASSERT_NOT_NULL(poly_model_buf_data_named(inst, "rope_sin", &numel));
  ASSERT_INT_EQ(numel, 16);
  ASSERT_NOT_NULL(poly_model_buf_data_named(inst, "output", &numel));
  ASSERT_INT_EQ(numel, 4 * 32);

  poly_model_free(inst);
  PASS();
}

extern void poly_test_bind_alloc_fail_after(int count);

static bool bind_allocation_rejected(int after) {
  GPT2Config cfg = poly_gpt2_config_default();
  cfg.n_embd = 8;
  cfg.n_head = 2;
  cfg.n_layer = 1;
  cfg.vocab_size = 4;
  cfg.max_seq_len = 2;
  cfg.batch_size = 1;
  PolyModel *model = poly_gpt2(&cfg, POLY_DEVICE_CPU);
  if (!model) return false;
  poly_test_bind_alloc_fail_after(after);
  PolyBindIndex *idx = poly_bind_index_create(model);
  poly_test_bind_alloc_fail_after(-1);
  bool rejected = idx == NULL;
  poly_bind_index_destroy(idx);
  idx = poly_bind_index_create(model);
  bool retried = idx != NULL;
  poly_bind_index_destroy(idx);
  poly_model_free(model);
  return rejected && retried;
}

TEST(hf, bind_index_candidate_allocation_failure) {
  ASSERT_TRUE(bind_allocation_rejected(0));
  PASS();
}

TEST(hf, bind_index_table_allocation_failure) {
  ASSERT_TRUE(bind_allocation_rejected(1));
  PASS();
}

TEST(hf, gguf_families_reject_failed_weights_and_bindings) {
  const char *suffix[] = {"embedding_length",        "attention.head_count", "block_count",
                          "attention.head_count_kv", "feed_forward_length",  "context_length"};
  int config[] = {8, 2, 1, 2, 16, 2};
  for (int family = 0; family < 2; family++) {
    char keys[6][64];
    PolyGgufKV kv[6] = {0};
    for (int i = 0; i < 6; i++) {
      snprintf(keys[i], sizeof(keys[i]), "%s.%s", family ? "qwen3" : "gpt2", suffix[i]);
      kv[i] = (PolyGgufKV){.key = keys[i], .type = 4, .val.u64 = (uint64_t)config[i]};
    }
    float data[16] = {0};
    for (int mode = 0; mode < 4; mode++) {
      PolyDecodedTensor t = {
          .name = "token_embd.weight",
          .data = data,
          .shape = {2, mode == 1 ? 4 : 8},
          .ndim = 2,
          .numel = mode == 1 ? 8 : 16,
          .dtype = mode == 0 ? POLY_DECODED_Q4_K : POLY_DECODED_F32};
      PolyGgufDecoded g = {
          .kv = kv, .n_kv = 6, .arch = family ? "qwen3" : "gpt2", .tensors = &t, .n_tensors = 1};
      if (mode == 2) poly_test_bind_alloc_fail_after(1);
      const PolyImportDesc *desc = poly_import_desc_find(g.arch);
      PolyGenericImportOpts opts = {.max_batch = 1, .max_seq_len = 2, .device = POLY_DEVICE_CPU};
      PolyModel *model = desc->from_gguf_decoded(&g, &opts);
      poly_test_bind_alloc_fail_after(-1);
      bool correct = (model != NULL) == (mode == 3);
      poly_model_free(model);
      ASSERT_TRUE(correct);
    }
  }
  PASS();
}

TEST(hf, model_import_rejects_failed_weight_conversion_or_copy) {
  const char *config = "{\"model_type\":\"gpt2\",\"vocab_size\":32,\"n_embd\":16,"
                       "\"n_head\":2,\"n_layer\":1,\"n_positions\":8}";
  double payload[512] = {0};
  bool rejected = true;
  for (int kind = 0; kind < 2; kind++) {
    int64_t shape[] = {kind ? 32 : 1, kind ? 16 : 1}, len = 0;
    uint8_t *file = make_st_file(
        "transformer.wte.weight", kind ? "F64" : "F32", payload, kind ? sizeof(payload) : 4, shape,
        2, &len
    );
    const uint8_t *files[] = {file};
    PolyModel *model =
        poly_hf_load(config, (int)strlen(config), files, &len, 1, 1, 2, POLY_DEVICE_CPU);
    rejected = rejected && model == NULL;
    poly_model_free(model);
    free(file);
  }
  ASSERT_TRUE(rejected);
  PASS();
}

TEST(hf, hf_load_tiny_gpt2) {
  const char *config = "{\"model_type\":\"gpt2\",\"vocab_size\":32,"
                       "\"n_embd\":16,\"n_head\":2,\"n_layer\":1,"
                       "\"n_positions\":8,\"layer_norm_epsilon\":1e-5}";

  /* Create a safetensors file with a few test weights */
  float wte_data[32 * 16];
  for (int i = 0; i < 32 * 16; i++)
    wte_data[i] = (float)i * 0.001f;

  int64_t wte_shape[] = {32, 16};
  int64_t file_len;
  uint8_t *file = make_st_file(
      "transformer.wte.weight", "F32", wte_data, sizeof(wte_data), wte_shape, 2, &file_len
  );

  const uint8_t *files[] = {file};
  int64_t lens[] = {file_len};

  PolyModel *inst =
      poly_hf_load(config, (int)strlen(config), files, lens, 1, 1, 8, POLY_DEVICE_AUTO);
  ASSERT_NOT_NULL(inst);

  /* Verify wte.weight was loaded */
  int n_params = poly_model_param_count(inst);
  for (int i = 0; i < n_params; i++) {
    if (strcmp(poly_model_param_name(inst, i), "wte.weight") == 0) {
      int64_t numel;
      float *data = poly_model_param_data(inst, i, &numel);
      ASSERT_INT_EQ(numel, 32 * 16);
      ASSERT_FLOAT_EQ(data[0], 0.0f, 1e-6f);
      ASSERT_FLOAT_EQ(data[1], 0.001f, 1e-6f);
      break;
    }
  }

  poly_model_free(inst);
  free(file);
  PASS();
}

TEST(hf, hf_load_ignores_attn_bias) {
  const char *config = "{\"model_type\":\"gpt2\",\"vocab_size\":32,"
                       "\"n_embd\":16,\"n_head\":2,\"n_layer\":1,"
                       "\"n_positions\":8,\"layer_norm_epsilon\":1e-5}";

  /* Create a safetensors file with attn.bias (should be ignored) */
  float bias_data[8 * 8];
  memset(bias_data, 0, sizeof(bias_data));
  int64_t bias_shape[] = {1, 1, 8, 8};
  int64_t file_len;
  uint8_t *file = make_st_file(
      "transformer.h.0.attn.bias", "F32", bias_data, sizeof(bias_data), bias_shape, 4, &file_len
  );

  const uint8_t *files[] = {file};
  int64_t lens[] = {file_len};

  /* Should not crash */
  PolyModel *inst =
      poly_hf_load(config, (int)strlen(config), files, lens, 1, 1, 8, POLY_DEVICE_AUTO);
  ASSERT_NOT_NULL(inst);

  poly_model_free(inst);
  free(file);
  PASS();
}

/* Frontend ops */

TEST(hf, poly_gather_basic) {
  PolyCtx *ctx = poly_ctx_new();

  /* table: (4, 3) weight matrix -- reshape buffer to give it a shape */
  int64_t table_shape[] = {4, 3};
  PolyUOp *table = poly_reshape(ctx, poly_buffer_f32(ctx, 12), table_shape, 2);

  /* indices: (2,) -- reshape buffer to give it a shape */
  int64_t idx_shape[] = {2};
  PolyUOp *indices = poly_reshape(ctx, poly_test_buffer(ctx, POLY_INT32, 2), idx_shape, 1);

  PolyUOp *result = poly_gather(ctx, table, indices);
  ASSERT_NOT_NULL(result);
  PolyShape s = poly_uop_max_shape(ctx, result);
  ASSERT_INT_EQ(s.ndim, 2);
  ASSERT_INT_EQ(s.dims[0], 2); /* num indices */
  ASSERT_INT_EQ(s.dims[1], 3); /* embedding dim */
  if (s.dims) free(s.dims);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(hf, poly_gather_2d_indices) {
  PolyCtx *ctx = poly_ctx_new();

  /* table: (10, 4) -- reshape buffer to give it a shape */
  int64_t table_shape[] = {10, 4};
  PolyUOp *table = poly_reshape(ctx, poly_buffer_f32(ctx, 40), table_shape, 2);

  /* indices: (2, 3) -- batch of indices, reshape buffer to give it a shape */
  int64_t idx_shape[] = {2, 3};
  PolyUOp *indices = poly_reshape(ctx, poly_test_buffer(ctx, POLY_INT32, 6), idx_shape, 2);

  PolyUOp *result = poly_gather(ctx, table, indices);
  ASSERT_NOT_NULL(result);
  PolyShape s = poly_uop_max_shape(ctx, result);
  ASSERT_INT_EQ(s.ndim, 3);
  ASSERT_INT_EQ(s.dims[0], 2); /* batch */
  ASSERT_INT_EQ(s.dims[1], 3); /* seq_len */
  ASSERT_INT_EQ(s.dims[2], 4); /* embed_dim */
  if (s.dims) free(s.dims);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(hf, poly_layernorm_shape) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *x = poly_reshape(ctx, poly_buffer_f32(ctx, 24), (int64_t[]){2, 3, 4}, 3);

  PolyUOp *result = poly_layernorm_apply(ctx, x, NULL, NULL, -1, 1e-5);
  ASSERT_NOT_NULL(result);
  PolyShape s = poly_uop_max_shape(ctx, result);
  ASSERT_INT_EQ(s.ndim, 3);
  ASSERT_INT_EQ(s.dims[0], 2);
  ASSERT_INT_EQ(s.dims[1], 3);
  ASSERT_INT_EQ(s.dims[2], 4);
  if (s.dims) free(s.dims);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(hf, poly_linear_shape) {
  PolyCtx *ctx = poly_ctx_new();

  /* x: (2, 3, 4), weight: (8, 4), bias: (8,) */
  PolyUOp *x = poly_reshape(ctx, poly_buffer_f32(ctx, 24), (int64_t[]){2, 3, 4}, 3);
  PolyUOp *w = poly_reshape(ctx, poly_buffer_f32(ctx, 32), (int64_t[]){8, 4}, 2);
  PolyUOp *b = poly_reshape(ctx, poly_buffer_f32(ctx, 8), (int64_t[]){8}, 1);

  PolyUOp *result = poly_linear_apply(ctx, x, w, b);
  ASSERT_NOT_NULL(result);

  PolyShape s = poly_uop_max_shape(ctx, result);
  ASSERT_INT_EQ(s.ndim, 3);
  ASSERT_INT_EQ(s.dims[0], 2);
  ASSERT_INT_EQ(s.dims[1], 3);
  ASSERT_INT_EQ(s.dims[2], 8);
  if (s.dims) free(s.dims);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(hf, poly_linear_no_bias) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *x = poly_reshape(ctx, poly_buffer_f32(ctx, 32), (int64_t[]){4, 8}, 2);
  PolyUOp *w = poly_reshape(ctx, poly_buffer_f32(ctx, 128), (int64_t[]){16, 8}, 2);

  PolyUOp *result = poly_linear_apply(ctx, x, w, NULL);
  ASSERT_NOT_NULL(result);

  PolyShape s = poly_uop_max_shape(ctx, result);
  ASSERT_INT_EQ(s.ndim, 2);
  ASSERT_INT_EQ(s.dims[0], 4);
  ASSERT_INT_EQ(s.dims[1], 16);
  if (s.dims) free(s.dims);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(hf, poly_causal_mask_shape) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *mask = poly_causal_mask(ctx, 5);
  ASSERT_NOT_NULL(mask);
  PolyShape s = poly_uop_max_shape(ctx, mask);
  ASSERT_INT_EQ(s.ndim, 2);
  ASSERT_INT_EQ(s.dims[0], 5);
  ASSERT_INT_EQ(s.dims[1], 5);
  if (s.dims) free(s.dims);

  poly_ctx_destroy(ctx);
  PASS();
}

/* GPT-2 forward pass e2e */

TEST(hf, gpt2_forward_e2e) {
  GPT2Config cfg = {
      .vocab_size = 32,
      .n_embd = 16,
      .n_head = 2,
      .n_layer = 1,
      .max_seq_len = 8,
      .batch_size = 1,
      .norm_eps = 1e-5f};

  PolyModel *inst = poly_gpt2(&cfg, POLY_DEVICE_AUTO);
  ASSERT_NOT_NULL(inst);

  /* Initialize weights with small random values */
  int np = poly_model_param_count(inst);
  for (int i = 0; i < np; i++) {
    int64_t numel;
    float *data = poly_model_param_data(inst, i, &numel);
    const char *name = poly_model_param_name(inst, i);
    /* LayerNorm weights init to 1, biases to 0 */
    if (strstr(name, "ln_") && strstr(name, "weight")) {
      for (int64_t j = 0; j < numel; j++)
        data[j] = 1.0f;
    } else if (strstr(name, "bias")) {
      for (int64_t j = 0; j < numel; j++)
        data[j] = 0.0f;
    } else {
      for (int64_t j = 0; j < numel; j++)
        data[j] = 0.02f * ((float)(j % 100) / 100.0f - 0.5f);
    }
  }

  /* Pinned GPT-style embedding inputs are integer token/position indices.
   * Use the typed byte API instead of the float-only convenience view. */
  int32_t token_data[8], position_data[8];
  for (int j = 0; j < 8; j++) {
    token_data[j] = j % 4;
    position_data[j] = j;
  }
  PolyIOBinding forward_io[] = {
      POLY_IO_BINDING_ARRAY("x", token_data, POLY_INT32),
      POLY_IO_BINDING_ARRAY("positions", position_data, POLY_INT32),
  };

  /* Run forward pass */
  int ret = poly_model_forward(inst, forward_io, 2);
  ASSERT_INT_EQ(ret, 0);

  /* Check output: (1, 8, 32) logits */
  int nb = poly_model_buf_count(inst);
  int out_idx = -1;
  for (int i = 0; i < nb; i++) {
    if (strcmp(poly_model_buf_name(inst, i), "output") == 0) {
      out_idx = i;
      break;
    }
  }
  ASSERT_TRUE(out_idx >= 0);

  int64_t numel;
  float *out = poly_model_buf_data(inst, out_idx, &numel);
  ASSERT_INT_EQ(numel, 1 * 8 * 32);

  /* Verify output is finite and not all zero */
  int all_zero = 1;
  for (int64_t i = 0; i < numel; i++) {
    ASSERT_TRUE(isfinite(out[i]));
    if (fabsf(out[i]) > 1e-10f) all_zero = 0;
  }
  ASSERT_FALSE(all_zero);

  float *baseline = malloc((size_t)numel * sizeof(*baseline));
  ASSERT_NOT_NULL(baseline);
  memcpy(baseline, out, (size_t)numel * sizeof(*baseline));

  int ir_len = 0, weights_len = 0;
  uint8_t *ir = poly_model_export_ir(inst, &ir_len);
  uint8_t *weights = poly_model_export_weights_ex(inst, &weights_len, POLY_EXPORT_WEIGHTS_PARAMS);
  ASSERT_NOT_NULL(ir);
  ASSERT_NOT_NULL(weights);

  PolyDevice place_devices[3] = {POLY_DEVICE_CPU, POLY_DEVICE_INTERP, POLY_DEVICE_AUTO};
  int n_place_devices = 2;
#ifdef POLY_HAS_CUDA
  if (poly_cuda_available()) place_devices[n_place_devices++] = POLY_DEVICE_CUDA;
#endif
  for (int d = 0; d < n_place_devices; d++) {
    PolyModel *placed = poly_model_from_ir(ir, ir_len, weights, weights_len);
    ASSERT_NOT_NULL(placed);
    ASSERT_INT_EQ(poly_model_set_device(placed, place_devices[d]), 0);
    ASSERT_INT_EQ(poly_model_forward(placed, forward_io, 2), 0);
    int64_t placed_numel = 0;
    float *placed_out = poly_model_buf_data_named(placed, "output", &placed_numel);
    ASSERT_NOT_NULL(placed_out);
    ASSERT_INT_EQ(placed_numel, numel);
    for (int64_t i = 0; i < numel; i++)
      ASSERT_FLOAT_EQ(placed_out[i], baseline[i], 2e-5f);
    poly_model_free(placed);
  }

  free(weights);
  free(ir);
  free(baseline);

  poly_model_free(inst);
  PASS();
}

/* GPT-2 training: loss decreases */

TEST(hf, gpt2_training_loss_decreases) {
  GPT2Config cfg = {
      .vocab_size = 32,
      .n_embd = 16,
      .n_head = 2,
      .n_layer = 1,
      .max_seq_len = 8,
      .batch_size = 1,
      .norm_eps = 1e-5f};

  PolyModel *inst = poly_gpt2(&cfg, POLY_DEVICE_AUTO);
  ASSERT_NOT_NULL(inst);

  /* Initialize weights with small random values */
  int np = poly_model_param_count(inst);
  for (int i = 0; i < np; i++) {
    int64_t numel;
    float *data = poly_model_param_data(inst, i, &numel);
    const char *name = poly_model_param_name(inst, i);
    if (strstr(name, "ln_") && strstr(name, "weight")) {
      for (int64_t j = 0; j < numel; j++)
        data[j] = 1.0f;
    } else if (strstr(name, "bias")) {
      for (int64_t j = 0; j < numel; j++)
        data[j] = 0.0f;
    } else {
      for (int64_t j = 0; j < numel; j++)
        data[j] = 0.02f * ((float)((j * 7 + 13) % 100) / 100.0f - 0.5f);
    }
  }

  int32_t token_data[8], position_data[8];
  for (int j = 0; j < 8; j++) {
    token_data[j] = j % 4;
    position_data[j] = j;
  }
  PolyIOBinding train_io[] = {
      POLY_IO_BINDING_ARRAY("x", token_data, POLY_INT32),
      POLY_IO_BINDING_ARRAY("positions", position_data, POLY_INT32),
  };

  /* Configure Adam optimizer */
  int ret = poly_model_set_optimizer(inst, POLY_OPTIM_ADAM, 0.001f, 0.9f, 0.999f, 1e-8f, 0.0f);
  ASSERT_INT_EQ(ret, 0);

  /* Train for 5 steps */
  float losses[5];
  for (int step = 0; step < 5; step++) {
    ret = poly_model_train_step(inst, NULL, train_io, 2, &losses[step]);
    ASSERT_INT_EQ(ret, 0);
    ASSERT_TRUE(isfinite(losses[step]));
  }

  /* Loss should decrease */
  ASSERT_TRUE(losses[4] < losses[0]);

  /* All losses should be finite and positive (sum of squares) */
  for (int i = 0; i < 5; i++) {
    ASSERT_TRUE(losses[i] >= 0.0f);
    ASSERT_TRUE(isfinite(losses[i]));
  }

  poly_model_free(inst);
  PASS();
}

/* Unsupported model type */

TEST(hf, hf_load_unsupported_type) {
  const char *config = "{\"model_type\":\"llama\",\"vocab_size\":100}";
  PolyModel *inst =
      poly_hf_load(config, (int)strlen(config), NULL, NULL, 0, 1, 64, POLY_DEVICE_AUTO);
  ASSERT_TRUE(inst == NULL);
  PASS();
}
