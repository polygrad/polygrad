/*
 * test_safetensors.c -- Tests for poly_safetensors encode/decode
 */

#include "test_harness.h"
#include "../src/safetensors.h"
#include <string.h>
#include <stdlib.h>

/* ── Round-trip: single tensor ─────────────────────────────────────── */

TEST(safetensors, round_trip_single) {
  float data[] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
  int64_t shape[] = { 2, 3 };
  PolySafetensorEntry entry = { "weight", data, shape, 2 };

  int out_len = 0;
  uint8_t *bytes = poly_safetensors_encode(&entry, 1, NULL, &out_len);
  ASSERT_NOT_NULL(bytes);
  ASSERT_TRUE(out_len > 8 + 6 * (int)sizeof(float));

  int n = 0;
  char *meta = NULL;
  PolySafetensorView *views = poly_safetensors_decode(bytes, out_len, &n, &meta);
  ASSERT_NOT_NULL(views);
  ASSERT_INT_EQ(n, 1);
  ASSERT_STR_EQ(views[0].name, "weight");
  ASSERT_INT_EQ(views[0].ndim, 2);
  ASSERT_INT_EQ(views[0].shape[0], 2);
  ASSERT_INT_EQ(views[0].shape[1], 3);
  ASSERT_INT_EQ(views[0].numel, 6);

  for (int i = 0; i < 6; i++)
    ASSERT_FLOAT_EQ(views[0].data[i], data[i], 0.0);

  ASSERT_TRUE(meta == NULL);

  free(views[0].name);
  free(views);
  free(bytes);
  PASS();
}

/* ── Round-trip: multiple tensors ──────────────────────────────────── */

TEST(safetensors, round_trip_multiple) {
  float w_data[] = { 1.0f, 2.0f, 3.0f, 4.0f };
  float b_data[] = { 0.5f, -0.5f };
  int64_t w_shape[] = { 2, 2 };
  int64_t b_shape[] = { 2 };

  PolySafetensorEntry entries[] = {
    { "layers.0.weight", w_data, w_shape, 2 },
    { "layers.0.bias", b_data, b_shape, 1 },
  };

  int out_len = 0;
  uint8_t *bytes = poly_safetensors_encode(entries, 2, NULL, &out_len);
  ASSERT_NOT_NULL(bytes);

  int n = 0;
  PolySafetensorView *views = poly_safetensors_decode(bytes, out_len, &n, NULL);
  ASSERT_NOT_NULL(views);
  ASSERT_INT_EQ(n, 2);

  /* Entries should be sorted by name (bias before weight) */
  ASSERT_STR_EQ(views[0].name, "layers.0.bias");
  ASSERT_INT_EQ(views[0].ndim, 1);
  ASSERT_INT_EQ(views[0].shape[0], 2);
  ASSERT_FLOAT_EQ(views[0].data[0], 0.5f, 0.0);
  ASSERT_FLOAT_EQ(views[0].data[1], -0.5f, 0.0);

  ASSERT_STR_EQ(views[1].name, "layers.0.weight");
  ASSERT_INT_EQ(views[1].ndim, 2);
  ASSERT_INT_EQ(views[1].numel, 4);
  for (int i = 0; i < 4; i++)
    ASSERT_FLOAT_EQ(views[1].data[i], w_data[i], 0.0);

  for (int i = 0; i < n; i++) free(views[i].name);
  free(views);
  free(bytes);
  PASS();
}

/* ── Round-trip: scalar (0-dim tensor) ─────────────────────────────── */

TEST(safetensors, round_trip_scalar) {
  float data = 42.0f;
  PolySafetensorEntry entry = { "loss", &data, NULL, 0 };

  int out_len = 0;
  uint8_t *bytes = poly_safetensors_encode(&entry, 1, NULL, &out_len);
  ASSERT_NOT_NULL(bytes);

  int n = 0;
  PolySafetensorView *views = poly_safetensors_decode(bytes, out_len, &n, NULL);
  ASSERT_NOT_NULL(views);
  ASSERT_INT_EQ(n, 1);
  ASSERT_INT_EQ(views[0].ndim, 0);
  ASSERT_INT_EQ(views[0].numel, 1);
  ASSERT_FLOAT_EQ(views[0].data[0], 42.0f, 0.0);

  free(views[0].name);
  free(views);
  free(bytes);
  PASS();
}

/* ── Metadata round-trip ──────────────────────────────────────────── */

TEST(safetensors, metadata_round_trip) {
  float data[] = { 1.0f };
  int64_t shape[] = { 1 };
  PolySafetensorEntry entry = { "x", data, shape, 1 };

  const char *meta_json = "{\"kind\":\"adamw\",\"lr\":0.001}";
  int out_len = 0;
  uint8_t *bytes = poly_safetensors_encode(&entry, 1, meta_json, &out_len);
  ASSERT_NOT_NULL(bytes);

  int n = 0;
  char *meta_out = NULL;
  PolySafetensorView *views = poly_safetensors_decode(bytes, out_len, &n, &meta_out);
  ASSERT_NOT_NULL(views);
  ASSERT_INT_EQ(n, 1);
  ASSERT_NOT_NULL(meta_out);

  /* Metadata should contain the kind and lr */
  ASSERT_TRUE(strstr(meta_out, "adamw") != NULL);
  ASSERT_TRUE(strstr(meta_out, "0.001") != NULL);

  free(meta_out);
  free(views[0].name);
  free(views);
  free(bytes);
  PASS();
}

/* ── Empty (zero tensors) ─────────────────────────────────────────── */

TEST(safetensors, encode_empty) {
  int out_len = 0;
  uint8_t *bytes = poly_safetensors_encode(NULL, 0, NULL, &out_len);
  ASSERT_NOT_NULL(bytes);
  ASSERT_TRUE(out_len >= 8);

  int n = 0;
  PolySafetensorView *views = poly_safetensors_decode(bytes, out_len, &n, NULL);
  /* Empty file: 0 tensors, should still decode */
  ASSERT_INT_EQ(n, 0);
  free(views);
  free(bytes);
  PASS();
}

/* ── Decode: truncated data ───────────────────────────────────────── */

TEST(safetensors, decode_truncated) {
  uint8_t short_buf[] = { 0, 0, 0, 0 };
  int n = 0;
  PolySafetensorView *views = poly_safetensors_decode(short_buf, 4, &n, NULL);
  ASSERT_TRUE(views == NULL);
  ASSERT_INT_EQ(n, 0);
  PASS();
}

/* ── Large tensor round-trip ──────────────────────────────────────── */

TEST(safetensors, round_trip_large) {
  int64_t numel = 1024;
  float *data = malloc(numel * sizeof(float));
  for (int64_t i = 0; i < numel; i++) data[i] = (float)i * 0.001f;

  int64_t shape[] = { 32, 32 };
  PolySafetensorEntry entry = { "big_weight", data, shape, 2 };

  int out_len = 0;
  uint8_t *bytes = poly_safetensors_encode(&entry, 1, NULL, &out_len);
  ASSERT_NOT_NULL(bytes);

  int n = 0;
  PolySafetensorView *views = poly_safetensors_decode(bytes, out_len, &n, NULL);
  ASSERT_NOT_NULL(views);
  ASSERT_INT_EQ(n, 1);
  ASSERT_INT_EQ(views[0].numel, 1024);

  for (int64_t i = 0; i < numel; i++)
    ASSERT_FLOAT_EQ(views[0].data[i], data[i], 0.0);

  free(views[0].name);
  free(views);
  free(bytes);
  free(data);
  PASS();
}

/* ── Deterministic ordering ───────────────────────────────────────── */

TEST(safetensors, deterministic_ordering) {
  float a_data[] = { 1.0f };
  float b_data[] = { 2.0f };
  float c_data[] = { 3.0f };
  int64_t shape[] = { 1 };

  /* Create entries in reverse order */
  PolySafetensorEntry entries[] = {
    { "z_param", a_data, shape, 1 },
    { "a_param", b_data, shape, 1 },
    { "m_param", c_data, shape, 1 },
  };

  int len1 = 0, len2 = 0;
  uint8_t *bytes1 = poly_safetensors_encode(entries, 3, NULL, &len1);
  uint8_t *bytes2 = poly_safetensors_encode(entries, 3, NULL, &len2);
  ASSERT_NOT_NULL(bytes1);
  ASSERT_NOT_NULL(bytes2);
  ASSERT_INT_EQ(len1, len2);
  ASSERT_TRUE(memcmp(bytes1, bytes2, len1) == 0);

  /* Verify decode order is sorted */
  int n = 0;
  PolySafetensorView *views = poly_safetensors_decode(bytes1, len1, &n, NULL);
  ASSERT_NOT_NULL(views);
  ASSERT_INT_EQ(n, 3);
  ASSERT_STR_EQ(views[0].name, "a_param");
  ASSERT_STR_EQ(views[1].name, "m_param");
  ASSERT_STR_EQ(views[2].name, "z_param");

  for (int i = 0; i < n; i++) free(views[i].name);
  free(views);
  free(bytes1);
  free(bytes2);
  PASS();
}
