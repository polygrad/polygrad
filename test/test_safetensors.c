/*
 * test_safetensors.c -- Tests for poly_safetensors encode/decode
 */

#include "test_harness.h"
#include "../src/safetensors.h"
#include <string.h>
#include <stdlib.h>

/* Round-trip: single tensor */

TEST(safetensors, round_trip_single) {
  float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  int64_t shape[] = {2, 3};
  PolySafetensorEntry entry = {
      .name = "weight",
      .data = data,
      .shape = shape,
      .ndim = 2,
      .dtype = POLY_ST_F32,
  };

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

/* Round-trip: multiple tensors */

TEST(safetensors, round_trip_multiple) {
  float w_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
  float b_data[] = {0.5f, -0.5f};
  int64_t w_shape[] = {2, 2};
  int64_t b_shape[] = {2};

  PolySafetensorEntry entries[] = {
      {.name = "layers.0.weight",
       .data = w_data,
       .shape = w_shape,
       .ndim = 2,
       .dtype = POLY_ST_F32},
      {.name = "layers.0.bias", .data = b_data, .shape = b_shape, .ndim = 1, .dtype = POLY_ST_F32},
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

  for (int i = 0; i < n; i++)
    free(views[i].name);
  free(views);
  free(bytes);
  PASS();
}

/* Round-trip: scalar (0-dim tensor) */

TEST(safetensors, round_trip_scalar) {
  float data = 42.0f;
  PolySafetensorEntry entry = {
      .name = "loss",
      .data = &data,
      .shape = NULL,
      .ndim = 0,
      .dtype = POLY_ST_F32,
  };

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

/* Metadata round-trip */

TEST(safetensors, metadata_round_trip) {
  float data[] = {1.0f};
  int64_t shape[] = {1};
  PolySafetensorEntry entry = {
      .name = "x",
      .data = data,
      .shape = shape,
      .ndim = 1,
      .dtype = POLY_ST_F32,
  };

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

/* Empty (zero tensors) */

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

/* Decode: truncated data */

TEST(safetensors, decode_truncated) {
  uint8_t short_buf[] = {0, 0, 0, 0};
  int n = 0;
  PolySafetensorView *views = poly_safetensors_decode(short_buf, 4, &n, NULL);
  ASSERT_TRUE(views == NULL);
  ASSERT_INT_EQ(n, 0);
  PASS();
}

TEST(safetensors, rejects_overflowing_header_and_missing_count) {
  uint8_t huge_header[8];
  memset(huge_header, 0xff, sizeof(huge_header));
  int n = 7;
  ASSERT_EQ(poly_safetensors_decode(huge_header, sizeof(huge_header), &n, NULL), NULL);
  ASSERT_INT_EQ(n, 0);
  ASSERT_EQ(poly_safetensors_decode_ex(huge_header, sizeof(huge_header), &n, NULL), NULL);
  ASSERT_INT_EQ(n, 0);
  ASSERT_EQ(poly_safetensors_decode(huge_header, sizeof(huge_header), NULL, NULL), NULL);
  ASSERT_EQ(poly_safetensors_decode_ex(huge_header, sizeof(huge_header), NULL, NULL), NULL);
  PASS();
}

/* Large tensor round-trip */

TEST(safetensors, round_trip_large) {
  int64_t numel = 1024;
  float *data = malloc(numel * sizeof(float));
  for (int64_t i = 0; i < numel; i++)
    data[i] = (float)i * 0.001f;

  int64_t shape[] = {32, 32};
  PolySafetensorEntry entry = {
      .name = "big_weight",
      .data = data,
      .shape = shape,
      .ndim = 2,
      .dtype = POLY_ST_F32,
  };

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

/* Deterministic ordering */

TEST(safetensors, deterministic_ordering) {
  float a_data[] = {1.0f};
  float b_data[] = {2.0f};
  float c_data[] = {3.0f};
  int64_t shape[] = {1};

  /* Create entries in reverse order */
  PolySafetensorEntry entries[] = {
      {.name = "z_param", .data = a_data, .shape = shape, .ndim = 1, .dtype = POLY_ST_F32},
      {.name = "a_param", .data = b_data, .shape = shape, .ndim = 1, .dtype = POLY_ST_F32},
      {.name = "m_param", .data = c_data, .shape = shape, .ndim = 1, .dtype = POLY_ST_F32},
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

  for (int i = 0; i < n; i++)
    free(views[i].name);
  free(views);
  free(bytes1);
  free(bytes2);
  PASS();
}

TEST(safetensors, mixed_dtypes_preserve_exact_storage_bytes) {
  uint16_t half_data[] = {0x3e00, 0xc000}; /* 1.5, -2.0 */
  int32_t int_data[] = {INT32_MIN, 7, INT32_MAX};
  double double_data[] = {1.0 / 3.0};
  int64_t half_shape[] = {2};
  int64_t int_shape[] = {3};
  int64_t double_shape[] = {1};
  PolySafetensorEntry entries[] = {
      {.name = "half", .data = half_data, .shape = half_shape, .ndim = 1, .dtype = POLY_ST_F16},
      {.name = "integer", .data = int_data, .shape = int_shape, .ndim = 1, .dtype = POLY_ST_I32},
      {.name = "double",
       .data = double_data,
       .shape = double_shape,
       .ndim = 1,
       .dtype = POLY_ST_F64},
  };

  int out_len = 0;
  uint8_t *bytes = poly_safetensors_encode(entries, 3, NULL, &out_len);
  ASSERT_NOT_NULL(bytes);
  ASSERT_TRUE(out_len > 0);

  int n = 0;
  PolySafetensorViewEx *views = poly_safetensors_decode_ex(bytes, out_len, &n, NULL);
  ASSERT_NOT_NULL(views);
  ASSERT_INT_EQ(n, 3);
  ASSERT_STR_EQ(views[0].name, "double");
  ASSERT_INT_EQ(views[0].dtype, POLY_ST_F64);
  ASSERT_INT_EQ(views[0].numel, 1);
  ASSERT_TRUE(memcmp(views[0].raw_data, double_data, sizeof(double_data)) == 0);
  ASSERT_STR_EQ(views[1].name, "half");
  ASSERT_INT_EQ(views[1].dtype, POLY_ST_F16);
  ASSERT_INT_EQ(views[1].numel, 2);
  ASSERT_TRUE(memcmp(views[1].raw_data, half_data, sizeof(half_data)) == 0);
  ASSERT_STR_EQ(views[2].name, "integer");
  ASSERT_INT_EQ(views[2].dtype, POLY_ST_I32);
  ASSERT_INT_EQ(views[2].numel, 3);
  ASSERT_TRUE(memcmp(views[2].raw_data, int_data, sizeof(int_data)) == 0);

  for (int i = 0; i < n; i++)
    free(views[i].name);
  free(views);
  free(bytes);
  PASS();
}

TEST(safetensors, rejects_invalid_fixed_shape_rows) {
  float value = 1.0f;
  int64_t rank_nine[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  PolySafetensorEntry too_wide = {
      .name = "x",
      .data = &value,
      .shape = rank_nine,
      .ndim = 9,
      .dtype = POLY_ST_F32,
  };
  int out_len = 7;
  ASSERT_EQ(poly_safetensors_encode(&too_wide, 1, NULL, &out_len), NULL);
  ASSERT_INT_EQ(out_len, 0);

  int64_t negative_shape[] = {-1};
  PolySafetensorEntry negative = {
      .name = "x",
      .data = &value,
      .shape = negative_shape,
      .ndim = 1,
      .dtype = POLY_ST_F32,
  };
  out_len = 7;
  ASSERT_EQ(poly_safetensors_encode(&negative, 1, NULL, &out_len), NULL);
  ASSERT_INT_EQ(out_len, 0);

  int64_t overflow_shape[] = {INT64_MAX, 2};
  PolySafetensorEntry overflow = {
      .name = "x",
      .data = &value,
      .shape = overflow_shape,
      .ndim = 2,
      .dtype = POLY_ST_F32,
  };
  out_len = 7;
  ASSERT_EQ(poly_safetensors_encode(&overflow, 1, NULL, &out_len), NULL);
  ASSERT_INT_EQ(out_len, 0);
  PASS();
}

TEST(safetensors, rejects_invalid_names_before_deterministic_sort) {
  float values[] = {1.0f, 2.0f};
  int64_t shape[] = {1};
  PolySafetensorEntry entries[] = {
      {.name = "valid", .data = &values[0], .shape = shape, .ndim = 1, .dtype = POLY_ST_F32},
      {.name = NULL, .data = &values[1], .shape = shape, .ndim = 1, .dtype = POLY_ST_F32},
  };
  int out_len = 7;
  ASSERT_EQ(poly_safetensors_encode(entries, 2, NULL, &out_len), NULL);
  ASSERT_INT_EQ(out_len, 0);
  entries[1].name = "";
  out_len = 7;
  ASSERT_EQ(poly_safetensors_encode(entries, 2, NULL, &out_len), NULL);
  ASSERT_INT_EQ(out_len, 0);
  PASS();
}

TEST(safetensors, rejects_non_object_json_roots) {
  uint8_t bytes[] = {
      4, 0, 0, 0, 0, 0, 0, 0, '[', '{', '}', ']',
  };
  int n = 7;
  ASSERT_EQ(poly_safetensors_decode(bytes, (int)sizeof(bytes), &n, NULL), NULL);
  ASSERT_INT_EQ(n, 0);
  n = 7;
  ASSERT_EQ(poly_safetensors_decode_ex(bytes, (int64_t)sizeof(bytes), &n, NULL), NULL);
  ASSERT_INT_EQ(n, 0);
  PASS();
}
