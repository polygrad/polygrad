/*
 * test_bundle.c -- Tests for poly.bundle@1 container format
 */

#include "test_harness.h"
#include "../src/bundle.h"
#include "../src/ir.h"
#include "../src/instance.h"
#include "../src/frontend.h"
#include "../src/model_mlp.h"
#include <string.h>

/* ── Basic encode/decode ─────────────────────────────────────────────── */

TEST(bundle, encode_decode_roundtrip) {
  /* Create dummy IR and weights */
  uint8_t ir[] = { 0x01, 0x02, 0x03, 0x04 };
  uint8_t weights[] = { 0xAA, 0xBB, 0xCC };
  const char *meta = "{\"model\":\"test\"}";

  int bundle_len = 0;
  uint8_t *bundle = poly_bundle_encode(ir, 4, weights, 3, meta, &bundle_len);
  ASSERT_NOT_NULL(bundle);
  ASSERT_TRUE(bundle_len > 0);

  /* Check magic */
  ASSERT_TRUE(memcmp(bundle, "POLYBNDL", 8) == 0);

  /* Decode */
  PolyBundleSections sec;
  ASSERT_INT_EQ(poly_bundle_decode(bundle, bundle_len, &sec), 0);

  ASSERT_INT_EQ(sec.version, 1);
  ASSERT_INT_EQ(sec.ir_len, 4);
  ASSERT_TRUE(memcmp(sec.ir_data, ir, 4) == 0);
  ASSERT_INT_EQ(sec.weights_len, 3);
  ASSERT_TRUE(memcmp(sec.weights_data, weights, 3) == 0);
  ASSERT_INT_EQ(sec.metadata_len, (int)strlen(meta));
  ASSERT_TRUE(memcmp(sec.metadata_json, meta, sec.metadata_len) == 0);

  free(bundle);
  PASS();
}

TEST(bundle, encode_ir_only) {
  uint8_t ir[] = { 0x42 };
  int bundle_len = 0;
  uint8_t *bundle = poly_bundle_encode(ir, 1, NULL, 0, NULL, &bundle_len);
  ASSERT_NOT_NULL(bundle);

  PolyBundleSections sec;
  ASSERT_INT_EQ(poly_bundle_decode(bundle, bundle_len, &sec), 0);
  ASSERT_INT_EQ(sec.ir_len, 1);
  ASSERT_TRUE(sec.ir_data[0] == 0x42);
  ASSERT_TRUE(sec.weights_data == NULL);
  ASSERT_TRUE(sec.metadata_json == NULL);

  free(bundle);
  PASS();
}

TEST(bundle, decode_bad_magic) {
  uint8_t bad[] = "NOTABNDL\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00";
  PolyBundleSections sec;
  ASSERT_INT_EQ(poly_bundle_decode(bad, sizeof(bad), &sec), -1);
  PASS();
}

TEST(bundle, decode_truncated) {
  PolyBundleSections sec;
  ASSERT_INT_EQ(poly_bundle_decode((const uint8_t *)"POLY", 4, &sec), -1);
  PASS();
}

TEST(bundle, decode_null_input) {
  PolyBundleSections sec;
  ASSERT_INT_EQ(poly_bundle_decode(NULL, 0, &sec), -1);
  PASS();
}

/* ── Instance round-trip via bundle ───────────────────────────────────── */

TEST(bundle, instance_save_load_roundtrip) {
  /* Create MLP instance */
  const char *spec = "{\"layers\":[2,4,1],\"activation\":\"relu\",\"bias\":true,"
                     "\"loss\":\"mse\",\"batch_size\":1,\"seed\":42}";
  PolyInstance *inst = poly_mlp_instance(spec, (int)strlen(spec));
  ASSERT_NOT_NULL(inst);

  /* Set some non-zero weight values */
  for (int p = 0; p < poly_instance_param_count(inst); p++) {
    int64_t numel;
    float *data = poly_instance_param_data(inst, p, &numel);
    for (int64_t j = 0; j < numel; j++)
      data[j] = (float)(j + 1) * 0.1f;
  }

  /* Forward on original */
  float input[] = { 1.0f, 2.0f };
  PolyIOBinding io[] = { {"x", input} };
  ASSERT_INT_EQ(poly_instance_forward(inst, io, 1), 0);

  /* Read original output */
  int out_idx = -1;
  for (int i = 0; i < poly_instance_buf_count(inst); i++)
    if (poly_instance_buf_role(inst, i) == POLY_ROLE_OUTPUT) { out_idx = i; break; }
  ASSERT_TRUE(out_idx >= 0);
  int64_t numel;
  float *orig_out = poly_instance_buf_data(inst, out_idx, &numel);
  float orig_val = orig_out[0];

  /* Save to bundle */
  int bundle_len = 0;
  uint8_t *bundle = poly_instance_save_bundle(inst, &bundle_len);
  ASSERT_NOT_NULL(bundle);
  ASSERT_TRUE(bundle_len > 0);

  /* Verify bundle magic */
  ASSERT_TRUE(memcmp(bundle, "POLYBNDL", 8) == 0);

  /* Load from bundle */
  PolyInstance *inst2 = poly_instance_from_bundle(bundle, bundle_len);
  ASSERT_NOT_NULL(inst2);

  /* Same structure */
  ASSERT_INT_EQ(poly_instance_buf_count(inst2), poly_instance_buf_count(inst));
  ASSERT_INT_EQ(poly_instance_param_count(inst2), poly_instance_param_count(inst));

  /* Same weights */
  for (int p = 0; p < poly_instance_param_count(inst2); p++) {
    int64_t n1, n2;
    float *d1 = poly_instance_param_data(inst, p, &n1);
    float *d2 = poly_instance_param_data(inst2, p, &n2);
    ASSERT_INT_EQ(n1, n2);
    for (int64_t j = 0; j < n1; j++)
      ASSERT_FLOAT_EQ(d1[j], d2[j], 0.0f);
  }

  /* Forward on loaded instance produces same output */
  ASSERT_INT_EQ(poly_instance_forward(inst2, io, 1), 0);
  float *loaded_out = poly_instance_buf_data(inst2, out_idx, &numel);
  ASSERT_FLOAT_EQ(loaded_out[0], orig_val, 1e-6);

  free(bundle);
  poly_instance_free(inst);
  poly_instance_free(inst2);
  PASS();
}
