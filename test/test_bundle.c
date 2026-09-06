/*
 * test_bundle.c -- Tests for poly.bundle@1 container format
 */

#include "test_harness.h"
#include "../src/bundle.h"
#include "../src/ir.h"
#include "../src/model.h"
#include "../src/frontend.h"
#include "../src/models/mlp.h"
#include <limits.h>
#include <string.h>

/* Basic encode/decode */

TEST(bundle, encode_decode_roundtrip) {
  /* Create dummy IR and weights */
  uint8_t ir[] = {0x01, 0x02, 0x03, 0x04};
  uint8_t weights[] = {0xAA, 0xBB, 0xCC};
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
  uint8_t ir[] = {0x42};
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

TEST(bundle, decode_rejects_section_length_above_signed_api_range) {
  uint8_t bundle[28] = {0};
  memcpy(bundle, POLY_BUNDLE_MAGIC, 8);
  bundle[8] = POLY_BUNDLE_VERSION;
  bundle[16] = 1; /* one section */
  bundle[20] = POLY_BUNDLE_IR;
  bundle[24] = 0xff;
  bundle[25] = 0xff;
  bundle[26] = 0xff;
  bundle[27] = 0xff;

  PolyBundleSections sec;
  ASSERT_INT_EQ(poly_bundle_decode(bundle, (int)sizeof(bundle), &sec), -1);
  PASS();
}

TEST(bundle, encode_rejects_signed_length_overflow_before_allocation) {
  uint8_t byte = 0;
  int out_len = 7;
  ASSERT_EQ(poly_bundle_encode(&byte, INT_MAX, NULL, 0, NULL, &out_len), NULL);
  ASSERT_INT_EQ(out_len, 0);
  PASS();
}

/* Model round-trip via bundle */

TEST(bundle, instance_save_load_roundtrip) {
  /* Create MLP instance */
  const char *spec = "{\"layers\":[2,4,1],\"activation\":\"relu\",\"bias\":true,"
                     "\"loss\":\"mse\",\"batch_size\":1,\"seed\":42}";
  PolyModel *inst = poly_mlp_from_json(spec, (int)strlen(spec), POLY_DEVICE_AUTO);
  ASSERT_NOT_NULL(inst);

  /* Set some non-zero weight values */
  for (int p = 0; p < poly_model_param_count(inst); p++) {
    int64_t numel;
    float *data = poly_model_param_data(inst, p, &numel);
    for (int64_t j = 0; j < numel; j++)
      data[j] = (float)(j + 1) * 0.1f;
  }

  /* Forward on original */
  float input[] = {1.0f, 2.0f};
  PolyIOBinding io[] = {POLY_IO_BINDING_ARRAY("x", input, POLY_FLOAT32)};
  ASSERT_INT_EQ(poly_model_forward(inst, io, 1), 0);

  /* Read original output */
  int out_idx = -1;
  for (int i = 0; i < poly_model_buf_count(inst); i++)
    if (poly_model_buf_role(inst, i) == POLY_ROLE_OUTPUT) {
      out_idx = i;
      break;
    }
  ASSERT_TRUE(out_idx >= 0);
  int64_t numel;
  float *orig_out = poly_model_buf_data(inst, out_idx, &numel);
  float orig_val = orig_out[0];

  /* Save to bundle */
  int bundle_len = 0;
  uint8_t *bundle = poly_model_save_bundle(inst, &bundle_len);
  ASSERT_NOT_NULL(bundle);
  ASSERT_TRUE(bundle_len > 0);

  /* Verify bundle magic */
  ASSERT_TRUE(memcmp(bundle, "POLYBNDL", 8) == 0);

  /* Load from bundle */
  PolyModel *inst2 = poly_model_from_bundle(bundle, bundle_len);
  ASSERT_NOT_NULL(inst2);

  /* Same structure */
  ASSERT_INT_EQ(poly_model_buf_count(inst2), poly_model_buf_count(inst));
  ASSERT_INT_EQ(poly_model_param_count(inst2), poly_model_param_count(inst));

  /* Same weights */
  for (int p = 0; p < poly_model_param_count(inst2); p++) {
    int64_t n1, n2;
    float *d1 = poly_model_param_data(inst, p, &n1);
    float *d2 = poly_model_param_data(inst2, p, &n2);
    ASSERT_INT_EQ(n1, n2);
    for (int64_t j = 0; j < n1; j++)
      ASSERT_FLOAT_EQ(d1[j], d2[j], 0.0f);
  }

  /* Forward on loaded instance produces same output */
  ASSERT_INT_EQ(poly_model_forward(inst2, io, 1), 0);
  float *loaded_out = poly_model_buf_data(inst2, out_idx, &numel);
  ASSERT_FLOAT_EQ(loaded_out[0], orig_val, 1e-6);

  free(bundle);
  poly_model_free(inst);
  poly_model_free(inst2);
  PASS();
}

TEST(bundle, instance_save_includes_entrypoint_manifest_metadata) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyModel *inst = poly_model_new(ctx, NULL);
  ASSERT_NOT_NULL(inst);

  int64_t shape[] = {1};
  PolyTensor *x = poly_model_input(inst, "x", POLY_FLOAT32, shape, 1);
  ASSERT_NOT_NULL(x);
  PolyTensor *y = poly_model_target(inst, "y", POLY_FLOAT32, shape, 1);
  ASSERT_NOT_NULL(y);

  ASSERT_INT_EQ(poly_model_output(inst, "logits", x), POLY_STATUS_OK);
  ASSERT_INT_EQ(poly_model_output(inst, "loss", y), POLY_STATUS_OK);

  const char *forward_inputs[] = {"x"};
  const char *forward_outputs[] = {"logits"};
  ASSERT_INT_EQ(
      poly_model_entrypoint(inst, "forward", forward_inputs, 1, forward_outputs, 1, NULL),
      POLY_STATUS_OK
  );

  const char *loss_inputs[] = {"x", "y"};
  const char *loss_outputs[] = {"loss"};
  PolyEntrypointOptions opts = {.objective = "loss", .flags = 7};
  ASSERT_INT_EQ(
      poly_model_entrypoint(inst, "loss", loss_inputs, 2, loss_outputs, 1, &opts), POLY_STATUS_OK
  );
  ASSERT_INT_EQ(poly_model_build(inst, NULL), POLY_STATUS_OK);

  int bundle_len = 0;
  uint8_t *bundle = poly_model_save_bundle(inst, &bundle_len);
  ASSERT_NOT_NULL(bundle);

  PolyBundleSections sections;
  ASSERT_INT_EQ(poly_bundle_decode(bundle, bundle_len, &sections), 0);
  ASSERT_NOT_NULL(sections.metadata_json);
  ASSERT_TRUE(sections.metadata_len > 0);

  char *meta = malloc((size_t)sections.metadata_len + 1);
  ASSERT_NOT_NULL(meta);
  memcpy(meta, sections.metadata_json, (size_t)sections.metadata_len);
  meta[sections.metadata_len] = '\0';

  ASSERT_TRUE(strstr(meta, "\"format\":\"poly.bundle@1\"") != NULL);
  char ir_format[64];
  snprintf(ir_format, sizeof(ir_format), "\"ir_format\":\"poly.ir.uops@%d\"", POLY_IR_VERSION);
  ASSERT_TRUE(strstr(meta, ir_format) != NULL);
  ASSERT_TRUE(strstr(meta, "\"name\":\"forward\"") != NULL);
  ASSERT_TRUE(strstr(meta, "\"inputs\":[\"x\"]") != NULL);
  ASSERT_TRUE(strstr(meta, "\"outputs\":[\"logits\"]") != NULL);
  ASSERT_TRUE(strstr(meta, "\"objective\":null") != NULL);
  ASSERT_TRUE(strstr(meta, "\"name\":\"loss\"") != NULL);
  ASSERT_TRUE(strstr(meta, "\"inputs\":[\"x\",\"y\"]") != NULL);
  ASSERT_TRUE(strstr(meta, "\"outputs\":[\"loss\"]") != NULL);
  ASSERT_TRUE(strstr(meta, "\"objective\":\"loss\"") != NULL);
  ASSERT_TRUE(strstr(meta, "\"flags\":7") != NULL);

  free(meta);
  free(bundle);
  poly_model_free(inst);
  poly_ctx_destroy(ctx);
  PASS();
}
