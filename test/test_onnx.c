#include "test_harness.h"
#include "../src/loaders/onnx_loader.h"
#include "../src/loaders/import_error.h"
#include "../vendor/cjson/cJSON.h"

static cJSON *onnx_fixture(void) {
  FILE *file = fopen("test/fixtures/onnx.json", "rb");
  if (!file) return NULL;
  char data[65536];
  size_t n = fread(data, 1, sizeof(data), file);
  bool ok = !ferror(file) && feof(file);
  fclose(file);
  return ok ? cJSON_ParseWithLength(data, n) : NULL;
}

static uint8_t *fixture_bytes(const char *str, size_t *size) {
  static const char alphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  uint8_t *out = malloc(strlen(str) + 1);
  if (!out) return NULL;
  unsigned acc = 0, bits = 0;
  *size = 0;
  for (; *str && *str != '='; str++) {
    const char *p = strchr(alphabet, *str);
    if (!p) {
      free(out);
      return NULL;
    }
    acc = (acc << 6) | (unsigned)(p - alphabet);
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      out[(*size)++] = (uint8_t)(acc >> bits);
    }
  }
  return out;
}

TEST(onnx, reference_graphs_preserve_context_and_storage) {
  cJSON *fixture = onnx_fixture();
  ASSERT_NOT_NULL(fixture);
  cJSON *cases = cJSON_GetObjectItemCaseSensitive(fixture, "cases");
  PolyCtx *ctx = poly_ctx_new();
  poly_ctx_set_logical_policy(ctx, POLY_LOGICAL_NEVER);
  cJSON *row;
  cJSON_ArrayForEach(row, cases) {
    size_t size, external_size = 0;
    uint8_t *data = fixture_bytes(cJSON_GetObjectItem(row, "onnx")->valuestring, &size);
    ASSERT_NOT_NULL(data);
    char *dims = cJSON_PrintUnformatted(cJSON_GetObjectItem(row, "dimensions"));
    cJSON *external = cJSON_GetObjectItem(row, "external")->child;
    uint8_t *weight = external ? fixture_bytes(external->valuestring, &external_size) : NULL;
    const char *name = external ? external->string : NULL;
    const uint8_t *ptr = weight;
    int64_t length = (int64_t)external_size;
    PolyOnnxOptions options = {dims, &name, &ptr, &length, external ? 1 : 0};
    int before = ctx->n_tensors;
    PolyModel *m = poly_onnx_load_into(ctx, data, (int64_t)size, &options, POLY_DEVICE_INTERP);
    ASSERT_NOT_NULL(m);
    ASSERT_EQ(ctx->n_tensors, before);
    ASSERT_EQ(poly_ctx_get_logical_policy(ctx), POLY_LOGICAL_NEVER);
    memset(data, 0, size);
    free(data);
    free(weight);
    free(dims);
    PolyIOBinding io[128];
    float *inputs[128];
    int ni = 0;
    cJSON *input, *output;
    cJSON_ArrayForEach(input, cJSON_GetObjectItem(row, "inputs")) {
      ASSERT_TRUE(ni < 128);
      cJSON *values = cJSON_GetObjectItem(input, "values");
      int n = cJSON_GetArraySize(values);
      inputs[ni] = malloc((size_t)n * sizeof(float));
      ASSERT_NOT_NULL(inputs[ni]);
      for (int i = 0; i < n; i++)
        inputs[ni][i] = (float)cJSON_GetArrayItem(values, i)->valuedouble;
      io[ni] =
          POLY_IO_BINDING_BYTES(input->string, inputs[ni], (size_t)n * sizeof(float), POLY_FLOAT32);
      ni++;
    }
    ASSERT_EQ(poly_model_call(m, "forward", io, ni), 0);
    for (int i = 0; i < ni; i++)
      free(inputs[i]);
    cJSON_ArrayForEach(output, cJSON_GetObjectItem(row, "outputs")) {
      cJSON *values = cJSON_GetObjectItem(output, "values");
      int n = cJSON_GetArraySize(values);
      float *y = malloc((size_t)n * sizeof(float));
      ASSERT_NOT_NULL(y);
      ASSERT_EQ(poly_model_read_buf_named(m, output->string, y, (size_t)n * sizeof(float)), 0);
      for (int i = 0; i < n; i++) {
        if (!(fabsf(y[i] - (float)cJSON_GetArrayItem(values, i)->valuedouble) <= 2e-5f))
          fprintf(
              stderr, "ONNX fixture %s, output %s[%d]\n",
              cJSON_GetObjectItem(row, "name")->valuestring, output->string, i
          );
        ASSERT_FLOAT_EQ(y[i], cJSON_GetArrayItem(values, i)->valuedouble, 2e-5);
      }
      free(y);
    }
    /* Imported graphs use normal logical arithmetic/reduction roots, never an
     * opaque foreign execution node or retained protobuf interpreter. */
    int count = 0;
    PolyUOp **nodes = poly_uop_toposort(ctx, poly_model_get_sink(m, "forward"), &count);
    ASSERT_TRUE(count > 0);
    int reductions = 0, selections = 0;
    for (int i = 0; i < count; i++) {
      ASSERT_TRUE(nodes[i]->op != POLY_OP_CUSTOM);
      if (nodes[i]->op == POLY_OP_REDUCE) reductions++;
      if (nodes[i]->op == POLY_OP_WHERE) selections++;
    }
    if (!strcmp(cJSON_GetObjectItem(row, "name")->valuestring, "mlp")) {
      ASSERT_EQ(reductions, 2);
      ASSERT_EQ(selections, 1);
    }
    if (!strncmp(cJSON_GetObjectItem(row, "name")->valuestring, "if_", 3))
      ASSERT_EQ(selections, 1); /* Tinygrad If selects equal-shaped branch values with WHERE. */
    poly_model_free(m);
  }
  poly_ctx_destroy(ctx);
  cJSON_Delete(fixture);
}

TEST(onnx, truncated_proto_and_varints_fail_without_leaking_handles) {
  cJSON *fixture = onnx_fixture();
  ASSERT_NOT_NULL(fixture);
  cJSON *row = cJSON_GetArrayItem(cJSON_GetObjectItem(fixture, "cases"), 0);
  size_t size;
  uint8_t *data = fixture_bytes(cJSON_GetObjectItem(row, "onnx")->valuestring, &size);
  ASSERT_NOT_NULL(data);
  PolyCtx *ctx = poly_ctx_new();
  PolyOnnxOptions options = {.dimensions_json = "{\"batch\":2}"};
  for (size_t n = 0; n < size; n++) {
    ASSERT_TRUE(poly_onnx_load_into(ctx, data, (int64_t)n, &options, POLY_DEVICE_INTERP) == NULL);
    ASSERT_TRUE(poly_import_last_error_code() != POLY_IMPORT_OK);
    ASSERT_EQ(ctx->n_tensors, 0);
  }
  const uint8_t invalid[] = {0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x01};
  ASSERT_TRUE(poly_onnx_load_into(ctx, invalid, sizeof(invalid), NULL, POLY_DEVICE_INTERP) == NULL);
  poly_ctx_destroy(ctx);
  free(data);
  cJSON_Delete(fixture);
}
