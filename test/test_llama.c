#include "test_harness.h"
#include "../src/models/models.h"
#include "../src/models/registry.h"
#include "../src/models/factory.h"
#include "../src/models/transformer.h"

/* Same dense block with Qwen's per-head norms and head_dim != dim/heads.
 * The shared builder must not inherit Llama's narrower head-width assumption. */
TEST(llama, shared_dense_qk_norm_cached_and_uncached) {
  /* Pinned llm/model.py TransformerBlock, same weights as below, FP32 cache
   * explicitly selected and output tied to token_embd. Last-position logits
   * also agree with the Python Polygrad Transformer (dense_reference.py probe). */
  const float reference[2][11] = {
      {.4031510353f, -.1170670763f, .3597702086f, -.1842437834f, .3163893521f, .0629191995f,
       .2730085552f, -.4285599291f, .2296277434f, -.4448931813f, .1862469465f},
      {.3968558013f, -.1096408293f, .3542748690f, -.1700790524f, .3116938770f, .0751177371f,
       .2691128850f, -.4423021674f, .2265319228f, -.4519312382f, .1839509606f}};
  const ModelTransformerNames names = {
      .label = "dense test",
      .input = "x",
      .output = "output",
      .cos = "rope_cos",
      .sin = "rope_sin",
      .embedding = "token_embd",
      .norm = "output_norm",
      .block = "blk.%d",
      .attn_norm = "attn_norm",
      .qkv = {"attn_q", "attn_k", "attn_v"},
      .qk_norm = {"attn_q_norm", "attn_k_norm"},
      .out = "attn_output",
      .ffn_norm = "ffn_norm",
      .gate = "ffn_gate",
      .up = "ffn_up",
      .down = "ffn_down"};
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  for (int normalized = 0; normalized < 2; normalized++) {
    Qwen3Config q = poly_qwen3_config_default();
    q.vocab_size = 11;
    q.dim = 12;
    q.n_heads = 2;
    q.n_kv_heads = 1;
    q.n_layers = 2;
    q.hidden_dim = 16;
    q.head_dim = 4;
    q.max_seq_len = 3;
    q.qk_norm = normalized ? 4 : 0;
    PolyModel *plain = poly_qwen3_into(ctx, &q, POLY_DEVICE_INTERP);
    ASSERT_NOT_NULL(plain);
    ModelTransformerConfig c = {
        .dim = q.dim,
        .hidden_dim = q.hidden_dim,
        .heads = q.n_heads,
        .kv_heads = q.n_kv_heads,
        .layers = q.n_layers,
        .vocab = q.vocab_size,
        .batch = 1,
        .length = q.max_seq_len,
        .head_dim = q.head_dim,
        .qk_norm = q.qk_norm,
        .cache_capacity = 5,
        .prefill_chunk = 4,
        .eps = q.norm_eps,
        .theta = q.rope_theta,
        .factor = 1,
        .tied = true,
        .materialize_intermediates = true};
    PolyModelFactoryScope scope;
    ASSERT_TRUE(model_factory_begin(&scope, ctx, POLY_DEVICE_INTERP));
    PolyModelError err = {0};
    PolyModel *cached = model_factory_end(&scope, model_transformer_build(ctx, &c, &names, &err));
    if (!cached) {
      poly_model_free(plain);
      poly_ctx_destroy(ctx);
      FAIL("shared cached construction: %s", err.message);
    }
    ASSERT_EQ(poly_model_param_count(plain), poly_model_param_count(cached));
    for (int i = 0; i < poly_model_buf_count(plain); i++) {
      if (poly_model_buf_role(plain, i) != POLY_ROLE_PARAM) continue;
      const char *name = poly_model_buf_name(plain, i);
      int64_t n = poly_model_buf_numel_named(plain, name);
      float *data = malloc((size_t)n * sizeof(*data));
      ASSERT_NOT_NULL(data);
      for (int64_t j = 0; j < n; j++)
        data[j] = (strstr(name, "norm") ? 1.f : 0.f) + (float)(j % 23 - 11) * .017f;
      ASSERT_EQ(poly_model_write_buf_named(plain, name, data, (size_t)n * sizeof(*data)), 0);
      ASSERT_EQ(poly_model_write_buf_named(cached, name, data, (size_t)n * sizeof(*data)), 0);
      free(data);
    }
    PolyTransformer *g = poly_transformer_from_model(cached, NULL);
    ASSERT_NOT_NULL(g);
    int32_t tokens[] = {1, 4, 2};
    PolyIOBinding io = POLY_IO_BINDING_ARRAY("x", tokens, POLY_INT32);
    ASSERT_EQ(poly_model_call(plain, "forward", &io, 1), 0);
    float expected[33], actual[11];
    ASSERT_EQ(poly_model_read_buf_named(plain, "output", expected, sizeof(expected)), 0);
    for (int j = 0; j < 11; j++)
      ASSERT_FLOAT_EQ(expected[22 + j], reference[normalized][j], 3e-5);
    for (int partition = 0; partition < 3; partition++) {
      ASSERT_EQ(poly_model_reset_transient(cached), 0);
      for (int pos = 0; pos < 3;) {
        int count = partition == 0 && pos == 0 ? 2 : partition == 2 ? 3 : 1;
        ASSERT_EQ(poly_transformer_append(g, tokens + pos, count, actual, 11), 0);
        pos += count;
        ASSERT_EQ(poly_transformer_position(g), pos);
        for (int j = 0; j < 11; j++)
          ASSERT_FLOAT_EQ(actual[j], expected[(pos - 1) * 11 + j], 3e-5);
      }
    }
    /* Both bindings fit independently. Reject their combined write window
     * before invalidating the completed prefix or touching cache storage. */
    PolyIOBinding invalid = POLY_IO_BINDING_ARRAY("tokens_prefill", tokens, POLY_INT32);
    PolyControlBinding control = {"start_pos", 3};
    float before[40], after[40];
    ASSERT_EQ(poly_model_read_buf_named(cached, "blk.0.cache_kv", before, sizeof(before)), 0);
    ASSERT_TRUE(poly_model_call_with_controls(cached, "prefill", &invalid, 1, &control, 1) != 0);
    ASSERT_EQ(poly_transformer_position(g), 3);
    ASSERT_EQ(poly_model_read_buf_named(cached, "blk.0.cache_kv", after, sizeof(after)), 0);
    ASSERT_EQ(memcmp(before, after, sizeof(before)), 0);
    poly_transformer_free(g);
    poly_model_free(plain);
    poly_ctx_collect(ctx);
  }
  poly_ctx_destroy(ctx);
  PASS();
}

/* Shared deterministic oracle, generated by test/generate_llama_fixture.py. */
static cJSON *llama_fixture(void) {
  FILE *f = fopen("test/fixtures/llama.json", "rb");
  if (!f) return NULL;
  char data[32768];
  size_t n = fread(data, 1, sizeof(data), f);
  bool ok = !ferror(f) && feof(f);
  fclose(f);
  return ok ? cJSON_ParseWithLength(data, n) : NULL;
}

TEST(llama, reference_logits_and_owned_state) {
  cJSON *fixture = llama_fixture();
  ASSERT_NOT_NULL(fixture);
  cJSON *cases = cJSON_GetObjectItemCaseSensitive(fixture, "cases");
  ASSERT_EQ(cJSON_GetArraySize(cases), 4);
  const PolyModelType *desc = model_type_find("llama");
  ASSERT_TRUE(desc && desc->from_hf_decoded);
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  for (int c = 0; c < 5; c++) {
    cJSON *item = cJSON_GetArrayItem(cases, c < 4 ? c : 3);
    cJSON *config = cJSON_GetObjectItemCaseSensitive(item, "config");
    cJSON *weights = cJSON_GetObjectItemCaseSensitive(item, "weights");
    int n = cJSON_GetArraySize(weights), idx = 0;
    PolyDecodedTensor *ts = calloc((size_t)n, sizeof(*ts));
    ASSERT_NOT_NULL(ts);
    cJSON *weight;
    cJSON_ArrayForEach(weight, weights) {
      PolyDecodedTensor *t = &ts[idx++];
      t->name = weight->string;
      t->ndim = cJSON_GetArraySize(weight);
      t->numel = 1;
      int offset = 0;
      for (const char *p = t->name; *p; p++)
        offset += (unsigned char)*p;
      for (int d = 0; d < t->ndim; d++) {
        t->shape[d] = cJSON_GetArrayItem(weight, d)->valueint;
        t->numel *= t->shape[d];
      }
      float *data = malloc((size_t)t->numel * sizeof(float));
      ASSERT_NOT_NULL(data);
      for (int64_t j = 0; j < t->numel; j++)
        data[j] = (float)((t->ndim == 1 ? 1.0 : 0.0) + ((j * 7 + offset) % 23 - 11) * 0.017);
      t->data = data;
      t->dtype = POLY_DECODED_F32;
      /* Safetensors may retain either member of a tied embedding/head pair. */
      if (c == 4 && !strcmp(t->name, "model.embed_tokens.weight")) t->name = "lm_head.weight";
    }
    PolyHfDecoded hf = {.config = config, .model_type = "llama", .tensors = ts, .n_tensors = n};
    PolyGenericImportOpts opts = {.ctx = ctx, .max_batch = 1, .max_seq_len = 3};
    PolyModel *m = desc->from_hf_decoded(&hf, &opts);
    for (int i = 0; i < n; i++)
      free((void *)ts[i].data);
    free(ts);
    ASSERT_NOT_NULL(m);
    int32_t tokens[] = {1, 4, 2};
    PolyIOBinding io = POLY_IO_BINDING_ARRAY("tokens", tokens, POLY_INT32);
    ASSERT_EQ(poly_model_call(m, "forward", &io, 1), 0);
    float values[33];
    ASSERT_EQ(poly_model_read_buf_named(m, "logits", values, sizeof(values)), 0);
    cJSON *expected = cJSON_GetObjectItemCaseSensitive(item, "logits");
    for (int j = 0; j < 33; j++)
      ASSERT_FLOAT_EQ(values[j], cJSON_GetArrayItem(expected, j)->valuedouble, 2e-5);
    poly_model_free(m);
    poly_ctx_collect(ctx);
  }
  poly_ctx_destroy(ctx);
  cJSON_Delete(fixture);
  PASS();
}

TEST(llama, malformed_config_preserves_context) {
  PolyCtx *ctx = poly_ctx_new();
  PolyLogicalPolicy policy = poly_ctx_get_logical_policy(ctx);
  const char *inputs[] = {"{}", "[]", "null", "{} false", "{\"hidden_size\":1.5}"};
  for (int i = 0; i < 5; i++) {
    PolyModelError err = {0};
    PolyModel *m = poly_model_from_config(
        ctx, "llama", inputs[i], (int)strlen(inputs[i]), POLY_DEVICE_AUTO, &err
    );
    ASSERT_TRUE(!m && err.message[0]);
    ASSERT_EQ(poly_ctx_get_logical_policy(ctx), policy);
  }
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(llama, missing_weights_reject_execution_and_export) {
  const char *json = "{\"hidden_size\":4,\"intermediate_size\":8,\"num_attention_heads\":2,"
                     "\"num_hidden_layers\":1,\"vocab_size\":8,\"tie_word_embeddings\":true}";
  PolyModelError err = {0};
  PolyModel *m =
      poly_model_from_config(NULL, "llama", json, (int)strlen(json), POLY_DEVICE_AUTO, &err);
  ASSERT_NOT_NULL(m);
  int32_t tokens[] = {1};
  PolyIOBinding io = POLY_IO_BINDING_ARRAY("tokens", tokens, POLY_INT32);
  ASSERT_INT_EQ(poly_model_call(m, "forward", &io, 1), -1);
  ASSERT_TRUE(strstr(poly_model_last_error(m)->message, "not initialized") != NULL);
  int len = 123;
  ASSERT_TRUE(poly_model_export_ir(m, &len) == NULL);
  ASSERT_INT_EQ(len, 0);
  ASSERT_TRUE(poly_model_export_weights(m, &len) == NULL);
  ASSERT_INT_EQ(len, 0);
  /* Zero is a valid explicit weight value. Tied aliases need one write, not
   * one per name; a wrong-sized write must not publish readiness. */
  for (int i = 0; i < poly_model_param_count(m); i++) {
    const char *name = poly_model_param_name(m, i);
    if (!strcmp(name, "lm_head.weight")) continue;
    size_t bytes = (size_t)poly_model_buf_numel_named(m, name) * sizeof(float);
    float *values = calloc(1, bytes);
    ASSERT_NOT_NULL(values);
    ASSERT_INT_EQ(poly_model_upload_param(m, i, values, bytes - 1), 0);
    ASSERT_INT_EQ(poly_model_call(m, "forward", &io, 1), -1);
    ASSERT_TRUE(poly_model_export_ir(m, &len) == NULL);
    ASSERT_INT_EQ(poly_model_upload_param(m, i, values, bytes + 1), -1);
    ASSERT_INT_EQ(poly_model_upload_param(m, i, values, bytes), 0);
    free(values);
  }
  ASSERT_INT_EQ(poly_model_call(m, "forward", &io, 1), 0);
  PolyModel *copy =
      poly_model_from_config(NULL, "llama", json, (int)strlen(json), POLY_DEVICE_AUTO, &err);
  ASSERT_NOT_NULL(copy);
  ASSERT_INT_EQ(poly_model_copy_prefixed_weights(copy, m, ""), 0);
  ASSERT_INT_EQ(poly_model_call(copy, "forward", &io, 1), 0);
  poly_model_free(copy);
  uint8_t *ir = poly_model_export_ir(m, &len);
  ASSERT_NOT_NULL(ir);
  free(ir);
  poly_model_free(m);
  PASS();
}

TEST(llama, rejected_bias_reports_invalid_status) {
  PolyCtx *ctx = poly_ctx_new();
  const char *json[] = {"{\"attention_bias\":true}", "{\"mlp_bias\":true}"};
  int codes[2];
  bool rejected[2];
  for (int i = 0; i < 2; i++) {
    PolyModelError err = {0};
    PolyModel *m =
        poly_model_from_config(ctx, "llama", json[i], (int)strlen(json[i]), POLY_DEVICE_AUTO, &err);
    codes[i] = err.code;
    rejected[i] = !m && strstr(err.message, "must be false");
    poly_model_free(m);
  }
  poly_ctx_destroy(ctx);
  for (int i = 0; i < 2; i++) {
    ASSERT_TRUE(rejected[i]);
    ASSERT_INT_EQ(codes[i], POLY_STATUS_INVALID);
  }
  PASS();
}
