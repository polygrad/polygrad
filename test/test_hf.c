/*
 * test_hf.c -- Tests for HuggingFace model loading infrastructure
 *
 * Covers: multi-dtype safetensors, PolyModelConfig, GPT-2 builder,
 *         HF loader, poly_gather, poly_layernorm, poly_linear.
 */

#include "test_harness.h"
#include "../src/safetensors.h"
#include "../src/modelzoo/modelzoo.h"
#include "../src/frontend.h"
#include "../src/sched.h"
#include <string.h>
#include <stdlib.h>
#include <math.h>

/* ── Safetensors multi-dtype ─────────────────────────────────────── */

/* Helper: create a minimal safetensors file with given dtype string */
static uint8_t *make_st_file(const char *name, const char *dtype_str,
                              const void *data, int64_t nbytes,
                              const int64_t *shape, int ndim,
                              int64_t *out_len) {
  /* Build JSON header manually */
  char header[512];
  char shape_str[128] = "[";
  for (int i = 0; i < ndim; i++) {
    char dim[32];
    snprintf(dim, sizeof(dim), "%s%lld", i > 0 ? "," : "", (long long)shape[i]);
    strcat(shape_str, dim);
  }
  strcat(shape_str, "]");

  snprintf(header, sizeof(header),
           "{\"%s\":{\"dtype\":\"%s\",\"shape\":%s,\"data_offsets\":[0,%lld]}}",
           name, dtype_str, shape_str, (long long)nbytes);

  uint64_t header_size = strlen(header);
  uint64_t total = 8 + header_size + (uint64_t)nbytes;

  uint8_t *buf = malloc((size_t)total);
  /* Write header size LE */
  for (int i = 0; i < 8; i++) buf[i] = (uint8_t)(header_size >> (i * 8));
  memcpy(buf + 8, header, header_size);
  memcpy(buf + 8 + header_size, data, nbytes);

  *out_len = (int64_t)total;
  return buf;
}

TEST(hf, safetensors_decode_ex_f32) {
  float data[] = { 1.0f, 2.0f, 3.0f };
  int64_t shape[] = { 3 };
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
  uint16_t data[] = { 0x3C00, 0x3800, 0xBC00 };
  int64_t shape[] = { 3 };
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
  uint16_t data[] = { 0x3F80, 0xC000 };
  int64_t shape[] = { 2 };
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
  int32_t data[] = { 42, -7, 0, 100 };
  int64_t shape[] = { 4 };
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
  float data[] = { 1.0f };
  /* Manually build with trailing spaces */
  const char *header = "{\"w\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}   ";
  uint64_t header_size = strlen(header);
  uint64_t total = 8 + header_size + 4;
  uint8_t *buf = malloc((size_t)total);
  for (int i = 0; i < 8; i++) buf[i] = (uint8_t)(header_size >> (i * 8));
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

/* ── PolyModelConfig ─────────────────────────────────────────────── */

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

/* ── GPT-2 builder ───────────────────────────────────────────────── */

TEST(hf, gpt2_build_tiny) {
  GPT2Config cfg = {
    .vocab_size = 32,
    .n_embd = 16,
    .n_head = 2,
    .n_layer = 1,
    .max_seq_len = 8,
    .norm_eps = 1e-5f
  };

  PolyInstance *inst = poly_gpt2_build(&cfg, 1);
  ASSERT_NOT_NULL(inst);

  /* Check param count: wte + wpe + 1 layer (12 params) + ln_f (2) = 16 */
  int n_params = poly_instance_param_count(inst);
  ASSERT_INT_EQ(n_params, 16);

  /* Check wte.weight shape */
  int64_t shape[8];
  for (int i = 0; i < n_params; i++) {
    const char *name = poly_instance_param_name(inst, i);
    if (strcmp(name, "wte.weight") == 0) {
      int ndim = poly_instance_param_shape(inst, i, shape, 8);
      ASSERT_INT_EQ(ndim, 2);
      ASSERT_INT_EQ(shape[0], 32);  /* vocab_size */
      ASSERT_INT_EQ(shape[1], 16);  /* n_embd */
    }
  }

  /* Check we have the expected buffer names */
  int n_bufs = poly_instance_buf_count(inst);
  ASSERT_TRUE(n_bufs >= 16 + 4);  /* params + x + output + positions + arange */

  int found_x = 0, found_output = 0;
  for (int i = 0; i < n_bufs; i++) {
    const char *name = poly_instance_buf_name(inst, i);
    if (strcmp(name, "x") == 0) found_x = 1;
    if (strcmp(name, "output") == 0) found_output = 1;
  }
  ASSERT_TRUE(found_x);
  ASSERT_TRUE(found_output);

  poly_instance_free(inst);
  PASS();
}

TEST(hf, gpt2_build_multi_layer) {
  GPT2Config cfg = {
    .vocab_size = 64,
    .n_embd = 32,
    .n_head = 4,
    .n_layer = 3,
    .max_seq_len = 16,
    .norm_eps = 1e-5f
  };

  PolyInstance *inst = poly_gpt2_build(&cfg, 2);
  ASSERT_NOT_NULL(inst);

  /* 2 (wte+wpe) + 3*12 (layers) + 2 (ln_f) = 40 params */
  ASSERT_INT_EQ(poly_instance_param_count(inst), 40);

  /* Verify a deep layer param exists */
  int found = 0;
  int n_params = poly_instance_param_count(inst);
  for (int i = 0; i < n_params; i++) {
    if (strcmp(poly_instance_param_name(inst, i), "h.2.mlp.c_proj.weight") == 0)
      found = 1;
  }
  ASSERT_TRUE(found);

  poly_instance_free(inst);
  PASS();
}

/* ── HF loader ───────────────────────────────────────────────────── */

TEST(hf, hf_load_tiny_gpt2) {
  const char *config = "{\"model_type\":\"gpt2\",\"vocab_size\":32,"
                       "\"n_embd\":16,\"n_head\":2,\"n_layer\":1,"
                       "\"n_positions\":8,\"layer_norm_epsilon\":1e-5}";

  /* Create a safetensors file with a few test weights */
  float wte_data[32 * 16];
  for (int i = 0; i < 32 * 16; i++) wte_data[i] = (float)i * 0.001f;

  int64_t wte_shape[] = { 32, 16 };
  int64_t file_len;
  uint8_t *file = make_st_file("transformer.wte.weight", "F32",
                                wte_data, sizeof(wte_data), wte_shape, 2,
                                &file_len);

  const uint8_t *files[] = { file };
  int64_t lens[] = { file_len };

  PolyInstance *inst = poly_hf_load(config, (int)strlen(config),
                                     files, lens, 1, 1, 8);
  ASSERT_NOT_NULL(inst);

  /* Verify wte.weight was loaded */
  int n_params = poly_instance_param_count(inst);
  for (int i = 0; i < n_params; i++) {
    if (strcmp(poly_instance_param_name(inst, i), "wte.weight") == 0) {
      int64_t numel;
      float *data = poly_instance_param_data(inst, i, &numel);
      ASSERT_INT_EQ(numel, 32 * 16);
      ASSERT_FLOAT_EQ(data[0], 0.0f, 1e-6f);
      ASSERT_FLOAT_EQ(data[1], 0.001f, 1e-6f);
      break;
    }
  }

  poly_instance_free(inst);
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
  int64_t bias_shape[] = { 1, 1, 8, 8 };
  int64_t file_len;
  uint8_t *file = make_st_file("transformer.h.0.attn.bias", "F32",
                                bias_data, sizeof(bias_data), bias_shape, 4,
                                &file_len);

  const uint8_t *files[] = { file };
  int64_t lens[] = { file_len };

  /* Should not crash */
  PolyInstance *inst = poly_hf_load(config, (int)strlen(config),
                                     files, lens, 1, 1, 8);
  ASSERT_NOT_NULL(inst);

  poly_instance_free(inst);
  free(file);
  PASS();
}

/* ── Frontend ops ────────────────────────────────────────────────── */

TEST(hf, poly_gather_basic) {
  PolyCtx *ctx = poly_ctx_new();

  /* table: (4, 3) weight matrix */
  int64_t table_shape[] = { 4, 3 };
  PolyUOp *table = poly_buffer_f32(ctx, 12);

  /* indices: (2,) */
  int64_t idx_shape[] = { 2 };
  PolyUOp *indices = poly_buffer_f32(ctx, 2);

  int64_t out_shape[8];
  int out_ndim;
  PolyUOp *result = poly_gather(ctx, table, table_shape, 2,
                                  indices, idx_shape, 1,
                                  out_shape, &out_ndim);
  ASSERT_NOT_NULL(result);
  ASSERT_INT_EQ(out_ndim, 2);
  ASSERT_INT_EQ(out_shape[0], 2);  /* num indices */
  ASSERT_INT_EQ(out_shape[1], 3);  /* embedding dim */

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(hf, poly_gather_2d_indices) {
  PolyCtx *ctx = poly_ctx_new();

  /* table: (10, 4) */
  int64_t table_shape[] = { 10, 4 };
  PolyUOp *table = poly_buffer_f32(ctx, 40);

  /* indices: (2, 3) -- batch of indices */
  int64_t idx_shape[] = { 2, 3 };
  PolyUOp *indices = poly_buffer_f32(ctx, 6);

  int64_t out_shape[8];
  int out_ndim;
  PolyUOp *result = poly_gather(ctx, table, table_shape, 2,
                                  indices, idx_shape, 2,
                                  out_shape, &out_ndim);
  ASSERT_NOT_NULL(result);
  ASSERT_INT_EQ(out_ndim, 3);
  ASSERT_INT_EQ(out_shape[0], 2);   /* batch */
  ASSERT_INT_EQ(out_shape[1], 3);   /* seq_len */
  ASSERT_INT_EQ(out_shape[2], 4);   /* embed_dim */

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(hf, poly_layernorm_shape) {
  PolyCtx *ctx = poly_ctx_new();

  int64_t shape[] = { 2, 3, 4 };
  PolyUOp *x = poly_buffer_f32(ctx, 24);

  int64_t out_shape[8];
  int out_ndim;
  PolyUOp *result = poly_layernorm(ctx, x, shape, 3, -1, 1e-5,
                                     out_shape, &out_ndim);
  ASSERT_NOT_NULL(result);
  ASSERT_INT_EQ(out_ndim, 3);
  ASSERT_INT_EQ(out_shape[0], 2);
  ASSERT_INT_EQ(out_shape[1], 3);
  ASSERT_INT_EQ(out_shape[2], 4);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(hf, poly_linear_shape) {
  PolyCtx *ctx = poly_ctx_new();

  /* x: (2, 3, 4), weight: (8, 4), bias: (8,) */
  int64_t x_shape[] = { 2, 3, 4 };
  PolyUOp *x = poly_buffer_f32(ctx, 24);
  int64_t w_shape[] = { 8, 4 };
  PolyUOp *w = poly_buffer_f32(ctx, 32);
  int64_t b_shape[] = { 8 };
  PolyUOp *b = poly_buffer_f32(ctx, 8);

  int64_t out_shape[8];
  int out_ndim;
  PolyUOp *result = poly_linear(ctx, x, x_shape, 3, w, w_shape, 2,
                                  b, b_shape, 1, out_shape, &out_ndim);
  ASSERT_NOT_NULL(result);
  ASSERT_INT_EQ(out_ndim, 3);
  ASSERT_INT_EQ(out_shape[0], 2);
  ASSERT_INT_EQ(out_shape[1], 3);
  ASSERT_INT_EQ(out_shape[2], 8);  /* out_features */

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(hf, poly_linear_no_bias) {
  PolyCtx *ctx = poly_ctx_new();

  int64_t x_shape[] = { 4, 8 };
  PolyUOp *x = poly_buffer_f32(ctx, 32);
  int64_t w_shape[] = { 16, 8 };
  PolyUOp *w = poly_buffer_f32(ctx, 128);

  int64_t out_shape[8];
  int out_ndim;
  PolyUOp *result = poly_linear(ctx, x, x_shape, 2, w, w_shape, 2,
                                  NULL, NULL, 0, out_shape, &out_ndim);
  ASSERT_NOT_NULL(result);
  ASSERT_INT_EQ(out_ndim, 2);
  ASSERT_INT_EQ(out_shape[0], 4);
  ASSERT_INT_EQ(out_shape[1], 16);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(hf, poly_causal_mask_shape) {
  PolyCtx *ctx = poly_ctx_new();

  int64_t out_shape[8];
  int out_ndim;
  PolyUOp *mask = poly_causal_mask(ctx, 5, out_shape, &out_ndim);
  ASSERT_NOT_NULL(mask);
  ASSERT_INT_EQ(out_ndim, 2);
  ASSERT_INT_EQ(out_shape[0], 5);
  ASSERT_INT_EQ(out_shape[1], 5);

  poly_ctx_destroy(ctx);
  PASS();
}

/* ── GPT-2 forward pass e2e ───────────────────────────────────── */

TEST(hf, gpt2_forward_e2e) {
  GPT2Config cfg = {
    .vocab_size = 32,
    .n_embd = 16,
    .n_head = 2,
    .n_layer = 1,
    .max_seq_len = 8,
    .norm_eps = 1e-5f
  };

  PolyInstance *inst = poly_gpt2_build(&cfg, 1);
  ASSERT_NOT_NULL(inst);

  /* Initialize weights with small random values */
  int np = poly_instance_param_count(inst);
  for (int i = 0; i < np; i++) {
    int64_t numel;
    float *data = poly_instance_param_data(inst, i, &numel);
    const char *name = poly_instance_param_name(inst, i);
    /* LayerNorm weights init to 1, biases to 0 */
    if (strstr(name, "ln_") && strstr(name, "weight")) {
      for (int64_t j = 0; j < numel; j++) data[j] = 1.0f;
    } else if (strstr(name, "bias")) {
      for (int64_t j = 0; j < numel; j++) data[j] = 0.0f;
    } else {
      for (int64_t j = 0; j < numel; j++)
        data[j] = 0.02f * ((float)(j % 100) / 100.0f - 0.5f);
    }
  }

  /* Set input tokens: [0, 1, 2, 3] */
  int nb = poly_instance_buf_count(inst);
  for (int i = 0; i < nb; i++) {
    const char *name = poly_instance_buf_name(inst, i);
    int64_t numel;
    float *data = poly_instance_buf_data(inst, i, &numel);
    if (strcmp(name, "x") == 0) {
      /* 1 batch, T=8 but only first 4 tokens matter */
      for (int64_t j = 0; j < numel; j++) data[j] = (float)(j % 4);
    } else if (strcmp(name, "positions") == 0) {
      for (int64_t j = 0; j < numel; j++) data[j] = (float)j;
    } else if (strcmp(name, "arange") == 0) {
      for (int64_t j = 0; j < numel; j++) data[j] = (float)j;
    }
  }

  /* Run forward pass */
  int ret = poly_instance_forward(inst, NULL, 0);
  ASSERT_INT_EQ(ret, 0);

  /* Check output: (1, 8, 32) logits */
  int out_idx = -1;
  for (int i = 0; i < nb; i++) {
    if (strcmp(poly_instance_buf_name(inst, i), "output") == 0) {
      out_idx = i;
      break;
    }
  }
  ASSERT_TRUE(out_idx >= 0);

  int64_t numel;
  float *out = poly_instance_buf_data(inst, out_idx, &numel);
  ASSERT_INT_EQ(numel, 1 * 8 * 32);

  /* Verify output is finite and not all zero */
  int all_zero = 1;
  for (int64_t i = 0; i < numel; i++) {
    ASSERT_TRUE(isfinite(out[i]));
    if (fabsf(out[i]) > 1e-10f) all_zero = 0;
  }
  ASSERT_FALSE(all_zero);

  poly_instance_free(inst);
  PASS();
}

/* ── GPT-2 training: loss decreases ─────────────────────────────── */

TEST(hf, gpt2_training_loss_decreases) {
  GPT2Config cfg = {
    .vocab_size = 32,
    .n_embd = 16,
    .n_head = 2,
    .n_layer = 1,
    .max_seq_len = 8,
    .norm_eps = 1e-5f
  };

  PolyInstance *inst = poly_gpt2_build(&cfg, 1);
  ASSERT_NOT_NULL(inst);

  /* Initialize weights with small random values */
  int np = poly_instance_param_count(inst);
  for (int i = 0; i < np; i++) {
    int64_t numel;
    float *data = poly_instance_param_data(inst, i, &numel);
    const char *name = poly_instance_param_name(inst, i);
    if (strstr(name, "ln_") && strstr(name, "weight")) {
      for (int64_t j = 0; j < numel; j++) data[j] = 1.0f;
    } else if (strstr(name, "bias")) {
      for (int64_t j = 0; j < numel; j++) data[j] = 0.0f;
    } else {
      for (int64_t j = 0; j < numel; j++)
        data[j] = 0.02f * ((float)((j * 7 + 13) % 100) / 100.0f - 0.5f);
    }
  }

  /* Set input tokens and positions */
  int nb = poly_instance_buf_count(inst);
  for (int i = 0; i < nb; i++) {
    const char *name = poly_instance_buf_name(inst, i);
    int64_t numel;
    float *data = poly_instance_buf_data(inst, i, &numel);
    if (strcmp(name, "x") == 0) {
      for (int64_t j = 0; j < numel; j++) data[j] = (float)(j % 4);
    } else if (strcmp(name, "positions") == 0) {
      for (int64_t j = 0; j < numel; j++) data[j] = (float)j;
    } else if (strcmp(name, "arange") == 0) {
      for (int64_t j = 0; j < numel; j++) data[j] = (float)j;
    }
  }

  /* Configure Adam optimizer */
  int ret = poly_instance_set_optimizer(inst, POLY_OPTIM_ADAM,
                                         0.001f, 0.9f, 0.999f, 1e-8f, 0.0f);
  ASSERT_INT_EQ(ret, 0);

  /* Train for 5 steps */
  float losses[5];
  for (int step = 0; step < 5; step++) {
    ret = poly_instance_train_step(inst, NULL, 0, &losses[step]);
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

  poly_instance_free(inst);
  PASS();
}

/* ── Unsupported model type ──────────────────────────────────────── */

TEST(hf, hf_load_unsupported_type) {
  const char *config = "{\"model_type\":\"llama\",\"vocab_size\":100}";
  PolyInstance *inst = poly_hf_load(config, (int)strlen(config),
                                     NULL, NULL, 0, 1, 64);
  ASSERT_TRUE(inst == NULL);
  PASS();
}
