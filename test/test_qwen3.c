/*
 * test_qwen3.c -- Qwen3 0.6B GGUF end-to-end tests
 *
 * Optional: requires GGUF model file. Set POLY_QWEN3_GGUF env var.
 *
 * Run: POLY_QWEN3_GGUF=/path/to/Qwen3-0.6B-Q8_0.gguf ./build/polygrad_test qwen3
 */

#include "test_harness.h"
#include "../src/models/qwen3.h"
#include "../src/codegen.h"
#include "../src/loaders/gguf_decode.h"
#include "../src/instance.h"
#include "../src/engine/schedule.h"
#include "../src/tokenizer.h"
#include <string.h>

/* GGUF file loading */

#define QWEN3_GGUF_EXPECTED_LEN 639447744LL
#define QWEN3_GGUF_EXPECTED_CRC32 0xa014a8efu

static const char *find_gguf_path(void) {
  const char *env = getenv("POLY_QWEN3_GGUF");
  if (env && env[0]) return env;
  return NULL;
}

static uint8_t *g_gguf_data = NULL;
static int64_t g_gguf_len = 0;
static PolyGgufDecoded *g_gguf = NULL;
static int g_gguf_state = 0; /* 0 unknown, 1 ok, 2 skip, 3 error */
static char g_gguf_error[256];

static uint32_t crc32_bytes(const uint8_t *data, int64_t len) {
  static uint32_t table[256];
  static int table_ready = 0;
  if (!table_ready) {
    for (uint32_t i = 0; i < 256; i++) {
      uint32_t c = i;
      for (int j = 0; j < 8; j++)
        c = (c & 1) ? (0xedb88320u ^ (c >> 1)) : (c >> 1);
      table[i] = c;
    }
    table_ready = 1;
  }

  uint32_t crc = 0xffffffffu;
  for (int64_t i = 0; i < len; i++)
    crc = table[(crc ^ data[i]) & 0xffu] ^ (crc >> 8);
  return crc ^ 0xffffffffu;
}

static int ensure_gguf(void) {
  if (g_gguf) return 1;
  if (g_gguf_state == 2) return 0;
  if (g_gguf_state == 3) return -1;

  const char *path = find_gguf_path();
  if (!path) {
    g_gguf_state = 2;
    return 0;
  }

  FILE *f = fopen(path, "rb");
  if (!f) {
    snprintf(g_gguf_error, sizeof(g_gguf_error), "failed to open %s", path);
    g_gguf_state = 3;
    return -1;
  }

  fseek(f, 0, SEEK_END);
  g_gguf_len = ftell(f);
  fseek(f, 0, SEEK_SET);
  g_gguf_data = malloc(g_gguf_len);
  if (fread(g_gguf_data, 1, g_gguf_len, f) != (size_t)g_gguf_len) {
    free(g_gguf_data);
    fclose(f);
    snprintf(g_gguf_error, sizeof(g_gguf_error), "failed to read %s", path);
    g_gguf_state = 3;
    return -1;
  }
  fclose(f);

  uint32_t crc = crc32_bytes(g_gguf_data, g_gguf_len);
  if (g_gguf_len != QWEN3_GGUF_EXPECTED_LEN || crc != QWEN3_GGUF_EXPECTED_CRC32) {
    snprintf(
        g_gguf_error, sizeof(g_gguf_error),
        "unexpected GGUF fixture len=%lld crc32=0x%08x", (long long)g_gguf_len, crc
    );
    free(g_gguf_data);
    g_gguf_data = NULL;
    g_gguf_state = 3;
    return -1;
  }

  if (poly_gguf_decode(g_gguf_data, g_gguf_len, &g_gguf) != 0 || !g_gguf) {
    free(g_gguf_data);
    g_gguf_data = NULL;
    snprintf(g_gguf_error, sizeof(g_gguf_error), "GGUF decode failed");
    g_gguf_state = 3;
    return -1;
  }
  g_gguf_state = 1;
  return 1;
}

#define SKIP_IF_NO_GGUF()                                                                          \
  do {                                                                                             \
    int _gguf_ok = ensure_gguf();                                                                  \
    if (_gguf_ok < 0) FAIL("%s", g_gguf_error);                                                    \
    if (!_gguf_ok) {                                                                               \
      fprintf(stderr, "    (skipped: POLY_QWEN3_GGUF not set)\n");                                 \
      PASS();                                                                                      \
    }                                                                                              \
  } while (0)

/* Helper: find I/O buffers */

static float *find_buf(PolyInstance *inst, const char *name, int64_t *numel) {
  int nb = poly_instance_buf_count(inst);
  for (int b = 0; b < nb; b++) {
    if (strcmp(poly_instance_buf_name(inst, b), name) == 0)
      return poly_instance_buf_data(inst, b, numel);
  }
  return NULL;
}

static int argmax_f32(const float *data, int n) {
  int best = 0;
  float best_val = data[0];
  for (int i = 1; i < n; i++)
    if (data[i] > best_val) {
      best_val = data[i];
      best = i;
    }
  return best;
}

/* Tests */

TEST(qwen3, gguf_decode) {
  SKIP_IF_NO_GGUF();
  ASSERT_NOT_NULL(g_gguf);
  ASSERT_TRUE(g_gguf->n_tensors > 300);
  ASSERT_NOT_NULL(g_gguf->arch);
  ASSERT_STR_EQ(g_gguf->arch, "qwen3");
  PASS();
}

TEST(qwen3, config_from_gguf) {
  SKIP_IF_NO_GGUF();
  int dim = poly_gguf_kv_int(g_gguf, "qwen3.embedding_length", -1);
  int heads = poly_gguf_kv_int(g_gguf, "qwen3.attention.head_count", -1);
  int layers = poly_gguf_kv_int(g_gguf, "qwen3.block_count", -1);
  ASSERT_INT_EQ(dim, 1024);
  ASSERT_INT_EQ(heads, 16);
  ASSERT_INT_EQ(layers, 28);
  PASS();
}

TEST(qwen3, model_build_and_load) {
  SKIP_IF_NO_GGUF();
  PolyInstance *inst = poly_qwen3_from_gguf_decoded(g_gguf, 1, 25);
  ASSERT_NOT_NULL(inst);

  /* Check buffer count: 4 I/O (x, output, rope_cos, rope_sin) + 310 params */
  int nb = poly_instance_buf_count(inst);
  ASSERT_TRUE(nb >= 310);

  /* Check I/O buffers exist */
  int64_t numel;
  ASSERT_NOT_NULL(find_buf(inst, "x", &numel));
  ASSERT_INT_EQ(numel, 25); /* batch=1 * seq_len=25 */
  ASSERT_NOT_NULL(find_buf(inst, "output", &numel));
  ASSERT_INT_EQ(numel, 25 * 151936); /* batch * seq_len * vocab */

  poly_instance_free(inst);
  PASS();
}

TEST(qwen3, forward_cpu) {
  SKIP_IF_NO_GGUF();
  PolyInstance *inst = poly_qwen3_from_gguf_decoded(g_gguf, 1, 25);
  ASSERT_NOT_NULL(inst);

  /* Fill input with prompt "The capital of France is" */
  int64_t numel;
  float *x = find_buf(inst, "x", &numel);
  ASSERT_NOT_NULL(x);
  memset(x, 0, numel * sizeof(float));
  float prompt[] = {785, 6722, 315, 9625, 374};
  memcpy(x, prompt, sizeof(prompt));

  int rc = poly_instance_forward(inst, NULL, 0);
  ASSERT_INT_EQ(rc, 0);

  float *out = find_buf(inst, "output", &numel);
  ASSERT_NOT_NULL(out);

  /* Check logits at last prompt position (pos 4) are finite */
  int V = 151936;
  float *logits = out + 4 * V;
  int n_nan = 0, n_inf = 0;
  for (int i = 0; i < V; i++) {
    if (isnan(logits[i])) n_nan++;
    if (isinf(logits[i])) n_inf++;
  }
  ASSERT_INT_EQ(n_nan, 0);
  ASSERT_INT_EQ(n_inf, 0);

  /* Greedy argmax should be "Paris" (id=12095) */
  int next_id = argmax_f32(logits, V);
  ASSERT_INT_EQ(next_id, 12095);

  poly_instance_free(inst);
  PASS();
}

#ifdef POLY_HAS_CUDA
TEST(qwen3, forward_cuda) {
  SKIP_IF_NO_GGUF();
  if (!poly_cuda_available()) {
    fprintf(stderr, "    (skipped: no CUDA GPU)\n");
    PASS();
  }

  PolyInstance *inst = poly_qwen3_from_gguf_decoded(g_gguf, 1, 25);
  ASSERT_NOT_NULL(inst);
  poly_instance_set_device(inst, POLY_DEVICE_CUDA);

  int64_t numel;
  float *x = find_buf(inst, "x", &numel);
  ASSERT_NOT_NULL(x);
  memset(x, 0, numel * sizeof(float));
  float prompt[] = {785, 6722, 315, 9625, 374};
  memcpy(x, prompt, sizeof(prompt));

  int rc = poly_instance_forward(inst, NULL, 0);
  ASSERT_INT_EQ(rc, 0);

  float *out = find_buf(inst, "output", &numel);
  ASSERT_NOT_NULL(out);

  /* Check logits at last prompt position are finite */
  int V = 151936;
  float *logits = out + 4 * V;
  int n_nan = 0;
  for (int i = 0; i < V; i++)
    if (isnan(logits[i])) n_nan++;
  ASSERT_INT_EQ(n_nan, 0);

  /* Same greedy argmax as CPU */
  int next_id = argmax_f32(logits, V);
  ASSERT_INT_EQ(next_id, 12095);

  poly_instance_free(inst);
  PASS();
}
#endif /* POLY_HAS_CUDA */

TEST(qwen3, tokenizer_from_gguf) {
  SKIP_IF_NO_GGUF();
  PolyTokenizer *tok = poly_tokenizer_from_gguf(g_gguf);
  ASSERT_NOT_NULL(tok);
  ASSERT_INT_EQ(poly_tokenizer_vocab_size(tok), 151936);

  /* Encode "The capital of France is" */
  int ids[64];
  int n = poly_tokenize(tok, "The capital of France is", ids, 64);
  ASSERT_INT_EQ(n, 5);
  ASSERT_INT_EQ(ids[0], 785);
  ASSERT_INT_EQ(ids[1], 6722);
  ASSERT_INT_EQ(ids[2], 315);
  ASSERT_INT_EQ(ids[3], 9625);
  ASSERT_INT_EQ(ids[4], 374);

  /* Decode round-trip */
  char text[256];
  poly_detokenize(tok, ids, n, text, sizeof(text));
  ASSERT_STR_EQ(text, "The capital of France is");

  poly_tokenizer_free(tok);
  PASS();
}
