/*
 * test_f16.c -- Float16 and BFloat16 end-to-end tests (TDD)
 *
 * These tests verify that float16/bfloat16 operations compile and execute
 * correctly through the C renderer. The strategy matches tinygrad:
 *   - float16 renders as __fp16 (via type_map)
 *   - bfloat16 ALU ops are emulated via f32 promotion
 *   - bfloat16 casts use manual bitwise manipulation with raw u16 storage on CPU C
 */

#include "test_harness.h"
#include "../src/polygrad.h"
#include "../src/frontend.h"
#include "../src/codegen.h"
#include "../src/engine/schedule.h"

/* Helper: f32 <-> f16 bit conversion (IEEE 754 half-precision) */

static uint16_t f32_to_f16_bits(float f) {
  uint32_t u;
  memcpy(&u, &f, sizeof(u));
  uint32_t sign = (u >> 16) & 0x8000;
  int32_t exp = (int32_t)(((u >> 23) & 0xFF)) - 127 + 15;
  uint32_t frac = (u >> 13) & 0x3FF;
  if (exp <= 0) return (uint16_t)sign;
  if (exp >= 31) return (uint16_t)(sign | 0x7C00);
  return (uint16_t)(sign | ((uint32_t)exp << 10) | frac);
}

static float f16_bits_to_f32(uint16_t h) {
  uint32_t sign = ((uint32_t)h & 0x8000) << 16;
  uint32_t exp = (h >> 10) & 0x1F;
  uint32_t frac = h & 0x3FF;
  uint32_t u;
  if (exp == 0) {
    u = sign; /* flush denorms to zero for simplicity */
  } else if (exp == 31) {
    u = sign | 0x7F800000 | (frac << 13);
  } else {
    u = sign | (((uint32_t)exp + 127 - 15) << 23) | (frac << 13);
  }
  float result;
  memcpy(&result, &u, sizeof(result));
  return result;
}

/* Helper: f32 <-> bf16 bit conversion */

static uint16_t f32_to_bf16_bits(float f) {
  uint32_t u;
  memcpy(&u, &f, sizeof(u));
  return (uint16_t)(u >> 16);
}

static float bf16_bits_to_f32(uint16_t b) {
  uint32_t u = (uint32_t)b << 16;
  float result;
  memcpy(&result, &u, sizeof(result));
  return result;
}

/* dtype classification */

TEST(f16, dtype_float16_basics) {
  ASSERT_INT_EQ(POLY_FLOAT16.bitsize, 16);
  ASSERT_INT_EQ(poly_dtype_itemsize(POLY_FLOAT16), 2);
  ASSERT_TRUE(poly_dtype_is_float(POLY_FLOAT16));
  ASSERT_FALSE(poly_dtype_is_int(POLY_FLOAT16));
  PASS();
}

TEST(f16, dtype_bfloat16_basics) {
  ASSERT_INT_EQ(POLY_BFLOAT16.bitsize, 16);
  ASSERT_INT_EQ(poly_dtype_itemsize(POLY_BFLOAT16), 2);
  ASSERT_TRUE(poly_dtype_is_float(POLY_BFLOAT16));
  ASSERT_FALSE(poly_dtype_is_int(POLY_BFLOAT16));
  PASS();
}

/* f32 -> f16 cast + realize */

TEST(f16, cast_f32_to_f16_e2e) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *in = poly_buffer_f32(ctx, 4);
  PolyUOp *casted = poly_cast(ctx, in, POLY_FLOAT16);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT16, 4);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, casted));

  float in_data[] = {1.0f, 2.0f, -0.5f, 0.0f};
  uint16_t out_data[4] = {0};

  PolyTestBufferView binds[] = {POLY_TEST_HOST_VIEW(in, in_data), POLY_TEST_HOST_VIEW(out, out_data)};
  int rc = poly_test_realize_buffer_views(ctx, sink, binds, 2);
  ASSERT_INT_EQ(rc, 0);

  ASSERT_INT_EQ(out_data[0], f32_to_f16_bits(1.0f));
  ASSERT_INT_EQ(out_data[1], f32_to_f16_bits(2.0f));
  ASSERT_INT_EQ(out_data[2], f32_to_f16_bits(-0.5f));
  ASSERT_INT_EQ(out_data[3], f32_to_f16_bits(0.0f));

  poly_ctx_destroy(ctx);
  PASS();
}

/* Pinned native float16 conversion uses IEEE half pack/unpack
 * (tinygrad/dtype.py:280-282), including subnormals and ties-to-even. */
TEST(f16, cast_f32_to_f16_ieee_edges_e2e) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *in = poly_buffer_f32(ctx, 9);
  PolyUOp *casted = poly_cast(ctx, in, POLY_FLOAT16);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT16, 9);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, casted));

  float in_data[] = {
      0x1p-24f,
      -0x1p-24f,
      0x1.ff8p-15f,
      0x1p-14f,
      1.0f + 0x1p-11f,
      1.0f + 0x3p-11f,
      INFINITY,
      -INFINITY,
      NAN,
  };
  uint16_t out_data[9] = {0};
  PolyTestBufferView binds[] = {
      POLY_TEST_HOST_VIEW(in, in_data),
      POLY_TEST_HOST_VIEW(out, out_data),
  };
  ASSERT_INT_EQ(poly_test_realize_buffer_views(ctx, sink, binds, 2), 0);

  const uint16_t native_expected[] = {
      0x0001u, 0x8001u, 0x03ffu, 0x0400u, 0x3c00u, 0x3c02u, 0x7c00u, 0xfc00u,
  };
  /* Pinned PythonRenderer excludes float16 on Python 3.11 and applies
   * pm_float_decomp, while the other tested renderers keep native IEEE half
   * (`runtime/ops_python.py:203-223`). */
  const uint16_t interp_expected[] = {
      0x0000u, 0x8000u, 0x0000u, 0x0400u, 0x3c00u, 0x3c02u, 0x7c00u, 0xfc00u,
  };
  const uint16_t *expected =
      poly_ctx_get_preferred_device(ctx) == POLY_DEVICE_INTERP ? interp_expected : native_expected;
  for (int i = 0; i < 8; i++)
    ASSERT_INT_EQ(out_data[i], expected[i]);
  ASSERT_TRUE((out_data[8] & 0x7c00u) == 0x7c00u && (out_data[8] & 0x03ffu) != 0);

  poly_ctx_destroy(ctx);
  PASS();
}

/* f16 -> f32 cast + realize */

TEST(f16, cast_f16_to_f32_e2e) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *in = poly_buffer(ctx, POLY_FLOAT16, 3);
  PolyUOp *casted = poly_cast(ctx, in, POLY_FLOAT32);
  PolyUOp *out = poly_buffer_f32(ctx, 3);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, casted));

  uint16_t in_data[] = {f32_to_f16_bits(1.0f), f32_to_f16_bits(3.5f), f32_to_f16_bits(-2.0f)};
  float out_data[3] = {0};

  PolyTestBufferView binds[] = {POLY_TEST_HOST_VIEW(in, in_data), POLY_TEST_HOST_VIEW(out, out_data)};
  int rc = poly_test_realize_buffer_views(ctx, sink, binds, 2);
  ASSERT_INT_EQ(rc, 0);

  ASSERT_FLOAT_EQ(out_data[0], 1.0f, 1e-3f);
  ASSERT_FLOAT_EQ(out_data[1], 3.5f, 1e-3f);
  ASSERT_FLOAT_EQ(out_data[2], -2.0f, 1e-3f);

  poly_ctx_destroy(ctx);
  PASS();
}

/* f16 add e2e */

TEST(f16, add_f16_e2e) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT16, 4);
  PolyUOp *b = poly_buffer(ctx, POLY_FLOAT16, 4);
  PolyUOp *sum = poly_alu2(ctx, POLY_OP_ADD, a, b);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT16, 4);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, sum));

  uint16_t a_data[] = {
      f32_to_f16_bits(1.0f), f32_to_f16_bits(2.0f), f32_to_f16_bits(3.0f), f32_to_f16_bits(-1.0f)
  };
  uint16_t b_data[] = {
      f32_to_f16_bits(10.0f), f32_to_f16_bits(20.0f), f32_to_f16_bits(30.0f), f32_to_f16_bits(1.0f)
  };
  uint16_t out_data[4] = {0};

  PolyTestBufferView binds[] = {
      POLY_TEST_HOST_VIEW(a, a_data), POLY_TEST_HOST_VIEW(b, b_data), POLY_TEST_HOST_VIEW(out, out_data)
  };
  int rc = poly_test_realize_buffer_views(ctx, sink, binds, 3);
  ASSERT_INT_EQ(rc, 0);

  ASSERT_FLOAT_EQ(f16_bits_to_f32(out_data[0]), 11.0f, 0.1f);
  ASSERT_FLOAT_EQ(f16_bits_to_f32(out_data[1]), 22.0f, 0.1f);
  ASSERT_FLOAT_EQ(f16_bits_to_f32(out_data[2]), 33.0f, 0.1f);
  ASSERT_FLOAT_EQ(f16_bits_to_f32(out_data[3]), 0.0f, 0.1f);

  poly_ctx_destroy(ctx);
  PASS();
}

/* f16 mul e2e */

TEST(f16, mul_f16_e2e) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT16, 3);
  PolyUOp *b = poly_buffer(ctx, POLY_FLOAT16, 3);
  PolyUOp *prod = poly_alu2(ctx, POLY_OP_MUL, a, b);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT16, 3);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, prod));

  uint16_t a_data[] = {f32_to_f16_bits(2.0f), f32_to_f16_bits(3.0f), f32_to_f16_bits(-4.0f)};
  uint16_t b_data[] = {f32_to_f16_bits(5.0f), f32_to_f16_bits(0.5f), f32_to_f16_bits(2.0f)};
  uint16_t out_data[3] = {0};

  PolyTestBufferView binds[] = {
      POLY_TEST_HOST_VIEW(a, a_data), POLY_TEST_HOST_VIEW(b, b_data), POLY_TEST_HOST_VIEW(out, out_data)
  };
  int rc = poly_test_realize_buffer_views(ctx, sink, binds, 3);
  ASSERT_INT_EQ(rc, 0);

  ASSERT_FLOAT_EQ(f16_bits_to_f32(out_data[0]), 10.0f, 0.1f);
  ASSERT_FLOAT_EQ(f16_bits_to_f32(out_data[1]), 1.5f, 0.1f);
  ASSERT_FLOAT_EQ(f16_bits_to_f32(out_data[2]), -8.0f, 0.1f);

  poly_ctx_destroy(ctx);
  PASS();
}

/* f16 neg e2e */

TEST(f16, neg_f16_e2e) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT16, 3);
  PolyUOp *neg = poly_alu1(ctx, POLY_OP_NEG, a);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT16, 3);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, neg));

  uint16_t a_data[] = {f32_to_f16_bits(1.0f), f32_to_f16_bits(-2.0f), f32_to_f16_bits(0.0f)};
  uint16_t out_data[3] = {0};

  PolyTestBufferView binds[] = {POLY_TEST_HOST_VIEW(a, a_data), POLY_TEST_HOST_VIEW(out, out_data)};
  int rc = poly_test_realize_buffer_views(ctx, sink, binds, 2);
  ASSERT_INT_EQ(rc, 0);

  ASSERT_FLOAT_EQ(f16_bits_to_f32(out_data[0]), -1.0f, 0.01f);
  ASSERT_FLOAT_EQ(f16_bits_to_f32(out_data[1]), 2.0f, 0.01f);
  ASSERT_FLOAT_EQ(f16_bits_to_f32(out_data[2]), 0.0f, 0.01f);

  poly_ctx_destroy(ctx);
  PASS();
}

/* f16 constant rendering */

TEST(f16, const_f16_e2e) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT16, 3);
  PolyUOp *c = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT16, poly_arg_float(10.0f));
  PolyUOp *sum = poly_alu2(ctx, POLY_OP_ADD, a, c);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT16, 3);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, sum));

  uint16_t a_data[] = {f32_to_f16_bits(1.0f), f32_to_f16_bits(2.0f), f32_to_f16_bits(3.0f)};
  uint16_t out_data[3] = {0};

  PolyTestBufferView binds[] = {POLY_TEST_HOST_VIEW(a, a_data), POLY_TEST_HOST_VIEW(out, out_data)};
  int rc = poly_test_realize_buffer_views(ctx, sink, binds, 2);
  ASSERT_INT_EQ(rc, 0);

  ASSERT_FLOAT_EQ(f16_bits_to_f32(out_data[0]), 11.0f, 0.1f);
  ASSERT_FLOAT_EQ(f16_bits_to_f32(out_data[1]), 12.0f, 0.1f);
  ASSERT_FLOAT_EQ(f16_bits_to_f32(out_data[2]), 13.0f, 0.1f);

  poly_ctx_destroy(ctx);
  PASS();
}

/* mixed precision: f16 -> f32 compute -> f32 output */

TEST(f16, mixed_f16_to_f32_chain_e2e) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *in = poly_buffer(ctx, POLY_FLOAT16, 3);
  PolyUOp *as_f32 = poly_cast(ctx, in, POLY_FLOAT32);
  PolyUOp *c = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(100.0f));
  PolyUOp *sum = poly_alu2(ctx, POLY_OP_ADD, as_f32, c);
  PolyUOp *out = poly_buffer_f32(ctx, 3);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, sum));

  uint16_t in_data[] = {f32_to_f16_bits(1.0f), f32_to_f16_bits(2.0f), f32_to_f16_bits(3.0f)};
  float out_data[3] = {0};

  PolyTestBufferView binds[] = {POLY_TEST_HOST_VIEW(in, in_data), POLY_TEST_HOST_VIEW(out, out_data)};
  int rc = poly_test_realize_buffer_views(ctx, sink, binds, 2);
  ASSERT_INT_EQ(rc, 0);

  ASSERT_FLOAT_EQ(out_data[0], 101.0f, 0.1f);
  ASSERT_FLOAT_EQ(out_data[1], 102.0f, 0.1f);
  ASSERT_FLOAT_EQ(out_data[2], 103.0f, 0.1f);

  poly_ctx_destroy(ctx);
  PASS();
}

/* f64 -> f16 cast (should go via f32 intermediate) */

TEST(f16, cast_f64_to_f16_e2e) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *in = poly_buffer_f64(ctx, 2);
  PolyUOp *casted = poly_cast(ctx, in, POLY_FLOAT16);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT16, 2);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, casted));

  double in_data[] = {1.5, -2.5};
  uint16_t out_data[2] = {0};

  PolyTestBufferView binds[] = {POLY_TEST_HOST_VIEW(in, in_data), POLY_TEST_HOST_VIEW(out, out_data)};
  int rc = poly_test_realize_buffer_views(ctx, sink, binds, 2);
  ASSERT_INT_EQ(rc, 0);

  ASSERT_FLOAT_EQ(f16_bits_to_f32(out_data[0]), 1.5f, 0.01f);
  ASSERT_FLOAT_EQ(f16_bits_to_f32(out_data[1]), -2.5f, 0.01f);

  poly_ctx_destroy(ctx);
  PASS();
}

/* bf16 cast f32 -> bf16 e2e */

TEST(f16, cast_f32_to_bf16_e2e) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *in = poly_buffer_f32(ctx, 3);
  PolyUOp *casted = poly_cast(ctx, in, POLY_BFLOAT16);
  PolyUOp *out = poly_buffer(ctx, POLY_BFLOAT16, 3);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, casted));

  float in_data[] = {1.0f, -2.0f, 0.5f};
  uint16_t out_data[3] = {0};

  PolyTestBufferView binds[] = {POLY_TEST_HOST_VIEW(in, in_data), POLY_TEST_HOST_VIEW(out, out_data)};
  int rc = poly_test_realize_buffer_views(ctx, sink, binds, 2);
  ASSERT_INT_EQ(rc, 0);

  ASSERT_FLOAT_EQ(bf16_bits_to_f32(out_data[0]), 1.0f, 0.01f);
  ASSERT_FLOAT_EQ(bf16_bits_to_f32(out_data[1]), -2.0f, 0.01f);
  ASSERT_FLOAT_EQ(bf16_bits_to_f32(out_data[2]), 0.5f, 0.01f);

  poly_ctx_destroy(ctx);
  PASS();
}

/* Pinned tinygrad f2f_store decomposes vector results lane-by-lane before
 * BF16 storage (uop/decompositions.py:423-429, 533-562). Four elements force
 * the CPU codegen upcast/STACK path that scalar and three-lane cases miss. */
TEST(f16, cast_f32_to_bf16_vector4_e2e) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *in = poly_buffer_f32(ctx, 4);
  PolyUOp *casted = poly_cast(ctx, in, POLY_BFLOAT16);
  PolyUOp *out = poly_buffer(ctx, POLY_BFLOAT16, 4);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, casted));

  float in_data[] = {1.0f, -2.0f, 0.5f, 3.25f};
  uint16_t out_data[4] = {0};
  PolyTestBufferView binds[] = {POLY_TEST_HOST_VIEW(in, in_data), POLY_TEST_HOST_VIEW(out, out_data)};
  ASSERT_INT_EQ(poly_test_realize_buffer_views(ctx, sink, binds, 2), 0);

  for (int i = 0; i < 4; i++)
    ASSERT_INT_EQ(out_data[i], f32_to_bf16_bits(in_data[i]));

  poly_ctx_destroy(ctx);
  PASS();
}

/* bf16 cast bf16 -> f32 e2e */

TEST(f16, cast_bf16_to_f32_e2e) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *in = poly_buffer(ctx, POLY_BFLOAT16, 3);
  PolyUOp *casted = poly_cast(ctx, in, POLY_FLOAT32);
  PolyUOp *out = poly_buffer_f32(ctx, 3);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, casted));

  uint16_t in_data[] = {f32_to_bf16_bits(1.0f), f32_to_bf16_bits(-3.0f), f32_to_bf16_bits(0.25f)};
  float out_data[3] = {0};

  PolyTestBufferView binds[] = {POLY_TEST_HOST_VIEW(in, in_data), POLY_TEST_HOST_VIEW(out, out_data)};
  int rc = poly_test_realize_buffer_views(ctx, sink, binds, 2);
  ASSERT_INT_EQ(rc, 0);

  ASSERT_FLOAT_EQ(out_data[0], 1.0f, 0.01f);
  ASSERT_FLOAT_EQ(out_data[1], -3.0f, 0.01f);
  ASSERT_FLOAT_EQ(out_data[2], 0.25f, 0.01f);

  poly_ctx_destroy(ctx);
  PASS();
}

/* Pinned pm_float_decomp keeps BITCAST as raw same-width reinterpretation for
 * every 16-bit destination/source, not only uint16
 * (tinygrad/uop/decompositions.py:541-545). */
TEST(f16, bitcast_bf16_int16_roundtrip_e2e) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *bf16_in = poly_buffer(ctx, POLY_BFLOAT16, 5);
  PolyUOp *int16_out = poly_buffer(ctx, POLY_INT16, 5);
  PolyUOp *to_i16 = poly_uop1(ctx, POLY_OP_BITCAST, POLY_INT16, bf16_in, poly_arg_none());
  PolyUOp *first_sink = poly_sink1(ctx, poly_store_val(ctx, int16_out, to_i16));
  uint16_t bf16_data[] = {
      f32_to_bf16_bits(1.0f), f32_to_bf16_bits(-2.0f), f32_to_bf16_bits(0.5f), 0x0001u, 0x8001u,
  };
  int16_t int16_data[5] = {0};
  PolyTestBufferView first_binds[] = {
      POLY_TEST_HOST_VIEW(bf16_in, bf16_data),
      POLY_TEST_HOST_VIEW(int16_out, int16_data),
  };
  ASSERT_INT_EQ(poly_test_realize_buffer_views(ctx, first_sink, first_binds, 2), 0);
  for (int i = 0; i < 5; i++)
    ASSERT_INT_EQ((uint16_t)int16_data[i], bf16_data[i]);

  PolyUOp *int16_in = poly_buffer(ctx, POLY_INT16, 5);
  PolyUOp *bf16_out = poly_buffer(ctx, POLY_BFLOAT16, 5);
  PolyUOp *to_bf16 = poly_uop1(ctx, POLY_OP_BITCAST, POLY_BFLOAT16, int16_in, poly_arg_none());
  PolyUOp *second_sink = poly_sink1(ctx, poly_store_val(ctx, bf16_out, to_bf16));
  int16_t reverse_input[5];
  for (int i = 0; i < 5; i++)
    reverse_input[i] = (int16_t)bf16_data[i];
  uint16_t roundtrip[5] = {0};
  PolyTestBufferView second_binds[] = {
      POLY_TEST_HOST_VIEW(int16_in, reverse_input),
      POLY_TEST_HOST_VIEW(bf16_out, roundtrip),
  };
  ASSERT_INT_EQ(poly_test_realize_buffer_views(ctx, second_sink, second_binds, 2), 0);
  for (int i = 0; i < 3; i++)
    ASSERT_INT_EQ(roundtrip[i], bf16_data[i]);
  /* Pinned pm_float_decomp applies only when BF16 is unsupported. CUDA SM80+
   * and HIP keep the native same-width BITCAST and preserve the raw words;
   * emulated renderers numerically convert the ordinary untagged STORE and
   * flush BF16 subnormals to signed zero
   * (uop/decompositions.py:533-562, renderer/cstyle.py:463). */
  /* Pinned PythonRenderer supports BF16 natively and its BITCAST operates on
   * the exact uint16 storage words (`ops_python.py:125`, `uop/ops.py:1199-1208`). */
  bool native_bf16 = poly_ctx_get_preferred_device(ctx) == POLY_DEVICE_INTERP;
#ifdef POLY_HAS_CUDA
  native_bf16 |=
      poly_ctx_get_preferred_device(ctx) == POLY_DEVICE_CUDA && poly_cuda_arch_major() >= 8;
#endif
  native_bf16 |= poly_ctx_get_preferred_device(ctx) == POLY_DEVICE_HIP;
  ASSERT_INT_EQ(roundtrip[3], native_bf16 ? 0x0001u : 0x0000u);
  ASSERT_INT_EQ(roundtrip[4], native_bf16 ? 0x8001u : 0x8000u);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(f16, bitcast_bf16_float16_roundtrip_e2e) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *bf16_in = poly_buffer(ctx, POLY_BFLOAT16, 5);
  PolyUOp *f16_out = poly_buffer(ctx, POLY_FLOAT16, 5);
  PolyUOp *to_f16 = poly_uop1(ctx, POLY_OP_BITCAST, POLY_FLOAT16, bf16_in, poly_arg_none());
  PolyUOp *first_sink = poly_sink1(ctx, poly_store_val(ctx, f16_out, to_f16));
  uint16_t bf16_data[] = {
      f32_to_bf16_bits(1.0f), f32_to_bf16_bits(-2.0f), f32_to_bf16_bits(0.5f), 0x0001u, 0x8001u,
  };
  uint16_t f16_data[5] = {0};
  PolyTestBufferView first_binds[] = {
      POLY_TEST_HOST_VIEW(bf16_in, bf16_data),
      POLY_TEST_HOST_VIEW(f16_out, f16_data),
  };
  ASSERT_INT_EQ(poly_test_realize_buffer_views(ctx, first_sink, first_binds, 2), 0);
  for (int i = 0; i < 3; i++)
    ASSERT_INT_EQ(f16_data[i], bf16_data[i]);
  bool non_native_f16 = poly_ctx_get_preferred_device(ctx) == POLY_DEVICE_INTERP;
  ASSERT_INT_EQ(f16_data[3], non_native_f16 ? 0x0000u : bf16_data[3]);
  ASSERT_INT_EQ(f16_data[4], non_native_f16 ? 0x8000u : bf16_data[4]);

  PolyUOp *f16_in = poly_buffer(ctx, POLY_FLOAT16, 5);
  PolyUOp *bf16_out = poly_buffer(ctx, POLY_BFLOAT16, 5);
  PolyUOp *to_bf16 = poly_uop1(ctx, POLY_OP_BITCAST, POLY_BFLOAT16, f16_in, poly_arg_none());
  PolyUOp *second_sink = poly_sink1(ctx, poly_store_val(ctx, bf16_out, to_bf16));
  uint16_t reverse_input[5];
  memcpy(reverse_input, bf16_data, sizeof(reverse_input));
  uint16_t roundtrip[5] = {0};
  PolyTestBufferView second_binds[] = {
      POLY_TEST_HOST_VIEW(f16_in, reverse_input),
      POLY_TEST_HOST_VIEW(bf16_out, roundtrip),
  };
  ASSERT_INT_EQ(poly_test_realize_buffer_views(ctx, second_sink, second_binds, 2), 0);
  bool native_bf16 = poly_ctx_get_preferred_device(ctx) == POLY_DEVICE_INTERP;
#ifdef POLY_HAS_CUDA
  native_bf16 |=
      poly_ctx_get_preferred_device(ctx) == POLY_DEVICE_CUDA && poly_cuda_arch_major() >= 8;
#endif
  native_bf16 |= poly_ctx_get_preferred_device(ctx) == POLY_DEVICE_HIP;
  for (int i = 0; i < 3; i++)
    ASSERT_INT_EQ(roundtrip[i], bf16_data[i]);
  ASSERT_INT_EQ(roundtrip[3], native_bf16 ? 0x0001u : 0x0000u);
  ASSERT_INT_EQ(roundtrip[4], native_bf16 ? 0x8001u : 0x8000u);

  poly_ctx_destroy(ctx);
  PASS();
}

/* Pinned native same-width BITCAST preserves representable destination-half
 * NaN payload/signaling bits (uop/ops.py:1199-1208). These words are finite
 * BF16 inputs, so the assertion isolates the destination F16 encoder. */
TEST(f16, bitcast_bf16_to_float16_preserves_nan_payload_bits_e2e) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *bf16_in = poly_buffer(ctx, POLY_BFLOAT16, 6);
  PolyUOp *f16_out = poly_buffer(ctx, POLY_FLOAT16, 6);
  PolyUOp *to_f16 = poly_uop1(ctx, POLY_OP_BITCAST, POLY_FLOAT16, bf16_in, poly_arg_none());
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, f16_out, to_f16));
  uint16_t input[] = {0x7c01u, 0x7d55u, 0x7e55u, 0xfc01u, 0xfd55u, 0xfe55u};
  uint16_t output[6] = {0};
  PolyTestBufferView binds[] = {
      POLY_TEST_HOST_VIEW(bf16_in, input),
      POLY_TEST_HOST_VIEW(f16_out, output),
  };
  ASSERT_INT_EQ(poly_test_realize_buffer_views(ctx, sink, binds, 2), 0);
  const uint16_t interp_expected[] = {
      0x7e01u, 0x7f55u, 0x7e55u, 0xfe01u, 0xff55u, 0xfe55u,
  };
  const uint16_t *expected =
      poly_ctx_get_preferred_device(ctx) == POLY_DEVICE_INTERP ? interp_expected : input;
  for (int i = 0; i < 6; i++)
    ASSERT_INT_EQ(output[i], expected[i]);

  poly_ctx_destroy(ctx);
  PASS();
}

/* bf16 add (via f32 emulation) e2e */

TEST(f16, add_bf16_e2e) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer(ctx, POLY_BFLOAT16, 3);
  PolyUOp *b = poly_buffer(ctx, POLY_BFLOAT16, 3);
  PolyUOp *sum = poly_alu2(ctx, POLY_OP_ADD, a, b);
  PolyUOp *out = poly_buffer(ctx, POLY_BFLOAT16, 3);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, sum));

  uint16_t a_data[] = {f32_to_bf16_bits(1.0f), f32_to_bf16_bits(2.0f), f32_to_bf16_bits(-1.0f)};
  uint16_t b_data[] = {f32_to_bf16_bits(10.0f), f32_to_bf16_bits(20.0f), f32_to_bf16_bits(1.0f)};
  uint16_t out_data[3] = {0};

  PolyTestBufferView binds[] = {
      POLY_TEST_HOST_VIEW(a, a_data), POLY_TEST_HOST_VIEW(b, b_data),
      POLY_TEST_HOST_VIEW(out, out_data)};
  int rc = poly_test_realize_buffer_views(ctx, sink, binds, 3);
  ASSERT_INT_EQ(rc, 0);

  ASSERT_FLOAT_EQ(bf16_bits_to_f32(out_data[0]), 11.0f, 0.2f);
  ASSERT_FLOAT_EQ(bf16_bits_to_f32(out_data[1]), 22.0f, 0.2f);
  ASSERT_FLOAT_EQ(bf16_bits_to_f32(out_data[2]), 0.0f, 0.2f);

  poly_ctx_destroy(ctx);
  PASS();
}

/* bf16 mul (via f32 emulation) e2e */

TEST(f16, mul_bf16_e2e) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer(ctx, POLY_BFLOAT16, 3);
  PolyUOp *b = poly_buffer(ctx, POLY_BFLOAT16, 3);
  PolyUOp *prod = poly_alu2(ctx, POLY_OP_MUL, a, b);
  PolyUOp *out = poly_buffer(ctx, POLY_BFLOAT16, 3);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, prod));

  uint16_t a_data[] = {f32_to_bf16_bits(2.0f), f32_to_bf16_bits(3.0f), f32_to_bf16_bits(-4.0f)};
  uint16_t b_data[] = {f32_to_bf16_bits(5.0f), f32_to_bf16_bits(0.5f), f32_to_bf16_bits(2.0f)};
  uint16_t out_data[3] = {0};

  PolyTestBufferView binds[] = {
      POLY_TEST_HOST_VIEW(a, a_data), POLY_TEST_HOST_VIEW(b, b_data),
      POLY_TEST_HOST_VIEW(out, out_data)};
  int rc = poly_test_realize_buffer_views(ctx, sink, binds, 3);
  ASSERT_INT_EQ(rc, 0);

  ASSERT_FLOAT_EQ(bf16_bits_to_f32(out_data[0]), 10.0f, 0.2f);
  ASSERT_FLOAT_EQ(bf16_bits_to_f32(out_data[1]), 1.5f, 0.2f);
  ASSERT_FLOAT_EQ(bf16_bits_to_f32(out_data[2]), -8.0f, 0.2f);

  poly_ctx_destroy(ctx);
  PASS();
}

/* Pinned CUDA SM80+ uses native BF16 and HIPRenderer.extra_matcher inserts a
 * BF16 cast after each ordinary ALU (renderer/cstyle.py:472-486,515-520).
 * Unsupported-renderer emulation rounds once when materializing storage. */
TEST(f16, bf16_chain_matches_native_or_emulated_renderer_rounding) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer(ctx, POLY_BFLOAT16, 1);
  PolyUOp *b = poly_buffer(ctx, POLY_BFLOAT16, 1);
  PolyUOp *c = poly_buffer(ctx, POLY_BFLOAT16, 1);
  PolyUOp *ab = poly_alu2(ctx, POLY_OP_ADD, a, b);
  PolyUOp *sum = poly_alu2(ctx, POLY_OP_ADD, ab, c);
  PolyUOp *out = poly_buffer(ctx, POLY_BFLOAT16, 1);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, sum));

  uint16_t a_data[] = {f32_to_bf16_bits(1.0f)};
  uint16_t b_data[] = {f32_to_bf16_bits(1.0f / 256.0f)};
  uint16_t c_data[] = {f32_to_bf16_bits(1.0f / 256.0f)};
  uint16_t out_data[1] = {0};
  PolyTestBufferView binds[] = {
      POLY_TEST_HOST_VIEW(a, a_data), POLY_TEST_HOST_VIEW(b, b_data),
      POLY_TEST_HOST_VIEW(c, c_data), POLY_TEST_HOST_VIEW(out, out_data)};
  ASSERT_INT_EQ(poly_test_realize_buffer_views(ctx, sink, binds, 4), 0);
  PolyDevice preferred = poly_ctx_get_preferred_device(ctx);
  bool per_op_bf16_rounding = preferred == POLY_DEVICE_HIP || preferred == POLY_DEVICE_INTERP;
#ifdef POLY_HAS_CUDA
  per_op_bf16_rounding |= preferred == POLY_DEVICE_CUDA && poly_cuda_arch_major() >= 8;
#endif
  ASSERT_FLOAT_EQ(bf16_bits_to_f32(out_data[0]), per_op_bf16_rounding ? 1.0f : 1.0078125f, 0.0f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(f16, saturated_gelu_family_backward_matches_pinned_backend) {
  /* Exact final-f16-STORE references are recorded by the backend probes.
   * Tensor.tolist() instead inserts CAST(f32) before realization and therefore
   * has a distinct, also pinned, observation boundary. Shared symbolic
   * stabilization is symbolic.py:478-480; renderer-supported ops and dtype
   * legalization then intentionally produce backend-specific stored bits. */
  uint16_t input_data[5] = {
      f32_to_f16_bits(-10.0f),
      f32_to_f16_bits(-8.0f),
      f32_to_f16_bits(-7.0f),
      f32_to_f16_bits(-6.0f),
      f32_to_f16_bits(-5.0f),
  };
  uint16_t expected_cpu[2][5] = {
      {0x0000u, 0x0000u, 0x0000u, 0x0000u, 0x0000u},
      {0x0000u, 0x0000u, 0x0000u, 0x8d8au, 0x9636u},
  };
  uint16_t expected_x86[2][5] = {
      {0x0000u, 0x0000u, 0x0000u, 0x0000u, 0x0000u},
      {0x0000u, 0x0000u, 0x0000u, 0x8d89u, 0x9636u},
  };
  uint16_t expected_cuda[2][5] = {
      {0x0000u, 0x0000u, 0x0000u, 0x0000u, 0x0000u},
      {0x0000u, 0x0000u, 0x0000u, 0x8d88u, 0x9636u},
  };
  uint16_t expected_interp[2][5] = {
      {0x8000u, 0x8000u, 0x8000u, 0x8000u, 0x8000u},
      {0x8000u, 0x8000u, 0x84cau, 0x8d8bu, 0x9632u},
  };
  const char *device = getenv("POLY_DEVICE");
  uint16_t(*expected)[5] = expected_cpu;
  if (device && strcmp(device, "interp") == 0)
    expected = expected_interp;
  else if (device && strcmp(device, "x86") == 0)
    expected = expected_x86;
  else if (device && strcmp(device, "cuda") == 0)
    expected = expected_cuda;

  for (int quick = 0; quick < 2; quick++) {
    PolyCtx *ctx = poly_ctx_new();
    ASSERT_NOT_NULL(ctx);
    PolyUOp *input = poly_buffer(ctx, POLY_FLOAT16, 5);
    PolyUOp *activated =
        quick ? poly_quick_gelu(ctx, input) : poly_gelu(ctx, input);
    PolyUOp *loss =
        poly_reduce_axis(ctx, POLY_OP_ADD, activated, (int64_t[]){0}, 1);
    PolyUOp *gradient = poly_grad(ctx, loss, input);
    PolyUOp *output = poly_buffer(ctx, POLY_FLOAT16, 5);
    PolyUOp *sink =
        gradient ? poly_sink1(ctx, poly_store_val(ctx, output, gradient)) : NULL;
    ASSERT_NOT_NULL(activated);
    ASSERT_NOT_NULL(loss);
    ASSERT_NOT_NULL(gradient);
    ASSERT_NOT_NULL(output);
    ASSERT_NOT_NULL(sink);

    uint16_t output_data[5] = {0};
    PolyTestBufferView binds[] = {
        POLY_TEST_HOST_VIEW(input, input_data),
        POLY_TEST_HOST_VIEW(output, output_data),
    };
    ASSERT_INT_EQ(poly_test_realize_buffer_views(ctx, sink, binds, 2), 0);
    for (int i = 0; i < 5; i++) {
      ASSERT_TRUE(isfinite(f16_bits_to_f32(output_data[i])));
      ASSERT_INT_EQ(output_data[i], expected[quick][i]);
    }

    poly_ctx_destroy(ctx);
  }
  PASS();
}
