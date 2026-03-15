/*
 * test_f16.c -- Float16 and BFloat16 end-to-end tests (TDD)
 *
 * These tests verify that float16/bfloat16 operations compile and execute
 * correctly through the C renderer. The strategy matches tinygrad:
 *   - float16 renders as __fp16 (via type_map)
 *   - bfloat16 ALU ops are emulated via f32 promotion
 *   - bfloat16 casts use manual bitwise manipulation
 */

#include "test_harness.h"
#include "../src/polygrad.h"
#include "../src/frontend.h"
#include "../src/scheduler.h"

/* ── Helper: f32 <-> f16 bit conversion (IEEE 754 half-precision) ─────── */

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

/* ── Helper: f32 <-> bf16 bit conversion ──────────────────────────────── */

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

/* ── dtype classification ─────────────────────────────────────────────── */

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

/* ── f32 -> f16 cast + realize ────────────────────────────────────────── */

TEST(f16, cast_f32_to_f16_e2e) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *in = poly_buffer_f32(ctx, 4);
  PolyUOp *casted = poly_cast(ctx, in, POLY_FLOAT16);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT16, 4);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, casted));

  float in_data[] = {1.0f, 2.0f, -0.5f, 0.0f};
  uint16_t out_data[4] = {0};

  PolyBufferBinding binds[] = { {in, in_data}, {out, out_data} };
  int rc = poly_realize(ctx, sink, binds, 2);
  ASSERT_INT_EQ(rc, 0);

  ASSERT_INT_EQ(out_data[0], f32_to_f16_bits(1.0f));
  ASSERT_INT_EQ(out_data[1], f32_to_f16_bits(2.0f));
  ASSERT_INT_EQ(out_data[2], f32_to_f16_bits(-0.5f));
  ASSERT_INT_EQ(out_data[3], f32_to_f16_bits(0.0f));

  poly_ctx_destroy(ctx);
  PASS();
}

/* ── f16 -> f32 cast + realize ────────────────────────────────────────── */

TEST(f16, cast_f16_to_f32_e2e) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *in = poly_buffer(ctx, POLY_FLOAT16, 3);
  PolyUOp *casted = poly_cast(ctx, in, POLY_FLOAT32);
  PolyUOp *out = poly_buffer_f32(ctx, 3);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, casted));

  uint16_t in_data[] = {
    f32_to_f16_bits(1.0f),
    f32_to_f16_bits(3.5f),
    f32_to_f16_bits(-2.0f)
  };
  float out_data[3] = {0};

  PolyBufferBinding binds[] = { {in, in_data}, {out, out_data} };
  int rc = poly_realize(ctx, sink, binds, 2);
  ASSERT_INT_EQ(rc, 0);

  ASSERT_FLOAT_EQ(out_data[0], 1.0f, 1e-3f);
  ASSERT_FLOAT_EQ(out_data[1], 3.5f, 1e-3f);
  ASSERT_FLOAT_EQ(out_data[2], -2.0f, 1e-3f);

  poly_ctx_destroy(ctx);
  PASS();
}

/* ── f16 add e2e ──────────────────────────────────────────────────────── */

TEST(f16, add_f16_e2e) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT16, 4);
  PolyUOp *b = poly_buffer(ctx, POLY_FLOAT16, 4);
  PolyUOp *sum = poly_alu2(ctx, POLY_OP_ADD, a, b);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT16, 4);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, sum));

  uint16_t a_data[] = {
    f32_to_f16_bits(1.0f), f32_to_f16_bits(2.0f),
    f32_to_f16_bits(3.0f), f32_to_f16_bits(-1.0f)
  };
  uint16_t b_data[] = {
    f32_to_f16_bits(10.0f), f32_to_f16_bits(20.0f),
    f32_to_f16_bits(30.0f), f32_to_f16_bits(1.0f)
  };
  uint16_t out_data[4] = {0};

  PolyBufferBinding binds[] = { {a, a_data}, {b, b_data}, {out, out_data} };
  int rc = poly_realize(ctx, sink, binds, 3);
  ASSERT_INT_EQ(rc, 0);

  ASSERT_FLOAT_EQ(f16_bits_to_f32(out_data[0]), 11.0f, 0.1f);
  ASSERT_FLOAT_EQ(f16_bits_to_f32(out_data[1]), 22.0f, 0.1f);
  ASSERT_FLOAT_EQ(f16_bits_to_f32(out_data[2]), 33.0f, 0.1f);
  ASSERT_FLOAT_EQ(f16_bits_to_f32(out_data[3]),  0.0f, 0.1f);

  poly_ctx_destroy(ctx);
  PASS();
}

/* ── f16 mul e2e ──────────────────────────────────────────────────────── */

TEST(f16, mul_f16_e2e) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT16, 3);
  PolyUOp *b = poly_buffer(ctx, POLY_FLOAT16, 3);
  PolyUOp *prod = poly_alu2(ctx, POLY_OP_MUL, a, b);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT16, 3);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, prod));

  uint16_t a_data[] = {
    f32_to_f16_bits(2.0f), f32_to_f16_bits(3.0f), f32_to_f16_bits(-4.0f)
  };
  uint16_t b_data[] = {
    f32_to_f16_bits(5.0f), f32_to_f16_bits(0.5f), f32_to_f16_bits(2.0f)
  };
  uint16_t out_data[3] = {0};

  PolyBufferBinding binds[] = { {a, a_data}, {b, b_data}, {out, out_data} };
  int rc = poly_realize(ctx, sink, binds, 3);
  ASSERT_INT_EQ(rc, 0);

  ASSERT_FLOAT_EQ(f16_bits_to_f32(out_data[0]), 10.0f, 0.1f);
  ASSERT_FLOAT_EQ(f16_bits_to_f32(out_data[1]),  1.5f, 0.1f);
  ASSERT_FLOAT_EQ(f16_bits_to_f32(out_data[2]), -8.0f, 0.1f);

  poly_ctx_destroy(ctx);
  PASS();
}

/* ── f16 neg e2e ──────────────────────────────────────────────────────── */

TEST(f16, neg_f16_e2e) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT16, 3);
  PolyUOp *neg = poly_alu1(ctx, POLY_OP_NEG, a);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT16, 3);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, neg));

  uint16_t a_data[] = {
    f32_to_f16_bits(1.0f), f32_to_f16_bits(-2.0f), f32_to_f16_bits(0.0f)
  };
  uint16_t out_data[3] = {0};

  PolyBufferBinding binds[] = { {a, a_data}, {out, out_data} };
  int rc = poly_realize(ctx, sink, binds, 2);
  ASSERT_INT_EQ(rc, 0);

  ASSERT_FLOAT_EQ(f16_bits_to_f32(out_data[0]), -1.0f, 0.01f);
  ASSERT_FLOAT_EQ(f16_bits_to_f32(out_data[1]),  2.0f, 0.01f);
  ASSERT_FLOAT_EQ(f16_bits_to_f32(out_data[2]),  0.0f, 0.01f);

  poly_ctx_destroy(ctx);
  PASS();
}

/* ── f16 constant rendering ───────────────────────────────────────────── */

TEST(f16, const_f16_e2e) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT16, 3);
  PolyUOp *c = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT16, poly_arg_float(10.0f));
  PolyUOp *sum = poly_alu2(ctx, POLY_OP_ADD, a, c);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT16, 3);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, sum));

  uint16_t a_data[] = {
    f32_to_f16_bits(1.0f), f32_to_f16_bits(2.0f), f32_to_f16_bits(3.0f)
  };
  uint16_t out_data[3] = {0};

  PolyBufferBinding binds[] = { {a, a_data}, {out, out_data} };
  int rc = poly_realize(ctx, sink, binds, 2);
  ASSERT_INT_EQ(rc, 0);

  ASSERT_FLOAT_EQ(f16_bits_to_f32(out_data[0]), 11.0f, 0.1f);
  ASSERT_FLOAT_EQ(f16_bits_to_f32(out_data[1]), 12.0f, 0.1f);
  ASSERT_FLOAT_EQ(f16_bits_to_f32(out_data[2]), 13.0f, 0.1f);

  poly_ctx_destroy(ctx);
  PASS();
}

/* ── mixed precision: f16 -> f32 compute -> f32 output ────────────────── */

TEST(f16, mixed_f16_to_f32_chain_e2e) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *in = poly_buffer(ctx, POLY_FLOAT16, 3);
  PolyUOp *as_f32 = poly_cast(ctx, in, POLY_FLOAT32);
  PolyUOp *c = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(100.0f));
  PolyUOp *sum = poly_alu2(ctx, POLY_OP_ADD, as_f32, c);
  PolyUOp *out = poly_buffer_f32(ctx, 3);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, sum));

  uint16_t in_data[] = {
    f32_to_f16_bits(1.0f), f32_to_f16_bits(2.0f), f32_to_f16_bits(3.0f)
  };
  float out_data[3] = {0};

  PolyBufferBinding binds[] = { {in, in_data}, {out, out_data} };
  int rc = poly_realize(ctx, sink, binds, 2);
  ASSERT_INT_EQ(rc, 0);

  ASSERT_FLOAT_EQ(out_data[0], 101.0f, 0.1f);
  ASSERT_FLOAT_EQ(out_data[1], 102.0f, 0.1f);
  ASSERT_FLOAT_EQ(out_data[2], 103.0f, 0.1f);

  poly_ctx_destroy(ctx);
  PASS();
}

/* ── f64 -> f16 cast (should go via f32 intermediate) ─────────────────── */

TEST(f16, cast_f64_to_f16_e2e) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *in = poly_buffer_f64(ctx, 2);
  PolyUOp *casted = poly_cast(ctx, in, POLY_FLOAT16);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT16, 2);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, casted));

  double in_data[] = {1.5, -2.5};
  uint16_t out_data[2] = {0};

  PolyBufferBinding binds[] = { {in, in_data}, {out, out_data} };
  int rc = poly_realize(ctx, sink, binds, 2);
  ASSERT_INT_EQ(rc, 0);

  ASSERT_FLOAT_EQ(f16_bits_to_f32(out_data[0]),  1.5f, 0.01f);
  ASSERT_FLOAT_EQ(f16_bits_to_f32(out_data[1]), -2.5f, 0.01f);

  poly_ctx_destroy(ctx);
  PASS();
}

/* ── bf16 cast f32 -> bf16 e2e ────────────────────────────────────────── */

TEST(f16, cast_f32_to_bf16_e2e) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *in = poly_buffer_f32(ctx, 3);
  PolyUOp *casted = poly_cast(ctx, in, POLY_BFLOAT16);
  PolyUOp *out = poly_buffer(ctx, POLY_BFLOAT16, 3);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, casted));

  float in_data[] = {1.0f, -2.0f, 0.5f};
  uint16_t out_data[3] = {0};

  PolyBufferBinding binds[] = { {in, in_data}, {out, out_data} };
  int rc = poly_realize(ctx, sink, binds, 2);
  ASSERT_INT_EQ(rc, 0);

  ASSERT_FLOAT_EQ(bf16_bits_to_f32(out_data[0]),  1.0f, 0.01f);
  ASSERT_FLOAT_EQ(bf16_bits_to_f32(out_data[1]), -2.0f, 0.01f);
  ASSERT_FLOAT_EQ(bf16_bits_to_f32(out_data[2]),  0.5f, 0.01f);

  poly_ctx_destroy(ctx);
  PASS();
}

/* ── bf16 cast bf16 -> f32 e2e ────────────────────────────────────────── */

TEST(f16, cast_bf16_to_f32_e2e) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *in = poly_buffer(ctx, POLY_BFLOAT16, 3);
  PolyUOp *casted = poly_cast(ctx, in, POLY_FLOAT32);
  PolyUOp *out = poly_buffer_f32(ctx, 3);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, casted));

  uint16_t in_data[] = {
    f32_to_bf16_bits(1.0f),
    f32_to_bf16_bits(-3.0f),
    f32_to_bf16_bits(0.25f)
  };
  float out_data[3] = {0};

  PolyBufferBinding binds[] = { {in, in_data}, {out, out_data} };
  int rc = poly_realize(ctx, sink, binds, 2);
  ASSERT_INT_EQ(rc, 0);

  ASSERT_FLOAT_EQ(out_data[0],  1.0f, 0.01f);
  ASSERT_FLOAT_EQ(out_data[1], -3.0f, 0.01f);
  ASSERT_FLOAT_EQ(out_data[2],  0.25f, 0.01f);

  poly_ctx_destroy(ctx);
  PASS();
}

/* ── bf16 add (via f32 emulation) e2e ─────────────────────────────────── */

TEST(f16, add_bf16_e2e) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer(ctx, POLY_BFLOAT16, 3);
  PolyUOp *b = poly_buffer(ctx, POLY_BFLOAT16, 3);
  PolyUOp *sum = poly_alu2(ctx, POLY_OP_ADD, a, b);
  PolyUOp *out = poly_buffer(ctx, POLY_BFLOAT16, 3);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, sum));

  uint16_t a_data[] = {
    f32_to_bf16_bits(1.0f), f32_to_bf16_bits(2.0f), f32_to_bf16_bits(-1.0f)
  };
  uint16_t b_data[] = {
    f32_to_bf16_bits(10.0f), f32_to_bf16_bits(20.0f), f32_to_bf16_bits(1.0f)
  };
  uint16_t out_data[3] = {0};

  PolyBufferBinding binds[] = { {a, a_data}, {b, b_data}, {out, out_data} };
  int rc = poly_realize(ctx, sink, binds, 3);
  ASSERT_INT_EQ(rc, 0);

  ASSERT_FLOAT_EQ(bf16_bits_to_f32(out_data[0]), 11.0f, 0.2f);
  ASSERT_FLOAT_EQ(bf16_bits_to_f32(out_data[1]), 22.0f, 0.2f);
  ASSERT_FLOAT_EQ(bf16_bits_to_f32(out_data[2]),  0.0f, 0.2f);

  poly_ctx_destroy(ctx);
  PASS();
}

/* ── bf16 mul (via f32 emulation) e2e ─────────────────────────────────── */

TEST(f16, mul_bf16_e2e) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer(ctx, POLY_BFLOAT16, 3);
  PolyUOp *b = poly_buffer(ctx, POLY_BFLOAT16, 3);
  PolyUOp *prod = poly_alu2(ctx, POLY_OP_MUL, a, b);
  PolyUOp *out = poly_buffer(ctx, POLY_BFLOAT16, 3);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, prod));

  uint16_t a_data[] = {
    f32_to_bf16_bits(2.0f), f32_to_bf16_bits(3.0f), f32_to_bf16_bits(-4.0f)
  };
  uint16_t b_data[] = {
    f32_to_bf16_bits(5.0f), f32_to_bf16_bits(0.5f), f32_to_bf16_bits(2.0f)
  };
  uint16_t out_data[3] = {0};

  PolyBufferBinding binds[] = { {a, a_data}, {b, b_data}, {out, out_data} };
  int rc = poly_realize(ctx, sink, binds, 3);
  ASSERT_INT_EQ(rc, 0);

  ASSERT_FLOAT_EQ(bf16_bits_to_f32(out_data[0]), 10.0f, 0.2f);
  ASSERT_FLOAT_EQ(bf16_bits_to_f32(out_data[1]),  1.5f, 0.2f);
  ASSERT_FLOAT_EQ(bf16_bits_to_f32(out_data[2]), -8.0f, 0.2f);

  poly_ctx_destroy(ctx);
  PASS();
}
