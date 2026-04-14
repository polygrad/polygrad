/*
 * test_realize.c — Tests for poly_realize_graph (graph-driven realize).
 *
 * Verifies that poly_realize_graph reads buffers from the side table
 * (attached via poly_buffer_set), schedules, compiles, and executes —
 * with no external bindings array.
 */

#include "test_harness.h"
#include "../src/realize.h"
#include "../src/device.h"
#include "../src/frontend.h"
#include "../src/polygrad.h"

TEST(realize, graph_vecadd) {
  PolyCtx *ctx = poly_ctx_new();

  /* Build: out = a + b */
  PolyUOp *a = poly_buffer_f32(ctx, 4);
  PolyUOp *b = poly_buffer_f32(ctx, 4);
  PolyUOp *out = poly_buffer_f32(ctx, 4);
  PolyUOp *add = poly_alu2(ctx, POLY_OP_ADD, a, b);
  PolyUOp *store = poly_store_val(ctx, out, add);
  PolyUOp *sink = poly_sink1(ctx, store);

  /* Attach data via side table — no external bindings */
  float da[] = {1.0f, 2.0f, 3.0f, 4.0f};
  float db[] = {10.0f, 20.0f, 30.0f, 40.0f};
  float dout[4] = {0};
  poly_buffer_set(ctx, a, da, sizeof(da), POLY_DEVICE_CPU);
  poly_buffer_set(ctx, b, db, sizeof(db), POLY_DEVICE_CPU);
  poly_buffer_set(ctx, out, dout, sizeof(dout), POLY_DEVICE_CPU);

  ASSERT_INT_EQ(poly_realize_graph(ctx, sink), 0);

  ASSERT_FLOAT_EQ(dout[0], 11.0f, 1e-5f);
  ASSERT_FLOAT_EQ(dout[1], 22.0f, 1e-5f);
  ASSERT_FLOAT_EQ(dout[2], 33.0f, 1e-5f);
  ASSERT_FLOAT_EQ(dout[3], 44.0f, 1e-5f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, graph_missing_buffer_errors) {
  /* Calling realize_graph without attaching all buffers should fail cleanly */
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer_f32(ctx, 4);
  PolyUOp *b = poly_buffer_f32(ctx, 4);
  PolyUOp *out = poly_buffer_f32(ctx, 4);
  PolyUOp *add = poly_alu2(ctx, POLY_OP_ADD, a, b);
  PolyUOp *store = poly_store_val(ctx, out, add);
  PolyUOp *sink = poly_sink1(ctx, store);

  /* Attach only a and out, leave b unattached */
  float da[] = {1.0f, 2.0f, 3.0f, 4.0f};
  float dout[4] = {0};
  poly_buffer_set(ctx, a, da, sizeof(da), POLY_DEVICE_CPU);
  poly_buffer_set(ctx, out, dout, sizeof(dout), POLY_DEVICE_CPU);

  /* Should error since b has no data */
  ASSERT_INT_EQ(poly_realize_graph(ctx, sink), -1);

  poly_ctx_destroy(ctx);
  PASS();
}
