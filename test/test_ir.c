/*
 * test_ir.c -- Tests for poly_ir binary export/import
 */

#include "test_harness.h"
#include "../src/ir.h"
#include "../src/frontend.h"
#include "../src/sched.h"
#include <string.h>
#include <stdlib.h>

/* ── Round-trip: simple add graph ──────────────────────────────────── */

TEST(ir, round_trip_add) {
  PolyCtx *ctx = poly_ctx_new();

  /* Build: out = a + b */
  PolyUOp *a = poly_buffer_f32(ctx, 4);
  PolyUOp *b = poly_buffer_f32(ctx, 4);
  PolyUOp *sum = poly_alu2(ctx, POLY_OP_ADD, a, b);
  PolyUOp *out_buf = poly_buffer_f32(ctx, 4);
  PolyUOp *store = poly_store_val(ctx, out_buf, sum);
  PolyUOp *sink = poly_sink1(ctx, store);

  int64_t shape4[] = { 4 };
  PolyIrBufEntry bufs[] = {
    { "a", POLY_IR_ROLE_INPUT, a, { 4 }, 1 },
    { "b", POLY_IR_ROLE_INPUT, b, { 4 }, 1 },
    { "output", POLY_IR_ROLE_OUTPUT, out_buf, { 4 }, 1 },
  };
  (void)shape4;

  PolyIrEntrypoint eps[] = {
    { "forward", sink },
  };

  PolyIrSpec spec = { ctx, bufs, 3, eps, 1 };

  int out_len = 0;
  uint8_t *bytes = poly_ir_export(&spec, &out_len);
  ASSERT_NOT_NULL(bytes);
  ASSERT_TRUE(out_len > 32);

  /* Import */
  PolyIrSpec imported;
  int ret = poly_ir_import(bytes, out_len, &imported);
  ASSERT_INT_EQ(ret, 0);
  ASSERT_NOT_NULL(imported.ctx);
  ASSERT_INT_EQ(imported.n_bufs, 3);
  ASSERT_INT_EQ(imported.n_entrypoints, 1);

  /* Check buffer names */
  ASSERT_STR_EQ(imported.bufs[0].name, "a");
  ASSERT_INT_EQ(imported.bufs[0].role, POLY_IR_ROLE_INPUT);
  ASSERT_INT_EQ(imported.bufs[0].ndim, 1);
  ASSERT_INT_EQ(imported.bufs[0].shape[0], 4);

  ASSERT_STR_EQ(imported.bufs[1].name, "b");
  ASSERT_STR_EQ(imported.bufs[2].name, "output");

  ASSERT_STR_EQ(imported.entrypoints[0].name, "forward");
  ASSERT_NOT_NULL(imported.entrypoints[0].sink);

  poly_ir_spec_free(&imported);
  poly_ctx_destroy(imported.ctx);
  poly_ctx_destroy(ctx);
  free(bytes);
  PASS();
}

/* ── Round-trip: graph with CONST args ─────────────────────────────── */

TEST(ir, round_trip_const) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer_f32(ctx, 4);
  PolyUOp *two = poly_const_float(ctx, 2.0);
  PolyUOp *scaled = poly_alu2(ctx, POLY_OP_MUL, a, two);
  PolyUOp *out = poly_buffer_f32(ctx, 4);
  PolyUOp *store = poly_store_val(ctx, out, scaled);
  PolyUOp *sink = poly_sink1(ctx, store);

  PolyIrBufEntry bufs[] = {
    { "input", POLY_IR_ROLE_INPUT, a, { 4 }, 1 },
    { "output", POLY_IR_ROLE_OUTPUT, out, { 4 }, 1 },
  };
  PolyIrEntrypoint eps[] = { { "forward", sink } };
  PolyIrSpec spec = { ctx, bufs, 2, eps, 1 };

  int out_len = 0;
  uint8_t *bytes = poly_ir_export(&spec, &out_len);
  ASSERT_NOT_NULL(bytes);

  PolyIrSpec imported;
  int ret = poly_ir_import(bytes, out_len, &imported);
  ASSERT_INT_EQ(ret, 0);
  ASSERT_INT_EQ(imported.n_bufs, 2);
  ASSERT_STR_EQ(imported.bufs[0].name, "input");
  ASSERT_STR_EQ(imported.bufs[1].name, "output");

  poly_ir_spec_free(&imported);
  poly_ctx_destroy(imported.ctx);
  poly_ctx_destroy(ctx);
  free(bytes);
  PASS();
}

/* ── Round-trip: multiple entrypoints ──────────────────────────────── */

TEST(ir, round_trip_multi_entry) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *x = poly_buffer_f32(ctx, 4);
  PolyUOp *fwd_out = poly_buffer_f32(ctx, 4);
  PolyUOp *fwd_store = poly_store_val(ctx, fwd_out, x);
  PolyUOp *fwd_sink = poly_sink1(ctx, fwd_store);

  PolyUOp *loss_out = poly_buffer_f32(ctx, 1);
  PolyUOp *loss_val = poly_const_float(ctx, 0.0);
  PolyUOp *loss_store = poly_store_val(ctx, loss_out, loss_val);
  PolyUOp *loss_sink = poly_sink1(ctx, loss_store);

  PolyIrBufEntry bufs[] = {
    { "x", POLY_IR_ROLE_INPUT, x, { 4 }, 1 },
    { "output", POLY_IR_ROLE_OUTPUT, fwd_out, { 4 }, 1 },
    { "loss", POLY_IR_ROLE_OUTPUT, loss_out, { 1 }, 1 },
  };
  PolyIrEntrypoint eps[] = {
    { "forward", fwd_sink },
    { "loss", loss_sink },
  };
  PolyIrSpec spec = { ctx, bufs, 3, eps, 2 };

  int out_len = 0;
  uint8_t *bytes = poly_ir_export(&spec, &out_len);
  ASSERT_NOT_NULL(bytes);

  PolyIrSpec imported;
  int ret = poly_ir_import(bytes, out_len, &imported);
  ASSERT_INT_EQ(ret, 0);
  ASSERT_INT_EQ(imported.n_entrypoints, 2);
  ASSERT_STR_EQ(imported.entrypoints[0].name, "forward");
  ASSERT_STR_EQ(imported.entrypoints[1].name, "loss");

  poly_ir_spec_free(&imported);
  poly_ctx_destroy(imported.ctx);
  poly_ctx_destroy(ctx);
  free(bytes);
  PASS();
}

/* ── Round-trip: param roles ──────────────────────────────────────── */

TEST(ir, round_trip_roles) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *w = poly_buffer_f32(ctx, 6);
  PolyUOp *x = poly_buffer_f32(ctx, 3);
  PolyUOp *out = poly_buffer_f32(ctx, 2);
  PolyUOp *sum = poly_alu2(ctx, POLY_OP_ADD, w, x);
  PolyUOp *store = poly_store_val(ctx, out, sum);
  PolyUOp *sink = poly_sink1(ctx, store);

  PolyIrBufEntry bufs[] = {
    { "layers.0.weight", POLY_IR_ROLE_PARAM, w, { 2, 3 }, 2 },
    { "x", POLY_IR_ROLE_INPUT, x, { 3 }, 1 },
    { "output", POLY_IR_ROLE_OUTPUT, out, { 2 }, 1 },
  };
  PolyIrEntrypoint eps[] = { { "forward", sink } };
  PolyIrSpec spec = { ctx, bufs, 3, eps, 1 };

  int out_len = 0;
  uint8_t *bytes = poly_ir_export(&spec, &out_len);
  ASSERT_NOT_NULL(bytes);

  PolyIrSpec imported;
  int ret = poly_ir_import(bytes, out_len, &imported);
  ASSERT_INT_EQ(ret, 0);

  /* Check roles preserved */
  ASSERT_STR_EQ(imported.bufs[0].name, "layers.0.weight");
  ASSERT_INT_EQ(imported.bufs[0].role, POLY_IR_ROLE_PARAM);
  ASSERT_INT_EQ(imported.bufs[0].ndim, 2);
  ASSERT_INT_EQ(imported.bufs[0].shape[0], 2);
  ASSERT_INT_EQ(imported.bufs[0].shape[1], 3);

  ASSERT_INT_EQ(imported.bufs[1].role, POLY_IR_ROLE_INPUT);
  ASSERT_INT_EQ(imported.bufs[2].role, POLY_IR_ROLE_OUTPUT);

  poly_ir_spec_free(&imported);
  poly_ctx_destroy(imported.ctx);
  poly_ctx_destroy(ctx);
  free(bytes);
  PASS();
}

/* ── Invalid data ─────────────────────────────────────────────────── */

TEST(ir, import_bad_magic) {
  uint8_t data[32] = { 0 };
  PolyIrSpec spec;
  int ret = poly_ir_import(data, 32, &spec);
  ASSERT_INT_EQ(ret, -1);
  PASS();
}

TEST(ir, import_truncated) {
  uint8_t data[16] = { 'P', 'G', 'I', 'R' };
  PolyIrSpec spec;
  int ret = poly_ir_import(data, 16, &spec);
  ASSERT_INT_EQ(ret, -1);
  PASS();
}

/* ── Round-trip: graph with int tuple args ─────────────────────────── */

TEST(ir, round_trip_reshape) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer_f32(ctx, 6);
  int64_t new_shape[] = { 2, 3 };
  PolyUOp *reshaped = poly_reshape(ctx, a, new_shape, 2);
  PolyUOp *out = poly_buffer_f32(ctx, 6);
  PolyUOp *store = poly_store_val(ctx, out, reshaped);
  PolyUOp *sink = poly_sink1(ctx, store);

  PolyIrBufEntry bufs[] = {
    { "input", POLY_IR_ROLE_INPUT, a, { 6 }, 1 },
    { "output", POLY_IR_ROLE_OUTPUT, out, { 2, 3 }, 2 },
  };
  PolyIrEntrypoint eps[] = { { "forward", sink } };
  PolyIrSpec spec = { ctx, bufs, 2, eps, 1 };

  int out_len = 0;
  uint8_t *bytes = poly_ir_export(&spec, &out_len);
  ASSERT_NOT_NULL(bytes);

  PolyIrSpec imported;
  int ret = poly_ir_import(bytes, out_len, &imported);
  ASSERT_INT_EQ(ret, 0);
  ASSERT_INT_EQ(imported.n_bufs, 2);

  poly_ir_spec_free(&imported);
  poly_ctx_destroy(imported.ctx);
  poly_ctx_destroy(ctx);
  free(bytes);
  PASS();
}
