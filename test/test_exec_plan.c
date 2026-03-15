/*
 * test_exec_plan.c -- Tests for PolyPreparedStep construction
 */

#include "test_harness.h"
#include "../src/frontend.h"
#include "../src/exec_plan.h"
#include "../src/scheduler.h"

/* ── Single-kernel: vecadd ─────────────────────────────────────────────── */

TEST(exec_plan, prepare_vecadd) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer_f32(ctx, 4);
  PolyUOp *b = poly_buffer_f32(ctx, 4);
  PolyUOp *c = poly_alu2(ctx, POLY_OP_ADD, a, b);
  PolyUOp *out = poly_buffer_f32(ctx, 4);
  PolyUOp *st = poly_store_val(ctx, out, c);
  PolyUOp *sink = poly_sink1(ctx, st);

  PolyPreparedStep *ps = poly_prepare_step(ctx, sink, POLY_MODE_CALL);
  ASSERT_TRUE(ps != NULL);

  /* Single kernel */
  ASSERT_INT_EQ(ps->n_items, 1);
  ASSERT_EQ(ps->items[0].kind, POLY_EXEC_COMPUTE);
  ASSERT_TRUE(ps->items[0].root != NULL);

  /* 3 external buffers (a, b, out), 0 intermediates */
  ASSERT_INT_EQ(ps->n_buf_slots, 3);
  for (int i = 0; i < 3; i++) {
    ASSERT_FALSE(ps->buf_slots[i].is_intermediate);
    ASSERT_INT_EQ(ps->buf_slots[i].numel, 4);
    ASSERT_INT_EQ(ps->buf_slots[i].external_buf_idx, i);
  }

  /* Exec order: single kernel at index 0 */
  ASSERT_TRUE(ps->exec_order != NULL);
  ASSERT_INT_EQ(ps->exec_order[0], 0);

  /* Mode and defaults */
  ASSERT_EQ(ps->mode, POLY_MODE_CALL);
  ASSERT_INT_EQ(ps->loss_buf_slot, -1);

  /* Kernel params should reference buf slots */
  ASSERT_TRUE(ps->items[0].n_buf_slots > 0);
  for (int i = 0; i < ps->items[0].n_buf_slots; i++)
    ASSERT_TRUE(ps->items[0].buf_slot_indices[i] >= 0);

  poly_prepared_step_free(ps);
  poly_ctx_destroy(ctx);
  PASS();
}

/* ── Multi-kernel: reduce -> scalar chain ──────────────────────────────── */

TEST(exec_plan, prepare_multikernel) {
  /* c = expand(reshape(sum(a), (1))) + b  -- sum produces intermediate */
  int N = 8;
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a   = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *b   = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, N);

  int64_t axes[] = {0};
  int64_t one_sh[] = {1};
  int64_t exp_sh[] = {N};
  PolyUOp *s = poly_reduce_axis(ctx, POLY_OP_ADD, a, axes, 1);
  PolyUOp *s1 = poly_reshape(ctx, s, one_sh, 1);
  PolyUOp *se = poly_expand(ctx, s1, exp_sh, 1);
  PolyUOp *c = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, se, b, poly_arg_none());
  PolyUOp *st = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out, c, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, st, poly_arg_none());

  PolyPreparedStep *ps = poly_prepare_step(ctx, sink, POLY_MODE_CALL);
  ASSERT_TRUE(ps != NULL);

  /* Should have multiple kernels (reduce produces intermediate) */
  ASSERT_TRUE(ps->n_items >= 2);

  /* Should have intermediate buffer slots */
  int n_inter = 0;
  for (int i = 0; i < ps->n_buf_slots; i++)
    if (ps->buf_slots[i].is_intermediate) n_inter++;
  ASSERT_TRUE(n_inter > 0);

  /* All exec items should be COMPUTE */
  for (int i = 0; i < ps->n_items; i++)
    ASSERT_EQ(ps->items[i].kind, POLY_EXEC_COMPUTE);

  /* All buf_slot_indices should be valid */
  for (int i = 0; i < ps->n_items; i++)
    for (int j = 0; j < ps->items[i].n_buf_slots; j++) {
      ASSERT_TRUE(ps->items[i].buf_slot_indices[j] >= 0);
      ASSERT_TRUE(ps->items[i].buf_slot_indices[j] < ps->n_buf_slots);
    }

  /* Exec order should cover all kernels */
  ASSERT_TRUE(ps->exec_order != NULL);

  poly_prepared_step_free(ps);
  poly_ctx_destroy(ctx);
  PASS();
}

/* ── Buffer slot metadata ──────────────────────────────────────────────── */

TEST(exec_plan, prepare_buf_slot_metadata) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer_f32(ctx, 8);
  PolyUOp *b = poly_buffer_f64(ctx, 8);
  PolyUOp *c = poly_alu2(ctx, POLY_OP_ADD, a, poly_cast_by_id(ctx, b, 12));
  PolyUOp *out = poly_buffer_f32(ctx, 8);
  PolyUOp *st = poly_store_val(ctx, out, c);
  PolyUOp *sink = poly_sink1(ctx, st);

  PolyPreparedStep *ps = poly_prepare_step(ctx, sink, POLY_MODE_CALL);
  ASSERT_TRUE(ps != NULL);

  /* Check dtype and size are populated */
  for (int i = 0; i < ps->n_buf_slots; i++) {
    if (!ps->buf_slots[i].is_intermediate) {
      ASSERT_TRUE(ps->buf_slots[i].numel > 0);
      ASSERT_TRUE(ps->buf_slots[i].nbytes > 0);
    }
  }

  poly_prepared_step_free(ps);
  poly_ctx_destroy(ctx);
  PASS();
}

/* ── Null/invalid input handling ───────────────────────────────────────── */

TEST(exec_plan, prepare_null_safety) {
  PolyCtx *ctx = poly_ctx_new();

  /* NULL sink */
  ASSERT_TRUE(poly_prepare_step(ctx, NULL, POLY_MODE_CALL) == NULL);

  /* Non-SINK UOp */
  PolyUOp *buf = poly_buffer_f32(ctx, 4);
  ASSERT_TRUE(poly_prepare_step(ctx, buf, POLY_MODE_CALL) == NULL);

  /* Free NULL is safe */
  poly_prepared_step_free(NULL);

  poly_ctx_destroy(ctx);
  PASS();
}

/* ── Graph hash is populated ───────────────────────────────────────────── */

TEST(exec_plan, prepare_graph_hash) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer_f32(ctx, 4);
  PolyUOp *b = poly_buffer_f32(ctx, 4);
  PolyUOp *c = poly_alu2(ctx, POLY_OP_ADD, a, b);
  PolyUOp *out = poly_buffer_f32(ctx, 4);
  PolyUOp *st = poly_store_val(ctx, out, c);
  PolyUOp *sink = poly_sink1(ctx, st);

  PolyPreparedStep *ps = poly_prepare_step(ctx, sink, POLY_MODE_CALL);
  ASSERT_TRUE(ps != NULL);
  ASSERT_TRUE(ps->graph_hash != 0);

  poly_prepared_step_free(ps);
  poly_ctx_destroy(ctx);
  PASS();
}

/* ── Phase 3: Lower + Run ─────────────────────────────────────────────── */

TEST(exec_plan, lower_and_run_vecadd) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer_f32(ctx, 4);
  PolyUOp *b = poly_buffer_f32(ctx, 4);
  PolyUOp *c = poly_alu2(ctx, POLY_OP_ADD, a, b);
  PolyUOp *out = poly_buffer_f32(ctx, 4);
  PolyUOp *st = poly_store_val(ctx, out, c);
  PolyUOp *sink = poly_sink1(ctx, st);

  PolyPreparedStep *ps = poly_prepare_step(ctx, sink, POLY_MODE_CALL);
  ASSERT_TRUE(ps != NULL);

  PolyExecutableStep *es = poly_lower_step(ctx, ps, POLY_DEVICE_CPU);
  ASSERT_TRUE(es != NULL);
  ASSERT_INT_EQ(es->n_runners, ps->n_items);
  ASSERT_EQ(es->device, POLY_DEVICE_CPU);

  /* Prepare data */
  float da[] = {1, 2, 3, 4};
  float db[] = {10, 20, 30, 40};
  float dout[4] = {0};

  /* Build slot_data array indexed by buf_slot */
  void *slot_data[3];
  for (int i = 0; i < ps->n_buf_slots; i++) {
    if (ps->buf_slots[i].buf_uop == a) slot_data[i] = da;
    else if (ps->buf_slots[i].buf_uop == b) slot_data[i] = db;
    else if (ps->buf_slots[i].buf_uop == out) slot_data[i] = dout;
    else slot_data[i] = NULL;
  }

  int ret = poly_executable_step_run(es, slot_data, ps->n_buf_slots, NULL, 0);
  ASSERT_INT_EQ(ret, 0);

  ASSERT_FLOAT_EQ(dout[0], 11.0f, 1e-6);
  ASSERT_FLOAT_EQ(dout[1], 22.0f, 1e-6);
  ASSERT_FLOAT_EQ(dout[2], 33.0f, 1e-6);
  ASSERT_FLOAT_EQ(dout[3], 44.0f, 1e-6);

  poly_executable_step_free(es);
  poly_prepared_step_free(ps);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(exec_plan, lower_and_run_multikernel) {
  /* sum(a) -> reshape -> expand -> add(b) -> out
   * Multi-kernel: reduce produces intermediate. */
  int N = 8;
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a   = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *b   = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, N);

  int64_t axes[] = {0};
  int64_t one_sh[] = {1};
  int64_t exp_sh[] = {N};
  PolyUOp *s = poly_reduce_axis(ctx, POLY_OP_ADD, a, axes, 1);
  PolyUOp *s1 = poly_reshape(ctx, s, one_sh, 1);
  PolyUOp *se = poly_expand(ctx, s1, exp_sh, 1);
  PolyUOp *c = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, se, b, poly_arg_none());
  PolyUOp *st = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out, c, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, st, poly_arg_none());

  PolyPreparedStep *ps = poly_prepare_step(ctx, sink, POLY_MODE_CALL);
  ASSERT_TRUE(ps != NULL);

  PolyExecutableStep *es = poly_lower_step(ctx, ps, POLY_DEVICE_CPU);
  ASSERT_TRUE(es != NULL);
  ASSERT_TRUE(es->n_intermediates > 0);

  /* a = [1..8], sum = 36, b = [10..17], out = 36 + b */
  float da[8], db[8], dout[8];
  for (int i = 0; i < N; i++) { da[i] = (float)(i + 1); db[i] = (float)(i + 10); }
  memset(dout, 0, sizeof(dout));

  void *slot_data[16] = {0};
  for (int i = 0; i < ps->n_buf_slots; i++) {
    if (ps->buf_slots[i].buf_uop == a) slot_data[i] = da;
    else if (ps->buf_slots[i].buf_uop == b) slot_data[i] = db;
    else if (ps->buf_slots[i].buf_uop == out) slot_data[i] = dout;
  }

  int ret = poly_executable_step_run(es, slot_data, ps->n_buf_slots, NULL, 0);
  ASSERT_INT_EQ(ret, 0);

  /* sum(1..8) = 36, out[i] = 36 + (i + 10) */
  for (int i = 0; i < N; i++)
    ASSERT_FLOAT_EQ(dout[i], 36.0f + (float)(i + 10), 1e-5);

  poly_executable_step_free(es);
  poly_prepared_step_free(ps);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(exec_plan, lower_matches_old_compile_step) {
  /* Same graph through both old (poly_compile_step) and new (prepare+lower+run).
   * Results must be identical. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer_f32(ctx, 4);
  PolyUOp *b = poly_buffer_f32(ctx, 4);
  PolyUOp *c = poly_alu2(ctx, POLY_OP_MUL, a, b);
  PolyUOp *out = poly_buffer_f32(ctx, 4);
  PolyUOp *st = poly_store_val(ctx, out, c);
  PolyUOp *sink = poly_sink1(ctx, st);

  float da[] = {2, 3, 4, 5};
  float db[] = {10, 20, 30, 40};

  /* Old path */
  float dout_old[4] = {0};
  PolyStep *old_step = poly_compile_step(ctx, sink);
  ASSERT_TRUE(old_step != NULL);
  PolyBufferBinding bindings[] = {
    {a, da}, {b, db}, {out, dout_old}
  };
  ASSERT_INT_EQ(poly_step_run(old_step, bindings, 3), 0);

  /* New path */
  float dout_new[4] = {0};
  PolyPreparedStep *ps = poly_prepare_step(ctx, sink, POLY_MODE_CALL);
  ASSERT_TRUE(ps != NULL);
  PolyExecutableStep *es = poly_lower_step(ctx, ps, POLY_DEVICE_CPU);
  ASSERT_TRUE(es != NULL);

  void *slot_data[16] = {0};
  for (int i = 0; i < ps->n_buf_slots; i++) {
    if (ps->buf_slots[i].buf_uop == a) slot_data[i] = da;
    else if (ps->buf_slots[i].buf_uop == b) slot_data[i] = db;
    else if (ps->buf_slots[i].buf_uop == out) slot_data[i] = dout_new;
  }
  ASSERT_INT_EQ(poly_executable_step_run(es, slot_data, ps->n_buf_slots, NULL, 0), 0);

  /* Must match exactly */
  for (int i = 0; i < 4; i++)
    ASSERT_FLOAT_EQ(dout_new[i], dout_old[i], 0.0);

  poly_executable_step_free(es);
  poly_prepared_step_free(ps);
  poly_step_destroy(old_step);
  poly_ctx_destroy(ctx);
  PASS();
}
