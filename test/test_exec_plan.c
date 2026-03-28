/*
 * test_exec_plan.c -- Tests for the cross-platform execution plan
 *
 * Phase 1-2: PolyPreparedStep construction
 * Phase 3:   CPU executable step (lower + run)
 * Phase 4:   Interpreter backend (CPU vs INTERP parity)
 */

#include "test_harness.h"
#include "../src/frontend.h"
#include "../src/exec_plan.h"
#include "../src/scheduler.h"

/* ── Helper: run same graph on CPU and INTERP, compare outputs ────────── */

static int cpu_interp_parity(PolyCtx *ctx, PolyUOp *sink,
                             PolyUOp **bufs, void **datas, int n_bufs,
                             PolyUOp *out_buf, float *out_cpu, float *out_interp,
                             int out_numel, float tol) {
  PolyPreparedStep *ps = poly_prepare_step(ctx, sink, POLY_MODE_CALL);
  if (!ps) return -1;

  /* CPU path */
  PolyExecutableStep *cpu = poly_lower_step(ctx, ps, POLY_DEVICE_CPU);
  if (!cpu) { poly_prepared_step_free(ps); return -2; }
  memset(out_cpu, 0, (size_t)out_numel * sizeof(float));
  void *slot_cpu[16] = {0};
  for (int i = 0; i < ps->n_buf_slots; i++) {
    for (int j = 0; j < n_bufs; j++) {
      if (ps->buf_slots[i].buf_uop == bufs[j])
        slot_cpu[i] = (bufs[j] == out_buf) ? out_cpu : datas[j];
    }
  }
  int rc = poly_executable_step_run(cpu, slot_cpu, ps->n_buf_slots, NULL, 0);
  poly_executable_step_free(cpu);
  if (rc < 0) { poly_prepared_step_free(ps); return -3; }

  /* INTERP path */
  PolyExecutableStep *interp = poly_lower_step(ctx, ps, POLY_DEVICE_INTERP);
  if (!interp) { poly_prepared_step_free(ps); return -4; }
  memset(out_interp, 0, (size_t)out_numel * sizeof(float));
  void *slot_interp[16] = {0};
  for (int i = 0; i < ps->n_buf_slots; i++) {
    for (int j = 0; j < n_bufs; j++) {
      if (ps->buf_slots[i].buf_uop == bufs[j])
        slot_interp[i] = (bufs[j] == out_buf) ? out_interp : datas[j];
    }
  }
  rc = poly_executable_step_run(interp, slot_interp, ps->n_buf_slots, NULL, 0);
  poly_executable_step_free(interp);
  poly_prepared_step_free(ps);
  if (rc < 0) return -5;

  /* Compare */
  for (int i = 0; i < out_numel; i++) {
    float diff = out_cpu[i] - out_interp[i];
    if (diff < 0) diff = -diff;
    if (diff > tol) return i + 1; /* 1-based index of first mismatch */
  }
  return 0;
}

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
  /* Plan has intermediate buffer slots (intermediates are per-run now) */
  int n_inter = 0;
  for (int i = 0; i < ps->n_buf_slots; i++)
    if (ps->buf_slots[i].is_intermediate) n_inter++;
  ASSERT_TRUE(n_inter > 0);

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
    POLY_BIND_HOST(a, da), POLY_BIND_HOST(b, db), POLY_BIND_HOST(out, dout_old)
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

/* ── Phase 4: Interpreter backend ─────────────────────────────────────── */

TEST(exec_plan, interp_vecadd) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer_f32(ctx, 4);
  PolyUOp *b = poly_buffer_f32(ctx, 4);
  PolyUOp *c = poly_alu2(ctx, POLY_OP_ADD, a, b);
  PolyUOp *out = poly_buffer_f32(ctx, 4);
  PolyUOp *st = poly_store_val(ctx, out, c);
  PolyUOp *sink = poly_sink1(ctx, st);

  PolyPreparedStep *ps = poly_prepare_step(ctx, sink, POLY_MODE_CALL);
  ASSERT_TRUE(ps != NULL);

  PolyExecutableStep *es = poly_lower_step(ctx, ps, POLY_DEVICE_INTERP);
  ASSERT_TRUE(es != NULL);
  ASSERT_EQ(es->device, POLY_DEVICE_INTERP);

  float da[] = {1, 2, 3, 4};
  float db[] = {10, 20, 30, 40};
  float dout[4] = {0};

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

TEST(exec_plan, interp_reduce) {
  /* sum(a) -> reshape -> expand -> add(b) -> out */
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

  PolyExecutableStep *es = poly_lower_step(ctx, ps, POLY_DEVICE_INTERP);
  ASSERT_TRUE(es != NULL);

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

  for (int i = 0; i < N; i++)
    ASSERT_FLOAT_EQ(dout[i], 36.0f + (float)(i + 10), 1e-5);

  poly_executable_step_free(es);
  poly_prepared_step_free(ps);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(exec_plan, interp_matches_cpu) {
  /* Same graph, CPU vs INTERP, must produce identical results */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer_f32(ctx, 4);
  PolyUOp *b = poly_buffer_f32(ctx, 4);
  PolyUOp *c = poly_alu2(ctx, POLY_OP_MUL, a, b);
  PolyUOp *out = poly_buffer_f32(ctx, 4);
  PolyUOp *st = poly_store_val(ctx, out, c);
  PolyUOp *sink = poly_sink1(ctx, st);

  float da[] = {2, 3, 4, 5};
  float db[] = {10, 20, 30, 40};

  PolyPreparedStep *ps = poly_prepare_step(ctx, sink, POLY_MODE_CALL);
  ASSERT_TRUE(ps != NULL);

  /* CPU path */
  float dout_cpu[4] = {0};
  PolyExecutableStep *cpu = poly_lower_step(ctx, ps, POLY_DEVICE_CPU);
  ASSERT_TRUE(cpu != NULL);

  void *slot_cpu[16] = {0};
  for (int i = 0; i < ps->n_buf_slots; i++) {
    if (ps->buf_slots[i].buf_uop == a) slot_cpu[i] = da;
    else if (ps->buf_slots[i].buf_uop == b) slot_cpu[i] = db;
    else if (ps->buf_slots[i].buf_uop == out) slot_cpu[i] = dout_cpu;
  }
  ASSERT_INT_EQ(poly_executable_step_run(cpu, slot_cpu, ps->n_buf_slots, NULL, 0), 0);

  /* INTERP path */
  float dout_interp[4] = {0};
  PolyExecutableStep *interp = poly_lower_step(ctx, ps, POLY_DEVICE_INTERP);
  ASSERT_TRUE(interp != NULL);

  void *slot_interp[16] = {0};
  for (int i = 0; i < ps->n_buf_slots; i++) {
    if (ps->buf_slots[i].buf_uop == a) slot_interp[i] = da;
    else if (ps->buf_slots[i].buf_uop == b) slot_interp[i] = db;
    else if (ps->buf_slots[i].buf_uop == out) slot_interp[i] = dout_interp;
  }
  ASSERT_INT_EQ(poly_executable_step_run(interp, slot_interp, ps->n_buf_slots, NULL, 0), 0);

  /* Bitwise identical (integer multiply, no FP rounding differences) */
  ASSERT_TRUE(memcmp(dout_cpu, dout_interp, sizeof(dout_cpu)) == 0);

  poly_executable_step_free(cpu);
  poly_executable_step_free(interp);
  poly_prepared_step_free(ps);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(exec_plan, interp_transcendental) {
  /* exp(log(a)) ~= a, tests the decomposed transcendental path */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer_f32(ctx, 4);
  PolyUOp *lg = poly_alu1(ctx, POLY_OP_LOG2, a);
  PolyUOp *ex = poly_alu1(ctx, POLY_OP_EXP2, lg);
  PolyUOp *out = poly_buffer_f32(ctx, 4);
  PolyUOp *st = poly_store_val(ctx, out, ex);
  PolyUOp *sink = poly_sink1(ctx, st);

  PolyPreparedStep *ps = poly_prepare_step(ctx, sink, POLY_MODE_CALL);
  ASSERT_TRUE(ps != NULL);

  PolyExecutableStep *es = poly_lower_step(ctx, ps, POLY_DEVICE_INTERP);
  ASSERT_TRUE(es != NULL);

  float da[] = {1.0f, 2.0f, 4.0f, 8.0f};
  float dout[4] = {0};

  void *slot_data[16] = {0};
  for (int i = 0; i < ps->n_buf_slots; i++) {
    if (ps->buf_slots[i].buf_uop == a) slot_data[i] = da;
    else if (ps->buf_slots[i].buf_uop == out) slot_data[i] = dout;
  }

  ASSERT_INT_EQ(poly_executable_step_run(es, slot_data, ps->n_buf_slots, NULL, 0), 0);

  /* exp2(log2(x)) ~= x, within polynomial approximation tolerance */
  for (int i = 0; i < 4; i++)
    ASSERT_FLOAT_EQ(dout[i], da[i], 1e-4);

  poly_executable_step_free(es);
  poly_prepared_step_free(ps);
  poly_ctx_destroy(ctx);
  PASS();
}

/* ── CPU vs INTERP parity suite ──────────────────────────────────────── */

TEST(exec_plan, parity_chain) {
  /* (a + b) * (a - b) -- 3-op elementwise chain */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer_f32(ctx, 8);
  PolyUOp *b = poly_buffer_f32(ctx, 8);
  PolyUOp *add = poly_alu2(ctx, POLY_OP_ADD, a, b);
  PolyUOp *sub = poly_alu2(ctx, POLY_OP_SUB, a, b);
  PolyUOp *mul = poly_alu2(ctx, POLY_OP_MUL, add, sub);
  PolyUOp *out = poly_buffer_f32(ctx, 8);
  PolyUOp *st = poly_store_val(ctx, out, mul);
  PolyUOp *sink = poly_sink1(ctx, st);

  float da[] = {1, 2, 3, 4, 5, 6, 7, 8};
  float db[] = {0.5f, 1, 1.5f, 2, 2.5f, 3, 3.5f, 4};
  float out_cpu[8], out_interp[8];
  PolyUOp *bufs[] = {a, b, out};
  void *datas[] = {da, db, NULL};

  int rc = cpu_interp_parity(ctx, sink, bufs, datas, 3,
                             out, out_cpu, out_interp, 8, 0.0f);
  ASSERT_INT_EQ(rc, 0);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(exec_plan, parity_neg_sqrt) {
  /* sqrt(a) + neg(b) */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer_f32(ctx, 4);
  PolyUOp *b = poly_buffer_f32(ctx, 4);
  PolyUOp *sa = poly_alu1(ctx, POLY_OP_SQRT, a);
  PolyUOp *nb = poly_alu1(ctx, POLY_OP_NEG, b);
  PolyUOp *r = poly_alu2(ctx, POLY_OP_ADD, sa, nb);
  PolyUOp *out = poly_buffer_f32(ctx, 4);
  PolyUOp *st = poly_store_val(ctx, out, r);
  PolyUOp *sink = poly_sink1(ctx, st);

  float da[] = {1, 4, 9, 16};
  float db[] = {0.5f, 1, 1.5f, 2};
  float out_cpu[4], out_interp[4];
  PolyUOp *bufs[] = {a, b, out};
  void *datas[] = {da, db, NULL};

  int rc = cpu_interp_parity(ctx, sink, bufs, datas, 3,
                             out, out_cpu, out_interp, 4, 1e-6f);
  ASSERT_INT_EQ(rc, 0);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(exec_plan, parity_reduce_sum) {
  /* sum(a) -> scalar output */
  int N = 8;
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, 1);

  int64_t axes[] = {0};
  PolyUOp *s = poly_reduce_axis(ctx, POLY_OP_ADD, a, axes, 1);
  PolyUOp *st = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out, s, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, st, poly_arg_none());

  float da[] = {1, 2, 3, 4, 5, 6, 7, 8};
  float out_cpu[1], out_interp[1];
  PolyUOp *bufs[] = {a, out};
  void *datas[] = {da, NULL};

  int rc = cpu_interp_parity(ctx, sink, bufs, datas, 2,
                             out, out_cpu, out_interp, 1, 1e-5f);
  ASSERT_INT_EQ(rc, 0);
  ASSERT_FLOAT_EQ(out_cpu[0], 36.0f, 1e-5);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(exec_plan, parity_where) {
  /* where(a > 0, a, b) */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer_f32(ctx, 4);
  PolyUOp *b = poly_buffer_f32(ctx, 4);
  PolyUOp *zero = poly_const_float(ctx, 0.0);
  PolyUOp *cmp = poly_alu2(ctx, POLY_OP_CMPLT, zero, a);
  PolyUOp *w = poly_alu3(ctx, POLY_OP_WHERE, cmp, a, b);
  PolyUOp *out = poly_buffer_f32(ctx, 4);
  PolyUOp *st = poly_store_val(ctx, out, w);
  PolyUOp *sink = poly_sink1(ctx, st);

  float da[] = {-1, 2, -3, 4};
  float db[] = {10, 20, 30, 40};
  float out_cpu[4], out_interp[4];
  PolyUOp *bufs[] = {a, b, out};
  void *datas[] = {da, db, NULL};

  int rc = cpu_interp_parity(ctx, sink, bufs, datas, 3,
                             out, out_cpu, out_interp, 4, 0.0f);
  ASSERT_INT_EQ(rc, 0);
  ASSERT_FLOAT_EQ(out_cpu[0], 10.0f, 1e-6);
  ASSERT_FLOAT_EQ(out_cpu[1],  2.0f, 1e-6);
  ASSERT_FLOAT_EQ(out_cpu[2], 30.0f, 1e-6);
  ASSERT_FLOAT_EQ(out_cpu[3],  4.0f, 1e-6);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(exec_plan, parity_exp2_log2) {
  /* exp2(log2(a)) ~= a (through decomposed polynomial path) */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer_f32(ctx, 4);
  PolyUOp *lg = poly_alu1(ctx, POLY_OP_LOG2, a);
  PolyUOp *ex = poly_alu1(ctx, POLY_OP_EXP2, lg);
  PolyUOp *out = poly_buffer_f32(ctx, 4);
  PolyUOp *st = poly_store_val(ctx, out, ex);
  PolyUOp *sink = poly_sink1(ctx, st);

  float da[] = {0.5f, 1.0f, 2.0f, 8.0f};
  float out_cpu[4], out_interp[4];
  PolyUOp *bufs[] = {a, out};
  void *datas[] = {da, NULL};

  int rc = cpu_interp_parity(ctx, sink, bufs, datas, 2,
                             out, out_cpu, out_interp, 4, 1e-5f);
  ASSERT_INT_EQ(rc, 0);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(exec_plan, parity_multikernel_reduce_chain) {
  /* sum(a) -> expand -> add(b): multi-kernel with intermediate */
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

  float da[8], db[8];
  for (int i = 0; i < N; i++) { da[i] = (float)(i + 1); db[i] = (float)(i * 10); }
  float out_cpu[8], out_interp[8];
  PolyUOp *bufs[] = {a, b, out};
  void *datas[] = {da, db, NULL};

  int rc = cpu_interp_parity(ctx, sink, bufs, datas, 3,
                             out, out_cpu, out_interp, N, 1e-5f);
  ASSERT_INT_EQ(rc, 0);

  poly_ctx_destroy(ctx);
  PASS();
}

/* ── Phase 5: Persistent workspace ───────────────────────────────────── */

TEST(exec_plan, workspace_reuse) {
  /* Run same plan 100 times with different data. Persistent intermediates
   * are zeroed each call (reduce accumulators). No per-call allocations. */
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

  PolySchedule *ps = poly_schedule_for(ctx, sink, POLY_MODE_CALL);
  ASSERT_TRUE(ps != NULL);

  PolyCompiledPlan *plan = poly_compile_schedule(ctx, ps, POLY_DEVICE_CPU);
  ASSERT_TRUE(plan != NULL);

  /* Verify persistent intermediates were allocated */
  int n_inter = 0;
  for (int i = 0; i < ps->n_buf_slots; i++)
    if (ps->buf_slots[i].is_intermediate) n_inter++;
  ASSERT_TRUE(n_inter > 0);
  ASSERT_INT_EQ(plan->n_intermediates, n_inter);

  /* Run 100 times with different multipliers */
  for (int iter = 0; iter < 100; iter++) {
    float da[8], db[8], dout[8];
    float mult = (float)(iter + 1);
    for (int i = 0; i < N; i++) {
      da[i] = (float)(i + 1) * mult;
      db[i] = (float)(i + 10);
    }
    memset(dout, 0, sizeof(dout));

    void *slot_data[16] = {0};
    for (int i = 0; i < ps->n_buf_slots; i++) {
      if (ps->buf_slots[i].buf_uop == a) slot_data[i] = da;
      else if (ps->buf_slots[i].buf_uop == b) slot_data[i] = db;
      else if (ps->buf_slots[i].buf_uop == out) slot_data[i] = dout;
    }

    int ret = poly_compiled_plan_run(plan, slot_data, ps->n_buf_slots, NULL, 0);
    ASSERT_INT_EQ(ret, 0);

    /* sum(1..8 * mult) = 36*mult, out[i] = 36*mult + (i+10) */
    float expected_sum = 36.0f * mult;
    for (int i = 0; i < N; i++)
      ASSERT_FLOAT_EQ(dout[i], expected_sum + (float)(i + 10), 1e-3);
  }

  poly_compiled_plan_free(plan);
  poly_schedule_free(ps);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(exec_plan, workspace_reduce_zeroed) {
  /* Verify reduce accumulators are zero at start of each run despite
   * persistent intermediates. Two consecutive reduce runs must each
   * produce correct independent results. */
  PolyCtx *ctx = poly_ctx_new();
  int N = 4;
  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, 1);

  int64_t axes[] = {0};
  PolyUOp *s = poly_reduce_axis(ctx, POLY_OP_ADD, a, axes, 1);
  PolyUOp *st = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out, s, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, st, poly_arg_none());

  PolySchedule *ps = poly_schedule_for(ctx, sink, POLY_MODE_CALL);
  ASSERT_TRUE(ps != NULL);

  PolyCompiledPlan *plan = poly_compile_schedule(ctx, ps, POLY_DEVICE_CPU);
  ASSERT_TRUE(plan != NULL);

  /* Run 1: sum([1,2,3,4]) = 10 */
  float da1[] = {1, 2, 3, 4};
  float dout1 = 0;
  void *slot1[8] = {0};
  for (int i = 0; i < ps->n_buf_slots; i++) {
    if (ps->buf_slots[i].buf_uop == a) slot1[i] = da1;
    else if (ps->buf_slots[i].buf_uop == out) slot1[i] = &dout1;
  }
  ASSERT_INT_EQ(poly_compiled_plan_run(plan, slot1, ps->n_buf_slots, NULL, 0), 0);
  ASSERT_FLOAT_EQ(dout1, 10.0f, 1e-6);

  /* Run 2: sum([10,20,30,40]) = 100, NOT 110 (accumulated from run 1) */
  float da2[] = {10, 20, 30, 40};
  float dout2 = 0;
  void *slot2[8] = {0};
  for (int i = 0; i < ps->n_buf_slots; i++) {
    if (ps->buf_slots[i].buf_uop == a) slot2[i] = da2;
    else if (ps->buf_slots[i].buf_uop == out) slot2[i] = &dout2;
  }
  ASSERT_INT_EQ(poly_compiled_plan_run(plan, slot2, ps->n_buf_slots, NULL, 0), 0);
  ASSERT_FLOAT_EQ(dout2, 100.0f, 1e-5);

  poly_compiled_plan_free(plan);
  poly_schedule_free(ps);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(exec_plan, interp_gated_load_pad_shrink) {
  /* Pad+shrink+expand+reduce: verifies gated INDEX (from move_where_on_load)
   * is correctly handled by the interpreter LOAD. Without the fix, pad guards
   * are ignored and the result is 800 instead of 740. */
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *x_flat = poly_buffer(ctx, POLY_FLOAT32, 75);
  int64_t x_shape[] = {1, 3, 5, 5};
  PolyUOp *x = poly_reshape(ctx, x_flat, x_shape, 4);
  int64_t pad_pairs[][2] = {{0,0},{0,0},{1,1},{1,1}};
  PolyUOp *xp = poly_pad(ctx, x, pad_pairs, 4);
  int64_t s1_pairs[][2] = {{0,1},{0,1},{0,5},{0,5}};
  int64_t s2_pairs[][2] = {{0,1},{0,1},{0,5},{1,6}};
  PolyUOp *s1 = poly_shrink(ctx, xp, s1_pairs, 4);
  PolyUOp *s2 = poly_shrink(ctx, xp, s2_pairs, 4);
  int64_t out_shape[] = {1, 2, 5, 5};
  PolyUOp *e1 = poly_expand(ctx, s1, out_shape, 4);
  PolyUOp *e2 = poly_expand(ctx, s2, out_shape, 4);
  PolyUOp *sum = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, e1, e2, poly_arg_none());
  int64_t red_axes[] = {1, 2, 3};
  PolyUOp *loss = poly_reduce_axis(ctx, POLY_OP_ADD, sum, red_axes, 3);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, 1);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out, loss, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  float x_d[75];
  for (int i = 0; i < 75; i++) x_d[i] = (float)(i + 1);
  float out_cpu[1], out_interp[1];
  PolyUOp *bufs[] = { x_flat, out };
  void *datas[] = { x_d, NULL };

  int rc = cpu_interp_parity(ctx, sink, bufs, datas, 2, out, out_cpu, out_interp, 1, 1e-5f);
  if (rc != 0)
    fprintf(stderr, "  gated_load: cpu=%.1f interp=%.1f rc=%d\n",
            (double)out_cpu[0], (double)out_interp[0], rc);
  ASSERT_INT_EQ(rc, 0);
  ASSERT_FLOAT_EQ(out_cpu[0], 740.0f, 1e-5);

  poly_ctx_destroy(ctx);
  PASS();
}
