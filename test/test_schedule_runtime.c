/*
 * test_schedule_runtime.c -- Tests for the engine/schedule runtime layer
 *
 * Phase 1-2: PolySchedule construction
 * Phase 3:   CPU compiled schedule (lower + run)
 * Phase 4:   Interpreter backend (CPU vs INTERP parity)
 */

#include "test_harness.h"
#include "../src/frontend.h"
#include "../src/frontend_internal.h"
#include "../src/engine/schedule.h"
#include "../src/codegen.h"
#include "../src/schedule/rangeify.h"
#include "../src/tensor.h"
#include "../src/device.h"
#include "../src/utils.h"

#include <stdbool.h>
#include <string.h>

typedef struct {
  const char *key;
  char *value;
  bool had_value;
} ScheduleEnvSave;

static ScheduleEnvSave schedule_save_env(const char *key) {
  const char *cur = getenv(key);
  return (ScheduleEnvSave){
      .key = key,
      .value = cur ? strdup(cur) : NULL,
      .had_value = cur != NULL,
  };
}

static void schedule_restore_env(ScheduleEnvSave *s) {
  if (!s) return;
  if (s->had_value)
    setenv(s->key, s->value ? s->value : "", 1);
  else
    unsetenv(s->key);
  free(s->value);
  s->value = NULL;
}

static int expected_to_program_cache_entries(PolyCtx *ctx, int source_backend_entries) {
  return poly_ctx_get_preferred_device(ctx) == POLY_DEVICE_INTERP ? 0 : source_backend_entries;
}

/* Helper: run same graph on CPU and INTERP, compare outputs */

static int cpu_interp_parity(
    PolyCtx *ctx,
    PolyUOp *sink,
    PolyUOp **bufs,
    void **datas,
    int n_bufs,
    PolyUOp *out_buf,
    float *out_cpu,
    float *out_interp,
    int out_numel,
    float tol
) {
  PolySchedule *ps = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  if (!ps) return -1;

  /* CPU path */
  PolyCompiledSchedule *cpu = poly_lower_schedule(ctx, ps, POLY_DEVICE_CPU);
  if (!cpu) {
    poly_schedule_free(ps);
    return -2;
  }
  memset(out_cpu, 0, (size_t)out_numel * sizeof(float));
  void *slot_cpu[16] = {0};
  for (int i = 0; i < ps->template->n_buf_slots; i++) {
    for (int j = 0; j < n_bufs; j++) {
      if (ps->template->buf_slots[i].buf_uop == bufs[j])
        slot_cpu[i] = (bufs[j] == out_buf) ? out_cpu : datas[j];
    }
  }
  int rc = poly_run_compiled_schedule(cpu, slot_cpu, ps->template->n_buf_slots, NULL, 0);
  poly_compiled_schedule_free(cpu);
  if (rc < 0) {
    poly_schedule_free(ps);
    return -3;
  }

  /* INTERP path */
  PolyCompiledSchedule *interp = poly_lower_schedule(ctx, ps, POLY_DEVICE_INTERP);
  if (!interp) {
    poly_schedule_free(ps);
    return -4;
  }
  memset(out_interp, 0, (size_t)out_numel * sizeof(float));
  void *slot_interp[16] = {0};
  for (int i = 0; i < ps->template->n_buf_slots; i++) {
    for (int j = 0; j < n_bufs; j++) {
      if (ps->template->buf_slots[i].buf_uop == bufs[j])
        slot_interp[i] = (bufs[j] == out_buf) ? out_interp : datas[j];
    }
  }
  rc = poly_run_compiled_schedule(interp, slot_interp, ps->template->n_buf_slots, NULL, 0);
  poly_compiled_schedule_free(interp);
  poly_schedule_free(ps);
  if (rc < 0) return -5;

  /* Compare */
  for (int i = 0; i < out_numel; i++) {
    float diff = out_cpu[i] - out_interp[i];
    if (diff < 0) diff = -diff;
    if (diff > tol) return i + 1; /* 1-based index of first mismatch */
  }
  return 0;
}

typedef struct {
  int store_target_index;
  int target_ptr_index;
  int read_ptr_index;
  int read_scalar_index;
} IndexKindCounts;

static int count_root_ops(PolyCtx *ctx, PolyUOp *root, PolyOps op) {
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, root, &n_topo);
  int count = 0;
  for (int i = 0; i < n_topo; i++) {
    if (topo[i] && topo[i]->op == op) count++;
  }
  return count;
}

static int count_root_ranges_of_type(PolyCtx *ctx, PolyUOp *root, PolyAxisType axis_type) {
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, root, &n_topo);
  int count = 0;
  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    if (!u || u->op != POLY_OP_RANGE || !poly_arg_is_range(u->arg)) continue;
    int64_t bound = 0;
    if (u->n_src > 0 && u->src[0]->op == POLY_OP_CONST && u->src[0]->arg.kind == POLY_ARG_INT)
      bound = u->src[0]->arg.i;
    if (bound <= 1) continue;
    if (poly_range_axis_type(u->arg) == axis_type) count++;
  }
  return count;
}

static PolyUOp *test_call_buffer_arg(PolyUOp *call, int arg_idx) {
  int seen = 0;
  for (int i = 1; call && i < call->n_src; i++) {
    if (call->src[i]->op == POLY_OP_DEFINE_VAR) continue;
    if (seen++ == arg_idx) return call->src[i];
  }
  return NULL;
}

static int count_body_call_arg_overlap(PolyCtx *ctx, PolyUOp *body, PolyUOp *call) {
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, body, &n_topo);
  int count = 0;
  int n_args = 0;
  for (int i = 1; call && i < call->n_src; i++)
    if (call->src[i]->op != POLY_OP_DEFINE_VAR) n_args++;
  for (int i = 0; i < n_topo; i++) {
    for (int a = 0; a < n_args; a++) {
      if (topo[i] == test_call_buffer_arg(call, a)) count++;
    }
  }
  return count;
}

static bool schedule_calls_are_parameterized(PolyCtx *ctx, PolySchedule *sched) {
  if (!sched) return false;
  for (int k = 0; k < sched->template->n_calls; k++) {
    PolyUOp *call = poly_schedule_call(sched, k);
    PolyUOp *body = poly_schedule_call_body(sched, k);
    if (!call || !body || poly_schedule_call_is_copy(sched, k) || body->op == POLY_OP_BUFFER_VIEW)
      continue;
    if (count_body_call_arg_overlap(ctx, body, call) != 0) return false;
    if (count_root_ops(ctx, body, POLY_OP_BUFFER) != 0) return false;
    if (count_root_ops(ctx, body, POLY_OP_PARAM) <= 0) return false;
  }
  return true;
}

TEST(schedule_runtime, compute_call_bodies_are_parameterized_like_tinygrad) {
  ScheduleEnvSave validate_env = schedule_save_env("POLY_VALIDATE_SCHEDULE");
  setenv("POLY_VALIDATE_SCHEDULE", "1", 1);

  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *a = poly_buffer_f32(ctx, 16);
  PolyUOp *b = poly_buffer_f32(ctx, 16);
  PolyUOp *out = poly_buffer_f32(ctx, 16);
  PolyUOp *add = poly_alu2(ctx, POLY_OP_ADD, a, b);
  PolySchedule *add_sched =
      poly_complete_create_schedule_with_vars(ctx, poly_sink1(ctx, poly_store_val(ctx, out, add)), POLY_MODE_CALL);
  ASSERT_TRUE(schedule_calls_are_parameterized(ctx, add_sched));
  poly_schedule_free(add_sched);

  PolyUOp *rout = poly_buffer_f32(ctx, 1);
  int64_t axes[1] = {0};
  PolyUOp *sum = poly_reduce_axis(ctx, POLY_OP_ADD, a, axes, 1);
  PolySchedule *reduce_sched =
      poly_complete_create_schedule_with_vars(ctx, poly_sink1(ctx, poly_store_val(ctx, rout, sum)), POLY_MODE_CALL);
  ASSERT_TRUE(schedule_calls_are_parameterized(ctx, reduce_sched));
  poly_schedule_free(reduce_sched);

  PolyUOp *base = poly_buffer_f32(ctx, 8);
  PolyUOp *unique = poly_uop0(ctx, POLY_OP_UNIQUE, POLY_VOID, poly_arg_int(9001));
  int64_t view_shape[1] = {8};
  PolyArg view_arg = {.kind = POLY_ARG_INT_TUPLE, .int_tuple = {view_shape, 1}};
  PolyUOp *view_src[2] = {base, unique};
  PolyUOp *view = poly_uop(ctx, POLY_OP_BUFFER_VIEW, POLY_FLOAT32, view_src, 2, view_arg);
  PolyUOp *view_out = poly_buffer_f32(ctx, 8);
  PolySchedule *view_sched = poly_complete_create_schedule_with_vars(
      ctx, poly_sink1(ctx, poly_store_val(ctx, view_out, poly_alu2(ctx, POLY_OP_ADD, view, b))),
      POLY_MODE_CALL
  );
  ASSERT_TRUE(schedule_calls_are_parameterized(ctx, view_sched));
  ASSERT_TRUE(count_root_ops(ctx, poly_schedule_call_body(view_sched, 0), POLY_OP_BUFFER_VIEW) > 0);
  poly_schedule_free(view_sched);

  poly_ctx_destroy(ctx);
  schedule_restore_env(&validate_env);
  PASS();
}

TEST(schedule_runtime, programinfo_uses_filtered_call_buffer_arg_indices) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *out = poly_buffer_f32(ctx, 1);
  PolyUOp *a = poly_buffer_f32(ctx, 1);
  PolyUOp *b = poly_buffer_f32(ctx, 1);
  PolyUOp *var = poly_define_var(ctx, "N", 1, 8);
  ASSERT_NOT_NULL(var);

  PolyUOp *p0 = poly_uop0(ctx, POLY_OP_PARAM, POLY_FLOAT32, poly_arg_int(0));
  PolyUOp *p2 = poly_uop0(ctx, POLY_OP_PARAM, POLY_FLOAT32, poly_arg_int(2));
  PolyUOp *sum = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, p0, p2, poly_arg_none());
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, p0, sum));

  PolyUOp *src[] = {sink, out, var, a, b};
  PolyUOp *call = poly_uop(ctx, POLY_OP_CALL, POLY_VOID, src, 5, poly_arg_none());
  PolyUOp *program = poly_program_from_call(ctx, call, "filtered_args");
  ASSERT_NOT_NULL(program);
  ASSERT_INT_EQ(program->op, POLY_OP_PROGRAM);

  const PolyProgramInfo *info = poly_program_info(ctx, program);
  ASSERT_NOT_NULL(info);
  ASSERT_STR_EQ(info->name, "filtered_args");
  ASSERT_INT_EQ(info->n_globals, 2);
  ASSERT_INT_EQ(info->globals[0], 0);
  ASSERT_INT_EQ(info->globals[1], 2);
  ASSERT_INT_EQ(info->n_vars, 1);
  ASSERT_PTR_EQ(info->vars[0], var);

  ASSERT_INT_EQ(info->n_outs, 1);
  ASSERT_INT_EQ(info->outs[0], 0);
  ASSERT_INT_EQ(info->n_ins, 2);
  ASSERT_INT_EQ(info->ins[0], 0);
  ASSERT_INT_EQ(info->ins[1], 2);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(schedule_runtime, programinfo_metadata_survives_program_cse_reuse) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *out = poly_buffer_f32(ctx, 1);
  PolyUOp *x = poly_buffer_f32(ctx, 1);
  PolyUOp *p0 = poly_uop0(ctx, POLY_OP_PARAM, POLY_FLOAT32, poly_arg_int(0));
  PolyUOp *p1 = poly_uop0(ctx, POLY_OP_PARAM, POLY_FLOAT32, poly_arg_int(1));
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, p0, p1));
  PolyUOp *src[] = {sink, out, x};
  PolyUOp *call = poly_uop(ctx, POLY_OP_CALL, POLY_VOID, src, 3, poly_arg_none());

  PolyUOp *p_a = poly_program_from_call(ctx, call, "same_program");
  PolyUOp *p_b = poly_program_from_call(ctx, call, "same_program");
  ASSERT_PTR_EQ(p_a, p_b);

  const PolyProgramInfo *info = poly_program_info(ctx, p_b);
  ASSERT_NOT_NULL(info);
  ASSERT_INT_EQ(info->n_globals, 2);
  ASSERT_INT_EQ(info->n_outs, 1);
  ASSERT_INT_EQ(info->outs[0], 0);
  ASSERT_INT_EQ(info->n_ins, 1);
  ASSERT_INT_EQ(info->ins[0], 1);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(schedule_runtime, programinfo_collects_special_launch_dims) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *out = poly_buffer_f32(ctx, 128);
  PolyUOp *x = poly_buffer_f32(ctx, 128);
  PolyUOp *p0 = poly_uop0(ctx, POLY_OP_PARAM, POLY_FLOAT32, poly_arg_int(0));
  PolyUOp *p1 = poly_uop0(ctx, POLY_OP_PARAM, POLY_FLOAT32, poly_arg_int(1));
  PolyUOp *g_bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(128));
  PolyUOp *l_bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(16));
  PolyUOp *gidx = poly_uop1(ctx, POLY_OP_SPECIAL, POLY_INT32, g_bound, poly_arg_str("gidx0"));
  PolyUOp *lidx = poly_uop1(ctx, POLY_OP_SPECIAL, POLY_INT32, l_bound, poly_arg_str("lidx0"));
  PolyUOp *idx = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, gidx, lidx, poly_arg_none());
  PolyUOp *dst = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, p0, idx, poly_arg_none());
  PolyUOp *src = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, p1, idx, poly_arg_none());
  PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, src, poly_arg_none());
  PolyUOp *sink = poly_sink1(ctx, poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, dst, load, poly_arg_none()));
  PolyUOp *call_src[] = {sink, out, x};
  PolyUOp *call = poly_uop(ctx, POLY_OP_CALL, POLY_VOID, call_src, 3, poly_arg_none());

  PolyUOp *program = poly_program_from_call(ctx, call, "launch_dims");
  ASSERT_NOT_NULL(program);
  const PolyProgramInfo *info = poly_program_info(ctx, program);
  ASSERT_NOT_NULL(info);
  ASSERT_INT_EQ(info->global_size[0], 128);
  ASSERT_INT_EQ(info->global_size[1], 1);
  ASSERT_INT_EQ(info->global_size[2], 1);
  ASSERT_TRUE(info->has_local_size);
  ASSERT_INT_EQ(info->local_size[0], 16);
  ASSERT_INT_EQ(info->local_size[1], 1);
  ASSERT_INT_EQ(info->local_size[2], 1);
  ASSERT_PTR_EQ(info->global_exprs[0], g_bound);
  ASSERT_PTR_EQ(info->local_exprs[0], l_bound);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(schedule_runtime, compute_schedule_calls_are_program_backed) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *a = poly_buffer_f32(ctx, 4);
  PolyUOp *b = poly_buffer_f32(ctx, 4);
  PolyUOp *out = poly_buffer_f32(ctx, 4);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, poly_alu2(ctx, POLY_OP_ADD, a, b)));

  PolySchedule *sched = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_NOT_NULL(sched);
  ASSERT_INT_EQ(sched->template->n_calls, 1);

  PolyUOp *call = poly_schedule_call(sched, 0);
  ASSERT_NOT_NULL(call);
  ASSERT_INT_EQ(call->op, POLY_OP_CALL);
  ASSERT_TRUE(call->n_src >= 1);
  ASSERT_NOT_NULL(call->src[0]);
  ASSERT_INT_EQ(call->src[0]->op, POLY_OP_PROGRAM);
  ASSERT_NOT_NULL(poly_program_info(ctx, call->src[0]));
  ASSERT_INT_EQ(poly_schedule_call_body(sched, 0)->op, POLY_OP_SINK);

  poly_schedule_free(sched);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(schedule_runtime, compute_call_lower_rejects_raw_sink_body) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *a = poly_buffer_f32(ctx, 4);
  PolyUOp *b = poly_buffer_f32(ctx, 4);
  PolyUOp *out = poly_buffer_f32(ctx, 4);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, poly_alu2(ctx, POLY_OP_ADD, a, b)));

  PolySchedule *sched = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_NOT_NULL(sched);
  ASSERT_INT_EQ(sched->template->n_calls, 1);

  PolyUOp *call = poly_schedule_call(sched, 0);
  ASSERT_NOT_NULL(call);
  ASSERT_TRUE(call->n_src >= 1);
  ASSERT_INT_EQ(call->src[0]->op, POLY_OP_PROGRAM);

  PolyUOp *body = poly_schedule_call_body(sched, 0);
  ASSERT_NOT_NULL(body);
  ASSERT_INT_EQ(body->op, POLY_OP_SINK);

  PolyUOp **src = malloc((size_t)call->n_src * sizeof(PolyUOp *));
  ASSERT_NOT_NULL(src);
  memcpy(src, call->src, (size_t)call->n_src * sizeof(PolyUOp *));
  src[0] = body;
  PolyUOp *raw_call = poly_uop(ctx, POLY_OP_CALL, POLY_VOID, src, call->n_src, poly_arg_none());
  free(src);
  ASSERT_NOT_NULL(raw_call);

  sched->template->linear->src[0] = raw_call;
  sched->run->calls[0].call = raw_call;
  ASSERT_INT_EQ(poly_schedule_call_lower(ctx, sched, 0, POLY_DEVICE_CPU), -1);

  poly_schedule_free(sched);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(schedule_runtime, to_program_attaches_linear_and_source_children) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  ASSERT_INT_EQ((int)poly_to_program_cache_len(ctx), 0);
  poly_program_source_render_count_reset();

  PolyUOp *a = poly_buffer_f32(ctx, 4);
  PolyUOp *b = poly_buffer_f32(ctx, 4);
  PolyUOp *out = poly_buffer_f32(ctx, 4);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, poly_alu2(ctx, POLY_OP_ADD, a, b)));

  PolySchedule *sched = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_NOT_NULL(sched);
  ASSERT_INT_EQ(sched->template->n_calls, 1);

  PolyUOp *program = poly_schedule_call_to_program(ctx, sched, 0, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(program);
  ASSERT_INT_EQ(program->op, POLY_OP_PROGRAM);
  ASSERT_INT_EQ(program->arg.kind, POLY_ARG_PROGRAM_INFO);
  ASSERT_TRUE(program->n_src >= 3);
  ASSERT_NOT_NULL(poly_program_info(ctx, program));

  PolyUOp *linear = poly_program_linear(program);
  ASSERT_NOT_NULL(linear);
  ASSERT_INT_EQ(linear->op, POLY_OP_LINEAR);
  ASSERT_TRUE(linear->n_src > 0);
  ASSERT_TRUE(program->n_src >= 4);
  ASSERT_NOT_NULL(program->src[3]);
  ASSERT_INT_EQ(program->src[3]->op, POLY_OP_SOURCE);
  ASSERT_INT_EQ(program->src[3]->arg.kind, POLY_ARG_STRING);
  ASSERT_NOT_NULL(program->src[3]->arg.str);
  ASSERT_TRUE(strstr(program->src[3]->arg.str, "poly_k_") != NULL);
  ASSERT_INT_EQ(poly_program_source_render_count(), 1);

  PolyUOp *again = poly_schedule_call_to_program(ctx, sched, 0, POLY_DEVICE_CPU);
  ASSERT_PTR_EQ(again, program);
  ASSERT_INT_EQ((int)poly_to_program_cache_len(ctx), 1);
  ASSERT_INT_EQ(poly_program_source_render_count(), 1);

  poly_schedule_free(sched);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(schedule_runtime, to_program_cache_reuses_source_across_fresh_schedules) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  poly_program_source_render_count_reset();

  PolyUOp *a1 = poly_buffer_f32(ctx, 4);
  PolyUOp *b1 = poly_buffer_f32(ctx, 4);
  PolyUOp *out1 = poly_buffer_f32(ctx, 4);
  PolyUOp *sink1 = poly_sink1(ctx, poly_store_val(ctx, out1, poly_alu2(ctx, POLY_OP_ADD, a1, b1)));
  PolySchedule *sched1 = poly_complete_create_schedule_with_vars(ctx, sink1, POLY_MODE_CALL);
  ASSERT_NOT_NULL(sched1);
  ASSERT_INT_EQ(sched1->template->n_calls, 1);

  PolyUOp *program1 = poly_schedule_call_to_program(ctx, sched1, 0, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(program1);
  ASSERT_TRUE(program1->n_src >= 4);
  ASSERT_INT_EQ(program1->src[3]->op, POLY_OP_SOURCE);
  ASSERT_INT_EQ(poly_program_source_render_count(), 1);

  PolyUOp *a2 = poly_buffer_f32(ctx, 4);
  PolyUOp *b2 = poly_buffer_f32(ctx, 4);
  PolyUOp *out2 = poly_buffer_f32(ctx, 4);
  PolyUOp *sink2 = poly_sink1(ctx, poly_store_val(ctx, out2, poly_alu2(ctx, POLY_OP_ADD, a2, b2)));
  PolySchedule *sched2 = poly_complete_create_schedule_with_vars(ctx, sink2, POLY_MODE_CALL);
  ASSERT_NOT_NULL(sched2);
  ASSERT_INT_EQ(sched2->template->n_calls, 1);

  PolyUOp *program2 = poly_schedule_call_to_program(ctx, sched2, 0, POLY_DEVICE_CPU);
  ASSERT_PTR_EQ(program2, program1);
  ASSERT_INT_EQ((int)poly_to_program_cache_len(ctx), 1);
  ASSERT_INT_EQ(poly_program_source_render_count(), 1);

  poly_schedule_free(sched2);
  poly_schedule_free(sched1);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(schedule_runtime, runner_launch_uses_programinfo_metadata) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *a = poly_buffer_f32(ctx, 4);
  PolyUOp *b = poly_buffer_f32(ctx, 4);
  PolyUOp *out = poly_buffer_f32(ctx, 4);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, poly_alu2(ctx, POLY_OP_ADD, a, b)));

  PolySchedule *sched = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_NOT_NULL(sched);
  ASSERT_INT_EQ(sched->template->n_calls, 1);
  PolyUOp *call = poly_schedule_call(sched, 0);
  ASSERT_NOT_NULL(call);
  ASSERT_TRUE(call->n_src >= 1);
  ASSERT_INT_EQ(call->src[0]->op, POLY_OP_PROGRAM);

  PolyUOp *program = poly_schedule_call_to_program(ctx, sched, 0, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(program);
  ASSERT_INT_EQ(program->op, POLY_OP_PROGRAM);
  PolyProgramInfo *info = (PolyProgramInfo *)poly_program_info(ctx, program);
  ASSERT_NOT_NULL(info);
  PolyUOp *g_bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(7));
  PolyUOp *l_bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(3));
  info->global_size[0] = 7;
  info->local_size[0] = 3;
  info->global_exprs[0] = g_bound;
  info->local_exprs[0] = l_bound;
  info->has_local_size = true;

  ASSERT_INT_EQ(poly_schedule_call_lower(ctx, sched, 0, POLY_DEVICE_CPU), 0);
  PolyRunner *runner = &sched->run->calls[0].prg;
  ASSERT_INT_EQ(runner->grid[0], 7);
  ASSERT_INT_EQ(runner->grid[1], 1);
  ASSERT_INT_EQ(runner->grid[2], 1);
  ASSERT_INT_EQ(runner->block[0], 3);
  ASSERT_INT_EQ(runner->block[1], 1);
  ASSERT_INT_EQ(runner->block[2], 1);
  ASSERT_PTR_EQ(runner->grid_exprs[0], g_bound);
  ASSERT_PTR_EQ(runner->block_exprs[0], l_bound);

  poly_schedule_free(sched);
  poly_ctx_destroy(ctx);
  PASS();
}

static PolyUOp *make_validator_comb_graph(PolyCtx *ctx, int depth, PolyUOp **bad_node) {
  PolyUOp *srcs[64];
  for (int i = 0; i < 64; i++)
    srcs[i] = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(i));

  PolyUOp *cur = poly_uop(ctx, POLY_OP_SINK, POLY_VOID, srcs, 64, poly_arg_none());
  if (bad_node) *bad_node = cur;

  for (int d = 0; d < depth; d++) {
    for (int i = 0; i < 63; i++)
      srcs[i] = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(d * 63 + i + 64));
    srcs[63] = cur;
    cur = poly_uop(ctx, POLY_OP_SINK, POLY_VOID, srcs, 64, poly_arg_none());
  }
  return cur;
}

static int count_root_gated_loads(PolyCtx *ctx, PolyUOp *root) {
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, root, &n_topo);
  int count = 0;
  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    if (!u || u->op != POLY_OP_LOAD || u->n_src < 1) continue;
    PolyUOp *idx = poly_find_index_through_cast(u->src[0]);
    if (idx && idx->op == POLY_OP_INDEX && idx->n_src >= 3) count++;
  }
  return count;
}

/* Mirror the tinygrad late WebGPU stage audit on a single triangular-mask root.
 * This keeps triu/tril checks on the exact same stage boundaries. */
static PolySchedule *build_tri_schedule(PolyCtx *ctx, int n, bool lower) {
  PolyUOp *in = poly_buffer_f32(ctx, n * n);
  PolyUOp *in2d = poly_reshape(ctx, in, (int64_t[]){n, n}, 2);
  PolyUOp *tri = lower ? poly_tril(ctx, in2d, 0) : poly_triu(ctx, in2d, 0);
  if (!tri) return NULL;
  PolyUOp *out = poly_buffer_f32(ctx, n * n);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, tri));
  return poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
}

static PolyUOp *apply_webgpu_tri_stage_root(
    PolyCtx *ctx,
    PolyUOp *root,
    bool with_add_loads,
    bool with_post_index_symbolic
) {
  PolyRendererCaps caps = {
      .has_mulacc = false,
      .has_threefry = false,
      .has_local = true,
      .max_vec_width = 1,
  };

  PolyUOp *u = poly_apply_opts_heuristic_ex(ctx, root, caps);
  u = poly_graph_rewrite(ctx, u, poly_symbolic());
  u = poly_graph_rewrite(ctx, u, poly_pm_move_where_on_load_pass());
  u = poly_graph_rewrite(ctx, u, poly_symbolic());
  u = poly_graph_rewrite(ctx, u, poly_pm_pre_expander_pass());
  u = poly_group_for_reduce(ctx, u, 256);
  u = poly_graph_rewrite(ctx, u, poly_pm_expander_pass());
  u = poly_graph_rewrite(ctx, u, poly_symbolic());
  u = poly_apply_pm_reduce(ctx, u);
  u = poly_add_gpudims(ctx, u);
  if (with_add_loads) u = poly_graph_rewrite(ctx, u, poly_pm_add_loads_pass());
  if (with_post_index_symbolic) {
    u = poly_apply_devectorize_stage(ctx, u, 1, caps);
    u = poly_apply_post_index_symbolic_stage(ctx, u, 1);
  }
  return u;
}

static IndexKindCounts count_index_kinds(PolyCtx *ctx, PolyUOp *root) {
  IndexKindCounts c = {0};
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, root, &n_topo);
  PolyUOp *store_targets[128] = {0};
  int n_store_targets = 0;

  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    if (!u || u->op != POLY_OP_STORE || u->n_src < 1) continue;
    if (u->src[0] && u->src[0]->op == POLY_OP_INDEX && n_store_targets < 128)
      store_targets[n_store_targets++] = u->src[0];
  }

  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    if (!u || u->op != POLY_OP_INDEX) continue;
    bool is_target = false;
    for (int j = 0; j < n_store_targets; j++) {
      if (store_targets[j] == u) {
        is_target = true;
        break;
      }
    }
    if (is_target) {
      c.store_target_index++;
      if (u->dtype.is_ptr) c.target_ptr_index++;
    } else if (u->dtype.is_ptr) {
      c.read_ptr_index++;
    } else {
      c.read_scalar_index++;
    }
  }

  return c;
}

/* Single-kernel: vecadd */

TEST(schedule_runtime, prepare_vecadd) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer_f32(ctx, 4);
  PolyUOp *b = poly_buffer_f32(ctx, 4);
  PolyUOp *c = poly_alu2(ctx, POLY_OP_ADD, a, b);
  PolyUOp *out = poly_buffer_f32(ctx, 4);
  PolyUOp *st = poly_store_val(ctx, out, c);
  PolyUOp *sink = poly_sink1(ctx, st);

  PolySchedule *ps = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_TRUE(ps != NULL);

  /* Single kernel */
  ASSERT_INT_EQ(ps->template->n_calls, 1);
  ASSERT_FALSE(poly_schedule_call_is_copy(ps, 0));
  ASSERT_TRUE(poly_schedule_call_body(ps, 0) != NULL);

  /* 3 external buffers (a, b, out), 0 intermediates */
  ASSERT_INT_EQ(ps->template->n_buf_slots, 3);
  for (int i = 0; i < 3; i++) {
    ASSERT_FALSE(ps->template->buf_slots[i].is_intermediate);
    ASSERT_INT_EQ(ps->template->buf_slots[i].numel, 4);
    ASSERT_INT_EQ(ps->template->buf_slots[i].external_buf_idx, i);
  }

  ASSERT_NOT_NULL(ps->template->linear);
  ASSERT_INT_EQ(ps->template->linear->n_src, 1);

  /* Mode and defaults */
  ASSERT_EQ(ps->template->mode, POLY_MODE_CALL);
  ASSERT_INT_EQ(ps->template->loss_buf_slot, -1);

  /* Kernel params should reference buf slots */
  ASSERT_TRUE(poly_schedule_call_n_buffer_args(ps, 0) > 0);
  for (int i = 0; i < poly_schedule_call_n_buffer_args(ps, 0); i++)
    ASSERT_TRUE(poly_schedule_call_buffer_slot(ps, 0, i) >= 0);

  poly_schedule_free(ps);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(schedule_runtime, create_schedule_matches_complete_schedule_vecadd) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer_f32(ctx, 8);
  PolyUOp *b = poly_buffer_f32(ctx, 8);
  PolyUOp *c = poly_alu2(ctx, POLY_OP_ADD, a, b);
  PolyUOp *out = poly_buffer_f32(ctx, 8);
  PolyUOp *st = poly_store_val(ctx, out, c);
  PolyUOp *sink = poly_sink1(ctx, st);

  PolySchedule *from_sink = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  PolyUOp *kernel_graph = poly_get_kernel_graph(ctx, sink);
  PolySchedule *from_kernel_graph = poly_create_schedule(ctx, kernel_graph);

  ASSERT_NOT_NULL(from_sink);
  ASSERT_NOT_NULL(kernel_graph);
  ASSERT_NOT_NULL(from_kernel_graph);
  ASSERT_INT_EQ(from_sink->template->n_calls, 1);
  ASSERT_INT_EQ(from_kernel_graph->template->n_calls, 1);
  ASSERT_TRUE(poly_structural_eq(
      poly_schedule_call_body(from_sink, 0), poly_schedule_call_body(from_kernel_graph, 0)
  ));

  int n_sink_lin = 0, n_graph_lin = 0;
  PolyUOp **sink_lin = poly_linearize(ctx, poly_schedule_call_body(from_sink, 0), &n_sink_lin);
  PolyUOp **graph_lin =
      poly_linearize(ctx, poly_schedule_call_body(from_kernel_graph, 0), &n_graph_lin);
  ASSERT_NOT_NULL(sink_lin);
  ASSERT_NOT_NULL(graph_lin);
  ASSERT_INT_EQ(n_sink_lin, n_graph_lin);

  int sink_end = 0, graph_end = 0;
  for (int i = 0; i < n_sink_lin; i++)
    if (sink_lin[i]->op == POLY_OP_END) sink_end++;
  for (int i = 0; i < n_graph_lin; i++)
    if (graph_lin[i]->op == POLY_OP_END) graph_end++;
  ASSERT_INT_EQ(sink_end, graph_end);

  free(sink_lin);
  free(graph_lin);
  poly_schedule_free(from_sink);
  poly_schedule_free(from_kernel_graph);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(schedule_runtime, validate_kernel_graph_grows_past_old_stack_cap) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *bad = NULL;
  PolyUOp *root = make_validator_comb_graph(ctx, 80, &bad);

  ASSERT_NOT_NULL(root);
  ASSERT_TRUE(bad != NULL && bad->n_src > 0);
  ASSERT_TRUE(poly_validate_kernel_graph(ctx, root));

  bad->src[0] = NULL;
  ASSERT_TRUE(!poly_validate_kernel_graph(ctx, root));

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(schedule_runtime, lower_sink_to_linear_matches_create_schedule_vecadd) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer_f32(ctx, 8);
  PolyUOp *b = poly_buffer_f32(ctx, 8);
  PolyUOp *c = poly_alu2(ctx, POLY_OP_ADD, a, b);
  PolyUOp *out = poly_buffer_f32(ctx, 8);
  PolyUOp *st = poly_store_val(ctx, out, c);
  PolyUOp *sink = poly_sink1(ctx, st);

  PolyUOp *kernel_graph = poly_get_kernel_graph(ctx, sink);
  PolySchedule *schedule = poly_create_schedule(ctx, kernel_graph);
  PolyUOp *linear = poly_lower_sink_to_linear(ctx, sink, POLY_MODE_CALL);

  ASSERT_NOT_NULL(kernel_graph);
  ASSERT_NOT_NULL(schedule);
  ASSERT_NOT_NULL(linear);
  ASSERT_EQ(linear->op, POLY_OP_LINEAR);
  ASSERT_INT_EQ(linear->n_src, schedule->template->n_calls);

  for (int step = 0; step < schedule->template->n_calls; step++) {
    PolyUOp *linear_item = linear->src[step];
    /* Cached LINEAR mirrors tinygrad's callified form: each entry is a CALL
     * carrying the reusable kernel root plus its parameter order. */
    if (linear_item && linear_item->op == POLY_OP_CALL && linear_item->n_src >= 1)
      linear_item = linear_item->src[0];
    if (linear_item && linear_item->op == POLY_OP_PROGRAM && linear_item->n_src >= 1)
      linear_item = linear_item->src[0];
    ASSERT_TRUE(poly_structural_eq(linear_item, poly_schedule_call_body(schedule, step)));
  }

  poly_schedule_free(schedule);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(schedule_runtime, lower_sink_to_linear_caches_structural_linear) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  ASSERT_INT_EQ((int)poly_schedule_cache_len(ctx), 0);

  PolyUOp *a1 = poly_buffer_f32(ctx, 8);
  PolyUOp *b1 = poly_buffer_f32(ctx, 8);
  PolyUOp *out1 = poly_buffer_f32(ctx, 8);
  PolyUOp *sink1 = poly_sink1(ctx, poly_store_val(ctx, out1, poly_alu2(ctx, POLY_OP_ADD, a1, b1)));

  PolyUOp *linear1 = poly_lower_sink_to_linear(ctx, sink1, POLY_MODE_CALL);
  ASSERT_NOT_NULL(linear1);
  ASSERT_EQ(linear1->op, POLY_OP_LINEAR);
  ASSERT_INT_EQ((int)poly_schedule_cache_len(ctx), 1);

  PolyUOp *a2 = poly_buffer_f32(ctx, 8);
  PolyUOp *b2 = poly_buffer_f32(ctx, 8);
  PolyUOp *out2 = poly_buffer_f32(ctx, 8);
  PolyUOp *sink2 = poly_sink1(ctx, poly_store_val(ctx, out2, poly_alu2(ctx, POLY_OP_ADD, a2, b2)));

  PolyUOp *linear2 = poly_lower_sink_to_linear(ctx, sink2, POLY_MODE_CALL);
  ASSERT_NOT_NULL(linear2);
  ASSERT_PTR_EQ(linear1, linear2);
  ASSERT_INT_EQ((int)poly_schedule_cache_len(ctx), 1);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(schedule_runtime, schedule_cache_key_separates_buffer_device) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  ASSERT_INT_EQ((int)poly_schedule_cache_len(ctx), 0);

  PolyUOp *a_cpu = poly_buffer_on_device(ctx, POLY_FLOAT32, 8, POLY_DEVICE_CPU);
  PolyUOp *b_cpu = poly_buffer_on_device(ctx, POLY_FLOAT32, 8, POLY_DEVICE_CPU);
  PolyUOp *out_cpu = poly_buffer_on_device(ctx, POLY_FLOAT32, 8, POLY_DEVICE_CPU);
  PolyUOp *sink_cpu =
      poly_sink1(ctx, poly_store_val(ctx, out_cpu, poly_alu2(ctx, POLY_OP_ADD, a_cpu, b_cpu)));

  PolyUOp *linear_cpu = poly_lower_sink_to_linear(ctx, sink_cpu, POLY_MODE_CALL);
  ASSERT_NOT_NULL(linear_cpu);
  ASSERT_INT_EQ((int)poly_schedule_cache_len(ctx), 1);

  PolyUOp *a_cuda = poly_buffer_on_device(ctx, POLY_FLOAT32, 8, POLY_DEVICE_CUDA);
  PolyUOp *b_cuda = poly_buffer_on_device(ctx, POLY_FLOAT32, 8, POLY_DEVICE_CUDA);
  PolyUOp *out_cuda = poly_buffer_on_device(ctx, POLY_FLOAT32, 8, POLY_DEVICE_CUDA);
  PolyUOp *sink_cuda =
      poly_sink1(ctx, poly_store_val(ctx, out_cuda, poly_alu2(ctx, POLY_OP_ADD, a_cuda, b_cuda)));

  PolyUOp *linear_cuda = poly_lower_sink_to_linear(ctx, sink_cuda, POLY_MODE_CALL);
  ASSERT_NOT_NULL(linear_cuda);
  (void)linear_cpu;
  ASSERT_INT_EQ((int)poly_schedule_cache_len(ctx), 2);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(schedule_runtime, structural_helpers_handle_model_scale_graphs) {
  PolyCtx *ctx1 = poly_ctx_new();
  PolyCtx *ctx2 = poly_ctx_new();
  ASSERT_NOT_NULL(ctx1);
  ASSERT_NOT_NULL(ctx2);

  const int n = POLY_MAX_STRUCT_NODES + 512;
  PolyUOp **src1 = malloc((size_t)n * sizeof(PolyUOp *));
  PolyUOp **src2 = malloc((size_t)n * sizeof(PolyUOp *));
  ASSERT_NOT_NULL(src1);
  ASSERT_NOT_NULL(src2);

  for (int i = 0; i < n; i++) {
    src1[i] = poly_buffer_f32(ctx1, i + 1);
    src2[i] = poly_buffer_f32(ctx2, i + 1);
  }

  PolyUOp *sink1 = poly_uop(ctx1, POLY_OP_SINK, POLY_VOID, src1, n, poly_arg_none());
  PolyUOp *sink2 = poly_uop(ctx2, POLY_OP_SINK, POLY_VOID, src2, n, poly_arg_none());
  ASSERT_NOT_NULL(sink1);
  ASSERT_NOT_NULL(sink2);

  uint32_t h1 = poly_structural_hash(sink1);
  uint32_t h2 = poly_structural_hash(sink2);
  ASSERT_TRUE(h1 != 0);
  ASSERT_INT_EQ((int)h1, (int)h2);
  ASSERT_TRUE(poly_structural_eq(sink1, sink2));

  PolyUOp *buf_order[POLY_MAX_REALIZE_BUFS];
  PolyUOp *visited[POLY_MAX_STRUCT_NODES];
  int n_bufs = 0, n_visited = 0;
  poly_collect_buf_order(sink1, buf_order, &n_bufs, visited, &n_visited);
  ASSERT_INT_EQ(n_bufs, n);
  ASSERT_TRUE(n_visited > POLY_MAX_STRUCT_NODES);

  free(src1);
  free(src2);
  poly_ctx_destroy(ctx1);
  poly_ctx_destroy(ctx2);
  PASS();
}

TEST(schedule_runtime, buffer_order_alloc_collects_past_old_realize_cap) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  const int n = POLY_MAX_REALIZE_BUFS + 9;
  PolyUOp **src = malloc((size_t)n * sizeof(PolyUOp *));
  ASSERT_NOT_NULL(src);
  for (int i = 0; i < n; i++)
    src[i] = poly_buffer_f32(ctx, i + 1);

  PolyUOp *sink = poly_uop(ctx, POLY_OP_SINK, POLY_VOID, src, n, poly_arg_none());
  ASSERT_NOT_NULL(sink);

  PolyUOp **buf_order = NULL;
  int n_bufs = 0, n_visited = 0;
  ASSERT_TRUE(poly_collect_buf_order_alloc(sink, &buf_order, &n_bufs, &n_visited));
  ASSERT_INT_EQ(n_bufs, n);
  ASSERT_TRUE(n_visited >= n);
  for (int i = 0; i < n; i++)
    ASSERT_PTR_EQ(buf_order[i], src[i]);

  free(buf_order);
  free(src);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(schedule_runtime, lower_sink_to_linear_handles_external_buffers_past_old_cap) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  const int n_inputs = POLY_MAX_REALIZE_BUFS + 8;
  PolyUOp **inputs = malloc((size_t)n_inputs * sizeof(PolyUOp *));
  ASSERT_NOT_NULL(inputs);
  for (int i = 0; i < n_inputs; i++)
    inputs[i] = poly_buffer_f32(ctx, 1);

  PolyUOp *acc = inputs[0];
  for (int i = 1; i < n_inputs; i++)
    acc = poly_alu2(ctx, POLY_OP_ADD, acc, inputs[i]);
  PolyUOp *out = poly_buffer_f32(ctx, 1);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, acc));

  PolyUOp *linear = poly_lower_sink_to_linear(ctx, sink, POLY_MODE_CALL);
  ASSERT_NOT_NULL(linear);
  ASSERT_INT_EQ(linear->op, POLY_OP_LINEAR);

  free(inputs);
  poly_ctx_destroy(ctx);
  PASS();
}

static PolyUOp *make_bufferview_cache_sink(PolyCtx *ctx, int64_t tag) {
  PolyUOp *base = poly_buffer(ctx, POLY_FLOAT32, 8);
  PolyUOp *unique = poly_uop0(ctx, POLY_OP_UNIQUE, POLY_VOID, poly_arg_int(tag));
  int64_t view_shape[1] = {8};
  PolyArg view_arg = {.kind = POLY_ARG_INT_TUPLE, .int_tuple = {view_shape, 1}};
  PolyUOp *view_src[2] = {base, unique};
  PolyUOp *view = poly_uop(ctx, POLY_OP_BUFFER_VIEW, POLY_FLOAT32, view_src, 2, view_arg);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, 8);
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(1.0));
  PolyUOp *add = poly_alu2(ctx, POLY_OP_ADD, view, one);
  return poly_sink1(ctx, poly_store_val(ctx, out, add));
}

static PolySchedule *make_manual_bufferview_call_schedule_ex(
    PolyCtx *ctx,
    PolyUOp **base_out,
    PolyUOp **view_out,
    bool offset_view
) {
  PolyUOp *base = poly_buffer(ctx, POLY_FLOAT32, 4);
  PolyUOp *unique = poly_uop0(ctx, POLY_OP_UNIQUE, POLY_VOID, poly_arg_int(7001));
  static int64_t full_view_shape[1] = {4};
  static int64_t offset_view_shape[2] = {2, (int64_t)sizeof(float)};
  int64_t *view_shape = offset_view ? offset_view_shape : full_view_shape;
  int view_arg_n = offset_view ? 2 : 1;
  int64_t view_numel = offset_view ? 2 : 4;
  PolyArg view_arg = {.kind = POLY_ARG_INT_TUPLE, .int_tuple = {view_shape, view_arg_n}};
  PolyUOp *view_src[2] = {base, unique};
  PolyUOp *view = poly_uop(ctx, POLY_OP_BUFFER_VIEW, POLY_FLOAT32, view_src, 2, view_arg);
  PolyUOp *call_src[3] = {view, view, base};
  PolyUOp *call = poly_uop(ctx, POLY_OP_CALL, POLY_VOID, call_src, 3, poly_arg_none());
  PolyUOp *linear = poly_uop1(ctx, POLY_OP_LINEAR, POLY_VOID, call, poly_arg_none());
  if (!base || !unique || !view || !call || !linear) return NULL;

  PolySchedule *sched = calloc(1, sizeof(PolySchedule));
  if (!sched) return NULL;
  sched->template = calloc(1, sizeof(PolyScheduleTemplate));
  sched->run = calloc(1, sizeof(PolyScheduleRuntime));
  if (!sched->template || !sched->run) return sched;

  sched->template->refcount = 1;
  sched->template->linear = linear;
  sched->template->n_calls = 1;
  sched->template->n_buf_slots = 2;
  sched->template->buf_slots = calloc(2, sizeof(PolyScheduleBufSlot));
  sched->template->call_access = calloc(1, sizeof(PolyCallAccess));
  sched->run->calls = calloc(1, sizeof(PolyCallRuntime));
  sched->run->call_io = calloc(1, sizeof(PolyCallIO));
  if (!sched->template->buf_slots || !sched->template->call_access || !sched->run->calls ||
      !sched->run->call_io)
    return sched;

  sched->template->buf_slots[0] = (PolyScheduleBufSlot){
      .dtype = POLY_FLOAT32,
      .numel = view_numel,
      .nbytes = view_numel * (int64_t)sizeof(float),
      .buf_uop = view,
      .device = POLY_DEVICE_CPU,
  };
  sched->template->buf_slots[1] = (PolyScheduleBufSlot){
      .dtype = POLY_FLOAT32,
      .numel = 4,
      .nbytes = 4 * (int64_t)sizeof(float),
      .buf_uop = base,
      .device = POLY_DEVICE_CPU,
  };
  sched->template->call_access[0].n_args = 2;
  sched->template->call_access[0].outs = calloc(2, sizeof(bool));
  sched->template->call_access[0].ins = calloc(2, sizeof(bool));
  sched->run->call_io[0].n_args = 2;
  sched->run->call_io[0].access = &sched->template->call_access[0];
  sched->run->call_io[0].arg_to_slot = calloc(2, sizeof(int));
  if (!sched->template->call_access[0].outs || !sched->template->call_access[0].ins ||
      !sched->run->call_io[0].arg_to_slot)
    return sched;
  sched->run->call_io[0].arg_to_slot[0] = 0;
  sched->run->call_io[0].arg_to_slot[1] = 1;
  sched->template->call_access[0].outs[0] = true;
  sched->template->call_access[0].ins[1] = true;
  sched->template->call_access[0].n_write_args = 1;
  sched->template->call_access[0].write_args = malloc(sizeof(int));
  sched->template->call_access[0].n_read_args = 1;
  sched->template->call_access[0].read_args = malloc(sizeof(int));
  sched->template->call_access[0].n_active_args = 2;
  sched->template->call_access[0].active_args = malloc(2 * sizeof(int));
  if (!sched->template->call_access[0].write_args || !sched->template->call_access[0].read_args ||
      !sched->template->call_access[0].active_args)
    return sched;
  sched->template->call_access[0].write_args[0] = 0;
  sched->template->call_access[0].read_args[0] = 1;
  sched->template->call_access[0].active_args[0] = 0;
  sched->template->call_access[0].active_args[1] = 1;

  if (base_out) *base_out = base;
  if (view_out) *view_out = view;
  return sched;
}

static PolySchedule *make_manual_bufferview_call_schedule(
    PolyCtx *ctx,
    PolyUOp **base_out,
    PolyUOp **view_out
) {
  return make_manual_bufferview_call_schedule_ex(ctx, base_out, view_out, false);
}

TEST(schedule_runtime, lower_sink_to_linear_cache_normalizes_buffer_view_identity) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *linear1 =
      poly_lower_sink_to_linear(ctx, make_bufferview_cache_sink(ctx, 1000), POLY_MODE_CALL);
  ASSERT_NOT_NULL(linear1);
  ASSERT_INT_EQ((int)poly_schedule_cache_len(ctx), 1);

  /* tinygrad callify.pm_replace_buf normalizes BUFFER_VIEW as a call param for
   * schedule-cache identity, the same as BUFFER and BIND. */
  PolyUOp *linear2 =
      poly_lower_sink_to_linear(ctx, make_bufferview_cache_sink(ctx, 2000), POLY_MODE_CALL);
  ASSERT_NOT_NULL(linear2);
  ASSERT_PTR_EQ(linear1, linear2);
  ASSERT_INT_EQ((int)poly_schedule_cache_len(ctx), 1);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(schedule_runtime, call_bufferview_installs_alias_residency_like_tinygrad_slice) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *base = NULL;
  PolyUOp *view = NULL;
  PolySchedule *sched = make_manual_bufferview_call_schedule(ctx, &base, &view);
  ASSERT_NOT_NULL(sched);
  ASSERT_NOT_NULL(base);
  ASSERT_NOT_NULL(view);
  ASSERT_NOT_NULL(sched->template);
  ASSERT_NOT_NULL(sched->run);
  ASSERT_NOT_NULL(sched->template->buf_slots);
  ASSERT_NOT_NULL(sched->template->call_access);
  ASSERT_NOT_NULL(sched->run->calls);
  ASSERT_NOT_NULL(sched->run->call_io);
  ASSERT_NOT_NULL(sched->run->call_io[0].arg_to_slot);
  ASSERT_NOT_NULL(sched->template->call_access[0].outs);
  ASSERT_NOT_NULL(sched->template->call_access[0].ins);

  float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  PolyBuffer base_handle = poly_buffer_make_host_view(data, sizeof(data));
  poly_buffer_attach(ctx, base, &base_handle);

  ASSERT_INT_EQ(poly_run_schedule(ctx, sched, NULL, 0), 0);
  PolyBuffer *alias = poly_buffer_get(ctx, view);
  ASSERT_NOT_NULL(alias);
  ASSERT_PTR_EQ(alias->ptr, data);
  ASSERT_FALSE(alias->owned);
  ASSERT_TRUE(alias->valid);
  ASSERT_INT_EQ((int)alias->nbytes, (int)sizeof(data));
  float out[4] = {0};
  ASSERT_INT_EQ(poly_buffer_read(ctx, view, out, sizeof(out)), 0);
  for (int i = 0; i < 4; i++)
    ASSERT_FLOAT_EQ(out[i], data[i], 0.0);

  poly_schedule_free(sched);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(schedule_runtime, compiled_call_bufferview_installs_alias_from_slot_data) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *base = NULL;
  PolyUOp *view = NULL;
  PolySchedule *sched = make_manual_bufferview_call_schedule(ctx, &base, &view);
  ASSERT_NOT_NULL(sched);
  ASSERT_NOT_NULL(base);
  ASSERT_NOT_NULL(view);

  PolyCompiledSchedule *plan = poly_lower_schedule(ctx, sched, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(plan);

  float data[4] = {5.0f, 6.0f, 7.0f, 8.0f};
  void *slot_data[2] = {NULL, data};
  ASSERT_INT_EQ(poly_run_compiled_schedule(plan, slot_data, 2, NULL, 0), 0);

  PolyBuffer *alias = poly_buffer_get(ctx, view);
  ASSERT_NOT_NULL(alias);
  ASSERT_PTR_EQ(alias->ptr, data);
  ASSERT_FALSE(alias->owned);
  ASSERT_TRUE(alias->valid);
  float out[4] = {0};
  ASSERT_INT_EQ(poly_buffer_read(ctx, view, out, sizeof(out)), 0);
  for (int i = 0; i < 4; i++)
    ASSERT_FLOAT_EQ(out[i], data[i], 0.0);

  poly_compiled_schedule_free(plan);
  poly_schedule_free(sched);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(schedule_runtime, call_bufferview_offset_aliases_source_slice) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *base = NULL;
  PolyUOp *view = NULL;
  PolySchedule *sched = make_manual_bufferview_call_schedule_ex(ctx, &base, &view, true);
  ASSERT_NOT_NULL(sched);
  ASSERT_NOT_NULL(base);
  ASSERT_NOT_NULL(view);

  float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  PolyBuffer base_handle = poly_buffer_make_host_view(data, sizeof(data));
  poly_buffer_attach(ctx, base, &base_handle);

  ASSERT_INT_EQ(poly_run_schedule(ctx, sched, NULL, 0), 0);
  PolyBuffer *alias = poly_buffer_get(ctx, view);
  ASSERT_NOT_NULL(alias);
  ASSERT_PTR_EQ(alias->ptr, data + 1);
  ASSERT_FALSE(alias->owned);
  ASSERT_TRUE(alias->valid);
  ASSERT_INT_EQ((int)alias->nbytes, (int)(2 * sizeof(float)));
  float out[2] = {0};
  ASSERT_INT_EQ(poly_buffer_read(ctx, view, out, sizeof(out)), 0);
  ASSERT_FLOAT_EQ(out[0], 2.0f, 0.0);
  ASSERT_FLOAT_EQ(out[1], 3.0f, 0.0);

  poly_schedule_free(sched);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(schedule_runtime, compiled_call_bufferview_offset_aliases_slot_slice) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *base = NULL;
  PolyUOp *view = NULL;
  PolySchedule *sched = make_manual_bufferview_call_schedule_ex(ctx, &base, &view, true);
  ASSERT_NOT_NULL(sched);
  ASSERT_NOT_NULL(base);
  ASSERT_NOT_NULL(view);

  PolyCompiledSchedule *plan = poly_lower_schedule(ctx, sched, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(plan);

  float data[4] = {5.0f, 6.0f, 7.0f, 8.0f};
  void *slot_data[2] = {NULL, data};
  ASSERT_INT_EQ(poly_run_compiled_schedule(plan, slot_data, 2, NULL, 0), 0);

  PolyBuffer *alias = poly_buffer_get(ctx, view);
  ASSERT_NOT_NULL(alias);
  ASSERT_PTR_EQ(alias->ptr, data + 1);
  ASSERT_FALSE(alias->owned);
  ASSERT_TRUE(alias->valid);
  ASSERT_INT_EQ((int)alias->nbytes, (int)(2 * sizeof(float)));
  float out[2] = {0};
  ASSERT_INT_EQ(poly_buffer_read(ctx, view, out, sizeof(out)), 0);
  ASSERT_FLOAT_EQ(out[0], 6.0f, 0.0);
  ASSERT_FLOAT_EQ(out[1], 7.0f, 0.0);

  poly_compiled_schedule_free(plan);
  poly_schedule_free(sched);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(schedule_runtime, lower_sink_to_linear_deep_graph_uses_heap_scratch) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *in = poly_buffer_f32(ctx, 1);
  PolyUOp *out = poly_buffer_f32(ctx, 1);
  PolyUOp *one = poly_const_float(ctx, 1.0);
  PolyUOp *acc = in;
  for (int i = 0; i < POLY_MAX_STRUCT_NODES + 16; i++)
    acc = poly_alu2(ctx, POLY_OP_ADD, acc, one);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, acc));

  PolyUOp *linear = poly_lower_sink_to_linear(ctx, sink, POLY_MODE_CALL);
  ASSERT_NOT_NULL(linear);
  ASSERT_INT_EQ(linear->op, POLY_OP_LINEAR);

  poly_ctx_destroy(ctx);
  PASS();
}

static PolyUOp *make_bound_vecadd_sink(
    PolyCtx *ctx,
    PolyUOp *N,
    int64_t value,
    int tag_base,
    PolyUOp **a_out,
    PolyUOp **b_out,
    PolyUOp **out_out
) {
  PolyUOp *bind_N = poly_bind_var(ctx, N, value);
  PolyUOp *unique_a = poly_uop0(ctx, POLY_OP_UNIQUE, POLY_VOID, poly_arg_int(tag_base));
  PolyUOp *unique_b = poly_uop0(ctx, POLY_OP_UNIQUE, POLY_VOID, poly_arg_int(tag_base + 1));
  PolyUOp *unique_o = poly_uop0(ctx, POLY_OP_UNIQUE, POLY_VOID, poly_arg_int(tag_base + 2));
  PolyUOp *src_a[2] = {unique_a, bind_N};
  PolyUOp *src_b[2] = {unique_b, bind_N};
  PolyUOp *src_o[2] = {unique_o, bind_N};
  PolyUOp *a = poly_uop(ctx, POLY_OP_BUFFER, POLY_FLOAT32, src_a, 2, poly_arg_int(16));
  PolyUOp *b = poly_uop(ctx, POLY_OP_BUFFER, POLY_FLOAT32, src_b, 2, poly_arg_int(16));
  PolyUOp *out = poly_uop(ctx, POLY_OP_BUFFER, POLY_FLOAT32, src_o, 2, poly_arg_int(16));
  if (a_out) *a_out = a;
  if (b_out) *b_out = b;
  if (out_out) *out_out = out;
  PolyUOp *add = poly_alu2(ctx, POLY_OP_ADD, a, b);
  return poly_sink1(ctx, poly_store_val(ctx, out, add));
}

TEST(schedule_runtime, lower_sink_to_linear_cache_ignores_bind_values) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *N = poly_define_var(ctx, "N", 1, 16);

  PolyUOp *sink4 = make_bound_vecadd_sink(ctx, N, 4, 3000000, NULL, NULL, NULL);
  PolyUOp *linear4 = poly_lower_sink_to_linear(ctx, sink4, POLY_MODE_CALL);
  ASSERT_NOT_NULL(linear4);
  ASSERT_EQ(linear4->op, POLY_OP_LINEAR);
  ASSERT_INT_EQ((int)poly_schedule_cache_len(ctx), 1);

  PolyUOp *sink8 = make_bound_vecadd_sink(ctx, N, 8, 3000000, NULL, NULL, NULL);
  PolyUOp *linear8 = poly_lower_sink_to_linear(ctx, sink8, POLY_MODE_CALL);
  ASSERT_NOT_NULL(linear8);
  ASSERT_PTR_EQ(linear4, linear8);
  ASSERT_INT_EQ((int)poly_schedule_cache_len(ctx), 1);

  poly_ctx_destroy(ctx);
  PASS();
}

static PolyUOp *make_deep_cache_key_sink(PolyCtx *ctx, int64_t tag_base, int depth) {
  PolyUOp *unique_a = poly_uop0(ctx, POLY_OP_UNIQUE, POLY_VOID, poly_arg_int(tag_base));
  PolyUOp *unique_b = poly_uop0(ctx, POLY_OP_UNIQUE, POLY_VOID, poly_arg_int(tag_base + 1));
  PolyUOp *unique_o = poly_uop0(ctx, POLY_OP_UNIQUE, POLY_VOID, poly_arg_int(tag_base + 2));
  PolyUOp *dim = poly_const_int(ctx, 4);
  PolyUOp *src_a[2] = {unique_a, dim};
  PolyUOp *src_b[2] = {unique_b, dim};
  PolyUOp *src_o[2] = {unique_o, dim};
  PolyUOp *a = poly_uop(ctx, POLY_OP_BUFFER, POLY_FLOAT32, src_a, 2, poly_arg_int(4));
  PolyUOp *b = poly_uop(ctx, POLY_OP_BUFFER, POLY_FLOAT32, src_b, 2, poly_arg_int(4));
  PolyUOp *out = poly_uop(ctx, POLY_OP_BUFFER, POLY_FLOAT32, src_o, 2, poly_arg_int(4));
  PolyUOp *expr = a;
  for (int i = 0; i < depth; i++)
    expr = poly_alu2(ctx, POLY_OP_ADD, expr, b);
  return poly_sink1(ctx, poly_store_val(ctx, out, expr));
}

TEST(schedule_runtime, lower_sink_to_linear_deep_cache_key_is_iterative) {
  ScheduleEnvSave scache = schedule_save_env("POLY_SCACHE");
  setenv("POLY_SCACHE", "1", 1);

  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  const int depth = 2048;
  PolyUOp *sink1 = make_deep_cache_key_sink(ctx, 3100000, depth);
  PolyUOp *linear1 = poly_lower_sink_to_linear(ctx, sink1, POLY_MODE_CALL);
  ASSERT_NOT_NULL(linear1);
  ASSERT_EQ(linear1->op, POLY_OP_LINEAR);
  ASSERT_INT_EQ((int)poly_schedule_cache_len(ctx), 1);

  PolyUOp *sink2 = make_deep_cache_key_sink(ctx, 3200000, depth);
  PolyUOp *linear2 = poly_lower_sink_to_linear(ctx, sink2, POLY_MODE_CALL);
  ASSERT_NOT_NULL(linear2);
  ASSERT_PTR_EQ(linear1, linear2);
  ASSERT_INT_EQ((int)poly_schedule_cache_len(ctx), 1);

  poly_ctx_destroy(ctx);
  schedule_restore_env(&scache);
  PASS();
}

static bool schedule_has_buf_slot(PolySchedule *sched, PolyUOp *buf) {
  for (int i = 0; sched && i < sched->template->n_buf_slots; i++)
    if (sched->template->buf_slots[i].buf_uop == buf) return true;
  return false;
}

TEST(schedule_runtime, complete_schedule_reuses_linear_cache_with_current_buf_slots) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *a1 = poly_buffer_f32(ctx, 8);
  PolyUOp *b1 = poly_buffer_f32(ctx, 8);
  PolyUOp *out1 = poly_buffer_f32(ctx, 8);
  PolyUOp *sink1 = poly_sink1(ctx, poly_store_val(ctx, out1, poly_alu2(ctx, POLY_OP_ADD, a1, b1)));
  PolySchedule *sched1 = poly_complete_create_schedule_with_vars(ctx, sink1, POLY_MODE_CALL);
  ASSERT_NOT_NULL(sched1);
  ASSERT_INT_EQ((int)poly_schedule_cache_len(ctx), 1);

  PolyUOp *a2 = poly_buffer_f32(ctx, 8);
  PolyUOp *b2 = poly_buffer_f32(ctx, 8);
  PolyUOp *out2 = poly_buffer_f32(ctx, 8);
  PolyUOp *sink2 = poly_sink1(ctx, poly_store_val(ctx, out2, poly_alu2(ctx, POLY_OP_ADD, a2, b2)));
  PolySchedule *sched2 = poly_complete_create_schedule_with_vars(ctx, sink2, POLY_MODE_CALL);
  ASSERT_NOT_NULL(sched2);
  ASSERT_INT_EQ((int)poly_schedule_cache_len(ctx), 1);

  ASSERT_INT_EQ(sched1->template->n_calls, 1);
  ASSERT_INT_EQ(sched2->template->n_calls, 1);
  ASSERT_PTR_EQ(poly_schedule_call_body(sched1, 0), poly_schedule_call_body(sched2, 0));
  ASSERT_NOT_NULL(sched2->template->call_access);
  ASSERT_NOT_NULL(sched2->run->call_io);
  ASSERT_PTR_EQ(sched2->run->call_io[0].access, &sched2->template->call_access[0]);
  ASSERT_INT_EQ(sched2->template->call_access[0].n_write_args, 1);
  ASSERT_INT_EQ(sched2->template->call_access[0].n_read_args, 2);
  ASSERT_TRUE(schedule_has_buf_slot(sched2, a2));
  ASSERT_TRUE(schedule_has_buf_slot(sched2, b2));
  ASSERT_TRUE(schedule_has_buf_slot(sched2, out2));
  ASSERT_FALSE(schedule_has_buf_slot(sched2, a1));
  ASSERT_FALSE(schedule_has_buf_slot(sched2, b1));
  ASSERT_FALSE(schedule_has_buf_slot(sched2, out1));
  int out_slot = poly_schedule_call_buffer_slot(sched2, 0, 0);
  int a_slot = poly_schedule_call_buffer_slot(sched2, 0, 1);
  int b_slot = poly_schedule_call_buffer_slot(sched2, 0, 2);
  ASSERT_TRUE(out_slot >= 0 && out_slot < sched2->template->n_buf_slots);
  ASSERT_TRUE(a_slot >= 0 && a_slot < sched2->template->n_buf_slots);
  ASSERT_TRUE(b_slot >= 0 && b_slot < sched2->template->n_buf_slots);
  ASSERT_PTR_EQ(sched2->template->buf_slots[out_slot].buf_uop, out2);
  ASSERT_PTR_EQ(sched2->template->buf_slots[a_slot].buf_uop, a2);
  ASSERT_PTR_EQ(sched2->template->buf_slots[b_slot].buf_uop, b2);

  poly_schedule_free(sched1);
  poly_schedule_free(sched2);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(schedule_runtime, complete_schedule_cached_linear_uses_current_bind_defaults) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *N = poly_define_var(ctx, "N", 1, 16);

  PolyUOp *a4 = NULL, *b4 = NULL, *out4 = NULL;
  PolyUOp *sink4 = make_bound_vecadd_sink(ctx, N, 4, 4000000, &a4, &b4, &out4);
  float a4_data[16], b4_data[16], out4_data[16];
  for (int i = 0; i < 16; i++) {
    a4_data[i] = (float)i;
    b4_data[i] = (float)(100 + i);
    out4_data[i] = -1.0f;
  }
  PolyTestBufferView views4[] = {
      POLY_TEST_HOST_VIEW(a4, a4_data),
      POLY_TEST_HOST_VIEW(b4, b4_data),
      POLY_TEST_HOST_VIEW(out4, out4_data),
  };
  ASSERT_INT_EQ(poly_test_realize_buffer_views_vars(ctx, sink4, views4, 3, NULL, 0), 0);
  ASSERT_INT_EQ((int)poly_schedule_cache_len(ctx), 1);
  ASSERT_FLOAT_EQ(out4_data[3], 106.0f, 1e-5);
  ASSERT_FLOAT_EQ(out4_data[4], -1.0f, 1e-5);

  PolyUOp *a8 = NULL, *b8 = NULL, *out8 = NULL;
  PolyUOp *sink8 = make_bound_vecadd_sink(ctx, N, 8, 4000000, &a8, &b8, &out8);
  float a8_data[16], b8_data[16], out8_data[16];
  for (int i = 0; i < 16; i++) {
    a8_data[i] = (float)(10 + i);
    b8_data[i] = (float)(200 + i);
    out8_data[i] = -1.0f;
  }
  PolyTestBufferView views8[] = {
      POLY_TEST_HOST_VIEW(a8, a8_data),
      POLY_TEST_HOST_VIEW(b8, b8_data),
      POLY_TEST_HOST_VIEW(out8, out8_data),
  };
  ASSERT_INT_EQ(poly_test_realize_buffer_views_vars(ctx, sink8, views8, 3, NULL, 0), 0);
  ASSERT_INT_EQ((int)poly_schedule_cache_len(ctx), 1);
  ASSERT_FLOAT_EQ(out8_data[7], 224.0f, 1e-5);
  ASSERT_FLOAT_EQ(out8_data[8], -1.0f, 1e-5);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(schedule_runtime, cached_linear_sub_e2e_uses_current_param_slots) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *a1 = poly_buffer_f32(ctx, 4);
  PolyUOp *b1 = poly_buffer_f32(ctx, 4);
  PolyUOp *out1 = poly_buffer_f32(ctx, 4);
  PolyUOp *sink1 = poly_sink1(
      ctx,
      poly_store_val(ctx, out1, poly_uop2(ctx, POLY_OP_SUB, POLY_FLOAT32, a1, b1, poly_arg_none()))
  );
  float a1_data[4] = {10, 20, 30, 40};
  float b1_data[4] = {1, 2, 3, 4};
  float out1_data[4] = {0};
  PolyTestBufferView views1[] = {
      POLY_TEST_HOST_VIEW(a1, a1_data),
      POLY_TEST_HOST_VIEW(b1, b1_data),
      POLY_TEST_HOST_VIEW(out1, out1_data),
  };
  ASSERT_INT_EQ(poly_test_realize_buffer_views(ctx, sink1, views1, 3), 0);
  ASSERT_INT_EQ((int)poly_schedule_cache_len(ctx), 1);
  ASSERT_FLOAT_EQ(out1_data[0], 9.0f, 1e-5);
  ASSERT_FLOAT_EQ(out1_data[3], 36.0f, 1e-5);

  PolyUOp *a2 = poly_buffer_f32(ctx, 4);
  PolyUOp *b2 = poly_buffer_f32(ctx, 4);
  PolyUOp *out2 = poly_buffer_f32(ctx, 4);
  PolyUOp *sink2 = poly_sink1(
      ctx,
      poly_store_val(ctx, out2, poly_uop2(ctx, POLY_OP_SUB, POLY_FLOAT32, a2, b2, poly_arg_none()))
  );
  float a2_data[4] = {100, 200, 300, 400};
  float b2_data[4] = {7, 8, 9, 10};
  float out2_data[4] = {0};
  PolyTestBufferView views2[] = {
      POLY_TEST_HOST_VIEW(a2, a2_data),
      POLY_TEST_HOST_VIEW(b2, b2_data),
      POLY_TEST_HOST_VIEW(out2, out2_data),
  };
  ASSERT_INT_EQ(poly_test_realize_buffer_views(ctx, sink2, views2, 3), 0);
  ASSERT_INT_EQ((int)poly_schedule_cache_len(ctx), 1);
  ASSERT_FLOAT_EQ(out2_data[0], 93.0f, 1e-5);
  ASSERT_FLOAT_EQ(out2_data[3], 390.0f, 1e-5);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(schedule_runtime, cached_linear_assign_chain_uses_raw_sink_param_slots) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *buf = poly_buffer_f32(ctx, 4);
  PolyUOp *v1 = poly_buffer_f32(ctx, 4);
  PolyUOp *v2 = poly_buffer_f32(ctx, 4);
  PolyUOp *inner_src[2] = {buf, v1};
  PolyUOp *inner = poly_uop(ctx, POLY_OP_ASSIGN, POLY_FLOAT32, inner_src, 2, poly_arg_none());
  PolyUOp *outer_src[2] = {inner, v2};
  PolyUOp *outer = poly_uop(ctx, POLY_OP_ASSIGN, POLY_FLOAT32, outer_src, 2, poly_arg_none());
  PolyUOp *sink = poly_sink1(ctx, outer);

  float buf_data[4] = {0, 0, 0, 0};
  float v1_data[4] = {1, 2, 3, 4};
  float v2_data[4] = {10, 20, 30, 40};
  PolyTestBufferView views[] = {
      POLY_TEST_HOST_VIEW(buf, buf_data),
      POLY_TEST_HOST_VIEW(v1, v1_data),
      POLY_TEST_HOST_VIEW(v2, v2_data),
  };

  /* earliest_rewrites collapses ASSIGN(ASSIGN(buf, v1), v2) to
   * ASSIGN(buf, v2), so the optimized kernel graph no longer uses v1. The
   * cached LINEAR replay must still number CALL params against the original
   * sink slots, otherwise PARAM(1) incorrectly binds v1 instead of v2. */
  ASSERT_INT_EQ(poly_test_realize_buffer_views(ctx, sink, views, 3), 0);
  ASSERT_INT_EQ((int)poly_schedule_cache_len(ctx), 1);
  ASSERT_FLOAT_EQ(buf_data[0], 10.0f, 1e-5);
  ASSERT_FLOAT_EQ(buf_data[3], 40.0f, 1e-5);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(schedule_runtime, runtime_cache_reuses_runner_across_fresh_schedules) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  ASSERT_INT_EQ((int)poly_to_program_cache_len(ctx), 0);
  ASSERT_INT_EQ((int)poly_runtime_cache_len(ctx), 0);

  /* This mirrors tinygrad's to_program/runtime cache layer: the schedule is
   * rebuilt for a fresh realization, but the backend program for the same
   * normalized kernel root is reused while current buffer slots are rebound. */
  PolyUOp *a1 = poly_buffer_f32(ctx, 4);
  PolyUOp *b1 = poly_buffer_f32(ctx, 4);
  PolyUOp *out1 = poly_buffer_f32(ctx, 4);
  PolyUOp *sink1 = poly_sink1(ctx, poly_store_val(ctx, out1, poly_alu2(ctx, POLY_OP_ADD, a1, b1)));
  float a1_data[4] = {1, 2, 3, 4};
  float b1_data[4] = {10, 20, 30, 40};
  float out1_data[4] = {0};
  PolyTestBufferView views1[] = {
      POLY_TEST_HOST_VIEW(a1, a1_data),
      POLY_TEST_HOST_VIEW(b1, b1_data),
      POLY_TEST_HOST_VIEW(out1, out1_data),
  };
  ASSERT_INT_EQ(poly_test_realize_buffer_views(ctx, sink1, views1, 3), 0);
  ASSERT_INT_EQ(
      (int)poly_to_program_cache_len(ctx), expected_to_program_cache_entries(ctx, 1)
  );
  ASSERT_INT_EQ((int)poly_runtime_cache_len(ctx), 1);
  ASSERT_FLOAT_EQ(out1_data[0], 11.0f, 1e-5);
  ASSERT_FLOAT_EQ(out1_data[3], 44.0f, 1e-5);

  PolyUOp *a2 = poly_buffer_f32(ctx, 4);
  PolyUOp *b2 = poly_buffer_f32(ctx, 4);
  PolyUOp *out2 = poly_buffer_f32(ctx, 4);
  PolyUOp *sink2 = poly_sink1(ctx, poly_store_val(ctx, out2, poly_alu2(ctx, POLY_OP_ADD, a2, b2)));
  float a2_data[4] = {5, 6, 7, 8};
  float b2_data[4] = {50, 60, 70, 80};
  float out2_data[4] = {0};
  PolyTestBufferView views2[] = {
      POLY_TEST_HOST_VIEW(a2, a2_data),
      POLY_TEST_HOST_VIEW(b2, b2_data),
      POLY_TEST_HOST_VIEW(out2, out2_data),
  };
  ASSERT_INT_EQ(poly_test_realize_buffer_views(ctx, sink2, views2, 3), 0);
  ASSERT_INT_EQ(
      (int)poly_to_program_cache_len(ctx), expected_to_program_cache_entries(ctx, 1)
  );
  ASSERT_INT_EQ((int)poly_runtime_cache_len(ctx), 1);
  ASSERT_FLOAT_EQ(out2_data[0], 55.0f, 1e-5);
  ASSERT_FLOAT_EQ(out2_data[3], 88.0f, 1e-5);

  PolyUOp *out3 = poly_buffer_f32(ctx, 4);
  PolyUOp *sink3 = poly_sink1(ctx, poly_store_val(ctx, out3, poly_alu2(ctx, POLY_OP_MUL, a2, b2)));
  float out3_data[4] = {0};
  PolyTestBufferView views3[] = {
      POLY_TEST_HOST_VIEW(a2, a2_data),
      POLY_TEST_HOST_VIEW(b2, b2_data),
      POLY_TEST_HOST_VIEW(out3, out3_data),
  };
  ASSERT_INT_EQ(poly_test_realize_buffer_views(ctx, sink3, views3, 3), 0);
  ASSERT_INT_EQ(
      (int)poly_to_program_cache_len(ctx), expected_to_program_cache_entries(ctx, 2)
  );
  ASSERT_INT_EQ((int)poly_runtime_cache_len(ctx), 2);
  ASSERT_FLOAT_EQ(out3_data[0], 250.0f, 1e-5);
  ASSERT_FLOAT_EQ(out3_data[3], 640.0f, 1e-5);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(schedule_runtime, runtime_cache_keys_distinct_program_wrappers) {
  ScheduleEnvSave pcache = schedule_save_env("POLY_PCACHE");
  setenv("POLY_PCACHE", "1", 1);

  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  ASSERT_INT_EQ((int)poly_to_program_cache_len(ctx), 0);
  ASSERT_INT_EQ((int)poly_runtime_cache_len(ctx), 0);

  PolyUOp *a = poly_buffer_f32(ctx, 4);
  PolyUOp *b = poly_buffer_f32(ctx, 4);
  PolyUOp *out = poly_buffer_f32(ctx, 4);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, poly_alu2(ctx, POLY_OP_ADD, a, b)));

  PolySchedule *sched1 = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  PolySchedule *sched2 = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_NOT_NULL(sched1);
  ASSERT_NOT_NULL(sched2);
  ASSERT_INT_EQ(sched1->template->n_calls, 1);
  ASSERT_INT_EQ(sched2->template->n_calls, 1);

  ASSERT_INT_EQ(poly_schedule_call_lower(ctx, sched1, 0, POLY_DEVICE_CPU), 0);
  ASSERT_INT_EQ((int)poly_to_program_cache_len(ctx), 1);
  ASSERT_INT_EQ((int)poly_runtime_cache_len(ctx), 1);

  PolyUOp *old_call = poly_schedule_call(sched2, 0);
  ASSERT_NOT_NULL(old_call);
  ASSERT_TRUE(old_call->n_src >= 1);
  PolyUOp *body = poly_schedule_call_body(sched2, 0);
  ASSERT_NOT_NULL(body);
  ASSERT_INT_EQ(body->op, POLY_OP_SINK);

  PolyUOp **base_src = malloc((size_t)old_call->n_src * sizeof(PolyUOp *));
  ASSERT_NOT_NULL(base_src);
  memcpy(base_src, old_call->src, (size_t)old_call->n_src * sizeof(PolyUOp *));
  base_src[0] = body;
  PolyUOp *base_call =
      poly_uop(ctx, POLY_OP_CALL, POLY_VOID, base_src, old_call->n_src, poly_arg_none());
  free(base_src);
  ASSERT_NOT_NULL(base_call);

  PolyUOp *alt_program = poly_program_from_call(ctx, base_call, "alt_program_cache_key");
  ASSERT_NOT_NULL(alt_program);
  ASSERT_PTR_NEQ(alt_program, old_call->src[0]);

  PolyUOp **alt_src = malloc((size_t)old_call->n_src * sizeof(PolyUOp *));
  ASSERT_NOT_NULL(alt_src);
  memcpy(alt_src, old_call->src, (size_t)old_call->n_src * sizeof(PolyUOp *));
  alt_src[0] = alt_program;
  PolyUOp *alt_call =
      poly_uop(ctx, POLY_OP_CALL, POLY_VOID, alt_src, old_call->n_src, poly_arg_none());
  free(alt_src);
  ASSERT_NOT_NULL(alt_call);

  sched2->template->linear->src[0] = alt_call;
  sched2->run->calls[0].call = alt_call;

  ASSERT_INT_EQ(poly_schedule_call_lower(ctx, sched2, 0, POLY_DEVICE_CPU), 0);
  ASSERT_INT_EQ((int)poly_to_program_cache_len(ctx), 2);
  ASSERT_INT_EQ((int)poly_runtime_cache_len(ctx), 2);

  poly_schedule_free(sched2);
  poly_schedule_free(sched1);
  poly_ctx_destroy(ctx);
  schedule_restore_env(&pcache);
  PASS();
}

TEST(schedule_runtime, runtime_cache_clear_keeps_live_schedule_runner_valid) {
  ScheduleEnvSave pcache = schedule_save_env("POLY_PCACHE");
  setenv("POLY_PCACHE", "1", 1);

  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *a = poly_buffer_f32(ctx, 4);
  PolyUOp *b = poly_buffer_f32(ctx, 4);
  PolyUOp *out = poly_buffer_f32(ctx, 4);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, poly_alu2(ctx, POLY_OP_ADD, a, b)));
  PolySchedule *sched = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_NOT_NULL(sched);

  float a_data[4] = {1, 2, 3, 4};
  float b_data[4] = {10, 20, 30, 40};
  float out_data[4] = {0};
  PolyBuffer a_view = poly_buffer_make_host_view(a_data, sizeof(a_data));
  PolyBuffer b_view = poly_buffer_make_host_view(b_data, sizeof(b_data));
  PolyBuffer out_view = poly_buffer_make_host_view(out_data, sizeof(out_data));
  poly_buffer_attach(ctx, a, &a_view);
  poly_buffer_attach(ctx, b, &b_view);
  poly_buffer_attach(ctx, out, &out_view);

  ASSERT_INT_EQ(poly_run_schedule(ctx, sched, NULL, 0), 0);
  ASSERT_INT_EQ((int)poly_runtime_cache_len(ctx), 1);
  PolyCtxStats cached = {0};
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &cached), 0);
  ASSERT_INT_EQ(cached.runtime_cache_entries, 1);
  ASSERT_TRUE(cached.runtime_artifact_entries > 0);
  ASSERT_TRUE(cached.compiled_artifact_bytes > 0);
  ASSERT_FLOAT_EQ(out_data[0], 11.0f, 1e-5);
  ASSERT_FLOAT_EQ(out_data[3], 44.0f, 1e-5);

  poly_runtime_cache_clear(ctx);
  ASSERT_INT_EQ((int)poly_runtime_cache_len(ctx), 0);
  PolyCtxStats evicted = {0};
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &evicted), 0);
  ASSERT_INT_EQ(evicted.runtime_cache_entries, 0);
  ASSERT_INT_EQ(evicted.runtime_artifact_entries, cached.runtime_artifact_entries);
  ASSERT_TRUE(evicted.compiled_artifact_bytes > 0);
  ASSERT_TRUE(evicted.compiled_artifact_bytes < cached.compiled_artifact_bytes);

  a_data[0] = 5.0f;
  a_data[1] = 6.0f;
  a_data[2] = 7.0f;
  a_data[3] = 8.0f;
  b_data[0] = 50.0f;
  b_data[1] = 60.0f;
  b_data[2] = 70.0f;
  b_data[3] = 80.0f;
  memset(out_data, 0, sizeof(out_data));

  ASSERT_INT_EQ(poly_run_schedule(ctx, sched, NULL, 0), 0);
  ASSERT_FLOAT_EQ(out_data[0], 55.0f, 1e-5);
  ASSERT_FLOAT_EQ(out_data[3], 88.0f, 1e-5);

  poly_schedule_free(sched);
  PolyCtxStats released = {0};
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &released), 0);
  ASSERT_INT_EQ(released.runtime_artifact_entries, 0);
  ASSERT_INT_EQ(released.compiled_artifact_bytes, 0);
  poly_ctx_destroy(ctx);
  schedule_restore_env(&pcache);
  PASS();
}

TEST(schedule_runtime, runtime_cache_clear_keeps_live_compiled_runner_valid) {
  ScheduleEnvSave pcache = schedule_save_env("POLY_PCACHE");
  setenv("POLY_PCACHE", "1", 1);

  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *a = poly_buffer_f32(ctx, 4);
  PolyUOp *b = poly_buffer_f32(ctx, 4);
  PolyUOp *out = poly_buffer_f32(ctx, 4);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, poly_alu2(ctx, POLY_OP_ADD, a, b)));
  PolySchedule *sched = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_NOT_NULL(sched);
  PolyCompiledSchedule *plan = poly_lower_schedule(ctx, sched, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(plan);
  ASSERT_INT_EQ((int)poly_runtime_cache_len(ctx), 1);
  PolyCtxStats cached = {0};
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &cached), 0);
  ASSERT_INT_EQ(cached.runtime_cache_entries, 1);
  ASSERT_TRUE(cached.runtime_artifact_entries > 0);
  ASSERT_TRUE(cached.compiled_artifact_bytes > 0);

  float a_data[4] = {1, 2, 3, 4};
  float b_data[4] = {10, 20, 30, 40};
  float out_data[4] = {0};
  void *slot_data[16] = {0};
  ASSERT_TRUE(sched->template->n_buf_slots <= 16);
  for (int i = 0; i < sched->template->n_buf_slots; i++) {
    if (sched->template->buf_slots[i].buf_uop == a)
      slot_data[i] = a_data;
    else if (sched->template->buf_slots[i].buf_uop == b)
      slot_data[i] = b_data;
    else if (sched->template->buf_slots[i].buf_uop == out)
      slot_data[i] = out_data;
  }

  ASSERT_INT_EQ(
      poly_run_compiled_schedule(plan, slot_data, sched->template->n_buf_slots, NULL, 0), 0
  );
  ASSERT_FLOAT_EQ(out_data[0], 11.0f, 1e-5);
  ASSERT_FLOAT_EQ(out_data[3], 44.0f, 1e-5);

  poly_runtime_cache_clear(ctx);
  ASSERT_INT_EQ((int)poly_runtime_cache_len(ctx), 0);
  PolyCtxStats evicted = {0};
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &evicted), 0);
  ASSERT_INT_EQ(evicted.runtime_cache_entries, 0);
  ASSERT_INT_EQ(evicted.runtime_artifact_entries, cached.runtime_artifact_entries);
  ASSERT_TRUE(evicted.compiled_artifact_bytes > 0);
  ASSERT_TRUE(evicted.compiled_artifact_bytes < cached.compiled_artifact_bytes);

  a_data[0] = 5.0f;
  a_data[1] = 6.0f;
  a_data[2] = 7.0f;
  a_data[3] = 8.0f;
  b_data[0] = 50.0f;
  b_data[1] = 60.0f;
  b_data[2] = 70.0f;
  b_data[3] = 80.0f;
  memset(out_data, 0, sizeof(out_data));

  ASSERT_INT_EQ(
      poly_run_compiled_schedule(plan, slot_data, sched->template->n_buf_slots, NULL, 0), 0
  );
  ASSERT_FLOAT_EQ(out_data[0], 55.0f, 1e-5);
  ASSERT_FLOAT_EQ(out_data[3], 88.0f, 1e-5);

  poly_compiled_schedule_free(plan);
  PolyCtxStats released = {0};
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &released), 0);
  ASSERT_INT_EQ(released.runtime_artifact_entries, 0);
  ASSERT_INT_EQ(released.compiled_artifact_bytes, 0);
  poly_schedule_free(sched);
  poly_ctx_destroy(ctx);
  schedule_restore_env(&pcache);
  PASS();
}

TEST(schedule_runtime, ctx_stats_reports_schedule_and_runtime_caches) {
  ScheduleEnvSave pcache = schedule_save_env("POLY_PCACHE");
  setenv("POLY_PCACHE", "1", 1);

  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyCtxStats stats = {0};
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &stats), 0);
  ASSERT_INT_EQ(stats.schedule_cache_entries, 0);
  ASSERT_INT_EQ(stats.to_program_cache_entries, 0);
  ASSERT_INT_EQ(stats.runtime_cache_entries, 0);
  ASSERT_INT_EQ(stats.program_cache_entries, 0);
  ASSERT_INT_EQ(stats.compiled_artifact_bytes, 0);

  PolyUOp *a = poly_buffer_f32(ctx, 4);
  PolyUOp *b = poly_buffer_f32(ctx, 4);
  PolyUOp *out = poly_buffer_f32(ctx, 4);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, poly_alu2(ctx, POLY_OP_ADD, a, b)));

  float a_data[4] = {1, 2, 3, 4};
  float b_data[4] = {10, 20, 30, 40};
  float out_data[4] = {0};
  PolyTestBufferView views[] = {
      POLY_TEST_HOST_VIEW(a, a_data),
      POLY_TEST_HOST_VIEW(b, b_data),
      POLY_TEST_HOST_VIEW(out, out_data),
  };

  ASSERT_INT_EQ(poly_test_realize_buffer_views(ctx, sink, views, 3), 0);
  ASSERT_FLOAT_EQ(out_data[0], 11.0f, 1e-5);
  ASSERT_FLOAT_EQ(out_data[3], 44.0f, 1e-5);

  ASSERT_INT_EQ(poly_ctx_stats(ctx, &stats), 0);
  ASSERT_INT_EQ(stats.schedule_cache_entries, poly_schedule_cache_len(ctx));
  ASSERT_INT_EQ(stats.to_program_cache_entries, poly_to_program_cache_len(ctx));
  ASSERT_INT_EQ(stats.runtime_cache_entries, poly_runtime_cache_len(ctx));
  ASSERT_INT_EQ(stats.program_cache_entries, stats.runtime_cache_entries);
  ASSERT_INT_EQ((int)poly_program_cache_len(ctx), (int)poly_runtime_cache_len(ctx));
  ASSERT_INT_EQ(stats.schedule_cache_entries, 1);
  ASSERT_INT_EQ(stats.to_program_cache_entries, expected_to_program_cache_entries(ctx, 1));
  ASSERT_INT_EQ(stats.runtime_cache_entries, 1);
  ASSERT_INT_EQ(stats.program_cache_entries, 1);
  ASSERT_INT_EQ(stats.compiled_artifact_bytes, poly_runtime_cache_artifact_bytes(ctx));
  ASSERT_INT_EQ(
      (int)poly_program_cache_artifact_bytes(ctx),
      (int)poly_runtime_cache_artifact_bytes(ctx)
  );
  if (poly_ctx_get_preferred_device(ctx) == POLY_DEVICE_INTERP)
    ASSERT_TRUE(stats.compiled_artifact_bytes > 0);
  else
    ASSERT_TRUE(stats.compiled_artifact_bytes > 4096);
  ASSERT_TRUE(stats.buffer_entries >= 3);

  poly_runtime_cache_clear(ctx);
  poly_to_program_cache_clear(ctx);
  poly_schedule_cache_clear(ctx);

  ASSERT_INT_EQ(poly_ctx_stats(ctx, &stats), 0);
  ASSERT_INT_EQ(stats.schedule_cache_entries, 0);
  ASSERT_INT_EQ(stats.to_program_cache_entries, 0);
  ASSERT_INT_EQ(stats.runtime_cache_entries, 0);
  ASSERT_INT_EQ(stats.program_cache_entries, 0);
  ASSERT_INT_EQ(stats.compiled_artifact_bytes, 0);

  poly_ctx_destroy(ctx);
  schedule_restore_env(&pcache);
  PASS();
}

TEST(schedule_runtime, ctx_stats_fixed_shape_replay_plateaus) {
  ScheduleEnvSave pcache = schedule_save_env("POLY_PCACHE");
  ScheduleEnvSave scache = schedule_save_env("POLY_SCACHE");
  setenv("POLY_PCACHE", "1", 1);
  setenv("POLY_SCACHE", "1", 1);

  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  enum { N = 1024 };
  PolyUOp *a = poly_buffer_f32(ctx, N);
  PolyUOp *b = poly_buffer_f32(ctx, N);
  PolyUOp *out = poly_buffer_f32(ctx, N);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, poly_alu2(ctx, POLY_OP_ADD, a, b)));

  static float a_data[N], b_data[N], out_data[N];
  for (int i = 0; i < N; i++) {
    a_data[i] = (float)i;
    b_data[i] = (float)(N - i);
  }
  PolyBuffer a_view = poly_buffer_make_host_view(a_data, sizeof(a_data));
  PolyBuffer b_view = poly_buffer_make_host_view(b_data, sizeof(b_data));
  PolyBuffer out_view = poly_buffer_make_host_view(out_data, sizeof(out_data));
  poly_buffer_attach(ctx, a, &a_view);
  poly_buffer_attach(ctx, b, &b_view);
  poly_buffer_attach(ctx, out, &out_view);

  ASSERT_INT_EQ(poly_realize_sink(ctx, sink), 0);
  ASSERT_FLOAT_EQ(out_data[0], (float)N, 1e-5);
  ASSERT_FLOAT_EQ(out_data[N - 1], (float)N, 1e-5);

  PolyCtxStats first = {0};
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &first), 0);
  ASSERT_INT_EQ(first.schedule_cache_entries, 1);
  ASSERT_INT_EQ(first.to_program_cache_entries, expected_to_program_cache_entries(ctx, 1));
  ASSERT_INT_EQ(first.runtime_cache_entries, 1);

  for (int iter = 0; iter < 32; iter++) {
    memset(out_data, 0, sizeof(out_data));
    ASSERT_INT_EQ(poly_realize_sink(ctx, sink), 0);
    ASSERT_FLOAT_EQ(out_data[0], (float)N, 1e-5);
    ASSERT_FLOAT_EQ(out_data[N - 1], (float)N, 1e-5);
  }

  PolyCtxStats replay = {0};
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &replay), 0);
  ASSERT_INT_EQ(replay.arena_bytes, first.arena_bytes);
  ASSERT_INT_EQ(replay.cse_entries, first.cse_entries);
  ASSERT_INT_EQ(replay.schedule_cache_entries, first.schedule_cache_entries);
  ASSERT_INT_EQ(replay.to_program_cache_entries, first.to_program_cache_entries);
  ASSERT_INT_EQ(replay.runtime_cache_entries, first.runtime_cache_entries);
  ASSERT_INT_EQ(replay.shape_cache_entries, first.shape_cache_entries);
  ASSERT_INT_EQ(replay.buffer_entries, first.buffer_entries);
  ASSERT_INT_EQ(replay.compiled_artifact_bytes, first.compiled_artifact_bytes);

  poly_ctx_destroy(ctx);
  schedule_restore_env(&scache);
  schedule_restore_env(&pcache);
  PASS();
}

TEST(schedule_runtime, ctx_stats_runtime_var_replay_plateaus) {
  ScheduleEnvSave pcache = schedule_save_env("POLY_PCACHE");
  ScheduleEnvSave scache = schedule_save_env("POLY_SCACHE");
  setenv("POLY_PCACHE", "1", 1);
  setenv("POLY_SCACHE", "1", 1);

  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *N = poly_define_var(ctx, "N", 1, 16);
  ASSERT_NOT_NULL(N);
  PolyUOp *a = poly_buffer_var(ctx, POLY_FLOAT32, N, NULL, 0);
  PolyUOp *out = poly_buffer_var(ctx, POLY_FLOAT32, N, NULL, 0);
  ASSERT_NOT_NULL(a);
  ASSERT_NOT_NULL(out);
  PolyUOp *sink = poly_sink1(
      ctx,
      poly_store_val(ctx, out, poly_alu2(ctx, POLY_OP_ADD, a, poly_const_float(ctx, 1.0f)))
  );

  float a_data[16];
  float out_data[16];
  for (int i = 0; i < 16; i++)
    a_data[i] = (float)(i + 1);
  poly_buffer_set(ctx, a, a_data, sizeof(a_data), POLY_DEVICE_CPU);
  poly_buffer_set(ctx, out, out_data, sizeof(out_data), POLY_DEVICE_CPU);

  PolySchedule *sched = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_NOT_NULL(sched);

  PolyVarBinding bind = {.var = N, .value = 6};
  for (int i = 0; i < 16; i++)
    out_data[i] = -999.0f;
  ASSERT_INT_EQ(poly_run_schedule(ctx, sched, &bind, 1), 0);
  for (int i = 0; i < 6; i++)
    ASSERT_FLOAT_EQ(out_data[i], (float)(i + 2), 1e-5f);
  ASSERT_FLOAT_EQ(out_data[6], -999.0f, 1e-5f);

  PolyCtxStats first = {0};
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &first), 0);
  ASSERT_INT_EQ(first.schedule_cache_entries, 1);
  ASSERT_INT_EQ(first.to_program_cache_entries, expected_to_program_cache_entries(ctx, 1));
  ASSERT_INT_EQ(first.runtime_cache_entries, 1);

  const int vals[] = {12, 6, 16, 12, 1, 15};
  for (int iter = 0; iter < (int)(sizeof(vals) / sizeof(vals[0])); iter++) {
    bind.value = vals[iter];
    for (int i = 0; i < 16; i++)
      out_data[i] = -999.0f;
    ASSERT_INT_EQ(poly_run_schedule(ctx, sched, &bind, 1), 0);
    for (int i = 0; i < vals[iter]; i++)
      ASSERT_FLOAT_EQ(out_data[i], (float)(i + 2), 1e-5f);
    if (vals[iter] < 16)
      ASSERT_FLOAT_EQ(out_data[vals[iter]], -999.0f, 1e-5f);
  }

  PolyCtxStats replay = {0};
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &replay), 0);
  ASSERT_INT_EQ(replay.arena_bytes, first.arena_bytes);
  ASSERT_INT_EQ(replay.cse_entries, first.cse_entries);
  ASSERT_INT_EQ(replay.schedule_cache_entries, first.schedule_cache_entries);
  ASSERT_INT_EQ(replay.to_program_cache_entries, first.to_program_cache_entries);
  ASSERT_INT_EQ(replay.runtime_cache_entries, first.runtime_cache_entries);
  ASSERT_INT_EQ(replay.shape_cache_entries, first.shape_cache_entries);
  ASSERT_INT_EQ(replay.buffer_entries, first.buffer_entries);
  ASSERT_INT_EQ(replay.compiled_artifact_bytes, first.compiled_artifact_bytes);

  poly_schedule_free(sched);
  poly_ctx_destroy(ctx);
  schedule_restore_env(&scache);
  schedule_restore_env(&pcache);
  PASS();
}

TEST(schedule_runtime, schedule_cache_clear_keeps_live_schedule_cache_entry_valid) {
  ScheduleEnvSave scache = schedule_save_env("POLY_SCACHE");
  setenv("POLY_SCACHE", "1", 1);

  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *a = poly_buffer_f32(ctx, 4);
  PolyUOp *b = poly_buffer_f32(ctx, 4);
  PolyUOp *out = poly_buffer_f32(ctx, 4);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, poly_alu2(ctx, POLY_OP_ADD, a, b)));
  PolySchedule *sched = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_NOT_NULL(sched);
  ASSERT_INT_EQ((int)poly_schedule_cache_len(ctx), 1);

  poly_schedule_cache_clear(ctx);
  ASSERT_INT_EQ((int)poly_schedule_cache_len(ctx), 0);

  float a_data[4] = {1, 2, 3, 4};
  float b_data[4] = {10, 20, 30, 40};
  float out_data[4] = {0};
  PolyBuffer a_view = poly_buffer_make_host_view(a_data, sizeof(a_data));
  PolyBuffer b_view = poly_buffer_make_host_view(b_data, sizeof(b_data));
  PolyBuffer out_view = poly_buffer_make_host_view(out_data, sizeof(out_data));
  poly_buffer_attach(ctx, a, &a_view);
  poly_buffer_attach(ctx, b, &b_view);
  poly_buffer_attach(ctx, out, &out_view);

  ASSERT_INT_EQ(poly_run_schedule(ctx, sched, NULL, 0), 0);
  ASSERT_FLOAT_EQ(out_data[0], 11.0f, 1e-5);
  ASSERT_FLOAT_EQ(out_data[3], 44.0f, 1e-5);

  poly_schedule_free(sched);
  ASSERT_INT_EQ((int)poly_schedule_cache_len(ctx), 0);

  PolyUOp *c = poly_buffer_f32(ctx, 4);
  PolyUOp *d = poly_buffer_f32(ctx, 4);
  PolyUOp *out2 = poly_buffer_f32(ctx, 4);
  PolyUOp *sink2 = poly_sink1(ctx, poly_store_val(ctx, out2, poly_alu2(ctx, POLY_OP_ADD, c, d)));
  PolySchedule *fresh = poly_complete_create_schedule_with_vars(ctx, sink2, POLY_MODE_CALL);
  ASSERT_NOT_NULL(fresh);
  ASSERT_INT_EQ((int)poly_schedule_cache_len(ctx), 1);
  poly_schedule_free(fresh);

  poly_ctx_destroy(ctx);
  schedule_restore_env(&scache);
  PASS();
}

TEST(schedule_runtime, schedule_cache_misses_on_changed_op_shape) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *a = poly_buffer_f32(ctx, 4);
  PolyUOp *b = poly_buffer_f32(ctx, 4);
  PolyUOp *out = poly_buffer_f32(ctx, 4);
  PolyUOp *sink_add = poly_sink1(ctx, poly_store_val(ctx, out, poly_alu2(ctx, POLY_OP_ADD, a, b)));
  PolyUOp *lin_add = poly_lower_sink_to_linear(ctx, sink_add, POLY_MODE_CALL);
  ASSERT_NOT_NULL(lin_add);
  ASSERT_INT_EQ((int)poly_schedule_cache_len(ctx), 1);

  PolyUOp *c = poly_buffer_f32(ctx, 4);
  PolyUOp *d = poly_buffer_f32(ctx, 4);
  PolyUOp *out2 = poly_buffer_f32(ctx, 4);
  PolyUOp *sink_mul = poly_sink1(ctx, poly_store_val(ctx, out2, poly_alu2(ctx, POLY_OP_MUL, c, d)));
  PolyUOp *lin_mul = poly_lower_sink_to_linear(ctx, sink_mul, POLY_MODE_CALL);
  ASSERT_NOT_NULL(lin_mul);
  ASSERT_TRUE(lin_mul != lin_add);
  ASSERT_INT_EQ((int)poly_schedule_cache_len(ctx), 2);

  int64_t axes[] = {0};
  PolyUOp *e = poly_buffer_f32(ctx, 4);
  PolyUOp *out3 = poly_buffer_f32(ctx, 1);
  PolyUOp *sink_reduce =
      poly_sink1(ctx, poly_store_val(ctx, out3, poly_reduce_axis(ctx, POLY_OP_ADD, e, axes, 1)));
  PolyUOp *lin_reduce = poly_lower_sink_to_linear(ctx, sink_reduce, POLY_MODE_CALL);
  ASSERT_NOT_NULL(lin_reduce);
  ASSERT_TRUE(lin_reduce != lin_add);
  ASSERT_TRUE(lin_reduce != lin_mul);
  ASSERT_INT_EQ((int)poly_schedule_cache_len(ctx), 3);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(schedule_runtime, schedule_cache_disabled_by_env_does_not_store_linear) {
  ScheduleEnvSave scache = schedule_save_env("POLY_SCACHE");
  setenv("POLY_SCACHE", "0", 1);

  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *a = poly_buffer_f32(ctx, 4);
  PolyUOp *b = poly_buffer_f32(ctx, 4);
  PolyUOp *out = poly_buffer_f32(ctx, 4);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, poly_alu2(ctx, POLY_OP_ADD, a, b)));

  /* mirrors tinygrad Context(SCACHE=0): lowering still returns LINEAR, but
   * neither the lookup nor the write path can populate the schedule cache. */
  PolyUOp *lin1 = poly_lower_sink_to_linear(ctx, sink, POLY_MODE_CALL);
  ASSERT_NOT_NULL(lin1);
  ASSERT_INT_EQ((int)poly_schedule_cache_len(ctx), 0);

  PolyUOp *lin2 = poly_lower_sink_to_linear(ctx, sink, POLY_MODE_CALL);
  ASSERT_NOT_NULL(lin2);
  ASSERT_INT_EQ((int)poly_schedule_cache_len(ctx), 0);

  PolySchedule *ps = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_NOT_NULL(ps);
  ASSERT_INT_EQ((int)poly_schedule_cache_len(ctx), 0);
  poly_schedule_free(ps);

  poly_ctx_destroy(ctx);
  schedule_restore_env(&scache);
  PASS();
}

TEST(schedule_runtime, cached_linear_runtime_override_wins_over_default) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *N = poly_define_var(ctx, "N", 1, 16);

  PolyUOp *a4 = NULL, *b4 = NULL, *out4 = NULL;
  PolyUOp *sink4 = make_bound_vecadd_sink(ctx, N, 4, 5000000, &a4, &b4, &out4);
  float a4_data[16], b4_data[16], out4_data[16];
  for (int i = 0; i < 16; i++) {
    a4_data[i] = (float)i;
    b4_data[i] = (float)(100 + i);
    out4_data[i] = -1.0f;
  }
  PolyTestBufferView views4[] = {
      POLY_TEST_HOST_VIEW(a4, a4_data),
      POLY_TEST_HOST_VIEW(b4, b4_data),
      POLY_TEST_HOST_VIEW(out4, out4_data),
  };
  ASSERT_INT_EQ(poly_test_realize_buffer_views_vars(ctx, sink4, views4, 3, NULL, 0), 0);
  ASSERT_INT_EQ((int)poly_schedule_cache_len(ctx), 1);

  PolyUOp *a_again = NULL, *b_again = NULL, *out_again = NULL;
  PolyUOp *sink_again = make_bound_vecadd_sink(ctx, N, 4, 5000100, &a_again, &b_again, &out_again);
  float a_data[16], b_data[16], out_data[16];
  for (int i = 0; i < 16; i++) {
    a_data[i] = (float)(10 + i);
    b_data[i] = (float)(200 + i);
    out_data[i] = -1.0f;
  }
  PolyTestBufferView views[] = {
      POLY_TEST_HOST_VIEW(a_again, a_data),
      POLY_TEST_HOST_VIEW(b_again, b_data),
      POLY_TEST_HOST_VIEW(out_again, out_data),
  };
  PolyVarBinding override = {.var = N, .value = 6};
  ASSERT_INT_EQ(poly_test_realize_buffer_views_vars(ctx, sink_again, views, 3, &override, 1), 0);
  ASSERT_INT_EQ((int)poly_schedule_cache_len(ctx), 1);
  ASSERT_FLOAT_EQ(out_data[5], 220.0f, 1e-5);
  ASSERT_FLOAT_EQ(out_data[6], -1.0f, 1e-5);

  poly_ctx_destroy(ctx);
  PASS();
}

static PolyUOp *make_copy_sink(PolyCtx *ctx, PolyUOp **src_out, PolyUOp **dst_out) {
  PolyUOp *src = poly_buffer_f32(ctx, 4);
  PolyUOp *dst = poly_buffer_f32(ctx, 4);
  PolyUOp *dev = poly_uop0(ctx, POLY_OP_DEVICE, POLY_VOID, poly_arg_int(POLY_DEVICE_CPU));
  PolyUOp *copy_src[2] = {src, dev};
  PolyUOp *copy = poly_uop(ctx, POLY_OP_COPY, POLY_FLOAT32, copy_src, 2, poly_arg_none());
  if (src_out) *src_out = src;
  if (dst_out) *dst_out = dst;
  return poly_sink1(ctx, poly_store_val(ctx, dst, copy));
}

TEST(schedule_runtime, cached_linear_copy_e2e_uses_current_copy_slots) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *src1 = NULL, *dst1 = NULL;
  PolyUOp *sink1 = make_copy_sink(ctx, &src1, &dst1);
  float src1_data[4] = {1, 2, 3, 4};
  float dst1_data[4] = {0};
  PolyTestBufferView views1[] = {
      POLY_TEST_HOST_VIEW(src1, src1_data),
      POLY_TEST_HOST_VIEW(dst1, dst1_data),
  };
  ASSERT_INT_EQ(poly_test_realize_buffer_views(ctx, sink1, views1, 2), 0);
  ASSERT_INT_EQ((int)poly_schedule_cache_len(ctx), 1);
  ASSERT_FLOAT_EQ(dst1_data[0], 1.0f, 1e-5);
  ASSERT_FLOAT_EQ(dst1_data[3], 4.0f, 1e-5);

  PolyUOp *src2 = NULL, *dst2 = NULL;
  PolyUOp *sink2 = make_copy_sink(ctx, &src2, &dst2);
  PolySchedule *sched2 = poly_complete_create_schedule_with_vars(ctx, sink2, POLY_MODE_CALL);
  ASSERT_NOT_NULL(sched2);
  ASSERT_INT_EQ((int)poly_schedule_cache_len(ctx), 1);
  int copy_call = -1;
  for (int i = 0; i < sched2->template->n_calls; i++)
    if (poly_schedule_call_is_copy(sched2, i)) copy_call = i;
  ASSERT_TRUE(copy_call >= 0);
  ASSERT_INT_EQ(poly_schedule_call_n_buffer_args(sched2, copy_call), 2);
  int dst_slot = poly_schedule_call_buffer_slot(sched2, copy_call, 0);
  int src_slot = poly_schedule_call_buffer_slot(sched2, copy_call, 1);
  ASSERT_TRUE(dst_slot >= 0);
  ASSERT_TRUE(src_slot >= 0);
  ASSERT_PTR_EQ(sched2->template->buf_slots[dst_slot].buf_uop, dst2);
  ASSERT_PTR_EQ(sched2->template->buf_slots[src_slot].buf_uop, src2);
  ASSERT_TRUE(schedule_has_buf_slot(sched2, src2));
  ASSERT_TRUE(schedule_has_buf_slot(sched2, dst2));
  ASSERT_FALSE(schedule_has_buf_slot(sched2, src1));
  ASSERT_FALSE(schedule_has_buf_slot(sched2, dst1));
  poly_schedule_free(sched2);

  float src2_data[4] = {10, 20, 30, 40};
  float dst2_data[4] = {0};
  PolyTestBufferView views2[] = {
      POLY_TEST_HOST_VIEW(src2, src2_data),
      POLY_TEST_HOST_VIEW(dst2, dst2_data),
  };
  ASSERT_INT_EQ(poly_test_realize_buffer_views(ctx, sink2, views2, 2), 0);
  ASSERT_INT_EQ((int)poly_schedule_cache_len(ctx), 1);
  ASSERT_FLOAT_EQ(dst2_data[0], 10.0f, 1e-5);
  ASSERT_FLOAT_EQ(dst2_data[3], 40.0f, 1e-5);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(schedule_runtime, copy_intermediate_slots_do_not_need_zero) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *a = poly_buffer_f32(ctx, 4);
  PolyUOp *b = poly_buffer_f32(ctx, 4);
  PolyUOp *out = poly_buffer_f32(ctx, 4);
  PolyUOp *dev = poly_uop0(ctx, POLY_OP_DEVICE, POLY_VOID, poly_arg_int(POLY_DEVICE_CPU));
  PolyUOp *copy_a_src[2] = {a, dev};
  PolyUOp *copy_b_src[2] = {b, dev};
  PolyUOp *copy_a = poly_uop(ctx, POLY_OP_COPY, POLY_FLOAT32, copy_a_src, 2, poly_arg_none());
  PolyUOp *copy_b = poly_uop(ctx, POLY_OP_COPY, POLY_FLOAT32, copy_b_src, 2, poly_arg_none());
  PolyUOp *sum = poly_alu2(ctx, POLY_OP_ADD, copy_a, copy_b);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, sum));

  PolySchedule *ps = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_NOT_NULL(ps);

  int n_copy_items = 0, n_intermediates = 0, n_zero = 0, n_arenas = 0, n_views = 0;
  for (int i = 0; i < ps->template->n_calls; i++)
    if (poly_schedule_call_is_copy(ps, i)) n_copy_items++;
  for (int i = 0; i < ps->template->n_buf_slots; i++) {
    if (!ps->template->buf_slots[i].is_intermediate) continue;
    n_intermediates++;
    if (ps->template->buf_slots[i].is_memory_arena) n_arenas++;
    if (ps->template->buf_slots[i].has_memory_parent) n_views++;
    if (ps->template->buf_slots[i].needs_zero) n_zero++;
  }

  ASSERT_INT_EQ(n_copy_items, 2);
  ASSERT_INT_EQ(n_intermediates, 3);
  ASSERT_INT_EQ(n_arenas, 1);
  ASSERT_INT_EQ(n_views, 2);
  ASSERT_INT_EQ(n_zero, 0);

  size_t expected_runtime_bytes = 0;
  for (int i = 0; i < ps->template->n_buf_slots; i++) {
    PolyScheduleBufSlot *slot = &ps->template->buf_slots[i];
    if (slot->is_intermediate && !slot->has_memory_parent)
      expected_runtime_bytes += (size_t)slot->nbytes;
  }
  ASSERT_TRUE(expected_runtime_bytes > 0);
  ASSERT_INT_EQ(poly_schedule_runtime_intermediate_bytes(ps), 0);

  float a_data[4] = {1, 2, 3, 4};
  float b_data[4] = {10, 20, 30, 40};
  float out_data[4] = {0};
  PolyBuffer a_view = poly_buffer_make_host_view(a_data, sizeof(a_data));
  PolyBuffer b_view = poly_buffer_make_host_view(b_data, sizeof(b_data));
  PolyBuffer out_view = poly_buffer_make_host_view(out_data, sizeof(out_data));
  poly_buffer_attach(ctx, a, &a_view);
  poly_buffer_attach(ctx, b, &b_view);
  poly_buffer_attach(ctx, out, &out_view);

  ASSERT_INT_EQ(poly_run_schedule(ctx, ps, NULL, 0), 0);
  ASSERT_INT_EQ(poly_schedule_runtime_intermediate_bytes(ps), expected_runtime_bytes);
  ASSERT_FLOAT_EQ(out_data[0], 11.0f, 1e-5);
  ASSERT_FLOAT_EQ(out_data[3], 44.0f, 1e-5);

  PolyCompiledSchedule *plan = poly_lower_schedule(ctx, ps, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(plan);
  ASSERT_INT_EQ(poly_compiled_schedule_runtime_intermediate_bytes(plan), expected_runtime_bytes);
  poly_compiled_schedule_free(plan);

  poly_schedule_free(ps);
  poly_ctx_destroy(ctx);
  PASS();
}

typedef struct {
  int n_calls;
  int n_intermediates;
  int n_owner_slots;
  int n_arenas;
  int n_views;
  size_t runtime_bytes;
  int64_t owner_slot_bytes;
  float out0;
  float out_last;
} MemoryPlanStats;

static int run_reduce_expand_memory_plan_case(MemoryPlanStats *stats) {
  PolyCtx *ctx = poly_ctx_new();
  if (!ctx || !stats) return -1;
  memset(stats, 0, sizeof(*stats));

  enum { N = 64, K = 3 };
  PolyUOp *inputs[K];
  PolyUOp *outs[K];
  PolyUOp *stores[K];
  float in_data[K][N];
  float out_data[K][N];
  for (int k = 0; k < K; k++) {
    inputs[k] = poly_buffer_f32(ctx, N);
    outs[k] = poly_buffer_f32(ctx, N);
    int64_t red_axes[1] = {0};
    int64_t one_shape[1] = {1};
    int64_t out_shape[1] = {N};
    PolyUOp *red = poly_reduce_axis(ctx, POLY_OP_ADD, inputs[k], red_axes, 1);
    PolyUOp *exp = poly_expand(ctx, poly_reshape(ctx, red, one_shape, 1), out_shape, 1);
    stores[k] =
        poly_store_val(ctx, outs[k], poly_alu2(ctx, POLY_OP_ADD, exp, poly_const_float(ctx, k)));
    for (int i = 0; i < N; i++) {
      in_data[k][i] = 1.0f;
      out_data[k][i] = 0.0f;
    }
  }

  PolySchedule *ps = poly_complete_create_schedule_with_vars(
      ctx,
      poly_sink_n(ctx, stores, K),
      POLY_MODE_CALL
  );
  if (!ps) {
    poly_ctx_destroy(ctx);
    return -1;
  }

  stats->n_calls = ps->template->n_calls;
  for (int i = 0; i < ps->template->n_buf_slots; i++) {
    PolyScheduleBufSlot *slot = &ps->template->buf_slots[i];
    if (!slot->is_intermediate) continue;
    stats->n_intermediates++;
    if (slot->is_memory_arena) stats->n_arenas++;
    if (slot->has_memory_parent) stats->n_views++;
    if (!slot->has_memory_parent) {
      stats->n_owner_slots++;
      stats->owner_slot_bytes += slot->nbytes;
    }
  }

  for (int k = 0; k < K; k++) {
    PolyBuffer in_view = poly_buffer_make_host_view(in_data[k], sizeof(in_data[k]));
    PolyBuffer out_view = poly_buffer_make_host_view(out_data[k], sizeof(out_data[k]));
    poly_buffer_attach(ctx, inputs[k], &in_view);
    poly_buffer_attach(ctx, outs[k], &out_view);
  }

  int rc = poly_run_schedule(ctx, ps, NULL, 0);
  stats->runtime_bytes = poly_schedule_runtime_intermediate_bytes(ps);
  stats->out0 = out_data[0][0];
  stats->out_last = out_data[K - 1][N - 1];

  poly_schedule_free(ps);
  poly_ctx_destroy(ctx);
  return rc;
}

TEST(schedule_runtime, compute_intermediates_are_memory_planned_into_views) {
  /* Mirrors tinygrad's memory_plan_rewrite shape for three scalar reductions:
   * tinygrad rewrites the three reduction output buffers to SLICEs into one
   * char arena. The 256-byte TLSF block size means this increases tiny scalar
   * runtime bytes; this test is about representation parity, not peak savings. */
  ScheduleEnvSave saved = schedule_save_env("POLY_NO_MEMORY_PLANNER");

  unsetenv("POLY_NO_MEMORY_PLANNER");
  MemoryPlanStats planned;
  ASSERT_INT_EQ(run_reduce_expand_memory_plan_case(&planned), 0);

  setenv("POLY_NO_MEMORY_PLANNER", "1", 1);
  MemoryPlanStats disabled;
  ASSERT_INT_EQ(run_reduce_expand_memory_plan_case(&disabled), 0);

  schedule_restore_env(&saved);

  ASSERT_INT_EQ(planned.n_calls, 6);
  ASSERT_INT_EQ(disabled.n_calls, planned.n_calls);
  ASSERT_INT_EQ(planned.n_arenas, 1);
  ASSERT_INT_EQ(planned.n_views, 3);
  ASSERT_INT_EQ(planned.n_owner_slots, 1);
  ASSERT_INT_EQ(disabled.n_arenas, 0);
  ASSERT_INT_EQ(disabled.n_views, 0);
  ASSERT_INT_EQ(disabled.n_owner_slots, 3);
  ASSERT_INT_EQ((int)disabled.runtime_bytes, 3 * (int)sizeof(float));
  ASSERT_TRUE(planned.runtime_bytes > disabled.runtime_bytes);
  ASSERT_FLOAT_EQ(planned.out0, 64.0f, 1e-5);
  ASSERT_FLOAT_EQ(planned.out_last, 66.0f, 1e-5);
  ASSERT_FLOAT_EQ(disabled.out0, 64.0f, 1e-5);
  ASSERT_FLOAT_EQ(disabled.out_last, 66.0f, 1e-5);
  PASS();
}

TEST(schedule_runtime, webgpu_intermediates_are_memory_planned_into_views) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *a = poly_buffer_f32(ctx, 4);
  PolyUOp *b = poly_buffer_f32(ctx, 4);
  PolyUOp *out = poly_buffer_f32(ctx, 4);
  PolyUOp *dev = poly_uop0(ctx, POLY_OP_DEVICE, POLY_VOID, poly_arg_int(POLY_DEVICE_WEBGPU));
  PolyUOp *copy_a_src[2] = {a, dev};
  PolyUOp *copy_b_src[2] = {b, dev};
  PolyUOp *copy_a = poly_uop(ctx, POLY_OP_COPY, POLY_FLOAT32, copy_a_src, 2, poly_arg_none());
  PolyUOp *copy_b = poly_uop(ctx, POLY_OP_COPY, POLY_FLOAT32, copy_b_src, 2, poly_arg_none());
  PolyUOp *sum = poly_alu2(ctx, POLY_OP_ADD, copy_a, copy_b);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, sum));

  PolySchedule *ps = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_NOT_NULL(ps);

  int n_arenas = 0, n_views = 0;
  for (int i = 0; i < ps->template->n_buf_slots; i++) {
    PolyScheduleBufSlot *slot = &ps->template->buf_slots[i];
    if (!slot->is_intermediate) continue;
    if (slot->is_memory_arena) {
      n_arenas++;
      ASSERT_INT_EQ(slot->device, POLY_DEVICE_WEBGPU);
    }
    if (slot->has_memory_parent) {
      n_views++;
      ASSERT_TRUE(slot->memory_parent_slot >= 0);
      ASSERT_TRUE(slot->memory_offset >= 0);
      ASSERT_INT_EQ(slot->device, POLY_DEVICE_WEBGPU);
    }
  }

  ASSERT_TRUE(n_arenas >= 1);
  ASSERT_TRUE(n_views >= 2);

  poly_schedule_free(ps);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(schedule_runtime, cached_linear_multikernel_e2e_uses_current_buffers) {
  int N = 4;
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *a1 = poly_buffer_f32(ctx, N);
  PolyUOp *b1 = poly_buffer_f32(ctx, N);
  PolyUOp *out1 = poly_buffer_f32(ctx, N);
  int64_t axes[] = {0};
  int64_t one_sh[] = {1};
  int64_t exp_sh[] = {N};
  PolyUOp *s1 = poly_reduce_axis(ctx, POLY_OP_ADD, a1, axes, 1);
  PolyUOp *se1 = poly_expand(ctx, poly_reshape(ctx, s1, one_sh, 1), exp_sh, 1);
  PolyUOp *sink1 = poly_sink1(ctx, poly_store_val(ctx, out1, poly_alu2(ctx, POLY_OP_ADD, se1, b1)));
  float a1_data[4] = {1, 2, 3, 4};
  float b1_data[4] = {10, 20, 30, 40};
  float out1_data[4] = {0};
  PolyTestBufferView views1[] = {
      POLY_TEST_HOST_VIEW(a1, a1_data),
      POLY_TEST_HOST_VIEW(b1, b1_data),
      POLY_TEST_HOST_VIEW(out1, out1_data),
  };
  PolySchedule *sched1 = poly_complete_create_schedule_with_vars(ctx, sink1, POLY_MODE_CALL);
  ASSERT_NOT_NULL(sched1);
  ASSERT_TRUE(sched1->template->n_calls > 1);
  poly_schedule_free(sched1);
  ASSERT_INT_EQ(poly_test_realize_buffer_views(ctx, sink1, views1, 3), 0);
  ASSERT_INT_EQ((int)poly_schedule_cache_len(ctx), 1);
  ASSERT_FLOAT_EQ(out1_data[0], 20.0f, 1e-5);
  ASSERT_FLOAT_EQ(out1_data[3], 50.0f, 1e-5);

  PolyUOp *a2 = poly_buffer_f32(ctx, N);
  PolyUOp *b2 = poly_buffer_f32(ctx, N);
  PolyUOp *out2 = poly_buffer_f32(ctx, N);
  PolyUOp *s2 = poly_reduce_axis(ctx, POLY_OP_ADD, a2, axes, 1);
  PolyUOp *se2 = poly_expand(ctx, poly_reshape(ctx, s2, one_sh, 1), exp_sh, 1);
  PolyUOp *sink2 = poly_sink1(ctx, poly_store_val(ctx, out2, poly_alu2(ctx, POLY_OP_ADD, se2, b2)));
  float a2_data[4] = {5, 6, 7, 8};
  float b2_data[4] = {100, 200, 300, 400};
  float out2_data[4] = {0};
  PolyTestBufferView views2[] = {
      POLY_TEST_HOST_VIEW(a2, a2_data),
      POLY_TEST_HOST_VIEW(b2, b2_data),
      POLY_TEST_HOST_VIEW(out2, out2_data),
  };
  ASSERT_INT_EQ(poly_test_realize_buffer_views(ctx, sink2, views2, 3), 0);
  ASSERT_INT_EQ((int)poly_schedule_cache_len(ctx), 1);
  ASSERT_FLOAT_EQ(out2_data[0], 126.0f, 1e-5);
  ASSERT_FLOAT_EQ(out2_data[3], 426.0f, 1e-5);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(schedule_runtime, full_rewrite_vecadd_upcast_preserves_end_ranges_like_tinygrad) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_TRUE(ctx != NULL);

  PolyUOp *a = poly_buffer_f32(ctx, 16);
  PolyUOp *b = poly_buffer_f32(ctx, 16);
  PolyUOp *c = poly_alu2(ctx, POLY_OP_ADD, a, b);
  PolyUOp *out = poly_buffer_f32(ctx, 16);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, c));

  PolySchedule *sched = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_TRUE(sched != NULL);
  ASSERT_INT_EQ(sched->template->n_calls, 1);

  PolyRewriteOpts opts = {
      .optimize = true,
      .devectorize = 1,
      .beam_width = 0,
      .caps = poly_c_renderer_caps(),
      .device = POLY_DEVICE_CPU,
      .opt_policy = POLY_OPT_HEURISTIC,
  };

  PolyUOp *rewritten = poly_full_rewrite_to_sink_ex(ctx, poly_schedule_call_body(sched, 0), opts);
  ASSERT_TRUE(rewritten != NULL);
  ASSERT_INT_EQ(count_root_ops(ctx, rewritten, POLY_OP_END), 1);
  ASSERT_INT_EQ(count_root_ops(ctx, rewritten, POLY_OP_RANGE), 1);

  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, rewritten, &n_topo);
  int end_with_non_range = 0;
  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    if (!u || u->op != POLY_OP_END) continue;
    for (int j = 1; j < u->n_src; j++) {
      if (!u->src[j] || u->src[j]->op != POLY_OP_RANGE) end_with_non_range++;
    }
  }
  ASSERT_INT_EQ(end_with_non_range, 0);

  poly_schedule_free(sched);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(schedule_runtime, linearize_vecadd_upcast_has_no_singleton_group_like_tinygrad) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_TRUE(ctx != NULL);

  PolyUOp *a = poly_buffer_f32(ctx, 16);
  PolyUOp *b = poly_buffer_f32(ctx, 16);
  PolyUOp *c = poly_alu2(ctx, POLY_OP_ADD, a, b);
  PolyUOp *out = poly_buffer_f32(ctx, 16);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, c));

  PolySchedule *sched = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_TRUE(sched != NULL);
  ASSERT_INT_EQ(sched->template->n_calls, 1);

  int n_lin = 0;
  PolyUOp **lin = poly_linearize(ctx, poly_schedule_call_body(sched, 0), &n_lin);
  ASSERT_TRUE(lin != NULL);
  ASSERT_TRUE(n_lin > 0);

  int n_group = 0;
  for (int i = 0; i < n_lin; i++) {
    if (lin[i] && lin[i]->op == POLY_OP_GROUP) n_group++;
  }
  ASSERT_INT_EQ(n_group, 0);

  free(lin);
  poly_schedule_free(sched);
  poly_ctx_destroy(ctx);
  PASS();
}

/* Multi-kernel: reduce -> scalar chain */

TEST(schedule_runtime, prepare_multikernel) {
  /* c = expand(reshape(sum(a), (1))) + b  -- sum produces intermediate */
  int N = 8;
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *b = poly_buffer(ctx, POLY_FLOAT32, N);
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

  PolySchedule *ps = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_TRUE(ps != NULL);

  /* Should have multiple kernels (reduce produces intermediate) */
  ASSERT_TRUE(ps->template->n_calls >= 2);

  /* Should have intermediate buffer slots */
  int n_inter = 0;
  for (int i = 0; i < ps->template->n_buf_slots; i++)
    if (ps->template->buf_slots[i].is_intermediate) n_inter++;
  ASSERT_TRUE(n_inter > 0);

  /* All calls should be COMPUTE */
  for (int i = 0; i < ps->template->n_calls; i++)
    ASSERT_FALSE(poly_schedule_call_is_copy(ps, i));

  /* All buf_slot_indices should be valid */
  for (int i = 0; i < ps->template->n_calls; i++)
    for (int j = 0; j < poly_schedule_call_n_buffer_args(ps, i); j++) {
      ASSERT_TRUE(poly_schedule_call_buffer_slot(ps, i, j) >= 0);
      ASSERT_TRUE(poly_schedule_call_buffer_slot(ps, i, j) < ps->template->n_buf_slots);
    }

  ASSERT_NOT_NULL(ps->template->linear);
  ASSERT_INT_EQ(ps->template->linear->n_src, ps->template->n_calls);

  poly_schedule_free(ps);
  poly_ctx_destroy(ctx);
  PASS();
}

/* Buffer slot metadata */

TEST(schedule_runtime, prepare_buf_slot_metadata) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer_f32(ctx, 8);
  PolyUOp *b = poly_buffer_f64(ctx, 8);
  PolyUOp *c = poly_alu2(ctx, POLY_OP_ADD, a, poly_cast_by_id(ctx, b, 12));
  PolyUOp *out = poly_buffer_f32(ctx, 8);
  PolyUOp *st = poly_store_val(ctx, out, c);
  PolyUOp *sink = poly_sink1(ctx, st);

  PolySchedule *ps = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_TRUE(ps != NULL);

  /* Check dtype and size are populated */
  for (int i = 0; i < ps->template->n_buf_slots; i++) {
    if (!ps->template->buf_slots[i].is_intermediate) {
      ASSERT_TRUE(ps->template->buf_slots[i].numel > 0);
      ASSERT_TRUE(ps->template->buf_slots[i].nbytes > 0);
    }
  }

  poly_schedule_free(ps);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(schedule_runtime, webgpu_triu_param_order_matches_runtime_slots) {
  if (!poly_backend_get(POLY_DEVICE_WEBGPU)) {
    SKIP("WebGPU runtime backend not available in this build");
  }

  PolyCtx *ctx = poly_ctx_new();
  ASSERT_TRUE(ctx != NULL);

  PolyUOp *in = poly_buffer_f32(ctx, 9);
  PolyUOp *in2d = poly_reshape(ctx, in, (int64_t[]){3, 3}, 2);
  PolyUOp *tri = poly_triu(ctx, in2d, 0);
  ASSERT_TRUE(tri != NULL);

  PolyUOp *out = poly_buffer_f32(ctx, 9);
  PolyUOp *store = poly_store_val(ctx, out, tri);
  PolyUOp *sink = poly_sink1(ctx, store);

  PolySchedule *sched = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_TRUE(sched != NULL);
  ASSERT_TRUE(sched->template->n_calls >= 1);

  bool checked_kernel = false;
  for (int k = 0; k < sched->template->n_calls; k++) {
    ASSERT_INT_EQ(poly_schedule_call_lower(ctx, sched, k, POLY_DEVICE_WEBGPU), 0);
    PolyCallRuntime *rt = &sched->run->calls[k];
    int n_params = poly_schedule_call_n_buffer_args(sched, k);
    ASSERT_INT_EQ(rt->prg.n_params, n_params);

    int n_lin = 0;
    PolyUOp **lin = poly_linearize_webgpu(ctx, poly_schedule_call_body(sched, k), &n_lin);
    ASSERT_TRUE(lin != NULL);

    int seen_params = 0;
    for (int i = 0; i < n_lin; i++) {
      if (lin[i]->op != POLY_OP_PARAM) continue;
      ASSERT_TRUE(seen_params < n_params);
      ASSERT_INT_EQ((int)lin[i]->arg.i, seen_params);
      ASSERT_INT_EQ(
          rt->prg.param_to_slot[seen_params],
          poly_schedule_call_buffer_slot(sched, k, (int)lin[i]->arg.i)
      );
      seen_params++;
    }
    ASSERT_INT_EQ(seen_params, n_params);
    free(lin);
    checked_kernel = true;
  }

  ASSERT_TRUE(checked_kernel);
  poly_schedule_free(sched);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(schedule_runtime, webgpu_unified_triu_renders_single_kernel_no_helper_params) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_TRUE(ctx != NULL);

  PolyUOp *in = poly_buffer_f32(ctx, 9);
  PolyUOp *in2d = poly_reshape(ctx, in, (int64_t[]){3, 3}, 2);
  PolyUOp *tri = poly_triu(ctx, in2d, 0);
  ASSERT_TRUE(tri != NULL);

  PolyUOp *out = poly_buffer_f32(ctx, 9);
  PolyUOp *store = poly_store_val(ctx, out, tri);
  PolyUOp *sink = poly_sink1(ctx, store);

  PolySchedule *sched = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_TRUE(sched != NULL);
  ASSERT_INT_EQ(sched->template->n_calls, 1);
  ASSERT_INT_EQ(poly_schedule_call_n_buffer_args(sched, 0), 2);

  int n_lin = 0;
  PolyUOp **lin = poly_linearize_webgpu(ctx, poly_schedule_call_body(sched, 0), &n_lin);
  ASSERT_TRUE(lin != NULL);
  ASSERT_TRUE(n_lin > 0);

  char *wgsl = poly_render_wgsl(lin, n_lin, "triu_unified");
  ASSERT_TRUE(wgsl != NULL);
  ASSERT_NOT_NULL(strstr(wgsl, "fn triu_unified("));
  ASSERT_NOT_NULL(strstr(wgsl, "var<storage,read_write> data0: array<f32>;"));
  ASSERT_NOT_NULL(strstr(wgsl, "var<storage,read_write> data1: array<f32>;"));
  ASSERT_TRUE(strstr(wgsl, "var<storage,read_write> data2: array<f32>;") == NULL);

  if (poly_dump_kernels_enabled())
    fprintf(stderr, "=== UNIFIED WEBGPU KERNEL triu_unified ===\n%s\n=== END ===\n", wgsl);

  free(wgsl);
  free(lin);
  poly_schedule_free(sched);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(schedule_runtime, webgpu_unified_triu_has_no_vector_gated_loads) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_TRUE(ctx != NULL);

  PolyUOp *in = poly_buffer_f32(ctx, 9);
  PolyUOp *in2d = poly_reshape(ctx, in, (int64_t[]){3, 3}, 2);
  PolyUOp *tri = poly_triu(ctx, in2d, 0);
  ASSERT_TRUE(tri != NULL);

  PolyUOp *out = poly_buffer_f32(ctx, 9);
  PolyUOp *store = poly_store_val(ctx, out, tri);
  PolyUOp *sink = poly_sink1(ctx, store);

  PolySchedule *sched = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_TRUE(sched != NULL);
  ASSERT_INT_EQ(sched->template->n_calls, 1);

  int n_lin = 0;
  PolyUOp **lin = poly_linearize_webgpu(ctx, poly_schedule_call_body(sched, 0), &n_lin);
  ASSERT_TRUE(lin != NULL);
  ASSERT_TRUE(n_lin > 0);

  int vec_gate_loads = 0;
  int vec_alt_loads = 0;
  for (int i = 0; i < n_lin; i++) {
    PolyUOp *u = lin[i];
    if (!u || u->op != POLY_OP_LOAD) continue;
    PolyUOp *idx = poly_find_index_through_cast(u->src[0]);
    if (idx && idx->op == POLY_OP_INDEX && idx->n_src >= 3 && idx->src[2] &&
        idx->src[2]->dtype.count > 1)
      vec_gate_loads++;
    if (u->n_src >= 2 && u->src[1] && u->src[1]->dtype.count > 1) vec_alt_loads++;
  }

  ASSERT_INT_EQ(vec_gate_loads, 0);
  ASSERT_INT_EQ(vec_alt_loads, 0);

  free(lin);
  poly_schedule_free(sched);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(schedule_runtime, webgpu_triu_move_where_keeps_residual_where_like_tinygrad) {
  for (int n = 3; n <= 9; n += 6) {
    PolyCtx *ctx = poly_ctx_new();
    ASSERT_TRUE(ctx != NULL);

    PolyUOp *in = poly_buffer_f32(ctx, n * n);
    PolyUOp *in2d = poly_reshape(ctx, in, (int64_t[]){n, n}, 2);
    PolyUOp *tri = poly_triu(ctx, in2d, 0);
    ASSERT_TRUE(tri != NULL);

    PolyUOp *out = poly_buffer_f32(ctx, n * n);
    PolyUOp *store = poly_store_val(ctx, out, tri);
    PolyUOp *sink = poly_sink1(ctx, store);

    PolySchedule *sched = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
    ASSERT_TRUE(sched != NULL);
    ASSERT_INT_EQ(sched->template->n_calls, 1);

    /* Match tinygrad's current boundary check from temp/tiny_triu_probe_stages.py:
     * postopt symbolic should keep a residual WHERE after only the safe clauses
     * move into INDEX.valid. */
    PolyRewriteOpts opts = {
        .optimize = true,
        .devectorize = 1,
        .beam_width = 0,
        .caps =
            {
                .has_mulacc = false,
                .has_threefry = false,
                .has_local = true,
                .max_vec_width = 1,
            },
        .device = POLY_DEVICE_WEBGPU,
        .opt_policy = POLY_OPT_HEURISTIC,
        .extra_matcher = poly_pm_wgsl_extra(),
        .gpu_block_size = 256,
    };

    PolyUOp *u = poly_schedule_call_body(sched, 0);
    u = poly_apply_opts_heuristic_ex(ctx, u, opts.caps);
    u = poly_graph_rewrite(ctx, u, poly_symbolic());
    u = poly_graph_rewrite(ctx, u, poly_pm_move_where_on_load_pass());

    ASSERT_INT_EQ(count_root_ops(ctx, u, POLY_OP_LOAD), 0);
    ASSERT_INT_EQ(count_root_ops(ctx, u, POLY_OP_WHERE), 1);

    poly_schedule_free(sched);
    poly_ctx_destroy(ctx);
  }
  PASS();
}

TEST(schedule_runtime, webgpu_triu_heuristic_adds_local_split_like_tinygrad) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_TRUE(ctx != NULL);

  PolyUOp *in = poly_buffer_f32(ctx, 81);
  PolyUOp *in2d = poly_reshape(ctx, in, (int64_t[]){9, 9}, 2);
  PolyUOp *tri = poly_triu(ctx, in2d, 0);
  ASSERT_TRUE(tri != NULL);

  PolyUOp *out = poly_buffer_f32(ctx, 81);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, tri));

  PolySchedule *sched = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_TRUE(sched != NULL);
  ASSERT_INT_EQ(sched->template->n_calls, 1);

  PolyRendererCaps caps = {
      .has_mulacc = false,
      .has_threefry = false,
      .has_local = true,
      .max_vec_width = 1,
  };
  PolyUOp *heur = poly_apply_opts_heuristic_ex(ctx, poly_schedule_call_body(sched, 0), caps);

  ASSERT_INT_EQ(count_root_ranges_of_type(ctx, heur, POLY_AXIS_LOOP), 2);
  ASSERT_INT_EQ(count_root_ranges_of_type(ctx, heur, POLY_AXIS_LOCAL), 2);
  ASSERT_INT_EQ(count_root_ranges_of_type(ctx, heur, POLY_AXIS_UPCAST), 0);

  poly_schedule_free(sched);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(schedule_runtime, webgpu_triu_gpudims_replaces_all_ranges_like_tinygrad) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_TRUE(ctx != NULL);

  PolyUOp *in = poly_buffer_f32(ctx, 81);
  PolyUOp *in2d = poly_reshape(ctx, in, (int64_t[]){9, 9}, 2);
  PolyUOp *tri = poly_triu(ctx, in2d, 0);
  ASSERT_TRUE(tri != NULL);

  PolyUOp *out = poly_buffer_f32(ctx, 81);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, tri));

  PolySchedule *sched = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_TRUE(sched != NULL);
  ASSERT_INT_EQ(sched->template->n_calls, 1);

  PolyRendererCaps caps = {
      .has_mulacc = false,
      .has_threefry = false,
      .has_local = true,
      .max_vec_width = 1,
  };

  PolyUOp *u = poly_apply_opts_heuristic_ex(ctx, poly_schedule_call_body(sched, 0), caps);
  u = poly_graph_rewrite(ctx, u, poly_symbolic());
  u = poly_graph_rewrite(ctx, u, poly_pm_move_where_on_load_pass());
  u = poly_graph_rewrite(ctx, u, poly_symbolic());
  u = poly_graph_rewrite(ctx, u, poly_pm_pre_expander_pass());
  u = poly_group_for_reduce(ctx, u, 256);
  u = poly_graph_rewrite(ctx, u, poly_pm_expander_pass());
  u = poly_graph_rewrite(ctx, u, poly_symbolic());
  u = poly_apply_pm_reduce(ctx, u);
  u = poly_add_gpudims(ctx, u);

  ASSERT_INT_EQ(count_root_ranges_of_type(ctx, u, POLY_AXIS_LOOP), 0);
  ASSERT_INT_EQ(count_root_ranges_of_type(ctx, u, POLY_AXIS_GLOBAL), 0);
  ASSERT_INT_EQ(count_root_ranges_of_type(ctx, u, POLY_AXIS_LOCAL), 0);
  ASSERT_INT_EQ(count_root_ops(ctx, u, POLY_OP_WHERE), 1);
  ASSERT_TRUE(count_root_ops(ctx, u, POLY_OP_SPECIAL) >= 1);

  poly_schedule_free(sched);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(schedule_runtime, webgpu_triu_small_expander_drops_residual_where_like_tinygrad) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_TRUE(ctx != NULL);

  PolySchedule *sched = build_tri_schedule(ctx, 3, false);
  ASSERT_TRUE(sched != NULL);
  ASSERT_INT_EQ(sched->template->n_calls, 1);

  PolyRendererCaps caps = {
      .has_mulacc = false,
      .has_threefry = false,
      .has_local = true,
      .max_vec_width = 1,
  };

  PolyUOp *u = poly_apply_opts_heuristic_ex(ctx, poly_schedule_call_body(sched, 0), caps);
  u = poly_graph_rewrite(ctx, u, poly_symbolic());
  u = poly_graph_rewrite(ctx, u, poly_pm_move_where_on_load_pass());
  u = poly_graph_rewrite(ctx, u, poly_symbolic());
  u = poly_graph_rewrite(ctx, u, poly_pm_pre_expander_pass());
  u = poly_group_for_reduce(ctx, u, 256);
  u = poly_graph_rewrite(ctx, u, poly_pm_expander_pass());
  u = poly_graph_rewrite(ctx, u, poly_symbolic());

  ASSERT_INT_EQ(count_root_ops(ctx, u, POLY_OP_WHERE), 0);

  poly_schedule_free(sched);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(schedule_runtime, webgpu_tril_small_expander_drops_residual_where_like_tinygrad) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_TRUE(ctx != NULL);

  PolySchedule *sched = build_tri_schedule(ctx, 3, true);
  ASSERT_TRUE(sched != NULL);
  ASSERT_INT_EQ(sched->template->n_calls, 1);

  PolyRendererCaps caps = {
      .has_mulacc = false,
      .has_threefry = false,
      .has_local = true,
      .max_vec_width = 1,
  };

  PolyUOp *u = poly_apply_opts_heuristic_ex(ctx, poly_schedule_call_body(sched, 0), caps);
  u = poly_graph_rewrite(ctx, u, poly_symbolic());
  u = poly_graph_rewrite(ctx, u, poly_pm_move_where_on_load_pass());
  u = poly_graph_rewrite(ctx, u, poly_symbolic());
  u = poly_graph_rewrite(ctx, u, poly_pm_pre_expander_pass());
  u = poly_group_for_reduce(ctx, u, 256);
  u = poly_graph_rewrite(ctx, u, poly_pm_expander_pass());
  u = poly_graph_rewrite(ctx, u, poly_symbolic());

  ASSERT_INT_EQ(count_root_ops(ctx, u, POLY_OP_WHERE), 0);

  poly_schedule_free(sched);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(schedule_runtime, webgpu_triu_small_devector_matches_tinygrad_load_count) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_TRUE(ctx != NULL);

  PolySchedule *sched = build_tri_schedule(ctx, 3, false);
  ASSERT_TRUE(sched != NULL);
  ASSERT_INT_EQ(sched->template->n_calls, 1);

  PolyRendererCaps caps = {
      .has_mulacc = false,
      .has_threefry = false,
      .has_local = true,
      .max_vec_width = 1,
  };

  PolyUOp *u = poly_apply_opts_heuristic_ex(ctx, poly_schedule_call_body(sched, 0), caps);
  u = poly_graph_rewrite(ctx, u, poly_symbolic());
  u = poly_graph_rewrite(ctx, u, poly_pm_move_where_on_load_pass());
  u = poly_graph_rewrite(ctx, u, poly_symbolic());
  u = poly_graph_rewrite(ctx, u, poly_pm_pre_expander_pass());
  u = poly_group_for_reduce(ctx, u, 256);
  u = poly_graph_rewrite(ctx, u, poly_pm_expander_pass());
  u = poly_graph_rewrite(ctx, u, poly_symbolic());
  u = poly_apply_pm_reduce(ctx, u);
  u = poly_add_gpudims(ctx, u);
  u = poly_graph_rewrite(ctx, u, poly_pm_add_loads_pass());
  u = poly_apply_devectorize_stage(ctx, u, 1, caps);

  /* tinygrad triu(3): devector LOAD=6, WHERE=0 on the checked-in reference. */
  ASSERT_INT_EQ(count_root_ops(ctx, u, POLY_OP_LOAD), 6);
  ASSERT_INT_EQ(count_root_ops(ctx, u, POLY_OP_WHERE), 0);

  poly_schedule_free(sched);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(schedule_runtime, webgpu_tril_small_devector_matches_tinygrad_load_count) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_TRUE(ctx != NULL);

  PolySchedule *sched = build_tri_schedule(ctx, 3, true);
  ASSERT_TRUE(sched != NULL);
  ASSERT_INT_EQ(sched->template->n_calls, 1);

  PolyRendererCaps caps = {
      .has_mulacc = false,
      .has_threefry = false,
      .has_local = true,
      .max_vec_width = 1,
  };

  PolyUOp *u = poly_apply_opts_heuristic_ex(ctx, poly_schedule_call_body(sched, 0), caps);
  u = poly_graph_rewrite(ctx, u, poly_symbolic());
  u = poly_graph_rewrite(ctx, u, poly_pm_move_where_on_load_pass());
  u = poly_graph_rewrite(ctx, u, poly_symbolic());
  u = poly_graph_rewrite(ctx, u, poly_pm_pre_expander_pass());
  u = poly_group_for_reduce(ctx, u, 256);
  u = poly_graph_rewrite(ctx, u, poly_pm_expander_pass());
  u = poly_graph_rewrite(ctx, u, poly_symbolic());
  u = poly_apply_pm_reduce(ctx, u);
  u = poly_add_gpudims(ctx, u);
  u = poly_graph_rewrite(ctx, u, poly_pm_add_loads_pass());
  u = poly_apply_devectorize_stage(ctx, u, 1, caps);

  /* tinygrad tril(3): devector LOAD=6, WHERE=0 on the checked-in reference. */
  ASSERT_INT_EQ(count_root_ops(ctx, u, POLY_OP_LOAD), 6);
  ASSERT_INT_EQ(count_root_ops(ctx, u, POLY_OP_WHERE), 0);

  poly_schedule_free(sched);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(schedule_runtime, webgpu_triu_add_loads_keeps_residual_where_like_tinygrad) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_TRUE(ctx != NULL);

  PolyUOp *in = poly_buffer_f32(ctx, 81);
  PolyUOp *in2d = poly_reshape(ctx, in, (int64_t[]){9, 9}, 2);
  PolyUOp *tri = poly_triu(ctx, in2d, 0);
  ASSERT_TRUE(tri != NULL);

  PolyUOp *out = poly_buffer_f32(ctx, 81);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, tri));

  PolySchedule *sched = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_TRUE(sched != NULL);
  ASSERT_INT_EQ(sched->template->n_calls, 1);

  PolyRendererCaps caps = {
      .has_mulacc = false,
      .has_threefry = false,
      .has_local = true,
      .max_vec_width = 1,
  };

  PolyUOp *u = poly_apply_opts_heuristic_ex(ctx, poly_schedule_call_body(sched, 0), caps);
  u = poly_graph_rewrite(ctx, u, poly_symbolic());
  u = poly_graph_rewrite(ctx, u, poly_pm_move_where_on_load_pass());
  u = poly_graph_rewrite(ctx, u, poly_symbolic());
  u = poly_graph_rewrite(ctx, u, poly_pm_pre_expander_pass());
  u = poly_group_for_reduce(ctx, u, 256);
  u = poly_graph_rewrite(ctx, u, poly_pm_expander_pass());
  u = poly_graph_rewrite(ctx, u, poly_symbolic());
  u = poly_apply_pm_reduce(ctx, u);
  u = poly_add_gpudims(ctx, u);
  u = poly_graph_rewrite(ctx, u, poly_pm_add_loads_pass());

  ASSERT_INT_EQ(count_root_ops(ctx, u, POLY_OP_WHERE), 1);
  ASSERT_INT_EQ(count_root_ops(ctx, u, POLY_OP_LOAD), 1);
  ASSERT_INT_EQ(count_root_gated_loads(ctx, u), 0);

  poly_schedule_free(sched);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(schedule_runtime, webgpu_triu_post_index_symbolic_lowers_where_to_gated_load_like_tinygrad) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_TRUE(ctx != NULL);

  PolyUOp *in = poly_buffer_f32(ctx, 81);
  PolyUOp *in2d = poly_reshape(ctx, in, (int64_t[]){9, 9}, 2);
  PolyUOp *tri = poly_triu(ctx, in2d, 0);
  ASSERT_TRUE(tri != NULL);

  PolyUOp *out = poly_buffer_f32(ctx, 81);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, tri));

  PolySchedule *sched = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_TRUE(sched != NULL);
  ASSERT_INT_EQ(sched->template->n_calls, 1);

  PolyRendererCaps caps = {
      .has_mulacc = false,
      .has_threefry = false,
      .has_local = true,
      .max_vec_width = 1,
  };

  PolyUOp *u = poly_apply_opts_heuristic_ex(ctx, poly_schedule_call_body(sched, 0), caps);
  u = poly_graph_rewrite(ctx, u, poly_symbolic());
  u = poly_graph_rewrite(ctx, u, poly_pm_move_where_on_load_pass());
  u = poly_graph_rewrite(ctx, u, poly_symbolic());
  u = poly_graph_rewrite(ctx, u, poly_pm_pre_expander_pass());
  u = poly_group_for_reduce(ctx, u, 256);
  u = poly_graph_rewrite(ctx, u, poly_pm_expander_pass());
  u = poly_graph_rewrite(ctx, u, poly_symbolic());
  u = poly_apply_pm_reduce(ctx, u);
  u = poly_add_gpudims(ctx, u);
  u = poly_graph_rewrite(ctx, u, poly_pm_add_loads_pass());
  u = poly_apply_devectorize_stage(ctx, u, 1, caps);
  u = poly_apply_post_index_symbolic_stage(ctx, u, 1);

  /* tinygrad_latest WGSL pipeline keeps the residual WHERE through add_loads,
   * then pm_lower_index_dtype/load_store_indexing lowers it into INDEX.valid. */
  ASSERT_INT_EQ(count_root_ops(ctx, u, POLY_OP_WHERE), 0);
  ASSERT_INT_EQ(count_root_ops(ctx, u, POLY_OP_LOAD), 1);
  ASSERT_INT_EQ(count_root_gated_loads(ctx, u), 1);

  poly_schedule_free(sched);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(schedule_runtime, webgpu_triu_final_rewrite_keeps_gated_load_like_tinygrad) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_TRUE(ctx != NULL);

  PolyUOp *in = poly_buffer_f32(ctx, 81);
  PolyUOp *in2d = poly_reshape(ctx, in, (int64_t[]){9, 9}, 2);
  PolyUOp *tri = poly_triu(ctx, in2d, 0);
  ASSERT_TRUE(tri != NULL);

  PolyUOp *out = poly_buffer_f32(ctx, 81);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, tri));

  PolySchedule *sched = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_TRUE(sched != NULL);
  ASSERT_INT_EQ(sched->template->n_calls, 1);

  PolyRewriteOpts opts = {
      .optimize = true,
      .devectorize = 1,
      .beam_width = 0,
      .caps =
          {
              .has_mulacc = false,
              .has_threefry = false,
              .has_local = true,
              .max_vec_width = 1,
          },
      .device = POLY_DEVICE_WEBGPU,
      .opt_policy = POLY_OPT_HEURISTIC,
      .extra_matcher = poly_pm_wgsl_extra(),
      .gpu_block_size = 256,
  };

  PolyUOp *rewritten = poly_full_rewrite_to_sink_ex(ctx, poly_schedule_call_body(sched, 0), opts);
  ASSERT_TRUE(rewritten != NULL);
  /* Matches temp/tg_webgpu_tri_stage_probe_current.py: current tinygrad
   * final WGSL rewrite removes the residual WHERE without leaving an
   * INDEX.valid gate. */
  ASSERT_INT_EQ(count_root_ops(ctx, rewritten, POLY_OP_WHERE), 0);
  ASSERT_INT_EQ(count_root_ops(ctx, rewritten, POLY_OP_LOAD), 1);
  ASSERT_INT_EQ(count_root_gated_loads(ctx, rewritten), 0);

  poly_schedule_free(sched);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(schedule_runtime, webgpu_unified_tril_has_no_vector_gated_loads) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_TRUE(ctx != NULL);

  PolySchedule *sched = build_tri_schedule(ctx, 3, true);
  ASSERT_TRUE(sched != NULL);
  ASSERT_INT_EQ(sched->template->n_calls, 1);

  int n_lin = 0;
  PolyUOp **lin = poly_linearize_webgpu(ctx, poly_schedule_call_body(sched, 0), &n_lin);
  ASSERT_TRUE(lin != NULL);
  ASSERT_TRUE(n_lin > 0);

  int vec_gate_loads = 0;
  int vec_alt_loads = 0;
  for (int i = 0; i < n_lin; i++) {
    PolyUOp *u = lin[i];
    if (!u || u->op != POLY_OP_LOAD) continue;
    PolyUOp *idx = poly_find_index_through_cast(u->src[0]);
    if (idx && idx->op == POLY_OP_INDEX && idx->n_src >= 3 && idx->src[2] &&
        idx->src[2]->dtype.count > 1)
      vec_gate_loads++;
    if (u->n_src >= 2 && u->src[1] && u->src[1]->dtype.count > 1) vec_alt_loads++;
  }

  ASSERT_INT_EQ(vec_gate_loads, 0);
  ASSERT_INT_EQ(vec_alt_loads, 0);

  free(lin);
  poly_schedule_free(sched);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(schedule_runtime, webgpu_tril_add_loads_keeps_residual_where_like_tinygrad) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_TRUE(ctx != NULL);

  PolySchedule *sched = build_tri_schedule(ctx, 9, true);
  ASSERT_TRUE(sched != NULL);
  ASSERT_INT_EQ(sched->template->n_calls, 1);

  PolyUOp *u = apply_webgpu_tri_stage_root(ctx, poly_schedule_call_body(sched, 0), true, false);

  ASSERT_INT_EQ(count_root_ops(ctx, u, POLY_OP_WHERE), 1);
  ASSERT_INT_EQ(count_root_ops(ctx, u, POLY_OP_LOAD), 1);
  ASSERT_INT_EQ(count_root_gated_loads(ctx, u), 0);

  poly_schedule_free(sched);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(schedule_runtime, webgpu_tril_post_index_symbolic_lowers_where_to_gated_load_like_tinygrad) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_TRUE(ctx != NULL);

  PolySchedule *sched = build_tri_schedule(ctx, 9, true);
  ASSERT_TRUE(sched != NULL);
  ASSERT_INT_EQ(sched->template->n_calls, 1);

  PolyUOp *u = apply_webgpu_tri_stage_root(ctx, poly_schedule_call_body(sched, 0), true, true);

  /* tinygrad_latest WGSL pipeline keeps the residual WHERE through add_loads,
   * then pm_lower_index_dtype/load_store_indexing lowers it into INDEX.valid. */
  ASSERT_INT_EQ(count_root_ops(ctx, u, POLY_OP_WHERE), 0);
  ASSERT_INT_EQ(count_root_ops(ctx, u, POLY_OP_LOAD), 1);
  ASSERT_INT_EQ(count_root_gated_loads(ctx, u), 1);

  poly_schedule_free(sched);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(schedule_runtime, webgpu_tril_final_rewrite_keeps_gated_load_like_tinygrad) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_TRUE(ctx != NULL);

  PolySchedule *sched = build_tri_schedule(ctx, 9, true);
  ASSERT_TRUE(sched != NULL);
  ASSERT_INT_EQ(sched->template->n_calls, 1);

  PolyRewriteOpts opts = {
      .optimize = true,
      .devectorize = 1,
      .beam_width = 0,
      .caps =
          {
              .has_mulacc = false,
              .has_threefry = false,
              .has_local = true,
              .max_vec_width = 1,
          },
      .device = POLY_DEVICE_WEBGPU,
      .opt_policy = POLY_OPT_HEURISTIC,
      .extra_matcher = poly_pm_wgsl_extra(),
      .gpu_block_size = 256,
  };

  PolyUOp *rewritten = poly_full_rewrite_to_sink_ex(ctx, poly_schedule_call_body(sched, 0), opts);
  ASSERT_TRUE(rewritten != NULL);
  /* Matches temp/tg_webgpu_tri_stage_probe_current.py: current tinygrad
   * final WGSL rewrite removes the residual WHERE without leaving an
   * INDEX.valid gate. */
  ASSERT_INT_EQ(count_root_ops(ctx, rewritten, POLY_OP_WHERE), 0);
  ASSERT_INT_EQ(count_root_ops(ctx, rewritten, POLY_OP_LOAD), 1);
  ASSERT_INT_EQ(count_root_gated_loads(ctx, rewritten), 0);

  poly_schedule_free(sched);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(schedule_runtime, kernel_graph_triu_keeps_load_in_codegen_stage_only) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_TRUE(ctx != NULL);

  PolyUOp *in = poly_buffer_f32(ctx, 9);
  PolyUOp *in2d = poly_reshape(ctx, in, (int64_t[]){3, 3}, 2);
  PolyUOp *tri = poly_triu(ctx, in2d, 0);
  ASSERT_TRUE(tri != NULL);

  PolyUOp *out = poly_buffer_f32(ctx, 9);
  PolyUOp *store = poly_store_val(ctx, out, tri);
  PolyUOp *sink = poly_sink1(ctx, store);

  PolyKernelScheduleResult sr = poly_build_kernel_schedule(ctx, sink);
  ASSERT_TRUE(sr.n_kernels >= 1);

  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, sr.kernels[0], &n_topo);
  ASSERT_TRUE(topo != NULL);
  int n_load = 0;
  for (int i = 0; i < n_topo; i++)
    if (topo[i] && topo[i]->op == POLY_OP_LOAD) n_load++;

  /* tinygrad create_schedule/get_kernel_graph boundary keeps scheduled roots
   * in INDEX form. LOAD first appears later in codegen pm_add_loads. */
  ASSERT_INT_EQ(n_load, 0);

  poly_kernel_schedule_result_free(&sr);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(schedule_runtime, kernel_graph_triu_read_indices_stay_scalar_until_codegen) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_TRUE(ctx != NULL);

  PolyUOp *in = poly_buffer_f32(ctx, 9);
  PolyUOp *in2d = poly_reshape(ctx, in, (int64_t[]){3, 3}, 2);
  PolyUOp *tri = poly_triu(ctx, in2d, 0);
  ASSERT_TRUE(tri != NULL);

  PolyUOp *out = poly_buffer_f32(ctx, 9);
  PolyUOp *store = poly_store_val(ctx, out, tri);
  PolyUOp *sink = poly_sink1(ctx, store);

  PolyKernelScheduleResult sr = poly_build_kernel_schedule(ctx, sink);
  ASSERT_TRUE(sr.n_kernels >= 1);

  IndexKindCounts counts = count_index_kinds(ctx, sr.kernels[0]);
  ASSERT_INT_EQ(counts.store_target_index, 1);
  ASSERT_INT_EQ(counts.target_ptr_index, 1);
  ASSERT_INT_EQ(counts.read_ptr_index, 0);
  ASSERT_TRUE(counts.read_scalar_index >= 1);

  poly_kernel_schedule_result_free(&sr);
  poly_ctx_destroy(ctx);
  PASS();
}

/* Null/invalid input handling */

TEST(schedule_runtime, prepare_null_safety) {
  PolyCtx *ctx = poly_ctx_new();

  /* NULL sink */
  ASSERT_TRUE(poly_complete_create_schedule_with_vars(ctx, NULL, POLY_MODE_CALL) == NULL);

  /* Non-SINK UOp */
  PolyUOp *buf = poly_buffer_f32(ctx, 4);
  ASSERT_TRUE(poly_complete_create_schedule_with_vars(ctx, buf, POLY_MODE_CALL) == NULL);

  /* Free NULL is safe */
  poly_schedule_free(NULL);

  poly_ctx_destroy(ctx);
  PASS();
}

/* Graph hash is populated */

TEST(schedule_runtime, prepare_graph_hash) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer_f32(ctx, 4);
  PolyUOp *b = poly_buffer_f32(ctx, 4);
  PolyUOp *c = poly_alu2(ctx, POLY_OP_ADD, a, b);
  PolyUOp *out = poly_buffer_f32(ctx, 4);
  PolyUOp *st = poly_store_val(ctx, out, c);
  PolyUOp *sink = poly_sink1(ctx, st);

  PolySchedule *ps = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_TRUE(ps != NULL);
  ASSERT_TRUE(ps->template->graph_hash != 0);

  poly_schedule_free(ps);
  poly_ctx_destroy(ctx);
  PASS();
}

/* Phase 3: Lower + Run */

TEST(schedule_runtime, lower_and_run_vecadd) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer_f32(ctx, 4);
  PolyUOp *b = poly_buffer_f32(ctx, 4);
  PolyUOp *c = poly_alu2(ctx, POLY_OP_ADD, a, b);
  PolyUOp *out = poly_buffer_f32(ctx, 4);
  PolyUOp *st = poly_store_val(ctx, out, c);
  PolyUOp *sink = poly_sink1(ctx, st);

  PolySchedule *ps = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_TRUE(ps != NULL);

  PolyCompiledSchedule *es = poly_lower_schedule(ctx, ps, POLY_DEVICE_CPU);
  ASSERT_TRUE(es != NULL);
  ASSERT_TRUE(es->run != NULL);
  ASSERT_TRUE(ps->template->n_calls == 0 || es->run->calls != NULL);
  ASSERT_EQ(es->device, POLY_DEVICE_CPU);

  /* Prepare data */
  float da[] = {1, 2, 3, 4};
  float db[] = {10, 20, 30, 40};
  float dout[4] = {0};

  /* Build slot_data array indexed by buf_slot */
  void *slot_data[3];
  for (int i = 0; i < ps->template->n_buf_slots; i++) {
    if (ps->template->buf_slots[i].buf_uop == a)
      slot_data[i] = da;
    else if (ps->template->buf_slots[i].buf_uop == b)
      slot_data[i] = db;
    else if (ps->template->buf_slots[i].buf_uop == out)
      slot_data[i] = dout;
    else
      slot_data[i] = NULL;
  }

  int ret = poly_run_compiled_schedule(es, slot_data, ps->template->n_buf_slots, NULL, 0);
  ASSERT_INT_EQ(ret, 0);

  ASSERT_FLOAT_EQ(dout[0], 11.0f, 1e-6);
  ASSERT_FLOAT_EQ(dout[1], 22.0f, 1e-6);
  ASSERT_FLOAT_EQ(dout[2], 33.0f, 1e-6);
  ASSERT_FLOAT_EQ(dout[3], 44.0f, 1e-6);

  poly_compiled_schedule_free(es);
  poly_schedule_free(ps);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(schedule_runtime, compiled_schedule_retains_template_after_schedule_free) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer_f32(ctx, 4);
  PolyUOp *b = poly_buffer_f32(ctx, 4);
  PolyUOp *c = poly_alu2(ctx, POLY_OP_ADD, a, b);
  PolyUOp *out = poly_buffer_f32(ctx, 4);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, c));

  PolySchedule *ps = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_TRUE(ps != NULL);
  PolyCompiledSchedule *es = poly_lower_schedule(ctx, ps, POLY_DEVICE_CPU);
  ASSERT_TRUE(es != NULL);

  float da[] = {1, 2, 3, 4};
  float db[] = {10, 20, 30, 40};
  float dout[4] = {0};
  void *slot_data[3] = {0};
  int n_slots = ps->template->n_buf_slots;
  for (int i = 0; i < n_slots; i++) {
    if (ps->template->buf_slots[i].buf_uop == a)
      slot_data[i] = da;
    else if (ps->template->buf_slots[i].buf_uop == b)
      slot_data[i] = db;
    else if (ps->template->buf_slots[i].buf_uop == out)
      slot_data[i] = dout;
  }

  poly_schedule_free(ps);
  ASSERT_INT_EQ(poly_run_compiled_schedule(es, slot_data, n_slots, NULL, 0), 0);
  ASSERT_FLOAT_EQ(dout[0], 11.0f, 1e-6);
  ASSERT_FLOAT_EQ(dout[1], 22.0f, 1e-6);
  ASSERT_FLOAT_EQ(dout[2], 33.0f, 1e-6);
  ASSERT_FLOAT_EQ(dout[3], 44.0f, 1e-6);

  poly_compiled_schedule_free(es);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(schedule_runtime, compiled_schedule_uses_ctx_buffers_when_slots_missing) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer_f32(ctx, 4);
  PolyUOp *b = poly_buffer_f32(ctx, 4);
  PolyUOp *out = poly_buffer_f32(ctx, 4);
  PolyUOp *c = poly_alu2(ctx, POLY_OP_ADD, a, b);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, c));

  float da[] = {1, 2, 3, 4};
  float db[] = {10, 20, 30, 40};
  float got[4] = {0};
  ASSERT_INT_EQ(poly_buffer_write(ctx, a, da, sizeof(da)), 0);
  ASSERT_INT_EQ(poly_buffer_write(ctx, b, db, sizeof(db)), 0);

  PolySchedule *ps = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_TRUE(ps != NULL);
  PolyCompiledSchedule *es = poly_lower_schedule(ctx, ps, POLY_DEVICE_CPU);
  ASSERT_TRUE(es != NULL);

  ASSERT_INT_EQ(poly_run_compiled_schedule(es, NULL, 0, NULL, 0), 0);
  ASSERT_INT_EQ(poly_buffer_read(ctx, out, got, sizeof(got)), 0);

  ASSERT_FLOAT_EQ(got[0], 11.0f, 1e-6);
  ASSERT_FLOAT_EQ(got[1], 22.0f, 1e-6);
  ASSERT_FLOAT_EQ(got[2], 33.0f, 1e-6);
  ASSERT_FLOAT_EQ(got[3], 44.0f, 1e-6);

  poly_compiled_schedule_free(es);
  poly_schedule_free(ps);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(schedule_runtime, compiled_schedule_ctx_readwrite_arg_updates_current_residency) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer_f32(ctx, 4);
  PolyUOp *b = poly_buffer_f32(ctx, 4);
  PolyUOp *c = poly_alu2(ctx, POLY_OP_ADD, a, b);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, a, c));

  float a1[] = {1, 2, 3, 4};
  float a2[] = {5, 6, 7, 8};
  float db[] = {10, 20, 30, 40};
  float got[4] = {0};
  ASSERT_INT_EQ(poly_buffer_write(ctx, a, a1, sizeof(a1)), 0);
  ASSERT_INT_EQ(poly_buffer_write(ctx, b, db, sizeof(db)), 0);

  PolySchedule *ps = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_TRUE(ps != NULL);
  PolyCompiledSchedule *es = poly_lower_schedule(ctx, ps, POLY_DEVICE_CPU);
  ASSERT_TRUE(es != NULL);

  ASSERT_INT_EQ(poly_run_compiled_schedule(es, NULL, 0, NULL, 0), 0);
  ASSERT_INT_EQ(poly_buffer_read(ctx, a, got, sizeof(got)), 0);
  ASSERT_FLOAT_EQ(got[0], 11.0f, 1e-6);
  ASSERT_FLOAT_EQ(got[1], 22.0f, 1e-6);
  ASSERT_FLOAT_EQ(got[2], 33.0f, 1e-6);
  ASSERT_FLOAT_EQ(got[3], 44.0f, 1e-6);

  ASSERT_INT_EQ(poly_buffer_write(ctx, a, a2, sizeof(a2)), 0);
  ASSERT_INT_EQ(poly_run_compiled_schedule(es, NULL, 0, NULL, 0), 0);
  ASSERT_INT_EQ(poly_buffer_read(ctx, a, got, sizeof(got)), 0);
  ASSERT_FLOAT_EQ(got[0], 15.0f, 1e-6);
  ASSERT_FLOAT_EQ(got[1], 26.0f, 1e-6);
  ASSERT_FLOAT_EQ(got[2], 37.0f, 1e-6);
  ASSERT_FLOAT_EQ(got[3], 48.0f, 1e-6);

  poly_compiled_schedule_free(es);
  poly_schedule_free(ps);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(schedule_runtime, compiled_schedule_runtime_var_override_uses_ctx_buffers) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *N = poly_define_var(ctx, "N", 1, 16);
  PolyUOp *a = poly_buffer_var(ctx, POLY_FLOAT32, N, NULL, 0);
  PolyUOp *out = poly_buffer_var(ctx, POLY_FLOAT32, N, NULL, 0);
  PolyUOp *add = poly_alu2(ctx, POLY_OP_ADD, a, poly_const_float(ctx, 1.0f));
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, add));

  float a_data[16];
  float out_data[16];
  float got[16];
  for (int i = 0; i < 16; i++) {
    a_data[i] = (float)(i + 1);
    out_data[i] = -999.0f;
    got[i] = 0.0f;
  }
  ASSERT_INT_EQ(poly_buffer_write(ctx, a, a_data, sizeof(a_data)), 0);
  ASSERT_INT_EQ(poly_buffer_write(ctx, out, out_data, sizeof(out_data)), 0);

  PolySchedule *ps = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_TRUE(ps != NULL);
  PolyCompiledSchedule *es = poly_lower_schedule(ctx, ps, POLY_DEVICE_CPU);
  ASSERT_TRUE(es != NULL);

  ASSERT_INT_EQ(poly_run_compiled_schedule(es, NULL, 0, NULL, 0), -1);

  PolyVarBinding bind = {.var = N, .value = 6};
  ASSERT_INT_EQ(poly_run_compiled_schedule(es, NULL, 0, &bind, 1), 0);
  ASSERT_INT_EQ(poly_buffer_read(ctx, out, got, sizeof(got)), 0);
  for (int i = 0; i < 6; i++)
    ASSERT_FLOAT_EQ(got[i], (float)(i + 2), 1e-5f);
  ASSERT_FLOAT_EQ(got[6], -999.0f, 1e-5f);

  poly_compiled_schedule_free(es);
  poly_schedule_free(ps);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(schedule_runtime, cpu_threaded_render_uses_core_id_and_arg_slots) {
  ScheduleEnvSave threads_env = schedule_save_env("THREADS");
  ScheduleEnvSave cpu_count_env = schedule_save_env("CPU_COUNT");
  setenv("THREADS", "1", 1);
  setenv("CPU_COUNT", "4", 1);

  PolyCtx *ctx = poly_ctx_new();
  const int N = 1000000;
  PolyUOp *a = poly_buffer_f32(ctx, N);
  PolyUOp *b = poly_buffer_f32(ctx, N);
  PolyUOp *c = poly_alu2(ctx, POLY_OP_ADD, a, b);
  PolyUOp *out = poly_buffer_f32(ctx, N);
  PolyUOp *st = poly_store_val(ctx, out, c);
  PolyUOp *sink = poly_sink1(ctx, st);

  PolySchedule *ps = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_TRUE(ps != NULL);
  ASSERT_INT_EQ(ps->template->n_calls, 1);

  PolyRewriteOpts opts = {
      .optimize = true,
      .devectorize = 1,
      .caps = poly_c_renderer_caps(),
      .device = POLY_DEVICE_CPU,
      .opt_policy = POLY_OPT_HEURISTIC,
      .extra_matcher = poly_pm_c_renderer_extra(),
  };
  int n_lin = 0;
  PolyUOp **lin = poly_linearize_ex(ctx, poly_schedule_call_body(ps, 0), opts, &n_lin);
  ASSERT_TRUE(lin != NULL);
  char *src = poly_render_c(lin, n_lin, "thread_probe");
  ASSERT_TRUE(src != NULL);

  ASSERT_TRUE(strstr(src, "const int core_id") != NULL);
  ASSERT_TRUE(strstr(src, "thread_probe_call_core") != NULL);
  ASSERT_TRUE(
      strstr(src, "thread_probe((float*)args[0], (float*)args[1], (float*)args[2], core_id);") !=
      NULL
  );
  ASSERT_TRUE(strstr(src, "__attribute__((vector_size(16)))") != NULL);
  ASSERT_TRUE(strstr(src, "*((float __attribute__((vector_size(16)))*") == NULL);
  ASSERT_TRUE(strstr(src, "*(data0+") != NULL || strstr(src, "*(data1+") != NULL);

  free(src);
  free(lin);
  poly_schedule_free(ps);
  poly_ctx_destroy(ctx);
  schedule_restore_env(&cpu_count_env);
  schedule_restore_env(&threads_env);
  PASS();
}

TEST(schedule_runtime, cpu_threaded_large_vecadd_e2e) {
  ScheduleEnvSave threads_env = schedule_save_env("THREADS");
  ScheduleEnvSave cpu_count_env = schedule_save_env("CPU_COUNT");
  setenv("THREADS", "1", 1);
  setenv("CPU_COUNT", "4", 1);

  PolyCtx *ctx = poly_ctx_new();
  const int N = 1000000;
  PolyUOp *a = poly_buffer_f32(ctx, N);
  PolyUOp *b = poly_buffer_f32(ctx, N);
  PolyUOp *c = poly_alu2(ctx, POLY_OP_ADD, a, b);
  PolyUOp *out = poly_buffer_f32(ctx, N);
  PolyUOp *st = poly_store_val(ctx, out, c);
  PolyUOp *sink = poly_sink1(ctx, st);

  PolySchedule *ps = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_TRUE(ps != NULL);
  PolyCompiledSchedule *es = poly_lower_schedule(ctx, ps, POLY_DEVICE_CPU);
  ASSERT_TRUE(es != NULL);
  ASSERT_TRUE(es->run != NULL);
  ASSERT_INT_EQ(ps->template->n_calls, 1);
  ASSERT_INT_EQ(es->run->calls[0].prg.grid[0], 4);

  float *da = malloc((size_t)N * sizeof(float));
  float *db = malloc((size_t)N * sizeof(float));
  float *dout = calloc((size_t)N, sizeof(float));
  void **slot_data = calloc((size_t)ps->template->n_buf_slots, sizeof(void *));
  ASSERT_TRUE(da != NULL);
  ASSERT_TRUE(db != NULL);
  ASSERT_TRUE(dout != NULL);
  ASSERT_TRUE(slot_data != NULL);

  for (int i = 0; i < N; i++) {
    da[i] = (float)i;
    db[i] = (float)(N - i);
  }
  for (int i = 0; i < ps->template->n_buf_slots; i++) {
    if (ps->template->buf_slots[i].buf_uop == a)
      slot_data[i] = da;
    else if (ps->template->buf_slots[i].buf_uop == b)
      slot_data[i] = db;
    else if (ps->template->buf_slots[i].buf_uop == out)
      slot_data[i] = dout;
  }

  ASSERT_INT_EQ(poly_run_compiled_schedule(es, slot_data, ps->template->n_buf_slots, NULL, 0), 0);
  ASSERT_FLOAT_EQ(dout[0], (float)N, 1e-6);
  ASSERT_FLOAT_EQ(dout[1], (float)N, 1e-6);
  ASSERT_FLOAT_EQ(dout[12345], (float)N, 1e-6);
  ASSERT_FLOAT_EQ(dout[N - 1], (float)N, 1e-6);

  free(slot_data);
  free(dout);
  free(db);
  free(da);
  poly_compiled_schedule_free(es);
  poly_schedule_free(ps);
  poly_ctx_destroy(ctx);
  schedule_restore_env(&cpu_count_env);
  schedule_restore_env(&threads_env);
  PASS();
}

TEST(schedule_runtime, cpu_threaded_reuses_workers_with_current_slots) {
  ScheduleEnvSave threads_env = schedule_save_env("THREADS");
  ScheduleEnvSave cpu_count_env = schedule_save_env("CPU_COUNT");
  setenv("THREADS", "1", 1);
  setenv("CPU_COUNT", "4", 1);

  PolyCtx *ctx = poly_ctx_new();
  const int N = 1000000;
  PolyUOp *a = poly_buffer_f32(ctx, N);
  PolyUOp *b = poly_buffer_f32(ctx, N);
  PolyUOp *c = poly_alu2(ctx, POLY_OP_ADD, a, b);
  PolyUOp *out = poly_buffer_f32(ctx, N);
  PolyUOp *st = poly_store_val(ctx, out, c);
  PolyUOp *sink = poly_sink1(ctx, st);

  PolySchedule *ps = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_TRUE(ps != NULL);
  PolyCompiledSchedule *es = poly_lower_schedule(ctx, ps, POLY_DEVICE_CPU);
  ASSERT_TRUE(es != NULL);
  ASSERT_TRUE(es->run != NULL);
  ASSERT_INT_EQ(ps->template->n_calls, 1);
  ASSERT_INT_EQ(es->run->calls[0].prg.grid[0], 4);

  float *a1 = malloc((size_t)N * sizeof(float));
  float *b1 = malloc((size_t)N * sizeof(float));
  float *o1 = calloc((size_t)N, sizeof(float));
  float *a2 = malloc((size_t)N * sizeof(float));
  float *b2 = malloc((size_t)N * sizeof(float));
  float *o2 = calloc((size_t)N, sizeof(float));
  void **slots1 = calloc((size_t)ps->template->n_buf_slots, sizeof(void *));
  void **slots2 = calloc((size_t)ps->template->n_buf_slots, sizeof(void *));
  ASSERT_TRUE(a1 != NULL);
  ASSERT_TRUE(b1 != NULL);
  ASSERT_TRUE(o1 != NULL);
  ASSERT_TRUE(a2 != NULL);
  ASSERT_TRUE(b2 != NULL);
  ASSERT_TRUE(o2 != NULL);
  ASSERT_TRUE(slots1 != NULL);
  ASSERT_TRUE(slots2 != NULL);

  for (int i = 0; i < N; i++) {
    a1[i] = (float)i;
    b1[i] = 1.0f;
    a2[i] = (float)(2 * i);
    b2[i] = 3.0f;
  }
  for (int i = 0; i < ps->template->n_buf_slots; i++) {
    if (ps->template->buf_slots[i].buf_uop == a) {
      slots1[i] = a1;
      slots2[i] = a2;
    } else if (ps->template->buf_slots[i].buf_uop == b) {
      slots1[i] = b1;
      slots2[i] = b2;
    } else if (ps->template->buf_slots[i].buf_uop == out) {
      slots1[i] = o1;
      slots2[i] = o2;
    }
  }

  ASSERT_INT_EQ(poly_run_compiled_schedule(es, slots1, ps->template->n_buf_slots, NULL, 0), 0);
  ASSERT_INT_EQ(poly_run_compiled_schedule(es, slots2, ps->template->n_buf_slots, NULL, 0), 0);

  ASSERT_FLOAT_EQ(o1[0], 1.0f, 1e-6);
  ASSERT_FLOAT_EQ(o1[12345], 12346.0f, 1e-6);
  ASSERT_FLOAT_EQ(o2[0], 3.0f, 1e-6);
  ASSERT_FLOAT_EQ(o2[12345], 24693.0f, 1e-6);
  ASSERT_FLOAT_EQ(o2[N - 1], (float)(2 * (N - 1) + 3), 1e-6);

  free(slots2);
  free(slots1);
  free(o2);
  free(b2);
  free(a2);
  free(o1);
  free(b1);
  free(a1);
  poly_compiled_schedule_free(es);
  poly_schedule_free(ps);
  poly_ctx_destroy(ctx);
  schedule_restore_env(&cpu_count_env);
  schedule_restore_env(&threads_env);
  PASS();
}

TEST(schedule_runtime, cpu_threads_env_zero_keeps_single_core) {
  ScheduleEnvSave threads_env = schedule_save_env("THREADS");
  ScheduleEnvSave cpu_count_env = schedule_save_env("CPU_COUNT");
  setenv("THREADS", "0", 1);
  setenv("CPU_COUNT", "4", 1);

  PolyCtx *ctx = poly_ctx_new();
  const int N = 1000000;
  PolyUOp *a = poly_buffer_f32(ctx, N);
  PolyUOp *b = poly_buffer_f32(ctx, N);
  PolyUOp *c = poly_alu2(ctx, POLY_OP_ADD, a, b);
  PolyUOp *out = poly_buffer_f32(ctx, N);
  PolyUOp *st = poly_store_val(ctx, out, c);
  PolyUOp *sink = poly_sink1(ctx, st);

  PolySchedule *ps = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_TRUE(ps != NULL);
  PolyCompiledSchedule *es = poly_lower_schedule(ctx, ps, POLY_DEVICE_CPU);
  ASSERT_TRUE(es != NULL);
  ASSERT_TRUE(es->run != NULL);
  ASSERT_INT_EQ(ps->template->n_calls, 1);
  ASSERT_INT_EQ(es->run->calls[0].prg.grid[0], 1);

  poly_compiled_schedule_free(es);
  poly_schedule_free(ps);
  poly_ctx_destroy(ctx);
  schedule_restore_env(&cpu_count_env);
  schedule_restore_env(&threads_env);
  PASS();
}

TEST(schedule_runtime, lower_and_run_multikernel) {
  /* sum(a) -> reshape -> expand -> add(b) -> out
   * Multi-kernel: reduce produces intermediate. */
  int N = 8;
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *b = poly_buffer(ctx, POLY_FLOAT32, N);
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

  PolySchedule *ps = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_TRUE(ps != NULL);

  PolyCompiledSchedule *es = poly_lower_schedule(ctx, ps, POLY_DEVICE_CPU);
  ASSERT_TRUE(es != NULL);
  /* Plan has intermediate buffer slots (intermediates are per-run now) */
  int n_inter = 0;
  for (int i = 0; i < ps->template->n_buf_slots; i++)
    if (ps->template->buf_slots[i].is_intermediate) n_inter++;
  ASSERT_TRUE(n_inter > 0);

  /* a = [1..8], sum = 36, b = [10..17], out = 36 + b */
  float da[8], db[8], dout[8];
  for (int i = 0; i < N; i++) {
    da[i] = (float)(i + 1);
    db[i] = (float)(i + 10);
  }
  memset(dout, 0, sizeof(dout));

  void *slot_data[16] = {0};
  for (int i = 0; i < ps->template->n_buf_slots; i++) {
    if (ps->template->buf_slots[i].buf_uop == a)
      slot_data[i] = da;
    else if (ps->template->buf_slots[i].buf_uop == b)
      slot_data[i] = db;
    else if (ps->template->buf_slots[i].buf_uop == out)
      slot_data[i] = dout;
  }

  int ret = poly_run_compiled_schedule(es, slot_data, ps->template->n_buf_slots, NULL, 0);
  ASSERT_INT_EQ(ret, 0);

  /* sum(1..8) = 36, out[i] = 36 + (i + 10) */
  for (int i = 0; i < N; i++)
    ASSERT_FLOAT_EQ(dout[i], 36.0f + (float)(i + 10), 1e-5);

  poly_compiled_schedule_free(es);
  poly_schedule_free(ps);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(schedule_runtime, lower_matches_realize_uops) {
  /* Same graph through direct realize and explicit prepare+lower+run.
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

  /* Direct realize path */
  float dout_direct[4] = {0};
  PolyTestBufferView bindings[] = {
      POLY_TEST_HOST_VIEW(a, da), POLY_TEST_HOST_VIEW(b, db),
      POLY_TEST_HOST_VIEW(out, dout_direct)};
  ASSERT_INT_EQ(poly_test_realize_buffer_views(ctx, sink, bindings, 3), 0);

  /* New path */
  float dout_new[4] = {0};
  PolySchedule *ps = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_TRUE(ps != NULL);
  PolyCompiledSchedule *es = poly_lower_schedule(ctx, ps, POLY_DEVICE_CPU);
  ASSERT_TRUE(es != NULL);

  void *slot_data[16] = {0};
  for (int i = 0; i < ps->template->n_buf_slots; i++) {
    if (ps->template->buf_slots[i].buf_uop == a)
      slot_data[i] = da;
    else if (ps->template->buf_slots[i].buf_uop == b)
      slot_data[i] = db;
    else if (ps->template->buf_slots[i].buf_uop == out)
      slot_data[i] = dout_new;
  }
  ASSERT_INT_EQ(poly_run_compiled_schedule(es, slot_data, ps->template->n_buf_slots, NULL, 0), 0);

  /* Must match exactly */
  for (int i = 0; i < 4; i++)
    ASSERT_FLOAT_EQ(dout_new[i], dout_direct[i], 0.0);

  poly_compiled_schedule_free(es);
  poly_schedule_free(ps);
  poly_ctx_destroy(ctx);
  PASS();
}

/* Phase 4: Interpreter backend */

TEST(schedule_runtime, interp_vecadd) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer_f32(ctx, 4);
  PolyUOp *b = poly_buffer_f32(ctx, 4);
  PolyUOp *c = poly_alu2(ctx, POLY_OP_ADD, a, b);
  PolyUOp *out = poly_buffer_f32(ctx, 4);
  PolyUOp *st = poly_store_val(ctx, out, c);
  PolyUOp *sink = poly_sink1(ctx, st);

  PolySchedule *ps = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_TRUE(ps != NULL);

  PolyCompiledSchedule *es = poly_lower_schedule(ctx, ps, POLY_DEVICE_INTERP);
  ASSERT_TRUE(es != NULL);
  ASSERT_EQ(es->device, POLY_DEVICE_INTERP);

  float da[] = {1, 2, 3, 4};
  float db[] = {10, 20, 30, 40};
  float dout[4] = {0};

  void *slot_data[3];
  for (int i = 0; i < ps->template->n_buf_slots; i++) {
    if (ps->template->buf_slots[i].buf_uop == a)
      slot_data[i] = da;
    else if (ps->template->buf_slots[i].buf_uop == b)
      slot_data[i] = db;
    else if (ps->template->buf_slots[i].buf_uop == out)
      slot_data[i] = dout;
    else
      slot_data[i] = NULL;
  }

  int ret = poly_run_compiled_schedule(es, slot_data, ps->template->n_buf_slots, NULL, 0);
  ASSERT_INT_EQ(ret, 0);

  ASSERT_FLOAT_EQ(dout[0], 11.0f, 1e-6);
  ASSERT_FLOAT_EQ(dout[1], 22.0f, 1e-6);
  ASSERT_FLOAT_EQ(dout[2], 33.0f, 1e-6);
  ASSERT_FLOAT_EQ(dout[3], 44.0f, 1e-6);

  poly_compiled_schedule_free(es);
  poly_schedule_free(ps);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(schedule_runtime, interp_reduce) {
  /* sum(a) -> reshape -> expand -> add(b) -> out */
  int N = 8;
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *b = poly_buffer(ctx, POLY_FLOAT32, N);
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

  PolySchedule *ps = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_TRUE(ps != NULL);

  PolyCompiledSchedule *es = poly_lower_schedule(ctx, ps, POLY_DEVICE_INTERP);
  ASSERT_TRUE(es != NULL);

  float da[8], db[8], dout[8];
  for (int i = 0; i < N; i++) {
    da[i] = (float)(i + 1);
    db[i] = (float)(i + 10);
  }
  memset(dout, 0, sizeof(dout));

  void *slot_data[16] = {0};
  for (int i = 0; i < ps->template->n_buf_slots; i++) {
    if (ps->template->buf_slots[i].buf_uop == a)
      slot_data[i] = da;
    else if (ps->template->buf_slots[i].buf_uop == b)
      slot_data[i] = db;
    else if (ps->template->buf_slots[i].buf_uop == out)
      slot_data[i] = dout;
  }

  int ret = poly_run_compiled_schedule(es, slot_data, ps->template->n_buf_slots, NULL, 0);
  ASSERT_INT_EQ(ret, 0);

  for (int i = 0; i < N; i++)
    ASSERT_FLOAT_EQ(dout[i], 36.0f + (float)(i + 10), 1e-5);

  poly_compiled_schedule_free(es);
  poly_schedule_free(ps);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(schedule_runtime, interp_matches_cpu) {
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

  PolySchedule *ps = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_TRUE(ps != NULL);

  /* CPU path */
  float dout_cpu[4] = {0};
  PolyCompiledSchedule *cpu = poly_lower_schedule(ctx, ps, POLY_DEVICE_CPU);
  ASSERT_TRUE(cpu != NULL);

  void *slot_cpu[16] = {0};
  for (int i = 0; i < ps->template->n_buf_slots; i++) {
    if (ps->template->buf_slots[i].buf_uop == a)
      slot_cpu[i] = da;
    else if (ps->template->buf_slots[i].buf_uop == b)
      slot_cpu[i] = db;
    else if (ps->template->buf_slots[i].buf_uop == out)
      slot_cpu[i] = dout_cpu;
  }
  ASSERT_INT_EQ(poly_run_compiled_schedule(cpu, slot_cpu, ps->template->n_buf_slots, NULL, 0), 0);

  /* INTERP path */
  float dout_interp[4] = {0};
  PolyCompiledSchedule *interp = poly_lower_schedule(ctx, ps, POLY_DEVICE_INTERP);
  ASSERT_TRUE(interp != NULL);

  void *slot_interp[16] = {0};
  for (int i = 0; i < ps->template->n_buf_slots; i++) {
    if (ps->template->buf_slots[i].buf_uop == a)
      slot_interp[i] = da;
    else if (ps->template->buf_slots[i].buf_uop == b)
      slot_interp[i] = db;
    else if (ps->template->buf_slots[i].buf_uop == out)
      slot_interp[i] = dout_interp;
  }
  ASSERT_INT_EQ(poly_run_compiled_schedule(interp, slot_interp, ps->template->n_buf_slots, NULL, 0), 0);

  /* Bitwise identical (integer multiply, no FP rounding differences) */
  ASSERT_TRUE(memcmp(dout_cpu, dout_interp, sizeof(dout_cpu)) == 0);

  poly_compiled_schedule_free(cpu);
  poly_compiled_schedule_free(interp);
  poly_schedule_free(ps);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(schedule_runtime, interp_transcendental) {
  /* exp(log(a)) ~= a, tests the decomposed transcendental path */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer_f32(ctx, 4);
  PolyUOp *lg = poly_alu1(ctx, POLY_OP_LOG2, a);
  PolyUOp *ex = poly_alu1(ctx, POLY_OP_EXP2, lg);
  PolyUOp *out = poly_buffer_f32(ctx, 4);
  PolyUOp *st = poly_store_val(ctx, out, ex);
  PolyUOp *sink = poly_sink1(ctx, st);

  PolySchedule *ps = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_TRUE(ps != NULL);

  PolyCompiledSchedule *es = poly_lower_schedule(ctx, ps, POLY_DEVICE_INTERP);
  ASSERT_TRUE(es != NULL);

  float da[] = {1.0f, 2.0f, 4.0f, 8.0f};
  float dout[4] = {0};

  void *slot_data[16] = {0};
  for (int i = 0; i < ps->template->n_buf_slots; i++) {
    if (ps->template->buf_slots[i].buf_uop == a)
      slot_data[i] = da;
    else if (ps->template->buf_slots[i].buf_uop == out)
      slot_data[i] = dout;
  }

  ASSERT_INT_EQ(poly_run_compiled_schedule(es, slot_data, ps->template->n_buf_slots, NULL, 0), 0);

  /* exp2(log2(x)) ~= x, within polynomial approximation tolerance */
  for (int i = 0; i < 4; i++)
    ASSERT_FLOAT_EQ(dout[i], da[i], 1e-4);

  poly_compiled_schedule_free(es);
  poly_schedule_free(ps);
  poly_ctx_destroy(ctx);
  PASS();
}

/* CPU vs INTERP parity suite */

TEST(schedule_runtime, parity_chain) {
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

  int rc = cpu_interp_parity(ctx, sink, bufs, datas, 3, out, out_cpu, out_interp, 8, 0.0f);
  ASSERT_INT_EQ(rc, 0);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(schedule_runtime, parity_neg_sqrt) {
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

  int rc = cpu_interp_parity(ctx, sink, bufs, datas, 3, out, out_cpu, out_interp, 4, 1e-6f);
  ASSERT_INT_EQ(rc, 0);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(schedule_runtime, parity_reduce_sum) {
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

  int rc = cpu_interp_parity(ctx, sink, bufs, datas, 2, out, out_cpu, out_interp, 1, 1e-5f);
  ASSERT_INT_EQ(rc, 0);
  ASSERT_FLOAT_EQ(out_cpu[0], 36.0f, 1e-5);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(schedule_runtime, parity_where) {
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

  int rc = cpu_interp_parity(ctx, sink, bufs, datas, 3, out, out_cpu, out_interp, 4, 0.0f);
  ASSERT_INT_EQ(rc, 0);
  ASSERT_FLOAT_EQ(out_cpu[0], 10.0f, 1e-6);
  ASSERT_FLOAT_EQ(out_cpu[1], 2.0f, 1e-6);
  ASSERT_FLOAT_EQ(out_cpu[2], 30.0f, 1e-6);
  ASSERT_FLOAT_EQ(out_cpu[3], 4.0f, 1e-6);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(schedule_runtime, parity_exp2_log2) {
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

  int rc = cpu_interp_parity(ctx, sink, bufs, datas, 2, out, out_cpu, out_interp, 4, 1e-5f);
  ASSERT_INT_EQ(rc, 0);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(schedule_runtime, parity_multikernel_reduce_chain) {
  /* sum(a) -> expand -> add(b): multi-kernel with intermediate */
  int N = 8;
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *b = poly_buffer(ctx, POLY_FLOAT32, N);
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
  for (int i = 0; i < N; i++) {
    da[i] = (float)(i + 1);
    db[i] = (float)(i * 10);
  }
  float out_cpu[8], out_interp[8];
  PolyUOp *bufs[] = {a, b, out};
  void *datas[] = {da, db, NULL};

  int rc = cpu_interp_parity(ctx, sink, bufs, datas, 3, out, out_cpu, out_interp, N, 1e-5f);
  ASSERT_INT_EQ(rc, 0);

  poly_ctx_destroy(ctx);
  PASS();
}

/* Phase 5: Persistent workspace */

TEST(schedule_runtime, workspace_reuse) {
  /* Run same plan 100 times with different data. Persistent intermediates
   * are zeroed each call (reduce accumulators). No per-call allocations. */
  int N = 8;
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *b = poly_buffer(ctx, POLY_FLOAT32, N);
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

  PolySchedule *ps = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_TRUE(ps != NULL);

  PolyCompiledSchedule *plan = poly_lower_schedule(ctx, ps, POLY_DEVICE_CPU);
  ASSERT_TRUE(plan != NULL);

  /* Verify persistent intermediates were allocated */
  int n_inter = 0;
  int n_owning_inter = 0;
  int n_zero = 0;
  for (int i = 0; i < ps->template->n_buf_slots; i++) {
    if (!ps->template->buf_slots[i].is_intermediate) continue;
    n_inter++;
    if (!ps->template->buf_slots[i].has_memory_parent) n_owning_inter++;
    if (ps->template->buf_slots[i].needs_zero) n_zero++;
  }
  ASSERT_TRUE(n_inter > 0);
  ASSERT_TRUE(n_owning_inter > 0);
  ASSERT_TRUE(n_zero > 0);
  ASSERT_TRUE(plan->run != NULL);
  ASSERT_INT_EQ(plan->run->n_intermediates, n_owning_inter);

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
    for (int i = 0; i < ps->template->n_buf_slots; i++) {
      if (ps->template->buf_slots[i].buf_uop == a)
        slot_data[i] = da;
      else if (ps->template->buf_slots[i].buf_uop == b)
        slot_data[i] = db;
      else if (ps->template->buf_slots[i].buf_uop == out)
        slot_data[i] = dout;
    }

    int ret = poly_run_compiled_schedule(plan, slot_data, ps->template->n_buf_slots, NULL, 0);
    ASSERT_INT_EQ(ret, 0);

    /* sum(1..8 * mult) = 36*mult, out[i] = 36*mult + (i+10) */
    float expected_sum = 36.0f * mult;
    for (int i = 0; i < N; i++)
      ASSERT_FLOAT_EQ(dout[i], expected_sum + (float)(i + 10), 1e-3);
  }

  poly_compiled_schedule_free(plan);
  poly_schedule_free(ps);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(schedule_runtime, workspace_reduce_zeroed) {
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

  PolySchedule *ps = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_TRUE(ps != NULL);

  PolyCompiledSchedule *plan = poly_lower_schedule(ctx, ps, POLY_DEVICE_CPU);
  ASSERT_TRUE(plan != NULL);

  /* Run 1: sum([1,2,3,4]) = 10 */
  float da1[] = {1, 2, 3, 4};
  float dout1 = 0;
  void *slot1[8] = {0};
  for (int i = 0; i < ps->template->n_buf_slots; i++) {
    if (ps->template->buf_slots[i].buf_uop == a)
      slot1[i] = da1;
    else if (ps->template->buf_slots[i].buf_uop == out)
      slot1[i] = &dout1;
  }
  ASSERT_INT_EQ(poly_run_compiled_schedule(plan, slot1, ps->template->n_buf_slots, NULL, 0), 0);
  ASSERT_FLOAT_EQ(dout1, 10.0f, 1e-6);

  /* Run 2: sum([10,20,30,40]) = 100, NOT 110 (accumulated from run 1) */
  float da2[] = {10, 20, 30, 40};
  float dout2 = 0;
  void *slot2[8] = {0};
  for (int i = 0; i < ps->template->n_buf_slots; i++) {
    if (ps->template->buf_slots[i].buf_uop == a)
      slot2[i] = da2;
    else if (ps->template->buf_slots[i].buf_uop == out)
      slot2[i] = &dout2;
  }
  ASSERT_INT_EQ(poly_run_compiled_schedule(plan, slot2, ps->template->n_buf_slots, NULL, 0), 0);
  ASSERT_FLOAT_EQ(dout2, 100.0f, 1e-5);

  poly_compiled_schedule_free(plan);
  poly_schedule_free(ps);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(schedule_runtime, interp_gated_load_pad_shrink) {
  /* Pad+shrink+expand+reduce: verifies gated INDEX (from move_where_on_load)
   * is correctly handled by the interpreter LOAD. Without the fix, pad guards
   * are ignored and the result is 800 instead of 740. */
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *x_flat = poly_buffer(ctx, POLY_FLOAT32, 75);
  int64_t x_shape[] = {1, 3, 5, 5};
  PolyUOp *x = poly_reshape(ctx, x_flat, x_shape, 4);
  int64_t pad_pairs[][2] = {{0, 0}, {0, 0}, {1, 1}, {1, 1}};
  PolyUOp *xp = poly_pad(ctx, x, pad_pairs, 4);
  int64_t s1_pairs[][2] = {{0, 1}, {0, 1}, {0, 5}, {0, 5}};
  int64_t s2_pairs[][2] = {{0, 1}, {0, 1}, {0, 5}, {1, 6}};
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
  for (int i = 0; i < 75; i++)
    x_d[i] = (float)(i + 1);
  float out_cpu[1], out_interp[1];
  PolyUOp *bufs[] = {x_flat, out};
  void *datas[] = {x_d, NULL};

  int rc = cpu_interp_parity(ctx, sink, bufs, datas, 2, out, out_cpu, out_interp, 1, 1e-5f);
  if (rc != 0)
    fprintf(
        stderr, "  gated_load: cpu=%.1f interp=%.1f rc=%d\n", (double)out_cpu[0],
        (double)out_interp[0], rc
    );
  ASSERT_INT_EQ(rc, 0);
  ASSERT_FLOAT_EQ(out_cpu[0], 740.0f, 1e-5);

  poly_ctx_destroy(ctx);
  PASS();
}
