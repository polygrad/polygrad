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
#include "../src/utils.h"

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
  for (int i = 0; i < ps->n_buf_slots; i++) {
    for (int j = 0; j < n_bufs; j++) {
      if (ps->buf_slots[i].buf_uop == bufs[j])
        slot_cpu[i] = (bufs[j] == out_buf) ? out_cpu : datas[j];
    }
  }
  int rc = poly_run_compiled_schedule(cpu, slot_cpu, ps->n_buf_slots, NULL, 0);
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
  for (int i = 0; i < ps->n_buf_slots; i++) {
    for (int j = 0; j < n_bufs; j++) {
      if (ps->buf_slots[i].buf_uop == bufs[j])
        slot_interp[i] = (bufs[j] == out_buf) ? out_interp : datas[j];
    }
  }
  rc = poly_run_compiled_schedule(interp, slot_interp, ps->n_buf_slots, NULL, 0);
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

static PolyUOp *apply_webgpu_tri_stage_root(PolyCtx *ctx, PolyUOp *root, bool with_add_loads, bool with_post_index_symbolic) {
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

TEST(exec_plan, prepare_vecadd) {
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

  poly_schedule_free(ps);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(exec_plan, create_schedule_matches_complete_schedule_vecadd) {
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
  ASSERT_INT_EQ(from_sink->n_items, 1);
  ASSERT_INT_EQ(from_kernel_graph->n_items, 1);
  ASSERT_TRUE(poly_structural_eq(from_sink->items[0].root, from_kernel_graph->items[0].root));

  int n_sink_lin = 0, n_graph_lin = 0;
  PolyUOp **sink_lin = poly_linearize(ctx, from_sink->items[0].root, &n_sink_lin);
  PolyUOp **graph_lin = poly_linearize(ctx, from_kernel_graph->items[0].root, &n_graph_lin);
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

TEST(exec_plan, lower_sink_to_linear_matches_create_schedule_vecadd) {
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
  ASSERT_INT_EQ(linear->n_src, schedule->n_items);

  for (int step = 0; step < schedule->n_items; step++) {
    int idx = schedule->exec_order ? schedule->exec_order[step] : step;
    ASSERT_TRUE(poly_structural_eq(linear->src[step], schedule->items[idx].root));
  }

  poly_schedule_free(schedule);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(exec_plan, full_rewrite_vecadd_upcast_preserves_end_ranges_like_tinygrad) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_TRUE(ctx != NULL);

  PolyUOp *a = poly_buffer_f32(ctx, 16);
  PolyUOp *b = poly_buffer_f32(ctx, 16);
  PolyUOp *c = poly_alu2(ctx, POLY_OP_ADD, a, b);
  PolyUOp *out = poly_buffer_f32(ctx, 16);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, c));

  PolySchedule *sched = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_TRUE(sched != NULL);
  ASSERT_INT_EQ(sched->n_items, 1);

  PolyRewriteOpts opts = {
      .optimize = true,
      .devectorize = 1,
      .beam_width = 0,
      .caps = poly_c_renderer_caps(),
      .device = POLY_DEVICE_CPU,
      .opt_policy = POLY_OPT_HEURISTIC,
  };

  PolyUOp *rewritten = poly_full_rewrite_to_sink_ex(ctx, sched->items[0].root, opts);
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

TEST(exec_plan, linearize_vecadd_upcast_has_no_singleton_group_like_tinygrad) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_TRUE(ctx != NULL);

  PolyUOp *a = poly_buffer_f32(ctx, 16);
  PolyUOp *b = poly_buffer_f32(ctx, 16);
  PolyUOp *c = poly_alu2(ctx, POLY_OP_ADD, a, b);
  PolyUOp *out = poly_buffer_f32(ctx, 16);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, c));

  PolySchedule *sched = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_TRUE(sched != NULL);
  ASSERT_INT_EQ(sched->n_items, 1);

  int n_lin = 0;
  PolyUOp **lin = poly_linearize(ctx, sched->items[0].root, &n_lin);
  ASSERT_TRUE(lin != NULL);
  ASSERT_INT_EQ(n_lin, 31);

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

TEST(exec_plan, prepare_multikernel) {
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

  poly_schedule_free(ps);
  poly_ctx_destroy(ctx);
  PASS();
}

/* Buffer slot metadata */

TEST(exec_plan, prepare_buf_slot_metadata) {
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
  for (int i = 0; i < ps->n_buf_slots; i++) {
    if (!ps->buf_slots[i].is_intermediate) {
      ASSERT_TRUE(ps->buf_slots[i].numel > 0);
      ASSERT_TRUE(ps->buf_slots[i].nbytes > 0);
    }
  }

  poly_schedule_free(ps);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(exec_plan, webgpu_triu_param_order_matches_runtime_slots) {
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
  ASSERT_TRUE(sched->n_items >= 1);

  PolyWebGpuStepPlan *plan = poly_render_step_webgpu_plan(ctx, sink);
  ASSERT_TRUE(plan != NULL);

  bool checked_kernel = false;
  for (int k = 0; k < sched->n_items; k++) {
    PolyExecItem *item = &sched->items[k];
    int n_lin = 0;
    PolyUOp **lin = poly_linearize_webgpu(ctx, item->root, &n_lin);
    ASSERT_TRUE(lin != NULL);

    int seen_params = 0;
    for (int i = 0; i < n_lin; i++) {
      if (lin[i]->op != POLY_OP_PARAM) continue;
      ASSERT_TRUE(seen_params < item->n_buf_slots);
      ASSERT_INT_EQ((int)lin[i]->arg.i, seen_params);
      ASSERT_INT_EQ(poly_webgpu_stepplan_kernel_param_buf_index(plan, k, seen_params),
                    item->buf_slot_indices[(int)lin[i]->arg.i]);
      seen_params++;
    }
    ASSERT_INT_EQ(poly_webgpu_stepplan_kernel_n_params(plan, k), item->n_buf_slots);
    ASSERT_INT_EQ(seen_params, item->n_buf_slots);
    free(lin);
    checked_kernel = true;
  }

  ASSERT_TRUE(checked_kernel);
  poly_webgpu_stepplan_destroy(plan);
  poly_schedule_free(sched);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(exec_plan, webgpu_unified_triu_renders_single_kernel_no_helper_params) {
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
  ASSERT_INT_EQ(sched->n_items, 1);
  ASSERT_INT_EQ(sched->items[0].n_buf_slots, 2);

  int n_lin = 0;
  PolyUOp **lin = poly_linearize_webgpu(ctx, sched->items[0].root, &n_lin);
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

TEST(exec_plan, webgpu_unified_triu_has_no_vector_gated_loads) {
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
  ASSERT_INT_EQ(sched->n_items, 1);

  int n_lin = 0;
  PolyUOp **lin = poly_linearize_webgpu(ctx, sched->items[0].root, &n_lin);
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

TEST(exec_plan, webgpu_triu_move_where_keeps_residual_where_like_tinygrad) {
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
    ASSERT_INT_EQ(sched->n_items, 1);

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

    PolyUOp *u = sched->items[0].root;
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

TEST(exec_plan, webgpu_triu_heuristic_adds_local_split_like_tinygrad) {
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
  ASSERT_INT_EQ(sched->n_items, 1);

  PolyRendererCaps caps = {
      .has_mulacc = false,
      .has_threefry = false,
      .has_local = true,
      .max_vec_width = 1,
  };
  PolyUOp *heur = poly_apply_opts_heuristic_ex(ctx, sched->items[0].root, caps);

  ASSERT_INT_EQ(count_root_ranges_of_type(ctx, heur, POLY_AXIS_LOOP), 2);
  ASSERT_INT_EQ(count_root_ranges_of_type(ctx, heur, POLY_AXIS_LOCAL), 2);
  ASSERT_INT_EQ(count_root_ranges_of_type(ctx, heur, POLY_AXIS_UPCAST), 0);

  poly_schedule_free(sched);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(exec_plan, webgpu_triu_gpudims_replaces_all_ranges_like_tinygrad) {
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
  ASSERT_INT_EQ(sched->n_items, 1);

  PolyRendererCaps caps = {
      .has_mulacc = false,
      .has_threefry = false,
      .has_local = true,
      .max_vec_width = 1,
  };

  PolyUOp *u = poly_apply_opts_heuristic_ex(ctx, sched->items[0].root, caps);
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

TEST(exec_plan, webgpu_triu_small_expander_drops_residual_where_like_tinygrad) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_TRUE(ctx != NULL);

  PolySchedule *sched = build_tri_schedule(ctx, 3, false);
  ASSERT_TRUE(sched != NULL);
  ASSERT_INT_EQ(sched->n_items, 1);

  PolyRendererCaps caps = {
      .has_mulacc = false,
      .has_threefry = false,
      .has_local = true,
      .max_vec_width = 1,
  };

  PolyUOp *u = poly_apply_opts_heuristic_ex(ctx, sched->items[0].root, caps);
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

TEST(exec_plan, webgpu_tril_small_expander_drops_residual_where_like_tinygrad) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_TRUE(ctx != NULL);

  PolySchedule *sched = build_tri_schedule(ctx, 3, true);
  ASSERT_TRUE(sched != NULL);
  ASSERT_INT_EQ(sched->n_items, 1);

  PolyRendererCaps caps = {
      .has_mulacc = false,
      .has_threefry = false,
      .has_local = true,
      .max_vec_width = 1,
  };

  PolyUOp *u = poly_apply_opts_heuristic_ex(ctx, sched->items[0].root, caps);
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

TEST(exec_plan, webgpu_triu_small_devector_matches_tinygrad_load_count) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_TRUE(ctx != NULL);

  PolySchedule *sched = build_tri_schedule(ctx, 3, false);
  ASSERT_TRUE(sched != NULL);
  ASSERT_INT_EQ(sched->n_items, 1);

  PolyRendererCaps caps = {
      .has_mulacc = false,
      .has_threefry = false,
      .has_local = true,
      .max_vec_width = 1,
  };

  PolyUOp *u = poly_apply_opts_heuristic_ex(ctx, sched->items[0].root, caps);
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

TEST(exec_plan, webgpu_tril_small_devector_matches_tinygrad_load_count) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_TRUE(ctx != NULL);

  PolySchedule *sched = build_tri_schedule(ctx, 3, true);
  ASSERT_TRUE(sched != NULL);
  ASSERT_INT_EQ(sched->n_items, 1);

  PolyRendererCaps caps = {
      .has_mulacc = false,
      .has_threefry = false,
      .has_local = true,
      .max_vec_width = 1,
  };

  PolyUOp *u = poly_apply_opts_heuristic_ex(ctx, sched->items[0].root, caps);
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

TEST(exec_plan, webgpu_triu_add_loads_keeps_residual_where_like_tinygrad) {
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
  ASSERT_INT_EQ(sched->n_items, 1);

  PolyRendererCaps caps = {
      .has_mulacc = false,
      .has_threefry = false,
      .has_local = true,
      .max_vec_width = 1,
  };

  PolyUOp *u = poly_apply_opts_heuristic_ex(ctx, sched->items[0].root, caps);
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

TEST(exec_plan, webgpu_triu_post_index_symbolic_keeps_residual_where_like_tinygrad) {
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
  ASSERT_INT_EQ(sched->n_items, 1);

  PolyRendererCaps caps = {
      .has_mulacc = false,
      .has_threefry = false,
      .has_local = true,
      .max_vec_width = 1,
  };

  PolyUOp *u = poly_apply_opts_heuristic_ex(ctx, sched->items[0].root, caps);
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

  ASSERT_INT_EQ(count_root_ops(ctx, u, POLY_OP_WHERE), 1);
  ASSERT_INT_EQ(count_root_ops(ctx, u, POLY_OP_LOAD), 1);
  ASSERT_INT_EQ(count_root_gated_loads(ctx, u), 0);

  poly_schedule_free(sched);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(exec_plan, webgpu_triu_final_rewrite_keeps_scalar_where_load_like_tinygrad) {
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
  ASSERT_INT_EQ(sched->n_items, 1);

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

  PolyUOp *rewritten = poly_full_rewrite_to_sink_ex(ctx, sched->items[0].root, opts);
  ASSERT_TRUE(rewritten != NULL);
  ASSERT_INT_EQ(count_root_ops(ctx, rewritten, POLY_OP_WHERE), 1);
  ASSERT_INT_EQ(count_root_ops(ctx, rewritten, POLY_OP_LOAD), 1);
  ASSERT_INT_EQ(count_root_gated_loads(ctx, rewritten), 0);

  poly_schedule_free(sched);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(exec_plan, webgpu_unified_tril_has_no_vector_gated_loads) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_TRUE(ctx != NULL);

  PolySchedule *sched = build_tri_schedule(ctx, 3, true);
  ASSERT_TRUE(sched != NULL);
  ASSERT_INT_EQ(sched->n_items, 1);

  int n_lin = 0;
  PolyUOp **lin = poly_linearize_webgpu(ctx, sched->items[0].root, &n_lin);
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

TEST(exec_plan, webgpu_tril_add_loads_keeps_residual_where_like_tinygrad) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_TRUE(ctx != NULL);

  PolySchedule *sched = build_tri_schedule(ctx, 9, true);
  ASSERT_TRUE(sched != NULL);
  ASSERT_INT_EQ(sched->n_items, 1);

  PolyUOp *u = apply_webgpu_tri_stage_root(ctx, sched->items[0].root, true, false);

  ASSERT_INT_EQ(count_root_ops(ctx, u, POLY_OP_WHERE), 1);
  ASSERT_INT_EQ(count_root_ops(ctx, u, POLY_OP_LOAD), 1);
  ASSERT_INT_EQ(count_root_gated_loads(ctx, u), 0);

  poly_schedule_free(sched);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(exec_plan, webgpu_tril_post_index_symbolic_keeps_residual_where_like_tinygrad) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_TRUE(ctx != NULL);

  PolySchedule *sched = build_tri_schedule(ctx, 9, true);
  ASSERT_TRUE(sched != NULL);
  ASSERT_INT_EQ(sched->n_items, 1);

  PolyUOp *u = apply_webgpu_tri_stage_root(ctx, sched->items[0].root, true, true);

  ASSERT_INT_EQ(count_root_ops(ctx, u, POLY_OP_WHERE), 1);
  ASSERT_INT_EQ(count_root_ops(ctx, u, POLY_OP_LOAD), 1);
  ASSERT_INT_EQ(count_root_gated_loads(ctx, u), 0);

  poly_schedule_free(sched);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(exec_plan, webgpu_tril_final_rewrite_keeps_scalar_where_load_like_tinygrad) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_TRUE(ctx != NULL);

  PolySchedule *sched = build_tri_schedule(ctx, 9, true);
  ASSERT_TRUE(sched != NULL);
  ASSERT_INT_EQ(sched->n_items, 1);

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

  PolyUOp *rewritten = poly_full_rewrite_to_sink_ex(ctx, sched->items[0].root, opts);
  ASSERT_TRUE(rewritten != NULL);
  ASSERT_INT_EQ(count_root_ops(ctx, rewritten, POLY_OP_WHERE), 1);
  ASSERT_INT_EQ(count_root_ops(ctx, rewritten, POLY_OP_LOAD), 1);
  ASSERT_INT_EQ(count_root_gated_loads(ctx, rewritten), 0);

  poly_schedule_free(sched);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(exec_plan, kernel_graph_triu_keeps_load_in_codegen_stage_only) {
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

TEST(exec_plan, kernel_graph_triu_read_indices_stay_scalar_until_codegen) {
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

TEST(exec_plan, prepare_null_safety) {
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

TEST(exec_plan, prepare_graph_hash) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer_f32(ctx, 4);
  PolyUOp *b = poly_buffer_f32(ctx, 4);
  PolyUOp *c = poly_alu2(ctx, POLY_OP_ADD, a, b);
  PolyUOp *out = poly_buffer_f32(ctx, 4);
  PolyUOp *st = poly_store_val(ctx, out, c);
  PolyUOp *sink = poly_sink1(ctx, st);

  PolySchedule *ps = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_TRUE(ps != NULL);
  ASSERT_TRUE(ps->graph_hash != 0);

  poly_schedule_free(ps);
  poly_ctx_destroy(ctx);
  PASS();
}

/* Phase 3: Lower + Run */

TEST(exec_plan, lower_and_run_vecadd) {
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
  ASSERT_INT_EQ(es->n_runners, ps->n_items);
  ASSERT_EQ(es->device, POLY_DEVICE_CPU);

  /* Prepare data */
  float da[] = {1, 2, 3, 4};
  float db[] = {10, 20, 30, 40};
  float dout[4] = {0};

  /* Build slot_data array indexed by buf_slot */
  void *slot_data[3];
  for (int i = 0; i < ps->n_buf_slots; i++) {
    if (ps->buf_slots[i].buf_uop == a)
      slot_data[i] = da;
    else if (ps->buf_slots[i].buf_uop == b)
      slot_data[i] = db;
    else if (ps->buf_slots[i].buf_uop == out)
      slot_data[i] = dout;
    else
      slot_data[i] = NULL;
  }

  int ret = poly_run_compiled_schedule(es, slot_data, ps->n_buf_slots, NULL, 0);
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

TEST(exec_plan, lower_and_run_multikernel) {
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
  for (int i = 0; i < ps->n_buf_slots; i++)
    if (ps->buf_slots[i].is_intermediate) n_inter++;
  ASSERT_TRUE(n_inter > 0);

  /* a = [1..8], sum = 36, b = [10..17], out = 36 + b */
  float da[8], db[8], dout[8];
  for (int i = 0; i < N; i++) {
    da[i] = (float)(i + 1);
    db[i] = (float)(i + 10);
  }
  memset(dout, 0, sizeof(dout));

  void *slot_data[16] = {0};
  for (int i = 0; i < ps->n_buf_slots; i++) {
    if (ps->buf_slots[i].buf_uop == a)
      slot_data[i] = da;
    else if (ps->buf_slots[i].buf_uop == b)
      slot_data[i] = db;
    else if (ps->buf_slots[i].buf_uop == out)
      slot_data[i] = dout;
  }

  int ret = poly_run_compiled_schedule(es, slot_data, ps->n_buf_slots, NULL, 0);
  ASSERT_INT_EQ(ret, 0);

  /* sum(1..8) = 36, out[i] = 36 + (i + 10) */
  for (int i = 0; i < N; i++)
    ASSERT_FLOAT_EQ(dout[i], 36.0f + (float)(i + 10), 1e-5);

  poly_compiled_schedule_free(es);
  poly_schedule_free(ps);
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
  PolySchedule *ps = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_TRUE(ps != NULL);
  PolyCompiledSchedule *es = poly_lower_schedule(ctx, ps, POLY_DEVICE_CPU);
  ASSERT_TRUE(es != NULL);

  void *slot_data[16] = {0};
  for (int i = 0; i < ps->n_buf_slots; i++) {
    if (ps->buf_slots[i].buf_uop == a)
      slot_data[i] = da;
    else if (ps->buf_slots[i].buf_uop == b)
      slot_data[i] = db;
    else if (ps->buf_slots[i].buf_uop == out)
      slot_data[i] = dout_new;
  }
  ASSERT_INT_EQ(poly_run_compiled_schedule(es, slot_data, ps->n_buf_slots, NULL, 0), 0);

  /* Must match exactly */
  for (int i = 0; i < 4; i++)
    ASSERT_FLOAT_EQ(dout_new[i], dout_old[i], 0.0);

  poly_compiled_schedule_free(es);
  poly_schedule_free(ps);
  poly_step_destroy(old_step);
  poly_ctx_destroy(ctx);
  PASS();
}

/* Phase 4: Interpreter backend */

TEST(exec_plan, interp_vecadd) {
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
  for (int i = 0; i < ps->n_buf_slots; i++) {
    if (ps->buf_slots[i].buf_uop == a)
      slot_data[i] = da;
    else if (ps->buf_slots[i].buf_uop == b)
      slot_data[i] = db;
    else if (ps->buf_slots[i].buf_uop == out)
      slot_data[i] = dout;
    else
      slot_data[i] = NULL;
  }

  int ret = poly_run_compiled_schedule(es, slot_data, ps->n_buf_slots, NULL, 0);
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

TEST(exec_plan, interp_reduce) {
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
  for (int i = 0; i < ps->n_buf_slots; i++) {
    if (ps->buf_slots[i].buf_uop == a)
      slot_data[i] = da;
    else if (ps->buf_slots[i].buf_uop == b)
      slot_data[i] = db;
    else if (ps->buf_slots[i].buf_uop == out)
      slot_data[i] = dout;
  }

  int ret = poly_run_compiled_schedule(es, slot_data, ps->n_buf_slots, NULL, 0);
  ASSERT_INT_EQ(ret, 0);

  for (int i = 0; i < N; i++)
    ASSERT_FLOAT_EQ(dout[i], 36.0f + (float)(i + 10), 1e-5);

  poly_compiled_schedule_free(es);
  poly_schedule_free(ps);
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

  PolySchedule *ps = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_TRUE(ps != NULL);

  /* CPU path */
  float dout_cpu[4] = {0};
  PolyCompiledSchedule *cpu = poly_lower_schedule(ctx, ps, POLY_DEVICE_CPU);
  ASSERT_TRUE(cpu != NULL);

  void *slot_cpu[16] = {0};
  for (int i = 0; i < ps->n_buf_slots; i++) {
    if (ps->buf_slots[i].buf_uop == a)
      slot_cpu[i] = da;
    else if (ps->buf_slots[i].buf_uop == b)
      slot_cpu[i] = db;
    else if (ps->buf_slots[i].buf_uop == out)
      slot_cpu[i] = dout_cpu;
  }
  ASSERT_INT_EQ(poly_run_compiled_schedule(cpu, slot_cpu, ps->n_buf_slots, NULL, 0), 0);

  /* INTERP path */
  float dout_interp[4] = {0};
  PolyCompiledSchedule *interp = poly_lower_schedule(ctx, ps, POLY_DEVICE_INTERP);
  ASSERT_TRUE(interp != NULL);

  void *slot_interp[16] = {0};
  for (int i = 0; i < ps->n_buf_slots; i++) {
    if (ps->buf_slots[i].buf_uop == a)
      slot_interp[i] = da;
    else if (ps->buf_slots[i].buf_uop == b)
      slot_interp[i] = db;
    else if (ps->buf_slots[i].buf_uop == out)
      slot_interp[i] = dout_interp;
  }
  ASSERT_INT_EQ(poly_run_compiled_schedule(interp, slot_interp, ps->n_buf_slots, NULL, 0), 0);

  /* Bitwise identical (integer multiply, no FP rounding differences) */
  ASSERT_TRUE(memcmp(dout_cpu, dout_interp, sizeof(dout_cpu)) == 0);

  poly_compiled_schedule_free(cpu);
  poly_compiled_schedule_free(interp);
  poly_schedule_free(ps);
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

  PolySchedule *ps = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_TRUE(ps != NULL);

  PolyCompiledSchedule *es = poly_lower_schedule(ctx, ps, POLY_DEVICE_INTERP);
  ASSERT_TRUE(es != NULL);

  float da[] = {1.0f, 2.0f, 4.0f, 8.0f};
  float dout[4] = {0};

  void *slot_data[16] = {0};
  for (int i = 0; i < ps->n_buf_slots; i++) {
    if (ps->buf_slots[i].buf_uop == a)
      slot_data[i] = da;
    else if (ps->buf_slots[i].buf_uop == out)
      slot_data[i] = dout;
  }

  ASSERT_INT_EQ(poly_run_compiled_schedule(es, slot_data, ps->n_buf_slots, NULL, 0), 0);

  /* exp2(log2(x)) ~= x, within polynomial approximation tolerance */
  for (int i = 0; i < 4; i++)
    ASSERT_FLOAT_EQ(dout[i], da[i], 1e-4);

  poly_compiled_schedule_free(es);
  poly_schedule_free(ps);
  poly_ctx_destroy(ctx);
  PASS();
}

/* CPU vs INTERP parity suite */

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

  int rc = cpu_interp_parity(ctx, sink, bufs, datas, 3, out, out_cpu, out_interp, 8, 0.0f);
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

  int rc = cpu_interp_parity(ctx, sink, bufs, datas, 3, out, out_cpu, out_interp, 4, 1e-6f);
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

  int rc = cpu_interp_parity(ctx, sink, bufs, datas, 2, out, out_cpu, out_interp, 1, 1e-5f);
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

  int rc = cpu_interp_parity(ctx, sink, bufs, datas, 3, out, out_cpu, out_interp, 4, 0.0f);
  ASSERT_INT_EQ(rc, 0);
  ASSERT_FLOAT_EQ(out_cpu[0], 10.0f, 1e-6);
  ASSERT_FLOAT_EQ(out_cpu[1], 2.0f, 1e-6);
  ASSERT_FLOAT_EQ(out_cpu[2], 30.0f, 1e-6);
  ASSERT_FLOAT_EQ(out_cpu[3], 4.0f, 1e-6);

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

  int rc = cpu_interp_parity(ctx, sink, bufs, datas, 2, out, out_cpu, out_interp, 4, 1e-5f);
  ASSERT_INT_EQ(rc, 0);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(exec_plan, parity_multikernel_reduce_chain) {
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

TEST(exec_plan, workspace_reuse) {
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
      if (ps->buf_slots[i].buf_uop == a)
        slot_data[i] = da;
      else if (ps->buf_slots[i].buf_uop == b)
        slot_data[i] = db;
      else if (ps->buf_slots[i].buf_uop == out)
        slot_data[i] = dout;
    }

    int ret = poly_run_compiled_schedule(plan, slot_data, ps->n_buf_slots, NULL, 0);
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

  PolySchedule *ps = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_TRUE(ps != NULL);

  PolyCompiledSchedule *plan = poly_lower_schedule(ctx, ps, POLY_DEVICE_CPU);
  ASSERT_TRUE(plan != NULL);

  /* Run 1: sum([1,2,3,4]) = 10 */
  float da1[] = {1, 2, 3, 4};
  float dout1 = 0;
  void *slot1[8] = {0};
  for (int i = 0; i < ps->n_buf_slots; i++) {
    if (ps->buf_slots[i].buf_uop == a)
      slot1[i] = da1;
    else if (ps->buf_slots[i].buf_uop == out)
      slot1[i] = &dout1;
  }
  ASSERT_INT_EQ(poly_run_compiled_schedule(plan, slot1, ps->n_buf_slots, NULL, 0), 0);
  ASSERT_FLOAT_EQ(dout1, 10.0f, 1e-6);

  /* Run 2: sum([10,20,30,40]) = 100, NOT 110 (accumulated from run 1) */
  float da2[] = {10, 20, 30, 40};
  float dout2 = 0;
  void *slot2[8] = {0};
  for (int i = 0; i < ps->n_buf_slots; i++) {
    if (ps->buf_slots[i].buf_uop == a)
      slot2[i] = da2;
    else if (ps->buf_slots[i].buf_uop == out)
      slot2[i] = &dout2;
  }
  ASSERT_INT_EQ(poly_run_compiled_schedule(plan, slot2, ps->n_buf_slots, NULL, 0), 0);
  ASSERT_FLOAT_EQ(dout2, 100.0f, 1e-5);

  poly_compiled_schedule_free(plan);
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
