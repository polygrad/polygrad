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

#include <stdbool.h>

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
  ASSERT_INT_EQ(linear->n_src, schedule->n_items);

  for (int step = 0; step < schedule->n_items; step++) {
    int idx = schedule->exec_order ? schedule->exec_order[step] : step;
    PolyUOp *linear_item = linear->src[step];
    /* Cached LINEAR mirrors tinygrad's callified form: each entry is a CALL
     * carrying the reusable kernel root plus its parameter order. */
    if (linear_item && linear_item->op == POLY_OP_CALL && linear_item->n_src >= 1)
      linear_item = linear_item->src[0];
    ASSERT_TRUE(poly_structural_eq(linear_item, schedule->items[idx].root));
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
  for (int i = 0; sched && i < sched->n_buf_slots; i++)
    if (sched->buf_slots[i].buf_uop == buf) return true;
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

  ASSERT_INT_EQ(sched1->n_items, 1);
  ASSERT_INT_EQ(sched2->n_items, 1);
  ASSERT_PTR_EQ(sched1->items[0].root, sched2->items[0].root);
  ASSERT_TRUE(schedule_has_buf_slot(sched2, a2));
  ASSERT_TRUE(schedule_has_buf_slot(sched2, b2));
  ASSERT_TRUE(schedule_has_buf_slot(sched2, out2));
  ASSERT_FALSE(schedule_has_buf_slot(sched2, a1));
  ASSERT_FALSE(schedule_has_buf_slot(sched2, b1));
  ASSERT_FALSE(schedule_has_buf_slot(sched2, out1));

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

TEST(schedule_runtime, program_cache_reuses_runner_across_fresh_schedules) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  ASSERT_INT_EQ((int)poly_program_cache_len(ctx), 0);

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
  ASSERT_INT_EQ((int)poly_program_cache_len(ctx), 1);
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
  ASSERT_INT_EQ((int)poly_program_cache_len(ctx), 1);
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
  ASSERT_INT_EQ((int)poly_program_cache_len(ctx), 2);
  ASSERT_FLOAT_EQ(out3_data[0], 250.0f, 1e-5);
  ASSERT_FLOAT_EQ(out3_data[3], 640.0f, 1e-5);

  poly_ctx_destroy(ctx);
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
  PolyExecItem *copy_item = NULL;
  for (int i = 0; i < sched2->n_items; i++)
    if (sched2->items[i].kind == POLY_EXEC_COPY) copy_item = &sched2->items[i];
  ASSERT_TRUE(copy_item != NULL);
  ASSERT_INT_EQ(copy_item->n_buf_slots, 2);
  ASSERT_TRUE(copy_item->buf_slot_indices[0] >= 0);
  ASSERT_TRUE(copy_item->buf_slot_indices[1] >= 0);
  ASSERT_PTR_EQ(sched2->buf_slots[copy_item->buf_slot_indices[0]].buf_uop, dst2);
  ASSERT_PTR_EQ(sched2->buf_slots[copy_item->buf_slot_indices[1]].buf_uop, src2);
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

  int n_copy_items = 0, n_intermediates = 0, n_zero = 0;
  for (int i = 0; i < ps->n_items; i++)
    if (ps->items[i].kind == POLY_EXEC_COPY) n_copy_items++;
  for (int i = 0; i < ps->n_buf_slots; i++) {
    if (!ps->buf_slots[i].is_intermediate) continue;
    n_intermediates++;
    if (ps->buf_slots[i].needs_zero) n_zero++;
  }

  ASSERT_INT_EQ(n_copy_items, 2);
  ASSERT_INT_EQ(n_intermediates, 2);
  ASSERT_INT_EQ(n_zero, 0);

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
  ASSERT_TRUE(sched1->n_items > 1);
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
  ASSERT_TRUE(sched->n_items >= 1);

  bool checked_kernel = false;
  for (int k = 0; k < sched->n_items; k++) {
    PolyExecItem *item = &sched->items[k];
    ASSERT_INT_EQ(poly_exec_item_lower(ctx, sched, k, POLY_DEVICE_WEBGPU), 0);
    ASSERT_INT_EQ(item->prg.n_params, item->n_buf_slots);

    int n_lin = 0;
    PolyUOp **lin = poly_linearize_webgpu(ctx, item->root, &n_lin);
    ASSERT_TRUE(lin != NULL);

    int seen_params = 0;
    for (int i = 0; i < n_lin; i++) {
      if (lin[i]->op != POLY_OP_PARAM) continue;
      ASSERT_TRUE(seen_params < item->n_buf_slots);
      ASSERT_INT_EQ((int)lin[i]->arg.i, seen_params);
      ASSERT_INT_EQ(
          item->prg.param_to_slot[seen_params], item->buf_slot_indices[(int)lin[i]->arg.i]
      );
      seen_params++;
    }
    ASSERT_INT_EQ(seen_params, item->n_buf_slots);
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

TEST(schedule_runtime, webgpu_triu_small_expander_drops_residual_where_like_tinygrad) {
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

TEST(schedule_runtime, webgpu_tril_small_expander_drops_residual_where_like_tinygrad) {
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

TEST(schedule_runtime, webgpu_triu_small_devector_matches_tinygrad_load_count) {
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

TEST(schedule_runtime, webgpu_tril_small_devector_matches_tinygrad_load_count) {
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
  /* Matches temp/tg_webgpu_tri_stage_probe.py: after final WGSL rewrite the
   * mask is still represented as the LOAD's INDEX.valid gate, not a WHERE. */
  ASSERT_INT_EQ(count_root_ops(ctx, rewritten, POLY_OP_WHERE), 0);
  ASSERT_INT_EQ(count_root_ops(ctx, rewritten, POLY_OP_LOAD), 1);
  ASSERT_INT_EQ(count_root_gated_loads(ctx, rewritten), 1);

  poly_schedule_free(sched);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(schedule_runtime, webgpu_unified_tril_has_no_vector_gated_loads) {
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

TEST(schedule_runtime, webgpu_tril_add_loads_keeps_residual_where_like_tinygrad) {
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

TEST(schedule_runtime, webgpu_tril_post_index_symbolic_lowers_where_to_gated_load_like_tinygrad) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_TRUE(ctx != NULL);

  PolySchedule *sched = build_tri_schedule(ctx, 9, true);
  ASSERT_TRUE(sched != NULL);
  ASSERT_INT_EQ(sched->n_items, 1);

  PolyUOp *u = apply_webgpu_tri_stage_root(ctx, sched->items[0].root, true, true);

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
  /* Matches temp/tg_webgpu_tri_stage_probe.py: after final WGSL rewrite the
   * mask is still represented as the LOAD's INDEX.valid gate, not a WHERE. */
  ASSERT_INT_EQ(count_root_ops(ctx, rewritten, POLY_OP_WHERE), 0);
  ASSERT_INT_EQ(count_root_ops(ctx, rewritten, POLY_OP_LOAD), 1);
  ASSERT_INT_EQ(count_root_gated_loads(ctx, rewritten), 1);

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
  ASSERT_TRUE(ps->graph_hash != 0);

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
  ASSERT_INT_EQ(ps->n_items, 1);

  PolyRewriteOpts opts = {
      .optimize = true,
      .devectorize = 1,
      .caps = poly_c_renderer_caps(),
      .device = POLY_DEVICE_CPU,
      .opt_policy = POLY_OPT_HEURISTIC,
      .extra_matcher = poly_pm_c_renderer_extra(),
  };
  int n_lin = 0;
  PolyUOp **lin = poly_linearize_ex(ctx, ps->items[0].root, opts, &n_lin);
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
  ASSERT_INT_EQ(es->n_runners, 1);
  ASSERT_INT_EQ(es->runners[0].grid[0], 4);

  float *da = malloc((size_t)N * sizeof(float));
  float *db = malloc((size_t)N * sizeof(float));
  float *dout = calloc((size_t)N, sizeof(float));
  void **slot_data = calloc((size_t)ps->n_buf_slots, sizeof(void *));
  ASSERT_TRUE(da != NULL);
  ASSERT_TRUE(db != NULL);
  ASSERT_TRUE(dout != NULL);
  ASSERT_TRUE(slot_data != NULL);

  for (int i = 0; i < N; i++) {
    da[i] = (float)i;
    db[i] = (float)(N - i);
  }
  for (int i = 0; i < ps->n_buf_slots; i++) {
    if (ps->buf_slots[i].buf_uop == a)
      slot_data[i] = da;
    else if (ps->buf_slots[i].buf_uop == b)
      slot_data[i] = db;
    else if (ps->buf_slots[i].buf_uop == out)
      slot_data[i] = dout;
  }

  ASSERT_INT_EQ(poly_run_compiled_schedule(es, slot_data, ps->n_buf_slots, NULL, 0), 0);
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
  ASSERT_INT_EQ(es->n_runners, 1);
  ASSERT_INT_EQ(es->runners[0].grid[0], 4);

  float *a1 = malloc((size_t)N * sizeof(float));
  float *b1 = malloc((size_t)N * sizeof(float));
  float *o1 = calloc((size_t)N, sizeof(float));
  float *a2 = malloc((size_t)N * sizeof(float));
  float *b2 = malloc((size_t)N * sizeof(float));
  float *o2 = calloc((size_t)N, sizeof(float));
  void **slots1 = calloc((size_t)ps->n_buf_slots, sizeof(void *));
  void **slots2 = calloc((size_t)ps->n_buf_slots, sizeof(void *));
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
  for (int i = 0; i < ps->n_buf_slots; i++) {
    if (ps->buf_slots[i].buf_uop == a) {
      slots1[i] = a1;
      slots2[i] = a2;
    } else if (ps->buf_slots[i].buf_uop == b) {
      slots1[i] = b1;
      slots2[i] = b2;
    } else if (ps->buf_slots[i].buf_uop == out) {
      slots1[i] = o1;
      slots2[i] = o2;
    }
  }

  ASSERT_INT_EQ(poly_run_compiled_schedule(es, slots1, ps->n_buf_slots, NULL, 0), 0);
  ASSERT_INT_EQ(poly_run_compiled_schedule(es, slots2, ps->n_buf_slots, NULL, 0), 0);

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
  ASSERT_INT_EQ(es->n_runners, 1);
  ASSERT_INT_EQ(es->runners[0].grid[0], 1);

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
      POLY_TEST_HOST_VIEW(a, da), POLY_TEST_HOST_VIEW(b, db), POLY_TEST_HOST_VIEW(out, dout_direct)
  };
  ASSERT_INT_EQ(poly_test_realize_buffer_views(ctx, sink, bindings, 3), 0);

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
  int n_zero = 0;
  for (int i = 0; i < ps->n_buf_slots; i++) {
    if (!ps->buf_slots[i].is_intermediate) continue;
    n_inter++;
    if (ps->buf_slots[i].needs_zero) n_zero++;
  }
  ASSERT_TRUE(n_inter > 0);
  ASSERT_TRUE(n_zero > 0);
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
