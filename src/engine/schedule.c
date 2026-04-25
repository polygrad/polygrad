/*
 * schedule.c -- Tinygrad-aligned engine scheduling and compiled execution.
 *
 * This file plays the role of tinygrad/engine/schedule.py in C:
 *   - schedule construction from a tensor sink
 *   - backend lowering of schedule items
 *   - schedule execution with reusable workspace
 */

#define _POSIX_C_SOURCE 200809L
#include "engine/schedule.h"
#include "utils.h"
#include "frontend_internal.h"
#include "codegen.h"
#include "interp.h"
#include "schedule/rangeify.h"
#include "runtime_webgpu.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>

static int webgpu_collect_param_order(PolyUOp **lin, int n_lin, int *order, int n_params) {
  int seen = 0;
  for (int i = 0; i < n_lin; i++) {
    if (lin[i]->op != POLY_OP_PARAM) continue;
    if (seen >= n_params) return -1;
    int idx = (int)lin[i]->arg.i;
    if (idx < 0 || idx >= n_params) return -1;
    order[seen++] = idx;
  }
  return (seen == n_params) ? 0 : -1;
}

static int webgpu_fill_param_slots(PolyCtx *ctx, PolyExecItem *item, PolyRunner *runner) {
  int n_params = item ? item->n_buf_slots : 0;
  runner->n_params = n_params;
  if (n_params <= 0) return 0;

  int n_lin = 0;
  PolyUOp **lin = poly_linearize_webgpu(ctx, item->root, &n_lin);
  if (!lin) return -1;

  int *param_order = malloc((size_t)n_params * sizeof(int));
  int *param_to_slot = malloc((size_t)n_params * sizeof(int));
  if (!param_order || !param_to_slot ||
      webgpu_collect_param_order(lin, n_lin, param_order, n_params) != 0) {
    free(param_order);
    free(param_to_slot);
    free(lin);
    return -1;
  }

  for (int i = 0; i < n_params; i++) {
    int src_idx = param_order[i];
    param_to_slot[i] = item->buf_slot_indices[src_idx];
  }

  free(param_order);
  free(lin);
  runner->param_to_slot = param_to_slot;
  return 0;
}

static bool poly_lookup_var_value(
    PolyUOp *var, const PolyVarBinding *bindings, int n_bindings, PolyArg *out
) {
  if (!var || !bindings || !out) return false;
  for (int i = 0; i < n_bindings; i++) {
    if (bindings[i].var != var) continue;
    if (poly_dtype_is_bool(var->dtype))
      *out = poly_arg_bool(bindings[i].value != 0);
    else if (poly_dtype_is_float(var->dtype))
      *out = poly_arg_float((double)bindings[i].value);
    else
      *out = poly_arg_int(bindings[i].value);
    return true;
  }
  return false;
}

static bool poly_eval_launch_expr(
    PolyUOp *u, const PolyVarBinding *bindings, int n_bindings, PolyArg *out
) {
  if (!u || !out) return false;

  if (u->op == POLY_OP_CONST) {
    *out = u->arg;
    return true;
  }
  if (u->op == POLY_OP_DEFINE_VAR) return poly_lookup_var_value(u, bindings, n_bindings, out);

  if (u->op == POLY_OP_CAST) {
    PolyArg operand;
    if (u->n_src < 1 || !poly_eval_launch_expr(u->src[0], bindings, n_bindings, &operand))
      return false;
    *out = poly_exec_alu(POLY_OP_CAST, u->dtype, &operand, 1);
    return out->kind != POLY_ARG_INVALID;
  }

  if (poly_opset_has(POLY_GROUP_ALU, u->op)) {
    if (u->n_src > 3) return false;
    PolyArg operands[3];
    for (int i = 0; i < u->n_src; i++) {
      if (!poly_eval_launch_expr(u->src[i], bindings, n_bindings, &operands[i])) return false;
    }
    *out = poly_exec_alu(u->op, u->dtype, operands, u->n_src);
    return out->kind != POLY_ARG_INVALID;
  }

  return false;
}

static int64_t poly_launch_arg_to_i64(PolyArg arg) {
  switch (arg.kind) {
  case POLY_ARG_BOOL:
    return arg.b ? 1 : 0;
  case POLY_ARG_FLOAT:
    return (int64_t)arg.f;
  case POLY_ARG_INT:
  default:
    return arg.i;
  }
}

static int poly_launch_dim_upper_bound(PolyCtx *ctx, PolyUOp *expr) {
  if (!expr) return 1;
  int64_t lo = 0, hi = 1;
  poly_uop_minmax(ctx, expr, &lo, &hi);
  if (hi <= 0) return 1;
  if (hi > INT32_MAX) return INT32_MAX;
  return (int)hi;
}

static int poly_resolve_runner_launch_dims(
    PolyRunner *runner, const PolyVarBinding *bindings, int n_bindings
) {
  if (!runner) return -1;

  for (int dim = 0; dim < 3; dim++) {
    if (runner->grid_exprs[dim]) {
      PolyArg value;
      if (!poly_eval_launch_expr(runner->grid_exprs[dim], bindings, n_bindings, &value)) return -1;
      int64_t resolved = poly_launch_arg_to_i64(value);
      runner->grid[dim] = (resolved > 0 && resolved <= INT32_MAX) ? (int)resolved : 1;
    }
    if (runner->block_exprs[dim]) {
      PolyArg value;
      if (!poly_eval_launch_expr(runner->block_exprs[dim], bindings, n_bindings, &value)) return -1;
      int64_t resolved = poly_launch_arg_to_i64(value);
      runner->block[dim] = (resolved > 0 && resolved <= INT32_MAX) ? (int)resolved : 1;
    }
  }

  return 0;
}

static void debug_dump_webgpu_runner_args(
    const PolyCompiledSchedule *plan,
    int exec_step,
    int kernel_idx,
    const PolyRunner *runner,
    void **args
) {
#ifdef __EMSCRIPTEN__
  if (!plan || !plan->schedule || !runner || !args) return;
  if (plan->device != POLY_DEVICE_WEBGPU || !poly_debug_at_least(7)) return;

  const PolySchedule *sched = plan->schedule;
  fprintf(
      stderr,
      "[webgpu-run] step=%d kernel=%d n_params=%d n_vars=%d\n",
      exec_step,
      kernel_idx,
      runner->n_params,
      runner->n_vars
  );
  for (int i = 0; i < runner->n_params; i++) {
    int slot = runner->param_to_slot ? runner->param_to_slot[i] : -1;
    const PolyScheduleBufSlot *bs =
        (slot >= 0 && slot < sched->n_buf_slots) ? &sched->buf_slots[slot] : NULL;
    fprintf(
        stderr,
        "  [param %d] slot=%d handle=%p buf_uop=%p interm=%d ext=%d nbytes=%lld numel=%lld\n",
        i,
        slot,
        args[i],
        bs ? (void *)bs->buf_uop : NULL,
        bs ? (int)bs->is_intermediate : -1,
        bs ? bs->external_buf_idx : -1,
        bs ? (long long)bs->nbytes : -1LL,
        bs ? (long long)bs->numel : -1LL
    );
  }
  for (int i = 0; i < runner->n_vars; i++) {
    void *var_ptr = args[runner->n_params + i];
    fprintf(
        stderr, "  [var %d] ptr=%p val=%d\n", i, var_ptr, var_ptr ? *(int *)var_ptr : 0
    );
  }
#else
  (void)plan;
  (void)exec_step;
  (void)kernel_idx;
  (void)runner;
  (void)args;
#endif
}

static void debug_dump_webgpu_schedule_args(
    const PolySchedule *sched,
    PolyDevice device,
    int exec_step,
    int kernel_idx,
    const PolyRunner *runner,
    void **args
) {
#ifdef __EMSCRIPTEN__
  if (!sched || !runner || !args) return;
  if (device != POLY_DEVICE_WEBGPU || !poly_debug_at_least(7)) return;

  fprintf(
      stderr,
      "[webgpu-run] step=%d kernel=%d n_params=%d n_vars=%d\n",
      exec_step,
      kernel_idx,
      runner->n_params,
      runner->n_vars
  );
  for (int i = 0; i < runner->n_params; i++) {
    int slot = runner->param_to_slot ? runner->param_to_slot[i] : -1;
    const PolyScheduleBufSlot *bs =
        (slot >= 0 && slot < sched->n_buf_slots) ? &sched->buf_slots[slot] : NULL;
    fprintf(
        stderr,
        "  [param %d] slot=%d handle=%p buf_uop=%p interm=%d ext=%d nbytes=%lld numel=%lld\n",
        i,
        slot,
        args[i],
        bs ? (void *)bs->buf_uop : NULL,
        bs ? (int)bs->is_intermediate : -1,
        bs ? bs->external_buf_idx : -1,
        bs ? (long long)bs->nbytes : -1LL,
        bs ? (long long)bs->numel : -1LL
    );
  }
#else
  (void)sched;
  (void)device;
  (void)exec_step;
  (void)kernel_idx;
  (void)runner;
  (void)args;
#endif
}

static bool is_intermediate_buffer_uop(PolyUOp *buf) {
  return buf && buf->op == POLY_OP_BUFFER && buf->n_src >= 1 && buf->src[0] &&
         buf->src[0]->op == POLY_OP_LUNIQUE;
}

static int collect_external_buf_order_from_kernel_graph(PolyUOp *kernel_graph, PolyUOp **buf_order) {
  PolyUOp *all_bufs[POLY_MAX_REALIZE_BUFS];
  PolyUOp *visited[POLY_MAX_STRUCT_NODES];
  int n_all = 0, n_visited = 0;
  poly_collect_buf_order(kernel_graph, all_bufs, &n_all, visited, &n_visited);

  int n_external = 0;
  for (int i = 0; i < n_all && i < POLY_MAX_REALIZE_BUFS; i++) {
    if (!is_intermediate_buffer_uop(all_bufs[i])) buf_order[n_external++] = all_bufs[i];
  }
  return n_external;
}

static PolyUOp *poly_strip_copy_end_chain(PolyUOp *u) {
  while (u && u->op == POLY_OP_END && u->n_src >= 1)
    u = u->src[0];
  return u;
}

static bool poly_copy_param_index(PolyUOp *u, PolyUOp **param_out, PolyUOp **idx_out) {
  if (!u || u->op != POLY_OP_INDEX || u->n_src != 2) return false;
  if (!u->src[0] || u->src[0]->op != POLY_OP_PARAM) return false;
  if (!u->src[1] || (u->src[1]->op != POLY_OP_RANGE && u->src[1]->op != POLY_OP_CONST)) return false;
  if (param_out) *param_out = u->src[0];
  if (idx_out) *idx_out = u->src[1];
  return true;
}

static bool poly_kernel_is_trivial_copy(PolyUOp *kernel) {
  if (!kernel || kernel->op != POLY_OP_SINK || kernel->n_src != 1) return false;

  PolyUOp *body = poly_strip_copy_end_chain(kernel->src[0]);
  if (!body || body->op != POLY_OP_STORE || body->n_src != 2) return false;

  PolyUOp *dst_param = NULL, *src_param = NULL;
  PolyUOp *dst_idx = NULL, *src_idx = NULL;
  if (!poly_copy_param_index(body->src[0], &dst_param, &dst_idx)) return false;
  if (!poly_copy_param_index(body->src[1], &src_param, &src_idx)) return false;
  if (dst_param == src_param || dst_idx != src_idx) return false;

  return poly_dtype_eq(poly_dtype_scalar(dst_param->dtype), poly_dtype_scalar(src_param->dtype));
}

static PolySchedule *build_schedule_from_kernel_graph(
    PolyCtx *ctx,
    PolyUOp *kernel_graph,
    PolyCompileMode mode,
    uint32_t graph_hash,
    PolyUOp **buf_order_orig,
    int n_bufs_orig,
    PolyUOp **buf_order_post,
    int n_bufs_post,
    const PolyVarBinding *default_vars,
    int n_default_vars
) {
  if (!kernel_graph || kernel_graph->op != POLY_OP_SINK) {
    fprintf(stderr, "polygrad: build_schedule_from_kernel_graph: expected SINK\n");
    return NULL;
  }

  PolyKernelScheduleResult sr = poly_build_kernel_schedule_from_kernel_graph(ctx, kernel_graph);
  if (sr.n_kernels < 1) {
    poly_kernel_schedule_result_free(&sr);
    return NULL;
  }

  PolySchedule *ps = calloc(1, sizeof(PolySchedule));
  if (!ps) {
    poly_kernel_schedule_result_free(&sr);
    return NULL;
  }
  ps->mode = mode;
  ps->graph_hash = graph_hash;
  ps->loss_buf_slot = -1;

  ps->n_default_vars = n_default_vars;
  if (n_default_vars > 0) {
    ps->default_vars = malloc((size_t)n_default_vars * sizeof(PolyVarBinding));
    memcpy(ps->default_vars, default_vars, (size_t)n_default_vars * sizeof(PolyVarBinding));
  }

  int n_external = n_bufs_orig;
  int n_intermediate = sr.n_intermediates;
  ps->n_buf_slots = n_external + n_intermediate;

  if (ps->n_buf_slots > 0) {
    ps->buf_slots = calloc((size_t)ps->n_buf_slots, sizeof(PolyScheduleBufSlot));

    for (int i = 0; i < n_external; i++) {
      PolyScheduleBufSlot *slot = &ps->buf_slots[i];
      PolyUOp *buf = buf_order_orig[i];
      slot->buf_uop = buf;
      slot->is_intermediate = false;
      slot->external_buf_idx = i;
      slot->dtype = buf ? poly_dtype_scalar(buf->dtype) : POLY_FLOAT32;
      slot->numel = (buf && buf->arg.kind == POLY_ARG_INT) ? buf->arg.i : 0;
      if (slot->numel > 0) slot->nbytes = slot->numel * poly_dtype_itemsize(slot->dtype);
    }

    for (int b = 0; b < n_intermediate; b++) {
      PolyScheduleBufSlot *slot = &ps->buf_slots[n_external + b];
      slot->is_intermediate = true;
      slot->external_buf_idx = -1;
      if (sr.intermediate_buf_uops && sr.intermediate_buf_uops[b]) {
        slot->buf_uop = sr.intermediate_buf_uops[b];
        slot->dtype = poly_dtype_scalar(sr.intermediate_buf_uops[b]->dtype);
      } else {
        slot->dtype = POLY_FLOAT32;
      }
      slot->numel = sr.intermediate_sizes ? sr.intermediate_sizes[b] : 0;
      int itemsize = (sr.intermediate_itemsizes && sr.intermediate_itemsizes[b] > 0)
                         ? sr.intermediate_itemsizes[b]
                         : (int)sizeof(float);
      slot->nbytes = slot->numel * itemsize;
    }
  }

  PolyMap *inter_set = NULL;
  if (n_intermediate > 0 && sr.intermediate_buf_uops) {
    inter_set = poly_map_new((size_t)(n_intermediate < 4 ? 4 : n_intermediate));
    for (int b = 0; b < n_intermediate; b++) {
      PolyUOp *ib = sr.intermediate_buf_uops[b];
      poly_map_set(
          inter_set, poly_ptr_hash(ib), ib, (PolyUOp *)(intptr_t)(n_external + b + 1), poly_ptr_eq
      );
    }
  }

  ps->n_items = sr.n_kernels;
  ps->items = calloc((size_t)sr.n_kernels, sizeof(PolyExecItem));

  for (int k = 0; k < sr.n_kernels; k++) {
    PolyExecItem *item = &ps->items[k];
    item->root = sr.kernels[k];
    item->kind = poly_kernel_is_trivial_copy(item->root) ? POLY_EXEC_COPY : POLY_EXEC_COMPUTE;

    int np = sr.kernel_n_params[k];
    item->n_buf_slots = np;
    item->buf_slot_indices = malloc((size_t)np * sizeof(int));

    for (int i = 0; i < np; i++) {
      PolyUOp *buf = sr.param_to_buf[k][i];
      item->buf_slot_indices[i] = -1;

      if (buf->op == POLY_OP_BUFFER) {
        int pos = poly_find_buf_position(buf, buf_order_post, n_bufs_post);
        if (pos >= 0) {
          item->buf_slot_indices[i] = pos;
        } else if (inter_set) {
          PolyUOp *v = poly_map_get(inter_set, poly_ptr_hash(buf), buf, poly_ptr_eq);
          if (v) item->buf_slot_indices[i] = (int)((intptr_t)v - 1);
        }
      }

      if (item->buf_slot_indices[i] < 0) {
        fprintf(
            stderr,
            "polygrad: build_schedule_from_kernel_graph: unresolved param %d in kernel %d\n",
            i,
            k
        );
        if (inter_set) poly_map_destroy(inter_set);
        goto cleanup;
      }
    }

    int nv = (sr.kernel_n_vars ? sr.kernel_n_vars[k] : 0);
    item->n_var_uops = nv;
    if (nv > 0 && sr.var_to_buf && sr.var_to_buf[k]) {
      item->var_uops = malloc((size_t)nv * sizeof(PolyUOp *));
      memcpy(item->var_uops, sr.var_to_buf[k], (size_t)nv * sizeof(PolyUOp *));
    }
  }

  if (inter_set) poly_map_destroy(inter_set);

  ps->exec_order = malloc((size_t)sr.n_kernels * sizeof(int));
  if (sr.exec_order)
    memcpy(ps->exec_order, sr.exec_order, (size_t)sr.n_kernels * sizeof(int));
  else
    for (int k = 0; k < sr.n_kernels; k++)
      ps->exec_order[k] = k;

  poly_kernel_schedule_result_free(&sr);
  return ps;

cleanup:
  poly_schedule_free(ps);
  poly_kernel_schedule_result_free(&sr);
  return NULL;
}

/* tinygrad/engine/schedule.py: complete_create_schedule_with_vars */

PolySchedule *poly_complete_create_schedule_with_vars(PolyCtx *ctx, PolyUOp *sink, PolyCompileMode mode) {
  if (!sink || sink->op != POLY_OP_SINK) {
    fprintf(stderr, "polygrad: complete_create_schedule_with_vars: expected SINK\n");
    return NULL;
  }

  /* --- Collect pre-strip buffer ordering -------------------------------- */
  PolyUOp **buf_order_orig = calloc(POLY_MAX_REALIZE_BUFS, sizeof(PolyUOp *));
  PolyUOp **dfs_visited = calloc(POLY_MAX_STRUCT_NODES, sizeof(PolyUOp *));
  int n_bufs_orig = 0, n_dfs = 0;
  poly_collect_buf_order(sink, buf_order_orig, &n_bufs_orig, dfs_visited, &n_dfs);

  /* --- Extract BIND defaults and strip ---------------------------------- */
  PolyVarBinding bind_vals[16];
  int n_bind_vals = 0;
  sink = poly_strip_bind_values(ctx, sink, bind_vals, &n_bind_vals, 16, NULL, 0);

  /* Post-strip buffer ordering (scheduler sees these pointers) */
  PolyUOp **buf_order_post = calloc(POLY_MAX_REALIZE_BUFS, sizeof(PolyUOp *));
  int n_bufs_post = 0;
  n_dfs = 0;
  poly_collect_buf_order(sink, buf_order_post, &n_bufs_post, dfs_visited, &n_dfs);
  free(dfs_visited);

  uint32_t ghash = poly_structural_hash(sink) ^ (POLY_SCHED_CACHE_VERSION * 2654435761u);

  /* Keep the first realization boundary explicit: bind/default-var handling
   * stays at the tensor sink, then scheduling consumes get_kernel_graph. */
  PolyUOp *kernel_graph = poly_get_kernel_graph(ctx, sink);
  PolySchedule *ps = build_schedule_from_kernel_graph(
      ctx,
      kernel_graph,
      mode,
      ghash,
      buf_order_orig,
      n_bufs_orig,
      buf_order_post,
      n_bufs_post,
      bind_vals,
      n_bind_vals
  );
  free(buf_order_orig);
  free(buf_order_post);
  return ps;
}

/* tinygrad/engine/schedule.py: create_schedule
 * Build the backend-neutral schedule from the kernel-graph boundary directly. */
PolySchedule *poly_create_schedule(PolyCtx *ctx, PolyUOp *kernel_graph) {
  if (!kernel_graph || kernel_graph->op != POLY_OP_SINK) {
    fprintf(stderr, "polygrad: create_schedule: expected SINK\n");
    return NULL;
  }

  PolyUOp *buf_order[POLY_MAX_REALIZE_BUFS];
  int n_external = collect_external_buf_order_from_kernel_graph(kernel_graph, buf_order);
  uint32_t ghash = poly_structural_hash(kernel_graph) ^ (POLY_SCHED_CACHE_VERSION * 2654435761u);
  return build_schedule_from_kernel_graph(
      ctx,
      kernel_graph,
      POLY_MODE_CALL,
      ghash,
      buf_order,
      n_external,
      buf_order,
      n_external,
      NULL,
      0
  );
}

static void poly_runner_cleanup(PolyRunner *runner, PolyDevice device) {
  if (!runner) return;
  const PolyBackendDesc *backend = poly_backend_get(device);
  if (runner->handle) {
    if (runner->free_handle) runner->free_handle(runner);
    else if (backend) backend->free_runner(runner);
  }
  free(runner->param_to_slot);
  free(runner->var_indices);
  memset(runner, 0, sizeof(*runner));
}

static void poly_schedule_runtime_destroy(PolySchedule *sched) {
  if (!sched) return;

  for (int i = 0; i < sched->n_items; i++) {
    PolyExecItem *item = &sched->items[i];
    if (item->prg_valid) poly_runner_cleanup(&item->prg, item->lowered_device);
    item->prg_valid = false;
    item->lowered_device = POLY_DEVICE_AUTO;
    item->lowered_env_stamp = 0;
  }

  if (sched->run_intermediates) {
    for (int i = 0; i < sched->n_run_intermediates; i++) {
      if (sched->run_intermediates[i].ptr && sched->run_allocator)
        sched->run_allocator->free(&sched->run_intermediates[i], sched->run_allocator->dev_ctx);
    }
    free(sched->run_intermediates);
  }
  if (sched->run_kernel_args) {
    for (int i = 0; i < sched->n_items; i++)
      free(sched->run_kernel_args[i]);
    free(sched->run_kernel_args);
  }
  free(sched->run_slot_to_data);
  free(sched->run_merged_vars);
  free(sched->run_var_int_storage);

  sched->run_device = POLY_DEVICE_AUTO;
  sched->run_allocator = NULL;
  sched->run_intermediates = NULL;
  sched->n_run_intermediates = 0;
  sched->run_kernel_args = NULL;
  sched->run_slot_to_data = NULL;
  sched->n_run_slot_to_data = 0;
  sched->run_merged_vars = NULL;
  sched->run_merged_vars_cap = 0;
  sched->run_var_int_storage = NULL;
  sched->run_var_int_cap = 0;
}

void poly_schedule_free(PolySchedule *step) {
  if (!step) return;
  poly_schedule_runtime_destroy(step);
  for (int i = 0; i < step->n_items; i++) {
    free(step->items[i].buf_slot_indices);
    free(step->items[i].var_uops);
  }
  free(step->items);
  free(step->buf_slots);
  free(step->exec_order);
  free(step->default_vars);
  free(step->grad_buf_slots);
  free(step);
}

/* tinygrad/engine/schedule.py: lower_sink_to_linear
 * Polygrad exposes a LINEAR UOp whose sources are the scheduled roots in
 * execution order. This keeps the stage boundary inspectable without
 * introducing a second schedule container type. */
PolyUOp *poly_lower_sink_to_linear(PolyCtx *ctx, PolyUOp *sink, PolyCompileMode mode) {
  (void)mode;
  /* Match tinygrad's stage composition here: lower_sink_to_linear should
   * consume get_kernel_graph -> create_schedule, not the full realize entry. */
  PolyUOp *kernel_graph = poly_get_kernel_graph(ctx, sink);
  PolySchedule *schedule = poly_create_schedule(ctx, kernel_graph);
  if (!schedule) return NULL;

  PolyUOp **linear_src = calloc((size_t)schedule->n_items, sizeof(PolyUOp *));
  if (!linear_src) {
    poly_schedule_free(schedule);
    return NULL;
  }
  for (int step = 0; step < schedule->n_items; step++) {
    int idx = schedule->exec_order ? schedule->exec_order[step] : step;
    linear_src[step] = schedule->items[idx].root;
  }
  PolyUOp *linear = poly_uop(ctx, POLY_OP_LINEAR, POLY_VOID, linear_src, schedule->n_items, poly_arg_none());
  free(linear_src);
  poly_schedule_free(schedule);
  return linear;
}

/* ══════════════════════════════════════════════════════════════════════ */
/*  Allocators                                                          */
/* ══════════════════════════════════════════════════════════════════════ */

/* CPU allocator */

static void *cpu_alloc(size_t nbytes, void *dev_ctx) {
  (void)dev_ctx;
  return calloc(1, nbytes);
}

static void cpu_free_alloc(const PolyBuffer *buffer, void *dev_ctx) {
  (void)dev_ctx;
  free(buffer ? buffer->ptr : NULL);
}

static int cpu_copy_in(const PolyBuffer *dst, const PolyBuffer *src, size_t n, void *dev_ctx) {
  (void)dev_ctx;
  memcpy(dst->ptr, src->ptr, n);
  return 0;
}

static int cpu_copy_out(const PolyBuffer *dst, const PolyBuffer *src, size_t n, void *dev_ctx) {
  (void)dev_ctx;
  memcpy(dst->ptr, src->ptr, n);
  return 0;
}

static int cpu_copy_between(const PolyBuffer *dst, const PolyBuffer *src, size_t n, void *dev_ctx) {
  (void)dev_ctx;
  memcpy(dst->ptr, src->ptr, n);
  return 0;
}

const PolyAllocator POLY_CPU_ALLOCATOR = {
    .alloc = cpu_alloc,
    .free = cpu_free_alloc,
    .copy_in = cpu_copy_in,
    .copy_out = cpu_copy_out,
    .copy_between = cpu_copy_between,
    .host_addressable = true,
    .dev_ctx = NULL,
};

/* HOST allocator
 *
 * Native builds borrow frontend-owned host memory directly. Retiring a HOST
 * residency should drop the frontend's strong owner entry keyed by this
 * PolyBuffer* address value; the actual host bytes remain frontend-managed.
 *
 * Emscripten builds currently stage imported host data into wasm heap in the
 * JS binding before calling poly_buffer_from_host. Retiring the HOST residency
 * still notifies the frontend so it can drop its strong owner entry, and then
 * frees the staged wasm heap pointer.
 */

static void *host_alloc(size_t nbytes, void *dev_ctx) {
  (void)nbytes;
  (void)dev_ctx;
  return NULL;
}

static void host_free_alloc(const PolyBuffer *buffer, void *dev_ctx) {
  (void)dev_ctx;
  if (!buffer) return;
  poly_frontend_buffer_release_key((uintptr_t)buffer);
#ifdef __EMSCRIPTEN__
  free(buffer->ptr);
#endif
}

static int host_copy_in(const PolyBuffer *dst, const PolyBuffer *src, size_t n, void *dev_ctx) {
  (void)dev_ctx;
#ifdef __EMSCRIPTEN__
  if (dst && dst->device == POLY_DEVICE_HOST && src && src->device == POLY_DEVICE_WEBGPU) {
    return poly_webgpu_get_allocator()->copy_out(dst, src, n, NULL);
  }
  if (dst && dst->device == POLY_DEVICE_HOST && src && src->ptr) {
    return poly_browser_host_copy_in((uintptr_t)dst, src->ptr, n);
  }
#endif
  if (!dst || !dst->ptr || !src || !src->ptr) return -1;
  memcpy(dst->ptr, src->ptr, n);
  return 0;
}

static int host_copy_out(const PolyBuffer *dst, const PolyBuffer *src, size_t n, void *dev_ctx) {
  (void)dev_ctx;
#ifdef __EMSCRIPTEN__
  if (src && src->device == POLY_DEVICE_HOST && dst && dst->ptr) {
    return poly_browser_host_copy_out((uintptr_t)src, dst->ptr, n);
  }
#endif
  if (!dst || !dst->ptr || !src || !src->ptr) return -1;
  memcpy(dst->ptr, src->ptr, n);
  return 0;
}

static int host_copy_between(const PolyBuffer *dst, const PolyBuffer *src, size_t n, void *dev_ctx) {
  (void)dev_ctx;
#ifdef __EMSCRIPTEN__
  if (dst && src && dst->device == POLY_DEVICE_HOST && src->device == POLY_DEVICE_HOST) {
    uint8_t *tmp = malloc(n);
    if (!tmp) return -1;
    int rc = poly_browser_host_copy_out((uintptr_t)src, tmp, n);
    if (rc == 0) rc = poly_browser_host_copy_in((uintptr_t)dst, tmp, n);
    free(tmp);
    return rc;
  }
#endif
  if (!dst || !dst->ptr || !src || !src->ptr) return -1;
  memcpy(dst->ptr, src->ptr, n);
  return 0;
}

static const PolyAllocator POLY_HOST_ALLOCATOR = {
    .alloc = host_alloc,
    .free = host_free_alloc,
    .copy_in = host_copy_in,
    .copy_out = host_copy_out,
    .copy_between = host_copy_between,
    .host_addressable =
#ifdef __EMSCRIPTEN__
        false,
#else
        true,
#endif
    .dev_ctx = NULL,
};

#ifdef __EMSCRIPTEN__
static void *wasm_alloc(size_t nbytes, void *dev_ctx) {
  (void)dev_ctx;
  return calloc(1, nbytes);
}

static void wasm_free_alloc(const PolyBuffer *buffer, void *dev_ctx) {
  (void)dev_ctx;
  free(buffer ? buffer->ptr : NULL);
}

static int wasm_copy_in(const PolyBuffer *dst, const PolyBuffer *src, size_t n, void *dev_ctx) {
  (void)dev_ctx;
  if (!dst || !dst->ptr || !src) return -1;
  if (src->device == POLY_DEVICE_HOST) {
    return poly_browser_host_copy_out((uintptr_t)src, dst->ptr, n);
  }
  if (!src->ptr) return -1;
  memcpy(dst->ptr, src->ptr, n);
  return 0;
}

static int wasm_copy_out(const PolyBuffer *dst, const PolyBuffer *src, size_t n, void *dev_ctx) {
  (void)dev_ctx;
  if (!dst || !src || !src->ptr) return -1;
  if (dst->device == POLY_DEVICE_HOST) {
    return poly_browser_host_copy_in((uintptr_t)dst, src->ptr, n);
  }
  if (!dst->ptr) return -1;
  memcpy(dst->ptr, src->ptr, n);
  return 0;
}

static int wasm_copy_between(const PolyBuffer *dst, const PolyBuffer *src, size_t n, void *dev_ctx) {
  (void)dev_ctx;
  if (!dst || !dst->ptr || !src || !src->ptr) return -1;
  memcpy(dst->ptr, src->ptr, n);
  return 0;
}

static const PolyAllocator POLY_WASM_ALLOCATOR = {
    .alloc = wasm_alloc,
    .free = wasm_free_alloc,
    .copy_in = wasm_copy_in,
    .copy_out = wasm_copy_out,
    .copy_between = wasm_copy_between,
    .host_addressable = true,
    .dev_ctx = NULL,
};
#endif

/* CUDA allocator */

#ifdef POLY_HAS_CUDA

static void *cuda_alloc_fn(size_t nbytes, void *dev_ctx) {
  (void)dev_ctx;
  unsigned long long dptr = poly_cuda_alloc(nbytes);
  return dptr ? (void *)(uintptr_t)dptr : NULL;
}

static void cuda_free_fn(const PolyBuffer *buffer, void *dev_ctx) {
  (void)dev_ctx;
  if (buffer && buffer->ptr) poly_cuda_free((unsigned long long)(uintptr_t)buffer->ptr);
}

static int cuda_copy_in_fn(const PolyBuffer *dst, const PolyBuffer *src, size_t n, void *dev_ctx) {
  (void)dev_ctx;
  return poly_cuda_copy_htod((unsigned long long)(uintptr_t)dst->ptr, src->ptr, n);
}

static int cuda_copy_out_fn(const PolyBuffer *dst, const PolyBuffer *src, size_t n, void *dev_ctx) {
  (void)dev_ctx;
  return poly_cuda_copy_dtoh(dst->ptr, (unsigned long long)(uintptr_t)src->ptr, n);
}

static int cuda_copy_between_fn(const PolyBuffer *dst, const PolyBuffer *src, size_t n, void *dev_ctx) {
  (void)dev_ctx;
  return poly_cuda_copy_dtod(
      (unsigned long long)(uintptr_t)dst->ptr, (unsigned long long)(uintptr_t)src->ptr, n
  );
}

const PolyAllocator POLY_CUDA_ALLOCATOR = {
    .alloc = cuda_alloc_fn,
    .free = cuda_free_fn,
    .copy_in = cuda_copy_in_fn,
    .copy_out = cuda_copy_out_fn,
    .copy_between = cuda_copy_between_fn,
    .dev_ctx = NULL,
};

#endif /* POLY_HAS_CUDA */

/* HIP allocator */

#ifdef POLY_HAS_HIP

static void *hip_alloc_fn(size_t nbytes, void *dev_ctx) {
  (void)dev_ctx;
  return poly_hip_alloc(nbytes);
}

static void hip_free_fn(const PolyBuffer *buffer, void *dev_ctx) {
  (void)dev_ctx;
  if (buffer && buffer->ptr) poly_hip_free(buffer->ptr);
}

static int hip_copy_in_fn(const PolyBuffer *dst, const PolyBuffer *src, size_t n, void *dev_ctx) {
  (void)dev_ctx;
  return poly_hip_copy_htod(dst->ptr, src->ptr, n);
}

static int hip_copy_out_fn(const PolyBuffer *dst, const PolyBuffer *src, size_t n, void *dev_ctx) {
  (void)dev_ctx;
  return poly_hip_copy_dtoh(dst->ptr, src->ptr, n);
}

static int hip_copy_between_fn(const PolyBuffer *dst, const PolyBuffer *src, size_t n, void *dev_ctx) {
  (void)dev_ctx;
  (void)dst;
  (void)src;
  (void)n;
  fprintf(stderr, "polygrad: hip_copy_between: not implemented\n");
  return -1;
}

const PolyAllocator POLY_HIP_ALLOCATOR = {
    .alloc = hip_alloc_fn,
    .free = hip_free_fn,
    .copy_in = hip_copy_in_fn,
    .copy_out = hip_copy_out_fn,
    .copy_between = hip_copy_between_fn,
    .dev_ctx = NULL,
};

#endif /* POLY_HAS_HIP */

/* ══════════════════════════════════════════════════════════════════════ */
/*  Backend-specific runner handle types                                 */
/* ══════════════════════════════════════════════════════════════════════ */

/* Interpreter: linearized UOp array */
typedef struct {
  PolyUOp **lin;
  int n_lin;
} InterpHandle;

/* CUDA: compiled program handle */
#ifdef POLY_HAS_CUDA
typedef struct {
  PolyCudaProgram *prog;
} CudaRunnerHandle;
#endif

/* HIP: compiled program handle */
#ifdef POLY_HAS_HIP
typedef struct {
  PolyHipProgram *prog;
} HipRunnerHandle;
#endif

/* WASM JIT: JS-side kernel cache index */
typedef struct {
  int kernel_id;
} WasmJitHandle;

typedef struct {
  size_t nbytes;
  PolyDevice device;
} CopyRunnerHandle;

static int copy_execute_fn(void *self, void **args, int n_args) {
  PolyRunner *runner = (PolyRunner *)self;
  CopyRunnerHandle *ch = (CopyRunnerHandle *)runner->handle;
  if (!ch || !args || n_args < 2 || !args[0] || !args[1]) return -1;

  const PolyBackendDesc *backend = poly_backend_get(ch->device);
  const PolyAllocator *alloc = backend ? backend->get_allocator() : NULL;
  if (!alloc) return -1;
  if (!alloc->copy_between && !alloc->host_addressable) return -1;

  PolyBuffer dst = {
      .ptr = args[0],
      .nbytes = ch->nbytes,
      .device = ch->device,
      .owned = false,
      .allocator = alloc,
      .src = NULL,
      .valid = true,
  };
  PolyBuffer src = {
      .ptr = args[1],
      .nbytes = ch->nbytes,
      .device = ch->device,
      .owned = false,
      .allocator = alloc,
      .src = NULL,
      .valid = true,
  };

  if (alloc->copy_between) return alloc->copy_between(&dst, &src, ch->nbytes, alloc->dev_ctx);
  return alloc->copy_in(&dst, &src, ch->nbytes, alloc->dev_ctx);
}

static void copy_free_fn(void *self) {
  PolyRunner *runner = (PolyRunner *)self;
  free(runner ? runner->handle : NULL);
}

static int poly_lower_copy_item(
    PolySchedule *schedule, PolyExecItem *item, PolyDevice device, PolyRunner *out
) {
  if (!schedule || !item || !out || item->n_buf_slots < 2) return -1;

  int dst_slot = item->buf_slot_indices[0];
  int src_slot = item->buf_slot_indices[1];
  if (dst_slot < 0 || dst_slot >= schedule->n_buf_slots || src_slot < 0 || src_slot >= schedule->n_buf_slots)
    return -1;

  const PolyBackendDesc *backend = poly_backend_get(device);
  const PolyAllocator *alloc = backend ? backend->get_allocator() : NULL;
  if (!alloc || (!alloc->copy_between && !alloc->host_addressable)) return -1;

  size_t dst_nbytes = (size_t)schedule->buf_slots[dst_slot].nbytes;
  size_t src_nbytes = (size_t)schedule->buf_slots[src_slot].nbytes;
  if (dst_nbytes == 0 || src_nbytes == 0 || dst_nbytes != src_nbytes) return -1;

  CopyRunnerHandle *ch = malloc(sizeof(CopyRunnerHandle));
  if (!ch) return -1;
  ch->nbytes = dst_nbytes;
  ch->device = device;

  out->kind = POLY_RUNNER_COPY;
  out->handle = ch;
  out->handle_size = (int)sizeof(*ch);
  out->execute = copy_execute_fn;
  out->free_handle = copy_free_fn;
  return 0;
}

/* WASM JIT EM_JS bridge (Emscripten only) */

#ifdef __EMSCRIPTEN__
#include <emscripten.h>

EM_JS(int, js_compile_wasm_kernel, (const uint8_t *bytes, int len), {
  var mod = new WebAssembly.Module(HEAPU8.subarray(bytes, bytes + len));
  var imports = {env : {memory : wasmMemory}, math : {exp2f : function(x){return Math.pow(2, x); },
      log2f: function(x) {
  return Math.log2(x); },
      sinf:  function(x) {
  return Math.sin(x); },
      powf:  function(x, y) {
  return Math.pow(x, y); }
}
}
;
var inst = new WebAssembly.Instance(mod, imports);
if (!Module._polyKernelCache) Module._polyKernelCache = [];
Module._polyKernelCache.push(inst);
return Module._polyKernelCache.length - 1;
});

EM_JS(int, js_exec_wasm_kernel, (int kernel_id, const int *args, int n_args), {
  var inst = Module._polyKernelCache[kernel_id];
  if (!inst) return -1;
  var params = [];
  for (var i = 0; i < n_args; i++) {
    params.push(HEAP32[(args >> 2) + i]);
  }
  inst.exports.kernel.apply(null, params);
  return 0;
});

EM_JS(void, js_free_wasm_kernel, (int kernel_id), {
  if (Module._polyKernelCache && kernel_id >= 0 && kernel_id < Module._polyKernelCache.length) {
    Module._polyKernelCache[kernel_id] = null;
  }
});
#endif /* __EMSCRIPTEN__ */

/* ══════════════════════════════════════════════════════════════════════ */
/*  Backend implementations (lower_item / execute / free_runner)         */
/* ══════════════════════════════════════════════════════════════════════ */

/* CPU backend */

#ifndef __EMSCRIPTEN__

static int cpu_execute_fn(void *self, void **args, int n_args);
static void cpu_free_fn(void *self);

static int cpu_lower_item(
    PolyCtx *ctx,
    PolyUOp *scheduled_root,
    const char *fn_name,
    PolyRunner *out
) {
  int n_lin;
  PolyUOp **lin = poly_linearize(ctx, scheduled_root, &n_lin);
  if (!lin) return -1;

  char *src = poly_render_c(lin, n_lin, fn_name);
  free(lin);
  if (!src) return -1;

  if (poly_dump_kernels_enabled())
    fprintf(stderr, "=== LOWER KERNEL %s ===\n%s\n=== END ===\n", fn_name, src);

  PolyProgram *prog = poly_compile_c(src, fn_name);
  if (!prog) {
    fprintf(stderr, "=== FAILED LOWER KERNEL %s ===\n%s\n=== END ===\n", fn_name, src);
    free(src);
    return -1;
  }
  free(src);

  out->kind = POLY_RUNNER_COMPILED;
  out->handle = prog;
  out->handle_size = 0;
  out->execute = cpu_execute_fn;
  out->free_handle = cpu_free_fn;
  return 0;
}

static int cpu_execute_fn(void *self, void **args, int n_args) {
  PolyRunner *runner = (PolyRunner *)self;
  poly_program_call((PolyProgram *)runner->handle, args, n_args);
  return 0;
}
static int cpu_execute(PolyRunner *runner, void **args, int n_args) {
  return cpu_execute_fn(runner, args, n_args);
}

static void cpu_free_fn(void *self) {
  PolyRunner *runner = (PolyRunner *)self;
  if (runner->handle) poly_program_destroy((PolyProgram *)runner->handle);
}
static void cpu_free_runner(PolyRunner *runner) {
  cpu_free_fn(runner);
}

static const PolyAllocator *cpu_get_allocator(void) {
  return &POLY_CPU_ALLOCATOR;
}

#endif /* !__EMSCRIPTEN__ */

static const PolyAllocator *host_get_allocator(void) {
  return &POLY_HOST_ALLOCATOR;
}

/* Interpreter backend */

static int interp_lower_item(
    PolyCtx *ctx,
    PolyUOp *scheduled_root,
    const char *fn_name,
    PolyRunner *out
) {
  (void)fn_name;
  int n_lin;
  PolyUOp **lin = poly_linearize(ctx, scheduled_root, &n_lin);
  if (!lin) return -1;

  InterpHandle *ih = malloc(sizeof(InterpHandle));
  if (!ih) {
    free(lin);
    return -1;
  }
  ih->lin = lin;
  ih->n_lin = n_lin;

  out->kind = POLY_RUNNER_INTERP;
  out->handle = ih;
  out->handle_size = 0;
  return 0;
}

static int interp_execute(PolyRunner *runner, void **args, int n_args) {
  InterpHandle *ih = (InterpHandle *)runner->handle;
  return poly_interp_eval(ih->lin, ih->n_lin, args, n_args);
}

static void interp_free_runner(PolyRunner *runner) {
  if (runner->handle) {
    InterpHandle *ih = (InterpHandle *)runner->handle;
    free(ih->lin);
    free(ih);
  }
}

static const PolyAllocator *interp_get_allocator(void) {
  return &POLY_CPU_ALLOCATOR;
}

/* WASM backend */

#ifdef __EMSCRIPTEN__

static int wasm_lower_item(
    PolyCtx *ctx,
    PolyUOp *scheduled_root,
    const char *fn_name,
    PolyRunner *out
) {
  (void)fn_name;
  int n_lin;
  PolyUOp **lin = poly_linearize_wasm_env(ctx, scheduled_root, &n_lin);
  if (!lin) return -1;

  int wasm_len = 0;
  uint8_t *wasm_bytes = poly_render_wasm(lin, n_lin, &wasm_len, false);
  free(lin);
  if (!wasm_bytes || wasm_len <= 0) return -1;

  int kernel_id = js_compile_wasm_kernel(wasm_bytes, wasm_len);
  free(wasm_bytes);
  if (kernel_id < 0) return -1;

  WasmJitHandle *wh = malloc(sizeof(WasmJitHandle));
  if (!wh) return -1;
  wh->kernel_id = kernel_id;

  out->kind = POLY_RUNNER_COMPILED;
  out->handle = wh;
  out->handle_size = 0;
  return 0;
}

static int wasm_execute(PolyRunner *runner, void **args, int n_args) {
  WasmJitHandle *wh = (WasmJitHandle *)runner->handle;
  int *iargs = malloc((size_t)n_args * sizeof(int));
  if (!iargs) return -1;
  for (int i = 0; i < n_args; i++)
    iargs[i] = (int)(intptr_t)args[i];
  int ret = js_exec_wasm_kernel(wh->kernel_id, iargs, n_args);
  free(iargs);
  return ret;
}

static void wasm_free_runner(PolyRunner *runner) {
  if (runner->handle) {
    WasmJitHandle *wh = (WasmJitHandle *)runner->handle;
    js_free_wasm_kernel(wh->kernel_id);
    free(wh);
  }
}

static const PolyAllocator *wasm_get_allocator(void) {
  return &POLY_WASM_ALLOCATOR;
}

#endif /* __EMSCRIPTEN__ */

/* CUDA backend */

#ifdef POLY_HAS_CUDA

static void cuda_extract_dims(
    PolyCtx *ctx,
    PolyUOp **lin,
    int n_lin,
    int grid[3],
    int local[3],
    PolyUOp *grid_exprs[3],
    PolyUOp *block_exprs[3],
    int *launch_bounds
) {
  grid[0] = 1;
  grid[1] = 1;
  grid[2] = 1;
  local[0] = 1;
  local[1] = 1;
  local[2] = 1;
  grid_exprs[0] = grid_exprs[1] = grid_exprs[2] = NULL;
  block_exprs[0] = block_exprs[1] = block_exprs[2] = NULL;

  for (int j = 0; j < n_lin; j++) {
    if (lin[j]->op != POLY_OP_SPECIAL || lin[j]->n_src <= 0)
      continue;
    const char *sn = lin[j]->arg.str;
    if (!sn || !sn[0]) continue;
    int slen = (int)strlen(sn);
    int dim_idx = (slen > 0) ? sn[slen - 1] - '0' : 0;
    if (dim_idx < 0 || dim_idx > 2) dim_idx = 0;
    PolyUOp *bound_expr = lin[j]->src[0];
    int bound = poly_launch_dim_upper_bound(ctx, bound_expr);
    if (sn[0] == 'l') {
      local[dim_idx] = bound;
      block_exprs[dim_idx] = bound_expr;
    } else {
      grid[dim_idx] = bound;
      grid_exprs[dim_idx] = bound_expr;
    }
  }

  *launch_bounds = local[0] * local[1] * local[2];
  if (*launch_bounds <= 0) *launch_bounds = 1;
}

static int cuda_lower_item(
    PolyCtx *ctx,
    PolyUOp *scheduled_root,
    const char *fn_name,
    PolyRunner *out
) {
  int n_lin;
  PolyUOp **lin = poly_linearize_cuda(ctx, scheduled_root, &n_lin);
  if (!lin) return -1;

  int grid[3], local[3], launch_bounds = 1;
  PolyUOp *grid_exprs[3], *block_exprs[3];
  cuda_extract_dims(ctx, lin, n_lin, grid, local, grid_exprs, block_exprs, &launch_bounds);

  char *src = poly_render_cuda(lin, n_lin, fn_name, launch_bounds);
  free(lin);
  if (!src) return -1;

  if (poly_dump_kernels_enabled())
    fprintf(stderr, "=== CUDA KERNEL %s ===\n%s\n=== END ===\n", fn_name, src);

  PolyCudaProgram *prog = poly_compile_cuda(src, fn_name);
  if (!prog) {
    fprintf(stderr, "=== FAILED CUDA KERNEL %s ===\n%s\n=== END ===\n", fn_name, src);
    free(src);
    return -1;
  }
  free(src);

  CudaRunnerHandle *ch = malloc(sizeof(CudaRunnerHandle));
  if (!ch) {
    poly_cuda_program_destroy(prog);
    return -1;
  }
  ch->prog = prog;

  out->kind = POLY_RUNNER_COMPILED;
  out->handle = ch;
  out->handle_size = 0;
  out->grid[0] = grid[0];
  out->grid[1] = grid[1];
  out->grid[2] = grid[2];
  out->block[0] = local[0];
  out->block[1] = local[1];
  out->block[2] = local[2];
  for (int i = 0; i < 3; i++) {
    out->grid_exprs[i] = grid_exprs[i];
    out->block_exprs[i] = block_exprs[i];
  }
  return 0;
}

static int cuda_execute(PolyRunner *runner, void **args, int n_args) {
  CudaRunnerHandle *ch = (CudaRunnerHandle *)runner->handle;

  /* cuLaunchKernel kernelParams: each element points TO the arg value.
   * Buffer params (0..n_params-1): args[i] is a device ptr; cast to CUdeviceptr.
   * Scalar params (n_params..): args[i] is already &int_val; use directly. */
  unsigned long long *dptrs = malloc((size_t)n_args * sizeof(unsigned long long));
  void **cuda_args = malloc((size_t)n_args * sizeof(void *));
  if (!dptrs || !cuda_args) {
    free(dptrs);
    free(cuda_args);
    return -1;
  }
  for (int i = 0; i < runner->n_params; i++) {
    dptrs[i] = (unsigned long long)(uintptr_t)args[i];
    cuda_args[i] = &dptrs[i];
  }
  for (int i = runner->n_params; i < n_args; i++) {
    cuda_args[i] = args[i];
  }

  int ret = poly_cuda_launch(
      ch->prog, cuda_args, n_args, runner->grid[0], runner->grid[1], runner->grid[2],
      runner->block[0], runner->block[1], runner->block[2]
  );
  if (ret == 0) ret = poly_cuda_sync();

  free(dptrs);
  free(cuda_args);
  return ret;
}

static void cuda_free_runner(PolyRunner *runner) {
  if (runner->handle) {
    CudaRunnerHandle *ch = (CudaRunnerHandle *)runner->handle;
    poly_cuda_program_destroy(ch->prog);
    free(ch);
  }
}

static const PolyAllocator *cuda_get_allocator(void) {
  return &POLY_CUDA_ALLOCATOR;
}

#endif /* POLY_HAS_CUDA */

/* HIP backend */

#ifdef POLY_HAS_HIP

static int hip_lower_item(
    PolyCtx *ctx,
    PolyUOp *scheduled_root,
    const char *fn_name,
    PolyRunner *out
) {
  int n_lin;
  PolyUOp **lin = poly_linearize_hip(ctx, scheduled_root, &n_lin);
  if (!lin) return -1;

  /* Extract grid/block from SPECIAL ops */
  int grid_size = 0, local_size = 0;
  for (int j = 0; j < n_lin; j++) {
    if (lin[j]->op == POLY_OP_SPECIAL && lin[j]->n_src > 0 && lin[j]->src[0]->op == POLY_OP_CONST) {
      const char *sn = lin[j]->arg.str;
      if (sn && sn[0] == 'l')
        local_size = (int)lin[j]->src[0]->arg.i;
      else
        grid_size = (int)lin[j]->src[0]->arg.i;
    }
  }

  int block_size = local_size > 0 ? local_size : 256;

  char *src = poly_render_hip(lin, n_lin, fn_name, block_size);
  free(lin);
  if (!src) return -1;

  if (poly_dump_kernels_enabled())
    fprintf(stderr, "=== HIP KERNEL %s ===\n%s\n=== END ===\n", fn_name, src);

  PolyHipProgram *prog = poly_compile_hip(src, fn_name);
  if (!prog) {
    fprintf(stderr, "=== FAILED HIP KERNEL %s ===\n%s\n=== END ===\n", fn_name, src);
    free(src);
    return -1;
  }
  free(src);

  /* Compute grid dimensions */
  int gx;
  if (local_size > 0 && grid_size > 0)
    gx = grid_size;
  else if (local_size > 0)
    gx = 1;
  else if (grid_size > 0)
    gx = (grid_size + block_size - 1) / block_size;
  else
    gx = 1;

  HipRunnerHandle *hh = malloc(sizeof(HipRunnerHandle));
  if (!hh) {
    poly_hip_program_destroy(prog);
    return -1;
  }
  hh->prog = prog;

  out->kind = POLY_RUNNER_COMPILED;
  out->handle = hh;
  out->handle_size = 0;
  out->grid[0] = gx;
  out->grid[1] = 1;
  out->grid[2] = 1;
  out->block[0] = block_size;
  out->block[1] = 1;
  out->block[2] = 1;
  return 0;
}

static int hip_execute(PolyRunner *runner, void **args, int n_args) {
  HipRunnerHandle *hh = (HipRunnerHandle *)runner->handle;

  /* hipModuleLaunchKernel kernelParams: each element points TO the arg value.
   * Buffer params (0..n_params-1): args[i] is the device ptr, so &args[i] works.
   * Scalar params (n_params..): args[i] is already &int_val, use it directly. */
  void **hip_args = malloc((size_t)n_args * sizeof(void *));
  if (!hip_args) return -1;
  for (int i = 0; i < runner->n_params; i++)
    hip_args[i] = &args[i];
  for (int i = runner->n_params; i < n_args; i++)
    hip_args[i] = args[i];

  int ret = poly_hip_launch(
      hh->prog, hip_args, n_args, runner->grid[0], runner->grid[1], runner->grid[2],
      runner->block[0], runner->block[1], runner->block[2]
  );
  if (ret == 0) ret = poly_hip_sync();

  free(hip_args);
  return ret;
}

static void hip_free_runner(PolyRunner *runner) {
  if (runner->handle) {
    HipRunnerHandle *hh = (HipRunnerHandle *)runner->handle;
    poly_hip_program_destroy(hh->prog);
    free(hh);
  }
}

static const PolyAllocator *hip_get_allocator(void) {
  return &POLY_HIP_ALLOCATOR;
}

#endif /* POLY_HAS_HIP */

/* x86-64 JIT backend */

#ifdef POLY_HAS_X64

static int x64_execute_fn(void *self, void **args, int n_args) {
  PolyRunner *runner = (PolyRunner *)self;
  poly_x64_program_call((PolyX64Program *)runner->handle, args, n_args);
  return 0;
}

static void x64_free_fn(void *self) {
  PolyRunner *runner = (PolyRunner *)self;
  if (runner->handle) poly_x64_program_destroy((PolyX64Program *)runner->handle);
}

/* Check if the kernel uses only dtypes the x64 renderer supports (f32, int32, bool).
 * Returns false if f64, f16, bf16 or other unsupported types are found.
 * Simple iterative DFS over the DAG. */
/* Check if kernel uses only features the x64 renderer handles correctly.
 * Rejects: f64/f16/bf16 dtypes, multi-range reduce patterns (DEFINE_REG with
 * nested RANGEs and AFTER chains — the renderer compiles but produces wrong code).
 * Iterative DFS with simple open-addressing pointer set. */
static bool x64_can_handle(PolyUOp *root) {
  int cap = 256, top = 0;
  PolyUOp **stack = malloc((size_t)cap * sizeof(PolyUOp *));
  if (!stack) return false;
  int set_cap = 512;
  PolyUOp **set = calloc((size_t)set_cap, sizeof(PolyUOp *));
  if (!set) {
    free(stack);
    return false;
  }
  bool ok = true;
  int n_ranges = 0, n_stores = 0;

  stack[top++] = root;
  while (top > 0) {
    PolyUOp *u = stack[--top];
    uint32_t h = (uint32_t)((uintptr_t)u >> 3) % (uint32_t)set_cap;
    bool found = false;
    for (int probe = 0; probe < set_cap; probe++) {
      uint32_t idx = (h + (uint32_t)probe) % (uint32_t)set_cap;
      if (!set[idx]) {
        set[idx] = u;
        break;
      }
      if (set[idx] == u) {
        found = true;
        break;
      }
    }
    if (found) continue;

    /* Reject unsupported dtypes: non-float32 floats (f64, f16, bf16) */
    PolyDType dt = u->dtype;
    if (!dt.is_ptr && !poly_dtype_eq(dt, POLY_VOID) && poly_dtype_is_float(dt) &&
        poly_dtype_scalar(dt).bitsize != 32) {
      ok = false;
      break;
    }
    /* Reject 64-bit integers (uint64 from THREEFRY, etc.) */
    if (!dt.is_ptr && !poly_dtype_eq(dt, POLY_VOID) && !poly_dtype_is_float(dt) &&
        poly_dtype_scalar(dt).bitsize > 32) {
      ok = false;
      break;
    }
    /* Reject unsupported ops */
    if (u->op == POLY_OP_THREEFRY) {
      ok = false;
      break;
    }
    if (u->op == POLY_OP_RANGE) n_ranges++;
    if (u->op == POLY_OP_STORE) n_stores++;

    for (int i = 0; i < u->n_src; i++) {
      if (top >= cap) {
        cap *= 2;
        stack = realloc(stack, (size_t)cap * sizeof(PolyUOp *));
      }
      stack[top++] = u->src[i];
    }
  }
  /* Previously rejected multi-store + multi-range patterns due to SHL R8
   * clobber in nested loops (fixed in commit 439f957). The renderer now
   * handles these correctly. Keeping the check commented for reference:
   * if (ok && n_stores > 1 && n_ranges > 1) ok = false; */

  free(stack);
  free(set);
  return ok;
}

static int x64_lower_item(
    PolyCtx *ctx,
    PolyUOp *scheduled_root,
    const char *fn_name,
    PolyRunner *out
) {
  /* Pre-check: fall back to CPU for unsupported patterns/dtypes */
  if (!x64_can_handle(scheduled_root)) goto fallback;

  int n_lin;
  /* Use renderer-specific x64 linearization here, matching tinygrad's
   * renderer-driven lowering contract. Falling back to CPU is still allowed
   * later if the x64 renderer cannot handle the resulting kernel. */
  PolyUOp **lin = poly_linearize_x64(ctx, scheduled_root, &n_lin);
  if (!lin) goto fallback;

  int code_size;
  uint8_t *code = poly_render_x64(lin, n_lin, &code_size);
  free(lin);
  if (!code) goto fallback;

  PolyX64Program *prog = poly_compile_x64(code, code_size);
  free(code);
  if (!prog) goto fallback;

  out->kind = POLY_RUNNER_COMPILED;
  out->handle = prog;
  out->handle_size = 0;
  out->execute = x64_execute_fn;
  out->free_handle = x64_free_fn;
  return 0;

fallback:
  /* x64 renderer doesn't support all ops yet (DEFINE_LOCAL, BARRIER, etc.).
   * Fall back to CPU compiled backend for unsupported kernels.
   * Both use host memory, so buffer layout is compatible. */
#ifndef __EMSCRIPTEN__
  return cpu_lower_item(ctx, scheduled_root, fn_name, out);
#else
  return -1;
#endif
}

static int x64_execute(PolyRunner *runner, void **args, int n_args) {
  return runner->execute(runner, args, n_args);
}

static void x64_free_runner(PolyRunner *runner) {
  if (runner->free_handle) runner->free_handle(runner);
}

#endif /* POLY_HAS_X64 */

/* ══════════════════════════════════════════════════════════════════════ */
/*  Backend registry                                                     */
/* ══════════════════════════════════════════════════════════════════════ */

static const PolyBackendDesc BACKENDS[] = {
    [POLY_DEVICE_AUTO] = {NULL, POLY_DEVICE_AUTO, false, NULL, NULL, NULL, NULL},
    [POLY_DEVICE_HOST] = {"host", POLY_DEVICE_HOST, false, NULL, NULL, NULL, host_get_allocator},
#ifndef __EMSCRIPTEN__
    [POLY_DEVICE_CPU] =
        {"cpu", POLY_DEVICE_CPU, false, cpu_lower_item, cpu_execute, cpu_free_runner,
         cpu_get_allocator},
#else
    [POLY_DEVICE_CPU] = {NULL, POLY_DEVICE_CPU, false, NULL, NULL, NULL, NULL},
#endif
    [POLY_DEVICE_INTERP] =
        {"interp", POLY_DEVICE_INTERP, false, interp_lower_item, interp_execute, interp_free_runner,
         interp_get_allocator},
#ifdef POLY_HAS_CUDA
    [POLY_DEVICE_CUDA] =
        {"cuda", POLY_DEVICE_CUDA, false, cuda_lower_item, cuda_execute, cuda_free_runner,
         cuda_get_allocator},
#else
    [POLY_DEVICE_CUDA] = {NULL, POLY_DEVICE_CUDA, false, NULL, NULL, NULL, NULL},
#endif
#ifdef __EMSCRIPTEN__
    [POLY_DEVICE_WASM] =
        {"wasm", POLY_DEVICE_WASM, true, wasm_lower_item, wasm_execute, wasm_free_runner,
         wasm_get_allocator},
#else
    [POLY_DEVICE_WASM] = {NULL, POLY_DEVICE_WASM, false, NULL, NULL, NULL, NULL},
#endif
#ifdef __EMSCRIPTEN__
    [POLY_DEVICE_WEBGPU] =
        {"webgpu", POLY_DEVICE_WEBGPU, true, poly_webgpu_lower_item, poly_webgpu_execute,
         poly_webgpu_free_runner, poly_webgpu_get_allocator},
#else
    [POLY_DEVICE_WEBGPU] = {NULL, POLY_DEVICE_WEBGPU, false, NULL, NULL, NULL, NULL},
#endif
#ifdef POLY_HAS_X64
    [POLY_DEVICE_X64_JIT] =
        {"x64_jit", POLY_DEVICE_X64_JIT, false, x64_lower_item, x64_execute, x64_free_runner,
         cpu_get_allocator},
#else
    [POLY_DEVICE_X64_JIT] = {NULL, POLY_DEVICE_X64_JIT, false, NULL, NULL, NULL, NULL},
#endif
#ifdef POLY_HAS_HIP
    [POLY_DEVICE_HIP] =
        {"hip", POLY_DEVICE_HIP, false, hip_lower_item, hip_execute, hip_free_runner,
         hip_get_allocator},
#else
    [POLY_DEVICE_HIP] = {NULL, POLY_DEVICE_HIP, false, NULL, NULL, NULL, NULL},
#endif
};

#define N_BACKENDS (sizeof(BACKENDS) / sizeof(BACKENDS[0]))

const PolyBackendDesc *poly_backend_get(PolyDevice device) {
  if (device < 0 || (size_t)device >= N_BACKENDS) return NULL;
  if (!BACKENDS[device].name) return NULL;
  return &BACKENDS[device];
}

bool poly_device_is_host_addressable(PolyDevice device) {
  const PolyBackendDesc *be = poly_backend_get(device);
  return be && be->get_allocator()->host_addressable;
}

/* Cache flush (called from napi_api.c) */

void poly_sched_cache_flush(void) {
  /* Per-context schedule caches are flushed when context is destroyed.
   * This global entry point is a no-op placeholder for the N-API layer. */
}

/* ══════════════════════════════════════════════════════════════════════ */
/*  Executable step: lower, run, free                                    */
/* ══════════════════════════════════════════════════════════════════════ */

PolyCompiledSchedule *poly_lower_schedule(PolyCtx *ctx, PolySchedule *schedule, PolyDevice device) {
  if (!ctx || !schedule) return NULL;

  const PolyBackendDesc *backend = poly_backend_get(device);
  if (!backend || !poly_device_can_execute(device) || !backend->lower_item) {
    fprintf(stderr, "polygrad: compile_schedule: unsupported device %d\n", device);
    return NULL;
  }

  /* x64 JIT uses per-runner dispatch: x64_execute delegates to runner->execute,
   * which is either x64_execute_fn (native) or cpu_execute_fn (fallback). */

  PolyCompiledSchedule *plan = calloc(1, sizeof(PolyCompiledSchedule));
  if (!plan) return NULL;
  plan->schedule = schedule;
  plan->device = device;
  plan->allocator = backend->get_allocator();
  plan->n_runners = schedule->n_items;
  plan->runners = calloc((size_t)schedule->n_items, sizeof(PolyRunner));
  if (!plan->runners) {
    free(plan);
    return NULL;
  }

  /* Lower each COMPUTE item via backend vtable */
  static int lower_counter = 0;
  for (int k = 0; k < schedule->n_items; k++) {
    PolyExecItem *item = &schedule->items[k];
    PolyRunner *runner = &plan->runners[k];
    bool lowered_as_copy = false;

    if (item->kind == POLY_EXEC_COPY) {
      if (poly_lower_copy_item(schedule, item, device, runner) == 0) lowered_as_copy = true;
    }
    if (!lowered_as_copy) {
      if (item->kind != POLY_EXEC_COMPUTE) {
        if (item->kind != POLY_EXEC_COPY) {
          fprintf(stderr, "polygrad: compile_schedule: non-COMPUTE item %d not supported\n", k);
          goto cleanup;
        }
      }

      if (!poly_validate_kernel_graph(ctx, item->root)) {
        fprintf(stderr, "polygrad: compile_schedule: kernel %d validation failed\n", k);
        goto cleanup;
      }

      char fn_name[64];
      snprintf(fn_name, sizeof(fn_name), "lower%d_k%d", lower_counter, k);

      int ret = backend->lower_item(ctx, item->root, fn_name, runner);
      if (ret != 0) {
        fprintf(
            stderr, "polygrad: compile_schedule: backend '%s' failed for kernel %d\n", backend->name,
            k
        );
        goto cleanup;
      }
    }

    /* PARAM binding order for WebGPU follows linear PARAM encounter order in
     * the WGSL renderer, not the original sr.param_to_buf order. Other
     * backends keep the existing schedule order. */
    if (lowered_as_copy) {
      runner->n_params = item->n_buf_slots;
      runner->param_to_slot = malloc((size_t)item->n_buf_slots * sizeof(int));
      if (!runner->param_to_slot) goto cleanup;
      memcpy(runner->param_to_slot, item->buf_slot_indices, (size_t)item->n_buf_slots * sizeof(int));
    } else if (device == POLY_DEVICE_WEBGPU) {
      if (webgpu_fill_param_slots(ctx, item, runner) != 0) {
        fprintf(stderr, "polygrad: compile_schedule: webgpu param remap failed for kernel %d\n", k);
        goto cleanup;
      }
    } else {
      runner->n_params = item->n_buf_slots;
      runner->param_to_slot = malloc((size_t)item->n_buf_slots * sizeof(int));
      memcpy(runner->param_to_slot, item->buf_slot_indices, (size_t)item->n_buf_slots * sizeof(int));
    }

    runner->n_vars = 0;
    runner->var_indices = NULL;
  }
  lower_counter++;

  /* Allocate persistent intermediates */
  plan->n_intermediates = 0;
  for (int i = 0; i < schedule->n_buf_slots; i++)
    if (schedule->buf_slots[i].is_intermediate) plan->n_intermediates++;

  if (plan->n_intermediates > 0) {
    plan->intermediates = calloc((size_t)plan->n_intermediates, sizeof(PolyBuffer));
    if (!plan->intermediates) goto cleanup;
    int idx = 0;
    for (int i = 0; i < schedule->n_buf_slots; i++) {
      if (!schedule->buf_slots[i].is_intermediate) continue;
      size_t nbytes = (size_t)schedule->buf_slots[i].nbytes;
      if (nbytes == 0) nbytes = sizeof(float);
      void *ptr = plan->allocator->alloc(nbytes, plan->allocator->dev_ctx);
      if (!ptr) goto cleanup;
      plan->intermediates[idx] = (PolyBuffer){
          .ptr = ptr,
          .nbytes = nbytes,
          .device = plan->device,
          .owned = true,
      };
      idx++;
    }
  }

  /* Allocate persistent per-kernel args arrays */
  plan->kernel_args = calloc((size_t)plan->n_runners, sizeof(void **));
  if (!plan->kernel_args) goto cleanup;
  for (int k = 0; k < plan->n_runners; k++) {
    PolyExecItem *item = &schedule->items[k];
    int n_args = plan->runners[k].n_params + item->n_var_uops;
    plan->kernel_args[k] = calloc((size_t)(n_args > 0 ? n_args : 1), sizeof(void *));
    if (!plan->kernel_args[k]) goto cleanup;
  }

  /* Allocate persistent slot_to_data */
  plan->n_slot_to_data = schedule->n_buf_slots;
  plan->slot_to_data =
      calloc((size_t)(plan->n_slot_to_data > 0 ? plan->n_slot_to_data : 1), sizeof(void *));
  if (!plan->slot_to_data) goto cleanup;

  /* Pre-fill intermediate slot pointers (these don't change between runs) */
  {
    int idx = 0;
    for (int i = 0; i < plan->n_slot_to_data; i++) {
      if (schedule->buf_slots[i].is_intermediate && idx < plan->n_intermediates) {
        plan->slot_to_data[i] = plan->intermediates[idx].ptr;
        idx++;
      }
    }
  }

  /* Allocate merged vars array */
  {
    int total_vars = schedule->n_default_vars + 16; /* room for runtime overrides */
    plan->merged_vars = calloc((size_t)total_vars, sizeof(PolyVarBinding));
    plan->merged_vars_cap = total_vars;
  }

  /* Allocate var int storage */
  {
    int total_var_ints = 0;
    for (int k = 0; k < schedule->n_items; k++)
      total_var_ints += schedule->items[k].n_var_uops;
    if (total_var_ints == 0) total_var_ints = 16;
    plan->var_int_storage = calloc((size_t)total_var_ints, sizeof(int));
    plan->var_int_cap = total_var_ints;
  }

  return plan;

cleanup:
  poly_compiled_schedule_free(plan);
  return NULL;
}

/* Infer the execution device from attached runtime buffers, matching the
 * graph-driven realize path. HOST buffers never force the executor. */
static PolyDevice poly_infer_schedule_device(PolyCtx *ctx, const PolySchedule *sched) {
  PolyDevice device = POLY_DEVICE_AUTO;
  for (int s = 0; sched && s < sched->n_buf_slots; s++) {
    if (sched->buf_slots[s].is_intermediate) continue;
    PolyBuffer *b = poly_buffer_get(ctx, sched->buf_slots[s].buf_uop);
    if (b && b->device != POLY_DEVICE_HOST && b->device != POLY_DEVICE_AUTO) {
      device = b->device;
      break;
    }
  }
  if (device == POLY_DEVICE_AUTO) {
    device = poly_ctx_get_preferred_device(ctx);
    if (device == POLY_DEVICE_AUTO) device = poly_device_default();
  }
  return device;
}

/* Build slot_data from ctx->buffers for the schedule-level run path. */
static int poly_fill_schedule_slot_data(
    PolyCtx *ctx, PolySchedule *sched, PolyDevice device, void **slot_data
) {
  if (!ctx || !sched || !slot_data) return -1;
  for (int s = 0; s < sched->n_buf_slots; s++) {
    if (sched->buf_slots[s].is_intermediate) continue;
    PolyUOp *buf_uop = sched->buf_slots[s].buf_uop;
    PolyBuffer *b = poly_buffer_get(ctx, buf_uop);
    if (!b) {
      fprintf(stderr, "polygrad: run_schedule: buffer slot %d has no data attached\n", s);
      return -1;
    }
    if (b->device != device && !poly_devices_share_storage(b->device, device)) {
      if (poly_buffer_allocate(ctx, buf_uop, device) != 0) {
        fprintf(stderr, "polygrad: run_schedule: buffer slot %d migration alloc failed\n", s);
        return -1;
      }
      PolyBuffer *cur = poly_buffer_get(ctx, buf_uop);
      const PolyBuffer *src = cur ? cur->src : NULL;
      if (!cur || !src || poly_buffer_copy(cur, src) != 0) {
        fprintf(stderr, "polygrad: run_schedule: buffer slot %d migration copy failed\n", s);
        return -1;
      }
      b = cur;
    } else if (!b->ptr && b->device != POLY_DEVICE_HOST) {
      fprintf(stderr, "polygrad: run_schedule: buffer slot %d has no residency pointer\n", s);
      return -1;
    }
    slot_data[s] = b->ptr;
  }
  return 0;
}

static uint32_t poly_schedule_lower_env_stamp(void) {
  const char *opt = getenv("POLY_OPTIMIZE");
  const char *devec = getenv("POLY_DEVECTORIZE");
  const char *tc_opt = getenv("POLY_TC_OPT");
  const char *use_tc = getenv("POLY_USE_TC");
  uint32_t stamp = 2166136261u;
  uint8_t bytes[] = {
      (uint8_t)(opt && opt[0] != '\0' && opt[0] != '0'),
      (uint8_t)(devec && devec[0] != '\0' ? (atoi(devec) & 0xFF) : 0),
      (uint8_t)(tc_opt && tc_opt[0] != '\0' ? (atoi(tc_opt) & 0xFF) : 0),
      (uint8_t)(use_tc && use_tc[0] != '\0' ? (atoi(use_tc) & 0xFF) : 1),
  };
  for (size_t i = 0; i < sizeof(bytes) / sizeof(bytes[0]); i++) {
    stamp ^= bytes[i];
    stamp *= 16777619u;
  }
  return stamp;
}

static void poly_zero_intermediate_buffers(PolyDevice device, PolyBuffer *bufs, int n_bufs) {
  for (int i = 0; i < n_bufs; i++) {
    PolyBuffer *h = &bufs[i];
    if (h->device == POLY_DEVICE_CPU || h->device == POLY_DEVICE_INTERP
#ifdef POLY_HAS_X64
        || h->device == POLY_DEVICE_X64_JIT
#endif
#ifdef __EMSCRIPTEN__
        || h->device == POLY_DEVICE_WASM
#endif
    ) {
      memset(h->ptr, 0, h->nbytes);
    }
#ifdef POLY_HAS_CUDA
    else if (h->device == POLY_DEVICE_CUDA) {
      poly_cuda_memset((unsigned long long)(uintptr_t)h->ptr, 0, h->nbytes);
    }
#endif
#ifdef POLY_HAS_HIP
    else if (h->device == POLY_DEVICE_HIP) {
      poly_hip_memset(h->ptr, 0, h->nbytes);
    }
#endif
#ifdef __EMSCRIPTEN__
    else if (h->device == POLY_DEVICE_WEBGPU) {
      poly_webgpu_memset_zero((uintptr_t)h->ptr, h->nbytes);
    }
#endif
  }
}

static int poly_schedule_runtime_prepare(PolyCtx *ctx, PolySchedule *sched, PolyDevice device) {
  if (!ctx || !sched) return -1;

  const PolyBackendDesc *backend = poly_backend_get(device);
  if (!backend || !poly_device_can_execute(device) || !backend->lower_item) {
    fprintf(stderr, "polygrad: run_schedule: unsupported device %d\n", device);
    return -1;
  }

  if (sched->run_device != POLY_DEVICE_AUTO && sched->run_device != device)
    poly_schedule_runtime_destroy(sched);

  if (sched->run_device == device && sched->run_slot_to_data) return 0;

  sched->run_device = device;
  sched->run_allocator = backend->get_allocator();

  sched->n_run_intermediates = 0;
  for (int i = 0; i < sched->n_buf_slots; i++)
    if (sched->buf_slots[i].is_intermediate) sched->n_run_intermediates++;

  if (sched->n_run_intermediates > 0) {
    sched->run_intermediates = calloc((size_t)sched->n_run_intermediates, sizeof(PolyBuffer));
    if (!sched->run_intermediates) goto fail;
    int idx = 0;
    for (int i = 0; i < sched->n_buf_slots; i++) {
      if (!sched->buf_slots[i].is_intermediate) continue;
      size_t nbytes = (size_t)sched->buf_slots[i].nbytes;
      if (nbytes == 0) nbytes = sizeof(float);
      void *ptr = sched->run_allocator->alloc(nbytes, sched->run_allocator->dev_ctx);
      if (!ptr) goto fail;
      sched->run_intermediates[idx] = (PolyBuffer){
          .ptr = ptr,
          .nbytes = nbytes,
          .device = device,
          .owned = true,
      };
      idx++;
    }
  }

  sched->run_kernel_args = calloc((size_t)sched->n_items, sizeof(void **));
  if (!sched->run_kernel_args) goto fail;

  sched->n_run_slot_to_data = sched->n_buf_slots;
  sched->run_slot_to_data = calloc(
      (size_t)(sched->n_run_slot_to_data > 0 ? sched->n_run_slot_to_data : 1), sizeof(void *)
  );
  if (!sched->run_slot_to_data) goto fail;

  {
    int idx = 0;
    for (int i = 0; i < sched->n_run_slot_to_data; i++) {
      if (sched->buf_slots[i].is_intermediate && idx < sched->n_run_intermediates)
        sched->run_slot_to_data[i] = sched->run_intermediates[idx++].ptr;
    }
  }

  {
    int total_vars = sched->n_default_vars + 16;
    sched->run_merged_vars = calloc((size_t)total_vars, sizeof(PolyVarBinding));
    if (!sched->run_merged_vars) goto fail;
    sched->run_merged_vars_cap = total_vars;
  }

  {
    int total_var_ints = 0;
    for (int k = 0; k < sched->n_items; k++)
      total_var_ints += sched->items[k].n_var_uops;
    if (total_var_ints == 0) total_var_ints = 16;
    sched->run_var_int_storage = calloc((size_t)total_var_ints, sizeof(int));
    if (!sched->run_var_int_storage) goto fail;
    sched->run_var_int_cap = total_var_ints;
  }

  return 0;

fail:
  poly_schedule_runtime_destroy(sched);
  return -1;
}

static int poly_schedule_merge_runtime_vars(
    PolySchedule *sched, PolyVarBinding *var_bindings, int n_var_bindings
) {
  int n_all = 0;
  int needed = sched->n_default_vars + n_var_bindings;
  if (needed > sched->run_merged_vars_cap) {
    free(sched->run_merged_vars);
    sched->run_merged_vars = calloc((size_t)needed, sizeof(PolyVarBinding));
    if (!sched->run_merged_vars) {
      sched->run_merged_vars_cap = 0;
      return -1;
    }
    sched->run_merged_vars_cap = needed;
  }

  for (int i = 0; i < sched->n_default_vars; i++)
    sched->run_merged_vars[n_all++] = sched->default_vars[i];
  for (int i = 0; i < n_var_bindings; i++) {
    bool found = false;
    for (int j = 0; j < n_all; j++) {
      if (sched->run_merged_vars[j].var == var_bindings[i].var) {
        sched->run_merged_vars[j].value = var_bindings[i].value;
        found = true;
        break;
      }
    }
    if (!found) sched->run_merged_vars[n_all++] = var_bindings[i];
  }
  return n_all;
}

int poly_exec_item_lower(PolyCtx *ctx, PolySchedule *schedule, int item_index, PolyDevice device) {
  if (!ctx || !schedule || item_index < 0 || item_index >= schedule->n_items) return -1;
  if (poly_schedule_runtime_prepare(ctx, schedule, device) != 0) return -1;

  PolyExecItem *item = &schedule->items[item_index];
  uint32_t env_stamp = poly_schedule_lower_env_stamp();
  if (item->prg_valid && item->lowered_device == device && item->lowered_env_stamp == env_stamp)
    return 0;
  if (item->prg_valid) poly_runner_cleanup(&item->prg, item->lowered_device);

  const PolyBackendDesc *backend = poly_backend_get(device);
  if (!backend || !backend->lower_item) return -1;
  bool lowered_as_copy = false;

  if (item->kind == POLY_EXEC_COPY) {
    if (poly_lower_copy_item(schedule, item, device, &item->prg) == 0) lowered_as_copy = true;
  }
  if (!lowered_as_copy) {
    if (item->kind != POLY_EXEC_COMPUTE) {
      if (item->kind != POLY_EXEC_COPY) {
        fprintf(stderr, "polygrad: exec_item_lower: non-COMPUTE item %d not supported\n", item_index);
        return -1;
      }
    }
    if (!poly_validate_kernel_graph(ctx, item->root)) {
      fprintf(stderr, "polygrad: exec_item_lower: kernel %d validation failed\n", item_index);
      return -1;
    }

    static int lower_counter = 0;
    char fn_name[64];
    snprintf(fn_name, sizeof(fn_name), "lower%d_k%d", lower_counter, item_index);
    if (backend->lower_item(ctx, item->root, fn_name, &item->prg) != 0) {
      fprintf(
          stderr, "polygrad: exec_item_lower: backend '%s' failed for kernel %d\n", backend->name,
          item_index
      );
      memset(&item->prg, 0, sizeof(item->prg));
      return -1;
    }
    lower_counter++;
  }

  if (lowered_as_copy) {
    item->prg.n_params = item->n_buf_slots;
    item->prg.param_to_slot = malloc((size_t)item->n_buf_slots * sizeof(int));
    if (!item->prg.param_to_slot) {
      poly_runner_cleanup(&item->prg, device);
      return -1;
    }
    memcpy(item->prg.param_to_slot, item->buf_slot_indices, (size_t)item->n_buf_slots * sizeof(int));
  } else if (device == POLY_DEVICE_WEBGPU) {
    if (webgpu_fill_param_slots(ctx, item, &item->prg) != 0) {
      poly_runner_cleanup(&item->prg, device);
      fprintf(stderr, "polygrad: exec_item_lower: webgpu param remap failed for kernel %d\n", item_index);
      return -1;
    }
  } else {
    item->prg.n_params = item->n_buf_slots;
    item->prg.param_to_slot = malloc((size_t)item->n_buf_slots * sizeof(int));
    if (!item->prg.param_to_slot) {
      poly_runner_cleanup(&item->prg, device);
      return -1;
    }
    memcpy(item->prg.param_to_slot, item->buf_slot_indices, (size_t)item->n_buf_slots * sizeof(int));
  }
  item->prg.n_vars = 0;
  item->prg.var_indices = NULL;

  free(schedule->run_kernel_args[item_index]);
  int n_args = item->prg.n_params + item->n_var_uops;
  schedule->run_kernel_args[item_index] =
      calloc((size_t)(n_args > 0 ? n_args : 1), sizeof(void *));
  if (!schedule->run_kernel_args[item_index]) {
    poly_runner_cleanup(&item->prg, device);
    return -1;
  }

  item->lowered_device = device;
  item->lowered_env_stamp = env_stamp;
  item->prg_valid = true;
  return 0;
}

static int poly_exec_item_run_prepared(
    PolySchedule *sched, int exec_step, int item_index, int n_all, int *var_int_idx
) {
  PolyExecItem *item = &sched->items[item_index];
  PolyRunner *runner = &item->prg;
  const PolyBackendDesc *backend = poly_backend_get(sched->run_device);
  if (!runner->handle) {
    fprintf(stderr, "polygrad: run_schedule: runner %d has no handle\n", item_index);
    return -1;
  }

  int n_vars = item->n_var_uops;
  int n_args = runner->n_params + n_vars;
  void **args = sched->run_kernel_args[item_index];

  for (int i = 0; i < runner->n_params; i++) {
    int slot = runner->param_to_slot[i];
    if (slot >= 0 && slot < sched->n_run_slot_to_data) args[i] = sched->run_slot_to_data[slot];
    if (!args[i]) {
      fprintf(
          stderr,
          "polygrad: run_schedule: missing data for param %d (slot %d) in kernel %d\n",
          i,
          slot,
          item_index
      );
      return -1;
    }
  }

  if (n_vars > 0 && item->var_uops) {
    for (int v = 0; v < n_vars; v++) {
      PolyUOp *var = item->var_uops[v];
      bool found = false;
      for (int vb = 0; vb < n_all; vb++) {
        if (sched->run_merged_vars[vb].var == var) {
          if (*var_int_idx >= sched->run_var_int_cap) {
            int new_cap = sched->run_var_int_cap * 2;
            sched->run_var_int_storage =
                realloc(sched->run_var_int_storage, (size_t)new_cap * sizeof(int));
            sched->run_var_int_cap = new_cap;
          }
          sched->run_var_int_storage[*var_int_idx] = (int)sched->run_merged_vars[vb].value;
          args[runner->n_params + v] = &sched->run_var_int_storage[*var_int_idx];
          (*var_int_idx)++;
          found = true;
          break;
        }
      }
      if (!found) {
        fprintf(stderr, "polygrad: run_schedule: no binding for DEFINE_VAR in kernel %d\n", item_index);
        return -1;
      }
    }
  }

  if (poly_resolve_runner_launch_dims(runner, sched->run_merged_vars, n_all) != 0) {
    fprintf(stderr, "polygrad: run_schedule: failed to resolve launch dims for kernel %d\n", item_index);
    return -1;
  }

  debug_dump_webgpu_schedule_args(sched, sched->run_device, exec_step, item_index, runner, args);
  int ret;
  if (runner->execute) {
    ret = runner->execute(runner, args, n_args);
  } else if (backend && backend->execute) {
    ret = backend->execute(runner, args, n_args);
  } else {
    fprintf(
        stderr,
        "polygrad: run_schedule: no execute hook for backend '%s' kernel %d\n",
        backend && backend->name ? backend->name : "<unknown>",
        item_index
    );
    return -1;
  }
  if (ret != 0) {
    fprintf(
        stderr, "polygrad: kernel %d/%d failed (params=%d grid=%d block=%d)\n", exec_step,
        sched->n_items, runner->n_params, runner->grid[0], runner->block[0]
    );
  }
  return ret;
}

int poly_exec_item_run(
    PolyCtx *ctx,
    PolySchedule *schedule,
    int item_index,
    PolyVarBinding *var_bindings,
    int n_var_bindings
) {
  if (!ctx || !schedule || item_index < 0 || item_index >= schedule->n_items) return -1;

  PolyDevice device = poly_infer_schedule_device(ctx, schedule);
  if (poly_exec_item_lower(ctx, schedule, item_index, device) != 0) return -1;
  if (poly_fill_schedule_slot_data(ctx, schedule, device, schedule->run_slot_to_data) != 0) return -1;
  poly_zero_intermediate_buffers(device, schedule->run_intermediates, schedule->n_run_intermediates);

  int n_all = poly_schedule_merge_runtime_vars(schedule, var_bindings, n_var_bindings);
  int var_int_idx = 0;
  return (n_all < 0) ? -1 : poly_exec_item_run_prepared(schedule, 0, item_index, n_all, &var_int_idx);
}

int poly_run_schedule(
    PolyCtx *ctx,
    PolySchedule *schedule,
    PolyVarBinding *var_bindings,
    int n_var_bindings
) {
  if (!ctx || !schedule) return -1;

  PolyDevice device = poly_infer_schedule_device(ctx, schedule);
  if (poly_schedule_runtime_prepare(ctx, schedule, device) != 0) {
    fprintf(stderr, "polygrad: run_schedule: compile failed\n");
    return -1;
  }
  if (poly_fill_schedule_slot_data(ctx, schedule, device, schedule->run_slot_to_data) != 0) return -1;

  poly_zero_intermediate_buffers(device, schedule->run_intermediates, schedule->n_run_intermediates);

  int n_all = poly_schedule_merge_runtime_vars(schedule, var_bindings, n_var_bindings);
  if (n_all < 0) return -1;

  int ret = 0;
  int var_int_idx = 0;
  for (int s = 0; s < schedule->n_items && ret == 0; s++) {
    int k = schedule->exec_order[s];
    if (poly_exec_item_lower(ctx, schedule, k, device) != 0) {
      ret = -1;
      break;
    }
    ret = poly_exec_item_run_prepared(schedule, s, k, n_all, &var_int_idx);
  }
  return ret;
}

int poly_run_compiled_schedule(
    PolyCompiledSchedule *plan,
    void **slot_data,
    int n_slots,
    PolyVarBinding *var_bindings,
    int n_var_bindings
) {
  if (!plan || !plan->schedule) return -1;
  PolySchedule *sched = plan->schedule;

  const PolyBackendDesc *backend = poly_backend_get(plan->device);
  if (!backend) return -1;

  int ret = 0;

  /* Zero persistent intermediates (reduce accumulators need this) */
  for (int i = 0; i < plan->n_intermediates; i++) {
    PolyBuffer *h = &plan->intermediates[i];
    if (h->device == POLY_DEVICE_CPU || h->device == POLY_DEVICE_INTERP
#ifdef POLY_HAS_X64
        || h->device == POLY_DEVICE_X64_JIT
#endif
#ifdef __EMSCRIPTEN__
        || h->device == POLY_DEVICE_WASM
#endif
    ) {
      memset(h->ptr, 0, h->nbytes);
    }
#ifdef POLY_HAS_CUDA
    else if (h->device == POLY_DEVICE_CUDA) {
      poly_cuda_memset((unsigned long long)(uintptr_t)h->ptr, 0, h->nbytes);
    }
#endif
#ifdef POLY_HAS_HIP
    else if (h->device == POLY_DEVICE_HIP) {
      poly_hip_memset(h->ptr, 0, h->nbytes);
    }
#endif
#ifdef __EMSCRIPTEN__
    else if (h->device == POLY_DEVICE_WEBGPU) {
      poly_webgpu_memset_zero((uintptr_t)h->ptr, h->nbytes);
    }
#endif
  }

  /* Fill external slots in persistent slot_to_data */
  for (int i = 0; i < plan->n_slot_to_data; i++) {
    if (!sched->buf_slots[i].is_intermediate) {
      plan->slot_to_data[i] = (i < n_slots && slot_data[i]) ? slot_data[i] : NULL;
    }
    /* intermediate slots are pre-filled at compile time and don't change */
  }

  /* Merge default vars with runtime overrides */
  int n_all = 0;

  /* Grow merged_vars if needed */
  int needed = sched->n_default_vars + n_var_bindings;
  if (needed > plan->merged_vars_cap) {
    free(plan->merged_vars);
    plan->merged_vars = calloc((size_t)needed, sizeof(PolyVarBinding));
    plan->merged_vars_cap = needed;
  }

  for (int i = 0; i < sched->n_default_vars; i++)
    plan->merged_vars[n_all++] = sched->default_vars[i];
  for (int i = 0; i < n_var_bindings; i++) {
    bool found = false;
    for (int j = 0; j < n_all; j++) {
      if (plan->merged_vars[j].var == var_bindings[i].var) {
        plan->merged_vars[j].value = var_bindings[i].value;
        found = true;
        break;
      }
    }
    if (!found) plan->merged_vars[n_all++] = var_bindings[i];
  }

  /* Execute runners in exec_order via backend vtable */
  int var_int_idx = 0;
  for (int s = 0; s < sched->n_items && ret == 0; s++) {
    int k = sched->exec_order[s];
    PolyRunner *runner = &plan->runners[k];

    if (!runner->handle) {
      fprintf(stderr, "polygrad: plan_run: runner %d has no handle\n", k);
      ret = -1;
      break;
    }

    PolyExecItem *item = &sched->items[k];
    int n_vars = item->n_var_uops;
    int n_args = runner->n_params + n_vars;
    void **args = plan->kernel_args[k];

    for (int i = 0; i < runner->n_params; i++) {
      int slot = runner->param_to_slot[i];
      if (slot >= 0 && slot < plan->n_slot_to_data) args[i] = plan->slot_to_data[slot];
      if (!args[i]) {
        fprintf(
            stderr,
            "polygrad: plan_run: missing data for param %d "
            "(slot %d) in kernel %d\n",
            i, slot, k
        );
        ret = -1;
        break;
      }
    }

    /* Fill var params (DEFINE_VAR values as int* pointers) */
    if (ret == 0 && n_vars > 0 && item->var_uops) {
      for (int v = 0; v < n_vars; v++) {
        PolyUOp *var = item->var_uops[v];
        bool found = false;
        for (int vb = 0; vb < n_all; vb++) {
          if (plan->merged_vars[vb].var == var) {
            if (var_int_idx >= plan->var_int_cap) {
              /* grow var int storage */
              int new_cap = plan->var_int_cap * 2;
              plan->var_int_storage = realloc(plan->var_int_storage, (size_t)new_cap * sizeof(int));
              plan->var_int_cap = new_cap;
            }
            plan->var_int_storage[var_int_idx] = (int)plan->merged_vars[vb].value;
            args[runner->n_params + v] = &plan->var_int_storage[var_int_idx];
            var_int_idx++;
            found = true;
            break;
          }
        }
        if (!found) {
          fprintf(
              stderr,
              "polygrad: plan_run: no binding for DEFINE_VAR "
              "in kernel %d\n",
              k
          );
          ret = -1;
          break;
        }
	    }
	  }

    if (ret == 0) {
      if (poly_resolve_runner_launch_dims(runner, plan->merged_vars, n_all) != 0) {
        fprintf(stderr, "polygrad: plan_run: failed to resolve launch dims for kernel %d\n", k);
        ret = -1;
        break;
      }
      debug_dump_webgpu_runner_args(plan, s, k, runner, args);
      if (runner->execute) ret = runner->execute(runner, args, n_args);
      else ret = backend->execute(runner, args, n_args);
      if (ret != 0) {
        fprintf(
            stderr, "polygrad: kernel %d/%d failed (params=%d grid=%d block=%d)\n", s,
            sched->n_items, runner->n_params, runner->grid[0], runner->block[0]
        );
      }
    }
  }

  return ret;
}

void poly_compiled_schedule_free(PolyCompiledSchedule *plan) {
  if (!plan) return;

  const PolyBackendDesc *backend = poly_backend_get(plan->device);

  for (int i = 0; i < plan->n_runners; i++) {
    PolyRunner *r = &plan->runners[i];
    if (r->handle) {
      if (r->free_handle) r->free_handle(r);
      else if (backend) backend->free_runner(r);
    }
    free(r->param_to_slot);
    free(r->var_indices);
  }
  free(plan->runners);

  /* Free persistent intermediates */
  if (plan->intermediates) {
    for (int i = 0; i < plan->n_intermediates; i++) {
      if (plan->intermediates[i].ptr)
        plan->allocator->free(&plan->intermediates[i], plan->allocator->dev_ctx);
    }
    free(plan->intermediates);
  }

  /* Free persistent per-kernel args arrays */
  if (plan->kernel_args) {
    for (int i = 0; i < plan->n_runners; i++)
      free(plan->kernel_args[i]);
    free(plan->kernel_args);
  }

  free(plan->slot_to_data);
  free(plan->merged_vars);
  free(plan->var_int_storage);
  free(plan);
}
