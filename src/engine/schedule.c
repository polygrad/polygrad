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
#include "ctx.h"
#include "utils.h"
#include "frontend_internal.h"
#include "codegen.h"
#include "interp.h"
#include "schedule/rangeify.h"
#include "runtime_wasm.h"
#include "runtime_webgpu.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>

static void stable_kernel_fn_name(char *out, size_t cap, PolyDevice device, PolyUOp *root) {
  uint32_t h = poly_structural_hash(root);
  /* The native CPU compiler hashes the full rendered source for its disk cache.
   * Counter-based function names make identical kernels render different source,
   * defeating tinygrad-style compile caching across eager realize calls. */
  snprintf(out, cap, "poly_k_%d_%08x", (int)device, h);
}

typedef struct {
  PolyUOp *root;
  PolyDevice device;
  uint32_t env_stamp;
  PolyRunner runner;
} PolyProgramCacheEntry;

static uint32_t poly_schedule_lower_env_stamp(void);

static const char *poly_exec_item_kind_name(PolyExecItemKind kind) {
  switch (kind) {
  case POLY_EXEC_COMPUTE:
    return "compute";
  case POLY_EXEC_COPY:
    return "copy";
  case POLY_EXEC_VIEW:
    return "view";
  case POLY_EXEC_ENCDEC:
    return "encdec";
  default:
    return "?";
  }
}

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
    PolyUOp *var,
    const PolyVarBinding *bindings,
    int n_bindings,
    PolyArg *out
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
    PolyUOp *u,
    const PolyVarBinding *bindings,
    int n_bindings,
    PolyArg *out
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
    PolyRunner *runner,
    const PolyVarBinding *bindings,
    int n_bindings
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
      stderr, "[webgpu-run] step=%d kernel=%d n_params=%d n_vars=%d\n", exec_step, kernel_idx,
      runner->n_params, runner->n_vars
  );
  for (int i = 0; i < runner->n_params; i++) {
    int slot = runner->param_to_slot ? runner->param_to_slot[i] : -1;
    const PolyScheduleBufSlot *bs =
        (slot >= 0 && slot < sched->n_buf_slots) ? &sched->buf_slots[slot] : NULL;
    fprintf(
        stderr,
        "  [param %d] slot=%d handle=%p buf_uop=%p interm=%d ext=%d nbytes=%lld numel=%lld\n", i,
        slot, args[i], bs ? (void *)bs->buf_uop : NULL, bs ? (int)bs->is_intermediate : -1,
        bs ? bs->external_buf_idx : -1, bs ? (long long)bs->nbytes : -1LL,
        bs ? (long long)bs->numel : -1LL
    );
  }
  for (int i = 0; i < runner->n_vars; i++) {
    void *var_ptr = args[runner->n_params + i];
    fprintf(stderr, "  [var %d] ptr=%p val=%d\n", i, var_ptr, var_ptr ? *(int *)var_ptr : 0);
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
      stderr, "[webgpu-run] step=%d kernel=%d n_params=%d n_vars=%d\n", exec_step, kernel_idx,
      runner->n_params, runner->n_vars
  );
  for (int i = 0; i < runner->n_params; i++) {
    int slot = runner->param_to_slot ? runner->param_to_slot[i] : -1;
    const PolyScheduleBufSlot *bs =
        (slot >= 0 && slot < sched->n_buf_slots) ? &sched->buf_slots[slot] : NULL;
    fprintf(
        stderr,
        "  [param %d] slot=%d handle=%p buf_uop=%p interm=%d ext=%d nbytes=%lld numel=%lld\n", i,
        slot, args[i], bs ? (void *)bs->buf_uop : NULL, bs ? (int)bs->is_intermediate : -1,
        bs ? bs->external_buf_idx : -1, bs ? (long long)bs->nbytes : -1LL,
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

static PolyDevice poly_schedule_slot_current_device(
    PolyCtx *ctx,
    const PolySchedule *sched,
    int slot_idx
) {
  if (!sched || slot_idx < 0 || slot_idx >= sched->n_buf_slots) return POLY_DEVICE_AUTO;
  PolyDevice dev = sched->buf_slots[slot_idx].device;
  if (dev != POLY_DEVICE_AUTO) return dev;
  if (ctx && !sched->buf_slots[slot_idx].is_intermediate) {
    PolyBuffer *b = poly_buffer_get(ctx, sched->buf_slots[slot_idx].buf_uop);
    if (b && b->device != POLY_DEVICE_AUTO) return b->device;
  }
  return POLY_DEVICE_AUTO;
}

static bool poly_schedule_slot_used_as_copy_source(const PolySchedule *sched, int slot_idx) {
  if (!sched) return false;
  for (int k = 0; k < sched->n_items; k++) {
    const PolyExecItem *item = &sched->items[k];
    if (item->kind != POLY_EXEC_COPY || item->n_buf_slots < 2) continue;
    if (item->buf_slot_indices[1] == slot_idx) return true;
  }
  return false;
}

static bool poly_schedule_slot_used_by_compute(const PolySchedule *sched, int slot_idx) {
  if (!sched) return false;
  for (int k = 0; k < sched->n_items; k++) {
    const PolyExecItem *item = &sched->items[k];
    if (item->kind != POLY_EXEC_COMPUTE) continue;
    for (int i = 0; i < item->n_buf_slots; i++)
      if (item->buf_slot_indices[i] == slot_idx) return true;
  }
  return false;
}

static PolyDevice poly_schedule_slot_target_device(
    PolyCtx *ctx,
    const PolySchedule *sched,
    int slot_idx,
    PolyDevice fallback
) {
  PolyDevice dev = poly_schedule_slot_current_device(ctx, sched, slot_idx);
  bool copy_source = poly_schedule_slot_used_as_copy_source(sched, slot_idx);
  bool compute_use = poly_schedule_slot_used_by_compute(sched, slot_idx);

  if (dev == POLY_DEVICE_AUTO) dev = fallback;
  if (dev == POLY_DEVICE_AUTO) dev = poly_device_default();

  /* HOST is valid as an explicit COPY source. It is not a kernel execution
   * domain, so a compute use must be migrated to the fallback executable
   * device unless that backend shares host-addressable storage. */
  if (dev == POLY_DEVICE_HOST && (!copy_source || compute_use)) {
    PolyDevice run = (fallback == POLY_DEVICE_AUTO) ? poly_device_default() : fallback;
    if (poly_device_can_execute(run) && !poly_device_is_host_addressable(run)) return run;
  }
  return dev;
}

static PolyDevice poly_exec_item_device(
    PolyCtx *ctx,
    const PolySchedule *sched,
    const PolyExecItem *item,
    PolyDevice fallback
) {
  if (!sched || !item) return fallback == POLY_DEVICE_AUTO ? poly_device_default() : fallback;
  for (int i = 0; i < item->n_buf_slots; i++) {
    int slot = item->buf_slot_indices[i];
    PolyDevice dev = poly_schedule_slot_target_device(ctx, sched, slot, fallback);
    if (dev != POLY_DEVICE_AUTO && dev != POLY_DEVICE_HOST && poly_device_can_execute(dev))
      return dev;
  }
  return fallback == POLY_DEVICE_AUTO ? poly_device_default() : fallback;
}

static bool collect_external_buf_order_from_kernel_graph(
    PolyUOp *kernel_graph,
    PolyUOp ***out_buf_order,
    int *out_n_external
) {
  if (!kernel_graph || !out_buf_order || !out_n_external) return false;
  *out_buf_order = NULL;
  *out_n_external = 0;

  PolyUOp **all_bufs = NULL;
  int n_all = 0, n_visited = 0;
  if (!poly_collect_buf_order_alloc(kernel_graph, &all_bufs, &n_all, &n_visited)) return false;

  int n_external = 0;
  for (int i = 0; i < n_all; i++) {
    if (!is_intermediate_buffer_uop(all_bufs[i])) all_bufs[n_external++] = all_bufs[i];
  }
  *out_buf_order = all_bufs;
  *out_n_external = n_external;
  return true;
}

static bool uop_ptr_in_list(PolyUOp *u, PolyUOp **list, int n) {
  for (int i = 0; i < n; i++)
    if (list[i] == u) return true;
  return false;
}

static int collect_define_vars(PolyCtx *ctx, PolyUOp *root, PolyUOp **out, int cap) {
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, root, &n_topo);
  if (!topo) return 0;

  int n = 0;
  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    if (!u || u->op != POLY_OP_DEFINE_VAR || uop_ptr_in_list(u, out, n)) continue;
    if (n >= cap) {
      fprintf(stderr, "polygrad: too many DEFINE_VARs in schedule (cap=%d)\n", cap);
      return -1;
    }
    out[n++] = u;
  }
  return n;
}

static int find_var_binding(const PolyVarBinding *bindings, int n, PolyUOp *var) {
  for (int i = 0; i < n; i++)
    if (bindings[i].var == var) return i;
  return -1;
}

static int collect_bind_defaults(
    PolyCtx *ctx,
    PolyUOp *root,
    PolyUOp **used_vars,
    int n_used_vars,
    PolyVarBinding *out,
    int cap
) {
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, root, &n_topo);
  if (!topo) return 0;

  int n = 0;
  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    if (!u || u->op != POLY_OP_BIND || u->n_src < 2) continue;
    PolyUOp *var = u->src[0];
    PolyUOp *val = u->src[1];
    if (!var || var->op != POLY_OP_DEFINE_VAR || !val || val->op != POLY_OP_CONST) continue;
    if (!uop_ptr_in_list(var, used_vars, n_used_vars)) continue;
    if (val->arg.kind != POLY_ARG_INT) {
      fprintf(stderr, "polygrad: BIND default for DEFINE_VAR must be integer CONST\n");
      return -1;
    }

    int32_t value = (int32_t)val->arg.i;
    int existing = find_var_binding(out, n, var);
    if (existing >= 0) {
      if (out[existing].value != value) {
        fprintf(
            stderr, "polygrad: BIND mismatch for DEFINE_VAR: %d != %d\n", out[existing].value, value
        );
        return -1;
      }
      continue;
    }
    if (n >= cap) {
      fprintf(stderr, "polygrad: too many BIND defaults (cap=%d)\n", cap);
      return -1;
    }
    out[n++] = (PolyVarBinding){.var = var, .value = value};
  }
  return n;
}

static int schedule_slot_for_buf(
    PolyUOp *buf,
    PolyUOp **buf_order_slots,
    int n_buf_order_slots,
    PolyMap *inter_set
) {
  if (!buf || buf->op != POLY_OP_BUFFER) return -1;
  int pos = poly_find_buf_position(buf, buf_order_slots, n_buf_order_slots);
  if (pos >= 0) return pos;
  if (inter_set) {
    PolyUOp *v = poly_map_get(inter_set, poly_ptr_hash(buf), buf, poly_ptr_eq);
    if (v) return (int)((intptr_t)v - 1);
  }
  return -1;
}

static int schedule_slot_for_kernel_param(
    const PolyKernelScheduleResult *sr,
    int kernel_idx,
    int param_idx,
    PolyUOp **buf_order_slots,
    int n_buf_order_slots,
    PolyMap *inter_set
) {
  if (!sr || kernel_idx < 0 || kernel_idx >= sr->n_kernels) return -1;
  if (!sr->param_to_buf || !sr->kernel_n_params || !sr->param_to_buf[kernel_idx]) return -1;
  if (param_idx < 0 || param_idx >= sr->kernel_n_params[kernel_idx]) return -1;
  return schedule_slot_for_buf(
      sr->param_to_buf[kernel_idx][param_idx], buf_order_slots, n_buf_order_slots, inter_set
  );
}

static bool linear_param_is_external(PolyUOp *u, int *out_idx) {
  if (!u || u->op != POLY_OP_PARAM || u->arg.kind != POLY_ARG_INT) return false;
  if (u->arg.i < 0 || u->arg.i > INT_MAX) return false;
  if (out_idx) *out_idx = (int)u->arg.i;
  return true;
}

static int linear_intermediate_slot(
    PolyUOp *buf,
    PolyUOp **intermediate_bufs,
    int n_intermediates,
    int n_external
) {
  for (int i = 0; i < n_intermediates; i++)
    if (intermediate_bufs[i] == buf) return n_external + i;
  return -1;
}

static int linear_param_to_slot(
    PolyUOp *param,
    PolyUOp **intermediate_bufs,
    int n_intermediates,
    int n_external
) {
  int external_idx = -1;
  if (linear_param_is_external(param, &external_idx))
    return (external_idx >= 0 && external_idx < n_external) ? external_idx : -1;
  if (param && param->op == POLY_OP_BUFFER)
    return linear_intermediate_slot(param, intermediate_bufs, n_intermediates, n_external);
  return -1;
}

static PolyUOp *linear_call_param_for_buf(PolyCtx *ctx, PolyUOp *buf, int external_slot) {
  if (external_slot >= 0)
    return poly_uop0(ctx, POLY_OP_PARAM, POLY_VOID, poly_arg_int(external_slot));
  return buf;
}

static bool uop_tree_contains_reduce(PolyCtx *ctx, PolyUOp *root) {
  int n = 0;
  PolyUOp **topo = poly_toposort(ctx, root, &n);
  for (int i = 0; i < n; i++)
    if (topo[i]->op == POLY_OP_REDUCE || topo[i]->op == POLY_OP_REDUCE_AXIS) return true;
  return false;
}

static bool linear_intermediate_needs_zero(PolyCtx *ctx, PolyUOp *linear, PolyUOp *buf) {
  if (!linear || !buf) return false;
  for (int k = 0; k < linear->n_src; k++) {
    PolyUOp *call = linear->src[k];
    if (!call || call->op != POLY_OP_CALL || call->n_src < 2) continue;
    bool uses_buf = false;
    for (int i = 1; i < call->n_src; i++) {
      if (call->src[i] == buf) {
        uses_buf = true;
        break;
      }
    }
    if (uses_buf && uop_tree_contains_reduce(ctx, call->src[0])) return true;
  }
  return false;
}

static PolySchedule *build_schedule_from_linear(
    PolyCtx *ctx,
    PolyUOp *linear,
    PolyCompileMode mode,
    uint32_t graph_hash,
    PolyUOp **buf_order_orig,
    int n_bufs_orig,
    const PolyVarBinding *default_vars,
    int n_default_vars
) {
  if (!linear || linear->op != POLY_OP_LINEAR) return NULL;

  PolyUOp **intermediate_bufs = NULL;
  int n_intermediates = 0, inter_cap = 0;

  for (int k = 0; k < linear->n_src; k++) {
    PolyUOp *call = linear->src[k];
    if (!call || call->op != POLY_OP_CALL || call->n_src < 1) goto fail_pre;
    for (int i = 1; i < call->n_src; i++) {
      PolyUOp *p = call->src[i];
      if (!p || p->op != POLY_OP_BUFFER) continue;
      if (linear_intermediate_slot(p, intermediate_bufs, n_intermediates, 0) >= 0) continue;
      if (n_intermediates >= inter_cap) {
        inter_cap = inter_cap ? inter_cap * 2 : 4;
        PolyUOp **tmp = realloc(intermediate_bufs, (size_t)inter_cap * sizeof(PolyUOp *));
        if (!tmp) goto fail_pre;
        intermediate_bufs = tmp;
      }
      intermediate_bufs[n_intermediates++] = p;
    }
  }

  PolySchedule *ps = calloc(1, sizeof(PolySchedule));
  if (!ps) goto fail_pre;
  ps->mode = mode;
  ps->graph_hash = graph_hash;
  ps->loss_buf_slot = -1;

  ps->n_default_vars = n_default_vars;
  if (n_default_vars > 0) {
    ps->default_vars = malloc((size_t)n_default_vars * sizeof(PolyVarBinding));
    if (!ps->default_vars) goto fail;
    memcpy(ps->default_vars, default_vars, (size_t)n_default_vars * sizeof(PolyVarBinding));
  }

  ps->n_buf_slots = n_bufs_orig + n_intermediates;
  if (ps->n_buf_slots > 0) {
    ps->buf_slots = calloc((size_t)ps->n_buf_slots, sizeof(PolyScheduleBufSlot));
    if (!ps->buf_slots) goto fail;
    for (int i = 0; i < n_bufs_orig; i++) {
      PolyScheduleBufSlot *slot = &ps->buf_slots[i];
      PolyUOp *buf = buf_order_orig[i];
      slot->buf_uop = buf;
      slot->is_intermediate = false;
      slot->external_buf_idx = i;
      slot->dtype = buf ? poly_dtype_scalar(buf->dtype) : POLY_FLOAT32;
      slot->numel = (buf && buf->arg.kind == POLY_ARG_INT) ? buf->arg.i : 0;
      slot->device = poly_uop_device(buf);
      if (slot->numel > 0) slot->nbytes = slot->numel * poly_dtype_itemsize(slot->dtype);
    }
    for (int i = 0; i < n_intermediates; i++) {
      PolyScheduleBufSlot *slot = &ps->buf_slots[n_bufs_orig + i];
      PolyUOp *buf = intermediate_bufs[i];
      slot->buf_uop = buf;
      slot->is_intermediate = true;
      slot->external_buf_idx = -1;
      slot->dtype = buf ? poly_dtype_scalar(buf->dtype) : POLY_FLOAT32;
      slot->device = poly_uop_device(buf);
      slot->numel = (buf && buf->arg.kind == POLY_ARG_INT) ? buf->arg.i : 0;
      if (slot->numel > 0) slot->nbytes = slot->numel * poly_dtype_itemsize(slot->dtype);
      slot->needs_zero = linear_intermediate_needs_zero(ctx, linear, buf);
    }
  }

  ps->n_items = linear->n_src;
  ps->items = calloc((size_t)ps->n_items, sizeof(PolyExecItem));
  ps->exec_order = malloc((size_t)ps->n_items * sizeof(int));
  if (!ps->items || !ps->exec_order) goto fail;

  for (int k = 0; k < ps->n_items; k++) {
    PolyUOp *call = linear->src[k];
    PolyExecItem *item = &ps->items[k];
    item->root = call->src[0];
    item->kind = (call->arg.kind == POLY_ARG_INT && call->arg.i == POLY_EXEC_COPY)
                     ? POLY_EXEC_COPY
                     : POLY_EXEC_COMPUTE;
    ps->exec_order[k] = k;

    int n_params = 0, n_vars = 0;
    for (int i = 1; i < call->n_src; i++) {
      if (call->src[i] && call->src[i]->op == POLY_OP_DEFINE_VAR)
        n_vars++;
      else
        n_params++;
    }

    item->n_buf_slots = n_params;
    if (n_params > 0) {
      item->buf_slot_indices = malloc((size_t)n_params * sizeof(int));
      if (!item->buf_slot_indices) goto fail;
    }
    item->n_var_uops = n_vars;
    if (n_vars > 0) {
      item->var_uops = malloc((size_t)n_vars * sizeof(PolyUOp *));
      if (!item->var_uops) goto fail;
    }

    int pidx = 0, vidx = 0;
    for (int i = 1; i < call->n_src; i++) {
      PolyUOp *p = call->src[i];
      if (p && p->op == POLY_OP_DEFINE_VAR) {
        item->var_uops[vidx++] = p;
      } else {
        int slot = linear_param_to_slot(p, intermediate_bufs, n_intermediates, n_bufs_orig);
        if (slot < 0) {
          fprintf(stderr, "polygrad: build_schedule_from_linear: unresolved param in item %d\n", k);
          goto fail;
        }
        item->buf_slot_indices[pidx++] = slot;
      }
    }
  }

  free(intermediate_bufs);
  return ps;

fail:
  free(intermediate_bufs);
  poly_schedule_free(ps);
  return NULL;
fail_pre:
  free(intermediate_bufs);
  return NULL;
}

static PolySchedule *build_schedule_from_kernel_graph(
    PolyCtx *ctx,
    PolyUOp *kernel_graph,
    PolyCompileMode mode,
    uint32_t graph_hash,
    PolyUOp **buf_order_orig,
    int n_bufs_orig,
    const PolyVarBinding *default_vars,
    int n_default_vars,
    PolyUOp *linear_template
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
      slot->device = poly_uop_device(buf);
      if (slot->numel > 0) slot->nbytes = slot->numel * poly_dtype_itemsize(slot->dtype);
    }

    for (int b = 0; b < n_intermediate; b++) {
      PolyScheduleBufSlot *slot = &ps->buf_slots[n_external + b];
      slot->is_intermediate = true;
      slot->external_buf_idx = -1;
      if (sr.intermediate_buf_uops && sr.intermediate_buf_uops[b]) {
        slot->buf_uop = sr.intermediate_buf_uops[b];
        slot->dtype = poly_dtype_scalar(sr.intermediate_buf_uops[b]->dtype);
        slot->device = poly_uop_device(sr.intermediate_buf_uops[b]);
      } else {
        slot->dtype = POLY_FLOAT32;
        slot->device = POLY_DEVICE_AUTO;
      }
      slot->numel = sr.intermediate_sizes ? sr.intermediate_sizes[b] : 0;
      int itemsize = (sr.intermediate_itemsizes && sr.intermediate_itemsizes[b] > 0)
                         ? sr.intermediate_itemsizes[b]
                         : (int)sizeof(float);
      slot->nbytes = slot->numel * itemsize;
      slot->needs_zero = sr.intermediate_needs_zero ? sr.intermediate_needs_zero[b] : true;
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
    PolyUOp *cached_root = NULL;
    if (linear_template && linear_template->op == POLY_OP_LINEAR &&
        linear_template->n_src == sr.n_kernels) {
      int linear_idx = k;
      if (sr.exec_order) {
        linear_idx = -1;
        for (int step = 0; step < sr.n_kernels; step++) {
          if (sr.exec_order[step] == k) {
            linear_idx = step;
            break;
          }
        }
      }
      if (linear_idx >= 0 && linear_idx < linear_template->n_src)
        cached_root = linear_template->src[linear_idx];
      if (cached_root && cached_root->op == POLY_OP_CALL && cached_root->n_src >= 1)
        cached_root = cached_root->src[0];
    }
    item->root = cached_root ? cached_root : sr.kernels[k];
    bool is_copy = sr.kernel_kinds && sr.kernel_kinds[k] == POLY_KERNEL_ITEM_COPY;
    item->kind = is_copy ? POLY_EXEC_COPY : POLY_EXEC_COMPUTE;

    if (is_copy) {
      item->n_buf_slots = 2;
      item->buf_slot_indices = malloc(2 * sizeof(int));
      if (!item->buf_slot_indices) goto cleanup;
      item->buf_slot_indices[0] = schedule_slot_for_kernel_param(
          &sr, k, sr.copy_dst_params ? sr.copy_dst_params[k] : -1, buf_order_orig, n_bufs_orig,
          inter_set
      );
      item->buf_slot_indices[1] = schedule_slot_for_kernel_param(
          &sr, k, sr.copy_src_params ? sr.copy_src_params[k] : -1, buf_order_orig, n_bufs_orig,
          inter_set
      );

      if (item->buf_slot_indices[0] < 0 || item->buf_slot_indices[1] < 0) {
        fprintf(
            stderr,
            "polygrad: build_schedule_from_kernel_graph: unresolved COPY params in kernel %d\n", k
        );
        if (inter_set) poly_map_destroy(inter_set);
        goto cleanup;
      }
    } else {
      int np = sr.kernel_n_params[k];
      item->n_buf_slots = np;
      item->buf_slot_indices = malloc((size_t)np * sizeof(int));

      for (int i = 0; i < np; i++) {
        item->buf_slot_indices[i] =
            schedule_slot_for_kernel_param(&sr, k, i, buf_order_orig, n_bufs_orig, inter_set);

        if (item->buf_slot_indices[i] < 0) {
          fprintf(
              stderr,
              "polygrad: build_schedule_from_kernel_graph: unresolved param %d in kernel %d\n", i, k
          );
          if (inter_set) poly_map_destroy(inter_set);
          goto cleanup;
        }
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

static PolyUOp *poly_lower_kernel_graph_to_linear(PolyCtx *ctx, PolyUOp *kernel_graph);

static bool poly_schedule_cache_enabled(void) {
  const char *v = getenv("POLY_SCACHE");
  return !v || v[0] != '0';
}

PolySchedule *poly_complete_create_schedule_with_vars(
    PolyCtx *ctx,
    PolyUOp *sink,
    PolyCompileMode mode
) {
  if (!sink || sink->op != POLY_OP_SINK) {
    fprintf(stderr, "polygrad: complete_create_schedule_with_vars: expected SINK\n");
    return NULL;
  }

  bool timing = poly_debug_at_least(2);
  double t0 = timing ? poly_now_ms() : 0.0;
  if (timing) {
    fprintf(stderr, "[polygrad:create_schedule] begin sink=%p mode=%d\n", (void *)sink, (int)mode);
    fflush(stderr);
  }

  /* --- Collect pre-strip buffer ordering -------------------------------- */
  PolyUOp **buf_order_orig = NULL;
  int n_bufs_orig = 0, n_dfs = 0;
  if (!poly_collect_buf_order_alloc(sink, &buf_order_orig, &n_bufs_orig, &n_dfs)) return NULL;
  double t_bufs = timing ? poly_now_ms() : 0.0;

  PolyUOp *used_vars[64];
  int n_used_vars = collect_define_vars(ctx, sink, used_vars, 64);
  if (n_used_vars < 0) {
    free(buf_order_orig);
    return NULL;
  }

  PolyVarBinding bind_vals[64];
  int n_bind_vals = collect_bind_defaults(ctx, sink, used_vars, n_used_vars, bind_vals, 64);
  if (n_bind_vals < 0) {
    free(buf_order_orig);
    return NULL;
  }
  double t_vars = timing ? poly_now_ms() : 0.0;
  if (timing) {
    fprintf(
        stderr,
        "[polygrad:create_schedule] prepass bufs=%d dfs=%d vars=%d binds=%d collect=%.3fms "
        "vars=%.3fms\n",
        n_bufs_orig, n_dfs, n_used_vars, n_bind_vals, t_bufs - t0, t_vars - t_bufs
    );
    fflush(stderr);
  }

  if (poly_schedule_cache_enabled()) {
    /* Tinygrad caches LINEAR templates, but it does not execute a cached
     * LINEAR directly. It resolves the cached CALL roots against the current
     * call's buffers before running. In Polygrad that current-call resolution
     * is build_schedule_from_kernel_graph(..., linear_template): the current
     * kernel graph supplies slots/intermediates/order, while the cached LINEAR
     * can donate already-lowered kernel roots when the schedule shape matches.
     */
    double t_lower0 = timing ? poly_now_ms() : 0.0;
    PolyUOp *kernel_graph = poly_get_kernel_graph(ctx, sink);
    if (!kernel_graph) {
      free(buf_order_orig);
      return NULL;
    }
    PolyUOp *linear_template = poly_lower_sink_to_linear(ctx, sink, mode);
    double t_lower1 = timing ? poly_now_ms() : 0.0;
    if (!linear_template) {
      free(buf_order_orig);
      return NULL;
    }
    uint32_t ghash = poly_structural_hash(kernel_graph) ^ (POLY_SCHED_CACHE_VERSION * 2654435761u);
    double t_build0 = timing ? poly_now_ms() : 0.0;
    PolySchedule *ps = build_schedule_from_kernel_graph(
        ctx, kernel_graph, mode, ghash, buf_order_orig, n_bufs_orig, bind_vals, n_bind_vals,
        linear_template
    );
    double t_build1 = timing ? poly_now_ms() : 0.0;
    if (timing) {
      fprintf(
          stderr,
          "[polygrad:create_schedule] done path=cached_linear_current_graph lower=%.3fms "
          "build=%.3fms total=%.3fms items=%d slots=%d\n",
          t_lower1 - t_lower0, t_build1 - t_build0, t_build1 - t0, ps ? ps->n_items : -1,
          ps ? ps->n_buf_slots : -1
      );
      fflush(stderr);
    }
    free(buf_order_orig);
    return ps;
  }

  double t_kernel0 = timing ? poly_now_ms() : 0.0;
  PolyUOp *kernel_graph = poly_get_kernel_graph(ctx, sink);
  double t_kernel1 = timing ? poly_now_ms() : 0.0;
  if (!kernel_graph) {
    free(buf_order_orig);
    return NULL;
  }
  uint32_t ghash = poly_structural_hash(kernel_graph) ^ (POLY_SCHED_CACHE_VERSION * 2654435761u);
  double t_build0 = timing ? poly_now_ms() : 0.0;
  PolySchedule *ps = build_schedule_from_kernel_graph(
      ctx, kernel_graph, mode, ghash, buf_order_orig, n_bufs_orig, bind_vals, n_bind_vals, NULL
  );
  double t_build1 = timing ? poly_now_ms() : 0.0;
  if (timing) {
    fprintf(
        stderr,
        "[polygrad:create_schedule] done path=kernel_graph kernel=%.3fms build=%.3fms total=%.3fms "
        "items=%d slots=%d\n",
        t_kernel1 - t_kernel0, t_build1 - t_build0, t_build1 - t0, ps ? ps->n_items : -1,
        ps ? ps->n_buf_slots : -1
    );
    fflush(stderr);
  }
  free(buf_order_orig);
  return ps;
}

static PolySchedule *poly_create_schedule_uncached(PolyCtx *ctx, PolyUOp *kernel_graph) {
  if (!kernel_graph || kernel_graph->op != POLY_OP_SINK) {
    fprintf(stderr, "polygrad: create_schedule: expected SINK\n");
    return NULL;
  }

  PolyUOp **buf_order = NULL;
  int n_external = 0;
  if (!collect_external_buf_order_from_kernel_graph(kernel_graph, &buf_order, &n_external))
    return NULL;
  uint32_t ghash = poly_structural_hash(kernel_graph) ^ (POLY_SCHED_CACHE_VERSION * 2654435761u);
  PolySchedule *schedule = build_schedule_from_kernel_graph(
      ctx, kernel_graph, POLY_MODE_CALL, ghash, buf_order, n_external, NULL, 0, NULL
  );
  free(buf_order);
  return schedule;
}

/* tinygrad/engine/schedule.py: create_schedule
 * Build the backend-neutral schedule from the kernel-graph boundary, resolving
 * cached LINEAR kernel roots against the current graph's buffers/vars. */
PolySchedule *poly_create_schedule(PolyCtx *ctx, PolyUOp *kernel_graph) {
  if (!kernel_graph || kernel_graph->op != POLY_OP_SINK) {
    fprintf(stderr, "polygrad: create_schedule: expected SINK\n");
    return NULL;
  }

  PolyUOp *linear_template =
      poly_schedule_cache_enabled() ? poly_lower_kernel_graph_to_linear(ctx, kernel_graph) : NULL;
  if (poly_schedule_cache_enabled() && !linear_template) return NULL;

  PolyUOp **buf_order = NULL;
  int n_external = 0;
  if (!collect_external_buf_order_from_kernel_graph(kernel_graph, &buf_order, &n_external))
    return NULL;
  uint32_t ghash = poly_structural_hash(kernel_graph) ^ (POLY_SCHED_CACHE_VERSION * 2654435761u);
  PolySchedule *schedule = build_schedule_from_kernel_graph(
      ctx, kernel_graph, POLY_MODE_CALL, ghash, buf_order, n_external, NULL, 0, linear_template
  );
  free(buf_order);
  return schedule;
}

static void poly_runner_cleanup(PolyRunner *runner, PolyDevice device) {
  if (!runner) return;
  const PolyBackendDesc *backend = poly_backend_get(device);
  if (runner->handle) {
    if (runner->free_handle)
      runner->free_handle(runner);
    else if (backend)
      backend->free_runner(runner);
  }
  free(runner->param_to_slot);
  free(runner->var_indices);
  memset(runner, 0, sizeof(*runner));
}

static void borrowed_runner_free_fn(void *self) {
  (void)self;
  /* Borrowed runners point at ctx-owned program-cache handles. Schedule/item
   * cleanup must release only the per-call param mapping, not the shared
   * backend runtime object. */
}

static bool poly_program_cache_enabled(void) {
  const char *v = getenv("POLY_PCACHE");
  return !v || v[0] != '0';
}

static uint32_t poly_program_cache_hash(PolyUOp *root, PolyDevice device, uint32_t env_stamp) {
  uint32_t h = poly_structural_hash(root);
  h ^= ((uint32_t)device + 0x9e3779b9u + (h << 6) + (h >> 2));
  h ^= (env_stamp + 0x85ebca6bu + (h << 6) + (h >> 2));
  return h;
}

static bool poly_program_cache_eq(const void *a, const void *b) {
  const PolyProgramCacheEntry *ka = (const PolyProgramCacheEntry *)a;
  const PolyProgramCacheEntry *kb = (const PolyProgramCacheEntry *)b;
  return ka && kb && ka->device == kb->device && ka->env_stamp == kb->env_stamp &&
         poly_structural_eq(ka->root, kb->root);
}

static void poly_program_cache_entry_free(const void *key, void *value, void *userdata) {
  (void)key;
  (void)userdata;
  PolyProgramCacheEntry *entry = (PolyProgramCacheEntry *)value;
  if (!entry) return;
  poly_runner_cleanup(&entry->runner, entry->device);
  free(entry);
}

void poly_program_cache_clear(PolyCtx *ctx) {
  if (!ctx || !ctx->program_cache) return;
  poly_map_foreach(ctx->program_cache, poly_program_cache_entry_free, NULL);
  poly_map_clear(ctx->program_cache);
}

size_t poly_program_cache_len(PolyCtx *ctx) {
  return (ctx && ctx->program_cache) ? poly_map_len(ctx->program_cache) : 0;
}

void poly_schedule_ctx_cleanup(PolyCtx *ctx) {
  /* ctx owns cached backend programs, matching tinygrad's global
   * to_program/runtime caches. LINEAR schedule templates are arena UOps and do
   * not need explicit cleanup; backend handles do. */
  poly_program_cache_clear(ctx);
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
      if (sched->run_intermediates[i].ptr && sched->run_intermediates[i].allocator)
        sched->run_intermediates[i].allocator->free(
            &sched->run_intermediates[i], sched->run_intermediates[i].allocator->dev_ctx
        );
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

static PolyArg schedule_cache_buffer_param_arg(PolyUOp *u, int pos, int64_t *vals, int cap);

typedef struct {
  PolyUOp *u;
  int state;
} ScheduleCacheVisit;

static int schedule_cache_rewrite_child_count(PolyUOp *u, bool normalize_buffers) {
  if (!u) return 0;
  if (normalize_buffers && (u->op == POLY_OP_BUFFER || u->op == POLY_OP_BUFFER_VIEW)) return 0;
  if (u->op == POLY_OP_BIND && u->n_src >= 1) return 1;
  return u->n_src;
}

static bool schedule_cache_visit_push(
    ScheduleCacheVisit **stack,
    int *sp,
    int *cap,
    PolyUOp *u,
    int state
) {
  if (*sp >= *cap) {
    int new_cap = *cap * 2;
    ScheduleCacheVisit *new_stack = realloc(*stack, (size_t)new_cap * sizeof(ScheduleCacheVisit));
    if (!new_stack) return false;
    *stack = new_stack;
    *cap = new_cap;
  }
  (*stack)[(*sp)++] = (ScheduleCacheVisit){u, state};
  return true;
}

static PolyUOp *schedule_cache_rewrite_iter(
    PolyCtx *ctx,
    PolyUOp *root,
    PolyUOp **input_order,
    int n_inputs,
    bool normalize_buffers
) {
  if (!ctx || !root) return NULL;

  PolyMap *memo = poly_map_new(64);
  if (!memo) return NULL;

  int stack_cap = 256;
  int sp = 0;
  ScheduleCacheVisit *stack = malloc((size_t)stack_cap * sizeof(ScheduleCacheVisit));
  if (!stack) {
    poly_map_destroy(memo);
    return NULL;
  }
  bool ok = schedule_cache_visit_push(&stack, &sp, &stack_cap, root, 0);

  while (ok && sp > 0) {
    ScheduleCacheVisit cur = stack[--sp];
    PolyUOp *u = cur.u;
    if (!u) {
      ok = false;
      break;
    }
    if (poly_map_get(memo, poly_ptr_hash(u), u, poly_ptr_eq)) continue;

    if (cur.state == 0) {
      ok = schedule_cache_visit_push(&stack, &sp, &stack_cap, u, 1);
      if (!ok) break;

      int n_child = schedule_cache_rewrite_child_count(u, normalize_buffers);
      for (int i = n_child - 1; i >= 0; i--) {
        PolyUOp *child = u->src[i];
        if (!child) {
          ok = false;
          break;
        }
        if (poly_map_get(memo, poly_ptr_hash(child), child, poly_ptr_eq)) continue;
        ok = schedule_cache_visit_push(&stack, &sp, &stack_cap, child, 0);
        if (!ok) break;
      }
      continue;
    }

    PolyUOp *result = u;
    if (normalize_buffers && (u->op == POLY_OP_BUFFER || u->op == POLY_OP_BUFFER_VIEW)) {
      int pos = poly_find_buf_position(u, input_order, n_inputs);
      if (pos >= 0) {
        /* This is Polygrad's C analogue of tinygrad callify.pm_replace_buf for
         * schedule-cache identity: replace concrete BUFFER/BUFFER_VIEW identity
         * with a stable parameter position, while retaining dtype, size/view
         * metadata, and concrete device. This must be a PARAM-like key node:
         * Polygrad structural equality intentionally ignores BUFFER sources. */
        int64_t key_vals[64];
        result = poly_uop0(
            ctx, POLY_OP_PARAM, u->dtype,
            schedule_cache_buffer_param_arg(
                u, pos, key_vals, (int)(sizeof(key_vals) / sizeof(key_vals[0]))
            )
        );
      }
    } else if (u->op == POLY_OP_BIND && u->n_src >= 1) {
      result = poly_map_get(memo, poly_ptr_hash(u->src[0]), u->src[0], poly_ptr_eq);
      if (!result) ok = false;
    } else {
      PolyUOp *stack_src[16];
      PolyUOp **new_src = (u->n_src > (int)(sizeof(stack_src) / sizeof(stack_src[0])))
                              ? malloc((size_t)u->n_src * sizeof(PolyUOp *))
                              : stack_src;
      if (!new_src) {
        ok = false;
      } else {
        bool changed = false;
        for (int i = 0; i < u->n_src; i++) {
          new_src[i] = poly_map_get(memo, poly_ptr_hash(u->src[i]), u->src[i], poly_ptr_eq);
          if (!new_src[i]) {
            ok = false;
            break;
          }
          if (new_src[i] != u->src[i]) changed = true;
        }
        if (ok && changed) result = poly_uop(ctx, u->op, u->dtype, new_src, u->n_src, u->arg);
        if (new_src != stack_src) free(new_src);
      }
    }
    if (ok && !result) ok = false;
    if (ok) poly_map_set(memo, poly_ptr_hash(u), u, result, poly_ptr_eq);
  }

  PolyUOp *ret = ok ? poly_map_get(memo, poly_ptr_hash(root), root, poly_ptr_eq) : NULL;
  free(stack);
  poly_map_destroy(memo);
  return ret;
}

static PolyUOp *schedule_cache_key_for_kernel_graph(PolyCtx *ctx, PolyUOp *kernel_graph) {
  return schedule_cache_rewrite_iter(ctx, kernel_graph, NULL, 0, false);
}

static PolyUOp *poly_build_linear_from_kernel_graph_uncached(
    PolyCtx *ctx,
    PolyUOp *kernel_graph,
    PolyUOp **external_buf_order,
    int n_external_buf_order
) {
  bool timing = poly_debug_at_least(2);
  double t0 = timing ? poly_now_ms() : 0.0;
  if (timing) {
    fprintf(
        stderr, "[polygrad:build_linear] begin kernel_graph=%p external_order=%d\n",
        (void *)kernel_graph, n_external_buf_order
    );
    fflush(stderr);
  }
  PolySchedule *schedule = poly_create_schedule_uncached(ctx, kernel_graph);
  if (!schedule) return NULL;
  double t_schedule = timing ? poly_now_ms() : 0.0;
  if (timing) {
    fprintf(
        stderr, "[polygrad:build_linear] schedule items=%d slots=%d ms=%.3f\n", schedule->n_items,
        schedule->n_buf_slots, t_schedule - t0
    );
    fflush(stderr);
  }

  PolyUOp **external_bufs = NULL;
  bool external_bufs_owned = false;
  int n_external = 0;
  if (external_buf_order) {
    /* LINEAR CALL params are replayed later against the schedule caller's
     * external buffer slots. For sink-level caching that slot list comes from
     * the original SINK, not from the optimized kernel graph. Keeping the same
     * numbering here prevents dropped/rewritten inputs from shifting params. */
    n_external = n_external_buf_order;
    external_bufs = external_buf_order;
  } else {
    /* Kernel-graph callers do not have a pre-rangeify SINK, so their natural
     * external order is the one discovered from the kernel graph itself. */
    if (!collect_external_buf_order_from_kernel_graph(kernel_graph, &external_bufs, &n_external)) {
      poly_schedule_free(schedule);
      return NULL;
    }
    external_bufs_owned = true;
  }
  PolyUOp **linear_src = calloc((size_t)schedule->n_items, sizeof(PolyUOp *));
  if (!linear_src) {
    if (external_bufs_owned) free(external_bufs);
    poly_schedule_free(schedule);
    return NULL;
  }
  for (int step = 0; step < schedule->n_items; step++) {
    int idx = schedule->exec_order ? schedule->exec_order[step] : step;
    PolyExecItem *item = &schedule->items[idx];
    int n_call_src = 1 + item->n_buf_slots + item->n_var_uops;
    PolyUOp **call_src = calloc((size_t)n_call_src, sizeof(PolyUOp *));
    if (!call_src) {
      free(linear_src);
      if (external_bufs_owned) free(external_bufs);
      poly_schedule_free(schedule);
      return NULL;
    }
    call_src[0] = item->root;
    for (int i = 0; i < item->n_buf_slots; i++) {
      int slot = item->buf_slot_indices[i];
      PolyUOp *buf =
          (slot >= 0 && slot < schedule->n_buf_slots) ? schedule->buf_slots[slot].buf_uop : NULL;
      int external_slot = poly_find_buf_position(buf, external_bufs, n_external);
      call_src[1 + i] = linear_call_param_for_buf(ctx, buf, external_slot);
    }
    for (int i = 0; i < item->n_var_uops; i++)
      call_src[1 + item->n_buf_slots + i] = item->var_uops[i];
    linear_src[step] =
        poly_uop(ctx, POLY_OP_CALL, POLY_VOID, call_src, n_call_src, poly_arg_int(item->kind));
    free(call_src);
  }
  PolyUOp *linear =
      poly_uop(ctx, POLY_OP_LINEAR, POLY_VOID, linear_src, schedule->n_items, poly_arg_none());
  free(linear_src);
  if (external_bufs_owned) free(external_bufs);
  poly_schedule_free(schedule);
  if (timing) {
    double t_done = poly_now_ms();
    fprintf(
        stderr, "[polygrad:build_linear] done calls=%d build_calls=%.3fms total=%.3fms\n",
        linear ? linear->n_src : -1, t_done - t_schedule, t_done - t0
    );
    fflush(stderr);
  }
  return linear;
}

static PolyUOp *poly_lower_kernel_graph_to_linear(PolyCtx *ctx, PolyUOp *kernel_graph) {
  PolyUOp *cache_key = schedule_cache_key_for_kernel_graph(ctx, kernel_graph);
  if (!cache_key) return NULL;
  uint32_t cache_hash = poly_structural_hash(cache_key) ^ (POLY_SCHED_CACHE_VERSION * 2654435761u);
  PolyUOp *cached = poly_map_get(ctx->schedule_cache, cache_hash, cache_key, poly_structural_eq);
  if (cached) return cached;

  PolyUOp *linear = poly_build_linear_from_kernel_graph_uncached(ctx, kernel_graph, NULL, 0);
  if (!linear) return NULL;
  poly_map_set(ctx->schedule_cache, cache_hash, cache_key, linear, poly_structural_eq);
  return linear;
}

static bool schedule_cache_stack_push(PolyUOp ***stack, int *sp, int *cap, PolyUOp *u) {
  if (*sp >= *cap) {
    int new_cap = *cap * 2;
    PolyUOp **new_stack = realloc(*stack, (size_t)new_cap * sizeof(PolyUOp *));
    if (!new_stack) return false;
    *stack = new_stack;
    *cap = new_cap;
  }
  (*stack)[(*sp)++] = u;
  return true;
}

static bool schedule_cache_collect_inputs(
    PolyUOp *u,
    PolyUOp ***out_input_order,
    int *n_inputs,
    int *n_visited
) {
  if (!u || !out_input_order || !n_inputs || !n_visited) return false;
  *out_input_order = NULL;
  *n_inputs = 0;
  *n_visited = 0;

  PolyMap *seen = poly_map_new(1024);
  if (!seen) return false;

  PolyUOp **input_order = NULL;
  int input_cap = 0;
  int cap = 1024;
  int sp = 0;
  PolyUOp **stack = malloc((size_t)cap * sizeof(PolyUOp *));
  bool ok = stack && schedule_cache_stack_push(&stack, &sp, &cap, u);

  while (ok && sp > 0) {
    PolyUOp *cur = stack[--sp];
    if (!cur) {
      ok = false;
      break;
    }
    if (poly_map_get(seen, poly_ptr_hash(cur), cur, poly_ptr_eq)) continue;
    poly_map_set(seen, poly_ptr_hash(cur), cur, cur, poly_ptr_eq);
    (*n_visited)++;

    if (cur->op == POLY_OP_BUFFER || cur->op == POLY_OP_BUFFER_VIEW) {
      if (*n_inputs >= input_cap) {
        int new_cap = input_cap ? input_cap * 2 : 16;
        PolyUOp **tmp = realloc(input_order, (size_t)new_cap * sizeof(PolyUOp *));
        if (!tmp) {
          ok = false;
          break;
        }
        input_order = tmp;
        input_cap = new_cap;
      }
      input_order[(*n_inputs)++] = cur;
      continue;
    }

    for (int i = cur->n_src - 1; i >= 0; i--) {
      ok = schedule_cache_stack_push(&stack, &sp, &cap, cur->src[i]);
      if (!ok) break;
    }
  }

  free(stack);
  poly_map_destroy(seen);
  if (!ok) {
    free(input_order);
    return false;
  }
  *out_input_order = input_order;
  return ok;
}

static PolyArg schedule_cache_buffer_param_arg(PolyUOp *u, int pos, int64_t *vals, int cap) {
  int n = 0;
  vals[n++] = (int64_t)(u ? u->op : POLY_OP_NOOP);
  vals[n++] = (int64_t)pos;
  vals[n++] = (int64_t)poly_uop_device(u);
  vals[n++] = (int64_t)(u ? u->arg.kind : POLY_ARG_NONE);
  if (u && u->arg.kind == POLY_ARG_INT) {
    vals[n++] = u->arg.i;
  } else if (u && u->arg.kind == POLY_ARG_INT_TUPLE) {
    vals[n++] = u->arg.int_tuple.n;
    for (int i = 0; i < u->arg.int_tuple.n && n < cap; i++)
      vals[n++] = u->arg.int_tuple.vals[i];
  }

  PolyArg arg = poly_arg_none();
  arg.kind = POLY_ARG_INT_TUPLE;
  arg.int_tuple.vals = vals;
  arg.int_tuple.n = n;
  return arg;
}

static PolyUOp *schedule_cache_key_for_sink(
    PolyCtx *ctx,
    PolyUOp *sink,
    PolyUOp **input_order,
    int n_inputs
) {
  return schedule_cache_rewrite_iter(ctx, sink, input_order, n_inputs, true);
}

PolyUOp *poly_lower_sink_to_linear(PolyCtx *ctx, PolyUOp *sink, PolyCompileMode mode) {
  (void)mode;
  if (!ctx || !sink || sink->op != POLY_OP_SINK) return NULL;

  bool timing = poly_debug_at_least(2);
  double t0 = timing ? poly_now_ms() : 0.0;
  if (timing) {
    fprintf(stderr, "[polygrad:lower_sink_to_linear] begin sink=%p\n", (void *)sink);
    fflush(stderr);
  }

  PolyUOp **input_order = NULL;
  int n_inputs = 0, n_visited = 0;
  if (!schedule_cache_collect_inputs(sink, &input_order, &n_inputs, &n_visited)) return NULL;
  double t_inputs = timing ? poly_now_ms() : 0.0;

  PolyUOp *cache_key = schedule_cache_key_for_sink(ctx, sink, input_order, n_inputs);
  if (!cache_key) {
    free(input_order);
    return NULL;
  }
  uint32_t cache_hash = poly_structural_hash(cache_key) ^ (POLY_SCHED_CACHE_VERSION * 2654435761u);
  PolyUOp *cached =
      poly_schedule_cache_enabled()
          ? poly_map_get(ctx->schedule_cache, cache_hash, cache_key, poly_structural_eq)
          : NULL;
  double t_key = timing ? poly_now_ms() : 0.0;
  if (cached) {
    if (timing) {
      fprintf(
          stderr,
          "[polygrad:lower_sink_to_linear] cache hit inputs=%d visited=%d inputs_ms=%.3f "
          "key_ms=%.3f total=%.3f\n",
          n_inputs, n_visited, t_inputs - t0, t_key - t_inputs, t_key - t0
      );
      fflush(stderr);
    }
    free(input_order);
    return cached;
  }
  if (timing) {
    fprintf(
        stderr,
        "[polygrad:lower_sink_to_linear] cache miss inputs=%d visited=%d inputs_ms=%.3f "
        "key_ms=%.3f\n",
        n_inputs, n_visited, t_inputs - t0, t_key - t_inputs
    );
    fflush(stderr);
  }

  PolyUOp **raw_external_bufs = NULL;
  int n_raw_external = 0, n_raw_visited = 0;
  if (!poly_collect_buf_order_alloc(sink, &raw_external_bufs, &n_raw_external, &n_raw_visited)) {
    free(input_order);
    return NULL;
  }
  double t_raw = timing ? poly_now_ms() : 0.0;

  PolyUOp *kernel_graph = poly_get_kernel_graph(ctx, sink);
  if (!kernel_graph) {
    free(input_order);
    free(raw_external_bufs);
    return NULL;
  }
  double t_kernel = timing ? poly_now_ms() : 0.0;
  PolyUOp *linear = poly_build_linear_from_kernel_graph_uncached(
      ctx, kernel_graph, raw_external_bufs, n_raw_external
  );
  free(input_order);
  free(raw_external_bufs);
  if (!linear) return NULL;
  double t_linear = timing ? poly_now_ms() : 0.0;
  if (poly_schedule_cache_enabled())
    poly_map_set(ctx->schedule_cache, cache_hash, cache_key, linear, poly_structural_eq);
  if (timing) {
    fprintf(
        stderr,
        "[polygrad:lower_sink_to_linear] done raw_bufs=%d raw_visited=%d raw=%.3fms kernel=%.3fms "
        "linear=%.3fms total=%.3fms calls=%d\n",
        n_raw_external, n_raw_visited, t_raw - t_key, t_kernel - t_raw, t_linear - t_kernel,
        t_linear - t0, linear ? linear->n_src : -1
    );
    fflush(stderr);
  }
  return linear;
}

size_t poly_schedule_cache_len(PolyCtx *ctx) {
  return (ctx && ctx->schedule_cache) ? poly_map_len(ctx->schedule_cache) : 0;
}

void poly_schedule_cache_clear(PolyCtx *ctx) {
  if (ctx && ctx->schedule_cache) poly_map_clear(ctx->schedule_cache);
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
  if (buffer->frontend_release)
    buffer->frontend_release((uintptr_t)buffer);
  else
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
  if (dst && dst->device == POLY_DEVICE_HOST && dst->ptr && src && src->ptr) {
    memcpy(dst->ptr, src->ptr, n);
    return 0;
  }
  if (dst && dst->device == POLY_DEVICE_HOST && src && src->ptr) {
    uintptr_t key = (uintptr_t)(dst->src ? dst->src : dst);
    return poly_browser_host_copy_in(key, src->ptr, n);
  }
#endif
  if (!dst || !dst->ptr || !src || !src->ptr) return -1;
  memcpy(dst->ptr, src->ptr, n);
  return 0;
}

static int host_copy_out(const PolyBuffer *dst, const PolyBuffer *src, size_t n, void *dev_ctx) {
  (void)dev_ctx;
#ifdef __EMSCRIPTEN__
  if (src && src->device == POLY_DEVICE_HOST && src->ptr && dst && dst->ptr) {
    memcpy(dst->ptr, src->ptr, n);
    return 0;
  }
  if (src && src->device == POLY_DEVICE_HOST && dst && dst->ptr) {
    uintptr_t key = (uintptr_t)(src->src ? src->src : src);
    return poly_browser_host_copy_out(key, dst->ptr, n);
  }
#endif
  if (!dst || !dst->ptr || !src || !src->ptr) return -1;
  memcpy(dst->ptr, src->ptr, n);
  return 0;
}

static int host_copy_between(
    const PolyBuffer *dst,
    const PolyBuffer *src,
    size_t n,
    void *dev_ctx
) {
  (void)dev_ctx;
#ifdef __EMSCRIPTEN__
  if (dst && src && dst->device == POLY_DEVICE_HOST && src->device == POLY_DEVICE_HOST) {
    if (dst->ptr && src->ptr) {
      memcpy(dst->ptr, src->ptr, n);
      return 0;
    }
    uint8_t *tmp = malloc(n);
    if (!tmp) return -1;
    uintptr_t src_key = (uintptr_t)(src->src ? src->src : src);
    uintptr_t dst_key = (uintptr_t)(dst->src ? dst->src : dst);
    int rc = poly_browser_host_copy_out(src_key, tmp, n);
    if (rc == 0) rc = poly_browser_host_copy_in(dst_key, tmp, n);
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

static int cuda_copy_between_fn(
    const PolyBuffer *dst,
    const PolyBuffer *src,
    size_t n,
    void *dev_ctx
) {
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

static int hip_copy_between_fn(
    const PolyBuffer *dst,
    const PolyBuffer *src,
    size_t n,
    void *dev_ctx
) {
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

typedef struct {
  size_t nbytes;
  PolyDevice dst_device;
  PolyDevice src_device;
  PolyBuffer *dst_identity;
  PolyBuffer *src_identity;
} CopyRunnerHandle;

static int copy_execute_fn(void *self, void **args, int n_args) {
  PolyRunner *runner = (PolyRunner *)self;
  CopyRunnerHandle *ch = (CopyRunnerHandle *)runner->handle;
  if (!ch || !args || n_args < 2) return -1;

  bool dst_keyed_host = ch->dst_device == POLY_DEVICE_HOST && ch->dst_identity != NULL;
  bool src_keyed_host = ch->src_device == POLY_DEVICE_HOST && ch->src_identity != NULL;
  /* Browser WebGPU keeps imported HOST bytes in JS-owned TypedArrays. Those
   * schedule slots intentionally have ptr=NULL; the copy backend uses the
   * retained PolyBuffer identity as the JS host-buffer key instead. */
  if ((!args[0] && !dst_keyed_host) || (!args[1] && !src_keyed_host)) return -1;

  const PolyBackendDesc *dst_backend = poly_backend_get(ch->dst_device);
  const PolyBackendDesc *src_backend = poly_backend_get(ch->src_device);
  if ((dst_backend && poly_backend_ensure_open(ch->dst_device) != 0) ||
      (src_backend && poly_backend_ensure_open(ch->src_device) != 0))
    return -1;
  const PolyAllocator *dst_alloc = dst_backend ? dst_backend->get_allocator() : NULL;
  const PolyAllocator *src_alloc = src_backend ? src_backend->get_allocator() : NULL;
  if (!dst_alloc || !src_alloc) return -1;

  PolyBuffer dst = {
      .ptr = args[0],
      .nbytes = ch->nbytes,
      .device = ch->dst_device,
      .owned = false,
      .allocator = dst_alloc,
      .src = ch->dst_identity,
      .valid = true,
  };
  PolyBuffer src = {
      .ptr = args[1],
      .nbytes = ch->nbytes,
      .device = ch->src_device,
      .owned = false,
      .allocator = src_alloc,
      .src = ch->src_identity,
      .valid = true,
  };

  return poly_buffer_copy(&dst, &src);
}

static bool runner_copy_param_allows_null(const PolyRunner *runner, int param_index) {
  if (!runner || runner->kind != POLY_RUNNER_COPY || !runner->handle) return false;
  CopyRunnerHandle *ch = (CopyRunnerHandle *)runner->handle;
  /* NULL is only valid for browser HOST endpoints where the real data lives in
   * the frontend registry and the copy handle captured its PolyBuffer key. */
  return (param_index == 0 && ch->dst_device == POLY_DEVICE_HOST && ch->dst_identity) ||
         (param_index == 1 && ch->src_device == POLY_DEVICE_HOST && ch->src_identity);
}

static void copy_free_fn(void *self) {
  PolyRunner *runner = (PolyRunner *)self;
  free(runner ? runner->handle : NULL);
}

static int poly_lower_copy_item(
    PolyCtx *ctx,
    PolySchedule *schedule,
    PolyExecItem *item,
    PolyDevice device,
    PolyRunner *out
) {
  if (!schedule || !item || !out || item->n_buf_slots < 2) return -1;

  int dst_slot = item->buf_slot_indices[0];
  int src_slot = item->buf_slot_indices[1];
  if (dst_slot < 0 || dst_slot >= schedule->n_buf_slots || src_slot < 0 ||
      src_slot >= schedule->n_buf_slots)
    return -1;

  PolyDevice dst_device = poly_schedule_slot_target_device(ctx, schedule, dst_slot, device);
  PolyDevice src_device = poly_schedule_slot_target_device(ctx, schedule, src_slot, device);
  if (dst_device == POLY_DEVICE_AUTO) dst_device = device;
  if (src_device == POLY_DEVICE_AUTO) src_device = device;

  size_t dst_nbytes = (size_t)schedule->buf_slots[dst_slot].nbytes;
  size_t src_nbytes = (size_t)schedule->buf_slots[src_slot].nbytes;
  if (dst_nbytes == 0 || src_nbytes == 0 || dst_nbytes != src_nbytes) return -1;

  CopyRunnerHandle *ch = malloc(sizeof(CopyRunnerHandle));
  if (!ch) return -1;
  ch->nbytes = dst_nbytes;
  ch->dst_device = dst_device;
  ch->src_device = src_device;
  PolyBuffer *dst_buf = poly_buffer_get(ctx, schedule->buf_slots[dst_slot].buf_uop);
  PolyBuffer *src_buf = poly_buffer_get(ctx, schedule->buf_slots[src_slot].buf_uop);
  ch->dst_identity = (dst_buf && dst_buf->device == POLY_DEVICE_HOST) ? dst_buf : NULL;
  ch->src_identity = (src_buf && src_buf->device == POLY_DEVICE_HOST) ? src_buf : NULL;

  out->kind = POLY_RUNNER_COPY;
  out->handle = ch;
  out->handle_size = (int)sizeof(*ch);
  out->execute = copy_execute_fn;
  out->free_handle = copy_free_fn;
  return 0;
}

static int poly_bind_runner_param_slots(
    PolyCtx *ctx,
    PolyExecItem *item,
    PolyRunner *runner,
    PolyDevice device
) {
  /* Backend lowering produces a reusable program/runtime object. The current
   * schedule still owns the PARAM -> buffer-slot mapping, so cached programs
   * can run against fresh buffers just like tinygrad resolves cached LINEAR
   * calls back to the current buffer inputs. */
  if (item->kind != POLY_EXEC_COPY && device == POLY_DEVICE_WEBGPU)
    return webgpu_fill_param_slots(ctx, item, runner);

  runner->n_params = item->n_buf_slots;
  if (item->n_buf_slots <= 0) return 0;
  runner->param_to_slot = malloc((size_t)item->n_buf_slots * sizeof(int));
  if (!runner->param_to_slot) return -1;
  memcpy(runner->param_to_slot, item->buf_slot_indices, (size_t)item->n_buf_slots * sizeof(int));
  return 0;
}

static int poly_lower_compute_item_cached(
    PolyCtx *ctx,
    PolyExecItem *item,
    PolyDevice device,
    uint32_t env_stamp,
    PolyRunner *out
) {
  const PolyBackendDesc *backend = poly_backend_get(device);
  if (!backend || !backend->lower_item) return -1;
  if (poly_backend_ensure_open(device) != 0) return -1;

  if (!poly_validate_kernel_graph(ctx, item->root)) return -2;

  PolyProgramCacheEntry key = {
      .root = item->root,
      .device = device,
      .env_stamp = env_stamp,
  };
  uint32_t hash = poly_program_cache_hash(item->root, device, env_stamp);
  PolyProgramCacheEntry *entry =
      (poly_program_cache_enabled() && ctx && ctx->program_cache)
          ? poly_map_get(ctx->program_cache, hash, &key, poly_program_cache_eq)
          : NULL;

  if (!entry) {
    char fn_name[64];
    stable_kernel_fn_name(fn_name, sizeof(fn_name), device, item->root);

    PolyRunner lowered = {0};
    if (backend->lower_item(ctx, item->root, fn_name, &lowered) != 0) return -1;

    if (poly_program_cache_enabled() && ctx && ctx->program_cache) {
      entry = calloc(1, sizeof(*entry));
      if (entry) {
        entry->root = item->root;
        entry->device = device;
        entry->env_stamp = env_stamp;
        entry->runner = lowered;
        /* Cached backend runners must not carry call-specific slot metadata.
         * Slot metadata is rebuilt below for each fresh PolySchedule. */
        entry->runner.param_to_slot = NULL;
        entry->runner.n_params = 0;
        entry->runner.var_indices = NULL;
        entry->runner.n_vars = 0;
        poly_map_set(ctx->program_cache, hash, entry, entry, poly_program_cache_eq);
      } else {
        *out = lowered;
        return 0;
      }
    } else {
      *out = lowered;
      return 0;
    }
  }

  *out = entry->runner;
  out->param_to_slot = NULL;
  out->n_params = 0;
  out->var_indices = NULL;
  out->n_vars = 0;
  out->free_handle = borrowed_runner_free_fn;
  return 0;
}

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
  PolyRewriteOpts opts = {
      .optimize = true,
      .devectorize = 1,
      .caps = poly_c_renderer_caps(),
      .device = POLY_DEVICE_CPU,
      .opt_policy = POLY_OPT_HEURISTIC,
      .extra_matcher = poly_pm_c_renderer_extra(),
  };
  PolyUOp **lin = poly_linearize_ex(ctx, scheduled_root, opts, &n_lin);
  if (!lin) return -1;
  int cpu_threads = 1;
  for (int j = 0; j < n_lin; j++) {
    PolyUOp *u = lin[j];
    if (!u || u->op != POLY_OP_DEFINE_VAR || u->arg.kind != POLY_ARG_DEFINE_VAR ||
        !u->arg.define_var.name || strcmp(u->arg.define_var.name, "core_id") != 0)
      continue;
    int64_t n = u->arg.define_var.max_val + 1;
    if (n > 1 && n <= INT32_MAX) cpu_threads = (int)n;
  }
  if (poly_debug_at_least(3)) {
    int n_weak = 0;
    for (int i = 0; i < n_lin; i++)
      if (lin[i] && poly_dtype_is_index(lin[i]->dtype)) n_weak++;
    fprintf(stderr, "[polygrad:cpu_lower] fn=%s linear=%d weakint=%d\n", fn_name, n_lin, n_weak);
    fflush(stderr);
  }

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
  out->grid[0] = cpu_threads;
  out->grid[1] = 1;
  out->grid[2] = 1;
  out->block[0] = 1;
  out->block[1] = 1;
  out->block[2] = 1;
  out->execute = cpu_execute_fn;
  out->free_handle = cpu_free_fn;
  return 0;
}

static int cpu_execute_fn(void *self, void **args, int n_args) {
  PolyRunner *runner = (PolyRunner *)self;
  int threads = runner->grid[0] > 1 ? runner->grid[0] : 1;
  if (threads > 1)
    poly_program_call_threaded((PolyProgram *)runner->handle, args, n_args, threads);
  else
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
    if (lin[j]->op != POLY_OP_SPECIAL || lin[j]->n_src <= 0) continue;
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

static int backend_noop_ensure_open(void) {
  return 0;
}

#ifdef POLY_HAS_CUDA
static int cuda_ensure_open(void) {
  return poly_cuda_init();
}
#endif

/* ══════════════════════════════════════════════════════════════════════ */
/*  Backend registry                                                     */
/* ══════════════════════════════════════════════════════════════════════ */

static const PolyBackendDesc BACKENDS[] = {
    [POLY_DEVICE_AUTO] = {NULL, POLY_DEVICE_AUTO, false, NULL, NULL, NULL, NULL, NULL},
    [POLY_DEVICE_HOST] =
        {"host", POLY_DEVICE_HOST, false, NULL, NULL, NULL, backend_noop_ensure_open,
         host_get_allocator},
#ifndef __EMSCRIPTEN__
    [POLY_DEVICE_CPU] =
        {"cpu", POLY_DEVICE_CPU, false, cpu_lower_item, cpu_execute, cpu_free_runner,
         backend_noop_ensure_open, cpu_get_allocator},
#else
    [POLY_DEVICE_CPU] = {NULL, POLY_DEVICE_CPU, false, NULL, NULL, NULL, NULL, NULL},
#endif
    [POLY_DEVICE_INTERP] =
        {"interp", POLY_DEVICE_INTERP, false, interp_lower_item, interp_execute, interp_free_runner,
         backend_noop_ensure_open, interp_get_allocator},
#ifdef POLY_HAS_CUDA
    [POLY_DEVICE_CUDA] =
        {"cuda", POLY_DEVICE_CUDA, false, cuda_lower_item, cuda_execute, cuda_free_runner,
         cuda_ensure_open, cuda_get_allocator},
#else
    [POLY_DEVICE_CUDA] = {NULL, POLY_DEVICE_CUDA, false, NULL, NULL, NULL, NULL, NULL},
#endif
#ifdef __EMSCRIPTEN__
    [POLY_DEVICE_WASM] =
        {"wasm", POLY_DEVICE_WASM, true, poly_wasm_lower_item, poly_wasm_execute,
         poly_wasm_free_runner, backend_noop_ensure_open, poly_wasm_get_allocator},
#else
    [POLY_DEVICE_WASM] = {NULL, POLY_DEVICE_WASM, false, NULL, NULL, NULL, NULL, NULL},
#endif
#ifdef __EMSCRIPTEN__
    [POLY_DEVICE_WEBGPU] =
        {"webgpu", POLY_DEVICE_WEBGPU, true, poly_webgpu_lower_item, poly_webgpu_execute,
         poly_webgpu_free_runner, backend_noop_ensure_open, poly_webgpu_get_allocator},
#else
    [POLY_DEVICE_WEBGPU] = {NULL, POLY_DEVICE_WEBGPU, false, NULL, NULL, NULL, NULL, NULL},
#endif
#ifdef POLY_HAS_X64
    [POLY_DEVICE_X64_JIT] =
        {"x64_jit", POLY_DEVICE_X64_JIT, false, x64_lower_item, x64_execute, x64_free_runner,
         backend_noop_ensure_open, cpu_get_allocator},
#else
    [POLY_DEVICE_X64_JIT] = {NULL, POLY_DEVICE_X64_JIT, false, NULL, NULL, NULL, NULL, NULL},
#endif
#ifdef POLY_HAS_HIP
    [POLY_DEVICE_HIP] =
        {"hip", POLY_DEVICE_HIP, false, hip_lower_item, hip_execute, hip_free_runner, poly_hip_init,
         hip_get_allocator},
#else
    [POLY_DEVICE_HIP] = {NULL, POLY_DEVICE_HIP, false, NULL, NULL, NULL, NULL, NULL},
#endif
};

#define N_BACKENDS (sizeof(BACKENDS) / sizeof(BACKENDS[0]))

const PolyBackendDesc *poly_backend_get(PolyDevice device) {
  if (device < 0 || (size_t)device >= N_BACKENDS) return NULL;
  if (!BACKENDS[device].name) return NULL;
  return &BACKENDS[device];
}

int poly_backend_ensure_open(PolyDevice device) {
  const PolyBackendDesc *be = poly_backend_get(device);
  if (!be) return -1;
  return be->ensure_open ? be->ensure_open() : 0;
}

bool poly_device_is_host_addressable(PolyDevice device) {
  const PolyBackendDesc *be = poly_backend_get(device);
  return be && be->get_allocator()->host_addressable;
}

/* Cache flush (called from napi_api.c) */

void poly_sched_cache_flush(void) {
  /* Per-context schedule caches are flushed when context is destroyed.
   * This ABI cleanup hook remains as a no-op for existing frontend cleanup
   * paths that still call it. */
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
  if (poly_backend_ensure_open(device) != 0) {
    fprintf(stderr, "polygrad: compile_schedule: backend '%s' failed to open\n", backend->name);
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

  /* Lower each COMPUTE item via backend vtable. Function names are structural so
   * backend compiler caches see identical kernels as identical source. */
  uint32_t env_stamp = poly_schedule_lower_env_stamp();
  for (int k = 0; k < schedule->n_items; k++) {
    PolyExecItem *item = &schedule->items[k];
    PolyRunner *runner = &plan->runners[k];
    bool lowered_as_copy = false;

    if (item->kind == POLY_EXEC_COPY) {
      if (poly_lower_copy_item(ctx, schedule, item, device, runner) == 0) lowered_as_copy = true;
    }
    if (!lowered_as_copy) {
      if (item->kind != POLY_EXEC_COMPUTE) {
        if (item->kind != POLY_EXEC_COPY) {
          fprintf(stderr, "polygrad: compile_schedule: non-COMPUTE item %d not supported\n", k);
          goto cleanup;
        }
      }

      int lower_rc = poly_lower_compute_item_cached(ctx, item, device, env_stamp, runner);
      if (lower_rc == -2) {
        fprintf(stderr, "polygrad: compile_schedule: kernel %d validation failed\n", k);
        goto cleanup;
      }
      if (lower_rc != 0) {
        fprintf(
            stderr, "polygrad: compile_schedule: backend '%s' failed for kernel %d\n",
            backend->name, k
        );
        goto cleanup;
      }
    }

    if (poly_bind_runner_param_slots(ctx, item, runner, device) != 0) {
      fprintf(stderr, "polygrad: compile_schedule: param remap failed for kernel %d\n", k);
      goto cleanup;
    }

    runner->n_vars = 0;
    runner->var_indices = NULL;
  }

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
      PolyDevice slot_device = poly_schedule_slot_target_device(ctx, schedule, i, device);
      if (slot_device == POLY_DEVICE_HOST || slot_device == POLY_DEVICE_AUTO) slot_device = device;
      const PolyBackendDesc *slot_backend = poly_backend_get(slot_device);
      const PolyAllocator *slot_alloc = slot_backend ? slot_backend->get_allocator() : NULL;
      if (!slot_alloc) goto cleanup;
      void *ptr = slot_alloc->alloc(nbytes, slot_alloc->dev_ctx);
      if (!ptr) goto cleanup;
      plan->intermediates[idx] = (PolyBuffer){
          .ptr = ptr,
          .nbytes = nbytes,
          .device = slot_device,
          .owned = true,
          .allocator = slot_alloc,
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
    PolyDevice hinted = sched->buf_slots[s].device;
    if (hinted != POLY_DEVICE_AUTO && hinted != POLY_DEVICE_HOST &&
        poly_device_can_execute(hinted)) {
      device = hinted;
      break;
    }
  }
  for (int s = 0; sched && s < sched->n_buf_slots; s++) {
    if (device != POLY_DEVICE_AUTO) break;
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
    PolyCtx *ctx,
    PolySchedule *sched,
    PolyDevice device,
    void **slot_data
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
    PolyDevice slot_device = poly_schedule_slot_target_device(ctx, sched, s, device);
    if (slot_device == POLY_DEVICE_AUTO) slot_device = device;
    if (slot_device == POLY_DEVICE_AUTO) slot_device = poly_device_default();
    if (b->device != slot_device && !poly_devices_share_storage(b->device, slot_device)) {
      if (poly_buffer_allocate(ctx, buf_uop, slot_device) != 0) {
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

static void poly_zero_buffer(PolyBuffer *h) {
  if (!h || !h->ptr) return;
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

static void poly_zero_schedule_intermediates(PolySchedule *sched, PolyBuffer *bufs, int n_bufs) {
  if (!sched || !bufs) return;
  int idx = 0;
  for (int s = 0; s < sched->n_buf_slots; s++) {
    if (!sched->buf_slots[s].is_intermediate) continue;
    if (idx >= n_bufs) break;
    if (sched->buf_slots[s].needs_zero) poly_zero_buffer(&bufs[idx]);
    idx++;
  }
}

static int poly_schedule_runtime_prepare(PolyCtx *ctx, PolySchedule *sched, PolyDevice device) {
  if (!ctx || !sched) return -1;

  const PolyBackendDesc *backend = poly_backend_get(device);
  if (!backend || !poly_device_can_execute(device) || !backend->lower_item) {
    fprintf(stderr, "polygrad: run_schedule: unsupported device %d\n", device);
    return -1;
  }
  if (poly_backend_ensure_open(device) != 0) {
    fprintf(stderr, "polygrad: run_schedule: backend '%s' failed to open\n", backend->name);
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
      PolyDevice slot_device = poly_schedule_slot_target_device(ctx, sched, i, device);
      if (slot_device == POLY_DEVICE_HOST || slot_device == POLY_DEVICE_AUTO) slot_device = device;
      const PolyBackendDesc *slot_backend = poly_backend_get(slot_device);
      const PolyAllocator *slot_alloc = slot_backend ? slot_backend->get_allocator() : NULL;
      if (!slot_alloc) goto fail;
      void *ptr = slot_alloc->alloc(nbytes, slot_alloc->dev_ctx);
      if (!ptr) goto fail;
      sched->run_intermediates[idx] = (PolyBuffer){
          .ptr = ptr,
          .nbytes = nbytes,
          .device = slot_device,
          .owned = true,
          .allocator = slot_alloc,
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
    PolySchedule *sched,
    PolyVarBinding *var_bindings,
    int n_var_bindings
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
  if (!schedule->run_slot_to_data && poly_schedule_runtime_prepare(ctx, schedule, device) != 0)
    return -1;

  PolyExecItem *item = &schedule->items[item_index];
  uint32_t env_stamp = poly_schedule_lower_env_stamp();
  bool timing = poly_debug_at_least(7);
  double t0 = timing ? poly_now_ms() : 0.0;
  if (timing) {
    fprintf(
        stderr,
        "[polygrad:exec_lower] begin item=%d/%d kind=%s device=%s root=%p slots=%d vars=%d\n",
        item_index, schedule->n_items, poly_exec_item_kind_name(item->kind),
        poly_device_name(device), (void *)item->root, item->n_buf_slots, item->n_var_uops
    );
    fflush(stderr);
  }
  if (item->prg_valid && item->lowered_device == device && item->lowered_env_stamp == env_stamp)
    return 0;
  if (item->prg_valid) poly_runner_cleanup(&item->prg, item->lowered_device);

  const PolyBackendDesc *backend = poly_backend_get(device);
  if (!backend || !backend->lower_item) return -1;
  if (poly_backend_ensure_open(device) != 0) return -1;
  bool lowered_as_copy = false;

  if (item->kind == POLY_EXEC_COPY) {
    if (poly_lower_copy_item(ctx, schedule, item, device, &item->prg) == 0) lowered_as_copy = true;
  }
  if (!lowered_as_copy) {
    if (item->kind != POLY_EXEC_COMPUTE) {
      if (item->kind != POLY_EXEC_COPY) {
        fprintf(
            stderr, "polygrad: exec_item_lower: non-COMPUTE item %d not supported\n", item_index
        );
        return -1;
      }
    }
    int lower_rc = poly_lower_compute_item_cached(ctx, item, device, env_stamp, &item->prg);
    if (lower_rc == -2) {
      fprintf(stderr, "polygrad: exec_item_lower: kernel %d validation failed\n", item_index);
      return -1;
    }
    if (lower_rc != 0) {
      fprintf(
          stderr, "polygrad: exec_item_lower: backend '%s' failed for kernel %d\n", backend->name,
          item_index
      );
      memset(&item->prg, 0, sizeof(item->prg));
      return -1;
    }
  }

  if (poly_bind_runner_param_slots(ctx, item, &item->prg, device) != 0) {
    poly_runner_cleanup(&item->prg, device);
    fprintf(stderr, "polygrad: exec_item_lower: param remap failed for kernel %d\n", item_index);
    return -1;
  }
  item->prg.n_vars = 0;
  item->prg.var_indices = NULL;

  free(schedule->run_kernel_args[item_index]);
  int n_args = item->prg.n_params + item->n_var_uops;
  schedule->run_kernel_args[item_index] = calloc((size_t)(n_args > 0 ? n_args : 1), sizeof(void *));
  if (!schedule->run_kernel_args[item_index]) {
    poly_runner_cleanup(&item->prg, device);
    return -1;
  }

  item->lowered_device = device;
  item->lowered_env_stamp = env_stamp;
  item->prg_valid = true;
  if (timing) {
    double t1 = poly_now_ms();
    fprintf(
        stderr,
        "[polygrad:exec_lower] done item=%d kind=%s runner=%d params=%d wgsl_or_size=%d ms=%.3f\n",
        item_index, poly_exec_item_kind_name(item->kind), (int)item->prg.kind, item->prg.n_params,
        item->prg.handle_size, t1 - t0
    );
    fflush(stderr);
  }
  return 0;
}

static int poly_exec_item_run_prepared(
    PolySchedule *sched,
    int exec_step,
    int item_index,
    int n_all,
    int *var_int_idx
) {
  PolyExecItem *item = &sched->items[item_index];
  PolyRunner *runner = &item->prg;
  const PolyBackendDesc *backend = poly_backend_get(item->lowered_device);
  if (backend && poly_backend_ensure_open(item->lowered_device) != 0) return -1;
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
    if (!args[i] && !runner_copy_param_allows_null(runner, i)) {
      fprintf(
          stderr, "polygrad: run_schedule: missing data for param %d (slot %d) in kernel %d\n", i,
          slot, item_index
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
        fprintf(
            stderr, "polygrad: run_schedule: no binding for DEFINE_VAR in kernel %d\n", item_index
        );
        return -1;
      }
    }
  }

  if (poly_resolve_runner_launch_dims(runner, sched->run_merged_vars, n_all) != 0) {
    fprintf(
        stderr, "polygrad: run_schedule: failed to resolve launch dims for kernel %d\n", item_index
    );
    return -1;
  }

  debug_dump_webgpu_schedule_args(sched, item->lowered_device, exec_step, item_index, runner, args);
  int ret;
  if (runner->execute) {
    ret = runner->execute(runner, args, n_args);
  } else if (backend && backend->execute) {
    ret = backend->execute(runner, args, n_args);
  } else {
    fprintf(
        stderr, "polygrad: run_schedule: no execute hook for backend '%s' kernel %d\n",
        backend && backend->name ? backend->name : "<unknown>", item_index
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
  if (poly_schedule_runtime_prepare(ctx, schedule, device) != 0) return -1;
  PolyDevice item_device =
      poly_exec_item_device(ctx, schedule, &schedule->items[item_index], device);
  if (poly_exec_item_lower(ctx, schedule, item_index, item_device) != 0) return -1;
  if (poly_fill_schedule_slot_data(ctx, schedule, device, schedule->run_slot_to_data) != 0)
    return -1;
  poly_zero_schedule_intermediates(
      schedule, schedule->run_intermediates, schedule->n_run_intermediates
  );

  int n_all = poly_schedule_merge_runtime_vars(schedule, var_bindings, n_var_bindings);
  int var_int_idx = 0;
  return (n_all < 0) ? -1
                     : poly_exec_item_run_prepared(schedule, 0, item_index, n_all, &var_int_idx);
}

int poly_run_schedule(
    PolyCtx *ctx,
    PolySchedule *schedule,
    PolyVarBinding *var_bindings,
    int n_var_bindings
) {
  if (!ctx || !schedule) return -1;

  bool timing = poly_debug_at_least(2);
  double t0 = timing ? poly_now_ms() : 0.0;
  PolyDevice device = poly_infer_schedule_device(ctx, schedule);
  if (timing) {
    fprintf(
        stderr,
        "[polygrad:run_schedule] begin device=%s items=%d slots=%d intermediates=%d "
        "default_vars=%d runtime_vars=%d\n",
        poly_device_name(device), schedule->n_items, schedule->n_buf_slots,
        schedule->n_run_intermediates, schedule->n_default_vars, n_var_bindings
    );
    fflush(stderr);
  }
  if (poly_schedule_runtime_prepare(ctx, schedule, device) != 0) {
    fprintf(stderr, "polygrad: run_schedule: compile failed\n");
    return -1;
  }
  double t_prepare = timing ? poly_now_ms() : 0.0;
  if (poly_fill_schedule_slot_data(ctx, schedule, device, schedule->run_slot_to_data) != 0)
    return -1;
  double t_slots = timing ? poly_now_ms() : 0.0;

  poly_zero_schedule_intermediates(
      schedule, schedule->run_intermediates, schedule->n_run_intermediates
  );
  double t_zero = timing ? poly_now_ms() : 0.0;

  int n_all = poly_schedule_merge_runtime_vars(schedule, var_bindings, n_var_bindings);
  if (n_all < 0) return -1;
  double t_vars = timing ? poly_now_ms() : 0.0;

  int ret = 0;
  int var_int_idx = 0;
  for (int s = 0; s < schedule->n_items && ret == 0; s++) {
    int k = schedule->exec_order[s];
    PolyDevice item_device = poly_exec_item_device(ctx, schedule, &schedule->items[k], device);
    if (poly_debug_at_least(7)) {
      fprintf(
          stderr, "[polygrad:run_schedule] step=%d/%d item=%d device=%s lower begin\n", s + 1,
          schedule->n_items, k, poly_device_name(item_device)
      );
      fflush(stderr);
    }
    if (poly_exec_item_lower(ctx, schedule, k, item_device) != 0) {
      ret = -1;
      break;
    }
    if (poly_debug_at_least(7)) {
      fprintf(
          stderr, "[polygrad:run_schedule] step=%d/%d item=%d run begin\n", s + 1,
          schedule->n_items, k
      );
      fflush(stderr);
    }
    ret = poly_exec_item_run_prepared(schedule, s, k, n_all, &var_int_idx);
    if (poly_debug_at_least(7)) {
      fprintf(
          stderr, "[polygrad:run_schedule] step=%d/%d item=%d run end ret=%d\n", s + 1,
          schedule->n_items, k, ret
      );
      fflush(stderr);
    }
  }
  if (timing) {
    double t_done = poly_now_ms();
    fprintf(
        stderr,
        "[polygrad:run_schedule] done prepare=%.3fms slots=%.3fms zero=%.3fms vars=%.3fms "
        "loop=%.3fms total=%.3fms ret=%d\n",
        t_prepare - t0, t_slots - t_prepare, t_zero - t_slots, t_vars - t_zero, t_done - t_vars,
        t_done - t0, ret
    );
    fflush(stderr);
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
  if (poly_backend_ensure_open(plan->device) != 0) return -1;

  int ret = 0;

  poly_zero_schedule_intermediates(sched, plan->intermediates, plan->n_intermediates);

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
      if (!args[i] && !runner_copy_param_allows_null(runner, i)) {
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
      if (runner->execute)
        ret = runner->execute(runner, args, n_args);
      else
        ret = backend->execute(runner, args, n_args);
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
      if (r->free_handle)
        r->free_handle(r);
      else if (backend)
        backend->free_runner(r);
    }
    free(r->param_to_slot);
    free(r->var_indices);
  }
  free(plan->runners);

  /* Free persistent intermediates */
  if (plan->intermediates) {
    for (int i = 0; i < plan->n_intermediates; i++) {
      if (plan->intermediates[i].ptr)
        plan->intermediates[i].allocator->free(
            &plan->intermediates[i], plan->intermediates[i].allocator->dev_ctx
        );
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
