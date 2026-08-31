/* Current Tinygrad CALL/END/AFTER schedule linearizer. */

#include "schedule/schedule.h"
#include "codegen/simplify.h"
#include "ctx.h"
#include "device.h"
#include "engine/jit.h"
#include "frontend_internal.h"
#include "uop/upat.h"
#include "schedule/memory.h"
#include "schedule/rangeify.h"
#include "utils.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* C row for current Tinygrad create_schedule's children/in_degree maps. */
typedef struct {
  PolyUOp *uop;
  int in_degree;
  int *children;
  int n_children;
  int cap_children;
} ScheduleKernel;

/* C row for current Tinygrad create_schedule's
 * writes[buffer] value: (AFTER, prior state, new kernels).  `buffer` stores
 * the Python dict key because this implementation uses a flat temporary array. */
typedef struct {
  PolyUOp *after;
  PolyUOp *prev_state;
  PolyUOp *buffer;
  PolyUOp **kernels;
  int n_kernels;
} ScheduleWrite;

/* C row for current Tinygrad create_schedule's reads tuple:
 * (reader AFTER, reader kernel, buffer state read). */
typedef struct {
  PolyUOp *after;
  PolyUOp *kernel;
  PolyUOp *state;
} ScheduleRead;

/* Temporary C storage for current Tinygrad create_schedule's children,
 * in_degree, writes, and reads collections.  No row survives LINEAR creation. */
typedef struct {
  PolyCtx *ctx;
  ScheduleKernel *kernels;
  int n_kernels;
  int cap_kernels;
  PolyMap *kernel_index;
  ScheduleWrite *writes;
  int n_writes;
  int cap_writes;
  ScheduleRead *reads;
  int n_reads;
  int cap_reads;
  bool failed;
} ScheduleContext;

/* tinygrad schedule/__init__.py:9-19. */
static PolyUOp *unwrap_src(PolyUOp *u) {
  while (u && u->n_src > 0 && u->op != POLY_OP_AFTER && u->op != POLY_OP_BUFFER &&
         u->op != POLY_OP_PARAM && u->op != POLY_OP_MSELECT && u->op != POLY_OP_MSTACK)
    u = u->src[0];
  return u;
}

/* C vector growth for the temporary Python lists in create_schedule. */
static bool append_uop(PolyUOp ***items, int *count, int *cap, PolyUOp *u) {
  if (*count >= *cap) {
    int next = *cap ? *cap * 2 : 4;
    PolyUOp **grown = realloc(*items, (size_t)next * sizeof(*grown));
    if (!grown) return false;
    *items = grown;
    *cap = next;
  }
  (*items)[(*count)++] = u;
  return true;
}

/* Current Tinygrad schedule/__init__.py:_states. */
static bool append_states(PolyUOp *u, PolyUOp ***states, int *count, int *cap) {
  u = unwrap_src(u);
  if (!u) return false;
  if (u->op == POLY_OP_MSELECT || u->op == POLY_OP_MSTACK) {
    for (int i = 0; i < u->n_src; i++)
      if (!append_states(u->src[i], states, count, cap)) return false;
    return true;
  }
  if (u->op != POLY_OP_AFTER && u->op != POLY_OP_BUFFER && u->op != POLY_OP_PARAM) return false;
  return append_uop(states, count, cap, u);
}

/* tinygrad schedule/__init__.py:21-26. */
static bool split_after(
    PolyUOp *after,
    PolyUOp ***kernels,
    int *n_kernels,
    PolyUOp ***deps,
    int *n_deps
) {
  int kernels_cap = 0, deps_cap = 0;
  *kernels = NULL;
  *deps = NULL;
  *n_kernels = 0;
  *n_deps = 0;
  if (!after || after->op != POLY_OP_AFTER) return false;
  for (int i = 1; i < after->n_src; i++) {
    PolyUOp *src = after->src[i];
    if (src->op == POLY_OP_CALL || src->op == POLY_OP_END) {
      if (!append_uop(kernels, n_kernels, &kernels_cap, src)) goto fail;
    } else if (src->op == POLY_OP_AFTER) {
      if (!append_uop(deps, n_deps, &deps_cap, src)) goto fail;
    } else if (src->op != POLY_OP_STORE) {
      fprintf(stderr, "polygrad: AFTER source must be CALL, END, STORE, or AFTER\n");
      goto fail;
    }
  }
  return true;
fail:
  free(*kernels);
  free(*deps);
  *kernels = NULL;
  *deps = NULL;
  *n_kernels = 0;
  *n_deps = 0;
  return false;
}

/* C index for current Tinygrad create_schedule's UOp-keyed maps. */
static int kernel_index(ScheduleContext *sctx, PolyUOp *kernel) {
  void *found = poly_map_get(sctx->kernel_index, poly_ptr_hash(kernel), kernel, poly_ptr_eq);
  if (found) return (int)((intptr_t)found - 1);
  if (sctx->n_kernels >= sctx->cap_kernels) {
    int next = sctx->cap_kernels ? sctx->cap_kernels * 2 : 8;
    ScheduleKernel *grown = realloc(sctx->kernels, (size_t)next * sizeof(*grown));
    if (!grown) return -1;
    sctx->kernels = grown;
    sctx->cap_kernels = next;
  }
  int idx = sctx->n_kernels++;
  sctx->kernels[idx] = (ScheduleKernel){.uop = kernel};
  poly_map_set(
      sctx->kernel_index, poly_ptr_hash(kernel), kernel, (void *)(intptr_t)(idx + 1), poly_ptr_eq
  );
  return idx;
}

/* C implementation of children[from].append(to) and in_degree[to] += 1. */
static bool add_edge(ScheduleContext *sctx, PolyUOp *from, PolyUOp *to) {
  int from_idx = kernel_index(sctx, from), to_idx = kernel_index(sctx, to);
  if (from_idx < 0 || to_idx < 0) return false;
  ScheduleKernel *producer = &sctx->kernels[from_idx];
  if (producer->n_children >= producer->cap_children) {
    int next = producer->cap_children ? producer->cap_children * 2 : 4;
    int *grown = realloc(producer->children, (size_t)next * sizeof(*grown));
    if (!grown) return false;
    producer->children = grown;
    producer->cap_children = next;
  }
  producer->children[producer->n_children++] = to_idx;
  sctx->kernels[to_idx].in_degree++;
  return true;
}

/* Append one temporary writes[buffer] row during create_schedule. */
static bool append_write(ScheduleContext *sctx, ScheduleWrite write) {
  if (sctx->n_writes >= sctx->cap_writes) {
    int next = sctx->cap_writes ? sctx->cap_writes * 2 : 8;
    ScheduleWrite *grown = realloc(sctx->writes, (size_t)next * sizeof(*grown));
    if (!grown) return false;
    sctx->writes = grown;
    sctx->cap_writes = next;
  }
  sctx->writes[sctx->n_writes++] = write;
  return true;
}

/* Append one temporary reads tuple during create_schedule. */
static bool append_read(ScheduleContext *sctx, ScheduleRead read) {
  if (sctx->n_reads >= sctx->cap_reads) {
    int next = sctx->cap_reads ? sctx->cap_reads * 2 : 16;
    ScheduleRead *grown = realloc(sctx->reads, (size_t)next * sizeof(*grown));
    if (!grown) return false;
    sctx->reads = grown;
    sctx->cap_reads = next;
  }
  sctx->reads[sctx->n_reads++] = read;
  return true;
}

/* C set-membership helper for current Tinygrad's prev_kernels set. */
static bool kernel_in_list(PolyUOp *kernel, PolyUOp **items, int count) {
  for (int i = 0; i < count; i++)
    if (items[i] == kernel) return true;
  return false;
}

/* Current Tinygrad create_schedule first loop: collect RAW facts for one AFTER. */
static bool collect_after(ScheduleContext *sctx, PolyUOp *after) {
  PolyUOp **kernels = NULL, **after_deps = NULL, **prev_kernels = NULL, **prev_deps = NULL;
  int n_kernels = 0, n_after_deps = 0, n_prev_kernels = 0, n_prev_deps = 0;
  if (!split_after(after, &kernels, &n_kernels, &after_deps, &n_after_deps)) return false;

  PolyUOp *prev_state = unwrap_src(after->src[0]);
  if (!prev_state) goto fail;
  if (prev_state->op == POLY_OP_AFTER &&
      !split_after(prev_state, &prev_kernels, &n_prev_kernels, &prev_deps, &n_prev_deps))
    goto fail;
  free(prev_deps);
  prev_deps = NULL;

  PolyUOp **new_kernels = n_kernels > 0 ? malloc((size_t)n_kernels * sizeof(*new_kernels)) : NULL;
  int n_new_kernels = 0;
  if (n_kernels > 0 && !new_kernels) goto fail;
  for (int i = 0; i < n_kernels; i++)
    if (!kernel_in_list(kernels[i], prev_kernels, n_prev_kernels))
      new_kernels[n_new_kernels++] = kernels[i];

  PolyUOp *write_buf = poly_uop_buf_uop(sctx->ctx, after);
  if (!write_buf ||
      !append_write(
          sctx, (ScheduleWrite){after, prev_state, write_buf, new_kernels, n_new_kernels}
      )) {
    free(new_kernels);
    goto fail;
  }

  for (int i = 0; i < n_kernels; i++) {
    PolyUOp *kernel = kernels[i];
    if (kernel_index(sctx, kernel) < 0) goto fail;
    PolyUOp *call = kernel;
    if (kernel->op == POLY_OP_END) {
      if (kernel->n_src < 1 || kernel->src[0]->op != POLY_OP_CALL) {
        fprintf(stderr, "polygrad: END src[0] must be CALL\n");
        goto fail;
      }
      call = kernel->src[0];
    }

    PolyUOp **read_states = NULL;
    int n_read_states = 0, cap_read_states = 0;
    for (int j = 1; j < call->n_src; j++) {
      if (!append_states(call->src[j], &read_states, &n_read_states, &cap_read_states)) {
        free(read_states);
        goto fail;
      }
    }
    for (int j = 0; j < n_read_states; j++) {
      if (!append_read(sctx, (ScheduleRead){after, kernel, read_states[j]})) {
        free(read_states);
        goto fail;
      }
    }

    PolyUOp **ordered_states = read_states;
    int n_ordered_states = n_read_states, cap_ordered_states = cap_read_states;
    for (int j = 0; j < n_after_deps; j++) {
      if (!append_states(after_deps[j], &ordered_states, &n_ordered_states, &cap_ordered_states)) {
        free(ordered_states);
        goto fail;
      }
    }
    for (int j = 0; j < n_ordered_states; j++) {
      PolyUOp *state = ordered_states[j];
      if (state->op != POLY_OP_AFTER) continue;
      PolyUOp **producers = NULL, **unused_deps = NULL;
      int n_producers = 0, n_unused_deps = 0;
      if (!split_after(state, &producers, &n_producers, &unused_deps, &n_unused_deps)) {
        free(ordered_states);
        goto fail;
      }
      for (int k = 0; k < n_producers; k++)
        if (!add_edge(sctx, producers[k], kernel)) sctx->failed = true;
      free(producers);
      free(unused_deps);
      if (sctx->failed) {
        free(ordered_states);
        goto fail;
      }
    }
    free(ordered_states);
  }

  free(kernels);
  free(after_deps);
  free(prev_kernels);
  free(prev_deps);
  return true;

fail:
  free(kernels);
  free(after_deps);
  free(prev_kernels);
  free(prev_deps);
  return false;
}

/* Current Tinygrad create_schedule WAR loop over reads and writes. */
static bool add_war_edges(ScheduleContext *sctx) {
  for (int i = 0; i < sctx->n_reads; i++) {
    ScheduleRead *read = &sctx->reads[i];
    PolyUOp *read_buf = poly_uop_buf_uop(sctx->ctx, read->state);
    if (!read_buf) return false;
    for (int j = 0; j < sctx->n_writes; j++) {
      ScheduleWrite *write = &sctx->writes[j];
      if (write->buffer != read_buf || write->after == read->after ||
          write->prev_state != read->state)
        continue;
      for (int k = 0; k < write->n_kernels; k++) {
        PolyUOp *writer = write->kernels[k];
        if (writer == read->kernel || poly_uop_reachable(sctx->ctx, read->kernel, writer)) continue;
        if (!add_edge(sctx, read->kernel, writer)) return false;
      }
    }
  }
  return true;
}

/* Current Tinygrad create_schedule linearization: normalize one queued kernel
 * into its resolved CALL occurrence and append it to LINEAR.src. */
static bool append_linear_call(
    PolyCtx *ctx,
    PolyUOp *kernel,
    PolyUOp ***calls,
    int *n_calls,
    int *cap_calls
) {
  if (kernel->op == POLY_OP_LINEAR) {
    for (int i = 0; i < kernel->n_src; i++)
      if (!append_uop(calls, n_calls, cap_calls, kernel->src[i])) return false;
    return true;
  }
  PolyUOp *call = kernel->op == POLY_OP_END && kernel->n_src > 0 ? kernel->src[0] : kernel;
  if (!call || call->op != POLY_OP_CALL || call->n_src < 1) return false;

  PolyUOp **src = malloc((size_t)call->n_src * sizeof(*src));
  if (!src) return false;
  int n_src = 1;
  src[0] = call->src[0];
  for (int i = 1; i < call->n_src; i++) {
    if (poly_uop_is_bound_var(call->src[i])) continue;
    src[n_src] = poly_uop_buf_uop(ctx, unwrap_src(call->src[i]));
    if (!src[n_src]) {
      free(src);
      return false;
    }
    n_src++;
  }
  PolyUOp *resolved = poly_uop(ctx, POLY_OP_CALL, POLY_VOID, src, n_src, poly_arg_none());
  free(src);
  return resolved && append_uop(calls, n_calls, cap_calls, resolved);
}

/* Free only create_schedule's temporary C collections. */
static void schedule_context_destroy(ScheduleContext *sctx) {
  for (int i = 0; i < sctx->n_kernels; i++)
    free(sctx->kernels[i].children);
  for (int i = 0; i < sctx->n_writes; i++)
    free(sctx->writes[i].kernels);
  free(sctx->kernels);
  free(sctx->writes);
  free(sctx->reads);
  if (sctx->kernel_index) poly_map_destroy(sctx->kernel_index);
}

/* Current Tinygrad schedule/__init__.py:create_schedule. */
PolyUOp *poly_create_schedule(PolyCtx *ctx, PolyUOp *kernel_graph) {
  if (!ctx || !kernel_graph || kernel_graph->op != POLY_OP_SINK) return NULL;
  ScheduleContext sctx = {.ctx = ctx, .kernel_index = poly_map_new(64)};
  if (!sctx.kernel_index) return NULL;

  int n_topo = 0;
  PolyUOp **topo = poly_toposort_ex_alloc(ctx, kernel_graph, &n_topo, NULL, false);
  if (!topo) goto fail;
  for (int i = 0; i < n_topo; i++) {
    if (topo[i]->op == POLY_OP_AFTER && !collect_after(&sctx, topo[i])) goto fail;
  }
  if (!add_war_edges(&sctx)) goto fail;

  int *queue = sctx.n_kernels > 0 ? malloc((size_t)sctx.n_kernels * sizeof(*queue)) : NULL;
  if (sctx.n_kernels > 0 && !queue) goto fail;
  int qhead = 0, qtail = 0;
  for (int i = 0; i < sctx.n_kernels; i++)
    if (sctx.kernels[i].in_degree == 0) queue[qtail++] = i;

  PolyUOp **calls = NULL;
  int n_calls = 0, cap_calls = 0, n_ordered = 0;
  while (qhead < qtail) {
    int idx = queue[qhead++];
    ScheduleKernel *kernel = &sctx.kernels[idx];
    n_ordered++;
    if (!append_linear_call(ctx, kernel->uop, &calls, &n_calls, &cap_calls)) {
      free(queue);
      free(calls);
      goto fail;
    }
    for (int i = 0; i < kernel->n_children; i++) {
      int child = kernel->children[i];
      if (--sctx.kernels[child].in_degree == 0) queue[qtail++] = child;
    }
  }
  free(queue);
  if (n_ordered != sctx.n_kernels) {
    fprintf(stderr, "polygrad: cycle detected in assign graph\n");
    free(calls);
    goto fail;
  }

  PolyUOp *linear = poly_uop(ctx, POLY_OP_LINEAR, POLY_VOID, calls, n_calls, poly_arg_none());
  free(calls);
  poly_toposort_free(topo);
  schedule_context_destroy(&sctx);
  return linear;

fail:
  poly_toposort_free(topo);
  schedule_context_destroy(&sctx);
  return NULL;
}

/* Current Tinygrad schedule/__init__.py:116-145. KernelInfo-backed SINKs are
 * already compiler kernels; every other SINK is recursively scheduled. */
static PolyUOp *lower_sink_to_linear(
    PolyCtx *ctx,
    PolyUOp *function,
    const PolyBindings *bindings
) {
  (void)bindings;
  if (!ctx || !function || function->op != POLY_OP_SINK ||
      function->arg.kind == POLY_ARG_KERNEL_INFO)
    return NULL;
  bool use_cache = poly_getenv_flag_default("SCACHE", true);
  PolyUOp *linear =
      use_cache ? poly_map_get(ctx->schedule_cache, poly_ptr_hash(function), function, poly_ptr_eq)
                : NULL;
  if (linear) return linear;
  PolyUOp *kernel_graph = poly_get_kernel_graph(ctx, function);
  linear = kernel_graph ? poly_create_schedule(ctx, kernel_graph) : NULL;
  if (linear && use_cache)
    poly_map_set(ctx->schedule_cache, poly_ptr_hash(function), function, linear, poly_ptr_eq);
  return linear;
}

/* Current Tinygrad schedule/__init__.py:143-145 pm_schedule. */
static PolyPatternMatcher *pm_schedule(void) {
  static _Thread_local PolyPatternMatcher *pm = NULL;
  if (pm) return pm;
  PolyRule rules[] = {{poly_upat_op(POLY_OP_SINK, NULL, 0, "function"), lower_sink_to_linear}};
  pm = poly_pm_thread_cache(poly_pm_new(rules, 1));
  return pm;
}

/* C rebuild mechanics for current graph_rewrite/substitute operations. */
static PolyUOp *rebuild_uop(PolyCtx *ctx, PolyUOp *u, PolyUOp **src) {
  if (u->tag || u->tag_arg.kind != POLY_ARG_NONE)
    return poly_uop_tagged_arg(
        ctx, u->op, poly_rebuild_dtype(u, src), src, u->n_src, u->arg, u->tag, u->tag_arg
    );
  return poly_uop(ctx, u->op, poly_rebuild_dtype(u, src), src, u->n_src, u->arg);
}

/* Current Tinygrad schedule/__init__.py:create_new_buffer. */
static PolyUOp *create_new_buffer(PolyCtx *ctx, PolyUOp *buffer) {
  if (!ctx || !buffer || buffer->op != POLY_OP_BUFFER || buffer->n_src != 1 ||
      buffer->arg.kind != POLY_ARG_PARAM || !buffer->arg.param ||
      buffer->arg.param->addrspace != POLY_ADDR_GLOBAL)
    return NULL;
  int64_t size = poly_uop_max_numel(ctx, buffer);
  if (size < 0) return NULL;
  PolyUOp *device = poly_uop_device_uop_cached(ctx, buffer, NULL);
  return device
             ? poly_uop_new_buffer(ctx, device, size, buffer->dtype, poly_ctx_next_unique_id(ctx))
             : NULL;
}

static bool is_buffer_template(PolyUOp *u) {
  return u && u->op == POLY_OP_BUFFER && u->n_src == 1 && u->arg.kind == POLY_ARG_PARAM &&
         u->arg.param && u->arg.param->addrspace == POLY_ADDR_GLOBAL;
}

static bool param_slot(PolyUOp *u, int *slot) {
  if (!u || u->op != POLY_OP_PARAM) return false;
  if (u->arg.kind == POLY_ARG_PARAM && u->arg.param && u->arg.param->slot >= 0) {
    *slot = (int)u->arg.param->slot;
    return true;
  }
  if (u->arg.kind == POLY_ARG_INT && u->arg.i >= 0) {
    *slot = (int)u->arg.i;
    return true;
  }
  return false;
}

/* Current pm_post_sched_cache: resolve positional PARAMs and allocate one
 * concrete BUFFER per template BUFFER across the complete inner LINEAR. */
static PolyUOp *resolve_schedule_arg(
    PolyCtx *ctx,
    PolyUOp *u,
    PolyUOp **args,
    int n_args,
    PolyMap *buffers,
    PolyMap *memo
) {
  PolyUOp *cached = poly_map_get(memo, poly_ptr_hash(u), u, poly_ptr_eq);
  if (cached) return cached;

  PolyUOp *ret = u;
  int slot = -1;
  if (param_slot(u, &slot)) {
    ret = slot < n_args ? args[slot] : NULL;
  } else if (is_buffer_template(u)) {
    ret = poly_map_get(buffers, poly_ptr_hash(u), u, poly_ptr_eq);
    if (!ret) {
      ret = create_new_buffer(ctx, u);
      if (ret) poly_map_set(buffers, poly_ptr_hash(u), u, ret, poly_ptr_eq);
    }
  } else if (u->n_src > 0) {
    PolyUOp *src_stack[16];
    PolyUOp **src = u->n_src > 16 ? malloc((size_t)u->n_src * sizeof(*src)) : src_stack;
    if (!src) return NULL;
    bool changed = false;
    int first = (u->op == POLY_OP_CALL || u->op == POLY_OP_FUNCTION) ? 1 : 0;
    if (first) src[0] = u->src[0];
    for (int i = first; i < u->n_src; i++) {
      src[i] = resolve_schedule_arg(ctx, u->src[i], args, n_args, buffers, memo);
      if (!src[i]) {
        if (src != src_stack) free(src);
        return NULL;
      }
      changed |= src[i] != u->src[i];
    }
    if (changed) ret = rebuild_uop(ctx, u, src);
    if (src != src_stack) free(src);
  }
  if (ret) poly_map_set(memo, poly_ptr_hash(u), u, ret, poly_ptr_eq);
  return ret;
}

typedef struct {
  int slot;
  PolyUOp *value;
} LinearBind;

static bool append_bind(LinearBind **binds, int *count, int *capacity, int slot, PolyUOp *value) {
  for (int i = 0; i < *count; i++) {
    if ((*binds)[i].slot != slot) continue;
    (*binds)[i].value = value;
    return true;
  }
  if (*count >= *capacity) {
    int next = *capacity ? *capacity * 2 : 8;
    LinearBind *grown = realloc(*binds, (size_t)next * sizeof(*grown));
    if (!grown) return false;
    *binds = grown;
    *capacity = next;
  }
  (*binds)[(*count)++] = (LinearBind){slot, value};
  return true;
}

static PolyUOp *bind_value(PolyCtx *ctx, PolyUOp *bound) {
  if (!poly_uop_is_bound_var(bound)) return NULL;
  PolyUOp *variable = bound->src[0];
  return poly_uop(
      ctx, POLY_OP_PARAM, variable->dtype, variable->src, variable->n_src, variable->arg
  );
}

static int bind_slot(PolyUOp *u, LinearBind *binds, int n_binds) {
  const char *expr = poly_uop_is_alu_param(u) ? poly_uop_expr(u) : NULL;
  if (!expr) return -1;
  for (int i = 0; i < n_binds; i++) {
    char expected[32];
    snprintf(expected, sizeof(expected), "p%d", binds[i].slot);
    if (strcmp(expr, expected) == 0) return i;
  }
  return -1;
}

/* Current resolve_linear_call.apply_binds substitutes scalar PARAMs inside
 * each non-nested CALL source while CALL bodies remain lexical scopes. */
static PolyUOp *apply_binds(PolyCtx *ctx, PolyUOp *call, LinearBind *binds, int n_binds) {
  if (n_binds == 0) return call;
  PolyUOp **from = NULL, **to = NULL;
  int n_sub = 0, cap = 0;
  for (int i = 0; i < call->n_src; i++) {
    int n_topo = 0;
    PolyUOp **topo = poly_toposort_ex_alloc(ctx, call->src[i], &n_topo, NULL, false);
    if (!topo) goto fail;
    for (int j = 0; j < n_topo; j++) {
      int found = bind_slot(topo[j], binds, n_binds);
      if (found < 0) continue;
      bool present = false;
      for (int k = 0; k < n_sub; k++)
        present |= from[k] == topo[j];
      if (present) continue;
      if (n_sub >= cap) {
        int next = cap ? cap * 2 : 8;
        PolyUOp **grown_from = realloc(from, (size_t)next * sizeof(*grown_from));
        if (!grown_from) {
          poly_toposort_free(topo);
          goto fail;
        }
        from = grown_from;
        PolyUOp **grown_to = realloc(to, (size_t)next * sizeof(*grown_to));
        if (!grown_to) {
          poly_toposort_free(topo);
          goto fail;
        }
        to = grown_to;
        cap = next;
      }
      from[n_sub] = topo[j];
      to[n_sub++] = binds[found].value;
    }
    poly_toposort_free(topo);
  }
  PolyUOp *ret = call;
  if (n_sub) {
    PolyUOp **src = malloc((size_t)call->n_src * sizeof(*src));
    if (!src) goto fail;
    bool changed = false;
    for (int i = 0; i < call->n_src; i++) {
      src[i] = poly_uop_substitute(ctx, call->src[i], from, to, n_sub);
      if (!src[i]) {
        free(src);
        goto fail;
      }
      changed |= src[i] != call->src[i];
    }
    ret = changed ? rebuild_uop(ctx, call, src) : call;
    free(src);
  }
  free(from);
  free(to);
  return ret;

fail:
  free(from);
  free(to);
  return NULL;
}

/* Current Tinygrad schedule/__init__.py:100-110 resolve_linear_call. */
static PolyUOp *resolve_linear_call(
    PolyCtx *ctx,
    PolyUOp *linear_call,
    LinearBind *outer_binds,
    int n_outer_binds
) {
  if (!ctx || !linear_call || linear_call->op != POLY_OP_CALL || linear_call->n_src < 1 ||
      linear_call->src[0]->op != POLY_OP_LINEAR)
    return NULL;

  PolyMap *buffers = poly_map_new(32), *memo = poly_map_new(128);
  if (!buffers || !memo) {
    if (buffers) poly_map_destroy(buffers);
    if (memo) poly_map_destroy(memo);
    return NULL;
  }
  PolyUOp *linear = resolve_schedule_arg(
      ctx, linear_call->src[0], linear_call->n_src > 1 ? &linear_call->src[1] : NULL,
      linear_call->n_src - 1, buffers, memo
  );
  poly_map_destroy(buffers);
  poly_map_destroy(memo);
  if (!linear || linear->op != POLY_OP_LINEAR) return NULL;

  LinearBind *binds = NULL;
  int n_binds = 0, cap_binds = 0;
  for (int i = 0; i < n_outer_binds; i++)
    if (!append_bind(&binds, &n_binds, &cap_binds, outer_binds[i].slot, outer_binds[i].value))
      goto fail;
  for (int i = 1; i < linear_call->n_src; i++) {
    PolyUOp *value = bind_value(ctx, linear_call->src[i]);
    if (value && !append_bind(&binds, &n_binds, &cap_binds, i - 1, value)) goto fail;
  }

  PolyUOp **calls = NULL;
  int n_calls = 0, cap_calls = 0;
  for (int i = 0; i < linear->n_src; i++) {
    PolyUOp *call = linear->src[i];
    if (!call) goto fail_calls;
    if (call->op == POLY_OP_CALL && call->n_src > 0 && call->src[0]->op == POLY_OP_LINEAR) {
      PolyUOp *nested = resolve_linear_call(ctx, call, binds, n_binds);
      if (!nested) goto fail_calls;
      for (int j = 0; j < nested->n_src; j++)
        if (!append_uop(&calls, &n_calls, &cap_calls, nested->src[j])) goto fail_calls;
    } else {
      PolyUOp *resolved = apply_binds(ctx, call, binds, n_binds);
      if (!resolved || !append_uop(&calls, &n_calls, &cap_calls, resolved)) goto fail_calls;
    }
  }
  PolyUOp *ret = poly_uop(ctx, POLY_OP_LINEAR, POLY_VOID, calls, n_calls, linear->arg);
  free(calls);
  free(binds);
  return ret;

fail_calls:
  free(calls);
fail:
  free(binds);
  return NULL;
}

static bool linear_var_eq(PolyUOp *a, PolyUOp *b) {
  const char *an = poly_uop_expr(a), *bn = poly_uop_expr(b);
  return a == b || (an && bn && strcmp(an, bn) == 0);
}

static bool append_var(PolyUOp ***vars, int *count, int *capacity, PolyUOp *var) {
  for (int i = 0; i < *count; i++)
    if (linear_var_eq((*vars)[i], var)) return true;
  if (*count >= *capacity) {
    int next = *capacity ? *capacity * 2 : 8;
    PolyUOp **grown = realloc(*vars, (size_t)next * sizeof(*grown));
    if (!grown) return false;
    *vars = grown;
    *capacity = next;
  }
  (*vars)[(*count)++] = var;
  return true;
}

static bool collect_used_vars(PolyCtx *ctx, PolyUOp *linear, PolyUOp ***vars_out, int *count_out) {
  PolyUOp **vars = NULL;
  int count = 0, capacity = 0;
  for (int i = 0; i < linear->n_src; i++) {
    PolyUOp *call = linear->src[i];
    if (!call || call->op != POLY_OP_CALL || call->n_src < 1) goto fail;
    int n_topo = 0;
    PolyUOp **topo = poly_toposort_alloc(ctx, call->src[0], &n_topo);
    if (!topo) goto fail;
    bool ok = true;
    for (int j = 0; j < n_topo; j++)
      if (poly_uop_is_alu_param(topo[j]) && !append_var(&vars, &count, &capacity, topo[j])) {
        ok = false;
        break;
      }
    poly_toposort_free(topo);
    if (!ok) goto fail;
  }
  *vars_out = vars;
  *count_out = count;
  return true;
fail:
  free(vars);
  return false;
}

static bool collect_bindings(
    PolyUOp *big_call,
    PolyUOp **used_vars,
    int n_used_vars,
    PolyVarBinding **bindings_out,
    int *count_out
) {
  PolyVarBinding *bindings = n_used_vars ? calloc((size_t)n_used_vars, sizeof(*bindings)) : NULL;
  if (n_used_vars && !bindings) return false;
  int count = 0;
  for (int i = 1; i < big_call->n_src; i++) {
    PolyUOp *bound = big_call->src[i];
    if (!poly_uop_is_bound_var(bound)) continue;
    PolyUOp *var = bound->src[0], *value = bound->src[1]->src[1], *used = NULL;
    if (!value || value->op != POLY_OP_CONST || value->arg.kind != POLY_ARG_INT) goto fail;
    for (int j = 0; j < n_used_vars; j++)
      if (linear_var_eq(used_vars[j], var)) {
        used = used_vars[j];
        break;
      }
    if (!used) continue;
    int existing = -1;
    for (int j = 0; j < count; j++)
      if (linear_var_eq(bindings[j].var, used)) existing = j;
    if (existing >= 0) {
      if (bindings[existing].value != value->arg.i) goto fail;
    } else {
      bindings[count++] = (PolyVarBinding){used, value->arg.i};
    }
  }
  *bindings_out = bindings;
  *count_out = count;
  return true;
fail:
  free(bindings);
  return false;
}

/* Current Tinygrad schedule/__init__.py:create_linear_with_vars. */
PolyUOp *poly_create_linear_with_vars(
    PolyCtx *ctx,
    PolyUOp *big_call,
    PolyVarBinding **var_bindings_out,
    int *n_var_bindings_out
) {
  if (!ctx || !big_call || !var_bindings_out || !n_var_bindings_out) return NULL;
  *var_bindings_out = NULL;
  *n_var_bindings_out = 0;

  /* tinygrad@2026-08-22/a9069c177a9d schedule/__init__.py:183-186 keeps
   * linear_call for held-buffer discovery after resolving its LINEAR body. */
  PolyUOp *linear_call =
      poly_graph_rewrite_ctx_ex2(ctx, big_call, pm_schedule(), NULL, false, true);
  if (!linear_call) return NULL;
  PolyUOp *linear = NULL;
  if (linear_call->op == POLY_OP_LINEAR)
    linear = linear_call;
  else if (linear_call->op == POLY_OP_CALL && linear_call->n_src >= 1 && linear_call->src[0]->op == POLY_OP_LINEAR)
    linear = resolve_linear_call(ctx, linear_call, NULL, 0);
  if (!linear || linear->op != POLY_OP_LINEAR) return NULL;
  linear = poly_copy_from_store(ctx, linear);
  if (!linear) return NULL;

  PolyUOp **used_vars = NULL;
  int n_used_vars = 0;
  if (!collect_used_vars(ctx, linear, &used_vars, &n_used_vars)) return NULL;
  PolyVarBinding *bindings = NULL;
  int n_bindings = 0;
  if (!collect_bindings(big_call, used_vars, n_used_vars, &bindings, &n_bindings)) {
    free(used_vars);
    return NULL;
  }
  free(used_vars);

  /* Current Tinygrad schedule/__init__.py:create_linear_with_vars records the
   * resolved LINEAR and returns LINEAR(), so capture never executes it. */
  if (ctx->active_jit_capture) {
    if (poly_jit_record_linear(ctx->active_jit_capture, linear, bindings, n_bindings) != 0) {
      free(bindings);
      return NULL;
    }
    *var_bindings_out = bindings;
    *n_var_bindings_out = n_bindings;
    return poly_uop0(ctx, POLY_OP_LINEAR, POLY_VOID, poly_arg_none());
  }

  PolyUOp **held = NULL;
  int n_held = 0;
  if (linear_call->op == POLY_OP_CALL && linear_call->n_src > 1) {
    held = malloc((size_t)(linear_call->n_src - 1) * sizeof(*held));
    if (!held) {
      free(bindings);
      return NULL;
    }
    for (int i = 1; i < linear_call->n_src; i++)
      if (linear_call->src[i] && linear_call->src[i]->op == POLY_OP_BUFFER)
        held[n_held++] = linear_call->src[i];
  }
  PolyUOp *planned = poly_memory_plan_rewrite(ctx, linear, held, n_held);
  free(held);
  if (!planned) {
    free(bindings);
    return NULL;
  }
  *var_bindings_out = bindings;
  *n_var_bindings_out = n_bindings;
  return planned;
}

static bool same_copy_index(PolyUOp *left, PolyUOp *right) {
  if (!left || !right) return false;
  if (left == right) return true;
  return left->op == POLY_OP_CONST && right->op == POLY_OP_CONST &&
         left->arg.kind == POLY_ARG_INT && right->arg.kind == POLY_ARG_INT && left->arg.i == 0 &&
         right->arg.i == 0;
}

/* Current Tinygrad schedule/__init__.py:assert_all_same_devices. */
static bool assert_all_same_devices(PolyCtx *ctx, PolyUOp *ast) {
  int n_topo = 0;
  PolyUOp **topo = poly_toposort_alloc(ctx, ast, &n_topo);
  if (!topo) return false;
  PolyUOp *device = NULL;
  bool ok = true;
  for (int i = 0; i < n_topo; i++) {
    if (topo[i]->op != POLY_OP_PARAM) continue;
    PolyUOp *current = poly_uop_device_uop_cached(ctx, topo[i], NULL);
    if (!current) continue;
    if (!device)
      device = current;
    else if (device != current) {
      fprintf(stderr, "polygrad: all buffers must be on the same device\n");
      ok = false;
      break;
    }
  }
  poly_toposort_free(topo);
  return ok;
}

/* Current tinygrad schedule/__init__.py:simplify_copy_kernel. */
static PolyUOp *simplify_copy_kernel(PolyCtx *ctx, PolyUOp *sink) {
  PolyPatternMatcher *symbolic_mops = poly_pm_concat(poly_symbolic(), poly_pm_mops());
  PolyPatternMatcher *with_flatten =
      symbolic_mops ? poly_pm_concat(symbolic_mops, poly_pm_flatten_range()) : NULL;
  PolyPatternMatcher *copy_simplify =
      with_flatten ? poly_pm_concat(with_flatten, poly_pm_simplify_ranges()) : NULL;
  poly_pm_destroy(symbolic_mops);
  poly_pm_destroy(with_flatten);
  if (!copy_simplify) return NULL;
  PolyMap *range_ctx = poly_map_new(16);
  PolyUOp *ret = poly_graph_rewrite_ctx(ctx, sink, copy_simplify, range_ctx);
  poly_map_destroy(range_ctx);
  poly_pm_destroy(copy_simplify);
  return ret;
}

/* Current tinygrad schedule/__init__.py:copy_kernel_to_copy_uop. */
static PolyUOp *copy_kernel_to_copy_uop(PolyCtx *ctx, PolyUOp *call) {
  if (!ctx || !call || call->op != POLY_OP_CALL || call->n_src < 3 || !call->src[0] ||
      call->src[0]->op != POLY_OP_SINK)
    return call;
  PolyUOp *dst_uop = call->src[1], *src_uop = call->src[2];
  PolyUOp *dst_device = poly_uop_device_uop_cached(ctx, dst_uop, NULL);
  PolyUOp *src_device = poly_uop_device_uop_cached(ctx, src_uop, NULL);
  bool same_device = dst_device == src_device;
  bool disk = dst_device && dst_device->arg.kind == POLY_ARG_STRING &&
              strncmp(dst_device->arg.str, "DISK", 4) == 0;
  if (same_device && !disk) return assert_all_same_devices(ctx, call->src[0]) ? call : NULL;

  PolyUOp *sink = simplify_copy_kernel(ctx, call->src[0]);
  if (!sink || sink->op != POLY_OP_SINK || sink->n_src != 1)
    return assert_all_same_devices(ctx, call->src[0]) ? call : NULL;
  PolyUOp *effect = sink->src[0], *range = NULL;
  if (effect->op == POLY_OP_END) {
    if (effect->n_src != 2 || !effect->src[0] || !effect->src[1] ||
        effect->src[1]->op != POLY_OP_RANGE)
      return assert_all_same_devices(ctx, sink) ? call : NULL;
    range = effect->src[1];
    effect = effect->src[0];
  }
  if (!effect || effect->op != POLY_OP_STORE || effect->n_src != 2)
    return assert_all_same_devices(ctx, sink) ? call : NULL;
  PolyUOp *dst = effect->src[0], *src = effect->src[1];
  if (!dst || !src || dst->op != POLY_OP_INDEX || src->op != POLY_OP_INDEX || dst->n_src != 2 ||
      src->n_src != 2 || !dst->src[0] || !src->src[0] || dst->src[0]->op != POLY_OP_PARAM ||
      src->src[0]->op != POLY_OP_PARAM || !same_copy_index(dst->src[1], src->src[1]) ||
      (range && dst->src[1] != range))
    return assert_all_same_devices(ctx, sink) ? call : NULL;
  PolyUOp *dst_param = dst->src[0], *src_param = src->src[0];
  if (dst_param->arg.kind != POLY_ARG_PARAM || !dst_param->arg.param ||
      src_param->arg.kind != POLY_ARG_PARAM || !src_param->arg.param ||
      dst_param->arg.param->slot != 0 || src_param->arg.param->slot != 1 || !dst_device)
    return assert_all_same_devices(ctx, sink) ? call : NULL;

  PolyUOp *copy = poly_copy_to_device_uop(ctx, src_param, dst_device);
  if (!copy) return NULL;
  PolyUOp **call_src = malloc((size_t)call->n_src * sizeof(*call_src));
  if (!call_src) return NULL;
  memcpy(call_src, call->src, (size_t)call->n_src * sizeof(*call_src));
  call_src[0] = copy;
  PolyUOp *ret = poly_uop(ctx, POLY_OP_CALL, call->dtype, call_src, call->n_src, call->arg);
  free(call_src);
  return ret;
}

/* Current tinygrad schedule/__init__.py:pm_copy_from_store. Cross-device
 * SINKs must prove one exact full-buffer copy; other cross-device kernels are
 * rejected instead of entering shared codegen. */
PolyUOp *poly_copy_from_store(PolyCtx *ctx, PolyUOp *linear) {
  if (!ctx || !linear || linear->op != POLY_OP_LINEAR) return NULL;
  PolyUOp **calls = linear->n_src > 0 ? malloc((size_t)linear->n_src * sizeof(*calls)) : NULL;
  if (linear->n_src > 0 && !calls) return NULL;
  bool changed = false;
  for (int i = 0; i < linear->n_src; i++) {
    calls[i] = copy_kernel_to_copy_uop(ctx, linear->src[i]);
    if (!calls[i]) {
      free(calls);
      return NULL;
    }
    changed |= calls[i] != linear->src[i];
  }
  PolyUOp *ret = changed
                     ? poly_uop(ctx, POLY_OP_LINEAR, POLY_VOID, calls, linear->n_src, linear->arg)
                     : linear;
  free(calls);
  return ret;
}

size_t poly_schedule_cache_len(PolyCtx *ctx) {
  return ctx && ctx->schedule_cache ? poly_map_len(ctx->schedule_cache) : 0;
}
