/* jit.c -- tinygrad-style JIT capture/replay for raw Tensor realizes. */

#include "engine/jit.h"
#include "codegen.h"
#include "ctx.h"
#include "device.h"
#include "frontend_internal.h"
#include "pat.h"
#include "tensor.h"
#include "utils.h"

#include <limits.h>
#include <stdlib.h>

struct PolyJit {
  PolyCtx *ctx;
  PolyUOp **input_buffers;
  PolyUOp **input_views;
  PolyDType *input_dtypes;
  PolyUOp **input_devices;
  int n_inputs;
  PolySchedule **schedules;
  int n_schedules;
  int schedules_cap;
  int n_recorded_schedules;
  PolySchedule *captured_linear;
  PolyCompiledSchedule *compiled_linear;
  PolyDevice compiled_device;
  bool prune;
  bool capturing;
  bool captured;
  PolyUOp *graphed_linear;
};

static int poly_jit_run_captured_linear(
    PolyJit *jit,
    PolyUOp **current_inputs,
    PolyVarBinding *var_bindings,
    int n_var_bindings
);

static void poly_jit_clear(PolyJit *jit) {
  if (!jit) return;
  if (jit->schedules) {
    for (int i = 0; i < jit->n_schedules; i++)
      poly_schedule_free(jit->schedules[i]);
  }
  free(jit->schedules);
  poly_compiled_schedule_free(jit->compiled_linear);
  poly_schedule_free(jit->captured_linear);
  free(jit->input_buffers);
  free(jit->input_views);
  free(jit->input_dtypes);
  free(jit->input_devices);
  jit->schedules = NULL;
  jit->captured_linear = NULL;
  jit->compiled_linear = NULL;
  jit->compiled_device = POLY_DEVICE_AUTO;
  jit->input_buffers = NULL;
  jit->input_views = NULL;
  jit->input_dtypes = NULL;
  jit->input_devices = NULL;
  jit->n_schedules = 0;
  jit->schedules_cap = 0;
  jit->n_recorded_schedules = 0;
  jit->n_inputs = 0;
  jit->captured = false;
  jit->graphed_linear = NULL;
}

static PolyUOp *poly_jit_tensor_buffer(PolyTensor *tensor) {
  PolyUOp *current = poly_tensor_uop(tensor);
  /* Pinned tinygrad prepares JIT inputs with u.base, where base traverses
   * movement views, MULTI, and DETACH. Keep this local to JIT: global
   * has_buffer_identity intentionally does not make SHRINK/PERMUTE/etc.
   * storage identities. */
  while (current && current->n_src >= 1 &&
         (poly_opset_has(POLY_GROUP_MOVEMENT, current->op) || current->op == POLY_OP_MULTI ||
          current->op == POLY_OP_DETACH))
    current = current->src[0];
  const PolyUOp *buf = poly_uop_get_buffer_identity(current);
  return (PolyUOp *)buf;
}

/* Pinned _prepare_jit_inputs applies rangeify.mop_cleanup after replacing the
 * physical input base with NOOP. Its only current rule merges adjacent
 * RESHAPEs (tinygrad/engine/jit.py:227-245 and schedule/rangeify.py:125-128). */
static PolyUOp *rule_jit_merge_adjacent_reshapes(
    PolyCtx *ctx,
    PolyUOp *root,
    const PolyBindings *bindings
) {
  PolyUOp *inner = poly_bind(bindings, "inner");
  if (!root || !inner || inner->n_src < 1 || root->n_src < 1) return NULL;
  PolyUOp *inline_src[16];
  PolyUOp **src = root->n_src <= 16 ? inline_src : malloc((size_t)root->n_src * sizeof(*src));
  if (!src) return NULL;
  for (int i = 0; i < root->n_src; i++)
    src[i] = root->src[i];
  src[0] = inner->src[0];
  PolyUOp *result = poly_uop_tagged_arg(
      ctx, root->op, root->dtype, src, root->n_src, root->arg, root->tag, root->tag_arg
  );
  if (src != inline_src) free(src);
  return result;
}

static _Thread_local PolyPatternMatcher *g_pm_jit_mop_cleanup = NULL;
static PolyPatternMatcher *poly_pm_jit_mop_cleanup(void) {
  if (g_pm_jit_mop_cleanup) return g_pm_jit_mop_cleanup;
  PolyPat *inner = poly_pat_op(POLY_OP_RESHAPE, NULL, 0, "inner");
  PolyPat *src[] = {inner};
  PolyRule rules[] = {{
      poly_pat_allow_any_len(poly_pat_op(POLY_OP_RESHAPE, src, 1, NULL)),
      rule_jit_merge_adjacent_reshapes,
  }};
  g_pm_jit_mop_cleanup = poly_pm_thread_cache(poly_pm_new(rules, 1));
  return g_pm_jit_mop_cleanup;
}

static int poly_jit_append_var_binding(
    PolyVarBinding **items,
    int *n_items,
    int *cap_items,
    PolyUOp *var,
    int64_t value
) {
  if (!items || !n_items || !cap_items || !var || value < INT32_MIN || value > INT32_MAX) return -1;
  for (int i = 0; i < *n_items; i++) {
    if ((*items)[i].var != var) continue;
    return (*items)[i].value == (int32_t)value ? 0 : -1;
  }
  if (*n_items >= *cap_items) {
    int new_cap = *cap_items ? *cap_items * 2 : 8;
    PolyVarBinding *tmp = realloc(*items, (size_t)new_cap * sizeof(*tmp));
    if (!tmp) return -1;
    *items = tmp;
    *cap_items = new_cap;
  }
  (*items)[*n_items] = (PolyVarBinding){.var = var, .value = (int32_t)value};
  (*n_items)++;
  return 0;
}

/* Mirror tinygrad's
 *   u.substitute({u.base:NOOP}, extra_pm=mop_cleanup).unbind_all()
 * for Polygrad's current physical input view. The returned view is a
 * hash-consed signature UOp; bindings are pass-local replay values. Logical
 * tensor provenance is intentionally outside this execution-layer boundary. */
static int poly_jit_prepare_input_view(
    PolyJit *jit,
    PolyTensor *tensor,
    PolyUOp *buf,
    PolyUOp **view_out,
    PolyVarBinding **bindings_out,
    int *n_bindings_out
) {
  if (!jit || !jit->ctx || !tensor || !buf || !view_out || !bindings_out || !n_bindings_out)
    return -1;
  *view_out = NULL;
  *bindings_out = NULL;
  *n_bindings_out = 0;

  PolyUOp *root = poly_tensor_uop(tensor);
  if (!root) return -1;
  PolyUOp *noop = poly_uop0(jit->ctx, POLY_OP_NOOP, POLY_VOID, poly_arg_none());
  if (!noop) return -1;
  PolyUOp *base_free = poly_uop_substitute(jit->ctx, root, &buf, &noop, 1);
  if (!base_free) return -1;

  int n_topo = 0;
  PolyUOp **topo = poly_toposort_alloc(NULL, base_free, &n_topo);
  if (!topo) return -1;
  PolyUOp **from = malloc((size_t)(n_topo ? n_topo : 1) * sizeof(*from));
  PolyUOp **to = malloc((size_t)(n_topo ? n_topo : 1) * sizeof(*to));
  PolyVarBinding *bindings = NULL;
  int n_sub = 0, n_bindings = 0, cap_bindings = 0;
  int rc = 0;
  if (!from || !to) rc = -1;

  /* Pinned UOp.unbind_all requires BIND(PARAM/DEFINE_VAR, CONST), records one
   * value per variable, and rewrites every BIND occurrence to that variable. */
  for (int i = 0; rc == 0 && i < n_topo; i++) {
    PolyUOp *u = topo[i];
    if (!u || u->op != POLY_OP_BIND) continue;
    if (u->n_src < 2 || u->src[0]->op != POLY_OP_DEFINE_VAR) {
      rc = -1;
      break;
    }
    int64_t value = 0;
    if (poly_uop_const_i64(u->src[1], &value) != 0 ||
        poly_jit_append_var_binding(&bindings, &n_bindings, &cap_bindings, u->src[0], value) != 0) {
      rc = -1;
      break;
    }
    from[n_sub] = u;
    to[n_sub] = u->src[0];
    n_sub++;
  }

  PolyUOp *unbound = base_free;
  if (rc == 0 && n_sub > 0) unbound = poly_uop_substitute(jit->ctx, base_free, from, to, n_sub);
  if (rc == 0 && unbound)
    unbound = poly_graph_rewrite(jit->ctx, unbound, poly_pm_jit_mop_cleanup());
  if (rc == 0 && !unbound) rc = -1;

  poly_toposort_free(topo);
  free(from);
  free(to);
  if (rc != 0) {
    free(bindings);
    return -1;
  }
  *view_out = unbound;
  *bindings_out = bindings;
  *n_bindings_out = n_bindings;
  return 0;
}

static int poly_jit_input_index(PolyJit *jit, PolyUOp *buf) {
  if (!jit || !buf) return -1;
  for (int i = 0; i < jit->n_inputs; i++)
    if (jit->input_buffers[i] == buf) return i;
  return -1;
}

static bool poly_jit_capture_input_spec(PolyJit *jit, int index, PolyTensor *tensor, PolyUOp *buf) {
  if (!jit || index < 0 || index >= jit->n_inputs || !tensor || !buf) return false;
  PolyUOp *root = poly_tensor_uop(tensor);
  if (!root) return false;
  PolyVarBinding *bindings = NULL;
  int n_bindings = 0;
  if (poly_jit_prepare_input_view(
          jit, tensor, buf, &jit->input_views[index], &bindings, &n_bindings
      ) != 0)
    return false;
  free(bindings);
  jit->input_buffers[index] = buf;
  jit->input_dtypes[index] = poly_dtype_scalar(root->dtype);
  jit->input_devices[index] = poly_uop_device_uop_cached(jit->ctx, buf, NULL);
  return jit->input_devices[index] != NULL;
}

static bool poly_jit_input_matches_spec(
    PolyJit *jit,
    int index,
    PolyTensor *tensor,
    PolyUOp *buf,
    PolyVarBinding **bindings,
    int *n_bindings,
    int *cap_bindings
) {
  if (!jit || index < 0 || index >= jit->n_inputs || !tensor || !buf) return false;
  PolyUOp *root = poly_tensor_uop(tensor);
  if (!root) return false;
  if (!poly_dtype_eq(poly_dtype_scalar(root->dtype), jit->input_dtypes[index])) return false;
  if (poly_uop_device_uop_cached(jit->ctx, buf, NULL) != jit->input_devices[index]) return false;
  PolyUOp *view = NULL;
  PolyVarBinding *input_bindings = NULL;
  int n_input_bindings = 0;
  if (poly_jit_prepare_input_view(jit, tensor, buf, &view, &input_bindings, &n_input_bindings) != 0)
    return false;
  if (view != jit->input_views[index]) {
    free(input_bindings);
    return false;
  }
  for (int i = 0; i < n_input_bindings; i++) {
    if (poly_jit_append_var_binding(
            bindings, n_bindings, cap_bindings, input_bindings[i].var, input_bindings[i].value
        ) != 0) {
      free(input_bindings);
      return false;
    }
  }
  free(input_bindings);
  return true;
}

static int poly_jit_append_unique_buffer(
    PolyUOp ***items,
    int *n_items,
    int *cap_items,
    PolyUOp *buf
) {
  if (!items || !n_items || !cap_items || !buf) return -1;
  for (int i = 0; i < *n_items; i++)
    if ((*items)[i] == buf) return i;
  if (*n_items >= *cap_items) {
    int new_cap = *cap_items ? *cap_items * 2 : 8;
    PolyUOp **tmp = realloc(*items, (size_t)new_cap * sizeof(*tmp));
    if (!tmp) return -1;
    *items = tmp;
    *cap_items = new_cap;
  }
  (*items)[*n_items] = buf;
  return (*n_items)++;
}

static int poly_jit_collect_external_buffers(
    PolySchedule **schedules,
    int n_schedules,
    PolyUOp ***external_out,
    int *n_external_out
) {
  if (!schedules || n_schedules <= 0 || !external_out || !n_external_out) return -1;
  PolyUOp **external = NULL;
  int n_external = 0, cap_external = 0;
  for (int s = 0; s < n_schedules; s++) {
    PolySchedule *captured = schedules[s];
    int n_sched_external = poly_schedule_external_slot_count(captured);
    if (n_sched_external < 0) goto fail;
    for (int i = 0; i < n_sched_external; i++) {
      PolyUOp *buf = poly_schedule_external_slot_buffer(captured, i);
      if (!buf || poly_jit_append_unique_buffer(&external, &n_external, &cap_external, buf) < 0)
        goto fail;
    }
  }
  *external_out = external;
  *n_external_out = n_external;
  return 0;

fail:
  free(external);
  return -1;
}

/* Pinned jit_lower substitutes exactly input_uops with shaped UOp.param nodes
 * before retaining CapturedJit.linear (tinygrad/engine/jit.py:67-76).
 * Polygrad's combined schedule has already memory-planned external inputs;
 * replacing only their identities cannot change intermediate allocation. */
static int poly_jit_parameterize_inputs(PolyJit *jit, PolySchedule *schedule) {
  if (!jit || !jit->ctx || !schedule || !schedule->template || !schedule->template->linear ||
      schedule->template->linear->op != POLY_OP_LINEAR || !schedule->run)
    return -1;
  if (jit->n_inputs == 0) return 0;

  PolyUOp **params = calloc((size_t)jit->n_inputs, sizeof(*params));
  int *input_slots = malloc((size_t)jit->n_inputs * sizeof(*input_slots));
  if (!params || !input_slots) {
    free(params);
    free(input_slots);
    return -1;
  }
  for (int i = 0; i < jit->n_inputs; i++) {
    input_slots[i] = -1;
    params[i] = poly_uop_param(jit->ctx, i, jit->input_buffers[i]);
    if (!params[i]) goto fail;
    for (int slot = 0; slot < schedule->template->n_buf_slots; slot++) {
      PolyScheduleBufSlot *meta = &schedule->template->buf_slots[slot];
      if (meta->buf_uop != jit->input_buffers[i]) continue;
      if (input_slots[i] >= 0 || meta->is_intermediate || meta->is_memory_arena ||
          meta->has_memory_parent)
        goto fail;
      input_slots[i] = slot;
    }
    if (input_slots[i] < 0 &&
        poly_uop_reachable(jit->ctx, schedule->template->linear, jit->input_buffers[i]))
      goto fail;
  }

  PolyUOp *linear = poly_uop_substitute(
      jit->ctx, schedule->template->linear, jit->input_buffers, params, jit->n_inputs
  );
  if (!linear || linear->op != POLY_OP_LINEAR || linear->n_src != schedule->template->linear->n_src)
    goto fail;
  for (int k = 0; k < linear->n_src; k++) {
    if (!linear->src[k] || linear->src[k]->op != POLY_OP_CALL) goto fail;
  }

  for (int i = 0; i < jit->n_inputs; i++)
    if (input_slots[i] >= 0) schedule->template->buf_slots[input_slots[i]].buf_uop = params[i];
  schedule->template->linear = linear;
  for (int k = 0; k < linear->n_src; k++)
    schedule->run->calls[k].call = linear->src[k];

  free(params);
  free(input_slots);
  return 0;

fail:
  free(params);
  free(input_slots);
  return -1;
}

static int poly_jit_append_unique_param(
    PolyUOp ***items,
    int *n_items,
    int *cap_items,
    PolyUOp *param
) {
  if (!items || !n_items || !cap_items || !param) return -1;
  for (int i = 0; i < *n_items; i++)
    if ((*items)[i] == param) return 0;
  if (*n_items >= *cap_items) {
    int new_cap = *cap_items ? *cap_items * 2 : 8;
    PolyUOp **grown = realloc(*items, (size_t)new_cap * sizeof(*grown));
    if (!grown) return -1;
    *items = grown;
    *cap_items = new_cap;
  }
  (*items)[(*n_items)++] = param;
  return 0;
}

/* Pinned create_graph_call(batch), tinygrad/engine/jit.py:25-29. The graph
 * body retains the exact compiled PROGRAM CALL occurrences; the outer CALL
 * exposes only deduplicated JIT PARAMs reachable from their arguments. */
static PolyUOp *poly_jit_create_graph_call(PolyCtx *ctx, PolyUOp **calls, int n_calls) {
  if (!ctx || !calls || n_calls <= 1) return NULL;
  PolyUOp **params = NULL;
  int n_params = 0, cap_params = 0;
  for (int k = 0; k < n_calls; k++) {
    PolyUOp *call = calls[k];
    if (!call || call->op != POLY_OP_CALL || call->n_src < 1 || !call->src[0] ||
        call->src[0]->op != POLY_OP_PROGRAM)
      goto fail;
    for (int a = 1; a < call->n_src; a++) {
      int n_topo = 0;
      PolyUOp **topo = poly_toposort_alloc(ctx, call->src[a], &n_topo);
      if (!topo && n_topo > 0) goto fail;
      for (int i = 0; i < n_topo; i++) {
        if (!topo[i] || topo[i]->op != POLY_OP_PARAM || topo[i]->arg.kind != POLY_ARG_PARAM ||
            !topo[i]->arg.param)
          continue;
        if (poly_jit_append_unique_param(&params, &n_params, &cap_params, topo[i]) != 0) {
          poly_toposort_free(topo);
          goto fail;
        }
      }
      poly_toposort_free(topo);
    }
  }

  PolyUOp *nested = poly_uop(ctx, POLY_OP_LINEAR, POLY_VOID, calls, n_calls, poly_arg_none());
  if (!nested) goto fail;
  PolyUOp *cf_src[1] = {nested};
  PolyUOp *cf = poly_uop(
      ctx, POLY_OP_CUSTOM_FUNCTION, POLY_VOID, cf_src, 1, poly_arg_str("graph")
  );
  if (!cf) goto fail;
  PolyUOp **call_src = malloc((size_t)(n_params + 1) * sizeof(*call_src));
  if (!call_src) goto fail;
  call_src[0] = cf;
  for (int i = 0; i < n_params; i++)
    call_src[i + 1] = params[i];
  PolyUOp *out = poly_uop(
      ctx, POLY_OP_CALL, POLY_VOID, call_src, n_params + 1, poly_arg_none()
  );
  free(call_src);
  free(params);
  return out;

fail:
  free(params);
  return NULL;
}

static int poly_jit_flush_graph_batch(
    PolyCtx *ctx,
    PolyUOp **calls,
    int n_calls,
    PolyUOp **out,
    int *n_out,
    int *max_batch_size
) {
  if (!ctx || !calls || n_calls <= 0 || !out || !n_out || !max_batch_size) return -1;
  if (n_calls == 1) {
    out[(*n_out)++] = calls[0];
    return 0;
  }
  PolyUOp *graph_call = poly_jit_create_graph_call(ctx, calls, n_calls);
  if (!graph_call) return -1;
  out[(*n_out)++] = graph_call;
  if (*max_batch_size > 0 && *max_batch_size <= INT_MAX / 2) *max_batch_size *= 2;
  return 0;
}

/* Direct port of pinned graph_split_rewrite's default PROGRAM batching for
 * CUDA (tinygrad/engine/jit.py:31-60). COPY batching stays disabled until the
 * required D2D COPY probe establishes its runtime dependency/update boundary. */
static PolyUOp *poly_jit_graph_split_rewrite(PolyCtx *ctx, PolyCompiledSchedule *compiled) {
  PolyUOp *linear = compiled ? compiled->linear : NULL;
  PolyDevice device = compiled ? compiled->device : POLY_DEVICE_AUTO;
  if (!ctx || !linear || linear->op != POLY_OP_LINEAR) return NULL;
#ifdef POLY_HAS_CUDA
  if (device != POLY_DEVICE_CUDA || !poly_cuda_graph_available()) return linear;
#else
  (void)device;
  return linear;
#endif

  PolyUOp **out = calloc((size_t)(linear->n_src > 0 ? linear->n_src : 1), sizeof(*out));
  if (!out) return NULL;
  int n_out = 0, batch_start = 0, n_batch = 0;
  int max_batch_size = 32; /* pinned JIT_BATCH_SIZE default, helpers.py:241 */
  for (int i = 0; i < linear->n_src; i++) {
    PolyUOp *call = linear->src[i];
    bool can_graph = call && call->op == POLY_OP_CALL && call->n_src >= 1 && call->src[0] &&
                     call->src[0]->op == POLY_OP_PROGRAM && compiled->run &&
                     compiled->run->calls[i].lowered_device == POLY_DEVICE_CUDA;
    bool can_extend = can_graph && (max_batch_size == 0 || n_batch < max_batch_size);
    if (!can_extend && n_batch > 0) {
      if (poly_jit_flush_graph_batch(
              ctx, &linear->src[batch_start], n_batch, out, &n_out, &max_batch_size
          ) != 0)
        goto fail;
      n_batch = 0;
    }
    if (can_graph) {
      if (n_batch == 0) batch_start = i;
      n_batch++;
    } else {
      out[n_out++] = call;
    }
  }
  if (n_batch > 0 &&
      poly_jit_flush_graph_batch(
          ctx, &linear->src[batch_start], n_batch, out, &n_out, &max_batch_size
      ) != 0)
    goto fail;

  PolyUOp *result = poly_uop(ctx, POLY_OP_LINEAR, POLY_VOID, out, n_out, linear->arg);
  free(out);
  return result;

fail:
  free(out);
  return NULL;
}

PolyCompiledSchedule *poly_jit_lower(PolyCtx *ctx, PolySchedule *schedule) {
  if (!ctx || !schedule) return NULL;
  PolyDevice device = poly_schedule_infer_device(ctx, schedule);
  PolyCompiledSchedule *compiled = poly_lower_schedule(ctx, schedule, device);
  if (!compiled) return NULL;
  PolyUOp *graph_linear = poly_jit_graph_split_rewrite(ctx, compiled);
  if (!graph_linear || poly_compiled_schedule_set_jit_graph(compiled, graph_linear) != 0) {
    poly_compiled_schedule_free(compiled);
    return NULL;
  }
  return compiled;
}

static int poly_jit_build_captured_linear(PolyJit *jit) {
  if (!jit || !jit->ctx || jit->n_schedules <= 0) return -1;
  PolyUOp **external = NULL;
  int n_external = 0;
  if (poly_jit_collect_external_buffers(jit->schedules, jit->n_schedules, &external, &n_external) !=
      0)
    return -1;
  PolySchedule *combined = poly_schedule_replay_many_with_buffers(
      jit->ctx, jit->schedules, jit->n_schedules, external, external, n_external
  );
  free(external);
  if (!combined) return -1;
  if (jit->prune) {
    PolySchedule *kept = NULL;
    PolySchedule *onetime = NULL;
    int prune_rc = poly_schedule_prune_for_buffers(
        jit->ctx, combined, jit->input_buffers, jit->n_inputs, &kept, &onetime
    );
    poly_schedule_free(combined);
    combined = NULL;
    if (prune_rc != 0 || !kept || !onetime) {
      poly_schedule_free(onetime);
      poly_schedule_free(kept);
      return -1;
    }
    /* Pinned prune_linear returns kept and onetime LINEARs. Capture deferred
     * both, so execute the complement once before lowering/retaining kept. */
    int onetime_rc = poly_run_schedule(jit->ctx, onetime, NULL, 0);
    poly_schedule_free(onetime);
    if (onetime_rc != 0) {
      poly_schedule_free(kept);
      return -1;
    }
    combined = kept;
  }
  if (poly_jit_parameterize_inputs(jit, combined) != 0) {
    poly_schedule_free(combined);
    return -1;
  }
  jit->captured_linear = combined;
  jit->n_recorded_schedules = jit->n_schedules;
  for (int i = 0; i < jit->n_schedules; i++)
    poly_schedule_free(jit->schedules[i]);
  free(jit->schedules);
  jit->schedules = NULL;
  jit->n_schedules = 0;
  jit->schedules_cap = 0;
  return 0;
}

PolyJit *poly_jit_new(PolyCtx *ctx) {
  if (!ctx) return NULL;
  PolyJit *jit = calloc(1, sizeof(*jit));
  if (!jit) return NULL;
  jit->ctx = ctx;
  return jit;
}

void poly_jit_free(PolyJit *jit) {
  if (!jit) return;
  if (jit->capturing && jit->ctx && jit->ctx->active_jit_capture == jit)
    jit->ctx->active_jit_capture = NULL;
  poly_jit_clear(jit);
  free(jit);
}

int poly_jit_set_prune(PolyJit *jit, bool prune) {
  if (!jit || jit->capturing || jit->captured) return -1;
  jit->prune = prune;
  return 0;
}

int poly_jit_begin_capture(PolyJit *jit, PolyTensor **inputs, int n_inputs) {
  if (!jit || !jit->ctx || n_inputs < 0 || (n_inputs > 0 && !inputs)) return -1;
  /* Pinned TinyJit rejects any nested capture before replacing capture state
   * (tinygrad/engine/jit.py:278-284). A raw same-object C re-entry has no
   * frontend finally block, so abort that partial capture before rejecting. */
  if (jit->ctx->active_jit_capture) {
    if (jit->ctx->active_jit_capture == jit) poly_jit_cancel_capture(jit);
    return -1;
  }

  poly_jit_clear(jit);
  if (n_inputs > 0) {
    jit->input_buffers = calloc((size_t)n_inputs, sizeof(*jit->input_buffers));
    jit->input_views = calloc((size_t)n_inputs, sizeof(*jit->input_views));
    jit->input_dtypes = calloc((size_t)n_inputs, sizeof(*jit->input_dtypes));
    jit->input_devices = calloc((size_t)n_inputs, sizeof(*jit->input_devices));
    if (!jit->input_buffers || !jit->input_views || !jit->input_dtypes || !jit->input_devices) {
      poly_jit_clear(jit);
      return -1;
    }
  }
  jit->n_inputs = n_inputs;
  for (int i = 0; i < n_inputs; i++) {
    PolyUOp *buf = poly_jit_tensor_buffer(inputs[i]);
    if (!buf) {
      poly_jit_clear(jit);
      return -1;
    }
    if (poly_jit_input_index(jit, buf) >= 0) {
      poly_jit_clear(jit);
      return -1;
    }
    if (!poly_jit_capture_input_spec(jit, i, inputs[i], buf)) {
      poly_jit_clear(jit);
      return -1;
    }
  }
  jit->capturing = true;
  jit->ctx->active_jit_capture = jit;
  return 0;
}

int poly_jit_end_capture(PolyJit *jit) {
  if (!jit || !jit->ctx || !jit->capturing) return -1;
  if (jit->ctx->active_jit_capture == jit) jit->ctx->active_jit_capture = NULL;
  jit->capturing = false;
  if (jit->n_schedules > 0 && poly_jit_build_captured_linear(jit) == 0) {
    /* Pinned TinyJit constructs CapturedJit and immediately invokes it once.
     * Use the same compiled schedule that subsequent exact-input replay uses. */
    if (poly_jit_run_captured_linear(jit, jit->input_buffers, NULL, 0) == 0) {
      jit->captured = true;
      return 0;
    }
  }
  poly_jit_clear(jit);
  return -1;
}

void poly_jit_cancel_capture(PolyJit *jit) {
  if (!jit || !jit->capturing) return;
  if (jit->ctx && jit->ctx->active_jit_capture == jit) jit->ctx->active_jit_capture = NULL;
  jit->capturing = false;
  poly_jit_clear(jit);
}

bool poly_jit_is_captured(PolyJit *jit) {
  return jit && jit->captured;
}

bool poly_jit_is_capturing(PolyJit *jit) {
  return jit && jit->capturing;
}

int poly_jit_schedule_count(PolyJit *jit) {
  if (!jit) return 0;
  return jit->captured ? jit->n_recorded_schedules : jit->n_schedules;
}

PolyUOp *poly_jit_captured_linear(PolyJit *jit) {
  if (!jit) return NULL;
  if (jit->graphed_linear) return jit->graphed_linear;
  return (jit->captured_linear && jit->captured_linear->template)
             ? jit->captured_linear->template->linear : NULL;
}

int poly_jit_record_schedule(PolyJit *jit, PolySchedule *sched) {
  if (!jit || !sched) return -1;
  if (jit->n_schedules >= jit->schedules_cap) {
    int new_cap = jit->schedules_cap ? jit->schedules_cap * 2 : 4;
    PolySchedule **new_schedules =
        realloc(jit->schedules, (size_t)new_cap * sizeof(*new_schedules));
    if (!new_schedules) return -1;
    jit->schedules = new_schedules;
    jit->schedules_cap = new_cap;
  }
  jit->schedules[jit->n_schedules++] = sched;
  return 0;
}

static int poly_jit_run_captured_linear(
    PolyJit *jit,
    PolyUOp **current_inputs,
    PolyVarBinding *var_bindings,
    int n_var_bindings
) {
  if (!jit || !jit->captured_linear || (jit->n_inputs > 0 && !current_inputs)) return -1;
  PolyDevice device = poly_schedule_infer_device(jit->ctx, jit->captured_linear);
  if (!jit->compiled_linear || jit->compiled_device != device) {
    poly_compiled_schedule_free(jit->compiled_linear);
    jit->compiled_linear = poly_jit_lower(jit->ctx, jit->captured_linear);
    jit->compiled_device = jit->compiled_linear ? device : POLY_DEVICE_AUTO;
    if (!jit->compiled_linear) return -1;
    jit->graphed_linear = jit->compiled_linear->jit_graph_linear;
  }
  return poly_run_compiled_schedule_with_input_uops(
      jit->compiled_linear, current_inputs, jit->n_inputs, var_bindings, n_var_bindings
  );
}

int poly_jit_run_with_vars(
    PolyJit *jit,
    PolyTensor **inputs,
    int n_inputs,
    PolyVarBinding *var_bindings,
    int n_var_bindings
) {
  if (!jit || !jit->captured || !jit->ctx || (n_inputs > 0 && !inputs) || n_var_bindings < 0 ||
      (n_var_bindings > 0 && !var_bindings))
    return -1;
  if (n_inputs != jit->n_inputs) return -1;

  bool timing = poly_debug_at_least(7);
  double t0 = timing ? poly_now_ms() : 0.0;
  int ret = -1;
  PolyVarBinding *effective_bindings = NULL;
  int n_effective_bindings = 0, cap_effective_bindings = 0;
  PolyUOp *current_inputs_stack[16];
  PolyUOp **current_inputs = current_inputs_stack;
  if (n_inputs > (int)(sizeof(current_inputs_stack) / sizeof(current_inputs_stack[0]))) {
    current_inputs = calloc((size_t)n_inputs, sizeof(*current_inputs));
    if (!current_inputs) return -1;
  }

  for (int i = 0; i < n_inputs; i++) {
    current_inputs[i] = poly_jit_tensor_buffer(inputs[i]);
    if (!current_inputs[i] || !poly_jit_input_matches_spec(
                                  jit, i, inputs[i], current_inputs[i], &effective_bindings,
                                  &n_effective_bindings, &cap_effective_bindings
                              ))
      goto cleanup;
  }
  for (int i = 0; i < n_var_bindings; i++) {
    if (poly_jit_append_var_binding(
            &effective_bindings, &n_effective_bindings, &cap_effective_bindings,
            var_bindings[i].var, var_bindings[i].value
        ) != 0)
      goto cleanup;
  }
  double t_inputs = timing ? poly_now_ms() : 0.0;

  ret = poly_jit_run_captured_linear(jit, current_inputs, effective_bindings, n_effective_bindings);
  if (timing) {
    double t_done = poly_now_ms();
    fprintf(
        stderr,
        "[polygrad:jit_run] path=retained inputs=%d input_check=%.3fms run=%.3fms "
        "total=%.3fms ret=%d\n",
        n_inputs, t_inputs - t0, t_done - t_inputs, t_done - t0, ret
    );
    fflush(stderr);
  }

cleanup:
  free(effective_bindings);
  if (current_inputs != current_inputs_stack) free(current_inputs);
  return ret;
}

int poly_jit_run(PolyJit *jit, PolyTensor **inputs, int n_inputs) {
  return poly_jit_run_with_vars(jit, inputs, n_inputs, NULL, 0);
}
