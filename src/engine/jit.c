/* jit.c -- tinygrad-style JIT capture/replay for raw Tensor realizes. */

#include "engine/jit.h"
#include "codegen/codegen.h"
#include "ctx.h"
#include "device.h"
#include "frontend_internal.h"
#include "frontend.h"
#include "uop/upat.h"
#include "schedule/memory.h"
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
  PolyUOp **linears;
  int n_linears;
  int linears_cap;
  int n_recorded_linears;
  PolyVarBinding *var_bindings;
  int n_var_bindings;
  int var_bindings_cap;
  PolyUOp *captured_linear;
  bool captured_root_retained;
  bool prune;
  bool capturing;
  bool captured;
};

static int poly_jit_run_captured_linear(
    PolyJit *jit,
    PolyUOp **current_inputs,
    PolyVarBinding *var_bindings,
    int n_var_bindings
);

static void poly_jit_clear(PolyJit *jit) {
  if (!jit) return;
  if (jit->captured_root_retained && jit->ctx && jit->captured_linear)
    poly_uop_release(jit->ctx, jit->captured_linear);
  for (int i = 0; jit->ctx && i < jit->n_linears; i++)
    poly_uop_release(jit->ctx, jit->linears[i]);
  for (int i = 0; jit->ctx && i < jit->n_inputs; i++) {
    if (jit->input_buffers[i]) poly_uop_release(jit->ctx, jit->input_buffers[i]);
    if (jit->input_views[i]) poly_uop_release(jit->ctx, jit->input_views[i]);
    if (jit->input_devices[i]) poly_uop_release(jit->ctx, jit->input_devices[i]);
  }
  for (int i = 0; jit->ctx && i < jit->n_var_bindings; i++)
    poly_uop_release(jit->ctx, jit->var_bindings[i].var);
  free(jit->linears);
  free(jit->var_bindings);
  free(jit->input_buffers);
  free(jit->input_views);
  free(jit->input_dtypes);
  free(jit->input_devices);
  jit->linears = NULL;
  jit->var_bindings = NULL;
  jit->captured_linear = NULL;
  jit->captured_root_retained = false;
  jit->input_buffers = NULL;
  jit->input_views = NULL;
  jit->input_dtypes = NULL;
  jit->input_devices = NULL;
  jit->n_linears = 0;
  jit->linears_cap = 0;
  jit->n_recorded_linears = 0;
  jit->n_var_bindings = 0;
  jit->var_bindings_cap = 0;
  jit->n_inputs = 0;
  jit->captured = false;
}

static PolyUOp *poly_jit_tensor_buffer(PolyTensor *tensor) {
  PolyUOp *current = poly_tensor_uop(tensor);
  /* Pinned tinygrad prepares JIT inputs with u.base, where base traverses
   * movement views, MULTI, and DETACH. Keep this local to JIT: global
   * has_buffer_identity intentionally does not make SHRINK/PERMUTE/etc.
   * storage identities. */
  while (current && current->n_src >= 1 &&
         (poly_opset_has(POLY_GROUP_MOVEMENT, current->op) || current->op == POLY_OP_UNSHARD ||
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
  PolyUPat *inner = poly_upat_op(POLY_OP_RESHAPE, NULL, 0, "inner");
  PolyUPat *src[] = {inner};
  PolyRule rules[] = {{
      poly_upat_allow_any_len(poly_upat_op(POLY_OP_RESHAPE, src, 1, NULL)),
      rule_jit_merge_adjacent_reshapes,
  }};
  g_pm_jit_mop_cleanup = poly_pm_thread_cache(poly_pm_new(rules, 1));
  return g_pm_jit_mop_cleanup;
}

/* Current _prepare_jit_inputs publishes var_vals as dict[str, int]. */
static bool poly_jit_same_var(PolyUOp *left, PolyUOp *right) {
  if (left == right) return true;
  const char *left_expr = poly_uop_expr(left), *right_expr = poly_uop_expr(right);
  return left_expr && right_expr && strcmp(left_expr, right_expr) == 0;
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
    if (!poly_jit_same_var((*items)[i].var, var)) continue;
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

static int append_unique_uop(PolyUOp ***items, int *n_items, int *capacity, PolyUOp *item) {
  for (int i = 0; i < *n_items; i++)
    if ((*items)[i] == item) return 0;
  if (*n_items >= *capacity) {
    int next = *capacity ? *capacity * 2 : 8;
    PolyUOp **grown = realloc(*items, (size_t)next * sizeof(*grown));
    if (!grown) return -1;
    *items = grown;
    *capacity = next;
  }
  (*items)[(*n_items)++] = item;
  return 0;
}

/* Current Tinygrad engine/jit.py:create_graph_call. */
PolyUOp *poly_create_graph_call(PolyCtx *ctx, PolyUOp **calls, int n_calls) {
  if (!ctx || !calls || n_calls <= 0) return NULL;
  PolyUOp **params = NULL;
  int n_params = 0, params_capacity = 0;
  for (int i = 0; i < n_calls; i++) {
    PolyUOp *call = calls[i];
    if (!call || call->op != POLY_OP_CALL || call->n_src < 1) goto fail;
    for (int j = 1; j < call->n_src; j++) {
      int n_topo = 0;
      PolyUOp **topo = poly_toposort_alloc(ctx, call->src[j], &n_topo);
      if (!topo) goto fail;
      bool ok = true;
      for (int k = 0; k < n_topo; k++)
        if (topo[k]->op == POLY_OP_PARAM &&
            append_unique_uop(&params, &n_params, &params_capacity, topo[k]) != 0) {
          ok = false;
          break;
        }
      poly_toposort_free(topo);
      if (!ok) goto fail;
    }
  }

  PolyUOp *nested = poly_uop(ctx, POLY_OP_LINEAR, POLY_VOID, calls, n_calls, poly_arg_none());
  PolyUOp *function =
      nested ? poly_uop1(ctx, POLY_OP_CUSTOM_FUNCTION, POLY_VOID, nested, poly_arg_str("graph"))
             : NULL;
  PolyUOp **src = function ? malloc((size_t)(n_params + 1) * sizeof(*src)) : NULL;
  if (!src) goto fail;
  src[0] = function;
  for (int i = 0; i < n_params; i++)
    src[i + 1] = params[i];
  PolyUOp *ret = poly_uop(ctx, POLY_OP_CALL, POLY_VOID, src, n_params + 1, poly_arg_none());
  free(src);
  free(params);
  return ret;

fail:
  free(params);
  return NULL;
}

static bool graph_cuda_device(PolyUOp *device) {
  if (!device || device->op != POLY_OP_DEVICE) return false;
  if (device->arg.kind == POLY_ARG_STRING)
    return device->arg.str && strncmp(device->arg.str, "CUDA", 4) == 0;
  if (device->arg.kind != POLY_ARG_STRING_TUPLE || device->arg.string_tuple.n <= 0 ||
      !device->arg.string_tuple.vals)
    return false;
  for (int i = 0; i < device->arg.string_tuple.n; i++)
    if (!device->arg.string_tuple.vals[i] ||
        strncmp(device->arg.string_tuple.vals[i], "CUDA", 4) != 0)
      return false;
  return true;
}

/* Current CUDAGraph is a MultiGraphRunner: PROGRAM and same-runtime COPY
 * calls are graphable when every concrete argument belongs to CUDA. */
static bool graph_cuda_supports_call(PolyCtx *ctx, PolyUOp *call) {
  if (!ctx || !call || call->op != POLY_OP_CALL || call->n_src < 2 || !call->src[0] ||
      (call->src[0]->op != POLY_OP_PROGRAM && call->src[0]->op != POLY_OP_COPY))
    return false;
  bool saw_device = false;
  for (int i = 1; i < call->n_src; i++) {
    if (poly_uop_is_bound_var(call->src[i])) continue;
    PolyUOp *device = poly_uop_device_uop_cached(ctx, call->src[i], NULL);
    if (!graph_cuda_device(device)) return false;
    saw_device = true;
  }
  return saw_device;
}

static int flush_graph_batch(
    PolyCtx *ctx,
    PolyUOp **batch,
    int n_batch,
    PolyUOp **out,
    int *n_out,
    int *max_batch_size
) {
  if (n_batch == 1 && !poly_getenv_flag("GRAPH_ONE_KERNEL")) {
    out[(*n_out)++] = batch[0];
    return 0;
  }
  PolyUOp *graph = poly_create_graph_call(ctx, batch, n_batch);
  if (!graph) return -1;
  out[(*n_out)++] = graph;
  if (*max_batch_size > 0 && *max_batch_size <= INT_MAX / 2) *max_batch_size *= 2;
  return 0;
}

/* Current Tinygrad engine/jit.py:graph_split_rewrite for the implemented
 * CUDAGraph runtime. */
static PolyUOp *poly_graph_split_rewrite(PolyCtx *ctx, PolyUOp *linear, int max_batch_size) {
  if (!ctx || !linear || linear->op != POLY_OP_LINEAR) return NULL;
#ifdef POLY_HAS_CUDA
  if (!poly_cuda_graph_available()) return linear;
#else
  return linear;
#endif
  PolyUOp **out = malloc((size_t)(linear->n_src > 0 ? linear->n_src : 1) * sizeof(*out));
  PolyUOp **batch = malloc((size_t)(linear->n_src > 0 ? linear->n_src : 1) * sizeof(*batch));
  if (!out || !batch) {
    free(out);
    free(batch);
    return NULL;
  }
  int n_out = 0, n_batch = 0;
  for (int i = 0; i < linear->n_src; i++) {
    PolyUOp *call = linear->src[i];
    bool graphable = graph_cuda_supports_call(ctx, call);
    bool extend = graphable && (max_batch_size == 0 || n_batch < max_batch_size);
    if (!extend && n_batch > 0) {
      if (flush_graph_batch(ctx, batch, n_batch, out, &n_out, &max_batch_size) != 0) goto fail;
      n_batch = 0;
    }
    if (graphable)
      batch[n_batch++] = call;
    else
      out[n_out++] = call;
  }
  if (n_batch > 0 && flush_graph_batch(ctx, batch, n_batch, out, &n_out, &max_batch_size) != 0)
    goto fail;
  PolyUOp *ret = poly_uop(ctx, POLY_OP_LINEAR, POLY_VOID, out, n_out, linear->arg);
  free(out);
  free(batch);
  return ret;

fail:
  free(out);
  free(batch);
  return NULL;
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

  /* Current UOp.unbind_all records AFTER(var, STORE(var, CONST)) and rewrites
   * each bound occurrence to its immutable ALU BUFFER variable. */
  for (int i = 0; rc == 0 && i < n_topo; i++) {
    PolyUOp *u = topo[i];
    if (!poly_uop_is_bound_var(u)) continue;
    PolyUOp *var = u->src[0];
    PolyUOp *value_uop = u->src[1]->src[1];
    int64_t value = 0;
    if (poly_uop_const_i64(value_uop, &value) != 0 ||
        poly_jit_append_var_binding(&bindings, &n_bindings, &cap_bindings, var, value) != 0) {
      rc = -1;
      break;
    }
    from[n_sub] = u;
    to[n_sub] = var;
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
  PolyUOp *view = NULL;
  if (poly_jit_prepare_input_view(jit, tensor, buf, &view, &bindings, &n_bindings) != 0)
    return false;
  free(bindings);
  PolyUOp *device = poly_uop_device_uop_cached(jit->ctx, buf, NULL);
  if (!device || poly_uop_retain(jit->ctx, buf) != 0) return false;
  if (poly_uop_retain(jit->ctx, view) != 0) {
    poly_uop_release(jit->ctx, buf);
    return false;
  }
  if (poly_uop_retain(jit->ctx, device) != 0) {
    poly_uop_release(jit->ctx, view);
    poly_uop_release(jit->ctx, buf);
    return false;
  }
  /* Tinygrad CapturedJit owns input buffers/views/devices and variable UOps
   * (engine/jit.py:250-281). These retains are the C ownership mechanics for
   * inputs removed from the captured LINEAR by PARAM substitution. */
  jit->input_buffers[index] = buf;
  jit->input_views[index] = view;
  jit->input_dtypes[index] = root->dtype;
  jit->input_devices[index] = device;
  return true;
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
  if (!poly_dtype_eq(root->dtype, jit->input_dtypes[index])) return false;
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

static bool poly_buffer_list_contains(PolyUOp **buffers, int count, PolyUOp *buffer) {
  for (int i = 0; i < count; i++)
    if (buffers[i] == buffer) return true;
  return false;
}

static bool poly_call_collect_bufs(PolyUOp *call, PolyUOp ***buffers, int *count, int *capacity) {
  if (!call || call->op != POLY_OP_CALL || call->n_src < 1) return false;
  for (int i = 1; i < call->n_src; i++)
    if (!poly_collect_bufs(call->src[i], buffers, count, capacity)) return false;
  return true;
}

/* Current Tinygrad engine/jit.py:prune_linear. */
static int poly_prune_linear(
    PolyCtx *ctx,
    PolyUOp *linear,
    PolyUOp **needed,
    int n_needed,
    PolyUOp **kept_out,
    PolyUOp **onetime_out
) {
  if (!ctx || !linear || linear->op != POLY_OP_LINEAR || n_needed < 0 ||
      (n_needed > 0 && !needed) || !kept_out || !onetime_out)
    return -1;
  PolyUOp **live = NULL, **kept = NULL, **onetime = NULL;
  int n_live = 0, cap_live = 0, n_kept = 0, n_onetime = 0;
  for (int i = 0; i < n_needed; i++)
    if (poly_jit_append_unique_buffer(&live, &n_live, &cap_live, needed[i]) < 0) goto fail;
  kept = linear->n_src > 0 ? malloc((size_t)linear->n_src * sizeof(*kept)) : NULL;
  onetime = linear->n_src > 0 ? malloc((size_t)linear->n_src * sizeof(*onetime)) : NULL;
  if (linear->n_src > 0 && (!kept || !onetime)) goto fail;

  for (int i = 0; i < linear->n_src; i++) {
    PolyUOp **call_bufs = NULL;
    int n_call_bufs = 0, cap_call_bufs = 0;
    if (!poly_call_collect_bufs(linear->src[i], &call_bufs, &n_call_bufs, &cap_call_bufs)) {
      free(call_bufs);
      goto fail;
    }
    bool keep = false;
    for (int j = 0; j < n_call_bufs && !keep; j++)
      keep = poly_buffer_list_contains(live, n_live, call_bufs[j]);
    if (keep) {
      kept[n_kept++] = linear->src[i];
      for (int j = 0; j < n_call_bufs; j++)
        if (poly_jit_append_unique_buffer(&live, &n_live, &cap_live, call_bufs[j]) < 0) {
          free(call_bufs);
          goto fail;
        }
    } else {
      onetime[n_onetime++] = linear->src[i];
    }
    free(call_bufs);
  }
  *kept_out = poly_uop(ctx, POLY_OP_LINEAR, POLY_VOID, kept, n_kept, linear->arg);
  *onetime_out = poly_uop(ctx, POLY_OP_LINEAR, POLY_VOID, onetime, n_onetime, linear->arg);
  free(live);
  free(kept);
  free(onetime);
  return *kept_out && *onetime_out ? 0 : -1;

fail:
  free(live);
  free(kept);
  free(onetime);
  return -1;
}

typedef struct {
  PolyUOp **items;
  int count;
  int capacity;
  bool failed;
} HeldBuffers;

static void collect_runtime_buffer(const void *key, void *value, void *userdata) {
  (void)value;
  HeldBuffers *held = userdata;
  if (!held || held->failed || !key) return;
  if (poly_jit_append_unique_buffer(&held->items, &held->count, &held->capacity, (PolyUOp *)key) <
      0)
    held->failed = true;
}

/* Current Tinygrad engine/jit.py:275 holds existing runtime buffers and
 * BUFFERs reachable from live Tensor.uop roots before arena planning. */
int poly_jit_collect_held_bufs(
    PolyCtx *ctx,
    PolyTensor **live_tensors,
    int n_live_tensors,
    PolyUOp ***held_out,
    int *n_held_out
) {
  if (!ctx || n_live_tensors < 0 || (n_live_tensors > 0 && !live_tensors) || !held_out ||
      !n_held_out)
    return -1;
  HeldBuffers held = {0};
  poly_map_foreach(ctx->buffers, collect_runtime_buffer, &held);
  for (int i = 0; i < n_live_tensors && !held.failed; i++) {
    PolyUOp *root = live_tensors[i] ? poly_tensor_uop(live_tensors[i]) : NULL;
    if (!root) continue;
    int n_topo = 0;
    PolyUOp **topo = poly_toposort_alloc(ctx, root, &n_topo);
    if (!topo) {
      held.failed = true;
      break;
    }
    for (int j = 0; j < n_topo; j++)
      if (topo[j]->op == POLY_OP_BUFFER &&
          poly_jit_append_unique_buffer(&held.items, &held.count, &held.capacity, topo[j]) < 0) {
        held.failed = true;
        break;
      }
    poly_toposort_free(topo);
  }
  if (held.failed) {
    free(held.items);
    return -1;
  }
  *held_out = held.items;
  *n_held_out = held.count;
  return 0;
}

/* Current Tinygrad engine/jit.py:jit_lower. */
PolyUOp *poly_jit_lower(
    PolyCtx *ctx,
    PolyUOp *linear,
    PolyUOp **held_bufs,
    int n_held_bufs,
    PolyUOp **input_uops,
    int n_input_uops
) {
  if (!ctx || !linear || linear->op != POLY_OP_LINEAR || n_held_bufs < 0 || n_input_uops < 0 ||
      (n_held_bufs > 0 && !held_bufs) || (n_input_uops > 0 && !input_uops))
    return NULL;
  PolyUOp **params = n_input_uops > 0 ? malloc((size_t)n_input_uops * sizeof(*params)) : NULL;
  if (n_input_uops > 0 && !params) return NULL;
  for (int i = 0; i < n_input_uops; i++) {
    params[i] = poly_uop_param(ctx, i, input_uops[i]);
    if (!params[i]) {
      free(params);
      return NULL;
    }
  }
  PolyUOp *parameterized = n_input_uops > 0
                               ? poly_uop_substitute(ctx, linear, input_uops, params, n_input_uops)
                               : linear;
  free(params);
  PolyUOp *planned = poly_memory_plan_rewrite(ctx, parameterized, held_bufs, n_held_bufs);
  int beam = poly_getenv_int("JITBEAM", poly_get_beam());
  PolyUOp *compiled = planned ? poly_compile_linear(ctx, planned, beam) : NULL;
  if (!compiled || poly_getenv_int("JIT", 1) >= 2) return compiled;
  return poly_graph_split_rewrite(ctx, compiled, poly_getenv_int("JIT_BATCH_SIZE", 32));
}

static int poly_jit_build_captured_linear(
    PolyJit *jit,
    PolyTensor **live_tensors,
    int n_live_tensors
) {
  if (!jit || !jit->ctx || jit->n_linears <= 0) return -1;
  int n_calls = 0;
  for (int i = 0; i < jit->n_linears; i++) {
    if (!jit->linears[i] || jit->linears[i]->op != POLY_OP_LINEAR ||
        jit->linears[i]->n_src > INT_MAX - n_calls)
      return -1;
    n_calls += jit->linears[i]->n_src;
  }
  PolyUOp **calls = n_calls > 0 ? malloc((size_t)n_calls * sizeof(*calls)) : NULL;
  if (n_calls > 0 && !calls) return -1;
  int at = 0;
  for (int i = 0; i < jit->n_linears; i++) {
    if (jit->linears[i]->n_src > 0)
      memcpy(&calls[at], jit->linears[i]->src, (size_t)jit->linears[i]->n_src * sizeof(*calls));
    at += jit->linears[i]->n_src;
  }
  PolyUOp *big_linear =
      poly_uop(jit->ctx, POLY_OP_LINEAR, POLY_VOID, calls, n_calls, poly_arg_none());
  free(calls);
  if (!big_linear) return -1;

  if (jit->prune) {
    PolyUOp *kept = NULL, *onetime = NULL;
    if (poly_prune_linear(
            jit->ctx, big_linear, jit->input_buffers, jit->n_inputs, &kept, &onetime
        ) != 0)
      return -1;
    if (poly_run_linear(
            jit->ctx, onetime, jit->var_bindings, jit->n_var_bindings, NULL, 0, true, false, false
        ) != 0)
      return -1;
    big_linear = kept;
  }

  PolyUOp **held = NULL;
  int n_held = 0;
  if (poly_jit_collect_held_bufs(jit->ctx, live_tensors, n_live_tensors, &held, &n_held) != 0)
    return -1;
  jit->captured_linear =
      poly_jit_lower(jit->ctx, big_linear, held, n_held, jit->input_buffers, jit->n_inputs);
  free(held);
  if (!jit->captured_linear) return -1;
  if (poly_uop_retain(jit->ctx, jit->captured_linear) != 0) return -1;
  jit->captured_root_retained = true;
  jit->n_recorded_linears = jit->n_linears;
  for (int i = 0; i < jit->n_linears; i++)
    poly_uop_release(jit->ctx, jit->linears[i]);
  free(jit->linears);
  jit->linears = NULL;
  jit->n_linears = 0;
  jit->linears_cap = 0;
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

int poly_jit_end_capture(PolyJit *jit, PolyTensor **live_tensors, int n_live_tensors) {
  if (!jit || !jit->ctx || !jit->capturing || n_live_tensors < 0 ||
      (n_live_tensors > 0 && !live_tensors))
    return -1;
  if (jit->ctx->active_jit_capture == jit) jit->ctx->active_jit_capture = NULL;
  jit->capturing = false;
  if (jit->n_linears > 0 &&
      poly_jit_build_captured_linear(jit, live_tensors, n_live_tensors) == 0) {
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
  return jit->captured ? jit->n_recorded_linears : jit->n_linears;
}

PolyUOp *poly_jit_captured_linear(PolyJit *jit) {
  return jit ? jit->captured_linear : NULL;
}

/* Current Tinygrad _TinyJit.add_linear. */
int poly_jit_record_linear(
    PolyJit *jit,
    PolyUOp *linear,
    PolyVarBinding *var_bindings,
    int n_var_bindings
) {
  if (!jit || !jit->capturing || !linear || linear->op != POLY_OP_LINEAR || n_var_bindings < 0 ||
      (n_var_bindings > 0 && !var_bindings))
    return -1;
  if (jit->n_linears >= jit->linears_cap) {
    int new_cap = jit->linears_cap ? jit->linears_cap * 2 : 4;
    PolyUOp **new_linears = realloc(jit->linears, (size_t)new_cap * sizeof(*new_linears));
    if (!new_linears) return -1;
    jit->linears = new_linears;
    jit->linears_cap = new_cap;
  }
  for (int i = 0; i < n_var_bindings; i++) {
    int before = jit->n_var_bindings;
    if (poly_jit_append_var_binding(
            &jit->var_bindings, &jit->n_var_bindings, &jit->var_bindings_cap, var_bindings[i].var,
            var_bindings[i].value
        ) != 0)
      return -1;
    if (jit->n_var_bindings != before && poly_uop_retain(jit->ctx, var_bindings[i].var) != 0) {
      jit->n_var_bindings--;
      return -1;
    }
  }
  /* Current _TinyJit.add_linear keeps every LINEAR live until capture creates
   * CapturedJit (engine/jit.py:193-209,250-281). This retain is C ownership
   * mechanics for the same interval. */
  if (poly_uop_retain(jit->ctx, linear) != 0) return -1;
  jit->linears[jit->n_linears++] = linear;
  return 0;
}

static int poly_jit_override_var_binding(
    PolyVarBinding **items,
    int *n_items,
    int *cap_items,
    PolyUOp *var,
    int32_t value
) {
  for (int i = 0; i < *n_items; i++) {
    if (!poly_jit_same_var((*items)[i].var, var)) continue;
    (*items)[i].value = value;
    return 0;
  }
  return poly_jit_append_var_binding(items, n_items, cap_items, var, value);
}

static int poly_jit_run_captured_linear(
    PolyJit *jit,
    PolyUOp **current_inputs,
    PolyVarBinding *var_bindings,
    int n_var_bindings
) {
  if (!jit || !jit->captured_linear || (jit->n_inputs > 0 && !current_inputs)) return -1;
  PolyVarBinding *effective = NULL;
  int n_effective = 0, cap_effective = 0;
  for (int i = 0; i < jit->n_var_bindings; i++)
    if (poly_jit_override_var_binding(
            &effective, &n_effective, &cap_effective, jit->var_bindings[i].var,
            jit->var_bindings[i].value
        ) != 0)
      goto fail;
  for (int i = 0; i < n_var_bindings; i++)
    if (poly_jit_override_var_binding(
            &effective, &n_effective, &cap_effective, var_bindings[i].var, var_bindings[i].value
        ) != 0)
      goto fail;
  int rc = poly_run_linear(
      jit->ctx, jit->captured_linear, effective, n_effective, current_inputs, jit->n_inputs, true,
      true, false
  );
  free(effective);
  return rc;

fail:
  free(effective);
  return -1;
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
