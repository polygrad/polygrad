/* realize.c -- graph-driven realize: reads buffers from ctx->buffers
 * side table (managed via device.h), no external bindings needed. */

#include "engine/realize.h"
#include "device.h"
#include "ctx.h"
#include "engine/schedule.h"
#include "frontend_internal.h"

#include <stdio.h>
#include <stdlib.h>

/* Current proven top-level view slice from tinygrad's transform_to_call:
 * realize the smaller base compute, then rebuild only the pinned outer views
 * on top of the fresh buffer for frontend retargeting. */
#define POLY_TRANSFORM_TO_CALL_MAX_VIEWS 16
#define POLY_TRANSFORM_TO_CALL_MAX_REDUCE_TMPS 64

static bool poly_transform_to_call_view_op(PolyOps op) {
  return op == POLY_OP_RESHAPE || op == POLY_OP_EXPAND || op == POLY_OP_PAD;
}

typedef struct {
  PolyUOp **stores;
  int n_stores;
  int stores_cap;
  PolyUOp *cached_orig[POLY_TRANSFORM_TO_CALL_MAX_REDUCE_TMPS];
  PolyUOp *cached_repl[POLY_TRANSFORM_TO_CALL_MAX_REDUCE_TMPS];
  int n_cached;
} PolyTransformToCallCtx;

static PolyUOp *poly_transform_to_call_root(PolyUOp *u, PolyUOp **views, int *n_views) {
  PolyUOp *root = u;
  *n_views = 0;
  while (root && root->n_src >= 1 && !poly_uop_has_buffer_identity(root) &&
         poly_transform_to_call_view_op(root->op)) {
    if (*n_views >= POLY_TRANSFORM_TO_CALL_MAX_VIEWS) break;
    views[(*n_views)++] = root;
    root = root->src[0];
  }
  return root ? root : u;
}

static PolyUOp *poly_transform_to_call_rebuild_view(
    PolyCtx *ctx, PolyUOp *buf, PolyShape root_shape, PolyUOp **views, int n_views
) {
  /* Tinygrad's becomes_map rewrites tensors back to their original outer view
   * stack after materializing the base compute into a fresh flat buffer. */
  PolyUOp *view = buf;
  if (root_shape.ndim != 1) view = poly_reshape(ctx, view, root_shape.dims, root_shape.ndim);
  for (int i = n_views - 1; i >= 0; i--) {
    PolyUOp *step = views[i];
    if (step->op == POLY_OP_RESHAPE) {
      view = poly_reshape(ctx, view, step->arg.int_tuple.vals, step->arg.int_tuple.n);
    } else if (step->op == POLY_OP_EXPAND) {
      view = poly_expand(ctx, view, step->arg.int_tuple.vals, step->arg.int_tuple.n);
    } else if (step->op == POLY_OP_PAD) {
      view = poly_pad(ctx, view, step->arg.pair_tuple.pairs, step->arg.pair_tuple.n);
    }
  }
  return view;
}

static PolyUOp *poly_transform_to_call_alloc_buffer_on_device(
    PolyCtx *ctx, PolyDType dtype, PolyShape shape, PolyDevice device
) {
  int64_t numel = (shape.ndim >= 0) ? poly_shape_numel(shape) : 1;
  if (numel < 1) numel = 1;

  PolyDevice out_dev = device;
  if (out_dev == POLY_DEVICE_AUTO) out_dev = poly_device_default();
  PolyUOp *buf = poly_buffer_on_device(ctx, poly_dtype_scalar(dtype), numel, out_dev);
  if (!buf || poly_buffer_allocate(ctx, buf, out_dev) != 0) return NULL;
  return buf;
}

static PolyUOp *poly_transform_to_call_alloc_buffer(PolyCtx *ctx, PolyDType dtype, PolyShape shape) {
  PolyDevice out_dev = poly_ctx_get_preferred_device(ctx);
  if (out_dev == POLY_DEVICE_AUTO) out_dev = poly_device_default();
  return poly_transform_to_call_alloc_buffer_on_device(ctx, dtype, shape, out_dev);
}

static bool poly_transform_to_call_append_store(PolyTransformToCallCtx *tctx, PolyUOp *store) {
  if (!tctx || !store) return false;
  if (tctx->n_stores >= tctx->stores_cap) return false;
  tctx->stores[tctx->n_stores++] = store;
  return true;
}

static bool poly_transform_to_call_after_store_assign(
    PolyUOp *u, PolyUOp **out_target, PolyUOp **out_store
) {
  if (!u || u->op != POLY_OP_AFTER || u->n_src < 2 || !u->src[0]) return false;
  PolyUOp *target = u->src[0];
  for (int i = 1; i < u->n_src; i++) {
    PolyUOp *store = u->src[i];
    if (!store || store->op != POLY_OP_STORE || store->n_src < 2) continue;
    if (store->src[0] != target) continue;
    if (out_target) *out_target = target;
    if (out_store) *out_store = store;
    return true;
  }
  return false;
}

static PolyUOp *poly_transform_to_call_wrap_call(PolyCtx *ctx, PolyUOp *sink) {
  if (!ctx || !sink || sink->op != POLY_OP_SINK) return sink;

  PolyUOp *ordered[POLY_MAX_REALIZE_BUFS] = {0};
  int n_ordered = poly_collect_ordered_buffers(ctx, sink, ordered, POLY_MAX_REALIZE_BUFS);
  int n_src = 1 + n_ordered;
  PolyUOp **src = calloc((size_t)n_src, sizeof(PolyUOp *));
  if (!src) return NULL;

  src[0] = sink;
  for (int i = 0; i < n_ordered; i++)
    src[1 + i] = ordered[i];

  PolyUOp *call = poly_uop(ctx, POLY_OP_CALL, POLY_VOID, src, n_src, poly_arg_none());
  free(src);
  return call;
}

static PolyUOp *poly_transform_to_call_cached_reduce(PolyTransformToCallCtx *tctx, PolyUOp *orig) {
  if (!tctx || !orig) return NULL;
  for (int i = 0; i < tctx->n_cached; i++) {
    if (tctx->cached_orig[i] == orig) return tctx->cached_repl[i];
  }
  return NULL;
}

static bool poly_transform_to_call_cache_reduce(
    PolyTransformToCallCtx *tctx, PolyUOp *orig, PolyUOp *repl
) {
  if (!tctx || !orig || !repl) return false;
  if (tctx->n_cached >= POLY_TRANSFORM_TO_CALL_MAX_REDUCE_TMPS) return false;
  tctx->cached_orig[tctx->n_cached] = orig;
  tctx->cached_repl[tctx->n_cached] = repl;
  tctx->n_cached++;
  return true;
}

static bool poly_transform_to_call_reduce_root(PolyUOp *u) {
  return u && (u->op == POLY_OP_REDUCE_AXIS || u->op == POLY_OP_REDUCE);
}

static bool poly_transform_to_call_depends_on_input_buffer(PolyCtx *ctx, PolyUOp *u) {
  if (!ctx || !u) return false;
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, u, &n_topo);
  for (int i = 0; i < n_topo; i++) {
    PolyUOp *t = topo[i];
    if (!t) continue;
    if (poly_uop_has_buffer_identity(t)) return true;
  }
  return false;
}

static PolyUOp *poly_transform_to_call_materialize_reduce_source(
    PolyCtx *ctx, PolyUOp *u, PolyTransformToCallCtx *tctx
) {
  if (!ctx || !u || !tctx) return NULL;
  PolyUOp *cached = poly_transform_to_call_cached_reduce(tctx, u);
  if (cached) return cached;

  PolyUOp *views[POLY_TRANSFORM_TO_CALL_MAX_VIEWS];
  int n_views = 0;
  PolyUOp *base = poly_transform_to_call_root(u, views, &n_views);
  if (n_views == 0) return NULL;
  if (!poly_transform_to_call_reduce_root(base)) return NULL;
  if (!poly_transform_to_call_depends_on_input_buffer(ctx, base)) return NULL;

  PolyShape base_shape = poly_uop_shape_cached(ctx, base);
  PolyUOp *buf = poly_transform_to_call_alloc_buffer(ctx, u->dtype, base_shape);
  if (!buf) {
    fprintf(stderr, "poly_realize: buffer allocate failed\n");
    return NULL;
  }

  PolyUOp *store = poly_store_val(ctx, buf, base);
  if (!poly_transform_to_call_append_store(tctx, store)) return NULL;

  PolyUOp *replacement = poly_transform_to_call_rebuild_view(ctx, buf, base_shape, views, n_views);
  if (!poly_transform_to_call_cache_reduce(tctx, u, replacement)) return NULL;
  return replacement;
}

static PolyUOp *poly_transform_to_call_rewrite_reduce_sources(
    PolyCtx *ctx, PolyUOp *u, PolyTransformToCallCtx *tctx, bool allow_materialize
) {
  if (!ctx || !u || !tctx) return NULL;
  if (poly_uop_has_buffer_identity(u) || u->n_src == 0) return u;

  PolyUOp *replacement =
      allow_materialize ? poly_transform_to_call_materialize_reduce_source(ctx, u, tctx) : NULL;
  if (replacement) return replacement;

  bool can_descend =
      poly_opset_has(POLY_GROUP_ELEMENTWISE, u->op) || poly_transform_to_call_view_op(u->op);
  if (!can_descend) return u;

  PolyUOp *new_src[16] = {0};
  bool changed = false;
  for (int i = 0; i < u->n_src && i < 16; i++) {
    bool child_allow = allow_materialize;
    if (!child_allow && poly_opset_has(POLY_GROUP_ELEMENTWISE, u->op)) {
      for (int j = 0; j < u->n_src; j++) {
        if (j == i || !u->src[j]) continue;
        if (poly_uop_has_buffer_identity(u->src[j]) ||
            poly_transform_to_call_depends_on_input_buffer(ctx, u->src[j])) {
          child_allow = true;
          break;
        }
      }
    }
    new_src[i] = poly_transform_to_call_rewrite_reduce_sources(ctx, u->src[i], tctx, child_allow);
    if (!new_src[i]) return NULL;
    if (new_src[i] != u->src[i]) changed = true;
  }
  if (!changed) return u;
  return poly_uop(ctx, u->op, u->dtype, new_src, u->n_src, u->arg);
}

/* Partial C analogue of tinygrad's transform_to_call(UOp.sink(...)).
 * Current scope:
 * - batch requested unrealized value UOps into one CALL whose body is a SINK
 *   of STOREs
 * - preserve already-realized values and ASSIGN targets in out_uops
 * - apply the proven top-level RESHAPE(compute) remap before scheduling
 * - materialize reduction-through-view inputs before outer elementwise stores */
PolyUOp *poly_transform_to_call(PolyCtx *ctx, PolyUOp **uops, int n, PolyUOp **out_uops) {
  if (!ctx || !uops || !out_uops || n < 0) return NULL;
  if (n == 0) return NULL;

  PolyUOp **stores = calloc((size_t)(n * 4), sizeof(PolyUOp *));
  if (!stores) return NULL;
  PolyTransformToCallCtx tctx = {
      .stores = stores,
      .n_stores = 0,
      .stores_cap = n * 4,
      .n_cached = 0,
  };

  for (int i = 0; i < n; i++) {
    PolyUOp *u = uops[i];
    if (!u) {
      out_uops[i] = NULL;
      continue;
    }
    if (poly_uop_has_buffer_identity(u)) {
      out_uops[i] = u;
      continue;
    }

    /* Tensor.assign follows tinygrad and builds AFTER(target, STORE(target,
     * value)). Real storage targets run the STORE in-place. Non-storage
     * targets are temporary values, so assignment materializes the stored value
     * instead of inventing a write into an expression such as COPY(...). */
    PolyUOp *assign_target = NULL;
    PolyUOp *assign_store = NULL;
    if (poly_transform_to_call_after_store_assign(u, &assign_target, &assign_store)) {
      if (poly_uop_has_buffer_identity(assign_target)) {
        if (!poly_transform_to_call_append_store(&tctx, assign_store)) {
          free(stores);
          return NULL;
        }
        out_uops[i] = assign_target;
        continue;
      }
      u = assign_store->src[1];
    }

    /* ASSIGN already writes to its target buffer, so keep it as the batched
     * store boundary and report the target buffer as the realized result. */
    if (u->op == POLY_OP_ASSIGN && u->n_src >= 1) {
      if (!poly_transform_to_call_append_store(&tctx, u)) {
        free(stores);
        return NULL;
      }
      out_uops[i] = u->src[0];
      continue;
    }

    PolyUOp *target_views[POLY_TRANSFORM_TO_CALL_MAX_VIEWS];
    int n_target_views = 0;
    PolyUOp *target_root = poly_transform_to_call_root(u, target_views, &n_target_views);
    if (!poly_transform_to_call_reduce_root(target_root)) {
      u = poly_transform_to_call_rewrite_reduce_sources(ctx, u, &tctx, false);
      if (!u) {
        free(stores);
        return NULL;
      }
    }

    PolyUOp *views[POLY_TRANSFORM_TO_CALL_MAX_VIEWS];
    int n_views = 0;
    PolyUOp *materialized = poly_transform_to_call_root(u, views, &n_views);
    PolyShape root_shape = poly_uop_shape_cached(ctx, materialized);
    if (materialized->op == POLY_OP_CONTIGUOUS && materialized->n_src >= 1 &&
        !poly_uop_has_buffer_identity(materialized->src[0])) {
      /* tinygrad engine/allocations.py pm_early_transform_tensor_graph:
       * CONTIGUOUS(src) -> buffer.after(buffer.store(src)).
       * Mirror that boundary here so top-level scalar/materialization paths
       * store the source compute directly into the final buffer instead of
       * scheduling an extra copy kernel for CONTIGUOUS itself. */
      materialized = materialized->src[0];
    }
    PolyDevice out_dev = poly_uop_device(materialized);
    if (out_dev == POLY_DEVICE_AUTO) out_dev = poly_uop_device(u);
    PolyUOp *buf = poly_transform_to_call_alloc_buffer_on_device(ctx, u->dtype, root_shape, out_dev);
    if (!buf) {
      fprintf(stderr, "poly_realize: buffer allocate failed\n");
      free(stores);
      return NULL;
    }

    if (!poly_transform_to_call_append_store(&tctx, poly_store_val(ctx, buf, materialized))) {
      free(stores);
      return NULL;
    }
    out_uops[i] = poly_transform_to_call_rebuild_view(ctx, buf, root_shape, views, n_views);
  }

  if (tctx.n_stores == 0) {
    free(stores);
    return NULL;
  }

  PolyUOp *sink_body = poly_sink_n(ctx, stores, tctx.n_stores);
  free(stores);
  return poly_transform_to_call_wrap_call(ctx, sink_body);
}

static PolySchedule *poly_schedule_effect_sink(PolyCtx *ctx, PolyUOp *sink) {
  if (!ctx || !sink || sink->op != POLY_OP_SINK) {
    fprintf(stderr, "poly_realize: expected effect SINK\n");
    return NULL;
  }

  /* This is the schedule-ready layer. Callers reaching this point already
   * own an effect sink; tensor values must be normalized by transform_to_call
   * before using this helper. */
  PolySchedule *sched = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  if (!sched) fprintf(stderr, "poly_realize: scheduling failed\n");
  return sched;
}

/* C-side analogue of tinygrad's Tensor.schedule_with_vars: normalize requested
 * top-level value UOps into one effect SINK and return its schedule. */
PolySchedule *poly_schedule_with_vars(PolyCtx *ctx, PolyUOp **uops, int n, PolyUOp **out_uops) {
  if (!ctx || !uops || !out_uops || n < 0) return NULL;
  if (n == 0) return NULL;

  /* Only tensor value roots belong on this path. Already-effectful SINKs used
   * by instance/imported graphs run through poly_realize_sink directly. */
  PolyUOp *big_call = poly_transform_to_call(ctx, uops, n, out_uops);
  if (!big_call) return NULL;

  PolyUOp *sched_sink = big_call;
  if (big_call->op == POLY_OP_CALL) {
    if (big_call->n_src < 1 || !big_call->src[0] || big_call->src[0]->op != POLY_OP_SINK) {
      fprintf(stderr, "poly_realize: transform_to_call returned malformed CALL\n");
      return NULL;
    }
    sched_sink = big_call->src[0];
  }

  PolySchedule *sched = poly_schedule_effect_sink(ctx, sched_sink);
  return sched;
}

int poly_realize_sink(PolyCtx *ctx, PolyUOp *sink) {
  PolySchedule *sched = poly_schedule_effect_sink(ctx, sink);
  if (!sched) return -1;
  int ret = poly_run_schedule(ctx, sched, NULL, 0);
  poly_schedule_free(sched);
  return ret;
}

int poly_realize_uops(PolyCtx *ctx, PolyUOp **uops, int n, PolyUOp **out_uops) {
  if (!ctx || !uops || !out_uops || n < 0) return -1;
  if (n == 0) return 0;

  bool needs_run = false;
  for (int i = 0; i < n; i++) {
    PolyUOp *u = uops[i];
    if (u && !poly_uop_has_buffer_identity(u)) {
      needs_run = true;
      break;
    }
  }

  PolySchedule *sched = poly_schedule_with_vars(ctx, uops, n, out_uops);
  if (!sched) return needs_run ? -1 : 0;
  int ret = poly_run_schedule(ctx, sched, NULL, 0);
  poly_schedule_free(sched);
  return ret;
}

int poly_realize_tensors(PolyCtx *ctx, PolyTensor **inputs, int n, PolyTensor **outputs) {
  if (!ctx || !inputs || !outputs || n < 0) return -1;
  if (n == 0) return 0;
  PolyUOp **physical = calloc((size_t)n, sizeof(PolyUOp *));
  PolyUOp **out_uops = calloc((size_t)n, sizeof(PolyUOp *));
  if (!physical) return -1;
  if (!out_uops) {
    free(physical);
    return -1;
  }

  for (int i = 0; i < n; i++) {
    outputs[i] = NULL;
    physical[i] = poly_tensor_physicalize(ctx, inputs[i]);
    if (!physical[i]) {
      free(physical);
      free(out_uops);
      return -1;
    }
  }

  int rc = poly_realize_uops(ctx, physical, n, out_uops);
  if (rc == 0) {
    for (int i = 0; i < n; i++) {
      if (!inputs[i] || !out_uops[i]) continue;
      PolyDevice device = poly_uop_device(out_uops[i]);
      if (device == POLY_DEVICE_AUTO) device = inputs[i]->device;
      if (device == POLY_DEVICE_AUTO) device = poly_ctx_get_preferred_device(ctx);
      if (device == POLY_DEVICE_AUTO) device = poly_device_default();
      if (poly_tensor_update(ctx, inputs[i], NULL, out_uops[i], POLY_TENSOR_VALUE, device) != 0) {
        rc = -1;
        continue;
      }
      outputs[i] = inputs[i];
    }
  }

  free(out_uops);
  free(physical);
  return rc;
}
