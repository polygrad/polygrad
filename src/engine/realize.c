/* realize.c -- graph-driven realize: reads buffers from ctx->buffers
 * side table (managed via device.h), no external bindings needed.
 *
 * This is the tensor-style realize path. The explicit bindings API lives in
 * frontend.c as poly_realize_with_bindings(...).
 */

#include "engine/realize.h"
#include "device.h"
#include "ctx.h"
#include "engine/schedule.h"
#include "frontend.h"
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

static PolyUOp *poly_transform_to_call_alloc_buffer(PolyCtx *ctx, PolyDType dtype, PolyShape shape) {
  int64_t numel = (shape.ndim >= 0) ? poly_shape_numel(shape) : 1;
  if (numel < 1) numel = 1;

  PolyUOp *buf = poly_buffer(ctx, poly_dtype_scalar(dtype), numel);
  PolyDevice out_dev = poly_ctx_get_preferred_device(ctx);
  if (out_dev == POLY_DEVICE_AUTO) out_dev = poly_device_default();
  if (!buf || poly_buffer_allocate(ctx, buf, out_dev) != 0) return NULL;
  return buf;
}

static bool poly_transform_to_call_append_store(PolyTransformToCallCtx *tctx, PolyUOp *store) {
  if (!tctx || !store) return false;
  if (tctx->n_stores >= tctx->stores_cap) return false;
  tctx->stores[tctx->n_stores++] = store;
  return true;
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
    PolyUOp *buf = poly_transform_to_call_alloc_buffer(ctx, u->dtype, root_shape);
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

/* C-side analogue of tinygrad's Tensor.schedule_with_vars: normalize the
 * requested top-level values into one realizable SINK and return the schedule.
 * The actual execution still happens in poly_realize below. */
PolySchedule *poly_schedule_with_vars(PolyCtx *ctx, PolyUOp **uops, int n, PolyUOp **out_uops) {
  if (!ctx || !uops || !out_uops || n < 0) return NULL;
  if (n == 0) return NULL;
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

  /* complete_create_schedule_with_vars strips BIND internally and stores the
   * default vars on the schedule so poly_run_schedule only needs runtime
   * overrides. */
  PolySchedule *sched = poly_complete_create_schedule_with_vars(ctx, sched_sink, POLY_MODE_CALL);
  if (!sched) fprintf(stderr, "poly_realize: scheduling failed\n");
  return sched;
}

int poly_realize(PolyCtx *ctx, PolyUOp **uops, int n, PolyUOp **out_uops) {
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
