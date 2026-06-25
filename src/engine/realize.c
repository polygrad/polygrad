/* realize.c -- graph-driven realize: reads buffers from ctx->buffers
 * side table (managed via device.h), no external bindings needed. */

#include "engine/realize.h"
#include "device.h"
#include "ctx.h"
#include "engine/jit.h"
#include "engine/schedule.h"
#include "frontend_internal.h"
#include "tensor.h"
#include "utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Tinygrad's callify state is list/dict-backed. Keep the same semantics in C:
 * the initial sizes match the old fixed caps, but both grow when needed. */
#define POLY_TRANSFORM_TO_CALL_INITIAL_VIEWS 16
#define POLY_TRANSFORM_TO_CALL_INITIAL_REDUCE_CACHE 64

static bool poly_transform_to_call_view_op(PolyOps op) {
  return op == POLY_OP_RESHAPE || op == POLY_OP_EXPAND || op == POLY_OP_PAD;
}

typedef struct {
  PolyUOp **items;
  int n;
  int cap;
  PolyUOp *stack[POLY_TRANSFORM_TO_CALL_INITIAL_VIEWS];
} PolyTransformViewStack;

typedef struct {
  PolyUOp **stores;
  int n_stores;
  int stores_cap;
  PolyUOp **cached_orig;
  PolyUOp **cached_repl;
  int n_cached;
  int cached_cap;
  bool failed;
} PolyTransformToCallCtx;

static void poly_transform_view_stack_free(PolyTransformViewStack *views) {
  if (!views) return;
  if (views->items != views->stack) free(views->items);
  views->items = NULL;
  views->n = 0;
  views->cap = 0;
}

static bool poly_transform_view_stack_push(PolyTransformViewStack *views, PolyUOp *u) {
  if (!views || !u) return false;
  if (!views->items) {
    views->items = views->stack;
    views->cap = (int)(sizeof(views->stack) / sizeof(views->stack[0]));
  }
  if (views->n >= views->cap) {
    int new_cap = views->cap ? views->cap * 2 : POLY_TRANSFORM_TO_CALL_INITIAL_VIEWS;
    PolyUOp **new_items = NULL;
    if (views->items == views->stack) {
      new_items = malloc((size_t)new_cap * sizeof(PolyUOp *));
      if (new_items) memcpy(new_items, views->stack, (size_t)views->n * sizeof(PolyUOp *));
    } else {
      new_items = realloc(views->items, (size_t)new_cap * sizeof(PolyUOp *));
    }
    if (!new_items) return false;
    views->items = new_items;
    views->cap = new_cap;
  }
  views->items[views->n++] = u;
  return true;
}

static PolyUOp *poly_transform_to_call_root(PolyUOp *u, PolyTransformViewStack *views) {
  PolyUOp *root = u;
  if (views) views->n = 0;
  while (root && root->n_src >= 1 && !poly_uop_has_buffer_identity(root) &&
         poly_transform_to_call_view_op(root->op)) {
    if (!poly_transform_view_stack_push(views, root)) return NULL;
    root = root->src[0];
  }
  return root ? root : u;
}

static PolyUOp *poly_transform_to_call_rebuild_view(
    PolyCtx *ctx,
    PolyUOp *buf,
    PolyShape root_shape,
    const PolyTransformViewStack *views
) {
  /* Tinygrad's becomes_map rewrites tensors back to their original outer view
   * stack after materializing the base compute into a fresh flat buffer. */
  PolyUOp *view = buf;
  if (root_shape.ndim != 1) view = poly_reshape(ctx, view, root_shape.dims, root_shape.ndim);
  int n_views = views ? views->n : 0;
  for (int i = n_views - 1; i >= 0; i--) {
    PolyUOp *step = views->items[i];
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

static PolyUOp *poly_transform_to_call_after_result_buffer(PolyCtx *ctx, PolyUOp *u) {
  PolyTransformViewStack views = {0};
  PolyUOp *root = poly_transform_to_call_root(u, &views);
  if (!root || root->op != POLY_OP_AFTER || root->n_src < 1 ||
      !poly_uop_has_buffer_identity(root->src[0])) {
    poly_transform_view_stack_free(&views);
    return NULL;
  }

  PolyShape root_shape = poly_uop_max_shape_cached(ctx, root);
  PolyUOp *ret = poly_transform_to_call_rebuild_view(ctx, root->src[0], root_shape, &views);
  poly_transform_view_stack_free(&views);
  return ret;
}

static PolyUOp *poly_transform_to_call_alloc_buffer_on_device(
    PolyCtx *ctx,
    PolyDType dtype,
    PolyShape shape,
    PolyDevice device
) {
  int64_t numel = (shape.ndim >= 0) ? poly_shape_numel(shape) : 1;
  if (numel < 1) numel = 1;

  PolyDevice out_dev = device;
  if (out_dev == POLY_DEVICE_AUTO) out_dev = poly_device_default();
  PolyUOp *buf = poly_buffer_on_device(ctx, poly_dtype_scalar(dtype), numel, out_dev);
  if (!buf || poly_buffer_allocate(ctx, buf, out_dev) != 0) return NULL;
  return buf;
}

static PolyUOp *poly_transform_to_call_alloc_buffer(
    PolyCtx *ctx,
    PolyDType dtype,
    PolyShape shape
) {
  PolyDevice out_dev = poly_ctx_get_preferred_device(ctx);
  if (out_dev == POLY_DEVICE_AUTO) out_dev = poly_device_default();
  return poly_transform_to_call_alloc_buffer_on_device(ctx, dtype, shape, out_dev);
}

static bool poly_transform_to_call_append_store(PolyTransformToCallCtx *tctx, PolyUOp *store) {
  if (!tctx || !store) return false;
  if (tctx->n_stores >= tctx->stores_cap) {
    int new_cap = tctx->stores_cap ? tctx->stores_cap * 2 : 8;
    PolyUOp **new_stores = realloc(tctx->stores, (size_t)new_cap * sizeof(PolyUOp *));
    if (!new_stores) return false;
    tctx->stores = new_stores;
    tctx->stores_cap = new_cap;
  }
  tctx->stores[tctx->n_stores++] = store;
  return true;
}

static void poly_transform_to_call_ctx_free(PolyTransformToCallCtx *tctx) {
  if (!tctx) return;
  free(tctx->stores);
  free(tctx->cached_orig);
  free(tctx->cached_repl);
  tctx->stores = NULL;
  tctx->cached_orig = NULL;
  tctx->cached_repl = NULL;
  tctx->n_stores = 0;
  tctx->stores_cap = 0;
  tctx->n_cached = 0;
  tctx->cached_cap = 0;
}

static bool poly_transform_to_call_after_store_assign(
    PolyUOp *u,
    PolyUOp **out_target,
    PolyUOp **out_store
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

static bool poly_transform_to_call_collect_after_stores(
    PolyTransformToCallCtx *tctx,
    PolyUOp *effect
) {
  if (!tctx || !effect) return false;
  if (effect->op == POLY_OP_STORE) return poly_transform_to_call_append_store(tctx, effect);
  if (effect->op != POLY_OP_AFTER) return true;

  /* View assign follows tinygrad's nested shape:
   * AFTER(base_identity, AFTER(view, STORE(view, value))).
   * The outer AFTER returns the storage identity; all nested STORE effects
   * under src[1:] must be scheduled before that identity is considered ready. */
  for (int i = 1; i < effect->n_src; i++) {
    if (!poly_transform_to_call_collect_after_stores(tctx, effect->src[i])) return false;
  }
  return true;
}

static bool poly_transform_to_call_collect_pending_effects(
    PolyTransformToCallCtx *tctx,
    PolyUOp *u,
    PolyMap *visited
) {
  if (!tctx || !u || !visited) return false;
  if (poly_map_get(visited, poly_ptr_hash(u), u, poly_ptr_eq)) return true;
  poly_map_set(visited, poly_ptr_hash(u), u, u, poly_ptr_eq);

  /* A requested view can contain the assign boundary below its root, e.g.
   * SHRINK(AFTER(BUFFER, AFTER(SHRINK(BUFFER), STORE(...)))).
   * Schedule those side-effect stores before materializing the requested
   * value, otherwise view.realize() reads the old base buffer. */
  if (u->op == POLY_OP_AFTER && u->n_src >= 2 && poly_uop_has_buffer_identity(u->src[0])) {
    for (int i = 1; i < u->n_src; i++) {
      if (!poly_transform_to_call_collect_after_stores(tctx, u->src[i])) return false;
    }
  }

  for (int i = 0; i < u->n_src; i++) {
    if (!poly_transform_to_call_collect_pending_effects(tctx, u->src[i], visited)) return false;
  }
  return true;
}

static PolyUOp *poly_transform_to_call_wrap_call(PolyCtx *ctx, PolyUOp *sink) {
  if (!ctx || !sink || sink->op != POLY_OP_SINK) return sink;

  PolyUOp *ordered_stack[16];
  PolyUOp **ordered = ordered_stack;
  int n_ordered = 0;
  n_ordered = poly_collect_ordered_buffers(
      ctx, sink, ordered_stack, (int)(sizeof(ordered_stack) / sizeof(ordered_stack[0]))
  );
  if (n_ordered > (int)(sizeof(ordered_stack) / sizeof(ordered_stack[0]))) {
    ordered = NULL;
    n_ordered = 0;
    if (!poly_collect_ordered_buffers_alloc(ctx, sink, &ordered, &n_ordered)) return NULL;
  }
  int n_src = 1 + n_ordered;
  PolyUOp **src = calloc((size_t)n_src, sizeof(PolyUOp *));
  if (!src) {
    if (ordered != ordered_stack) free(ordered);
    return NULL;
  }

  src[0] = sink;
  for (int i = 0; i < n_ordered; i++)
    src[1 + i] = ordered[i];

  PolyUOp *call = poly_uop(ctx, POLY_OP_CALL, POLY_VOID, src, n_src, poly_arg_none());
  if (ordered != ordered_stack) free(ordered);
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
    PolyTransformToCallCtx *tctx,
    PolyUOp *orig,
    PolyUOp *repl
) {
  if (!tctx || !orig || !repl) return false;
  if (tctx->n_cached >= tctx->cached_cap) {
    int new_cap =
        tctx->cached_cap ? tctx->cached_cap * 2 : POLY_TRANSFORM_TO_CALL_INITIAL_REDUCE_CACHE;
    PolyUOp **new_orig = realloc(tctx->cached_orig, (size_t)new_cap * sizeof(PolyUOp *));
    if (!new_orig) return false;
    tctx->cached_orig = new_orig;
    PolyUOp **new_repl = realloc(tctx->cached_repl, (size_t)new_cap * sizeof(PolyUOp *));
    if (!new_repl) return false;
    tctx->cached_repl = new_repl;
    tctx->cached_cap = new_cap;
  }
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
  PolyScratchMark scratch = poly_ctx_scratch_mark(ctx);
  PolyUOp **topo = poly_toposort_scratch(ctx, u, &n_topo);
  bool depends = false;
  for (int i = 0; i < n_topo; i++) {
    PolyUOp *t = topo[i];
    if (!t) continue;
    if (poly_uop_has_buffer_identity(t)) {
      depends = true;
      break;
    }
  }
  poly_ctx_scratch_rewind(ctx, scratch);
  return depends;
}

static PolyUOp *poly_transform_to_call_materialize_reduce_source(
    PolyCtx *ctx,
    PolyUOp *u,
    PolyTransformToCallCtx *tctx
) {
  if (!ctx || !u || !tctx) return NULL;
  PolyUOp *cached = poly_transform_to_call_cached_reduce(tctx, u);
  if (cached) return cached;

  PolyTransformViewStack views = {0};
  PolyUOp *base = poly_transform_to_call_root(u, &views);
  if (!base) {
    tctx->failed = true;
    poly_transform_view_stack_free(&views);
    return NULL;
  }
  if (views.n == 0 || !poly_transform_to_call_reduce_root(base) ||
      !poly_transform_to_call_depends_on_input_buffer(ctx, base)) {
    poly_transform_view_stack_free(&views);
    return NULL;
  }

  PolyShape base_shape = poly_uop_max_shape_cached(ctx, base);
  PolyUOp *buf = poly_transform_to_call_alloc_buffer(ctx, u->dtype, base_shape);
  if (!buf) {
    fprintf(stderr, "poly_realize: buffer allocate failed\n");
    tctx->failed = true;
    poly_transform_view_stack_free(&views);
    return NULL;
  }

  PolyUOp *replacement = poly_transform_to_call_rebuild_view(ctx, buf, base_shape, &views);
  poly_transform_view_stack_free(&views);
  if (!replacement || !poly_transform_to_call_cache_reduce(tctx, u, replacement)) {
    tctx->failed = true;
    return NULL;
  }

  PolyUOp *store = poly_store_val(ctx, buf, base);
  if (!poly_transform_to_call_append_store(tctx, store)) {
    tctx->failed = true;
    return NULL;
  }
  return replacement;
}

static PolyUOp *poly_transform_to_call_rewrite_reduce_sources(
    PolyCtx *ctx,
    PolyUOp *u,
    PolyTransformToCallCtx *tctx,
    bool allow_materialize
) {
  if (!ctx || !u || !tctx) return NULL;
  if (poly_uop_has_buffer_identity(u) || u->n_src == 0) return u;

  PolyUOp *replacement =
      allow_materialize ? poly_transform_to_call_materialize_reduce_source(ctx, u, tctx) : NULL;
  if (replacement) return replacement;
  if (tctx->failed) return NULL;

  bool can_descend =
      poly_opset_has(POLY_GROUP_ELEMENTWISE, u->op) || poly_transform_to_call_view_op(u->op);
  if (!can_descend) return u;

  PolyUOp *src_buf[16];
  PolyUOp **new_src = src_buf;
  if (u->n_src > (int)(sizeof(src_buf) / sizeof(src_buf[0]))) {
    new_src = malloc((size_t)u->n_src * sizeof(PolyUOp *));
    if (!new_src) {
      tctx->failed = true;
      return NULL;
    }
  }

  bool changed = false;
  for (int i = 0; i < u->n_src; i++) {
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
    if (!new_src[i]) {
      if (new_src != src_buf) free(new_src);
      return NULL;
    }
    if (new_src[i] != u->src[i]) changed = true;
  }

  PolyUOp *ret = changed ? poly_uop(ctx, u->op, u->dtype, new_src, u->n_src, u->arg) : u;
  if (new_src != src_buf) free(new_src);
  return ret;
}

/* Partial C analogue of tinygrad's transform_to_call(UOp.sink(...)).
 * Current scope:
 * - batch requested unrealized value UOps into one CALL whose body is a SINK
 *   of STOREs
 * - preserve already-realized values and ASSIGN targets in out_uops
 * - apply the proven top-level RESHAPE(compute) remap before scheduling
 * - materialize reduction-through-view inputs before outer elementwise stores */
static PolyUOp *poly_transform_to_call_ex(PolyCtx *ctx, PolyUOp **uops, int n, PolyUOp **out_uops) {
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

    if (u->op == POLY_OP_AFTER && u->n_src >= 2 && poly_uop_has_buffer_identity(u->src[0])) {
      int before = tctx.n_stores;
      if (!poly_transform_to_call_collect_after_stores(&tctx, u)) {
        poly_transform_to_call_ctx_free(&tctx);
        return NULL;
      }
      if (tctx.n_stores > before) {
        out_uops[i] = u->src[0];
        continue;
      }
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
          poly_transform_to_call_ctx_free(&tctx);
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
        poly_transform_to_call_ctx_free(&tctx);
        return NULL;
      }
      out_uops[i] = u->src[0];
      continue;
    }

    PolyTransformViewStack target_views = {0};
    PolyUOp *target_root = poly_transform_to_call_root(u, &target_views);
    if (!target_root) {
      poly_transform_view_stack_free(&target_views);
      poly_transform_to_call_ctx_free(&tctx);
      return NULL;
    }
    if (!poly_transform_to_call_reduce_root(target_root)) {
      u = poly_transform_to_call_rewrite_reduce_sources(ctx, u, &tctx, false);
      if (!u) {
        poly_transform_view_stack_free(&target_views);
        poly_transform_to_call_ctx_free(&tctx);
        return NULL;
      }
    }
    poly_transform_view_stack_free(&target_views);

    PolyMap *pending_visited = poly_map_new(64);
    if (!pending_visited) {
      poly_transform_to_call_ctx_free(&tctx);
      return NULL;
    }
    bool pending_ok = poly_transform_to_call_collect_pending_effects(&tctx, u, pending_visited);
    poly_map_destroy(pending_visited);
    if (!pending_ok) {
      poly_transform_to_call_ctx_free(&tctx);
      return NULL;
    }

    PolyUOp *after_result = poly_transform_to_call_after_result_buffer(ctx, u);
    if (after_result) {
      out_uops[i] = after_result;
      continue;
    }

    PolyTransformViewStack views = {0};
    PolyUOp *materialized = poly_transform_to_call_root(u, &views);
    if (!materialized) {
      poly_transform_view_stack_free(&views);
      poly_transform_to_call_ctx_free(&tctx);
      return NULL;
    }
    PolyShape root_shape = poly_uop_max_shape_cached(ctx, materialized);
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
    PolyUOp *buf =
        poly_transform_to_call_alloc_buffer_on_device(ctx, u->dtype, root_shape, out_dev);
    if (!buf) {
      fprintf(stderr, "poly_realize: buffer allocate failed\n");
      poly_transform_view_stack_free(&views);
      poly_transform_to_call_ctx_free(&tctx);
      return NULL;
    }

    if (!poly_transform_to_call_append_store(&tctx, poly_store_val(ctx, buf, materialized))) {
      poly_transform_view_stack_free(&views);
      poly_transform_to_call_ctx_free(&tctx);
      return NULL;
    }
    out_uops[i] = poly_transform_to_call_rebuild_view(ctx, buf, root_shape, &views);
    poly_transform_view_stack_free(&views);
  }

  if (tctx.n_stores == 0) {
    poly_transform_to_call_ctx_free(&tctx);
    return NULL;
  }

  PolyUOp *sink_body = poly_sink_n(ctx, tctx.stores, tctx.n_stores);
  poly_transform_to_call_ctx_free(&tctx);
  PolyUOp *call = poly_transform_to_call_wrap_call(ctx, sink_body);
  if (!call) return NULL;
  return call;
}

PolyUOp *poly_transform_to_call(PolyCtx *ctx, PolyUOp **uops, int n, PolyUOp **out_uops) {
  return poly_transform_to_call_ex(ctx, uops, n, out_uops);
}

static PolySchedule *poly_schedule_effect_sink(PolyCtx *ctx, PolyUOp *sink) {
  if (!ctx || !sink || sink->op != POLY_OP_SINK) {
    fprintf(stderr, "poly_realize: expected effect SINK\n");
    return NULL;
  }

  /* This is the schedule-ready layer. Callers reaching this point already
   * own an effect sink; tensor values must be normalized by transform_to_call
   * before using this helper. */
  bool timing = poly_debug_at_least(2);
  double t0 = timing ? poly_now_ms() : 0.0;
  if (timing) {
    fprintf(
        stderr, "[polygrad:schedule_effect] begin sink=%p n_src=%d\n", (void *)sink, sink->n_src
    );
    fflush(stderr);
  }
  PolySchedule *sched = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  if (!sched) fprintf(stderr, "poly_realize: scheduling failed\n");
  if (timing) {
    double t1 = poly_now_ms();
    fprintf(
        stderr,
        "[polygrad:schedule_effect] done schedule=%.3fms calls=%d slots=%d default_vars=%d\n",
        t1 - t0, sched ? sched->template->n_calls : -1, sched ? sched->template->n_buf_slots : -1,
        sched ? sched->template->n_default_vars : -1
    );
    fflush(stderr);
  }
  return sched;
}

/* C-side analogue of tinygrad's Tensor.schedule_with_vars: normalize requested
 * top-level value UOps into one effect SINK and return its schedule. */
static PolySchedule *poly_schedule_with_vars_ex(
    PolyCtx *ctx,
    PolyUOp **uops,
    int n,
    PolyUOp **out_uops
) {
  if (!ctx || !uops || !out_uops || n < 0) return NULL;
  if (n == 0) return NULL;

  /* Only tensor value roots belong on this path. Already-effectful SINKs used
   * by instance/imported graphs run through poly_realize_sink directly. */
  PolyUOp *big_call = poly_transform_to_call_ex(ctx, uops, n, out_uops);
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

PolySchedule *poly_schedule_with_vars(PolyCtx *ctx, PolyUOp **uops, int n, PolyUOp **out_uops) {
  return poly_schedule_with_vars_ex(ctx, uops, n, out_uops);
}

int poly_realize_sink(PolyCtx *ctx, PolyUOp *sink) {
  bool timing = poly_debug_at_least(2);
  double t0 = timing ? poly_now_ms() : 0.0;
  if (timing) {
    fprintf(stderr, "[polygrad:realize_sink] begin sink=%p\n", (void *)sink);
    fflush(stderr);
  }
  PolySchedule *sched = poly_schedule_effect_sink(ctx, sink);
  double t_sched = timing ? poly_now_ms() : 0.0;
  if (!sched) return -1;
  if (timing) {
    fprintf(stderr, "[polygrad:realize_sink] run begin calls=%d\n", sched->template->n_calls);
    fflush(stderr);
  }
  int ret = poly_run_schedule(ctx, sched, NULL, 0);
  double t_run = timing ? poly_now_ms() : 0.0;
  poly_schedule_free(sched);
  if (timing) {
    fprintf(
        stderr, "[polygrad:realize_sink] schedule=%.3fms run=%.3fms total=%.3fms ret=%d\n",
        t_sched - t0, t_run - t_sched, t_run - t0, ret
    );
  }
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

  PolySchedule *sched = poly_schedule_with_vars_ex(ctx, uops, n, out_uops);
  if (!sched) {
    bool all_outputs_resolved = true;
    for (int i = 0; i < n; i++) {
      if (!out_uops[i]) {
        all_outputs_resolved = false;
        break;
      }
    }
    return (!needs_run || all_outputs_resolved) ? 0 : -1;
  }
  bool captured = false;
  PolyJit *cap = ctx->active_jit_capture;
  if (poly_jit_is_capturing(cap)) {
    if (poly_jit_record_schedule(cap, sched) != 0) {
      poly_schedule_free(sched);
      return -1;
    }
    captured = true;
  }
  int ret = poly_run_schedule(ctx, sched, NULL, 0);
  if (!captured) poly_schedule_free(sched);
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
