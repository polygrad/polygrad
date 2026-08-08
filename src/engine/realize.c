/* realize.c -- graph-driven realize: reads buffers from ctx->buffers
 * side table (managed via device.h), no external bindings needed. */

#include "engine/realize.h"
#include "device.h"
#include "ctx.h"
#include "engine/jit.h"
#include "engine/schedule.h"
#include "frontend_internal.h"
#include "pat.h"
#include "tensor.h"
#include "utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Tinygrad's callify state is list/dict-backed. Keep the same semantics in C:
 * the initial sizes match the old fixed caps, but both grow when needed. */
#define POLY_TRANSFORM_TO_CALL_INITIAL_VIEWS 16
#define POLY_TRANSFORM_TO_CALL_INITIAL_REPLACEMENTS 64

static bool poly_transform_to_call_view_op(PolyOps op) {
  /* Pinned UOp.base/multibase and callify's becomes-map preserve the complete
   * GroupOp.Movement set (uop/ops.py:675-686). POLY_GROUP_MOVEMENT is the
   * exact existing six-op counterpart; using a subset rematerializes requested
   * PERMUTE/SHRINK/FLIP views after their base was already materialized. */
  return poly_opset_has(POLY_GROUP_MOVEMENT, op);
}

static bool poly_transform_to_call_after_result_view_op(PolyOps op) {
  return poly_transform_to_call_view_op(op) || op == POLY_OP_CONTIGUOUS;
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
  PolyUOp **calls;
  int n_calls;
  int calls_cap;
  PolyUOp **cached_orig;
  PolyUOp **cached_repl;
  int n_cached;
  int cached_cap;
  PolyMap *requested_bases;
  PolyMap *view_copy_memo;
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

/* Pinned UOp.multibase strips a leading movement/DETACH and then follows the
 * complete movement/MULTI/DETACH base chain (uop/ops.py:675-686). Callify uses
 * those exact requested bases to prevent a requested descendant from
 * recomputing a requested parent. */
static PolyUOp *poly_transform_to_call_multibase(PolyUOp *u) {
  if (!u) return NULL;
  bool enters_base = poly_opset_has(POLY_GROUP_MOVEMENT, u->op) || u->op == POLY_OP_DETACH;
  if (!enters_base || u->n_src < 1) return u;
  PolyUOp *base = u->src[0];
  while (base && base->n_src >= 1 &&
         (poly_opset_has(POLY_GROUP_MOVEMENT, base->op) || base->op == POLY_OP_MULTI ||
          base->op == POLY_OP_DETACH))
    base = base->src[0];
  return base ? base : u;
}

static PolyUOp *poly_transform_to_call_after_result_root(PolyUOp *u, PolyTransformViewStack *views) {
  PolyUOp *root = u;
  if (views) views->n = 0;
  while (root && root->n_src >= 1 && !poly_uop_has_buffer_identity(root) &&
         poly_transform_to_call_after_result_view_op(root->op)) {
    if (!poly_transform_view_stack_push(views, root)) return NULL;
    root = root->src[0];
  }
  return root ? root : u;
}

static PolyUOp *poly_transform_to_call_canonical_shape_dim(
    PolyCtx *ctx, PolyUOp *u, PolyShape shape, int dim
) {
  if (!ctx || !u || dim < 0 || dim >= shape.ndim) return NULL;
  PolyUOp *value = poly_uop_shape_dim(ctx, u, dim);
  if (!value)
    value =
        poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(shape.dims[dim]));
  if (value && !poly_dtype_eq(poly_dtype_scalar(value->dtype), POLY_INDEX))
    value = poly_uop1(ctx, POLY_OP_CAST, POLY_INDEX, value, poly_arg_none());
  return value ? poly_graph_rewrite(ctx, value, poly_symbolic()) : NULL;
}

static bool poly_transform_to_call_same_exact_shape(
    PolyCtx *ctx, PolyUOp *a, PolyShape a_shape, PolyUOp *b, PolyShape b_shape
) {
  if (!ctx || !a || !b || a_shape.ndim != b_shape.ndim || a_shape.ndim < 0)
    return false;
  for (int i = 0; i < a_shape.ndim; i++) {
    PolyUOp *a_dim =
        poly_transform_to_call_canonical_shape_dim(ctx, a, a_shape, i);
    PolyUOp *b_dim =
        poly_transform_to_call_canonical_shape_dim(ctx, b, b_shape, i);
    if (!a_dim || !b_dim || a_dim != b_dim) return false;
  }
  return true;
}

static bool poly_transform_to_call_is_static_max_shape(
    PolyCtx *ctx, PolyUOp *u, PolyShape shape
) {
  if (!ctx || !u || shape.ndim < 0) return false;
  for (int i = 0; i < shape.ndim; i++) {
    PolyUOp *dim =
        poly_transform_to_call_canonical_shape_dim(ctx, u, shape, i);
    int64_t value = 0;
    if (!dim || poly_uop_const_i64(dim, &value) != 0 ||
        value != shape.dims[i])
      return false;
  }
  return true;
}

static PolyUOp *poly_transform_to_call_rebuild_view(
    PolyCtx *ctx,
    PolyUOp *buf,
    PolyUOp *root,
    const PolyTransformViewStack *views
) {
  if (!ctx || !buf || !root) return NULL;
  PolyShape root_shape = poly_uop_max_shape_cached(ctx, root);
  if (root_shape.ndim < 0) return NULL;

  /* Pinned callify.py:180 maps the materialized maximum-size buffer through
   * shrink_to(original_uop.shape) before rebuilding outer views. Pinned
   * movement.py:173-194 returns the input unchanged when it already has that
   * exact shape; only raw/static-max allocation storage is normalized here. */
  PolyUOp *view = buf;
  PolyShape view_shape = poly_uop_max_shape_cached(ctx, view);
  if (view_shape.ndim < 0) return NULL;
  if (!poly_transform_to_call_same_exact_shape(
          ctx, view, view_shape, root, root_shape
      )) {
    bool at_static_max =
        poly_shape_eq(view_shape, root_shape) &&
        poly_transform_to_call_is_static_max_shape(ctx, view, view_shape);
    if (!at_static_max) {
      int64_t root_numel = poly_shape_numel(root_shape);
      if (view_shape.ndim != 1 || root_numel < 0 ||
          view_shape.dims[0] != root_numel)
        return NULL;
      view = poly_reshape(ctx, view, root_shape.dims, root_shape.ndim);
      if (!view) return NULL;
    }

    PolyUOp *starts[POLY_MAX_DIMS];
    PolyUOp *sizes[POLY_MAX_DIMS];
    PolyUOp *zero =
        poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(0));
    for (int i = 0; i < root_shape.ndim; i++) {
      starts[i] = zero;
      sizes[i] = poly_uop_shape_dim(ctx, root, i);
      if (!sizes[i])
        sizes[i] = poly_uop0(
            ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(root_shape.dims[i]));
      if (!starts[i] || !sizes[i]) return NULL;
    }
    if (!poly_transform_to_call_is_static_max_shape(ctx, root, root_shape)) {
      view = poly_shrink_uop(ctx, view, starts, sizes, root_shape.ndim);
      if (!view) return NULL;
    }
  }
  int n_views = views ? views->n : 0;
  for (int i = n_views - 1; i >= 0; i--) {
    PolyUOp *step = views->items[i];
    if (step->op == POLY_OP_RESHAPE) {
      if (step->arg.kind != POLY_ARG_NONE || step->n_src != 2)
        return NULL;
      /* Pinned callify substitutes only the materialized base in the immutable
       * movement graph (callify.py:204-220, tensor.py:202-206). Preserve the
       * exact symbolic RESHAPE shape source instead of its cached max shape. */
      PolyUOp *reshape_src[2] = {view, step->src[1]};
      view = poly_uop(
          ctx, POLY_OP_RESHAPE, step->dtype, reshape_src, 2, poly_arg_none());
    } else if (step->op == POLY_OP_EXPAND && step->arg.kind == POLY_ARG_NONE &&
               step->n_src == 2) {
      /* Same immutable-source substitution as RESHAPE and pinned callify.py:
       * replace only the materialized base, not the shape-value UOp. */
      PolyUOp *expand_src[2] = {view, step->src[1]};
      view = poly_uop(
          ctx, POLY_OP_EXPAND, step->dtype, expand_src, 2, poly_arg_none());
    } else if (step->op == POLY_OP_PERMUTE && step->arg.kind == POLY_ARG_INT_TUPLE) {
      view = poly_permute(ctx, view, step->arg.int_tuple.vals, step->arg.int_tuple.n);
    } else if (step->op == POLY_OP_PAD && step->arg.kind == POLY_ARG_NONE &&
               step->n_src == 3 && step->src[1]->op == POLY_OP_STACK &&
               step->src[2]->op == POLY_OP_STACK &&
               step->src[1]->n_src == step->src[2]->n_src) {
      view = poly_pad_uop(ctx, view, step->src[1]->src, step->src[2]->src, step->src[1]->n_src);
    } else if (step->op == POLY_OP_PAD && step->arg.kind == POLY_ARG_PAIR_TUPLE) {
      view = poly_pad(ctx, view, step->arg.pair_tuple.pairs, step->arg.pair_tuple.n);
    } else if (step->op == POLY_OP_SHRINK && step->arg.kind == POLY_ARG_PAIR_TUPLE) {
      view = poly_shrink(ctx, view, step->arg.pair_tuple.pairs, step->arg.pair_tuple.n);
    } else if (step->op == POLY_OP_SHRINK && step->arg.kind == POLY_ARG_NONE &&
               step->n_src >= 3 && step->src[1]->op == POLY_OP_STACK &&
               step->src[2]->op == POLY_OP_STACK) {
      view = poly_shrink_uop(ctx, view, step->src[1]->src, step->src[2]->src, step->src[1]->n_src);
    } else if (step->op == POLY_OP_FLIP && step->arg.kind == POLY_ARG_INT_TUPLE) {
      view = poly_uop1(ctx, POLY_OP_FLIP, view->dtype, view, step->arg);
    }
  }
  return view;
}

static PolyUOp *poly_transform_to_call_after_result_buffer(PolyCtx *ctx, PolyUOp *u) {
  PolyTransformViewStack views = {0};
  PolyUOp *root = poly_transform_to_call_after_result_root(u, &views);
  if (!root || root->op != POLY_OP_AFTER || root->n_src < 1) {
    poly_transform_view_stack_free(&views);
    return NULL;
  }

  /* Pinned tinygrad callify.apply_after strips the complete pending-version
   * chain, not just one AFTER layer. Distinct STORE nodes remain distinct
   * effects in the CALL body; this only selects their shared result storage. */
  PolyUOp *base = root->src[0];
  while (base && base->op == POLY_OP_AFTER && base->n_src >= 1)
    base = base->src[0];

  /* A STORE target may be a writable movement view rather than a contiguous
   * buffer identity. Pinned tinygrad keeps raw
   * AFTER(view, STORE(view, value)) in the assignment sink and returns the
   * view after executing it; it does not replace the write with a fresh
   * materialization. Follow the same movement/AFTER chain only to prove that
   * storage exists, while preserving the exact view as the realized result. */
  PolyUOp *storage = base;
  while (storage && !poly_uop_has_buffer_identity(storage)) {
    if (storage->op == POLY_OP_AFTER && storage->n_src >= 1) {
      storage = storage->src[0];
      continue;
    }
    if ((poly_opset_has(POLY_GROUP_MOVEMENT, storage->op) || storage->op == POLY_OP_BITCAST) &&
        storage->n_src >= 1) {
      storage = storage->src[0];
      continue;
    }
    storage = NULL;
  }
  if (!base || !storage) {
    poly_transform_view_stack_free(&views);
    return NULL;
  }

  PolyUOp *ret = poly_transform_to_call_rebuild_view(ctx, base, root, &views);
  poly_transform_view_stack_free(&views);
  return ret;
}

/* Pinned callify's UOp.empty_like creates only a BUFFER identity. Physical
 * allocation belongs to exec_copy/exec_kernel, where CALL output slots are
 * prepared. Keeping construction lazy also lets a LINEAR be inspected or
 * cached without opening its eventual backend. */
static PolyUOp *poly_transform_to_call_empty_buffer_on_device(
    PolyCtx *ctx,
    PolyDType dtype,
    PolyShape shape,
    PolyDevice device
) {
  int64_t numel = 1;
  if (shape.ndim >= 0) {
    numel = poly_shape_numel(shape);
    if (numel < 0) return NULL;
  }

  PolyDevice out_dev = device;
  if (out_dev == POLY_DEVICE_AUTO) out_dev = poly_device_default();
  return poly_buffer_on_device(ctx, poly_dtype_scalar(dtype), numel, out_dev);
}

static bool poly_transform_to_call_append_store(PolyTransformToCallCtx *tctx, PolyUOp *store) {
  if (!tctx || !store) return false;
  /* graph_rewrite visits one immutable UOp once. Preserve that exact-identity
   * behavior when several requested roots share the same effect. Never merge
   * different STORE versions merely because they write the same buffer. */
  for (int i = 0; i < tctx->n_stores; i++)
    if (tctx->stores[i] == store) return true;
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

static bool poly_transform_to_call_append_call(PolyTransformToCallCtx *tctx, PolyUOp *call) {
  if (!tctx || !call || call->op != POLY_OP_CALL) return false;
  for (int i = 0; i < tctx->n_calls; i++)
    if (tctx->calls[i] == call) return true;
  if (tctx->n_calls >= tctx->calls_cap) {
    int new_cap = tctx->calls_cap ? tctx->calls_cap * 2 : 4;
    PolyUOp **new_calls = realloc(tctx->calls, (size_t)new_cap * sizeof(PolyUOp *));
    if (!new_calls) return false;
    tctx->calls = new_calls;
    tctx->calls_cap = new_cap;
  }
  tctx->calls[tctx->n_calls++] = call;
  return true;
}

static bool poly_transform_to_call_contains_uop(PolyUOp *u, PolyUOp *needle, PolyMap *visited) {
  if (!u || !needle || !visited) return false;
  if (u == needle) return true;
  if (poly_map_get(visited, poly_ptr_hash(u), u, poly_ptr_eq)) return false;
  poly_map_set(visited, poly_ptr_hash(u), u, u, poly_ptr_eq);
  for (int i = 0; i < u->n_src; i++)
    if (poly_transform_to_call_contains_uop(u->src[i], needle, visited)) return true;
  return false;
}

static bool poly_transform_to_call_arg_depends_on_call(PolyUOp *arg, PolyUOp *producer) {
  PolyMap *visited = poly_map_new(64);
  if (!visited) return false;
  bool ret = poly_transform_to_call_contains_uop(arg, producer, visited);
  poly_map_destroy(visited);
  return ret;
}

static bool poly_transform_to_call_call_depends_on_call(PolyUOp *consumer, PolyUOp *producer) {
  if (!consumer || consumer->op != POLY_OP_CALL || !producer || producer->op != POLY_OP_CALL)
    return false;
  for (int i = 1; i < consumer->n_src; i++)
    if (poly_transform_to_call_arg_depends_on_call(consumer->src[i], producer)) return true;
  return false;
}

static void poly_transform_to_call_order_calls(PolyTransformToCallCtx *tctx) {
  if (!tctx || tctx->n_calls <= 1) return;
  bool changed = true;
  while (changed) {
    changed = false;
    for (int i = 0; i < tctx->n_calls; i++) {
      for (int j = i + 1; j < tctx->n_calls; j++) {
        if (!poly_transform_to_call_call_depends_on_call(tctx->calls[i], tctx->calls[j]))
          continue;
        PolyUOp *tmp = tctx->calls[i];
        tctx->calls[i] = tctx->calls[j];
        tctx->calls[j] = tmp;
        changed = true;
      }
    }
  }
}

static void poly_transform_to_call_ctx_free(PolyTransformToCallCtx *tctx) {
  if (!tctx) return;
  free(tctx->stores);
  free(tctx->calls);
  free(tctx->cached_orig);
  free(tctx->cached_repl);
  poly_map_destroy(tctx->requested_bases);
  poly_map_destroy(tctx->view_copy_memo);
  tctx->stores = NULL;
  tctx->calls = NULL;
  tctx->cached_orig = NULL;
  tctx->cached_repl = NULL;
  tctx->requested_bases = NULL;
  tctx->view_copy_memo = NULL;
  tctx->n_stores = 0;
  tctx->stores_cap = 0;
  tctx->n_calls = 0;
  tctx->calls_cap = 0;
  tctx->n_cached = 0;
  tctx->cached_cap = 0;
}

/* Pinned RewriteContext rebuilds a source-changing UOp with the original
 * op/dtype/arg/tag. Polygrad's tag is split into tag/tag_arg, so both fields
 * are part of the same reconstruction identity. */
static PolyUOp *poly_rebuild_with_sources(
    PolyCtx *ctx,
    PolyUOp *u,
    PolyUOp **src
) {
  if (!ctx || !u || (u->n_src > 0 && !src)) return NULL;
  return (u->tag != 0 || u->tag_arg.kind != POLY_ARG_NONE)
             ? poly_uop_tagged_arg(
                   ctx, u->op, u->dtype, src, u->n_src, u->arg, u->tag,
                   u->tag_arg
               )
             : poly_uop(ctx, u->op, u->dtype, src, u->n_src, u->arg);
}

static PolyUOp *poly_transform_to_call_fail(
    PolyTransformToCallCtx *tctx,
    PolyUOp **out_uops,
    int n
) {
  if (out_uops)
    for (int i = 0; i < n; i++) out_uops[i] = NULL;
  poly_transform_to_call_ctx_free(tctx);
  return NULL;
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

static bool poly_transform_to_call_cache_replacement(
    PolyTransformToCallCtx *tctx,
    PolyUOp *orig,
    PolyUOp *repl
);

/* Read-only analogue of pinned callify.add_tags/apply_after. Record the
 * executable result of every reachable AFTER before later callification
 * rewrites replace nested CONTIGUOUS/COPY nodes. CALL/FUNCTION bodies remain
 * opaque; only their caller-visible arguments participate in the map. */
static bool poly_transform_to_call_apply_after(
    PolyTransformToCallCtx *tctx,
    PolyUOp *u,
    PolyMap *visited
) {
  if (!tctx || !u || !visited) return false;
  if (poly_map_get(visited, poly_ptr_hash(u), u, poly_ptr_eq)) return true;
  poly_map_set(visited, poly_ptr_hash(u), u, u, poly_ptr_eq);

  if (u->op == POLY_OP_AFTER && u->n_src >= 1) {
    PolyUOp *base = u->src[0];
    while (base && base->op == POLY_OP_AFTER && base->n_src >= 1) base = base->src[0];
    if (!base || !poly_transform_to_call_cache_replacement(tctx, u, base)) return false;
  }

  int first_src =
      ((u->op == POLY_OP_CALL || u->op == POLY_OP_FUNCTION) && u->n_src > 0) ? 1 : 0;
  for (int i = first_src; i < u->n_src; i++)
    if (!poly_transform_to_call_apply_after(tctx, u->src[i], visited)) return false;
  return true;
}

/* C ownership form of tinygrad transform_to_call's returned buffer_map. The
 * physical transform owns these already-existing original/replacement arrays
 * while it runs; the Tensor.linear_with_vars analogue takes them before the
 * transform context is destroyed and applies them outside callify. */
static void poly_transform_to_call_take_replacements(
    PolyTransformToCallCtx *tctx,
    PolyUOp ***out_orig,
    PolyUOp ***out_repl,
    int *out_n
) {
  if (!tctx || !out_orig || !out_repl || !out_n) return;
  *out_orig = tctx->cached_orig;
  *out_repl = tctx->cached_repl;
  *out_n = tctx->n_cached;
  tctx->cached_orig = NULL;
  tctx->cached_repl = NULL;
  tctx->n_cached = 0;
  tctx->cached_cap = 0;
}

static bool poly_transform_to_call_collect_after_stores(
    PolyTransformToCallCtx *tctx,
    PolyUOp *effect
);

static bool poly_transform_to_call_collect_after_stores_ex(
    PolyTransformToCallCtx *tctx,
    PolyUOp *effect,
    bool *out_found
);

static bool poly_transform_to_call_collect_arg_effects(
    PolyTransformToCallCtx *tctx,
    PolyUOp *u,
    PolyMap *visited
) {
  if (!tctx || !u || !visited) return false;
  if (poly_map_get(visited, poly_ptr_hash(u), u, poly_ptr_eq)) return true;
  poly_map_set(visited, poly_ptr_hash(u), u, u, poly_ptr_eq);

  if (u->op == POLY_OP_CALL)
    return poly_transform_to_call_collect_after_stores(tctx, u);

  if (u->op == POLY_OP_AFTER && u->n_src >= 2 && poly_uop_has_buffer_identity(u->src[0])) {
    PolyUOp *base = u->src[0];
    while (base && base->op == POLY_OP_AFTER && base->n_src >= 1) base = base->src[0];
    if (!base || !poly_transform_to_call_cache_replacement(tctx, u, base)) return false;
    for (int i = 1; i < u->n_src; i++)
      if (!poly_transform_to_call_collect_after_stores(tctx, u->src[i])) return false;
  }

  int first_src = (u->op == POLY_OP_FUNCTION && u->n_src > 0) ? 1 : 0;
  for (int i = first_src; i < u->n_src; i++)
    if (!poly_transform_to_call_collect_arg_effects(tctx, u->src[i], visited)) return false;
  return true;
}

static bool poly_transform_to_call_collect_after_stores(
    PolyTransformToCallCtx *tctx,
    PolyUOp *effect
) {
  return poly_transform_to_call_collect_after_stores_ex(tctx, effect, NULL);
}

static bool poly_transform_to_call_collect_after_stores_ex(
    PolyTransformToCallCtx *tctx,
    PolyUOp *effect,
    bool *out_found
) {
  if (!tctx || !effect) return false;
  if (effect->op == POLY_OP_STORE) {
    if (out_found) *out_found = true;
    return poly_transform_to_call_append_store(tctx, effect);
  }
  if (effect->op == POLY_OP_CALL) {
    PolyMap *visited = poly_map_new(64);
    if (!visited) return false;
    bool ok = true;
    for (int i = 1; i < effect->n_src && ok; i++)
      ok = poly_transform_to_call_collect_arg_effects(tctx, effect->src[i], visited);
    poly_map_destroy(visited);
    if (ok && out_found) *out_found = true;
    return ok && poly_transform_to_call_append_call(tctx, effect);
  }
  if (effect->op != POLY_OP_AFTER) return true;

  PolyUOp *target = NULL;
  if (poly_transform_to_call_after_store_assign(effect, &target, NULL)) {
    /* Keep the canonical AFTER+STORE effect root. This is the identity that a
     * downstream assignment consumes and the unit tinygrad split_kernels
     * rewrites once. Extracting only STORE loses that sharing and executes the
     * same immutable effect twice when it is also nested in another target. */
    PolyUOp *base = target;
    while (base && base->op == POLY_OP_AFTER && base->n_src >= 1) base = base->src[0];
    if (!base || !poly_transform_to_call_cache_replacement(tctx, effect, base)) return false;
    if (out_found) *out_found = true;
    return poly_transform_to_call_append_store(tctx, effect);
  }

  /* View assign follows tinygrad's nested shape:
   * AFTER(base_identity, AFTER(view, STORE(view, value))).
   * The outer AFTER returns the storage identity; all nested STORE effects
   * under src[1:] must be scheduled before that identity is considered ready. */
  for (int i = 1; i < effect->n_src; i++) {
    if (!poly_transform_to_call_collect_after_stores_ex(
            tctx, effect->src[i], out_found))
      return false;
  }
  if (effect->n_src >= 1) {
    PolyUOp *base = effect->src[0];
    while (base && base->op == POLY_OP_AFTER && base->n_src >= 1) base = base->src[0];
    if (!base || !poly_transform_to_call_cache_replacement(tctx, effect, base)) return false;
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

  /* CALL/FUNCTION bodies are opaque. Their internal STORE/AFTER nodes must not
   * be re-collected into the caller's materialization sink. */
  if (u->op == POLY_OP_CALL || u->op == POLY_OP_FUNCTION) return true;

  /* Pinned pm_finalize_call rewrites bottom-up: nested canonical AFTER
   * dependencies are recorded before the current AFTER. Recurse through the
   * target and STORE value, but not through the owned raw STORE node, then
   * append this whole immutable effect as the canonical assignment unit. */
  PolyUOp *canonical_target = NULL;
  PolyUOp *canonical_store = NULL;
  if (poly_transform_to_call_after_store_assign(
          u, &canonical_target, &canonical_store)) {
    if (!poly_transform_to_call_collect_pending_effects(
            tctx, canonical_target, visited))
      return false;
    if (!poly_transform_to_call_collect_pending_effects(
            tctx, canonical_store->src[1], visited))
      return false;
    return poly_transform_to_call_collect_after_stores(tctx, u);
  }

  /* A requested view can contain the assign boundary below its root, e.g.
   * SHRINK(AFTER(BUFFER, AFTER(SHRINK(BUFFER), STORE(...)))).
   * Schedule those side-effect stores before materializing the requested
   * value, otherwise view.realize() reads the old base buffer. */
  if (u->op == POLY_OP_AFTER && u->n_src >= 2 && poly_uop_has_buffer_identity(u->src[0])) {
    PolyUOp *base = u->src[0];
    while (base && base->op == POLY_OP_AFTER && base->n_src >= 1) base = base->src[0];
    if (!base || !poly_transform_to_call_cache_replacement(tctx, u, base)) return false;
    for (int i = 1; i < u->n_src; i++) {
      if (!poly_transform_to_call_collect_after_stores(tctx, u->src[i])) return false;
    }
  }

  for (int i = 0; i < u->n_src; i++) {
    if (!poly_transform_to_call_collect_pending_effects(tctx, u->src[i], visited)) return false;
  }
  return true;
}

/* Pinned pm_finalize_call appends exact AFTER identities while visiting the
 * rewritten graph bottom-up. Polygrad discovers some materialized
 * CONTIGUOUS parents before pending assignment children, so recover the same
 * dependency-before-parent order by filtering the collected identities from
 * their own UOp postorder. Independent SINK sources retain source order. */
static bool poly_transform_to_call_finalize_after_order(
    PolyCtx *ctx,
    PolyTransformToCallCtx *tctx
) {
  if (!ctx || !tctx) return false;
  if (tctx->n_stores <= 1) return true;

  PolyUOp *collected = poly_sink_n(ctx, tctx->stores, tctx->n_stores);
  if (!collected) return false;
  int map_cap = tctx->n_stores * 2;
  if (map_cap < 64) map_cap = 64;
  PolyMap *effect_set = poly_map_new(map_cap);
  if (!effect_set) return false;
  for (int i = 0; i < tctx->n_stores; i++)
    poly_map_set(
        effect_set, poly_ptr_hash(tctx->stores[i]), tctx->stores[i],
        tctx->stores[i], poly_ptr_eq
    );

  PolyScratchMark scratch = poly_ctx_scratch_mark(ctx);
  int n_topo = 0;
  PolyUOp **topo = poly_toposort_scratch(ctx, collected, &n_topo);
  int n_ordered = 0;
  if (topo) {
    for (int i = 0; i < n_topo; i++) {
      PolyUOp *effect = poly_map_get(
          effect_set, poly_ptr_hash(topo[i]), topo[i], poly_ptr_eq
      );
      if (effect) tctx->stores[n_ordered++] = effect;
    }
  }
  poly_ctx_scratch_rewind(ctx, scratch);
  poly_map_destroy(effect_set);
  return topo && n_ordered == tctx->n_stores;
}

static PolyUOp *poly_transform_to_call_strip_pending_after(
    PolyCtx *ctx,
    PolyUOp *u,
    PolyMap *visited
) {
  if (!ctx || !u || !visited) return u;
  if (poly_map_get(visited, poly_ptr_hash(u), u, poly_ptr_eq)) return u;
  poly_map_set(visited, poly_ptr_hash(u), u, u, poly_ptr_eq);

  /* Once pending effects have been collected into earlier CALLs, consumers
   * should read the AFTER target. This mirrors tinygrad create_schedule:
   * AFTER carries dependency edges, while split consumer kernels see src[0].
   * Do not enter CALL bodies; they are opaque schedule items. */
  if (u->op == POLY_OP_AFTER && u->n_src >= 2 && poly_uop_has_buffer_identity(u->src[0]))
    return u->src[0];
  if (u->op == POLY_OP_CALL || u->op == POLY_OP_FUNCTION) return u;

  PolyUOp *stack_src[16];
  PolyUOp **new_src = stack_src;
  if (u->n_src > (int)(sizeof(stack_src) / sizeof(stack_src[0]))) {
    new_src = malloc((size_t)u->n_src * sizeof(*new_src));
    if (!new_src) return NULL;
  }

  bool changed = false;
  for (int i = 0; i < u->n_src; i++) {
    new_src[i] = poly_transform_to_call_strip_pending_after(ctx, u->src[i], visited);
    if (!new_src[i]) {
      if (new_src != stack_src) free(new_src);
      return NULL;
    }
    if (new_src[i] != u->src[i]) changed = true;
  }

  PolyUOp *ret = changed ? poly_rebuild_with_sources(ctx, u, new_src) : u;
  if (new_src != stack_src) free(new_src);
  return ret;
}

static PolyUOp *poly_transform_to_call_normalize_call_args(PolyCtx *ctx, PolyUOp *call) {
  if (!ctx || !call || call->op != POLY_OP_CALL || call->n_src < 1) return call;
  PolyUOp **new_src = malloc((size_t)call->n_src * sizeof(*new_src));
  if (!new_src) return NULL;
  new_src[0] = call->src[0];
  bool changed = false;
  for (int i = 1; i < call->n_src; i++) {
    PolyMap *visited = poly_map_new(64);
    if (!visited) {
      free(new_src);
      return NULL;
    }
    new_src[i] = poly_transform_to_call_strip_pending_after(ctx, call->src[i], visited);
    poly_map_destroy(visited);
    if (!new_src[i]) {
      free(new_src);
      return NULL;
    }
    if (new_src[i] != call->src[i]) changed = true;
  }
  PolyUOp *ret = changed ? poly_rebuild_with_sources(ctx, call, new_src) : call;
  free(new_src);
  return ret;
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
  PolyUOp **params = n_ordered > 0 ? calloc((size_t)n_ordered, sizeof(*params)) : NULL;
  if (n_ordered > 0 && !params) {
    if (ordered != ordered_stack) free(ordered);
    return NULL;
  }
  for (int i = 0; i < n_ordered; i++) {
    params[i] = poly_uop_param(ctx, i, ordered[i]);
    if (!params[i]) {
      free(params);
      if (ordered != ordered_stack) free(ordered);
      return NULL;
    }
  }
  PolyUOp *function = n_ordered > 0 ?
      poly_uop_substitute(ctx, sink, ordered, params, n_ordered) : sink;
  free(params);
  if (!function) {
    if (ordered != ordered_stack) free(ordered);
    return NULL;
  }

  int n_src = 1 + n_ordered;
  PolyUOp **src = calloc((size_t)n_src, sizeof(PolyUOp *));
  if (!src) {
    if (ordered != ordered_stack) free(ordered);
    return NULL;
  }

  src[0] = function;
  for (int i = 0; i < n_ordered; i++)
    src[1 + i] = ordered[i];

  PolyUOp *call = poly_uop(ctx, POLY_OP_CALL, POLY_VOID, src, n_src, poly_arg_none());
  if (ordered != ordered_stack) free(ordered);
  free(src);
  return call;
}

/* tinygrad schedule.pm_resolve_linear_call: resolve a cached kernel-level
 * LINEAR CALL's external PARAM slots against the concrete buffers from the
 * high-level SINK that was just lowered. Intermediate BUFFER arguments and
 * scalar DEFINE_VAR arguments already belong to the lowered LINEAR. */
static PolyUOp *poly_transform_to_call_resolve_linear_call(
    PolyCtx *ctx,
    PolyUOp *call,
    PolyUOp **external,
    int n_external
) {
  if (!ctx || !call || call->op != POLY_OP_CALL || call->n_src < 1 || n_external < 0 ||
      (n_external > 0 && !external))
    return NULL;

  PolyUOp **src = malloc((size_t)call->n_src * sizeof(*src));
  if (!src) return NULL;
  src[0] = call->src[0];
  bool ok = true;
  for (int i = 1; i < call->n_src; i++) {
    PolyUOp *arg = call->src[i];
    int slot = -1;
    if (arg && arg->op == POLY_OP_PARAM && arg->arg.kind == POLY_ARG_INT)
      slot = (int)arg->arg.i;
    else if (arg && arg->op == POLY_OP_PARAM && arg->arg.kind == POLY_ARG_PARAM &&
             arg->arg.param)
      slot = (int)arg->arg.param->slot;
    if (slot >= 0) {
      src[i] = (slot >= 0 && slot < n_external) ? external[slot] : NULL;
    } else {
      src[i] = arg;
    }
    if (!src[i]) {
      ok = false;
      break;
    }
  }
  PolyUOp *resolved = ok ? poly_rebuild_with_sources(ctx, call, src) : NULL;
  free(src);
  return resolved;
}

/* Pinned create_linear_with_vars first lowers the shaped-PARAM function in an
 * outer CALL, then resolves the resulting LINEAR CALL arguments against that
 * outer CALL's concrete storage arguments. Keep that boundary in one helper so
 * tensor-value and already-effectful entrypoints cannot drift apart. */
static PolyUOp *poly_transform_to_call_resolve_outer_linear(
    PolyCtx *ctx,
    PolyUOp *outer_call,
    PolyCompileMode mode
) {
  if (!ctx || !outer_call || outer_call->op != POLY_OP_CALL ||
      outer_call->n_src < 1 || !outer_call->src[0] ||
      outer_call->src[0]->op != POLY_OP_SINK)
    return NULL;

  PolyUOp *linear = poly_lower_sink_to_linear(ctx, outer_call->src[0], mode);
  if (!linear || linear->op != POLY_OP_LINEAR) return NULL;

  PolyUOp **resolved = linear->n_src > 0 ?
      calloc((size_t)linear->n_src, sizeof(*resolved)) : NULL;
  if (linear->n_src > 0 && !resolved) return NULL;
  PolyUOp **external = outer_call->n_src > 1 ? &outer_call->src[1] : NULL;
  int n_external = outer_call->n_src - 1;
  bool ok = true;
  for (int i = 0; i < linear->n_src; i++) {
    resolved[i] = poly_transform_to_call_resolve_linear_call(
        ctx, linear->src[i], external, n_external
    );
    if (!resolved[i]) {
      ok = false;
      break;
    }
  }
  PolyUOp *resolved_linear = ok ? poly_rebuild_with_sources(ctx, linear, resolved) : NULL;
  free(resolved);
  return resolved_linear;
}

/* Return the last call in producer_linear that must run before consumer.
 * transform_to_call can materialize an inner CONTIGUOUS into the STORE sink
 * and then build an opaque COPY/CALL that reads the fresh buffer. Pinned
 * tinygrad's LINEAR keeps that producer before the consumer. Polygrad lowers
 * the STORE sink separately, so merge the two ordered call streams at their
 * concrete buffer read/write boundary instead of concatenating opaque calls
 * before all STORE calls. */
static int poly_transform_to_call_producer_prefix(
    PolyCtx *ctx,
    PolyUOp *consumer,
    PolyUOp **producer_linear,
    int n_producer
) {
  if (!ctx || !consumer || consumer->op != POLY_OP_CALL || n_producer <= 0 ||
      !producer_linear)
    return -1;

  int n_consumer_args = poly_call_n_buffer_args(consumer);
  if (n_consumer_args <= 0) return -1;
  bool *consumer_outs = calloc((size_t)n_consumer_args, sizeof(*consumer_outs));
  bool *consumer_ins = calloc((size_t)n_consumer_args, sizeof(*consumer_ins));
  if (!consumer_outs || !consumer_ins ||
      poly_call_get_outs_ins(ctx, consumer, consumer_outs, consumer_ins, n_consumer_args) != 0) {
    free(consumer_outs);
    free(consumer_ins);
    return -2;
  }

  int prefix = -1;
  for (int ci = 0; ci < n_consumer_args; ci++) {
    PolyUOp *consumer_arg = poly_call_buffer_arg(consumer, ci);
    const PolyUOp *input = poly_uop_get_buffer_identity(consumer_arg);
    if (!consumer_ins[ci]) continue;
    if (!input) continue;

    for (int pi = 0; pi < n_producer; pi++) {
      PolyUOp *producer = producer_linear[pi];
      int n_producer_args = poly_call_n_buffer_args(producer);
      if (!producer || producer->op != POLY_OP_CALL || n_producer_args <= 0) continue;
      bool *producer_outs = calloc((size_t)n_producer_args, sizeof(*producer_outs));
      bool *producer_ins = calloc((size_t)n_producer_args, sizeof(*producer_ins));
      if (!producer_outs || !producer_ins ||
          poly_call_get_outs_ins(
              ctx, producer, producer_outs, producer_ins, n_producer_args
          ) != 0) {
        free(producer_outs);
        free(producer_ins);
        prefix = -2;
        goto done;
      }
      for (int po = 0; po < n_producer_args; po++) {
        PolyUOp *producer_arg = poly_call_buffer_arg(producer, po);
        const PolyUOp *output = poly_uop_get_buffer_identity(producer_arg);
        if (!producer_outs[po]) continue;
        if (output == input && pi > prefix) prefix = pi;
      }
      free(producer_outs);
      free(producer_ins);
    }
  }

done:
  free(consumer_outs);
  free(consumer_ins);
  return prefix;
}

static PolyUOp *poly_transform_to_call_cached_replacement(
    PolyTransformToCallCtx *tctx,
    PolyUOp *orig
) {
  if (!tctx || !orig) return NULL;
  for (int i = 0; i < tctx->n_cached; i++) {
    if (tctx->cached_orig[i] == orig) return tctx->cached_repl[i];
  }
  return NULL;
}

static bool poly_transform_to_call_cache_replacement(
    PolyTransformToCallCtx *tctx,
    PolyUOp *orig,
    PolyUOp *repl
) {
  if (!tctx || !orig || !repl) return false;
  for (int i = 0; i < tctx->n_cached; i++) {
    if (tctx->cached_orig[i] != orig) continue;
    tctx->cached_repl[i] = repl;
    return true;
  }
  if (tctx->n_cached >= tctx->cached_cap) {
    int new_cap =
        tctx->cached_cap ? tctx->cached_cap * 2 : POLY_TRANSFORM_TO_CALL_INITIAL_REPLACEMENTS;
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

static PolyUOp *poly_transform_to_call_materialize_view_copy(
    PolyCtx *ctx,
    PolyUOp *copy,
    PolyTransformToCallCtx *tctx
) {
  if (!ctx || !copy || !tctx || copy->op != POLY_OP_COPY || copy->n_src < 2)
    return NULL;

  PolyUOp *cached = poly_transform_to_call_cached_replacement(tctx, copy);
  if (cached) return cached;

  const PolyUOp *view = poly_uop_get_buffer_identity(copy->src[0]);
  const PolyUOp *base =
      (view && view->op == POLY_OP_BUFFER_VIEW && view->n_src >= 1)
          ? poly_uop_get_buffer_identity(view->src[0])
          : NULL;
  if (!base) return NULL;

  PolyShape shape = poly_uop_max_shape_cached(ctx, copy);
  PolyUOp *buf = poly_transform_to_call_empty_buffer_on_device(
      ctx, copy->dtype, shape, poly_uop_device(copy)
  );
  if (!buf) {
    fprintf(stderr, "poly_realize: output buffer creation failed\n");
    tctx->failed = true;
    return NULL;
  }

  /* This is tinygrad's creation-device COPY callification in Polygrad's
   * physical vocabulary. BUFFER_VIEW is the realized counterpart to SLICE:
   * install that alias first, then copy from it into the target buffer. */
  PolyUOp *view_call_src[3] = {(PolyUOp *)view, (PolyUOp *)view, (PolyUOp *)base};
  PolyUOp *copy_call_src[3] = {copy, buf, (PolyUOp *)view};
  PolyUOp *view_call =
      poly_uop(ctx, POLY_OP_CALL, POLY_VOID, view_call_src, 3, poly_arg_none());
  PolyUOp *copy_call =
      poly_uop(ctx, POLY_OP_CALL, POLY_VOID, copy_call_src, 3, poly_arg_none());
  PolyUOp *replacement = poly_transform_to_call_rebuild_view(ctx, buf, copy, NULL);
  if (!view_call || !copy_call || !replacement ||
      !poly_transform_to_call_append_call(tctx, view_call) ||
      !poly_transform_to_call_append_call(tctx, copy_call) ||
      !poly_transform_to_call_cache_replacement(tctx, copy, replacement)) {
    tctx->failed = true;
    return NULL;
  }
  return replacement;
}

static bool poly_transform_to_call_view_copy_opaque(PolyOps op) {
  switch (op) {
  case POLY_OP_FUNCTION:
  case POLY_OP_CALL:
  case POLY_OP_PROGRAM:
  case POLY_OP_LINEAR:
  case POLY_OP_SOURCE:
  case POLY_OP_BINARY:
  case POLY_OP_SINK:
  case POLY_OP_AFTER:
  case POLY_OP_STORE:
  case POLY_OP_ASSIGN:
    return true;
  default:
    return false;
  }
}

static PolyUOp *poly_transform_to_call_rewrite_view_copies(
    PolyCtx *ctx,
    PolyUOp *u,
    PolyTransformToCallCtx *tctx
) {
  if (!ctx || !u || !tctx) return NULL;

  PolyUOp *memoized = poly_map_get(
      tctx->view_copy_memo, poly_ptr_hash(u), u, poly_ptr_eq
  );
  if (memoized) return memoized;

  PolyUOp *replacement = poly_transform_to_call_materialize_view_copy(ctx, u, tctx);
  if (replacement) {
    poly_map_set(tctx->view_copy_memo, poly_ptr_hash(u), u, replacement, poly_ptr_eq);
    return replacement;
  }
  if (tctx->failed) return NULL;
  if (poly_uop_has_buffer_identity(u) || u->n_src == 0 ||
      poly_transform_to_call_view_copy_opaque(u->op)) {
    poly_map_set(tctx->view_copy_memo, poly_ptr_hash(u), u, u, poly_ptr_eq);
    return u;
  }

  PolyUOp *src_buf[16];
  PolyUOp **new_src = src_buf;
  if (u->n_src > (int)(sizeof(src_buf) / sizeof(src_buf[0]))) {
    new_src = malloc((size_t)u->n_src * sizeof(*new_src));
    if (!new_src) {
      tctx->failed = true;
      return NULL;
    }
  }

  bool changed = false;
  for (int i = 0; i < u->n_src; i++) {
    new_src[i] = poly_transform_to_call_rewrite_view_copies(ctx, u->src[i], tctx);
    if (!new_src[i]) {
      if (new_src != src_buf) free(new_src);
      return NULL;
    }
    if (new_src[i] != u->src[i]) changed = true;
  }

  PolyUOp *ret = changed ? poly_rebuild_with_sources(ctx, u, new_src) : u;
  if (new_src != src_buf) free(new_src);
  if (ret) poly_map_set(tctx->view_copy_memo, poly_ptr_hash(u), u, ret, poly_ptr_eq);
  return ret;
}

static PolyUOp *poly_transform_to_call_materialize_contiguous(
    PolyCtx *ctx,
    PolyUOp *u,
    PolyTransformToCallCtx *tctx,
    PolyUOp **out_replacement
) {
  if (out_replacement) *out_replacement = NULL;
  if (!ctx || !u || !tctx || u->op != POLY_OP_CONTIGUOUS || u->n_src != 1 ||
      poly_uop_has_buffer_identity(u->src[0]) || !out_replacement)
    return NULL;

  PolyShape shape = poly_uop_max_shape_cached(ctx, u);
  if (shape.ndim < 0) return NULL;
  if (poly_shape_numel(shape) == 0) {
    PolyUOp *replacement = poly_transform_to_call_cached_replacement(tctx, u);
    if (!replacement) {
      replacement = u->src[0];
      if (!poly_transform_to_call_cache_replacement(tctx, u, replacement)) {
        tctx->failed = true;
        return NULL;
      }
    }
    *out_replacement = replacement;
    return u->src[0];
  }

  PolyUOp *cached = poly_transform_to_call_cached_replacement(tctx, u);
  if (cached) {
    const PolyUOp *identity = poly_uop_get_buffer_identity(cached);
    PolyUOp *store = identity ? poly_store_val(ctx, cached, u->src[0]) : NULL;
    PolyUOp *after_src[2] = {cached, store};
    PolyUOp *after = (identity && store)
                         ? poly_uop(
                               ctx, POLY_OP_AFTER, cached->dtype, after_src, 2,
                               poly_arg_none())
                         : NULL;
    if (!after || !poly_transform_to_call_append_store(tctx, after)) {
      tctx->failed = true;
      return NULL;
    }
    *out_replacement = cached;
    return after;
  }

  PolyDevice out_dev = poly_uop_device(u);
  if (out_dev == POLY_DEVICE_AUTO) out_dev = poly_uop_device(u->src[0]);
  PolyUOp *buf = poly_transform_to_call_empty_buffer_on_device(ctx, u->dtype, shape, out_dev);
  PolyUOp *replacement = poly_transform_to_call_rebuild_view(ctx, buf, u, NULL);
  PolyUOp *store = replacement ? poly_store_val(ctx, replacement, u->src[0]) : NULL;
  PolyUOp *after_src[2] = {replacement, store};
  PolyUOp *after = (replacement && store)
                       ? poly_uop(
                             ctx, POLY_OP_AFTER, replacement->dtype, after_src, 2,
                             poly_arg_none())
                       : NULL;
  if (!buf || !replacement || !store || !after ||
      !poly_transform_to_call_append_store(tctx, after) ||
      !poly_transform_to_call_cache_replacement(tctx, u, replacement)) {
    tctx->failed = true;
    return NULL;
  }
  *out_replacement = replacement;
  return after;
}

/* tinygrad callify tags every CONTIGUOUS, not only the requested outer root.
 * Materialize nested boundaries into the same batched STORE sink and retain a
 * becomes-map entry so already-built live consumers read the realized value.
 * The requested outer CONTIGUOUS remains owned by the existing top-level path
 * to avoid allocating a second output buffer. */
static PolyUOp *poly_transform_to_call_rewrite_nested_contiguous(
    PolyCtx *ctx,
    PolyUOp *u,
    PolyUOp *outer,
    PolyTransformToCallCtx *tctx,
    PolyMap *memo
) {
  if (!ctx || !u || !tctx || !memo) return NULL;
  PolyUOp *memoized = poly_map_get(memo, poly_ptr_hash(u), u, poly_ptr_eq);
  if (memoized) return memoized;

  bool opaque_program =
      u->op == POLY_OP_PROGRAM || u->op == POLY_OP_LINEAR || u->op == POLY_OP_SOURCE ||
      u->op == POLY_OP_BINARY;
  /* AFTER returns a storage identity, but its STORE values are still part of
   * the shared tensor graph that pinned callify rewrites before collecting
   * effects. Treat concrete buffers/views as leaves; keep walking through the
   * immutable AFTER effect so a shared creation COPY is rewritten everywhere. */
  if ((poly_uop_has_buffer_identity(u) && u->op != POLY_OP_AFTER) ||
      u->n_src == 0 || opaque_program) {
    poly_map_set(memo, poly_ptr_hash(u), u, u, poly_ptr_eq);
    return u;
  }

  PolyUOp *src_buf[16];
  PolyUOp **new_src = src_buf;
  if (u->n_src > (int)(sizeof(src_buf) / sizeof(src_buf[0]))) {
    new_src = malloc((size_t)u->n_src * sizeof(*new_src));
    if (!new_src) {
      tctx->failed = true;
      return NULL;
    }
  }

  bool changed = false;
  int first_src = (u->op == POLY_OP_CALL || u->op == POLY_OP_FUNCTION) ? 1 : 0;
  for (int i = 0; i < first_src; i++) new_src[i] = u->src[i];
  for (int i = first_src; i < u->n_src; i++) {
    new_src[i] =
        poly_transform_to_call_rewrite_nested_contiguous(ctx, u->src[i], outer, tctx, memo);
    if (!new_src[i]) {
      if (new_src != src_buf) free(new_src);
      return NULL;
    }
    if (new_src[i] != u->src[i]) changed = true;
  }

  PolyUOp *ret = changed ? poly_rebuild_with_sources(ctx, u, new_src) : u;
  if (new_src != src_buf) free(new_src);
  if (!ret) return NULL;

  /* Pinned callify turns a contiguous movement view into a SLICE effect before
   * splitting the creation-device COPY that consumes it.  Polygrad's existing
   * physical counterpart is BUFFER_VIEW followed by COPY.  Route that case
   * through the dedicated materializer before the generic creation-COPY rule
   * can hide the view inside AFTER/STORE compute IR. */
  if (ret->op == POLY_OP_COPY) {
    PolyUOp *view_replacement =
        poly_transform_to_call_materialize_view_copy(ctx, ret, tctx);
    if (view_replacement) {
      if (ret != u &&
          !poly_transform_to_call_cache_replacement(tctx, u, view_replacement)) {
        tctx->failed = true;
        return NULL;
      }
      poly_map_set(
          memo, poly_ptr_hash(u), u, view_replacement, poly_ptr_eq
      );
      return view_replacement;
    }
    if (tctx->failed) return NULL;
  }

  /* Pinned callify tags COPYs from creation devices and rewrites the COPY
   * itself to AFTER(buffer, STORE(buffer, COPY)) before rewriting its parent.
   * Match callify.py:19-25 by reading COPY.src[0].device through movement
   * nodes. Requiring buffer identity here misses
   * COPY(RESHAPE(SHRINK(BUFFER@DISK)), DEVICE), leaking DEVICE into ordinary
   * scalar codegen. HOST/DISK are Polygrad's creation-device counterparts to
   * tinygrad's PYTHON/NPY/DISK/TINYFS set. */
  if (ret->op == POLY_OP_COPY && ret->n_src >= 2) {
    PolyDevice source_device = poly_uop_device(ret->src[0]);
    PolyDevice copy_device = poly_uop_device(ret);
    bool from_creation =
        source_device == POLY_DEVICE_HOST || source_device == POLY_DEVICE_DISK;
    /* The preserved Path-A/raw-UOp route can still carry BUFFER(UNIQUE) with
     * device=AUTO and express HOST/DISK only in ctx->buffers. Keep that
     * boundary fallback solely for an incomplete graph. Complete Path-B
     * roots take the pinned graph-device branch above, including movement-
     * wrapped DISK sources that the old identity-only rule missed. */
    if (!from_creation && source_device == POLY_DEVICE_AUTO) {
      const PolyUOp *legacy_identity =
          poly_uop_get_buffer_identity(ret->src[0]);
      PolyBuffer *legacy_storage =
          legacy_identity ? poly_buffer_get(ctx, (PolyUOp *)legacy_identity) : NULL;
      from_creation =
          legacy_storage &&
          (legacy_storage->device == POLY_DEVICE_HOST ||
           legacy_storage->device == POLY_DEVICE_DISK);
    }
    if (from_creation && copy_device != POLY_DEVICE_AUTO && copy_device != POLY_DEVICE_HOST &&
        copy_device != POLY_DEVICE_DISK) {
      PolyUOp *contiguous = poly_contiguous(ctx, ret);
      PolyUOp *replacement = NULL;
      PolyUOp *executable = contiguous ? poly_transform_to_call_materialize_contiguous(
                                             ctx, contiguous, tctx, &replacement
                                         )
                                       : NULL;
      if (!executable || !replacement ||
          !poly_transform_to_call_cache_replacement(tctx, u, replacement)) {
        tctx->failed = true;
        return NULL;
      }
      poly_map_set(memo, poly_ptr_hash(u), u, executable, poly_ptr_eq);
      return executable;
    }
  }

  /* Pinned callify.replace_store_after_with_contig turns
   * AFTER(non-storage, STORE(non-storage, value)) into a materialized
   * CONTIGUOUS(value). finalize_after then maps the original AFTER directly
   * to that final buffer. This is required for placement targets such as a
   * creation-device COPY: leaving AFTER -> COPY in the live-Tensor map keeps
   * the old imported bytes current after the assignment has executed. */
  PolyUOp *assign_target = NULL;
  PolyUOp *assign_store = NULL;
  if (ret->n_src == 2 &&
      poly_transform_to_call_after_store_assign(ret, &assign_target, &assign_store) &&
      ret->src[1] == assign_store &&
      !poly_transform_to_call_after_result_buffer(ctx, ret)) {
    PolyUOp *contiguous = poly_contiguous(ctx, assign_store->src[1]);
    PolyUOp *replacement = NULL;
    PolyUOp *executable = contiguous ? poly_transform_to_call_materialize_contiguous(
                                           ctx, contiguous, tctx, &replacement
                                       )
                                     : NULL;
    if (!executable || !replacement ||
        !poly_transform_to_call_cache_replacement(tctx, u, replacement)) {
      tctx->failed = true;
      return NULL;
    }

    poly_map_set(memo, poly_ptr_hash(u), u, executable, poly_ptr_eq);
    return executable;
  }

  /* Pinned callify.finalize_after maps each tagged AFTER to the final storage
   * left after its complete version chain is stripped. Keep the executable
   * effects in ret, while exposing that same final value to live tensors. */
  if (ret->op == POLY_OP_AFTER) {
    PolyUOp *result = poly_transform_to_call_after_result_buffer(ctx, ret);
    if (result && !poly_transform_to_call_cache_replacement(tctx, u, result)) {
      tctx->failed = true;
      return NULL;
    }
  }

  /* Pinned add_tags tags every requested multibase during the one shared
   * bottom-up rewrite. Its early transform then inserts CONTIGUOUS and lowers
   * it to AFTER(buffer, STORE(buffer, value)) before rebuilding any descendant
   * (callify.py:32-52,155-180). Do the same with AllocCtx.bases' pass-local C
   * representation; the resulting executable UOps are identical. */
  bool requested_base = tctx->requested_bases &&
                        poly_map_get(
                            tctx->requested_bases, poly_ptr_hash(u), u, poly_ptr_eq
                        );
  if (requested_base && ret->op != POLY_OP_CONTIGUOUS &&
      !poly_uop_has_buffer_identity(ret)) {
    PolyUOp *contiguous = poly_contiguous(ctx, ret);
    PolyUOp *replacement = NULL;
    PolyUOp *executable = contiguous ? poly_transform_to_call_materialize_contiguous(
                                           ctx, contiguous, tctx, &replacement
                                       )
                                     : NULL;
    if (!executable || !replacement ||
        !poly_transform_to_call_cache_replacement(tctx, u, replacement)) {
      tctx->failed = true;
      return NULL;
    }
    poly_map_set(memo, poly_ptr_hash(u), u, executable, poly_ptr_eq);
    return executable;
  }

  /* add_tags in pinned tinygrad is bottom-up: every inner CONTIGUOUS must be
   * replaced before an enclosing CONTIGUOUS store captures its source. A
   * preorder replacement leaves those inner buffers allocated but unwritten,
   * so the enclosing kernel can observe an invalid input residency. */
  if (u != outer && ret->op == POLY_OP_CONTIGUOUS && ret->n_src == 1) {
    /* Pinned pm_early_transform_tensor_graph removes an extra CONTIGUOUS on
     * AFTER when the AFTER target already has buffer identity. Preserve the
     * AFTER in the executable graph so its producer remains a dependency, but
     * map the original CONTIGUOUS to the producer's final storage for live
     * tensors. This is what keeps custom producer -> CONTIGUOUS -> COPY at two
     * calls instead of inserting an unnecessary materialization kernel. */
    PolyUOp *after = ret->src[0];
    if (after && after->op == POLY_OP_AFTER && after->n_src >= 1 &&
        poly_uop_has_buffer_identity(after->src[0])) {
      PolyUOp *result = poly_transform_to_call_after_result_buffer(ctx, after);
      if (!result || !poly_transform_to_call_cache_replacement(tctx, u, result)) {
        tctx->failed = true;
        return NULL;
      }
      poly_map_set(memo, poly_ptr_hash(u), u, after, poly_ptr_eq);
      return after;
    }

    PolyUOp *replacement = NULL;
    PolyUOp *executable = NULL;
    if (poly_uop_has_buffer_identity(ret->src[0])) {
      replacement = ret->src[0];
      executable = replacement;
    } else {
      executable =
          poly_transform_to_call_materialize_contiguous(ctx, ret, tctx, &replacement);
    }
    if (executable && replacement) {
      if (!poly_transform_to_call_cache_replacement(tctx, u, replacement)) {
        tctx->failed = true;
        return NULL;
      }
      poly_map_set(memo, poly_ptr_hash(u), u, executable, poly_ptr_eq);
      return executable;
    }
    if (tctx->failed) return NULL;
  }

  poly_map_set(memo, poly_ptr_hash(u), u, ret, poly_ptr_eq);
  return ret;
}

/* Partial C analogue of tinygrad's transform_to_call(UOp.sink(...)).
 * Current scope:
 * - batch requested unrealized value UOps into one CALL whose body is a SINK
 *   of STOREs
 * - preserve already-realized values and ASSIGN targets in out_uops
 * - return the complete physical original-to-result map to the caller */
PolyUOp *poly_transform_to_call_with_map(
    PolyCtx *ctx,
    PolyUOp **uops,
    int n,
    PolyUOp **out_uops,
    PolyUOp ***out_map_orig,
    PolyUOp ***out_map_repl,
    int *out_map_n
) {
  if (out_map_orig) *out_map_orig = NULL;
  if (out_map_repl) *out_map_repl = NULL;
  if (out_map_n) *out_map_n = 0;
  if (!ctx || !uops || !out_uops || n < 0) return NULL;
  if ((out_map_orig || out_map_repl || out_map_n) &&
      (!out_map_orig || !out_map_repl || !out_map_n))
    return NULL;
  if (n == 0) return NULL;
  for (int i = 0; i < n; i++) out_uops[i] = NULL;

  PolyUOp **stores = calloc((size_t)(n * 4), sizeof(PolyUOp *));
  if (!stores) return NULL;
  PolyTransformToCallCtx tctx = {
      .stores = stores,
      .n_stores = 0,
      .stores_cap = n * 4,
      .n_cached = 0,
      .requested_bases = poly_map_new((size_t)n * 2 + 16),
      .view_copy_memo = poly_map_new(256),
  };
  if (!tctx.requested_bases || !tctx.view_copy_memo)
    return poly_transform_to_call_fail(&tctx, out_uops, n);
  for (int i = 0; i < n; i++) {
    PolyUOp *base = poly_transform_to_call_multibase(uops[i]);
    if (!base || base->op == POLY_OP_CONST || base->op == POLY_OP_BUFFER ||
        base->op == POLY_OP_BIND || base->op == POLY_OP_AFTER ||
        poly_uop_has_buffer_identity(base))
      continue;
    poly_map_set(
        tctx.requested_bases, poly_ptr_hash(base), base, base, poly_ptr_eq
    );
  }

  PolyMap *after_visited = poly_map_new(256);
  if (!after_visited) return poly_transform_to_call_fail(&tctx, out_uops, n);
  bool after_ok = true;
  for (int i = 0; i < n && after_ok; i++)
    if (uops[i]) after_ok = poly_transform_to_call_apply_after(&tctx, uops[i], after_visited);
  poly_map_destroy(after_visited);
  if (!after_ok) return poly_transform_to_call_fail(&tctx, out_uops, n);

  /* Pinned transform_to_call rewrites one shared big_sink before
   * pm_finalize_call collects any assignment. Do the same here so a creation
   * COPY shared by state and consumer roots is materialized once throughout
   * the batched graph; collecting an original direct AFTER first would freeze
   * one stale COPY consumer before a later root sees the rewrite. */
  PolyUOp *big_sink = poly_sink_n(ctx, uops, n);
  PolyMap *shared_rewrite_memo = poly_map_new(256);
  if (!big_sink || !shared_rewrite_memo) {
    if (shared_rewrite_memo) poly_map_destroy(shared_rewrite_memo);
    return poly_transform_to_call_fail(&tctx, out_uops, n);
  }
  big_sink = poly_transform_to_call_rewrite_nested_contiguous(
      ctx, big_sink, NULL, &tctx, shared_rewrite_memo
  );
  poly_map_destroy(shared_rewrite_memo);
  if (!big_sink || big_sink->op != POLY_OP_SINK || big_sink->n_src != n)
    return poly_transform_to_call_fail(&tctx, out_uops, n);

  for (int i = 0; i < n; i++) {
    PolyUOp *u = big_sink->src[i];
    if (!u) {
      out_uops[i] = NULL;
      continue;
    }
    if (poly_uop_has_buffer_identity(u)) {
      out_uops[i] = u;
      continue;
    }

    /* tinygrad callify.transform_to_call does not tag a requested movement
     * chain whose multibase is already BUFFER/SLICE/PARAM. Tensor.realize is
     * therefore a zero-CALL operation for pure views; a later contiguous/data
     * request performs any required materialization. Keep the placed view as
     * the physical root without changing its logical provenance. */
    PolyUOp *pure_view_base = u;
    while (pure_view_base && pure_view_base->n_src >= 1 &&
           poly_opset_has(POLY_GROUP_MOVEMENT, pure_view_base->op))
      pure_view_base = pure_view_base->src[0];
    if (pure_view_base != u && poly_uop_has_buffer_identity(pure_view_base)) {
      out_uops[i] = u;
      continue;
    }

    PolyUOp *direct_after_result = poly_transform_to_call_after_result_buffer(ctx, u);
    if (direct_after_result) {
      bool found_effect = false;
      if (!poly_transform_to_call_collect_after_stores_ex(&tctx, u, &found_effect))
        return poly_transform_to_call_fail(&tctx, out_uops, n);
      /* Shared callify can have registered this exact immutable AFTER while
       * rewriting an earlier root. Pinned graph_rewrite still treats that as
       * the root's executable effect; exact-identity deduplication must not be
       * mistaken for absence and followed by a second raw STORE append. */
      if (found_effect) {
        out_uops[i] = direct_after_result;
        continue;
      }
    }

    /* Tensor.assign follows tinygrad and builds AFTER(target, STORE(target,
     * value)). An already-realized storage target can take the direct in-place
     * path. Leave every other target in the complete graph: pinned callify
     * rewrites nested targets bottom-up before deciding whether AFTER+STORE
     * needs CONTIGUOUS materialization. Dropping the STORE here would prevent
     * a rewritten child AFTER from becoming the next assignment's target. */
    PolyUOp *assign_target = NULL;
    PolyUOp *assign_store = NULL;
    if (poly_transform_to_call_after_store_assign(u, &assign_target, &assign_store)) {
      if (poly_uop_has_buffer_identity(assign_target)) {
        if (!poly_transform_to_call_append_store(&tctx, assign_store))
          return poly_transform_to_call_fail(&tctx, out_uops, n);
        out_uops[i] = assign_target;
        continue;
      }
    }

    /* ASSIGN already writes to its target buffer, so keep it as the batched
     * store boundary and report the target buffer as the realized result. */
    if (u->op == POLY_OP_ASSIGN && u->n_src >= 1) {
      if (!poly_transform_to_call_append_store(&tctx, u))
        return poly_transform_to_call_fail(&tctx, out_uops, n);
      out_uops[i] = u->src[0];
      continue;
    }

    PolyTransformViewStack target_views = {0};
    PolyUOp *target_root = poly_transform_to_call_root(u, &target_views);
    if (!target_root) {
      poly_transform_view_stack_free(&target_views);
      return poly_transform_to_call_fail(&tctx, out_uops, n);
    }

    /* Leave the existing top-level view COPY path below intact. For a COPY
     * nested in a consumer value graph, tinygrad's add_tags/finalize_call
     * materializes the creation-device dependency before the consumer kernel. */
    const PolyUOp *target_view =
        (target_root->op == POLY_OP_COPY && target_root->n_src >= 2)
            ? poly_uop_get_buffer_identity(target_root->src[0])
            : NULL;
    bool top_level_view_copy = target_view && target_view->op == POLY_OP_BUFFER_VIEW;
    if (!top_level_view_copy) {
      poly_transform_view_stack_free(&target_views);
      u = poly_transform_to_call_rewrite_view_copies(ctx, u, &tctx);
      if (!u) return poly_transform_to_call_fail(&tctx, out_uops, n);
      target_views = (PolyTransformViewStack){0};
      target_root = poly_transform_to_call_root(u, &target_views);
      if (!target_root) {
        poly_transform_view_stack_free(&target_views);
        return poly_transform_to_call_fail(&tctx, out_uops, n);
      }
    }

    PolyUOp *outer_contiguous =
        target_root->op == POLY_OP_CONTIGUOUS ? target_root : NULL;
    if (outer_contiguous) {
      PolyUOp *cached =
          poly_transform_to_call_cached_replacement(&tctx, outer_contiguous);
      const PolyUOp *cached_identity = poly_uop_get_buffer_identity(cached);
      if (cached_identity) {
        /* Another target already materialized this exact CONTIGUOUS. Rebuild
         * only this target's outer movement views from the shared realized
         * identity, making batched callification independent of target order. */
        out_uops[i] = poly_transform_to_call_rebuild_view(
            ctx, (PolyUOp *)cached_identity, outer_contiguous, &target_views
        );
        poly_transform_view_stack_free(&target_views);
        if (!out_uops[i]) return poly_transform_to_call_fail(&tctx, out_uops, n);
        continue;
      }
    }
    /* Pinned callify runs its early tensor-graph rewrite once on the shared
     * big_sink. The shared pass above has already rewritten every nested
     * CONTIGUOUS and creation-device COPY. Re-entering those replacement
     * AFTER/STORE graphs here would callify the owned COPY a second time and
     * duplicate each already-materialized dependency. */
    if (poly_uop_has_buffer_identity(u)) {
      out_uops[i] = u;
      poly_transform_view_stack_free(&target_views);
      continue;
    }
    poly_transform_view_stack_free(&target_views);

    PolyMap *pending_visited = poly_map_new(64);
    if (!pending_visited) return poly_transform_to_call_fail(&tctx, out_uops, n);
    bool pending_ok = poly_transform_to_call_collect_pending_effects(&tctx, u, pending_visited);
    poly_map_destroy(pending_visited);
    if (!pending_ok) return poly_transform_to_call_fail(&tctx, out_uops, n);

    PolyUOp *after_result = poly_transform_to_call_after_result_buffer(ctx, u);
    if (after_result) {
      out_uops[i] = after_result;
      continue;
    }

    /* Pinned callify.py:222-241 keeps nested AFTER values in ctx.assigns;
     * rangeify.py:497-514 later rewrites them to buffers while retaining the
     * exact producer version as the CALL dependency. Keep the same graph here:
     * stripping now would make COPY consumers read storage before its creator. */

    PolyShape output_shape = poly_uop_max_shape_cached(ctx, u);
    int64_t output_numel =
        (output_shape.ndim >= 0) ? poly_shape_numel(output_shape) : -1;
    if (output_numel == 0) {
      /* tinygrad callify removes size-zero CONTIGUOUS work and retargets the
       * Tensor to an unallocated BUFFER. Keep the distinct placed identity and
       * any producer effects collected above, but do not invent a residency:
       * there is no CALL whose execution could allocate or write one. */
      PolyDevice out_dev = poly_uop_device(u);
      PolyUOp *buf = poly_transform_to_call_empty_buffer_on_device(
          ctx, u->dtype, output_shape, out_dev
      );
      if (!buf) return poly_transform_to_call_fail(&tctx, out_uops, n);
      PolyUOp *result =
          (output_shape.ndim == 1 && output_shape.dims[0] == 0)
              ? buf
              : poly_reshape(ctx, buf, output_shape.dims, output_shape.ndim);
      if (!result) return poly_transform_to_call_fail(&tctx, out_uops, n);
      out_uops[i] = result;
      if (outer_contiguous &&
          !poly_transform_to_call_cache_replacement(&tctx, outer_contiguous, result)) {
        return poly_transform_to_call_fail(&tctx, out_uops, n);
      }
      continue;
    }

    PolyTransformViewStack views = {0};
    PolyUOp *materialized = poly_transform_to_call_root(u, &views);
    if (!materialized) {
      poly_transform_view_stack_free(&views);
      return poly_transform_to_call_fail(&tctx, out_uops, n);
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
        poly_transform_to_call_empty_buffer_on_device(ctx, u->dtype, root_shape, out_dev);
    if (!buf) {
      fprintf(stderr, "poly_realize: output buffer creation failed\n");
      poly_transform_view_stack_free(&views);
      return poly_transform_to_call_fail(&tctx, out_uops, n);
    }

    /* A realized BUFFER_VIEW is Polygrad's physical counterpart to tinygrad's
     * SLICE call. When a cross-device COPY consumes it, keep both ordered CALL
     * boundaries: first install the alias from its base storage, then copy the
     * selected range. Sending BUFFER_VIEW through compute codegen would embed
     * a storage identity as a value expression and lose the slice offset. */
    PolyUOp *copy_source =
        (materialized->op == POLY_OP_COPY && materialized->n_src >= 2)
            ? materialized->src[0]
            : NULL;
    while (copy_source && copy_source->op == POLY_OP_CONTIGUOUS && copy_source->n_src >= 1)
      copy_source = copy_source->src[0];
    const PolyUOp *copy_identity = poly_uop_get_buffer_identity(copy_source);
    if (!copy_identity && copy_source) {
      PolyUOp *copy_result = poly_transform_to_call_after_result_buffer(ctx, copy_source);
      copy_identity = poly_uop_get_buffer_identity(copy_result);
    }
    PolyUOp *copy_body = materialized;
    if (copy_identity && copy_source && !poly_uop_has_buffer_identity(copy_source)) {
      /* Pinned rangeify.py:573-590 keeps AFTER in lctx.map.values() but
       * rewrites it out of the executable COPY body. Preserve the exact source
       * below as the CALL dependency and normalize only this body boundary. */
      PolyMap *strip_visited = poly_map_new(64);
      if (!strip_visited) {
        poly_transform_view_stack_free(&views);
        return poly_transform_to_call_fail(&tctx, out_uops, n);
      }
      copy_body = poly_transform_to_call_strip_pending_after(
          ctx, materialized, strip_visited
      );
      poly_map_destroy(strip_visited);
      if (!copy_body) {
        poly_transform_view_stack_free(&views);
        return poly_transform_to_call_fail(&tctx, out_uops, n);
      }
    }
    const PolyUOp *view_base =
        (copy_identity && copy_identity->op == POLY_OP_BUFFER_VIEW && copy_identity->n_src >= 1)
            ? poly_uop_get_buffer_identity(copy_identity->src[0])
            : NULL;
    if (view_base) {
      PolyUOp *view_call_src[3] = {
          (PolyUOp *)copy_identity, (PolyUOp *)copy_identity, (PolyUOp *)view_base
      };
      PolyUOp *copy_call_src[3] = {copy_body, buf, copy_source};
      PolyUOp *view_call =
          poly_uop(ctx, POLY_OP_CALL, POLY_VOID, view_call_src, 3, poly_arg_none());
      PolyUOp *copy_call =
          poly_uop(ctx, POLY_OP_CALL, POLY_VOID, copy_call_src, 3, poly_arg_none());
      if (!view_call || !copy_call ||
          !poly_transform_to_call_append_call(&tctx, view_call) ||
          !poly_transform_to_call_append_call(&tctx, copy_call)) {
        poly_transform_view_stack_free(&views);
        return poly_transform_to_call_fail(&tctx, out_uops, n);
      }
      out_uops[i] = poly_transform_to_call_rebuild_view(ctx, buf, materialized, &views);
      if (!out_uops[i] ||
          (outer_contiguous && !poly_transform_to_call_cache_replacement(
                                   &tctx, outer_contiguous, out_uops[i]
                               ))) {
        poly_transform_view_stack_free(&views);
        return poly_transform_to_call_fail(&tctx, out_uops, n);
      }
      poly_transform_view_stack_free(&views);
      continue;
    }

    /* COPY is already a complete schedule item. Call it directly with its
     * destination and source identities; wrapping it in STORE would make
     * rangeify emit a second redundant COPY. */
    if (copy_identity) {
      PolyUOp *copy_call_src[3] = {copy_body, buf, copy_source};
      PolyUOp *copy_call =
          poly_uop(ctx, POLY_OP_CALL, POLY_VOID, copy_call_src, 3, poly_arg_none());
      if (!copy_call || !poly_transform_to_call_append_call(&tctx, copy_call)) {
        poly_transform_view_stack_free(&views);
        return poly_transform_to_call_fail(&tctx, out_uops, n);
      }
      out_uops[i] = poly_transform_to_call_rebuild_view(ctx, buf, materialized, &views);
      if (!out_uops[i] ||
          (outer_contiguous && !poly_transform_to_call_cache_replacement(
                                   &tctx, outer_contiguous, out_uops[i]
                               ))) {
        poly_transform_view_stack_free(&views);
        return poly_transform_to_call_fail(&tctx, out_uops, n);
      }
      poly_transform_view_stack_free(&views);
      continue;
    }

    /* Pinned callify tags every requested base, adds CONTIGUOUS, then lowers
     * it to AFTER(buffer, STORE(buffer, source)). The caller-visible physical
     * result remains the stripped buffer below. */
    PolyUOp *output_store = poly_store_val(ctx, buf, materialized);
    PolyUOp *after_src[2] = {buf, output_store};
    PolyUOp *output_effect = output_store ?
        poly_uop(ctx, POLY_OP_AFTER, buf->dtype, after_src, 2, poly_arg_none()) : NULL;
    if (!output_effect || !poly_transform_to_call_append_store(&tctx, output_effect)) {
      poly_transform_view_stack_free(&views);
      return poly_transform_to_call_fail(&tctx, out_uops, n);
    }
    out_uops[i] = poly_transform_to_call_rebuild_view(ctx, buf, materialized, &views);
    if (!out_uops[i] ||
        (outer_contiguous &&
         !poly_transform_to_call_cache_replacement(&tctx, outer_contiguous, out_uops[i]))) {
      poly_transform_view_stack_free(&views);
      return poly_transform_to_call_fail(&tctx, out_uops, n);
    }
    poly_transform_view_stack_free(&views);
  }

  /* Pinned add_tags tags each requested multibase and finalize_after returns a
   * buffer_map entry for it. Preserve the same complete physical map even when
   * the requested root is an ordinary ADD/MUL rather than an explicitly cached
   * CONTIGUOUS/AFTER. Without this entry a live dependent recomputes the just-
   * realized value from its inputs instead of consuming the final BUFFER. */
  for (int i = 0; i < n; i++) {
    if (uops[i] && out_uops[i] && uops[i] != out_uops[i] &&
        !poly_transform_to_call_cache_replacement(&tctx, uops[i], out_uops[i])) {
      return poly_transform_to_call_fail(&tctx, out_uops, n);
    }
  }

  PolyUOp *store_call = NULL;
  if (tctx.n_stores > 0) {
    if (!poly_transform_to_call_finalize_after_order(ctx, &tctx))
      return poly_transform_to_call_fail(&tctx, out_uops, n);
    PolyUOp *sink_body = poly_sink_n(ctx, tctx.stores, tctx.n_stores);
    store_call = poly_transform_to_call_wrap_call(ctx, sink_body);
    if (!store_call) return poly_transform_to_call_fail(&tctx, out_uops, n);
  }

  if (tctx.n_calls == 0) {
    PolyUOp *result = store_call ? store_call :
        poly_uop(ctx, POLY_OP_LINEAR, POLY_VOID, NULL, 0, poly_arg_none());
    if (!result) return poly_transform_to_call_fail(&tctx, out_uops, n);
    poly_transform_to_call_take_replacements(
        &tctx, out_map_orig, out_map_repl, out_map_n
    );
    poly_transform_to_call_ctx_free(&tctx);
    return result;
  }

  poly_transform_to_call_order_calls(&tctx);
  for (int i = 0; i < tctx.n_calls; i++) {
    tctx.calls[i] = poly_transform_to_call_normalize_call_args(ctx, tctx.calls[i]);
    if (!tctx.calls[i]) return poly_transform_to_call_fail(&tctx, out_uops, n);
  }

  PolyUOp *store_linear = NULL;
  PolyUOp **store_external = NULL;
  int n_store_external = 0;
  bool store_external_owned = false;
  if (store_call) {
    PolyUOp *sink_body = store_call->src[0];
    n_store_external = store_call->n_src - 1;
    store_external = n_store_external > 0 ? &store_call->src[1] : NULL;
    store_linear = poly_lower_sink_to_linear(ctx, sink_body, POLY_MODE_CALL);
    if (!store_linear || store_linear->op != POLY_OP_LINEAR) {
      if (store_external_owned) free(store_external);
      return poly_transform_to_call_fail(&tctx, out_uops, n);
    }
  }

  int n_store_calls = store_linear ? store_linear->n_src : 0;
  PolyUOp **resolved_store_calls =
      n_store_calls > 0 ? calloc((size_t)n_store_calls, sizeof(*resolved_store_calls)) : NULL;
  if (n_store_calls > 0 && !resolved_store_calls) {
    if (store_external_owned) free(store_external);
    return poly_transform_to_call_fail(&tctx, out_uops, n);
  }
  for (int i = 0; i < n_store_calls; i++) {
    resolved_store_calls[i] = poly_transform_to_call_resolve_linear_call(
        ctx, store_linear->src[i], store_external, n_store_external
    );
    if (!resolved_store_calls[i]) {
      free(resolved_store_calls);
      if (store_external_owned) free(store_external);
      return poly_transform_to_call_fail(&tctx, out_uops, n);
    }
  }

  int n_linear = tctx.n_calls + n_store_calls;
  PolyUOp **linear_src = calloc((size_t)n_linear, sizeof(*linear_src));
  if (!linear_src) {
    free(resolved_store_calls);
    if (store_external_owned) free(store_external);
    return poly_transform_to_call_fail(&tctx, out_uops, n);
  }
  int n_emitted = 0;
  int store_cursor = 0;
  for (int i = 0; i < tctx.n_calls; i++) {
    int prefix = poly_transform_to_call_producer_prefix(
        ctx, tctx.calls[i], resolved_store_calls, n_store_calls
    );
    if (prefix == -2) {
      free(linear_src);
      free(resolved_store_calls);
      if (store_external_owned) free(store_external);
      return poly_transform_to_call_fail(&tctx, out_uops, n);
    }
    while (store_cursor <= prefix) linear_src[n_emitted++] = resolved_store_calls[store_cursor++];
    linear_src[n_emitted++] = tctx.calls[i];
  }
  while (store_cursor < n_store_calls)
    linear_src[n_emitted++] = resolved_store_calls[store_cursor++];
  if (n_emitted != n_linear) {
    free(linear_src);
    free(resolved_store_calls);
    if (store_external_owned) free(store_external);
    return poly_transform_to_call_fail(&tctx, out_uops, n);
  }
  PolyUOp *linear = poly_uop(ctx, POLY_OP_LINEAR, POLY_VOID, linear_src, n_linear, poly_arg_none());
  free(linear_src);
  free(resolved_store_calls);
  if (store_external_owned) free(store_external);
  if (!linear) return poly_transform_to_call_fail(&tctx, out_uops, n);
  poly_transform_to_call_take_replacements(
      &tctx, out_map_orig, out_map_repl, out_map_n
  );
  poly_transform_to_call_ctx_free(&tctx);
  return linear;
}

PolyUOp *poly_transform_to_call(PolyCtx *ctx, PolyUOp **uops, int n, PolyUOp **out_uops) {
  return poly_transform_to_call_with_map(ctx, uops, n, out_uops, NULL, NULL, NULL);
}

PolySchedule *poly_schedule_effect_sink(PolyCtx *ctx, PolyUOp *sink) {
  if (!ctx || !sink || sink->op != POLY_OP_SINK) {
    fprintf(stderr, "poly_realize: expected effect SINK\n");
    return NULL;
  }

  /* Imported/Instance graphs already own their output/effect storage, so they
   * skip tensor output allocation. They do not skip tinygrad's input-buffer
   * normalization: rangeify must see a shaped-PARAM function body, with the
   * concrete BUFFER/BUFFER_VIEW identities retained as outer CALL arguments. */
  bool timing = poly_debug_at_least(2);
  double t0 = timing ? poly_now_ms() : 0.0;
  if (timing) {
    fprintf(
        stderr, "[polygrad:schedule_effect] begin sink=%p n_src=%d\n", (void *)sink, sink->n_src
    );
    fflush(stderr);
  }
  PolyUOp *outer_call = poly_transform_to_call_wrap_call(ctx, sink);
  PolyUOp *linear = outer_call ?
      poly_transform_to_call_resolve_outer_linear(ctx, outer_call, POLY_MODE_CALL) : NULL;
  PolySchedule *sched = linear ?
      poly_create_schedule_from_linear_with_vars(ctx, linear, outer_call, POLY_MODE_CALL) : NULL;
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

/* C-side scheduling half of tinygrad create_linear_with_vars after
 * transform_to_call and live-Tensor map publication have already completed. */
static PolySchedule *poly_schedule_callified_with_vars(PolyCtx *ctx, PolyUOp *big_call) {
  if (!ctx || !big_call) return NULL;
  PolySchedule *sched = NULL;
  if (big_call->op == POLY_OP_LINEAR) {
    sched = poly_create_schedule_from_linear_with_vars(
        ctx, big_call, big_call, POLY_MODE_CALL
    );
  } else if (big_call->op == POLY_OP_CALL) {
    if (big_call->n_src < 1 || !big_call->src[0] || big_call->src[0]->op != POLY_OP_SINK) {
      fprintf(stderr, "poly_realize: transform_to_call returned malformed CALL\n");
      return NULL;
    }
    PolyUOp *resolved_linear =
        poly_transform_to_call_resolve_outer_linear(ctx, big_call, POLY_MODE_CALL);
    if (!resolved_linear) return NULL;
    sched = poly_create_schedule_from_linear_with_vars(
        ctx, resolved_linear, big_call, POLY_MODE_CALL
    );
  } else {
    sched = poly_schedule_effect_sink(ctx, big_call);
  }
  return sched;
}

/* Raw physical-UOp analogue of Tensor.schedule_with_vars. Tensor placement
 * adaptation belongs to poly_realize_tensors_impl, not this entrypoint. */
PolySchedule *poly_schedule_with_vars(PolyCtx *ctx, PolyUOp **uops, int n, PolyUOp **out_uops) {
  if (!ctx || !uops || !out_uops || n < 0) return NULL;
  if (n == 0) return NULL;

  PolyUOp *big_call = poly_transform_to_call(ctx, uops, n, out_uops);
  if (!big_call) goto fail;
  PolySchedule *sched = poly_schedule_callified_with_vars(ctx, big_call);
  if (sched) return sched;

fail:
  for (int i = 0; i < n; i++) out_uops[i] = NULL;
  return NULL;
}

static PolyDevice poly_realize_tensor_requested_device(PolyCtx *ctx, PolyTensor *tensor) {
  PolyDevice device = tensor ? tensor->device : POLY_DEVICE_AUTO;
  if (device == POLY_DEVICE_AUTO) device = poly_ctx_get_preferred_device(ctx);
  if (device == POLY_DEVICE_AUTO) device = poly_device_default();
  return device;
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

/* Shared schedule ownership for raw-UOp and Tensor realization. Pinned
 * create_linear_with_vars records the real LINEAR during capture and returns
 * an empty LINEAR to the realizing call, so captured work is not run here. */
static int poly_realize_schedule(PolyCtx *ctx, PolySchedule *sched) {
  if (!ctx || !sched) return -1;
  PolyJit *cap = ctx->active_jit_capture;
  if (poly_jit_is_capturing(cap)) {
    if (poly_jit_record_schedule(cap, sched) != 0) {
      poly_schedule_free(sched);
      return -1;
    }
    return 0;
  }
  int ret = poly_run_schedule(ctx, sched, NULL, 0);
  poly_schedule_free(sched);
  return ret;
}

static bool poly_tensor_root_has_unplaced_buffer(PolyCtx *ctx, PolyUOp *root) {
  if (!ctx || !root) return true;
  PolyScratchMark scratch = poly_ctx_scratch_mark(ctx);
  int n = 0;
  /* CALL/FUNCTION bodies contain placeholders, not Tensor storage. Their
   * caller-visible arguments remain part of the traversal. */
  PolyUOp **topo =
      poly_toposort_ex_user_scratch(ctx, root, &n, NULL, NULL, false);
  if (!topo) {
    poly_ctx_scratch_rewind(ctx, scratch);
    return true;
  }
  bool unplaced = false;
  for (int i = 0; i < n; i++) {
    if (topo[i] && topo[i]->op == POLY_OP_BUFFER &&
        poly_uop_device(topo[i]) == POLY_DEVICE_AUTO) {
      unplaced = true;
      break;
    }
  }
  poly_ctx_scratch_rewind(ctx, scratch);
  return unplaced;
}

int poly_realize_uops(PolyCtx *ctx, PolyUOp **uops, int n, PolyUOp **out_uops) {
  if (!ctx || !uops || !out_uops || n < 0) return -1;
  if (n == 0) return 0;
  PolySchedule *sched = poly_schedule_with_vars(ctx, uops, n, out_uops);
  return sched ? poly_realize_schedule(ctx, sched) : -1;
}

static int poly_realize_tensors_impl(
    PolyCtx *ctx,
    PolyTensor **inputs,
    int n,
    PolyTensor **outputs
) {
  if (!ctx || !inputs || !outputs || n < 0) return -1;
  if (n == 0) return 0;
  PolyUOp **physical_roots = calloc((size_t)n, sizeof(PolyUOp *));
  PolyTensor **pending_tensors = calloc((size_t)n, sizeof(PolyTensor *));
  PolyUOp **pending_roots = calloc((size_t)n, sizeof(PolyUOp *));
  PolyUOp **pending_out = calloc((size_t)n, sizeof(PolyUOp *));
  int *pending_indices = calloc((size_t)n, sizeof(int));
  PolyMap *placement_memo[POLY_DEVICE_DISK + 1] = {0};
  PolyUOp **map_orig = NULL;
  PolyUOp **map_repl = NULL;
  int map_n = 0;
  PolySchedule *sched = NULL;
  int rc = -1;
  int n_pending = 0;
  bool used_placement = false;
  if (!physical_roots || !pending_tensors || !pending_roots || !pending_out ||
      !pending_indices)
    goto cleanup;

  for (int i = 0; i < n; i++) {
    outputs[i] = NULL;
    if (!inputs[i]) goto cleanup;
    if (!inputs[i]->uop_physical ||
        poly_tensor_root_has_unplaced_buffer(ctx, inputs[i]->uop_physical))
      used_placement = true;
  }

  /* Pinned Tensor.realize consumes its sole deviceful Tensor.uop directly
   * (tensor.py:202-218); it has no logical->physical admission fallback.
   * Keep Path A available while Path B is experimental, but make every
   * remaining admission observable and fail-loud under the migration gate. */
  if (used_placement && poly_getenv_flag("POLY_PATH_B_REQUIRE_PHYSICAL")) {
    for (int i = 0; i < n; i++) {
      PolyUOp *physical = inputs[i]->uop_physical;
      bool missing = physical == NULL;
      bool unplaced = !missing && poly_tensor_root_has_unplaced_buffer(ctx, physical);
      if (!missing && !unplaced) continue;
      fprintf(
          stderr,
          "PATH_B_ADMISSION_FAIL target=%d reason=%s logical_op=%s "
          "physical_op=%s role=%d device=%d\n",
          i, missing ? "missing_physical" : "unplaced_buffer",
          inputs[i]->uop_logical ? poly_op_name(inputs[i]->uop_logical->op) : "NULL",
          physical ? poly_op_name(physical->op) : "NULL", (int)inputs[i]->role,
          (int)inputs[i]->device
      );
    }
    goto cleanup;
  }

  /* Pinned Tensor.realize filters its already-deviceful Tensor.uop directly
   * (tensor.py:214-219), then applies transform_to_call's exact becomes-map.
   * Path B stores a complete graph in uop_physical at construction. During
   * migration, a legacy non-NULL graph can still contain device-free BUFFERs;
   * keep the preserved Path-A aggregate placer for that entire batch and
   * never mix placement and direct roots within one realization. */
  if (used_placement) {
    if (poly_tensor_physicalize_many(
            ctx, inputs, n, physical_roots, placement_memo
        ) != 0)
      goto cleanup;
  } else {
    for (int i = 0; i < n; i++)
      physical_roots[i] = inputs[i]->uop_physical;
  }

  for (int i = 0; i < n; i++) {
    PolyUOp *physical = physical_roots[i];
    /* Pinned Tensor.realize skips roots whose UOp.device is None
     * (tensor.py:214-219). Path B represents that state as AUTO; wrapper
     * device metadata must not force a pure CONST graph through callify. */
    if (!used_placement && poly_uop_device(physical) == POLY_DEVICE_AUTO) {
      outputs[i] = inputs[i];
      continue;
    }
    /* Pinned callify.py:60-95 keeps only a provable contiguous movement view
     * over already-valid storage as a zero-CALL Buffer.view. Do not use the
     * recursive UOp.buffer accessor as the proof: it also walks through
     * AFTER, whose STORE effects must first run through callify. */
    if (!used_placement && physical->op == POLY_OP_CONTIGUOUS &&
        physical->n_src == 1) {
      PolyUOp *view_identity = NULL;
      PolyShape view_shape = {.ndim = -1};
      int64_t view_numel = -1;
      size_t view_byte_offset = 0;
      if (poly_uop_contiguous_view_info(
              ctx, physical->src[0], &view_identity, &view_shape,
              &view_numel, &view_byte_offset
          ) &&
          poly_uop_buffer(ctx, physical)) {
        outputs[i] = inputs[i];
        continue;
      }
    }
    const PolyUOp *identity = poly_uop_get_buffer_identity(physical);
    if (identity) {
      /* tinygrad uop/ops.py:825-829 passes BUFFER/SLICE/PARAM identities
       * through without callify. Reconcile only an existing valid Polygrad
       * residency: creating a valid row for an absent mapped BUFFER would
       * falsely execute capture-time state. */
      PolyBuffer *storage = poly_buffer_get(ctx, (PolyUOp *)identity);
      PolyDevice requested = poly_realize_tensor_requested_device(ctx, inputs[i]);
      if (storage && (storage->valid || (storage->src && storage->src->valid)) &&
          poly_buffer_ensure_device_current(ctx, (PolyUOp *)identity, requested) != 0)
        goto cleanup;
      if (poly_tensor_set_physical(
              ctx, inputs[i], physical, inputs[i]->role, inputs[i]->device
          ) != 0)
        goto cleanup;
      outputs[i] = inputs[i];
      continue;
    }
    pending_tensors[n_pending] = inputs[i];
    pending_roots[n_pending] = physical;
    pending_indices[n_pending++] = i;
  }

  if (n_pending == 0) {
    rc = 0;
    goto cleanup;
  }

  /* Pinned Tensor.linear_with_vars publishes transform_to_call's complete map
   * before create_linear_with_vars. The exact placement memo exists only to
   * adapt that physical map back onto Polygrad's live uop_physical roots. */
  PolyUOp *big_call = poly_transform_to_call_with_map(
      ctx, pending_roots, n_pending, pending_out, &map_orig, &map_repl, &map_n
  );
  if (!big_call) goto cleanup;
  if (map_n > 0 && poly_tensor_apply_realize_map(
                       ctx, map_orig, map_repl, map_n, POLY_DEVICE_AUTO,
                       used_placement ? placement_memo : NULL
                   ) != 0)
    goto cleanup;

  for (int pending = 0; pending < n_pending; pending++) {
    PolyUOp *out = pending_out[pending];
    PolyUOp *current = poly_tensor_uop(pending_tensors[pending]);
    if (!out) goto cleanup;
    /* A pure movement root can pass through callify unchanged, so it has no
     * becomes-map row. tinygrad already stores that physical root in
     * Tensor.uop; publish the exact requested physical counterpart here. A
     * changed output must still have reached the wrapper through the complete
     * becomes-map above, preserving the live-dependent correctness check. */
    if (current != out && out != pending_roots[pending]) goto cleanup;
    if (poly_tensor_uop_physical(pending_tensors[pending]) != out &&
        poly_tensor_set_physical(
            ctx, pending_tensors[pending], out,
            pending_tensors[pending]->role, pending_tensors[pending]->device
        ) != 0)
      goto cleanup;
  }

  sched = poly_schedule_callified_with_vars(ctx, big_call);
  if (!sched) goto cleanup;
  rc = poly_realize_schedule(ctx, sched);
  sched = NULL; /* poly_realize_schedule consumes schedule ownership */
  if (rc != 0) goto cleanup;
  for (int pending = 0; pending < n_pending; pending++)
    outputs[pending_indices[pending]] = pending_tensors[pending];

cleanup:
  poly_schedule_free(sched);
  free(map_repl);
  free(map_orig);
  poly_tensor_physicalize_memo_destroy(placement_memo);
  free(pending_indices);
  free(pending_out);
  free(pending_roots);
  free(pending_tensors);
  free(physical_roots);
  return rc;
}

int poly_realize_tensors_ex(
    PolyCtx *ctx,
    PolyTensor **inputs,
    int n,
    PolyTensor **outputs,
    bool update_stats
) {
  if (!ctx) return -1;
  if (!update_stats) ctx->stats_suppression_depth++;
  int rc = poly_realize_tensors_impl(ctx, inputs, n, outputs);
  if (!update_stats) ctx->stats_suppression_depth--;
  return rc;
}

int poly_realize_tensors(PolyCtx *ctx, PolyTensor **inputs, int n, PolyTensor **outputs) {
  return poly_realize_tensors_ex(ctx, inputs, n, outputs, true);
}
