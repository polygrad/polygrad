/* realize.c -- graph-driven realize: reads buffers from ctx->buffers
 * side table (managed via device.h), no external bindings needed. */

#include "engine/realize.h"
#include "device.h"
#include "ctx.h"
#include "engine/jit.h"
#include "engine/schedule.h"
#include "frontend_internal.h"
#include "uop/upat.h"
#include "schedule/rangeify.h"
#include "schedule/schedule.h"
#include "tensor.h"
#include "utils.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Tinygrad's callify state is list/dict-backed. Keep the same semantics in C:
 * the initial sizes match the old fixed caps, but both grow when needed. */
#define POLY_TRANSFORM_TO_CALL_INITIAL_VIEWS 16
#define POLY_TRANSFORM_TO_CALL_INITIAL_REPLACEMENTS 64
#define POLY_CALLIFY_TAG_MARKER INT32_MIN

static bool poly_transform_to_call_view_op(PolyOps op) {
  /* Pinned UOp.base and callify's buffer_map preserve the complete
   * GroupOp.Movement set (uop/ops.py:758-762). POLY_GROUP_MOVEMENT is the
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
  PolyUOp **cached_orig;
  PolyUOp **cached_repl;
  int n_cached;
  int cached_cap;
  PolyUOp **public_orig;
  PolyUOp **public_repl;
  int n_public;
  int public_cap;
  PolyUOp **tag_orig;
  int n_tag_orig;
  int tag_orig_cap;
  PolyMap *requested_bases;
  PolyMap *original_uops;
  PolyMap *device_memo;
  PolyMap *axis_memo;
  bool publication_phase;
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

/* Pinned tensor.py:transform_to_call excludes virtual and ALU values from
 * requested storage. Reuse that predicate before C's fallback allocation too. */
static bool poly_callify_can_store(PolyCtx *ctx, PolyTransformToCallCtx *tctx, PolyUOp *u) {
  PolyAddrSpace addrspace;
  return u && !poly_dtype_is_weak(u->dtype) &&
         poly_uop_device_uop_cached(ctx, u, tctx->device_memo) &&
         !(poly_uop_addrspace(u, &addrspace) && addrspace == POLY_ADDR_ALU);
}

static PolyUOp *poly_transform_to_call_after_result_root(
    PolyUOp *u,
    PolyTransformViewStack *views
) {
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
    PolyCtx *ctx,
    PolyUOp *u,
    PolyShape shape,
    int dim
) {
  if (!ctx || !u || dim < 0 || dim >= shape.ndim) return NULL;
  PolyUOp *value = poly_uop_shape_dim(ctx, u, dim);
  if (!value) value = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(shape.dims[dim]));
  if (value && !poly_dtype_eq(value->dtype, POLY_WEAKINT))
    value = poly_uop1(ctx, POLY_OP_CAST, POLY_WEAKINT, value, poly_arg_none());
  return value ? poly_graph_rewrite(ctx, value, poly_symbolic()) : NULL;
}

static bool poly_transform_to_call_same_exact_shape(
    PolyCtx *ctx,
    PolyUOp *a,
    PolyShape a_shape,
    PolyUOp *b,
    PolyShape b_shape
) {
  if (!ctx || !a || !b || a_shape.ndim != b_shape.ndim || a_shape.ndim < 0) return false;
  for (int i = 0; i < a_shape.ndim; i++) {
    PolyUOp *a_dim = poly_transform_to_call_canonical_shape_dim(ctx, a, a_shape, i);
    PolyUOp *b_dim = poly_transform_to_call_canonical_shape_dim(ctx, b, b_shape, i);
    if (!a_dim || !b_dim || a_dim != b_dim) return false;
  }
  return true;
}

static bool poly_transform_to_call_is_static_max_shape(PolyCtx *ctx, PolyUOp *u, PolyShape shape) {
  if (!ctx || !u || shape.ndim < 0) return false;
  for (int i = 0; i < shape.ndim; i++) {
    PolyUOp *dim = poly_transform_to_call_canonical_shape_dim(ctx, u, shape, i);
    int64_t value = 0;
    if (!dim || poly_uop_const_i64(dim, &value) != 0 || value != shape.dims[i]) return false;
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
  if (!poly_transform_to_call_same_exact_shape(ctx, view, view_shape, root, root_shape)) {
    bool at_static_max = poly_shape_eq(view_shape, root_shape) &&
                         poly_transform_to_call_is_static_max_shape(ctx, view, view_shape);
    if (!at_static_max) {
      int64_t root_numel = poly_shape_numel(root_shape);
      if (view_shape.ndim != 1 || root_numel < 0 || view_shape.dims[0] != root_numel) return NULL;
      view = poly_reshape(ctx, view, root_shape.dims, root_shape.ndim);
      if (!view) return NULL;
    }

    PolyUOp *starts[POLY_MAX_DIMS];
    PolyUOp *sizes[POLY_MAX_DIMS];
    PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(0));
    for (int i = 0; i < root_shape.ndim; i++) {
      starts[i] = zero;
      sizes[i] = poly_uop_shape_dim(ctx, root, i);
      if (!sizes[i])
        sizes[i] = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(root_shape.dims[i]));
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
      if (step->arg.kind != POLY_ARG_NONE || step->n_src != 2) return NULL;
      /* Pinned callify substitutes only the materialized base in the immutable
       * movement graph (callify.py:204-220, tensor.py:202-206). Preserve the
       * exact symbolic RESHAPE shape source instead of its cached max shape. */
      PolyUOp *reshape_src[2] = {view, step->src[1]};
      view = poly_uop(ctx, POLY_OP_RESHAPE, step->dtype, reshape_src, 2, poly_arg_none());
    } else if (step->op == POLY_OP_EXPAND && step->arg.kind == POLY_ARG_NONE && step->n_src == 2) {
      /* Same immutable-source substitution as RESHAPE and pinned callify.py:
       * replace only the materialized base, not the shape-value UOp. */
      PolyUOp *expand_src[2] = {view, step->src[1]};
      view = poly_uop(ctx, POLY_OP_EXPAND, step->dtype, expand_src, 2, poly_arg_none());
    } else if (step->op == POLY_OP_PERMUTE && step->arg.kind == POLY_ARG_INT_TUPLE) {
      view = poly_permute(ctx, view, step->arg.int_tuple.vals, step->arg.int_tuple.n);
    } else if (step->op == POLY_OP_PAD && step->arg.kind == POLY_ARG_NONE && step->n_src == 3) {
      PolyUOp *offsets[POLY_MAX_DIMS], *sizes[POLY_MAX_DIMS];
      int n_offsets = poly_uop_as_shape(ctx, step->src[1], offsets, POLY_MAX_DIMS);
      int n_sizes = poly_uop_as_shape(ctx, step->src[2], sizes, POLY_MAX_DIMS);
      view = n_offsets >= 0 && n_offsets == n_sizes
                 ? poly_pad_uop(ctx, view, offsets, sizes, n_offsets)
                 : NULL;
    } else if (step->op == POLY_OP_SHRINK && step->arg.kind == POLY_ARG_NONE && step->n_src >= 3) {
      PolyUOp *starts[POLY_MAX_DIMS], *sizes[POLY_MAX_DIMS];
      int n_starts = poly_uop_as_shape(ctx, step->src[1], starts, POLY_MAX_DIMS);
      int n_sizes = poly_uop_as_shape(ctx, step->src[2], sizes, POLY_MAX_DIMS);
      view = n_starts >= 0 && n_starts == n_sizes
                 ? poly_shrink_uop(ctx, view, starts, sizes, n_starts)
                 : NULL;
    } else if (step->op == POLY_OP_FLIP && step->arg.kind == POLY_ARG_INT_TUPLE) {
      view = poly_uop1(ctx, POLY_OP_FLIP, view->dtype, view, step->arg);
    }
  }
  return view;
}

static PolyUOp *poly_transform_to_call_after_result_buffer(PolyCtx *ctx, PolyUOp *u) {
  if (poly_uop_is_bound_var(u)) return NULL;
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
static PolyUOp *poly_transform_to_call_empty_buffer_like(
    PolyCtx *ctx,
    PolyTransformToCallCtx *tctx,
    PolyDType dtype,
    PolyShape shape,
    PolyUOp *like
) {
  /* Pinned UOp.empty_like forwards the exact self.device into new_buffer
   * (uop/ops.py:733-750). Keep the canonical DEVICE UOp already present in
   * the physical graph; reducing CPU:1 to the CPU backend here aliases the
   * result before ParamArg/Bufferize can preserve it. */
  /* Pinned UOp.device is a recursive_property (uop/ops.py:770-783): shared
   * immutable subgraphs are evaluated once. Keep the equivalent cache scoped
   * to this callify pass so exact-device discovery stays linear in UOp count. */
  PolyUOp *device = poly_uop_device_uop_cached(ctx, like, tctx ? tctx->device_memo : NULL);
  if (!device) device = poly_device_uop(ctx, poly_device_default());
  if (!device || shape.ndim < 0 || shape.ndim > POLY_MAX_DIMS) return NULL;

  /* Pinned UOp.empty_like uses self.axis and shard_shape before UOp.empty
   * (uop/ops.py:742-750). UOp.axis is the source-backed proof that a wrapper
   * such as CONTIGUOUS still carries the same shard axis; descendant search is
   * not equivalent because COPY and partial SHRINK deliberately clear it. */
  int multi_axis = -1;
  int tuple_count = 0;
  if (device->arg.kind == POLY_ARG_STRING_TUPLE) {
    tuple_count = device->arg.string_tuple.n;
    if (tuple_count <= 0) return NULL;
    int axis = -1;
    if (poly_uop_axis_cached(ctx, like, tctx ? tctx->axis_memo : NULL, &axis)) {
      if (axis < 0 || axis >= shape.ndim) return NULL;
      multi_axis = axis;
    }
  }

  int64_t buffer_shape[POLY_MAX_DIMS];
  for (int i = 0; i < shape.ndim; i++)
    buffer_shape[i] = shape.dims[i];
  if (multi_axis >= 0) {
    if (buffer_shape[multi_axis] < 0 || buffer_shape[multi_axis] % tuple_count != 0) return NULL;
    buffer_shape[multi_axis] /= tuple_count;
  }
  PolyShape allocation_shape = {.dims = buffer_shape, .ndim = shape.ndim};
  int64_t numel = poly_shape_numel(allocation_shape);
  if (numel < 0) return NULL;
  PolyDType storage_dtype = poly_dtype_is_weak(dtype) ? poly_dtype_strong(dtype) : dtype;
  PolyUOp *buffer =
      poly_uop_new_buffer(ctx, device, numel, storage_dtype, poly_ctx_next_unique_id(ctx));
  if (!buffer) return NULL;
  /* Pinned UOp.empty is BUFFER(prod(max_shape)).reshape(max_shape) before
   * symbolic shrink_to (uop/ops.py:742-750), while MovementMixin.reshape
   * returns self for an unchanged shape (mixin/movement.py:145-165).  Preserve
   * the natural flat BUFFER for rank-1 and shape only scalar/rank>1 outputs. */
  PolyShape natural_shape = poly_uop_max_shape_cached(ctx, buffer);
  PolyUOp *local = poly_shape_eq(natural_shape, allocation_shape)
                       ? buffer
                       : poly_reshape(ctx, buffer, buffer_shape, shape.ndim);
  if (!local || multi_axis < 0) return local;
  PolyUOp *device_range = poly_range(ctx, tuple_count, -1, POLY_AXIS_DEVICE);
  int64_t axis = multi_axis;
  PolyUOp *ranges[] = {device_range};
  return device_range ? poly_unshard(ctx, local, &axis, ranges, 1) : NULL;
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

static void poly_transform_to_call_ctx_free(PolyTransformToCallCtx *tctx) {
  if (!tctx) return;
  free(tctx->stores);
  free(tctx->cached_orig);
  free(tctx->cached_repl);
  free(tctx->public_orig);
  free(tctx->public_repl);
  free(tctx->tag_orig);
  poly_map_destroy(tctx->requested_bases);
  poly_map_destroy(tctx->original_uops);
  poly_map_destroy(tctx->device_memo);
  poly_map_destroy(tctx->axis_memo);
  tctx->stores = NULL;
  tctx->cached_orig = NULL;
  tctx->cached_repl = NULL;
  tctx->public_orig = NULL;
  tctx->public_repl = NULL;
  tctx->tag_orig = NULL;
  tctx->requested_bases = NULL;
  tctx->original_uops = NULL;
  tctx->device_memo = NULL;
  tctx->axis_memo = NULL;
  tctx->n_stores = 0;
  tctx->stores_cap = 0;
  tctx->n_cached = 0;
  tctx->cached_cap = 0;
  tctx->n_public = 0;
  tctx->public_cap = 0;
  tctx->n_tag_orig = 0;
  tctx->tag_orig_cap = 0;
  tctx->publication_phase = false;
}

/* Pinned RewriteContext rebuilds a source-changing UOp with the original
 * op/dtype/arg/tag. Polygrad's tag is split into tag/tag_arg, so both fields
 * are part of the same reconstruction identity. */
static PolyUOp *poly_rebuild_with_sources(PolyCtx *ctx, PolyUOp *u, PolyUOp **src) {
  if (!ctx || !u || (u->n_src > 0 && !src)) return NULL;
  return (u->tag != 0 || u->tag_arg.kind != POLY_ARG_NONE)
             ? poly_uop_tagged_arg(ctx, u->op, u->dtype, src, u->n_src, u->arg, u->tag, u->tag_arg)
             : poly_uop(ctx, u->op, u->dtype, src, u->n_src, u->arg);
}

static PolyUOp *poly_uop_with_metadata_from(
    PolyCtx *ctx,
    PolyUOp *prototype,
    PolyOps op,
    PolyDType dtype,
    PolyUOp **src,
    int n_src,
    PolyArg arg
) {
  if (!ctx || !prototype || (n_src > 0 && !src)) return NULL;
  return (prototype->tag != 0 || prototype->tag_arg.kind != POLY_ARG_NONE)
             ? poly_uop_tagged_arg(
                   ctx, op, dtype, src, n_src, arg, prototype->tag, prototype->tag_arg
               )
             : poly_uop(ctx, op, dtype, src, n_src, arg);
}

static PolyUOp *poly_transform_to_call_fail(
    PolyTransformToCallCtx *tctx,
    PolyUOp **out_uops,
    int n
) {
  if (out_uops)
    for (int i = 0; i < n; i++)
      out_uops[i] = NULL;
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

static bool poly_transform_to_call_publish_replacement(
    PolyTransformToCallCtx *tctx,
    PolyUOp *orig,
    PolyUOp *repl
);

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
  *out_orig = tctx->public_orig;
  *out_repl = tctx->public_repl;
  *out_n = tctx->n_public;
  tctx->public_orig = NULL;
  tctx->public_repl = NULL;
  tctx->n_public = 0;
  tctx->public_cap = 0;
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

  if (u->op == POLY_OP_CALL) return poly_transform_to_call_collect_after_stores(tctx, u);

  if (u->op == POLY_OP_AFTER && u->n_src >= 2 && poly_uop_has_buffer_identity(u->src[0])) {
    PolyUOp *base = u->src[0];
    while (base && base->op == POLY_OP_AFTER && base->n_src >= 1)
      base = base->src[0];
    if (!base || !poly_transform_to_call_cache_replacement(tctx, u, base)) return false;
    /* Pinned finalize_after records the canonical AFTER occurrence, not a
     * second raw STORE spelling of that same effect (callify.py:169-181).
     * Keep traversing below so distinct nested effects remain visible; the
     * exact-identity store set suppresses only repeated routes to this AFTER. */
    if (!poly_transform_to_call_collect_after_stores(tctx, u)) return false;
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
    /* Current finalize_after records AFTER occurrences, not a parallel CALL
     * list (tinygrad/tensor.py:178-240). Opaque CALLs remain dependencies of
     * their enclosing AFTER; only effects reachable through arguments are
     * collected here. */
    return ok;
  }
  if (effect->op != POLY_OP_AFTER) return true;

  PolyUOp *target = NULL;
  if (poly_transform_to_call_after_store_assign(effect, &target, NULL)) {
    /* Keep the canonical AFTER+STORE effect root. This is the identity that a
     * downstream assignment consumes and the unit tinygrad split_kernels
     * rewrites once. Extracting only STORE loses that sharing and executes the
     * same immutable effect twice when it is also nested in another target. */
    PolyUOp *base = target;
    while (base && base->op == POLY_OP_AFTER && base->n_src >= 1)
      base = base->src[0];
    if (!base || !poly_transform_to_call_cache_replacement(tctx, effect, base)) return false;
    if (out_found) *out_found = true;
    return poly_transform_to_call_append_store(tctx, effect);
  }

  /* Current finalize_after records every remaining AFTER as one assignment
   * occurrence (tensor.py:178-192). Preserve nested CALL/END dependencies in
   * that node; get_kernel_graph and create_schedule own their extraction. */
  if (effect->n_src > 1 && poly_uop_has_buffer_identity(effect->src[0])) {
    if (out_found) *out_found = true;
    return poly_transform_to_call_append_store(tctx, effect);
  }

  /* View assign follows tinygrad's nested shape:
   * AFTER(base_identity, AFTER(view, STORE(view, value))).
   * The outer AFTER returns the storage identity; all nested STORE effects
   * under src[1:] must be scheduled before that identity is considered ready. */
  for (int i = 1; i < effect->n_src; i++) {
    if (!poly_transform_to_call_collect_after_stores_ex(tctx, effect->src[i], out_found))
      return false;
  }
  if (effect->n_src >= 1) {
    PolyUOp *base = effect->src[0];
    while (base && base->op == POLY_OP_AFTER && base->n_src >= 1)
      base = base->src[0];
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
  if (poly_uop_is_bound_var(u)) return true;

  /* Pinned pm_finalize_call rewrites bottom-up: nested canonical AFTER
   * dependencies are recorded before the current AFTER. Recurse through the
   * target and STORE value, but not through the owned raw STORE node, then
   * append this whole immutable effect as the canonical assignment unit. */
  PolyUOp *canonical_target = NULL;
  PolyUOp *canonical_store = NULL;
  if (poly_transform_to_call_after_store_assign(u, &canonical_target, &canonical_store)) {
    if (!poly_transform_to_call_collect_pending_effects(tctx, canonical_target, visited))
      return false;
    if (!poly_transform_to_call_collect_pending_effects(tctx, canonical_store->src[1], visited))
      return false;
    return poly_transform_to_call_collect_after_stores(tctx, u);
  }

  /* A requested view can contain the assign boundary below its root, e.g.
   * SHRINK(AFTER(BUFFER, AFTER(SHRINK(BUFFER), STORE(...)))).
   * Schedule those side-effect stores before materializing the requested
   * value, otherwise view.realize() reads the old base buffer. */
  if (u->op == POLY_OP_AFTER && u->n_src >= 2 && poly_uop_has_buffer_identity(u->src[0])) {
    PolyUOp *base = u->src[0];
    while (base && base->op == POLY_OP_AFTER && base->n_src >= 1)
      base = base->src[0];
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
        effect_set, poly_ptr_hash(tctx->stores[i]), tctx->stores[i], tctx->stores[i], poly_ptr_eq
    );

  PolyScratchMark scratch = poly_ctx_scratch_mark(ctx);
  int n_topo = 0;
  PolyUOp **topo = poly_toposort_scratch(ctx, collected, &n_topo);
  int n_ordered = 0;
  if (topo) {
    for (int i = 0; i < n_topo; i++) {
      PolyUOp *effect = poly_map_get(effect_set, poly_ptr_hash(topo[i]), topo[i], poly_ptr_eq);
      if (effect) tctx->stores[n_ordered++] = effect;
    }
  }
  poly_ctx_scratch_rewind(ctx, scratch);
  poly_map_destroy(effect_set);
  return topo && n_ordered == tctx->n_stores;
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
  PolyUOp *function =
      n_ordered > 0 ? poly_uop_substitute(ctx, sink, ordered, params, n_ordered) : sink;
  free(params);
  if (!function) {
    if (ordered != ordered_stack) free(ordered);
    return NULL;
  }

  /* Tinygrad tensor.py:224-242 records every callified write as
   * AFTER(target, STORE(target, value)).  Raw C/Model effect sinks already
   * own storage, so add only that missing state edge before scheduling. */
  PolyUOp **effects =
      function->n_src > 0 ? malloc((size_t)function->n_src * sizeof(*effects)) : NULL;
  if (function->n_src > 0 && !effects) {
    if (ordered != ordered_stack) free(ordered);
    return NULL;
  }
  bool changed = false;
  for (int i = 0; i < function->n_src; i++) {
    PolyUOp *effect = function->src[i];
    effects[i] = effect;
    if (!effect || effect->op != POLY_OP_STORE || effect->n_src < 1) continue;
    PolyUOp *target = poly_uop_buf_uop(ctx, effect->src[0]);
    if (!target) {
      free(effects);
      if (ordered != ordered_stack) free(ordered);
      return NULL;
    }
    PolyUOp *after_src[] = {target, effect};
    effects[i] = poly_uop(ctx, POLY_OP_AFTER, target->dtype, after_src, 2, poly_arg_none());
    if (!effects[i]) {
      free(effects);
      if (ordered != ordered_stack) free(ordered);
      return NULL;
    }
    changed = true;
  }
  if (changed) function = poly_sink_n(ctx, effects, function->n_src);
  free(effects);
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

  PolyCallInfo info = {0};
  PolyUOp *call = poly_uop(ctx, POLY_OP_CALL, POLY_VOID, src, n_src, poly_arg_call_info(&info));
  if (ordered != ordered_stack) free(ordered);
  free(src);
  return call;
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

static bool poly_transform_to_call_publish_replacement(
    PolyTransformToCallCtx *tctx,
    PolyUOp *orig,
    PolyUOp *repl
) {
  if (!tctx || !orig || !repl) return false;
  for (int i = 0; i < tctx->n_public; i++) {
    if (tctx->public_orig[i] != orig) continue;
    tctx->public_repl[i] = repl;
    return true;
  }
  if (tctx->n_public >= tctx->public_cap) {
    int new_cap =
        tctx->public_cap ? tctx->public_cap * 2 : POLY_TRANSFORM_TO_CALL_INITIAL_REPLACEMENTS;
    PolyUOp **new_orig = realloc(tctx->public_orig, (size_t)new_cap * sizeof(*new_orig));
    if (!new_orig) return false;
    tctx->public_orig = new_orig;
    PolyUOp **new_repl = realloc(tctx->public_repl, (size_t)new_cap * sizeof(*new_repl));
    if (!new_repl) return false;
    tctx->public_repl = new_repl;
    tctx->public_cap = new_cap;
  }
  tctx->public_orig[tctx->n_public] = orig;
  tctx->public_repl[tctx->n_public] = repl;
  tctx->n_public++;
  return true;
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
    bool original = tctx->original_uops &&
                    poly_map_get(tctx->original_uops, poly_ptr_hash(orig), orig, poly_ptr_eq);
    return !tctx->publication_phase || !original ||
           poly_transform_to_call_publish_replacement(tctx, orig, repl);
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
  bool original = tctx->original_uops &&
                  poly_map_get(tctx->original_uops, poly_ptr_hash(orig), orig, poly_ptr_eq);
  return !tctx->publication_phase || !original ||
         poly_transform_to_call_publish_replacement(tctx, orig, repl);
}

static bool poly_callify_tag_ids(PolyUOp *u, const int64_t **out_ids, int *out_n) {
  if (out_ids) *out_ids = NULL;
  if (out_n) *out_n = 0;
  if (!u || u->tag != POLY_CALLIFY_TAG_MARKER || u->tag_arg.kind != POLY_ARG_INT_TUPLE ||
      u->tag_arg.int_tuple.n <= 0 || !u->tag_arg.int_tuple.vals)
    return false;
  if (out_ids) *out_ids = u->tag_arg.int_tuple.vals;
  if (out_n) *out_n = u->tag_arg.int_tuple.n;
  return true;
}

static PolyArg poly_callify_tag_arg(int64_t *ids, int n) {
  PolyArg arg = poly_arg_none();
  arg.kind = POLY_ARG_INT_TUPLE;
  arg.int_tuple.vals = ids;
  arg.int_tuple.n = n;
  return arg;
}

static PolyUOp *poly_callify_rebuild_with_tag_ids(
    PolyCtx *ctx,
    PolyUOp *prototype,
    PolyUOp **src,
    int64_t *ids,
    int n_ids
) {
  if (!ctx || !prototype || n_ids < 0 || (n_ids > 0 && !ids)) return NULL;
  return poly_uop_tagged_arg(
      ctx, prototype->op, prototype->dtype, src ? src : prototype->src, prototype->n_src,
      prototype->arg, POLY_CALLIFY_TAG_MARKER, poly_callify_tag_arg(ids, n_ids)
  );
}

static PolyUOp *poly_callify_rebuild_without_tag(PolyCtx *ctx, PolyUOp *u, PolyUOp **src) {
  if (!ctx || !u) return NULL;
  return poly_uop(ctx, u->op, u->dtype, src ? src : u->src, u->n_src, u->arg);
}

static bool poly_callify_append_tag_original(
    PolyTransformToCallCtx *tctx,
    PolyUOp *original,
    int64_t *out_id
) {
  if (!tctx || !original || !out_id) return false;
  if (tctx->n_tag_orig >= tctx->tag_orig_cap) {
    int new_cap = tctx->tag_orig_cap ? tctx->tag_orig_cap * 2 : 64;
    PolyUOp **new_orig = realloc(tctx->tag_orig, (size_t)new_cap * sizeof(*new_orig));
    if (!new_orig) return false;
    tctx->tag_orig = new_orig;
    tctx->tag_orig_cap = new_cap;
  }
  *out_id = tctx->n_tag_orig;
  tctx->tag_orig[tctx->n_tag_orig++] = original;
  return true;
}

static bool poly_callify_publish_tagged_originals(
    PolyTransformToCallCtx *tctx,
    PolyUOp *tagged,
    PolyUOp *replacement
) {
  if (!tctx || !tagged || !replacement) return false;
  const int64_t *ids = NULL;
  int n_ids = 0;
  if (!poly_callify_tag_ids(tagged, &ids, &n_ids)) return true;
  for (int i = 0; i < n_ids; i++) {
    if (ids[i] < 0 || ids[i] >= tctx->n_tag_orig ||
        !poly_transform_to_call_publish_replacement(tctx, tctx->tag_orig[ids[i]], replacement))
      return false;
  }
  return true;
}

/* Pinned callify.py:14-17 tags only previously untagged UOps and stores the
 * original in AllocCtx.uop_list. Polygrad's existing tag/tag_arg CSE metadata
 * represents the one-element tuple for this pass; it is stripped before
 * scheduling and never becomes Tensor or context state. */
static PolyUOp *poly_callify_tag_uop(
    PolyCtx *ctx,
    PolyTransformToCallCtx *tctx,
    PolyUOp *original,
    PolyUOp *u
) {
  if (!ctx || !tctx || !original || !u) return NULL;
  if (u->tag != 0 || u->tag_arg.kind != POLY_ARG_NONE) return u;
  int64_t id = -1;
  if (!poly_callify_append_tag_original(tctx, original, &id)) return NULL;
  return poly_callify_rebuild_with_tag_ids(ctx, u, NULL, &id, 1);
}

static PolyUOp *poly_callify_merge_tagged_uops(
    PolyCtx *ctx,
    PolyUOp *prototype,
    PolyUOp **src,
    PolyUOp *first,
    PolyUOp *second
) {
  const int64_t *first_ids = NULL, *second_ids = NULL;
  int n_first = 0, n_second = 0;
  if (!poly_callify_tag_ids(first, &first_ids, &n_first) ||
      !poly_callify_tag_ids(second, &second_ids, &n_second))
    return NULL;
  int n_ids = n_first + n_second;
  int64_t stack_ids[16];
  int64_t *ids = stack_ids;
  if (n_ids > (int)(sizeof(stack_ids) / sizeof(stack_ids[0]))) {
    ids = malloc((size_t)n_ids * sizeof(*ids));
    if (!ids) return NULL;
  }
  memcpy(ids, first_ids, (size_t)n_first * sizeof(*ids));
  memcpy(ids + n_first, second_ids, (size_t)n_second * sizeof(*ids));
  PolyUOp *ret = poly_callify_rebuild_with_tag_ids(ctx, prototype, src, ids, n_ids);
  if (ids != stack_ids) free(ids);
  return ret;
}

static bool poly_callify_copy_from_creation(PolyUOp *copy) {
  if (!copy || copy->op != POLY_OP_COPY || copy->n_src < 1) return false;
  PolyDevice source_device = poly_uop_device(copy->src[0]);
  return source_device == POLY_DEVICE_HOST || source_device == POLY_DEVICE_DISK;
}

static PolyUOp *poly_transform_to_call_add_tags(
    PolyCtx *ctx,
    PolyTransformToCallCtx *tctx,
    PolyUOp *u,
    PolyMap *memo
) {
  if (!ctx || !tctx || !u || !memo) return NULL;
  PolyUOp *memoized = poly_map_get(memo, poly_ptr_hash(u), u, poly_ptr_eq);
  if (memoized) return memoized;

  PolyUOp *src_buf[16];
  PolyUOp **new_src = src_buf;
  if (u->n_src > (int)(sizeof(src_buf) / sizeof(src_buf[0]))) {
    new_src = malloc((size_t)u->n_src * sizeof(*new_src));
    if (!new_src) return NULL;
  }
  bool changed = false;
  int first_src = ((u->op == POLY_OP_CALL || u->op == POLY_OP_FUNCTION) && u->n_src > 0) ? 1 : 0;
  for (int i = 0; i < first_src; i++)
    new_src[i] = u->src[i];
  for (int i = first_src; i < u->n_src; i++) {
    new_src[i] = poly_transform_to_call_add_tags(ctx, tctx, u->src[i], memo);
    if (!new_src[i]) {
      if (new_src != src_buf) free(new_src);
      return NULL;
    }
    if (new_src[i] != u->src[i]) changed = true;
  }
  PolyUOp *ret = changed ? poly_rebuild_with_sources(ctx, u, new_src) : u;
  if (new_src != src_buf) free(new_src);
  if (!ret) return NULL;

  /* tensor.py:disk_copy_is_buffer publishes an independent file BUFFER;
   * the COPY remains an effect, with an empty tag to prevent materialization
   * as a compute kernel. The allocator makes both identities see the file. */
  if (ret->op == POLY_OP_COPY && poly_uop_device(ret) == POLY_DEVICE_DISK && ret->tag == 0) {
    PolyUOp *buffer = poly_transform_to_call_empty_buffer_like(
        ctx, tctx, ret->dtype, poly_uop_max_shape_cached(ctx, ret), ret
    );
    if (!buffer || !poly_transform_to_call_cache_replacement(tctx, u, buffer)) return NULL;
    ret = poly_callify_rebuild_with_tag_ids(ctx, ret, NULL, NULL, 0);
    if (!ret) return NULL;
  } else if (ret->op == POLY_OP_COPY && poly_callify_copy_from_creation(ret)) {
    ret = poly_callify_tag_uop(ctx, tctx, u, ret);
    if (!ret) return NULL;
  }

  PolyUOp *assign_store = NULL;
  bool has_assignment_effect = !poly_uop_is_bound_var(ret) &&
                               poly_transform_to_call_after_store_assign(ret, NULL, &assign_store);
  bool assignment = has_assignment_effect && ret->n_src == 2 && ret->src[1] == assign_store;
  if (assignment) {
    ret = poly_callify_tag_uop(ctx, tctx, u, ret);
    if (!ret) return NULL;

    /* Pinned callify.py:32-39 merges a creation-COPY tag into the assignment
     * AFTER, clears the COPY tag, and preserves ordered (AFTER, COPY)
     * provenance so both originals finalize to the assignment destination. */
    PolyUOp *copy = assign_store->n_src >= 2 ? assign_store->src[1] : NULL;
    if (assign_store->n_src == 2 && copy && copy->op == POLY_OP_COPY &&
        poly_callify_tag_ids(ret, NULL, NULL) && poly_callify_tag_ids(copy, NULL, NULL)) {
      PolyUOp *untagged_copy = poly_callify_rebuild_without_tag(ctx, copy, NULL);
      PolyUOp *store_src[2] = {assign_store->src[0], untagged_copy};
      PolyUOp *untagged_store =
          untagged_copy ? poly_rebuild_with_sources(ctx, assign_store, store_src) : NULL;
      PolyUOp *after_src[2] = {ret->src[0], untagged_store};
      PolyUOp *merged =
          untagged_store ? poly_callify_merge_tagged_uops(ctx, ret, after_src, ret, copy) : NULL;
      if (!merged) return NULL;
      ret = merged;
    }
  }

  /* Pinned apply_after records the original untagged AFTER immediately, even
   * when bottom-up child tagging rebuilt the pass-local occurrence. Tagged
   * assignment candidates publish only if their provenance reaches finalize. */
  if (ret->op == POLY_OP_AFTER && !poly_uop_is_bound_var(ret)) {
    PolyUOp *base = ret->n_src >= 1 ? ret->src[0] : NULL;
    while (base && base->op == POLY_OP_AFTER && base->n_src >= 1)
      base = base->src[0];
    if (!base) return NULL;
    if (poly_callify_tag_ids(ret, NULL, NULL)) {
      if (!poly_transform_to_call_cache_replacement(tctx, ret, base)) return NULL;
    } else if (!poly_transform_to_call_publish_replacement(tctx, u, base)) {
      return NULL;
    }
  }

  if (ret->op == POLY_OP_CONTIGUOUS) {
    ret = poly_callify_tag_uop(ctx, tctx, u, ret);
    if (!ret) return NULL;
  }

  if (tctx->requested_bases &&
      poly_map_get(tctx->requested_bases, poly_ptr_hash(u), u, poly_ptr_eq)) {
    ret = poly_callify_tag_uop(ctx, tctx, u, ret);
    if (!ret) return NULL;
  }

  poly_map_set(memo, poly_ptr_hash(u), u, ret, poly_ptr_eq);
  return ret;
}

static PolyUOp *poly_transform_to_call_finalize_tags(
    PolyCtx *ctx,
    PolyTransformToCallCtx *tctx,
    PolyUOp *u,
    PolyMap *memo
) {
  if (!ctx || !tctx || !u || !memo) return NULL;
  PolyUOp *memoized = poly_map_get(memo, poly_ptr_hash(u), u, poly_ptr_eq);
  if (memoized) return memoized;

  PolyUOp *src_buf[16];
  PolyUOp **new_src = src_buf;
  if (u->n_src > (int)(sizeof(src_buf) / sizeof(src_buf[0]))) {
    new_src = malloc((size_t)u->n_src * sizeof(*new_src));
    if (!new_src) return NULL;
  }
  bool changed = false;
  int first_src = ((u->op == POLY_OP_CALL || u->op == POLY_OP_FUNCTION) && u->n_src > 0) ? 1 : 0;
  for (int i = 0; i < first_src; i++)
    new_src[i] = u->src[i];
  for (int i = first_src; i < u->n_src; i++) {
    new_src[i] = poly_transform_to_call_finalize_tags(ctx, tctx, u->src[i], memo);
    if (!new_src[i]) {
      if (new_src != src_buf) free(new_src);
      return NULL;
    }
    if (new_src[i] != u->src[i]) changed = true;
  }
  PolyUOp *ret = changed ? poly_rebuild_with_sources(ctx, u, new_src) : u;
  if (new_src != src_buf) free(new_src);
  if (!ret) return NULL;

  const int64_t *ids = NULL;
  int n_ids = 0;
  if (ret->op == POLY_OP_AFTER && !poly_uop_is_bound_var(ret) &&
      poly_callify_tag_ids(ret, &ids, &n_ids)) {
    PolyUOp *replacement = poly_transform_to_call_after_result_buffer(ctx, ret);
    if (!replacement || !poly_callify_publish_tagged_originals(tctx, ret, replacement)) return NULL;
    ret = poly_callify_rebuild_without_tag(ctx, ret, NULL);
    if (!ret) return NULL;
  }

  /* pm_finalize_call retains COPY-to-DISK even without STORE/AFTER. Store
   * the input occurrence until the shared finalize map is applied below. */
  if (ret->op == POLY_OP_COPY && poly_uop_device(ret) == POLY_DEVICE_DISK &&
      !poly_transform_to_call_append_store(tctx, u))
    return NULL;

  poly_map_set(memo, poly_ptr_hash(u), u, ret, poly_ptr_eq);
  return ret;
}

/* Tinygrad 2026-08-22/a9069c177a9d tensor.py:_make_buffer_view. Build the
 * canonical SHRINK/BITCAST storage argument and attach Buffer.view metadata
 * to that exact UOp. */
static PolyUOp *make_buffer_view(PolyCtx *ctx, PolyUOp *value) {
  if (!ctx || !value) return NULL;
  PolyUOp *view_key = poly_uop_buffer(ctx, value);
  if (!view_key || poly_uop_has_buffer_identity(view_key)) return NULL;
  PolyBuffer *view_storage = poly_buffer_get(ctx, view_key);
  PolyUOp *base = value;
  while (base && base->n_src > 0 && !poly_uop_has_buffer_identity(base) &&
         (poly_opset_has(POLY_GROUP_MOVEMENT, base->op) || base->op == POLY_OP_BITCAST))
    base = base->src[0];
  base = (PolyUOp *)poly_uop_get_buffer_identity(base);
  PolyBuffer *base_storage = base ? poly_buffer_get(ctx, base) : NULL;
  if (!base || !view_storage || !base_storage) return NULL;

  PolyShape base_shape = poly_uop_max_shape_cached(ctx, base);
  size_t base_itemsize = poly_dtype_itemsize(base->dtype);
  if (base_shape.ndim != 1 || base_itemsize == 0 || view_storage->offset % base_itemsize != 0 ||
      view_storage->nbytes % base_itemsize != 0)
    return NULL;
  uint64_t begin = view_storage->offset / base_itemsize;
  uint64_t length = view_storage->nbytes / base_itemsize;
  if (begin > INT64_MAX || length > INT64_MAX - begin) return NULL;
  int64_t bounds[1][2] = {{(int64_t)begin, (int64_t)(begin + length)}};
  PolyUOp *view = poly_shrink(ctx, base, bounds, 1);
  if (view && !poly_dtype_eq(view->dtype, value->dtype))
    view = poly_uop1(ctx, POLY_OP_BITCAST, value->dtype, view, poly_arg_none());
  return view && poly_uop_buffer(ctx, view) == view ? view : NULL;
}

static PolyUOp *poly_transform_to_call_materialize_view_copy(
    PolyCtx *ctx,
    PolyUOp *copy,
    PolyTransformToCallCtx *tctx
) {
  if (!ctx || !copy || !tctx || copy->op != POLY_OP_COPY || copy->n_src < 1) return NULL;

  PolyUOp *view = make_buffer_view(ctx, copy->src[0]);
  PolyUOp *copy_value =
      view ? poly_transform_to_call_rebuild_view(ctx, view, copy->src[0], NULL) : NULL;
  if (!copy_value) return NULL;
  PolyUOp **copy_src = malloc((size_t)copy->n_src * sizeof(*copy_src));
  if (!copy_src) return NULL;
  memcpy(copy_src, copy->src, (size_t)copy->n_src * sizeof(*copy_src));
  copy_src[0] = copy_value;
  PolyUOp *copy_body = poly_callify_tag_ids(copy, NULL, NULL)
                           ? poly_callify_rebuild_without_tag(ctx, copy, copy_src)
                           : poly_rebuild_with_sources(ctx, copy, copy_src);
  free(copy_src);
  if (!copy_body) return NULL;

  /* Current contiguous_mops_to_view only rewrites the COPY source when the
   * COPY is not a tagged requested/creation value (tinygrad/tensor.py:69-88,
   * 145-173). The enclosing tagged value owns materialization. */
  if (!poly_callify_tag_ids(copy, NULL, NULL)) return copy_body;

  PolyShape shape = poly_uop_max_shape_cached(ctx, copy);
  PolyUOp *buf = poly_transform_to_call_empty_buffer_like(ctx, tctx, copy->dtype, shape, copy);
  if (!buf) {
    fprintf(stderr, "poly_realize: output buffer creation failed\n");
    tctx->failed = true;
    return NULL;
  }

  PolyUOp *replacement = poly_transform_to_call_rebuild_view(ctx, buf, copy, NULL);
  PolyUOp *store = replacement ? poly_store_val(ctx, replacement, copy_body) : NULL;
  PolyUOp *after_src[2] = {replacement, store};
  PolyUOp *after = replacement && store ? poly_uop_with_metadata_from(
                                              ctx, copy, POLY_OP_AFTER, replacement->dtype,
                                              after_src, 2, poly_arg_none()
                                          )
                                        : NULL;
  /* Current callify keeps COPY in AFTER(buffer, STORE(buffer, COPY));
   * create_linear_with_vars later extracts the COPY CALL
   * (tinygrad/tensor.py:54-62,178-240; schedule/rangeify.py:565-590). */
  if (!replacement || !store || !after || !poly_transform_to_call_append_store(tctx, after) ||
      !poly_transform_to_call_cache_replacement(tctx, copy, replacement)) {
    tctx->failed = true;
    return NULL;
  }
  return after;
}

/* Pinned UOp.shrink_to for callify's precompiled FUNCTION boundary
 * (tinygrad/callify.py:117-142).  Read the exact symbolic dimensions from
 * `like`; max-shape allocation and the caller-visible symbolic view are
 * separate, just as UOp.empty(...).shrink_to(shape) is in tinygrad. */
static PolyUOp *poly_callify_shrink_to_like(PolyCtx *ctx, PolyUOp *value, PolyUOp *like) {
  if (!ctx || !value || !like) return NULL;
  int ndim = poly_uop_ndim(ctx, like);
  if (ndim < 0 || ndim > POLY_MAX_DIMS || poly_uop_ndim(ctx, value) != ndim) return NULL;
  PolyUOp *starts[POLY_MAX_DIMS], *sizes[POLY_MAX_DIMS];
  for (int i = 0; i < ndim; i++) {
    starts[i] = poly_const_int(ctx, 0);
    sizes[i] = poly_uop_shape_dim(ctx, like, i);
    if (!starts[i] || !sizes[i]) return NULL;
  }
  return poly_shrink_uop(ctx, value, starts, sizes, ndim);
}

/* Pinned Tensor.transform_precompiled_call (tinygrad/tensor.py:109-141):
 * allocate explicit outputs, redirect the
 * TUPLE body into output PARAMs, turn FUNCTION into opaque CALL, and expose
 * each result as AFTER(output, CALL). */
static PolyUOp *poly_transform_precompiled_call(
    PolyCtx *ctx,
    PolyTransformToCallCtx *tctx,
    PolyUOp *function
) {
  if (!ctx || !tctx || !function || function->op != POLY_OP_FUNCTION || function->n_src < 1 ||
      function->arg.kind != POLY_ARG_CALL_INFO || !function->arg.call_info ||
      !function->arg.call_info->precompile)
    return NULL;
  PolyUOp *body = function->src[0];
  if (!body || body->op != POLY_OP_TUPLE || body->n_src <= 0) return NULL;

  int n_inputs = function->n_src - 1;
  int n_outputs = body->n_src;
  PolyUOp **inputs = n_inputs > 0 ? malloc((size_t)n_inputs * sizeof(*inputs)) : NULL;
  PolyUOp **resolved = malloc((size_t)n_outputs * sizeof(*resolved));
  PolyUOp **outputs = malloc((size_t)n_outputs * sizeof(*outputs));
  PolyUOp **targets = malloc((size_t)n_outputs * sizeof(*targets));
  PolyUOp **items = malloc((size_t)n_outputs * sizeof(*items));
  PolyUOp **from = malloc((size_t)n_outputs * sizeof(*from));
  PolyUOp **to = malloc((size_t)n_outputs * sizeof(*to));
  PolyUOp **rewritten = malloc((size_t)n_outputs * sizeof(*rewritten));
  PolyUOp **call_src = malloc((size_t)(1 + n_inputs + n_outputs) * sizeof(*call_src));
  PolyUOp **tuple_src = malloc((size_t)n_outputs * sizeof(*tuple_src));
  if ((n_inputs > 0 && !inputs) || !resolved || !outputs || !targets || !items || !from || !to ||
      !rewritten || !call_src || !tuple_src) {
    free(inputs);
    free(resolved);
    free(outputs);
    free(targets);
    free(items);
    free(from);
    free(to);
    free(rewritten);
    free(call_src);
    free(tuple_src);
    return NULL;
  }

  bool ok = true;
  for (int i = 0; ok && i < n_inputs; i++) {
    PolyUOp *input = function->src[i + 1];
    inputs[i] = input && input->op != POLY_OP_AFTER ? poly_contiguous(ctx, input) : input;
    ok = inputs[i] != NULL;
  }
  for (int i = 0; ok && i < n_outputs; i++) {
    resolved[i] = poly_uop1(ctx, POLY_OP_GETTUPLE, body->src[i]->dtype, function, poly_arg_int(i));
    PolyShape shape =
        resolved[i] ? poly_uop_max_shape_cached(ctx, resolved[i]) : (PolyShape){.ndim = -1};
    PolyUOp *allocated = resolved[i] ? poly_transform_to_call_empty_buffer_like(
                                           ctx, tctx, resolved[i]->dtype, shape, resolved[i]
                                       )
                                     : NULL;
    outputs[i] = allocated ? poly_callify_shrink_to_like(ctx, allocated, resolved[i]) : NULL;
    PolyUOp *param = outputs[i] ? poly_uop_param(ctx, n_inputs + i, outputs[i]) : NULL;
    targets[i] = param ? poly_callify_shrink_to_like(ctx, param, body->src[i]) : NULL;
    ok = resolved[i] && outputs[i] && targets[i];
  }

  int n_sub = 0;
  for (int i = 0; ok && i < n_outputs; i++) {
    PolyUOp *source = body->src[i];
    int n_deps = 0;
    for (PolyUOp *cur = source; cur && cur->op == POLY_OP_AFTER && cur->n_src >= 1;
         cur = cur->src[0]) {
      /* AFTER needs one value source in addition to the flattened effects.
       * Respect the existing C arity ceiling before summing or allocating. */
      if (cur->n_src - 1 > UINT16_MAX - 1 - n_deps) {
        ok = false;
        break;
      }
      n_deps += cur->n_src - 1;
    }
    if (!ok) break;
    PolyUOp **deps = n_deps > 0 ? malloc((size_t)n_deps * sizeof(*deps)) : NULL;
    if (n_deps > 0 && !deps) {
      ok = false;
      break;
    }
    int dep_i = 0;
    while (source && source->op == POLY_OP_AFTER && source->n_src >= 1) {
      for (int j = 1; j < source->n_src; j++)
        deps[dep_i++] = source->src[j];
      source = source->src[0];
    }
    PolyUOp *placed = NULL;
    if (source && source->op == POLY_OP_CONTIGUOUS && source->n_src == 1) {
      PolyUOp *store = poly_store_val(ctx, targets[i], source->src[0]);
      PolyUOp *after_src[2] = {targets[i], store};
      placed = store
                   ? poly_uop(ctx, POLY_OP_AFTER, targets[i]->dtype, after_src, 2, poly_arg_none())
                   : NULL;
    } else if (source && (source->op == POLY_OP_BUFFER || source->op == POLY_OP_UNSHARD) && poly_uop_has_buffer_identity(source)) {
      placed = targets[i];
    }

    bool already_subbed = false;
    if (placed)
      for (int j = 0; j < n_sub; j++)
        if (from[j] == source) already_subbed = true;
    if (placed && !already_subbed) {
      from[n_sub] = source;
      to[n_sub] = placed;
      n_sub++;
      if (n_deps == 0) {
        items[i] = source;
      } else {
        PolyUOp **after_src = malloc((size_t)(1 + n_deps) * sizeof(*after_src));
        if (!after_src) {
          free(deps);
          ok = false;
          break;
        }
        after_src[0] = source;
        memcpy(after_src + 1, deps, (size_t)n_deps * sizeof(*deps));
        items[i] =
            poly_uop(ctx, POLY_OP_AFTER, source->dtype, after_src, 1 + n_deps, poly_arg_none());
        free(after_src);
      }
    } else {
      /* Pinned s.after(*after_deps) is the STORE value: sibling effects on
       * AFTER(target, STORE, ...) do not establish that dependency. */
      PolyUOp *value = source;
      if (n_deps > 0) {
        PolyUOp **value_src = malloc((size_t)(1 + n_deps) * sizeof(*value_src));
        if (!value_src) {
          free(deps);
          ok = false;
          break;
        }
        value_src[0] = source;
        memcpy(value_src + 1, deps, (size_t)n_deps * sizeof(*deps));
        value = poly_uop(ctx, POLY_OP_AFTER, source->dtype, value_src, 1 + n_deps, poly_arg_none());
        free(value_src);
      }
      PolyUOp *store = value ? poly_store_val(ctx, targets[i], value) : NULL;
      items[i] =
          store
              ? poly_uop2(ctx, POLY_OP_AFTER, targets[i]->dtype, targets[i], store, poly_arg_none())
              : NULL;
    }
    free(deps);
    ok = items[i] != NULL;
  }

  if (ok && n_sub > 0 &&
      poly_uop_substitute_many(ctx, items, n_outputs, from, to, n_sub, rewritten) != 0)
    ok = false;
  if (ok && n_sub == 0) memcpy(rewritten, items, (size_t)n_outputs * sizeof(*rewritten));

  PolyUOp *call = NULL;
  if (ok) {
    PolyUOp *sink = poly_sink_n(ctx, rewritten, n_outputs);
    call_src[0] = sink;
    for (int i = 0; i < n_inputs; i++)
      call_src[1 + i] = inputs[i];
    for (int i = 0; i < n_outputs; i++)
      call_src[1 + n_inputs + i] = outputs[i];
    call = sink ? poly_uop(
                      ctx, POLY_OP_CALL, function->dtype, call_src, 1 + n_inputs + n_outputs,
                      function->arg
                  )
                : NULL;
    ok = call != NULL;
  }
  for (int i = 0; ok && i < n_outputs; i++) {
    PolyUOp *after_src[2] = {outputs[i], call};
    PolyUOp *after = poly_uop(ctx, POLY_OP_AFTER, outputs[i]->dtype, after_src, 2, poly_arg_none());
    tuple_src[i] = after ? poly_callify_shrink_to_like(ctx, after, resolved[i]) : NULL;
    ok = tuple_src[i] != NULL;
  }
  PolyUOp *ret =
      ok ? poly_uop(ctx, POLY_OP_TUPLE, POLY_VOID, tuple_src, n_outputs, poly_arg_none()) : NULL;

  free(inputs);
  free(resolved);
  free(outputs);
  free(targets);
  free(items);
  free(from);
  free(to);
  free(rewritten);
  free(call_src);
  free(tuple_src);
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
                         ? poly_uop_with_metadata_from(
                               ctx, u, POLY_OP_AFTER, cached->dtype, after_src, 2, poly_arg_none()
                           )
                         : NULL;
    if (!after || !poly_transform_to_call_append_store(tctx, after)) {
      tctx->failed = true;
      return NULL;
    }
    *out_replacement = cached;
    return after;
  }

  PolyUOp *buf = poly_transform_to_call_empty_buffer_like(ctx, tctx, u->dtype, shape, u);
  PolyUOp *replacement = poly_transform_to_call_rebuild_view(ctx, buf, u, NULL);
  PolyUOp *store = replacement ? poly_store_val(ctx, replacement, u->src[0]) : NULL;
  PolyUOp *after_src[2] = {replacement, store};
  PolyUOp *after = (replacement && store) ? poly_uop_with_metadata_from(
                                                ctx, u, POLY_OP_AFTER, replacement->dtype,
                                                after_src, 2, poly_arg_none()
                                            )
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

static PolyUOp *poly_transform_to_call_contiguous_mops_to_view(PolyCtx *ctx, PolyUOp *contiguous) {
  if (!ctx || !contiguous || contiguous->op != POLY_OP_CONTIGUOUS || contiguous->n_src != 1 ||
      !poly_opset_has(POLY_GROUP_MOVEMENT, contiguous->src[0]->op))
    return NULL;

  PolyShape shape = poly_uop_max_shape_cached(ctx, contiguous);
  /* Pinned contiguous_mops_to_view accepts only static movement graphs whose
   * flattened index is one contiguous range (callify.py:59-89).  Polygrad's
   * existing proof returns that same base/range for realized storage. */
  if (shape.ndim < 0 || !poly_transform_to_call_is_static_max_shape(ctx, contiguous, shape))
    return NULL;

  PolyUOp *view = make_buffer_view(ctx, contiguous->src[0]);
  PolyUOp *shaped =
      view ? poly_transform_to_call_rebuild_view(ctx, view, contiguous->src[0], NULL) : NULL;
  return shaped && poly_uop_buffer(ctx, shaped) ? shaped : NULL;
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

  bool opaque_program = u->op == POLY_OP_PROGRAM || u->op == POLY_OP_LINEAR ||
                        u->op == POLY_OP_SOURCE || u->op == POLY_OP_BINARY;
  /* AFTER returns a storage identity, but its STORE values are still part of
   * the shared tensor graph that pinned callify rewrites before collecting
   * effects. Treat concrete buffers/views as leaves; keep walking through the
   * immutable AFTER effect so a shared creation COPY is rewritten everywhere. */
  if ((poly_uop_has_buffer_identity(u) && u->op != POLY_OP_AFTER) || u->n_src == 0 ||
      opaque_program) {
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
  for (int i = 0; i < first_src; i++)
    new_src[i] = u->src[i];
  for (int i = first_src; i < u->n_src; i++) {
    new_src[i] =
        poly_transform_to_call_rewrite_nested_contiguous(ctx, u->src[i], outer, tctx, memo);
    if (!new_src[i]) {
      if (new_src != src_buf) free(new_src);
      return NULL;
    }
    /* RewriteContext.unified_rewrite retains completed replacement nodes as
     * well as original-to-result links. Generated parents must not re-enter
     * an already materialized COPY/AFTER subtree and allocate it again. */
    poly_map_set(memo, poly_ptr_hash(new_src[i]), new_src[i], new_src[i], poly_ptr_eq);
    if (new_src[i] != u->src[i]) changed = true;
  }

  PolyUOp *ret = changed ? poly_rebuild_with_sources(ctx, u, new_src) : u;
  if (new_src != src_buf) free(new_src);
  if (!ret) return NULL;

  /* Pinned pm_early_transform_tensor_graph transforms precompiled FUNCTIONs
   * before resolving their enclosing GETTUPLE(TUPLE(...)) selectors
   * (tinygrad/callify.py:146-151).  CALL bodies stay opaque to this outer
   * traversal; they are scheduled independently downstream. */
  if (ret->op == POLY_OP_FUNCTION && ret->arg.kind == POLY_ARG_CALL_INFO && ret->arg.call_info &&
      ret->arg.call_info->precompile) {
    PolyUOp *precompiled = poly_transform_precompiled_call(ctx, tctx, ret);
    if (!precompiled) {
      tctx->failed = true;
      return NULL;
    }
    poly_map_set(memo, poly_ptr_hash(u), u, precompiled, poly_ptr_eq);
    return precompiled;
  }
  if (ret->op == POLY_OP_GETTUPLE && ret->n_src == 1 && ret->src[0] &&
      ret->src[0]->op == POLY_OP_TUPLE && ret->arg.kind == POLY_ARG_INT && ret->arg.i >= 0 &&
      ret->arg.i < ret->src[0]->n_src) {
    PolyUOp *selected = ret->src[0]->src[ret->arg.i];
    poly_map_set(memo, poly_ptr_hash(u), u, selected, poly_ptr_eq);
    return selected;
  }

  /* Pinned pm_early_transform_tensor_graph runs contiguous_mops_to_view
   * before replace_contig_with_store_after (callify.py:152-164).  Returning
   * the storage view here removes the CONTIGUOUS through ordinary buffer
   * identity instead of allocating an intermediate materialization. */
  PolyUOp *movement_view = poly_transform_to_call_contiguous_mops_to_view(ctx, ret);
  if (movement_view) {
    poly_map_set(memo, poly_ptr_hash(u), u, movement_view, poly_ptr_eq);
    return movement_view;
  }

  /* Tinygrad 2026-08-22/a9069c177a9d callify keeps a contiguous movement view
   * as the COPY argument and attaches Buffer.view storage outside IR. */
  if (ret->op == POLY_OP_COPY) {
    PolyUOp *view_rewrite = poly_transform_to_call_materialize_view_copy(ctx, ret, tctx);
    if (view_rewrite) {
      poly_map_set(memo, poly_ptr_hash(u), u, view_rewrite, poly_ptr_eq);
      return view_rewrite;
    }
    if (tctx->failed) return NULL;
  }

  /* Pinned callify tags COPYs from creation devices and rewrites the COPY
   * itself to AFTER(buffer, STORE(buffer, COPY)) before rewriting its parent.
   * Match callify.py:19-25 by reading COPY.src[0].device through movement
   * nodes. Requiring buffer identity here misses
   * COPY(RESHAPE(SHRINK(BUFFER@DISK)), arg=device), leaking placement into ordinary
   * scalar codegen. HOST/DISK are Polygrad's creation-device counterparts to
   * tinygrad's PYTHON/NPY/DISK/TINYFS set. */
  if (ret->op == POLY_OP_COPY && ret->n_src >= 1) {
    PolyDevice source_device = poly_uop_device(ret->src[0]);
    PolyDevice copy_device = poly_uop_device(ret);
    bool from_creation = source_device == POLY_DEVICE_HOST || source_device == POLY_DEVICE_DISK;
    if (from_creation && copy_device != POLY_DEVICE_AUTO && copy_device != POLY_DEVICE_HOST &&
        copy_device != POLY_DEVICE_DISK) {
      PolyUOp *untagged_ret = poly_callify_tag_ids(ret, NULL, NULL)
                                  ? poly_callify_rebuild_without_tag(ctx, ret, NULL)
                                  : ret;
      PolyUOp *contiguous_src[1] = {untagged_ret};
      PolyUOp *contiguous = untagged_ret ? poly_uop_with_metadata_from(
                                               ctx, ret, POLY_OP_CONTIGUOUS, ret->dtype,
                                               contiguous_src, 1, poly_arg_none()
                                           )
                                         : NULL;
      PolyUOp *replacement = NULL;
      PolyUOp *executable =
          contiguous
              ? poly_transform_to_call_materialize_contiguous(ctx, contiguous, tctx, &replacement)
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
  if (!poly_uop_is_bound_var(ret) && ret->n_src == 2 &&
      poly_transform_to_call_after_store_assign(ret, &assign_target, &assign_store) &&
      ret->src[1] == assign_store && assign_store->n_src == 2 &&
      !poly_transform_to_call_after_result_buffer(ctx, ret)) {
    /* Pinned callify.py:54-57 carries the assignment AFTER's provenance tuple
     * onto this synthetic CONTIGUOUS. Distinct originals may share the exact
     * STORE value; preserving the tag keeps their rewrite occurrences
     * distinct until finalize consumes the surviving provenance. */
    PolyUOp *contiguous_src[1] = {assign_store->src[1]};
    PolyUOp *contiguous = poly_uop_with_metadata_from(
        ctx, ret, POLY_OP_CONTIGUOUS, assign_store->src[1]->dtype, contiguous_src, 1,
        poly_arg_none()
    );
    PolyUOp *replacement = NULL;
    PolyUOp *executable =
        contiguous
            ? poly_transform_to_call_materialize_contiguous(ctx, contiguous, tctx, &replacement)
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
  if (ret->op == POLY_OP_AFTER && !poly_uop_is_bound_var(ret)) {
    PolyUOp *result = poly_transform_to_call_after_result_buffer(ctx, ret);
    if (result && !poly_transform_to_call_cache_replacement(tctx, u, result)) {
      tctx->failed = true;
      return NULL;
    }
    /* Current finalize_after records the whole immutable AFTER occurrence in
     * AllocCtx.assigns (tensor.py:178-192). CALL/END/STORE dependencies stay
     * nested until get_kernel_graph; callify never schedules them itself. */
    if (!poly_transform_to_call_append_store(tctx, ret)) {
      tctx->failed = true;
      return NULL;
    }
  }

  /* Pinned add_tags tags every requested base during the one shared
   * bottom-up rewrite. Its early transform then inserts CONTIGUOUS and lowers
   * it to AFTER(buffer, STORE(buffer, value)) before rebuilding any descendant
   * (tensor.py:add_tags,pm_early_transform_tensor_graph). Use the same pass-local
   * AllocCtx.bases representation; the resulting executable UOps are identical. */
  bool tagged_value = poly_callify_tag_ids(ret, NULL, NULL);
  if (tagged_value && ret->op != POLY_OP_CONTIGUOUS && ret->op != POLY_OP_AFTER &&
      ret->op != POLY_OP_STORE && !poly_uop_has_buffer_identity(ret)) {
    PolyUOp *untagged_ret = poly_callify_tag_ids(ret, NULL, NULL)
                                ? poly_callify_rebuild_without_tag(ctx, ret, NULL)
                                : ret;
    PolyUOp *contiguous_src[1] = {untagged_ret};
    PolyUOp *contiguous = untagged_ret ? poly_uop_with_metadata_from(
                                             ctx, ret, POLY_OP_CONTIGUOUS, ret->dtype,
                                             contiguous_src, 1, poly_arg_none()
                                         )
                                       : NULL;
    /* Pinned replace_contig_with_store_after leaves DISK/TINYFS CONTIGUOUS
     * unmaterialized (callify.py:44-53). The underlying allocator exposes a
     * zero-copy Buffer.view, so allocating and scalar-bitcasting into a fresh
     * buffer would be both slower and wrong for unequal item sizes. */
    if (contiguous && poly_uop_device(contiguous) == POLY_DEVICE_DISK) {
      poly_map_set(memo, poly_ptr_hash(u), u, contiguous, poly_ptr_eq);
      return contiguous;
    }
    /* Run the generated CONTIGUOUS through the same early rules. In
     * particular, marker removal can expose an existing AFTER: its tag must
     * merge there instead of allocating a second output buffer. */
    PolyUOp *executable =
        contiguous
            ? poly_transform_to_call_rewrite_nested_contiguous(ctx, contiguous, outer, tctx, memo)
            : NULL;
    PolyUOp *replacement =
        executable ? poly_transform_to_call_cached_replacement(tctx, contiguous) : NULL;
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
    /* replace_contig_with_store_after cannot allocate virtual values. */
    if (!poly_callify_can_store(ctx, tctx, ret)) {
      poly_map_set(memo, poly_ptr_hash(u), u, ret, poly_ptr_eq);
      return ret;
    }
    /* Same pinned DISK/TINYFS exception for an explicit CONTIGUOUS already in
     * the graph. Strip callify's pass-local tag, but keep the operation and do
     * not publish a realized replacement. */
    if (poly_uop_device(ret) == POLY_DEVICE_DISK) {
      PolyUOp *unmaterialized = poly_callify_tag_ids(ret, NULL, NULL)
                                    ? poly_callify_rebuild_without_tag(ctx, ret, NULL)
                                    : ret;
      if (!unmaterialized) {
        tctx->failed = true;
        return NULL;
      }
      poly_map_set(memo, poly_ptr_hash(u), u, unmaterialized, poly_ptr_eq);
      return unmaterialized;
    }
    /* Pinned pm_early_transform_tensor_graph removes an extra CONTIGUOUS on
     * AFTER when the AFTER target already has buffer identity. Preserve the
     * AFTER in the executable graph so its producer remains a dependency, but
     * map the original CONTIGUOUS to the producer's final storage for live
     * tensors. This is what keeps custom producer -> CONTIGUOUS -> COPY at two
     * calls instead of inserting an unnecessary materialization kernel. */
    PolyUOp *after = ret->src[0];
    if (after && after->op == POLY_OP_AFTER && after->n_src >= 1 &&
        poly_uop_has_buffer_identity(after->src[0])) {
      PolyUOp *merged_after = NULL;
      if (poly_callify_tag_ids(after, NULL, NULL) && poly_callify_tag_ids(ret, NULL, NULL))
        merged_after = poly_callify_merge_tagged_uops(ctx, after, NULL, after, ret);
      else if (poly_callify_tag_ids(ret, NULL, NULL))
        merged_after = poly_uop_with_metadata_from(
            ctx, ret, POLY_OP_AFTER, after->dtype, after->src, after->n_src, after->arg
        );
      if (merged_after) after = merged_after;
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
      executable = poly_transform_to_call_materialize_contiguous(ctx, ret, tctx, &replacement);
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

  /* Final pm_early_transform_tensor_graph rule (tensor.py:174): these are
   * autograd markers, not executable storage or computation boundaries. */
  if (ret->n_src == 1 && (ret->op == POLY_OP_DETACH || ret->op == POLY_OP_CONTIGUOUS_BACKWARD))
    ret = ret->src[0];
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
  if ((out_map_orig || out_map_repl || out_map_n) && (!out_map_orig || !out_map_repl || !out_map_n))
    return NULL;
  if (n == 0) return NULL;
  for (int i = 0; i < n; i++)
    out_uops[i] = NULL;

  PolyUOp **stores = calloc((size_t)(n * 4), sizeof(PolyUOp *));
  if (!stores) return NULL;
  PolyTransformToCallCtx tctx = {
      .stores = stores,
      .n_stores = 0,
      .stores_cap = n * 4,
      .n_cached = 0,
      .requested_bases = poly_map_new((size_t)n * 2 + 16),
      .original_uops = poly_map_new(256),
      .device_memo = poly_map_new(256),
      .axis_memo = poly_map_new(256),
  };
  if (!tctx.requested_bases || !tctx.original_uops || !tctx.device_memo || !tctx.axis_memo)
    return poly_transform_to_call_fail(&tctx, out_uops, n);
  for (int i = 0; i < n; i++) {
    PolyUOp *base = poly_uop_base(uops[i]);
    if (!poly_callify_can_store(ctx, &tctx, base) || base->op == POLY_OP_AFTER ||
        poly_uop_has_buffer_identity(base))
      continue;
    poly_map_set(tctx.requested_bases, poly_ptr_hash(base), base, base, poly_ptr_eq);
  }

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
  big_sink = poly_transform_to_call_add_tags(ctx, &tctx, big_sink, tctx.original_uops);
  if (!big_sink) {
    poly_map_destroy(shared_rewrite_memo);
    return poly_transform_to_call_fail(&tctx, out_uops, n);
  }
  big_sink = poly_transform_to_call_rewrite_nested_contiguous(
      ctx, big_sink, NULL, &tctx, shared_rewrite_memo
  );
  if (!big_sink || big_sink->op != POLY_OP_SINK || big_sink->n_src != n) {
    poly_map_destroy(shared_rewrite_memo);
    return poly_transform_to_call_fail(&tctx, out_uops, n);
  }
  poly_map_destroy(shared_rewrite_memo);

  PolyMap *finalize_memo = poly_map_new(256);
  if (!finalize_memo) return poly_transform_to_call_fail(&tctx, out_uops, n);
  big_sink = poly_transform_to_call_finalize_tags(ctx, &tctx, big_sink, finalize_memo);
  if (!big_sink || big_sink->op != POLY_OP_SINK || big_sink->n_src != n) {
    poly_map_destroy(finalize_memo);
    return poly_transform_to_call_fail(&tctx, out_uops, n);
  }

  /* Current finalize_after collects only effects reachable from the rewritten
   * SINK (tinygrad/tensor.py:178-240). */
  int write_store = 0;
  for (int i = 0; i < tctx.n_stores; i++) {
    PolyUOp *mapped =
        poly_map_get(finalize_memo, poly_ptr_hash(tctx.stores[i]), tctx.stores[i], poly_ptr_eq);
    if (!mapped) continue;
    tctx.stores[write_store++] = mapped;
  }
  tctx.n_stores = write_store;
  poly_map_destroy(finalize_memo);
  tctx.publication_phase = true;

  for (int i = 0; i < n; i++) {
    PolyUOp *u = big_sink->src[i];
    if (!u) {
      out_uops[i] = NULL;
      continue;
    }
    if (!poly_callify_can_store(ctx, &tctx, u)) {
      out_uops[i] = u;
      continue;
    }
    if (poly_uop_has_buffer_identity(u)) {
      out_uops[i] = u;
      continue;
    }

    if (uops[i]->op == POLY_OP_COPY && poly_uop_device(uops[i]) == POLY_DEVICE_DISK) {
      out_uops[i] = poly_transform_to_call_cached_replacement(&tctx, uops[i]);
      if (!out_uops[i]) return poly_transform_to_call_fail(&tctx, out_uops, n);
      continue;
    }

    /* Pinned callify leaves a requested DISK/TINYFS CONTIGUOUS with no
     * assignment and returns an empty CALL (callify.py:44-53,204-222). Keep
     * the caller's original typed view root; its runtime Buffer.view is
     * resolved lazily by poly_uop_buffer. */
    PolyUOp *disk_contiguous = u;
    while (disk_contiguous && disk_contiguous->n_src >= 1 &&
           poly_opset_has(POLY_GROUP_MOVEMENT, disk_contiguous->op))
      disk_contiguous = disk_contiguous->src[0];
    if (disk_contiguous && disk_contiguous->op == POLY_OP_CONTIGUOUS &&
        disk_contiguous->n_src == 1 && poly_uop_device(disk_contiguous) == POLY_DEVICE_DISK) {
      out_uops[i] = uops[i];
      continue;
    }

    /* tinygrad tensor.transform_to_call does not tag a requested movement
     * chain whose base already has buffer identity. Tensor.realize is
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

    PolyTransformViewStack target_views = {0};
    PolyUOp *target_root = poly_transform_to_call_root(u, &target_views);
    if (!target_root) {
      poly_transform_view_stack_free(&target_views);
      return poly_transform_to_call_fail(&tctx, out_uops, n);
    }

    PolyUOp *outer_contiguous = target_root->op == POLY_OP_CONTIGUOUS ? target_root : NULL;
    if (outer_contiguous) {
      PolyUOp *cached = poly_transform_to_call_cached_replacement(&tctx, outer_contiguous);
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
    int64_t output_numel = (output_shape.ndim >= 0) ? poly_shape_numel(output_shape) : -1;
    if (output_numel == 0) {
      /* tinygrad callify removes size-zero CONTIGUOUS work and retargets the
       * Tensor to an unallocated BUFFER. Keep the distinct placed identity and
       * any producer effects collected above, but do not invent a residency:
       * there is no CALL whose execution could allocate or write one. */
      PolyUOp *buf =
          poly_transform_to_call_empty_buffer_like(ctx, &tctx, u->dtype, output_shape, u);
      if (!buf) return poly_transform_to_call_fail(&tctx, out_uops, n);
      PolyUOp *result = (output_shape.ndim == 1 && output_shape.dims[0] == 0)
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
    PolyUOp *buf =
        poly_transform_to_call_empty_buffer_like(ctx, &tctx, u->dtype, root_shape, materialized);
    if (!buf) {
      fprintf(stderr, "poly_realize: output buffer creation failed\n");
      poly_transform_view_stack_free(&views);
      return poly_transform_to_call_fail(&tctx, out_uops, n);
    }

    /* Pinned callify tags every requested base, adds CONTIGUOUS, then lowers
     * it to AFTER(buffer, STORE(buffer, source)). The caller-visible physical
     * result remains the stripped buffer below. */
    PolyUOp *output_store = poly_store_val(ctx, buf, materialized);
    PolyUOp *after_src[2] = {buf, output_store};
    PolyUOp *output_effect =
        output_store ? poly_uop(ctx, POLY_OP_AFTER, buf->dtype, after_src, 2, poly_arg_none())
                     : NULL;
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

  /* Pinned add_tags tags each requested base and finalize_after returns a
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

  /* Current tinygrad Tensor.transform_to_call returns one outer CALL over all
   * immutable effects (tensor.py:224-242). Kernel splitting, dependency
   * ordering, held-buffer selection, and memory planning belong exclusively
   * to create_linear_with_vars; callify must not construct LINEAR itself. */
  if (tctx.n_stores > 0 && !poly_transform_to_call_finalize_after_order(ctx, &tctx))
    return poly_transform_to_call_fail(&tctx, out_uops, n);
  int n_effects = tctx.n_stores;
  PolyUOp **effects = n_effects > 0 ? malloc((size_t)n_effects * sizeof(*effects)) : NULL;
  if (n_effects > 0 && !effects) return poly_transform_to_call_fail(&tctx, out_uops, n);
  int effect = 0;
  for (int i = 0; i < tctx.n_stores; i++)
    effects[effect++] = tctx.stores[i];
  if (poly_debug_at_least(7)) {
    fprintf(stderr, "[polygrad:transform_to_call] stores=%d effects=", tctx.n_stores);
    for (int i = 0; i < n_effects; i++)
      fprintf(
          stderr, "%s%s%s", i ? "," : "", poly_op_name(effects[i]->op),
          effects[i]->op == POLY_OP_CALL && effects[i]->n_src ? poly_op_name(effects[i]->src[0]->op)
                                                              : ""
      );
    fputc('\n', stderr);
  }
  PolyUOp *sink_body = poly_sink_n(ctx, effects, n_effects);
  free(effects);
  PolyUOp *outer_call = sink_body ? poly_transform_to_call_wrap_call(ctx, sink_body) : NULL;
  if (!outer_call) return poly_transform_to_call_fail(&tctx, out_uops, n);
  poly_transform_to_call_take_replacements(&tctx, out_map_orig, out_map_repl, out_map_n);
  poly_transform_to_call_ctx_free(&tctx);
  return outer_call;
}

PolyUOp *poly_transform_to_call(PolyCtx *ctx, PolyUOp **uops, int n, PolyUOp **out_uops) {
  return poly_transform_to_call_with_map(ctx, uops, n, out_uops, NULL, NULL, NULL);
}

static bool poly_roots_explicit_devices_supported(PolyCtx *ctx, PolyUOp **roots, int n) {
  if (!ctx || !roots || n < 0) return false;
  for (int i = 0; i < n; i++)
    if (!poly_uop_explicit_devices_supported(ctx, roots[i])) return false;
  return true;
}

PolyUOp *poly_linear_effect_sink(
    PolyCtx *ctx,
    PolyUOp *sink,
    PolyVarBinding **var_bindings_out,
    int *n_var_bindings_out
) {
  if (!ctx || !sink || sink->op != POLY_OP_SINK) {
    fprintf(stderr, "poly_realize: expected effect SINK\n");
    return NULL;
  }
  if (!poly_uop_explicit_devices_supported(ctx, sink)) {
    fprintf(stderr, "poly_realize: unsupported explicit device identity\n");
    return NULL;
  }
  if (poly_tensor_root_has_unplaced_buffer(ctx, sink)) {
    fprintf(stderr, "poly_realize: physical effect root contains device-free BUFFER\n");
    return NULL;
  }

  /* Imported/Model graphs already own their output/effect storage, so they
   * skip tensor output allocation. They do not skip tinygrad's input-buffer
   * normalization: rangeify must see a shaped-PARAM function body, with exact
   * BUFFER/SHRINK/BITCAST storage occurrences as outer CALL arguments. */
  bool timing = poly_debug_at_least(2);
  double t0 = timing ? poly_now_ms() : 0.0;
  if (timing) {
    fprintf(stderr, "[polygrad:linear_effect] begin sink=%p n_src=%d\n", (void *)sink, sink->n_src);
    fflush(stderr);
  }
  PolyUOp *outer_call = poly_transform_to_call_wrap_call(ctx, sink);
  PolyUOp *linear =
      outer_call
          ? poly_create_linear_with_vars(ctx, outer_call, var_bindings_out, n_var_bindings_out)
          : NULL;
  if (!linear) fprintf(stderr, "poly_realize: scheduling failed\n");
  if (timing) {
    double t1 = poly_now_ms();
    fprintf(
        stderr, "[polygrad:linear_effect] done schedule=%.3fms calls=%d vars=%d\n", t1 - t0,
        linear ? linear->n_src : -1, linear ? *n_var_bindings_out : -1
    );
    fflush(stderr);
  }
  return linear;
}

/* Raw physical-UOp analogue of Tensor.linear_with_vars. Default Tensor
 * execution requires a complete eagerly constructed physical root; explicit
 * logical-to-physical compilation stays outside both scheduling entrypoints. */
PolyUOp *poly_linear_with_vars(
    PolyCtx *ctx,
    PolyUOp **uops,
    int n,
    PolyUOp **out_uops,
    PolyVarBinding **var_bindings_out,
    int *n_var_bindings_out
) {
  if (!ctx || !uops || !out_uops || !var_bindings_out || !n_var_bindings_out || n < 0) return NULL;
  if (n == 0) return NULL;
  if (!poly_roots_explicit_devices_supported(ctx, uops, n)) {
    fprintf(stderr, "poly_realize: unsupported explicit device identity\n");
    for (int i = 0; i < n; i++)
      out_uops[i] = NULL;
    return NULL;
  }
  for (int i = 0; i < n; i++) {
    /* Tensor.linear_with_vars rejects weak storage requests. Tensor.realize
     * filters virtual roots before reaching this scheduling boundary. */
    if (!uops[i] ||
        (poly_dtype_is_weak(uops[i]->dtype) && poly_uop_device_uop_cached(ctx, uops[i], NULL))) {
      fprintf(stderr, "poly_linear_with_vars: cannot schedule a deviceful weak dtype\n");
      for (int j = 0; j < n; j++)
        out_uops[j] = NULL;
      return NULL;
    }
    if (poly_tensor_root_has_unplaced_buffer(ctx, uops[i])) {
      fprintf(stderr, "poly_realize: physical root %d contains device-free BUFFER\n", i);
      for (int j = 0; j < n; j++)
        out_uops[j] = NULL;
      return NULL;
    }
  }

  PolyUOp *big_call = poly_transform_to_call(ctx, uops, n, out_uops);
  if (!big_call) goto fail;
  PolyUOp *linear =
      poly_create_linear_with_vars(ctx, big_call, var_bindings_out, n_var_bindings_out);
  if (linear) return linear;

fail:
  for (int i = 0; i < n; i++)
    out_uops[i] = NULL;
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
  PolyVarBinding *var_bindings = NULL;
  int n_var_bindings = 0;
  PolyUOp *linear = poly_linear_effect_sink(ctx, sink, &var_bindings, &n_var_bindings);
  double t_sched = timing ? poly_now_ms() : 0.0;
  if (!linear) return -1;
  if (timing) {
    fprintf(stderr, "[polygrad:realize_sink] run begin calls=%d\n", linear->n_src);
    fflush(stderr);
  }
  int ret = poly_run_linear(ctx, linear, var_bindings, n_var_bindings, NULL, 0, true, false, false);
  double t_run = timing ? poly_now_ms() : 0.0;
  free(var_bindings);
  if (timing) {
    fprintf(
        stderr, "[polygrad:realize_sink] schedule=%.3fms run=%.3fms total=%.3fms ret=%d\n",
        t_sched - t0, t_run - t_sched, t_run - t0, ret
    );
  }
  return ret;
}

/* Current Tinygrad Tensor realization always runs the returned LINEAR;
 * create_linear_with_vars already replaced captured work with LINEAR(). */
static int poly_realize_linear(
    PolyCtx *ctx,
    PolyUOp *linear,
    PolyVarBinding *var_bindings,
    int n_var_bindings
) {
  if (!ctx || !linear || linear->op != POLY_OP_LINEAR) return -1;
  return poly_run_linear(ctx, linear, var_bindings, n_var_bindings, NULL, 0, true, false, false);
}

bool poly_tensor_root_has_unplaced_buffer(PolyCtx *ctx, PolyUOp *root) {
  if (!ctx || !root) return true;
  PolyScratchMark scratch = poly_ctx_scratch_mark(ctx);
  int n = 0;
  /* CALL/FUNCTION bodies contain placeholders, not Tensor storage. Their
   * caller-visible arguments remain part of the traversal. */
  PolyUOp **topo = poly_toposort_ex_user_scratch(ctx, root, &n, NULL, NULL, false);
  if (!topo) {
    poly_ctx_scratch_rewind(ctx, scratch);
    return true;
  }
  bool unplaced = false;
  for (int i = 0; i < n; i++) {
    if (topo[i] && topo[i]->op == POLY_OP_BUFFER && !poly_uop_is_variable(topo[i]) &&
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
  PolyVarBinding *var_bindings = NULL;
  int n_var_bindings = 0;
  PolyUOp *linear = poly_linear_with_vars(ctx, uops, n, out_uops, &var_bindings, &n_var_bindings);
  int ret = linear ? poly_realize_linear(ctx, linear, var_bindings, n_var_bindings) : -1;
  free(var_bindings);
  return ret;
}

static int poly_realize_tensors_impl(
    PolyCtx *ctx,
    PolyTensor **inputs,
    int n,
    PolyTensor **outputs
) {
  if (!ctx || !inputs || !outputs || n < 0) return -1;
  if (n == 0) return 0;
  PolyTensor **pending_tensors = calloc((size_t)n, sizeof(PolyTensor *));
  PolyUOp **pending_roots = calloc((size_t)n, sizeof(PolyUOp *));
  PolyUOp **pending_out = calloc((size_t)n, sizeof(PolyUOp *));
  int *pending_indices = calloc((size_t)n, sizeof(int));
  PolyUOp **map_orig = NULL;
  PolyUOp **map_repl = NULL;
  int map_n = 0;
  PolyUOp *linear = NULL;
  PolyVarBinding *var_bindings = NULL;
  int n_var_bindings = 0;
  int rc = -1;
  int n_pending = 0;
  if (!pending_tensors || !pending_roots || !pending_out || !pending_indices) goto cleanup;

  for (int i = 0; i < n; i++) {
    outputs[i] = NULL;
    if (!inputs[i]) goto cleanup;
    if (!inputs[i]->uop_physical ||
        poly_tensor_root_has_unplaced_buffer(ctx, inputs[i]->uop_physical)) {
      fprintf(stderr, "poly_realize_tensors: tensor %d has no complete physical root\n", i);
      goto cleanup;
    }
  }

  for (int i = 0; i < n; i++) {
    PolyUOp *physical = inputs[i]->uop_physical;
    /* Pinned Tensor.realize skips virtual roots: no device OR weak dtype
     * (tensor.py:419-424; UOp.is_virtual). Ask for the DEVICE UOp rather than a
     * scalar backend enum: tuple devices are deviceful graphs even though
     * they deliberately have no single PolyDevice execution value. */
    if (poly_dtype_is_weak(physical->dtype) || !poly_uop_device_uop_cached(ctx, physical, NULL)) {
      outputs[i] = inputs[i];
      continue;
    }
    /* Pinned callify.py:60-95 keeps only a provable contiguous movement view
     * over already-valid storage as a zero-CALL Buffer.view. Do not use the
     * recursive UOp.buffer accessor as the proof: it also walks through
     * AFTER, whose STORE effects must first run through callify. */
    if (physical->op == POLY_OP_CONTIGUOUS && physical->n_src == 1) {
      PolyUOp *view_identity = NULL;
      PolyShape view_shape = {.ndim = -1};
      int64_t view_numel = -1;
      size_t view_byte_offset = 0;
      if (poly_uop_contiguous_view_info(
              ctx, physical->src[0], &view_identity, &view_shape, &view_numel, &view_byte_offset
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
      if (poly_tensor_set_physical(ctx, inputs[i], physical, inputs[i]->role, inputs[i]->device) !=
          0)
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
   * to live current Tensor.uop roots before create_linear_with_vars
   * (tensor.py:195-206). Polygrad's current parity surface is uop_physical. */
  PolyUOp *big_call = poly_transform_to_call_with_map(
      ctx, pending_roots, n_pending, pending_out, &map_orig, &map_repl, &map_n
  );
  if (!big_call) goto cleanup;
  if (map_n > 0 &&
      poly_tensor_apply_realize_map(ctx, map_orig, map_repl, map_n, POLY_DEVICE_AUTO) != 0)
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
            ctx, pending_tensors[pending], out, pending_tensors[pending]->role,
            pending_tensors[pending]->device
        ) != 0)
      goto cleanup;
  }

  linear = poly_create_linear_with_vars(ctx, big_call, &var_bindings, &n_var_bindings);
  if (!linear) goto cleanup;
  rc = poly_realize_linear(ctx, linear, var_bindings, n_var_bindings);
  if (rc != 0) goto cleanup;
  int n_materialized = 0;
  for (int pending = 0; pending < n_pending; pending++) {
    outputs[pending_indices[pending]] = pending_tensors[pending];
    /* Tinygrad 2026-08-22 leaves a zero-CALL movement root unchanged. The
     * approved Polygrad lifetime boundary retires only a result that callify
     * replaced with a materialized current resource. */
    if (pending_out[pending] != pending_roots[pending])
      pending_tensors[n_materialized++] = pending_tensors[pending];
  }
  if (poly_tensor_retire_logical_resources(ctx, pending_tensors, n_materialized) != 0) {
    rc = -1;
    goto cleanup;
  }

cleanup:
  free(var_bindings);
  free(map_repl);
  free(map_orig);
  free(pending_indices);
  free(pending_out);
  free(pending_roots);
  free(pending_tensors);
  /* Tensor realization is a context-thread safe point after current roots,
   * cache entries and JIT capture owners are published. Tinygrad's weak UOps
   * release the same unreachable producer/compiler rows without a stats call. */
  if (rc == 0 && poly_ctx_collect_at_safe_point(ctx) != 0) rc = -1;
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
