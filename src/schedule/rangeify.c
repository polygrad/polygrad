/* C implementation of tinygrad schedule/rangeify.py. */

#include "schedule/rangeify.h"
#include "schedule/allreduce.h"
#include "schedule/indexing.h"
#include "schedule/multi.h"
#include "ctx.h"
#include "device.h"
#include "codegen/simplify.h"
#include "uop/movement.h"
#include "uop/spec.h"
#include "tensor.h"
#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "frontend_internal.h"
#include "utils.h"

#ifndef POLY_RANGEIFY_DEBUG
#define POLY_RANGEIFY_DEBUG 0
#endif

#define RDBG(...)                                                                                  \
  do {                                                                                             \
    if (POLY_RANGEIFY_DEBUG) fprintf(stderr, __VA_ARGS__);                                         \
  } while (0)

#define POLY_RANGEIFY_RANGE_TAG 0x5247 /* 'RG': tinygrad RANGE.rtag(()) marker */
#define POLY_RANGEIFY_PARAM_TAG 0x5047 /* 'PG': tinygrad PARAM.rtag(()) marker */

static PolyUOp *rangeify_index_const(PolyCtx *ctx, int64_t value) {
  return poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(value));
}

static PolyUOp *rmap_get(PolyMap *map, PolyUOp *key) {
  return poly_map_get(map, poly_ptr_hash(key), key, poly_ptr_eq);
}

static void rmap_set(PolyMap *map, PolyUOp *key, PolyUOp *value) {
  poly_map_set(map, poly_ptr_hash(key), key, value, poly_ptr_eq);
}

static PolyUOp *flatten_bufferize(PolyCtx *ctx, PolyUOp *stage) {
  if (!stage || stage->op != POLY_OP_STAGE || stage->n_src == 2) return NULL;
  const int ndim = stage->n_src - 1;
  if (ndim > POLY_MAX_DIMS) return NULL;

  PolyUOp *shape[POLY_MAX_DIMS];
  PolyUOp *flat_shape = rangeify_index_const(ctx, 1);
  for (int i = 0; i < ndim; i++) {
    shape[i] = poly_uop_shape_dim(ctx, stage, i);
    flat_shape = shape[i] ? poly_binop(ctx, POLY_OP_MUL, flat_shape, shape[i]) : NULL;
    if (!shape[i] || !flat_shape) return NULL;
  }
  flat_shape = poly_graph_rewrite(ctx, flat_shape, poly_symbolic());
  PolyUOp *flat_idx[POLY_MAX_DIMS];
  int n_flat = 0;
  if (!flat_shape || !poly_apply_reshape(
                         ctx, &flat_shape, 1, shape, ndim, stage->src + 1, ndim,
                         flat_idx, &n_flat
                     ) ||
      n_flat != 1)
    return NULL;

  PolyUOp *stage_src[2] = {stage->src[0], flat_idx[0]};
  PolyUOp *flat_stage = poly_uop_tagged_arg(
      ctx, POLY_OP_STAGE, stage->dtype, stage_src, 2, stage->arg, stage->tag, stage->tag_arg
  );
  PolyUOp *ret = flat_stage ? poly_reshape_uop(ctx, flat_stage, shape, ndim) : NULL;
  if (!ret) return NULL;

  bool symbolic = false;
  PolyUOp *starts[POLY_MAX_DIMS], *sizes[POLY_MAX_DIMS];
  for (int i = 0; i < ndim; i++) {
    PolyUOp *r = stage->src[i + 1];
    starts[i] = rangeify_index_const(ctx, 0);
    sizes[i] = r->op == POLY_OP_CONST ? rangeify_index_const(ctx, 1)
                                      : (r->n_src > 0 ? r->src[0] : NULL);
    if (!sizes[i]) return NULL;
    symbolic |= r->op == POLY_OP_RANGE && r->n_src > 0 && r->src[0]->op != POLY_OP_CONST;
  }
  return symbolic ? poly_shrink_uop(ctx, ret, starts, sizes, ndim) : ret;
}

/* Current Tinygrad schedule/rangeify.py:392-424. */
static PolyUOp *bufferize_to_store(PolyCtx *ctx, int *slot_counter, PolyUOp *stage) {
  if (!stage || stage->op != POLY_OP_STAGE || stage->n_src != 2 ||
      stage->arg.kind != POLY_ARG_BUFFERIZE_OPTS ||
      poly_bufferize_arg_addrspace(stage->arg) != POLY_ADDR_GLOBAL)
    return NULL;
  PolyUOp *value = stage->src[0], *flat_idx = stage->src[1];
  PolyUOp *ranges[POLY_MAX_DIMS];
  int n_ranges = poly_uop_ranges(ctx, flat_idx, ranges, POLY_MAX_DIMS);
  for (int i = 0; i < n_ranges - 1; i++)
    for (int j = i + 1; j < n_ranges; j++)
      if (poly_range_axis_id(ranges[i]->arg) > poly_range_axis_id(ranges[j]->arg)) {
        PolyUOp *tmp = ranges[i]; ranges[i] = ranges[j]; ranges[j] = tmp;
      }

  if (value->op == POLY_OP_AFTER) {
    PolyUOp *buf = poly_uop_buf_uop(ctx, value);
    if (!buf) return NULL;
    PolyUOp **after_src = malloc((size_t)value->n_src * sizeof(*after_src));
    if (!after_src) return NULL;
    int n_after = 1;
    after_src[0] = buf;
    for (int i = 1; i < value->n_src; i++) {
      PolyUOp *store = value->src[i];
      if (!store || store->op != POLY_OP_STORE || store->n_src != 2 ||
          store->src[0]->op != POLY_OP_INDEX)
        continue;
      PolyUOp *target = store->src[0];
      if (target->src[0]->op == POLY_OP_STAGE && target->src[0]->n_src > 0 &&
          target->src[0]->src[0]->op == POLY_OP_INDEX)
        target = target->src[0]->src[0];
      if (store->src[1] == target) continue;

      PolyUOp *merged[2 * POLY_MAX_DIMS];
      int n_merged = poly_uop_ranges(ctx, target, merged, POLY_MAX_DIMS);
      for (int j = 0; j < n_ranges; j++) {
        bool seen = false;
        for (int k = 0; k < n_merged; k++) seen |= merged[k] == ranges[j];
        if (!seen && n_merged < 2 * POLY_MAX_DIMS) merged[n_merged++] = ranges[j];
      }
      for (int j = 0; j < n_merged - 1; j++)
        for (int k = j + 1; k < n_merged; k++)
          if (poly_range_axis_id(merged[j]->arg) > poly_range_axis_id(merged[k]->arg)) {
            PolyUOp *tmp = merged[j]; merged[j] = merged[k]; merged[k] = tmp;
          }
      PolyUOp *ended = poly_store_val(ctx, target, store->src[1]);
      if (!ended) { free(after_src); return NULL; }
      if (n_merged) {
        PolyUOp *end_src[1 + 2 * POLY_MAX_DIMS];
        end_src[0] = ended;
        for (int j = 0; j < n_merged; j++) end_src[j + 1] = merged[j];
        ended = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, n_merged + 1, poly_arg_none());
      }
      after_src[n_after++] = ended;
    }
    PolyUOp *ret = n_after == 1 ? buf : poly_uop(
        ctx, POLY_OP_AFTER, buf->dtype, after_src, n_after, poly_arg_none()
    );
    free(after_src);
    return ret;
  }

  PolyDType storage_dtype = poly_dtype_strong(stage->dtype);
  int64_t size = poly_uop_max_numel(ctx, stage);
  if (size <= 0) return NULL;
  PolyUOp *device = poly_bufferize_arg_device_is_tuple(stage->arg)
                        ? poly_device_uop_from_names(
                              ctx, poly_bufferize_arg_devices(stage->arg),
                              poly_bufferize_arg_n_devices(stage->arg)
                          )
                        : poly_device_uop_from_name(
                              ctx, poly_bufferize_arg_device(stage->arg)
                          );
  PolyUOp *buf = device
                     ? poly_uop_new_buffer(
                           ctx, device, size, storage_dtype, (*slot_counter)++
                       )
                     : NULL;
  PolyUOp *idx = buf ? poly_uop2(ctx, POLY_OP_INDEX, storage_dtype, buf, flat_idx, poly_arg_none()) : NULL;
  PolyUOp *stored = poly_dtype_eq(value->dtype, storage_dtype)
                        ? value : poly_uop1(ctx, POLY_OP_CAST, storage_dtype, value, poly_arg_none());
  PolyUOp *store = idx ? poly_store_val(ctx, idx, stored) : NULL;
  if (!store) return NULL;
  if (n_ranges) {
    PolyUOp *end_src[1 + POLY_MAX_DIMS];
    end_src[0] = store;
    for (int i = 0; i < n_ranges; i++) end_src[i + 1] = ranges[i];
    store = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, n_ranges + 1, poly_arg_none());
  }
  PolyUOp *after = poly_uop2(ctx, POLY_OP_AFTER, storage_dtype, buf, store, poly_arg_none());
  return poly_dtype_eq(storage_dtype, stage->dtype)
             ? after : poly_uop1(ctx, POLY_OP_CAST, stage->dtype, after, poly_arg_none());
}

static PolyUOp *rangeify_clone_preserving_metadata(
    PolyCtx *ctx,
    PolyUOp *u,
    PolyUOp **src,
    int n_src
);

/* Pinned `is_noop_after_dep` recognizes only a zero-source NOOP, optionally
 * wrapped by END nodes. Value-carrying NOOP(source) remains a real source
 * boundary (schedule/rangeify.py:459-461). */
static bool rangeify_is_noop_after_dep(PolyUOp *u) {
  if (!u) return false;
  if (u->op == POLY_OP_NOOP) return u->n_src == 0;
  return u->op == POLY_OP_END && u->n_src >= 1 && rangeify_is_noop_after_dep(u->src[0]);
}

static PolyUOp *flatten_bufferize_match(
    PolyCtx *ctx,
    PolyUOp *stage,
    const PolyBindings *bindings
) {
  (void)bindings;
  return flatten_bufferize(ctx, stage);
}

static PolyUOp *bufferize_to_store_match(
    PolyCtx *ctx,
    PolyUOp *stage,
    const PolyBindings *bindings
) {
  (void)bindings;
  int *slot_counter = poly_graph_rewrite_userctx();
  return slot_counter ? bufferize_to_store(ctx, slot_counter, stage) : NULL;
}

static PolyUOp *index_weak_buffer(
    PolyCtx *ctx,
    PolyUOp *index,
    const PolyBindings *bindings
) {
  (void)bindings;
  if (!index || index->op != POLY_OP_INDEX || index->n_src < 1 ||
      index->src[0]->op != POLY_OP_CAST || index->src[0]->n_src != 1 ||
      !poly_dtype_is_weak(index->src[0]->dtype))
    return NULL;
  PolyUOp **src = malloc((size_t)index->n_src * sizeof(*src));
  if (!src) return NULL;
  src[0] = index->src[0]->src[0];
  for (int i = 1; i < index->n_src; i++) src[i] = index->src[i];
  /* Tinygrad 2026-08-22/a9069c177a9d rangeify.py:449-451 uses
   * `replace(dtype=None, ...)`: INDEX re-derives the strong storage dtype. */
  PolyDType dtype = index->dtype;
  PolyUOp *unwrapped =
      poly_dtype_from_uop(
          index->op, src, index->n_src, index->arg, index->dtype, &dtype)
          ? (index->tag != 0 || index->tag_arg.kind != POLY_ARG_NONE)
                ? poly_uop_tagged_arg(
                      ctx, index->op, dtype, src, index->n_src, index->arg,
                      index->tag, index->tag_arg)
                : poly_uop(ctx, index->op, dtype, src, index->n_src, index->arg)
          : NULL;
  free(src);
  return unwrapped
             ? poly_uop1(ctx, POLY_OP_CAST, index->dtype, unwrapped, poly_arg_dtype(index->dtype))
             : NULL;
}

static PolyUOp *move_reshape_through_multi(
    PolyCtx *ctx,
    PolyUOp *multi,
    const PolyBindings *bindings
) {
  (void)bindings;
  if (!multi || (multi->op != POLY_OP_MSELECT && multi->op != POLY_OP_MSTACK) ||
      multi->n_src < 1)
    return NULL;
  PolyUOp **src = malloc((size_t)multi->n_src * sizeof(*src));
  if (!src) return NULL;
  bool ok = true;
  for (int i = 0; i < multi->n_src; i++) {
    if (!multi->src[i] || multi->src[i]->op != POLY_OP_RESHAPE ||
        multi->src[i]->n_src < 1) {
      ok = false;
      break;
    }
    src[i] = poly_uop_unsharded_base(multi->src[i]->src[0]);
  }
  PolyUOp *base = ok ? rangeify_clone_preserving_metadata(ctx, multi, src, multi->n_src) : NULL;
  free(src);
  PolyShape shape = poly_uop_max_shape_cached(ctx, multi);
  return base && shape.ndim >= 0 ? poly_reshape(ctx, base, shape.dims, shape.ndim) : NULL;
}

static PolyUOp *remove_call_reshapes(
    PolyCtx *ctx,
    PolyUOp *call,
    const PolyBindings *bindings
) {
  (void)bindings;
  if (!call || call->op != POLY_OP_CALL) return NULL;
  PolyUOp **src = malloc((size_t)call->n_src * sizeof(*src));
  if (!src) return NULL;
  bool changed = false;
  for (int i = 0; i < call->n_src; i++) {
    src[i] = call->src[i] && call->src[i]->op == POLY_OP_RESHAPE && call->src[i]->n_src > 0
                 ? call->src[i]->src[0]
                 : call->src[i];
    changed |= src[i] != call->src[i];
  }
  PolyUOp *ret = changed
                     ? rangeify_clone_preserving_metadata(ctx, call, src, call->n_src)
                     : NULL;
  free(src);
  return ret;
}

static PolyUOp *remove_invalid_store(
    PolyCtx *ctx,
    PolyUOp *store,
    const PolyBindings *bindings
) {
  (void)bindings;
  if (!store || store->op != POLY_OP_STORE || store->n_src != 2) return NULL;
  PolyUOp *value = store->src[1];
  bool invalid = value && value->op == POLY_OP_CONST && value->arg.kind == POLY_ARG_INVALID;
  invalid |= value && value->op == POLY_OP_CONTIGUOUS && value->n_src == 1 &&
             value->src[0]->op == POLY_OP_CONST &&
             value->src[0]->arg.kind == POLY_ARG_INVALID;
  return invalid ? poly_uop0(ctx, POLY_OP_NOOP, POLY_VOID, poly_arg_none()) : NULL;
}

static PolyUOp *remove_noop_afters(
    PolyCtx *ctx,
    PolyUOp *after,
    const PolyBindings *bindings
) {
  (void)bindings;
  if (!after || after->op != POLY_OP_AFTER || after->n_src < 1) return NULL;
  PolyUOp **src = malloc((size_t)after->n_src * sizeof(*src));
  if (!src) return NULL;
  int n_src = 1;
  src[0] = after->src[0];
  for (int i = 1; i < after->n_src; i++)
    if (!rangeify_is_noop_after_dep(after->src[i])) src[n_src++] = after->src[i];
  PolyUOp *ret = n_src == after->n_src
                     ? NULL
                     : n_src == 1
                           ? src[0]
                           : rangeify_clone_preserving_metadata(ctx, after, src, n_src);
  free(src);
  return ret;
}

PolyPatternMatcher *poly_pm_add_buffers(void) {
  static _Thread_local PolyPatternMatcher *pm = NULL;
  if (pm) return pm;
  PolyRule rules[] = {
      {poly_upat_allow_any_len(poly_upat_op(POLY_OP_STAGE, NULL, 0, "x")),
       flatten_bufferize_match},
      {poly_upat_allow_any_len(poly_upat_op(POLY_OP_STAGE, NULL, 0, "x")),
       bufferize_to_store_match},
      {poly_upat_allow_any_len(poly_upat_op(POLY_OP_INDEX, NULL, 0, "u")),
       index_weak_buffer},
      {poly_upat_allow_any_len(poly_upat_ops(
           poly_opset_add(poly_opset_add((PolyOpSet){{0, 0}}, POLY_OP_MSELECT), POLY_OP_MSTACK),
           NULL, 0, "m")),
       move_reshape_through_multi},
      {poly_upat_allow_any_len(poly_upat_op(POLY_OP_CALL, NULL, 0, "k")),
       remove_call_reshapes},
      {poly_upat_allow_any_len(poly_upat_op(POLY_OP_STORE, NULL, 0, "st")),
       remove_invalid_store},
      {poly_upat_allow_any_len(poly_upat_op(POLY_OP_AFTER, NULL, 0, "x")),
       remove_noop_afters},
  };
  PolyPatternMatcher *rules_pm =
      poly_pm_new(rules, (int)(sizeof(rules) / sizeof(rules[0])));
  PolyPatternMatcher *with_flatten = poly_pm_concat(poly_pm_mops(), rules_pm);
  pm = poly_pm_thread_cache(with_flatten);
  poly_pm_destroy(rules_pm);
  return pm;
}

/* Current Tinygrad schedule/rangeify.py:LocalAddBufferContext. */

typedef struct {
  PolyCtx *ctx;
  int dg;
  int range;
  PolyUOp **map_keys;
  PolyUOp **map_values;
  int map_count;
  int map_cap;
  bool failed;
} PolyLocalAddBufferContext;

/* C storage for current Tinygrad LocalAddBufferContext.map. */
static void local_add_buffer_map_setdefault(
    PolyLocalAddBufferContext *lctx,
    PolyUOp *key,
    PolyUOp *value
) {
  for (int i = 0; i < lctx->map_count; i++)
    if (lctx->map_keys[i] == key) return;
  if (lctx->map_count >= lctx->map_cap) {
    int cap = lctx->map_cap ? lctx->map_cap * 2 : 8;
    PolyUOp **keys = realloc(lctx->map_keys, (size_t)cap * sizeof(*keys));
    PolyUOp **values = realloc(lctx->map_values, (size_t)cap * sizeof(*values));
    assert(keys && values && "OOM: LocalAddBufferContext.map");
    lctx->map_keys = keys;
    lctx->map_values = values;
    lctx->map_cap = cap;
  }
  lctx->map_keys[lctx->map_count] = key;
  lctx->map_values[lctx->map_count++] = value;
}

static PolyUOp *debuf(
    PolyCtx *ctx,
    PolyLocalAddBufferContext *lctx,
    PolyUOp *root,
    PolyUOp *binding
) {
  if (!lctx || !root) return NULL;
  if (poly_uop_is_variable(root)) {
    return poly_uop(
        ctx, POLY_OP_PARAM, root->dtype, root->src, root->n_src, root->arg
    );
  }
  bool shaped_param = poly_uop_is_shaped_value_param(root);
  /* Tinygrad 2026-08-22/a9069c177a9d schedule/rangeify.py:497-503 debufs
   * BUFFER, MSTACK, MSELECT, and callified shaped PARAMs. Exact runtime views
   * are outer CALL arguments and have already become PARAMs in the body. */
  bool aggregate = root->op == POLY_OP_MSTACK || root->op == POLY_OP_MSELECT;
  if (root->op != POLY_OP_BUFFER && !aggregate &&
      !(shaped_param && root->tag == POLY_RANGEIFY_PARAM_TAG))
    return NULL;

  PolyDType scalar = root->dtype;
  PolyShape max_shape = poly_uop_max_shape_cached(ctx, root);
  int64_t max_numel = max_shape.ndim >= 0 ? poly_shape_numel(max_shape) : -1;
  if (max_numel <= 0 || max_shape.ndim > POLY_MAX_DIMS) return NULL;

  PolyAddrSpace addrspace = POLY_ADDR_GLOBAL;
  (void)poly_uop_addrspace(root, &addrspace);
  PolyUOp *device = poly_uop_device_uop_cached(ctx, root, NULL);
  PolyParamArg param_arg = {
      .slot = lctx->dg,
      .addrspace = addrspace,
      .device = device && device->arg.kind == POLY_ARG_STRING ? device->arg.str : NULL,
      .devices = device && device->arg.kind == POLY_ARG_STRING_TUPLE
                     ? device->arg.string_tuple.vals
                     : NULL,
      .n_devices = device && device->arg.kind == POLY_ARG_STRING_TUPLE
                       ? device->arg.string_tuple.n
                       : 0,
      .device_is_tuple = device && device->arg.kind == POLY_ARG_STRING_TUPLE,
  };
  PolyUOp *flat_shape = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(max_numel));
  PolyUOp *param = flat_shape
                       ? poly_uop1(ctx, POLY_OP_PARAM, scalar, flat_shape, poly_arg_param(&param_arg))
                       : NULL;
  if (!param) return NULL;

  PolyUOp *ret = poly_reshape(ctx, param, max_shape.dims, max_shape.ndim);
  if (!ret) return NULL;
  bool needs_shrink = false;
  PolyUOp *starts[POLY_MAX_DIMS], *sizes[POLY_MAX_DIMS];
  for (int i = 0; i < max_shape.ndim; i++) {
    starts[i] = rangeify_index_const(ctx, 0);
    sizes[i] = poly_uop_shape_dim(ctx, root, i);
    int64_t actual = -1;
    if (!starts[i] || !sizes[i]) return NULL;
    if (poly_uop_const_i64(sizes[i], &actual) != 0 || actual != max_shape.dims[i])
      needs_shrink = true;
  }
  if (needs_shrink) {
    ret = poly_shrink_uop(ctx, ret, starts, sizes, max_shape.ndim);
    if (!ret) return NULL;
  }
  local_add_buffer_map_setdefault(lctx, root, binding);
  lctx->dg++;
  return ret;
}

static PolyUOp *to_define_global_debuf(
    PolyCtx *ctx,
    PolyUOp *root,
    const PolyBindings *b
) {
  (void)b;
  PolyLocalAddBufferContext *lctx =
      (PolyLocalAddBufferContext *)poly_graph_rewrite_userctx();
  return debuf(ctx, lctx, root, root);
}

/* Current tinygrad schedule/rangeify.py:to_define_global removes the
 * call-argument slot from named symbolic PARAMs. codegen pm_number_params
 * assigns their final ABI slots after global buffers. */
static PolyUOp *reset_named_param_slot(
    PolyCtx *ctx,
    PolyUOp *root,
    const PolyBindings *b
) {
  (void)b;
  if (!poly_uop_is_alu_param(root) || !root->arg.param->name ||
      !root->arg.param->has_minmax || root->arg.param->slot == -1)
    return NULL;
  PolyParamArg arg = *root->arg.param;
  arg.slot = -1;
  return poly_uop_tagged_arg(
      ctx, root->op, root->dtype, root->src, root->n_src,
      poly_arg_param(&arg), root->tag, root->tag_arg
  );
}

static PolyUOp *index_alu_param(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)ctx;
  (void)b;
  if (!root || root->op != POLY_OP_INDEX || root->n_src != 1 || !root->src[0] ||
      root->src[0]->op != POLY_OP_PARAM || root->src[0]->arg.kind != POLY_ARG_PARAM ||
      !root->src[0]->arg.param || root->src[0]->arg.param->addrspace != POLY_ADDR_ALU)
    return NULL;
  return root->src[0];
}

static PolyUOp *strip_bound_variable(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)ctx;
  (void)b;
  return poly_uop_is_bound_var(root) ? root->src[0] : NULL;
}

static PolyUOp *handle_after(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  if (!root || root->op != POLY_OP_AFTER || root->n_src < 1) return NULL;
  PolyLocalAddBufferContext *lctx =
      (PolyLocalAddBufferContext *)poly_graph_rewrite_userctx();
  PolyUOp *buf = poly_uop_buf_uop(ctx, root);
  if (!lctx || !buf) return NULL;

  const PolyUOp *identity = poly_uop_get_buffer_identity(buf);
  if (identity && identity->arg.kind == POLY_ARG_PARAM && identity->arg.param &&
      identity->arg.param->addrspace == POLY_ADDR_LOCAL)
    return NULL;

  /* Current Tinygrad schedule/rangeify.py:handle_after records the producer
   * version as the CALL binding, then lets ordinary debuf number `buf`. */
  local_add_buffer_map_setdefault(lctx, buf, root);
  return buf;
}

static PolyUOp *remove_bufferize_device(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  if (!root || root->op != POLY_OP_STAGE || root->arg.kind != POLY_ARG_BUFFERIZE_OPTS) return NULL;
  PolyArg arg = poly_arg_bufferize_opts(
      NULL, root->arg.bufferize_opts.addrspace, root->arg.bufferize_opts.removable
  );
  if (poly_arg_eq(arg, root->arg)) return NULL;
  return poly_uop(ctx, root->op, root->dtype, root->src, root->n_src, arg);
}

static PolyUOp *renumber_range(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  PolyLocalAddBufferContext *lctx =
      (PolyLocalAddBufferContext *)poly_graph_rewrite_userctx();
  if (!lctx || !root || root->op != POLY_OP_RANGE || root->tag != POLY_RANGEIFY_RANGE_TAG)
    return NULL;

  PolyArg new_arg = poly_arg_range(lctx->range++, poly_range_axis_type(root->arg));
  return (root->n_src > 0) ? poly_uop1(ctx, POLY_OP_RANGE, root->dtype, root->src[0], new_arg)
                           : poly_uop0(ctx, POLY_OP_RANGE, root->dtype, new_arg);
}

static PolyUOp *get_contiguous(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)ctx;
  (void)b;
  if (!root || root->n_src < 1 || root->op != POLY_OP_CONTIGUOUS) return NULL;
  return root->src[0];
}

static PolyUOp *remove_noop(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)ctx;
  (void)b;
  if (!root || root->op != POLY_OP_NOOP || root->n_src < 1) return NULL;
  return root->src[0];
}

static bool find_bufs_gate(PolyUOp *u) {
  return u && u->op != POLY_OP_AFTER;
}

/* Current Tinygrad schedule/rangeify.py:find_bufs. */
bool poly_find_bufs(PolyCtx *ctx, PolyUOp *store) {
  if (!ctx || !store || store->op != POLY_OP_STORE) return false;
  int n_topo = 0;
  /* tinygrad@2026-08-22/a9069c177a9d schedule/rangeify.py:501-505 uses
   * UOp.toposort's default enter_calls=False. Each opaque CALL body is a
   * separate kernel and cannot participate in this STORE's index-cycle test. */
  PolyUOp **topo = poly_toposort_ex_alloc(
      ctx, store, &n_topo, find_bufs_gate, false);
  PolyMap *read_from = poly_map_new(n_topo > 16 ? (uint32_t)n_topo : 16);
  if (!topo || !read_from) {
    poly_toposort_free(topo);
    if (read_from) poly_map_destroy(read_from);
    return false;
  }

  bool valid = true;
  for (int i = 0; i < n_topo && valid; i++) {
    PolyUOp *idx = topo[i];
    if (!idx || idx->op != POLY_OP_INDEX || idx->n_src < 1 || !idx->src[0]) continue;
    PolyUOp *buf = poly_uop_buf_uop(ctx, idx);
    if (!buf || (buf->op != POLY_OP_BUFFER && buf->op != POLY_OP_PARAM)) continue;
    uintptr_t mode = (uintptr_t)idx->src[0]->op + 1;
    void *seen = poly_map_get(read_from, poly_ptr_hash(buf), buf, poly_ptr_eq);
    if (seen && (uintptr_t)seen != mode) {
      fprintf(stderr, "polygrad: cycle detected while indexing %p\n", (void *)buf);
      valid = false;
    } else if (!seen) {
      poly_map_set(
          read_from, poly_ptr_hash(buf), buf, (void *)mode, poly_ptr_eq);
    }
  }
  poly_map_destroy(read_from);
  poly_toposort_free(topo);
  return valid;
}

static PolyUOp *validate_find_bufs(
    PolyCtx *ctx,
    PolyUOp *root,
    const PolyBindings *b
) {
  (void)b;
  PolyLocalAddBufferContext *lctx =
      (PolyLocalAddBufferContext *)poly_graph_rewrite_userctx();
  if (!lctx || !poly_find_bufs(ctx, root)) {
    if (lctx) lctx->failed = true;
  }
  return NULL;
}

static PolyPatternMatcher *poly_to_define_global(void) {
  static _Thread_local PolyPatternMatcher *pm = NULL;
  if (pm) return pm;
  PolyOpSet storage_set = poly_opset_add((PolyOpSet){{0, 0}}, POLY_OP_BUFFER);
  storage_set = poly_opset_add(storage_set, POLY_OP_MSTACK);
  storage_set = poly_opset_add(storage_set, POLY_OP_MSELECT);
  PolyRule rules[] = {
      {poly_upat_op(POLY_OP_STORE, NULL, 0, "x"), validate_find_bufs},
      {poly_upat_ops(storage_set, NULL, 0, "buf"), to_define_global_debuf},
      {poly_upat_op(POLY_OP_PARAM, NULL, 0, "v"), reset_named_param_slot},
      {poly_upat_op(POLY_OP_PARAM, NULL, 0, "buf"), to_define_global_debuf},
      {poly_upat_op(POLY_OP_INDEX, NULL, 0, "idx"), index_alu_param},
      {poly_upat_op(POLY_OP_AFTER, NULL, 0, "bound"), strip_bound_variable},
      {poly_upat_op(POLY_OP_AFTER, NULL, 0, "after"), handle_after},
      {poly_upat_op(POLY_OP_STAGE, NULL, 0, "b"), remove_bufferize_device},
      {poly_upat_op(POLY_OP_RANGE, NULL, 0, "r"), renumber_range},
  };
  pm = poly_pm_thread_cache(poly_pm_new(rules, (int)(sizeof(rules) / sizeof(rules[0]))));
  return pm;
}

static PolyPatternMatcher *poly_rangeify_codegen(void) {
  static _Thread_local PolyPatternMatcher *pm = NULL;
  if (pm) return pm;
  PolyRule rules[] = {
      {poly_upat_op(POLY_OP_CONTIGUOUS, NULL, 0, "x"), get_contiguous},
      {poly_upat_op(POLY_OP_NOOP, NULL, 0, "x"), remove_noop},
  };
  pm = poly_pm_thread_cache(poly_pm_new(rules, (int)(sizeof(rules) / sizeof(rules[0]))));
  return pm;
}

static PolyPatternMatcher *poly_kernel_split(void) {
  static _Thread_local PolyPatternMatcher *pm = NULL;
  if (pm) return pm;
  PolyPatternMatcher *to_define_and_flatten =
      poly_pm_concat(poly_to_define_global(), poly_pm_flatten_range());
  PolyPatternMatcher *all =
      poly_pm_concat(to_define_and_flatten, poly_rangeify_codegen());
  poly_pm_destroy(to_define_and_flatten);
  pm = poly_pm_thread_cache(all);
  return pm;
}

/* C traversal for Tinygrad's `graph_rewrite(..., name="kernel split")`. */
static PolyUOp *rewrite_kernel_split(PolyLocalAddBufferContext *lctx, PolyUOp *u) {
  /* tinygrad@2026-08-22/a9069c177a9d schedule/rangeify.py:546-563 keeps
   * opaque CALL/FUNCTION bodies for their recursive scheduling boundary. */
  return poly_graph_rewrite_ctx_ex2(
      lctx->ctx, u, poly_kernel_split(), lctx, true, false
  );
}

typedef struct {
  bool failed;
} PolySplitKernelsContext;

/* Current tinygrad schedule/rangeify.py:547-563.  Kernel splitting owns one
 * fresh LocalAddBufferContext per closed STORE/END occurrence. */
static PolyUOp *split_store(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  if (!ctx || !root || (root->op != POLY_OP_STORE && root->op != POLY_OP_END)) return NULL;
  PolySplitKernelsContext *split_ctx =
      (PolySplitKernelsContext *)poly_graph_rewrite_userctx();

  int n_topo = 0;
  PolyUOp **topo = poly_toposort_ex_alloc(ctx, root, &n_topo, NULL, false);
  PolyUOp **ranges = n_topo > 0 ? malloc((size_t)n_topo * sizeof(*ranges)) : NULL;
  if (!topo || (n_topo > 0 && !ranges)) {
    poly_toposort_free(topo);
    free(ranges);
    return NULL;
  }
  int n_ranges = poly_uop_ranges(ctx, root, ranges, n_topo);
  bool closed = true;
  for (int i = 0; i < n_ranges; i++) {
    if (poly_range_axis_type(ranges[i]->arg) != POLY_AXIS_DEVICE) {
      closed = false;
      break;
    }
  }
  poly_toposort_free(topo);
  free(ranges);
  if (!closed) return NULL;

  PolyUOp *store = root->op == POLY_OP_END && root->n_src > 0 ? root->src[0] : root;
  if (store && store->op == POLY_OP_STORE && store->n_src > 0 &&
      poly_uop_is_variable(store->src[0]))
    return NULL;

  PolyLocalAddBufferContext lctx = {.ctx = ctx};
  PolyUOp *ret = NULL;

  PolyUOp *body_root = rewrite_kernel_split(&lctx, root);
  if (lctx.failed) {
    if (split_ctx) split_ctx->failed = true;
    free(lctx.map_values);
    free(lctx.map_keys);
    return NULL;
  }
  PolyKernelInfo kernel_info = {.name = "test"};
  PolyUOp *body = body_root
                      ? poly_uop(
                            ctx, POLY_OP_SINK, POLY_VOID, &body_root, 1,
                            poly_arg_kernel_info(&kernel_info))
                      : NULL;
  if (body) {
    PolyUOp **call_src = malloc((size_t)(lctx.map_count + 1) * sizeof(*call_src));
    if (call_src) {
      call_src[0] = body;
      for (int i = 0; i < lctx.map_count; i++) call_src[i + 1] = lctx.map_values[i];
      ret = poly_uop(
          ctx, POLY_OP_CALL, POLY_VOID, call_src, lctx.map_count + 1, poly_arg_none()
      );
      free(call_src);
    }
  }

  free(lctx.map_values);
  free(lctx.map_keys);
  return ret;
}

static PolyPatternMatcher *poly_split_kernels(void) {
  static _Thread_local PolyPatternMatcher *pm = NULL;
  if (pm) return pm;
  PolyOpSet ops = poly_opset_add((PolyOpSet){{0, 0}}, POLY_OP_STORE);
  ops = poly_opset_add(ops, POLY_OP_END);
  PolyRule rules[] = {{poly_upat_ops(ops, NULL, 0, "x"), split_store}};
  pm = poly_pm_thread_cache(poly_pm_new(rules, 1));
  return pm;
}

static bool shrink_has_zero_offset(PolyUOp *shrink) {
  if (!shrink || shrink->op != POLY_OP_SHRINK) return false;
  if (shrink->arg.kind == POLY_ARG_NONE && shrink->n_src >= 3) {
    PolyUOp *starts = shrink->src[1];
    if (!starts) return false;
    if (starts->op != POLY_OP_STACK) {
      int64_t value = 0;
      return poly_uop_const_i64(starts, &value) == 0 && value == 0;
    }
    for (int i = 0; i < starts->n_src; i++) {
      int64_t value = 0;
      if (poly_uop_const_i64(starts->src[i], &value) != 0 || value != 0) return false;
    }
    return true;
  }
  return false;
}

static PolyUOp *strip_zero_offset_shrink(PolyUOp *u) {
  return shrink_has_zero_offset(u) && u->n_src > 0 ? u->src[0] : u;
}

/* Current tinygrad schedule/rangeify.py:330-352 removes caller-side indexing
 * only after kernel splitting; the compiler body remains opaque. */
static PolyUOp *no_indexing_calls(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  if (!root || root->op != POLY_OP_CALL || root->n_src < 1) return NULL;
  PolyUOp **src = malloc((size_t)root->n_src * sizeof(*src));
  if (!src) return NULL;
  bool changed = false;
  for (int i = 0; i < root->n_src; i++) {
    PolyUOp *arg = root->src[i];
    if (arg && arg->op == POLY_OP_INDEX && arg->n_src > 0) {
      src[i] = arg->src[0];
    } else if (arg && arg->op == POLY_OP_SHRINK) {
      src[i] = strip_zero_offset_shrink(arg);
    } else if (arg && arg->op == POLY_OP_MSTACK) {
      PolyUOp **members = malloc((size_t)arg->n_src * sizeof(*members));
      if (!members) {
        free(src);
        return NULL;
      }
      bool member_changed = false;
      for (int j = 0; j < arg->n_src; j++) {
        members[j] = strip_zero_offset_shrink(arg->src[j]);
        member_changed |= members[j] != arg->src[j];
      }
      src[i] = member_changed
                   ? poly_uop(ctx, POLY_OP_MSTACK, arg->dtype, members, arg->n_src, arg->arg)
                   : arg;
      free(members);
    } else {
      src[i] = arg;
    }
    changed |= src[i] != root->src[i];
  }
  PolyUOp *ret = changed ? poly_uop(ctx, root->op, root->dtype, src, root->n_src, root->arg) : NULL;
  free(src);
  return ret;
}

static PolyPatternMatcher *poly_pm_no_indexing_calls(void) {
  static _Thread_local PolyPatternMatcher *pm = NULL;
  if (pm) return pm;
  PolyRule rules[] = {{poly_upat_op(POLY_OP_CALL, NULL, 0, "u"), no_indexing_calls}};
  pm = poly_pm_thread_cache(poly_pm_new(rules, 1));
  return pm;
}

static bool fix_store_hazard_gate(PolyUOp *u) {
  return u->op != POLY_OP_CONTIGUOUS;
}

/* Exact port of tinygrad schedule/rangeify.py:fix_store_hazard. */
static PolyUOp *fix_store_hazard(PolyCtx *ctx, PolyUOp *target, PolyUOp *src) {
  bool target_has_shrink = poly_uop_op_in_backward_slice_with_self(ctx, target, POLY_OP_SHRINK);
  PolyUOp *base = poly_uop_unsharded_base(target);

  int n_topo = 0;
  PolyUOp **topo = poly_toposort_ex_alloc(ctx, src, &n_topo, fix_store_hazard_gate, true);
  PolyMap *reaches_base = poly_map_new(n_topo < 16 ? 16 : (uint32_t)n_topo);
  if (!topo || !reaches_base) {
    poly_toposort_free(topo);
    if (reaches_base) poly_map_destroy(reaches_base);
    return NULL;
  }

  bool hazard = false;
  for (int i = 0; i < n_topo && !hazard; i++) {
    PolyUOp *u = topo[i];
    bool reaches = u == base;
    for (int j = 0; j < u->n_src && !reaches; j++) {
      reaches =
          poly_map_get(reaches_base, poly_ptr_hash(u->src[j]), u->src[j], poly_ptr_eq) != NULL;
    }
    if (reaches) poly_map_set(reaches_base, poly_ptr_hash(u), u, (void *)(uintptr_t)1, poly_ptr_eq);

    bool unsafe = u->op == POLY_OP_PERMUTE || u->op == POLY_OP_FLIP ||
                  (target_has_shrink && u->op == POLY_OP_SHRINK);
    if (reaches && unsafe && !(u == target && u->op == POLY_OP_SHRINK)) hazard = true;
  }

  poly_map_destroy(reaches_base);
  poly_toposort_free(topo);
  if (!hazard) return NULL;
  PolyUOp *contiguous = (src->op == POLY_OP_CONTIGUOUS || poly_uop_has_buffer_identity(src))
                            ? src
                            : poly_uop1(ctx, POLY_OP_CONTIGUOUS, src->dtype, src, poly_arg_none());
  return poly_store_val(ctx, target, contiguous);
}

static PolyUOp *rangeify_clone_preserving_metadata(
    PolyCtx *ctx,
    PolyUOp *u,
    PolyUOp **src,
    int n_src
) {
  if (u->tag != 0 || u->tag_arg.kind != POLY_ARG_NONE)
    return poly_uop_tagged_arg(ctx, u->op, u->dtype, src, n_src, u->arg, u->tag, u->tag_arg);
  return poly_uop(ctx, u->op, u->dtype, src, n_src, u->arg);
}

static bool mops_shape_equal(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  int a_ndim = poly_uop_ndim(ctx, a), b_ndim = poly_uop_ndim(ctx, b);
  if (a_ndim < 0 || a_ndim != b_ndim) return false;
  for (int axis = 0; axis < a_ndim; axis++) {
    PolyUOp *ad = poly_uop_shape_dim(ctx, a, axis);
    PolyUOp *bd = poly_uop_shape_dim(ctx, b, axis);
    if (ad == bd) continue;
    int64_t av = 0, bv = 0;
    if (!ad || !bd || poly_uop_const_i64(ad, &av) != 0 ||
        poly_uop_const_i64(bd, &bv) != 0 || av != bv)
      return false;
  }
  return true;
}

/* Current tinygrad schedule/rangeify.py::_mop_index. */
static PolyUOp *mop_index(PolyCtx *ctx, PolyUOp *idx, const PolyBindings *bindings) {
  (void)bindings;
  if (!ctx || !idx || idx->op != POLY_OP_INDEX || idx->n_src < 2) return NULL;
  PolyUOp *movement = idx->src[0];
  if (!movement || !poly_opset_has(POLY_GROUP_MOVEMENT, movement->op) ||
      movement->n_src < 1)
    return NULL;

  int n_idx = idx->n_src - 1;
  int movement_ndim = poly_uop_ndim(ctx, movement);
  if (movement_ndim < 0 || n_idx > POLY_MAX_DIMS) return NULL;
  PolyShape source_shape = poly_uop_max_shape_cached(ctx, movement->src[0]);
  if (source_shape.ndim < 0 || source_shape.ndim > POLY_MAX_DIMS) return NULL;

  PolyUOp *transformed[POLY_MAX_DIMS];
  PolyUOp *valid = NULL;
  int n_transformed = 0;
  if (n_idx == movement_ndim) {
    if (!poly_apply_movement_op(
            ctx, movement, movement->op, source_shape, movement->arg, idx->src + 1, n_idx,
            transformed, &n_transformed, &valid
        ))
      return NULL;
  } else {
    if (movement->op != POLY_OP_RESHAPE ||
        !poly_reshape_indices(
            ctx, movement, idx->src + 1, n_idx, transformed, &n_transformed
        ))
      return NULL;
    if (n_transformed == 0)
      return poly_dtype_eq(movement->src[0]->dtype, idx->dtype) ? movement->src[0] : NULL;
  }
  if (n_transformed < 0 || n_transformed > POLY_MAX_DIMS) return NULL;

  PolyUOp *src[POLY_MAX_DIMS + 1];
  src[0] = movement->src[0];
  for (int i = 0; i < n_transformed; i++) src[i + 1] = transformed[i];
  PolyUOp *ret = poly_uop(ctx, POLY_OP_INDEX, idx->dtype, src, n_transformed + 1, idx->arg);
  if (!ret) return NULL;
  return n_idx == movement_ndim || mops_shape_equal(ctx, ret, idx) ? ret : NULL;
}

/* Pinned tinygrad schedule/rangeify.py::pm_mops:
 *
 *   AFTER(MOVEMENT_OR_INDEX(x, ...), effects...)
 *     -> MOVEMENT_OR_INDEX(AFTER(x, effects...), ...)
 *
 * The fresh outer movement/INDEX intentionally has no tag, matching tinygrad's
 * direct UOp(...) constructor. AFTER metadata and every effect source are
 * preserved by the inner replacement. */
static PolyUOp *mops_move_after(PolyCtx *ctx, PolyUOp *after, const PolyBindings *bindings) {
  PolyUOp *moved = poly_bind(bindings, "moved");
  if (!ctx || !after || after->op != POLY_OP_AFTER || after->n_src < 1 || !moved ||
      moved != after->src[0] || moved->n_src < 1)
    return NULL;

  PolyUOp *after_inline[16];
  PolyUOp **after_src =
      after->n_src <= 16 ? after_inline : malloc((size_t)after->n_src * sizeof(*after_src));
  if (!after_src) return NULL;
  after_src[0] = moved->src[0];
  for (int i = 1; i < after->n_src; i++)
    after_src[i] = after->src[i];
  PolyUOp *inner_after = rangeify_clone_preserving_metadata(ctx, after, after_src, after->n_src);
  if (after_src != after_inline) free(after_src);
  if (!inner_after) return NULL;

  PolyUOp *moved_inline[16];
  PolyUOp **moved_src =
      moved->n_src <= 16 ? moved_inline : malloc((size_t)moved->n_src * sizeof(*moved_src));
  if (!moved_src) return NULL;
  moved_src[0] = inner_after;
  for (int i = 1; i < moved->n_src; i++)
    moved_src[i] = moved->src[i];
  PolyUOp *result = poly_uop(ctx, moved->op, moved->dtype, moved_src, moved->n_src, moved->arg);
  if (moved_src != moved_inline) free(moved_src);
  return result;
}

/* Pinned tinygrad schedule/rangeify.py::pm_mops:
 * END(MOVEMENT(x, ...), ranges...) -> END(x, ranges...). */
static PolyUOp *mops_end_movement(PolyCtx *ctx, PolyUOp *end, const PolyBindings *bindings) {
  PolyUOp *movement = poly_bind(bindings, "movement");
  if (!ctx || !end || end->op != POLY_OP_END || end->n_src < 1 || !movement ||
      movement != end->src[0] || movement->n_src < 1)
    return NULL;

  PolyUOp *inline_src[16];
  PolyUOp **src = end->n_src <= 16 ? inline_src : malloc((size_t)end->n_src * sizeof(*src));
  if (!src) return NULL;
  src[0] = movement->src[0];
  for (int i = 1; i < end->n_src; i++)
    src[i] = end->src[i];
  PolyUOp *result = rangeify_clone_preserving_metadata(ctx, end, src, end->n_src);
  if (src != inline_src) free(src);
  return result;
}

static _Thread_local PolyPatternMatcher *g_pm_mops = NULL;
PolyPatternMatcher *poly_pm_mops(void) {
  if (g_pm_mops) return g_pm_mops;
  PolyOpSet movement_or_index = poly_opset_add(POLY_GROUP_MOVEMENT, POLY_OP_INDEX);

  PolyUPat *after_child = poly_upat_ops(movement_or_index, NULL, 0, "moved");
  PolyUPat *after_src[] = {after_child};
  PolyUPat *end_child = poly_upat_ops(POLY_GROUP_MOVEMENT, NULL, 0, "movement");
  PolyUPat *end_src[] = {end_child};
  PolyRule rules[] = {
      {poly_upat_allow_any_len(poly_upat_op(POLY_OP_INDEX, NULL, 0, "idx")), mop_index},
      {poly_upat_allow_any_len(poly_upat_op(POLY_OP_AFTER, after_src, 1, "after")),
       mops_move_after},
      {poly_upat_allow_any_len(poly_upat_op(POLY_OP_END, end_src, 1, "end")), mops_end_movement},
  };
  g_pm_mops =
      poly_pm_thread_cache(poly_pm_new(rules, (int)(sizeof(rules) / sizeof(rules[0]))));
  return g_pm_mops;
}

/* Pinned tinygrad schedule/rangeify.py:102-123 split_reduceop.
 *
 * Static shape extraction deliberately requires literal CONST dimensions,
 * matching tinygrad's all_int(x.shape) gate rather than accepting bound
 * symbolic maxima. */
static bool split_reduceop_static_shape(
    PolyCtx *ctx,
    PolyUOp *u,
    int64_t dims[POLY_MAX_DIMS],
    int *ndim_out
) {
  int ndim = poly_uop_ndim(ctx, u);
  if (ndim < 0 || ndim > POLY_MAX_DIMS) return false;
  for (int i = 0; i < ndim; i++) {
    PolyUOp *dim = poly_uop_shape_dim(ctx, u, i);
    if (poly_uop_const_i64(dim, &dims[i]) != 0 || dims[i] < 0) return false;
  }
  *ndim_out = ndim;
  return true;
}

static bool split_reduceop_shape_product(const int64_t *dims, int ndim, int64_t *out) {
  int64_t product = 1;
  for (int i = 0; i < ndim; i++) {
    if (dims[i] == 0) {
      *out = 0;
      return true;
    }
    if (__builtin_mul_overflow(product, dims[i], &product)) return false;
  }
  *out = product;
  return true;
}

static PolyUOp *split_reduceop_range(PolyCtx *ctx, int64_t bound, int axis) {
  PolyUOp *bound_uop = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(bound));
  return bound_uop
             ? poly_uop1(
                   ctx, POLY_OP_RANGE, POLY_INT32, bound_uop, poly_arg_range(axis, POLY_AXIS_WEAK)
               )
             : NULL;
}

/* Match split_reduceop's INDEX(...).substitute({x.base:NOOP}, pm_mops).ranges
 * test. Only a top-level movement chain can remove one of x's own range
 * markers: for an ALU/value base tinygrad substitutes x itself with NOOP, so
 * every non-singleton output marker remains. */
static bool split_reduceop_expanded_axes(
    PolyCtx *ctx,
    PolyUOp *x,
    const int64_t *x_dims,
    int x_ndim,
    bool expanded[POLY_MAX_DIMS]
) {
  PolyUOp *markers[POLY_MAX_DIMS] = {0};
  PolyUOp *ranges[POLY_MAX_DIMS] = {0};
  for (int i = 0; i < x_ndim; i++) {
    if (x_dims[i] > 1) {
      markers[i] = split_reduceop_range(ctx, x_dims[i], i);
      if (!markers[i]) return false;
      ranges[i] = markers[i];
    } else {
      ranges[i] = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
      if (!ranges[i]) return false;
    }
  }

  PolyUOp *current = x;
  int n_ranges = x_ndim;
  while (current && current->n_src > 0 && poly_opset_has(POLY_GROUP_MOVEMENT, current->op)) {
    int64_t source_dims[POLY_MAX_DIMS];
    int source_ndim = 0;
    if (!split_reduceop_static_shape(ctx, current->src[0], source_dims, &source_ndim)) return false;
    PolyShape source_shape = {.dims = source_dims, .ndim = source_ndim};
    PolyUOp *transformed[POLY_MAX_DIMS] = {0};
    PolyUOp *valid = NULL;
    int n_transformed = 0;
    if (!poly_apply_movement_op(
            ctx, current, current->op, source_shape, current->arg, ranges, n_ranges, transformed,
            &n_transformed, &valid
        ) ||
        n_transformed < 0 || n_transformed > POLY_MAX_DIMS)
      return false;
    memcpy(ranges, transformed, (size_t)n_transformed * sizeof(*ranges));
    n_ranges = n_transformed;
    current = current->src[0];
  }

  for (int axis = 0; axis < x_ndim; axis++) {
    expanded[axis] = true;
    for (int i = 0; markers[axis] && i < n_ranges; i++) {
      if (poly_uop_reachable(ctx, ranges[i], markers[axis])) {
        expanded[axis] = false;
        break;
      }
    }
  }
  return true;
}

static PolyUOp *split_reduceop(PolyCtx *ctx, PolyUOp *reduce, PolyUOp *x) {
  if (!ctx || !(reduce->op == POLY_OP_REDUCE && reduce->arg.kind == POLY_ARG_REDUCE) || !x || !poly_getenv_flag_default("SPLIT_REDUCEOP", true))
    return NULL;

  int64_t x_dims[POLY_MAX_DIMS], out_dims[POLY_MAX_DIMS];
  int x_ndim = 0, out_ndim = 0;
  if (!split_reduceop_static_shape(ctx, x, x_dims, &x_ndim) ||
      !split_reduceop_static_shape(ctx, reduce, out_dims, &out_ndim) || x_ndim >= POLY_MAX_DIMS ||
      out_ndim != x_ndim - reduce->arg.reduce.num_axes)
    return NULL;

  int64_t x_product = 0, out_product = 0;
  if (!split_reduceop_shape_product(x_dims, x_ndim, &x_product) ||
      !split_reduceop_shape_product(out_dims, out_ndim, &out_product) || out_product <= 0)
    return NULL;
  int64_t threshold = poly_getenv_int("REDUCEOP_SPLIT_THRESHOLD", 32768);
  if (x_product / out_product < threshold) return NULL;

  bool expanded[POLY_MAX_DIMS] = {false};
  if (!split_reduceop_expanded_axes(ctx, x, x_dims, x_ndim, expanded)) return NULL;

  int split_size = poly_getenv_int("REDUCEOP_SPLIT_SIZE", 22);
  int64_t split_budget = split_size < 0     ? 0
                         : split_size >= 62 ? INT64_MAX
                                            : ((int64_t)1 << split_size);
  int64_t max_divisor = split_budget / out_product;
  if (max_divisor > 256) max_divisor = 256;

  int split_axis = -1;
  int64_t divisor = 0;
  for (int axis = 0; axis < reduce->arg.reduce.num_axes && split_axis < 0; axis++) {
    for (int64_t candidate = max_divisor; candidate >= 8; candidate--) {
      if (x_dims[axis] % candidate == 0 && !expanded[axis]) {
        split_axis = axis;
        divisor = candidate;
        break;
      }
    }
  }
  if (split_axis < 0) return NULL;

  int64_t split_shape[POLY_MAX_DIMS];
  int split_ndim = x_ndim + 1;
  int write = 0;
  for (int i = 0; i < x_ndim; i++) {
    if (i == split_axis) {
      split_shape[write++] = divisor;
      split_shape[write++] = x_dims[i] / divisor;
    } else {
      split_shape[write++] = x_dims[i];
    }
  }

  int64_t permutation[POLY_MAX_DIMS];
  write = 0;
  for (int i = 0; i < split_ndim; i++)
    if (i != split_axis) permutation[write++] = i;
  permutation[write++] = split_axis;

  PolyUOp *reshaped = poly_reshape(ctx, x, split_shape, split_ndim);
  PolyUOp *permuted = reshaped ? poly_permute(ctx, reshaped, permutation, split_ndim) : NULL;
  int64_t first_axes[POLY_MAX_DIMS];
  for (int i = 0; i < reduce->arg.reduce.num_axes; i++) first_axes[i] = i;
  PolyUOp *first =
      permuted ? poly_reduce_axis(
                     ctx, reduce->arg.reduce.op, permuted, first_axes,
                     reduce->arg.reduce.num_axes
                 )
               : NULL;
  PolyUOp *contiguous =
      first ? poly_uop1(ctx, POLY_OP_CONTIGUOUS, first->dtype, first, poly_arg_none()) : NULL;
  int64_t second_axis[] = {out_ndim};
  PolyUOp *second =
      contiguous ? poly_reduce_axis(ctx, reduce->arg.reduce.op, contiguous, second_axis, 1)
                 : NULL;
  PolyUOp *result = second ? poly_reshape(ctx, second, out_dims, out_ndim) : NULL;

  if (result && poly_debug_at_least(3)) {
    fprintf(
        stderr, "split %lld: axis=%d input=%lld output=%lld\n", (long long)divisor, split_axis,
        (long long)x_product, (long long)out_product
    );
  }
  return result;
}

/* Pinned schedule/allreduce.py:6-18 chooses the naive branch for every
 * two-device collective unless RING/ALL2ALL is explicitly forced, and for
 * small collectives on any device count. Keep the unported ring/all-to-all
 * branches fail-closed instead of allowing raw ALLREDUCE to reach codegen. */
static PolyUOp *rangeify_resolve_function(PolyCtx *ctx, PolyUOp *call) {
  if (!ctx || !call || call->op != POLY_OP_FUNCTION || call->n_src < 1 ||
      !call->src[0] || call->src[0]->op != POLY_OP_TUPLE)
    return NULL;
  PolyUOp *body = call->src[0];
  int n_topo = 0;
  PolyUOp **topo = poly_toposort_ex_alloc(ctx, body, &n_topo, NULL, false);
  if (!topo) return NULL;

  PolyMap *sub_map = poly_map_new(n_topo < 16 ? 16 : (uint32_t)n_topo);
  PolyMap *memo = poly_map_new(n_topo < 16 ? 16 : (uint32_t)n_topo);
  if (!sub_map || !memo) {
    if (sub_map) poly_map_destroy(sub_map);
    if (memo) poly_map_destroy(memo);
    poly_toposort_free(topo);
    return NULL;
  }

  bool ok = true;
  for (int i = 0; ok && i < n_topo; i++) {
    PolyUOp *param = topo[i];
    if (!param || param->op != POLY_OP_PARAM) continue;
    int64_t slot = -1;
    if (param->arg.kind == POLY_ARG_PARAM && param->arg.param)
      slot = param->arg.param->slot;
    else if (param->arg.kind == POLY_ARG_INT)
      slot = param->arg.i;
    if (slot < 0) continue;
    if (slot >= call->n_src - 1) {
      ok = false;
      break;
    }
    PolyUOp *arg = call->src[1 + slot];
    int param_axis = -1, arg_axis = -1;
    bool param_has_axis = poly_uop_axis(ctx, param, &param_axis);
    bool arg_has_axis = poly_uop_axis(ctx, arg, &arg_axis);
    PolyShape param_shape = poly_uop_max_shape_cached(ctx, param);
    PolyShape arg_shape = poly_uop_max_shape_cached(ctx, arg);
    if (!arg || param_has_axis != arg_has_axis ||
        (param_has_axis && param_axis != arg_axis) ||
        param_shape.ndim < 0 || arg_shape.ndim != param_shape.ndim ||
        !poly_dtype_eq(param->dtype, arg->dtype)) {
      ok = false;
      break;
    }
    for (int d = 0; d < param_shape.ndim; d++)
      if (param_shape.dims[d] != arg_shape.dims[d]) ok = false;
    if (ok)
      poly_map_set(sub_map, poly_ptr_hash(param), param, arg, poly_ptr_eq);
  }

  for (int i = 0; ok && i < n_topo; i++) {
    PolyUOp *u = topo[i];
    PolyUOp *result = poly_map_get(
        sub_map, poly_ptr_hash(u), u, poly_ptr_eq);
    if (!result) {
      PolyUOp *src_stack[16];
      PolyUOp **src = u->n_src > 16
                          ? malloc((size_t)u->n_src * sizeof(*src))
                          : src_stack;
      if (!src) {
        ok = false;
        break;
      }
      bool changed = false;
      for (int s = 0; s < u->n_src; s++) {
        PolyUOp *mapped = poly_map_get(
            memo, poly_ptr_hash(u->src[s]), u->src[s], poly_ptr_eq);
        src[s] = mapped ? mapped : u->src[s];
        if (src[s] != u->src[s]) changed = true;
      }
      result = changed
                   ? rangeify_clone_preserving_metadata(ctx, u, src, u->n_src)
                   : u;
      if (src != src_stack) free(src);
    }
    if (!result) {
      ok = false;
      break;
    }
    if (result != u)
      poly_map_set(memo, poly_ptr_hash(u), u, result, poly_ptr_eq);
  }

  PolyUOp *ret = ok
                     ? poly_map_get(memo, poly_ptr_hash(body), body, poly_ptr_eq)
                     : NULL;
  if (ok && !ret) ret = body;
  poly_map_destroy(sub_map);
  poly_map_destroy(memo);
  poly_toposort_free(topo);
  return ret;
}

/* Pinned schedule/rangeify.py:150-164 runs FUNCTION resolution and DETACH
 * removal through the same bottom-up graph_rewrite.  The generic C driver
 * already has the corresponding fixed-point/replacement traversal: after a
 * FUNCTION becomes its substituted TUPLE body, it descends into that body.
 * Keep only FUNCTION resolution in this matcher; the existing earliest pass
 * below then applies the remaining ordered rules to the exposed graph. */
static PolyUOp *resolve_function_match(
    PolyCtx *ctx,
    PolyUOp *call,
    const PolyBindings *bindings
) {
  (void)bindings;
  if (!ctx || !call || call->op != POLY_OP_FUNCTION) return NULL;
  if (call->arg.kind == POLY_ARG_CALL_INFO && call->arg.call_info &&
      call->arg.call_info->precompile)
    return NULL;
  return rangeify_resolve_function(ctx, call);
}

static PolyDType bitcast_uint_dtype(int itemsize) {
  switch (itemsize) {
    case 1: return POLY_UINT8;
    case 2: return POLY_UINT16;
    case 4: return POLY_UINT32;
    case 8: return POLY_UINT64;
    default: return POLY_VOID;
  }
}

/* Exact C port of current schedule/rangeify.py:111-125 expand_bitcast.
 * Tensor construction deliberately retains one raw unequal-width BITCAST;
 * this earliest scheduler rule exposes its byte lanes for ordinary devices.
 * DISK/TINYFS remain typed storage views and are not expanded. */
static PolyUOp *expand_bitcast(PolyCtx *ctx, PolyUOp *bc, PolyUOp *x) {
  if (!ctx || !bc || bc->op != POLY_OP_BITCAST || !x) return NULL;
  int ns = poly_dtype_itemsize(bc->dtype);
  int os = poly_dtype_itemsize(x->dtype);
  if (ns <= 0 || os <= 0 || ns == os) return NULL;
  const char *device = poly_uop_device_name(ctx, x);
  if (device &&
      (strncmp(device, "DISK", 4) == 0 || strncmp(device, "TINYFS", 6) == 0))
    return NULL;

  PolyDType old_uint = bitcast_uint_dtype(os);
  PolyDType new_uint = bitcast_uint_dtype(ns);
  if (poly_dtype_eq(old_uint, POLY_VOID) || poly_dtype_eq(new_uint, POLY_VOID)) return NULL;
  PolyUOp *tmp = poly_dtype_eq(x->dtype, old_uint)
                       ? x
                       : poly_uop1(ctx, POLY_OP_BITCAST, old_uint, x, poly_arg_none());
  int ndim = poly_uop_ndim(ctx, x);
  if (!tmp || ndim <= 0 || ndim > POLY_MAX_DIMS) return NULL;

  if (ns > os) {
    if (ns % os != 0 || ndim >= POLY_MAX_DIMS) return NULL;
    int rate = ns / os;
    PolyUOp *reshape_shape[POLY_MAX_DIMS];
    for (int axis = 0; axis < ndim - 1; axis++)
      reshape_shape[axis] = poly_uop_shape_dim(ctx, x, axis);
    PolyUOp *last = poly_uop_shape_dim(ctx, x, ndim - 1);
    PolyUOp *rate_uop = rangeify_index_const(ctx, rate);
    reshape_shape[ndim - 1] = last && rate_uop
                                  ? poly_uop2(
                                        ctx, POLY_OP_FLOORDIV, POLY_WEAKINT, last,
                                        rate_uop, poly_arg_none())
                                  : NULL;
    reshape_shape[ndim - 1] = reshape_shape[ndim - 1]
                                  ? poly_graph_rewrite(
                                        ctx, reshape_shape[ndim - 1], poly_symbolic())
                                  : NULL;
    reshape_shape[ndim] = rate_uop;
    for (int axis = 0; axis <= ndim; axis++)
      if (!reshape_shape[axis]) return NULL;
    tmp = poly_reshape_uop(ctx, tmp, reshape_shape, ndim + 1);
    if (!tmp) return NULL;

    PolyUOp *sum = NULL;
    for (int lane = 0; lane < rate; lane++) {
      PolyUOp *starts[POLY_MAX_DIMS], *sizes[POLY_MAX_DIMS];
      for (int axis = 0; axis <= ndim; axis++) {
        starts[axis] = rangeify_index_const(ctx, 0);
        sizes[axis] = poly_uop_shape_dim(ctx, tmp, axis);
      }
      starts[ndim] = rangeify_index_const(ctx, lane);
      sizes[ndim] = rangeify_index_const(ctx, 1);
      PolyUOp *part = poly_shrink_uop(ctx, tmp, starts, sizes, ndim + 1);
      part = part ? poly_cast(ctx, part, new_uint) : NULL;
      PolyUOp *shift = rangeify_index_const(ctx, 8LL * lane * os);
      part = part && shift ? poly_alu2(ctx, POLY_OP_SHL, part, shift) : NULL;
      sum = !part ? NULL : sum ? poly_alu2(ctx, POLY_OP_ADD, sum, part) : part;
      if (!sum) return NULL;
    }
    PolyUOp *squeezed_shape[POLY_MAX_DIMS];
    for (int axis = 0; axis < ndim; axis++)
      squeezed_shape[axis] = poly_uop_shape_dim(ctx, sum, axis);
    sum = poly_reshape_uop(ctx, sum, squeezed_shape, ndim);
    return sum ? poly_uop1(ctx, POLY_OP_BITCAST, bc->dtype, sum, poly_arg_none()) : NULL;
  }

  if (os % ns != 0) return NULL;
  int rate = os / ns;
  PolyUOp *parts[8];
  if (rate <= 0 || rate > (int)(sizeof(parts) / sizeof(parts[0]))) return NULL;
  for (int lane = 0; lane < rate; lane++) {
    PolyUOp *shift = rangeify_index_const(ctx, 8LL * lane * ns);
    parts[lane] = shift ? poly_alu2(ctx, POLY_OP_SHR, tmp, shift) : NULL;
    if (!parts[lane]) return NULL;
  }
  PolyUOp *stacked = poly_stack(ctx, parts, rate, -1);
  if (!stacked || poly_uop_ndim(ctx, stacked) != ndim + 1) return NULL;
  PolyUOp *flattened_shape[POLY_MAX_DIMS];
  for (int axis = 0; axis < ndim - 1; axis++)
    flattened_shape[axis] = poly_uop_shape_dim(ctx, stacked, axis);
  PolyUOp *left = poly_uop_shape_dim(ctx, stacked, ndim - 1);
  PolyUOp *right = poly_uop_shape_dim(ctx, stacked, ndim);
  flattened_shape[ndim - 1] = left && right
                                  ? poly_uop2(
                                        ctx, POLY_OP_MUL, POLY_WEAKINT, left, right,
                                        poly_arg_none())
                                  : NULL;
  flattened_shape[ndim - 1] = flattened_shape[ndim - 1]
                                  ? poly_graph_rewrite(
                                        ctx, flattened_shape[ndim - 1], poly_symbolic())
                                  : NULL;
  if (!flattened_shape[ndim - 1]) return NULL;
  PolyUOp *flattened = poly_reshape_uop(ctx, stacked, flattened_shape, ndim);
  flattened = flattened ? poly_cast(ctx, flattened, new_uint) : NULL;
  return flattened
             ? poly_uop1(ctx, POLY_OP_BITCAST, bc->dtype, flattened, poly_arg_none())
             : NULL;
}

/* Current Tinygrad schedule/rangeify.py:earliest_rewrites. */
static PolyUOp *earliest_gettuple(
    PolyCtx *ctx, PolyUOp *g, const PolyBindings *bindings
) {
  (void)ctx;
  (void)bindings;
  if (!g || g->op != POLY_OP_GETTUPLE || g->n_src != 1 ||
      g->src[0]->op != POLY_OP_TUPLE || g->arg.kind != POLY_ARG_INT ||
      g->arg.i < 0 || g->arg.i >= g->src[0]->n_src)
    return NULL;
  return g->src[0]->src[g->arg.i];
}

static PolyUOp *earliest_allreduce(
    PolyCtx *ctx, PolyUOp *red, const PolyBindings *bindings
) {
  (void)bindings;
  return poly_create_allreduce_function(ctx, red);
}

static PolyUOp *earliest_split_reduceop(
    PolyCtx *ctx, PolyUOp *reduce, const PolyBindings *bindings
) {
  (void)bindings;
  return reduce && reduce->n_src == 1
             ? split_reduceop(ctx, reduce, reduce->src[0])
             : NULL;
}

static PolyUOp *earliest_remove_passthrough(
    PolyCtx *ctx, PolyUOp *x, const PolyBindings *bindings
) {
  (void)ctx;
  (void)bindings;
  return x && x->n_src == 1 ? x->src[0] : NULL;
}

static PolyUOp *earliest_sink_bases(
    PolyCtx *ctx, PolyUOp *sink, const PolyBindings *bindings
) {
  (void)bindings;
  if (!sink || sink->op != POLY_OP_SINK) return NULL;
  PolyUOp *inline_src[16];
  PolyUOp **src = sink->n_src <= 16
                      ? inline_src
                      : malloc((size_t)sink->n_src * sizeof(*src));
  if (!src) return NULL;
  bool changed = false;
  for (int i = 0; i < sink->n_src; i++) {
    src[i] = poly_uop_unsharded_base(sink->src[i]);
    changed |= src[i] != sink->src[i];
  }
  PolyUOp *ret = changed
                     ? rangeify_clone_preserving_metadata(ctx, sink, src, sink->n_src)
                     : NULL;
  if (src != inline_src) free(src);
  return ret;
}

static PolyUOp *copy_target_device(PolyCtx *ctx, PolyUOp *copy) {
  if (!copy || copy->op != POLY_OP_COPY || copy->n_src != 1) return NULL;
  if (copy->arg.kind == POLY_ARG_STRING)
    return poly_device_uop_from_name(ctx, copy->arg.str);
  if (copy->arg.kind == POLY_ARG_STRING_TUPLE)
    return poly_device_uop_from_names(
        ctx, copy->arg.string_tuple.vals, copy->arg.string_tuple.n);
  return NULL;
}

static PolyUOp *earliest_copy_movement(
    PolyCtx *ctx, PolyUOp *copy, const PolyBindings *bindings
) {
  (void)bindings;
  if (!copy || copy->op != POLY_OP_COPY || copy->n_src != 1 ||
      !poly_opset_has(POLY_GROUP_MOVEMENT, copy->src[0]->op))
    return NULL;
  PolyUOp *movement = copy->src[0];
  PolyUOp *base = poly_uop_unsharded_base(movement);
  PolyShape shape = poly_uop_max_shape_cached(ctx, movement);
  PolyShape base_shape = poly_uop_max_shape_cached(ctx, base);
  int64_t numel = shape.ndim >= 0 ? poly_shape_numel(shape) : -1;
  int64_t base_numel = base_shape.ndim >= 0 ? poly_shape_numel(base_shape) : -1;
  int64_t view_offset = 0;
  bool contiguous_view = poly_uop_contiguous_view_offset(ctx, movement, &view_offset) == 0;
  if (numel < 0 || base_numel < 0 || (numel == base_numel && contiguous_view)) return NULL;
  PolyUOp *contiguous = poly_contiguous(ctx, movement);
  PolyUOp *src[] = {contiguous};
  return contiguous
             ? rangeify_clone_preserving_metadata(ctx, copy, src, 1)
             : NULL;
}

static PolyUOp *earliest_copy_same_device(
    PolyCtx *ctx, PolyUOp *copy, const PolyBindings *bindings
) {
  (void)bindings;
  PolyUOp *device = copy_target_device(ctx, copy);
  PolyUOp *source_device = copy && copy->n_src == 1
                               ? poly_uop_device_uop_cached(ctx, copy->src[0], NULL)
                               : NULL;
  return device && source_device == device ? copy->src[0] : NULL;
}

static PolyUOp *earliest_copy_reshape(
    PolyCtx *ctx, PolyUOp *copy, const PolyBindings *bindings
) {
  (void)bindings;
  if (!copy || copy->op != POLY_OP_COPY || copy->n_src != 1 ||
      copy->src[0]->op != POLY_OP_RESHAPE || copy->src[0]->n_src < 2)
    return NULL;
  PolyUOp *reshape = copy->src[0];
  PolyUOp *device = copy_target_device(ctx, copy);
  PolyUOp *inner = device
                       ? poly_copy_to_device_uop(ctx, reshape->src[0], device)
                       : NULL;
  PolyUOp *src[] = {inner, reshape->src[1]};
  return inner
             ? rangeify_clone_preserving_metadata(ctx, reshape, src, 2)
             : NULL;
}

static PolyUOp *earliest_store_reshape(
    PolyCtx *ctx, PolyUOp *store, const PolyBindings *bindings
) {
  (void)bindings;
  if (!store || store->op != POLY_OP_STORE || store->n_src != 2 ||
      store->src[0]->op != POLY_OP_RESHAPE || store->src[1]->op != POLY_OP_RESHAPE ||
      store->src[0]->n_src < 1 || store->src[1]->n_src < 1)
    return NULL;
  PolyShape dst = poly_uop_max_shape_cached(ctx, store->src[0]->src[0]);
  PolyShape src = poly_uop_max_shape_cached(ctx, store->src[1]->src[0]);
  return dst.ndim >= 0 && poly_shape_eq(dst, src)
             ? poly_store_val(ctx, store->src[0]->src[0], store->src[1]->src[0])
             : NULL;
}

static PolyUOp *earliest_dedup_store(
    PolyCtx *ctx, PolyUOp *outer, const PolyBindings *bindings
) {
  (void)ctx;
  (void)bindings;
  if (!outer || outer->op != POLY_OP_AFTER || outer->n_src != 2 ||
      outer->src[0]->op != POLY_OP_AFTER || outer->src[0]->n_src != 2 ||
      outer->src[1]->op != POLY_OP_STORE || outer->src[1]->n_src != 2)
    return NULL;
  PolyUOp *inner = outer->src[0];
  PolyUOp *first = inner->src[1];
  PolyUOp *second = outer->src[1];
  if (first->op != POLY_OP_STORE || first->n_src != 2 ||
      inner->src[0] != first->src[0] || inner != second->src[0] ||
      first->src[1] != second->src[1])
    return NULL;
  return inner;
}

static PolyUOp *earliest_remove_nested_store(
    PolyCtx *ctx, PolyUOp *after, const PolyBindings *bindings
) {
  (void)ctx;
  (void)bindings;
  if (!after || after->op != POLY_OP_AFTER || after->n_src != 2 ||
      after->src[1]->op != POLY_OP_STORE || after->src[1]->n_src != 2 ||
      after->src[1]->src[1]->op != POLY_OP_AFTER ||
      after->src[1]->src[1]->n_src != 2)
    return NULL;
  PolyUOp *buf = after->src[0];
  PolyUOp *outer_store = after->src[1];
  PolyUOp *inner = outer_store->src[1];
  PolyUOp *inner_store = inner->src[1];
  return inner_store->op == POLY_OP_STORE && inner_store->n_src == 2 &&
                 outer_store->src[0] == buf && inner->src[0] == buf &&
                 inner_store->src[0] == buf
             ? inner
             : NULL;
}

static PolyUOp *earliest_fix_store_hazard(
    PolyCtx *ctx, PolyUOp *store, const PolyBindings *bindings
) {
  (void)bindings;
  return store && store->n_src == 2
             ? fix_store_hazard(ctx, store->src[0], store->src[1])
             : NULL;
}

static PolyUOp *earliest_store_bitcast(
    PolyCtx *ctx, PolyUOp *store, const PolyBindings *bindings
) {
  (void)bindings;
  if (!store || store->op != POLY_OP_STORE || store->n_src != 2 ||
      store->src[0]->op != POLY_OP_BITCAST || store->src[0]->n_src != 1)
    return NULL;
  PolyUOp *target = store->src[0]->src[0];
  PolyUOp *value = poly_uop1(
      ctx, POLY_OP_BITCAST, target->dtype, store->src[1], poly_arg_dtype(target->dtype));
  return value ? poly_store_val(ctx, target, value) : NULL;
}

static PolyUOp *earliest_expand_bitcast(
    PolyCtx *ctx, PolyUOp *bc, const PolyBindings *bindings
) {
  (void)bindings;
  return bc && bc->n_src == 1 ? expand_bitcast(ctx, bc, bc->src[0]) : NULL;
}

static PolyUOp *earliest_zero_reduce(
    PolyCtx *ctx, PolyUOp *reduce, const PolyBindings *bindings
) {
  (void)bindings;
  if (!reduce || reduce->op != POLY_OP_REDUCE || reduce->n_src != 1 ||
      reduce->arg.kind != POLY_ARG_REDUCE)
    return NULL;
  PolyShape input = poly_uop_max_shape_cached(ctx, reduce->src[0]);
  PolyShape output = poly_uop_max_shape_cached(ctx, reduce);
  bool input_zero = false, output_zero = false;
  for (int i = 0; i < input.ndim; i++) input_zero |= input.dims[i] == 0;
  for (int i = 0; i < output.ndim; i++) output_zero |= output.dims[i] == 0;
  if (!input_zero || output_zero) return NULL;
  PolyUOp *identity = poly_identity_element(ctx, reduce->arg.reduce.op, reduce->dtype);
  return identity ? poly_const_like(ctx, reduce, identity->arg) : NULL;
}

static PolyUOp *earliest_zero_shape(
    PolyCtx *ctx, PolyUOp *x, const PolyBindings *bindings
) {
  (void)bindings;
  if (!x || x->op == POLY_OP_SINK) return NULL;
  PolyShape shape = poly_uop_max_shape_cached(ctx, x);
  bool has_zero = false;
  for (int i = 0; i < shape.ndim; i++) has_zero |= shape.dims[i] == 0;
  if (!has_zero) return NULL;
  PolyArg zero = poly_dtype_is_float(x->dtype)  ? poly_arg_float(0.0)
                 : poly_dtype_is_bool(x->dtype) ? poly_arg_bool(false)
                                                : poly_arg_int(0);
  PolyUOp *ret = poly_const_like(ctx, x, zero);
  return ret && (x->tag != 0 || x->tag_arg.kind != POLY_ARG_NONE)
             ? poly_uop_tagged_arg(
                   ctx, ret->op, ret->dtype, ret->src, ret->n_src, ret->arg,
                   x->tag, x->tag_arg)
             : ret;
}

static _Thread_local PolyPatternMatcher *g_earliest_rewrites = NULL;
static PolyPatternMatcher *earliest_rewrites(void) {
  if (g_earliest_rewrites) return g_earliest_rewrites;
  PolyOpSet passthrough = poly_opset_add(
      poly_opset_add((PolyOpSet){{0, 0}}, POLY_OP_DETACH),
      POLY_OP_CONTIGUOUS_BACKWARD);
  PolyNamedRule rules[] = {
      POLY_RULE(poly_upat_op(POLY_OP_FUNCTION, NULL, 0, "c"), resolve_function_match),
      POLY_RULE(poly_upat_op(POLY_OP_GETTUPLE, NULL, 0, "g"), earliest_gettuple),
      POLY_RULE(poly_upat_op(POLY_OP_ALLREDUCE, NULL, 0, "red"), earliest_allreduce),
      POLY_RULE(poly_upat_op(POLY_OP_REDUCE, NULL, 0, "reduce"), earliest_split_reduceop),
      POLY_RULE(poly_upat_ops(passthrough, NULL, 0, "x"), earliest_remove_passthrough),
      POLY_RULE(poly_upat_allow_any_len(poly_upat_op(POLY_OP_SINK, NULL, 0, "x")), earliest_sink_bases),
      POLY_RULE(poly_upat_op(POLY_OP_COPY, NULL, 0, "copy"), earliest_copy_movement),
      POLY_RULE(poly_upat_op(POLY_OP_COPY, NULL, 0, "copy"), earliest_copy_same_device),
      POLY_RULE(poly_upat_op(POLY_OP_COPY, NULL, 0, "copy"), earliest_copy_reshape),
      POLY_RULE(poly_upat_op(POLY_OP_STORE, NULL, 0, "store"), earliest_store_reshape),
      POLY_RULE(poly_upat_op(POLY_OP_STORE, NULL, 0, "store"), earliest_fix_store_hazard),
      POLY_RULE(poly_upat_op(POLY_OP_AFTER, NULL, 0, "after"), earliest_dedup_store),
      POLY_RULE(poly_upat_op(POLY_OP_AFTER, NULL, 0, "after"), earliest_remove_nested_store),
      POLY_RULE(poly_upat_op(POLY_OP_STORE, NULL, 0, "store"), earliest_store_bitcast),
      POLY_RULE(poly_upat_op(POLY_OP_BITCAST, NULL, 0, "bc"), earliest_expand_bitcast),
      POLY_RULE(poly_upat_op(POLY_OP_REDUCE, NULL, 0, "reduce"), earliest_zero_reduce),
      POLY_RULE(poly_upat_any("x"), earliest_zero_shape),
  };
  PolyPatternMatcher *specific = poly_pm_new_named(
      rules, (int)(sizeof(rules) / sizeof(rules[0])));
  PolyPatternMatcher *mops_cleanup = poly_pm_concat(poly_pm_mops(), poly_mop_cleanup());
  PolyPatternMatcher *combined = poly_pm_concat(mops_cleanup, specific);
  g_earliest_rewrites = poly_pm_thread_cache(combined);
  poly_pm_destroy(mops_cleanup);
  poly_pm_destroy(specific);
  return g_earliest_rewrites;
}

/* Current tinygrad schedule/rangeify.py:earliest_rewrites. */
PolyUOp *poly_apply_earliest_rewrites(PolyCtx *ctx, PolyUOp *sink) {
  return poly_graph_rewrite_ctx_ex2(
      ctx, sink, earliest_rewrites(), NULL, true, false);
}

static bool rangeify_always_run(PolyUOp *u) {
  return u && (u->op == POLY_OP_CONTIGUOUS || u->op == POLY_OP_NOOP);
}

/* Current Tinygrad schedule/rangeify.py:cleanup_dead_axes. */
static PolyUOp *cleanup_dead_axes(
    PolyCtx *ctx,
    PolyUOp *stage,
    const PolyBindings *bindings
) {
  (void)bindings;
  if (!stage || stage->op != POLY_OP_STAGE || stage->n_src < 1 ||
      !poly_bufferize_arg_removable(stage->arg) || rangeify_always_run(stage->src[0]) ||
      stage->src[0]->op == POLY_OP_AFTER)
    return NULL;
  int n_axes = stage->n_src - 1;
  PolyShape shape = poly_uop_max_shape_cached(ctx, stage);
  if (n_axes < 0 || n_axes > POLY_MAX_DIMS || shape.ndim != n_axes) return NULL;

  PolyUOp *src[POLY_MAX_DIMS + 1];
  int64_t reshape[POLY_MAX_DIMS];
  int n_live = 0;
  bool hit = false;
  src[0] = stage->src[0];
  for (int i = 0; i < n_axes; i++) {
    PolyUOp *range = stage->src[i + 1];
    if (range->op == POLY_OP_RANGE &&
        (range->n_src != 1 || range->src[0]->op != POLY_OP_CONST))
      return NULL;
    bool dead = range->op == POLY_OP_CONST ||
                (range->op == POLY_OP_RANGE &&
                 !poly_uop_in_ranges(ctx, stage->src[0], range));
    reshape[i] = dead ? 1 : shape.dims[i];
    if (dead)
      hit = true;
    else
      src[++n_live] = range;
  }
  if (!hit) return NULL;
  PolyUOp *trimmed = poly_uop_tagged_arg(
      ctx, stage->op, stage->dtype, src, n_live + 1, stage->arg, stage->tag,
      stage->tag_arg
  );
  PolyUOp *reshaped = trimmed ? poly_reshape(ctx, trimmed, reshape, n_axes) : NULL;
  return reshaped ? poly_expand(ctx, reshaped, shape.dims, shape.ndim) : NULL;
}

/* Current Tinygrad schedule/rangeify.py:remove_noop_bufferize. */
static PolyUOp *remove_noop_bufferize(
    PolyCtx *ctx,
    PolyUOp *stage,
    const PolyBindings *bindings
) {
  (void)bindings;
  if (!stage || stage->op != POLY_OP_STAGE || stage->n_src < 1) return NULL;
  PolyUOp *index = stage->src[0];
  if (index && index->op == POLY_OP_NOOP && index->n_src == 1) index = index->src[0];
  if (!index || index->op != POLY_OP_INDEX || index->n_src != stage->n_src) return NULL;
  for (int i = 1; i < stage->n_src; i++)
    if (index->src[i] != stage->src[i]) return NULL;
  PolyShape shape = poly_uop_max_shape_cached(ctx, stage);
  if (shape.ndim == 0) return index->src[0];
  if (shape.ndim < 0 || shape.ndim > POLY_MAX_DIMS) return NULL;
  int64_t bounds[POLY_MAX_DIMS][2];
  for (int i = 0; i < shape.ndim; i++) {
    bounds[i][0] = 0;
    bounds[i][1] = shape.dims[i];
  }
  return poly_shrink(ctx, index->src[0], bounds, shape.ndim);
}

static PolyUOp *fold_const_buffer(
    PolyCtx *ctx,
    PolyUOp *root,
    const PolyBindings *bindings
) {
  (void)bindings;
  if (root->op == POLY_OP_STAGE && root->n_src > 0 &&
      root->src[0]->op == POLY_OP_CONST)
    return poly_const_like(ctx, root, root->src[0]->arg);
  if (root->op == POLY_OP_INDEX && root->n_src > 0 &&
      root->src[0]->op == POLY_OP_CONST)
    return root->src[0];
  if (root->op == POLY_OP_NOOP && root->n_src == 1 &&
      root->src[0]->op == POLY_OP_CONST)
    return root->src[0];
  return NULL;
}

/* Current Tinygrad schedule/rangeify.py:after_all_invalid.  INDEX(AFTER)
 * is invalid only when every effect stores Invalid across the whole same
 * buffer; partial, padded, expanded, or foreign-buffer stores stay explicit. */
static bool after_all_invalid(PolyCtx *ctx, PolyUOp *after) {
  if (!ctx || !after || after->op != POLY_OP_AFTER || after->n_src < 1) return false;
  PolyUOp *buf = poly_uop_buf_uop(ctx, after->src[0]);
  if (!buf) return false;

  for (int i = 1; i < after->n_src; i++) {
    PolyUOp *end = after->src[i];
    if (!end || end->op != POLY_OP_END || end->n_src < 1) return false;
    PolyUOp *store = end->src[0];
    PolyUOp *value_base = store && store->n_src == 2 ? poly_uop_base(store->src[1]) : NULL;
    if (!store || store->op != POLY_OP_STORE || store->n_src != 2 ||
        !value_base || value_base->op != POLY_OP_CONST ||
        value_base->arg.kind != POLY_ARG_INVALID ||
        poly_uop_buf_uop(ctx, store->src[0]) != buf)
      return false;

    PolyUOp *ended_numel = rangeify_index_const(ctx, 1);
    for (int r = 1; r < end->n_src; r++) {
      PolyUOp *range = end->src[r];
      if (!range || range->op != POLY_OP_RANGE || range->n_src < 1 ||
          !poly_uop_in_ranges(ctx, store->src[0], range))
        return false;
      ended_numel = poly_binop(ctx, POLY_OP_MUL, ended_numel, range->src[0]);
      if (!ended_numel) return false;
    }

    int ndim = poly_uop_ndim(ctx, buf);
    if (ndim < 0 || ndim > POLY_MAX_DIMS) return false;
    PolyUOp *buf_numel = rangeify_index_const(ctx, 1);
    for (int axis = 0; axis < ndim; axis++) {
      PolyUOp *dim = poly_uop_shape_dim(ctx, buf, axis);
      buf_numel = dim ? poly_binop(ctx, POLY_OP_MUL, buf_numel, dim) : NULL;
      if (!buf_numel) return false;
    }
    PolyUOp *same_numel = poly_binop(ctx, POLY_OP_CMPEQ, ended_numel, buf_numel);
    if (!same_numel || poly_uop_resolve(ctx, same_numel, 0) != 1) return false;
  }
  return true;
}

static PolyUOp *index_after_all_invalid(
    PolyCtx *ctx,
    PolyUOp *index,
    const PolyBindings *bindings
) {
  (void)bindings;
  if (!index || index->op != POLY_OP_INDEX || index->n_src < 1 ||
      !index->src[0] || index->src[0]->op != POLY_OP_AFTER ||
      !after_all_invalid(ctx, index->src[0]))
    return NULL;
  return poly_const_like(ctx, index, poly_arg_invalid());
}

/* Current pm_const_buffer_folding removes a deviceless replicated MSTACK
 * from INDEX because every device lane contains the same value. */
static PolyUOp *index_deviceless_mstack(
    PolyCtx *ctx,
    PolyUOp *index,
    const PolyBindings *bindings
) {
  (void)bindings;
  if (!index || index->op != POLY_OP_INDEX || index->n_src < 1 ||
      !index->src[0] || index->src[0]->op != POLY_OP_MSTACK ||
      index->src[0]->n_src < 1)
    return NULL;
  PolyUOp *stack = index->src[0];
  PolyUOp *value = stack->src[0];
  for (int i = 1; i < stack->n_src; i++)
    if (stack->src[i] != value) return NULL;
  if (poly_uop_device_uop_cached(ctx, value, NULL)) return NULL;
  PolyUOp **src = malloc((size_t)index->n_src * sizeof(*src));
  if (!src) return NULL;
  src[0] = value;
  for (int i = 1; i < index->n_src; i++) src[i] = index->src[i];
  PolyUOp *ret = poly_uop_replace_src(ctx, index, src);
  free(src);
  return ret;
}

static _Thread_local PolyPatternMatcher *g_pm_const_buffer_folding = NULL;
PolyPatternMatcher *poly_pm_const_buffer_folding(void) {
  if (g_pm_const_buffer_folding) return g_pm_const_buffer_folding;
  PolyRule rules[] = {
      {poly_upat_allow_any_len(poly_upat_op(POLY_OP_STAGE, NULL, 0, "stage")),
       cleanup_dead_axes},
      {poly_upat_allow_any_len(poly_upat_op(POLY_OP_STAGE, NULL, 0, "stage")),
       remove_noop_bufferize},
      {poly_upat_allow_any_len(poly_upat_op(POLY_OP_STAGE, NULL, 0, "stage")),
       fold_const_buffer},
      {poly_upat_allow_any_len(poly_upat_op(POLY_OP_INDEX, NULL, 0, "index")),
       fold_const_buffer},
      {poly_upat_allow_any_len(poly_upat_op(POLY_OP_INDEX, NULL, 0, "index")),
       index_after_all_invalid},
      {poly_upat_allow_any_len(poly_upat_op(POLY_OP_NOOP, NULL, 0, "noop")),
       fold_const_buffer},
      {poly_upat_allow_any_len(poly_upat_op(POLY_OP_INDEX, NULL, 0, "index")),
       index_deviceless_mstack},
  };
  PolyPatternMatcher *local = poly_pm_new(rules, (int)(sizeof(rules) / sizeof(rules[0])));
  g_pm_const_buffer_folding =
      poly_pm_thread_cache(poly_pm_concat(poly_pm_mops(), local));
  poly_pm_destroy(local);
  return g_pm_const_buffer_folding;
}

/* pm_remove_bufferize helpers */

/* Current Tinygrad remove_bufferize keeps the STAGE once red_gate observes
 * more than three distinct storage/effect identities. */
enum { REMOVE_BUFFERIZE_MAX_ACCESSED = 3 };

/* Exact port of tinygrad's red_gate in pm_remove_bufferize.
 * AFTER contributes one buf_uop and stops, global STAGE/BUFFERIZE and MSTACK
 * contribute one identity and stop, STORE stops without contributing, and
 * PARAM contributes its identity while allowing its shape metadata traversal.
 *
 * Returns the unique identity count and an allocated REDUCE list. Allocation
 * failure returns one past the acceptance threshold so removal fails closed.
 */
static int poly_red_gate_collect(
    PolyCtx *ctx,
    PolyUOp *root,
    PolyUOp ***reduces_out,
    int *n_reduces
) {
  (void)ctx; /* arena not needed here — iterative DFS uses malloc/free */

  int cap = 64;
  PolyUOp **stack = malloc(cap * sizeof(PolyUOp *));
  if (!stack) {
    *reduces_out = NULL;
    *n_reduces = 0;
    return 0;
  }
  int top = 0;

  PolyMap *visited = poly_map_new(64);
  PolyMap *accessed_seen = poly_map_new(16);
  if (!visited || !accessed_seen) {
    if (visited) poly_map_destroy(visited);
    if (accessed_seen) poly_map_destroy(accessed_seen);
    free(stack);
    *reduces_out = NULL;
    *n_reduces = 0;
    return 0;
  }
  int reduces_cap = 16;
  PolyUOp **reduces = malloc((size_t)reduces_cap * sizeof(*reduces));
  if (!reduces) {
    poly_map_destroy(visited);
    poly_map_destroy(accessed_seen);
    free(stack);
    *reduces_out = NULL;
    *n_reduces = 0;
    return REMOVE_BUFFERIZE_MAX_ACCESSED + 1;
  }
  int accessed = 0;
  *n_reduces = 0;

  stack[top++] = root;
  while (top > 0) {
    PolyUOp *u = stack[--top];
    if (rmap_get(visited, u)) continue;
    rmap_set(visited, u, u);

    PolyUOp *access_identity = NULL;
    if (u->op == POLY_OP_AFTER) {
      access_identity = poly_uop_buf_uop(ctx, u);
      if (access_identity &&
          !poly_map_get(
              accessed_seen, poly_ptr_hash(access_identity), access_identity, poly_ptr_eq
          )) {
        poly_map_set(
            accessed_seen, poly_ptr_hash(access_identity), access_identity, access_identity,
            poly_ptr_eq
        );
        accessed++;
      }
      continue;
    }
    if ((u->op == POLY_OP_STAGE && poly_bufferize_arg_addrspace(u->arg) == POLY_ADDR_GLOBAL) ||
        u->op == POLY_OP_MSTACK) {
      access_identity = u;
      if (!poly_map_get(
              accessed_seen, poly_ptr_hash(access_identity), access_identity, poly_ptr_eq
          )) {
        poly_map_set(
            accessed_seen, poly_ptr_hash(access_identity), access_identity, access_identity,
            poly_ptr_eq
        );
        accessed++;
      }
      continue;
    }
    if (u->op == POLY_OP_STORE) continue;
    if (u->op == POLY_OP_PARAM) {
      access_identity = u;
      if (!poly_map_get(
              accessed_seen, poly_ptr_hash(access_identity), access_identity, poly_ptr_eq
          )) {
        poly_map_set(
            accessed_seen, poly_ptr_hash(access_identity), access_identity, access_identity,
            poly_ptr_eq
        );
        accessed++;
      }
    }
    if (u->op == POLY_OP_REDUCE) {
      if (*n_reduces == reduces_cap) {
        reduces_cap *= 2;
        PolyUOp **grown = realloc(reduces, (size_t)reduces_cap * sizeof(*reduces));
        if (!grown) {
          free(reduces);
          free(stack);
          poly_map_destroy(visited);
          poly_map_destroy(accessed_seen);
          *reduces_out = NULL;
          *n_reduces = 0;
          return REMOVE_BUFFERIZE_MAX_ACCESSED + 1;
        }
        reduces = grown;
      }
      reduces[(*n_reduces)++] = u;
    }

    /* Push sources in reverse order for consistent DFS ordering. */
    for (int i = u->n_src - 1; i >= 0; i--) {
      PolyUOp *src = u->src[i];
      if (!rmap_get(visited, src)) {
        if (top >= cap) {
          cap *= 2;
          void *tmp = realloc(stack, cap * sizeof(PolyUOp *));
          if (!tmp) {
            free(reduces);
            free(stack);
            poly_map_destroy(visited);
            poly_map_destroy(accessed_seen);
            *reduces_out = NULL;
            *n_reduces = 0;
            return REMOVE_BUFFERIZE_MAX_ACCESSED + 1;
          }
          stack = tmp;
        }
        stack[top++] = src;
      }
    }
  }

  free(stack);
  poly_map_destroy(visited);
  poly_map_destroy(accessed_seen);
  *reduces_out = reduces;
  return accessed;
}

/* Current Tinygrad schedule/rangeify.py:remove_bufferize, default PCONTIG=0
 * path.  Non-default PCONTIG remains registered parity debt. */
static PolyUOp *remove_bufferize(
    PolyCtx *ctx,
    PolyUOp *index,
    const PolyBindings *bindings
) {
  (void)bindings;
  if (!index || index->op != POLY_OP_INDEX || index->n_src < 1 ||
      !index->src[0] || index->src[0]->op != POLY_OP_STAGE)
    return NULL;
  PolyUOp *stage = index->src[0];
  if (stage->n_src != index->n_src) return NULL;
  for (int i = 1; i < stage->n_src; i++)
    if (stage->src[i]->op != POLY_OP_RANGE && stage->src[i]->op != POLY_OP_CONST)
      return NULL;
  PolyUOp *value = stage->src[0];
  if (rangeify_always_run(value) || !poly_bufferize_arg_removable(stage->arg)) return NULL;

  PolyUOp **reduces = NULL;
  int n_reduces = 0;
  int accessed = poly_red_gate_collect(ctx, value, &reduces, &n_reduces);
  if (accessed > REMOVE_BUFFERIZE_MAX_ACCESSED) {
    free(reduces);
    return NULL;
  }

  bool buffer_in_reduce = false;
  for (int i = 0; i < n_reduces && !buffer_in_reduce; i++) {
    if (!reduces[i] || reduces[i]->n_src < 1) continue;
    int n_topo = 0;
    PolyUOp **topo = poly_toposort_alloc(ctx, reduces[i]->src[0], &n_topo);
    for (int j = 0; j < n_topo; j++) {
      PolyOps op = topo[j]->op;
      if (op == POLY_OP_PARAM || op == POLY_OP_STAGE || op == POLY_OP_AFTER) {
        buffer_in_reduce = true;
        break;
      }
    }
    poly_toposort_free(topo);
  }
  free(reduces);
  if (buffer_in_reduce) return NULL;

  int n_ranges = stage->n_src - 1;
  PolyUOp **from = n_ranges ? malloc((size_t)n_ranges * sizeof(*from)) : NULL;
  PolyUOp **to = n_ranges ? malloc((size_t)n_ranges * sizeof(*to)) : NULL;
  if (n_ranges && (!from || !to)) {
    free(from);
    free(to);
    return NULL;
  }
  int n_sub = 0;
  for (int i = 0; i < n_ranges; i++) {
    PolyUOp *replacement = index->src[i + 1];
    if (stage->src[i + 1]->op == POLY_OP_CONST ||
        (replacement->op == POLY_OP_CONST &&
         replacement->arg.kind == POLY_ARG_INVALID))
      continue;
    from[n_sub] = stage->src[i + 1];
    to[n_sub++] = replacement;
  }
  PolyUOp *ret = n_sub ? poly_uop_substitute(ctx, value, from, to, n_sub) : value;
  free(from);
  free(to);
  return ret;
}

static PolyUOp *store_self_noop(
    PolyCtx *ctx,
    PolyUOp *root,
    const PolyBindings *bindings
) {
  (void)bindings;
  if (root->op == POLY_OP_STORE && root->n_src == 2 && root->src[0] == root->src[1])
    return poly_uop0(ctx, POLY_OP_NOOP, POLY_VOID, poly_arg_none());
  if (root->op == POLY_OP_END && root->n_src >= 1 &&
      root->src[0]->op == POLY_OP_NOOP)
    return root->src[0];
  return NULL;
}

static _Thread_local PolyPatternMatcher *g_pm_remove_bufferize = NULL;
static PolyPatternMatcher *poly_pm_remove_bufferize(void) {
  if (g_pm_remove_bufferize) return g_pm_remove_bufferize;
  PolyRule rules[] = {
      {poly_upat_allow_any_len(poly_upat_op(POLY_OP_INDEX, NULL, 0, "index")),
       remove_bufferize},
      {poly_upat_allow_any_len(poly_upat_op(POLY_OP_STORE, NULL, 0, "store")),
       store_self_noop},
      {poly_upat_allow_any_len(poly_upat_op(POLY_OP_END, NULL, 0, "end")),
       store_self_noop},
  };
  g_pm_remove_bufferize = poly_pm_thread_cache(
      poly_pm_new(rules, (int)(sizeof(rules) / sizeof(rules[0])))
  );
  return g_pm_remove_bufferize;
}

static _Thread_local PolyPatternMatcher *g_pm_rangeify_cleanup = NULL;
static PolyPatternMatcher *poly_pm_rangeify_cleanup(void) {
  if (g_pm_rangeify_cleanup) return g_pm_rangeify_cleanup;
  PolyPatternMatcher *symbolic_reduce =
      poly_pm_concat(poly_symbolic(), poly_pm_reduce_simplify());
  PolyPatternMatcher *with_const =
      poly_pm_concat(symbolic_reduce, poly_pm_const_buffer_folding());
  g_pm_rangeify_cleanup = poly_pm_thread_cache(
      poly_pm_concat(with_const, poly_pm_remove_bufferize())
  );
  poly_pm_destroy(symbolic_reduce);
  poly_pm_destroy(with_const);
  return g_pm_rangeify_cleanup;
}

/* Current Tinygrad schedule/rangeify.py:_limit_bufs visitor.  STAGE, AFTER,
 * PARAM, MSELECT, and MSTACK are distinct kernel arguments; their sources are
 * intentionally not traversed. */
static int poly_limit_bufs_count(PolyUOp *root) {
  int cap = 32;
  PolyUOp **stack = malloc((size_t)cap * sizeof(*stack));
  PolyMap *visited = poly_map_new(64);
  if (!stack || !visited) {
    free(stack);
    if (visited) poly_map_destroy(visited);
    return -1;
  }
  int top = 0;
  int count = 0;

  stack[top++] = root;
  while (top > 0) {
    PolyUOp *u = stack[--top];
    if (rmap_get(visited, u)) continue;
    rmap_set(visited, u, u);

    if (u->op == POLY_OP_STAGE || u->op == POLY_OP_AFTER || u->op == POLY_OP_PARAM ||
        u->op == POLY_OP_MSELECT || u->op == POLY_OP_MSTACK) {
      count++;
      continue;
    }

    for (int i = u->n_src - 1; i >= 0; i--) {
      if (!rmap_get(visited, u->src[i])) {
        if (top >= cap) {
          cap *= 2;
          PolyUOp **grown = realloc(stack, (size_t)cap * sizeof(*stack));
          if (!grown) {
            free(stack);
            poly_map_destroy(visited);
            return -1;
          }
          stack = grown;
        }
        stack[top++] = u->src[i];
      }
    }
  }

  free(stack);
  poly_map_destroy(visited);
  return count;
}

/* Current Tinygrad DEVICE_MAX_BUFS in schedule/rangeify.py.  The optional C
 * environment override is the existing spelling of MAX_KERNEL_BUFFERS. */
static int poly_limit_bufs_max_for_device(PolyUOp *device) {
  const char *override = getenv("POLY_MAX_KERNEL_BUFFERS");
  int configured = override ? atoi(override) : 0;
  if (configured != 0) return configured;
  if (!device || device->op != POLY_OP_DEVICE) return 0;
  const char *name = device->arg.kind == POLY_ARG_STRING
                         ? device->arg.str
                         : device->arg.kind == POLY_ARG_STRING_TUPLE &&
                                   device->arg.string_tuple.n > 0
                               ? device->arg.string_tuple.vals[0]
                               : NULL;
  if (!name) return 0;
  /* tinygrad@2026-08-22/a9069c177a9d keeps CPU renderer selections such as
   * CPU:X86 on device CPU. Polygrad exposes those renderers as C devices. */
  if ((strncmp(name, "CPU", 3) == 0 && (name[3] == '\0' || name[3] == ':')) ||
      strcmp(name, "X86") == 0 || strcmp(name, "INTERP") == 0 ||
      strcmp(name, "WASM") == 0)
    return 31;
  if (strncmp(name, "WEBGPU", 6) == 0 && (name[6] == '\0' || name[6] == ':')) return 8;
  if (strncmp(name, "METAL", 5) == 0 && (name[5] == '\0' || name[5] == ':')) return 31;
  return 0;
}

/* Current Tinygrad schedule/rangeify.py:pm_limit_bufs. */
static PolyUOp *poly_limit_bufs(PolyCtx *ctx, PolyUOp *sink) {
  int n_topo;
  PolyUOp **topo = poly_toposort_ex_alloc(ctx, sink, &n_topo, NULL, false);
  if (!topo) return sink;
  PolyMap *rmap = poly_map_new(n_topo < 16 ? 16 : (uint32_t)n_topo);
  PolyMap *device_memo = poly_map_new(n_topo < 16 ? 16 : (uint32_t)n_topo);
  if (!rmap || !device_memo) {
    if (rmap) poly_map_destroy(rmap);
    if (device_memo) poly_map_destroy(device_memo);
    poly_toposort_free(topo);
    return sink;
  }
  int64_t next_range = 0;
  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    if (u->op == POLY_OP_RANGE && u->arg.kind == POLY_ARG_RANGE &&
        poly_range_axis_id(u->arg) >= next_range) {
      if (poly_range_axis_id(u->arg) == INT64_MAX) {
        poly_map_destroy(device_memo);
        poly_map_destroy(rmap);
        poly_toposort_free(topo);
        return sink;
      }
      next_range = poly_range_axis_id(u->arg) + 1;
    }
  }

  for (int t = 0; t < n_topo; t++) {
    PolyUOp *u = topo[t];

    bool src_changed = false;
    PolyUOp *ns_buf[POLY_MAX_DIMS + 4];
    PolyUOp **ns = ((uint32_t)u->n_src > (uint32_t)(sizeof ns_buf / sizeof *ns_buf))
                       ? malloc(u->n_src * sizeof(PolyUOp *))
                       : ns_buf;
    for (int i = 0; i < u->n_src; i++) {
      PolyUOp *m = rmap_get(rmap, u->src[i]);
      ns[i] = m ? m : u->src[i];
      if (ns[i] != u->src[i]) src_changed = true;
    }

    PolyUOp *result = NULL;

    if (poly_opset_has(POLY_GROUP_BINARY, u->op) ||
        poly_opset_has(POLY_GROUP_TERNARY, u->op)) {
      PolyUOp *check_node = src_changed ? poly_uop_replace_src(ctx, u, ns) : u;
      PolyUOp *root_device = poly_bufferize_device_hint(ctx, check_node, device_memo);
      int max_bufs = poly_limit_bufs_max_for_device(root_device);
      if (!max_bufs) {
        if (src_changed) rmap_set(rmap, u, check_node);
        if (ns != ns_buf) free(ns);
        continue;
      }
      int buf_count = max_bufs ? poly_limit_bufs_count(check_node) : 0;

      if (buf_count > max_bufs - 1) {
        bool any_wrapped = false;
        for (int i = 0; i < u->n_src; i++) {
          PolyUOp *s = ns[i];
          if (!poly_opset_has(POLY_GROUP_ELEMENTWISE, s->op)) continue;
          PolyUOp *source_device = poly_bufferize_device_hint(ctx, s, device_memo);
          if (!source_device) continue;

          PolyUOp *orig_rngs[POLY_MAX_DIMS];
          int n_rngs = poly_uop_ranges(ctx, s, orig_rngs, POLY_MAX_DIMS);
          PolyUOp *end_rngs[POLY_MAX_DIMS];
          for (int d = 0; d < n_rngs; d++) {
            PolyUOp *r = orig_rngs[d];
            if (poly_range_axis_type(r->arg) == POLY_AXIS_DEVICE) {
              end_rngs[d] = r;
              continue;
            }
            end_rngs[d] = poly_uop_tagged_arg(
                ctx, r->op, r->dtype, r->src, r->n_src,
                poly_arg_range(next_range++, POLY_AXIS_WEAK), r->tag, r->tag_arg
            );
          }

          PolyUOp *sub_s = n_rngs
                               ? poly_uop_substitute(ctx, s, orig_rngs, end_rngs, n_rngs)
                               : s;
          PolyUOp *buf_src[POLY_MAX_DIMS + 1];
          buf_src[0] = sub_s;
          for (int d = 0; d < n_rngs; d++)
            buf_src[1 + d] = end_rngs[d];
          PolyUOp *bufferize = poly_uop(
              ctx, POLY_OP_STAGE, s->dtype, buf_src, n_rngs + 1,
              poly_bufferize_opts_for_device(source_device, POLY_ADDR_GLOBAL, false)
          );

          PolyUOp *idx_src[POLY_MAX_DIMS + 1];
          idx_src[0] = bufferize;
          for (int d = 0; d < n_rngs; d++)
            idx_src[1 + d] = orig_rngs[d];
          ns[i] = poly_uop(
              ctx, POLY_OP_INDEX, s->dtype, idx_src, n_rngs + 1,
              poly_arg_none()
          );
          any_wrapped = true;
        }

        if (any_wrapped) result = poly_uop_replace_src(ctx, u, ns);
      }
    }

    if (!result && src_changed) result = poly_uop_replace_src(ctx, u, ns);
    if (result && result != u) rmap_set(rmap, u, result);
    if (ns != ns_buf) free(ns);
  }

  PolyUOp *new_sink = rmap_get(rmap, sink);
  if (device_memo) poly_map_destroy(device_memo);
  poly_map_destroy(rmap);
  poly_toposort_free(topo);
  return new_sink ? new_sink : sink;
}

typedef struct {
  int64_t next_slot;
} CopyToStoreContext;

static bool copy_has_buffer_identity(PolyUOp *u) {
  while (u) {
    if (u->op == POLY_OP_RESHAPE || u->op == POLY_OP_UNSHARD ||
        u->op == POLY_OP_MSELECT || u->op == POLY_OP_AFTER) {
      if (u->n_src < 1) return false;
      u = u->src[0];
      continue;
    }
    return u->op == POLY_OP_BUFFER || u->op == POLY_OP_PARAM;
  }
  return false;
}

static PolyUOp *copy_flatten(PolyCtx *ctx, PolyUOp *u) {
  PolyShape shape = poly_uop_max_shape_cached(ctx, u);
  int64_t numel = shape.ndim >= 0 ? poly_shape_numel(shape) : -1;
  if (numel < 0) return NULL;
  if (shape.ndim == 1 && shape.dims[0] == numel) return u;
  return poly_reshape(ctx, u, &numel, 1);
}

static PolyUOp *copy_reshape_to_source(PolyCtx *ctx, PolyUOp *u, PolyUOp *source) {
  int ndim = poly_uop_ndim(ctx, source);
  if (ndim < 0 || ndim > POLY_MAX_DIMS) return NULL;
  PolyShape current = poly_uop_max_shape_cached(ctx, u);
  PolyShape target = poly_uop_max_shape_cached(ctx, source);
  if (poly_shape_eq(current, target)) return u;
  PolyUOp *dims[POLY_MAX_DIMS];
  for (int i = 0; i < ndim; i++) {
    dims[i] = poly_uop_shape_dim(ctx, source, i);
    if (!dims[i] && target.dims)
      dims[i] = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(target.dims[i]));
    if (!dims[i]) return NULL;
  }
  return poly_reshape_uop(ctx, u, dims, ndim);
}

/* Current tinygrad schedule/rangeify.py:565-581. COPY is Tensor IR only: it
 * becomes a destination BUFFER plus a plain STORE before rangeification.  A
 * later schedule pass can then identify exact parameter-to-parameter stores
 * as runtime copies without allowing COPY to enter shared codegen. */
static PolyUOp *convert_copy_to_store(
    PolyCtx *ctx,
    CopyToStoreContext *cctx,
    PolyUOp *copy,
    PolyUOp *existing_buf
) {
  if (!ctx || !cctx || !copy || copy->op != POLY_OP_COPY || copy->n_src != 1 ||
      (copy->arg.kind != POLY_ARG_STRING && copy->arg.kind != POLY_ARG_STRING_TUPLE))
    return NULL;

  PolyUOp *input_src = copy->src[0];
  if (!copy_has_buffer_identity(input_src)) input_src = poly_contiguous(ctx, input_src);
  input_src = input_src ? copy_flatten(ctx, input_src) : NULL;
  if (!input_src) return NULL;

  if (existing_buf) {
    if (!copy_has_buffer_identity(existing_buf)) return NULL;
    PolyUOp *flat_buf = copy_flatten(ctx, existing_buf);
    return flat_buf ? poly_store_val(ctx, flat_buf, input_src) : NULL;
  }

  int64_t size = poly_uop_max_numel(ctx, input_src);
  if (size < 0) return NULL;
  PolyUOp *device = copy->arg.kind == POLY_ARG_STRING
                        ? poly_device_uop_from_name(ctx, copy->arg.str)
                        : poly_device_uop_from_names(
                              ctx, copy->arg.string_tuple.vals,
                              copy->arg.string_tuple.n
                          );
  PolyUOp *buf = device
                     ? poly_uop_new_buffer(
                           ctx, device, size, copy->dtype, cctx->next_slot++
                       )
                     : NULL;
  PolyUOp *store = buf ? poly_store_val(ctx, buf, input_src) : NULL;
  PolyUOp *after = store
                       ? poly_uop2(ctx, POLY_OP_AFTER, copy->dtype, buf, store, poly_arg_none())
                       : NULL;
  return after ? copy_reshape_to_source(ctx, after, copy) : NULL;
}

static PolyUOp *convert_store_copy(
    PolyCtx *ctx,
    PolyUOp *root,
    const PolyBindings *bindings
) {
  (void)bindings;
  if (!root || root->op != POLY_OP_STORE || root->n_src != 2 || !root->src[1] ||
      root->src[1]->op != POLY_OP_COPY)
    return NULL;
  return convert_copy_to_store(
      ctx, (CopyToStoreContext *)poly_graph_rewrite_userctx(), root->src[1], root->src[0]
  );
}

static PolyUOp *convert_copy(
    PolyCtx *ctx,
    PolyUOp *root,
    const PolyBindings *bindings
) {
  (void)bindings;
  if (!root || root->op != POLY_OP_COPY) return NULL;
  return convert_copy_to_store(
      ctx, (CopyToStoreContext *)poly_graph_rewrite_userctx(), root, NULL
  );
}

static PolyPatternMatcher *poly_pm_copy_to_store(void) {
  static _Thread_local PolyPatternMatcher *pm = NULL;
  if (pm) return pm;
  PolyRule rules[] = {
      {poly_upat_allow_any_len(poly_upat_op(POLY_OP_STORE, NULL, 0, "store")),
       convert_store_copy},
      {poly_upat_allow_any_len(poly_upat_op(POLY_OP_COPY, NULL, 0, "copy")),
       convert_copy},
  };
  pm = poly_pm_thread_cache(poly_pm_new(rules, (int)(sizeof(rules) / sizeof(rules[0]))));
  return pm;
}

static PolyUOp *poly_apply_copy_to_store(PolyCtx *ctx, PolyUOp *sink) {
  CopyToStoreContext cctx = {0};
  return poly_graph_rewrite_ctx_ex2(
      ctx, sink, poly_pm_copy_to_store(), &cctx, true, false
  );
}

static PolyUOp *add_param_range_tag(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  if (!root || (root->op != POLY_OP_RANGE && root->op != POLY_OP_PARAM)) return NULL;
  int32_t tag = root->op == POLY_OP_RANGE ? POLY_RANGEIFY_RANGE_TAG : POLY_RANGEIFY_PARAM_TAG;
  if (root->tag == tag) return NULL;
  /* Current tinygrad rangeify.py:pm_add_param_range_tags marks every
   * pre-split PARAM and RANGE with tag=(). The split pass rewrites only those
   * tagged occurrences and creates untagged replacements, preventing a
   * bottom-up fixed point from debuffering or renumbering them again. */
  return poly_uop_tagged(
      ctx, root->op, root->dtype, root->src, root->n_src, root->arg, tag
  );
}

static PolyPatternMatcher *poly_pm_add_param_range_tags(void) {
  static _Thread_local PolyPatternMatcher *pm = NULL;
  if (pm) return pm;
  PolyOpSet ops = poly_opset_add(poly_opset_add((PolyOpSet){{0, 0}}, POLY_OP_PARAM), POLY_OP_RANGE);
  PolyRule rules[] = {
      {poly_upat_ops(ops, NULL, 0, "x"), add_param_range_tag},
  };
  pm = poly_pm_thread_cache(poly_pm_new(rules, (int)(sizeof(rules) / sizeof(rules[0]))));
  return pm;
}

PolyUOp *poly_get_kernel_graph(PolyCtx *ctx, PolyUOp *tensor_sink) {
  if (!ctx || !tensor_sink || tensor_sink->op != POLY_OP_SINK) {
    fprintf(stderr, "polygrad: poly_get_kernel_graph: expected SINK\n");
    return NULL;
  }

  bool timing = poly_debug_at_least(2);
  double t0 = timing ? poly_now_ms() : 0.0;
  if (timing) {
    fprintf(
        stderr, "[polygrad:get_kernel_graph] begin sink=%p n_src=%d\n", (void *)tensor_sink,
        tensor_sink->n_src
    );
    fflush(stderr);
  }

  /* Current Tinygrad schedule/rangeify.py:get_kernel_graph. */
  tensor_sink = poly_apply_multi_pm(ctx, tensor_sink);
  if (!tensor_sink) return NULL;
  if (timing) {
    fprintf(stderr, "[polygrad:get_kernel_graph] stage earliest_rewrites begin\n");
    fflush(stderr);
  }
  tensor_sink = poly_apply_earliest_rewrites(ctx, tensor_sink);
  if (!tensor_sink) return NULL;
  tensor_sink = poly_apply_copy_to_store(ctx, tensor_sink);
  double t_earliest = timing ? poly_now_ms() : 0.0;
  if (timing) {
    fprintf(
        stderr, "[polygrad:get_kernel_graph] stage earliest_rewrites done %.3fms\n", t_earliest - t0
    );
    fflush(stderr);
  }

  if (timing) {
    fprintf(stderr, "[polygrad:get_kernel_graph] stage run_rangeify begin\n");
    fflush(stderr);
  }
  PolyUOp *rangeified = poly_run_rangeify(ctx, tensor_sink, poly_debug_at_least(4));
  if (!rangeified) return NULL;
  double t_rangeify = timing ? poly_now_ms() : 0.0;
  if (timing) {
    fprintf(
        stderr, "[polygrad:get_kernel_graph] stage run_rangeify done %.3fms\n",
        t_rangeify - t_earliest
    );
    fflush(stderr);
  }
  if (timing) {
    fprintf(stderr, "[polygrad:get_kernel_graph] stage symbolic+reduce_collapse+debuf begin\n");
    fflush(stderr);
  }
  PolyUOp *cleaned = poly_graph_rewrite_ctx_ex2(
      ctx, rangeified, poly_pm_rangeify_cleanup(), NULL, false, false
  );
  if (!cleaned) return NULL;
  double t_cleanup = timing ? poly_now_ms() : 0.0;
  if (timing) {
    fprintf(
        stderr, "[polygrad:get_kernel_graph] stage symbolic+reduce_collapse+debuf done %.3fms\n",
        t_cleanup - t_rangeify
    );
    fflush(stderr);
  }
  if (timing) {
    fprintf(stderr, "[polygrad:get_kernel_graph] stage limit_bufs begin\n");
    fflush(stderr);
  }
  PolyUOp *limited = poly_limit_bufs(ctx, cleaned);
  double t_limit = timing ? poly_now_ms() : 0.0;
  if (timing) {
    fprintf(
        stderr, "[polygrad:get_kernel_graph] stage limit_bufs done %.3fms\n", t_limit - t_cleanup
    );
    fflush(stderr);
  }
  if (timing) {
    fprintf(stderr, "[polygrad:get_kernel_graph] stage stage_to_store begin\n");
    fflush(stderr);
  }
  int n_topo = 0;
  PolyUOp **topo = poly_toposort_ex_alloc(ctx, limited, &n_topo, NULL, false);
  if (!topo && n_topo != 0) return NULL;
  int slot_counter = 0;
  /* Current get_kernel_graph starts temporary slots after existing global
   * BUFFER ParamArg slots (schedule/rangeify.py:603-606). */
  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    if (!u || u->op != POLY_OP_BUFFER || u->arg.kind != POLY_ARG_PARAM ||
        !u->arg.param || u->arg.param->addrspace != POLY_ADDR_GLOBAL)
      continue;
    if (u->arg.param->slot >= slot_counter && u->arg.param->slot < INT_MAX)
      slot_counter = (int)u->arg.param->slot + 1;
  }
  poly_toposort_free(topo);
  PolyPatternMatcher *stage_to_store =
      poly_pm_concat(poly_pm_add_buffers(), poly_pm_add_param_range_tags());
  PolyUOp *kernel_graph = stage_to_store
                              ? poly_graph_rewrite_ctx_ex2(
                                    ctx, limited, stage_to_store, &slot_counter, true, false
                                )
                              : NULL;
  poly_pm_destroy(stage_to_store);
  double t_stage_to_store = timing ? poly_now_ms() : 0.0;
  if (timing) {
    fprintf(
        stderr, "[polygrad:get_kernel_graph] stage stage_to_store done %.3fms\n",
        t_stage_to_store - t_limit
    );
    fflush(stderr);
  }
  if (timing) {
    fprintf(stderr, "[polygrad:get_kernel_graph] stage split_kernels begin\n");
    fflush(stderr);
  }
  PolySplitKernelsContext split_ctx = {0};
  kernel_graph = poly_graph_rewrite_ctx_ex2(
      ctx, kernel_graph, poly_split_kernels(), &split_ctx, true, false
  );
  double t_split = timing ? poly_now_ms() : 0.0;
  if (!kernel_graph || split_ctx.failed) return NULL;
  if (timing) {
    fprintf(
        stderr, "[polygrad:get_kernel_graph] stage split_kernels done %.3fms\n",
        t_split - t_stage_to_store
    );
    fflush(stderr);
  }
  kernel_graph = poly_graph_rewrite_ctx_ex2(
      ctx, kernel_graph, poly_pm_no_indexing_calls(), NULL, false, false
  );
  double t_no_index = timing ? poly_now_ms() : 0.0;
  if (!kernel_graph || !poly_type_verify_kernel_graph(ctx, kernel_graph)) return NULL;
  if (timing) {
    int n_topo = 0;
    PolyUOp **topo_dbg = poly_toposort_alloc(ctx, kernel_graph, &n_topo);
    poly_toposort_free(topo_dbg);
    fprintf(
        stderr,
        "[polygrad:get_kernel_graph] done earliest=%.3fms rangeify=%.3fms "
        "symbolic_reduce_debuf=%.3fms limit=%.3fms "
        "stage_to_store=%.3fms split=%.3fms no_index=%.3fms "
        "total=%.3fms topo=%d\n",
        t_earliest - t0, t_rangeify - t_earliest, t_cleanup - t_rangeify,
        t_limit - t_cleanup, t_stage_to_store - t_limit,
        t_split - t_stage_to_store, t_no_index - t_split, t_no_index - t0, n_topo
    );
    fflush(stderr);
  }
  return kernel_graph;
}
