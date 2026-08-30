/* C implementation of tinygrad schedule/multi.py. */

#include "schedule/multi.h"

#include "ctx.h"
#include "device.h"
#include "frontend_internal.h"
#include "uop/upat.h"
#include "schedule/indexing.h"
#include "tensor.h"
#include "utils.h"

#include <limits.h>
#include <stdlib.h>

static PolyUOp *multi_index_const(PolyCtx *ctx, int64_t value) {
  return poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(value));
}

static PolyUOp *multi_clone(
    PolyCtx *ctx,
    PolyUOp *u,
    PolyUOp **src,
    int n_src
) {
  if (u->tag != 0 || u->tag_arg.kind != POLY_ARG_NONE)
    return poly_uop_tagged_arg(ctx, u->op, u->dtype, src, n_src, u->arg, u->tag, u->tag_arg);
  return poly_uop(ctx, u->op, u->dtype, src, n_src, u->arg);
}

static PolyUOp *multi_map_get(PolyMap *map, PolyUOp *key) {
  return poly_map_get(map, poly_ptr_hash(key), key, poly_ptr_eq);
}

static void multi_map_set(PolyMap *map, PolyUOp *key, PolyUOp *value) {
  poly_map_set(map, poly_ptr_hash(key), key, value, poly_ptr_eq);
}
static bool multi_pm_is_device_range(PolyUOp *u) {
  return u && u->op == POLY_OP_RANGE &&
         poly_range_axis_type(u->arg) == POLY_AXIS_DEVICE;
}

/* Current `_apply_shrink` substitutes DEVICE RANGE occurrences in the SHRINK
 * bounds (`tinygrad/schedule/multi.py:8-12`). */
static bool multi_pm_collect_device_ranges(
    PolyCtx *ctx,
    PolyUOp *starts,
    PolyUOp *sizes,
    PolyUOp ***out_vars,
    int *out_n
) {
  *out_vars = NULL;
  *out_n = 0;
  int n_starts = 0, n_sizes = 0;
  PolyUOp **start_topo = poly_toposort_alloc(ctx, starts, &n_starts);
  PolyUOp **size_topo = poly_toposort_alloc(ctx, sizes, &n_sizes);
  if (!start_topo || !size_topo) {
    poly_toposort_free(start_topo);
    poly_toposort_free(size_topo);
    return false;
  }
  int cap = n_starts + n_sizes;
  PolyUOp **vars = cap > 0 ? malloc((size_t)cap * sizeof(*vars)) : NULL;
  if (cap > 0 && !vars) {
    poly_toposort_free(start_topo);
    poly_toposort_free(size_topo);
    return false;
  }
  PolyUOp **topos[] = {start_topo, size_topo};
  int counts[] = {n_starts, n_sizes};
  int n_vars = 0;
  for (int ti = 0; ti < 2; ti++) {
    for (int i = 0; i < counts[ti]; i++) {
      PolyUOp *candidate = topos[ti][i];
      if (!multi_pm_is_device_range(candidate)) continue;
      bool duplicate = false;
      for (int j = 0; j < n_vars; j++)
        if (vars[j] == candidate) { duplicate = true; break; }
      if (!duplicate) vars[n_vars++] = candidate;
    }
  }
  poly_toposort_free(start_topo);
  poly_toposort_free(size_topo);
  *out_vars = vars;
  *out_n = n_vars;
  return true;
}

static PolyUOp *multi_pm_local_shrink(
    PolyCtx *ctx,
    PolyUOp *base,
    PolyUOp *starts,
    PolyUOp *sizes,
    PolyUOp **device_ranges,
    int n_device_ranges,
    int device_index
) {
  PolyUOp *shape_roots[2] = {starts, sizes};
  PolyUOp *substituted[2] = {starts, sizes};
  PolyUOp **values =
      n_device_ranges > 0 ? malloc((size_t)n_device_ranges * sizeof(*values)) : NULL;
  if (n_device_ranges > 0 && !values) return NULL;
  for (int i = 0; i < n_device_ranges; i++) {
    values[i] = poly_uop0(
        ctx, POLY_OP_CONST, device_ranges[i]->dtype, poly_arg_int(device_index));
    if (!values[i]) {
      free(values);
      return NULL;
    }
  }
  if (n_device_ranges > 0 && poly_uop_substitute_many(
          ctx, shape_roots, 2, device_ranges, values, n_device_ranges, substituted
      ) != 0) {
    free(values);
    return NULL;
  }
  free(values);

  /* Pinned UOp._mop simplifies the reconstructed shape sources through
   * UOp.sink(*usrcs).simplify() before it creates SHRINK (ops.py:710-721). */
  PolyUOp *shape_sink = poly_sink_n(ctx, substituted, 2);
  PolyUOp *simplified = shape_sink ? poly_graph_rewrite(ctx, shape_sink, poly_symbolic()) : NULL;
  if (!simplified || simplified->op != POLY_OP_SINK || simplified->n_src != 2) return NULL;
  PolyUOp *shrink_src[3] = {base, simplified->src[0], simplified->src[1]};
  return poly_uop(ctx, POLY_OP_SHRINK, base->dtype, shrink_src, 3, poly_arg_none());
}

/* Exact port of pinned schedule/multi.py:mstack_early_shrink. COPY children
 * keep their explicit scalar destination; other children materialize a local
 * CONTIGUOUS after the per-device SHRINK. The returned node is the original
 * MSTACK with only its ordered sources replaced. */
static PolyUOp *multi_pm_mstack_early_shrink(
    PolyCtx *ctx,
    PolyUOp *shrink,
    PolyUOp *mstack,
    PolyUOp *starts,
    PolyUOp *sizes,
    bool *failed
) {
  *failed = false;
  if (!shrink || shrink->n_src != 3 || !mstack || mstack->op != POLY_OP_MSTACK ||
      mstack->n_src == 0)
    return NULL;
  PolyUOp **device_ranges = NULL;
  int n_device_ranges = 0;
  if (!multi_pm_collect_device_ranges(
          ctx, starts, sizes, &device_ranges, &n_device_ranges
      )) {
    *failed = true;
    return NULL;
  }
  PolyUOp **local = malloc((size_t)mstack->n_src * sizeof(*local));
  if (!local) {
    free(device_ranges);
    *failed = true;
    return NULL;
  }
  for (int i = 0; i < mstack->n_src; i++) {
    PolyUOp *child = mstack->src[i];
    PolyUOp *base = child;
    if (child && child->op == POLY_OP_COPY) {
      if (child->n_src != 1) {
        *failed = true;
        break;
      }
      base = child->src[0];
    }
    PolyUOp *local_shrink = base ? multi_pm_local_shrink(
        ctx, base, starts, sizes, device_ranges, n_device_ranges, i
    ) : NULL;
    if (!local_shrink) {
      *failed = true;
      break;
    }
    if (child->op == POLY_OP_COPY) {
      /* Current tinygrad schedule/multi.py:mstack_early_shrink rebuilds the
       * current unary COPY with x.device after applying the lane-specific
       * shrink to x.src[0]. */
      PolyUOp *target = poly_uop_device_uop_cached(ctx, child, NULL);
      local[i] = target ? poly_copy_to_device_uop(ctx, local_shrink, target) : NULL;
    } else {
      local[i] = poly_uop1(
          ctx, POLY_OP_CONTIGUOUS, local_shrink->dtype, local_shrink, poly_arg_none());
    }
    if (!local[i]) {
      *failed = true;
      break;
    }
  }
  PolyUOp *ret = *failed ? NULL :
      multi_clone(ctx, mstack, local, mstack->n_src);
  if (!ret) *failed = true;
  free(local);
  free(device_ranges);
  return ret;
}

static PolyUOp *multi_pm_tuple_device(PolyCtx *ctx, PolyUOp *u, int *n_devices) {
  if (n_devices) *n_devices = 0;
  PolyUOp *device = u ? poly_uop_device_uop_cached(ctx, u, NULL) : NULL;
  if (!device || device->op != POLY_OP_DEVICE ||
      device->arg.kind != POLY_ARG_STRING_TUPLE || device->arg.string_tuple.n <= 0)
    return NULL;
  if (n_devices) *n_devices = device->arg.string_tuple.n;
  return device;
}

/* Current `value.unshard(template.arg, template.src[1:])`. */
static PolyUOp *multi_pm_restore_sharding(
    PolyCtx *ctx,
    PolyUOp *value,
    PolyUOp *template
) {
  return value && template && template->op == POLY_OP_UNSHARD &&
                 template->arg.kind == POLY_ARG_INT_TUPLE &&
                 template->n_src == template->arg.int_tuple.n + 1
             ? poly_unshard(
                   ctx, value, template->arg.int_tuple.vals, template->src + 1,
                   template->arg.int_tuple.n)
             : NULL;
}

static PolyUOp *multi_pm_wrap_axis(
    PolyCtx *ctx,
    PolyUOp *u,
    int axis,
    PolyUOp *range
);

/* Pinned UOp._shard (uop/ops.py:654-660). Shape divisibility is proved by
 * the same symbolic rewrite used for shape expressions; an unproved shard is
 * an error, never a truncating FLOORDIV guess. */
static PolyUOp *multi_pm_shard_value(
    PolyCtx *ctx,
    PolyUOp *value,
    int axis,
    PolyUOp *sharding_range,
    bool *failed
) {
  *failed = false;
  int64_t range_vmin = 0, range_vmax = -1;
  if (!value || !sharding_range || sharding_range->op != POLY_OP_RANGE) {
    *failed = true;
    return NULL;
  }
  poly_uop_minmax(ctx, sharding_range, &range_vmin, &range_vmax);
  if (range_vmax < 0 || range_vmax >= INT_MAX) {
    *failed = true;
    return NULL;
  }
  int n_devices = (int)range_vmax + 1;
  PolyShape shape = poly_uop_max_shape_cached(ctx, value);
  if (shape.ndim == 0) return value;
  if (shape.ndim < 0 || shape.ndim > POLY_MAX_DIMS || axis < 0 || axis >= shape.ndim) {
    *failed = true;
    return NULL;
  }

  PolyUOp *dim = poly_uop_shape_dim(ctx, value, axis);
  PolyUOp *count = multi_index_const(ctx, n_devices);
  PolyUOp *mod = dim && count
                     ? poly_uop2(ctx, POLY_OP_FLOORMOD, POLY_WEAKINT, dim, count, poly_arg_none())
                     : NULL;
  mod = mod ? poly_graph_rewrite(ctx, mod, poly_symbolic()) : NULL;
  int64_t remainder = -1;
  if (!mod || poly_uop_const_i64(mod, &remainder) != 0 || remainder != 0) {
    *failed = true;
    return NULL;
  }
  PolyUOp *local_dim =
      poly_uop2(ctx, POLY_OP_FLOORDIV, POLY_WEAKINT, dim, count, poly_arg_none());
  local_dim = local_dim ? poly_graph_rewrite(ctx, local_dim, poly_symbolic()) : NULL;
  if (!local_dim) {
    *failed = true;
    return NULL;
  }

  PolyUOp *start = sharding_range
                       ? poly_uop2(
                             ctx, POLY_OP_MUL, POLY_WEAKINT, sharding_range, local_dim,
                             poly_arg_none())
                       : NULL;
  PolyUOp *starts[POLY_MAX_DIMS], *sizes[POLY_MAX_DIMS];
  for (int i = 0; i < shape.ndim; i++) {
    starts[i] = i == axis ? start : multi_index_const(ctx, 0);
    sizes[i] = i == axis ? local_dim : poly_uop_shape_dim(ctx, value, i);
    if (!starts[i] || !sizes[i]) {
      *failed = true;
      return NULL;
    }
  }
  PolyUOp *ret = poly_shrink_uop(ctx, value, starts, sizes, shape.ndim);
  if (!ret) *failed = true;
  return ret;
}

static bool multi_pm_sharding(
    PolyCtx *ctx,
    PolyUOp *multi,
    int64_t *axes,
    PolyUOp **ranges,
    int *counts,
    int *n_sharding
);
static bool multi_pm_shape_equal(PolyCtx *ctx, PolyUOp *a, PolyUOp *b);
static PolyUOp *multi_pm_shard_subview(
    PolyCtx *ctx,
    PolyUOp *full,
    PolyUOp *multi,
    bool *failed
);

/* Current schedule/multi.py:_shard_idx substitutes every DEVICE RANGE in a
 * sharding expression with the concrete tuple occurrence index. */
static bool multi_pm_shard_idx(PolyCtx *ctx, PolyUOp *range, int device_index, int *out) {
  PolyUOp **device_ranges = NULL;
  int n_device_ranges = 0;
  if (!multi_pm_collect_device_ranges(
          ctx, range, range, &device_ranges, &n_device_ranges))
    return false;
  if (n_device_ranges == 0) {
    free(device_ranges);
    *out = 0;
    return true;
  }
  PolyUOp **values = malloc((size_t)n_device_ranges * sizeof(*values));
  if (!values) {
    free(device_ranges);
    return false;
  }
  bool ok = true;
  for (int i = 0; i < n_device_ranges; i++) {
    values[i] = poly_uop0(
        ctx, POLY_OP_CONST, device_ranges[i]->dtype, poly_arg_int(device_index));
    if (!values[i]) ok = false;
  }
  PolyUOp *root = range, *rewritten = range;
  if (ok && poly_uop_substitute_many(
                ctx, &root, 1, device_ranges, values, n_device_ranges,
                &rewritten) != 0)
    ok = false;
  rewritten = ok ? poly_graph_rewrite(ctx, rewritten, poly_symbolic()) : NULL;
  int64_t value = 0;
  if (!rewritten || poly_uop_const_i64(rewritten, &value) != 0 ||
      value < 0 || value > INT_MAX)
    ok = false;
  free(values);
  free(device_ranges);
  if (ok) *out = (int)value;
  return ok;
}

typedef struct {
  int index[POLY_MAX_DIMS];
  PolyUOp *value;
} MultiPiece;

/* Current schedule/multi.py:copy_multi. Scalar targets reconstruct every
 * sharded axis from last to first. Tuple targets pad every shard into the full
 * logical shape before ALLREDUCE(ADD). */
static PolyUOp *multi_pm_copy_multi(
    PolyCtx *ctx,
    PolyUOp *multi,
    PolyUOp *target_device,
    bool *failed
) {
  *failed = false;
  int64_t axes[POLY_MAX_DIMS];
  PolyUOp *ranges[POLY_MAX_DIMS];
  int counts[POLY_MAX_DIMS], n_sharding = 0, n_devices = 0;
  if (!multi_pm_sharding(
          ctx, multi, axes, ranges, counts, &n_sharding) ||
      !target_device || target_device->op != POLY_OP_DEVICE ||
      !multi_pm_tuple_device(ctx, multi, &n_devices)) {
    *failed = true;
    return NULL;
  }
  if (target_device->arg.kind == POLY_ARG_STRING) {
    MultiPiece *pieces = calloc((size_t)n_devices, sizeof(*pieces));
    if (!pieces) {
      *failed = true;
      return NULL;
    }
    bool ok = true;
    for (int i = 0; i < n_devices; i++) {
      PolyUOp *selected =
          poly_uop1(ctx, POLY_OP_MSELECT, multi->dtype, multi->src[0], poly_arg_int(i));
      pieces[i].value = selected
                            ? poly_copy_to_device_uop(ctx, selected, target_device)
                            : NULL;
      if (!pieces[i].value) ok = false;
      for (int j = 0; ok && j < n_sharding; j++)
        ok = multi_pm_shard_idx(ctx, ranges[j], i, &pieces[i].index[j]);
    }
    int n_pieces = n_devices;
    for (int j = n_sharding - 1; ok && j >= 0; j--) {
      MultiPiece *next = calloc((size_t)n_pieces, sizeof(*next));
      bool *used = calloc((size_t)n_pieces, sizeof(*used));
      PolyUOp **group = malloc((size_t)n_pieces * sizeof(*group));
      int *group_index = malloc((size_t)n_pieces * sizeof(*group_index));
      if (!next || !used || !group || !group_index) {
        free(next); free(used); free(group); free(group_index);
        ok = false;
        break;
      }
      int n_next = 0;
      for (int p = 0; p < n_pieces && ok; p++) {
        if (used[p]) continue;
        int n_group = 0;
        for (int q = p; q < n_pieces; q++) {
          if (used[q]) continue;
          bool same = true;
          for (int k = 0; k < j; k++)
            if (pieces[p].index[k] != pieces[q].index[k]) same = false;
          if (!same) continue;
          int pos = n_group++;
          while (pos > 0 && group_index[pos - 1] > pieces[q].index[j]) {
            group[pos] = group[pos - 1];
            group_index[pos] = group_index[pos - 1];
            pos--;
          }
          group[pos] = pieces[q].value;
          group_index[pos] = pieces[q].index[j];
          used[q] = true;
        }
        if (n_group != counts[j]) {
          ok = false;
          break;
        }
        for (int k = 0; k < n_group; k++)
          if (group_index[k] != k) ok = false;
        if (!ok) break;
        next[n_next] = pieces[p];
        next[n_next].value = poly_cat(ctx, group, n_group, (int)axes[j]);
        if (!next[n_next].value) ok = false;
        n_next++;
      }
      free(used); free(group); free(group_index); free(pieces);
      pieces = next;
      n_pieces = n_next;
    }
    PolyUOp *ret = ok && n_pieces == 1 ? pieces[0].value : NULL;
    free(pieces);
    if (!ret) *failed = true;
    return ret;
  }
  if (target_device->arg.kind == POLY_ARG_STRING_TUPLE) {
    PolyUOp *value = multi->src[0];
    for (int j = 0; value && j < n_sharding; j++) {
      int ndim = poly_uop_ndim(ctx, value);
      int axis = (int)axes[j];
      if (ndim < 0 || ndim > POLY_MAX_DIMS || axis < 0 || axis >= ndim) {
        value = NULL;
        break;
      }
      PolyUOp *local_size = poly_uop_shape_dim(ctx, value, axis);
      PolyUOp *offset = local_size
                            ? poly_binop(ctx, POLY_OP_MUL, ranges[j], local_size)
                            : NULL;
      PolyUOp *full_size = poly_uop_shape_dim(ctx, multi, axis);
      PolyUOp *offsets[POLY_MAX_DIMS], *sizes[POLY_MAX_DIMS];
      for (int k = 0; k < ndim; k++) {
        offsets[k] = k == axis ? offset : multi_index_const(ctx, 0);
        sizes[k] = k == axis ? full_size : poly_uop_shape_dim(ctx, value, k);
      }
      value = offset && full_size
                  ? poly_pad_uop(ctx, value, offsets, sizes, ndim)
                  : NULL;
    }
    PolyUOp *ret = value
                       ? poly_allreduce(ctx, value, POLY_OP_ADD, target_device)
                       : NULL;
    if (!ret) *failed = true;
    return ret;
  }
  *failed = true;
  return NULL;
}

static bool multi_pm_same_sharding(PolyUOp *a, PolyUOp *b) {
  if (!a || !b || a->op != POLY_OP_UNSHARD || b->op != POLY_OP_UNSHARD ||
      a->arg.kind != POLY_ARG_INT_TUPLE || b->arg.kind != POLY_ARG_INT_TUPLE ||
      a->arg.int_tuple.n != b->arg.int_tuple.n || a->n_src != b->n_src)
    return false;
  for (int i = 0; i < a->arg.int_tuple.n; i++)
    if (a->arg.int_tuple.vals[i] != b->arg.int_tuple.vals[i] ||
        a->src[i + 1] != b->src[i + 1])
      return false;
  return true;
}

static bool multi_pm_is_broadcast_axis(
    PolyCtx *ctx,
    PolyUOp *source,
    PolyUOp **out_shape,
    int out_ndim,
    int axis
) {
  int source_ndim = poly_uop_ndim(ctx, source);
  int source_axis = axis - (out_ndim - source_ndim);
  if (source_axis < 0) return true;
  PolyUOp *source_dim = poly_uop_shape_dim(ctx, source, source_axis);
  int64_t source_value = 0, out_value = 0;
  return poly_uop_const_i64(source_dim, &source_value) == 0 && source_value == 1 &&
         !(poly_uop_const_i64(out_shape[axis], &out_value) == 0 && out_value == 1);
}

/* Current schedule/multi.py:shard_srcs. This is the single-axis fallback for
 * ALU/STACK operands whose sharding tuples differ. */
static PolyUOp **multi_pm_shard_srcs(
    PolyCtx *ctx,
    PolyUOp **src,
    int n_src,
    int axis,
    PolyUOp **out_range,
    bool *failed
) {
  *failed = false;
  PolyUOp *common_device = NULL, *sharding_range = NULL;
  for (int i = 0; i < n_src; i++) {
    PolyUOp *device = poly_uop_device_uop_cached(ctx, src[i], NULL);
    if (device && common_device && device != common_device) {
      *failed = true;
      return NULL;
    }
    if (device) common_device = device;
    if (!sharding_range && src[i] && src[i]->op == POLY_OP_UNSHARD &&
        src[i]->n_src > 1)
      sharding_range = src[i]->src[1];
  }
  if (common_device) {
    if (common_device->arg.kind != POLY_ARG_STRING_TUPLE ||
        common_device->arg.string_tuple.n <= 0) {
      *failed = true;
      return NULL;
    }
    sharding_range = poly_range(
        ctx, common_device->arg.string_tuple.n, -1, POLY_AXIS_DEVICE);
  }
  if (!sharding_range) {
    *failed = true;
    return NULL;
  }
  PolyUOp *out_shape[POLY_MAX_DIMS];
  int out_ndim = poly_broadcast_shape(ctx, src, n_src, out_shape, POLY_MAX_DIMS);
  if (out_ndim < 0 || axis < 0 || axis >= out_ndim) {
    *failed = true;
    return NULL;
  }
  PolyUOp **local = malloc((size_t)n_src * sizeof(*local));
  if (!local) {
    *failed = true;
    return NULL;
  }
  for (int i = 0; i < n_src; i++) {
    int source_ndim = poly_uop_ndim(ctx, src[i]);
    int source_axis = axis - (out_ndim - source_ndim);
    int existing_axis = -1;
    if (poly_uop_axis(ctx, src[i], &existing_axis) && existing_axis == source_axis) {
      local[i] = src[i]->n_src > 0 ? src[i]->src[0] : NULL;
    } else {
      PolyUOp *full = src[i];
      if (poly_uop_axis(ctx, src[i], &existing_axis)) {
        PolyUOp *device = poly_uop_device_uop_cached(ctx, src[i], NULL);
        full = device ? multi_pm_copy_multi(ctx, src[i], device, failed) : NULL;
      }
      local[i] = full && multi_pm_is_broadcast_axis(
                             ctx, src[i], out_shape, out_ndim, axis)
                     ? full
                     : full && source_axis >= 0
                           ? multi_pm_shard_value(
                                 ctx, full, source_axis, sharding_range, failed)
                           : NULL;
    }
    if (!local[i] || *failed) {
      free(local);
      *failed = true;
      return NULL;
    }
  }
  if (out_range) *out_range = sharding_range;
  return local;
}

/* Current schedule/multi.py:alu_multi. Same-sharding inputs execute locally;
 * full-shape inputs take shard_subview; all other cases use shard_srcs. */
static PolyUOp *multi_pm_alu(
    PolyCtx *ctx,
    PolyUOp *root,
    PolyUOp **src,
    bool *failed
) {
  *failed = false;
  if (!root || root->n_src == 0) return NULL;
  PolyUOp *target = NULL;
  for (int i = 0; i < root->n_src && !target; i++)
    if (src[i] && src[i]->op == POLY_OP_UNSHARD) target = src[i];
  if (!target) return NULL;
  int64_t axes[POLY_MAX_DIMS];
  PolyUOp *ranges[POLY_MAX_DIMS];
  int counts[POLY_MAX_DIMS], n_sharding = 0;
  if (!multi_pm_sharding(
          ctx, target, axes, ranges, counts, &n_sharding)) {
    *failed = true;
    return NULL;
  }
  bool can_handle = true;
  for (int i = 0; i < root->n_src; i++) {
    if (src[i]->op == POLY_OP_UNSHARD)
      can_handle = can_handle && multi_pm_same_sharding(src[i], target);
    else
      can_handle = can_handle &&
                   (poly_uop_ndim(ctx, src[i]) == 0 ||
                    multi_pm_shape_equal(ctx, src[i], target));
  }
  PolyUOp **local = NULL;
  PolyUOp *sharding_range = NULL;
  int axis = -1;
  if (can_handle) {
    local = malloc((size_t)root->n_src * sizeof(*local));
    if (!local) {
      *failed = true;
      return NULL;
    }
    for (int i = 0; i < root->n_src; i++)
      local[i] = src[i]->op == POLY_OP_UNSHARD
                     ? src[i]->src[0]
                     : poly_uop_ndim(ctx, src[i]) == 0
                           ? src[i]
                           : multi_pm_shard_subview(ctx, src[i], target, failed);
  } else {
    if (!poly_uop_axis(ctx, root, &axis)) {
      *failed = true;
      return NULL;
    }
    local = multi_pm_shard_srcs(
        ctx, src, root->n_src, axis, &sharding_range, failed);
  }
  if (!local || *failed) {
    free(local);
    return NULL;
  }
  PolyUOp *inner = poly_uop(
      ctx, root->op, root->dtype, local, root->n_src, root->arg);
  PolyUOp *ret = can_handle
                     ? poly_unshard(ctx, inner, axes, ranges, n_sharding)
                     : multi_pm_wrap_axis(ctx, inner, axis, sharding_range);
  free(local);
  if (!ret) *failed = true;
  return ret;
}

/* Exact current schedule/multi.py:103-119 reduce_multi prefix semantics. */
static PolyUOp *multi_pm_reduce(
    PolyCtx *ctx,
    PolyUOp *root,
    PolyUOp *multi,
    bool *failed
) {
  *failed = false;
  if (!root || root->op != POLY_OP_REDUCE || root->n_src != 1 || !multi ||
      multi->op != POLY_OP_UNSHARD || multi->arg.kind != POLY_ARG_INT_TUPLE ||
      multi->arg.int_tuple.n <= 0 || multi->n_src != multi->arg.int_tuple.n + 1 ||
      root->arg.kind != POLY_ARG_REDUCE || root->arg.reduce.num_axes <= 0) {
    *failed = true;
    return NULL;
  }
  int num_axes = root->arg.reduce.num_axes;
  int n_reduced = 0, n_remaining = 0;
  int64_t remaining_axes[POLY_MAX_DIMS];
  PolyUOp *remaining_ranges[POLY_MAX_DIMS];
  for (int i = 0; i < multi->arg.int_tuple.n; i++) {
    int64_t axis = multi->arg.int_tuple.vals[i];
    if (axis < num_axes) {
      n_reduced++;
    } else {
      remaining_axes[n_remaining] = axis - num_axes;
      remaining_ranges[n_remaining++] = multi->src[i + 1];
    }
  }
  int64_t reduce_axes[POLY_MAX_DIMS];
  for (int i = 0; i < num_axes; i++) reduce_axes[i] = i;
  PolyUOp *local =
      poly_reduce_axis(ctx, root->arg.reduce.op, multi->src[0], reduce_axes, num_axes);
  if (!local) {
    *failed = true;
    return NULL;
  }
  if (n_reduced == 0) {
    PolyUOp *ret = poly_unshard(
        ctx, local, remaining_axes, remaining_ranges, n_remaining);
    if (!ret) *failed = true;
    return ret;
  }
  if (n_remaining != 0) {
    *failed = true;
    return NULL;
  }

  PolyUOp *device = multi_pm_tuple_device(ctx, multi, NULL);
  if (!device) {
    *failed = true;
    return NULL;
  }
  PolyUOp *allreduce_input = local;
  PolyDType local_dtype = local->dtype;
  bool cast_collective =
      poly_getenv_int("ALLREDUCE_CAST", 1) != 0 && multi->src[0]->op == POLY_OP_CAST &&
      multi->src[0]->n_src == 1 &&
      (poly_dtype_eq(multi->src[0]->src[0]->dtype, POLY_BFLOAT16) ||
       poly_dtype_eq(multi->src[0]->src[0]->dtype, POLY_FLOAT16));
  if (cast_collective)
    allreduce_input = poly_uop1(
        ctx, POLY_OP_CAST, multi->src[0]->src[0]->dtype, local, poly_arg_none());
  PolyUOp *ret = allreduce_input
                     ? poly_allreduce(
                           ctx, allreduce_input, root->arg.reduce.op, device)
                     : NULL;
  if (ret && cast_collective)
    ret = poly_uop1(ctx, POLY_OP_CAST, local_dtype, ret, poly_arg_none());
  if (!ret) *failed = true;
  return ret;
}

static PolyUOp *multi_pm_shape_arg_item(PolyUOp *shape, int axis) {
  if (!shape || axis < 0) return NULL;
  if (shape->op == POLY_OP_STACK)
    return axis < shape->n_src ? shape->src[axis] : NULL;
  return axis == 0 && shape->op == POLY_OP_CONST ? shape : NULL;
}

static bool multi_pm_expr_equal(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  if (!a || !b) return false;
  a = poly_graph_rewrite(ctx, a, poly_symbolic());
  b = poly_graph_rewrite(ctx, b, poly_symbolic());
  if (!a || !b) return false;
  if (a == b) return true;
  int64_t av = 0, bv = 0;
  return poly_uop_const_i64(a, &av) == 0 && poly_uop_const_i64(b, &bv) == 0 && av == bv;
}

static PolyUOp *multi_pm_shape_product(PolyCtx *ctx, PolyUOp *u) {
  int ndim = poly_uop_ndim(ctx, u);
  if (ndim < 0 || ndim > POLY_MAX_DIMS) return NULL;
  PolyUOp *product = multi_index_const(ctx, 1);
  for (int i = 0; product && i < ndim; i++) {
    PolyUOp *dim = poly_uop_shape_dim(ctx, u, i);
    product = dim ? poly_binop(ctx, POLY_OP_MUL, product, dim) : NULL;
    product = product ? poly_graph_rewrite(ctx, product, poly_symbolic()) : NULL;
  }
  return product;
}

static bool multi_pm_sharding(
    PolyCtx *ctx,
    PolyUOp *multi,
    int64_t *axes,
    PolyUOp **ranges,
    int *counts,
    int *n_sharding
);

static bool multi_pm_shape_equal(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  int a_ndim = poly_uop_ndim(ctx, a), b_ndim = poly_uop_ndim(ctx, b);
  if (a_ndim < 0 || a_ndim != b_ndim) return false;
  for (int i = 0; i < a_ndim; i++)
    if (!multi_pm_expr_equal(
            ctx, poly_uop_shape_dim(ctx, a, i), poly_uop_shape_dim(ctx, b, i)))
      return false;
  return true;
}

/* Current schedule/multi.py:shard_subview. */
static PolyUOp *multi_pm_shard_subview(
    PolyCtx *ctx,
    PolyUOp *full,
    PolyUOp *multi,
    bool *failed
) {
  *failed = false;
  if (!full || !multi || !multi_pm_shape_equal(ctx, full, multi)) {
    *failed = true;
    return NULL;
  }
  int64_t axes[POLY_MAX_DIMS];
  PolyUOp *ranges[POLY_MAX_DIMS];
  int counts[POLY_MAX_DIMS], n_sharding = 0;
  if (!multi_pm_sharding(ctx, multi, axes, ranges, counts, &n_sharding)) {
    *failed = true;
    return NULL;
  }
  if (full->op == POLY_OP_EXPAND && full->n_src == 2 &&
      poly_uop_ndim(ctx, full->src[0]) == 0) {
    int ndim = poly_uop_ndim(ctx, multi->src[0]);
    PolyUOp *dims[POLY_MAX_DIMS];
    for (int i = 0; i < ndim; i++) dims[i] = poly_uop_shape_dim(ctx, multi->src[0], i);
    PolyUOp *ret = poly_expand_uop(ctx, full->src[0], dims, ndim);
    if (!ret) *failed = true;
    return ret;
  }
  PolyUOp *ret = full;
  for (int i = 0; i < n_sharding; i++) {
    ret = multi_pm_shard_value(ctx, ret, (int)axes[i], ranges[i], failed);
    if (!ret || *failed) return NULL;
  }
  return ret;
}

static PolyUOp *multi_pm_exact_divide(
    PolyCtx *ctx,
    PolyUOp *value,
    int64_t divisor
) {
  if (!value || divisor <= 0) return NULL;
  PolyUOp *count = multi_index_const(ctx, divisor);
  PolyUOp *remainder = count
                           ? poly_uop2(
                                 ctx, POLY_OP_FLOORMOD, POLY_WEAKINT, value, count,
                                 poly_arg_none())
                           : NULL;
  remainder = remainder ? poly_graph_rewrite(ctx, remainder, poly_symbolic()) : NULL;
  int64_t rem = -1;
  if (!remainder || poly_uop_const_i64(remainder, &rem) != 0 || rem != 0) return NULL;
  PolyUOp *quotient = poly_uop2(
      ctx, POLY_OP_FLOORDIV, POLY_WEAKINT, value, count, poly_arg_none());
  return quotient ? poly_graph_rewrite(ctx, quotient, poly_symbolic()) : NULL;
}

/* Current UOp._mop for PAD/SHRINK (uop/ops.py:785-806). Multi rewrites
 * construct raw movement nodes; public MovementMixin methods own later
 * shape-identity elision. */
static PolyUOp *multi_pm_mop_bounds(
    PolyCtx *ctx,
    PolyOps op,
    PolyUOp *value,
    PolyUOp **first,
    PolyUOp **second,
    int ndim
) {
  if (!ctx || !value || (op != POLY_OP_PAD && op != POLY_OP_SHRINK) ||
      ndim < 0 || ndim > POLY_MAX_DIMS)
    return NULL;
  if (ndim == 0) return value;
  PolyUOp *first_shape = poly_shape_to_shape_arg(ctx, first, ndim);
  PolyUOp *second_shape = poly_shape_to_shape_arg(ctx, second, ndim);
  if (!first_shape || !second_shape) return NULL;
  PolyUOp *src[] = {value, first_shape, second_shape};
  return poly_uop(ctx, op, value->dtype, src, 3, poly_arg_none());
}

static PolyUOp *multi_pm_wrap_axis(
    PolyCtx *ctx,
    PolyUOp *u,
    int axis,
    PolyUOp *range
) {
  int64_t axis_value = axis;
  PolyUOp *ranges[] = {range};
  return u ? poly_unshard(ctx, u, &axis_value, ranges, 1) : NULL;
}

/* Exact schedule/multi.py:param_to_multi (lines 138-144). An axis-bearing
 * public PARAM carries one caller slot for the aggregate tuple. multi_pm makes
 * its local shard shape explicit, clears only the axis metadata on the inner
 * PARAM, and restores the public shape with MULTI(axis). */
static PolyUOp *multi_pm_param(PolyCtx *ctx, PolyUOp *param, bool *failed) {
  *failed = false;
  if (!param || param->op != POLY_OP_PARAM || param->arg.kind != POLY_ARG_PARAM ||
      !param->arg.param || !param->arg.param->has_axis ||
      !param->arg.param->device_is_tuple || param->arg.param->n_devices <= 0) {
    *failed = true;
    return NULL;
  }
  int axis = param->arg.param->axis;
  int ndim = poly_uop_ndim(ctx, param);
  if (ndim <= 0 || ndim > POLY_MAX_DIMS || axis < 0 || axis >= ndim) {
    *failed = true;
    return NULL;
  }

  PolyUOp *local_dims[POLY_MAX_DIMS];
  for (int i = 0; i < ndim; i++) {
    local_dims[i] = poly_uop_shape_dim(ctx, param, i);
    if (i == axis)
      local_dims[i] = multi_pm_exact_divide(
          ctx, local_dims[i], param->arg.param->n_devices);
    if (!local_dims[i]) {
      *failed = true;
      return NULL;
    }
  }
  PolyUOp *shape = poly_uop_stack(ctx, local_dims, ndim);
  PolyParamArg local_arg = *param->arg.param;
  local_arg.axis = 0;
  local_arg.has_axis = false;
  PolyUOp *local = shape
                       ? poly_uop1(
                             ctx, POLY_OP_PARAM, param->dtype, shape,
                             poly_arg_param(&local_arg))
                       : NULL;
  PolyUOp *range = poly_range(
      ctx, param->arg.param->n_devices, -1, POLY_AXIS_DEVICE);
  PolyUOp *ret = range ? multi_pm_wrap_axis(ctx, local, axis, range) : NULL;
  if (!ret) *failed = true;
  return ret;
}

static bool multi_pm_sharding(
    PolyCtx *ctx,
    PolyUOp *multi,
    int64_t *axes,
    PolyUOp **ranges,
    int *counts,
    int *n_sharding
) {
  if (!multi || multi->op != POLY_OP_UNSHARD ||
      multi->arg.kind != POLY_ARG_INT_TUPLE || multi->arg.int_tuple.n <= 0 ||
      multi->arg.int_tuple.n > POLY_MAX_DIMS ||
      multi->n_src != multi->arg.int_tuple.n + 1)
    return false;
  for (int i = 0; i < multi->arg.int_tuple.n; i++) {
    PolyUOp *range = multi->src[i + 1];
    int64_t vmin = 0, vmax = -1;
    if (!range || !poly_dtype_is_int(range->dtype)) return false;
    poly_uop_minmax(ctx, range, &vmin, &vmax);
    if (vmin < 0 || vmax < 0 || vmax >= INT_MAX) return false;
    axes[i] = multi->arg.int_tuple.vals[i];
    ranges[i] = range;
    counts[i] = (int)vmax + 1;
  }
  *n_sharding = multi->arg.int_tuple.n;
  return true;
}

/* Current schedule/multi.py:reshape_multi through flip_multi. Each rule uses
 * the rewritten UNSHARD's complete (axis, RANGE) set; no parent-axis cache or
 * device-only assumption participates in movement lowering. */
static PolyUOp *multi_pm_movement(
    PolyCtx *ctx,
    PolyUOp *root,
    PolyUOp *multi,
    PolyUOp **src,
    bool *failed
) {
  *failed = false;
  int64_t axes[POLY_MAX_DIMS];
  PolyUOp *ranges[POLY_MAX_DIMS];
  int counts[POLY_MAX_DIMS], n_sharding = 0;
  if (!root || !multi_pm_sharding(ctx, multi, axes, ranges, counts, &n_sharding)) {
    *failed = true;
    return NULL;
  }
  int local_ndim = poly_uop_ndim(ctx, multi->src[0]);
  int public_ndim = poly_uop_ndim(ctx, multi);
  if (local_ndim < 0 || local_ndim > POLY_MAX_DIMS || public_ndim != local_ndim) {
    *failed = true;
    return NULL;
  }
  for (int i = 0; i < n_sharding; i++)
    if (axes[i] < 0 || axes[i] >= local_ndim) {
      *failed = true;
      return NULL;
    }

  if (root->op == POLY_OP_RESHAPE && root->n_src == 2) {
    PolyUOp *old_product = multi_pm_shape_product(ctx, multi);
    PolyUOp *new_product = multi_pm_shape_product(ctx, root);
    if (!old_product || !new_product || !multi_pm_expr_equal(ctx, old_product, new_product)) {
      *failed = true;
      return NULL;
    }
    int ndim = poly_uop_ndim(ctx, root);
    if (ndim < 0 || ndim > POLY_MAX_DIMS) {
      *failed = true;
      return NULL;
    }
    PolyUOp *dims[POLY_MAX_DIMS], *prefix[POLY_MAX_DIMS + 1];
    for (int i = 0; i < ndim; i++) dims[i] = poly_uop_shape_dim(ctx, root, i);
    prefix[0] = multi_index_const(ctx, 1);
    for (int i = 0; i < ndim; i++) {
      prefix[i + 1] = prefix[i] && dims[i]
                          ? poly_uop2(
                                ctx, POLY_OP_MUL, POLY_WEAKINT, prefix[i], dims[i],
                                poly_arg_none())
                          : NULL;
      prefix[i + 1] = prefix[i + 1]
                          ? poly_graph_rewrite(ctx, prefix[i + 1], poly_symbolic())
                          : NULL;
    }
    int64_t new_axes[POLY_MAX_DIMS];
    for (int i = 0; i < n_sharding; i++) {
      PolyUOp *target = multi_index_const(ctx, 1);
      for (int j = 0; target && j < axes[i]; j++) {
        PolyUOp *dim = poly_uop_shape_dim(ctx, multi, j);
        target = dim ? poly_uop2(
                           ctx, POLY_OP_MUL, POLY_WEAKINT, target, dim,
                           poly_arg_none())
                     : NULL;
        target = target ? poly_graph_rewrite(ctx, target, poly_symbolic()) : NULL;
      }
      int new_axis = -1;
      for (int j = 0; j <= ndim; j++)
        if (prefix[j] && multi_pm_expr_equal(ctx, target, prefix[j])) new_axis = j;
      if (new_axis < 0 || new_axis >= ndim) {
        *failed = true;
        return NULL;
      }
      for (int j = 0; j < i; j++)
        if (new_axes[j] == new_axis) {
          *failed = true;
          return NULL;
        }
      new_axes[i] = new_axis;
      dims[new_axis] = multi_pm_exact_divide(ctx, dims[new_axis], counts[i]);
      if (!dims[new_axis]) {
        *failed = true;
        return NULL;
      }
    }
    PolyUOp *local = poly_reshape_uop(ctx, multi->src[0], dims, ndim);
    PolyUOp *ret = local
                       ? poly_unshard(ctx, local, new_axes, ranges, n_sharding)
                       : NULL;
    if (!ret) *failed = true;
    return ret;
  }

  if (root->op == POLY_OP_EXPAND && root->n_src == 2) {
    PolyUOp *marg[POLY_MAX_DIMS];
    int n_marg = poly_uop_as_shape(ctx, root->src[1], marg, POLY_MAX_DIMS);
    if (n_marg < 0) {
      *failed = true;
      return NULL;
    }
    PolyUOp *expand_src[2] = {multi->src[0], root->src[1]};
    PolyUOp *local = poly_uop(
        ctx, POLY_OP_EXPAND, root->dtype, expand_src, 2, poly_arg_none()
    );
    int64_t new_axes[POLY_MAX_DIMS];
    for (int i = 0; i < n_sharding; i++) new_axes[i] = axes[i] + n_marg;
    PolyUOp *ret = local
                       ? poly_unshard(ctx, local, new_axes, ranges, n_sharding)
                       : NULL;
    if (!ret) *failed = true;
    return ret;
  }

  if (root->op == POLY_OP_PAD && root->n_src == 3) {
    PolyUOp *zero = multi_index_const(ctx, 0);
    PolyUOp *offsets[POLY_MAX_DIMS], *sizes[POLY_MAX_DIMS];
    for (int i = 0; i < local_ndim; i++) {
      bool sharded = false;
      for (int j = 0; j < n_sharding; j++)
        if (axes[j] == i) sharded = true;
      offsets[i] = sharded ? zero : multi_pm_shape_arg_item(src[1], i);
      sizes[i] = sharded ? poly_uop_shape_dim(ctx, multi->src[0], i)
                         : multi_pm_shape_arg_item(src[2], i);
      if (!offsets[i] || !sizes[i]) {
        *failed = true;
        return NULL;
      }
      if (sharded &&
          (!multi_pm_expr_equal(ctx, multi_pm_shape_arg_item(src[1], i), zero) ||
           !multi_pm_expr_equal(
               ctx, multi_pm_shape_arg_item(src[2], i),
               poly_uop_shape_dim(ctx, multi, i)))) {
        *failed = true;
        return NULL;
      }
    }
    PolyUOp *local = multi_pm_mop_bounds(
        ctx, POLY_OP_PAD, multi->src[0], offsets, sizes, local_ndim);
    PolyUOp *ret = local
                       ? poly_unshard(ctx, local, axes, ranges, n_sharding)
                       : NULL;
    if (!ret) *failed = true;
    return ret;
  }

  if (root->op == POLY_OP_PERMUTE && root->n_src == 1 &&
      root->arg.kind == POLY_ARG_INT_TUPLE) {
    int64_t new_axes[POLY_MAX_DIMS];
    for (int i = 0; i < n_sharding; i++) {
      new_axes[i] = -1;
      for (int j = 0; j < root->arg.int_tuple.n; j++)
        if (root->arg.int_tuple.vals[j] == axes[i]) new_axes[i] = j;
      if (new_axes[i] < 0) {
        *failed = true;
        return NULL;
      }
    }
    PolyUOp *local = poly_permute(
        ctx, multi->src[0], root->arg.int_tuple.vals, root->arg.int_tuple.n);
    PolyUOp *ret = local
                       ? poly_unshard(ctx, local, new_axes, ranges, n_sharding)
                       : NULL;
    if (!ret) *failed = true;
    return ret;
  }

  if (root->op == POLY_OP_SHRINK && root->n_src == 3) {
    PolyUOp *zero = multi_index_const(ctx, 0);
    int selected_partition = -1;
    PolyUOp *starts[POLY_MAX_DIMS], *sizes[POLY_MAX_DIMS];
    for (int i = 0; i < local_ndim; i++) {
      starts[i] = multi_pm_shape_arg_item(src[1], i);
      sizes[i] = multi_pm_shape_arg_item(src[2], i);
      if (!starts[i] || !sizes[i]) {
        *failed = true;
        return NULL;
      }
    }
    int64_t remaining_axes[POLY_MAX_DIMS];
    PolyUOp *remaining_ranges[POLY_MAX_DIMS];
    int n_remaining = 0;
    for (int i = 0; i < n_sharding; i++) {
      int axis = (int)axes[i];
      PolyUOp *start = multi_pm_shape_arg_item(src[1], axis);
      PolyUOp *size = multi_pm_shape_arg_item(src[2], axis);
      PolyUOp *local_axis = poly_uop_shape_dim(ctx, multi->src[0], axis);
      PolyUOp *public_axis = poly_uop_shape_dim(ctx, multi, axis);
      PolyUOp *owned_start = local_axis
                                 ? poly_uop2(
                                       ctx, POLY_OP_MUL, POLY_WEAKINT, ranges[i],
                                       local_axis, poly_arg_none())
                                 : NULL;
      owned_start = owned_start
                        ? poly_graph_rewrite(ctx, owned_start, poly_symbolic())
                        : NULL;
      if (multi_pm_expr_equal(ctx, size, local_axis) &&
          multi_pm_expr_equal(ctx, start, owned_start)) {
        starts[axis] = zero;
        sizes[axis] = local_axis;
        continue;
      }
      if (multi_pm_expr_equal(ctx, start, zero) &&
          multi_pm_expr_equal(ctx, size, public_axis)) {
        starts[axis] = zero;
        sizes[axis] = local_axis;
        remaining_axes[n_remaining] = axis;
        remaining_ranges[n_remaining++] = ranges[i];
        continue;
      }
      PolyUOp *device = multi_pm_tuple_device(ctx, multi, NULL);
      if (n_sharding != 1 || !device ||
          device->arg.kind != POLY_ARG_STRING_TUPLE) {
        *failed = true;
        return NULL;
      }
      for (int lane = 0; lane < counts[i]; lane++) {
        PolyUOp *partition_start = lane == 0
                                       ? zero
                                       : poly_uop2(
                                             ctx, POLY_OP_MUL, POLY_WEAKINT,
                                             multi_index_const(ctx, lane), local_axis,
                                             poly_arg_none());
        partition_start = partition_start
                              ? poly_graph_rewrite(ctx, partition_start, poly_symbolic())
                              : NULL;
        if (multi_pm_expr_equal(ctx, start, partition_start) &&
            multi_pm_expr_equal(ctx, size, local_axis))
          selected_partition = lane;
      }
      if (selected_partition < 0) {
        *failed = true;
        return NULL;
      }
      starts[axis] = zero;
      sizes[axis] = local_axis;
    }
    if (selected_partition >= 0) {
      PolyUOp *selected = poly_uop1(
          ctx, POLY_OP_MSELECT, multi->dtype, multi->src[0],
          poly_arg_int(selected_partition));
      PolyUOp *device = multi_pm_tuple_device(ctx, multi, NULL);
      PolyUOp *copy = selected && device
                          ? poly_copy_to_device_uop(ctx, selected, device)
                          : NULL;
      PolyUOp *ret = copy
                         ? multi_pm_mop_bounds(
                               ctx, POLY_OP_SHRINK, copy, starts, sizes, local_ndim)
                         : NULL;
      if (!ret) *failed = true;
      return ret;
    }
    PolyUOp *local = multi_pm_mop_bounds(
        ctx, POLY_OP_SHRINK, multi->src[0], starts, sizes, local_ndim);
    PolyUOp *ret = !local ? NULL
                          : n_remaining == 0
                                ? local
                                : poly_unshard(
                                      ctx, local, remaining_axes, remaining_ranges,
                                      n_remaining);
    if (!ret) *failed = true;
    return ret;
  }

  if (root->op == POLY_OP_FLIP && root->n_src == 1 &&
      root->arg.kind == POLY_ARG_INT_TUPLE && root->arg.int_tuple.n == local_ndim) {
    for (int i = 0; i < n_sharding; i++)
      if (root->arg.int_tuple.vals[axes[i]] != 0) {
        *failed = true;
        return NULL;
      }
    PolyUOp *local = poly_uop1(
        ctx, POLY_OP_FLIP, multi->src[0]->dtype, multi->src[0], root->arg);
    PolyUOp *ret = local
                       ? poly_unshard(ctx, local, axes, ranges, n_sharding)
                       : NULL;
    if (!ret) *failed = true;
    return ret;
  }

  *failed = true;
  return NULL;
}

/* Current schedule/multi.py:stack_multi. Equal sharding shifts every axis by
 * one; mismatched single-axis inputs use the shared shard_srcs fallback. */
static PolyUOp *multi_pm_stack(
    PolyCtx *ctx,
    PolyUOp *root,
    PolyUOp **src,
    bool *failed
) {
  *failed = false;
  PolyUOp *target = NULL;
  for (int i = 0; root && i < root->n_src && !target; i++)
    if (src[i] && src[i]->op == POLY_OP_UNSHARD) target = src[i];
  if (!root || root->op != POLY_OP_STACK || !target) return NULL;
  int64_t axes[POLY_MAX_DIMS];
  PolyUOp *ranges[POLY_MAX_DIMS];
  int counts[POLY_MAX_DIMS], n_sharding = 0;
  if (!multi_pm_sharding(ctx, target, axes, ranges, counts, &n_sharding)) {
    *failed = true;
    return NULL;
  }
  PolyUOp **local = malloc((size_t)root->n_src * sizeof(*local));
  if (!local) {
    *failed = true;
    return NULL;
  }
  bool same = true;
  for (int i = 0; i < root->n_src; i++) {
    if (src[i] && src[i]->op == POLY_OP_UNSHARD) {
      if (src[i]->arg.kind != POLY_ARG_INT_TUPLE ||
          src[i]->arg.int_tuple.n != n_sharding || src[i]->n_src != n_sharding + 1) {
        same = false;
        break;
      }
      for (int j = 0; j < n_sharding; j++)
        if (src[i]->arg.int_tuple.vals[j] != axes[j] ||
            src[i]->src[j + 1] != ranges[j])
          same = false;
      local[i] = src[i]->src[0];
    } else {
      local[i] = src[i];
    }
  }
  if (!same) {
    free(local);
    int axis = -1;
    PolyUOp *sharding_range = NULL;
    if (!poly_uop_axis(ctx, root, &axis) || axis <= 0) {
      *failed = true;
      return NULL;
    }
    local = multi_pm_shard_srcs(
        ctx, src, root->n_src, axis - 1, &sharding_range, failed);
    if (!local || *failed) {
      free(local);
      return NULL;
    }
    PolyUOp *stack = poly_uop(
        ctx, POLY_OP_STACK, root->dtype, local, root->n_src, root->arg);
    PolyUOp *ret = stack
                       ? multi_pm_wrap_axis(ctx, stack, axis, sharding_range)
                       : NULL;
    free(local);
    if (!ret) *failed = true;
    return ret;
  }
  PolyUOp *stack = poly_uop(
      ctx, POLY_OP_STACK, root->dtype, local, root->n_src, root->arg);
  for (int i = 0; i < n_sharding; i++) axes[i]++;
  PolyUOp *ret = stack ? poly_unshard(ctx, stack, axes, ranges, n_sharding) : NULL;
  free(local);
  if (!ret) *failed = true;
  return ret;
}

/* Current schedule/multi.py:index_multi. */
static PolyUOp *multi_pm_index(
    PolyCtx *ctx,
    PolyUOp *root,
    PolyUOp *multi,
    PolyUOp **src,
    bool *failed
) {
  *failed = false;
  if (!root || root->op != POLY_OP_INDEX || root->n_src < 2 || !multi ||
      multi->op != POLY_OP_UNSHARD) {
    *failed = true;
    return NULL;
  }
  int64_t axes[POLY_MAX_DIMS];
  PolyUOp *ranges[POLY_MAX_DIMS];
  int counts[POLY_MAX_DIMS], n_sharding = 0;
  if (!multi_pm_sharding(ctx, multi, axes, ranges, counts, &n_sharding)) {
    *failed = true;
    return NULL;
  }
  PolyUOp **local_src = malloc((size_t)root->n_src * sizeof(*local_src));
  if (!local_src) {
    *failed = true;
    return NULL;
  }
  local_src[0] = multi->src[0];
  for (int i = 1; i < root->n_src; i++) local_src[i] = src[i];
  for (int i = 0; i < n_sharding; i++) {
    int axis = (int)axes[i];
    if (axis < 0 || axis + 1 >= root->n_src) {
      *failed = true;
      break;
    }
    PolyUOp *shard_size = poly_uop_shape_dim(ctx, multi->src[0], axis);
    PolyUOp *offset = shard_size
                          ? poly_uop2(
                                ctx, POLY_OP_MUL, POLY_WEAKINT, ranges[i],
                                shard_size, poly_arg_none())
                          : NULL;
    PolyUOp *local = offset ? poly_sub(ctx, src[axis + 1], offset) : NULL;
    local = local ? poly_graph_rewrite(ctx, local, poly_symbolic()) : NULL;
    int64_t vmin = -1, vmax = -1, shard_min = -1, shard_max = -1;
    if (local) poly_uop_minmax(ctx, local, &vmin, &vmax);
    if (shard_size) poly_uop_minmax(ctx, shard_size, &shard_min, &shard_max);
    if (!local || vmin < 0 || shard_max < 0 || vmax >= shard_max) {
      PolyUOp *diff = poly_sub(ctx, src[axis + 1], ranges[i]);
      PolyUOp *mod = diff && shard_size
                         ? poly_uop2(
                               ctx, POLY_OP_FLOORMOD, POLY_WEAKINT, diff,
                               shard_size, poly_arg_none())
                         : NULL;
      mod = mod ? poly_graph_rewrite(ctx, mod, poly_symbolic()) : NULL;
      int64_t mod_value = -1;
      if (!mod || poly_uop_const_i64(mod, &mod_value) != 0 || mod_value != 0) {
        *failed = true;
        break;
      }
      local = poly_uop2(
          ctx, POLY_OP_FLOORDIV, POLY_WEAKINT, diff, shard_size,
          poly_arg_none());
      local = local ? poly_graph_rewrite(ctx, local, poly_symbolic()) : NULL;
      vmin = vmax = -1;
      if (local) poly_uop_minmax(ctx, local, &vmin, &vmax);
      if (!local || vmin < 0 || vmax >= shard_max) {
        *failed = true;
        break;
      }
    }
    local_src[axis + 1] = local;
  }
  PolyUOp *ret = !*failed
                     ? poly_uop(
                           ctx, POLY_OP_INDEX, root->dtype, local_src,
                           root->n_src, root->arg)
                     : NULL;
  free(local_src);
  if (!ret) *failed = true;
  return ret;
}

/* Current schedule/multi.py:store_value_multi. */
static PolyUOp *multi_pm_store_value(
    PolyCtx *ctx,
    PolyUOp *root,
    PolyUOp *dest,
    PolyUOp *multi,
    bool *failed
) {
  PolyUOp *local_dest = multi_pm_shard_subview(ctx, dest, multi, failed);
  PolyUOp *ret = local_dest && !*failed
                     ? poly_store_val(ctx, local_dest, multi->src[0])
                     : NULL;
  if (!ret) *failed = true;
  return ret;
}

/* Current schedule/multi.py:store_dest_multi. */
static PolyUOp *multi_pm_store_dest(
    PolyCtx *ctx,
    PolyUOp *root,
    PolyUOp *multi,
    PolyUOp **src,
    bool *failed
) {
  *failed = false;
  PolyUOp **local = malloc((size_t)root->n_src * sizeof(*local));
  if (!local) {
    *failed = true;
    return NULL;
  }
  local[0] = multi->src[0];
  for (int i = 1; i < root->n_src; i++) {
    if (src[i] && src[i]->op == POLY_OP_UNSHARD)
      local[i] = src[i]->src[0];
    else if (multi_pm_shape_equal(ctx, src[i], multi))
      local[i] = multi_pm_shard_subview(ctx, src[i], multi, failed);
    else
      local[i] = src[i];
    if (!local[i] || *failed) break;
  }
  PolyUOp *ret = !*failed
                     ? poly_uop(
                           ctx, root->op, root->dtype, local, root->n_src,
                           root->arg)
                     : NULL;
  free(local);
  if (!ret) *failed = true;
  return ret;
}

/* Pinned schedule/multi.py:124-125. Localize an operation whose value source
 * is MULTI, unwrap every other direct MULTI source, then restore the first
 * source's exact shard axis. The local operation is a fresh UOp in tinygrad,
 * so it intentionally does not inherit root tags. */
static PolyUOp *multi_pm_passthrough(
    PolyCtx *ctx,
    PolyUOp *root,
    PolyUOp **src,
    bool wrap_multi,
    bool *failed
) {
  if (!ctx || !root || !src || !failed || root->n_src < 1 ||
      (wrap_multi &&
       (!src[0] || src[0]->op != POLY_OP_UNSHARD ||
        src[0]->arg.kind != POLY_ARG_INT_TUPLE ||
        src[0]->n_src != src[0]->arg.int_tuple.n + 1))) {
    *failed = true;
    return NULL;
  }
  PolyUOp *local_src_stack[16];
  PolyUOp **local_src = root->n_src > 16
                           ? malloc((size_t)root->n_src * sizeof(*local_src))
                           : local_src_stack;
  if (!local_src) {
    *failed = true;
    return NULL;
  }
  for (int i = 0; i < root->n_src; i++) {
    PolyUOp *candidate = src[i];
    if (candidate && candidate->op == POLY_OP_UNSHARD) {
      if (candidate->arg.kind != POLY_ARG_INT_TUPLE ||
          candidate->n_src != candidate->arg.int_tuple.n + 1) {
        if (local_src != local_src_stack) free(local_src);
        *failed = true;
        return NULL;
      }
      candidate = candidate->src[0];
    }
    local_src[i] = candidate;
  }
  PolyUOp *local = poly_uop(
      ctx, root->op, root->dtype, local_src, root->n_src, root->arg);
  if (local_src != local_src_stack) free(local_src);
  if (!local) {
    *failed = true;
    return NULL;
  }
  if (!wrap_multi) return local;
  PolyUOp *ret = multi_pm_restore_sharding(ctx, local, src[0]);
  if (!ret) *failed = true;
  return ret;
}

/* Pinned schedule/multi.py:127-136. CALL/FUNCTION bodies remain opaque to the
 * outer traversal; a non-precompiled value-producing FUNCTION is the one rule
 * that explicitly runs multi_pm over its TUPLE body. */
static PolyUOp *multi_pm_function(
    PolyCtx *ctx,
    PolyUOp *call,
    PolyUOp **src,
    bool *failed
) {
  if (!ctx || !call || call->op != POLY_OP_FUNCTION || call->n_src < 1 ||
      !call->src[0] || !src || !failed) {
    *failed = true;
    return NULL;
  }
  if (call->arg.kind == POLY_ARG_CALL_INFO && call->arg.call_info &&
      call->arg.call_info->precompile) {
    bool changed = false;
    for (int i = 1; i < call->n_src; i++)
      if (src[i] != call->src[i]) changed = true;
    if (!changed) return call;
    return (call->tag != 0 || call->tag_arg.kind != POLY_ARG_NONE)
               ? poly_uop_tagged_arg(
                     ctx, call->op, call->dtype, src, call->n_src,
                     call->arg, call->tag, call->tag_arg)
               : poly_uop(
                     ctx, call->op, call->dtype, src, call->n_src, call->arg);
  }
  PolyUOp *new_body = poly_apply_multi_pm(ctx, call->src[0]);
  if (!new_body || new_body->op != POLY_OP_TUPLE) {
    *failed = true;
    return NULL;
  }

  PolyUOp *call_src_stack[16];
  PolyUOp **call_src = call->n_src > 16
                          ? malloc((size_t)call->n_src * sizeof(*call_src))
                          : call_src_stack;
  if (!call_src) {
    *failed = true;
    return NULL;
  }
  call_src[0] = new_body;
  for (int i = 1; i < call->n_src; i++) {
    if (src[i] && src[i]->op == POLY_OP_UNSHARD) {
      if (src[i]->arg.kind != POLY_ARG_INT_TUPLE ||
          src[i]->n_src != src[i]->arg.int_tuple.n + 1) {
        if (call_src != call_src_stack) free(call_src);
        *failed = true;
        return NULL;
      }
      call_src[i] = src[i]->src[0];
    } else {
      call_src[i] = src[i];
    }
  }

  bool has_multi_output = false;
  for (int i = 0; i < new_body->n_src; i++) {
    PolyUOp *output = new_body->src[i];
    if (output && output->op == POLY_OP_UNSHARD) {
      if (output->arg.kind != POLY_ARG_INT_TUPLE || output->arg.int_tuple.n <= 0 ||
          output->n_src != output->arg.int_tuple.n + 1) {
        if (call_src != call_src_stack) free(call_src);
        *failed = true;
        return NULL;
      }
      has_multi_output = true;
    }
  }

  PolyUOp *result = NULL;
  if (!has_multi_output) {
    result = multi_clone(
        ctx, call, call_src, call->n_src);
  } else {
    PolyUOp *body_src_stack[16], *result_src_stack[16];
    PolyUOp **body_src = new_body->n_src > 16
                            ? malloc((size_t)new_body->n_src * sizeof(*body_src))
                            : body_src_stack;
    PolyUOp **result_src = new_body->n_src > 16
                              ? malloc((size_t)new_body->n_src * sizeof(*result_src))
                              : result_src_stack;
    if (!body_src || !result_src) {
      if (body_src && body_src != body_src_stack) free(body_src);
      if (result_src && result_src != result_src_stack) free(result_src);
      if (call_src != call_src_stack) free(call_src);
      *failed = true;
      return NULL;
    }
    for (int i = 0; i < new_body->n_src; i++)
      body_src[i] = new_body->src[i]->op == POLY_OP_UNSHARD
                        ? new_body->src[i]->src[0]
                        : new_body->src[i];
    PolyUOp *shard_body = poly_uop(
        ctx, POLY_OP_TUPLE, POLY_VOID, body_src, new_body->n_src,
        poly_arg_none());
    call_src[0] = shard_body;
    PolyUOp *shard_call = shard_body
                              ? multi_clone(
                                    ctx, call, call_src, call->n_src)
                              : NULL;
    bool ok = shard_call != NULL;
    for (int i = 0; ok && i < new_body->n_src; i++) {
      PolyUOp *selected = poly_uop1(
          ctx, POLY_OP_GETTUPLE, new_body->src[i]->dtype, shard_call,
          poly_arg_int(i));
      if (new_body->src[i]->op == POLY_OP_UNSHARD)
        selected = multi_pm_restore_sharding(ctx, selected, new_body->src[i]);
      result_src[i] = selected;
      ok = selected != NULL;
    }
    if (ok)
      result = poly_uop(
          ctx, POLY_OP_TUPLE, POLY_VOID, result_src, new_body->n_src,
          poly_arg_none());
    if (body_src != body_src_stack) free(body_src);
    if (result_src != result_src_stack) free(result_src);
  }
  if (call_src != call_src_stack) free(call_src);
  if (!result) *failed = true;
  return result;
}

/* Current tinygrad schedule/multi.py:23-34 lower_broadcast_copy and
 * COPY_TO_ONE.  COPY is unary and carries its exact scalar/tuple destination
 * in arg.  Tuple broadcast creates one ordered physical occurrence per
 * destination; tuple-to-scalar selects lane zero before copying. */
static PolyUOp *multi_pm_copy(PolyCtx *ctx, PolyUOp *copy, PolyUOp *source, bool *failed) {
  if (!ctx || !copy || copy->op != POLY_OP_COPY || copy->n_src != 1 || !source) return NULL;

  PolyUOp *source_device = poly_uop_device_uop_cached(ctx, source, NULL);
  if (copy->arg.kind == POLY_ARG_STRING_TUPLE) {
    if (source_device && source_device->arg.kind != POLY_ARG_STRING) return NULL;
    int n_devices = copy->arg.string_tuple.n;
    if (n_devices <= 0 || !copy->arg.string_tuple.vals) {
      if (failed) *failed = true;
      return NULL;
    }

    PolyUOp *simplified = poly_graph_rewrite(ctx, source, poly_symbolic());
    if (!simplified) {
      if (failed) *failed = true;
      return NULL;
    }
    PolyUOp *simplified_device = poly_uop_device_uop_cached(ctx, simplified, NULL);
    PolyUOp **copies = malloc((size_t)n_devices * sizeof(*copies));
    if (!copies) {
      if (failed) *failed = true;
      return NULL;
    }
    bool ok = true;
    for (int i = 0; i < n_devices; i++) {
      if (!simplified_device) {
        copies[i] = simplified;
        continue;
      }
      PolyUOp *device =
          poly_device_uop_from_name(ctx, copy->arg.string_tuple.vals[i]);
      copies[i] = device ? poly_copy_to_device_uop(ctx, simplified, device) : NULL;
      if (!copies[i]) ok = false;
    }
    PolyUOp *result = ok ? poly_uop(
                               ctx, POLY_OP_MSTACK, copy->dtype, copies, n_devices,
                               poly_arg_none())
                         : NULL;
    free(copies);
    if (!result && failed) *failed = true;
    return result;
  }

  if (copy->arg.kind == POLY_ARG_STRING && source_device &&
      source_device->arg.kind == POLY_ARG_STRING_TUPLE) {
    PolyUOp *selected = poly_uop1(
        ctx, POLY_OP_MSELECT, source->dtype, source, poly_arg_int(0));
    PolyUOp *device = poly_device_uop_from_name(ctx, copy->arg.str);
    PolyUOp *result = selected && device
                          ? poly_copy_to_device_uop(ctx, selected, device)
                          : NULL;
    if (!result && failed) *failed = true;
    return result;
  }
  return NULL;
}

/* Pinned tinygrad schedule/multi.py:replace_allreduce, first dependency-closed
 * rules. get_kernel_graph applies multi_pm before earliest_rewrites, turning a
 * tuple DEVICE request into exact scalar-device occurrences before scheduling.
 * graph_rewrite defaults to enter_calls=False and reaches a fixed point; keep
 * both properties here. */
PolyUOp *poly_apply_multi_pm(PolyCtx *ctx, PolyUOp *sink) {
  if (!ctx || !sink) return NULL;
  for (;;) {
    int n_topo = 0;
    PolyUOp **topo = poly_toposort_ex_alloc(ctx, sink, &n_topo, NULL, false);
    if (!topo) return NULL;
    PolyMap *rmap = poly_map_new(n_topo < 16 ? 16 : (uint32_t)n_topo);
    if (!rmap) {
      poly_toposort_free(topo);
      return NULL;
    }
    bool changed = false;

    for (int t = 0; t < n_topo; t++) {
      PolyUOp *u = topo[t];
      PolyUOp *ns_buf[16];
      PolyUOp **ns = u->n_src > 16 ? malloc((size_t)u->n_src * sizeof(*ns)) : ns_buf;
      if (u->n_src > 16 && !ns) {
        poly_map_destroy(rmap);
        poly_toposort_free(topo);
        return NULL;
      }
      bool src_changed = false;
      for (int i = 0; i < u->n_src; i++) {
        PolyUOp *mapped = multi_map_get(rmap, u->src[i]);
        ns[i] = mapped ? mapped : u->src[i];
        if (ns[i] != u->src[i]) src_changed = true;
      }

      PolyUOp *result = NULL;
      bool rule_failed = false;
      if (u->op == POLY_OP_PARAM && u->arg.kind == POLY_ARG_PARAM &&
          u->arg.param && u->arg.param->has_axis) {
        result = multi_pm_param(ctx, u, &rule_failed);
      } else if (u->op == POLY_OP_COPY && u->n_src == 1 && ns[0] &&
                 ns[0]->op == POLY_OP_UNSHARD) {
        PolyUOp *target = NULL;
        if (u->arg.kind == POLY_ARG_STRING)
          target = poly_device_uop_from_name(ctx, u->arg.str);
        else if (u->arg.kind == POLY_ARG_STRING_TUPLE)
          target = poly_device_uop_from_names(
              ctx, u->arg.string_tuple.vals, u->arg.string_tuple.n);
        result = target
                     ? multi_pm_copy_multi(ctx, ns[0], target, &rule_failed)
                     : NULL;
        if (!target) rule_failed = true;
      } else if (u->op == POLY_OP_COPY && u->n_src == 1) {
        result = multi_pm_copy(ctx, u, ns[0], &rule_failed);
      } else if (u->op == POLY_OP_MSELECT && u->n_src == 1 && ns[0] &&
                 ns[0]->op == POLY_OP_MSTACK && u->arg.kind == POLY_ARG_INT &&
                 u->arg.i >= 0 && u->arg.i < ns[0]->n_src) {
        result = ns[0]->src[u->arg.i];
      } else if (u->op == POLY_OP_SHRINK && u->n_src == 3 && ns[0] &&
                 ns[0]->op == POLY_OP_MSTACK) {
        result = multi_pm_mstack_early_shrink(
            ctx, u, ns[0], ns[1], ns[2], &rule_failed);
      } else if (u->op == POLY_OP_MSELECT && u->n_src == 1 && ns[0] &&
                 poly_opset_has(POLY_GROUP_MOVEMENT, ns[0]->op) &&
                 ns[0]->n_src >= 1) {
        /* Pinned schedule/multi.py:32-34 moves selection before movement. */
        PolyUOp *selected = poly_uop1(
            ctx, POLY_OP_MSELECT, ns[0]->src[0]->dtype, ns[0]->src[0], u->arg);
        PolyUOp *movement_src_stack[16];
        PolyUOp **movement_src = ns[0]->n_src > 16
                                     ? malloc((size_t)ns[0]->n_src * sizeof(*movement_src))
                                     : movement_src_stack;
        if (!selected || !movement_src) {
          if (movement_src && movement_src != movement_src_stack) free(movement_src);
          rule_failed = true;
        } else {
          movement_src[0] = selected;
          for (int i = 1; i < ns[0]->n_src; i++) movement_src[i] = ns[0]->src[i];
          result = multi_clone(
              ctx, ns[0], movement_src, ns[0]->n_src);
          if (movement_src != movement_src_stack) free(movement_src);
          if (!result) rule_failed = true;
        }
      } else if (poly_opset_has(POLY_GROUP_ALU, u->op)) {
        bool has_multi_source = false;
        for (int i = 0; i < u->n_src; i++)
          if (ns[i] && ns[i]->op == POLY_OP_UNSHARD) has_multi_source = true;
        if (has_multi_source) result = multi_pm_alu(ctx, u, ns, &rule_failed);
      } else if (u->op == POLY_OP_STACK) {
        bool has_multi_source = false;
        for (int i = 0; i < u->n_src; i++)
          if (ns[i] && ns[i]->op == POLY_OP_UNSHARD) has_multi_source = true;
        if (has_multi_source) result = multi_pm_stack(ctx, u, ns, &rule_failed);
      } else if (u->op == POLY_OP_REDUCE && u->n_src == 1 && ns[0] &&
                 ns[0]->op == POLY_OP_UNSHARD) {
        result = multi_pm_reduce(ctx, u, ns[0], &rule_failed);
      } else if ((u->op == POLY_OP_RESHAPE || u->op == POLY_OP_EXPAND ||
                  u->op == POLY_OP_PAD || u->op == POLY_OP_SHRINK ||
                  u->op == POLY_OP_PERMUTE || u->op == POLY_OP_FLIP) &&
                 u->n_src >= 1 && ns[0] && ns[0]->op == POLY_OP_UNSHARD) {
        result = multi_pm_movement(ctx, u, ns[0], ns, &rule_failed);
      } else if (u->op == POLY_OP_INDEX && u->n_src >= 2 && ns[0] &&
                 ns[0]->op == POLY_OP_UNSHARD) {
        result = multi_pm_index(ctx, u, ns[0], ns, &rule_failed);
      } else if (u->op == POLY_OP_ALLREDUCE && u->n_src == 1 && ns[0] &&
                 ns[0]->op == POLY_OP_UNSHARD &&
                 ns[0]->arg.kind == POLY_ARG_INT_TUPLE &&
                 ns[0]->n_src == ns[0]->arg.int_tuple.n + 1 &&
                 u->arg.kind == POLY_ARG_ALLREDUCE) {
        PolyUOp *device = poly_uop_device_uop_cached(ctx, u, NULL);
        PolyUOp *local = device
                             ? poly_allreduce(
                                   ctx, ns[0]->src[0], u->arg.allreduce.op, device)
                             : NULL;
        result = multi_pm_restore_sharding(ctx, local, ns[0]);
        if (!result) rule_failed = true;
      } else if (u->op == POLY_OP_AFTER && u->n_src == 2 && ns[0] &&
                 ns[0]->op == POLY_OP_UNSHARD && ns[1] &&
                 ns[1]->op == POLY_OP_STORE && ns[1]->n_src == 2 && ns[1]->src[0] &&
                 ns[1]->src[0]->op == POLY_OP_UNSHARD &&
                 ns[1]->src[1] && ns[1]->src[1]->op == POLY_OP_UNSHARD &&
                 ns[1]->src[1]->arg.kind == POLY_ARG_INT_TUPLE) {
        /* Pinned schedule/multi.py:store_after_multi resolves the assignment
         * locally, then restores the value's shard axis around the AFTER. */
        PolyUOp *dest = ns[1]->src[0]->src[0];
        PolyUOp *value = ns[1]->src[1]->src[0];
        PolyUOp *store = poly_uop2(
            ctx, POLY_OP_STORE, POLY_VOID, dest, value, poly_arg_none());
        PolyUOp *after =
            store ? poly_uop2(ctx, POLY_OP_AFTER, dest->dtype, dest, store, poly_arg_none()) : NULL;
        result = multi_pm_restore_sharding(ctx, after, ns[1]->src[1]);
      } else if (u->op == POLY_OP_GETTUPLE && u->n_src == 1 && ns[0] &&
                 ns[0]->op == POLY_OP_TUPLE && u->arg.kind == POLY_ARG_INT &&
                 u->arg.i >= 0 && u->arg.i < ns[0]->n_src) {
        /* Pinned schedule/multi.py:158-159. */
        result = ns[0]->src[u->arg.i];
      } else if (u->op == POLY_OP_GETTUPLE && u->n_src == 1 && ns[0] &&
                 ns[0]->op == POLY_OP_UNSHARD &&
                 ns[0]->arg.kind == POLY_ARG_INT_TUPLE &&
                 ns[0]->n_src == ns[0]->arg.int_tuple.n + 1 &&
                 u->arg.kind == POLY_ARG_INT &&
                 u->arg.i >= 0 &&
                 (ns[0]->src[0]->op == POLY_OP_FUNCTION ||
                  ns[0]->src[0]->op == POLY_OP_TUPLE)) {
        PolyUOp *aggregate = ns[0]->src[0];
        PolyUOp *tuple = aggregate->op == POLY_OP_FUNCTION && aggregate->n_src > 0
                             ? aggregate->src[0]
                             : aggregate;
        if (!tuple || tuple->op != POLY_OP_TUPLE || u->arg.i >= tuple->n_src) {
          rule_failed = true;
        } else {
          PolyUOp *selected = poly_uop1(
              ctx, POLY_OP_GETTUPLE, tuple->src[u->arg.i]->dtype, aggregate,
              u->arg);
          result = multi_pm_restore_sharding(ctx, selected, ns[0]);
          if (!result) rule_failed = true;
        }
      } else if (u->op == POLY_OP_FUNCTION) {
        result = multi_pm_function(ctx, u, ns, &rule_failed);
      } else if ((u->op == POLY_OP_CALL || u->op == POLY_OP_FUNCTION ||
                  u->op == POLY_OP_AFTER) &&
                 u->n_src >= 1 && ns[0] && ns[0]->op == POLY_OP_UNSHARD) {
        result = multi_pm_passthrough(ctx, u, ns, true, &rule_failed);
      } else if (u->op == POLY_OP_CALL && poly_dtype_eq(u->dtype, POLY_VOID)) {
        bool has_multi_source = false;
        for (int i = 0; i < u->n_src; i++)
          if (ns[i] && ns[i]->op == POLY_OP_UNSHARD) has_multi_source = true;
        if (has_multi_source)
          result = multi_pm_passthrough(ctx, u, ns, false, &rule_failed);
      } else if ((u->op == POLY_OP_CAST || u->op == POLY_OP_BITCAST ||
                  u->op == POLY_OP_CONTIGUOUS || u->op == POLY_OP_DETACH ||
                  u->op == POLY_OP_CONTIGUOUS_BACKWARD) &&
                 u->n_src == 1 && ns[0] && ns[0]->op == POLY_OP_UNSHARD) {
        result = multi_pm_passthrough(ctx, u, ns, true, &rule_failed);
      } else if (u->op == POLY_OP_STORE && u->n_src >= 2 && ns[1] &&
                 ns[1]->op == POLY_OP_UNSHARD) {
        /* Current schedule/multi.py orders store_value_multi before
         * store_dest_multi. A doubly sharded STORE therefore subviews its
         * destination using the value's sharding before resolving the dest. */
        result = multi_pm_store_value(ctx, u, ns[0], ns[1], &rule_failed);
      } else if (u->op == POLY_OP_STORE && u->n_src >= 1 && ns[0] &&
                 ns[0]->op == POLY_OP_UNSHARD) {
        result = multi_pm_store_dest(ctx, u, ns[0], ns, &rule_failed);
      }

      if (rule_failed) {
        if (ns != ns_buf) free(ns);
        poly_map_destroy(rmap);
        poly_toposort_free(topo);
        return NULL;
      }

      if (!result && src_changed) result = multi_clone(ctx, u, ns, u->n_src);
      if (result && result != u) {
        multi_map_set(rmap, u, result);
        changed = true;
      }
      if (ns != ns_buf) free(ns);
    }

    PolyUOp *next = changed ? multi_map_get(rmap, sink) : NULL;
    poly_map_destroy(rmap);
    poly_toposort_free(topo);
    if (!next || next == sink) return sink;
    sink = next;
  }
}
