/*
 * shape.c — Shape inference for tensor-level UOp graphs
 *
 * Computes the output shape (tuple of dimension sizes) for any UOp.
 * Walks the graph in toposort order, caches results per UOp pointer.
 *
 * Reference: tinygrad uop/ops.py lines 206-296 (_shape property)
 */

#include "polygrad.h"
#include "ctx.h"
#include "device.h"
#include "frontend_internal.h"
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pat.h"
#include "utils.h"

/* Local helpers */

/* tinygrad-style graph constructors for shape-bearing UOps live here now.
 * They were previously stranded in the retired single-kernel scheduler alongside the old
 * scheduler wrapper, but they are core IR-building APIs rather than a
 * scheduling implementation detail. */

static bool rank_tuple_valid(const void *data, int n) {
  return n >= 0 && n <= POLY_MAX_DIMS && (n == 0 || data != NULL);
}

static bool shape_value_is_const(PolyUOp *u, int64_t expected) {
  int64_t value = 0;
  return u && u->op == POLY_OP_CONST && poly_uop_const_i64(u, &value) == 0 && value == expected;
}

static bool movement_shape_is_identity(
    PolyCtx *ctx,
    PolyUOp *src,
    PolyUOp **offsets,
    PolyUOp **sizes,
    int ndim
) {
  PolyShape shape = poly_uop_max_shape_cached(ctx, src);
  if (shape.ndim != ndim) return false;
  for (int i = 0; i < ndim; i++) {
    if (!shape_value_is_const(offsets[i], 0)) return false;
    PolyUOp *dim = poly_uop_shape_dim(ctx, src, i);
    if (sizes[i] != dim && !shape_value_is_const(sizes[i], shape.dims[i])) return false;
    if (dim && dim->op != POLY_OP_CONST && sizes[i] != dim) return false;
  }
  return true;
}

PolyUOp *poly_buffer_on_device(
    PolyCtx *ctx,
    PolyDType scalar_dtype,
    int64_t size,
    PolyDevice device
) {
  if (!ctx) return NULL;
  int64_t id = poly_ctx_next_unique_id(ctx);
  PolyUOp *unique = poly_uop0(ctx, POLY_OP_UNIQUE, POLY_VOID, poly_arg_int(id));
  if (device == POLY_DEVICE_AUTO)
    return poly_uop1(ctx, POLY_OP_BUFFER, scalar_dtype, unique, poly_arg_int(size));

  PolyUOp *dev = poly_device_uop(ctx, device);
  PolyUOp *src[2] = {unique, dev};
  return poly_uop(ctx, POLY_OP_BUFFER, scalar_dtype, src, 2, poly_arg_int(size));
}

PolyUOp *poly_buffer(PolyCtx *ctx, PolyDType scalar_dtype, int64_t size) {
  return poly_buffer_on_device(ctx, scalar_dtype, size, POLY_DEVICE_AUTO);
}

/* Pinned tinygrad `shape_to_shape_arg` represents a shape as
 * STACK(CONST(weakint), ...), including rank-0 as STACK(void). Keep this
 * tensor-stage spelling exact so movement topology is directly comparable. */
static PolyUOp *poly_shape_stack(PolyCtx *ctx, PolyUOp **vals, int n) {
  if (!ctx || !rank_tuple_valid(vals, n)) return NULL;
  PolyDType dt =
      n == 0 ? POLY_VOID : (n > 1 ? poly_dtype_vec(POLY_INDEX, n) : POLY_INDEX);
  return poly_uop(ctx, POLY_OP_STACK, dt, vals, n, poly_arg_none());
}

static PolyUOp *poly_static_shape_arg(PolyCtx *ctx, const int64_t *dims, int ndim) {
  if (!ctx || !rank_tuple_valid(dims, ndim)) return NULL;
  PolyUOp *vals[POLY_MAX_DIMS];
  for (int i = 0; i < ndim; i++)
    vals[i] = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(dims[i]));
  return poly_shape_stack(ctx, vals, ndim);
}

PolyUOp *poly_reshape(PolyCtx *ctx, PolyUOp *src, int64_t *dims, int ndim) {
  if (!ctx || !src || !rank_tuple_valid(dims, ndim)) return NULL;
  PolyUOp *shape = poly_static_shape_arg(ctx, dims, ndim);
  if (!shape) return NULL;
  PolyUOp *srcs[2] = {src, shape};
  return poly_uop(ctx, POLY_OP_RESHAPE, src->dtype, srcs, 2, poly_arg_none());
}

PolyUOp *poly_reshape_uop(PolyCtx *ctx, PolyUOp *src, PolyUOp **dims, int ndim) {
  if (!ctx || !src || !rank_tuple_valid(dims, ndim)) return NULL;
  PolyUOp *shape = poly_shape_stack(ctx, dims, ndim);
  if (!shape) return NULL;
  PolyUOp *srcs[2] = {src, shape};
  return poly_uop(ctx, POLY_OP_RESHAPE, src->dtype, srcs, 2, poly_arg_none());
}

PolyUOp *poly_expand(PolyCtx *ctx, PolyUOp *src, int64_t *dims, int ndim) {
  if (!ctx || !src || !rank_tuple_valid(dims, ndim)) return NULL;
  PolyUOp *shape = poly_static_shape_arg(ctx, dims, ndim);
  if (!shape) return NULL;
  PolyUOp *srcs[2] = {src, shape};
  return poly_uop(ctx, POLY_OP_EXPAND, src->dtype, srcs, 2, poly_arg_none());
}

PolyUOp *poly_reduce_axis(
    PolyCtx *ctx,
    PolyOps reduce_op,
    PolyUOp *src,
    int64_t *axes,
    int n_axes
) {
  /* tinygrad UOp._rop sorts axes and drops singleton reductions up front.
   * Matching that here keeps no-op singleton reductions out of the graph
   * instead of relying on later schedule/rangeify cleanup to discover them. */
  if (!ctx || !src) return NULL;
  if (n_axes == 0) return src;
  if (!rank_tuple_valid(axes, n_axes)) return NULL;

  /* Use the ctx-owned cached shape here. poly_reduce_axis only inspects dims;
   * requesting a heap copy would make this hot constructor responsible for
   * extra ownership bookkeeping on every reduce. */
  PolyShape shape = poly_uop_max_shape_cached(ctx, src);
  int64_t filtered_buf[POLY_MAX_DIMS];
  int filtered_n = 0;
  for (int i = 0; i < n_axes; i++) {
    int64_t ax = axes[i];
    if (shape.ndim > 0 && ax >= 0 && ax < shape.ndim && shape.dims[ax] == 1) continue;
    filtered_buf[filtered_n++] = ax;
  }
  if (filtered_n == 0) return src;

  for (int i = 1; i < filtered_n; i++) {
    int64_t ax = filtered_buf[i];
    int j = i - 1;
    while (j >= 0 && filtered_buf[j] > ax) {
      filtered_buf[j + 1] = filtered_buf[j];
      j--;
    }
    filtered_buf[j + 1] = ax;
  }

  int64_t *stored_axes = filtered_buf;
  if (filtered_n > 0) {
    stored_axes =
        poly_arena_alloc(poly_ctx_arena(ctx), filtered_n * sizeof(int64_t), _Alignof(int64_t));
    memcpy(stored_axes, filtered_buf, filtered_n * sizeof(int64_t));
  }

  PolyArg arg;
  arg.kind = POLY_ARG_REDUCE_AXIS;
  arg.reduce_axis.op = reduce_op;
  arg.reduce_axis.axes = stored_axes;
  arg.reduce_axis.n = filtered_n;
  /* Pinned UOp._rop emits REDUCE(value, arg=(op, axes)) before rangeify
   * (uop/ops.py:567-569). POLY_ARG_REDUCE_AXIS is that tensor-stage arg;
   * lowered REDUCE uses POLY_ARG_OPS plus RANGE sources. */
  return poly_uop1(ctx, POLY_OP_REDUCE, src->dtype, src, arg);
}

PolyUOp *poly_permute(PolyCtx *ctx, PolyUOp *src, int64_t *perm, int ndim) {
  if (!ctx || !src || !rank_tuple_valid(perm, ndim)) return NULL;
  PolyArg arg;
  arg.kind = POLY_ARG_INT_TUPLE;
  arg.int_tuple.vals = perm;
  arg.int_tuple.n = ndim;
  return poly_uop1(ctx, POLY_OP_PERMUTE, src->dtype, src, arg);
}

PolyUOp *poly_expand_uop(PolyCtx *ctx, PolyUOp *src, PolyUOp **dims, int ndim) {
  if (!ctx || !src || !rank_tuple_valid(dims, ndim)) return NULL;
  PolyUOp *dim_stack = poly_shape_stack(ctx, dims, ndim);
  if (!dim_stack) return NULL;
  PolyUOp *srcs[2] = {src, dim_stack};
  return poly_uop(ctx, POLY_OP_EXPAND, src->dtype, srcs, 2, poly_arg_none());
}

PolyUOp *poly_shrink_uop(PolyCtx *ctx, PolyUOp *src, PolyUOp **starts, PolyUOp **sizes, int ndim) {
  if (!ctx || !src || !rank_tuple_valid(starts, ndim) || !rank_tuple_valid(sizes, ndim))
    return NULL;
  /* MovementMixin.shrink returns self when the resulting shape is unchanged
   * (mixin/movement.py:192-193). Elide only on proved shape-value identity. */
  if (movement_shape_is_identity(ctx, src, starts, sizes, ndim)) return src;
  PolyUOp *start_stack = poly_shape_stack(ctx, starts, ndim);
  PolyUOp *size_stack = poly_shape_stack(ctx, sizes, ndim);
  if (!start_stack || !size_stack) return NULL;
  PolyUOp *srcs[3] = {src, start_stack, size_stack};
  return poly_uop(ctx, POLY_OP_SHRINK, src->dtype, srcs, 3, poly_arg_none());
}

PolyUOp *poly_pad_uop(PolyCtx *ctx, PolyUOp *src, PolyUOp **offsets, PolyUOp **sizes, int ndim) {
  if (!ctx || !src || !rank_tuple_valid(offsets, ndim) || !rank_tuple_valid(sizes, ndim))
    return NULL;
  /* MovementMixin.pad has the same shape-identity return rule
   * (mixin/movement.py:170-171). */
  if (movement_shape_is_identity(ctx, src, offsets, sizes, ndim)) return src;
  PolyUOp *offset_stack = poly_shape_stack(ctx, offsets, ndim);
  PolyUOp *size_stack = poly_shape_stack(ctx, sizes, ndim);
  if (!offset_stack || !size_stack) return NULL;
  PolyUOp *srcs[3] = {src, offset_stack, size_stack};
  return poly_uop(ctx, POLY_OP_PAD, src->dtype, srcs, 3, poly_arg_none());
}

PolyUOp *poly_shrink(PolyCtx *ctx, PolyUOp *src, int64_t (*pairs)[2], int ndim) {
  if (!ctx || !src || !rank_tuple_valid(pairs, ndim)) return NULL;
  /* Pinned UOp._mop returns scalar PAD/SHRINK unchanged when the movement
   * argument is empty (uop/ops.py:712-713). */
  if (ndim == 0) return src;
  PolyUOp *starts[POLY_MAX_DIMS];
  PolyUOp *sizes[POLY_MAX_DIMS];
  for (int i = 0; i < ndim; i++) {
    int64_t size;
    if (pairs[i][0] < 0 || pairs[i][1] < pairs[i][0] ||
        __builtin_sub_overflow(pairs[i][1], pairs[i][0], &size))
      return NULL;
    starts[i] = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(pairs[i][0]));
    sizes[i] = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(size));
  }
  return poly_shrink_uop(ctx, src, starts, sizes, ndim);
}

PolyUOp *poly_flip(PolyCtx *ctx, PolyUOp *src, int64_t *axes, int n_axes) {
  if (!ctx || !src || !rank_tuple_valid(axes, n_axes)) return NULL;
  PolyShape shape = poly_uop_max_shape_cached(ctx, src);
  if (shape.ndim < 0 || shape.ndim > POLY_MAX_DIMS) return NULL;
  if (n_axes == 0) return src;
  int64_t mask[POLY_MAX_DIMS] = {0};
  for (int i = 0; i < n_axes; i++) {
    int64_t axis = axes[i];
    if (axis < 0 || axis >= shape.ndim || mask[axis]) return NULL;
    mask[axis] = 1;
  }
  PolyArg arg;
  arg.kind = POLY_ARG_INT_TUPLE;
  arg.int_tuple.vals = mask;
  arg.int_tuple.n = shape.ndim;
  return poly_uop1(ctx, POLY_OP_FLIP, src->dtype, src, arg);
}

PolyUOp *poly_pad(PolyCtx *ctx, PolyUOp *src, int64_t (*pairs)[2], int ndim) {
  if (!ctx || !src || !rank_tuple_valid(pairs, ndim)) return NULL;
  if (ndim == 0) return src;
  PolyShape shape = poly_uop_max_shape_cached(ctx, src);
  if (shape.ndim != ndim) return NULL;
  PolyUOp *offsets[POLY_MAX_DIMS];
  PolyUOp *sizes[POLY_MAX_DIMS];
  for (int i = 0; i < ndim; i++) {
    if (pairs[i][0] < 0 || pairs[i][1] < 0) return NULL;
    offsets[i] = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(pairs[i][0]));
    PolyUOp *dim = poly_uop_shape_dim(ctx, src, i);
    if (!dim) dim = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(shape.dims[i]));
    int64_t delta;
    if (__builtin_add_overflow(pairs[i][0], pairs[i][1], &delta)) return NULL;
    if (delta == 0) {
      sizes[i] = dim;
    } else {
      PolyUOp *amount = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(delta));
      sizes[i] = poly_uop2(ctx, POLY_OP_ADD, POLY_INDEX, dim, amount, poly_arg_none());
      sizes[i] = poly_graph_rewrite(ctx, sizes[i], poly_symbolic_simple());
    }
  }
  return poly_pad_uop(ctx, src, offsets, sizes, ndim);
}

/* Heap-allocate a shape with copied dims */
static PolyShape heap_shape(int64_t *dims, int ndim) {
  if (ndim < 0) return POLY_SHAPE_NONE;
  if (ndim == 0) return (PolyShape){NULL, 0};
  int64_t *copy = malloc(ndim * sizeof(int64_t));
  memcpy(copy, dims, ndim * sizeof(int64_t));
  return (PolyShape){copy, ndim};
}

/* Shape cache entry (arena-allocated) */

typedef struct {
  int8_t ndim; /* -1 = no shape, 0 = scalar, >0 = tensor */
  int64_t *dims; /* arena-allocated, NULL if scalar/none */
  PolyUOp **dim_uops; /* arena-allocated symbolic dims, NULL if scalar/none */
} ShapeCacheEntry;

static ShapeCacheEntry *ensure_shape(PolyCtx *ctx, PolyUOp *u);

PolyMap *poly_ctx_shape_cache(PolyCtx *ctx); /* defined in uop.c */

/* Public lazy accessors */

int poly_uop_ndim(PolyCtx *ctx, const PolyUOp *u) {
  if (!u) return -1;
  return ensure_shape(ctx, (PolyUOp *)u)->ndim;
}

const int64_t *poly_uop_max_shape_dims(PolyCtx *ctx, const PolyUOp *u) {
  if (!u) return NULL;
  return ensure_shape(ctx, (PolyUOp *)u)->dims;
}

PolyUOp *poly_uop_shape_dim(PolyCtx *ctx, const PolyUOp *u, int dim) {
  if (!u || dim < 0) return NULL;
  ShapeCacheEntry *e = ensure_shape(ctx, (PolyUOp *)u);
  if (!e || dim >= e->ndim || !e->dim_uops) return NULL;
  return e->dim_uops[dim];
}

static PolyUOp *axis_shape_arg_item(PolyUOp *shape, int axis) {
  if (!shape || axis < 0) return NULL;
  if (shape->op == POLY_OP_STACK)
    return axis < shape->n_src ? shape->src[axis] : NULL;
  return axis == 0 && shape->op == POLY_OP_CONST ? shape : NULL;
}

static bool axis_expr_equal(PolyUOp *a, PolyUOp *b) {
  if (a == b) return true;
  int64_t av = 0, bv = 0;
  return poly_uop_const_i64(a, &av) == 0 && poly_uop_const_i64(b, &bv) == 0 && av == bv;
}

static PolyUOp *axis_shape_product(PolyCtx *ctx, const PolyUOp *u, int end) {
  if (!ctx || !u || end < 0 || end > poly_uop_ndim(ctx, u)) return NULL;
  PolyUOp *product = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(1));
  const int64_t *max_shape = poly_uop_max_shape_dims(ctx, u);
  for (int i = 0; i < end; i++) {
    PolyUOp *dim = poly_uop_shape_dim(ctx, u, i);
    if (!dim && max_shape) dim = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(max_shape[i]));
    if (!dim) return NULL;
    product = poly_uop2(ctx, POLY_OP_MUL, POLY_INDEX, product, dim, poly_arg_none());
    product = product ? poly_graph_rewrite(ctx, product, poly_symbolic_simple()) : NULL;
    if (!product) return NULL;
  }
  return product;
}

/* Exact C port of pinned UOp.axis (tinygrad/uop/ops.py:623-651). The cache
 * stores 1 for None and axis+2 for a concrete shard axis. */
bool poly_uop_axis_cached(PolyCtx *ctx, const PolyUOp *u, PolyMap *cache, int *out_axis) {
  if (out_axis) *out_axis = -1;
  if (!ctx || !u) return false;
  if (cache) {
    void *cached = poly_map_get(cache, poly_ptr_hash(u), u, poly_ptr_eq);
    if (cached) {
      intptr_t encoded = (intptr_t)cached;
      if (encoded == 1) return false;
      if (out_axis) *out_axis = (int)(encoded - 2);
      return true;
    }
  }

  bool has_axis = false;
  int axis = -1;
  if (u->op == POLY_OP_COPY) {
    has_axis = false;
  } else if (u->op == POLY_OP_MULTI && u->arg.kind == POLY_ARG_INT && u->arg.i >= 0 &&
             u->arg.i <= INT_MAX) {
    axis = (int)u->arg.i;
    has_axis = true;
  } else if (u->op == POLY_OP_GETTUPLE && u->n_src == 1 &&
             u->arg.kind == POLY_ARG_INT && u->arg.i >= 0) {
    /* Pinned UOp.axis selects the requested TUPLE result, including through
     * a value-producing FUNCTION (tinygrad/uop/ops.py:628-630). */
    PolyUOp *aggregate = u->src[0];
    PolyUOp *tuple = aggregate && aggregate->op == POLY_OP_FUNCTION && aggregate->n_src > 0
                         ? aggregate->src[0]
                         : aggregate;
    if (tuple && tuple->op == POLY_OP_TUPLE && u->arg.i < tuple->n_src)
      has_axis = poly_uop_axis_cached(ctx, tuple->src[u->arg.i], cache, &axis);
  } else if (u->op == POLY_OP_PARAM && u->arg.kind == POLY_ARG_PARAM && u->arg.param &&
             u->arg.param->has_axis && u->arg.param->axis >= 0) {
    axis = u->arg.param->axis;
    has_axis = true;
  } else if (poly_opset_has(POLY_GROUP_ALU, u->op)) {
    /* Pinned dedup(...)[-1] chooses the last non-None source axis. */
    for (int i = 0; i < u->n_src; i++) {
      int source_axis = -1;
      if (poly_uop_axis_cached(ctx, u->src[i], cache, &source_axis)) {
        axis = source_axis;
        has_axis = true;
      }
    }
  } else if (u->n_src > 0) {
    has_axis = poly_uop_axis_cached(ctx, u->src[0], cache, &axis);
    if (has_axis && u->op == POLY_OP_SHRINK) {
      bool full = false;
      if (u->arg.kind == POLY_ARG_NONE && u->n_src >= 3) {
        PolyUOp *start = axis_shape_arg_item(u->src[1], axis);
        PolyUOp *size = axis_shape_arg_item(u->src[2], axis);
        PolyUOp *source_dim = poly_uop_shape_dim(ctx, u->src[0], axis);
        int64_t start_value = -1;
        full = poly_uop_const_i64(start, &start_value) == 0 && start_value == 0 &&
               axis_expr_equal(size, source_dim);
      } else if (u->arg.kind == POLY_ARG_PAIR_TUPLE &&
                 axis < u->arg.pair_tuple.n) {
        const int64_t *source_shape = poly_uop_max_shape_dims(ctx, u->src[0]);
        full = source_shape && u->arg.pair_tuple.pairs[axis][0] == 0 &&
               u->arg.pair_tuple.pairs[axis][1] == source_shape[axis];
      }
      if (!full) has_axis = false;
    } else if (has_axis && (u->op == POLY_OP_REDUCE || u->op == POLY_OP_REDUCE_AXIS) &&
               u->arg.kind == POLY_ARG_REDUCE_AXIS) {
      for (int i = 0; i < u->arg.reduce_axis.n; i++)
        if (u->arg.reduce_axis.axes[i] == axis) {
          has_axis = false;
          break;
        }
    } else if (has_axis && u->op == POLY_OP_RESHAPE) {
      PolyUOp *source_prefix = axis_shape_product(ctx, u->src[0], axis);
      PolyUOp *output_prefix = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(1));
      int output_ndim = poly_uop_ndim(ctx, u);
      int new_axis = axis_expr_equal(output_prefix, source_prefix) ? 0 : -1;
      for (int i = 0; i < output_ndim; i++) {
        PolyUOp *dim = poly_uop_shape_dim(ctx, u, i);
        const int64_t *max_shape = poly_uop_max_shape_dims(ctx, u);
        if (!dim && max_shape)
          dim = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(max_shape[i]));
        output_prefix = dim
                            ? poly_uop2(
                                  ctx, POLY_OP_MUL, POLY_INDEX, output_prefix, dim,
                                  poly_arg_none()
                              )
                            : NULL;
        output_prefix = output_prefix
                            ? poly_graph_rewrite(ctx, output_prefix, poly_symbolic_simple())
                            : NULL;
        if (!output_prefix) break;
        if (axis_expr_equal(output_prefix, source_prefix)) new_axis = i + 1;
      }
      PolyUOp *device = poly_uop_device_uop_cached(ctx, (PolyUOp *)u, NULL);
      int device_count = device && device->arg.kind == POLY_ARG_STRING_TUPLE
                             ? device->arg.string_tuple.n
                             : 0;
      if (!source_prefix || new_axis < 0 || new_axis >= output_ndim || device_count <= 0) {
        has_axis = false;
      } else {
        PolyUOp *new_dim = poly_uop_shape_dim(ctx, u, new_axis);
        const int64_t *max_shape = poly_uop_max_shape_dims(ctx, u);
        if (!new_dim && max_shape)
          new_dim = poly_uop0(
              ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(max_shape[new_axis])
          );
        PolyUOp *count = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(device_count));
        PolyUOp *rem = new_dim
                           ? poly_uop2(ctx, POLY_OP_MOD, POLY_INDEX, new_dim, count, poly_arg_none())
                           : NULL;
        rem = rem ? poly_graph_rewrite(ctx, rem, poly_symbolic_simple()) : NULL;
        int64_t rem_value = -1;
        if (poly_uop_const_i64(rem, &rem_value) != 0 || rem_value != 0) {
          has_axis = false;
        } else {
          axis = new_axis;
        }
      }
    } else if (has_axis && u->op == POLY_OP_PERMUTE &&
               u->arg.kind == POLY_ARG_INT_TUPLE) {
      int new_axis = -1;
      for (int i = 0; i < u->arg.int_tuple.n; i++)
        if (u->arg.int_tuple.vals[i] == axis) {
          new_axis = i;
          break;
        }
      if (new_axis < 0)
        has_axis = false;
      else
        axis = new_axis;
    }
  }

  if (cache)
    poly_map_set(
        cache, poly_ptr_hash(u), (void *)u,
        (void *)(intptr_t)(has_axis ? axis + 2 : 1), poly_ptr_eq
    );
  if (has_axis && out_axis) *out_axis = axis;
  return has_axis;
}

bool poly_uop_axis(PolyCtx *ctx, const PolyUOp *u, int *out_axis) {
  PolyMap *cache = poly_map_new(64);
  if (!cache) {
    if (out_axis) *out_axis = -1;
    return false;
  }
  bool ret = poly_uop_axis_cached(ctx, u, cache, out_axis);
  poly_map_destroy(cache);
  return ret;
}

int poly_uop_const_i64(const PolyUOp *u, int64_t *out) {
  if (!u || u->op != POLY_OP_CONST || u->arg.kind != POLY_ARG_INT) return -1;
  if (out) *out = u->arg.i;
  return 0;
}

PolyUOp *poly_uop_unbind_var(PolyUOp *u) {
  if (!u) return NULL;
  if (u->op == POLY_OP_DEFINE_VAR) return u;
  if (u->op == POLY_OP_BIND && u->n_src >= 1 && u->src[0]->op == POLY_OP_DEFINE_VAR)
    return u->src[0];
  return NULL;
}

int poly_uop_bind_value(PolyUOp *u, int64_t *out) {
  if (!u) return -1;
  if (u->op == POLY_OP_CONST) return poly_uop_const_i64(u, out);
  if (u->op == POLY_OP_BIND && u->n_src >= 2) return poly_uop_const_i64(u->src[1], out);
  if ((u->op == POLY_OP_ADD || u->op == POLY_OP_SUB || u->op == POLY_OP_MUL) && u->n_src >= 2) {
    int64_t a = 0, b = 0, r = 0;
    if (poly_uop_bind_value(u->src[0], &a) != 0 || poly_uop_bind_value(u->src[1], &b) != 0)
      return -1;
    bool ov = false;
    if (u->op == POLY_OP_ADD) ov = __builtin_add_overflow(a, b, &r);
    else if (u->op == POLY_OP_SUB) ov = __builtin_sub_overflow(a, b, &r);
    else ov = __builtin_mul_overflow(a, b, &r);
    if (ov) return -1;
    if (out) *out = r;
    return 0;
  }
  return -1;
}

/* Pinned UOp.as_shape (uop/ops.py:697-700): CONST represents dtype.count
 * repeated lanes, STACK exposes each source, and any other UOp is one symbolic
 * dimension. */
static bool shape_arg_values(
    PolyCtx *ctx, PolyUOp *shape_arg, int64_t *dims, PolyUOp **dim_uops, int *n_out
) {
  if (!shape_arg) return false;
  if (shape_arg->op == POLY_OP_STACK && shape_arg->n_src > POLY_MAX_DIMS) return false;

  int n = shape_arg->op == POLY_OP_STACK
              ? shape_arg->n_src
              : (shape_arg->op == POLY_OP_CONST ? shape_arg->dtype.count : 1);
  if (n < 0 || n > POLY_MAX_DIMS) return false;

  for (int i = 0; i < n; i++) {
    PolyUOp *sz = shape_arg->op == POLY_OP_STACK ? shape_arg->src[i] : shape_arg;
    /* Pinned tinygrad/uop/ops.py:697-705 shape_value.as_shape calls
     * ssimplify(), which runs the full symbolic matcher for every lane. */
    sz = poly_graph_rewrite(ctx, sz, poly_symbolic());
    if (!sz) return false;
    int64_t v = 0;
    bool is_const = poly_uop_const_i64(sz, &v) == 0;
    if (!is_const && sz && sz->op == POLY_OP_CONST &&
        sz->arg.kind == POLY_ARG_INT_TUPLE && i < sz->arg.int_tuple.n) {
      v = sz->arg.int_tuple.vals[i];
      is_const = true;
    }
    if (is_const) {
      /* Pinned movement shape validation rejects negative target dimensions
       * (uop/ops.py:330-336). */
      if (v < 0) return false;
      dims[i] = v;
      if (dim_uops) dim_uops[i] = NULL;
      continue;
    }
    int64_t vmin = 0, vmax = 0;
    poly_uop_minmax(ctx, sz, &vmin, &vmax);
    if (vmin < 0 || vmax < 0) return false;
    dims[i] = vmax;
    if (dim_uops) dim_uops[i] = sz;
  }
  if (n_out) *n_out = n;
  return true;
}

static bool shape_dim_is_static(PolyUOp *u) {
  int64_t unused = 0;
  return !u || poly_uop_const_i64(u, &unused) == 0;
}

static PolyUOp *canonical_shape_dim(
    PolyCtx *ctx, int64_t max_dim, PolyUOp *dim_uop
) {
  PolyUOp *dim = dim_uop
                     ? dim_uop
                     : poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(max_dim));
  if (dim && !poly_dtype_eq(poly_dtype_scalar(dim->dtype), POLY_INDEX))
    dim = poly_uop1(ctx, POLY_OP_CAST, POLY_INDEX, dim, poly_arg_none());
  /* Pinned UOp.ssimplify uses the complete symbolic matcher
   * (uop/ops.py:427-434), including commutative canonicalization. */
  return dim ? poly_graph_rewrite(ctx, dim, poly_symbolic()) : NULL;
}

static PolyUOp *exact_shape_product(
    PolyCtx *ctx, const int64_t *max_dims, PolyUOp *const *dim_uops, int ndim
) {
  if (!ctx || ndim < 0 || ndim > POLY_MAX_DIMS || (ndim > 0 && !max_dims)) return NULL;
  PolyUOp *product = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(1));
  for (int i = 0; i < ndim; i++) {
    PolyUOp *dim =
        canonical_shape_dim(ctx, max_dims[i], dim_uops ? dim_uops[i] : NULL);
    if (!dim) return NULL;
    product =
        poly_uop2(ctx, POLY_OP_MUL, POLY_INDEX, product, dim, poly_arg_none());
    if (!product) return NULL;
  }
  return poly_graph_rewrite(ctx, product, poly_symbolic());
}

static bool expand_dim_compatible(
    PolyCtx *ctx,
    int64_t in_dim,
    PolyUOp *in_uop,
    int64_t out_dim,
    PolyUOp *out_uop
) {
  /* Pinned uop/ops.py:334-336 compares canonical exact dimensions:
   * s == ns or s == 1. Allocation maxima are not semantic dimensions. */
  PolyUOp *in_exact = canonical_shape_dim(ctx, in_dim, in_uop);
  PolyUOp *out_exact = canonical_shape_dim(ctx, out_dim, out_uop);
  if (!in_exact || !out_exact) return false;
  int64_t in_const = 0;
  if (poly_uop_const_i64(in_exact, &in_const) == 0 && in_const == 1) return true;
  return in_exact == out_exact;
}

PolyShape poly_uop_max_shape_cached(PolyCtx *ctx, const PolyUOp *u) {
  if (!u) return POLY_SHAPE_NONE;
  ShapeCacheEntry *e = ensure_shape(ctx, (PolyUOp *)u);
  return (PolyShape){e->dims, e->ndim};
}

/* Public API */

int64_t poly_shape_numel(PolyShape s) {
  if (s.ndim <= 0) return (s.ndim == 0) ? 1 : 0;
  if (!s.dims || s.ndim > POLY_MAX_DIMS) return -1;

  bool has_zero = false;
  for (int i = 0; i < s.ndim; i++) {
    if (s.dims[i] < 0) return -1;
    if (s.dims[i] == 0) has_zero = true;
  }
  if (has_zero) return 0;

  int64_t prod = 1;
  for (int i = 0; i < s.ndim; i++) {
    if (prod > INT64_MAX / s.dims[i]) return -1;
    prod *= s.dims[i];
  }
  return prod;
}

bool poly_shape_eq(PolyShape a, PolyShape b) {
  if (a.ndim != b.ndim) return false;
  if (a.ndim <= 0) return true;
  return memcmp(a.dims, b.dims, a.ndim * sizeof(int64_t)) == 0;
}

/* Main entry point */

/* Delegates to the ctx-cached engine (ensure_shape) and heap-copies the
 * result.  Callers that received a PolyShape from this function must free
 * dims when ndim > 0.  New code should prefer poly_uop_max_shape_cached()
 * or poly_uop_ndim()/poly_uop_max_shape_dims() which avoid the heap copy. */
PolyShape poly_uop_max_shape(PolyCtx *ctx, PolyUOp *u) {
  PolyShape cached = poly_uop_max_shape_cached(ctx, (const PolyUOp *)u);
  return heap_shape(cached.dims, cached.ndim);
}

/* ═══════════════════════════════════════════════════════════════════════ */
/*  Lazy shape computation -- cached on PolyCtx, computed on first access */
/*  Corrected rules verified against tinygrad ops.py:206-318              */
/* ═══════════════════════════════════════════════════════════════════════ */

static ShapeCacheEntry *make_entry_none(PolyCtx *ctx) {
  ShapeCacheEntry *e = poly_arena_alloc(poly_ctx_arena(ctx), sizeof(ShapeCacheEntry), 8);
  e->ndim = -1;
  e->dims = NULL;
  e->dim_uops = NULL;
  return e;
}

static ShapeCacheEntry *make_entry_scalar(PolyCtx *ctx) {
  ShapeCacheEntry *e = poly_arena_alloc(poly_ctx_arena(ctx), sizeof(ShapeCacheEntry), 8);
  e->ndim = 0;
  e->dims = NULL;
  e->dim_uops = NULL;
  return e;
}

static PolyUOp *shape_dim_const(PolyCtx *ctx, int64_t value) {
  /* Pinned shape_to_shape_arg/sint_to_uop use weakint for every Python
   * integer dimension (uop/ops.py:85-89,1652). */
  return poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(value));
}

static ShapeCacheEntry *make_entry_dims_uops(
    PolyCtx *ctx,
    const int64_t *dims,
    PolyUOp *const *dim_uops,
    int ndim
) {
  if (ndim < 0 || ndim > POLY_MAX_DIMS || (ndim > 0 && !dims)) return make_entry_none(ctx);
  ShapeCacheEntry *e = poly_arena_alloc(poly_ctx_arena(ctx), sizeof(ShapeCacheEntry), 8);
  e->ndim = (int8_t)ndim;
  if (ndim > 0) {
    e->dims = poly_arena_alloc(poly_ctx_arena(ctx), ndim * sizeof(int64_t), _Alignof(int64_t));
    memcpy(e->dims, dims, ndim * sizeof(int64_t));
    e->dim_uops =
        poly_arena_alloc(poly_ctx_arena(ctx), ndim * sizeof(PolyUOp *), _Alignof(PolyUOp *));
    for (int i = 0; i < ndim; i++)
      e->dim_uops[i] = dim_uops && dim_uops[i] ? dim_uops[i] : shape_dim_const(ctx, dims[i]);
  } else {
    e->dims = NULL;
    e->dim_uops = NULL;
  }
  return e;
}

static ShapeCacheEntry *make_entry_1d(PolyCtx *ctx, int64_t dim0) {
  return make_entry_dims_uops(ctx, &dim0, NULL, 1);
}

static ShapeCacheEntry *make_entry_dims(PolyCtx *ctx, const int64_t *dims, int ndim) {
  return make_entry_dims_uops(ctx, dims, NULL, ndim);
}

static ShapeCacheEntry *shape_cache_lookup(PolyCtx *ctx, PolyUOp *u) {
  if (!ctx || !u) return NULL;
  PolyMap *cache = poly_ctx_shape_cache(ctx);
  return poly_map_get(cache, poly_ptr_hash(u), u, poly_ptr_eq);
}

static int8_t src_ndim(PolyCtx *ctx, PolyUOp *u, int idx) {
  if (!u || idx < 0 || idx >= u->n_src) return -1;
  ShapeCacheEntry *e = shape_cache_lookup(ctx, u->src[idx]);
  return e ? e->ndim : -1;
}

static const int64_t *src_dims(PolyCtx *ctx, PolyUOp *u, int idx) {
  if (!u || idx < 0 || idx >= u->n_src) return NULL;
  ShapeCacheEntry *e = shape_cache_lookup(ctx, u->src[idx]);
  return e ? e->dims : NULL;
}

static PolyUOp *const *src_dim_uops(PolyCtx *ctx, PolyUOp *u, int idx) {
  if (!u || idx < 0 || idx >= u->n_src) return NULL;
  ShapeCacheEntry *e = shape_cache_lookup(ctx, u->src[idx]);
  return e ? (PolyUOp *const *)e->dim_uops : NULL;
}

/* Read source shape from the ctx cache. ensure_shape() fills the cache in
 * topological order, matching tinygrad's recursive_property behavior without
 * recursive C calls through deep UOp chains. */
#define SRC_NDIM(i) src_ndim(ctx, u, (i))
#define SRC_DIMS(i) src_dims(ctx, u, (i))
#define SRC_DIM_UOPS(i) src_dim_uops(ctx, u, (i))

static ShapeCacheEntry *compute_and_cache(PolyCtx *ctx, PolyUOp *u);

/* Pinned GETTUPLE(FUNCTION)._shape rewrites every symbolic dimension PARAM
 * with the FUNCTION's ordered argument at ParamArg.slot
 * (tinygrad/uop/ops.py:242-253,1691). Keep this query pass-local: it derives
 * a shape expression and does not add graph or lifecycle state. */
static PolyUOp *shape_resolve_function_dim(
    PolyCtx *ctx, PolyUOp *dim, PolyUOp *function
) {
  if (!ctx || !dim || !function || function->op != POLY_OP_FUNCTION ||
      function->n_src < 1)
    return NULL;
  int n_topo = 0;
  PolyUOp **topo = poly_toposort_ex_alloc(ctx, dim, &n_topo, NULL, false);
  if (!topo) return NULL;
  PolyUOp **from = n_topo > 0 ? malloc((size_t)n_topo * sizeof(*from)) : NULL;
  PolyUOp **to = n_topo > 0 ? malloc((size_t)n_topo * sizeof(*to)) : NULL;
  if (n_topo > 0 && (!from || !to)) {
    free(from);
    free(to);
    poly_toposort_free(topo);
    return NULL;
  }

  int n_sub = 0;
  bool valid = true;
  for (int i = 0; i < n_topo; i++) {
    PolyUOp *param = topo[i];
    if (!param || param->op != POLY_OP_PARAM) continue;
    if (param->arg.kind != POLY_ARG_PARAM || !param->arg.param ||
        param->arg.param->slot < 0 || param->arg.param->slot >= function->n_src - 1) {
      valid = false;
      break;
    }
    from[n_sub] = param;
    to[n_sub] = function->src[1 + param->arg.param->slot];
    n_sub++;
  }

  PolyUOp *resolved = dim;
  if (valid && n_sub > 0 &&
      poly_uop_substitute_many(ctx, &dim, 1, from, to, n_sub, &resolved) != 0)
    valid = false;
  free(from);
  free(to);
  poly_toposort_free(topo);
  return valid ? resolved : NULL;
}

static ShapeCacheEntry *ensure_shape(PolyCtx *ctx, PolyUOp *u) {
  ShapeCacheEntry *cached = shape_cache_lookup(ctx, u);
  if (cached) return cached;

  PolyScratchMark scratch = poly_ctx_scratch_mark(ctx);
  int n_topo = 0;
  PolyUOp **topo = poly_toposort_scratch(ctx, u, &n_topo);
  PolyMap *cache = poly_ctx_shape_cache(ctx);
  if (!topo) {
    poly_ctx_scratch_rewind(ctx, scratch);
    ShapeCacheEntry *entry = make_entry_none(ctx);
    poly_map_set(cache, poly_ptr_hash(u), u, entry, poly_ptr_eq);
    return entry;
  }

  for (int i = 0; i < n_topo; i++) {
    PolyUOp *cur = topo[i];
    if (shape_cache_lookup(ctx, cur)) continue;
    ShapeCacheEntry *entry = compute_and_cache(ctx, cur);
    poly_map_set(cache, poly_ptr_hash(cur), cur, entry, poly_ptr_eq);
  }

  cached = shape_cache_lookup(ctx, u);
  if (!cached) {
    ShapeCacheEntry *entry = make_entry_none(ctx);
    poly_map_set(cache, poly_ptr_hash(u), u, entry, poly_ptr_eq);
    cached = entry;
  }
  poly_ctx_scratch_rewind(ctx, scratch);
  return cached;
}

static ShapeCacheEntry *compute_and_cache(PolyCtx *ctx, PolyUOp *u) {
  PolyOps op = u->op;

  /* Pinned GETTUPLE extracts shape from the requested TUPLE element. A
   * FUNCTION selector additionally resolves symbolic dimension PARAMs from
   * the ordered call arguments before exposing allocation maxima. */
  if (op == POLY_OP_GETTUPLE && u->n_src == 1 && u->arg.kind == POLY_ARG_INT &&
      u->arg.i >= 0) {
    PolyUOp *aggregate = u->src[0];
    PolyUOp *function = aggregate && aggregate->op == POLY_OP_FUNCTION ? aggregate : NULL;
    PolyUOp *tuple = function && function->n_src > 0 ? function->src[0] : aggregate;
    if (!tuple || tuple->op != POLY_OP_TUPLE || u->arg.i >= tuple->n_src)
      return make_entry_none(ctx);
    ShapeCacheEntry *selected = shape_cache_lookup(ctx, tuple->src[u->arg.i]);
    if (!selected || selected->ndim < 0) return make_entry_none(ctx);
    if (!function || selected->ndim == 0)
      return make_entry_dims_uops(ctx, selected->dims, selected->dim_uops, selected->ndim);

    int64_t dims[POLY_MAX_DIMS];
    PolyUOp *dim_uops[POLY_MAX_DIMS];
    for (int i = 0; i < selected->ndim; i++) {
      PolyUOp *dim = selected->dim_uops ? selected->dim_uops[i] : NULL;
      if (!dim) dim = shape_dim_const(ctx, selected->dims[i]);
      dim = shape_resolve_function_dim(ctx, dim, function);
      if (!dim) return make_entry_none(ctx);
      int64_t value = 0;
      if (poly_uop_const_i64(dim, &value) == 0) {
        if (value < 0) return make_entry_none(ctx);
        dims[i] = value;
      } else {
        int64_t vmin = 0, vmax = 0;
        poly_uop_minmax(ctx, dim, &vmin, &vmax);
        if (vmin < 0 || vmax < 0) return make_entry_none(ctx);
        dims[i] = vmax;
      }
      dim_uops[i] = dim;
    }
    return make_entry_dims_uops(ctx, dims, dim_uops, selected->ndim);
  }

  /* STORE: inherit shape from value (src[1]) */
  if (op == POLY_OP_STORE && u->n_src >= 2 && SRC_NDIM(1) >= 0)
    return make_entry_dims_uops(ctx, SRC_DIMS(1), SRC_DIM_UOPS(1), SRC_NDIM(1));

  /* Late ops with no shape. RANGE and SPECIAL are scalar-shaped in pinned
   * tinygrad and are handled with the other scalar symbolic values below. */
  if (op == POLY_OP_LOAD || op == POLY_OP_SINK || op == POLY_OP_IF ||
      op == POLY_OP_ENDIF || op == POLY_OP_BARRIER || op == POLY_OP_VECTORIZE ||
      op == POLY_OP_GEP || op == POLY_OP_LINEAR ||
      op == POLY_OP_PROGRAM || op == POLY_OP_SOURCE || op == POLY_OP_BINARY || op == POLY_OP_INS ||
      op == POLY_OP_CUSTOM || op == POLY_OP_CUSTOMI || op == POLY_OP_UNIQUE ||
      op == POLY_OP_LUNIQUE || op == POLY_OP_UNROLL || op == POLY_OP_CONTRACT ||
      op == POLY_OP_VCAT || op == POLY_OP_PTRCAT || op == POLY_OP_CALL) {
    return make_entry_none(ctx);
  }

  /* Scalar constants and symbolic index values. */
  if (op == POLY_OP_CONST || op == POLY_OP_VCONST || op == POLY_OP_DEFINE_VAR ||
      op == POLY_OP_BIND || op == POLY_OP_RANGE || op == POLY_OP_SPECIAL) {
    return make_entry_scalar(ctx);
  }

  /* BUFFER */
  if (op == POLY_OP_BUFFER) {
    if (u->arg.kind == POLY_ARG_INT) {
      /* Dynamic buffer: BUFFER(src=(UNIQUE, DEFINE_VAR/BIND, CONST...)).
       * dims stores max allocation shape; dim_uops stores the symbolic shape
       * expression, matching tinygrad's shape tuple carrying BIND/PARAM UOps. */
      PolyUOp *dynamic_var = u->n_src >= 2 ? poly_uop_unbind_var(u->src[1]) : NULL;
      if (dynamic_var) {
        int ndim = u->n_src - 1;
        if (ndim <= 0 || ndim > POLY_MAX_DIMS) return make_entry_none(ctx);
        int64_t dims[POLY_MAX_DIMS];
        PolyUOp *dim_uops[POLY_MAX_DIMS];
        dims[0] = dynamic_var->arg.define_var.max_val;
        dim_uops[0] = u->src[1];
        for (int i = 1; i < ndim && i < POLY_MAX_DIMS; i++) {
          if (u->src[1 + i]->op != POLY_OP_CONST || u->src[1 + i]->arg.kind != POLY_ARG_INT)
            return make_entry_none(ctx);
          dims[i] = u->src[1 + i]->arg.i;
          dim_uops[i] = u->src[1 + i];
        }
        return make_entry_dims_uops(ctx, dims, dim_uops, ndim);
      } else {
        return make_entry_1d(ctx, u->arg.i);
      }
    } else {
      return make_entry_none(ctx);
    }
  }

  /* BUFFER_VIEW: 1D shape from arg tuple first element */
  if (op == POLY_OP_BUFFER_VIEW && u->arg.kind == POLY_ARG_INT_TUPLE && u->arg.int_tuple.n > 0)
    return make_entry_1d(ctx, u->arg.int_tuple.vals[0]);

  /* DEFINE_LOCAL, DEFINE_REG: shape from pointer dtype size */
  if (op == POLY_OP_DEFINE_LOCAL || op == POLY_OP_DEFINE_REG) {
    if (u->dtype.is_ptr && u->dtype.ptr_size > 0) {
      return make_entry_1d(ctx, u->dtype.ptr_size);
    } else {
      return make_entry_none(ctx);
    }
  }

  /* tinygrad uop/ops.py::UOp._shape, Ops.PARAM:
   * pointer PARAMs use their pointer extent; non-pointer PARAMs use
   * src[0].as_shape. This covers both shaped CALL values (STACK) and the
   * final scalar PARAM(CONST(ptr_size)) emitted by pm_remove_vec_dtypes. */
  if (op == POLY_OP_PARAM) {
    if (u->dtype.is_ptr && u->dtype.ptr_size >= 0) {
      return make_entry_1d(ctx, u->dtype.ptr_size);
    } else if (u->n_src >= 1 && u->src[0] && u->src[0]->op == POLY_OP_STACK) {
      int64_t dims[POLY_MAX_DIMS];
      PolyUOp *dim_uops[POLY_MAX_DIMS];
      int ndim = 0;
      if (shape_arg_values(ctx, u->src[0], dims, dim_uops, &ndim))
        return make_entry_dims_uops(ctx, dims, dim_uops, ndim);
      return make_entry_none(ctx);
    } else if (u->n_src >= 1 && u->src[0]) {
      /* Pinned tinygrad/uop/ops.py:697-700: a scalar shape source uses the
       * same full ssimplify path as each STACK lane. */
      PolyUOp *dim = poly_graph_rewrite(ctx, u->src[0], poly_symbolic());
      if (!dim) return make_entry_none(ctx);
      int64_t value = 0;
      if (poly_uop_const_i64(dim, &value) == 0)
        return value >= 0 ? make_entry_1d(ctx, value) : make_entry_none(ctx);
      int64_t vmin = 0, vmax = 0;
      poly_uop_minmax(ctx, dim, &vmin, &vmax);
      if (vmin < 0 || vmax < 0) return make_entry_none(ctx);
      return make_entry_dims_uops(ctx, &vmax, &dim, 1);
    }
    return make_entry_none(ctx);
  }

  /* Pinned tinygrad INDEX shape is independent of pointer dtype:
   *   concat(index.shape for index in src[1:]) + src[0].shape[n_indices:]
   * Scalar indexes therefore consume source axes without contributing axes. */
  if (op == POLY_OP_INDEX) {
    if (u->n_src < 1 || SRC_NDIM(0) < 0) return make_entry_none(ctx);
    int8_t source_ndim = SRC_NDIM(0);
    int n_indices = u->n_src - 1;
    int tail_start = n_indices < source_ndim ? n_indices : source_ndim;
    int out_ndim = source_ndim - tail_start;
    for (int i = 1; i < u->n_src; i++) {
      int8_t index_ndim = SRC_NDIM(i);
      if (index_ndim < 0 || out_ndim > POLY_MAX_DIMS - index_ndim)
        return make_entry_none(ctx);
      out_ndim += index_ndim;
    }

    int64_t dims[POLY_MAX_DIMS];
    PolyUOp *dim_uops[POLY_MAX_DIMS];
    int pos = 0;
    for (int i = 1; i < u->n_src; i++) {
      PolyUOp *const *index_dim_uops = SRC_DIM_UOPS(i);
      for (int d = 0; d < SRC_NDIM(i); d++) {
        dims[pos] = SRC_DIMS(i)[d];
        dim_uops[pos++] = index_dim_uops ? index_dim_uops[d] : NULL;
      }
    }
    PolyUOp *const *source_dim_uops = SRC_DIM_UOPS(0);
    for (int d = tail_start; d < source_ndim; d++) {
      dims[pos] = SRC_DIMS(0)[d];
      dim_uops[pos++] = source_dim_uops ? source_dim_uops[d] : NULL;
    }
    return make_entry_dims_uops(ctx, dims, dim_uops, out_ndim);
  }

  /* Pinned tinygrad STAGE adds closed-range extents to the front of the
   * existing value shape. Extents are `range.vmax + 1`, which also handles a
   * singleton CONST(0) and symbolic RANGE bounds without syntax heuristics. */
  if (op == POLY_OP_STAGE) {
    if (u->n_src < 1 || SRC_NDIM(0) < 0) return make_entry_none(ctx);
    int n_ranges = u->n_src - 1;
    int source_ndim = SRC_NDIM(0);
    if (n_ranges > POLY_MAX_DIMS - source_ndim) return make_entry_none(ctx);
    int out_ndim = n_ranges + source_ndim;
    int64_t dims[POLY_MAX_DIMS];
    PolyUOp *dim_uops[POLY_MAX_DIMS];
    for (int i = 0; i < n_ranges; i++) {
      int64_t vmin = 0, vmax = 0;
      poly_uop_minmax(ctx, u->src[1 + i], &vmin, &vmax);
      (void)vmin;
      if (__builtin_add_overflow(vmax, (int64_t)1, &dims[i])) return make_entry_none(ctx);
      dim_uops[i] = NULL;
    }
    PolyUOp *const *source_dim_uops = SRC_DIM_UOPS(0);
    for (int i = 0; i < source_ndim; i++) {
      dims[n_ranges + i] = SRC_DIMS(0)[i];
      dim_uops[n_ranges + i] = source_dim_uops ? source_dim_uops[i] : NULL;
    }
    return make_entry_dims_uops(ctx, dims, dim_uops, out_ndim);
  }

  /* Pinned UOp._shape treats MSTACK, MSELECT, and ALLREDUCE as source-0
   * passthroughs (uop/ops.py:308-315). */
  if (op == POLY_OP_MSTACK || op == POLY_OP_MSELECT || op == POLY_OP_ALLREDUCE) {
    if (u->n_src >= 1 && SRC_NDIM(0) >= 0)
      return make_entry_dims_uops(ctx, SRC_DIMS(0), SRC_DIM_UOPS(0), SRC_NDIM(0));
    return make_entry_none(ctx);
  }

  /* Pinned MULTI is movement-like: src[0] is one local shard and the public
   * shape multiplies the shard axis by the exact tuple-device count
   * (uop/ops.py:322-355,612-625). */
  if (op == POLY_OP_MULTI) {
    if (u->n_src != 1 || u->arg.kind != POLY_ARG_INT || SRC_NDIM(0) < 0)
      return make_entry_none(ctx);
    PolyUOp *device = poly_uop_device_uop_cached(ctx, u, NULL);
    int axis = (int)u->arg.i;
    if (!device || device->arg.kind != POLY_ARG_STRING_TUPLE || u->arg.i != axis || axis < 0 ||
        axis >= SRC_NDIM(0))
      return make_entry_none(ctx);
    int64_t dims[POLY_MAX_DIMS];
    PolyUOp *dim_uops[POLY_MAX_DIMS];
    memcpy(dims, SRC_DIMS(0), (size_t)SRC_NDIM(0) * sizeof(*dims));
    PolyUOp *const *source_dim_uops = SRC_DIM_UOPS(0);
    for (int i = 0; i < SRC_NDIM(0); i++)
      dim_uops[i] = source_dim_uops ? source_dim_uops[i] : NULL;
    int count = device->arg.string_tuple.n;
    if (__builtin_mul_overflow(dims[axis], (int64_t)count, &dims[axis]))
      return make_entry_none(ctx);
    if (count == 0) {
      dim_uops[axis] = NULL;
    } else if (count > 1) {
      if (shape_dim_is_static(dim_uops[axis])) {
        dim_uops[axis] = shape_dim_const(ctx, dims[axis]);
      } else {
        dim_uops[axis] =
            poly_alu2(ctx, POLY_OP_MUL, dim_uops[axis], shape_dim_const(ctx, count));
        if (!dim_uops[axis]) return make_entry_none(ctx);
      }
    }
    return make_entry_dims_uops(ctx, dims, dim_uops, SRC_NDIM(0));
  }

  /* RESHAPE: pinned tensor form is (value, shape_to_shape_arg(shape)). */
  if (op == POLY_OP_RESHAPE && u->arg.kind == POLY_ARG_NONE && u->n_src == 2) {
    int8_t in_ndim = SRC_NDIM(0);
    if (in_ndim < 0) return make_entry_none(ctx);
    int64_t dims[POLY_MAX_DIMS];
    PolyUOp *dim_uops[POLY_MAX_DIMS];
    int n = 0;
    if (!shape_arg_values(ctx, u->src[1], dims, dim_uops, &n)) return make_entry_none(ctx);
    /* Pinned uop/ops.py:330-333 requires exact input/output cardinality.
     * Compare canonical products, not products of allocation maxima. */
    PolyUOp *input_product = exact_shape_product(ctx, SRC_DIMS(0), SRC_DIM_UOPS(0), in_ndim);
    PolyUOp *output_product = exact_shape_product(ctx, dims, dim_uops, n);
    if (!input_product || !output_product || input_product != output_product)
      return make_entry_none(ctx);
    return make_entry_dims_uops(ctx, dims, dim_uops, n);
  }

  /* EXPAND: pinned movement form reads src[1].as_shape. The max-shape stores
   * allocation bounds; dim_uops preserves the public symbolic shape. */
  if (op == POLY_OP_EXPAND && u->arg.kind == POLY_ARG_NONE && u->n_src == 2) {
    int8_t in_ndim = SRC_NDIM(0);
    if (in_ndim < 0) return make_entry_none(ctx);
    int64_t dims[POLY_MAX_DIMS];
    PolyUOp *dim_uops[POLY_MAX_DIMS];
    int n = 0;
    if (!shape_arg_values(ctx, u->src[1], dims, dim_uops, &n) || n != in_ndim)
      return make_entry_none(ctx);
    PolyUOp *const *in_dim_uops = SRC_DIM_UOPS(0);
    for (int i = 0; i < n; i++) {
      PolyUOp *out_uop = dim_uops[i];
      if (!expand_dim_compatible(
              ctx, SRC_DIMS(0)[i], in_dim_uops ? in_dim_uops[i] : NULL, dims[i],
              out_uop
          ))
        return make_entry_none(ctx);
    }
    return make_entry_dims_uops(ctx, dims, dim_uops, n);
  }

  /* PERMUTE: reorder src[0] shape */
  if (op == POLY_OP_PERMUTE && u->n_src >= 1 && u->arg.kind == POLY_ARG_INT_TUPLE) {
    int8_t in_ndim = SRC_NDIM(0);
    if (in_ndim <= 0) {
      return make_entry_none(ctx);
    }
    int n = u->arg.int_tuple.n;
    if (!rank_tuple_valid(u->arg.int_tuple.vals, n) || n > in_ndim) return make_entry_none(ctx);
    int64_t dims[POLY_MAX_DIMS];
    for (int i = 0; i < n; i++) {
      int64_t idx = u->arg.int_tuple.vals[i];
      if (idx < 0 || idx >= in_ndim) return make_entry_none(ctx);
      dims[i] = SRC_DIMS(0)[idx];
    }
    PolyUOp *dim_uops[POLY_MAX_DIMS];
    PolyUOp *const *in_dim_uops = SRC_DIM_UOPS(0);
    for (int i = 0; i < n; i++) {
      int64_t idx = u->arg.int_tuple.vals[i];
      dim_uops[i] = in_dim_uops ? in_dim_uops[idx] : NULL;
    }
    return make_entry_dims_uops(ctx, dims, dim_uops, n);
  }

  /* PAD: pinned tensor form stores offset and output-size shape values as
   * src[1]/src[2] (uop/ops.py:710-721). */
  if (op == POLY_OP_PAD && u->arg.kind == POLY_ARG_NONE && u->n_src == 3 &&
      !u->src[0]->dtype.is_ptr) {
    int8_t in_ndim = SRC_NDIM(0);
    if (in_ndim < 0) return make_entry_none(ctx);
    int64_t dims[POLY_MAX_DIMS];
    PolyUOp *dim_uops[POLY_MAX_DIMS];
    int n = 0;
    if (!shape_arg_values(ctx, u->src[2], dims, dim_uops, &n) || n != in_ndim)
      return make_entry_none(ctx);
    return make_entry_dims_uops(ctx, dims, dim_uops, n);
  }

  /* PAD: legacy pair-tuple form retained for imported/raw old graphs. */
  if (op == POLY_OP_PAD && u->n_src >= 1 && u->arg.kind == POLY_ARG_PAIR_TUPLE) {
    int8_t in_ndim = SRC_NDIM(0);
    if (in_ndim <= 0) {
      return make_entry_none(ctx);
    }
    if (!rank_tuple_valid(u->arg.pair_tuple.pairs, u->arg.pair_tuple.n) ||
        u->arg.pair_tuple.n != in_ndim)
      return make_entry_none(ctx);
    int64_t dims[POLY_MAX_DIMS];
    for (int i = 0; i < in_ndim; i++)
      dims[i] = SRC_DIMS(0)[i] + u->arg.pair_tuple.pairs[i][0] + u->arg.pair_tuple.pairs[i][1];
    return make_entry_dims(ctx, dims, in_ndim);
  }

  /* SHRINK: tinygrad-form movement shrink.
   * src = (tensor, STACK(offsets), STACK(sizes)). Late codegen memory shrink
   * uses a pointer src[0] and remains a no-shape address node. */
  if (op == POLY_OP_SHRINK && u->arg.kind == POLY_ARG_NONE && u->n_src >= 3 &&
      !u->src[0]->dtype.is_ptr) {
    int8_t in_ndim = SRC_NDIM(0);
    if (in_ndim < 0) return make_entry_none(ctx);
    int64_t dims[POLY_MAX_DIMS];
    PolyUOp *dim_uops[POLY_MAX_DIMS];
    int n = 0;
    if (!shape_arg_values(ctx, u->src[2], dims, dim_uops, &n) || n != in_ndim)
      return make_entry_none(ctx);
    return make_entry_dims_uops(ctx, dims, dim_uops, n);
  }

  if (op == POLY_OP_SHRINK && u->arg.kind == POLY_ARG_NONE && u->n_src >= 1 &&
      u->src[0]->dtype.is_ptr)
    return make_entry_none(ctx);

  /* SHRINK: legacy pair-tuple movement shrink, output = end - begin per axis */
  if (op == POLY_OP_SHRINK && u->n_src >= 1 && u->arg.kind == POLY_ARG_PAIR_TUPLE) {
    int8_t in_ndim = SRC_NDIM(0);
    if (in_ndim <= 0) {
      return make_entry_none(ctx);
    }
    if (!rank_tuple_valid(u->arg.pair_tuple.pairs, u->arg.pair_tuple.n) ||
        u->arg.pair_tuple.n != in_ndim)
      return make_entry_none(ctx);
    int64_t dims[POLY_MAX_DIMS];
    for (int i = 0; i < in_ndim; i++)
      dims[i] = u->arg.pair_tuple.pairs[i][1] - u->arg.pair_tuple.pairs[i][0];
    return make_entry_dims(ctx, dims, in_ndim);
  }

  /* FLIP: same shape as src[0] */
  if (op == POLY_OP_FLIP) {
    if (u->arg.kind == POLY_ARG_INT_TUPLE &&
        !rank_tuple_valid(u->arg.int_tuple.vals, u->arg.int_tuple.n))
      return make_entry_none(ctx);
    if (u->n_src >= 1 && SRC_NDIM(0) >= 0)
      return make_entry_dims_uops(ctx, SRC_DIMS(0), SRC_DIM_UOPS(0), SRC_NDIM(0));
    return make_entry_none(ctx);
  }

  /* Tensor REDUCE: dims at reduction axes become 1. Accept legacy
   * REDUCE_AXIS raw/import graphs while their migration is incomplete. */
  if ((op == POLY_OP_REDUCE || op == POLY_OP_REDUCE_AXIS) && u->n_src >= 1 &&
      u->arg.kind == POLY_ARG_REDUCE_AXIS) {
    int8_t in_ndim = SRC_NDIM(0);
    if (in_ndim <= 0) {
      return make_entry_none(ctx);
    }
    if (!rank_tuple_valid(u->arg.reduce_axis.axes, u->arg.reduce_axis.n))
      return make_entry_none(ctx);
    int64_t dims[POLY_MAX_DIMS];
    PolyUOp *dim_uops[POLY_MAX_DIMS];
    PolyUOp *const *in_dim_uops = SRC_DIM_UOPS(0);
    memcpy(dims, SRC_DIMS(0), in_ndim * sizeof(int64_t));
    for (int i = 0; i < in_ndim; i++)
      dim_uops[i] = in_dim_uops ? in_dim_uops[i] : NULL;
    for (int i = 0; i < u->arg.reduce_axis.n; i++) {
      int ax = (int)u->arg.reduce_axis.axes[i];
      if (ax >= 0 && ax < in_ndim) {
        dims[ax] = 1;
        dim_uops[ax] = NULL;
      }
    }
    return make_entry_dims_uops(ctx, dims, dim_uops, in_ndim);
  }

  /* ASSIGN: use stored logical shape (arg) if present */
  if (op == POLY_OP_ASSIGN) {
    if (u->arg.kind == POLY_ARG_INT_TUPLE && u->arg.int_tuple.n > 0) {
      if (!rank_tuple_valid(u->arg.int_tuple.vals, u->arg.int_tuple.n)) return make_entry_none(ctx);
      return make_entry_dims(ctx, u->arg.int_tuple.vals, u->arg.int_tuple.n);
    }
    if (u->n_src >= 1 && SRC_NDIM(0) >= 0)
      return make_entry_dims_uops(ctx, SRC_DIMS(0), SRC_DIM_UOPS(0), SRC_NDIM(0));
    return make_entry_none(ctx);
  }

  /* Passthrough ops: inherit src[0] shape */
  if (op == POLY_OP_CONTIGUOUS || op == POLY_OP_DETACH || op == POLY_OP_CONTIGUOUS_BACKWARD ||
      op == POLY_OP_COPY || op == POLY_OP_NOOP || op == POLY_OP_REDUCE || op == POLY_OP_AFTER ||
      op == POLY_OP_END || op == POLY_OP_GROUP) {
    if (u->n_src >= 1 && SRC_NDIM(0) >= 0)
      return make_entry_dims_uops(ctx, SRC_DIMS(0), SRC_DIM_UOPS(0), SRC_NDIM(0));
    return make_entry_none(ctx);
  }

  /* BITCAST: scale last dim if itemsize differs */
  if (op == POLY_OP_BITCAST && u->n_src >= 1) {
    int8_t in_ndim = SRC_NDIM(0);
    if (in_ndim < 0) {
      return make_entry_none(ctx);
    }
    if (in_ndim == 0) {
      return make_entry_scalar(ctx);
    }
    int out_sz = poly_dtype_itemsize(poly_dtype_scalar(u->dtype));
    int in_sz = poly_dtype_itemsize(poly_dtype_scalar(u->src[0]->dtype));
    if (out_sz != in_sz && in_sz > 0 && out_sz > 0) {
      int64_t dims[POLY_MAX_DIMS];
      PolyUOp *dim_uops[POLY_MAX_DIMS];
      PolyUOp *const *in_dim_uops = SRC_DIM_UOPS(0);
      memcpy(dims, SRC_DIMS(0), in_ndim * sizeof(int64_t));
      for (int i = 0; i < in_ndim; i++)
        dim_uops[i] = in_dim_uops ? in_dim_uops[i] : NULL;
      dims[in_ndim - 1] = (SRC_DIMS(0)[in_ndim - 1] * in_sz) / out_sz;
      dim_uops[in_ndim - 1] = NULL;
      return make_entry_dims_uops(ctx, dims, dim_uops, in_ndim);
    } else {
      if (u->n_src >= 1 && SRC_NDIM(0) >= 0)
        return make_entry_dims_uops(ctx, SRC_DIMS(0), SRC_DIM_UOPS(0), SRC_NDIM(0));
      return make_entry_none(ctx);
    }
  }

  /* CAST: ptr→non-ptr returns no shape; else same as ALU */
  if (op == POLY_OP_CAST && u->n_src >= 1) {
    if (u->src[0]->dtype.is_ptr && !u->dtype.is_ptr) {
      return make_entry_none(ctx);
    }
    /* Fall through to ALU handling */
  }

  /* ALU + CAST: broadcast shapes across all sources */
  /* Full NumPy broadcasting: align trailing dims, max(a,b) per axis,
   * a==1 or b==1 for expansion. Ported from old compute_shape(). */
  if (poly_opset_has(POLY_GROUP_ALU, op) || op == POLY_OP_CAST) {
    int64_t out_dims[POLY_MAX_DIMS];
    PolyUOp *out_dim_uops[POLY_MAX_DIMS];
    int out_ndim = -1;
    for (int i = 0; i < u->n_src; i++) {
      int8_t si_ndim = SRC_NDIM(i);
      if (si_ndim < 0) continue;
      if (si_ndim > POLY_MAX_DIMS) return make_entry_none(ctx);
      const int64_t *si_dims = SRC_DIMS(i);
      PolyUOp *const *si_dim_uops = SRC_DIM_UOPS(i);
      if (out_ndim < 0) {
        out_ndim = si_ndim;
        if (si_ndim > 0) memcpy(out_dims, si_dims, si_ndim * sizeof(int64_t));
        for (int ax = 0; ax < si_ndim; ax++)
          out_dim_uops[ax] = si_dim_uops ? si_dim_uops[ax] : NULL;
        continue;
      }
      int ndim = (out_ndim > si_ndim) ? out_ndim : si_ndim;
      if (ndim > POLY_MAX_DIMS) return make_entry_none(ctx);
      int64_t merged[POLY_MAX_DIMS];
      PolyUOp *merged_dim_uops[POLY_MAX_DIMS];
      for (int ax = 0; ax < ndim; ax++) {
        int ai = out_ndim - 1 - ax;
        int bi = si_ndim - 1 - ax;
        int64_t a = (ai >= 0) ? out_dims[ai] : 1;
        int64_t b = (bi >= 0) ? si_dims[bi] : 1;
        PolyUOp *au = (ai >= 0) ? out_dim_uops[ai] : NULL;
        PolyUOp *bu = (bi >= 0 && si_dim_uops) ? si_dim_uops[bi] : NULL;
        if (a != b && a != 1 && b != 1) {
          return make_entry_none(ctx);
        }
        if (a == 1) {
          merged[ndim - 1 - ax] = b;
          merged_dim_uops[ndim - 1 - ax] = bu;
        } else if (b == 1) {
          merged[ndim - 1 - ax] = a;
          merged_dim_uops[ndim - 1 - ax] = au;
        } else {
          bool au_static = shape_dim_is_static(au);
          bool bu_static = shape_dim_is_static(bu);
          if (!au_static || !bu_static) {
            if (au && bu && au != bu) return make_entry_none(ctx);
            if ((au && !bu) || (!au && bu)) return make_entry_none(ctx);
          }
          merged[ndim - 1 - ax] = a;
          merged_dim_uops[ndim - 1 - ax] = (au_static && bu_static) ? NULL : (au ? au : bu);
        }
      }
      out_ndim = ndim;
      memcpy(out_dims, merged, ndim * sizeof(int64_t));
      for (int ax = 0; ax < ndim; ax++)
        out_dim_uops[ax] = merged_dim_uops[ax];
    }
    if (out_ndim < 0) {
      return make_entry_none(ctx);
    }
    return make_entry_dims_uops(ctx, out_dims, out_dim_uops, out_ndim);
  }

  /* Default: no shape */
  return make_entry_none(ctx);
}
