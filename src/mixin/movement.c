/*
 * mixin/movement.c -- MovementMixin graph composition
 *
 * Mirrors tinygrad/mixin/movement.py. Raw movement UOps keep their current
 * Tinygrad source arities; this layer owns broadcast, reshape, pad, shrink,
 * flip, and permute composition over those UOps.
 */

#include "polygrad.h"
#include "uop/upat.h"

static bool rank_tuple_valid(const void *data, int n) {
  return n >= 0 && n <= POLY_MAX_DIMS && (n == 0 || data != NULL);
}

static bool shape_value_is_const(PolyUOp *u, int64_t expected) {
  int64_t value = 0;
  return u && u->op == POLY_OP_CONST && poly_uop_const_i64(u, &value) == 0 && value == expected;
}

static bool shape_value_equal(PolyUOp *a, PolyUOp *b) {
  if (a == b) return true;
  int64_t av = 0, bv = 0;
  return poly_uop_const_i64(a, &av) == 0 && poly_uop_const_i64(b, &bv) == 0 && av == bv;
}

/* UOp._mop simplifies shape operands, never the tensor source. Constant-only
 * tuples are already canonical; keep static construction off the rewriter. */
static PolyUOp *movement_shape_arg(PolyCtx *ctx, PolyUOp **dims, int ndim) {
  PolyUOp *shape = poly_shape_to_shape_arg(ctx, dims, ndim);
  if (!shape) return NULL;
  for (int i = 0; i < ndim; i++)
    if (dims[i]->op != POLY_OP_CONST) return poly_graph_rewrite(ctx, shape, poly_symbolic());
  return shape;
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

static PolyUOp *static_shape_arg(PolyCtx *ctx, const int64_t *dims, int ndim) {
  if (!ctx || !rank_tuple_valid(dims, ndim)) return NULL;
  PolyUOp *vals[POLY_MAX_DIMS];
  for (int i = 0; i < ndim; i++)
    vals[i] = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(dims[i]));
  return poly_shape_to_shape_arg(ctx, vals, ndim);
}

PolyUOp *poly_reshape(PolyCtx *ctx, PolyUOp *src, int64_t *dims, int ndim) {
  if (!ctx || !src || !rank_tuple_valid(dims, ndim)) return NULL;
  if (poly_uop_ndim(ctx, src) == ndim) {
    bool same = true;
    for (int i = 0; i < ndim; i++)
      if (!shape_value_is_const(poly_uop_shape_dim(ctx, src, i), dims[i])) same = false;
    if (same) return src;
  }
  PolyUOp *shape = static_shape_arg(ctx, dims, ndim);
  if (!shape) return NULL;
  PolyUOp *srcs[2] = {src, shape};
  return poly_uop(ctx, POLY_OP_RESHAPE, src->dtype, srcs, 2, poly_arg_none());
}

PolyUOp *poly_reshape_uop(PolyCtx *ctx, PolyUOp *src, PolyUOp **dims, int ndim) {
  if (!ctx || !src || !rank_tuple_valid(dims, ndim)) return NULL;
  PolyUOp *shape = movement_shape_arg(ctx, dims, ndim);
  if (!shape) return NULL;
  dims = ndim == 1 ? &shape : shape->src;
  if (poly_uop_ndim(ctx, src) == ndim) {
    bool same = true;
    for (int i = 0; i < ndim; i++)
      if (!shape_value_equal(poly_uop_shape_dim(ctx, src, i), dims[i])) same = false;
    if (same) return src;
  }
  PolyUOp *srcs[2] = {src, shape};
  return poly_uop(ctx, POLY_OP_RESHAPE, src->dtype, srcs, 2, poly_arg_none());
}

/* Current UOp._mop(Ops.EXPAND) stores only dimensions prepended to the
 * existing source shape (uop/ops.py:419-426,787-806). This is the raw
 * movement constructor used by MovementMixin._broadcast_to after it has
 * squeezed the source axes that actually expand. */
static PolyUOp *mop_expand(PolyCtx *ctx, PolyUOp *src, PolyUOp **arg, int n_arg) {
  if (!ctx || !src || !rank_tuple_valid(arg, n_arg)) return NULL;
  if (n_arg == 0) return src;
  PolyUOp *shape = movement_shape_arg(ctx, arg, n_arg);
  if (!shape) return NULL;
  PolyUOp *srcs[2] = {src, shape};
  return poly_uop(ctx, POLY_OP_EXPAND, src->dtype, srcs, 2, poly_arg_none());
}

/* Exact C port of current MovementMixin._broadcast_to
 * (tinygrad/mixin/movement.py:129-147). Public expand, elementwise
 * broadcasting, const_like, and both language frontends all cross this one
 * core construction rule. */
static PolyUOp *broadcast_to(PolyCtx *ctx, PolyUOp *src, PolyUOp **new_shape, int new_ndim) {
  if (!ctx || !src || !rank_tuple_valid(new_shape, new_ndim)) return NULL;
  int source_ndim = poly_uop_ndim(ctx, src);
  if (source_ndim < 0) {
    if (poly_dtype_eq(src->dtype, POLY_VOID)) return NULL;
    source_ndim = 0;
  }
  if (source_ndim > new_ndim) return NULL;

  bool same = source_ndim == new_ndim;
  for (int i = 0; i < new_ndim; i++) {
    int64_t vmin = 0, vmax = 0;
    if (!new_shape[i]) return NULL;
    poly_uop_minmax(ctx, new_shape[i], &vmin, &vmax);
    if (vmin < 0 || vmax < 0) return NULL;
    if (same && !shape_value_equal(poly_uop_shape_dim(ctx, src, i), new_shape[i])) same = false;
  }
  if (same) return src;

  int n_left = new_ndim - source_ndim;
  int expand_at[POLY_MAX_DIMS], kept[POLY_MAX_DIMS];
  int n_expand = 0, n_kept = 0;
  for (int i = 0; i < source_ndim; i++) {
    PolyUOp *source_dim = poly_uop_shape_dim(ctx, src, i);
    PolyUOp *new_dim = new_shape[n_left + i];
    if (!source_dim || !new_dim) return NULL;
    if (shape_value_equal(source_dim, new_dim)) {
      kept[n_kept++] = i;
    } else if (shape_value_is_const(source_dim, 1)) {
      expand_at[n_expand++] = i;
    } else {
      return NULL;
    }
  }

  PolyUOp *squeezed = src;
  if (n_expand > 0) {
    PolyUOp *kept_dims[POLY_MAX_DIMS];
    for (int i = 0; i < n_kept; i++) {
      kept_dims[i] = poly_uop_shape_dim(ctx, src, kept[i]);
      if (!kept_dims[i]) return NULL;
    }
    squeezed = poly_reshape_uop(ctx, src, kept_dims, n_kept);
    if (!squeezed) return NULL;
  }

  PolyUOp *expand_shape[POLY_MAX_DIMS];
  int n_expand_shape = 0;
  for (int i = 0; i < n_left; i++)
    expand_shape[n_expand_shape++] = new_shape[i];
  for (int i = 0; i < n_expand; i++)
    expand_shape[n_expand_shape++] = new_shape[n_left + expand_at[i]];
  PolyUOp *expanded = mop_expand(ctx, squeezed, expand_shape, n_expand_shape);
  if (!expanded) return NULL;

  int64_t perm[POLY_MAX_DIMS];
  bool identity = true;
  for (int i = 0; i < n_left; i++)
    perm[i] = i;
  for (int i = 0; i < source_ndim; i++) {
    int position = -1;
    for (int j = 0; j < n_expand; j++)
      if (expand_at[j] == i) {
        position = n_left + j;
        break;
      }
    if (position < 0) {
      for (int j = 0; j < n_kept; j++)
        if (kept[j] == i) {
          position = n_left + n_expand + j;
          break;
        }
    }
    if (position < 0) return NULL;
    perm[n_left + i] = position;
  }
  for (int i = 0; i < new_ndim; i++)
    if (perm[i] != i) {
      identity = false;
      break;
    }
  return identity ? expanded : poly_permute(ctx, expanded, perm, new_ndim);
}

PolyUOp *poly_expand(PolyCtx *ctx, PolyUOp *src, int64_t *dims, int ndim) {
  if (!ctx || !src || !rank_tuple_valid(dims, ndim)) return NULL;
  PolyUOp *new_shape[POLY_MAX_DIMS];
  for (int i = 0; i < ndim; i++)
    new_shape[i] = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(dims[i]));
  return broadcast_to(ctx, src, new_shape, ndim);
}

PolyUOp *poly_permute(PolyCtx *ctx, PolyUOp *src, int64_t *perm, int ndim) {
  if (!ctx || !src || !rank_tuple_valid(perm, ndim) || poly_uop_ndim(ctx, src) != ndim) return NULL;
  /* MovementMixin.permute resolves/validates axes before _mop and preserves
   * source identity for a no-op, including callers without a Tensor frontend. */
  int64_t order[POLY_MAX_DIMS];
  bool seen[POLY_MAX_DIMS] = {0}, identity = true;
  for (int i = 0; i < ndim; i++) {
    int64_t axis = perm[i];
    if (axis < -ndim || axis >= ndim) return NULL;
    if (axis < 0) axis += ndim;
    if (seen[axis]) return NULL;
    seen[axis] = true;
    order[i] = axis;
    if (axis != i) identity = false;
  }
  if (identity) return src;
  PolyArg arg;
  arg.kind = POLY_ARG_INT_TUPLE;
  arg.int_tuple.vals = order;
  arg.int_tuple.n = ndim;
  return poly_uop1(ctx, POLY_OP_PERMUTE, src->dtype, src, arg);
}

PolyUOp *poly_stack(PolyCtx *ctx, PolyUOp **src, int n_src, int dim) {
  if (!ctx || !src || n_src <= 0) return NULL;
  int ndim = poly_uop_ndim(ctx, src[0]);
  if (ndim < 0 || ndim >= POLY_MAX_DIMS) return NULL;
  if (dim < 0) dim += ndim + 1;
  if (dim < 0 || dim > ndim) return NULL;

  /* Current MovementMixin.stack requires identical shapes before building
   * one UOp STACK and moves its leading axis into the requested position
   * (mixin/movement.py:257-273). */
  for (int i = 1; i < n_src; i++) {
    if (!src[i] || poly_uop_ndim(ctx, src[i]) != ndim) return NULL;
    for (int axis = 0; axis < ndim; axis++)
      if (!shape_value_equal(
              poly_uop_shape_dim(ctx, src[0], axis), poly_uop_shape_dim(ctx, src[i], axis)
          ))
        return NULL;
  }

  PolyUOp *stacked = poly_uop_stack(ctx, src, n_src);
  if (!stacked || dim == 0) return stacked;
  int64_t perm[POLY_MAX_DIMS];
  int out_ndim = ndim + 1;
  int pos = 0;
  for (int axis = 1; axis <= dim; axis++)
    perm[pos++] = axis;
  perm[pos++] = 0;
  for (int axis = dim + 1; axis < out_ndim; axis++)
    perm[pos++] = axis;
  return poly_permute(ctx, stacked, perm, out_ndim);
}

PolyUOp *poly_expand_uop(PolyCtx *ctx, PolyUOp *src, PolyUOp **dims, int ndim) {
  return broadcast_to(ctx, src, dims, ndim);
}

PolyUOp *poly_shrink_uop(PolyCtx *ctx, PolyUOp *src, PolyUOp **starts, PolyUOp **sizes, int ndim) {
  if (!ctx || !src || !rank_tuple_valid(starts, ndim) || !rank_tuple_valid(sizes, ndim) ||
      poly_uop_ndim(ctx, src) != ndim)
    return NULL;
  PolyUOp *start_stack = movement_shape_arg(ctx, starts, ndim);
  PolyUOp *size_stack = movement_shape_arg(ctx, sizes, ndim);
  if (!start_stack || !size_stack) return NULL;
  starts = ndim == 1 ? &start_stack : start_stack->src;
  sizes = ndim == 1 ? &size_stack : size_stack->src;
  /* MovementMixin.shrink returns self when the resulting shape is unchanged
   * (mixin/movement.py:192-193). Elide only on proved shape-value identity. */
  if (movement_shape_is_identity(ctx, src, starts, sizes, ndim)) return src;
  PolyUOp *srcs[3] = {src, start_stack, size_stack};
  return poly_uop(ctx, POLY_OP_SHRINK, src->dtype, srcs, 3, poly_arg_none());
}

PolyUOp *poly_pad_uop(PolyCtx *ctx, PolyUOp *src, PolyUOp **offsets, PolyUOp **sizes, int ndim) {
  if (!ctx || !src || !rank_tuple_valid(offsets, ndim) || !rank_tuple_valid(sizes, ndim) ||
      poly_uop_ndim(ctx, src) != ndim)
    return NULL;
  PolyUOp *offset_stack = movement_shape_arg(ctx, offsets, ndim);
  PolyUOp *size_stack = movement_shape_arg(ctx, sizes, ndim);
  if (!offset_stack || !size_stack) return NULL;
  offsets = ndim == 1 ? &offset_stack : offset_stack->src;
  sizes = ndim == 1 ? &size_stack : size_stack->src;
  /* MovementMixin.pad has the same shape-identity return rule
   * (mixin/movement.py:170-171). */
  if (movement_shape_is_identity(ctx, src, offsets, sizes, ndim)) return src;
  PolyUOp *srcs[3] = {src, offset_stack, size_stack};
  return poly_uop(ctx, POLY_OP_PAD, src->dtype, srcs, 3, poly_arg_none());
}

PolyUOp *poly_shrink(PolyCtx *ctx, PolyUOp *src, int64_t (*pairs)[2], int ndim) {
  if (!ctx || !src || !rank_tuple_valid(pairs, ndim) || poly_uop_ndim(ctx, src) != ndim)
    return NULL;
  /* Pinned UOp._mop returns scalar PAD/SHRINK unchanged when the movement
   * argument is empty, after validating rank. */
  if (ndim == 0) return src;
  PolyUOp *starts[POLY_MAX_DIMS];
  PolyUOp *sizes[POLY_MAX_DIMS];
  for (int i = 0; i < ndim; i++) {
    int64_t size;
    if (pairs[i][0] < 0 || pairs[i][1] < pairs[i][0] ||
        __builtin_sub_overflow(pairs[i][1], pairs[i][0], &size))
      return NULL;
    starts[i] = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(pairs[i][0]));
    sizes[i] = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(size));
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
  PolyShape shape = poly_uop_max_shape_cached(ctx, src);
  if (shape.ndim != ndim) return NULL;
  if (ndim == 0) return src;
  PolyUOp *offsets[POLY_MAX_DIMS];
  PolyUOp *sizes[POLY_MAX_DIMS];
  for (int i = 0; i < ndim; i++) {
    if (pairs[i][0] < 0 || pairs[i][1] < 0) return NULL;
    offsets[i] = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(pairs[i][0]));
    PolyUOp *dim = poly_uop_shape_dim(ctx, src, i);
    if (!dim) dim = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(shape.dims[i]));
    int64_t delta;
    if (__builtin_add_overflow(pairs[i][0], pairs[i][1], &delta)) return NULL;
    if (delta == 0) {
      sizes[i] = dim;
    } else {
      PolyUOp *amount = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(delta));
      sizes[i] = poly_uop2(ctx, POLY_OP_ADD, POLY_WEAKINT, dim, amount, poly_arg_none());
      sizes[i] = poly_graph_rewrite(ctx, sizes[i], poly_symbolic_simple());
    }
  }
  return poly_pad_uop(ctx, src, offsets, sizes, ndim);
}
