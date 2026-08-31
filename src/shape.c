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
#include "uop/upat.h"
#include "utils.h"

/* Local shape-inference helpers. */

static bool rank_tuple_valid(const void *data, int n) {
  return n >= 0 && n <= POLY_MAX_DIMS && (n == 0 || data != NULL);
}

static bool axis_expr_equal(PolyUOp *a, PolyUOp *b);

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

PolyMap *poly_ctx_shape_cache(PolyCtx *ctx); /* defined in uop/ops.c */

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

static bool resolved_shape_dim(PolyCtx *ctx, const PolyUOp *u, int axis, int64_t *out) {
  if (!ctx || !u || axis < 0) return false;
  PolyUOp *dim = poly_uop_shape_dim(ctx, u, axis);
  if (dim) return poly_uop_bind_value(dim, out) == 0;
  const int64_t *dims = poly_uop_max_shape_dims(ctx, u);
  if (!dims || axis >= poly_uop_ndim(ctx, u)) return false;
  if (out) *out = dims[axis];
  return true;
}

int poly_broadcast_axes(
    PolyCtx *ctx,
    const PolyUOp *src,
    const PolyUOp *out,
    int *axes,
    int max_axes
) {
  if (!ctx || !src || !out || max_axes < 0) return -1;
  int src_ndim = poly_uop_ndim(ctx, src);
  int out_ndim = poly_uop_ndim(ctx, out);
  if (src_ndim < 0 || out_ndim < src_ndim || out_ndim > POLY_MAX_DIMS) return -1;

  int nleft = out_ndim - src_ndim;
  int n_axes = 0;
  for (int axis = 0; axis < nleft; axis++) {
    if (axes && n_axes < max_axes) axes[n_axes] = axis;
    n_axes++;
  }
  for (int axis = 0; axis < src_ndim; axis++) {
    int64_t src_dim = 0, out_dim = 0;
    /* tinygrad resolve(..., default=False): only dimensions proven to be 1
     * and expanded to a proven non-1 dimension are broadcast axes. */
    if (!resolved_shape_dim(ctx, src, axis, &src_dim) || src_dim != 1 ||
        !resolved_shape_dim(ctx, out, nleft + axis, &out_dim) || out_dim == 1)
      continue;
    if (axes && n_axes < max_axes) axes[n_axes] = nleft + axis;
    n_axes++;
  }
  return n_axes > max_axes ? -1 : n_axes;
}

static PolyUOp *axis_shape_arg_item(PolyUOp *shape, int axis) {
  if (!shape || axis < 0) return NULL;
  if (shape->op == POLY_OP_STACK) return axis < shape->n_src ? shape->src[axis] : NULL;
  return axis == 0 ? shape : NULL;
}

static bool axis_expr_equal(PolyUOp *a, PolyUOp *b) {
  if (a == b) return true;
  int64_t av = 0, bv = 0;
  return poly_uop_const_i64(a, &av) == 0 && poly_uop_const_i64(b, &bv) == 0 && av == bv;
}

static PolyUOp *axis_shape_product(PolyCtx *ctx, const PolyUOp *u, int end) {
  if (!ctx || !u || end < 0 || end > poly_uop_ndim(ctx, u)) return NULL;
  PolyUOp *product = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(1));
  const int64_t *max_shape = poly_uop_max_shape_dims(ctx, u);
  for (int i = 0; i < end; i++) {
    PolyUOp *dim = poly_uop_shape_dim(ctx, u, i);
    if (!dim && max_shape)
      dim = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(max_shape[i]));
    if (!dim) return NULL;
    product = poly_binop(ctx, POLY_OP_MUL, product, dim);
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
  } else if (u->op == POLY_OP_UNSHARD && u->arg.kind == POLY_ARG_INT_TUPLE &&
             u->arg.int_tuple.n == 1 && u->arg.int_tuple.vals[0] >= 0 &&
             u->arg.int_tuple.vals[0] <= INT_MAX) {
    axis = (int)u->arg.int_tuple.vals[0];
    has_axis = true;
  } else if (u->op == POLY_OP_GETTUPLE && u->n_src == 1 && u->arg.kind == POLY_ARG_INT && u->arg.i >= 0) {
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
      }
      if (!full) has_axis = false;
    } else if (has_axis && u->op == POLY_OP_REDUCE && u->arg.kind == POLY_ARG_REDUCE) {
      if (axis < u->arg.reduce.num_axes)
        has_axis = false;
      else
        axis -= u->arg.reduce.num_axes;
    } else if (has_axis && u->op == POLY_OP_RESHAPE) {
      PolyUOp *source_prefix = axis_shape_product(ctx, u->src[0], axis);
      PolyUOp *output_prefix = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(1));
      int output_ndim = poly_uop_ndim(ctx, u);
      int new_axis = axis_expr_equal(output_prefix, source_prefix) ? 0 : -1;
      for (int i = 0; i < output_ndim; i++) {
        PolyUOp *dim = poly_uop_shape_dim(ctx, u, i);
        const int64_t *max_shape = poly_uop_max_shape_dims(ctx, u);
        if (!dim && max_shape)
          dim = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(max_shape[i]));
        output_prefix =
            dim ? poly_uop2(ctx, POLY_OP_MUL, POLY_WEAKINT, output_prefix, dim, poly_arg_none())
                : NULL;
        output_prefix =
            output_prefix ? poly_graph_rewrite(ctx, output_prefix, poly_symbolic_simple()) : NULL;
        if (!output_prefix) break;
        if (axis_expr_equal(output_prefix, source_prefix)) new_axis = i + 1;
      }
      PolyUOp *device = poly_uop_device_uop_cached(ctx, (PolyUOp *)u, NULL);
      int device_count =
          device && device->arg.kind == POLY_ARG_STRING_TUPLE ? device->arg.string_tuple.n : 0;
      if (!source_prefix || new_axis < 0 || new_axis >= output_ndim || device_count <= 0) {
        has_axis = false;
      } else {
        PolyUOp *new_dim = poly_uop_shape_dim(ctx, u, new_axis);
        const int64_t *max_shape = poly_uop_max_shape_dims(ctx, u);
        if (!new_dim && max_shape)
          new_dim = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(max_shape[new_axis]));
        PolyUOp *count = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(device_count));
        PolyUOp *rem =
            new_dim ? poly_uop2(ctx, POLY_OP_MOD, POLY_WEAKINT, new_dim, count, poly_arg_none())
                    : NULL;
        rem = rem ? poly_graph_rewrite(ctx, rem, poly_symbolic_simple()) : NULL;
        int64_t rem_value = -1;
        if (poly_uop_const_i64(rem, &rem_value) != 0 || rem_value != 0) {
          has_axis = false;
        } else {
          axis = new_axis;
        }
      }
    } else if (has_axis && u->op == POLY_OP_PERMUTE && u->arg.kind == POLY_ARG_INT_TUPLE) {
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
        cache, poly_ptr_hash(u), (void *)u, (void *)(intptr_t)(has_axis ? axis + 2 : 1), poly_ptr_eq
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
  if (poly_uop_is_variable(u)) return u;
  if (poly_uop_is_bound_var(u)) return u->src[0];
  return NULL;
}

int poly_uop_bind_value(PolyUOp *u, int64_t *out) {
  if (!u) return -1;
  if (u->op == POLY_OP_CONST) return poly_uop_const_i64(u, out);
  if (poly_uop_is_bound_var(u)) return poly_uop_const_i64(u->src[1]->src[1], out);
  if ((u->op == POLY_OP_ADD || u->op == POLY_OP_SUB || u->op == POLY_OP_MUL) && u->n_src >= 2) {
    int64_t a = 0, b = 0, r = 0;
    if (poly_uop_bind_value(u->src[0], &a) != 0 || poly_uop_bind_value(u->src[1], &b) != 0)
      return -1;
    bool ov = false;
    if (u->op == POLY_OP_ADD)
      ov = __builtin_add_overflow(a, b, &r);
    else if (u->op == POLY_OP_SUB)
      ov = __builtin_sub_overflow(a, b, &r);
    else
      ov = __builtin_mul_overflow(a, b, &r);
    if (ov) return -1;
    if (out) *out = r;
    return 0;
  }
  return -1;
}

/* Current UOp.as_shape (uop/ops.py:774-777): STACK exposes each source and
 * every other scalar integer UOp is one symbolic dimension. */
static bool shape_arg_values(
    PolyCtx *ctx,
    PolyUOp *shape_arg,
    int64_t *dims,
    PolyUOp **dim_uops,
    int *n_out
) {
  if (!shape_arg) return false;
  PolyUOp *items[POLY_MAX_DIMS];
  int n = poly_uop_as_shape(ctx, shape_arg, items, POLY_MAX_DIMS);
  if (n < 0) return false;

  for (int i = 0; i < n; i++) {
    PolyUOp *sz = items[i];
    int64_t v = 0;
    bool is_const = poly_uop_const_i64(sz, &v) == 0;
    if (!is_const && sz && sz->op == POLY_OP_CONST && sz->arg.kind == POLY_ARG_INT_TUPLE &&
        i < sz->arg.int_tuple.n) {
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

static PolyUOp *canonical_shape_dim(PolyCtx *ctx, int64_t max_dim, PolyUOp *dim_uop) {
  PolyUOp *dim =
      dim_uop ? dim_uop : poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(max_dim));
  /* Current UOp._shape keeps symbolic dimension dtypes and applies ssimplify;
   * it does not cast int dimensions to weakint (uop/ops.py:427-434). */
  return dim ? poly_graph_rewrite(ctx, dim, poly_symbolic()) : NULL;
}

static PolyUOp *exact_shape_product(
    PolyCtx *ctx,
    const int64_t *max_dims,
    PolyUOp *const *dim_uops,
    int ndim
) {
  if (!ctx || ndim < 0 || ndim > POLY_MAX_DIMS || (ndim > 0 && !max_dims)) return NULL;
  PolyUOp *product = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(1));
  for (int i = 0; i < ndim; i++) {
    PolyUOp *dim = canonical_shape_dim(ctx, max_dims[i], dim_uops ? dim_uops[i] : NULL);
    if (!dim) return NULL;
    product = poly_binop(ctx, POLY_OP_MUL, product, dim);
    if (!product) return NULL;
  }
  return poly_graph_rewrite(ctx, product, poly_symbolic());
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

int64_t poly_uop_max_numel(PolyCtx *ctx, const PolyUOp *u) {
  return ctx && u ? poly_shape_numel(poly_uop_max_shape_cached(ctx, u)) : -1;
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
  return poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(value));
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

/* Tinygrad 2026-08-22/a9069c177a9d uop/ops.py:70-77 `_broadcast_shape`.
 * `ranks` permits WMMA to broadcast each fragment prefix (`shape[:-1]`). */
static bool broadcast_shape_entries(
    PolyCtx *ctx,
    ShapeCacheEntry *const *entries,
    const int *ranks,
    int n_entries,
    int64_t out_dims[POLY_MAX_DIMS],
    PolyUOp *out_dim_uops[POLY_MAX_DIMS],
    int *out_ndim
) {
  if (!ctx || !entries || n_entries <= 0 || !out_dims || !out_dim_uops || !out_ndim) return false;
  int ndim = 0;
  for (int i = 0; i < n_entries; i++) {
    if (!entries[i]) return false;
    int rank = ranks ? ranks[i] : entries[i]->ndim;
    if (rank < 0 || rank > entries[i]->ndim || rank > POLY_MAX_DIMS) return false;
    if (rank > ndim) ndim = rank;
  }

  for (int axis = 0; axis < ndim; axis++) {
    bool selected = false;
    int64_t selected_max = 1;
    PolyUOp *selected_dim = NULL;
    for (int i = 0; i < n_entries; i++) {
      int rank = ranks ? ranks[i] : entries[i]->ndim;
      int source_axis = axis - (ndim - rank);
      int64_t dim_max = source_axis >= 0 ? entries[i]->dims[source_axis] : 1;
      PolyUOp *dim = source_axis >= 0 && entries[i]->dim_uops ? entries[i]->dim_uops[source_axis]
                                                              : shape_dim_const(ctx, 1);
      int64_t dim_value = 0;
      bool is_one = poly_uop_const_i64(dim, &dim_value) == 0 && dim_value == 1;
      if (is_one) continue;
      if (!selected) {
        selected = true;
        selected_max = dim_max;
        selected_dim = dim;
      } else if (selected_dim != dim) {
        int64_t selected_value = 0;
        bool both_same_const = poly_uop_const_i64(selected_dim, &selected_value) == 0 &&
                               poly_uop_const_i64(dim, &dim_value) == 0 &&
                               selected_value == dim_value;
        if (!both_same_const) return false;
      }
    }
    out_dims[axis] = selected ? selected_max : 1;
    out_dim_uops[axis] = selected ? selected_dim : shape_dim_const(ctx, 1);
  }
  *out_ndim = ndim;
  return true;
}

/* Pinned GETTUPLE(FUNCTION)._shape rewrites every symbolic dimension PARAM
 * with the FUNCTION's ordered argument at ParamArg.slot
 * (tinygrad/uop/ops.py:242-253,1691). Keep this query pass-local: it derives
 * a shape expression and does not add graph or lifecycle state. */
static PolyUOp *shape_resolve_function_dim(PolyCtx *ctx, PolyUOp *dim, PolyUOp *function) {
  if (!ctx || !dim || !function || function->op != POLY_OP_FUNCTION || function->n_src < 1)
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
    if (param->arg.kind != POLY_ARG_PARAM || !param->arg.param || param->arg.param->slot < 0 ||
        param->arg.param->slot >= function->n_src - 1) {
      valid = false;
      break;
    }
    from[n_sub] = param;
    to[n_sub] = function->src[1 + param->arg.param->slot];
    n_sub++;
  }

  PolyUOp *resolved = dim;
  if (valid && n_sub > 0 && poly_uop_substitute_many(ctx, &dim, 1, from, to, n_sub, &resolved) != 0)
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
  if (op == POLY_OP_GETTUPLE && u->n_src == 1 && u->arg.kind == POLY_ARG_INT && u->arg.i >= 0) {
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

  /* Tinygrad 2026-08-22/a9069c177a9d uop/ops.py:330-344 late ops have no
   * shape. Value CALL/INS and BINARY are handled below. */
  if (op == POLY_OP_SINK || op == POLY_OP_IF || op == POLY_OP_ENDIF || op == POLY_OP_BARRIER ||
      op == POLY_OP_LINEAR || op == POLY_OP_PROGRAM || op == POLY_OP_SOURCE ||
      op == POLY_OP_GROUP || op == POLY_OP_TUPLE || op == POLY_OP_FUNCTION ||
      op == POLY_OP_REWRITE_ERROR || op == POLY_OP_CUSTOM_FUNCTION || op == POLY_OP_UNIQUE) {
    return make_entry_none(ctx);
  }

  if (op == POLY_OP_CALL || op == POLY_OP_INS)
    return poly_dtype_eq(u->dtype, POLY_VOID) ? make_entry_none(ctx) : make_entry_scalar(ctx);

  if (op == POLY_OP_BINARY)
    return u->arg.kind == POLY_ARG_BYTES && u->arg.bytes.n >= 0 ? make_entry_1d(ctx, u->arg.bytes.n)
                                                                : make_entry_none(ctx);

  /* Scalar constants and symbolic index values. */
  if (op == POLY_OP_CONST || op == POLY_OP_RANGE || op == POLY_OP_SPECIAL ||
      poly_uop_is_variable(u) || poly_uop_is_bound_var(u)) {
    return make_entry_scalar(ctx);
  }

  /* Current UOp._shape gives STACK a new leading source-count axis followed
   * by the first source shape (uop/ops.py:367-369). This is the same rule for
   * tensor-value STACK and the scalar STACK used as a shape argument. */
  if (op == POLY_OP_STACK) {
    if (u->n_src == 0) return make_entry_scalar(ctx);
    if (SRC_NDIM(0) < 0 || SRC_NDIM(0) >= POLY_MAX_DIMS) return make_entry_none(ctx);
    int64_t dims[POLY_MAX_DIMS];
    PolyUOp *dim_uops[POLY_MAX_DIMS];
    dims[0] = u->n_src;
    dim_uops[0] = NULL;
    PolyUOp *const *source_dim_uops = SRC_DIM_UOPS(0);
    for (int i = 0; i < SRC_NDIM(0); i++) {
      dims[i + 1] = SRC_DIMS(0)[i];
      dim_uops[i + 1] = source_dim_uops ? source_dim_uops[i] : NULL;
    }
    return make_entry_dims_uops(ctx, dims, dim_uops, SRC_NDIM(0) + 1);
  }

  /* Current BUFFER shape is src[0].as_shape. Retain the approved logical
   * BUFFER forms below because they encode Polygrad's portable storage. */
  if (op == POLY_OP_BUFFER) {
    if (u->n_src == 1 && u->src[0] && poly_dtype_is_int(u->src[0]->dtype)) {
      int64_t dims[POLY_MAX_DIMS];
      PolyUOp *dim_uops[POLY_MAX_DIMS];
      int ndim = 0;
      if (!shape_arg_values(ctx, u->src[0], dims, dim_uops, &ndim)) return make_entry_none(ctx);
      return make_entry_dims_uops(ctx, dims, dim_uops, ndim);
    } else if (u->n_src == 0 && u->arg.kind == POLY_ARG_NONE) {
      return make_entry_scalar(ctx);
    } else if (u->arg.kind == POLY_ARG_INT) {
      /* Dynamic buffer: BUFFER(src=(UNIQUE, Variable/bound Variable, CONST...)).
       * dims stores max allocation shape; dim_uops stores the symbolic shape
       * expression, matching tinygrad's shape tuple carrying bound/PARAM UOps. */
      PolyUOp *dynamic_var = u->n_src >= 2 ? poly_uop_unbind_var(u->src[1]) : NULL;
      if (dynamic_var) {
        int ndim = u->n_src - 1;
        if (ndim <= 0 || ndim > POLY_MAX_DIMS) return make_entry_none(ctx);
        int64_t dims[POLY_MAX_DIMS];
        PolyUOp *dim_uops[POLY_MAX_DIMS];
        dims[0] = dynamic_var->arg.param->max_val;
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

  if (op == POLY_OP_CUSTOM || op == POLY_OP_CUSTOMI) {
    if (poly_dtype_eq(u->dtype, POLY_VOID)) return make_entry_none(ctx);
    ShapeCacheEntry **entries = u->n_src ? malloc((size_t)u->n_src * sizeof(*entries)) : NULL;
    if (u->n_src && !entries) return make_entry_none(ctx);
    int n_entries = 0;
    for (int i = 0; i < u->n_src; i++) {
      ShapeCacheEntry *entry = shape_cache_lookup(ctx, u->src[i]);
      if (entry && entry->ndim >= 0) entries[n_entries++] = entry;
    }
    int64_t dims[POLY_MAX_DIMS];
    PolyUOp *dim_uops[POLY_MAX_DIMS];
    int ndim = 0;
    bool ok = n_entries > 0 &&
              broadcast_shape_entries(ctx, entries, NULL, n_entries, dims, dim_uops, &ndim);
    free(entries);
    return ok ? make_entry_dims_uops(ctx, dims, dim_uops, ndim) : make_entry_none(ctx);
  }

  /* Tinygrad 2026-08-22/a9069c177a9d UOp._shape: PARAM shape is src[0]. */
  if (op == POLY_OP_PARAM) {
    if (u->n_src >= 1 && u->src[0] && u->src[0]->op == POLY_OP_STACK) {
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
      if (index_ndim < 0 || out_ndim > POLY_MAX_DIMS - index_ndim) return make_entry_none(ctx);
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

  /* Current tinygrad UOp._shape for UNSHARD multiplies every sharded local
   * axis by its explicit RANGE count (`tinygrad/uop/ops.py:417-443`). */
  if (op == POLY_OP_UNSHARD) {
    if (u->arg.kind != POLY_ARG_INT_TUPLE || u->arg.int_tuple.n <= 0 ||
        u->n_src != u->arg.int_tuple.n + 1 || SRC_NDIM(0) < 0)
      return make_entry_none(ctx);
    int64_t dims[POLY_MAX_DIMS];
    PolyUOp *dim_uops[POLY_MAX_DIMS];
    memcpy(dims, SRC_DIMS(0), (size_t)SRC_NDIM(0) * sizeof(*dims));
    PolyUOp *const *source_dim_uops = SRC_DIM_UOPS(0);
    for (int i = 0; i < SRC_NDIM(0); i++)
      dim_uops[i] = source_dim_uops ? source_dim_uops[i] : NULL;
    for (int i = 0; i < u->arg.int_tuple.n; i++) {
      int64_t axis_value = u->arg.int_tuple.vals[i];
      if (axis_value < 0 || axis_value >= SRC_NDIM(0)) return make_entry_none(ctx);
      int axis = (int)axis_value;
      int64_t vmin = 0, vmax = 0;
      poly_uop_minmax(ctx, u->src[i + 1], &vmin, &vmax);
      if (vmax < 0 || vmax == INT64_MAX) return make_entry_none(ctx);
      int64_t count = vmax + 1;
      if (__builtin_mul_overflow(dims[axis], count, &dims[axis])) return make_entry_none(ctx);
      if (count > 1) {
        if (shape_dim_is_static(dim_uops[axis])) {
          dim_uops[axis] = shape_dim_const(ctx, dims[axis]);
        } else {
          dim_uops[axis] = poly_alu2(ctx, POLY_OP_MUL, dim_uops[axis], shape_dim_const(ctx, count));
          if (!dim_uops[axis]) return make_entry_none(ctx);
        }
      }
    }
    return make_entry_dims_uops(ctx, dims, dim_uops, SRC_NDIM(0));
  }

  if (op == POLY_OP_RESHAPE && u->n_src == 2 && u->src[0] && u->src[0]->op == POLY_OP_NOOP) {
    int64_t dims[POLY_MAX_DIMS];
    PolyUOp *dim_uops[POLY_MAX_DIMS];
    int ndim = 0;
    return shape_arg_values(ctx, u->src[1], dims, dim_uops, &ndim)
               ? make_entry_dims_uops(ctx, dims, dim_uops, ndim)
               : make_entry_none(ctx);
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
    if (!input_product || !output_product) return make_entry_none(ctx);
    if (input_product != output_product) {
      PolyUOp *different = poly_binop(ctx, POLY_OP_CMPNE, input_product, output_product);
      if (!different || poly_uop_resolve(ctx, different, 1) != 0) return make_entry_none(ctx);
    }
    return make_entry_dims_uops(ctx, dims, dim_uops, n);
  }

  /* Current raw EXPAND prepends src[1].as_shape to the existing source shape
   * (uop/ops.py:419-426).  Broadcast compatibility is established by the
   * higher-level _broadcast_to construction before this node is emitted. */
  if (op == POLY_OP_EXPAND && u->arg.kind == POLY_ARG_NONE && u->n_src == 2) {
    int8_t in_ndim = SRC_NDIM(0);
    if (in_ndim < 0) return make_entry_none(ctx);
    int64_t prefix_dims[POLY_MAX_DIMS];
    PolyUOp *prefix_uops[POLY_MAX_DIMS];
    int n_prefix = 0;
    if (!shape_arg_values(ctx, u->src[1], prefix_dims, prefix_uops, &n_prefix) ||
        n_prefix > POLY_MAX_DIMS - in_ndim)
      return make_entry_none(ctx);
    int out_ndim = n_prefix + in_ndim;
    int64_t dims[POLY_MAX_DIMS];
    PolyUOp *dim_uops[POLY_MAX_DIMS];
    for (int i = 0; i < n_prefix; i++) {
      dims[i] = prefix_dims[i];
      dim_uops[i] = prefix_uops[i];
    }
    PolyUOp *const *in_dim_uops = SRC_DIM_UOPS(0);
    for (int i = 0; i < in_ndim; i++) {
      dims[n_prefix + i] = SRC_DIMS(0)[i];
      dim_uops[n_prefix + i] = in_dim_uops ? in_dim_uops[i] : NULL;
    }
    return make_entry_dims_uops(ctx, dims, dim_uops, out_ndim);
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
  if (op == POLY_OP_PAD && u->arg.kind == POLY_ARG_NONE && u->n_src == 3) {
    int8_t in_ndim = SRC_NDIM(0);
    if (in_ndim < 0) return make_entry_none(ctx);
    int64_t dims[POLY_MAX_DIMS];
    PolyUOp *dim_uops[POLY_MAX_DIMS];
    int n = 0;
    if (!shape_arg_values(ctx, u->src[2], dims, dim_uops, &n) || n != in_ndim)
      return make_entry_none(ctx);
    return make_entry_dims_uops(ctx, dims, dim_uops, n);
  }

  /* Tinygrad SHRINK shape is the size source for movement and memory slices. */
  if (op == POLY_OP_SHRINK && u->arg.kind == POLY_ARG_NONE && u->n_src >= 3) {
    int8_t in_ndim = SRC_NDIM(0);
    if (in_ndim < 0) return make_entry_none(ctx);
    int64_t dims[POLY_MAX_DIMS];
    PolyUOp *dim_uops[POLY_MAX_DIMS];
    int n = 0;
    if (!shape_arg_values(ctx, u->src[2], dims, dim_uops, &n) || n != in_ndim)
      return make_entry_none(ctx);
    return make_entry_dims_uops(ctx, dims, dim_uops, n);
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

  /* Current UOp._shape removes the prefix named by REDUCE.arg[1]
   * (tinygrad/uop/ops.py:444-449). */
  if (op == POLY_OP_REDUCE && u->n_src >= 1 && u->arg.kind == POLY_ARG_REDUCE) {
    int8_t in_ndim = SRC_NDIM(0);
    int num_axes = u->arg.reduce.num_axes;
    if (in_ndim < 0 || num_axes < 0 || num_axes > in_ndim) return make_entry_none(ctx);
    const int64_t *in_dims = SRC_DIMS(0);
    PolyUOp *const *in_dim_uops = SRC_DIM_UOPS(0);
    return make_entry_dims_uops(
        ctx, in_dims ? in_dims + num_axes : NULL, in_dim_uops ? in_dim_uops + num_axes : NULL,
        in_ndim - num_axes
    );
  }

  /* Current tinygrad UOp._shape passthrough ops inherit src[0], including
   * LOAD and STORE.  The address carries the vectorized memory shape; STORE's
   * value is broadcast to that shape before devectorization.  Late GROUP has
   * no shape and therefore is intentionally not a passthrough. */
  if (op == POLY_OP_CONTIGUOUS || op == POLY_OP_DETACH || op == POLY_OP_CONTIGUOUS_BACKWARD ||
      op == POLY_OP_COPY || op == POLY_OP_NOOP || op == POLY_OP_REDUCE || op == POLY_OP_AFTER ||
      op == POLY_OP_LOAD || op == POLY_OP_STORE || op == POLY_OP_END) {
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
    int out_sz = poly_dtype_itemsize(u->dtype);
    int in_sz = poly_dtype_itemsize(u->src[0]->dtype);
    if (out_sz != in_sz && in_sz > 0 && out_sz > 0) {
      int64_t dims[POLY_MAX_DIMS];
      PolyUOp *dim_uops[POLY_MAX_DIMS];
      PolyUOp *const *in_dim_uops = SRC_DIM_UOPS(0);
      memcpy(dims, SRC_DIMS(0), in_ndim * sizeof(int64_t));
      for (int i = 0; i < in_ndim; i++)
        dim_uops[i] = in_dim_uops ? in_dim_uops[i] : NULL;
      int last = in_ndim - 1;
      PolyUOp *input_dim = canonical_shape_dim(ctx, dims[last], dim_uops[last]);
      int64_t static_dim = 0;
      if (!input_dim) return make_entry_none(ctx);
      /* Current UOp._shape rejects only a statically non-divisible byte
       * extent; symbolic dimensions retain the exact simplified expression
       * (uop/ops.py:404-411). */
      if (poly_uop_const_i64(input_dim, &static_dim) == 0) {
        int64_t bytes = 0;
        if (__builtin_mul_overflow(static_dim, (int64_t)in_sz, &bytes) || bytes % out_sz != 0)
          return make_entry_none(ctx);
      }
      PolyUOp *input_size = shape_dim_const(ctx, in_sz);
      PolyUOp *output_size = shape_dim_const(ctx, out_sz);
      PolyUOp *bytes =
          poly_uop2(ctx, POLY_OP_MUL, POLY_WEAKINT, input_dim, input_size, poly_arg_none());
      PolyUOp *scaled =
          bytes
              ? poly_uop2(ctx, POLY_OP_FLOORDIV, POLY_WEAKINT, bytes, output_size, poly_arg_none())
              : NULL;
      scaled = scaled ? poly_graph_rewrite(ctx, scaled, poly_symbolic()) : NULL;
      if (!scaled) return make_entry_none(ctx);
      int64_t vmin = 0, vmax = 0;
      poly_uop_minmax(ctx, scaled, &vmin, &vmax);
      if (vmin < 0 || vmax < 0) return make_entry_none(ctx);
      dims[last] = vmax;
      dim_uops[last] = scaled;
      return make_entry_dims_uops(ctx, dims, dim_uops, in_ndim);
    } else {
      if (u->n_src >= 1 && SRC_NDIM(0) >= 0)
        return make_entry_dims_uops(ctx, SRC_DIMS(0), SRC_DIM_UOPS(0), SRC_NDIM(0));
      return make_entry_none(ctx);
    }
  }

  if (op == POLY_OP_WMMA) {
    if (u->n_src != 3) return make_entry_none(ctx);
    ShapeCacheEntry *entries[3];
    int ranks[3];
    for (int i = 0; i < 3; i++) {
      entries[i] = shape_cache_lookup(ctx, u->src[i]);
      if (!entries[i] || entries[i]->ndim < 1) return make_entry_none(ctx);
      ranks[i] = entries[i]->ndim - 1;
    }
    int64_t dims[POLY_MAX_DIMS];
    PolyUOp *dim_uops[POLY_MAX_DIMS];
    int ndim = 0;
    if (!broadcast_shape_entries(ctx, entries, ranks, 3, dims, dim_uops, &ndim) ||
        ndim >= POLY_MAX_DIMS)
      return make_entry_none(ctx);
    dims[ndim] = entries[2]->dims[entries[2]->ndim - 1];
    dim_uops[ndim] = entries[2]->dim_uops[entries[2]->ndim - 1];
    return make_entry_dims_uops(ctx, dims, dim_uops, ndim + 1);
  }

  if (poly_opset_has(POLY_GROUP_ALU, op) || op == POLY_OP_CAST) {
    ShapeCacheEntry **entries = u->n_src ? malloc((size_t)u->n_src * sizeof(*entries)) : NULL;
    if (u->n_src && !entries) return make_entry_none(ctx);
    bool valid = u->n_src > 0;
    for (int i = 0; i < u->n_src; i++) {
      entries[i] = shape_cache_lookup(ctx, u->src[i]);
      valid &= entries[i] && entries[i]->ndim >= 0;
    }
    int64_t dims[POLY_MAX_DIMS];
    PolyUOp *dim_uops[POLY_MAX_DIMS];
    int ndim = 0;
    bool ok = valid && broadcast_shape_entries(ctx, entries, NULL, u->n_src, dims, dim_uops, &ndim);
    free(entries);
    return ok ? make_entry_dims_uops(ctx, dims, dim_uops, ndim) : make_entry_none(ctx);
  }

  /* Default: no shape */
  return make_entry_none(ctx);
}
