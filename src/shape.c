/*
 * shape.c — Shape inference for tensor-level UOp graphs
 *
 * Computes the output shape (tuple of dimension sizes) for any UOp.
 * Walks the graph in toposort order, caches results per UOp pointer.
 *
 * Reference: tinygrad uop/ops.py lines 206-296 (_shape property)
 */

#include "polygrad.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "utils.h"

/* Local helpers */

/* tinygrad-style graph constructors for shape-bearing UOps live here now.
 * They were previously stranded in the retired single-kernel scheduler alongside the old
 * scheduler wrapper, but they are core IR-building APIs rather than a
 * scheduling implementation detail. */

static int poly_buffer_id = 0;

PolyUOp *poly_buffer_on_device(PolyCtx *ctx, PolyDType scalar_dtype, int64_t size, PolyDevice device) {
  int id = poly_buffer_id++;
  PolyUOp *unique = poly_uop0(ctx, POLY_OP_UNIQUE, POLY_VOID, poly_arg_int(id));
  if (device == POLY_DEVICE_AUTO) return poly_uop1(ctx, POLY_OP_BUFFER, scalar_dtype, unique, poly_arg_int(size));

  PolyUOp *dev = poly_uop0(ctx, POLY_OP_DEVICE, POLY_VOID, poly_arg_int((int64_t)device));
  PolyUOp *src[2] = {unique, dev};
  return poly_uop(ctx, POLY_OP_BUFFER, scalar_dtype, src, 2, poly_arg_int(size));
}

PolyUOp *poly_buffer(PolyCtx *ctx, PolyDType scalar_dtype, int64_t size) {
  return poly_buffer_on_device(ctx, scalar_dtype, size, POLY_DEVICE_AUTO);
}

PolyUOp *poly_reshape(PolyCtx *ctx, PolyUOp *src, int64_t *dims, int ndim) {
  PolyArg arg;
  arg.kind = POLY_ARG_INT_TUPLE;
  arg.int_tuple.vals = dims;
  arg.int_tuple.n = ndim;
  return poly_uop1(ctx, POLY_OP_RESHAPE, src->dtype, src, arg);
}

PolyUOp *poly_expand(PolyCtx *ctx, PolyUOp *src, int64_t *dims, int ndim) {
  PolyArg arg;
  arg.kind = POLY_ARG_INT_TUPLE;
  arg.int_tuple.vals = dims;
  arg.int_tuple.n = ndim;
  return poly_uop1(ctx, POLY_OP_EXPAND, src->dtype, src, arg);
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
  if (!ctx || !src || n_axes <= 0) return src;

  /* Use the ctx-owned cached shape here. poly_reduce_axis only inspects dims;
   * requesting a heap copy would make this hot constructor responsible for
   * extra ownership bookkeeping on every reduce. */
  PolyShape shape = poly_uop_shape_cached(ctx, src);
  int64_t filtered_buf[POLY_MAX_DIMS];
  int filtered_n = 0;
  for (int i = 0; i < n_axes; i++) {
    int64_t ax = axes[i];
    if (shape.ndim > 0 && ax >= 0 && ax < shape.ndim && shape.dims[ax] == 1) continue;
    if (filtered_n < POLY_MAX_DIMS) filtered_buf[filtered_n++] = ax;
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
  return poly_uop1(ctx, POLY_OP_REDUCE_AXIS, src->dtype, src, arg);
}

PolyUOp *poly_permute(PolyCtx *ctx, PolyUOp *src, int64_t *perm, int ndim) {
  PolyArg arg;
  arg.kind = POLY_ARG_INT_TUPLE;
  arg.int_tuple.vals = perm;
  arg.int_tuple.n = ndim;
  return poly_uop1(ctx, POLY_OP_PERMUTE, src->dtype, src, arg);
}

PolyUOp *poly_shrink(PolyCtx *ctx, PolyUOp *src, int64_t (*pairs)[2], int ndim) {
  PolyArg arg;
  arg.kind = POLY_ARG_PAIR_TUPLE;
  arg.pair_tuple.pairs = pairs;
  arg.pair_tuple.n = ndim;
  return poly_uop1(ctx, POLY_OP_SHRINK, src->dtype, src, arg);
}

PolyUOp *poly_flip(PolyCtx *ctx, PolyUOp *src, int64_t *axes, int n_axes) {
  PolyArg arg;
  arg.kind = POLY_ARG_INT_TUPLE;
  arg.int_tuple.vals = axes;
  arg.int_tuple.n = n_axes;
  return poly_uop1(ctx, POLY_OP_FLIP, src->dtype, src, arg);
}

PolyUOp *poly_pad(PolyCtx *ctx, PolyUOp *src, int64_t (*pairs)[2], int ndim) {
  PolyArg arg;
  arg.kind = POLY_ARG_PAIR_TUPLE;
  arg.pair_tuple.pairs = pairs;
  arg.pair_tuple.n = ndim;
  return poly_uop1(ctx, POLY_OP_PAD, src->dtype, src, arg);
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
} ShapeCacheEntry;

static ShapeCacheEntry *ensure_shape(PolyCtx *ctx, PolyUOp *u);

PolyMap *poly_ctx_shape_cache(PolyCtx *ctx); /* defined in uop.c */

/* Public lazy accessors */

int poly_uop_ndim(PolyCtx *ctx, const PolyUOp *u) {
  if (!u) return -1;
  return ensure_shape(ctx, (PolyUOp *)u)->ndim;
}

const int64_t *poly_uop_dims(PolyCtx *ctx, const PolyUOp *u) {
  if (!u) return NULL;
  return ensure_shape(ctx, (PolyUOp *)u)->dims;
}

PolyShape poly_uop_shape_cached(PolyCtx *ctx, const PolyUOp *u) {
  if (!u) return POLY_SHAPE_NONE;
  ShapeCacheEntry *e = ensure_shape(ctx, (PolyUOp *)u);
  return (PolyShape){e->dims, e->ndim};
}

/* Public API */

int64_t poly_shape_numel(PolyShape s) {
  if (s.ndim <= 0) return (s.ndim == 0) ? 1 : 0;
  int64_t prod = 1;
  for (int i = 0; i < s.ndim; i++)
    prod *= s.dims[i];
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
 * dims when ndim > 0.  New code should prefer poly_uop_shape_cached()
 * or poly_uop_ndim()/poly_uop_dims() which avoid the heap copy. */
PolyShape poly_uop_shape(PolyCtx *ctx, PolyUOp *u) {
  PolyShape cached = poly_uop_shape_cached(ctx, (const PolyUOp *)u);
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
  return e;
}

static ShapeCacheEntry *make_entry_scalar(PolyCtx *ctx) {
  ShapeCacheEntry *e = poly_arena_alloc(poly_ctx_arena(ctx), sizeof(ShapeCacheEntry), 8);
  e->ndim = 0;
  e->dims = NULL;
  return e;
}

static ShapeCacheEntry *make_entry_1d(PolyCtx *ctx, int64_t dim0) {
  ShapeCacheEntry *e = poly_arena_alloc(poly_ctx_arena(ctx), sizeof(ShapeCacheEntry), 8);
  e->ndim = 1;
  e->dims = poly_arena_alloc(poly_ctx_arena(ctx), sizeof(int64_t), _Alignof(int64_t));
  e->dims[0] = dim0;
  return e;
}

static ShapeCacheEntry *make_entry_dims(PolyCtx *ctx, const int64_t *dims, int ndim) {
  ShapeCacheEntry *e = poly_arena_alloc(poly_ctx_arena(ctx), sizeof(ShapeCacheEntry), 8);
  e->ndim = (int8_t)ndim;
  if (ndim > 0) {
    e->dims = poly_arena_alloc(poly_ctx_arena(ctx), ndim * sizeof(int64_t), _Alignof(int64_t));
    memcpy(e->dims, dims, ndim * sizeof(int64_t));
  } else {
    e->dims = NULL;
  }
  return e;
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

/* Read source shape from the ctx cache. ensure_shape() fills the cache in
 * topological order, matching tinygrad's recursive_property behavior without
 * recursive C calls through deep UOp chains. */
#define SRC_NDIM(i) src_ndim(ctx, u, (i))
#define SRC_DIMS(i) src_dims(ctx, u, (i))

static ShapeCacheEntry *compute_and_cache(PolyCtx *ctx, PolyUOp *u);

static ShapeCacheEntry *ensure_shape(PolyCtx *ctx, PolyUOp *u) {
  ShapeCacheEntry *cached = shape_cache_lookup(ctx, u);
  if (cached) return cached;

  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, u, &n_topo);
  PolyMap *cache = poly_ctx_shape_cache(ctx);
  if (!topo) {
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
  if (cached) return cached;
  ShapeCacheEntry *entry = make_entry_none(ctx);
  poly_map_set(cache, poly_ptr_hash(u), u, entry, poly_ptr_eq);
  return entry;
}

static ShapeCacheEntry *compute_and_cache(PolyCtx *ctx, PolyUOp *u) {
  PolyOps op = u->op;

  /* STORE: inherit shape from value (src[1]) */
  if (op == POLY_OP_STORE && u->n_src >= 2 && SRC_NDIM(1) >= 0)
    return make_entry_dims(ctx, SRC_DIMS(1), SRC_NDIM(1));

  /* No-shape ops (kernel-level, never have tensor shapes) */
  if (op == POLY_OP_RANGE || op == POLY_OP_LOAD || op == POLY_OP_SINK || op == POLY_OP_IF ||
      op == POLY_OP_ENDIF || op == POLY_OP_BARRIER || op == POLY_OP_SPECIAL ||
      op == POLY_OP_VECTORIZE || op == POLY_OP_GEP || op == POLY_OP_LINEAR ||
      op == POLY_OP_PROGRAM || op == POLY_OP_SOURCE || op == POLY_OP_BINARY || op == POLY_OP_INS ||
      op == POLY_OP_CUSTOM || op == POLY_OP_CUSTOMI || op == POLY_OP_UNIQUE ||
      op == POLY_OP_LUNIQUE || op == POLY_OP_UNROLL || op == POLY_OP_CONTRACT ||
      op == POLY_OP_VCAT || op == POLY_OP_PTRCAT || op == POLY_OP_CALL) {
    return make_entry_none(ctx);
  }

  /* Scalar constants */
  if (op == POLY_OP_CONST || op == POLY_OP_VCONST || op == POLY_OP_DEFINE_VAR ||
      op == POLY_OP_BIND) {
    return make_entry_scalar(ctx);
  }

  /* BUFFER */
  if (op == POLY_OP_BUFFER) {
    if (u->arg.kind == POLY_ARG_INT) {
      /* Dynamic buffer: BUFFER(src=(UNIQUE, DEFINE_VAR, CONST...)) → (max_val, K, ...) */
      if (u->n_src >= 2 && u->src[1]->op == POLY_OP_DEFINE_VAR) {
        int ndim = u->n_src - 1;
        int64_t dims[POLY_MAX_DIMS];
        dims[0] = u->src[1]->arg.define_var.max_val;
        for (int i = 1; i < ndim && i < POLY_MAX_DIMS; i++)
          dims[i] = u->src[1 + i]->arg.i;
        return make_entry_dims(ctx, dims, ndim);
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

  /* PARAM: shape from pointer dtype size, else no shape */
  if (op == POLY_OP_PARAM) {
    if (u->dtype.is_ptr && u->dtype.ptr_size > 0) {
      return make_entry_1d(ctx, u->dtype.ptr_size);
    } else {
      return make_entry_none(ctx);
    }
  }

  /* INDEX: conditional shape for pointer types */
  if (op == POLY_OP_INDEX) {
    if (!u->dtype.is_ptr) {
      return make_entry_none(ctx);
    }
    if (u->n_src < 1 || SRC_NDIM(0) <= 0) {
      return make_entry_none(ctx);
    }
    int8_t source_ndim = SRC_NDIM(0);
    int n_indices = u->n_src - 1;
    if (n_indices >= source_ndim) {
      return make_entry_none(ctx);
    }
    int remaining = source_ndim - n_indices;
    return make_entry_dims(ctx, SRC_DIMS(0) + n_indices, remaining);
  }

  /* BUFFERIZE: shape from range bounds */
  if (op == POLY_OP_BUFFERIZE) {
    int n_ranges = u->n_src - 1;
    if (n_ranges <= 0) {
      return make_entry_none(ctx);
    }
    int64_t dims[POLY_MAX_DIMS];
    for (int i = 0; i < n_ranges && i < POLY_MAX_DIMS; i++) {
      PolyUOp *rng = u->src[1 + i];
      if (rng->op == POLY_OP_RANGE && rng->n_src >= 1 && rng->src[0]->op == POLY_OP_CONST &&
          rng->src[0]->arg.kind == POLY_ARG_INT) {
        dims[i] = rng->src[0]->arg.i;
      } else {
        dims[i] = -1; /* symbolic, resolved later */
      }
    }
    return make_entry_dims(ctx, dims, n_ranges);
  }

  /* RESHAPE, EXPAND: shape from int_tuple arg */
  if ((op == POLY_OP_RESHAPE || op == POLY_OP_EXPAND) && u->arg.kind == POLY_ARG_INT_TUPLE) {
    return make_entry_dims(ctx, u->arg.int_tuple.vals, u->arg.int_tuple.n);
  }

  /* PERMUTE: reorder src[0] shape */
  if (op == POLY_OP_PERMUTE && u->n_src >= 1 && u->arg.kind == POLY_ARG_INT_TUPLE) {
    int8_t in_ndim = SRC_NDIM(0);
    if (in_ndim <= 0) {
      return make_entry_none(ctx);
    }
    int n = u->arg.int_tuple.n;
    int64_t dims[POLY_MAX_DIMS];
    for (int i = 0; i < n && i < in_ndim; i++)
      dims[i] = SRC_DIMS(0)[u->arg.int_tuple.vals[i]];
    return make_entry_dims(ctx, dims, n);
  }

  /* PAD: output = input + begin + end per axis */
  if (op == POLY_OP_PAD && u->n_src >= 1 && u->arg.kind == POLY_ARG_PAIR_TUPLE) {
    int8_t in_ndim = SRC_NDIM(0);
    if (in_ndim <= 0) {
      return make_entry_none(ctx);
    }
    int64_t dims[POLY_MAX_DIMS];
    for (int i = 0; i < in_ndim && i < u->arg.pair_tuple.n; i++)
      dims[i] = SRC_DIMS(0)[i] + u->arg.pair_tuple.pairs[i][0] + u->arg.pair_tuple.pairs[i][1];
    return make_entry_dims(ctx, dims, in_ndim);
  }

  /* SHRINK: output = end - begin per axis */
  if (op == POLY_OP_SHRINK && u->n_src >= 1 && u->arg.kind == POLY_ARG_PAIR_TUPLE) {
    int8_t in_ndim = SRC_NDIM(0);
    if (in_ndim <= 0) {
      return make_entry_none(ctx);
    }
    int64_t dims[POLY_MAX_DIMS];
    for (int i = 0; i < in_ndim && i < u->arg.pair_tuple.n; i++)
      dims[i] = u->arg.pair_tuple.pairs[i][1] - u->arg.pair_tuple.pairs[i][0];
    return make_entry_dims(ctx, dims, in_ndim);
  }

  /* FLIP: same shape as src[0] */
  if (op == POLY_OP_FLIP) {
    if (u->n_src >= 1 && SRC_NDIM(0) >= 0) return make_entry_dims(ctx, SRC_DIMS(0), SRC_NDIM(0));
    return make_entry_none(ctx);
  }

  /* REDUCE_AXIS: dims at reduction axes become 1 */
  if (op == POLY_OP_REDUCE_AXIS && u->n_src >= 1 && u->arg.kind == POLY_ARG_REDUCE_AXIS) {
    int8_t in_ndim = SRC_NDIM(0);
    if (in_ndim <= 0) {
      return make_entry_none(ctx);
    }
    int64_t dims[POLY_MAX_DIMS];
    memcpy(dims, SRC_DIMS(0), in_ndim * sizeof(int64_t));
    for (int i = 0; i < u->arg.reduce_axis.n; i++) {
      int ax = (int)u->arg.reduce_axis.axes[i];
      if (ax >= 0 && ax < in_ndim) dims[ax] = 1;
    }
    return make_entry_dims(ctx, dims, in_ndim);
  }

  /* ASSIGN: use stored logical shape (arg) if present */
  if (op == POLY_OP_ASSIGN) {
    if (u->arg.kind == POLY_ARG_INT_TUPLE && u->arg.int_tuple.n > 0)
      return make_entry_dims(ctx, u->arg.int_tuple.vals, u->arg.int_tuple.n);
    if (u->n_src >= 1 && SRC_NDIM(0) >= 0) return make_entry_dims(ctx, SRC_DIMS(0), SRC_NDIM(0));
    return make_entry_none(ctx);
  }

  /* Passthrough ops: inherit src[0] shape */
  if (op == POLY_OP_CONTIGUOUS || op == POLY_OP_DETACH || op == POLY_OP_CONTIGUOUS_BACKWARD ||
      op == POLY_OP_COPY || op == POLY_OP_NOOP || op == POLY_OP_REDUCE || op == POLY_OP_AFTER ||
      op == POLY_OP_END || op == POLY_OP_GROUP) {
    if (u->n_src >= 1 && SRC_NDIM(0) >= 0) return make_entry_dims(ctx, SRC_DIMS(0), SRC_NDIM(0));
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
      memcpy(dims, SRC_DIMS(0), in_ndim * sizeof(int64_t));
      dims[in_ndim - 1] = (SRC_DIMS(0)[in_ndim - 1] * in_sz) / out_sz;
      return make_entry_dims(ctx, dims, in_ndim);
    } else {
      if (u->n_src >= 1 && SRC_NDIM(0) >= 0) return make_entry_dims(ctx, SRC_DIMS(0), SRC_NDIM(0));
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
    int out_ndim = -1;
    for (int i = 0; i < u->n_src; i++) {
      int8_t si_ndim = SRC_NDIM(i);
      if (si_ndim < 0) continue;
      const int64_t *si_dims = SRC_DIMS(i);
      if (out_ndim < 0) {
        out_ndim = si_ndim;
        if (si_ndim > 0) memcpy(out_dims, si_dims, si_ndim * sizeof(int64_t));
        continue;
      }
      int ndim = (out_ndim > si_ndim) ? out_ndim : si_ndim;
      int64_t merged[POLY_MAX_DIMS];
      for (int ax = 0; ax < ndim; ax++) {
        int ai = out_ndim - 1 - ax;
        int bi = si_ndim - 1 - ax;
        int64_t a = (ai >= 0) ? out_dims[ai] : 1;
        int64_t b = (bi >= 0) ? si_dims[bi] : 1;
        if (a != b && a != 1 && b != 1) {
          return make_entry_none(ctx);
        }
        merged[ndim - 1 - ax] = (a > b) ? a : b;
      }
      out_ndim = ndim;
      memcpy(out_dims, merged, ndim * sizeof(int64_t));
    }
    if (out_ndim < 0) {
      return make_entry_none(ctx);
    }
    return make_entry_dims(ctx, out_dims, out_ndim);
  }

  /* Default: no shape */
  return make_entry_none(ctx);
}
