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

/* ── Local helpers ────────────────────────────────────────────────────── */

static bool ptr_eq(const void *a, const void *b) { return a == b; }

static uint32_t ptr_hash(const void *p) {
  uintptr_t v = (uintptr_t)p;
  return (uint32_t)(v ^ (v >> 16) ^ (sizeof(v) > 4 ? (uint32_t)(v >> 32) : 0));
}

/* Heap-allocate a shape with copied dims */
static PolyShape heap_shape(int64_t *dims, int ndim) {
  if (ndim < 0) return POLY_SHAPE_NONE;
  if (ndim == 0) return (PolyShape){ NULL, 0 };
  int64_t *copy = malloc(ndim * sizeof(int64_t));
  memcpy(copy, dims, ndim * sizeof(int64_t));
  return (PolyShape){ copy, ndim };
}

/* ── Shape accessors ──────────────────────────────────────────────────── */

int poly_uop_ndim(const PolyUOp *u) { return u ? u->_shape_ndim : -1; }
const int64_t *poly_uop_dims(const PolyUOp *u) { return u ? u->_shape_dims : NULL; }

/* ── Public API ───────────────────────────────────────────────────────── */

int64_t poly_shape_numel(PolyShape s) {
  if (s.ndim <= 0) return (s.ndim == 0) ? 1 : 0;
  int64_t prod = 1;
  for (int i = 0; i < s.ndim; i++) prod *= s.dims[i];
  return prod;
}

bool poly_shape_eq(PolyShape a, PolyShape b) {
  if (a.ndim != b.ndim) return false;
  if (a.ndim <= 0) return true;
  return memcmp(a.dims, b.dims, a.ndim * sizeof(int64_t)) == 0;
}

/* ── Shape cache ─────────────────────────────────────────────────────── */

static PolyShape get_cached(PolyMap *cache, PolyUOp *u) {
  PolyShape *s = poly_map_get(cache, ptr_hash(u), u, ptr_eq);
  if (s) return *s;
  return POLY_SHAPE_NONE;
}

static void set_cached(PolyMap *cache, PolyUOp *u, PolyShape s) {
  PolyShape *stored = malloc(sizeof(PolyShape));
  *stored = s;
  poly_map_set(cache, ptr_hash(u), u, stored, ptr_eq);
}

/* ── Shape computation (returns heap-allocated dims) ─────────────────── */

static PolyShape compute_shape(PolyUOp *u, PolyMap *cache) {
  PolyOps op = u->op;

  /* Ops with no tensor shape (kernel-level) */
  if (op == POLY_OP_RANGE || op == POLY_OP_INDEX || op == POLY_OP_LOAD ||
      op == POLY_OP_END || op == POLY_OP_SINK || op == POLY_OP_PARAM ||
      op == POLY_OP_IF || op == POLY_OP_ENDIF || op == POLY_OP_BARRIER ||
      op == POLY_OP_SPECIAL || op == POLY_OP_VECTORIZE || op == POLY_OP_GEP ||
      op == POLY_OP_VCONST || op == POLY_OP_DEFINE_LOCAL || op == POLY_OP_DEFINE_REG ||
      op == POLY_OP_LINEAR || op == POLY_OP_PROGRAM || op == POLY_OP_SOURCE ||
      op == POLY_OP_BINARY || op == POLY_OP_INS || op == POLY_OP_CUSTOM ||
      op == POLY_OP_CUSTOMI || op == POLY_OP_UNIQUE || op == POLY_OP_LUNIQUE ||
      op == POLY_OP_UNROLL || op == POLY_OP_CONTRACT)
    return POLY_SHAPE_NONE;

  /* BUFFER: shape depends on sources.
   * Dynamic buffer: BUFFER(src=(UNIQUE, DEFINE_VAR, CONST(K)...)) → shape = (max_val, K, ...)
   * Static buffer:  BUFFER(src=(UNIQUE,), arg=size) → shape = (size,) */
  if (op == POLY_OP_BUFFER && u->arg.kind == POLY_ARG_INT) {
    if (u->n_src >= 2 && u->src[1]->op == POLY_OP_DEFINE_VAR) {
      /* Dynamic buffer: src[0]=UNIQUE, src[1]=DEFINE_VAR, src[2..]=CONST inner dims */
      int ndim = u->n_src - 1;  /* skip UNIQUE */
      int64_t dims[POLY_MAX_DIMS];
      dims[0] = u->src[1]->arg.define_var.max_val;
      for (int i = 1; i < ndim && i < POLY_MAX_DIMS; i++)
        dims[i] = u->src[1 + i]->arg.i;  /* CONST inner dims */
      return heap_shape(dims, ndim);
    }
    return heap_shape(&u->arg.i, 1);
  }

  /* CONST / DEFINE_VAR / BIND: scalar () */
  if (op == POLY_OP_CONST || op == POLY_OP_DEFINE_VAR || op == POLY_OP_BIND)
    return (PolyShape){ NULL, 0 };

  /* STORE: inherit shape from value (src[1]) */
  if (op == POLY_OP_STORE && u->n_src >= 2) {
    PolyShape s = get_cached(cache, u->src[1]);
    return heap_shape(s.dims, s.ndim);
  }

  /* RESHAPE / EXPAND: shape from int_tuple arg */
  if ((op == POLY_OP_RESHAPE || op == POLY_OP_EXPAND) &&
      u->arg.kind == POLY_ARG_INT_TUPLE)
    return heap_shape(u->arg.int_tuple.vals, u->arg.int_tuple.n);

  /* PERMUTE: reorder src[0] shape by arg tuple */
  if (op == POLY_OP_PERMUTE && u->n_src >= 1 && u->arg.kind == POLY_ARG_INT_TUPLE) {
    PolyShape in = get_cached(cache, u->src[0]);
    if (in.ndim < 0) return POLY_SHAPE_NONE;
    int n = u->arg.int_tuple.n;
    int64_t out[POLY_MAX_DIMS];
    for (int i = 0; i < n && i < in.ndim; i++)
      out[i] = in.dims[u->arg.int_tuple.vals[i]];
    return heap_shape(out, n);
  }

  /* PAD: output = input + begin + end per axis */
  if (op == POLY_OP_PAD && u->n_src >= 1 && u->arg.kind == POLY_ARG_PAIR_TUPLE) {
    PolyShape in = get_cached(cache, u->src[0]);
    if (in.ndim < 0) return POLY_SHAPE_NONE;
    int64_t out[POLY_MAX_DIMS];
    for (int i = 0; i < in.ndim && i < u->arg.pair_tuple.n; i++)
      out[i] = in.dims[i] + u->arg.pair_tuple.pairs[i][0] + u->arg.pair_tuple.pairs[i][1];
    return heap_shape(out, in.ndim);
  }

  /* SHRINK: output = end - begin per axis */
  if (op == POLY_OP_SHRINK && u->n_src >= 1 && u->arg.kind == POLY_ARG_PAIR_TUPLE) {
    PolyShape in = get_cached(cache, u->src[0]);
    if (in.ndim < 0) return POLY_SHAPE_NONE;
    int64_t out[POLY_MAX_DIMS];
    for (int i = 0; i < in.ndim && i < u->arg.pair_tuple.n; i++)
      out[i] = u->arg.pair_tuple.pairs[i][1] - u->arg.pair_tuple.pairs[i][0];
    return heap_shape(out, in.ndim);
  }

  /* FLIP: same shape as input */
  if (op == POLY_OP_FLIP && u->n_src >= 1) {
    PolyShape s = get_cached(cache, u->src[0]);
    return heap_shape(s.dims, s.ndim);
  }

  /* REDUCE_AXIS: dims at reduction axes become 1 */
  if (op == POLY_OP_REDUCE_AXIS && u->n_src >= 1 &&
      u->arg.kind == POLY_ARG_REDUCE_AXIS) {
    PolyShape in = get_cached(cache, u->src[0]);
    if (in.ndim < 0) return POLY_SHAPE_NONE;
    int64_t out[POLY_MAX_DIMS];
    memcpy(out, in.dims, in.ndim * sizeof(int64_t));
    for (int i = 0; i < u->arg.reduce_axis.n; i++) {
      int ax = (int)u->arg.reduce_axis.axes[i];
      if (ax >= 0 && ax < in.ndim) out[ax] = 1;
    }
    return heap_shape(out, in.ndim);
  }

  /* Pass-through ops: inherit src[0] shape */
  if (op == POLY_OP_CONTIGUOUS || op == POLY_OP_DETACH ||
      op == POLY_OP_CONTIGUOUS_BACKWARD || op == POLY_OP_COPY ||
      op == POLY_OP_NOOP || op == POLY_OP_BUFFERIZE || op == POLY_OP_ASSIGN) {
    if (u->n_src >= 1) {
      PolyShape s = get_cached(cache, u->src[0]);
      return heap_shape(s.dims, s.ndim);
    }
    return POLY_SHAPE_NONE;
  }

  /* Elementwise ops (ALU, CAST, BITCAST): broadcast shapes using
   * NumPy/tinygrad rules (align trailing dims, allow dim==1 expansion). */
  if (poly_opset_has(POLY_GROUP_ALU, op) || op == POLY_OP_CAST || op == POLY_OP_BITCAST) {
    int64_t out_dims[POLY_MAX_DIMS];
    int out_ndim = -1;
    for (int i = 0; i < u->n_src; i++) {
      PolyShape si = get_cached(cache, u->src[i]);
      if (si.ndim < 0) continue;

      if (out_ndim < 0) {
        /* First known shape initializes the broadcast accumulator.
         * Scalars initialize to ndim=0 and can still expand later. */
        out_ndim = si.ndim;
        for (int d = 0; d < si.ndim; d++) out_dims[d] = si.dims[d];
        continue;
      }

      int ndim = (out_ndim > si.ndim) ? out_ndim : si.ndim;
      int64_t merged[POLY_MAX_DIMS];
      for (int ax = 0; ax < ndim; ax++) {
        int ai = out_ndim - 1 - ax;
        int bi = si.ndim - 1 - ax;
        int64_t a = (ai >= 0) ? out_dims[ai] : 1;
        int64_t b = (bi >= 0) ? si.dims[bi] : 1;
        if (!(a == b || a == 1 || b == 1)) {
          fprintf(stderr, "polygrad: shape mismatch in %s axis %d: %lld vs %lld (shapes: [",
                  poly_op_name(op), ax, (long long)a, (long long)b);
          for (int d2 = 0; d2 < out_ndim; d2++)
            fprintf(stderr, "%s%lld", d2 ? "," : "", (long long)out_dims[d2]);
          fprintf(stderr, "] vs [");
          for (int d2 = 0; d2 < si.ndim; d2++)
            fprintf(stderr, "%s%lld", d2 ? "," : "", (long long)si.dims[d2]);
          fprintf(stderr, "])\n");
          return POLY_SHAPE_NONE;
        }
        merged[ndim - 1 - ax] = (a == 1) ? b : a;
      }
      out_ndim = ndim;
      for (int d = 0; d < out_ndim; d++) out_dims[d] = merged[d];
    }

    if (out_ndim < 0) return POLY_SHAPE_NONE;
    return heap_shape(out_dims, out_ndim);
  }

  return POLY_SHAPE_NONE;
}

/* ── Main entry point ─────────────────────────────────────────────────── */

PolyShape poly_uop_shape(PolyCtx *ctx, PolyUOp *u) {
  int n_topo;
  PolyUOp **topo = poly_toposort(ctx, u, &n_topo);
  PolyMap *cache = poly_map_new(n_topo * 2);

  for (int i = 0; i < n_topo; i++) {
    PolyShape s = compute_shape(topo[i], cache);
    set_cached(cache, topo[i], s);
  }

  /* Copy result before freeing cache */
  PolyShape result = get_cached(cache, u);
  PolyShape out = POLY_SHAPE_NONE;
  if (result.ndim >= 0) {
    if (result.ndim == 0) {
      out = (PolyShape){ NULL, 0 };
    } else {
      int64_t *dims = malloc(result.ndim * sizeof(int64_t));
      memcpy(dims, result.dims, result.ndim * sizeof(int64_t));
      out = (PolyShape){ dims, result.ndim };
    }
  }

  /* Free all cache entries */
  for (int i = 0; i < n_topo; i++) {
    PolyShape *s = poly_map_get(cache, ptr_hash(topo[i]), topo[i], ptr_eq);
    if (s) {
      if (s->ndim > 0 && s->dims) free(s->dims);
      free(s);
    }
  }
  poly_map_destroy(cache);

  return out;
}

/* ═══════════════════════════════════════════════════════════════════════ */
/*  Eager shape computation — called once at UOp creation time            */
/*  Reads source shapes from src[i]->_shape_* (already computed)          */
/*  Corrected rules verified against tinygrad ops.py:206-318              */
/* ═══════════════════════════════════════════════════════════════════════ */

static void shape_set_none(PolyUOp *u) {
  u->_shape_ndim = -1;
  u->_shape_dims = NULL;
}

static void shape_set_scalar(PolyUOp *u) {
  u->_shape_ndim = 0;
  u->_shape_dims = NULL;
}

static void shape_set_1d(PolyCtx *ctx, PolyUOp *u, int64_t dim0) {
  u->_shape_ndim = 1;
  u->_shape_dims = poly_arena_alloc(poly_ctx_arena(ctx), sizeof(int64_t), _Alignof(int64_t));
  u->_shape_dims[0] = dim0;
}

static void shape_set_dims(PolyCtx *ctx, PolyUOp *u, const int64_t *dims, int ndim) {
  u->_shape_ndim = (int8_t)ndim;
  if (ndim > 0) {
    u->_shape_dims = poly_arena_alloc(poly_ctx_arena(ctx), ndim * sizeof(int64_t), _Alignof(int64_t));
    memcpy(u->_shape_dims, dims, ndim * sizeof(int64_t));
  } else {
    u->_shape_dims = NULL;
  }
}

static void shape_passthrough_src0(PolyCtx *ctx, PolyUOp *u) {
  if (u->n_src >= 1 && u->src[0]->_shape_ndim >= 0) {
    shape_set_dims(ctx, u, u->src[0]->_shape_dims, u->src[0]->_shape_ndim);
  } else {
    shape_set_none(u);
  }
}

void poly_uop_compute_shape(PolyCtx *ctx, PolyUOp *u) {
  PolyOps op = u->op;

  /* ── No-shape ops (kernel-level, never have tensor shapes) ────────── */
  if (op == POLY_OP_RANGE || op == POLY_OP_LOAD || op == POLY_OP_STORE ||
      op == POLY_OP_SINK || op == POLY_OP_IF || op == POLY_OP_ENDIF ||
      op == POLY_OP_BARRIER || op == POLY_OP_SPECIAL ||
      op == POLY_OP_VECTORIZE || op == POLY_OP_GEP ||
      op == POLY_OP_LINEAR || op == POLY_OP_PROGRAM || op == POLY_OP_SOURCE ||
      op == POLY_OP_BINARY || op == POLY_OP_INS || op == POLY_OP_CUSTOM ||
      op == POLY_OP_CUSTOMI || op == POLY_OP_UNIQUE || op == POLY_OP_LUNIQUE ||
      op == POLY_OP_UNROLL || op == POLY_OP_CONTRACT ||
      op == POLY_OP_VCAT || op == POLY_OP_PTRCAT || op == POLY_OP_CALL) {
    shape_set_none(u);
    return;
  }

  /* ── Scalar constants ─────────────────────────────────────────────── */
  if (op == POLY_OP_CONST || op == POLY_OP_VCONST ||
      op == POLY_OP_DEFINE_VAR || op == POLY_OP_BIND) {
    shape_set_scalar(u);
    return;
  }

  /* ── BUFFER ───────────────────────────────────────────────────────── */
  if (op == POLY_OP_BUFFER) {
    if (u->arg.kind == POLY_ARG_INT) {
      shape_set_1d(ctx, u, u->arg.i);
    } else {
      shape_set_none(u);
    }
    return;
  }

  /* ── DEFINE_LOCAL, DEFINE_REG: shape from pointer dtype size ──────── */
  if (op == POLY_OP_DEFINE_LOCAL || op == POLY_OP_DEFINE_REG) {
    if (u->dtype.is_ptr && u->dtype.ptr_size > 0) {
      shape_set_1d(ctx, u, u->dtype.ptr_size);
    } else {
      shape_set_none(u);
    }
    return;
  }

  /* ── PARAM: shape from pointer dtype size, else no shape ──────────── */
  if (op == POLY_OP_PARAM) {
    if (u->dtype.is_ptr && u->dtype.ptr_size > 0) {
      shape_set_1d(ctx, u, u->dtype.ptr_size);
    } else {
      shape_set_none(u);
    }
    return;
  }

  /* ── INDEX: conditional shape for pointer types ───────────────────── */
  if (op == POLY_OP_INDEX) {
    if (!u->dtype.is_ptr) { shape_set_none(u); return; }
    if (u->n_src < 1 || u->src[0]->_shape_ndim <= 0) { shape_set_none(u); return; }
    int8_t src_ndim = u->src[0]->_shape_ndim;
    int n_indices = u->n_src - 1;
    if (n_indices >= src_ndim) { shape_set_none(u); return; }
    int remaining = src_ndim - n_indices;
    shape_set_dims(ctx, u, u->src[0]->_shape_dims + n_indices, remaining);
    return;
  }

  /* ── BUFFERIZE: shape from range bounds ───────────────────────────── */
  if (op == POLY_OP_BUFFERIZE) {
    int n_ranges = u->n_src - 1;
    if (n_ranges <= 0) { shape_set_none(u); return; }
    int64_t dims[POLY_MAX_DIMS];
    for (int i = 0; i < n_ranges && i < POLY_MAX_DIMS; i++) {
      PolyUOp *rng = u->src[1 + i];
      if (rng->op == POLY_OP_RANGE && rng->n_src >= 1 &&
          rng->src[0]->op == POLY_OP_CONST && rng->src[0]->arg.kind == POLY_ARG_INT) {
        dims[i] = rng->src[0]->arg.i;
      } else {
        dims[i] = -1;  /* symbolic, resolved later */
      }
    }
    shape_set_dims(ctx, u, dims, n_ranges);
    return;
  }

  /* ── RESHAPE, EXPAND: shape from int_tuple arg ────────────────────── */
  if ((op == POLY_OP_RESHAPE || op == POLY_OP_EXPAND) &&
      u->arg.kind == POLY_ARG_INT_TUPLE) {
    shape_set_dims(ctx, u, u->arg.int_tuple.vals, u->arg.int_tuple.n);
    return;
  }

  /* ── PERMUTE: reorder src[0] shape ────────────────────────────────── */
  if (op == POLY_OP_PERMUTE && u->n_src >= 1 && u->arg.kind == POLY_ARG_INT_TUPLE) {
    int8_t in_ndim = u->src[0]->_shape_ndim;
    if (in_ndim <= 0) { shape_set_none(u); return; }
    int n = u->arg.int_tuple.n;
    int64_t dims[POLY_MAX_DIMS];
    for (int i = 0; i < n && i < in_ndim; i++)
      dims[i] = u->src[0]->_shape_dims[u->arg.int_tuple.vals[i]];
    shape_set_dims(ctx, u, dims, n);
    return;
  }

  /* ── PAD: output = input + begin + end per axis ───────────────────── */
  if (op == POLY_OP_PAD && u->n_src >= 1 && u->arg.kind == POLY_ARG_PAIR_TUPLE) {
    int8_t in_ndim = u->src[0]->_shape_ndim;
    if (in_ndim <= 0) { shape_set_none(u); return; }
    int64_t dims[POLY_MAX_DIMS];
    for (int i = 0; i < in_ndim && i < u->arg.pair_tuple.n; i++)
      dims[i] = u->src[0]->_shape_dims[i] + u->arg.pair_tuple.pairs[i][0] + u->arg.pair_tuple.pairs[i][1];
    shape_set_dims(ctx, u, dims, in_ndim);
    return;
  }

  /* ── SHRINK: output = end - begin per axis ────────────────────────── */
  if (op == POLY_OP_SHRINK && u->n_src >= 1 && u->arg.kind == POLY_ARG_PAIR_TUPLE) {
    int8_t in_ndim = u->src[0]->_shape_ndim;
    if (in_ndim <= 0) { shape_set_none(u); return; }
    int64_t dims[POLY_MAX_DIMS];
    for (int i = 0; i < in_ndim && i < u->arg.pair_tuple.n; i++)
      dims[i] = u->arg.pair_tuple.pairs[i][1] - u->arg.pair_tuple.pairs[i][0];
    shape_set_dims(ctx, u, dims, in_ndim);
    return;
  }

  /* ── FLIP: same shape as src[0] ───────────────────────────────────── */
  if (op == POLY_OP_FLIP) { shape_passthrough_src0(ctx, u); return; }

  /* ── REDUCE_AXIS: dims at reduction axes become 1 ─────────────────── */
  if (op == POLY_OP_REDUCE_AXIS && u->n_src >= 1 && u->arg.kind == POLY_ARG_REDUCE_AXIS) {
    int8_t in_ndim = u->src[0]->_shape_ndim;
    if (in_ndim <= 0) { shape_set_none(u); return; }
    int64_t dims[POLY_MAX_DIMS];
    memcpy(dims, u->src[0]->_shape_dims, in_ndim * sizeof(int64_t));
    for (int i = 0; i < u->arg.reduce_axis.n; i++) {
      int ax = (int)u->arg.reduce_axis.axes[i];
      if (ax >= 0 && ax < in_ndim) dims[ax] = 1;
    }
    shape_set_dims(ctx, u, dims, in_ndim);
    return;
  }

  /* ── Passthrough ops: inherit src[0] shape ────────────────────────── */
  if (op == POLY_OP_CONTIGUOUS || op == POLY_OP_DETACH ||
      op == POLY_OP_CONTIGUOUS_BACKWARD || op == POLY_OP_COPY ||
      op == POLY_OP_NOOP || op == POLY_OP_ASSIGN ||
      op == POLY_OP_REDUCE || op == POLY_OP_AFTER || op == POLY_OP_END ||
      op == POLY_OP_GROUP) {
    shape_passthrough_src0(ctx, u);
    return;
  }

  /* ── BITCAST: scale last dim if itemsize differs ──────────────────── */
  if (op == POLY_OP_BITCAST && u->n_src >= 1) {
    int8_t in_ndim = u->src[0]->_shape_ndim;
    if (in_ndim < 0) { shape_set_none(u); return; }
    if (in_ndim == 0) { shape_set_scalar(u); return; }
    int out_sz = poly_dtype_itemsize(poly_dtype_scalar(u->dtype));
    int in_sz = poly_dtype_itemsize(poly_dtype_scalar(u->src[0]->dtype));
    if (out_sz != in_sz && in_sz > 0 && out_sz > 0) {
      int64_t dims[POLY_MAX_DIMS];
      memcpy(dims, u->src[0]->_shape_dims, in_ndim * sizeof(int64_t));
      dims[in_ndim - 1] = (u->src[0]->_shape_dims[in_ndim - 1] * in_sz) / out_sz;
      shape_set_dims(ctx, u, dims, in_ndim);
    } else {
      shape_passthrough_src0(ctx, u);
    }
    return;
  }

  /* ── CAST: ptr→non-ptr returns no shape; else same as ALU ─────────── */
  if (op == POLY_OP_CAST && u->n_src >= 1) {
    if (u->src[0]->dtype.is_ptr && !u->dtype.is_ptr) {
      shape_set_none(u);
      return;
    }
    /* Fall through to ALU handling */
  }

  /* ── ALU + CAST: take the source with highest ndim ───────────────── */
  /* In tinygrad, all ALU sources have the same shape (broadcasting done
   * before ALU creation). In polygrad, scalar CONSTs can appear as ALU
   * sources alongside tensors. Take the highest-ndim source's shape. */
  if (poly_opset_has(POLY_GROUP_ALU, op) || op == POLY_OP_CAST) {
    int8_t best_ndim = -1;
    const int64_t *best_dims = NULL;
    for (int i = 0; i < u->n_src; i++) {
      if (u->src[i]->_shape_ndim > best_ndim) {
        best_ndim = u->src[i]->_shape_ndim;
        best_dims = u->src[i]->_shape_dims;
      }
    }
    if (best_ndim < 0) { shape_set_none(u); return; }
    shape_set_dims(ctx, u, best_dims, best_ndim);
    return;
  }

  /* ── Default: no shape ────────────────────────────────────────────── */
  shape_set_none(u);
}
