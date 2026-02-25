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
          fprintf(stderr, "polygrad: shape mismatch in %s\n", poly_op_name(op));
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
