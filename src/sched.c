/*
 * sched.c — Tensor-to-kernel scheduler
 *
 * Converts tensor-level graphs (BUFFER + ALU + movement ops)
 * into kernel-level IR (PARAM/RANGE/INDEX/LOAD/STORE/END/SINK).
 *
 * Simplified rangeify: recursive lowering for elementwise ops,
 * with RESHAPE and EXPAND index transforms.
 *
 * Reference: tinygrad schedule/rangeify.py, schedule/indexing.py
 */

#include "sched.h"
#include "rangeify.h"
#include "codegen.h"  /* for linearize/render/compile if needed */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── Local helpers ────────────────────────────────────────────────────── */

static bool ptr_eq(const void *a, const void *b) { return a == b; }

static uint32_t ptr_hash(const void *p) {
  uintptr_t v = (uintptr_t)p;
  return (uint32_t)(v ^ (v >> 16) ^ (sizeof(v) > 4 ? (uint32_t)(v >> 32) : 0));
}

/* Return the identity element for a reduction op */
static double reduce_identity(PolyOps op) {
  switch (op) {
    case POLY_OP_ADD: return 0.0;
    case POLY_OP_MUL: return 1.0;
    case POLY_OP_MAX: return -__builtin_inf();
    default:
      fprintf(stderr, "polygrad: sched: unsupported reduce op %s\n", poly_op_name(op));
      return 0.0;
  }
}

/* ── Scheduling context ──────────────────────────────────────────────── */

typedef struct {
  PolyUOp *acc;
  PolyUOp *dep;
} RealizedScalarReduce;

typedef struct {
  PolyCtx *ctx;
  PolyMap *buf_to_param;   /* BUFFER UOp* → PARAM UOp* */
  PolyMap *shape_cache;    /* UOp* → PolyShape* (heap-allocated) */
  PolyMap *lower_cache;    /* UOp* keyed by (uop, ranges_hash) → lowered UOp* */
  PolyMap *scalar_reduce_cache; /* REDUCE_AXIS UOp* -> RealizedScalarReduce* */
  PolyUOp *replace_from;   /* optional substitution during lowering */
  PolyUOp *replace_to;
  int next_range_id;      /* unique RANGE arg ids across scheduled kernels */
  int n_params;
} SchedCtx;

/* Get cached shape, or compute and cache it */
static PolyShape sched_shape(SchedCtx *sctx, PolyUOp *u) {
  /* Check cache */
  PolyShape *cached = poly_map_get(sctx->shape_cache, ptr_hash(u), u, ptr_eq);
  if (cached) return *cached;

  /* Compute */
  PolyShape s = poly_uop_shape(sctx->ctx, u);
  PolyShape *stored = malloc(sizeof(PolyShape));
  *stored = s;
  poly_map_set(sctx->shape_cache, ptr_hash(u), u, stored, ptr_eq);
  return s;
}

/* ── Flat index computation ──────────────────────────────────────────── */

/* Build UOp expression: ranges[0]*stride[0] + ranges[1]*stride[1] + ... */
static PolyUOp *compute_flat_index(PolyCtx *ctx, PolyUOp **ranges, int ndim,
                                   PolyShape shape) {
  if (ndim == 0) {
    return poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  }

  /* For 1D, flat index is just the range variable */
  if (ndim == 1) return ranges[0];

  /* Multi-dimensional: compute strides and accumulate */
  int64_t strides[POLY_MAX_DIMS];
  strides[ndim - 1] = 1;
  for (int i = ndim - 2; i >= 0; i--)
    strides[i] = strides[i + 1] * shape.dims[i + 1];

  PolyUOp *flat = NULL;
  for (int i = 0; i < ndim; i++) {
    PolyUOp *term;
    if (strides[i] == 1) {
      term = ranges[i];
    } else {
      PolyUOp *s = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(strides[i]));
      term = poly_uop2(ctx, POLY_OP_MUL, POLY_INT32, ranges[i], s, poly_arg_none());
    }
    if (flat == NULL) {
      flat = term;
    } else {
      flat = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, flat, term, poly_arg_none());
    }
  }
  return flat;
}

/* ── Reshape index transform ─────────────────────────────────────────── */

/* Given output ranges for out_shape, compute input ranges for in_shape.
 * Method: flatten to linear index, then decompose via div/mod. */
static void reshape_indices(PolyCtx *ctx,
                            PolyUOp **out_ranges, int out_ndim, PolyShape out_shape,
                            PolyUOp **in_ranges,  int in_ndim,  PolyShape in_shape) {
  /* Compute combined flat index from output ranges */
  PolyUOp *combined = compute_flat_index(ctx, out_ranges, out_ndim, out_shape);

  /* Decompose into input dimension ranges */
  int64_t in_stride = 1;
  for (int j = in_ndim - 1; j >= 0; j--) {
    PolyUOp *dim_val = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(in_shape.dims[j]));

    PolyUOp *shifted;
    if (in_stride == 1) {
      shifted = combined;
    } else {
      PolyUOp *s = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(in_stride));
      shifted = poly_uop2(ctx, POLY_OP_IDIV, POLY_INT32, combined, s, poly_arg_none());
    }

    in_ranges[j] = poly_uop2(ctx, POLY_OP_MOD, POLY_INT32, shifted, dim_val, poly_arg_none());
    in_stride *= in_shape.dims[j];
  }
}

static int alloc_range_id(SchedCtx *sctx) {
  return sctx->next_range_id++;
}

/* STORE with optional dependency sources.
 * Extra sources are ignored by renderers but force topological dependence
 * so the store remains inside the intended loop nest. */
static PolyUOp *store_with_deps(PolyCtx *ctx, PolyUOp *dst, PolyUOp *val,
                                PolyUOp **deps, int n_deps) {
  if (n_deps <= 0) return poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, dst, val, poly_arg_none());
  PolyUOp *src[2 + POLY_MAX_DIMS];
  int n_src = 0;
  src[n_src++] = dst;
  src[n_src++] = val;
  for (int i = 0; i < n_deps && i < POLY_MAX_DIMS; i++) {
    if (deps[i]) src[n_src++] = deps[i];
  }
  return poly_uop(ctx, POLY_OP_STORE, POLY_VOID, src, n_src, poly_arg_none());
}

/* Binary op with optional dependency sources.
 * Extra sources are dependency-only; renderers consume the first two. */
static PolyUOp *binary_with_deps(PolyCtx *ctx, PolyOps op, PolyDType dtype,
                                 PolyUOp *lhs, PolyUOp *rhs,
                                 PolyUOp **deps, int n_deps) {
  if (n_deps <= 0) return poly_uop2(ctx, op, dtype, lhs, rhs, poly_arg_none());
  PolyUOp *src[2 + POLY_MAX_DIMS];
  int n_src = 0;
  src[n_src++] = lhs;
  src[n_src++] = rhs;
  for (int i = 0; i < n_deps && i < POLY_MAX_DIMS; i++) {
    if (deps[i]) src[n_src++] = deps[i];
  }
  return poly_uop(ctx, op, dtype, src, n_src, poly_arg_none());
}

/* Find a single unique REDUCE_AXIS node in a value graph.
 * Returns false if multiple distinct reductions are found. */
static bool find_single_reduce(PolyUOp *u, PolyUOp **found) {
  if (u->op == POLY_OP_REDUCE_AXIS) {
    if (*found && *found != u) return false;
    *found = u;
  }
  for (int i = 0; i < u->n_src; i++)
    if (!find_single_reduce(u->src[i], found)) return false;
  return true;
}

static PolyUOp *lower_uop(SchedCtx *sctx, PolyUOp *u,
                          PolyUOp **ranges, int n_ranges);

/* Materialize a scalar REDUCE_AXIS once and cache the result for reuse
 * across multiple scheduled STOREs. */
static bool realize_scalar_reduce(SchedCtx *sctx, PolyUOp *reduce_uop,
                                  PolyUOp **acc_out, PolyUOp **dep_out) {
  PolyCtx *ctx = sctx->ctx;
  RealizedScalarReduce *cached = poly_map_get(
    sctx->scalar_reduce_cache, ptr_hash(reduce_uop), reduce_uop, ptr_eq);
  if (cached) {
    *acc_out = cached->acc;
    *dep_out = cached->dep;
    return true;
  }

  if (reduce_uop->op != POLY_OP_REDUCE_AXIS ||
      reduce_uop->arg.kind != POLY_ARG_REDUCE_AXIS) {
    return false;
  }

  PolyShape rshape = sched_shape(sctx, reduce_uop);
  if (rshape.ndim < 0 || poly_shape_numel(rshape) != 1) return false;

  PolyOps reduce_op = reduce_uop->arg.reduce_axis.op;
  PolyShape in_shape = sched_shape(sctx, reduce_uop->src[0]);
  if (in_shape.ndim < 0) return false;

  double ident = reduce_identity(reduce_op);
  PolyUOp *acc = poly_uop0(ctx, POLY_OP_DEFINE_LOCAL, reduce_uop->dtype,
                           poly_arg_float(ident));

  PolyUOp *rranges[POLY_MAX_DIMS];
  for (int i = 0; i < in_shape.ndim; i++) {
    PolyUOp *bnd = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32,
                             poly_arg_int(in_shape.dims[i]));
    rranges[i] = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bnd,
                            poly_arg_range(alloc_range_id(sctx), POLY_AXIS_LOOP));
  }

  PolyUOp *rval = lower_uop(sctx, reduce_uop->src[0], rranges, in_shape.ndim);
  if (!rval) return false;

  PolyUOp *racc = binary_with_deps(ctx, reduce_op, reduce_uop->dtype,
                                    acc, rval, rranges, in_shape.ndim);
  PolyUOp *rstore = store_with_deps(ctx, acc, racc, rranges, in_shape.ndim);
  PolyUOp *rchain = rstore;
  for (int i = in_shape.ndim - 1; i >= 0; i--) {
    PolyUOp *end_src[2] = { rchain, rranges[i] };
    rchain = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  }

  RealizedScalarReduce *entry = malloc(sizeof(RealizedScalarReduce));
  entry->acc = acc;
  entry->dep = rchain;
  poly_map_set(sctx->scalar_reduce_cache, ptr_hash(reduce_uop), reduce_uop, entry, ptr_eq);

  *acc_out = acc;
  *dep_out = rchain;
  return true;
}

/* ── Recursive lowering ──────────────────────────────────────────────── */

/* Lower a tensor-level UOp into kernel-level IR, given the current
 * set of range variables representing "which position we're computing". */
static PolyUOp *lower_uop(SchedCtx *sctx, PolyUOp *u,
                          PolyUOp **ranges, int n_ranges) {
  PolyCtx *ctx = sctx->ctx;

  /* Optional substitution hook for schedule-time realizes. */
  if (sctx->replace_from && u == sctx->replace_from) return sctx->replace_to;

  /* BUFFER → PARAM + flat_index + INDEX + LOAD */
  if (u->op == POLY_OP_BUFFER) {
    PolyUOp *param = poly_map_get(sctx->buf_to_param, ptr_hash(u), u, ptr_eq);
    if (!param) {
      fprintf(stderr, "polygrad: sched: unknown buffer\n");
      return NULL;
    }
    PolyShape buf_shape = sched_shape(sctx, u);
    PolyUOp *flat = compute_flat_index(ctx, ranges, n_ranges, buf_shape);
    PolyUOp *idx = poly_uop2(ctx, POLY_OP_INDEX, param->dtype, param, flat, poly_arg_none());
    PolyDType scalar = poly_dtype_scalar(u->dtype);
    return poly_uop1(ctx, POLY_OP_LOAD, scalar, idx, poly_arg_none());
  }

  /* CONST: broadcasts — just return as-is */
  if (u->op == POLY_OP_CONST) {
    return u;
  }

  /* CAST/BITCAST: lower source, apply cast */
  if (u->op == POLY_OP_CAST || u->op == POLY_OP_BITCAST) {
    PolyUOp *lowered_src = lower_uop(sctx, u->src[0], ranges, n_ranges);
    if (!lowered_src) return NULL;
    return poly_uop1(ctx, u->op, u->dtype, lowered_src, u->arg);
  }

  /* EXPAND: zero out expanded dims, recurse into source */
  if (u->op == POLY_OP_EXPAND) {
    PolyShape in_shape = sched_shape(sctx, u->src[0]);
    PolyShape out_shape = sched_shape(sctx, u);
    if (in_shape.ndim < 0 || out_shape.ndim < 0) return NULL;

    /* Zero-init: in_shape.ndim may exceed n_ranges when shapes differ */
    PolyUOp *in_ranges[POLY_MAX_DIMS] = {0};
    for (int i = 0; i < in_shape.ndim; i++) {
      if (in_shape.dims[i] == 1 && out_shape.dims[i] != 1) {
        /* Expanded dim: always index 0 */
        in_ranges[i] = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
      } else if (i < n_ranges) {
        in_ranges[i] = ranges[i];
      }
    }
    return lower_uop(sctx, u->src[0], in_ranges, in_shape.ndim);
  }

  /* RESHAPE: transform indices via flatten + decompose */
  if (u->op == POLY_OP_RESHAPE) {
    PolyShape in_shape = sched_shape(sctx, u->src[0]);
    PolyShape out_shape = sched_shape(sctx, u);
    if (in_shape.ndim < 0 || out_shape.ndim < 0) return NULL;

    /* Zero-init: reshape_indices may leave gaps when ndims differ */
    PolyUOp *in_ranges[POLY_MAX_DIMS] = {0};
    reshape_indices(ctx, ranges, n_ranges, out_shape,
                    in_ranges, in_shape.ndim, in_shape);
    return lower_uop(sctx, u->src[0], in_ranges, in_shape.ndim);
  }

  /* PERMUTE: reorder range variables by permutation */
  if (u->op == POLY_OP_PERMUTE && u->arg.kind == POLY_ARG_INT_TUPLE) {
    int n = u->arg.int_tuple.n;
    /* Zero-init + bounds check: permutation may reference more dims than ranges */
    PolyUOp *in_ranges[POLY_MAX_DIMS] = {0};
    for (int i = 0; i < n && i < n_ranges; i++)
      in_ranges[u->arg.int_tuple.vals[i]] = ranges[i];
    return lower_uop(sctx, u->src[0], in_ranges, n);
  }

  /* SHRINK: offset range by slice start */
  if (u->op == POLY_OP_SHRINK && u->arg.kind == POLY_ARG_PAIR_TUPLE) {
    int n = u->arg.pair_tuple.n;
    /* Zero-init + bounds check: pair_tuple.n may exceed n_ranges */
    PolyUOp *in_ranges[POLY_MAX_DIMS] = {0};
    for (int i = 0; i < n && i < n_ranges; i++) {
      int64_t start = u->arg.pair_tuple.pairs[i][0];
      if (start == 0) {
        in_ranges[i] = ranges[i];
      } else {
        PolyUOp *off = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(start));
        in_ranges[i] = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32,
                                  ranges[i], off, poly_arg_none());
      }
    }
    return lower_uop(sctx, u->src[0], in_ranges, n);
  }

  /* FLIP: reverse range for flipped axes */
  if (u->op == POLY_OP_FLIP && u->arg.kind == POLY_ARG_INT_TUPLE) {
    PolyShape in_shape = sched_shape(sctx, u->src[0]);
    if (in_shape.ndim < 0) return NULL;
    bool flipped[POLY_MAX_DIMS] = {false};
    for (int i = 0; i < u->arg.int_tuple.n; i++) {
      int ax = (int)u->arg.int_tuple.vals[i];
      if (ax >= 0 && ax < in_shape.ndim) flipped[ax] = true;
    }
    /* Zero-init + bounds check: in_shape.ndim may exceed n_ranges */
    PolyUOp *in_ranges[POLY_MAX_DIMS] = {0};
    for (int i = 0; i < in_shape.ndim && i < n_ranges; i++) {
      if (flipped[i]) {
        PolyUOp *max_idx = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32,
                                     poly_arg_int(in_shape.dims[i] - 1));
        in_ranges[i] = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, max_idx,
            poly_uop1(ctx, POLY_OP_NEG, POLY_INT32, ranges[i], poly_arg_none()),
            poly_arg_none());
      } else {
        in_ranges[i] = ranges[i];
      }
    }
    return lower_uop(sctx, u->src[0], in_ranges, in_shape.ndim);
  }

  /* PAD: offset + bounds check + clamped load + WHERE */
  if (u->op == POLY_OP_PAD && u->arg.kind == POLY_ARG_PAIR_TUPLE) {
    PolyShape in_shape = sched_shape(sctx, u->src[0]);
    if (in_shape.ndim < 0) return NULL;
    int n = u->arg.pair_tuple.n;

    /* Zero-init + bounds check: pair_tuple.n may exceed n_ranges */
    PolyUOp *in_ranges[POLY_MAX_DIMS] = {0};
    PolyUOp *valid = NULL;

    for (int i = 0; i < n && i < n_ranges; i++) {
      int64_t begin = u->arg.pair_tuple.pairs[i][0];
      if (begin == 0) {
        in_ranges[i] = ranges[i];
      } else {
        PolyUOp *off = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(begin));
        in_ranges[i] = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, ranges[i],
            poly_uop1(ctx, POLY_OP_NEG, POLY_INT32, off, poly_arg_none()),
            poly_arg_none());
      }
      /* valid_i = NOT(in_idx < 0) AND (in_idx < in_dim) */
      PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
      PolyUOp *dim = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32,
                               poly_arg_int(in_shape.dims[i]));
      PolyUOp *ge_zero = poly_uop1(ctx, POLY_OP_NEG, POLY_BOOL,
          poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, in_ranges[i], zero,
                    poly_arg_none()), poly_arg_none());
      PolyUOp *lt_dim = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL,
                                  in_ranges[i], dim, poly_arg_none());
      PolyUOp *dv = poly_uop2(ctx, POLY_OP_AND, POLY_BOOL,
                              ge_zero, lt_dim, poly_arg_none());
      valid = valid ? poly_uop2(ctx, POLY_OP_AND, POLY_BOOL,
                                valid, dv, poly_arg_none()) : dv;
    }

    /* Clamp indices to [0, dim-1] for safe memory access */
    PolyUOp *clamped[POLY_MAX_DIMS];
    for (int i = 0; i < n; i++) {
      PolyUOp *z = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
      PolyUOp *m = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32,
                             poly_arg_int(in_shape.dims[i] - 1));
      PolyUOp *lt_z = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL,
                                in_ranges[i], z, poly_arg_none());
      PolyUOp *cl = poly_uop(ctx, POLY_OP_WHERE, POLY_INT32,
          (PolyUOp*[]){lt_z, z, in_ranges[i]}, 3, poly_arg_none());
      PolyUOp *gt_m = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL,
                                m, cl, poly_arg_none());
      clamped[i] = poly_uop(ctx, POLY_OP_WHERE, POLY_INT32,
          (PolyUOp*[]){gt_m, m, cl}, 3, poly_arg_none());
    }

    PolyUOp *loaded = lower_uop(sctx, u->src[0], clamped, n);
    if (!loaded) return NULL;
    PolyUOp *pad_val = poly_uop0(ctx, POLY_OP_CONST, u->dtype, poly_arg_float(0.0));
    PolyUOp *where_src[3] = {valid, loaded, pad_val};
    return poly_uop(ctx, POLY_OP_WHERE, u->dtype, where_src, 3, poly_arg_none());
  }

  /* ALU ops: elementwise — lower each source with same ranges */
  if (poly_opset_has(POLY_GROUP_ALU, u->op)) {
    PolyUOp *lowered_srcs[3];
    for (int i = 0; i < u->n_src; i++) {
      /* Scalars (CONST) don't need ranges */
      PolyShape si = sched_shape(sctx, u->src[i]);
      if (si.ndim == 0) {
        /* Scalar source — broadcast, no indexing needed */
        lowered_srcs[i] = lower_uop(sctx, u->src[i], ranges, n_ranges);
      } else {
        lowered_srcs[i] = lower_uop(sctx, u->src[i], ranges, n_ranges);
      }
      if (!lowered_srcs[i]) return NULL;
    }
    return poly_uop(ctx, u->op, u->dtype, lowered_srcs, u->n_src, u->arg);
  }

  fprintf(stderr, "polygrad: sched: unsupported op %s in lower_uop\n", poly_op_name(u->op));
  return NULL;
}

/* ── Schedule a single STORE ─────────────────────────────────────────── */

/* Schedule one STORE(buffer, value) into kernel-level IR.
 * Returns the END node (wrapping STORE and all RANGEs). */
static PolyUOp *schedule_store(SchedCtx *sctx, PolyUOp *store_uop) {
  PolyCtx *ctx = sctx->ctx;

  if (store_uop->op != POLY_OP_STORE || store_uop->n_src < 2) {
    fprintf(stderr, "polygrad: sched: expected STORE with 2 sources\n");
    return NULL;
  }

  PolyUOp *out_buf = store_uop->src[0];  /* output BUFFER */
  PolyUOp *value   = store_uop->src[1];  /* value expression */

  /* Get output shape from the value */
  PolyShape out_shape = sched_shape(sctx, value);
  if (out_shape.ndim < 0) {
    fprintf(stderr, "polygrad: sched: cannot determine output shape\n");
    return NULL;
  }

  /* For scalar outputs, use the buffer shape */
  if (out_shape.ndim == 0) {
    out_shape = sched_shape(sctx, out_buf);
    if (out_shape.ndim < 0) {
      fprintf(stderr, "polygrad: sched: cannot determine buffer shape\n");
      return NULL;
    }
  }

  /* Look up output buffer PARAM */
  PolyUOp *out_param = poly_map_get(sctx->buf_to_param, ptr_hash(out_buf), out_buf, ptr_eq);
  if (!out_param) {
    fprintf(stderr, "polygrad: sched: unknown output buffer\n");
    return NULL;
  }

  /* ── Chained scalar-reduce → elementwise path ─────────────────────── */
  /* Handles patterns like add(reshape(sum(a), ()), b) by computing
   * the scalar reduction once, then using it in a second loop nest. */
  PolyUOp *nested_reduce = NULL;
  if (find_single_reduce(value, &nested_reduce) &&
      nested_reduce && nested_reduce != value &&
      nested_reduce->arg.kind == POLY_ARG_REDUCE_AXIS) {
    PolyShape rshape = sched_shape(sctx, nested_reduce);
    if (rshape.ndim >= 0 && poly_shape_numel(rshape) == 1) {
      PolyUOp *acc = NULL, *rchain = NULL;
      if (!realize_scalar_reduce(sctx, nested_reduce, &acc, &rchain)) {
        fprintf(stderr, "polygrad: sched: failed to realize nested scalar reduce\n");
        return NULL;
      }

      /* Elementwise loop nest using the reduced scalar */
      int ndim = out_shape.ndim;
      PolyUOp *eranges[POLY_MAX_DIMS];
      for (int i = 0; i < ndim; i++) {
        PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(out_shape.dims[i]));
        eranges[i] = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound,
                                poly_arg_range(alloc_range_id(sctx), POLY_AXIS_LOOP));
      }

      PolyUOp *old_from = sctx->replace_from, *old_to = sctx->replace_to;
      sctx->replace_from = nested_reduce;
      sctx->replace_to = acc;
      PolyUOp *eval = lower_uop(sctx, value, eranges, ndim);
      sctx->replace_from = old_from;
      sctx->replace_to = old_to;
      if (!eval) return NULL;

      PolyUOp *out_flat = (ndim == 0)
        ? poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0))
        : compute_flat_index(ctx, eranges, ndim, out_shape);
      PolyUOp *out_idx = poly_uop2(ctx, POLY_OP_INDEX, out_param->dtype,
                                   out_param, out_flat, poly_arg_none());

      /* 3rd source carries dependency on reduction chain. */
      PolyUOp *s_src[3] = { out_idx, eval, rchain };
      PolyUOp *estore = poly_uop(ctx, POLY_OP_STORE, POLY_VOID, s_src, 3, poly_arg_none());

      PolyUOp *echain = estore;
      for (int i = ndim - 1; i >= 0; i--) {
        PolyUOp *end_src[2] = { echain, eranges[i] };
        echain = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
      }
      return echain;
    }

    /* General non-scalar nested reduce:
     * compute the reduction for the current output coordinate, then
     * substitute nested_reduce with the per-coordinate accumulator.
     *
     * Current supported mapping: output shape keeps reduce output ndim and
     * only expands dims that are 1 in the reduce output (reduce->expand chain). */
    if (rshape.ndim > 0 && poly_shape_numel(rshape) > 1) {
      PolyOps reduce_op = nested_reduce->arg.reduce_axis.op;
      int n_rax = nested_reduce->arg.reduce_axis.n;
      int64_t *rax = nested_reduce->arg.reduce_axis.axes;
      PolyShape in_shape = sched_shape(sctx, nested_reduce->src[0]);
      if (in_shape.ndim < 0) {
        fprintf(stderr, "polygrad: sched: cannot determine nested reduce input shape\n");
        return NULL;
      }

      /* Classify reduced axes (needed for both map_ok checks) */
      bool is_reduced[POLY_MAX_DIMS] = {false};
      for (int i = 0; i < n_rax; i++) {
        int ax = (int)rax[i];
        if (ax >= 0 && ax < in_shape.ndim) is_reduced[ax] = true;
      }

      /* Check: same-ndim case (reduce→expand chain) */
      bool map_ok = (in_shape.ndim == rshape.ndim) && (out_shape.ndim == rshape.ndim);
      bool squeeze = false;
      if (map_ok) {
        for (int i = 0; i < rshape.ndim; i++) {
          if (rshape.dims[i] != 1 && rshape.dims[i] != out_shape.dims[i]) {
            map_ok = false;
            break;
          }
        }
      }

      /* Check: squeezed case — RESHAPE removes reduced (size-1) dims */
      if (!map_ok && in_shape.ndim == rshape.ndim && out_shape.ndim < rshape.ndim) {
        int n_non_reduced = 0;
        for (int i = 0; i < rshape.ndim; i++)
          if (!is_reduced[i]) n_non_reduced++;
        if (out_shape.ndim == n_non_reduced) {
          squeeze = true;
          map_ok = true;
          int oi = 0;
          for (int i = 0; i < rshape.ndim && map_ok; i++) {
            if (!is_reduced[i]) {
              if (oi >= out_shape.ndim || rshape.dims[i] != out_shape.dims[oi])
                map_ok = false;
              oi++;
            }
          }
        }
      }

      if (map_ok) {
        /* Output loops */
        int ndim = out_shape.ndim;
        PolyUOp *out_ranges[POLY_MAX_DIMS];
        for (int i = 0; i < ndim; i++) {
          PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(out_shape.dims[i]));
          out_ranges[i] = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound,
                                     poly_arg_range(alloc_range_id(sctx), POLY_AXIS_LOOP));
        }

        /* For non-reduced axes, coordinate comes from output loop.
         * In squeeze mode, output ranges map sequentially to non-reduced dims.
         * In same-ndim mode, output ranges map 1:1 (with 0 for broadcast).
         * Zero-init: reduced dims are filled later in the inner-loop block. */
        PolyUOp *all_ranges[POLY_MAX_DIMS] = {0};
        PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
        if (squeeze) {
          int oi = 0;
          for (int i = 0; i < in_shape.ndim; i++) {
            /* Guard oi < ndim: n_non_reduced == ndim is validated above,
             * but the analyzer can't prove it across the loop. */
            if (!is_reduced[i] && oi < ndim)
              all_ranges[i] = out_ranges[oi++];
          }
        } else {
          for (int i = 0; i < in_shape.ndim; i++) {
            if (!is_reduced[i])
              all_ranges[i] = (rshape.dims[i] == 1) ? zero : out_ranges[i];
          }
        }

        /* Accumulator lives inside output loops. */
        double ident = reduce_identity(reduce_op);
        PolyUOp *acc = poly_uop(ctx, POLY_OP_DEFINE_LOCAL, nested_reduce->dtype,
                                out_ranges, ndim, poly_arg_float(ident));

        /* Inner reduction loops */
        for (int i = 0; i < in_shape.ndim; i++) {
          if (is_reduced[i]) {
            PolyUOp *bnd = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32,
                                     poly_arg_int(in_shape.dims[i]));
            all_ranges[i] = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bnd,
                                       poly_arg_range(alloc_range_id(sctx), POLY_AXIS_LOOP));
          }
        }

        PolyUOp *loaded = lower_uop(sctx, nested_reduce->src[0], all_ranges, in_shape.ndim);
        if (!loaded) return NULL;
        PolyUOp *alu = binary_with_deps(ctx, reduce_op, nested_reduce->dtype,
                                         acc, loaded, all_ranges, in_shape.ndim);
        PolyUOp *acc_store = store_with_deps(ctx, acc, alu, all_ranges, in_shape.ndim);

        PolyUOp *rchain = acc_store;
        for (int i = in_shape.ndim - 1; i >= 0; i--) {
          if (is_reduced[i]) {
            PolyUOp *end_src[2] = { rchain, all_ranges[i] };
            rchain = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
          }
        }

        /* Lower final expression with nested reduce replaced by acc. */
        PolyUOp *old_from = sctx->replace_from, *old_to = sctx->replace_to;
        sctx->replace_from = nested_reduce;
        sctx->replace_to = acc;
        PolyUOp *eval = lower_uop(sctx, value, out_ranges, ndim);
        sctx->replace_from = old_from;
        sctx->replace_to = old_to;
        if (!eval) return NULL;

        PolyUOp *out_flat = compute_flat_index(ctx, out_ranges, ndim, out_shape);
        PolyUOp *out_idx = poly_uop2(ctx, POLY_OP_INDEX, out_param->dtype,
                                     out_param, out_flat, poly_arg_none());

        /* Dependency source keeps out_store after reduction loop. */
        PolyUOp *s_src[3] = { out_idx, eval, rchain };
        PolyUOp *out_store = poly_uop(ctx, POLY_OP_STORE, POLY_VOID,
                                      s_src, 3, poly_arg_none());

        PolyUOp *echain = out_store;
        for (int i = ndim - 1; i >= 0; i--) {
          PolyUOp *end_src[2] = { echain, out_ranges[i] };
          echain = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
        }
        return echain;
      }
    }
  }

  /* ── REDUCE_AXIS path ──────────────────────────────────────────────── */
  if (value->op == POLY_OP_REDUCE_AXIS &&
      value->arg.kind == POLY_ARG_REDUCE_AXIS) {
    PolyOps reduce_op = value->arg.reduce_axis.op;
    int n_rax = value->arg.reduce_axis.n;
    int64_t *rax = value->arg.reduce_axis.axes;

    /* Get input shape (the source of the reduce) */
    PolyShape in_shape = sched_shape(sctx, value->src[0]);
    if (in_shape.ndim < 0) {
      fprintf(stderr, "polygrad: sched: cannot determine reduce input shape\n");
      return NULL;
    }

    /* Classify axes: reduced vs non-reduced */
    bool is_reduced[POLY_MAX_DIMS] = {false};
    for (int i = 0; i < n_rax; i++) {
      int ax = (int)rax[i];
      if (ax >= 0 && ax < in_shape.ndim) is_reduced[ax] = true;
    }

    /* Create OUTER RANGE loops (non-reduced dims) */
    PolyUOp *all_ranges[POLY_MAX_DIMS];
    PolyUOp *outer_ranges[POLY_MAX_DIMS];
    int64_t outer_dims[POLY_MAX_DIMS];
    int n_outer = 0;
    for (int i = 0; i < in_shape.ndim; i++) {
      if (!is_reduced[i]) {
        PolyUOp *bnd = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32,
                                 poly_arg_int(in_shape.dims[i]));
        all_ranges[i] = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32,
                                   bnd, poly_arg_range(alloc_range_id(sctx), POLY_AXIS_LOOP));
        outer_ranges[n_outer] = all_ranges[i];
        outer_dims[n_outer] = in_shape.dims[i];
        n_outer++;
      }
    }

    /* DEFINE_LOCAL: accumulator with identity value.
     * Give outer ranges as sources so the linearizer places it
     * inside the outer loop (not hoisted before it). */
    double ident = reduce_identity(reduce_op);
    PolyUOp *acc = poly_uop(ctx, POLY_OP_DEFINE_LOCAL, value->dtype,
                            outer_ranges, n_outer,
                            poly_arg_float(ident));

    /* Create INNER RANGE loops (reduced dims) */
    for (int i = 0; i < in_shape.ndim; i++) {
      if (is_reduced[i]) {
        PolyUOp *bnd = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32,
                                 poly_arg_int(in_shape.dims[i]));
        all_ranges[i] = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32,
                                   bnd, poly_arg_range(alloc_range_id(sctx), POLY_AXIS_LOOP));
      }
    }

    /* Lower the reduce source using ALL ranges (outer + inner) */
    PolyUOp *loaded = lower_uop(sctx, value->src[0], all_ranges, in_shape.ndim);
    if (!loaded) return NULL;

    /* Accumulate: alu = reduce_op(acc, loaded) */
    PolyUOp *alu = binary_with_deps(ctx, reduce_op, value->dtype,
                                     acc, loaded, all_ranges, in_shape.ndim);

    /* Store to accumulator */
    PolyUOp *acc_store = store_with_deps(ctx, acc, alu, all_ranges, in_shape.ndim);

    /* Close inner ENDs (reduced dims, innermost first) */
    PolyUOp *current = acc_store;
    for (int i = in_shape.ndim - 1; i >= 0; i--) {
      if (is_reduced[i]) {
        PolyUOp *end_src[2] = { current, all_ranges[i] };
        current = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
      }
    }

    /* Output flat index from non-reduced ranges */
    PolyUOp *out_flat;
    if (n_outer == 0) {
      /* Full reduction: scalar output at index 0 */
      out_flat = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
    } else {
      PolyShape nr_shape = { outer_dims, n_outer };
      out_flat = compute_flat_index(ctx, outer_ranges, n_outer, nr_shape);
    }

    PolyUOp *out_idx = poly_uop2(ctx, POLY_OP_INDEX, out_param->dtype,
                                 out_param, out_flat, poly_arg_none());

    /* Store acc to output.  3rd source = inner END chain for dependency. */
    PolyUOp *dep_src[3] = { out_idx, acc, current };
    PolyUOp *out_store = poly_uop(ctx, POLY_OP_STORE, POLY_VOID,
                                  dep_src, 3, poly_arg_none());

    /* Close outer ENDs */
    current = out_store;
    for (int i = in_shape.ndim - 1; i >= 0; i--) {
      if (!is_reduced[i]) {
        PolyUOp *end_src[2] = { current, all_ranges[i] };
        current = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
      }
    }

    return current;
  }

  /* ── Elementwise path ──────────────────────────────────────────────── */
  int ndim = out_shape.ndim;

  /* Create RANGE UOps (one per dimension) */
  PolyUOp *ranges[POLY_MAX_DIMS];
  for (int i = 0; i < ndim; i++) {
    PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(out_shape.dims[i]));
    ranges[i] = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound,
                           poly_arg_range(alloc_range_id(sctx), POLY_AXIS_LOOP));
  }

  /* Lower the value expression */
  PolyUOp *lowered_value = lower_uop(sctx, value, ranges, ndim);
  if (!lowered_value) return NULL;

  PolyUOp *out_flat = compute_flat_index(ctx, ranges, ndim, out_shape);
  PolyUOp *out_idx = poly_uop2(ctx, POLY_OP_INDEX, out_param->dtype, out_param, out_flat, poly_arg_none());
  PolyUOp *kernel_store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out_idx, lowered_value, poly_arg_none());

  /* Chain ENDs for all ranges (innermost first) */
  PolyUOp *current = kernel_store;
  for (int i = ndim - 1; i >= 0; i--) {
    PolyUOp *end_src[2] = { current, ranges[i] };
    current = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  }

  return current;
}

/* ── Collect all BUFFERs in a tensor graph ────────────────────────────── */

static void collect_buffers(SchedCtx *sctx, PolyUOp *root) {
  int n_topo;
  PolyUOp **topo = poly_toposort(sctx->ctx, root, &n_topo);

  for (int i = 0; i < n_topo; i++) {
    if (topo[i]->op == POLY_OP_BUFFER) {
      /* Check if already assigned */
      if (poly_map_get(sctx->buf_to_param, ptr_hash(topo[i]), topo[i], ptr_eq))
        continue;

      /* Create PARAM for this buffer */
      PolyDType scalar = poly_dtype_scalar(topo[i]->dtype);
      PolyDType ptr_dt = poly_dtype_ptr(scalar, -1, POLY_ADDR_GLOBAL);
      PolyUOp *param = poly_uop0(sctx->ctx, POLY_OP_PARAM, ptr_dt,
                                poly_arg_int(sctx->n_params));
      poly_map_set(sctx->buf_to_param, ptr_hash(topo[i]), topo[i], param, ptr_eq);
      sctx->n_params++;
    }
  }
}

/* ── Public API ───────────────────────────────────────────────────────── */

PolyUOp *poly_schedule(PolyCtx *ctx, PolyUOp *tensor_sink) {
  PolyScheduleResult sr = poly_schedule_v2(ctx, tensor_sink);
  if (sr.n_kernels < 1) { poly_schedule_result_free(&sr); return NULL; }
  /* For single-kernel API, return the last kernel (consumer).
   * Multi-kernel callers should use poly_schedule_v2() directly. */
  PolyUOp *kernel = sr.kernels[sr.n_kernels - 1];
  poly_schedule_result_free(&sr);
  return kernel;
}

/* ── Convenience constructors ─────────────────────────────────────────── */

static int poly_buffer_id = 0;

PolyUOp *poly_buffer(PolyCtx *ctx, PolyDType scalar_dtype, int64_t size) {
  /* Each buffer gets a UNIQUE source to prevent CSE merging (same as tinygrad) */
  int id = poly_buffer_id++;
  PolyUOp *unique = poly_uop0(ctx, POLY_OP_UNIQUE, POLY_VOID, poly_arg_int(id));
  return poly_uop1(ctx, POLY_OP_BUFFER, scalar_dtype, unique, poly_arg_int(size));
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

PolyUOp *poly_reduce_axis(PolyCtx *ctx, PolyOps reduce_op, PolyUOp *src,
                         int64_t *axes, int n_axes) {
  PolyArg arg;
  arg.kind = POLY_ARG_REDUCE_AXIS;
  arg.reduce_axis.op = reduce_op;
  arg.reduce_axis.axes = axes;
  arg.reduce_axis.n = n_axes;
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
