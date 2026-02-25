/*
 * rangeify.c — Range-propagation scheduling pipeline
 *
 * Ports tinygrad's rangeify.py + indexing.py to C11.
 * Phase 1: consumer map + realize-point detection.
 * Phase 2: range propagation.
 *
 * Reference: tinygrad schedule/rangeify.py, schedule/indexing.py
 */

#include "rangeify.h"
#include "indexing.h"
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifndef POLY_RANGEIFY_DEBUG
#define POLY_RANGEIFY_DEBUG 0
#endif

#define RDBG(...) do { if (POLY_RANGEIFY_DEBUG) fprintf(stderr, __VA_ARGS__); } while (0)

/* Global rangeify stats (debug/parity observability). */
static PolyRangeifyStats g_rangeify_stats;

void poly_rangeify_stats_reset(void) {
  memset(&g_rangeify_stats, 0, sizeof(g_rangeify_stats));
}

PolyRangeifyStats poly_rangeify_stats_get(void) {
  return g_rangeify_stats;
}

/* ── Pointer hashing/equality (shared with sched.c) ──────────────────── */

static bool ptr_eq(const void *a, const void *b) { return a == b; }

static uint32_t ptr_hash(const void *p) {
  uintptr_t v = (uintptr_t)p;
  return (uint32_t)(v ^ (v >> 16) ^ (sizeof(v) > 4 ? (uint32_t)(v >> 32) : 0));
}

/* ── Consumer list helpers ───────────────────────────────────────────── */

static PolyConsumerList *consumer_list_new(void) {
  PolyConsumerList *cl = malloc(sizeof(PolyConsumerList));
  cl->items = NULL;
  cl->count = 0;
  cl->cap = 0;
  return cl;
}

static void consumer_list_add(PolyConsumerList *cl, PolyUOp *consumer) {
  if (cl->count >= cl->cap) {
    cl->cap = cl->cap ? cl->cap * 2 : 4;
    cl->items = realloc(cl->items, cl->cap * sizeof(PolyUOp *));
  }
  cl->items[cl->count++] = consumer;
}

static void consumer_list_free(PolyConsumerList *cl) {
  free(cl->items);
  free(cl);
}

/* ── Consumer map ────────────────────────────────────────────────────── */

PolyMap *poly_consumer_map_build(PolyCtx *ctx, PolyUOp *sink) {
  PolyMap *cmap = poly_map_new(64);

  /* Toposort to get all UOps in dependency order */
  int n_uops;
  PolyUOp **topo = poly_toposort(ctx, sink, &n_uops);
  if (!topo) return cmap;

  /* Ensure every UOp has an entry (even if 0 consumers) */
  for (int i = 0; i < n_uops; i++) {
    uint32_t h = ptr_hash(topo[i]);
    if (!poly_map_get(cmap, h, topo[i], ptr_eq)) {
      poly_map_set(cmap, h, topo[i], consumer_list_new(), ptr_eq);
    }
  }

  /* For each UOp, register it as a consumer of each of its sources */
  for (int i = 0; i < n_uops; i++) {
    PolyUOp *u = topo[i];
    for (int j = 0; j < u->n_src; j++) {
      PolyUOp *src = u->src[j];
      uint32_t h = ptr_hash(src);
      PolyConsumerList *cl = poly_map_get(cmap, h, src, ptr_eq);
      if (cl) consumer_list_add(cl, u);
    }
  }

  return cmap;
}

PolyConsumerList *poly_consumer_map_get(PolyMap *cmap, PolyUOp *u) {
  return poly_map_get(cmap, ptr_hash(u), u, ptr_eq);
}

/* Cleanup helper for consumer map */
static void free_consumer_list_entry(const void *key, void *value, void *ud) {
  (void)key; (void)ud;
  consumer_list_free((PolyConsumerList *)value);
}

/* Cleanup helper for realize map */
static void free_realize_entry(const void *key, void *value, void *ud) {
  (void)key; (void)ud;
  PolyRealizeInfo *ri = value;
  free(ri->axes);
  free(ri);
}

/* Cleanup helper for range map entries */
static void free_range_entry(const void *key, void *value, void *ud) {
  (void)key; (void)ud;
  PolyRangeEntry *re = value;
  free(re->in_rngs);
  free(re->out_rngs);
  free(re);
}

/* Cleanup helper for shape cache entries */
static void free_shape_entry(const void *key, void *value, void *ud) {
  (void)key; (void)ud;
  PolyShape *s = value;
  if (s->ndim > 0 && s->dims) free(s->dims);
  free(s);
}

/* Cleanup helper for buffer_alt_rngs entries */
static void free_alt_rngs_entry(const void *key, void *value, void *ud) {
  (void)key; (void)ud;
  free(value);
}

static bool alt_entry_equals(PolyBufferAltRngs *alt, PolyUOp **rngs, int n) {
  if (!alt || !rngs || n <= 0) return false;
  for (int i = 0; i < alt->count; i++) {
    if (alt->lens[i] != n) continue;
    bool same = true;
    for (int d = 0; d < n; d++) {
      if (alt->rngs[i][d] != rngs[d]) { same = false; break; }
    }
    if (same) return true;
  }
  return false;
}

static void buffer_alt_add(PolyIndexingCtx *ictx, PolyUOp *buf,
                           PolyUOp **rngs, int n_rngs) {
  if (!ictx || !buf || !rngs || n_rngs <= 0 || n_rngs > POLY_MAX_DIMS) return;
  PolyBufferAltRngs *alt = poly_map_get(ictx->buffer_alt_rngs, ptr_hash(buf),
                                        buf, ptr_eq);
  if (!alt) {
    alt = malloc(sizeof(PolyBufferAltRngs));
    if (!alt) return;
    memset(alt, 0, sizeof(*alt));
    poly_map_set(ictx->buffer_alt_rngs, ptr_hash(buf), buf, alt, ptr_eq);
  }
  if (alt_entry_equals(alt, rngs, n_rngs)) return;
  if (alt->count >= POLY_MAX_ALT_RNGS) return;
  int ai = alt->count++;
  alt->lens[ai] = n_rngs;
  for (int d = 0; d < n_rngs; d++) alt->rngs[ai][d] = rngs[d];
}

/* Per-UOp ending-ranges list (tinygrad run_rangeify parity helper). */
typedef struct {
  PolyUOp **items;
  int count;
  int cap;
} PolyEndingRanges;

static PolyEndingRanges *ending_new(void) {
  PolyEndingRanges *er = malloc(sizeof(PolyEndingRanges));
  er->count = 0;
  er->cap = 4;
  er->items = malloc(er->cap * sizeof(PolyUOp *));
  return er;
}

static void ending_destroy(const void *key, void *value, void *ud) {
  (void)key; (void)ud;
  PolyEndingRanges *er = value;
  free(er->items);
  free(er);
}

static void ending_clear(PolyEndingRanges *er) { er->count = 0; }

static bool ending_contains(PolyEndingRanges *er, PolyUOp *r) {
  for (int i = 0; i < er->count; i++) if (er->items[i] == r) return true;
  return false;
}

static void ending_add(PolyEndingRanges *er, PolyUOp *r) {
  if (!r || ending_contains(er, r)) return;
  if (er->count >= er->cap) {
    er->cap *= 2;
    er->items = realloc(er->items, er->cap * sizeof(PolyUOp *));
  }
  er->items[er->count++] = r;
}

static PolyEndingRanges *ending_get_or_create(PolyMap *m, PolyUOp *key) {
  PolyEndingRanges *er = poly_map_get(m, ptr_hash(key), key, ptr_eq);
  if (er) return er;
  er = ending_new();
  poly_map_set(m, ptr_hash(key), key, er, ptr_eq);
  return er;
}

/* ── Indexing context ────────────────────────────────────────────────── */

PolyIndexingCtx *poly_indexing_ctx_new(PolyCtx *ctx) {
  PolyIndexingCtx *ictx = malloc(sizeof(PolyIndexingCtx));
  ictx->ctx = ctx;
  ictx->consumer_map = NULL;
  ictx->realize_map = poly_map_new(32);
  ictx->range_map = poly_map_new(64);
  ictx->shape_cache = poly_map_new(64);
  ictx->reduce_origin = poly_map_new(16);
  ictx->buffer_alt_rngs = poly_map_new(16);
  ictx->realized_to_bufferize = poly_map_new(32);
  ictx->bufferize_to_realized = poly_map_new(32);
  ictx->next_range_id = 0;
  ictx->add_buffer_indices = false;
  const char *mkb = getenv("POLY_MAX_KERNEL_BUFFERS");
  ictx->max_kernel_bufs = mkb ? atoi(mkb) : 0;
  return ictx;
}

void poly_indexing_ctx_destroy(PolyIndexingCtx *ictx) {
  if (ictx->consumer_map) {
    poly_map_foreach(ictx->consumer_map, free_consumer_list_entry, NULL);
    poly_map_destroy(ictx->consumer_map);
  }
  poly_map_foreach(ictx->range_map, free_range_entry, NULL);
  poly_map_destroy(ictx->range_map);
  poly_map_destroy(ictx->reduce_origin);
  poly_map_foreach(ictx->buffer_alt_rngs, free_alt_rngs_entry, NULL);
  poly_map_destroy(ictx->buffer_alt_rngs);
  poly_map_foreach(ictx->realize_map, free_realize_entry, NULL);
  poly_map_destroy(ictx->realize_map);
  poly_map_foreach(ictx->shape_cache, free_shape_entry, NULL);
  poly_map_destroy(ictx->shape_cache);
  if (ictx->realized_to_bufferize)
    poly_map_destroy(ictx->realized_to_bufferize);
  if (ictx->bufferize_to_realized)
    poly_map_destroy(ictx->bufferize_to_realized);
  free(ictx);
}

/* ── Realize map ─────────────────────────────────────────────────────── */

/*
 * Realize-point detection. Ports tinygrad's pm_generate_realize_map.
 *
 * An op is "realized" if it must become a kernel boundary — its output
 * goes to a buffer rather than being fused into a consumer's kernel.
 *
 * Rules (simplified for CPU-only, no OUTER ranges yet):
 *  1. SINK sources are always realized (they are final outputs)
 *  2. CONTIGUOUS, COPY, STORE, ASSIGN are always realized
 *  3. Everything else is potentially fusable
 */

/* Ops that are always contiguous / never need realization */
static bool is_always_contiguous(PolyOps op) {
  switch (op) {
    case POLY_OP_CONTIGUOUS:
    case POLY_OP_ASSIGN:
    case POLY_OP_COPY:
    case POLY_OP_BUFFER:
    case POLY_OP_BUFFER_VIEW:
    case POLY_OP_CONST:
    case POLY_OP_BIND:
    case POLY_OP_DEVICE:
    case POLY_OP_MSELECT:
    case POLY_OP_MSTACK:
    case POLY_OP_PARAM:
    case POLY_OP_DEFINE_LOCAL:
    case POLY_OP_DEFINE_REG:
    case POLY_OP_LOAD:
    case POLY_OP_CALL:
    case POLY_OP_ENCDEC:
      return true;
    default:
      return false;
  }
}

static void realize_mark(PolyIndexingCtx *ictx, PolyUOp *u) {
  uint32_t h = ptr_hash(u);
  if (poly_map_get(ictx->realize_map, h, u, ptr_eq)) return;
  PolyRealizeInfo *ri = malloc(sizeof(PolyRealizeInfo));
  ri->axes = NULL;
  ri->n_axes = -1;  /* -1 = all axes (not yet populated with specific list) */
  poly_map_set(ictx->realize_map, h, u, ri, ptr_eq);
}

void poly_realize_map_build(PolyIndexingCtx *ictx, PolyUOp *sink) {
  if (sink->op != POLY_OP_SINK) return;

  /* Rule 1: SINK sources are always realized (unless always-contiguous) */
  for (int i = 0; i < sink->n_src; i++) {
    PolyUOp *s = sink->src[i];
    /* Walk past STORE to find the value being stored */
    if (s->op == POLY_OP_STORE && s->n_src >= 2) {
      /* The STORE's value (src[1]) is the thing that needs to be realized
       * as a kernel. But the STORE itself is a realize point. */
      realize_mark(ictx, s);
    } else if (!is_always_contiguous(s->op)) {
      realize_mark(ictx, s);
    }
  }

  /* Rule 2: Walk graph for ops that always realize */
  int n_uops;
  PolyUOp **topo = poly_toposort(ictx->ctx, sink, &n_uops);
  if (!topo) return;

  for (int i = 0; i < n_uops; i++) {
    PolyUOp *u = topo[i];
    switch (u->op) {
      case POLY_OP_CONTIGUOUS:
      case POLY_OP_COPY:
      case POLY_OP_ASSIGN:
        realize_mark(ictx, u);
        break;
      default:
        break;
    }
  }

}

bool poly_is_realized(PolyIndexingCtx *ictx, PolyUOp *u) {
  return poly_map_get(ictx->realize_map, ptr_hash(u), u, ptr_eq) != NULL;
}

/* ── Range map helpers ───────────────────────────────────────────────── */

/* Get cached shape, or compute and cache it */
static PolyShape ictx_shape(PolyIndexingCtx *ictx, PolyUOp *u) {
  PolyShape *cached = poly_map_get(ictx->shape_cache, ptr_hash(u), u, ptr_eq);
  if (cached) return *cached;
  PolyShape s = poly_uop_shape(ictx->ctx, u);
  PolyShape *stored = malloc(sizeof(PolyShape));
  *stored = s;
  poly_map_set(ictx->shape_cache, ptr_hash(u), u, stored, ptr_eq);
  return s;
}

/* Store a range entry in the range map */
static void range_map_set_valid(PolyIndexingCtx *ictx, PolyUOp *u,
                                 PolyUOp **in_rngs, int n_in,
                                 PolyUOp **out_rngs, int n_out,
                                 PolyUOp *valid) {
  PolyRangeEntry *re = malloc(sizeof(PolyRangeEntry));
  re->in_rngs = malloc(n_in * sizeof(PolyUOp *));
  memcpy(re->in_rngs, in_rngs, n_in * sizeof(PolyUOp *));
  re->n_in = n_in;
  re->out_rngs = malloc(n_out * sizeof(PolyUOp *));
  memcpy(re->out_rngs, out_rngs, n_out * sizeof(PolyUOp *));
  re->n_out = n_out;
  re->valid = valid;
  poly_map_set(ictx->range_map, ptr_hash(u), u, re, ptr_eq);
}

PolyRangeEntry *poly_range_map_get(PolyIndexingCtx *ictx, PolyUOp *u) {
  return poly_map_get(ictx->range_map, ptr_hash(u), u, ptr_eq);
}

/* Create a new RANGE UOp for a given dimension size.
 * axis_type: POLY_AXIS_LOOP for elementwise, POLY_AXIS_REDUCE for reduction dims.
 * Matches tinygrad's new_range(s, axistype) in indexing.py:45. */
static PolyUOp *new_range_ex(PolyIndexingCtx *ictx, int64_t dim_size,
                              PolyAxisType axis_type) {
  PolyCtx *ctx = ictx->ctx;
  /* If dim_size == 1, this range is trivially 0 */
  if (dim_size == 1)
    return poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(dim_size));
  return poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound,
                    poly_arg_range(ictx->next_range_id++, axis_type));
}

/* Convenience: create LOOP range (default for elementwise iteration). */
static PolyUOp *new_range(PolyIndexingCtx *ictx, int64_t dim_size) {
  return new_range_ex(ictx, dim_size, POLY_AXIS_LOOP);
}

/* Create a RANGE with a UOp bound (for dynamic shapes).
 * If bound is CONST(1), returns CONST(0) (singleton dim).
 * If bound is CONST(N), delegates to new_range_ex.
 * If bound is DEFINE_VAR, creates RANGE(DEFINE_VAR) with symbolic bound. */
static PolyUOp *new_range_uop(PolyIndexingCtx *ictx, PolyUOp *bound,
                                PolyAxisType axis_type) {
  PolyCtx *ctx = ictx->ctx;
  if (bound->op == POLY_OP_CONST) {
    /* Static bound: use existing helper */
    return new_range_ex(ictx, bound->arg.i, axis_type);
  }
  /* Symbolic bound (DEFINE_VAR): create RANGE with UOp bound */
  return poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound,
                    poly_arg_range(ictx->next_range_id++, axis_type));
}

/* ── Range propagation ───────────────────────────────────────────────── */
/*
 * Port of tinygrad's run_rangeify() (indexing.py:158-276).
 *
 * Walks the tensor graph in reverse topological order. For each UOp:
 *  1. If realized: create fresh RANGE UOps (new kernel boundary)
 *  2. If single consumer: inherit consumer's ranges (fusion)
 *  3. If multi-consumer: merge if same, or realize if different
 *  4. Apply movement op transforms to compute input ranges
 *  5. Add reduction ranges for REDUCE_AXIS
 */

void poly_range_propagate(PolyIndexingCtx *ictx, PolyUOp *sink) {
  PolyCtx *ctx = ictx->ctx;

  /* Get toposort (we'll walk in reverse) */
  int n_uops;
  PolyUOp **topo = poly_toposort(ctx, sink, &n_uops);
  if (!topo) return;


  /* Build consumer map if not already built */
  if (!ictx->consumer_map)
    ictx->consumer_map = poly_consumer_map_build(ctx, sink);

  /* tinygrad parity: per-node propagated ending ranges */
  PolyMap *ending_map = poly_map_new(n_uops * 2);

  /* Reverse topological traversal */
  for (int ti = n_uops - 1; ti >= 0; ti--) {
    PolyUOp *x = topo[ti];

    /* Skip ops that don't participate in range propagation */
    if (x->op == POLY_OP_DEVICE || x->op == POLY_OP_UNIQUE) continue;

    /* Get shape of this UOp */
    PolyShape shape = ictx_shape(ictx, x);

    /* Collect consumer ranges: for each consumer that has ranges,
     * get the input ranges that consumer assigned to this UOp */
    PolyUOp *consumer_rngs_buf[16][POLY_MAX_DIMS];
    int consumer_rngs_lens[16];
    memset(consumer_rngs_buf, 0, sizeof(consumer_rngs_buf));
    memset(consumer_rngs_lens, 0, sizeof(consumer_rngs_lens));
    int n_consumer_rngs = 0;

    PolyConsumerList *consumers = poly_consumer_map_get(ictx->consumer_map, x);
    PolyEndingRanges *ending = ending_get_or_create(ending_map, x);
    ending_clear(ending);
    if (consumers) {
      for (int ci = 0; ci < consumers->count && n_consumer_rngs < 16; ci++) {
        PolyUOp *consumer = consumers->items[ci];
        PolyRangeEntry *cre = poly_range_map_get(ictx, consumer);
        if (!cre) continue;

        /* The consumer's in_rngs contain the ranges it passes to its
         * sources. We need to find which source position x occupies
         * in the consumer, then extract the corresponding ranges.
         *
         * For now: the consumer's in_rngs ARE the ranges x should see
         * (since in a simple graph, the consumer passes its ranges
         * unchanged to all sources except for movement ops).
         *
         * This is the "output ranges" from x's perspective. */
        memcpy(consumer_rngs_buf[n_consumer_rngs], cre->in_rngs,
               cre->n_in * sizeof(PolyUOp *));
        consumer_rngs_lens[n_consumer_rngs] = cre->n_in;
        n_consumer_rngs++;
      }
      /* ending_ranges[x] = concat(ending_ranges[consumer]) */
      for (int ci = 0; ci < consumers->count; ci++) {
        PolyUOp *consumer = consumers->items[ci];
        PolyEndingRanges *ec = poly_map_get(ending_map, ptr_hash(consumer), consumer, ptr_eq);
        if (!ec) continue;
        for (int ei = 0; ei < ec->count; ei++) ending_add(ending, ec->items[ei]);
      }
    }

    /* Determine output ranges for x */
    PolyUOp *out_rngs[POLY_MAX_DIMS];
    memset(out_rngs, 0, sizeof(out_rngs));
    int n_out = 0;

    if (poly_is_realized(ictx, x)) {
      /* Case 1: Realized — create fresh ranges */
      if (shape.ndim <= 0) {
        /* Scalar or no shape — no ranges needed */
        n_out = 0;
      } else {
        /* Find DEFINE_VAR for dim 0 (dynamic batch dimension).
         * Check both the node itself (if BUFFER) and the STORE target (src[0]). */
        PolyUOp *dim0_var = NULL;
        if (x->op == POLY_OP_BUFFER && x->n_src >= 2 &&
            x->src[1]->op == POLY_OP_DEFINE_VAR)
          dim0_var = x->src[1];
        else if (x->op == POLY_OP_STORE && x->n_src >= 1 &&
                 x->src[0]->op == POLY_OP_BUFFER && x->src[0]->n_src >= 2 &&
                 x->src[0]->src[1]->op == POLY_OP_DEFINE_VAR)
          dim0_var = x->src[0]->src[1];
        for (int i = 0; i < shape.ndim; i++) {
          if (dim0_var && i == 0)
            out_rngs[i] = new_range_uop(ictx, dim0_var, POLY_AXIS_LOOP);
          else
            out_rngs[i] = new_range(ictx, shape.dims[i]);
        }
        n_out = shape.ndim;
      }

      /* Update realize map with specific axes */
      PolyRealizeInfo *ri = poly_map_get(ictx->realize_map, ptr_hash(x), x, ptr_eq);
      if (ri && n_out > 0) {
        free(ri->axes);
        ri->axes = malloc(n_out * sizeof(int));
        ri->n_axes = n_out;
        for (int i = 0; i < n_out; i++) ri->axes[i] = i;
      }
    } else if (n_consumer_rngs == 0) {
      /* Case 2: No consumers with ranges — skip */
      continue;
    } else if (n_consumer_rngs == 1) {
      /* Case 3: Single consumer — inherit (fusion!) */
      memcpy(out_rngs, consumer_rngs_buf[0],
             consumer_rngs_lens[0] * sizeof(PolyUOp *));
      n_out = consumer_rngs_lens[0];
    } else {
      /* Case 4: Multiple consumers — check if ranges agree */
      bool all_same = true;
      int ref_len = consumer_rngs_lens[0];

      for (int ci = 1; ci < n_consumer_rngs; ci++) {
        if (consumer_rngs_lens[ci] != ref_len) { all_same = false; break; }
        for (int d = 0; d < ref_len; d++) {
          if (consumer_rngs_buf[ci][d] != consumer_rngs_buf[0][d]) {
            all_same = false; break;
          }
        }
        if (!all_same) break;
      }

      if (all_same) {
        /* All consumers agree — inherit (fuse) */
        memcpy(out_rngs, consumer_rngs_buf[0], ref_len * sizeof(PolyUOp *));
        n_out = ref_len;
      } else {
        /* Consumers disagree — must realize this op */
        if (shape.ndim <= 0) {
          /* Scalar with disagreeing consumers: propagate empty ranges
           * so sources (e.g. REDUCE_AXIS below) still get range entries */
          n_out = 0;
        } else {
          realize_mark(ictx, x);
          PolyUOp *dim0_var_c4 = NULL;
          if (x->op == POLY_OP_BUFFER && x->n_src >= 2 &&
              x->src[1]->op == POLY_OP_DEFINE_VAR)
            dim0_var_c4 = x->src[1];
          else if (x->op == POLY_OP_STORE && x->n_src >= 1 &&
                   x->src[0]->op == POLY_OP_BUFFER && x->src[0]->n_src >= 2 &&
                   x->src[0]->src[1]->op == POLY_OP_DEFINE_VAR)
            dim0_var_c4 = x->src[0]->src[1];
          for (int i = 0; i < shape.ndim; i++) {
            if (dim0_var_c4 && i == 0)
              out_rngs[i] = new_range_uop(ictx, dim0_var_c4, POLY_AXIS_LOOP);
            else
              out_rngs[i] = new_range(ictx, shape.dims[i]);
          }
          n_out = shape.ndim;

          /* For BUFFER nodes with disagreeing consumers: save
           * per-consumer ranges for the BUFFER handler. */
          if (x->op == POLY_OP_BUFFER && n_consumer_rngs > 0) {
            PolyBufferAltRngs *alt = malloc(sizeof(PolyBufferAltRngs));
            alt->count = 0;
            for (int ci = 0; ci < n_consumer_rngs && alt->count < POLY_MAX_ALT_RNGS; ci++) {
              int len = consumer_rngs_lens[ci];
              if (len <= 0 || len > POLY_MAX_DIMS) continue;
              int ai = alt->count;
              alt->lens[ai] = len;
              memcpy(alt->rngs[ai], consumer_rngs_buf[ci], len * sizeof(PolyUOp *));
              alt->count++;
            }
            if (alt->count > 0)
              poly_map_set(ictx->buffer_alt_rngs, ptr_hash(x), x, alt, ptr_eq);
            else
              free(alt);
            if (alt->count > 0) g_rangeify_stats.buffer_alt_created++;
          }

          PolyRealizeInfo *ri = poly_map_get(ictx->realize_map, ptr_hash(x), x, ptr_eq);
          if (ri && n_out > 0) {
            free(ri->axes);
            ri->axes = malloc(n_out * sizeof(int));
            ri->n_axes = n_out;
            for (int i = 0; i < n_out; i++) ri->axes[i] = i;
          }
        }
      }
    }

    /* tinygrad parity: if ended ranges flow into elementwise/reduce, realize axes */
    if (ending->count > 0 &&
        (poly_opset_has(POLY_GROUP_ELEMENTWISE, x->op) || x->op == POLY_OP_REDUCE_AXIS)) {
      int realize_axes[POLY_MAX_DIMS];
      int n_realize_axes = 0;
      PolyRealizeInfo *ri = poly_map_get(ictx->realize_map, ptr_hash(x), x, ptr_eq);
      if (ri && ri->axes && ri->n_axes > 0) {
        for (int i = 0; i < ri->n_axes && n_realize_axes < POLY_MAX_DIMS; i++) {
          int ax = ri->axes[i];
          bool seen = false;
          for (int j = 0; j < n_realize_axes; j++) {
            if (realize_axes[j] == ax) { seen = true; break; }
          }
          if (!seen) realize_axes[n_realize_axes++] = ax;
        }
      }

      for (int i = 0; i < n_out && n_realize_axes < POLY_MAX_DIMS; i++) {
        bool seen = false;
        for (int j = 0; j < n_realize_axes; j++) {
          if (realize_axes[j] == i) { seen = true; break; }
        }
        if (!seen) realize_axes[n_realize_axes++] = i;
      }

      ending_clear(ending);
      if (n_realize_axes > 0) {
        realize_mark(ictx, x);
        ri = poly_map_get(ictx->realize_map, ptr_hash(x), x, ptr_eq);
        if (ri) {
          free(ri->axes);
          ri->axes = malloc(n_realize_axes * sizeof(int));
          ri->n_axes = n_realize_axes;
          for (int i = 0; i < n_realize_axes; i++) ri->axes[i] = realize_axes[i];
        }
        PolyUOp *dim0_var_c4b = NULL;
        if (x->op == POLY_OP_BUFFER && x->n_src >= 2 &&
            x->src[1]->op == POLY_OP_DEFINE_VAR)
          dim0_var_c4b = x->src[1];
        else if (x->op == POLY_OP_STORE && x->n_src >= 1 &&
                 x->src[0]->op == POLY_OP_BUFFER && x->src[0]->n_src >= 2 &&
                 x->src[0]->src[1]->op == POLY_OP_DEFINE_VAR)
          dim0_var_c4b = x->src[0]->src[1];
        for (int i = 0; i < n_realize_axes; i++) {
          int ax = realize_axes[i];
          if (ax >= 0 && ax < n_out && ax < shape.ndim) {
            if (dim0_var_c4b && ax == 0)
              out_rngs[ax] = new_range_uop(ictx, dim0_var_c4b, POLY_AXIS_LOOP);
            else
              out_rngs[ax] = new_range(ictx, shape.dims[ax]);
          }
        }
      }
    }

    if (n_out == 0 && !poly_is_realized(ictx, x) && x->n_src == 0) continue;

    /* Compute input ranges (what this op's sources see) */
    PolyUOp *in_rngs[POLY_MAX_DIMS];
    memset(in_rngs, 0, sizeof(in_rngs));
    int n_in = n_out;
    memcpy(in_rngs, out_rngs, n_out * sizeof(PolyUOp *));

    /* Apply movement op transforms */
    PolyUOp *valid_mask = NULL;
    if (poly_opset_has(POLY_GROUP_MOVEMENT, x->op) && x->n_src > 0) {
      PolyShape src_shape = ictx_shape(ictx, x->src[0]);
      if (src_shape.ndim >= 0) {
        PolyUOp *transformed[POLY_MAX_DIMS];
        int n_transformed = 0;
        if (poly_apply_movement_op(ctx, x->op, src_shape, x->arg,
                                   out_rngs, n_out,
                                   transformed, &n_transformed, &valid_mask)) {
          memcpy(in_rngs, transformed, n_transformed * sizeof(PolyUOp *));
          n_in = n_transformed;
        }
      }
    }

    /* tinygrad parity: EXPAND can inject ranges that must end upstream */
    if (x->op == POLY_OP_EXPAND) {
      int n_cmp = n_in < n_out ? n_in : n_out;
      for (int i = 0; i < n_cmp; i++) {
        if (in_rngs[i] != out_rngs[i]) ending_add(ending, out_rngs[i]);
      }
    }

    /* REDUCE_AXIS: add new reduction ranges for reduced axes */
    if (x->op == POLY_OP_REDUCE_AXIS && x->arg.kind == POLY_ARG_REDUCE_AXIS) {
      PolyShape src_shape = ictx_shape(ictx, x->src[0]);
      if (src_shape.ndim > 0) {
        int n_axes = x->arg.reduce_axis.n;
        int64_t *axes = x->arg.reduce_axis.axes;

        /* Input ranges = output ranges + new ranges for reduced axes */
        PolyUOp *new_in[POLY_MAX_DIMS];
        bool is_reduced[POLY_MAX_DIMS] = {false};
        for (int i = 0; i < n_axes; i++) {
          int ax = (int)axes[i];
          if (ax >= 0 && ax < src_shape.ndim) is_reduced[ax] = true;
        }

        for (int i = 0; i < src_shape.ndim; i++) {
          if (is_reduced[i]) {
            /* New range for reduced axis — POLY_AXIS_REDUCE matches
             * tinygrad's new_range(s, axistype=AxisType.REDUCE) */
            new_in[i] = new_range_ex(ictx, src_shape.dims[i], POLY_AXIS_REDUCE);
          } else if (i < n_out) {
            new_in[i] = out_rngs[i];
          } else {
            new_in[i] = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
          }
        }
        memcpy(in_rngs, new_in, src_shape.ndim * sizeof(PolyUOp *));
        n_in = src_shape.ndim;
      }
    }

    /* Store in range map (with valid mask for PAD ops) */
    range_map_set_valid(ictx, x, in_rngs, n_in, out_rngs, n_out, valid_mask);
  }

  poly_map_foreach(ending_map, ending_destroy, NULL);
  poly_map_destroy(ending_map);
}

/* ── Apply rangeify graph rewrite ────────────────────────────────────── */
/*
 * Transforms tensor-level IR to kernel-level IR using range annotations.
 *
 * Walks toposort order, applying:
 *  1. Movement ops (RESHAPE, EXPAND, PERMUTE, SHRINK, FLIP) → src[0]
 *  2. PAD → WHERE(valid_mask, src[0], 0.0)
 *  3. REDUCE_AXIS → REDUCE(value, reduce_ranges...) with arg = reduce_op
 *  4. Realized computed ops → BUFFERIZE(op, out_ranges...)
 *  5. Everything else → pass through (rebuild if sources changed)
 */

static PolyUOp *rmap_get(PolyMap *m, PolyUOp *key) {
  return poly_map_get(m, ptr_hash(key), key, ptr_eq);
}

static void rmap_set(PolyMap *m, PolyUOp *key, PolyUOp *val) {
  poly_map_set(m, ptr_hash(key), key, val, ptr_eq);
}

/* Map consumer ranges through a movement-op chain down to a base source.
 * Used when a movement source is removed (RESHAPE/EXPAND/...) but consumers
 * still need source-specific indexing expressions rooted in local ranges. */
static bool map_ranges_through_movement_chain(PolyCtx *ctx, PolyIndexingCtx *ictx,
                                              PolyUOp *from, PolyUOp *to_base,
                                              PolyUOp **start_rngs, int n_start,
                                              PolyUOp **out_rngs, int *out_n) {
  if (!from || !to_base || !start_rngs || !out_rngs || !out_n) return false;
  if (n_start < 0 || n_start > POLY_MAX_DIMS) return false;

  PolyUOp *cur = from;
  PolyUOp *cur_rngs[POLY_MAX_DIMS];
  PolyUOp *next_rngs[POLY_MAX_DIMS];
  memcpy(cur_rngs, start_rngs, n_start * sizeof(PolyUOp *));
  int n_cur = n_start;

  while (cur != to_base) {
    if (!poly_opset_has(POLY_GROUP_MOVEMENT, cur->op) || cur->n_src <= 0)
      return false;
    PolyShape src_shape = ictx_shape(ictx, cur->src[0]);
    PolyUOp *valid = NULL;
    int n_next = 0;
    if (!poly_apply_movement_op(ctx, cur->op, src_shape, cur->arg,
                                cur_rngs, n_cur, next_rngs, &n_next, &valid))
      return false;
    if (n_next < 0 || n_next > POLY_MAX_DIMS) return false;
    memcpy(cur_rngs, next_rngs, n_next * sizeof(PolyUOp *));
    n_cur = n_next;
    cur = cur->src[0];
  }

  memcpy(out_rngs, cur_rngs, n_cur * sizeof(PolyUOp *));
  *out_n = n_cur;
  return true;
}

static PolyUOp *movement_chain_base_buffer(PolyUOp *u) {
  PolyUOp *cur = u;
  bool saw_movement = false;
  while (cur && poly_opset_has(POLY_GROUP_MOVEMENT, cur->op) && cur->n_src > 0) {
    saw_movement = true;
    cur = cur->src[0];
  }
  if (!saw_movement) return NULL;
  return (cur && cur->op == POLY_OP_BUFFER) ? cur : NULL;
}

PolyUOp *poly_apply_rangeify(PolyIndexingCtx *ictx, PolyUOp *sink) {
  PolyCtx *ctx = ictx->ctx;

  int n_uops;
  PolyUOp **topo = poly_toposort(ctx, sink, &n_uops);
  if (!topo) return sink;

  /* Replace map: old UOp → new UOp */
  PolyMap *rmap = poly_map_new(n_uops * 2);

  for (int i = 0; i < n_uops; i++) {
    PolyUOp *u = topo[i];
    PolyRangeEntry *u_re = poly_range_map_get(ictx, u);

    /* Remap sources through replace map.
     * Zero-init both paths so the analyzer knows new_src[i] is never garbage
     * even if accessed before the loop fills it (e.g. REDUCE_AXIS with n_src>0). */
    bool src_changed = false;
    PolyUOp *new_src_buf[16] = {0};
    PolyUOp **new_src = (u->n_src > 16) ?
      calloc(u->n_src, sizeof(PolyUOp *)) : new_src_buf;

    for (int j = 0; j < u->n_src; j++) {
      PolyUOp *rs = rmap_get(rmap, u->src[j]);
      new_src[j] = rs ? rs : u->src[j];
      if (new_src[j] != u->src[j]) src_changed = true;

      /* Structural per-consumer BUFFER indexing fallback:
       * capture source-specific ranges through movement chains ending at BUFFER
       * so lowering can pick local expressions without foreign-range remap. */
      if (u_re && u_re->n_in > 0) {
        PolyUOp *base_buf = movement_chain_base_buffer(u->src[j]);
        if (!base_buf) continue;
        PolyUOp *mapped_rngs[POLY_MAX_DIMS];
        int n_mapped = 0;
        if (map_ranges_through_movement_chain(ctx, ictx,
                                              u->src[j], base_buf,
                                              u_re->in_rngs, u_re->n_in,
                                              mapped_rngs, &n_mapped) &&
            n_mapped > 0) {
          buffer_alt_add(ictx, base_buf, mapped_rngs, n_mapped);
        }
      }
    }

    /* Wrap BUFFERIZE and BUFFER sources with INDEX.
     * Port of tinygrad's create_bufferize_and_index_based_on_ranges:
     * BUFFERIZE sources get consumer-side INDEX for intermediate buffer reads.
     * When add_buffer_indices is set, BUFFER sources also get flat INDEX,
     * and BUFFERIZE INDEX is flattened (2-src instead of multi-dim). */
    for (int j = 0; j < u->n_src; j++) {
      PolyRangeEntry *re = poly_range_map_get(ictx, u);

      if (new_src[j]->op == POLY_OP_BUFFERIZE) {
        int n_buf_rngs = new_src[j]->n_src - 1;
        if (re && n_buf_rngs > 0) {
          PolyUOp *orig_src = u->src[j];
          PolyUOp *idx_rngs[POLY_MAX_DIMS];
          int n_idx = 0;
          bool idx_ok = false;

          /* Prefer rebuilding consumer indexing from this op's local ranges
           * through any removed movement chain on the original source. */
          if (orig_src != new_src[j]) {
            idx_ok = map_ranges_through_movement_chain(ctx, ictx,
                                                       orig_src, new_src[j],
                                                       re->in_rngs, re->n_in,
                                                       idx_rngs, &n_idx);
          }

          /* Prefer realized-axis mapping from the original source node. */
          if (!idx_ok) {
            n_idx = 0;
            PolyRealizeInfo *sri = poly_map_get(ictx->realize_map,
                                                ptr_hash(orig_src), orig_src,
                                                ptr_eq);
            if (sri && sri->n_axes > 0 && sri->axes) {
              for (int ai = 0; ai < sri->n_axes && n_idx < POLY_MAX_DIMS; ai++) {
                int ax = sri->axes[ai];
                if (ax >= 0 && ax < re->n_in)
                  idx_rngs[n_idx++] = re->in_rngs[ax];
              }
            } else if (re->n_in == n_buf_rngs) {
              for (int d = 0; d < re->n_in && n_idx < POLY_MAX_DIMS; d++)
                idx_rngs[n_idx++] = re->in_rngs[d];
            }
          }

          if (n_idx != n_buf_rngs) continue;

          if (ictx->add_buffer_indices) {
            /* New split path: multi-range INDEX with POINTER dtype.
             * pm_remove_bufferize requires arity equality (idx.n_src == bufferize.n_src).
             * add_kernel_loads checks dtype.is_ptr to insert LOAD. */
            int64_t buf_dims[POLY_MAX_DIMS];
            bool all_const = true;
            for (int d = 0; d < n_buf_rngs; d++) {
              PolyUOp *rng = new_src[j]->src[1 + d];
              if (rng->op == POLY_OP_RANGE && rng->n_src > 0 &&
                  rng->src[0]->op == POLY_OP_CONST)
                buf_dims[d] = rng->src[0]->arg.i;
              else {
                buf_dims[d] = 1;
                all_const = false;
              }
            }
            PolyShape bsh = { .dims = buf_dims, .ndim = n_buf_rngs };
            int64_t numel = all_const ? poly_shape_numel(bsh) : -1;
            PolyDType ptr_dt = poly_dtype_ptr(poly_dtype_scalar(new_src[j]->dtype),
                                               numel, POLY_ADDR_GLOBAL);
            PolyUOp *idx_src[POLY_MAX_DIMS + 1];
            idx_src[0] = new_src[j];
            for (int d = 0; d < n_idx; d++)
              idx_src[d + 1] = idx_rngs[d];
            new_src[j] = poly_uop(ctx, POLY_OP_INDEX, ptr_dt,
                                   idx_src, n_idx + 1, poly_arg_none());
          } else {
            /* Legacy path: multi-dim INDEX with element dtype.
             * lower_v2 handles INDEX(BUFFERIZE) via positional zip, inserts LOAD.
             * Pointer dtype here would corrupt lower_v2's LOAD dtype computation. */
            PolyUOp *idx_src[POLY_MAX_DIMS + 1];
            idx_src[0] = new_src[j];
            for (int d = 0; d < n_idx; d++)
              idx_src[d + 1] = idx_rngs[d];
            new_src[j] = poly_uop(ctx, POLY_OP_INDEX, new_src[j]->dtype,
                                   idx_src, n_idx + 1, poly_arg_none());
          }
          src_changed = true;
        }
      }

      /* Wrap bare BUFFER sources with flat INDEX (new split path only).
       * Store target (STORE src[0]) uses out_rngs; read sources use
       * movement-chain-mapped in_rngs or direct in_rngs. */
      else if (ictx->add_buffer_indices &&
               new_src[j]->op == POLY_OP_BUFFER &&
               u->op != POLY_OP_SINK) {
        if (!re) continue;
        PolyUOp *idx_rngs[POLY_MAX_DIMS];
        int n_idx = 0;

        if ((u->op == POLY_OP_STORE || u->op == POLY_OP_ASSIGN) && j == 0) {
          /* Store/ASSIGN target: use output ranges. */
          for (int d = 0; d < re->n_out && d < POLY_MAX_DIMS; d++)
            idx_rngs[d] = re->out_rngs[d];
          n_idx = re->n_out;
        } else {
          /* Read source: map consumer's in_rngs through movement chain. */
          PolyUOp *orig_src = u->src[j];
          if (orig_src != new_src[j]) {
            bool ok = map_ranges_through_movement_chain(ctx, ictx,
                orig_src, new_src[j],
                re->in_rngs, re->n_in,
                idx_rngs, &n_idx);
            if (!ok) n_idx = 0;
          }
          if (n_idx == 0 && re->n_in > 0) {
            /* Direct source or chain mapping failed: use consumer's in_rngs. */
            for (int d = 0; d < re->n_in && d < POLY_MAX_DIMS; d++)
              idx_rngs[d] = re->in_rngs[d];
            n_idx = re->n_in;
          }
        }

        if (n_idx > 0) {
          /* Derive shape from range bounds for flat index computation.
           * The BUFFER is always 1D (flat), but ranges may be multi-dim. */
          int64_t rng_dims[POLY_MAX_DIMS];
          for (int d = 0; d < n_idx; d++) {
            if (idx_rngs[d]->op == POLY_OP_RANGE && idx_rngs[d]->n_src > 0 &&
                idx_rngs[d]->src[0]->op == POLY_OP_CONST)
              rng_dims[d] = idx_rngs[d]->src[0]->arg.i;
            else if (idx_rngs[d]->op == POLY_OP_CONST)
              rng_dims[d] = 1;  /* CONST(0) = singleton dim */
            else
              rng_dims[d] = 1;
          }
          PolyShape idx_shape = { .dims = rng_dims, .ndim = n_idx };
          PolyUOp *flat_idx = poly_compute_flat_index(ctx, idx_rngs, n_idx, idx_shape);
          int64_t numel = poly_shape_numel(idx_shape);
          PolyDType ptr_dt = poly_dtype_ptr(poly_dtype_scalar(new_src[j]->dtype),
                                             numel, POLY_ADDR_GLOBAL);
          new_src[j] = poly_uop2(ctx, POLY_OP_INDEX, ptr_dt,
                                   new_src[j], flat_idx, poly_arg_none());
          src_changed = true;
        }
      }
    }

    PolyUOp *result = NULL;
    bool transformed_compute = false;

    switch (u->op) {

    /* ── Rule 1: Movement ops → src[0] ─────────────────────────────── */
    case POLY_OP_RESHAPE:
    case POLY_OP_EXPAND:
    case POLY_OP_PERMUTE:
    case POLY_OP_SHRINK:
    case POLY_OP_FLIP:
      if (u->n_src > 0) result = new_src[0];
      break;

    /* ── Rule 2: PAD → WHERE(valid, src[0], 0.0) ──────────────────── */
    case POLY_OP_PAD: {
      if (u->n_src == 0) break;
      PolyRangeEntry *re = poly_range_map_get(ictx, u);
      if (re && re->valid) {
        PolyUOp *zero;
        if (poly_dtype_is_float(u->dtype))
          zero = poly_uop0(ctx, POLY_OP_CONST, u->dtype, poly_arg_float(0.0));
        else
          zero = poly_uop0(ctx, POLY_OP_CONST, u->dtype, poly_arg_int(0));
        result = poly_uop3(ctx, POLY_OP_WHERE, u->dtype,
                            re->valid, new_src[0], zero, poly_arg_none());
        transformed_compute = true;
      } else {
        result = new_src[0];
      }
      break;
    }

    /* ── Rule 3: REDUCE_AXIS → REDUCE with range sources ──────────── */
    case POLY_OP_REDUCE_AXIS: {
      PolyRangeEntry *re = poly_range_map_get(ictx, u);
      if (!re) break;

      /* Build REDUCE sources: [value, reduce_range_0, reduce_range_1, ...]
       * Reduce ranges = ranges in in_rngs that are NOT in out_rngs */
      PolyUOp *reduce_src[POLY_MAX_DIMS + 1];
      int n_rsrc = 0;

      /* src[0] = value being reduced */
      if (u->n_src > 0)
        reduce_src[n_rsrc++] = new_src[0];

      /* Identify reduce ranges (in input but not output) */
      for (int d = 0; d < re->n_in; d++) {
        bool is_output = false;
        for (int k = 0; k < re->n_out; k++) {
          if (re->in_rngs[d] == re->out_rngs[k]) { is_output = true; break; }
        }
        if (!is_output)
          reduce_src[n_rsrc++] = re->in_rngs[d];
      }

      /* arg = the reduction op (ADD/MUL/MAX) */
      PolyArg reduce_arg = poly_arg_int((int64_t)u->arg.reduce_axis.op);
      result = poly_uop(ctx, POLY_OP_REDUCE, u->dtype,
                         reduce_src, n_rsrc, reduce_arg);
      transformed_compute = true;
      /* Record mapping: post-rangeify REDUCE → pre-rangeify REDUCE_AXIS */
      rmap_set(ictx->reduce_origin, result, u);
      break;
    }

    default:
      break;
    }

    /* ── Rule 4: Realized computed ops → BUFFERIZE ────────────────── */
    if (poly_is_realized(ictx, u) &&
        u->op != POLY_OP_STORE && u->op != POLY_OP_SINK &&
        u->op != POLY_OP_BUFFER) {
      /* Check if global context already has a BUFFERIZE for this op
       * (child contexts reuse global BUFFERIZE for intermediate buffer identity) */
      PolyUOp *existing_buf = NULL;
      if (ictx->realized_to_bufferize)
        existing_buf = poly_map_get(ictx->realized_to_bufferize,
                                     ptr_hash(u), u, ptr_eq);
      if (existing_buf) {
        result = existing_buf;
        if (ictx->bufferize_to_realized)
          poly_map_set(ictx->bufferize_to_realized, ptr_hash(existing_buf),
                        existing_buf, u, ptr_eq);
      } else {
        PolyRangeEntry *re = poly_range_map_get(ictx, u);
        if (re) {
          /* BUFFERIZE(value, range_0, range_1, ...) */
          PolyUOp *buf_src[POLY_MAX_DIMS + 1];
          int n_bsrc = 0;

          /* src[0] = the computed value (with remapped sources).
           * For movement ops: result is the absorbed source (correct).
           * For REDUCE_AXIS: result is the REDUCE UOp (transformed_compute).
           * For unhandled ops: result is NULL, rebuild or use original. */
          PolyUOp *val;
          if (result) {
            val = result;
          } else if (src_changed) {
            val = poly_uop(ctx, u->op, u->dtype, new_src, u->n_src, u->arg);
          } else {
            val = u;
          }
          buf_src[n_bsrc++] = val;

          for (int d = 0; d < re->n_out; d++)
            buf_src[n_bsrc++] = re->out_rngs[d];

          /* Stage 3 gate: store removable flag. removable=false for ops that must always
           * materialize (CONTIGUOUS, COPY, ASSIGN, ENCDEC, BUFFER, BUFFER_VIEW). */
          bool removable = (u->op != POLY_OP_CONTIGUOUS &&
                            u->op != POLY_OP_COPY &&
                            u->op != POLY_OP_ASSIGN &&
                            u->op != POLY_OP_ENCDEC &&
                            u->op != POLY_OP_BUFFER &&
                            u->op != POLY_OP_BUFFER_VIEW);
          result = poly_uop(ctx, POLY_OP_BUFFERIZE, u->dtype,
                             buf_src, n_bsrc, poly_arg_int(removable ? 1 : 0));

          /* Register for child contexts to reuse */
          if (ictx->realized_to_bufferize)
            poly_map_set(ictx->realized_to_bufferize, ptr_hash(u),
                          u, result, ptr_eq);
          if (ictx->bufferize_to_realized)
            poly_map_set(ictx->bufferize_to_realized, ptr_hash(result),
                          result, u, ptr_eq);
        }
      }
    }

    /* ── Rule 5: Rebuild with new sources if changed ──────────────── */
    if (!result && src_changed) {
      result = poly_uop(ctx, u->op, u->dtype, new_src, u->n_src, u->arg);
    }

    if (result && result != u)
      rmap_set(rmap, u, result);

    /* Free heap-allocated new_src. Compare against stack buffer rather than
     * re-checking n_src>16 — clearer for the analyzer's malloc/free tracking. */
    if (new_src != new_src_buf) free(new_src);
  }

  PolyUOp *new_sink = rmap_get(rmap, sink);
  poly_map_destroy(rmap);
  return new_sink ? new_sink : sink;
}

/* ── pm_add_buffers: BUFFERIZE → BUFFER + STORE + END + AFTER ───────── */
/*
 * Port of tinygrad's bufferize_to_store() (rangeify.py:338-378).
 *
 * After apply_rangeify, the graph has BUFFERIZE(value, range_0, ..., range_N-1)
 * nodes marking intermediate buffers. This pass converts each BUFFERIZE into:
 *
 *   BUFFER(LUNIQUE(id), DEVICE())           — new intermediate buffer
 *   INDEX(BUFFER, flat_index, ptr_dtype)    — compute store index
 *   STORE(INDEX, value)                     — write computed value
 *   END(STORE, range_0)                     — close ranges
 *   ...
 *   END(..., range_N-1)
 *   AFTER(BUFFER, final_END)                — buffer available after store
 *
 * The result AFTER node replaces the BUFFERIZE. Consumers that read from
 * the BUFFERIZE now read from the BUFFER (with data dependency via AFTER).
 *
 * This is the CPU-only (GLOBAL addrspace) path. The local/DEFINE_LOCAL
 * path is handled by pm_add_buffers_local in codegen (Phase 3).
 */

/* Plain struct for storing intermediate buffer dimension info.
 * Stored in buf_dims_map cast to PolyUOp* to avoid UOp hash-consing CSE
 * (which would redirect the pointer to arena memory and cause bad-free). */
typedef struct {
  int n;
  int64_t dims[POLY_MAX_DIMS];
} PolyBufDimsInfo;

/* Convert a single BUFFERIZE node to a BUFFER+STORE+END+AFTER chain.
 * If buf_dims_map is non-NULL, stores a BUFFER→PolyBufDimsInfo* mapping
 * encoding the BUFFERIZE's full per-dim sizes (including 1 for CONST(0)
 * dims). Cast to PolyUOp* for storage in PolyMap. */
static PolyUOp *bufferize_to_store_global(PolyCtx *ctx, int *lunique_counter,
                                           PolyUOp *bufferize,
                                           PolyMap *buf_dims_map) {
  /* src[0] = value, src[1:] = range UOps */
  PolyUOp *value = bufferize->src[0];
  int n_ranges = bufferize->n_src - 1;

  /* ── ASSIGN branch: write to existing target buffer (no LUNIQUE) ──
   * Matches tinygrad rangeify.py:345-354.
   * BUFFERIZE(ASSIGN(INDEX(BUFFER, flat_idx), computed_value), rngs)
   * → AFTER(BUFFER, END(STORE(INDEX_ptr, computed_value), rngs)) */
  if (value->op == POLY_OP_ASSIGN && value->n_src >= 2) {
    PolyUOp *assign_target = value->src[0];  /* INDEX(BUFFER, flat_idx) */
    PolyUOp *assign_src = value->src[1];      /* computed value expression */

    /* assign_target should be INDEX from apply_rangeify BUFFER wrapping */
    PolyUOp *target_buf;
    if (assign_target->op == POLY_OP_INDEX && assign_target->n_src >= 1)
      target_buf = assign_target->src[0];
    else
      target_buf = assign_target;  /* bare BUFFER (shouldn't happen after rangeify) */

    /* Compute size for ptr dtype */
    int64_t asize = 1;
    for (int i = 0; i < n_ranges; i++) {
      PolyUOp *rng = bufferize->src[1 + i];
      if (rng->op == POLY_OP_RANGE && rng->n_src > 0 &&
          rng->src[0]->op == POLY_OP_CONST)
        asize *= rng->src[0]->arg.i;
    }
    if (asize <= 0) asize = 1;
    PolyDType sdtype = poly_dtype_ptr(bufferize->dtype, asize, POLY_ADDR_GLOBAL);

    /* Rebuild INDEX with pointer dtype (matching tinygrad's assign_target.replace(dtype=sdtype)) */
    PolyUOp *store_idx;
    if (assign_target->op == POLY_OP_INDEX) {
      store_idx = poly_uop(ctx, POLY_OP_INDEX, sdtype,
                            assign_target->src, assign_target->n_src,
                            assign_target->arg);
    } else {
      PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
      store_idx = poly_uop2(ctx, POLY_OP_INDEX, sdtype,
                              target_buf, zero, poly_arg_none());
    }

    /* STORE(INDEX_ptr, computed_value) */
    PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID,
                                 store_idx, assign_src, poly_arg_none());

    /* END chain wrapping: sort ranges by axis_id */
    PolyUOp *sorted_ranges[POLY_MAX_DIMS];
    int n_sorted = 0;
    for (int i = 0; i < n_ranges; i++) {
      PolyUOp *rng = bufferize->src[1 + i];
      if (rng->op == POLY_OP_RANGE)
        sorted_ranges[n_sorted++] = rng;
    }
    for (int i = 0; i < n_sorted - 1; i++) {
      for (int j = i + 1; j < n_sorted; j++) {
        int64_t ai = poly_range_axis_id(sorted_ranges[i]->arg);
        int64_t aj = poly_range_axis_id(sorted_ranges[j]->arg);
        if (ai > aj) {
          PolyUOp *tmp = sorted_ranges[i];
          sorted_ranges[i] = sorted_ranges[j];
          sorted_ranges[j] = tmp;
        }
      }
    }

    PolyUOp *current = store;
    for (int i = 0; i < n_sorted; i++) {
      PolyUOp *end_src[2] = { current, sorted_ranges[i] };
      current = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2,
                           poly_arg_none());
    }

    /* AFTER(existing_BUFFER, end_chain) — uses EXISTING buffer, not LUNIQUE */
    PolyUOp *after_src[2] = { target_buf, current };
    return poly_uop(ctx, POLY_OP_AFTER, target_buf->dtype, after_src, 2,
                     poly_arg_none());
  }

  /* ── Normal path: new intermediate buffer (LUNIQUE) ── */

  /* Compute size = product of range bounds */
  int64_t size = 1;
  for (int i = 0; i < n_ranges; i++) {
    PolyUOp *rng = bufferize->src[1 + i];
    if (rng->op == POLY_OP_RANGE && rng->n_src > 0 &&
        rng->src[0]->op == POLY_OP_CONST) {
      size *= rng->src[0]->arg.i;
    } else if (rng->op == POLY_OP_CONST) {
      /* CONST(0) range = size-1 dimension */
      size *= 1;
    }
  }
  if (size <= 0) size = 1;

  /* Create BUFFER(LUNIQUE(id), DEVICE()) with arg=size */
  PolyUOp *lunique = poly_uop0(ctx, POLY_OP_LUNIQUE, POLY_VOID,
                                 poly_arg_int((*lunique_counter)++));
  PolyUOp *device = poly_uop0(ctx, POLY_OP_DEVICE, POLY_VOID,
                                poly_arg_none());
  PolyUOp *buf_src[2] = { lunique, device };
  PolyUOp *buf = poly_uop(ctx, POLY_OP_BUFFER, bufferize->dtype,
                            buf_src, 2, poly_arg_int(size));

  /* Store the BUFFERIZE's full per-dim sizes in buf_dims_map.
   * This includes 1 for CONST(0) (singleton) dims and N for RANGE(N) dims.
   * Consumers need this to compute correct flat indices into the intermediate
   * buffer (using buffer strides, not consumer iteration strides).
   *
   * We use a plain malloc'd PolyBufDimsInfo struct (cast to PolyUOp*) instead
   * of a UOp because poly_uop0 uses hash-consing CSE — if two BUFFERIZEs
   * produce the same dims, the UOp gets deduplicated and the malloc'd vals
   * pointer is lost, causing bad-free on arena memory during cleanup. */
  if (buf_dims_map) {
    PolyBufDimsInfo *info = malloc(sizeof(PolyBufDimsInfo));
    info->n = n_ranges;
    for (int i = 0; i < n_ranges; i++) {
      PolyUOp *rng = bufferize->src[1 + i];
      if (rng->op == POLY_OP_RANGE && rng->n_src > 0 &&
          rng->src[0]->op == POLY_OP_CONST)
        info->dims[i] = rng->src[0]->arg.i;
      else
        info->dims[i] = 1;
    }
    poly_map_set(buf_dims_map, ptr_hash(buf), buf, (PolyUOp *)info, ptr_eq);
  }

  /* Compute flat index from ranges */
  PolyUOp *ranges[POLY_MAX_DIMS];
  int64_t dims[POLY_MAX_DIMS];
  int n_active = 0;
  for (int i = 0; i < n_ranges; i++) {
    PolyUOp *rng = bufferize->src[1 + i];
    if (rng->op == POLY_OP_RANGE && rng->n_src > 0 &&
        rng->src[0]->op == POLY_OP_CONST) {
      ranges[n_active] = rng;
      dims[n_active] = rng->src[0]->arg.i;
      n_active++;
    }
    /* Skip CONST(0) ranges — they contribute nothing to index */
  }

  PolyUOp *flat_idx;
  if (n_active > 0) {
    PolyShape sh = { .dims = dims, .ndim = n_active };
    flat_idx = poly_compute_flat_index(ctx, ranges, n_active, sh);
  } else {
    flat_idx = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  }

  /* INDEX(BUFFER, flat_idx) with ptr dtype */
  PolyDType sdtype = poly_dtype_ptr(bufferize->dtype, size, POLY_ADDR_GLOBAL);
  PolyUOp *idx = poly_uop2(ctx, POLY_OP_INDEX, sdtype, buf, flat_idx,
                             poly_arg_none());

  /* STORE(INDEX, value) */
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx, value,
                               poly_arg_none());

  /* END(STORE, range_0), END(END, range_1), ... for each active RANGE
   * Sort by axis_id (tinygrad: sorted(idx.ranges, key=lambda x: x.arg)) */
  PolyUOp *sorted_ranges[POLY_MAX_DIMS];
  int n_sorted = 0;
  for (int i = 0; i < n_ranges; i++) {
    PolyUOp *rng = bufferize->src[1 + i];
    if (rng->op == POLY_OP_RANGE)
      sorted_ranges[n_sorted++] = rng;
  }
  /* Sort by axis_id */
  for (int i = 0; i < n_sorted - 1; i++) {
    for (int j = i + 1; j < n_sorted; j++) {
      int64_t ai = poly_range_axis_id(sorted_ranges[i]->arg);
      int64_t aj = poly_range_axis_id(sorted_ranges[j]->arg);
      if (ai > aj) {
        PolyUOp *tmp = sorted_ranges[i];
        sorted_ranges[i] = sorted_ranges[j];
        sorted_ranges[j] = tmp;
      }
    }
  }

  PolyUOp *current = store;
  for (int i = 0; i < n_sorted; i++) {
    PolyUOp *end_src[2] = { current, sorted_ranges[i] };
    current = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2,
                         poly_arg_none());
  }

  /* AFTER(BUFFER, end_chain): buffer is available after store completes */
  PolyUOp *after_src[2] = { buf, current };
  return poly_uop(ctx, POLY_OP_AFTER, buf->dtype, after_src, 2,
                   poly_arg_none());
}

/* Apply pm_add_buffers: walk graph bottom-up, replace BUFFERIZE nodes
 * with BUFFER+STORE+END+AFTER chains.
 * If buf_dims_map is non-NULL, stores BUFFER→shape_info mapping for each
 * intermediate buffer (see bufferize_to_store_global).
 * Returns the rewritten graph (no BUFFERIZE nodes remain). */
PolyUOp *poly_apply_add_buffers(PolyCtx *ctx, PolyUOp *sink,
                                 PolyMap *buf_dims_map) {
  int n_topo;
  PolyUOp **topo = poly_toposort(ctx, sink, &n_topo);
  PolyMap *rmap = poly_map_new(n_topo < 16 ? 16 : (uint32_t)n_topo);
  int lunique_counter = 0;

  for (int t = 0; t < n_topo; t++) {
    PolyUOp *u = topo[t];
    PolyUOp *result = NULL;

    /* First, rebuild sources if any were remapped */
    bool src_changed = false;
    PolyUOp *new_src_buf[16];
    PolyUOp **new_src = (u->n_src > 16) ?
        malloc(u->n_src * sizeof(PolyUOp *)) : new_src_buf;
    for (int i = 0; i < u->n_src; i++) {
      PolyUOp *mapped = rmap_get(rmap, u->src[i]);
      new_src[i] = mapped ? mapped : u->src[i];
      if (new_src[i] != u->src[i]) src_changed = true;
    }

    if (u->op == POLY_OP_BUFFERIZE) {
      /* Create a temporary node with remapped sources for the conversion */
      PolyUOp *remapped = src_changed ?
          poly_uop(ctx, u->op, u->dtype, new_src, u->n_src, u->arg) : u;
      result = bufferize_to_store_global(ctx, &lunique_counter, remapped,
                                         buf_dims_map);
    } else if (src_changed) {
      result = poly_uop(ctx, u->op, u->dtype, new_src, u->n_src, u->arg);
    }

    if (result && result != u)
      rmap_set(rmap, u, result);

    if (new_src != new_src_buf) free(new_src);
  }

  PolyUOp *new_sink = rmap_get(rmap, sink);
  poly_map_destroy(rmap);
  return new_sink ? new_sink : sink;
}


/* ═══════════════════════════════════════════════════════════════════════
 * split_kernel_rewrite — tinygrad-aligned kernel extraction
 *
 * Port of tinygrad's to_define_global (rangeify.py:453-468).
 * Pure bottom-up substitution — no heuristic matching, no buf_dims_map:
 *   BUFFER → PARAM (sequential numbering)
 *   AFTER  → src[0] (intermediate BUFFER, which then becomes PARAM)
 *   RANGE  → RANGE (renumbered from 0)
 *   CONTIGUOUS → src[0]
 * ═══════════════════════════════════════════════════════════════════════ */

typedef struct {
  PolyCtx *ctx;
  PolyMap *remap;            /* UOp* → rewritten UOp* (cache) */
  int param_count;           /* sequential PARAM numbering (BUFFERs only) */
  int range_count;           /* sequential RANGE renumbering */
  PolyUOp **param_bufs;     /* param_bufs[i] = original BUFFER for PARAM(i) (dynamic) */
  int param_bufs_cap;        /* allocated capacity of param_bufs */
  PolyMap *buf_to_param;     /* BUFFER/DEFINE_VAR UOp* → rewritten UOp* (dedup) */
  PolyUOp **var_bufs;       /* var_bufs[i] = DEFINE_VAR UOp for var param i (dynamic) */
  int var_count;             /* number of DEFINE_VAR params */
  int var_bufs_cap;          /* allocated capacity of var_bufs */
} PolySplitCtx;

static PolyUOp *split_kernel_rewrite(PolySplitCtx *sctx, PolyUOp *u) {
  /* Check cache */
  PolyUOp *cached = poly_map_get(sctx->remap, ptr_hash(u), u, ptr_eq);
  if (cached) return cached;

  PolyCtx *ctx = sctx->ctx;

  /* AFTER(BUFFER, END_chain): only walk src[0] (the intermediate BUFFER).
   * src[1] is the producer's END chain — it belongs to the producer kernel
   * and must NOT be walked here. Walking it would allocate ghost PARAM/RANGE
   * indices for producer BUFFERs, creating non-sequential consumer params. */
  if (u->op == POLY_OP_AFTER) {
    PolyUOp *rewritten_buf = split_kernel_rewrite(sctx, u->src[0]);
    poly_map_set(sctx->remap, ptr_hash(u), u, rewritten_buf, ptr_eq);
    return rewritten_buf;
  }

  /* Recursively process sources first (bottom-up) */
  PolyUOp *new_src_buf[16];
  PolyUOp **new_src = (u->n_src > 16) ?
      malloc(u->n_src * sizeof(PolyUOp *)) : new_src_buf;
  bool changed = false;
  for (int i = 0; i < u->n_src; i++) {
    new_src[i] = split_kernel_rewrite(sctx, u->src[i]);
    if (new_src[i] != u->src[i]) changed = true;
  }

  PolyUOp *result = NULL;

  switch (u->op) {

  case POLY_OP_BUFFER: {
    /* debuf: BUFFER → PARAM with sequential numbering.
     * Reuse existing PARAM if this BUFFER was already seen. */
    PolyUOp *existing = poly_map_get(sctx->buf_to_param, ptr_hash(u), u, ptr_eq);
    if (existing) {
      result = existing;
    } else {
      PolyDType scalar = poly_dtype_scalar(u->dtype);
      int64_t numel = (u->arg.kind == POLY_ARG_INT) ? u->arg.i : 1;
      PolyDType ptr_dt = poly_dtype_ptr(scalar, numel, POLY_ADDR_GLOBAL);
      int idx = sctx->param_count;
      result = poly_uop0(ctx, POLY_OP_PARAM, ptr_dt, poly_arg_int(idx));
      poly_map_set(sctx->buf_to_param, ptr_hash(u), u, result, ptr_eq);
      if (idx >= sctx->param_bufs_cap) {
        int new_cap = sctx->param_bufs_cap ? sctx->param_bufs_cap * 2 : 8;
        void *tmp = realloc(sctx->param_bufs, new_cap * sizeof(PolyUOp *));
        assert(tmp && "OOM: param_bufs realloc");
        sctx->param_bufs = tmp;
        sctx->param_bufs_cap = new_cap;
      }
      sctx->param_bufs[idx] = u;
      sctx->param_count++;
    }
    break;
  }

  case POLY_OP_DEFINE_VAR: {
    /* Track DEFINE_VAR separately from BUFFER params.
     * DEFINE_VAR stays as-is in the kernel graph — the renderer emits it as
     * `const int N`. It's tracked in a separate var_bufs array (not param_bufs)
     * because args[] in the _call wrapper must have buffers first, then vars,
     * matching tinygrad's [globals...] + [var_vals...] convention. */
    PolyUOp *existing = poly_map_get(sctx->buf_to_param, ptr_hash(u), u, ptr_eq);
    if (existing) {
      result = existing;
    } else {
      poly_map_set(sctx->buf_to_param, ptr_hash(u), u, u, ptr_eq);
      if (sctx->var_count >= sctx->var_bufs_cap) {
        int new_cap = sctx->var_bufs_cap ? sctx->var_bufs_cap * 2 : 4;
        void *tmp = realloc(sctx->var_bufs, new_cap * sizeof(PolyUOp *));
        assert(tmp && "OOM: var_bufs realloc");
        sctx->var_bufs = tmp;
        sctx->var_bufs_cap = new_cap;
      }
      sctx->var_bufs[sctx->var_count++] = u;
      result = u;
    }
    break;
  }

  case POLY_OP_RANGE: {
    /* renumber_range: RANGE → RANGE with sequential ID from 0.
     * Preserve the bound (src[0]) and axis_type from the arg. */
    int64_t new_id = sctx->range_count++;
    int64_t axis_type = poly_range_axis_type(u->arg);
    PolyArg new_arg = poly_arg_range(new_id, axis_type);
    if (u->n_src > 0) {
      result = poly_uop1(ctx, POLY_OP_RANGE, u->dtype, new_src[0], new_arg);
    } else {
      result = poly_uop0(ctx, POLY_OP_RANGE, u->dtype, new_arg);
    }
    break;
  }

  case POLY_OP_CONTIGUOUS:
    /* CONTIGUOUS → src[0] (strip the CONTIGUOUS marker). */
    result = new_src[0];
    break;

  default:
    /* Rebuild with rewritten sources if any changed. */
    result = changed
        ? poly_uop(ctx, u->op, u->dtype, new_src, u->n_src, u->arg)
        : u;
    break;
  }

  if (new_src != new_src_buf) free(new_src);

  poly_map_set(sctx->remap, ptr_hash(u), u, result, ptr_eq);
  return result;
}

/* add_kernel_loads: post-pass that wraps non-STORE INDEX nodes with LOAD.
 *
 * After split_kernel_rewrite, the kernel graph has INDEX(PARAM, flat) nodes.
 * Store targets: STORE(INDEX(PARAM, flat), value) — INDEX stays as-is.
 * Read accesses: INDEX(PARAM, flat) consumed by ALU/REDUCE — needs LOAD.
 *
 * This is the polygrad equivalent of tinygrad's pm_add_loads. */
static PolyUOp *add_kernel_loads(PolyCtx *ctx, PolyUOp *kernel_sink) {
  int n_topo;
  PolyUOp **topo = poly_toposort(ctx, kernel_sink, &n_topo);

  /* Bottom-up rewrite: wrap ALL ptr-dtype INDEX with LOAD.
   * STORE's src[0] (store target) is never remapped, so it keeps the
   * original INDEX. This handles the case where the same INDEX UOp is used
   * as both a store target and a read operand (CSE deduplication). */
  PolyMap *rmap = poly_map_new(n_topo < 16 ? 16 : (uint32_t)n_topo);

  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    bool src_changed = false;
    PolyUOp *new_src_buf[16];
    PolyUOp **new_src = (u->n_src > 16) ?
        malloc(u->n_src * sizeof(PolyUOp *)) : new_src_buf;
    for (int j = 0; j < u->n_src; j++) {
      /* STORE's src[0] is the store target — do NOT remap (keep as INDEX). */
      if (u->op == POLY_OP_STORE && j == 0) {
        new_src[j] = u->src[j];
        continue;
      }
      PolyUOp *mapped = poly_map_get(rmap, ptr_hash(u->src[j]), u->src[j], ptr_eq);
      new_src[j] = mapped ? mapped : u->src[j];
      if (new_src[j] != u->src[j]) src_changed = true;
    }

    PolyUOp *result = NULL;

    if (u->op == POLY_OP_INDEX && u->n_src >= 2 && u->dtype.is_ptr) {
      /* Wrap INDEX with LOAD — strip pointer flag to get scalar element type. */
      PolyUOp *idx = src_changed
          ? poly_uop(ctx, u->op, u->dtype, new_src, u->n_src, u->arg)
          : u;
      PolyDType elem = u->dtype;
      elem.is_ptr = false;
      elem.ptr_size = 0;
      result = poly_uop1(ctx, POLY_OP_LOAD, elem, idx, poly_arg_none());
    } else if (src_changed) {
      result = poly_uop(ctx, u->op, u->dtype, new_src, u->n_src, u->arg);
    }

    if (result && result != u)
      poly_map_set(rmap, ptr_hash(u), u, result, ptr_eq);

    if (new_src != new_src_buf) free(new_src);
  }

  PolyUOp *new_sink = poly_map_get(rmap, ptr_hash(kernel_sink), kernel_sink, ptr_eq);
  poly_map_destroy(rmap);
  return new_sink ? new_sink : kernel_sink;
}

/* flatten_range_chain: ensure the END chain includes all NON-REDUCE RANGEs.
 *
 * Collects outer (store-loop) RANGEs from the subtree and rebuilds the END
 * chain to close every one. Reduce RANGEs are excluded — they're handled by
 * pm_reduce in the codegen pipeline (which creates its own inner RANGE/END).
 *
 * This is structurally equivalent to tinygrad's pm_flatten_range + pm_split_ends,
 * adapted for polygrad's convention of creating END chains during scheduling
 * (tinygrad defers to codegen). */
static PolyUOp *flatten_range_chain(PolyCtx *ctx, PolyUOp *node) {
  /* Unwrap existing END chain to find the STORE. */
  PolyUOp *store = node;
  while (store->op == POLY_OP_END && store->n_src >= 1)
    store = store->src[0];
  if (store->op != POLY_OP_STORE) return node;

  /* First: collect RANGEs that are REDUCE sources (handled by pm_reduce). */
  int n_topo;
  PolyUOp **topo = poly_toposort(ctx, store, &n_topo);
  PolyMap *reduce_ranges = poly_map_new(32);
  for (int i = 0; i < n_topo; i++) {
    if (topo[i]->op == POLY_OP_REDUCE) {
      for (int s = 1; s < topo[i]->n_src; s++) {
        if (topo[i]->src[s]->op == POLY_OP_RANGE)
          poly_map_set(reduce_ranges, ptr_hash(topo[i]->src[s]),
                        topo[i]->src[s], topo[i]->src[s], ptr_eq);
      }
    }
  }

  /* Collect all unique non-reduce RANGE UOps from the STORE's subtree. */
  PolyUOp *all_ranges[POLY_MAX_DIMS * 2];
  int n_ranges = 0;
  for (int i = 0; i < n_topo; i++) {
    if (topo[i]->op == POLY_OP_RANGE && n_ranges < POLY_MAX_DIMS * 2) {
      /* Skip reduce ranges */
      if (poly_map_get(reduce_ranges, ptr_hash(topo[i]),
                        topo[i], ptr_eq))
        continue;
      /* Dedup */
      bool dup = false;
      for (int k = 0; k < n_ranges; k++) {
        if (all_ranges[k] == topo[i]) { dup = true; break; }
      }
      if (!dup) all_ranges[n_ranges++] = topo[i];
    }
  }
  poly_map_destroy(reduce_ranges);

  /* Sort by axis_id for consistent END chain ordering. */
  for (int i = 0; i < n_ranges - 1; i++) {
    for (int j = i + 1; j < n_ranges; j++) {
      int64_t ai = poly_range_axis_id(all_ranges[i]->arg);
      int64_t aj = poly_range_axis_id(all_ranges[j]->arg);
      if (ai > aj) {
        PolyUOp *tmp = all_ranges[i];
        all_ranges[i] = all_ranges[j];
        all_ranges[j] = tmp;
      }
    }
  }

  if (getenv("POLY_DEBUG_FLATTEN")) {
    fprintf(stderr, "[flatten_range_chain] n_ranges=%d (non-reduce)\n", n_ranges);
    for (int i = 0; i < n_ranges; i++) {
      fprintf(stderr, "  range[%d]: axis_id=%ld bound=%ld ptr=%p\n", i,
              poly_range_axis_id(all_ranges[i]->arg),
              (all_ranges[i]->n_src > 0 && all_ranges[i]->src[0]->op == POLY_OP_CONST)
                  ? all_ranges[i]->src[0]->arg.i : -1,
              (void*)all_ranges[i]);
    }
    /* Also dump reduce ranges */
    fprintf(stderr, "  reduce_ranges:\n");
    for (int i = 0; i < n_topo; i++) {
      if (topo[i]->op == POLY_OP_REDUCE) {
        fprintf(stderr, "    REDUCE at topo[%d], n_src=%d\n", i, topo[i]->n_src);
        for (int s = 1; s < topo[i]->n_src; s++) {
          PolyUOp *rr = topo[i]->src[s];
          fprintf(stderr, "      src[%d]: %s axis_id=%ld bound=%ld ptr=%p\n", s,
                  poly_op_name(rr->op),
                  rr->op == POLY_OP_RANGE ? poly_range_axis_id(rr->arg) : -1,
                  (rr->n_src > 0 && rr->src[0]->op == POLY_OP_CONST) ? rr->src[0]->arg.i : -1,
                  (void*)rr);
        }
      }
    }
    fflush(stderr);
  }

  /* Rebuild END chain: END(END(...END(STORE, range_0)..., range_n-2), range_n-1) */
  PolyUOp *current = store;
  for (int i = 0; i < n_ranges; i++) {
    PolyUOp *end_src[2] = { current, all_ranges[i] };
    current = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2,
                         poly_arg_none());
  }
  return current;
}


/* Pre-rangeify graph normalization.
 *
 * Port of tinygrad's earliest_rewrites (rangeify.py:101-157), subset:
 *   C1: DETACH / CONTIGUOUS_BACKWARD removal (stops grad flow in backward)
 *   C2: Zero-sized reduce -> identity element (empty tensor edge case)
 *   C3: Merge adjacent RESHAPEs: RESHAPE(RESHAPE(x, s1), s2) -> RESHAPE(x, s2)
 *
 * Applied before rangeify sees the graph, so later passes can assume
 * these ops are already stripped.
 */
static PolyUOp *poly_earliest_rewrites(PolyCtx *ctx, PolyUOp *sink) {
  int n_topo;
  PolyUOp **topo = poly_toposort(ctx, sink, &n_topo);
  PolyMap *rmap = poly_map_new(n_topo < 16 ? 16 : (uint32_t)n_topo);
  bool changed = false;

  for (int t = 0; t < n_topo; t++) {
    PolyUOp *u = topo[t];

    /* Rebuild sources using remap */
    bool src_changed = false;
    PolyUOp *ns_buf[16];
    PolyUOp **ns = (u->n_src > 16) ?
        malloc(u->n_src * sizeof(PolyUOp *)) : ns_buf;
    for (int i = 0; i < u->n_src; i++) {
      PolyUOp *m = rmap_get(rmap, u->src[i]);
      ns[i] = m ? m : u->src[i];
      if (ns[i] != u->src[i]) src_changed = true;
    }

    PolyUOp *result = NULL;

    /* C1: DETACH / CONTIGUOUS_BACKWARD removal.
     * These exist after autograd (autograd.c:276 / autograd.c:299).
     * Pass through to src[0]. */
    if (u->op == POLY_OP_DETACH ||
        u->op == POLY_OP_CONTIGUOUS_BACKWARD) {
      result = ns[0];
    }

    /* C3: Merge adjacent RESHAPEs.
     * RESHAPE(RESHAPE(x, s1), s2) -> RESHAPE(x, s2).
     * Intermediate shape is irrelevant. Fires on backward graphs where
     * poly_grad's reduce_to_shape produces RESHAPE chains. */
    else if (u->op == POLY_OP_RESHAPE && u->n_src >= 1 &&
             ns[0]->op == POLY_OP_RESHAPE && ns[0]->n_src >= 1) {
      PolyUOp *inner_x = ns[0]->src[0];
      PolyUOp *new_srcs[2] = { inner_x, u->n_src >= 2 ? ns[1] : NULL };
      result = poly_uop(ctx, POLY_OP_RESHAPE, u->dtype,
                         new_srcs, u->n_src, u->arg);
    }

    /* C2: Zero-sized reduce -> identity element.
     * Only when input shape is statically known and provably zero. */
    else if (u->op == POLY_OP_REDUCE_AXIS && u->n_src >= 1) {
      /* Check if input has a zero dimension in the reduce axes */
      PolyUOp *input = ns[0];
      if (input->op == POLY_OP_CONST) {
        /* Scalar CONST has size 1, not 0 -- skip. Reduce of CONST
         * is handled by sym constant folding, not here. */
      } else if (u->arg.kind == POLY_ARG_REDUCE_AXIS) {
        /* Check shape of input -- if any reduce axis has size 0 */
        PolyShape sh = poly_uop_shape(ctx, input);
        bool has_zero = false;
        if (sh.ndim > 0) {
          int n_axes = u->arg.reduce_axis.n;
          for (int a = 0; a < n_axes; a++) {
            int axis = u->arg.reduce_axis.axes[a];
            if (axis >= 0 && axis < sh.ndim && sh.dims[axis] == 0) {
              has_zero = true;
              break;
            }
          }
        }
        if (sh.ndim > 0 && sh.dims) free(sh.dims);
        if (has_zero) {
          PolyOps rop = u->arg.reduce_axis.op;
          double ident = (rop == POLY_OP_ADD) ? 0.0 :
                         (rop == POLY_OP_MUL) ? 1.0 :
                         (rop == POLY_OP_MAX) ? -INFINITY : 0.0;
          result = poly_uop(ctx, POLY_OP_CONST, u->dtype, NULL, 0,
                            poly_arg_float(ident));
        }
      }
    }

    /* C4: ASSIGN rules (tinygrad rangeify.py:143-156).
     * C4a: Collapse nested ASSIGN to the same buffer.
     *      ASSIGN(target, ASSIGN(target, src)) → inner ASSIGN.
     * C4b: fix_assign_hazard — if value reads target through PERMUTE/FLIP,
     *      wrap value with CONTIGUOUS to force materialization.
     *      (Full-buffer ASSIGN only — target must be BUFFER-rooted.) */
    else if (u->op == POLY_OP_ASSIGN && u->n_src >= 2) {
      PolyUOp *target = ns[0];
      PolyUOp *value = ns[1];

      /* C4a: collapse nested ASSIGN */
      if (value->op == POLY_OP_ASSIGN && value->n_src >= 2 &&
          value->src[0] == target) {
        result = value;
      }

      /* C4b: fix_assign_hazard — walk value subtree, check for PERMUTE/FLIP
       * that can reach target's base BUFFER. If found, wrap value with
       * CONTIGUOUS to force it into a separate kernel before the ASSIGN. */
      if (!result) {
        /* Find target's base (walk past movement ops) */
        PolyUOp *target_base = target;
        while (target_base->n_src > 0 &&
               poly_opset_has(POLY_GROUP_MOVEMENT, target_base->op))
          target_base = target_base->src[0];

        /* Walk value subtree looking for PERMUTE/FLIP */
        int n_val_topo;
        PolyUOp **val_topo = poly_toposort(ctx, value, &n_val_topo);
        bool needs_contiguous = false;
        for (int v = 0; v < n_val_topo && !needs_contiguous; v++) {
          PolyUOp *vn = val_topo[v];
          if (is_always_contiguous(vn->op)) continue;
          if (vn->op != POLY_OP_PERMUTE && vn->op != POLY_OP_FLIP)
            continue;
          /* Check if target_base is reachable from this hazard node */
          int n_h_topo;
          PolyUOp **h_topo = poly_toposort(ctx, vn, &n_h_topo);
          for (int h = 0; h < n_h_topo; h++) {
            if (h_topo[h] == target_base) {
              needs_contiguous = true;
              break;
            }
          }
        }
        if (needs_contiguous) {
          PolyUOp *cont_val = poly_uop1(ctx, POLY_OP_CONTIGUOUS, value->dtype,
                                          value, poly_arg_none());
          PolyUOp *assign_src[2] = { target, cont_val };
          result = poly_uop(ctx, POLY_OP_ASSIGN, u->dtype, assign_src, 2,
                             u->arg);
        }
      }
    }

    if (!result && src_changed)
      result = poly_uop(ctx, u->op, u->dtype, ns, u->n_src, u->arg);

    if (result && result != u) {
      rmap_set(rmap, u, result);
      changed = true;
    }
    if (ns != ns_buf) free(ns);
  }

  PolyUOp *new_sink = changed ? rmap_get(rmap, sink) : NULL;
  poly_map_destroy(rmap);
  return new_sink ? new_sink : sink;
}


/* Stage 2: Remove dead (unused) range axes from BUFFERIZE nodes.
 *
 * For each INDEX(BUFFERIZE(val, buf_r0..buf_rN), idx_r0..idx_rN) pair:
 * - Collect RANGE UOps reachable from val's subtree.
 * - Drop position i if bufferize.src[1+i] is a RANGE NOT used by val.
 * - CONST positions always kept (preserve singleton-dim semantics).
 * - Rebuilds the INDEX+BUFFERIZE pair with matching arity.
 *
 * For the current test graph (each BUFFERIZE has 1 range, used by val),
 * this is a no-op. It runs to satisfy the pipeline contract for Stage 3.
 */
static PolyUOp *poly_cleanup_dead_bufferize_axes(PolyCtx *ctx, PolyUOp *sink) {
  int n_topo;
  PolyUOp **topo = poly_toposort(ctx, sink, &n_topo);
  PolyMap *rmap = poly_map_new(n_topo < 16 ? 16 : (uint32_t)n_topo);

  for (int t = 0; t < n_topo; t++) {
    PolyUOp *u = topo[t];

    bool src_changed = false;
    PolyUOp *new_src_buf[POLY_MAX_DIMS + 2];
    PolyUOp **new_src = ((uint32_t)u->n_src > (uint32_t)(sizeof new_src_buf / sizeof *new_src_buf))
        ? malloc(u->n_src * sizeof(PolyUOp *)) : new_src_buf;
    for (int i = 0; i < u->n_src; i++) {
      PolyUOp *m = rmap_get(rmap, u->src[i]);
      new_src[i] = m ? m : u->src[i];
      if (new_src[i] != u->src[i]) src_changed = true;
    }

    PolyUOp *result = NULL;

    if (u->op == POLY_OP_INDEX && u->n_src >= 2 &&
        new_src[0]->op == POLY_OP_BUFFERIZE) {
      PolyUOp *bufferize = new_src[0];
      PolyUOp *val = bufferize->src[0];
      int n_buf_rngs = bufferize->n_src - 1;
      int n_idx = u->n_src - 1;

      bool always_run = (val->op == POLY_OP_CONTIGUOUS ||
                         val->op == POLY_OP_COPY ||
                         val->op == POLY_OP_ASSIGN ||
                         val->op == POLY_OP_ENCDEC);

      if (!always_run && n_buf_rngs == n_idx && n_buf_rngs > 0) {
        /* Collect RANGEs reachable from val's subtree */
        int n_val_topo;
        PolyUOp **val_topo = poly_toposort(ctx, val, &n_val_topo);
        PolyMap *val_ranges = poly_map_new(n_val_topo < 8 ? 8 : (uint32_t)n_val_topo);
        for (int vi = 0; vi < n_val_topo; vi++) {
          if (val_topo[vi]->op == POLY_OP_RANGE)
            rmap_set(val_ranges, val_topo[vi], val_topo[vi]);
        }

        /* Build live mask.
         * CONSTs are dead axes (always index 0, contribute nothing).
         * RANGEs not in val's subtree are also dead.
         * Matches tinygrad cleanup_dead_axes: "CONSTs are already dead axes". */
        bool live[POLY_MAX_DIMS];
        bool any_dead = false;
        for (int d = 0; d < n_buf_rngs && d < POLY_MAX_DIMS; d++) {
          PolyUOp *br = bufferize->src[1 + d];
          if (br->op == POLY_OP_CONST) {
            live[d] = false;
            any_dead = true;
          } else {
            live[d] = (rmap_get(val_ranges, br) != NULL);
            if (!live[d]) any_dead = true;
          }
        }
        poly_map_destroy(val_ranges);

        if (any_dead) {
          PolyUOp *new_buf_src[POLY_MAX_DIMS + 1];
          PolyUOp *new_idx_src[POLY_MAX_DIMS + 1];
          int n_new = 0;
          new_buf_src[0] = val;
          for (int d = 0; d < n_buf_rngs && d < POLY_MAX_DIMS; d++) {
            if (live[d]) {
              new_buf_src[1 + n_new] = bufferize->src[1 + d];
              new_idx_src[1 + n_new] = new_src[1 + d];
              n_new++;
            }
          }
          PolyUOp *new_buf = poly_uop(ctx, POLY_OP_BUFFERIZE, bufferize->dtype,
                                       new_buf_src, n_new + 1, bufferize->arg);
          new_idx_src[0] = new_buf;
          result = poly_uop(ctx, POLY_OP_INDEX, u->dtype,
                             new_idx_src, n_new + 1, u->arg);
        }
      }
    }

    if (!result && src_changed)
      result = poly_uop(ctx, u->op, u->dtype, new_src, u->n_src, u->arg);
    if (result && result != u)
      rmap_set(rmap, u, result);
    if (new_src != new_src_buf) free(new_src);
  }

  PolyUOp *new_sink = rmap_get(rmap, sink);
  poly_map_destroy(rmap);
  return new_sink ? new_sink : sink;
}

/* Stage 3 helpers */

/* Red-gate traversal (port of tinygrad's red_gate in pm_remove_bufferize).
 *
 * Traverses root's subtree via iterative DFS with a stopping condition:
 *   - BUFFERIZE: counted as accessed buffer, traversal STOPS (sources not explored).
 *   - BUFFER, PARAM, BUFFER_VIEW: counted as accessed buffer, traversal CONTINUES.
 *   - REDUCE: added to reduces[] array, traversal CONTINUES.
 *   - Other nodes: traversal CONTINUES.
 *
 * This ensures that BUFFERIZE nodes are treated as opaque intermediate buffers —
 * their internal computations (REDUCE, BUFFER_A, etc.) are NOT traversed.
 * Without this, BUFFERIZE_2's val would see BUFFERIZE_1's internal REDUCE,
 * incorrectly triggering the buffer_in_reduce gate and preventing removal.
 *
 * Parameters:
 *   reduces[0..max_reduces-1]: output array for REDUCE UOps found
 *   n_reduces: output count of REDUCE UOps
 *   Returns: count of unique accessed buffer nodes
 */
static int poly_red_gate_collect(PolyCtx *ctx, PolyUOp *root,
                                  PolyUOp **reduces, int *n_reduces, int max_reduces) {
  (void)ctx; /* arena not needed here — iterative DFS uses malloc/free */

  int cap = 64;
  PolyUOp **stack = malloc(cap * sizeof(PolyUOp *));
  if (!stack) { *n_reduces = 0; return 0; }
  int top = 0;

  PolyMap *visited = poly_map_new(64);
  int accessed = 0;
  *n_reduces = 0;

  stack[top++] = root;
  while (top > 0) {
    PolyUOp *u = stack[--top];
    if (rmap_get(visited, u)) continue;
    rmap_set(visited, u, u);

    if (u->op == POLY_OP_BUFFERIZE) {
      accessed++; /* count it, but DO NOT recurse into its sources */
      continue;
    }
    if (u->op == POLY_OP_BUFFER || u->op == POLY_OP_PARAM ||
        u->op == POLY_OP_BUFFER_VIEW) {
      accessed++; /* count it, recurse normally */
    }
    if (u->op == POLY_OP_REDUCE && *n_reduces < max_reduces) {
      reduces[(*n_reduces)++] = u;
    }

    /* Push sources in reverse order for consistent DFS ordering */
    for (int i = u->n_src - 1; i >= 0; i--) {
      PolyUOp *src = u->src[i];
      if (!rmap_get(visited, src)) {
        if (top >= cap) {
          cap *= 2;
          stack = realloc(stack, cap * sizeof(PolyUOp *));
          if (!stack) { poly_map_destroy(visited); return accessed; }
        }
        stack[top++] = src;
      }
    }
  }

  free(stack);
  poly_map_destroy(visited);
  return accessed;
}

/* Substitute RANGE UOps in val's subgraph per sub_map, return rewritten val. */
static PolyUOp *poly_substitute_ranges(PolyCtx *ctx, PolyUOp *val, PolyMap *sub_map) {
  int n;
  PolyUOp **topo = poly_toposort(ctx, val, &n);
  PolyMap *rmap = poly_map_new(n < 8 ? 8 : (uint32_t)n);

  for (int t = 0; t < n; t++) {
    PolyUOp *u = topo[t];

    /* Direct substitution target? */
    PolyUOp *sub = rmap_get(sub_map, u);
    if (sub && sub != u) {
      rmap_set(rmap, u, sub);
      continue;
    }

    /* Rebuild with remapped sources */
    bool src_changed = false;
    PolyUOp *ns_buf[POLY_MAX_DIMS + 2];
    PolyUOp **ns = ((uint32_t)u->n_src > (uint32_t)(sizeof ns_buf / sizeof *ns_buf))
        ? malloc(u->n_src * sizeof(PolyUOp *)) : ns_buf;
    for (int i = 0; i < u->n_src; i++) {
      PolyUOp *m = rmap_get(rmap, u->src[i]);
      ns[i] = m ? m : u->src[i];
      if (ns[i] != u->src[i]) src_changed = true;
    }
    if (src_changed) {
      PolyUOp *rebuilt = poly_uop(ctx, u->op, u->dtype, ns, u->n_src, u->arg);
      if (rebuilt != u) rmap_set(rmap, u, rebuilt);
    }
    if (ns != ns_buf) free(ns);
  }

  PolyUOp *result = rmap_get(rmap, val);
  poly_map_destroy(rmap);
  return result ? result : val;
}

/* Stage 3: pm_remove_bufferize — remove BUFFERIZEs that don't need materialization.
 *
 * For each INDEX(BUFFERIZE(val, buf_r0..rN), idx_r0..rN) with matching arity,
 * check 4 gates. If all pass: substitute val's buf ranges with idx ranges,
 * replacing the INDEX(BUFFERIZE(...)) node with the inlined val expression.
 *
 * Gates (any fires → keep BUFFERIZE):
 *   1. ALWAYS_RUN: val->op in {CONTIGUOUS, COPY, ASSIGN, ENCDEC}
 *   2. removable flag: bufferize->arg.i == 0 (stored at creation time)
 *   3. accessed_buffers: count distinct {BUFFER,PARAM,BUFFERIZE,BUFFER_VIEW} in val > 3
 *   4. buffer_in_reduce: any REDUCE in val has a buffer op under its value operand
 *
 * Corresponds to tinygrad's pm_remove_bufferize (PCONTIG<=2 path).
 */
static PolyUOp *poly_remove_bufferize(PolyCtx *ctx, PolyUOp *sink) {
  int n_topo;
  PolyUOp **topo = poly_toposort(ctx, sink, &n_topo);
  PolyMap *rmap = poly_map_new(n_topo < 16 ? 16 : (uint32_t)n_topo);

  bool dbg_all = getenv("POLY_DEBUG_REMOVE") != NULL;
  for (int t = 0; t < n_topo; t++) {
    PolyUOp *u = topo[t];

    if (dbg_all && u->op == POLY_OP_INDEX) {
      fprintf(stderr, "[remove_bufferize] topo INDEX ptr=%p n_src=%d src0_op=%s\n",
              (void*)u, u->n_src, u->n_src > 0 ? poly_op_name(u->src[0]->op) : "none");
      fflush(stderr);
    }

    bool src_changed = false;
    PolyUOp *ns_buf[POLY_MAX_DIMS + 2];
    PolyUOp **ns = ((uint32_t)u->n_src > (uint32_t)(sizeof ns_buf / sizeof *ns_buf))
        ? malloc(u->n_src * sizeof(PolyUOp *)) : ns_buf;
    for (int i = 0; i < u->n_src; i++) {
      PolyUOp *m = rmap_get(rmap, u->src[i]);
      ns[i] = m ? m : u->src[i];
      if (ns[i] != u->src[i]) src_changed = true;
    }

    PolyUOp *result = NULL;

    if (u->op == POLY_OP_INDEX && u->n_src >= 1 &&
        ns[0]->op == POLY_OP_BUFFERIZE) {
      PolyUOp *bufferize = ns[0];
      int n_buf_rngs = bufferize->n_src - 1;
      int n_idx = u->n_src - 1;

      bool dbg = getenv("POLY_DEBUG_REMOVE") != NULL;

      if (dbg) {
        fprintf(stderr, "[remove_bufferize] INDEX ptr=%p n_idx=%d bufferize_ptr=%p n_buf_rngs=%d arity_match=%d\n",
                (void*)u, n_idx, (void*)bufferize, n_buf_rngs, n_idx == n_buf_rngs);
        fflush(stderr);
      }

      if (n_idx == n_buf_rngs) {
        PolyUOp *val = bufferize->src[0];

        /* A2: Noop BUFFERIZE -- INDEX and BUFFERIZE have identical range
         * sources, so the BUFFERIZE is an identity reindex.
         * Requires n_idx > 0: a 0-range BUFFERIZE (scalar materialization)
         * is NOT a noop — it needs the cost gates (buffer_in_reduce etc).
         * Matches tinygrad rangeify.py:255-258 (remove_noop_bufferize). */
        {
          bool noop = (n_idx > 0);
          for (int d = 0; d < n_idx; d++) {
            if (ns[1 + d] != bufferize->src[1 + d]) { noop = false; break; }
          }
          if (noop) {
            if (dbg) {
              fprintf(stderr, "[remove_bufferize]   noop removed (identical ranges)\n");
              fflush(stderr);
            }
            result = val;
          }
        }

        /* A1: CONST-through-BUFFERIZE -- constant doesn't need a buffer.
         * Matches tinygrad rangeify.py:268 (pm_const_buffer_folding). */
        if (!result && val->op == POLY_OP_CONST) {
          PolyDType scalar_dt = poly_dtype_scalar(u->dtype);
          result = poly_uop(ctx, POLY_OP_CONST, scalar_dt, NULL, 0, val->arg);
          if (dbg) {
            fprintf(stderr, "[remove_bufferize]   CONST-through-BUFFERIZE folded\n");
            fflush(stderr);
          }
        }

        if (!result) {
        /* Gate 1: ALWAYS_RUN ops always materialize */
        bool always_run = (val->op == POLY_OP_CONTIGUOUS ||
                           val->op == POLY_OP_COPY ||
                           val->op == POLY_OP_ASSIGN ||
                           val->op == POLY_OP_ENCDEC);

        /* Gate 2: removable flag stored in BUFFERIZE arg */
        bool removable = (bufferize->arg.kind == POLY_ARG_INT &&
                          bufferize->arg.i == 1);

        /* B: THREEFRY bypass -- RNG ops skip cost gates and are always
         * inlined. Buffering RNG state breaks determinism when consumers
         * execute in different order.
         * Matches tinygrad rangeify.py:200. */
        bool skip_cost_gates = (val->op == POLY_OP_THREEFRY);

        bool buf_count_ok = true;
        bool buf_in_reduce = false;
        int accessed = 0, n_reduces = 0;

        if (!skip_cost_gates) {
        /* Gates 3 and 4: traverse val's subtree with red_gate (stopping at BUFFERIZE).
         * Mirrors tinygrad's accessed_buffers + buffer_in_reduce logic exactly.
         * Stops at BUFFERIZE so we don't traverse into nested intermediates' internals. */
        PolyUOp *reduces[POLY_MAX_DIMS];
        accessed = poly_red_gate_collect(ctx, val,
                                         reduces, &n_reduces, POLY_MAX_DIMS);

        /* Gate 3: too many accessed buffer nodes -> fusing not worth it */
        buf_count_ok = (accessed <= 3);

        /* Gate 4: any REDUCE in val's subtree reads from a buffer op under its
         * value operand (src[0]) -> must keep materialization boundary */
        for (int ri = 0; ri < n_reduces && !buf_in_reduce; ri++) {
          PolyUOp *reduce = reduces[ri];
          if (reduce->n_src >= 1) {
            int nv;
            PolyUOp **vtopo = poly_toposort(ctx, reduce->src[0], &nv);
            for (int j = 0; j < nv; j++) {
              PolyOps op = vtopo[j]->op;
              if (op == POLY_OP_BUFFER || op == POLY_OP_PARAM ||
                  op == POLY_OP_BUFFERIZE || op == POLY_OP_BUFFER_VIEW) {
                buf_in_reduce = true;
                break;
              }
            }
          }
        }
        } /* !skip_cost_gates */

        if (dbg) {
          fprintf(stderr, "[remove_bufferize]   val_op=%s always_run=%d removable=%d (arg.kind=%d arg.i=%lld) accessed=%d buf_count_ok=%d buf_in_reduce=%d n_reduces=%d skip_cost=%d\n",
                  poly_op_name(val->op), always_run, removable,
                  bufferize->arg.kind, (long long)bufferize->arg.i,
                  accessed, buf_count_ok, buf_in_reduce, n_reduces, skip_cost_gates);
          fflush(stderr);
        }

        if (!always_run && removable && buf_count_ok && !buf_in_reduce) {
          /* All gates pass: inline val by substituting buf ranges -> idx ranges */
          PolyMap *sub_map = poly_map_new(n_idx < 8 ? 8 : (uint32_t)n_idx);
          for (int d = 0; d < n_idx && d < n_buf_rngs; d++) {
            PolyUOp *buf_r = bufferize->src[1 + d];
            PolyUOp *idx_r = ns[1 + d];
            if (buf_r->op != POLY_OP_CONST)
              rmap_set(sub_map, buf_r, idx_r);
          }
          result = poly_substitute_ranges(ctx, val, sub_map);
          poly_map_destroy(sub_map);
        }
        } /* !result: A2/A1 didn't fire */
      }
    }

    if (!result && src_changed)
      result = poly_uop(ctx, u->op, u->dtype, ns, u->n_src, u->arg);
    if (result && result != u)
      rmap_set(rmap, u, result);
    if (ns != ns_buf) free(ns);
  }

  PolyUOp *new_sink = rmap_get(rmap, sink);
  poly_map_destroy(rmap);
  return new_sink ? new_sink : sink;
}

/* ── Stage 3.25: poly_limit_bufs helpers ─────────────────────────────── */

/* Count distinct buffer-like nodes reachable from root via gated DFS.
 * Gate set = counted set = {BUFFERIZE, AFTER, BUFFER, PARAM, DEFINE_VAR}.
 * Stops traversal at counted nodes (they are binding roots).
 * Port of tinygrad's gate_input in limit_bufs (rangeify.py:313-316). */
static int poly_count_buf_like_nodes(PolyCtx *ctx, PolyUOp *root) {
  (void)ctx;
  int cap = 32;
  PolyUOp **stack = malloc(cap * sizeof(PolyUOp *));
  if (!stack) return 0;
  int top = 0;

  PolyMap *visited = poly_map_new(64);
  int count = 0;

  stack[top++] = root;
  while (top > 0) {
    PolyUOp *u = stack[--top];
    if (rmap_get(visited, u)) continue;
    rmap_set(visited, u, u);

    bool is_gate = (u->op == POLY_OP_BUFFERIZE || u->op == POLY_OP_AFTER ||
                    u->op == POLY_OP_BUFFER || u->op == POLY_OP_PARAM ||
                    u->op == POLY_OP_DEFINE_VAR);
    if (is_gate) {
      count++;
      continue; /* do NOT recurse into sources of gate nodes */
    }

    for (int i = u->n_src - 1; i >= 0; i--) {
      if (!rmap_get(visited, u->src[i])) {
        if (top >= cap) {
          cap *= 2;
          stack = realloc(stack, cap * sizeof(PolyUOp *));
          if (!stack) { poly_map_destroy(visited); return count; }
        }
        stack[top++] = u->src[i];
      }
    }
  }

  free(stack);
  poly_map_destroy(visited);
  return count;
}

/* Collect live RANGE UOps reachable from root, in first-seen DFS order.
 * Stops traversal at BUFFERIZE boundaries (which "end" their ranges).
 * Does NOT enter REDUCE range sources (src[1:]) — only the value (src[0]).
 * Returns count written to ranges_out[0..max_ranges-1]. */
static int poly_collect_live_ranges(PolyCtx *ctx, PolyUOp *root,
                                     PolyUOp **ranges_out, int max_ranges) {
  (void)ctx;
  int cap = 32;
  PolyUOp **stack = malloc(cap * sizeof(PolyUOp *));
  if (!stack) return 0;
  int top = 0;

  PolyMap *visited = poly_map_new(64);
  int n_ranges = 0;

  stack[top++] = root;
  while (top > 0) {
    PolyUOp *u = stack[--top];
    if (rmap_get(visited, u)) continue;
    rmap_set(visited, u, u);

    if (u->op == POLY_OP_RANGE) {
      if (n_ranges < max_ranges)
        ranges_out[n_ranges++] = u;
      /* Still recurse into RANGE's src[0] (the bound), but that's
       * typically a CONST leaf — no further ranges there. */
    }

    if (u->op == POLY_OP_BUFFERIZE) {
      continue; /* BUFFERIZE ends its ranges — do not recurse */
    }

    /* For REDUCE, only recurse into src[0] (value operand).
     * src[1:] are the reduce ranges that are "ended" by the REDUCE. */
    int n_children = (u->op == POLY_OP_REDUCE && u->n_src > 1) ? 1 : u->n_src;

    for (int i = n_children - 1; i >= 0; i--) {
      if (!rmap_get(visited, u->src[i])) {
        if (top >= cap) {
          cap *= 2;
          stack = realloc(stack, cap * sizeof(PolyUOp *));
          if (!stack) { poly_map_destroy(visited); return n_ranges; }
        }
        stack[top++] = u->src[i];
      }
    }
  }

  free(stack);
  poly_map_destroy(visited);
  return n_ranges;
}

/* Check if an op is in the buffer-like gate set (same as poly_count_buf_like_nodes). */
static bool poly_is_buf_gate_op(PolyOps op) {
  return op == POLY_OP_BUFFERIZE || op == POLY_OP_AFTER ||
         op == POLY_OP_BUFFER || op == POLY_OP_PARAM ||
         op == POLY_OP_DEFINE_VAR;
}

/* Check if an op is Elementwise (ALU + CAST + BITCAST).
 * Matches tinygrad's GroupOp.Elementwise. */
static bool poly_is_elementwise_op(PolyOps op) {
  return poly_opset_has(POLY_GROUP_ELEMENTWISE, op);
}

/* Stage 3.25: poly_limit_bufs — enforce per-kernel buffer count limits.
 *
 * Port of tinygrad's pm_limit_bufs (rangeify.py:306-328).
 * For each Binary/Ternary node with too many buffer-like descendants,
 * wraps Elementwise sources in BUFFERIZE+INDEX to force materialization.
 *
 * Must run after poly_remove_bufferize and before poly_flatten_bufferize_indices
 * (within the multi-range rewrite window).
 *
 * ictx->max_kernel_bufs: 0 = disabled (noop), >0 = limit.
 * ictx->next_range_id: consumed to create fresh LOOP ranges. */
static PolyUOp *poly_limit_bufs(PolyCtx *ctx, PolyIndexingCtx *ictx, PolyUOp *sink) {
  if (ictx->max_kernel_bufs <= 0) return sink;

  int n_topo;
  PolyUOp **topo = poly_toposort(ctx, sink, &n_topo);
  PolyMap *rmap = poly_map_new(n_topo < 16 ? 16 : (uint32_t)n_topo);

  int max_bufs = ictx->max_kernel_bufs;

  for (int t = 0; t < n_topo; t++) {
    PolyUOp *u = topo[t];

    /* Rebuild sources from rmap (standard bottom-up pattern) */
    bool src_changed = false;
    PolyUOp *ns_buf[POLY_MAX_DIMS + 4];
    PolyUOp **ns = ((uint32_t)u->n_src > (uint32_t)(sizeof ns_buf / sizeof *ns_buf))
        ? malloc(u->n_src * sizeof(PolyUOp *)) : ns_buf;
    for (int i = 0; i < u->n_src; i++) {
      PolyUOp *m = rmap_get(rmap, u->src[i]);
      ns[i] = m ? m : u->src[i];
      if (ns[i] != u->src[i]) src_changed = true;
    }

    PolyUOp *result = NULL;

    /* Only check Binary/Ternary ALU ops */
    bool is_bin = poly_opset_has(POLY_GROUP_BINARY, u->op);
    bool is_tern = poly_opset_has(POLY_GROUP_TERNARY, u->op);

    if (is_bin || is_tern) {
      /* Build a temporary node with remapped sources for counting */
      PolyUOp *check_node = src_changed
          ? poly_uop(ctx, u->op, u->dtype, ns, u->n_src, u->arg) : u;

      int buf_count = poly_count_buf_like_nodes(ctx, check_node);

      if (buf_count > max_bufs - 1) {
        /* Over limit — wrap Elementwise sources in BUFFERIZE+INDEX */
        bool any_wrapped = false;
        for (int i = 0; i < u->n_src; i++) {
          PolyUOp *s = ns[i];

          /* Skip load roots — wrapping won't reduce count */
          if (poly_is_buf_gate_op(s->op)) continue;

          /* Skip non-Elementwise — REDUCE etc. are already materialization boundaries */
          if (!poly_is_elementwise_op(s->op)) continue;

          /* Collect live ranges from this source */
          PolyUOp *orig_rngs[POLY_MAX_DIMS];
          int n_rngs = poly_collect_live_ranges(ctx, s, orig_rngs, POLY_MAX_DIMS);

          /* Skip scalar expressions (0 ranges) — wrapping costs more than it saves */
          if (n_rngs == 0) continue;

          /* Create fresh LOOP ranges with same bounds */
          PolyUOp *end_rngs[POLY_MAX_DIMS];
          for (int d = 0; d < n_rngs; d++) {
            PolyUOp *bound = (orig_rngs[d]->n_src > 0) ? orig_rngs[d]->src[0] : NULL;
            if (!bound) continue; /* safety: shouldn't happen for valid RANGE */
            end_rngs[d] = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound,
                                     poly_arg_range(ictx->next_range_id++, POLY_AXIS_LOOP));
          }

          /* Substitute orig ranges → end ranges in s */
          PolyMap *sub_map = poly_map_new(n_rngs < 8 ? 8 : (uint32_t)n_rngs);
          for (int d = 0; d < n_rngs; d++)
            rmap_set(sub_map, orig_rngs[d], end_rngs[d]);
          PolyUOp *sub_s = poly_substitute_ranges(ctx, s, sub_map);
          poly_map_destroy(sub_map);

          /* Build BUFFERIZE(sub_s, end_r0, ..., end_rN).
           * Non-removable (arg=0) — these boundaries are load-bearing. */
          PolyUOp *buf_src[POLY_MAX_DIMS + 1];
          buf_src[0] = sub_s;
          for (int d = 0; d < n_rngs; d++)
            buf_src[1 + d] = end_rngs[d];
          PolyUOp *bufferize = poly_uop(ctx, POLY_OP_BUFFERIZE,
                                         poly_dtype_scalar(u->dtype),
                                         buf_src, n_rngs + 1,
                                         poly_arg_int(0)); /* removable=false */

          /* Compute ptr_dt for INDEX (same pattern as Stage 1.5) */
          int64_t numel = 1;
          for (int d = 0; d < n_rngs; d++) {
            PolyUOp *bound = orig_rngs[d]->src[0];
            if (bound->op == POLY_OP_CONST)
              numel *= bound->arg.i;
          }
          PolyDType ptr_dt = poly_dtype_ptr(poly_dtype_scalar(u->dtype),
                                             numel, POLY_ADDR_GLOBAL);

          /* Build INDEX(bufferize, orig_r0, ..., orig_rN) in multi-range form */
          PolyUOp *idx_src[POLY_MAX_DIMS + 1];
          idx_src[0] = bufferize;
          for (int d = 0; d < n_rngs; d++)
            idx_src[1 + d] = orig_rngs[d];
          ns[i] = poly_uop(ctx, POLY_OP_INDEX, ptr_dt,
                            idx_src, n_rngs + 1, poly_arg_none());
          any_wrapped = true;
        }

        if (any_wrapped) {
          result = poly_uop(ctx, u->op, u->dtype, ns, u->n_src, u->arg);
        }
      }
    }

    if (!result && src_changed)
      result = poly_uop(ctx, u->op, u->dtype, ns, u->n_src, u->arg);
    if (result && result != u)
      rmap_set(rmap, u, result);
    if (ns != ns_buf) free(ns);
  }

  PolyUOp *new_sink = rmap_get(rmap, sink);
  poly_map_destroy(rmap);
  return new_sink ? new_sink : sink;
}

/* Stage 3.5: Flatten multi-range INDEX(BUFFERIZE) back to flat-scalar form.
 *
 * After poly_apply_rangeify (with add_buffer_indices=true), every BUFFERIZE
 * consumer INDEX has the multi-range form:
 *   INDEX(BUFFERIZE(val, buf_r0..buf_rN), idx_r0..idx_rN)  -- (N+1)-source, ptr_dt
 *
 * This form exists in the "rewrite window" for Stage 3 (pm_remove_bufferize).
 * Before poly_apply_add_buffers we must flatten survivors back to:
 *   INDEX(BUFFERIZE(val, buf_r0..buf_rN), flat_scalar)  -- 2-source, ptr_dt
 *
 * split_kernel_rewrite and add_kernel_loads require flat-scalar INDEX.
 *
 * Without Stage 3 this pass is behavior-neutral: it un-flattens then re-flattens.
 * With Stage 3, only BUFFERIZEs that were NOT removed reach this pass.
 */
static PolyUOp *poly_flatten_bufferize_indices(PolyCtx *ctx, PolyUOp *sink) {
  int n_topo;
  PolyUOp **topo = poly_toposort(ctx, sink, &n_topo);
  PolyMap *rmap = poly_map_new(n_topo < 16 ? 16 : (uint32_t)n_topo);

  for (int t = 0; t < n_topo; t++) {
    PolyUOp *u = topo[t];

    /* Rebuild sources using remap */
    bool src_changed = false;
    PolyUOp *new_src_buf[16];
    PolyUOp **new_src = (u->n_src > 16) ?
        malloc(u->n_src * sizeof(PolyUOp *)) : new_src_buf;
    for (int i = 0; i < u->n_src; i++) {
      PolyUOp *mapped = rmap_get(rmap, u->src[i]);
      new_src[i] = mapped ? mapped : u->src[i];
      if (new_src[i] != u->src[i]) src_changed = true;
    }

    PolyUOp *result = NULL;

    /* Flatten: INDEX(BUFFERIZE, r0..rN) → INDEX(BUFFERIZE, flat_scalar)
     * Handles both multi-range (n_idx > 0) and 0-index (n_idx == 0) forms.
     * Stage 2 may drop all range axes from a BUFFERIZE (e.g. when the store-loop
     * RANGE differs from the reduce-loop RANGE that appears in val's subtree),
     * producing INDEX(BUFFERIZE(val)) with n_src=1. Stage 3 removes such
     * BUFFERIZEs, leaving arity-1 INDEX(surviving_BUFFERIZE) — we must flatten
     * those too, inserting CONST(0) as the flat index for the scalar case. */
    if (u->op == POLY_OP_INDEX &&
        u->n_src >= 1 &&
        new_src[0]->op == POLY_OP_BUFFERIZE) {
      PolyUOp *bufferize = new_src[0];
      int n_idx = u->n_src - 1;
      int n_buf_rngs = bufferize->n_src - 1;

      /* Only flatten when arities match (multi-range form from Stage 1.5).
       * Arity mismatch means the INDEX was already flat (legacy path). */
      if (n_idx == n_buf_rngs) {
        PolyUOp *flat_idx;
        PolyDType ptr_dt;

        if (n_idx == 0) {
          /* 0-index case: BUFFERIZE has 0 ranges (scalar output after Stage 2).
           * INDEX(BUFFERIZE(val)) → INDEX(BUFFERIZE(val), CONST(0)) */
          flat_idx = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
          ptr_dt = poly_dtype_ptr(poly_dtype_scalar(bufferize->dtype),
                                   1, POLY_ADDR_GLOBAL);
        } else {
          /* Multi-range case: extract consumer ranges and compute flat index.
           * Extract bounds as UOp* (supports both CONST and DEFINE_VAR bounds). */
          PolyUOp *idx_rngs[POLY_MAX_DIMS];
          PolyUOp *bounds[POLY_MAX_DIMS];
          bool all_const = true;
          int64_t numel = 1;
          for (int d = 0; d < n_idx && d < POLY_MAX_DIMS; d++) {
            idx_rngs[d] = new_src[1 + d];
            PolyUOp *rng = bufferize->src[1 + d];
            /* Extract bound from RANGE node (src[0] is the bound).
             * After control flow, RANGE may have extra sources; only src[0] is bound. */
            if (rng->op == POLY_OP_RANGE && rng->n_src > 0)
              bounds[d] = rng->src[0];
            else
              bounds[d] = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(1));
            if (bounds[d]->op == POLY_OP_CONST)
              numel *= bounds[d]->arg.i;
            else
              all_const = false;
          }

          flat_idx = poly_compute_flat_index_symbolic(ctx, idx_rngs, bounds, n_idx);
          ptr_dt = poly_dtype_ptr(poly_dtype_scalar(bufferize->dtype),
                                   all_const ? numel : -1, POLY_ADDR_GLOBAL);
        }

        result = poly_uop2(ctx, POLY_OP_INDEX, ptr_dt, bufferize, flat_idx,
                            poly_arg_none());
      }
    }

    if (!result && src_changed)
      result = poly_uop(ctx, u->op, u->dtype, new_src, u->n_src, u->arg);

    if (result && result != u)
      rmap_set(rmap, u, result);

    if (new_src != new_src_buf) free(new_src);
  }

  PolyUOp *new_sink = rmap_get(rmap, sink);
  poly_map_destroy(rmap);
  return new_sink ? new_sink : sink;
}

/* Schedule multi-kernel IR from tensor SINK.
 * Produces PolyScheduleResult with per-kernel SINKs, param mappings,
 * intermediates, and execution order. */
static PolyScheduleResult schedule_v2_new(PolyCtx *ctx, PolyUOp *tensor_sink) {
  PolyScheduleResult result;
  memset(&result, 0, sizeof(result));

  if (tensor_sink->op != POLY_OP_SINK) {
    fprintf(stderr, "polygrad: schedule_v2_new: expected SINK\n");
    return result;
  }

  /* 0. Pre-rangeify graph normalization: strip DETACH/CONTIGUOUS_BACKWARD,
   * merge adjacent RESHAPEs, handle zero-sized reduces.
   * Matches tinygrad's earliest_rewrites (rangeify.py:101-157). */
  tensor_sink = poly_earliest_rewrites(ctx, tensor_sink);

  /* 1. Global rangeify (with flat INDEX wrapping for BUFFER sources) */
  PolyIndexingCtx *ictx = poly_indexing_ctx_new(ctx);
  ictx->add_buffer_indices = true;
  poly_realize_map_build(ictx, tensor_sink);
  poly_range_propagate(ictx, tensor_sink);
  PolyUOp *rangeified = poly_apply_rangeify(ictx, tensor_sink);

  /* Stage 0: Diagnostic dump of BUFFERIZE/INDEX structure post-rangeify.
   * Enabled by POLY_DEBUG_BUFFERIZE env var. Run against shared_scalar_reduce_branches
   * to confirm: flat vs axis-wise INDEX, and BUFFERIZE dedup (1 vs 2 pointers). */
  if (getenv("POLY_DEBUG_BUFFERIZE")) {
    int n_diag;
    PolyUOp **diag_topo = poly_toposort(ctx, rangeified, &n_diag);
    /* Collect distinct BUFFERIZE pointers for dedup check */
    PolyUOp *seen_bufs[64];
    int n_seen = 0;
    for (int i = 0; i < n_diag; i++) {
      PolyUOp *u = diag_topo[i];
      if (u->op != POLY_OP_BUFFERIZE) continue;
      bool already = false;
      for (int k = 0; k < n_seen; k++)
        if (seen_bufs[k] == u) { already = true; break; }
      if (!already && n_seen < 64) seen_bufs[n_seen++] = u;
      int n_rngs = u->n_src - 1;
      /* Compute shape = product of range bounds */
      int64_t shape = 1;
      for (int d = 0; d < n_rngs; d++) {
        PolyUOp *rng = u->src[1 + d];
        if (rng->op == POLY_OP_RANGE && rng->n_src > 0 &&
            rng->src[0]->op == POLY_OP_CONST)
          shape *= rng->src[0]->arg.i;
      }
      fprintf(stderr, "[BUFFERIZE] ptr=%p src0=%s n_rngs=%d shape=%lld\n",
              (void *)u, poly_op_name(u->src[0]->op), n_rngs, (long long)shape);
    }
    fprintf(stderr, "[BUFFERIZE] distinct_ptrs=%d\n", n_seen);
    for (int i = 0; i < n_diag; i++) {
      PolyUOp *u = diag_topo[i];
      if (u->op != POLY_OP_INDEX) continue;
      if (u->src[0]->op != POLY_OP_BUFFERIZE) continue;
      /* src[1] is RANGE or CONST → axis-wise; otherwise → flat (ALU chain) */
      bool axis_wise = (u->n_src >= 2 && (u->src[1]->op == POLY_OP_RANGE ||
                                           u->src[1]->op == POLY_OP_CONST));
      fprintf(stderr, "[INDEX(BUFFERIZE)] ptr=%p n_src=%d form=%s src1_op=%s\n",
              (void *)u, u->n_src, axis_wise ? "axis-wise" : "flat",
              u->n_src >= 2 ? poly_op_name(u->src[1]->op) : "none");
    }
    fflush(stderr);
  }

  /* Stage 2: Remove dead (unused) range axes from BUFFERIZE+INDEX pairs. */
  PolyUOp *cleaned = poly_cleanup_dead_bufferize_axes(ctx, rangeified);

  /* Stage 3: Remove BUFFERIZEs that don't need materialization (pm_remove_bufferize).
   * Uses multi-range INDEX arity-equality match enabled by Stage 1.5. */
  PolyUOp *removed = poly_remove_bufferize(ctx, cleaned);

  /* Stage 3.25: Limit buffer count per kernel (pm_limit_bufs).
   * Only active when max_kernel_bufs > 0 (set via POLY_MAX_KERNEL_BUFFERS
   * env var or programmatically on ictx). Wraps Elementwise sources of
   * over-limit Binary/Ternary ops in non-removable BUFFERIZE+INDEX pairs. */
  PolyUOp *limited = poly_limit_bufs(ctx, ictx, removed);

  /* Stage 3.5: Flatten surviving multi-range INDEX(BUFFERIZE) back to flat-scalar form.
   * Required before add_buffers because split_kernel_rewrite and add_kernel_loads
   * expect INDEX(PARAM/BUFFERIZE, flat_scalar), not multi-range INDEX. */
  PolyUOp *flattened = poly_flatten_bufferize_indices(ctx, limited);

  /* 2. add_buffers: BUFFERIZE → BUFFER+STORE+END+AFTER.
   * No buf_dims_map needed — flat INDEX is computed during apply_rangeify. */
  PolyUOp *abuf_sink = poly_apply_add_buffers(ctx, flattened, NULL);

  /* 3. Collect AFTER nodes from post-add_buffers graph.
   * Two kinds: intermediate (BUFFER with LUNIQUE) and ASSIGN (existing BUFFER).
   * Both produce kernels. Only intermediates get buffer allocation. */
  int n_topo;
  PolyUOp **topo = poly_toposort(ctx, abuf_sink, &n_topo);
  PolyUOp **after_nodes = NULL;
  bool *after_is_intermediate = NULL;
  int n_after = 0, after_cap = 0;
  for (int i = 0; i < n_topo; i++) {
    if (topo[i]->op == POLY_OP_AFTER && topo[i]->n_src >= 2 &&
        topo[i]->src[0]->op == POLY_OP_BUFFER) {
      if (n_after >= after_cap) {
        after_cap = after_cap ? after_cap * 2 : 8;
        void *tmp = realloc(after_nodes, after_cap * sizeof(PolyUOp *));
        assert(tmp && "OOM: after_nodes realloc");
        after_nodes = tmp;
        void *tmp2 = realloc(after_is_intermediate, after_cap * sizeof(bool));
        assert(tmp2 && "OOM: after_is_intermediate realloc");
        after_is_intermediate = tmp2;
      }
      after_nodes[n_after] = topo[i];
      /* Intermediate: BUFFER has LUNIQUE child. ASSIGN: existing BUFFER (no LUNIQUE). */
      after_is_intermediate[n_after] = (topo[i]->src[0]->n_src >= 1 &&
                                         topo[i]->src[0]->src[0]->op == POLY_OP_LUNIQUE);
      n_after++;
    }
  }

  /* Count real intermediates (not ASSIGN AFTERs) */
  int n_intermediates = 0;
  for (int a = 0; a < n_after; a++)
    if (after_is_intermediate[a]) n_intermediates++;

  /* Count consumer SINK sources that are STOREs (not ASSIGN).
   * ASSIGN SINK sources are handled as AFTER producer kernels. */
  int n_sink_src = tensor_sink->n_src;
  int n_consumer_stores = 0;
  for (int i = 0; i < n_sink_src; i++) {
    PolyUOp *s = tensor_sink->src[i];
    if (s->op == POLY_OP_STORE) n_consumer_stores++;
    /* ASSIGN sources become AFTERs, handled above */
  }

  int total_kernels = n_after + n_consumer_stores;

  if (getenv("POLY_DEBUG_FLATTEN")) {
    fprintf(stderr, "[schedule_v2_new] n_after=%d (inter=%d assign=%d) n_consumer=%d total=%d\n",
            n_after, n_intermediates, n_after - n_intermediates, n_consumer_stores, total_kernels);
    fflush(stderr);
  }

  /* 4. Allocate result arrays */
  result.kernels = malloc(total_kernels * sizeof(PolyUOp *));
  result.param_to_buf = calloc(total_kernels, sizeof(PolyUOp **));
  result.kernel_n_params = calloc(total_kernels, sizeof(int));
  result.var_to_buf = calloc(total_kernels, sizeof(PolyUOp **));
  result.kernel_n_vars = calloc(total_kernels, sizeof(int));
  result.n_intermediates = n_intermediates;

  if (n_intermediates > 0) {
    result.intermediate_sizes = malloc(n_intermediates * sizeof(int64_t));
    result.intermediate_itemsizes = malloc(n_intermediates * sizeof(int));
    result.intermediate_buf_uops = malloc(n_intermediates * sizeof(PolyUOp *));
  }

  /* Build intermediate_idx: maps after_nodes index → intermediate array index.
   * -1 for ASSIGN AFTERs (no intermediate allocation). */
  int *intermediate_idx = calloc(n_after, sizeof(int));
  {
    int ii = 0;
    for (int a = 0; a < n_after; a++) {
      if (after_is_intermediate[a])
        intermediate_idx[a] = ii++;
      else
        intermediate_idx[a] = -1;
    }
  }

  /* 5. Build AFTER kernels (both intermediate producers and ASSIGN writers)
   * using split_kernel_rewrite. */
  for (int a = 0; a < n_after; a++) {
    PolyUOp *after = after_nodes[a];
    PolyUOp *after_buf = after->src[0];
    PolyUOp *end_chain = after->src[1];

    PolySplitCtx sctx;
    memset(&sctx, 0, sizeof(sctx));
    sctx.ctx = ctx;
    sctx.remap = poly_map_new(256);
    sctx.buf_to_param = poly_map_new(64);

    PolyUOp *rewritten = split_kernel_rewrite(&sctx, end_chain);
    rewritten = flatten_range_chain(ctx, rewritten);
    PolyUOp *kernel = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, rewritten,
                                 poly_arg_none());
    kernel = add_kernel_loads(ctx, kernel);

    result.kernels[a] = kernel;
    result.param_to_buf[a] = malloc(sctx.param_count * sizeof(PolyUOp *));
    memcpy(result.param_to_buf[a], sctx.param_bufs,
           sctx.param_count * sizeof(PolyUOp *));
    result.kernel_n_params[a] = sctx.param_count;
    if (sctx.var_count > 0) {
      result.var_to_buf[a] = malloc(sctx.var_count * sizeof(PolyUOp *));
      memcpy(result.var_to_buf[a], sctx.var_bufs,
             sctx.var_count * sizeof(PolyUOp *));
    }
    result.kernel_n_vars[a] = sctx.var_count;

    /* Populate intermediate arrays only for real intermediates */
    if (after_is_intermediate[a]) {
      int ii = intermediate_idx[a];
      result.intermediate_sizes[ii] =
          (after_buf->arg.kind == POLY_ARG_INT) ? after_buf->arg.i : 1;
      result.intermediate_itemsizes[ii] =
          poly_dtype_itemsize(poly_dtype_scalar(after_buf->dtype));
      result.intermediate_buf_uops[ii] = after_buf;
    }

    free(sctx.param_bufs);
    free(sctx.var_bufs);
    poly_map_destroy(sctx.remap);
    poly_map_destroy(sctx.buf_to_param);
  }

  /* 6. Build consumer kernels for STORE SINK sources.
   * ASSIGN SINK sources are already handled as AFTER kernels above.
   * The STORE from abuf_sink has INDEX-wrapped BUFFERs and AFTER nodes.
   * split_kernel_rewrite: BUFFER→PARAM, AFTER→src[0], RANGE→renumber. */
  if (abuf_sink->op != POLY_OP_SINK || abuf_sink->n_src != n_sink_src) {
    fprintf(stderr, "polygrad: new_split: SINK mismatch (%d vs %d)\n",
            abuf_sink->n_src, n_sink_src);
    goto cleanup;
  }

  {
    int consumer_idx = 0;
    for (int i = 0; i < n_sink_src; i++) {
      PolyUOp *orig = tensor_sink->src[i];

      /* Skip ASSIGN sources — they're handled as AFTER kernels */
      if (orig->op == POLY_OP_ASSIGN) continue;

      PolyUOp *post_store = abuf_sink->src[i];

      if (!orig || orig->op != POLY_OP_STORE) {
        fprintf(stderr, "polygrad: new_split: SINK source %d is %s\n",
                i, orig ? poly_op_name(orig->op) : "NULL");
        goto cleanup;
      }

      PolyRangeEntry *store_re = poly_range_map_get(ictx, orig);
      if (!store_re) {
        fprintf(stderr, "polygrad: new_split: no range entry for STORE %d\n", i);
        goto cleanup;
      }

      PolySplitCtx sctx;
      memset(&sctx, 0, sizeof(sctx));
      sctx.ctx = ctx;
      sctx.remap = poly_map_new(256);
      sctx.buf_to_param = poly_map_new(64);

      PolyUOp *rewritten_store = split_kernel_rewrite(&sctx, post_store);

      PolyUOp *outer_rngs[POLY_MAX_DIMS];
      int n_outer = 0;
      for (int d = 0; d < store_re->n_out; d++) {
        if (store_re->out_rngs[d]->op == POLY_OP_RANGE) {
          PolyUOp *renumbered = poly_map_get(sctx.remap,
              ptr_hash(store_re->out_rngs[d]),
              store_re->out_rngs[d], ptr_eq);
          if (renumbered)
            outer_rngs[n_outer++] = renumbered;
        }
      }

      /* Build END chain: innermost first → outermost last */
      PolyUOp *current = rewritten_store;
      for (int r = n_outer - 1; r >= 0; r--) {
        PolyUOp *end_src[2] = { current, outer_rngs[r] };
        current = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2,
                             poly_arg_none());
      }

      PolyUOp *kernel = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, current,
                                   poly_arg_none());
      kernel = add_kernel_loads(ctx, kernel);

      int kidx = n_after + consumer_idx;
      result.kernels[kidx] = kernel;
      result.param_to_buf[kidx] = malloc(sctx.param_count * sizeof(PolyUOp *));
      memcpy(result.param_to_buf[kidx], sctx.param_bufs,
             sctx.param_count * sizeof(PolyUOp *));
      result.kernel_n_params[kidx] = sctx.param_count;
      if (sctx.var_count > 0) {
        result.var_to_buf[kidx] = malloc(sctx.var_count * sizeof(PolyUOp *));
        memcpy(result.var_to_buf[kidx], sctx.var_bufs,
               sctx.var_count * sizeof(PolyUOp *));
      }
      result.kernel_n_vars[kidx] = sctx.var_count;

      free(sctx.param_bufs);
      free(sctx.var_bufs);
      poly_map_destroy(sctx.remap);
      poly_map_destroy(sctx.buf_to_param);
      consumer_idx++;
    }
    assert(consumer_idx == n_consumer_stores);
  }

  result.n_kernels = total_kernels;

  /* ── Stage 5: Build exec_order from RAW + WAR + WAW dependency graph ── */
  if (total_kernels <= 1) {
    result.exec_order = NULL;  /* trivial: no reordering needed */
  } else {
    int n = total_kernels;

    /* 5a. Build buf_to_prod: BUFFER → writer kernel index for INTERMEDIATE
     * buffers only. ASSIGN AFTERs are NOT included — their ordering is
     * handled purely by WAR/WAW edges (5c/5d). Including ASSIGN buffers
     * here would create RAW "producer before consumer" edges that conflict
     * with WAR "reader before writer" edges, causing cycles. */
    PolyMap *buf_to_prod = poly_map_new((size_t)(n_after < 8 ? 8 : n_after));
    for (int b = 0; b < n_after; b++) {
      if (!after_is_intermediate[b]) continue;  /* skip ASSIGN AFTERs */
      PolyUOp *buf = after_nodes[b]->src[0];
      if (!poly_map_get(buf_to_prod, ptr_hash(buf), buf, ptr_eq))
        poly_map_set(buf_to_prod, ptr_hash(buf), buf,
                     (PolyUOp *)(intptr_t)(b + 1), ptr_eq);
    }

    /* 5b. Build dep[from*n + to] = true if kernel `from` must run before `to`.
     * RAW edges: producer → consumer (consumer reads intermediate written by producer). */
    bool *dep = calloc((size_t)n * n, sizeof(bool));
    for (int k = 0; k < n; k++) {
      for (int p = 0; p < result.kernel_n_params[k]; p++) {
        PolyUOp *buf = result.param_to_buf[k][p];
        if (!buf) continue;
        PolyUOp *val = poly_map_get(buf_to_prod, ptr_hash(buf), buf, ptr_eq);
        if (!val) continue;
        int b = (int)((intptr_t)val - 1);
        if (b != k) dep[b * n + k] = true;  /* producer b before consumer k */
      }
    }

    /* 5c. WAR edges: for each ASSIGN writer W, any NON-ASSIGN kernel K that
     * reads from W's target buffer must complete before W starts.
     * Skip WAR between ASSIGN kernels — their ordering is handled by
     * program-order edges (5c2). Without this skip, WAR creates conflicts:
     * e.g. weight ASSIGN reads m_buffer, m ASSIGN writes m_buffer →
     * WAR says weight before m, but program order says m before weight. */
    for (int w = 0; w < n_after; w++) {
      if (after_is_intermediate[w]) continue;  /* WAR only for ASSIGN writes */
      PolyUOp *write_buf = after_nodes[w]->src[0];
      for (int k = 0; k < n; k++) {
        if (k == w) continue;
        /* Skip WAR between ASSIGN kernels — program order handles this */
        if (k < n_after && !after_is_intermediate[k]) continue;
        for (int p = 0; p < result.kernel_n_params[k]; p++) {
          if (result.param_to_buf[k][p] == write_buf) {
            dep[k * n + w] = true;  /* reader k before writer w */
            break;
          }
        }
      }
    }

    /* 5c2. Program-order edges for ASSIGN kernels: enforce the order in which
     * assign().realize() was called during tracing. Essential for Adam where
     * m/v updates must run before param update (param reads NEW m/v).
     * ASSIGN kernels appear in after_nodes[] in program order already
     * (they're added in the order SINK sources appear). */
    {
      int prev_assign = -1;
      for (int k = 0; k < n_after; k++) {
        if (after_is_intermediate[k]) continue;  /* only ASSIGN kernels */
        if (prev_assign >= 0)
          dep[prev_assign * n + k] = true;  /* prev before current */
        prev_assign = k;
      }
    }

    /* 5d. WAW edges: if multiple ASSIGN kernels write the same buffer,
     * enforce program order between consecutive writers.
     * Build per-buffer writer lists, add edges between consecutive writes. */
    {
      /* Collect ASSIGN writer indices per buffer */
      for (int w1 = 0; w1 < n_after; w1++) {
        if (after_is_intermediate[w1]) continue;
        PolyUOp *buf1 = after_nodes[w1]->src[0];
        for (int w2 = w1 + 1; w2 < n_after; w2++) {
          if (after_is_intermediate[w2]) continue;
          if (after_nodes[w2]->src[0] == buf1) {
            /* Two ASSIGN writers to same buffer: w1 before w2 (program order) */
            dep[w1 * n + w2] = true;
          }
        }
      }
    }

    poly_map_destroy(buf_to_prod);

    /* 5e. Kahn's topological sort with min-index tie-break. O(n² + E). */
    int *in_degree = calloc(n, sizeof(int));
    for (int from = 0; from < n; from++)
      for (int to = 0; to < n; to++)
        if (dep[from * n + to]) in_degree[to]++;

    int *order = malloc(n * sizeof(int));
    int n_ordered = 0;
    bool *ready = calloc(n, sizeof(bool));
    for (int k = 0; k < n; k++)
      if (in_degree[k] == 0) ready[k] = true;

    while (n_ordered < n) {
      int pick = -1;
      for (int k = 0; k < n; k++)
        if (ready[k]) { pick = k; break; }
      if (pick < 0) {
        fprintf(stderr, "polygrad: exec_order: cycle detected in kernel "
                "dependency graph (ordered %d of %d)\n", n_ordered, n);
        free(order); order = NULL;
        break;
      }
      ready[pick] = false;
      order[n_ordered++] = pick;
      for (int to = 0; to < n; to++) {
        if (dep[pick * n + to]) {
          in_degree[to]--;
          if (in_degree[to] == 0) ready[to] = true;
        }
      }
    }

    free(dep); free(in_degree); free(ready);
    result.exec_order = order;  /* NULL if cycle detected → sequential fallback */
  }

#ifndef NDEBUG
  /* Group A: intermediate_buf_uops populated and valid */
  assert(result.n_intermediates == 0 || result.intermediate_buf_uops != NULL);
  for (int _b = 0; _b < result.n_intermediates; _b++) {
    assert(result.intermediate_buf_uops[_b] != NULL);
    assert(result.intermediate_buf_uops[_b]->op == POLY_OP_BUFFER &&
           "schedule_v2_new: intermediate must be BUFFER op");
  }

  /* Group B: exec_order is a valid permutation */
  if (result.exec_order != NULL) {
    int _nk = result.n_kernels;
    bool *_seen = calloc(_nk, sizeof(bool));
    assert(_seen != NULL);
    for (int _step = 0; _step < _nk; _step++) {
      int _k = result.exec_order[_step];
      assert(_k >= 0 && _k < _nk && "exec_order: kernel index out of range");
      assert(!_seen[_k] && "exec_order: duplicate kernel index");
      _seen[_k] = true;
    }
    free(_seen);
  }

  /* Group C: dependency ordering — every dependency edge is satisfied.
   * Build buf→writer map, check that exec_order positions satisfy
   * pos[consumer] > pos[producer] for RAW edges. */
  if (result.exec_order != NULL && n_after > 0 &&
      result.kernel_n_params != NULL && result.param_to_buf != NULL) {
    int _nk = result.n_kernels;
    int *_pos = malloc(_nk * sizeof(int));
    assert(_pos != NULL);
    for (int _step = 0; _step < _nk; _step++)
      _pos[result.exec_order[_step]] = _step;
    /* Build buf→writer map for INTERMEDIATE buffers only.
     * ASSIGN AFTERs have reversed ordering (reader before writer)
     * enforced by WAR edges, so RAW assertion doesn't apply. */
    PolyMap *_btp = poly_map_new((size_t)(n_after < 4 ? 4 : n_after));
    for (int _b = 0; _b < n_after; _b++) {
      if (!after_is_intermediate[_b]) continue;  /* skip ASSIGN AFTERs */
      PolyUOp *_buf = after_nodes[_b]->src[0];
      if (!poly_map_get(_btp, ptr_hash(_buf), _buf, ptr_eq))
        poly_map_set(_btp, ptr_hash(_buf), _buf,
                     (PolyUOp *)(intptr_t)(_b + 1), ptr_eq);
    }
    for (int _k = 0; _k < _nk; _k++) {
      if (!result.param_to_buf[_k]) continue;
      for (int _p = 0; _p < result.kernel_n_params[_k]; _p++) {
        PolyUOp *_buf = result.param_to_buf[_k][_p];
        if (!_buf) continue;
        PolyUOp *_v = poly_map_get(_btp, ptr_hash(_buf), _buf, ptr_eq);
        if (_v) {
          int _prod = (int)((intptr_t)_v - 1);
          if (_k == _prod) continue;
          assert(_pos[_k] > _pos[_prod] &&
                 "exec_order: dependency ordering violated");
        }
      }
    }
    poly_map_destroy(_btp);
    free(_pos);
  }
#endif

  free(after_nodes);
  free(after_is_intermediate);
  free(intermediate_idx);
  poly_indexing_ctx_destroy(ictx);
  return result;

cleanup:
  free(after_nodes);
  free(after_is_intermediate);
  free(intermediate_idx);
  poly_schedule_result_free(&result);
  memset(&result, 0, sizeof(result));
  poly_indexing_ctx_destroy(ictx);
  return result;
}

/* ── Public API: poly_schedule_v2 ─────────────────────────────────────── */

PolyScheduleResult poly_schedule_v2(PolyCtx *ctx, PolyUOp *tensor_sink) {
  return schedule_v2_new(ctx, tensor_sink);
}


/* Free dynamically allocated fields of a PolyScheduleResult. */
void poly_schedule_result_free(PolyScheduleResult *sr) {
  if (sr->param_to_buf) {
    for (int k = 0; k < sr->n_kernels; k++)
      free(sr->param_to_buf[k]);
    free(sr->param_to_buf);
  }
  free(sr->kernel_n_params);
  free(sr->intermediate_sizes);
  free(sr->intermediate_itemsizes);
  free(sr->intermediate_buf_uops);
  free(sr->exec_order);
  if (sr->var_to_buf) {
    for (int k = 0; k < sr->n_kernels; k++)
      free(sr->var_to_buf[k]);
    free(sr->var_to_buf);
  }
  free(sr->kernel_n_vars);
  free(sr->kernels);
  memset(sr, 0, sizeof(*sr));
}
