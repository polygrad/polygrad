/*
 * rangeify.h — Range-propagation scheduling pipeline
 *
 * Ports tinygrad's rangeify.py + indexing.py to C11.
 * Converts tensor-level UOp graphs into multi-kernel programs
 * via consumer analysis, realize-point detection, and range propagation.
 */

#ifndef POLY_RANGEIFY_H
#define POLY_RANGEIFY_H

#include "polygrad.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ── Consumer map ────────────────────────────────────────────────────── */
/* For each UOp, tracks which UOps consume it as a source.               */

typedef struct {
    PolyUOp **items;
    int count;
    int cap;
} PolyConsumerList;

/* Build consumer map from a SINK root. Returns a PolyMap of UOp* → PolyConsumerList*.
 * Uses pointer hashing/equality. Caller must free the map with poly_map_destroy()
 * and free each PolyConsumerList with free(). */
PolyMap *poly_consumer_map_build(PolyCtx *ctx, PolyUOp *sink);

/* Look up consumers for a UOp. Returns NULL if not found. */
PolyConsumerList *poly_consumer_map_get(PolyMap *cmap, PolyUOp *u);

/* ── Realize map ─────────────────────────────────────────────────────── */
/* Marks UOps that must be realized as kernel boundaries.                 */

typedef struct {
    int *axes;    /* NULL = all axes realized, non-NULL = specific axes */
    int n_axes;   /* number of entries in axes (-1 = all) */
} PolyRealizeInfo;

/* ── Range map ───────────────────────────────────────────────────────── */
/* For each UOp, stores (input_ranges, output_ranges).                   */

typedef struct {
    PolyUOp **in_rngs;   /* input ranges (what the sources see) */
    int n_in;
    PolyUOp **out_rngs;  /* output ranges (what consumers see) */
    int n_out;
    PolyUOp *valid;      /* validity predicate from PAD (NULL if none) */
} PolyRangeEntry;

/* ── Per-consumer BUFFER ranges ──────────────────────────────────────── */
/* When a BUFFER has multiple consumers that disagree on ranges           */
/* (Case 4b in range_propagate), we save each consumer's ranges so the   */
/* BUFFER handler can pick the correct indexing for each store context.   */

#define POLY_MAX_ALT_RNGS 8

typedef struct {
    int count;
    int lens[POLY_MAX_ALT_RNGS];
    PolyUOp *rngs[POLY_MAX_ALT_RNGS][POLY_MAX_DIMS];
} PolyBufferAltRngs;

/* ── Indexing context ────────────────────────────────────────────────── */
/* Aggregated state for the entire rangeify pass.                        */

typedef struct {
    PolyCtx *ctx;
    PolyMap *consumer_map;  /* UOp* → PolyConsumerList* */
    PolyMap *realize_map;   /* UOp* → PolyRealizeInfo*  */
    PolyMap *range_map;     /* UOp* → PolyRangeEntry*   */
    PolyMap *shape_cache;   /* UOp* → PolyShape*        */
    PolyMap *reduce_origin; /* post-rangeify REDUCE UOp* → pre-rangeify REDUCE_AXIS UOp* */
    PolyMap *buffer_alt_rngs; /* BUFFER UOp* → PolyBufferAltRngs* (disagreeing consumers) */
    PolyMap *realized_to_bufferize; /* original UOp* → BUFFERIZE UOp* */
    PolyMap *bufferize_to_realized; /* BUFFERIZE UOp* → original UOp* */
    int next_range_id;
    bool add_buffer_indices; /* if true, wrap BUFFER sources with flat INDEX during apply_rangeify */
    int max_kernel_bufs; /* 0 = disabled, >0 = max buffer params per kernel (incl. output) */
} PolyIndexingCtx;

/* ── Rangeify stats ───────────────────────────────────────────────────── */
/* Runtime counters for tracking workaround/fallback path usage.          */

typedef struct {
    int remap_calls;
    int remap_id_matches;
    int remap_pos_matches;
    int remap_unique_bound_matches;
    int remap_bound_matches;
    int remap_failures;
    int orphan_top_level_hits;
    int deep_orphan_hits;
    int buffer_alt_created;
    int buffer_alt_used;
} PolyRangeifyStats;

/* Reset/get global rangeify stats counters. */
void poly_rangeify_stats_reset(void);
PolyRangeifyStats poly_rangeify_stats_get(void);

/* Initialize an indexing context. All maps are allocated. */
PolyIndexingCtx *poly_indexing_ctx_new(PolyCtx *ctx);

/* Free an indexing context and all its maps. */
void poly_indexing_ctx_destroy(PolyIndexingCtx *ictx);

/* Build the realize map for a tensor SINK.
 * Marks SINK sources, CONTIGUOUS, COPY, STORE, ASSIGN as realize points. */
void poly_realize_map_build(PolyIndexingCtx *ictx, PolyUOp *sink);

/* Check if a UOp is in the realize map. */
bool poly_is_realized(PolyIndexingCtx *ictx, PolyUOp *u);

/* Propagate ranges through the tensor graph.
 * Walks reverse topological order assigning RANGE UOps to each op.
 * Builds consumer_map if not already present. */
void poly_range_propagate(PolyIndexingCtx *ictx, PolyUOp *sink);

/* Look up range entry for a UOp. Returns NULL if not found. */
PolyRangeEntry *poly_range_map_get(PolyIndexingCtx *ictx, PolyUOp *u);

/* Apply rangeify graph rewrite: transforms tensor-level IR to kernel-level IR.
 *
 * Transformations:
 *  - Movement ops (RESHAPE, EXPAND, PERMUTE, SHRINK, FLIP) → src[0]
 *  - REDUCE_AXIS → REDUCE with range sources
 *  - PAD → WHERE(valid_mask, src[0], 0.0)
 *  - Realized computed ops → BUFFERIZE(op, ranges...)
 *
 * Must be called after poly_realize_map_build() and poly_range_propagate().
 * Returns the rewritten SINK. */
PolyUOp *poly_apply_rangeify(PolyIndexingCtx *ictx, PolyUOp *sink);

/* ── pm_add_buffers ─────────────────────────────────────────────────── */
/* Convert BUFFERIZE nodes to BUFFER+STORE+END+AFTER chains.
 * Port of tinygrad's pm_add_buffers (rangeify.py:392-401).
 * Must be called after poly_apply_rangeify().
 * Returns the rewritten graph (no BUFFERIZE nodes remain). */
PolyUOp *poly_apply_add_buffers(PolyCtx *ctx, PolyUOp *sink,
                                 PolyMap *buf_dims_map);

/* ── Schedule result ─────────────────────────────────────────────────── */

typedef struct {
    PolyUOp **kernels;   /* array of per-kernel SINK UOps */
    int n_kernels;

    /* Per-kernel PARAM-to-buffer mappings.
     * param_to_buf[k][i] = BUFFER UOp (original) or BUFFERIZE UOp (intermediate)
     * that PARAM i in kernel k corresponds to.
     * NULL for single-kernel (use collect_ordered_buffers convention). */
    PolyUOp ***param_to_buf;
    int *kernel_n_params;

    /* Intermediate buffer metadata (one per intermediate). */
    int64_t *intermediate_sizes;   /* size in elements per intermediate buf */
    int *intermediate_itemsizes;   /* itemsize in bytes per element (e.g. 4 for f32, 1 for bool) */
    int n_intermediates;

    /* Canonical intermediate buffer UOps, one per intermediate.
     * New split path: BUFFER(LUNIQUE) from after_nodes[b]->src[0].
     * Old split path: BUFFERIZE node from bufferize_nodes[b].
     * Used by all realize paths for intermediate identification.
     * Length = n_intermediates. NULL if n_intermediates == 0. */
    PolyUOp **intermediate_buf_uops;

    /* Execution order: exec_order[step] = kernel_index.
     * NULL means use sequential index order (0, 1, 2, ...).
     * Built from RAW dependency graph via topological sort. */
    int *exec_order;

    /* Per-kernel DEFINE_VAR params (var_to_buf[k][v] = DEFINE_VAR UOp for var v).
     * DEFINE_VAR args come AFTER buffer args in the kernel's args[] array.
     * NULL if no dynamic shapes are used. */
    PolyUOp ***var_to_buf;
    int *kernel_n_vars;
} PolyScheduleResult;

/* Free the dynamically allocated fields of a PolyScheduleResult. */
void poly_schedule_result_free(PolyScheduleResult *sr);

/* Schedule v2: full rangeify pipeline.
 * Returns a list of kernel SINKs ready for linearize → render → compile.
 * For single-kernel cases, n_kernels == 1 and result is equivalent to
 * poly_schedule(). */
PolyScheduleResult poly_schedule_v2(PolyCtx *ctx, PolyUOp *tensor_sink);

#ifdef __cplusplus
}
#endif

#endif /* POLY_RANGEIFY_H */
