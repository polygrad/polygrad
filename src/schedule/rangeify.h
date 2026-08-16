/*
 * rangeify.h -- Tinygrad-aligned kernel graph construction.
 *
 * Mirrors tinygrad's schedule/rangeify.py and schedule/indexing.py.
 * This header carries both the public kernel-graph stage and the current
 * kernel-schedule bridge types used by engine/schedule.c.
 */

#ifndef POLY_SCHEDULE_RANGEIFY_H
#define POLY_SCHEDULE_RANGEIFY_H

#include "polygrad.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  PolyUOp **items;
  int count;
  int cap;
} PolyConsumerList;

typedef struct {
  int *axes;
  int n_axes;
} PolyRealizeInfo;

typedef struct {
  PolyUOp **in_rngs;
  int n_in;
  PolyUOp **out_rngs;
  int n_out;
  PolyUOp *valid;
} PolyRangeEntry;

typedef struct {
  int count;
  int cap;
  int *lens;
  PolyUOp *(*rngs)[POLY_MAX_DIMS];
} PolyBufferAltRngs;

typedef struct {
  PolyCtx *ctx;
  PolyMap *consumer_map;
  PolyMap *realize_map;
  PolyMap *range_map;
  PolyMap *shape_cache;
  PolyMap *reduce_origin;
  PolyMap *buffer_alt_rngs;
  PolyMap *realized_to_bufferize;
  PolyMap *bufferize_to_realized;
  int next_range_id;
  bool add_buffer_indices;
  int max_kernel_bufs;
} PolyIndexingCtx;

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
  int buffer_alt_max_count;
} PolyRangeifyStats;

void poly_rangeify_stats_reset(void);
PolyRangeifyStats poly_rangeify_stats_get(void);

PolyIndexingCtx *poly_indexing_ctx_new(PolyCtx *ctx);
void poly_indexing_ctx_destroy(PolyIndexingCtx *ictx);

PolyMap *poly_consumer_map_build(PolyCtx *ctx, PolyUOp *sink);
PolyConsumerList *poly_consumer_map_get(PolyMap *cmap, PolyUOp *u);

void poly_realize_map_build(PolyIndexingCtx *ictx, PolyUOp *sink);
bool poly_is_realized(PolyIndexingCtx *ictx, PolyUOp *u);
void poly_range_propagate(PolyIndexingCtx *ictx, PolyUOp *sink);
PolyRangeEntry *poly_range_map_get(PolyIndexingCtx *ictx, PolyUOp *u);

/* Tinygrad schedule/indexing.py analogue: apply rangeify to a graph. */
PolyUOp *poly_run_rangeify(PolyIndexingCtx *ictx, PolyUOp *sink);

/* Internal stage helpers kept visible for parity probes and focused tests. */
PolyUOp *poly_apply_multi_pm(PolyCtx *ctx, PolyUOp *sink);
PolyUOp *poly_apply_earliest_rewrites(PolyCtx *ctx, PolyUOp *sink);
PolyUOp *poly_apply_add_buffers(PolyCtx *ctx, PolyUOp *sink, PolyMap *buf_dims_map);
PolyUOp *poly_apply_add_buffers_local(PolyCtx *ctx, PolyUOp *sink);

/* Tinygrad schedule/rangeify.py analogue: build the kernel graph for a sink. */
PolyUOp *poly_get_kernel_graph(PolyCtx *ctx, PolyUOp *sink);

typedef enum {
  POLY_KERNEL_ITEM_COMPUTE = 0,
  POLY_KERNEL_ITEM_COPY = 1,
  POLY_KERNEL_ITEM_CALL = 2,
} PolyKernelItemKind;

typedef struct {
  PolyUOp **kernels;
  int n_kernels;

  int *kernel_kinds; /* PolyKernelItemKind */
  int *copy_dst_params;
  int *copy_src_params;

  /* Exact split-time CALL bindings. An entry may retain AFTER for dependency
   * ordering; schedule creation resolves its existing buffer identity. */
  PolyUOp ***param_to_buf;
  int *kernel_n_params;

  int64_t *intermediate_sizes;
  int *intermediate_itemsizes;
  int n_intermediates;
  PolyUOp **intermediate_buf_uops;

  int *exec_order;

  PolyUOp ***var_to_buf;
  int *kernel_n_vars;
} PolyKernelScheduleResult;

/* Tinygrad create_schedule consumes get_kernel_graph output directly. The C
 * bridge still needs an explicit split-kernel result while schedule creation
 * remains a richer struct than tinygrad's Python list of ExecItems. */
PolyKernelScheduleResult poly_build_kernel_schedule_from_kernel_graph(
    PolyCtx *ctx,
    PolyUOp *kernel_graph
);
PolyKernelScheduleResult poly_build_kernel_schedule(PolyCtx *ctx, PolyUOp *tensor_sink);
void poly_kernel_schedule_result_free(PolyKernelScheduleResult *sr);

#ifdef __cplusplus
}
#endif

#endif /* POLY_SCHEDULE_RANGEIFY_H */
