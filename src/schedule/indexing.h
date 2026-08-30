/* C declarations for tinygrad schedule/indexing.py. */

#ifndef POLY_INDEXING_H
#define POLY_INDEXING_H

#include "polygrad.h"
#include "uop/ops.h"

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
  PolyCtx *ctx;
  PolyMap *consumer_map;
  PolyMap *realize_map;
  PolyMap *non_removable;
  PolyMap *range_map;
  PolyMap *shape_cache;
  int next_range_id;
} PolyIndexingCtx;

PolyIndexingCtx *poly_indexing_ctx_new(PolyCtx *ctx);
void poly_indexing_ctx_destroy(PolyIndexingCtx *ictx);
PolyMap *poly_consumer_map_build(PolyCtx *ctx, PolyUOp *sink);
PolyConsumerList *poly_consumer_map_get(PolyMap *cmap, PolyUOp *u);
void poly_realize_map_build(PolyIndexingCtx *ictx, PolyUOp *sink);
bool poly_is_realized(PolyIndexingCtx *ictx, PolyUOp *u);
void poly_range_propagate(PolyIndexingCtx *ictx, PolyUOp *sink);
PolyRangeEntry *poly_range_map_get(PolyIndexingCtx *ictx, PolyUOp *u);
PolyUOp *poly_apply_rangeify(PolyIndexingCtx *ictx, PolyUOp *sink);
PolyUOp *poly_run_rangeify(PolyCtx *ctx, PolyUOp *sink, bool debug);

/* C encoding of indexing.py:BufferizeOpts(device=s.device, ...). */
PolyUOp *poly_bufferize_device_hint(PolyCtx *ctx, PolyUOp *value, PolyMap *device_memo);
PolyArg poly_bufferize_opts_for_device(
    PolyUOp *device,
    PolyAddrSpace addrspace,
    bool removable
);

/* Apply a movement op's index transform to output ranges.
 *
 * Given output ranges (what the consumer sees), compute input ranges
 * (what the movement op's source sees) after applying the op's semantics.
 *
 * op:        the movement op (RESHAPE, EXPAND, PERMUTE, SHRINK, FLIP, PAD)
 * in_shape:  shape of the movement op's source (input)
 * arg:       the movement op's argument
 * out_rngs:  output ranges (from consumer), length n_out
 * n_out:     number of output ranges
 * in_rngs:   [out] input ranges for the source, must hold at least in_shape.ndim entries
 * n_in_out:  [out] number of input ranges written
 *
 * For PAD, also outputs:
 * valid_out: [out] validity predicate UOp (AND of bounds checks), or NULL if no padding
 *
 * Returns true on success, false on error.
 */
bool poly_apply_movement_op(
    PolyCtx *ctx,
    PolyUOp *movement,
    PolyOps op,
    PolyShape in_shape,
    PolyArg arg,
    PolyUOp **out_rngs,
    int n_out,
    PolyUOp **in_rngs,
    int *n_in_out,
    PolyUOp **valid_out
);

/* Current tinygrad schedule/indexing.py:_apply_reshape. */
bool poly_apply_reshape(
    PolyCtx *ctx,
    PolyUOp **in_shape,
    int n_in,
    PolyUOp **out_shape,
    int n_out,
    PolyUOp **out_ranges,
    int n_ranges,
    PolyUOp **in_ranges,
    int *n_in_out
);

/* Compute the exact RESHAPE index transform from the movement UOp's symbolic
 * input/output shapes. Partial indexing follows pinned _mop_index: an
 * unindexed output suffix must exactly match an input suffix, and n_in_out
 * reports the mapped input-prefix rank. */
bool poly_reshape_indices(
    PolyCtx *ctx,
    PolyUOp *reshape,
    PolyUOp **out_ranges,
    int n_out,
    PolyUOp **in_ranges,
    int *n_in_out
);

#ifdef __cplusplus
}
#endif

#endif /* POLY_INDEXING_H */
