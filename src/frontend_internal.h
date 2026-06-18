/*
 * frontend_internal.h -- Private helpers shared inside the C core
 *
 * NOT part of the public API. Not installed. Not included by language frontends.
 * Contains declarations for utility functions used across frontend helpers,
 * placement, scheduling, and execution lowering.
 */

#ifndef POLY_FRONTEND_INTERNAL_H
#define POLY_FRONTEND_INTERNAL_H

#include "polygrad.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Constants */

#define POLY_MAX_REALIZE_BUFS 2048
#define POLY_MAX_STRUCT_NODES 8192
#define POLY_SCHED_CACHE_VERSION 3u

/* Structural hashing/equality (for cache keying by graph shape) */

uint32_t poly_structural_hash(PolyUOp *u);
bool poly_structural_eq(const void *a, const void *b);

/* Buffer ordering */

/* DFS to collect BUFFER nodes in structural order (same order as
 * structural_hash). buf_order written up to POLY_MAX_REALIZE_BUFS.
 * visited/n_visited are caller-provided scratch (POLY_MAX_STRUCT_NODES). */
void poly_collect_buf_order(
    PolyUOp *u,
    PolyUOp **buf_order,
    int *n_bufs,
    PolyUOp **visited,
    int *n_visited
);

/* Owned dynamic version of poly_collect_buf_order. Caller frees *out_buf_order
 * with free(). */
bool poly_collect_buf_order_alloc(
    PolyUOp *u,
    PolyUOp ***out_buf_order,
    int *out_n_bufs,
    int *out_n_visited
);

/* Linear scan for a BUFFER UOp in a buf_order array. Returns index or -1. */
int poly_find_buf_position(PolyUOp *buf, PolyUOp **buf_order, int n_bufs);

/* Collect ordered external buffers (output-first, then inputs).
 * Returns count of buffers found, up to max_bufs. */
int poly_collect_ordered_buffers(
    PolyCtx *ctx,
    PolyUOp *tensor_sink,
    PolyUOp **ordered,
    int max_bufs
);

/* Owned dynamic version of poly_collect_ordered_buffers. Caller frees
 * *out_ordered with free(). */
bool poly_collect_ordered_buffers_alloc(
    PolyCtx *ctx,
    PolyUOp *tensor_sink,
    PolyUOp ***out_ordered,
    int *out_n_ordered
);

/* Collect output BUFFER UOps from SINK -> STORE -> BUFFER chain.
 * Returns count written to out[], up to cap. */
int poly_collect_output_buffers_in_sink(PolyUOp *tensor_sink, PolyUOp **out, int cap);

/* Graph validation */

/* Validates that all UOps in the kernel graph are owned by ctx and have
 * no NULL sources. Returns true if valid, false with diagnostic on stderr. */
bool poly_validate_kernel_graph(PolyCtx *ctx, PolyUOp *root);

/* Memoized backing helper for poly_uop_device(), matching tinygrad's cached
 * UOp._device property without storing pass-local cache state on every UOp. */
PolyDevice poly_uop_device_cached(PolyUOp *u, PolyMap *cache);

#ifdef __cplusplus
}
#endif

#endif /* POLY_FRONTEND_INTERNAL_H */
