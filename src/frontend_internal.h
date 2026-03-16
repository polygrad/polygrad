/*
 * frontend_internal.h -- Private helpers shared between frontend.c and exec_plan.c
 *
 * NOT part of the public API. Not installed. Not included by language frontends.
 * Contains declarations for utility functions that both the old realize/compile
 * path (frontend.c) and the new exec_plan path (exec_plan.c) need.
 */

#ifndef POLY_FRONTEND_INTERNAL_H
#define POLY_FRONTEND_INTERNAL_H

#include "polygrad.h"
#include "frontend.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Constants ───────────────────────────────────────────────────────── */

#define POLY_MAX_REALIZE_BUFS 256
#define POLY_MAX_STRUCT_NODES 8192
#define POLY_SCHED_CACHE_VERSION 3u

/* ── Pointer hashing (for PolyMap keying by UOp pointer) ─────────────── */

bool poly_ptr_eq(const void *a, const void *b);
uint32_t poly_ptr_hash(const void *p);

/* ── Structural hashing/equality (for cache keying by graph shape) ───── */

uint32_t poly_structural_hash(PolyUOp *u);
bool poly_structural_eq(const void *a, const void *b);

/* ── Buffer ordering ─────────────────────────────────────────────────── */

/* DFS to collect BUFFER nodes in structural order (same order as
 * structural_hash). buf_order written up to POLY_MAX_REALIZE_BUFS.
 * visited/n_visited are caller-provided scratch (POLY_MAX_STRUCT_NODES). */
void poly_collect_buf_order(PolyUOp *u, PolyUOp **buf_order, int *n_bufs,
                            PolyUOp **visited, int *n_visited);

/* Linear scan for a BUFFER UOp in a buf_order array. Returns index or -1. */
int poly_find_buf_position(PolyUOp *buf, PolyUOp **buf_order, int n_bufs);

/* Collect ordered external buffers (output-first, then inputs).
 * Returns count of buffers found, up to max_bufs. */
int poly_collect_ordered_buffers(PolyCtx *ctx, PolyUOp *tensor_sink,
                                 PolyUOp **ordered, int max_bufs);

/* Collect output BUFFER UOps from SINK -> STORE -> BUFFER chain.
 * Returns count written to out[], up to cap. */
int poly_collect_output_buffers_in_sink(PolyUOp *tensor_sink,
                                        PolyUOp **out, int cap);

/* ── Graph validation ────────────────────────────────────────────────── */

/* Validates that all UOps in the kernel graph are owned by ctx and have
 * no NULL sources. Returns true if valid, false with diagnostic on stderr. */
bool poly_validate_kernel_graph(PolyCtx *ctx, PolyUOp *root);

/* ── BIND stripping ──────────────────────────────────────────────────── */

/* Rewrites BIND(DEFINE_VAR, CONST) -> DEFINE_VAR throughout the graph.
 * Extracts var->value bindings into out_vals[0..n_out-1].
 * Remaps buf_bindings[].buffer pointers to their rewritten equivalents. */
PolyUOp *poly_strip_bind_values(PolyCtx *ctx, PolyUOp *sink,
                                PolyVarBinding *out_vals, int *n_out, int max_out,
                                PolyBufferBinding *buf_bindings, int n_buf_bindings);

#ifdef __cplusplus
}
#endif

#endif /* POLY_FRONTEND_INTERNAL_H */
