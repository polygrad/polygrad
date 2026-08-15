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

/* tinygrad's callified function uses value PARAMs as external storage
 * identities until rangeify lowers them to kernel pointer PARAMs. */
static inline bool poly_uop_is_shaped_value_param(const PolyUOp *u) {
  return u && u->op == POLY_OP_PARAM && u->arg.kind == POLY_ARG_PARAM && u->arg.param &&
         !u->dtype.is_ptr && !u->arg.param->name && u->n_src == 1 && u->src[0] &&
         u->src[0]->op == POLY_OP_STACK;
}

/* DFS to collect BUFFER/BUFFER_VIEW and callified shaped PARAM identities in
 * structural order. buf_order written up to POLY_MAX_REALIZE_BUFS.
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

/* Linear scan for an external storage identity in a buf_order array. */
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

/* Validates Polygrad ownership/non-NULL-source structure and the integer
 * INDEX-coordinate predicate from pinned spec_tensor. This is not a complete
 * spec_tensor implementation. Returns false with a diagnostic on stderr. */
bool poly_validate_kernel_graph(PolyCtx *ctx, PolyUOp *root);

/* Reject caller-visible explicit DEVICE identities which the current runtime
 * cannot address.  An absent DEVICE and internal DEVICE(None) remain valid;
 * CALL/FUNCTION bodies are opaque, matching pinned realization traversal. */
bool poly_uop_explicit_devices_supported(PolyCtx *ctx, PolyUOp *root);

/* Memoized backing helper for poly_uop_device(), matching tinygrad's cached
 * UOp._device property without storing pass-local cache state on every UOp. */
PolyDevice poly_uop_device_cached(PolyUOp *u, PolyMap *cache);

/* Pass-local exact concrete DEVICE identity query.  The returned UOp is owned
 * by the input graph's context; the cache owns neither keys nor values. */
PolyUOp *poly_uop_device_uop_cached(PolyCtx *ctx, PolyUOp *u, PolyMap *cache);

/* Pass-local backing cache for pinned UOp.axis. NULL/false is represented
 * explicitly in the map so shared unsharded subgraphs are not revisited. */
bool poly_uop_axis_cached(PolyCtx *ctx, const PolyUOp *u, PolyMap *cache, int *out_axis);

/* Apply one substitution map to several roots with one pass-local rewrite
 * memo. This is the allocation-free-root equivalent of tinygrad substituting
 * one temporary SINK: shared UOps are rewritten once, but no aggregate UOp is
 * interned in Polygrad's ctx-lifetime arena/CSE. */
int poly_uop_substitute_many(
    PolyCtx *ctx,
    PolyUOp **roots,
    int n_roots,
    PolyUOp **from,
    PolyUOp **to,
    int n,
    PolyUOp **out
);

/* Compile aggregate portable roots into complete physical roots using exact
 * storage-binding rows.  When physical_templates is non-NULL, those exact
 * occurrence graphs are the source and template_bindings supplies their
 * storage identities; otherwise logical_roots/logical_bindings are used.
 * This is a pure placement kernel: caller output slots change only when every
 * candidate validates, and no Tensor/Instance/cache state is mutated. */
int poly_place_roots(
    PolyCtx *ctx,
    PolyUOp **logical_roots,
    PolyUOp **physical_templates,
    int n_roots,
    PolyUOp **logical_bindings,
    PolyUOp **template_bindings,
    PolyUOp **target_bindings,
    int n_bindings,
    PolyUOp **out_roots
);

/* One explicit module region for the non-uniform scalar-device placement
 * policy.  The descriptor is pass input, not graph or Instance state: output
 * is the exact portable value root produced by the module, and inputs are the
 * exact portable roots at which its backward slice must stop. */
typedef struct {
  const char *name;
  PolyUOp *output;
  PolyUOp **inputs;
  int n_inputs;
  PolyUOp *device;
} PolyPlaceModule;

/* Compile pure portable roots under an ordered, explicit module/device map.
 * Each module is rebuilt from declared boundary inputs on its target device;
 * exact cross-module identity changes become ordinary COPY UOps.  The policy
 * derives physical binding homes from exact module regions and STORE values.
 * The operation is aggregate and atomic with respect to both output arrays. */
int poly_place_module_map(
    PolyCtx *ctx,
    PolyUOp **logical_roots,
    int n_roots,
    PolyUOp **logical_bindings,
    int n_bindings,
    const PolyPlaceModule *modules,
    int n_modules,
    PolyUOp **out_bindings,
    PolyUOp **out_roots
);

/* Resolve the exact execution device used by the physicalizer for a Tensor. */
PolyDevice poly_tensor_resolved_device(PolyCtx *ctx, PolyTensor *tensor);

/* Physicalize a snapshot of portable Tensor roots with one invocation-local
 * memo. This is an explicit re-placement boundary, never default execution;
 * no logical->physical correspondence escapes the call. */
int poly_tensor_physicalize_many(PolyCtx *ctx, PolyTensor **tensors, int n, PolyUOp **out);

/* Physical-only counterpart to pinned transform_to_call's `(graph,
 * buffer_map)` return. The caller owns and frees out_map_orig/out_map_repl;
 * no Tensor, placement, Instance, or residency state is mutated here. */
PolyUOp *poly_transform_to_call_with_map(
    PolyCtx *ctx,
    PolyUOp **uops,
    int n,
    PolyUOp **out_uops,
    PolyUOp ***out_map_orig,
    PolyUOp ***out_map_repl,
    int *out_map_n
);

#ifdef __cplusplus
}
#endif

#endif /* POLY_FRONTEND_INTERNAL_H */
