#ifndef POLYGRAD_UOP_OPS_H
#define POLYGRAD_UOP_OPS_H

#include "../polygrad.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Current tinygrad/uop/ops.py:axis_letters and range_str. */
char poly_axis_letter(PolyArg arg);
char *poly_range_str(PolyArg arg);
/* Compare the complete (axis id, split path, AxisType) range key. */
int poly_range_arg_cmp(PolyArg a, PolyArg b);

/* UOp.vconst_like: a scalar or flat STACK after movement lowering. */
PolyUOp *poly_vconst_like(PolyCtx *ctx, PolyUOp *ref, PolyArg val);

/* Heap-backed result of current Tinygrad UOp.split_uop. Caller frees it. */
PolyUOp **poly_uop_split(PolyUOp *u, PolyOps sep, int *n_out);

/* Complete UOp.ranges in insertion order. Borrowed immutable cache storage:
 * keep u alive, do not free/sort the array, and do not cross a GC safe point
 * without retaining u. Empty sets still return a non-NULL array. */
PolyUOp *const *poly_uop_ranges_view(PolyCtx *ctx, PolyUOp *u, int *n_out);

/* UOp.divides: prove exact divisibility structurally, not by sampled bounds. */
PolyUOp *poly_uop_divides(PolyCtx *ctx, PolyUOp *u, int64_t factor);

/* Tinygrad helpers.py:is_image_shape on the UOp's encoded shape. */
bool poly_uop_is_image_shape(PolyCtx *ctx, const PolyUOp *u);

/* Tinygrad 2026-08-22/a9069c177a9d uop/ops.py:UOp.get_idx/get_valid. */
PolyUOp *poly_uop_get_idx(PolyCtx *ctx, PolyUOp *u);
PolyUOp *poly_uop_get_valid(PolyCtx *ctx, PolyUOp *u);

/* Tinygrad 2026-08-22/a9069c177a9d uop/ops.py:UOp.addrspace. */
bool poly_uop_addrspace(const PolyUOp *u, PolyAddrSpace *out);

/* UOp.key content identity, excluding tags. Caller frees the returned bytes;
 * NULL disables persistent caching for unsupported process-local payloads. */
uint8_t *poly_uop_key(PolyCtx *ctx, PolyUOp *root, size_t *size);

/* C mechanics for Tinygrad's weak UOpMetaClass.ucache and Python-owned UOps. */
int poly_uop_cse_evict_unmarked(PolyCtx *ctx, PolyMap *live);
bool poly_uop_storage_contains(PolyCtx *ctx, const void *ptr);
void poly_uop_storage_destroy_all(PolyCtx *ctx);

/* tinygrad's callified function uses value PARAMs as external storage
 * identities until rangeify lowers them to kernel pointer PARAMs. */
static inline bool poly_uop_is_shaped_value_param(const PolyUOp *u) {
  return u && u->op == POLY_OP_PARAM && u->arg.kind == POLY_ARG_PARAM && u->arg.param &&
         !u->arg.param->name && u->n_src == 1 && u->src[0] &&
         (u->src[0]->op == POLY_OP_STACK || poly_dtype_is_int(u->src[0]->dtype));
}

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

#ifdef POLY_TESTING
/* Fail one nonempty substitution after count successful calls; -1 disables. */
void poly_test_substitute_fail_after(int count);
#endif

#ifdef __cplusplus
}
#endif

#endif
