/* realize.h -- tinygrad-parity realize: graph is source of truth.
 *
 * poly_realize_graph(ctx, sink) walks the SINK, finds BUFFER UOps,
 * looks up handles from ctx's buffer_handles side table, extracts
 * BIND values for var_vals, and executes. No external bindings needed.
 *
 * Callers attach handles before realize:
 *   poly_ctx_set_handle(ctx, buf_uop, handle);
 *   poly_realize_graph(ctx, sink);
 *   PolyBuffer *out = poly_ctx_get_handle(ctx, out_buf);
 */

#ifndef POLYGRAD_REALIZE_H
#define POLYGRAD_REALIZE_H

#include "polygrad.h"
#include "exec_plan.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Attach a buffer handle to a BUFFER UOp in the context side table.
 * The handle is copied; caller retains ownership of the underlying memory. */
void poly_ctx_set_handle(PolyCtx *ctx, PolyUOp *buf, PolyBuffer handle);

/* Look up the handle for a BUFFER UOp. Returns NULL if not attached. */
PolyBuffer *poly_ctx_get_handle(PolyCtx *ctx, PolyUOp *buf);

/* Remove a handle from the side table. Returns true if found and removed. */
bool poly_ctx_remove_handle(PolyCtx *ctx, PolyUOp *buf);

/* Tinygrad-parity realize: walks the SINK, finds buffers from side table,
 * extracts BIND values, schedules, compiles, executes.
 * Returns 0 on success, -1 on error.
 *
 * All BUFFER UOps referenced by the SINK must have handles attached
 * via poly_ctx_set_handle before calling. Intermediate buffers created
 * by the scheduler are allocated automatically.
 *
 * Equivalent to tinygrad's Tensor.realize() — the graph is the source
 * of truth; no external bindings are needed. */
int poly_realize_graph(PolyCtx *ctx, PolyUOp *tensor_sink);

#ifdef __cplusplus
}
#endif

#endif /* POLYGRAD_REALIZE_H */
