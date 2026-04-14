/* realize.h -- graph-driven realize.
 *
 * poly_realize_sink(ctx, sink) walks the SINK, finds BUFFER UOps,
 * looks up data from ctx->buffers side table (managed via device.h),
 * extracts BIND values, schedules, compiles, executes.
 *
 * Usage:
 *   poly_buffer_set(ctx, buf_uop, ptr, nbytes, device);  // attach data
 *   poly_realize_sink(ctx, sink);                       // execute
 *   PolyBuffer *out = poly_buffer_get(ctx, out_buf);     // read result
 */

#ifndef POLYGRAD_REALIZE_H
#define POLYGRAD_REALIZE_H

#include "polygrad.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Walks the SINK, reads buffers from ctx->buffers (attached via poly_buffer_set
 * from device.h), extracts BIND values, schedules, compiles, executes.
 * Returns 0 on success, -1 on error.
 *
 * All BUFFER UOps in the SINK must have data attached via poly_buffer_set
 * before calling. */
int poly_realize_sink(PolyCtx *ctx, PolyUOp *tensor_sink);

/* Frontend-facing realize. Materializes a single value UOp and returns its
 * realized buffer-identity UOp. If `uop` already has buffer identity it is
 * returned unchanged. Otherwise a fresh BUFFER (same dtype, numel from shape)
 * is allocated host-side, attached to ctx->buffers, a STORE(buf, uop) is
 * wrapped in a SINK and executed via poly_realize_sink. The returned UOp is
 * a RESHAPE(BUFFER) preserving the value's shape when ndim > 1, else the
 * BUFFER itself.
 *
 * Read output bytes via poly_buffer_get(ctx, poly_uop_get_buffer_identity(ret)).
 *
 * Returns NULL on error. */
PolyUOp *poly_realize_uop(PolyCtx *ctx, PolyUOp *uop);

/* Batched form: materializes all unrealized `uops[i]` in a single combined
 * SINK, then fills `out_uops[i]` with each realized UOp (already-identity
 * inputs pass through). Returns 0 on success, -1 on error. */
int poly_realize_uops(PolyCtx *ctx, PolyUOp **uops, int n, PolyUOp **out_uops);

#ifdef __cplusplus
}
#endif

#endif /* POLYGRAD_REALIZE_H */
