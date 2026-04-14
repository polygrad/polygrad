/* realize.h -- graph-driven realize.
 *
 * poly_realize_graph(ctx, sink) walks the SINK, finds BUFFER UOps,
 * looks up data from ctx->buffers side table (managed via device.h),
 * extracts BIND values, schedules, compiles, executes.
 *
 * Usage:
 *   poly_buffer_set(ctx, buf_uop, ptr, nbytes, device);  // attach data
 *   poly_realize_graph(ctx, sink);                       // execute
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
int poly_realize_graph(PolyCtx *ctx, PolyUOp *tensor_sink);

#ifdef __cplusplus
}
#endif

#endif /* POLYGRAD_REALIZE_H */
