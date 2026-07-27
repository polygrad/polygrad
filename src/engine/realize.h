/* realize.h -- graph-driven realize.
 *
 * This is the C-core analogue of tinygrad's Tensor.realize ->
 * Tensor.schedule_with_vars -> run_schedule path. Callers pass the unrealized
 * top-level value UOps they want materialized; poly_realize_uops callifies
 * them into an internal effect SINK, then shares the same sink runner used by
 * already-schedule-ready instance/imported graphs.
 */

#ifndef POLYGRAD_REALIZE_H
#define POLYGRAD_REALIZE_H

#include "polygrad.h"
#include "engine/schedule.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Explicit tensor-graph to call-graph stage, mirroring tinygrad's
 * transform_to_call(...) boundary. It batches the requested top-level value
 * UOps into one realizable CALL whose src[0] is the SINK body to schedule,
 * and returns the buffer-identity replacements in out_uops. Returns NULL when
 * there is nothing to run or on error. This raw-UOp entrypoint receives an
 * already-physical graph and therefore performs no logical Tensor placement
 * inference; poly_realize_tensors retains exact placement correspondence
 * through the private production variant. */
PolyUOp *poly_transform_to_call(PolyCtx *ctx, PolyUOp **uops, int n, PolyUOp **out_uops);

/* C-core analogue of tinygrad's Tensor.schedule_with_vars. Runs the local
 * transform-to-call stage on the requested top-level value UOps, allocates
 * fresh output buffers for unrealized values, and returns the schedule-level
 * object that can be passed to poly_run_schedule. Already-realized inputs pass
 * through unchanged in out_uops. Successful zero-work lowering is an explicit
 * zero-CALL schedule; NULL means error. Like poly_transform_to_call, this
 * raw-UOp entrypoint performs no logical Tensor placement inference. */
PolySchedule *poly_schedule_with_vars(PolyCtx *ctx, PolyUOp **uops, int n, PolyUOp **out_uops);

/* Normalize an already-effectful concrete SINK to tinygrad's shaped-PARAM
 * function/outer-CALL boundary and return its resolved schedule. This skips
 * tensor output allocation but not input-buffer callification. */
PolySchedule *poly_schedule_effect_sink(PolyCtx *ctx, PolyUOp *sink);

/* Run an already-effectful concrete SINK using ctx->buffers runtime state. This
 * is the execution layer for imported/Instance STORE/ASSIGN/AFTER graphs. */
int poly_realize_sink(PolyCtx *ctx, PolyUOp *sink);

/* Materialize the requested top-level value UOps using ctx->buffers runtime
 * state. Already-realized inputs pass through unchanged in out_uops. Non-leaf
 * values are assigned fresh output buffers internally, scheduled as one batch,
 * and the realized replacements are returned in out_uops.
 *
 * Returns 0 on success, -1 on error. */
int poly_realize_uops(PolyCtx *ctx, PolyUOp **uops, int n, PolyUOp **out_uops);

#ifdef __cplusplus
}
#endif

#endif /* POLYGRAD_REALIZE_H */
