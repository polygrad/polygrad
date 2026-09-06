/* realize.h -- graph-driven realize.
 *
 * This is the C-core analogue of current Tinygrad's Tensor.realize ->
 * Tensor.linear_with_vars -> run_linear path. Callers pass the unrealized
 * top-level value UOps they want materialized; poly_realize_uops callifies
 * them into an internal effect SINK, then shares the same sink runner used by
 * already-schedule-ready Model/imported graphs.
 */

#ifndef POLYGRAD_REALIZE_H
#define POLYGRAD_REALIZE_H

#include "polygrad.h"
#include "engine/schedule.h"
#include "schedule/schedule.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Explicit tensor-graph to call-graph stage, mirroring tinygrad's
 * transform_to_call(...) boundary. It batches the requested top-level value
 * UOps into one realizable CALL whose src[0] is the SINK body to schedule,
 * and returns replacements in out_uops. Nonmaterialized values pass through;
 * no effects yields CALL(SINK()). Returns NULL for zero inputs or on error.
 * This raw-UOp entrypoint receives an already-physical graph and therefore
 * performs no logical Tensor placement
 * inference. Default Tensor realization uses this same physical-only path. */
PolyUOp *poly_transform_to_call(PolyCtx *ctx, PolyUOp **uops, int n, PolyUOp **out_uops);

/* C-core analogue of tinygrad's Tensor.linear_with_vars. Runs the local
 * transform-to-call stage on the requested top-level value UOps, allocates
 * fresh output buffers for unrealized values, and returns `LINEAR`. */
PolyUOp *poly_linear_with_vars(
    PolyCtx *ctx,
    PolyUOp **uops,
    int n,
    PolyUOp **out_uops,
    PolyVarBinding **var_bindings_out,
    int *n_var_bindings_out
);

/* Normalize an already-effectful concrete SINK to tinygrad's shaped-PARAM
 * function/outer-CALL boundary and return its resolved LINEAR. This skips
 * tensor output allocation but not input-buffer callification. */
PolyUOp *poly_linear_effect_sink(
    PolyCtx *ctx,
    PolyUOp *sink,
    PolyVarBinding **var_bindings_out,
    int *n_var_bindings_out
);

/* Admission predicate shared by default Tensor realization and retained
 * physical Model templates. It follows caller-visible graph inputs while
 * treating CALL/FUNCTION bodies as opaque, and rejects any BUFFER whose device
 * remains AUTO. */
bool poly_tensor_root_has_unplaced_buffer(PolyCtx *ctx, PolyUOp *root);

/* Current Tinygrad engine/realize.py:run_linear. LINEAR is the schedule;
 * execution resolves its CALL arguments directly through UOp buffer identity. */
int poly_run_linear(
    PolyCtx *ctx,
    PolyUOp *linear,
    PolyVarBinding *var_bindings,
    int n_var_bindings,
    PolyUOp **input_uops,
    int n_input_uops,
    bool update_stats,
    bool jit,
    bool wait
);

/* Run an already-effectful concrete SINK using ctx->buffers runtime state. This
 * is the execution layer for imported/Model STORE/ASSIGN/AFTER graphs. */
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
