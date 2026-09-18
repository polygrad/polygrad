/* jit.h -- Tensor JIT capture/replay and engine hooks. */

#ifndef POLYGRAD_ENGINE_JIT_H
#define POLYGRAD_ENGINE_JIT_H

#include "../core.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Tinygrad-style raw Tensor JIT capture/replay.
 *
 * This is deliberately a tensor/schedule-layer object, not an Model
 * entrypoint plan. Capture records the LINEAR schedules produced by normal
 * poly_realize_tensors calls. The retained physical LINEAR substitutes only
 * JIT input BUFFERs with shaped PARAMs and replays the same compiled plan with
 * current input identities, matching tinygrad's CapturedJit input_uops
 * substitution boundary. Input
 * compatibility and runtime variable values follow
 * tinygrad/engine/jit.py:_prepare_jit_inputs: the physical base is replaced by
 * NOOP, the remaining view is unbound, and current AFTER/STORE values override the
 * captured schedule defaults. Polygrad's logical provenance root is not part
 * of this execution-layer signature.
 */
PolyJit *poly_jit_new(PolyCtx *ctx);
void poly_jit_free(PolyJit *jit);
int poly_jit_set_prune(PolyJit *jit, bool prune);
int poly_jit_begin_capture(PolyJit *jit, PolyTensor **inputs, int n_inputs);
int poly_jit_end_capture(PolyJit *jit, PolyTensor **live_tensors, int n_live_tensors);
void poly_jit_cancel_capture(PolyJit *jit);
bool poly_jit_is_captured(PolyJit *jit);
int poly_jit_schedule_count(PolyJit *jit);
int poly_jit_run(PolyJit *jit, PolyTensor **inputs, int n_inputs);
int poly_jit_run_with_vars(
    PolyJit *jit,
    PolyTensor **inputs,
    int n_inputs,
    PolyVarBinding *var_bindings,
    int n_var_bindings
);

bool poly_jit_is_capturing(PolyJit *jit);
/* Current Tinygrad engine/jit.py:create_graph_call. */
PolyUOp *poly_create_graph_call(PolyCtx *ctx, PolyUOp **calls, int n_calls);
int poly_jit_record_linear(
    PolyJit *jit,
    PolyUOp *linear,
    PolyVarBinding *var_bindings,
    int n_var_bindings
);
/* Pinned CapturedJit.linear analogue for topology tests and graph lowering. */
PolyUOp *poly_jit_captured_linear(PolyJit *jit);
/* C mechanics for Tinygrad engine/jit.py's buffers|all_tensors held set. */
int poly_jit_collect_held_bufs(
    PolyCtx *ctx,
    PolyTensor **live_tensors,
    int n_live_tensors,
    PolyUOp ***held_out,
    int *n_held_out
);
/* Current Tinygrad engine/jit.py:jit_lower. */
PolyUOp *poly_jit_lower(
    PolyCtx *ctx,
    PolyUOp *linear,
    PolyUOp **held_bufs,
    int n_held_bufs,
    PolyUOp **input_uops,
    int n_input_uops
);

#ifdef __cplusplus
}
#endif

#endif /* POLYGRAD_ENGINE_JIT_H */
