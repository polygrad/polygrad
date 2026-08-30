/* jit.h -- internal JIT capture/replay hooks.
 *
 * Public JIT API is declared in polygrad.h. This header only exposes the
 * capture hook needed by realize.c, matching tinygrad's separate engine/jit.py
 * and engine/realize.py responsibilities.
 */

#ifndef POLYGRAD_ENGINE_JIT_H
#define POLYGRAD_ENGINE_JIT_H

#include "polygrad.h"
#include "engine/realize.h"

#ifdef __cplusplus
extern "C" {
#endif

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
