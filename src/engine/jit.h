/* jit.h -- internal JIT capture/replay hooks.
 *
 * Public JIT API is declared in polygrad.h. This header only exposes the
 * capture hook needed by realize.c, matching tinygrad's separate engine/jit.py
 * and engine/realize.py responsibilities.
 */

#ifndef POLYGRAD_ENGINE_JIT_H
#define POLYGRAD_ENGINE_JIT_H

#include "polygrad.h"
#include "engine/schedule.h"

#ifdef __cplusplus
extern "C" {
#endif

bool poly_jit_is_capturing(PolyJit *jit);
int poly_jit_record_schedule(PolyJit *jit, PolySchedule *sched);
/* Pinned CapturedJit.linear analogue for topology tests and graph lowering. */
PolyUOp *poly_jit_captured_linear(PolyJit *jit);
/* Pinned jit_lower retained-executable boundary for an already-built schedule.
 * Tensor-facing PolyJit capture and logical-program Instance execution share
 * this core lowering path; callers retain the returned compiled handle and may
 * free the input schedule immediately. */
PolyCompiledSchedule *poly_jit_lower(PolyCtx *ctx, PolySchedule *schedule);

#ifdef __cplusplus
}
#endif

#endif /* POLYGRAD_ENGINE_JIT_H */
