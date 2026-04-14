/* realize.c -- graph-driven realize: reads buffers from ctx->buffers
 * side table (managed via device.h), no external bindings needed.
 *
 * Mirrors poly_realize() in frontend.c but reads input data from the side
 * table instead of an external PolyBufferBinding[] array.
 */

#include "realize.h"
#include "device.h"
#include "ctx.h"
#include "exec_plan.h"
#include "frontend.h"

#include <stdio.h>
#include <stdlib.h>

int poly_realize_graph(PolyCtx *ctx, PolyUOp *tensor_sink) {
  if (!ctx || !tensor_sink || tensor_sink->op != POLY_OP_SINK) {
    fprintf(stderr, "poly_realize_graph: expected SINK\n");
    return -1;
  }

  /* Schedule (poly_schedule_for handles BIND stripping internally and stores
   * default_vars on the schedule, merged with runtime vars in
   * poly_compiled_plan_run). */
  PolySchedule *sched = poly_schedule_for(ctx, tensor_sink, POLY_MODE_CALL);
  if (!sched) {
    fprintf(stderr, "poly_realize_graph: scheduling failed\n");
    return -1;
  }

  /* Infer target device from buffers in the side table. Walk non-intermediate
   * buf_slots, find the first non-CPU domain. Fall back to CPU. */
  PolyDevice device = POLY_DEVICE_CPU;
  for (int s = 0; s < sched->n_buf_slots; s++) {
    if (sched->buf_slots[s].is_intermediate) continue;
    PolyBuffer *b = poly_buffer_get(ctx, sched->buf_slots[s].buf_uop);
    if (b && b->device != POLY_DEVICE_CPU && b->device != POLY_DEVICE_AUTO) {
      device = b->device;
      break;
    }
  }

  /* Compile */
  PolyCompiledPlan *plan = poly_compile_schedule(ctx, sched, device);
  if (!plan) {
    fprintf(stderr, "poly_realize_graph: compile failed\n");
    poly_schedule_free(sched);
    return -1;
  }

  /* Build slot_data from side table */
  void **slot_data = calloc((size_t)sched->n_buf_slots, sizeof(void *));
  int ret = -1;
  if (!slot_data) goto cleanup;

  for (int s = 0; s < sched->n_buf_slots; s++) {
    if (sched->buf_slots[s].is_intermediate) continue;
    PolyUOp *buf_uop = sched->buf_slots[s].buf_uop;
    PolyBuffer *b = poly_buffer_get(ctx, buf_uop);
    if (!b || !b->ptr) {
      fprintf(stderr, "poly_realize_graph: buffer slot %d has no data attached\n", s);
      goto cleanup;
    }
    if (b->device != device) {
      fprintf(
          stderr,
          "poly_realize_graph: buffer slot %d on device %d but kernel target is %d "
          "(cross-device transfer not yet supported)\n",
          s, b->device, device
      );
      goto cleanup;
    }
    slot_data[s] = b->ptr;
  }

  ret = poly_compiled_plan_run(plan, slot_data, sched->n_buf_slots, NULL, 0);

cleanup:
  /* TODO: cache schedule + plan per (ctx, sink, device) for repeated realize.
   * For now, free inline to avoid leaks. */
  free(slot_data);
  poly_compiled_plan_free(plan);
  poly_schedule_free(sched);
  return ret;
}
