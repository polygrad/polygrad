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

int poly_realize_sink(PolyCtx *ctx, PolyUOp *tensor_sink) {
  if (!ctx || !tensor_sink || tensor_sink->op != POLY_OP_SINK) {
    fprintf(stderr, "poly_realize_sink: expected SINK\n");
    return -1;
  }

  /* Schedule (poly_schedule_for handles BIND stripping internally and stores
   * default_vars on the schedule, merged with runtime vars in
   * poly_compiled_plan_run). */
  PolySchedule *sched = poly_schedule_for(ctx, tensor_sink, POLY_MODE_CALL);
  if (!sched) {
    fprintf(stderr, "poly_realize_sink: scheduling failed\n");
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
    fprintf(stderr, "poly_realize_sink: compile failed\n");
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
      fprintf(stderr, "poly_realize_sink: buffer slot %d has no data attached\n", s);
      goto cleanup;
    }
    if (b->device != device) {
      fprintf(
          stderr,
          "poly_realize_sink: buffer slot %d on device %d but kernel target is %d "
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

int poly_realize_uops(PolyCtx *ctx, PolyUOp **uops, int n, PolyUOp **out_uops) {
  if (!ctx || !uops || !out_uops || n < 0) return -1;
  if (n == 0) return 0;

  PolyUOp **stores = calloc((size_t)n, sizeof(PolyUOp *));
  if (!stores) return -1;
  int n_stores = 0;

  for (int i = 0; i < n; i++) {
    PolyUOp *u = uops[i];
    if (!u) {
      out_uops[i] = NULL;
      continue;
    }
    if (poly_uop_has_buffer_identity(u)) {
      out_uops[i] = u;
      continue;
    }

    /* ASSIGN writes back to its target buffer (src[0]) in place. Don't
     * wrap it in STORE(new_buf, ASSIGN); emit it directly into the SINK
     * and report the target buffer as the realized UOp. */
    if (u->op == POLY_OP_ASSIGN && u->n_src >= 1) {
      stores[n_stores++] = u;
      out_uops[i] = u->src[0];
      continue;
    }

    PolyShape shape = poly_uop_shape_cached(ctx, u);
    int64_t numel = (shape.ndim >= 0) ? poly_shape_numel(shape) : 1;
    if (numel < 1) numel = 1;

    PolyUOp *buf = poly_buffer(ctx, poly_dtype_scalar(u->dtype), numel);
    if (!buf || poly_buffer_allocate(ctx, buf, POLY_DEVICE_HOST) != 0) {
      fprintf(stderr, "poly_realize_uops: buffer allocate failed\n");
      free(stores);
      return -1;
    }
    stores[n_stores++] = poly_store_val(ctx, buf, u);
    out_uops[i] = (shape.ndim > 1) ? poly_reshape(ctx, buf, shape.dims, shape.ndim) : buf;
  }

  int ret = 0;
  if (n_stores > 0) {
    PolyUOp *sink = poly_sink_n(ctx, stores, n_stores);
    ret = poly_realize_sink(ctx, sink);
  }
  free(stores);
  return ret;
}

PolyUOp *poly_realize_uop(PolyCtx *ctx, PolyUOp *uop) {
  PolyUOp *out = NULL;
  if (poly_realize_uops(ctx, &uop, 1, &out) != 0) return NULL;
  return out;
}
