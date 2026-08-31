#ifndef POLYGRAD_BENCH_BUFFER_H
#define POLYGRAD_BENCH_BUFFER_H

#include "../src/ctx.h"
#include "../src/device.h"

/* C argument adaptation for current Tinygrad UOp.new_buffer. */
static inline PolyUOp *poly_bench_buffer(
    PolyCtx *ctx,
    PolyDType dtype,
    int64_t size,
    PolyDevice device
) {
  PolyUOp *device_uop = poly_device_uop(ctx, device);
  return device_uop
             ? poly_uop_new_buffer(ctx, device_uop, size, dtype, poly_ctx_next_unique_id(ctx))
             : NULL;
}

#endif
