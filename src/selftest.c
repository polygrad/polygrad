/* selftest.c -- embeddable runtime sanity checks */

#include "polygrad.h"
#include "device.h"
#include "engine/schedule.h"
#include "pat.h"

#include <math.h>
#include <stddef.h>

static int selftest_alu(void) {
  PolyArg add_ops[2] = {poly_arg_int(2), poly_arg_int(3)};
  PolyArg add = poly_exec_alu(POLY_OP_ADD, POLY_INT32, add_ops, 2);
  if (add.kind != POLY_ARG_INT || add.i != 5) return -1;

  PolyArg cmp_ops[2] = {
      poly_arg_int(INT64_C(9007199254740992)),
      poly_arg_int(INT64_C(9007199254740993)),
  };
  PolyArg cmp = poly_exec_alu(POLY_OP_CMPNE, POLY_INT64, cmp_ops, 2);
  if (cmp.kind != POLY_ARG_BOOL || !cmp.b) return -1;

  PolyArg div_ops[2] = {poly_arg_float(0.0), poly_arg_float(0.0)};
  PolyArg div = poly_exec_alu(POLY_OP_FDIV, POLY_FLOAT32, div_ops, 2);
  if (div.kind != POLY_ARG_FLOAT || !isnan(div.f)) return -1;

  return 0;
}

int poly_selftest_device(PolyDevice device) {
  if (selftest_alu() != 0) return -1;

  if (device == POLY_DEVICE_AUTO) device = POLY_DEVICE_INTERP;
  if (device == POLY_DEVICE_HOST || !poly_device_can_execute(device)) return -1;

  int rc = -1;
  PolyCtx *ctx = poly_ctx_new();
  PolySchedule *sched = NULL;
  if (!ctx) return -1;
  poly_ctx_set_preferred_device(ctx, device);

  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, 4);
  PolyUOp *b = poly_buffer(ctx, POLY_FLOAT32, 4);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, 4);
  if (!a || !b || !out) goto cleanup;

  const float av[4] = {1.0f, -2.0f, 3.5f, 8.0f};
  const float bv[4] = {10.0f, 20.0f, -4.5f, 0.25f};
  const float expect[4] = {11.0f, 18.0f, -1.0f, 8.25f};
  float got[4] = {0};

  if (poly_buffer_write(ctx, a, av, sizeof(av)) != 0) goto cleanup;
  if (poly_buffer_write(ctx, b, bv, sizeof(bv)) != 0) goto cleanup;

  PolyUOp *sum = poly_alu2(ctx, POLY_OP_ADD, a, b);
  PolyUOp *store = poly_store_val(ctx, out, sum);
  PolyUOp *sink = poly_sink1(ctx, store);
  if (!sum || !store || !sink) goto cleanup;

  sched = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  if (!sched) goto cleanup;
  if (poly_run_schedule(ctx, sched, NULL, 0) != 0) goto cleanup;
  if (poly_buffer_read(ctx, out, got, sizeof(got)) != 0) goto cleanup;

  for (int i = 0; i < 4; i++) {
    if (fabsf(got[i] - expect[i]) > 1e-6f) goto cleanup;
  }
  rc = 0;

cleanup:
  poly_schedule_free(sched);
  poly_ctx_destroy(ctx);
  return rc;
}

int poly_selftest(void) {
  return poly_selftest_device(POLY_DEVICE_INTERP);
}
