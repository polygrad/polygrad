/* selftest.c -- embeddable runtime sanity checks */

#include "polygrad.h"
#include "ctx.h"
#include "device.h"
#include "engine/realize.h"
#include "uop/upat.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>

/* C argument adaptation for current Tinygrad UOp.new_buffer. */
static PolyUOp *selftest_new_buffer(
    PolyCtx *ctx, PolyDType dtype, int64_t size, PolyDevice device
) {
  PolyUOp *device_uop = poly_device_uop(ctx, device);
  return device_uop
             ? poly_uop_new_buffer(
                   ctx, device_uop, size, dtype, poly_ctx_next_unique_id(ctx)
               )
             : NULL;
}

static int selftest_alu(void) {
  PolyArg add_ops[2] = {poly_arg_int(2), poly_arg_int(3)};
  PolyArg add = poly_exec_alu(POLY_OP_ADD, POLY_INT32, add_ops, 2, true);
  if (add.kind != POLY_ARG_INT || add.i != 5) return -1;

  PolyArg cmp_ops[2] = {
      poly_arg_int(INT64_C(9007199254740992)),
      poly_arg_int(INT64_C(9007199254740993)),
  };
  PolyArg cmp = poly_exec_alu(POLY_OP_CMPNE, POLY_INT64, cmp_ops, 2, true);
  if (cmp.kind != POLY_ARG_BOOL || !cmp.b) return -1;

  PolyArg div_ops[2] = {poly_arg_float(0.0), poly_arg_float(0.0)};
  PolyArg div = poly_exec_alu(POLY_OP_FDIV, POLY_FLOAT32, div_ops, 2, true);
  if (div.kind != POLY_ARG_FLOAT || !isnan(div.f)) return -1;

  return 0;
}

int poly_selftest_device(PolyDevice device) {
  if (selftest_alu() != 0) return -1;

  if (device == POLY_DEVICE_AUTO) device = POLY_DEVICE_INTERP;
  if (!poly_device_can_execute(device)) return -1;

  int rc = -1;
  PolyCtx *ctx = poly_ctx_new();
  PolyVarBinding *var_bindings = NULL;
  int n_var_bindings = 0;
  if (!ctx) return -1;
  poly_ctx_set_preferred_device(ctx, device);

  /* Tinygrad@2026-08-22/a9069c177a9d UOp.new_buffer creates executable
   * BUFFER(shape, ParamArg(device)); the self-test exercises execution. */
  PolyUOp *a = selftest_new_buffer(ctx, POLY_FLOAT32, 4, device);
  PolyUOp *b = selftest_new_buffer(ctx, POLY_FLOAT32, 4, device);
  PolyUOp *out = selftest_new_buffer(ctx, POLY_FLOAT32, 4, device);
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

  PolyUOp *linear = poly_linear_effect_sink(
      ctx, sink, &var_bindings, &n_var_bindings
  );
  if (!linear) goto cleanup;
  if (poly_run_linear(
          ctx, linear, var_bindings, n_var_bindings,
          NULL, 0, true, false, false
      ) != 0)
    goto cleanup;
  if (poly_buffer_read(ctx, out, got, sizeof(got)) != 0) goto cleanup;

  for (int i = 0; i < 4; i++) {
    if (fabsf(got[i] - expect[i]) > 1e-6f) goto cleanup;
  }
  rc = 0;

cleanup:
  free(var_bindings);
  poly_ctx_destroy(ctx);
  return rc;
}

int poly_selftest(void) {
  return poly_selftest_device(POLY_DEVICE_INTERP);
}
