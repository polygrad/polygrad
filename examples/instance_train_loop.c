/*
 * Manual graph + custom training loop through PolyInstance.
 *
 * Build:
 *   cc -Isrc examples/instance_train_loop.c -Lbuild -lpolygrad -lm -ldl -o temp/instance_train_loop
 *   LD_LIBRARY_PATH=build ./temp/instance_train_loop
 */

#include "instance.h"
#include "polygrad.h"
#include "tensor.h"
#include <stdio.h>

int main(void) {
  PolyCtx *ctx = poly_ctx_new();
  int64_t shape[] = {1};

  PolyUOp *w = poly_buffer(ctx, POLY_FLOAT32, 1);
  float w_data[] = {1.0f};
  poly_buffer_set(ctx, w, w_data, sizeof(w_data), POLY_DEVICE_CPU);
  poly_register_existing_buffer(ctx, POLY_ROLE_PARAM, w, shape, 1, "w", true);

  PolyUOp *x =
      poly_register_buffer_by_id(ctx, POLY_ROLE_INPUT, poly_dtype_id_by_name("float32"), shape, 1, "x");
  PolyUOp *y =
      poly_register_buffer_by_id(ctx, POLY_ROLE_TARGET, poly_dtype_id_by_name("float32"), shape, 1, "y");
  PolyUOp *loss_buf = poly_register_buffer_by_id(
      ctx, POLY_ROLE_OUTPUT, poly_dtype_id_by_name("float32"), shape, 1, "loss"
  );

  PolyUOp *pred = poly_alu2(ctx, POLY_OP_MUL, x, w);
  PolyUOp *loss = poly_mse_loss(ctx, pred, y);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, loss_buf, loss));
  const char *names[] = {"loss"};
  PolyUOp *sinks[] = {sink};
  PolyInstance *inst = poly_instance_from_sinks(ctx, names, sinks, 1);

  poly_instance_set_optimizer(inst, POLY_OPTIM_SGD, 0.1f, 0.0f, 0.0f, 0.0f, 0.0f);
  float x_data[] = {1.0f};
  float y_data[] = {3.0f};
  PolyIOBinding io[] = {{"x", x_data}, {"y", y_data}};
  float first = 0.0f, last = 0.0f;
  for (int step = 0; step < 8; step++) {
    poly_instance_train_step(inst, io, 2, &last);
    if (step == 0) first = last;
  }
  printf("loss %.6f -> %.6f\n", first, last);

  poly_instance_free(inst);
  poly_ctx_destroy(ctx);
  return 0;
}
