/*
 * Manual tensor graph -> portable Instance.
 *
 * Build:
 *   cc -Isrc examples/tensor_instance_export.c -Lbuild -lpolygrad -lm -ldl -o temp/tensor_instance_export
 *   LD_LIBRARY_PATH=build ./temp/tensor_instance_export
 */

#include "instance.h"
#include "polygrad.h"
#include <stdio.h>

int main(void) {
  PolyCtx *ctx = poly_ctx_new();
  int64_t shape[] = {4};

  PolyUOp *w = poly_buffer(ctx, POLY_FLOAT32, 4);
  float w_data[] = {2.0f, 3.0f, 4.0f, 5.0f};
  poly_buffer_set(ctx, w, w_data, sizeof(w_data), POLY_DEVICE_CPU);
  poly_register_existing_buffer(ctx, POLY_ROLE_PARAM, w, shape, 1, "w", true);

  PolyUOp *x =
      poly_register_buffer_by_id(ctx, POLY_ROLE_INPUT, poly_dtype_id_by_name("float32"), shape, 1, "x");
  PolyUOp *out = poly_register_buffer_by_id(
      ctx, POLY_ROLE_OUTPUT, poly_dtype_id_by_name("float32"), shape, 1, "output"
  );

  PolyUOp *prod = poly_alu2(ctx, POLY_OP_MUL, x, w);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, prod));
  const char *names[] = {"forward"};
  PolyUOp *sinks[] = {sink};
  PolyInstance *inst = poly_instance_from_sinks(ctx, names, sinks, 1);

  float x_data[] = {10.0f, 10.0f, 10.0f, 10.0f};
  PolyIOBinding io[] = {{"x", x_data}};
  if (poly_instance_forward(inst, io, 1) != 0) return 1;

  int64_t numel = 0;
  float *y = poly_instance_buf_data_named(inst, "output", &numel);
  printf("output:");
  for (int64_t i = 0; i < numel; i++) printf(" %.1f", y[i]);
  printf("\n");

  poly_instance_free(inst);
  poly_ctx_destroy(ctx);
  return 0;
}
