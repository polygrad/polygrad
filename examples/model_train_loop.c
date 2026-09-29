/*
 * Model-owned construction and a custom training loop.
 *
 * Build:
 *   cc -Isrc examples/model_train_loop.c -Lbuild -lpolygrad -lm -ldl -o temp/model_train_loop
 *   LD_LIBRARY_PATH=build ./temp/model_train_loop
 */
#include "polygrad.h"
#include "model.h"
#include <math.h>
#include <stdio.h>

int main(void) {
  PolyCtx *ctx = poly_ctx_new();
  if (!ctx) return 1;
  int rc = 1;
  PolyModel *model = poly_model_new(ctx, NULL);
  if (!model) goto done;
  int64_t shape[] = {1};
  PolyTensor *w = poly_model_param(model, "w", POLY_FLOAT32, shape, 1);
  PolyTensor *x = poly_model_input(model, "x", POLY_FLOAT32, shape, 1);
  PolyTensor *y = poly_model_target(model, "y", POLY_FLOAT32, shape, 1);
  PolyTensor *prediction = poly_tensor_alu2(ctx, POLY_OP_MUL, x, w);
  PolyTensor *error = poly_tensor_alu2(ctx, POLY_OP_SUB, prediction, y);
  PolyTensor *loss = poly_tensor_alu2(ctx, POLY_OP_MUL, error, error);
  const char *inputs[] = {"x", "y"}, *outputs[] = {"loss"};
  PolyEntrypointOptions options = {.objective = "loss"};
  int failed = poly_model_output(model, "loss", loss) ||
               poly_model_entrypoint(model, "loss", inputs, 2, outputs, 1, &options) ||
               poly_model_build(model, NULL);
  poly_tensor_release(loss);
  poly_tensor_release(error);
  poly_tensor_release(prediction);
  if (failed) goto done;
  float initial = 1;
  if (poly_model_write_buf_named(model, "w", &initial, sizeof(initial)) ||
      poly_model_set_optimizer(model, POLY_OPTIM_SGD, .1f, 0, 0, 0, 0))
    goto done;
  float x_data[] = {1}, y_data[] = {3};
  PolyIOBinding io[] = {
      POLY_IO_BINDING_ARRAY("x", x_data, POLY_FLOAT32),
      POLY_IO_BINDING_ARRAY("y", y_data, POLY_FLOAT32)};
  float first = 0, last = 0;
  for (int step = 0; step < 8; step++) {
    if (poly_model_train_step(model, NULL, io, 2, &last)) goto done;
    if (!step) first = last;
  }
  printf("loss %.6f -> %.6f\n", first, last);
  rc = !(isfinite(last) && last < first);
done:
  poly_model_free(model);
  poly_ctx_destroy(ctx);
  return rc;
}
