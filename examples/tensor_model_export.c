/*
 * Tensor graph -> portable Model with explicit, Model-owned bindings.
 *
 * Build:
 *   cc -Isrc examples/tensor_model_export.c -Lbuild -lpolygrad -lm -ldl -o temp/tensor_model_export
 *   LD_LIBRARY_PATH=build ./temp/tensor_model_export
 */
#include "polygrad.h"
#include "model.h"
#include "bundle.h"
#include <stdio.h>
#include <stdlib.h>

int main(void) {
  PolyCtx *ctx = poly_ctx_new();
  if (!ctx) return 1;
  int rc = 1, size = 0;
  uint8_t *bundle = NULL;
  PolyModel *restored = NULL;
  int64_t shape[] = {4};
  float weights[] = {2, 3, 4, 5};
  PolyTensor *w = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_CPU);
  PolyTensor *x = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_CPU);
  PolyTensor *out = poly_tensor_alu2(ctx, POLY_OP_MUL, x, w);
  PolyBindingSpec bindings[] = {
      {"w", POLY_ROLE_PARAM, w, 0},
      {"x", POLY_ROLE_INPUT, x, 0},
      {"output", POLY_ROLE_OUTPUT, out, 0}};
  const char *inputs[] = {"x"}, *outputs[] = {"output"};
  PolyEntrypointSpec entry = {
      .name = "forward", .inputs = inputs, .n_inputs = 1, .outputs = outputs, .n_outputs = 1};
  PolyModel *model = poly_model_from_bindings(ctx, bindings, 3, &entry, 1, NULL, NULL);
  poly_tensor_release(out);
  poly_tensor_release(x);
  poly_tensor_release(w);
  if (!model || poly_model_write_buf_named(model, "w", weights, sizeof(weights))) goto done;
  bundle = poly_model_save_bundle(model, &size);
  if (!bundle) goto done;
  restored = poly_model_from_bundle_into(ctx, bundle, size, POLY_DEVICE_CPU);
  if (!restored) goto done;
  float values[] = {10, 10, 10, 10}, result[4];
  PolyIOBinding io = POLY_IO_BINDING_ARRAY("x", values, POLY_FLOAT32);
  if (poly_model_forward(restored, &io, 1) ||
      poly_model_read_buf_named(restored, "output", result, sizeof(result)))
    goto done;
  printf("output:");
  for (int i = 0; i < 4; i++) {
    printf(" %.1f", result[i]);
    if (result[i] != weights[i] * values[i]) goto done;
  }
  printf("\n");
  rc = 0;
done:
  free(bundle);
  poly_model_free(restored);
  poly_model_free(model);
  /* Both Models borrow ctx; destroy it only after their handles. */
  poly_ctx_destroy(ctx);
  return rc;
}
