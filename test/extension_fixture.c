/* Independently compiled author. No Polygrad implementation is linked. */
#include "polygrad.h"

int extension_density(PolyCtx *ctx, PolyTensor **inputs, int kind, PolyTensor **outputs) {
  PolyTensor *owned[16] = {0};
  int count = 0, ok = 0;
  int64_t axis = 0;
  outputs[0] = outputs[1] = NULL;
  if (kind == 3) return poly_tensor_sort(ctx, inputs[0], 0, 0, outputs, outputs + 1) == 0;
  if (kind == 4) {
    int64_t pairs[1][2] = {{1, 3}}, kernel = 1, stride = 1, dilation = 1;
    outputs[0] = poly_tensor_shrink(ctx, inputs[0], pairs, 1);
    outputs[1] = poly_tensor_pool(ctx, inputs[0], &kernel, 1, &stride, &dilation);
    if (outputs[0] && outputs[1]) return 1;
    poly_tensor_release(outputs[0]);
    poly_tensor_release(outputs[1]);
    outputs[0] = outputs[1] = NULL;
    return 0;
  }
#define OWN(expr)                                                                                  \
  do {                                                                                             \
    owned[count] = (expr);                                                                         \
    if (!owned[count++]) goto done;                                                                \
  } while (0)
  if (kind >= 5) {
    /* Import the shared functions, not copies of their mathematical formulas. */
    switch (kind) {
    case 5:
      OWN(poly_tensor_erfc(ctx, inputs[0]));
      break;
    case 6:
      OWN(poly_tensor_erfinv(ctx, inputs[0]));
      break;
    case 7:
      OWN(poly_tensor_ndtri(ctx, inputs[0]));
      break;
    case 8:
      OWN(poly_tensor_digamma(ctx, inputs[0]));
      break;
    case 9:
      OWN(poly_tensor_lgamma(ctx, inputs[0]));
      break;
    case 10:
      OWN(poly_tensor_log_ndtr(ctx, inputs[0]));
      break;
    case 11:
      OWN(poly_tensor_ndtri_exp(ctx, inputs[0]));
      break;
    default:
      goto done;
    }
    OWN(poly_tensor_sum(ctx, owned[0], &axis, 1, false));
    OWN(poly_tensor_gradient(ctx, owned[1], inputs[0]));
    outputs[0] = poly_tensor_retain(owned[0]);
    outputs[1] = poly_tensor_retain(owned[2]);
    ok = 1;
    goto done;
  }
  OWN(poly_tensor_alu2(ctx, POLY_OP_MUL, inputs[0], inputs[0]));
  OWN(poly_tensor_sum(ctx, owned[0], &axis, 1, false));
  OWN(poly_tensor_const_like_float(ctx, owned[1], -.5));
  OWN(poly_tensor_alu2(ctx, POLY_OP_MUL, owned[1], owned[2]));
  PolyTensor *density = owned[3];
  if (kind == 1) {
    OWN(poly_tensor_dot(ctx, inputs[1], inputs[0]));
    OWN(poly_tensor_softplus(ctx, owned[4], 1));
    OWN(poly_tensor_alu2(ctx, POLY_OP_MUL, inputs[2], owned[4]));
    OWN(poly_tensor_alu2(ctx, POLY_OP_SUB, owned[6], owned[5]));
    OWN(poly_tensor_sum(ctx, owned[7], &axis, 1, false));
    OWN(poly_tensor_alu2(ctx, POLY_OP_ADD, density, owned[8]));
    density = owned[9];
  } else if (kind == 2) {
    /* Failed construction must not publish a pending mutation on a borrowed
     * input. Capture owns rollback; ending it never executes the assignment. */
    PolyTensorCapture *capture = poly_tensor_capture_begin(ctx);
    if (capture) {
      poly_tensor_assign(ctx, inputs[0], owned[0]);
      poly_tensor_capture_end(capture);
    }
    goto done;
  } else if (kind != 0)
    goto done;
  OWN(poly_tensor_gradient(ctx, density, inputs[0]));
  outputs[0] = poly_tensor_retain(density);
  outputs[1] = poly_tensor_retain(owned[count - 1]);
  ok = 1;
done:
  while (count)
    poly_tensor_release(owned[--count]);
  return ok;
#undef OWN
}
