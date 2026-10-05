/* Independently compiled author. No Polygrad implementation is linked. */
#include "polygrad.h"

int extension_density(PolyCtx *ctx, PolyTensor **inputs, int kind, PolyTensor **outputs) {
  PolyTensor *owned[16] = {0};
  int count = 0, ok = 0;
  int64_t axis = 0;
  outputs[0] = outputs[1] = NULL;
  if (kind == 3) return poly_tensor_sort(ctx, inputs[0], 0, 0, outputs, outputs + 1) == 0;
#define OWN(expr)                                                                                  \
  do {                                                                                             \
    owned[count] = (expr);                                                                         \
    if (!owned[count++]) goto done;                                                                \
  } while (0)
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
