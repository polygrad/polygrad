#include "polygrad.h"
#define OP(fn, x) fn(ctx, x)
static PolyTensor *helper(PolyCtx *ctx, PolyTensor *x) {
  return OP(poly_tensor_exp, x);
}
/* Uncalled definitions must not enlarge the required extension imports. */
static PolyTensor *unused(PolyCtx *ctx, PolyTensor *x) {
  return poly_tensor_log(ctx, x);
}
int macro_build(PolyCtx *ctx, PolyTensor **inputs, PolyTensor **outputs) {
  outputs[0] = helper(ctx, inputs[0]);
  return outputs[0] != 0;
}
