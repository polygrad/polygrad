/* Private scalar construction shared by ElementwiseMixin and Tensor compositions. */
#ifndef POLY_ELEMENTWISE_INTERNAL_H
#define POLY_ELEMENTWISE_INTERNAL_H

#include "polygrad.h"

#ifdef __cplusplus
extern "C" {
#endif

/* C scalar construction and ElementwiseMixin composition shared by Tensor
 * reductions and raw UOp operators; no Tensor handles or residency state. */
PolyUOp *poly_elementwise_float_const(PolyCtx *ctx, PolyUOp *ref, double v);
PolyUOp *poly_elementwise_neg(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_elementwise_inverse(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_elementwise_scalar_binop(
    PolyCtx *ctx,
    PolyOps op,
    PolyUOp *self,
    PolyDType from_py_dtype,
    double value,
    bool reverse
);
PolyUOp *poly_const_exact_int(PolyCtx *ctx, PolyDType dt, int64_t value);
PolyUOp *poly_const_exact_float(PolyCtx *ctx, PolyDType dt, double value);
bool poly_dtype_bound_const(PolyCtx *ctx, PolyDType dt, bool use_min, PolyUOp **out);

#ifdef __cplusplus
}
#endif
#endif
