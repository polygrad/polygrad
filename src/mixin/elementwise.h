/* mixin/elementwise.h -- Elementwise compositions on UOp graphs. */
#ifndef POLY_MIXIN_ELEMENTWISE_H
#define POLY_MIXIN_ELEMENTWISE_H

#include "../core.h"

#ifdef __cplusplus
extern "C" {
#endif

/* C scalar construction and ElementwiseMixin composition shared by Tensor
 * reductions and raw UOp operators; no Tensor handles or residency state. */
PolyUOp *poly_uop_elementwise_float_const(PolyCtx *ctx, PolyUOp *ref, double v);
PolyUOp *poly_uop_elementwise_neg(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_uop_elementwise_inverse(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_uop_elementwise_scalar_binop(
    PolyCtx *ctx,
    PolyOps op,
    PolyUOp *self,
    PolyDType from_py_dtype,
    double value,
    bool reverse
);
PolyUOp *poly_uop_const_exact_int(PolyCtx *ctx, PolyDType dt, int64_t value);
PolyUOp *poly_uop_const_exact_float(PolyCtx *ctx, PolyDType dt, double value);
bool poly_dtype_bound_const(PolyCtx *ctx, PolyDType dt, bool use_min, PolyUOp **out);

/* Shared ElementwiseMixin graph operations (mixin/elementwise.c). */
PolyUOp *poly_uop_div(PolyCtx *ctx, PolyUOp *a, PolyUOp *b);

PolyUOp *poly_uop_contiguous(PolyCtx *ctx, PolyUOp *x);

PolyUOp *poly_uop_exp(PolyCtx *ctx, PolyUOp *x);

PolyUOp *poly_uop_log(PolyCtx *ctx, PolyUOp *x);

PolyUOp *poly_uop_log1p(PolyCtx *ctx, PolyUOp *x);

PolyUOp *poly_uop_expm1(PolyCtx *ctx, PolyUOp *x);

PolyUOp *poly_uop_sin(PolyCtx *ctx, PolyUOp *x);

PolyUOp *poly_uop_cos(PolyCtx *ctx, PolyUOp *x);

PolyUOp *poly_uop_tan(PolyCtx *ctx, PolyUOp *x);

PolyUOp *poly_uop_log10(PolyCtx *ctx, PolyUOp *x);

PolyUOp *poly_uop_atanh(PolyCtx *ctx, PolyUOp *x);

PolyUOp *poly_uop_asinh(PolyCtx *ctx, PolyUOp *x);

PolyUOp *poly_uop_acosh(PolyCtx *ctx, PolyUOp *x);

PolyUOp *poly_uop_asin(PolyCtx *ctx, PolyUOp *x);

PolyUOp *poly_uop_acos(PolyCtx *ctx, PolyUOp *x);

PolyUOp *poly_uop_atan(PolyCtx *ctx, PolyUOp *x);

PolyUOp *poly_uop_celu(PolyCtx *ctx, PolyUOp *x, PolyUOp *alpha);

PolyUOp *poly_uop_selu(PolyCtx *ctx, PolyUOp *x, PolyUOp *alpha, PolyUOp *gamma);

PolyUOp *poly_uop_sinh(PolyCtx *ctx, PolyUOp *x);

PolyUOp *poly_uop_cosh(PolyCtx *ctx, PolyUOp *x);

PolyUOp *poly_uop_softsign(PolyCtx *ctx, PolyUOp *x);

PolyUOp *poly_uop_erf(PolyCtx *ctx, PolyUOp *x);

PolyUOp *poly_uop_erfc(PolyCtx *ctx, PolyUOp *x);

PolyUOp *poly_uop_erfinv(PolyCtx *ctx, PolyUOp *x);

PolyUOp *poly_uop_ndtri(PolyCtx *ctx, PolyUOp *x);

PolyUOp *poly_uop_digamma(PolyCtx *ctx, PolyUOp *x);

PolyUOp *poly_uop_lgamma(PolyCtx *ctx, PolyUOp *x);

PolyUOp *poly_uop_sigmoid(PolyCtx *ctx, PolyUOp *x);

PolyUOp *poly_uop_tanh(PolyCtx *ctx, PolyUOp *x);

PolyUOp *poly_uop_abs(PolyCtx *ctx, PolyUOp *x);

PolyUOp *poly_uop_sign(PolyCtx *ctx, PolyUOp *x);

PolyUOp *poly_uop_square(PolyCtx *ctx, PolyUOp *x);

PolyUOp *poly_uop_rsqrt(PolyCtx *ctx, PolyUOp *x);

PolyUOp *poly_uop_ceil(PolyCtx *ctx, PolyUOp *x);

PolyUOp *poly_uop_floor(PolyCtx *ctx, PolyUOp *x);

PolyUOp *poly_uop_round(PolyCtx *ctx, PolyUOp *x);

PolyUOp *poly_uop_isinf(PolyCtx *ctx, PolyUOp *x);

PolyUOp *poly_uop_isnan(PolyCtx *ctx, PolyUOp *x);

PolyUOp *poly_uop_isfinite(PolyCtx *ctx, PolyUOp *x);

PolyUOp *poly_uop_isclose(
    PolyCtx *ctx,
    PolyUOp *x,
    PolyUOp *other,
    PolyUOp *rtol,
    PolyUOp *atol,
    bool equal_nan
);

PolyUOp *poly_uop_copysign(PolyCtx *ctx, PolyUOp *x, PolyUOp *other);

PolyUOp *poly_uop_lerp(PolyCtx *ctx, PolyUOp *x, PolyUOp *end, PolyUOp *weight, bool scalar_weight);

PolyUOp *poly_uop_relu(PolyCtx *ctx, PolyUOp *x);

PolyUOp *poly_uop_relu6(PolyCtx *ctx, PolyUOp *x);

PolyUOp *poly_uop_leaky_relu(PolyCtx *ctx, PolyUOp *x, double neg_slope);

PolyUOp *poly_uop_gelu(PolyCtx *ctx, PolyUOp *x);

PolyUOp *poly_uop_quick_gelu(PolyCtx *ctx, PolyUOp *x);

PolyUOp *poly_uop_silu(PolyCtx *ctx, PolyUOp *x);

PolyUOp *poly_uop_elu(PolyCtx *ctx, PolyUOp *x, double alpha);

PolyUOp *poly_uop_softplus(PolyCtx *ctx, PolyUOp *x, double beta);

PolyUOp *poly_uop_logsigmoid(PolyCtx *ctx, PolyUOp *x);

PolyUOp *poly_uop_mish(PolyCtx *ctx, PolyUOp *x);

PolyUOp *poly_uop_hardtanh(PolyCtx *ctx, PolyUOp *x, double min_val, double max_val);

PolyUOp *poly_uop_hardswish(PolyCtx *ctx, PolyUOp *x);

PolyUOp *poly_uop_hardsigmoid(PolyCtx *ctx, PolyUOp *x);

PolyUOp *poly_uop_ne(PolyCtx *ctx, PolyUOp *a, PolyUOp *b);

PolyUOp *poly_uop_gt(PolyCtx *ctx, PolyUOp *a, PolyUOp *b);

PolyUOp *poly_uop_ge(PolyCtx *ctx, PolyUOp *a, PolyUOp *b);

PolyUOp *poly_uop_le(PolyCtx *ctx, PolyUOp *a, PolyUOp *b);

PolyUOp *poly_uop_maximum(PolyCtx *ctx, PolyUOp *a, PolyUOp *b);

PolyUOp *poly_uop_minimum(PolyCtx *ctx, PolyUOp *a, PolyUOp *b);

PolyUOp *poly_uop_clamp(PolyCtx *ctx, PolyUOp *x, double lo, double hi);

PolyUOp *poly_uop_detach(PolyCtx *ctx, PolyUOp *x);

PolyUOp *poly_uop_elementwise_promote(PolyCtx *ctx, PolyUOp *root, PolyDType common);

bool poly_uop_broadcasted_pair(PolyCtx *ctx, PolyUOp **a, PolyUOp **b);

PolyUOp *poly_uop_binop(PolyCtx *ctx, PolyOps op, PolyUOp *a, PolyUOp *b);

PolyUOp *poly_uop_add(PolyCtx *ctx, PolyUOp *a, PolyUOp *b);

PolyUOp *poly_uop_sub(PolyCtx *ctx, PolyUOp *a, PolyUOp *b);

PolyUOp *poly_uop_mul(PolyCtx *ctx, PolyUOp *a, PolyUOp *b);

PolyUOp *poly_uop_eq(PolyCtx *ctx, PolyUOp *a, PolyUOp *b);

PolyUOp *poly_uop_logical_not(PolyCtx *ctx, PolyUOp *x);

PolyUOp *poly_uop_where(PolyCtx *ctx, PolyUOp *cond, PolyUOp *x, PolyUOp *y);

#ifdef __cplusplus
}
#endif

#endif
