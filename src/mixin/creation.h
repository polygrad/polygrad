/* mixin/creation.h -- Stateless creation compositions on UOp graphs. */
#ifndef POLY_MIXIN_CREATION_H
#define POLY_MIXIN_CREATION_H

#include "../core.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Deterministic RNG helpers (stateless seed -> tensor). */
PolyUOp *poly_uop_rand(PolyCtx *ctx, const int64_t *shape, int ndim, uint64_t seed);

PolyUOp *poly_uop_randn(PolyCtx *ctx, const int64_t *shape, int ndim, uint64_t seed);

PolyUOp *poly_uop_rand_dtype(
    PolyCtx *ctx,
    const int64_t *shape,
    int ndim,
    uint64_t seed,
    PolyDType supplied_dtype
);

PolyUOp *poly_uop_randn_dtype(
    PolyCtx *ctx,
    const int64_t *shape,
    int ndim,
    uint64_t seed,
    PolyDType supplied_dtype
);

/* Creation helpers (constant-backed tensors). */
PolyUOp *poly_uop_full_invalid_dtype(
    PolyCtx *ctx,
    const int64_t *shape,
    int ndim,
    PolyDType supplied_dtype
);

PolyUOp *poly_uop_arange(PolyCtx *ctx, double start, double stop, double step);

PolyUOp *poly_uop_eye(PolyCtx *ctx, int64_t n);

PolyUOp *poly_uop_linspace(PolyCtx *ctx, double start, double stop, int64_t steps);

PolyUOp *poly_uop_full(PolyCtx *ctx, const int64_t *shape, int ndim, double fill_value);

PolyUOp *poly_uop_const_int_dtype(PolyCtx *ctx, int64_t value, PolyDType supplied_dtype);

PolyUOp *poly_uop_const_uint_dtype(PolyCtx *ctx, uint64_t value, PolyDType supplied_dtype);

PolyUOp *poly_uop_full_uint_dtype(
    PolyCtx *ctx,
    const int64_t *shape,
    int ndim,
    uint64_t value,
    PolyDType supplied_dtype
);

PolyUOp *poly_uop_const_float_dtype(PolyCtx *ctx, double value, PolyDType supplied_dtype);

PolyUOp *poly_uop_full_int_dtype(
    PolyCtx *ctx,
    const int64_t *shape,
    int ndim,
    int64_t fill_value,
    PolyDType supplied_dtype
);

PolyUOp *poly_uop_full_float_dtype(
    PolyCtx *ctx,
    const int64_t *shape,
    int ndim,
    double fill_value,
    PolyDType supplied_dtype
);

PolyUOp *poly_uop_arange_int_dtype(
    PolyCtx *ctx,
    int64_t start,
    int64_t stop,
    int64_t step,
    PolyDType supplied_dtype
);

PolyUOp *poly_uop_arange_float_dtype(
    PolyCtx *ctx,
    double start,
    double stop,
    double step,
    PolyDType supplied_dtype
);

PolyUOp *poly_uop_linspace_dtype(
    PolyCtx *ctx,
    double start,
    double stop,
    int64_t steps,
    PolyDType supplied_dtype
);

PolyUOp *poly_uop_eye_dtype(PolyCtx *ctx, int64_t n, int64_t m, PolyDType supplied_dtype);

#ifdef __cplusplus
}
#endif

#endif
