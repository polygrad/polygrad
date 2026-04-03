/*
 * tensor.h -- Composed tensor ops (elementwise, reduction, creation, etc.)
 *
 * These are higher-level ops built from the core UOp primitives in frontend.h.
 * Separated from frontend.c to keep the FFI surface (frontend.c) focused on
 * graph construction and the realize pipeline.
 */

#ifndef POLY_TENSOR_H
#define POLY_TENSOR_H

#include "polygrad.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ── Const buffer registry (shared between tensor.c and frontend.c) ──── */

void poly_const_registry_add(PolyCtx *ctx, PolyUOp *buf, void *data);
void *poly_const_registry_lookup(PolyCtx *ctx, PolyUOp *buf);
bool poly_const_registry_has(PolyCtx *ctx, PolyUOp *buf);
void poly_const_registry_cleanup(PolyCtx *ctx);

/* ── Shape helpers (shared) ──────────────────────────────────────────── */

int64_t poly_shape_numel_checked(const int64_t *shape, int ndim);
bool poly_shape_equal(const int64_t *a, int a_ndim, const int64_t *b, int b_ndim);

/* ── Broadcasting (matches tinygrad's _broadcasted) ─────────────────── */

/* Broadcast a UOp to a target shape via reshape + expand.
 * Equivalent to tinygrad's _broadcast_to: left-pad dims with 1, then expand. */
PolyUOp *poly_broadcast_to(PolyCtx *ctx, PolyUOp *x, const int64_t *shape, int ndim);

/* Broadcast two UOps to a common shape (tinygrad's _broadcasted).
 * Returns the broadcast shape via out_shape/out_ndim. Returns false on
 * incompatible shapes. */
bool poly_broadcast_pair(PolyCtx *ctx, PolyUOp **a, PolyUOp **b,
                         int64_t *out_shape, int *out_ndim);

/* ── Broadcasting binary ops (like tinygrad Tensor.add/mul/sub) ─────── */

PolyUOp *poly_add(PolyCtx *ctx, PolyUOp *a, PolyUOp *b);
PolyUOp *poly_sub(PolyCtx *ctx, PolyUOp *a, PolyUOp *b);
PolyUOp *poly_mul(PolyCtx *ctx, PolyUOp *a, PolyUOp *b);
PolyUOp *poly_div(PolyCtx *ctx, PolyUOp *a, PolyUOp *b);

/* ── Contiguous (realize barrier) ───────────────────────────────────── */

PolyUOp *poly_contiguous(PolyCtx *ctx, PolyUOp *x);

/* ── Composed elementwise ops (shape-free, UOp-level) ────────────────── */

/* Math */
PolyUOp *poly_exp(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_log(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_log1p(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_expm1(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_sin(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_cos(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_tan(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_erf(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_erfc(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_erfinv(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_ndtri(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_digamma(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_lgamma(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_sigmoid(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_tanh_act(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_abs(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_sign(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_square(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_rsqrt(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_ceil(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_floor(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_round_f(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_isinf(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_isnan(PolyCtx *ctx, PolyUOp *x);

/* Activations */
PolyUOp *poly_relu(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_relu6(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_leaky_relu(PolyCtx *ctx, PolyUOp *x, double neg_slope);
PolyUOp *poly_gelu(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_quick_gelu(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_silu(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_elu(PolyCtx *ctx, PolyUOp *x, double alpha);
PolyUOp *poly_softplus(PolyCtx *ctx, PolyUOp *x, double beta);
PolyUOp *poly_mish(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_hardtanh(PolyCtx *ctx, PolyUOp *x, double min_val, double max_val);
PolyUOp *poly_hardswish(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_hardsigmoid(PolyCtx *ctx, PolyUOp *x);

/* Comparisons */
PolyUOp *poly_eq(PolyCtx *ctx, PolyUOp *a, PolyUOp *b);
PolyUOp *poly_ne(PolyCtx *ctx, PolyUOp *a, PolyUOp *b);
PolyUOp *poly_gt(PolyCtx *ctx, PolyUOp *a, PolyUOp *b);
PolyUOp *poly_ge(PolyCtx *ctx, PolyUOp *a, PolyUOp *b);
PolyUOp *poly_le(PolyCtx *ctx, PolyUOp *a, PolyUOp *b);
PolyUOp *poly_cast(PolyCtx *ctx, PolyUOp *x, PolyDType target);
PolyUOp *poly_cast_by_id(PolyCtx *ctx, PolyUOp *x, int dtype_id);
PolyUOp *poly_where_op(PolyCtx *ctx, PolyUOp *cond, PolyUOp *x, PolyUOp *y);
PolyUOp *poly_maximum(PolyCtx *ctx, PolyUOp *a, PolyUOp *b);
PolyUOp *poly_minimum(PolyCtx *ctx, PolyUOp *a, PolyUOp *b);
PolyUOp *poly_clamp(PolyCtx *ctx, PolyUOp *x, double lo, double hi);
PolyUOp *poly_detach(PolyCtx *ctx, PolyUOp *x);

/* Deterministic RNG helpers (stateless seed -> tensor). */
PolyUOp *poly_rand(PolyCtx *ctx, const int64_t *shape, int ndim, uint64_t seed);
PolyUOp *poly_randn(PolyCtx *ctx, const int64_t *shape, int ndim, uint64_t seed);

/* Creation helpers (constant-backed tensors). */
PolyUOp *poly_arange(PolyCtx *ctx, double start, double stop, double step);
PolyUOp *poly_eye(PolyCtx *ctx, int64_t n);
PolyUOp *poly_linspace(PolyCtx *ctx, double start, double stop, int64_t steps);
PolyUOp *poly_full(PolyCtx *ctx, const int64_t *shape, int ndim, double fill_value);
PolyUOp *poly_tril(PolyCtx *ctx, PolyUOp *x, int diagonal);
PolyUOp *poly_triu(PolyCtx *ctx, PolyUOp *x, int diagonal);
PolyUOp *poly_cholesky(PolyCtx *ctx, PolyUOp *x, int upper);
PolyUOp *poly_triangular_solve(PolyCtx *ctx, PolyUOp *a, PolyUOp *b,
                               int upper, int transpose_a, int unit_diagonal);

/* ── Shape-aware composed ops (shape read from UOp) ──────────────────── */

PolyUOp *poly_sum_reduce(PolyCtx *ctx, PolyUOp *x, int axis, int keepdim);
PolyUOp *poly_max_reduce(PolyCtx *ctx, PolyUOp *x, int axis, int keepdim);
PolyUOp *poly_mean_reduce(PolyCtx *ctx, PolyUOp *x, int axis, int keepdim);
PolyUOp *poly_var_reduce(PolyCtx *ctx, PolyUOp *x, int axis, int keepdim, int correction);
PolyUOp *poly_logsumexp(PolyCtx *ctx, PolyUOp *x, int axis, int keepdim);

PolyUOp *poly_dot(PolyCtx *ctx, PolyUOp *x, PolyUOp *w);

PolyUOp *poly_softmax(PolyCtx *ctx, PolyUOp *x, int axis);
PolyUOp *poly_log_softmax(PolyCtx *ctx, PolyUOp *x, int axis);
PolyUOp *poly_cross_entropy(PolyCtx *ctx, PolyUOp *logits, PolyUOp *target, int axis);

/* ── Einsum ────────────────────────────────────────────────────────── */

PolyUOp *poly_einsum(PolyCtx *ctx, const char *formula,
                     PolyUOp **tensors, int n_tensors);

/* ── Rearrange (einops) ───────────────────────────────────────────── */

PolyUOp *poly_rearrange(PolyCtx *ctx, const char *formula,
                        PolyUOp *x, const char *axis_names,
                        const int64_t *axis_values, int n_axis_sizes);

/* ── Gather (embedding lookup) ───────────────────────────────────── */

PolyUOp *poly_gather(PolyCtx *ctx, PolyUOp *table, PolyUOp *indices);

/* ── Additional composed ops ─────────────────────────────────────── */

PolyUOp *poly_sdpa(PolyCtx *ctx, PolyUOp *q, PolyUOp *k, PolyUOp *v,
                    PolyUOp *mask, int is_causal);
PolyUOp *poly_rope(PolyCtx *ctx, PolyUOp *x, PolyUOp *freqs_cos, PolyUOp *freqs_sin);
PolyUOp *poly_repeat_interleave(PolyCtx *ctx, PolyUOp *x, int repeats, int dim);
PolyUOp *poly_argmax(PolyCtx *ctx, PolyUOp *x, int axis);
PolyUOp *poly_mse_loss(PolyCtx *ctx, PolyUOp *pred, PolyUOp *target);
PolyUOp *poly_mae_loss(PolyCtx *ctx, PolyUOp *pred, PolyUOp *target);

/* Query constant buffer data. Returns NULL if buf is not a registered constant. */
const void *poly_const_buffer_data(PolyCtx *ctx, PolyUOp *buf);

#ifdef __cplusplus
}
#endif

#endif /* POLY_TENSOR_H */
