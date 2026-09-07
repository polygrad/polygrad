/*
 * tensor.h -- Composed tensor ops (elementwise, reduction, creation, etc.)
 *
 * These are tensor-level graph helpers built from the core UOp primitives.
 * Keep them out of frontend.h so the language-binding ABI stays small and
 * limited to FFI-safe wrappers.
 */

#ifndef POLY_TENSOR_H
#define POLY_TENSOR_H

#include "polygrad.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Shape helpers (shared) */

int64_t poly_shape_numel_checked(const int64_t *shape, int ndim);
bool poly_shape_equal(const int64_t *a, int a_ndim, const int64_t *b, int b_ndim);

/* Polygrad logical-lifetime boundary for composed Tensor operations. Physical
 * operands remain mandatory; this controls only the independent portable
 * result and propagates NEVER/UNSUPPORTED operand state. */
int poly_tensor_result_builds_logical(PolyCtx *ctx, PolyTensor *const *inputs, int n_inputs);
PolyTensor *poly_tensor_create_result(
    PolyCtx *ctx,
    PolyTensor *const *inputs,
    int n_inputs,
    PolyUOp *logical,
    PolyUOp *physical,
    PolyTensorRole role,
    PolyDevice device
);

/* Publish an exact executable root while preserving the Tensor's retained
 * logical root. Frontend graph construction replaces both roots explicitly
 * through poly_tensor_replace_roots. */
int poly_tensor_set_physical(
    PolyCtx *ctx,
    PolyTensor *tensor,
    PolyUOp *uop_physical,
    PolyTensorRole role,
    PolyDevice device
);

/* Apply realized replacements to live executable PolyTensor roots, preserving
 * logical roots. POLY_DEVICE_AUTO applies the map globally. */
int poly_tensor_apply_realize_map(
    PolyCtx *ctx,
    PolyUOp **from,
    PolyUOp **to,
    int n,
    PolyDevice device
);

/* Approved logical/physical boundary: after successful own materialization,
 * replace an UNTIL_REALIZE producer with an exact device-free current
 * resource, or mark unsupported forms unavailable without altering physical. */
int poly_tensor_retire_logical_resources(PolyCtx *ctx, PolyTensor **tensors, int n_tensors);

/* Full-buffer STORE effect for optimizer/direct core SINKs.
 * Tensor.assign itself still uses tinygrad's current-value shape:
 * AFTER(target, STORE(target, value)). Direct effect SINKs already sequence
 * stores explicitly, so they should contain STORE(target, value) entries.
 * Movement views are normalized to their base buffer for whole-storage updates. */
PolyUOp *poly_store_buffer_update(PolyCtx *ctx, PolyUOp *target, PolyUOp *value);

/* Movement-op helpers (port of tinygrad mixin/movement.py) */

/* Tensor.repeat -- movement.py:465. n_repeats >= input ndim. */
PolyUOp *poly_repeat(PolyCtx *ctx, PolyUOp *x, const int64_t *repeats, int n_repeats);

/* Tensor.shrink_to -- movement.py:168. ends[i] == -1 means no-op (keep dim). */
PolyUOp *poly_shrink_to(PolyCtx *ctx, PolyUOp *x, const int64_t *ends, int n_ends);

/* Tensor._pool -- movement.py:487. General N-d pool via repeat/shrink/reshape/permute.
 * stride/dilation NULL means default of 1. Output adds a kernel axis per pooled dim. */
PolyUOp *poly_pool(
    PolyCtx *ctx,
    PolyUOp *x,
    const int64_t *k_,
    int nk,
    const int64_t *stride_,
    const int64_t *dilation_
);

PolyUOp *poly_max_pool2d(
    PolyCtx *ctx,
    PolyUOp *x,
    const int64_t *kernel,
    int n_kernel,
    const int64_t *stride,
    const int64_t *dilation,
    const int64_t *padding,
    int n_padding
);

PolyUOp *poly_conv2d(
    PolyCtx *ctx,
    PolyUOp *x,
    PolyUOp *weight,
    PolyUOp *bias,
    int groups,
    const int64_t *stride,
    const int64_t *dilation,
    const int64_t *padding,
    int n_padding
);

PolyUOp *poly_batchnorm(
    PolyCtx *ctx,
    PolyUOp *x,
    PolyUOp *weight,
    PolyUOp *bias,
    PolyUOp *mean,
    PolyUOp *invstd,
    const int64_t *axes,
    int n_axes
);

PolyUOp *poly_one_hot(PolyCtx *ctx, PolyUOp *x, int64_t num_classes);
PolyUOp *poly_index_select(PolyCtx *ctx, PolyUOp *x, int dim, PolyUOp *index);

/* Tensor.cat -- tensor.py:1364. Concatenate tensors along `dim`.
 * All tensors must have identical shape except along `dim`. */
PolyUOp *poly_cat(PolyCtx *ctx, PolyUOp **tensors, int n_tensors, int dim);

/* Tensor._pad_constant -- tensor.py:1067. Constant pad with `value`.
 * Supports negative pads (which shrink that side). For value==0 this is
 * equivalent to poly_pad on non-negative pairs. */
PolyUOp *poly_pad_value(PolyCtx *ctx, PolyUOp *x, int64_t (*pads)[2], int ndim, double value);

/* Tensor._pad_circular -- tensor.py:1075. Circular (wrap-around) padding.
 * Negative pads not supported. Each pad must be <= corresponding dim size. */
PolyUOp *poly_pad_circular(PolyCtx *ctx, PolyUOp *x, int64_t (*pads)[2], int ndim);

/* Tensor._pad_reflect_replicate (mode="reflect") -- tensor.py:1081.
 * Reflect padding without repeating the edge. Each pad must be < dim size. */
PolyUOp *poly_pad_reflect(PolyCtx *ctx, PolyUOp *x, int64_t (*pads)[2], int ndim);

/* Tensor._pad_reflect_replicate (mode="replicate") -- tensor.py:1081.
 * Replicate (edge-extend) padding. Repeats the boundary element. */
PolyUOp *poly_pad_replicate(PolyCtx *ctx, PolyUOp *x, int64_t (*pads)[2], int ndim);

/* Tensor._cumalu -- tensor.py:2048. Cumulative reduction along `axis`.
 * Supports POLY_OP_ADD, POLY_OP_MAX, POLY_OP_MUL (uses poly_pad_value with
 * the operator's identity element). include_initial=true uses negative pad
 * on the right (tinygrad parity). */
PolyUOp *poly_cumalu(PolyCtx *ctx, PolyUOp *x, int axis, PolyOps op);
PolyUOp *poly_split_cumalu(PolyCtx *ctx, PolyUOp *x, int axis, PolyOps op);

/* Broadcasting (matches tinygrad's _broadcasted) */

/* Broadcast a UOp to a target shape via reshape + expand.
 * Equivalent to tinygrad's _broadcast_to: left-pad dims with 1, then expand. */
PolyUOp *poly_broadcast_to(PolyCtx *ctx, PolyUOp *x, const int64_t *shape, int ndim);

/* Broadcast two UOps to a common shape (tinygrad's _broadcasted).
 * Returns the broadcast shape via out_shape/out_ndim. Returns false on
 * incompatible shapes. */
bool poly_broadcast_pair(PolyCtx *ctx, PolyUOp **a, PolyUOp **b, int64_t *out_shape, int *out_ndim);

/* Find a live PolyTensor representative for this storage identity, preferring
 * trainable/state metadata over anonymous aliases. This is for instance/export
 * validation, not placement decisions. */
PolyTensor *poly_tensor_find_storage_identity(PolyCtx *ctx, const PolyUOp *storage);

/* Broadcasting binary ops (like tinygrad Tensor.add/mul/sub) */

PolyUOp *poly_add(PolyCtx *ctx, PolyUOp *a, PolyUOp *b);
PolyUOp *poly_sub(PolyCtx *ctx, PolyUOp *a, PolyUOp *b);
PolyUOp *poly_mul(PolyCtx *ctx, PolyUOp *a, PolyUOp *b);
PolyUOp *poly_div(PolyCtx *ctx, PolyUOp *a, PolyUOp *b);
bool poly_broadcasted_pair(PolyCtx *ctx, PolyUOp **a, PolyUOp **b);
PolyUOp *poly_binop(PolyCtx *ctx, PolyOps op, PolyUOp *a, PolyUOp *b);

/* Contiguous (realize barrier) */

PolyUOp *poly_contiguous(PolyCtx *ctx, PolyUOp *x);

/* Composed elementwise ops (shape-free, UOp-level) */

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
PolyUOp *poly_log10(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_atanh(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_asinh(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_acosh(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_asin(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_acos(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_atan(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_logsigmoid(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_sinh(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_cosh(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_softsign(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_isfinite(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_celu(PolyCtx *ctx, PolyUOp *x, PolyUOp *alpha);
PolyUOp *poly_selu(PolyCtx *ctx, PolyUOp *x, PolyUOp *alpha, PolyUOp *gamma);
PolyUOp *poly_copysign(PolyCtx *ctx, PolyUOp *x, PolyUOp *other);
PolyUOp *poly_lerp(PolyCtx *ctx, PolyUOp *x, PolyUOp *end, PolyUOp *weight, bool scalar_weight);
PolyUOp *poly_isclose(
    PolyCtx *ctx,
    PolyUOp *x,
    PolyUOp *other,
    PolyUOp *rtol,
    PolyUOp *atol,
    bool equal_nan
);
PolyUOp *poly_binary_crossentropy(PolyCtx *ctx, PolyUOp *x, PolyUOp *target, int reduction);
PolyUOp *poly_binary_crossentropy_logits(
    PolyCtx *ctx,
    PolyUOp *x,
    PolyUOp *target,
    PolyUOp *weight,
    int reduction
);
PolyUOp *poly_nll_loss(
    PolyCtx *ctx,
    PolyUOp *x,
    PolyUOp *target,
    PolyUOp *weight,
    PolyUOp *ignore_index,
    int reduction
);
PolyUOp *poly_mish(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_hardtanh(PolyCtx *ctx, PolyUOp *x, double min_val, double max_val);
PolyUOp *poly_hardswish(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_hardsigmoid(PolyCtx *ctx, PolyUOp *x);

/* Paired Tensor boundaries for composed activations. These apply the exact
 * raw tinygrad-shaped program independently to retained logical and current
 * physical roots. */
PolyTensor *poly_tensor_relu(PolyCtx *ctx, PolyTensor *src);
PolyTensor *poly_tensor_sigmoid(PolyCtx *ctx, PolyTensor *src);
PolyTensor *poly_tensor_tanh(PolyCtx *ctx, PolyTensor *src);
PolyTensor *poly_tensor_silu(PolyCtx *ctx, PolyTensor *src);

/* Comparisons. All return BOOL, mirroring tinygrad mixin/elementwise.py:
 *   eq/ne via CMPNE (and double-CMPNE for eq), gt/lt via CMPLT,
 *   ge/le via logical_not of the strict form. */
PolyUOp *poly_eq(PolyCtx *ctx, PolyUOp *a, PolyUOp *b);
PolyUOp *poly_ne(PolyCtx *ctx, PolyUOp *a, PolyUOp *b);
PolyUOp *poly_gt(PolyCtx *ctx, PolyUOp *a, PolyUOp *b);
PolyUOp *poly_ge(PolyCtx *ctx, PolyUOp *a, PolyUOp *b);
PolyUOp *poly_le(PolyCtx *ctx, PolyUOp *a, PolyUOp *b);
/* Polygrad's canonical bool NOT, matching tinygrad's logical_not()
 * (mixin/elementwise.py:25-33) after CAST elision: CMPNE(x, CONST(true)). */
PolyUOp *poly_logical_not(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_cast(PolyCtx *ctx, PolyUOp *x, PolyDType target);
PolyUOp *poly_cast_by_id(PolyCtx *ctx, PolyUOp *x, int dtype_id);
PolyUOp *poly_bitcast_by_id(PolyCtx *ctx, PolyUOp *x, int dtype_id);
PolyUOp *poly_where_op(PolyCtx *ctx, PolyUOp *cond, PolyUOp *x, PolyUOp *y);
PolyUOp *poly_maximum(PolyCtx *ctx, PolyUOp *a, PolyUOp *b);
PolyUOp *poly_minimum(PolyCtx *ctx, PolyUOp *a, PolyUOp *b);
PolyUOp *poly_clamp(PolyCtx *ctx, PolyUOp *x, double lo, double hi);
PolyUOp *poly_detach(PolyCtx *ctx, PolyUOp *x);

/* Deterministic RNG helpers (stateless seed -> tensor). */
PolyUOp *poly_rand(PolyCtx *ctx, const int64_t *shape, int ndim, uint64_t seed);
PolyUOp *poly_randn(PolyCtx *ctx, const int64_t *shape, int ndim, uint64_t seed);
PolyUOp *poly_rand_by_id(PolyCtx *ctx, const int64_t *shape, int ndim, uint64_t seed, int dtype_id);
PolyUOp *poly_randn_by_id(
    PolyCtx *ctx,
    const int64_t *shape,
    int ndim,
    uint64_t seed,
    int dtype_id
);

/* Creation helpers (constant-backed tensors). */
PolyUOp *poly_arange(PolyCtx *ctx, double start, double stop, double step);
PolyUOp *poly_eye(PolyCtx *ctx, int64_t n);
PolyUOp *poly_linspace(PolyCtx *ctx, double start, double stop, int64_t steps);
PolyUOp *poly_full(PolyCtx *ctx, const int64_t *shape, int ndim, double fill_value);
PolyUOp *poly_const_int_by_id(PolyCtx *ctx, int64_t value, int dtype_id);
PolyUOp *poly_const_uint_by_id(PolyCtx *ctx, uint64_t value, int dtype_id);
PolyUOp *poly_full_uint_by_id(
    PolyCtx *ctx,
    const int64_t *shape,
    int ndim,
    uint64_t value,
    int dtype_id
);
PolyUOp *poly_const_float_by_id(PolyCtx *ctx, double value, int dtype_id);
PolyUOp *poly_full_int_by_id(
    PolyCtx *ctx,
    const int64_t *shape,
    int ndim,
    int64_t fill_value,
    int dtype_id
);
PolyUOp *poly_full_float_by_id(
    PolyCtx *ctx,
    const int64_t *shape,
    int ndim,
    double fill_value,
    int dtype_id
);
PolyUOp *poly_arange_int_by_id(
    PolyCtx *ctx,
    int64_t start,
    int64_t stop,
    int64_t step,
    int dtype_id
);
PolyUOp *poly_arange_float_by_id(
    PolyCtx *ctx,
    double start,
    double stop,
    double step,
    int dtype_id
);
PolyUOp *poly_linspace_by_id(PolyCtx *ctx, double start, double stop, int64_t steps, int dtype_id);
PolyUOp *poly_eye_by_id(PolyCtx *ctx, int64_t n, int64_t m, int dtype_id);
PolyUOp *poly_tril(PolyCtx *ctx, PolyUOp *x, int diagonal);
PolyUOp *poly_triu(PolyCtx *ctx, PolyUOp *x, int diagonal);
PolyUOp *poly_cholesky(PolyCtx *ctx, PolyUOp *x, int upper);
PolyUOp *poly_cholesky_solve(PolyCtx *ctx, PolyUOp *chol, PolyUOp *b, int upper);
PolyUOp *poly_triangular_solve(
    PolyCtx *ctx,
    PolyUOp *a,
    PolyUOp *b,
    int upper,
    int transpose_a,
    int unit_diagonal
);

/* Shape-aware composed ops (shape read from UOp) */

PolyUOp *poly_sum_reduce(PolyCtx *ctx, PolyUOp *x, int axis, int keepdim);
PolyUOp *poly_max_reduce(PolyCtx *ctx, PolyUOp *x, int axis, int keepdim);
PolyUOp *poly_mean_reduce(PolyCtx *ctx, PolyUOp *x, int axis, int keepdim);
PolyUOp *poly_var_reduce(PolyCtx *ctx, PolyUOp *x, int axis, int keepdim, int correction);
PolyUOp *poly_logsumexp(PolyCtx *ctx, PolyUOp *x, int axis, int keepdim);

PolyUOp *poly_dot(PolyCtx *ctx, PolyUOp *x, PolyUOp *w);
#define POLY_QR_COMPLETE 0
#define POLY_QR_REDUCED 1
#define POLY_QR_R_ONLY 2
int poly_qr(PolyCtx *ctx, PolyUOp *x, PolyUOp **out_q, PolyUOp **out_r);
int poly_qr_ex(PolyCtx *ctx, PolyUOp *x, int mode, PolyUOp **out_q, PolyUOp **out_r);
PolyUOp *poly_solve(PolyCtx *ctx, PolyUOp *a, PolyUOp *b);
PolyUOp *poly_lstsq(PolyCtx *ctx, PolyUOp *a, PolyUOp *b);

PolyUOp *poly_softmax(PolyCtx *ctx, PolyUOp *x, int axis);
PolyUOp *poly_log_softmax(PolyCtx *ctx, PolyUOp *x, int axis);
PolyUOp *poly_cross_entropy(PolyCtx *ctx, PolyUOp *logits, PolyUOp *target, int axis);

/* Tinygrad-style sort/topk composed helpers.
 * Return 0 on success and populate both outputs. */
int poly_sort(
    PolyCtx *ctx,
    PolyUOp *x,
    int dim,
    int descending,
    PolyUOp **out_values,
    PolyUOp **out_indices
);
PolyUOp *poly_argsort(PolyCtx *ctx, PolyUOp *x, int dim, int descending);
int poly_topk(
    PolyCtx *ctx,
    PolyUOp *x,
    int64_t k,
    int dim,
    int largest,
    int sorted,
    PolyUOp **out_values,
    PolyUOp **out_indices
);

/* Einsum */

PolyUOp *poly_einsum(PolyCtx *ctx, const char *formula, PolyUOp **tensors, int n_tensors);

/* Rearrange (einops) */

PolyUOp *poly_rearrange(
    PolyCtx *ctx,
    const char *formula,
    PolyUOp *x,
    const char *axis_names,
    const int64_t *axis_values,
    int n_axis_sizes
);

/* Gather (embedding lookup) */

PolyUOp *poly_gather(PolyCtx *ctx, PolyUOp *table, PolyUOp *indices);
PolyUOp *poly_gather_dim(PolyCtx *ctx, PolyUOp *x, int dim, PolyUOp *index);
PolyUOp *poly_scatter(
    PolyCtx *ctx,
    PolyUOp *self,
    int dim,
    PolyUOp *index,
    PolyUOp *src,
    const char *reduce
);
PolyUOp *poly_scatter_reduce(
    PolyCtx *ctx,
    PolyUOp *self,
    int dim,
    PolyUOp *index,
    PolyUOp *src,
    const char *reduce,
    int include_self
);

/* Additional composed ops */

PolyUOp *poly_rope(PolyCtx *ctx, PolyUOp *x, PolyUOp *freqs_cos, PolyUOp *freqs_sin);
PolyTensor *poly_tensor_rope(
    PolyCtx *ctx,
    PolyTensor *x,
    PolyTensor *freqs_cos,
    PolyTensor *freqs_sin
);
PolyUOp *poly_repeat_interleave(PolyCtx *ctx, PolyUOp *x, int repeats, int dim);
PolyUOp *poly_argmax(PolyCtx *ctx, PolyUOp *x, int axis, int keepdim);
PolyUOp *poly_mse_loss(PolyCtx *ctx, PolyUOp *pred, PolyUOp *target);
PolyUOp *poly_mae_loss(PolyCtx *ctx, PolyUOp *pred, PolyUOp *target);

#ifdef __cplusplus
}
#endif

#endif /* POLY_TENSOR_H */
