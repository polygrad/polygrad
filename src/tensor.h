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

/* Apply realized replacements to live PolyTensors, preserving logical roots
 * and updating only executable physical roots. POLY_DEVICE_AUTO applies the
 * map globally; an exact device keeps placement aliases from retargeting a
 * distinct source-device tensor that shares the same portable logical root. */
int poly_tensor_apply_realize_map(
    PolyCtx *ctx,
    PolyUOp **from,
    PolyUOp **to,
    int n,
    PolyDevice device,
    PolyMap **placement_memo
);

/* Dynamic BUFFER helper used by C tests/probes and low-level callers. The
 * first runtime dimension is a DEFINE_VAR or BIND, while allocation reserves
 * the variable max bound times the fixed inner dimensions. */
PolyUOp *poly_buffer_var(
    PolyCtx *ctx,
    PolyDType dt,
    PolyUOp *batch_var,
    const int64_t *inner_dims,
    int n_inner_dims
);

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
PolyUOp *poly_cumalu(PolyCtx *ctx, PolyUOp *x, int axis, PolyOps op, bool include_initial);

/* Broadcasting (matches tinygrad's _broadcasted) */

/* Broadcast a UOp to a target shape via reshape + expand.
 * Equivalent to tinygrad's _broadcast_to: left-pad dims with 1, then expand. */
PolyUOp *poly_broadcast_to(PolyCtx *ctx, PolyUOp *x, const int64_t *shape, int ndim);

/* Broadcast two UOps to a common shape (tinygrad's _broadcasted).
 * Returns the broadcast shape via out_shape/out_ndim. Returns false on
 * incompatible shapes. */
bool poly_broadcast_pair(PolyCtx *ctx, PolyUOp **a, PolyUOp **b, int64_t *out_shape, int *out_ndim);

/* Current-placement index lookup. `role == (PolyTensorRole)-1` means any role.
 * The index is keyed by poly_tensor_uop(t), not by the preserved logical root. */
PolyTensor *poly_tensor_find_current(
    PolyCtx *ctx,
    PolyUOp *current,
    PolyDevice device,
    PolyTensorRole role
);

/* Find a live PolyTensor representative for this storage identity, preferring
 * trainable/state metadata over anonymous aliases. This is for instance/export
 * validation, not placement decisions. */
PolyTensor *poly_tensor_find_storage_identity(PolyCtx *ctx, const PolyUOp *storage);

typedef struct {
  PolyTensor *selected;
  PolyUOp *selected_current;
  PolyUOp *selected_logical;
  PolyUOp *selected_physical;
  PolyTensorRole selected_role;
  PolyDevice selected_device;
  PolyTensor *selected_source;

  PolyUOp *query_current;
  PolyDevice query_device;
  PolyTensor *place_fact;
  PolyTensor *value_fact;
  PolyTensor *matched_fact;
  PolyTensorRole matched_role;

  PolyUOp *physical_root;
} PolyPlacementAudit;

/* Test/probe-facing placement inspection. This is intentionally not part of
 * the frontend ABI: it mirrors the physicalizer's current fact lookup order so
 * placement tests can assert which VALUE/PLACE record is active before judging
 * the generated physical COPY/DEVICE graph. `query_current == NULL` audits the
 * selected tensor's current root; `query_device == AUTO` uses the selected
 * tensor's resolved device. */
int poly_tensor_placement_audit(
    PolyCtx *ctx,
    PolyTensor *selected,
    PolyUOp *query_current,
    PolyDevice query_device,
    PolyPlacementAudit *out
);

/* Broadcasting binary ops (like tinygrad Tensor.add/mul/sub) */

PolyUOp *poly_add(PolyCtx *ctx, PolyUOp *a, PolyUOp *b);
PolyUOp *poly_sub(PolyCtx *ctx, PolyUOp *a, PolyUOp *b);
PolyUOp *poly_mul(PolyCtx *ctx, PolyUOp *a, PolyUOp *b);
PolyUOp *poly_div(PolyCtx *ctx, PolyUOp *a, PolyUOp *b);

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
PolyUOp *poly_mish(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_hardtanh(PolyCtx *ctx, PolyUOp *x, double min_val, double max_val);
PolyUOp *poly_hardswish(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_hardsigmoid(PolyCtx *ctx, PolyUOp *x);

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
PolyUOp *poly_scatter(PolyCtx *ctx, PolyUOp *self, int dim, PolyUOp *index, PolyUOp *src, const char *reduce);
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
PolyUOp *poly_repeat_interleave(PolyCtx *ctx, PolyUOp *x, int repeats, int dim);
PolyUOp *poly_argmax(PolyCtx *ctx, PolyUOp *x, int axis);
PolyUOp *poly_mse_loss(PolyCtx *ctx, PolyUOp *pred, PolyUOp *target);
PolyUOp *poly_mae_loss(PolyCtx *ctx, PolyUOp *pred, PolyUOp *target);

#ifdef __cplusplus
}
#endif

#endif /* POLY_TENSOR_H */
