/*
 * tensor.h -- Tensor compositions and logical/physical handle boundaries
 *
 * Shared UOp/ElementwiseMixin operators are declared by polygrad.h. This
 * header adds Tensor-level creation, movement, reduction and handle APIs;
 * language argument adaptation belongs to frontend.h.
 */

#ifndef POLY_TENSOR_H
#define POLY_TENSOR_H

#include "polygrad.h"

#ifdef __cplusplus
extern "C" {
#endif

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

/* Tensor.cat -- tensor.py:1364. Concatenate tensors along `dim`.
 * All tensors must have identical shape except along `dim`. */
PolyUOp *poly_cat(PolyCtx *ctx, PolyUOp **tensors, int n_tensors, int dim);

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

/* CreationMixin.full: optional storage materialization of a captured value. */
PolyTensor *poly_tensor_full_from_value(
    PolyCtx *ctx,
    PolyUOp *value_uop,
    const int64_t *dims,
    int ndim,
    PolyDevice device,
    PolyDType dtype,
    bool dtype_explicit,
    bool buffer
);

/* Tensor losses compose shared elementwise graphs and reductions. */
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

/* Paired Tensor boundaries for composed activations. These apply the exact
 * raw tinygrad-shaped program independently to retained logical and current
 * physical roots. */
PolyTensor *poly_tensor_relu(PolyCtx *ctx, PolyTensor *src);
PolyTensor *poly_tensor_sigmoid(PolyCtx *ctx, PolyTensor *src);
PolyTensor *poly_tensor_tanh(PolyCtx *ctx, PolyTensor *src);
PolyTensor *poly_tensor_silu(PolyCtx *ctx, PolyTensor *src);

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
PolyUOp *poly_repeat_interleave(PolyCtx *ctx, PolyUOp *x, int repeats, int dim);
PolyUOp *poly_argmax(PolyCtx *ctx, PolyUOp *x, int axis, int keepdim);
PolyUOp *poly_mse_loss(PolyCtx *ctx, PolyUOp *pred, PolyUOp *target);
PolyUOp *poly_mae_loss(PolyCtx *ctx, PolyUOp *pred, PolyUOp *target);

#ifdef __cplusplus
}
#endif

#endif /* POLY_TENSOR_H */
