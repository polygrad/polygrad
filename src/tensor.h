/* tensor.h -- Tensor ownership and paired physical/logical operations. */
#ifndef POLY_TENSOR_H
#define POLY_TENSOR_H

#include "core.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Core frontend tensor handle.
 *
 * PolyUOp stays the pure logical value graph. PolyTensor is the C-side value
 * reference used by frontends. The tensor carries two roots: logical is the
 * exportable/re-placement expression, while physical is the mandatory current
 * tinygrad-shaped execution/readback root. poly_tensor_uop() returns physical;
 * portable export/provenance uses uop_logical explicitly.
 */

typedef enum {
  POLY_TENSOR_VALUE = 0,
  POLY_TENSOR_PLACE = 1,
  POLY_TENSOR_BARRIER = 2,
} PolyTensorRole;

typedef enum {
  POLY_TENSOR_PROVENANCE_UNKNOWN = 0,
  POLY_TENSOR_PROVENANCE_USER_INPUT = 1,
  POLY_TENSOR_PROVENANCE_PARAM_INIT = 2,
  POLY_TENSOR_PROVENANCE_STATE_LOADED = 3,
  POLY_TENSOR_PROVENANCE_CONST_INIT = 4,
  POLY_TENSOR_PROVENANCE_COMPUTED = 5,
} PolyTensorProvenance;

struct PolyTensor {
  PolyUOp *uop_logical;
  PolyUOp *uop_physical;
  PolyLogicalPolicy logical_policy;
  PolyLogicalState logical_state;
  PolyTensorRole role;
  PolyDevice device;
  uint64_t order;
  PolyTensor *source;
  PolyTensorProvenance provenance;
  /* C mechanics for Tinygrad's weak live-Tensor registry. */
  PolyCtx *owner_ctx;
  uint32_t owner_refs;
  int owner_slot;
};

PolyTensor *poly_tensor_create_with_roots(
    PolyCtx *ctx,
    PolyUOp *uop_logical,
    PolyUOp *uop_physical,
    PolyTensorRole role,
    PolyDevice device
);

/* FFI adaptation for one-operand results whose portable availability follows
 * the source Tensor independently of the ambient context policy. */
PolyTensor *poly_tensor_create_result_like(
    PolyCtx *ctx,
    PolyTensor *input,
    PolyUOp *uop_logical,
    PolyUOp *uop_physical,
    PolyTensorRole role,
    PolyDevice device
);

/* Tensor constructors return one owned handle, including identity returns.
 * Internal accessors and same-object mutators are borrowed. */
PolyTensor *poly_tensor_retain(PolyTensor *tensor);

void poly_tensor_release(PolyTensor *tensor);

PolyTensor *poly_tensor_empty(
    PolyCtx *ctx,
    PolyDType scalar_dtype,
    const int64_t *dims,
    int ndim,
    PolyDevice device
);

PolyTensor *poly_tensor_empty_uop(
    PolyCtx *ctx,
    PolyDType scalar_dtype,
    PolyUOp **dims,
    int ndim,
    PolyDevice device
);

/* CreationMixin.empty preserves the complete device identity (e.g. DISK:path). */
PolyTensor *poly_tensor_empty_uop_name(
    PolyCtx *ctx,
    PolyDType scalar_dtype,
    PolyUOp **dims,
    int ndim,
    const char *device
);

PolyTensor *poly_tensor_from_host(
    PolyCtx *ctx,
    void *ptr,
    size_t nbytes,
    PolyDType scalar_dtype,
    const int64_t *dims,
    int ndim
);

/* Pinned Tensor.manual_seed/Tensor.rand stateful RNG surface. RNG state is
 * owned by ctx and separated by exact device identity. */
void poly_tensor_manual_seed(PolyCtx *ctx, int64_t seed);

int poly_tensor_replace_roots(
    PolyCtx *ctx,
    PolyTensor *tensor,
    PolyUOp *uop_logical,
    PolyUOp *uop_physical,
    PolyTensorRole role,
    PolyDevice device
);

PolyTensor *poly_tensor_to_device(PolyCtx *ctx, PolyTensor *tensor, PolyDevice device);

PolyTensor *poly_tensor_to_device_name(PolyCtx *ctx, PolyTensor *tensor, const char *device);

PolyTensor *poly_tensor_assign(PolyCtx *ctx, PolyTensor *target, PolyTensor *value);

PolyTensor *poly_tensor_clone_into(PolyCtx *ctx, PolyTensor *target, PolyTensor *source);

PolyTensor *poly_tensor_clone(PolyCtx *ctx, PolyTensor *source, PolyDevice device);

int poly_tensor_custom_kernel(
    PolyCtx *ctx,
    PolyUOp *body,
    PolyTensor **inputs,
    int n_inputs,
    uint32_t grad_fxn_key,
    PolyTensor **outputs
);

/* Build value-producing FUNCTION roots from result Tensors and ordered input
 * UOps captured before body execution (tinygrad/function.py:43-79). */
int poly_tensor_function(
    PolyCtx *ctx,
    PolyTensor **results,
    int n_results,
    PolyUOp **logical_inputs,
    PolyUOp **physical_inputs,
    int n_inputs,
    const char *name,
    bool allow_implicit,
    bool precompile,
    bool precompile_backward,
    PolyTensor **outputs
);

PolyTensor *poly_tensor_alu1(PolyCtx *ctx, PolyOps op, PolyTensor *src);

PolyTensor *poly_tensor_alu2(PolyCtx *ctx, PolyOps op, PolyTensor *a, PolyTensor *b);

PolyTensor *poly_tensor_alu3(PolyCtx *ctx, PolyOps op, PolyTensor *a, PolyTensor *b, PolyTensor *c);

/* rounding: 0 true division, 1 truncate, 2 floor; selected after promotion. */
PolyTensor *poly_tensor_div(PolyCtx *ctx, PolyTensor *dividend, PolyTensor *divisor, int rounding);

PolyTensor *poly_tensor_exp(PolyCtx *ctx, PolyTensor *src);

PolyTensor *poly_tensor_log(PolyCtx *ctx, PolyTensor *src);

PolyTensor *poly_tensor_cos(PolyCtx *ctx, PolyTensor *src);

PolyTensor *poly_tensor_tan(PolyCtx *ctx, PolyTensor *src);

PolyTensor *poly_tensor_log1p(PolyCtx *ctx, PolyTensor *src);

PolyTensor *poly_tensor_expm1(PolyCtx *ctx, PolyTensor *src);

/* Composed pointwise/loss methods; reduction: 0=none, 1=sum, 2=mean.
 * Optional weights/ignore_index may be NULL. Scalar arguments are Tensor
 * handles to preserve their original weak dtype through promotion. */
PolyTensor *poly_tensor_log10(PolyCtx *ctx, PolyTensor *x);

PolyTensor *poly_tensor_atanh(PolyCtx *ctx, PolyTensor *x);

PolyTensor *poly_tensor_asinh(PolyCtx *ctx, PolyTensor *x);

PolyTensor *poly_tensor_acosh(PolyCtx *ctx, PolyTensor *x);

PolyTensor *poly_tensor_asin(PolyCtx *ctx, PolyTensor *x);

PolyTensor *poly_tensor_acos(PolyCtx *ctx, PolyTensor *x);

PolyTensor *poly_tensor_atan(PolyCtx *ctx, PolyTensor *x);

PolyTensor *poly_tensor_logsigmoid(PolyCtx *ctx, PolyTensor *x);

PolyTensor *poly_tensor_sinh(PolyCtx *ctx, PolyTensor *x);

PolyTensor *poly_tensor_cosh(PolyCtx *ctx, PolyTensor *x);

PolyTensor *poly_tensor_erf(PolyCtx *ctx, PolyTensor *x);

PolyTensor *poly_tensor_softsign(PolyCtx *ctx, PolyTensor *x);

PolyTensor *poly_tensor_isfinite(PolyCtx *ctx, PolyTensor *x);

PolyTensor *poly_tensor_celu(PolyCtx *ctx, PolyTensor *x, PolyTensor *alpha);

PolyTensor *poly_tensor_selu(PolyCtx *ctx, PolyTensor *x, PolyTensor *alpha, PolyTensor *gamma);

PolyTensor *poly_tensor_copysign(PolyCtx *ctx, PolyTensor *x, PolyTensor *other);

PolyTensor *poly_tensor_lerp(
    PolyCtx *ctx,
    PolyTensor *x,
    PolyTensor *end,
    PolyTensor *weight,
    bool scalar_weight
);

PolyTensor *poly_tensor_isclose(
    PolyCtx *ctx,
    PolyTensor *x,
    PolyTensor *other,
    PolyTensor *rtol,
    PolyTensor *atol,
    bool equal_nan
);

PolyTensor *poly_tensor_binary_crossentropy(
    PolyCtx *ctx,
    PolyTensor *x,
    PolyTensor *target,
    int reduction
);

/* Cross-entropy follows Tensor.cross_entropy; class axis is explicit in C.
 * Reduction: 0=none, 1=sum, 2=mean. Weight/ignore_index belong to nll_loss. */
PolyTensor *poly_tensor_cross_entropy(
    PolyCtx *ctx,
    PolyTensor *x,
    PolyTensor *target,
    int axis,
    int reduction,
    double smoothing
);

PolyTensor *poly_tensor_mse_loss(PolyCtx *ctx, PolyTensor *x, PolyTensor *target);

PolyTensor *poly_tensor_binary_crossentropy_logits(
    PolyCtx *ctx,
    PolyTensor *x,
    PolyTensor *target,
    PolyTensor *weight,
    int reduction
);

PolyTensor *poly_tensor_nll_loss(
    PolyCtx *ctx,
    PolyTensor *x,
    PolyTensor *target,
    PolyTensor *weight,
    PolyTensor *ignore_index,
    int reduction
);

PolyTensor *poly_tensor_prod(
    PolyCtx *ctx,
    PolyTensor *src,
    int64_t *axes,
    int n_axes,
    bool keepdim
);

PolyTensor *poly_tensor_logsumexp(
    PolyCtx *ctx,
    PolyTensor *src,
    int64_t *axes,
    int n_axes,
    bool keepdim
);

PolyTensor *poly_tensor_normalize(PolyCtx *ctx, PolyTensor *src, double p, int axis, double eps);

PolyTensor *poly_tensor_logcumsumexp(PolyCtx *ctx, PolyTensor *src, int axis);

PolyTensor *poly_tensor_gelu_exact(PolyCtx *ctx, PolyTensor *src);

PolyTensor *poly_tensor_diag(PolyCtx *ctx, PolyTensor *src);

PolyTensor *poly_tensor_diagonal(PolyCtx *ctx, PolyTensor *src, int64_t offset, int dim1, int dim2);

PolyTensor *poly_tensor_unfold(PolyCtx *ctx, PolyTensor *src, int dim, int64_t size, int64_t step);

PolyTensor *poly_tensor_argmin(PolyCtx *ctx, PolyTensor *src, int axis, bool keepdim);

PolyTensor *poly_tensor_pad_mode(PolyCtx *ctx, PolyTensor *src, int64_t *pairs, int ndim, int mode);

PolyTensor *poly_tensor_gelu(PolyCtx *ctx, PolyTensor *src);

PolyTensor *poly_tensor_stack(PolyCtx *ctx, PolyTensor **inputs, int n_inputs, int dim);

PolyTensor *poly_tensor_bitwise_not(PolyCtx *ctx, PolyTensor *src);

/* reduction: 0 none, 1 sum, 2 mean, shared with the other loss boundaries. */
PolyTensor *poly_tensor_sparse_categorical_crossentropy(
    PolyCtx *ctx,
    PolyTensor *x,
    PolyTensor *target,
    int64_t ignore_index,
    double smoothing,
    int reduction
);

PolyTensor *poly_tensor_quick_gelu(PolyCtx *ctx, PolyTensor *src);

PolyTensor *poly_tensor_detach(PolyCtx *ctx, PolyTensor *src);

PolyTensor *poly_tensor_contiguous_backward(PolyCtx *ctx, PolyTensor *src);

PolyTensor *poly_tensor_sum(PolyCtx *ctx, PolyTensor *src, int64_t *axes, int n_axes, bool keepdim);

PolyTensor *poly_tensor_mean(
    PolyCtx *ctx,
    PolyTensor *src,
    int64_t *axes,
    int n_axes,
    bool keepdim
);

PolyTensor *poly_tensor_max(PolyCtx *ctx, PolyTensor *src, int64_t *axes, int n_axes, bool keepdim);

PolyTensor *poly_tensor_min(PolyCtx *ctx, PolyTensor *src, int64_t *axes, int n_axes, bool keepdim);

PolyTensor *poly_tensor_all(PolyCtx *ctx, PolyTensor *src, int64_t *axes, int n_axes, bool keepdim);

PolyTensor *poly_tensor_any(PolyCtx *ctx, PolyTensor *src, int64_t *axes, int n_axes, bool keepdim);

PolyTensor *poly_tensor_cumsum(PolyCtx *ctx, PolyTensor *src, int axis);

PolyTensor *poly_tensor_cumprod(PolyCtx *ctx, PolyTensor *src, int axis);

int poly_tensor_cummax(
    PolyCtx *ctx,
    PolyTensor *src,
    int axis,
    PolyTensor **values,
    PolyTensor **indices
);

int poly_tensor_cummin(
    PolyCtx *ctx,
    PolyTensor *src,
    int axis,
    PolyTensor **values,
    PolyTensor **indices
);

PolyTensor *poly_tensor_argmax(PolyCtx *ctx, PolyTensor *src, int axis, bool keepdim);

PolyTensor *poly_tensor_minimum(PolyCtx *ctx, PolyTensor *a, PolyTensor *b);

PolyTensor *poly_tensor_dot(PolyCtx *ctx, PolyTensor *src, PolyTensor *weight);

int poly_tensor_qr_ex(
    PolyCtx *ctx,
    PolyTensor *src,
    int mode,
    PolyTensor **out_q,
    PolyTensor **out_r
);

PolyTensor *poly_tensor_triangular_solve(
    PolyCtx *ctx,
    PolyTensor *a,
    PolyTensor *b,
    int upper,
    int transpose_a,
    int unit_diagonal
);

PolyTensor *poly_tensor_cholesky(PolyCtx *ctx, PolyTensor *src, int upper);

PolyTensor *poly_tensor_cholesky_solve(PolyCtx *ctx, PolyTensor *chol, PolyTensor *b, int upper);

PolyTensor *poly_tensor_solve(PolyCtx *ctx, PolyTensor *a, PolyTensor *b);

PolyTensor *poly_tensor_lstsq(PolyCtx *ctx, PolyTensor *a, PolyTensor *b);

int poly_tensor_sort(
    PolyCtx *ctx,
    PolyTensor *src,
    int dim,
    int descending,
    PolyTensor **out_values,
    PolyTensor **out_indices
);

int poly_tensor_topk(
    PolyCtx *ctx,
    PolyTensor *src,
    int64_t k,
    int dim,
    int largest,
    int sorted,
    PolyTensor **out_values,
    PolyTensor **out_indices
);

PolyTensor *poly_tensor_softmax(PolyCtx *ctx, PolyTensor *src, int axis);

PolyTensor *poly_tensor_log_softmax(PolyCtx *ctx, PolyTensor *src, int axis);

PolyTensor *poly_tensor_rope(
    PolyCtx *ctx,
    PolyTensor *x,
    PolyTensor *freqs_cos,
    PolyTensor *freqs_sin
);

PolyTensor *poly_tensor_const_like_int(PolyCtx *ctx, PolyTensor *ref, int64_t value);

PolyTensor *poly_tensor_const_like_float(PolyCtx *ctx, PolyTensor *ref, double value);

PolyTensor *poly_tensor_contiguous(PolyCtx *ctx, PolyTensor *src);

/* Tensor.cat: shared logical/physical construction and runtime-owner validation. */
PolyTensor *poly_tensor_cat(PolyCtx *ctx, PolyTensor **tensors, int n_tensors, int dim);

PolyTensor *poly_tensor_reshape(PolyCtx *ctx, PolyTensor *src, int64_t *dims, int ndim);

PolyTensor *poly_tensor_reshape_uop(PolyCtx *ctx, PolyTensor *src, PolyUOp **dims, int ndim);

PolyTensor *poly_tensor_expand(PolyCtx *ctx, PolyTensor *src, int64_t *dims, int ndim);

PolyTensor *poly_tensor_expand_uop(PolyCtx *ctx, PolyTensor *src, PolyUOp **dims, int ndim);

PolyTensor *poly_tensor_permute(PolyCtx *ctx, PolyTensor *src, int64_t *perm, int ndim);

PolyTensor *poly_tensor_shrink(PolyCtx *ctx, PolyTensor *src, int64_t (*pairs)[2], int ndim);

PolyTensor *poly_tensor_shrink_uop(
    PolyCtx *ctx,
    PolyTensor *src,
    PolyUOp **starts,
    PolyUOp **sizes,
    int ndim
);

PolyTensor *poly_tensor_flip(PolyCtx *ctx, PolyTensor *src, int64_t *axes, int n_axes);

PolyTensor *poly_tensor_pad_value_bool(
    PolyCtx *ctx,
    PolyTensor *src,
    int64_t (*pairs)[2],
    int ndim,
    bool value
);

PolyTensor *poly_tensor_pad_value_int(
    PolyCtx *ctx,
    PolyTensor *src,
    int64_t (*pairs)[2],
    int ndim,
    int64_t value
);

PolyTensor *poly_tensor_pad_value_float(
    PolyCtx *ctx,
    PolyTensor *src,
    int64_t (*pairs)[2],
    int ndim,
    double value
);

PolyTensor *poly_tensor_pool(
    PolyCtx *ctx,
    PolyTensor *src,
    const int64_t *kernel,
    int n_kernel,
    const int64_t *stride,
    const int64_t *dilation
);

/* Optional indices receives a separately owned Tensor handle. Failure leaves
 * it NULL; values and indices follow the input's logical-retention policy. */
PolyTensor *poly_tensor_max_pool2d(
    PolyCtx *ctx,
    PolyTensor *src,
    const int64_t *kernel,
    int n_kernel,
    const int64_t *stride,
    const int64_t *dilation,
    const int64_t *padding,
    int n_padding,
    bool ceil_mode,
    PolyTensor **indices
);

PolyTensor *poly_tensor_avg_pool2d(
    PolyCtx *ctx,
    PolyTensor *src,
    const int64_t *kernel,
    int n_kernel,
    const int64_t *stride,
    const int64_t *dilation,
    const int64_t *padding,
    int n_padding,
    bool ceil_mode,
    bool count_include_pad
);

PolyTensor *poly_tensor_interpolate(
    PolyCtx *ctx,
    PolyTensor *src,
    const int64_t *size,
    int n_size,
    const char *mode,
    bool align_corners
);

PolyTensor *poly_tensor_max_unpool2d(
    PolyCtx *ctx,
    PolyTensor *src,
    PolyTensor *indices,
    const int64_t *kernel,
    int n_kernel,
    const int64_t *stride,
    const int64_t *dilation,
    const int64_t *padding,
    int n_padding,
    const int64_t *output_size,
    int n_output
);

PolyTensor *poly_tensor_conv_transpose2d(
    PolyCtx *ctx,
    PolyTensor *src,
    PolyTensor *weight,
    PolyTensor *bias,
    int groups,
    const int64_t *stride,
    const int64_t *dilation,
    const int64_t *padding,
    int n_padding,
    const int64_t *output_padding,
    int n_output_padding
);

PolyTensor *poly_tensor_conv2d(
    PolyCtx *ctx,
    PolyTensor *src,
    PolyTensor *weight,
    PolyTensor *bias,
    int groups,
    const int64_t *stride,
    const int64_t *dilation,
    const int64_t *padding,
    int n_padding
);

PolyTensor *poly_tensor_batchnorm(
    PolyCtx *ctx,
    PolyTensor *src,
    PolyTensor *weight,
    PolyTensor *bias,
    PolyTensor *mean,
    PolyTensor *invstd,
    const int64_t *axes,
    int n_axes
);

PolyTensor *poly_tensor_one_hot(PolyCtx *ctx, PolyTensor *x, int64_t num_classes);

PolyTensor *poly_tensor_gather_dim(PolyCtx *ctx, PolyTensor *x, int dim, PolyTensor *index);

PolyTensor *poly_tensor_index_select(PolyCtx *ctx, PolyTensor *x, int dim, PolyTensor *index);

/* OpMixin._getitem's normalized syntax: one row per index, including new
 * axes; starts/sizes are scalar shape UOps before striding. NORMALIZED means
 * a host-list index whose negative entries were adjusted before _frompy.
 * These borrowed call arguments are not retained as another graph format. */
typedef enum {
  POLY_INDEX_NONE = 0,
  POLY_INDEX_INT = 1,
  POLY_INDEX_SLICE = 2,
  POLY_INDEX_TENSOR = 3,
  POLY_INDEX_NORMALIZED = 4
} PolyIndexKind;

PolyTensor *poly_tensor_getitem(
    PolyCtx *ctx,
    PolyTensor *self,
    const int *kinds,
    PolyUOp **starts,
    PolyUOp **sizes,
    const int64_t *steps,
    PolyTensor **indices,
    int n
);

/* Same normalized arguments; success0, invalid-1, conflicting live uses-2,
 * dtype mismatch-3, weak target-4, unsupported advanced DISK write-5,
 * incompatible index broadcast-6. */
int poly_tensor_setitem(
    PolyCtx *ctx,
    PolyTensor *self,
    const int *kinds,
    PolyUOp **starts,
    PolyUOp **sizes,
    const int64_t *steps,
    PolyTensor **indices,
    int n,
    PolyTensor *value
);

PolyTensor *poly_tensor_scatter(
    PolyCtx *ctx,
    PolyTensor *self,
    int dim,
    PolyTensor *index,
    PolyTensor *src,
    const char *reduce
);

PolyTensor *poly_tensor_scatter_reduce(
    PolyCtx *ctx,
    PolyTensor *self,
    int dim,
    PolyTensor *index,
    PolyTensor *src,
    const char *reduce,
    int include_self
);

PolyTensor *poly_tensor_einsum(
    PolyCtx *ctx,
    const char *formula,
    PolyTensor **tensors,
    int n_tensors
);

PolyTensor *poly_tensor_rearrange(
    PolyCtx *ctx,
    const char *formula,
    PolyTensor *tensor,
    const char *axis_names,
    const int64_t *axis_values,
    int n_axis_sizes
);

PolyUOp *poly_tensor_uop(PolyTensor *tensor);

PolyUOp *poly_tensor_uop_logical(PolyTensor *tensor);

PolyUOp *poly_tensor_uop_physical(PolyTensor *tensor);

PolyLogicalPolicy poly_tensor_logical_policy(const PolyTensor *tensor);

PolyLogicalState poly_tensor_logical_state(const PolyTensor *tensor);

int poly_tensor_set_logical_policy(PolyCtx *ctx, PolyTensor *tensor, PolyLogicalPolicy policy);

PolyDevice poly_tensor_device(PolyTensor *tensor);

PolyTensorProvenance poly_tensor_provenance(PolyTensor *tensor);

void poly_tensor_set_provenance(PolyTensor *tensor, PolyTensorProvenance provenance);

int poly_realize_tensors(PolyCtx *ctx, PolyTensor **inputs, int n, PolyTensor **outputs);

int poly_realize_tensors_ex(
    PolyCtx *ctx,
    PolyTensor **inputs,
    int n,
    PolyTensor **outputs,
    bool update_stats
);

/* C-only publication of already-built Tensor.assign effects. Shares view
 * alias retargeting with optimizer batches; does not realize or place roots. */
PolyTensor *poly_tensor_assign_after(
    PolyCtx *ctx,
    PolyTensor *target,
    PolyUOp *logical_after,
    PolyUOp *physical_after
);

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

/* Construction-only ownership for Model train/eval capture. The caller keeps
 * ctx alive and ends every successful begin, including after an error. Wrap
 * returns owned Tensors and restores authoring roots between passes; sealing
 * turns their auxiliary dependencies into ordinary SINK stores. RNG returns
 * two owned handles and a device, or -1 at end (failure poisons the capture).
 * No materialization of STORE effects, nested capture or seed reset is allowed. */
typedef struct PolyTensorCapture PolyTensorCapture;

PolyTensorCapture *poly_tensor_capture_begin(PolyCtx *ctx);

int poly_tensor_capture_wrap(
    PolyTensorCapture *capture,
    PolyTensor **states,
    const int *mutable_state,
    int n_states,
    PolyTensor **outputs,
    int n_outputs,
    PolyTensor **wrapped
);

int poly_tensor_capture_rng(
    PolyTensorCapture *capture,
    int index,
    PolyTensor **seed,
    PolyTensor **counter
);

void poly_tensor_capture_end(PolyTensorCapture *capture);

bool poly_tensor_capture_allows_realize(PolyCtx *ctx, PolyTensor **inputs, int n);

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

/* Paired Tensor boundaries for composed activations. These apply the exact
 * raw tinygrad-shaped program independently to retained logical and current
 * physical roots. */
PolyTensor *poly_tensor_relu(PolyCtx *ctx, PolyTensor *src);

PolyTensor *poly_tensor_sigmoid(PolyCtx *ctx, PolyTensor *src);

PolyTensor *poly_tensor_tanh(PolyCtx *ctx, PolyTensor *src);

PolyTensor *poly_tensor_silu(PolyCtx *ctx, PolyTensor *src);

/* Typed dtype options; ID conversion belongs to frontend.h. */
PolyTensor *poly_tensor_sum_dtype(
    PolyCtx *ctx,
    PolyTensor *src,
    int64_t *axes,
    int n_axes,
    bool keepdim,
    const PolyDType *dtype
);
PolyTensor *poly_tensor_dot_dtype(
    PolyCtx *ctx,
    PolyTensor *src,
    PolyTensor *weight,
    const PolyDType *dtype
);
PolyTensor *poly_tensor_conv2d_dtype(
    PolyCtx *ctx,
    PolyTensor *src,
    PolyTensor *weight,
    PolyTensor *bias,
    int groups,
    const int64_t *stride,
    const int64_t *dilation,
    const int64_t *padding,
    int n_padding,
    const PolyDType *dtype
);
PolyTensor *poly_tensor_cast(PolyCtx *ctx, PolyTensor *src, PolyDType dtype);
PolyTensor *poly_tensor_bitcast(PolyCtx *ctx, PolyTensor *src, PolyDType dtype);
PolyTensor *poly_tensor_rand(
    PolyCtx *ctx,
    const int64_t *dims,
    int ndim,
    PolyDType dtype,
    PolyDevice device,
    int contiguous
);
PolyTensor *poly_tensor_randn(
    PolyCtx *ctx,
    const int64_t *dims,
    int ndim,
    PolyDType dtype,
    PolyDevice device
);

#ifdef __cplusplus
}
#endif

#endif
