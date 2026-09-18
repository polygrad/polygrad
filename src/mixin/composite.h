/* mixin/composite.h -- Shape-aware reductions, indexing, losses and linear algebra on UOp graphs.
 */
#ifndef POLY_MIXIN_COMPOSITE_H
#define POLY_MIXIN_COMPOSITE_H

#include "../core.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Tensor.dot(dtype=): widen the reduction, not the multiplied operands. */
PolyUOp *poly_uop_dot_dtype(PolyCtx *ctx, PolyUOp *x, PolyUOp *w, const PolyDType *dtype);

/* Tensor losses compose shared elementwise graphs and reductions. */
PolyUOp *poly_uop_binary_crossentropy(PolyCtx *ctx, PolyUOp *x, PolyUOp *target, int reduction);

PolyUOp *poly_uop_binary_crossentropy_logits(
    PolyCtx *ctx,
    PolyUOp *x,
    PolyUOp *target,
    PolyUOp *weight,
    int reduction
);

PolyUOp *poly_uop_nll_loss(
    PolyCtx *ctx,
    PolyUOp *x,
    PolyUOp *target,
    PolyUOp *weight,
    PolyUOp *ignore_index,
    int reduction
);

PolyUOp *poly_uop_tril(PolyCtx *ctx, PolyUOp *x, int diagonal);

PolyUOp *poly_uop_triu(PolyCtx *ctx, PolyUOp *x, int diagonal);

PolyUOp *poly_uop_cholesky(PolyCtx *ctx, PolyUOp *x, int upper);

PolyUOp *poly_uop_cholesky_solve(PolyCtx *ctx, PolyUOp *chol, PolyUOp *b, int upper);

PolyUOp *poly_uop_triangular_solve(
    PolyCtx *ctx,
    PolyUOp *a,
    PolyUOp *b,
    int upper,
    int transpose_a,
    int unit_diagonal
);

/* Shape-aware composed ops (shape read from UOp) */

PolyUOp *poly_uop_sum_reduce(PolyCtx *ctx, PolyUOp *x, int axis, int keepdim);

PolyUOp *poly_uop_max_reduce(PolyCtx *ctx, PolyUOp *x, int axis, int keepdim);

PolyUOp *poly_uop_mean_reduce(PolyCtx *ctx, PolyUOp *x, int axis, int keepdim);

PolyUOp *poly_uop_mean_axes(PolyCtx *ctx, PolyUOp *x, int64_t *axes, int n_axes, bool keepdim);

PolyUOp *poly_uop_var_reduce(PolyCtx *ctx, PolyUOp *x, int axis, int keepdim, int correction);

PolyUOp *poly_uop_logsumexp(PolyCtx *ctx, PolyUOp *x, int axis, int keepdim);

PolyUOp *poly_uop_dot(PolyCtx *ctx, PolyUOp *x, PolyUOp *w);

PolyUOp *poly_uop_newton_schulz(
    PolyCtx *ctx,
    PolyUOp *x,
    int steps,
    const double *coefficients,
    int n_coefficients,
    double eps
);

#define POLY_QR_COMPLETE 0
#define POLY_QR_REDUCED 1
#define POLY_QR_R_ONLY 2
int poly_uop_qr(PolyCtx *ctx, PolyUOp *x, PolyUOp **out_q, PolyUOp **out_r);

int poly_uop_qr_ex(PolyCtx *ctx, PolyUOp *x, int mode, PolyUOp **out_q, PolyUOp **out_r);

PolyUOp *poly_uop_solve(PolyCtx *ctx, PolyUOp *a, PolyUOp *b);

PolyUOp *poly_uop_lstsq(PolyCtx *ctx, PolyUOp *a, PolyUOp *b);

PolyUOp *poly_uop_softmax(PolyCtx *ctx, PolyUOp *x, int axis);

PolyUOp *poly_uop_log_softmax(PolyCtx *ctx, PolyUOp *x, int axis);

PolyUOp *poly_uop_cross_entropy(PolyCtx *ctx, PolyUOp *logits, PolyUOp *target, int axis);

/* Tinygrad-style sort/topk composed helpers.
 * Return 0 on success and populate both outputs. */
int poly_uop_sort(
    PolyCtx *ctx,
    PolyUOp *x,
    int dim,
    int descending,
    PolyUOp **out_values,
    PolyUOp **out_indices
);

PolyUOp *poly_uop_argsort(PolyCtx *ctx, PolyUOp *x, int dim, int descending);

int poly_uop_topk(
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

PolyUOp *poly_uop_einsum(PolyCtx *ctx, const char *formula, PolyUOp **tensors, int n_tensors);

/* Rearrange (einops) */

PolyUOp *poly_uop_rearrange(
    PolyCtx *ctx,
    const char *formula,
    PolyUOp *x,
    const char *axis_names,
    const int64_t *axis_values,
    int n_axis_sizes
);

/* Gather (embedding lookup) */

PolyUOp *poly_uop_gather(PolyCtx *ctx, PolyUOp *table, PolyUOp *indices);

PolyUOp *poly_uop_gather_dim(PolyCtx *ctx, PolyUOp *x, int dim, PolyUOp *index);

PolyUOp *poly_uop_scatter(
    PolyCtx *ctx,
    PolyUOp *self,
    int dim,
    PolyUOp *index,
    PolyUOp *src,
    const char *reduce
);

PolyUOp *poly_uop_scatter_reduce(
    PolyCtx *ctx,
    PolyUOp *self,
    int dim,
    PolyUOp *index,
    PolyUOp *src,
    const char *reduce,
    int include_self
);

/* Additional composed ops */

PolyUOp *poly_uop_rope(PolyCtx *ctx, PolyUOp *x, PolyUOp *freqs_cos, PolyUOp *freqs_sin);

PolyUOp *poly_uop_repeat_interleave(PolyCtx *ctx, PolyUOp *x, int repeats, int dim);

PolyUOp *poly_uop_argmax(PolyCtx *ctx, PolyUOp *x, int axis, int keepdim);

PolyUOp *poly_uop_mse_loss(PolyCtx *ctx, PolyUOp *pred, PolyUOp *target);

PolyUOp *poly_uop_mae_loss(PolyCtx *ctx, PolyUOp *pred, PolyUOp *target);

/* MovementMixin._pool: append one window axis per pooled dimension.
 * NULL stride/dilation selects the default of one. */
PolyUOp *poly_uop_pool(
    PolyCtx *ctx,
    PolyUOp *x,
    const int64_t *k_,
    int nk,
    const int64_t *stride_,
    const int64_t *dilation_
);

PolyUOp *poly_uop_max_pool2d(
    PolyCtx *ctx,
    PolyUOp *x,
    const int64_t *kernel,
    int n_kernel,
    const int64_t *stride,
    const int64_t *dilation,
    const int64_t *padding,
    int n_padding
);

PolyUOp *poly_uop_conv2d(
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

PolyUOp *poly_uop_batchnorm(
    PolyCtx *ctx,
    PolyUOp *x,
    PolyUOp *weight,
    PolyUOp *bias,
    PolyUOp *mean,
    PolyUOp *invstd,
    const int64_t *axes,
    int n_axes
);

PolyUOp *poly_uop_one_hot(PolyCtx *ctx, PolyUOp *x, int64_t num_classes);

PolyUOp *poly_uop_index_select(PolyCtx *ctx, PolyUOp *x, int dim, PolyUOp *index);

#ifdef __cplusplus
}
#endif

#endif
