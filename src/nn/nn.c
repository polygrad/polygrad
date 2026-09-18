/* nn.c — Reusable raw-UOp and paired-Tensor layer programs. */

#include "nn/nn.h"
#include "tensor.h" /* poly_uop_mean_reduce */
#include "engine/schedule.h" /* poly_uop_reshape, poly_uop_permute, poly_uop_expand */
#include <stdint.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

/* Polygrad retains a portable logical twin, but default execution follows the
 * same ordered Tensor.uop construction as pinned Tensor._apply_uop
 * (tinygrad/tensor.py:128-140). Layer programs therefore run independently
 * over the exact logical and physical occurrences; neither root is recovered
 * from the other. */
static PolyTensor *nn_tensor_result(
    PolyCtx *ctx,
    PolyUOp *logical,
    PolyUOp *physical,
    PolyTensor **inputs,
    int n_inputs
) {
  int build_logical = poly_tensor_result_builds_logical(ctx, inputs, n_inputs);
  if (build_logical < 0 || !physical || (build_logical && !logical) ||
      (logical && !poly_ctx_owns_ptr(ctx, logical)) || !poly_ctx_owns_ptr(ctx, physical))
    return NULL;
  PolyDevice device = POLY_DEVICE_AUTO;
  for (int i = 0; i < n_inputs; i++) {
    PolyTensor *input = inputs[i];
    if (!input) continue;
    if (input->device != POLY_DEVICE_AUTO) {
      if (device != POLY_DEVICE_AUTO && device != input->device) return NULL;
      device = input->device;
    }
  }
  PolyTensor *out = poly_tensor_create_result(
      ctx, inputs, n_inputs, logical, physical, POLY_TENSOR_VALUE, device
  );
  if (!out) return NULL;
  out->provenance = POLY_TENSOR_PROVENANCE_COMPUTED;
  return out;
}

/* nn.LSTMCell: gate order is input, forget, candidate, output. There is no
 * recurrent executor: callers connect the returned state into the next call. */
int poly_uop_lstm_cell(
    PolyCtx *ctx,
    PolyUOp *x,
    PolyUOp *h,
    PolyUOp *c,
    PolyUOp *weight_ih,
    PolyUOp *weight_hh,
    PolyUOp *bias_ih,
    PolyUOp *bias_hh,
    PolyUOp **new_h,
    PolyUOp **new_c
) {
  if (!new_h || !new_c || new_h == new_c) return -1;
  *new_h = *new_c = NULL;
  if (!ctx || !x || !weight_ih || !weight_hh || (!!h != !!c)) return -1;
  if (poly_uop_ndim(ctx, x) != 2 || poly_uop_ndim(ctx, weight_hh) != 2) return -1;
  const int64_t *xs = poly_uop_max_shape_dims(ctx, x),
                *ws = poly_uop_max_shape_dims(ctx, weight_hh);
  if (!xs || !ws || ws[1] <= 0 || ws[1] > INT64_MAX / 4 || ws[0] != 4 * ws[1]) return -1;
  int64_t batch = xs[0], hidden = ws[1];
  if (!h) {
    PolyUOp *zero = poly_uop_const_typed(ctx, x->dtype, 0);
    h = poly_uop_expand(
        ctx, poly_uop_reshape(ctx, zero, (int64_t[]){1, 1}, 2), (int64_t[]){batch, hidden}, 2
    );
    c = h;
  }
  PolyUOp *gates = poly_uop_add(
      ctx, poly_uop_linear_apply(ctx, x, weight_ih, bias_ih),
      poly_uop_linear_apply(ctx, h, weight_hh, bias_hh)
  );
  if (!gates) return -1;
  PolyUOp *parts[4];
  for (int i = 0; i < 4; i++) {
    parts[i] =
        poly_uop_shrink(ctx, gates, (int64_t[][2]){{0, batch}, {i * hidden, (i + 1) * hidden}}, 2);
    parts[i] = i == 2 ? poly_uop_tanh(ctx, parts[i]) : poly_uop_sigmoid(ctx, parts[i]);
    if (!parts[i]) return -1;
  }
  PolyUOp *nc =
      poly_uop_add(ctx, poly_uop_mul(ctx, parts[1], c), poly_uop_mul(ctx, parts[0], parts[2]));
  PolyUOp *nh = poly_uop_mul(ctx, parts[3], poly_uop_tanh(ctx, nc));
  if (!nc || !nh) return -1;
  *new_h = nh;
  *new_c = nc;
  return 0;
}

int poly_tensor_lstm_cell(
    PolyCtx *ctx,
    PolyTensor *x,
    PolyTensor *h,
    PolyTensor *c,
    PolyTensor *weight_ih,
    PolyTensor *weight_hh,
    PolyTensor *bias_ih,
    PolyTensor *bias_hh,
    PolyTensor **new_h,
    PolyTensor **new_c
) {
  if (!new_h || !new_c || new_h == new_c) return -1;
  *new_h = *new_c = NULL;
  if (!ctx || !x || !weight_ih || !weight_hh || (!!h != !!c)) return -1;
  PolyTensor *inputs[] = {x, weight_ih, weight_hh, h, c, bias_ih, bias_hh};
  int logical = poly_tensor_result_builds_logical(ctx, inputs, 7);
  if (logical < 0) return -1;
  PolyUOp *hp = NULL, *cp = NULL, *hl = NULL, *cl = NULL;
  if (poly_uop_lstm_cell(
          ctx, x->uop_physical, h ? h->uop_physical : NULL, c ? c->uop_physical : NULL,
          weight_ih->uop_physical, weight_hh->uop_physical, bias_ih ? bias_ih->uop_physical : NULL,
          bias_hh ? bias_hh->uop_physical : NULL, &hp, &cp
      ) != 0)
    return -1;
  if (logical &&
      poly_uop_lstm_cell(
          ctx, x->uop_logical, h ? h->uop_logical : NULL, c ? c->uop_logical : NULL,
          weight_ih->uop_logical, weight_hh->uop_logical, bias_ih ? bias_ih->uop_logical : NULL,
          bias_hh ? bias_hh->uop_logical : NULL, &hl, &cl
      ) != 0)
    return -1;
  PolyTensor *ht = nn_tensor_result(ctx, hl, hp, inputs, 7);
  PolyTensor *ct = ht ? nn_tensor_result(ctx, cl, cp, inputs, 7) : NULL;
  if (!ct) {
    poly_tensor_release(ht);
    return -1;
  }
  *new_h = ht;
  *new_c = ct;
  return 0;
}

/* Linear */

PolyUOp *poly_uop_linear_apply(PolyCtx *ctx, PolyUOp *x, PolyUOp *w, PolyUOp *b) {
  if (!ctx || !x || !w) return NULL;
  int64_t perm[] = {1, 0};
  PolyUOp *out = poly_uop_dot(ctx, x, poly_uop_permute(ctx, w, perm, 2));
  if (!out) return NULL;
  if (b) out = poly_uop_add(ctx, out, b);
  return out;
}

PolyTensor *poly_tensor_linear_apply(PolyCtx *ctx, PolyTensor *x, PolyTensor *w, PolyTensor *b) {
  if (!ctx || !x || !w) return NULL;
  PolyTensor *inputs[3] = {x, w, b};
  int build_logical = poly_tensor_result_builds_logical(ctx, inputs, b ? 3 : 2);
  if (build_logical < 0) return NULL;
  /* Pinned nn.Linear stores (out,in), transposes it, then calls Tensor.linear;
   * Tensor.linear is dot followed by optional add
   * (nn/__init__.py:156-177; mixin/__init__.py:1335-1350). */
  PolyUOp *physical =
      poly_uop_linear_apply(ctx, x->uop_physical, w->uop_physical, b ? b->uop_physical : NULL);
  PolyUOp *logical =
      build_logical
          ? poly_uop_linear_apply(ctx, x->uop_logical, w->uop_logical, b ? b->uop_logical : NULL)
          : NULL;
  return nn_tensor_result(ctx, logical, physical, inputs, b ? 3 : 2);
}

/* LayerNorm */

PolyUOp *poly_uop_layernorm_apply(
    PolyCtx *ctx,
    PolyUOp *x,
    PolyUOp *w,
    PolyUOp *b,
    int axis,
    double eps
) {
  int64_t a = axis;
  return poly_uop_layernorm_axes_apply(ctx, x, w, b, &a, 1, eps);
}

PolyUOp *poly_uop_layernorm_axes_apply(
    PolyCtx *ctx,
    PolyUOp *x,
    PolyUOp *w,
    PolyUOp *b,
    const int64_t *axes,
    int n_axes,
    double eps
) {
  if (!ctx || !x || n_axes < 0 || n_axes > POLY_MAX_DIMS || (n_axes && !axes)) return NULL;
  int64_t reduce_axes[POLY_MAX_DIMS];
  if (n_axes) memcpy(reduce_axes, axes, (size_t)n_axes * sizeof(int64_t));
  /* nn.LayerNorm / Tensor.layernorm: reduce the declared axes together,
   * preserving the centered value and the input accumulation/cast rules. */
  PolyUOp *mean = poly_uop_mean_axes(ctx, x, reduce_axes, n_axes, true);
  PolyUOp *centered = poly_uop_sub(ctx, x, mean);
  PolyUOp *sq = poly_uop_alu2(ctx, POLY_OP_MUL, centered, centered);
  PolyUOp *var = poly_uop_mean_axes(ctx, sq, reduce_axes, n_axes, true);
  PolyUOp *normed = poly_uop_mul(
      ctx, centered,
      poly_uop_rsqrt(ctx, poly_uop_add(ctx, var, poly_uop_const_typed(ctx, POLY_WEAKFLOAT, eps)))
  );

  if (w) normed = poly_uop_mul(ctx, normed, w);
  if (b) normed = poly_uop_add(ctx, normed, b);

  return normed;
}

PolyTensor *poly_tensor_layernorm_apply(
    PolyCtx *ctx,
    PolyTensor *x,
    PolyTensor *w,
    PolyTensor *b,
    int axis,
    double eps
) {
  int64_t a = axis;
  return poly_tensor_layernorm_axes_apply(ctx, x, w, b, &a, 1, eps);
}

PolyTensor *poly_tensor_layernorm_axes_apply(
    PolyCtx *ctx,
    PolyTensor *x,
    PolyTensor *w,
    PolyTensor *b,
    const int64_t *axes,
    int n_axes,
    double eps
) {
  if (!ctx || !x || (!!w != !!b)) return NULL;
  PolyTensor *inputs[3] = {x, w, b};
  int build_logical = poly_tensor_result_builds_logical(ctx, inputs, w ? 3 : 1);
  if (build_logical < 0) return NULL;
  /* Pinned LayerNorm first runs Tensor.layernorm, then applies the affine
   * weight and bias (nn/__init__.py:235-261; mixin/__init__.py:1548-1564). */
  PolyUOp *physical = poly_uop_layernorm_axes_apply(
      ctx, x->uop_physical, w ? w->uop_physical : NULL, b ? b->uop_physical : NULL, axes, n_axes,
      eps
  );
  PolyUOp *logical = build_logical ? poly_uop_layernorm_axes_apply(
                                         ctx, x->uop_logical, w ? w->uop_logical : NULL,
                                         b ? b->uop_logical : NULL, axes, n_axes, eps
                                     )
                                   : NULL;
  return nn_tensor_result(ctx, logical, physical, inputs, w ? 3 : 1);
}

/* RMSNorm */

PolyUOp *poly_uop_groupnorm_apply(
    PolyCtx *ctx,
    PolyUOp *x,
    PolyUOp *w,
    PolyUOp *b,
    int groups,
    double eps
) {
  if (!ctx || !x || groups <= 0) return NULL;
  int ndim = poly_uop_ndim(ctx, x);
  int64_t shape[POLY_MAX_DIMS], affine[POLY_MAX_DIMS];
  PolyUOp *dims[POLY_MAX_DIMS];
  if (ndim < 2 || ndim > POLY_MAX_DIMS) return NULL;
  for (int i = 0; i < ndim; i++) {
    dims[i] = poly_uop_shape_dim(ctx, x, i);
    if (!dims[i] || (i && poly_uop_const_i64(dims[i], &shape[i]) != 0)) return NULL;
    affine[i] = i == 1 ? shape[i] : 1;
  }
  int64_t count = poly_shape_numel_checked(shape + 1, ndim - 1);
  if (count < 0 || shape[1] % groups || count % groups) return NULL;
  /* nn.GroupNorm reshapes channels/spatial dimensions into groups, then
   * invokes the same LayerNorm program. Affine parameters remain per-channel. */
  /* The batch is not reduced or folded into a group. Preserve its actual
   * shape expression, as GroupNorm.reshape(x.shape[0], groups, -1) does. */
  PolyUOp *group_shape[] = {
      dims[0], poly_uop_const_int(ctx, groups), poly_uop_const_int(ctx, count / groups)};
  PolyUOp *flat = poly_uop_reshape_symbolic(ctx, x, group_shape, 3);
  PolyUOp *out = poly_uop_reshape_symbolic(
      ctx, poly_uop_layernorm_apply(ctx, flat, NULL, NULL, -1, eps), dims, ndim
  );
  if (!w || !b) return out;
  return poly_uop_add(
      ctx, poly_uop_mul(ctx, out, poly_uop_reshape(ctx, w, affine, ndim)),
      poly_uop_reshape(ctx, b, affine, ndim)
  );
}

PolyTensor *poly_tensor_groupnorm_apply(
    PolyCtx *ctx,
    PolyTensor *x,
    PolyTensor *w,
    PolyTensor *b,
    int groups,
    double eps
) {
  if (!ctx || !x) return NULL;
  PolyTensor *inputs[] = {x, w, b};
  int logical = poly_tensor_result_builds_logical(ctx, inputs, 3);
  if (logical < 0) return NULL;
  PolyUOp *p = poly_uop_groupnorm_apply(
      ctx, x->uop_physical, w ? w->uop_physical : NULL, b ? b->uop_physical : NULL, groups, eps
  );
  PolyUOp *l = logical ? poly_uop_groupnorm_apply(
                             ctx, x->uop_logical, w ? w->uop_logical : NULL,
                             b ? b->uop_logical : NULL, groups, eps
                         )
                       : NULL;
  return nn_tensor_result(ctx, l, p, inputs, 3);
}

/* nn.BatchNorm.calc_stats: variance uses a detached mean, not a detached x.
 * Inference broadcasts running variance while leaving the mean channel-shaped. */
static int batchnorm_stats(
    PolyCtx *ctx,
    PolyUOp *x,
    PolyUOp *running_mean,
    PolyUOp *running_var,
    bool training,
    PolyUOp **mean,
    PolyUOp **var
) {
  if (!ctx || !x || (!!running_mean != !!running_var)) return -1;
  int ndim = poly_uop_ndim(ctx, x);
  if (ndim < 2 || ndim > POLY_MAX_DIMS) return -1;
  PolyUOp *shape[POLY_MAX_DIMS];
  int64_t axes[POLY_MAX_DIMS];
  int n_axes = 0;
  for (int i = 0; i < ndim; i++) {
    shape[i] = i == 1 ? poly_uop_shape_dim(ctx, x, 1) : poly_uop_const_int(ctx, 1);
    if (i != 1) axes[n_axes++] = i;
  }
  if (running_mean && !training) {
    *mean = running_mean;
    *var = poly_uop_reshape_symbolic(ctx, running_var, shape, ndim);
  } else {
    *mean = poly_uop_mean_axes(ctx, x, axes, n_axes, false);
    PolyUOp *centered = poly_uop_sub(
        ctx, x, poly_uop_reshape_symbolic(ctx, poly_uop_detach(ctx, *mean), shape, ndim)
    );
    *var = poly_uop_mean_axes(ctx, poly_uop_mul(ctx, centered, centered), axes, n_axes, false);
  }
  return *mean && *var ? 0 : -1;
}

int poly_tensor_batchnorm_stats(
    PolyCtx *ctx,
    PolyTensor *x,
    PolyTensor *running_mean,
    PolyTensor *running_var,
    bool training,
    PolyTensor **mean,
    PolyTensor **var
) {
  if (!mean || !var || mean == var) return -1;
  *mean = *var = NULL;
  if (!ctx || !x || (!!running_mean != !!running_var)) return -1;
  PolyTensor *inputs[] = {x, running_mean, running_var};
  int logical = poly_tensor_result_builds_logical(ctx, inputs, 3);
  if (logical < 0) return -1;
  PolyUOp *p[2] = {0}, *l[2] = {0};
  if (batchnorm_stats(
          ctx, x->uop_physical, running_mean ? running_mean->uop_physical : NULL,
          running_var ? running_var->uop_physical : NULL, training, &p[0], &p[1]
      ) != 0 ||
      (logical && batchnorm_stats(
                      ctx, x->uop_logical, running_mean ? running_mean->uop_logical : NULL,
                      running_var ? running_var->uop_logical : NULL, training, &l[0], &l[1]
                  ) != 0))
    return -1;
  PolyTensor *m = nn_tensor_result(ctx, l[0], p[0], inputs, 3);
  PolyTensor *v = m ? nn_tensor_result(ctx, l[1], p[1], inputs, 3) : NULL;
  if (!v) {
    if (m) poly_tensor_release(m);
    return -1;
  }
  *mean = m;
  *var = v;
  return 0;
}

PolyTensor *poly_tensor_batchnorm_apply(
    PolyCtx *ctx,
    PolyTensor *x,
    PolyTensor *w,
    PolyTensor *b,
    PolyTensor *running_mean,
    PolyTensor *running_var,
    PolyTensor *num_batches,
    bool training,
    double eps,
    double momentum
) {
  if (!ctx || !x || (!!running_mean != !!running_var) || (running_mean && !num_batches))
    return NULL;
  PolyTensor *inputs[] = {x, w, b, running_mean, running_var, num_batches};
  int logical = poly_tensor_result_builds_logical(ctx, inputs, 6);
  if (logical < 0) return NULL;
  PolyTensor *mean = NULL, *var = NULL, *out = NULL, *updates[3] = {0};
  if (poly_tensor_batchnorm_stats(ctx, x, running_mean, running_var, training, &mean, &var) != 0)
    return NULL;
  PolyUOp *p[4] = {0}, *l[4] = {0};
  bool update = training && running_mean;
  for (int domain = 0; domain < 1 + logical; domain++) {
    PolyUOp *roots[6];
    for (int i = 0; i < 6; i++)
      roots[i] = inputs[i] ? (domain ? inputs[i]->uop_logical : inputs[i]->uop_physical) : NULL;
    PolyUOp **r = domain ? l : p;
    PolyUOp *m = domain ? mean->uop_logical : mean->uop_physical;
    PolyUOp *v = domain ? var->uop_logical : var->uop_physical;
    int64_t channel = 1;
    r[0] = poly_uop_batchnorm(
        ctx, roots[0], roots[1], roots[2], m,
        poly_uop_rsqrt(ctx, poly_uop_add(ctx, v, poly_uop_const_typed(ctx, POLY_WEAKFLOAT, eps))),
        &channel, 1
    );
    if (update) {
      int ndim = poly_uop_ndim(ctx, roots[0]);
      int64_t shape[POLY_MAX_DIMS];
      for (int i = 0; i < ndim; i++)
        if (poly_uop_const_i64(poly_uop_shape_dim(ctx, roots[0], i), &shape[i]) != 0) goto done;
      int64_t count = poly_shape_numel_checked(shape, ndim);
      if (count <= shape[1]) goto done;
      PolyUOp *remain = poly_uop_const_typed(ctx, POLY_WEAKFLOAT, 1 - momentum);
      r[1] = poly_uop_add(
          ctx, poly_uop_mul(ctx, remain, roots[3]),
          poly_uop_mul(
              ctx, poly_uop_const_typed(ctx, POLY_WEAKFLOAT, momentum), poly_uop_detach(ctx, m)
          )
      );
      r[2] = poly_uop_add(
          ctx, poly_uop_mul(ctx, remain, roots[4]),
          poly_uop_mul(
              ctx, poly_uop_const_typed(ctx, POLY_WEAKFLOAT, momentum * count / (count - shape[1])),
              poly_uop_detach(ctx, v)
          )
      );
      r[3] = poly_uop_add(ctx, roots[5], poly_uop_const_int(ctx, 1));
    }
  }
  /* Prepare every result before the pinned ordered Tensor.assign calls.
   * Assignment/view alias publication remains owned by Tensor, not NN. */
  out = nn_tensor_result(ctx, l[0], p[0], inputs, 6);
  if (!out) goto done;
  if (update) {
    for (int i = 0; i < 3; i++) {
      updates[i] = nn_tensor_result(ctx, l[i + 1], p[i + 1], inputs, 6);
      if (!updates[i] || !poly_dtype_eq(p[i + 1]->dtype, inputs[i + 3]->uop_physical->dtype))
        goto fail;
    }
    for (int i = 0; i < 3; i++)
      if (!poly_tensor_assign(ctx, inputs[i + 3], updates[i])) goto fail;
  }
  goto done;
fail:
  poly_tensor_release(out);
  out = NULL;
done:
  for (int i = 0; i < 3; i++)
    if (updates[i]) poly_tensor_release(updates[i]);
  poly_tensor_release(mean);
  poly_tensor_release(var);
  return out;
}

PolyUOp *poly_uop_rmsnorm_apply(PolyCtx *ctx, PolyUOp *x, PolyUOp *w, double eps) {
  if (!ctx || !x) return NULL;
  /* nn.RMSNorm._norm/__call__: normalize in float32 even for half/double,
   * cast back before affine multiplication, and let UOp infer broadcasting. */
  PolyUOp *xf = poly_uop_cast(ctx, x, POLY_FLOAT32);
  PolyUOp *mean = poly_uop_mean_reduce(ctx, poly_uop_mul(ctx, xf, xf), -1, 1);
  PolyUOp *scale =
      poly_uop_rsqrt(ctx, poly_uop_add(ctx, mean, poly_uop_const_typed(ctx, POLY_WEAKFLOAT, eps)));
  PolyUOp *normed = poly_uop_cast(ctx, poly_uop_mul(ctx, xf, scale), x->dtype);
  return w ? poly_uop_mul(ctx, normed, w) : normed;
}

PolyTensor *poly_tensor_rmsnorm_apply(PolyCtx *ctx, PolyTensor *x, PolyTensor *w, double eps) {
  if (!ctx || !x) return NULL;
  PolyTensor *inputs[2] = {x, w};
  int build_logical = poly_tensor_result_builds_logical(ctx, inputs, w ? 2 : 1);
  if (build_logical < 0) return NULL;
  /* Pinned RMSNorm normalizes x.float(), casts back, and applies the optional
   * affine weight (nn/__init__.py:281-304). The existing raw program is run
   * over both retained occurrences without correspondence. */
  PolyUOp *physical = poly_uop_rmsnorm_apply(ctx, x->uop_physical, w ? w->uop_physical : NULL, eps);
  PolyUOp *logical =
      build_logical ? poly_uop_rmsnorm_apply(ctx, x->uop_logical, w ? w->uop_logical : NULL, eps)
                    : NULL;
  return nn_tensor_result(ctx, logical, physical, inputs, w ? 2 : 1);
}

/* Embedding */

/* nn.InstanceNorm reshapes spatial dimensions into one normalization axis.
 * Affine state remains caller-owned, so C models and frontends share this graph. */
PolyUOp *poly_uop_instancenorm_apply(
    PolyCtx *ctx,
    PolyUOp *x,
    PolyUOp *w,
    PolyUOp *b,
    int num_features,
    double eps
) {
  if (!ctx || !x || num_features <= 0) return NULL;
  int ndim = poly_uop_ndim(ctx, x);
  const int64_t *dims = poly_uop_max_shape_dims(ctx, x);
  if (ndim < 2 || ndim > POLY_MAX_DIMS || !dims) return NULL;
  int64_t shape[POLY_MAX_DIMS], affine[POLY_MAX_DIMS];
  memcpy(shape, dims, (size_t)ndim * sizeof(int64_t));
  int64_t count = poly_shape_numel_checked(shape + 1, ndim - 1);
  if (count < 0 || count % num_features) return NULL;
  PolyUOp *flat =
      poly_uop_reshape(ctx, x, (int64_t[]){shape[0], num_features, count / num_features}, 3);
  PolyUOp *result =
      poly_uop_reshape(ctx, poly_uop_layernorm_apply(ctx, flat, NULL, NULL, -1, eps), shape, ndim);
  if (!w || !b) return result;
  for (int i = 0; i < ndim; i++)
    affine[i] = i == 1 ? num_features : 1;
  return poly_uop_add(
      ctx, poly_uop_mul(ctx, result, poly_uop_reshape(ctx, w, affine, ndim)),
      poly_uop_reshape(ctx, b, affine, ndim)
  );
}

PolyTensor *poly_tensor_instancenorm_apply(
    PolyCtx *ctx,
    PolyTensor *x,
    PolyTensor *w,
    PolyTensor *b,
    int num_features,
    double eps
) {
  if (!ctx || !x) return NULL;
  PolyTensor *inputs[] = {x, w, b};
  int logical = poly_tensor_result_builds_logical(ctx, inputs, 3);
  if (logical < 0) return NULL;
  PolyUOp *p = poly_uop_instancenorm_apply(
      ctx, x->uop_physical, w ? w->uop_physical : NULL, b ? b->uop_physical : NULL, num_features,
      eps
  );
  PolyUOp *l = logical ? poly_uop_instancenorm_apply(
                             ctx, x->uop_logical, w ? w->uop_logical : NULL,
                             b ? b->uop_logical : NULL, num_features, eps
                         )
                       : NULL;
  return nn_tensor_result(ctx, l, p, inputs, 3);
}

PolyUOp *poly_uop_embedding_apply(PolyCtx *ctx, PolyUOp *tokens, PolyUOp *table) {
  return poly_uop_gather(ctx, table, tokens);
}

PolyTensor *poly_tensor_embedding_apply(PolyCtx *ctx, PolyTensor *tokens, PolyTensor *table) {
  if (!ctx || !tokens || !table) return NULL;
  PolyTensor *inputs[2] = {tokens, table};
  int build_logical = poly_tensor_result_builds_logical(ctx, inputs, 2);
  if (build_logical < 0) return NULL;
  /* Pinned Embedding is its one-hot WHERE/SUM program over the ordered weight
   * and index Tensor.uops (nn/__init__.py:368-391). */
  PolyUOp *physical = poly_uop_embedding_apply(ctx, tokens->uop_physical, table->uop_physical);
  PolyUOp *logical =
      build_logical ? poly_uop_embedding_apply(ctx, tokens->uop_logical, table->uop_logical) : NULL;
  return nn_tensor_result(ctx, logical, physical, inputs, 2);
}

/* Causal attention mask */

PolyUOp *poly_uop_causal_mask(PolyCtx *ctx, int64_t T) {
  if (!ctx || T <= 0) return NULL;

  PolyUOp *arange_buf = poly_uop_arange(ctx, 0.0, (double)T, 1.0);
  int64_t row_shape[] = {T, 1};
  PolyUOp *row =
      poly_uop_expand(ctx, poly_uop_reshape(ctx, arange_buf, row_shape, 2), (int64_t[]){T, T}, 2);
  int64_t col_shape[] = {1, T};
  PolyUOp *col =
      poly_uop_expand(ctx, poly_uop_reshape(ctx, arange_buf, col_shape, 2), (int64_t[]){T, T}, 2);

  PolyUOp *mask = poly_uop_alu2(ctx, POLY_OP_CMPLT, row, col);
  return poly_uop_where(ctx, mask, poly_uop_const_float(ctx, -1e9), poly_uop_const_float(ctx, 0.0));
}

PolyTensor *poly_tensor_causal_mask(PolyCtx *ctx, int64_t T) {
  if (!ctx || T <= 0) return NULL;
  bool build_logical = poly_ctx_get_logical_policy(ctx) != POLY_LOGICAL_NEVER;
  /* Pinned GPT-2/LLaMA mask construction is a pure Tensor graph. With no
   * BUFFER occurrence, retained and executable roots may CSE to the same
   * node, but both approved roots are stored explicitly. */
  PolyUOp *physical = poly_uop_causal_mask(ctx, T);
  PolyUOp *logical = build_logical ? poly_uop_causal_mask(ctx, T) : NULL;
  if (!physical || (build_logical && !logical)) return NULL;
  PolyTensor *out = poly_tensor_create_with_roots(
      ctx, logical, physical, POLY_TENSOR_VALUE, poly_ctx_get_preferred_device(ctx)
  );
  if (!out) return NULL;
  out->provenance = POLY_TENSOR_PROVENANCE_COMPUTED;
  return out;
}

/* Scaled Dot-Product Attention */

/* RandMixin.scaled_dot_product_attention: return probabilities and the
 * optionally repeated V. Both public boundaries use this exact program. */
static PolyUOp *sdpa_weights(
    PolyCtx *ctx,
    PolyUOp *q,
    PolyUOp *k,
    PolyUOp **value,
    PolyUOp *mask,
    int is_causal,
    int enable_gqa
) {
  PolyUOp *v = *value;
  if (!ctx || !q || !k || !v || (is_causal && mask)) return NULL;
  int q_ndim = poly_uop_ndim(ctx, q);
  int k_ndim = poly_uop_ndim(ctx, k);
  int v_ndim = poly_uop_ndim(ctx, v);
  if (q_ndim < 2 || k_ndim < 2 || v_ndim < 2) return NULL;
  const int64_t *q_dims = poly_uop_max_shape_dims(ctx, q);
  if (!q_dims) return NULL;

  if (enable_gqa) {
    if (q_ndim < 3 || k_ndim < 3 || v_ndim < 3) return NULL;
    int64_t heads = q_dims[q_ndim - 3];
    PolyUOp **kv[] = {&k, &v};
    for (int i = 0; i < 2; i++) {
      int ndim = poly_uop_ndim(ctx, *kv[i]);
      const int64_t *dims = poly_uop_max_shape_dims(ctx, *kv[i]);
      if (!dims || dims[ndim - 3] <= 0 || heads / dims[ndim - 3] > INT_MAX) return NULL;
      *kv[i] = poly_uop_repeat_interleave(ctx, *kv[i], (int)(heads / dims[ndim - 3]), -3);
      if (!*kv[i]) return NULL;
    }
  }
  *value = v;

  int64_t d_k = q_dims[q_ndim - 1];
  /* Tinygrad permits zero-width heads: dot / sqrt(0) propagates NaNs. */
  if (d_k < 0) return NULL;

  int64_t k_perm[POLY_MAX_DIMS];
  for (int i = 0; i < k_ndim; i++)
    k_perm[i] = i;
  k_perm[k_ndim - 2] = k_ndim - 1;
  k_perm[k_ndim - 1] = k_ndim - 2;
  PolyUOp *k_t = poly_uop_permute(ctx, k, k_perm, k_ndim);

  /* RandMixin.scaled_dot_product_attention: dot in at least float32, divide
   * by weak sqrt(head_dim), then cast scores back before softmax. */
  PolyDType acc;
  if (!poly_dtype_least_upper(q->dtype, k->dtype, &acc) ||
      !poly_dtype_least_upper(acc, POLY_FLOAT32, &acc))
    return NULL;
  PolyUOp *scores = poly_uop_dot_dtype(ctx, q, k_t, &acc);
  scores = poly_uop_div(ctx, scores, poly_uop_const_typed(ctx, POLY_WEAKFLOAT, sqrt((double)d_k)));

  if (is_causal) {
    /* const_like preserves the score dtype unless explicitly overridden. */
    mask = poly_uop_tril(
        ctx, poly_uop_const_like_dtype(ctx, scores, poly_arg_bool(true), POLY_BOOL), 0
    );
    if (!mask) return NULL;
  }

  if (mask && poly_dtype_eq(mask->dtype, POLY_BOOL)) {
    mask = poly_uop_where(
        ctx, mask, poly_uop_const_int(ctx, 0), poly_uop_const_typed(ctx, POLY_WEAKFLOAT, -INFINITY)
    );
    if (!mask) return NULL;
  }
  if (mask) scores = poly_uop_add(ctx, scores, mask);

  return poly_uop_softmax(ctx, poly_uop_cast(ctx, scores, q->dtype), -1);
}

PolyUOp *poly_uop_sdpa(
    PolyCtx *ctx,
    PolyUOp *q,
    PolyUOp *k,
    PolyUOp *v,
    PolyUOp *mask,
    int is_causal,
    int enable_gqa
) {
  PolyUOp *weights = sdpa_weights(ctx, q, k, &v, mask, is_causal, enable_gqa);
  return weights ? poly_uop_dot(ctx, weights, v) : NULL;
}

static PolyUOp *dropout(PolyCtx *ctx, PolyUOp *x, PolyUOp *noise, double p) {
  if (p == 1) return poly_uop_const_like(ctx, x, poly_arg_int(0));
  PolyUOp *keep = poly_uop_contiguous(
      ctx, poly_uop_ge(ctx, noise, poly_uop_const_typed(ctx, POLY_WEAKFLOAT, p))
  );
  return poly_uop_div(
      ctx, poly_uop_where(ctx, keep, x, poly_uop_const_int(ctx, 0)),
      poly_uop_const_typed(ctx, POLY_WEAKFLOAT, 1 - p)
  );
}

PolyTensor *poly_tensor_dropout(PolyCtx *ctx, PolyTensor *x, double p, int training) {
  if (!ctx || !x) return NULL;
  PolyTensor *inputs[] = {x};
  int build_logical = poly_tensor_result_builds_logical(ctx, inputs, 1);
  if (build_logical < 0 || !(p >= 0 && p <= 1)) return NULL;
  if (!training || p == 0) return poly_tensor_retain(x);
  PolyTensor *noise = NULL;
  if (p != 1) {
    int ndim = poly_uop_ndim(ctx, x->uop_physical);
    int64_t dims[POLY_MAX_DIMS];
    if (ndim < 0 || ndim > POLY_MAX_DIMS) return NULL;
    for (int i = 0; i < ndim; i++) {
      PolyUOp *dim = poly_uop_shape_dim(ctx, x->uop_physical, i);
      /* RandMixin.rand requires concrete extents; never draw at a symbolic
       * maximum and silently change the RNG stream for a smaller binding. */
      if (!dim || dim->op != POLY_OP_CONST || dim->arg.kind != POLY_ARG_INT) return NULL;
      dims[i] = dim->arg.i;
    }
    /* One RNG draw owns both graph occurrences; do not advance twice for
     * logical retention. The existing RNG owner retains seed/counter state. */
    PolyDevice device = x->device;
    if (device == POLY_DEVICE_AUTO) device = poly_ctx_get_preferred_device(ctx);
    if (device == POLY_DEVICE_AUTO) device = poly_device_default();
    PolyDType dtype;
    if (!poly_dtype_by_id(poly_get_default_float(), &dtype)) return NULL;
    noise = poly_tensor_rand(ctx, dims, ndim, dtype, device, 0);
    if (!noise || (build_logical && !noise->uop_logical)) {
      poly_tensor_release(noise);
      return NULL;
    }
  }
  PolyUOp *physical = dropout(ctx, x->uop_physical, noise ? noise->uop_physical : NULL, p);
  PolyUOp *logical =
      build_logical ? dropout(ctx, x->uop_logical, noise ? noise->uop_logical : NULL, p) : NULL;
  PolyTensor *out = nn_tensor_result(ctx, logical, physical, inputs, 1);
  poly_tensor_release(noise);
  return out;
}

PolyTensor *poly_tensor_sdpa(
    PolyCtx *ctx,
    PolyTensor *q,
    PolyTensor *k,
    PolyTensor *v,
    PolyTensor *mask,
    double dropout_p,
    int is_causal,
    int enable_gqa,
    int training
) {
  if (!ctx || !q || !k || !v) return NULL;
  PolyTensor *inputs[4] = {q, k, v, mask};
  int build_logical = poly_tensor_result_builds_logical(ctx, inputs, mask ? 4 : 3);
  if (build_logical < 0 || !(dropout_p >= 0 && dropout_p <= 1)) return NULL;
  PolyUOp *vp = v->uop_physical, *vl = v->uop_logical;
  PolyUOp *physical = sdpa_weights(
      ctx, q->uop_physical, k->uop_physical, &vp, mask ? mask->uop_physical : NULL, is_causal,
      enable_gqa
  );
  PolyUOp *logical = build_logical ? sdpa_weights(
                                         ctx, q->uop_logical, k->uop_logical, &vl,
                                         mask ? mask->uop_logical : NULL, is_causal, enable_gqa
                                     )
                                   : NULL;
  PolyTensor *weights = nn_tensor_result(ctx, logical, physical, inputs, mask ? 4 : 3);
  if (!weights) return NULL;
  PolyTensor *dropped = poly_tensor_dropout(ctx, weights, dropout_p, training);
  PolyTensor *out =
      dropped ? nn_tensor_result(
                    ctx, build_logical ? poly_uop_dot(ctx, dropped->uop_logical, vl) : NULL,
                    poly_uop_dot(ctx, dropped->uop_physical, vp), inputs, mask ? 4 : 3
                )
              : NULL;
  poly_tensor_release(dropped);
  poly_tensor_release(weights);
  return out;
}
