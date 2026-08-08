/*
 * optim.c -- Open optimizer graph helpers.
 */

#include "optim.h"
#include "tensor.h"
#include <stdlib.h>
#include <string.h>

static int64_t optim_uop_numel(PolyCtx *ctx, PolyUOp *u) {
  if (!ctx || !u) return -1;
  int ndim = poly_uop_ndim(ctx, u);
  if (ndim < 0) return -1;
  if (ndim == 0) return 1;
  const int64_t *dims = poly_uop_max_shape_dims(ctx, u);
  return poly_shape_numel_checked(dims, ndim);
}

static PolyUOp *optim_lr_uop(PolyCtx *ctx, PolyUOp *lr) {
  PolyDType dtype;
  if (!ctx || !lr || !poly_ctx_owns_ptr(ctx, lr)) return NULL;
  dtype = lr->dtype;
  if (dtype.is_ptr || dtype.count != 1 || !poly_dtype_is_float(dtype) || dtype.bitsize < 32)
    return NULL;
  int ndim = poly_uop_ndim(ctx, lr);
  if (ndim == 0) return lr;
  if (ndim != 1) return NULL;
  const int64_t *dims = poly_uop_max_shape_dims(ctx, lr);
  if (!dims || dims[0] != 1) return NULL;
  PolyUOp *dim = poly_uop_shape_dim(ctx, lr, 0);
  int64_t static_dim = 0;
  if (!dim || poly_uop_const_i64(dim, &static_dim) != 0 || static_dim != 1) return NULL;
  return lr;
}

static bool optim_same_shape(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  if (!ctx || !a || !b) return false;
  int a_ndim = poly_uop_ndim(ctx, a);
  int b_ndim = poly_uop_ndim(ctx, b);
  if (a_ndim < 0 || a_ndim != b_ndim) return false;
  const int64_t *a_dims = poly_uop_max_shape_dims(ctx, a);
  const int64_t *b_dims = poly_uop_max_shape_dims(ctx, b);
  if (a_ndim > 0 && (!a_dims || !b_dims)) return false;
  for (int i = 0; i < a_ndim; i++)
    if (a_dims[i] != b_dims[i]) return false;
  return true;
}

/* Pinned OpMixin._broadcasted/_broadcast_to
 * (mixin/__init__.py:439-450): align dimensions on the right, introduce
 * leading singleton dimensions with RESHAPE, then EXPAND only when a
 * singleton actually broadcasts. Optimizer scalar constants use this exact
 * Tensor spelling instead of relying on implicit scheduler broadcasting. */
static PolyUOp *optim_broadcast_like(PolyCtx *ctx, PolyUOp *value, PolyUOp *like) {
  if (!ctx || !value || !like) return NULL;
  if (optim_same_shape(ctx, value, like)) return value;
  int value_ndim = poly_uop_ndim(ctx, value);
  int like_ndim = poly_uop_ndim(ctx, like);
  if (value_ndim < 0 || like_ndim < 0 || value_ndim > like_ndim ||
      like_ndim > POLY_MAX_DIMS)
    return NULL;
  const int64_t *value_dims = poly_uop_max_shape_dims(ctx, value);
  const int64_t *like_dims = poly_uop_max_shape_dims(ctx, like);
  if ((value_ndim > 0 && !value_dims) || (like_ndim > 0 && !like_dims)) return NULL;

  PolyUOp *shaped = value;
  int64_t aligned[POLY_MAX_DIMS];
  bool needs_reshape = value_ndim != like_ndim;
  for (int i = 0; i < like_ndim; i++) {
    int source_axis = i - (like_ndim - value_ndim);
    aligned[i] = source_axis >= 0 ? value_dims[source_axis] : 1;
    if (aligned[i] != 1 && aligned[i] != like_dims[i]) return NULL;
  }
  if (needs_reshape) {
    shaped = poly_reshape(ctx, value, aligned, like_ndim);
    if (!shaped) return NULL;
  }
  if (optim_same_shape(ctx, shaped, like)) return shaped;
  return poly_expand(ctx, shaped, (int64_t *)like_dims, like_ndim);
}

static PolyUOp *optim_float_like(PolyCtx *ctx, double value, PolyUOp *like) {
  if (!ctx || !like) return NULL;
  PolyDType dtype = poly_dtype_scalar(like->dtype);
  if (!poly_dtype_is_float(dtype)) return NULL;
  return optim_broadcast_like(ctx, poly_const_typed(ctx, dtype, value), like);
}

static PolyUOp *optim_lr_for_param(PolyCtx *ctx, PolyUOp *lr, PolyUOp *param) {
  lr = optim_lr_uop(ctx, lr);
  return (lr && param) ? optim_broadcast_like(ctx, lr, param) : NULL;
}

static PolyUOp *optim_reshape_like(PolyCtx *ctx, PolyUOp *value, PolyUOp *like) {
  if (!ctx || !value || !like) return NULL;
  if (optim_same_shape(ctx, value, like)) return value;
  int ndim = poly_uop_ndim(ctx, like);
  if (ndim < 0) return value;
  const int64_t *dims = poly_uop_max_shape_dims(ctx, like);
  return poly_reshape(ctx, value, (int64_t *)dims, ndim);
}

/* Pinned ElementwiseMixin.sub (mixin/elementwise.py:90-109) is
 * `a + (-b)`, and negation is `b * -1` (elementwise.py:57-67). */
static PolyUOp *optim_sub(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  if (!ctx || !a || !b) return NULL;
  PolyUOp *minus_one = optim_float_like(ctx, -1.0, b);
  PolyUOp *negative = minus_one ? poly_alu2(ctx, POLY_OP_MUL, b, minus_one) : NULL;
  return negative ? poly_alu2(ctx, POLY_OP_ADD, a, negative) : NULL;
}

/* Pinned ElementwiseMixin.div performs `a * b.reciprocal()` after
 * broadcasting (mixin/elementwise.py:206-231). */
static PolyUOp *optim_div(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  if (!ctx || !a || !b) return NULL;
  PolyUOp *reciprocal = poly_alu1(ctx, POLY_OP_RECIPROCAL, b);
  return reciprocal ? poly_alu2(ctx, POLY_OP_MUL, a, reciprocal) : NULL;
}

static PolyUOp *optim_assign_uop_target(PolyCtx *ctx, PolyUOp *target_uop, PolyUOp *value) {
  if (!ctx || !target_uop || !value) return NULL;
  if (!poly_dtype_eq(poly_dtype_scalar(target_uop->dtype), poly_dtype_scalar(value->dtype)))
    return NULL;

  /* Build the same tinygrad effect shape as Tensor.assign without mutating the
   * target yet: AFTER(target, STORE(target, next_value)). */
  PolyUOp *store = poly_store_val(ctx, target_uop, value);
  if (!store) return NULL;
  PolyUOp *src[2] = {target_uop, store};
  PolyUOp *after = poly_uop(ctx, POLY_OP_AFTER, target_uop->dtype, src, 2, poly_arg_none());
  return after;
}

static PolyUOp *optim_assign_uop(PolyCtx *ctx, PolyTensor *target, PolyUOp *value) {
  return target ? optim_assign_uop_target(ctx, poly_tensor_uop(target), value) : NULL;
}

int poly_optim_build_update(
    PolyCtx *ctx,
    const PolyOptimConfig *cfg,
    PolyUOp *lr,
    PolyUOp *param,
    PolyUOp *grad,
    PolyUOp *m_buf,
    PolyUOp *v_buf,
    PolyUOp *bc1_buf,
    PolyUOp *bc2_buf,
    int64_t numel,
    PolyOptimUpdate *out
) {
  if (!ctx || !cfg || !param || !grad || !out || numel <= 0)
    return -1;
  memset(out, 0, sizeof(*out));

  PolyUOp *param_value = param;
  PolyUOp *grad_value = optim_reshape_like(ctx, grad, param);
  PolyUOp *lr_value = optim_lr_uop(ctx, lr);
  if (!grad_value || !lr_value) return -1;

  switch (cfg->kind) {
  case POLY_OPTIM_SGD: {
    PolyUOp *base = poly_detach(ctx, param_value);
    PolyUOp *g = grad_value;
    PolyUOp *lr_param = optim_lr_for_param(ctx, lr_value, param_value);
    if (!base || !lr_param) return -1;
    if (cfg->weight_decay > 0.0f) {
      PolyUOp *wd = optim_float_like(ctx, cfg->weight_decay, base);
      g = poly_alu2(ctx, POLY_OP_ADD, g, poly_alu2(ctx, POLY_OP_MUL, wd, base));
    }
    if (cfg->classic) {
      PolyUOp *r = optim_float_like(ctx, 1.0, g);
      g = r ? poly_alu2(ctx, POLY_OP_MUL, g, r) : NULL;
      g = g ? poly_alu2(ctx, POLY_OP_MUL, g, lr_param) : NULL;
    }
    if (cfg->momentum > 0.0f) {
      if (!m_buf) return -1;
      PolyUOp *m_value_root = optim_reshape_like(ctx, m_buf, param_value);
      PolyUOp *mom = optim_float_like(ctx, cfg->momentum, m_value_root);
      if (!m_value_root || !mom) return -1;
      PolyUOp *m_value =
          poly_alu2(ctx, POLY_OP_ADD, poly_alu2(ctx, POLY_OP_MUL, mom, m_value_root), g);
      out->m_new = optim_assign_uop_target(ctx, m_buf, m_value);
      PolyUOp *m_current = out->m_new;
      if (!m_value || !out->m_new || !m_current) return -1;
      g = cfg->nesterov ? poly_alu2(ctx, POLY_OP_ADD, g, poly_alu2(ctx, POLY_OP_MUL, mom, m_current))
                        : m_current;
    }
    if (!cfg->classic) {
      PolyUOp *r = optim_float_like(ctx, 1.0, g);
      g = r ? poly_alu2(ctx, POLY_OP_MUL, g, r) : NULL;
      g = g ? poly_alu2(ctx, POLY_OP_MUL, g, lr_param) : NULL;
    }
    out->param_new = optim_sub(ctx, base, g);
    return (out->param_new && (cfg->momentum <= 0.0f || out->m_new)) ? 0 : -1;
  }
  case POLY_OPTIM_ADAM:
  case POLY_OPTIM_ADAMW: {
    if (!m_buf || !v_buf || !bc1_buf || !bc2_buf) return -1;
    PolyUOp *m_value_root = optim_reshape_like(ctx, m_buf, param_value);
    PolyUOp *v_value_root = optim_reshape_like(ctx, v_buf, param_value);
    if (!m_value_root || !v_value_root) return -1;

    PolyUOp *b1 = optim_float_like(ctx, cfg->beta1, m_value_root);
    PolyUOp *b2 = optim_float_like(ctx, cfg->beta2, v_value_root);
    PolyUOp *one_minus_b1 = optim_float_like(ctx, 1.0 - cfg->beta1, grad_value);
    PolyUOp *one_minus_b2 = optim_float_like(ctx, 1.0 - cfg->beta2, grad_value);
    PolyUOp *bc1_scale = optim_float_like(ctx, cfg->beta1, bc1_buf);
    PolyUOp *bc2_scale = optim_float_like(ctx, cfg->beta2, bc2_buf);
    if (!b1 || !b2 || !one_minus_b1 || !one_minus_b2 || !bc1_scale || !bc2_scale)
      return -1;

    PolyUOp *m_value = poly_alu2(
        ctx, POLY_OP_ADD, poly_alu2(ctx, POLY_OP_MUL, b1, m_value_root),
        poly_alu2(ctx, POLY_OP_MUL, one_minus_b1, grad_value)
    );
    PolyUOp *g_sq = poly_alu2(ctx, POLY_OP_MUL, grad_value, grad_value);
    PolyUOp *v_value = poly_alu2(
        ctx, POLY_OP_ADD, poly_alu2(ctx, POLY_OP_MUL, b2, v_value_root),
        poly_alu2(ctx, POLY_OP_MUL, one_minus_b2, g_sq)
    );

    PolyUOp *bc1_value = poly_alu2(ctx, POLY_OP_MUL, bc1_buf, bc1_scale);
    PolyUOp *bc2_value = poly_alu2(ctx, POLY_OP_MUL, bc2_buf, bc2_scale);
    m_value = optim_reshape_like(ctx, m_value, m_buf);
    v_value = optim_reshape_like(ctx, v_value, v_buf);
    bc1_value = optim_reshape_like(ctx, bc1_value, bc1_buf);
    bc2_value = optim_reshape_like(ctx, bc2_value, bc2_buf);
    out->m_new = optim_assign_uop_target(ctx, m_buf, m_value);
    out->v_new = optim_assign_uop_target(ctx, v_buf, v_value);
    out->bc1_new = optim_assign_uop_target(ctx, bc1_buf, bc1_value);
    out->bc2_new = optim_assign_uop_target(ctx, bc2_buf, bc2_value);
    PolyUOp *m_current = out->m_new;
    PolyUOp *v_current = out->v_new;
    if (!m_value || !v_value || !bc1_value || !bc2_value || !out->m_new || !out->v_new ||
        !out->bc1_new || !out->bc2_new || !m_current || !v_current)
      return -1;

    PolyUOp *one_bc1 = optim_float_like(ctx, 1.0, out->bc1_new);
    PolyUOp *one_bc2 = optim_float_like(ctx, 1.0, out->bc2_new);
    PolyUOp *bc1_denom = optim_sub(ctx, one_bc1, out->bc1_new);
    PolyUOp *bc2_denom = optim_sub(ctx, one_bc2, out->bc2_new);
    bc1_denom = optim_broadcast_like(ctx, bc1_denom, m_current);
    bc2_denom = optim_broadcast_like(ctx, bc2_denom, v_current);
    PolyUOp *m_hat = optim_div(ctx, m_current, bc1_denom);
    PolyUOp *v_hat = optim_div(ctx, v_current, bc2_denom);
    PolyUOp *eps = optim_float_like(ctx, cfg->eps, v_hat);
    PolyUOp *denom =
        (v_hat && eps) ? poly_alu2(ctx, POLY_OP_ADD, poly_alu1(ctx, POLY_OP_SQRT, v_hat), eps)
                       : NULL;
    PolyUOp *up = denom ? optim_div(ctx, m_hat, denom) : NULL;
    PolyUOp *base = poly_detach(ctx, param_value);
    PolyUOp *wd = optim_float_like(ctx, cfg->weight_decay, base);
    up = (up && wd) ? poly_alu2(
                          ctx, POLY_OP_ADD, up, poly_alu2(ctx, POLY_OP_MUL, wd, base)
                      )
                    : NULL;
    /* Pinned LAMB returns `self.lr * r * up` even when Adam fixes r=1.0
     * (nn/optim.py:171-178). Keep the shape-[1] LR multiplication before
     * broadcasting that product to the parameter shape. */
    PolyUOp *r = optim_float_like(ctx, 1.0, lr_value);
    PolyUOp *lr_scaled = r ? poly_alu2(ctx, POLY_OP_MUL, lr_value, r) : NULL;
    lr_scaled = lr_scaled ? optim_broadcast_like(ctx, lr_scaled, up) : NULL;
    PolyUOp *step = lr_scaled ? poly_alu2(ctx, POLY_OP_MUL, lr_scaled, up) : NULL;
    out->param_new = optim_sub(ctx, poly_detach(ctx, param_value), step);
    return (out->param_new && out->m_new && out->v_new && out->bc1_new && out->bc2_new) ? 0 : -1;
  }
  default:
    return -1;
  }
}

int poly_optim_build_step(
    PolyCtx *ctx,
    const PolyOptimConfig *cfg,
    PolyTensor *lr,
    PolyTensor **params,
    PolyTensor **grads,
    int n_params,
    PolyTensor **m_tensors,
    PolyTensor **v_tensors,
    PolyTensor *bc1_tensor,
    PolyTensor *bc2_tensor,
    PolyTensor **out_tensors,
    int out_cap
) {
  if (!ctx || !cfg || !lr || !params || !grads || n_params <= 0) return -1;
  PolyUOp *lr_uop = optim_lr_uop(ctx, poly_tensor_uop(lr));
  if (!lr_uop) return -1;
  bool adam = (cfg->kind == POLY_OPTIM_ADAM || cfg->kind == POLY_OPTIM_ADAMW);
  bool sgd_momentum = (cfg->kind == POLY_OPTIM_SGD && cfg->momentum > 0.0f);
  int needed = n_params;
  if (adam) needed += 2 * n_params + 2;
  else if (sgd_momentum) needed += n_params;
  if (!out_tensors || out_cap < needed) return needed;

  if ((adam || sgd_momentum) && !m_tensors) return -1;
  if (adam && (!v_tensors || !bc1_tensor || !bc2_tensor)) return -1;

  PolyUOp *bc1_uop = adam ? poly_tensor_uop(bc1_tensor) : NULL;
  PolyUOp *bc2_uop = adam ? poly_tensor_uop(bc2_tensor) : NULL;
  if (adam && (!bc1_uop || !bc2_uop || !poly_ctx_owns_ptr(ctx, bc1_uop) ||
               !poly_ctx_owns_ptr(ctx, bc2_uop) || optim_uop_numel(ctx, bc1_uop) != 1 ||
               optim_uop_numel(ctx, bc2_uop) != 1 ||
               poly_tensor_device(bc1_tensor) != poly_tensor_device(lr) ||
               poly_tensor_device(bc2_tensor) != poly_tensor_device(lr)))
    return -1;

  /* Validate the complete batch before constructing any assignment effects. */
  for (int i = 0; i < n_params; i++) {
    if (!params[i] || !grads[i] || poly_tensor_device(lr) != poly_tensor_device(params[i]))
      return -1;
    PolyUOp *param_uop = poly_tensor_uop(params[i]);
    PolyUOp *grad_uop = poly_tensor_uop(grads[i]);
    if (!param_uop || !grad_uop || !poly_ctx_owns_ptr(ctx, param_uop) ||
        !poly_ctx_owns_ptr(ctx, grad_uop))
      return -1;
    int64_t numel = optim_uop_numel(ctx, param_uop);
    if (numel <= 0 || optim_uop_numel(ctx, grad_uop) != numel) return -1;

    if (adam || sgd_momentum) {
      if (!m_tensors[i] || poly_tensor_device(m_tensors[i]) != poly_tensor_device(params[i]))
        return -1;
      PolyUOp *m_uop = poly_tensor_uop(m_tensors[i]);
      if (!m_uop || !poly_ctx_owns_ptr(ctx, m_uop) || optim_uop_numel(ctx, m_uop) != numel)
        return -1;
    }
    if (adam) {
      if (!v_tensors[i] || poly_tensor_device(v_tensors[i]) != poly_tensor_device(params[i]))
        return -1;
      PolyUOp *v_uop = poly_tensor_uop(v_tensors[i]);
      if (!v_uop || !poly_ctx_owns_ptr(ctx, v_uop) || optim_uop_numel(ctx, v_uop) != numel)
        return -1;
    }
  }

  PolyTensor **targets = calloc((size_t)needed, sizeof(PolyTensor *));
  PolyUOp **effects = calloc((size_t)needed, sizeof(PolyUOp *));
  if (!targets || !effects) {
    free(targets);
    free(effects);
    return -1;
  }

  int rc = -1;
  int m_base = -1;
  int v_base = -1;
  int param_base = 0;
  if (adam) {
    m_base = 2;
    v_base = 2 + n_params;
    param_base = 2 + 2 * n_params;
  } else if (sgd_momentum) {
    m_base = 0;
    param_base = n_params;
  }

  for (int i = 0; i < n_params; i++) {
    PolyUOp *param_uop = poly_tensor_uop(params[i]);
    PolyUOp *grad_uop = poly_tensor_uop(grads[i]);

    int64_t numel = optim_uop_numel(ctx, param_uop);
    PolyOptimUpdate upd;
    if (poly_optim_build_update(
            ctx, cfg, lr_uop, param_uop, grad_uop,
            (adam || sgd_momentum) ? poly_tensor_uop(m_tensors[i]) : NULL,
            adam ? poly_tensor_uop(v_tensors[i]) : NULL, bc1_uop, bc2_uop, numel, &upd
        ) != 0)
      goto done;

    PolyUOp *param_new = optim_reshape_like(ctx, upd.param_new, param_uop);
    PolyUOp *param_effect = optim_assign_uop(ctx, params[i], param_new);
    if (!param_new || !param_effect) goto done;
    targets[param_base + i] = params[i];
    effects[param_base + i] = param_effect;

    if (sgd_momentum) {
      if (!upd.m_new) goto done;
      targets[m_base + i] = m_tensors[i];
      effects[m_base + i] = upd.m_new;
    }

    if (adam) {
      if (!upd.m_new || !upd.v_new || !upd.bc1_new || !upd.bc2_new) goto done;
      if (i == 0) {
        targets[0] = bc1_tensor;
        effects[0] = upd.bc1_new;
        targets[1] = bc2_tensor;
        effects[1] = upd.bc2_new;
      } else if (effects[0] != upd.bc1_new || effects[1] != upd.bc2_new) {
        goto done;
      }
      targets[m_base + i] = m_tensors[i];
      effects[m_base + i] = upd.m_new;
      targets[v_base + i] = v_tensors[i];
      effects[v_base + i] = upd.v_new;
    }
  }

  for (int i = 0; i < needed; i++)
    if (!targets[i] || !effects[i]) goto done;

  /* No PolyTensor root is changed until the whole batch has validated and all
   * assignment-effect UOps have been constructed successfully. Pinned
   * Optimizer.schedule_step mutates each current Tensor.uop through assign
   * (nn/optim.py:41-57, tensor.py:230-257), so the Tensor boundary stores that exact
   * current-derived AFTER/STORE effect as the physical root. The same pointer
   * remains in the mandatory hybrid logical slot until retained optimizer
   * export is specified. */
  for (int i = 0; i < needed; i++) {
    if (poly_tensor_replace_roots(
            ctx, targets[i], effects[i], effects[i], targets[i]->role, targets[i]->device
        ) != 0)
      goto done;
  }
  memcpy(out_tensors, targets, (size_t)needed * sizeof(PolyTensor *));
  rc = needed;

done:
  free(effects);
  free(targets);
  return rc;
}
