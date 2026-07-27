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

static PolyUOp *optim_lr_for_param(PolyCtx *ctx, PolyUOp *lr, PolyUOp *param) {
  lr = optim_lr_uop(ctx, lr);
  if (!lr || !param) return NULL;
  if (poly_uop_ndim(ctx, lr) == 0) return lr;

  int ndim = poly_uop_ndim(ctx, param);
  if (ndim < 0 || ndim > POLY_MAX_DIMS) return NULL;
  if (ndim == 0) return poly_reshape(ctx, lr, NULL, 0);

  const int64_t *dims = poly_uop_max_shape_dims(ctx, param);
  if (!dims) return NULL;
  PolyUOp *shaped = lr;
  if (ndim > 1) {
    int64_t singleton[POLY_MAX_DIMS];
    for (int i = 0; i < ndim; i++) singleton[i] = 1;
    shaped = poly_reshape(ctx, lr, singleton, ndim);
    if (!shaped) return NULL;
  }
  return poly_expand(ctx, shaped, (int64_t *)dims, ndim);
}

static PolyUOp *optim_flatten(PolyCtx *ctx, PolyUOp *u, int64_t numel) {
  if (!ctx || !u || numel <= 0) return NULL;
  int ndim = poly_uop_ndim(ctx, u);
  const int64_t *dims = poly_uop_max_shape_dims(ctx, u);
  if (ndim == 1 && dims && dims[0] == numel) return u;
  int64_t flat[1] = {numel};
  return poly_reshape(ctx, u, flat, 1);
}

static PolyUOp *optim_reshape_like(PolyCtx *ctx, PolyUOp *value, PolyUOp *like) {
  if (!ctx || !value || !like) return NULL;
  int ndim = poly_uop_ndim(ctx, like);
  if (ndim < 0) return value;
  const int64_t *dims = poly_uop_max_shape_dims(ctx, like);
  return poly_reshape(ctx, value, (int64_t *)dims, ndim);
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

  PolyUOp *param_flat = optim_flatten(ctx, param, numel);
  PolyUOp *grad_flat = optim_flatten(ctx, grad, numel);
  if (!param_flat || !grad_flat || !(lr = optim_lr_for_param(ctx, lr, param_flat))) return -1;

  switch (cfg->kind) {
  case POLY_OPTIM_SGD: {
    PolyUOp *base = poly_detach(ctx, param_flat);
    PolyUOp *g = grad_flat;
    if (cfg->weight_decay > 0.0f) {
      PolyUOp *wd = poly_const_float(ctx, (double)cfg->weight_decay);
      g = poly_alu2(ctx, POLY_OP_ADD, g, poly_alu2(ctx, POLY_OP_MUL, wd, base));
    }
    if (cfg->classic) g = poly_alu2(ctx, POLY_OP_MUL, lr, g);
    if (cfg->momentum > 0.0f) {
      if (!m_buf) return -1;
      PolyUOp *m_flat = optim_flatten(ctx, m_buf, numel);
      if (!m_flat) return -1;
      PolyUOp *mom = poly_const_float(ctx, (double)cfg->momentum);
      PolyUOp *m_value =
          poly_alu2(ctx, POLY_OP_ADD, poly_alu2(ctx, POLY_OP_MUL, mom, m_flat), g);
      m_value = optim_reshape_like(ctx, m_value, m_buf);
      out->m_new = optim_assign_uop_target(ctx, m_buf, m_value);
      PolyUOp *m_current = optim_flatten(ctx, out->m_new, numel);
      if (!m_value || !out->m_new || !m_current) return -1;
      g = cfg->nesterov ? poly_alu2(ctx, POLY_OP_ADD, g, poly_alu2(ctx, POLY_OP_MUL, mom, m_current))
                        : m_current;
    }
    if (!cfg->classic) g = poly_alu2(ctx, POLY_OP_MUL, lr, g);
    out->param_new = poly_alu2(ctx, POLY_OP_SUB, base, g);
    return (out->param_new && (cfg->momentum <= 0.0f || out->m_new)) ? 0 : -1;
  }
  case POLY_OPTIM_ADAM:
  case POLY_OPTIM_ADAMW: {
    if (!m_buf || !v_buf || !bc1_buf || !bc2_buf) return -1;
    PolyUOp *m_flat = optim_flatten(ctx, m_buf, numel);
    PolyUOp *v_flat = optim_flatten(ctx, v_buf, numel);
    if (!m_flat || !v_flat) return -1;

    PolyUOp *b1 = poly_const_float(ctx, (double)cfg->beta1);
    PolyUOp *b2 = poly_const_float(ctx, (double)cfg->beta2);
    PolyUOp *one_minus_b1 = poly_const_float(ctx, 1.0 - (double)cfg->beta1);
    PolyUOp *one_minus_b2 = poly_const_float(ctx, 1.0 - (double)cfg->beta2);
    PolyUOp *eps = poly_const_float(ctx, (double)cfg->eps);
    PolyUOp *one = poly_const_float(ctx, 1.0);

    PolyUOp *m_value = poly_alu2(
        ctx, POLY_OP_ADD, poly_alu2(ctx, POLY_OP_MUL, b1, m_flat),
        poly_alu2(ctx, POLY_OP_MUL, one_minus_b1, grad_flat)
    );
    PolyUOp *g_sq = poly_alu2(ctx, POLY_OP_MUL, grad_flat, grad_flat);
    PolyUOp *v_value = poly_alu2(
        ctx, POLY_OP_ADD, poly_alu2(ctx, POLY_OP_MUL, b2, v_flat),
        poly_alu2(ctx, POLY_OP_MUL, one_minus_b2, g_sq)
    );

    PolyUOp *bc1_value = poly_alu2(ctx, POLY_OP_MUL, bc1_buf, b1);
    PolyUOp *bc2_value = poly_alu2(ctx, POLY_OP_MUL, bc2_buf, b2);
    m_value = optim_reshape_like(ctx, m_value, m_buf);
    v_value = optim_reshape_like(ctx, v_value, v_buf);
    bc1_value = optim_reshape_like(ctx, bc1_value, bc1_buf);
    bc2_value = optim_reshape_like(ctx, bc2_value, bc2_buf);
    out->m_new = optim_assign_uop_target(ctx, m_buf, m_value);
    out->v_new = optim_assign_uop_target(ctx, v_buf, v_value);
    out->bc1_new = optim_assign_uop_target(ctx, bc1_buf, bc1_value);
    out->bc2_new = optim_assign_uop_target(ctx, bc2_buf, bc2_value);
    PolyUOp *m_current = optim_flatten(ctx, out->m_new, numel);
    PolyUOp *v_current = optim_flatten(ctx, out->v_new, numel);
    if (!m_value || !v_value || !bc1_value || !bc2_value || !out->m_new || !out->v_new ||
        !out->bc1_new || !out->bc2_new || !m_current || !v_current)
      return -1;

    PolyUOp *bc1_corr = poly_alu1(
        ctx, POLY_OP_RECIPROCAL, poly_alu2(ctx, POLY_OP_SUB, one, out->bc1_new)
    );
    PolyUOp *bc2_corr = poly_alu1(
        ctx, POLY_OP_RECIPROCAL, poly_alu2(ctx, POLY_OP_SUB, one, out->bc2_new)
    );
    int64_t param_shape[1] = {numel};
    bc1_corr = poly_expand(ctx, bc1_corr, param_shape, 1);
    bc2_corr = poly_expand(ctx, bc2_corr, param_shape, 1);
    if (!bc1_corr || !bc2_corr) return -1;

    PolyUOp *m_hat = poly_alu2(ctx, POLY_OP_MUL, m_current, bc1_corr);
    PolyUOp *v_hat = poly_alu2(ctx, POLY_OP_MUL, v_current, bc2_corr);
    PolyUOp *denom = poly_alu2(ctx, POLY_OP_ADD, poly_alu1(ctx, POLY_OP_SQRT, v_hat), eps);
    PolyUOp *up = poly_alu2(ctx, POLY_OP_MUL, m_hat, poly_alu1(ctx, POLY_OP_RECIPROCAL, denom));
    if (cfg->kind == POLY_OPTIM_ADAMW && cfg->weight_decay > 0.0f) {
      PolyUOp *wd = poly_const_float(ctx, (double)cfg->weight_decay);
      up = poly_alu2(
          ctx, POLY_OP_ADD, up, poly_alu2(ctx, POLY_OP_MUL, wd, poly_detach(ctx, param_flat))
      );
    }
    PolyUOp *step = poly_alu2(ctx, POLY_OP_MUL, lr, up);
    out->param_new = poly_alu2(ctx, POLY_OP_SUB, poly_detach(ctx, param_flat), step);
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
   * assignment-effect UOps have been constructed successfully. Match
   * tinygrad Optimizer.schedule_step ordering: optimizer state, parameters. */
  for (int i = 0; i < needed; i++) {
    if (poly_tensor_replace_roots(
            ctx, targets[i], effects[i], NULL, targets[i]->role, targets[i]->device
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
