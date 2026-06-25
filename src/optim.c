/*
 * optim.c -- Open optimizer graph helpers.
 */

#include "optim.h"
#include "tensor.h"
#include <string.h>

static int64_t optim_uop_numel(PolyCtx *ctx, PolyUOp *u) {
  if (!ctx || !u) return -1;
  int ndim = poly_uop_ndim(ctx, u);
  if (ndim < 0) return -1;
  if (ndim == 0) return 1;
  const int64_t *dims = poly_uop_max_shape_dims(ctx, u);
  return poly_shape_numel_checked(dims, ndim);
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

static PolyTensor *optim_assign_uop(PolyCtx *ctx, PolyTensor *target, PolyUOp *value) {
  if (!ctx || !target || !value) return NULL;
  PolyUOp *target_uop = poly_tensor_uop(target);
  if (!target_uop) return NULL;
  if (!poly_dtype_eq(poly_dtype_scalar(target_uop->dtype), poly_dtype_scalar(value->dtype)))
    return NULL;

  /* Optimizers build the same tinygrad effect shape as Tensor.assign:
   * AFTER(target, STORE(target, next_value)). This helper avoids creating a
   * throwaway PolyTensor just to pass the computed value through
   * poly_tensor_assign(). */
  PolyUOp *store = poly_store_val(ctx, target_uop, value);
  if (!store) return NULL;
  PolyUOp *src[2] = {target_uop, store};
  PolyUOp *after = poly_uop(ctx, POLY_OP_AFTER, target_uop->dtype, src, 2, poly_arg_none());
  if (!after) return NULL;
  if (poly_tensor_update(ctx, target, after, NULL, target->role, target->device) != 0) return NULL;
  return target;
}

int poly_optim_build_update(
    PolyCtx *ctx,
    const PolyOptimConfig *cfg,
    PolyUOp *param,
    PolyUOp *grad,
    PolyUOp *m_buf,
    PolyUOp *v_buf,
    PolyUOp *bc1_buf,
    PolyUOp *bc2_buf,
    int64_t numel,
    PolyOptimUpdate *out
) {
  if (!ctx || !cfg || !param || !grad || !out) return -1;
  memset(out, 0, sizeof(*out));

  PolyUOp *lr = poly_const_float(ctx, (double)cfg->lr);
  switch (cfg->kind) {
  case POLY_OPTIM_SGD: {
    PolyUOp *base = poly_detach(ctx, param);
    PolyUOp *g = grad;
    if (cfg->weight_decay > 0.0f) {
      PolyUOp *wd = poly_const_float(ctx, (double)cfg->weight_decay);
      g = poly_alu2(ctx, POLY_OP_ADD, g, poly_alu2(ctx, POLY_OP_MUL, wd, base));
    }
    if (cfg->classic) g = poly_alu2(ctx, POLY_OP_MUL, lr, g);
    if (cfg->momentum > 0.0f) {
      if (!m_buf) return -1;
      PolyUOp *mom = poly_const_float(ctx, (double)cfg->momentum);
      out->m_new = poly_alu2(ctx, POLY_OP_ADD, poly_alu2(ctx, POLY_OP_MUL, mom, m_buf), g);
      g = cfg->nesterov ? poly_alu2(ctx, POLY_OP_ADD, g, poly_alu2(ctx, POLY_OP_MUL, mom, out->m_new))
                        : out->m_new;
    }
    if (!cfg->classic) g = poly_alu2(ctx, POLY_OP_MUL, lr, g);
    out->param_new = poly_alu2(ctx, POLY_OP_SUB, base, g);
    return out->param_new ? 0 : -1;
  }
  case POLY_OPTIM_ADAM:
  case POLY_OPTIM_ADAMW: {
    if (!m_buf || !v_buf || !bc1_buf || !bc2_buf || numel <= 0) return -1;

    PolyUOp *b1 = poly_const_float(ctx, (double)cfg->beta1);
    PolyUOp *b2 = poly_const_float(ctx, (double)cfg->beta2);
    PolyUOp *one_minus_b1 = poly_const_float(ctx, 1.0 - (double)cfg->beta1);
    PolyUOp *one_minus_b2 = poly_const_float(ctx, 1.0 - (double)cfg->beta2);
    PolyUOp *eps = poly_const_float(ctx, (double)cfg->eps);
    PolyUOp *one = poly_const_float(ctx, 1.0);

    out->m_new = poly_alu2(
        ctx, POLY_OP_ADD, poly_alu2(ctx, POLY_OP_MUL, b1, m_buf),
        poly_alu2(ctx, POLY_OP_MUL, one_minus_b1, grad)
    );
    PolyUOp *g_sq = poly_alu2(ctx, POLY_OP_MUL, grad, grad);
    out->v_new = poly_alu2(
        ctx, POLY_OP_ADD, poly_alu2(ctx, POLY_OP_MUL, b2, v_buf),
        poly_alu2(ctx, POLY_OP_MUL, one_minus_b2, g_sq)
    );

    out->bc1_new = poly_alu2(ctx, POLY_OP_MUL, bc1_buf, b1);
    out->bc2_new = poly_alu2(ctx, POLY_OP_MUL, bc2_buf, b2);
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

    PolyUOp *m_hat = poly_alu2(ctx, POLY_OP_MUL, out->m_new, bc1_corr);
    PolyUOp *v_hat = poly_alu2(ctx, POLY_OP_MUL, out->v_new, bc2_corr);
    PolyUOp *denom = poly_alu2(ctx, POLY_OP_ADD, poly_alu1(ctx, POLY_OP_SQRT, v_hat), eps);
    PolyUOp *up = poly_alu2(ctx, POLY_OP_MUL, m_hat, poly_alu1(ctx, POLY_OP_RECIPROCAL, denom));
    if (cfg->kind == POLY_OPTIM_ADAMW && cfg->weight_decay > 0.0f) {
      PolyUOp *wd = poly_const_float(ctx, (double)cfg->weight_decay);
      up = poly_alu2(ctx, POLY_OP_ADD, up, poly_alu2(ctx, POLY_OP_MUL, wd, poly_detach(ctx, param)));
    }
    PolyUOp *step = poly_alu2(ctx, POLY_OP_MUL, lr, up);
    out->param_new = poly_alu2(ctx, POLY_OP_SUB, poly_detach(ctx, param), step);
    return (out->param_new && out->m_new && out->v_new && out->bc1_new && out->bc2_new) ? 0 : -1;
  }
  default:
    return -1;
  }
}

int poly_optim_build_step(
    PolyCtx *ctx,
    const PolyOptimConfig *cfg,
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
  if (!ctx || !cfg || !params || !grads || n_params <= 0) return -1;
  bool adam = (cfg->kind == POLY_OPTIM_ADAM || cfg->kind == POLY_OPTIM_ADAMW);
  bool sgd_momentum = (cfg->kind == POLY_OPTIM_SGD && cfg->momentum > 0.0f);
  int needed = n_params;
  if (adam) needed += 2 * n_params + 2;
  else if (sgd_momentum) needed += n_params;
  if (!out_tensors || out_cap < needed) return needed;

  if ((adam || sgd_momentum) && !m_tensors) return -1;
  if (adam && (!v_tensors || !bc1_tensor || !bc2_tensor)) return -1;

  int out_i = 0;
  PolyUOp *bc1_new = NULL;
  PolyUOp *bc2_new = NULL;

  for (int i = 0; i < n_params; i++) {
    if (!params[i] || !grads[i]) return -1;
    PolyUOp *param_uop = poly_tensor_uop(params[i]);
    PolyUOp *grad_uop = poly_tensor_uop(grads[i]);
    if (!param_uop || !grad_uop) return -1;

    int64_t numel = optim_uop_numel(ctx, param_uop);
    if (numel <= 0) return -1;
    PolyUOp *param_flat = optim_flatten(ctx, param_uop, numel);
    PolyUOp *grad_flat = optim_flatten(ctx, grad_uop, numel);
    if (!param_flat || !grad_flat) return -1;

    PolyUOp *m_flat = NULL;
    PolyUOp *v_flat = NULL;
    PolyUOp *bc1_uop = NULL;
    PolyUOp *bc2_uop = NULL;

    if (adam || sgd_momentum) {
      if (!m_tensors[i]) return -1;
      m_flat = optim_flatten(ctx, poly_tensor_uop(m_tensors[i]), numel);
      if (!m_flat) return -1;
    }
    if (adam) {
      if (!v_tensors[i]) return -1;
      v_flat = optim_flatten(ctx, poly_tensor_uop(v_tensors[i]), numel);
      bc1_uop = poly_tensor_uop(bc1_tensor);
      bc2_uop = poly_tensor_uop(bc2_tensor);
      if (!v_flat || !bc1_uop || !bc2_uop) return -1;
    }

    PolyOptimUpdate upd;
    if (poly_optim_build_update(
            ctx, cfg, param_flat, grad_flat, m_flat, v_flat, bc1_uop, bc2_uop, numel, &upd
        ) != 0)
      return -1;

    PolyUOp *param_new = optim_reshape_like(ctx, upd.param_new, param_uop);
    if (!param_new || !optim_assign_uop(ctx, params[i], param_new)) return -1;
    out_tensors[out_i++] = params[i];

    if (sgd_momentum) {
      PolyUOp *m_new = optim_reshape_like(ctx, upd.m_new, poly_tensor_uop(m_tensors[i]));
      if (!m_new || !optim_assign_uop(ctx, m_tensors[i], m_new)) return -1;
      out_tensors[out_i++] = m_tensors[i];
    }

    if (adam) {
      if (!bc1_new) bc1_new = upd.bc1_new;
      if (!bc2_new) bc2_new = upd.bc2_new;
      PolyUOp *m_new = optim_reshape_like(ctx, upd.m_new, poly_tensor_uop(m_tensors[i]));
      PolyUOp *v_new = optim_reshape_like(ctx, upd.v_new, poly_tensor_uop(v_tensors[i]));
      if (!m_new || !v_new) return -1;
      if (!optim_assign_uop(ctx, m_tensors[i], m_new)) return -1;
      if (!optim_assign_uop(ctx, v_tensors[i], v_new)) return -1;
      out_tensors[out_i++] = m_tensors[i];
      out_tensors[out_i++] = v_tensors[i];
    }
  }

  if (adam) {
    if (!bc1_new || !bc2_new) return -1;
    bc1_new = optim_reshape_like(ctx, bc1_new, poly_tensor_uop(bc1_tensor));
    bc2_new = optim_reshape_like(ctx, bc2_new, poly_tensor_uop(bc2_tensor));
    if (!bc1_new || !bc2_new) return -1;
    if (!optim_assign_uop(ctx, bc1_tensor, bc1_new)) return -1;
    if (!optim_assign_uop(ctx, bc2_tensor, bc2_new)) return -1;
    out_tensors[out_i++] = bc1_tensor;
    out_tensors[out_i++] = bc2_tensor;
  }

  return out_i;
}
