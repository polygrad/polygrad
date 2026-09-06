/*
 * optim.h -- Open optimizer graph helpers.
 *
 * Optimizers are normal tensor graph builders. Model convenience training
 * uses these helpers, and custom loops can use the same update expressions
 * before deciding how to realize/apply them.
 */

#ifndef POLY_OPTIM_H
#define POLY_OPTIM_H

#include "polygrad.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef POLY_OPTIM_NONE
#define POLY_OPTIM_NONE 0
#define POLY_OPTIM_SGD 1
#define POLY_OPTIM_ADAM 2
#define POLY_OPTIM_ADAMW 3
#endif

typedef struct {
  int kind;
  double beta1;
  double beta2;
  double eps;
  double weight_decay;
  double momentum;
  bool nesterov;
  bool classic;
} PolyOptimConfig;

typedef struct {
  PolyUOp *param_new;
  PolyUOp *m_new;
  PolyUOp *v_new;
  PolyUOp *bc1_new;
  PolyUOp *bc2_new;
} PolyOptimUpdate;

/* Build one parameter's tinygrad-style optimizer graph. param_new is the flat
 * next parameter value. State fields are AFTER(target, STORE(target, value))
 * assignment roots, and param_new consumes those roots exactly as tinygrad
 * Optimizer._step does. lr must be an at-least-32-bit floating static scalar
 * or static shape-[1] UOp owned by ctx. Adam/AdamW require m/v and beta-power
 * scalar state; SGD ignores those inputs unless momentum is enabled. */
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
);

/* Tensor-level optimizer step builder shared by frontends and Model-like
 * callers. This is the C-side graph-construction part of tinygrad's
 * Optimizer.schedule_step(), not the engine scheduler. It owns no state:
 * lr, params, grads, and optional optimizer-state tensors are supplied by the
 * caller. lr must be an at-least-32-bit floating static scalar or static
 * shape-[1] Tensor in the same ctx and on the same device as the params. The
 * function validates and builds the complete batch before mutating target
 * tensors into assignment-effect roots, then writes those tensors into
 * out_tensors in tinygrad schedule order (optimizer state, parameters) so
 * callers can realize the whole optimizer step as one batch. If out_tensors is
 * NULL or out_cap is too small, returns the number of output tensors required
 * without mutating anything. */
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
);

#ifdef __cplusplus
}
#endif

#endif /* POLY_OPTIM_H */
