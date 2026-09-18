/* mixin/gradient.h -- Reverse-mode differentiation of UOp graphs. */
#ifndef POLY_MIXIN_GRADIENT_H
#define POLY_MIXIN_GRADIENT_H

#include "../core.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Autograd */
/* Reverse-mode gradient of loss w.r.t. wrt.
 * Returns a UOp expression for d(loss)/d(wrt), or NULL on unsupported path. */
PolyUOp *poly_uop_grad(PolyCtx *ctx, PolyUOp *loss, PolyUOp *wrt);

/* Compute gradients for multiple targets in a single reverse pass.
 * initial_grad: the upstream gradient (NULL = ones_like(loss)).
 * wrts[0..n-1]: target UOps to differentiate w.r.t.
 * out_grads[0..n-1]: receives gradient UOps (zero if no path).
 * Returns 0 on success, -1 on failure. */
int poly_uop_grad_many(
    PolyCtx *ctx,
    PolyUOp *loss,
    PolyUOp *initial_grad,
    PolyUOp **wrts,
    int n,
    PolyUOp **out_grads
);

/* Extended multi-target gradient result. out_present is optional; when
 * supplied, each slot records whether reverse-mode produced a gradient before
 * the public zero-for-no-path fallback. This preserves tinygrad's distinction
 * between an absent/NOOP gradient and a numerically zero gradient. */
int poly_uop_grad_many_ex(
    PolyCtx *ctx,
    PolyUOp *loss,
    PolyUOp *initial_grad,
    PolyUOp **wrts,
    int n,
    PolyUOp **out_grads,
    uint8_t *out_present
);

#ifdef __cplusplus
}
#endif

#endif
