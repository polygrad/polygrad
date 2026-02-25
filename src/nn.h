/*
 * nn.h — Tensor + neural network API for polygrad
 *
 * High-level C API that wraps polygrad's UOp compiler core into a
 * usable neural network interface with Tensor, layers, and optimizers.
 *
 * All operations are lazy — they build UOp graphs without computation.
 * Call poly_tensor_realize() or poly_sgd_step() to execute.
 */

#ifndef POLY_NN_H
#define POLY_NN_H

#include "polygrad.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define POLY_NN_MAX_DIMS 8

/* ── Tensor ──────────────────────────────────────────────────────────── */

typedef struct PolyTensor {
    PolyCtx *ctx;                        /* UOp context (NULL for persistent params) */
    PolyUOp *uop;                        /* lazy tensor-level UOp (NULL before wrap) */
    int64_t shape[POLY_NN_MAX_DIMS];
    int ndim;
    PolyDType dtype;
    float *data;                        /* host data (NULL if not realized) */
    int64_t numel;
    bool requires_grad;
    bool owns_data;                     /* true → free(data) on tensor_free */
} PolyTensor;

/* ── Creation ────────────────────────────────────────────────────────── */

/* Set the nn RNG seed (for reproducible weight initialization) */
void poly_nn_seed(uint32_t seed);

/* Persistent tensors (params): ctx=NULL, data allocated, owns_data=true */
PolyTensor *poly_tensor_zeros(int64_t *shape, int ndim);
PolyTensor *poly_tensor_ones(int64_t *shape, int ndim);
PolyTensor *poly_tensor_rand(int64_t *shape, int ndim);
PolyTensor *poly_tensor_from_data(const float *data, int64_t *shape, int ndim);

/* Wrap external data as a lazy tensor in a UOp context (does not own data) */
PolyTensor *poly_tensor_input(PolyCtx *ctx, float *data, int64_t *shape, int ndim);

/* Wrap a persistent tensor into a step's UOp context (creates BUFFER + RESHAPE) */
PolyTensor *poly_tensor_wrap(PolyCtx *ctx, PolyTensor *persistent);

void poly_tensor_free(PolyTensor *t);

/* ── Elementwise ─────────────────────────────────────────────────────── */

PolyTensor *poly_tensor_add(PolyTensor *a, PolyTensor *b);
PolyTensor *poly_tensor_sub(PolyTensor *a, PolyTensor *b);
PolyTensor *poly_tensor_mul(PolyTensor *a, PolyTensor *b);
PolyTensor *poly_tensor_div(PolyTensor *a, PolyTensor *b);
PolyTensor *poly_tensor_neg(PolyTensor *t);

/* ── Activations ─────────────────────────────────────────────────────── */

PolyTensor *poly_tensor_relu(PolyTensor *t);
PolyTensor *poly_tensor_sigmoid(PolyTensor *t);

/* ── Movement ────────────────────────────────────────────────────────── */

PolyTensor *poly_tensor_reshape(PolyTensor *t, int64_t *shape, int ndim);
PolyTensor *poly_tensor_permute(PolyTensor *t, int64_t *perm, int ndim);
PolyTensor *poly_tensor_expand(PolyTensor *t, int64_t *shape, int ndim);
PolyTensor *poly_tensor_transpose(PolyTensor *t);

/* ── Reductions ──────────────────────────────────────────────────────── */

PolyTensor *poly_tensor_sum(PolyTensor *t, int axis);   /* axis=-1 for all */
PolyTensor *poly_tensor_max_reduce(PolyTensor *t, int axis);
PolyTensor *poly_tensor_mean(PolyTensor *t, int axis);

/* ── Matrix ops ──────────────────────────────────────────────────────── */

PolyTensor *poly_tensor_matmul(PolyTensor *a, PolyTensor *b);
PolyTensor *poly_tensor_linear(PolyTensor *x, PolyTensor *w, PolyTensor *bias);

/* ── Loss functions ──────────────────────────────────────────────────── */

PolyTensor *poly_tensor_mse(PolyTensor *pred, PolyTensor *target);

/* ── Realization ─────────────────────────────────────────────────────── */

int poly_tensor_realize(PolyTensor *t);
float poly_tensor_item(PolyTensor *t);

/* ── Autograd ────────────────────────────────────────────────────────── */

/* Compute gradients of loss w.r.t. each param.
 * Returns malloc'd array of PolyTensor* (one per param, caller frees array + tensors). */
PolyTensor **poly_tensor_backward(PolyCtx *ctx, PolyTensor *loss,
                                PolyTensor **params, int n_params);

/* ── nn Layers ───────────────────────────────────────────────────────── */

typedef struct {
    PolyTensor *weight;    /* (out_features, in_features) */
    PolyTensor *bias;      /* (out_features,) or NULL */
    int in_features;
    int out_features;
} PolyLinear;

PolyLinear *poly_nn_linear(int in_features, int out_features, bool use_bias);
PolyTensor *poly_nn_linear_forward(PolyLinear *l, PolyCtx *ctx, PolyTensor *x);
int poly_nn_linear_params(PolyLinear *l, PolyTensor **out, int max);
void poly_nn_linear_free(PolyLinear *l);

/* ── Optimizers ──────────────────────────────────────────────────────── */

typedef struct {
    PolyTensor **params;
    int n_params;
    float lr;
} PolySGD;

PolySGD poly_sgd_new(PolyTensor **params, int n_params, float lr);
int poly_sgd_step(PolySGD *opt, PolyCtx *ctx, PolyTensor *loss, float *loss_out);

#ifdef __cplusplus
}
#endif

#endif /* POLY_NN_H */
