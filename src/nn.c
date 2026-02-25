/*
 * nn.c — Tensor + neural network implementation
 *
 * Composes polygrad's low-level UOp primitives (sched.h, frontend.h, autograd.c)
 * into a high-level Tensor API with layers and optimizers.
 */

#define _POSIX_C_SOURCE 200809L

#include "nn.h"
#include "sched.h"
#include "frontend.h"
#include "codegen.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/* ── Buffer→data registry ────────────────────────────────────────────── */
/* Maps BUFFER UOps to their host data pointers.
 * Populated by poly_tensor_input() and poly_tensor_wrap().
 * Read by poly_tensor_realize() and poly_sgd_step(). */

#define NN_MAX_BUF_REG 512
static struct { PolyUOp *buf; void *data; } nn_buf_reg[NN_MAX_BUF_REG];
static int nn_buf_n = 0;

static void nn_reg_buf(PolyUOp *buf, void *data) {
  if (nn_buf_n < NN_MAX_BUF_REG) {
    nn_buf_reg[nn_buf_n].buf = buf;
    nn_buf_reg[nn_buf_n].data = data;
    nn_buf_n++;
  }
}

static void *nn_find_buf(PolyUOp *buf) {
  for (int i = 0; i < nn_buf_n; i++)
    if (nn_buf_reg[i].buf == buf) return nn_buf_reg[i].data;
  return NULL;
}

/* ── Helpers ─────────────────────────────────────────────────────────── */

static int64_t compute_numel(const int64_t *shape, int ndim) {
  int64_t n = 1;
  for (int i = 0; i < ndim; i++) n *= shape[i];
  return n;
}

static PolyTensor *make_tensor(PolyCtx *ctx, PolyUOp *uop,
                              const int64_t *shape, int ndim,
                              float *data, bool owns_data) {
  PolyTensor *t = calloc(1, sizeof(PolyTensor));
  t->ctx = ctx;
  t->uop = uop;
  t->ndim = ndim;
  for (int i = 0; i < ndim; i++) t->shape[i] = shape[i];
  t->dtype = POLY_FLOAT32;
  t->numel = compute_numel(shape, ndim);
  t->data = data;
  t->owns_data = owns_data;
  return t;
}

/* Simple LCG random number generator */
static uint32_t nn_rng_state = 12345;

void poly_nn_seed(uint32_t seed) { nn_rng_state = seed; }

static float rand_uniform(void) {
  nn_rng_state = nn_rng_state * 1103515245 + 12345;
  return (float)(nn_rng_state >> 16) / 32768.0f - 1.0f; /* [-1, 1] */
}

/* ── Creation ────────────────────────────────────────────────────────── */

PolyTensor *poly_tensor_zeros(int64_t *shape, int ndim) {
  int64_t n = compute_numel(shape, ndim);
  float *data = calloc(n, sizeof(float));
  return make_tensor(NULL, NULL, shape, ndim, data, true);
}

PolyTensor *poly_tensor_ones(int64_t *shape, int ndim) {
  int64_t n = compute_numel(shape, ndim);
  float *data = malloc(n * sizeof(float));
  for (int64_t i = 0; i < n; i++) data[i] = 1.0f;
  return make_tensor(NULL, NULL, shape, ndim, data, true);
}

PolyTensor *poly_tensor_rand(int64_t *shape, int ndim) {
  int64_t n = compute_numel(shape, ndim);
  float *data = malloc(n * sizeof(float));
  /* Xavier-style initialization: uniform(-scale, scale) */
  float scale = sqrtf(6.0f / (float)n);
  for (int64_t i = 0; i < n; i++)
    data[i] = rand_uniform() * scale;
  return make_tensor(NULL, NULL, shape, ndim, data, true);
}

PolyTensor *poly_tensor_from_data(const float *data, int64_t *shape, int ndim) {
  int64_t n = compute_numel(shape, ndim);
  float *copy = malloc(n * sizeof(float));
  memcpy(copy, data, n * sizeof(float));
  return make_tensor(NULL, NULL, shape, ndim, copy, true);
}

PolyTensor *poly_tensor_input(PolyCtx *ctx, float *data, int64_t *shape, int ndim) {
  int64_t n = compute_numel(shape, ndim);
  PolyUOp *buf = poly_buffer_f32(ctx, n);
  nn_reg_buf(buf, data);
  PolyUOp *uop = buf;
  if (ndim != 1 || shape[0] != n)
    uop = poly_reshape(ctx, buf, shape, ndim);
  return make_tensor(ctx, uop, shape, ndim, data, false);
}

PolyTensor *poly_tensor_wrap(PolyCtx *ctx, PolyTensor *persistent) {
  PolyUOp *buf = poly_buffer_f32(ctx, persistent->numel);
  nn_reg_buf(buf, persistent->data);
  PolyUOp *uop = buf;
  if (persistent->ndim != 1 || persistent->shape[0] != persistent->numel)
    uop = poly_reshape(ctx, buf, persistent->shape, persistent->ndim);
  PolyTensor *t = make_tensor(ctx, uop, persistent->shape, persistent->ndim,
                             persistent->data, false);
  t->requires_grad = persistent->requires_grad;
  return t;
}

void poly_tensor_free(PolyTensor *t) {
  if (!t) return;
  if (t->owns_data) free(t->data);
  free(t);
}

/* ── Elementwise ─────────────────────────────────────────────────────── */

PolyTensor *poly_tensor_add(PolyTensor *a, PolyTensor *b) {
  PolyUOp *r = poly_alu2(a->ctx, POLY_OP_ADD, a->uop, b->uop);
  return make_tensor(a->ctx, r, a->shape, a->ndim, NULL, false);
}

PolyTensor *poly_tensor_sub(PolyTensor *a, PolyTensor *b) {
  PolyUOp *r = poly_alu2(a->ctx, POLY_OP_SUB, a->uop, b->uop);
  return make_tensor(a->ctx, r, a->shape, a->ndim, NULL, false);
}

PolyTensor *poly_tensor_mul(PolyTensor *a, PolyTensor *b) {
  PolyUOp *r = poly_alu2(a->ctx, POLY_OP_MUL, a->uop, b->uop);
  return make_tensor(a->ctx, r, a->shape, a->ndim, NULL, false);
}

PolyTensor *poly_tensor_div(PolyTensor *a, PolyTensor *b) {
  PolyUOp *r = poly_alu2(a->ctx, POLY_OP_FDIV, a->uop, b->uop);
  return make_tensor(a->ctx, r, a->shape, a->ndim, NULL, false);
}

PolyTensor *poly_tensor_neg(PolyTensor *t) {
  PolyUOp *r = poly_alu1(t->ctx, POLY_OP_NEG, t->uop);
  return make_tensor(t->ctx, r, t->shape, t->ndim, NULL, false);
}

/* ── Activations ─────────────────────────────────────────────────────── */

PolyTensor *poly_tensor_relu(PolyTensor *t) {
  return make_tensor(t->ctx, poly_relu(t->ctx, t->uop), t->shape, t->ndim, NULL, false);
}

PolyTensor *poly_tensor_sigmoid(PolyTensor *t) {
  return make_tensor(t->ctx, poly_sigmoid(t->ctx, t->uop), t->shape, t->ndim, NULL, false);
}

/* ── Movement ────────────────────────────────────────────────────────── */

PolyTensor *poly_tensor_reshape(PolyTensor *t, int64_t *shape, int ndim) {
  PolyUOp *r = poly_reshape(t->ctx, t->uop, shape, ndim);
  return make_tensor(t->ctx, r, shape, ndim, NULL, false);
}

PolyTensor *poly_tensor_permute(PolyTensor *t, int64_t *perm, int ndim) {
  PolyUOp *r = poly_permute(t->ctx, t->uop, perm, ndim);
  int64_t new_shape[POLY_NN_MAX_DIMS];
  for (int i = 0; i < ndim; i++) new_shape[i] = t->shape[perm[i]];
  return make_tensor(t->ctx, r, new_shape, ndim, NULL, false);
}

PolyTensor *poly_tensor_expand(PolyTensor *t, int64_t *shape, int ndim) {
  PolyUOp *r = poly_expand(t->ctx, t->uop, shape, ndim);
  return make_tensor(t->ctx, r, shape, ndim, NULL, false);
}

PolyTensor *poly_tensor_transpose(PolyTensor *t) {
  if (t->ndim < 2) return t;
  int64_t perm[POLY_NN_MAX_DIMS];
  for (int i = 0; i < t->ndim; i++) perm[i] = i;
  perm[t->ndim - 2] = t->ndim - 1;
  perm[t->ndim - 1] = t->ndim - 2;
  return poly_tensor_permute(t, perm, t->ndim);
}

/* ── Reductions ──────────────────────────────────────────────────────── */

PolyTensor *poly_tensor_sum(PolyTensor *t, int axis) {
  PolyCtx *ctx = t->ctx;

  if (axis == -1) {
    /* Reduce all axes */
    int64_t all_axes[POLY_NN_MAX_DIMS];
    for (int i = 0; i < t->ndim; i++) all_axes[i] = i;
    PolyUOp *r = poly_reduce_axis(ctx, POLY_OP_ADD, t->uop, all_axes, t->ndim);
    int64_t scalar_shape[] = {1};
    PolyUOp *reshaped = poly_reshape(ctx, r, scalar_shape, 1);
    return make_tensor(ctx, reshaped, scalar_shape, 1, NULL, false);
  }

  if (axis < 0) axis += t->ndim;
  int64_t axes[] = {axis};
  PolyUOp *r = poly_reduce_axis(ctx, POLY_OP_ADD, t->uop, axes, 1);
  /* reduce_axis sets axis dim to 1; reshape to remove it */
  int64_t new_shape[POLY_NN_MAX_DIMS];
  int new_ndim = 0;
  for (int i = 0; i < t->ndim; i++) {
    if (i != axis) new_shape[new_ndim++] = t->shape[i];
  }
  if (new_ndim == 0) { new_shape[0] = 1; new_ndim = 1; }
  PolyUOp *reshaped = poly_reshape(ctx, r, new_shape, new_ndim);
  return make_tensor(ctx, reshaped, new_shape, new_ndim, NULL, false);
}

PolyTensor *poly_tensor_max_reduce(PolyTensor *t, int axis) {
  PolyCtx *ctx = t->ctx;

  if (axis == -1) {
    int64_t all_axes[POLY_NN_MAX_DIMS];
    for (int i = 0; i < t->ndim; i++) all_axes[i] = i;
    PolyUOp *r = poly_reduce_axis(ctx, POLY_OP_MAX, t->uop, all_axes, t->ndim);
    int64_t scalar_shape[] = {1};
    PolyUOp *reshaped = poly_reshape(ctx, r, scalar_shape, 1);
    return make_tensor(ctx, reshaped, scalar_shape, 1, NULL, false);
  }

  if (axis < 0) axis += t->ndim;
  int64_t axes[] = {axis};
  PolyUOp *r = poly_reduce_axis(ctx, POLY_OP_MAX, t->uop, axes, 1);
  int64_t new_shape[POLY_NN_MAX_DIMS];
  int new_ndim = 0;
  for (int i = 0; i < t->ndim; i++) {
    if (i != axis) new_shape[new_ndim++] = t->shape[i];
  }
  if (new_ndim == 0) { new_shape[0] = 1; new_ndim = 1; }
  PolyUOp *reshaped = poly_reshape(ctx, r, new_shape, new_ndim);
  return make_tensor(ctx, reshaped, new_shape, new_ndim, NULL, false);
}

PolyTensor *poly_tensor_mean(PolyTensor *t, int axis) {
  PolyCtx *ctx = t->ctx;
  int64_t count;
  if (axis == -1) {
    count = t->numel;
  } else {
    if (axis < 0) axis += t->ndim;
    count = t->shape[axis];
  }
  PolyTensor *s = poly_tensor_sum(t, axis);
  PolyUOp *divisor = poly_const_float(ctx, (double)count);
  PolyUOp *r = poly_alu2(ctx, POLY_OP_FDIV, s->uop, divisor);
  PolyTensor *result = make_tensor(ctx, r, s->shape, s->ndim, NULL, false);
  free(s); /* free wrapper only, UOp is arena-managed */
  return result;
}

/* ── Matrix ops ──────────────────────────────────────────────────────── */

PolyTensor *poly_tensor_matmul(PolyTensor *a, PolyTensor *b) {
  PolyCtx *ctx = a->ctx;
  int64_t M = a->shape[0], K = a->shape[1], N = b->shape[1];

  /* a: (M, K) → (M, K, 1) → expand (M, K, N) */
  int64_t a3[] = {M, K, 1};
  PolyUOp *a_3d = poly_reshape(ctx, a->uop, a3, 3);
  int64_t a_exp[] = {M, K, N};
  PolyUOp *a_expanded = poly_expand(ctx, a_3d, a_exp, 3);

  /* b: (K, N) → (1, K, N) → expand (M, K, N) */
  int64_t b3[] = {1, K, N};
  PolyUOp *b_3d = poly_reshape(ctx, b->uop, b3, 3);
  int64_t b_exp[] = {M, K, N};
  PolyUOp *b_expanded = poly_expand(ctx, b_3d, b_exp, 3);

  /* multiply elementwise: (M, K, N) */
  PolyUOp *mul = poly_alu2(ctx, POLY_OP_MUL, a_expanded, b_expanded);

  /* reduce over K (axis 1): (M, 1, N) */
  int64_t axes[] = {1};
  PolyUOp *reduced = poly_reduce_axis(ctx, POLY_OP_ADD, mul, axes, 1);

  /* reshape to (M, N) */
  int64_t out_shape[] = {M, N};
  PolyUOp *result = poly_reshape(ctx, reduced, out_shape, 2);

  return make_tensor(ctx, result, out_shape, 2, NULL, false);
}

PolyTensor *poly_tensor_linear(PolyTensor *x, PolyTensor *w, PolyTensor *bias) {
  /* linear(x, w, bias) = x @ w^T + bias
   * x: (batch, in), w: (out, in), bias: (out,) */
  /* Note: poly_tensor_transpose returns t unchanged for ndim < 2, so
   * wt may alias w. Guard frees with wt != w to avoid double-free. */
  PolyTensor *wt = poly_tensor_transpose(w);
  PolyTensor *mm = poly_tensor_matmul(x, wt);
  if (!bias) { if (wt != w) free(wt); return mm; }

  PolyCtx *ctx = x->ctx;
  /* Broadcast bias (out,) → (1, out) → (batch, out) */
  int64_t bias_2d[] = {1, bias->shape[0]};
  PolyUOp *bias_reshaped = poly_reshape(ctx, bias->uop, bias_2d, 2);
  int64_t bias_exp[] = {mm->shape[0], mm->shape[1]};
  PolyUOp *bias_expanded = poly_expand(ctx, bias_reshaped, bias_exp, 2);

  PolyUOp *r = poly_alu2(ctx, POLY_OP_ADD, mm->uop, bias_expanded);
  PolyTensor *result = make_tensor(ctx, r, mm->shape, mm->ndim, NULL, false);
  if (wt != w) free(wt);
  free(mm);
  return result;
}

/* ── Loss functions ──────────────────────────────────────────────────── */

PolyTensor *poly_tensor_mse(PolyTensor *pred, PolyTensor *target) {
  /* MSE = mean((pred - target)^2) */
  PolyTensor *diff = poly_tensor_sub(pred, target);
  PolyTensor *sq = poly_tensor_mul(diff, diff);
  PolyTensor *result = poly_tensor_mean(sq, -1);
  free(diff);
  free(sq);
  return result;
}

/* ── Realization ─────────────────────────────────────────────────────── */

/* Walk graph and collect unique BUFFER UOps */
static int collect_buffers(PolyCtx *ctx, PolyUOp *root,
                           PolyUOp **out, int max) {
  int n;
  PolyUOp **topo = poly_toposort(ctx, root, &n);
  /* topo is arena-allocated — do not free */
  int count = 0;
  for (int i = 0; i < n && count < max; i++) {
    if (topo[i]->op == POLY_OP_BUFFER) {
      bool dup = false;
      for (int j = 0; j < count; j++) {
        if (out[j] == topo[i]) { dup = true; break; }
      }
      if (!dup) out[count++] = topo[i];
    }
  }
  return count;
}

int poly_tensor_realize(PolyTensor *t) {
  if (!t->uop || !t->ctx) return -1;
  PolyCtx *ctx = t->ctx;

  /* Allocate output data. Use calloc (not malloc) so the buffer is
   * zero-initialized — the kernel will overwrite it, but calloc lets the
   * analyzer verify poly_tensor_item() never returns garbage. */
  if (!t->data) {
    t->data = calloc(t->numel, sizeof(float));
    t->owns_data = true;
  }

  /* Create output buffer + store + sink */
  PolyUOp *out_buf = poly_buffer_f32(ctx, t->numel);
  PolyUOp *store = poly_store_val(ctx, out_buf, t->uop);
  PolyUOp *sink = poly_sink1(ctx, store);

  /* Collect all buffers in the graph */
  PolyUOp *all_bufs[256];
  int n_bufs = collect_buffers(ctx, sink, all_bufs, 256);

  /* Build bindings */
  poly_realize_begin(ctx);
  for (int i = 0; i < n_bufs; i++) {
    if (all_bufs[i] == out_buf) {
      poly_realize_bind(ctx, all_bufs[i], t->data);
    } else {
      void *data = nn_find_buf(all_bufs[i]);
      if (!data) {
        fprintf(stderr, "polygrad nn: realize: no data for buffer\n");
        nn_buf_n = 0;
        return -1;
      }
      poly_realize_bind(ctx, all_bufs[i], data);
    }
  }

  int rc = poly_realize_exec(ctx, sink);
  nn_buf_n = 0;
  return rc;
}

float poly_tensor_item(PolyTensor *t) {
  if (t->data && t->numel > 0) return t->data[0];
  poly_tensor_realize(t);
  if (t->data && t->numel > 0) return t->data[0];
  return 0.0f;
}

/* ── Autograd ────────────────────────────────────────────────────────── */

PolyTensor **poly_tensor_backward(PolyCtx *ctx, PolyTensor *loss,
                                PolyTensor **params, int n_params) {
  PolyTensor **grads = malloc(n_params * sizeof(PolyTensor *));
  for (int i = 0; i < n_params; i++) {
    PolyUOp *grad_uop = poly_grad(ctx, loss->uop, params[i]->uop);
    grads[i] = make_tensor(ctx, grad_uop, params[i]->shape, params[i]->ndim,
                           NULL, false);
  }
  return grads;
}

/* ── nn Layers ───────────────────────────────────────────────────────── */

PolyLinear *poly_nn_linear(int in_features, int out_features, bool use_bias) {
  PolyLinear *l = malloc(sizeof(PolyLinear));
  l->in_features = in_features;
  l->out_features = out_features;

  int64_t w_shape[] = {out_features, in_features};
  l->weight = poly_tensor_rand(w_shape, 2);
  l->weight->requires_grad = true;

  if (use_bias) {
    int64_t b_shape[] = {out_features};
    l->bias = poly_tensor_zeros(b_shape, 1);
    l->bias->requires_grad = true;
  } else {
    l->bias = NULL;
  }

  return l;
}

PolyTensor *poly_nn_linear_forward(PolyLinear *l, PolyCtx *ctx, PolyTensor *x) {
  PolyTensor *w = poly_tensor_wrap(ctx, l->weight);
  PolyTensor *b = l->bias ? poly_tensor_wrap(ctx, l->bias) : NULL;
  /* poly_tensor_linear may free w internally (via transpose alias),
   * so use poly_tensor_free which handles NULL and owns_data safely. */
  PolyTensor *result = poly_tensor_linear(x, w, b);
  poly_tensor_free(w);
  poly_tensor_free(b);
  return result;
}

int poly_nn_linear_params(PolyLinear *l, PolyTensor **out, int max) {
  int n = 0;
  if (n < max) out[n++] = l->weight;
  if (l->bias && n < max) out[n++] = l->bias;
  return n;
}

void poly_nn_linear_free(PolyLinear *l) {
  if (!l) return;
  poly_tensor_free(l->weight);
  poly_tensor_free(l->bias);
  free(l);
}

/* ── SGD Optimizer ───────────────────────────────────────────────────── */

PolySGD poly_sgd_new(PolyTensor **params, int n_params, float lr) {
  PolyTensor **copy = malloc(n_params * sizeof(PolyTensor *));
  memcpy(copy, params, n_params * sizeof(PolyTensor *));
  return (PolySGD){ .params = copy, .n_params = n_params, .lr = lr };
}

#define SGD_MAX_PARAMS 64

int poly_sgd_step(PolySGD *opt, PolyCtx *ctx, PolyTensor *loss, float *loss_out) {
  int n = opt->n_params;
  if (n > SGD_MAX_PARAMS) return -1;

  /* 1. Walk loss graph to find param BUFFER UOps via registry.
   *    During forward pass, poly_tensor_wrap() registered each param's
   *    BUFFER→data mapping. We match by data pointer identity. */
  int n_topo;
  PolyUOp **topo = poly_toposort(ctx, loss->uop, &n_topo);
  /* topo is arena-allocated — do not free */

  PolyUOp *param_bufs[SGD_MAX_PARAMS];  /* BUFFER UOps */
  PolyUOp *param_uops[SGD_MAX_PARAMS];  /* RESHAPE or BUFFER UOps (for poly_grad) */

  for (int i = 0; i < n; i++) {
    param_bufs[i] = NULL;
    param_uops[i] = NULL;

    /* Find the BUFFER whose registered data matches this param */
    for (int j = 0; j < n_topo; j++) {
      if (topo[j]->op == POLY_OP_BUFFER) {
        void *data = nn_find_buf(topo[j]);
        if (data == opt->params[i]->data) {
          param_bufs[i] = topo[j];
          break;
        }
      }
    }
    if (!param_bufs[i]) {
      fprintf(stderr, "polygrad nn: sgd_step: param %d not found in loss graph\n", i);
      return -1;
    }

    /* Find the RESHAPE wrapper that matches the persistent param shape.
     * The loss graph may contain other reshapes of the same buffer. */
    for (int j = 0; j < n_topo; j++) {
      if (topo[j]->op != POLY_OP_RESHAPE ||
          topo[j]->n_src <= 0 ||
          topo[j]->src[0] != param_bufs[i]) continue;

      PolyShape rs = poly_uop_shape(ctx, topo[j]);
      bool shape_match = (rs.ndim == opt->params[i]->ndim);
      if (shape_match) {
        for (int d = 0; d < rs.ndim; d++) {
          if (rs.dims[d] != opt->params[i]->shape[d]) {
            shape_match = false;
            break;
          }
        }
      }
      if (rs.ndim > 0 && rs.dims) free(rs.dims);
      if (shape_match) {
        param_uops[i] = topo[j];
        break;
      }
    }
    if (!param_uops[i]) param_uops[i] = param_bufs[i];
  }

  /* 2. Build ALL stores in ONE SINK (tinygrad optimizer.step() pattern).
   *    Scheduling all outputs together lets the rangeify pipeline see
   *    shared forward-pass nodes and auto-partition via BUFFERIZE.
   *    stores[0] = loss, stores[1..n] = gradients */
  int n_stores = 1 + n;
  PolyUOp *stores[1 + SGD_MAX_PARAMS];

  /* Loss store */
  float loss_val = 0.0f;
  PolyUOp *loss_out_buf = poly_buffer_f32(ctx, 1);
  stores[0] = poly_store_val(ctx, loss_out_buf, loss->uop);

  /* Gradient stores */
  float *grad_temps[SGD_MAX_PARAMS];
  PolyUOp *grad_out_bufs[SGD_MAX_PARAMS];

  for (int i = 0; i < n; i++) {
    PolyUOp *grad = poly_grad(ctx, loss->uop, param_uops[i]);
    if (!grad) {
      fprintf(stderr, "polygrad nn: sgd_step: gradient NULL for param %d\n", i);
      for (int g = 0; g < i; g++) free(grad_temps[g]);
      nn_buf_n = 0;
      return -1;
    }
    grad_temps[i] = malloc(opt->params[i]->numel * sizeof(float));
    for (int64_t j = 0; j < opt->params[i]->numel; j++)
      grad_temps[i][j] = -999.0f;  /* sentinel to detect unwritten */
    grad_out_bufs[i] = poly_buffer_f32(ctx, opt->params[i]->numel);
    stores[1 + i] = poly_store_val(ctx, grad_out_bufs[i], grad);

  }

  /* Single SINK with ALL stores */
  PolyUOp *sink = poly_sink_n(ctx, stores, n_stores);

  /* Collect all buffers and bind */
  PolyUOp *all_bufs[256];
  int n_bufs = collect_buffers(ctx, sink, all_bufs, 256);

  poly_realize_begin(ctx);
  for (int i = 0; i < n_bufs; i++) {
    if (all_bufs[i] == loss_out_buf) {
      poly_realize_bind(ctx, all_bufs[i], &loss_val);
      continue;
    }
    bool is_grad_buf = false;
    for (int g = 0; g < n; g++) {
      if (all_bufs[i] == grad_out_bufs[g]) {
        poly_realize_bind(ctx, all_bufs[i], grad_temps[g]);
        is_grad_buf = true;
        break;
      }
    }
    if (is_grad_buf) continue;

    void *data = nn_find_buf(all_bufs[i]);
    if (!data) {
      fprintf(stderr, "polygrad nn: sgd_step: no data for buffer\n");
      for (int g = 0; g < n; g++) free(grad_temps[g]);
      nn_buf_n = 0;
      return -1;
    }
    poly_realize_bind(ctx, all_bufs[i], data);
  }

  int rc = poly_realize_exec(ctx, sink);
  if (rc != 0) {
    fprintf(stderr, "polygrad nn: sgd_step: realize failed\n");
    for (int g = 0; g < n; g++) free(grad_temps[g]);
    nn_buf_n = 0;
    return -1;
  }
  /* 3. CPU-side SGD update */
  for (int i = 0; i < n; i++) {
    /* Guard: param data should always be set after realize, but check
     * defensively since the analyzer can't track through dlopen'd kernels. */
    if (!opt->params[i]->data) {
      fprintf(stderr, "polygrad nn: sgd_step: param[%d] data is NULL\n", i);
      free(grad_temps[i]);
      continue;
    }
    for (int64_t j = 0; j < opt->params[i]->numel; j++)
      opt->params[i]->data[j] -= opt->lr * grad_temps[i][j];
    free(grad_temps[i]);
  }

  if (loss_out) *loss_out = loss_val;

  /* 4. Cleanup */
  nn_buf_n = 0;
  return 0;
}
