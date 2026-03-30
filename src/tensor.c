/*
 * tensor.c -- Internal shape-tracked expression wrapper
 *
 * PolyExpr pairs a UOp with its logical shape. Thin wrappers around
 * frontend.c and scheduler.c ops that track output shapes automatically.
 *
 * Used by nn.c layers and models/ builders. Not exported via FFI.
 */

#define _POSIX_C_SOURCE 200809L

#include "tensor.h"
#include <string.h>
#include <math.h>
#include <stdio.h>

/* ── Internal helpers ─────────────────────────────────────────────────── */

static PolyExpr make(PolyCtx *ctx, PolyUOp *uop, const int64_t *shape, int ndim, PolyDType dtype) {
  PolyExpr e = { .ctx = ctx, .uop = uop, .buf = NULL, .ndim = ndim, .dtype = dtype };
  if (shape && ndim > 0) memcpy(e.shape, shape, ndim * sizeof(int64_t));
  return e;
}

static PolyExpr fail(void) { return pe_null(); }

/* Copy shape from PolyExpr, return same shape (for same-shape ops) */
static PolyExpr same_shape(PolyExpr x, PolyUOp *uop) {
  return make(x.ctx, uop, x.shape, x.ndim, x.dtype);
}

static PolyExpr same_shape_dt(PolyExpr x, PolyUOp *uop, PolyDType dt) {
  return make(x.ctx, uop, x.shape, x.ndim, dt);
}

/* Broadcast two shapes, return the result shape */
static int broadcast_shapes(const int64_t *a, int an, const int64_t *b, int bn,
                            int64_t *out, int *out_n) {
  int max_n = an > bn ? an : bn;
  for (int i = 0; i < max_n; i++) {
    int64_t da = (i < an) ? a[an - 1 - i] : 1;
    int64_t db = (i < bn) ? b[bn - 1 - i] : 1;
    if (da != db && da != 1 && db != 1) return -1;
    out[max_n - 1 - i] = da > db ? da : db;
  }
  *out_n = max_n;
  return 0;
}

/* ── Creation ─────────────────────────────────────────────────────────── */

PolyExpr pe_buffer(PolyCtx *ctx, PolyDType dt, const int64_t *shape, int ndim) {
  int64_t numel = 1;
  for (int i = 0; i < ndim; i++) numel *= shape[i];
  PolyUOp *raw = poly_buffer(ctx, dt, numel);
  if (!raw) return fail();
  PolyUOp *view = (ndim > 1) ? poly_reshape(ctx, raw, (int64_t *)shape, ndim) : raw;
  PolyExpr e = make(ctx, view, shape, ndim, dt);
  e.buf = raw;  /* keep reference to underlying BUFFER for realize binding */
  return e;
}

PolyExpr pe_const_float(PolyCtx *ctx, double val) {
  PolyUOp *c = poly_const_float(ctx, val);
  int64_t shape[] = {1};
  return make(ctx, c, shape, 1, POLY_FLOAT32);
}

PolyExpr pe_const_int(PolyCtx *ctx, int64_t val) {
  PolyUOp *c = poly_const_int(ctx, val);
  int64_t shape[] = {1};
  return make(ctx, c, shape, 1, POLY_INT32);
}

PolyExpr pe_full(PolyCtx *ctx, const int64_t *shape, int ndim, double val) {
  PolyUOp *u = poly_full(ctx, shape, ndim, val);
  if (!u) return fail();
  return make(ctx, u, shape, ndim, POLY_FLOAT32);
}

PolyExpr pe_zeros(PolyCtx *ctx, const int64_t *shape, int ndim) {
  return pe_full(ctx, shape, ndim, 0.0);
}

PolyExpr pe_ones(PolyCtx *ctx, const int64_t *shape, int ndim) {
  return pe_full(ctx, shape, ndim, 1.0);
}

PolyExpr pe_rand(PolyCtx *ctx, const int64_t *shape, int ndim, uint64_t seed) {
  PolyUOp *u = poly_rand(ctx, shape, ndim, seed);
  if (!u) return fail();
  return make(ctx, u, shape, ndim, POLY_FLOAT32);
}

PolyExpr pe_arange(PolyCtx *ctx, double start, double stop, double step) {
  PolyUOp *u = poly_arange(ctx, start, stop, step);
  if (!u) return fail();
  int64_t n = (int64_t)ceil((stop - start) / step);
  if (n < 0) n = 0;
  int64_t shape[] = {n};
  return make(ctx, u, shape, 1, POLY_FLOAT32);
}

PolyExpr pe_eye(PolyCtx *ctx, int64_t n) {
  PolyUOp *u = poly_eye(ctx, n);
  if (!u) return fail();
  int64_t shape[] = {n, n};
  return make(ctx, u, shape, 2, POLY_FLOAT32);
}

/* ── Elementwise (binary) ─────────────────────────────────────────────── */

static PolyExpr binop(PolyExpr a, PolyExpr b, PolyOps op) {
  if (!pe_valid(a) || !pe_valid(b)) return fail();
  /* Broadcast shapes */
  int64_t out_shape[PE_MAX_DIMS];
  int out_ndim;
  if (broadcast_shapes(a.shape, a.ndim, b.shape, b.ndim, out_shape, &out_ndim) < 0)
    return fail();
  /* Broadcast UOps to common shape if needed */
  PolyUOp *au = a.uop, *bu = b.uop;
  if (a.ndim != out_ndim || memcmp(a.shape, out_shape, out_ndim * sizeof(int64_t)) != 0) {
    if (a.ndim < out_ndim) au = poly_reshape(a.ctx, au, out_shape, out_ndim);
    au = poly_expand(a.ctx, au, out_shape, out_ndim);
  }
  if (b.ndim != out_ndim || memcmp(b.shape, out_shape, out_ndim * sizeof(int64_t)) != 0) {
    if (b.ndim < out_ndim) bu = poly_reshape(b.ctx, bu, out_shape, out_ndim);
    bu = poly_expand(b.ctx, bu, out_shape, out_ndim);
  }
  PolyUOp *u = poly_alu2(a.ctx, op, au, bu);
  return make(a.ctx, u, out_shape, out_ndim, a.dtype);
}

PolyExpr pe_add(PolyExpr a, PolyExpr b) { return binop(a, b, POLY_OP_ADD); }
PolyExpr pe_sub(PolyExpr a, PolyExpr b) { return binop(a, b, POLY_OP_SUB); }
PolyExpr pe_mul(PolyExpr a, PolyExpr b) { return binop(a, b, POLY_OP_MUL); }
PolyExpr pe_div(PolyExpr a, PolyExpr b) { return binop(a, b, POLY_OP_FDIV); }

PolyExpr pe_maximum(PolyExpr a, PolyExpr b) {
  if (!pe_valid(a) || !pe_valid(b)) return fail();
  PolyUOp *u = poly_maximum(a.ctx, a.uop, b.uop);
  return same_shape(a, u);
}

PolyExpr pe_minimum(PolyExpr a, PolyExpr b) {
  if (!pe_valid(a) || !pe_valid(b)) return fail();
  PolyUOp *u = poly_minimum(a.ctx, a.uop, b.uop);
  return same_shape(a, u);
}

/* ── Elementwise (unary) ──────────────────────────────────────────────── */

static PolyExpr unop(PolyExpr x, PolyOps op) {
  if (!pe_valid(x)) return fail();
  return same_shape(x, poly_alu1(x.ctx, op, x.uop));
}

PolyExpr pe_neg(PolyExpr x)    { return unop(x, POLY_OP_NEG); }
PolyExpr pe_sqrt(PolyExpr x)   { return unop(x, POLY_OP_SQRT); }

PolyExpr pe_exp(PolyExpr x) {
  if (!pe_valid(x)) return fail();
  return same_shape(x, poly_exp(x.ctx, x.uop));
}

PolyExpr pe_log(PolyExpr x) {
  if (!pe_valid(x)) return fail();
  return same_shape(x, poly_log(x.ctx, x.uop));
}

PolyExpr pe_rsqrt(PolyExpr x) {
  if (!pe_valid(x)) return fail();
  return same_shape(x, poly_rsqrt(x.ctx, x.uop));
}

PolyExpr pe_square(PolyExpr x) {
  if (!pe_valid(x)) return fail();
  return same_shape(x, poly_square(x.ctx, x.uop));
}

PolyExpr pe_abs(PolyExpr x) {
  if (!pe_valid(x)) return fail();
  return same_shape(x, poly_abs(x.ctx, x.uop));
}

PolyExpr pe_sign(PolyExpr x) {
  if (!pe_valid(x)) return fail();
  return same_shape(x, poly_sign(x.ctx, x.uop));
}

/* ── Activations ──────────────────────────────────────────────────────── */

PolyExpr pe_relu(PolyExpr x) {
  if (!pe_valid(x)) return fail();
  return same_shape(x, poly_relu(x.ctx, x.uop));
}

PolyExpr pe_gelu(PolyExpr x) {
  if (!pe_valid(x)) return fail();
  return same_shape(x, poly_gelu(x.ctx, x.uop));
}

PolyExpr pe_silu(PolyExpr x) {
  if (!pe_valid(x)) return fail();
  return same_shape(x, poly_silu(x.ctx, x.uop));
}

PolyExpr pe_sigmoid(PolyExpr x) {
  if (!pe_valid(x)) return fail();
  return same_shape(x, poly_sigmoid(x.ctx, x.uop));
}

PolyExpr pe_tanh(PolyExpr x) {
  if (!pe_valid(x)) return fail();
  return same_shape(x, poly_tanh_act(x.ctx, x.uop));
}

/* ── Comparison ───────────────────────────────────────────────────────── */

static PolyExpr cmpop(PolyExpr a, PolyExpr b, PolyUOp *(*fn)(PolyCtx *, PolyUOp *, PolyUOp *)) {
  if (!pe_valid(a) || !pe_valid(b)) return fail();
  PolyUOp *u = fn(a.ctx, a.uop, b.uop);
  return same_shape_dt(a, u, POLY_BOOL);
}

PolyExpr pe_eq(PolyExpr a, PolyExpr b) { return cmpop(a, b, poly_eq); }
PolyExpr pe_ne(PolyExpr a, PolyExpr b) { return cmpop(a, b, poly_ne); }
PolyExpr pe_gt(PolyExpr a, PolyExpr b) { return cmpop(a, b, poly_gt); }
PolyExpr pe_ge(PolyExpr a, PolyExpr b) { return cmpop(a, b, poly_ge); }
PolyExpr pe_le(PolyExpr a, PolyExpr b) { return cmpop(a, b, poly_le); }

PolyExpr pe_lt(PolyExpr a, PolyExpr b) {
  if (!pe_valid(a) || !pe_valid(b)) return fail();
  PolyUOp *u = poly_alu2(a.ctx, POLY_OP_CMPLT, a.uop, b.uop);
  return same_shape_dt(a, u, POLY_BOOL);
}

/* ── Selection ────────────────────────────────────────────────────────── */

PolyExpr pe_where(PolyExpr cond, PolyExpr x, PolyExpr y) {
  if (!pe_valid(cond) || !pe_valid(x) || !pe_valid(y)) return fail();
  PolyUOp *u = poly_where_op(cond.ctx, cond.uop, x.uop, y.uop);
  return same_shape(x, u);
}

PolyExpr pe_clamp(PolyExpr x, double lo, double hi) {
  if (!pe_valid(x)) return fail();
  return same_shape(x, poly_clamp(x.ctx, x.uop, lo, hi));
}

/* ── Scalar broadcast ─────────────────────────────────────────────────── */

PolyExpr pe_add_scalar(PolyExpr x, double val) {
  if (!pe_valid(x)) return fail();
  PolyUOp *c = poly_const_float(x.ctx, val);
  return same_shape(x, poly_alu2(x.ctx, POLY_OP_ADD, x.uop, c));
}

PolyExpr pe_mul_scalar(PolyExpr x, double val) {
  if (!pe_valid(x)) return fail();
  PolyUOp *c = poly_const_float(x.ctx, val);
  return same_shape(x, poly_alu2(x.ctx, POLY_OP_MUL, x.uop, c));
}

/* ── Type casting ─────────────────────────────────────────────────────── */

PolyExpr pe_cast(PolyExpr x, PolyDType target) {
  if (!pe_valid(x)) return fail();
  PolyUOp *u = poly_cast(x.ctx, x.uop, target);
  PolyExpr e = x;
  e.uop = u;
  e.dtype = target;
  return e;
}

/* ── Movement ─────────────────────────────────────────────────────────── */

PolyExpr pe_reshape(PolyExpr x, const int64_t *shape, int ndim) {
  if (!pe_valid(x)) return fail();
  PolyUOp *u = poly_reshape(x.ctx, x.uop, (int64_t *)shape, ndim);
  return make(x.ctx, u, shape, ndim, x.dtype);
}

PolyExpr pe_permute(PolyExpr x, const int64_t *perm, int ndim) {
  if (!pe_valid(x) || ndim != x.ndim) return fail();
  PolyUOp *u = poly_permute(x.ctx, x.uop, (int64_t *)perm, ndim);
  int64_t out_shape[PE_MAX_DIMS];
  for (int i = 0; i < ndim; i++) out_shape[i] = x.shape[perm[i]];
  return make(x.ctx, u, out_shape, ndim, x.dtype);
}

PolyExpr pe_expand(PolyExpr x, const int64_t *shape, int ndim) {
  if (!pe_valid(x)) return fail();
  PolyUOp *u = poly_expand(x.ctx, x.uop, (int64_t *)shape, ndim);
  return make(x.ctx, u, shape, ndim, x.dtype);
}

PolyExpr pe_shrink(PolyExpr x, int64_t (*pairs)[2], int ndim) {
  if (!pe_valid(x) || ndim != x.ndim) return fail();
  PolyUOp *u = poly_shrink(x.ctx, x.uop, pairs, ndim);
  int64_t out_shape[PE_MAX_DIMS];
  for (int i = 0; i < ndim; i++) out_shape[i] = pairs[i][1] - pairs[i][0];
  return make(x.ctx, u, out_shape, ndim, x.dtype);
}

PolyExpr pe_pad(PolyExpr x, int64_t (*pairs)[2], int ndim) {
  if (!pe_valid(x) || ndim != x.ndim) return fail();
  PolyUOp *u = poly_pad(x.ctx, x.uop, pairs, ndim);
  int64_t out_shape[PE_MAX_DIMS];
  for (int i = 0; i < ndim; i++) out_shape[i] = x.shape[i] + pairs[i][0] + pairs[i][1];
  return make(x.ctx, u, out_shape, ndim, x.dtype);
}

PolyExpr pe_flip(PolyExpr x, int64_t *axes, int n_axes) {
  if (!pe_valid(x)) return fail();
  PolyUOp *u = poly_flip(x.ctx, x.uop, axes, n_axes);
  return same_shape(x, u);
}

PolyExpr pe_transpose(PolyExpr x, int dim0, int dim1) {
  if (!pe_valid(x) || dim0 >= x.ndim || dim1 >= x.ndim) return fail();
  if (dim0 < 0) dim0 += x.ndim;
  if (dim1 < 0) dim1 += x.ndim;
  int64_t perm[PE_MAX_DIMS];
  for (int i = 0; i < x.ndim; i++) perm[i] = i;
  perm[dim0] = dim1;
  perm[dim1] = dim0;
  return pe_permute(x, perm, x.ndim);
}

/* ── Reductions ───────────────────────────────────────────────────────── */

PolyExpr pe_sum(PolyExpr x, int axis, int keepdim) {
  if (!pe_valid(x)) return fail();
  int64_t out_shape[PE_MAX_DIMS]; int out_ndim;
  PolyUOp *u = poly_sum_reduce(x.ctx, x.uop, x.shape, x.ndim,
                                axis, keepdim, out_shape, &out_ndim);
  if (!u) return fail();
  return make(x.ctx, u, out_shape, out_ndim, x.dtype);
}

PolyExpr pe_max(PolyExpr x, int axis, int keepdim) {
  if (!pe_valid(x)) return fail();
  int64_t out_shape[PE_MAX_DIMS]; int out_ndim;
  PolyUOp *u = poly_max_reduce(x.ctx, x.uop, x.shape, x.ndim,
                                axis, keepdim, out_shape, &out_ndim);
  if (!u) return fail();
  return make(x.ctx, u, out_shape, out_ndim, x.dtype);
}

PolyExpr pe_mean(PolyExpr x, int axis, int keepdim) {
  if (!pe_valid(x)) return fail();
  int64_t out_shape[PE_MAX_DIMS]; int out_ndim;
  PolyUOp *u = poly_mean_reduce(x.ctx, x.uop, x.shape, x.ndim,
                                 axis, keepdim, out_shape, &out_ndim);
  if (!u) return fail();
  return make(x.ctx, u, out_shape, out_ndim, x.dtype);
}

PolyExpr pe_var(PolyExpr x, int axis, int keepdim, int correction) {
  if (!pe_valid(x)) return fail();
  int64_t out_shape[PE_MAX_DIMS]; int out_ndim;
  PolyUOp *u = poly_var_reduce(x.ctx, x.uop, x.shape, x.ndim,
                                axis, keepdim, correction, out_shape, &out_ndim);
  if (!u) return fail();
  return make(x.ctx, u, out_shape, out_ndim, x.dtype);
}

/* ── Matrix ops ───────────────────────────────────────────────────────── */

PolyExpr pe_dot(PolyExpr a, PolyExpr b) {
  if (!pe_valid(a) || !pe_valid(b)) return fail();
  int64_t out_shape[PE_MAX_DIMS]; int out_ndim;
  PolyUOp *u = poly_dot(a.ctx, a.uop, a.shape, a.ndim,
                         b.uop, b.shape, b.ndim, out_shape, &out_ndim);
  if (!u) return fail();
  return make(a.ctx, u, out_shape, out_ndim, a.dtype);
}

PolyExpr pe_linear(PolyExpr x, PolyExpr weight, PolyExpr *bias) {
  if (!pe_valid(x) || !pe_valid(weight)) return fail();
  int64_t out_shape[PE_MAX_DIMS]; int out_ndim;
  PolyUOp *bu = (bias && pe_valid(*bias)) ? bias->uop : NULL;
  const int64_t *bs = (bias && pe_valid(*bias)) ? bias->shape : NULL;
  int bn = (bias && pe_valid(*bias)) ? bias->ndim : 0;
  PolyUOp *u = poly_linear(x.ctx, x.uop, x.shape, x.ndim,
                            weight.uop, weight.shape, weight.ndim,
                            bu, bs, bn, out_shape, &out_ndim);
  if (!u) return fail();
  return make(x.ctx, u, out_shape, out_ndim, x.dtype);
}

/* ── NN composed ops ──────────────────────────────────────────────────── */

PolyExpr pe_softmax(PolyExpr x, int axis) {
  if (!pe_valid(x)) return fail();
  PolyUOp *u = poly_softmax(x.ctx, x.uop, x.shape, x.ndim, axis);
  if (!u) return fail();
  return same_shape(x, u);
}

PolyExpr pe_log_softmax(PolyExpr x, int axis) {
  if (!pe_valid(x)) return fail();
  PolyUOp *u = poly_log_softmax(x.ctx, x.uop, x.shape, x.ndim, axis);
  if (!u) return fail();
  return same_shape(x, u);
}

PolyExpr pe_layernorm(PolyExpr x, int axis, double eps) {
  if (!pe_valid(x)) return fail();
  int64_t out_shape[PE_MAX_DIMS]; int out_ndim;
  PolyUOp *u = poly_layernorm(x.ctx, x.uop, x.shape, x.ndim, axis, eps,
                               out_shape, &out_ndim);
  if (!u) return fail();
  return make(x.ctx, u, out_shape, out_ndim, x.dtype);
}

PolyExpr pe_cross_entropy(PolyExpr logits, PolyExpr target, int axis) {
  if (!pe_valid(logits) || !pe_valid(target)) return fail();
  int64_t out_shape[PE_MAX_DIMS]; int out_ndim;
  PolyUOp *u = poly_cross_entropy(logits.ctx,
                                   logits.uop, logits.shape, logits.ndim,
                                   target.uop, target.shape, target.ndim,
                                   axis, out_shape, &out_ndim);
  if (!u) return fail();
  return make(logits.ctx, u, out_shape, out_ndim, logits.dtype);
}

PolyExpr pe_gather(PolyExpr table, PolyExpr indices) {
  if (!pe_valid(table) || !pe_valid(indices)) return fail();
  int64_t out_shape[PE_MAX_DIMS]; int out_ndim;
  PolyUOp *u = poly_gather(table.ctx,
                            table.uop, table.shape, table.ndim,
                            indices.uop, indices.shape, indices.ndim,
                            out_shape, &out_ndim);
  if (!u) return fail();
  return make(table.ctx, u, out_shape, out_ndim, table.dtype);
}

/* ── New transformer primitives ───────────────────────────────────────── */

PolyExpr pe_rmsnorm(PolyExpr x, PolyExpr *weight, double eps) {
  if (!pe_valid(x)) return fail();
  /* x * rsqrt(mean(x^2, axis=-1, keepdim=True) + eps) */
  PolyExpr x2 = pe_square(x);
  PolyExpr m = pe_mean(x2, -1, 1);  /* keepdim */
  PolyExpr m_eps = pe_add_scalar(m, eps);
  PolyExpr rrms = pe_rsqrt(m_eps);
  PolyExpr normed = pe_mul(x, rrms);
  if (weight && pe_valid(*weight))
    normed = pe_mul(normed, *weight);
  return normed;
}

PolyExpr pe_scaled_dot_product_attention(PolyExpr q, PolyExpr k, PolyExpr v,
                                         PolyExpr *mask, int is_causal) {
  if (!pe_valid(q) || !pe_valid(k) || !pe_valid(v)) return fail();
  /* q: (..., seq_q, d_k), k: (..., seq_k, d_k), v: (..., seq_k, d_v) */
  int64_t d_k = q.shape[q.ndim - 1];
  double scale = 1.0 / sqrt((double)d_k);

  /* scores = q @ k^T / sqrt(d_k) */
  PolyExpr k_t = pe_transpose(k, -2, -1);
  PolyExpr scores = pe_dot(q, k_t);
  scores = pe_mul_scalar(scores, scale);

  /* Causal mask: upper triangle = -inf */
  if (is_causal) {
    int64_t seq_q = q.shape[q.ndim - 2];
    int64_t seq_k = k.shape[k.ndim - 2];
    /* Create boolean lower-triangular mask */
    PolyExpr ones_mat = pe_ones(q.ctx, (int64_t[]){seq_q, seq_k}, 2);
    /* tril: zero upper triangle */
    PolyUOp *tril_uop = poly_tril(q.ctx, ones_mat.uop,
                                   (int64_t[]){seq_q, seq_k}, 2, 0);
    PolyExpr tril_mask = make(q.ctx, tril_uop, (int64_t[]){seq_q, seq_k}, 2, POLY_FLOAT32);
    /* Convert: 0 -> -inf, 1 -> 0 */
    PolyExpr zero = pe_const_float(q.ctx, 0.0);
    PolyExpr neg_inf = pe_const_float(q.ctx, -1e9);  /* large negative, not actual inf */
    PolyExpr cond = pe_eq(tril_mask, zero);
    PolyExpr causal = pe_where(cond, neg_inf, zero);
    scores = pe_add(scores, causal);
  }

  /* Explicit mask */
  if (mask && pe_valid(*mask))
    scores = pe_add(scores, *mask);

  /* attn_weights = softmax(scores, axis=-1) */
  PolyExpr attn_weights = pe_softmax(scores, -1);

  /* output = attn_weights @ v */
  return pe_dot(attn_weights, v);
}

PolyExpr pe_rope(PolyExpr x, PolyExpr freqs_cos, PolyExpr freqs_sin) {
  if (!pe_valid(x) || !pe_valid(freqs_cos) || !pe_valid(freqs_sin)) return fail();
  /* Vectorized RoPE: split last dim in half, rotate */
  int64_t half_dim = x.shape[x.ndim - 1] / 2;
  if (half_dim <= 0) return fail();

  /* x1 = x[..., :half_dim], x2 = x[..., half_dim:] via shrink */
  int64_t pairs1[PE_MAX_DIMS][2], pairs2[PE_MAX_DIMS][2];
  for (int i = 0; i < x.ndim - 1; i++) {
    pairs1[i][0] = 0; pairs1[i][1] = x.shape[i];
    pairs2[i][0] = 0; pairs2[i][1] = x.shape[i];
  }
  pairs1[x.ndim - 1][0] = 0;         pairs1[x.ndim - 1][1] = half_dim;
  pairs2[x.ndim - 1][0] = half_dim;  pairs2[x.ndim - 1][1] = x.shape[x.ndim - 1];

  PolyExpr x1 = pe_shrink(x, pairs1, x.ndim);
  PolyExpr x2 = pe_shrink(x, pairs2, x.ndim);

  /* out = cat(x1*cos - x2*sin, x2*cos + x1*sin, dim=-1) */
  PolyExpr r1 = pe_sub(pe_mul(x1, freqs_cos), pe_mul(x2, freqs_sin));
  PolyExpr r2 = pe_add(pe_mul(x2, freqs_cos), pe_mul(x1, freqs_sin));

  /* Concatenate along last dim via pad */
  int64_t pad1[PE_MAX_DIMS][2], pad2[PE_MAX_DIMS][2];
  for (int i = 0; i < r1.ndim; i++) {
    pad1[i][0] = 0; pad1[i][1] = 0;
    pad2[i][0] = 0; pad2[i][1] = 0;
  }
  pad1[r1.ndim - 1][1] = half_dim;  /* pad r1 with zeros on the right */
  pad2[r1.ndim - 1][0] = half_dim;  /* pad r2 with zeros on the left */

  PolyExpr r1_padded = pe_pad(r1, pad1, r1.ndim);
  PolyExpr r2_padded = pe_pad(r2, pad2, r2.ndim);
  return pe_add(r1_padded, r2_padded);
}

PolyExpr pe_repeat_interleave(PolyExpr x, int repeats, int dim) {
  if (!pe_valid(x) || repeats <= 0) return fail();
  if (dim < 0) dim += x.ndim;
  if (dim < 0 || dim >= x.ndim) return fail();

  /* Insert dim of size 1 after target: (..., d, 1, ...) */
  int64_t ins_shape[PE_MAX_DIMS];
  int ins_ndim = x.ndim + 1;
  for (int i = 0; i <= dim; i++) ins_shape[i] = x.shape[i];
  ins_shape[dim + 1] = 1;
  for (int i = dim + 1; i < x.ndim; i++) ins_shape[i + 1] = x.shape[i];

  PolyExpr reshaped = pe_reshape(x, ins_shape, ins_ndim);

  /* Expand: (..., d, repeats, ...) */
  int64_t exp_shape[PE_MAX_DIMS];
  memcpy(exp_shape, ins_shape, ins_ndim * sizeof(int64_t));
  exp_shape[dim + 1] = repeats;
  PolyExpr expanded = pe_expand(reshaped, exp_shape, ins_ndim);

  /* Flatten back: (..., d*repeats, ...) */
  int64_t flat_shape[PE_MAX_DIMS];
  int flat_ndim = x.ndim;
  for (int i = 0; i < dim; i++) flat_shape[i] = x.shape[i];
  flat_shape[dim] = x.shape[dim] * repeats;
  for (int i = dim + 1; i < x.ndim; i++) flat_shape[i] = x.shape[i];

  return pe_reshape(expanded, flat_shape, flat_ndim);
}

PolyExpr pe_chunk(PolyExpr x, int n_chunks, int dim, PolyExpr *out_chunks) {
  if (!pe_valid(x) || n_chunks <= 0) return fail();
  if (dim < 0) dim += x.ndim;
  if (dim < 0 || dim >= x.ndim) return fail();

  int64_t total = x.shape[dim];
  int64_t chunk_size = (total + n_chunks - 1) / n_chunks;

  for (int c = 0; c < n_chunks; c++) {
    int64_t start = c * chunk_size;
    int64_t end = start + chunk_size;
    if (end > total) end = total;

    int64_t pairs[PE_MAX_DIMS][2];
    for (int i = 0; i < x.ndim; i++) {
      pairs[i][0] = 0;
      pairs[i][1] = x.shape[i];
    }
    pairs[dim][0] = start;
    pairs[dim][1] = end;

    out_chunks[c] = pe_shrink(x, pairs, x.ndim);
  }
  return out_chunks[0];  /* return first chunk for convenience */
}

/* ── Loss functions ────────────────────────────────────────────────────── */

PolyExpr pe_mse_loss(PolyExpr pred, PolyExpr target) {
  /* mean((pred - target)^2) */
  if (!pe_valid(pred) || !pe_valid(target)) return fail();
  PolyExpr diff = pe_sub(pred, target);
  PolyExpr sq = pe_square(diff);
  /* Reduce all dims */
  PolyExpr r = sq;
  for (int i = r.ndim - 1; i >= 0; i--)
    r = pe_mean(r, i, 0);
  return r;
}

PolyExpr pe_mae_loss(PolyExpr pred, PolyExpr target) {
  /* mean(|pred - target|) */
  if (!pe_valid(pred) || !pe_valid(target)) return fail();
  PolyExpr diff = pe_abs(pe_sub(pred, target));
  PolyExpr r = diff;
  for (int i = r.ndim - 1; i >= 0; i--)
    r = pe_mean(r, i, 0);
  return r;
}

PolyExpr pe_bce_loss(PolyExpr input, PolyExpr target) {
  /* -mean(target * log(input) + (1-target) * log(1-input)) */
  if (!pe_valid(input) || !pe_valid(target)) return fail();
  PolyExpr log_in = pe_log(input);
  PolyExpr one = pe_const_float(input.ctx, 1.0);
  PolyExpr log_1m = pe_log(pe_sub(pe_expand(one, input.shape, input.ndim), input));
  PolyExpr loss = pe_neg(pe_add(pe_mul(target, log_in),
                                pe_mul(pe_sub(pe_expand(one, target.shape, target.ndim), target), log_1m)));
  PolyExpr r = loss;
  for (int i = r.ndim - 1; i >= 0; i--)
    r = pe_mean(r, i, 0);
  return r;
}

/* ── Inference utilities ───────────────────────────────────────────────── */

PolyExpr pe_argmax(PolyExpr x, int axis) {
  /*
   * Port of tinygrad Tensor.argmax:
   *   m = (x == x.max(axis, keepdim=True))
   *   idx = m.float() * arange(N, 0, -1).reshape(...)
   *   return N - idx.max(axis).cast(int32)
   *
   * The descending arange ensures first-occurrence wins on ties.
   */
  if (!pe_valid(x)) return fail();
  if (axis < 0) axis += x.ndim;
  if (axis < 0 || axis >= x.ndim) return fail();

  int64_t N = x.shape[axis];

  /* max with keepdim */
  PolyExpr x_max = pe_max(x, axis, 1);

  /* Broadcast x_max to match x shape for comparison */
  PolyExpr x_max_bc = pe_expand(x_max, x.shape, x.ndim);

  /* m = (x == x_max) as float */
  PolyExpr m = pe_eq(x, x_max_bc);
  PolyExpr m_f = pe_cast(m, POLY_FLOAT32);

  /* Build descending arange: [N, N-1, ..., 1] reshaped for broadcasting */
  /* arange(N, 0, -1) = N - arange(0, N, 1) */
  PolyExpr rng = pe_arange(x.ctx, 0.0, (double)N, 1.0);
  PolyExpr desc = pe_add_scalar(pe_neg(rng), (double)N);  /* N - arange */

  /* Reshape desc to broadcast along the target axis */
  int64_t bc_shape[PE_MAX_DIMS];
  for (int i = 0; i < x.ndim; i++) bc_shape[i] = 1;
  bc_shape[axis] = N;
  PolyExpr desc_r = pe_reshape(desc, bc_shape, x.ndim);
  PolyExpr desc_bc = pe_expand(desc_r, x.shape, x.ndim);

  /* idx = m * desc_arange */
  PolyExpr idx = pe_mul(m_f, desc_bc);

  /* result = N - idx.max(axis) */
  PolyExpr idx_max = pe_max(idx, axis, 0);
  PolyExpr result = pe_add_scalar(pe_neg(idx_max), (double)N);

  /* Cast to int32 */
  return pe_cast(result, POLY_INT32);
}

PolyExpr pe_argmin(PolyExpr x, int axis) {
  /* argmin(x) = argmax(-x) */
  return pe_argmax(pe_neg(x), axis);
}

/* ── Precompute helpers ───────────────────────────────────────────────── */

/* Precompute cos/sin for RoPE: freqs = 1/(theta^(arange(0,dim,2)/dim))
 * table[i,j] = i * freqs[j], return cos(table) or sin(table) */
static PolyExpr precompute_freqs_table(PolyCtx *ctx, int64_t dim, int64_t seq_len,
                                        double theta) {
  /* freqs = 1.0 / (theta ^ (arange(0, dim, 2) / dim)) */
  /* = theta ^ (-arange(0, dim, 2) / dim) */
  /* Compute as constant data */
  int64_t half = dim / 2;
  int64_t numel = seq_len * half;
  float *data = (float *)malloc(numel * sizeof(float));
  if (!data) return fail();

  for (int64_t i = 0; i < seq_len; i++) {
    for (int64_t j = 0; j < half; j++) {
      double freq = 1.0 / pow(theta, (double)(2 * j) / (double)dim);
      data[i * half + j] = (float)((double)i * freq);
    }
  }

  /* Create buffer and bind data */
  PolyUOp *buf = poly_buffer(ctx, POLY_FLOAT32, numel);
  int64_t shape[] = {seq_len, half};
  PolyUOp *view = poly_reshape(ctx, buf, shape, 2);
  PolyExpr e = make(ctx, view, shape, 2, POLY_FLOAT32);

  /* Store the data pointer for later binding */
  /* Note: caller must bind this buffer with the computed data at realize time */
  free(data);  /* For now, return just the expression; data binding is caller's job */
  return e;
}

PolyExpr pe_precompute_freqs_cos(PolyCtx *ctx, int64_t dim, int64_t seq_len, double theta) {
  return precompute_freqs_table(ctx, dim, seq_len, theta);
}

PolyExpr pe_precompute_freqs_sin(PolyCtx *ctx, int64_t dim, int64_t seq_len, double theta) {
  return precompute_freqs_table(ctx, dim, seq_len, theta);
}

PolyExpr pe_causal_mask(PolyCtx *ctx, int64_t T) {
  int64_t out_shape[PE_MAX_DIMS]; int out_ndim;
  PolyUOp *u = poly_causal_mask(ctx, T, out_shape, &out_ndim);
  if (!u) return fail();
  return make(ctx, u, out_shape, out_ndim, POLY_FLOAT32);
}

/* ── Graph construction helpers ───────────────────────────────────────── */

PolyUOp *pe_store(PolyExpr x, PolyExpr buf) {
  if (!pe_valid(x) || !pe_valid(buf)) return NULL;
  return poly_store_val(x.ctx, buf.uop, x.uop);
}

PolyUOp *pe_sink1(PolyUOp *store) {
  if (!store) return NULL;
  /* Need a ctx to call poly_sink1 -- extract from store's sources */
  return NULL;  /* TODO: poly_sink1 takes ctx */
}

PolyExpr pe_assign(PolyExpr target, PolyExpr value) {
  if (!pe_valid(target) || !pe_valid(value)) return fail();
  PolyUOp *u = poly_assign(target.ctx, target.uop, value.uop);
  return same_shape(target, u);
}

PolyExpr pe_detach(PolyExpr x) {
  if (!pe_valid(x)) return fail();
  return same_shape(x, poly_detach(x.ctx, x.uop));
}

/* ═══════════════════════════════════════════════════════════════════════ */
/*  NN Layers                                                             */
/* ═══════════════════════════════════════════════════════════════════════ */

/* SplitMix64 PRNG for weight init (same as model_mlp.c) */
static uint64_t splitmix64(uint64_t *state) {
  uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

static float splitmix64_float(uint64_t *state) {
  return (float)(splitmix64(state) >> 40) / (float)(1ULL << 24);
}

/* Kaiming uniform init: U(-bound, bound) where bound = sqrt(1/in_features) */
static void kaiming_init(float *data, int64_t numel, int in_features, uint64_t *state) {
  float bound = sqrtf(1.0f / (float)in_features);
  for (int64_t i = 0; i < numel; i++)
    data[i] = (2.0f * splitmix64_float(state) - 1.0f) * bound;
}

/* Create a PolyExpr buffer and fill it with data from a host array.
 * The data is embedded as a constant buffer auto-bound at realize time. */
static PolyExpr pe_buffer_with_data(PolyCtx *ctx, PolyDType dt,
                                     const int64_t *shape, int ndim,
                                     const float *data, int64_t numel) {
  /* Use poly_full for each element would be expensive.
   * Instead, create a buffer and rely on the caller to bind data at realize time.
   * For nn layers used in PolyInstance builders, the data gets exported via IR. */
  PolyExpr e = pe_buffer(ctx, dt, shape, ndim);
  /* Mark the data pointer on the underlying buffer for later binding */
  (void)data; (void)numel;  /* data binding happens at Instance level */
  return e;
}

/* ── Linear ───────────────────────────────────────────────────────────── */

PeLinear pe_nn_linear(PolyCtx *ctx, int in_features, int out_features,
                      int use_bias, uint64_t seed) {
  PeLinear l = {0};
  l.in_features = in_features;
  l.out_features = out_features;
  l.has_bias = use_bias;

  int64_t w_shape[] = {out_features, in_features};
  l.weight = pe_buffer(ctx, POLY_FLOAT32, w_shape, 2);

  if (use_bias) {
    int64_t b_shape[] = {out_features};
    l.bias = pe_buffer(ctx, POLY_FLOAT32, b_shape, 1);
  }

  return l;
}

PolyExpr pe_nn_linear_forward(PeLinear *l, PolyExpr x) {
  if (!l || !pe_valid(x) || !pe_valid(l->weight)) return pe_null();
  PolyExpr *bias_ptr = l->has_bias ? &l->bias : NULL;
  return pe_linear(x, l->weight, bias_ptr);
}

/* ── RMSNorm ──────────────────────────────────────────────────────────── */

PeRMSNorm pe_nn_rmsnorm(PolyCtx *ctx, int dim, double eps, uint64_t seed) {
  (void)seed;
  PeRMSNorm l = {0};
  l.dim = dim;
  l.eps = eps;

  int64_t w_shape[] = {dim};
  l.weight = pe_buffer(ctx, POLY_FLOAT32, w_shape, 1);
  /* Weight initialized to ones (caller binds data) */

  return l;
}

PolyExpr pe_nn_rmsnorm_forward(PeRMSNorm *l, PolyExpr x) {
  if (!l || !pe_valid(x)) return pe_null();
  return pe_rmsnorm(x, &l->weight, l->eps);
}

/* ── Embedding ────────────────────────────────────────────────────────── */

PeEmbedding pe_nn_embedding(PolyCtx *ctx, int vocab_size, int embed_dim, uint64_t seed) {
  (void)seed;
  PeEmbedding l = {0};
  l.vocab_size = vocab_size;
  l.embed_dim = embed_dim;

  int64_t w_shape[] = {vocab_size, embed_dim};
  l.weight = pe_buffer(ctx, POLY_FLOAT32, w_shape, 2);

  return l;
}

PolyExpr pe_nn_embedding_forward(PeEmbedding *l, PolyExpr indices) {
  if (!l || !pe_valid(indices)) return pe_null();
  return pe_gather(l->weight, indices);
}

/* ── Dropout ──────────────────────────────────────────────────────────── */

PolyExpr pe_nn_dropout(PolyExpr x, double p, uint64_t seed) {
  if (!pe_valid(x) || p <= 0.0) return x;
  if (p >= 1.0) return pe_mul_scalar(x, 0.0);

  /* mask = (rand(shape) > p) as float, scaled by 1/(1-p) */
  PolyExpr mask = pe_rand(x.ctx, x.shape, x.ndim, seed);
  PolyExpr threshold = pe_const_float(x.ctx, p);
  PolyExpr keep = pe_gt(mask, threshold);
  /* Cast bool to float */
  PolyExpr keep_f = pe_cast(keep, POLY_FLOAT32);
  float scale = 1.0f / (1.0f - (float)p);
  PolyExpr scaled = pe_mul_scalar(keep_f, scale);
  return pe_mul(x, scaled);
}

/* ── Attention ────────────────────────────────────────────────────────── */

PeAttention pe_nn_attention(PolyCtx *ctx, int dim, int n_heads, int n_kv_heads,
                            int use_bias, uint64_t seed) {
  PeAttention a = {0};
  a.dim = dim;
  a.n_heads = n_heads;
  a.n_kv_heads = n_kv_heads > 0 ? n_kv_heads : n_heads;
  a.head_dim = dim / n_heads;

  int kv_dim = a.n_kv_heads * a.head_dim;
  a.wq = pe_nn_linear(ctx, dim, n_heads * a.head_dim, use_bias, seed);
  a.wk = pe_nn_linear(ctx, dim, kv_dim, use_bias, seed + 1);
  a.wv = pe_nn_linear(ctx, dim, kv_dim, use_bias, seed + 2);
  a.wo = pe_nn_linear(ctx, n_heads * a.head_dim, dim, use_bias, seed + 3);

  return a;
}

PolyExpr pe_nn_attention_forward(PeAttention *a, PolyExpr x,
                                  PolyExpr *freqs_cos, PolyExpr *freqs_sin,
                                  PolyExpr *mask, int is_causal) {
  if (!a || !pe_valid(x)) return pe_null();

  int64_t batch = x.shape[0];
  int64_t seq = x.shape[1];
  int hd = a->head_dim;
  int nh = a->n_heads;
  int nkv = a->n_kv_heads;

  /* QKV projections */
  PolyExpr q = pe_nn_linear_forward(&a->wq, x);  /* (B, T, nh*hd) */
  PolyExpr k = pe_nn_linear_forward(&a->wk, x);  /* (B, T, nkv*hd) */
  PolyExpr v = pe_nn_linear_forward(&a->wv, x);  /* (B, T, nkv*hd) */

  /* Reshape to multi-head: (B, T, nh, hd) */
  q = pe_reshape(q, (int64_t[]){batch, seq, nh, hd}, 4);
  k = pe_reshape(k, (int64_t[]){batch, seq, nkv, hd}, 4);
  v = pe_reshape(v, (int64_t[]){batch, seq, nkv, hd}, 4);

  /* Apply RoPE if provided */
  if (freqs_cos && freqs_sin && pe_valid(*freqs_cos) && pe_valid(*freqs_sin)) {
    q = pe_rope(q, *freqs_cos, *freqs_sin);
    k = pe_rope(k, *freqs_cos, *freqs_sin);
  }

  /* GQA: repeat KV heads if needed */
  if (nkv < nh) {
    int n_rep = nh / nkv;
    k = pe_repeat_interleave(k, n_rep, 2);
    v = pe_repeat_interleave(v, n_rep, 2);
  }

  /* Transpose to (B, nh, T, hd) for attention */
  q = pe_transpose(q, 1, 2);
  k = pe_transpose(k, 1, 2);
  v = pe_transpose(v, 1, 2);

  /* Scaled dot-product attention */
  PolyExpr attn = pe_scaled_dot_product_attention(q, k, v, mask, is_causal);

  /* Transpose back and reshape: (B, T, nh*hd) */
  attn = pe_transpose(attn, 1, 2);
  attn = pe_reshape(attn, (int64_t[]){batch, seq, nh * hd}, 3);

  /* Output projection */
  return pe_nn_linear_forward(&a->wo, attn);
}

/* ── Parameter collection ─────────────────────────────────────────────── */

int pe_nn_linear_params(PeLinear *l, PolyExpr *out, int max) {
  int n = 0;
  if (pe_valid(l->weight) && n < max) out[n++] = l->weight;
  if (l->has_bias && pe_valid(l->bias) && n < max) out[n++] = l->bias;
  return n;
}

int pe_nn_rmsnorm_params(PeRMSNorm *l, PolyExpr *out, int max) {
  int n = 0;
  if (pe_valid(l->weight) && n < max) out[n++] = l->weight;
  return n;
}

int pe_nn_embedding_params(PeEmbedding *l, PolyExpr *out, int max) {
  int n = 0;
  if (pe_valid(l->weight) && n < max) out[n++] = l->weight;
  return n;
}

int pe_nn_attention_params(PeAttention *a, PolyExpr *out, int max) {
  int n = 0;
  n += pe_nn_linear_params(&a->wq, out + n, max - n);
  n += pe_nn_linear_params(&a->wk, out + n, max - n);
  n += pe_nn_linear_params(&a->wv, out + n, max - n);
  n += pe_nn_linear_params(&a->wo, out + n, max - n);
  return n;
}
