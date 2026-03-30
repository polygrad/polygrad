/*
 * tensor.h -- Internal shape-tracked expression wrapper
 *
 * PolyExpr pairs a UOp with its logical shape for use by C nn layers
 * and model builders. NOT a public FFI surface -- Python/JS frontends
 * have their own Tensor classes with language-native semantics.
 *
 * No owned host data, no optimizer, no realize, no autograd state.
 * Just shape tracking so C model code doesn't manually carry shape arrays.
 */

#ifndef POLY_TENSOR_H
#define POLY_TENSOR_H

#include "polygrad.h"
#include "frontend.h"
#include "scheduler.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PE_MAX_DIMS POLY_MAX_DIMS

typedef struct {
  PolyCtx *ctx;
  PolyUOp *uop;          /* the expression (may be reshaped view) */
  PolyUOp *buf;           /* underlying BUFFER UOp if this is a buffer, else NULL */
  int64_t shape[PE_MAX_DIMS];
  int ndim;
  PolyDType dtype;
} PolyExpr;

/* ── Null / validity ──────────────────────────────────────────────────── */

static inline PolyExpr pe_null(void) {
  return (PolyExpr){ .ctx = NULL, .uop = NULL, .ndim = 0 };
}

static inline int pe_valid(PolyExpr e) { return e.uop != NULL; }

static inline int64_t pe_numel(PolyExpr e) {
  int64_t n = 1;
  for (int i = 0; i < e.ndim; i++) n *= e.shape[i];
  return n;
}

/* ── Creation ─────────────────────────────────────────────────────────── */

PolyExpr pe_buffer(PolyCtx *ctx, PolyDType dt, const int64_t *shape, int ndim);
PolyExpr pe_const_float(PolyCtx *ctx, double val);
PolyExpr pe_const_int(PolyCtx *ctx, int64_t val);
PolyExpr pe_full(PolyCtx *ctx, const int64_t *shape, int ndim, double val);
PolyExpr pe_zeros(PolyCtx *ctx, const int64_t *shape, int ndim);
PolyExpr pe_ones(PolyCtx *ctx, const int64_t *shape, int ndim);
PolyExpr pe_rand(PolyCtx *ctx, const int64_t *shape, int ndim, uint64_t seed);
PolyExpr pe_arange(PolyCtx *ctx, double start, double stop, double step);
PolyExpr pe_eye(PolyCtx *ctx, int64_t n);

/* ── Elementwise (binary) ─────────────────────────────────────────────── */

PolyExpr pe_add(PolyExpr a, PolyExpr b);
PolyExpr pe_sub(PolyExpr a, PolyExpr b);
PolyExpr pe_mul(PolyExpr a, PolyExpr b);
PolyExpr pe_div(PolyExpr a, PolyExpr b);
PolyExpr pe_maximum(PolyExpr a, PolyExpr b);
PolyExpr pe_minimum(PolyExpr a, PolyExpr b);

/* ── Elementwise (unary) ──────────────────────────────────────────────── */

PolyExpr pe_neg(PolyExpr x);
PolyExpr pe_exp(PolyExpr x);
PolyExpr pe_log(PolyExpr x);
PolyExpr pe_sqrt(PolyExpr x);
PolyExpr pe_rsqrt(PolyExpr x);
PolyExpr pe_square(PolyExpr x);
PolyExpr pe_abs(PolyExpr x);
PolyExpr pe_sign(PolyExpr x);

/* ── Activations ──────────────────────────────────────────────────────── */

PolyExpr pe_relu(PolyExpr x);
PolyExpr pe_gelu(PolyExpr x);
PolyExpr pe_silu(PolyExpr x);
PolyExpr pe_sigmoid(PolyExpr x);
PolyExpr pe_tanh(PolyExpr x);

/* ── Comparison ───────────────────────────────────────────────────────── */

PolyExpr pe_eq(PolyExpr a, PolyExpr b);
PolyExpr pe_ne(PolyExpr a, PolyExpr b);
PolyExpr pe_gt(PolyExpr a, PolyExpr b);
PolyExpr pe_lt(PolyExpr a, PolyExpr b);
PolyExpr pe_ge(PolyExpr a, PolyExpr b);
PolyExpr pe_le(PolyExpr a, PolyExpr b);

/* ── Selection ────────────────────────────────────────────────────────── */

PolyExpr pe_where(PolyExpr cond, PolyExpr x, PolyExpr y);
PolyExpr pe_clamp(PolyExpr x, double lo, double hi);

/* ── Scalar broadcast ─────────────────────────────────────────────────── */

PolyExpr pe_add_scalar(PolyExpr x, double val);
PolyExpr pe_mul_scalar(PolyExpr x, double val);

/* ── Type casting ─────────────────────────────────────────────────────── */

PolyExpr pe_cast(PolyExpr x, PolyDType target);

/* ── Movement ─────────────────────────────────────────────────────────── */

PolyExpr pe_reshape(PolyExpr x, const int64_t *shape, int ndim);
PolyExpr pe_permute(PolyExpr x, const int64_t *perm, int ndim);
PolyExpr pe_expand(PolyExpr x, const int64_t *shape, int ndim);
PolyExpr pe_shrink(PolyExpr x, int64_t (*pairs)[2], int ndim);
PolyExpr pe_pad(PolyExpr x, int64_t (*pairs)[2], int ndim);
PolyExpr pe_flip(PolyExpr x, int64_t *axes, int n_axes);
PolyExpr pe_transpose(PolyExpr x, int dim0, int dim1);

/* ── Reductions ───────────────────────────────────────────────────────── */

PolyExpr pe_sum(PolyExpr x, int axis, int keepdim);
PolyExpr pe_max(PolyExpr x, int axis, int keepdim);
PolyExpr pe_mean(PolyExpr x, int axis, int keepdim);
PolyExpr pe_var(PolyExpr x, int axis, int keepdim, int correction);

/* ── Matrix ops ───────────────────────────────────────────────────────── */

PolyExpr pe_dot(PolyExpr a, PolyExpr b);
PolyExpr pe_linear(PolyExpr x, PolyExpr weight, PolyExpr *bias);

/* ── NN composed ops ──────────────────────────────────────────────────── */

PolyExpr pe_softmax(PolyExpr x, int axis);
PolyExpr pe_log_softmax(PolyExpr x, int axis);
PolyExpr pe_layernorm(PolyExpr x, int axis, double eps);
PolyExpr pe_cross_entropy(PolyExpr logits, PolyExpr target, int axis);
PolyExpr pe_gather(PolyExpr table, PolyExpr indices);

/* ── New transformer primitives ───────────────────────────────────────── */

PolyExpr pe_rmsnorm(PolyExpr x, PolyExpr *weight, double eps);
PolyExpr pe_scaled_dot_product_attention(PolyExpr q, PolyExpr k, PolyExpr v,
                                         PolyExpr *mask, int is_causal);
PolyExpr pe_rope(PolyExpr x, PolyExpr freqs_cos, PolyExpr freqs_sin);
PolyExpr pe_repeat_interleave(PolyExpr x, int repeats, int dim);
PolyExpr pe_chunk(PolyExpr x, int n_chunks, int dim, PolyExpr *out_chunks);

/* ── Precompute helpers ───────────────────────────────────────────────── */

PolyExpr pe_precompute_freqs_cos(PolyCtx *ctx, int64_t dim, int64_t seq_len, double theta);
PolyExpr pe_precompute_freqs_sin(PolyCtx *ctx, int64_t dim, int64_t seq_len, double theta);
PolyExpr pe_causal_mask(PolyCtx *ctx, int64_t T);

/* ── Graph construction helpers ───────────────────────────────────────── */

PolyUOp *pe_store(PolyExpr x, PolyExpr buf);
PolyUOp *pe_sink1(PolyUOp *store);
PolyExpr pe_assign(PolyExpr target, PolyExpr value);
PolyExpr pe_detach(PolyExpr x);

/* ═══════════════════════════════════════════════════════════════════════ */
/*  NN Layers -- stateful modules using PolyExpr                          */
/*  Mirrors tinygrad nn/__init__.py                                       */
/* ═══════════════════════════════════════════════════════════════════════ */

/* ── Linear ───────────────────────────────────────────────────────────── */

typedef struct {
  PolyExpr weight;       /* (out_features, in_features) */
  PolyExpr bias;         /* (out_features,) -- .uop=NULL if no bias */
  int in_features;
  int out_features;
  int has_bias;
} PeLinear;

PeLinear pe_nn_linear(PolyCtx *ctx, int in_features, int out_features,
                      int use_bias, uint64_t seed);
PolyExpr pe_nn_linear_forward(PeLinear *l, PolyExpr x);

/* ── RMSNorm ──────────────────────────────────────────────────────────── */

typedef struct {
  PolyExpr weight;       /* (dim,) */
  int dim;
  double eps;
} PeRMSNorm;

PeRMSNorm pe_nn_rmsnorm(PolyCtx *ctx, int dim, double eps, uint64_t seed);
PolyExpr  pe_nn_rmsnorm_forward(PeRMSNorm *l, PolyExpr x);

/* ── Embedding ────────────────────────────────────────────────────────── */

typedef struct {
  PolyExpr weight;       /* (vocab_size, embed_dim) */
  int vocab_size;
  int embed_dim;
} PeEmbedding;

PeEmbedding pe_nn_embedding(PolyCtx *ctx, int vocab_size, int embed_dim, uint64_t seed);
PolyExpr    pe_nn_embedding_forward(PeEmbedding *l, PolyExpr indices);

/* ── Dropout ──────────────────────────────────────────────────────────── */

PolyExpr pe_nn_dropout(PolyExpr x, double p, uint64_t seed);

/* ── Attention (multi-head) ───────────────────────────────────────────── */

typedef struct {
  PeLinear wq, wk, wv, wo;
  int n_heads;
  int n_kv_heads;       /* for GQA; == n_heads if no GQA */
  int head_dim;
  int dim;
} PeAttention;

PeAttention pe_nn_attention(PolyCtx *ctx, int dim, int n_heads, int n_kv_heads,
                            int use_bias, uint64_t seed);
PolyExpr    pe_nn_attention_forward(PeAttention *a, PolyExpr x,
                                    PolyExpr *freqs_cos, PolyExpr *freqs_sin,
                                    PolyExpr *mask, int is_causal);

/* ── Layer parameter collection ───────────────────────────────────────── */

/* Append all trainable PolyExpr params from a layer to the array.
 * Returns number of params added. */
int pe_nn_linear_params(PeLinear *l, PolyExpr *out, int max);
int pe_nn_rmsnorm_params(PeRMSNorm *l, PolyExpr *out, int max);
int pe_nn_embedding_params(PeEmbedding *l, PolyExpr *out, int max);
int pe_nn_attention_params(PeAttention *a, PolyExpr *out, int max);

#ifdef __cplusplus
}
#endif

#endif /* POLY_TENSOR_H */
