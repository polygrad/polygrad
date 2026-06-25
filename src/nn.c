/*
 * nn.c — Neural network layers
 *
 * Two functions per layer:
 *   poly_X_apply(ctx, x, w, b, ...) — the math, takes explicit weight UOps
 *   poly_X(ctx, prefix, x, ...)     — registers named params, calls apply
 *
 * All math uses shape-on-UOp (no explicit shape parameters).
 */

#include "nn.h"
#include "instance.h"
#include "tensor.h" /* poly_mean_reduce */
#include "engine/schedule.h" /* poly_reshape, poly_permute, poly_expand */
#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

/* RNG */

static uint32_t nn_rng_state = 12345;

void poly_nn_seed(uint32_t seed) {
  nn_rng_state = seed;
}

/* Linear */

PolyUOp *poly_linear_apply(PolyCtx *ctx, PolyUOp *x, PolyUOp *w, PolyUOp *b) {
  if (!ctx || !x || !w) return NULL;
  int64_t perm[] = {1, 0};
  PolyUOp *out = poly_dot(ctx, x, poly_permute(ctx, w, perm, 2));
  if (!out) return NULL;
  if (b) out = poly_add(ctx, out, b);
  return out;
}

PolyUOp *poly_linear(
    PolyCtx *ctx,
    const char *prefix,
    PolyUOp *x,
    int in_features,
    int out_features,
    bool use_bias
) {
  int64_t ws[] = {out_features, in_features};
  PolyUOp *w = poly_param(ctx, POLY_FLOAT32, ws, 2, "%s.weight", prefix);
  if (!w) return NULL;
  w = poly_reshape(ctx, w, ws, 2);

  PolyUOp *b = NULL;
  if (use_bias) {
    int64_t bs[] = {out_features};
    b = poly_param(ctx, POLY_FLOAT32, bs, 1, "%s.bias", prefix);
    if (!b) return NULL;
  }

  return poly_linear_apply(ctx, x, w, b);
}

PolyUOp *poly_instance_linear(
    PolyInstance *inst,
    const char *prefix,
    PolyUOp *x,
    int in_features,
    int out_features,
    bool use_bias
) {
  if (!inst || !x) return NULL;
  PolyCtx *ctx = poly_instance_ctx(inst);
  if (!ctx) return NULL;
  bool scoped = prefix && prefix[0];
  if (scoped && poly_instance_scope_push(inst, "%s", prefix) != POLY_STATUS_OK) return NULL;

  int64_t ws[] = {out_features, in_features};
  PolyTensor *w_tensor = poly_instance_param(inst, "weight", POLY_FLOAT32, ws, 2);
  PolyUOp *w = w_tensor ? poly_tensor_uop(w_tensor) : NULL;

  PolyUOp *b = NULL;
  if (w && use_bias) {
    int64_t bs[] = {out_features};
    PolyTensor *b_tensor = poly_instance_param(inst, "bias", POLY_FLOAT32, bs, 1);
    b = b_tensor ? poly_tensor_uop(b_tensor) : NULL;
  }

  if (scoped && poly_instance_scope_pop(inst) != POLY_STATUS_OK) return NULL;
  if (!w || (use_bias && !b)) return NULL;
  return poly_linear_apply(ctx, x, w, b);
}

/* LayerNorm */

PolyUOp *poly_layernorm_apply(
    PolyCtx *ctx,
    PolyUOp *x,
    PolyUOp *w,
    PolyUOp *b,
    int axis,
    double eps
) {
  if (!ctx || !x) return NULL;

  PolyUOp *mean = poly_mean_reduce(ctx, x, axis, 1);
  PolyUOp *centered = poly_sub(ctx, x, mean);
  PolyUOp *sq = poly_alu2(ctx, POLY_OP_MUL, centered, centered);
  PolyUOp *var = poly_mean_reduce(ctx, sq, axis, 1);
  PolyUOp *denom = poly_alu1(ctx, POLY_OP_SQRT, poly_add(ctx, var, poly_const_float(ctx, eps)));
  PolyUOp *normed = poly_mul(ctx, centered, poly_alu1(ctx, POLY_OP_RECIPROCAL, denom));

  if (w) normed = poly_mul(ctx, normed, w);
  if (b) normed = poly_add(ctx, normed, b);

  return normed;
}

PolyUOp *poly_layernorm(PolyCtx *ctx, const char *prefix, PolyUOp *x, int dim, double eps) {
  int64_t ds[] = {dim};
  PolyUOp *w = poly_param(ctx, POLY_FLOAT32, ds, 1, "%s.weight", prefix);
  PolyUOp *b = poly_param(ctx, POLY_FLOAT32, ds, 1, "%s.bias", prefix);
  if (!w || !b) return NULL;

  return poly_layernorm_apply(
      ctx, x, poly_reshape(ctx, w, ds, 1), poly_reshape(ctx, b, ds, 1), -1, eps
  );
}

PolyUOp *poly_instance_layernorm(
    PolyInstance *inst,
    const char *prefix,
    PolyUOp *x,
    int dim,
    double eps
) {
  if (!inst || !x) return NULL;
  PolyCtx *ctx = poly_instance_ctx(inst);
  if (!ctx) return NULL;
  bool scoped = prefix && prefix[0];
  if (scoped && poly_instance_scope_push(inst, "%s", prefix) != POLY_STATUS_OK) return NULL;

  int64_t ds[] = {dim};
  PolyTensor *w_tensor = poly_instance_param(inst, "weight", POLY_FLOAT32, ds, 1);
  PolyTensor *b_tensor = poly_instance_param(inst, "bias", POLY_FLOAT32, ds, 1);

  if (scoped && poly_instance_scope_pop(inst) != POLY_STATUS_OK) return NULL;
  if (!w_tensor || !b_tensor) return NULL;
  return poly_layernorm_apply(
      ctx, x, poly_reshape(ctx, poly_tensor_uop(w_tensor), ds, 1),
      poly_reshape(ctx, poly_tensor_uop(b_tensor), ds, 1), -1, eps
  );
}

/* RMSNorm */

PolyUOp *poly_rmsnorm_apply(PolyCtx *ctx, PolyUOp *x, PolyUOp *w, double eps) {
  if (!ctx || !x) return NULL;
  int ndim = poly_uop_ndim(ctx, x);
  if (ndim < 1) return NULL;
  const int64_t *dims = poly_uop_max_shape_dims(ctx, x);
  if (!dims) return NULL;
  int64_t shape[POLY_MAX_DIMS];
  memcpy(shape, dims, ndim * sizeof(int64_t));
  int axis = ndim - 1;

  PolyUOp *x2 = poly_alu2(ctx, POLY_OP_MUL, x, x);
  PolyUOp *m = poly_mean_reduce(ctx, x2, axis, 1);
  m = poly_alu2(ctx, POLY_OP_ADD, m, poly_const_float(ctx, eps));
  PolyUOp *rrms = poly_alu1(ctx, POLY_OP_SQRT, m);
  rrms = poly_alu2(ctx, POLY_OP_FDIV, poly_const_float(ctx, 1.0), rrms);
  rrms = poly_expand(ctx, rrms, shape, ndim);
  PolyUOp *normed = poly_alu2(ctx, POLY_OP_MUL, x, rrms);

  if (w) {
    int w_ndim = poly_uop_ndim(ctx, w);
    if (w_ndim == 1) {
      const int64_t *w_dims = poly_uop_max_shape_dims(ctx, w);
      int64_t bc[POLY_MAX_DIMS];
      for (int i = 0; i < ndim - 1; i++)
        bc[i] = 1;
      bc[ndim - 1] = w_dims ? w_dims[0] : 0;
      PolyUOp *w_r = poly_reshape(ctx, w, bc, ndim);
      PolyUOp *w_e = poly_expand(ctx, w_r, shape, ndim);
      normed = poly_alu2(ctx, POLY_OP_MUL, normed, w_e);
    }
  }
  return normed;
}

PolyUOp *poly_rmsnorm(PolyCtx *ctx, const char *prefix, PolyUOp *x, int dim, double eps) {
  int64_t ds[] = {dim};
  PolyUOp *w = poly_param(ctx, POLY_FLOAT32, ds, 1, "%s.weight", prefix);
  if (!w) return NULL;
  return poly_rmsnorm_apply(ctx, x, poly_reshape(ctx, w, ds, 1), eps);
}

PolyUOp *poly_instance_rmsnorm(
    PolyInstance *inst,
    const char *prefix,
    PolyUOp *x,
    int dim,
    double eps
) {
  if (!inst || !x) return NULL;
  PolyCtx *ctx = poly_instance_ctx(inst);
  if (!ctx) return NULL;
  bool scoped = prefix && prefix[0];
  if (scoped && poly_instance_scope_push(inst, "%s", prefix) != POLY_STATUS_OK) return NULL;

  int64_t ds[] = {dim};
  PolyTensor *w_tensor = poly_instance_param(inst, "weight", POLY_FLOAT32, ds, 1);

  if (scoped && poly_instance_scope_pop(inst) != POLY_STATUS_OK) return NULL;
  if (!w_tensor) return NULL;
  return poly_rmsnorm_apply(ctx, x, poly_reshape(ctx, poly_tensor_uop(w_tensor), ds, 1), eps);
}

/* Embedding */

PolyUOp *poly_embedding_apply(PolyCtx *ctx, PolyUOp *tokens, PolyUOp *table) {
  return poly_gather(ctx, table, tokens);
}

PolyUOp *poly_embedding(
    PolyCtx *ctx,
    const char *prefix,
    PolyUOp *tokens,
    int vocab_size,
    int embed_dim
) {
  int64_t ws[] = {vocab_size, embed_dim};
  PolyUOp *w = poly_param(ctx, POLY_FLOAT32, ws, 2, "%s.weight", prefix);
  if (!w) return NULL;
  return poly_embedding_apply(ctx, tokens, poly_reshape(ctx, w, ws, 2));
}

PolyUOp *poly_instance_embedding(
    PolyInstance *inst,
    const char *prefix,
    PolyUOp *tokens,
    int vocab_size,
    int embed_dim
) {
  if (!inst || !tokens) return NULL;
  PolyCtx *ctx = poly_instance_ctx(inst);
  if (!ctx) return NULL;
  bool scoped = prefix && prefix[0];
  if (scoped && poly_instance_scope_push(inst, "%s", prefix) != POLY_STATUS_OK) return NULL;

  int64_t ws[] = {vocab_size, embed_dim};
  PolyTensor *w_tensor = poly_instance_param(inst, "weight", POLY_FLOAT32, ws, 2);

  if (scoped && poly_instance_scope_pop(inst) != POLY_STATUS_OK) return NULL;
  if (!w_tensor) return NULL;
  return poly_embedding_apply(ctx, tokens, poly_reshape(ctx, poly_tensor_uop(w_tensor), ws, 2));
}

/* Causal attention mask */

PolyUOp *poly_causal_mask(PolyCtx *ctx, int64_t T) {
  if (!ctx || T <= 0) return NULL;

  PolyUOp *arange_buf = poly_arange(ctx, 0.0, (double)T, 1.0);
  int64_t row_shape[] = {T, 1};
  PolyUOp *row =
      poly_expand(ctx, poly_reshape(ctx, arange_buf, row_shape, 2), (int64_t[]){T, T}, 2);
  int64_t col_shape[] = {1, T};
  PolyUOp *col =
      poly_expand(ctx, poly_reshape(ctx, arange_buf, col_shape, 2), (int64_t[]){T, T}, 2);

  PolyUOp *mask = poly_alu2(ctx, POLY_OP_CMPLT, row, col);
  return poly_where_op(ctx, mask, poly_const_float(ctx, -1e9), poly_const_float(ctx, 0.0));
}

/* Scaled Dot-Product Attention */

/*
 * Repeat K/V heads for Grouped Query Attention (GQA).
 * Input:  (B, n_kv_heads, T, head_dim)
 * Output: (B, n_heads, T, head_dim)  where n_heads = n_kv_heads * n_rep
 *
 * Matches tinygrad's repeat_kv: x.repeat((1,1,1,n_rep)).reshape(...)
 * but using expand (no data copy).
 */
static PolyUOp *repeat_kv(PolyCtx *ctx, PolyUOp *kv, int n_rep) {
  if (n_rep <= 1) return kv;
  const int64_t *dims = poly_uop_max_shape_dims(ctx, kv);
  int ndim = poly_uop_ndim(ctx, kv);
  if (ndim != 4 || !dims) return NULL;
  /* (B, n_kv_heads, T, hd) -> (B, n_kv_heads, 1, T, hd) */
  int64_t rs[] = {dims[0], dims[1], 1, dims[2], dims[3]};
  PolyUOp *r = poly_reshape(ctx, kv, rs, 5);
  /* expand the new dim to n_rep */
  int64_t ex[] = {dims[0], dims[1], n_rep, dims[2], dims[3]};
  r = poly_expand(ctx, r, ex, 5);
  /* flatten back: (B, n_kv_heads * n_rep, T, hd) */
  int64_t fl[] = {dims[0], dims[1] * n_rep, dims[2], dims[3]};
  return poly_reshape(ctx, r, fl, 4);
}

PolyUOp *poly_sdpa(PolyCtx *ctx, PolyUOp *q, PolyUOp *k, PolyUOp *v, PolyUOp *mask, int is_causal) {
  int q_ndim = poly_uop_ndim(ctx, q);
  int k_ndim = poly_uop_ndim(ctx, k);
  int v_ndim = poly_uop_ndim(ctx, v);
  if (q_ndim < 2 || k_ndim < 2 || v_ndim < 2) return NULL;
  const int64_t *q_dims = poly_uop_max_shape_dims(ctx, q);
  const int64_t *k_dims = poly_uop_max_shape_dims(ctx, k);
  if (!q_dims || !k_dims) return NULL;

  /* GQA: if Q has more heads than K/V, repeat K/V */
  if (q_ndim == 4 && k_ndim == 4 && q_dims[1] != k_dims[1]) {
    int n_rep = (int)(q_dims[1] / k_dims[1]);
    k = repeat_kv(ctx, k, n_rep);
    v = repeat_kv(ctx, v, n_rep);
    k_dims = poly_uop_max_shape_dims(ctx, k);
  }

  int64_t d_k = q_dims[q_ndim - 1];
  double scale = 1.0 / sqrt((double)d_k);

  int64_t k_perm[POLY_MAX_DIMS];
  for (int i = 0; i < k_ndim; i++)
    k_perm[i] = i;
  k_perm[k_ndim - 2] = k_ndim - 1;
  k_perm[k_ndim - 1] = k_ndim - 2;
  PolyUOp *k_t = poly_permute(ctx, k, k_perm, k_ndim);

  PolyUOp *scores = poly_dot(ctx, q, k_t);
  scores = poly_alu2(ctx, POLY_OP_MUL, scores, poly_const_float(ctx, scale));

  if (is_causal) {
    int64_t seq_q = q_dims[q_ndim - 2];
    int64_t seq_k = poly_uop_max_shape_dims(ctx, k)[k_ndim - 2];
    PolyUOp *ones = poly_full(ctx, (int64_t[]){seq_q, seq_k}, 2, 1.0);
    PolyUOp *tril_m = poly_tril(ctx, ones, 0);
    PolyUOp *cond = poly_alu2(ctx, POLY_OP_CMPLT, tril_m, poly_const_float(ctx, 0.5));
    PolyUOp *cmask = poly_alu3(
        ctx, POLY_OP_WHERE, cond, poly_const_float(ctx, -1e9), poly_const_float(ctx, 0.0)
    );
    scores = poly_add(ctx, scores, cmask);
  }

  if (mask) scores = poly_add(ctx, scores, mask);

  return poly_dot(ctx, poly_softmax(ctx, scores, -1), v);
}
