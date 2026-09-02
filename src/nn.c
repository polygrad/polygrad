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

/* Polygrad retains a portable logical twin, but default execution follows the
 * same ordered Tensor.uop construction as pinned Tensor._apply_uop
 * (tinygrad/tensor.py:128-140). Layer programs therefore run independently
 * over the exact logical and physical occurrences; neither root is recovered
 * from the other. */
static PolyTensor *nn_tensor_result(
    PolyCtx *ctx,
    PolyUOp *logical,
    PolyUOp *physical,
    PolyTensor **inputs,
    int n_inputs
) {
  int build_logical = poly_tensor_result_builds_logical(ctx, inputs, n_inputs);
  if (build_logical < 0 || !physical || (build_logical && !logical) ||
      (logical && !poly_ctx_owns_ptr(ctx, logical)) || !poly_ctx_owns_ptr(ctx, physical))
    return NULL;
  PolyDevice device = POLY_DEVICE_AUTO;
  bool requires_grad = false;
  bool requires_grad_set = false;
  for (int i = 0; i < n_inputs; i++) {
    PolyTensor *input = inputs[i];
    if (!input) continue;
    if (input->device != POLY_DEVICE_AUTO) {
      if (device != POLY_DEVICE_AUTO && device != input->device) return NULL;
      device = input->device;
    }
    requires_grad |= input->requires_grad;
    requires_grad_set |= input->requires_grad_set;
  }
  PolyTensor *out = poly_tensor_create_result(
      ctx, inputs, n_inputs, logical, physical, POLY_TENSOR_VALUE, device
  );
  if (!out) return NULL;
  out->requires_grad = requires_grad;
  out->requires_grad_set = requires_grad_set;
  out->provenance = POLY_TENSOR_PROVENANCE_COMPUTED;
  return out;
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

PolyTensor *poly_tensor_linear_apply(PolyCtx *ctx, PolyTensor *x, PolyTensor *w, PolyTensor *b) {
  if (!ctx || !x || !w) return NULL;
  PolyTensor *inputs[3] = {x, w, b};
  int build_logical = poly_tensor_result_builds_logical(ctx, inputs, b ? 3 : 2);
  if (build_logical < 0) return NULL;
  /* Pinned nn.Linear stores (out,in), transposes it, then calls Tensor.linear;
   * Tensor.linear is dot followed by optional add
   * (nn/__init__.py:156-177; mixin/__init__.py:1335-1350). */
  PolyUOp *physical =
      poly_linear_apply(ctx, x->uop_physical, w->uop_physical, b ? b->uop_physical : NULL);
  PolyUOp *logical =
      build_logical
          ? poly_linear_apply(ctx, x->uop_logical, w->uop_logical, b ? b->uop_logical : NULL)
          : NULL;
  return nn_tensor_result(ctx, logical, physical, inputs, b ? 3 : 2);
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

PolyTensor *poly_instance_linear(
    PolyInstance *inst,
    const char *prefix,
    PolyTensor *x,
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
  PolyTensor *w = poly_instance_param(inst, "weight", POLY_FLOAT32, ws, 2);
  PolyTensor *b = NULL;
  if (w && use_bias) {
    int64_t bs[] = {out_features};
    b = poly_instance_param(inst, "bias", POLY_FLOAT32, bs, 1);
  }

  if (scoped && poly_instance_scope_pop(inst) != POLY_STATUS_OK) return NULL;
  if (!w || (use_bias && !b)) return NULL;
  return poly_tensor_linear_apply(ctx, x, w, b);
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

PolyTensor *poly_tensor_layernorm_apply(
    PolyCtx *ctx,
    PolyTensor *x,
    PolyTensor *w,
    PolyTensor *b,
    int axis,
    double eps
) {
  if (!ctx || !x || (!!w != !!b)) return NULL;
  PolyTensor *inputs[3] = {x, w, b};
  int build_logical = poly_tensor_result_builds_logical(ctx, inputs, w ? 3 : 1);
  if (build_logical < 0) return NULL;
  /* Pinned LayerNorm first runs Tensor.layernorm, then applies the affine
   * weight and bias (nn/__init__.py:235-261; mixin/__init__.py:1548-1564). */
  PolyUOp *physical = poly_layernorm_apply(
      ctx, x->uop_physical, w ? w->uop_physical : NULL, b ? b->uop_physical : NULL, axis, eps
  );
  PolyUOp *logical = build_logical ? poly_layernorm_apply(
                                         ctx, x->uop_logical, w ? w->uop_logical : NULL,
                                         b ? b->uop_logical : NULL, axis, eps
                                     )
                                   : NULL;
  return nn_tensor_result(ctx, logical, physical, inputs, w ? 3 : 1);
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

PolyTensor *poly_instance_layernorm(
    PolyInstance *inst,
    const char *prefix,
    PolyTensor *x,
    int dim,
    double eps
) {
  if (!inst || !x) return NULL;
  PolyCtx *ctx = poly_instance_ctx(inst);
  if (!ctx) return NULL;
  bool scoped = prefix && prefix[0];
  if (scoped && poly_instance_scope_push(inst, "%s", prefix) != POLY_STATUS_OK) return NULL;

  int64_t ds[] = {dim};
  PolyTensor *w = poly_instance_param(inst, "weight", POLY_FLOAT32, ds, 1);
  PolyTensor *b = poly_instance_param(inst, "bias", POLY_FLOAT32, ds, 1);

  if (scoped && poly_instance_scope_pop(inst) != POLY_STATUS_OK) return NULL;
  if (!w || !b) return NULL;
  return poly_tensor_layernorm_apply(ctx, x, w, b, -1, eps);
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

PolyTensor *poly_tensor_rmsnorm_apply(PolyCtx *ctx, PolyTensor *x, PolyTensor *w, double eps) {
  if (!ctx || !x) return NULL;
  PolyTensor *inputs[2] = {x, w};
  int build_logical = poly_tensor_result_builds_logical(ctx, inputs, w ? 2 : 1);
  if (build_logical < 0) return NULL;
  /* Pinned RMSNorm normalizes x.float(), casts back, and applies the optional
   * affine weight (nn/__init__.py:281-304). The existing raw program is run
   * over both retained occurrences without correspondence. */
  PolyUOp *physical = poly_rmsnorm_apply(ctx, x->uop_physical, w ? w->uop_physical : NULL, eps);
  PolyUOp *logical = build_logical
                         ? poly_rmsnorm_apply(ctx, x->uop_logical, w ? w->uop_logical : NULL, eps)
                         : NULL;
  return nn_tensor_result(ctx, logical, physical, inputs, w ? 2 : 1);
}

PolyUOp *poly_rmsnorm(PolyCtx *ctx, const char *prefix, PolyUOp *x, int dim, double eps) {
  int64_t ds[] = {dim};
  PolyUOp *w = poly_param(ctx, POLY_FLOAT32, ds, 1, "%s.weight", prefix);
  if (!w) return NULL;
  return poly_rmsnorm_apply(ctx, x, poly_reshape(ctx, w, ds, 1), eps);
}

PolyTensor *poly_instance_rmsnorm(
    PolyInstance *inst,
    const char *prefix,
    PolyTensor *x,
    int dim,
    double eps
) {
  if (!inst || !x) return NULL;
  PolyCtx *ctx = poly_instance_ctx(inst);
  if (!ctx) return NULL;
  bool scoped = prefix && prefix[0];
  if (scoped && poly_instance_scope_push(inst, "%s", prefix) != POLY_STATUS_OK) return NULL;

  int64_t ds[] = {dim};
  PolyTensor *w = poly_instance_param(inst, "weight", POLY_FLOAT32, ds, 1);

  if (scoped && poly_instance_scope_pop(inst) != POLY_STATUS_OK) return NULL;
  if (!w) return NULL;
  return poly_tensor_rmsnorm_apply(ctx, x, w, eps);
}

/* Embedding */

PolyUOp *poly_embedding_apply(PolyCtx *ctx, PolyUOp *tokens, PolyUOp *table) {
  return poly_gather(ctx, table, tokens);
}

PolyTensor *poly_tensor_embedding_apply(PolyCtx *ctx, PolyTensor *tokens, PolyTensor *table) {
  if (!ctx || !tokens || !table) return NULL;
  PolyTensor *inputs[2] = {tokens, table};
  int build_logical = poly_tensor_result_builds_logical(ctx, inputs, 2);
  if (build_logical < 0) return NULL;
  /* Pinned Embedding is its one-hot WHERE/SUM program over the ordered weight
   * and index Tensor.uops (nn/__init__.py:368-391). */
  PolyUOp *physical = poly_embedding_apply(ctx, tokens->uop_physical, table->uop_physical);
  PolyUOp *logical =
      build_logical ? poly_embedding_apply(ctx, tokens->uop_logical, table->uop_logical) : NULL;
  return nn_tensor_result(ctx, logical, physical, inputs, 2);
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

PolyTensor *poly_instance_embedding(
    PolyInstance *inst,
    const char *prefix,
    PolyTensor *tokens,
    int vocab_size,
    int embed_dim
) {
  if (!inst || !tokens) return NULL;
  PolyCtx *ctx = poly_instance_ctx(inst);
  if (!ctx) return NULL;
  bool scoped = prefix && prefix[0];
  if (scoped && poly_instance_scope_push(inst, "%s", prefix) != POLY_STATUS_OK) return NULL;

  int64_t ws[] = {vocab_size, embed_dim};
  PolyTensor *w = poly_instance_param(inst, "weight", POLY_FLOAT32, ws, 2);

  if (scoped && poly_instance_scope_pop(inst) != POLY_STATUS_OK) return NULL;
  if (!w) return NULL;
  return poly_tensor_embedding_apply(ctx, tokens, w);
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

PolyTensor *poly_tensor_causal_mask(PolyCtx *ctx, int64_t T) {
  if (!ctx || T <= 0) return NULL;
  bool build_logical = poly_ctx_get_logical_policy(ctx) != POLY_LOGICAL_NEVER;
  /* Pinned GPT-2/LLaMA mask construction is a pure Tensor graph. With no
   * BUFFER occurrence, retained and executable roots may CSE to the same
   * node, but both approved roots are stored explicitly. */
  PolyUOp *physical = poly_causal_mask(ctx, T);
  PolyUOp *logical = build_logical ? poly_causal_mask(ctx, T) : NULL;
  if (!physical || (build_logical && !logical)) return NULL;
  PolyTensor *out = poly_tensor_create_with_roots(
      ctx, logical, physical, POLY_TENSOR_VALUE, poly_ctx_get_preferred_device(ctx)
  );
  if (!out) return NULL;
  out->requires_grad = false;
  out->requires_grad_set = true;
  out->provenance = POLY_TENSOR_PROVENANCE_COMPUTED;
  return out;
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

PolyTensor *poly_tensor_sdpa(
    PolyCtx *ctx,
    PolyTensor *q,
    PolyTensor *k,
    PolyTensor *v,
    PolyTensor *mask,
    int is_causal
) {
  if (!ctx || !q || !k || !v) return NULL;
  PolyTensor *inputs[4] = {q, k, v, mask};
  int build_logical = poly_tensor_result_builds_logical(ctx, inputs, mask ? 4 : 3);
  if (build_logical < 0) return NULL;
  PolyUOp *physical = poly_sdpa(
      ctx, q->uop_physical, k->uop_physical, v->uop_physical, mask ? mask->uop_physical : NULL,
      is_causal
  );
  PolyUOp *logical = build_logical ? poly_sdpa(
                                         ctx, q->uop_logical, k->uop_logical, v->uop_logical,
                                         mask ? mask->uop_logical : NULL, is_causal
                                     )
                                   : NULL;
  return nn_tensor_result(ctx, logical, physical, inputs, mask ? 4 : 3);
}
