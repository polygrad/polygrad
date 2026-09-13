/* Model-owned layer construction. Reuses nn programs; owns no runtime.
 * Deprecated ctx-registry constructors remain only for the existing C ABI. */
#include "layers.h"

int poly_model_lstm_cell(
    PolyModel *model,
    const char *prefix,
    PolyTensor *x,
    PolyTensor *h,
    PolyTensor *c,
    int input_size,
    int hidden_size,
    bool bias,
    PolyTensor **new_h,
    PolyTensor **new_c
) {
  if (!new_h || !new_c || new_h == new_c) return -1;
  *new_h = *new_c = NULL;
  if (!model || !x || input_size <= 0 || hidden_size <= 0) return -1;
  PolyCtx *ctx = poly_model_ctx(model);
  bool scoped = prefix && prefix[0];
  if (!ctx || (scoped && poly_model_scope_push(model, "%s", prefix) != POLY_STATUS_OK)) return -1;
  int64_t gates = (int64_t)hidden_size * 4;
  PolyTensor *wi =
      poly_model_param(model, "weight_ih", POLY_FLOAT32, (int64_t[]){gates, input_size}, 2);
  PolyTensor *wh =
      wi ? poly_model_param(model, "weight_hh", POLY_FLOAT32, (int64_t[]){gates, hidden_size}, 2)
         : NULL;
  PolyTensor *bi = bias && wh ? poly_model_param(model, "bias_ih", POLY_FLOAT32, &gates, 1) : NULL;
  PolyTensor *bh = bi ? poly_model_param(model, "bias_hh", POLY_FLOAT32, &gates, 1) : NULL;
  if (scoped && poly_model_scope_pop(model) != POLY_STATUS_OK) return -1;
  if (!wi || !wh || (bias && (!bi || !bh))) return -1;
  return poly_tensor_lstm_cell(ctx, x, h, c, wi, wh, bi, bh, new_h, new_c);
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

PolyUOp *poly_layernorm(PolyCtx *ctx, const char *prefix, PolyUOp *x, int dim, double eps) {
  int64_t ds[] = {dim};
  PolyUOp *w = poly_param(ctx, POLY_FLOAT32, ds, 1, "%s.weight", prefix);
  PolyUOp *b = poly_param(ctx, POLY_FLOAT32, ds, 1, "%s.bias", prefix);
  if (!w || !b) return NULL;

  return poly_layernorm_apply(
      ctx, x, poly_reshape(ctx, w, ds, 1), poly_reshape(ctx, b, ds, 1), -1, eps
  );
}

PolyUOp *poly_rmsnorm(PolyCtx *ctx, const char *prefix, PolyUOp *x, int dim, double eps) {
  int64_t ds[] = {dim};
  PolyUOp *w = poly_param(ctx, POLY_FLOAT32, ds, 1, "%s.weight", prefix);
  if (!w) return NULL;
  return poly_rmsnorm_apply(ctx, x, poly_reshape(ctx, w, ds, 1), eps);
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

PolyTensor *poly_model_linear(
    PolyModel *inst,
    const char *prefix,
    PolyTensor *x,
    int in_features,
    int out_features,
    bool use_bias
) {
  if (!inst || !x) return NULL;
  PolyCtx *ctx = poly_model_ctx(inst);
  if (!ctx) return NULL;
  bool scoped = prefix && prefix[0];
  if (scoped && poly_model_scope_push(inst, "%s", prefix) != POLY_STATUS_OK) return NULL;

  int64_t ws[] = {out_features, in_features};
  PolyTensor *w = poly_model_param(inst, "weight", POLY_FLOAT32, ws, 2);
  PolyTensor *b = NULL;
  if (w && use_bias) {
    int64_t bs[] = {out_features};
    b = poly_model_param(inst, "bias", POLY_FLOAT32, bs, 1);
  }

  if (scoped && poly_model_scope_pop(inst) != POLY_STATUS_OK) return NULL;
  if (!w || (use_bias && !b)) return NULL;
  return poly_tensor_linear_apply(ctx, x, w, b);
}

PolyTensor *poly_model_layernorm(
    PolyModel *inst,
    const char *prefix,
    PolyTensor *x,
    int dim,
    double eps
) {
  if (!inst || !x) return NULL;
  PolyCtx *ctx = poly_model_ctx(inst);
  if (!ctx) return NULL;
  bool scoped = prefix && prefix[0];
  if (scoped && poly_model_scope_push(inst, "%s", prefix) != POLY_STATUS_OK) return NULL;

  int64_t ds[] = {dim};
  PolyTensor *w = poly_model_param(inst, "weight", POLY_FLOAT32, ds, 1);
  PolyTensor *b = poly_model_param(inst, "bias", POLY_FLOAT32, ds, 1);

  if (scoped && poly_model_scope_pop(inst) != POLY_STATUS_OK) return NULL;
  if (!w || !b) return NULL;
  return poly_tensor_layernorm_apply(ctx, x, w, b, -1, eps);
}

PolyTensor *poly_model_rmsnorm(
    PolyModel *inst,
    const char *prefix,
    PolyTensor *x,
    int dim,
    double eps
) {
  if (!inst || !x) return NULL;
  PolyCtx *ctx = poly_model_ctx(inst);
  if (!ctx) return NULL;
  bool scoped = prefix && prefix[0];
  if (scoped && poly_model_scope_push(inst, "%s", prefix) != POLY_STATUS_OK) return NULL;

  int64_t ds[] = {dim};
  PolyTensor *w = poly_model_param(inst, "weight", POLY_FLOAT32, ds, 1);

  if (scoped && poly_model_scope_pop(inst) != POLY_STATUS_OK) return NULL;
  if (!w) return NULL;
  return poly_tensor_rmsnorm_apply(ctx, x, w, eps);
}

PolyTensor *poly_model_embedding(
    PolyModel *inst,
    const char *prefix,
    PolyTensor *tokens,
    int vocab_size,
    int embed_dim
) {
  if (!inst || !tokens) return NULL;
  PolyCtx *ctx = poly_model_ctx(inst);
  if (!ctx) return NULL;
  bool scoped = prefix && prefix[0];
  if (scoped && poly_model_scope_push(inst, "%s", prefix) != POLY_STATUS_OK) return NULL;

  int64_t ws[] = {vocab_size, embed_dim};
  PolyTensor *w = poly_model_param(inst, "weight", POLY_FLOAT32, ws, 2);

  if (scoped && poly_model_scope_pop(inst) != POLY_STATUS_OK) return NULL;
  if (!w) return NULL;
  return poly_tensor_embedding_apply(ctx, tokens, w);
}
