/*
 * gpt2.c -- GPT-2 model builder for PolyInstance
 *
 * Builds tensor-level UOp graph from GPT-2 config, exports to IR,
 * creates PolyInstance with named parameter buffers matching HF keys.
 *
 * Reference: py/polygrad/nn/gpt2.py (Python GPT-2 implementation)
 * Weight naming: matches HF GPT2 minus "transformer." prefix.
 *
 * Important: CONTIGUOUS barriers are placed at the same points as
 * the Python GPT-2's .realize() calls to keep kernel sizes manageable.
 */

#define _POSIX_C_SOURCE 200809L
#include "../modelzoo.h"
#include "../../frontend.h"
#include "../../instance.h"
#include "../../ir.h"
#include "../../safetensors.h"
#include "../../sched.h"
#include "../../nn.h"
#include "../../../vendor/cjson/cJSON.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/* Max buffers: params + input + output + aux.
 * Per layer: ln_1 weight/bias, attn c_attn weight/bias, attn c_proj weight/bias,
 *            ln_2 weight/bias, mlp c_fc weight/bias, mlp c_proj weight/bias = 12
 * Plus: wte, wpe, ln_f weight/bias = 4
 * Plus: x (input), output, positions, arange = 4
 * Plus: vocab_arange (aux) = 1 */
#define GPT2_MAX_LAYERS 128
#define GPT2_MAX_BUFS (GPT2_MAX_LAYERS * 12 + 4 + 5)

/* Insert a CONTIGUOUS realize barrier (same as Python .realize()) */
#define REALIZE(u) poly_uop1(ctx, POLY_OP_CONTIGUOUS, (u)->dtype, (u), poly_arg_none())

/* ── Helper: add named param buffer ─────────────────────────────── */

typedef struct {
  PolyIrBufEntry *bufs;
  int n_bufs;
  int cap;
} BufList;

static void buflist_add(BufList *bl, const char *name, uint8_t role,
                         PolyUOp *buffer, const int64_t *shape, int ndim) {
  if (bl->n_bufs >= bl->cap) return;
  PolyIrBufEntry *e = &bl->bufs[bl->n_bufs++];
  e->name = name;
  e->role = role;
  e->buffer = buffer;
  e->ndim = ndim;
  for (int i = 0; i < ndim; i++) e->shape[i] = shape[i];
}

/* ── HF Conv1D linear (no weight transpose) ─────────────────────── */

/* HF GPT-2 uses Conv1D convention: y = x @ weight + bias
 * Weight stored as (in_features, out_features), NOT PyTorch (out, in).
 * poly_linear() expects PyTorch convention, so we use poly_dot directly. */
static PolyUOp *hf_linear(PolyCtx *ctx,
                            PolyUOp *x, const int64_t *x_shape, int x_ndim,
                            PolyUOp *weight, const int64_t *w_shape, int w_ndim,
                            PolyUOp *bias, int64_t bias_features,
                            int64_t *out_shape, int *out_ndim) {
  /* x @ weight (no transpose) */
  PolyUOp *result = poly_dot(ctx, x, x_shape, x_ndim, weight, w_shape, w_ndim,
                               out_shape, out_ndim);

  /* + bias: reshape to (1,...,1, out_features) and broadcast */
  if (bias) {
    int64_t b_shape[POLY_MAX_DIMS];
    for (int i = 0; i < *out_ndim - 1; i++) b_shape[i] = 1;
    b_shape[*out_ndim - 1] = bias_features;
    PolyUOp *b_r = poly_reshape(ctx, bias, b_shape, *out_ndim);
    PolyUOp *b_exp = poly_expand(ctx, b_r, out_shape, *out_ndim);
    result = poly_alu2(ctx, POLY_OP_ADD, result, b_exp);
  }
  return result;
}

/* ── Inline embedding (no anonymous buffers) ────────────────────── */

/* Embedding lookup using registered arange buffer.
 * Same algorithm as poly_gather but uses caller-provided arange buffer
 * instead of creating an anonymous one.
 *
 * table: (N, D), indices: (...), arange_buf: (N,) with 0,1,...,N-1
 * result: (..., D) */
static PolyUOp *embedding(PolyCtx *ctx,
                           PolyUOp *table, int64_t N, int64_t D,
                           PolyUOp *indices, const int64_t *idx_shape, int idx_ndim,
                           PolyUOp *arange_buf) {
  /* indices.unsqueeze(-1) -> (..., 1) */
  int64_t idx_us_shape[POLY_MAX_DIMS];
  int idx_us_ndim = idx_ndim + 1;
  for (int i = 0; i < idx_ndim; i++) idx_us_shape[i] = idx_shape[i];
  idx_us_shape[idx_ndim] = 1;
  PolyUOp *idx_us = poly_reshape(ctx, indices, idx_us_shape, idx_us_ndim);

  /* Broadcast to (..., N) */
  int64_t idx_bcast[POLY_MAX_DIMS];
  for (int i = 0; i < idx_ndim; i++) idx_bcast[i] = idx_shape[i];
  idx_bcast[idx_ndim] = N;
  PolyUOp *idx_exp = poly_expand(ctx, idx_us, idx_bcast, idx_us_ndim);

  /* Reshape arange to (1,...,1, N) and broadcast to (..., N) */
  int64_t arange_shape[POLY_MAX_DIMS];
  for (int i = 0; i < idx_ndim; i++) arange_shape[i] = 1;
  arange_shape[idx_ndim] = N;
  PolyUOp *arange_r = poly_reshape(ctx, arange_buf, arange_shape, idx_us_ndim);
  PolyUOp *arange_exp = poly_expand(ctx, arange_r, idx_bcast, idx_us_ndim);

  /* mask = eq(idx, arange) -> (..., N) */
  PolyUOp *mask = poly_eq(ctx, idx_exp, arange_exp);

  /* mask.unsqueeze(-1) -> (..., N, 1), broadcast to (..., N, D) */
  int mask_us_ndim = idx_us_ndim + 1;
  int64_t mask_us_shape[POLY_MAX_DIMS];
  for (int i = 0; i < idx_us_ndim; i++) mask_us_shape[i] = idx_bcast[i];
  mask_us_shape[idx_us_ndim] = 1;
  PolyUOp *mask_us = poly_reshape(ctx, mask, mask_us_shape, mask_us_ndim);

  int64_t mask_bcast[POLY_MAX_DIMS];
  for (int i = 0; i < idx_us_ndim; i++) mask_bcast[i] = idx_bcast[i];
  mask_bcast[idx_us_ndim] = D;
  PolyUOp *mask_exp = poly_expand(ctx, mask_us, mask_bcast, mask_us_ndim);

  /* table -> (1,...,1, N, D) -> (..., N, D) */
  int64_t tbl_shape[POLY_MAX_DIMS];
  for (int i = 0; i < idx_ndim; i++) tbl_shape[i] = 1;
  tbl_shape[idx_ndim] = N;
  tbl_shape[idx_ndim + 1] = D;
  PolyUOp *tbl_r = poly_reshape(ctx, table, tbl_shape, mask_us_ndim);
  PolyUOp *tbl_exp = poly_expand(ctx, tbl_r, mask_bcast, mask_us_ndim);

  /* where(mask, table, 0) -> sum over N axis */
  PolyUOp *selected = poly_where_op(ctx, mask_exp, tbl_exp,
                                      poly_const_float(ctx, 0.0));
  int64_t reduce_axes[] = { idx_ndim };
  PolyUOp *gathered = poly_reduce_axis(ctx, POLY_OP_ADD, selected, reduce_axes, 1);

  /* REDUCE_AXIS keeps dim (keepdim=true): (..., 1, D).
   * Reshape to squeeze the reduced V axis: (..., D). */
  int64_t out_shape[POLY_MAX_DIMS];
  int out_ndim = idx_ndim + 1;
  for (int i = 0; i < idx_ndim; i++) out_shape[i] = idx_shape[i];
  out_shape[idx_ndim] = D;
  return poly_reshape(ctx, gathered, out_shape, out_ndim);
}

/* ── Post-init: populate arange AUX buffers ─────────────────────── */

static void populate_arange_bufs(PolyInstance *inst) {
  int n = poly_instance_buf_count(inst);
  for (int i = 0; i < n; i++) {
    const char *name = poly_instance_buf_name(inst, i);
    if (strcmp(name, "vocab_arange") == 0) {
      int64_t numel;
      float *data = poly_instance_buf_data(inst, i, &numel);
      if (data) {
        for (int64_t j = 0; j < numel; j++) data[j] = (float)j;
      }
    }
  }
}

/* ── GPT-2 Builder ───────────────────────────────────────────────── */

PolyInstance *poly_gpt2_build(const GPT2Config *cfg, int max_batch) {
  if (!cfg || cfg->n_layer < 1 || cfg->n_embd < 1 || cfg->vocab_size < 1)
    return NULL;

  int V = cfg->vocab_size;
  int D = cfg->n_embd;
  int H = cfg->n_head;
  int L = cfg->n_layer;
  int T_max = cfg->max_seq_len;
  int B_max = max_batch > 0 ? max_batch : 1;
  int head_dim = D / H;
  float eps = cfg->norm_eps > 0 ? cfg->norm_eps : 1e-5f;

  if (D % H != 0) {
    fprintf(stderr, "poly_gpt2_build: n_embd (%d) not divisible by n_head (%d)\n", D, H);
    return NULL;
  }

  PolyCtx *ctx = poly_ctx_new();

  /* Allocate buf list */
  BufList bl;
  bl.cap = GPT2_MAX_BUFS;
  bl.bufs = calloc(bl.cap, sizeof(PolyIrBufEntry));
  bl.n_bufs = 0;

  /* String arena for param names (heap-alloc, freed at cleanup) */
  char **name_strs = calloc(bl.cap, sizeof(char *));
  int n_names = 0;

  #define MKNAME(fmt, ...) do { \
    name_strs[n_names] = malloc(128); \
    snprintf(name_strs[n_names], 128, fmt, ##__VA_ARGS__); \
  } while(0)

  #define ADD_PARAM(nm, buf, shp, nd) do { \
    MKNAME("%s", nm); \
    buflist_add(&bl, name_strs[n_names], POLY_IR_ROLE_PARAM, buf, shp, nd); \
    n_names++; \
  } while(0)

  #define ADD_PARAM_FMT(fmt, buf, shp, nd, ...) do { \
    name_strs[n_names] = malloc(128); \
    snprintf(name_strs[n_names], 128, fmt, ##__VA_ARGS__); \
    buflist_add(&bl, name_strs[n_names], POLY_IR_ROLE_PARAM, buf, shp, nd); \
    n_names++; \
  } while(0)

  /* ── Create parameter buffers ────────────────────────────────── */

  /* Token embedding: wte.weight (V, D) */
  int64_t wte_shape[] = { V, D };
  PolyUOp *wte = poly_buffer_f32(ctx, (int64_t)V * D);
  ADD_PARAM("wte.weight", wte, wte_shape, 2);

  /* Position embedding: wpe.weight (T_max, D) */
  int64_t wpe_shape[] = { T_max, D };
  PolyUOp *wpe = poly_buffer_f32(ctx, (int64_t)T_max * D);
  ADD_PARAM("wpe.weight", wpe, wpe_shape, 2);

  /* Per-layer params */
  typedef struct {
    PolyUOp *ln1_w, *ln1_b;
    PolyUOp *c_attn_w, *c_attn_b;
    PolyUOp *c_proj_w, *c_proj_b;
    PolyUOp *ln2_w, *ln2_b;
    PolyUOp *fc_w, *fc_b;
    PolyUOp *proj_w, *proj_b;
  } LayerParams;

  LayerParams *lp = calloc(L, sizeof(LayerParams));

  for (int i = 0; i < L; i++) {
    int64_t d_shape[] = { D };
    int64_t attn_w_shape[] = { D, 3 * D };
    int64_t attn_b_shape[] = { 3 * D };
    int64_t proj_w_shape[] = { D, D };
    int64_t fc_w_shape[] = { D, 4 * D };
    int64_t fc_b_shape[] = { 4 * D };

    lp[i].ln1_w = poly_buffer_f32(ctx, D);
    ADD_PARAM_FMT("h.%d.ln_1.weight", lp[i].ln1_w, d_shape, 1, i);
    lp[i].ln1_b = poly_buffer_f32(ctx, D);
    ADD_PARAM_FMT("h.%d.ln_1.bias", lp[i].ln1_b, d_shape, 1, i);

    lp[i].c_attn_w = poly_buffer_f32(ctx, (int64_t)D * 3 * D);
    ADD_PARAM_FMT("h.%d.attn.c_attn.weight", lp[i].c_attn_w, attn_w_shape, 2, i);
    lp[i].c_attn_b = poly_buffer_f32(ctx, 3 * D);
    ADD_PARAM_FMT("h.%d.attn.c_attn.bias", lp[i].c_attn_b, attn_b_shape, 1, i);

    lp[i].c_proj_w = poly_buffer_f32(ctx, (int64_t)D * D);
    ADD_PARAM_FMT("h.%d.attn.c_proj.weight", lp[i].c_proj_w, proj_w_shape, 2, i);
    lp[i].c_proj_b = poly_buffer_f32(ctx, D);
    ADD_PARAM_FMT("h.%d.attn.c_proj.bias", lp[i].c_proj_b, d_shape, 1, i);

    lp[i].ln2_w = poly_buffer_f32(ctx, D);
    ADD_PARAM_FMT("h.%d.ln_2.weight", lp[i].ln2_w, d_shape, 1, i);
    lp[i].ln2_b = poly_buffer_f32(ctx, D);
    ADD_PARAM_FMT("h.%d.ln_2.bias", lp[i].ln2_b, d_shape, 1, i);

    lp[i].fc_w = poly_buffer_f32(ctx, (int64_t)D * 4 * D);
    ADD_PARAM_FMT("h.%d.mlp.c_fc.weight", lp[i].fc_w, fc_w_shape, 2, i);
    lp[i].fc_b = poly_buffer_f32(ctx, 4 * D);
    ADD_PARAM_FMT("h.%d.mlp.c_fc.bias", lp[i].fc_b, fc_b_shape, 1, i);

    int64_t proj_w_shape2[] = { 4 * D, D };
    lp[i].proj_w = poly_buffer_f32(ctx, (int64_t)4 * D * D);
    ADD_PARAM_FMT("h.%d.mlp.c_proj.weight", lp[i].proj_w, proj_w_shape2, 2, i);
    lp[i].proj_b = poly_buffer_f32(ctx, D);
    ADD_PARAM_FMT("h.%d.mlp.c_proj.bias", lp[i].proj_b, d_shape, 1, i);
  }

  /* Final layer norm */
  int64_t d_shape[] = { D };
  PolyUOp *ln_f_w = poly_buffer_f32(ctx, D);
  ADD_PARAM("ln_f.weight", ln_f_w, d_shape, 1);
  PolyUOp *ln_f_b = poly_buffer_f32(ctx, D);
  ADD_PARAM("ln_f.bias", ln_f_b, d_shape, 1);

  /* ── Input/output/aux buffers ──────────────────────────────────── */

  /* Input tokens: (B_max, T_max) */
  int64_t x_shape[] = { B_max, T_max };
  PolyUOp *x_buf = poly_buffer_f32(ctx, (int64_t)B_max * T_max);
  buflist_add(&bl, "x", POLY_IR_ROLE_INPUT, x_buf, x_shape, 2);

  /* Output logits: (B_max, T_max, V) */
  int64_t out_shape[] = { B_max, T_max, V };
  PolyUOp *out_buf = poly_buffer_f32(ctx, (int64_t)B_max * T_max * V);
  buflist_add(&bl, "output", POLY_IR_ROLE_OUTPUT, out_buf, out_shape, 3);

  /* Position indices: (1, T_max) -- filled at runtime with 0,1,...,T-1 */
  int64_t pos_shape[] = { 1, T_max };
  PolyUOp *pos_buf = poly_buffer_f32(ctx, T_max);
  buflist_add(&bl, "positions", POLY_IR_ROLE_INPUT, pos_buf, pos_shape, 2);

  /* Arange buffer for causal mask + position embedding: (T_max,)
   * Filled at runtime with 0,1,...,T-1 */
  int64_t arange_shape[] = { T_max };
  PolyUOp *arange_buf = poly_buffer_f32(ctx, T_max);
  buflist_add(&bl, "arange", POLY_IR_ROLE_INPUT, arange_buf, arange_shape, 1);

  /* Vocab arange: (V,) -- AUX buffer, auto-populated with 0,1,...,V-1 */
  int64_t varange_shape[] = { V };
  PolyUOp *varange_buf = poly_buffer_f32(ctx, V);
  buflist_add(&bl, "vocab_arange", POLY_IR_ROLE_AUX, varange_buf, varange_shape, 1);

  /* ── Build forward graph ─────────────────────────────────────── */

  int B = B_max;
  int T = T_max;

  /* Token embeddings: embedding(wte, tokens, vocab_arange) -> (B, T, D) */
  PolyUOp *tok_emb = embedding(ctx, wte, V, D, x_buf, x_shape, 2, varange_buf);
  tok_emb = REALIZE(tok_emb);

  /* Position embeddings: embedding(wpe, positions, arange_buf)
   * Note: we reuse arange_buf (size T_max) as the position arange since
   * position indices are also 0..T-1. But wpe has T_max rows, so the
   * arange for position embedding lookup needs size T_max -- same buffer. */
  PolyUOp *pos_emb = embedding(ctx, wpe, T_max, D, pos_buf, pos_shape, 2, arange_buf);
  pos_emb = REALIZE(pos_emb);

  /* Broadcast pos_emb to (B, T, D) */
  int64_t h_shape[] = { B, T, D };
  PolyUOp *pos_exp = poly_expand(ctx, pos_emb, h_shape, 3);

  /* h = tok_emb + pos_emb, realize */
  PolyUOp *h = poly_alu2(ctx, POLY_OP_ADD, tok_emb, pos_exp);
  h = REALIZE(h);

  /* Build causal mask: (T, T) -> (1, 1, T, T), realize */
  int64_t mask_row_shape[] = { T, 1 };
  PolyUOp *mask_row = poly_reshape(ctx, arange_buf, mask_row_shape, 2);
  int64_t mask_row_exp_shape[] = { T, T };
  PolyUOp *mask_row_exp = poly_expand(ctx, mask_row, mask_row_exp_shape, 2);
  int64_t mask_col_shape[] = { 1, T };
  PolyUOp *mask_col = poly_reshape(ctx, arange_buf, mask_col_shape, 2);
  PolyUOp *mask_col_exp = poly_expand(ctx, mask_col, mask_row_exp_shape, 2);
  PolyUOp *mask_cmp = poly_alu2(ctx, POLY_OP_CMPLT, mask_row_exp, mask_col_exp);
  PolyUOp *mask_val = poly_where_op(ctx, mask_cmp,
                                      poly_const_float(ctx, -1e9),
                                      poly_const_float(ctx, 0.0));
  int64_t mask_4d[] = { 1, 1, T, T };
  PolyUOp *mask = poly_reshape(ctx, mask_val, mask_4d, 4);
  mask = REALIZE(mask);

  /* ── Transformer blocks ──────────────────────────────────────── */

  for (int i = 0; i < L; i++) {
    /* LayerNorm 1 + affine, realize */
    int64_t ln_shape[8];
    int ln_ndim;
    PolyUOp *ln1 = poly_layernorm(ctx, h, h_shape, 3, -1, (double)eps,
                                    ln_shape, &ln_ndim);
    int64_t w1d[] = { 1, 1, D };
    PolyUOp *ln1_w_r = poly_reshape(ctx, lp[i].ln1_w, w1d, 3);
    PolyUOp *ln1_w_e = poly_expand(ctx, ln1_w_r, h_shape, 3);
    PolyUOp *ln1_b_r = poly_reshape(ctx, lp[i].ln1_b, w1d, 3);
    PolyUOp *ln1_b_e = poly_expand(ctx, ln1_b_r, h_shape, 3);
    ln1 = poly_alu2(ctx, POLY_OP_ADD,
                     poly_alu2(ctx, POLY_OP_MUL, ln1, ln1_w_e), ln1_b_e);
    ln1 = REALIZE(ln1);

    /* QKV projection: (B, T, D) @ (D, 3D) + bias -> (B, T, 3D), realize */
    int64_t qkv_shape[8];
    int qkv_ndim;
    PolyUOp *qkv = hf_linear(ctx, ln1, h_shape, 3,
                               lp[i].c_attn_w, (int64_t[]){ D, 3 * D }, 2,
                               lp[i].c_attn_b, 3 * D,
                               qkv_shape, &qkv_ndim);
    qkv = REALIZE(qkv);

    /* Split Q, K, V via shrink, realize each */
    int64_t shrink_q[][2] = { {0, B}, {0, T}, {0, D} };
    int64_t shrink_k[][2] = { {0, B}, {0, T}, {D, 2*D} };
    int64_t shrink_v[][2] = { {0, B}, {0, T}, {2*D, 3*D} };
    PolyUOp *q = REALIZE(poly_shrink(ctx, qkv, shrink_q, 3));
    PolyUOp *k = REALIZE(poly_shrink(ctx, qkv, shrink_k, 3));
    PolyUOp *v = REALIZE(poly_shrink(ctx, qkv, shrink_v, 3));

    /* Reshape to multi-head: (B, T, H, head_dim) -> permute (B, H, T, head_dim) */
    int64_t mh_shape[] = { B, T, H, head_dim };
    q = poly_reshape(ctx, q, mh_shape, 4);
    k = poly_reshape(ctx, k, mh_shape, 4);
    v = poly_reshape(ctx, v, mh_shape, 4);

    int64_t perm_0213[] = { 0, 2, 1, 3 };
    q = poly_permute(ctx, q, perm_0213, 4);
    k = poly_permute(ctx, k, perm_0213, 4);
    v = poly_permute(ctx, v, perm_0213, 4);

    /* K transpose: (B, H, T, head_dim) -> (B, H, head_dim, T) */
    int64_t q_4d[] = { B, H, T, head_dim };
    int64_t perm_0132[] = { 0, 1, 3, 2 };
    PolyUOp *kt = poly_permute(ctx, k, perm_0132, 4);
    int64_t kt_shape[] = { B, H, head_dim, T };

    /* scores = Q @ K.T, realize */
    int64_t scores_shape[8];
    int scores_ndim;
    PolyUOp *scores = poly_dot(ctx, q, q_4d, 4, kt, kt_shape, 4,
                                scores_shape, &scores_ndim);
    scores = REALIZE(scores);

    /* Scale by 1/sqrt(head_dim) */
    double scale = 1.0 / sqrt((double)head_dim);
    scores = poly_alu2(ctx, POLY_OP_MUL, scores,
                        poly_const_float(ctx, scale));

    /* Apply causal mask: scores + mask, realize */
    int64_t mask_bcast[] = { B, H, T, T };
    PolyUOp *mask_exp = poly_expand(ctx, mask, mask_bcast, 4);
    scores = poly_alu2(ctx, POLY_OP_ADD, scores, mask_exp);
    scores = REALIZE(scores);

    /* Softmax, realize */
    int64_t scores_4d[] = { B, H, T, T };
    PolyUOp *attn = poly_softmax(ctx, scores, scores_4d, 4, -1);
    attn = REALIZE(attn);

    /* Attention @ V, realize */
    int64_t attn_4d[] = { B, H, T, T };
    int64_t v_4d[] = { B, H, T, head_dim };
    int64_t attn_out_shape[8];
    int attn_out_ndim;
    PolyUOp *attn_out = poly_dot(ctx, attn, attn_4d, 4, v, v_4d, 4,
                                   attn_out_shape, &attn_out_ndim);
    attn_out = REALIZE(attn_out);

    /* Merge heads: (B, H, T, head_dim) -> (B, T, H, head_dim) -> (B, T, D) */
    int64_t perm_0213_back[] = { 0, 2, 1, 3 };
    attn_out = poly_permute(ctx, attn_out, perm_0213_back, 4);
    int64_t merge_shape[] = { B, T, D };
    attn_out = poly_reshape(ctx, attn_out, merge_shape, 3);

    /* Output projection, realize */
    int64_t proj_out_shape[8];
    int proj_out_ndim;
    attn_out = hf_linear(ctx, attn_out, merge_shape, 3,
                          lp[i].c_proj_w, (int64_t[]){ D, D }, 2,
                          lp[i].c_proj_b, D,
                          proj_out_shape, &proj_out_ndim);
    attn_out = REALIZE(attn_out);

    /* Residual: h = h + attn_out, realize */
    h = poly_alu2(ctx, POLY_OP_ADD, h, attn_out);
    h = REALIZE(h);

    /* LayerNorm 2 + affine, realize */
    int64_t ln2_shape[8];
    int ln2_ndim;
    PolyUOp *ln2 = poly_layernorm(ctx, h, h_shape, 3, -1, (double)eps,
                                    ln2_shape, &ln2_ndim);
    PolyUOp *ln2_w_r = poly_reshape(ctx, lp[i].ln2_w, w1d, 3);
    PolyUOp *ln2_w_e = poly_expand(ctx, ln2_w_r, h_shape, 3);
    PolyUOp *ln2_b_r = poly_reshape(ctx, lp[i].ln2_b, w1d, 3);
    PolyUOp *ln2_b_e = poly_expand(ctx, ln2_b_r, h_shape, 3);
    ln2 = poly_alu2(ctx, POLY_OP_ADD,
                     poly_alu2(ctx, POLY_OP_MUL, ln2, ln2_w_e), ln2_b_e);
    ln2 = REALIZE(ln2);

    /* FFN: fc, realize, gelu, realize, proj, realize */
    int64_t fc_out_shape[8];
    int fc_out_ndim;
    PolyUOp *ffn = hf_linear(ctx, ln2, h_shape, 3,
                              lp[i].fc_w, (int64_t[]){ D, 4 * D }, 2,
                              lp[i].fc_b, 4 * D,
                              fc_out_shape, &fc_out_ndim);
    ffn = REALIZE(ffn);
    ffn = poly_gelu(ctx, ffn);
    ffn = REALIZE(ffn);

    int64_t ffn_shape[] = { B, T, 4 * D };
    int64_t proj_shape[8];
    int proj_ndim;
    ffn = hf_linear(ctx, ffn, ffn_shape, 3,
                     lp[i].proj_w, (int64_t[]){ 4 * D, D }, 2,
                     lp[i].proj_b, D,
                     proj_shape, &proj_ndim);
    ffn = REALIZE(ffn);

    /* Residual: h = h + ffn, realize */
    h = poly_alu2(ctx, POLY_OP_ADD, h, ffn);
    h = REALIZE(h);
  }

  /* Final layer norm + affine, realize */
  int64_t lnf_shape[8];
  int lnf_ndim;
  h = poly_layernorm(ctx, h, h_shape, 3, -1, (double)eps, lnf_shape, &lnf_ndim);
  int64_t w1d_f[] = { 1, 1, D };
  PolyUOp *lnf_w_r = poly_reshape(ctx, ln_f_w, w1d_f, 3);
  PolyUOp *lnf_w_e = poly_expand(ctx, lnf_w_r, h_shape, 3);
  PolyUOp *lnf_b_r = poly_reshape(ctx, ln_f_b, w1d_f, 3);
  PolyUOp *lnf_b_e = poly_expand(ctx, lnf_b_r, h_shape, 3);
  h = poly_alu2(ctx, POLY_OP_ADD,
                 poly_alu2(ctx, POLY_OP_MUL, h, lnf_w_e), lnf_b_e);
  h = REALIZE(h);

  /* LM head: h @ wte.T -> logits (B, T, V) -- weight tying */
  int64_t wte_T_shape[] = { D, V };
  int64_t perm_01[] = { 1, 0 };
  PolyUOp *wte_2d = poly_reshape(ctx, wte, wte_shape, 2);
  PolyUOp *wte_T = poly_permute(ctx, wte_2d, perm_01, 2);

  int64_t logits_shape[8];
  int logits_ndim;
  PolyUOp *logits = poly_dot(ctx, h, h_shape, 3, wte_T, wte_T_shape, 2,
                               logits_shape, &logits_ndim);

  /* Store logits -> output buffer */
  PolyUOp *fwd_store = poly_store_val(ctx, out_buf, logits);
  PolyUOp *fwd_sink = poly_sink1(ctx, fwd_store);

  /* ── Loss computation (surrogate: sum of logits^2) ──────────── */
  /* Matches Python training test: loss = (logits * logits).sum()
   * This is sufficient to verify autograd through the full GPT-2 graph.
   * The loss is computed from the logits UOp (not out_buf) so autograd
   * can trace back to all parameters. */

  PolyUOp *loss_buf = poly_buffer_f32(ctx, 1);
  buflist_add(&bl, "loss", POLY_IR_ROLE_OUTPUT, loss_buf, (int64_t[]){1}, 1);

  PolyUOp *logits_sq = poly_alu2(ctx, POLY_OP_MUL, logits, logits);

  /* Reduce over all 3 axes: (B, T, V) -> (1, 1, 1) */
  int64_t reduce_all[] = { 0, 1, 2 };
  PolyUOp *loss_sum = poly_reduce_axis(ctx, POLY_OP_ADD, logits_sq, reduce_all, 3);

  /* Reshape to (1,) scalar for loss buffer */
  int64_t scalar_shape[] = { 1 };
  PolyUOp *loss_val = poly_reshape(ctx, loss_sum, scalar_shape, 1);

  PolyUOp *loss_store = poly_store_val(ctx, loss_buf, loss_val);
  PolyUOp *loss_sink = poly_sink1(ctx, loss_store);

  /* ── Export to IR and create Instance ────────────────────────── */

  PolyIrEntrypoint eps_arr[2] = {
    { "forward", fwd_sink },
    { "loss", loss_sink }
  };
  PolyIrSpec spec = { ctx, bl.bufs, bl.n_bufs, eps_arr, 2 };

  int ir_len = 0;
  uint8_t *ir_data = poly_ir_export(&spec, &ir_len);

  PolyInstance *inst = NULL;
  if (ir_data) {
    inst = poly_instance_from_ir(ir_data, ir_len, NULL, 0);
    free(ir_data);
  }

  /* Populate AUX arange buffers */
  if (inst) {
    populate_arange_bufs(inst);
  }

  /* Cleanup */
  free(lp);
  for (int i = 0; i < n_names; i++) free(name_strs[i]);
  free(name_strs);
  free(bl.bufs);
  poly_ctx_destroy(ctx);

  #undef MKNAME
  #undef ADD_PARAM
  #undef ADD_PARAM_FMT

  return inst;
}

/* ── Modelzoo vtable integration (legacy API) ──────────────────── */

static PolyUOp **gpt2_forward(PolyModel *model, PolyUOp **inputs,
                                int num_inputs, int *out_num_outputs) {
  (void)model; (void)inputs; (void)num_inputs;
  *out_num_outputs = 0;
  return NULL;
}

static PolyUOp *gpt2_get_parameter(PolyModel *model, const char *name) {
  (void)model; (void)name;
  return NULL;
}

static void gpt2_free(PolyModel *model) { (void)model; }

static PolyModel *create_gpt2_model(PolyCtx *ctx, PolyModelConfig *config) {
  (void)ctx; (void)config;
  PolyModel *model = calloc(1, sizeof(PolyModel));
  model->forward = gpt2_forward;
  model->get_parameter = gpt2_get_parameter;
  model->free = gpt2_free;
  return model;
}

void __attribute__((constructor)) register_gpt2(void) {
  poly_model_register("gpt2", create_gpt2_model);
}
