/*
 * gpt2.c -- GPT-2 model builder for PolyInstance
 *
 * Builds tensor-level UOp graph from GPT-2 config, exports to IR,
 * creates PolyInstance with named parameter buffers matching HF keys.
 *
 * Reference: py/polygrad/nn/gpt2.py (Python GPT-2 implementation)
 * Weight naming: matches HF GPT2 minus "transformer." prefix.
 */

#define _POSIX_C_SOURCE 200809L
#include "../modelzoo.h"
#include "../../frontend.h"
#include "../../poly_ir.h"
#include "../../poly_safetensors.h"
#include "../../sched.h"
#include "../../nn.h"
#include "../../../vendor/cjson/cJSON.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/* Max buffers: params + input + output.
 * Per layer: ln_1 weight/bias, attn c_attn weight/bias, attn c_proj weight/bias,
 *            ln_2 weight/bias, mlp c_fc weight/bias, mlp c_proj weight/bias = 12
 * Plus: wte, wpe, ln_f weight/bias = 4
 * Plus: x (input), output, positions = 3
 * Plus: causal_mask arange buffer = 1 */
#define GPT2_MAX_LAYERS 128
#define GPT2_MAX_BUFS (GPT2_MAX_LAYERS * 12 + 4 + 4)

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

  /* ── Input/output buffers ────────────────────────────────────── */

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

  /* Arange buffer for causal mask: (T_max,) -- filled with 0,1,...,T-1 */
  int64_t arange_shape[] = { T_max };
  PolyUOp *arange_buf = poly_buffer_f32(ctx, T_max);
  buflist_add(&bl, "arange", POLY_IR_ROLE_INPUT, arange_buf, arange_shape, 1);

  /* ── Build forward graph ─────────────────────────────────────── */

  int B = B_max;
  int T = T_max;

  /* Token embeddings: gather(wte, tokens) -> (B, T, D) */
  int64_t tok_emb_shape[8];
  int tok_emb_ndim;
  PolyUOp *tok_emb = poly_gather(ctx, wte, wte_shape, 2,
                                   x_buf, x_shape, 2,
                                   tok_emb_shape, &tok_emb_ndim);

  /* Position embeddings: gather(wpe, positions) -> (1, T, D) */
  int64_t pos_emb_shape[8];
  int pos_emb_ndim;
  PolyUOp *pos_emb = poly_gather(ctx, wpe, wpe_shape, 2,
                                   pos_buf, pos_shape, 2,
                                   pos_emb_shape, &pos_emb_ndim);

  /* Broadcast pos_emb to (B, T, D) */
  int64_t h_shape[] = { B, T, D };
  PolyUOp *pos_exp = poly_expand(ctx, pos_emb, h_shape, 3);

  /* h = tok_emb + pos_emb */
  PolyUOp *h = poly_alu2(ctx, POLY_OP_ADD, tok_emb, pos_exp);

  /* Build causal mask: (T, T) -> (1, 1, T, T) */
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
  /* Reshape to (1, 1, T, T) for broadcast with (B, H, T, T) */
  int64_t mask_4d[] = { 1, 1, T, T };
  PolyUOp *mask = poly_reshape(ctx, mask_val, mask_4d, 4);

  /* ── Transformer blocks ──────────────────────────────────────── */

  for (int i = 0; i < L; i++) {
    /* LayerNorm 1 */
    int64_t ln_shape[8];
    int ln_ndim;
    PolyUOp *ln1 = poly_layernorm(ctx, h, h_shape, 3, -1, (double)eps,
                                    ln_shape, &ln_ndim);

    /* Apply affine: ln1 * weight + bias */
    int64_t w1d[] = { 1, 1, D };
    PolyUOp *ln1_w_r = poly_reshape(ctx, lp[i].ln1_w, w1d, 3);
    PolyUOp *ln1_w_e = poly_expand(ctx, ln1_w_r, h_shape, 3);
    PolyUOp *ln1_b_r = poly_reshape(ctx, lp[i].ln1_b, w1d, 3);
    PolyUOp *ln1_b_e = poly_expand(ctx, ln1_b_r, h_shape, 3);
    ln1 = poly_alu2(ctx, POLY_OP_ADD,
                     poly_alu2(ctx, POLY_OP_MUL, ln1, ln1_w_e), ln1_b_e);

    /* Multi-head attention */
    /* QKV projection: (B, T, D) @ (D, 3D).T + bias -> (B, T, 3D) */
    int64_t qkv_shape[8];
    int qkv_ndim;
    PolyUOp *qkv = poly_linear(ctx, ln1, h_shape, 3,
                                 lp[i].c_attn_w, (int64_t[]){ D, 3 * D }, 2,
                                 lp[i].c_attn_b, (int64_t[]){ 3 * D }, 1,
                                 qkv_shape, &qkv_ndim);

    /* Split Q, K, V via shrink */
    int64_t shrink_q[][2] = { {0, B}, {0, T}, {0, D} };
    int64_t shrink_k[][2] = { {0, B}, {0, T}, {D, 2*D} };
    int64_t shrink_v[][2] = { {0, B}, {0, T}, {2*D, 3*D} };
    PolyUOp *q = poly_shrink(ctx, qkv, shrink_q, 3);
    PolyUOp *k = poly_shrink(ctx, qkv, shrink_k, 3);
    PolyUOp *v = poly_shrink(ctx, qkv, shrink_v, 3);

    /* Reshape to multi-head: (B, T, H, head_dim) -> permute (B, H, T, head_dim) */
    int64_t mh_shape[] = { B, T, H, head_dim };
    q = poly_reshape(ctx, q, mh_shape, 4);
    k = poly_reshape(ctx, k, mh_shape, 4);
    v = poly_reshape(ctx, v, mh_shape, 4);

    int64_t perm_0213[] = { 0, 2, 1, 3 };
    q = poly_permute(ctx, q, perm_0213, 4);  /* (B, H, T, head_dim) */
    k = poly_permute(ctx, k, perm_0213, 4);
    v = poly_permute(ctx, v, perm_0213, 4);

    /* Scaled dot-product attention */
    /* scores = Q @ K.T / sqrt(head_dim) */
    int64_t q_4d[] = { B, H, T, head_dim };

    /* K transpose: (B, H, T, head_dim) -> (B, H, head_dim, T) */
    int64_t perm_0132[] = { 0, 1, 3, 2 };
    PolyUOp *kt = poly_permute(ctx, k, perm_0132, 4);
    int64_t kt_shape[] = { B, H, head_dim, T };

    int64_t scores_shape[8];
    int scores_ndim;
    PolyUOp *scores = poly_dot(ctx, q, q_4d, 4, kt, kt_shape, 4,
                                scores_shape, &scores_ndim);

    /* Scale by 1/sqrt(head_dim) */
    double scale = 1.0 / sqrt((double)head_dim);
    scores = poly_alu2(ctx, POLY_OP_MUL, scores,
                        poly_const_float(ctx, scale));

    /* Apply causal mask: scores + mask */
    int64_t mask_bcast[] = { B, H, T, T };
    PolyUOp *mask_exp = poly_expand(ctx, mask, mask_bcast, 4);
    scores = poly_alu2(ctx, POLY_OP_ADD, scores, mask_exp);

    /* Softmax over last axis */
    int64_t scores_4d[] = { B, H, T, T };
    PolyUOp *attn = poly_softmax(ctx, scores, scores_4d, 4, -1);

    /* Attention @ V: (B, H, T, T) @ (B, H, T, head_dim) -> (B, H, T, head_dim) */
    int64_t attn_4d[] = { B, H, T, T };
    int64_t v_4d[] = { B, H, T, head_dim };
    int64_t attn_out_shape[8];
    int attn_out_ndim;
    PolyUOp *attn_out = poly_dot(ctx, attn, attn_4d, 4, v, v_4d, 4,
                                   attn_out_shape, &attn_out_ndim);

    /* Merge heads: (B, H, T, head_dim) -> (B, T, H, head_dim) -> (B, T, D) */
    int64_t perm_0213_back[] = { 0, 2, 1, 3 };
    attn_out = poly_permute(ctx, attn_out, perm_0213_back, 4);
    int64_t merge_shape[] = { B, T, D };
    attn_out = poly_reshape(ctx, attn_out, merge_shape, 3);

    /* Output projection */
    int64_t proj_out_shape[8];
    int proj_out_ndim;
    attn_out = poly_linear(ctx, attn_out, merge_shape, 3,
                            lp[i].c_proj_w, (int64_t[]){ D, D }, 2,
                            lp[i].c_proj_b, d_shape, 1,
                            proj_out_shape, &proj_out_ndim);

    /* Residual: h = h + attn_out */
    h = poly_alu2(ctx, POLY_OP_ADD, h, attn_out);

    /* LayerNorm 2 */
    int64_t ln2_shape[8];
    int ln2_ndim;
    PolyUOp *ln2 = poly_layernorm(ctx, h, h_shape, 3, -1, (double)eps,
                                    ln2_shape, &ln2_ndim);

    /* Apply affine */
    PolyUOp *ln2_w_r = poly_reshape(ctx, lp[i].ln2_w, w1d, 3);
    PolyUOp *ln2_w_e = poly_expand(ctx, ln2_w_r, h_shape, 3);
    PolyUOp *ln2_b_r = poly_reshape(ctx, lp[i].ln2_b, w1d, 3);
    PolyUOp *ln2_b_e = poly_expand(ctx, ln2_b_r, h_shape, 3);
    ln2 = poly_alu2(ctx, POLY_OP_ADD,
                     poly_alu2(ctx, POLY_OP_MUL, ln2, ln2_w_e), ln2_b_e);

    /* FFN: fc -> gelu -> proj */
    int64_t fc_out_shape[8];
    int fc_out_ndim;
    PolyUOp *ffn = poly_linear(ctx, ln2, h_shape, 3,
                                lp[i].fc_w, (int64_t[]){ D, 4 * D }, 2,
                                lp[i].fc_b, (int64_t[]){ 4 * D }, 1,
                                fc_out_shape, &fc_out_ndim);
    ffn = poly_gelu(ctx, ffn);

    int64_t ffn_shape[] = { B, T, 4 * D };
    int64_t proj_shape[8];
    int proj_ndim;
    ffn = poly_linear(ctx, ffn, ffn_shape, 3,
                       lp[i].proj_w, (int64_t[]){ 4 * D, D }, 2,
                       lp[i].proj_b, d_shape, 1,
                       proj_shape, &proj_ndim);

    /* Residual: h = h + ffn */
    h = poly_alu2(ctx, POLY_OP_ADD, h, ffn);
  }

  /* Final layer norm */
  int64_t lnf_shape[8];
  int lnf_ndim;
  h = poly_layernorm(ctx, h, h_shape, 3, -1, (double)eps, lnf_shape, &lnf_ndim);

  /* Apply affine */
  int64_t w1d_f[] = { 1, 1, D };
  PolyUOp *lnf_w_r = poly_reshape(ctx, ln_f_w, w1d_f, 3);
  PolyUOp *lnf_w_e = poly_expand(ctx, lnf_w_r, h_shape, 3);
  PolyUOp *lnf_b_r = poly_reshape(ctx, ln_f_b, w1d_f, 3);
  PolyUOp *lnf_b_e = poly_expand(ctx, lnf_b_r, h_shape, 3);
  h = poly_alu2(ctx, POLY_OP_ADD,
                 poly_alu2(ctx, POLY_OP_MUL, h, lnf_w_e), lnf_b_e);

  /* LM head: h @ wte.T -> logits (B, T, V) -- weight tying */
  int64_t wte_T_shape[] = { D, V };
  int64_t perm_01[] = { 1, 0 };
  PolyUOp *wte_T = poly_permute(ctx, wte, perm_01, 2);

  int64_t logits_shape[8];
  int logits_ndim;
  PolyUOp *logits = poly_dot(ctx, h, h_shape, 3, wte_T, wte_T_shape, 2,
                               logits_shape, &logits_ndim);

  /* Store logits -> output buffer */
  PolyUOp *fwd_store = poly_store_val(ctx, out_buf, logits);
  PolyUOp *fwd_sink = poly_sink1(ctx, fwd_store);

  /* ── Export to IR and create Instance ────────────────────────── */

  PolyIrEntrypoint eps_arr[1] = {{ "forward", fwd_sink }};
  PolyIrSpec spec = { ctx, bl.bufs, bl.n_bufs, eps_arr, 1 };

  int ir_len = 0;
  uint8_t *ir_data = poly_ir_export(&spec, &ir_len);

  PolyInstance *inst = NULL;
  if (ir_data) {
    inst = poly_instance_from_ir(ir_data, ir_len, NULL, 0);
    free(ir_data);
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
