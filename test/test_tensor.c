/*
 * test_tensor.c -- Tests for PolyExpr (internal shape-tracked expression wrapper)
 *
 * Reference values verified against tinygrad (conda env 'tiny').
 */

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "test_harness.h"
#include "../src/tensor.h"

/* ── Helper: realize a PolyExpr into a float array ────────────────────── */

static int realize_expr(PolyExpr x, PolyExpr out_buf, float *out_data,
                        PolyExpr *leaf_bufs, float **leaf_datas, int n_leaves) {
  PolyUOp *store = poly_store_val(x.ctx, out_buf.buf ? out_buf.buf : out_buf.uop, x.uop);
  PolyUOp *sink = poly_sink1(x.ctx, store);

  int n_total = n_leaves + 1;
  PolyUOp *bufs[64];
  void *datas[64];
  for (int i = 0; i < n_leaves; i++) {
    bufs[i] = leaf_bufs[i].buf ? leaf_bufs[i].buf : leaf_bufs[i].uop;
    datas[i] = leaf_datas[i];
  }
  bufs[n_leaves] = out_buf.buf ? out_buf.buf : out_buf.uop;
  datas[n_leaves] = out_data;

  return poly_realize_flat(x.ctx, sink, bufs, datas, n_total);
}

/* ── Shape tracking ───────────────────────────────────────────────────── */

TEST(pe, buffer_shape) {
  PolyCtx *ctx = poly_ctx_new();
  int64_t shape[] = {2, 3, 4};
  PolyExpr e = pe_buffer(ctx, POLY_FLOAT32, shape, 3);
  ASSERT_TRUE(pe_valid(e));
  ASSERT_INT_EQ(e.ndim, 3);
  ASSERT_INT_EQ(e.shape[0], 2);
  ASSERT_INT_EQ(e.shape[1], 3);
  ASSERT_INT_EQ(e.shape[2], 4);
  ASSERT_INT_EQ(pe_numel(e), 24);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, reshape_shape) {
  PolyCtx *ctx = poly_ctx_new();
  PolyExpr e = pe_buffer(ctx, POLY_FLOAT32, (int64_t[]){2, 6}, 2);
  PolyExpr r = pe_reshape(e, (int64_t[]){3, 4}, 2);
  ASSERT_INT_EQ(r.ndim, 2);
  ASSERT_INT_EQ(r.shape[0], 3);
  ASSERT_INT_EQ(r.shape[1], 4);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, transpose_shape) {
  PolyCtx *ctx = poly_ctx_new();
  PolyExpr e = pe_buffer(ctx, POLY_FLOAT32, (int64_t[]){2, 3, 4}, 3);
  PolyExpr t = pe_transpose(e, -2, -1);
  ASSERT_INT_EQ(t.ndim, 3);
  ASSERT_INT_EQ(t.shape[0], 2);
  ASSERT_INT_EQ(t.shape[1], 4);
  ASSERT_INT_EQ(t.shape[2], 3);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, permute_shape) {
  PolyCtx *ctx = poly_ctx_new();
  PolyExpr e = pe_buffer(ctx, POLY_FLOAT32, (int64_t[]){2, 3, 4}, 3);
  PolyExpr p = pe_permute(e, (int64_t[]){2, 0, 1}, 3);
  ASSERT_INT_EQ(p.shape[0], 4);
  ASSERT_INT_EQ(p.shape[1], 2);
  ASSERT_INT_EQ(p.shape[2], 3);
  poly_ctx_destroy(ctx);
  PASS();
}

/* ── Elementwise e2e ──────────────────────────────────────────────────── */

TEST(pe, add_e2e) {
  PolyCtx *ctx = poly_ctx_new();
  PolyExpr a = pe_buffer(ctx, POLY_FLOAT32, (int64_t[]){3}, 1);
  PolyExpr b = pe_buffer(ctx, POLY_FLOAT32, (int64_t[]){3}, 1);
  PolyExpr out = pe_buffer(ctx, POLY_FLOAT32, (int64_t[]){3}, 1);
  PolyExpr r = pe_add(a, b);

  float da[] = {1.0f, 2.0f, 3.0f};
  float db[] = {4.0f, 5.0f, 6.0f};
  float dout[3] = {0};
  PolyExpr leaves[] = {a, b};
  float *leaf_data[] = {da, db};
  ASSERT_INT_EQ(realize_expr(r, out, dout, leaves, leaf_data, 2), 0);
  ASSERT_FLOAT_EQ(dout[0], 5.0f, 1e-6);
  ASSERT_FLOAT_EQ(dout[1], 7.0f, 1e-6);
  ASSERT_FLOAT_EQ(dout[2], 9.0f, 1e-6);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, dot_e2e) {
  PolyCtx *ctx = poly_ctx_new();
  /* (2,3) @ (3,2) -> (2,2) */
  PolyExpr a = pe_buffer(ctx, POLY_FLOAT32, (int64_t[]){2, 3}, 2);
  PolyExpr b = pe_buffer(ctx, POLY_FLOAT32, (int64_t[]){3, 2}, 2);
  PolyExpr out = pe_buffer(ctx, POLY_FLOAT32, (int64_t[]){2, 2}, 2);
  PolyExpr r = pe_dot(a, b);
  ASSERT_INT_EQ(r.ndim, 2);
  ASSERT_INT_EQ(r.shape[0], 2);
  ASSERT_INT_EQ(r.shape[1], 2);

  float da[] = {1,2,3, 4,5,6};
  float db[] = {1,2, 3,4, 5,6};
  float dout[4] = {0};
  PolyExpr leaves[] = {a, b};
  float *leaf_data[] = {da, db};
  ASSERT_INT_EQ(realize_expr(r, out, dout, leaves, leaf_data, 2), 0);
  /* [1*1+2*3+3*5, 1*2+2*4+3*6] = [22, 28] */
  ASSERT_FLOAT_EQ(dout[0], 22.0f, 1e-4);
  ASSERT_FLOAT_EQ(dout[1], 28.0f, 1e-4);
  /* [4*1+5*3+6*5, 4*2+5*4+6*6] = [49, 64] */
  ASSERT_FLOAT_EQ(dout[2], 49.0f, 1e-4);
  ASSERT_FLOAT_EQ(dout[3], 64.0f, 1e-4);
  poly_ctx_destroy(ctx);
  PASS();
}

/* ── RMSNorm ──────────────────────────────────────────────────────────── */

TEST(pe, rmsnorm_e2e) {
  /* Reference (tinygrad): [[0.4629, 0.9258, 1.3887], [0.7895, 0.9869, 1.1843]] */
  PolyCtx *ctx = poly_ctx_new();
  PolyExpr x = pe_buffer(ctx, POLY_FLOAT32, (int64_t[]){2, 3}, 2);
  PolyExpr w = pe_buffer(ctx, POLY_FLOAT32, (int64_t[]){3}, 1);
  PolyExpr out = pe_buffer(ctx, POLY_FLOAT32, (int64_t[]){2, 3}, 2);

  PolyExpr r = pe_rmsnorm(x, &w, 1e-5);
  ASSERT_TRUE(pe_valid(r));
  ASSERT_INT_EQ(r.ndim, 2);
  ASSERT_INT_EQ(r.shape[0], 2);
  ASSERT_INT_EQ(r.shape[1], 3);

  float dx[] = {1, 2, 3, 4, 5, 6};
  float dw[] = {1, 1, 1};
  float dout[6] = {0};
  PolyExpr leaves[] = {x, w};
  float *leaf_data[] = {dx, dw};
  ASSERT_INT_EQ(realize_expr(r, out, dout, leaves, leaf_data, 2), 0);

  ASSERT_FLOAT_EQ(dout[0], 0.46290955f, 1e-4);
  ASSERT_FLOAT_EQ(dout[1], 0.92581910f, 1e-4);
  ASSERT_FLOAT_EQ(dout[2], 1.38872860f, 1e-4);
  ASSERT_FLOAT_EQ(dout[3], 0.78954184f, 1e-4);
  ASSERT_FLOAT_EQ(dout[4], 0.98692730f, 1e-4);
  ASSERT_FLOAT_EQ(dout[5], 1.18431280f, 1e-4);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, rmsnorm_no_weight) {
  PolyCtx *ctx = poly_ctx_new();
  PolyExpr x = pe_buffer(ctx, POLY_FLOAT32, (int64_t[]){2, 3}, 2);
  PolyExpr out = pe_buffer(ctx, POLY_FLOAT32, (int64_t[]){2, 3}, 2);
  PolyExpr r = pe_rmsnorm(x, NULL, 1e-5);
  ASSERT_TRUE(pe_valid(r));

  float dx[] = {1, 2, 3, 4, 5, 6};
  float dout[6] = {0};
  PolyExpr leaves[] = {x};
  float *leaf_data[] = {dx};
  ASSERT_INT_EQ(realize_expr(r, out, dout, leaves, leaf_data, 1), 0);
  /* Same as with weight=[1,1,1] */
  ASSERT_FLOAT_EQ(dout[0], 0.46290955f, 1e-4);
  poly_ctx_destroy(ctx);
  PASS();
}

/* ── Scaled Dot-Product Attention ─────────────────────────────────────── */

TEST(pe, sdpa_e2e) {
  /* Reference (tinygrad): [[[1.6605, 2.6605], [2.3395, 3.3395]]] */
  PolyCtx *ctx = poly_ctx_new();
  PolyExpr q = pe_buffer(ctx, POLY_FLOAT32, (int64_t[]){1, 2, 2}, 3);
  PolyExpr k = pe_buffer(ctx, POLY_FLOAT32, (int64_t[]){1, 2, 2}, 3);
  PolyExpr v = pe_buffer(ctx, POLY_FLOAT32, (int64_t[]){1, 2, 2}, 3);
  PolyExpr out = pe_buffer(ctx, POLY_FLOAT32, (int64_t[]){1, 2, 2}, 3);

  PolyExpr r = pe_scaled_dot_product_attention(q, k, v, NULL, 0);
  ASSERT_TRUE(pe_valid(r));
  ASSERT_INT_EQ(r.ndim, 3);
  ASSERT_INT_EQ(r.shape[0], 1);
  ASSERT_INT_EQ(r.shape[1], 2);
  ASSERT_INT_EQ(r.shape[2], 2);

  float dq[] = {1, 0, 0, 1};
  float dk[] = {1, 0, 0, 1};
  float dv[] = {1, 2, 3, 4};
  float dout[4] = {0};
  PolyExpr leaves[] = {q, k, v};
  float *leaf_data[] = {dq, dk, dv};
  ASSERT_INT_EQ(realize_expr(r, out, dout, leaves, leaf_data, 3), 0);

  ASSERT_FLOAT_EQ(dout[0], 1.6604769f, 1e-3);
  ASSERT_FLOAT_EQ(dout[1], 2.6604769f, 1e-3);
  ASSERT_FLOAT_EQ(dout[2], 2.3395231f, 1e-3);
  ASSERT_FLOAT_EQ(dout[3], 3.3395231f, 1e-3);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, sdpa_causal_e2e) {
  /* Reference (tinygrad): [[[1.0, 0.0], [0.3302, 0.6698], [0.7517, 0.7517]]] */
  PolyCtx *ctx = poly_ctx_new();
  PolyExpr q = pe_buffer(ctx, POLY_FLOAT32, (int64_t[]){1, 3, 2}, 3);
  PolyExpr k = pe_buffer(ctx, POLY_FLOAT32, (int64_t[]){1, 3, 2}, 3);
  PolyExpr v = pe_buffer(ctx, POLY_FLOAT32, (int64_t[]){1, 3, 2}, 3);
  PolyExpr out = pe_buffer(ctx, POLY_FLOAT32, (int64_t[]){1, 3, 2}, 3);

  PolyExpr r = pe_scaled_dot_product_attention(q, k, v, NULL, 1);
  ASSERT_TRUE(pe_valid(r));

  float dq[] = {1,0, 0,1, 1,1};
  float dk[] = {1,0, 0,1, 1,1};
  float dv[] = {1,0, 0,1, 1,1};
  float dout[6] = {0};
  PolyExpr leaves[] = {q, k, v};
  float *leaf_data[] = {dq, dk, dv};
  ASSERT_INT_EQ(realize_expr(r, out, dout, leaves, leaf_data, 3), 0);

  ASSERT_FLOAT_EQ(dout[0], 1.0f, 1e-3);
  ASSERT_FLOAT_EQ(dout[1], 0.0f, 1e-3);
  ASSERT_FLOAT_EQ(dout[2], 0.33023846f, 1e-3);
  ASSERT_FLOAT_EQ(dout[3], 0.66976154f, 1e-3);
  ASSERT_FLOAT_EQ(dout[4], 0.7517449f, 1e-3);
  ASSERT_FLOAT_EQ(dout[5], 0.7517449f, 1e-3);
  poly_ctx_destroy(ctx);
  PASS();
}

/* ── repeat_interleave ────────────────────────────────────────────────── */

TEST(pe, repeat_interleave_e2e) {
  /* Reference (tinygrad): [[1,1,2,2,3,3], [4,4,5,5,6,6]] */
  PolyCtx *ctx = poly_ctx_new();
  PolyExpr x = pe_buffer(ctx, POLY_FLOAT32, (int64_t[]){2, 3}, 2);
  PolyExpr out = pe_buffer(ctx, POLY_FLOAT32, (int64_t[]){2, 6}, 2);

  PolyExpr r = pe_repeat_interleave(x, 2, 1);
  ASSERT_TRUE(pe_valid(r));
  ASSERT_INT_EQ(r.ndim, 2);
  ASSERT_INT_EQ(r.shape[0], 2);
  ASSERT_INT_EQ(r.shape[1], 6);

  float dx[] = {1,2,3, 4,5,6};
  float dout[12] = {0};
  PolyExpr leaves[] = {x};
  float *leaf_data[] = {dx};
  ASSERT_INT_EQ(realize_expr(r, out, dout, leaves, leaf_data, 1), 0);

  float expected[] = {1,1,2,2,3,3, 4,4,5,5,6,6};
  for (int i = 0; i < 12; i++)
    ASSERT_FLOAT_EQ(dout[i], expected[i], 1e-6);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, repeat_interleave_dim0) {
  /* [1,2,3] repeat 3 on dim 0 -> [1,1,1,2,2,2,3,3,3] */
  PolyCtx *ctx = poly_ctx_new();
  PolyExpr x = pe_buffer(ctx, POLY_FLOAT32, (int64_t[]){3}, 1);
  PolyExpr out = pe_buffer(ctx, POLY_FLOAT32, (int64_t[]){9}, 1);

  PolyExpr r = pe_repeat_interleave(x, 3, 0);
  ASSERT_INT_EQ(r.shape[0], 9);

  float dx[] = {1, 2, 3};
  float dout[9] = {0};
  PolyExpr leaves[] = {x};
  float *leaf_data[] = {dx};
  ASSERT_INT_EQ(realize_expr(r, out, dout, leaves, leaf_data, 1), 0);

  float expected[] = {1,1,1, 2,2,2, 3,3,3};
  for (int i = 0; i < 9; i++)
    ASSERT_FLOAT_EQ(dout[i], expected[i], 1e-6);
  poly_ctx_destroy(ctx);
  PASS();
}

/* ── chunk ────────────────────────────────────────────────────────────── */

TEST(pe, chunk_e2e) {
  /* Reference (tinygrad): [1,2,3] and [4,5,6] */
  PolyCtx *ctx = poly_ctx_new();
  PolyExpr x = pe_buffer(ctx, POLY_FLOAT32, (int64_t[]){6}, 1);

  PolyExpr chunks[2];
  pe_chunk(x, 2, 0, chunks);
  ASSERT_TRUE(pe_valid(chunks[0]));
  ASSERT_TRUE(pe_valid(chunks[1]));
  ASSERT_INT_EQ(chunks[0].shape[0], 3);
  ASSERT_INT_EQ(chunks[1].shape[0], 3);

  /* Realize first chunk */
  PolyExpr out0 = pe_buffer(ctx, POLY_FLOAT32, (int64_t[]){3}, 1);
  float dx[] = {1,2,3,4,5,6};
  float dout0[3] = {0};
  PolyExpr leaves0[] = {x};
  float *ld0[] = {dx};
  ASSERT_INT_EQ(realize_expr(chunks[0], out0, dout0, leaves0, ld0, 1), 0);
  ASSERT_FLOAT_EQ(dout0[0], 1.0f, 1e-6);
  ASSERT_FLOAT_EQ(dout0[1], 2.0f, 1e-6);
  ASSERT_FLOAT_EQ(dout0[2], 3.0f, 1e-6);

  /* Realize second chunk */
  PolyExpr out1 = pe_buffer(ctx, POLY_FLOAT32, (int64_t[]){3}, 1);
  float dout1[3] = {0};
  PolyExpr leaves1[] = {x};
  float *ld1[] = {dx};
  ASSERT_INT_EQ(realize_expr(chunks[1], out1, dout1, leaves1, ld1, 1), 0);
  ASSERT_FLOAT_EQ(dout1[0], 4.0f, 1e-6);
  ASSERT_FLOAT_EQ(dout1[1], 5.0f, 1e-6);
  ASSERT_FLOAT_EQ(dout1[2], 6.0f, 1e-6);

  poly_ctx_destroy(ctx);
  PASS();
}

/* ── RoPE ─────────────────────────────────────────────────────────────── */

TEST(pe, rope_e2e) {
  /* Reference (tinygrad):
   * x = [[[[1,0,0,1], [1,0,0,1], [1,0,0,1]]]] shape (1,1,3,4)
   * Result: [[[[1.0, 0.0, 0.0, 1.0],
   *            [0.5403, -0.01, 0.8415, 0.9999],
   *            [-0.4161, -0.02, 0.9093, 0.9998]]]]
   */
  PolyCtx *ctx = poly_ctx_new();
  int64_t dim = 4, seq = 3;
  int64_t half = dim / 2;

  /* Precompute freqs: for each (pos, freq_idx): pos * 1/(theta^(2*idx/dim)) */
  float cos_data[6], sin_data[6];  /* 3 x 2 */
  double theta = 10000.0;
  for (int64_t i = 0; i < seq; i++) {
    for (int64_t j = 0; j < half; j++) {
      double freq = 1.0 / pow(theta, (double)(2 * j) / (double)dim);
      double angle = (double)i * freq;
      cos_data[i * half + j] = (float)cos(angle);
      sin_data[i * half + j] = (float)sin(angle);
    }
  }

  /* Need to broadcast freqs to (1, 1, 3, 2) to match x's (1, 1, 3, 4/2) */
  PolyExpr x = pe_buffer(ctx, POLY_FLOAT32, (int64_t[]){1, 1, 3, 4}, 4);
  PolyExpr fc_buf = pe_buffer(ctx, POLY_FLOAT32, (int64_t[]){3, 2}, 2);
  PolyExpr fs_buf = pe_buffer(ctx, POLY_FLOAT32, (int64_t[]){3, 2}, 2);

  /* Reshape freqs to (1, 1, 3, 2) for broadcasting */
  PolyExpr fc = pe_reshape(fc_buf, (int64_t[]){1, 1, 3, 2}, 4);
  PolyExpr fs = pe_reshape(fs_buf, (int64_t[]){1, 1, 3, 2}, 4);

  PolyExpr r = pe_rope(x, fc, fs);
  ASSERT_TRUE(pe_valid(r));
  ASSERT_INT_EQ(r.ndim, 4);
  ASSERT_INT_EQ(r.shape[3], 4);

  PolyExpr out = pe_buffer(ctx, POLY_FLOAT32, (int64_t[]){1, 1, 3, 4}, 4);
  float dx[] = {1,0,0,1, 1,0,0,1, 1,0,0,1};
  float dout[12] = {0};
  PolyExpr leaves[] = {x, fc_buf, fs_buf};
  float *leaf_data[] = {dx, cos_data, sin_data};
  ASSERT_INT_EQ(realize_expr(r, out, dout, leaves, leaf_data, 3), 0);

  /* pos=0: cos=[1,1], sin=[0,0] -> [1*1-0*0, 0*1-1*0, 0*1+1*0, 1*1+0*0] = [1,0,0,1] */
  ASSERT_FLOAT_EQ(dout[0], 1.0f, 1e-3);
  ASSERT_FLOAT_EQ(dout[1], 0.0f, 1e-3);
  ASSERT_FLOAT_EQ(dout[2], 0.0f, 1e-3);
  ASSERT_FLOAT_EQ(dout[3], 1.0f, 1e-3);

  /* pos=1: [0.5403, -0.01, 0.8415, 0.9999] */
  ASSERT_FLOAT_EQ(dout[4], 0.5403f, 1e-3);
  ASSERT_FLOAT_EQ(dout[5], -0.01f, 1e-2);
  ASSERT_FLOAT_EQ(dout[6], 0.8415f, 1e-3);
  ASSERT_FLOAT_EQ(dout[7], 0.9999f, 1e-3);

  /* pos=2: [-0.4161, -0.02, 0.9093, 0.9998] */
  ASSERT_FLOAT_EQ(dout[8], -0.4161f, 1e-3);
  ASSERT_FLOAT_EQ(dout[9], -0.02f, 1e-2);
  ASSERT_FLOAT_EQ(dout[10], 0.9093f, 1e-3);
  ASSERT_FLOAT_EQ(dout[11], 0.9998f, 1e-3);

  poly_ctx_destroy(ctx);
  PASS();
}

/* ═══════════════════════════════════════════════════════════════════════ */
/*  NN Layer tests                                                        */
/* ═══════════════════════════════════════════════════════════════════════ */

TEST(pe, nn_linear_shape) {
  PolyCtx *ctx = poly_ctx_new();
  PeLinear l = pe_nn_linear(ctx, 3, 4, 1, 42);
  ASSERT_TRUE(pe_valid(l.weight));
  ASSERT_INT_EQ(l.weight.ndim, 2);
  ASSERT_INT_EQ(l.weight.shape[0], 4);
  ASSERT_INT_EQ(l.weight.shape[1], 3);
  ASSERT_TRUE(l.has_bias);
  ASSERT_TRUE(pe_valid(l.bias));
  ASSERT_INT_EQ(l.bias.shape[0], 4);

  PolyExpr x = pe_buffer(ctx, POLY_FLOAT32, (int64_t[]){2, 3}, 2);
  PolyExpr out = pe_nn_linear_forward(&l, x);
  ASSERT_TRUE(pe_valid(out));
  ASSERT_INT_EQ(out.ndim, 2);
  ASSERT_INT_EQ(out.shape[0], 2);
  ASSERT_INT_EQ(out.shape[1], 4);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, nn_linear_no_bias) {
  PolyCtx *ctx = poly_ctx_new();
  PeLinear l = pe_nn_linear(ctx, 5, 3, 0, 42);
  ASSERT_TRUE(!l.has_bias);
  ASSERT_TRUE(!pe_valid(l.bias));

  PolyExpr params[4];
  int np = pe_nn_linear_params(&l, params, 4);
  ASSERT_INT_EQ(np, 1);  /* weight only */
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, nn_rmsnorm_shape) {
  PolyCtx *ctx = poly_ctx_new();
  PeRMSNorm l = pe_nn_rmsnorm(ctx, 64, 1e-5, 42);
  ASSERT_INT_EQ(l.dim, 64);
  ASSERT_TRUE(pe_valid(l.weight));
  ASSERT_INT_EQ(l.weight.shape[0], 64);

  PolyExpr x = pe_buffer(ctx, POLY_FLOAT32, (int64_t[]){2, 8, 64}, 3);
  PolyExpr out = pe_nn_rmsnorm_forward(&l, x);
  ASSERT_TRUE(pe_valid(out));
  ASSERT_INT_EQ(out.ndim, 3);
  ASSERT_INT_EQ(out.shape[2], 64);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, nn_embedding_shape) {
  PolyCtx *ctx = poly_ctx_new();
  PeEmbedding l = pe_nn_embedding(ctx, 100, 32, 42);
  ASSERT_INT_EQ(l.vocab_size, 100);
  ASSERT_INT_EQ(l.embed_dim, 32);
  ASSERT_TRUE(pe_valid(l.weight));
  ASSERT_INT_EQ(l.weight.shape[0], 100);
  ASSERT_INT_EQ(l.weight.shape[1], 32);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, nn_attention_shape) {
  PolyCtx *ctx = poly_ctx_new();
  /* 4 heads, 4 kv heads (no GQA), dim=32 */
  PeAttention a = pe_nn_attention(ctx, 32, 4, 4, 0, 42);
  ASSERT_INT_EQ(a.n_heads, 4);
  ASSERT_INT_EQ(a.n_kv_heads, 4);
  ASSERT_INT_EQ(a.head_dim, 8);
  ASSERT_INT_EQ(a.dim, 32);

  /* Count params: 4 linear layers, no bias = 4 weights */
  PolyExpr params[16];
  int np = pe_nn_attention_params(&a, params, 16);
  ASSERT_INT_EQ(np, 4);

  /* Forward shape: (1, 4, 32) -> (1, 4, 32) */
  PolyExpr x = pe_buffer(ctx, POLY_FLOAT32, (int64_t[]){1, 4, 32}, 3);
  PolyExpr out = pe_nn_attention_forward(&a, x, NULL, NULL, NULL, 1);
  ASSERT_TRUE(pe_valid(out));
  ASSERT_INT_EQ(out.ndim, 3);
  ASSERT_INT_EQ(out.shape[0], 1);
  ASSERT_INT_EQ(out.shape[1], 4);
  ASSERT_INT_EQ(out.shape[2], 32);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, nn_attention_gqa_shape) {
  PolyCtx *ctx = poly_ctx_new();
  /* 8 heads, 2 kv heads (GQA 4:1) */
  PeAttention a = pe_nn_attention(ctx, 64, 8, 2, 0, 42);
  ASSERT_INT_EQ(a.n_heads, 8);
  ASSERT_INT_EQ(a.n_kv_heads, 2);
  ASSERT_INT_EQ(a.head_dim, 8);

  /* wq: (64, 64), wk: (16, 64), wv: (16, 64), wo: (64, 64) */
  ASSERT_INT_EQ(a.wq.weight.shape[0], 64);
  ASSERT_INT_EQ(a.wk.weight.shape[0], 16);
  ASSERT_INT_EQ(a.wv.weight.shape[0], 16);
  ASSERT_INT_EQ(a.wo.weight.shape[0], 64);

  PolyExpr x = pe_buffer(ctx, POLY_FLOAT32, (int64_t[]){1, 4, 64}, 3);
  PolyExpr out = pe_nn_attention_forward(&a, x, NULL, NULL, NULL, 1);
  ASSERT_TRUE(pe_valid(out));
  ASSERT_INT_EQ(out.shape[0], 1);
  ASSERT_INT_EQ(out.shape[1], 4);
  ASSERT_INT_EQ(out.shape[2], 64);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, nn_params_collection) {
  PolyCtx *ctx = poly_ctx_new();
  PeLinear lin = pe_nn_linear(ctx, 8, 4, 1, 1);    /* 2 params */
  PeRMSNorm rms = pe_nn_rmsnorm(ctx, 4, 1e-5, 2);  /* 1 param */
  PeEmbedding emb = pe_nn_embedding(ctx, 10, 4, 3); /* 1 param */

  PolyExpr params[16];
  int n = 0;
  n += pe_nn_linear_params(&lin, params + n, 16 - n);
  n += pe_nn_rmsnorm_params(&rms, params + n, 16 - n);
  n += pe_nn_embedding_params(&emb, params + n, 16 - n);
  ASSERT_INT_EQ(n, 4);
  poly_ctx_destroy(ctx);
  PASS();
}

/* ── GroupNorm ─────────────────────────────────────────────────────────── */

TEST(pe, nn_groupnorm_shape) {
  PolyCtx *ctx = poly_ctx_new();
  PeGroupNorm gn = pe_nn_groupnorm(ctx, 2, 6, 1e-5, 1, 42);
  ASSERT_INT_EQ(gn.num_groups, 2);
  ASSERT_INT_EQ(gn.num_channels, 6);
  ASSERT_TRUE(gn.has_affine);
  ASSERT_TRUE(pe_valid(gn.weight));
  ASSERT_TRUE(pe_valid(gn.bias));

  PolyExpr params[4];
  int np = pe_nn_groupnorm_params(&gn, params, 4);
  ASSERT_INT_EQ(np, 2);

  PolyExpr x = pe_buffer(ctx, POLY_FLOAT32, (int64_t[]){1, 6, 2}, 3);
  PolyExpr out = pe_nn_groupnorm_forward(&gn, x);
  ASSERT_TRUE(pe_valid(out));
  ASSERT_INT_EQ(out.ndim, 3);
  ASSERT_INT_EQ(out.shape[0], 1);
  ASSERT_INT_EQ(out.shape[1], 6);
  ASSERT_INT_EQ(out.shape[2], 2);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, nn_groupnorm_e2e) {
  /* Reference (tinygrad, weight=1, bias=0):
   * GroupNorm(2, 6) on (1, 6, 2):
   * [[[-1.4638, -0.8783], [-0.2928, 0.2928], [0.8783, 1.4638],
   *   [-1.4638, -0.8783], [-0.2928, 0.2928], [0.8783, 1.4638]]] */
  PolyCtx *ctx = poly_ctx_new();
  PeGroupNorm gn = pe_nn_groupnorm(ctx, 2, 6, 1e-5, 1, 42);
  PolyExpr x = pe_buffer(ctx, POLY_FLOAT32, (int64_t[]){1, 6, 2}, 3);
  PolyExpr outb = pe_buffer(ctx, POLY_FLOAT32, (int64_t[]){1, 6, 2}, 3);
  PolyExpr r = pe_nn_groupnorm_forward(&gn, x);

  float dx[] = {1,2,3,4,5,6, 7,8,9,10,11,12};
  float dw[] = {1,1,1,1,1,1};
  float db[] = {0,0,0,0,0,0};
  float dout[12] = {0};
  PolyExpr leaves[] = {x, gn.weight, gn.bias};
  float *ld[] = {dx, dw, db};
  ASSERT_INT_EQ(realize_expr(r, outb, dout, leaves, ld, 3), 0);

  ASSERT_FLOAT_EQ(dout[0], -1.4638f, 1e-3);
  ASSERT_FLOAT_EQ(dout[1], -0.8783f, 1e-3);
  ASSERT_FLOAT_EQ(dout[2], -0.2928f, 1e-3);
  ASSERT_FLOAT_EQ(dout[3],  0.2928f, 1e-3);
  ASSERT_FLOAT_EQ(dout[4],  0.8783f, 1e-3);
  ASSERT_FLOAT_EQ(dout[5],  1.4638f, 1e-3);
  /* Group 2 (channels 3-5) should be same pattern */
  ASSERT_FLOAT_EQ(dout[6], -1.4638f, 1e-3);
  ASSERT_FLOAT_EQ(dout[11], 1.4638f, 1e-3);
  poly_ctx_destroy(ctx);
  PASS();
}

/* ── Loss functions ───────────────────────────────────────────────────── */

TEST(pe, mse_loss_e2e) {
  /* mse([1,2,3], [1,2,3]) = 0 */
  PolyCtx *ctx = poly_ctx_new();
  PolyExpr pred = pe_buffer(ctx, POLY_FLOAT32, (int64_t[]){3}, 1);
  PolyExpr tgt = pe_buffer(ctx, POLY_FLOAT32, (int64_t[]){3}, 1);
  PolyExpr out = pe_buffer(ctx, POLY_FLOAT32, (int64_t[]){1}, 1);
  PolyExpr r = pe_mse_loss(pred, tgt);
  ASSERT_TRUE(pe_valid(r));

  float dp[] = {1, 2, 3}, dt[] = {1, 2, 3};
  float dout[1] = {999};
  PolyExpr leaves[] = {pred, tgt};
  float *ld[] = {dp, dt};
  ASSERT_INT_EQ(realize_expr(r, out, dout, leaves, ld, 2), 0);
  ASSERT_FLOAT_EQ(dout[0], 0.0f, 1e-6);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, mse_loss_nonzero) {
  /* mse([1,2,3], [4,5,6]) = mean([9,9,9]) = 9 */
  PolyCtx *ctx = poly_ctx_new();
  PolyExpr pred = pe_buffer(ctx, POLY_FLOAT32, (int64_t[]){3}, 1);
  PolyExpr tgt = pe_buffer(ctx, POLY_FLOAT32, (int64_t[]){3}, 1);
  PolyExpr out = pe_buffer(ctx, POLY_FLOAT32, (int64_t[]){1}, 1);
  PolyExpr r = pe_mse_loss(pred, tgt);

  float dp[] = {1, 2, 3}, dt[] = {4, 5, 6};
  float dout[1] = {0};
  PolyExpr leaves[] = {pred, tgt};
  float *ld[] = {dp, dt};
  ASSERT_INT_EQ(realize_expr(r, out, dout, leaves, ld, 2), 0);
  ASSERT_FLOAT_EQ(dout[0], 9.0f, 1e-4);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, mae_loss_e2e) {
  /* mae([1,2,3], [4,6,3]) = mean([3,4,0]) = 7/3 = 2.333 */
  PolyCtx *ctx = poly_ctx_new();
  PolyExpr pred = pe_buffer(ctx, POLY_FLOAT32, (int64_t[]){3}, 1);
  PolyExpr tgt = pe_buffer(ctx, POLY_FLOAT32, (int64_t[]){3}, 1);
  PolyExpr out = pe_buffer(ctx, POLY_FLOAT32, (int64_t[]){1}, 1);
  PolyExpr r = pe_mae_loss(pred, tgt);

  float dp[] = {1, 2, 3}, dt[] = {4, 6, 3};
  float dout[1] = {0};
  PolyExpr leaves[] = {pred, tgt};
  float *ld[] = {dp, dt};
  ASSERT_INT_EQ(realize_expr(r, out, dout, leaves, ld, 2), 0);
  ASSERT_FLOAT_EQ(dout[0], 7.0f / 3.0f, 1e-4);
  poly_ctx_destroy(ctx);
  PASS();
}

/* ── Argmax / Argmin ──────────────────────────────────────────────────── */

TEST(pe, argmax_1d) {
  /* Reference (tinygrad): argmax([1,5,3,2,4]) = 1 */
  PolyCtx *ctx = poly_ctx_new();
  PolyExpr x = pe_buffer(ctx, POLY_FLOAT32, (int64_t[]){5}, 1);
  PolyExpr out = pe_buffer(ctx, POLY_FLOAT32, (int64_t[]){1}, 1);
  PolyExpr r = pe_argmax(x, 0);
  ASSERT_TRUE(pe_valid(r));

  /* argmax returns int32, cast to float for output buffer */
  PolyExpr r_f = pe_cast(r, POLY_FLOAT32);
  float dx[] = {1.0f, 5.0f, 3.0f, 2.0f, 4.0f};
  float dout[1] = {0};
  PolyExpr leaves[] = {x};
  float *ld[] = {dx};
  ASSERT_INT_EQ(realize_expr(r_f, out, dout, leaves, ld, 1), 0);
  ASSERT_FLOAT_EQ(dout[0], 1.0f, 1e-4);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, argmax_2d_axis1) {
  /* Reference (tinygrad): argmax([[1,5,3],[4,2,6]], axis=1) = [1, 2] */
  PolyCtx *ctx = poly_ctx_new();
  PolyExpr x = pe_buffer(ctx, POLY_FLOAT32, (int64_t[]){2, 3}, 2);
  PolyExpr out = pe_buffer(ctx, POLY_FLOAT32, (int64_t[]){2}, 1);
  PolyExpr r = pe_argmax(x, 1);
  ASSERT_TRUE(pe_valid(r));

  PolyExpr r_f = pe_cast(r, POLY_FLOAT32);
  float dx[] = {1,5,3, 4,2,6};
  float dout[2] = {0};
  PolyExpr leaves[] = {x};
  float *ld[] = {dx};
  ASSERT_INT_EQ(realize_expr(r_f, out, dout, leaves, ld, 1), 0);
  ASSERT_FLOAT_EQ(dout[0], 1.0f, 1e-4);  /* index of 5 in [1,5,3] */
  ASSERT_FLOAT_EQ(dout[1], 2.0f, 1e-4);  /* index of 6 in [4,2,6] */
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, argmax_2d_axis0) {
  /* Reference (tinygrad): argmax([[1,5,3],[4,2,6]], axis=0) = [1, 0, 1] */
  PolyCtx *ctx = poly_ctx_new();
  PolyExpr x = pe_buffer(ctx, POLY_FLOAT32, (int64_t[]){2, 3}, 2);
  PolyExpr out = pe_buffer(ctx, POLY_FLOAT32, (int64_t[]){3}, 1);
  PolyExpr r = pe_argmax(x, 0);
  ASSERT_TRUE(pe_valid(r));

  PolyExpr r_f = pe_cast(r, POLY_FLOAT32);
  float dx[] = {1,5,3, 4,2,6};
  float dout[3] = {0};
  PolyExpr leaves[] = {x};
  float *ld[] = {dx};
  ASSERT_INT_EQ(realize_expr(r_f, out, dout, leaves, ld, 1), 0);
  ASSERT_FLOAT_EQ(dout[0], 1.0f, 1e-4);  /* max of col 0: 4 at row 1 */
  ASSERT_FLOAT_EQ(dout[1], 0.0f, 1e-4);  /* max of col 1: 5 at row 0 */
  ASSERT_FLOAT_EQ(dout[2], 1.0f, 1e-4);  /* max of col 2: 6 at row 1 */
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, argmin_1d) {
  /* argmin([1,5,3,2,4]) = 0 */
  PolyCtx *ctx = poly_ctx_new();
  PolyExpr x = pe_buffer(ctx, POLY_FLOAT32, (int64_t[]){5}, 1);
  PolyExpr out = pe_buffer(ctx, POLY_FLOAT32, (int64_t[]){1}, 1);
  PolyExpr r = pe_argmin(x, 0);
  PolyExpr r_f = pe_cast(r, POLY_FLOAT32);

  float dx[] = {1.0f, 5.0f, 3.0f, 2.0f, 4.0f};
  float dout[1] = {0};
  PolyExpr leaves[] = {x};
  float *ld[] = {dx};
  ASSERT_INT_EQ(realize_expr(r_f, out, dout, leaves, ld, 1), 0);
  ASSERT_FLOAT_EQ(dout[0], 0.0f, 1e-4);
  poly_ctx_destroy(ctx);
  PASS();
}

/* ── Softmax via PolyExpr ─────────────────────────────────────────────── */

TEST(pe, softmax_e2e) {
  PolyCtx *ctx = poly_ctx_new();
  PolyExpr x = pe_buffer(ctx, POLY_FLOAT32, (int64_t[]){3}, 1);
  PolyExpr out = pe_buffer(ctx, POLY_FLOAT32, (int64_t[]){3}, 1);
  PolyExpr r = pe_softmax(x, 0);

  float dx[] = {1.0f, 2.0f, 3.0f};
  float dout[3] = {0};
  PolyExpr leaves[] = {x};
  float *leaf_data[] = {dx};
  ASSERT_INT_EQ(realize_expr(r, out, dout, leaves, leaf_data, 1), 0);

  /* softmax([1,2,3]) = [0.0900, 0.2447, 0.6652] */
  ASSERT_FLOAT_EQ(dout[0], 0.0900f, 1e-3);
  ASSERT_FLOAT_EQ(dout[1], 0.2447f, 1e-3);
  ASSERT_FLOAT_EQ(dout[2], 0.6652f, 1e-3);
  poly_ctx_destroy(ctx);
  PASS();
}

/* ── Linear via PolyExpr ──────────────────────────────────────────────── */

TEST(pe, linear_e2e) {
  PolyCtx *ctx = poly_ctx_new();
  /* x: (2, 3), weight: (4, 3), bias: (4,) -> out: (2, 4) */
  PolyExpr x = pe_buffer(ctx, POLY_FLOAT32, (int64_t[]){2, 3}, 2);
  PolyExpr w = pe_buffer(ctx, POLY_FLOAT32, (int64_t[]){4, 3}, 2);
  PolyExpr b = pe_buffer(ctx, POLY_FLOAT32, (int64_t[]){4}, 1);
  PolyExpr out = pe_buffer(ctx, POLY_FLOAT32, (int64_t[]){2, 4}, 2);

  PolyExpr r = pe_linear(x, w, &b);
  ASSERT_TRUE(pe_valid(r));
  ASSERT_INT_EQ(r.ndim, 2);
  ASSERT_INT_EQ(r.shape[0], 2);
  ASSERT_INT_EQ(r.shape[1], 4);

  float dx[] = {1,0,0, 0,1,0};
  float dw[] = {1,2,3, 4,5,6, 7,8,9, 10,11,12};  /* rows of weight */
  float db[] = {0.1f, 0.2f, 0.3f, 0.4f};
  float dout[8] = {0};
  PolyExpr leaves[] = {x, w, b};
  float *leaf_data[] = {dx, dw, db};
  ASSERT_INT_EQ(realize_expr(r, out, dout, leaves, leaf_data, 3), 0);

  /* x[0]=[1,0,0]: out = w[:,0] + bias = [1.1, 4.2, 7.3, 10.4] */
  ASSERT_FLOAT_EQ(dout[0], 1.1f, 1e-4);
  ASSERT_FLOAT_EQ(dout[1], 4.2f, 1e-4);
  ASSERT_FLOAT_EQ(dout[2], 7.3f, 1e-4);
  ASSERT_FLOAT_EQ(dout[3], 10.4f, 1e-4);
  poly_ctx_destroy(ctx);
  PASS();
}
