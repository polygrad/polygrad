/*
 * test_tensor.c -- Tests for shape-on-UOp and v2 composed ops
 *
 * Reference values verified against tinygrad (conda env 'tiny').
 * All tests use the v2 API (shape read from UOp) or raw UOp construction.
 * No PolyExpr dependency.
 */

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "test_harness.h"
#include "../src/polygrad.h"
#include "../src/frontend.h"
#include "../src/engine/schedule.h"
#include "../src/schedule/rangeify.h"
#include "../src/nn.h"
#include "../src/tensor.h"
#include "../src/codegen.h"

/* Helper: realize a UOp into a float array */

static int realize_uop(
    PolyCtx *ctx,
    PolyUOp *val,
    PolyUOp *out_buf,
    float *out_data,
    PolyUOp **leaf_bufs,
    float **leaf_datas,
    int n_leaves
) {
  PolyUOp *store = poly_store_val(ctx, out_buf, val);
  PolyUOp *sink = poly_sink1(ctx, store);
  int n = n_leaves + 1;
  PolyUOp *bufs[64];
  void *datas[64];
  for (int i = 0; i < n_leaves; i++) {
    bufs[i] = leaf_bufs[i];
    datas[i] = leaf_datas[i];
  }
  bufs[n_leaves] = out_buf;
  datas[n_leaves] = out_data;
  return poly_realize_with_bindings_flat(ctx, sink, bufs, datas, n);
}

/* Helper: make a shaped buffer (RESHAPE(BUFFER, shape)) */

static PolyUOp *make_buf(PolyCtx *ctx, const int64_t *shape, int ndim) {
  int64_t numel = 1;
  for (int i = 0; i < ndim; i++)
    numel *= shape[i];
  PolyUOp *buf = poly_buffer_f32(ctx, numel);
  if (ndim > 1) return poly_reshape(ctx, buf, (int64_t *)shape, ndim);
  return buf;
}

/* Get underlying BUFFER from possibly-reshaped UOp */
static PolyUOp *base_buf(PolyUOp *u) {
  while (u->op == POLY_OP_RESHAPE && u->n_src > 0)
    u = u->src[0];
  return u;
}

/* Structural constructor tests that count LOAD/RANGE/STORE ops need the
 * executable scheduled root rather than the earlier public kernel graph. */
static PolyUOp *single_scheduled_root(PolyCtx *ctx, PolyUOp *sink) {
  PolySchedule *schedule = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  if (!schedule) return NULL;
  if (schedule->n_items != 1 || !schedule->items[0].root) {
    poly_schedule_free(schedule);
    return NULL;
  }
  PolyUOp *root = schedule->items[0].root;
  poly_schedule_free(schedule);
  return root;
}

/* ═══════════════════════════════════════════════════════════════════════ */
/*  v2 composed op e2e tests (tinygrad-verified reference values)         */
/* ═══════════════════════════════════════════════════════════════════════ */

TEST(pe, rmsnorm_e2e) {
  /* Reference: [[0.4629, 0.9258, 1.3887], [0.7895, 0.9869, 1.1843]] */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = make_buf(ctx, (int64_t[]){2, 3}, 2);
  PolyUOp *w = poly_buffer_f32(ctx, 3);
  PolyUOp *out_buf = poly_buffer_f32(ctx, 6);
  PolyUOp *r = poly_rmsnorm_apply(ctx, x, w, 1e-5);
  ASSERT_NOT_NULL(r);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, r), 2);

  float dx[] = {1, 2, 3, 4, 5, 6}, dw[] = {1, 1, 1}, dout[6] = {0};
  PolyUOp *leaves[] = {base_buf(x), w};
  float *ld[] = {dx, dw};
  ASSERT_INT_EQ(realize_uop(ctx, r, out_buf, dout, leaves, ld, 2), 0);
  ASSERT_FLOAT_EQ(dout[0], 0.46290955f, 1e-4);
  ASSERT_FLOAT_EQ(dout[3], 0.78954184f, 1e-4);
  ASSERT_FLOAT_EQ(dout[5], 1.18431280f, 1e-4);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, sdpa_e2e) {
  /* Reference: [[[1.6605, 2.6605], [2.3395, 3.3395]]] */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *q = make_buf(ctx, (int64_t[]){1, 2, 2}, 3);
  PolyUOp *k = make_buf(ctx, (int64_t[]){1, 2, 2}, 3);
  PolyUOp *v = make_buf(ctx, (int64_t[]){1, 2, 2}, 3);
  PolyUOp *out_buf = poly_buffer_f32(ctx, 4);
  PolyUOp *r = poly_sdpa(ctx, q, k, v, NULL, 0);
  ASSERT_NOT_NULL(r);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, r), 3);

  float dq[] = {1, 0, 0, 1}, dk[] = {1, 0, 0, 1}, dv[] = {1, 2, 3, 4}, dout[4] = {0};
  PolyUOp *leaves[] = {base_buf(q), base_buf(k), base_buf(v)};
  float *ld[] = {dq, dk, dv};
  ASSERT_INT_EQ(realize_uop(ctx, r, out_buf, dout, leaves, ld, 3), 0);
  ASSERT_FLOAT_EQ(dout[0], 1.6604769f, 1e-3);
  ASSERT_FLOAT_EQ(dout[3], 3.3395231f, 1e-3);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, sdpa_causal_e2e) {
  /* Reference: [[[1.0, 0.0], [0.3302, 0.6698], [0.7517, 0.7517]]] */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *q = make_buf(ctx, (int64_t[]){1, 3, 2}, 3);
  PolyUOp *k = make_buf(ctx, (int64_t[]){1, 3, 2}, 3);
  PolyUOp *v = make_buf(ctx, (int64_t[]){1, 3, 2}, 3);
  PolyUOp *out_buf = poly_buffer_f32(ctx, 6);
  PolyUOp *r = poly_sdpa(ctx, q, k, v, NULL, 1);

  float dq[] = {1, 0, 0, 1, 1, 1}, dk[] = {1, 0, 0, 1, 1, 1}, dv[] = {1, 0, 0, 1, 1, 1};
  float dout[6] = {0};
  PolyUOp *leaves[] = {base_buf(q), base_buf(k), base_buf(v)};
  float *ld[] = {dq, dk, dv};
  ASSERT_INT_EQ(realize_uop(ctx, r, out_buf, dout, leaves, ld, 3), 0);
  ASSERT_FLOAT_EQ(dout[0], 1.0f, 1e-3);
  ASSERT_FLOAT_EQ(dout[2], 0.33023846f, 1e-3);
  ASSERT_FLOAT_EQ(dout[4], 0.7517449f, 1e-3);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, sdpa_single_token_multihead_returns_v) {
  /* For T=1, softmax(q @ k^T) is exactly 1, so SDPA must return v. */
  PolyCtx *ctx = poly_ctx_new();
  const int B = 1, H = 12, T = 1, D = 64;
  const int64_t shape[] = {B, H, T, D};
  const int64_t numel = (int64_t)B * H * T * D;

  PolyUOp *q = make_buf(ctx, (int64_t *)shape, 4);
  PolyUOp *k = make_buf(ctx, (int64_t *)shape, 4);
  PolyUOp *v = make_buf(ctx, (int64_t *)shape, 4);
  PolyUOp *out_buf = poly_buffer_f32(ctx, numel);
  PolyUOp *r = poly_sdpa(ctx, q, k, v, NULL, 0);
  ASSERT_NOT_NULL(r);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, r), 4);

  float *dq = calloc((size_t)numel, sizeof(float));
  float *dk = calloc((size_t)numel, sizeof(float));
  float *dv = calloc((size_t)numel, sizeof(float));
  float *dout = calloc((size_t)numel, sizeof(float));
  ASSERT_NOT_NULL(dq);
  ASSERT_NOT_NULL(dk);
  ASSERT_NOT_NULL(dv);
  ASSERT_NOT_NULL(dout);

  for (int64_t i = 0; i < numel; i++) {
    dq[i] = (float)((i % 17) - 8) * 0.25f;
    dk[i] = (float)((i % 13) - 6) * 0.5f;
    dv[i] = (float)(i % 101) * 0.1f - 5.0f;
  }

  PolyUOp *leaves[] = {base_buf(q), base_buf(k), base_buf(v)};
  float *ld[] = {dq, dk, dv};
  ASSERT_INT_EQ(realize_uop(ctx, r, out_buf, dout, leaves, ld, 3), 0);

  for (int64_t i = 0; i < numel; i++)
    ASSERT_FLOAT_EQ(dout[i], dv[i], 1e-4f);

  free(dq);
  free(dk);
  free(dv);
  free(dout);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, gpt2_qkv_v_path_single_token_layout) {
  /* GPT-2 qkv split path for T=1:
   * qkv: (B,T,3*D) -> v shrink -> reshape(B,T,H,hd) -> permute(B,H,T,hd)
   * must preserve the exact lane order of the final third of qkv.
   */
  PolyCtx *ctx = poly_ctx_new();
  const int B = 1, T = 1, H = 12, hd = 64, D = H * hd;
  const int64_t qkv_shape[] = {B, T, 3 * D};
  const int64_t numel = (int64_t)B * T * 3 * D;
  const int64_t out_numel = (int64_t)B * H * T * hd;

  PolyUOp *qkv = make_buf(ctx, (int64_t *)qkv_shape, 3);
  int64_t shrink_v[][2] = {{0, B}, {0, T}, {2 * D, 3 * D}};
  PolyUOp *v = poly_shrink(ctx, qkv, shrink_v, 3);
  ASSERT_NOT_NULL(v);

  int64_t mh[] = {B, T, H, hd};
  int64_t perm[] = {0, 2, 1, 3};
  PolyUOp *vp = poly_permute(ctx, poly_reshape(ctx, v, mh, 4), perm, 4);
  ASSERT_NOT_NULL(vp);

  PolyUOp *out_buf = poly_buffer_f32(ctx, out_numel);
  float *in = calloc((size_t)numel, sizeof(float));
  float *out = calloc((size_t)out_numel, sizeof(float));
  ASSERT_NOT_NULL(in);
  ASSERT_NOT_NULL(out);

  for (int64_t i = 0; i < numel; i++)
    in[i] = (float)i;

  PolyUOp *leaves[] = {base_buf(qkv)};
  float *ld[] = {in};
  ASSERT_INT_EQ(realize_uop(ctx, vp, out_buf, out, leaves, ld, 1), 0);

  for (int h = 0; h < H; h++) {
    for (int d = 0; d < hd; d++) {
      int64_t got_idx = ((int64_t)h * T + 0) * hd + d;
      int64_t src_idx = (int64_t)2 * D + (int64_t)h * hd + d;
      ASSERT_FLOAT_EQ(out[got_idx], in[src_idx], 1e-6f);
    }
  }

  free(in);
  free(out);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, gpt2_qkv_v_path_single_token_contiguous) {
  PolyCtx *ctx = poly_ctx_new();
  const int B = 1, T = 1, H = 12, hd = 64, D = H * hd;
  const int64_t qkv_shape[] = {B, T, 3 * D};
  const int64_t numel = (int64_t)B * T * 3 * D;
  const int64_t out_numel = (int64_t)B * H * T * hd;

  PolyUOp *qkv = make_buf(ctx, (int64_t *)qkv_shape, 3);
  int64_t shrink_v[][2] = {{0, B}, {0, T}, {2 * D, 3 * D}};
  PolyUOp *v = poly_shrink(ctx, qkv, shrink_v, 3);
  ASSERT_NOT_NULL(v);

  int64_t mh[] = {B, T, H, hd};
  int64_t perm[] = {0, 2, 1, 3};
  PolyUOp *vp = poly_permute(ctx, poly_reshape(ctx, v, mh, 4), perm, 4);
  ASSERT_NOT_NULL(vp);
  vp = poly_contiguous(ctx, vp);
  ASSERT_NOT_NULL(vp);

  PolyUOp *out_buf = poly_buffer_f32(ctx, out_numel);
  float *in = calloc((size_t)numel, sizeof(float));
  float *out = calloc((size_t)out_numel, sizeof(float));
  ASSERT_NOT_NULL(in);
  ASSERT_NOT_NULL(out);

  for (int64_t i = 0; i < numel; i++)
    in[i] = (float)i;

  PolyUOp *leaves[] = {base_buf(qkv)};
  float *ld[] = {in};
  ASSERT_INT_EQ(realize_uop(ctx, vp, out_buf, out, leaves, ld, 1), 0);

  for (int h = 0; h < H; h++) {
    for (int d = 0; d < hd; d++) {
      int64_t got_idx = ((int64_t)h * T + 0) * hd + d;
      int64_t src_idx = (int64_t)2 * D + (int64_t)h * hd + d;
      ASSERT_FLOAT_EQ(out[got_idx], in[src_idx], 1e-6f);
    }
  }

  free(in);
  free(out);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, gpt2_c_attn_linear_single_token_e2e) {
  PolyCtx *ctx = poly_ctx_new();
  const int B = 1, T = 1, IN = 768, OUT = 2304;
  const int64_t x_shape[] = {B, T, IN};
  const int64_t w_shape[] = {OUT, IN};
  const int64_t b_shape[] = {OUT};
  const int64_t out_numel = (int64_t)B * T * OUT;

  PolyUOp *x = make_buf(ctx, (int64_t *)x_shape, 3);
  PolyUOp *w = make_buf(ctx, (int64_t *)w_shape, 2);
  PolyUOp *b = make_buf(ctx, (int64_t *)b_shape, 1);
  PolyUOp *out_buf = poly_buffer_f32(ctx, out_numel);
  PolyUOp *r = poly_linear_apply(ctx, x, w, b);
  ASSERT_NOT_NULL(r);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, r), 3);

  float *dx = calloc((size_t)IN, sizeof(float));
  float *dw = calloc((size_t)OUT * (size_t)IN, sizeof(float));
  float *db = calloc((size_t)OUT, sizeof(float));
  float *dout = calloc((size_t)out_numel, sizeof(float));
  ASSERT_NOT_NULL(dx);
  ASSERT_NOT_NULL(dw);
  ASSERT_NOT_NULL(db);
  ASSERT_NOT_NULL(dout);

  for (int i = 0; i < IN; i++)
    dx[i] = (float)((i % 23) - 11) * 0.03125f;
  for (int o = 0; o < OUT; o++) {
    db[o] = (float)((o % 19) - 9) * 0.05f;
    for (int i = 0; i < IN; i++)
      dw[(size_t)o * (size_t)IN + (size_t)i] = (float)(((o * 7 + i * 3) % 29) - 14) * 0.0078125f;
  }

  PolyUOp *leaves[] = {base_buf(x), base_buf(w), base_buf(b)};
  float *ld[] = {dx, dw, db};
  ASSERT_INT_EQ(realize_uop(ctx, r, out_buf, dout, leaves, ld, 3), 0);

  for (int o = 0; o < OUT; o++) {
    double acc = db[o];
    for (int i = 0; i < IN; i++)
      acc += (double)dx[i] * (double)dw[(size_t)o * (size_t)IN + (size_t)i];
    ASSERT_FLOAT_EQ(dout[o], (float)acc, 1e-3f);
  }

  free(dx);
  free(dw);
  free(db);
  free(dout);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, gpt2_c_attn_linear_single_token_contiguous_e2e) {
  PolyCtx *ctx = poly_ctx_new();
  const int B = 1, T = 1, IN = 768, OUT = 2304;
  const int64_t x_shape[] = {B, T, IN};
  const int64_t w_shape[] = {OUT, IN};
  const int64_t b_shape[] = {OUT};
  const int64_t out_numel = (int64_t)B * T * OUT;

  PolyUOp *x = make_buf(ctx, (int64_t *)x_shape, 3);
  PolyUOp *w = make_buf(ctx, (int64_t *)w_shape, 2);
  PolyUOp *b = make_buf(ctx, (int64_t *)b_shape, 1);
  PolyUOp *out_buf = poly_buffer_f32(ctx, out_numel);
  PolyUOp *r = poly_contiguous(ctx, poly_linear_apply(ctx, x, w, b));
  ASSERT_NOT_NULL(r);

  float *dx = calloc((size_t)IN, sizeof(float));
  float *dw = calloc((size_t)OUT * (size_t)IN, sizeof(float));
  float *db = calloc((size_t)OUT, sizeof(float));
  float *dout = calloc((size_t)out_numel, sizeof(float));
  ASSERT_NOT_NULL(dx);
  ASSERT_NOT_NULL(dw);
  ASSERT_NOT_NULL(db);
  ASSERT_NOT_NULL(dout);

  for (int i = 0; i < IN; i++)
    dx[i] = (float)((i % 23) - 11) * 0.03125f;
  for (int o = 0; o < OUT; o++) {
    db[o] = (float)((o % 19) - 9) * 0.05f;
    for (int i = 0; i < IN; i++)
      dw[(size_t)o * (size_t)IN + (size_t)i] = (float)(((o * 7 + i * 3) % 29) - 14) * 0.0078125f;
  }

  PolyUOp *leaves[] = {base_buf(x), base_buf(w), base_buf(b)};
  float *ld[] = {dx, dw, db};
  ASSERT_INT_EQ(realize_uop(ctx, r, out_buf, dout, leaves, ld, 3), 0);

  for (int o = 0; o < OUT; o++) {
    double acc = db[o];
    for (int i = 0; i < IN; i++)
      acc += (double)dx[i] * (double)dw[(size_t)o * (size_t)IN + (size_t)i];
    ASSERT_FLOAT_EQ(dout[o], (float)acc, 1e-3f);
  }

  free(dx);
  free(dw);
  free(db);
  free(dout);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, gpt2_c_proj_linear_single_token_e2e) {
  PolyCtx *ctx = poly_ctx_new();
  const int B = 1, T = 1, IN = 768, OUT = 768;
  const int64_t x_shape[] = {B, T, IN};
  const int64_t w_shape[] = {OUT, IN};
  const int64_t b_shape[] = {OUT};
  const int64_t out_numel = (int64_t)B * T * OUT;

  PolyUOp *x = make_buf(ctx, (int64_t *)x_shape, 3);
  PolyUOp *w = make_buf(ctx, (int64_t *)w_shape, 2);
  PolyUOp *b = make_buf(ctx, (int64_t *)b_shape, 1);
  PolyUOp *out_buf = poly_buffer_f32(ctx, out_numel);
  PolyUOp *r = poly_linear_apply(ctx, x, w, b);
  ASSERT_NOT_NULL(r);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, r), 3);

  float *dx = calloc((size_t)IN, sizeof(float));
  float *dw = calloc((size_t)OUT * (size_t)IN, sizeof(float));
  float *db = calloc((size_t)OUT, sizeof(float));
  float *dout = calloc((size_t)out_numel, sizeof(float));
  ASSERT_NOT_NULL(dx);
  ASSERT_NOT_NULL(dw);
  ASSERT_NOT_NULL(db);
  ASSERT_NOT_NULL(dout);

  for (int i = 0; i < IN; i++)
    dx[i] = (float)((i % 31) - 15) * 0.015625f;
  for (int o = 0; o < OUT; o++) {
    db[o] = (float)((o % 17) - 8) * 0.03125f;
    for (int i = 0; i < IN; i++)
      dw[(size_t)o * (size_t)IN + (size_t)i] = (float)(((o * 5 + i * 11) % 37) - 18) * 0.00390625f;
  }

  PolyUOp *leaves[] = {base_buf(x), base_buf(w), base_buf(b)};
  float *ld[] = {dx, dw, db};
  ASSERT_INT_EQ(realize_uop(ctx, r, out_buf, dout, leaves, ld, 3), 0);

  for (int o = 0; o < OUT; o++) {
    double acc = db[o];
    for (int i = 0; i < IN; i++)
      acc += (double)dx[i] * (double)dw[(size_t)o * (size_t)IN + (size_t)i];
    ASSERT_FLOAT_EQ(dout[o], (float)acc, 1e-3f);
  }

  free(dx);
  free(dw);
  free(db);
  free(dout);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, repeat_interleave_e2e) {
  /* Reference: [[1,1,2,2,3,3], [4,4,5,5,6,6]] */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = make_buf(ctx, (int64_t[]){2, 3}, 2);
  PolyUOp *out_buf = poly_buffer_f32(ctx, 12);
  PolyUOp *r = poly_repeat_interleave(ctx, x, 2, 1);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, r), 2);
  ASSERT_INT_EQ(poly_uop_dims(ctx, r)[1], 6);

  float dx[] = {1, 2, 3, 4, 5, 6}, dout[12] = {0};
  PolyUOp *leaves[] = {base_buf(x)};
  float *ld[] = {dx};
  ASSERT_INT_EQ(realize_uop(ctx, r, out_buf, dout, leaves, ld, 1), 0);
  float expected[] = {1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6};
  for (int i = 0; i < 12; i++)
    ASSERT_FLOAT_EQ(dout[i], expected[i], 1e-6);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, argmax_e2e) {
  /* Reference: argmax([1,5,3,2,4]) = 1 */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_buffer_f32(ctx, 5);
  PolyUOp *out_buf = poly_buffer_f32(ctx, 1);
  PolyUOp *r = poly_argmax(ctx, x, 0);
  /* Cast int32 result to float for output */
  r = poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, r, poly_arg_none());

  float dx[] = {1, 5, 3, 2, 4}, dout[1] = {0};
  PolyUOp *leaves[] = {x};
  float *ld[] = {dx};
  ASSERT_INT_EQ(realize_uop(ctx, r, out_buf, dout, leaves, ld, 1), 0);
  ASSERT_FLOAT_EQ(dout[0], 1.0f, 1e-4);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, argmax_2d_e2e) {
  /* Reference: argmax([[1,5,3],[4,2,6]], axis=1) = [1, 2] */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = make_buf(ctx, (int64_t[]){2, 3}, 2);
  PolyUOp *out_buf = poly_buffer_f32(ctx, 2);
  PolyUOp *r = poly_argmax(ctx, x, 1);
  r = poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, r, poly_arg_none());

  float dx[] = {1, 5, 3, 4, 2, 6}, dout[2] = {0};
  PolyUOp *leaves[] = {base_buf(x)};
  float *ld[] = {dx};
  ASSERT_INT_EQ(realize_uop(ctx, r, out_buf, dout, leaves, ld, 1), 0);
  ASSERT_FLOAT_EQ(dout[0], 1.0f, 1e-4);
  ASSERT_FLOAT_EQ(dout[1], 2.0f, 1e-4);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, mse_loss_e2e) {
  /* mse([1,2,3], [4,5,6]) = mean([9,9,9]) = 9 */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *pred = poly_buffer_f32(ctx, 3);
  PolyUOp *tgt = poly_buffer_f32(ctx, 3);
  PolyUOp *out_buf = poly_buffer_f32(ctx, 1);
  PolyUOp *r = poly_mse_loss(ctx, pred, tgt);

  float dp[] = {1, 2, 3}, dt[] = {4, 5, 6}, dout[1] = {0};
  PolyUOp *leaves[] = {pred, tgt};
  float *ld[] = {dp, dt};
  ASSERT_INT_EQ(realize_uop(ctx, r, out_buf, dout, leaves, ld, 2), 0);
  ASSERT_FLOAT_EQ(dout[0], 9.0f, 1e-4);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, softmax_v2_e2e) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_buffer_f32(ctx, 3);
  PolyUOp *out_buf = poly_buffer_f32(ctx, 3);
  PolyUOp *sm = poly_softmax(ctx, x, 0);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, sm), 1);

  float dx[] = {1, 2, 3}, dout[3] = {0};
  poly_realize_begin(ctx);
  poly_realize_bind(ctx, x, dx);
  poly_realize_bind(ctx, out_buf, dout);
  PolyUOp *store = poly_store_val(ctx, out_buf, sm);
  ASSERT_INT_EQ(poly_realize_exec(ctx, poly_sink1(ctx, store)), 0);
  ASSERT_FLOAT_EQ(dout[0], 0.0900f, 1e-3);
  ASSERT_FLOAT_EQ(dout[2], 0.6652f, 1e-3);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, dot_v2_e2e) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = make_buf(ctx, (int64_t[]){2, 3}, 2);
  PolyUOp *b = make_buf(ctx, (int64_t[]){3, 2}, 2);
  PolyUOp *r = poly_dot(ctx, a, b);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, r), 2);
  ASSERT_INT_EQ(poly_uop_dims(ctx, r)[0], 2);
  ASSERT_INT_EQ(poly_uop_dims(ctx, r)[1], 2);

  PolyUOp *out_buf = poly_buffer_f32(ctx, 4);
  float da[] = {1, 2, 3, 4, 5, 6}, db[] = {1, 2, 3, 4, 5, 6}, dout[4] = {0};
  PolyUOp *leaves[] = {base_buf(a), base_buf(b)};
  float *ld[] = {da, db};
  ASSERT_INT_EQ(realize_uop(ctx, r, out_buf, dout, leaves, ld, 2), 0);
  ASSERT_FLOAT_EQ(dout[0], 22.0f, 1e-4);
  ASSERT_FLOAT_EQ(dout[3], 64.0f, 1e-4);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, layernorm_v2_shape) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = make_buf(ctx, (int64_t[]){2, 3}, 2);
  PolyUOp *r = poly_layernorm_apply(ctx, x, NULL, NULL, -1, 1e-5);
  ASSERT_NOT_NULL(r);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, r), 2);
  ASSERT_INT_EQ(poly_uop_dims(ctx, r)[0], 2);
  ASSERT_INT_EQ(poly_uop_dims(ctx, r)[1], 3);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, rope_e2e) {
  /* Reference (tinygrad): pos=0 → [1,0,0,1], pos=1 → [0.5403,-0.01,0.8415,0.9999] */
  PolyCtx *ctx = poly_ctx_new();
  int64_t dim = 4, seq = 3, half = 2;

  float cos_data[6], sin_data[6];
  double theta = 10000.0;
  for (int64_t i = 0; i < seq; i++)
    for (int64_t j = 0; j < half; j++) {
      double freq = 1.0 / pow(theta, (double)(2 * j) / (double)dim);
      double angle = (double)i * freq;
      cos_data[i * half + j] = (float)cos(angle);
      sin_data[i * half + j] = (float)sin(angle);
    }

  PolyUOp *x = make_buf(ctx, (int64_t[]){1, 1, 3, 4}, 4);
  PolyUOp *fc_buf = poly_buffer_f32(ctx, 6);
  PolyUOp *fs_buf = poly_buffer_f32(ctx, 6);
  PolyUOp *fc = poly_reshape(ctx, fc_buf, (int64_t[]){1, 1, 3, 2}, 4);
  PolyUOp *fs = poly_reshape(ctx, fs_buf, (int64_t[]){1, 1, 3, 2}, 4);
  PolyUOp *out_buf = poly_buffer_f32(ctx, 12);
  PolyUOp *r = poly_rope(ctx, x, fc, fs);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, r), 4);
  ASSERT_INT_EQ(poly_uop_dims(ctx, r)[3], 4);

  float dx[] = {1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1}, dout[12] = {0};
  PolyUOp *leaves[] = {base_buf(x), fc_buf, fs_buf};
  float *ld[] = {dx, cos_data, sin_data};
  ASSERT_INT_EQ(realize_uop(ctx, r, out_buf, dout, leaves, ld, 3), 0);
  ASSERT_FLOAT_EQ(dout[0], 1.0f, 1e-3);
  ASSERT_FLOAT_EQ(dout[4], 0.5403f, 1e-3);
  ASSERT_FLOAT_EQ(dout[6], 0.8415f, 1e-3);
  poly_ctx_destroy(ctx);
  PASS();
}

/* ═══════════════════════════════════════════════════════════════════════ */
/*  Shape-on-UOp rule tests                                               */
/* ═══════════════════════════════════════════════════════════════════════ */

TEST(shape_uop, buffer_static) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *b = poly_buffer(ctx, POLY_FLOAT32, 100);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, b), 1);
  ASSERT_INT_EQ(poly_uop_dims(ctx, b)[0], 100);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(shape_uop, buffer_dynamic) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *var = poly_define_var(ctx, "batch", 1, 32);
  PolyUOp *buf = poly_buffer_var(ctx, POLY_FLOAT32, var, (int64_t[]){10}, 1);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, buf), 2);
  ASSERT_INT_EQ(poly_uop_dims(ctx, buf)[0], 32);
  ASSERT_INT_EQ(poly_uop_dims(ctx, buf)[1], 10);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(shape_uop, reshape) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *r = poly_reshape(ctx, poly_buffer(ctx, POLY_FLOAT32, 24), (int64_t[]){2, 3, 4}, 3);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, r), 3);
  ASSERT_INT_EQ(poly_uop_dims(ctx, r)[0], 2);
  ASSERT_INT_EQ(poly_uop_dims(ctx, r)[1], 3);
  ASSERT_INT_EQ(poly_uop_dims(ctx, r)[2], 4);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(shape_uop, permute) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *b = make_buf(ctx, (int64_t[]){2, 3, 4}, 3);
  PolyUOp *p = poly_permute(ctx, b, (int64_t[]){2, 0, 1}, 3);
  ASSERT_INT_EQ(poly_uop_dims(ctx, p)[0], 4);
  ASSERT_INT_EQ(poly_uop_dims(ctx, p)[1], 2);
  ASSERT_INT_EQ(poly_uop_dims(ctx, p)[2], 3);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(shape_uop, pad) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *b = make_buf(ctx, (int64_t[]){2, 3}, 2);
  PolyUOp *p = poly_pad(ctx, b, (int64_t[][2]){{1, 1}, {2, 0}}, 2);
  ASSERT_INT_EQ(poly_uop_dims(ctx, p)[0], 4);
  ASSERT_INT_EQ(poly_uop_dims(ctx, p)[1], 5);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(shape_uop, shrink) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *b = make_buf(ctx, (int64_t[]){4, 5}, 2);
  PolyUOp *s = poly_shrink(ctx, b, (int64_t[][2]){{1, 3}, {0, 4}}, 2);
  ASSERT_INT_EQ(poly_uop_dims(ctx, s)[0], 2);
  ASSERT_INT_EQ(poly_uop_dims(ctx, s)[1], 4);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(shape_uop, reduce_axis) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *b = make_buf(ctx, (int64_t[]){2, 3, 4}, 3);
  PolyUOp *r = poly_reduce_axis(ctx, POLY_OP_ADD, b, (int64_t[]){1}, 1);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, r), 3);
  ASSERT_INT_EQ(poly_uop_dims(ctx, r)[0], 2);
  ASSERT_INT_EQ(poly_uop_dims(ctx, r)[1], 1);
  ASSERT_INT_EQ(poly_uop_dims(ctx, r)[2], 4);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(shape_uop, reduce_axis_drops_singleton_axes_like_tinygrad_rop) {
  PolyCtx *ctx = poly_ctx_new();

  /* tinygrad UOp._rop filters size-1 axes before constructing REDUCE_AXIS.
   * Keep that parity at the constructor boundary so later schedule/rangeify
   * stages never see no-op singleton reductions. */
  PolyUOp *b = make_buf(ctx, (int64_t[]){1, 4, 3}, 3);

  PolyUOp *only_singleton = poly_reduce_axis(ctx, POLY_OP_ADD, b, (int64_t[]){0}, 1);
  ASSERT_PTR_EQ(only_singleton, b);

  PolyUOp *mixed = poly_reduce_axis(ctx, POLY_OP_ADD, b, (int64_t[]){2, 0}, 2);
  ASSERT_NOT_NULL(mixed);
  ASSERT_EQ(mixed->op, POLY_OP_REDUCE_AXIS);
  ASSERT_EQ(mixed->arg.kind, POLY_ARG_REDUCE_AXIS);
  ASSERT_INT_EQ(mixed->arg.reduce_axis.n, 1);
  ASSERT_INT_EQ(mixed->arg.reduce_axis.axes[0], 2);
  ASSERT_INT_EQ(poly_uop_dims(ctx, mixed)[0], 1);
  ASSERT_INT_EQ(poly_uop_dims(ctx, mixed)[1], 4);
  ASSERT_INT_EQ(poly_uop_dims(ctx, mixed)[2], 1);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(shape_uop, alu_broadcast_scalar) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = make_buf(ctx, (int64_t[]){3, 4}, 2);
  PolyUOp *c = poly_const_float(ctx, 1.0);
  PolyUOp *r = poly_alu2(ctx, POLY_OP_ADD, a, c);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, r), 2);
  ASSERT_INT_EQ(poly_uop_dims(ctx, r)[0], 3);
  ASSERT_INT_EQ(poly_uop_dims(ctx, r)[1], 4);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(shape_uop, alu_broadcast_ndim) {
  /* (3,5,1) + (5,4) → (3,5,4) -- the embedding WHERE bug case */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = make_buf(ctx, (int64_t[]){3, 5, 1}, 3);
  PolyUOp *b = make_buf(ctx, (int64_t[]){5, 4}, 2);
  PolyUOp *r = poly_alu2(ctx, POLY_OP_ADD, a, b);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, r), 3);
  ASSERT_INT_EQ(poly_uop_dims(ctx, r)[0], 3);
  ASSERT_INT_EQ(poly_uop_dims(ctx, r)[1], 5);
  ASSERT_INT_EQ(poly_uop_dims(ctx, r)[2], 4);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(shape_uop, cmplt_broadcast) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = make_buf(ctx, (int64_t[]){2, 3}, 2);
  PolyUOp *c = poly_const_float(ctx, 0.5);
  PolyUOp *r = poly_alu2(ctx, POLY_OP_CMPLT, c, a);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, r), 2);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(shape_uop, const_scalar) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_INT_EQ(poly_uop_ndim(ctx, poly_const_float(ctx, 42.0)), 0);
  ASSERT_TRUE(poly_uop_dims(ctx, poly_const_float(ctx, 42.0)) == NULL);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(shape_uop, store_inherits_value_shape) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *b = poly_buffer_f32(ctx, 4);
  PolyUOp *st = poly_store_val(ctx, b, poly_const_float(ctx, 1.0));
  /* STORE inherits shape from its value source (src[1]).
   * A scalar const has shape (), so ndim=0. */
  ASSERT_INT_EQ(poly_uop_ndim(ctx, st), 0);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(shape_uop, contiguous_passthrough) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *b = make_buf(ctx, (int64_t[]){2, 3}, 2);
  PolyUOp *c = poly_uop1(ctx, POLY_OP_CONTIGUOUS, b->dtype, b, poly_arg_none());
  ASSERT_INT_EQ(poly_uop_ndim(ctx, c), 2);
  ASSERT_INT_EQ(poly_uop_dims(ctx, c)[0], 2);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(shape_uop, assign_flat) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *buf = poly_buffer(ctx, POLY_FLOAT32, 12);
  PolyUOp *r = make_buf(ctx, (int64_t[]){3, 4}, 2);
  PolyUOp *a = poly_assign(ctx, r, poly_alu2(ctx, POLY_OP_ADD, r, poly_const_float(ctx, 1.0)));
  /* ASSIGN normalizes to flat BUFFER (shape fix deferred) */
  ASSERT_INT_EQ(poly_uop_ndim(ctx, a), 1);
  ASSERT_INT_EQ(poly_uop_dims(ctx, a)[0], 12);
  poly_ctx_destroy(ctx);
  PASS();
}

/* Shape parity oracle */

static int check_shape_parity(PolyCtx *ctx, PolyUOp *root) {
  int n_topo;
  PolyUOp **topo = poly_toposort(ctx, root, &n_topo);
  int mismatches = 0;
  for (int i = 0; i < n_topo; i++) {
    PolyShape computed = poly_uop_shape(ctx, topo[i]);
    int cached_ndim = poly_uop_ndim(ctx, topo[i]);
    if (cached_ndim != computed.ndim) {
      fprintf(
          stderr, "  parity: op=%s cached=%d computed=%d\n", poly_op_name(topo[i]->op), cached_ndim,
          computed.ndim
      );
      mismatches++;
    } else if (cached_ndim > 0 && computed.dims) {
      for (int j = 0; j < cached_ndim; j++)
        if (poly_uop_dims(ctx, topo[i])[j] != computed.dims[j]) {
          fprintf(
              stderr, "  parity: op=%s dim[%d] cached=%ld computed=%ld\n",
              poly_op_name(topo[i]->op), j, (long)poly_uop_dims(ctx, topo[i])[j],
              (long)computed.dims[j]
          );
          mismatches++;
          break;
        }
    }
    if (computed.ndim > 0 && computed.dims) free(computed.dims);
  }
  return mismatches;
}

TEST(shape_uop, parity_softmax) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = make_buf(ctx, (int64_t[]){3, 4}, 2);
  PolyUOp *sm = poly_softmax(ctx, x, -1);
  ASSERT_INT_EQ(check_shape_parity(ctx, sm), 0);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(shape_uop, parity_cross_entropy) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *logits = make_buf(ctx, (int64_t[]){3, 5}, 2);
  PolyUOp *target = poly_reshape(ctx, poly_buffer_f32(ctx, 3), (int64_t[]){3}, 1);
  PolyUOp *ce = poly_cross_entropy(ctx, logits, target, -1);
  ASSERT_INT_EQ(check_shape_parity(ctx, ce), 0);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(shape_uop, parity_gather) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *table = make_buf(ctx, (int64_t[]){5, 4}, 2);
  PolyUOp *idx = poly_reshape(ctx, poly_buffer_f32(ctx, 3), (int64_t[]){3}, 1);
  PolyUOp *g = poly_gather(ctx, table, idx);
  ASSERT_INT_EQ(check_shape_parity(ctx, g), 0);
  poly_ctx_destroy(ctx);
  PASS();
}

/* v2 reduce shape */

TEST(pe, v2_reduce_shape) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = make_buf(ctx, (int64_t[]){3, 4}, 2);
  PolyUOp *s = poly_sum_reduce(ctx, x, 1, 0);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, s), 1);
  ASSERT_INT_EQ(poly_uop_dims(ctx, s)[0], 3);
  PolyUOp *sk = poly_sum_reduce(ctx, x, 1, 1);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, sk), 2);
  ASSERT_INT_EQ(poly_uop_dims(ctx, sk)[0], 3);
  ASSERT_INT_EQ(poly_uop_dims(ctx, sk)[1], 1);
  poly_ctx_destroy(ctx);
  PASS();
}

/* Contiguous */

TEST(tensor, contiguous_passthrough) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a_buf = poly_buffer_f32(ctx, 4);
  PolyUOp *a = poly_reshape(ctx, a_buf, (int64_t[]){4}, 1);
  PolyUOp *c = poly_contiguous(ctx, a);

  PolyUOp *out_buf = poly_buffer_f32(ctx, 4);
  PolyUOp *store = poly_store_val(ctx, out_buf, c);
  PolyUOp *sink = poly_sink1(ctx, store);

  float in[] = {1, 2, 3, 4};
  float out[4] = {0};
  PolyBufferBinding bindings[] = {
      POLY_BIND_HOST(out_buf, out),
      POLY_BIND_HOST(a_buf, in),
  };
  ASSERT_INT_EQ(poly_realize_with_bindings(ctx, sink, bindings, 2), 0);
  for (int i = 0; i < 4; i++)
    ASSERT_FLOAT_EQ(out[i], in[i], 1e-6);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(tensor, contiguous_expand_materializes) {
  /* expand (4,1)->(4,4) then contiguous forces a real copy */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a_buf = poly_buffer_f32(ctx, 4);
  PolyUOp *a = poly_reshape(ctx, a_buf, (int64_t[]){4, 1}, 2);
  PolyUOp *expanded = poly_expand(ctx, a, (int64_t[]){4, 4}, 2);
  PolyUOp *c = poly_contiguous(ctx, expanded);
  PolyUOp *result = poly_add(ctx, c, poly_const_float(ctx, 1.0));

  PolyUOp *out_buf = poly_buffer_f32(ctx, 16);
  PolyUOp *store = poly_store_val(ctx, out_buf, result);
  PolyUOp *sink = poly_sink1(ctx, store);

  float in[] = {10, 20, 30, 40};
  float out[16] = {0};
  PolyBufferBinding bindings[] = {
      POLY_BIND_HOST(out_buf, out),
      POLY_BIND_HOST(a_buf, in),
  };
  ASSERT_INT_EQ(poly_realize_with_bindings(ctx, sink, bindings, 2), 0);
  for (int r = 0; r < 4; r++)
    for (int c2 = 0; c2 < 4; c2++)
      ASSERT_FLOAT_EQ(out[r * 4 + c2], in[r] + 1.0f, 1e-6);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(tensor, contiguous_chain) {
  /* a*2 -> contiguous -> +1 -> contiguous -> output */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a_buf = poly_buffer_f32(ctx, 3);
  PolyUOp *a = poly_reshape(ctx, a_buf, (int64_t[]){3}, 1);

  PolyUOp *doubled =
      poly_contiguous(ctx, poly_alu2(ctx, POLY_OP_MUL, a, poly_const_float(ctx, 2.0)));
  PolyUOp *result = poly_contiguous(ctx, poly_add(ctx, doubled, poly_const_float(ctx, 1.0)));

  PolyUOp *out_buf = poly_buffer_f32(ctx, 3);
  PolyUOp *store = poly_store_val(ctx, out_buf, result);
  PolyUOp *sink = poly_sink1(ctx, store);

  float in[] = {5, 10, 15};
  float out[3] = {0};
  PolyBufferBinding bindings[] = {
      POLY_BIND_HOST(out_buf, out),
      POLY_BIND_HOST(a_buf, in),
  };
  ASSERT_INT_EQ(poly_realize_with_bindings(ctx, sink, bindings, 2), 0);
  ASSERT_FLOAT_EQ(out[0], 11.0f, 1e-6);
  ASSERT_FLOAT_EQ(out[1], 21.0f, 1e-6);
  ASSERT_FLOAT_EQ(out[2], 31.0f, 1e-6);
  poly_ctx_destroy(ctx);
  PASS();
}

/* ═══════════════════════════════════════════════════════════════════════ */
/*  Movement-op helper tests (poly_repeat / poly_shrink_to / poly_pool)   */
/*                                                                        */
/*  Reference values verified against tinygrad_latest, conda env tiny:    */
/*    PYTHONPATH=references/tinygrad_latest python -c "from tinygrad ..." */
/*                                                                        */
/*  These cover the helpers that poly_cumalu (and conv) need. Inputs are  */
/*  bound via POLY_BIND_HOST -- no const-registry path involved.          */
/* ═══════════════════════════════════════════════════════════════════════ */

TEST(pe, repeat_1d_simple) {
  PolyCtx *ctx = poly_ctx_new();

  /* tinygrad: Tensor([1,2,3]).repeat([4]) -> 12 elements [1,2,3,1,2,3,...] */
  PolyUOp *in = poly_buffer_f32(ctx, 3);
  int64_t reps[1] = {4};
  PolyUOp *out_val = poly_repeat(ctx, in, reps, 1);
  ASSERT_NOT_NULL(out_val);

  PolyUOp *out_buf = poly_buffer_f32(ctx, 12);
  float in_data[3] = {1.0f, 2.0f, 3.0f};
  float out_data[12] = {0};
  PolyUOp *leaves[] = {in};
  float *ld[] = {in_data};
  ASSERT_INT_EQ(realize_uop(ctx, out_val, out_buf, out_data, leaves, ld, 1), 0);

  float expected[12] = {1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3};
  for (int i = 0; i < 12; i++)
    ASSERT_FLOAT_EQ(out_data[i], expected[i], 1e-5);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, repeat_1d_to_2d) {
  PolyCtx *ctx = poly_ctx_new();

  /* tinygrad: Tensor([1,2,3]).repeat([2,3]) -> shape (2,9), each row [1,2,3]*3 */
  PolyUOp *in = poly_buffer_f32(ctx, 3);
  int64_t reps[2] = {2, 3};
  PolyUOp *out_val = poly_repeat(ctx, in, reps, 2);
  ASSERT_NOT_NULL(out_val);

  PolyUOp *out_buf = poly_buffer_f32(ctx, 18);
  float in_data[3] = {1.0f, 2.0f, 3.0f};
  float out_data[18] = {0};
  PolyUOp *leaves[] = {in};
  float *ld[] = {in_data};
  ASSERT_INT_EQ(realize_uop(ctx, out_val, out_buf, out_data, leaves, ld, 1), 0);

  float expected[18] = {1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3};
  for (int i = 0; i < 18; i++)
    ASSERT_FLOAT_EQ(out_data[i], expected[i], 1e-5);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, repeat_2d_2x2) {
  PolyCtx *ctx = poly_ctx_new();

  /* tinygrad: Tensor([[1,2],[3,4]]).repeat([2,3]) -> shape (4,6) */
  int64_t shape[2] = {2, 2};
  PolyUOp *in = make_buf(ctx, shape, 2);
  int64_t reps[2] = {2, 3};
  PolyUOp *out_val = poly_repeat(ctx, in, reps, 2);
  ASSERT_NOT_NULL(out_val);

  PolyUOp *out_buf = poly_buffer_f32(ctx, 24);
  float in_data[4] = {1, 2, 3, 4};
  float out_data[24] = {0};
  PolyUOp *leaves[] = {base_buf(in)};
  float *ld[] = {in_data};
  ASSERT_INT_EQ(realize_uop(ctx, out_val, out_buf, out_data, leaves, ld, 1), 0);

  /* tinygrad output:
   * [[1,2,1,2,1,2],[3,4,3,4,3,4],[1,2,1,2,1,2],[3,4,3,4,3,4]] */
  float expected[24] = {
      1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4, 1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4,
  };
  for (int i = 0; i < 24; i++)
    ASSERT_FLOAT_EQ(out_data[i], expected[i], 1e-5);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, shrink_to_2d) {
  PolyCtx *ctx = poly_ctx_new();

  /* tinygrad: arange(12).reshape(3,4).shrink_to((2,3)) -> [[0,1,2],[4,5,6]] */
  int64_t shape[2] = {3, 4};
  PolyUOp *in = make_buf(ctx, shape, 2);
  int64_t ends[2] = {2, 3};
  PolyUOp *out_val = poly_shrink_to(ctx, in, ends, 2);
  ASSERT_NOT_NULL(out_val);

  PolyUOp *out_buf = poly_buffer_f32(ctx, 6);
  float in_data[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  float out_data[6] = {0};
  PolyUOp *leaves[] = {base_buf(in)};
  float *ld[] = {in_data};
  ASSERT_INT_EQ(realize_uop(ctx, out_val, out_buf, out_data, leaves, ld, 1), 0);

  float expected[6] = {0, 1, 2, 4, 5, 6};
  for (int i = 0; i < 6; i++)
    ASSERT_FLOAT_EQ(out_data[i], expected[i], 1e-5);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, pool_1d_k3_stride1) {
  PolyCtx *ctx = poly_ctx_new();

  /* tinygrad: arange(5)._pool((3,)) -> shape (3,3) [[0,1,2],[1,2,3],[2,3,4]] */
  PolyUOp *in = poly_buffer_f32(ctx, 5);
  int64_t k[1] = {3};
  PolyUOp *out_val = poly_pool(ctx, in, k, 1, NULL, NULL);
  ASSERT_NOT_NULL(out_val);

  PolyUOp *out_buf = poly_buffer_f32(ctx, 9);
  float in_data[5] = {0, 1, 2, 3, 4};
  float out_data[9] = {0};
  PolyUOp *leaves[] = {in};
  float *ld[] = {in_data};
  ASSERT_INT_EQ(realize_uop(ctx, out_val, out_buf, out_data, leaves, ld, 1), 0);

  /* row-major (3,3): [[0,1,2],[1,2,3],[2,3,4]] */
  float expected[9] = {0, 1, 2, 1, 2, 3, 2, 3, 4};
  for (int i = 0; i < 9; i++)
    ASSERT_FLOAT_EQ(out_data[i], expected[i], 1e-5);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, pool_1d_k3_stride2) {
  PolyCtx *ctx = poly_ctx_new();

  /* tinygrad: arange(5)._pool((3,), stride=2) -> (2,3) [[0,1,2],[2,3,4]] */
  PolyUOp *in = poly_buffer_f32(ctx, 5);
  int64_t k[1] = {3}, s[1] = {2};
  PolyUOp *out_val = poly_pool(ctx, in, k, 1, s, NULL);
  ASSERT_NOT_NULL(out_val);

  PolyUOp *out_buf = poly_buffer_f32(ctx, 6);
  float in_data[5] = {0, 1, 2, 3, 4};
  float out_data[6] = {0};
  PolyUOp *leaves[] = {in};
  float *ld[] = {in_data};
  ASSERT_INT_EQ(realize_uop(ctx, out_val, out_buf, out_data, leaves, ld, 1), 0);

  float expected[6] = {0, 1, 2, 2, 3, 4};
  for (int i = 0; i < 6; i++)
    ASSERT_FLOAT_EQ(out_data[i], expected[i], 1e-5);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, pool_1d_k2_dilation2) {
  PolyCtx *ctx = poly_ctx_new();

  /* tinygrad: arange(8)._pool((2,), dilation=2)
   * -> (6,2) [[0,2],[1,3],[2,4],[3,5],[4,6],[5,7]] */
  PolyUOp *in = poly_buffer_f32(ctx, 8);
  int64_t k[1] = {2}, d[1] = {2};
  PolyUOp *out_val = poly_pool(ctx, in, k, 1, NULL, d);
  ASSERT_NOT_NULL(out_val);

  PolyUOp *out_buf = poly_buffer_f32(ctx, 12);
  float in_data[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  float out_data[12] = {0};
  PolyUOp *leaves[] = {in};
  float *ld[] = {in_data};
  ASSERT_INT_EQ(realize_uop(ctx, out_val, out_buf, out_data, leaves, ld, 1), 0);

  float expected[12] = {0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7};
  for (int i = 0; i < 12; i++)
    ASSERT_FLOAT_EQ(out_data[i], expected[i], 1e-5);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, pool_2d_k22_4x4) {
  PolyCtx *ctx = poly_ctx_new();

  /* tinygrad: arange(16).reshape(4,4)._pool((2,2)) -> shape (3,3,2,2) */
  int64_t shape[2] = {4, 4};
  PolyUOp *in = make_buf(ctx, shape, 2);
  int64_t k[2] = {2, 2};
  PolyUOp *out_val = poly_pool(ctx, in, k, 2, NULL, NULL);
  ASSERT_NOT_NULL(out_val);

  PolyUOp *out_buf = poly_buffer_f32(ctx, 36);
  float in_data[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  float out_data[36] = {0};
  PolyUOp *leaves[] = {base_buf(in)};
  float *ld[] = {in_data};
  ASSERT_INT_EQ(realize_uop(ctx, out_val, out_buf, out_data, leaves, ld, 1), 0);

  /* tinygrad output (3,3,2,2) flattened row-major:
   *  [[ [[0,1],[4,5]],   [[1,2],[5,6]],   [[2,3],[6,7]]   ],
   *   [ [[4,5],[8,9]],   [[5,6],[9,10]],  [[6,7],[10,11]] ],
   *   [ [[8,9],[12,13]], [[9,10],[13,14]],[[10,11],[14,15]] ]] */
  float expected[36] = {
      0, 1,  4, 5, 1,  2,  5, 6, 2,  3,  6, 7,  4,  5,  8,  9,  5,  6,
      9, 10, 6, 7, 10, 11, 8, 9, 12, 13, 9, 10, 13, 14, 10, 11, 14, 15,
  };
  for (int i = 0; i < 36; i++)
    ASSERT_FLOAT_EQ(out_data[i], expected[i], 1e-5);

  poly_ctx_destroy(ctx);
  PASS();
}

/* ═══════════════════════════════════════════════════════════════════════ */
/*  poly_cumalu (ADD-only) tests -- tinygrad-verified                     */
/* ═══════════════════════════════════════════════════════════════════════ */

TEST(pe, cumalu_add_1d) {
  PolyCtx *ctx = poly_ctx_new();

  /* tinygrad: Tensor([1..5])._cumalu(0, ADD) -> [1, 3, 6, 10, 15] */
  PolyUOp *in = poly_buffer_f32(ctx, 5);
  PolyUOp *out_val = poly_cumalu(ctx, in, 0, POLY_OP_ADD, false);
  ASSERT_NOT_NULL(out_val);

  PolyUOp *out_buf = poly_buffer_f32(ctx, 5);
  float in_data[5] = {1, 2, 3, 4, 5};
  float out_data[5] = {0};
  PolyUOp *leaves[] = {in};
  float *ld[] = {in_data};
  ASSERT_INT_EQ(realize_uop(ctx, out_val, out_buf, out_data, leaves, ld, 1), 0);

  float expected[5] = {1, 3, 6, 10, 15};
  for (int i = 0; i < 5; i++)
    ASSERT_FLOAT_EQ(out_data[i], expected[i], 1e-5);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, cumalu_add_1d_const) {
  PolyCtx *ctx = poly_ctx_new();

  /* tinygrad: Tensor([3,3,3,3,3])._cumalu(0, ADD) -> [3, 6, 9, 12, 15]
   * This is the exact shape arange(0, 15, 3) builds internally. */
  PolyUOp *in = poly_buffer_f32(ctx, 5);
  PolyUOp *out_val = poly_cumalu(ctx, in, 0, POLY_OP_ADD, false);
  ASSERT_NOT_NULL(out_val);

  PolyUOp *out_buf = poly_buffer_f32(ctx, 5);
  float in_data[5] = {3, 3, 3, 3, 3};
  float out_data[5] = {0};
  PolyUOp *leaves[] = {in};
  float *ld[] = {in_data};
  ASSERT_INT_EQ(realize_uop(ctx, out_val, out_buf, out_data, leaves, ld, 1), 0);

  float expected[5] = {3, 6, 9, 12, 15};
  for (int i = 0; i < 5; i++)
    ASSERT_FLOAT_EQ(out_data[i], expected[i], 1e-5);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, cumalu_add_1d_include_initial) {
  PolyCtx *ctx = poly_ctx_new();

  /* tinygrad: Tensor([1..5])._cumalu(0, ADD, _include_initial=True)
   *   -> [0, 1, 3, 6, 10] */
  PolyUOp *in = poly_buffer_f32(ctx, 5);
  PolyUOp *out_val = poly_cumalu(ctx, in, 0, POLY_OP_ADD, true);
  ASSERT_NOT_NULL(out_val);

  PolyUOp *out_buf = poly_buffer_f32(ctx, 5);
  float in_data[5] = {1, 2, 3, 4, 5};
  float out_data[5] = {0};
  PolyUOp *leaves[] = {in};
  float *ld[] = {in_data};
  ASSERT_INT_EQ(realize_uop(ctx, out_val, out_buf, out_data, leaves, ld, 1), 0);

  float expected[5] = {0, 1, 3, 6, 10};
  for (int i = 0; i < 5; i++)
    ASSERT_FLOAT_EQ(out_data[i], expected[i], 1e-5);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, cumalu_add_2d_axis1) {
  PolyCtx *ctx = poly_ctx_new();

  /* tinygrad: arange(12).reshape(3,4)._cumalu(1, ADD)
   *   -> [[0,1,3,6],[4,9,15,22],[8,17,27,38]] */
  int64_t shape[2] = {3, 4};
  PolyUOp *in = make_buf(ctx, shape, 2);
  PolyUOp *out_val = poly_cumalu(ctx, in, 1, POLY_OP_ADD, false);
  ASSERT_NOT_NULL(out_val);

  PolyUOp *out_buf = poly_buffer_f32(ctx, 12);
  float in_data[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  float out_data[12] = {0};
  PolyUOp *leaves[] = {base_buf(in)};
  float *ld[] = {in_data};
  ASSERT_INT_EQ(realize_uop(ctx, out_val, out_buf, out_data, leaves, ld, 1), 0);

  float expected[12] = {0, 1, 3, 6, 4, 9, 15, 22, 8, 17, 27, 38};
  for (int i = 0; i < 12; i++)
    ASSERT_FLOAT_EQ(out_data[i], expected[i], 1e-5);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, cumalu_add_2d_axis0) {
  PolyCtx *ctx = poly_ctx_new();

  /* tinygrad: arange(12).reshape(3,4)._cumalu(0, ADD)
   *   -> [[0,1,2,3],[4,6,8,10],[12,15,18,21]] */
  int64_t shape[2] = {3, 4};
  PolyUOp *in = make_buf(ctx, shape, 2);
  PolyUOp *out_val = poly_cumalu(ctx, in, 0, POLY_OP_ADD, false);
  ASSERT_NOT_NULL(out_val);

  PolyUOp *out_buf = poly_buffer_f32(ctx, 12);
  float in_data[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  float out_data[12] = {0};
  PolyUOp *leaves[] = {base_buf(in)};
  float *ld[] = {in_data};
  ASSERT_INT_EQ(realize_uop(ctx, out_val, out_buf, out_data, leaves, ld, 1), 0);

  float expected[12] = {0, 1, 2, 3, 4, 6, 8, 10, 12, 15, 18, 21};
  for (int i = 0; i < 12; i++)
    ASSERT_FLOAT_EQ(out_data[i], expected[i], 1e-5);

  poly_ctx_destroy(ctx);
  PASS();
}

/* ═══════════════════════════════════════════════════════════════════════ */
/*  poly_cat tests -- tinygrad-verified                                   */
/* ═══════════════════════════════════════════════════════════════════════ */

TEST(pe, cat_1d) {
  PolyCtx *ctx = poly_ctx_new();

  /* tinygrad: [1,2].cat([3,4,5], dim=0) -> [1,2,3,4,5] */
  PolyUOp *a = poly_buffer_f32(ctx, 2);
  PolyUOp *b = poly_buffer_f32(ctx, 3);
  PolyUOp *parts[2] = {a, b};
  PolyUOp *out_val = poly_cat(ctx, parts, 2, 0);
  ASSERT_NOT_NULL(out_val);

  PolyUOp *out_buf = poly_buffer_f32(ctx, 5);
  float a_d[2] = {1, 2}, b_d[3] = {3, 4, 5}, out[5] = {0};
  PolyUOp *leaves[] = {a, b};
  float *ld[] = {a_d, b_d};
  ASSERT_INT_EQ(realize_uop(ctx, out_val, out_buf, out, leaves, ld, 2), 0);

  float expected[5] = {1, 2, 3, 4, 5};
  for (int i = 0; i < 5; i++)
    ASSERT_FLOAT_EQ(out[i], expected[i], 1e-5);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, cat_2d_axis0) {
  PolyCtx *ctx = poly_ctx_new();

  /* tinygrad: [[1,2],[3,4]].cat([[5,6]], dim=0) -> [[1,2],[3,4],[5,6]] */
  int64_t s_a[2] = {2, 2}, s_b[2] = {1, 2};
  PolyUOp *a = make_buf(ctx, s_a, 2);
  PolyUOp *b = make_buf(ctx, s_b, 2);
  PolyUOp *parts[2] = {a, b};
  PolyUOp *out_val = poly_cat(ctx, parts, 2, 0);
  ASSERT_NOT_NULL(out_val);

  PolyUOp *out_buf = poly_buffer_f32(ctx, 6);
  float a_d[4] = {1, 2, 3, 4}, b_d[2] = {5, 6}, out[6] = {0};
  PolyUOp *leaves[] = {base_buf(a), base_buf(b)};
  float *ld[] = {a_d, b_d};
  ASSERT_INT_EQ(realize_uop(ctx, out_val, out_buf, out, leaves, ld, 2), 0);

  float expected[6] = {1, 2, 3, 4, 5, 6};
  for (int i = 0; i < 6; i++)
    ASSERT_FLOAT_EQ(out[i], expected[i], 1e-5);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, cat_2d_axis1) {
  PolyCtx *ctx = poly_ctx_new();

  /* tinygrad: [[1,2],[3,4]].cat([[7],[8]], dim=1) -> [[1,2,7],[3,4,8]] */
  int64_t s_a[2] = {2, 2}, s_b[2] = {2, 1};
  PolyUOp *a = make_buf(ctx, s_a, 2);
  PolyUOp *b = make_buf(ctx, s_b, 2);
  PolyUOp *parts[2] = {a, b};
  PolyUOp *out_val = poly_cat(ctx, parts, 2, 1);
  ASSERT_NOT_NULL(out_val);

  PolyUOp *out_buf = poly_buffer_f32(ctx, 6);
  float a_d[4] = {1, 2, 3, 4}, b_d[2] = {7, 8}, out[6] = {0};
  PolyUOp *leaves[] = {base_buf(a), base_buf(b)};
  float *ld[] = {a_d, b_d};
  ASSERT_INT_EQ(realize_uop(ctx, out_val, out_buf, out, leaves, ld, 2), 0);

  float expected[6] = {1, 2, 7, 3, 4, 8};
  for (int i = 0; i < 6; i++)
    ASSERT_FLOAT_EQ(out[i], expected[i], 1e-5);

  poly_ctx_destroy(ctx);
  PASS();
}

/* ═══════════════════════════════════════════════════════════════════════ */
/*  poly_pad_value tests (constant pad mode) -- tinygrad-verified         */
/* ═══════════════════════════════════════════════════════════════════════ */

TEST(pe, pad_value_1d_basic) {
  PolyCtx *ctx = poly_ctx_new();

  /* tinygrad: [1,2,3].pad(((2,1),), value=9) -> [9,9,1,2,3,9] */
  PolyUOp *in = poly_buffer_f32(ctx, 3);
  int64_t pads[1][2] = {{2, 1}};
  PolyUOp *out_val = poly_pad_value(ctx, in, pads, 1, 9.0);
  ASSERT_NOT_NULL(out_val);

  PolyUOp *out_buf = poly_buffer_f32(ctx, 6);
  float in_d[3] = {1, 2, 3}, out[6] = {0};
  PolyUOp *leaves[] = {in};
  float *ld[] = {in_d};
  ASSERT_INT_EQ(realize_uop(ctx, out_val, out_buf, out, leaves, ld, 1), 0);

  float expected[6] = {9, 9, 1, 2, 3, 9};
  for (int i = 0; i < 6; i++)
    ASSERT_FLOAT_EQ(out[i], expected[i], 1e-5);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, pad_value_1d_negative_left) {
  PolyCtx *ctx = poly_ctx_new();

  /* tinygrad: [1,2,3].pad(((-1,1),), value=7) -> [2, 3, 7] */
  PolyUOp *in = poly_buffer_f32(ctx, 3);
  int64_t pads[1][2] = {{-1, 1}};
  PolyUOp *out_val = poly_pad_value(ctx, in, pads, 1, 7.0);
  ASSERT_NOT_NULL(out_val);

  PolyUOp *out_buf = poly_buffer_f32(ctx, 3);
  float in_d[3] = {1, 2, 3}, out[3] = {0};
  PolyUOp *leaves[] = {in};
  float *ld[] = {in_d};
  ASSERT_INT_EQ(realize_uop(ctx, out_val, out_buf, out, leaves, ld, 1), 0);

  float expected[3] = {2, 3, 7};
  for (int i = 0; i < 3; i++)
    ASSERT_FLOAT_EQ(out[i], expected[i], 1e-5);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, pad_value_1d_negative_right_zero_value) {
  PolyCtx *ctx = poly_ctx_new();

  /* tinygrad: [1,2,3].pad(((1,-1),), value=0) -> [0,1,2] */
  PolyUOp *in = poly_buffer_f32(ctx, 3);
  int64_t pads[1][2] = {{1, -1}};
  PolyUOp *out_val = poly_pad_value(ctx, in, pads, 1, 0.0);
  ASSERT_NOT_NULL(out_val);

  PolyUOp *out_buf = poly_buffer_f32(ctx, 3);
  float in_d[3] = {1, 2, 3}, out[3] = {0};
  PolyUOp *leaves[] = {in};
  float *ld[] = {in_d};
  ASSERT_INT_EQ(realize_uop(ctx, out_val, out_buf, out, leaves, ld, 1), 0);

  float expected[3] = {0, 1, 2};
  for (int i = 0; i < 3; i++)
    ASSERT_FLOAT_EQ(out[i], expected[i], 1e-5);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, pad_value_2d) {
  PolyCtx *ctx = poly_ctx_new();

  /* tinygrad: [[1,2],[3,4]].pad(((1,1),(0,2)), value=-1) ->
   *   [[-1,-1,-1,-1],[1,2,-1,-1],[3,4,-1,-1],[-1,-1,-1,-1]] */
  int64_t shape[2] = {2, 2};
  PolyUOp *in = make_buf(ctx, shape, 2);
  int64_t pads[2][2] = {{1, 1}, {0, 2}};
  PolyUOp *out_val = poly_pad_value(ctx, in, pads, 2, -1.0);
  ASSERT_NOT_NULL(out_val);

  PolyUOp *out_buf = poly_buffer_f32(ctx, 16);
  float in_d[4] = {1, 2, 3, 4}, out[16] = {0};
  PolyUOp *leaves[] = {base_buf(in)};
  float *ld[] = {in_d};
  ASSERT_INT_EQ(realize_uop(ctx, out_val, out_buf, out, leaves, ld, 1), 0);

  float expected[16] = {
      -1, -1, -1, -1, 1, 2, -1, -1, 3, 4, -1, -1, -1, -1, -1, -1,
  };
  for (int i = 0; i < 16; i++)
    ASSERT_FLOAT_EQ(out[i], expected[i], 1e-5);

  poly_ctx_destroy(ctx);
  PASS();
}

/* ═══════════════════════════════════════════════════════════════════════ */
/*  poly_pad_circular tests -- tinygrad-verified                          */
/* ═══════════════════════════════════════════════════════════════════════ */

TEST(pe, pad_circular_1d) {
  PolyCtx *ctx = poly_ctx_new();

  /* tinygrad: [1,2,3].pad(((1,2),), mode='circular') -> [3, 1, 2, 3, 1, 2] */
  PolyUOp *in = poly_buffer_f32(ctx, 3);
  int64_t pads[1][2] = {{1, 2}};
  PolyUOp *out_val = poly_pad_circular(ctx, in, pads, 1);
  ASSERT_NOT_NULL(out_val);

  PolyUOp *out_buf = poly_buffer_f32(ctx, 6);
  float in_d[3] = {1, 2, 3}, out[6] = {0};
  PolyUOp *leaves[] = {in};
  float *ld[] = {in_d};
  ASSERT_INT_EQ(realize_uop(ctx, out_val, out_buf, out, leaves, ld, 1), 0);

  float expected[6] = {3, 1, 2, 3, 1, 2};
  for (int i = 0; i < 6; i++)
    ASSERT_FLOAT_EQ(out[i], expected[i], 1e-5);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, pad_circular_2d) {
  PolyCtx *ctx = poly_ctx_new();

  /* tinygrad: [[1,2],[3,4]].pad(((1,0),(0,1)), mode='circular')
   *   -> [[3,4,3],[1,2,1],[3,4,3]] */
  int64_t shape[2] = {2, 2};
  PolyUOp *in = make_buf(ctx, shape, 2);
  int64_t pads[2][2] = {{1, 0}, {0, 1}};
  PolyUOp *out_val = poly_pad_circular(ctx, in, pads, 2);
  ASSERT_NOT_NULL(out_val);

  PolyUOp *out_buf = poly_buffer_f32(ctx, 9);
  float in_d[4] = {1, 2, 3, 4}, out[9] = {0};
  PolyUOp *leaves[] = {base_buf(in)};
  float *ld[] = {in_d};
  ASSERT_INT_EQ(realize_uop(ctx, out_val, out_buf, out, leaves, ld, 1), 0);

  float expected[9] = {3, 4, 3, 1, 2, 1, 3, 4, 3};
  for (int i = 0; i < 9; i++)
    ASSERT_FLOAT_EQ(out[i], expected[i], 1e-5);

  poly_ctx_destroy(ctx);
  PASS();
}

/* ═══════════════════════════════════════════════════════════════════════ */
/*  poly_pad_reflect tests -- tinygrad-verified                           */
/* ═══════════════════════════════════════════════════════════════════════ */

TEST(pe, pad_reflect_1d_left_heavy) {
  PolyCtx *ctx = poly_ctx_new();

  /* tinygrad: [1,2,3,4].pad(((2,1),), mode='reflect') -> [3,2, 1,2,3,4, 3] */
  PolyUOp *in = poly_buffer_f32(ctx, 4);
  int64_t pads[1][2] = {{2, 1}};
  PolyUOp *out_val = poly_pad_reflect(ctx, in, pads, 1);
  ASSERT_NOT_NULL(out_val);

  PolyUOp *out_buf = poly_buffer_f32(ctx, 7);
  float in_d[4] = {1, 2, 3, 4}, out[7] = {0};
  PolyUOp *leaves[] = {in};
  float *ld[] = {in_d};
  ASSERT_INT_EQ(realize_uop(ctx, out_val, out_buf, out, leaves, ld, 1), 0);

  float expected[7] = {3, 2, 1, 2, 3, 4, 3};
  for (int i = 0; i < 7; i++)
    ASSERT_FLOAT_EQ(out[i], expected[i], 1e-5);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, pad_reflect_1d_right_heavy) {
  PolyCtx *ctx = poly_ctx_new();

  /* tinygrad: [1,2,3,4].pad(((1,2),), mode='reflect') -> [2, 1,2,3,4, 3,2] */
  PolyUOp *in = poly_buffer_f32(ctx, 4);
  int64_t pads[1][2] = {{1, 2}};
  PolyUOp *out_val = poly_pad_reflect(ctx, in, pads, 1);
  ASSERT_NOT_NULL(out_val);

  PolyUOp *out_buf = poly_buffer_f32(ctx, 7);
  float in_d[4] = {1, 2, 3, 4}, out[7] = {0};
  PolyUOp *leaves[] = {in};
  float *ld[] = {in_d};
  ASSERT_INT_EQ(realize_uop(ctx, out_val, out_buf, out, leaves, ld, 1), 0);

  float expected[7] = {2, 1, 2, 3, 4, 3, 2};
  for (int i = 0; i < 7; i++)
    ASSERT_FLOAT_EQ(out[i], expected[i], 1e-5);

  poly_ctx_destroy(ctx);
  PASS();
}

/* ═══════════════════════════════════════════════════════════════════════ */
/*  poly_pad_replicate tests -- tinygrad-verified                         */
/* ═══════════════════════════════════════════════════════════════════════ */

TEST(pe, pad_replicate_1d_left_heavy) {
  PolyCtx *ctx = poly_ctx_new();

  /* tinygrad: [1,2,3,4].pad(((2,1),), mode='replicate') -> [1,1, 1,2,3,4, 4] */
  PolyUOp *in = poly_buffer_f32(ctx, 4);
  int64_t pads[1][2] = {{2, 1}};
  PolyUOp *out_val = poly_pad_replicate(ctx, in, pads, 1);
  ASSERT_NOT_NULL(out_val);

  PolyUOp *out_buf = poly_buffer_f32(ctx, 7);
  float in_d[4] = {1, 2, 3, 4}, out[7] = {0};
  PolyUOp *leaves[] = {in};
  float *ld[] = {in_d};
  ASSERT_INT_EQ(realize_uop(ctx, out_val, out_buf, out, leaves, ld, 1), 0);

  float expected[7] = {1, 1, 1, 2, 3, 4, 4};
  for (int i = 0; i < 7; i++)
    ASSERT_FLOAT_EQ(out[i], expected[i], 1e-5);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, pad_replicate_1d_right_heavy) {
  PolyCtx *ctx = poly_ctx_new();

  /* tinygrad: [1,2,3,4].pad(((1,2),), mode='replicate') -> [1, 1,2,3,4, 4,4] */
  PolyUOp *in = poly_buffer_f32(ctx, 4);
  int64_t pads[1][2] = {{1, 2}};
  PolyUOp *out_val = poly_pad_replicate(ctx, in, pads, 1);
  ASSERT_NOT_NULL(out_val);

  PolyUOp *out_buf = poly_buffer_f32(ctx, 7);
  float in_d[4] = {1, 2, 3, 4}, out[7] = {0};
  PolyUOp *leaves[] = {in};
  float *ld[] = {in_d};
  ASSERT_INT_EQ(realize_uop(ctx, out_val, out_buf, out, leaves, ld, 1), 0);

  float expected[7] = {1, 1, 2, 3, 4, 4, 4};
  for (int i = 0; i < 7; i++)
    ASSERT_FLOAT_EQ(out[i], expected[i], 1e-5);

  poly_ctx_destroy(ctx);
  PASS();
}

/* ═══════════════════════════════════════════════════════════════════════ */
/*  poly_cumalu MAX/MUL tests -- tinygrad-verified                        */
/* ═══════════════════════════════════════════════════════════════════════ */

TEST(pe, cumalu_max_1d) {
  PolyCtx *ctx = poly_ctx_new();

  /* tinygrad: cummax([1,3,2,5,4]) -> [1, 3, 3, 5, 5] */
  PolyUOp *in = poly_buffer_f32(ctx, 5);
  PolyUOp *out_val = poly_cumalu(ctx, in, 0, POLY_OP_MAX, false);
  ASSERT_NOT_NULL(out_val);

  PolyUOp *out_buf = poly_buffer_f32(ctx, 5);
  float in_d[5] = {1, 3, 2, 5, 4}, out[5] = {0};
  PolyUOp *leaves[] = {in};
  float *ld[] = {in_d};
  ASSERT_INT_EQ(realize_uop(ctx, out_val, out_buf, out, leaves, ld, 1), 0);

  float expected[5] = {1, 3, 3, 5, 5};
  for (int i = 0; i < 5; i++)
    ASSERT_FLOAT_EQ(out[i], expected[i], 1e-5);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, cumalu_max_1d_decreasing) {
  PolyCtx *ctx = poly_ctx_new();

  /* tinygrad: cummax([5,3,4,1,2]) -> [5, 5, 5, 5, 5] */
  PolyUOp *in = poly_buffer_f32(ctx, 5);
  PolyUOp *out_val = poly_cumalu(ctx, in, 0, POLY_OP_MAX, false);
  ASSERT_NOT_NULL(out_val);

  PolyUOp *out_buf = poly_buffer_f32(ctx, 5);
  float in_d[5] = {5, 3, 4, 1, 2}, out[5] = {0};
  PolyUOp *leaves[] = {in};
  float *ld[] = {in_d};
  ASSERT_INT_EQ(realize_uop(ctx, out_val, out_buf, out, leaves, ld, 1), 0);

  float expected[5] = {5, 5, 5, 5, 5};
  for (int i = 0; i < 5; i++)
    ASSERT_FLOAT_EQ(out[i], expected[i], 1e-5);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, cumalu_max_negative) {
  PolyCtx *ctx = poly_ctx_new();

  /* Critical: input is all-negative. If pad-with-value used 0 instead of -inf,
   * the cummax would incorrectly be 0 for the first element.
   * tinygrad: cummax([-3,-1,-2]) -> [-3, -1, -1] */
  PolyUOp *in = poly_buffer_f32(ctx, 3);
  PolyUOp *out_val = poly_cumalu(ctx, in, 0, POLY_OP_MAX, false);
  ASSERT_NOT_NULL(out_val);

  PolyUOp *out_buf = poly_buffer_f32(ctx, 3);
  float in_d[3] = {-3, -1, -2}, out[3] = {0};
  PolyUOp *leaves[] = {in};
  float *ld[] = {in_d};
  ASSERT_INT_EQ(realize_uop(ctx, out_val, out_buf, out, leaves, ld, 1), 0);

  float expected[3] = {-3, -1, -1};
  for (int i = 0; i < 3; i++)
    ASSERT_FLOAT_EQ(out[i], expected[i], 1e-5);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, cumalu_max_include_initial) {
  PolyCtx *ctx = poly_ctx_new();

  /* tinygrad: cummax([1,3,2], include_initial=True) -> [-inf, 1, 3] */
  PolyUOp *in = poly_buffer_f32(ctx, 3);
  PolyUOp *out_val = poly_cumalu(ctx, in, 0, POLY_OP_MAX, true);
  ASSERT_NOT_NULL(out_val);

  PolyUOp *out_buf = poly_buffer_f32(ctx, 3);
  float in_d[3] = {1, 3, 2}, out[3] = {0};
  PolyUOp *leaves[] = {in};
  float *ld[] = {in_d};
  ASSERT_INT_EQ(realize_uop(ctx, out_val, out_buf, out, leaves, ld, 1), 0);

  ASSERT_FLOAT_INF(out[0], -1); /* -inf */
  ASSERT_FLOAT_EQ(out[1], 1.0f, 1e-5);
  ASSERT_FLOAT_EQ(out[2], 3.0f, 1e-5);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, cumalu_mul_1d) {
  PolyCtx *ctx = poly_ctx_new();

  /* tinygrad: cumprod([1,2,3,4]) -> [1, 2, 6, 24] */
  PolyUOp *in = poly_buffer_f32(ctx, 4);
  PolyUOp *out_val = poly_cumalu(ctx, in, 0, POLY_OP_MUL, false);
  ASSERT_NOT_NULL(out_val);

  PolyUOp *out_buf = poly_buffer_f32(ctx, 4);
  float in_d[4] = {1, 2, 3, 4}, out[4] = {0};
  PolyUOp *leaves[] = {in};
  float *ld[] = {in_d};
  ASSERT_INT_EQ(realize_uop(ctx, out_val, out_buf, out, leaves, ld, 1), 0);

  float expected[4] = {1, 2, 6, 24};
  for (int i = 0; i < 4; i++)
    ASSERT_FLOAT_EQ(out[i], expected[i], 1e-5);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, cumalu_mul_1d_const) {
  PolyCtx *ctx = poly_ctx_new();

  /* Critical: if pad used 0 instead of 1 (the MUL identity), the cumulative
   * product would be 0 everywhere.
   * tinygrad: cumprod([2,2,2]) -> [2, 4, 8] */
  PolyUOp *in = poly_buffer_f32(ctx, 3);
  PolyUOp *out_val = poly_cumalu(ctx, in, 0, POLY_OP_MUL, false);
  ASSERT_NOT_NULL(out_val);

  PolyUOp *out_buf = poly_buffer_f32(ctx, 3);
  float in_d[3] = {2, 2, 2}, out[3] = {0};
  PolyUOp *leaves[] = {in};
  float *ld[] = {in_d};
  ASSERT_INT_EQ(realize_uop(ctx, out_val, out_buf, out, leaves, ld, 1), 0);

  float expected[3] = {2, 4, 8};
  for (int i = 0; i < 3; i++)
    ASSERT_FLOAT_EQ(out[i], expected[i], 1e-5);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, cumalu_mul_include_initial) {
  PolyCtx *ctx = poly_ctx_new();

  /* tinygrad: cumprod([2,3,4], include_initial=True) -> [1, 2, 6] */
  PolyUOp *in = poly_buffer_f32(ctx, 3);
  PolyUOp *out_val = poly_cumalu(ctx, in, 0, POLY_OP_MUL, true);
  ASSERT_NOT_NULL(out_val);

  PolyUOp *out_buf = poly_buffer_f32(ctx, 3);
  float in_d[3] = {2, 3, 4}, out[3] = {0};
  PolyUOp *leaves[] = {in};
  float *ld[] = {in_d};
  ASSERT_INT_EQ(realize_uop(ctx, out_val, out_buf, out, leaves, ld, 1), 0);

  float expected[3] = {1, 2, 6};
  for (int i = 0; i < 3; i++)
    ASSERT_FLOAT_EQ(out[i], expected[i], 1e-5);

  poly_ctx_destroy(ctx);
  PASS();
}

/* ═══════════════════════════════════════════════════════════════════════ */
/*  Pure-UOp poly_full / poly_arange numerical equivalence tests          */
/*                                                                        */
/*  Verify that the rewritten helpers produce the same values as the      */
/*  old host-materialized versions. Reference values are independent of   */
/*  tinygrad here -- they are just `start + i*step` for arange and        */
/*  `value` everywhere for full. The cumalu path is currently O(N^2)      */
/*  until the range-collapse simplify pass lands in Phase D, so keep the  */
/*  test sizes small.                                                     */
/* ═══════════════════════════════════════════════════════════════════════ */

TEST(pe, full_pure_uop_1d) {
  PolyCtx *ctx = poly_ctx_new();
  int64_t shape[1] = {5};
  PolyUOp *out_val = poly_full(ctx, shape, 1, 7.5);
  ASSERT_NOT_NULL(out_val);

  PolyUOp *out_buf = poly_buffer_f32(ctx, 5);
  float out[5] = {0};
  ASSERT_INT_EQ(realize_uop(ctx, out_val, out_buf, out, NULL, NULL, 0), 0);
  for (int i = 0; i < 5; i++)
    ASSERT_FLOAT_EQ(out[i], 7.5f, 1e-5);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, full_pure_uop_2d) {
  PolyCtx *ctx = poly_ctx_new();
  int64_t shape[2] = {3, 4};
  PolyUOp *out_val = poly_full(ctx, shape, 2, -2.0);
  ASSERT_NOT_NULL(out_val);

  PolyUOp *out_buf = poly_buffer_f32(ctx, 12);
  float out[12] = {0};
  ASSERT_INT_EQ(realize_uop(ctx, out_val, out_buf, out, NULL, NULL, 0), 0);
  for (int i = 0; i < 12; i++)
    ASSERT_FLOAT_EQ(out[i], -2.0f, 1e-5);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, full_pure_uop_zero_size) {
  PolyCtx *ctx = poly_ctx_new();
  int64_t shape[1] = {0};
  PolyUOp *out_val = poly_full(ctx, shape, 1, 1.0);
  ASSERT_NOT_NULL(out_val); /* should return an empty buffer, not NULL */
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, arange_pure_uop_simple) {
  PolyCtx *ctx = poly_ctx_new();
  /* arange(0, 5, 1) -> [0, 1, 2, 3, 4] */
  PolyUOp *out_val = poly_arange(ctx, 0.0, 5.0, 1.0);
  ASSERT_NOT_NULL(out_val);

  PolyUOp *out_buf = poly_buffer_f32(ctx, 5);
  float out[5] = {0};
  ASSERT_INT_EQ(realize_uop(ctx, out_val, out_buf, out, NULL, NULL, 0), 0);
  float expected[5] = {0, 1, 2, 3, 4};
  for (int i = 0; i < 5; i++)
    ASSERT_FLOAT_EQ(out[i], expected[i], 1e-5);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, arange_pure_uop_start_step) {
  PolyCtx *ctx = poly_ctx_new();
  /* arange(2, 8, 3) -> [2, 5] */
  PolyUOp *out_val = poly_arange(ctx, 2.0, 8.0, 3.0);
  ASSERT_NOT_NULL(out_val);

  PolyUOp *out_buf = poly_buffer_f32(ctx, 2);
  float out[2] = {0};
  ASSERT_INT_EQ(realize_uop(ctx, out_val, out_buf, out, NULL, NULL, 0), 0);
  float expected[2] = {2, 5};
  for (int i = 0; i < 2; i++)
    ASSERT_FLOAT_EQ(out[i], expected[i], 1e-5);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, arange_pure_uop_float_step) {
  PolyCtx *ctx = poly_ctx_new();
  /* arange(1.5, 4.0, 0.5) -> [1.5, 2.0, 2.5, 3.0, 3.5] */
  PolyUOp *out_val = poly_arange(ctx, 1.5, 4.0, 0.5);
  ASSERT_NOT_NULL(out_val);

  PolyUOp *out_buf = poly_buffer_f32(ctx, 5);
  float out[5] = {0};
  ASSERT_INT_EQ(realize_uop(ctx, out_val, out_buf, out, NULL, NULL, 0), 0);
  float expected[5] = {1.5f, 2.0f, 2.5f, 3.0f, 3.5f};
  for (int i = 0; i < 5; i++)
    ASSERT_FLOAT_EQ(out[i], expected[i], 1e-5);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, linspace_pure_uop_basic) {
  PolyCtx *ctx = poly_ctx_new();
  /* tinygrad: linspace(0, 10, 5) -> [0, 2.5, 5, 7.5, 10] */
  PolyUOp *out_val = poly_linspace(ctx, 0.0, 10.0, 5);
  ASSERT_NOT_NULL(out_val);

  PolyUOp *out_buf = poly_buffer_f32(ctx, 5);
  float out[5] = {0};
  ASSERT_INT_EQ(realize_uop(ctx, out_val, out_buf, out, NULL, NULL, 0), 0);
  float expected[5] = {0.0f, 2.5f, 5.0f, 7.5f, 10.0f};
  for (int i = 0; i < 5; i++)
    ASSERT_FLOAT_EQ(out[i], expected[i], 1e-5);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, linspace_pure_uop_negative_range) {
  PolyCtx *ctx = poly_ctx_new();
  /* tinygrad: linspace(-1, 1, 5) -> [-1, -0.5, 0, 0.5, 1] */
  PolyUOp *out_val = poly_linspace(ctx, -1.0, 1.0, 5);
  ASSERT_NOT_NULL(out_val);

  PolyUOp *out_buf = poly_buffer_f32(ctx, 5);
  float out[5] = {0};
  ASSERT_INT_EQ(realize_uop(ctx, out_val, out_buf, out, NULL, NULL, 0), 0);
  float expected[5] = {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f};
  for (int i = 0; i < 5; i++)
    ASSERT_FLOAT_EQ(out[i], expected[i], 1e-5);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, linspace_pure_uop_single_step) {
  PolyCtx *ctx = poly_ctx_new();
  /* tinygrad: linspace(0, 1, 1) -> [0] */
  PolyUOp *out_val = poly_linspace(ctx, 0.0, 1.0, 1);
  ASSERT_NOT_NULL(out_val);

  PolyUOp *out_buf = poly_buffer_f32(ctx, 1);
  float out[1] = {0};
  ASSERT_INT_EQ(realize_uop(ctx, out_val, out_buf, out, NULL, NULL, 0), 0);
  ASSERT_FLOAT_EQ(out[0], 0.0f, 1e-5);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, arange_pure_uop_negative_step) {
  PolyCtx *ctx = poly_ctx_new();
  /* arange(5, 0, -1) -> [5, 4, 3, 2, 1] */
  PolyUOp *out_val = poly_arange(ctx, 5.0, 0.0, -1.0);
  ASSERT_NOT_NULL(out_val);

  PolyUOp *out_buf = poly_buffer_f32(ctx, 5);
  float out[5] = {0};
  ASSERT_INT_EQ(realize_uop(ctx, out_val, out_buf, out, NULL, NULL, 0), 0);
  float expected[5] = {5, 4, 3, 2, 1};
  for (int i = 0; i < 5; i++)
    ASSERT_FLOAT_EQ(out[i], expected[i], 1e-5);

  poly_ctx_destroy(ctx);
  PASS();
}

/* ═══════════════════════════════════════════════════════════════════════ */
/*  Pure-UOp poly_eye / poly_tril / poly_triu tests -- tinygrad-verified  */
/* ═══════════════════════════════════════════════════════════════════════ */

TEST(pe, eye_pure_uop_3) {
  PolyCtx *ctx = poly_ctx_new();
  /* tinygrad: eye(3) -> [[1,0,0],[0,1,0],[0,0,1]] */
  PolyUOp *out_val = poly_eye(ctx, 3);
  ASSERT_NOT_NULL(out_val);

  PolyUOp *out_buf = poly_buffer_f32(ctx, 9);
  float out[9] = {0};
  ASSERT_INT_EQ(realize_uop(ctx, out_val, out_buf, out, NULL, NULL, 0), 0);
  float expected[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  for (int i = 0; i < 9; i++)
    ASSERT_FLOAT_EQ(out[i], expected[i], 1e-5);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, eye_pure_uop_4) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *out_val = poly_eye(ctx, 4);
  ASSERT_NOT_NULL(out_val);

  PolyUOp *out_buf = poly_buffer_f32(ctx, 16);
  float out[16] = {0};
  ASSERT_INT_EQ(realize_uop(ctx, out_val, out_buf, out, NULL, NULL, 0), 0);
  float expected[16] = {
      1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
  };
  for (int i = 0; i < 16; i++)
    ASSERT_FLOAT_EQ(out[i], expected[i], 1e-5);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, triu_frontend_uses_int_mask_and_broadcast_zero) {
  PolyCtx *ctx = poly_ctx_new();
  int64_t shape[2] = {3, 4};
  PolyUOp *in = make_buf(ctx, shape, 2);
  PolyUOp *out_val = poly_triu(ctx, in, 0);
  ASSERT_NOT_NULL(out_val);

  ASSERT_INT_EQ(out_val->op, POLY_OP_WHERE);
  ASSERT_TRUE(poly_dtype_eq(out_val->dtype, POLY_FLOAT32));

  PolyUOp *mask = out_val->src[0];
  ASSERT_NOT_NULL(mask);
  ASSERT_INT_EQ(mask->op, POLY_OP_CMPNE);
  ASSERT_TRUE(poly_dtype_eq(mask->dtype, POLY_BOOL));

  PolyUOp *lt = mask->src[0];
  ASSERT_NOT_NULL(lt);
  ASSERT_INT_EQ(lt->op, POLY_OP_CMPLT);
  ASSERT_TRUE(poly_dtype_eq(lt->dtype, POLY_BOOL));
  ASSERT_TRUE(poly_dtype_eq(lt->src[0]->dtype, POLY_INT32));
  ASSERT_TRUE(poly_dtype_eq(lt->src[1]->dtype, POLY_INT32));

  ASSERT_NOT_NULL(out_val->src[1]);
  ASSERT_INT_EQ(out_val->src[1]->op, POLY_OP_RESHAPE);

  PolyUOp *zero = out_val->src[2];
  ASSERT_NOT_NULL(zero);
  ASSERT_INT_EQ(zero->op, POLY_OP_EXPAND);
  ASSERT_TRUE(poly_dtype_eq(zero->dtype, POLY_FLOAT32));
  ASSERT_NOT_NULL(zero->src[0]);
  ASSERT_INT_EQ(zero->src[0]->op, POLY_OP_RESHAPE);
  ASSERT_NOT_NULL(zero->src[0]->src[0]);
  ASSERT_INT_EQ(zero->src[0]->src[0]->op, POLY_OP_CONST);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, tril_pure_uop_diag0) {
  PolyCtx *ctx = poly_ctx_new();
  /* tinygrad: arange(1,13).reshape(3,4).tril(0)
   *   -> [[1,0,0,0],[5,6,0,0],[9,10,11,0]] */
  int64_t shape[2] = {3, 4};
  PolyUOp *in = make_buf(ctx, shape, 2);
  PolyUOp *out_val = poly_tril(ctx, in, 0);
  ASSERT_NOT_NULL(out_val);

  PolyUOp *out_buf = poly_buffer_f32(ctx, 12);
  float in_d[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  float out[12] = {0};
  PolyUOp *leaves[] = {base_buf(in)};
  float *ld[] = {in_d};
  ASSERT_INT_EQ(realize_uop(ctx, out_val, out_buf, out, leaves, ld, 1), 0);

  float expected[12] = {1, 0, 0, 0, 5, 6, 0, 0, 9, 10, 11, 0};
  for (int i = 0; i < 12; i++)
    ASSERT_FLOAT_EQ(out[i], expected[i], 1e-5);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, tril_pure_uop_diag_pos1) {
  PolyCtx *ctx = poly_ctx_new();
  /* tinygrad: tril(1) -> [[1,2,0,0],[5,6,7,0],[9,10,11,12]] */
  int64_t shape[2] = {3, 4};
  PolyUOp *in = make_buf(ctx, shape, 2);
  PolyUOp *out_val = poly_tril(ctx, in, 1);
  ASSERT_NOT_NULL(out_val);

  PolyUOp *out_buf = poly_buffer_f32(ctx, 12);
  float in_d[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  float out[12] = {0};
  PolyUOp *leaves[] = {base_buf(in)};
  float *ld[] = {in_d};
  ASSERT_INT_EQ(realize_uop(ctx, out_val, out_buf, out, leaves, ld, 1), 0);

  float expected[12] = {1, 2, 0, 0, 5, 6, 7, 0, 9, 10, 11, 12};
  for (int i = 0; i < 12; i++)
    ASSERT_FLOAT_EQ(out[i], expected[i], 1e-5);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, tril_pure_uop_diag_neg1) {
  PolyCtx *ctx = poly_ctx_new();
  /* tinygrad: tril(-1) -> [[0,0,0,0],[5,0,0,0],[9,10,0,0]] */
  int64_t shape[2] = {3, 4};
  PolyUOp *in = make_buf(ctx, shape, 2);
  PolyUOp *out_val = poly_tril(ctx, in, -1);
  ASSERT_NOT_NULL(out_val);

  PolyUOp *out_buf = poly_buffer_f32(ctx, 12);
  float in_d[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  float out[12] = {0};
  PolyUOp *leaves[] = {base_buf(in)};
  float *ld[] = {in_d};
  ASSERT_INT_EQ(realize_uop(ctx, out_val, out_buf, out, leaves, ld, 1), 0);

  float expected[12] = {0, 0, 0, 0, 5, 0, 0, 0, 9, 10, 0, 0};
  for (int i = 0; i < 12; i++)
    ASSERT_FLOAT_EQ(out[i], expected[i], 1e-5);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, triu_pure_uop_diag0) {
  PolyCtx *ctx = poly_ctx_new();
  /* tinygrad: triu(0) -> [[1,2,3,4],[0,6,7,8],[0,0,11,12]] */
  int64_t shape[2] = {3, 4};
  PolyUOp *in = make_buf(ctx, shape, 2);
  PolyUOp *out_val = poly_triu(ctx, in, 0);
  ASSERT_NOT_NULL(out_val);

  PolyUOp *out_buf = poly_buffer_f32(ctx, 12);
  float in_d[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  float out[12] = {0};
  PolyUOp *leaves[] = {base_buf(in)};
  float *ld[] = {in_d};
  ASSERT_INT_EQ(realize_uop(ctx, out_val, out_buf, out, leaves, ld, 1), 0);

  float expected[12] = {1, 2, 3, 4, 0, 6, 7, 8, 0, 0, 11, 12};
  for (int i = 0; i < 12; i++)
    ASSERT_FLOAT_EQ(out[i], expected[i], 1e-5);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, triu_pure_uop_diag_pos1) {
  PolyCtx *ctx = poly_ctx_new();
  /* tinygrad: triu(1) -> [[0,2,3,4],[0,0,7,8],[0,0,0,12]] */
  int64_t shape[2] = {3, 4};
  PolyUOp *in = make_buf(ctx, shape, 2);
  PolyUOp *out_val = poly_triu(ctx, in, 1);
  ASSERT_NOT_NULL(out_val);

  PolyUOp *out_buf = poly_buffer_f32(ctx, 12);
  float in_d[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  float out[12] = {0};
  PolyUOp *leaves[] = {base_buf(in)};
  float *ld[] = {in_d};
  ASSERT_INT_EQ(realize_uop(ctx, out_val, out_buf, out, leaves, ld, 1), 0);

  float expected[12] = {0, 2, 3, 4, 0, 0, 7, 8, 0, 0, 0, 12};
  for (int i = 0; i < 12; i++)
    ASSERT_FLOAT_EQ(out[i], expected[i], 1e-5);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, triu_pure_uop_diag_neg1) {
  PolyCtx *ctx = poly_ctx_new();
  /* tinygrad: triu(-1) -> [[1,2,3,4],[5,6,7,8],[0,10,11,12]] */
  int64_t shape[2] = {3, 4};
  PolyUOp *in = make_buf(ctx, shape, 2);
  PolyUOp *out_val = poly_triu(ctx, in, -1);
  ASSERT_NOT_NULL(out_val);

  PolyUOp *out_buf = poly_buffer_f32(ctx, 12);
  float in_d[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  float out[12] = {0};
  PolyUOp *leaves[] = {base_buf(in)};
  float *ld[] = {in_d};
  ASSERT_INT_EQ(realize_uop(ctx, out_val, out_buf, out, leaves, ld, 1), 0);

  float expected[12] = {1, 2, 3, 4, 5, 6, 7, 8, 0, 10, 11, 12};
  for (int i = 0; i < 12; i++)
    ASSERT_FLOAT_EQ(out[i], expected[i], 1e-5);
  poly_ctx_destroy(ctx);
  PASS();
}

/* ═══════════════════════════════════════════════════════════════════════ */
/*  Structural gate for the const-registry root fix                       */
/*                                                                        */
/*  poly_arange must lower to a pure RANGE-driven kernel after the        */
/*  range-collapse simplify pass. Verified against tinygrad_latest:       */
/*    arange(5)        -> {RANGE:1, LOAD:0, REDUCE:0, STORE:1}            */
/*    arange(0,5,1)    -> same                                            */
/*    arange(2,8,3)    -> {RANGE:1, LOAD:0, REDUCE:0, STORE:1, ADD:2,     */
/*                         MUL:1}                                         */
/*                                                                        */
/*  Currently FAILS because poly_arange host-materializes via the         */
/*  const-registry. This is the gate for steps 4/5 (rewrite poly_arange   */
/*  via _cumalu) AND step 2 (range-collapse simplify pass) of the         */
/*  const-registry root fix.                                              */
/*                                                                        */
/*  TODO: this currently checks the rewritten tensor sink directly,       */
/*  which bypasses rangeify. Once the reduce-collapse pass is wired into  */
/*  rangeify (codex audit recommended insertion at rangeify.c:2791), this */
/*  test should schedule first then inspect the kernel UOps.              */
/* ═══════════════════════════════════════════════════════════════════════ */
TEST(pe, arange_range_collapse_structural) {
  /* Phase D structural assertion: after pm_reduce_simplify lands in
   * rangeify, poly_arange's REDUCE-based cumsum collapses to a closed-form
   * `i*step + start` expression. Schedule the arange (which runs the full
   * rangeify+reduce_simplify pipeline) and inspect the kernel UOps:
   *
   *   exactly 1 RANGE   (the output index)
   *   0 LOAD            (no buffer reads -- pure compute)
   *   0 REDUCE          (the cumsum REDUCE was eliminated)
   *   1 STORE           (single store per element)
   *
   * Mirrors tinygrad's E_5 kernel for Tensor.arange(5):
   *   *(data0+gidx0) = gidx0;
   */
  PolyCtx *ctx = poly_ctx_new();

  const struct {
    double start, stop, step;
  } cases[] = {
      {0.0, 5.0, 1.0}, /* trivial out[i] = i */
      {2.0, 8.0, 3.0}, /* non-trivial start+step */
  };

  for (size_t k = 0; k < sizeof(cases) / sizeof(cases[0]); k++) {
    PolyUOp *ar = poly_arange(ctx, cases[k].start, cases[k].stop, cases[k].step);
    ASSERT_NOT_NULL(ar);
    int64_t n = (int64_t)((cases[k].stop - cases[k].start) / cases[k].step);
    if (n <= 0) n = 1;
    PolyUOp *out = poly_buffer_f32(ctx, n);
    PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, ar));
    ASSERT_NOT_NULL(sink);

    /* Schedule (runs rangeify + reduce_simplify), then linearize. */
    PolyUOp *kernel = single_scheduled_root(ctx, sink);
    ASSERT_NOT_NULL(kernel);
    int n_lin = 0;
    PolyUOp **lin = poly_linearize(ctx, kernel, &n_lin);
    ASSERT_NOT_NULL(lin);
    ASSERT_TRUE(n_lin > 0);

    int n_range = 0, n_load = 0, n_reduce = 0, n_store = 0;
    for (int i = 0; i < n_lin; i++) {
      switch (lin[i]->op) {
      case POLY_OP_RANGE:
        n_range++;
        break;
      case POLY_OP_LOAD:
        n_load++;
        break;
      case POLY_OP_REDUCE:
        n_reduce++;
        break;
      case POLY_OP_REDUCE_AXIS:
        n_reduce++;
        break;
      case POLY_OP_STORE:
        n_store++;
        break;
      default:
        break;
      }
    }

    ASSERT_INT_EQ(n_range, 1);
    ASSERT_INT_EQ(n_load, 0);
    ASSERT_INT_EQ(n_reduce, 0);
    ASSERT_INT_EQ(n_store, 1);
  }

  poly_ctx_destroy(ctx);
  PASS();
}
