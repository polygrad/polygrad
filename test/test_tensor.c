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
#include "../src/scheduler.h"

/* ── Helper: realize a UOp into a float array ─────────────────────────── */

static int realize_uop(PolyCtx *ctx, PolyUOp *val, PolyUOp *out_buf,
                       float *out_data, PolyUOp **leaf_bufs, float **leaf_datas,
                       int n_leaves) {
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
  return poly_realize_flat(ctx, sink, bufs, datas, n);
}

/* ── Helper: make a shaped buffer (RESHAPE(BUFFER, shape)) ────────────── */

static PolyUOp *make_buf(PolyCtx *ctx, const int64_t *shape, int ndim) {
  int64_t numel = 1;
  for (int i = 0; i < ndim; i++) numel *= shape[i];
  PolyUOp *buf = poly_buffer_f32(ctx, numel);
  if (ndim > 1) return poly_reshape(ctx, buf, (int64_t *)shape, ndim);
  return buf;
}

/* Get underlying BUFFER from possibly-reshaped UOp */
static PolyUOp *base_buf(PolyUOp *u) {
  while (u->op == POLY_OP_RESHAPE && u->n_src > 0) u = u->src[0];
  return u;
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
  PolyUOp *r = poly_rmsnorm_v2(ctx, x, w, 1e-5);
  ASSERT_NOT_NULL(r);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, r), 2);

  float dx[] = {1,2,3, 4,5,6}, dw[] = {1,1,1}, dout[6] = {0};
  PolyUOp *leaves[] = {base_buf(x), w};
  float *ld[] = {dx, dw};
  ASSERT_INT_EQ(realize_uop(ctx, r, out_buf, dout, leaves, ld, 2), 0);
  ASSERT_FLOAT_EQ(dout[0], 0.46290955f, 1e-4);
  ASSERT_FLOAT_EQ(dout[3], 0.78954184f, 1e-4);
  ASSERT_FLOAT_EQ(dout[5], 1.18431280f, 1e-4);
  poly_ctx_destroy(ctx); PASS();
}

TEST(pe, sdpa_e2e) {
  /* Reference: [[[1.6605, 2.6605], [2.3395, 3.3395]]] */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *q = make_buf(ctx, (int64_t[]){1, 2, 2}, 3);
  PolyUOp *k = make_buf(ctx, (int64_t[]){1, 2, 2}, 3);
  PolyUOp *v = make_buf(ctx, (int64_t[]){1, 2, 2}, 3);
  PolyUOp *out_buf = poly_buffer_f32(ctx, 4);
  PolyUOp *r = poly_sdpa_v2(ctx, q, k, v, NULL, 0);
  ASSERT_NOT_NULL(r);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, r), 3);

  float dq[] = {1,0, 0,1}, dk[] = {1,0, 0,1}, dv[] = {1,2, 3,4}, dout[4] = {0};
  PolyUOp *leaves[] = {base_buf(q), base_buf(k), base_buf(v)};
  float *ld[] = {dq, dk, dv};
  ASSERT_INT_EQ(realize_uop(ctx, r, out_buf, dout, leaves, ld, 3), 0);
  ASSERT_FLOAT_EQ(dout[0], 1.6604769f, 1e-3);
  ASSERT_FLOAT_EQ(dout[3], 3.3395231f, 1e-3);
  poly_ctx_destroy(ctx); PASS();
}

TEST(pe, sdpa_causal_e2e) {
  /* Reference: [[[1.0, 0.0], [0.3302, 0.6698], [0.7517, 0.7517]]] */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *q = make_buf(ctx, (int64_t[]){1, 3, 2}, 3);
  PolyUOp *k = make_buf(ctx, (int64_t[]){1, 3, 2}, 3);
  PolyUOp *v = make_buf(ctx, (int64_t[]){1, 3, 2}, 3);
  PolyUOp *out_buf = poly_buffer_f32(ctx, 6);
  PolyUOp *r = poly_sdpa_v2(ctx, q, k, v, NULL, 1);

  float dq[] = {1,0, 0,1, 1,1}, dk[] = {1,0, 0,1, 1,1}, dv[] = {1,0, 0,1, 1,1};
  float dout[6] = {0};
  PolyUOp *leaves[] = {base_buf(q), base_buf(k), base_buf(v)};
  float *ld[] = {dq, dk, dv};
  ASSERT_INT_EQ(realize_uop(ctx, r, out_buf, dout, leaves, ld, 3), 0);
  ASSERT_FLOAT_EQ(dout[0], 1.0f, 1e-3);
  ASSERT_FLOAT_EQ(dout[2], 0.33023846f, 1e-3);
  ASSERT_FLOAT_EQ(dout[4], 0.7517449f, 1e-3);
  poly_ctx_destroy(ctx); PASS();
}

TEST(pe, repeat_interleave_e2e) {
  /* Reference: [[1,1,2,2,3,3], [4,4,5,5,6,6]] */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = make_buf(ctx, (int64_t[]){2, 3}, 2);
  PolyUOp *out_buf = poly_buffer_f32(ctx, 12);
  PolyUOp *r = poly_repeat_interleave_v2(ctx, x, 2, 1);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, r), 2);
  ASSERT_INT_EQ(poly_uop_dims(ctx, r)[1], 6);

  float dx[] = {1,2,3, 4,5,6}, dout[12] = {0};
  PolyUOp *leaves[] = {base_buf(x)};
  float *ld[] = {dx};
  ASSERT_INT_EQ(realize_uop(ctx, r, out_buf, dout, leaves, ld, 1), 0);
  float expected[] = {1,1,2,2,3,3, 4,4,5,5,6,6};
  for (int i = 0; i < 12; i++) ASSERT_FLOAT_EQ(dout[i], expected[i], 1e-6);
  poly_ctx_destroy(ctx); PASS();
}

TEST(pe, argmax_e2e) {
  /* Reference: argmax([1,5,3,2,4]) = 1 */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_buffer_f32(ctx, 5);
  PolyUOp *out_buf = poly_buffer_f32(ctx, 1);
  PolyUOp *r = poly_argmax_v2(ctx, x, 0);
  /* Cast int32 result to float for output */
  r = poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, r, poly_arg_none());

  float dx[] = {1,5,3,2,4}, dout[1] = {0};
  PolyUOp *leaves[] = {x};
  float *ld[] = {dx};
  ASSERT_INT_EQ(realize_uop(ctx, r, out_buf, dout, leaves, ld, 1), 0);
  ASSERT_FLOAT_EQ(dout[0], 1.0f, 1e-4);
  poly_ctx_destroy(ctx); PASS();
}

TEST(pe, argmax_2d_e2e) {
  /* Reference: argmax([[1,5,3],[4,2,6]], axis=1) = [1, 2] */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = make_buf(ctx, (int64_t[]){2, 3}, 2);
  PolyUOp *out_buf = poly_buffer_f32(ctx, 2);
  PolyUOp *r = poly_argmax_v2(ctx, x, 1);
  r = poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, r, poly_arg_none());

  float dx[] = {1,5,3, 4,2,6}, dout[2] = {0};
  PolyUOp *leaves[] = {base_buf(x)};
  float *ld[] = {dx};
  ASSERT_INT_EQ(realize_uop(ctx, r, out_buf, dout, leaves, ld, 1), 0);
  ASSERT_FLOAT_EQ(dout[0], 1.0f, 1e-4);
  ASSERT_FLOAT_EQ(dout[1], 2.0f, 1e-4);
  poly_ctx_destroy(ctx); PASS();
}

TEST(pe, mse_loss_e2e) {
  /* mse([1,2,3], [4,5,6]) = mean([9,9,9]) = 9 */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *pred = poly_buffer_f32(ctx, 3);
  PolyUOp *tgt = poly_buffer_f32(ctx, 3);
  PolyUOp *out_buf = poly_buffer_f32(ctx, 1);
  PolyUOp *r = poly_mse_loss_v2(ctx, pred, tgt);

  float dp[] = {1,2,3}, dt[] = {4,5,6}, dout[1] = {0};
  PolyUOp *leaves[] = {pred, tgt};
  float *ld[] = {dp, dt};
  ASSERT_INT_EQ(realize_uop(ctx, r, out_buf, dout, leaves, ld, 2), 0);
  ASSERT_FLOAT_EQ(dout[0], 9.0f, 1e-4);
  poly_ctx_destroy(ctx); PASS();
}

TEST(pe, softmax_v2_e2e) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_buffer_f32(ctx, 3);
  PolyUOp *out_buf = poly_buffer_f32(ctx, 3);
  PolyUOp *sm = poly_softmax_v2(ctx, x, 0);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, sm), 1);

  float dx[] = {1,2,3}, dout[3] = {0};
  poly_realize_begin(ctx);
  poly_realize_bind(ctx, x, dx);
  poly_realize_bind(ctx, out_buf, dout);
  PolyUOp *store = poly_store_val(ctx, out_buf, sm);
  ASSERT_INT_EQ(poly_realize_exec(ctx, poly_sink1(ctx, store)), 0);
  ASSERT_FLOAT_EQ(dout[0], 0.0900f, 1e-3);
  ASSERT_FLOAT_EQ(dout[2], 0.6652f, 1e-3);
  poly_ctx_destroy(ctx); PASS();
}

TEST(pe, dot_v2_e2e) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = make_buf(ctx, (int64_t[]){2, 3}, 2);
  PolyUOp *b = make_buf(ctx, (int64_t[]){3, 2}, 2);
  PolyUOp *r = poly_dot_v2(ctx, a, b);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, r), 2);
  ASSERT_INT_EQ(poly_uop_dims(ctx, r)[0], 2);
  ASSERT_INT_EQ(poly_uop_dims(ctx, r)[1], 2);

  PolyUOp *out_buf = poly_buffer_f32(ctx, 4);
  float da[] = {1,2,3, 4,5,6}, db[] = {1,2, 3,4, 5,6}, dout[4] = {0};
  PolyUOp *leaves[] = {base_buf(a), base_buf(b)};
  float *ld[] = {da, db};
  ASSERT_INT_EQ(realize_uop(ctx, r, out_buf, dout, leaves, ld, 2), 0);
  ASSERT_FLOAT_EQ(dout[0], 22.0f, 1e-4);
  ASSERT_FLOAT_EQ(dout[3], 64.0f, 1e-4);
  poly_ctx_destroy(ctx); PASS();
}

TEST(pe, layernorm_v2_shape) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = make_buf(ctx, (int64_t[]){2, 3}, 2);
  PolyUOp *r = poly_layernorm_v2(ctx, x, -1, 1e-5);
  ASSERT_NOT_NULL(r);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, r), 2);
  ASSERT_INT_EQ(poly_uop_dims(ctx, r)[0], 2);
  ASSERT_INT_EQ(poly_uop_dims(ctx, r)[1], 3);
  poly_ctx_destroy(ctx); PASS();
}

TEST(pe, rope_e2e) {
  /* Reference (tinygrad): pos=0 → [1,0,0,1], pos=1 → [0.5403,-0.01,0.8415,0.9999] */
  PolyCtx *ctx = poly_ctx_new();
  int64_t dim = 4, seq = 3, half = 2;

  float cos_data[6], sin_data[6];
  double theta = 10000.0;
  for (int64_t i = 0; i < seq; i++)
    for (int64_t j = 0; j < half; j++) {
      double freq = 1.0 / pow(theta, (double)(2*j) / (double)dim);
      double angle = (double)i * freq;
      cos_data[i*half+j] = (float)cos(angle);
      sin_data[i*half+j] = (float)sin(angle);
    }

  PolyUOp *x = make_buf(ctx, (int64_t[]){1,1,3,4}, 4);
  PolyUOp *fc_buf = poly_buffer_f32(ctx, 6);
  PolyUOp *fs_buf = poly_buffer_f32(ctx, 6);
  PolyUOp *fc = poly_reshape(ctx, fc_buf, (int64_t[]){1,1,3,2}, 4);
  PolyUOp *fs = poly_reshape(ctx, fs_buf, (int64_t[]){1,1,3,2}, 4);
  PolyUOp *out_buf = poly_buffer_f32(ctx, 12);
  PolyUOp *r = poly_rope_v2(ctx, x, fc, fs);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, r), 4);
  ASSERT_INT_EQ(poly_uop_dims(ctx, r)[3], 4);

  float dx[] = {1,0,0,1, 1,0,0,1, 1,0,0,1}, dout[12] = {0};
  PolyUOp *leaves[] = {base_buf(x), fc_buf, fs_buf};
  float *ld[] = {dx, cos_data, sin_data};
  ASSERT_INT_EQ(realize_uop(ctx, r, out_buf, dout, leaves, ld, 3), 0);
  ASSERT_FLOAT_EQ(dout[0], 1.0f, 1e-3);
  ASSERT_FLOAT_EQ(dout[4], 0.5403f, 1e-3);
  ASSERT_FLOAT_EQ(dout[6], 0.8415f, 1e-3);
  poly_ctx_destroy(ctx); PASS();
}

/* ═══════════════════════════════════════════════════════════════════════ */
/*  Shape-on-UOp rule tests                                               */
/* ═══════════════════════════════════════════════════════════════════════ */

TEST(shape_uop, buffer_static) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *b = poly_buffer(ctx, POLY_FLOAT32, 100);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, b), 1);
  ASSERT_INT_EQ(poly_uop_dims(ctx, b)[0], 100);
  poly_ctx_destroy(ctx); PASS();
}

TEST(shape_uop, buffer_dynamic) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *var = poly_define_var(ctx, "batch", 1, 32);
  PolyUOp *buf = poly_buffer_var(ctx, POLY_FLOAT32, var, (int64_t[]){10}, 1);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, buf), 2);
  ASSERT_INT_EQ(poly_uop_dims(ctx, buf)[0], 32);
  ASSERT_INT_EQ(poly_uop_dims(ctx, buf)[1], 10);
  poly_ctx_destroy(ctx); PASS();
}

TEST(shape_uop, reshape) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *r = poly_reshape(ctx, poly_buffer(ctx, POLY_FLOAT32, 24), (int64_t[]){2,3,4}, 3);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, r), 3);
  ASSERT_INT_EQ(poly_uop_dims(ctx, r)[0], 2);
  ASSERT_INT_EQ(poly_uop_dims(ctx, r)[1], 3);
  ASSERT_INT_EQ(poly_uop_dims(ctx, r)[2], 4);
  poly_ctx_destroy(ctx); PASS();
}

TEST(shape_uop, permute) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *b = make_buf(ctx, (int64_t[]){2,3,4}, 3);
  PolyUOp *p = poly_permute(ctx, b, (int64_t[]){2,0,1}, 3);
  ASSERT_INT_EQ(poly_uop_dims(ctx, p)[0], 4);
  ASSERT_INT_EQ(poly_uop_dims(ctx, p)[1], 2);
  ASSERT_INT_EQ(poly_uop_dims(ctx, p)[2], 3);
  poly_ctx_destroy(ctx); PASS();
}

TEST(shape_uop, pad) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *b = make_buf(ctx, (int64_t[]){2,3}, 2);
  PolyUOp *p = poly_pad(ctx, b, (int64_t[][2]){{1,1},{2,0}}, 2);
  ASSERT_INT_EQ(poly_uop_dims(ctx, p)[0], 4);
  ASSERT_INT_EQ(poly_uop_dims(ctx, p)[1], 5);
  poly_ctx_destroy(ctx); PASS();
}

TEST(shape_uop, shrink) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *b = make_buf(ctx, (int64_t[]){4,5}, 2);
  PolyUOp *s = poly_shrink(ctx, b, (int64_t[][2]){{1,3},{0,4}}, 2);
  ASSERT_INT_EQ(poly_uop_dims(ctx, s)[0], 2);
  ASSERT_INT_EQ(poly_uop_dims(ctx, s)[1], 4);
  poly_ctx_destroy(ctx); PASS();
}

TEST(shape_uop, reduce_axis) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *b = make_buf(ctx, (int64_t[]){2,3,4}, 3);
  PolyUOp *r = poly_reduce_axis(ctx, POLY_OP_ADD, b, (int64_t[]){1}, 1);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, r), 3);
  ASSERT_INT_EQ(poly_uop_dims(ctx, r)[0], 2);
  ASSERT_INT_EQ(poly_uop_dims(ctx, r)[1], 1);
  ASSERT_INT_EQ(poly_uop_dims(ctx, r)[2], 4);
  poly_ctx_destroy(ctx); PASS();
}

TEST(shape_uop, alu_broadcast_scalar) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = make_buf(ctx, (int64_t[]){3,4}, 2);
  PolyUOp *c = poly_const_float(ctx, 1.0);
  PolyUOp *r = poly_alu2(ctx, POLY_OP_ADD, a, c);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, r), 2);
  ASSERT_INT_EQ(poly_uop_dims(ctx, r)[0], 3);
  ASSERT_INT_EQ(poly_uop_dims(ctx, r)[1], 4);
  poly_ctx_destroy(ctx); PASS();
}

TEST(shape_uop, alu_broadcast_ndim) {
  /* (3,5,1) + (5,4) → (3,5,4) -- the embedding WHERE bug case */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = make_buf(ctx, (int64_t[]){3,5,1}, 3);
  PolyUOp *b = make_buf(ctx, (int64_t[]){5,4}, 2);
  PolyUOp *r = poly_alu2(ctx, POLY_OP_ADD, a, b);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, r), 3);
  ASSERT_INT_EQ(poly_uop_dims(ctx, r)[0], 3);
  ASSERT_INT_EQ(poly_uop_dims(ctx, r)[1], 5);
  ASSERT_INT_EQ(poly_uop_dims(ctx, r)[2], 4);
  poly_ctx_destroy(ctx); PASS();
}

TEST(shape_uop, cmplt_broadcast) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = make_buf(ctx, (int64_t[]){2,3}, 2);
  PolyUOp *c = poly_const_float(ctx, 0.5);
  PolyUOp *r = poly_alu2(ctx, POLY_OP_CMPLT, c, a);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, r), 2);
  poly_ctx_destroy(ctx); PASS();
}

TEST(shape_uop, const_scalar) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_INT_EQ(poly_uop_ndim(ctx, poly_const_float(ctx, 42.0)), 0);
  ASSERT_TRUE(poly_uop_dims(ctx, poly_const_float(ctx, 42.0)) == NULL);
  poly_ctx_destroy(ctx); PASS();
}

TEST(shape_uop, store_inherits_value_shape) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *b = poly_buffer_f32(ctx, 4);
  PolyUOp *st = poly_store_val(ctx, b, poly_const_float(ctx, 1.0));
  /* STORE inherits shape from its value source (src[1]).
   * A scalar const has shape (), so ndim=0. */
  ASSERT_INT_EQ(poly_uop_ndim(ctx, st), 0);
  poly_ctx_destroy(ctx); PASS();
}

TEST(shape_uop, contiguous_passthrough) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *b = make_buf(ctx, (int64_t[]){2,3}, 2);
  PolyUOp *c = poly_uop1(ctx, POLY_OP_CONTIGUOUS, b->dtype, b, poly_arg_none());
  ASSERT_INT_EQ(poly_uop_ndim(ctx, c), 2);
  ASSERT_INT_EQ(poly_uop_dims(ctx, c)[0], 2);
  poly_ctx_destroy(ctx); PASS();
}

TEST(shape_uop, assign_flat) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *buf = poly_buffer(ctx, POLY_FLOAT32, 12);
  PolyUOp *r = make_buf(ctx, (int64_t[]){3,4}, 2);
  PolyUOp *a = poly_assign(ctx, r, poly_alu2(ctx, POLY_OP_ADD, r, poly_const_float(ctx, 1.0)));
  /* ASSIGN normalizes to flat BUFFER (shape fix deferred) */
  ASSERT_INT_EQ(poly_uop_ndim(ctx, a), 1);
  ASSERT_INT_EQ(poly_uop_dims(ctx, a)[0], 12);
  poly_ctx_destroy(ctx); PASS();
}

/* ── Shape parity oracle ──────────────────────────────────────────────── */

static int check_shape_parity(PolyCtx *ctx, PolyUOp *root) {
  int n_topo;
  PolyUOp **topo = poly_toposort(ctx, root, &n_topo);
  int mismatches = 0;
  for (int i = 0; i < n_topo; i++) {
    PolyShape computed = poly_uop_shape(ctx, topo[i]);
    int cached_ndim = poly_uop_ndim(ctx, topo[i]);
    if (cached_ndim != computed.ndim) {
      fprintf(stderr, "  parity: op=%s cached=%d computed=%d\n",
              poly_op_name(topo[i]->op), cached_ndim, computed.ndim);
      mismatches++;
    } else if (cached_ndim > 0 && computed.dims) {
      for (int j = 0; j < cached_ndim; j++)
        if (poly_uop_dims(ctx, topo[i])[j] != computed.dims[j]) {
          fprintf(stderr, "  parity: op=%s dim[%d] cached=%ld computed=%ld\n",
                  poly_op_name(topo[i]->op), j,
                  (long)poly_uop_dims(ctx, topo[i])[j], (long)computed.dims[j]);
          mismatches++; break;
        }
    }
    if (computed.ndim > 0 && computed.dims) free(computed.dims);
  }
  return mismatches;
}

TEST(shape_uop, parity_softmax) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = make_buf(ctx, (int64_t[]){3,4}, 2);
  PolyUOp *sm = poly_softmax(ctx, x, (int64_t[]){3,4}, 2, -1);
  ASSERT_INT_EQ(check_shape_parity(ctx, sm), 0);
  poly_ctx_destroy(ctx); PASS();
}

TEST(shape_uop, parity_cross_entropy) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *logits = make_buf(ctx, (int64_t[]){3,5}, 2);
  PolyUOp *target = poly_buffer_f32(ctx, 3);
  int64_t out_shape[8]; int out_ndim;
  PolyUOp *ce = poly_cross_entropy(ctx, logits, (int64_t[]){3,5}, 2,
                                    target, (int64_t[]){3}, 1, -1, out_shape, &out_ndim);
  ASSERT_INT_EQ(check_shape_parity(ctx, ce), 0);
  poly_ctx_destroy(ctx); PASS();
}

TEST(shape_uop, parity_gather) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *table = make_buf(ctx, (int64_t[]){5,4}, 2);
  PolyUOp *idx = poly_buffer_f32(ctx, 3);
  int64_t out_shape[8]; int out_ndim;
  PolyUOp *g = poly_gather(ctx, table, (int64_t[]){5,4}, 2,
                            idx, (int64_t[]){3}, 1, out_shape, &out_ndim);
  ASSERT_INT_EQ(check_shape_parity(ctx, g), 0);
  poly_ctx_destroy(ctx); PASS();
}

/* ── v2 reduce shape ──────────────────────────────────────────────────── */

TEST(pe, v2_reduce_shape) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = make_buf(ctx, (int64_t[]){3,4}, 2);
  PolyUOp *s = poly_sum_reduce_v2(ctx, x, 1, 0);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, s), 1);
  ASSERT_INT_EQ(poly_uop_dims(ctx, s)[0], 3);
  PolyUOp *sk = poly_sum_reduce_v2(ctx, x, 1, 1);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, sk), 2);
  ASSERT_INT_EQ(poly_uop_dims(ctx, sk)[0], 3);
  ASSERT_INT_EQ(poly_uop_dims(ctx, sk)[1], 1);
  poly_ctx_destroy(ctx); PASS();
}
