/*
 * test_shape.c — Tests for shape inference
 */

#include "test_harness.h"
#include "../src/polygrad.h"
#include "../src/ctx.h"
#include "../src/device.h"
#include "../src/tensor.h"

/* Helper: create int tuple arg */
static PolyArg int_tuple(int64_t *vals, int n) {
  PolyArg a;
  a.kind = POLY_ARG_INT_TUPLE;
  a.int_tuple.vals = vals;
  a.int_tuple.n = n;
  return a;
}

/* Helper: create reduce_axis arg */
static PolyArg reduce_ax(PolyOps op, int64_t *axes, int n) {
  PolyArg a;
  a.kind = POLY_ARG_REDUCE_AXIS;
  a.reduce_axis.op = op;
  a.reduce_axis.axes = axes;
  a.reduce_axis.n = n;
  return a;
}

static int g_frontend_release_a = 0;
static int g_frontend_release_b = 0;

static void test_frontend_release_a(uintptr_t buffer_key) {
  (void)buffer_key;
  g_frontend_release_a++;
}

static void test_frontend_release_b(uintptr_t buffer_key) {
  (void)buffer_key;
  g_frontend_release_b++;
}

/* Shape tests */

TEST(device, frontend_release_callback_is_captured_per_host_buffer) {
  g_frontend_release_a = 0;
  g_frontend_release_b = 0;

  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  poly_ctx_set_frontend_buffer_release(ctx, test_frontend_release_a);

  PolyUOp *buf = poly_buffer(ctx, POLY_FLOAT32, 1);
  ASSERT_NOT_NULL(buf);
  float data = 1.0f;
  poly_buffer_set(ctx, buf, &data, sizeof(data), POLY_DEVICE_HOST);

  poly_ctx_set_frontend_buffer_release(ctx, test_frontend_release_b);
  poly_ctx_destroy(ctx);

  ASSERT_INT_EQ(g_frontend_release_a, 1);
  ASSERT_INT_EQ(g_frontend_release_b, 0);
  PASS();
}

TEST(shape, buffer) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *buf = poly_uop0(ctx, POLY_OP_BUFFER, POLY_FLOAT32, poly_arg_int(100));
  PolyShape s = poly_uop_shape(ctx, buf);
  ASSERT_INT_EQ(s.ndim, 1);
  ASSERT_INT_EQ(s.dims[0], 100);
  if (s.dims) free(s.dims);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(shape, const_scalar) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *c = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(3.14));
  PolyShape s = poly_uop_shape(ctx, c);
  ASSERT_INT_EQ(s.ndim, 0);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(shape, elementwise) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_uop0(ctx, POLY_OP_BUFFER, POLY_FLOAT32, poly_arg_int(100));
  PolyUOp *b = poly_uop0(ctx, POLY_OP_BUFFER, POLY_FLOAT32, poly_arg_int(100));
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, a, b, poly_arg_none());
  PolyShape s = poly_uop_shape(ctx, add);
  ASSERT_INT_EQ(s.ndim, 1);
  ASSERT_INT_EQ(s.dims[0], 100);
  if (s.dims) free(s.dims);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(shape, reshape) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *buf = poly_uop0(ctx, POLY_OP_BUFFER, POLY_FLOAT32, poly_arg_int(100));
  int64_t dims[] = {10, 10};
  PolyUOp *r = poly_uop1(ctx, POLY_OP_RESHAPE, POLY_FLOAT32, buf, int_tuple(dims, 2));
  PolyShape s = poly_uop_shape(ctx, r);
  ASSERT_INT_EQ(s.ndim, 2);
  ASSERT_INT_EQ(s.dims[0], 10);
  ASSERT_INT_EQ(s.dims[1], 10);
  if (s.dims) free(s.dims);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(shape, expand) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *buf = poly_uop0(ctx, POLY_OP_BUFFER, POLY_FLOAT32, poly_arg_int(10));
  int64_t rdims[] = {1, 10};
  PolyUOp *r = poly_uop1(ctx, POLY_OP_RESHAPE, POLY_FLOAT32, buf, int_tuple(rdims, 2));
  int64_t edims[] = {5, 10};
  PolyUOp *e = poly_uop1(ctx, POLY_OP_EXPAND, POLY_FLOAT32, r, int_tuple(edims, 2));
  PolyShape s = poly_uop_shape(ctx, e);
  ASSERT_INT_EQ(s.ndim, 2);
  ASSERT_INT_EQ(s.dims[0], 5);
  ASSERT_INT_EQ(s.dims[1], 10);
  if (s.dims) free(s.dims);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(shape, reduce_axis) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *buf = poly_uop0(ctx, POLY_OP_BUFFER, POLY_FLOAT32, poly_arg_int(100));
  int64_t rdims[] = {10, 10};
  PolyUOp *r = poly_uop1(ctx, POLY_OP_RESHAPE, POLY_FLOAT32, buf, int_tuple(rdims, 2));
  int64_t axes[] = {1};
  PolyUOp *red =
      poly_uop1(ctx, POLY_OP_REDUCE_AXIS, POLY_FLOAT32, r, reduce_ax(POLY_OP_ADD, axes, 1));
  PolyShape s = poly_uop_shape(ctx, red);
  ASSERT_INT_EQ(s.ndim, 2);
  ASSERT_INT_EQ(s.dims[0], 10);
  ASSERT_INT_EQ(s.dims[1], 1);
  if (s.dims) free(s.dims);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(shape, chain) {
  /* ADD(RESHAPE(buf, (5,4)), EXPAND(RESHAPE(buf2, (1,4)), (5,4))) → (5,4) */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *buf1 = poly_uop0(ctx, POLY_OP_BUFFER, POLY_FLOAT32, poly_arg_int(20));
  PolyUOp *buf2 = poly_uop0(ctx, POLY_OP_BUFFER, POLY_FLOAT32, poly_arg_int(4));
  int64_t d1[] = {5, 4};
  PolyUOp *r1 = poly_uop1(ctx, POLY_OP_RESHAPE, POLY_FLOAT32, buf1, int_tuple(d1, 2));
  int64_t d2[] = {1, 4};
  PolyUOp *r2 = poly_uop1(ctx, POLY_OP_RESHAPE, POLY_FLOAT32, buf2, int_tuple(d2, 2));
  int64_t d3[] = {5, 4};
  PolyUOp *e2 = poly_uop1(ctx, POLY_OP_EXPAND, POLY_FLOAT32, r2, int_tuple(d3, 2));
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, r1, e2, poly_arg_none());
  PolyShape s = poly_uop_shape(ctx, add);
  ASSERT_INT_EQ(s.ndim, 2);
  ASSERT_INT_EQ(s.dims[0], 5);
  ASSERT_INT_EQ(s.dims[1], 4);
  if (s.dims) free(s.dims);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(shape, ensure_shape_uses_rewound_scratch_toposort) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_uop0(ctx, POLY_OP_BUFFER, POLY_FLOAT32, poly_arg_int(8));
  for (int i = 0; i < 32; i++)
    x = poly_uop1(ctx, POLY_OP_NEG, POLY_FLOAT32, x, poly_arg_none());

  size_t scratch_before = poly_arena_used(ctx->scratch);
  PolyShape first = poly_uop_shape(ctx, x);
  ASSERT_INT_EQ(first.ndim, 1);
  ASSERT_INT_EQ(first.dims[0], 8);
  if (first.dims) free(first.dims);
  ASSERT_INT_EQ(poly_arena_used(ctx->scratch), scratch_before);

  size_t main_after_first = poly_arena_used(poly_ctx_arena(ctx));
  PolyShape second = poly_uop_shape(ctx, x);
  ASSERT_INT_EQ(second.ndim, 1);
  ASSERT_INT_EQ(second.dims[0], 8);
  if (second.dims) free(second.dims);
  ASSERT_INT_EQ(poly_arena_used(ctx->scratch), scratch_before);
  ASSERT_INT_EQ(poly_arena_used(poly_ctx_arena(ctx)), main_after_first);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(shape, mismatch) {
  /* ADD(buf_10, buf_20) should detect mismatch (returns NONE) */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_uop0(ctx, POLY_OP_BUFFER, POLY_FLOAT32, poly_arg_int(10));
  PolyUOp *b = poly_uop0(ctx, POLY_OP_BUFFER, POLY_FLOAT32, poly_arg_int(20));
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, a, b, poly_arg_none());
  PolyShape s = poly_uop_shape(ctx, add);
  ASSERT_INT_EQ(s.ndim, -1); /* mismatch → no shape */
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(shape, noshape) {
  /* RANGE has no tensor shape */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(10));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));
  PolyShape s = poly_uop_shape(ctx, range);
  ASSERT_INT_EQ(s.ndim, -1);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(shape, rank_cap_rejects_overrank_movement_constructors) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *buf = poly_buffer(ctx, POLY_FLOAT32, 1);
  ASSERT_NOT_NULL(buf);

  int64_t dims[POLY_MAX_DIMS + 1];
  int64_t pairs[POLY_MAX_DIMS + 1][2];
  for (int i = 0; i < POLY_MAX_DIMS + 1; i++) {
    dims[i] = 1;
    pairs[i][0] = 0;
    pairs[i][1] = 1;
  }

  ASSERT_TRUE(poly_reshape(ctx, buf, dims, POLY_MAX_DIMS + 1) == NULL);
  ASSERT_TRUE(poly_expand(ctx, buf, dims, POLY_MAX_DIMS + 1) == NULL);
  ASSERT_TRUE(poly_permute(ctx, buf, dims, POLY_MAX_DIMS + 1) == NULL);
  ASSERT_TRUE(poly_shrink(ctx, buf, pairs, POLY_MAX_DIMS + 1) == NULL);
  ASSERT_TRUE(poly_pad(ctx, buf, pairs, POLY_MAX_DIMS + 1) == NULL);
  ASSERT_TRUE(poly_flip(ctx, buf, dims, POLY_MAX_DIMS + 1) == NULL);
  ASSERT_TRUE(poly_reduce_axis(ctx, POLY_OP_ADD, buf, dims, POLY_MAX_DIMS + 1) == NULL);
  ASSERT_PTR_EQ(poly_reduce_axis(ctx, POLY_OP_ADD, buf, NULL, 0), buf);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(shape, rank_cap_rejects_low_level_overrank_uops) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *buf = poly_buffer(ctx, POLY_FLOAT32, 1);
  ASSERT_NOT_NULL(buf);

  int64_t dims[POLY_MAX_DIMS + 1];
  int64_t pairs[POLY_MAX_DIMS + 1][2];
  for (int i = 0; i < POLY_MAX_DIMS + 1; i++) {
    dims[i] = 1;
    pairs[i][0] = 0;
    pairs[i][1] = 1;
  }
  PolyArg tuple = int_tuple(dims, POLY_MAX_DIMS + 1);
  PolyArg pair_tuple;
  pair_tuple.kind = POLY_ARG_PAIR_TUPLE;
  pair_tuple.pair_tuple.pairs = pairs;
  pair_tuple.pair_tuple.n = POLY_MAX_DIMS + 1;
  PolyArg reduce = reduce_ax(POLY_OP_ADD, dims, POLY_MAX_DIMS + 1);

  ASSERT_TRUE(poly_uop1(ctx, POLY_OP_RESHAPE, POLY_FLOAT32, buf, tuple) == NULL);
  ASSERT_TRUE(poly_uop1(ctx, POLY_OP_EXPAND, POLY_FLOAT32, buf, tuple) == NULL);
  ASSERT_TRUE(poly_uop1(ctx, POLY_OP_PERMUTE, POLY_FLOAT32, buf, tuple) == NULL);
  ASSERT_TRUE(poly_uop1(ctx, POLY_OP_FLIP, POLY_FLOAT32, buf, tuple) == NULL);
  ASSERT_TRUE(poly_uop1(ctx, POLY_OP_SHRINK, POLY_FLOAT32, buf, pair_tuple) == NULL);
  ASSERT_TRUE(poly_uop1(ctx, POLY_OP_PAD, POLY_FLOAT32, buf, pair_tuple) == NULL);
  ASSERT_TRUE(poly_uop1(ctx, POLY_OP_REDUCE_AXIS, POLY_FLOAT32, buf, reduce) == NULL);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(shape, rank_cap_rejects_ffi_host_and_dynamic_buffers) {
  PolyCtx *ctx = poly_ctx_new();
  float data[1] = {0.0f};
  int64_t dims[POLY_MAX_DIMS + 1];
  for (int i = 0; i < POLY_MAX_DIMS + 1; i++)
    dims[i] = 1;

  ASSERT_TRUE(
      poly_buffer_from_host(
          ctx, data, sizeof(data), poly_dtype_id_by_name("float32"), dims, POLY_MAX_DIMS + 1
      ) == NULL
  );

  PolyUOp *n = poly_uop0(ctx, POLY_OP_DEFINE_VAR, POLY_INT32, poly_arg_define_var("N", 1, 4));
  ASSERT_NOT_NULL(n);
  ASSERT_TRUE(poly_buffer_var(ctx, POLY_FLOAT32, n, dims, POLY_MAX_DIMS) == NULL);
  ASSERT_TRUE(poly_buffer_var(ctx, POLY_FLOAT32, n, NULL, 1) == NULL);

  poly_ctx_destroy(ctx);
  PASS();
}
