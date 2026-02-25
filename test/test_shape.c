/*
 * test_shape.c — Tests for shape inference
 */

#include "test_harness.h"
#include "../src/polygrad.h"

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

/* ── Shape tests ──────────────────────────────────────────────────────── */

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
  int64_t dims[] = { 10, 10 };
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
  int64_t rdims[] = { 1, 10 };
  PolyUOp *r = poly_uop1(ctx, POLY_OP_RESHAPE, POLY_FLOAT32, buf, int_tuple(rdims, 2));
  int64_t edims[] = { 5, 10 };
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
  int64_t rdims[] = { 10, 10 };
  PolyUOp *r = poly_uop1(ctx, POLY_OP_RESHAPE, POLY_FLOAT32, buf, int_tuple(rdims, 2));
  int64_t axes[] = { 1 };
  PolyUOp *red = poly_uop1(ctx, POLY_OP_REDUCE_AXIS, POLY_FLOAT32, r, reduce_ax(POLY_OP_ADD, axes, 1));
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
  int64_t d1[] = { 5, 4 };
  PolyUOp *r1 = poly_uop1(ctx, POLY_OP_RESHAPE, POLY_FLOAT32, buf1, int_tuple(d1, 2));
  int64_t d2[] = { 1, 4 };
  PolyUOp *r2 = poly_uop1(ctx, POLY_OP_RESHAPE, POLY_FLOAT32, buf2, int_tuple(d2, 2));
  int64_t d3[] = { 5, 4 };
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
