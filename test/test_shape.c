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
  PolyShape s = poly_uop_max_shape(ctx, buf);
  ASSERT_INT_EQ(s.ndim, 1);
  ASSERT_INT_EQ(s.dims[0], 100);
  if (s.dims) free(s.dims);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(shape, numel_zero_precedes_overflow_and_nonzero_overflow_is_rejected) {
  int64_t zero_dims[3] = {INT64_MAX, 2, 0};
  PolyShape zero_shape = {.dims = zero_dims, .ndim = 3};
  ASSERT_TRUE(poly_shape_numel(zero_shape) == 0);

  int64_t overflow_dims[2] = {INT64_MAX, 2};
  PolyShape overflow_shape = {.dims = overflow_dims, .ndim = 2};
  ASSERT_TRUE(poly_shape_numel(overflow_shape) == -1);
  PASS();
}

TEST(shape, const_scalar) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *c = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(3.14));
  PolyShape s = poly_uop_max_shape(ctx, c);
  ASSERT_INT_EQ(s.ndim, 0);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(shape, elementwise) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_uop0(ctx, POLY_OP_BUFFER, POLY_FLOAT32, poly_arg_int(100));
  PolyUOp *b = poly_uop0(ctx, POLY_OP_BUFFER, POLY_FLOAT32, poly_arg_int(100));
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, a, b, poly_arg_none());
  PolyShape s = poly_uop_max_shape(ctx, add);
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
  PolyUOp *r = poly_reshape(ctx, buf, dims, 2);
  ASSERT_NOT_NULL(r);
  ASSERT_INT_EQ(r->op, POLY_OP_RESHAPE);
  ASSERT_INT_EQ(r->arg.kind, POLY_ARG_NONE);
  ASSERT_INT_EQ(r->n_src, 2);
  ASSERT_PTR_EQ(r->src[0], buf);
  ASSERT_INT_EQ(r->src[1]->op, POLY_OP_STACK);
  ASSERT_TRUE(poly_dtype_eq(r->src[1]->dtype, poly_dtype_vec(POLY_INDEX, 2)));
  ASSERT_INT_EQ(r->src[1]->n_src, 2);
  ASSERT_INT_EQ(r->src[1]->src[0]->op, POLY_OP_CONST);
  ASSERT_INT_EQ(r->src[1]->src[0]->arg.i, 10);
  ASSERT_INT_EQ(r->src[1]->src[1]->op, POLY_OP_CONST);
  ASSERT_INT_EQ(r->src[1]->src[1]->arg.i, 10);
  PolyShape s = poly_uop_max_shape(ctx, r);
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
  PolyUOp *r = poly_reshape(ctx, buf, rdims, 2);
  int64_t edims[] = {5, 10};
  PolyUOp *e = poly_expand(ctx, r, edims, 2);
  ASSERT_NOT_NULL(e);
  ASSERT_INT_EQ(e->arg.kind, POLY_ARG_NONE);
  ASSERT_INT_EQ(e->n_src, 2);
  ASSERT_PTR_EQ(e->src[0], r);
  ASSERT_INT_EQ(e->src[1]->op, POLY_OP_STACK);
  ASSERT_TRUE(poly_dtype_eq(e->src[1]->dtype, poly_dtype_vec(POLY_INDEX, 2)));
  ASSERT_INT_EQ(e->src[1]->n_src, 2);
  ASSERT_INT_EQ(e->src[1]->src[0]->arg.i, 5);
  ASSERT_INT_EQ(e->src[1]->src[1]->arg.i, 10);
  PolyShape s = poly_uop_max_shape(ctx, e);
  ASSERT_INT_EQ(s.ndim, 2);
  ASSERT_INT_EQ(s.dims[0], 5);
  ASSERT_INT_EQ(s.dims[1], 10);
  if (s.dims) free(s.dims);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(shape, reshape_expand_accept_scalar_shape_sources) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  /* Pinned uop/spec.py:151 permits any shape-value UOp in src[1], and
   * UOp.as_shape treats a scalar expression as one symbolic dimension
   * (uop/ops.py:697-700). */
  PolyUOp *buf = poly_buffer(ctx, POLY_FLOAT32, 8);
  PolyUOp *eight = poly_const_int(ctx, 8);
  ASSERT_NOT_NULL(buf);
  ASSERT_INT_EQ(buf->n_src, 1);
  ASSERT_NOT_NULL(buf->src);
  ASSERT_NOT_NULL(buf->src[0]);
  ASSERT_INT_EQ(buf->src[0]->op, POLY_OP_UNIQUE);
  ASSERT_INT_EQ(buf->src[0]->n_src, 0);
  ASSERT_NOT_NULL(eight);
  ASSERT_INT_EQ(eight->n_src, 0);
  PolyUOp *reshape_src[] = {buf, eight};
  PolyUOp *reshape = poly_uop(
      ctx, POLY_OP_RESHAPE, POLY_FLOAT32, reshape_src, 2, poly_arg_none());
  ASSERT_NOT_NULL(reshape);
  ASSERT_INT_EQ(reshape->n_src, 2);
  ASSERT_NOT_NULL(reshape->src);
  ASSERT_PTR_EQ(reshape->src[0], buf);
  ASSERT_PTR_EQ(reshape->src[1], eight);
  PolyShape reshape_shape = poly_uop_max_shape_cached(ctx, reshape);
  ASSERT_INT_EQ(reshape_shape.ndim, 1);
  ASSERT_INT_EQ(reshape_shape.dims[0], 8);

  PolyUOp *one_shape =
      poly_reshape(ctx, poly_buffer(ctx, POLY_FLOAT32, 1), (int64_t[]){1}, 1);
  PolyUOp *n = poly_uop0(
      ctx, POLY_OP_DEFINE_VAR, POLY_INT32,
      poly_arg_define_var("n", 1, 8));
  PolyUOp *expand_src[] = {one_shape, n};
  PolyUOp *expand = poly_uop(
      ctx, POLY_OP_EXPAND, POLY_FLOAT32, expand_src, 2, poly_arg_none());
  ASSERT_NOT_NULL(expand);
  ASSERT_INT_EQ(expand->n_src, 2);
  ASSERT_PTR_EQ(expand->src[1], n);
  PolyShape expand_shape = poly_uop_max_shape_cached(ctx, expand);
  ASSERT_INT_EQ(expand_shape.ndim, 1);
  ASSERT_INT_EQ(expand_shape.dims[0], 8);
  ASSERT_PTR_EQ(poly_uop_shape_dim(ctx, expand, 0), n);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(shape, reshape_expand_reject_negative_shape_ranges) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  /* Pinned uop/ops.py:330-336 rejects a negative RESHAPE dimension and an
   * EXPAND target whose symbolic range is not wholly nonnegative. */
  PolyUOp *value = poly_buffer(ctx, POLY_FLOAT32, 1);
  PolyUOp *negative = poly_const_int(ctx, -1);
  PolyUOp *reshape_srcs[] = {value, negative};
  PolyUOp *reshape = poly_uop(
      ctx, POLY_OP_RESHAPE, POLY_FLOAT32, reshape_srcs, 2, poly_arg_none());
  ASSERT_NOT_NULL(reshape);
  ASSERT_INT_EQ(poly_uop_max_shape_cached(ctx, reshape).ndim, -1);

  PolyUOp *n = poly_uop0(
      ctx, POLY_OP_DEFINE_VAR, POLY_INT32,
      poly_arg_define_var("n", -1, 8));
  PolyUOp *expand_srcs[] = {value, n};
  PolyUOp *expand = poly_uop(
      ctx, POLY_OP_EXPAND, POLY_FLOAT32, expand_srcs, 2, poly_arg_none());
  ASSERT_NOT_NULL(expand);
  ASSERT_INT_EQ(poly_uop_max_shape_cached(ctx, expand).ndim, -1);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(shape, reshape_cardinality_and_expand_use_exact_dimensions) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  /* Pinned uop/ops.py:330-336 validates the exact canonical shape:
   * RESHAPE products must match, while EXPAND requires exact equality or an
   * exact input singleton. Allocation maxima are not semantic dimensions. */
  PolyUOp *bad_reshape =
      poly_reshape(ctx, poly_buffer(ctx, POLY_FLOAT32, 6), (int64_t[]){2, 4}, 2);
  ASSERT_NOT_NULL(bad_reshape);
  ASSERT_INT_EQ(poly_uop_max_shape_cached(ctx, bad_reshape).ndim, -1);

  const int64_t ranges[][3] = {
      {0, 0, 0}, {2, 2, 2}, {2, 2, 3}, {0, 1, 1}, {0, 1, 4}, {1, 1, 4},
  };
  const bool accepted[] = {true, true, false, false, false, true};
  for (int i = 0; i < (int)(sizeof(ranges) / sizeof(ranges[0])); i++) {
    char name[16];
    snprintf(name, sizeof(name), "n%d", i);
    PolyUOp *n = poly_uop0(
        ctx, POLY_OP_DEFINE_VAR, POLY_INDEX,
        poly_arg_define_var(name, ranges[i][0], ranges[i][1]));
    PolyUOp *source = poly_buffer_var(ctx, POLY_FLOAT32, n, NULL, 0);
    PolyUOp *expanded =
        poly_expand(ctx, source, (int64_t[]){ranges[i][2]}, 1);
    ASSERT_NOT_NULL(expanded);
    PolyShape shape = poly_uop_max_shape_cached(ctx, expanded);
    ASSERT_INT_EQ(shape.ndim, accepted[i] ? 1 : -1);
    if (accepted[i]) ASSERT_INT_EQ(shape.dims[0], ranges[i][2]);
  }

  PolyUOp *n = poly_uop0(
      ctx, POLY_OP_DEFINE_VAR, POLY_INDEX,
      poly_arg_define_var("valid_n", 1, 8));
  PolyUOp *two = poly_const_int(ctx, 2);
  PolyUOp *base =
      poly_reshape(ctx, poly_buffer(ctx, POLY_FLOAT32, 2), (int64_t[]){2, 1}, 2);
  PolyUOp *input_shape_srcs[] = {two, n};
  PolyUOp *input_shape = poly_uop(
      ctx, POLY_OP_STACK, poly_dtype_vec(POLY_INDEX, 2),
      input_shape_srcs, 2, poly_arg_none());
  PolyUOp *expand_srcs[] = {base, input_shape};
  PolyUOp *expanded = poly_uop(
      ctx, POLY_OP_EXPAND, POLY_FLOAT32,
      expand_srcs, 2, poly_arg_none());
  ASSERT_NOT_NULL(expanded);
  ASSERT_INT_EQ(poly_uop_max_shape_cached(ctx, expanded).ndim, 2);

  PolyUOp *output_shape_srcs[] = {n, two};
  PolyUOp *output_shape = poly_uop(
      ctx, POLY_OP_STACK, poly_dtype_vec(POLY_INDEX, 2),
      output_shape_srcs, 2, poly_arg_none());
  PolyUOp *reshape_srcs[] = {expanded, output_shape};
  PolyUOp *valid_reshape = poly_uop(
      ctx, POLY_OP_RESHAPE, POLY_FLOAT32,
      reshape_srcs, 2, poly_arg_none());
  ASSERT_NOT_NULL(valid_reshape);
  PolyShape valid_shape = poly_uop_max_shape_cached(ctx, valid_reshape);
  ASSERT_INT_EQ(valid_shape.ndim, 2);
  ASSERT_INT_EQ(valid_shape.dims[0], 8);
  ASSERT_INT_EQ(valid_shape.dims[1], 2);
  ASSERT_PTR_EQ(poly_uop_shape_dim(ctx, valid_reshape, 0), n);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(shape, reduce_axis) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *buf = poly_uop0(ctx, POLY_OP_BUFFER, POLY_FLOAT32, poly_arg_int(100));
  int64_t rdims[] = {10, 10};
  PolyUOp *r = poly_reshape(ctx, buf, rdims, 2);
  int64_t axes[] = {1};
  PolyUOp *red =
      poly_uop1(ctx, POLY_OP_REDUCE_AXIS, POLY_FLOAT32, r, reduce_ax(POLY_OP_ADD, axes, 1));
  PolyShape s = poly_uop_max_shape(ctx, red);
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
  PolyUOp *r1 = poly_reshape(ctx, buf1, d1, 2);
  int64_t d2[] = {1, 4};
  PolyUOp *r2 = poly_reshape(ctx, buf2, d2, 2);
  int64_t d3[] = {5, 4};
  PolyUOp *e2 = poly_expand(ctx, r2, d3, 2);
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, r1, e2, poly_arg_none());
  PolyShape s = poly_uop_max_shape(ctx, add);
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
  PolyShape first = poly_uop_max_shape(ctx, x);
  ASSERT_INT_EQ(first.ndim, 1);
  ASSERT_INT_EQ(first.dims[0], 8);
  if (first.dims) free(first.dims);
  ASSERT_INT_EQ(poly_arena_used(ctx->scratch), scratch_before);

  size_t main_after_first = poly_arena_used(poly_ctx_arena(ctx));
  PolyShape second = poly_uop_max_shape(ctx, x);
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
  PolyShape s = poly_uop_max_shape(ctx, add);
  ASSERT_INT_EQ(s.ndim, -1); /* mismatch → no shape */
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(shape, range_is_scalar_like_tinygrad) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(10));
  PolyUOp *range =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_range(0, POLY_AXIS_LOOP));
  PolyShape s = poly_uop_max_shape(ctx, range);
  ASSERT_INT_EQ(s.ndim, 0);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(shape, index_and_stage_match_tinygrad_topology) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *two = poly_const_int(ctx, 2);
  PolyUOp *three = poly_const_int(ctx, 3);
  PolyUOp *four = poly_const_int(ctx, 4);
  PolyUOp *zero = poly_const_int(ctx, 0);
  PolyUOp *shape_src[] = {two, three};
  PolyUOp *shape = poly_uop(
      ctx, POLY_OP_STACK, poly_dtype_vec(POLY_INDEX, 2), shape_src, 2, poly_arg_none()
  );
  PolyParamArg param_arg = {
      .slot = 0,
      .addrspace = POLY_ADDR_GLOBAL,
      .device = POLY_DEVICE_CPU,
  };
  PolyUOp *param = poly_uop1(
      ctx, POLY_OP_PARAM, POLY_FLOAT32, shape, poly_arg_param(&param_arg)
  );
  PolyUOp *r2 = poly_uop1(
      ctx, POLY_OP_RANGE, POLY_INDEX, two, poly_arg_range(0, POLY_AXIS_LOOP)
  );
  PolyUOp *r3 = poly_uop1(
      ctx, POLY_OP_RANGE, POLY_INDEX, three, poly_arg_range(1, POLY_AXIS_LOOP)
  );
  PolyUOp *r4 = poly_uop1(
      ctx, POLY_OP_RANGE, POLY_INDEX, four, poly_arg_range(2, POLY_AXIS_LOOP)
  );

  PolyUOp *partial = poly_uop2(
      ctx, POLY_OP_INDEX, POLY_FLOAT32, param, r2, poly_arg_none()
  );
  PolyUOp *full_src[] = {param, r2, r3};
  PolyUOp *full =
      poly_uop(ctx, POLY_OP_INDEX, POLY_FLOAT32, full_src, 3, poly_arg_none());
  PolyUOp *stage_singleton_src[] = {full, zero};
  PolyUOp *stage_singleton = poly_uop(
      ctx, POLY_OP_STAGE, POLY_FLOAT32, stage_singleton_src, 2,
      poly_arg_bufferize_opts(POLY_DEVICE_CPU, POLY_ADDR_GLOBAL, true)
  );
  PolyUOp *stage_tensor_src[] = {param, r4};
  PolyUOp *stage_tensor = poly_uop(
      ctx, POLY_OP_STAGE, POLY_FLOAT32, stage_tensor_src, 2,
      poly_arg_bufferize_opts(POLY_DEVICE_CPU, POLY_ADDR_GLOBAL, true)
  );

  PolyShape partial_shape = poly_uop_max_shape_cached(ctx, partial);
  ASSERT_INT_EQ(partial_shape.ndim, 1);
  ASSERT_INT_EQ(partial_shape.dims[0], 3);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, full), 0);

  PolyShape singleton_shape = poly_uop_max_shape_cached(ctx, stage_singleton);
  ASSERT_INT_EQ(singleton_shape.ndim, 1);
  ASSERT_INT_EQ(singleton_shape.dims[0], 1);

  PolyShape tensor_shape = poly_uop_max_shape_cached(ctx, stage_tensor);
  ASSERT_INT_EQ(tensor_shape.ndim, 3);
  ASSERT_INT_EQ(tensor_shape.dims[0], 4);
  ASSERT_INT_EQ(tensor_shape.dims[1], 2);
  ASSERT_INT_EQ(tensor_shape.dims[2], 3);

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
  PolyUOp *shape_dims[POLY_MAX_DIMS + 1];
  for (int i = 0; i < POLY_MAX_DIMS + 1; i++)
    shape_dims[i] = poly_const_int(ctx, 1);
  PolyUOp *overrank_shape = poly_uop(
      ctx, POLY_OP_STACK, poly_dtype_vec(POLY_INDEX, POLY_MAX_DIMS + 1),
      shape_dims, POLY_MAX_DIMS + 1, poly_arg_none());
  ASSERT_NOT_NULL(overrank_shape);
  PolyUOp *movement_src[] = {buf, overrank_shape};

  ASSERT_TRUE(poly_uop1(ctx, POLY_OP_RESHAPE, POLY_FLOAT32, buf, tuple) == NULL);
  ASSERT_TRUE(poly_uop1(ctx, POLY_OP_EXPAND, POLY_FLOAT32, buf, tuple) == NULL);
  ASSERT_TRUE(
      poly_uop(
          ctx, POLY_OP_RESHAPE, POLY_FLOAT32,
          movement_src, 2, poly_arg_none()) == NULL);
  ASSERT_TRUE(
      poly_uop(
          ctx, POLY_OP_EXPAND, POLY_FLOAT32,
          movement_src, 2, poly_arg_none()) == NULL);
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

TEST(shape, movement_shape_lanes_use_full_pinned_symbolic_canonicalization) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  /* Pinned tinygrad/uop/ops.py:697-705 applies ssimplify() to every as_shape
   * lane before movement shape inference and rangeify suffix comparison. */
  PolyUOp *n =
      poly_uop0(ctx, POLY_OP_DEFINE_VAR, POLY_INT32, poly_arg_define_var("n", 1, 8));
  PolyUOp *y =
      poly_uop0(ctx, POLY_OP_DEFINE_VAR, POLY_INT32, poly_arg_define_var("y", 1, 8));
  PolyUOp *one = poly_const_int(ctx, 1);
  PolyUOp *two = poly_const_int(ctx, 2);
  PolyUOp *three = poly_const_int(ctx, 3);
  PolyUOp *five = poly_const_int(ctx, 5);
  ASSERT_NOT_NULL(n);

  PolyUOp *left[5] = {
      poly_uop2(ctx, POLY_OP_ADD, POLY_INDEX, n, n, poly_arg_none()),
      poly_uop2(
          ctx, POLY_OP_ADD, POLY_INDEX,
          poly_uop2(ctx, POLY_OP_ADD, POLY_INDEX, n, one, poly_arg_none()),
          two, poly_arg_none()),
      poly_uop2(
          ctx, POLY_OP_MUL, POLY_INDEX, two,
          poly_uop2(ctx, POLY_OP_ADD, POLY_INDEX, n, one, poly_arg_none()),
          poly_arg_none()),
      poly_uop2(
          ctx, POLY_OP_ADD, POLY_INDEX,
          poly_uop2(ctx, POLY_OP_MUL, POLY_INDEX, n, two, poly_arg_none()),
          poly_uop2(ctx, POLY_OP_MUL, POLY_INDEX, n, three, poly_arg_none()),
          poly_arg_none()),
      poly_uop2(
          ctx, POLY_OP_ADD, POLY_INDEX,
          poly_uop2(ctx, POLY_OP_ADD, POLY_INDEX, y, n, poly_arg_none()), n,
          poly_arg_none()),
  };
  PolyUOp *right[5] = {
      poly_uop2(ctx, POLY_OP_MUL, POLY_INDEX, n, two, poly_arg_none()),
      poly_uop2(ctx, POLY_OP_ADD, POLY_INDEX, n, three, poly_arg_none()),
      poly_uop2(
          ctx, POLY_OP_ADD, POLY_INDEX,
          poly_uop2(ctx, POLY_OP_MUL, POLY_INDEX, n, two, poly_arg_none()),
          two, poly_arg_none()),
      poly_uop2(ctx, POLY_OP_MUL, POLY_INDEX, n, five, poly_arg_none()),
      poly_uop2(
          ctx, POLY_OP_ADD, POLY_INDEX, y,
          poly_uop2(ctx, POLY_OP_MUL, POLY_INDEX, n, two, poly_arg_none()),
          poly_arg_none()),
  };

  PolyUOp *base =
      poly_reshape(ctx, poly_buffer(ctx, POLY_FLOAT32, 3), (int64_t[]){3, 1}, 2);
  ASSERT_NOT_NULL(base);
  for (int i = 0; i < 5; i++) {
    PolyUOp *expand_shape_src[] = {three, left[i]};
    PolyUOp *expand_shape = poly_uop(
        ctx, POLY_OP_STACK, poly_dtype_vec(POLY_INDEX, 2),
        expand_shape_src, 2, poly_arg_none());
    PolyUOp *expand_src[] = {base, expand_shape};
    PolyUOp *expanded = poly_uop(
        ctx, POLY_OP_EXPAND, POLY_FLOAT32, expand_src, 2, poly_arg_none());

    PolyUOp *reshape_shape_src[] = {three, right[i]};
    PolyUOp *reshape_shape = poly_uop(
        ctx, POLY_OP_STACK, poly_dtype_vec(POLY_INDEX, 2),
        reshape_shape_src, 2, poly_arg_none());
    PolyUOp *reshape_src[] = {expanded, reshape_shape};
    PolyUOp *reshaped = poly_uop(
        ctx, POLY_OP_RESHAPE, POLY_FLOAT32, reshape_src, 2, poly_arg_none());

    ASSERT_NOT_NULL(expanded);
    ASSERT_NOT_NULL(reshaped);
    ASSERT_INT_EQ(poly_uop_ndim(ctx, expanded), 2);
    ASSERT_INT_EQ(poly_uop_ndim(ctx, reshaped), 2);
    ASSERT_PTR_EQ(poly_uop_shape_dim(ctx, expanded, 1), right[i]);
    ASSERT_PTR_EQ(poly_uop_shape_dim(ctx, reshaped, 1), right[i]);
  }

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(shape, scalar_param_shape_uses_full_pinned_symbolic_canonicalization) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  /* Pinned tinygrad/uop/ops.py:697-700 applies ssimplify to any non-STACK
   * scalar shape source too, not only to STACK lanes. */
  PolyUOp *n =
      poly_uop0(ctx, POLY_OP_DEFINE_VAR, POLY_INT32, poly_arg_define_var("n", 1, 8));
  PolyUOp *one = poly_const_int(ctx, 1);
  PolyUOp *two = poly_const_int(ctx, 2);
  PolyUOp *three = poly_const_int(ctx, 3);
  PolyUOp *source = poly_uop2(
      ctx, POLY_OP_ADD, POLY_INDEX,
      poly_uop2(ctx, POLY_OP_ADD, POLY_INDEX, n, one, poly_arg_none()), two,
      poly_arg_none());
  PolyUOp *expected =
      poly_uop2(ctx, POLY_OP_ADD, POLY_INDEX, n, three, poly_arg_none());
  PolyParamArg arg = {
      .slot = 0,
      .addrspace = POLY_ADDR_GLOBAL,
      .device = POLY_DEVICE_CPU,
  };
  PolyUOp *param =
      poly_uop1(ctx, POLY_OP_PARAM, POLY_FLOAT32, source, poly_arg_param(&arg));

  ASSERT_INT_EQ(poly_uop_ndim(ctx, param), 1);
  ASSERT_PTR_EQ(poly_uop_shape_dim(ctx, param, 0), expected);

  poly_ctx_destroy(ctx);
  PASS();
}
