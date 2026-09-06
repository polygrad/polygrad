/*
 * test_shape.c — Tests for shape inference
 */

#include "test_harness.h"
#include "../src/polygrad.h"
#include "../src/ctx.h"
#include "../src/device.h"
#include "../src/frontend.h"
#include "../src/tensor.h"

/* Helper: create int tuple arg */
static PolyArg int_tuple(int64_t *vals, int n) {
  PolyArg a;
  a.kind = POLY_ARG_INT_TUPLE;
  a.int_tuple.vals = vals;
  a.int_tuple.n = n;
  return a;
}

TEST(shape, current_late_value_and_binary_shapes) {
  /* Tinygrad 2026-08-22/a9069c177a9d uop/ops.py:338-386: value CALL/INS
   * are scalar and BINARY's shape is its byte length. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *call = poly_uop(ctx, POLY_OP_CALL, POLY_FLOAT32, NULL, 0, poly_arg_none());
  PolyUOp *void_call = poly_uop(ctx, POLY_OP_CALL, POLY_VOID, NULL, 0, poly_arg_none());
  PolyUOp *ins = poly_uop(ctx, POLY_OP_INS, POLY_FLOAT32, NULL, 0, poly_arg_int(0));
  PolyUOp *void_ins = poly_uop(ctx, POLY_OP_INS, POLY_VOID, NULL, 0, poly_arg_int(0));
  PolyUOp *binary =
      poly_uop(ctx, POLY_OP_BINARY, POLY_UINT8, NULL, 0, poly_arg_bytes((const uint8_t *)"abc", 3));
  ASSERT_NOT_NULL(call);
  ASSERT_NOT_NULL(void_call);
  ASSERT_NOT_NULL(ins);
  ASSERT_NOT_NULL(void_ins);
  ASSERT_NOT_NULL(binary);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, call), 0);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, void_call), -1);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, ins), 0);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, void_ins), -1);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, binary), 1);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, binary)[0], 3);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(shape, current_buffer_custom_and_noop_shapes) {
  /* Tinygrad 2026-08-22/a9069c177a9d uop/ops.py:346-391: BUFFER reads its
   * shape source, value CUSTOM broadcasts inputs, and RESHAPE(NOOP) uses the
   * requested shape even though the marker has no source shape. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *two = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(2));
  PolyUOp *three = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(3));
  PolyUOp *shape = poly_shape_to_shape_arg(ctx, (PolyUOp *[]){two, three}, 2);
  PolyUOp *buffer = poly_uop1(ctx, POLY_OP_BUFFER, POLY_FLOAT32, shape, poly_arg_none());
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(1.0));
  PolyUOp *vec = poly_uop_stack(ctx, (PolyUOp *[]){one, one}, 2);
  PolyUOp *custom =
      poly_uop(ctx, POLY_OP_CUSTOM, POLY_FLOAT32, (PolyUOp *[]){one, vec}, 2, poly_arg_none());
  PolyUOp *void_custom = poly_uop1(ctx, POLY_OP_CUSTOM, POLY_VOID, vec, poly_arg_none());
  PolyUOp *noop = poly_uop(ctx, POLY_OP_NOOP, POLY_FLOAT32, NULL, 0, poly_arg_none());
  PolyUOp *reshape = poly_uop2(ctx, POLY_OP_RESHAPE, POLY_FLOAT32, noop, shape, poly_arg_none());
  ASSERT_NOT_NULL(buffer);
  ASSERT_NOT_NULL(custom);
  ASSERT_NOT_NULL(void_custom);
  ASSERT_NOT_NULL(noop);
  ASSERT_NOT_NULL(reshape);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, buffer), 2);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, buffer)[0], 2);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, buffer)[1], 3);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, custom), 1);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, custom)[0], 2);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, void_custom), -1);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, noop), -1);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, reshape), 2);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, reshape)[0], 2);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, reshape)[1], 3);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(shape, current_wmma_uses_accumulator_fragment_shape) {
  /* Tinygrad 2026-08-22/a9069c177a9d uop/ops.py:395-397 broadcasts fragment
   * prefixes, then retains the accumulator's final lane dimension. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *half[8], *zero[6];
  for (int i = 0; i < 8; i++)
    half[i] = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT16, poly_arg_float(i));
  for (int i = 0; i < 6; i++)
    zero[i] = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(0));
  PolyUOp *a = poly_reshape(ctx, poly_uop_stack(ctx, half, 4), (int64_t[]){1, 4}, 2);
  PolyUOp *b = poly_reshape(ctx, poly_uop_stack(ctx, half, 8), (int64_t[]){2, 4}, 2);
  PolyUOp *acc = poly_reshape(ctx, poly_uop_stack(ctx, zero, 6), (int64_t[]){2, 3}, 2);
  int dims[] = {1, 1, 1};
  PolyUOp *wmma = poly_uop(
      ctx, POLY_OP_WMMA, POLY_FLOAT32, (PolyUOp *[]){a, b, acc}, 3,
      poly_arg_tensor_core(dims, POLY_FLOAT16, "CUDA", 32, NULL, NULL, false)
  );
  PolyUOp *cast = poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT16, wmma, poly_arg_none());
  ASSERT_NOT_NULL(wmma);
  ASSERT_NOT_NULL(cast);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, wmma), 2);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, wmma)[0], 2);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, wmma)[1], 3);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, cast), 2);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, cast)[0], 2);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, cast)[1], 3);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(shape, symbolic_empty_separates_logical_and_physical_storage) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *n = poly_uop_variable(ctx, "N", 1, 4, POLY_WEAKINT, 1, false);
  PolyUOp *ten = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(10));
  PolyUOp *shape[] = {n, ten};
  PolyTensor *tensor = poly_tensor_empty_uop(ctx, POLY_FLOAT32, shape, 2, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(tensor);
  PolyUOp *logical = poly_tensor_uop_logical(tensor);
  PolyUOp *physical = poly_tensor_uop_physical(tensor);
  ASSERT_NOT_NULL(logical);
  ASSERT_NOT_NULL(physical);
  ASSERT_INT_EQ(logical->op, POLY_OP_SHRINK);
  ASSERT_INT_EQ(physical->op, POLY_OP_SHRINK);
  /* tinygrad@2026-08-22 UOp.has_buffer_identity does not cross SHRINK;
   * UOp.base reaches the storage beneath the symbolic movement graph. */
  const PolyUOp *logical_buffer = poly_uop_base(logical);
  const PolyUOp *physical_buffer = poly_uop_base(physical);
  ASSERT_NOT_NULL(logical_buffer);
  ASSERT_NOT_NULL(physical_buffer);
  ASSERT_INT_EQ(logical_buffer->src[0]->op, POLY_OP_UNIQUE);
  ASSERT_INT_EQ(logical_buffer->arg.kind, POLY_ARG_INT);
  ASSERT_INT_EQ(physical_buffer->src[0]->op, POLY_OP_CONST);
  ASSERT_INT_EQ(physical_buffer->arg.kind, POLY_ARG_PARAM);
  ASSERT_STR_EQ(physical_buffer->arg.param->device, "CPU");
  ASSERT_INT_EQ(logical_buffer->src[0]->arg.i, physical_buffer->arg.param->slot);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(shape, multi_family_matches_pinned_tuple_device_shapes) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *cpu = poly_device_uop_from_name(ctx, "CPU");
  PolyUOp *cpu1 = poly_device_uop_from_name(ctx, "CPU:1");
  PolyUOp *buffer0 = poly_uop_new_buffer(ctx, cpu, 4, POLY_FLOAT32, 201);
  PolyUOp *buffer1 = poly_uop_new_buffer(ctx, cpu1, 4, POLY_FLOAT32, 202);
  PolyUOp *stack_src[] = {buffer0, buffer1};
  PolyUOp *stack = poly_uop(ctx, POLY_OP_MSTACK, POLY_FLOAT32, stack_src, 2, poly_arg_none());
  PolyUOp *select = poly_uop1(ctx, POLY_OP_MSELECT, POLY_FLOAT32, stack, poly_arg_int(1));

  const char *devices[] = {"CPU", "CPU:1"};
  PolyUOp *tuple = poly_device_uop_from_names(ctx, devices, 2);
  PolyUOp *local = poly_uop_new_buffer(ctx, tuple, 2, POLY_FLOAT32, 203);
  PolyUOp *device_range = poly_range(ctx, 2, -1, POLY_AXIS_DEVICE);
  int64_t axis0 = 0;
  PolyUOp *ranges[] = {device_range};
  PolyUOp *multi = poly_unshard(ctx, local, &axis0, ranges, 1);
  PolyUOp *allreduce = poly_allreduce(ctx, multi, POLY_OP_ADD, tuple);

  PolyUOp *roots[] = {stack, select, multi, allreduce};
  for (int i = 0; i < 4; i++) {
    PolyShape shape = poly_uop_max_shape(ctx, roots[i]);
    ASSERT_INT_EQ(shape.ndim, 1);
    ASSERT_INT_EQ(shape.dims[0], 4);
    free(shape.dims);
  }

  int64_t axis1 = 1;
  PolyUOp *bad_axis = poly_unshard(ctx, local, &axis1, ranges, 1);
  PolyShape bad_axis_shape = poly_uop_max_shape(ctx, bad_axis);
  ASSERT_INT_EQ(bad_axis_shape.ndim, -1);
  free(bad_axis_shape.dims);

  poly_ctx_destroy(ctx);
  PASS();
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

  PolyUOp *buf = poly_test_buffer(ctx, POLY_FLOAT32, 1);
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
  PolyUOp *buf = poly_test_buffer(ctx, POLY_FLOAT32, 100);
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
  ASSERT_TRUE(poly_shape_numel_checked(zero_dims, 3) == 0);

  int64_t overflow_dims[2] = {INT64_MAX, 2};
  PolyShape overflow_shape = {.dims = overflow_dims, .ndim = 2};
  ASSERT_TRUE(poly_shape_numel(overflow_shape) == -1);
  ASSERT_TRUE(poly_shape_numel_checked(overflow_dims, 2) == -1);
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
  PolyUOp *a = poly_test_buffer(ctx, POLY_FLOAT32, 100);
  PolyUOp *b = poly_test_buffer(ctx, POLY_FLOAT32, 100);
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
  PolyUOp *buf = poly_test_buffer(ctx, POLY_FLOAT32, 100);
  int64_t dims[] = {10, 10};
  PolyUOp *r = poly_reshape(ctx, buf, dims, 2);
  ASSERT_NOT_NULL(r);
  ASSERT_INT_EQ(r->op, POLY_OP_RESHAPE);
  ASSERT_INT_EQ(r->arg.kind, POLY_ARG_NONE);
  ASSERT_INT_EQ(r->n_src, 2);
  ASSERT_PTR_EQ(r->src[0], buf);
  ASSERT_INT_EQ(r->src[1]->op, POLY_OP_STACK);
  ASSERT_TRUE(poly_dtype_eq(r->src[1]->dtype, POLY_WEAKINT));
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
  PolyUOp *buf = poly_test_buffer(ctx, POLY_FLOAT32, 10);
  int64_t rdims[] = {1, 10};
  PolyUOp *r = poly_reshape(ctx, buf, rdims, 2);
  int64_t edims[] = {5, 10};
  PolyUOp *e = poly_expand(ctx, r, edims, 2);
  ASSERT_NOT_NULL(e);
  ASSERT_INT_EQ(e->op, POLY_OP_EXPAND);
  ASSERT_INT_EQ(e->arg.kind, POLY_ARG_NONE);
  ASSERT_INT_EQ(e->n_src, 2);
  ASSERT_INT_EQ(e->src[0]->op, POLY_OP_RESHAPE);
  ASSERT_PTR_EQ(e->src[0]->src[0], r);
  ASSERT_INT_EQ(e->src[1]->op, POLY_OP_CONST);
  ASSERT_INT_EQ(e->src[1]->arg.i, 5);
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
  PolyUOp *buf = poly_test_buffer(ctx, POLY_FLOAT32, 8);
  PolyUOp *eight = poly_const_int(ctx, 8);
  ASSERT_NOT_NULL(buf);
  ASSERT_INT_EQ(buf->n_src, 1);
  ASSERT_NOT_NULL(buf->src);
  ASSERT_NOT_NULL(buf->src[0]);
  ASSERT_INT_EQ(buf->src[0]->op, POLY_OP_CONST);
  ASSERT_INT_EQ(buf->src[0]->arg.i, 8);
  ASSERT_INT_EQ(buf->arg.kind, POLY_ARG_PARAM);
  ASSERT_NOT_NULL(buf->arg.param);
  ASSERT_INT_EQ(buf->arg.param->slot, 0);
  ASSERT_NOT_NULL(eight);
  ASSERT_INT_EQ(eight->n_src, 0);
  PolyUOp *reshape_src[] = {buf, eight};
  PolyUOp *reshape = poly_uop(ctx, POLY_OP_RESHAPE, POLY_FLOAT32, reshape_src, 2, poly_arg_none());
  ASSERT_NOT_NULL(reshape);
  ASSERT_INT_EQ(reshape->n_src, 2);
  ASSERT_NOT_NULL(reshape->src);
  ASSERT_PTR_EQ(reshape->src[0], buf);
  ASSERT_PTR_EQ(reshape->src[1], eight);
  PolyShape reshape_shape = poly_uop_max_shape_cached(ctx, reshape);
  ASSERT_INT_EQ(reshape_shape.ndim, 1);
  ASSERT_INT_EQ(reshape_shape.dims[0], 8);

  PolyUOp *one_shape = poly_reshape(ctx, poly_test_buffer(ctx, POLY_FLOAT32, 1), (int64_t[]){1}, 1);
  PolyUOp *n = poly_uop_variable(ctx, "n", 1, 8, POLY_INT32, 1, false);
  PolyUOp *expand_src[] = {one_shape, n};
  PolyUOp *expand = poly_uop(ctx, POLY_OP_EXPAND, POLY_FLOAT32, expand_src, 2, poly_arg_none());
  ASSERT_NOT_NULL(expand);
  ASSERT_INT_EQ(expand->n_src, 2);
  ASSERT_PTR_EQ(expand->src[1], n);
  PolyShape expand_shape = poly_uop_max_shape_cached(ctx, expand);
  ASSERT_INT_EQ(expand_shape.ndim, 2);
  ASSERT_INT_EQ(expand_shape.dims[0], 8);
  ASSERT_INT_EQ(expand_shape.dims[1], 1);
  ASSERT_PTR_EQ(poly_uop_shape_dim(ctx, expand, 0), n);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(shape, reshape_expand_reject_negative_shape_ranges) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  /* Pinned uop/ops.py:330-336 rejects a negative RESHAPE dimension and an
   * EXPAND target whose symbolic range is not wholly nonnegative. */
  PolyUOp *value = poly_test_buffer(ctx, POLY_FLOAT32, 1);
  PolyUOp *negative = poly_const_int(ctx, -1);
  PolyUOp *reshape_srcs[] = {value, negative};
  PolyUOp *reshape = poly_uop(ctx, POLY_OP_RESHAPE, POLY_FLOAT32, reshape_srcs, 2, poly_arg_none());
  ASSERT_NOT_NULL(reshape);
  ASSERT_INT_EQ(poly_uop_max_shape_cached(ctx, reshape).ndim, -1);

  PolyUOp *n = poly_uop_variable(ctx, "n", -1, 8, POLY_INT32, 1, false);
  PolyUOp *expand_srcs[] = {value, n};
  PolyUOp *expand = poly_uop(ctx, POLY_OP_EXPAND, POLY_FLOAT32, expand_srcs, 2, poly_arg_none());
  ASSERT_NOT_NULL(expand);
  ASSERT_INT_EQ(poly_uop_max_shape_cached(ctx, expand).ndim, -1);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(shape, reshape_cardinality_and_expand_use_exact_dimensions) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  /* Current MovementMixin._broadcast_to uses structural dimension equality
   * or a literal input singleton (mixin/movement.py:129-148). Allocation
   * maxima and fixed variable ranges are not semantic dimension identities. */
  PolyUOp *bad_reshape =
      poly_reshape(ctx, poly_test_buffer(ctx, POLY_FLOAT32, 6), (int64_t[]){2, 4}, 2);
  ASSERT_NOT_NULL(bad_reshape);
  ASSERT_INT_EQ(poly_uop_max_shape_cached(ctx, bad_reshape).ndim, -1);

  const int64_t ranges[][3] = {
      {0, 0, 0}, {2, 2, 2}, {2, 2, 3}, {0, 1, 1}, {0, 1, 4}, {1, 1, 4},
  };
  for (int i = 0; i < (int)(sizeof(ranges) / sizeof(ranges[0])); i++) {
    char name[16];
    snprintf(name, sizeof(name), "n%d", i);
    PolyUOp *n = poly_uop_variable(ctx, name, ranges[i][0], ranges[i][1], POLY_WEAKINT, 1, false);
    PolyUOp *source = poly_test_buffer_var(ctx, POLY_FLOAT32, n, NULL, 0);
    PolyUOp *expanded = poly_expand(ctx, source, (int64_t[]){ranges[i][2]}, 1);
    ASSERT_TRUE(expanded == NULL);
  }

  PolyUOp *n = poly_uop_variable(ctx, "valid_n", 1, 8, POLY_WEAKINT, 1, false);
  PolyUOp *two = poly_const_int(ctx, 2);
  PolyUOp *base = poly_reshape(ctx, poly_test_buffer(ctx, POLY_FLOAT32, 2), (int64_t[]){2, 1}, 2);
  PolyUOp *input_shape_srcs[] = {two, n};
  PolyUOp *expanded = poly_expand_uop(ctx, base, input_shape_srcs, 2);
  ASSERT_NOT_NULL(expanded);
  ASSERT_INT_EQ(poly_uop_max_shape_cached(ctx, expanded).ndim, 2);

  PolyUOp *output_shape_srcs[] = {n, two};
  PolyUOp *output_shape =
      poly_uop(ctx, POLY_OP_STACK, POLY_WEAKINT, output_shape_srcs, 2, poly_arg_none());
  PolyUOp *reshape_srcs[] = {expanded, output_shape};
  PolyUOp *valid_reshape =
      poly_uop(ctx, POLY_OP_RESHAPE, POLY_FLOAT32, reshape_srcs, 2, poly_arg_none());
  ASSERT_NOT_NULL(valid_reshape);
  PolyShape valid_shape = poly_uop_max_shape_cached(ctx, valid_reshape);
  ASSERT_INT_EQ(valid_shape.ndim, 2);
  ASSERT_INT_EQ(valid_shape.dims[0], 8);
  ASSERT_INT_EQ(valid_shape.dims[1], 2);
  ASSERT_PTR_EQ(poly_uop_shape_dim(ctx, valid_reshape, 0), n);

  PolyUOp *same_source = poly_test_buffer_var(ctx, POLY_FLOAT32, n, NULL, 0);
  ASSERT_PTR_EQ(poly_expand_uop(ctx, same_source, &n, 1), same_source);
  PolyUOp *literal_one = poly_test_buffer(ctx, POLY_FLOAT32, 1);
  PolyUOp *symbolic_expand = poly_expand_uop(ctx, literal_one, &n, 1);
  ASSERT_NOT_NULL(symbolic_expand);
  ASSERT_INT_EQ(symbolic_expand->op, POLY_OP_EXPAND);
  ASSERT_PTR_EQ(poly_uop_shape_dim(ctx, symbolic_expand, 0), n);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(shape, reduce) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *buf = poly_test_buffer(ctx, POLY_FLOAT32, 100);
  int64_t rdims[] = {10, 10};
  PolyUOp *r = poly_reshape(ctx, buf, rdims, 2);
  int64_t axes[] = {1};
  PolyUOp *red = poly_reduce_axis(ctx, POLY_OP_ADD, r, axes, 1);
  PolyShape s = poly_uop_max_shape(ctx, red);
  ASSERT_INT_EQ(s.ndim, 1);
  ASSERT_INT_EQ(s.dims[0], 10);
  if (s.dims) free(s.dims);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(shape, chain) {
  /* ADD(RESHAPE(buf, (5,4)), EXPAND(RESHAPE(buf2, (1,4)), (5,4))) → (5,4) */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *buf1 = poly_test_buffer(ctx, POLY_FLOAT32, 20);
  PolyUOp *buf2 = poly_test_buffer(ctx, POLY_FLOAT32, 4);
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
  PolyUOp *x = poly_test_buffer(ctx, POLY_FLOAT32, 8);
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
  PolyUOp *a = poly_test_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *b = poly_test_buffer(ctx, POLY_FLOAT32, 20);
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
  PolyUOp *shape = poly_uop(ctx, POLY_OP_STACK, POLY_WEAKINT, shape_src, 2, poly_arg_none());
  PolyParamArg param_arg = {
      .slot = 0,
      .addrspace = POLY_ADDR_GLOBAL,
      .device = "CPU",
  };
  PolyUOp *param = poly_uop1(ctx, POLY_OP_PARAM, POLY_FLOAT32, shape, poly_arg_param(&param_arg));
  PolyUOp *r2 = poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, two, poly_arg_range(0, POLY_AXIS_LOOP));
  PolyUOp *r3 =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, three, poly_arg_range(1, POLY_AXIS_LOOP));
  PolyUOp *r4 =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, four, poly_arg_range(2, POLY_AXIS_LOOP));

  PolyUOp *partial = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, param, r2, poly_arg_none());
  PolyUOp *full_src[] = {param, r2, r3};
  PolyUOp *full = poly_uop(ctx, POLY_OP_INDEX, POLY_FLOAT32, full_src, 3, poly_arg_none());
  PolyUOp *stage_singleton_src[] = {full, zero};
  PolyUOp *stage_singleton = poly_uop(
      ctx, POLY_OP_STAGE, POLY_FLOAT32, stage_singleton_src, 2,
      poly_arg_bufferize_opts("CPU", POLY_ADDR_GLOBAL, true)
  );
  PolyUOp *stage_tensor_src[] = {param, r4};
  PolyUOp *stage_tensor = poly_uop(
      ctx, POLY_OP_STAGE, POLY_FLOAT32, stage_tensor_src, 2,
      poly_arg_bufferize_opts("CPU", POLY_ADDR_GLOBAL, true)
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
  PolyUOp *buf = poly_test_buffer(ctx, POLY_FLOAT32, 1);
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
  PolyUOp *buf = poly_test_buffer(ctx, POLY_FLOAT32, 1);
  ASSERT_NOT_NULL(buf);

  int64_t dims[POLY_MAX_DIMS + 1];
  for (int i = 0; i < POLY_MAX_DIMS + 1; i++) {
    dims[i] = 1;
  }
  PolyArg tuple = int_tuple(dims, POLY_MAX_DIMS + 1);
  PolyUOp *shape_dims[POLY_MAX_DIMS + 1];
  for (int i = 0; i < POLY_MAX_DIMS + 1; i++)
    shape_dims[i] = poly_const_int(ctx, 1);
  PolyUOp *overrank_shape =
      poly_uop(ctx, POLY_OP_STACK, POLY_WEAKINT, shape_dims, POLY_MAX_DIMS + 1, poly_arg_none());
  ASSERT_NOT_NULL(overrank_shape);
  PolyUOp *movement_src[] = {buf, overrank_shape};
  PolyUOp *sized_movement_src[] = {buf, overrank_shape, overrank_shape};

  ASSERT_TRUE(poly_uop1(ctx, POLY_OP_RESHAPE, POLY_FLOAT32, buf, tuple) == NULL);
  ASSERT_TRUE(poly_uop1(ctx, POLY_OP_EXPAND, POLY_FLOAT32, buf, tuple) == NULL);
  ASSERT_TRUE(
      poly_uop(ctx, POLY_OP_RESHAPE, POLY_FLOAT32, movement_src, 2, poly_arg_none()) == NULL
  );
  ASSERT_TRUE(
      poly_uop(ctx, POLY_OP_EXPAND, POLY_FLOAT32, movement_src, 2, poly_arg_none()) == NULL
  );
  ASSERT_TRUE(poly_uop1(ctx, POLY_OP_PERMUTE, POLY_FLOAT32, buf, tuple) == NULL);
  ASSERT_TRUE(poly_uop1(ctx, POLY_OP_FLIP, POLY_FLOAT32, buf, tuple) == NULL);
  ASSERT_TRUE(
      poly_uop(ctx, POLY_OP_SHRINK, POLY_FLOAT32, sized_movement_src, 3, poly_arg_none()) == NULL
  );
  ASSERT_TRUE(
      poly_uop(ctx, POLY_OP_PAD, POLY_FLOAT32, sized_movement_src, 3, poly_arg_none()) == NULL
  );
  ASSERT_TRUE(poly_reduce_axis(ctx, POLY_OP_ADD, buf, dims, POLY_MAX_DIMS + 1) == NULL);

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

  PolyUOp *n = poly_uop_variable(ctx, "N", 1, 4, POLY_INT32, 1, false);
  ASSERT_NOT_NULL(n);
  ASSERT_TRUE(poly_test_buffer_var(ctx, POLY_FLOAT32, n, dims, POLY_MAX_DIMS) == NULL);
  ASSERT_TRUE(poly_test_buffer_var(ctx, POLY_FLOAT32, n, NULL, 1) == NULL);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(shape, movement_construction_canonicalizes_shape_sources) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  /* UOp._mop simplifies shape operands before publishing the movement graph,
   * not just when as_shape later computes metadata. */
  PolyUOp *n = poly_uop_variable(ctx, "n", 1, 8, POLY_WEAKINT, 1, false);
  PolyUOp *twice = poly_uop2(ctx, POLY_OP_ADD, POLY_WEAKINT, n, n, poly_arg_none());
  PolyUOp *canonical =
      poly_uop2(ctx, POLY_OP_MUL, POLY_WEAKINT, n, poly_const_int(ctx, 2), poly_arg_none());
  PolyUOp *base = poly_const_float(ctx, 1.0);
  PolyUOp *expanded = poly_expand_uop(ctx, base, &twice, 1);
  ASSERT_NOT_NULL(expanded);
  ASSERT_INT_EQ(expanded->op, POLY_OP_EXPAND);
  ASSERT_INT_EQ(expanded->n_src, 2);
  ASSERT_PTR_EQ(expanded->src[0], base);
  ASSERT_PTR_EQ(expanded->src[1], canonical);
  PolyUOp *dims[] = {poly_const_int(ctx, 1), twice};
  PolyUOp *reshaped = poly_reshape_uop(ctx, expanded, dims, 2);
  ASSERT_NOT_NULL(reshaped);
  ASSERT_INT_EQ(reshaped->op, POLY_OP_RESHAPE);
  ASSERT_INT_EQ(reshaped->n_src, 2);
  ASSERT_PTR_EQ(reshaped->src[0], expanded);
  ASSERT_PTR_EQ(
      reshaped->src[1], poly_shape_to_shape_arg(ctx, (PolyUOp *[]){dims[0], canonical}, 2)
  );
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(shape, pad_shrink_construction_canonicalizes_both_shape_sources) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *n = poly_uop_variable(ctx, "n", 1, 8, POLY_WEAKINT, 1, false);
  PolyUOp *twice = poly_uop2(ctx, POLY_OP_ADD, POLY_WEAKINT, n, n, poly_arg_none());
  PolyUOp *canonical =
      poly_uop2(ctx, POLY_OP_MUL, POLY_WEAKINT, n, poly_const_int(ctx, 2), poly_arg_none());
  PolyUOp *zero = poly_const_int(ctx, 0);
  PolyUOp *offset = poly_uop2(
      ctx, POLY_OP_ADD, POLY_WEAKINT, n,
      poly_uop2(ctx, POLY_OP_MUL, POLY_WEAKINT, n, poly_const_int(ctx, -1), poly_arg_none()),
      poly_arg_none()
  );
  PolyUOp *base = poly_const_float(ctx, 1.0);
  PolyUOp *wide = poly_expand(ctx, base, (int64_t[]){32}, 1);
  PolyUOp *small = poly_expand(ctx, base, (int64_t[]){1}, 1);
  PolyUOp *roots[] = {
      poly_shrink_uop(ctx, wide, &offset, &twice, 1),
      poly_pad_uop(ctx, small, &offset, &twice, 1),
  };
  for (int i = 0; i < 2; i++) {
    ASSERT_NOT_NULL(roots[i]);
    ASSERT_INT_EQ(roots[i]->op, i == 0 ? POLY_OP_SHRINK : POLY_OP_PAD);
    ASSERT_INT_EQ(roots[i]->n_src, 3);
    ASSERT_PTR_EQ(roots[i]->src[0], i == 0 ? wide : small);
    ASSERT_PTR_EQ(roots[i]->src[1], zero);
    ASSERT_PTR_EQ(roots[i]->src[2], canonical);
  }
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(shape, movement_canonical_shape_identity_returns_source) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *n = poly_uop_variable(ctx, "n", 1, 8, POLY_WEAKINT, 1, false);
  PolyUOp *twice = poly_uop2(ctx, POLY_OP_ADD, POLY_WEAKINT, n, n, poly_arg_none());
  PolyUOp *canonical =
      poly_uop2(ctx, POLY_OP_MUL, POLY_WEAKINT, n, poly_const_int(ctx, 2), poly_arg_none());
  PolyUOp *offset = poly_uop2(
      ctx, POLY_OP_ADD, POLY_WEAKINT, n,
      poly_uop2(ctx, POLY_OP_MUL, POLY_WEAKINT, n, poly_const_int(ctx, -1), poly_arg_none()),
      poly_arg_none()
  );
  PolyUOp *base = poly_expand_uop(ctx, poly_const_float(ctx, 1.0), &canonical, 1);
  ASSERT_PTR_EQ(poly_reshape_uop(ctx, base, &twice, 1), base);
  ASSERT_PTR_EQ(poly_shrink_uop(ctx, base, &offset, &twice, 1), base);
  ASSERT_PTR_EQ(poly_pad_uop(ctx, base, &offset, &twice, 1), base);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(shape, pad_shrink_reject_wrong_rank_before_scalar_noop) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *scalar = poly_const_float(ctx, 1.0);
  PolyUOp *vector = poly_expand(ctx, scalar, (int64_t[]){3}, 1);
  ASSERT_PTR_EQ(poly_pad(ctx, scalar, NULL, 0), scalar);
  ASSERT_PTR_EQ(poly_shrink(ctx, scalar, NULL, 0), scalar);
  ASSERT_PTR_EQ(poly_pad_uop(ctx, scalar, NULL, NULL, 0), scalar);
  ASSERT_PTR_EQ(poly_shrink_uop(ctx, scalar, NULL, NULL, 0), scalar);
  ASSERT_TRUE(poly_pad(ctx, vector, NULL, 0) == NULL);
  ASSERT_TRUE(poly_shrink(ctx, vector, NULL, 0) == NULL);
  ASSERT_TRUE(poly_pad_uop(ctx, vector, NULL, NULL, 0) == NULL);
  ASSERT_TRUE(poly_shrink_uop(ctx, vector, NULL, NULL, 0) == NULL);
  PolyUOp *offsets[] = {poly_const_int(ctx, 0), poly_const_int(ctx, 0)};
  PolyUOp *sizes[] = {poly_const_int(ctx, 1), poly_const_int(ctx, 1)};
  int64_t pairs[2][2] = {{0, 1}, {0, 1}};
  ASSERT_TRUE(poly_pad(ctx, vector, pairs, 2) == NULL);
  ASSERT_TRUE(poly_shrink(ctx, vector, pairs, 2) == NULL);
  ASSERT_TRUE(poly_pad_uop(ctx, vector, offsets, sizes, 2) == NULL);
  ASSERT_TRUE(poly_shrink_uop(ctx, vector, offsets, sizes, 2) == NULL);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(shape, movement_shape_lanes_use_full_pinned_symbolic_canonicalization) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  /* Pinned tinygrad/uop/ops.py:697-705 applies ssimplify() to every as_shape
   * lane before movement shape inference and rangeify suffix comparison. */
  PolyUOp *n = poly_uop_variable(ctx, "n", 1, 8, POLY_WEAKINT, 1, false);
  PolyUOp *y = poly_uop_variable(ctx, "y", 1, 8, POLY_WEAKINT, 1, false);
  PolyUOp *one = poly_const_int(ctx, 1);
  PolyUOp *two = poly_const_int(ctx, 2);
  PolyUOp *three = poly_const_int(ctx, 3);
  PolyUOp *five = poly_const_int(ctx, 5);
  ASSERT_NOT_NULL(n);

  PolyUOp *left[5] = {
      poly_uop2(ctx, POLY_OP_ADD, POLY_WEAKINT, n, n, poly_arg_none()),
      poly_uop2(
          ctx, POLY_OP_ADD, POLY_WEAKINT,
          poly_uop2(ctx, POLY_OP_ADD, POLY_WEAKINT, n, one, poly_arg_none()), two, poly_arg_none()
      ),
      poly_uop2(
          ctx, POLY_OP_MUL, POLY_WEAKINT, two,
          poly_uop2(ctx, POLY_OP_ADD, POLY_WEAKINT, n, one, poly_arg_none()), poly_arg_none()
      ),
      poly_uop2(
          ctx, POLY_OP_ADD, POLY_WEAKINT,
          poly_uop2(ctx, POLY_OP_MUL, POLY_WEAKINT, n, two, poly_arg_none()),
          poly_uop2(ctx, POLY_OP_MUL, POLY_WEAKINT, n, three, poly_arg_none()), poly_arg_none()
      ),
      poly_uop2(
          ctx, POLY_OP_ADD, POLY_WEAKINT,
          poly_uop2(ctx, POLY_OP_ADD, POLY_WEAKINT, y, n, poly_arg_none()), n, poly_arg_none()
      ),
  };
  PolyUOp *right[5] = {
      poly_uop2(ctx, POLY_OP_MUL, POLY_WEAKINT, n, two, poly_arg_none()),
      poly_uop2(ctx, POLY_OP_ADD, POLY_WEAKINT, n, three, poly_arg_none()),
      poly_uop2(
          ctx, POLY_OP_ADD, POLY_WEAKINT,
          poly_uop2(ctx, POLY_OP_MUL, POLY_WEAKINT, n, two, poly_arg_none()), two, poly_arg_none()
      ),
      poly_uop2(ctx, POLY_OP_MUL, POLY_WEAKINT, n, five, poly_arg_none()),
      poly_uop2(
          ctx, POLY_OP_ADD, POLY_WEAKINT, y,
          poly_uop2(ctx, POLY_OP_MUL, POLY_WEAKINT, n, two, poly_arg_none()), poly_arg_none()
      ),
  };

  PolyUOp *base = poly_reshape(ctx, poly_test_buffer(ctx, POLY_FLOAT32, 3), (int64_t[]){3, 1}, 2);
  ASSERT_NOT_NULL(base);
  for (int i = 0; i < 5; i++) {
    PolyUOp *expand_shape_src[] = {three, left[i]};
    PolyUOp *expanded = poly_expand_uop(ctx, base, expand_shape_src, 2);

    PolyUOp *reshape_shape_src[] = {three, right[i]};
    PolyUOp *reshaped = poly_reshape_uop(ctx, expanded, reshape_shape_src, 2);

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

TEST(shape, unequal_width_bitcast_preserves_current_symbolic_last_axis) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  /* Current UOp._shape retains
   * ssimplify((last * input_itemsize) // output_itemsize) rather than only
   * its allocation maximum (uop/ops.py:404-411). */
  PolyUOp *n = poly_uop_variable(ctx, "n", 4, 8, POLY_WEAKINT, 1, false);
  PolyUOp *source = poly_test_buffer_var(ctx, POLY_UINT8, n, NULL, 0);
  PolyUOp *wide = poly_uop1(ctx, POLY_OP_BITCAST, POLY_UINT32, source, poly_arg_none());
  ASSERT_NOT_NULL(source);
  ASSERT_NOT_NULL(wide);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, wide), 1);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, wide)[0], 2);
  PolyUOp *last = poly_uop_shape_dim(ctx, wide, 0);
  ASSERT_NOT_NULL(last);
  ASSERT_INT_EQ(last->op, POLY_OP_FLOORDIV);
  ASSERT_INT_EQ(last->n_src, 2);
  ASSERT_PTR_EQ(last->src[0], n);
  ASSERT_INT_EQ(last->src[1]->op, POLY_OP_CONST);
  ASSERT_INT_EQ(last->src[1]->arg.kind, POLY_ARG_INT);
  ASSERT_INT_EQ(last->src[1]->arg.i, 4);

  PolyUOp *invalid_source =
      poly_reshape(ctx, poly_test_buffer(ctx, POLY_UINT8, 3), (int64_t[]){3}, 1);
  PolyUOp *invalid = poly_uop1(ctx, POLY_OP_BITCAST, POLY_UINT32, invalid_source, poly_arg_none());
  ASSERT_NOT_NULL(invalid);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, invalid), -1);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(shape, scalar_param_shape_uses_full_pinned_symbolic_canonicalization) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  /* Pinned tinygrad/uop/ops.py:697-700 applies ssimplify to any non-STACK
   * scalar shape source too, not only to STACK lanes. */
  PolyUOp *n = poly_uop_variable(ctx, "n", 1, 8, POLY_WEAKINT, 1, false);
  PolyUOp *one = poly_const_int(ctx, 1);
  PolyUOp *two = poly_const_int(ctx, 2);
  PolyUOp *three = poly_const_int(ctx, 3);
  PolyUOp *source = poly_uop2(
      ctx, POLY_OP_ADD, POLY_WEAKINT,
      poly_uop2(ctx, POLY_OP_ADD, POLY_WEAKINT, n, one, poly_arg_none()), two, poly_arg_none()
  );
  PolyUOp *expected = poly_uop2(ctx, POLY_OP_ADD, POLY_WEAKINT, n, three, poly_arg_none());
  PolyParamArg arg = {
      .slot = 0,
      .addrspace = POLY_ADDR_GLOBAL,
      .device = "CPU",
  };
  PolyUOp *param = poly_uop1(ctx, POLY_OP_PARAM, POLY_FLOAT32, source, poly_arg_param(&arg));

  ASSERT_INT_EQ(poly_uop_ndim(ctx, param), 1);
  ASSERT_PTR_EQ(poly_uop_shape_dim(ctx, param, 0), expected);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(shape, multi_axis_propagates_like_pinned_uop_axis) {
  /* Pinned UOp.axis rules are source-backed by uop/ops.py:623-651 and the
   * paired temp/tg_multi_axis_propagation_probe_20260811.py probe. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  const char *names[] = {"CPU", "CPU:1"};
  PolyUOp *device_tuple = poly_device_uop_from_names(ctx, names, 2);
  PolyUOp *buffer = poly_uop_new_buffer(ctx, device_tuple, 4, POLY_FLOAT32, 511);
  PolyUOp *local = poly_reshape(ctx, buffer, (int64_t[]){2, 2}, 2);
  PolyUOp *device_range = poly_range(ctx, 2, -1, POLY_AXIS_DEVICE);
  int64_t axis0 = 0;
  PolyUOp *ranges[] = {device_range};
  PolyUOp *multi = poly_unshard(ctx, local, &axis0, ranges, 1);
  ASSERT_NOT_NULL(multi);

  int axis = -1;
  ASSERT_TRUE(poly_uop_axis(ctx, multi, &axis));
  ASSERT_INT_EQ(axis, 0);

  PolyUOp *contiguous = poly_contiguous(ctx, multi);
  ASSERT_TRUE(poly_uop_axis(ctx, contiguous, &axis));
  ASSERT_INT_EQ(axis, 0);

  PolyUOp *cpu = poly_device_uop_from_name(ctx, "CPU");
  PolyUOp *copy = poly_copy_to_device_uop(ctx, multi, cpu);
  ASSERT_FALSE(poly_uop_axis(ctx, copy, &axis));

  PolyUOp *full = poly_shrink(ctx, multi, (int64_t[][2]){{0, 4}, {0, 2}}, 2);
  PolyUOp *partial = poly_shrink(ctx, multi, (int64_t[][2]){{0, 2}, {0, 2}}, 2);
  ASSERT_TRUE(poly_uop_axis(ctx, full, &axis));
  ASSERT_INT_EQ(axis, 0);
  ASSERT_FALSE(poly_uop_axis(ctx, partial, &axis));

  PolyUOp *permuted = poly_permute(ctx, multi, (int64_t[]){1, 0}, 2);
  ASSERT_TRUE(poly_uop_axis(ctx, permuted, &axis));
  ASSERT_INT_EQ(axis, 1);

  PolyUOp *reshaped = poly_reshape(ctx, multi, (int64_t[]){2, 4}, 2);
  ASSERT_TRUE(poly_uop_axis(ctx, reshaped, &axis));
  ASSERT_INT_EQ(axis, 0);

  int64_t other_axis[] = {1};
  int64_t shard_axis[] = {0};
  PolyUOp *reduce_other = poly_reduce_axis(ctx, POLY_OP_ADD, multi, other_axis, 1);
  PolyUOp *reduce_shard = poly_reduce_axis(ctx, POLY_OP_ADD, multi, shard_axis, 1);
  ASSERT_TRUE(poly_uop_axis(ctx, reduce_other, &axis));
  ASSERT_INT_EQ(axis, 0);
  ASSERT_FALSE(poly_uop_axis(ctx, reduce_shard, &axis));

  poly_ctx_destroy(ctx);
  PASS();
}
