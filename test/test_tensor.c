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
#include "../src/bigint.h"
#include "../src/polygrad.h"
#include "../src/frontend.h"
#include "../src/device.h"
#include "../src/engine/schedule.h"
#include "../src/schedule/rangeify.h"
#include "../src/nn.h"
#include "../src/tensor.h"
#include "../src/codegen.h"

/* Helper: realize a UOp into a host array */

static int realize_uop(
    PolyCtx *ctx,
    PolyUOp *val,
    PolyUOp *out_buf,
    void *out_data,
    PolyUOp **leaf_bufs,
    float **leaf_datas,
    int n_leaves
) {
  PolyUOp *store = poly_store_val(ctx, out_buf, val);
  PolyUOp *sink = poly_sink1(ctx, store);
  int n = n_leaves + 1;
  PolyUOp *bufs[64];
  void *datas[64];
  PolyTestBufferView views[64];
  for (int i = 0; i < n_leaves; i++) {
    bufs[i] = leaf_bufs[i];
    datas[i] = leaf_datas[i];
  }
  bufs[n_leaves] = out_buf;
  datas[n_leaves] = out_data;
  for (int i = 0; i < n; i++)
    views[i] = POLY_TEST_HOST_VIEW(bufs[i], datas[i]);
  return poly_test_realize_buffer_views(ctx, sink, views, n);
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

static int count_op_in_root(PolyCtx *ctx, PolyUOp *root, int op) {
  int n_topo = 0, count = 0;
  PolyUOp **topo = poly_toposort_alloc(ctx, root, &n_topo);
  if (!topo) return -1;
  for (int i = 0; i < n_topo; i++) count += topo[i]->op == op;
  poly_toposort_free(topo);
  return count;
}

static int read_tensor_bytes(PolyCtx *ctx, PolyTensor *tensor, void *out, size_t nbytes) {
  PolyTensor *realized = NULL;
  if (!ctx || !tensor || !out || poly_realize_tensors(ctx, &tensor, 1, &realized) != 0 ||
      !realized)
    return -1;
  const PolyUOp *buffer = poly_uop_get_buffer_identity(realized->uop_physical);
  return buffer ? poly_buffer_read(ctx, (PolyUOp *)buffer, out, nbytes) : -1;
}

static int read_tensor_f32(PolyCtx *ctx, PolyTensor *tensor, float *out, size_t n) {
  return read_tensor_bytes(ctx, tensor, out, n * sizeof(*out));
}

static void set_unsigned_values(void *dst, PolyDType dtype, const uint64_t *values, int n) {
  for (int i = 0; i < n; i++) {
    if (dtype.bitsize == 8)
      ((uint8_t *)dst)[i] = (uint8_t)values[i];
    else if (dtype.bitsize == 16)
      ((uint16_t *)dst)[i] = (uint16_t)values[i];
    else if (dtype.bitsize == 32)
      ((uint32_t *)dst)[i] = (uint32_t)values[i];
    else
      ((uint64_t *)dst)[i] = values[i];
  }
}

static uint64_t get_unsigned_value(const void *src, PolyDType dtype, int i) {
  if (dtype.bitsize == 8) return ((const uint8_t *)src)[i];
  if (dtype.bitsize == 16) return ((const uint16_t *)src)[i];
  if (dtype.bitsize == 32) return ((const uint32_t *)src)[i];
  return ((const uint64_t *)src)[i];
}

/* Structural constructor tests that count LOAD/RANGE/STORE ops need the
 * executable scheduled root rather than the earlier public kernel graph. */
static PolyUOp *single_scheduled_root(PolyCtx *ctx, PolyUOp *sink) {
  PolySchedule *schedule = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  if (!schedule) return NULL;
  if (schedule->template->n_calls != 1 || !poly_schedule_call_body(schedule, 0)) {
    poly_schedule_free(schedule);
    return NULL;
  }
  PolyUOp *root = poly_schedule_call_body(schedule, 0);
  poly_schedule_free(schedule);
  return root;
}

/* ═══════════════════════════════════════════════════════════════════════ */
/*  v2 composed op e2e tests (tinygrad-verified reference values)         */
/* ═══════════════════════════════════════════════════════════════════════ */

TEST(tensor, function_builds_ordered_logical_and_physical_value_calls) {
  /* Pinned function.py:39-94 and uop/ops.py:1077-1092 substitute ordered
   * inputs with PARAMs and expose one TUPLE/FUNCTION/GETTUPLE value call. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  float av[2] = {1.0f, 2.0f}, bv[2] = {3.0f, 4.0f};
  int64_t shape[1] = {2};
  PolyTensor *a_host = poly_tensor_from_host(
      ctx, av, sizeof(av), POLY_FLOAT32, shape, 1);
  PolyTensor *b_host = poly_tensor_from_host(
      ctx, bv, sizeof(bv), POLY_FLOAT32, shape, 1);
  PolyTensor *a = poly_tensor_to_device(ctx, a_host, POLY_DEVICE_CPU);
  PolyTensor *b = poly_tensor_to_device(ctx, b_host, POLY_DEVICE_CPU);
  PolyTensor *sum = poly_tensor_alu2(ctx, POLY_OP_ADD, a, b);
  ASSERT_NOT_NULL(sum);
  PolyTensor *results[1] = {sum};
  PolyTensor *inputs[2] = {a, b};
  PolyTensor *outputs[1] = {NULL};
  ASSERT_INT_EQ(
      poly_tensor_function(
          ctx, results, 1, inputs, 2, "ordered_add", false, false, false, outputs),
      0);
  ASSERT_NOT_NULL(outputs[0]);

  PolyUOp *surfaces[2] = {
      outputs[0]->uop_logical, outputs[0]->uop_physical};
  for (int surface = 0; surface < 2; surface++) {
    PolyUOp *selected = surfaces[surface];
    ASSERT_NOT_NULL(selected);
    ASSERT_INT_EQ(selected->op, POLY_OP_GETTUPLE);
    ASSERT_INT_EQ(selected->arg.kind, POLY_ARG_INT);
    ASSERT_INT_EQ(selected->arg.i, 0);
    PolyUOp *function = selected->src[0];
    ASSERT_INT_EQ(function->op, POLY_OP_FUNCTION);
    ASSERT_INT_EQ(function->n_src, 3);
    ASSERT_INT_EQ(function->arg.kind, POLY_ARG_CALL_INFO);
    ASSERT_STR_EQ(function->arg.call_info->name, "ordered_add");
    ASSERT_INT_EQ(function->src[0]->op, POLY_OP_TUPLE);
    ASSERT_INT_EQ(function->src[0]->n_src, 1);
    PolyUOp *add = function->src[0]->src[0];
    ASSERT_INT_EQ(add->op, POLY_OP_ADD);
    ASSERT_INT_EQ(add->src[0]->op, POLY_OP_PARAM);
    ASSERT_INT_EQ(add->src[1]->op, POLY_OP_PARAM);
    ASSERT_INT_EQ(add->src[0]->arg.param->slot, 0);
    ASSERT_INT_EQ(add->src[1]->arg.param->slot, 1);
  }

  PolyUOp *grad = poly_grad(
      ctx, outputs[0]->uop_physical, a->uop_physical);
  ASSERT_NOT_NULL(grad);
  int n_topo = 0;
  PolyUOp **topo = poly_toposort_alloc(ctx, grad, &n_topo);
  ASSERT_NOT_NULL(topo);
  int functions = 0, gettuples = 0;
  for (int i = 0; i < n_topo; i++) {
    functions += topo[i]->op == POLY_OP_FUNCTION;
    gettuples += topo[i]->op == POLY_OP_GETTUPLE;
  }
  /* For ADD the derivative is constant, so pinned gradient.py eliminates the
   * forward call and retains only the backward FUNCTION/GETTUPLE pair. */
  ASSERT_INT_EQ(functions, 1);
  ASSERT_INT_EQ(gettuples, 1);
  poly_toposort_free(topo);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(tensor, static_empty_shares_unique_with_deviceful_physical_root) {
  PolyCtx *ctx = poly_ctx_new();
  int64_t shape[2] = {2, 3};
  ASSERT_EQ(poly_tensor_empty(ctx, POLY_FLOAT32, shape, 2, POLY_DEVICE_AUTO), NULL);
  ASSERT_EQ(poly_tensor_empty(ctx, POLY_FLOAT32, shape, 2, POLY_DEVICE_HOST), NULL);
  ASSERT_EQ(poly_tensor_empty(ctx, POLY_FLOAT32, shape, 2, POLY_DEVICE_DISK), NULL);
  PolyTensor *tensor = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 2, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(tensor);
  ASSERT_NOT_NULL(tensor->uop_logical);
  ASSERT_NOT_NULL(tensor->uop_physical);
  ASSERT_INT_EQ(tensor->uop_logical->op, POLY_OP_RESHAPE);
  ASSERT_INT_EQ(tensor->uop_physical->op, POLY_OP_RESHAPE);

  PolyUOp *logical_buffer = base_buf(tensor->uop_logical);
  PolyUOp *physical_buffer = base_buf(tensor->uop_physical);
  ASSERT_NOT_NULL(logical_buffer);
  ASSERT_NOT_NULL(physical_buffer);
  ASSERT_INT_EQ(logical_buffer->op, POLY_OP_BUFFER);
  ASSERT_INT_EQ(physical_buffer->op, POLY_OP_BUFFER);
  ASSERT_INT_EQ(logical_buffer->n_src, 1);
  ASSERT_INT_EQ(physical_buffer->n_src, 2);
  ASSERT_TRUE(logical_buffer->src[0] == physical_buffer->src[0]);
  ASSERT_INT_EQ(physical_buffer->src[1]->op, POLY_OP_DEVICE);
  ASSERT_INT_EQ(poly_device_from_device_uop(physical_buffer->src[1]), POLY_DEVICE_CPU);
  ASSERT_EQ(physical_buffer->src[1]->arg.kind, POLY_ARG_STRING);
  ASSERT_STR_EQ(physical_buffer->src[1]->arg.str, "CPU");

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(tensor, root_mutators_require_complete_physical_root) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *logical = poly_buffer_f32(ctx, 2);
  PolyUOp *physical = poly_buffer_on_device(ctx, POLY_FLOAT32, 2, POLY_DEVICE_CPU);
  PolyTensor *tensor = poly_tensor_create_with_roots(
      ctx, logical, physical, POLY_TENSOR_VALUE, POLY_DEVICE_CPU
  );
  ASSERT_NOT_NULL(tensor);

  PolyUOp *realized = poly_buffer_on_device(ctx, POLY_FLOAT32, 2, POLY_DEVICE_CPU);
  ASSERT_INT_EQ(
      poly_tensor_set_physical(
          ctx, tensor, realized, POLY_TENSOR_VALUE, POLY_DEVICE_CPU
      ),
      0
  );
  ASSERT_PTR_EQ(tensor->uop_logical, logical);
  ASSERT_PTR_EQ(tensor->uop_physical, realized);

  PolyUOp *next_logical = poly_add(ctx, logical, poly_const_float(ctx, 1.0));
  PolyUOp *next_physical = poly_add(ctx, realized, poly_const_float(ctx, 1.0));
  ASSERT_NOT_NULL(next_logical);
  ASSERT_NOT_NULL(next_physical);
  ASSERT_INT_EQ(
      poly_tensor_replace_roots(
          ctx, tensor, next_logical, NULL, POLY_TENSOR_VALUE, POLY_DEVICE_CPU
      ),
      -1
  );
  ASSERT_PTR_EQ(tensor->uop_logical, logical);
  ASSERT_PTR_EQ(tensor->uop_physical, realized);
  ASSERT_INT_EQ(
      poly_tensor_replace_roots(
          ctx, tensor, next_logical, next_physical, POLY_TENSOR_VALUE, POLY_DEVICE_CPU
      ),
      0
  );
  ASSERT_PTR_EQ(tensor->uop_logical, next_logical);
  ASSERT_PTR_EQ(tensor->uop_physical, next_physical);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(tensor, host_array_uses_deviceful_source_and_real_cpu_copy) {
  PolyCtx *ctx = poly_ctx_new();
  float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  int64_t shape[2] = {2, 2};
  PolyTensor *source =
      poly_tensor_from_host(ctx, data, sizeof(data), POLY_FLOAT32, shape, 2);
  ASSERT_NOT_NULL(source);
  ASSERT_INT_EQ(source->device, POLY_DEVICE_HOST);
  ASSERT_INT_EQ(source->uop_logical->op, POLY_OP_RESHAPE);
  ASSERT_INT_EQ(source->uop_physical->op, POLY_OP_RESHAPE);

  PolyUOp *logical_buffer = base_buf(source->uop_logical);
  PolyUOp *physical_buffer = base_buf(source->uop_physical);
  ASSERT_NOT_NULL(logical_buffer);
  ASSERT_NOT_NULL(physical_buffer);
  ASSERT_INT_EQ(logical_buffer->n_src, 1);
  ASSERT_INT_EQ(physical_buffer->n_src, 2);
  ASSERT_PTR_EQ(logical_buffer->src[0], physical_buffer->src[0]);
  ASSERT_INT_EQ(
      poly_device_from_device_uop(physical_buffer->src[1]), POLY_DEVICE_HOST
  );
  ASSERT_EQ(poly_buffer_get_ptr(ctx, logical_buffer), NULL);
  ASSERT_PTR_EQ(poly_buffer_get_ptr(ctx, physical_buffer), data);

  PolyTensor *moved = poly_tensor_to_device(ctx, source, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(moved);
  ASSERT_INT_EQ(moved->uop_physical->op, POLY_OP_COPY);
  ASSERT_PTR_EQ(moved->uop_physical->src[0], source->uop_physical);
  ASSERT_INT_EQ(
      poly_device_from_device_uop(moved->uop_physical->src[1]), POLY_DEVICE_CPU
  );

  PolyTensor *realized = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &moved, 1, &realized), 0);
  ASSERT_NOT_NULL(realized);
  PolyUOp *realized_buffer =
      (PolyUOp *)poly_uop_get_buffer_identity(realized->uop_physical);
  ASSERT_NOT_NULL(realized_buffer);
  ASSERT_PTR_NEQ(realized_buffer, physical_buffer);
  float out[4] = {0};
  ASSERT_INT_EQ(poly_buffer_read(ctx, realized_buffer, out, sizeof(out)), 0);
  for (int i = 0; i < 4; i++) ASSERT_FLOAT_NEAR(out[i], data[i], 4, 1e-6);

  ASSERT_EQ(
      poly_tensor_from_host(ctx, data, sizeof(data) - 1, POLY_FLOAT32, shape, 2),
      NULL
  );
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(tensor, stateful_rand_matches_pinned_graph_and_values) {
  /* Pinned Tensor._next_counter plus RandMixin._rand
   * (tensor.py:493-504, mixin/rand.py:12-39): one public draw advances the
   * storage-backed uint32 counter through AFTER/STORE and emits two THREEFRY
   * nodes. The exact float32 values below are the pinned seed-1337 CPU words. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  int f32_id = poly_dtype_id_by_name("float32");
  int64_t shape[1] = {8};
  poly_tensor_manual_seed(ctx, 1337);
  PolyTensor *first = poly_tensor_rand_by_id(
      ctx, shape, 1, f32_id, POLY_DEVICE_CPU, 1
  );
  ASSERT_NOT_NULL(first);
  ASSERT_INT_EQ(count_op_in_root(ctx, first->uop_physical, POLY_OP_AFTER), 1);
  ASSERT_INT_EQ(count_op_in_root(ctx, first->uop_physical, POLY_OP_STORE), 1);
  ASSERT_INT_EQ(count_op_in_root(ctx, first->uop_physical, POLY_OP_THREEFRY), 2);
  ASSERT_INT_EQ(count_op_in_root(ctx, first->uop_physical, POLY_OP_COPY), 2);
  ASSERT_INT_EQ(count_op_in_root(ctx, first->uop_logical, POLY_OP_COPY), 0);

  float first_values[8] = {0};
  ASSERT_INT_EQ(read_tensor_f32(ctx, first, first_values, 8), 0);
  const uint32_t expected_bits[8] = {
      UINT32_C(0x3efa31a0), UINT32_C(0x3eb22b7c), UINT32_C(0x3f28c97e),
      UINT32_C(0x3f22effe), UINT32_C(0x3ef13c94), UINT32_C(0x3e10dd30),
      UINT32_C(0x3e8e61ec), UINT32_C(0x3d4c9dc0),
  };
  ASSERT_TRUE(memcmp(first_values, expected_bits, sizeof(expected_bits)) == 0);

  poly_tensor_manual_seed(ctx, 1337);
  PolyTensor *reset = poly_tensor_rand_by_id(
      ctx, shape, 1, f32_id, POLY_DEVICE_CPU, 1
  );
  ASSERT_NOT_NULL(reset);
  float reset_values[8] = {0};
  ASSERT_INT_EQ(read_tensor_f32(ctx, reset, reset_values, 8), 0);
  ASSERT_TRUE(memcmp(first_values, reset_values, sizeof(first_values)) == 0);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(tensor, stateful_rand_is_context_local_and_advances) {
  /* Tensor's pinned dictionaries are process-global only because Tensor owns
   * the runtime. Polygrad adapts that state to PolyCtx ownership: equal fresh
   * contexts reproduce the stream, while consecutive draws in one context
   * advance it. */
  PolyCtx *ctx0 = poly_ctx_new();
  PolyCtx *ctx1 = poly_ctx_new();
  ASSERT_NOT_NULL(ctx0);
  ASSERT_NOT_NULL(ctx1);
  int f32_id = poly_dtype_id_by_name("float32");
  int64_t shape[1] = {4};
  poly_tensor_manual_seed(ctx0, 123);
  poly_tensor_manual_seed(ctx1, 123);
  PolyTensor *a0 = poly_tensor_rand_by_id(ctx0, shape, 1, f32_id, POLY_DEVICE_CPU, 1);
  PolyTensor *a1 = poly_tensor_rand_by_id(ctx1, shape, 1, f32_id, POLY_DEVICE_CPU, 1);
  PolyTensor *b0 = poly_tensor_rand_by_id(ctx0, shape, 1, f32_id, POLY_DEVICE_CPU, 1);
  ASSERT_NOT_NULL(a0);
  ASSERT_NOT_NULL(a1);
  ASSERT_NOT_NULL(b0);
  float av0[4] = {0}, av1[4] = {0}, bv0[4] = {0};
  ASSERT_INT_EQ(read_tensor_f32(ctx0, a0, av0, 4), 0);
  ASSERT_INT_EQ(read_tensor_f32(ctx1, a1, av1, 4), 0);
  ASSERT_INT_EQ(read_tensor_f32(ctx0, b0, bv0, 4), 0);
  ASSERT_TRUE(memcmp(av0, av1, sizeof(av0)) == 0);
  ASSERT_TRUE(memcmp(av0, bv0, sizeof(av0)) != 0);
  poly_ctx_destroy(ctx1);
  poly_ctx_destroy(ctx0);
  PASS();
}

TEST(tensor, scalar_constructors_store_exact_const_as_both_roots) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  int int_id = poly_dtype_id_by_name("int32");
  int float_id = poly_dtype_id_by_name("float32");
  ASSERT_TRUE(int_id >= 0);
  ASSERT_TRUE(float_id >= 0);

  PolyTensor *i =
      poly_tensor_const_int_by_id(ctx, 7, int_id, POLY_DEVICE_CUDA);
  PolyTensor *f =
      poly_tensor_const_float_by_id(ctx, 1.5, float_id, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(i);
  ASSERT_NOT_NULL(f);
  ASSERT_PTR_EQ(i->uop_logical, i->uop_physical);
  ASSERT_PTR_EQ(f->uop_logical, f->uop_physical);
  ASSERT_EQ(i->uop_physical->op, POLY_OP_CONST);
  ASSERT_EQ(f->uop_physical->op, POLY_OP_CONST);
  ASSERT_INT_EQ(i->uop_physical->arg.i, 7);
  ASSERT_FLOAT_NEAR(f->uop_physical->arg.f, 1.5, 4, 0.0);
  ASSERT_INT_EQ(i->device, POLY_DEVICE_CUDA);
  ASSERT_INT_EQ(f->device, POLY_DEVICE_CPU);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(tensor, pure_constructors_store_one_device_free_root) {
  /* Pinned Tensor.__init__ stores a device-free constructor UOp directly, and
   * Tensor.to returns self while UOp.device is None (tensor.py:76-119,
   * 327-335). The Tensor boundary stores that exact root as both twins; no placement map is
   * involved. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  int i32 = poly_dtype_id_by_name("int32");
  int f32 = poly_dtype_id_by_name("float32");
  int64_t shape[] = {2, 3};
  PolyTensor *values[] = {
      poly_tensor_full_int_by_id(ctx, shape, 2, 2, i32, POLY_DEVICE_CPU),
      poly_tensor_full_float_by_id(ctx, shape, 2, 2.0, f32, POLY_DEVICE_CPU),
      poly_tensor_arange_int_by_id(ctx, 0, 4, 1, i32, POLY_DEVICE_CPU),
      poly_tensor_arange_float_by_id(ctx, 0.0, 4.0, 1.0, f32, POLY_DEVICE_CPU),
      poly_tensor_linspace_by_id(ctx, 0.0, 1.0, 4, f32, POLY_DEVICE_CPU),
      poly_tensor_eye_by_id(ctx, 3, 3, f32, POLY_DEVICE_CPU),
  };

  for (int i = 0; i < (int)(sizeof(values) / sizeof(values[0])); i++) {
    ASSERT_NOT_NULL(values[i]);
    ASSERT_PTR_EQ(values[i]->uop_logical, values[i]->uop_physical);
    ASSERT_INT_EQ(poly_uop_device(values[i]->uop_physical), POLY_DEVICE_AUTO);
  }

  int64_t singleton_shape[] = {1};
  PolyTensor *singleton_full =
      poly_tensor_full_int_by_id(ctx, singleton_shape, 1, 2, i32, POLY_DEVICE_CPU);
  PolyTensor *singleton_arange =
      poly_tensor_arange_int_by_id(ctx, 0, 1, 1, i32, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(singleton_full);
  ASSERT_NOT_NULL(singleton_arange);
  ASSERT_EQ(singleton_full->uop_physical->op, POLY_OP_RESHAPE);
  ASSERT_EQ(singleton_full->uop_physical->src[0]->op, POLY_OP_CONST);

  PolyTensor *moved = poly_tensor_to_device(ctx, values[2], POLY_DEVICE_CUDA);
  ASSERT_PTR_EQ(moved, values[2]);
  ASSERT_PTR_EQ(moved->uop_physical, values[2]->uop_physical);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(tensor, movement_constructors_use_exact_logical_and_physical_sources) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  int64_t source_shape[2] = {1, 3};
  PolyTensor *source = poly_tensor_empty(ctx, POLY_FLOAT32, source_shape, 2, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(source);
  ASSERT_PTR_NEQ(source->uop_logical, source->uop_physical);

  int64_t reshape_dims[2] = {3, 1};
  int64_t expand_dims[2] = {2, 3};
  int64_t perm[2] = {1, 0};
  int64_t shrink_pairs[2][2] = {{0, 1}, {1, 3}};
  int64_t pad_pairs[2][2] = {{1, 0}, {0, 1}};
  int64_t axes[1] = {1};

  PolyTensor *reshape = poly_tensor_reshape(ctx, source, reshape_dims, 2);
  PolyTensor *expand = poly_tensor_expand(ctx, source, expand_dims, 2);
  PolyTensor *permute = poly_tensor_permute(ctx, source, perm, 2);
  PolyTensor *shrink = poly_tensor_shrink(ctx, source, shrink_pairs, 2);
  PolyTensor *pad = poly_tensor_pad_value(ctx, source, pad_pairs, 2, 0.0);
  PolyTensor *flip = poly_tensor_flip(ctx, source, axes, 1);
  PolyTensor *results[6] = {reshape, expand, permute, shrink, pad, flip};
  PolyOps ops[6] = {
      POLY_OP_RESHAPE, POLY_OP_EXPAND, POLY_OP_PERMUTE, POLY_OP_SHRINK, POLY_OP_PAD, POLY_OP_FLIP,
  };

  for (int i = 0; i < 6; i++) {
    ASSERT_NOT_NULL(results[i]);
    ASSERT_INT_EQ(results[i]->uop_logical->op, ops[i]);
    ASSERT_INT_EQ(results[i]->uop_physical->op, ops[i]);
    ASSERT_PTR_EQ(results[i]->uop_logical->src[0], source->uop_logical);
    ASSERT_PTR_EQ(results[i]->uop_physical->src[0], source->uop_physical);
    ASSERT_PTR_NEQ(results[i]->uop_logical, results[i]->uop_physical);
  }

  /* Raw logical-only construction is inert import/re-placement input. Default
   * Tensor composition must reject it instead of reconstructing execution
   * state from the logical graph. */
  PolyUOp *legacy_uop = poly_buffer_f32(ctx, 3);
  PolyTensor *legacy = poly_tensor_create_with_roots(
      ctx, legacy_uop, NULL, POLY_TENSOR_VALUE, POLY_DEVICE_CPU
  );
  int64_t legacy_shape[2] = {3, 1};
  PolyTensor *legacy_reshape = poly_tensor_reshape(ctx, legacy, legacy_shape, 2);
  ASSERT_NOT_NULL(legacy);
  ASSERT_EQ(legacy->uop_physical, NULL);
  ASSERT_EQ(legacy_reshape, NULL);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(tensor, symbolic_shrink_uses_exact_logical_and_physical_sources) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *logical = make_buf(ctx, (int64_t[]){10, 2}, 2);
  PolyUOp *physical_buffer =
      poly_buffer_on_device(ctx, POLY_FLOAT32, 20, POLY_DEVICE_CPU);
  PolyUOp *physical =
      poly_reshape(ctx, physical_buffer, (int64_t[]){10, 2}, 2);
  PolyTensor *source = poly_tensor_create_with_roots(
      ctx, logical, physical, POLY_TENSOR_VALUE, POLY_DEVICE_CPU
  );
  ASSERT_NOT_NULL(source);
  ASSERT_PTR_NEQ(source->uop_logical, source->uop_physical);

  float data[20];
  for (int i = 0; i < 20; i++) data[i] = (float)i;

  PolyUOp *i = poly_define_var(ctx, "i", 0, 8);
  PolyUOp *ib = poly_bind_var(ctx, i, 4);
  PolyUOp *starts[2] = {ib, poly_const_int(ctx, 0)};
  PolyUOp *sizes[2] = {poly_const_int(ctx, 2), poly_const_int(ctx, 2)};
  PolyUOp *expected_logical =
      poly_shrink_uop(ctx, source->uop_logical, starts, sizes, 2);
  PolyUOp *expected_physical =
      poly_shrink_uop(ctx, source->uop_physical, starts, sizes, 2);

  PolyTensor *slice =
      poly_tensor_shrink_uop(ctx, source, starts, sizes, 2);
  ASSERT_NOT_NULL(slice);
  ASSERT_PTR_EQ(slice->uop_logical, expected_logical);
  ASSERT_PTR_EQ(slice->uop_physical, expected_physical);
  ASSERT_INT_EQ(slice->uop_logical->op, POLY_OP_SHRINK);
  ASSERT_INT_EQ(slice->uop_physical->op, POLY_OP_SHRINK);
  ASSERT_PTR_EQ(slice->uop_logical->src[0], source->uop_logical);
  ASSERT_PTR_EQ(slice->uop_physical->src[0], source->uop_physical);

  float out[4] = {0};
  PolyUOp *leaf_buffers[1] = {physical_buffer};
  float *leaf_data[1] = {data};
  ASSERT_INT_EQ(
      realize_uop(
          ctx, slice->uop_physical, poly_buffer_f32(ctx, 4), out,
          leaf_buffers, leaf_data, 1
      ),
      0
  );
  ASSERT_FLOAT_EQ(out[0], 8.0f, 0.0);
  ASSERT_FLOAT_EQ(out[1], 9.0f, 0.0);
  ASSERT_FLOAT_EQ(out[2], 10.0f, 0.0);
  ASSERT_FLOAT_EQ(out[3], 11.0f, 0.0);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(tensor, symbolic_expand_uses_exact_logical_and_physical_sources) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *logical = make_buf(ctx, (int64_t[]){1, 2}, 2);
  PolyUOp *physical_buffer =
      poly_buffer_on_device(ctx, POLY_FLOAT32, 2, POLY_DEVICE_CPU);
  PolyUOp *physical =
      poly_reshape(ctx, physical_buffer, (int64_t[]){1, 2}, 2);
  PolyTensor *source = poly_tensor_create_with_roots(
      ctx, logical, physical, POLY_TENSOR_VALUE, POLY_DEVICE_CPU
  );
  ASSERT_NOT_NULL(source);
  ASSERT_PTR_NEQ(source->uop_logical, source->uop_physical);

  PolyUOp *n = poly_define_var(ctx, "n", 1, 4);
  PolyUOp *nb = poly_bind_var(ctx, n, 3);
  PolyUOp *dims[2] = {nb, poly_const_int(ctx, 2)};
  PolyUOp *expected_logical =
      poly_expand_uop(ctx, source->uop_logical, dims, 2);
  PolyUOp *expected_physical =
      poly_expand_uop(ctx, source->uop_physical, dims, 2);

  PolyTensor *expanded =
      poly_tensor_expand_uop(ctx, source, dims, 2);
  ASSERT_NOT_NULL(expanded);
  ASSERT_PTR_EQ(expanded->uop_logical, expected_logical);
  ASSERT_PTR_EQ(expanded->uop_physical, expected_physical);
  ASSERT_INT_EQ(expanded->uop_logical->op, POLY_OP_EXPAND);
  ASSERT_INT_EQ(expanded->uop_physical->op, POLY_OP_EXPAND);
  ASSERT_PTR_EQ(expanded->uop_logical->src[0], source->uop_logical);
  ASSERT_PTR_EQ(expanded->uop_physical->src[0], source->uop_physical);

  PolyUOp *starts[2] = {poly_const_int(ctx, 0), poly_const_int(ctx, 0)};
  PolyUOp *sizes[2] = {poly_const_int(ctx, 3), poly_const_int(ctx, 2)};
  PolyUOp *static_view =
      poly_shrink_uop(ctx, expanded->uop_physical, starts, sizes, 2);
  ASSERT_NOT_NULL(static_view);
  float data[2] = {1.0f, 2.0f};
  float out[6] = {0};
  PolyUOp *leaf_buffers[1] = {physical_buffer};
  float *leaf_data[1] = {data};
  ASSERT_INT_EQ(
      realize_uop(
          ctx, static_view, poly_buffer_f32(ctx, 6), out,
          leaf_buffers, leaf_data, 1
      ),
      0
  );
  for (int row = 0; row < 3; row++) {
    ASSERT_FLOAT_EQ(out[row * 2], 1.0f, 0.0);
    ASSERT_FLOAT_EQ(out[row * 2 + 1], 2.0f, 0.0);
  }

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(tensor, symbolic_reshape_uses_exact_logical_and_physical_sources) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *logical = make_buf(ctx, (int64_t[]){1, 8, 1, 8}, 4);
  PolyUOp *physical_buffer =
      poly_buffer_on_device(ctx, POLY_FLOAT32, 64, POLY_DEVICE_CPU);
  PolyUOp *physical =
      poly_reshape(ctx, physical_buffer, (int64_t[]){1, 8, 1, 8}, 4);
  PolyTensor *source = poly_tensor_create_with_roots(
      ctx, logical, physical, POLY_TENSOR_VALUE, POLY_DEVICE_CPU
  );
  ASSERT_NOT_NULL(source);

  PolyUOp *n = poly_define_var(ctx, "n", 1, 7);
  PolyUOp *nb = poly_bind_var(ctx, n, 3);
  PolyUOp *starts[4] = {
      poly_const_int(ctx, 0), poly_const_int(ctx, 0),
      poly_const_int(ctx, 0), poly_const_int(ctx, 0),
  };
  PolyUOp *sizes[4] = {
      poly_const_int(ctx, 1), nb,
      poly_const_int(ctx, 1), poly_const_int(ctx, 8),
  };
  PolyTensor *slice = poly_tensor_shrink_uop(ctx, source, starts, sizes, 4);
  ASSERT_NOT_NULL(slice);

  PolyUOp *dims[4] = {
      poly_const_int(ctx, 1), nb,
      poly_const_int(ctx, 2), poly_const_int(ctx, 4),
  };
  PolyUOp *expected_logical =
      poly_reshape_uop(ctx, slice->uop_logical, dims, 4);
  PolyUOp *expected_physical =
      poly_reshape_uop(ctx, slice->uop_physical, dims, 4);
  PolyTensor *reshaped = poly_tensor_reshape_uop(ctx, slice, dims, 4);
  ASSERT_NOT_NULL(reshaped);
  ASSERT_PTR_EQ(reshaped->uop_logical, expected_logical);
  ASSERT_PTR_EQ(reshaped->uop_physical, expected_physical);
  ASSERT_INT_EQ(reshaped->uop_logical->op, POLY_OP_RESHAPE);
  ASSERT_INT_EQ(reshaped->uop_physical->op, POLY_OP_RESHAPE);
  ASSERT_PTR_EQ(reshaped->uop_logical->src[0], slice->uop_logical);
  ASSERT_PTR_EQ(reshaped->uop_physical->src[0], slice->uop_physical);
  ASSERT_INT_EQ(reshaped->uop_physical->src[1]->op, POLY_OP_STACK);
  ASSERT_INT_EQ(reshaped->uop_physical->src[1]->n_src, 4);
  ASSERT_PTR_EQ(reshaped->uop_physical->src[1]->src[1], nb);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, reshaped->uop_logical), 4);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, reshaped->uop_physical), 4);

  /* Pinned UOp._shape rejects symbolic reshapes whose exact products differ
   * (uop/ops.py:318-336); the Tensor bridge must fail closed as well. */
  PolyUOp *bad_dims[4] = {
      poly_const_int(ctx, 1), nb,
      poly_const_int(ctx, 3), poly_const_int(ctx, 4),
  };
  ASSERT_EQ(poly_tensor_reshape_uop(ctx, slice, bad_dims, 4), NULL);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(tensor, symbolic_dot_preserves_noncontracted_bind_dimension) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *n = poly_define_var(ctx, "n", 1, 7);
  PolyUOp *nb = poly_bind_var(ctx, n, 3);

  PolyUOp *q_logical = make_buf(ctx, (int64_t[]){1, 2, 1, 4}, 4);
  PolyUOp *q_buffer = poly_buffer_on_device(ctx, POLY_FLOAT32, 8, POLY_DEVICE_CPU);
  PolyUOp *q_physical = poly_reshape(ctx, q_buffer, (int64_t[]){1, 2, 1, 4}, 4);
  PolyTensor *query = poly_tensor_create_with_roots(
      ctx, q_logical, q_physical, POLY_TENSOR_VALUE, POLY_DEVICE_CPU
  );

  PolyUOp *k_logical_base = make_buf(ctx, (int64_t[]){1, 8, 2, 4}, 4);
  PolyUOp *k_buffer = poly_buffer_on_device(ctx, POLY_FLOAT32, 64, POLY_DEVICE_CPU);
  PolyUOp *k_physical_base = poly_reshape(ctx, k_buffer, (int64_t[]){1, 8, 2, 4}, 4);
  PolyTensor *key_base = poly_tensor_create_with_roots(
      ctx, k_logical_base, k_physical_base, POLY_TENSOR_VALUE, POLY_DEVICE_CPU
  );
  PolyUOp *starts[4] = {
      poly_const_int(ctx, 0), poly_const_int(ctx, 0),
      poly_const_int(ctx, 0), poly_const_int(ctx, 0),
  };
  PolyUOp *sizes[4] = {
      poly_const_int(ctx, 1), nb,
      poly_const_int(ctx, 2), poly_const_int(ctx, 4),
  };
  PolyTensor *key = poly_tensor_shrink_uop(ctx, key_base, starts, sizes, 4);
  int64_t perm[4] = {0, 2, 3, 1};
  PolyTensor *weight = poly_tensor_permute(ctx, key, perm, 4);
  ASSERT_NOT_NULL(query);
  ASSERT_NOT_NULL(weight);

  PolyTensor *output = poly_tensor_dot(ctx, query, weight);
  ASSERT_NOT_NULL(output);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, output->uop_logical), 4);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, output->uop_physical), 4);
  ASSERT_PTR_EQ(poly_uop_shape_dim(ctx, output->uop_logical, 3), nb);
  ASSERT_PTR_EQ(poly_uop_shape_dim(ctx, output->uop_physical, 3), nb);
  ASSERT_INT_EQ(output->uop_logical->op, POLY_OP_RESHAPE);
  ASSERT_INT_EQ(output->uop_physical->op, POLY_OP_RESHAPE);
  ASSERT_INT_EQ(output->uop_physical->src[0]->op, POLY_OP_REDUCE);
  ASSERT_INT_EQ(output->uop_physical->src[1]->op, POLY_OP_STACK);
  ASSERT_INT_EQ(output->uop_physical->src[1]->n_src, 4);
  ASSERT_PTR_EQ(output->uop_physical->src[1]->src[3], nb);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(tensor, symbolic_softmax_and_max_preserve_exact_bind_dimension) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *n = poly_define_var(ctx, "n", 1, 7);
  PolyUOp *nb = poly_bind_var(ctx, n, 3);
  PolyUOp *base = make_buf(ctx, (int64_t[]){1, 2, 1, 7}, 4);
  PolyUOp *starts[4] = {
      poly_const_int(ctx, 0), poly_const_int(ctx, 0),
      poly_const_int(ctx, 0), poly_const_int(ctx, 0),
  };
  PolyUOp *sizes[4] = {
      poly_const_int(ctx, 1), poly_const_int(ctx, 2),
      poly_const_int(ctx, 1), nb,
  };
  PolyUOp *x = poly_shrink_uop(ctx, base, starts, sizes, 4);
  ASSERT_NOT_NULL(x);

  /* Pinned _softmax is max(keepdim)->detach->subtract->exp, followed by
   * sum(keepdim)->reciprocal->multiply (mixin/__init__.py:743-770). */
  PolyUOp *m = poly_max_reduce(ctx, x, -1, 1);
  PolyUOp *shifted = poly_sub(ctx, x, poly_detach(ctx, m));
  PolyUOp *e = poly_exp(ctx, shifted);
  PolyUOp *s = poly_sum_reduce(ctx, e, -1, 1);
  PolyUOp *manual = poly_mul(ctx, e, poly_alu1(ctx, POLY_OP_RECIPROCAL, s));
  PolyUOp *softmax = poly_softmax(ctx, x, -1);
  ASSERT_NOT_NULL(manual);
  ASSERT_PTR_EQ(softmax, manual);
  ASSERT_PTR_EQ(poly_uop_shape_dim(ctx, softmax, 3), nb);

  /* Pinned log_softmax reuses the exact same prefix (mixin/__init__.py:772-793). */
  PolyUOp *manual_log = poly_sub(ctx, shifted, poly_log(ctx, s));
  PolyUOp *log_softmax = poly_log_softmax(ctx, x, -1);
  ASSERT_NOT_NULL(manual_log);
  ASSERT_PTR_EQ(log_softmax, manual_log);
  ASSERT_PTR_EQ(poly_uop_shape_dim(ctx, log_softmax, 3), nb);

  /* Pinned ReduceMixin._reduce reshapes with surviving symbolic dimensions
   * instead of their allocation maxima (mixin/reduce.py:13-17). */
  PolyUOp *other_axis = poly_max_reduce(ctx, x, 1, 0);
  ASSERT_NOT_NULL(other_axis);
  ASSERT_INT_EQ(other_axis->op, POLY_OP_RESHAPE);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, other_axis), 3);
  ASSERT_PTR_EQ(poly_uop_shape_dim(ctx, other_axis, 2), nb);
  ASSERT_INT_EQ(other_axis->src[1]->op, POLY_OP_STACK);
  ASSERT_PTR_EQ(other_axis->src[1]->src[2], nb);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(tensor, custom_kernel_uses_ordered_roots_and_one_call_per_graph) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  int64_t shape[1] = {4};
  PolyTensor *c =
      poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_CPU);
  PolyTensor *a =
      poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_CPU);
  PolyTensor *b =
      poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(c);
  ASSERT_NOT_NULL(a);
  ASSERT_NOT_NULL(b);

  PolyUOp *pc = poly_uop_placeholder_like(ctx, c->uop_physical, 0);
  PolyUOp *pa = poly_uop_placeholder_like(ctx, a->uop_physical, 1);
  PolyUOp *pb = poly_uop_placeholder_like(ctx, b->uop_physical, 2);
  PolyUOp *r = poly_uop_range(ctx, 4, 0, POLY_AXIS_LOOP);
  PolyUOp *idxs[1] = {r};
  PolyUOp *ci = poly_uop_index(ctx, pc, idxs, 1, 0);
  PolyUOp *ai = poly_uop_index(ctx, pa, idxs, 1, 0);
  PolyUOp *bi = poly_uop_index(ctx, pb, idxs, 1, 0);
  PolyUOp *sum = poly_alu2(
      ctx, POLY_OP_ADD, poly_uop_load(ctx, ai), poly_uop_load(ctx, bi)
  );
  PolyUOp *store = poly_uop_store(ctx, ci, sum);
  PolyUOp *end = poly_uop_end(ctx, store, &r, 1);
  PolyUOp *body = poly_uop_sink_ex(ctx, &end, 1, "custom_add_4", 1);
  ASSERT_NOT_NULL(body);

  PolyTensor *inputs[3] = {c, a, b};
  PolyTensor *outputs[3] = {0};
  ASSERT_INT_EQ(poly_tensor_custom_kernel(ctx, body, inputs, 3, outputs), 0);
  for (int i = 0; i < 3; i++) {
    ASSERT_NOT_NULL(outputs[i]);
    ASSERT_INT_EQ(outputs[i]->uop_logical->op, POLY_OP_AFTER);
    ASSERT_INT_EQ(outputs[i]->uop_physical->op, POLY_OP_AFTER);
    ASSERT_PTR_EQ(outputs[i]->uop_logical->src[0], inputs[i]->uop_logical);
    ASSERT_PTR_EQ(outputs[i]->uop_physical->src[0], inputs[i]->uop_physical);
    ASSERT_PTR_EQ(outputs[i]->uop_logical->src[1], outputs[0]->uop_logical->src[1]);
    ASSERT_PTR_EQ(outputs[i]->uop_physical->src[1], outputs[0]->uop_physical->src[1]);
  }
  PolyUOp *logical_call = outputs[0]->uop_logical->src[1];
  PolyUOp *physical_call = outputs[0]->uop_physical->src[1];
  ASSERT_INT_EQ(logical_call->op, POLY_OP_CALL);
  ASSERT_INT_EQ(physical_call->op, POLY_OP_CALL);
  ASSERT_INT_EQ(logical_call->n_src, 4);
  ASSERT_INT_EQ(physical_call->n_src, 4);
  ASSERT_PTR_EQ(logical_call->src[0], body);
  ASSERT_PTR_EQ(physical_call->src[0], body);
  for (int i = 0; i < 3; i++) {
    ASSERT_PTR_EQ(logical_call->src[i + 1], inputs[i]->uop_logical);
    ASSERT_PTR_EQ(physical_call->src[i + 1], inputs[i]->uop_physical);
  }

  /* Pinned pm_schedule leaves SINK(KernelInfo) opaque and create_schedule
   * preserves its three ordered buffer arguments. The following consumer is
   * a separate raw SINK until compile_linear (schedule/__init__.py:94-105,
   * 118-128). */
  PolyUOp *probe_store =
      poly_store_val(ctx, c->uop_physical, outputs[0]->uop_physical);
  PolyUOp *probe_sink = poly_sink1(ctx, probe_store);
  PolySchedule *probe_schedule = poly_schedule_effect_sink(ctx, probe_sink);
  ASSERT_NOT_NULL(probe_schedule);
  ASSERT_INT_EQ(probe_schedule->template->n_calls, 2);
  PolyUOp *custom_call = poly_schedule_call(probe_schedule, 0);
  ASSERT_NOT_NULL(custom_call);
  ASSERT_INT_EQ(custom_call->op, POLY_OP_CALL);
  ASSERT_INT_EQ(custom_call->n_src, 4);
  ASSERT_PTR_EQ(custom_call->src[0], body);
  ASSERT_INT_EQ(poly_schedule_call(probe_schedule, 1)->src[0]->op, POLY_OP_SINK);
  poly_schedule_free(probe_schedule);

  float c_data[4] = {0};
  float a_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float b_data[4] = {10.0f, 20.0f, 30.0f, 40.0f};
  float out[4] = {0};
  PolyUOp *leaf_buffers[3] = {
      c->uop_physical, a->uop_physical, b->uop_physical
  };
  float *leaf_data[3] = {c_data, a_data, b_data};
  ASSERT_INT_EQ(
      realize_uop(
          ctx, outputs[0]->uop_physical, c->uop_physical, out,
          leaf_buffers, leaf_data, 3
      ),
      0
  );
  ASSERT_FLOAT_EQ(out[0], 11.0f, 0.0);
  ASSERT_FLOAT_EQ(out[1], 22.0f, 0.0);
  ASSERT_FLOAT_EQ(out[2], 33.0f, 0.0);
  ASSERT_FLOAT_EQ(out[3], 44.0f, 0.0);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(tensor, dtype_constructors_use_exact_logical_and_physical_sources) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  int64_t shape[1] = {4};
  PolyTensor *source = poly_tensor_empty(ctx, POLY_INT32, shape, 1, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(source);
  ASSERT_PTR_NEQ(source->uop_logical, source->uop_physical);

  int f32 = poly_dtype_id_by_name("float32");
  PolyTensor *casted = poly_tensor_cast_by_id(ctx, source, f32);
  PolyTensor *bitcasted = poly_tensor_bitcast_by_id(ctx, source, f32);
  ASSERT_NOT_NULL(casted);
  ASSERT_NOT_NULL(bitcasted);
  ASSERT_EQ(casted->uop_logical->op, POLY_OP_CAST);
  ASSERT_EQ(casted->uop_physical->op, POLY_OP_CAST);
  ASSERT_PTR_EQ(casted->uop_logical->src[0], source->uop_logical);
  ASSERT_PTR_EQ(casted->uop_physical->src[0], source->uop_physical);
  ASSERT_EQ(bitcasted->uop_logical->op, POLY_OP_BITCAST);
  ASSERT_EQ(bitcasted->uop_physical->op, POLY_OP_BITCAST);
  ASSERT_PTR_EQ(bitcasted->uop_logical->src[0], source->uop_logical);
  ASSERT_PTR_EQ(bitcasted->uop_physical->src[0], source->uop_physical);

  /* Pinned Tensor.bitcast decomposes this ordinary CPU width change and ends
   * in one equal-width BITCAST after reshaping/combining the uint lanes
   * (tensor.py:881-904). Shape inference scales the last dimension. */
  int64_t bytes_shape[1] = {8};
  PolyTensor *bytes = poly_tensor_empty(ctx, POLY_UINT8, bytes_shape, 1, POLY_DEVICE_CPU);
  PolyTensor *wide = poly_tensor_bitcast_by_id(ctx, bytes, f32);
  ASSERT_NOT_NULL(wide);
  ASSERT_INT_EQ(wide->uop_logical->op, POLY_OP_BITCAST);
  ASSERT_INT_EQ(wide->uop_physical->op, POLY_OP_BITCAST);
  PolyShape wide_shape = poly_uop_max_shape_cached(ctx, wide->uop_physical);
  ASSERT_INT_EQ(wide_shape.ndim, 1);
  ASSERT_INT_EQ(wide_shape.dims[0], 2);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(tensor, unequal_width_bitcast_matches_pinned_tensor_topology_and_values) {
  /* Pinned Tensor.bitcast decomposes non-DISK width changes through uint
   * shifts plus Tensor.usum/stack movement (tensor.py:881-904). These exact
   * operation counts are paired with bitcast_widen_u8_u32 and
   * bitcast_narrow_u32_u8 in the canonical graph corpus. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  int u8_id = poly_dtype_id_by_name("uint8");
  int u32_id = poly_dtype_id_by_name("uint32");

  int64_t wide_source_shape[1] = {8};
  PolyTensor *wide_source =
      poly_tensor_empty(ctx, POLY_UINT8, wide_source_shape, 1, POLY_DEVICE_CPU);
  PolyUOp *wide_fill_uop =
      poly_full_int_by_id(ctx, wide_source_shape, 1, 1, u8_id);
  PolyTensor *wide_fill = poly_tensor_create_with_roots(
      ctx, wide_fill_uop, wide_fill_uop, POLY_TENSOR_VALUE, POLY_DEVICE_CPU
  );
  ASSERT_NOT_NULL(poly_tensor_assign(ctx, wide_source, wide_fill));
  PolyTensor *wide = poly_tensor_bitcast_by_id(ctx, wide_source, u32_id);
  ASSERT_NOT_NULL(wide);
  PolyShape wide_shape = poly_uop_max_shape_cached(ctx, wide->uop_physical);
  ASSERT_INT_EQ(wide_shape.ndim, 1);
  ASSERT_INT_EQ(wide_shape.dims[0], 2);
  ASSERT_INT_EQ(count_op_in_root(ctx, wide->uop_physical, POLY_OP_SHRINK), 4);
  ASSERT_INT_EQ(count_op_in_root(ctx, wide->uop_physical, POLY_OP_CAST), 4);
  ASSERT_INT_EQ(count_op_in_root(ctx, wide->uop_physical, POLY_OP_SHL), 4);
  ASSERT_INT_EQ(count_op_in_root(ctx, wide->uop_physical, POLY_OP_ADD), 3);
  uint32_t wide_values[2] = {0};
  ASSERT_INT_EQ(read_tensor_bytes(ctx, wide, wide_values, sizeof(wide_values)), 0);
  ASSERT_INT_EQ(wide_values[0], UINT32_C(0x01010101));
  ASSERT_INT_EQ(wide_values[1], UINT32_C(0x01010101));

  int64_t narrow_source_shape[1] = {2};
  PolyTensor *narrow_source =
      poly_tensor_empty(ctx, POLY_UINT32, narrow_source_shape, 1, POLY_DEVICE_CPU);
  PolyUOp *narrow_fill_uop =
      poly_full_int_by_id(ctx, narrow_source_shape, 1, 1, u32_id);
  PolyTensor *narrow_fill = poly_tensor_create_with_roots(
      ctx, narrow_fill_uop, narrow_fill_uop, POLY_TENSOR_VALUE, POLY_DEVICE_CPU
  );
  ASSERT_NOT_NULL(poly_tensor_assign(ctx, narrow_source, narrow_fill));
  PolyTensor *narrow = poly_tensor_bitcast_by_id(ctx, narrow_source, u8_id);
  ASSERT_NOT_NULL(narrow);
  PolyShape narrow_shape = poly_uop_max_shape_cached(ctx, narrow->uop_physical);
  ASSERT_INT_EQ(narrow_shape.ndim, 1);
  ASSERT_INT_EQ(narrow_shape.dims[0], 8);
  ASSERT_INT_EQ(count_op_in_root(ctx, narrow->uop_physical, POLY_OP_SHR), 4);
  ASSERT_INT_EQ(count_op_in_root(ctx, narrow->uop_physical, POLY_OP_PAD), 4);
  ASSERT_INT_EQ(count_op_in_root(ctx, narrow->uop_physical, POLY_OP_ADD), 3);
  ASSERT_INT_EQ(count_op_in_root(ctx, narrow->uop_physical, POLY_OP_CAST), 1);
  uint8_t narrow_values[8] = {0};
  ASSERT_INT_EQ(read_tensor_bytes(ctx, narrow, narrow_values, sizeof(narrow_values)), 0);
  const uint8_t expected_narrow[8] = {1, 0, 0, 0, 1, 0, 0, 0};
  ASSERT_TRUE(memcmp(narrow_values, expected_narrow, sizeof(expected_narrow)) == 0);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(tensor, einsum_c_api_rejects_oversized_formula_parts) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = make_buf(ctx, (int64_t[]){1}, 1);
  PolyUOp *inputs[] = {x};

  PolyUOp *same = poly_einsum(ctx, "a->a", inputs, 1);
  ASSERT_NOT_NULL(same);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, same), 1);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, same)[0], 1);

  char long_rhs[128];
  memset(long_rhs, 'a', sizeof(long_rhs));
  long_rhs[0] = 'a';
  long_rhs[1] = '-';
  long_rhs[2] = '>';
  for (int i = 3; i < 126; i++)
    long_rhs[i] = 'a';
  long_rhs[126] = '\0';
  ASSERT_EQ(poly_einsum(ctx, long_rhs, inputs, 1), NULL);

  char long_formula[320];
  memset(long_formula, 'a', sizeof(long_formula) - 1);
  long_formula[sizeof(long_formula) - 1] = '\0';
  ASSERT_EQ(poly_einsum(ctx, long_formula, inputs, 1), NULL);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(tensor, einsum_c_api_matmul_shape_matches_tinygrad) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = make_buf(ctx, (int64_t[]){2, 2}, 2);
  PolyUOp *b = make_buf(ctx, (int64_t[]){2, 2}, 2);
  PolyUOp *inputs[] = {a, b};

  PolyUOp *out = poly_einsum(ctx, "ij,jk->ik", inputs, 2);
  ASSERT_NOT_NULL(out);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, out), 2);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, out)[0], 2);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, out)[1], 2);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(tensor, einsum_rearrange_reject_invalid_and_foreign_inputs) {
  PolyCtx *ctx = poly_ctx_new();
  PolyCtx *foreign_ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  ASSERT_NOT_NULL(foreign_ctx);
  PolyUOp *x = make_buf(ctx, (int64_t[]){6}, 1);
  PolyUOp *foreign = make_buf(foreign_ctx, (int64_t[]){6}, 1);
  ASSERT_NOT_NULL(x);
  ASSERT_NOT_NULL(foreign);

  PolyUOp *foreign_inputs[] = {foreign};
  ASSERT_EQ(poly_einsum(ctx, "a->a", foreign_inputs, 1), NULL);
  ASSERT_EQ(poly_rearrange(ctx, "a->a", foreign, NULL, NULL, 0), NULL);

  int64_t axis_values[] = {2, 3};
  PolyUOp *valid = poly_rearrange(ctx, "(h w)->h w", x, "h w", axis_values, 2);
  ASSERT_NOT_NULL(valid);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, valid), 2);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, valid)[0], 2);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, valid)[1], 3);

  PolyUOp *cuda =
      poly_device_uop(ctx, POLY_DEVICE_CUDA);
  PolyUOp *cpu =
      poly_device_uop(ctx, POLY_DEVICE_CPU);
  PolyUOp *to_cuda =
      poly_uop2(ctx, POLY_OP_COPY, x->dtype, x, cuda, poly_arg_none());
  PolyUOp *physical =
      poly_uop2(ctx, POLY_OP_COPY, x->dtype, to_cuda, cpu, poly_arg_none());
  PolyTensor *tensor = poly_tensor_create_with_roots(
      ctx, x, physical, POLY_TENSOR_VALUE, POLY_DEVICE_CPU
  );
  ASSERT_NOT_NULL(tensor);
  PolyUOp *expected_logical =
      poly_rearrange(ctx, "(h w)->h w", x, "h w", axis_values, 2);
  PolyUOp *expected_physical =
      poly_rearrange(ctx, "(h w)->h w", physical, "h w", axis_values, 2);
  PolyTensor *tensor_valid = poly_tensor_rearrange(
      ctx, "(h w)->h w", tensor, "h w", axis_values, 2
  );
  ASSERT_NOT_NULL(tensor_valid);
  ASSERT_EQ(poly_tensor_uop_logical(tensor_valid), expected_logical);
  ASSERT_EQ(poly_tensor_uop_physical(tensor_valid), expected_physical);
  ASSERT_EQ(
      poly_tensor_rearrange(
          foreign_ctx, "(h w)->h w", tensor, "h w", axis_values, 2
      ),
      NULL
  );

  char long_formula[320];
  memset(long_formula, 'a', sizeof(long_formula) - 1);
  memcpy(long_formula + sizeof(long_formula) - 5, "->a", 4);
  long_formula[sizeof(long_formula) - 1] = '\0';
  ASSERT_EQ(poly_rearrange(ctx, long_formula, x, NULL, NULL, 0), NULL);
  ASSERT_EQ(poly_rearrange(ctx, "a", x, NULL, NULL, 0), NULL);
  ASSERT_EQ(poly_rearrange(ctx, "a->a->a", x, NULL, NULL, 0), NULL);
  ASSERT_EQ(poly_rearrange(ctx, "((a))->a", x, NULL, NULL, 0), NULL);
  ASSERT_EQ(poly_rearrange(ctx, "(a->a", x, NULL, NULL, 0), NULL);
  ASSERT_EQ(poly_tensor_rearrange(ctx, "(a->a", tensor, NULL, NULL, 0), NULL);
  ASSERT_EQ(
      poly_rearrange(
          ctx,
          "a b c d e f g h i j k l m n o p q->"
          "a b c d e f g h i j k l m n o p q",
          x, NULL, NULL, 0
      ),
      NULL
  );

  int64_t overflow_values[] = {INT64_MAX, 2};
  ASSERT_EQ(
      poly_rearrange(ctx, "(h w)->h w", x, "h w", overflow_values, 2),
      NULL
  );

  poly_ctx_destroy(foreign_ctx);
  poly_ctx_destroy(ctx);
  PASS();
}

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

TEST(pe, relu_e2e_preserves_false_branch_zero) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_buffer_f32(ctx, 4);
  PolyUOp *out_buf = poly_buffer_f32(ctx, 4);
  PolyUOp *r = poly_relu(ctx, x);
  ASSERT_NOT_NULL(r);

  float dx[] = {-1.0f, 0.0f, 1.0f, 2.0f};
  float dout[4] = {0};
  PolyUOp *leaves[] = {x};
  float *ld[] = {dx};
  ASSERT_INT_EQ(realize_uop(ctx, r, out_buf, dout, leaves, ld, 1), 0);
  ASSERT_FLOAT_EQ(dout[0], 0.0f, 1e-6f);
  ASSERT_FLOAT_EQ(dout[1], 0.0f, 1e-6f);
  ASSERT_FLOAT_EQ(dout[2], 1.0f, 1e-6f);
  ASSERT_FLOAT_EQ(dout[3], 2.0f, 1e-6f);

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
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, r)[1], 6);

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

TEST(pe, cat_many_more_than_max_dims_e2e) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *parts[19];
  for (int i = 0; i < 19; i++)
    parts[i] = poly_full(ctx, (int64_t[]){1}, 1, (double)(i + 1));
  PolyUOp *r = poly_cat(ctx, parts, 19, 0);
  ASSERT_NOT_NULL(r);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, r), 1);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, r)[0], 19);

  PolyUOp *out_buf = poly_buffer_f32(ctx, 19);
  float dout[19] = {0};
  ASSERT_INT_EQ(realize_uop(ctx, r, out_buf, dout, NULL, NULL, 0), 0);
  for (int i = 0; i < 19; i++)
    ASSERT_FLOAT_EQ(dout[i], (float)(i + 1), 1e-6f);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, argmax_e2e) {
  /* Reference: argmax([1,5,3,2,4]) = 1 */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_buffer_f32(ctx, 5);
  PolyUOp *out_buf = poly_buffer_f32(ctx, 1);
  PolyUOp *r = poly_argmax(ctx, x, 0, 0);
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
  PolyUOp *r = poly_argmax(ctx, x, 1, 0);
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

TEST(pe, tensor_argmax_builds_both_roots_from_exact_occurrences) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *logical = make_buf(ctx, (int64_t[]){2, 3}, 2);
  PolyUOp *cuda = poly_device_uop(ctx, POLY_DEVICE_CUDA);
  PolyUOp *cpu = poly_device_uop(ctx, POLY_DEVICE_CPU);
  PolyUOp *to_cuda =
      poly_uop2(ctx, POLY_OP_COPY, logical->dtype, logical, cuda, poly_arg_none());
  PolyUOp *physical =
      poly_uop2(ctx, POLY_OP_COPY, logical->dtype, to_cuda, cpu, poly_arg_none());
  PolyTensor *src = poly_tensor_create_with_roots(
      ctx, logical, physical, POLY_TENSOR_VALUE, POLY_DEVICE_CPU
  );
  ASSERT_NOT_NULL(src);
  poly_tensor_set_requires_grad(src, true);

  PolyUOp *expected_logical = poly_argmax(ctx, logical, 1, 1);
  PolyUOp *expected_physical = poly_argmax(ctx, physical, 1, 1);
  ASSERT_NOT_NULL(expected_logical);
  ASSERT_NOT_NULL(expected_physical);

  PolyTensor *out = poly_tensor_argmax(ctx, src, 1, true);
  ASSERT_NOT_NULL(out);
  ASSERT_EQ(poly_tensor_uop_logical(out), expected_logical);
  ASSERT_EQ(poly_tensor_uop_physical(out), expected_physical);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, poly_tensor_uop_physical(out)), 2);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, poly_tensor_uop_physical(out))[0], 2);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, poly_tensor_uop_physical(out))[1], 1);
  ASSERT_FALSE(poly_tensor_requires_grad(out));
  ASSERT_TRUE(poly_tensor_requires_grad_is_set(out));

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, tensor_argmax_rejects_foreign_context) {
  PolyCtx *owner = poly_ctx_new();
  PolyCtx *foreign = poly_ctx_new();
  ASSERT_NOT_NULL(owner);
  ASSERT_NOT_NULL(foreign);
  PolyTensor *src =
      poly_tensor_empty(owner, POLY_FLOAT32, (int64_t[]){2, 3}, 2, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(src);
  ASSERT_EQ(poly_tensor_argmax(foreign, src, 1, false), NULL);
  poly_ctx_destroy(foreign);
  poly_ctx_destroy(owner);
  PASS();
}

TEST(pe, tensor_gelu_family_builds_both_roots_from_exact_occurrences) {
  PolyCtx *ctx = poly_ctx_new();
  PolyCtx *foreign = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  ASSERT_NOT_NULL(foreign);

  PolyUOp *logical = make_buf(ctx, (int64_t[]){2, 2}, 2);
  PolyUOp *cuda = poly_device_uop(ctx, POLY_DEVICE_CUDA);
  PolyUOp *cpu = poly_device_uop(ctx, POLY_DEVICE_CPU);
  PolyUOp *to_cuda =
      poly_uop2(ctx, POLY_OP_COPY, logical->dtype, logical, cuda, poly_arg_none());
  PolyUOp *physical =
      poly_uop2(ctx, POLY_OP_COPY, logical->dtype, to_cuda, cpu, poly_arg_none());
  PolyTensor *src = poly_tensor_create_with_roots(
      ctx, logical, physical, POLY_TENSOR_VALUE, POLY_DEVICE_CPU
  );
  ASSERT_NOT_NULL(src);
  poly_tensor_set_requires_grad(src, true);

  PolyUOp *expected_gelu_logical = poly_gelu(ctx, logical);
  PolyUOp *expected_gelu_physical = poly_gelu(ctx, physical);
  PolyTensor *gelu = poly_tensor_gelu(ctx, src);
  ASSERT_NOT_NULL(expected_gelu_logical);
  ASSERT_NOT_NULL(expected_gelu_physical);
  ASSERT_NOT_NULL(gelu);
  ASSERT_EQ(poly_tensor_uop_logical(gelu), expected_gelu_logical);
  ASSERT_EQ(poly_tensor_uop_physical(gelu), expected_gelu_physical);
  ASSERT_TRUE(poly_tensor_requires_grad(gelu));
  ASSERT_TRUE(poly_tensor_requires_grad_is_set(gelu));

  PolyUOp *expected_quick_logical = poly_quick_gelu(ctx, logical);
  PolyUOp *expected_quick_physical = poly_quick_gelu(ctx, physical);
  PolyTensor *quick = poly_tensor_quick_gelu(ctx, src);
  ASSERT_NOT_NULL(expected_quick_logical);
  ASSERT_NOT_NULL(expected_quick_physical);
  ASSERT_NOT_NULL(quick);
  ASSERT_EQ(poly_tensor_uop_logical(quick), expected_quick_logical);
  ASSERT_EQ(poly_tensor_uop_physical(quick), expected_quick_physical);
  ASSERT_TRUE(poly_tensor_requires_grad(quick));
  ASSERT_TRUE(poly_tensor_requires_grad_is_set(quick));

  ASSERT_EQ(poly_tensor_gelu(foreign, src), NULL);
  ASSERT_EQ(poly_tensor_quick_gelu(foreign, src), NULL);

  poly_ctx_destroy(foreign);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, tensor_log1p_expm1_build_both_roots_from_exact_occurrences) {
  PolyCtx *ctx = poly_ctx_new();
  PolyCtx *foreign = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  ASSERT_NOT_NULL(foreign);

  PolyUOp *logical = make_buf(ctx, (int64_t[]){4}, 1);
  PolyUOp *cuda = poly_device_uop(ctx, POLY_DEVICE_CUDA);
  PolyUOp *cpu = poly_device_uop(ctx, POLY_DEVICE_CPU);
  PolyUOp *to_cuda =
      poly_uop2(ctx, POLY_OP_COPY, logical->dtype, logical, cuda, poly_arg_none());
  PolyUOp *physical =
      poly_uop2(ctx, POLY_OP_COPY, logical->dtype, to_cuda, cpu, poly_arg_none());
  PolyTensor *src = poly_tensor_create_with_roots(
      ctx, logical, physical, POLY_TENSOR_VALUE, POLY_DEVICE_CPU
  );
  ASSERT_NOT_NULL(src);
  poly_tensor_set_requires_grad(src, true);

  PolyUOp *expected_log1p_logical = poly_log1p(ctx, logical);
  PolyUOp *expected_log1p_physical = poly_log1p(ctx, physical);
  PolyTensor *log1p = poly_tensor_log1p(ctx, src);
  ASSERT_NOT_NULL(expected_log1p_logical);
  ASSERT_NOT_NULL(expected_log1p_physical);
  ASSERT_NOT_NULL(log1p);
  ASSERT_EQ(poly_tensor_uop_logical(log1p), expected_log1p_logical);
  ASSERT_EQ(poly_tensor_uop_physical(log1p), expected_log1p_physical);
  ASSERT_TRUE(poly_tensor_requires_grad(log1p));
  ASSERT_TRUE(poly_tensor_requires_grad_is_set(log1p));

  PolyUOp *expected_expm1_logical = poly_expm1(ctx, logical);
  PolyUOp *expected_expm1_physical = poly_expm1(ctx, physical);
  PolyTensor *expm1 = poly_tensor_expm1(ctx, src);
  ASSERT_NOT_NULL(expected_expm1_logical);
  ASSERT_NOT_NULL(expected_expm1_physical);
  ASSERT_NOT_NULL(expm1);
  ASSERT_EQ(poly_tensor_uop_logical(expm1), expected_expm1_logical);
  ASSERT_EQ(poly_tensor_uop_physical(expm1), expected_expm1_physical);
  ASSERT_TRUE(poly_tensor_requires_grad(expm1));
  ASSERT_TRUE(poly_tensor_requires_grad_is_set(expm1));

  ASSERT_EQ(poly_tensor_log1p(foreign, src), NULL);
  ASSERT_EQ(poly_tensor_expm1(foreign, src), NULL);

  poly_ctx_destroy(foreign);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, cast_same_dtype_elides_and_scalar_target_preserves_vector_lanes) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *scalar = poly_const_int(ctx, 3);
  ASSERT_EQ(poly_cast(ctx, scalar, POLY_INT32), scalar);

  PolyUOp *lanes[2] = {poly_const_int(ctx, 1), poly_const_int(ctx, 2)};
  PolyUOp *vector =
      poly_uop(ctx, POLY_OP_STACK, poly_dtype_vec(POLY_INT32, 2), lanes, 2, poly_arg_none());
  PolyUOp *cast = poly_cast(ctx, vector, POLY_FLOAT32);
  ASSERT_NOT_NULL(cast);
  ASSERT_INT_EQ(cast->op, POLY_OP_CAST);
  ASSERT_INT_EQ(cast->dtype.count, 2);
  ASSERT_TRUE(poly_dtype_eq(poly_dtype_scalar(cast->dtype), POLY_FLOAT32));
  ASSERT_EQ(cast->src[0], vector);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, gather_dim_e2e_matches_tinygrad_probe) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = make_buf(ctx, (int64_t[]){2, 3, 4}, 3);
  PolyUOp *idx = poly_reshape(ctx, poly_buffer(ctx, POLY_INT32, 8), (int64_t[]){2, 2, 2}, 3);
  PolyUOp *out_buf = poly_buffer_f32(ctx, 8);
  PolyUOp *r = poly_gather_dim(ctx, x, 1, idx);
  ASSERT_NOT_NULL(r);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, r), 3);

  float dx[24];
  for (int i = 0; i < 24; i++)
    dx[i] = (float)i;
  int32_t di[] = {0, 2, 1, 0, 2, 1, 0, 2};
  float dout[8] = {0};
  PolyUOp *leaves[] = {base_buf(x), base_buf(idx)};
  float *ld[] = {dx, (float *)di};
  ASSERT_INT_EQ(realize_uop(ctx, r, out_buf, dout, leaves, ld, 2), 0);
  const float expected[] = {0, 9, 4, 1, 20, 17, 12, 21};
  for (int i = 0; i < 8; i++)
    ASSERT_FLOAT_EQ(dout[i], expected[i], 1e-5f);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, scatter_e2e_matches_tinygrad_probe) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *self0 = make_buf(ctx, (int64_t[]){3, 5}, 2);
  PolyUOp *idx0 = poly_reshape(ctx, poly_buffer(ctx, POLY_INT32, 4), (int64_t[]){1, 4}, 2);
  PolyUOp *src0 = make_buf(ctx, (int64_t[]){2, 5}, 2);
  PolyUOp *r0 = poly_scatter(ctx, self0, 0, idx0, src0, NULL);
  ASSERT_NOT_NULL(r0);
  float dself0[15] = {0};
  int32_t didx0[] = {0, 1, 2, 0};
  float dsrc0[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  float got0[15] = {0};
  PolyUOp *leaves0[] = {base_buf(self0), base_buf(idx0), base_buf(src0)};
  float *ld0[] = {dself0, (float *)didx0, dsrc0};
  ASSERT_INT_EQ(realize_uop(ctx, r0, poly_buffer_f32(ctx, 15), got0, leaves0, ld0, 3), 0);
  const float exp0[] = {1, 0, 0, 4, 0, 0, 2, 0, 0, 0, 0, 0, 3, 0, 0};
  for (int i = 0; i < 15; i++)
    ASSERT_FLOAT_EQ(got0[i], exp0[i], 1e-5f);

  PolyUOp *self = make_buf(ctx, (int64_t[]){3, 5}, 2);
  PolyUOp *idx = poly_reshape(ctx, poly_buffer(ctx, POLY_INT32, 9), (int64_t[]){3, 3}, 2);
  PolyUOp *src = make_buf(ctx, (int64_t[]){3, 3}, 2);
  PolyUOp *out_buf = poly_buffer_f32(ctx, 15);

  PolyUOp *r = poly_scatter(ctx, self, 1, idx, src, NULL);
  ASSERT_NOT_NULL(r);

  float dself[15] = {0};
  int32_t didx[] = {0, 1, 2, 0, 1, 4, 2, 3, 4};
  float dsrc[] = {1, 2, 3, 6, 7, 8, 9, 10, 11};
  float dout[15] = {0};
  PolyUOp *leaves[] = {base_buf(self), base_buf(idx), base_buf(src)};
  float *ld[] = {dself, (float *)didx, dsrc};
  ASSERT_INT_EQ(realize_uop(ctx, r, out_buf, dout, leaves, ld, 3), 0);
  const float exp[] = {1, 2, 3, 0, 0, 6, 7, 0, 0, 8, 0, 0, 9, 10, 11};
  for (int i = 0; i < 15; i++)
    ASSERT_FLOAT_EQ(dout[i], exp[i], 1e-5f);

  PolyUOp *self_dup = make_buf(ctx, (int64_t[]){1, 4}, 2);
  PolyUOp *idx_dup = poly_reshape(ctx, poly_buffer(ctx, POLY_INT32, 3), (int64_t[]){1, 3}, 2);
  PolyUOp *src_dup = make_buf(ctx, (int64_t[]){1, 3}, 2);
  PolyUOp *dup = poly_scatter(ctx, self_dup, 1, idx_dup, src_dup, NULL);
  ASSERT_NOT_NULL(dup);
  float dself_dup[] = {0, 0, 0, 0};
  int32_t didx_dup[] = {1, 1, 2};
  float dsrc_dup[] = {7, 9, 8};
  float got_dup[4] = {0};
  PolyUOp *dup_leaves[] = {base_buf(self_dup), base_buf(idx_dup), base_buf(src_dup)};
  float *dup_ld[] = {dself_dup, (float *)didx_dup, dsrc_dup};
  ASSERT_INT_EQ(realize_uop(ctx, dup, poly_buffer_f32(ctx, 4), got_dup, dup_leaves, dup_ld, 3), 0);
  const float exp_dup[] = {0, 9, 8, 0};
  for (int i = 0; i < 4; i++)
    ASSERT_FLOAT_EQ(got_dup[i], exp_dup[i], 1e-5f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, scatter_reduce_e2e_matches_tinygrad_probe) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *self = make_buf(ctx, (int64_t[]){1, 5}, 2);
  PolyUOp *idx = poly_reshape(ctx, poly_buffer(ctx, POLY_INT32, 10), (int64_t[]){1, 10}, 2);
  PolyUOp *src = make_buf(ctx, (int64_t[]){1, 10}, 2);
  float dself[] = {1, 2, 3, 4, 5};
  int32_t didx[] = {0, 0, 1, 1, 2, 2, 3, 3, 4, 4};
  float dsrc[] = {1, 6, 2, 7, 3, 8, 4, 9, 5, 10};
  PolyUOp *leaves[] = {base_buf(self), base_buf(idx), base_buf(src)};
  float *ld[] = {dself, (float *)didx, dsrc};

  const struct {
    const char *reduce;
    int include_self;
    float expected[5];
  } cases[] = {
      {"sum", 1, {8, 11, 14, 17, 20}},
      {"prod", 1, {6, 28, 72, 144, 250}},
      {"mean", 0, {3.5f, 4.5f, 5.5f, 6.5f, 7.5f}},
      {"amax", 1, {6, 7, 8, 9, 10}},
      {"amin", 1, {1, 2, 3, 4, 5}},
  };

  for (int c = 0; c < (int)(sizeof(cases) / sizeof(cases[0])); c++) {
    PolyUOp *r =
        poly_scatter_reduce(ctx, self, 1, idx, src, cases[c].reduce, cases[c].include_self);
    ASSERT_NOT_NULL(r);
    float got[5] = {0};
    ASSERT_INT_EQ(realize_uop(ctx, r, poly_buffer_f32(ctx, 5), got, leaves, ld, 3), 0);
    for (int i = 0; i < 5; i++)
      ASSERT_FLOAT_EQ(got[i], cases[c].expected[i], 1e-5f);
  }

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, tensor_scatter_builds_both_roots_from_exact_occurrences) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  int64_t shape[2] = {1, 5};
  PolyTensor *self =
      poly_tensor_empty(ctx, POLY_FLOAT32, shape, 2, POLY_DEVICE_CPU);
  PolyTensor *index =
      poly_tensor_empty(ctx, POLY_INT32, shape, 2, POLY_DEVICE_CPU);
  PolyTensor *src =
      poly_tensor_empty(ctx, POLY_FLOAT32, shape, 2, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(self);
  ASSERT_NOT_NULL(index);
  ASSERT_NOT_NULL(src);

  PolyUOp *logical_scatter = poly_scatter(
      ctx, self->uop_logical, 1, index->uop_logical, src->uop_logical, NULL
  );
  PolyUOp *physical_scatter = poly_scatter(
      ctx, self->uop_physical, 1, index->uop_physical, src->uop_physical, NULL
  );
  PolyTensor *scatter = poly_tensor_scatter(ctx, self, 1, index, src, NULL);
  ASSERT_NOT_NULL(scatter);
  ASSERT_PTR_EQ(scatter->uop_logical, logical_scatter);
  ASSERT_PTR_EQ(scatter->uop_physical, physical_scatter);

  PolyUOp *logical_sum = poly_scatter_reduce(
      ctx, self->uop_logical, 1, index->uop_logical, src->uop_logical, "sum", 1
  );
  PolyUOp *physical_sum = poly_scatter_reduce(
      ctx, self->uop_physical, 1, index->uop_physical, src->uop_physical, "sum", 1
  );
  PolyTensor *sum =
      poly_tensor_scatter_reduce(ctx, self, 1, index, src, "sum", 1);
  ASSERT_NOT_NULL(sum);
  ASSERT_PTR_EQ(sum->uop_logical, logical_sum);
  ASSERT_PTR_EQ(sum->uop_physical, physical_sum);

  float self_data[5] = {1, 2, 3, 4, 5};
  int32_t index_data[5] = {0, 1, 1, 3, 4};
  float src_data[5] = {6, 7, 8, 9, 10};
  PolyUOp *leaves[3] = {
      base_buf(self->uop_physical),
      base_buf(index->uop_physical),
      base_buf(src->uop_physical),
  };
  float *leaf_data[3] = {
      self_data,
      (float *)index_data,
      src_data,
  };
  float scatter_values[5] = {0};
  ASSERT_INT_EQ(
      realize_uop(
          ctx, scatter->uop_physical, poly_buffer_f32(ctx, 5), scatter_values,
          leaves, leaf_data, 3
      ),
      0
  );
  const float expected_scatter[5] = {6, 8, 3, 9, 10};
  for (int i = 0; i < 5; i++)
    ASSERT_FLOAT_EQ(scatter_values[i], expected_scatter[i], 1e-5f);

  float sum_values[5] = {0};
  ASSERT_INT_EQ(
      realize_uop(
          ctx, sum->uop_physical, poly_buffer_f32(ctx, 5), sum_values,
          leaves, leaf_data, 3
      ),
      0
  );
  const float expected_sum[5] = {7, 17, 3, 13, 15};
  for (int i = 0; i < 5; i++)
    ASSERT_FLOAT_EQ(sum_values[i], expected_sum[i], 1e-5f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, tensor_einsum_builds_both_roots_from_exact_occurrences) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  int64_t shape[2] = {2, 2};
  PolyTensor *a =
      poly_tensor_empty(ctx, POLY_FLOAT32, shape, 2, POLY_DEVICE_CPU);
  PolyTensor *b =
      poly_tensor_empty(ctx, POLY_FLOAT32, shape, 2, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(a);
  ASSERT_NOT_NULL(b);

  PolyUOp *logical_inputs[2] = {a->uop_logical, b->uop_logical};
  PolyUOp *physical_inputs[2] = {a->uop_physical, b->uop_physical};
  PolyUOp *logical =
      poly_einsum(ctx, "ij,jk->ik", logical_inputs, 2);
  PolyUOp *physical =
      poly_einsum(ctx, "ij,jk->ik", physical_inputs, 2);
  PolyTensor *inputs[2] = {a, b};
  PolyTensor *out =
      poly_tensor_einsum(ctx, "ij,jk->ik", inputs, 2);
  ASSERT_NOT_NULL(out);
  ASSERT_PTR_EQ(out->uop_logical, logical);
  ASSERT_PTR_EQ(out->uop_physical, physical);

  float a_data[4] = {1, 2, 3, 4};
  float b_data[4] = {5, 6, 7, 8};
  PolyUOp *leaves[2] = {
      base_buf(a->uop_physical),
      base_buf(b->uop_physical),
  };
  float *leaf_data[2] = {a_data, b_data};
  float values[4] = {0};
  ASSERT_INT_EQ(
      realize_uop(
          ctx, out->uop_physical, poly_buffer_f32(ctx, 4), values,
          leaves, leaf_data, 2
      ),
      0
  );
  const float expected[4] = {19, 22, 43, 50};
  for (int i = 0; i < 4; i++)
    ASSERT_FLOAT_EQ(values[i], expected[i], 1e-5f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, unsigned_scatter_amin_matches_pinned_inverse_max_inverse) {
  /* Pinned scatter amin fills with the positive dtype.max and reduces through
   * Tensor.min's inverse/MAX/inverse program
   * (mixin/__init__.py:1206-1211, mixin/elementwise.py:379-393). */
  PolyDType dtypes[] = {POLY_UINT8, POLY_UINT16, POLY_UINT32, POLY_UINT64};
  const char *maxima[] = {
      "255", "65535", "4294967295", "18446744073709551615",
  };
  for (int d = 0; d < 4; d++) {
    PolyCtx *ctx = poly_ctx_new();
    PolyDType dtype = dtypes[d];
    PolyUOp *self =
        poly_reshape(ctx, poly_buffer(ctx, dtype, 2), (int64_t[]){1, 2}, 2);
    PolyUOp *idx =
        poly_reshape(ctx, poly_buffer(ctx, POLY_INT32, 2), (int64_t[]){1, 2}, 2);
    PolyUOp *src =
        poly_reshape(ctx, poly_buffer(ctx, dtype, 2), (int64_t[]){1, 2}, 2);
    PolyUOp *result = poly_scatter_reduce(ctx, self, 1, idx, src, "amin", 0);
    ASSERT_NOT_NULL(result);

    int n_topo = 0, neg_count = 0;
    bool saw_positive_max = false;
    PolyUOp **topo = poly_toposort_alloc(ctx, result, &n_topo);
    ASSERT_NOT_NULL(topo);
    for (int i = 0; i < n_topo; i++) {
      PolyUOp *u = topo[i];
      if (u->op == POLY_OP_NEG) neg_count++;
      if (u->op != POLY_OP_CONST || !poly_dtype_eq(u->dtype, dtype) ||
          (u->arg.kind != POLY_ARG_INT && u->arg.kind != POLY_ARG_BIGINT))
        continue;
      char *decimal = poly_arg_integer_to_decimal(u->arg);
      if (decimal && strcmp(decimal, maxima[d]) == 0) saw_positive_max = true;
      free(decimal);
    }
    ASSERT_INT_EQ(neg_count, 0);
    ASSERT_TRUE(saw_positive_max);
    poly_toposort_free(topo);

    union {
      uint64_t align;
      uint8_t bytes[16];
    } self_data = {0}, src_data = {0}, output = {0};
    uint64_t self_values[2] = {100, 100}, src_values[2] = {0, 1};
    int32_t idx_data[2] = {0, 0};
    set_unsigned_values(self_data.bytes, dtype, self_values, 2);
    set_unsigned_values(src_data.bytes, dtype, src_values, 2);
    PolyUOp *leaves[] = {base_buf(self), base_buf(idx), base_buf(src)};
    float *data[] = {
        (float *)self_data.bytes, (float *)idx_data, (float *)src_data.bytes,
    };
    ASSERT_INT_EQ(
        realize_uop(
            ctx, result, poly_buffer(ctx, dtype, 2), output.bytes, leaves, data, 3
        ),
        0
    );
    ASSERT_TRUE(get_unsigned_value(output.bytes, dtype, 0) == 0);
    ASSERT_TRUE(get_unsigned_value(output.bytes, dtype, 1) == 100);
    poly_ctx_destroy(ctx);
  }
  PASS();
}

TEST(pe, sort_topk_e2e_matches_tinygrad_probe) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = make_buf(ctx, (int64_t[]){2, 5}, 2);
  float dx[] = {0.1f, 0.5f, 1.2f, 3.4f, 2.1f, 2.2f, 1.9f, 0.3f, 4.5f, 0.8f};
  PolyUOp *leaves[] = {base_buf(x)};
  float *ld[] = {dx};

  PolyUOp *vals = NULL, *idx = NULL;
  ASSERT_INT_EQ(poly_sort(ctx, x, 1, 0, &vals, &idx), 0);
  ASSERT_NOT_NULL(vals);
  ASSERT_NOT_NULL(idx);

  PolyUOp *out_vals = poly_buffer_f32(ctx, 10);
  PolyUOp *out_idx = poly_buffer_f32(ctx, 10);
  PolyUOp *out_idx_i32 = poly_buffer(ctx, POLY_INT32, 10);
  float got_vals[10] = {0}, got_idx[10] = {0};
  int32_t got_idx_i32[10] = {0};
  ASSERT_INT_EQ(realize_uop(ctx, vals, out_vals, got_vals, leaves, ld, 1), 0);
  ASSERT_INT_EQ(realize_uop(ctx, poly_cast(ctx, idx, POLY_FLOAT32), out_idx, got_idx, leaves, ld, 1), 0);
  ASSERT_INT_EQ(realize_uop(ctx, idx, out_idx_i32, got_idx_i32, leaves, ld, 1), 0);
  const float exp_vals[] = {0.1f, 0.5f, 1.2f, 2.1f, 3.4f, 0.3f, 0.8f, 1.9f, 2.2f, 4.5f};
  const float exp_idx[] = {0, 1, 2, 4, 3, 2, 4, 1, 0, 3};
  for (int i = 0; i < 10; i++) {
    ASSERT_FLOAT_EQ(got_vals[i], exp_vals[i], 1e-5f);
    ASSERT_FLOAT_EQ(got_idx[i], exp_idx[i], 1e-5f);
    ASSERT_INT_EQ(got_idx_i32[i], (int32_t)exp_idx[i]);
  }

  PolyUOp *top_vals = NULL, *top_idx = NULL;
  ASSERT_INT_EQ(poly_topk(ctx, x, 2, 1, 1, 1, &top_vals, &top_idx), 0);
  ASSERT_NOT_NULL(top_vals);
  ASSERT_NOT_NULL(top_idx);
  PolyUOp *out_top_vals = poly_buffer_f32(ctx, 4);
  PolyUOp *out_top_idx = poly_buffer_f32(ctx, 4);
  PolyUOp *out_top_idx_i32 = poly_buffer(ctx, POLY_INT32, 4);
  float got_top_vals[4] = {0}, got_top_idx[4] = {0};
  int32_t got_top_idx_i32[4] = {0};
  ASSERT_INT_EQ(realize_uop(ctx, top_vals, out_top_vals, got_top_vals, leaves, ld, 1), 0);
  ASSERT_INT_EQ(realize_uop(ctx, poly_cast(ctx, top_idx, POLY_FLOAT32), out_top_idx, got_top_idx, leaves, ld, 1), 0);
  ASSERT_INT_EQ(realize_uop(ctx, top_idx, out_top_idx_i32, got_top_idx_i32, leaves, ld, 1), 0);
  const float exp_top_vals[] = {3.4f, 2.1f, 4.5f, 2.2f};
  const float exp_top_idx[] = {3, 4, 3, 0};
  for (int i = 0; i < 4; i++) {
    ASSERT_FLOAT_EQ(got_top_vals[i], exp_top_vals[i], 1e-5f);
    ASSERT_FLOAT_EQ(got_top_idx[i], exp_top_idx[i], 1e-5f);
    ASSERT_INT_EQ(got_top_idx_i32[i], (int32_t)exp_top_idx[i]);
  }

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, tensor_topk_builds_both_roots_from_exact_occurrences) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *logical = make_buf(ctx, (int64_t[]){4}, 1);
  PolyUOp *cuda = poly_device_uop(ctx, POLY_DEVICE_CUDA);
  PolyUOp *cpu = poly_device_uop(ctx, POLY_DEVICE_CPU);
  PolyUOp *to_cuda =
      poly_uop2(ctx, POLY_OP_COPY, logical->dtype, logical, cuda, poly_arg_none());
  PolyUOp *physical =
      poly_uop2(ctx, POLY_OP_COPY, logical->dtype, to_cuda, cpu, poly_arg_none());
  PolyTensor *src = poly_tensor_create_with_roots(
      ctx, logical, physical, POLY_TENSOR_VALUE, POLY_DEVICE_CPU
  );
  ASSERT_NOT_NULL(src);

  PolyUOp *expected_logical_values = NULL, *expected_logical_indices = NULL;
  PolyUOp *expected_physical_values = NULL, *expected_physical_indices = NULL;
  ASSERT_INT_EQ(
      poly_topk(
          ctx, logical, 2, 0, 1, 1, &expected_logical_values, &expected_logical_indices
      ),
      0
  );
  ASSERT_INT_EQ(
      poly_topk(
          ctx, physical, 2, 0, 1, 1, &expected_physical_values, &expected_physical_indices
      ),
      0
  );

  PolyTensor *values = NULL, *indices = NULL;
  ASSERT_INT_EQ(poly_tensor_topk(ctx, src, 2, 0, 1, 1, &values, &indices), 0);
  ASSERT_NOT_NULL(values);
  ASSERT_NOT_NULL(indices);
  ASSERT_EQ(poly_tensor_uop_logical(values), expected_logical_values);
  ASSERT_EQ(poly_tensor_uop_physical(values), expected_physical_values);
  ASSERT_EQ(poly_tensor_uop_logical(indices), expected_logical_indices);
  ASSERT_EQ(poly_tensor_uop_physical(indices), expected_physical_indices);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, tensor_linalg_builds_both_roots_from_exact_occurrences) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  int64_t matrix_shape[2] = {2, 2};
  int64_t vector_shape[1] = {2};
  PolyTensor *a =
      poly_tensor_empty(ctx, POLY_FLOAT32, matrix_shape, 2, POLY_DEVICE_CPU);
  PolyTensor *b =
      poly_tensor_empty(ctx, POLY_FLOAT32, vector_shape, 1, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(a);
  ASSERT_NOT_NULL(b);

  PolyUOp *logical_q = NULL, *logical_r = NULL;
  PolyUOp *physical_q = NULL, *physical_r = NULL;
  ASSERT_INT_EQ(
      poly_qr_ex(
          ctx, a->uop_logical, POLY_QR_COMPLETE, &logical_q, &logical_r
      ),
      0
  );
  ASSERT_INT_EQ(
      poly_qr_ex(
          ctx, a->uop_physical, POLY_QR_COMPLETE, &physical_q, &physical_r
      ),
      0
  );
  PolyTensor *q = NULL, *r = NULL;
  ASSERT_INT_EQ(
      poly_tensor_qr_ex(ctx, a, POLY_QR_COMPLETE, &q, &r),
      0
  );
  ASSERT_NOT_NULL(q);
  ASSERT_NOT_NULL(r);
  ASSERT_PTR_EQ(q->uop_logical, logical_q);
  ASSERT_PTR_EQ(q->uop_physical, physical_q);
  ASSERT_PTR_EQ(r->uop_logical, logical_r);
  ASSERT_PTR_EQ(r->uop_physical, physical_r);

  PolyUOp *logical_tri =
      poly_triangular_solve(ctx, a->uop_logical, b->uop_logical, 0, 0, 0);
  PolyUOp *physical_tri =
      poly_triangular_solve(ctx, a->uop_physical, b->uop_physical, 0, 0, 0);
  PolyTensor *tri = poly_tensor_triangular_solve(ctx, a, b, 0, 0, 0);
  ASSERT_NOT_NULL(tri);
  ASSERT_PTR_EQ(tri->uop_logical, logical_tri);
  ASSERT_PTR_EQ(tri->uop_physical, physical_tri);

  PolyUOp *logical_chol = poly_cholesky(ctx, a->uop_logical, 0);
  PolyUOp *physical_chol = poly_cholesky(ctx, a->uop_physical, 0);
  PolyTensor *chol = poly_tensor_cholesky(ctx, a, 0);
  ASSERT_NOT_NULL(chol);
  ASSERT_PTR_EQ(chol->uop_logical, logical_chol);
  ASSERT_PTR_EQ(chol->uop_physical, physical_chol);

  PolyUOp *logical_chol_solve =
      poly_cholesky_solve(ctx, chol->uop_logical, b->uop_logical, 0);
  PolyUOp *physical_chol_solve =
      poly_cholesky_solve(ctx, chol->uop_physical, b->uop_physical, 0);
  PolyTensor *chol_solve = poly_tensor_cholesky_solve(ctx, chol, b, 0);
  ASSERT_NOT_NULL(chol_solve);
  ASSERT_PTR_EQ(chol_solve->uop_logical, logical_chol_solve);
  ASSERT_PTR_EQ(chol_solve->uop_physical, physical_chol_solve);

  PolyUOp *logical_solve = poly_solve(ctx, a->uop_logical, b->uop_logical);
  PolyUOp *physical_solve = poly_solve(ctx, a->uop_physical, b->uop_physical);
  PolyTensor *solve = poly_tensor_solve(ctx, a, b);
  ASSERT_NOT_NULL(solve);
  ASSERT_PTR_EQ(solve->uop_logical, logical_solve);
  ASSERT_PTR_EQ(solve->uop_physical, physical_solve);

  int64_t tall_shape[2] = {3, 2};
  int64_t tall_rhs_shape[1] = {3};
  PolyTensor *tall =
      poly_tensor_empty(ctx, POLY_FLOAT32, tall_shape, 2, POLY_DEVICE_CPU);
  PolyTensor *tall_rhs =
      poly_tensor_empty(ctx, POLY_FLOAT32, tall_rhs_shape, 1, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(tall);
  ASSERT_NOT_NULL(tall_rhs);
  PolyUOp *logical_lstsq =
      poly_lstsq(ctx, tall->uop_logical, tall_rhs->uop_logical);
  PolyUOp *physical_lstsq =
      poly_lstsq(ctx, tall->uop_physical, tall_rhs->uop_physical);
  PolyTensor *lstsq = poly_tensor_lstsq(ctx, tall, tall_rhs);
  ASSERT_NOT_NULL(lstsq);
  ASSERT_PTR_EQ(lstsq->uop_logical, logical_lstsq);
  ASSERT_PTR_EQ(lstsq->uop_physical, physical_lstsq);

  float a_data[4] = {4.0f, 2.0f, 2.0f, 5.0f};
  float b_data[2] = {1.0f, 3.0f};
  PolyUOp *a_leaf = base_buf(a->uop_physical);
  PolyUOp *b_leaf = base_buf(b->uop_physical);
  float *a_leaf_data[1] = {a_data};
  PolyUOp *a_leaves[1] = {a_leaf};
  float q_values[4] = {0}, r_values[4] = {0};
  ASSERT_INT_EQ(
      realize_uop(
          ctx, q->uop_physical, poly_buffer_f32(ctx, 4), q_values,
          a_leaves, a_leaf_data, 1
      ),
      0
  );
  ASSERT_INT_EQ(
      realize_uop(
          ctx, r->uop_physical, poly_buffer_f32(ctx, 4), r_values,
          a_leaves, a_leaf_data, 1
      ),
      0
  );
  float expected_q[4] = {
      -0.89442706f, 0.44721359f, -0.44721359f, -0.89442718f
  };
  float expected_r[4] = {
      -4.47213554f, -4.02492189f, 5.4168083e-9f, -3.57770872f
  };
  for (int i = 0; i < 4; i++) {
    ASSERT_FLOAT_EQ(q_values[i], expected_q[i], 2e-5f);
    ASSERT_FLOAT_EQ(r_values[i], expected_r[i], 2e-5f);
  }

  float solve_values[2] = {0};
  PolyUOp *solve_leaves[2] = {a_leaf, b_leaf};
  float *solve_leaf_data[2] = {a_data, b_data};
  ASSERT_INT_EQ(
      realize_uop(
          ctx, solve->uop_physical, poly_buffer_f32(ctx, 2),
          solve_values, solve_leaves, solve_leaf_data, 2
      ),
      0
  );
  ASSERT_FLOAT_EQ(solve_values[0], -0.0625f, 2e-5f);
  ASSERT_FLOAT_EQ(solve_values[1], 0.625f, 2e-5f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, tensor_sort_topk_reject_foreign_context) {
  PolyCtx *owner = poly_ctx_new();
  PolyCtx *foreign = poly_ctx_new();
  ASSERT_NOT_NULL(owner);
  ASSERT_NOT_NULL(foreign);
  PolyTensor *src =
      poly_tensor_empty(owner, POLY_FLOAT32, (int64_t[]){4}, 1, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(src);

  PolyTensor *sort_values = NULL, *sort_indices = NULL;
  PolyTensor *topk_values = NULL, *topk_indices = NULL;
  int sort_rc = poly_tensor_sort(
      foreign, src, 0, 0, &sort_values, &sort_indices
  );
  int topk_rc = poly_tensor_topk(
      foreign, src, 2, 0, 1, 1, &topk_values, &topk_indices
  );

  /* Destroy the contaminated destination first on the pre-fix run: its CSE
   * can contain source pointers into owner. Assertions follow cleanup so the
   * failing evidence does not leak either context. */
  poly_ctx_destroy(foreign);
  poly_ctx_destroy(owner);

  ASSERT_INT_EQ(sort_rc, -1);
  ASSERT_EQ(sort_values, NULL);
  ASSERT_EQ(sort_indices, NULL);
  ASSERT_INT_EQ(topk_rc, -1);
  ASSERT_EQ(topk_values, NULL);
  ASSERT_EQ(topk_indices, NULL);
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
  PolyUOp *store = poly_store_val(ctx, out_buf, sm);
  PolyUOp *sink = poly_sink1(ctx, store);
  PolyTestBufferView bindings[] = {
      POLY_TEST_HOST_VIEW(x, dx),
      POLY_TEST_HOST_VIEW(out_buf, dout),
  };
  ASSERT_INT_EQ(poly_test_realize_buffer_views(ctx, sink, bindings, 2), 0);
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
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, r)[0], 2);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, r)[1], 2);

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

TEST(tensor, sum_and_dot_accumulation_dtype_match_pinned_topology) {
  /* Pinned ReduceMixin.sum and dot cast the product/value to the selected
   * accumulation dtype before REDUCE, and default half sums cast back
   * (mixin/reduce.py:13-44, mixin/__init__.py:451-490). */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  int f32 = poly_dtype_id_by_name("float32");
  ASSERT_TRUE(f32 >= 0);
  int64_t a_shape[] = {2, 3}, b_shape[] = {3, 2}, axis[] = {1};
  PolyTensor *a = poly_tensor_empty(ctx, POLY_FLOAT16, a_shape, 2, POLY_DEVICE_CPU);
  PolyTensor *b = poly_tensor_empty(ctx, POLY_FLOAT16, b_shape, 2, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(a);
  ASSERT_NOT_NULL(b);

  PolyTensor *sum_default = poly_tensor_sum(ctx, a, axis, 1, false);
  PolyUOp *sum_default_root = poly_tensor_uop(sum_default);
  ASSERT_NOT_NULL(sum_default_root);
  ASSERT_INT_EQ(sum_default_root->op, POLY_OP_CAST);
  ASSERT_TRUE(poly_dtype_eq(sum_default_root->dtype, POLY_FLOAT16));
  ASSERT_INT_EQ(sum_default_root->src[0]->op, POLY_OP_RESHAPE);
  ASSERT_INT_EQ(sum_default_root->src[0]->src[0]->op, POLY_OP_REDUCE);
  ASSERT_INT_EQ(sum_default_root->src[0]->src[0]->src[0]->op, POLY_OP_CAST);
  ASSERT_TRUE(poly_dtype_eq(sum_default_root->src[0]->src[0]->src[0]->dtype, POLY_FLOAT32));

  PolyTensor *sum_explicit =
      poly_tensor_sum_dtype_by_id(ctx, a, axis, 1, false, f32);
  PolyUOp *sum_explicit_root = poly_tensor_uop(sum_explicit);
  ASSERT_NOT_NULL(sum_explicit_root);
  ASSERT_INT_EQ(sum_explicit_root->op, POLY_OP_RESHAPE);
  ASSERT_INT_EQ(sum_explicit_root->src[0]->op, POLY_OP_REDUCE);
  ASSERT_INT_EQ(sum_explicit_root->src[0]->src[0]->op, POLY_OP_CAST);
  ASSERT_TRUE(poly_dtype_eq(sum_explicit_root->dtype, POLY_FLOAT32));

  PolyTensor *dot_default = poly_tensor_dot(ctx, a, b);
  PolyUOp *dot_default_root = poly_tensor_uop(dot_default);
  ASSERT_NOT_NULL(dot_default_root);
  ASSERT_INT_EQ(dot_default_root->op, POLY_OP_CAST);
  ASSERT_INT_EQ(dot_default_root->src[0]->op, POLY_OP_RESHAPE);
  ASSERT_INT_EQ(dot_default_root->src[0]->src[0]->op, POLY_OP_REDUCE);
  ASSERT_INT_EQ(dot_default_root->src[0]->src[0]->src[0]->op, POLY_OP_CAST);
  ASSERT_INT_EQ(dot_default_root->src[0]->src[0]->src[0]->src[0]->op, POLY_OP_MUL);

  PolyTensor *dot_explicit = poly_tensor_dot_dtype_by_id(ctx, a, b, f32);
  PolyUOp *dot_explicit_root = poly_tensor_uop(dot_explicit);
  ASSERT_NOT_NULL(dot_explicit_root);
  ASSERT_INT_EQ(dot_explicit_root->op, POLY_OP_RESHAPE);
  ASSERT_INT_EQ(dot_explicit_root->src[0]->op, POLY_OP_REDUCE);
  ASSERT_INT_EQ(dot_explicit_root->src[0]->src[0]->op, POLY_OP_CAST);
  ASSERT_INT_EQ(dot_explicit_root->src[0]->src[0]->src[0]->op, POLY_OP_MUL);
  ASSERT_TRUE(poly_dtype_eq(dot_explicit_root->dtype, POLY_FLOAT32));

  PolyTensor *b_f32 = poly_tensor_empty(ctx, POLY_FLOAT32, b_shape, 2, POLY_DEVICE_CPU);
  PolyTensor *dot_mixed = poly_tensor_dot(ctx, a, b_f32);
  PolyUOp *dot_mixed_root = poly_tensor_uop(dot_mixed);
  ASSERT_NOT_NULL(dot_mixed_root);
  ASSERT_INT_EQ(dot_mixed_root->op, POLY_OP_RESHAPE);
  ASSERT_TRUE(poly_dtype_eq(dot_mixed_root->dtype, POLY_FLOAT32));
  ASSERT_INT_EQ(dot_mixed_root->src[0]->op, POLY_OP_REDUCE);
  PolyUOp *mixed_mul = dot_mixed_root->src[0]->src[0];
  ASSERT_INT_EQ(mixed_mul->op, POLY_OP_MUL);
  ASSERT_TRUE(poly_dtype_eq(mixed_mul->dtype, POLY_FLOAT32));
  ASSERT_INT_EQ(mixed_mul->src[0]->op, POLY_OP_CAST);
  ASSERT_TRUE(poly_dtype_eq(mixed_mul->src[0]->dtype, POLY_FLOAT32));
  ASSERT_FALSE(mixed_mul->src[1]->op == POLY_OP_CAST);

  PolyTensor *v0 =
      poly_tensor_empty(ctx, POLY_FLOAT32, (int64_t[]){3}, 1, POLY_DEVICE_CPU);
  PolyTensor *v1 =
      poly_tensor_empty(ctx, POLY_FLOAT32, (int64_t[]){3}, 1, POLY_DEVICE_CPU);
  PolyTensor *scalar_dot = poly_tensor_dot(ctx, v0, v1);
  ASSERT_NOT_NULL(scalar_dot);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, poly_tensor_uop(scalar_dot)), 0);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(tensor, conv2d_promotion_accumulation_and_bias_match_pinned_topology) {
  /* Pinned conv2d uses ordinary promoted multiplication, ReduceMixin.sum's
   * accumulation/cast-back rule, and ordinary promoted bias addition
   * (mixin/__init__.py:439-449,1493-1507; mixin/reduce.py:19-44). */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  int f32 = poly_dtype_id_by_name("float32");
  ASSERT_TRUE(f32 >= 0);
  int64_t x_shape[] = {1, 1, 3, 3}, w_shape[] = {1, 1, 2, 2}, b_shape[] = {1};
  PolyTensor *x = poly_tensor_empty(ctx, POLY_FLOAT16, x_shape, 4, POLY_DEVICE_CPU);
  PolyTensor *w_half = poly_tensor_empty(ctx, POLY_FLOAT16, w_shape, 4, POLY_DEVICE_CPU);
  PolyTensor *w_float = poly_tensor_empty(ctx, POLY_FLOAT32, w_shape, 4, POLY_DEVICE_CPU);
  PolyTensor *b_half = poly_tensor_empty(ctx, POLY_FLOAT16, b_shape, 1, POLY_DEVICE_CPU);
  PolyTensor *b_float = poly_tensor_empty(ctx, POLY_FLOAT32, b_shape, 1, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(x);
  ASSERT_NOT_NULL(w_half);
  ASSERT_NOT_NULL(w_float);
  ASSERT_NOT_NULL(b_half);
  ASSERT_NOT_NULL(b_float);

  PolyTensor *mixed = poly_tensor_conv2d(
      ctx, x, w_float, b_float, 1, NULL, NULL, NULL, 0);
  PolyTensor *half_default = poly_tensor_conv2d(
      ctx, x, w_half, b_half, 1, NULL, NULL, NULL, 0);
  PolyTensor *half_explicit = poly_tensor_conv2d_dtype_by_id(
      ctx, x, w_half, b_half, 1, NULL, NULL, NULL, 0, f32);
  ASSERT_NOT_NULL(mixed);
  ASSERT_NOT_NULL(half_default);
  ASSERT_NOT_NULL(half_explicit);
  ASSERT_TRUE(poly_dtype_eq(poly_tensor_uop(mixed)->dtype, POLY_FLOAT32));
  ASSERT_TRUE(poly_dtype_eq(poly_tensor_uop(half_default)->dtype, POLY_FLOAT16));
  ASSERT_TRUE(poly_dtype_eq(poly_tensor_uop(half_explicit)->dtype, POLY_FLOAT32));

  PolyUOp *roots[] = {
      poly_tensor_uop(mixed), poly_tensor_uop(half_default), poly_tensor_uop(half_explicit)};
  int expected_casts[] = {1, 2, 2};
  for (int r = 0; r < 3; r++) {
    int n_topo = 0, casts = 0, muls = 0, reduces = 0, adds = 0;
    PolyUOp **topo = poly_toposort_alloc(ctx, roots[r], &n_topo);
    ASSERT_NOT_NULL(topo);
    for (int i = 0; i < n_topo; i++) {
      casts += topo[i]->op == POLY_OP_CAST;
      muls += topo[i]->op == POLY_OP_MUL;
      reduces += topo[i]->op == POLY_OP_REDUCE;
      adds += topo[i]->op == POLY_OP_ADD;
      if (topo[i]->op == POLY_OP_REDUCE)
        ASSERT_TRUE(poly_dtype_eq(topo[i]->dtype, POLY_FLOAT32));
      if (topo[i]->op == POLY_OP_MUL)
        ASSERT_TRUE(poly_dtype_eq(topo[i]->dtype, r == 0 ? POLY_FLOAT32 : POLY_FLOAT16));
    }
    ASSERT_INT_EQ(casts, expected_casts[r]);
    ASSERT_INT_EQ(muls, 1);
    ASSERT_INT_EQ(reduces, 1);
    ASSERT_INT_EQ(adds, 1);
    free(topo);
  }

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, dot_singleton_batch_omits_noop_weight_expand) {
  /* Pinned Tensor.dot reshapes a (K,N) weight to (1,N,K), then
   * _broadcast_to returns it unchanged for a (1,K) input. Only the input
   * requires EXPAND from (1,1,K) to (1,N,K). */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = make_buf(ctx, (int64_t[]){1, 12}, 2);
  PolyUOp *w = make_buf(ctx, (int64_t[]){12, 10}, 2);
  PolyUOp *r = poly_dot(ctx, x, w);
  ASSERT_NOT_NULL(r);

  int n_topo = 0, n_expand = 0;
  PolyUOp **topo = poly_toposort_alloc(ctx, r, &n_topo);
  ASSERT_NOT_NULL(topo);
  for (int i = 0; i < n_topo; i++)
    n_expand += topo[i]->op == POLY_OP_EXPAND;
  ASSERT_INT_EQ(n_expand, 1);
  poly_toposort_free(topo);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, qr_e2e_matches_tinygrad_probe) {
  PolyCtx *ctx = poly_ctx_new();

  const int64_t shape_square[] = {2, 2};
  const int64_t q_square[] = {2, 2};
  float data_square[] = {1, 2, 3, 4};

  const int64_t shape_tall[] = {3, 2};
  const int64_t q_tall[] = {3, 3};
  float data_tall[] = {1, 2, 3, 4, 5, 6};

  const int64_t shape_wide[] = {2, 3};
  const int64_t q_wide[] = {2, 2};
  float data_wide[] = {1, 2, 3, 4, 5, 6};

  const int64_t shape_zero[] = {2, 2};
  const int64_t q_zero[] = {2, 2};
  float data_zero[] = {0, 1, 0, 2};

  const int64_t shape_batched_square[] = {2, 2, 2};
  const int64_t q_batched_square[] = {2, 2, 2};
  float data_batched_square[] = {1, 2, 3, 4, 2, 0, 0, 2};

  const int64_t shape_batched_tall[] = {2, 3, 2};
  const int64_t q_batched_tall[] = {2, 3, 3};
  float data_batched_tall[] = {1, 2, 3, 4, 5, 6, 2, 1, 0, 3, 4, 5};

  const int64_t shape_batched_wide[] = {2, 2, 3};
  const int64_t q_batched_wide[] = {2, 2, 2};
  float data_batched_wide[] = {1, 2, 3, 4, 5, 6, 2, 1, 0, 0, 3, 4};

  struct {
    const int64_t *shape;
    int ndim;
    const int64_t *q_shape;
    float *data;
    int64_t numel;
  } cases[] = {
      {shape_square, 2, q_square, data_square, 4},
      {shape_tall, 2, q_tall, data_tall, 6},
      {shape_wide, 2, q_wide, data_wide, 6},
      {shape_zero, 2, q_zero, data_zero, 4},
      {shape_batched_square, 3, q_batched_square, data_batched_square, 8},
      {shape_batched_tall, 3, q_batched_tall, data_batched_tall, 12},
      {shape_batched_wide, 3, q_batched_wide, data_batched_wide, 12},
  };

  for (int c = 0; c < (int)(sizeof(cases) / sizeof(cases[0])); c++) {
    PolyUOp *a = make_buf(ctx, cases[c].shape, cases[c].ndim);
    PolyUOp *q = NULL, *r = NULL;
    ASSERT_INT_EQ(poly_qr(ctx, a, &q, &r), 0);
    ASSERT_NOT_NULL(q);
    ASSERT_NOT_NULL(r);
    ASSERT_INT_EQ(poly_uop_ndim(ctx, q), cases[c].ndim);
    ASSERT_INT_EQ(poly_uop_ndim(ctx, r), cases[c].ndim);
    for (int i = 0; i < cases[c].ndim; i++) {
      ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, q)[i], cases[c].q_shape[i]);
      ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, r)[i], cases[c].shape[i]);
    }

    PolyUOp *recon = poly_dot(ctx, q, r);
    PolyUOp *out_buf = poly_buffer_f32(ctx, cases[c].numel);
    float got[32] = {0};
    PolyUOp *leaves[] = {base_buf(a)};
    float *ld[] = {cases[c].data};
    ASSERT_INT_EQ(realize_uop(ctx, recon, out_buf, got, leaves, ld, 1), 0);
    for (int64_t i = 0; i < cases[c].numel; i++) {
      ASSERT_TRUE(isfinite(got[i]));
      ASSERT_FLOAT_EQ(got[i], cases[c].data[i], 2e-3f);
    }
  }

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, qr_reduced_and_r_modes_match_reference_shapes) {
  PolyCtx *ctx = poly_ctx_new();

  const int64_t shape_tall[] = {3, 2};
  const int64_t q_tall[] = {3, 2};
  const int64_t r_tall[] = {2, 2};
  float data_tall[] = {1, 2, 3, 4, 5, 6};

  const int64_t shape_wide[] = {2, 3};
  const int64_t q_wide[] = {2, 2};
  const int64_t r_wide[] = {2, 3};
  float data_wide[] = {1, 2, 3, 4, 5, 6};

  const int64_t shape_batched_tall[] = {2, 3, 2};
  const int64_t q_batched_tall[] = {2, 3, 2};
  const int64_t r_batched_tall[] = {2, 2, 2};
  float data_batched_tall[] = {1, 2, 3, 4, 5, 6, 2, 1, 0, 3, 4, 5};

  struct {
    const int64_t *shape;
    int ndim;
    const int64_t *q_shape;
    const int64_t *r_shape;
    float *data;
    int64_t numel;
  } cases[] = {
      {shape_tall, 2, q_tall, r_tall, data_tall, 6},
      {shape_wide, 2, q_wide, r_wide, data_wide, 6},
      {shape_batched_tall, 3, q_batched_tall, r_batched_tall, data_batched_tall, 12},
  };

  for (int c = 0; c < (int)(sizeof(cases) / sizeof(cases[0])); c++) {
    PolyUOp *a = make_buf(ctx, cases[c].shape, cases[c].ndim);
    PolyUOp *q = NULL, *r = NULL;
    ASSERT_INT_EQ(poly_qr_ex(ctx, a, POLY_QR_REDUCED, &q, &r), 0);
    ASSERT_NOT_NULL(q);
    ASSERT_NOT_NULL(r);
    for (int i = 0; i < cases[c].ndim; i++) {
      ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, q)[i], cases[c].q_shape[i]);
      ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, r)[i], cases[c].r_shape[i]);
    }

    PolyUOp *recon = poly_dot(ctx, q, r);
    PolyUOp *out_buf = poly_buffer_f32(ctx, cases[c].numel);
    float got[32] = {0};
    PolyUOp *leaves[] = {base_buf(a)};
    float *ld[] = {cases[c].data};
    ASSERT_INT_EQ(realize_uop(ctx, recon, out_buf, got, leaves, ld, 1), 0);
    for (int64_t i = 0; i < cases[c].numel; i++) {
      ASSERT_TRUE(isfinite(got[i]));
      ASSERT_FLOAT_EQ(got[i], cases[c].data[i], 2e-3f);
    }

    PolyUOp *q_r_only = (PolyUOp *)(uintptr_t)1;
    PolyUOp *r_only = NULL;
    ASSERT_INT_EQ(poly_qr_ex(ctx, a, POLY_QR_R_ONLY, &q_r_only, &r_only), 0);
    ASSERT_TRUE(q_r_only == NULL);
    ASSERT_NOT_NULL(r_only);
    for (int i = 0; i < cases[c].ndim; i++) {
      ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, r_only)[i], cases[c].r_shape[i]);
    }
  }

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, triangular_solve_matches_numpy_torch_probe) {
  PolyCtx *ctx = poly_ctx_new();

  const int64_t a_shape[] = {3, 3};
  const int64_t b_vec_shape[] = {3};
  const int64_t b_mat_shape[] = {3, 2};
  const int64_t a_batch_shape[] = {2, 3, 3};
  const int64_t b_batch_shape[] = {2, 3, 2};

  float lower[] = {2, 0, 0, 1, 3, 0, -2, 0.5f, 4};
  float upper[] = {2, -1, 0.5f, 0, 3, 2, 0, 0, 4};
  float lower_unit[] = {5, 0, 0, 1, 7, 0, -2, 0.5f, 9};
  float b_vec[] = {2, 7, 9};
  float b_mat[] = {2, 1, 7, 2, 9, 3};
  float lower_batch[] = {
      2, 0, 0, 1, 3, 0, -2, 0.5f, 4,
      3, 0, 0, 1, 4, 0, -2, 0.5f, 5,
  };
  float b_batch[] = {
      2, 1, 7, 2, 9, 3,
      3, 2, 8, 3, 10, 4,
  };

  float expect_lower_vec[] = {1, 2, 2.5f};
  float expect_lower_mat[] = {1, 0.5f, 2, 0.5f, 2.5f, 0.9375f};
  float expect_upper_mat[] = {
      0.8541666865f, 0.3958333433f, 0.8333333135f, 0.1666666716f, 2.25f, 0.75f};
  float expect_lower_trans[] = {
      2.2708332539f, 0.9791666865f, 1.9583333731f, 0.5416666865f, 2.25f, 0.75f};
  float expect_upper_trans[] = {1, 0.5f, 2.6666667461f, 0.8333333135f, 0.7916666865f, 0.2708333433f};
  float expect_lower_unit[] = {2, 1, 5, 1, 10.5f, 4.5f};
  float expect_batch[] = {
      1, 0.5f, 2, 0.5f, 2.5f, 0.9375f,
      1, 0.6666666865f, 1.75f, 0.5833333135f, 2.2249999046f, 1.0083333254f,
  };

  struct {
    const int64_t *a_shape;
    int a_ndim;
    float *a_data;
    const int64_t *b_shape;
    int b_ndim;
    float *b_data;
    int upper;
    int transpose_a;
    int unit_diagonal;
    float *expected;
    int64_t n_out;
  } cases[] = {
      {a_shape, 2, lower, b_vec_shape, 1, b_vec, 0, 0, 0, expect_lower_vec, 3},
      {a_shape, 2, lower, b_mat_shape, 2, b_mat, 0, 0, 0, expect_lower_mat, 6},
      {a_shape, 2, upper, b_mat_shape, 2, b_mat, 1, 0, 0, expect_upper_mat, 6},
      {a_shape, 2, lower, b_mat_shape, 2, b_mat, 0, 1, 0, expect_lower_trans, 6},
      {a_shape, 2, upper, b_mat_shape, 2, b_mat, 1, 1, 0, expect_upper_trans, 6},
      {a_shape, 2, lower_unit, b_mat_shape, 2, b_mat, 0, 0, 1, expect_lower_unit, 6},
      {a_batch_shape, 3, lower_batch, b_batch_shape, 3, b_batch, 0, 0, 0, expect_batch, 12},
  };

  for (int c = 0; c < (int)(sizeof(cases) / sizeof(cases[0])); c++) {
    PolyUOp *a = make_buf(ctx, cases[c].a_shape, cases[c].a_ndim);
    PolyUOp *b = make_buf(ctx, cases[c].b_shape, cases[c].b_ndim);
    PolyUOp *x = poly_triangular_solve(
        ctx, a, b, cases[c].upper, cases[c].transpose_a, cases[c].unit_diagonal
    );
    ASSERT_NOT_NULL(x);
    ASSERT_INT_EQ(poly_uop_ndim(ctx, x), cases[c].b_ndim);
    for (int i = 0; i < cases[c].b_ndim; i++)
      ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, x)[i], cases[c].b_shape[i]);

    PolyUOp *out_buf = poly_buffer_f32(ctx, cases[c].n_out);
    float got[32] = {0};
    PolyUOp *leaves[] = {base_buf(a), base_buf(b)};
    float *ld[] = {cases[c].a_data, cases[c].b_data};
    ASSERT_INT_EQ(realize_uop(ctx, x, out_buf, got, leaves, ld, 2), 0);
    for (int64_t i = 0; i < cases[c].n_out; i++) {
      ASSERT_TRUE(isfinite(got[i]));
      ASSERT_FLOAT_EQ(got[i], cases[c].expected[i], 2e-4f);
    }
  }

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, cholesky_matches_numpy_torch_probe) {
  PolyCtx *ctx = poly_ctx_new();

  const int64_t shape1[] = {1, 1};
  const int64_t shape2[] = {2, 2};
  const int64_t shape3[] = {3, 3};
  const int64_t shape4[] = {4, 4};
  const int64_t shape_batch[] = {2, 2, 2};

  float a1[] = {4};
  float e1[] = {2};
  float a2[] = {4, 2, 2, 5};
  float e2[] = {2, 0, 1, 2};
  float e2_upper[] = {2, 1, 0, 2};
  float a3[] = {6, 2, 1, 2, 5, 2, 1, 2, 4};
  float e3[] = {
      2.4494898319f, 0, 0,
      0.8164966106f, 2.0816659927f, 0,
      0.4082483053f, 0.8006407619f, 1.7867029905f,
  };
  float a4_eye[] = {
      4, 0, 0, 0,
      0, 4, 0, 0,
      0, 0, 4, 0,
      0, 0, 0, 4,
  };
  float e4_eye[] = {
      2, 0, 0, 0,
      0, 2, 0, 0,
      0, 0, 2, 0,
      0, 0, 0, 2,
  };
  float abat[] = {4, 2, 2, 5, 9, 3, 3, 2};
  float ebat[] = {2, 0, 1, 2, 3, 0, 1, 1};

  struct {
    const int64_t *shape;
    int ndim;
    float *data;
    int upper;
    float *expected;
    int64_t n_out;
  } cases[] = {
      {shape1, 2, a1, 0, e1, 1},
      {shape2, 2, a2, 0, e2, 4},
      {shape2, 2, a2, 1, e2_upper, 4},
      {shape3, 2, a3, 0, e3, 9},
      {shape4, 2, a4_eye, 0, e4_eye, 16},
      {shape_batch, 3, abat, 0, ebat, 8},
  };

  for (int c = 0; c < (int)(sizeof(cases) / sizeof(cases[0])); c++) {
    PolyUOp *a = make_buf(ctx, cases[c].shape, cases[c].ndim);
    PolyUOp *l = poly_cholesky(ctx, a, cases[c].upper);
    ASSERT_NOT_NULL(l);
    ASSERT_INT_EQ(poly_uop_ndim(ctx, l), cases[c].ndim);
    for (int i = 0; i < cases[c].ndim; i++)
      ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, l)[i], cases[c].shape[i]);

    PolyUOp *out_buf = poly_buffer_f32(ctx, cases[c].n_out);
    float got[32] = {0};
    PolyUOp *leaves[] = {base_buf(a)};
    float *ld[] = {cases[c].data};
    ASSERT_INT_EQ(realize_uop(ctx, l, out_buf, got, leaves, ld, 1), 0);
    for (int64_t i = 0; i < cases[c].n_out; i++) {
      ASSERT_TRUE(isfinite(got[i]));
      ASSERT_FLOAT_EQ(got[i], cases[c].expected[i], 2e-4f);
    }
  }

  PolyUOp *bad = make_buf(ctx, shape2, 2);
  float dbad[] = {1, 2, 2, 1};
  PolyUOp *bad_l = poly_cholesky(ctx, bad, 0);
  ASSERT_NOT_NULL(bad_l);
  PolyUOp *out_buf = poly_buffer_f32(ctx, 4);
  float got_bad[4] = {0};
  PolyUOp *bad_leaves[] = {base_buf(bad)};
  float *bad_ld[] = {dbad};
  ASSERT_INT_EQ(realize_uop(ctx, bad_l, out_buf, got_bad, bad_leaves, bad_ld, 1), 0);
  bool has_nonfinite = false;
  for (int i = 0; i < 4; i++)
    if (!isfinite(got_bad[i])) has_nonfinite = true;
  ASSERT_TRUE(has_nonfinite);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, cholesky_solve_and_solve_match_numpy_torch_probe) {
  PolyCtx *ctx = poly_ctx_new();

  const int64_t shape2[] = {2, 2};
  const int64_t vec_shape[] = {2};
  const int64_t batch_shape[] = {2, 2, 2};

  float spd[] = {4, 2, 2, 5};
  float rhs_mat[] = {1, 2, 3, 4};
  float b_vec[] = {1, 4};
  float expect_cholsolve[] = {-0.0625f, 0.125f, 0.625f, 0.75f};
  float spd_batch[] = {4, 2, 2, 5, 9, 3, 3, 2};
  float chol_rhs_batch_vec[] = {1, 3, 2, 4};
  float expect_cholsolve_batch_vec[] = {-0.0625f, 0.625f, -0.8888889f, 3.3333333f};
  float expect_cholsolve_broadcast_vec[] = {-0.1875f, 0.875f, -1.1111112f, 3.6666667f};

  for (int upper = 0; upper <= 1; upper++) {
    PolyUOp *a = make_buf(ctx, shape2, 2);
    PolyUOp *b = make_buf(ctx, shape2, 2);
    PolyUOp *factor = poly_cholesky(ctx, a, upper);
    ASSERT_NOT_NULL(factor);
    PolyUOp *x = poly_cholesky_solve(ctx, factor, b, upper);
    ASSERT_NOT_NULL(x);
    PolyUOp *out_buf = poly_buffer_f32(ctx, 4);
    float got[4] = {0};
    PolyUOp *leaves[] = {base_buf(a), base_buf(b)};
    float *ld[] = {spd, rhs_mat};
    ASSERT_INT_EQ(realize_uop(ctx, x, out_buf, got, leaves, ld, 2), 0);
    for (int i = 0; i < 4; i++)
      ASSERT_FLOAT_EQ(got[i], expect_cholsolve[i], 2e-4f);
  }

  for (int upper = 0; upper <= 1; upper++) {
    PolyUOp *a = make_buf(ctx, batch_shape, 3);
    PolyUOp *b = make_buf(ctx, shape2, 2);
    PolyUOp *factor = poly_cholesky(ctx, a, upper);
    ASSERT_NOT_NULL(factor);
    PolyUOp *x = poly_cholesky_solve(ctx, factor, b, upper);
    ASSERT_NOT_NULL(x);
    float got[4] = {0};
    PolyUOp *leaves[] = {base_buf(a), base_buf(b)};
    float *ld[] = {spd_batch, chol_rhs_batch_vec};
    ASSERT_INT_EQ(realize_uop(ctx, x, poly_buffer_f32(ctx, 4), got, leaves, ld, 2), 0);
    for (int i = 0; i < 4; i++)
      ASSERT_FLOAT_EQ(got[i], expect_cholsolve_batch_vec[i], 4e-4f);
  }

  for (int upper = 0; upper <= 1; upper++) {
    PolyUOp *a = make_buf(ctx, batch_shape, 3);
    PolyUOp *b = make_buf(ctx, vec_shape, 1);
    PolyUOp *factor = poly_cholesky(ctx, a, upper);
    ASSERT_NOT_NULL(factor);
    PolyUOp *x = poly_cholesky_solve(ctx, factor, b, upper);
    ASSERT_NOT_NULL(x);
    float got[4] = {0};
    PolyUOp *leaves[] = {base_buf(a), base_buf(b)};
    float *ld[] = {spd_batch, b_vec};
    ASSERT_INT_EQ(realize_uop(ctx, x, poly_buffer_f32(ctx, 4), got, leaves, ld, 2), 0);
    for (int i = 0; i < 4; i++)
      ASSERT_FLOAT_EQ(got[i], expect_cholsolve_broadcast_vec[i], 5e-4f);
  }

  float a2[] = {2, 1, 1, 3};
  float expect_vec[] = {-0.2f, 1.4f};
  PolyUOp *a_vec = make_buf(ctx, shape2, 2);
  PolyUOp *b_v = make_buf(ctx, vec_shape, 1);
  PolyUOp *x_vec = poly_solve(ctx, a_vec, b_v);
  ASSERT_NOT_NULL(x_vec);
  float got_vec[2] = {0};
  PolyUOp *vec_leaves[] = {base_buf(a_vec), base_buf(b_v)};
  float *vec_ld[] = {a2, b_vec};
  ASSERT_INT_EQ(realize_uop(ctx, x_vec, poly_buffer_f32(ctx, 2), got_vec, vec_leaves, vec_ld, 2), 0);
  for (int i = 0; i < 2; i++)
    ASSERT_FLOAT_EQ(got_vec[i], expect_vec[i], 3e-4f);

  float expect_mat[] = {0, 0.4f, 1, 1.2f};
  PolyUOp *a_mat = make_buf(ctx, shape2, 2);
  PolyUOp *b_m = make_buf(ctx, shape2, 2);
  PolyUOp *x_mat = poly_solve(ctx, a_mat, b_m);
  ASSERT_NOT_NULL(x_mat);
  float got_mat[4] = {0};
  PolyUOp *mat_leaves[] = {base_buf(a_mat), base_buf(b_m)};
  float *mat_ld[] = {a2, rhs_mat};
  ASSERT_INT_EQ(realize_uop(ctx, x_mat, poly_buffer_f32(ctx, 4), got_mat, mat_leaves, mat_ld, 2), 0);
  for (int i = 0; i < 4; i++)
    ASSERT_FLOAT_EQ(got_mat[i], expect_mat[i], 3e-4f);

  float pivot_a[] = {0, 2, 1, 3};
  float pivot_b_vec[] = {4, 5};
  float expect_pivot_vec[] = {-1, 2};
  PolyUOp *pa_vec = make_buf(ctx, shape2, 2);
  PolyUOp *pb_vec = make_buf(ctx, vec_shape, 1);
  PolyUOp *px_vec = poly_solve(ctx, pa_vec, pb_vec);
  ASSERT_NOT_NULL(px_vec);
  float got_pivot_vec[2] = {0};
  PolyUOp *pivot_vec_leaves[] = {base_buf(pa_vec), base_buf(pb_vec)};
  float *pivot_vec_ld[] = {pivot_a, pivot_b_vec};
  ASSERT_INT_EQ(
      realize_uop(ctx, px_vec, poly_buffer_f32(ctx, 2), got_pivot_vec, pivot_vec_leaves, pivot_vec_ld, 2),
      0
  );
  for (int i = 0; i < 2; i++)
    ASSERT_FLOAT_EQ(got_pivot_vec[i], expect_pivot_vec[i], 3e-4f);

  float pivot_b_mat[] = {4, 1, 5, 2};
  float expect_pivot_mat[] = {-1, 0.5f, 2, 0.5f};
  PolyUOp *pa_mat = make_buf(ctx, shape2, 2);
  PolyUOp *pb_mat = make_buf(ctx, shape2, 2);
  PolyUOp *px_mat = poly_solve(ctx, pa_mat, pb_mat);
  ASSERT_NOT_NULL(px_mat);
  float got_pivot_mat[4] = {0};
  PolyUOp *pivot_mat_leaves[] = {base_buf(pa_mat), base_buf(pb_mat)};
  float *pivot_mat_ld[] = {pivot_a, pivot_b_mat};
  ASSERT_INT_EQ(
      realize_uop(ctx, px_mat, poly_buffer_f32(ctx, 4), got_pivot_mat, pivot_mat_leaves, pivot_mat_ld, 2),
      0
  );
  for (int i = 0; i < 4; i++)
    ASSERT_FLOAT_EQ(got_pivot_mat[i], expect_pivot_mat[i], 3e-4f);

  int f32_id = poly_dtype_id_by_name("float32");
  ASSERT_TRUE(f32_id >= 0);
  float public_a[] = {2, 1, 1, 3};
  float public_b[] = {1, 4};
  int64_t public_a_shape[] = {2, 2};
  int64_t public_b_shape[] = {2};
  PolyTensor *public_ah = poly_tensor_from_host_by_id(
      ctx, public_a, sizeof(public_a), f32_id, public_a_shape, 2
  );
  PolyTensor *public_bh = poly_tensor_from_host_by_id(
      ctx, public_b, sizeof(public_b), f32_id, public_b_shape, 1
  );
  PolyTensor *public_at = poly_tensor_to_device(ctx, public_ah, POLY_DEVICE_CPU);
  PolyTensor *public_bt = poly_tensor_to_device(ctx, public_bh, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(public_ah);
  ASSERT_NOT_NULL(public_bh);
  ASSERT_NOT_NULL(public_at);
  ASSERT_NOT_NULL(public_bt);
  PolyTensor *public_xt = poly_tensor_solve(ctx, public_at, public_bt);
  ASSERT_NOT_NULL(public_xt);
  PolyTensor *public_out = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &public_xt, 1, &public_out), 0);
  ASSERT_NOT_NULL(public_out);
  const PolyUOp *public_buf = poly_uop_get_buffer_identity(poly_tensor_uop(public_out));
  ASSERT_NOT_NULL(public_buf);
  float public_got[2] = {0};
  ASSERT_INT_EQ(poly_buffer_read(ctx, (PolyUOp *)public_buf, public_got, sizeof(public_got)), 0);
  ASSERT_FLOAT_EQ(public_got[0], -0.2f, 3e-4f);
  ASSERT_FLOAT_EQ(public_got[1], 1.4f, 3e-4f);

  float a_batch[] = {2, 1, 1, 3, 3, 1, 1, 4};
  float b_batch[] = {1, 2, 3, 4, 2, 3, 4, 5};
  float b_batch_vec[] = {1, 4, 2, 5};
  float expect_batch[] = {
      0, 0.4f, 1, 1.2f,
      0.3636363745f, 0.6363636255f, 0.9090909362f, 1.0909091234f,
  };
  float expect_batch_vec[] = {-0.2f, 1.4f, 0.27272728f, 1.1818182f};
  PolyUOp *ab = make_buf(ctx, batch_shape, 3);
  PolyUOp *bb = make_buf(ctx, batch_shape, 3);
  PolyUOp *xb = poly_solve(ctx, ab, bb);
  ASSERT_NOT_NULL(xb);
  float got_batch[8] = {0};
  PolyUOp *batch_leaves[] = {base_buf(ab), base_buf(bb)};
  float *batch_ld[] = {a_batch, b_batch};
  ASSERT_INT_EQ(realize_uop(ctx, xb, poly_buffer_f32(ctx, 8), got_batch, batch_leaves, batch_ld, 2), 0);
  for (int i = 0; i < 8; i++)
    ASSERT_FLOAT_EQ(got_batch[i], expect_batch[i], 3e-4f);

  PolyUOp *bbv = make_buf(ctx, shape2, 2);
  PolyUOp *xbv = poly_solve(ctx, ab, bbv);
  ASSERT_NOT_NULL(xbv);
  float got_batch_vec[4] = {0};
  PolyUOp *batch_vec_leaves[] = {base_buf(ab), base_buf(bbv)};
  float *batch_vec_ld[] = {a_batch, b_batch_vec};
  ASSERT_INT_EQ(realize_uop(ctx, xbv, poly_buffer_f32(ctx, 4), got_batch_vec, batch_vec_leaves, batch_vec_ld, 2), 0);
  for (int i = 0; i < 4; i++)
    ASSERT_FLOAT_EQ(got_batch_vec[i], expect_batch_vec[i], 4e-4f);

  float pivot_batch_a[] = {0, 2, 1, 3, 3, 1, 0, 2};
  float pivot_batch_bv[] = {4, 5, 7, 4};
  float expect_pivot_batch_vec[] = {-1, 2, 1.6666667f, 2};
  PolyUOp *pab = make_buf(ctx, batch_shape, 3);
  PolyUOp *pbbv = make_buf(ctx, shape2, 2);
  PolyUOp *pxbv = poly_solve(ctx, pab, pbbv);
  ASSERT_NOT_NULL(pxbv);
  float got_pivot_batch_vec[4] = {0};
  PolyUOp *pivot_batch_leaves[] = {base_buf(pab), base_buf(pbbv)};
  float *pivot_batch_ld[] = {pivot_batch_a, pivot_batch_bv};
  ASSERT_INT_EQ(
      realize_uop(ctx, pxbv, poly_buffer_f32(ctx, 4), got_pivot_batch_vec, pivot_batch_leaves, pivot_batch_ld, 2),
      0
  );
  for (int i = 0; i < 4; i++)
    ASSERT_FLOAT_EQ(got_pivot_batch_vec[i], expect_pivot_batch_vec[i], 5e-4f);

  PolyUOp *bbv_single = make_buf(ctx, vec_shape, 1);
  PolyUOp *xbv_single = poly_solve(ctx, ab, bbv_single);
  ASSERT_NOT_NULL(xbv_single);
  float expect_broadcast_vec[] = {-0.2f, 1.4f, 0.0f, 1.0f};
  float got_broadcast_vec[4] = {0};
  PolyUOp *broadcast_vec_leaves[] = {base_buf(ab), base_buf(bbv_single)};
  float *broadcast_vec_ld[] = {a_batch, b_vec};
  ASSERT_INT_EQ(
      realize_uop(ctx, xbv_single, poly_buffer_f32(ctx, 4), got_broadcast_vec, broadcast_vec_leaves, broadcast_vec_ld, 2),
      0
  );
  for (int i = 0; i < 4; i++)
    ASSERT_FLOAT_EQ(got_broadcast_vec[i], expect_broadcast_vec[i], 5e-4f);

  const int64_t singleton_batch_shape[] = {1, 2, 2};
  PolyUOp *bb_single_batch = make_buf(ctx, singleton_batch_shape, 3);
  PolyUOp *xb_single_batch = poly_solve(ctx, ab, bb_single_batch);
  ASSERT_NOT_NULL(xb_single_batch);
  float expect_single_batch[] = {
      0, 0.4f, 1, 1.2f,
      0.09090909f, 0.36363637f, 0.7272727f, 0.9090909f,
  };
  float got_single_batch[8] = {0};
  PolyUOp *single_batch_leaves[] = {base_buf(ab), base_buf(bb_single_batch)};
  float *single_batch_ld[] = {a_batch, rhs_mat};
  ASSERT_INT_EQ(
      realize_uop(ctx, xb_single_batch, poly_buffer_f32(ctx, 8), got_single_batch, single_batch_leaves, single_batch_ld, 2),
      0
  );
  for (int i = 0; i < 8; i++)
    ASSERT_FLOAT_EQ(got_single_batch[i], expect_single_batch[i], 5e-4f);

  float tri_batch[] = {2, 0, 1, 3, 3, 0, 1, 4};
  float expect_tri_broadcast_vec[] = {0.5f, 1.1666667f, 0.33333334f, 0.9166667f};
  PolyUOp *tri_a = make_buf(ctx, batch_shape, 3);
  PolyUOp *tri_b = make_buf(ctx, vec_shape, 1);
  PolyUOp *tri_x = poly_triangular_solve(ctx, tri_a, tri_b, 0, 0, 0);
  ASSERT_NOT_NULL(tri_x);
  float got_tri_broadcast_vec[4] = {0};
  PolyUOp *tri_leaves[] = {base_buf(tri_a), base_buf(tri_b)};
  float *tri_ld[] = {tri_batch, b_vec};
  ASSERT_INT_EQ(
      realize_uop(ctx, tri_x, poly_buffer_f32(ctx, 4), got_tri_broadcast_vec, tri_leaves, tri_ld, 2),
      0
  );
  for (int i = 0; i < 4; i++)
    ASSERT_FLOAT_EQ(got_tri_broadcast_vec[i], expect_tri_broadcast_vec[i], 5e-4f);

  const int64_t tall_shape[] = {3, 2};
  const int64_t tall_vec_shape[] = {3};
  const int64_t tall_mat_shape[] = {3, 2};
  const int64_t tall_batch_shape[] = {2, 3, 2};
  const int64_t tall_batch_rhs_shape[] = {2, 3, 2};
  const int64_t tall_batch_vec_shape[] = {2, 3};
  float tall_a[] = {1, 0, 1, 1, 1, 2};
  float tall_b_vec[] = {1, 2, 2.5f};
  float tall_b_mat[] = {1, 0.5f, 2, 1, 2.5f, 1.5f};
  float tall_a_batch[] = {
      1, 0, 1, 1, 1, 2,
      1, 0, 1, 1.5f, 1, 3,
  };
  float tall_b_batch[] = {
      1, 0.5f, 2, 1, 2.5f, 1.5f,
      1.25f, 0.75f, 2.25f, 1.25f, 2.75f, 1.75f,
  };
  float tall_b_batch_vec[] = {1, 2, 2.5f, 1.25f, 2.25f, 2.75f};
  float expect_lstsq_vec[] = {1.0833334f, 0.75f};
  float expect_lstsq_mat[] = {1.0833334f, 0.5f, 0.75f, 0.5f};
  float expect_lstsq_batch[] = {
      1.0833334f, 0.5f, 0.75f, 0.5f,
      1.3333334f, 0.75f, 0.5f, 0.33333334f,
  };
  float expect_lstsq_batch_vec[] = {1.0833334f, 0.75f, 1.3333334f, 0.5f};

  PolyUOp *la = make_buf(ctx, tall_shape, 2);
  PolyUOp *lbv = make_buf(ctx, tall_vec_shape, 1);
  PolyUOp *lxv = poly_lstsq(ctx, la, lbv);
  ASSERT_NOT_NULL(lxv);
  float got_lxv[2] = {0};
  PolyUOp *lvec_leaves[] = {base_buf(la), base_buf(lbv)};
  float *lvec_ld[] = {tall_a, tall_b_vec};
  ASSERT_INT_EQ(realize_uop(ctx, lxv, poly_buffer_f32(ctx, 2), got_lxv, lvec_leaves, lvec_ld, 2), 0);
  for (int i = 0; i < 2; i++)
    ASSERT_FLOAT_EQ(got_lxv[i], expect_lstsq_vec[i], 4e-4f);

  PolyUOp *lB = make_buf(ctx, tall_mat_shape, 2);
  PolyUOp *lxm = poly_lstsq(ctx, la, lB);
  ASSERT_NOT_NULL(lxm);
  float got_lxm[4] = {0};
  PolyUOp *lmat_leaves[] = {base_buf(la), base_buf(lB)};
  float *lmat_ld[] = {tall_a, tall_b_mat};
  ASSERT_INT_EQ(realize_uop(ctx, lxm, poly_buffer_f32(ctx, 4), got_lxm, lmat_leaves, lmat_ld, 2), 0);
  for (int i = 0; i < 4; i++)
    ASSERT_FLOAT_EQ(got_lxm[i], expect_lstsq_mat[i], 4e-4f);

  PolyUOp *lab = make_buf(ctx, tall_batch_shape, 3);
  PolyUOp *lbb = make_buf(ctx, tall_batch_rhs_shape, 3);
  PolyUOp *lxb = poly_lstsq(ctx, lab, lbb);
  ASSERT_NOT_NULL(lxb);
  float got_lxb[8] = {0};
  PolyUOp *lbatch_leaves[] = {base_buf(lab), base_buf(lbb)};
  float *lbatch_ld[] = {tall_a_batch, tall_b_batch};
  ASSERT_INT_EQ(realize_uop(ctx, lxb, poly_buffer_f32(ctx, 8), got_lxb, lbatch_leaves, lbatch_ld, 2), 0);
  for (int i = 0; i < 8; i++)
    ASSERT_FLOAT_EQ(got_lxb[i], expect_lstsq_batch[i], 5e-4f);

  PolyUOp *lbbv = make_buf(ctx, tall_batch_vec_shape, 2);
  PolyUOp *lxbv = poly_lstsq(ctx, lab, lbbv);
  ASSERT_NOT_NULL(lxbv);
  float got_lxbv[4] = {0};
  PolyUOp *lbatch_vec_leaves[] = {base_buf(lab), base_buf(lbbv)};
  float *lbatch_vec_ld[] = {tall_a_batch, tall_b_batch_vec};
  ASSERT_INT_EQ(realize_uop(ctx, lxbv, poly_buffer_f32(ctx, 4), got_lxbv, lbatch_vec_leaves, lbatch_vec_ld, 2), 0);
  for (int i = 0; i < 4; i++)
    ASSERT_FLOAT_EQ(got_lxbv[i], expect_lstsq_batch_vec[i], 5e-4f);

  PolyUOp *lbbv_single = make_buf(ctx, tall_vec_shape, 1);
  PolyUOp *lxbv_single = poly_lstsq(ctx, lab, lbbv_single);
  ASSERT_NOT_NULL(lxbv_single);
  float expect_lstsq_broadcast_vec[] = {1.0833334f, 0.75f, 1.0833334f, 0.5f};
  float got_lxbv_single[4] = {0};
  PolyUOp *lbatch_broadcast_leaves[] = {base_buf(lab), base_buf(lbbv_single)};
  float *lbatch_broadcast_ld[] = {tall_a_batch, tall_b_vec};
  ASSERT_INT_EQ(
      realize_uop(ctx, lxbv_single, poly_buffer_f32(ctx, 4), got_lxbv_single, lbatch_broadcast_leaves, lbatch_broadcast_ld, 2),
      0
  );
  for (int i = 0; i < 4; i++)
    ASSERT_FLOAT_EQ(got_lxbv_single[i], expect_lstsq_broadcast_vec[i], 5e-4f);

  const int64_t wide_shape[] = {2, 3};
  const int64_t wide_vec_shape[] = {2};
  const int64_t wide_mat_shape[] = {2, 2};
  float wide_a[] = {1, 2, 0, 0, 1, 1};
  float wide_b_vec[] = {1, 2};
  float wide_b_mat[] = {1, 3, 2, 4};
  float expect_wide_vec[] = {-0.33333334f, 0.6666667f, 1.3333334f};
  float expect_wide_mat[] = {
      -0.33333334f, -0.33333334f,
      0.6666667f, 1.6666666f,
      1.3333334f, 2.3333333f,
  };
  PolyUOp *lwa = make_buf(ctx, wide_shape, 2);
  PolyUOp *lwv = make_buf(ctx, wide_vec_shape, 1);
  PolyUOp *lwxv = poly_lstsq(ctx, lwa, lwv);
  ASSERT_NOT_NULL(lwxv);
  float got_lwxv[3] = {0};
  PolyUOp *lwide_vec_leaves[] = {base_buf(lwa), base_buf(lwv)};
  float *lwide_vec_ld[] = {wide_a, wide_b_vec};
  ASSERT_INT_EQ(realize_uop(ctx, lwxv, poly_buffer_f32(ctx, 3), got_lwxv, lwide_vec_leaves, lwide_vec_ld, 2), 0);
  for (int i = 0; i < 3; i++)
    ASSERT_FLOAT_EQ(got_lwxv[i], expect_wide_vec[i], 6e-4f);

  PolyUOp *lwB = make_buf(ctx, wide_mat_shape, 2);
  PolyUOp *lwxm = poly_lstsq(ctx, lwa, lwB);
  ASSERT_NOT_NULL(lwxm);
  float got_lwxm[6] = {0};
  PolyUOp *lwide_mat_leaves[] = {base_buf(lwa), base_buf(lwB)};
  float *lwide_mat_ld[] = {wide_a, wide_b_mat};
  ASSERT_INT_EQ(realize_uop(ctx, lwxm, poly_buffer_f32(ctx, 6), got_lwxm, lwide_mat_leaves, lwide_mat_ld, 2), 0);
  for (int i = 0; i < 6; i++)
    ASSERT_FLOAT_EQ(got_lwxm[i], expect_wide_mat[i], 8e-4f);

  const int64_t rankdef_square_shape[] = {2, 2};
  const int64_t rankdef_tall_shape[] = {3, 2};
  const int64_t rankdef_wide_shape[] = {2, 3};
  float rankdef_square_a[] = {1, 1, 2, 2};
  float rankdef_tall_a[] = {1, 1, 2, 2, 3, 3};
  float rankdef_wide_a[] = {1, 1, 0, 2, 2, 0};
  float rankdef_square_bv[] = {3, 6};
  float rankdef_tall_bv[] = {1, 2, 3};
  float rankdef_square_bm[] = {3, 1, 6, 2};

  PolyUOp *rd_sq_a = make_buf(ctx, rankdef_square_shape, 2);
  PolyUOp *rd_sq_bv = make_buf(ctx, wide_vec_shape, 1);
  PolyUOp *rd_sq_xv = poly_lstsq(ctx, rd_sq_a, rd_sq_bv);
  ASSERT_NOT_NULL(rd_sq_xv);
  float got_rd_sq_v[2] = {0};
  PolyUOp *rd_sq_v_leaves[] = {base_buf(rd_sq_a), base_buf(rd_sq_bv)};
  float *rd_sq_v_ld[] = {rankdef_square_a, rankdef_square_bv};
  ASSERT_INT_EQ(realize_uop(ctx, rd_sq_xv, poly_buffer_f32(ctx, 2), got_rd_sq_v, rd_sq_v_leaves, rd_sq_v_ld, 2), 0);
  ASSERT_FLOAT_EQ(got_rd_sq_v[0], 1.5f, 1e-3f);
  ASSERT_FLOAT_EQ(got_rd_sq_v[1], 1.5f, 1e-3f);

  PolyUOp *rd_sq_bm = make_buf(ctx, wide_mat_shape, 2);
  PolyUOp *rd_sq_xm = poly_lstsq(ctx, rd_sq_a, rd_sq_bm);
  ASSERT_NOT_NULL(rd_sq_xm);
  float got_rd_sq_m[4] = {0};
  PolyUOp *rd_sq_m_leaves[] = {base_buf(rd_sq_a), base_buf(rd_sq_bm)};
  float *rd_sq_m_ld[] = {rankdef_square_a, rankdef_square_bm};
  ASSERT_INT_EQ(realize_uop(ctx, rd_sq_xm, poly_buffer_f32(ctx, 4), got_rd_sq_m, rd_sq_m_leaves, rd_sq_m_ld, 2), 0);
  float expect_rd_sq_m[] = {1.5f, 0.5f, 1.5f, 0.5f};
  for (int i = 0; i < 4; i++)
    ASSERT_FLOAT_EQ(got_rd_sq_m[i], expect_rd_sq_m[i], 1e-3f);

  PolyUOp *rd_tall_a = make_buf(ctx, rankdef_tall_shape, 2);
  PolyUOp *rd_tall_bv = make_buf(ctx, tall_vec_shape, 1);
  PolyUOp *rd_tall_x = poly_lstsq(ctx, rd_tall_a, rd_tall_bv);
  ASSERT_NOT_NULL(rd_tall_x);
  float got_rd_tall[2] = {0};
  PolyUOp *rd_tall_leaves[] = {base_buf(rd_tall_a), base_buf(rd_tall_bv)};
  float *rd_tall_ld[] = {rankdef_tall_a, rankdef_tall_bv};
  ASSERT_INT_EQ(realize_uop(ctx, rd_tall_x, poly_buffer_f32(ctx, 2), got_rd_tall, rd_tall_leaves, rd_tall_ld, 2), 0);
  ASSERT_FLOAT_EQ(got_rd_tall[0], 0.5f, 1e-3f);
  ASSERT_FLOAT_EQ(got_rd_tall[1], 0.5f, 1e-3f);

  PolyUOp *rd_wide_a = make_buf(ctx, rankdef_wide_shape, 2);
  PolyUOp *rd_wide_bv = make_buf(ctx, wide_vec_shape, 1);
  PolyUOp *rd_wide_x = poly_lstsq(ctx, rd_wide_a, rd_wide_bv);
  ASSERT_NOT_NULL(rd_wide_x);
  float got_rd_wide[3] = {0};
  PolyUOp *rd_wide_leaves[] = {base_buf(rd_wide_a), base_buf(rd_wide_bv)};
  float *rd_wide_ld[] = {rankdef_wide_a, rankdef_square_bv};
  ASSERT_INT_EQ(realize_uop(ctx, rd_wide_x, poly_buffer_f32(ctx, 3), got_rd_wide, rd_wide_leaves, rd_wide_ld, 2), 0);
  ASSERT_FLOAT_EQ(got_rd_wide[0], 1.5f, 1e-3f);
  ASSERT_FLOAT_EQ(got_rd_wide[1], 1.5f, 1e-3f);
  ASSERT_FLOAT_EQ(got_rd_wide[2], 0.0f, 1e-3f);

  PolyUOp *bad_a = make_buf(ctx, (int64_t[]){2, 3}, 2);
  ASSERT_TRUE(poly_solve(ctx, bad_a, b_m) == NULL);
  ASSERT_TRUE(poly_lstsq(ctx, bad_a, lbv) == NULL);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST_COMMON(pe, linalg_structured_boundaries_backend_common) {
  PolyCtx *ctx = poly_ctx_new();
  int f32_id = poly_dtype_id_by_name("float32");
  ASSERT_TRUE(f32_id >= 0);
  PolyDevice device = poly_ctx_get_preferred_device(ctx);
  if (!poly_device_can_execute(device)) device = POLY_DEVICE_CPU;
  int64_t shape2[] = {2, 2};
  int64_t vec_shape[] = {2};

  float qr_data[] = {1, 2, 3, 4};
  PolyTensor *qr_host =
      poly_tensor_from_host_by_id(ctx, qr_data, sizeof(qr_data), f32_id, shape2, 2);
  PolyTensor *qr_a = poly_tensor_to_device(ctx, qr_host, device);
  ASSERT_NOT_NULL(qr_host);
  ASSERT_NOT_NULL(qr_a);
  PolyTensor *q = NULL, *r = NULL;
  ASSERT_INT_EQ(poly_tensor_qr_ex(ctx, qr_a, POLY_QR_REDUCED, &q, &r), 0);
  ASSERT_NOT_NULL(q);
  ASSERT_NOT_NULL(r);
  PolyTensor *qr_t = poly_tensor_dot(ctx, q, r);
  ASSERT_NOT_NULL(qr_t);
  PolyTensor *qr_out = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &qr_t, 1, &qr_out), 0);
  ASSERT_NOT_NULL(qr_out);
  const PolyUOp *qr_buf = poly_uop_get_buffer_identity(poly_tensor_uop(qr_out));
  ASSERT_NOT_NULL(qr_buf);
  float qr_got[4] = {0};
  ASSERT_INT_EQ(poly_buffer_read(ctx, (PolyUOp *)qr_buf, qr_got, sizeof(qr_got)), 0);
  for (int i = 0; i < 4; i++)
    ASSERT_FLOAT_EQ(qr_got[i], qr_data[i], 2e-3f);

  float spd[] = {4, 2, 2, 5};
  float rhs_mat[] = {1, 2, 3, 4};
  float expect_cholsolve[] = {-0.0625f, 0.125f, 0.625f, 0.75f};
  PolyTensor *chol_ah =
      poly_tensor_from_host_by_id(ctx, spd, sizeof(spd), f32_id, shape2, 2);
  PolyTensor *chol_bh = poly_tensor_from_host_by_id(
      ctx, rhs_mat, sizeof(rhs_mat), f32_id, shape2, 2
  );
  PolyTensor *chol_a = poly_tensor_to_device(ctx, chol_ah, device);
  PolyTensor *chol_b = poly_tensor_to_device(ctx, chol_bh, device);
  ASSERT_NOT_NULL(chol_ah);
  ASSERT_NOT_NULL(chol_bh);
  ASSERT_NOT_NULL(chol_a);
  ASSERT_NOT_NULL(chol_b);
  PolyTensor *factor = poly_tensor_cholesky(ctx, chol_a, 0);
  ASSERT_NOT_NULL(factor);
  PolyTensor *chol_t = poly_tensor_cholesky_solve(ctx, factor, chol_b, 0);
  ASSERT_NOT_NULL(chol_t);
  PolyTensor *chol_out = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &chol_t, 1, &chol_out), 0);
  ASSERT_NOT_NULL(chol_out);
  const PolyUOp *chol_buf = poly_uop_get_buffer_identity(poly_tensor_uop(chol_out));
  ASSERT_NOT_NULL(chol_buf);
  float chol_got[4] = {0};
  ASSERT_INT_EQ(poly_buffer_read(ctx, (PolyUOp *)chol_buf, chol_got, sizeof(chol_got)), 0);
  for (int i = 0; i < 4; i++)
    ASSERT_FLOAT_EQ(chol_got[i], expect_cholsolve[i], 2e-4f);

  float a2[] = {2, 1, 1, 3};
  float b_vec[] = {1, 4};
  float expect_vec[] = {-0.2f, 1.4f};
  PolyTensor *solve_ah =
      poly_tensor_from_host_by_id(ctx, a2, sizeof(a2), f32_id, shape2, 2);
  PolyTensor *solve_bh =
      poly_tensor_from_host_by_id(ctx, b_vec, sizeof(b_vec), f32_id, vec_shape, 1);
  PolyTensor *solve_a = poly_tensor_to_device(ctx, solve_ah, device);
  PolyTensor *solve_b = poly_tensor_to_device(ctx, solve_bh, device);
  ASSERT_NOT_NULL(solve_ah);
  ASSERT_NOT_NULL(solve_bh);
  ASSERT_NOT_NULL(solve_a);
  ASSERT_NOT_NULL(solve_b);
  PolyTensor *solve_t = poly_tensor_solve(ctx, solve_a, solve_b);
  ASSERT_NOT_NULL(solve_t);
  PolyTensor *solve_out = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &solve_t, 1, &solve_out), 0);
  ASSERT_NOT_NULL(solve_out);
  const PolyUOp *solve_buf = poly_uop_get_buffer_identity(poly_tensor_uop(solve_out));
  ASSERT_NOT_NULL(solve_buf);
  float solve_got[2] = {0};
  ASSERT_INT_EQ(poly_buffer_read(ctx, (PolyUOp *)solve_buf, solve_got, sizeof(solve_got)), 0);
  for (int i = 0; i < 2; i++)
    ASSERT_FLOAT_EQ(solve_got[i], expect_vec[i], 3e-4f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, layernorm_v2_shape) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = make_buf(ctx, (int64_t[]){2, 3}, 2);
  PolyUOp *r = poly_layernorm_apply(ctx, x, NULL, NULL, -1, 1e-5);
  ASSERT_NOT_NULL(r);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, r), 2);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, r)[0], 2);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, r)[1], 3);
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
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, r)[3], 4);

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
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, b)[0], 100);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(shape_uop, buffer_dynamic) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *var = poly_define_var(ctx, "batch", 1, 32);
  PolyUOp *buf = poly_buffer_var(ctx, POLY_FLOAT32, var, (int64_t[]){10}, 1);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, buf), 2);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, buf)[0], 32);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, buf)[1], 10);
  ASSERT_PTR_EQ(poly_uop_shape_dim(ctx, buf, 0), var);
  ASSERT_INT_EQ(poly_uop_shape_dim(ctx, buf, 1)->op, POLY_OP_CONST);
  ASSERT_INT_EQ(poly_uop_shape_dim(ctx, buf, 1)->arg.i, 10);

  PolyUOp *bound = poly_bind_var(ctx, var, 7);
  PolyUOp *bound_buf = poly_buffer_var(ctx, POLY_FLOAT32, bound, (int64_t[]){10}, 1);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, bound_buf), 2);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, bound_buf)[0], 32);
  ASSERT_PTR_EQ(poly_uop_shape_dim(ctx, bound_buf, 0), bound);
  ASSERT_PTR_EQ(poly_uop_unbind_var(poly_uop_shape_dim(ctx, bound_buf, 0)), var);
  int64_t value = -1;
  ASSERT_INT_EQ(poly_uop_bind_value(poly_uop_shape_dim(ctx, bound_buf, 0), &value), 0);
  ASSERT_INT_EQ(value, 7);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(shape_uop, gettuple_function_resolves_symbolic_param_shape_and_axis) {
  /* Pinned GETTUPLE(FUNCTION)._shape substitutes PARAM(slot) dimensions with
   * ordered FUNCTION arguments, while BIND._min_max retains the PARAM range
   * (tinygrad/uop/ops.py:242-253,628-630,1010-1012,1691). */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *empty = poly_uop(ctx, POLY_OP_STACK, POLY_VOID, NULL, 0, poly_arg_none());
  PolyParamArg dim_arg = {
      .slot = 1,
      .name = "size",
      .min_val = 1,
      .max_val = 8,
      .has_minmax = true,
      .addrspace = POLY_ADDR_GLOBAL,
  };
  PolyUOp *dim_param = poly_uop1(
      ctx, POLY_OP_PARAM, POLY_INDEX, empty, poly_arg_param(&dim_arg));
  PolyUOp *body_dims[2] = {dim_param, poly_const_int(ctx, 4)};
  PolyUOp *body_shape = poly_uop(
      ctx, POLY_OP_STACK, poly_dtype_vec(POLY_INDEX, 2), body_dims, 2,
      poly_arg_none());
  PolyParamArg value_arg = {
      .slot = 0,
      .addrspace = POLY_ADDR_GLOBAL,
      .axis = 1,
      .has_axis = true,
  };
  PolyUOp *value = poly_uop1(
      ctx, POLY_OP_PARAM, POLY_FLOAT32, body_shape, poly_arg_param(&value_arg));
  PolyUOp *body = poly_uop1(ctx, POLY_OP_TUPLE, POLY_VOID, value, poly_arg_none());

  PolyUOp *actual = poly_buffer_on_device(ctx, POLY_FLOAT32, 20, POLY_DEVICE_CPU);
  actual = poly_reshape(ctx, actual, (int64_t[]){5, 4}, 2);
  PolyUOp *size = poly_uop0(
      ctx, POLY_OP_DEFINE_VAR, POLY_INDEX, poly_arg_define_var("size", 1, 8));
  PolyUOp *bound = poly_uop2(
      ctx, POLY_OP_BIND, POLY_INDEX, size, poly_const_int(ctx, 5), poly_arg_none());
  PolyUOp *function_src[3] = {body, actual, bound};
  PolyUOp *function = poly_uop(
      ctx, POLY_OP_FUNCTION, POLY_VOID, function_src, 3,
      poly_arg_str("symbolic_shape"));
  PolyUOp *selected = poly_uop1(
      ctx, POLY_OP_GETTUPLE, POLY_FLOAT32, function, poly_arg_int(0));

  ASSERT_INT_EQ(poly_uop_ndim(ctx, selected), 2);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, selected)[0], 8);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, selected)[1], 4);
  ASSERT_PTR_EQ(poly_uop_shape_dim(ctx, selected, 0), bound);
  int axis = -1;
  ASSERT_TRUE(poly_uop_axis(ctx, selected, &axis));
  ASSERT_INT_EQ(axis, 1);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(shape_uop, shrink_uop_with_bound_start_matches_tinygrad_form) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *x = make_buf(ctx, (int64_t[]){10, 2}, 2);
  PolyUOp *i = poly_define_var(ctx, "i", 0, 8);
  PolyUOp *ib = poly_bind_var(ctx, i, 4);
  PolyUOp *starts[2] = {ib, poly_const_int(ctx, 0)};
  PolyUOp *sizes[2] = {poly_const_int(ctx, 2), poly_const_int(ctx, 2)};

  PolyUOp *slice = poly_shrink_uop(ctx, x, starts, sizes, 2);
  ASSERT_NOT_NULL(slice);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, slice), 2);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, slice)[0], 2);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, slice)[1], 2);
  PolyUOp *slice_dim0 = poly_uop_shape_dim(ctx, slice, 0);
  ASSERT_NOT_NULL(slice_dim0);
  ASSERT_INT_EQ(slice_dim0->op, POLY_OP_CONST);
  ASSERT_INT_EQ(slice_dim0->arg.i, 2);

  float data[20];
  for (int j = 0; j < 20; j++) data[j] = (float)j;
  float out[4] = {0};
  PolyUOp *out_buf = poly_buffer_f32(ctx, 4);
  PolyUOp *leaves[1] = {base_buf(x)};
  float *leaf_data[1] = {data};
  ASSERT_INT_EQ(realize_uop(ctx, slice, out_buf, out, leaves, leaf_data, 1), 0);
  ASSERT_FLOAT_EQ(out[0], 8.0f, 1e-6f);
  ASSERT_FLOAT_EQ(out[1], 9.0f, 1e-6f);
  ASSERT_FLOAT_EQ(out[2], 10.0f, 1e-6f);
  ASSERT_FLOAT_EQ(out[3], 11.0f, 1e-6f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(shape_uop, shrink_uop_preserves_variable_extent_like_tinygrad) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *x = make_buf(ctx, (int64_t[]){10, 2}, 2);
  PolyUOp *n = poly_define_var(ctx, "n", 1, 8);
  PolyUOp *nb = poly_bind_var(ctx, n, 3);
  PolyUOp *starts[2] = {poly_const_int(ctx, 0), poly_const_int(ctx, 0)};
  PolyUOp *sizes[2] = {nb, poly_const_int(ctx, 2)};

  PolyUOp *slice = poly_shrink_uop(ctx, x, starts, sizes, 2);
  ASSERT_NOT_NULL(slice);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, slice), 2);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, slice)[0], 8);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, slice)[1], 2);
  PolyUOp *slice_dim0 = poly_uop_shape_dim(ctx, slice, 0);
  ASSERT_NOT_NULL(slice_dim0);
  ASSERT_PTR_EQ(poly_uop_unbind_var(slice_dim0), n);
  int64_t bound_value = -1;
  ASSERT_INT_EQ(poly_uop_bind_value(slice_dim0, &bound_value), 0);
  ASSERT_INT_EQ(bound_value, 3);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(shape_uop, expand_uop_preserves_variable_extent_like_tinygrad) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *x = make_buf(ctx, (int64_t[]){2}, 1);
  PolyUOp *xr = poly_reshape(ctx, x, (int64_t[]){1, 2}, 2);
  PolyUOp *n = poly_define_var(ctx, "n", 1, 4);
  PolyUOp *nb = poly_bind_var(ctx, n, 3);
  PolyUOp *dims[2] = {nb, poly_const_int(ctx, 2)};

  PolyUOp *expanded = poly_expand_uop(ctx, xr, dims, 2);
  ASSERT_NOT_NULL(expanded);
  ASSERT_INT_EQ(expanded->op, POLY_OP_EXPAND);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, expanded), 2);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, expanded)[0], 4);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, expanded)[1], 2);
  PolyUOp *dim0 = poly_uop_shape_dim(ctx, expanded, 0);
  ASSERT_NOT_NULL(dim0);
  ASSERT_PTR_EQ(poly_uop_unbind_var(dim0), n);
  int64_t bound_value = -1;
  ASSERT_INT_EQ(poly_uop_bind_value(dim0, &bound_value), 0);
  ASSERT_INT_EQ(bound_value, 3);
  PolyUOp *dim1 = poly_uop_shape_dim(ctx, expanded, 1);
  ASSERT_NOT_NULL(dim1);
  ASSERT_INT_EQ(dim1->op, POLY_OP_CONST);
  ASSERT_INT_EQ(dim1->arg.i, 2);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(shape_uop, alu_after_symbolic_expand_accepts_equal_static_dims) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *n = poly_define_var(ctx, "n", 1, 4);
  PolyUOp *nb = poly_bind_var(ctx, n, 3);
  PolyUOp *lhs = poly_buffer_var(ctx, POLY_FLOAT32, nb, (int64_t[]){10}, 1);
  ASSERT_NOT_NULL(lhs);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, lhs), 2);
  ASSERT_PTR_EQ(poly_uop_unbind_var(poly_uop_shape_dim(ctx, lhs, 0)), n);

  PolyUOp *target[2] = {nb, poly_const_int(ctx, 10)};

  PolyUOp *rhs_1x10 = poly_reshape(ctx, poly_buffer_f32(ctx, 10), (int64_t[]){1, 10}, 2);
  PolyUOp *rhs_1x10_expanded = poly_expand_uop(ctx, rhs_1x10, target, 2);
  ASSERT_NOT_NULL(rhs_1x10_expanded);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, rhs_1x10_expanded), 2);
  ASSERT_PTR_EQ(poly_uop_unbind_var(poly_uop_shape_dim(ctx, rhs_1x10_expanded, 0)), n);
  ASSERT_PTR_NEQ(poly_uop_shape_dim(ctx, lhs, 1), poly_uop_shape_dim(ctx, rhs_1x10_expanded, 1));

  PolyUOp *prod = poly_mul(ctx, lhs, rhs_1x10_expanded);
  ASSERT_NOT_NULL(prod);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, prod), 2);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, prod)[0], 4);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, prod)[1], 10);
  ASSERT_PTR_EQ(poly_uop_unbind_var(poly_uop_shape_dim(ctx, prod, 0)), n);
  int64_t dim_value = -1;
  ASSERT_INT_EQ(poly_uop_const_i64(poly_uop_shape_dim(ctx, prod, 1), &dim_value), 0);
  ASSERT_INT_EQ(dim_value, 10);

  PolyUOp *rhs_10 = make_buf(ctx, (int64_t[]){10}, 1);
  PolyUOp *rhs_10_reshaped = poly_reshape(ctx, rhs_10, (int64_t[]){1, 10}, 2);
  PolyUOp *rhs_10_expanded = poly_expand_uop(ctx, rhs_10_reshaped, target, 2);
  ASSERT_NOT_NULL(rhs_10_expanded);
  ASSERT_PTR_NEQ(poly_uop_shape_dim(ctx, lhs, 1), poly_uop_shape_dim(ctx, rhs_10_expanded, 1));

  PolyUOp *sum = poly_add(ctx, lhs, rhs_10_expanded);
  ASSERT_NOT_NULL(sum);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, sum), 2);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, sum)[0], 4);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, sum)[1], 10);
  ASSERT_PTR_EQ(poly_uop_unbind_var(poly_uop_shape_dim(ctx, sum, 0)), n);
  dim_value = -1;
  ASSERT_INT_EQ(poly_uop_const_i64(poly_uop_shape_dim(ctx, sum, 1), &dim_value), 0);
  ASSERT_INT_EQ(dim_value, 10);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(shape_uop, smoothed_bound_slice_multiplies_static_batch_shape) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *labels_all = make_buf(ctx, (int64_t[]){8, 10}, 2);
  PolyUOp *i = poly_define_var(ctx, "i", 0, 6);
  PolyUOp *ib = poly_bind_var(ctx, i, 0);
  PolyUOp *starts[2] = {ib, poly_const_int(ctx, 0)};
  PolyUOp *sizes[2] = {poly_const_int(ctx, 2), poly_const_int(ctx, 10)};
  PolyUOp *labels = poly_shrink_uop(ctx, labels_all, starts, sizes, 2);
  ASSERT_NOT_NULL(labels);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, labels), 2);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, labels)[0], 2);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, labels)[1], 10);

  PolyUOp *smoothed = poly_add(
      ctx,
      poly_mul(ctx, labels, poly_const_float(ctx, 0.9f)),
      poly_const_float(ctx, 0.01f)
  );
  PolyUOp *static_logp = make_buf(ctx, (int64_t[]){2, 10}, 2);
  PolyUOp *prod = poly_mul(ctx, smoothed, static_logp);
  ASSERT_NOT_NULL(prod);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, prod), 2);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, prod)[0], 2);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, prod)[1], 10);

  PolyUOp *per_row = poly_reduce_axis(ctx, POLY_OP_ADD, prod, (int64_t[]){1}, 1);
  ASSERT_NOT_NULL(per_row);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, per_row), 2);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, per_row)[0], 2);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, per_row)[1], 1);
  PolyUOp *loss_rows = poly_reshape(ctx, per_row, (int64_t[]){2}, 1);
  ASSERT_NOT_NULL(loss_rows);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, loss_rows), 1);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, loss_rows)[0], 2);

  float labels_data[80] = {0};
  for (int r = 0; r < 8; r++)
    labels_data[r * 10 + (r % 10)] = 1.0f;
  float logp_data[20];
  for (int j = 0; j < 20; j++)
    logp_data[j] = 1.0f;
  float out[2] = {0};
  PolyUOp *leaves[2] = {base_buf(labels_all), base_buf(static_logp)};
  float *leaf_data[2] = {labels_data, logp_data};
  ASSERT_INT_EQ(realize_uop(ctx, loss_rows, poly_buffer_f32(ctx, 2), out, leaves, leaf_data, 2), 0);
  ASSERT_FLOAT_EQ(out[0], 1.0f, 1e-6f);
  ASSERT_FLOAT_EQ(out[1], 1.0f, 1e-6f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(shape_uop, buffer_unique_ids_are_ctx_local) {
  PolyCtx *ctx1 = poly_ctx_new();
  PolyCtx *ctx2 = poly_ctx_new();

  PolyUOp *a0 = poly_buffer(ctx1, POLY_FLOAT32, 4);
  PolyUOp *a1 = poly_buffer(ctx1, POLY_FLOAT32, 4);
  PolyUOp *n = poly_define_var(ctx1, "N", 1, 8);
  PolyUOp *ad = poly_buffer_var(ctx1, POLY_FLOAT32, n, NULL, 0);
  PolyUOp *b0 = poly_buffer(ctx2, POLY_FLOAT32, 4);
  PolyUOp *m = poly_define_var(ctx2, "M", 1, 8);
  PolyUOp *bd = poly_buffer_var(ctx2, POLY_FLOAT32, m, NULL, 0);

  ASSERT_NOT_NULL(a0);
  ASSERT_NOT_NULL(a1);
  ASSERT_NOT_NULL(ad);
  ASSERT_NOT_NULL(b0);
  ASSERT_NOT_NULL(bd);
  ASSERT_INT_EQ(a0->src[0]->op, POLY_OP_UNIQUE);
  ASSERT_INT_EQ(a1->src[0]->op, POLY_OP_UNIQUE);
  ASSERT_INT_EQ(ad->src[0]->op, POLY_OP_UNIQUE);
  ASSERT_INT_EQ(b0->src[0]->op, POLY_OP_UNIQUE);
  ASSERT_INT_EQ(bd->src[0]->op, POLY_OP_UNIQUE);
  ASSERT_INT_EQ(a0->src[0]->arg.i, 0);
  ASSERT_INT_EQ(a1->src[0]->arg.i, 1);
  ASSERT_INT_EQ(ad->src[0]->arg.i, 2);
  ASSERT_INT_EQ(b0->src[0]->arg.i, 0);
  ASSERT_INT_EQ(bd->src[0]->arg.i, 1);

  poly_ctx_destroy(ctx2);
  poly_ctx_destroy(ctx1);
  PASS();
}

TEST(shape_uop, reshape) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *r = poly_reshape(ctx, poly_buffer(ctx, POLY_FLOAT32, 24), (int64_t[]){2, 3, 4}, 3);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, r), 3);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, r)[0], 2);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, r)[1], 3);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, r)[2], 4);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(shape_uop, permute) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *b = make_buf(ctx, (int64_t[]){2, 3, 4}, 3);
  PolyUOp *p = poly_permute(ctx, b, (int64_t[]){2, 0, 1}, 3);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, p)[0], 4);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, p)[1], 2);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, p)[2], 3);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(shape_uop, pad) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *b = make_buf(ctx, (int64_t[]){2, 3}, 2);
  PolyUOp *p = poly_pad(ctx, b, (int64_t[][2]){{1, 1}, {2, 0}}, 2);
  /* Pinned UOp._mop stores PAD offsets and output sizes as shape-value
   * sources (uop/ops.py:710-721), never as a pair-tuple arg. */
  ASSERT_INT_EQ(p->op, POLY_OP_PAD);
  ASSERT_INT_EQ(p->arg.kind, POLY_ARG_NONE);
  ASSERT_INT_EQ(p->n_src, 3);
  ASSERT_INT_EQ(p->src[1]->op, POLY_OP_STACK);
  ASSERT_INT_EQ(p->src[2]->op, POLY_OP_STACK);
  ASSERT_INT_EQ(p->src[1]->src[0]->arg.i, 1);
  ASSERT_INT_EQ(p->src[1]->src[1]->arg.i, 2);
  ASSERT_INT_EQ(p->src[2]->src[0]->arg.i, 4);
  ASSERT_INT_EQ(p->src[2]->src[1]->arg.i, 5);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, p)[0], 4);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, p)[1], 5);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(shape_uop, shrink) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *b = make_buf(ctx, (int64_t[]){4, 5}, 2);
  PolyUOp *s = poly_shrink(ctx, b, (int64_t[][2]){{1, 3}, {0, 4}}, 2);
  /* Pinned UOp._mop stores SHRINK starts and sizes as shape-value sources
   * (uop/ops.py:710-721). */
  ASSERT_INT_EQ(s->op, POLY_OP_SHRINK);
  ASSERT_INT_EQ(s->arg.kind, POLY_ARG_NONE);
  ASSERT_INT_EQ(s->n_src, 3);
  ASSERT_INT_EQ(s->src[1]->op, POLY_OP_STACK);
  ASSERT_INT_EQ(s->src[2]->op, POLY_OP_STACK);
  ASSERT_INT_EQ(s->src[1]->src[0]->arg.i, 1);
  ASSERT_INT_EQ(s->src[1]->src[1]->arg.i, 0);
  ASSERT_INT_EQ(s->src[2]->src[0]->arg.i, 2);
  ASSERT_INT_EQ(s->src[2]->src[1]->arg.i, 4);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, s)[0], 2);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, s)[1], 4);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(shape_uop, scalar_pad_and_shrink_are_noops) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *scalar = poly_reshape(ctx, poly_buffer_f32(ctx, 1), NULL, 0);
  ASSERT_NOT_NULL(scalar);
  ASSERT_PTR_EQ(poly_pad(ctx, scalar, NULL, 0), scalar);
  ASSERT_PTR_EQ(poly_pad_value(ctx, scalar, NULL, 0, 0.0), scalar);
  PolyUOp *filled = poly_pad_value(ctx, scalar, NULL, 0, 5.0);
  ASSERT_NOT_NULL(filled);
  ASSERT_INT_EQ(filled->op, POLY_OP_WHERE);
  ASSERT_INT_EQ(filled->n_src, 3);
  ASSERT_INT_EQ(filled->src[0]->op, POLY_OP_CAST);
  ASSERT_INT_EQ(filled->src[0]->src[0]->op, POLY_OP_CONST);
  ASSERT_PTR_EQ(filled->src[1], scalar);
  ASSERT_PTR_EQ(poly_shrink(ctx, scalar, NULL, 0), scalar);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(shape_uop, unchanged_pad_and_shrink_are_noops) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *value = make_buf(ctx, (int64_t[]){2, 3}, 2);
  ASSERT_NOT_NULL(value);
  ASSERT_PTR_EQ(poly_pad(ctx, value, (int64_t[][2]){{0, 0}, {0, 0}}, 2), value);
  ASSERT_PTR_EQ(poly_shrink(ctx, value, (int64_t[][2]){{0, 2}, {0, 3}}, 2), value);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(shape_uop, flip_uses_rank_sized_boolean_mask) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *b = make_buf(ctx, (int64_t[]){2, 3}, 2);
  PolyUOp *f = poly_flip(ctx, b, (int64_t[]){1}, 1);
  ASSERT_NOT_NULL(f);
  ASSERT_INT_EQ(f->op, POLY_OP_FLIP);
  ASSERT_INT_EQ(f->arg.kind, POLY_ARG_INT_TUPLE);
  ASSERT_INT_EQ(f->arg.int_tuple.n, 2);
  ASSERT_INT_EQ(f->arg.int_tuple.vals[0], 0);
  ASSERT_INT_EQ(f->arg.int_tuple.vals[1], 1);
  ASSERT_PTR_EQ(poly_flip(ctx, b, NULL, 0), b);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(shape_uop, reduce_axis) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *b = make_buf(ctx, (int64_t[]){2, 3, 4}, 3);
  PolyUOp *r = poly_reduce_axis(ctx, POLY_OP_ADD, b, (int64_t[]){1}, 1);
  /* Pinned UOp._rop emits tensor-stage REDUCE(value, (op, axes))
   * (uop/ops.py:567-569). */
  ASSERT_EQ(r->op, POLY_OP_REDUCE);
  ASSERT_EQ(r->arg.kind, POLY_ARG_REDUCE_AXIS);
  ASSERT_EQ(r->arg.reduce_axis.op, POLY_OP_ADD);
  ASSERT_INT_EQ(r->arg.reduce_axis.n, 1);
  ASSERT_INT_EQ(r->arg.reduce_axis.axes[0], 1);
  ASSERT_INT_EQ(r->n_src, 1);
  ASSERT_PTR_EQ(r->src[0], b);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, r), 3);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, r)[0], 2);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, r)[1], 1);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, r)[2], 4);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(shape_uop, reduce_axis_drops_singleton_axes_like_tinygrad_rop) {
  PolyCtx *ctx = poly_ctx_new();

  /* tinygrad UOp._rop filters size-1 axes before constructing REDUCE.
   * Keep that parity at the constructor boundary so later schedule/rangeify
   * stages never see no-op singleton reductions. */
  PolyUOp *b = make_buf(ctx, (int64_t[]){1, 4, 3}, 3);

  PolyUOp *only_singleton = poly_reduce_axis(ctx, POLY_OP_ADD, b, (int64_t[]){0}, 1);
  ASSERT_PTR_EQ(only_singleton, b);

  PolyUOp *mixed = poly_reduce_axis(ctx, POLY_OP_ADD, b, (int64_t[]){2, 0}, 2);
  ASSERT_NOT_NULL(mixed);
  ASSERT_EQ(mixed->op, POLY_OP_REDUCE);
  ASSERT_EQ(mixed->arg.kind, POLY_ARG_REDUCE_AXIS);
  ASSERT_INT_EQ(mixed->arg.reduce_axis.n, 1);
  ASSERT_INT_EQ(mixed->arg.reduce_axis.axes[0], 2);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, mixed)[0], 1);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, mixed)[1], 4);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, mixed)[2], 1);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(shape_uop, alu_broadcast_scalar) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = make_buf(ctx, (int64_t[]){3, 4}, 2);
  PolyUOp *c = poly_const_float(ctx, 1.0);
  PolyUOp *r = poly_alu2(ctx, POLY_OP_ADD, a, c);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, r), 2);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, r)[0], 3);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, r)[1], 4);
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
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, r)[0], 3);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, r)[1], 5);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, r)[2], 4);
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
  ASSERT_TRUE(poly_uop_max_shape_dims(ctx, poly_const_float(ctx, 42.0)) == NULL);
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
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, c)[0], 2);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(shape_uop, assign_flat) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *r = make_buf(ctx, (int64_t[]){3, 4}, 2);
  PolyUOp *a =
      poly_store_buffer_update(ctx, r, poly_alu2(ctx, POLY_OP_ADD, r, poly_const_float(ctx, 1.0)));
  /* Whole-buffer update helper normalizes to flat BUFFER target. */
  ASSERT_TRUE(a->op == POLY_OP_STORE);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, a->src[0]), 1);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, a->src[0])[0], 12);
  poly_ctx_destroy(ctx);
  PASS();
}

/* Shape parity oracle */

static int check_shape_parity(PolyCtx *ctx, PolyUOp *root) {
  int n_topo;
  PolyUOp **topo = poly_toposort(ctx, root, &n_topo);
  int mismatches = 0;
  for (int i = 0; i < n_topo; i++) {
    PolyShape computed = poly_uop_max_shape(ctx, topo[i]);
    int cached_ndim = poly_uop_ndim(ctx, topo[i]);
    if (cached_ndim != computed.ndim) {
      fprintf(
          stderr, "  parity: op=%s cached=%d computed=%d\n", poly_op_name(topo[i]->op), cached_ndim,
          computed.ndim
      );
      mismatches++;
    } else if (cached_ndim > 0 && computed.dims) {
      for (int j = 0; j < cached_ndim; j++)
        if (poly_uop_max_shape_dims(ctx, topo[i])[j] != computed.dims[j]) {
          fprintf(
              stderr, "  parity: op=%s dim[%d] cached=%ld computed=%ld\n",
              poly_op_name(topo[i]->op), j, (long)poly_uop_max_shape_dims(ctx, topo[i])[j],
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
  PolyUOp *idx =
      poly_reshape(ctx, poly_buffer(ctx, POLY_INT32, 3), (int64_t[]){3}, 1);
  PolyUOp *g = poly_gather(ctx, table, idx);
  ASSERT_INT_EQ(check_shape_parity(ctx, g), 0);
  PolyUOp *float_idx =
      poly_reshape(ctx, poly_buffer(ctx, POLY_FLOAT32, 3), (int64_t[]){3}, 1);
  ASSERT_TRUE(poly_gather(ctx, table, float_idx) == NULL);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(shape_uop, gather_integer_promotion_scalar_and_rank_match_tinygrad) {
  /* Pinned nn/__init__.py:371-392 and mixin/__init__.py:439-449 accept scalar
   * integer indices and promote the index/class-range pair before CMPNE. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *table =
      poly_reshape(ctx, poly_buffer(ctx, POLY_FLOAT32, 12), (int64_t[]){4, 3}, 2);
  ASSERT_NOT_NULL(table);

  struct {
    PolyDType input;
    PolyDType expected_cmp;
  } cases[] = {
      {POLY_INT8, POLY_INT32},   {POLY_UINT8, POLY_INT32},
      {POLY_INT32, POLY_INT32}, {POLY_INT64, POLY_INT64},
      {POLY_UINT64, POLY_UINT64}, {POLY_INDEX, POLY_INT32},
  };
  for (int ci = 0; ci < (int)(sizeof(cases) / sizeof(cases[0])); ci++) {
    PolyUOp *idx = poly_reshape(
        ctx, poly_buffer(ctx, cases[ci].input, 2), (int64_t[]){2}, 1
    );
    PolyUOp *g = poly_gather(ctx, table, idx);
    ASSERT_NOT_NULL(g);
    PolyUOp *value_cmp = NULL;
    int n_topo = 0;
    PolyUOp **topo = poly_toposort(ctx, g, &n_topo);
    for (int i = 0; i < n_topo; i++) {
      PolyUOp *u = topo[i];
      if (u && u->op == POLY_OP_CMPNE && u->n_src == 2 &&
          !poly_dtype_is_bool(u->src[0]->dtype)) {
        value_cmp = u;
        break;
      }
    }
    ASSERT_NOT_NULL(value_cmp);
    ASSERT_TRUE(poly_dtype_eq(value_cmp->src[0]->dtype, cases[ci].expected_cmp));
    ASSERT_TRUE(poly_dtype_eq(value_cmp->src[1]->dtype, cases[ci].expected_cmp));
  }

  PolyUOp *scalar = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(2));
  PolyUOp *scalar_gather = poly_gather(ctx, table, scalar);
  ASSERT_NOT_NULL(scalar_gather);
  PolyShape scalar_shape = poly_uop_max_shape(ctx, scalar_gather);
  ASSERT_INT_EQ(scalar_shape.ndim, 1);
  ASSERT_INT_EQ(scalar_shape.dims[0], 3);
  free(scalar_shape.dims);

  PolyDType int_ptr = poly_dtype_ptr(POLY_INT32, 2, POLY_ADDR_GLOBAL);
  PolyUOp *ptr_indices = poly_uop0(ctx, POLY_OP_PARAM, int_ptr, poly_arg_int(0));
  ASSERT_TRUE(poly_gather(ctx, table, ptr_indices) == NULL);

  int64_t ones[POLY_MAX_DIMS];
  for (int i = 0; i < POLY_MAX_DIMS; i++) ones[i] = 1;
  PolyUOp *rank14 =
      poly_reshape(ctx, poly_buffer(ctx, POLY_INT32, 1), ones, POLY_MAX_DIMS - 2);
  PolyUOp *rank15 =
      poly_reshape(ctx, poly_buffer(ctx, POLY_INT32, 1), ones, POLY_MAX_DIMS - 1);
  PolyUOp *rank16 = poly_reshape(ctx, poly_buffer(ctx, POLY_INT32, 1), ones, POLY_MAX_DIMS);
  ASSERT_NOT_NULL(rank14);
  ASSERT_NOT_NULL(rank15);
  ASSERT_NOT_NULL(rank16);
  PolyUOp *rank14_gather = poly_gather(ctx, table, rank14);
  ASSERT_NOT_NULL(rank14_gather);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, rank14_gather), POLY_MAX_DIMS - 1);
  ASSERT_TRUE(poly_gather(ctx, table, rank15) == NULL);
  ASSERT_TRUE(poly_gather(ctx, table, rank16) == NULL);

  poly_ctx_destroy(ctx);
  PASS();
}

/* v2 reduce shape */

TEST(pe, v2_reduce_shape) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = make_buf(ctx, (int64_t[]){3, 4}, 2);
  PolyUOp *s = poly_sum_reduce(ctx, x, 1, 0);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, s), 1);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, s)[0], 3);
  PolyUOp *sk = poly_sum_reduce(ctx, x, 1, 1);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, sk), 2);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, sk)[0], 3);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, sk)[1], 1);
  poly_ctx_destroy(ctx);
  PASS();
}

/* Contiguous */

TEST(tensor, contiguous_tensor_matches_pinned_device_rules) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  int i32 = poly_dtype_id_by_name("int32");

  PolyTensor *pure = poly_tensor_arange_int_by_id(ctx, 0, 4, 1, i32, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(pure);
  PolyUOp *pure_physical = poly_tensor_uop_physical(pure);
  PolyTensor *pure_contiguous = poly_tensor_contiguous(ctx, pure);
  ASSERT_NOT_NULL(pure_contiguous);
  ASSERT_INT_EQ(poly_tensor_uop_logical(pure_contiguous)->op, POLY_OP_CONTIGUOUS);
  ASSERT_PTR_EQ(poly_tensor_uop_logical(pure_contiguous)->src[0], poly_tensor_uop_logical(pure));
  ASSERT_PTR_EQ(poly_tensor_uop_physical(pure_contiguous), pure_physical);
  ASSERT_INT_EQ(poly_tensor_uop_physical(pure_contiguous)->op, POLY_OP_ADD);

  int64_t shape[2] = {2, 3};
  int64_t order[2] = {1, 0};
  PolyTensor *storage = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 2, POLY_DEVICE_CPU);
  PolyTensor *permuted = poly_tensor_permute(ctx, storage, order, 2);
  PolyTensor *materialized = poly_tensor_contiguous(ctx, permuted);
  ASSERT_NOT_NULL(materialized);
  ASSERT_INT_EQ(poly_tensor_uop_logical(materialized)->op, POLY_OP_CONTIGUOUS);
  ASSERT_PTR_EQ(poly_tensor_uop_logical(materialized)->src[0], poly_tensor_uop_logical(permuted));
  ASSERT_INT_EQ(poly_tensor_uop_physical(materialized)->op, POLY_OP_CONTIGUOUS);
  ASSERT_PTR_EQ(poly_tensor_uop_physical(materialized)->src[0], poly_tensor_uop_physical(permuted));

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(tensor, contiguous_backward_constructs_exact_barrier_and_gradient) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  int64_t shape[] = {3};
  PolyTensor *source =
      poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_CPU);
  PolyTensor *result = poly_tensor_contiguous_backward(ctx, source);
  ASSERT_NOT_NULL(source);
  ASSERT_NOT_NULL(result);
  ASSERT_INT_EQ(poly_tensor_uop_logical(result)->op, POLY_OP_CONTIGUOUS_BACKWARD);
  ASSERT_PTR_EQ(
      poly_tensor_uop_logical(result)->src[0], poly_tensor_uop_logical(source)
  );
  ASSERT_INT_EQ(poly_tensor_uop_physical(result)->op, POLY_OP_CONTIGUOUS_BACKWARD);
  ASSERT_PTR_EQ(
      poly_tensor_uop_physical(result)->src[0], poly_tensor_uop_physical(source)
  );

  PolyUOp *grad = poly_grad(
      ctx, poly_tensor_uop_physical(result), poly_tensor_uop_physical(source)
  );
  ASSERT_NOT_NULL(grad);
  int n_topo = 0, contiguous_count = 0;
  PolyUOp **topo = poly_toposort_alloc(ctx, grad, &n_topo);
  ASSERT_NOT_NULL(topo);
  for (int i = 0; i < n_topo; i++) contiguous_count += topo[i]->op == POLY_OP_CONTIGUOUS;
  ASSERT_INT_EQ(contiguous_count, 1);
  free(topo);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(tensor, binary_alu_broadcasts_before_construction_like_tinygrad) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  int64_t value_shape[] = {4, 3};
  int64_t bias_shape[] = {3};
  PolyTensor *value =
      poly_tensor_empty(ctx, POLY_FLOAT32, value_shape, 2, POLY_DEVICE_CPU);
  PolyTensor *bias =
      poly_tensor_empty(ctx, POLY_FLOAT32, bias_shape, 1, POLY_DEVICE_CPU);
  PolyTensor *sum = poly_tensor_alu2(ctx, POLY_OP_ADD, value, bias);
  ASSERT_NOT_NULL(value);
  ASSERT_NOT_NULL(bias);
  ASSERT_NOT_NULL(sum);

  /* Pinned _broadcast_to left-pads [3] to [1,3], then expands to [4,3]
   * before the binary ALU (mixin/movement.py:116-128). */
  PolyUOp *physical = poly_tensor_uop_physical(sum);
  ASSERT_NOT_NULL(physical);
  ASSERT_INT_EQ(physical->op, POLY_OP_ADD);
  ASSERT_INT_EQ(physical->n_src, 2);
  ASSERT_INT_EQ(physical->src[1]->op, POLY_OP_EXPAND);
  ASSERT_INT_EQ(physical->src[1]->src[0]->op, POLY_OP_RESHAPE);
  ASSERT_PTR_EQ(
      physical->src[1]->src[0]->src[0],
      poly_tensor_uop_physical(bias)
  );
  PolyShape physical_shape = poly_uop_max_shape(ctx, physical);
  ASSERT_INT_EQ(physical_shape.ndim, 2);
  ASSERT_INT_EQ(physical_shape.dims[0], 4);
  ASSERT_INT_EQ(physical_shape.dims[1], 3);
  free(physical_shape.dims);

  float value_data[12];
  for (int i = 0; i < 12; i++) value_data[i] = (float)i;
  float bias_data[] = {10.0f, 20.0f, 30.0f};
  PolyUOp *value_buffer =
      (PolyUOp *)poly_uop_get_buffer_identity(poly_tensor_uop_physical(value));
  PolyUOp *bias_buffer =
      (PolyUOp *)poly_uop_get_buffer_identity(poly_tensor_uop_physical(bias));
  ASSERT_NOT_NULL(value_buffer);
  ASSERT_NOT_NULL(bias_buffer);
  ASSERT_INT_EQ(poly_buffer_write(ctx, value_buffer, value_data, sizeof(value_data)), 0);
  ASSERT_INT_EQ(poly_buffer_write(ctx, bias_buffer, bias_data, sizeof(bias_data)), 0);

  PolyTensor *realized = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &sum, 1, &realized), 0);
  ASSERT_NOT_NULL(realized);
  PolyUOp *result_buffer =
      (PolyUOp *)poly_uop_get_buffer_identity(poly_tensor_uop_physical(realized));
  float result[12] = {0};
  ASSERT_NOT_NULL(result_buffer);
  ASSERT_INT_EQ(poly_buffer_read(ctx, result_buffer, result, sizeof(result)), 0);
  for (int row = 0; row < 4; row++)
    for (int col = 0; col < 3; col++)
      ASSERT_FLOAT_NEAR(
          result[row * 3 + col], value_data[row * 3 + col] + bias_data[col], 4, 1e-6
      );

  poly_ctx_destroy(ctx);
  PASS();
}

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
  PolyTestBufferView bindings[] = {
      POLY_TEST_HOST_VIEW(out_buf, out),
      POLY_TEST_HOST_VIEW(a_buf, in),
  };
  ASSERT_INT_EQ(poly_test_realize_buffer_views(ctx, sink, bindings, 2), 0);
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
  PolyTestBufferView bindings[] = {
      POLY_TEST_HOST_VIEW(out_buf, out),
      POLY_TEST_HOST_VIEW(a_buf, in),
  };
  ASSERT_INT_EQ(poly_test_realize_buffer_views(ctx, sink, bindings, 2), 0);
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
  PolyTestBufferView bindings[] = {
      POLY_TEST_HOST_VIEW(out_buf, out),
      POLY_TEST_HOST_VIEW(a_buf, in),
  };
  ASSERT_INT_EQ(poly_test_realize_buffer_views(ctx, sink, bindings, 2), 0);
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
/*  bound via POLY_TEST_HOST_VIEW -- no const-registry path involved.          */
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

TEST(pe, conv2d_padded_3x3_4x4_devectorize_regression) {
  PolyCtx *ctx = poly_ctx_new();

  /* tinygrad:
   * Tensor(arange(16).reshape(1,1,4,4)).conv2d(ones(1,1,3,3), padding=1)
   * -> [[[[10,18,24,18],[27,45,54,39],[51,81,90,63],[42,66,72,50]]]]
   * This shape creates a 144-lane vectorized INDEX in late codegen. */
  int64_t x_shape[4] = {1, 1, 4, 4};
  int64_t w_shape[4] = {1, 1, 3, 3};
  PolyUOp *x = make_buf(ctx, x_shape, 4);
  PolyUOp *w = make_buf(ctx, w_shape, 4);
  int64_t padding[1] = {1};
  PolyUOp *out_val = poly_conv2d(ctx, x, w, NULL, 1, NULL, NULL, padding, 1);
  ASSERT_NOT_NULL(out_val);

  PolyUOp *out_buf = poly_buffer_f32(ctx, 16);
  float x_data[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  float w_data[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  float out_data[16] = {0};
  PolyUOp *leaves[] = {base_buf(x), base_buf(w)};
  float *ld[] = {x_data, w_data};
  ASSERT_INT_EQ(realize_uop(ctx, out_val, out_buf, out_data, leaves, ld, 2), 0);

  float expected[16] = {10, 18, 24, 18, 27, 45, 54, 39, 51, 81, 90, 63, 42, 66, 72, 50};
  for (int i = 0; i < 16; i++)
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

TEST(pe, arange_empty_uses_pinned_pure_full_topology) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  int i32 = poly_dtype_id_by_name("int32");
  int f32 = poly_dtype_id_by_name("float32");
  PolyUOp *empty_i = poly_arange_int_by_id(ctx, 0, 0, -1, i32);
  PolyUOp *empty_f = poly_arange_float_by_id(ctx, 0.0, 0.0, -1.0, f32);
  PolyUOp *roots[2] = {empty_i, empty_f};
  for (int i = 0; i < 2; i++) {
    PolyUOp *root = roots[i];
    ASSERT_NOT_NULL(root);
    ASSERT_EQ(root->op, POLY_OP_EXPAND);
    ASSERT_INT_EQ(poly_uop_ndim(ctx, root), 1);
    ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, root)[0], 0);
    ASSERT_INT_EQ(root->n_src, 2);
    ASSERT_EQ(root->src[0]->op, POLY_OP_RESHAPE);
    ASSERT_EQ(root->src[0]->src[0]->op, POLY_OP_CONST);
  }
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, reduce_identity_element_is_typed_like_pinned) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *i32 = poly_identity_element(ctx, POLY_OP_MAX, POLY_INT32);
  PolyUOp *i64 = poly_identity_element(ctx, POLY_OP_MAX, POLY_INT64);
  PolyUOp *u32 = poly_identity_element(ctx, POLY_OP_MAX, POLY_UINT32);
  PolyUOp *f32 = poly_identity_element(ctx, POLY_OP_MAX, POLY_FLOAT32);
  PolyUOp *mul = poly_identity_element(ctx, POLY_OP_MUL, POLY_INT32);
  PolyUOp *add = poly_identity_element(ctx, POLY_OP_ADD, POLY_FLOAT32);
  ASSERT_NOT_NULL(i32);
  ASSERT_NOT_NULL(i64);
  ASSERT_NOT_NULL(u32);
  ASSERT_NOT_NULL(f32);
  ASSERT_NOT_NULL(mul);
  ASSERT_NOT_NULL(add);
  ASSERT_INT_EQ(i32->arg.kind, POLY_ARG_INT);
  ASSERT_INT_EQ(i32->arg.i, INT32_MIN);
  ASSERT_INT_EQ(i64->arg.i, INT64_MIN);
  ASSERT_INT_EQ(u32->arg.i, 0);
  ASSERT_INT_EQ(f32->arg.kind, POLY_ARG_FLOAT);
  ASSERT_TRUE(isinf(f32->arg.f) && f32->arg.f < 0.0);
  ASSERT_INT_EQ(mul->arg.i, 1);
  ASSERT_INT_EQ(add->arg.kind, POLY_ARG_FLOAT);
  ASSERT_FLOAT_EQ(add->arg.f, 0.0, 0.0);
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

TEST(pe, zero_dimension_broadcast_matches_tinygrad) {
  /* Pinned _broadcast_shape chooses zero when either aligned dimension is
   * zero (uop/ops.py:63-71). This is the core topology reached by
   * Tensor.zeros(5, 0, 3).triu(). */
  PolyCtx *ctx = poly_ctx_new();
  int f32 = poly_dtype_id_by_name("float32");
  int64_t mask_shape[] = {0, 3};
  int64_t value_shape[] = {5, 0, 3};
  PolyUOp *mask_value = poly_full_float_by_id(ctx, mask_shape, 2, 1.0, f32);
  PolyUOp *mask = poly_eq(ctx, mask_value, mask_value);
  PolyUOp *value = poly_full_float_by_id(ctx, value_shape, 3, 2.0, f32);
  PolyUOp *out = poly_where_op(ctx, mask, value, poly_const_float(ctx, 0.0));

  ASSERT_NOT_NULL(out);
  ASSERT_INT_EQ(mask_value->op, POLY_OP_EXPAND);
  ASSERT_INT_EQ(mask_value->src[0]->op, POLY_OP_RESHAPE);
  ASSERT_INT_EQ(mask_value->src[0]->src[0]->op, POLY_OP_CONST);
  ASSERT_INT_EQ(out->op, POLY_OP_WHERE);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, out), 3);
  const int64_t *shape = poly_uop_max_shape_dims(ctx, out);
  ASSERT_NOT_NULL(shape);
  ASSERT_INT_EQ(shape[0], 5);
  ASSERT_INT_EQ(shape[1], 0);
  ASSERT_INT_EQ(shape[2], 3);
  ASSERT_INT_EQ(out->src[0]->op, POLY_OP_EXPAND);
  ASSERT_INT_EQ(out->src[2]->op, POLY_OP_EXPAND);

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

TEST(pe, triu_tril_batched_last_two_dims_match_tinygrad) {
  PolyCtx *ctx = poly_ctx_new();
  const int64_t shape[3] = {2, 3, 3};
  PolyUOp *in = make_buf(ctx, shape, 3);

  PolyUOp *upper = poly_triu(ctx, in, 0);
  PolyUOp *lower = poly_tril(ctx, in, 1);
  ASSERT_NOT_NULL(upper);
  ASSERT_NOT_NULL(lower);

  float in_d[18];
  for (int i = 0; i < 18; i++)
    in_d[i] = (float)(i + 1);
  PolyUOp *leaves[] = {base_buf(in)};
  float *ld[] = {in_d};

  float got_upper[18] = {0};
  float got_lower[18] = {0};
  ASSERT_INT_EQ(realize_uop(ctx, upper, poly_buffer_f32(ctx, 18), got_upper, leaves, ld, 1), 0);
  ASSERT_INT_EQ(realize_uop(ctx, lower, poly_buffer_f32(ctx, 18), got_lower, leaves, ld, 1), 0);

  float expect_upper[18] = {
      1, 2, 3, 0, 5, 6, 0, 0, 9,
      10, 11, 12, 0, 14, 15, 0, 0, 18,
  };
  float expect_lower[18] = {
      1, 2, 0, 4, 5, 6, 7, 8, 9,
      10, 11, 0, 13, 14, 15, 16, 17, 18,
  };
  for (int i = 0; i < 18; i++) {
    ASSERT_FLOAT_EQ(got_upper[i], expect_upper[i], 1e-5);
    ASSERT_FLOAT_EQ(got_lower[i], expect_lower[i], 1e-5);
  }

  const int64_t zero_shape[3] = {5, 0, 3};
  PolyUOp *zero_in = make_buf(ctx, zero_shape, 3);
  ASSERT_NOT_NULL(poly_triu(ctx, zero_in, 0));
  ASSERT_NOT_NULL(poly_tril(ctx, zero_in, 0));

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
    free(lin);
  }

  poly_ctx_destroy(ctx);
  PASS();
}
