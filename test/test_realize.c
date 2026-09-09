/*
 * test_realize.c — Tests for graph-driven poly_realize.
 *
 * Verifies that poly_realize materializes top-level value UOps through the
 * ctx->buffers side table with no external bindings array.
 */

#include "test_harness.h"
#include "../src/ctx.h"
#include "../src/engine/jit.h"
#include "../src/engine/realize.h"
#include "../src/engine/schedule.h"
#include "../src/device.h"
#include "../src/frontend.h"
#include "../src/frontend_internal.h"
#include "../src/polygrad.h"
#include "../src/tensor.h"
#include "../src/utils.h"
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static PolyBuffer *realized_buffer(PolyCtx *ctx, PolyUOp *realized) {
  const PolyUOp *buf_uop = poly_uop_get_buffer_identity(realized);
  return buf_uop ? poly_buffer_get(ctx, (PolyUOp *)buf_uop) : NULL;
}

/* Raw physical-UOp fixture. Device policy must already be encoded in the UOp;
 * it never infers placement from the Tensor wrapper or buffer residency. */
static PolyTensor *physical_tensor_from_uop(
    PolyCtx *ctx,
    PolyUOp *physical,
    PolyTensorRole role,
    PolyDevice device
) {
  if (!ctx || !physical || poly_tensor_root_has_unplaced_buffer(ctx, physical)) return NULL;
  return poly_tensor_create_with_roots(ctx, NULL, physical, role, device);
}

static PolyTensor *initialized_f32_tensor(
    PolyCtx *ctx,
    const int64_t *shape,
    int ndim,
    float *data,
    PolyDevice device,
    PolyUOp **out_buffer
) {
  int64_t numel = 1;
  for (int i = 0; i < ndim; i++)
    numel *= shape[i];
  if (numel < 0) return NULL;
  size_t nbytes = (size_t)numel * sizeof(float);
  PolyTensor *tensor = NULL;
  if (device == POLY_DEVICE_HOST) {
    /* Internal placement fixture: expose Tinygrad _frompy's staging BUFFER,
     * not the public Tensor constructor's default-device COPY. */
    PolyUOp *physical = poly_buffer_from_host(
        ctx, data, nbytes, poly_dtype_id_by_name("float32"), (int64_t *)shape, ndim
    );
    PolyUOp *logical = poly_uop_new_logical_buffer(ctx, POLY_FLOAT32, numel);
    if (logical && ndim != 1) logical = poly_reshape(ctx, logical, (int64_t *)shape, ndim);
    if (logical && physical)
      tensor = poly_tensor_create_with_roots(
          ctx, logical, physical, POLY_TENSOR_VALUE, POLY_DEVICE_HOST
      );
  } else {
    tensor = poly_tensor_empty(ctx, POLY_FLOAT32, shape, ndim, device);
  }
  PolyUOp *buffer =
      tensor ? (PolyUOp *)poly_uop_get_buffer_identity(poly_tensor_uop_physical(tensor)) : NULL;
  if (!buffer) return NULL;
  if (device != POLY_DEVICE_HOST) poly_buffer_set(ctx, buffer, data, nbytes, device);
  if (out_buffer) *out_buffer = buffer;
  return tensor;
}

static PolyTensor *initialized_i32_tensor8(PolyCtx *ctx, int start, PolyUOp **out_buffer) {
  int64_t shape[] = {8};
  PolyTensor *tensor = poly_tensor_empty(ctx, POLY_INT32, shape, 1, POLY_DEVICE_CPU);
  PolyUOp *buffer =
      tensor ? (PolyUOp *)poly_uop_get_buffer_identity(poly_tensor_uop_physical(tensor)) : NULL;
  if (!buffer) return NULL;
  int32_t values[8];
  for (int i = 0; i < 8; i++)
    values[i] = start + i;
  if (poly_buffer_allocate(ctx, buffer, POLY_DEVICE_CPU) != 0 ||
      poly_buffer_copyin(ctx, buffer, values, sizeof(values)) != 0)
    return NULL;
  if (out_buffer) *out_buffer = buffer;
  return tensor;
}

TEST(realize, virtual_weak_realize_noop_but_scheduling_requires_concrete_cast) {
  /* Tensor.realize filters UOp.is_virtual; linear_with_vars rejects a direct
   * weak storage request. Do not conflate the two public boundaries. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  ASSERT_INT_EQ(poly_ctx_set_logical_policy(ctx, POLY_LOGICAL_UNTIL_REALIZE), 0);
  poly_ctx_set_preferred_device(ctx, POLY_DEVICE_CPU);
  int64_t shape[] = {2};
  PolyTensor *base = poly_tensor_full_int_by_id(
      ctx, shape, 1, 2, poly_dtype_id_by_name("int32"), POLY_DEVICE_CPU, false, true
  );
  PolyTensor *exponent =
      poly_tensor_const_float_by_id(ctx, 2.0, poly_dtype_id_by_name("weakfloat"), POLY_DEVICE_CPU);
  PolyTensor *weak = poly_tensor_alu2(ctx, POLY_OP_POW, base, exponent);
  ASSERT_NOT_NULL(weak);
  ASSERT_TRUE(poly_dtype_eq(poly_tensor_uop_physical(weak)->dtype, POLY_WEAKFLOAT));
  PolyUOp *weak_logical = poly_tensor_uop_logical(weak);
  ASSERT_NOT_NULL(weak_logical);

  PolyTensor *realized = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &weak, 1, &realized), 0);
  ASSERT_PTR_EQ(realized, weak);
  PolyUOp *weak_root = poly_tensor_uop_physical(weak), *output = NULL;
  PolyVarBinding *bindings = NULL;
  int n_bindings = 0;
  ASSERT_TRUE(poly_linear_with_vars(ctx, &weak_root, 1, &output, &bindings, &n_bindings) == NULL);
  ASSERT_TRUE(output == NULL);
  ASSERT_PTR_EQ(poly_tensor_uop_logical(weak), weak_logical);
  ASSERT_INT_EQ(poly_tensor_logical_state(weak), POLY_LOGICAL_AVAILABLE);

  PolyTensor *strong = poly_tensor_cast_by_id(ctx, weak, poly_dtype_id_by_name("float32"));
  ASSERT_NOT_NULL(strong);
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &strong, 1, &realized), 0);
  const PolyUOp *identity = poly_uop_get_buffer_identity(poly_tensor_uop_physical(realized));
  ASSERT_NOT_NULL(identity);
  float actual[2] = {0};
  ASSERT_INT_EQ(poly_buffer_read(ctx, (PolyUOp *)identity, actual, sizeof(actual)), 0);
  ASSERT_FLOAT_EQ(actual[0], 4.0f, 1e-6);
  ASSERT_FLOAT_EQ(actual[1], 4.0f, 1e-6);

  poly_ctx_destroy(ctx);
  PASS();
}

static PolyTensor *sharded_i32_tensor(PolyCtx *ctx, PolyTensor *input) {
  if (!ctx || !input) return NULL;
  int64_t shape[] = {4, 2};
  PolyUOp *logical = poly_reshape(ctx, poly_tensor_uop_logical(input), shape, 2);
  PolyUOp *current = poly_reshape(ctx, poly_tensor_uop_physical(input), shape, 2);
  const char *names[] = {"CPU", "CPU:1"};
  PolyUOp *tuple = poly_device_uop_from_names(ctx, names, 2);
  PolyUOp *broadcast = poly_copy_to_device_uop(ctx, current, tuple);
  PolyUOp *device_num = poly_uop_range(ctx, 2, -1, POLY_AXIS_DEVICE);
  PolyUOp *two = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(2));
  PolyUOp *starts[] = {
      poly_alu2(ctx, POLY_OP_MUL, device_num, two),
      poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(0)),
  };
  PolyUOp *sizes[] = {two, two};
  PolyUOp *local = poly_shrink_uop(ctx, broadcast, starts, sizes, 2);
  int64_t axes[] = {0};
  PolyUOp *ranges[] = {device_num};
  PolyUOp *physical = poly_unshard(ctx, local, axes, ranges, 1);
  if (!logical || !tuple || !broadcast || !device_num || !two || !local || !physical) return NULL;
  return poly_tensor_create_with_roots(ctx, logical, physical, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
}

#define READ_REALIZED_F32(ctx, realized, out, count)                                               \
  do {                                                                                             \
    const PolyUOp *_read_buf = poly_uop_get_buffer_identity(realized);                             \
    ASSERT_NOT_NULL(_read_buf);                                                                    \
    ASSERT_INT_EQ(                                                                                 \
        poly_buffer_read((ctx), (PolyUOp *)_read_buf, (out), (count) * sizeof(float)), 0           \
    );                                                                                             \
  } while (0)

static int count_root_ops(PolyCtx *ctx, PolyUOp *root, PolyOps op) {
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, root, &n_topo);
  int count = 0;
  for (int i = 0; i < n_topo; i++) {
    if (topo[i] && topo[i]->op == op) count++;
  }
  return count;
}

static int count_bound_vars(PolyCtx *ctx, PolyUOp *root) {
  int n_topo = 0, count = 0;
  PolyUOp **topo = poly_toposort_alloc(ctx, root, &n_topo);
  if (!topo) return -1;
  for (int i = 0; i < n_topo; i++)
    count += poly_uop_is_bound_var(topo[i]);
  poly_toposort_free(topo);
  return count;
}

static bool realize_next_permutation(int *items, int n) {
  int i = n - 2;
  while (i >= 0 && items[i] >= items[i + 1])
    i--;
  if (i < 0) return false;
  int j = n - 1;
  while (items[j] <= items[i])
    j--;
  int tmp = items[i];
  items[i] = items[j];
  items[j] = tmp;
  for (int lo = i + 1, hi = n - 1; lo < hi; lo++, hi--) {
    tmp = items[lo];
    items[lo] = items[hi];
    items[hi] = tmp;
  }
  return true;
}

static bool is_shaped_value_param(PolyUOp *u) {
  return u && u->op == POLY_OP_PARAM && u->arg.kind == POLY_ARG_PARAM && u->arg.param &&
         !u->arg.param->name && u->n_src == 1 && u->src[0] &&
         (u->src[0]->op == POLY_OP_STACK || poly_dtype_is_int(u->src[0]->dtype));
}

static int count_shaped_value_params(PolyCtx *ctx, PolyUOp *root) {
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, root, &n_topo);
  int count = 0;
  for (int i = 0; i < n_topo; i++)
    if (is_shaped_value_param(topo[i])) count++;
  return count;
}

static PolyTensor *custom_add_tensor(
    PolyCtx *ctx,
    PolyTensor *a,
    PolyTensor *b,
    PolyUOp **out_buf
) {
  int64_t shape[] = {4};
  PolyTensor *c = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_CPU);
  if (!c) return NULL;
  if (out_buf) *out_buf = poly_tensor_uop_physical(c);
  PolyUOp *pc = poly_uop_placeholder_like(ctx, poly_tensor_uop_physical(c), 0);
  PolyUOp *pa = poly_uop_placeholder_like(ctx, poly_tensor_uop(a), 1);
  PolyUOp *pb = poly_uop_placeholder_like(ctx, poly_tensor_uop(b), 2);
  PolyUOp *r = poly_uop_range(ctx, 4, 0, POLY_AXIS_WEAK);
  PolyUOp *idxs[1] = {r};
  PolyUOp *ci = poly_uop_index(ctx, pc, idxs, 1);
  PolyUOp *ai = poly_uop_index(ctx, pa, idxs, 1);
  PolyUOp *bi = poly_uop_index(ctx, pb, idxs, 1);
  PolyUOp *av = poly_uop_load(ctx, ai);
  PolyUOp *bv = poly_uop_load(ctx, bi);
  PolyUOp *sum = poly_alu2(ctx, POLY_OP_ADD, av, bv);
  PolyUOp *store = poly_uop_store(ctx, ci, sum);
  PolyUOp *end = poly_uop_end(ctx, store, &r, 1);
  PolyUOp *sink = poly_uop_sink_ex(ctx, &end, 1, "custom_add_4", 1);
  PolyTensor *inputs[3] = {c, a, b};
  PolyTensor *outputs[3] = {0};
  return sink && poly_tensor_custom_kernel(ctx, sink, inputs, 3, 0, outputs) == 0 ? outputs[0]
                                                                                  : NULL;
}

static PolyTensor *custom_summary_tensor(PolyCtx *ctx, PolyTensor *a, PolyUOp **out_buf) {
  int64_t shape[] = {1};
  PolyTensor *out = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_CPU);
  if (!out) return NULL;
  if (out_buf) *out_buf = poly_tensor_uop_physical(out);
  PolyUOp *pout = poly_uop_placeholder_like(ctx, poly_tensor_uop_physical(out), 0);
  PolyUOp *pa = poly_uop_placeholder_like(ctx, poly_tensor_uop(a), 1);
  PolyUOp *r = poly_uop_range(ctx, 8, 0, POLY_AXIS_REDUCE);
  PolyUOp *zero_index = poly_const_int(ctx, 0);
  PolyUOp *zero = poly_const_typed(ctx, POLY_FLOAT32, 0.0);
  PolyUOp *out_index[] = {zero_index};
  PolyUOp *initialized = poly_uop_set(ctx, poly_uop_index(ctx, pout, out_index, 1), zero, NULL, 0);
  PolyUOp *after_range = poly_uop_after(ctx, initialized, r);
  PolyUOp *input_index[] = {r};
  PolyUOp *sum = poly_alu2(
      ctx, POLY_OP_ADD, poly_uop_index(ctx, after_range, out_index, 1),
      poly_uop_index(ctx, pa, input_index, 1)
  );
  PolyUOp *reduced = poly_uop_set(ctx, poly_uop_index(ctx, initialized, out_index, 1), sum, &r, 1);
  PolyUOp *sink = poly_uop_sink_ex(ctx, &reduced, 1, "custom_sum_8", 1);
  PolyTensor *inputs[2] = {out, a};
  PolyTensor *outputs[2] = {0};
  return sink && poly_tensor_custom_kernel(ctx, sink, inputs, 2, 0, outputs) == 0 ? outputs[0]
                                                                                  : NULL;
}

static void custom_multi_summary_tensors(
    PolyCtx *ctx,
    PolyTensor *a,
    PolyTensor *b,
    PolyTensor **out0_tensor,
    PolyTensor **out1_tensor,
    PolyUOp **out0_buf,
    PolyUOp **out1_buf
) {
  int64_t shape[] = {4};
  PolyTensor *out0 = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_CPU);
  PolyTensor *out1 = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_CPU);
  if (!out0 || !out1) return;
  if (out0_buf) *out0_buf = poly_tensor_uop_physical(out0);
  if (out1_buf) *out1_buf = poly_tensor_uop_physical(out1);
  PolyUOp *pout0 = poly_uop_placeholder_like(ctx, poly_tensor_uop_physical(out0), 0);
  PolyUOp *pout1 = poly_uop_placeholder_like(ctx, poly_tensor_uop_physical(out1), 1);
  PolyUOp *pa = poly_uop_placeholder_like(ctx, poly_tensor_uop(a), 2);
  PolyUOp *pb = poly_uop_placeholder_like(ctx, poly_tensor_uop(b), 3);
  PolyUOp *r = poly_uop_range(ctx, 4, 0, POLY_AXIS_WEAK);
  PolyUOp *index[] = {r};
  PolyUOp *av = poly_uop_index(ctx, pa, index, 1);
  PolyUOp *bv = poly_uop_index(ctx, pb, index, 1);
  PolyUOp *st0 = poly_uop_store(
      ctx, poly_uop_index(ctx, pout0, index, 1), poly_alu2(ctx, POLY_OP_ADD, av, bv)
  );
  PolyUOp *st1 = poly_uop_store(
      ctx, poly_uop_index(ctx, pout1, index, 1), poly_alu2(ctx, POLY_OP_MUL, av, bv)
  );
  PolyUOp *stores[] = {st0, st1};
  PolyUOp *group = poly_uop_group(ctx, stores, 2);
  PolyUOp *end = poly_uop_end(ctx, group, &r, 1);
  PolyUOp *sink = poly_uop_sink_ex(ctx, &end, 1, "custom_addmul_4", 1);
  PolyTensor *inputs[4] = {out0, out1, a, b};
  PolyTensor *outputs[4] = {0};
  if (!sink || poly_tensor_custom_kernel(ctx, sink, inputs, 4, 0, outputs) != 0) return;
  *out0_tensor = outputs[0];
  *out1_tensor = outputs[1];
}

TEST(realize, custom_call_copy_output_schedules_copy_call) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *c = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CPU);
  PolyUOp *a = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CPU);
  PolyUOp *b = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CPU);
  PolyUOp *pc = poly_uop_placeholder_like(ctx, c, 0);
  PolyUOp *pa = poly_uop_placeholder_like(ctx, a, 1);
  PolyUOp *pb = poly_uop_placeholder_like(ctx, b, 2);
  PolyUOp *r = poly_uop_range(ctx, 4, 0, POLY_AXIS_GLOBAL);
  PolyUOp *idxs[1] = {r};
  PolyUOp *ci = poly_uop_index(ctx, pc, idxs, 1);
  PolyUOp *ai = poly_uop_index(ctx, pa, idxs, 1);
  PolyUOp *bi = poly_uop_index(ctx, pb, idxs, 1);
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, ai, bi, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, ci, add, poly_arg_none());
  PolyUOp *end = poly_uop_end(ctx, store, &r, 1);
  /* Pinned lower_sink_to_linear leaves KernelInfo SINKs as custom kernels;
   * a plain SINK is instead recursively scheduled as a tensor function. */
  PolyUOp *sink = poly_uop_sink_ex(ctx, &end, 1, "custom_copy_source", 1);
  PolyUOp *args[3] = {c, a, b};
  PolyUOp *call = poly_uop_call(ctx, sink, args, 3);
  PolyUOp *after = poly_uop_after(ctx, c, call);
  PolyUOp *contig = poly_contiguous(ctx, after);
  PolyUOp *device = poly_device_uop(ctx, POLY_DEVICE_CUDA);
  PolyUOp *copy = poly_copy_to_device_uop(ctx, contig, device);

  PolyUOp *out = NULL;
  PolyUOp *callified = poly_transform_to_call(ctx, &copy, 1, &out);
  ASSERT_NOT_NULL(callified);
  /* Pinned transform_to_call always returns CALL(SINK(...), ...); only
   * create_linear_with_vars splits that body into ordered custom/COPY calls
   * (tinygrad/callify.py:203-221, schedule/__init__.py:21-68). */
  ASSERT_EQ(callified->op, POLY_OP_CALL);
  ASSERT_NOT_NULL(callified->src[0]);
  ASSERT_EQ(callified->src[0]->op, POLY_OP_SINK);
  ASSERT_NOT_NULL(out);
  ASSERT_TRUE(poly_uop_has_buffer_identity(out));
  const PolyUOp *out_identity = poly_uop_get_buffer_identity(out);
  ASSERT_NOT_NULL(out_identity);
  ASSERT_INT_EQ(poly_uop_device((PolyUOp *)out_identity), POLY_DEVICE_CUDA);
  ASSERT_FALSE(poly_buffer_is_allocated(ctx, (PolyUOp *)out_identity));
  ASSERT_TRUE(poly_buffer_get(ctx, (PolyUOp *)out_identity) == NULL);
  ASSERT_FALSE(poly_buffer_is_allocated(ctx, c));

  PolyUOp *scheduled_out = NULL;
  PolyUOp *sched = poly_test_linear_values(ctx, &copy, 1, &scheduled_out);
  ASSERT_NOT_NULL(sched);
  ASSERT_NOT_NULL(scheduled_out);
  ASSERT_INT_EQ(sched->n_src, 2);
  ASSERT_FALSE(poly_test_linear_call_is_copy(sched, 0));
  ASSERT_TRUE(poly_test_linear_call_is_copy(sched, 1));
  ASSERT_INT_EQ(count_root_ops(ctx, poly_test_linear_call_body(sched, 1), POLY_OP_CALL), 0);
  ASSERT_INT_EQ(count_root_ops(ctx, poly_test_linear_call_body(sched, 1), POLY_OP_AFTER), 0);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, callification_defers_output_allocation_until_execution) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  poly_ctx_set_preferred_device(ctx, POLY_DEVICE_CPU);

  PolyUOp *src = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CPU);
  float src_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  poly_buffer_set(ctx, src, src_data, sizeof(src_data), POLY_DEVICE_CPU);
  PolyUOp *value = poly_alu2(ctx, POLY_OP_ADD, src, poly_const_float(ctx, 1.0));
  PolyUOp *out = NULL;
  PolyUOp *sched = poly_test_linear_values(ctx, &value, 1, &out);
  ASSERT_NOT_NULL(sched);
  ASSERT_INT_EQ(sched->n_src, 1);
  ASSERT_FALSE(poly_test_linear_call_is_copy(sched, 0));
  ASSERT_NOT_NULL(out);
  ASSERT_EQ(out->op, POLY_OP_BUFFER);
  const PolyUOp *identity = poly_uop_get_buffer_identity(out);
  ASSERT_NOT_NULL(identity);
  ASSERT_PTR_EQ(identity, out);
  ASSERT_TRUE(poly_buffer_get(ctx, (PolyUOp *)identity) == NULL);
  ASSERT_FALSE(poly_buffer_is_allocated(ctx, (PolyUOp *)identity));
  PolyCtxStats stats = {0};
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &stats), 0);
  ASSERT_TRUE(stats.mem_used == 0);

  ASSERT_INT_EQ(poly_run_linear(ctx, sched, NULL, 0, NULL, 0, true, false, false), 0);
  PolyBuffer *storage = poly_buffer_get(ctx, (PolyUOp *)identity);
  ASSERT_NOT_NULL(storage);
  ASSERT_TRUE(storage->valid);
  ASSERT_TRUE(poly_buffer_is_allocated(ctx, (PolyUOp *)identity));
  float values[4] = {0};
  ASSERT_INT_EQ(poly_buffer_read(ctx, (PolyUOp *)identity, values, sizeof(values)), 0);
  for (int i = 0; i < 4; i++)
    ASSERT_FLOAT_EQ(values[i], (float)(i + 2), 1e-5f);
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &stats), 0);
  ASSERT_TRUE(stats.mem_used == sizeof(values));

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, tensor_to_device_stores_tinygrad_copy_device_graph) {
  PolyCtx *ctx = poly_ctx_new();
  PolyTensor *a_tensor = poly_tensor_empty(ctx, POLY_FLOAT32, (int64_t[]){4}, 1, POLY_DEVICE_CPU);
  PolyTensor *one =
      poly_tensor_const_float_by_id(ctx, 1.0, poly_dtype_id_by_name("float32"), POLY_DEVICE_CPU);
  PolyTensor *base = poly_tensor_alu2(ctx, POLY_OP_ADD, a_tensor, one);
  PolyTensor *cuda_tensor = poly_tensor_to_device(ctx, base, POLY_DEVICE_CUDA);
  PolyTensor *cpu_tensor = poly_tensor_to_device(ctx, base, POLY_DEVICE_CPU);
  PolyUOp *a = a_tensor ? poly_tensor_uop_physical(a_tensor) : NULL;
  PolyUOp *x = base ? poly_tensor_uop_logical(base) : NULL;
  ASSERT_NOT_NULL(a_tensor);
  ASSERT_NOT_NULL(one);
  ASSERT_NOT_NULL(base);
  ASSERT_NOT_NULL(cuda_tensor);
  ASSERT_NOT_NULL(cpu_tensor);
  ASSERT_PTR_EQ(poly_tensor_uop_logical(cuda_tensor), x);
  ASSERT_NOT_NULL(poly_tensor_uop_physical(cuda_tensor));
  ASSERT_PTR_EQ(poly_tensor_uop(cuda_tensor), poly_tensor_uop_physical(cuda_tensor));
  ASSERT_INT_EQ(cuda_tensor->role, POLY_TENSOR_PLACE);
  ASSERT_INT_EQ(poly_tensor_device(cuda_tensor), POLY_DEVICE_CUDA);

  PolyUOp *physical = poly_tensor_uop_physical(cuda_tensor);
  ASSERT_NOT_NULL(physical);
  /* tinygrad tensor.py:327-335 stores copy_to_device immediately. */
  ASSERT_PTR_EQ(physical, poly_tensor_uop_physical(cuda_tensor));
  ASSERT_INT_EQ(physical->op, POLY_OP_COPY);
  ASSERT_INT_EQ(poly_uop_device(physical), POLY_DEVICE_CUDA);
  ASSERT_INT_EQ(count_root_ops(ctx, physical, POLY_OP_COPY), 1);
  ASSERT_INT_EQ(physical->n_src, 1);

  PolyUOp *compute = physical->src[0];
  ASSERT_NOT_NULL(compute);
  ASSERT_INT_EQ(compute->op, POLY_OP_ADD);
  ASSERT_INT_EQ(poly_uop_device(compute), POLY_DEVICE_CPU);

  ASSERT_PTR_EQ(compute->src[0], a);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, tensor_to_device_preserves_distinct_host_execution_target) {
  PolyCtx *ctx = poly_ctx_new();

  /* The source DEVICE, not wrapper metadata or the test runner's default,
   * determines whether Tensor.to needs a COPY. */
  PolyUOp *buf = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CPU);
  PolyTensor *cpu = physical_tensor_from_uop(ctx, buf, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  PolyTensor *interp = poly_tensor_to_device(ctx, cpu, POLY_DEVICE_INTERP);
  ASSERT_NOT_NULL(cpu);
  ASSERT_NOT_NULL(interp);
  ASSERT_PTR_NEQ(interp, cpu);
  ASSERT_INT_EQ(interp->role, POLY_TENSOR_PLACE);
  ASSERT_INT_EQ(poly_tensor_device(interp), POLY_DEVICE_INTERP);
  ASSERT_PTR_EQ(interp->source, cpu);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, tensor_clone_into_separates_logical_and_cross_device_physical_graphs) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *source_logical = poly_test_buffer(ctx, POLY_FLOAT32, 1);
  PolyUOp *source_physical = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 1, POLY_DEVICE_CPU);
  float source_data[] = {3.0f};
  poly_buffer_set(ctx, source_physical, source_data, sizeof(source_data), POLY_DEVICE_CPU);
  PolyTensor *source = poly_tensor_create_with_roots(
      ctx, source_logical, source_physical, POLY_TENSOR_VALUE, POLY_DEVICE_CPU
  );

  PolyUOp *target_logical = poly_test_buffer(ctx, POLY_FLOAT32, 1);
  PolyUOp *target_physical = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 1, POLY_DEVICE_INTERP);
  PolyTensor *target = poly_tensor_create_with_roots(
      ctx, target_logical, target_physical, POLY_TENSOR_VALUE, POLY_DEVICE_INTERP
  );
  ASSERT_NOT_NULL(source);
  ASSERT_NOT_NULL(target);
  ASSERT_PTR_EQ(poly_tensor_clone_into(ctx, target, source), target);

  PolyUOp *logical = poly_tensor_uop_logical(target);
  PolyUOp *physical = poly_tensor_uop_physical(target);
  ASSERT_NOT_NULL(logical);
  ASSERT_NOT_NULL(physical);
  ASSERT_INT_EQ(logical->op, POLY_OP_AFTER);
  ASSERT_PTR_EQ(logical->src[0], target_logical);
  ASSERT_PTR_EQ(logical->src[1]->src[1], source_logical);
  ASSERT_INT_EQ(physical->op, POLY_OP_AFTER);
  ASSERT_PTR_EQ(physical->src[0], target_physical);
  ASSERT_INT_EQ(physical->src[1]->src[1]->op, POLY_OP_COPY);
  ASSERT_PTR_EQ(physical->src[1]->src[1]->src[0], source_physical);
  ASSERT_INT_EQ(poly_uop_device(physical->src[1]->src[1]), POLY_DEVICE_INTERP);

  PolyTensor *realized = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &target, 1, &realized), 0);
  float out[] = {0.0f};
  READ_REALIZED_F32(ctx, poly_tensor_uop(realized), out, 1);
  ASSERT_FLOAT_EQ(out[0], 3.0f, 1e-6f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, tensor_clone_into_same_device_uses_after_store_without_copy) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *source_logical = poly_test_logical_buffer(ctx, POLY_FLOAT32, 1);
  PolyUOp *source_physical = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 1, POLY_DEVICE_CPU);
  PolyTensor *source = poly_tensor_create_with_roots(
      ctx, source_logical, source_physical, POLY_TENSOR_VALUE, POLY_DEVICE_CPU
  );
  PolyUOp *target_logical = poly_test_logical_buffer(ctx, POLY_FLOAT32, 1);
  PolyUOp *target_physical = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 1, POLY_DEVICE_CPU);
  PolyTensor *target = poly_tensor_create_with_roots(
      ctx, target_logical, target_physical, POLY_TENSOR_VALUE, POLY_DEVICE_CPU
  );
  ASSERT_NOT_NULL(source);
  ASSERT_NOT_NULL(target);
  ASSERT_PTR_EQ(poly_tensor_clone_into(ctx, target, source), target);

  PolyUOp *physical = poly_tensor_uop_physical(target);
  ASSERT_NOT_NULL(physical);
  ASSERT_INT_EQ(physical->op, POLY_OP_AFTER);
  ASSERT_PTR_EQ(physical->src[0], target_physical);
  ASSERT_INT_EQ(physical->src[1]->op, POLY_OP_STORE);
  ASSERT_PTR_EQ(physical->src[1]->src[1], source_physical);
  ASSERT_INT_EQ(count_root_ops(ctx, physical, POLY_OP_COPY), 0);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, tensor_assign_chains_logical_and_physical_versions_separately) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *target_logical = poly_test_logical_buffer(ctx, POLY_FLOAT32, 1);
  PolyUOp *target_physical = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 1, POLY_DEVICE_CPU);
  float target_data[] = {1.0f};
  poly_buffer_set(ctx, target_physical, target_data, sizeof(target_data), POLY_DEVICE_CPU);
  PolyTensor *target = poly_tensor_create_with_roots(
      ctx, target_logical, target_physical, POLY_TENSOR_VALUE, POLY_DEVICE_CPU
  );
  ASSERT_NOT_NULL(target);

  PolyUOp *one = poly_const_float(ctx, 1.0f);
  PolyTensor *first_value = poly_tensor_create_with_roots(
      ctx, poly_add(ctx, target_logical, one), poly_add(ctx, target_physical, one),
      POLY_TENSOR_VALUE, POLY_DEVICE_CPU
  );
  ASSERT_NOT_NULL(first_value);
  ASSERT_PTR_EQ(poly_tensor_assign(ctx, target, first_value), target);
  PolyUOp *first_logical = poly_tensor_uop_logical(target);
  PolyUOp *first_physical = poly_tensor_uop_physical(target);
  ASSERT_NOT_NULL(first_logical);
  ASSERT_NOT_NULL(first_physical);
  ASSERT_INT_EQ(first_logical->op, POLY_OP_AFTER);
  ASSERT_INT_EQ(first_physical->op, POLY_OP_AFTER);
  ASSERT_PTR_EQ(first_logical->src[0], target_logical);
  ASSERT_PTR_EQ(first_physical->src[0], target_physical);
  ASSERT_INT_EQ(poly_uop_device(first_logical), POLY_DEVICE_AUTO);
  ASSERT_INT_EQ(poly_uop_device(first_physical), POLY_DEVICE_CPU);

  PolyTensor *second_value = poly_tensor_create_with_roots(
      ctx, poly_add(ctx, first_logical, one), poly_add(ctx, first_physical, one), POLY_TENSOR_VALUE,
      POLY_DEVICE_CPU
  );
  ASSERT_NOT_NULL(second_value);
  ASSERT_PTR_EQ(poly_tensor_assign(ctx, target, second_value), target);
  PolyUOp *second_logical = poly_tensor_uop_logical(target);
  PolyUOp *second_physical = poly_tensor_uop_physical(target);
  ASSERT_NOT_NULL(second_logical);
  ASSERT_NOT_NULL(second_physical);
  ASSERT_PTR_EQ(second_logical->src[0], first_logical);
  ASSERT_PTR_EQ(second_physical->src[0], first_physical);
  ASSERT_INT_EQ(poly_uop_device(second_logical), POLY_DEVICE_AUTO);
  ASSERT_INT_EQ(poly_uop_device(second_physical), POLY_DEVICE_CPU);

  PolyTensor *realized = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &target, 1, &realized), 0);
  float out[] = {0.0f};
  READ_REALIZED_F32(ctx, poly_tensor_uop(realized), out, 1);
  ASSERT_FLOAT_EQ(out[0], 3.0f, 1e-6f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, host_buffer_value_executes_permute_on_interp_without_device_instruction) {
  PolyCtx *ctx = poly_ctx_new();
  poly_ctx_set_preferred_device(ctx, POLY_DEVICE_INTERP);

  float data[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  int64_t flat_shape[] = {6};
  PolyTensor *host = poly_tensor_from_host(ctx, data, sizeof(data), POLY_FLOAT32, flat_shape, 1);
  PolyTensor *source = poly_tensor_to_device(ctx, host, POLY_DEVICE_INTERP);
  ASSERT_NOT_NULL(host);
  ASSERT_NOT_NULL(source);
  int64_t matrix_shape[] = {2, 3};
  int64_t order[] = {1, 0};
  PolyTensor *tensor = poly_tensor_contiguous(
      ctx, poly_tensor_permute(ctx, poly_tensor_reshape(ctx, source, matrix_shape, 2), order, 2)
  );
  ASSERT_NOT_NULL(tensor);

  PolyTensor *realized = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &tensor, 1, &realized), 0);
  ASSERT_PTR_EQ(realized, tensor);
  ASSERT_INT_EQ(poly_tensor_device(realized), POLY_DEVICE_INTERP);
  float out[6] = {0};
  READ_REALIZED_F32(ctx, poly_tensor_uop(realized), out, 6);
  ASSERT_FLOAT_EQ(out[0], 1.0f, 1e-5f);
  ASSERT_FLOAT_EQ(out[1], 4.0f, 1e-5f);
  ASSERT_FLOAT_EQ(out[2], 2.0f, 1e-5f);
  ASSERT_FLOAT_EQ(out[3], 5.0f, 1e-5f);
  ASSERT_FLOAT_EQ(out[4], 3.0f, 1e-5f);
  ASSERT_FLOAT_EQ(out[5], 6.0f, 1e-5f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, allocated_place_commits_value_and_reuses_target_across_assign) {
  PolyCtx *ctx = poly_ctx_new();

  float source_data[3] = {1.0f, 2.0f, 3.0f};
  int64_t shape[] = {3};
  PolyTensor *source = initialized_f32_tensor(ctx, shape, 1, source_data, POLY_DEVICE_HOST, NULL);
  PolyTensor *target = poly_tensor_to_device(ctx, source, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(source);
  ASSERT_NOT_NULL(target);
  ASSERT_INT_EQ(target->role, POLY_TENSOR_PLACE);
  ASSERT_PTR_EQ(target->source, source);

  PolyTensor *realized = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &target, 1, &realized), 0);
  ASSERT_PTR_EQ(realized, target);
  ASSERT_INT_EQ(target->role, POLY_TENSOR_VALUE);
  ASSERT_TRUE(target->source == NULL);
  const PolyUOp *target_identity = poly_uop_get_buffer_identity(poly_tensor_uop(target));
  ASSERT_NOT_NULL(target_identity);

  for (int pass = 0; pass < 2; pass++) {
    float value_data[3] = {5.0f + (float)pass, 6.0f + (float)pass, 7.0f + (float)pass};
    PolyTensor *value = initialized_f32_tensor(ctx, shape, 1, value_data, POLY_DEVICE_CPU, NULL);
    ASSERT_NOT_NULL(value);
    ASSERT_PTR_EQ(poly_tensor_assign(ctx, target, value), target);
    ASSERT_INT_EQ(poly_realize_tensors(ctx, &target, 1, &realized), 0);
    ASSERT_PTR_EQ(poly_uop_get_buffer_identity(poly_tensor_uop(target)), target_identity);
    float out[3] = {0};
    READ_REALIZED_F32(ctx, poly_tensor_uop(target), out, 3);
    ASSERT_FLOAT_EQ(out[0], value_data[0], 1e-5f);
    ASSERT_FLOAT_EQ(out[1], value_data[1], 1e-5f);
    ASSERT_FLOAT_EQ(out[2], value_data[2], 1e-5f);
  }

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, shared_allocator_place_copies_and_preserves_source_across_assign) {
  PolyCtx *ctx = poly_ctx_new();

  float source_data[3] = {1.0f, 2.0f, 3.0f};
  int64_t shape[] = {3};
  PolyTensor *source = initialized_f32_tensor(ctx, shape, 1, source_data, POLY_DEVICE_CPU, NULL);
  PolyTensor *target = poly_tensor_to_device(ctx, source, POLY_DEVICE_INTERP);
  ASSERT_NOT_NULL(source);
  ASSERT_NOT_NULL(target);
  ASSERT_INT_EQ(target->role, POLY_TENSOR_PLACE);
  ASSERT_PTR_EQ(target->source, source);

  PolyTensor *realized = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &target, 1, &realized), 0);
  ASSERT_PTR_EQ(realized, target);
  ASSERT_INT_EQ(target->role, POLY_TENSOR_VALUE);
  ASSERT_TRUE(target->source == NULL);
  const PolyUOp *source_identity = poly_uop_get_buffer_identity(poly_tensor_uop(source));
  const PolyUOp *target_identity = poly_uop_get_buffer_identity(poly_tensor_uop(target));
  ASSERT_NOT_NULL(source_identity);
  ASSERT_NOT_NULL(target_identity);
  ASSERT_PTR_NEQ(target_identity, source_identity);
  float copied_out[3] = {0};
  READ_REALIZED_F32(ctx, poly_tensor_uop(target), copied_out, 3);
  for (int i = 0; i < 3; i++)
    ASSERT_FLOAT_EQ(copied_out[i], source_data[i], 1e-5f);

  for (int pass = 0; pass < 2; pass++) {
    float value_data[3] = {5.0f + (float)pass, 6.0f + (float)pass, 7.0f + (float)pass};
    PolyTensor *value = initialized_f32_tensor(ctx, shape, 1, value_data, POLY_DEVICE_INTERP, NULL);
    ASSERT_NOT_NULL(value);
    ASSERT_PTR_EQ(poly_tensor_assign(ctx, target, value), target);
    ASSERT_INT_EQ(poly_realize_tensors(ctx, &target, 1, &realized), 0);
    ASSERT_PTR_EQ(poly_uop_get_buffer_identity(poly_tensor_uop(target)), target_identity);

    float source_out[3] = {0};
    float target_out[3] = {0};
    READ_REALIZED_F32(ctx, poly_tensor_uop(source), source_out, 3);
    READ_REALIZED_F32(ctx, poly_tensor_uop(target), target_out, 3);
    for (int i = 0; i < 3; i++) {
      ASSERT_FLOAT_EQ(source_out[i], source_data[i], 1e-5f);
      ASSERT_FLOAT_EQ(target_out[i], value_data[i], 1e-5f);
    }
  }

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, indirect_immutable_place_input_commits_realized_copy_once) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  float source_data[2] = {1.0f, 2.0f};
  int64_t shape[] = {2};
  PolyUOp *source_physical = NULL;
  PolyTensor *source =
      initialized_f32_tensor(ctx, shape, 1, source_data, POLY_DEVICE_HOST, &source_physical);
  PolyUOp *source_logical = poly_tensor_uop_logical(source);
  PolyTensor *placed = poly_tensor_to_device(ctx, source, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(source);
  ASSERT_NOT_NULL(placed);
  ASSERT_INT_EQ(placed->role, POLY_TENSOR_PLACE);
  ASSERT_PTR_EQ(poly_tensor_uop_logical(placed), source_logical);
  ASSERT_NOT_NULL(poly_tensor_uop_physical(placed));
  ASSERT_PTR_EQ(poly_tensor_uop(placed), poly_tensor_uop_physical(placed));

  /* tinygrad tensor.py:327-335 stores this creation COPY directly in
   * Tensor.uop. Polygrad retains the portable source separately but must use
   * the exact eager physical COPY for the first indirect consumer. */
  PolyUOp *projected = poly_tensor_uop_physical(placed);
  ASSERT_NOT_NULL(projected);
  ASSERT_PTR_EQ(projected, poly_tensor_uop_physical(placed));
  ASSERT_INT_EQ(projected->op, POLY_OP_COPY);
  ASSERT_INT_EQ(projected->n_src, 1);
  ASSERT_PTR_EQ(projected->src[0], source_physical);
  ASSERT_INT_EQ(poly_uop_device(projected), POLY_DEVICE_CPU);

  /* tinygrad tensor.py:128-136 applies the same op to its current UOp.
   * Preserve Polygrad's exportable twin without feeding the physical COPY
   * back into uop_logical. */
  PolyUOp *first_logical =
      poly_add(ctx, poly_tensor_uop_logical(placed), poly_const_float(ctx, 1.0f));
  PolyUOp *first_uop = poly_add(ctx, poly_tensor_uop(placed), poly_const_float(ctx, 1.0f));
  PolyTensor *first = poly_tensor_create_with_roots(
      ctx, first_logical, first_uop, POLY_TENSOR_VALUE, POLY_DEVICE_CPU
  );
  ASSERT_NOT_NULL(first);

  poly_ctx_reset_counters(ctx);
  PolyTensor *first_out = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &first, 1, &first_out), 0);
  ASSERT_PTR_EQ(first_out, first);

  float first_values[2] = {0};
  READ_REALIZED_F32(ctx, poly_tensor_uop(first), first_values, 2);
  ASSERT_FLOAT_EQ(first_values[0], 2.0f, 1e-6f);
  ASSERT_FLOAT_EQ(first_values[1], 3.0f, 1e-6f);
  ASSERT_INT_EQ(placed->role, POLY_TENSOR_VALUE);
  ASSERT_TRUE(placed->source == NULL);
  ASSERT_PTR_EQ(poly_tensor_uop_logical(placed), source_logical);
  const PolyUOp *placed_identity = poly_uop_get_buffer_identity(poly_tensor_uop(placed));
  ASSERT_NOT_NULL(placed_identity);
  ASSERT_PTR_NEQ(placed_identity, source_physical);
  ASSERT_INT_EQ(poly_uop_device(poly_tensor_uop(placed)), POLY_DEVICE_CPU);
  PolyBuffer *placed_storage = poly_buffer_get(ctx, (PolyUOp *)placed_identity);
  ASSERT_NOT_NULL(placed_storage);
  ASSERT_TRUE(placed_storage->valid);
  ASSERT_INT_EQ(placed_storage->device, POLY_DEVICE_CPU);

  PolyCtxStats after_first = {0};
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &after_first), 0);
  ASSERT_INT_EQ(after_first.kernel_count, 2);

  PolyUOp *second_logical =
      poly_add(ctx, poly_tensor_uop_logical(placed), poly_const_float(ctx, 2.0f));
  PolyUOp *second_uop =
      poly_add(ctx, poly_tensor_uop_physical(placed), poly_const_float(ctx, 2.0f));
  PolyTensor *second = poly_tensor_create_with_roots(
      ctx, second_logical, second_uop, POLY_TENSOR_VALUE, POLY_DEVICE_CPU
  );
  ASSERT_NOT_NULL(second);
  PolyUOp *second_out = NULL;
  PolyUOp *second_schedule = poly_test_linear_values(ctx, &second_uop, 1, &second_out);
  ASSERT_NOT_NULL(second_schedule);
  ASSERT_NOT_NULL(second_out);
  ASSERT_INT_EQ(second_schedule->n_src, 1);
  ASSERT_INT_EQ(
      count_root_ops(ctx, poly_test_linear_call_body(second_schedule, 0), POLY_OP_COPY), 0
  );
  ASSERT_INT_EQ(poly_run_linear(ctx, second_schedule, NULL, 0, NULL, 0, true, false, false), 0);

  float second_values[2] = {0};
  READ_REALIZED_F32(ctx, second_out, second_values, 2);
  ASSERT_FLOAT_EQ(second_values[0], 3.0f, 1e-6f);
  ASSERT_FLOAT_EQ(second_values[1], 4.0f, 1e-6f);
  PolyCtxStats after_second = {0};
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &after_second), 0);
  ASSERT_INT_EQ(after_second.kernel_count, after_first.kernel_count + 1);
  ASSERT_PTR_EQ(poly_uop_get_buffer_identity(poly_tensor_uop(placed)), placed_identity);
  float source_values[2] = {0};
  ASSERT_INT_EQ(poly_buffer_read(ctx, source_physical, source_values, sizeof(source_values)), 0);
  ASSERT_FLOAT_EQ(source_values[0], source_data[0], 1e-6f);
  ASSERT_FLOAT_EQ(source_values[1], source_data[1], 1e-6f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, indirect_place_assignment_commits_value_after_consumer_run) {
  PolyCtx *ctx = poly_ctx_new();

  float source_data[] = {2.0f};
  int64_t shape[] = {1};
  PolyTensor *source = initialized_f32_tensor(ctx, shape, 1, source_data, POLY_DEVICE_HOST, NULL);
  PolyTensor *target = poly_tensor_to_device(ctx, source, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(source);
  ASSERT_NOT_NULL(target);

  PolyUOp *incremented_logical =
      poly_add(ctx, poly_tensor_uop_logical(target), poly_const_float(ctx, 1.0f));
  PolyUOp *incremented = poly_add(ctx, poly_tensor_uop(target), poly_const_float(ctx, 1.0f));
  PolyTensor *value = poly_tensor_create_with_roots(
      ctx, incremented_logical, incremented, POLY_TENSOR_VALUE, POLY_DEVICE_CPU
  );
  ASSERT_NOT_NULL(value);
  ASSERT_PTR_EQ(poly_tensor_assign(ctx, target, value), target);
  ASSERT_INT_EQ(target->role, POLY_TENSOR_PLACE);
  ASSERT_PTR_EQ(target->source, source);

  PolyUOp *consumer_logical =
      poly_add(ctx, poly_tensor_uop_logical(target), poly_const_float(ctx, 5.0f));
  PolyUOp *consumer_uop = poly_add(ctx, poly_tensor_uop(target), poly_const_float(ctx, 5.0f));
  PolyTensor *consumer = poly_tensor_create_with_roots(
      ctx, consumer_logical, consumer_uop, POLY_TENSOR_VALUE, POLY_DEVICE_CPU
  );
  ASSERT_NOT_NULL(consumer);

  /* Keep an unrelated nested placement whose inherited current residency is
   * already valid on its final requested device. A post-run lifecycle commit
   * must be tied to a physical root changed by this callification pass, not to
   * a broad scan of all valid PLACE buffers in the context. */
  PolyUOp *other_logical_buf = poly_buffer_f32(ctx, 1);
  PolyUOp *other_logical = poly_add(ctx, other_logical_buf, poly_const_float(ctx, 1.0f));
  PolyUOp *other_physical = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 1, POLY_DEVICE_CPU);
  float other_data[] = {9.0f};
  poly_buffer_set(ctx, other_physical, other_data, sizeof(other_data), POLY_DEVICE_CPU);
  PolyTensor *other = poly_tensor_create_with_roots(
      ctx, other_logical, other_physical, POLY_TENSOR_VALUE, POLY_DEVICE_CPU
  );
  PolyTensor *other_interp = poly_tensor_to_device(ctx, other, POLY_DEVICE_INTERP);
  PolyTensor *other_cpu_again = poly_tensor_to_device(ctx, other_interp, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(other);
  ASSERT_NOT_NULL(other_interp);
  ASSERT_NOT_NULL(other_cpu_again);
  PolyUOp *other_roundtrip = poly_tensor_uop_physical(other_cpu_again);
  ASSERT_NOT_NULL(other_roundtrip);
  ASSERT_EQ(other_roundtrip->op, POLY_OP_COPY);
  ASSERT_PTR_EQ(other_roundtrip->src[0], poly_tensor_uop_physical(other_interp));
  ASSERT_EQ(other_roundtrip->src[0]->op, POLY_OP_COPY);
  ASSERT_PTR_EQ(other_roundtrip->src[0]->src[0], other_physical);
  ASSERT_INT_EQ(other_cpu_again->role, POLY_TENSOR_PLACE);

  poly_ctx_reset_counters(ctx);
  PolyTensor *consumer_out = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &consumer, 1, &consumer_out), 0);
  ASSERT_PTR_EQ(consumer_out, consumer);

  /* Callification realizes the assignment as a dependency of the consumer.
   * That successful indirect execution is the same PLACE completion boundary
   * as realizing the target directly: retain its concrete target storage and
   * do not let a later readback rebuild/replay the placement COPY + STORE. */
  ASSERT_INT_EQ(target->role, POLY_TENSOR_VALUE);
  ASSERT_TRUE(target->source == NULL);
  ASSERT_INT_EQ(other_cpu_again->role, POLY_TENSOR_PLACE);
  ASSERT_PTR_EQ(other_cpu_again->source, other_interp);
  ASSERT_PTR_EQ(poly_tensor_uop_physical(other_cpu_again), other_roundtrip);
  const PolyUOp *target_identity = poly_uop_get_buffer_identity(poly_tensor_uop(target));
  ASSERT_NOT_NULL(target_identity);
  PolyBuffer *target_storage = poly_buffer_get(ctx, (PolyUOp *)target_identity);
  ASSERT_NOT_NULL(target_storage);
  ASSERT_TRUE(target_storage->valid);
  ASSERT_INT_EQ(target_storage->device, POLY_DEVICE_CPU);

  float consumer_value = 0.0f;
  READ_REALIZED_F32(ctx, poly_tensor_uop(consumer), &consumer_value, 1);
  ASSERT_FLOAT_EQ(consumer_value, 8.0f, 1e-6f);

  PolyCtxStats before_rerealize;
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &before_rerealize), 0);
  PolyTensor *target_out = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &target, 1, &target_out), 0);
  ASSERT_PTR_EQ(target_out, target);
  PolyCtxStats after_rerealize;
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &after_rerealize), 0);
  ASSERT_TRUE(after_rerealize.kernel_count == before_rerealize.kernel_count);

  float target_value = 0.0f;
  READ_REALIZED_F32(ctx, poly_tensor_uop(target), &target_value, 1);
  ASSERT_FLOAT_EQ(target_value, 3.0f, 1e-6f);
  ASSERT_FLOAT_EQ(source_data[0], 2.0f, 1e-6f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, indirect_nested_place_assignments_commit_latest_value) {
  PolyCtx *ctx = poly_ctx_new();

  float source_data[] = {0.0f};
  int64_t shape[] = {1};
  PolyTensor *source = initialized_f32_tensor(ctx, shape, 1, source_data, POLY_DEVICE_HOST, NULL);
  PolyTensor *target = poly_tensor_to_device(ctx, source, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(source);
  ASSERT_NOT_NULL(target);

  PolyTensor *assignment_values[2] = {NULL, NULL};
  for (int assignment = 0; assignment < 2; assignment++) {
    PolyUOp *incremented_logical =
        poly_add(ctx, poly_tensor_uop_logical(target), poly_const_float(ctx, 8.0f));
    PolyUOp *incremented_physical =
        poly_add(ctx, poly_tensor_uop_physical(target), poly_const_float(ctx, 8.0f));
    PolyTensor *value = poly_tensor_create_with_roots(
        ctx, incremented_logical, incremented_physical, POLY_TENSOR_VALUE, POLY_DEVICE_CPU
    );
    ASSERT_NOT_NULL(value);
    assignment_values[assignment] = value;
    ASSERT_PTR_EQ(poly_tensor_assign(ctx, target, value), target);
  }
  ASSERT_INT_EQ(target->role, POLY_TENSOR_PLACE);
  ASSERT_INT_EQ(count_root_ops(ctx, poly_tensor_uop(target), POLY_OP_AFTER), 2);

  PolyUOp *placed = poly_tensor_uop_physical(target);
  ASSERT_NOT_NULL(placed);
  int n_placed = 0;
  PolyUOp **placed_topo = poly_toposort(ctx, placed, &n_placed);
  int placed_afters = 0;
  for (int i = 0; i < n_placed; i++) {
    PolyUOp *u = placed_topo[i];
    if (!u || u->op != POLY_OP_AFTER) continue;
    placed_afters++;
    ASSERT_INT_EQ(u->n_src, 2);
    ASSERT_EQ(u->src[1]->op, POLY_OP_STORE);
    ASSERT_TRUE(u->src[1]->n_src >= 2);
    ASSERT_PTR_EQ(u->src[1]->src[0], u->src[0]);
  }
  ASSERT_INT_EQ(placed_afters, 2);

  PolyUOp *consumer_logical =
      poly_add(ctx, poly_tensor_uop_logical(target), poly_const_float(ctx, 5.0f));
  PolyUOp *consumer_uop =
      poly_add(ctx, poly_tensor_uop_physical(target), poly_const_float(ctx, 5.0f));
  PolyTensor *consumer = poly_tensor_create_with_roots(
      ctx, consumer_logical, consumer_uop, POLY_TENSOR_VALUE, POLY_DEVICE_CPU
  );
  ASSERT_NOT_NULL(consumer);

  poly_ctx_reset_counters(ctx);
  PolyTensor *consumer_out = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &consumer, 1, &consumer_out), 0);
  ASSERT_PTR_EQ(consumer_out, consumer);

  /* Pinned callify.finalize_after maps every tagged original AFTER in the
   * nested assignment chain directly to the stripped final buffer. The live
   * PLACE must therefore finish at the second value version, not at a rebuilt
   * one-AFTER remainder that a later readback can execute again. */
  ASSERT_INT_EQ(target->role, POLY_TENSOR_VALUE);
  ASSERT_TRUE(target->source == NULL);
  ASSERT_INT_EQ(count_root_ops(ctx, poly_tensor_uop(target), POLY_OP_AFTER), 0);
  const PolyUOp *target_identity = poly_uop_get_buffer_identity(poly_tensor_uop(target));
  ASSERT_NOT_NULL(target_identity);
  PolyUOp *mapped_second_value = poly_tensor_uop_physical(assignment_values[1]);
  ASSERT_NOT_NULL(mapped_second_value);
  ASSERT_EQ(mapped_second_value->op, POLY_OP_ADD);
  ASSERT_INT_EQ(count_root_ops(ctx, mapped_second_value, POLY_OP_AFTER), 0);
  ASSERT_TRUE(poly_uop_reachable(ctx, mapped_second_value, (PolyUOp *)target_identity));

  float consumer_value = 0.0f;
  READ_REALIZED_F32(ctx, poly_tensor_uop(consumer), &consumer_value, 1);
  ASSERT_FLOAT_EQ(consumer_value, 21.0f, 1e-6f);

  PolyCtxStats before_rerealize;
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &before_rerealize), 0);
  PolyTensor *target_out = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &target, 1, &target_out), 0);
  ASSERT_PTR_EQ(target_out, target);
  PolyCtxStats after_rerealize;
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &after_rerealize), 0);
  ASSERT_TRUE(after_rerealize.kernel_count == before_rerealize.kernel_count);

  float target_value = 0.0f;
  READ_REALIZED_F32(ctx, poly_tensor_uop(target), &target_value, 1);
  ASSERT_FLOAT_EQ(target_value, 16.0f, 1e-6f);
  ASSERT_FLOAT_EQ(source_data[0], 0.0f, 1e-6f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, earlier_place_version_uses_live_target_in_multi_version_consumer) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  float source_data[] = {0.0f};
  int64_t shape[] = {1};
  PolyUOp *source_physical = NULL;
  PolyTensor *source =
      initialized_f32_tensor(ctx, shape, 1, source_data, POLY_DEVICE_HOST, &source_physical);
  PolyTensor *target = poly_tensor_to_device(ctx, source, POLY_DEVICE_CUDA);
  ASSERT_NOT_NULL(source);
  ASSERT_NOT_NULL(target);

  PolyUOp *first_value_logical =
      poly_add(ctx, poly_tensor_uop_logical(target), poly_const_float(ctx, 8.0f));
  PolyUOp *first_value =
      poly_add(ctx, poly_tensor_uop_physical(target), poly_const_float(ctx, 8.0f));
  PolyTensor *first_tensor = poly_tensor_create_with_roots(
      ctx, first_value_logical, first_value, POLY_TENSOR_VALUE, POLY_DEVICE_CUDA
  );
  ASSERT_NOT_NULL(first_tensor);
  ASSERT_PTR_EQ(poly_tensor_assign(ctx, target, first_tensor), target);
  PolyUOp *first_version_logical = poly_tensor_uop_logical(target);
  PolyUOp *first_version = poly_tensor_uop_physical(target);

  /* Visit the earlier version first, as the first random output does before a
   * later RNG draw reaches the live outer counter PLACE. */
  PolyUOp *earlier_read = poly_add(ctx, first_version, poly_const_float(ctx, 1.0f));
  PolyUOp *second_value_logical =
      poly_add(ctx, poly_tensor_uop_logical(target), poly_const_float(ctx, 8.0f));
  PolyUOp *second_value =
      poly_add(ctx, poly_tensor_uop_physical(target), poly_const_float(ctx, 8.0f));
  PolyTensor *second_tensor = poly_tensor_create_with_roots(
      ctx, second_value_logical, second_value, POLY_TENSOR_VALUE, POLY_DEVICE_CUDA
  );
  ASSERT_NOT_NULL(second_tensor);
  ASSERT_PTR_EQ(poly_tensor_assign(ctx, target, second_tensor), target);
  PolyUOp *consumer_logical = poly_add(
      ctx, poly_add(ctx, first_version_logical, poly_const_float(ctx, 1.0f)),
      poly_tensor_uop_logical(target)
  );
  PolyUOp *consumer_uop = poly_add(ctx, earlier_read, poly_tensor_uop_physical(target));
  PolyTensor *consumer = poly_tensor_create_with_roots(
      ctx, consumer_logical, consumer_uop, POLY_TENSOR_VALUE, POLY_DEVICE_CUDA
  );
  ASSERT_NOT_NULL(consumer);

  PolyUOp *physical = poly_tensor_uop_physical(consumer);
  ASSERT_NOT_NULL(physical);
  int n_physical = 0;
  PolyUOp **topo = poly_toposort(ctx, physical, &n_physical);
  int assignment_afters = 0;
  PolyUOp *inner_target = NULL;
  for (int i = 0; i < n_physical; i++) {
    PolyUOp *u = topo[i];
    if (!u || u->op != POLY_OP_AFTER || u->n_src != 2 || !u->src[1] ||
        u->src[1]->op != POLY_OP_STORE)
      continue;
    assignment_afters++;
    ASSERT_PTR_EQ(u->src[1]->src[0], u->src[0]);
    if (!inner_target || u->src[0]->op != POLY_OP_AFTER) inner_target = u->src[0];
  }
  ASSERT_INT_EQ(assignment_afters, 2);
  ASSERT_NOT_NULL(inner_target);
  ASSERT_EQ(inner_target->op, POLY_OP_COPY);
  ASSERT_INT_EQ(poly_uop_device(inner_target), POLY_DEVICE_CUDA);
  ASSERT_PTR_NEQ(inner_target, source_physical);

  /* Physicalizing the live PLACE directly and reaching only the saved first
   * version must select the same placement-owned COPY.  Pinned tinygrad puts
   * that COPY in the Tensor UOp before assign, so neither consumer source
   * order nor the presence of the later version can change this identity. */
  PolyUOp *live_physical = poly_tensor_uop_physical(target);
  ASSERT_NOT_NULL(live_physical);
  int n_live = 0;
  PolyUOp **live_topo = poly_toposort(ctx, live_physical, &n_live);
  PolyUOp *live_inner_target = NULL;
  for (int i = 0; i < n_live; i++) {
    PolyUOp *u = live_topo[i];
    if (!u || u->op != POLY_OP_AFTER || u->n_src != 2 || !u->src[1] ||
        u->src[1]->op != POLY_OP_STORE)
      continue;
    if (!live_inner_target || u->src[0]->op != POLY_OP_AFTER) live_inner_target = u->src[0];
  }
  ASSERT_PTR_EQ(live_inner_target, inner_target);

  PolyUOp *earlier_read_logical = poly_add(ctx, first_version_logical, poly_const_float(ctx, 1.0f));
  PolyTensor *earlier_only = poly_tensor_create_with_roots(
      ctx, earlier_read_logical, earlier_read, POLY_TENSOR_VALUE, POLY_DEVICE_CUDA
  );
  ASSERT_NOT_NULL(earlier_only);
  PolyUOp *earlier_physical = poly_tensor_uop_physical(earlier_only);
  ASSERT_NOT_NULL(earlier_physical);
  ASSERT_EQ(earlier_physical->op, POLY_OP_ADD);
  ASSERT_INT_EQ(poly_uop_device(earlier_physical), POLY_DEVICE_CUDA);
  int n_earlier = 0;
  PolyUOp **earlier_topo = poly_toposort(ctx, earlier_physical, &n_earlier);
  int earlier_afters = 0;
  PolyUOp *earlier_target = NULL;
  for (int i = 0; i < n_earlier; i++) {
    PolyUOp *u = earlier_topo[i];
    if (!u || u->op != POLY_OP_AFTER || u->n_src != 2 || !u->src[1] ||
        u->src[1]->op != POLY_OP_STORE)
      continue;
    earlier_afters++;
    ASSERT_PTR_EQ(u->src[1]->src[0], u->src[0]);
    earlier_target = u->src[0];
  }
  ASSERT_INT_EQ(earlier_afters, 1);
  ASSERT_PTR_EQ(earlier_target, inner_target);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, shared_allocator_versions_preserve_exact_physical_occurrences) {
  for (int cpu_newest = 0; cpu_newest < 2; cpu_newest++) {
    PolyCtx *ctx = poly_ctx_new();
    ASSERT_NOT_NULL(ctx);

    float source_data[] = {0.0f};
    PolyTensor *source =
        initialized_f32_tensor(ctx, (int64_t[]){1}, 1, source_data, POLY_DEVICE_HOST, NULL);
    ASSERT_NOT_NULL(source);

    PolyTensor *cpu = NULL;
    PolyTensor *interp = NULL;
    if (cpu_newest) {
      interp = poly_tensor_to_device(ctx, source, POLY_DEVICE_INTERP);
      cpu = poly_tensor_to_device(ctx, source, POLY_DEVICE_CPU);
    } else {
      cpu = poly_tensor_to_device(ctx, source, POLY_DEVICE_CPU);
      interp = poly_tensor_to_device(ctx, source, POLY_DEVICE_INTERP);
    }
    ASSERT_NOT_NULL(cpu);
    ASSERT_NOT_NULL(interp);
    ASSERT_PTR_NEQ(cpu, interp);
    ASSERT_TRUE(poly_devices_share_storage(POLY_DEVICE_CPU, POLY_DEVICE_INTERP));

    PolyTensor *targets[2] = {cpu, interp};
    PolyDevice devices[2] = {POLY_DEVICE_CPU, POLY_DEVICE_INTERP};
    PolyUOp *first_versions[2] = {NULL, NULL};
    PolyUOp *first_physical_versions[2] = {NULL, NULL};
    for (int target_idx = 0; target_idx < 2; target_idx++) {
      PolyUOp *value_logical =
          poly_add(ctx, poly_tensor_uop_logical(targets[target_idx]), poly_const_float(ctx, 1.0f));
      PolyUOp *value_uop =
          poly_add(ctx, poly_tensor_uop(targets[target_idx]), poly_const_float(ctx, 1.0f));
      PolyTensor *value = poly_tensor_create_with_roots(
          ctx, value_logical, value_uop, POLY_TENSOR_VALUE, devices[target_idx]
      );
      ASSERT_NOT_NULL(value);
      ASSERT_PTR_EQ(poly_tensor_assign(ctx, targets[target_idx], value), targets[target_idx]);
      first_versions[target_idx] = poly_tensor_uop_logical(targets[target_idx]);
      first_physical_versions[target_idx] = poly_tensor_uop(targets[target_idx]);
    }
    ASSERT_PTR_EQ(first_versions[0], first_versions[1]);
    ASSERT_PTR_NEQ(first_physical_versions[0], first_physical_versions[1]);

    /* Advance both PLACE records to the same second logical version. Consumers
     * of the saved first version must now use exact-chain owner lookup. */
    for (int target_idx = 0; target_idx < 2; target_idx++) {
      PolyUOp *value_logical =
          poly_add(ctx, poly_tensor_uop_logical(targets[target_idx]), poly_const_float(ctx, 1.0f));
      PolyUOp *value_uop =
          poly_add(ctx, poly_tensor_uop(targets[target_idx]), poly_const_float(ctx, 1.0f));
      PolyTensor *value = poly_tensor_create_with_roots(
          ctx, value_logical, value_uop, POLY_TENSOR_VALUE, devices[target_idx]
      );
      ASSERT_NOT_NULL(value);
      ASSERT_PTR_EQ(poly_tensor_assign(ctx, targets[target_idx], value), targets[target_idx]);
    }
    ASSERT_PTR_EQ(poly_tensor_uop_logical(cpu), poly_tensor_uop_logical(interp));
    ASSERT_PTR_NEQ(poly_tensor_uop(cpu), poly_tensor_uop(interp));
    for (int target_idx = 0; target_idx < 2; target_idx++) {
      PolyUOp *consumer_logical =
          poly_add(ctx, first_versions[target_idx], poly_const_float(ctx, 2.0f + target_idx));
      PolyUOp *consumer_uop = poly_add(
          ctx, first_physical_versions[target_idx], poly_const_float(ctx, 2.0f + target_idx)
      );
      PolyTensor *consumer = poly_tensor_create_with_roots(
          ctx, consumer_logical, consumer_uop, POLY_TENSOR_VALUE, devices[target_idx]
      );
      ASSERT_NOT_NULL(consumer);
      PolyUOp *physical = poly_tensor_uop_physical(consumer);
      ASSERT_NOT_NULL(physical);

      int n_topo = 0;
      PolyUOp **topo = poly_toposort(ctx, physical, &n_topo);
      ASSERT_NOT_NULL(topo);
      int assignment_afters = 0;
      for (int i = 0; i < n_topo; i++) {
        PolyUOp *u = topo[i];
        if (!u || u->op != POLY_OP_AFTER || u->n_src != 2 || !u->src[1] ||
            u->src[1]->op != POLY_OP_STORE)
          continue;
        assignment_afters++;
        ASSERT_EQ(u->src[0]->op, POLY_OP_COPY);
        ASSERT_INT_EQ(poly_uop_device(u->src[0]), devices[target_idx]);
        ASSERT_TRUE(u->src[1]->n_src >= 1);
        ASSERT_PTR_EQ(u->src[1]->src[0], u->src[0]);
      }
      ASSERT_INT_EQ(assignment_afters, 1);
    }

    poly_ctx_destroy(ctx);
  }
  PASS();
}

TEST(realize, retained_place_version_replays_after_live_callify_map) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  float source_data[] = {0.0f};
  int64_t shape[] = {1};
  PolyTensor *source = initialized_f32_tensor(ctx, shape, 1, source_data, POLY_DEVICE_HOST, NULL);
  PolyTensor *target = poly_tensor_to_device(ctx, source, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(source);
  ASSERT_NOT_NULL(target);

  PolyUOp *first_value_logical =
      poly_add(ctx, poly_tensor_uop_logical(target), poly_const_float(ctx, 8.0f));
  PolyUOp *first_value_uop =
      poly_add(ctx, poly_tensor_uop_physical(target), poly_const_float(ctx, 8.0f));
  PolyTensor *first_value = poly_tensor_create_with_roots(
      ctx, first_value_logical, first_value_uop, POLY_TENSOR_VALUE, POLY_DEVICE_CPU
  );
  ASSERT_NOT_NULL(first_value);
  ASSERT_PTR_EQ(poly_tensor_assign(ctx, target, first_value), target);
  PolyUOp *first_version_logical = poly_tensor_uop_logical(target);
  PolyUOp *first_version = poly_tensor_uop_physical(target);
  PolyTensor *saved_version = poly_tensor_create_with_roots(
      ctx, first_version_logical, first_version, POLY_TENSOR_VALUE, POLY_DEVICE_CPU
  );
  ASSERT_NOT_NULL(saved_version);
  /* Pinned tinygrad retains one already-deviceful UOp graph. Polygrad's raw
   * UOp APIs are physical-only, so retain the exact placed counterpart while
   * the live PLACE still owns this version; uop_logical remains provenance. */
  PolyUOp *retained_first_physical = poly_tensor_uop_physical(saved_version);
  ASSERT_NOT_NULL(retained_first_physical);
  ASSERT_INT_EQ(poly_uop_retain(ctx, retained_first_physical), 0);

  PolyUOp *second_value_logical =
      poly_add(ctx, poly_tensor_uop_logical(target), poly_const_float(ctx, 8.0f));
  PolyUOp *second_value_uop =
      poly_add(ctx, poly_tensor_uop_physical(target), poly_const_float(ctx, 8.0f));
  PolyTensor *second_value = poly_tensor_create_with_roots(
      ctx, second_value_logical, second_value_uop, POLY_TENSOR_VALUE, POLY_DEVICE_CPU
  );
  ASSERT_NOT_NULL(second_value);
  ASSERT_PTR_EQ(poly_tensor_assign(ctx, target, second_value), target);
  PolyUOp *consumer_logical =
      poly_add(ctx, poly_tensor_uop_logical(target), poly_const_float(ctx, 5.0f));
  PolyUOp *consumer_uop =
      poly_add(ctx, poly_tensor_uop_physical(target), poly_const_float(ctx, 5.0f));
  PolyTensor *consumer = poly_tensor_create_with_roots(
      ctx, consumer_logical, consumer_uop, POLY_TENSOR_VALUE, POLY_DEVICE_CPU
  );
  ASSERT_NOT_NULL(consumer);

  PolyTensor *realized_consumer = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &consumer, 1, &realized_consumer), 0);
  ASSERT_PTR_EQ(realized_consumer, consumer);
  ASSERT_INT_EQ(target->role, POLY_TENSOR_VALUE);
  ASSERT_PTR_EQ(target->source, NULL);
  ASSERT_EQ(poly_tensor_uop(target)->op, POLY_OP_BUFFER);
  ASSERT_EQ(poly_tensor_uop(saved_version)->op, POLY_OP_BUFFER);

  PolyUOp *late_current_uop =
      poly_add(ctx, poly_tensor_uop(saved_version), poly_const_float(ctx, 2.0f));
  PolyUOp *late_current_logical =
      poly_add(ctx, poly_tensor_uop_logical(saved_version), poly_const_float(ctx, 2.0f));
  PolyTensor *late_current = poly_tensor_create_with_roots(
      ctx, late_current_logical, late_current_uop, POLY_TENSOR_VALUE, POLY_DEVICE_CPU
  );
  ASSERT_NOT_NULL(late_current);
  PolyUOp *late_current_physical = poly_tensor_uop_physical(late_current);
  ASSERT_NOT_NULL(late_current_physical);
  ASSERT_INT_EQ(count_root_ops(ctx, late_current_physical, POLY_OP_AFTER), 0);

  /* The raw physical UOp was retained outside the live Tensor map. Pinned
   * tinygrad callifies that same deviceful immutable graph again. */
  PolyUOp *late_raw_uop = poly_add(ctx, retained_first_physical, poly_const_float(ctx, 3.0f));
  PolyUOp *late_raw_logical = poly_add(ctx, first_version_logical, poly_const_float(ctx, 3.0f));
  PolyTensor *late_raw = poly_tensor_create_with_roots(
      ctx, late_raw_logical, late_raw_uop, POLY_TENSOR_VALUE, POLY_DEVICE_CPU
  );
  ASSERT_NOT_NULL(late_raw);
  PolyUOp *late_raw_physical = poly_tensor_uop_physical(late_raw);
  ASSERT_NOT_NULL(late_raw_physical);

  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, late_raw_physical, &n_topo);
  ASSERT_NOT_NULL(topo);
  int assignment_afters = 0;
  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    if (!u || u->op != POLY_OP_AFTER || u->n_src != 2 || !u->src[1] ||
        u->src[1]->op != POLY_OP_STORE)
      continue;
    assignment_afters++;
    ASSERT_EQ(u->src[0]->op, POLY_OP_COPY);
    ASSERT_INT_EQ(poly_uop_device(u->src[0]), POLY_DEVICE_CPU);
    ASSERT_TRUE(u->src[1]->n_src >= 1);
    ASSERT_PTR_EQ(u->src[1]->src[0], u->src[0]);
  }
  ASSERT_INT_EQ(assignment_afters, 1);

  PolyUOp *scheduled_out = NULL;
  PolyUOp *schedule = poly_test_linear_values(ctx, &late_raw_physical, 1, &scheduled_out);
  ASSERT_NOT_NULL(schedule);
  ASSERT_NOT_NULL(scheduled_out);
  ASSERT_INT_EQ(schedule->n_src, 3);

  poly_uop_release(ctx, retained_first_physical);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, aggregate_map_preserves_saved_place_version_topology) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  float source_data[] = {0.0f};
  int64_t shape[] = {1};
  PolyUOp *source_physical = NULL;
  PolyTensor *source =
      initialized_f32_tensor(ctx, shape, 1, source_data, POLY_DEVICE_HOST, &source_physical);
  PolyTensor *target = poly_tensor_to_device(ctx, source, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(source);
  ASSERT_NOT_NULL(target);

  PolyUOp *versions[4] = {NULL};
  PolyUOp *physical_versions[4] = {NULL};
  PolyTensor *saved[4] = {NULL};
  for (int version = 0; version < 4; version++) {
    PolyUOp *value_logical =
        poly_add(ctx, poly_tensor_uop_logical(target), poly_const_float(ctx, 1.0f));
    PolyUOp *value_uop = poly_add(ctx, poly_tensor_uop(target), poly_const_float(ctx, 1.0f));
    PolyTensor *value = poly_tensor_create_with_roots(
        ctx, value_logical, value_uop, POLY_TENSOR_VALUE, POLY_DEVICE_CPU
    );
    ASSERT_NOT_NULL(value);
    ASSERT_PTR_EQ(poly_tensor_assign(ctx, target, value), target);
    versions[version] = poly_tensor_uop_logical(target);
    physical_versions[version] = poly_tensor_uop(target);
    saved[version] = poly_tensor_create_with_roots(
        ctx, versions[version], physical_versions[version], POLY_TENSOR_VALUE, POLY_DEVICE_CPU
    );
    ASSERT_NOT_NULL(saved[version]);
  }

  PolyUOp *retained_raw_first = physical_versions[0];
  ASSERT_INT_EQ(poly_uop_retain(ctx, retained_raw_first), 0);
  PolyUOp *consumer_logical =
      poly_add(ctx, poly_tensor_uop_logical(saved[0]), poly_const_float(ctx, 2.0f));
  PolyUOp *consumer_uop = poly_add(ctx, poly_tensor_uop(saved[0]), poly_const_float(ctx, 2.0f));
  PolyTensor *consumer = poly_tensor_create_with_roots(
      ctx, consumer_logical, consumer_uop, POLY_TENSOR_VALUE, POLY_DEVICE_CPU
  );
  ASSERT_NOT_NULL(consumer);
  poly_ctx_reset_counters(ctx);
  PolyTensor *realized_consumer = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &consumer, 1, &realized_consumer), 0);
  ASSERT_PTR_EQ(realized_consumer, consumer);
  PolyCtxStats stats = {0};
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &stats), 0);
  ASSERT_INT_EQ(stats.kernel_count, 3);

  /* Pinned _apply_map_to_tensors rewrites all live roots in one SINK. Mapping
   * the first saved version strips exactly that prefix from each later live
   * version, while the raw immutable UOp retained outside a Tensor is intact. */
  ASSERT_INT_EQ(count_root_ops(ctx, poly_tensor_uop(target), POLY_OP_AFTER), 3);
  ASSERT_INT_EQ(target->role, POLY_TENSOR_PLACE);
  ASSERT_PTR_EQ(poly_tensor_uop_logical(target), versions[3]);
  const PolyUOp *placed_identity = poly_uop_get_buffer_identity(poly_tensor_uop(saved[0]));
  ASSERT_NOT_NULL(placed_identity);
  ASSERT_INT_EQ(poly_uop_device((PolyUOp *)placed_identity), POLY_DEVICE_CPU);

  for (int version = 0; version < 4; version++) {
    PolyUOp *current = poly_tensor_uop(saved[version]);
    ASSERT_PTR_EQ(poly_tensor_uop_logical(saved[version]), versions[version]);
    ASSERT_INT_EQ(count_root_ops(ctx, current, POLY_OP_AFTER), version);
    ASSERT_TRUE(poly_uop_reachable(ctx, current, (PolyUOp *)placed_identity));

    int n_topo = 0;
    PolyUOp **topo = poly_toposort(ctx, current, &n_topo);
    ASSERT_NOT_NULL(topo);
    for (int i = 0; i < n_topo; i++) {
      PolyUOp *u = topo[i];
      if (!u || u->op != POLY_OP_AFTER) continue;
      ASSERT_INT_EQ(u->n_src, 2);
      ASSERT_EQ(u->src[1]->op, POLY_OP_STORE);
      ASSERT_TRUE(u->src[1]->n_src >= 1);
      ASSERT_PTR_EQ(u->src[1]->src[0], u->src[0]);
    }
  }
  ASSERT_INT_EQ(count_root_ops(ctx, retained_raw_first, POLY_OP_AFTER), 1);
  ASSERT_TRUE(poly_uop_reachable(ctx, retained_raw_first, source_physical));
  ASSERT_FALSE(poly_uop_reachable(ctx, retained_raw_first, (PolyUOp *)placed_identity));

  poly_uop_release(ctx, retained_raw_first);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, aggregate_map_prefers_exact_root_placement_over_exact_descendant) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *state_buf = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 1, POLY_DEVICE_CPU);
  PolyUOp *param_buf = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 1, POLY_DEVICE_CPU);
  PolyUOp *portable_value = poly_buffer_f32(ctx, 1);
  ASSERT_NOT_NULL(state_buf);
  ASSERT_NOT_NULL(param_buf);
  ASSERT_NOT_NULL(portable_value);

  PolyUOp *state_store = poly_store_val(ctx, state_buf, poly_const_float(ctx, 1.0f));
  PolyUOp *state_src[2] = {state_buf, state_store};
  PolyUOp *state_after = poly_uop(ctx, POLY_OP_AFTER, POLY_FLOAT32, state_src, 2, poly_arg_none());
  ASSERT_NOT_NULL(state_store);
  ASSERT_NOT_NULL(state_after);

  PolyUOp *logical_value = poly_add(ctx, state_after, portable_value);
  PolyUOp *logical_store = poly_store_val(ctx, param_buf, logical_value);
  PolyUOp *logical_src[2] = {param_buf, logical_store};
  PolyUOp *param_logical =
      poly_uop(ctx, POLY_OP_AFTER, POLY_FLOAT32, logical_src, 2, poly_arg_none());

  PolyUOp *cpu_device = poly_device_uop(ctx, POLY_DEVICE_CPU);
  PolyUOp *placed_value = poly_copy_to_device_uop(ctx, portable_value, cpu_device);
  PolyUOp *physical_value = poly_add(ctx, state_after, placed_value);
  PolyUOp *physical_store = poly_store_val(ctx, param_buf, physical_value);
  PolyUOp *physical_src[2] = {param_buf, physical_store};
  PolyUOp *param_physical =
      poly_uop(ctx, POLY_OP_AFTER, POLY_FLOAT32, physical_src, 2, poly_arg_none());
  ASSERT_NOT_NULL(logical_value);
  ASSERT_NOT_NULL(logical_store);
  ASSERT_NOT_NULL(param_logical);
  ASSERT_NOT_NULL(cpu_device);
  ASSERT_NOT_NULL(placed_value);
  ASSERT_NOT_NULL(physical_value);
  ASSERT_NOT_NULL(physical_store);
  ASSERT_NOT_NULL(param_physical);
  ASSERT_PTR_NEQ(param_logical, param_physical);

  PolyTensor *state = poly_tensor_create_with_roots(
      ctx, state_after, state_after, POLY_TENSOR_VALUE, POLY_DEVICE_CPU
  );
  PolyTensor *param = poly_tensor_create_with_roots(
      ctx, param_logical, param_physical, POLY_TENSOR_VALUE, POLY_DEVICE_CPU
  );
  ASSERT_NOT_NULL(state);
  ASSERT_NOT_NULL(param);

  PolyUOp *from[2] = {state_after, param_physical};
  PolyUOp *to[2] = {state_buf, param_buf};

  ASSERT_INT_EQ(poly_tensor_apply_realize_map(ctx, from, to, 2, POLY_DEVICE_AUTO), 0);
  ASSERT_PTR_EQ(poly_tensor_uop_logical(state), state_after);
  ASSERT_PTR_EQ(poly_tensor_uop_logical(param), param_logical);
  ASSERT_PTR_EQ(poly_tensor_uop(state), state_buf);
  ASSERT_PTR_EQ(poly_tensor_uop(param), param_buf);
  ASSERT_PTR_EQ(poly_tensor_uop_physical(state), state_buf);
  ASSERT_PTR_EQ(poly_tensor_uop_physical(param), param_buf);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, aggregate_map_keeps_shared_storage_devices_distinct) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  float source_data[] = {0.0f};
  PolyTensor *source =
      initialized_f32_tensor(ctx, (int64_t[]){1}, 1, source_data, POLY_DEVICE_HOST, NULL);
  PolyTensor *cpu = poly_tensor_to_device(ctx, source, POLY_DEVICE_CPU);
  PolyTensor *interp = poly_tensor_to_device(ctx, source, POLY_DEVICE_INTERP);
  ASSERT_NOT_NULL(source);
  ASSERT_NOT_NULL(cpu);
  ASSERT_NOT_NULL(interp);

  PolyTensor *targets[2] = {cpu, interp};
  PolyDevice devices[2] = {POLY_DEVICE_CPU, POLY_DEVICE_INTERP};
  PolyTensor *saved[2] = {NULL};
  PolyUOp *logical[2] = {NULL};
  PolyUOp *placed[2] = {NULL};
  PolyUOp *replacement[2] = {NULL};
  for (int i = 0; i < 2; i++) {
    PolyUOp *value_logical =
        poly_add(ctx, poly_tensor_uop_logical(targets[i]), poly_const_float(ctx, 1.0f));
    PolyUOp *value_uop = poly_add(ctx, poly_tensor_uop(targets[i]), poly_const_float(ctx, 1.0f));
    PolyTensor *value =
        poly_tensor_create_with_roots(ctx, value_logical, value_uop, POLY_TENSOR_VALUE, devices[i]);
    ASSERT_NOT_NULL(value);
    ASSERT_PTR_EQ(poly_tensor_assign(ctx, targets[i], value), targets[i]);
    logical[i] = poly_tensor_uop_logical(targets[i]);
    saved[i] = poly_tensor_create_with_roots(
        ctx, logical[i], poly_tensor_uop(targets[i]), POLY_TENSOR_VALUE, devices[i]
    );
    replacement[i] = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 1, devices[i]);
    ASSERT_NOT_NULL(saved[i]);
    ASSERT_NOT_NULL(replacement[i]);
  }
  for (int i = 0; i < 2; i++) {
    placed[i] = poly_tensor_uop_physical(targets[i]);
    ASSERT_NOT_NULL(placed[i]);
    ASSERT_INT_EQ(poly_uop_device(placed[i]), devices[i]);
  }
  ASSERT_PTR_EQ(logical[0], logical[1]);
  ASSERT_PTR_NEQ(placed[0], placed[1]);

  ASSERT_INT_EQ(poly_tensor_apply_realize_map(ctx, placed, replacement, 2, POLY_DEVICE_AUTO), 0);
  for (int i = 0; i < 2; i++) {
    ASSERT_PTR_EQ(poly_tensor_uop_logical(targets[i]), logical[i]);
    ASSERT_PTR_EQ(poly_tensor_uop_logical(saved[i]), logical[i]);
    ASSERT_PTR_EQ(poly_tensor_uop_physical(targets[i]), replacement[i]);
    ASSERT_PTR_EQ(poly_tensor_uop_physical(saved[i]), replacement[i]);
    ASSERT_INT_EQ(poly_uop_device(poly_tensor_uop(targets[i])), devices[i]);
    ASSERT_PTR_NEQ(poly_tensor_uop(targets[i]), replacement[1 - i]);
  }

  poly_ctx_destroy(ctx);
  PASS();
}

typedef struct {
  ptrdiff_t arena_delta;
  ptrdiff_t cse_delta;
} AggregateMapArenaResult;

static int aggregate_map_unrelated_arena_case(
    int n_values,
    int n_places,
    AggregateMapArenaResult *out
) {
  if (n_values < 0 || n_places < 0 || !out) return -1;
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp **value_roots = n_values > 0 ? calloc((size_t)n_values, sizeof(*value_roots)) : NULL;
  PolyTensor **values = n_values > 0 ? calloc((size_t)n_values, sizeof(*values)) : NULL;
  PolyUOp **place_roots = n_places > 0 ? calloc((size_t)n_places, sizeof(*place_roots)) : NULL;
  PolyTensor **places = n_places > 0 ? calloc((size_t)n_places, sizeof(*places)) : NULL;
  int rc = -1;
  bool retained_new_leaf = false;
  if (!ctx || (n_values > 0 && (!value_roots || !values)) ||
      (n_places > 0 && (!place_roots || !places)))
    goto cleanup;

  PolyUOp *old_leaf = poly_buffer_f32(ctx, 1);
  PolyUOp *new_leaf = poly_buffer_f32(ctx, 1);
  PolyUOp *affected_root = poly_add(ctx, old_leaf, poly_const_float(ctx, 1.0f));
  PolyTensor *affected =
      physical_tensor_from_uop(ctx, affected_root, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  if (!old_leaf || !new_leaf || !affected_root || !affected) goto cleanup;
  if (poly_uop_retain(ctx, new_leaf) != 0) goto cleanup;
  retained_new_leaf = true;

  for (int i = 0; i < n_values; i++) {
    PolyUOp *logical =
        poly_add(ctx, poly_buffer_f32(ctx, 1), poly_const_float(ctx, (double)(i + 2)));
    values[i] = physical_tensor_from_uop(ctx, logical, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
    value_roots[i] = values[i] ? poly_tensor_uop_physical(values[i]) : NULL;
    if (!logical || !values[i] || !value_roots[i]) goto cleanup;
  }
  for (int i = 0; i < n_places; i++) {
    PolyUOp *source_root =
        poly_add(ctx, poly_buffer_f32(ctx, 1), poly_const_float(ctx, (double)(i + 2)));
    PolyTensor *source =
        physical_tensor_from_uop(ctx, source_root, POLY_TENSOR_VALUE, POLY_DEVICE_HOST);
    places[i] = poly_tensor_to_device(ctx, source, POLY_DEVICE_CUDA);
    place_roots[i] = places[i] ? poly_tensor_uop_physical(places[i]) : NULL;
    if (!source_root || !source || !places[i] || !place_roots[i] ||
        poly_tensor_uop(places[i]) != place_roots[i])
      goto cleanup;
  }

  PolyCtxStats before = {0}, after = {0};
  PolyUOp *from[1] = {old_leaf};
  PolyUOp *to[1] = {new_leaf};
  if (poly_ctx_collect(ctx) != 0 || poly_ctx_stats(ctx, &before) != 0 ||
      poly_tensor_apply_realize_map(ctx, from, to, 1, POLY_DEVICE_AUTO) != 0 ||
      poly_ctx_collect(ctx) != 0 || poly_ctx_stats(ctx, &after) != 0 ||
      !poly_uop_reachable(ctx, poly_tensor_uop(affected), new_leaf) ||
      poly_uop_reachable(ctx, poly_tensor_uop(affected), old_leaf))
    goto cleanup;
  for (int i = 0; i < n_values; i++)
    if (poly_tensor_uop_physical(values[i]) != value_roots[i]) goto cleanup;
  for (int i = 0; i < n_places; i++)
    if (poly_tensor_uop(places[i]) != place_roots[i] ||
        poly_tensor_uop_physical(places[i]) != place_roots[i])
      goto cleanup;

  out->arena_delta = (ptrdiff_t)after.arena_bytes - (ptrdiff_t)before.arena_bytes;
  out->cse_delta = (ptrdiff_t)after.cse_entries - (ptrdiff_t)before.cse_entries;
  rc = 0;

cleanup:
  if (retained_new_leaf) poly_uop_release(ctx, new_leaf);
  free(places);
  free(place_roots);
  free(values);
  free(value_roots);
  if (ctx) poly_ctx_destroy(ctx);
  return rc;
}

TEST(realize, aggregate_map_ignores_unrelated_roots_without_retention) {
  AggregateMapArenaResult baseline = {0};
  AggregateMapArenaResult values = {0};
  AggregateMapArenaResult places = {0};
  ASSERT_INT_EQ(aggregate_map_unrelated_arena_case(0, 0, &baseline), 0);
  ASSERT_INT_EQ(aggregate_map_unrelated_arena_case(100, 0, &values), 0);
  ASSERT_INT_EQ(aggregate_map_unrelated_arena_case(0, 100, &places), 0);
  ASSERT_INT_EQ(values.arena_delta, baseline.arena_delta);
  ASSERT_INT_EQ(values.cse_delta, baseline.cse_delta);
  ASSERT_INT_EQ(places.arena_delta, baseline.arena_delta);
  ASSERT_INT_EQ(places.cse_delta, baseline.cse_delta);
  PASS();
}

static int aggregate_map_unrelated_contiguous_arena_case(
    int n_unrelated,
    bool share_input,
    AggregateMapArenaResult *out
) {
  if (n_unrelated < 0 || !out) return -1;
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp **unrelated_roots =
      n_unrelated > 0 ? calloc((size_t)n_unrelated, sizeof(*unrelated_roots)) : NULL;
  PolyTensor **unrelated_tensors =
      n_unrelated > 0 ? calloc((size_t)n_unrelated, sizeof(*unrelated_tensors)) : NULL;
  int rc = -1;
  bool retained_replacement = false;
  if (!ctx || (n_unrelated > 0 && (!unrelated_roots || !unrelated_tensors))) goto cleanup;

  PolyUOp *input = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 1, POLY_DEVICE_CPU);
  float input_value = 1.0f;
  if (!input) goto cleanup;
  poly_buffer_set(ctx, input, &input_value, sizeof(input_value), POLY_DEVICE_CPU);
  PolyUOp *logical_contiguous =
      poly_contiguous(ctx, poly_add(ctx, input, poly_const_float(ctx, 1.0f)));
  PolyUOp *cuda_input = poly_copy_to_device_uop(ctx, input, poly_device_uop(ctx, POLY_DEVICE_CUDA));
  PolyUOp *placed_contiguous =
      poly_contiguous(ctx, poly_add(ctx, cuda_input, poly_const_float(ctx, 1.0f)));
  PolyTensor *boundary = poly_tensor_create_with_roots(
      ctx, logical_contiguous, placed_contiguous, POLY_TENSOR_VALUE, POLY_DEVICE_CUDA
  );
  PolyUOp *consumer = poly_add(ctx, logical_contiguous, poly_const_float(ctx, 5.0f));
  PolyUOp *physical_consumer = poly_add(ctx, placed_contiguous, poly_const_float(ctx, 5.0f));
  PolyTensor *consumer_tensor = poly_tensor_create_with_roots(
      ctx, consumer, physical_consumer, POLY_TENSOR_VALUE, POLY_DEVICE_CUDA
  );
  PolyUOp *replacement = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 1, POLY_DEVICE_CUDA);
  if (!logical_contiguous || !boundary || !placed_contiguous || !consumer || !consumer_tensor ||
      !replacement || placed_contiguous->op != POLY_OP_CONTIGUOUS ||
      placed_contiguous == logical_contiguous)
    goto cleanup;
  if (poly_uop_retain(ctx, replacement) != 0) goto cleanup;
  retained_replacement = true;

  for (int i = 0; i < n_unrelated; i++) {
    PolyUOp *other_input =
        share_input ? input : poly_test_buffer_on_device(ctx, POLY_FLOAT32, 1, POLY_DEVICE_CPU);
    PolyUOp *other_source = poly_add(ctx, other_input, poly_const_float(ctx, (double)(i + 11)));
    PolyUOp *logical_unrelated = poly_contiguous(ctx, other_source);
    PolyUOp *other_cuda =
        poly_copy_to_device_uop(ctx, other_input, poly_device_uop(ctx, POLY_DEVICE_CUDA));
    PolyUOp *physical_unrelated =
        poly_contiguous(ctx, poly_add(ctx, other_cuda, poly_const_float(ctx, (double)(i + 11))));
    unrelated_tensors[i] = poly_tensor_create_with_roots(
        ctx, logical_unrelated, physical_unrelated, POLY_TENSOR_VALUE, POLY_DEVICE_CUDA
    );
    unrelated_roots[i] =
        unrelated_tensors[i] ? poly_tensor_uop_physical(unrelated_tensors[i]) : NULL;
    if (!other_input || !other_source || !logical_unrelated || !unrelated_roots[i] ||
        !unrelated_tensors[i])
      goto cleanup;
  }

  PolyCtxStats before = {0}, after = {0};
  PolyUOp *from[1] = {placed_contiguous};
  PolyUOp *to[1] = {replacement};
  if (poly_ctx_collect(ctx) != 0 || poly_ctx_stats(ctx, &before) != 0 ||
      poly_tensor_apply_realize_map(ctx, from, to, 1, POLY_DEVICE_CUDA) != 0 ||
      poly_ctx_collect(ctx) != 0 || poly_ctx_stats(ctx, &after) != 0 ||
      poly_tensor_uop_logical(boundary) != logical_contiguous ||
      poly_tensor_uop_physical(boundary) != replacement ||
      poly_tensor_uop_logical(consumer_tensor) != consumer ||
      !poly_uop_reachable(ctx, poly_tensor_uop_physical(consumer_tensor), replacement))
    goto cleanup;
  for (int i = 0; i < n_unrelated; i++)
    if (poly_tensor_uop_physical(unrelated_tensors[i]) != unrelated_roots[i]) goto cleanup;

  out->arena_delta = (ptrdiff_t)after.arena_bytes - (ptrdiff_t)before.arena_bytes;
  out->cse_delta = (ptrdiff_t)after.cse_entries - (ptrdiff_t)before.cse_entries;
  rc = 0;

cleanup:
  if (retained_replacement) poly_uop_release(ctx, replacement);
  free(unrelated_tensors);
  free(unrelated_roots);
  if (ctx) poly_ctx_destroy(ctx);
  return rc;
}

TEST(realize, aggregate_map_ignores_same_metadata_contiguous_without_retention) {
  AggregateMapArenaResult baseline = {0};
  AggregateMapArenaResult distinct_input = {0};
  AggregateMapArenaResult shared_input = {0};
  ASSERT_INT_EQ(aggregate_map_unrelated_contiguous_arena_case(0, false, &baseline), 0);
  ASSERT_INT_EQ(aggregate_map_unrelated_contiguous_arena_case(100, false, &distinct_input), 0);
  ASSERT_INT_EQ(aggregate_map_unrelated_contiguous_arena_case(100, true, &shared_input), 0);
  ASSERT_INT_EQ(distinct_input.arena_delta, baseline.arena_delta);
  ASSERT_INT_EQ(distinct_input.cse_delta, baseline.cse_delta);
  ASSERT_TRUE(shared_input.arena_delta <= 0);
  /* Shared placed descendants may survive the weak-CSE sweep. Map work must
   * stay bounded and leave no new cache rows, independent of live sharing. */
  ASSERT_TRUE(shared_input.cse_delta <= 0);
  PASS();
}

TEST(realize, aggregate_map_updates_only_exact_physical_occurrences) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  float source_value = 1.0f;
  PolyTensor *source =
      initialized_f32_tensor(ctx, (int64_t[]){1}, 1, &source_value, POLY_DEVICE_HOST, NULL);
  PolyTensor *target = poly_tensor_to_device(ctx, source, POLY_DEVICE_CUDA);
  ASSERT_NOT_NULL(source);
  ASSERT_NOT_NULL(target);
  ASSERT_NOT_NULL(poly_tensor_uop_physical(target));

  PolyUOp *projected = poly_tensor_uop_physical(target);
  PolyUOp *replacement = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 1, POLY_DEVICE_CUDA);
  ASSERT_NOT_NULL(projected);
  ASSERT_NOT_NULL(replacement);
  ASSERT_PTR_EQ(projected, poly_tensor_uop_physical(target));
  ASSERT_EQ(projected->op, POLY_OP_COPY);
  ASSERT_PTR_EQ(projected->src[0], poly_tensor_uop_physical(source));

  PolyUOp *from[1] = {projected};
  PolyUOp *to[1] = {replacement};
  ASSERT_INT_EQ(poly_tensor_apply_realize_map(ctx, from, to, 1, POLY_DEVICE_CUDA), 0);
  ASSERT_PTR_EQ(poly_tensor_uop_logical(target), poly_tensor_uop_logical(source));
  ASSERT_PTR_EQ(poly_tensor_uop_physical(target), replacement);
  ASSERT_PTR_EQ(poly_tensor_uop(target), replacement);
  /* tinygrad tensor.py:202-206 publishes transform_to_call's BUFFER map
   * before scheduling. Polygrad's exact requested-device counterpart therefore
   * completes PLACE graph state here, while runtime allocation remains absent. */
  ASSERT_EQ(target->role, POLY_TENSOR_VALUE);
  ASSERT_PTR_EQ(target->source, NULL);
  ASSERT_FALSE(poly_buffer_is_allocated(ctx, replacement));

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, nested_shrink_assignment_materializes_before_live_store) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  poly_ctx_set_preferred_device(ctx, POLY_DEVICE_CPU);

  float initial[2] = {1.0f, 2.0f};
  int64_t shape[] = {2};
  PolyTensor *counter = initialized_f32_tensor(ctx, shape, 1, initial, POLY_DEVICE_CPU, NULL);
  ASSERT_NOT_NULL(counter);

  for (int update = 0; update < 2; update++) {
    PolyUOp *logical = poly_tensor_uop_logical(counter);
    PolyUOp *physical = poly_tensor_uop_physical(counter);
    PolyUOp *low_logical = poly_shrink(ctx, logical, (int64_t[1][2]){{0, 1}}, 1);
    PolyUOp *high_logical = poly_shrink(ctx, logical, (int64_t[1][2]){{1, 2}}, 1);
    PolyUOp *low_physical = poly_shrink(ctx, physical, (int64_t[1][2]){{0, 1}}, 1);
    PolyUOp *high_physical = poly_shrink(ctx, physical, (int64_t[1][2]){{1, 2}}, 1);
    PolyUOp *logical_parts[2] = {
        poly_add(ctx, low_logical, poly_const_float(ctx, 1.0f)), high_logical};
    PolyUOp *physical_parts[2] = {
        poly_add(ctx, low_physical, poly_const_float(ctx, 1.0f)), high_physical};
    PolyTensor *value = poly_tensor_create_with_roots(
        ctx, poly_cat(ctx, logical_parts, 2, 0), poly_cat(ctx, physical_parts, 2, 0),
        POLY_TENSOR_VALUE, POLY_DEVICE_CPU
    );
    ASSERT_NOT_NULL(value);
    ASSERT_PTR_EQ(poly_tensor_assign(ctx, counter, value), counter);
  }

  PolyUOp *physical = poly_tensor_uop_physical(counter);
  ASSERT_NOT_NULL(physical);
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, physical, &n_topo);
  int n_afters = 0;
  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    if (!u || u->op != POLY_OP_AFTER) continue;
    n_afters++;
    ASSERT_INT_EQ(u->n_src, 2);
    ASSERT_EQ(u->src[1]->op, POLY_OP_STORE);
    ASSERT_INT_EQ(u->src[1]->n_src, 2);
    ASSERT_PTR_EQ(u->src[0], u->src[1]->src[0]);
  }
  ASSERT_INT_EQ(n_afters, 2);

  PolyUOp *scheduled_out = NULL;
  PolyUOp *schedule = poly_test_linear_values(ctx, &physical, 1, &scheduled_out);
  ASSERT_NOT_NULL(schedule);
  ASSERT_NOT_NULL(scheduled_out);

  /* Pinned fix_store_hazard preserves this exact three-CALL topology:
   * first in-place update, temporary second value, temporary-to-counter
   * STORE. The temporary is a schedule-owned buffer shared by CALLs 1/2. */
  ASSERT_INT_EQ(schedule->n_src, 3);
  ASSERT_INT_EQ(poly_test_linear_call_n_buffers(schedule, 0), 1);
  ASSERT_INT_EQ(poly_test_linear_call_n_buffers(schedule, 1), 2);
  ASSERT_INT_EQ(poly_test_linear_call_n_buffers(schedule, 2), 2);
  PolyUOp *counter_buffer = poly_test_linear_call_buffer(schedule, 0, 0);
  PolyUOp *temporary_buffer = poly_test_linear_call_buffer(schedule, 1, 0);
  ASSERT_PTR_EQ(poly_test_linear_call_buffer(schedule, 1, 1), counter_buffer);
  ASSERT_PTR_EQ(poly_test_linear_call_buffer(schedule, 2, 0), counter_buffer);
  ASSERT_PTR_EQ(poly_test_linear_call_buffer(schedule, 2, 1), temporary_buffer);
  ASSERT_PTR_NEQ(counter_buffer, temporary_buffer);

  ASSERT_INT_EQ(poly_run_linear(ctx, schedule, NULL, 0, NULL, 0, true, false, false), 0);
  float result[2] = {0.0f, 0.0f};
  ASSERT_INT_EQ(
      poly_buffer_read(
          ctx, (PolyUOp *)poly_uop_get_buffer_identity(scheduled_out), result, sizeof(result)
      ),
      0
  );
  ASSERT_FLOAT_EQ(result[0], 3.0f, 1e-6f);
  ASSERT_FLOAT_EQ(result[1], 2.0f, 1e-6f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, creation_copy_nested_shrink_assignment_preserves_version_topology) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  poly_ctx_set_preferred_device(ctx, POLY_DEVICE_CPU);

  float initial[2] = {1.0f, 2.0f};
  int64_t shape[] = {2};
  PolyTensor *source = initialized_f32_tensor(ctx, shape, 1, initial, POLY_DEVICE_HOST, NULL);
  PolyTensor *counter = poly_tensor_to_device(ctx, source, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(source);
  ASSERT_NOT_NULL(counter);

  for (int update = 0; update < 2; update++) {
    PolyUOp *logical = poly_tensor_uop_logical(counter);
    PolyUOp *physical = poly_tensor_uop_physical(counter);
    PolyUOp *logical_parts[2] = {
        poly_add(
            ctx, poly_shrink(ctx, logical, (int64_t[1][2]){{0, 1}}, 1), poly_const_float(ctx, 1.0f)
        ),
        poly_shrink(ctx, logical, (int64_t[1][2]){{1, 2}}, 1)};
    PolyUOp *physical_parts[2] = {
        poly_add(
            ctx, poly_shrink(ctx, physical, (int64_t[1][2]){{0, 1}}, 1), poly_const_float(ctx, 1.0f)
        ),
        poly_shrink(ctx, physical, (int64_t[1][2]){{1, 2}}, 1)};
    PolyTensor *value = poly_tensor_create_with_roots(
        ctx, poly_cat(ctx, logical_parts, 2, 0), poly_cat(ctx, physical_parts, 2, 0),
        POLY_TENSOR_VALUE, POLY_DEVICE_CPU
    );
    ASSERT_NOT_NULL(value);
    ASSERT_PTR_EQ(poly_tensor_assign(ctx, counter, value), counter);
  }

  PolyUOp *physical = poly_tensor_uop_physical(counter);
  ASSERT_NOT_NULL(physical);
  ASSERT_INT_EQ(count_root_ops(ctx, physical, POLY_OP_COPY), 1);
  ASSERT_INT_EQ(count_root_ops(ctx, physical, POLY_OP_AFTER), 2);

  PolyUOp *scheduled_out = NULL;
  PolyUOp *schedule = poly_test_linear_values(ctx, &physical, 1, &scheduled_out);
  ASSERT_NOT_NULL(schedule);
  ASSERT_NOT_NULL(scheduled_out);

  /* Pinned callify keeps the materialized AFTER in the executable graph while
   * its buffer_map points live tensors at the stripped buffer. Exact order:
   * creation COPY, first in-place update, temporary second value, final STORE. */
  ASSERT_INT_EQ(schedule->n_src, 4);
  ASSERT_TRUE(poly_test_linear_call_is_copy(schedule, 0));
  ASSERT_FALSE(poly_test_linear_call_is_copy(schedule, 1));
  ASSERT_FALSE(poly_test_linear_call_is_copy(schedule, 2));
  ASSERT_FALSE(poly_test_linear_call_is_copy(schedule, 3));
  ASSERT_INT_EQ(poly_test_linear_call_n_buffers(schedule, 0), 2);
  ASSERT_INT_EQ(poly_test_linear_call_n_buffers(schedule, 1), 1);
  ASSERT_INT_EQ(poly_test_linear_call_n_buffers(schedule, 2), 2);
  ASSERT_INT_EQ(poly_test_linear_call_n_buffers(schedule, 3), 2);

  PolyUOp *counter_buffer = poly_test_linear_call_buffer(schedule, 0, 0);
  PolyUOp *temporary_buffer = poly_test_linear_call_buffer(schedule, 2, 0);
  ASSERT_PTR_EQ(poly_test_linear_call_buffer(schedule, 1, 0), counter_buffer);
  ASSERT_PTR_EQ(poly_test_linear_call_buffer(schedule, 2, 1), counter_buffer);
  ASSERT_PTR_EQ(poly_test_linear_call_buffer(schedule, 3, 0), counter_buffer);
  ASSERT_PTR_EQ(poly_test_linear_call_buffer(schedule, 3, 1), temporary_buffer);
  ASSERT_PTR_NEQ(counter_buffer, temporary_buffer);

  ASSERT_INT_EQ(poly_run_linear(ctx, schedule, NULL, 0, NULL, 0, true, false, false), 0);
  float result[2] = {0.0f, 0.0f};
  ASSERT_INT_EQ(
      poly_buffer_read(
          ctx, (PolyUOp *)poly_uop_get_buffer_identity(scheduled_out), result, sizeof(result)
      ),
      0
  );
  ASSERT_FLOAT_EQ(result[0], 3.0f, 1e-6f);
  ASSERT_FLOAT_EQ(result[1], 2.0f, 1e-6f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, creation_copy_cuda_roundtrip_keeps_exact_call_dependency_topology) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  float input[3] = {1.0f, 2.0f, 3.0f};
  int64_t shape[] = {3};
  PolyTensor *source = initialized_f32_tensor(ctx, shape, 1, input, POLY_DEVICE_CPU, NULL);
  PolyUOp *one = poly_const_float(ctx, 1.0);
  PolyUOp *logical_add = poly_add(ctx, poly_tensor_uop_logical(source), one);
  PolyUOp *physical_add = poly_add(ctx, poly_tensor_uop_physical(source), one);
  PolyTensor *computed = poly_tensor_create_with_roots(
      ctx, logical_add, physical_add, POLY_TENSOR_VALUE, POLY_DEVICE_CPU
  );
  PolyTensor *cuda = poly_tensor_to_device(ctx, computed, POLY_DEVICE_CUDA);
  PolyTensor *roundtrip = poly_tensor_to_device(ctx, cuda, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(source);
  ASSERT_NOT_NULL(one);
  ASSERT_NOT_NULL(logical_add);
  ASSERT_NOT_NULL(physical_add);
  ASSERT_NOT_NULL(computed);
  ASSERT_NOT_NULL(cuda);
  ASSERT_NOT_NULL(roundtrip);

  PolyUOp *physical = poly_tensor_uop_physical(roundtrip);
  ASSERT_NOT_NULL(physical);
  PolyUOp *scheduled_out = NULL;
  PolyUOp *schedule = poly_test_linear_values(ctx, &physical, 1, &scheduled_out);
  ASSERT_NOT_NULL(schedule);
  ASSERT_NOT_NULL(scheduled_out);

  /* tinygrad@2026-08-22/a9069c177a9d schedules the exact cross-device chain
   * as CPU SINK -> CUDA COPY -> CUDA SINK -> CPU COPY. */
  ASSERT_INT_EQ(schedule->n_src, 4);
  ASSERT_FALSE(poly_test_linear_call_is_copy(schedule, 0));
  ASSERT_TRUE(poly_test_linear_call_is_copy(schedule, 1));
  ASSERT_FALSE(poly_test_linear_call_is_copy(schedule, 2));
  ASSERT_TRUE(poly_test_linear_call_is_copy(schedule, 3));
  for (int k = 0; k < 4; k++)
    ASSERT_INT_EQ(poly_test_linear_call_n_buffers(schedule, k), 2);

  PolyUOp *compute_out = poly_test_linear_call_buffer(schedule, 0, 0);
  PolyUOp *cuda_copy_out = poly_test_linear_call_buffer(schedule, 1, 0);
  PolyUOp *cuda_out = poly_test_linear_call_buffer(schedule, 2, 0);
  PolyUOp *roundtrip_out = poly_test_linear_call_buffer(schedule, 3, 0);
  ASSERT_PTR_EQ(poly_test_linear_call_buffer(schedule, 1, 1), compute_out);
  ASSERT_PTR_EQ(poly_test_linear_call_buffer(schedule, 2, 1), cuda_copy_out);
  ASSERT_PTR_EQ(poly_test_linear_call_buffer(schedule, 3, 1), cuda_out);
  ASSERT_INT_EQ(poly_uop_device_cached(compute_out, NULL), POLY_DEVICE_CPU);
  ASSERT_INT_EQ(poly_uop_device_cached(cuda_copy_out, NULL), POLY_DEVICE_CUDA);
  ASSERT_INT_EQ(poly_uop_device_cached(cuda_out, NULL), POLY_DEVICE_CUDA);
  ASSERT_INT_EQ(poly_uop_device_cached(roundtrip_out, NULL), POLY_DEVICE_CPU);
  ASSERT_PTR_EQ(roundtrip_out, poly_uop_get_buffer_identity(scheduled_out));

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, tensor_alu_preserves_ordered_roundtrip_occurrences) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *logical = poly_buffer_f32(ctx, 2);
  PolyUOp *physical = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 2, POLY_DEVICE_CPU);
  PolyTensor *x =
      poly_tensor_create_with_roots(ctx, logical, physical, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  PolyTensor *x_cuda = poly_tensor_to_device(ctx, x, POLY_DEVICE_CUDA);
  PolyTensor *x_cpu = poly_tensor_to_device(ctx, x_cuda, POLY_DEVICE_CPU);
  PolyTensor *out = poly_tensor_alu2(ctx, POLY_OP_ADD, x, x_cpu);
  ASSERT_NOT_NULL(x);
  ASSERT_NOT_NULL(x_cuda);
  ASSERT_NOT_NULL(x_cpu);
  ASSERT_NOT_NULL(out);

  /* Pinned Tensor._apply_uop/Tensor.alu (tensor.py:128-140) consumes ordered
   * current Tensor.uop operands. The same logical X occurs twice here, but the
   * executable second operand must remain the exact CUDA->CPU COPY chain. */
  PolyUOp *out_logical = poly_tensor_uop_logical(out);
  PolyUOp *out_physical = poly_tensor_uop_physical(out);
  ASSERT_NOT_NULL(out_logical);
  ASSERT_NOT_NULL(out_physical);
  ASSERT_EQ(out_logical->op, POLY_OP_ADD);
  ASSERT_PTR_EQ(out_logical->src[0], logical);
  ASSERT_PTR_EQ(out_logical->src[1], logical);
  ASSERT_EQ(out_physical->op, POLY_OP_ADD);
  ASSERT_PTR_EQ(out_physical->src[0], physical);
  ASSERT_PTR_EQ(out_physical->src[1], poly_tensor_uop_physical(x_cpu));
  ASSERT_EQ(out_physical->src[1]->op, POLY_OP_COPY);
  ASSERT_PTR_EQ(out_physical->src[1]->src[0], poly_tensor_uop_physical(x_cuda));
  ASSERT_EQ(out_physical->src[1]->src[0]->op, POLY_OP_COPY);
  ASSERT_PTR_EQ(out_physical->src[1]->src[0]->src[0], physical);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, tensor_minimum_preserves_ordered_roundtrip_occurrences) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *logical = poly_buffer_f32(ctx, 2);
  PolyUOp *physical = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 2, POLY_DEVICE_CPU);
  PolyTensor *x =
      poly_tensor_create_with_roots(ctx, logical, physical, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  PolyTensor *x_cuda = poly_tensor_to_device(ctx, x, POLY_DEVICE_CUDA);
  PolyTensor *x_cpu = poly_tensor_to_device(ctx, x_cuda, POLY_DEVICE_CPU);
  PolyTensor *out = poly_tensor_minimum(ctx, x, x_cpu);
  ASSERT_NOT_NULL(x);
  ASSERT_NOT_NULL(x_cuda);
  ASSERT_NOT_NULL(x_cpu);
  ASSERT_NOT_NULL(out);

  /* Pinned minimum builds inverse -> MAX -> inverse from ordered Tensor.uop
   * operands (tensor.py:128-140; mixin/elementwise.py:366-393). */
  PolyUOp *out_logical = poly_tensor_uop_logical(out);
  PolyUOp *out_physical = poly_tensor_uop_physical(out);
  PolyUOp *expected_logical = poly_minimum(ctx, logical, logical);
  PolyUOp *expected_physical = poly_minimum(ctx, physical, poly_tensor_uop_physical(x_cpu));
  ASSERT_PTR_EQ(out_logical, expected_logical);
  ASSERT_PTR_EQ(out_physical, expected_physical);
  ASSERT_EQ(out_logical->op, POLY_OP_MUL);
  ASSERT_EQ(out_logical->src[0]->op, POLY_OP_MAX);
  ASSERT_PTR_EQ(out_logical->src[0]->src[0], out_logical->src[0]->src[1]);
  ASSERT_EQ(out_physical->op, POLY_OP_MUL);
  ASSERT_EQ(out_physical->src[0]->op, POLY_OP_MAX);
  ASSERT_EQ(out_physical->src[0]->src[0]->op, POLY_OP_MUL);
  ASSERT_EQ(out_physical->src[0]->src[1]->op, POLY_OP_MUL);
  ASSERT_PTR_EQ(out_physical->src[0]->src[0]->src[0], physical);
  ASSERT_PTR_EQ(out_physical->src[0]->src[1]->src[0], poly_tensor_uop_physical(x_cpu));
  ASSERT_EQ(out_physical->src[0]->src[1]->src[0]->op, POLY_OP_COPY);
  ASSERT_EQ(out_physical->src[0]->src[1]->src[0]->src[0]->op, POLY_OP_COPY);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, tensor_alu_rejects_wrong_operation_arity) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *logical = poly_buffer_f32(ctx, 2);
  PolyUOp *physical = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 2, POLY_DEVICE_CPU);
  PolyTensor *x =
      poly_tensor_create_with_roots(ctx, logical, physical, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(x);

  /* Pinned GroupOp partitions ALU operations by arity
   * (tinygrad/uop/__init__.py:114-119). UOp.alu can temporarily construct
   * malformed source counts, but the exact pinned probe shows every such
   * graph fails during realization. The public C/FFI boundary rejects them
   * before constructing an invalid UOp. */
  ASSERT_TRUE(poly_tensor_alu1(ctx, POLY_OP_ADD, x) == NULL);
  ASSERT_TRUE(poly_tensor_alu2(ctx, POLY_OP_NEG, x, x) == NULL);
  ASSERT_TRUE(poly_tensor_alu3(ctx, POLY_OP_ADD, x, x, x) == NULL);
  ASSERT_NOT_NULL(poly_tensor_alu1(ctx, POLY_OP_NEG, x));
  ASSERT_NOT_NULL(poly_tensor_alu2(ctx, POLY_OP_ADD, x, x));
  ASSERT_NOT_NULL(poly_tensor_alu3(ctx, POLY_OP_MULACC, x, x, x));

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, tensor_value_buffer_identity_is_residency_not_copy) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *logical_buf = poly_buffer_f32(ctx, 6);
  PolyUOp *physical_buf = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 6, POLY_DEVICE_CPU);
  int64_t shape[2] = {2, 3};
  PolyUOp *logical = poly_reshape(ctx, logical_buf, shape, 2);
  PolyUOp *physical = poly_reshape(ctx, physical_buf, shape, 2);
  PolyTensor *direct =
      poly_tensor_create_with_roots(ctx, logical, physical, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(direct);

  PolyUOp *direct_physical = poly_tensor_uop_physical(direct);
  ASSERT_NOT_NULL(direct_physical);
  ASSERT_PTR_EQ(poly_uop_get_buffer_identity(direct_physical), physical_buf);
  ASSERT_INT_EQ(count_root_ops(ctx, direct_physical, POLY_OP_COPY), 0);

  PolyTensor *placed = poly_tensor_to_device(ctx, direct, POLY_DEVICE_CUDA);
  ASSERT_NOT_NULL(placed);
  PolyUOp *placed_physical = poly_tensor_uop_physical(placed);
  ASSERT_NOT_NULL(placed_physical);
  ASSERT_INT_EQ(placed_physical->op, POLY_OP_COPY);
  ASSERT_INT_EQ(count_root_ops(ctx, placed_physical, POLY_OP_COPY), 1);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, eager_physical_operation_uses_exact_moved_operand) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CPU);
  PolyUOp *x = poly_alu2(ctx, POLY_OP_ADD, a, poly_const_float(ctx, 1.0));
  PolyTensor *x_cpu = physical_tensor_from_uop(ctx, x, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  PolyTensor *x_cuda = poly_tensor_to_device(ctx, x_cpu, POLY_DEVICE_CUDA);
  ASSERT_NOT_NULL(x_cpu);
  ASSERT_NOT_NULL(x_cuda);

  int f32 = poly_dtype_id_by_name("float32");
  PolyTensor *two = poly_tensor_const_float_by_id(ctx, 2.0, f32, POLY_DEVICE_CUDA);
  PolyTensor *y_cuda = poly_tensor_alu2(ctx, POLY_OP_ADD, x_cuda, two);
  ASSERT_NOT_NULL(two);
  ASSERT_NOT_NULL(y_cuda);

  PolyUOp *physical = poly_tensor_uop_physical(y_cuda);
  ASSERT_NOT_NULL(physical);
  ASSERT_INT_EQ(physical->op, POLY_OP_ADD);

  PolyUOp *placed_x = physical->src[0];
  ASSERT_NOT_NULL(placed_x);
  ASSERT_INT_EQ(placed_x->op, POLY_OP_COPY);
  ASSERT_INT_EQ(poly_uop_device(placed_x), POLY_DEVICE_CUDA);
  ASSERT_INT_EQ(placed_x->src[0]->op, POLY_OP_ADD);
  ASSERT_PTR_EQ(placed_x->src[0]->src[0], a);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, eager_physical_operation_distinguishes_realized_occurrences) {
  PolyCtx *ctx = poly_ctx_new();

  /* x1 and x2 intentionally preserve the same logical ADD after realization,
   * but they materialize into distinct buffers. A later .to(CUDA) must carry
   * the selected tensor's current buffer root, not the shared logical ADD. */
  float da[] = {1.0f};
  int64_t shape[] = {1};
  int f32 = poly_dtype_id_by_name("float32");
  PolyTensor *a = initialized_f32_tensor(ctx, shape, 1, da, POLY_DEVICE_CPU, NULL);
  PolyTensor *one = poly_tensor_const_float_by_id(ctx, 1.0, f32, POLY_DEVICE_CPU);
  PolyTensor *x1 = poly_tensor_alu2(ctx, POLY_OP_ADD, a, one);
  ASSERT_NOT_NULL(x1);
  PolyTensor *out = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &x1, 1, &out), 0);
  ASSERT_PTR_EQ(out, x1);
  PolyUOp *x1_buf = poly_tensor_uop(x1);
  ASSERT_TRUE(poly_uop_has_buffer_identity(x1_buf));

  PolyTensor *x2 = poly_tensor_alu2(ctx, POLY_OP_ADD, a, one);
  ASSERT_NOT_NULL(x2);
  out = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &x2, 1, &out), 0);
  ASSERT_PTR_EQ(out, x2);
  PolyUOp *x2_buf = poly_tensor_uop(x2);
  ASSERT_TRUE(poly_uop_has_buffer_identity(x2_buf));
  ASSERT_PTR_NEQ(x1_buf, x2_buf);

  PolyTensor *x1_cuda = poly_tensor_to_device(ctx, x1, POLY_DEVICE_CUDA);
  PolyTensor *x2_cuda = poly_tensor_to_device(ctx, x2, POLY_DEVICE_CUDA);
  ASSERT_NOT_NULL(x1_cuda);
  ASSERT_NOT_NULL(x2_cuda);
  ASSERT_PTR_NEQ(x1_cuda, x2_cuda);

  PolyTensor *one_cuda = poly_tensor_const_float_by_id(ctx, 1.0f, f32, POLY_DEVICE_CUDA);
  PolyTensor *y = poly_tensor_alu2(ctx, POLY_OP_ADD, x1_cuda, one_cuda);
  ASSERT_NOT_NULL(one_cuda);
  ASSERT_NOT_NULL(y);

  PolyUOp *physical = poly_tensor_uop_physical(y);
  ASSERT_NOT_NULL(physical);
  ASSERT_INT_EQ(physical->op, POLY_OP_ADD);
  ASSERT_INT_EQ(physical->src[0]->op, POLY_OP_COPY);
  ASSERT_PTR_EQ(physical->src[0]->src[0], x1_buf);
  ASSERT_PTR_NEQ(physical->src[0]->src[0], x2_buf);
  ASSERT_INT_EQ(poly_uop_device(physical->src[0]), POLY_DEVICE_CUDA);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, eager_to_device_preserves_nested_copy_occurrences) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CPU);
  PolyUOp *x = poly_alu2(ctx, POLY_OP_ADD, a, poly_const_float(ctx, 1.0));
  PolyTensor *x_cpu = physical_tensor_from_uop(ctx, x, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  PolyTensor *x_cuda = poly_tensor_to_device(ctx, x_cpu, POLY_DEVICE_CUDA);
  PolyTensor *x_cpu_again = poly_tensor_to_device(ctx, x_cuda, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(x_cpu);
  ASSERT_NOT_NULL(x_cuda);
  ASSERT_NOT_NULL(x_cpu_again);

  PolyUOp *eager_cuda = poly_tensor_uop_physical(x_cuda);
  PolyUOp *eager_cpu = poly_tensor_uop_physical(x_cpu_again);
  ASSERT_NOT_NULL(eager_cuda);
  ASSERT_NOT_NULL(eager_cpu);
  ASSERT_INT_EQ(eager_cuda->op, POLY_OP_COPY);
  ASSERT_INT_EQ(poly_uop_device(eager_cuda), POLY_DEVICE_CUDA);
  ASSERT_INT_EQ(eager_cpu->op, POLY_OP_COPY);
  ASSERT_INT_EQ(poly_uop_device(eager_cpu), POLY_DEVICE_CPU);
  ASSERT_PTR_EQ(eager_cpu->src[0], eager_cuda);

  PolyUOp *physical = poly_tensor_uop_physical(x_cpu_again);
  ASSERT_NOT_NULL(physical);
  ASSERT_PTR_EQ(physical, eager_cpu);
  ASSERT_INT_EQ(physical->op, POLY_OP_COPY);
  ASSERT_INT_EQ(poly_uop_device(physical), POLY_DEVICE_CPU);
  ASSERT_INT_EQ(physical->src[0]->op, POLY_OP_COPY);
  ASSERT_INT_EQ(poly_uop_device(physical->src[0]), POLY_DEVICE_CUDA);
  ASSERT_INT_EQ(physical->src[0]->src[0]->op, POLY_OP_ADD);
  ASSERT_PTR_EQ(physical->src[0]->src[0]->src[0], a);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, eager_operation_preserves_nested_copy_from_realized_source) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer_f32(ctx, 1);
  float da[] = {1.0f};
  poly_buffer_set(ctx, a, da, sizeof(da), POLY_DEVICE_CPU);

  PolyUOp *x_expr = poly_alu2(ctx, POLY_OP_ADD, a, poly_const_float(ctx, 1.0f));
  PolyUOp *x_buf = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 1, POLY_DEVICE_CPU);
  float dx[] = {2.0f};
  poly_buffer_set(ctx, x_buf, dx, sizeof(dx), POLY_DEVICE_CPU);

  /* Simulate the post-realize PolyTensor state directly: logical expression
   * is preserved for export, while current/physical root is the realized CPU
   * buffer. This keeps the test focused on placement, not scheduler runtime. */
  PolyTensor *x_cpu =
      poly_tensor_create_with_roots(ctx, x_expr, x_buf, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(x_cpu);
  ASSERT_TRUE(poly_uop_has_buffer_identity(x_buf));

  PolyTensor *x_cuda = poly_tensor_to_device(ctx, x_cpu, POLY_DEVICE_CUDA);
  PolyTensor *x_cpu_again = poly_tensor_to_device(ctx, x_cuda, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(x_cuda);
  ASSERT_NOT_NULL(x_cpu_again);

  /* Eager construction consumes the exact ordered physical occurrence. There
   * is no global logical/device lookup to reconstruct the roundtrip. */
  int f32 = poly_dtype_id_by_name("float32");
  PolyTensor *one = poly_tensor_const_float_by_id(ctx, 1.0f, f32, POLY_DEVICE_CPU);
  PolyTensor *y_cpu = poly_tensor_alu2(ctx, POLY_OP_ADD, x_cpu_again, one);
  ASSERT_NOT_NULL(one);
  ASSERT_NOT_NULL(y_cpu);

  PolyUOp *physical = poly_tensor_uop_physical(y_cpu);
  ASSERT_NOT_NULL(physical);
  ASSERT_INT_EQ(physical->op, POLY_OP_ADD);
  ASSERT_INT_EQ(count_root_ops(ctx, physical, POLY_OP_COPY), 2);

  PolyUOp *copy_cpu = physical->src[0];
  ASSERT_NOT_NULL(copy_cpu);
  ASSERT_INT_EQ(copy_cpu->op, POLY_OP_COPY);
  ASSERT_INT_EQ(poly_uop_device(copy_cpu), POLY_DEVICE_CPU);

  PolyUOp *copy_cuda = copy_cpu->src[0];
  ASSERT_NOT_NULL(copy_cuda);
  ASSERT_INT_EQ(copy_cuda->op, POLY_OP_COPY);
  ASSERT_INT_EQ(poly_uop_device(copy_cuda), POLY_DEVICE_CUDA);
  ASSERT_PTR_EQ(copy_cuda->src[0], x_buf);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, tensor_assign_targets_exact_moved_physical_occurrence) {
  PolyCtx *ctx = poly_ctx_new();

  float da[] = {1.0f, 2.0f, 3.0f};
  PolyTensor *a_cpu = initialized_f32_tensor(ctx, (int64_t[]){3}, 1, da, POLY_DEVICE_CPU, NULL);
  PolyTensor *a_cuda = poly_tensor_to_device(ctx, a_cpu, POLY_DEVICE_CUDA);
  ASSERT_NOT_NULL(a_cpu);
  ASSERT_NOT_NULL(a_cuda);

  float dv[] = {5.0f, 5.0f, 5.0f};
  PolyTensor *v_cpu = initialized_f32_tensor(ctx, (int64_t[]){3}, 1, dv, POLY_DEVICE_CPU, NULL);
  PolyTensor *v_cuda = poly_tensor_to_device(ctx, v_cpu, POLY_DEVICE_CUDA);
  ASSERT_NOT_NULL(v_cpu);
  ASSERT_NOT_NULL(v_cuda);

  PolyUOp *target_occurrence = poly_tensor_uop_physical(a_cuda);
  PolyUOp *value_occurrence = poly_tensor_uop_physical(v_cuda);
  PolyTensor *assigned = poly_tensor_assign(ctx, a_cuda, v_cuda);
  ASSERT_PTR_EQ(assigned, a_cuda);
  PolyUOp *physical = poly_tensor_uop_physical(assigned);
  ASSERT_NOT_NULL(physical);
  ASSERT_INT_EQ(physical->op, POLY_OP_AFTER);

  PolyUOp *target_copy = physical->src[0];
  PolyUOp *store = physical->src[1];
  ASSERT_NOT_NULL(target_copy);
  ASSERT_NOT_NULL(store);
  ASSERT_INT_EQ(target_copy->op, POLY_OP_COPY);
  ASSERT_PTR_EQ(target_copy, target_occurrence);
  ASSERT_INT_EQ(poly_uop_device(target_copy), POLY_DEVICE_CUDA);
  ASSERT_INT_EQ(store->op, POLY_OP_STORE);
  ASSERT_PTR_EQ(store->src[0], target_copy);
  ASSERT_PTR_EQ(store->src[1], value_occurrence);
  ASSERT_INT_EQ(store->src[1]->op, POLY_OP_COPY);
  ASSERT_INT_EQ(poly_uop_device(store->src[1]), POLY_DEVICE_CUDA);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, tensor_assign_rejects_device_and_dtype_mismatch) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *target_buf = poly_buffer_f32(ctx, 1);
  float target_data[] = {1.0f};
  poly_buffer_set(ctx, target_buf, target_data, sizeof(target_data), POLY_DEVICE_CPU);
  PolyTensor *target_cpu =
      physical_tensor_from_uop(ctx, target_buf, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(target_cpu);

  PolyUOp *value_buf = poly_buffer_f32(ctx, 1);
  float value_data[] = {5.0f};
  poly_buffer_set(ctx, value_buf, value_data, sizeof(value_data), POLY_DEVICE_CPU);
  PolyTensor *value_cpu =
      physical_tensor_from_uop(ctx, value_buf, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  PolyTensor *value_cuda = poly_tensor_to_device(ctx, value_cpu, POLY_DEVICE_CUDA);
  ASSERT_NOT_NULL(value_cpu);
  ASSERT_NOT_NULL(value_cuda);

  /* tinygrad rejects CPU.assign(CUDA) before scheduling. Polygrad must not
   * silently lower the CUDA placement request back into a CPU STORE value. */
  ASSERT_TRUE(poly_tensor_assign(ctx, target_cpu, value_cuda) == NULL);

  PolyUOp *value64_buf = poly_test_buffer(ctx, POLY_FLOAT64, 1);
  double value64_data[] = {5.0};
  poly_buffer_set(ctx, value64_buf, value64_data, sizeof(value64_data), POLY_DEVICE_CPU);
  PolyTensor *value64_cpu =
      physical_tensor_from_uop(ctx, value64_buf, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(value64_cpu);
  ASSERT_TRUE(poly_tensor_assign(ctx, target_cpu, value64_cpu) == NULL);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, tensor_place_assign_realize_materializes_value_without_mutating_source) {
  PolyCtx *ctx = poly_ctx_new();

  float da[] = {1.0f, 2.0f, 3.0f};
  int64_t shape[] = {3};
  PolyUOp *a = NULL;
  PolyTensor *a_host = initialized_f32_tensor(ctx, shape, 1, da, POLY_DEVICE_HOST, &a);
  PolyTensor *a_cpu_copy = poly_tensor_to_device(ctx, a_host, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(a_host);
  ASSERT_NOT_NULL(a_cpu_copy);

  float dv[] = {5.0f, 6.0f, 7.0f};
  PolyTensor *v_cpu = initialized_f32_tensor(ctx, shape, 1, dv, POLY_DEVICE_CPU, NULL);
  ASSERT_NOT_NULL(v_cpu);

  PolyTensor *assigned = poly_tensor_assign(ctx, a_cpu_copy, v_cpu);
  ASSERT_PTR_EQ(assigned, a_cpu_copy);
  PolyTensor *out_tensor = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &assigned, 1, &out_tensor), 0);
  ASSERT_PTR_EQ(out_tensor, assigned);

  PolyUOp *out_uop = poly_tensor_uop(out_tensor);
  ASSERT_TRUE(poly_uop_has_buffer_identity(out_uop));
  ASSERT_PTR_NEQ(poly_uop_get_buffer_identity(out_uop), a);
  PolyBuffer *out = realized_buffer(ctx, out_uop);
  ASSERT_NOT_NULL(out);
  float out_vals[3];
  READ_REALIZED_F32(ctx, out_uop, out_vals, 3);
  ASSERT_FLOAT_EQ(out_vals[0], 5.0f, 1e-5f);
  ASSERT_FLOAT_EQ(out_vals[1], 6.0f, 1e-5f);
  ASSERT_FLOAT_EQ(out_vals[2], 7.0f, 1e-5f);
  ASSERT_FLOAT_EQ(da[0], 1.0f, 1e-5f);
  ASSERT_FLOAT_EQ(da[1], 2.0f, 1e-5f);
  ASSERT_FLOAT_EQ(da[2], 3.0f, 1e-5f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, prebuilt_graph_observes_placed_assign_only_after_assign_realize) {
  PolyCtx *ctx = poly_ctx_new();

  float source_data[] = {0.1f};
  int64_t shape[] = {1};
  int f32 = poly_dtype_id_by_name("float32");
  PolyTensor *source_host =
      initialized_f32_tensor(ctx, shape, 1, source_data, POLY_DEVICE_HOST, NULL);
  PolyTensor *placed = poly_tensor_to_device(ctx, source_host, POLY_DEVICE_INTERP);
  ASSERT_NOT_NULL(source_host);
  ASSERT_NOT_NULL(placed);

  PolyTensor *one = poly_tensor_const_float_by_id(ctx, 1.0, f32, POLY_DEVICE_INTERP);
  PolyTensor *prebuilt = poly_tensor_alu2(ctx, POLY_OP_MUL, placed, one);
  ASSERT_NOT_NULL(prebuilt);

  float value_data[] = {0.2f};
  PolyTensor *value_host =
      initialized_f32_tensor(ctx, shape, 1, value_data, POLY_DEVICE_HOST, NULL);
  PolyTensor *value_interp = poly_tensor_to_device(ctx, value_host, POLY_DEVICE_INTERP);
  ASSERT_NOT_NULL(value_host);
  ASSERT_NOT_NULL(value_interp);
  ASSERT_PTR_EQ(poly_tensor_assign(ctx, placed, value_interp), placed);

  PolyTensor *placed_out = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &placed, 1, &placed_out), 0);
  ASSERT_PTR_EQ(placed_out, placed);

  PolyTensor *prebuilt_out = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &prebuilt, 1, &prebuilt_out), 0);
  ASSERT_PTR_EQ(prebuilt_out, prebuilt);
  float out[1];
  READ_REALIZED_F32(ctx, poly_tensor_uop(prebuilt_out), out, 1);
  ASSERT_FLOAT_EQ(out[0], 0.2f, 1e-6f);
  ASSERT_FLOAT_EQ(source_data[0], 0.1f, 1e-6f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, placed_effect_uses_realized_alias_in_prebuilt_value) {
  PolyCtx *ctx = poly_ctx_new();

  float param_data[] = {1.0f};
  int64_t shape[] = {1};
  PolyTensor *param_host =
      initialized_f32_tensor(ctx, shape, 1, param_data, POLY_DEVICE_HOST, NULL);
  PolyTensor *param = poly_tensor_to_device(ctx, param_host, POLY_DEVICE_INTERP);

  float lr_data[] = {0.1f};
  PolyTensor *lr_host = initialized_f32_tensor(ctx, shape, 1, lr_data, POLY_DEVICE_HOST, NULL);
  PolyTensor *lr = poly_tensor_to_device(ctx, lr_host, POLY_DEVICE_INTERP);
  ASSERT_NOT_NULL(param_host);
  ASSERT_NOT_NULL(param);
  ASSERT_NOT_NULL(lr_host);
  ASSERT_NOT_NULL(lr);

  PolyTensor *update = poly_tensor_alu2(ctx, POLY_OP_SUB, param, lr);
  ASSERT_NOT_NULL(update);
  ASSERT_PTR_EQ(poly_tensor_assign(ctx, param, update), param);

  float next_lr_data[] = {0.2f};
  PolyTensor *next_lr_host =
      initialized_f32_tensor(ctx, shape, 1, next_lr_data, POLY_DEVICE_HOST, NULL);
  PolyTensor *next_lr = poly_tensor_to_device(ctx, next_lr_host, POLY_DEVICE_INTERP);
  ASSERT_NOT_NULL(next_lr_host);
  ASSERT_NOT_NULL(next_lr);
  ASSERT_PTR_EQ(poly_tensor_assign(ctx, lr, next_lr), lr);

  PolyTensor *lr_out = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &lr, 1, &lr_out), 0);
  ASSERT_PTR_EQ(lr_out, lr);
  PolyTensor *param_out = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &param, 1, &param_out), 0);
  ASSERT_PTR_EQ(param_out, param);

  float out[1];
  READ_REALIZED_F32(ctx, poly_tensor_uop(param_out), out, 1);
  ASSERT_FLOAT_EQ(out[0], 0.8f, 1e-6f);
  ASSERT_FLOAT_EQ(param_data[0], 1.0f, 1e-6f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, uop_device_follows_device_and_after_value) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *cpu_device = poly_device_uop(ctx, POLY_DEVICE_CPU);
  PolyUOp *cuda_device = poly_device_uop(ctx, POLY_DEVICE_CUDA);
  ASSERT_INT_EQ(poly_uop_device(cpu_device), POLY_DEVICE_CPU);
  ASSERT_INT_EQ(poly_uop_device(cuda_device), POLY_DEVICE_CUDA);

  PolyUOp *cpu_value = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CPU);
  PolyUOp *cuda_target = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CUDA);
  PolyUOp *cuda_value = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CUDA);
  PolyUOp *effect =
      poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, cuda_target, cuda_value, poly_arg_none());
  PolyUOp *after = poly_uop2(ctx, POLY_OP_AFTER, POLY_FLOAT32, cpu_value, effect, poly_arg_none());
  ASSERT_INT_EQ(poly_uop_device(after), POLY_DEVICE_CPU);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST_COMMON(realize, tensor_realize_cpu_e2e) {
  PolyCtx *ctx = poly_ctx_new();

  float da[] = {1.0f, 2.0f, 3.0f, 4.0f};
  float db[] = {10.0f, 20.0f, 30.0f, 40.0f};
  int64_t shape[] = {4};
  PolyTensor *a = initialized_f32_tensor(ctx, shape, 1, da, POLY_DEVICE_CPU, NULL);
  PolyTensor *b = initialized_f32_tensor(ctx, shape, 1, db, POLY_DEVICE_CPU, NULL);
  PolyTensor *tensor = poly_tensor_alu2(ctx, POLY_OP_ADD, a, b);
  ASSERT_NOT_NULL(tensor);
  PolyTensor *out_tensor = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &tensor, 1, &out_tensor), 0);
  ASSERT_NOT_NULL(out_tensor);
  ASSERT_NOT_NULL(poly_tensor_uop(out_tensor));

  PolyBuffer *buf = realized_buffer(ctx, poly_tensor_uop(out_tensor));
  ASSERT_NOT_NULL(buf);
  float out[4];
  READ_REALIZED_F32(ctx, poly_tensor_uop(out_tensor), out, 4);
  ASSERT_FLOAT_EQ(out[0], 11.0f, 1e-5f);
  ASSERT_FLOAT_EQ(out[1], 22.0f, 1e-5f);
  ASSERT_FLOAT_EQ(out[2], 33.0f, 1e-5f);
  ASSERT_FLOAT_EQ(out[3], 44.0f, 1e-5f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, tensor_host_to_cpu_copy_feeds_compute_e2e) {
  PolyCtx *ctx = poly_ctx_new();

  float *da = malloc(3 * sizeof(float));
  ASSERT_NOT_NULL(da);
  da[0] = 1.0f;
  da[1] = 2.0f;
  da[2] = 3.0f;
  int64_t shape[] = {3};
  int f32 = poly_dtype_id_by_name("float32");
  PolyTensor *host = poly_tensor_from_host(ctx, da, 3 * sizeof(float), POLY_FLOAT32, shape, 1);
  PolyTensor *a = poly_tensor_to_device(ctx, host, POLY_DEVICE_CPU);
  PolyTensor *one = poly_tensor_const_float_by_id(ctx, 1.0, f32, POLY_DEVICE_CPU);
  PolyTensor *tensor = poly_tensor_alu2(ctx, POLY_OP_ADD, a, one);
  ASSERT_NOT_NULL(tensor);
  PolyTensor *out_tensor = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &tensor, 1, &out_tensor), 0);
  ASSERT_NOT_NULL(out_tensor);
  ASSERT_NOT_NULL(poly_tensor_uop(out_tensor));

  PolyBuffer *buf = realized_buffer(ctx, poly_tensor_uop(out_tensor));
  ASSERT_NOT_NULL(buf);
  float out[3];
  READ_REALIZED_F32(ctx, poly_tensor_uop(out_tensor), out, 3);
  ASSERT_FLOAT_EQ(out[0], 2.0f, 1e-5f);
  ASSERT_FLOAT_EQ(out[1], 3.0f, 1e-5f);
  ASSERT_FLOAT_EQ(out[2], 4.0f, 1e-5f);

  poly_ctx_destroy(ctx);
  free(da);
  PASS();
}

TEST(realize, tensor_host_to_cpu_copy_feeds_reduce_broadcast_e2e) {
  PolyCtx *ctx = poly_ctx_new();

  float *da = malloc(4 * sizeof(float));
  ASSERT_NOT_NULL(da);
  da[0] = 1.0f;
  da[1] = 2.0f;
  da[2] = 3.0f;
  da[3] = 4.0f;
  int64_t dims[2] = {1, 4};
  int64_t host_shape[1] = {4};
  int64_t axes[1] = {1};
  int f32 = poly_dtype_id_by_name("float32");
  PolyTensor *host = poly_tensor_from_host(ctx, da, 4 * sizeof(float), POLY_FLOAT32, host_shape, 1);
  PolyTensor *a = poly_tensor_to_device(ctx, host, POLY_DEVICE_CPU);
  PolyTensor *x = poly_tensor_reshape(ctx, a, dims, 2);
  PolyTensor *sum = poly_tensor_sum(ctx, x, axes, 1, true);
  PolyTensor *four = poly_tensor_const_float_by_id(ctx, 4.0, f32, POLY_DEVICE_CPU);
  PolyTensor *mean = poly_tensor_div(ctx, sum, four, 0);
  PolyTensor *tensor = poly_tensor_alu2(ctx, POLY_OP_SUB, x, mean);
  ASSERT_NOT_NULL(tensor);
  PolyTensor *out_tensor = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &tensor, 1, &out_tensor), 0);
  ASSERT_NOT_NULL(out_tensor);
  ASSERT_NOT_NULL(poly_tensor_uop(out_tensor));

  PolyBuffer *buf = realized_buffer(ctx, poly_tensor_uop(out_tensor));
  ASSERT_NOT_NULL(buf);
  float out[4];
  READ_REALIZED_F32(ctx, poly_tensor_uop(out_tensor), out, 4);
  ASSERT_FLOAT_EQ(out[0], -1.5f, 1e-5f);
  ASSERT_FLOAT_EQ(out[1], -0.5f, 1e-5f);
  ASSERT_FLOAT_EQ(out[2], 0.5f, 1e-5f);
  ASSERT_FLOAT_EQ(out[3], 1.5f, 1e-5f);

  poly_ctx_destroy(ctx);
  free(da);
  PASS();
}

TEST(realize, tensor_assign_buffer_updates_original_storage) {
  PolyCtx *ctx = poly_ctx_new();

  float *da = malloc(3 * sizeof(float));
  ASSERT_NOT_NULL(da);
  da[0] = 1.0f;
  da[1] = 2.0f;
  da[2] = 3.0f;
  int64_t shape[] = {3};
  int f32 = poly_dtype_id_by_name("float32");
  PolyUOp *a = NULL;
  PolyTensor *target = initialized_f32_tensor(ctx, shape, 1, da, POLY_DEVICE_CPU, &a);
  PolyTensor *ten = poly_tensor_const_float_by_id(ctx, 10.0, f32, POLY_DEVICE_CPU);
  PolyTensor *value = poly_tensor_alu2(ctx, POLY_OP_ADD, target, ten);
  ASSERT_NOT_NULL(target);
  ASSERT_NOT_NULL(value);

  PolyTensor *assigned = poly_tensor_assign(ctx, target, value);
  ASSERT_NOT_NULL(assigned);
  PolyTensor *out_tensor = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &assigned, 1, &out_tensor), 0);
  ASSERT_NOT_NULL(out_tensor);
  ASSERT_PTR_EQ(poly_uop_get_buffer_identity(poly_tensor_uop(out_tensor)), a);
  ASSERT_INT_EQ(poly_buffer_read(ctx, a, da, 3 * sizeof(float)), 0);

  ASSERT_FLOAT_EQ(da[0], 11.0f, 1e-5f);
  ASSERT_FLOAT_EQ(da[1], 12.0f, 1e-5f);
  ASSERT_FLOAT_EQ(da[2], 13.0f, 1e-5f);

  poly_ctx_destroy(ctx);
  free(da);
  PASS();
}

TEST(realize, tensor_assign_shrink_view_updates_base_storage) {
  PolyCtx *ctx = poly_ctx_new();

  float data[8] = {0.0f, 0.0f, 0.0f, 0.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  int64_t base_shape[] = {8};
  PolyUOp *base = NULL;
  PolyTensor *base_tensor =
      initialized_f32_tensor(ctx, base_shape, 1, data, POLY_DEVICE_CPU, &base);
  ASSERT_NOT_NULL(base_tensor);

  PolyTensor *view_tensor = poly_tensor_shrink(ctx, base_tensor, (int64_t[1][2]){{0, 4}}, 1);
  ASSERT_NOT_NULL(view_tensor);

  float src_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  int64_t src_shape[] = {4};
  PolyTensor *src_tensor =
      initialized_f32_tensor(ctx, src_shape, 1, src_data, POLY_DEVICE_CPU, NULL);
  ASSERT_NOT_NULL(src_tensor);

  ASSERT_PTR_EQ(poly_tensor_assign(ctx, view_tensor, src_tensor), view_tensor);
  /* tinygrad Tensor.assign retargets live wrappers at the base-buffer level:
   * c[:4].assign(v) makes c.uop an AFTER over the original BUFFER, so realizing
   * c or the view executes a STORE into the base storage. */
  ASSERT_INT_EQ(poly_tensor_uop_logical(base_tensor)->op, POLY_OP_AFTER);
  ASSERT_INT_EQ(poly_tensor_uop_physical(base_tensor)->op, POLY_OP_AFTER);
  ASSERT_INT_EQ(poly_tensor_uop_logical(view_tensor)->op, POLY_OP_SHRINK);
  ASSERT_INT_EQ(poly_tensor_uop_physical(view_tensor)->op, POLY_OP_SHRINK);

  PolyTensor *out = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &base_tensor, 1, &out), 0);
  ASSERT_PTR_EQ(out, base_tensor);
  ASSERT_PTR_EQ(poly_uop_get_buffer_identity(poly_tensor_uop(base_tensor)), base);
  ASSERT_INT_EQ(poly_buffer_read(ctx, base, data, sizeof(data)), 0);
  ASSERT_FLOAT_EQ(data[0], 1.0f, 1e-5f);
  ASSERT_FLOAT_EQ(data[1], 2.0f, 1e-5f);
  ASSERT_FLOAT_EQ(data[2], 3.0f, 1e-5f);
  ASSERT_FLOAT_EQ(data[3], 4.0f, 1e-5f);
  ASSERT_FLOAT_EQ(data[4], 5.0f, 1e-5f);
  ASSERT_FLOAT_EQ(data[7], 8.0f, 1e-5f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, tensor_assign_shrink_view_realize_view_updates_base_storage) {
  PolyCtx *ctx = poly_ctx_new();

  float data[8] = {0.0f, 0.0f, 0.0f, 0.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  int64_t base_shape[] = {8};
  PolyUOp *base = NULL;
  PolyTensor *base_tensor =
      initialized_f32_tensor(ctx, base_shape, 1, data, POLY_DEVICE_CPU, &base);
  ASSERT_NOT_NULL(base_tensor);

  PolyTensor *view_tensor = poly_tensor_shrink(ctx, base_tensor, (int64_t[1][2]){{0, 4}}, 1);
  ASSERT_NOT_NULL(view_tensor);

  float src_data[4] = {9.0f, 8.0f, 7.0f, 6.0f};
  int64_t src_shape[] = {4};
  PolyTensor *src_tensor =
      initialized_f32_tensor(ctx, src_shape, 1, src_data, POLY_DEVICE_CPU, NULL);
  ASSERT_NOT_NULL(src_tensor);

  ASSERT_PTR_EQ(poly_tensor_assign(ctx, view_tensor, src_tensor), view_tensor);
  ASSERT_INT_EQ(poly_tensor_uop_logical(base_tensor)->op, POLY_OP_AFTER);
  ASSERT_INT_EQ(poly_tensor_uop_physical(base_tensor)->op, POLY_OP_AFTER);
  ASSERT_INT_EQ(poly_tensor_uop_logical(view_tensor)->op, POLY_OP_SHRINK);
  ASSERT_INT_EQ(poly_tensor_uop_physical(view_tensor)->op, POLY_OP_SHRINK);

  PolyTensor *out = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &view_tensor, 1, &out), 0);
  ASSERT_PTR_EQ(out, view_tensor);
  ASSERT_INT_EQ(poly_buffer_read(ctx, base, data, sizeof(data)), 0);
  ASSERT_FLOAT_EQ(data[0], 9.0f, 1e-5f);
  ASSERT_FLOAT_EQ(data[1], 8.0f, 1e-5f);
  ASSERT_FLOAT_EQ(data[2], 7.0f, 1e-5f);
  ASSERT_FLOAT_EQ(data[3], 6.0f, 1e-5f);
  ASSERT_FLOAT_EQ(data[4], 5.0f, 1e-5f);
  ASSERT_FLOAT_EQ(data[7], 8.0f, 1e-5f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, tensor_assign_realized_contiguous_cache_view_retargets_both_roots) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  /* This gate inspects the original producer after realization. */
  ASSERT_INT_EQ(poly_ctx_set_logical_policy(ctx, POLY_LOGICAL_ALWAYS), 0);
  int f32 = poly_dtype_id_by_name("float32");
  int64_t cache_shape[] = {2, 1, 8, 1, 4};
  int64_t value_shape[] = {2, 1, 3, 1, 4};

  /* Tensor.zeros(...).contiguous() is the Llama KV-cache construction. The
   * device-free logical graph keeps its effect value while the deviceful
   * physical graph materializes CONTIGUOUS. */
  PolyTensor *zero =
      poly_tensor_full_float_by_id(ctx, cache_shape, 5, 0.0, f32, POLY_DEVICE_CPU, true, false);
  PolyTensor *cache = poly_tensor_empty(ctx, POLY_FLOAT32, cache_shape, 5, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(zero);
  ASSERT_NOT_NULL(cache);
  ASSERT_PTR_EQ(poly_tensor_clone_into(ctx, cache, zero), cache);
  cache = poly_tensor_contiguous(ctx, cache);
  ASSERT_NOT_NULL(cache);
  ASSERT_INT_EQ(cache->uop_logical->op, POLY_OP_AFTER);
  ASSERT_INT_EQ(cache->uop_physical->op, POLY_OP_CONTIGUOUS);
  PolyUOp *logical_materialization = cache->uop_logical;

  PolyTensor *realized = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &cache, 1, &realized), 0);
  ASSERT_PTR_EQ(realized, cache);
  ASSERT_PTR_EQ(cache->uop_logical, logical_materialization);
  PolyUOp *physical_materialization = cache->uop_physical;
  PolyUOp *physical_identity = (PolyUOp *)poly_uop_get_buffer_identity(physical_materialization);
  ASSERT_NOT_NULL(physical_identity);

  int64_t bounds[5][2] = {{0, 2}, {0, 1}, {0, 3}, {0, 1}, {0, 4}};
  PolyTensor *view = poly_tensor_shrink(ctx, cache, bounds, 5);
  ASSERT_NOT_NULL(view);
  float values[24];
  for (int i = 0; i < 12; i++) {
    values[i] = (float)i;
    values[12 + i] = (float)(100 + i);
  }
  PolyTensor *value = initialized_f32_tensor(ctx, value_shape, 5, values, POLY_DEVICE_CPU, NULL);
  ASSERT_NOT_NULL(value);

  ASSERT_PTR_EQ(poly_tensor_assign(ctx, view, value), view);
  ASSERT_INT_EQ(cache->uop_logical->op, POLY_OP_AFTER);
  ASSERT_PTR_EQ(cache->uop_logical->src[0], logical_materialization);
  ASSERT_INT_EQ(cache->uop_physical->op, POLY_OP_AFTER);
  ASSERT_PTR_EQ(cache->uop_physical->src[0], physical_materialization);
  ASSERT_INT_EQ(view->uop_logical->op, POLY_OP_SHRINK);
  ASSERT_PTR_EQ(view->uop_logical->src[0], cache->uop_logical);
  ASSERT_INT_EQ(view->uop_physical->op, POLY_OP_SHRINK);
  ASSERT_PTR_EQ(view->uop_physical->src[0], cache->uop_physical);
  ASSERT_INT_EQ(count_root_ops(ctx, cache->uop_physical, POLY_OP_AFTER), 2);
  ASSERT_INT_EQ(count_root_ops(ctx, cache->uop_physical, POLY_OP_STORE), 1);

  PolyTensor *view_out = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &view, 1, &view_out), 0);
  ASSERT_PTR_EQ(view_out, view);
  float actual[64] = {0};
  ASSERT_INT_EQ(poly_buffer_read(ctx, physical_identity, actual, sizeof(actual)), 0);
  for (int i = 0; i < 12; i++) {
    ASSERT_FLOAT_EQ(actual[i], (float)i, 1e-6f);
    ASSERT_FLOAT_EQ(actual[32 + i], (float)(100 + i), 1e-6f);
  }
  for (int i = 12; i < 32; i++)
    ASSERT_FLOAT_EQ(actual[i], 0.0f, 1e-6f);
  for (int i = 44; i < 64; i++)
    ASSERT_FLOAT_EQ(actual[i], 0.0f, 1e-6f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, tensor_assign_permute_view_updates_base_storage) {
  PolyCtx *ctx = poly_ctx_new();

  float data[6] = {0};
  int64_t matrix_shape[] = {2, 3};
  PolyUOp *base = NULL;
  PolyTensor *matrix_tensor =
      initialized_f32_tensor(ctx, matrix_shape, 2, data, POLY_DEVICE_CPU, &base);
  ASSERT_NOT_NULL(matrix_tensor);

  PolyTensor *view_tensor = poly_tensor_permute(ctx, matrix_tensor, (int64_t[]){1, 0}, 2);
  ASSERT_NOT_NULL(view_tensor);

  float src_data[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  int64_t src_shape[] = {3, 2};
  PolyTensor *src_tensor =
      initialized_f32_tensor(ctx, src_shape, 2, src_data, POLY_DEVICE_CPU, NULL);
  ASSERT_NOT_NULL(src_tensor);

  ASSERT_PTR_EQ(poly_tensor_assign(ctx, view_tensor, src_tensor), view_tensor);
  ASSERT_INT_EQ(poly_tensor_uop_logical(matrix_tensor)->op, POLY_OP_AFTER);
  ASSERT_INT_EQ(poly_tensor_uop_physical(matrix_tensor)->op, POLY_OP_AFTER);
  ASSERT_INT_EQ(poly_tensor_uop_logical(view_tensor)->op, POLY_OP_PERMUTE);
  ASSERT_INT_EQ(poly_tensor_uop_physical(view_tensor)->op, POLY_OP_PERMUTE);

  PolyTensor *out = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &matrix_tensor, 1, &out), 0);
  ASSERT_PTR_EQ(out, matrix_tensor);
  ASSERT_PTR_EQ(poly_uop_get_buffer_identity(poly_tensor_uop(matrix_tensor)), base);
  ASSERT_INT_EQ(poly_buffer_read(ctx, base, data, sizeof(data)), 0);
  /* Assigning to m.T writes source[r,c] into m[c,r]. */
  ASSERT_FLOAT_EQ(data[0], 1.0f, 1e-5f);
  ASSERT_FLOAT_EQ(data[1], 3.0f, 1e-5f);
  ASSERT_FLOAT_EQ(data[2], 5.0f, 1e-5f);
  ASSERT_FLOAT_EQ(data[3], 2.0f, 1e-5f);
  ASSERT_FLOAT_EQ(data[4], 4.0f, 1e-5f);
  ASSERT_FLOAT_EQ(data[5], 6.0f, 1e-5f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, tensor_physical_root_tracks_realize_and_assign) {
  PolyCtx *ctx = poly_ctx_new();
  /* Compare current-buffer updates without retiring producer identity. */
  ASSERT_INT_EQ(poly_ctx_set_logical_policy(ctx, POLY_LOGICAL_ALWAYS), 0);

  float *da = malloc(sizeof(float));
  ASSERT_NOT_NULL(da);
  da[0] = 1.0f;
  int f32 = poly_dtype_id_by_name("float32");
  int64_t shape[] = {1};
  PolyTensor *a = initialized_f32_tensor(ctx, shape, 1, da, POLY_DEVICE_CPU, NULL);
  PolyTensor *one = poly_tensor_const_float_by_id(ctx, 1.0, f32, POLY_DEVICE_CPU);
  PolyTensor *x = poly_tensor_alu2(ctx, POLY_OP_ADD, a, one);
  ASSERT_NOT_NULL(x);
  PolyUOp *x_logical_expr = poly_tensor_uop_logical(x);
  PolyUOp *x_physical_expr = poly_tensor_uop_physical(x);

  PolyTensor *out = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &x, 1, &out), 0);
  ASSERT_PTR_EQ(out, x);
  PolyUOp *x_buf = poly_tensor_uop(x);
  ASSERT_PTR_NEQ(x_buf, x_physical_expr);
  ASSERT_PTR_EQ(poly_tensor_uop_logical(x), x_logical_expr);
  ASSERT_TRUE(poly_uop_has_buffer_identity(x_buf));

  float *dv = malloc(sizeof(float));
  ASSERT_NOT_NULL(dv);
  dv[0] = 5.0f;
  PolyTensor *vt = initialized_f32_tensor(ctx, shape, 1, dv, POLY_DEVICE_CPU, NULL);
  ASSERT_NOT_NULL(vt);

  ASSERT_PTR_EQ(poly_tensor_assign(ctx, x, vt), x);
  PolyUOp *after = poly_tensor_uop(x);
  ASSERT_NOT_NULL(after);
  ASSERT_INT_EQ(after->op, POLY_OP_AFTER);

  out = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &x, 1, &out), 0);
  ASSERT_PTR_EQ(out, x);
  ASSERT_PTR_EQ(poly_tensor_uop(x), x_buf);
  PolyBuffer *x_storage = realized_buffer(ctx, poly_tensor_uop(x));
  ASSERT_NOT_NULL(x_storage);
  float x_val = 0.0f;
  READ_REALIZED_F32(ctx, poly_tensor_uop(x), &x_val, 1);
  ASSERT_FLOAT_EQ(x_val, 5.0f, 1e-5f);

  PolyTensor *fresh_inner = poly_tensor_alu2(ctx, POLY_OP_ADD, a, one);
  ASSERT_NOT_NULL(fresh_inner);
  ASSERT_PTR_EQ(poly_tensor_uop_logical(fresh_inner), x_logical_expr);
  ASSERT_INT_EQ(poly_tensor_uop_physical(fresh_inner)->op, POLY_OP_ADD);
  ASSERT_PTR_EQ(poly_tensor_uop_physical(fresh_inner)->src[0], poly_tensor_uop_physical(a));
  ASSERT_PTR_EQ(poly_tensor_uop_physical(fresh_inner)->src[1], poly_tensor_uop_physical(one));
  PolyTensor *z = poly_tensor_alu2(ctx, POLY_OP_ADD, fresh_inner, one);
  ASSERT_NOT_NULL(z);
  out = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &z, 1, &out), 0);
  ASSERT_PTR_EQ(out, z);
  PolyBuffer *z_storage = realized_buffer(ctx, poly_tensor_uop(z));
  ASSERT_NOT_NULL(z_storage);
  ASSERT_PTR_NEQ(z_storage, x_storage);
  float z_val = 0.0f;
  READ_REALIZED_F32(ctx, poly_tensor_uop(z), &z_val, 1);
  ASSERT_FLOAT_EQ(z_val, 3.0f, 1e-5f);

  poly_ctx_destroy(ctx);
  free(da);
  free(dv);
  PASS();
}

TEST(realize, tensor_realized_current_feeds_later_ops_after_source_mutation) {
  PolyCtx *ctx = poly_ctx_new();

  float *da = malloc(sizeof(float));
  ASSERT_NOT_NULL(da);
  da[0] = 1.0f;
  int f32 = poly_dtype_id_by_name("float32");
  int64_t shape[] = {1};
  PolyTensor *at = initialized_f32_tensor(ctx, shape, 1, da, POLY_DEVICE_CPU, NULL);
  ASSERT_NOT_NULL(at);

  PolyTensor *one = poly_tensor_const_float_by_id(ctx, 1.0, f32, POLY_DEVICE_CPU);
  PolyTensor *ar = poly_tensor_alu2(ctx, POLY_OP_ADD, at, one);
  ASSERT_NOT_NULL(ar);
  PolyTensor *out = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &ar, 1, &out), 0);
  ASSERT_PTR_EQ(out, ar);
  PolyBuffer *ar_storage = realized_buffer(ctx, poly_tensor_uop(ar));
  ASSERT_NOT_NULL(ar_storage);
  float ar_val = 0.0f;
  READ_REALIZED_F32(ctx, poly_tensor_uop(ar), &ar_val, 1);
  ASSERT_FLOAT_EQ(ar_val, 2.0f, 1e-5f);

  float *dten = malloc(sizeof(float));
  ASSERT_NOT_NULL(dten);
  dten[0] = 10.0f;
  PolyTensor *tent = initialized_f32_tensor(ctx, shape, 1, dten, POLY_DEVICE_CPU, NULL);
  ASSERT_NOT_NULL(tent);
  ASSERT_PTR_EQ(poly_tensor_assign(ctx, at, tent), at);
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &at, 1, &out), 0);
  READ_REALIZED_F32(ctx, poly_tensor_uop(at), da, 1);
  ASSERT_FLOAT_EQ(da[0], 10.0f, 1e-5f);

  PolyTensor *b = poly_tensor_alu2(ctx, POLY_OP_ADD, ar, one);
  ASSERT_NOT_NULL(b);
  ASSERT_PTR_EQ(poly_tensor_uop_physical(b)->src[0], poly_tensor_uop_physical(ar));
  out = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &b, 1, &out), 0);
  PolyBuffer *b_storage = realized_buffer(ctx, poly_tensor_uop(b));
  ASSERT_NOT_NULL(b_storage);
  float b_val = 0.0f;
  READ_REALIZED_F32(ctx, poly_tensor_uop(b), &b_val, 1);
  ASSERT_FLOAT_EQ(b_val, 3.0f, 1e-5f);

  poly_ctx_destroy(ctx);
  free(da);
  free(dten);
  PASS();
}

TEST(realize, tensor_chained_assign_versions_execute_once) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  float *counter_data = malloc(sizeof(float));
  ASSERT_NOT_NULL(counter_data);
  counter_data[0] = 0.0f;
  int f32 = poly_dtype_id_by_name("float32");
  int64_t shape[] = {1};
  PolyUOp *counter_buf = NULL;
  PolyTensor *counter =
      initialized_f32_tensor(ctx, shape, 1, counter_data, POLY_DEVICE_CPU, &counter_buf);
  ASSERT_NOT_NULL(counter);

  PolyTensor *one = poly_tensor_const_float_by_id(ctx, 1.0, f32, POLY_DEVICE_CPU);
  PolyTensor *next = poly_tensor_alu2(ctx, POLY_OP_ADD, counter, one);
  ASSERT_NOT_NULL(next);
  ASSERT_PTR_EQ(poly_tensor_assign(ctx, counter, next), counter);
  PolyUOp *first_logical_after = poly_tensor_uop_logical(counter);
  PolyUOp *first_physical_after = poly_tensor_uop_physical(counter);
  ASSERT_INT_EQ(first_logical_after->op, POLY_OP_AFTER);
  ASSERT_INT_EQ(first_physical_after->op, POLY_OP_AFTER);
  ASSERT_PTR_EQ(first_physical_after->src[0], counter_buf);

  PolyTensor *version0 =
      poly_tensor_contiguous(ctx, poly_tensor_alu2(ctx, POLY_OP_SUB, counter, one));
  ASSERT_NOT_NULL(version0);

  next = poly_tensor_alu2(ctx, POLY_OP_ADD, counter, one);
  ASSERT_NOT_NULL(next);
  ASSERT_PTR_EQ(poly_tensor_assign(ctx, counter, next), counter);
  PolyUOp *second_logical_after = poly_tensor_uop_logical(counter);
  PolyUOp *second_physical_after = poly_tensor_uop_physical(counter);
  ASSERT_INT_EQ(second_logical_after->op, POLY_OP_AFTER);
  ASSERT_INT_EQ(second_physical_after->op, POLY_OP_AFTER);
  ASSERT_PTR_EQ(second_logical_after->src[0], first_logical_after);
  ASSERT_PTR_EQ(second_physical_after->src[0], first_physical_after);

  PolyTensor *version1 =
      poly_tensor_contiguous(ctx, poly_tensor_alu2(ctx, POLY_OP_SUB, counter, one));
  ASSERT_NOT_NULL(version1);

  PolyUOp *versions[2] = {poly_tensor_uop_logical(version0), poly_tensor_uop_logical(version1)};
  PolyUOp *physical_versions[2] = {
      poly_tensor_uop_physical(version0), poly_tensor_uop_physical(version1)};
  PolyUOp *joined_logical = poly_cat(ctx, versions, 2, 0);
  PolyUOp *joined_physical = poly_cat(ctx, physical_versions, 2, 0);
  ASSERT_NOT_NULL(joined_logical);
  ASSERT_NOT_NULL(joined_physical);
  PolyTensor *result = poly_tensor_create_with_roots(
      ctx, joined_logical, joined_physical, POLY_TENSOR_VALUE, POLY_DEVICE_CPU
  );
  ASSERT_NOT_NULL(result);

  PolyTensor *realized = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &result, 1, &realized), 0);
  ASSERT_PTR_EQ(realized, result);
  float values[2] = {-1.0f, -1.0f};
  READ_REALIZED_F32(ctx, poly_tensor_uop(result), values, 2);
  ASSERT_FLOAT_EQ(values[0], 0.0f, 1e-5f);
  ASSERT_FLOAT_EQ(values[1], 1.0f, 1e-5f);

  ASSERT_PTR_EQ(poly_tensor_uop_logical(counter), second_logical_after);
  PolyTensor *counter_out = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &counter, 1, &counter_out), 0);
  ASSERT_PTR_EQ(counter_out, counter);
  ASSERT_TRUE(poly_uop_has_buffer_identity(poly_tensor_uop(counter)));
  float counter_value = -1.0f;
  READ_REALIZED_F32(ctx, poly_tensor_uop(counter), &counter_value, 1);
  ASSERT_FLOAT_EQ(counter_value, 2.0f, 1e-5f);

  counter_out = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &counter, 1, &counter_out), 0);
  ASSERT_PTR_EQ(counter_out, counter);
  READ_REALIZED_F32(ctx, poly_tensor_uop(counter), &counter_value, 1);
  ASSERT_FLOAT_EQ(counter_value, 2.0f, 1e-5f);

  poly_ctx_destroy(ctx);
  free(counter_data);
  PASS();
}

TEST(realize, chained_assign_schedule_preserves_repeated_effect_call) {
  /* Pinned create_schedule preserves two ordered occurrences of the same
   * immutable assignment CALL in this versioned graph. Only nested
   * precompiled LINEAR flattening deduplicates shared inner work
   * (tinygrad/schedule/__init__.py:21-68,92-128). */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  float *counter_data = malloc(sizeof(float));
  ASSERT_NOT_NULL(counter_data);
  counter_data[0] = 0.0f;
  int f32 = poly_dtype_id_by_name("float32");
  int64_t shape[] = {1};
  PolyTensor *counter = initialized_f32_tensor(ctx, shape, 1, counter_data, POLY_DEVICE_CPU, NULL);
  PolyTensor *one = poly_tensor_const_float_by_id(ctx, 1.0, f32, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(counter);
  ASSERT_NOT_NULL(one);

  PolyTensor *next = poly_tensor_alu2(ctx, POLY_OP_ADD, counter, one);
  ASSERT_NOT_NULL(next);
  ASSERT_PTR_EQ(poly_tensor_assign(ctx, counter, next), counter);
  PolyTensor *version0 =
      poly_tensor_contiguous(ctx, poly_tensor_alu2(ctx, POLY_OP_SUB, counter, one));
  ASSERT_NOT_NULL(version0);

  next = poly_tensor_alu2(ctx, POLY_OP_ADD, counter, one);
  ASSERT_NOT_NULL(next);
  ASSERT_PTR_EQ(poly_tensor_assign(ctx, counter, next), counter);
  PolyTensor *version1 =
      poly_tensor_contiguous(ctx, poly_tensor_alu2(ctx, POLY_OP_SUB, counter, one));
  ASSERT_NOT_NULL(version1);

  PolyUOp *versions[2] = {
      poly_tensor_uop_physical(version0),
      poly_tensor_uop_physical(version1),
  };
  PolyUOp *root = poly_cat(ctx, versions, 2, 0);
  ASSERT_NOT_NULL(root);
  PolyUOp *scheduled = NULL;
  PolyUOp *schedule = poly_test_linear_values(ctx, &root, 1, &scheduled);
  ASSERT_NOT_NULL(schedule);
  ASSERT_NOT_NULL(scheduled);
  ASSERT_INT_EQ(schedule->n_src, 5);
  ASSERT_PTR_EQ(poly_test_linear_call(schedule, 0), poly_test_linear_call(schedule, 2));

  poly_ctx_destroy(ctx);
  free(counter_data);
  PASS();
}

TEST(realize, tensor_shared_lazy_retarget_keeps_distinct_tensor_records) {
  PolyCtx *ctx = poly_ctx_new();

  float *da = malloc(sizeof(float));
  ASSERT_NOT_NULL(da);
  da[0] = 1.0f;
  int f32 = poly_dtype_id_by_name("float32");
  int64_t shape[] = {1};
  PolyTensor *a = initialized_f32_tensor(ctx, shape, 1, da, POLY_DEVICE_CPU, NULL);
  PolyTensor *one = poly_tensor_const_float_by_id(ctx, 1.0, f32, POLY_DEVICE_CPU);
  PolyTensor *expr = poly_tensor_alu2(ctx, POLY_OP_ADD, a, one);
  ASSERT_NOT_NULL(expr);
  PolyTensor *x1 = poly_tensor_create_with_roots(
      ctx, poly_tensor_uop_logical(expr), poly_tensor_uop_physical(expr), POLY_TENSOR_VALUE,
      POLY_DEVICE_CPU
  );
  PolyTensor *x2 = poly_tensor_create_with_roots(
      ctx, poly_tensor_uop_logical(expr), poly_tensor_uop_physical(expr), POLY_TENSOR_VALUE,
      POLY_DEVICE_CPU
  );
  ASSERT_NOT_NULL(x1);
  ASSERT_NOT_NULL(x2);
  ASSERT_PTR_NEQ(x1, x2);
  ASSERT_PTR_EQ(poly_tensor_uop(x1), poly_tensor_uop(x2));

  PolyTensor *out = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &x1, 1, &out), 0);
  ASSERT_PTR_EQ(out, x1);
  PolyUOp *shared_buf = poly_tensor_uop(x1);
  ASSERT_TRUE(poly_uop_has_buffer_identity(shared_buf));
  ASSERT_PTR_EQ(poly_tensor_uop_physical(x2), shared_buf);
  ASSERT_PTR_NEQ(x1, x2);
  ASSERT_PTR_EQ(poly_tensor_uop(x1), poly_tensor_uop(x2));

  float *dv = malloc(sizeof(float));
  ASSERT_NOT_NULL(dv);
  dv[0] = 5.0f;
  PolyTensor *vt = initialized_f32_tensor(ctx, shape, 1, dv, POLY_DEVICE_CPU, NULL);
  ASSERT_NOT_NULL(vt);
  ASSERT_PTR_EQ(poly_tensor_assign(ctx, x2, vt), x2);
  ASSERT_PTR_EQ(poly_tensor_uop(x1), shared_buf);
  ASSERT_INT_EQ(poly_tensor_uop(x2)->op, POLY_OP_AFTER);

  out = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &x2, 1, &out), 0);
  ASSERT_PTR_EQ(out, x2);
  ASSERT_PTR_EQ(poly_tensor_uop(x2), shared_buf);
  PolyBuffer *storage = realized_buffer(ctx, shared_buf);
  ASSERT_NOT_NULL(storage);
  float stored_val = 0.0f;
  READ_REALIZED_F32(ctx, shared_buf, &stored_val, 1);
  ASSERT_FLOAT_EQ(stored_val, 5.0f, 1e-5f);

  poly_ctx_destroy(ctx);
  free(da);
  free(dv);
  PASS();
}

TEST_COMMON(realize, graph_vecadd) {
  PolyCtx *ctx = poly_ctx_new();

  /* Build: add = a + b */
  PolyUOp *a = poly_buffer_f32(ctx, 4);
  PolyUOp *b = poly_buffer_f32(ctx, 4);
  PolyUOp *add = poly_alu2(ctx, POLY_OP_ADD, a, b);

  /* Attach leaf data via side table — poly_realize allocates the output. */
  float da[] = {1.0f, 2.0f, 3.0f, 4.0f};
  float db[] = {10.0f, 20.0f, 30.0f, 40.0f};
  poly_buffer_set(ctx, a, da, sizeof(da), POLY_DEVICE_CPU);
  poly_buffer_set(ctx, b, db, sizeof(db), POLY_DEVICE_CPU);

  PolyUOp *targets[] = {add};
  PolyUOp *realized[] = {NULL};
  ASSERT_INT_EQ(poly_realize_uops(ctx, targets, 1, realized), 0);
  ASSERT_TRUE(realized[0] != NULL);

  const PolyUOp *out = poly_uop_get_buffer_identity(realized[0]);
  ASSERT_TRUE(out != NULL);
  PolyBuffer *buf = poly_buffer_get(ctx, (PolyUOp *)out);
  ASSERT_TRUE(buf != NULL);
  float dout[4];
  READ_REALIZED_F32(ctx, realized[0], dout, 4);
  ASSERT_FLOAT_EQ(dout[0], 11.0f, 1e-5f);
  ASSERT_FLOAT_EQ(dout[1], 22.0f, 1e-5f);
  ASSERT_FLOAT_EQ(dout[2], 33.0f, 1e-5f);
  ASSERT_FLOAT_EQ(dout[3], 44.0f, 1e-5f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, schedule_with_vars_all_realized_passthrough) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *buf = poly_buffer_f32(ctx, 4);
  PolyUOp *targets[] = {buf};
  PolyUOp *realized[] = {NULL};

  PolyUOp *sched = poly_test_linear_values(ctx, targets, 1, realized);
  ASSERT_NOT_NULL(sched);
  ASSERT_INT_EQ(sched->n_src, 0);
  ASSERT_PTR_EQ(realized[0], buf);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, contiguous_realized_scalar_passthrough) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *src = poly_buffer_f32(ctx, 4);
  float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
  poly_buffer_set(ctx, src, data, sizeof(data), POLY_DEVICE_CPU);

  PolyUOp *sum = poly_sum_reduce(ctx, src, 0, 0);
  PolyUOp *targets[] = {sum};
  PolyUOp *realized[] = {NULL};
  ASSERT_INT_EQ(poly_realize_uops(ctx, targets, 1, realized), 0);
  ASSERT_NOT_NULL(realized[0]);
  ASSERT_TRUE(poly_uop_has_buffer_identity(realized[0]));

  PolyUOp *contig = poly_contiguous(ctx, realized[0]);
  ASSERT_PTR_EQ(contig, realized[0]);

  PolyUOp *contig_targets[] = {contig};
  PolyUOp *contig_realized[] = {NULL};
  PolyUOp *sched = poly_test_linear_values(ctx, contig_targets, 1, contig_realized);
  ASSERT_NOT_NULL(sched);
  ASSERT_INT_EQ(sched->n_src, 0);
  ASSERT_PTR_EQ(contig_realized[0], contig);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, requested_plain_root_retargets_live_dependent_to_final_buffer) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  /* This becomes-map test inspects producer-complete logical roots after
   * realization; request that capability explicitly. */
  ASSERT_INT_EQ(poly_ctx_set_logical_policy(ctx, POLY_LOGICAL_ALWAYS), 0);
  float input_value = 1.0f;
  int f32 = poly_dtype_id_by_name("float32");
  int64_t shape[] = {1};
  PolyTensor *host =
      poly_tensor_from_host(ctx, &input_value, sizeof(input_value), POLY_FLOAT32, shape, 1);
  PolyUOp *host_buffer =
      host ? (PolyUOp *)poly_uop_get_buffer_identity(poly_tensor_uop_physical(host)) : NULL;
  PolyTensor *input = poly_tensor_to_device(ctx, host, POLY_DEVICE_CPU);
  PolyTensor *one = poly_tensor_const_float_by_id(ctx, 1.0, f32, POLY_DEVICE_CPU);
  PolyTensor *two = poly_tensor_const_float_by_id(ctx, 2.0, f32, POLY_DEVICE_CPU);
  PolyTensor *requested = poly_tensor_alu2(ctx, POLY_OP_ADD, input, one);
  PolyTensor *dependent = poly_tensor_alu2(ctx, POLY_OP_MUL, requested, two);
  ASSERT_NOT_NULL(host_buffer);
  ASSERT_NOT_NULL(requested);
  ASSERT_NOT_NULL(dependent);
  PolyUOp *requested_before = poly_tensor_uop_physical(requested);
  PolyUOp *requested_logical = poly_tensor_uop_logical(requested);
  PolyUOp *dependent_logical = poly_tensor_uop_logical(dependent);

  PolyTensor *realized = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &requested, 1, &realized), 0);
  ASSERT_PTR_EQ(realized, requested);
  ASSERT_PTR_EQ(poly_tensor_uop_logical(requested), requested_logical);
  ASSERT_PTR_EQ(poly_tensor_uop_logical(dependent), dependent_logical);

  PolyUOp *requested_physical = poly_tensor_uop_physical(requested);
  PolyUOp *dependent_physical = poly_tensor_uop_physical(dependent);
  ASSERT_EQ(requested_physical->op, POLY_OP_BUFFER);
  ASSERT_EQ(dependent_physical->op, POLY_OP_MUL);
  /* Pinned Tensor.linear_with_vars applies transform_to_call's becomes-map to
   * every live Tensor (tensor.py:22-36,202-218). The dependent consumes the
   * finalized BUFFER, never the now-materialized ADD occurrence. */
  ASSERT_TRUE(poly_uop_reachable(ctx, dependent_physical, requested_physical));
  ASSERT_FALSE(poly_uop_reachable(ctx, dependent_physical, requested_before));
  ASSERT_INT_EQ(count_root_ops(ctx, dependent_physical, POLY_OP_ADD), 0);

  input_value = 10.0f;
  ASSERT_INT_EQ(poly_buffer_write(ctx, host_buffer, &input_value, sizeof(input_value)), 0);
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &dependent, 1, &realized), 0);
  float dependent_value = 0.0f;
  READ_REALIZED_F32(ctx, poly_tensor_uop(dependent), &dependent_value, 1);
  ASSERT_FLOAT_EQ(dependent_value, 4.0f, 1e-6f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, aggregate_map_preserves_opaque_bodies_and_updates_external_args) {
  const PolyOps opaque_ops[] = {POLY_OP_CALL, POLY_OP_FUNCTION};
  for (int op_idx = 0; op_idx < 2; op_idx++) {
    for (int key_in_external = 0; key_in_external < 2; key_in_external++) {
      PolyCtx *ctx = poly_ctx_new();
      ASSERT_NOT_NULL(ctx);

      float input_value = 1.0f;
      PolyUOp *input = poly_buffer_f32(ctx, 1);
      ASSERT_NOT_NULL(input);
      poly_buffer_set(ctx, input, &input_value, sizeof(input_value), POLY_DEVICE_HOST);
      PolyUOp *key = poly_contiguous(ctx, poly_add(ctx, input, poly_const_float(ctx, 1.0)));
      PolyTensor *boundary = physical_tensor_from_uop(ctx, key, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
      ASSERT_NOT_NULL(key);
      ASSERT_NOT_NULL(boundary);
      PolyUOp *placed_key = poly_tensor_uop_physical(boundary);
      ASSERT_NOT_NULL(placed_key);

      PolyUOp *value = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 1, POLY_DEVICE_CUDA);
      PolyUOp *body = poly_sink1(ctx, key_in_external ? value : key);
      PolyUOp *external = key_in_external ? key : poly_const_float(ctx, 3.0);
      PolyUOp *opaque_src[2] = {body, external};
      PolyUOp *opaque =
          poly_uop(ctx, opaque_ops[op_idx], POLY_VOID, opaque_src, 2, poly_arg_none());
      PolyUOp *after_src[2] = {value, opaque};
      PolyUOp *after = poly_uop(ctx, POLY_OP_AFTER, value->dtype, after_src, 2, poly_arg_none());
      PolyTensor *tensor =
          physical_tensor_from_uop(ctx, after, POLY_TENSOR_VALUE, POLY_DEVICE_CUDA);
      PolyUOp *physical_before = tensor ? poly_tensor_uop_physical(tensor) : NULL;
      PolyUOp *replacement = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 1, POLY_DEVICE_CUDA);
      ASSERT_NOT_NULL(value);
      ASSERT_NOT_NULL(body);
      ASSERT_NOT_NULL(external);
      ASSERT_NOT_NULL(opaque);
      ASSERT_NOT_NULL(after);
      ASSERT_NOT_NULL(tensor);
      ASSERT_NOT_NULL(physical_before);
      ASSERT_NOT_NULL(replacement);
      ASSERT_INT_EQ(poly_uop_retain(ctx, replacement), 0);

      PolyUOp *from[1] = {placed_key};
      PolyUOp *to[1] = {replacement};
      PolyCtxStats before = {0}, after_stats = {0};
      ASSERT_INT_EQ(poly_ctx_stats(ctx, &before), 0);
      ASSERT_INT_EQ(poly_tensor_apply_realize_map(ctx, from, to, 1, POLY_DEVICE_CUDA), 0);
      ASSERT_INT_EQ(poly_ctx_stats(ctx, &after_stats), 0);

      PolyUOp *physical = poly_tensor_uop_physical(tensor);
      if (!key_in_external) {
        /* Exact-map scope enters the opaque body for ownership, while
         * substitution pins that body. With no caller-visible occurrence the
         * stored physical graph remains unchanged. */
        ASSERT_PTR_EQ(physical, physical_before);
        ASSERT_TRUE(after_stats.arena_bytes <= before.arena_bytes);
        /* Retargeted sibling roots may make weak CSE rows collectible. The
         * opaque no-op path must allocate nothing and cannot grow the cache. */
        ASSERT_TRUE(after_stats.cse_entries <= before.cse_entries);
      } else {
        ASSERT_NOT_NULL(physical);
        ASSERT_EQ(physical->op, POLY_OP_AFTER);
        ASSERT_PTR_EQ(physical->src[0], value);
        ASSERT_NOT_NULL(physical->src[1]);
        ASSERT_EQ(physical->src[1]->op, opaque_ops[op_idx]);
        ASSERT_PTR_EQ(physical->src[1]->src[0], body);
        ASSERT_PTR_EQ(physical->src[1]->src[1], replacement);
      }

      poly_uop_release(ctx, replacement);
      poly_ctx_destroy(ctx);
    }
  }
  PASS();
}

TEST(realize, multi_root_substitution_matches_ordered_opaque_body_pins) {
  const PolyOps opaque_ops[] = {POLY_OP_CALL, POLY_OP_FUNCTION};
  for (int op_idx = 0; op_idx < 2; op_idx++) {
    int permutation[4] = {0, 1, 2, 3};
    do {
      PolyCtx *ctx = poly_ctx_new();
      ASSERT_NOT_NULL(ctx);
      PolyUOp *old = poly_buffer_f32(ctx, 1);
      PolyUOp *replacement = poly_buffer_f32(ctx, 1);
      PolyUOp *inner_body = poly_sink1(ctx, old);
      PolyUOp *inner_src[2] = {inner_body, old};
      PolyUOp *inner = poly_uop(ctx, opaque_ops[op_idx], POLY_VOID, inner_src, 2, poly_arg_none());
      PolyUOp *outer_body_src[2] = {inner, old};
      PolyUOp *outer_body = poly_sink_n(ctx, outer_body_src, 2);
      PolyUOp *outer_src[3] = {outer_body, inner, old};
      PolyUOp *outer = poly_uop(ctx, POLY_OP_CALL, POLY_VOID, outer_src, 3, poly_arg_none());
      PolyUOp *nodes[4] = {inner_body, inner, outer_body, outer};
      PolyUOp *roots[4] = {NULL};
      PolyUOp *out[4] = {NULL};
      PolyUOp *mapped[4] = {NULL};
      PolyUOp *from[1] = {old};
      PolyUOp *to[1] = {replacement};
      ASSERT_NOT_NULL(old);
      ASSERT_NOT_NULL(replacement);
      ASSERT_NOT_NULL(inner_body);
      ASSERT_NOT_NULL(inner);
      ASSERT_NOT_NULL(outer_body);
      ASSERT_NOT_NULL(outer);
      for (int i = 0; i < 4; i++)
        roots[i] = nodes[permutation[i]];

      PolyCtxStats before = {0}, after_stats = {0};
      ASSERT_INT_EQ(poly_ctx_stats(ctx, &before), 0);
      ASSERT_INT_EQ(poly_uop_substitute_many(ctx, roots, 4, from, to, 1, out), 0);
      ASSERT_INT_EQ(poly_ctx_stats(ctx, &after_stats), 0);
      for (int i = 0; i < 4; i++)
        mapped[permutation[i]] = out[i];

      ASSERT_PTR_EQ(mapped[0], inner_body);
      ASSERT_PTR_EQ(mapped[1]->src[0], inner_body);
      ASSERT_PTR_EQ(mapped[1]->src[1], replacement);
      ASSERT_PTR_EQ(mapped[2], outer_body);
      ASSERT_PTR_EQ(mapped[3]->src[0], outer_body);
      ASSERT_PTR_EQ(mapped[3]->src[1], mapped[1]);
      ASSERT_PTR_EQ(mapped[3]->src[2], replacement);
      /* Only the rebuilt inner and outer owners are retained, independent of
       * root order; no overwritten body rewrite remains in the CSE. */
      ASSERT_TRUE(after_stats.cse_entries == before.cse_entries + 2);
      poly_ctx_destroy(ctx);
    } while (realize_next_permutation(permutation, 4));

    int hidden_permutation[3] = {0, 1, 2};
    do {
      PolyCtx *ctx = poly_ctx_new();
      ASSERT_NOT_NULL(ctx);
      PolyUOp *old = poly_buffer_f32(ctx, 1);
      PolyUOp *replacement = poly_buffer_f32(ctx, 1);
      PolyUOp *inner_body = poly_sink1(ctx, old);
      PolyUOp *inner_src[2] = {inner_body, poly_const_float(ctx, 7.0)};
      PolyUOp *inner = poly_uop(ctx, opaque_ops[op_idx], POLY_VOID, inner_src, 2, poly_arg_none());
      PolyUOp *outer_body = poly_sink1(ctx, inner);
      PolyUOp *outer_src[2] = {outer_body, poly_const_float(ctx, 9.0)};
      PolyUOp *outer = poly_uop(ctx, POLY_OP_CALL, POLY_VOID, outer_src, 2, poly_arg_none());
      PolyUOp *nodes[3] = {inner_body, outer_body, outer};
      PolyUOp *roots[3] = {NULL};
      PolyUOp *out[3] = {NULL};
      PolyUOp *mapped[3] = {NULL};
      PolyUOp *from[1] = {old};
      PolyUOp *to[1] = {replacement};
      for (int i = 0; i < 3; i++)
        roots[i] = nodes[hidden_permutation[i]];

      PolyCtxStats before = {0}, after_stats = {0};
      ASSERT_INT_EQ(poly_ctx_stats(ctx, &before), 0);
      ASSERT_INT_EQ(poly_uop_substitute_many(ctx, roots, 3, from, to, 1, out), 0);
      ASSERT_INT_EQ(poly_ctx_stats(ctx, &after_stats), 0);
      for (int i = 0; i < 3; i++)
        mapped[hidden_permutation[i]] = out[i];
      int outer_body_pos = -1, outer_pos = -1;
      for (int i = 0; i < 3; i++) {
        if (hidden_permutation[i] == 1) outer_body_pos = i;
        if (hidden_permutation[i] == 2) outer_pos = i;
      }
      bool inner_body_frozen = outer_body_pos < outer_pos;
      if (inner_body_frozen)
        ASSERT_PTR_EQ(mapped[0], inner_body);
      else
        ASSERT_PTR_NEQ(mapped[0], inner_body);
      ASSERT_PTR_EQ(mapped[1], outer_body);
      ASSERT_PTR_EQ(mapped[2]->src[0], outer_body);
      ASSERT_TRUE(after_stats.cse_entries == before.cse_entries + (inner_body_frozen ? 0 : 1));
      poly_ctx_destroy(ctx);
    } while (realize_next_permutation(hidden_permutation, 3));
  }
  PASS();
}

TEST(realize, nested_contiguous_retargets_live_consumer_to_realized_value) {
  PolyCtx *ctx = poly_ctx_new();
  poly_ctx_set_preferred_device(ctx, POLY_DEVICE_INTERP);

  float initial = 1.0f;
  int f32 = poly_dtype_id_by_name("float32");
  int64_t shape[] = {1};
  PolyUOp *input_buffer = NULL;
  PolyTensor *input =
      initialized_f32_tensor(ctx, shape, 1, &initial, POLY_DEVICE_INTERP, &input_buffer);
  PolyTensor *one = poly_tensor_const_float_by_id(ctx, 1.0, f32, POLY_DEVICE_INTERP);
  PolyTensor *two = poly_tensor_const_float_by_id(ctx, 2.0, f32, POLY_DEVICE_INTERP);
  PolyTensor *five = poly_tensor_const_float_by_id(ctx, 5.0, f32, POLY_DEVICE_INTERP);
  PolyTensor *inner = poly_tensor_contiguous(ctx, poly_tensor_alu2(ctx, POLY_OP_ADD, input, one));
  PolyTensor *requested_tensor =
      poly_tensor_contiguous(ctx, poly_tensor_alu2(ctx, POLY_OP_MUL, inner, two));
  PolyTensor *consumer_tensor = poly_tensor_alu2(ctx, POLY_OP_ADD, inner, five);
  ASSERT_NOT_NULL(requested_tensor);
  ASSERT_NOT_NULL(consumer_tensor);
  PolyUOp *consumer_logical = poly_tensor_uop_logical(consumer_tensor);

  PolyTensor *realized = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &requested_tensor, 1, &realized), 0);
  ASSERT_PTR_EQ(realized, requested_tensor);
  ASSERT_PTR_EQ(poly_tensor_uop_logical(consumer_tensor), consumer_logical);
  ASSERT_NOT_NULL(poly_tensor_uop_physical(consumer_tensor));

  float requested_out = 0.0f;
  READ_REALIZED_F32(ctx, poly_tensor_uop(requested_tensor), &requested_out, 1);
  ASSERT_FLOAT_EQ(requested_out, 4.0f, 1e-5f);

  float updated = 10.0f;
  ASSERT_INT_EQ(poly_buffer_write(ctx, input_buffer, &updated, sizeof(updated)), 0);
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &consumer_tensor, 1, &realized), 0);
  ASSERT_PTR_EQ(realized, consumer_tensor);
  float consumer_out = 0.0f;
  READ_REALIZED_F32(ctx, poly_tensor_uop(consumer_tensor), &consumer_out, 1);
  ASSERT_FLOAT_EQ(consumer_out, 7.0f, 1e-5f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, batched_explicit_and_nested_contiguous_materializes_once) {
  PolyCtx *ctx = poly_ctx_new();
  poly_ctx_set_preferred_device(ctx, POLY_DEVICE_INTERP);

  float initial = 1.0f;
  int f32 = poly_dtype_id_by_name("float32");
  int64_t shape[] = {1};
  PolyTensor *input = initialized_f32_tensor(ctx, shape, 1, &initial, POLY_DEVICE_INTERP, NULL);
  PolyTensor *one = poly_tensor_const_float_by_id(ctx, 1.0, f32, POLY_DEVICE_INTERP);
  PolyTensor *five = poly_tensor_const_float_by_id(ctx, 5.0, f32, POLY_DEVICE_INTERP);
  PolyTensor *inner_tensor =
      poly_tensor_contiguous(ctx, poly_tensor_alu2(ctx, POLY_OP_ADD, input, one));
  PolyTensor *consumer_tensor = poly_tensor_alu2(ctx, POLY_OP_ADD, inner_tensor, five);
  ASSERT_NOT_NULL(inner_tensor);
  ASSERT_NOT_NULL(consumer_tensor);

  poly_ctx_reset_counters(ctx);
  PolyTensor *targets[2] = {inner_tensor, consumer_tensor};
  PolyTensor *outputs[2] = {NULL, NULL};
  ASSERT_INT_EQ(poly_realize_tensors(ctx, targets, 2, outputs), 0);
  ASSERT_PTR_EQ(outputs[0], inner_tensor);
  ASSERT_PTR_EQ(outputs[1], consumer_tensor);

  float inner_out = 0.0f, consumer_out = 0.0f;
  READ_REALIZED_F32(ctx, poly_tensor_uop(inner_tensor), &inner_out, 1);
  READ_REALIZED_F32(ctx, poly_tensor_uop(consumer_tensor), &consumer_out, 1);
  ASSERT_FLOAT_EQ(inner_out, 2.0f, 1e-5f);
  ASSERT_FLOAT_EQ(consumer_out, 7.0f, 1e-5f);

  PolyCtxStats stats = {0};
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &stats), 0);
  ASSERT_TRUE(stats.kernel_count == 2);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, placed_after_map_retargets_live_assignment_across_copy) {
  PolyCtx *ctx = poly_ctx_new();

  int64_t shape[] = {2};
  PolyTensor *host_counter = poly_tensor_empty(ctx, POLY_UINT32, shape, 1, POLY_DEVICE_CPU);
  PolyTensor *counter = poly_tensor_to_device(ctx, host_counter, POLY_DEVICE_CUDA);
  ASSERT_NOT_NULL(host_counter);
  ASSERT_NOT_NULL(counter);

  PolyUOp *incremented_logical =
      poly_add(ctx, poly_tensor_uop_logical(counter), poly_const_int(ctx, 1));
  PolyUOp *incremented = poly_add(ctx, poly_tensor_uop_physical(counter), poly_const_int(ctx, 1));
  PolyTensor *value = poly_tensor_create_with_roots(
      ctx, incremented_logical, incremented, POLY_TENSOR_VALUE, POLY_DEVICE_CUDA
  );
  ASSERT_NOT_NULL(value);
  ASSERT_PTR_EQ(poly_tensor_assign(ctx, counter, value), counter);

  PolyUOp *logical_after = poly_tensor_uop_logical(counter);
  PolyUOp *placed_after = poly_tensor_uop_physical(counter);
  ASSERT_NOT_NULL(logical_after);
  ASSERT_NOT_NULL(placed_after);
  ASSERT_INT_EQ(logical_after->op, POLY_OP_AFTER);
  ASSERT_INT_EQ(logical_after->src[0]->op, POLY_OP_BUFFER);
  ASSERT_INT_EQ(placed_after->op, POLY_OP_AFTER);
  ASSERT_INT_EQ(placed_after->src[0]->op, POLY_OP_COPY);
  ASSERT_TRUE(poly_uop_get_buffer_identity(placed_after->src[0]) == NULL);

  /* Pinned finalize_after maps the current AFTER directly to final storage.
   * The live physical Tensor is retargeted by exact occurrence identity. */
  PolyUOp *realized = poly_test_buffer_on_device(ctx, POLY_UINT32, 2, POLY_DEVICE_CUDA);
  PolyUOp *from[1] = {placed_after};
  PolyUOp *to[1] = {realized};
  ASSERT_NOT_NULL(realized);
  ASSERT_INT_EQ(poly_tensor_apply_realize_map(ctx, from, to, 1, POLY_DEVICE_CUDA), 0);
  ASSERT_PTR_EQ(poly_tensor_uop_logical(counter), logical_after);
  ASSERT_PTR_EQ(poly_tensor_uop_physical(counter), realized);
  ASSERT_PTR_EQ(poly_tensor_uop(counter), realized);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, transform_to_call_finalizes_creation_copy_assignment_after) {
  PolyCtx *ctx = poly_ctx_new();

  float initial = 2.0f;
  PolyUOp *host = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 1, POLY_DEVICE_HOST);
  poly_buffer_set(ctx, host, &initial, sizeof(initial), POLY_DEVICE_HOST);
  PolyUOp *device = poly_device_uop(ctx, POLY_DEVICE_CPU);
  PolyUOp *copy = poly_copy_to_device_uop(ctx, host, device);
  PolyUOp *value = poly_add(ctx, copy, poly_const_float(ctx, 1.0));
  PolyUOp *store = poly_store_val(ctx, copy, value);
  PolyUOp *after_src[2] = {copy, store};
  PolyUOp *after = poly_uop(ctx, POLY_OP_AFTER, POLY_FLOAT32, after_src, 2, poly_arg_none());
  ASSERT_NOT_NULL(after);

  /* Pinned add_tags merges a creation COPY tag into this assignment AFTER.
   * Keep live tensors for both original UOps to prove finalize_after's two
   * becomes-map entries, while the executable consumer reads the new value. */
  PolyTensor *counter = physical_tensor_from_uop(ctx, after, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  PolyTensor *creation_copy =
      physical_tensor_from_uop(ctx, copy, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  PolyUOp *consumer = poly_add(ctx, after, poly_const_float(ctx, 5.0));
  PolyUOp *realized = NULL;
  PolyUOp **map_orig = NULL;
  PolyUOp **map_repl = NULL;
  int map_n = 0;
  PolyUOp *call =
      poly_transform_to_call_with_map(ctx, &consumer, 1, &realized, &map_orig, &map_repl, &map_n);
  ASSERT_NOT_NULL(counter);
  ASSERT_NOT_NULL(creation_copy);
  ASSERT_NOT_NULL(call);
  ASSERT_NOT_NULL(realized);
  ASSERT_NOT_NULL(poly_uop_get_buffer_identity(realized));
  ASSERT_TRUE(map_n > 0);

  PolyUOp *after_replacement = NULL;
  PolyUOp *copy_replacement = NULL;
  int after_rows = 0, copy_rows = 0;
  for (int i = 0; i < map_n; i++) {
    if (map_orig[i] == after) {
      after_rows++;
      after_replacement = map_repl[i];
    }
    if (map_orig[i] == copy) {
      copy_rows++;
      copy_replacement = map_repl[i];
    }
  }
  ASSERT_INT_EQ(after_rows, 1);
  ASSERT_INT_EQ(copy_rows, 1);
  ASSERT_PTR_EQ(after_replacement, copy_replacement);
  ASSERT_INT_EQ(poly_tensor_apply_realize_map(ctx, map_orig, map_repl, map_n, POLY_DEVICE_AUTO), 0);

  const PolyUOp *realized_identity = poly_uop_get_buffer_identity(realized);
  const PolyUOp *counter_identity = poly_uop_get_buffer_identity(poly_tensor_uop_physical(counter));
  const PolyUOp *creation_copy_identity =
      poly_uop_get_buffer_identity(poly_tensor_uop_physical(creation_copy));
  ASSERT_NOT_NULL(counter_identity);
  ASSERT_PTR_EQ(creation_copy_identity, counter_identity);
  ASSERT_PTR_NEQ(counter_identity, realized_identity);
  ASSERT_TRUE(poly_uop_reachable(ctx, call, (PolyUOp *)counter_identity));

  free(map_repl);
  free(map_orig);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, transform_to_call_merges_contiguous_after_provenance) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *target = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CPU);
  PolyUOp *value = poly_add(ctx, target, poly_const_float(ctx, 1.0f));
  PolyUOp *store = poly_store_val(ctx, target, value);
  PolyUOp *after_src[2] = {target, store};
  PolyUOp *after = poly_uop(ctx, POLY_OP_AFTER, target->dtype, after_src, 2, poly_arg_none());
  PolyUOp *contiguous = poly_contiguous(ctx, after);
  ASSERT_NOT_NULL(after);
  ASSERT_NOT_NULL(contiguous);

  /* Pinned callify.py:159-160 concatenates AFTER then CONTIGUOUS provenance.
   * finalize_after maps both original roots to the same stripped storage and
   * removes the pass-local tag before returning the executable graph. */
  PolyUOp *realized = NULL;
  PolyUOp **map_orig = NULL;
  PolyUOp **map_repl = NULL;
  int map_n = 0;
  PolyUOp *call =
      poly_transform_to_call_with_map(ctx, &contiguous, 1, &realized, &map_orig, &map_repl, &map_n);
  ASSERT_NOT_NULL(call);
  ASSERT_NOT_NULL(realized);

  PolyUOp *after_replacement = NULL;
  PolyUOp *contiguous_replacement = NULL;
  int after_rows = 0, contiguous_rows = 0;
  for (int i = 0; i < map_n; i++) {
    if (map_orig[i] == after) {
      after_rows++;
      after_replacement = map_repl[i];
    }
    if (map_orig[i] == contiguous) {
      contiguous_rows++;
      contiguous_replacement = map_repl[i];
    }
  }
  ASSERT_INT_EQ(after_rows, 1);
  ASSERT_INT_EQ(contiguous_rows, 1);
  ASSERT_PTR_EQ(after_replacement, contiguous_replacement);
  ASSERT_PTR_EQ(
      poly_uop_get_buffer_identity(after_replacement),
      poly_uop_get_buffer_identity(contiguous_replacement)
  );

  PolyUOp *roots[4] = {call, realized, after_replacement, contiguous_replacement};
  for (int root = 0; root < 4; root++) {
    int n_topo = 0;
    PolyUOp **topo = poly_toposort(ctx, roots[root], &n_topo);
    for (int i = 0; i < n_topo; i++)
      ASSERT_FALSE(topo[i]->tag == INT32_MIN && topo[i]->tag_arg.kind == POLY_ARG_INT_TUPLE);
  }

  free(map_repl);
  free(map_orig);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, transform_to_call_three_source_after_keeps_independent_effects) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *target = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CPU);
  PolyUOp *host = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_HOST);
  PolyUOp *device = poly_device_uop(ctx, POLY_DEVICE_CPU);
  PolyUOp *creation_copy = poly_copy_to_device_uop(ctx, host, device);
  PolyUOp *primary_store = poly_store_val(ctx, target, creation_copy);

  PolyUOp *extra_target = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CPU);
  PolyUOp *extra_store = poly_store_val(ctx, extra_target, poly_const_float(ctx, 7.0f));
  PolyUOp *after_src[3] = {target, primary_store, extra_store};
  PolyUOp *after = poly_uop(ctx, POLY_OP_AFTER, target->dtype, after_src, 3, poly_arg_none());
  ASSERT_NOT_NULL(creation_copy);
  ASSERT_NOT_NULL(primary_store);
  ASSERT_NOT_NULL(extra_store);
  ASSERT_NOT_NULL(after);

  /* Pinned callify.py:35-39 uses strict two-source assignment patterns. This
   * three-source AFTER is handled by untagged apply_after; its creation COPY is
   * materialized independently and its extra effect is not dropped. */
  PolyUOp *realized = NULL;
  PolyUOp **map_orig = NULL;
  PolyUOp **map_repl = NULL;
  int map_n = 0;
  PolyUOp *call =
      poly_transform_to_call_with_map(ctx, &after, 1, &realized, &map_orig, &map_repl, &map_n);
  ASSERT_NOT_NULL(call);
  ASSERT_NOT_NULL(realized);
  ASSERT_INT_EQ(count_root_ops(ctx, call, POLY_OP_STORE), 3);

  PolyUOp *after_replacement = NULL;
  PolyUOp *copy_replacement = NULL;
  int after_rows = 0, copy_rows = 0;
  for (int i = 0; i < map_n; i++) {
    if (map_orig[i] == after) {
      after_rows++;
      after_replacement = map_repl[i];
    }
    if (map_orig[i] == creation_copy) {
      copy_rows++;
      copy_replacement = map_repl[i];
    }
  }
  ASSERT_INT_EQ(after_rows, 1);
  ASSERT_INT_EQ(copy_rows, 1);
  ASSERT_PTR_EQ(poly_uop_get_buffer_identity(after_replacement), target);
  ASSERT_PTR_NEQ(
      poly_uop_get_buffer_identity(copy_replacement),
      poly_uop_get_buffer_identity(after_replacement)
  );

  free(map_repl);
  free(map_orig);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, transform_to_call_rejects_nonstorage_gated_store) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *base = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 1, POLY_DEVICE_CPU);
  PolyUOp *one = poly_const_float(ctx, 1.0f);
  PolyUOp *target = poly_add(ctx, base, one);
  PolyUOp *value_base = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 1, POLY_DEVICE_CPU);
  PolyUOp *value = poly_add(ctx, value_base, one);
  PolyUOp *gate = poly_uop0(ctx, POLY_OP_CONST, POLY_BOOL, poly_arg_bool(false));
  PolyUOp *store_src[3] = {target, value, gate};
  PolyUOp *store = poly_uop(ctx, POLY_OP_STORE, POLY_VOID, store_src, 3, poly_arg_none());
  PolyUOp *after_src[2] = {target, store};
  PolyUOp *after = poly_uop(ctx, POLY_OP_AFTER, target->dtype, after_src, 2, poly_arg_none());
  ASSERT_NOT_NULL(after);

  /* Pinned callify.py:161-164 matches an exact two-source STORE before
   * replacing non-storage AFTER+STORE with CONTIGUOUS. spec_tensor rejects a
   * gated STORE whose target is not INDEX/SHRINK. Fail this malformed stage
   * graph instead of silently deleting its third source and false gate. */
  PolyUOp *realized = NULL;
  PolyUOp **map_orig = NULL;
  PolyUOp **map_repl = NULL;
  int map_n = 0;
  PolyUOp *call =
      poly_transform_to_call_with_map(ctx, &after, 1, &realized, &map_orig, &map_repl, &map_n);
  ASSERT_PTR_EQ(call, NULL);
  ASSERT_PTR_EQ(realized, NULL);
  ASSERT_PTR_EQ(map_orig, NULL);
  ASSERT_PTR_EQ(map_repl, NULL);
  ASSERT_INT_EQ(map_n, 0);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, transform_to_call_nested_three_source_after_keeps_map_key) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *target = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 1, POLY_DEVICE_CPU);
  PolyUOp *host = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 1, POLY_DEVICE_HOST);
  PolyUOp *device = poly_device_uop(ctx, POLY_DEVICE_CPU);
  PolyUOp *creation_copy = poly_copy_to_device_uop(ctx, host, device);
  PolyUOp *primary_store = poly_store_val(ctx, target, creation_copy);
  PolyUOp *extra_target = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 1, POLY_DEVICE_CPU);
  PolyUOp *extra_store = poly_store_val(ctx, extra_target, poly_const_float(ctx, 7.0f));
  PolyUOp *after_src[3] = {target, primary_store, extra_store};
  PolyUOp *after = poly_uop(ctx, POLY_OP_AFTER, target->dtype, after_src, 3, poly_arg_none());

  PolyUOp *one = poly_const_float(ctx, 1.0f);
  PolyUOp *outer_target = poly_add(ctx, after, one);
  PolyUOp *outer_store = poly_store_val(ctx, outer_target, poly_add(ctx, target, one));
  PolyUOp *outer_src[2] = {outer_target, outer_store};
  PolyUOp *outer = poly_uop(ctx, POLY_OP_AFTER, outer_target->dtype, outer_src, 2, poly_arg_none());
  ASSERT_NOT_NULL(after);
  ASSERT_NOT_NULL(outer);

  PolyUOp *realized = NULL;
  PolyUOp **map_orig = NULL;
  PolyUOp **map_repl = NULL;
  int map_n = 0;
  PolyUOp *call =
      poly_transform_to_call_with_map(ctx, &outer, 1, &realized, &map_orig, &map_repl, &map_n);
  ASSERT_NOT_NULL(call);
  ASSERT_NOT_NULL(realized);

  int after_rows = 0, outer_rows = 0;
  for (int i = 0; i < map_n; i++) {
    after_rows += map_orig[i] == after;
    outer_rows += map_orig[i] == outer;
  }
  ASSERT_INT_EQ(after_rows, 1);
  ASSERT_INT_EQ(outer_rows, 1);

  free(map_repl);
  free(map_orig);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, transform_to_call_does_not_publish_discarded_assignment_target_rows) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *source_logical = poly_test_buffer(ctx, POLY_FLOAT32, 1);
  PolyUOp *source_physical = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 1, POLY_DEVICE_HOST);
  PolyTensor *source = poly_tensor_create_with_roots(
      ctx, source_logical, source_physical, POLY_TENSOR_VALUE, POLY_DEVICE_HOST
  );
  PolyTensor *counter = poly_tensor_to_device(ctx, source, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(source);
  ASSERT_NOT_NULL(counter);
  PolyUOp *creation_copy = poly_tensor_uop_physical(counter);
  ASSERT_NOT_NULL(creation_copy);
  ASSERT_EQ(creation_copy->op, POLY_OP_COPY);

  PolyUOp *one = poly_const_float(ctx, 1.0f);
  PolyUOp *first_logical = NULL;
  PolyUOp *first_physical = NULL;
  for (int version = 0; version < 2; version++) {
    PolyTensor *next = poly_tensor_create_with_roots(
        ctx, poly_add(ctx, poly_tensor_uop_logical(counter), one),
        poly_add(ctx, poly_tensor_uop_physical(counter), one), POLY_TENSOR_VALUE, POLY_DEVICE_CPU
    );
    ASSERT_NOT_NULL(next);
    ASSERT_PTR_EQ(poly_tensor_assign(ctx, counter, next), counter);
    if (version == 0) {
      first_logical = poly_tensor_uop_logical(counter);
      first_physical = poly_tensor_uop_physical(counter);
    }
  }
  ASSERT_NOT_NULL(first_logical);
  ASSERT_NOT_NULL(first_physical);
  ASSERT_EQ(first_physical->op, POLY_OP_AFTER);

  PolyUOp *live_before = poly_tensor_uop_physical(counter);
  ASSERT_NOT_NULL(live_before);
  ASSERT_INT_EQ(count_root_ops(ctx, live_before, POLY_OP_AFTER), 2);
  ASSERT_INT_EQ(count_root_ops(ctx, live_before, POLY_OP_STORE), 2);
  ASSERT_INT_EQ(count_root_ops(ctx, live_before, POLY_OP_COPY), 1);

  /* Pinned callify.py:54-57,161-164 rewrites this non-storage assignment
   * target to CONTIGUOUS(replacement). Its separate finalize pass at
   * callify.py:203-220 therefore never publishes rows for the discarded
   * first version or creation COPY, even though another live Tensor retains
   * them. */
  PolyTensor *target = poly_tensor_create_with_roots(
      ctx, poly_add(ctx, first_logical, one), poly_add(ctx, first_physical, one), POLY_TENSOR_VALUE,
      POLY_DEVICE_CPU
  );
  PolyUOp *replacement_logical = poly_test_buffer(ctx, POLY_FLOAT32, 1);
  PolyUOp *replacement_host = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 1, POLY_DEVICE_HOST);
  PolyTensor *replacement_source = poly_tensor_create_with_roots(
      ctx, replacement_logical, replacement_host, POLY_TENSOR_VALUE, POLY_DEVICE_HOST
  );
  PolyTensor *replacement = poly_tensor_to_device(ctx, replacement_source, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(target);
  ASSERT_NOT_NULL(replacement_source);
  ASSERT_NOT_NULL(replacement);
  ASSERT_PTR_EQ(poly_tensor_assign(ctx, target, replacement), target);

  PolyUOp *requested = poly_tensor_uop_physical(target);
  PolyUOp *realized = NULL;
  PolyUOp **map_orig = NULL;
  PolyUOp **map_repl = NULL;
  int map_n = 0;
  PolyUOp *call =
      poly_transform_to_call_with_map(ctx, &requested, 1, &realized, &map_orig, &map_repl, &map_n);
  ASSERT_NOT_NULL(call);
  ASSERT_NOT_NULL(realized);
  ASSERT_TRUE(map_n > 0);
  for (int i = 0; i < map_n; i++) {
    ASSERT_PTR_NEQ(map_orig[i], first_physical);
    ASSERT_PTR_NEQ(map_orig[i], creation_copy);
  }

  ASSERT_INT_EQ(poly_tensor_apply_realize_map(ctx, map_orig, map_repl, map_n, POLY_DEVICE_AUTO), 0);
  ASSERT_PTR_EQ(poly_tensor_uop_physical(counter), live_before);
  ASSERT_INT_EQ(count_root_ops(ctx, poly_tensor_uop_physical(counter), POLY_OP_AFTER), 2);
  ASSERT_INT_EQ(count_root_ops(ctx, poly_tensor_uop_physical(counter), POLY_OP_STORE), 2);
  ASSERT_INT_EQ(count_root_ops(ctx, poly_tensor_uop_physical(counter), POLY_OP_COPY), 1);

  free(map_repl);
  free(map_orig);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, transform_to_call_view_copy_is_reachable_store_effect) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *base = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 2, POLY_DEVICE_HOST);
  float base_data[2] = {1.0f, 2.0f};
  PolyBuffer base_storage = {
      .ptr = base_data,
      .nbytes = sizeof(base_data),
      .device = POLY_DEVICE_HOST,
      .owned = false,
      .valid = true,
  };
  poly_buffer_attach(ctx, base, &base_storage);
  int64_t bounds[1][2] = {{0, 1}};
  PolyUOp *view = poly_shrink(ctx, base, bounds, 1);
  ASSERT_PTR_EQ(poly_uop_buffer(ctx, view), view);
  PolyUOp *device = poly_device_uop(ctx, POLY_DEVICE_CPU);
  PolyUOp *copy = poly_copy_to_device_uop(ctx, view, device);
  PolyUOp *realized = NULL;
  PolyUOp *call = poly_transform_to_call(ctx, &copy, 1, &realized);
  ASSERT_NOT_NULL(base);
  ASSERT_NOT_NULL(view);
  ASSERT_NOT_NULL(device);
  ASSERT_NOT_NULL(copy);
  ASSERT_NOT_NULL(call);
  ASSERT_NOT_NULL(realized);

  /* Current Tensor.transform_to_call keeps COPY beneath
   * AFTER(buffer, STORE(buffer, COPY)); scheduling creates the COPY CALL later
   * (tinygrad/tensor.py:35-62,178-240; schedule/rangeify.py:565-590). */
  ASSERT_INT_EQ(call->op, POLY_OP_CALL);
  ASSERT_INT_EQ(count_root_ops(ctx, call, POLY_OP_CALL), 1);
  ASSERT_INT_EQ(count_root_ops(ctx, call, POLY_OP_SINK), 1);
  ASSERT_INT_EQ(count_root_ops(ctx, call, POLY_OP_AFTER), 1);
  ASSERT_INT_EQ(count_root_ops(ctx, call, POLY_OP_STORE), 1);
  ASSERT_INT_EQ(count_root_ops(ctx, call, POLY_OP_COPY), 1);
  ASSERT_TRUE(poly_uop_reachable(ctx, call, view));

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, view_copy_uses_existing_movement_uop_as_storage_argument) {
  /* Tinygrad 2026-08-22/a9069c177a9d keeps the exact SHRINK as the COPY input
   * and attaches Buffer.view metadata outside shared IR (tensor.py:178-220,
   * uop/ops.py:907-934). */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  PolyUOp *base = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_HOST);
  PolyBuffer storage = {
      .ptr = data,
      .nbytes = sizeof(data),
      .device = POLY_DEVICE_HOST,
      .owned = false,
      .valid = true,
  };
  poly_buffer_attach(ctx, base, &storage);
  int64_t bounds[1][2] = {{1, 3}};
  PolyUOp *view = poly_shrink(ctx, base, bounds, 1);
  ASSERT_NOT_NULL(view);
  ASSERT_PTR_EQ(poly_uop_buffer(ctx, view), view);

  PolyUOp *copy = poly_copy_to_device_uop(ctx, view, poly_device_uop(ctx, POLY_DEVICE_CPU));
  PolyUOp *realized = NULL;
  PolyUOp *call = poly_transform_to_call(ctx, &copy, 1, &realized);
  ASSERT_NOT_NULL(call);
  ASSERT_NOT_NULL(realized);
  ASSERT_TRUE(poly_uop_reachable(ctx, call, view));

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, transform_to_call_keeps_only_reachable_view_copy_effects) {
  for (int mode = 0; mode < 3; mode++) {
    bool live = mode != 0;
    bool nested = mode == 2;
    PolyCtx *ctx = poly_ctx_new();
    ASSERT_NOT_NULL(ctx);

    float data[2] = {3.0f, 4.0f};
    PolyUOp *base = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 2, POLY_DEVICE_HOST);
    ASSERT_NOT_NULL(base);
    PolyBuffer host_storage = {
        .ptr = data,
        .nbytes = sizeof(data),
        .device = POLY_DEVICE_HOST,
        .owned = false,
        .valid = true,
    };
    poly_buffer_attach(ctx, base, &host_storage);
    int64_t bounds[1][2] = {{0, 1}};
    PolyUOp *view = poly_shrink(ctx, base, bounds, 1);
    ASSERT_PTR_EQ(poly_uop_buffer(ctx, view), view);
    PolyUOp *device = poly_device_uop(ctx, POLY_DEVICE_CPU);
    PolyUOp *copy = poly_copy_to_device_uop(ctx, view, device);
    ASSERT_NOT_NULL(view);
    ASSERT_NOT_NULL(device);
    ASSERT_NOT_NULL(copy);

    PolyUOp *one = poly_const_float(ctx, 1.0f);
    PolyUOp *dead_target = poly_add(ctx, copy, one);
    PolyUOp *live_buffer = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 1, POLY_DEVICE_CPU);
    PolyUOp *live_value = poly_add(ctx, live_buffer, one);
    PolyUOp *requested = nested ? dead_target : copy;
    if (!live) {
      PolyUOp *store = poly_store_val(ctx, dead_target, live_value);
      PolyUOp *after_src[2] = {dead_target, store};
      requested = poly_uop(ctx, POLY_OP_AFTER, dead_target->dtype, after_src, 2, poly_arg_none());
    }
    ASSERT_NOT_NULL(one);
    ASSERT_NOT_NULL(dead_target);
    ASSERT_NOT_NULL(live_buffer);
    ASSERT_NOT_NULL(live_value);
    ASSERT_NOT_NULL(requested);
    PolyTensor *retained_copy =
        nested ? poly_tensor_create_with_roots(ctx, copy, copy, POLY_TENSOR_VALUE, POLY_DEVICE_CPU)
               : NULL;
    if (nested) ASSERT_NOT_NULL(retained_copy);

    PolyUOp *realized = NULL;
    PolyUOp **map_orig = NULL;
    PolyUOp **map_repl = NULL;
    int map_n = 0;
    PolyUOp *call_graph = poly_transform_to_call_with_map(
        ctx, &requested, 1, &realized, &map_orig, &map_repl, &map_n
    );
    ASSERT_NOT_NULL(call_graph);
    ASSERT_EQ(call_graph->op, POLY_OP_CALL);
    ASSERT_NOT_NULL(realized);

    int copy_rows = 0, requested_rows = 0, synthetic_rows = 0;
    PolyUOp *copy_replacement = NULL;
    for (int i = 0; i < map_n; i++) {
      if (map_orig[i] == copy) {
        copy_rows++;
        copy_replacement = map_repl[i];
      }
      requested_rows += map_orig[i] == requested;
      synthetic_rows += !poly_uop_reachable(ctx, requested, map_orig[i]);
    }
    int n_topo = 0, marker_nodes = 0;
    PolyUOp **topo = poly_toposort(ctx, call_graph, &n_topo);
    for (int i = 0; i < n_topo; i++)
      marker_nodes += topo[i]->tag == INT32_MIN && topo[i]->tag_arg.kind == POLY_ARG_INT_TUPLE;
    ASSERT_INT_EQ(marker_nodes, 0);
    if (live) {
      /* Current callify keeps the view COPY inside reachable AFTER/STORE
       * effects; create_linear_with_vars extracts ordered calls later
       * (tinygrad/tensor.py:178-240; schedule/rangeify.py:565-590). */
      ASSERT_INT_EQ(count_root_ops(ctx, call_graph, POLY_OP_CALL), 1);
      ASSERT_INT_EQ(count_root_ops(ctx, call_graph, POLY_OP_SINK), 1);
      ASSERT_INT_EQ(count_root_ops(ctx, call_graph, POLY_OP_AFTER), nested ? 2 : 1);
      ASSERT_INT_EQ(count_root_ops(ctx, call_graph, POLY_OP_STORE), nested ? 2 : 1);
      ASSERT_INT_EQ(count_root_ops(ctx, call_graph, POLY_OP_COPY), 1);
      ASSERT_TRUE(poly_uop_reachable(ctx, call_graph, view));
      ASSERT_INT_EQ(copy_rows, 1);
      ASSERT_INT_EQ(requested_rows, 1);
      ASSERT_INT_EQ(synthetic_rows, 0);
      ASSERT_INT_EQ(map_n, nested ? 2 : 1);
      if (nested) {
        ASSERT_NOT_NULL(copy_replacement);
        ASSERT_INT_EQ(
            poly_tensor_apply_realize_map(ctx, map_orig, map_repl, map_n, POLY_DEVICE_AUTO), 0
        );
        ASSERT_PTR_EQ(poly_tensor_uop_physical(retained_copy), copy_replacement);
        ASSERT_PTR_NEQ(poly_tensor_uop_physical(retained_copy), copy);
      }
    } else {
      /* Current callify drops the replaced non-storage assignment target;
       * its dead view/COPY cannot reach the returned CALL. */
      ASSERT_INT_EQ(count_root_ops(ctx, call_graph, POLY_OP_CALL), 1);
      ASSERT_INT_EQ(count_root_ops(ctx, call_graph, POLY_OP_SINK), 1);
      ASSERT_INT_EQ(count_root_ops(ctx, call_graph, POLY_OP_AFTER), 1);
      ASSERT_INT_EQ(count_root_ops(ctx, call_graph, POLY_OP_STORE), 1);
      ASSERT_INT_EQ(count_root_ops(ctx, call_graph, POLY_OP_COPY), 0);
      ASSERT_FALSE(poly_uop_reachable(ctx, call_graph, view));
      ASSERT_INT_EQ(copy_rows, 0);
      ASSERT_INT_EQ(requested_rows, 1);
      ASSERT_INT_EQ(synthetic_rows, 0);
    }

    free(map_repl);
    free(map_orig);
    poly_ctx_destroy(ctx);
  }
  PASS();
}

TEST(realize, transform_to_call_dead_map_rows_ignore_opaque_callee_bodies) {
  const PolyOps opaque_ops[] = {POLY_OP_CALL, POLY_OP_FUNCTION};
  for (int op_index = 0; op_index < 2; op_index++) {
    PolyCtx *ctx = poly_ctx_new();
    ASSERT_NOT_NULL(ctx);

    PolyUOp *state = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 1, POLY_DEVICE_CPU);
    PolyUOp *one = poly_const_float(ctx, 1.0f);
    ASSERT_NOT_NULL(state);
    ASSERT_NOT_NULL(one);

    PolyUOp *state_store = poly_store_val(ctx, state, poly_add(ctx, state, one));
    PolyUOp *state_after_src[2] = {state, state_store};
    PolyUOp *discarded =
        poly_uop(ctx, POLY_OP_AFTER, state->dtype, state_after_src, 2, poly_arg_none());
    ASSERT_NOT_NULL(discarded);
    ASSERT_EQ(discarded->op, POLY_OP_AFTER);

    PolyUOp *target = poly_add(ctx, discarded, one);
    PolyUOp *replacement_host = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 1, POLY_DEVICE_HOST);
    PolyUOp *replacement_device = poly_device_uop(ctx, POLY_DEVICE_CPU);
    PolyUOp *replacement = poly_copy_to_device_uop(ctx, replacement_host, replacement_device);
    PolyUOp *target_store = poly_store_val(ctx, target, replacement);
    PolyUOp *requested_src[2] = {target, target_store};
    PolyUOp *requested_assign =
        poly_uop(ctx, POLY_OP_AFTER, target->dtype, requested_src, 2, poly_arg_none());
    ASSERT_NOT_NULL(target);
    ASSERT_NOT_NULL(replacement_host);
    ASSERT_NOT_NULL(replacement_device);
    ASSERT_NOT_NULL(replacement);
    ASSERT_NOT_NULL(requested_assign);

    /* Pinned graph_rewrite(..., enter_calls=False) keeps src[0] opaque.  The
     * same discarded state version appearing only as a callee body must not
     * make its early-rewrite map row caller-visible. */
    PolyUOp *body = poly_sink_n(ctx, &discarded, 1);
    PolyUOp *opaque_src[2] = {body, state};
    PolyUOp *opaque =
        poly_uop(ctx, opaque_ops[op_index], POLY_VOID, opaque_src, 2, poly_arg_none());
    PolyUOp *keep_buffer = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 1, POLY_DEVICE_CPU);
    PolyUOp *keep_src[2] = {keep_buffer, opaque};
    PolyUOp *keep = poly_uop(ctx, POLY_OP_AFTER, keep_buffer->dtype, keep_src, 2, poly_arg_none());
    ASSERT_NOT_NULL(body);
    ASSERT_NOT_NULL(opaque);
    ASSERT_NOT_NULL(keep);
    ASSERT_PTR_EQ(opaque->src[0], body);
    ASSERT_PTR_EQ(body->src[0], discarded);

    PolyUOp *requested[2] = {requested_assign, keep};
    PolyUOp *realized[2] = {NULL, NULL};
    PolyUOp **map_orig = NULL;
    PolyUOp **map_repl = NULL;
    int map_n = 0;
    PolyUOp *call =
        poly_transform_to_call_with_map(ctx, requested, 2, realized, &map_orig, &map_repl, &map_n);
    ASSERT_NOT_NULL(call);
    ASSERT_NOT_NULL(realized[0]);
    ASSERT_NOT_NULL(realized[1]);
    ASSERT_TRUE(map_n > 0);
    for (int i = 0; i < map_n; i++) {
      ASSERT_PTR_NEQ(map_orig[i], discarded);
    }
    ASSERT_PTR_EQ(opaque->src[0], body);
    ASSERT_PTR_EQ(body->src[0], discarded);

    free(map_repl);
    free(map_orig);
    poly_ctx_destroy(ctx);
  }
  PASS();
}

TEST(realize, transform_to_call_does_not_publish_synthetic_early_rewrite_keys) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *state = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 1, POLY_DEVICE_CPU);
  PolyUOp *one = poly_const_float(ctx, 1.0f);
  PolyUOp *state_store = poly_store_val(ctx, state, poly_add(ctx, state, one));
  PolyUOp *state_after_src[2] = {state, state_store};
  PolyUOp *first = poly_uop(ctx, POLY_OP_AFTER, state->dtype, state_after_src, 2, poly_arg_none());
  PolyUOp *target = poly_add(ctx, first, one);

  PolyUOp *host = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 1, POLY_DEVICE_HOST);
  PolyUOp *cpu_device = poly_device_uop(ctx, POLY_DEVICE_CPU);
  PolyUOp *replacement = poly_copy_to_device_uop(ctx, host, cpu_device);
  PolyUOp *alias = poly_contiguous(ctx, replacement);
  PolyTensor *alias_tensor =
      poly_tensor_create_with_roots(ctx, alias, alias, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);

  PolyUOp *requested_store = poly_store_val(ctx, target, replacement);
  PolyUOp *requested_src[2] = {target, requested_store};
  PolyUOp *requested =
      poly_uop(ctx, POLY_OP_AFTER, target->dtype, requested_src, 2, poly_arg_none());
  ASSERT_NOT_NULL(state);
  ASSERT_NOT_NULL(first);
  ASSERT_NOT_NULL(replacement);
  ASSERT_NOT_NULL(alias);
  ASSERT_NOT_NULL(alias_tensor);
  ASSERT_NOT_NULL(requested);

  PolyUOp *realized = NULL;
  PolyUOp **map_orig = NULL;
  PolyUOp **map_repl = NULL;
  int map_n = 0;
  PolyUOp *call =
      poly_transform_to_call_with_map(ctx, &requested, 1, &realized, &map_orig, &map_repl, &map_n);
  ASSERT_NOT_NULL(call);
  ASSERT_NOT_NULL(realized);
  ASSERT_TRUE(map_n > 0);

  bool found_requested = false;
  bool found_replacement = false;
  for (int i = 0; i < map_n; i++) {
    ASSERT_PTR_NEQ(map_orig[i], alias);
    if (map_orig[i] == requested) found_requested = true;
    if (map_orig[i] == replacement) found_replacement = true;
  }
  ASSERT_TRUE(found_requested);
  ASSERT_TRUE(found_replacement);

  ASSERT_PTR_EQ(poly_tensor_uop_physical(alias_tensor), alias);

  free(map_repl);
  free(map_orig);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, transform_to_call_surviving_tags_disambiguate_shared_rewrites) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *buffer = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 1, POLY_DEVICE_CPU);
  PolyUOp *one = poly_const_float(ctx, 1.0f);
  PolyUOp *two = poly_const_float(ctx, 2.0f);
  PolyUOp *shared_replacement = poly_add(ctx, buffer, one);

  PolyUOp *dead_target = poly_add(ctx, buffer, two);
  PolyUOp *dead_store = poly_store_val(ctx, dead_target, shared_replacement);
  PolyUOp *dead_src[2] = {dead_target, dead_store};
  PolyUOp *dead = poly_uop(ctx, POLY_OP_AFTER, dead_target->dtype, dead_src, 2, poly_arg_none());
  PolyUOp *outer_target = poly_add(ctx, dead, one);
  PolyUOp *outer_value = poly_mul(ctx, buffer, two);
  PolyUOp *outer_store = poly_store_val(ctx, outer_target, outer_value);
  PolyUOp *outer_src[2] = {outer_target, outer_store};
  PolyUOp *outer = poly_uop(ctx, POLY_OP_AFTER, outer_target->dtype, outer_src, 2, poly_arg_none());

  PolyUOp *live_target = poly_mul(ctx, buffer, one);
  PolyUOp *live_store = poly_store_val(ctx, live_target, shared_replacement);
  PolyUOp *live_src[2] = {live_target, live_store};
  PolyUOp *live = poly_uop(ctx, POLY_OP_AFTER, live_target->dtype, live_src, 2, poly_arg_none());
  ASSERT_NOT_NULL(dead);
  ASSERT_NOT_NULL(outer);
  ASSERT_NOT_NULL(live);

  PolyUOp *requested[2] = {outer, live};
  PolyUOp *realized[2] = {NULL, NULL};
  PolyUOp **map_orig = NULL;
  PolyUOp **map_repl = NULL;
  int map_n = 0;
  PolyUOp *call =
      poly_transform_to_call_with_map(ctx, requested, 2, realized, &map_orig, &map_repl, &map_n);
  ASSERT_NOT_NULL(call);
  ASSERT_NOT_NULL(realized[0]);
  ASSERT_NOT_NULL(realized[1]);

  int dead_rows = 0, outer_rows = 0, live_rows = 0;
  for (int i = 0; i < map_n; i++) {
    dead_rows += map_orig[i] == dead;
    outer_rows += map_orig[i] == outer;
    live_rows += map_orig[i] == live;
  }
  ASSERT_INT_EQ(dead_rows, 0);
  ASSERT_INT_EQ(outer_rows, 1);
  ASSERT_INT_EQ(live_rows, 1);

  PolyUOp *roots[3] = {call, realized[0], realized[1]};
  for (int root = 0; root < 3; root++) {
    int n_topo = 0;
    PolyUOp **topo = poly_toposort(ctx, roots[root], &n_topo);
    for (int i = 0; i < n_topo; i++)
      ASSERT_FALSE(topo[i]->tag == INT32_MIN && topo[i]->tag_arg.kind == POLY_ARG_INT_TUPLE);
  }

  free(map_repl);
  free(map_orig);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, nested_contiguous_inside_after_store_precedes_consumer) {
  PolyCtx *ctx = poly_ctx_new();
  poly_ctx_set_preferred_device(ctx, POLY_DEVICE_INTERP);

  float source_value = 1.0f, target_value = 0.0f;
  int f32 = poly_dtype_id_by_name("float32");
  int64_t shape[] = {1};
  PolyUOp *target_buffer = NULL;
  PolyTensor *source =
      initialized_f32_tensor(ctx, shape, 1, &source_value, POLY_DEVICE_INTERP, NULL);
  PolyTensor *target =
      initialized_f32_tensor(ctx, shape, 1, &target_value, POLY_DEVICE_INTERP, &target_buffer);
  PolyTensor *one = poly_tensor_const_float_by_id(ctx, 1.0, f32, POLY_DEVICE_INTERP);
  PolyTensor *five = poly_tensor_const_float_by_id(ctx, 5.0, f32, POLY_DEVICE_INTERP);
  PolyTensor *inner = poly_tensor_contiguous(ctx, poly_tensor_alu2(ctx, POLY_OP_ADD, source, one));
  ASSERT_PTR_EQ(poly_tensor_assign(ctx, target, inner), target);
  PolyTensor *tensor =
      poly_tensor_contiguous(ctx, poly_tensor_alu2(ctx, POLY_OP_ADD, target, five));
  ASSERT_NOT_NULL(tensor);
  PolyTensor *realized = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &tensor, 1, &realized), 0);
  ASSERT_PTR_EQ(realized, tensor);

  float requested_out = 0.0f, target_out = 0.0f;
  READ_REALIZED_F32(ctx, poly_tensor_uop(tensor), &requested_out, 1);
  READ_REALIZED_F32(ctx, target_buffer, &target_out, 1);
  ASSERT_FLOAT_EQ(requested_out, 7.0f, 1e-5f);
  ASSERT_FLOAT_EQ(target_out, 2.0f, 1e-5f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, nested_after_store_consumer_indexes_producer_buffer_once) {
  PolyCtx *ctx = poly_ctx_new();
  poly_ctx_set_preferred_device(ctx, POLY_DEVICE_CPU);

  PolyUOp *inner_buffer = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CPU);
  PolyUOp *outer_buffer = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(inner_buffer);
  ASSERT_NOT_NULL(outer_buffer);
  ASSERT_INT_EQ(poly_buffer_allocate(ctx, inner_buffer, POLY_DEVICE_CPU), 0);
  ASSERT_INT_EQ(poly_buffer_allocate(ctx, outer_buffer, POLY_DEVICE_CPU), 0);

  int64_t shape[1] = {4};
  int64_t singleton[1] = {1};
  PolyUOp *ones =
      poly_expand(ctx, poly_reshape(ctx, poly_const_float(ctx, 1.0), singleton, 1), shape, 1);
  PolyUOp *inner_store = poly_store_val(ctx, inner_buffer, ones);
  PolyUOp *inner_after_src[2] = {inner_buffer, inner_store};
  PolyUOp *inner_after =
      poly_uop(ctx, POLY_OP_AFTER, inner_buffer->dtype, inner_after_src, 2, poly_arg_none());
  PolyUOp *outer_store = poly_store_val(ctx, outer_buffer, inner_after);
  PolyUOp *outer_after_src[2] = {outer_buffer, outer_store};
  PolyUOp *outer_after =
      poly_uop(ctx, POLY_OP_AFTER, outer_buffer->dtype, outer_after_src, 2, poly_arg_none());
  ASSERT_NOT_NULL(ones);
  ASSERT_NOT_NULL(inner_store);
  ASSERT_NOT_NULL(inner_after);
  ASSERT_NOT_NULL(outer_store);
  ASSERT_NOT_NULL(outer_after);

  PolyUOp *scheduled_out = NULL;
  PolyUOp *schedule = poly_test_linear_values(ctx, &outer_after, 1, &scheduled_out);
  ASSERT_NOT_NULL(schedule);
  ASSERT_NOT_NULL(scheduled_out);
  ASSERT_INT_EQ(schedule->n_src, 2);
  ASSERT_INT_EQ(count_root_ops(ctx, poly_test_linear_call_body(schedule, 1), POLY_OP_ADD), 0);
  ASSERT_INT_EQ(poly_run_linear(ctx, schedule, NULL, 0, NULL, 0, true, false, false), 0);

  float inner_values[4] = {0};
  float outer_values[4] = {0};
  ASSERT_INT_EQ(poly_buffer_read(ctx, inner_buffer, inner_values, sizeof(inner_values)), 0);
  ASSERT_INT_EQ(poly_buffer_read(ctx, outer_buffer, outer_values, sizeof(outer_values)), 0);
  for (int i = 0; i < 4; i++) {
    ASSERT_FLOAT_EQ(inner_values[i], 1.0f, 1e-6f);
    ASSERT_FLOAT_EQ(outer_values[i], 1.0f, 1e-6f);
  }

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, nested_after_store_through_view_commits_dependency_before_assign) {
  PolyCtx *ctx = poly_ctx_new();
  poly_ctx_set_preferred_device(ctx, POLY_DEVICE_CPU);

  PolyUOp *inner_buffer = poly_buffer_f32(ctx, 1);
  PolyUOp *outer_buffer = poly_buffer_f32(ctx, 1);
  PolyUOp *source = poly_buffer_f32(ctx, 1);
  float inner_init = 0.0f, outer_init = 0.0f, source_init = 1.0f;
  poly_buffer_set(ctx, inner_buffer, &inner_init, sizeof(inner_init), POLY_DEVICE_CPU);
  poly_buffer_set(ctx, outer_buffer, &outer_init, sizeof(outer_init), POLY_DEVICE_CPU);
  poly_buffer_set(ctx, source, &source_init, sizeof(source_init), POLY_DEVICE_CPU);

  PolyUOp *inner_store = poly_store_val(ctx, inner_buffer, source);
  PolyUOp *inner_after_src[2] = {inner_buffer, inner_store};
  PolyUOp *inner_after =
      poly_uop(ctx, POLY_OP_AFTER, inner_buffer->dtype, inner_after_src, 2, poly_arg_none());

  int64_t one[1] = {1};
  PolyUOp *outer_view =
      poly_expand(ctx, poly_reshape(ctx, poly_reshape(ctx, outer_buffer, one, 1), one, 1), one, 1);
  PolyUOp *outer_value = poly_reshape(
      ctx,
      poly_alu2(
          ctx, POLY_OP_ADD, poly_alu2(ctx, POLY_OP_MUL, poly_const_float(ctx, 0.85f), outer_view),
          inner_after
      ),
      one, 1
  );
  PolyUOp *outer_store = poly_store_val(ctx, outer_view, outer_value);
  PolyUOp *outer_after_src[2] = {outer_view, outer_store};
  PolyUOp *outer_after =
      poly_uop(ctx, POLY_OP_AFTER, outer_view->dtype, outer_after_src, 2, poly_arg_none());
  ASSERT_NOT_NULL(inner_after);
  ASSERT_NOT_NULL(outer_view);
  ASSERT_NOT_NULL(outer_after);

  PolyUOp *scheduled_out = NULL;
  PolyUOp *schedule = poly_test_linear_values(ctx, &outer_after, 1, &scheduled_out);
  ASSERT_NOT_NULL(schedule);
  ASSERT_PTR_EQ(scheduled_out, outer_view);
  ASSERT_INT_EQ(schedule->n_src, 2);
  ASSERT_INT_EQ(poly_run_linear(ctx, schedule, NULL, 0, NULL, 0, true, false, false), 0);

  float inner_out = 0.0f, outer_out = 0.0f;
  ASSERT_INT_EQ(poly_buffer_read(ctx, inner_buffer, &inner_out, sizeof(inner_out)), 0);
  ASSERT_INT_EQ(poly_buffer_read(ctx, outer_buffer, &outer_out, sizeof(outer_out)), 0);
  ASSERT_FLOAT_EQ(inner_out, 1.0f, 1e-6f);
  ASSERT_FLOAT_EQ(outer_out, 1.0f, 1e-6f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, batched_contiguous_inside_after_store_precedes_consumer) {
  uint64_t kernel_counts[2] = {0, 0};
  for (int reverse = 0; reverse < 2; reverse++) {
    PolyCtx *ctx = poly_ctx_new();
    poly_ctx_set_preferred_device(ctx, POLY_DEVICE_INTERP);

    float source_value = 1.0f, target_value = 0.0f;
    int f32 = poly_dtype_id_by_name("float32");
    int64_t shape[] = {1};
    PolyUOp *target_buffer = NULL;
    PolyTensor *source =
        initialized_f32_tensor(ctx, shape, 1, &source_value, POLY_DEVICE_INTERP, NULL);
    PolyTensor *target =
        initialized_f32_tensor(ctx, shape, 1, &target_value, POLY_DEVICE_INTERP, &target_buffer);
    PolyTensor *one = poly_tensor_const_float_by_id(ctx, 1.0, f32, POLY_DEVICE_INTERP);
    PolyTensor *five = poly_tensor_const_float_by_id(ctx, 5.0, f32, POLY_DEVICE_INTERP);
    PolyTensor *inner_tensor =
        poly_tensor_contiguous(ctx, poly_tensor_alu2(ctx, POLY_OP_ADD, source, one));
    ASSERT_PTR_EQ(poly_tensor_assign(ctx, target, inner_tensor), target);
    PolyTensor *requested_tensor =
        poly_tensor_contiguous(ctx, poly_tensor_alu2(ctx, POLY_OP_ADD, target, five));
    ASSERT_NOT_NULL(inner_tensor);
    ASSERT_NOT_NULL(requested_tensor);

    PolyTensor *inputs[2] = {
        reverse ? requested_tensor : inner_tensor,
        reverse ? inner_tensor : requested_tensor,
    };
    PolyTensor *outputs[2] = {NULL, NULL};
    poly_ctx_reset_counters(ctx);
    ASSERT_INT_EQ(poly_realize_tensors(ctx, inputs, 2, outputs), 0);
    ASSERT_PTR_EQ(outputs[0], inputs[0]);
    ASSERT_PTR_EQ(outputs[1], inputs[1]);

    float inner_out = 0.0f, requested_out = 0.0f, target_out = 0.0f;
    READ_REALIZED_F32(ctx, poly_tensor_uop(inner_tensor), &inner_out, 1);
    READ_REALIZED_F32(ctx, poly_tensor_uop(requested_tensor), &requested_out, 1);
    READ_REALIZED_F32(ctx, target_buffer, &target_out, 1);
    ASSERT_FLOAT_EQ(inner_out, 2.0f, 1e-5f);
    ASSERT_FLOAT_EQ(requested_out, 7.0f, 1e-5f);
    ASSERT_FLOAT_EQ(target_out, 2.0f, 1e-5f);

    PolyCtxStats stats = {0};
    ASSERT_INT_EQ(poly_ctx_stats(ctx, &stats), 0);
    kernel_counts[reverse] = stats.kernel_count;
    poly_ctx_destroy(ctx);
  }
  ASSERT_TRUE(kernel_counts[0] == kernel_counts[1]);
  PASS();
}

TEST(realize, transform_to_call_wraps_sink_body_in_call) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer_f32(ctx, 4);
  PolyUOp *b = poly_buffer_f32(ctx, 4);
  PolyUOp *add = poly_alu2(ctx, POLY_OP_ADD, a, b);
  PolyUOp *mul = poly_alu2(ctx, POLY_OP_MUL, a, b);

  PolyUOp *targets[] = {add, mul};
  PolyUOp *realized[] = {NULL, NULL};
  PolyUOp *big_call = poly_transform_to_call(ctx, targets, 2, realized);
  ASSERT_NOT_NULL(big_call);
  ASSERT_INT_EQ(big_call->op, POLY_OP_CALL);
  ASSERT_TRUE(big_call->n_src >= 3);
  ASSERT_TRUE(big_call->src[0] != NULL);
  ASSERT_INT_EQ(big_call->src[0]->op, POLY_OP_SINK);
  ASSERT_INT_EQ(count_root_ops(ctx, big_call, POLY_OP_CALL), 1);
  ASSERT_INT_EQ(count_root_ops(ctx, big_call, POLY_OP_STORE), 2);
  ASSERT_NOT_NULL(realized[0]);
  ASSERT_NOT_NULL(realized[1]);
  ASSERT_TRUE(poly_uop_get_buffer_identity(realized[0]) != NULL);
  ASSERT_TRUE(poly_uop_get_buffer_identity(realized[1]) != NULL);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, precompiled_copied_output_keeps_dependency_on_store_value) {
  /* tensor.py:transform_precompiled_call places effects on the STORE value,
   * not as unordered siblings of the STORE in its destination AFTER. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  poly_ctx_set_preferred_device(ctx, POLY_DEVICE_INTERP);
  PolyUOp *buffer = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 2, POLY_DEVICE_INTERP);
  PolyUOp *param = poly_uop_param(ctx, 0, buffer);
  PolyUOp *dep = poly_uop2(
      ctx, POLY_OP_CALL, POLY_VOID, poly_uop0(ctx, POLY_OP_NOOP, POLY_VOID, poly_arg_none()), param,
      poly_arg_none()
  );
  PolyUOp *value = poly_add(ctx, param, param);
  PolyUOp *after = poly_uop2(ctx, POLY_OP_AFTER, value->dtype, value, dep, poly_arg_none());
  PolyUOp *tuple = poly_uop1(ctx, POLY_OP_TUPLE, POLY_VOID, after, poly_arg_none());
  PolyCallInfo info = {.precompile = true};
  PolyUOp *function =
      poly_uop2(ctx, POLY_OP_FUNCTION, POLY_VOID, tuple, buffer, poly_arg_call_info(&info));
  PolyUOp *requested = poly_uop1(ctx, POLY_OP_GETTUPLE, POLY_FLOAT32, function, poly_arg_int(0));
  PolyUOp *realized = NULL;
  PolyUOp *callified = poly_transform_to_call(ctx, &requested, 1, &realized);
  ASSERT_NOT_NULL(callified);
  int n_topo = 0;
  PolyUOp **topo = poly_toposort_alloc(ctx, callified, &n_topo);
  PolyUOp *call = NULL;
  for (int i = 0; i < n_topo; i++)
    if (topo[i]->op == POLY_OP_CALL && topo[i]->arg.kind == POLY_ARG_CALL_INFO &&
        topo[i]->arg.call_info->precompile)
      call = topo[i];
  ASSERT_NOT_NULL(call);
  PolyUOp *item = call->src[0]->src[0];
  bool correct = item->op == POLY_OP_AFTER && item->n_src == 2 &&
                 item->src[1]->op == POLY_OP_STORE && item->src[1]->src[1]->op == POLY_OP_AFTER &&
                 item->src[1]->src[1]->src[1] == dep;
  poly_toposort_free(topo);
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(correct);
  PASS();
}

TEST(realize, precompiled_function_becomes_opaque_output_call) {
  /* Pinned callify.py:101-142 replaces a precompiled value FUNCTION with an
   * opaque CALL whose SINK stores into explicit output PARAMs, then exposes
   * each result as AFTER(output, CALL). */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  poly_ctx_set_preferred_device(ctx, POLY_DEVICE_INTERP);

  float a_data[2] = {1.0f, -5.0f};
  float b_data[2] = {3.0f, 4.0f};
  int64_t shape[3] = {1, 2, 1};
  PolyTensor *a_host = poly_tensor_from_host(ctx, a_data, sizeof(a_data), POLY_FLOAT32, shape, 3);
  PolyTensor *b_host = poly_tensor_from_host(ctx, b_data, sizeof(b_data), POLY_FLOAT32, shape, 3);
  PolyTensor *a = poly_tensor_to_device(ctx, a_host, POLY_DEVICE_INTERP);
  PolyTensor *b = poly_tensor_to_device(ctx, b_host, POLY_DEVICE_INTERP);
  PolyTensor *sum = poly_tensor_alu2(ctx, POLY_OP_ADD, a, b);
  ASSERT_NOT_NULL(sum);

  PolyTensor *results[1] = {sum};
  PolyUOp *logical_inputs[2] = {a->uop_logical, b->uop_logical};
  PolyUOp *physical_inputs[2] = {a->uop_physical, b->uop_physical};
  PolyTensor *output = NULL;
  ASSERT_INT_EQ(
      poly_tensor_function(
          ctx, results, 1, logical_inputs, physical_inputs, 2, "precompiled_add", false, true,
          false, &output
      ),
      0
  );
  ASSERT_NOT_NULL(output);
  ASSERT_INT_EQ(output->uop_physical->op, POLY_OP_GETTUPLE);
  ASSERT_INT_EQ(output->uop_physical->src[0]->op, POLY_OP_FUNCTION);

  PolyUOp *requested = output->uop_physical;
  PolyUOp *realized = NULL;
  PolyUOp *callified = poly_transform_to_call(ctx, &requested, 1, &realized);
  ASSERT_NOT_NULL(callified);
  ASSERT_INT_EQ(count_root_ops(ctx, callified, POLY_OP_FUNCTION), 0);
  ASSERT_INT_EQ(count_root_ops(ctx, callified, POLY_OP_GETTUPLE), 0);
  ASSERT_INT_EQ(count_root_ops(ctx, callified, POLY_OP_TUPLE), 0);

  int n_topo = 0;
  PolyUOp **topo = poly_toposort_alloc(ctx, callified, &n_topo);
  ASSERT_NOT_NULL(topo);
  PolyUOp *precompiled = NULL;
  for (int i = 0; i < n_topo; i++)
    if (topo[i]->op == POLY_OP_CALL && topo[i]->arg.kind == POLY_ARG_CALL_INFO &&
        topo[i]->arg.call_info && topo[i]->arg.call_info->precompile)
      precompiled = topo[i];
  ASSERT_NOT_NULL(precompiled);
  ASSERT_INT_EQ(precompiled->n_src, 4);
  ASSERT_INT_EQ(precompiled->src[0]->op, POLY_OP_SINK);
  ASSERT_INT_EQ(precompiled->src[3]->op, POLY_OP_RESHAPE);
  ASSERT_INT_EQ(precompiled->src[3]->src[0]->op, POLY_OP_PARAM);
  PolyShape output_shape = poly_uop_max_shape_cached(ctx, precompiled->src[3]);
  ASSERT_INT_EQ(output_shape.ndim, 3);
  ASSERT_INT_EQ(output_shape.dims[0], 1);
  ASSERT_INT_EQ(output_shape.dims[1], 2);
  ASSERT_INT_EQ(output_shape.dims[2], 1);
  ASSERT_TRUE(poly_uop_has_buffer_identity(precompiled->src[3]));
  ASSERT_TRUE(poly_uop_has_buffer_identity(realized));
  poly_toposort_free(topo);

  PolyTensor *executed = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &output, 1, &executed), 0);
  ASSERT_PTR_EQ(executed, output);
  float values[2] = {0.0f, 0.0f};
  READ_REALIZED_F32(ctx, poly_tensor_uop_physical(output), values, 2);
  ASSERT_FLOAT_EQ(values[0], 4.0f, 1e-5f);
  ASSERT_FLOAT_EQ(values[1], -1.0f, 1e-5f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, precompiled_multioutput_orders_one_shared_input_copy_first) {
  /* Pinned create_schedule uses an identity-keyed dependency graph and emits
   * the one shared creation COPY before both precompiled output kernels
   * (schedule/__init__.py:21-68). */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  poly_ctx_set_preferred_device(ctx, POLY_DEVICE_INTERP);

  float x_data[2] = {2.0f, 3.0f};
  int64_t shape[1] = {2};
  PolyTensor *x_host = poly_tensor_from_host(ctx, x_data, sizeof(x_data), POLY_FLOAT32, shape, 1);
  PolyTensor *x = poly_tensor_to_device(ctx, x_host, POLY_DEVICE_INTERP);
  PolyTensor *add = poly_tensor_alu2(ctx, POLY_OP_ADD, x, x);
  PolyTensor *mul = poly_tensor_alu2(ctx, POLY_OP_MUL, x, x);
  ASSERT_NOT_NULL(x);
  ASSERT_NOT_NULL(add);
  ASSERT_NOT_NULL(mul);

  PolyTensor *results[2] = {add, mul};
  PolyUOp *logical_inputs[1] = {x->uop_logical};
  PolyUOp *physical_inputs[1] = {x->uop_physical};
  PolyTensor *outputs[2] = {NULL, NULL};
  ASSERT_INT_EQ(
      poly_tensor_function(
          ctx, results, 2, logical_inputs, physical_inputs, 1, "precompiled_pair", false, true,
          false, outputs
      ),
      0
  );
  ASSERT_NOT_NULL(outputs[0]);
  ASSERT_NOT_NULL(outputs[1]);

  PolyUOp *roots[2] = {
      poly_tensor_uop_physical(outputs[0]),
      poly_tensor_uop_physical(outputs[1]),
  };
  PolyUOp *scheduled[2] = {NULL, NULL};
  PolyUOp *schedule = poly_test_linear_values(ctx, roots, 2, scheduled);
  ASSERT_NOT_NULL(schedule);
  ASSERT_NOT_NULL(scheduled[0]);
  ASSERT_NOT_NULL(scheduled[1]);
  ASSERT_INT_EQ(schedule->n_src, 3);
  ASSERT_TRUE(poly_test_linear_call_is_copy(schedule, 0));
  /* Pinned symbolic rewrites x+x to x*2, so both bodies contain MUL; their
   * CONST counts distinguish x+x from x*x exactly as the paired probe does. */
  ASSERT_INT_EQ(count_root_ops(ctx, poly_test_linear_call_body(schedule, 1), POLY_OP_MUL), 1);
  ASSERT_INT_EQ(count_root_ops(ctx, poly_test_linear_call_body(schedule, 1), POLY_OP_CONST), 2);
  ASSERT_INT_EQ(count_root_ops(ctx, poly_test_linear_call_body(schedule, 2), POLY_OP_MUL), 1);
  ASSERT_INT_EQ(count_root_ops(ctx, poly_test_linear_call_body(schedule, 2), POLY_OP_CONST), 1);

  const PolyUOp *copy_dst =
      poly_uop_get_buffer_identity(poly_call_buffer_arg(poly_test_linear_call(schedule, 0), 0));
  const PolyUOp *add_input =
      poly_uop_get_buffer_identity(poly_call_buffer_arg(poly_test_linear_call(schedule, 1), 1));
  const PolyUOp *mul_input =
      poly_uop_get_buffer_identity(poly_call_buffer_arg(poly_test_linear_call(schedule, 2), 1));
  ASSERT_NOT_NULL(copy_dst);
  ASSERT_PTR_EQ(add_input, copy_dst);
  ASSERT_PTR_EQ(mul_input, copy_dst);

  ASSERT_INT_EQ(poly_run_linear(ctx, schedule, NULL, 0, NULL, 0, true, false, false), 0);
  float add_out[2] = {0};
  float mul_out[2] = {0};
  READ_REALIZED_F32(ctx, scheduled[0], add_out, 2);
  READ_REALIZED_F32(ctx, scheduled[1], mul_out, 2);
  ASSERT_FLOAT_EQ(add_out[0], 4.0f, 1e-5f);
  ASSERT_FLOAT_EQ(add_out[1], 6.0f, 1e-5f);
  ASSERT_FLOAT_EQ(mul_out[0], 4.0f, 1e-5f);
  ASSERT_FLOAT_EQ(mul_out[1], 9.0f, 1e-5f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, transform_to_call_materialized_contiguous_keeps_shaped_effect_target) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  float a_data[4] = {2.0f, 1.0f, 0.0f, 2.5f};
  float b_data[1] = {3.5f};
  int64_t a_shape[2] = {2, 2};
  int64_t b_shape[2] = {1, 1};
  int64_t selected_bounds[2][2] = {{1, 2}, {1, 2}};
  int f32 = poly_dtype_id_by_name("float32");
  ASSERT_TRUE(f32 >= 0);
  PolyUOp *a = poly_buffer_from_host(ctx, a_data, sizeof(a_data), f32, a_shape, 2);
  PolyUOp *b = poly_buffer_from_host(ctx, b_data, sizeof(b_data), f32, b_shape, 2);
  PolyUOp *cpu = poly_device_uop(ctx, POLY_DEVICE_CPU);
  PolyUOp *a_cpu = poly_copy_to_device_uop(ctx, a, cpu);
  PolyUOp *b_cpu = poly_copy_to_device_uop(ctx, b, cpu);
  PolyUOp *materialized = poly_contiguous(ctx, poly_add(ctx, a_cpu, poly_const_float(ctx, 0.0f)));
  PolyUOp *selected = poly_shrink(ctx, materialized, selected_bounds, 2);
  PolyUOp *value = poly_div(ctx, b_cpu, selected);
  ASSERT_NOT_NULL(a);
  ASSERT_NOT_NULL(b);
  ASSERT_NOT_NULL(a_cpu);
  ASSERT_NOT_NULL(b_cpu);
  ASSERT_NOT_NULL(materialized);
  ASSERT_NOT_NULL(selected);
  ASSERT_NOT_NULL(value);
  PolyTensor *tensor = physical_tensor_from_uop(ctx, value, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(tensor);
  PolyUOp *physical = poly_tensor_uop_physical(tensor);
  ASSERT_NOT_NULL(physical);

  PolyUOp *callified_out = NULL;
  PolyUOp *call = poly_transform_to_call(ctx, &physical, 1, &callified_out);
  ASSERT_NOT_NULL(call);
  ASSERT_INT_EQ(call->op, POLY_OP_CALL);
  ASSERT_NOT_NULL(call->src[0]);
  ASSERT_INT_EQ(call->src[0]->op, POLY_OP_SINK);

  PolyUOp *materialized_effect = NULL;
  int n_body = 0;
  PolyUOp **body_topo = poly_toposort(ctx, call->src[0], &n_body);
  for (int i = 0; i < n_body; i++) {
    PolyUOp *u = body_topo[i];
    if (!u || u->op != POLY_OP_AFTER || u->n_src != 2 || !u->src[0] ||
        u->src[0]->op != POLY_OP_RESHAPE || !u->src[1] || u->src[1]->op != POLY_OP_STORE ||
        u->src[1]->n_src != 2 || u->src[1]->src[0] != u->src[0])
      continue;
    PolyShape effect_shape = poly_uop_max_shape_cached(ctx, u);
    if (effect_shape.ndim == 2 && effect_shape.dims[0] == 2 && effect_shape.dims[1] == 2) {
      materialized_effect = u;
      break;
    }
  }
  ASSERT_NOT_NULL(materialized_effect);
  ASSERT_PTR_EQ(materialized_effect->src[1]->src[0], materialized_effect->src[0]);
  ASSERT_INT_EQ(materialized_effect->src[0]->n_src, 2);
  ASSERT_INT_EQ(materialized_effect->src[0]->src[1]->op, POLY_OP_STACK);
  ASSERT_INT_EQ(materialized_effect->src[0]->src[1]->n_src, 2);
  ASSERT_TRUE(is_shaped_value_param(materialized_effect->src[0]->src[0]));
  PolyShape storage_shape = poly_uop_max_shape_cached(ctx, materialized_effect->src[0]->src[0]);
  ASSERT_INT_EQ(storage_shape.ndim, 1);
  ASSERT_INT_EQ(storage_shape.dims[0], 4);

  PolyUOp *scheduled_out = NULL;
  PolyUOp *schedule = poly_test_linear_values(ctx, &physical, 1, &scheduled_out);
  ASSERT_NOT_NULL(schedule);
  ASSERT_NOT_NULL(scheduled_out);
  ASSERT_INT_EQ(schedule->n_src, 4);

  PolyUOp *selected_index = NULL;
  for (int i = 0; i < schedule->n_src; i++) {
    PolyUOp *body = poly_test_linear_call_body(schedule, i);
    int n_call = 0;
    PolyUOp **call_topo = poly_toposort(ctx, body, &n_call);
    for (int j = 0; j < n_call; j++) {
      PolyUOp *u = call_topo[j];
      if (!u || u->op != POLY_OP_INDEX || u->n_src != 2 || !u->src[0] || !u->src[1] ||
          u->src[0]->op != POLY_OP_PARAM || u->src[1]->op != POLY_OP_CONST ||
          u->src[1]->arg.kind != POLY_ARG_INT)
        continue;
      PolyShape base_shape = poly_uop_max_shape_cached(ctx, u->src[0]);
      if (base_shape.ndim == 1 && base_shape.dims[0] == 4) selected_index = u;
    }
  }
  ASSERT_NOT_NULL(selected_index);
  ASSERT_INT_EQ(selected_index->src[1]->arg.i, 3);

  ASSERT_INT_EQ(poly_run_linear(ctx, schedule, NULL, 0, NULL, 0, true, false, false), 0);
  float result = 0.0f;
  READ_REALIZED_F32(ctx, scheduled_out, &result, 1);
  ASSERT_FLOAT_EQ(result, 1.4f, 1e-6f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, call_body_paramarg_survives_linear_then_resolves_concrete_args) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  poly_ctx_set_preferred_device(ctx, POLY_DEVICE_INTERP);

  PolyUOp *a = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_INTERP);
  PolyUOp *b = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_INTERP);
  float da[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float db[4] = {10.0f, 20.0f, 30.0f, 40.0f};
  ASSERT_NOT_NULL(a);
  ASSERT_NOT_NULL(b);
  poly_buffer_set(ctx, a, da, sizeof(da), POLY_DEVICE_INTERP);
  poly_buffer_set(ctx, b, db, sizeof(db), POLY_DEVICE_INTERP);

  PolyUOp *targets[2] = {
      poly_alu2(ctx, POLY_OP_ADD, a, b),
      poly_alu2(ctx, POLY_OP_MUL, a, b),
  };
  PolyUOp *callified_out[2] = {NULL, NULL};
  PolyUOp *outer = poly_transform_to_call(ctx, targets, 2, callified_out);
  ASSERT_NOT_NULL(outer);
  ASSERT_INT_EQ(outer->op, POLY_OP_CALL);
  ASSERT_NOT_NULL(outer->src[0]);
  ASSERT_INT_EQ(outer->src[0]->op, POLY_OP_SINK);
  ASSERT_INT_EQ(outer->n_src, 5);

  PolyUOp *params[4] = {NULL, NULL, NULL, NULL};
  int n_body = 0;
  PolyUOp **body_topo = poly_toposort(ctx, outer->src[0], &n_body);
  for (int i = 0; i < n_body; i++) {
    PolyUOp *u = body_topo[i];
    if (!is_shaped_value_param(u)) continue;
    ASSERT_TRUE(u->arg.param->slot >= 0 && u->arg.param->slot < 4);
    ASSERT_TRUE(params[u->arg.param->slot] == NULL);
    params[u->arg.param->slot] = u;
  }
  ASSERT_INT_EQ(count_shaped_value_params(ctx, outer->src[0]), 4);
  for (int slot = 0; slot < 4; slot++) {
    PolyUOp *concrete = outer->src[1 + slot];
    ASSERT_NOT_NULL(params[slot]);
    ASSERT_NOT_NULL(poly_uop_get_buffer_identity(concrete));
    ASSERT_TRUE(poly_dtype_eq(params[slot]->dtype, concrete->dtype));
    ASSERT_STR_EQ(params[slot]->arg.param->device, poly_uop_device_name(ctx, concrete));
    ASSERT_INT_EQ(params[slot]->src[0]->op, POLY_OP_CONST);
    ASSERT_INT_EQ(params[slot]->src[0]->arg.i, 4);
  }

  PolyVarBinding *linear_vars = NULL;
  int n_linear_vars = 0;
  PolyUOp *linear = poly_create_linear_with_vars(ctx, outer, &linear_vars, &n_linear_vars);
  ASSERT_NOT_NULL(linear);
  ASSERT_INT_EQ(n_linear_vars, 0);
  ASSERT_INT_EQ(linear->op, POLY_OP_LINEAR);
  ASSERT_INT_EQ(linear->n_src, 2);
  const int expected_outer_slots[2][3] = {{0, 1, 2}, {3, 1, 2}};
  for (int k = 0; k < linear->n_src; k++) {
    PolyUOp *call = linear->src[k];
    ASSERT_NOT_NULL(call);
    ASSERT_INT_EQ(call->op, POLY_OP_CALL);
    ASSERT_INT_EQ(call->n_src, 4);
    ASSERT_INT_EQ(count_shaped_value_params(ctx, call->src[0]), 3);
    for (int i = 1; i < call->n_src; i++) {
      int slot = expected_outer_slots[k][i - 1];
      ASSERT_PTR_EQ(call->src[i], outer->src[1 + slot]);
      ASSERT_NOT_NULL(poly_uop_get_buffer_identity(call->src[i]));
    }

    PolyUOp *program = call->src[0];
    ASSERT_INT_EQ(count_root_ops(ctx, program, POLY_OP_PARAM), 3);
    ASSERT_INT_EQ(count_root_ops(ctx, program, POLY_OP_INDEX), 3);
    ASSERT_INT_EQ(count_root_ops(ctx, program, POLY_OP_RANGE), 1);
    ASSERT_INT_EQ(count_root_ops(ctx, program, POLY_OP_STORE), 1);
    ASSERT_INT_EQ(count_root_ops(ctx, program, POLY_OP_END), 1);
    ASSERT_INT_EQ(count_root_ops(ctx, program, POLY_OP_STAGE), 0);
  }
  free(linear_vars);

  PolyUOp *resolved_out[2] = {NULL, NULL};
  PolyUOp *schedule = poly_test_linear_values(ctx, targets, 2, resolved_out);
  ASSERT_NOT_NULL(schedule);
  ASSERT_INT_EQ(schedule->n_src, 2);
  ASSERT_INT_EQ(poly_test_linear_call_n_buffers(schedule, 0), 3);
  ASSERT_INT_EQ(poly_test_linear_call_n_buffers(schedule, 1), 3);
  for (int k = 0; k < schedule->n_src; k++) {
    PolyUOp *call = poly_test_linear_call(schedule, k);
    ASSERT_NOT_NULL(call);
    ASSERT_INT_EQ(call->op, POLY_OP_CALL);
    ASSERT_INT_EQ(count_shaped_value_params(ctx, call->src[0]), 3);
    for (int i = 1; i < call->n_src; i++)
      ASSERT_NOT_NULL(poly_uop_get_buffer_identity(call->src[i]));
  }
  ASSERT_INT_EQ(poly_run_linear(ctx, schedule, NULL, 0, NULL, 0, true, false, false), 0);

  float add_out[4] = {0};
  float mul_out[4] = {0};
  READ_REALIZED_F32(ctx, resolved_out[0], add_out, 4);
  READ_REALIZED_F32(ctx, resolved_out[1], mul_out, 4);
  for (int i = 0; i < 4; i++) {
    ASSERT_FLOAT_EQ(add_out[i], da[i] + db[i], 1e-6f);
    ASSERT_FLOAT_EQ(mul_out[i], da[i] * db[i], 1e-6f);
  }

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, transform_to_call_keeps_after_version_in_executable_value_graph) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *state = poly_buffer_f32(ctx, 1);
  PolyUOp *scale = poly_buffer_f32(ctx, 1);
  PolyUOp *state_store = poly_store_val(ctx, state, poly_const_float(ctx, 2.0f));
  PolyUOp *state_after_src[2] = {state, state_store};
  PolyUOp *state_after =
      poly_uop(ctx, POLY_OP_AFTER, state->dtype, state_after_src, 2, poly_arg_none());
  PolyUOp *consumer = poly_alu2(ctx, POLY_OP_MUL, scale, state_after);
  ASSERT_NOT_NULL(state_store);
  ASSERT_NOT_NULL(state_after);
  ASSERT_NOT_NULL(consumer);

  PolyUOp *targets[2] = {state_after, consumer};
  PolyUOp *realized[2] = {NULL, NULL};
  PolyUOp *big_call = poly_transform_to_call(ctx, targets, 2, realized);
  ASSERT_NOT_NULL(big_call);
  ASSERT_INT_EQ(big_call->op, POLY_OP_CALL);
  ASSERT_NOT_NULL(big_call->src[0]);
  ASSERT_INT_EQ(big_call->src[0]->op, POLY_OP_SINK);
  ASSERT_INT_EQ(big_call->src[0]->n_src, 2);

  PolyUOp *consumer_after = big_call->src[0]->src[1];
  ASSERT_NOT_NULL(consumer_after);
  ASSERT_INT_EQ(consumer_after->op, POLY_OP_AFTER);
  ASSERT_INT_EQ(consumer_after->n_src, 2);
  PolyUOp *consumer_store = consumer_after->src[1];
  ASSERT_NOT_NULL(consumer_store);
  ASSERT_INT_EQ(consumer_store->op, POLY_OP_STORE);
  ASSERT_PTR_EQ(consumer_store->src[0], consumer_after->src[0]);
  ASSERT_INT_EQ(count_root_ops(ctx, consumer_store->src[1], POLY_OP_AFTER), 1);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, transform_to_call_preserves_virtual_roots_without_storage) {
  /* Pinned tensor.py:transform_to_call excludes device-free and weak values
   * from AllocCtx.bases. They have no width/place to materialize. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *buffer = poly_buffer_f32(ctx, 4);
  PolyUOp *one = poly_uop_const(ctx, poly_arg_float(1.0), POLY_FLOAT32);
  PolyUOp *roots[] = {
      poly_alu2(ctx, POLY_OP_ADD, one, one),
      poly_cast(ctx, buffer, POLY_WEAKFLOAT),
  };
  for (int i = 0; i < 2; i++) {
    PolyUOp *out = NULL, **map_orig = NULL, **map_repl = NULL;
    int map_n = 0;
    PolyUOp *call =
        poly_transform_to_call_with_map(ctx, &roots[i], 1, &out, &map_orig, &map_repl, &map_n);
    ASSERT_NOT_NULL(call);
    ASSERT_INT_EQ(call->op, POLY_OP_CALL);
    ASSERT_INT_EQ(call->n_src, 1);
    ASSERT_INT_EQ(call->src[0]->op, POLY_OP_SINK);
    ASSERT_INT_EQ(call->src[0]->n_src, 0);
    ASSERT_PTR_EQ(out, roots[i]);
    ASSERT_INT_EQ(map_n, 0);
    free(map_orig);
    free(map_repl);
  }
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, transform_to_call_preserves_alu_roots_without_storage) {
  /* A deviceful LOAD and its ALU consumers are kernel values, not requested
   * Tensor storage. The pinned predicate excludes their ALU address space. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *buffer = poly_buffer_f32(ctx, 4);
  PolyUOp *zero = poly_const_int(ctx, 0);
  PolyUOp *index = poly_uop_index(ctx, buffer, &zero, 1);
  PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, index, poly_arg_none());
  PolyUOp *one = poly_uop_const(ctx, poly_arg_float(1.0), POLY_FLOAT32);
  PolyUOp *roots[] = {load, poly_alu2(ctx, POLY_OP_ADD, load, one)};
  for (int i = 0; i < 2; i++) {
    PolyAddrSpace addrspace;
    ASSERT_TRUE(poly_uop_addrspace(roots[i], &addrspace));
    ASSERT_INT_EQ(addrspace, POLY_ADDR_ALU);
    PolyUOp *out = NULL, **map_orig = NULL, **map_repl = NULL;
    int map_n = 0;
    PolyUOp *call =
        poly_transform_to_call_with_map(ctx, &roots[i], 1, &out, &map_orig, &map_repl, &map_n);
    ASSERT_NOT_NULL(call);
    ASSERT_INT_EQ(call->op, POLY_OP_CALL);
    ASSERT_INT_EQ(call->n_src, 1);
    ASSERT_INT_EQ(call->src[0]->op, POLY_OP_SINK);
    ASSERT_INT_EQ(call->src[0]->n_src, 0);
    ASSERT_PTR_EQ(out, roots[i]);
    ASSERT_INT_EQ(map_n, 0);
    free(map_orig);
    free(map_repl);
  }
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, transform_to_call_requested_unshard_preserves_capture_boundary) {
  /* UOp.base stops at UNSHARD. A requested movement/DETACH over it must
   * materialize that global value, not publish its unrequested local producer. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *buffer = poly_buffer_f32(ctx, 4);
  float values[] = {1, 2, 3, 4};
  poly_buffer_set(ctx, buffer, values, sizeof(values), POLY_DEVICE_CPU);
  PolyUOp *one = poly_uop_const(ctx, poly_arg_float(1.0), POLY_FLOAT32);
  PolyUOp *parent = poly_alu2(ctx, POLY_OP_ADD, buffer, one);
  PolyUOp *range = poly_uop_range(ctx, 1, -1, POLY_AXIS_DEVICE);
  int64_t axis[] = {0};
  PolyUOp *unshard = poly_unshard(ctx, parent, axis, &range, 1);
  PolyUOp *roots[] = {
      poly_reshape(ctx, unshard, (int64_t[]){2, 2}, 2),
      poly_alu1(ctx, POLY_OP_DETACH, unshard),
  };
  for (int i = 0; i < 2; i++)
    ASSERT_INT_EQ(poly_uop_retain(ctx, roots[i]), 0);
  for (int i = 0; i < 2; i++) {
    PolyUOp *out = NULL, **map_orig = NULL, **map_repl = NULL;
    int map_n = 0;
    PolyUOp *call =
        poly_transform_to_call_with_map(ctx, &roots[i], 1, &out, &map_orig, &map_repl, &map_n);
    ASSERT_NOT_NULL(call);
    bool mapped_unshard = false;
    for (int j = 0; j < map_n; j++) {
      ASSERT_PTR_NEQ(map_orig[j], parent);
      if (map_orig[j] == unshard) mapped_unshard = true;
    }
    ASSERT_TRUE(mapped_unshard);
    ASSERT_INT_EQ(call->op, POLY_OP_CALL);
    ASSERT_INT_EQ(call->src[0]->op, POLY_OP_SINK);
    ASSERT_INT_EQ(call->src[0]->n_src, 1);
    PolyUOp *effect = call->src[0]->src[0];
    ASSERT_INT_EQ(effect->op, POLY_OP_AFTER);
    ASSERT_INT_EQ(effect->src[1]->op, POLY_OP_STORE);
    ASSERT_INT_EQ(effect->src[1]->src[1]->op, POLY_OP_UNSHARD);
    ASSERT_INT_EQ(effect->src[1]->src[1]->src[0]->op, POLY_OP_ADD);
    ASSERT_INT_EQ(count_root_ops(ctx, call, POLY_OP_STORE), 1);
    ASSERT_NOT_NULL(poly_uop_get_buffer_identity(out));
    free(map_orig);
    free(map_repl);

    PolyUOp *resolved = NULL;
    PolyUOp *linear = poly_test_linear_values(ctx, &roots[i], 1, &resolved);
    ASSERT_NOT_NULL(linear);
    ASSERT_INT_EQ(poly_uop_retain(ctx, resolved), 0);
    ASSERT_INT_EQ(poly_run_linear(ctx, linear, NULL, 0, NULL, 0, true, false, false), 0);
    float got[4] = {0};
    READ_REALIZED_F32(ctx, resolved, got, 4);
    for (int j = 0; j < 4; j++)
      ASSERT_FLOAT_EQ(got[j], values[j] + 1, 1e-6);
    poly_uop_release(ctx, resolved);
  }
  for (int i = 0; i < 2; i++)
    poly_uop_release(ctx, roots[i]);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, transform_to_call_removes_training_markers_before_storage) {
  /* Pinned pm_early_transform_tensor_graph removes both markers. A tagged
   * CONTIGUOUS_BACKWARD must still publish its requested output provenance. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *buffer = poly_buffer_f32(ctx, 4);
  PolyUOp *one = poly_uop_const(ctx, poly_arg_float(1.0), POLY_FLOAT32);
  PolyUOp *value = poly_alu2(ctx, POLY_OP_ADD, buffer, one);
  PolyOps ops[] = {POLY_OP_DETACH, POLY_OP_CONTIGUOUS_BACKWARD};
  for (int i = 0; i < 2; i++) {
    PolyUOp *root = poly_alu1(ctx, ops[i], value);
    PolyUOp *out = NULL;
    PolyUOp *call = poly_transform_to_call(ctx, &root, 1, &out);
    ASSERT_NOT_NULL(call);
    ASSERT_INT_EQ(count_root_ops(ctx, call, POLY_OP_DETACH), 0);
    ASSERT_INT_EQ(count_root_ops(ctx, call, POLY_OP_CONTIGUOUS_BACKWARD), 0);
    ASSERT_INT_EQ(call->src[0]->op, POLY_OP_SINK);
    ASSERT_INT_EQ(call->src[0]->n_src, 1);
    PolyUOp *effect = call->src[0]->src[0];
    ASSERT_INT_EQ(effect->op, POLY_OP_AFTER);
    ASSERT_INT_EQ(effect->src[1]->op, POLY_OP_STORE);
    ASSERT_INT_EQ(effect->src[1]->src[1]->op, POLY_OP_ADD);
    ASSERT_NOT_NULL(poly_uop_get_buffer_identity(out));
  }
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, transform_to_call_virtual_contiguous_has_no_publication) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *buffer = poly_buffer_f32(ctx, 4);
  PolyUOp *root = poly_contiguous(ctx, poly_cast(ctx, buffer, POLY_WEAKFLOAT));
  PolyUOp *out = NULL, **map_orig = NULL, **map_repl = NULL;
  int map_n = 0;
  PolyUOp *call =
      poly_transform_to_call_with_map(ctx, &root, 1, &out, &map_orig, &map_repl, &map_n);
  ASSERT_NOT_NULL(call);
  ASSERT_INT_EQ(call->n_src, 1);
  ASSERT_INT_EQ(call->src[0]->n_src, 0);
  ASSERT_PTR_EQ(out, root);
  ASSERT_INT_EQ(map_n, 0);
  free(map_orig);
  free(map_repl);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, transform_to_call_training_marker_reuses_requested_parent) {
  /* CONTIGUOUS_BACKWARD's tag must merge into the parent's AFTER when both
   * are requested, not allocate a second buffer by bypassing early rewrites. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *buffer = poly_buffer_f32(ctx, 4);
  PolyUOp *one = poly_uop_const(ctx, poly_arg_float(1.0), POLY_FLOAT32);
  PolyUOp *parent = poly_alu2(ctx, POLY_OP_ADD, buffer, one);
  PolyUOp *marker = poly_alu1(ctx, POLY_OP_CONTIGUOUS_BACKWARD, parent);
  for (int reverse = 0; reverse < 2; reverse++) {
    PolyUOp *roots[] = {reverse ? marker : parent, reverse ? parent : marker};
    PolyUOp *out[2] = {NULL, NULL};
    PolyUOp *call = poly_transform_to_call(ctx, roots, 2, out);
    ASSERT_NOT_NULL(call);
    ASSERT_INT_EQ(call->src[0]->op, POLY_OP_SINK);
    ASSERT_INT_EQ(call->src[0]->n_src, 1);
    ASSERT_INT_EQ(count_root_ops(ctx, call, POLY_OP_STORE), 1);
    ASSERT_INT_EQ(count_root_ops(ctx, call, POLY_OP_CONTIGUOUS_BACKWARD), 0);
    ASSERT_NOT_NULL(poly_uop_get_buffer_identity(out[0]));
    ASSERT_PTR_EQ(out[0], out[1]);
  }
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, transform_to_call_requested_parent_feeds_requested_descendant) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *input = poly_buffer_f32(ctx, 4);
  PolyUOp *bias = poly_buffer_f32(ctx, 4);
  PolyUOp *scale = poly_buffer_f32(ctx, 4);
  float input_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
  float bias_data[] = {10.0f, 20.0f, 30.0f, 40.0f};
  float scale_data[] = {2.0f, 3.0f, 4.0f, 5.0f};
  poly_buffer_set(ctx, input, input_data, sizeof(input_data), POLY_DEVICE_CPU);
  poly_buffer_set(ctx, bias, bias_data, sizeof(bias_data), POLY_DEVICE_CPU);
  poly_buffer_set(ctx, scale, scale_data, sizeof(scale_data), POLY_DEVICE_CPU);

  PolyUOp *parent = poly_alu2(ctx, POLY_OP_ADD, input, bias);
  PolyUOp *descendant = poly_alu2(ctx, POLY_OP_MUL, parent, scale);
  PolyUOp *targets[] = {parent, descendant};
  PolyUOp *callified_out[] = {NULL, NULL};
  PolyUOp *big_call = poly_transform_to_call(ctx, targets, 2, callified_out);
  ASSERT_NOT_NULL(big_call);
  ASSERT_INT_EQ(big_call->op, POLY_OP_CALL);
  ASSERT_NOT_NULL(big_call->src[0]);
  ASSERT_INT_EQ(big_call->src[0]->op, POLY_OP_SINK);
  ASSERT_INT_EQ(big_call->src[0]->n_src, 2);

  PolyUOp *parent_effect = big_call->src[0]->src[0];
  PolyUOp *descendant_effect = big_call->src[0]->src[1];
  ASSERT_INT_EQ(parent_effect->op, POLY_OP_AFTER);
  ASSERT_INT_EQ(descendant_effect->op, POLY_OP_AFTER);
  ASSERT_INT_EQ(parent_effect->src[1]->op, POLY_OP_STORE);
  ASSERT_INT_EQ(descendant_effect->src[1]->op, POLY_OP_STORE);
  ASSERT_INT_EQ(count_root_ops(ctx, descendant_effect->src[1]->src[1], POLY_OP_AFTER), 1);
  ASSERT_INT_EQ(count_root_ops(ctx, descendant_effect->src[1]->src[1], POLY_OP_STORE), 1);

  PolyUOp *resolved[] = {NULL, NULL};
  PolyUOp *schedule = poly_test_linear_values(ctx, targets, 2, resolved);
  ASSERT_NOT_NULL(schedule);
  ASSERT_INT_EQ(schedule->n_src, 2);
  /* Tinygrad Tensor.realize keeps every output Tensor/UOp live through all reads;
   * raw C callers must retain borrowed output roots across allocation safe points. */
  ASSERT_INT_EQ(poly_uop_retain(ctx, resolved[0]), 0);
  ASSERT_INT_EQ(poly_uop_retain(ctx, resolved[1]), 0);
  const PolyUOp *parent_buffer = poly_uop_get_buffer_identity(resolved[0]);
  ASSERT_NOT_NULL(parent_buffer);
  for (int call_index = 0; call_index < 2; call_index++) {
    PolyUOp *call = poly_test_linear_call(schedule, call_index);
    bool has_parent_buffer = false;
    ASSERT_NOT_NULL(call);
    for (int i = 1; i < call->n_src; i++)
      if (poly_uop_get_buffer_identity(call->src[i]) == parent_buffer) has_parent_buffer = true;
    ASSERT_TRUE(has_parent_buffer);
  }
  ASSERT_INT_EQ(poly_run_linear(ctx, schedule, NULL, 0, NULL, 0, true, false, false), 0);

  float parent_out[4] = {0};
  float descendant_out[4] = {0};
  READ_REALIZED_F32(ctx, resolved[0], parent_out, 4);
  READ_REALIZED_F32(ctx, resolved[1], descendant_out, 4);
  for (int i = 0; i < 4; i++) {
    ASSERT_FLOAT_EQ(parent_out[i], input_data[i] + bias_data[i], 1e-6f);
    ASSERT_FLOAT_EQ(descendant_out[i], parent_out[i] * scale_data[i], 1e-6f);
  }

  poly_uop_release(ctx, resolved[1]);
  poly_uop_release(ctx, resolved[0]);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, transform_to_call_requested_parent_movement_views_reuse_parent_effect) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *input = poly_reshape(ctx, poly_buffer_f32(ctx, 6), (int64_t[]){2, 3}, 2);
  PolyUOp *bias = poly_reshape(ctx, poly_buffer_f32(ctx, 6), (int64_t[]){2, 3}, 2);
  PolyUOp *parent = poly_alu2(ctx, POLY_OP_ADD, input, bias);
  ASSERT_NOT_NULL(input);
  ASSERT_NOT_NULL(bias);
  ASSERT_NOT_NULL(parent);

  PolyUOp *views[] = {
      poly_permute(ctx, parent, (int64_t[]){1, 0}, 2),
      poly_shrink(ctx, parent, (int64_t[][2]){{0, 2}, {1, 3}}, 2),
      poly_flip(ctx, parent, (int64_t[]){1}, 1),
  };
  PolyOps expected_ops[] = {POLY_OP_PERMUTE, POLY_OP_SHRINK, POLY_OP_FLIP};
  int expected_n_src[] = {1, 3, 1};

  for (int i = 0; i < 3; i++) {
    PolyUOp *targets[] = {parent, views[i]};
    PolyUOp *callified_out[] = {NULL, NULL};
    PolyUOp *callified = poly_transform_to_call(ctx, targets, 2, callified_out);
    ASSERT_NOT_NULL(callified);
    ASSERT_EQ(callified->op, POLY_OP_CALL);
    ASSERT_NOT_NULL(callified->src[0]);
    ASSERT_EQ(callified->src[0]->op, POLY_OP_SINK);
    ASSERT_INT_EQ(callified->src[0]->n_src, 1);
    ASSERT_EQ(callified->src[0]->src[0]->op, POLY_OP_AFTER);
    ASSERT_EQ(callified->src[0]->src[0]->src[1]->op, POLY_OP_STORE);
    ASSERT_NOT_NULL(callified_out[0]);
    ASSERT_NOT_NULL(callified_out[1]);
    ASSERT_EQ(callified_out[1]->op, expected_ops[i]);
    ASSERT_INT_EQ(callified_out[1]->n_src, expected_n_src[i]);
    ASSERT_PTR_EQ(callified_out[1]->src[0], callified_out[0]);
    if (expected_ops[i] == POLY_OP_SHRINK) {
      ASSERT_EQ(callified_out[1]->src[1]->op, POLY_OP_STACK);
      ASSERT_EQ(callified_out[1]->src[2]->op, POLY_OP_STACK);
      ASSERT_INT_EQ(callified_out[1]->src[1]->n_src, 2);
      ASSERT_INT_EQ(callified_out[1]->src[2]->n_src, 2);
    }

    PolyUOp *scheduled_out[] = {NULL, NULL};
    PolyUOp *schedule = poly_test_linear_values(ctx, targets, 2, scheduled_out);
    ASSERT_NOT_NULL(schedule);
    ASSERT_INT_EQ(schedule->n_src, 1);
    ASSERT_NOT_NULL(scheduled_out[0]);
    ASSERT_NOT_NULL(scheduled_out[1]);
    ASSERT_EQ(scheduled_out[1]->op, expected_ops[i]);
    ASSERT_INT_EQ(scheduled_out[1]->n_src, expected_n_src[i]);
    ASSERT_PTR_EQ(scheduled_out[1]->src[0], scheduled_out[0]);
    if (expected_ops[i] == POLY_OP_SHRINK) {
      ASSERT_EQ(scheduled_out[1]->src[1]->op, POLY_OP_STACK);
      ASSERT_EQ(scheduled_out[1]->src[2]->op, POLY_OP_STACK);
    }
  }

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, transform_to_call_shared_dag_stays_linear) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *value = poly_buffer_f32(ctx, 4);
  for (int i = 0; i < 26; i++)
    value = poly_alu2(ctx, POLY_OP_ADD, value, value);
  ASSERT_NOT_NULL(value);

  struct timespec cpu_begin, cpu_end;
  ASSERT_INT_EQ(clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cpu_begin), 0);
  PolyUOp *realized = NULL;
  PolyUOp *big_call = poly_transform_to_call(ctx, &value, 1, &realized);
  ASSERT_INT_EQ(clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cpu_end), 0);
  double cpu_seconds = (double)(cpu_end.tv_sec - cpu_begin.tv_sec) +
                       (double)(cpu_end.tv_nsec - cpu_begin.tv_nsec) * 1e-9;

  ASSERT_NOT_NULL(big_call);
  ASSERT_INT_EQ(big_call->op, POLY_OP_CALL);
  ASSERT_NOT_NULL(realized);
  ASSERT_NOT_NULL(poly_uop_get_buffer_identity(realized));
  if (cpu_seconds >= 0.5) FAIL("shared-DAG callify used %.6f CPU seconds", cpu_seconds);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, transform_to_call_deep_view_stack_retargets_all_views) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer_f32(ctx, 1);
  PolyUOp *u = poly_alu2(ctx, POLY_OP_ADD, a, poly_const_float(ctx, 1.0f));
  for (int i = 0; i < 20; i++) {
    int64_t pads[1][2] = {{1, 1}};
    u = poly_pad(ctx, u, pads, 1);
  }

  PolyUOp *targets[] = {u};
  PolyUOp *realized[] = {NULL};
  PolyUOp *big_call = poly_transform_to_call(ctx, targets, 1, realized);
  ASSERT_NOT_NULL(big_call);
  ASSERT_INT_EQ(big_call->op, POLY_OP_CALL);
  ASSERT_NOT_NULL(realized[0]);
  ASSERT_INT_EQ(count_root_ops(ctx, realized[0], POLY_OP_PAD), 20);

  PolyShape shape = poly_uop_max_shape_cached(ctx, realized[0]);
  ASSERT_INT_EQ(shape.ndim, 1);
  ASSERT_INT_EQ(shape.dims[0], 41);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, transform_to_call_preserves_symbolic_reshape_shape_source) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  /* Pinned UOp._shape requires RESHAPE source and target products to match
   * (uop/ops.py:330-333). Build a valid symbolic (n,2) source before checking
   * that callify preserves the exact target shape-value graph. */
  PolyUOp *source = poly_buffer_f32(ctx, 2);
  source = poly_reshape(ctx, source, (int64_t[]){1, 2}, 2);
  PolyUOp *n = poly_uop_variable(ctx, "n", 1, 8, POLY_WEAKINT, 1, false);
  PolyUOp *two = poly_const_int(ctx, 2);
  PolyUOp *shape_srcs[] = {n, two};
  PolyUOp *shape = poly_uop(ctx, POLY_OP_STACK, POLY_WEAKINT, shape_srcs, 2, poly_arg_none());
  PolyUOp *expanded = poly_expand_uop(ctx, source, shape_srcs, 2);
  ASSERT_NOT_NULL(expanded);
  PolyUOp *value = poly_alu2(ctx, POLY_OP_ADD, expanded, poly_const_float(ctx, 1.0f));
  PolyUOp *reshape_srcs[] = {value, shape};
  PolyUOp *root = poly_uop(ctx, POLY_OP_RESHAPE, POLY_FLOAT32, reshape_srcs, 2, poly_arg_none());
  ASSERT_NOT_NULL(root);
  PolyShape root_shape = poly_uop_max_shape_cached(ctx, root);
  ASSERT_INT_EQ(root_shape.ndim, 2);
  ASSERT_INT_EQ(root_shape.dims[0], 8);
  ASSERT_INT_EQ(root_shape.dims[1], 2);
  ASSERT_PTR_EQ(poly_uop_shape_dim(ctx, root, 0), n);

  PolyUOp *realized = NULL;
  PolyUOp *big_call = poly_transform_to_call(ctx, &root, 1, &realized);
  ASSERT_NOT_NULL(big_call);
  ASSERT_INT_EQ(big_call->op, POLY_OP_CALL);
  ASSERT_NOT_NULL(realized);
  ASSERT_INT_EQ(realized->op, POLY_OP_RESHAPE);
  ASSERT_INT_EQ(realized->n_src, 2);
  ASSERT_TRUE(realized->src[0] != value);
  ASSERT_PTR_EQ(realized->src[1], shape);
  ASSERT_PTR_EQ(realized->src[1]->src[0], n);
  ASSERT_PTR_EQ(root->src[1], shape);
  ASSERT_INT_EQ(realized->src[0]->op, POLY_OP_SHRINK);
  ASSERT_INT_EQ(realized->src[0]->n_src, 3);
  ASSERT_INT_EQ(realized->src[0]->src[2]->op, POLY_OP_STACK);
  ASSERT_PTR_EQ(realized->src[0]->src[2]->src[0], n);
  PolyShape realized_shape = poly_uop_max_shape_cached(ctx, realized);
  ASSERT_INT_EQ(realized_shape.ndim, 2);
  ASSERT_INT_EQ(realized_shape.dims[0], 8);
  ASSERT_INT_EQ(realized_shape.dims[1], 2);
  ASSERT_PTR_EQ(poly_uop_shape_dim(ctx, realized, 0), n);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, symbolic_reshape_callify_runs_nonmax_binding) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  /* Pinned callify.py:180 materializes the maximum buffer, then shrink_to's
   * the exact symbolic root before replaying outer views. */
  PolyUOp *base = poly_buffer_f32(ctx, 2);
  PolyUOp *source = poly_reshape(ctx, base, (int64_t[]){1, 2}, 2);
  PolyUOp *n = poly_uop_variable(ctx, "n_nonmax", 1, 8, POLY_WEAKINT, 1, false);
  PolyUOp *shape_srcs[] = {n, poly_const_int(ctx, 2)};
  PolyUOp *shape = poly_uop(ctx, POLY_OP_STACK, POLY_WEAKINT, shape_srcs, 2, poly_arg_none());
  PolyUOp *expanded = poly_expand_uop(ctx, source, shape_srcs, 2);
  PolyUOp *value = poly_alu2(ctx, POLY_OP_ADD, expanded, poly_const_float(ctx, 1.0f));
  PolyUOp *reshape_srcs[] = {value, shape};
  PolyUOp *root = poly_uop(ctx, POLY_OP_RESHAPE, POLY_FLOAT32, reshape_srcs, 2, poly_arg_none());
  ASSERT_NOT_NULL(root);
  ASSERT_INT_EQ(poly_uop_max_shape_cached(ctx, root).ndim, 2);

  float input[2] = {1.0f, 2.0f};
  poly_buffer_set(ctx, base, input, sizeof(input), POLY_DEVICE_CPU);
  PolyUOp *scheduled_out = NULL;
  PolyUOp *schedule = poly_test_linear_values(ctx, &root, 1, &scheduled_out);
  ASSERT_NOT_NULL(schedule);
  ASSERT_NOT_NULL(scheduled_out);
  ASSERT_INT_EQ(poly_uop_max_shape_cached(ctx, scheduled_out).ndim, 2);
  ASSERT_INT_EQ(count_root_ops(ctx, scheduled_out, POLY_OP_SHRINK), 1);

  PolyVarBinding bind = {.var = n, .value = 4};
  ASSERT_INT_EQ(poly_run_linear(ctx, schedule, &bind, 1, NULL, 0, true, false, false), 0);
  /* Pinned UOp.has_buffer_identity (uop/ops.py:825-829) does not unwrap
   * SHRINK. Read the backing max-allocation buffer below the exact symbolic
   * view instead of broadening Polygrad's buffer-identity contract. */
  ASSERT_INT_EQ(scheduled_out->op, POLY_OP_RESHAPE);
  ASSERT_INT_EQ(scheduled_out->src[0]->op, POLY_OP_SHRINK);
  const PolyUOp *identity = poly_uop_get_buffer_identity(scheduled_out->src[0]->src[0]);
  ASSERT_NOT_NULL(identity);
  float output[8] = {0};
  ASSERT_INT_EQ(poly_buffer_read(ctx, (PolyUOp *)identity, output, sizeof(output)), 0);
  for (int i = 0; i < 4; i++) {
    ASSERT_FLOAT_EQ(output[2 * i], 2.0f, 1e-5f);
    ASSERT_FLOAT_EQ(output[2 * i + 1], 3.0f, 1e-5f);
  }

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, transform_to_call_preserves_scalar_movement_shape_sources) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *reshape_value =
      poly_alu2(ctx, POLY_OP_ADD, poly_buffer_f32(ctx, 8), poly_const_float(ctx, 1.0f));
  PolyUOp *n = poly_uop_variable(ctx, "n", 1, 8, POLY_WEAKINT, 1, false);
  PolyUOp *reshape_src[] = {reshape_value, n};
  PolyUOp *reshape = poly_uop(ctx, POLY_OP_RESHAPE, POLY_FLOAT32, reshape_src, 2, poly_arg_none());
  ASSERT_NOT_NULL(reshape);

  PolyUOp *realized_reshape = NULL;
  PolyUOp *reshape_call = poly_transform_to_call(ctx, &reshape, 1, &realized_reshape);
  ASSERT_NOT_NULL(reshape_call);
  ASSERT_NOT_NULL(realized_reshape);
  ASSERT_INT_EQ(realized_reshape->op, POLY_OP_RESHAPE);
  ASSERT_INT_EQ(realized_reshape->n_src, 2);
  ASSERT_PTR_EQ(realized_reshape->src[1], n);

  PolyUOp *unit = poly_reshape(ctx, poly_buffer_f32(ctx, 1), (int64_t[]){1}, 1);
  PolyUOp *expand_value = poly_alu2(ctx, POLY_OP_ADD, unit, poly_const_float(ctx, 1.0f));
  PolyUOp *expand_src[] = {expand_value, n};
  PolyUOp *expand = poly_uop(ctx, POLY_OP_EXPAND, POLY_FLOAT32, expand_src, 2, poly_arg_none());
  ASSERT_NOT_NULL(expand);

  PolyUOp *realized_expand = NULL;
  PolyUOp *expand_call = poly_transform_to_call(ctx, &expand, 1, &realized_expand);
  ASSERT_NOT_NULL(expand_call);
  ASSERT_NOT_NULL(realized_expand);
  ASSERT_INT_EQ(realized_expand->op, POLY_OP_EXPAND);
  ASSERT_INT_EQ(realized_expand->n_src, 2);
  ASSERT_PTR_EQ(realized_expand->src[1], n);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, transform_to_call_reduce_views_stay_in_output_graph) {
  PolyCtx *ctx = poly_ctx_new();

  enum { N_TERMS = 70 };
  PolyUOp *sum = NULL;
  for (int i = 0; i < N_TERMS; i++) {
    PolyUOp *buf = poly_buffer_f32(ctx, 4);
    int64_t axis[] = {0};
    PolyUOp *reduced = poly_reduce_axis(ctx, POLY_OP_ADD, buf, axis, 1);
    PolyUOp *view = poly_reshape(ctx, reduced, (int64_t[]){1}, 1);
    sum = sum ? poly_alu2(ctx, POLY_OP_ADD, sum, view) : view;
  }

  PolyUOp *targets[] = {sum};
  PolyUOp *realized[] = {NULL};
  PolyUOp *big_call = poly_transform_to_call(ctx, targets, 1, realized);
  ASSERT_NOT_NULL(big_call);
  ASSERT_INT_EQ(big_call->op, POLY_OP_CALL);
  ASSERT_NOT_NULL(big_call->src[0]);
  ASSERT_INT_EQ(big_call->src[0]->op, POLY_OP_SINK);
  ASSERT_INT_EQ(big_call->src[0]->n_src, 1);
  ASSERT_INT_EQ(count_root_ops(ctx, big_call->src[0], POLY_OP_AFTER), 1);
  ASSERT_INT_EQ(count_root_ops(ctx, big_call->src[0], POLY_OP_STORE), 1);
  ASSERT_INT_EQ(count_root_ops(ctx, big_call->src[0], POLY_OP_REDUCE), N_TERMS);

  PolyUOp *output_effect = big_call->src[0]->src[0];
  ASSERT_NOT_NULL(output_effect);
  ASSERT_INT_EQ(output_effect->op, POLY_OP_AFTER);
  ASSERT_INT_EQ(output_effect->n_src, 2);
  ASSERT_INT_EQ(output_effect->src[1]->op, POLY_OP_STORE);
  ASSERT_INT_EQ(count_root_ops(ctx, output_effect->src[1]->src[1], POLY_OP_REDUCE), N_TERMS);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, transform_to_call_shared_reduce_matches_tinygrad_topology) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer_f32(ctx, 4);
  PolyUOp *c = poly_buffer_f32(ctx, 4);
  PolyUOp *e = poly_buffer_f32(ctx, 4);
  int64_t axis[] = {0};
  PolyUOp *reduced = poly_reduce_axis(ctx, POLY_OP_ADD, a, axis, 1);
  PolyUOp *targets[] = {
      poly_add(ctx, reduced, c),
      poly_mul(ctx, reduced, e),
  };

  size_t scratch_before = poly_arena_used(ctx->scratch);
  PolyUOp *realized[] = {NULL, NULL};
  PolyUOp *big_call = poly_transform_to_call(ctx, targets, 2, realized);
  ASSERT_NOT_NULL(big_call);
  ASSERT_INT_EQ(big_call->op, POLY_OP_CALL);
  ASSERT_NOT_NULL(big_call->src[0]);
  ASSERT_INT_EQ(big_call->src[0]->op, POLY_OP_SINK);
  ASSERT_INT_EQ(big_call->src[0]->n_src, 2);
  ASSERT_NOT_NULL(realized[0]);
  ASSERT_NOT_NULL(realized[1]);
  ASSERT_INT_EQ(count_root_ops(ctx, big_call->src[0], POLY_OP_AFTER), 2);
  ASSERT_INT_EQ(count_root_ops(ctx, big_call->src[0], POLY_OP_STORE), 2);
  ASSERT_INT_EQ(count_root_ops(ctx, big_call->src[0], POLY_OP_REDUCE), 1);
  ASSERT_INT_EQ(count_root_ops(ctx, big_call->src[0], POLY_OP_RESHAPE), 0);
  ASSERT_INT_EQ(count_root_ops(ctx, big_call->src[0], POLY_OP_EXPAND), 0);
  ASSERT_INT_EQ(big_call->src[0]->src[0]->op, POLY_OP_AFTER);
  ASSERT_INT_EQ(big_call->src[0]->src[1]->op, POLY_OP_AFTER);
  ASSERT_INT_EQ(poly_arena_used(ctx->scratch), scratch_before);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, linear_with_vars_then_run_vecadd) {
  PolyCtx *ctx = poly_ctx_new();

  /* Current Tinygrad Tensor.linear_with_vars -> run_linear boundary. */
  PolyUOp *a = poly_buffer_f32(ctx, 4);
  PolyUOp *b = poly_buffer_f32(ctx, 4);
  PolyUOp *add = poly_alu2(ctx, POLY_OP_ADD, a, b);

  float da[] = {1.0f, 2.0f, 3.0f, 4.0f};
  float db[] = {10.0f, 20.0f, 30.0f, 40.0f};
  poly_buffer_set(ctx, a, da, sizeof(da), POLY_DEVICE_CPU);
  poly_buffer_set(ctx, b, db, sizeof(db), POLY_DEVICE_CPU);

  PolyUOp *targets[] = {add};
  PolyUOp *realized[] = {NULL};
  PolyUOp *sched = poly_test_linear_values(ctx, targets, 1, realized);
  ASSERT_NOT_NULL(sched);
  ASSERT_TRUE(realized[0] != NULL);
  ASSERT_INT_EQ(sched->n_src, 1);

  ASSERT_INT_EQ(poly_run_linear(ctx, sched, NULL, 0, NULL, 0, true, false, false), 0);

  const PolyUOp *out = poly_uop_get_buffer_identity(realized[0]);
  ASSERT_TRUE(out != NULL);
  PolyBuffer *buf = poly_buffer_get(ctx, (PolyUOp *)out);
  ASSERT_NOT_NULL(buf);
  float dout[4];
  READ_REALIZED_F32(ctx, realized[0], dout, 4);
  ASSERT_FLOAT_EQ(dout[0], 11.0f, 1e-5f);
  ASSERT_FLOAT_EQ(dout[1], 22.0f, 1e-5f);
  ASSERT_FLOAT_EQ(dout[2], 33.0f, 1e-5f);
  ASSERT_FLOAT_EQ(dout[3], 44.0f, 1e-5f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, run_linear_uses_bind_default_from_ctx_buffers) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *N = poly_uop_variable(ctx, "N", 1, 16, POLY_WEAKINT, 1, false);
  PolyUOp *bind_N = poly_uop_bind(ctx, N, 4);

  int f32 = poly_dtype_id_by_name("float32");
  PolyUOp *a = poly_test_buffer_var_by_id(ctx, f32, bind_N, NULL, 0, POLY_DEVICE_CPU);
  PolyUOp *out = poly_test_buffer_var_by_id(ctx, f32, bind_N, NULL, 0, POLY_DEVICE_CPU);
  PolyUOp *add = poly_alu2(ctx, POLY_OP_ADD, a, poly_const_float(ctx, 1.0));
  PolyUOp *store = poly_store_val(ctx, out, add);
  PolyUOp *sink = poly_sink1(ctx, store);
  PolyUOp *a_storage = poly_uop_buf_uop(ctx, a);
  PolyUOp *out_storage = poly_uop_buf_uop(ctx, out);
  ASSERT_NOT_NULL(a_storage);
  ASSERT_NOT_NULL(out_storage);

  float a_data[16];
  float out_data[16];
  for (int i = 0; i < 16; i++) {
    a_data[i] = (float)(i + 1);
    out_data[i] = -999.0f;
  }
  poly_buffer_set(ctx, a_storage, a_data, sizeof(a_data), POLY_DEVICE_CPU);
  poly_buffer_set(ctx, out_storage, out_data, sizeof(out_data), POLY_DEVICE_CPU);

  PolyVarBinding *default_vars = NULL;
  int n_default_vars = 0;
  PolyUOp *sched = poly_linear_effect_sink(ctx, sink, &default_vars, &n_default_vars);
  ASSERT_NOT_NULL(sched);
  ASSERT_INT_EQ(sched->n_src, 1);
  ASSERT_INT_EQ(n_default_vars, 1);
  ASSERT_TRUE(poly_uop_is_alu_param(default_vars[0].var));
  ASSERT_STR_EQ(default_vars[0].var->arg.param->name, "N");
  ASSERT_INT_EQ(default_vars[0].value, 4);
  ASSERT_INT_EQ(
      poly_run_linear(ctx, sched, default_vars, n_default_vars, NULL, 0, true, false, false), 0
  );
  ASSERT_INT_EQ(poly_buffer_read(ctx, out_storage, out_data, sizeof(out_data)), 0);

  for (int i = 0; i < 4; i++)
    ASSERT_FLOAT_EQ(out_data[i], (float)(i + 2), 1e-5f);
  ASSERT_FLOAT_EQ(out_data[4], -999.0f, 1e-5f);

  for (int i = 0; i < 16; i++)
    out_data[i] = -999.0f;
  ASSERT_INT_EQ(poly_buffer_write(ctx, out_storage, out_data, sizeof(out_data)), 0);
  PolyVarBinding bind_6 = {.var = default_vars[0].var, .value = 6};
  ASSERT_INT_EQ(poly_run_linear(ctx, sched, &bind_6, 1, NULL, 0, true, false, false), 0);
  ASSERT_INT_EQ(poly_buffer_read(ctx, out_storage, out_data, sizeof(out_data)), 0);
  for (int i = 0; i < 6; i++)
    ASSERT_FLOAT_EQ(out_data[i], (float)(i + 2), 1e-5f);
  ASSERT_FLOAT_EQ(out_data[6], -999.0f, 1e-5f);

  free(default_vars);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, run_linear_dynamic_var_override_from_ctx_buffers) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *N = poly_uop_variable(ctx, "N", 1, 16, POLY_WEAKINT, 1, false);
  PolyUOp *a = poly_test_buffer_var(ctx, POLY_FLOAT32, N, NULL, 0);
  PolyUOp *out = poly_test_buffer_var(ctx, POLY_FLOAT32, N, NULL, 0);
  PolyUOp *add = poly_alu2(ctx, POLY_OP_ADD, a, poly_const_float(ctx, 1.0));
  PolyUOp *store = poly_store_val(ctx, out, add);
  PolyUOp *sink = poly_sink1(ctx, store);
  PolyUOp *a_storage = poly_uop_buf_uop(ctx, a);
  PolyUOp *out_storage = poly_uop_buf_uop(ctx, out);
  ASSERT_NOT_NULL(a_storage);
  ASSERT_NOT_NULL(out_storage);

  float a_data[16];
  float out_data[16];
  for (int i = 0; i < 16; i++)
    a_data[i] = (float)(i + 1);
  for (int i = 0; i < 16; i++)
    out_data[i] = -999.0f;
  poly_buffer_set(ctx, a_storage, a_data, sizeof(a_data), POLY_DEVICE_CPU);
  poly_buffer_set(ctx, out_storage, out_data, sizeof(out_data), POLY_DEVICE_CPU);

  PolyUOp *sched = poly_test_create_linear(ctx, sink);
  ASSERT_NOT_NULL(sched);
  ASSERT_INT_EQ(poly_run_linear(ctx, sched, NULL, 0, NULL, 0, true, false, false), -1);

  PolyVarBinding bind = {.var = N, .value = 6};
  ASSERT_INT_EQ(poly_run_linear(ctx, sched, &bind, 1, NULL, 0, true, false, false), 0);
  ASSERT_INT_EQ(poly_buffer_read(ctx, out_storage, out_data, sizeof(out_data)), 0);
  for (int i = 0; i < 6; i++)
    ASSERT_FLOAT_EQ(out_data[i], (float)(i + 2), 1e-5f);
  ASSERT_FLOAT_EQ(out_data[6], -999.0f, 1e-5f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, contiguous_scalar_reduce_materializes_without_extra_copy_kernel) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer_f32(ctx, 4);
  int64_t axis[] = {0};
  PolyUOp *sum1 = poly_reduce_axis(ctx, POLY_OP_ADD, a, axis, 1);
  PolyUOp *sum0 = poly_reshape(ctx, sum1, NULL, 0);
  PolyUOp *target = poly_contiguous(ctx, sum0);

  float da[] = {1.0f, 2.0f, 3.0f, 4.0f};
  poly_buffer_set(ctx, a, da, sizeof(da), POLY_DEVICE_CPU);

  PolyUOp *targets[] = {target};
  PolyUOp *realized[] = {NULL};
  PolyUOp *sched = poly_test_linear_values(ctx, targets, 1, realized);
  ASSERT_NOT_NULL(sched);
  ASSERT_INT_EQ(sched->n_src, 1);
  ASSERT_INT_EQ(count_root_ops(ctx, poly_test_linear_call_body(sched, 0), POLY_OP_REDUCE), 1);

  ASSERT_INT_EQ(poly_run_linear(ctx, sched, NULL, 0, NULL, 0, true, false, false), 0);
  ASSERT_NOT_NULL(realized[0]);
  const PolyUOp *out = poly_uop_get_buffer_identity(realized[0]);
  ASSERT_NOT_NULL(out);
  PolyBuffer *buf = poly_buffer_get(ctx, (PolyUOp *)out);
  ASSERT_NOT_NULL(buf);
  float reduced = 0.0f;
  READ_REALIZED_F32(ctx, realized[0], &reduced, 1);
  ASSERT_FLOAT_EQ(reduced, 10.0f, 1e-5f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, schedule_with_vars_chained_singleton_reduce_matches_tinygrad_counts) {
  PolyCtx *ctx = poly_ctx_new();

  /* tinygrad reference for Tensor.empty(1,4,3).sum(axis=0).sum(axis=1):
   * SCHEDULE_LEN=1 and the scheduled root contains exactly
   * STORE=1 REDUCE=1 RANGE=2 END=1 INDEX=2 WHERE=0. */
  PolyUOp *x_buf = poly_buffer_f32(ctx, 12);
  PolyUOp *x = poly_reshape(ctx, x_buf, (int64_t[]){1, 4, 3}, 3);
  PolyUOp *r0 = poly_reduce_axis(ctx, POLY_OP_ADD, x, (int64_t[]){0}, 1);
  PolyUOp *r1 = poly_reduce_axis(ctx, POLY_OP_ADD, r0, (int64_t[]){1}, 1);

  PolyUOp *targets[] = {r1};
  PolyUOp *realized[] = {NULL};
  PolyUOp *sched = poly_test_linear_values(ctx, targets, 1, realized);
  ASSERT_NOT_NULL(sched);
  ASSERT_INT_EQ(sched->n_src, 1);
  ASSERT_NOT_NULL(realized[0]);

  PolyUOp *root = poly_test_linear_call_body(sched, 0);
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_STORE), 1);
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_REDUCE), 1);
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_RANGE), 2);
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_END), 1);
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_INDEX), 2);
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_WHERE), 0);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, schedule_with_vars_mixed_multiaxis_singleton_reduce_matches_tinygrad_counts) {
  PolyCtx *ctx = poly_ctx_new();

  /* tinygrad reference for Tensor.empty(1,2,3).sum(axis=(0,2)):
   * SCHEDULE_LEN=1 and the scheduled root contains exactly
   * STORE=1 REDUCE=1 RANGE=2 END=1 INDEX=2 WHERE=0. */
  PolyUOp *x_buf = poly_buffer_f32(ctx, 6);
  PolyUOp *x = poly_reshape(ctx, x_buf, (int64_t[]){1, 2, 3}, 3);
  PolyUOp *r = poly_reduce_axis(ctx, POLY_OP_ADD, x, (int64_t[]){0, 2}, 2);

  PolyUOp *targets[] = {r};
  PolyUOp *realized[] = {NULL};
  PolyUOp *sched = poly_test_linear_values(ctx, targets, 1, realized);
  ASSERT_NOT_NULL(sched);
  ASSERT_INT_EQ(sched->n_src, 1);
  ASSERT_NOT_NULL(realized[0]);

  PolyUOp *root = poly_test_linear_call_body(sched, 0);
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_STORE), 1);
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_REDUCE), 1);
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_RANGE), 2);
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_END), 1);
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_INDEX), 2);
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_WHERE), 0);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, schedule_with_vars_flip_add_matches_tinygrad_counts) {
  PolyCtx *ctx = poly_ctx_new();

  /* tinygrad reference for Tensor.empty(2,4).flip(1) + Tensor.empty(2,4):
   * SCHEDULE_LEN=1 and the scheduled root contains exactly
   * STORE=1 REDUCE=0 RANGE=2 END=1 INDEX=3 WHERE=0 LOAD=0 BUFFERIZE=0. */
  PolyUOp *x_buf = poly_buffer_f32(ctx, 8);
  PolyUOp *x = poly_reshape(ctx, x_buf, (int64_t[]){2, 4}, 2);
  PolyUOp *y = poly_alu2(ctx, POLY_OP_ADD, poly_flip(ctx, x, (int64_t[]){1}, 1), x);

  PolyUOp *targets[] = {y};
  PolyUOp *realized[] = {NULL};
  PolyUOp *sched = poly_test_linear_values(ctx, targets, 1, realized);
  ASSERT_NOT_NULL(sched);
  ASSERT_INT_EQ(sched->n_src, 1);
  ASSERT_NOT_NULL(realized[0]);

  PolyUOp *root = poly_test_linear_call_body(sched, 0);
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_STORE), 1);
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_REDUCE), 0);
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_RANGE), 2);
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_END), 1);
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_INDEX), 3);
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_WHERE), 0);
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_LOAD), 0);
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_STAGE), 0);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, schedule_with_vars_reshape_add_matches_tinygrad_counts) {
  PolyCtx *ctx = poly_ctx_new();

  /* tinygrad reference for (Tensor.empty(6) + Tensor.empty(6)).reshape(2,3):
   * SCHEDULE_LEN=1 and the scheduled root contains exactly
   * STORE=1 REDUCE=0 RANGE=1 END=1 INDEX=3 WHERE=0 LOAD=0 BUFFERIZE=0. */
  PolyUOp *a = poly_buffer_f32(ctx, 6);
  PolyUOp *b = poly_buffer_f32(ctx, 6);
  PolyUOp *x = poly_alu2(ctx, POLY_OP_ADD, a, b);
  PolyUOp *y = poly_reshape(ctx, x, (int64_t[]){2, 3}, 2);

  PolyUOp *targets[] = {y};
  PolyUOp *realized[] = {NULL};
  PolyUOp *sched = poly_test_linear_values(ctx, targets, 1, realized);
  ASSERT_NOT_NULL(sched);
  ASSERT_INT_EQ(sched->n_src, 1);
  ASSERT_NOT_NULL(realized[0]);

  PolyUOp *root = poly_test_linear_call_body(sched, 0);
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_STORE), 1);
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_REDUCE), 0);
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_RANGE), 1);
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_END), 1);
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_INDEX), 3);
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_WHERE), 0);
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_LOAD), 0);
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_STAGE), 0);
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_RESHAPE), 0);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, schedule_with_vars_pad_add_matches_tinygrad_counts) {
  PolyCtx *ctx = poly_ctx_new();

  /* tinygrad reference for (Tensor.empty(3) + Tensor.empty(3)).pad(((1,2),)):
   * SCHEDULE_LEN=1 and the scheduled root contains exactly
   * STORE=1 REDUCE=0 RANGE=1 END=1 INDEX=3 WHERE=0 LOAD=0 BUFFERIZE=0.
   * The frontend-visible realized value remains PAD(BUFFER, ...). */
  PolyUOp *a = poly_buffer_f32(ctx, 3);
  PolyUOp *b = poly_buffer_f32(ctx, 3);
  PolyUOp *x = poly_alu2(ctx, POLY_OP_ADD, a, b);
  PolyUOp *y = poly_pad(ctx, x, (int64_t[1][2]){{1, 2}}, 1);

  PolyUOp *targets[] = {y};
  PolyUOp *realized[] = {NULL};
  PolyUOp *sched = poly_test_linear_values(ctx, targets, 1, realized);
  ASSERT_NOT_NULL(sched);
  ASSERT_INT_EQ(sched->n_src, 1);
  ASSERT_NOT_NULL(realized[0]);

  PolyUOp *root = poly_test_linear_call_body(sched, 0);
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_STORE), 1);
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_REDUCE), 0);
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_RANGE), 1);
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_END), 1);
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_INDEX), 3);
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_WHERE), 0);
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_LOAD), 0);
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_STAGE), 0);
  ASSERT_INT_EQ(realized[0]->op, POLY_OP_PAD);
  ASSERT_TRUE(realized[0]->n_src >= 1);
  ASSERT_TRUE(poly_uop_get_buffer_identity(realized[0]->src[0]) != NULL);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, schedule_with_vars_expand_add_matches_tinygrad_counts) {
  PolyCtx *ctx = poly_ctx_new();

  /* Current Tinygrad reference for
   * (Tensor.empty(2,1) + Tensor.empty(2,1)).expand(2,4):
   * SCHEDULE_LEN=1 and the scheduled root contains exactly
   * STORE=1 REDUCE=0 RANGE=1 END=1 INDEX=3 WHERE=0 LOAD=0 BUFFERIZE=0.
   * The frontend-visible value follows _broadcast_to's
   * PERMUTE(EXPAND(RESHAPE(BUFFER))) layout. */
  PolyUOp *a_buf = poly_buffer_f32(ctx, 2);
  PolyUOp *b_buf = poly_buffer_f32(ctx, 2);
  PolyUOp *a = poly_reshape(ctx, a_buf, (int64_t[]){2, 1}, 2);
  PolyUOp *b = poly_reshape(ctx, b_buf, (int64_t[]){2, 1}, 2);
  PolyUOp *x = poly_alu2(ctx, POLY_OP_ADD, a, b);
  PolyUOp *y = poly_expand(ctx, x, (int64_t[]){2, 4}, 2);

  PolyUOp *targets[] = {y};
  PolyUOp *realized[] = {NULL};
  PolyUOp *sched = poly_test_linear_values(ctx, targets, 1, realized);
  ASSERT_NOT_NULL(sched);
  ASSERT_INT_EQ(sched->n_src, 1);
  ASSERT_NOT_NULL(realized[0]);

  PolyUOp *root = poly_test_linear_call_body(sched, 0);
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_STORE), 1);
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_REDUCE), 0);
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_RANGE), 1);
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_END), 1);
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_INDEX), 3);
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_WHERE), 0);
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_LOAD), 0);
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_STAGE), 0);
  ASSERT_INT_EQ(realized[0]->op, POLY_OP_PERMUTE);
  ASSERT_TRUE(realized[0]->n_src >= 1);
  ASSERT_INT_EQ(realized[0]->src[0]->op, POLY_OP_EXPAND);
  ASSERT_INT_EQ(realized[0]->src[0]->src[0]->op, POLY_OP_RESHAPE);
  ASSERT_TRUE(poly_uop_get_buffer_identity(realized[0]->src[0]->src[0]) != NULL);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, schedule_with_vars_cross_entropy_sparse_last_axis_matches_tinygrad_kernel_count) {
  PolyCtx *ctx = poly_ctx_new();

  /* tinygrad reference for Tensor.zeros(2,3).cross_entropy(Tensor([0,2], dtype=int32)):
   * sched_len=3. The final *-1/N stays fused into the last reduction kernel. */
  PolyUOp *logits_buf = poly_buffer_f32(ctx, 6);
  PolyUOp *target_buf = poly_test_buffer(ctx, POLY_INT32, 2);
  PolyUOp *logits = poly_reshape(ctx, logits_buf, (int64_t[]){2, 3}, 2);
  PolyUOp *target = poly_reshape(ctx, target_buf, (int64_t[]){2}, 1);
  PolyUOp *loss = poly_cross_entropy(ctx, logits, target, 1);

  PolyUOp *targets[] = {loss};
  PolyUOp *realized[] = {NULL};
  PolyUOp *sched = poly_test_linear_values(ctx, targets, 1, realized);
  ASSERT_NOT_NULL(sched);
  ASSERT_NOT_NULL(realized[0]);
  ASSERT_INT_EQ(sched->n_src, 3);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, schedule_with_vars_cross_entropy_sparse_non_last_axis_matches_tinygrad_kernel_count) {
  PolyCtx *ctx = poly_ctx_new();

  /* tinygrad reference for Tensor.zeros(2,3,2).cross_entropy(target int32[2,2]):
   * sched_len=3. The final *-1/N stays fused into the last reduction kernel. */
  PolyUOp *logits_buf = poly_buffer_f32(ctx, 12);
  PolyUOp *target_buf = poly_test_buffer(ctx, POLY_INT32, 4);
  PolyUOp *logits = poly_reshape(ctx, logits_buf, (int64_t[]){2, 3, 2}, 3);
  PolyUOp *target = poly_reshape(ctx, target_buf, (int64_t[]){2, 2}, 2);
  PolyUOp *loss = poly_cross_entropy(ctx, logits, target, -2);

  PolyUOp *targets[] = {loss};
  PolyUOp *realized[] = {NULL};
  PolyUOp *sched = poly_test_linear_values(ctx, targets, 1, realized);
  ASSERT_NOT_NULL(sched);
  ASSERT_NOT_NULL(realized[0]);
  ASSERT_INT_EQ(sched->n_src, 3);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, schedule_with_vars_mixed_passthrough_and_unrealized) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer_f32(ctx, 4);
  PolyUOp *b = poly_buffer_f32(ctx, 4);
  PolyUOp *add = poly_alu2(ctx, POLY_OP_ADD, a, b);

  float da[] = {1.0f, 2.0f, 3.0f, 4.0f};
  float db[] = {10.0f, 20.0f, 30.0f, 40.0f};
  poly_buffer_set(ctx, a, da, sizeof(da), POLY_DEVICE_CPU);
  poly_buffer_set(ctx, b, db, sizeof(db), POLY_DEVICE_CPU);

  PolyUOp *targets[] = {a, add};
  PolyUOp *realized[] = {NULL, NULL};
  PolyUOp *sched = poly_test_linear_values(ctx, targets, 2, realized);
  ASSERT_NOT_NULL(sched);
  ASSERT_PTR_EQ(realized[0], a);
  ASSERT_NOT_NULL(realized[1]);
  ASSERT_PTR_NEQ(realized[1], add);

  ASSERT_INT_EQ(poly_run_linear(ctx, sched, NULL, 0, NULL, 0, true, false, false), 0);

  PolyBuffer *buf = realized_buffer(ctx, realized[1]);
  ASSERT_NOT_NULL(buf);
  float dout[4];
  READ_REALIZED_F32(ctx, realized[1], dout, 4);
  ASSERT_FLOAT_EQ(dout[0], 11.0f, 1e-5f);
  ASSERT_FLOAT_EQ(dout[1], 22.0f, 1e-5f);
  ASSERT_FLOAT_EQ(dout[2], 33.0f, 1e-5f);
  ASSERT_FLOAT_EQ(dout[3], 44.0f, 1e-5f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, schedule_with_vars_multi_target_batch) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer_f32(ctx, 4);
  PolyUOp *b = poly_buffer_f32(ctx, 4);
  PolyUOp *add = poly_alu2(ctx, POLY_OP_ADD, a, b);
  PolyUOp *mul = poly_alu2(ctx, POLY_OP_MUL, a, b);

  float da[] = {1.0f, 2.0f, 3.0f, 4.0f};
  float db[] = {10.0f, 20.0f, 30.0f, 40.0f};
  poly_buffer_set(ctx, a, da, sizeof(da), POLY_DEVICE_CPU);
  poly_buffer_set(ctx, b, db, sizeof(db), POLY_DEVICE_CPU);

  PolyUOp *targets[] = {add, mul};
  PolyUOp *realized[] = {NULL, NULL};
  PolyUOp *sched = poly_test_linear_values(ctx, targets, 2, realized);
  ASSERT_NOT_NULL(sched);
  ASSERT_NOT_NULL(realized[0]);
  ASSERT_NOT_NULL(realized[1]);
  /* Tinygrad Tensor.realize keeps every output Tensor/UOp live through all reads;
   * raw C callers must retain borrowed output roots across allocation safe points. */
  ASSERT_INT_EQ(poly_uop_retain(ctx, realized[0]), 0);
  ASSERT_INT_EQ(poly_uop_retain(ctx, realized[1]), 0);

  ASSERT_INT_EQ(poly_run_linear(ctx, sched, NULL, 0, NULL, 0, true, false, false), 0);

  PolyBuffer *add_buf = realized_buffer(ctx, realized[0]);
  PolyBuffer *mul_buf = realized_buffer(ctx, realized[1]);
  ASSERT_NOT_NULL(add_buf);
  ASSERT_NOT_NULL(mul_buf);
  float add_out[4];
  float mul_out[4];
  READ_REALIZED_F32(ctx, realized[0], add_out, 4);
  READ_REALIZED_F32(ctx, realized[1], mul_out, 4);
  ASSERT_FLOAT_EQ(add_out[0], 11.0f, 1e-5f);
  ASSERT_FLOAT_EQ(add_out[3], 44.0f, 1e-5f);
  ASSERT_FLOAT_EQ(mul_out[0], 10.0f, 1e-5f);
  ASSERT_FLOAT_EQ(mul_out[3], 160.0f, 1e-5f);

  poly_uop_release(ctx, realized[1]);
  poly_uop_release(ctx, realized[0]);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, tensor_zero_size_placement_copy_adds_no_call) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  int64_t dims[] = {0};
  PolyTensor *empty = poly_tensor_empty(ctx, POLY_FLOAT32, dims, 1, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(empty);
  ASSERT_NOT_NULL(poly_tensor_uop_physical(empty));

  PolyTensor *realized_tensor = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &empty, 1, &realized_tensor), 0);
  ASSERT_PTR_EQ(realized_tensor, empty);
  PolyUOp *realized = poly_tensor_uop(empty);
  ASSERT_NOT_NULL(realized);
  const PolyUOp *identity = poly_uop_get_buffer_identity(realized);
  ASSERT_NOT_NULL(identity);
  ASSERT_EQ(identity->op, POLY_OP_BUFFER);
  ASSERT_EQ(identity->arg.kind, POLY_ARG_PARAM);
  ASSERT_NOT_NULL(identity->arg.param);
  ASSERT_INT_EQ(identity->n_src, 1);
  ASSERT_EQ(identity->src[0]->op, POLY_OP_CONST);
  ASSERT_INT_EQ(identity->src[0]->arg.i, 0);
  ASSERT_TRUE(poly_buffer_get(ctx, (PolyUOp *)identity) == NULL);
  ASSERT_FALSE(poly_buffer_is_allocated(ctx, (PolyUOp *)identity));

  PolyShape shape = poly_uop_max_shape_cached(ctx, realized);
  ASSERT_INT_EQ(shape.ndim, 1);
  ASSERT_INT_EQ(shape.dims[0], 0);
  PolyCtxStats stats = {0};
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &stats), 0);
  ASSERT_INT_EQ(stats.kernel_count, 0);
  ASSERT_TRUE(stats.mem_used == 0);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, default_realize_rejects_logical_only_tensor) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  int64_t dims[1] = {2};
  PolyTensor *direct = poly_tensor_empty(ctx, POLY_FLOAT32, dims, 1, POLY_DEVICE_CPU);
  PolyUOp *legacy_root = poly_test_buffer(ctx, POLY_FLOAT32, 2);
  PolyTensor *legacy =
      poly_tensor_create_with_roots(ctx, legacy_root, NULL, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(direct);
  ASSERT_NOT_NULL(direct->uop_physical);
  ASSERT_NOT_NULL(legacy);
  ASSERT_EQ(legacy->uop_physical, NULL);

  PolyTensor *direct_out = NULL;
  PolyTensor *legacy_out = NULL;
  int direct_rc = poly_realize_tensors(ctx, &direct, 1, &direct_out);
  int rejected_rc = poly_realize_tensors(ctx, &legacy, 1, &legacy_out);
  bool rejected_kept_logical_only = legacy->uop_physical == NULL;
  ASSERT_INT_EQ(direct_rc, 0);
  ASSERT_PTR_EQ(direct_out, direct);
  ASSERT_INT_EQ(rejected_rc, -1);
  ASSERT_TRUE(rejected_kept_logical_only);
  ASSERT_TRUE(legacy_out == NULL);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, tensor_zero_axis_pure_full_short_circuits_earlier_numel_overflow) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  int64_t dims[3] = {INT64_MAX, 2, 0};
  PolyTensor *empty = poly_tensor_full_float_by_id(
      ctx, dims, 3, 0.0, poly_dtype_id_by_name("float32"), POLY_DEVICE_CPU, true, false
  );
  ASSERT_NOT_NULL(empty);

  PolyTensor *realized_tensor = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &empty, 1, &realized_tensor), 0);
  ASSERT_PTR_EQ(realized_tensor, empty);
  PolyUOp *realized = poly_tensor_uop(empty);
  ASSERT_NOT_NULL(realized);
  /* Pinned tinygrad full(buffer=False) remains the pure EXPAND graph after
   * realize when numel is zero (mixin/__init__.py:55-77). */
  ASSERT_EQ(realized->op, POLY_OP_EXPAND);
  PolyShape shape = poly_uop_max_shape_cached(ctx, realized);
  ASSERT_INT_EQ(shape.ndim, 3);
  ASSERT_TRUE(shape.dims[0] == INT64_MAX);
  ASSERT_INT_EQ(shape.dims[1], 2);
  ASSERT_INT_EQ(shape.dims[2], 0);

  ASSERT_TRUE(poly_uop_get_buffer_identity(realized) == NULL);
  ASSERT_TRUE(poly_uop_buffer(ctx, realized) == NULL);
  PolyCtxStats stats = {0};
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &stats), 0);
  ASSERT_INT_EQ(stats.kernel_count, 0);
  ASSERT_TRUE(stats.mem_used == 0);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, schedule_with_vars_preserves_2d_shape) {
  PolyCtx *ctx = poly_ctx_new();
  int64_t dims[] = {2, 3};

  PolyUOp *a_buf = poly_buffer_f32(ctx, 6);
  PolyUOp *b_buf = poly_buffer_f32(ctx, 6);
  PolyUOp *a = poly_reshape(ctx, a_buf, dims, 2);
  PolyUOp *b = poly_reshape(ctx, b_buf, dims, 2);
  PolyUOp *add = poly_alu2(ctx, POLY_OP_ADD, a, b);

  float da[] = {1, 2, 3, 4, 5, 6};
  float db[] = {10, 20, 30, 40, 50, 60};
  poly_buffer_set(ctx, a_buf, da, sizeof(da), POLY_DEVICE_CPU);
  poly_buffer_set(ctx, b_buf, db, sizeof(db), POLY_DEVICE_CPU);

  PolyUOp *targets[] = {add};
  PolyUOp *realized[] = {NULL};
  PolyUOp *sched = poly_test_linear_values(ctx, targets, 1, realized);
  ASSERT_NOT_NULL(sched);
  ASSERT_NOT_NULL(realized[0]);
  ASSERT_EQ(realized[0]->op, POLY_OP_RESHAPE);
  PolyShape s = poly_uop_max_shape_cached(ctx, realized[0]);
  PolyShape expected = {dims, 2};
  ASSERT_TRUE(poly_shape_eq(s, expected));

  ASSERT_INT_EQ(poly_run_linear(ctx, sched, NULL, 0, NULL, 0, true, false, false), 0);
  PolyBuffer *buf = realized_buffer(ctx, realized[0]);
  ASSERT_NOT_NULL(buf);
  float dout[6];
  READ_REALIZED_F32(ctx, realized[0], dout, 6);
  ASSERT_FLOAT_EQ(dout[0], 11.0f, 1e-5f);
  ASSERT_FLOAT_EQ(dout[5], 66.0f, 1e-5f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, schedule_with_vars_assign_in_place) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer_f32(ctx, 4);
  PolyUOp *two = poly_const_float(ctx, 2.0f);
  PolyUOp *mul = poly_alu2(ctx, POLY_OP_MUL, a, two);
  PolyUOp *store = poly_store_val(ctx, a, mul);
  PolyUOp *assign_srcs[2] = {a, store};
  PolyUOp *assign = poly_uop(ctx, POLY_OP_AFTER, a->dtype, assign_srcs, 2, poly_arg_none());

  float da[] = {1.0f, 2.0f, 3.0f, 4.0f};
  poly_buffer_set(ctx, a, da, sizeof(da), POLY_DEVICE_CPU);

  PolyUOp *targets[] = {assign};
  PolyUOp *realized[] = {NULL};
  PolyUOp *sched = poly_test_linear_values(ctx, targets, 1, realized);
  ASSERT_NOT_NULL(sched);
  ASSERT_PTR_EQ(realized[0], a);

  ASSERT_INT_EQ(poly_run_linear(ctx, sched, NULL, 0, NULL, 0, true, false, false), 0);
  ASSERT_INT_EQ(poly_buffer_read(ctx, a, da, sizeof(da)), 0);
  ASSERT_FLOAT_EQ(da[0], 2.0f, 1e-5f);
  ASSERT_FLOAT_EQ(da[1], 4.0f, 1e-5f);
  ASSERT_FLOAT_EQ(da[2], 6.0f, 1e-5f);
  ASSERT_FLOAT_EQ(da[3], 8.0f, 1e-5f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, schedule_batched_dependent_assign_executes_shared_effect_once) {
  for (int reverse = 0; reverse < 2; reverse++) {
    PolyCtx *ctx = poly_ctx_new();
    ASSERT_NOT_NULL(ctx);

    PolyUOp *a = poly_buffer_f32(ctx, 1);
    PolyUOp *b = poly_buffer_f32(ctx, 1);
    float da[] = {0.0f};
    float db[] = {0.0f};
    poly_buffer_set(ctx, a, da, sizeof(da), POLY_DEVICE_CPU);
    poly_buffer_set(ctx, b, db, sizeof(db), POLY_DEVICE_CPU);

    PolyUOp *a_value = poly_alu2(ctx, POLY_OP_ADD, a, poly_const_float(ctx, 1.0f));
    PolyUOp *a_store = poly_store_val(ctx, a, a_value);
    PolyUOp *a_after_src[2] = {a, a_store};
    PolyUOp *a_after = poly_uop(ctx, POLY_OP_AFTER, a->dtype, a_after_src, 2, poly_arg_none());
    PolyUOp *b_store = poly_store_val(ctx, b, a_after);
    PolyUOp *b_after_src[2] = {b, b_store};
    PolyUOp *b_after = poly_uop(ctx, POLY_OP_AFTER, b->dtype, b_after_src, 2, poly_arg_none());

    PolyUOp *targets[2] = {
        reverse ? b_after : a_after,
        reverse ? a_after : b_after,
    };
    PolyUOp *realized[2] = {NULL, NULL};
    PolyUOp *schedule = poly_test_linear_values(ctx, targets, 2, realized);
    ASSERT_NOT_NULL(schedule);
    ASSERT_INT_EQ(schedule->n_src, 2);
    ASSERT_INT_EQ(poly_run_linear(ctx, schedule, NULL, 0, NULL, 0, true, false, false), 0);
    ASSERT_INT_EQ(poly_buffer_read(ctx, a, da, sizeof(da)), 0);
    ASSERT_INT_EQ(poly_buffer_read(ctx, b, db, sizeof(db)), 0);
    ASSERT_FLOAT_EQ(da[0], 1.0f, 1e-5f);
    ASSERT_FLOAT_EQ(db[0], 1.0f, 1e-5f);

    poly_ctx_destroy(ctx);
  }
  PASS();
}

TEST(realize, schedule_with_vars_empty_input_allocates_like_tinygrad) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer_f32(ctx, 4);
  PolyUOp *b = poly_buffer_f32(ctx, 4);
  PolyUOp *add = poly_alu2(ctx, POLY_OP_ADD, a, b);

  float da[] = {1.0f, 2.0f, 3.0f, 4.0f};
  poly_buffer_set(ctx, a, da, sizeof(da), POLY_DEVICE_CPU);

  PolyUOp *targets[] = {add};
  PolyUOp *realized[] = {NULL};
  PolyUOp *sched = poly_test_linear_values(ctx, targets, 1, realized);
  ASSERT_NOT_NULL(sched);
  ASSERT_INT_EQ(poly_run_linear(ctx, sched, NULL, 0, NULL, 0, true, false, false), 0);
  ASSERT_NOT_NULL(poly_buffer_get(ctx, b));
  ASSERT_NOT_NULL(realized[0]);
  ASSERT_NOT_NULL(realized_buffer(ctx, realized[0]));

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, schedule_with_vars_triu_root_has_no_early_loads) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *in = poly_buffer_f32(ctx, 9);
  PolyUOp *in2d = poly_reshape(ctx, in, (int64_t[]){3, 3}, 2);
  PolyUOp *tri = poly_triu(ctx, in2d, 0);
  ASSERT_NOT_NULL(tri);

  float din[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  poly_buffer_set(ctx, in, din, sizeof(din), POLY_DEVICE_CPU);

  PolyUOp *targets[] = {tri};
  PolyUOp *realized[] = {NULL};
  PolyUOp *sched = poly_test_linear_values(ctx, targets, 1, realized);
  ASSERT_NOT_NULL(sched);
  ASSERT_INT_EQ(sched->n_src, 1);
  ASSERT_NOT_NULL(realized[0]);

  /* tinygrad schedule_with_vars boundary: triu root still has WHERE and no
   * LOAD. Late load insertion happens later in codegen pm_add_loads. */
  ASSERT_INT_EQ(count_root_ops(ctx, poly_test_linear_call_body(sched, 0), POLY_OP_LOAD), 0);
  ASSERT_TRUE(count_root_ops(ctx, poly_test_linear_call_body(sched, 0), POLY_OP_WHERE) > 0);

  ASSERT_INT_EQ(poly_run_linear(ctx, sched, NULL, 0, NULL, 0, true, false, false), 0);
  PolyBuffer *buf = realized_buffer(ctx, realized[0]);
  ASSERT_NOT_NULL(buf);
  float dout[9];
  READ_REALIZED_F32(ctx, realized[0], dout, 9);
  ASSERT_FLOAT_EQ(dout[0], 1.0f, 1e-5f);
  ASSERT_FLOAT_EQ(dout[1], 2.0f, 1e-5f);
  ASSERT_FLOAT_EQ(dout[2], 3.0f, 1e-5f);
  ASSERT_FLOAT_EQ(dout[3], 0.0f, 1e-5f);
  ASSERT_FLOAT_EQ(dout[4], 5.0f, 1e-5f);
  ASSERT_FLOAT_EQ(dout[8], 9.0f, 1e-5f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, schedule_with_vars_tril_root_has_no_early_loads) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *in = poly_buffer_f32(ctx, 9);
  PolyUOp *in2d = poly_reshape(ctx, in, (int64_t[]){3, 3}, 2);
  PolyUOp *tri = poly_tril(ctx, in2d, 0);
  ASSERT_NOT_NULL(tri);

  float din[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  poly_buffer_set(ctx, in, din, sizeof(din), POLY_DEVICE_CPU);

  PolyUOp *targets[] = {tri};
  PolyUOp *realized[] = {NULL};
  PolyUOp *sched = poly_test_linear_values(ctx, targets, 1, realized);
  ASSERT_NOT_NULL(sched);
  ASSERT_INT_EQ(sched->n_src, 1);
  ASSERT_NOT_NULL(realized[0]);

  ASSERT_INT_EQ(count_root_ops(ctx, poly_test_linear_call_body(sched, 0), POLY_OP_LOAD), 0);
  ASSERT_TRUE(count_root_ops(ctx, poly_test_linear_call_body(sched, 0), POLY_OP_WHERE) > 0);

  ASSERT_INT_EQ(poly_run_linear(ctx, sched, NULL, 0, NULL, 0, true, false, false), 0);
  PolyBuffer *buf = realized_buffer(ctx, realized[0]);
  ASSERT_NOT_NULL(buf);
  float dout[9];
  READ_REALIZED_F32(ctx, realized[0], dout, 9);
  ASSERT_FLOAT_EQ(dout[0], 1.0f, 1e-5f);
  ASSERT_FLOAT_EQ(dout[1], 0.0f, 1e-5f);
  ASSERT_FLOAT_EQ(dout[2], 0.0f, 1e-5f);
  ASSERT_FLOAT_EQ(dout[3], 4.0f, 1e-5f);
  ASSERT_FLOAT_EQ(dout[4], 5.0f, 1e-5f);
  ASSERT_FLOAT_EQ(dout[8], 9.0f, 1e-5f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, poly_jit_replays_raw_tensor_realize_with_new_input) {
  PolyCtx *ctx = poly_ctx_new();

  float a_data[3] = {1.0f, 2.0f, 3.0f};
  int64_t shape[] = {3};
  int f32 = poly_dtype_id_by_name("float32");
  PolyUOp *a_buf = NULL;
  PolyTensor *a = initialized_f32_tensor(ctx, shape, 1, a_data, POLY_DEVICE_CPU, &a_buf);
  ASSERT_NOT_NULL(a);

  PolyJit *jit = poly_jit_new(ctx);
  ASSERT_NOT_NULL(jit);
  ASSERT_INT_EQ(poly_jit_begin_capture(jit, &a, 1), 0);
  poly_ctx_reset_counters(ctx);

  PolyTensor *one = poly_tensor_const_float_by_id(ctx, 1.0, f32, POLY_DEVICE_CPU);
  PolyTensor *out = poly_tensor_alu2(ctx, POLY_OP_ADD, a, one);
  ASSERT_NOT_NULL(out);
  PolyTensor *realized = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &out, 1, &realized), 0);
  ASSERT_PTR_EQ(realized, out);
  const PolyUOp *out_identity = poly_uop_get_buffer_identity(poly_tensor_uop(out));
  ASSERT_NOT_NULL(out_identity);
  ASSERT_FALSE(poly_buffer_is_allocated(ctx, (PolyUOp *)out_identity));
  ASSERT_INT_EQ(poly_jit_schedule_count(jit), 1);
  PolyCtxStats capture_stats = {0};
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &capture_stats), 0);
  ASSERT_TRUE(capture_stats.global_ops == 0);
  ASSERT_TRUE(capture_stats.global_mem == 0);
  ASSERT_TRUE(capture_stats.kernel_count == 0);

  ASSERT_INT_EQ(poly_jit_end_capture(jit, ctx->tensors, ctx->n_tensors), 0);
  ASSERT_TRUE(poly_jit_is_captured(jit));
  ASSERT_INT_EQ(poly_jit_schedule_count(jit), 1);
  PolyUOp *captured = poly_jit_captured_linear(jit);
  ASSERT_NOT_NULL(captured);
  ASSERT_INT_EQ(captured->op, POLY_OP_LINEAR);
  /* Current compiled ADD has output/input PROGRAM PARAMs plus replay PARAM 0. */
  ASSERT_INT_EQ(count_shaped_value_params(ctx, captured), 3);
  bool saw_input_param = false, saw_fixed_output = false, saw_captured_input = false;
  for (int k = 0; k < captured->n_src; k++) {
    PolyUOp *call = captured->src[k];
    ASSERT_NOT_NULL(call);
    ASSERT_INT_EQ(call->op, POLY_OP_CALL);
    for (int i = 1; i < call->n_src; i++) {
      PolyUOp *arg = call->src[i];
      if (is_shaped_value_param(arg)) {
        ASSERT_INT_EQ(arg->arg.param->slot, 0);
        saw_input_param = true;
      }
      const PolyUOp *identity = poly_uop_get_buffer_identity(arg);
      if (identity == out_identity) saw_fixed_output = true;
      if (identity == a_buf) saw_captured_input = true;
    }
  }
  ASSERT_TRUE(saw_input_param);
  ASSERT_TRUE(saw_fixed_output);
  ASSERT_FALSE(saw_captured_input);
  ASSERT_TRUE(poly_buffer_is_allocated(ctx, (PolyUOp *)out_identity));
  PolyCtxStats first_stats = {0};
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &first_stats), 0);
  ASSERT_TRUE(first_stats.global_ops == 3);
  ASSERT_TRUE(first_stats.global_mem == 24);
  ASSERT_TRUE(first_stats.kernel_count == 1);

  float first[3] = {0};
  ASSERT_INT_EQ(
      poly_buffer_read(
          ctx, (PolyUOp *)poly_uop_get_buffer_identity(poly_tensor_uop(out)), first, sizeof(first)
      ),
      0
  );
  ASSERT_FLOAT_EQ(first[0], 2.0f, 1e-5f);
  ASSERT_FLOAT_EQ(first[1], 3.0f, 1e-5f);
  ASSERT_FLOAT_EQ(first[2], 4.0f, 1e-5f);

  float b_data[3] = {10.0f, 20.0f, 30.0f};
  PolyUOp *b_buf = NULL;
  PolyTensor *b = initialized_f32_tensor(ctx, shape, 1, b_data, POLY_DEVICE_CPU, &b_buf);
  ASSERT_NOT_NULL(b);

  poly_ctx_reset_counters(ctx);
  ASSERT_INT_EQ(poly_jit_run(jit, &b, 1), 0);
  PolyCtxStats replay_stats = {0};
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &replay_stats), 0);
  ASSERT_TRUE(replay_stats.global_ops == 3);
  ASSERT_TRUE(replay_stats.global_mem == 24);
  ASSERT_TRUE(replay_stats.kernel_count == 1);

  float second[3] = {0};
  ASSERT_INT_EQ(
      poly_buffer_read(
          ctx, (PolyUOp *)poly_uop_get_buffer_identity(poly_tensor_uop(out)), second, sizeof(second)
      ),
      0
  );
  ASSERT_FLOAT_EQ(second[0], 11.0f, 1e-5f);
  ASSERT_FLOAT_EQ(second[1], 21.0f, 1e-5f);
  ASSERT_FLOAT_EQ(second[2], 31.0f, 1e-5f);

  float captured_input[3] = {0};
  ASSERT_INT_EQ(poly_buffer_read(ctx, a_buf, captured_input, sizeof(captured_input)), 0);
  ASSERT_FLOAT_EQ(captured_input[0], 1.0f, 1e-5f);
  ASSERT_FLOAT_EQ(captured_input[1], 2.0f, 1e-5f);
  ASSERT_FLOAT_EQ(captured_input[2], 3.0f, 1e-5f);

  float replay_input[3] = {0};
  ASSERT_INT_EQ(poly_buffer_read(ctx, b_buf, replay_input, sizeof(replay_input)), 0);
  ASSERT_FLOAT_EQ(replay_input[0], 10.0f, 1e-5f);
  ASSERT_FLOAT_EQ(replay_input[1], 20.0f, 1e-5f);
  ASSERT_FLOAT_EQ(replay_input[2], 30.0f, 1e-5f);

  poly_jit_free(jit);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, poly_jit_capture_and_replay_require_only_physical_tensor_roots) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  ASSERT_INT_EQ(poly_ctx_set_logical_policy(ctx, POLY_LOGICAL_NEVER), 0);
  int64_t shape[] = {3};
  int f32 = poly_dtype_id_by_name("float32");
  float a_data[] = {1.0f, 2.0f, 3.0f};
  PolyTensor *a = initialized_f32_tensor(ctx, shape, 1, a_data, POLY_DEVICE_CPU, NULL);
  ASSERT_NOT_NULL(a);
  ASSERT_PTR_EQ(poly_tensor_uop_logical(a), NULL);

  PolyJit *jit = poly_jit_new(ctx);
  ASSERT_NOT_NULL(jit);
  ASSERT_INT_EQ(poly_jit_begin_capture(jit, &a, 1), 0);
  PolyTensor *one = poly_tensor_const_float_by_id(ctx, 1.0, f32, POLY_DEVICE_CPU);
  PolyTensor *out = poly_tensor_alu2(ctx, POLY_OP_ADD, a, one);
  ASSERT_NOT_NULL(out);
  ASSERT_PTR_EQ(poly_tensor_uop_logical(out), NULL);
  PolyTensor *realized = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &out, 1, &realized), 0);
  ASSERT_PTR_EQ(realized, out);
  ASSERT_INT_EQ(poly_jit_end_capture(jit, ctx->tensors, ctx->n_tensors), 0);
  ASSERT_TRUE(poly_jit_is_captured(jit));

  float b_data[] = {10.0f, 20.0f, 30.0f};
  PolyTensor *b = initialized_f32_tensor(ctx, shape, 1, b_data, POLY_DEVICE_CPU, NULL);
  ASSERT_NOT_NULL(b);
  ASSERT_PTR_EQ(poly_tensor_uop_logical(b), NULL);
  ASSERT_INT_EQ(poly_jit_run(jit, &b, 1), 0);
  float values[3] = {0};
  const PolyUOp *identity = poly_uop_get_buffer_identity(poly_tensor_uop_physical(out));
  ASSERT_NOT_NULL(identity);
  ASSERT_INT_EQ(poly_buffer_read(ctx, (PolyUOp *)identity, values, sizeof(values)), 0);
  ASSERT_FLOAT_EQ(values[0], 11.0f, 1e-5f);
  ASSERT_FLOAT_EQ(values[1], 21.0f, 1e-5f);
  ASSERT_FLOAT_EQ(values[2], 31.0f, 1e-5f);

  poly_jit_free(jit);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, live_jit_owns_captured_residency_after_tensor_release) {
  /* Tinygrad 2026-08-22 engine/jit.py:250-280 retains the final captured
   * LINEAR after dropping the pre-plan graph and frontend Tensor owners. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  int64_t shape[1] = {1024};
  float first_values[1024] = {0};
  PolyTensor *input = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(input);
  ASSERT_INT_EQ(poly_buffer_write(ctx, input->uop_physical, first_values, sizeof(first_values)), 0);

  PolyJit *jit = poly_jit_new(ctx);
  ASSERT_NOT_NULL(jit);
  ASSERT_INT_EQ(poly_jit_begin_capture(jit, &input, 1), 0);
  int f32 = poly_dtype_id_by_name("float32");
  PolyTensor *one = poly_tensor_const_float_by_id(ctx, 1.0, f32, POLY_DEVICE_CPU);
  PolyTensor *out = poly_tensor_alu2(ctx, POLY_OP_ADD, input, one);
  ASSERT_NOT_NULL(one);
  ASSERT_NOT_NULL(out);
  PolyTensor *realized = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &out, 1, &realized), 0);
  ASSERT_INT_EQ(poly_jit_end_capture(jit, ctx->tensors, ctx->n_tensors), 0);
  PolyUOp *out_buffer = (PolyUOp *)poly_uop_get_buffer_identity(out->uop_physical);
  ASSERT_NOT_NULL(out_buffer);
  ASSERT_TRUE(poly_buffer_is_allocated(ctx, out_buffer));

  poly_tensor_release(input);
  poly_tensor_release(one);
  poly_tensor_release(out);
  ASSERT_INT_EQ(poly_ctx_collect(ctx), 0);
  ASSERT_TRUE(poly_buffer_is_allocated(ctx, out_buffer));
  PolyCtxStats stats = {0};
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &stats), 0);
  /* CapturedJit owns the 4 KiB input and output after Tensor release. */
  ASSERT_TRUE(stats.mem_used == 8192);

  float next_values[1024];
  for (int i = 0; i < 1024; i++)
    next_values[i] = (float)i;
  PolyTensor *next = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(next);
  ASSERT_INT_EQ(poly_buffer_write(ctx, next->uop_physical, next_values, sizeof(next_values)), 0);
  ASSERT_INT_EQ(poly_jit_run(jit, &next, 1), 0);
  float got[1024] = {0};
  ASSERT_INT_EQ(poly_buffer_read(ctx, out_buffer, got, sizeof(got)), 0);
  ASSERT_FLOAT_EQ(got[0], 1.0f, 1e-6f);
  ASSERT_FLOAT_EQ(got[1023], 1024.0f, 1e-6f);

  poly_tensor_release(next);
  poly_jit_free(jit);
  ASSERT_INT_EQ(poly_ctx_collect(ctx), 0);
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &stats), 0);
  ASSERT_TRUE(stats.mem_used == 0);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, poly_jit_replays_tuple_device_mstack_calls_like_pinned) {
  /* Pinned TinyJit concatenates captured LINEAR calls, substitutes input
   * PARAM leaves with walk=True, and resolves MSTACK on capture execution and
   * replay (engine/jit.py:67-76,220-229,285-302;
   * engine/realize.py:142-180). The same topology also survives prune=True,
   * whose _collect_bufs walk recurses through MSTACK/MSELECT. */
  for (int prune = 0; prune < 2; prune++) {
    PolyCtx *ctx = poly_ctx_new();
    ASSERT_NOT_NULL(ctx);
    poly_ctx_set_preferred_device(ctx, POLY_DEVICE_CPU);
    PolyUOp *capture_buffer = NULL;
    PolyTensor *capture_input = initialized_i32_tensor8(ctx, 8, &capture_buffer);
    ASSERT_NOT_NULL(capture_input);
    ASSERT_NOT_NULL(capture_buffer);

    PolyJit *jit = poly_jit_new(ctx);
    ASSERT_NOT_NULL(jit);
    if (prune) ASSERT_INT_EQ(poly_jit_set_prune(jit, true), 0);
    ASSERT_INT_EQ(poly_jit_begin_capture(jit, &capture_input, 1), 0);
    PolyTensor *out = sharded_i32_tensor(ctx, capture_input);
    ASSERT_NOT_NULL(out);
    PolyTensor *realized = NULL;
    ASSERT_INT_EQ(poly_realize_tensors(ctx, &out, 1, &realized), 0);
    ASSERT_PTR_EQ(realized, out);
    ASSERT_INT_EQ(poly_jit_schedule_count(jit), 1);

    poly_ctx_reset_counters(ctx);
    ASSERT_INT_EQ(poly_jit_end_capture(jit, ctx->tensors, ctx->n_tensors), 0);
    ASSERT_TRUE(poly_jit_is_captured(jit));
    PolyUOp *captured = poly_jit_captured_linear(jit);
    ASSERT_NOT_NULL(captured);
    ASSERT_INT_EQ(captured->op, POLY_OP_LINEAR);
    ASSERT_INT_EQ(captured->n_src, 4);
    ASSERT_INT_EQ(count_shaped_value_params(ctx, captured), 6);
    ASSERT_FALSE(poly_uop_reachable(ctx, captured, capture_buffer));
    PolyUOp *final_call = captured->src[3];
    ASSERT_NOT_NULL(final_call);
    ASSERT_INT_EQ(final_call->op, POLY_OP_CALL);
    ASSERT_INT_EQ(final_call->n_src, 3);
    ASSERT_INT_EQ(final_call->src[1]->op, POLY_OP_BUFFER);
    ASSERT_INT_EQ(final_call->src[2]->op, POLY_OP_MSTACK);
    ASSERT_INT_EQ(final_call->src[2]->n_src, 2);
    ASSERT_INT_EQ(final_call->src[2]->src[0]->op, POLY_OP_BITCAST);
    ASSERT_INT_EQ(final_call->src[2]->src[1]->op, POLY_OP_BITCAST);
    ASSERT_INT_EQ(ctx->kernel_count, 5);

    PolyBuffer *multi = poly_uop_buffer_handle(ctx, poly_tensor_uop_physical(out));
    ASSERT_NOT_NULL(multi);
    ASSERT_TRUE(poly_buffer_is_multi(multi));
    ASSERT_INT_EQ(multi->n_bufs, 2);
    const char *devices[] = {"CPU", "CPU:1"};
    for (int lane = 0; lane < 2; lane++) {
      PolyBuffer *child = poly_buffer_multi_child(multi, lane);
      ASSERT_NOT_NULL(child);
      ASSERT_TRUE(child->valid);
      ASSERT_NOT_NULL(child->ptr);
      ASSERT_STR_EQ(child->device_uop->arg.str, devices[lane]);
      for (int i = 0; i < 4; i++)
        ASSERT_INT_EQ(((int32_t *)child->ptr)[i], 8 + lane * 4 + i);
    }

    PolyTensor *replay_input = initialized_i32_tensor8(ctx, 16, NULL);
    ASSERT_NOT_NULL(replay_input);
    poly_ctx_reset_counters(ctx);
    ASSERT_INT_EQ(poly_jit_run(jit, &replay_input, 1), 0);
    ASSERT_INT_EQ(ctx->kernel_count, 5);
    for (int lane = 0; lane < 2; lane++) {
      PolyBuffer *child = poly_buffer_multi_child(multi, lane);
      ASSERT_NOT_NULL(child);
      ASSERT_TRUE(child->valid);
      ASSERT_STR_EQ(child->device_uop->arg.str, devices[lane]);
      for (int i = 0; i < 4; i++)
        ASSERT_INT_EQ(((int32_t *)child->ptr)[i], 16 + lane * 4 + i);
    }

    poly_jit_free(jit);
    poly_ctx_destroy(ctx);
  }
  PASS();
}

TEST(realize, poly_jit_capture_skips_already_current_tensor) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  float data[3] = {1.0f, 2.0f, 3.0f};
  int64_t shape[] = {3};
  PolyUOp *buf = NULL;
  PolyTensor *input = initialized_f32_tensor(ctx, shape, 1, data, POLY_DEVICE_CPU, &buf);
  ASSERT_NOT_NULL(input);
  ASSERT_TRUE(poly_buffer_is_allocated(ctx, buf));

  PolyJit *jit = poly_jit_new(ctx);
  ASSERT_NOT_NULL(jit);
  ASSERT_INT_EQ(poly_jit_begin_capture(jit, &input, 1), 0);
  poly_ctx_reset_counters(ctx);

  PolyTensor *realized = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &input, 1, &realized), 0);
  ASSERT_PTR_EQ(realized, input);
  ASSERT_PTR_EQ(poly_tensor_uop(input), buf);
  ASSERT_INT_EQ(poly_jit_schedule_count(jit), 0);
  PolyCtxStats stats = {0};
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &stats), 0);
  ASSERT_TRUE(stats.global_ops == 0);
  ASSERT_TRUE(stats.global_mem == 0);
  ASSERT_TRUE(stats.kernel_count == 0);
  ASSERT_TRUE(poly_buffer_is_allocated(ctx, buf));

  poly_jit_cancel_capture(jit);
  poly_jit_free(jit);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, poly_jit_nested_capture_aborts_without_stranding_context) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  float data[3] = {1.0f, 2.0f, 3.0f};
  int64_t shape[] = {3};
  int f32 = poly_dtype_id_by_name("float32");
  PolyTensor *input = initialized_f32_tensor(ctx, shape, 1, data, POLY_DEVICE_CPU, NULL);
  ASSERT_NOT_NULL(input);

  PolyJit *outer = poly_jit_new(ctx);
  PolyJit *other = poly_jit_new(ctx);
  ASSERT_NOT_NULL(outer);
  ASSERT_NOT_NULL(other);
  ASSERT_INT_EQ(poly_jit_begin_capture(outer, &input, 1), 0);

  PolyTensor *one = poly_tensor_const_float_by_id(ctx, 1.0, f32, POLY_DEVICE_CPU);
  PolyTensor *out = poly_tensor_alu2(ctx, POLY_OP_ADD, input, one);
  ASSERT_NOT_NULL(out);
  PolyTensor *realized = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &out, 1, &realized), 0);
  ASSERT_PTR_EQ(realized, out);
  ASSERT_INT_EQ(poly_jit_schedule_count(outer), 1);

  /* Pinned TinyJit rejects before admitting a nested capture and its finally
   * block clears capture state (tinygrad/engine/jit.py:278-284). */
  ASSERT_INT_EQ(poly_jit_begin_capture(outer, &input, 1), -1);
  ASSERT_FALSE(poly_jit_is_capturing(outer));
  ASSERT_INT_EQ(poly_jit_schedule_count(outer), 0);
  ASSERT_INT_EQ(poly_jit_begin_capture(other, &input, 1), 0);
  ASSERT_TRUE(poly_jit_is_capturing(other));
  poly_jit_cancel_capture(other);

  poly_jit_free(other);
  poly_jit_free(outer);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, poly_jit_uses_movement_base_as_input_buffer) {
  PolyCtx *ctx = poly_ctx_new();

  float a_data[5] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  int64_t shape[] = {5};
  int f32 = poly_dtype_id_by_name("float32");
  PolyUOp *a_buf = NULL;
  PolyTensor *a_base = initialized_f32_tensor(ctx, shape, 1, a_data, POLY_DEVICE_CPU, &a_buf);
  int64_t a_bounds[1][2] = {{1, 4}};
  PolyTensor *a = poly_tensor_shrink(ctx, a_base, a_bounds, 1);
  ASSERT_NOT_NULL(a);
  ASSERT_TRUE(poly_uop_get_buffer_identity(poly_tensor_uop_physical(a)) == NULL);

  PolyJit *jit = poly_jit_new(ctx);
  ASSERT_NOT_NULL(jit);
  ASSERT_INT_EQ(poly_jit_begin_capture(jit, &a, 1), 0);

  PolyTensor *one = poly_tensor_const_float_by_id(ctx, 1.0, f32, POLY_DEVICE_CPU);
  PolyTensor *out = poly_tensor_alu2(ctx, POLY_OP_ADD, a, one);
  ASSERT_NOT_NULL(out);
  PolyTensor *realized = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &out, 1, &realized), 0);
  ASSERT_PTR_EQ(realized, out);
  ASSERT_INT_EQ(poly_jit_end_capture(jit, ctx->tensors, ctx->n_tensors), 0);

  float b_data[5] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f};
  PolyTensor *b_base = initialized_f32_tensor(ctx, shape, 1, b_data, POLY_DEVICE_CPU, NULL);
  int64_t b_bounds[1][2] = {{1, 4}};
  PolyTensor *b = poly_tensor_shrink(ctx, b_base, b_bounds, 1);
  ASSERT_NOT_NULL(b);
  ASSERT_INT_EQ(poly_jit_run(jit, &b, 1), 0);

  float got[3] = {0};
  ASSERT_INT_EQ(
      poly_buffer_read(
          ctx, (PolyUOp *)poly_uop_get_buffer_identity(poly_tensor_uop(out)), got, sizeof(got)
      ),
      0
  );
  ASSERT_FLOAT_EQ(got[0], 21.0f, 1e-5f);
  ASSERT_FLOAT_EQ(got[1], 31.0f, 1e-5f);
  ASSERT_FLOAT_EQ(got[2], 41.0f, 1e-5f);

  poly_jit_free(jit);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, poly_jit_replays_symbolic_input_view_with_current_bind) {
  PolyCtx *ctx = poly_ctx_new();

  float input_data[6] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  int f32 = poly_dtype_id_by_name("float32");
  int64_t input_shape[] = {6};
  PolyTensor *input =
      initialized_f32_tensor(ctx, input_shape, 1, input_data, POLY_DEVICE_CPU, NULL);

  PolyUOp *i = poly_uop_variable(ctx, "i", 0, 4, POLY_WEAKINT, 1, false);
  PolyUOp *size_2 = poly_const_int(ctx, 2);
  PolyUOp *capture_start = poly_uop_bind(ctx, i, 2);
  PolyUOp *capture_starts[] = {capture_start};
  PolyUOp *capture_sizes[] = {size_2};
  PolyTensor *capture_input = poly_tensor_shrink_uop(ctx, input, capture_starts, capture_sizes, 1);
  ASSERT_NOT_NULL(capture_input);
  PolyUOp *capture_view = poly_tensor_uop_physical(capture_input);
  ASSERT_INT_EQ(capture_view->op, POLY_OP_SHRINK);
  ASSERT_INT_EQ(count_bound_vars(ctx, capture_view), 1);

  PolyJit *jit = poly_jit_new(ctx);
  ASSERT_NOT_NULL(jit);
  ASSERT_INT_EQ(poly_jit_begin_capture(jit, &capture_input, 1), 0);
  PolyTensor *hundred = poly_tensor_const_float_by_id(ctx, 100.0, f32, POLY_DEVICE_CPU);
  PolyTensor *out = poly_tensor_alu2(ctx, POLY_OP_ADD, capture_input, hundred);
  ASSERT_NOT_NULL(out);
  PolyTensor *realized = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &out, 1, &realized), 0);
  ASSERT_PTR_EQ(realized, out);
  ASSERT_INT_EQ(poly_jit_end_capture(jit, ctx->tensors, ctx->n_tensors), 0);
  ASSERT_TRUE(poly_jit_is_captured(jit));
  ASSERT_INT_EQ(count_bound_vars(ctx, capture_view), 1);

  PolyUOp *out_buf = (PolyUOp *)poly_uop_get_buffer_identity(poly_tensor_uop(out));
  ASSERT_NOT_NULL(out_buf);
  float got[2] = {0};
  ASSERT_INT_EQ(poly_buffer_read(ctx, out_buf, got, sizeof(got)), 0);
  ASSERT_FLOAT_EQ(got[0], 102.0f, 1e-5f);
  ASSERT_FLOAT_EQ(got[1], 103.0f, 1e-5f);

  PolyUOp *replay_start_4 = poly_uop_bind(ctx, i, 4);
  PolyUOp *replay_starts_4[] = {replay_start_4};
  PolyTensor *replay_input_4 =
      poly_tensor_shrink_uop(ctx, input, replay_starts_4, capture_sizes, 1);
  ASSERT_NOT_NULL(replay_input_4);
  ASSERT_INT_EQ(count_bound_vars(ctx, poly_tensor_uop_physical(replay_input_4)), 1);
  ASSERT_INT_EQ(poly_jit_run(jit, &replay_input_4, 1), 0);
  ASSERT_INT_EQ(poly_buffer_read(ctx, out_buf, got, sizeof(got)), 0);
  ASSERT_FLOAT_EQ(got[0], 104.0f, 1e-5f);
  ASSERT_FLOAT_EQ(got[1], 105.0f, 1e-5f);

  PolyUOp *replay_start_0 = poly_uop_bind(ctx, i, 0);
  PolyUOp *replay_starts_0[] = {replay_start_0};
  PolyTensor *replay_input_0 =
      poly_tensor_shrink_uop(ctx, input, replay_starts_0, capture_sizes, 1);
  ASSERT_NOT_NULL(replay_input_0);
  ASSERT_INT_EQ(poly_jit_run(jit, &replay_input_0, 1), 0);
  ASSERT_INT_EQ(poly_buffer_read(ctx, out_buf, got, sizeof(got)), 0);
  ASSERT_FLOAT_EQ(got[0], 100.0f, 1e-5f);
  ASSERT_FLOAT_EQ(got[1], 101.0f, 1e-5f);

  /* Pinned expected_input_info includes the unbound input view, so changing
   * the fixed slice size is an input-signature mismatch, not a new binding. */
  PolyUOp *size_3 = poly_const_int(ctx, 3);
  PolyUOp *mismatch_sizes[] = {size_3};
  PolyTensor *mismatch_input =
      poly_tensor_shrink_uop(ctx, input, replay_starts_0, mismatch_sizes, 1);
  ASSERT_NOT_NULL(mismatch_input);
  ASSERT_INT_EQ(poly_jit_run(jit, &mismatch_input, 1), -1);

  poly_jit_free(jit);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, poly_jit_captures_custom_call_and_replays_after_input_mutation) {
  PolyCtx *ctx = poly_ctx_new();

  float a_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float b_data[4] = {10.0f, 20.0f, 30.0f, 40.0f};
  int64_t shape[] = {4};
  PolyUOp *a_buf = NULL;
  PolyUOp *b_buf = NULL;
  PolyTensor *a = initialized_f32_tensor(ctx, shape, 1, a_data, POLY_DEVICE_CPU, &a_buf);
  PolyTensor *b = initialized_f32_tensor(ctx, shape, 1, b_data, POLY_DEVICE_CPU, &b_buf);
  ASSERT_NOT_NULL(a);
  ASSERT_NOT_NULL(b);

  PolyJit *jit = poly_jit_new(ctx);
  ASSERT_NOT_NULL(jit);
  PolyTensor *inputs[2] = {a, b};
  ASSERT_INT_EQ(poly_jit_begin_capture(jit, inputs, 2), 0);

  PolyUOp *out_buf = NULL;
  PolyTensor *out = custom_add_tensor(ctx, a, b, &out_buf);
  ASSERT_NOT_NULL(out);
  ASSERT_NOT_NULL(out_buf);
  PolyUOp *logical_root = poly_tensor_uop_logical(out);
  PolyUOp *physical_root = poly_tensor_uop(out);
  ASSERT_INT_EQ(logical_root->op, POLY_OP_AFTER);
  ASSERT_INT_EQ(physical_root->op, POLY_OP_AFTER);
  ASSERT_TRUE(logical_root != physical_root);
  ASSERT_INT_EQ(poly_uop_device(logical_root->src[0]), POLY_DEVICE_AUTO);
  ASSERT_PTR_EQ(physical_root->src[0], out_buf);
  ASSERT_INT_EQ(poly_uop_device(out_buf), POLY_DEVICE_CPU);
  ASSERT_INT_EQ(logical_root->src[1]->op, POLY_OP_CALL);
  ASSERT_INT_EQ(physical_root->src[1]->op, POLY_OP_CALL);
  ASSERT_PTR_EQ(logical_root->src[1]->src[0], physical_root->src[1]->src[0]);
  ASSERT_PTR_EQ(physical_root->src[1]->src[1], out_buf);
  PolyTensor *realized = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &out, 1, &realized), 0);
  ASSERT_PTR_EQ(realized, out);
  ASSERT_INT_EQ(poly_jit_schedule_count(jit), 1);
  PolyTensor *live[3] = {a, b, out};
  ASSERT_INT_EQ(poly_jit_end_capture(jit, live, 3), 0);
  ASSERT_TRUE(poly_jit_is_captured(jit));
  ASSERT_INT_EQ(poly_jit_schedule_count(jit), 1);

  float got[4] = {0};
  ASSERT_INT_EQ(poly_buffer_read(ctx, out_buf, got, sizeof(got)), 0);
  ASSERT_FLOAT_EQ(got[0], 11.0f, 1e-5f);
  ASSERT_FLOAT_EQ(got[1], 22.0f, 1e-5f);
  ASSERT_FLOAT_EQ(got[2], 33.0f, 1e-5f);
  ASSERT_FLOAT_EQ(got[3], 44.0f, 1e-5f);

  float a_update[4] = {5.0f, 6.0f, 7.0f, 8.0f};
  ASSERT_INT_EQ(poly_buffer_write(ctx, a_buf, a_update, sizeof(a_update)), 0);
  ASSERT_INT_EQ(poly_jit_run(jit, inputs, 2), 0);
  memset(got, 0, sizeof(got));
  ASSERT_INT_EQ(poly_buffer_read(ctx, out_buf, got, sizeof(got)), 0);
  ASSERT_FLOAT_EQ(got[0], 15.0f, 1e-5f);
  ASSERT_FLOAT_EQ(got[1], 26.0f, 1e-5f);
  ASSERT_FLOAT_EQ(got[2], 37.0f, 1e-5f);
  ASSERT_FLOAT_EQ(got[3], 48.0f, 1e-5f);

  poly_jit_free(jit);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, poly_jit_captures_custom_reduction_call_and_replays_after_input_mutation) {
  PolyCtx *ctx = poly_ctx_new();

  float a_data[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  int64_t a_shape[] = {8};
  PolyUOp *a_buf = NULL;
  PolyTensor *a = initialized_f32_tensor(ctx, a_shape, 1, a_data, POLY_DEVICE_CPU, &a_buf);
  ASSERT_NOT_NULL(a);

  PolyJit *jit = poly_jit_new(ctx);
  ASSERT_NOT_NULL(jit);
  ASSERT_INT_EQ(poly_jit_begin_capture(jit, &a, 1), 0);

  PolyUOp *out_buf = NULL;
  PolyTensor *out = custom_summary_tensor(ctx, a, &out_buf);
  ASSERT_NOT_NULL(out);
  ASSERT_NOT_NULL(out_buf);
  PolyUOp *logical_root = poly_tensor_uop_logical(out);
  PolyUOp *physical_root = poly_tensor_uop(out);
  ASSERT_INT_EQ(logical_root->op, POLY_OP_AFTER);
  ASSERT_INT_EQ(physical_root->op, POLY_OP_AFTER);
  ASSERT_TRUE(logical_root != physical_root);
  ASSERT_INT_EQ(poly_uop_device(logical_root->src[0]), POLY_DEVICE_AUTO);
  ASSERT_PTR_EQ(physical_root->src[0], out_buf);
  ASSERT_INT_EQ(poly_uop_device(out_buf), POLY_DEVICE_CPU);
  ASSERT_PTR_EQ(logical_root->src[1]->src[0], physical_root->src[1]->src[0]);
  ASSERT_PTR_EQ(physical_root->src[1]->src[1], out_buf);
  PolyTensor *realized = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &out, 1, &realized), 0);
  ASSERT_PTR_EQ(realized, out);
  ASSERT_INT_EQ(poly_jit_schedule_count(jit), 1);
  PolyTensor *live[2] = {a, out};
  PolyUOp **held = NULL;
  int n_held = 0;
  ASSERT_INT_EQ(poly_jit_collect_held_bufs(ctx, live, 2, &held, &n_held), 0);
  bool holds_out = false;
  for (int i = 0; i < n_held; i++)
    holds_out |= held[i] == out_buf;
  free(held);
  ASSERT_TRUE(holds_out);
  ASSERT_INT_EQ(poly_jit_end_capture(jit, live, 2), 0);
  ASSERT_TRUE(poly_jit_is_captured(jit));
  PolyUOp *captured = poly_jit_captured_linear(jit);
  ASSERT_NOT_NULL(captured);
  ASSERT_INT_EQ(captured->op, POLY_OP_LINEAR);
  ASSERT_INT_EQ(captured->n_src, 1);
  PolyUOp *captured_call = captured->src[0];
  ASSERT_NOT_NULL(captured_call);
  ASSERT_INT_EQ(captured_call->op, POLY_OP_CALL);
  ASSERT_INT_EQ(captured_call->src[0]->op, POLY_OP_PROGRAM);
  const PolyProgramInfo *info = poly_program_info(ctx, captured_call->src[0]);
  ASSERT_NOT_NULL(info);
  ASSERT_INT_EQ(info->n_globals, 2);
  ASSERT_INT_EQ(info->n_outs, 1);
  ASSERT_INT_EQ(info->outs[0], 0);
  ASSERT_INT_EQ(info->n_ins, 2);
  ASSERT_INT_EQ(info->ins[0], 0);
  ASSERT_INT_EQ(info->ins[1], 1);

  float got[1] = {0};
  ASSERT_INT_EQ(poly_buffer_read(ctx, out_buf, got, sizeof(got)), 0);
  ASSERT_FLOAT_EQ(got[0], 36.0f, 1e-5f);

  float a_update[8] = {2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
  ASSERT_INT_EQ(poly_buffer_write(ctx, a_buf, a_update, sizeof(a_update)), 0);
  ASSERT_INT_EQ(poly_jit_run(jit, &a, 1), 0);
  memset(got, 0, sizeof(got));
  ASSERT_INT_EQ(poly_buffer_read(ctx, out_buf, got, sizeof(got)), 0);
  ASSERT_FLOAT_EQ(got[0], 44.0f, 1e-5f);

  poly_jit_free(jit);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, poly_jit_captures_custom_multi_output_call_and_replays_after_input_mutation) {
  PolyCtx *ctx = poly_ctx_new();

  float a_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float b_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  int64_t shape[] = {4};
  PolyUOp *a_buf = NULL;
  PolyUOp *b_buf = NULL;
  PolyTensor *a = initialized_f32_tensor(ctx, shape, 1, a_data, POLY_DEVICE_CPU, &a_buf);
  PolyTensor *b = initialized_f32_tensor(ctx, shape, 1, b_data, POLY_DEVICE_CPU, &b_buf);
  ASSERT_NOT_NULL(a);
  ASSERT_NOT_NULL(b);

  PolyJit *jit = poly_jit_new(ctx);
  ASSERT_NOT_NULL(jit);
  PolyTensor *inputs[2] = {a, b};
  ASSERT_INT_EQ(poly_jit_begin_capture(jit, inputs, 2), 0);

  PolyTensor *out0 = NULL;
  PolyTensor *out1 = NULL;
  PolyUOp *out0_buf = NULL;
  PolyUOp *out1_buf = NULL;
  custom_multi_summary_tensors(ctx, a, b, &out0, &out1, &out0_buf, &out1_buf);
  ASSERT_NOT_NULL(out0);
  ASSERT_NOT_NULL(out1);
  PolyUOp *logical0 = poly_tensor_uop_logical(out0);
  PolyUOp *logical1 = poly_tensor_uop_logical(out1);
  PolyUOp *physical0 = poly_tensor_uop(out0);
  PolyUOp *physical1 = poly_tensor_uop(out1);
  ASSERT_INT_EQ(logical0->op, POLY_OP_AFTER);
  ASSERT_INT_EQ(logical1->op, POLY_OP_AFTER);
  ASSERT_INT_EQ(physical0->op, POLY_OP_AFTER);
  ASSERT_INT_EQ(physical1->op, POLY_OP_AFTER);
  ASSERT_PTR_EQ(logical0->src[1], logical1->src[1]);
  ASSERT_PTR_EQ(physical0->src[1], physical1->src[1]);
  ASSERT_PTR_EQ(logical0->src[1]->src[0], physical0->src[1]->src[0]);
  ASSERT_PTR_EQ(physical0->src[0], out0_buf);
  ASSERT_PTR_EQ(physical1->src[0], out1_buf);
  ASSERT_PTR_EQ(physical0->src[1]->src[1], out0_buf);
  ASSERT_PTR_EQ(physical0->src[1]->src[2], out1_buf);
  ASSERT_INT_EQ(poly_uop_device(out0_buf), POLY_DEVICE_CPU);
  ASSERT_INT_EQ(poly_uop_device(out1_buf), POLY_DEVICE_CPU);
  PolyTensor *outs[2] = {out0, out1};
  PolyTensor *realized[2] = {NULL, NULL};
  ASSERT_INT_EQ(poly_realize_tensors(ctx, outs, 2, realized), 0);
  ASSERT_PTR_EQ(realized[0], out0);
  ASSERT_PTR_EQ(realized[1], out1);
  PolyTensor *live[4] = {a, b, out0, out1};
  ASSERT_INT_EQ(poly_jit_end_capture(jit, live, 4), 0);
  ASSERT_TRUE(poly_jit_is_captured(jit));

  float got0[4] = {0};
  float got1[4] = {0};
  ASSERT_INT_EQ(poly_buffer_read(ctx, out0_buf, got0, sizeof(got0)), 0);
  ASSERT_INT_EQ(poly_buffer_read(ctx, out1_buf, got1, sizeof(got1)), 0);
  float expected0[4] = {2.0f, 4.0f, 6.0f, 8.0f};
  float expected1[4] = {1.0f, 4.0f, 9.0f, 16.0f};
  for (int i = 0; i < 4; i++) {
    ASSERT_FLOAT_EQ(got0[i], expected0[i], 1e-5f);
    ASSERT_FLOAT_EQ(got1[i], expected1[i], 1e-5f);
  }

  float a_update[4] = {2.0f, 3.0f, 4.0f, 5.0f};
  ASSERT_INT_EQ(poly_buffer_write(ctx, a_buf, a_update, sizeof(a_update)), 0);
  ASSERT_INT_EQ(poly_jit_run(jit, inputs, 2), 0);
  memset(got0, 0, sizeof(got0));
  memset(got1, 0, sizeof(got1));
  ASSERT_INT_EQ(poly_buffer_read(ctx, out0_buf, got0, sizeof(got0)), 0);
  ASSERT_INT_EQ(poly_buffer_read(ctx, out1_buf, got1, sizeof(got1)), 0);
  float replay_expected0[4] = {3.0f, 5.0f, 7.0f, 9.0f};
  float replay_expected1[4] = {2.0f, 6.0f, 12.0f, 20.0f};
  for (int i = 0; i < 4; i++) {
    ASSERT_FLOAT_EQ(got0[i], replay_expected0[i], 1e-5f);
    ASSERT_FLOAT_EQ(got1[i], replay_expected1[i], 1e-5f);
  }

  poly_jit_free(jit);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, poly_jit_replays_assign_against_current_input) {
  PolyCtx *ctx = poly_ctx_new();

  float a_data[3] = {1.0f, 2.0f, 3.0f};
  int64_t shape[] = {3};
  int f32 = poly_dtype_id_by_name("float32");
  PolyUOp *a_buf = NULL;
  PolyTensor *a = initialized_f32_tensor(ctx, shape, 1, a_data, POLY_DEVICE_CPU, &a_buf);
  ASSERT_NOT_NULL(a);

  PolyJit *jit = poly_jit_new(ctx);
  ASSERT_NOT_NULL(jit);
  ASSERT_INT_EQ(poly_jit_begin_capture(jit, &a, 1), 0);

  PolyTensor *one = poly_tensor_const_float_by_id(ctx, 1.0, f32, POLY_DEVICE_CPU);
  PolyTensor *value = poly_tensor_alu2(ctx, POLY_OP_ADD, a, one);
  ASSERT_NOT_NULL(value);
  ASSERT_PTR_EQ(poly_tensor_assign(ctx, a, value), a);
  PolyTensor *realized = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &a, 1, &realized), 0);
  ASSERT_PTR_EQ(realized, a);
  ASSERT_INT_EQ(poly_jit_end_capture(jit, ctx->tensors, ctx->n_tensors), 0);

  float captured_after[3] = {0};
  ASSERT_INT_EQ(poly_buffer_read(ctx, a_buf, captured_after, sizeof(captured_after)), 0);
  ASSERT_FLOAT_EQ(captured_after[0], 2.0f, 1e-5f);
  ASSERT_FLOAT_EQ(captured_after[1], 3.0f, 1e-5f);
  ASSERT_FLOAT_EQ(captured_after[2], 4.0f, 1e-5f);

  float b_data[3] = {100.0f, 200.0f, 300.0f};
  PolyUOp *b_buf = NULL;
  PolyTensor *b = initialized_f32_tensor(ctx, shape, 1, b_data, POLY_DEVICE_CPU, &b_buf);
  ASSERT_NOT_NULL(b);

  ASSERT_INT_EQ(poly_jit_run(jit, &b, 1), 0);

  float captured_still[3] = {0};
  ASSERT_INT_EQ(poly_buffer_read(ctx, a_buf, captured_still, sizeof(captured_still)), 0);
  ASSERT_FLOAT_EQ(captured_still[0], 2.0f, 1e-5f);
  ASSERT_FLOAT_EQ(captured_still[1], 3.0f, 1e-5f);
  ASSERT_FLOAT_EQ(captured_still[2], 4.0f, 1e-5f);

  float replay_input[3] = {0};
  ASSERT_INT_EQ(poly_buffer_read(ctx, b_buf, replay_input, sizeof(replay_input)), 0);
  ASSERT_FLOAT_EQ(replay_input[0], 101.0f, 1e-5f);
  ASSERT_FLOAT_EQ(replay_input[1], 201.0f, 1e-5f);
  ASSERT_FLOAT_EQ(replay_input[2], 301.0f, 1e-5f);

  poly_jit_free(jit);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, poly_jit_replay_uses_replaced_buffer_binding) {
  PolyCtx *ctx = poly_ctx_new();
  int64_t shape[] = {3};
  float original[3] = {1, 2, 3}, replacement[3] = {10, 20, 30};
  PolyTensor *input = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_CPU);
  PolyUOp *buffer = poly_uop_buffer(ctx, poly_tensor_uop_physical(input));
  ASSERT_NOT_NULL(buffer);
  ASSERT_INT_EQ(poly_buffer_set(ctx, buffer, original, sizeof(original), POLY_DEVICE_CPU), 0);
  PolyJit *jit = poly_jit_new(ctx);
  ASSERT_INT_EQ(poly_jit_begin_capture(jit, &input, 1), 0);
  PolyTensor *one =
      poly_tensor_const_float_by_id(ctx, 1.0, poly_dtype_id_by_name("float32"), POLY_DEVICE_CPU);
  PolyTensor *sum = poly_tensor_alu2(ctx, POLY_OP_ADD, input, one);
  ASSERT_PTR_EQ(poly_tensor_assign(ctx, input, sum), input);
  PolyTensor *realized = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &input, 1, &realized), 0);
  ASSERT_INT_EQ(poly_jit_end_capture(jit, ctx->tensors, ctx->n_tensors), 0);
  ASSERT_INT_EQ(poly_buffer_set(ctx, buffer, replacement, sizeof(replacement), POLY_DEVICE_CPU), 0);
  ASSERT_INT_EQ(poly_jit_run(jit, &input, 1), 0);
  for (int i = 0; i < 3; i++) {
    ASSERT_FLOAT_EQ(original[i], i + 2, 1e-6);
    ASSERT_FLOAT_EQ(replacement[i], (i + 1) * 10 + 1, 1e-6);
  }
  poly_jit_free(jit);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, poly_jit_combines_multiple_captured_realizes) {
  PolyCtx *ctx = poly_ctx_new();

  float a_data[3] = {1.0f, 2.0f, 3.0f};
  int64_t shape[] = {3};
  int f32 = poly_dtype_id_by_name("float32");
  PolyTensor *a = initialized_f32_tensor(ctx, shape, 1, a_data, POLY_DEVICE_CPU, NULL);
  ASSERT_NOT_NULL(a);

  PolyJit *jit = poly_jit_new(ctx);
  ASSERT_NOT_NULL(jit);
  ASSERT_INT_EQ(poly_jit_begin_capture(jit, &a, 1), 0);

  PolyTensor *one = poly_tensor_const_float_by_id(ctx, 1.0, f32, POLY_DEVICE_CPU);
  PolyTensor *y = poly_tensor_alu2(ctx, POLY_OP_ADD, a, one);
  ASSERT_NOT_NULL(y);
  PolyTensor *realized_y = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &y, 1, &realized_y), 0);

  PolyTensor *two = poly_tensor_const_float_by_id(ctx, 2.0, f32, POLY_DEVICE_CPU);
  PolyTensor *z = poly_tensor_alu2(ctx, POLY_OP_MUL, a, two);
  ASSERT_NOT_NULL(z);
  PolyTensor *realized_z = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &z, 1, &realized_z), 0);

  ASSERT_INT_EQ(poly_jit_end_capture(jit, ctx->tensors, ctx->n_tensors), 0);
  ASSERT_INT_EQ(poly_jit_schedule_count(jit), 2);

  float b_data[3] = {10.0f, 20.0f, 30.0f};
  PolyUOp *b_buf = NULL;
  PolyTensor *b = initialized_f32_tensor(ctx, shape, 1, b_data, POLY_DEVICE_CPU, &b_buf);
  ASSERT_NOT_NULL(b);

  ASSERT_INT_EQ(poly_jit_run(jit, &b, 1), 0);

  float y_out[3] = {0};
  ASSERT_INT_EQ(
      poly_buffer_read(
          ctx, (PolyUOp *)poly_uop_get_buffer_identity(poly_tensor_uop(y)), y_out, sizeof(y_out)
      ),
      0
  );
  ASSERT_FLOAT_EQ(y_out[0], 11.0f, 1e-5f);
  ASSERT_FLOAT_EQ(y_out[1], 21.0f, 1e-5f);
  ASSERT_FLOAT_EQ(y_out[2], 31.0f, 1e-5f);

  float z_out[3] = {0};
  ASSERT_INT_EQ(
      poly_buffer_read(
          ctx, (PolyUOp *)poly_uop_get_buffer_identity(poly_tensor_uop(z)), z_out, sizeof(z_out)
      ),
      0
  );
  ASSERT_FLOAT_EQ(z_out[0], 20.0f, 1e-5f);
  ASSERT_FLOAT_EQ(z_out[1], 40.0f, 1e-5f);
  ASSERT_FLOAT_EQ(z_out[2], 60.0f, 1e-5f);

  float replay_input[3] = {0};
  ASSERT_INT_EQ(poly_buffer_read(ctx, b_buf, replay_input, sizeof(replay_input)), 0);
  ASSERT_FLOAT_EQ(replay_input[0], 10.0f, 1e-5f);
  ASSERT_FLOAT_EQ(replay_input[1], 20.0f, 1e-5f);
  ASSERT_FLOAT_EQ(replay_input[2], 30.0f, 1e-5f);

  poly_jit_free(jit);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, poly_jit_prune_skips_onetime_side_realize_on_replay) {
  PolyCtx *ctx = poly_ctx_new();

  float a_data[3] = {1.0f, 2.0f, 3.0f};
  int64_t shape[] = {3};
  int f32 = poly_dtype_id_by_name("float32");
  PolyTensor *a = initialized_f32_tensor(ctx, shape, 1, a_data, POLY_DEVICE_CPU, NULL);
  ASSERT_NOT_NULL(a);

  float side_data[3] = {-1.0f, -1.0f, -1.0f};
  float seed_data[3] = {7.0f, 8.0f, 9.0f};
  PolyUOp *side_buf = NULL;
  PolyTensor *side = initialized_f32_tensor(ctx, shape, 1, side_data, POLY_DEVICE_CPU, &side_buf);
  PolyTensor *seed = initialized_f32_tensor(ctx, shape, 1, seed_data, POLY_DEVICE_CPU, NULL);
  ASSERT_NOT_NULL(side);
  ASSERT_NOT_NULL(seed);

  PolyJit *jit = poly_jit_new(ctx);
  ASSERT_NOT_NULL(jit);
  ASSERT_INT_EQ(poly_jit_set_prune(jit, true), 0);
  ASSERT_INT_EQ(poly_jit_begin_capture(jit, &a, 1), 0);

  ASSERT_PTR_EQ(poly_tensor_assign(ctx, side, seed), side);
  PolyTensor *realized_side = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &side, 1, &realized_side), 0);
  ASSERT_PTR_EQ(realized_side, side);

  PolyTensor *one = poly_tensor_const_float_by_id(ctx, 1.0, f32, POLY_DEVICE_CPU);
  PolyTensor *out = poly_tensor_alu2(ctx, POLY_OP_ADD, a, one);
  ASSERT_NOT_NULL(out);
  PolyTensor *realized_out = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &out, 1, &realized_out), 0);
  ASSERT_INT_EQ(poly_jit_end_capture(jit, ctx->tensors, ctx->n_tensors), 0);
  ASSERT_INT_EQ(poly_jit_schedule_count(jit), 2);

  float side_after_capture[3] = {0};
  ASSERT_INT_EQ(poly_buffer_read(ctx, side_buf, side_after_capture, sizeof(side_after_capture)), 0);
  ASSERT_FLOAT_EQ(side_after_capture[0], 7.0f, 1e-5f);
  ASSERT_FLOAT_EQ(side_after_capture[1], 8.0f, 1e-5f);
  ASSERT_FLOAT_EQ(side_after_capture[2], 9.0f, 1e-5f);

  float reset_side[3] = {-9.0f, -9.0f, -9.0f};
  ASSERT_INT_EQ(poly_buffer_write(ctx, side_buf, reset_side, sizeof(reset_side)), 0);

  float b_data[3] = {100.0f, 200.0f, 300.0f};
  PolyTensor *b = initialized_f32_tensor(ctx, shape, 1, b_data, POLY_DEVICE_CPU, NULL);
  ASSERT_NOT_NULL(b);
  ASSERT_INT_EQ(poly_jit_run(jit, &b, 1), 0);

  float side_after_replay[3] = {0};
  ASSERT_INT_EQ(poly_buffer_read(ctx, side_buf, side_after_replay, sizeof(side_after_replay)), 0);
  ASSERT_FLOAT_EQ(side_after_replay[0], -9.0f, 1e-5f);
  ASSERT_FLOAT_EQ(side_after_replay[1], -9.0f, 1e-5f);
  ASSERT_FLOAT_EQ(side_after_replay[2], -9.0f, 1e-5f);

  float out_data[3] = {0};
  ASSERT_INT_EQ(
      poly_buffer_read(
          ctx, (PolyUOp *)poly_uop_get_buffer_identity(poly_tensor_uop(out)), out_data,
          sizeof(out_data)
      ),
      0
  );
  ASSERT_FLOAT_EQ(out_data[0], 101.0f, 1e-5f);
  ASSERT_FLOAT_EQ(out_data[1], 201.0f, 1e-5f);
  ASSERT_FLOAT_EQ(out_data[2], 301.0f, 1e-5f);

  poly_jit_free(jit);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, poly_jit_accepts_replay_buffer_size_mismatch_like_tinygrad) {
  PolyCtx *ctx = poly_ctx_new();

  float a_data[3] = {1.0f, 2.0f, 3.0f};
  int64_t a_shape[] = {3};
  int f32 = poly_dtype_id_by_name("float32");
  PolyTensor *a = initialized_f32_tensor(ctx, a_shape, 1, a_data, POLY_DEVICE_CPU, NULL);
  ASSERT_NOT_NULL(a);

  PolyJit *jit = poly_jit_new(ctx);
  ASSERT_NOT_NULL(jit);
  ASSERT_INT_EQ(poly_jit_begin_capture(jit, &a, 1), 0);
  PolyTensor *one = poly_tensor_const_float_by_id(ctx, 1.0, f32, POLY_DEVICE_CPU);
  PolyTensor *out = poly_tensor_alu2(ctx, POLY_OP_ADD, a, one);
  ASSERT_NOT_NULL(out);
  PolyTensor *realized = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &out, 1, &realized), 0);
  ASSERT_INT_EQ(poly_jit_end_capture(jit, ctx->tensors, ctx->n_tensors), 0);

  float b_data[4] = {10.0f, 20.0f, 30.0f, 40.0f};
  int64_t b_shape[] = {4};
  PolyTensor *b = initialized_f32_tensor(ctx, b_shape, 1, b_data, POLY_DEVICE_CPU, NULL);
  ASSERT_NOT_NULL(b);

  ASSERT_INT_EQ(poly_jit_run(jit, &b, 1), 0);

  float out_data[3] = {0};
  ASSERT_INT_EQ(
      poly_buffer_read(
          ctx, (PolyUOp *)poly_uop_get_buffer_identity(poly_tensor_uop(out)), out_data,
          sizeof(out_data)
      ),
      0
  );
  ASSERT_FLOAT_EQ(out_data[0], 11.0f, 1e-5f);
  ASSERT_FLOAT_EQ(out_data[1], 21.0f, 1e-5f);
  ASSERT_FLOAT_EQ(out_data[2], 31.0f, 1e-5f);

  poly_jit_free(jit);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, poly_jit_rejects_conflicting_runtime_var_override) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *N = poly_uop_variable(ctx, "N", 1, 16, POLY_WEAKINT, 1, false);
  PolyUOp *bind_N = poly_uop_bind(ctx, N, 4);
  ASSERT_NOT_NULL(N);
  ASSERT_NOT_NULL(bind_N);

  /* Pinned UOp.empty allocates the maximum deviceful BUFFER and represents a
   * symbolic runtime extent with movement, not extra BUFFER sources
   * (uop/ops.py:733-746). */
  PolyUOp *a_buf = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 16, POLY_DEVICE_CPU);
  PolyUOp *out_buf = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 16, POLY_DEVICE_CPU);
  PolyUOp *starts[1] = {poly_const_int(ctx, 0)};
  PolyUOp *sizes[1] = {bind_N};
  PolyUOp *a_view = poly_shrink_uop(ctx, a_buf, starts, sizes, 1);
  PolyUOp *out_view = poly_shrink_uop(ctx, out_buf, starts, sizes, 1);
  ASSERT_NOT_NULL(a_buf);
  ASSERT_NOT_NULL(out_buf);
  ASSERT_NOT_NULL(a_view);
  ASSERT_NOT_NULL(out_view);

  float a_data[16];
  float out_data[16];
  for (int i = 0; i < 16; i++) {
    a_data[i] = (float)(i + 1);
    out_data[i] = -999.0f;
  }
  poly_buffer_set(ctx, a_buf, a_data, sizeof(a_data), POLY_DEVICE_CPU);
  poly_buffer_set(ctx, out_buf, out_data, sizeof(out_data), POLY_DEVICE_CPU);

  PolyTensor *a =
      poly_tensor_create_with_roots(ctx, a_view, a_view, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(a);
  PolyJit *jit = poly_jit_new(ctx);
  ASSERT_NOT_NULL(jit);
  ASSERT_INT_EQ(poly_jit_begin_capture(jit, &a, 1), 0);

  PolyUOp *store = poly_store_val(
      ctx, out_view, poly_alu2(ctx, POLY_OP_ADD, poly_tensor_uop(a), poly_const_float(ctx, 1.0f))
  );
  PolyUOp *after_src[2] = {out_view, store};
  PolyUOp *assign = poly_uop(ctx, POLY_OP_AFTER, out_view->dtype, after_src, 2, poly_arg_none());
  PolyUOp *realized[1] = {NULL};
  ASSERT_INT_EQ(poly_realize_uops(ctx, &assign, 1, realized), 0);
  /* Pinned Tensor.realize retains the caller-visible SHRINK view over the
   * realized BUFFER; SHRINK itself has no buffer identity (tensor.py:202-218,
   * paired probe temp/tg_shrink_assign_realize_probe_20260810.py). */
  ASSERT_NOT_NULL(realized[0]);
  ASSERT_INT_EQ(realized[0]->op, POLY_OP_SHRINK);
  ASSERT_PTR_EQ(poly_uop_get_buffer_identity(realized[0]->src[0]), out_buf);
  ASSERT_INT_EQ(poly_jit_end_capture(jit, ctx->tensors, ctx->n_tensors), 0);
  ASSERT_INT_EQ(poly_buffer_read(ctx, out_buf, out_data, sizeof(out_data)), 0);

  for (int i = 0; i < 4; i++)
    ASSERT_FLOAT_EQ(out_data[i], (float)(i + 2), 1e-5f);
  ASSERT_FLOAT_EQ(out_data[4], -999.0f, 1e-5f);

  for (int i = 0; i < 16; i++)
    out_data[i] = -999.0f;
  ASSERT_INT_EQ(poly_buffer_write(ctx, out_buf, out_data, sizeof(out_data)), 0);
  ASSERT_INT_EQ(poly_jit_run(jit, &a, 1), 0);
  ASSERT_INT_EQ(poly_buffer_read(ctx, out_buf, out_data, sizeof(out_data)), 0);
  for (int i = 0; i < 4; i++)
    ASSERT_FLOAT_EQ(out_data[i], (float)(i + 2), 1e-5f);
  ASSERT_FLOAT_EQ(out_data[4], -999.0f, 1e-5f);

  for (int i = 0; i < 16; i++)
    out_data[i] = -999.0f;
  ASSERT_INT_EQ(poly_buffer_write(ctx, out_buf, out_data, sizeof(out_data)), 0);
  PolyVarBinding bind_4 = {.var = N, .value = 4};
  ASSERT_INT_EQ(poly_jit_run_with_vars(jit, &a, 1, &bind_4, 1), 0);
  ASSERT_INT_EQ(poly_buffer_read(ctx, out_buf, out_data, sizeof(out_data)), 0);
  for (int i = 0; i < 4; i++)
    ASSERT_FLOAT_EQ(out_data[i], (float)(i + 2), 1e-5f);
  ASSERT_FLOAT_EQ(out_data[4], -999.0f, 1e-5f);

  for (int i = 0; i < 16; i++)
    out_data[i] = -999.0f;
  ASSERT_INT_EQ(poly_buffer_write(ctx, out_buf, out_data, sizeof(out_data)), 0);
  PolyVarBinding bind_6 = {.var = N, .value = 6};
  /* Pinned _prepare_jit_inputs rejects conflicting bindings collected from
   * one input signature and explicit arguments (engine/jit.py:240-244). */
  ASSERT_INT_EQ(poly_jit_run_with_vars(jit, &a, 1, &bind_6, 1), -1);
  ASSERT_INT_EQ(poly_buffer_read(ctx, out_buf, out_data, sizeof(out_data)), 0);
  for (int i = 0; i < 16; i++)
    ASSERT_FLOAT_EQ(out_data[i], -999.0f, 1e-5f);

  poly_jit_free(jit);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, poly_jit_rejects_replay_dtype_mismatch) {
  PolyCtx *ctx = poly_ctx_new();

  float a_data[3] = {1.0f, 2.0f, 3.0f};
  int64_t shape[] = {3};
  int f32 = poly_dtype_id_by_name("float32");
  PolyTensor *a = initialized_f32_tensor(ctx, shape, 1, a_data, POLY_DEVICE_CPU, NULL);
  ASSERT_NOT_NULL(a);

  PolyJit *jit = poly_jit_new(ctx);
  ASSERT_NOT_NULL(jit);
  ASSERT_INT_EQ(poly_jit_begin_capture(jit, &a, 1), 0);
  PolyTensor *one = poly_tensor_const_float_by_id(ctx, 1.0, f32, POLY_DEVICE_CPU);
  PolyTensor *out = poly_tensor_alu2(ctx, POLY_OP_ADD, a, one);
  ASSERT_NOT_NULL(out);
  PolyTensor *realized = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &out, 1, &realized), 0);
  ASSERT_INT_EQ(poly_jit_end_capture(jit, ctx->tensors, ctx->n_tensors), 0);

  int32_t b_data[3] = {10, 20, 30};
  PolyTensor *b = poly_tensor_empty(ctx, POLY_INT32, shape, 1, POLY_DEVICE_CPU);
  PolyUOp *b_buf = b ? (PolyUOp *)poly_uop_get_buffer_identity(poly_tensor_uop_physical(b)) : NULL;
  ASSERT_NOT_NULL(b_buf);
  poly_buffer_set(ctx, b_buf, b_data, sizeof(b_data), POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(b);

  ASSERT_INT_EQ(poly_jit_run(jit, &b, 1), -1);

  poly_jit_free(jit);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, poly_jit_rejects_replay_device_identity_mismatch) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  int64_t shape[] = {3};
  float capture_data[3] = {1.0f, 2.0f, 3.0f};
  PolyTensor *capture = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(capture);
  PolyUOp *capture_logical = poly_tensor_uop_logical(capture);
  ASSERT_NOT_NULL(capture_logical);
  ASSERT_INT_EQ(capture_logical->op, POLY_OP_BUFFER);
  PolyUOp *cpu1 = poly_device_uop_from_name(ctx, "CPU:1");
  PolyUOp *capture_buffer =
      poly_uop_new_buffer(ctx, cpu1, 3, POLY_FLOAT32, capture_logical->src[0]->arg.i);
  ASSERT_NOT_NULL(capture_buffer);
  ASSERT_INT_EQ(
      poly_tensor_set_physical(ctx, capture, capture_buffer, POLY_TENSOR_VALUE, POLY_DEVICE_CPU), 0
  );
  poly_buffer_set(ctx, capture_buffer, capture_data, sizeof(capture_data), POLY_DEVICE_CPU);

  PolyJit *jit = poly_jit_new(ctx);
  ASSERT_NOT_NULL(jit);
  ASSERT_INT_EQ(poly_jit_begin_capture(jit, &capture, 1), 0);
  PolyTensor *one =
      poly_tensor_const_float_by_id(ctx, 1.0, poly_dtype_id_by_name("float32"), POLY_DEVICE_CPU);
  PolyTensor *out = poly_tensor_alu2(ctx, POLY_OP_ADD, capture, one);
  PolyTensor *realized = NULL;
  ASSERT_NOT_NULL(out);
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &out, 1, &realized), 0);
  ASSERT_INT_EQ(poly_jit_end_capture(jit, ctx->tensors, ctx->n_tensors), 0);

  float replay_data[3] = {10.0f, 20.0f, 30.0f};
  PolyTensor *same_identity = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(same_identity);
  PolyUOp *same_logical = poly_tensor_uop_logical(same_identity);
  ASSERT_NOT_NULL(same_logical);
  PolyUOp *same_buffer =
      poly_uop_new_buffer(ctx, cpu1, 3, POLY_FLOAT32, same_logical->src[0]->arg.i);
  ASSERT_NOT_NULL(same_buffer);
  ASSERT_INT_EQ(
      poly_tensor_set_physical(ctx, same_identity, same_buffer, POLY_TENSOR_VALUE, POLY_DEVICE_CPU),
      0
  );
  poly_buffer_set(ctx, same_buffer, replay_data, sizeof(replay_data), POLY_DEVICE_CPU);
  ASSERT_INT_EQ(poly_jit_run(jit, &same_identity, 1), 0);

  PolyTensor *different_identity =
      initialized_f32_tensor(ctx, shape, 1, replay_data, POLY_DEVICE_CPU, NULL);
  ASSERT_NOT_NULL(different_identity);
  /* Pinned expected_input_info includes the exact device string. CPU and
   * CPU:1 use the same implementation class but are distinct JIT signatures
   * (tinygrad/engine/jit.py:240-244,301,307-308). */
  ASSERT_INT_EQ(poly_jit_run(jit, &different_identity, 1), -1);

  poly_jit_free(jit);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, graph_empty_input_allocates_like_tinygrad) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer_f32(ctx, 4);
  PolyUOp *b = poly_buffer_f32(ctx, 4);
  PolyUOp *add = poly_alu2(ctx, POLY_OP_ADD, a, b);

  float da[] = {1.0f, 2.0f, 3.0f, 4.0f};
  poly_buffer_set(ctx, a, da, sizeof(da), POLY_DEVICE_CPU);

  PolyUOp *targets[] = {add};
  PolyUOp *realized[] = {NULL};
  ASSERT_INT_EQ(poly_realize_uops(ctx, targets, 1, realized), 0);
  ASSERT_NOT_NULL(poly_buffer_get(ctx, b));
  ASSERT_NOT_NULL(realized[0]);
  ASSERT_NOT_NULL(realized_buffer(ctx, realized[0]));

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, global_counters_track_calls_and_preserve_live_memory_on_reset) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  float x_data[8];
  for (int i = 0; i < 8; i++)
    x_data[i] = (float)i;
  int64_t shape[] = {8};
  int f32 = poly_dtype_id_by_name("float32");
  PolyTensor *x = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_CPU);
  PolyUOp *x_buffer =
      x ? (PolyUOp *)poly_uop_get_buffer_identity(poly_tensor_uop_physical(x)) : NULL;
  PolyBuffer *x_storage = NULL;
  ASSERT_NOT_NULL(x_buffer);
  ASSERT_INT_EQ(poly_buffer_alloc_owned_host(ctx, x_buffer, sizeof(x_data), false, &x_storage), 0);
  ASSERT_NOT_NULL(x_storage);
  memcpy(x_storage->ptr, x_data, sizeof(x_data));
  ASSERT_INT_EQ(poly_buffer_mark_host_written(ctx, x_buffer), 0);
  PolyTensor *one = poly_tensor_const_float_by_id(ctx, 1.0, f32, POLY_DEVICE_CPU);
  PolyTensor *out = poly_tensor_alu2(ctx, POLY_OP_ADD, x, one);
  ASSERT_NOT_NULL(x);
  ASSERT_NOT_NULL(out);

  PolyCtxStats stats = {0};
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &stats), 0);
  ASSERT_INT_EQ(stats.mem_used, 32);
  poly_ctx_reset_counters(ctx);
  PolyTensor *out_inputs[] = {out};
  PolyTensor *out_outputs[] = {NULL};
  ASSERT_INT_EQ(poly_realize_tensors_ex(ctx, out_inputs, 1, out_outputs, true), 0);
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &stats), 0);
  ASSERT_TRUE(stats.global_ops == 8);
  ASSERT_TRUE(stats.global_mem == 64);
  ASSERT_TRUE(stats.kernel_count == 1);
  ASSERT_INT_EQ(stats.mem_used, 64);

  poly_ctx_reset_counters(ctx);
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &stats), 0);
  ASSERT_TRUE(stats.global_ops == 0 && stats.global_mem == 0 && stats.kernel_count == 0);
  ASSERT_TRUE(stats.time_sum_s == 0.0);
  ASSERT_INT_EQ(stats.mem_used, 64);

  PolyTensor *two = poly_tensor_const_float_by_id(ctx, 2.0, f32, POLY_DEVICE_CPU);
  PolyTensor *suppressed = poly_tensor_alu2(ctx, POLY_OP_ADD, x, two);
  ASSERT_NOT_NULL(suppressed);
  PolyTensor *suppressed_inputs[] = {suppressed};
  PolyTensor *suppressed_outputs[] = {NULL};
  ASSERT_INT_EQ(poly_realize_tensors_ex(ctx, suppressed_inputs, 1, suppressed_outputs, false), 0);
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &stats), 0);
  ASSERT_TRUE(stats.global_ops == 0 && stats.global_mem == 0 && stats.kernel_count == 0);
  ASSERT_INT_EQ(stats.mem_used, 96);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, bound_view_materialization_executes_runtime_extent) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  /* Keep the original BIND producer for this callify topology assertion. */
  ASSERT_INT_EQ(poly_ctx_set_logical_policy(ctx, POLY_LOGICAL_ALWAYS), 0);

  float x_data[8] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
  int64_t shape[] = {8};
  int f32 = poly_dtype_id_by_name("float32");
  PolyTensor *x = initialized_f32_tensor(ctx, shape, 1, x_data, POLY_DEVICE_CPU, NULL);

  PolyUOp *n = poly_uop_variable(ctx, "n", 1, 8, POLY_WEAKINT, 1, false);
  PolyUOp *bound_n = poly_uop_bind(ctx, n, 4);
  PolyUOp *starts[1] = {poly_const_int(ctx, 0)};
  PolyUOp *sizes[1] = {bound_n};
  PolyTensor *view = poly_tensor_shrink_uop(ctx, x, starts, sizes, 1);
  PolyTensor *one = poly_tensor_const_float_by_id(ctx, 1.0, f32, POLY_DEVICE_CPU);
  PolyTensor *out = poly_tensor_alu2(ctx, POLY_OP_ADD, view, one);
  ASSERT_NOT_NULL(n);
  ASSERT_NOT_NULL(bound_n);
  ASSERT_NOT_NULL(view);
  ASSERT_NOT_NULL(out);
  ASSERT_PTR_EQ(poly_uop_shape_dim(ctx, poly_tensor_uop_physical(out), 0), bound_n);
  PolyUOp *logical = poly_tensor_uop_logical(out);

  poly_ctx_reset_counters(ctx);
  PolyTensor *realized = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &out, 1, &realized), 0);
  ASSERT_NOT_NULL(realized);
  ASSERT_PTR_EQ(poly_tensor_uop_logical(out), logical);
  PolyUOp *physical = poly_tensor_uop_physical(out);
  ASSERT_NOT_NULL(physical);
  /* Pinned callify.py:169-181 maps the materialized max-size BUFFER back
   * through shrink_to(original.shape). UOp.has_buffer_identity deliberately
   * excludes SHRINK (uop/ops.py:825-829), while its src[0] owns the storage. */
  ASSERT_INT_EQ(physical->op, POLY_OP_SHRINK);
  ASSERT_INT_EQ(physical->n_src, 3);
  ASSERT_INT_EQ(physical->src[0]->op, POLY_OP_BUFFER);
  ASSERT_INT_EQ(physical->src[1]->op, POLY_OP_CONST);
  ASSERT_INT_EQ(physical->src[1]->arg.i, 0);
  ASSERT_PTR_EQ(physical->src[2], bound_n);
  ASSERT_TRUE(poly_uop_get_buffer_identity(physical) == NULL);
  const PolyUOp *physical_storage = poly_uop_get_buffer_identity(physical->src[0]);
  ASSERT_NOT_NULL(physical_storage);

  float values[4] = {0};
  ASSERT_INT_EQ(poly_buffer_read(ctx, (PolyUOp *)physical_storage, values, sizeof(values)), 0);
  for (int i = 0; i < 4; i++)
    ASSERT_FLOAT_EQ(values[i], (float)(i + 1), 1e-5f);

  PolyCtxStats stats = {0};
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &stats), 0);
  ASSERT_TRUE(stats.global_ops == 4);
  ASSERT_TRUE(stats.global_mem == 32);
  ASSERT_TRUE(stats.kernel_count == 1);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, pure_movement_view_realize_is_zero_call) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  float data[8] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
  int64_t base_shape[] = {8};
  PolyUOp *base = NULL;
  PolyTensor *base_tensor =
      initialized_f32_tensor(ctx, base_shape, 1, data, POLY_DEVICE_CPU, &base);

  int64_t shape[2] = {2, 4};
  int64_t order[2] = {1, 0};
  PolyTensor *tensor =
      poly_tensor_permute(ctx, poly_tensor_reshape(ctx, base_tensor, shape, 2), order, 2);
  ASSERT_NOT_NULL(tensor);
  PolyUOp *logical = poly_tensor_uop_logical(tensor);

  PolyCtxStats before = {0};
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &before), 0);
  poly_ctx_reset_counters(ctx);
  PolyTensor *realized = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &tensor, 1, &realized), 0);
  ASSERT_PTR_EQ(realized, tensor);
  ASSERT_PTR_EQ(poly_tensor_uop_logical(tensor), logical);
  PolyUOp *physical = poly_tensor_uop_physical(tensor);
  ASSERT_NOT_NULL(physical);
  ASSERT_EQ(physical->op, POLY_OP_PERMUTE);
  while (physical && poly_opset_has(POLY_GROUP_MOVEMENT, physical->op))
    physical = physical->src[0];
  ASSERT_PTR_EQ(poly_uop_get_buffer_identity(physical), base);

  PolyCtxStats stats = {0};
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &stats), 0);
  ASSERT_TRUE(stats.global_ops == 0);
  ASSERT_TRUE(stats.global_mem == 0);
  ASSERT_TRUE(stats.kernel_count == 0);
  ASSERT_TRUE(stats.mem_used == before.mem_used);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, disk_contiguous_shrink_is_lazy_and_copies_only_selected_range) {
  char path[256];
  snprintf(path, sizeof(path), "temp/polygrad_disk_view_%ld.bin", (long)getpid());
  FILE *file = fopen(path, "wb");
  ASSERT_NOT_NULL(file);
  uint8_t file_data[16];
  for (int i = 0; i < 16; i++)
    file_data[i] = (uint8_t)i;
  ASSERT_TRUE(fwrite(file_data, 1, sizeof(file_data), file) == sizeof(file_data));
  ASSERT_INT_EQ(fclose(file), 0);

  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  int uint8_id = poly_dtype_id_by_name("uint8");
  ASSERT_TRUE(uint8_id >= 0);
  PolyUOp *disk_buffer = poly_buffer_from_file(ctx, path, uint8_id);
  ASSERT_NOT_NULL(disk_buffer);
  ASSERT_INT_EQ(poly_uop_device(disk_buffer), POLY_DEVICE_DISK);
  ASSERT_FALSE(poly_device_can_execute(POLY_DEVICE_DISK));
  ASSERT_TRUE(poly_device_is_host_addressable(POLY_DEVICE_DISK));

  PolyBuffer *disk_storage = poly_buffer_get(ctx, disk_buffer);
  ASSERT_NOT_NULL(disk_storage);
  ASSERT_INT_EQ(disk_storage->device, POLY_DEVICE_DISK);
  ASSERT_TRUE(disk_storage->nbytes == sizeof(file_data));
  ASSERT_TRUE(disk_storage->valid);

  PolyCtxStats stats = {0};
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &stats), 0);
  ASSERT_TRUE(stats.mem_used == 0);

  int64_t bounds[1][2] = {{3, 9}};
  PolyUOp *slice = poly_shrink(ctx, disk_buffer, bounds, 1);
  ASSERT_NOT_NULL(slice);
  PolyTensor *disk_tensor =
      poly_tensor_create_with_roots(ctx, slice, slice, POLY_TENSOR_VALUE, POLY_DEVICE_DISK);
  ASSERT_NOT_NULL(disk_tensor);

  poly_ctx_reset_counters(ctx);
  PolyTensor *disk_outputs[1] = {NULL};
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &disk_tensor, 1, disk_outputs), 0);
  ASSERT_NOT_NULL(disk_outputs[0]);
  PolyUOp *disk_realized = poly_tensor_uop(disk_outputs[0]);
  ASSERT_NOT_NULL(disk_realized);
  /* Pinned tinygrad leaves the immutable SHRINK root in place and resolves
   * its runtime Buffer.view lazily (uop/ops.py:838-852). */
  ASSERT_EQ(disk_realized->op, POLY_OP_SHRINK);
  PolyUOp *view_key = poly_uop_buffer(ctx, disk_realized);
  ASSERT_PTR_EQ(view_key, disk_realized);
  PolyBuffer *view_storage = poly_buffer_get(ctx, view_key);
  ASSERT_NOT_NULL(view_storage);
  ASSERT_INT_EQ(view_storage->device, POLY_DEVICE_DISK);
  ASSERT_TRUE(view_storage->nbytes == 6);
  ASSERT_PTR_EQ(view_storage->base, disk_storage);
  ASSERT_TRUE(view_storage->offset == 3);
  uint8_t view_data[6] = {0};
  ASSERT_INT_EQ(poly_buffer_read(ctx, disk_realized, view_data, sizeof(view_data)), 0);
  for (int i = 0; i < 6; i++)
    ASSERT_INT_EQ(view_data[i], i + 3);
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &stats), 0);
  ASSERT_TRUE(stats.global_ops == 0 && stats.global_mem == 0 && stats.kernel_count == 0);
  ASSERT_TRUE(stats.mem_used == 0);

  PolyTensor *cpu_tensor = poly_tensor_to_device(ctx, disk_outputs[0], POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(cpu_tensor);
  PolyUOp *schedule_input = poly_tensor_uop_physical(cpu_tensor);
  PolyUOp *schedule_output = NULL;
  PolyUOp *copy_schedule = poly_test_linear_values(ctx, &schedule_input, 1, &schedule_output);
  ASSERT_NOT_NULL(copy_schedule);
  ASSERT_INT_EQ(copy_schedule->n_src, 1);
  ASSERT_TRUE(poly_test_linear_call_is_copy(copy_schedule, 0));
  poly_ctx_reset_counters(ctx);
  PolyTensor *cpu_outputs[1] = {NULL};
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &cpu_tensor, 1, cpu_outputs), 0);
  ASSERT_NOT_NULL(cpu_outputs[0]);
  const PolyUOp *cpu_identity = poly_uop_get_buffer_identity(poly_tensor_uop(cpu_outputs[0]));
  ASSERT_NOT_NULL(cpu_identity);
  PolyBuffer *cpu_storage = poly_buffer_get(ctx, (PolyUOp *)cpu_identity);
  ASSERT_NOT_NULL(cpu_storage);
  ASSERT_INT_EQ(cpu_storage->device, POLY_DEVICE_CPU);
  ASSERT_TRUE(cpu_storage->nbytes == 6);
  ASSERT_PTR_NEQ(cpu_storage->ptr, view_storage->ptr);
  uint8_t cpu_data[6] = {0};
  ASSERT_INT_EQ(poly_buffer_read(ctx, (PolyUOp *)cpu_identity, cpu_data, sizeof(cpu_data)), 0);
  for (int i = 0; i < 6; i++)
    ASSERT_INT_EQ(cpu_data[i], i + 3);
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &stats), 0);
  ASSERT_TRUE(stats.global_ops == 0);
  ASSERT_TRUE(stats.global_mem == 6);
  ASSERT_TRUE(stats.kernel_count == 1);
  ASSERT_TRUE(stats.mem_used == 6);

  poly_ctx_destroy(ctx);
  ASSERT_INT_EQ(unlink(path), 0);
  PASS();
}

TEST(realize, contiguous_empty_view_has_no_storage_shortcut) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *buffer = poly_test_buffer(ctx, POLY_FLOAT32, 0);
  PolyUOp *matrix = poly_reshape(ctx, buffer, (int64_t[]){2, 0}, 2);
  PolyUOp *slice = poly_shrink(ctx, matrix, (int64_t[2][2]){{0, 1}, {0, 0}}, 2);
  ASSERT_NOT_NULL(slice);
  PolyUOp *identity = NULL;
  PolyShape shape;
  int64_t numel = -1;
  size_t offset = 99;
  ASSERT_TRUE(!poly_uop_contiguous_view_info(ctx, slice, &identity, &shape, &numel, &offset));
  ASSERT_TRUE(identity == NULL);
  ASSERT_INT_EQ(numel, -1);
  ASSERT_INT_EQ(offset, 0);
  poly_ctx_destroy(ctx);
  PASS();
}

static int test_copy_callback(
    const PolyBuffer *dst,
    const PolyBuffer *src,
    size_t nbytes,
    void *user
) {
  (void)dst;
  (void)src;
  (void)nbytes;
  (*(int *)user)++;
  return 0;
}

TEST(realize, buffer_copy_rejects_missing_transfer_callbacks) {
  int calls = 0;
  for (int missing = 0; missing < 2; missing++) {
    PolyAllocator source = {
        .copy_out = missing == 0 ? NULL : test_copy_callback, .dev_ctx = &calls};
    PolyAllocator target = {.copy_in = missing == 1 ? NULL : test_copy_callback, .dev_ctx = &calls};
    float a = 3, b = 7;
    PolyBuffer src = {.ptr = &a, .nbytes = sizeof(a), .allocator = &source, .valid = true};
    PolyBuffer dst = {.ptr = &b, .nbytes = sizeof(b), .allocator = &target};
    ASSERT_INT_EQ(poly_buffer_copy(&dst, &src), -1);
    ASSERT_INT_EQ(calls, 0);
    ASSERT_FLOAT_EQ(b, 7, 0);
    ASSERT_TRUE(!dst.valid);
  }
  PASS();
}

TEST(realize, disk_copy_publishes_named_storage_and_preserves_distinct_paths) {
  char paths[2][64] = {"temp/polygrad_Disk_First_XXXXXX", "temp/polygrad_Disk_Second_XXXXXX"};
  for (int i = 0; i < 2; i++) {
    int fd = mkstemp(paths[i]);
    ASSERT_TRUE(fd >= 0);
    ASSERT_INT_EQ(close(fd), 0);
  }
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  float values[] = {1, 2, 3, 4}, got[4];
  PolyUOp *buffer = poly_test_buffer(ctx, POLY_FLOAT32, 4);
  ASSERT_INT_EQ(poly_buffer_write(ctx, buffer, values, sizeof(values)), 0);
  PolyTensor *current =
      poly_tensor_create_with_roots(ctx, buffer, buffer, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(current);
  for (int i = 0; i < 2; i++) {
    char device[80];
    snprintf(device, sizeof(device), "DISK:%s", paths[i]);
    PolyTensor *next = poly_tensor_to_device_name(ctx, current, device), *out = NULL;
    ASSERT_NOT_NULL(next);
    ASSERT_TRUE(next != current);
    ASSERT_INT_EQ(poly_tensor_uop_physical(next)->op, POLY_OP_COPY);
    ASSERT_INT_EQ(poly_realize_tensors(ctx, &next, 1, &out), 0);
    ASSERT_PTR_EQ(next, out);
    PolyUOp *root = poly_tensor_uop_physical(out);
    ASSERT_INT_EQ(root->op, POLY_OP_BUFFER);
    ASSERT_STR_EQ(poly_uop_device_name(ctx, root), device);
    PolyUOp *identity = poly_uop_buffer(ctx, root);
    ASSERT_PTR_EQ(identity, root);
    /* No intervening read may initialize the first file's runtime metadata. */
    if (i == 1) {
      ASSERT_INT_EQ(poly_buffer_read(ctx, identity, got, sizeof(got)), 0);
      ASSERT_TRUE(memcmp(got, values, sizeof(got)) == 0);
    }
    poly_tensor_release(current);
    current = next;
  }
  poly_tensor_release(current);
  poly_ctx_destroy(ctx);
  for (int i = 0; i < 2; i++) {
    FILE *file = fopen(paths[i], "rb");
    ASSERT_NOT_NULL(file);
    ASSERT_TRUE(fread(got, 1, sizeof(got), file) == sizeof(got));
    ASSERT_INT_EQ(fclose(file), 0);
    ASSERT_TRUE(memcmp(got, values, sizeof(got)) == 0);
    ASSERT_INT_EQ(unlink(paths[i]), 0);
  }
  PASS();
}

TEST(realize, contiguous_realized_view_is_only_physical_at_tensor_boundary) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  float data[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  PolyUOp *base = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 8, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(base);
  PolyBuffer handle = poly_buffer_make_host_view(data, sizeof(data));
  poly_buffer_attach(ctx, base, &handle);

  int64_t nested_bounds[1][2] = {{1, 6}};
  PolyUOp *nested_view = poly_shrink(ctx, base, nested_bounds, 1);
  ASSERT_NOT_NULL(nested_view);
  PolyUOp *value = poly_alu2(ctx, POLY_OP_ADD, nested_view, poly_const_float(ctx, 1.0));
  ASSERT_NOT_NULL(value);
  PolyTensor *value_tensor =
      physical_tensor_from_uop(ctx, value, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(value_tensor);
  PolyUOp *value_physical = poly_tensor_uop_physical(value_tensor);
  ASSERT_NOT_NULL(value_physical);
  ASSERT_INT_EQ(count_root_ops(ctx, value_physical, POLY_OP_SHRINK), 1);

  int64_t root_bounds[1][2] = {{2, 5}};
  PolyUOp *root_view = poly_shrink(ctx, base, root_bounds, 1);
  ASSERT_NOT_NULL(root_view);
  PolyTensor *root_tensor =
      physical_tensor_from_uop(ctx, root_view, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(root_tensor);
  PolyUOp *root_physical = poly_tensor_uop_physical(root_tensor);
  ASSERT_NOT_NULL(root_physical);
  ASSERT_PTR_EQ(root_physical, root_view);
  ASSERT_PTR_EQ(poly_uop_buffer(ctx, root_physical), root_physical);
  PolyBuffer *alias = poly_buffer_get(ctx, root_physical);
  ASSERT_NOT_NULL(alias);
  ASSERT_PTR_EQ(alias->base, poly_buffer_get(ctx, base));
  ASSERT_TRUE(alias->offset == 2 * sizeof(float));
  ASSERT_INT_EQ((int)alias->nbytes, 3 * (int)sizeof(float));

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, direct_deviceless_root_realize_matches_pinned_noop) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *root =
      poly_alu2(ctx, POLY_OP_ADD, poly_const_float(ctx, 2.0), poly_const_float(ctx, 3.0));
  ASSERT_NOT_NULL(root);
  ASSERT_INT_EQ(poly_uop_device(root), POLY_DEVICE_AUTO);
  ASSERT_FALSE(poly_uop_has_buffer_identity(root));
  PolyTensor *tensor =
      poly_tensor_create_with_roots(ctx, root, root, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(tensor);

  poly_ctx_reset_counters(ctx);
  PolyTensor *out = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &tensor, 1, &out), 0);
  ASSERT_PTR_EQ(out, tensor);
  ASSERT_PTR_EQ(poly_tensor_uop_physical(out), root);
  ASSERT_FALSE(poly_uop_has_buffer_identity(poly_tensor_uop_physical(out)));

  PolyCtxStats stats = {0};
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &stats), 0);
  ASSERT_TRUE(stats.global_ops == 0);
  ASSERT_TRUE(stats.global_mem == 0);
  ASSERT_TRUE(stats.kernel_count == 0);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, unallocated_cpu_metadata_is_not_retained_storage) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *buffer = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 3, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(buffer);

  /* Tinygrad 2026-08-22 a9069c17 Buffer.ensure_allocated turns metadata into
   * storage; the prior unallocated object is not retained as a byte source. */
  PolyBuffer *metadata = poly_uop_buffer_handle(ctx, buffer);
  ASSERT_NOT_NULL(metadata);
  ASSERT_PTR_EQ(metadata->ptr, NULL);
  ASSERT_INT_EQ(poly_buffer_ensure_allocated(ctx, buffer, POLY_DEVICE_CPU), 0);

  float values[] = {5.5f, 8.5f, 11.5f};
  ASSERT_INT_EQ(poly_buffer_copyin(ctx, buffer, values, sizeof(values)), 0);
  float got[3] = {0};
  ASSERT_INT_EQ(poly_buffer_read(ctx, buffer, got, sizeof(got)), 0);
  for (int i = 0; i < 3; i++)
    ASSERT_FLOAT_EQ(got[i], values[i], 0.0f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, raw_cpu_nonzero_identity_executes_without_aliasing_device_metadata) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  poly_ctx_set_preferred_device(ctx, POLY_DEVICE_CPU);

  /* Pinned CPU:1 opens a distinct CPUDevice/allocator identity while using the
   * same CPU implementation class (device.py:15-35,101-171;
   * runtime/ops_cpu.py:122-142). */
  PolyUOp *device = poly_device_uop_from_name(ctx, "CPU:1");
  PolyUOp *buffer = poly_uop_new_buffer(ctx, device, 4, POLY_FLOAT32, poly_ctx_next_unique_id(ctx));
  float input[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  ASSERT_INT_EQ(poly_buffer_allocate(ctx, buffer, POLY_DEVICE_CPU), 0);
  ASSERT_INT_EQ(poly_buffer_copyin(ctx, buffer, input, sizeof(input)), 0);
  PolyUOp *root = poly_uop2(
      ctx, POLY_OP_ADD, POLY_FLOAT32, buffer,
      poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(1.0)), poly_arg_none()
  );
  ASSERT_NOT_NULL(root);

  PolyUOp *realized = NULL;
  ASSERT_INT_EQ(poly_realize_uops(ctx, &root, 1, &realized), 0);
  ASSERT_NOT_NULL(realized);
  const PolyUOp *realized_identity = poly_uop_get_buffer_identity(realized);
  ASSERT_NOT_NULL(realized_identity);
  PolyBuffer *residency = poly_buffer_get(ctx, (PolyUOp *)realized_identity);
  ASSERT_NOT_NULL(residency);
  ASSERT_INT_EQ(residency->device, POLY_DEVICE_CPU);
  ASSERT_PTR_EQ(residency->device_uop, device);
  ASSERT_PTR_EQ(residency->memory_device_uop, device);
  float output[4] = {0};
  ASSERT_INT_EQ(poly_buffer_read(ctx, (PolyUOp *)realized_identity, output, sizeof(output)), 0);
  for (int i = 0; i < 4; i++)
    ASSERT_FLOAT_EQ(output[i], input[i] + 1.0f, 0.0f);
  ASSERT_TRUE(poly_ctx_mem_used_for_device_uop(ctx, device) >= sizeof(output));
  ASSERT_INT_EQ(
      (int)poly_ctx_mem_used_for_device_uop(ctx, poly_device_uop(ctx, POLY_DEVICE_CPU)), 0
  );
  ASSERT_INT_EQ((int)poly_runtime_cache_len(ctx), 1);

  /* Pinned runtime_cache is keyed by (PROGRAM.key, exact device string), so
   * the same kernel class on CPU and CPU:1 must compile/cache independently
   * (tinygrad/engine/realize.py:113-119). */
  PolyUOp *cpu_buffer = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(cpu_buffer);
  poly_buffer_set(ctx, cpu_buffer, input, sizeof(input), POLY_DEVICE_CPU);
  PolyUOp *cpu_root = poly_uop2(
      ctx, POLY_OP_ADD, POLY_FLOAT32, cpu_buffer,
      poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(1.0)), poly_arg_none()
  );
  PolyUOp *cpu_realized = NULL;
  ASSERT_INT_EQ(poly_realize_uops(ctx, &cpu_root, 1, &cpu_realized), 0);
  ASSERT_NOT_NULL(cpu_realized);
  ASSERT_INT_EQ((int)poly_runtime_cache_len(ctx), 2);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, raw_explicit_unsupported_device_fails_before_schedule_fallback) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  poly_ctx_set_preferred_device(ctx, POLY_DEVICE_CPU);

  /* Pinned opens the exact ordinal and reports CUDA error 101 on this one-GPU
   * host. Polygrad must not substitute its ordinal-zero CUDA singleton. */
  PolyUOp *device = poly_device_uop_from_name(ctx, "CUDA:1");
  PolyUOp *buffer = poly_uop_new_buffer(ctx, device, 4, POLY_FLOAT32, poly_ctx_next_unique_id(ctx));
  float input[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  poly_buffer_set(ctx, buffer, input, sizeof(input), POLY_DEVICE_HOST);
  PolyUOp *root = poly_uop2(
      ctx, POLY_OP_ADD, POLY_FLOAT32, buffer,
      poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(1.0)), poly_arg_none()
  );
  ASSERT_NOT_NULL(root);

  PolyUOp *realized = (PolyUOp *)(uintptr_t)1;
  ASSERT_INT_EQ(poly_realize_uops(ctx, &root, 1, &realized), -1);
  ASSERT_TRUE(realized == NULL);

  PolyUOp *store = poly_store_val(ctx, buffer, root);
  PolyUOp *sink = store ? poly_sink1(ctx, store) : NULL;
  ASSERT_NOT_NULL(sink);
  ASSERT_TRUE(poly_test_create_linear(ctx, sink) == NULL);

  /* Direct prebuilt LINEAR is a lower-level schedule ingress. It must apply
   * the same exact-device rule rather than infer ordinal-zero CPU from AUTO. */
  PolyUOp *param = poly_test_program_param(ctx, POLY_FLOAT32, 4, 0);
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(0));
  PolyUOp *index = poly_uop_index(ctx, param, &zero, 1);
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(1.0));
  PolyUOp *direct_store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, index, one, poly_arg_none());
  PolyUOp *direct_body = poly_test_kernel_sink(ctx, &direct_store, 1, "unsupported_device");
  PolyUOp *direct_call_src[2] = {direct_body, buffer};
  PolyUOp *direct_call =
      poly_uop(ctx, POLY_OP_CALL, POLY_VOID, direct_call_src, 2, poly_arg_none());
  PolyUOp *linear = poly_uop1(ctx, POLY_OP_LINEAR, POLY_VOID, direct_call, poly_arg_none());
  ASSERT_NOT_NULL(linear);
  ASSERT_INT_EQ(poly_run_linear(ctx, linear, NULL, 0, NULL, 0, true, false, false), -1);

  PolyCtxStats stats = {0};
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &stats), 0);
  ASSERT_TRUE(stats.kernel_count == 0);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, raw_logical_buffer_requires_explicit_placement) {
  /* Tinygrad 2026-08-22/a9069c177a9d UOp.new_buffer always records a concrete
   * device (uop/ops.py:814-818). Polygrad's device-free BUFFER is portable
   * source IR; attached residency must not let it bypass explicit place. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *logical = poly_uop_new_logical_buffer(ctx, POLY_FLOAT32, 2);
  float values[2] = {1.0f, 2.0f};
  PolyBuffer host = poly_buffer_make_host_view(values, sizeof(values));
  ASSERT_NOT_NULL(logical);
  poly_buffer_attach(ctx, logical, &host);
  ASSERT_NOT_NULL(poly_buffer_get(ctx, logical));
  PolyUOp *root = poly_add(ctx, logical, poly_const_float(ctx, 1.0f));
  PolyUOp *out = (PolyUOp *)(uintptr_t)1;
  ASSERT_NOT_NULL(root);
  ASSERT_INT_EQ(poly_realize_uops(ctx, &root, 1, &out), -1);
  ASSERT_PTR_EQ(out, NULL);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, runtime_buffer_retains_exact_device_identity) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *buffer = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(buffer);
  float input[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  poly_buffer_set(ctx, buffer, input, sizeof(input), POLY_DEVICE_HOST);
  PolyBuffer *host = poly_buffer_get(ctx, buffer);
  ASSERT_NOT_NULL(host);
  ASSERT_NOT_NULL(host->device_uop);
  ASSERT_STR_EQ(host->device_uop->arg.str, "HOST");

  ASSERT_INT_EQ(poly_buffer_ensure_device_current(ctx, buffer, POLY_DEVICE_CPU), 0);
  PolyBuffer *cpu = poly_buffer_get(ctx, buffer);
  ASSERT_NOT_NULL(cpu);
  ASSERT_NOT_NULL(cpu->device_uop);
  ASSERT_STR_EQ(cpu->device_uop->arg.str, "CPU");
  ASSERT_PTR_EQ(cpu->device_uop, poly_uop_device_uop_cached(ctx, buffer, NULL));
  ASSERT_NOT_NULL(cpu->memory_device_uop);
  ASSERT_PTR_EQ(cpu->memory_device_uop, cpu->device_uop);
  ASSERT_TRUE(poly_ctx_mem_used_for_device_uop(ctx, cpu->device_uop) >= sizeof(input));

  PolyUOp *cpu1 = poly_device_uop_from_name(ctx, "CPU:1");
  ASSERT_NOT_NULL(cpu1);
  poly_ctx_record_memory_alloc_exact(ctx, cpu1, POLY_DEVICE_CPU, 64);
  ASSERT_INT_EQ((int)poly_ctx_mem_used_for_device_uop(ctx, cpu1), 64);
  ASSERT_TRUE(poly_ctx_mem_used_for_device_uop(ctx, cpu->device_uop) >= sizeof(input));
  poly_ctx_record_memory_free_exact(ctx, cpu1, POLY_DEVICE_CPU, 64);
  ASSERT_INT_EQ((int)poly_ctx_mem_used_for_device_uop(ctx, cpu1), 0);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, direct_lazy_contiguous_after_runs_producer_and_maps_storage) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  int f32 = poly_dtype_id_by_name("float32");
  int64_t shape[] = {3, 4};
  PolyTensor *value =
      poly_tensor_full_float_by_id(ctx, shape, 2, 0.0, f32, POLY_DEVICE_CPU, true, false);
  PolyTensor *source = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 2, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(value);
  ASSERT_NOT_NULL(source);
  ASSERT_PTR_EQ(poly_tensor_clone_into(ctx, source, value), source);
  ASSERT_INT_EQ(poly_tensor_uop_physical(source)->op, POLY_OP_AFTER);

  PolyUOp *logical = poly_contiguous(ctx, poly_tensor_uop_logical(source));
  PolyUOp *physical = poly_contiguous(ctx, poly_tensor_uop_physical(source));
  ASSERT_NOT_NULL(logical);
  ASSERT_NOT_NULL(physical);
  ASSERT_INT_EQ(physical->op, POLY_OP_CONTIGUOUS);
  ASSERT_INT_EQ(physical->src[0]->op, POLY_OP_AFTER);
  PolyTensor *out =
      poly_tensor_create_with_roots(ctx, logical, physical, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(out);

  /* Pinned Tensor.realize schedules deviceful roots without buffer identity
   * (tensor.py:214-219). callify removes CONTIGUOUS(AFTER) only after retaining
   * the AFTER effects and maps the original root to final storage
   * (callify.py:32-52,142-180,204-220). */
  poly_ctx_reset_counters(ctx);
  PolyTensor *realized = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &out, 1, &realized), 0);
  ASSERT_PTR_EQ(realized, out);
  PolyUOp *realized_root = poly_tensor_uop_physical(realized);
  ASSERT_NOT_NULL(realized_root);
  ASSERT_PTR_NEQ(realized_root, physical);
  ASSERT_INT_EQ(realized_root->op, POLY_OP_RESHAPE);
  const PolyUOp *identity = poly_uop_get_buffer_identity(realized_root);
  ASSERT_NOT_NULL(identity);

  float got[12];
  ASSERT_INT_EQ(poly_buffer_read(ctx, (PolyUOp *)identity, got, sizeof(got)), 0);
  for (int i = 0; i < 12; i++)
    ASSERT_FLOAT_EQ(got[i], 0.0, 0.0);

  PolyCtxStats stats = {0};
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &stats), 0);
  ASSERT_TRUE(stats.kernel_count == 1);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, tuple_buffer_and_mstack_use_pinned_multibuffer_ownership) {
  /* Pinned UOp.buffer constructs an owning MultiBuffer for tuple BUFFER and a
   * borrowing MultiBuffer for MSTACK; MSELECT returns one exact child
   * (tinygrad/uop/ops.py:853-879, tinygrad/device.py:86-97). */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  const char *names[] = {"CPU", "CPU:1"};
  PolyUOp *tuple_device = poly_device_uop_from_names(ctx, names, 2);
  PolyUOp *tuple_buffer =
      poly_uop_new_buffer(ctx, tuple_device, 4, POLY_INT32, poly_ctx_next_unique_id(ctx));
  ASSERT_NOT_NULL(tuple_buffer);
  ASSERT_PTR_EQ(poly_uop_buffer(ctx, tuple_buffer), tuple_buffer);
  PolyBuffer *owned = poly_uop_buffer_handle(ctx, tuple_buffer);
  ASSERT_NOT_NULL(owned);
  ASSERT_TRUE(poly_buffer_is_multi(owned));
  ASSERT_TRUE(owned->owns_bufs);
  ASSERT_INT_EQ(owned->n_bufs, 2);
  ASSERT_TRUE(owned->nbytes == 4 * sizeof(int32_t));
  PolyBuffer *owned0 = poly_buffer_multi_child(owned, 0);
  PolyBuffer *owned1 = poly_buffer_multi_child(owned, 1);
  ASSERT_NOT_NULL(owned0);
  ASSERT_NOT_NULL(owned1);
  ASSERT_INT_EQ(owned0->device, POLY_DEVICE_CPU);
  ASSERT_INT_EQ(owned1->device, POLY_DEVICE_CPU);
  ASSERT_STR_EQ(owned0->device_uop->arg.str, "CPU");
  ASSERT_STR_EQ(owned1->device_uop->arg.str, "CPU:1");
  ASSERT_TRUE(!owned0->ptr && !owned1->ptr);

  PolyUOp *cpu = poly_test_buffer_on_device(ctx, POLY_INT32, 4, POLY_DEVICE_CPU);
  PolyUOp *cpu1_device = poly_device_uop_from_name(ctx, "CPU:1");
  PolyUOp *cpu1 =
      poly_uop_new_buffer(ctx, cpu1_device, 4, POLY_INT32, poly_ctx_next_unique_id(ctx));
  ASSERT_NOT_NULL(cpu);
  ASSERT_NOT_NULL(cpu1);
  ASSERT_INT_EQ(poly_buffer_allocate(ctx, cpu, POLY_DEVICE_CPU), 0);
  ASSERT_INT_EQ(poly_buffer_allocate(ctx, cpu1, POLY_DEVICE_CPU), 0);
  PolyBuffer *scalar0 = poly_buffer_get(ctx, cpu);
  PolyBuffer *scalar1 = poly_buffer_get(ctx, cpu1);
  ASSERT_NOT_NULL(scalar0);
  ASSERT_NOT_NULL(scalar1);

  PolyUOp *stack_src[] = {cpu, cpu1};
  PolyUOp *stack = poly_uop(ctx, POLY_OP_MSTACK, POLY_INT32, stack_src, 2, poly_arg_none());
  ASSERT_PTR_EQ(poly_uop_buffer(ctx, stack), stack);
  PolyBuffer *borrowed = poly_uop_buffer_handle(ctx, stack);
  ASSERT_NOT_NULL(borrowed);
  ASSERT_TRUE(poly_buffer_is_multi(borrowed));
  ASSERT_FALSE(borrowed->owns_bufs);
  ASSERT_PTR_EQ(poly_buffer_multi_child(borrowed, 0), scalar0);
  ASSERT_PTR_EQ(poly_buffer_multi_child(borrowed, 1), scalar1);

  PolyUOp *select1 = poly_uop1(ctx, POLY_OP_MSELECT, POLY_INT32, stack, poly_arg_int(1));
  ASSERT_PTR_EQ(poly_uop_buffer_handle(ctx, select1), scalar1);
  ASSERT_PTR_EQ(poly_uop_buffer(ctx, select1), cpu1);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, collecting_tuple_buffer_view_preserves_live_parent_lanes) {
  /* Tinygrad 2026-08-22 a9069c17 UOp.buffer builds one Buffer.view per
   * MultiBuffer lane; collecting those views cannot deallocate their bases
   * (uop/ops.py:922-934, device.py:215-217). */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  const char *names[] = {"CPU", "CPU:1"};
  PolyUOp *tuple_device = poly_device_uop_from_names(ctx, names, 2);
  PolyUOp *tuple_buffer =
      poly_uop_new_buffer(ctx, tuple_device, 4, POLY_INT32, poly_ctx_next_unique_id(ctx));
  ASSERT_NOT_NULL(tuple_buffer);
  ASSERT_PTR_EQ(poly_uop_buffer(ctx, tuple_buffer), tuple_buffer);
  PolyBuffer *parent = poly_uop_buffer_handle(ctx, tuple_buffer);
  ASSERT_NOT_NULL(parent);
  ASSERT_TRUE(poly_buffer_is_multi(parent));
  ASSERT_TRUE(parent->owns_bufs);

  void *parent_ptrs[2] = {0};
  for (int lane = 0; lane < 2; lane++) {
    PolyBuffer *child = poly_buffer_multi_child(parent, lane);
    ASSERT_NOT_NULL(child);
    ASSERT_INT_EQ(poly_buffer_handle_ensure_allocated(ctx, child), 0);
    ASSERT_NOT_NULL(child->ptr);
    parent_ptrs[lane] = child->ptr;
    for (int i = 0; i < 4; i++)
      ((int32_t *)child->ptr)[i] = lane * 10 + i;
    child->valid = true;
  }

  PolyTensor *parent_tensor = poly_tensor_create_with_roots(
      ctx, tuple_buffer, tuple_buffer, POLY_TENSOR_VALUE, POLY_DEVICE_CPU
  );
  ASSERT_NOT_NULL(parent_tensor);
  int64_t bounds[1][2] = {{0, 2}};
  PolyUOp *view = poly_shrink(ctx, tuple_buffer, bounds, 1);
  ASSERT_NOT_NULL(view);
  ASSERT_PTR_EQ(poly_uop_buffer(ctx, view), view);
  PolyBuffer *view_handle = poly_buffer_get(ctx, view);
  ASSERT_NOT_NULL(view_handle);
  ASSERT_TRUE(poly_buffer_is_multi(view_handle));

  PolyTensor *view_tensor =
      poly_tensor_create_with_roots(ctx, view, view, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(view_tensor);
  poly_tensor_release(view_tensor);
  ASSERT_INT_EQ(poly_ctx_collect(ctx), 0);
  ASSERT_TRUE(poly_buffer_get(ctx, view) == NULL);

  ASSERT_PTR_EQ(poly_uop_buffer_handle(ctx, tuple_buffer), parent);
  for (int lane = 0; lane < 2; lane++) {
    PolyBuffer *child = poly_buffer_multi_child(parent, lane);
    ASSERT_NOT_NULL(child);
    ASSERT_PTR_EQ(child->ptr, parent_ptrs[lane]);
    ASSERT_TRUE(child->valid);
    for (int i = 0; i < 4; i++)
      ASSERT_INT_EQ(((int32_t *)child->ptr)[i], lane * 10 + i);
  }

  poly_tensor_release(parent_tensor);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, mselect_call_executes_selected_scalar_lane_like_pinned) {
  /* Pinned resolve_params replaces MSELECT(PARAM, lane) with the current
   * aggregate input before unwrap_multi, then exec_kernel allocates and runs
   * the one selected Buffer (engine/realize.py:142-180). */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *cpu_device = poly_device_uop_from_name(ctx, "CPU");
  PolyUOp *cpu1_device = poly_device_uop_from_name(ctx, "CPU:1");
  PolyUOp *in0 = poly_uop_new_buffer(ctx, cpu_device, 4, POLY_INT32, poly_ctx_next_unique_id(ctx));
  PolyUOp *in1 = poly_uop_new_buffer(ctx, cpu1_device, 4, POLY_INT32, poly_ctx_next_unique_id(ctx));
  PolyUOp *out = poly_uop_new_buffer(ctx, cpu1_device, 4, POLY_INT32, poly_ctx_next_unique_id(ctx));
  ASSERT_NOT_NULL(in0);
  ASSERT_NOT_NULL(in1);
  ASSERT_NOT_NULL(out);

  int32_t values0[] = {1, 2, 3, 4};
  int32_t values1[] = {9, 8, 7, 6};
  ASSERT_INT_EQ(poly_buffer_allocate(ctx, in0, POLY_DEVICE_CPU), 0);
  ASSERT_INT_EQ(poly_buffer_allocate(ctx, in1, POLY_DEVICE_CPU), 0);
  ASSERT_INT_EQ(poly_buffer_copyin(ctx, in0, values0, sizeof(values0)), 0);
  ASSERT_INT_EQ(poly_buffer_copyin(ctx, in1, values1, sizeof(values1)), 0);
  ASSERT_TRUE(poly_buffer_get(ctx, out) == NULL);

  PolyUOp *stack_src[] = {in0, in1};
  PolyUOp *stack = poly_uop(ctx, POLY_OP_MSTACK, POLY_INT32, stack_src, 2, poly_arg_none());
  PolyUOp *selected = poly_uop1(ctx, POLY_OP_MSELECT, POLY_INT32, stack, poly_arg_int(1));
  ASSERT_NOT_NULL(stack);
  ASSERT_NOT_NULL(selected);

  PolyUOp *dst_param = poly_test_program_param(ctx, POLY_INT32, 4, 0);
  PolyUOp *src_param = poly_test_program_param(ctx, POLY_INT32, 4, 1);
  PolyUOp *stores[4] = {0};
  for (int i = 0; i < 4; i++) {
    PolyUOp *idx = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(i));
    PolyUOp *dst_index = poly_uop_index(ctx, dst_param, &idx, 1);
    PolyUOp *src_index = poly_uop_index(ctx, src_param, &idx, 1);
    PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, POLY_INT32, src_index, poly_arg_none());
    stores[i] = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, dst_index, load, poly_arg_none());
  }
  PolyUOp *body = poly_test_kernel_sink(ctx, stores, 4, "mselect_copy");
  PolyUOp *call_src[] = {body, out, selected};
  PolyUOp *call = poly_uop(ctx, POLY_OP_CALL, POLY_VOID, call_src, 3, poly_arg_none());
  PolyUOp *linear = poly_uop1(ctx, POLY_OP_LINEAR, POLY_VOID, call, poly_arg_none());
  PolyUOp *schedule = linear;
  ASSERT_NOT_NULL(schedule);
  ASSERT_INT_EQ(schedule->n_src, 1);
  PolyUOp *scheduled_arg = poly_call_buffer_arg(poly_test_linear_call(schedule, 0), 1);
  ASSERT_NOT_NULL(scheduled_arg);
  ASSERT_INT_EQ(scheduled_arg->op, POLY_OP_MSELECT);
  ASSERT_INT_EQ(scheduled_arg->n_src, 1);
  ASSERT_INT_EQ(scheduled_arg->src[0]->op, POLY_OP_MSTACK);
  ASSERT_INT_EQ(scheduled_arg->src[0]->n_src, 2);
  ASSERT_STR_EQ(poly_uop_device_uop_cached(ctx, scheduled_arg, NULL)->arg.str, "CPU:1");

  poly_ctx_reset_counters(ctx);
  ASSERT_INT_EQ(poly_run_linear(ctx, schedule, NULL, 0, NULL, 0, true, false, false), 0);
  PolyBuffer *out_buffer = poly_buffer_get(ctx, out);
  ASSERT_NOT_NULL(out_buffer);
  ASSERT_TRUE(out_buffer->valid);
  ASSERT_STR_EQ(out_buffer->device_uop->arg.str, "CPU:1");
  int32_t got[4] = {0};
  ASSERT_INT_EQ(poly_buffer_read(ctx, out, got, sizeof(got)), 0);
  for (int i = 0; i < 4; i++)
    ASSERT_INT_EQ(got[i], values1[i]);
  PolyCtxStats stats = {0};
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &stats), 0);
  ASSERT_TRUE(stats.kernel_count == 1);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, direct_contiguous_view_buffer_accessor_is_zero_call_and_bounded) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  float data[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  PolyUOp *base = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 8, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(base);
  PolyBuffer handle = poly_buffer_make_host_view(data, sizeof(data));
  poly_buffer_attach(ctx, base, &handle);

  int64_t bounds[1][2] = {{2, 6}};
  PolyUOp *slice = poly_shrink(ctx, base, bounds, 1);
  PolyUOp *contiguous = poly_contiguous(ctx, slice);
  ASSERT_NOT_NULL(slice);
  ASSERT_NOT_NULL(contiguous);
  ASSERT_FALSE(poly_uop_has_buffer_identity(slice));
  ASSERT_FALSE(poly_uop_has_buffer_identity(contiguous));
  ASSERT_TRUE(poly_buffer_get(ctx, slice) == NULL);

  PolyTensor *tensor = poly_tensor_create_with_roots(
      ctx, contiguous, contiguous, POLY_TENSOR_VALUE, POLY_DEVICE_CPU
  );
  ASSERT_NOT_NULL(tensor);
  int64_t unique_before = ctx->next_unique_id;

  poly_ctx_reset_counters(ctx);
  PolyTensor *out = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &tensor, 1, &out), 0);
  ASSERT_PTR_EQ(out, tensor);
  ASSERT_PTR_EQ(poly_tensor_uop_physical(out), contiguous);
  ASSERT_INT_EQ(ctx->next_unique_id, unique_before);

  /* Pinned UOp.buffer returns a runtime Buffer.view without rewriting the
   * movement graph (uop/ops.py:838-852). Polygrad keys that runtime alias by
   * the exact immutable SHRINK and refreshes the same side-table row. */
  PolyUOp *view_key = poly_uop_buffer(ctx, contiguous);
  ASSERT_PTR_EQ(view_key, slice);
  PolyBuffer *view = poly_buffer_get(ctx, view_key);
  ASSERT_NOT_NULL(view);
  ASSERT_PTR_EQ(view->base, poly_buffer_get(ctx, base));
  ASSERT_TRUE(view->offset == 2 * sizeof(float));
  ASSERT_TRUE(view->nbytes == 4 * sizeof(float));
  ASSERT_FALSE(view->owned);

  PolyCtxStats first = {0};
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &first), 0);
  for (int i = 0; i < 100; i++)
    ASSERT_PTR_EQ(poly_uop_buffer(ctx, contiguous), view_key);
  PolyCtxStats replay = {0};
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &replay), 0);
  ASSERT_INT_EQ(ctx->next_unique_id, unique_before);
  ASSERT_TRUE(replay.arena_bytes == first.arena_bytes);
  ASSERT_TRUE(replay.buffer_entries == first.buffer_entries);
  ASSERT_PTR_EQ(poly_buffer_get(ctx, view_key), view);

  float got[4] = {0};
  ASSERT_INT_EQ(poly_buffer_read(ctx, view_key, got, sizeof(got)), 0);
  for (int i = 0; i < 4; i++)
    ASSERT_FLOAT_NEAR(got[i], (float)(i + 2), 1, 1e-6f);
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &replay), 0);
  ASSERT_TRUE(replay.global_ops == 0);
  ASSERT_TRUE(replay.global_mem == 0);
  ASSERT_TRUE(replay.kernel_count == 0);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, nested_contiguous_movement_view_is_one_consumer_call) {
  /* Pinned callify.py:59-89,152-164 replaces a static contiguous movement
   * with SLICE.reshape before generic CONTIGUOUS materialization. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  uint8_t data[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  PolyUOp *base = poly_test_buffer_on_device(ctx, POLY_UINT8, 8, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(base);
  ASSERT_INT_EQ(poly_buffer_allocate(ctx, base, POLY_DEVICE_CPU), 0);
  ASSERT_INT_EQ(poly_buffer_copyin(ctx, base, data, sizeof(data)), 0);

  int64_t bounds[1][2] = {{2, 6}};
  int64_t shape[2] = {2, 2};
  PolyUOp *movement = poly_reshape(ctx, poly_shrink(ctx, base, bounds, 1), shape, 2);
  PolyUOp *inner = poly_contiguous(ctx, movement);
  PolyUOp *cast = inner ? poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, inner, poly_arg_none()) : NULL;
  PolyUOp *outer = cast ? poly_contiguous(ctx, cast) : NULL;
  ASSERT_NOT_NULL(movement);
  ASSERT_NOT_NULL(inner);
  ASSERT_NOT_NULL(outer);
  ASSERT_INT_EQ(inner->op, POLY_OP_CONTIGUOUS);
  ASSERT_INT_EQ(inner->src[0]->op, POLY_OP_RESHAPE);
  ASSERT_INT_EQ(inner->src[0]->src[0]->op, POLY_OP_SHRINK);

  PolyUOp *scheduled_out = NULL;
  PolyUOp *schedule = poly_test_linear_values(ctx, &outer, 1, &scheduled_out);
  ASSERT_NOT_NULL(schedule);
  ASSERT_NOT_NULL(scheduled_out);
  ASSERT_INT_EQ(schedule->n_src, 1);
  PolyUOp *call = poly_test_linear_call(schedule, 0);
  ASSERT_NOT_NULL(call);
  ASSERT_INT_EQ(poly_call_n_buffer_args(call), 2);
  PolyUOp *view = poly_call_buffer_arg(call, 1);
  ASSERT_NOT_NULL(view);
  ASSERT_INT_EQ(view->op, POLY_OP_SHRINK);
  ASSERT_PTR_EQ(view->src[0], base);
  ASSERT_NOT_NULL(poly_buffer_get(ctx, view));

  poly_ctx_reset_counters(ctx);
  ASSERT_INT_EQ(poly_run_linear(ctx, schedule, NULL, 0, NULL, 0, true, false, false), 0);
  const PolyUOp *identity = poly_uop_get_buffer_identity(scheduled_out);
  ASSERT_NOT_NULL(identity);
  float values[4] = {0};
  ASSERT_INT_EQ(poly_buffer_read(ctx, (PolyUOp *)identity, values, sizeof(values)), 0);
  for (int i = 0; i < 4; i++)
    ASSERT_FLOAT_EQ(values[i], (float)(i + 2), 1e-6f);
  PolyCtxStats stats = {0};
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &stats), 0);
  ASSERT_TRUE(stats.kernel_count == 1);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, non_contiguous_expand_still_materializes_before_consumer) {
  /* Pinned contiguous_mops_to_view rejects an EXPAND whose flattened index is
   * not one RANGE plus a constant offset (uop/ops.py:809-823). */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  uint8_t data[2] = {2, 4};
  PolyUOp *base = poly_test_buffer_on_device(ctx, POLY_UINT8, 2, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(base);
  ASSERT_INT_EQ(poly_buffer_allocate(ctx, base, POLY_DEVICE_CPU), 0);
  ASSERT_INT_EQ(poly_buffer_copyin(ctx, base, data, sizeof(data)), 0);

  int64_t input_shape[2] = {2, 1};
  int64_t output_shape[2] = {2, 2};
  PolyUOp *expanded = poly_expand(ctx, poly_reshape(ctx, base, input_shape, 2), output_shape, 2);
  PolyUOp *inner = expanded ? poly_contiguous(ctx, expanded) : NULL;
  PolyUOp *cast = inner ? poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, inner, poly_arg_none()) : NULL;
  PolyUOp *outer = cast ? poly_contiguous(ctx, cast) : NULL;
  ASSERT_NOT_NULL(outer);

  PolyUOp *scheduled_out = NULL;
  PolyUOp *schedule = poly_test_linear_values(ctx, &outer, 1, &scheduled_out);
  ASSERT_NOT_NULL(schedule);
  ASSERT_NOT_NULL(scheduled_out);
  ASSERT_INT_EQ(schedule->n_src, 2);
  poly_ctx_reset_counters(ctx);
  ASSERT_INT_EQ(poly_run_linear(ctx, schedule, NULL, 0, NULL, 0, true, false, false), 0);
  const PolyUOp *identity = poly_uop_get_buffer_identity(scheduled_out);
  ASSERT_NOT_NULL(identity);
  float values[4] = {0};
  ASSERT_INT_EQ(poly_buffer_read(ctx, (PolyUOp *)identity, values, sizeof(values)), 0);
  ASSERT_FLOAT_EQ(values[0], 2.0f, 1e-6f);
  ASSERT_FLOAT_EQ(values[1], 2.0f, 1e-6f);
  ASSERT_FLOAT_EQ(values[2], 4.0f, 1e-6f);
  ASSERT_FLOAT_EQ(values[3], 4.0f, 1e-6f);
  PolyCtxStats stats = {0};
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &stats), 0);
  ASSERT_TRUE(stats.kernel_count == 2);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, disk_reshaped_views_batch_copy_preserves_offsets_and_counts) {
  char path[256];
  snprintf(path, sizeof(path), "temp/polygrad_disk_batch_%ld.bin", (long)getpid());
  FILE *file = fopen(path, "wb");
  ASSERT_NOT_NULL(file);
  uint8_t file_data[32];
  for (int i = 0; i < 32; i++)
    file_data[i] = (uint8_t)i;
  ASSERT_TRUE(fwrite(file_data, 1, sizeof(file_data), file) == sizeof(file_data));
  ASSERT_INT_EQ(fclose(file), 0);

  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  int uint8_id = poly_dtype_id_by_name("uint8");
  ASSERT_TRUE(uint8_id >= 0);
  PolyUOp *disk_buffer = poly_buffer_from_file(ctx, path, uint8_id);
  ASSERT_NOT_NULL(disk_buffer);
  PolyBuffer *disk_storage = poly_buffer_get(ctx, disk_buffer);
  ASSERT_NOT_NULL(disk_storage);

  int64_t left_bounds[1][2] = {{4, 10}};
  int64_t right_bounds[1][2] = {{17, 25}};
  int64_t left_shape[2] = {2, 3};
  int64_t right_shape[2] = {2, 4};
  PolyUOp *left = poly_reshape(ctx, poly_shrink(ctx, disk_buffer, left_bounds, 1), left_shape, 2);
  PolyUOp *right =
      poly_reshape(ctx, poly_shrink(ctx, disk_buffer, right_bounds, 1), right_shape, 2);
  ASSERT_NOT_NULL(left);
  ASSERT_NOT_NULL(right);
  PolyTensor *disk_inputs[2] = {
      poly_tensor_create_with_roots(ctx, left, left, POLY_TENSOR_VALUE, POLY_DEVICE_DISK),
      poly_tensor_create_with_roots(ctx, right, right, POLY_TENSOR_VALUE, POLY_DEVICE_DISK),
  };
  ASSERT_NOT_NULL(disk_inputs[0]);
  ASSERT_NOT_NULL(disk_inputs[1]);

  poly_ctx_reset_counters(ctx);
  PolyTensor *disk_outputs[2] = {NULL, NULL};
  ASSERT_INT_EQ(poly_realize_tensors(ctx, disk_inputs, 2, disk_outputs), 0);
  PolyUOp *disk_roots[2] = {poly_tensor_uop(disk_outputs[0]), poly_tensor_uop(disk_outputs[1])};
  ASSERT_NOT_NULL(disk_roots[0]);
  ASSERT_NOT_NULL(disk_roots[1]);
  ASSERT_EQ(disk_roots[0]->op, POLY_OP_RESHAPE);
  ASSERT_EQ(disk_roots[1]->op, POLY_OP_RESHAPE);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, disk_roots[0]), 2);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, disk_roots[0])[0], 2);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, disk_roots[0])[1], 3);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, disk_roots[1]), 2);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, disk_roots[1])[0], 2);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, disk_roots[1])[1], 4);

  /* Pinned UOp.buffer creates Buffer.view objects for contiguous movements
   * without replacing the RESHAPE/SHRINK graph (uop/ops.py:838-852). */
  PolyUOp *disk_ids[2] = {
      poly_uop_buffer(ctx, disk_roots[0]),
      poly_uop_buffer(ctx, disk_roots[1]),
  };
  ASSERT_NOT_NULL(disk_ids[0]);
  ASSERT_NOT_NULL(disk_ids[1]);
  ASSERT_EQ(disk_ids[0]->op, POLY_OP_SHRINK);
  ASSERT_EQ(disk_ids[1]->op, POLY_OP_SHRINK);
  PolyBuffer *disk_views[2] = {
      poly_buffer_get(ctx, (PolyUOp *)disk_ids[0]),
      poly_buffer_get(ctx, (PolyUOp *)disk_ids[1]),
  };
  ASSERT_NOT_NULL(disk_views[0]);
  ASSERT_NOT_NULL(disk_views[1]);
  ASSERT_INT_EQ(disk_views[0]->device, POLY_DEVICE_DISK);
  ASSERT_INT_EQ(disk_views[1]->device, POLY_DEVICE_DISK);
  ASSERT_TRUE(disk_views[0]->nbytes == 6);
  ASSERT_TRUE(disk_views[1]->nbytes == 8);
  ASSERT_PTR_EQ(disk_views[0]->base, disk_storage);
  ASSERT_PTR_EQ(disk_views[1]->base, disk_storage);
  ASSERT_TRUE(disk_views[0]->offset == 4);
  ASSERT_TRUE(disk_views[1]->offset == 17);

  PolyCtxStats stats = {0};
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &stats), 0);
  ASSERT_TRUE(stats.global_ops == 0 && stats.global_mem == 0 && stats.kernel_count == 0);
  ASSERT_TRUE(stats.mem_used == 0);

  PolyTensor *cpu_inputs[2] = {
      poly_tensor_to_device(ctx, disk_outputs[0], POLY_DEVICE_CPU),
      poly_tensor_to_device(ctx, disk_outputs[1], POLY_DEVICE_CPU),
  };
  ASSERT_NOT_NULL(cpu_inputs[0]);
  ASSERT_NOT_NULL(cpu_inputs[1]);
  poly_ctx_reset_counters(ctx);
  PolyTensor *cpu_outputs[2] = {NULL, NULL};
  ASSERT_INT_EQ(poly_realize_tensors(ctx, cpu_inputs, 2, cpu_outputs), 0);

  const PolyUOp *cpu_ids[2] = {
      poly_uop_get_buffer_identity(poly_tensor_uop(cpu_outputs[0])),
      poly_uop_get_buffer_identity(poly_tensor_uop(cpu_outputs[1])),
  };
  ASSERT_NOT_NULL(cpu_ids[0]);
  ASSERT_NOT_NULL(cpu_ids[1]);
  uint8_t left_data[6] = {0};
  uint8_t right_data[8] = {0};
  ASSERT_INT_EQ(poly_buffer_read(ctx, (PolyUOp *)cpu_ids[0], left_data, sizeof(left_data)), 0);
  ASSERT_INT_EQ(poly_buffer_read(ctx, (PolyUOp *)cpu_ids[1], right_data, sizeof(right_data)), 0);
  for (int i = 0; i < 6; i++)
    ASSERT_INT_EQ(left_data[i], i + 4);
  for (int i = 0; i < 8; i++)
    ASSERT_INT_EQ(right_data[i], i + 17);

  ASSERT_INT_EQ(poly_ctx_stats(ctx, &stats), 0);
  ASSERT_TRUE(stats.global_ops == 0);
  ASSERT_TRUE(stats.global_mem == 14);
  ASSERT_TRUE(stats.kernel_count == 2);
  ASSERT_TRUE(stats.mem_used == 14);

  poly_ctx_destroy(ctx);
  ASSERT_INT_EQ(unlink(path), 0);
  PASS();
}

TEST(realize, direct_movement_disk_copy_matches_pinned_creation_pipeline) {
  char path[256];
  snprintf(path, sizeof(path), "temp/polygrad_disk_direct_%ld.bin", (long)getpid());
  FILE *file = fopen(path, "wb");
  ASSERT_NOT_NULL(file);
  uint8_t file_data[32];
  for (int i = 0; i < 32; i++)
    file_data[i] = (uint8_t)i;
  ASSERT_TRUE(fwrite(file_data, 1, sizeof(file_data), file) == sizeof(file_data));
  ASSERT_INT_EQ(fclose(file), 0);

  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  int uint8_id = poly_dtype_id_by_name("uint8");
  ASSERT_TRUE(uint8_id >= 0);
  PolyUOp *disk_buffer = poly_buffer_from_file(ctx, path, uint8_id);
  ASSERT_NOT_NULL(disk_buffer);

  int64_t bounds[1][2] = {{4, 10}};
  int64_t shape[2] = {2, 3};
  PolyUOp *movement = poly_reshape(ctx, poly_shrink(ctx, disk_buffer, bounds, 1), shape, 2);
  ASSERT_NOT_NULL(movement);
  PolyUOp *device = poly_device_uop(ctx, POLY_DEVICE_CPU);
  PolyUOp *copy = poly_copy_to_device_uop(ctx, movement, device);
  ASSERT_NOT_NULL(copy);

  /* Pinned Tensor.to builds COPY(RESHAPE(SHRINK(BUFFER@DISK)), arg=device)
   * directly (tensor.py:3661-3663). This is the stored execution root:
   * no placement or storage-view projection is involved. */
  ASSERT_INT_EQ(copy->op, POLY_OP_COPY);
  ASSERT_INT_EQ(copy->n_src, 1);
  ASSERT_INT_EQ(copy->src[0]->op, POLY_OP_RESHAPE);
  ASSERT_INT_EQ(copy->src[0]->src[0]->op, POLY_OP_SHRINK);
  ASSERT_PTR_EQ(copy->src[0]->src[0]->src[0], disk_buffer);
  ASSERT_INT_EQ(copy->n_src, 1);
  ASSERT_INT_EQ(poly_uop_device(copy), POLY_DEVICE_CPU);

  /* tinygrad@2026-08-22/a9069c177a9d keeps the source view in COPY's buffer
   * metadata and emits one COPY CALL (schedule/__init__.py:155-176). */
  PolyUOp *scheduled_out = NULL;
  PolyUOp *schedule = poly_test_linear_values(ctx, &copy, 1, &scheduled_out);
  ASSERT_NOT_NULL(schedule);
  ASSERT_NOT_NULL(scheduled_out);
  ASSERT_INT_EQ(schedule->n_src, 1);
  ASSERT_TRUE(poly_test_linear_call_is_copy(schedule, 0));
  ASSERT_INT_EQ(poly_test_linear_call_body(schedule, 0)->op, POLY_OP_COPY);

  PolyTensor *tensor =
      poly_tensor_create_with_roots(ctx, movement, copy, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(tensor);
  ASSERT_PTR_EQ(poly_tensor_uop_physical(tensor), copy);

  poly_ctx_reset_counters(ctx);
  PolyTensor *realized = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &tensor, 1, &realized), 0);
  ASSERT_PTR_EQ(realized, tensor);
  const PolyUOp *identity = poly_uop_get_buffer_identity(poly_tensor_uop_physical(realized));
  ASSERT_NOT_NULL(identity);
  ASSERT_INT_EQ(poly_uop_device((PolyUOp *)identity), POLY_DEVICE_CPU);
  uint8_t values[6] = {0};
  ASSERT_INT_EQ(poly_buffer_read(ctx, (PolyUOp *)identity, values, sizeof(values)), 0);
  for (int i = 0; i < 6; i++)
    ASSERT_INT_EQ(values[i], i + 4);

  PolyCtxStats stats = {0};
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &stats), 0);
  ASSERT_TRUE(stats.global_ops == 0);
  ASSERT_TRUE(stats.global_mem == 6);
  ASSERT_TRUE(stats.kernel_count == 1);

  poly_ctx_destroy(ctx);
  ASSERT_INT_EQ(unlink(path), 0);
  PASS();
}

TEST(realize, host_reshape_contiguous_lowers_to_one_copy_call) {
  /* tinygrad@2026-08-22/a9069c177a9d schedule/__init__.py:155-176
   * simplifies movement around a cross-device full-buffer STORE before
   * replacing the CALL body with COPY(PARAM@source, target). */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  float values[4 * 3 * 6 * 6];
  for (int i = 0; i < 4 * 3 * 6 * 6; i++)
    values[i] = (float)i;
  int64_t shape[] = {4, 3, 6, 6};
  PolyTensor *host = poly_tensor_from_host(ctx, values, sizeof(values), POLY_FLOAT32, shape, 4);
  PolyTensor *cpu = poly_tensor_to_device(ctx, host, POLY_DEVICE_CPU);
  PolyTensor *contiguous = poly_tensor_contiguous(ctx, cpu);
  ASSERT_NOT_NULL(host);
  ASSERT_NOT_NULL(cpu);
  ASSERT_NOT_NULL(contiguous);

  PolyUOp *root = poly_tensor_uop_physical(contiguous);
  PolyUOp *realized = NULL;
  PolyVarBinding *bindings = NULL;
  int n_bindings = 0;
  PolyUOp *linear = poly_linear_with_vars(ctx, &root, 1, &realized, &bindings, &n_bindings);
  ASSERT_NOT_NULL(linear);
  ASSERT_INT_EQ(linear->op, POLY_OP_LINEAR);
  ASSERT_INT_EQ(linear->n_src, 1);
  PolyUOp *call = linear->src[0];
  ASSERT_INT_EQ(call->op, POLY_OP_CALL);
  ASSERT_INT_EQ(call->n_src, 3);
  ASSERT_INT_EQ(call->src[0]->op, POLY_OP_COPY);
  ASSERT_INT_EQ(call->src[0]->n_src, 1);
  ASSERT_INT_EQ(call->src[0]->src[0]->op, POLY_OP_PARAM);
  ASSERT_STR_EQ(poly_uop_device_name(ctx, call->src[0]->src[0]), "HOST");
  ASSERT_STR_EQ(poly_uop_device_name(ctx, call->src[1]), "CPU");
  ASSERT_STR_EQ(poly_uop_device_name(ctx, call->src[2]), "HOST");
  ASSERT_INT_EQ(n_bindings, 0);

  free(bindings);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, typed_disk_view_copy_matches_pinned_slice_pipeline) {
  char path[256];
  snprintf(path, sizeof(path), "temp/polygrad_disk_bitcast_%ld.bin", (long)getpid());
  FILE *file = fopen(path, "wb");
  ASSERT_NOT_NULL(file);
  uint8_t file_data[32];
  for (int i = 0; i < 32; i++)
    file_data[i] = (uint8_t)i;
  ASSERT_TRUE(fwrite(file_data, 1, sizeof(file_data), file) == sizeof(file_data));
  ASSERT_INT_EQ(fclose(file), 0);

  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  int uint8_id = poly_dtype_id_by_name("uint8");
  int uint16_id = poly_dtype_id_by_name("uint16");
  ASSERT_TRUE(uint8_id >= 0 && uint16_id >= 0);
  PolyUOp *disk_buffer = poly_buffer_from_file(ctx, path, uint8_id);
  ASSERT_NOT_NULL(disk_buffer);
  PolyTensor *source = poly_tensor_create_with_roots(
      ctx, disk_buffer, disk_buffer, POLY_TENSOR_VALUE, POLY_DEVICE_DISK
  );
  ASSERT_NOT_NULL(source);

  int64_t bounds[1][2] = {{8, 12}};
  PolyTensor *bytes = poly_tensor_shrink(ctx, source, bounds, 1);
  PolyTensor *typed = poly_tensor_bitcast_by_id(ctx, bytes, uint16_id);
  PolyTensor *cpu = poly_tensor_to_device(ctx, typed, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(bytes);
  ASSERT_NOT_NULL(typed);
  ASSERT_NOT_NULL(cpu);

  /* tinygrad@2026-08-22/a9069c177a9d constructs unary COPY over the typed
   * movement and schedules it as one COPY CALL (tensor.py:534-550). */
  PolyUOp *copy = poly_tensor_uop_physical(cpu);
  ASSERT_INT_EQ(copy->op, POLY_OP_COPY);
  ASSERT_INT_EQ(copy->src[0]->op, POLY_OP_BITCAST);
  ASSERT_INT_EQ(copy->src[0]->src[0]->op, POLY_OP_SHRINK);
  ASSERT_PTR_EQ(copy->src[0]->src[0]->src[0], disk_buffer);

  PolyUOp *scheduled_out = NULL;
  PolyUOp *schedule = poly_test_linear_values(ctx, &copy, 1, &scheduled_out);
  ASSERT_NOT_NULL(schedule);
  ASSERT_NOT_NULL(scheduled_out);
  ASSERT_INT_EQ(schedule->n_src, 1);
  ASSERT_TRUE(poly_test_linear_call_is_copy(schedule, 0));

  poly_ctx_reset_counters(ctx);
  PolyTensor *realized = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &cpu, 1, &realized), 0);
  ASSERT_PTR_EQ(realized, cpu);
  const PolyUOp *identity = poly_uop_get_buffer_identity(poly_tensor_uop_physical(realized));
  ASSERT_NOT_NULL(identity);
  uint16_t values[2] = {0};
  ASSERT_INT_EQ(poly_buffer_read(ctx, (PolyUOp *)identity, values, sizeof(values)), 0);
  ASSERT_INT_EQ(values[0], 0x0908);
  ASSERT_INT_EQ(values[1], 0x0b0a);

  PolyCtxStats stats = {0};
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &stats), 0);
  ASSERT_TRUE(stats.global_ops == 0);
  ASSERT_TRUE(stats.global_mem == sizeof(values));
  ASSERT_TRUE(stats.kernel_count == 1);

  poly_ctx_destroy(ctx);
  ASSERT_INT_EQ(unlink(path), 0);
  PASS();
}

TEST(realize, nested_disk_view_copies_feed_compute_as_ordered_calls) {
  char path[256];
  snprintf(path, sizeof(path), "temp/polygrad_disk_nested_%ld.bin", (long)getpid());
  FILE *file = fopen(path, "wb");
  ASSERT_NOT_NULL(file);
  uint8_t file_data[32];
  for (int i = 0; i < 32; i++)
    file_data[i] = (uint8_t)i;
  ASSERT_TRUE(fwrite(file_data, 1, sizeof(file_data), file) == sizeof(file_data));
  ASSERT_INT_EQ(fclose(file), 0);

  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  int uint8_id = poly_dtype_id_by_name("uint8");
  ASSERT_TRUE(uint8_id >= 0);
  PolyUOp *disk_buffer = poly_buffer_from_file(ctx, path, uint8_id);
  ASSERT_NOT_NULL(disk_buffer);

  int64_t left_bounds[1][2] = {{4, 7}};
  int64_t right_bounds[1][2] = {{17, 20}};
  int64_t row_shape[2] = {1, 3};
  PolyUOp *left = poly_reshape(ctx, poly_shrink(ctx, disk_buffer, left_bounds, 1), row_shape, 2);
  PolyUOp *right = poly_reshape(ctx, poly_shrink(ctx, disk_buffer, right_bounds, 1), row_shape, 2);
  ASSERT_NOT_NULL(left);
  ASSERT_NOT_NULL(right);

  PolyTensor *left_disk =
      poly_tensor_create_with_roots(ctx, left, left, POLY_TENSOR_VALUE, POLY_DEVICE_DISK);
  PolyTensor *right_disk =
      poly_tensor_create_with_roots(ctx, right, right, POLY_TENSOR_VALUE, POLY_DEVICE_DISK);
  ASSERT_NOT_NULL(left_disk);
  ASSERT_NOT_NULL(right_disk);
  PolyTensor *left_interp = poly_tensor_to_device(ctx, left_disk, POLY_DEVICE_INTERP);
  PolyTensor *right_interp = poly_tensor_to_device(ctx, right_disk, POLY_DEVICE_INTERP);
  ASSERT_NOT_NULL(left_interp);
  ASSERT_NOT_NULL(right_interp);

  PolyUOp *logical_rows[2] = {
      poly_tensor_uop_logical(left_interp), poly_tensor_uop_logical(right_interp)};
  PolyUOp *physical_rows[2] = {poly_tensor_uop(left_interp), poly_tensor_uop(right_interp)};
  /* tinygrad tensor.py:128-136 applies CAT and each following movement op to
   * the exact current COPY roots. Keep the same physical chain alongside the
   * exportable DISK provenance chain. */
  PolyUOp *joined = poly_cat(ctx, logical_rows, 2, 0);
  PolyUOp *joined_physical = poly_cat(ctx, physical_rows, 2, 0);
  ASSERT_NOT_NULL(joined);
  ASSERT_NOT_NULL(joined_physical);
  int64_t first_col[2][2] = {{0, 2}, {0, 1}};
  int64_t result_shape[1] = {2};
  PolyUOp *selected = poly_reshape(ctx, poly_shrink(ctx, joined, first_col, 2), result_shape, 1);
  PolyUOp *selected_physical =
      poly_reshape(ctx, poly_shrink(ctx, joined_physical, first_col, 2), result_shape, 1);
  ASSERT_NOT_NULL(selected);
  ASSERT_NOT_NULL(selected_physical);
  PolyTensor *result = poly_tensor_create_with_roots(
      ctx, selected, selected_physical, POLY_TENSOR_VALUE, POLY_DEVICE_INTERP
  );
  ASSERT_NOT_NULL(result);

  PolyUOp *physical = poly_tensor_uop_physical(result);
  PolyUOp *scheduled_out = NULL;
  PolyUOp *schedule = poly_test_linear_values(ctx, &physical, 1, &scheduled_out);
  ASSERT_NOT_NULL(schedule);
  ASSERT_NOT_NULL(scheduled_out);
  ASSERT_INT_EQ(schedule->n_src, 3);
  int copy_calls = 0;
  for (int i = 0; i < 2; i++) {
    PolyUOp *body = poly_test_linear_call_body(schedule, i);
    ASSERT_NOT_NULL(body);
    if (poly_test_linear_call_is_copy(schedule, i)) copy_calls++;
  }
  ASSERT_INT_EQ(copy_calls, 2);
  PolyUOp *compute = poly_test_linear_call_body(schedule, 2);
  ASSERT_NOT_NULL(compute);
  ASSERT_INT_EQ(count_root_ops(ctx, compute, POLY_OP_RESHAPE), 0);
  ASSERT_INT_EQ(count_root_ops(ctx, compute, POLY_OP_SHRINK), 0);
  ASSERT_INT_EQ(count_root_ops(ctx, compute, POLY_OP_PAD), 0);
  ASSERT_INT_EQ(count_root_ops(ctx, compute, POLY_OP_BUFFER), 0);
  ASSERT_INT_EQ(count_root_ops(ctx, compute, POLY_OP_UNIQUE), 0);
  ASSERT_INT_EQ(count_root_ops(ctx, compute, POLY_OP_PARAM), 3);
  ASSERT_TRUE(count_root_ops(ctx, compute, POLY_OP_INDEX) > 0);
  ASSERT_TRUE(count_root_ops(ctx, compute, POLY_OP_STORE) > 0);

  poly_ctx_reset_counters(ctx);
  PolyTensor *realized = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &result, 1, &realized), 0);
  ASSERT_NOT_NULL(realized);
  PolyCtxStats stats = {0};
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &stats), 0);
  ASSERT_TRUE(stats.kernel_count == 3);

  /* tinygrad Tensor._buffer and Polygrad Tensor._buffer both materialize a
   * non-contiguous realized view before flat byte readback. Keep that readback
   * call outside the three-call execution-topology assertion above. */
  PolyUOp *read_uop = poly_contiguous(ctx, poly_tensor_uop(realized));
  PolyUOp *read_out = NULL;
  ASSERT_NOT_NULL(read_uop);
  ASSERT_INT_EQ(poly_realize_uops(ctx, &read_uop, 1, &read_out), 0);
  const PolyUOp *identity = poly_uop_get_buffer_identity(read_out);
  ASSERT_NOT_NULL(identity);
  uint8_t values[2] = {0};
  ASSERT_INT_EQ(poly_buffer_read(ctx, (PolyUOp *)identity, values, sizeof(values)), 0);
  ASSERT_INT_EQ(values[0], 4);
  ASSERT_INT_EQ(values[1], 17);

  poly_ctx_destroy(ctx);
  ASSERT_INT_EQ(unlink(path), 0);
  PASS();
}

TEST(realize, nested_disk_view_copies_hold_contiguous_reduction_without_extra_copy) {
  char path[256];
  snprintf(path, sizeof(path), "temp/polygrad_disk_reduce_%ld.bin", (long)getpid());
  FILE *file = fopen(path, "wb");
  ASSERT_NOT_NULL(file);
  uint8_t file_data[] = {1, 10, 11, 12, 99, 2, 20, 21, 22};
  ASSERT_TRUE(fwrite(file_data, 1, sizeof(file_data), file) == sizeof(file_data));
  ASSERT_INT_EQ(fclose(file), 0);

  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  int uint8_id = poly_dtype_id_by_name("uint8");
  ASSERT_TRUE(uint8_id >= 0);
  PolyUOp *disk_buffer = poly_buffer_from_file(ctx, path, uint8_id);
  ASSERT_NOT_NULL(disk_buffer);

  int64_t left_bounds[1][2] = {{0, 4}};
  int64_t right_bounds[1][2] = {{5, 9}};
  int64_t row_shape[2] = {1, 4};
  PolyUOp *left = poly_reshape(ctx, poly_shrink(ctx, disk_buffer, left_bounds, 1), row_shape, 2);
  PolyUOp *right = poly_reshape(ctx, poly_shrink(ctx, disk_buffer, right_bounds, 1), row_shape, 2);
  ASSERT_NOT_NULL(left);
  ASSERT_NOT_NULL(right);

  PolyTensor *left_disk =
      poly_tensor_create_with_roots(ctx, left, left, POLY_TENSOR_VALUE, POLY_DEVICE_DISK);
  PolyTensor *right_disk =
      poly_tensor_create_with_roots(ctx, right, right, POLY_TENSOR_VALUE, POLY_DEVICE_DISK);
  ASSERT_NOT_NULL(left_disk);
  ASSERT_NOT_NULL(right_disk);
  PolyTensor *left_interp = poly_tensor_to_device(ctx, left_disk, POLY_DEVICE_INTERP);
  PolyTensor *right_interp = poly_tensor_to_device(ctx, right_disk, POLY_DEVICE_INTERP);
  ASSERT_NOT_NULL(left_interp);
  ASSERT_NOT_NULL(right_interp);

  PolyUOp *logical_rows[2] = {
      poly_tensor_uop_logical(left_interp), poly_tensor_uop_logical(right_interp)};
  PolyUOp *physical_rows[2] = {poly_tensor_uop(left_interp), poly_tensor_uop(right_interp)};
  PolyUOp *joined = poly_cat(ctx, logical_rows, 2, 0);
  PolyUOp *joined_physical = poly_cat(ctx, physical_rows, 2, 0);
  PolyUOp *joined_i32 = poly_cast(ctx, joined, POLY_INT32);
  PolyUOp *joined_i32_physical = poly_cast(ctx, joined_physical, POLY_INT32);
  int64_t reduce_axis[1] = {1};
  PolyUOp *reduced = poly_reduce_axis(ctx, POLY_OP_ADD, joined_i32, reduce_axis, 1);
  PolyUOp *reduced_physical =
      poly_reduce_axis(ctx, POLY_OP_ADD, joined_i32_physical, reduce_axis, 1);
  PolyUOp *materialized = poly_contiguous(ctx, reduced);
  PolyUOp *materialized_physical = poly_contiguous(ctx, reduced_physical);
  PolyUOp *value = poly_add(ctx, materialized, poly_const_int(ctx, 1));
  PolyUOp *value_physical = poly_add(ctx, materialized_physical, poly_const_int(ctx, 1));
  ASSERT_NOT_NULL(value);
  ASSERT_NOT_NULL(value_physical);
  PolyTensor *result = poly_tensor_create_with_roots(
      ctx, value, value_physical, POLY_TENSOR_VALUE, POLY_DEVICE_INTERP
  );
  ASSERT_NOT_NULL(result);

  PolyUOp *physical = poly_tensor_uop_physical(result);
  PolyUOp *scheduled_out = NULL;
  PolyUOp *schedule = poly_test_linear_values(ctx, &physical, 1, &scheduled_out);
  ASSERT_NOT_NULL(schedule);
  ASSERT_NOT_NULL(scheduled_out);
  ASSERT_INT_EQ(schedule->n_src, 4);
  /* Pinned tinygrad callify exposes the CONTIGUOUS buffer as a CALL argument;
   * create_linear_with_vars therefore holds it out of the memory planner. */
  ASSERT_FALSE(poly_test_linear_call_is_copy(schedule, 2));
  ASSERT_FALSE(poly_test_linear_call_is_copy(schedule, 3));
  PolyUOp *reduction = poly_test_linear_call_buffer(schedule, 2, 0);
  ASSERT_NOT_NULL(reduction);
  ASSERT_PTR_EQ(poly_test_linear_call_buffer(schedule, 3, 1), reduction);
  ASSERT_TRUE(poly_buffer_get(ctx, reduction) == NULL);
  ASSERT_FALSE(poly_buffer_is_allocated(ctx, reduction));

  int reduction_writes = 0;
  int reduction_reads = 0;
  for (int k = 0; k < schedule->n_src; k++) {
    PolyUOp *call = poly_test_linear_call(schedule, k);
    int n_args = poly_call_n_buffer_args(call);
    bool *outs = calloc((size_t)n_args, sizeof(*outs));
    bool *ins = calloc((size_t)n_args, sizeof(*ins));
    ASSERT_TRUE(n_args == 0 || (outs && ins));
    ASSERT_INT_EQ(poly_call_get_outs_ins(ctx, call, outs, ins, n_args), 0);
    for (int i = 0; i < n_args; i++) {
      if (poly_call_buffer_arg(call, i) != reduction) continue;
      reduction_writes += outs[i] ? 1 : 0;
      reduction_reads += ins[i] ? 1 : 0;
    }
    free(outs);
    free(ins);
  }
  ASSERT_INT_EQ(reduction_writes, 1);
  ASSERT_INT_EQ(reduction_reads, 1);

  ASSERT_INT_EQ(poly_run_linear(ctx, schedule, NULL, 0, NULL, 0, true, false, false), 0);
  PolyBuffer *reduction_storage = poly_buffer_get(ctx, reduction);
  ASSERT_NOT_NULL(reduction_storage);
  ASSERT_TRUE(reduction_storage->valid);
  ASSERT_TRUE(poly_buffer_is_allocated(ctx, reduction));
  const PolyUOp *identity = poly_uop_get_buffer_identity(scheduled_out);
  ASSERT_NOT_NULL(identity);
  int32_t values[2] = {0};
  ASSERT_INT_EQ(poly_buffer_read(ctx, (PolyUOp *)identity, values, sizeof(values)), 0);
  ASSERT_INT_EQ(values[0], 35);
  ASSERT_INT_EQ(values[1], 66);

  poly_ctx_destroy(ctx);
  ASSERT_INT_EQ(unlink(path), 0);
  PASS();
}
