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
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static PolyBuffer *realized_buffer(PolyCtx *ctx, PolyUOp *realized) {
  const PolyUOp *buf_uop = poly_uop_get_buffer_identity(realized);
  return buf_uop ? poly_buffer_get(ctx, (PolyUOp *)buf_uop) : NULL;
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
         !u->dtype.is_ptr && !u->arg.param->name && u->n_src == 1 && u->src[0] &&
         u->src[0]->op == POLY_OP_STACK;
}

static int count_shaped_value_params(PolyCtx *ctx, PolyUOp *root) {
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, root, &n_topo);
  int count = 0;
  for (int i = 0; i < n_topo; i++)
    if (is_shaped_value_param(topo[i])) count++;
  return count;
}

static int count_copy_to_device(PolyCtx *ctx, PolyUOp *root, PolyDevice device) {
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, root, &n_topo);
  int count = 0;
  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    if (u && u->op == POLY_OP_COPY && u->n_src >= 2 &&
        poly_device_from_device_uop(u->src[1]) == device)
      count++;
  }
  return count;
}

static PolyTensor *custom_add_tensor(
    PolyCtx *ctx,
    PolyTensor *a,
    PolyTensor *b,
    PolyUOp **out_buf
) {
  /* Pinned tinygrad UOp.custom_kernel receives deviceful sources
   * (uop/ops.py:1093-1097). Mirror the frontend boundary: portable logical
   * storage plus an exact deviceful physical output, with one opaque body. */
  PolyUOp *c_logical = poly_buffer(ctx, POLY_FLOAT32, 4);
  PolyUOp *c_physical = poly_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CPU);
  if (out_buf) *out_buf = c_physical;
  PolyUOp *pc = poly_uop_placeholder_like(ctx, c_logical, 0);
  PolyUOp *pa = poly_uop_placeholder_like(ctx, poly_tensor_uop(a), 1);
  PolyUOp *pb = poly_uop_placeholder_like(ctx, poly_tensor_uop(b), 2);
  PolyUOp *r = poly_uop_range(ctx, 4, 0, POLY_AXIS_GLOBAL);
  PolyUOp *idxs[1] = {r};
  PolyUOp *ci = poly_uop_index(ctx, pc, idxs, 1, 0);
  PolyUOp *ai = poly_uop_index(ctx, pa, idxs, 1, 0);
  PolyUOp *bi = poly_uop_index(ctx, pb, idxs, 1, 0);
  PolyUOp *av = poly_uop_load(ctx, ai);
  PolyUOp *bv = poly_uop_load(ctx, bi);
  PolyUOp *sum = poly_alu2(ctx, POLY_OP_ADD, av, bv);
  PolyUOp *store = poly_uop_store(ctx, ci, sum);
  PolyUOp *end = poly_uop_end(ctx, store, &r, 1);
  PolyUOp *sink = poly_uop_sink(ctx, &end, 1);
  PolyUOp *logical_args[3] = {c_logical, poly_tensor_uop(a), poly_tensor_uop(b)};
  PolyUOp *physical_args[3] = {c_physical, poly_tensor_uop(a), poly_tensor_uop(b)};
  PolyUOp *logical_call = poly_uop_call(ctx, sink, logical_args, 3);
  PolyUOp *physical_call = poly_uop_call(ctx, sink, physical_args, 3);
  return poly_tensor_create_with_roots(
      ctx, poly_uop_after(ctx, c_logical, logical_call),
      poly_uop_after(ctx, c_physical, physical_call), POLY_TENSOR_VALUE, POLY_DEVICE_CPU
  );
}

static PolyTensor *custom_summary_tensor(
    PolyCtx *ctx,
    PolyTensor *a,
    PolyTensor *b,
    PolyUOp **out_buf
) {
  PolyUOp *out_logical = poly_buffer(ctx, POLY_FLOAT32, 2);
  PolyUOp *out_physical = poly_buffer_on_device(ctx, POLY_FLOAT32, 2, POLY_DEVICE_CPU);
  if (out_buf) *out_buf = out_physical;
  PolyUOp *pout = poly_uop_placeholder_like(ctx, out_logical, 0);
  PolyUOp *pa = poly_uop_placeholder_like(ctx, poly_tensor_uop(a), 1);
  PolyUOp *pb = poly_uop_placeholder_like(ctx, poly_tensor_uop(b), 2);
  PolyUOp *c = poly_uop_range(ctx, 2, 0, POLY_AXIS_GLOBAL);
  PolyUOp *r = poly_uop_range(ctx, 4, 1, POLY_AXIS_LOOP);
  PolyUOp *four = poly_const_int(ctx, 4);
  PolyUOp *offset = poly_alu2(ctx, POLY_OP_ADD, poly_alu2(ctx, POLY_OP_MUL, c, four), r);
  PolyUOp *out_idx[1] = {c};
  PolyUOp *a_idx[1] = {offset};
  PolyUOp *b_idx[1] = {r};
  PolyUOp *prod = poly_alu2(
      ctx, POLY_OP_MUL, poly_uop_index(ctx, pa, a_idx, 1, 0), poly_uop_index(ctx, pb, b_idx, 1, 0)
  );
  PolyUOp *sum = poly_uop_reduce(ctx, POLY_OP_ADD, prod, &r, 1);
  PolyUOp *store = poly_uop_store(ctx, poly_uop_index(ctx, pout, out_idx, 1, 0), sum);
  PolyUOp *end = poly_uop_end(ctx, store, &c, 1);
  PolyUOp *sink = poly_uop_sink(ctx, &end, 1);
  PolyUOp *logical_args[3] = {out_logical, poly_tensor_uop(a), poly_tensor_uop(b)};
  PolyUOp *physical_args[3] = {out_physical, poly_tensor_uop(a), poly_tensor_uop(b)};
  PolyUOp *logical_call = poly_uop_call(ctx, sink, logical_args, 3);
  PolyUOp *physical_call = poly_uop_call(ctx, sink, physical_args, 3);
  return poly_tensor_create_with_roots(
      ctx, poly_uop_after(ctx, out_logical, logical_call),
      poly_uop_after(ctx, out_physical, physical_call), POLY_TENSOR_VALUE, POLY_DEVICE_CPU
  );
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
  PolyUOp *out0_logical = poly_buffer(ctx, POLY_FLOAT32, 2);
  PolyUOp *out1_logical = poly_buffer(ctx, POLY_FLOAT32, 2);
  PolyUOp *out0_physical = poly_buffer_on_device(ctx, POLY_FLOAT32, 2, POLY_DEVICE_CPU);
  PolyUOp *out1_physical = poly_buffer_on_device(ctx, POLY_FLOAT32, 2, POLY_DEVICE_CPU);
  if (out0_buf) *out0_buf = out0_physical;
  if (out1_buf) *out1_buf = out1_physical;
  PolyUOp *pout0 = poly_uop_placeholder_like(ctx, out0_logical, 0);
  PolyUOp *pout1 = poly_uop_placeholder_like(ctx, out1_logical, 1);
  PolyUOp *pa = poly_uop_placeholder_like(ctx, poly_tensor_uop(a), 2);
  PolyUOp *pb = poly_uop_placeholder_like(ctx, poly_tensor_uop(b), 3);
  PolyUOp *c = poly_uop_range(ctx, 2, 0, POLY_AXIS_GLOBAL);
  PolyUOp *r = poly_uop_range(ctx, 4, 1, POLY_AXIS_LOOP);
  PolyUOp *four = poly_const_int(ctx, 4);
  PolyUOp *offset = poly_alu2(ctx, POLY_OP_ADD, poly_alu2(ctx, POLY_OP_MUL, c, four), r);
  PolyUOp *out_idx[1] = {c};
  PolyUOp *a_idx[1] = {offset};
  PolyUOp *b_idx[1] = {r};
  PolyUOp *term = poly_uop_index(ctx, pa, a_idx, 1, 0);
  PolyUOp *s0 = poly_uop_reduce(ctx, POLY_OP_ADD, term, &r, 1);
  PolyUOp *prod = poly_alu2(ctx, POLY_OP_MUL, term, poly_uop_index(ctx, pb, b_idx, 1, 0));
  PolyUOp *s1 = poly_uop_reduce(ctx, POLY_OP_ADD, prod, &r, 1);
  PolyUOp *st0 = poly_uop_store(ctx, poly_uop_index(ctx, pout0, out_idx, 1, 0), s0);
  PolyUOp *st1 = poly_uop_store(ctx, poly_uop_index(ctx, pout1, out_idx, 1, 0), s1);
  PolyUOp *end0 = poly_uop_end(ctx, st0, &c, 1);
  PolyUOp *end1 = poly_uop_end(ctx, st1, &c, 1);
  PolyUOp *ends[2] = {end0, end1};
  PolyUOp *sink = poly_uop_sink(ctx, ends, 2);
  PolyUOp *logical_args[4] = {out0_logical, out1_logical, poly_tensor_uop(a), poly_tensor_uop(b)};
  PolyUOp *physical_args[4] = {
      out0_physical, out1_physical, poly_tensor_uop(a), poly_tensor_uop(b)};
  PolyUOp *logical_call = poly_uop_call(ctx, sink, logical_args, 4);
  PolyUOp *physical_call = poly_uop_call(ctx, sink, physical_args, 4);
  *out0_tensor = poly_tensor_create_with_roots(
      ctx, poly_uop_after(ctx, out0_logical, logical_call),
      poly_uop_after(ctx, out0_physical, physical_call), POLY_TENSOR_VALUE, POLY_DEVICE_CPU
  );
  *out1_tensor = poly_tensor_create_with_roots(
      ctx, poly_uop_after(ctx, out1_logical, logical_call),
      poly_uop_after(ctx, out1_physical, physical_call), POLY_TENSOR_VALUE, POLY_DEVICE_CPU
  );
}

static PolyTensor *custom_grouped_intercept_summary_tensor(
    PolyCtx *ctx,
    PolyTensor *x,
    PolyTensor *y,
    PolyUOp **out_buf
) {
  PolyUOp *out_logical = poly_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *out_physical = poly_buffer_on_device(ctx, POLY_FLOAT32, 10, POLY_DEVICE_CPU);
  if (out_buf) *out_buf = out_physical;
  PolyUOp *pout = poly_uop_placeholder_like(ctx, out_logical, 0);
  PolyUOp *px = poly_uop_placeholder_like(ctx, poly_tensor_uop(x), 1);
  PolyUOp *py = poly_uop_placeholder_like(ctx, poly_tensor_uop(y), 2);
  PolyUOp *c = poly_uop_range(ctx, 2, 0, POLY_AXIS_LOOP);
  PolyUOp *r = poly_uop_range(ctx, 128, 1, POLY_AXIS_REDUCE);
  PolyUOp *rows = poly_const_int(ctx, 128);
  PolyUOp *two = poly_const_int(ctx, 2);
  PolyUOp *offset = poly_alu2(ctx, POLY_OP_ADD, poly_alu2(ctx, POLY_OP_MUL, c, rows), r);
  PolyUOp *x_idx[1] = {offset};
  PolyUOp *y_idx[1] = {r};
  PolyUOp *xv = poly_uop_index(ctx, px, x_idx, 1, 0);
  PolyUOp *yv = poly_uop_index(ctx, py, y_idx, 1, 0);
  PolyUOp *one = poly_const_float(ctx, 1.0f);

  PolyUOp *vals[5];
  vals[0] = poly_uop_reduce(ctx, POLY_OP_ADD, one, &r, 1);
  vals[1] = poly_uop_reduce(ctx, POLY_OP_ADD, xv, &r, 1);
  vals[2] = poly_uop_reduce(ctx, POLY_OP_ADD, poly_alu2(ctx, POLY_OP_MUL, xv, xv), &r, 1);
  vals[3] = poly_uop_reduce(ctx, POLY_OP_ADD, yv, &r, 1);
  vals[4] = poly_uop_reduce(ctx, POLY_OP_ADD, poly_alu2(ctx, POLY_OP_MUL, xv, yv), &r, 1);

  PolyUOp *stores[5];
  for (int i = 0; i < 5; i++) {
    PolyUOp *stat = poly_const_int(ctx, i);
    PolyUOp *idx = poly_alu2(ctx, POLY_OP_ADD, c, poly_alu2(ctx, POLY_OP_MUL, stat, two));
    PolyUOp *out_idx[1] = {idx};
    stores[i] = poly_uop_store(ctx, poly_uop_index(ctx, pout, out_idx, 1, 0), vals[i]);
  }
  PolyUOp *group = poly_uop_group(ctx, stores, 5);
  PolyUOp *end = poly_uop_end(ctx, group, &c, 1);
  PolyUOp *sink = poly_uop_sink(ctx, &end, 1);
  PolyUOp *logical_args[3] = {out_logical, poly_tensor_uop_logical(x), poly_tensor_uop_logical(y)};
  PolyUOp *physical_args[3] = {out_physical, poly_tensor_uop(x), poly_tensor_uop(y)};
  PolyUOp *logical_call = poly_uop_call(ctx, sink, logical_args, 3);
  PolyUOp *physical_call = poly_uop_call(ctx, sink, physical_args, 3);
  return poly_tensor_create_with_roots(
      ctx, poly_uop_after(ctx, out_logical, logical_call),
      poly_uop_after(ctx, out_physical, physical_call), POLY_TENSOR_VALUE, POLY_DEVICE_CPU
  );
}

TEST(realize, custom_call_copy_output_schedules_copy_call) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *c = poly_buffer(ctx, POLY_FLOAT32, 4);
  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, 4);
  PolyUOp *b = poly_buffer(ctx, POLY_FLOAT32, 4);
  PolyUOp *pc = poly_uop_placeholder_like(ctx, c, 0);
  PolyUOp *pa = poly_uop_placeholder_like(ctx, a, 1);
  PolyUOp *pb = poly_uop_placeholder_like(ctx, b, 2);
  PolyUOp *r = poly_uop_range(ctx, 4, 0, POLY_AXIS_GLOBAL);
  PolyUOp *idxs[1] = {r};
  PolyUOp *ci = poly_uop_index(ctx, pc, idxs, 1, 1);
  PolyUOp *ai = poly_uop_index(ctx, pa, idxs, 1, 1);
  PolyUOp *bi = poly_uop_index(ctx, pb, idxs, 1, 1);
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, ai, bi, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, ci, add, poly_arg_none());
  PolyUOp *end = poly_uop_end(ctx, store, &r, 1);
  PolyUOp *sink = poly_uop_sink(ctx, &end, 1);
  PolyUOp *args[3] = {c, a, b};
  PolyUOp *call = poly_uop_call(ctx, sink, args, 3);
  PolyUOp *after = poly_uop_after(ctx, c, call);
  PolyUOp *contig = poly_contiguous(ctx, after);
  PolyUOp *device = poly_uop0(ctx, POLY_OP_DEVICE, POLY_VOID, poly_arg_int(POLY_DEVICE_CUDA));
  PolyUOp *copy_src[2] = {contig, device};
  PolyUOp *copy = poly_uop(ctx, POLY_OP_COPY, POLY_FLOAT32, copy_src, 2, poly_arg_none());

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
  PolySchedule *sched = poly_schedule_with_vars(ctx, &copy, 1, &scheduled_out);
  ASSERT_NOT_NULL(sched);
  ASSERT_NOT_NULL(scheduled_out);
  ASSERT_INT_EQ(sched->template->n_calls, 2);
  ASSERT_FALSE(poly_schedule_call_is_copy(sched, 0));
  ASSERT_TRUE(poly_schedule_call_is_copy(sched, 1));
  ASSERT_INT_EQ(count_root_ops(ctx, poly_schedule_call_body(sched, 1), POLY_OP_CALL), 0);
  ASSERT_INT_EQ(count_root_ops(ctx, poly_schedule_call_body(sched, 1), POLY_OP_AFTER), 0);

  poly_schedule_free(sched);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, callification_defers_output_allocation_until_execution) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  poly_ctx_set_preferred_device(ctx, POLY_DEVICE_CPU);

  PolyUOp *src = poly_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CPU);
  float src_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  poly_buffer_set(ctx, src, src_data, sizeof(src_data), POLY_DEVICE_CPU);
  PolyUOp *value = poly_alu2(ctx, POLY_OP_ADD, src, poly_const_float(ctx, 1.0));
  PolyUOp *out = NULL;
  PolySchedule *sched = poly_schedule_with_vars(ctx, &value, 1, &out);
  ASSERT_NOT_NULL(sched);
  ASSERT_INT_EQ(sched->template->n_calls, 1);
  ASSERT_FALSE(poly_schedule_call_is_copy(sched, 0));
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

  ASSERT_INT_EQ(poly_run_schedule(ctx, sched, NULL, 0), 0);
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

  poly_schedule_free(sched);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, tensor_place_physicalizes_to_tinygrad_copy_device_graph) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer_f32(ctx, 4);
  PolyUOp *x = poly_alu2(ctx, POLY_OP_ADD, a, poly_const_float(ctx, 1.0));

  PolyTensor *base = poly_tensor_create(ctx, x, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  PolyTensor *cuda_tensor = poly_tensor_to_device(ctx, base, POLY_DEVICE_CUDA);
  PolyTensor *cpu_tensor = poly_tensor_to_device(ctx, base, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(base);
  ASSERT_NOT_NULL(cuda_tensor);
  ASSERT_NOT_NULL(cpu_tensor);
  ASSERT_PTR_EQ(poly_tensor_uop_logical(cuda_tensor), x);
  ASSERT_NOT_NULL(poly_tensor_uop_physical(cuda_tensor));
  ASSERT_PTR_EQ(poly_tensor_uop(cuda_tensor), poly_tensor_uop_physical(cuda_tensor));
  ASSERT_INT_EQ(cuda_tensor->role, POLY_TENSOR_PLACE);
  ASSERT_INT_EQ(poly_tensor_device(cuda_tensor), POLY_DEVICE_CUDA);

  PolyUOp *physical = poly_tensor_physicalize(ctx, cuda_tensor);
  ASSERT_NOT_NULL(physical);
  /* tinygrad tensor.py:327-335 stores copy_to_device immediately. The
   * physicalizer must therefore preserve this exact current COPY occurrence. */
  ASSERT_PTR_EQ(physical, poly_tensor_uop_physical(cuda_tensor));
  ASSERT_INT_EQ(physical->op, POLY_OP_COPY);
  ASSERT_INT_EQ(poly_uop_device(physical), POLY_DEVICE_CUDA);
  ASSERT_INT_EQ(count_root_ops(ctx, physical, POLY_OP_COPY), 2);
  ASSERT_INT_EQ(poly_device_from_device_uop(physical->src[1]), POLY_DEVICE_CUDA);

  PolyUOp *compute = physical->src[0];
  ASSERT_NOT_NULL(compute);
  ASSERT_INT_EQ(compute->op, POLY_OP_ADD);
  ASSERT_INT_EQ(poly_uop_device(compute), POLY_DEVICE_CPU);

  PolyUOp *input_copy = compute->src[0];
  ASSERT_NOT_NULL(input_copy);
  ASSERT_INT_EQ(input_copy->op, POLY_OP_COPY);
  ASSERT_INT_EQ(input_copy->n_src, 2);
  ASSERT_PTR_EQ(input_copy->src[0], a);
  ASSERT_INT_EQ(poly_device_from_device_uop(input_copy->src[1]), POLY_DEVICE_CPU);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, tensor_to_device_preserves_distinct_host_execution_target) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *buf = poly_buffer_f32(ctx, 4);
  PolyTensor *cpu = poly_tensor_create(ctx, buf, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
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

  PolyUOp *source_logical = poly_buffer(ctx, POLY_FLOAT32, 1);
  PolyUOp *source_physical = poly_buffer_on_device(ctx, POLY_FLOAT32, 1, POLY_DEVICE_CPU);
  float source_data[] = {3.0f};
  poly_buffer_set(ctx, source_physical, source_data, sizeof(source_data), POLY_DEVICE_CPU);
  PolyTensor *source = poly_tensor_create_with_roots(
      ctx, source_logical, source_physical, POLY_TENSOR_VALUE, POLY_DEVICE_CPU
  );

  PolyUOp *target_logical = poly_buffer(ctx, POLY_FLOAT32, 1);
  PolyUOp *target_physical = poly_buffer_on_device(ctx, POLY_FLOAT32, 1, POLY_DEVICE_INTERP);
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
  ASSERT_INT_EQ(poly_device_from_device_uop(physical->src[1]->src[1]->src[1]), POLY_DEVICE_INTERP);

  PolyTensor *realized = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &target, 1, &realized), 0);
  float out[] = {0.0f};
  READ_REALIZED_F32(ctx, poly_tensor_uop(realized), out, 1);
  ASSERT_FLOAT_EQ(out[0], 3.0f, 1e-6f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, tensor_assign_chains_logical_and_physical_versions_separately) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *target_logical = poly_buffer(ctx, POLY_FLOAT32, 1);
  PolyUOp *target_physical = poly_buffer_on_device(ctx, POLY_FLOAT32, 1, POLY_DEVICE_CPU);
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

  PolyUOp *buf = poly_buffer_f32(ctx, 6);
  float data[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  poly_buffer_set(ctx, buf, data, sizeof(data), POLY_DEVICE_HOST);
  PolyUOp *matrix = poly_reshape(ctx, buf, (int64_t[]){2, 3}, 2);
  PolyUOp *permuted = poly_permute(ctx, matrix, (int64_t[]){1, 0}, 2);
  PolyUOp *contiguous = poly_contiguous(ctx, permuted);
  PolyTensor *tensor = poly_tensor_create(ctx, contiguous, POLY_TENSOR_VALUE, POLY_DEVICE_INTERP);
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

  PolyUOp *source_buf = poly_buffer_f32(ctx, 3);
  float source_data[3] = {1.0f, 2.0f, 3.0f};
  poly_buffer_set(ctx, source_buf, source_data, sizeof(source_data), POLY_DEVICE_HOST);
  PolyTensor *source = poly_tensor_create(ctx, source_buf, POLY_TENSOR_VALUE, POLY_DEVICE_HOST);
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
    PolyUOp *value_buf = poly_buffer_f32(ctx, 3);
    float value_data[3] = {5.0f + (float)pass, 6.0f + (float)pass, 7.0f + (float)pass};
    poly_buffer_set(ctx, value_buf, value_data, sizeof(value_data), POLY_DEVICE_CPU);
    PolyTensor *value = poly_tensor_create(ctx, value_buf, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
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

  PolyUOp *source_buf = poly_buffer_f32(ctx, 3);
  float source_data[3] = {1.0f, 2.0f, 3.0f};
  poly_buffer_set(ctx, source_buf, source_data, sizeof(source_data), POLY_DEVICE_CPU);
  PolyTensor *source = poly_tensor_create(ctx, source_buf, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
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
    PolyUOp *value_buf = poly_buffer_f32(ctx, 3);
    float value_data[3] = {5.0f + (float)pass, 6.0f + (float)pass, 7.0f + (float)pass};
    poly_buffer_set(ctx, value_buf, value_data, sizeof(value_data), POLY_DEVICE_INTERP);
    PolyTensor *value = poly_tensor_create(ctx, value_buf, POLY_TENSOR_VALUE, POLY_DEVICE_INTERP);
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

  PolyUOp *source_buf = poly_buffer_f32(ctx, 2);
  float source_data[2] = {1.0f, 2.0f};
  poly_buffer_set(ctx, source_buf, source_data, sizeof(source_data), POLY_DEVICE_HOST);
  PolyTensor *source = poly_tensor_create(ctx, source_buf, POLY_TENSOR_VALUE, POLY_DEVICE_HOST);
  PolyTensor *placed = poly_tensor_to_device(ctx, source, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(source);
  ASSERT_NOT_NULL(placed);
  ASSERT_INT_EQ(placed->role, POLY_TENSOR_PLACE);
  ASSERT_PTR_EQ(poly_tensor_uop_logical(placed), source_buf);
  ASSERT_NOT_NULL(poly_tensor_uop_physical(placed));
  ASSERT_PTR_EQ(poly_tensor_uop(placed), poly_tensor_uop_physical(placed));

  /* tinygrad tensor.py:327-335 stores this creation COPY directly in
   * Tensor.uop. Polygrad retains the portable source separately but must use
   * the exact eager physical COPY for the first indirect consumer. */
  PolyUOp *projected = poly_tensor_physicalize(ctx, placed);
  ASSERT_NOT_NULL(projected);
  ASSERT_PTR_EQ(projected, poly_tensor_uop_physical(placed));
  ASSERT_INT_EQ(projected->op, POLY_OP_COPY);
  ASSERT_INT_EQ(projected->n_src, 2);
  ASSERT_PTR_EQ(projected->src[0], source_buf);
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
  ASSERT_PTR_EQ(poly_tensor_uop_logical(placed), source_buf);
  const PolyUOp *placed_identity = poly_uop_get_buffer_identity(poly_tensor_uop(placed));
  ASSERT_NOT_NULL(placed_identity);
  ASSERT_PTR_NEQ(placed_identity, source_buf);
  ASSERT_INT_EQ(poly_uop_device(poly_tensor_uop(placed)), POLY_DEVICE_CPU);
  PolyBuffer *placed_storage = poly_buffer_get(ctx, (PolyUOp *)placed_identity);
  ASSERT_NOT_NULL(placed_storage);
  ASSERT_TRUE(placed_storage->valid);
  ASSERT_INT_EQ(placed_storage->device, POLY_DEVICE_CPU);

  PolyCtxStats after_first = {0};
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &after_first), 0);
  ASSERT_INT_EQ(after_first.kernel_count, 2);

  PolyUOp *second_uop = poly_add(ctx, poly_tensor_uop(placed), poly_const_float(ctx, 2.0f));
  PolyTensor *second = poly_tensor_create(ctx, second_uop, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(second);
  PolyUOp *second_out = NULL;
  PolySchedule *second_schedule = poly_schedule_with_vars(ctx, &second_uop, 1, &second_out);
  ASSERT_NOT_NULL(second_schedule);
  ASSERT_NOT_NULL(second_out);
  ASSERT_INT_EQ(second_schedule->template->n_calls, 1);
  ASSERT_INT_EQ(count_root_ops(ctx, poly_schedule_call_body(second_schedule, 0), POLY_OP_COPY), 0);
  ASSERT_INT_EQ(poly_run_schedule(ctx, second_schedule, NULL, 0), 0);

  float second_values[2] = {0};
  READ_REALIZED_F32(ctx, second_out, second_values, 2);
  ASSERT_FLOAT_EQ(second_values[0], 3.0f, 1e-6f);
  ASSERT_FLOAT_EQ(second_values[1], 4.0f, 1e-6f);
  PolyCtxStats after_second = {0};
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &after_second), 0);
  ASSERT_INT_EQ(after_second.kernel_count, after_first.kernel_count + 1);
  ASSERT_PTR_EQ(poly_uop_get_buffer_identity(poly_tensor_uop(placed)), placed_identity);
  float source_values[2] = {0};
  ASSERT_INT_EQ(poly_buffer_read(ctx, source_buf, source_values, sizeof(source_values)), 0);
  ASSERT_FLOAT_EQ(source_values[0], source_data[0], 1e-6f);
  ASSERT_FLOAT_EQ(source_values[1], source_data[1], 1e-6f);

  poly_schedule_free(second_schedule);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, indirect_place_assignment_commits_value_after_consumer_run) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *source_buf = poly_buffer_f32(ctx, 1);
  float source_data[] = {2.0f};
  poly_buffer_set(ctx, source_buf, source_data, sizeof(source_data), POLY_DEVICE_HOST);
  PolyTensor *source = poly_tensor_create(ctx, source_buf, POLY_TENSOR_VALUE, POLY_DEVICE_HOST);
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
  PolyUOp *other_physical = poly_buffer_on_device(ctx, POLY_FLOAT32, 1, POLY_DEVICE_CPU);
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

  PolyUOp *source_buf = poly_buffer_f32(ctx, 1);
  float source_data[] = {0.0f};
  poly_buffer_set(ctx, source_buf, source_data, sizeof(source_data), POLY_DEVICE_HOST);
  PolyTensor *source = poly_tensor_create(ctx, source_buf, POLY_TENSOR_VALUE, POLY_DEVICE_HOST);
  PolyTensor *target = poly_tensor_to_device(ctx, source, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(source);
  ASSERT_NOT_NULL(target);

  PolyTensor *assignment_values[2] = {NULL, NULL};
  for (int assignment = 0; assignment < 2; assignment++) {
    PolyUOp *incremented = poly_add(ctx, poly_tensor_uop(target), poly_const_float(ctx, 8.0f));
    PolyTensor *value = poly_tensor_create(ctx, incremented, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
    ASSERT_NOT_NULL(value);
    assignment_values[assignment] = value;
    ASSERT_PTR_EQ(poly_tensor_assign(ctx, target, value), target);
  }
  ASSERT_INT_EQ(target->role, POLY_TENSOR_PLACE);
  ASSERT_INT_EQ(count_root_ops(ctx, poly_tensor_uop(target), POLY_OP_AFTER), 2);

  PolyUOp *placed = poly_tensor_physicalize(ctx, target);
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

  PolyUOp *consumer_uop = poly_add(ctx, poly_tensor_uop(target), poly_const_float(ctx, 5.0f));
  PolyTensor *consumer = poly_tensor_create(ctx, consumer_uop, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
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

  PolyUOp *source_buf = poly_buffer_f32(ctx, 1);
  float source_data[] = {0.0f};
  poly_buffer_set(ctx, source_buf, source_data, sizeof(source_data), POLY_DEVICE_HOST);
  PolyTensor *source = poly_tensor_create(ctx, source_buf, POLY_TENSOR_VALUE, POLY_DEVICE_HOST);
  PolyTensor *target = poly_tensor_to_device(ctx, source, POLY_DEVICE_CUDA);
  ASSERT_NOT_NULL(source);
  ASSERT_NOT_NULL(target);

  PolyUOp *first_value = poly_add(ctx, poly_tensor_uop(target), poly_const_float(ctx, 8.0f));
  PolyTensor *first_tensor =
      poly_tensor_create(ctx, first_value, POLY_TENSOR_VALUE, POLY_DEVICE_CUDA);
  ASSERT_NOT_NULL(first_tensor);
  ASSERT_PTR_EQ(poly_tensor_assign(ctx, target, first_tensor), target);
  PolyUOp *first_version = poly_tensor_uop(target);

  /* Visit the earlier version first, as the first random output does before a
   * later RNG draw reaches the live outer counter PLACE. */
  PolyUOp *earlier_read = poly_add(ctx, first_version, poly_const_float(ctx, 1.0f));
  PolyUOp *second_value = poly_add(ctx, poly_tensor_uop(target), poly_const_float(ctx, 8.0f));
  PolyTensor *second_tensor =
      poly_tensor_create(ctx, second_value, POLY_TENSOR_VALUE, POLY_DEVICE_CUDA);
  ASSERT_NOT_NULL(second_tensor);
  ASSERT_PTR_EQ(poly_tensor_assign(ctx, target, second_tensor), target);
  PolyUOp *consumer_uop = poly_add(ctx, earlier_read, poly_tensor_uop(target));
  PolyTensor *consumer = poly_tensor_create(ctx, consumer_uop, POLY_TENSOR_VALUE, POLY_DEVICE_CUDA);
  ASSERT_NOT_NULL(consumer);

  PolyUOp *physical = poly_tensor_physicalize(ctx, consumer);
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
  ASSERT_PTR_NEQ(inner_target, source_buf);

  /* Physicalizing the live PLACE directly and reaching only the saved first
   * version must select the same placement-owned COPY.  Pinned tinygrad puts
   * that COPY in the Tensor UOp before assign, so neither consumer source
   * order nor the presence of the later version can change this identity. */
  PolyUOp *live_physical = poly_tensor_physicalize(ctx, target);
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

  PolyTensor *earlier_only =
      poly_tensor_create(ctx, earlier_read, POLY_TENSOR_VALUE, POLY_DEVICE_CUDA);
  ASSERT_NOT_NULL(earlier_only);
  PolyUOp *earlier_physical = poly_tensor_physicalize(ctx, earlier_only);
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

TEST(realize, shared_allocator_place_versions_use_exact_execution_device) {
  for (int cpu_newest = 0; cpu_newest < 2; cpu_newest++) {
    PolyCtx *ctx = poly_ctx_new();
    ASSERT_NOT_NULL(ctx);

    float source_data[] = {0.0f};
    PolyUOp *source_buf = poly_buffer_f32(ctx, 1);
    ASSERT_NOT_NULL(source_buf);
    poly_buffer_set(ctx, source_buf, source_data, sizeof(source_data), POLY_DEVICE_HOST);
    PolyTensor *source = poly_tensor_create(ctx, source_buf, POLY_TENSOR_VALUE, POLY_DEVICE_HOST);
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
    ASSERT_PTR_EQ(
        poly_tensor_find_current(ctx, poly_tensor_uop(cpu), POLY_DEVICE_CPU, POLY_TENSOR_PLACE), cpu
    );
    ASSERT_PTR_EQ(
        poly_tensor_find_current(
            ctx, poly_tensor_uop(interp), POLY_DEVICE_INTERP, POLY_TENSOR_PLACE
        ),
        interp
    );

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
      PolyUOp *physical = poly_tensor_physicalize(ctx, consumer);
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
  PolyUOp *source_buf = poly_buffer_f32(ctx, 1);
  ASSERT_NOT_NULL(source_buf);
  poly_buffer_set(ctx, source_buf, source_data, sizeof(source_data), POLY_DEVICE_HOST);
  PolyTensor *source = poly_tensor_create(ctx, source_buf, POLY_TENSOR_VALUE, POLY_DEVICE_HOST);
  PolyTensor *target = poly_tensor_to_device(ctx, source, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(source);
  ASSERT_NOT_NULL(target);

  PolyUOp *first_value_uop = poly_add(ctx, poly_tensor_uop(target), poly_const_float(ctx, 8.0f));
  PolyTensor *first_value =
      poly_tensor_create(ctx, first_value_uop, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(first_value);
  ASSERT_PTR_EQ(poly_tensor_assign(ctx, target, first_value), target);
  PolyUOp *first_version = poly_tensor_uop(target);
  PolyTensor *saved_version =
      poly_tensor_create(ctx, first_version, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(saved_version);
  /* Pinned tinygrad retains one already-deviceful UOp graph. Polygrad's raw
   * UOp APIs are physical-only, so retain the exact placed counterpart while
   * the live PLACE still owns this version; uop_logical remains provenance. */
  PolyUOp *retained_first_physical = poly_tensor_physicalize(ctx, saved_version);
  ASSERT_NOT_NULL(retained_first_physical);

  PolyUOp *second_value_uop = poly_add(ctx, poly_tensor_uop(target), poly_const_float(ctx, 8.0f));
  PolyTensor *second_value =
      poly_tensor_create(ctx, second_value_uop, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(second_value);
  ASSERT_PTR_EQ(poly_tensor_assign(ctx, target, second_value), target);
  PolyUOp *consumer_uop = poly_add(ctx, poly_tensor_uop(target), poly_const_float(ctx, 5.0f));
  PolyTensor *consumer = poly_tensor_create(ctx, consumer_uop, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
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
  PolyTensor *late_current =
      poly_tensor_create(ctx, late_current_uop, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(late_current);
  PolyUOp *late_current_physical = poly_tensor_physicalize(ctx, late_current);
  ASSERT_NOT_NULL(late_current_physical);
  ASSERT_INT_EQ(count_root_ops(ctx, late_current_physical, POLY_OP_AFTER), 0);

  /* The raw physical UOp was retained outside the live Tensor map. Pinned
   * tinygrad callifies that same deviceful immutable graph again. */
  PolyUOp *late_raw_uop = poly_add(ctx, retained_first_physical, poly_const_float(ctx, 3.0f));
  PolyTensor *late_raw = poly_tensor_create(ctx, late_raw_uop, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(late_raw);
  PolyUOp *late_raw_physical = poly_tensor_physicalize(ctx, late_raw);
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
  PolySchedule *schedule = poly_schedule_with_vars(ctx, &late_raw_physical, 1, &scheduled_out);
  ASSERT_NOT_NULL(schedule);
  ASSERT_NOT_NULL(scheduled_out);
  ASSERT_INT_EQ(schedule->template->n_calls, 3);
  poly_schedule_free(schedule);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, aggregate_map_preserves_saved_place_version_topology) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  float source_data[] = {0.0f};
  PolyUOp *source_buf = poly_buffer_f32(ctx, 1);
  ASSERT_NOT_NULL(source_buf);
  poly_buffer_set(ctx, source_buf, source_data, sizeof(source_data), POLY_DEVICE_HOST);
  PolyTensor *source = poly_tensor_create(ctx, source_buf, POLY_TENSOR_VALUE, POLY_DEVICE_HOST);
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
  ASSERT_TRUE(poly_uop_reachable(ctx, retained_raw_first, source_buf));
  ASSERT_FALSE(poly_uop_reachable(ctx, retained_raw_first, (PolyUOp *)placed_identity));

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, aggregate_map_prefers_exact_root_placement_over_exact_descendant) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *state_buf = poly_buffer_on_device(ctx, POLY_FLOAT32, 1, POLY_DEVICE_CPU);
  PolyUOp *param_buf = poly_buffer_on_device(ctx, POLY_FLOAT32, 1, POLY_DEVICE_CPU);
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

  PolyUOp *cpu_device = poly_uop0(ctx, POLY_OP_DEVICE, POLY_VOID, poly_arg_int(POLY_DEVICE_CPU));
  PolyUOp *copy_src[2] = {portable_value, cpu_device};
  PolyUOp *placed_value = poly_uop(ctx, POLY_OP_COPY, POLY_FLOAT32, copy_src, 2, poly_arg_none());
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

  PolyTensor *state = poly_tensor_create(ctx, state_after, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  PolyTensor *param = poly_tensor_create(ctx, param_logical, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(state);
  ASSERT_NOT_NULL(param);

  PolyMap *placement_memo[POLY_DEVICE_DISK + 1] = {0};
  placement_memo[POLY_DEVICE_CPU] = poly_map_new(16);
  ASSERT_NOT_NULL(placement_memo[POLY_DEVICE_CPU]);
  poly_map_set(
      placement_memo[POLY_DEVICE_CPU], poly_ptr_hash(param_logical), param_logical, param_physical,
      poly_ptr_eq
  );
  PolyUOp *from[2] = {state_after, param_physical};
  PolyUOp *to[2] = {state_buf, param_buf};

  ASSERT_INT_EQ(
      poly_tensor_apply_realize_map(ctx, from, to, 2, POLY_DEVICE_AUTO, placement_memo), 0
  );
  ASSERT_PTR_EQ(poly_tensor_uop_logical(state), state_after);
  ASSERT_PTR_EQ(poly_tensor_uop_logical(param), param_logical);
  ASSERT_PTR_EQ(poly_tensor_uop(state), state_buf);
  ASSERT_PTR_EQ(poly_tensor_uop(param), param_buf);
  ASSERT_PTR_EQ(poly_tensor_uop_physical(state), state_buf);
  ASSERT_PTR_EQ(poly_tensor_uop_physical(param), param_buf);

  poly_tensor_physicalize_memo_destroy(placement_memo);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, aggregate_map_keeps_shared_storage_devices_distinct) {
  PolyCtx *ctx = poly_ctx_new();
  PolyMap *placement_memo[POLY_DEVICE_DISK + 1] = {0};
  ASSERT_NOT_NULL(ctx);

  float source_data[] = {0.0f};
  PolyUOp *source_buf = poly_buffer_f32(ctx, 1);
  ASSERT_NOT_NULL(source_buf);
  poly_buffer_set(ctx, source_buf, source_data, sizeof(source_data), POLY_DEVICE_HOST);
  PolyTensor *source = poly_tensor_create(ctx, source_buf, POLY_TENSOR_VALUE, POLY_DEVICE_HOST);
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
    replacement[i] = poly_buffer_on_device(ctx, POLY_FLOAT32, 1, devices[i]);
    ASSERT_NOT_NULL(saved[i]);
    ASSERT_NOT_NULL(replacement[i]);
  }
  ASSERT_INT_EQ(poly_tensor_physicalize_many(ctx, targets, 2, placed, placement_memo), 0);
  for (int i = 0; i < 2; i++) {
    ASSERT_NOT_NULL(placed[i]);
    ASSERT_INT_EQ(poly_uop_device(placed[i]), devices[i]);
  }
  ASSERT_PTR_EQ(logical[0], logical[1]);
  ASSERT_PTR_NEQ(placed[0], placed[1]);

  ASSERT_INT_EQ(
      poly_tensor_apply_realize_map(ctx, placed, replacement, 2, POLY_DEVICE_AUTO, placement_memo),
      0
  );
  for (int i = 0; i < 2; i++) {
    ASSERT_PTR_EQ(poly_tensor_uop_logical(targets[i]), logical[i]);
    ASSERT_PTR_EQ(poly_tensor_uop_logical(saved[i]), logical[i]);
    ASSERT_PTR_EQ(poly_tensor_uop_physical(targets[i]), replacement[i]);
    ASSERT_PTR_EQ(poly_tensor_uop_physical(saved[i]), replacement[i]);
    ASSERT_INT_EQ(poly_uop_device(poly_tensor_uop(targets[i])), devices[i]);
    ASSERT_PTR_NEQ(poly_tensor_uop(targets[i]), replacement[1 - i]);
  }

  poly_tensor_physicalize_memo_destroy(placement_memo);
  poly_ctx_destroy(ctx);
  PASS();
}

typedef struct {
  size_t arena_delta;
  size_t cse_delta;
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
  if (!ctx || (n_values > 0 && (!value_roots || !values)) ||
      (n_places > 0 && (!place_roots || !places)))
    goto cleanup;

  PolyUOp *old_leaf = poly_buffer_f32(ctx, 1);
  PolyUOp *new_leaf = poly_buffer_f32(ctx, 1);
  PolyUOp *affected_root = poly_add(ctx, old_leaf, poly_const_float(ctx, 1.0f));
  PolyTensor *affected = poly_tensor_create(ctx, affected_root, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  if (!old_leaf || !new_leaf || !affected_root || !affected) goto cleanup;

  for (int i = 0; i < n_values; i++) {
    value_roots[i] = poly_add(ctx, poly_buffer_f32(ctx, 1), poly_const_float(ctx, (double)(i + 2)));
    values[i] = poly_tensor_create(ctx, value_roots[i], POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
    if (!value_roots[i] || !values[i]) goto cleanup;
  }
  for (int i = 0; i < n_places; i++) {
    PolyUOp *source_root =
        poly_add(ctx, poly_buffer_f32(ctx, 1), poly_const_float(ctx, (double)(i + 2)));
    PolyTensor *source = poly_tensor_create(ctx, source_root, POLY_TENSOR_VALUE, POLY_DEVICE_HOST);
    places[i] = poly_tensor_to_device(ctx, source, POLY_DEVICE_CUDA);
    place_roots[i] = places[i] ? poly_tensor_uop_physical(places[i]) : NULL;
    if (!source_root || !source || !places[i] || !place_roots[i] ||
        poly_tensor_uop(places[i]) != place_roots[i])
      goto cleanup;
  }

  PolyCtxStats before = {0}, after = {0};
  PolyUOp *from[1] = {old_leaf};
  PolyUOp *to[1] = {new_leaf};
  if (poly_ctx_stats(ctx, &before) != 0 ||
      poly_tensor_apply_realize_map(ctx, from, to, 1, POLY_DEVICE_AUTO, NULL) != 0 ||
      poly_ctx_stats(ctx, &after) != 0 ||
      !poly_uop_reachable(ctx, poly_tensor_uop(affected), new_leaf) ||
      poly_uop_reachable(ctx, poly_tensor_uop(affected), old_leaf))
    goto cleanup;
  for (int i = 0; i < n_values; i++)
    if (poly_tensor_uop(values[i]) != value_roots[i] || poly_tensor_uop_physical(values[i]))
      goto cleanup;
  for (int i = 0; i < n_places; i++)
    if (poly_tensor_uop(places[i]) != place_roots[i] ||
        poly_tensor_uop_physical(places[i]) != place_roots[i])
      goto cleanup;

  out->arena_delta = after.arena_bytes - before.arena_bytes;
  out->cse_delta = after.cse_entries - before.cse_entries;
  rc = 0;

cleanup:
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
  PolyMap *placement_memo[POLY_DEVICE_DISK + 1] = {0};
  PolyUOp **unrelated_roots =
      n_unrelated > 0 ? calloc((size_t)n_unrelated, sizeof(*unrelated_roots)) : NULL;
  PolyTensor **unrelated_tensors =
      n_unrelated > 0 ? calloc((size_t)n_unrelated, sizeof(*unrelated_tensors)) : NULL;
  int rc = -1;
  if (!ctx || (n_unrelated > 0 && (!unrelated_roots || !unrelated_tensors))) goto cleanup;

  PolyUOp *input = poly_buffer_f32(ctx, 1);
  float input_value = 1.0f;
  if (!input) goto cleanup;
  poly_buffer_set(ctx, input, &input_value, sizeof(input_value), POLY_DEVICE_CPU);
  PolyUOp *logical_contiguous =
      poly_contiguous(ctx, poly_add(ctx, input, poly_const_float(ctx, 1.0f)));
  PolyTensor *boundary =
      poly_tensor_create(ctx, logical_contiguous, POLY_TENSOR_VALUE, POLY_DEVICE_CUDA);
  PolyTensor *boundary_roots[1] = {boundary};
  PolyUOp *placed_roots[1] = {NULL};
  if (!boundary ||
      poly_tensor_physicalize_many(ctx, boundary_roots, 1, placed_roots, placement_memo) != 0)
    goto cleanup;
  PolyUOp *placed_contiguous = placed_roots[0];
  PolyUOp *consumer = poly_add(ctx, logical_contiguous, poly_const_float(ctx, 5.0f));
  PolyTensor *consumer_tensor =
      poly_tensor_create(ctx, consumer, POLY_TENSOR_VALUE, POLY_DEVICE_CUDA);
  PolyUOp *replacement = poly_buffer_on_device(ctx, POLY_FLOAT32, 1, POLY_DEVICE_CUDA);
  if (!logical_contiguous || !boundary || !placed_contiguous || !consumer || !consumer_tensor ||
      !replacement || placed_contiguous->op != POLY_OP_CONTIGUOUS ||
      placed_contiguous == logical_contiguous)
    goto cleanup;

  for (int i = 0; i < n_unrelated; i++) {
    PolyUOp *other_input = share_input ? input : poly_buffer_f32(ctx, 1);
    PolyUOp *other_source = poly_add(ctx, other_input, poly_const_float(ctx, (double)(i + 11)));
    unrelated_roots[i] = poly_contiguous(ctx, other_source);
    unrelated_tensors[i] =
        poly_tensor_create(ctx, unrelated_roots[i], POLY_TENSOR_VALUE, POLY_DEVICE_CUDA);
    if (!other_input || !other_source || !unrelated_roots[i] || !unrelated_tensors[i]) goto cleanup;
  }

  PolyCtxStats before = {0}, after = {0};
  PolyUOp *from[1] = {placed_contiguous};
  PolyUOp *to[1] = {replacement};
  if (poly_ctx_stats(ctx, &before) != 0 ||
      poly_tensor_apply_realize_map(ctx, from, to, 1, POLY_DEVICE_CUDA, placement_memo) != 0 ||
      poly_ctx_stats(ctx, &after) != 0 || poly_tensor_uop_logical(boundary) != logical_contiguous ||
      poly_tensor_uop_physical(boundary) != replacement ||
      poly_tensor_uop_logical(consumer_tensor) != consumer ||
      !poly_uop_reachable(ctx, poly_tensor_uop_physical(consumer_tensor), replacement))
    goto cleanup;
  for (int i = 0; i < n_unrelated; i++)
    if (poly_tensor_uop(unrelated_tensors[i]) != unrelated_roots[i] ||
        poly_tensor_uop_physical(unrelated_tensors[i]))
      goto cleanup;

  out->arena_delta = after.arena_bytes - before.arena_bytes;
  out->cse_delta = after.cse_entries - before.cse_entries;
  rc = 0;

cleanup:
  poly_tensor_physicalize_memo_destroy(placement_memo);
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
  ASSERT_INT_EQ(shared_input.arena_delta, baseline.arena_delta);
  ASSERT_INT_EQ(shared_input.cse_delta, baseline.cse_delta);
  PASS();
}

TEST(realize, aggregate_map_projects_only_map_relevant_pending_place) {
  PolyCtx *ctx = poly_ctx_new();
  PolyMap *placement_memo[POLY_DEVICE_DISK + 1] = {0};
  ASSERT_NOT_NULL(ctx);

  PolyUOp *source_buffer = poly_buffer_f32(ctx, 1);
  float source_value = 1.0f;
  poly_buffer_set(ctx, source_buffer, &source_value, sizeof(source_value), POLY_DEVICE_HOST);
  PolyTensor *source = poly_tensor_create(ctx, source_buffer, POLY_TENSOR_VALUE, POLY_DEVICE_HOST);
  PolyTensor *target = poly_tensor_to_device(ctx, source, POLY_DEVICE_CUDA);
  ASSERT_NOT_NULL(source_buffer);
  ASSERT_NOT_NULL(source);
  ASSERT_NOT_NULL(target);
  ASSERT_NOT_NULL(poly_tensor_uop_physical(target));

  PolyTensor *project_tensors[1] = {target};
  PolyUOp *projected_roots[1] = {NULL};
  ASSERT_INT_EQ(
      poly_tensor_physicalize_many(ctx, project_tensors, 1, projected_roots, placement_memo), 0
  );
  PolyUOp *projected = projected_roots[0];
  PolyUOp *replacement = poly_buffer_on_device(ctx, POLY_FLOAT32, 1, POLY_DEVICE_CUDA);
  ASSERT_NOT_NULL(projected);
  ASSERT_NOT_NULL(replacement);
  ASSERT_PTR_EQ(projected, poly_tensor_uop_physical(target));
  ASSERT_EQ(projected->op, POLY_OP_COPY);
  ASSERT_PTR_EQ(projected->src[0], source_buffer);
  ASSERT_PTR_EQ(
      poly_map_get(
          placement_memo[POLY_DEVICE_CUDA], poly_ptr_hash(source_buffer), source_buffer, poly_ptr_eq
      ),
      projected
  );

  PolyUOp *from[1] = {projected};
  PolyUOp *to[1] = {replacement};
  ASSERT_INT_EQ(
      poly_tensor_apply_realize_map(ctx, from, to, 1, POLY_DEVICE_CUDA, placement_memo), 0
  );
  ASSERT_PTR_EQ(poly_tensor_uop_logical(target), source_buffer);
  ASSERT_PTR_EQ(poly_tensor_uop_physical(target), replacement);
  ASSERT_PTR_EQ(poly_tensor_uop(target), replacement);
  /* tinygrad tensor.py:202-206 publishes transform_to_call's BUFFER map
   * before scheduling. Polygrad's exact requested-device counterpart therefore
   * completes PLACE graph state here, while runtime allocation remains absent. */
  ASSERT_EQ(target->role, POLY_TENSOR_VALUE);
  ASSERT_PTR_EQ(target->source, NULL);
  ASSERT_FALSE(poly_buffer_is_allocated(ctx, replacement));

  poly_tensor_physicalize_memo_destroy(placement_memo);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, nested_shrink_assignment_materializes_before_live_store) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  poly_ctx_set_preferred_device(ctx, POLY_DEVICE_CPU);

  PolyUOp *counter_buffer = poly_buffer(ctx, POLY_FLOAT32, 2);
  float initial[2] = {1.0f, 2.0f};
  poly_buffer_set(ctx, counter_buffer, initial, sizeof(initial), POLY_DEVICE_CPU);
  PolyTensor *counter = poly_tensor_create(ctx, counter_buffer, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(counter);

  for (int update = 0; update < 2; update++) {
    PolyUOp *current = poly_tensor_uop(counter);
    PolyUOp *low = poly_shrink(ctx, current, (int64_t[1][2]){{0, 1}}, 1);
    PolyUOp *high = poly_shrink(ctx, current, (int64_t[1][2]){{1, 2}}, 1);
    PolyUOp *next_low = poly_add(ctx, low, poly_const_float(ctx, 1.0f));
    PolyUOp *parts[2] = {next_low, high};
    PolyUOp *next = poly_cat(ctx, parts, 2, 0);
    PolyTensor *value = poly_tensor_create(ctx, next, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
    ASSERT_NOT_NULL(low);
    ASSERT_NOT_NULL(high);
    ASSERT_NOT_NULL(next_low);
    ASSERT_NOT_NULL(next);
    ASSERT_NOT_NULL(value);
    ASSERT_PTR_EQ(poly_tensor_assign(ctx, counter, value), counter);
  }

  PolyUOp *physical = poly_tensor_physicalize(ctx, counter);
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
  PolySchedule *schedule = poly_schedule_with_vars(ctx, &physical, 1, &scheduled_out);
  ASSERT_NOT_NULL(schedule);
  ASSERT_NOT_NULL(scheduled_out);

  /* Pinned fix_store_hazard preserves this exact three-CALL topology:
   * first in-place update, temporary second value, temporary-to-counter
   * STORE. The temporary is a schedule-owned buffer shared by CALLs 1/2. */
  ASSERT_INT_EQ(schedule->template->n_calls, 3);
  ASSERT_INT_EQ(poly_schedule_call_n_buffer_args(schedule, 0), 1);
  ASSERT_INT_EQ(poly_schedule_call_n_buffer_args(schedule, 1), 2);
  ASSERT_INT_EQ(poly_schedule_call_n_buffer_args(schedule, 2), 2);
  int counter_slot = poly_schedule_call_buffer_slot(schedule, 0, 0);
  int temporary_slot = poly_schedule_call_buffer_slot(schedule, 1, 0);
  ASSERT_INT_EQ(poly_schedule_call_buffer_slot(schedule, 1, 1), counter_slot);
  ASSERT_INT_EQ(poly_schedule_call_buffer_slot(schedule, 2, 0), counter_slot);
  ASSERT_INT_EQ(poly_schedule_call_buffer_slot(schedule, 2, 1), temporary_slot);
  ASSERT_TRUE(counter_slot != temporary_slot);

  ASSERT_INT_EQ(poly_run_schedule(ctx, schedule, NULL, 0), 0);
  float result[2] = {0.0f, 0.0f};
  ASSERT_INT_EQ(
      poly_buffer_read(
          ctx, (PolyUOp *)poly_uop_get_buffer_identity(scheduled_out), result, sizeof(result)
      ),
      0
  );
  ASSERT_FLOAT_EQ(result[0], 3.0f, 1e-6f);
  ASSERT_FLOAT_EQ(result[1], 2.0f, 1e-6f);

  poly_schedule_free(schedule);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, creation_copy_nested_shrink_assignment_preserves_version_topology) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  poly_ctx_set_preferred_device(ctx, POLY_DEVICE_CPU);

  PolyUOp *source_buffer = poly_buffer(ctx, POLY_FLOAT32, 2);
  float initial[2] = {1.0f, 2.0f};
  poly_buffer_set(ctx, source_buffer, initial, sizeof(initial), POLY_DEVICE_HOST);
  PolyTensor *source = poly_tensor_create(ctx, source_buffer, POLY_TENSOR_VALUE, POLY_DEVICE_HOST);
  PolyTensor *counter = poly_tensor_to_device(ctx, source, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(source);
  ASSERT_NOT_NULL(counter);

  for (int update = 0; update < 2; update++) {
    PolyUOp *current = poly_tensor_uop(counter);
    PolyUOp *low = poly_shrink(ctx, current, (int64_t[1][2]){{0, 1}}, 1);
    PolyUOp *high = poly_shrink(ctx, current, (int64_t[1][2]){{1, 2}}, 1);
    PolyUOp *next_low = poly_add(ctx, low, poly_const_float(ctx, 1.0f));
    PolyUOp *parts[2] = {next_low, high};
    PolyUOp *next = poly_cat(ctx, parts, 2, 0);
    PolyTensor *value = poly_tensor_create(ctx, next, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
    ASSERT_NOT_NULL(value);
    ASSERT_PTR_EQ(poly_tensor_assign(ctx, counter, value), counter);
  }

  PolyUOp *physical = poly_tensor_physicalize(ctx, counter);
  ASSERT_NOT_NULL(physical);
  ASSERT_INT_EQ(count_root_ops(ctx, physical, POLY_OP_COPY), 1);
  ASSERT_INT_EQ(count_root_ops(ctx, physical, POLY_OP_AFTER), 2);

  PolyUOp *scheduled_out = NULL;
  PolySchedule *schedule = poly_schedule_with_vars(ctx, &physical, 1, &scheduled_out);
  ASSERT_NOT_NULL(schedule);
  ASSERT_NOT_NULL(scheduled_out);

  /* Pinned callify keeps the materialized AFTER in the executable graph while
   * its buffer_map points live tensors at the stripped buffer. Exact order:
   * creation COPY, first in-place update, temporary second value, final STORE. */
  ASSERT_INT_EQ(schedule->template->n_calls, 4);
  ASSERT_TRUE(poly_schedule_call_is_copy(schedule, 0));
  ASSERT_FALSE(poly_schedule_call_is_copy(schedule, 1));
  ASSERT_FALSE(poly_schedule_call_is_copy(schedule, 2));
  ASSERT_FALSE(poly_schedule_call_is_copy(schedule, 3));
  ASSERT_INT_EQ(poly_schedule_call_n_buffer_args(schedule, 0), 2);
  ASSERT_INT_EQ(poly_schedule_call_n_buffer_args(schedule, 1), 1);
  ASSERT_INT_EQ(poly_schedule_call_n_buffer_args(schedule, 2), 2);
  ASSERT_INT_EQ(poly_schedule_call_n_buffer_args(schedule, 3), 2);

  int counter_slot = poly_schedule_call_buffer_slot(schedule, 0, 0);
  int temporary_slot = poly_schedule_call_buffer_slot(schedule, 2, 0);
  ASSERT_INT_EQ(poly_schedule_call_buffer_slot(schedule, 1, 0), counter_slot);
  ASSERT_INT_EQ(poly_schedule_call_buffer_slot(schedule, 2, 1), counter_slot);
  ASSERT_INT_EQ(poly_schedule_call_buffer_slot(schedule, 3, 0), counter_slot);
  ASSERT_INT_EQ(poly_schedule_call_buffer_slot(schedule, 3, 1), temporary_slot);
  ASSERT_TRUE(counter_slot != temporary_slot);

  ASSERT_INT_EQ(poly_run_schedule(ctx, schedule, NULL, 0), 0);
  float result[2] = {0.0f, 0.0f};
  ASSERT_INT_EQ(
      poly_buffer_read(
          ctx, (PolyUOp *)poly_uop_get_buffer_identity(scheduled_out), result, sizeof(result)
      ),
      0
  );
  ASSERT_FLOAT_EQ(result[0], 3.0f, 1e-6f);
  ASSERT_FLOAT_EQ(result[1], 2.0f, 1e-6f);

  poly_schedule_free(schedule);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, creation_copy_cuda_roundtrip_keeps_exact_call_dependency_topology) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  float input[3] = {1.0f, 2.0f, 3.0f};
  PolyUOp *source_buffer = poly_buffer_f32(ctx, 3);
  ASSERT_NOT_NULL(source_buffer);
  poly_buffer_set(ctx, source_buffer, input, sizeof(input), POLY_DEVICE_HOST);

  PolyTensor *source = poly_tensor_create(ctx, source_buffer, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  PolyUOp *add = poly_add(ctx, source_buffer, poly_const_float(ctx, 1.0f));
  PolyTensor *computed = poly_tensor_create(ctx, add, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  PolyTensor *cuda = poly_tensor_to_device(ctx, computed, POLY_DEVICE_CUDA);
  PolyTensor *roundtrip = poly_tensor_to_device(ctx, cuda, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(source);
  ASSERT_NOT_NULL(add);
  ASSERT_NOT_NULL(computed);
  ASSERT_NOT_NULL(cuda);
  ASSERT_NOT_NULL(roundtrip);

  PolyUOp *physical = poly_tensor_uop_physical(roundtrip);
  ASSERT_NOT_NULL(physical);
  PolyUOp *scheduled_out = NULL;
  PolySchedule *schedule = poly_schedule_with_vars(ctx, &physical, 1, &scheduled_out);
  ASSERT_NOT_NULL(schedule);
  ASSERT_NOT_NULL(scheduled_out);

  /* Pinned callify.py:222-241 and schedule/rangeify.py:497-514 retain
   * creation AFTER as the compute CALL dependency. Exact order is creation
   * COPY, compute, CUDA COPY, CPU COPY. */
  ASSERT_INT_EQ(schedule->template->n_calls, 4);
  ASSERT_TRUE(poly_schedule_call_is_copy(schedule, 0));
  ASSERT_FALSE(poly_schedule_call_is_copy(schedule, 1));
  ASSERT_TRUE(poly_schedule_call_is_copy(schedule, 2));
  ASSERT_TRUE(poly_schedule_call_is_copy(schedule, 3));
  for (int k = 0; k < 4; k++)
    ASSERT_INT_EQ(poly_schedule_call_n_buffer_args(schedule, k), 2);

  int creation_out = poly_schedule_call_buffer_slot(schedule, 0, 0);
  int compute_out = poly_schedule_call_buffer_slot(schedule, 1, 0);
  int cuda_out = poly_schedule_call_buffer_slot(schedule, 2, 0);
  int roundtrip_out = poly_schedule_call_buffer_slot(schedule, 3, 0);
  ASSERT_INT_EQ(poly_schedule_call_buffer_slot(schedule, 1, 1), creation_out);
  ASSERT_INT_EQ(poly_schedule_call_buffer_slot(schedule, 2, 1), compute_out);
  ASSERT_INT_EQ(poly_schedule_call_buffer_slot(schedule, 3, 1), cuda_out);
  ASSERT_INT_EQ(schedule->template->buf_slots[creation_out].device, POLY_DEVICE_CPU);
  ASSERT_INT_EQ(schedule->template->buf_slots[compute_out].device, POLY_DEVICE_CPU);
  ASSERT_INT_EQ(schedule->template->buf_slots[cuda_out].device, POLY_DEVICE_CUDA);
  ASSERT_INT_EQ(schedule->template->buf_slots[roundtrip_out].device, POLY_DEVICE_CPU);
  ASSERT_PTR_EQ(
      schedule->template->buf_slots[roundtrip_out].buf_uop,
      poly_uop_get_buffer_identity(scheduled_out)
  );

  poly_schedule_free(schedule);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, tensor_value_buffer_identity_is_residency_not_copy) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *logical_buf = poly_buffer_f32(ctx, 6);
  PolyUOp *physical_buf = poly_buffer_on_device(ctx, POLY_FLOAT32, 6, POLY_DEVICE_CUDA);
  int64_t shape[2] = {2, 3};
  PolyUOp *logical = poly_reshape(ctx, logical_buf, shape, 2);
  PolyUOp *physical = poly_reshape(ctx, physical_buf, shape, 2);
  PolyTensor *direct =
      poly_tensor_create_with_roots(ctx, logical, physical, POLY_TENSOR_VALUE, POLY_DEVICE_CUDA);
  ASSERT_NOT_NULL(direct);

  PolyUOp *direct_physical = poly_tensor_physicalize(ctx, direct);
  ASSERT_NOT_NULL(direct_physical);
  ASSERT_PTR_EQ(poly_uop_get_buffer_identity(direct_physical), physical_buf);
  ASSERT_INT_EQ(count_root_ops(ctx, direct_physical, POLY_OP_COPY), 0);

  float host[6] = {0};
  poly_buffer_set(ctx, logical_buf, host, sizeof(host), POLY_DEVICE_CPU);
  PolyTensor *source = poly_tensor_create(ctx, logical, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  PolyTensor *placed = poly_tensor_to_device(ctx, source, POLY_DEVICE_CUDA);
  ASSERT_NOT_NULL(source);
  ASSERT_NOT_NULL(placed);
  PolyUOp *placed_physical = poly_tensor_physicalize(ctx, placed);
  ASSERT_NOT_NULL(placed_physical);
  ASSERT_INT_EQ(placed_physical->op, POLY_OP_COPY);
  ASSERT_INT_EQ(count_root_ops(ctx, placed_physical, POLY_OP_COPY), 1);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, tensor_place_fact_feeds_later_value_on_target_device) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer_f32(ctx, 4);
  PolyUOp *x = poly_alu2(ctx, POLY_OP_ADD, a, poly_const_float(ctx, 1.0));
  PolyTensor *x_cpu = poly_tensor_create(ctx, x, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  PolyTensor *x_cuda = poly_tensor_to_device(ctx, x_cpu, POLY_DEVICE_CUDA);
  ASSERT_NOT_NULL(x_cpu);
  ASSERT_NOT_NULL(x_cuda);

  PolyUOp *y = poly_alu2(ctx, POLY_OP_ADD, poly_tensor_uop(x_cuda), poly_const_float(ctx, 2.0));
  PolyTensor *y_cuda = poly_tensor_create(ctx, y, POLY_TENSOR_VALUE, POLY_DEVICE_CUDA);
  ASSERT_NOT_NULL(y_cuda);

  PolyUOp *physical = poly_tensor_physicalize(ctx, y_cuda);
  ASSERT_NOT_NULL(physical);
  ASSERT_INT_EQ(physical->op, POLY_OP_ADD);

  PolyUOp *placed_x = physical->src[0];
  ASSERT_NOT_NULL(placed_x);
  ASSERT_INT_EQ(placed_x->op, POLY_OP_COPY);
  ASSERT_INT_EQ(poly_device_from_device_uop(placed_x->src[1]), POLY_DEVICE_CUDA);
  ASSERT_INT_EQ(placed_x->src[0]->op, POLY_OP_ADD);
  ASSERT_INT_EQ(placed_x->src[0]->src[0]->op, POLY_OP_COPY);
  ASSERT_INT_EQ(poly_device_from_device_uop(placed_x->src[0]->src[0]->src[1]), POLY_DEVICE_CPU);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, tensor_placement_audit_reports_place_fact_for_later_value) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer_f32(ctx, 4);
  PolyUOp *x = poly_alu2(ctx, POLY_OP_ADD, a, poly_const_float(ctx, 1.0));
  PolyTensor *x_cpu = poly_tensor_create(ctx, x, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  PolyTensor *x_cuda = poly_tensor_to_device(ctx, x_cpu, POLY_DEVICE_CUDA);
  ASSERT_NOT_NULL(x_cpu);
  ASSERT_NOT_NULL(x_cuda);

  PolyUOp *x_cuda_current = poly_tensor_uop(x_cuda);
  PolyUOp *y_logical =
      poly_alu2(ctx, POLY_OP_ADD, poly_tensor_uop_logical(x_cuda), poly_const_float(ctx, 2.0));
  PolyUOp *y = poly_alu2(ctx, POLY_OP_ADD, x_cuda_current, poly_const_float(ctx, 2.0));
  PolyTensor *y_cuda =
      poly_tensor_create_with_roots(ctx, y_logical, y, POLY_TENSOR_VALUE, POLY_DEVICE_CUDA);
  ASSERT_NOT_NULL(y_cuda);

  PolyPlacementAudit audit;
  /* The audit mirrors the current-root index. tinygrad tensor.py:327-335 makes
   * the COPY current immediately, so query that exact occurrence rather than
   * its separate Polygrad logical provenance root. */
  ASSERT_INT_EQ(
      poly_tensor_placement_audit(ctx, y_cuda, x_cuda_current, POLY_DEVICE_CUDA, &audit), 0
  );
  ASSERT_PTR_EQ(audit.selected, y_cuda);
  ASSERT_PTR_EQ(audit.selected_current, y);
  ASSERT_PTR_EQ(audit.selected_logical, y_logical);
  ASSERT_PTR_EQ(audit.selected_physical, y);
  ASSERT_INT_EQ(audit.selected_role, POLY_TENSOR_VALUE);
  ASSERT_INT_EQ(audit.selected_device, POLY_DEVICE_CUDA);
  ASSERT_PTR_EQ(audit.query_current, x_cuda_current);
  ASSERT_INT_EQ(audit.query_device, POLY_DEVICE_CUDA);
  ASSERT_PTR_EQ(audit.place_fact, x_cuda);
  ASSERT_TRUE(audit.value_fact == NULL);
  ASSERT_PTR_EQ(audit.matched_fact, x_cuda);
  ASSERT_INT_EQ(audit.matched_role, POLY_TENSOR_PLACE);
  ASSERT_NOT_NULL(audit.physical_root);
  ASSERT_INT_EQ(audit.physical_root->op, POLY_OP_ADD);
  ASSERT_INT_EQ(audit.physical_root->src[0]->op, POLY_OP_COPY);
  ASSERT_INT_EQ(poly_device_from_device_uop(audit.physical_root->src[0]->src[1]), POLY_DEVICE_CUDA);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, tensor_place_fact_distinguishes_separate_realized_same_logical_sources) {
  PolyCtx *ctx = poly_ctx_new();

  /* x1 and x2 intentionally preserve the same logical ADD after realization,
   * but they materialize into distinct buffers. A later .to(CUDA) must carry
   * the selected tensor's current buffer root, not the shared logical ADD. */
  PolyUOp *a = poly_buffer_f32(ctx, 1);
  float *da = malloc(sizeof(float));
  ASSERT_NOT_NULL(da);
  da[0] = 1.0f;
  poly_buffer_set(ctx, a, da, sizeof(float), POLY_DEVICE_CPU);

  PolyUOp *expr = poly_alu2(ctx, POLY_OP_ADD, a, poly_const_float(ctx, 1.0f));
  PolyTensor *x1 = poly_tensor_create(ctx, expr, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(x1);
  PolyTensor *out = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &x1, 1, &out), 0);
  ASSERT_PTR_EQ(out, x1);
  PolyUOp *x1_buf = poly_tensor_uop(x1);
  ASSERT_TRUE(poly_uop_has_buffer_identity(x1_buf));

  PolyTensor *x2 = poly_tensor_create(ctx, expr, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
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

  PolyUOp *y_expr =
      poly_alu2(ctx, POLY_OP_ADD, poly_tensor_uop(x1_cuda), poly_const_float(ctx, 1.0f));
  PolyTensor *y = poly_tensor_create(ctx, y_expr, POLY_TENSOR_VALUE, POLY_DEVICE_CUDA);
  ASSERT_NOT_NULL(y);

  PolyUOp *physical = poly_tensor_physicalize(ctx, y);
  ASSERT_NOT_NULL(physical);
  ASSERT_INT_EQ(physical->op, POLY_OP_ADD);
  ASSERT_INT_EQ(physical->src[0]->op, POLY_OP_COPY);
  ASSERT_PTR_EQ(physical->src[0]->src[0], x1_buf);
  ASSERT_PTR_NEQ(physical->src[0]->src[0], x2_buf);
  ASSERT_INT_EQ(poly_device_from_device_uop(physical->src[0]->src[1]), POLY_DEVICE_CUDA);

  poly_ctx_destroy(ctx);
  free(da);
  PASS();
}

TEST(realize, tensor_placement_audit_prefers_nested_place_over_value_fact) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer_f32(ctx, 1);
  float da[] = {1.0f};
  poly_buffer_set(ctx, a, da, sizeof(da), POLY_DEVICE_CPU);

  PolyUOp *x_expr = poly_alu2(ctx, POLY_OP_ADD, a, poly_const_float(ctx, 1.0f));
  PolyUOp *x_buf = poly_buffer_f32(ctx, 1);
  float dx[] = {2.0f};
  poly_buffer_set(ctx, x_buf, dx, sizeof(dx), POLY_DEVICE_CPU);
  PolyTensor *x_cpu =
      poly_tensor_create_with_roots(ctx, x_expr, x_buf, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(x_cpu);

  PolyTensor *x_cuda = poly_tensor_to_device(ctx, x_cpu, POLY_DEVICE_CUDA);
  PolyTensor *x_cpu_again = poly_tensor_to_device(ctx, x_cuda, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(x_cuda);
  ASSERT_NOT_NULL(x_cpu_again);

  PolyUOp *x_cpu_again_current = poly_tensor_uop(x_cpu_again);
  PolyUOp *y_logical = poly_alu2(
      ctx, POLY_OP_ADD, poly_tensor_uop_logical(x_cpu_again), poly_const_float(ctx, 1.0f)
  );
  PolyUOp *y_expr = poly_alu2(ctx, POLY_OP_ADD, x_cpu_again_current, poly_const_float(ctx, 1.0f));
  PolyTensor *y_cpu =
      poly_tensor_create_with_roots(ctx, y_logical, y_expr, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(y_cpu);

  PolyPlacementAudit audit;
  ASSERT_INT_EQ(
      poly_tensor_placement_audit(ctx, y_cpu, x_cpu_again_current, POLY_DEVICE_CPU, &audit), 0
  );
  ASSERT_PTR_EQ(audit.selected, y_cpu);
  ASSERT_PTR_EQ(audit.query_current, x_cpu_again_current);
  ASSERT_PTR_EQ(audit.place_fact, x_cpu_again);
  ASSERT_TRUE(audit.value_fact == NULL);
  ASSERT_PTR_EQ(audit.matched_fact, x_cpu_again);
  ASSERT_INT_EQ(audit.matched_role, POLY_TENSOR_PLACE);
  ASSERT_NOT_NULL(audit.physical_root);
  ASSERT_INT_EQ(audit.physical_root->op, POLY_OP_ADD);
  ASSERT_INT_EQ(audit.physical_root->src[0]->op, POLY_OP_COPY);
  ASSERT_INT_EQ(poly_device_from_device_uop(audit.physical_root->src[0]->src[1]), POLY_DEVICE_CPU);
  ASSERT_INT_EQ(audit.physical_root->src[0]->src[0]->op, POLY_OP_COPY);
  ASSERT_INT_EQ(
      poly_device_from_device_uop(audit.physical_root->src[0]->src[0]->src[1]), POLY_DEVICE_CUDA
  );
  ASSERT_PTR_EQ(audit.physical_root->src[0]->src[0]->src[0], x_buf);

  /* The original CPU value remains separately discoverable at its own exact
   * current root; eager COPY occurrences must not alias the two index facts. */
  PolyPlacementAudit value_audit;
  ASSERT_INT_EQ(poly_tensor_placement_audit(ctx, y_cpu, x_buf, POLY_DEVICE_CPU, &value_audit), 0);
  ASSERT_TRUE(value_audit.place_fact == NULL);
  ASSERT_PTR_EQ(value_audit.value_fact, x_cpu);
  ASSERT_TRUE(value_audit.matched_fact == NULL);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, tensor_nested_place_physicalizes_copy_chain_without_self_selecting) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer_f32(ctx, 4);
  PolyUOp *x = poly_alu2(ctx, POLY_OP_ADD, a, poly_const_float(ctx, 1.0));
  PolyTensor *x_cpu = poly_tensor_create(ctx, x, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
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
  ASSERT_INT_EQ(poly_device_from_device_uop(eager_cuda->src[1]), POLY_DEVICE_CUDA);
  ASSERT_INT_EQ(eager_cpu->op, POLY_OP_COPY);
  ASSERT_INT_EQ(poly_device_from_device_uop(eager_cpu->src[1]), POLY_DEVICE_CPU);
  ASSERT_PTR_EQ(eager_cpu->src[0], eager_cuda);

  PolyUOp *physical = poly_tensor_physicalize(ctx, x_cpu_again);
  ASSERT_NOT_NULL(physical);
  ASSERT_PTR_EQ(physical, eager_cpu);
  ASSERT_INT_EQ(physical->op, POLY_OP_COPY);
  ASSERT_INT_EQ(poly_device_from_device_uop(physical->src[1]), POLY_DEVICE_CPU);
  ASSERT_INT_EQ(physical->src[0]->op, POLY_OP_COPY);
  ASSERT_INT_EQ(poly_device_from_device_uop(physical->src[0]->src[1]), POLY_DEVICE_CUDA);
  ASSERT_INT_EQ(physical->src[0]->src[0]->op, POLY_OP_ADD);
  ASSERT_INT_EQ(physical->src[0]->src[0]->src[0]->op, POLY_OP_COPY);
  ASSERT_INT_EQ(
      poly_device_from_device_uop(physical->src[0]->src[0]->src[0]->src[1]), POLY_DEVICE_CPU
  );

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, tensor_nested_place_fact_feeds_later_value_from_realized_source) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer_f32(ctx, 1);
  float da[] = {1.0f};
  poly_buffer_set(ctx, a, da, sizeof(da), POLY_DEVICE_CPU);

  PolyUOp *x_expr = poly_alu2(ctx, POLY_OP_ADD, a, poly_const_float(ctx, 1.0f));
  PolyUOp *x_buf = poly_buffer_f32(ctx, 1);
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

  /* The downstream value graph only sees the current realized buffer UOp.
   * The physicalizer must still recover the explicit CUDA->CPU roundtrip from
   * the selected PLACE fact, matching tinygrad's nested COPY graph. */
  PolyUOp *y_expr =
      poly_alu2(ctx, POLY_OP_ADD, poly_tensor_uop(x_cpu_again), poly_const_float(ctx, 1.0f));
  PolyTensor *y_cpu = poly_tensor_create(ctx, y_expr, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(y_cpu);

  PolyUOp *physical = poly_tensor_physicalize(ctx, y_cpu);
  ASSERT_NOT_NULL(physical);
  ASSERT_INT_EQ(physical->op, POLY_OP_ADD);
  ASSERT_INT_EQ(count_root_ops(ctx, physical, POLY_OP_COPY), 2);

  PolyUOp *copy_cpu = physical->src[0];
  ASSERT_NOT_NULL(copy_cpu);
  ASSERT_INT_EQ(copy_cpu->op, POLY_OP_COPY);
  ASSERT_INT_EQ(poly_device_from_device_uop(copy_cpu->src[1]), POLY_DEVICE_CPU);

  PolyUOp *copy_cuda = copy_cpu->src[0];
  ASSERT_NOT_NULL(copy_cuda);
  ASSERT_INT_EQ(copy_cuda->op, POLY_OP_COPY);
  ASSERT_INT_EQ(poly_device_from_device_uop(copy_cuda->src[1]), POLY_DEVICE_CUDA);
  ASSERT_PTR_EQ(copy_cuda->src[0], x_buf);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, tensor_place_assign_targets_copy_not_source_buffer) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer_f32(ctx, 3);
  float da[] = {1.0f, 2.0f, 3.0f};
  poly_buffer_set(ctx, a, da, sizeof(da), POLY_DEVICE_CPU);
  PolyTensor *a_cpu = poly_tensor_create(ctx, a, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  PolyTensor *a_cuda = poly_tensor_to_device(ctx, a_cpu, POLY_DEVICE_CUDA);
  ASSERT_NOT_NULL(a_cpu);
  ASSERT_NOT_NULL(a_cuda);

  PolyUOp *v = poly_buffer_f32(ctx, 3);
  float dv[] = {5.0f, 5.0f, 5.0f};
  poly_buffer_set(ctx, v, dv, sizeof(dv), POLY_DEVICE_CPU);
  PolyTensor *v_cpu = poly_tensor_create(ctx, v, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  PolyTensor *v_cuda = poly_tensor_to_device(ctx, v_cpu, POLY_DEVICE_CUDA);
  ASSERT_NOT_NULL(v_cpu);
  ASSERT_NOT_NULL(v_cuda);

  PolyTensor *assigned = poly_tensor_assign(ctx, a_cuda, v_cuda);
  ASSERT_PTR_EQ(assigned, a_cuda);
  PolyUOp *physical = poly_tensor_physicalize(ctx, assigned);
  ASSERT_NOT_NULL(physical);
  ASSERT_INT_EQ(physical->op, POLY_OP_AFTER);

  PolyUOp *target_copy = physical->src[0];
  PolyUOp *store = physical->src[1];
  ASSERT_NOT_NULL(target_copy);
  ASSERT_NOT_NULL(store);
  ASSERT_INT_EQ(target_copy->op, POLY_OP_COPY);
  ASSERT_PTR_EQ(target_copy->src[0], a);
  ASSERT_INT_EQ(poly_device_from_device_uop(target_copy->src[1]), POLY_DEVICE_CUDA);
  ASSERT_INT_EQ(store->op, POLY_OP_STORE);
  ASSERT_PTR_EQ(store->src[0], target_copy);
  ASSERT_INT_EQ(store->src[1]->op, POLY_OP_COPY);
  ASSERT_PTR_EQ(store->src[1]->src[0], v);
  ASSERT_INT_EQ(poly_device_from_device_uop(store->src[1]->src[1]), POLY_DEVICE_CUDA);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, tensor_assign_rejects_device_and_dtype_mismatch) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *target_buf = poly_buffer_f32(ctx, 1);
  float target_data[] = {1.0f};
  poly_buffer_set(ctx, target_buf, target_data, sizeof(target_data), POLY_DEVICE_CPU);
  PolyTensor *target_cpu = poly_tensor_create(ctx, target_buf, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(target_cpu);

  PolyUOp *value_buf = poly_buffer_f32(ctx, 1);
  float value_data[] = {5.0f};
  poly_buffer_set(ctx, value_buf, value_data, sizeof(value_data), POLY_DEVICE_CPU);
  PolyTensor *value_cpu = poly_tensor_create(ctx, value_buf, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  PolyTensor *value_cuda = poly_tensor_to_device(ctx, value_cpu, POLY_DEVICE_CUDA);
  ASSERT_NOT_NULL(value_cpu);
  ASSERT_NOT_NULL(value_cuda);

  /* tinygrad rejects CPU.assign(CUDA) before scheduling. Polygrad must not
   * silently lower the CUDA placement request back into a CPU STORE value. */
  ASSERT_TRUE(poly_tensor_assign(ctx, target_cpu, value_cuda) == NULL);

  PolyUOp *value64_buf = poly_buffer(ctx, POLY_FLOAT64, 1);
  double value64_data[] = {5.0};
  poly_buffer_set(ctx, value64_buf, value64_data, sizeof(value64_data), POLY_DEVICE_CPU);
  PolyTensor *value64_cpu =
      poly_tensor_create(ctx, value64_buf, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(value64_cpu);
  ASSERT_TRUE(poly_tensor_assign(ctx, target_cpu, value64_cpu) == NULL);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, tensor_place_assign_realize_materializes_value_without_mutating_source) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer_f32(ctx, 3);
  float *da = malloc(3 * sizeof(float));
  ASSERT_NOT_NULL(da);
  da[0] = 1.0f;
  da[1] = 2.0f;
  da[2] = 3.0f;
  poly_buffer_set(ctx, a, da, 3 * sizeof(float), POLY_DEVICE_HOST);
  PolyTensor *a_host = poly_tensor_create(ctx, a, POLY_TENSOR_VALUE, POLY_DEVICE_HOST);
  PolyTensor *a_cpu_copy = poly_tensor_to_device(ctx, a_host, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(a_host);
  ASSERT_NOT_NULL(a_cpu_copy);

  PolyUOp *v = poly_buffer_f32(ctx, 3);
  float *dv = malloc(3 * sizeof(float));
  ASSERT_NOT_NULL(dv);
  dv[0] = 5.0f;
  dv[1] = 6.0f;
  dv[2] = 7.0f;
  poly_buffer_set(ctx, v, dv, 3 * sizeof(float), POLY_DEVICE_CPU);
  PolyTensor *v_cpu = poly_tensor_create(ctx, v, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
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
  free(da);
  free(dv);
  PASS();
}

TEST(realize, prebuilt_graph_observes_placed_assign_only_after_assign_realize) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *source = poly_buffer_f32(ctx, 1);
  float source_data[] = {0.1f};
  poly_buffer_set(ctx, source, source_data, sizeof(source_data), POLY_DEVICE_HOST);
  PolyTensor *source_host = poly_tensor_create(ctx, source, POLY_TENSOR_VALUE, POLY_DEVICE_HOST);
  PolyTensor *placed = poly_tensor_to_device(ctx, source_host, POLY_DEVICE_INTERP);
  ASSERT_NOT_NULL(source_host);
  ASSERT_NOT_NULL(placed);

  PolyUOp *one = poly_const_float(ctx, 1.0);
  PolyUOp *prebuilt_uop = poly_alu2(ctx, POLY_OP_MUL, poly_tensor_uop(placed), one);
  PolyTensor *prebuilt =
      poly_tensor_create(ctx, prebuilt_uop, POLY_TENSOR_VALUE, POLY_DEVICE_INTERP);
  ASSERT_NOT_NULL(prebuilt);

  PolyUOp *value_buf = poly_buffer_f32(ctx, 1);
  float value_data[] = {0.2f};
  poly_buffer_set(ctx, value_buf, value_data, sizeof(value_data), POLY_DEVICE_HOST);
  PolyTensor *value_host = poly_tensor_create(ctx, value_buf, POLY_TENSOR_VALUE, POLY_DEVICE_HOST);
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

  PolyUOp *param_buf = poly_buffer_f32(ctx, 1);
  float param_data[] = {1.0f};
  poly_buffer_set(ctx, param_buf, param_data, sizeof(param_data), POLY_DEVICE_HOST);
  PolyTensor *param_host = poly_tensor_create(ctx, param_buf, POLY_TENSOR_VALUE, POLY_DEVICE_HOST);
  PolyTensor *param = poly_tensor_to_device(ctx, param_host, POLY_DEVICE_INTERP);

  PolyUOp *lr_buf = poly_buffer_f32(ctx, 1);
  float lr_data[] = {0.1f};
  poly_buffer_set(ctx, lr_buf, lr_data, sizeof(lr_data), POLY_DEVICE_HOST);
  PolyTensor *lr_host = poly_tensor_create(ctx, lr_buf, POLY_TENSOR_VALUE, POLY_DEVICE_HOST);
  PolyTensor *lr = poly_tensor_to_device(ctx, lr_host, POLY_DEVICE_INTERP);
  ASSERT_NOT_NULL(param_host);
  ASSERT_NOT_NULL(param);
  ASSERT_NOT_NULL(lr_host);
  ASSERT_NOT_NULL(lr);

  PolyUOp *update_uop = poly_alu2(ctx, POLY_OP_SUB, poly_tensor_uop(param), poly_tensor_uop(lr));
  PolyTensor *update = poly_tensor_create(ctx, update_uop, POLY_TENSOR_VALUE, POLY_DEVICE_INTERP);
  ASSERT_NOT_NULL(update);
  ASSERT_PTR_EQ(poly_tensor_assign(ctx, param, update), param);

  PolyUOp *next_lr_buf = poly_buffer_f32(ctx, 1);
  float next_lr_data[] = {0.2f};
  poly_buffer_set(ctx, next_lr_buf, next_lr_data, sizeof(next_lr_data), POLY_DEVICE_HOST);
  PolyTensor *next_lr_host =
      poly_tensor_create(ctx, next_lr_buf, POLY_TENSOR_VALUE, POLY_DEVICE_HOST);
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
  PolyUOp *cpu_device = poly_uop0(ctx, POLY_OP_DEVICE, POLY_VOID, poly_arg_int(POLY_DEVICE_CPU));
  PolyUOp *cuda_device = poly_uop0(ctx, POLY_OP_DEVICE, POLY_VOID, poly_arg_int(POLY_DEVICE_CUDA));
  ASSERT_INT_EQ(poly_uop_device(cpu_device), POLY_DEVICE_CPU);
  ASSERT_INT_EQ(poly_uop_device(cuda_device), POLY_DEVICE_CUDA);

  PolyUOp *cpu_value = poly_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CPU);
  PolyUOp *cuda_target = poly_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CUDA);
  PolyUOp *cuda_value = poly_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CUDA);
  PolyUOp *effect =
      poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, cuda_target, cuda_value, poly_arg_none());
  PolyUOp *after = poly_uop2(ctx, POLY_OP_AFTER, POLY_FLOAT32, cpu_value, effect, poly_arg_none());
  ASSERT_INT_EQ(poly_uop_device(after), POLY_DEVICE_CPU);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, tensor_physicalize_uses_selected_tensor_device) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer_f32(ctx, 4);
  float *da = malloc(4 * sizeof(float));
  ASSERT_NOT_NULL(da);
  da[0] = 1.0f;
  da[1] = 2.0f;
  da[2] = 3.0f;
  da[3] = 4.0f;
  poly_buffer_set(ctx, a, da, 4 * sizeof(float), POLY_DEVICE_HOST);
  PolyUOp *x = poly_alu2(ctx, POLY_OP_ADD, a, poly_const_float(ctx, 1.0));
  PolyUOp *y = poly_alu2(ctx, POLY_OP_MUL, x, poly_const_float(ctx, 2.0));

  PolyTensor *cuda_tensor = poly_tensor_create(ctx, y, POLY_TENSOR_VALUE, POLY_DEVICE_CUDA);
  PolyTensor *cpu_tensor = poly_tensor_create(ctx, y, POLY_TENSOR_PLACE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(cuda_tensor);
  ASSERT_NOT_NULL(cpu_tensor);

  PolyUOp *physical = poly_tensor_physicalize(ctx, cpu_tensor);
  ASSERT_NOT_NULL(physical);
  ASSERT_INT_EQ(physical->op, POLY_OP_MUL);
  ASSERT_INT_EQ(poly_uop_device(physical), POLY_DEVICE_CPU);
  ASSERT_INT_EQ(count_root_ops(ctx, physical, POLY_OP_COPY), 1);
  ASSERT_INT_EQ(count_copy_to_device(ctx, physical, POLY_DEVICE_CPU), 1);
  ASSERT_INT_EQ(count_copy_to_device(ctx, physical, POLY_DEVICE_CUDA), 0);

  poly_ctx_destroy(ctx);
  free(da);
  PASS();
}

TEST_COMMON(realize, tensor_realize_cpu_e2e) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer_f32(ctx, 4);
  PolyUOp *b = poly_buffer_f32(ctx, 4);
  PolyUOp *x = poly_alu2(ctx, POLY_OP_ADD, a, b);
  float da[] = {1.0f, 2.0f, 3.0f, 4.0f};
  float db[] = {10.0f, 20.0f, 30.0f, 40.0f};
  poly_buffer_set(ctx, a, da, sizeof(da), POLY_DEVICE_CPU);
  poly_buffer_set(ctx, b, db, sizeof(db), POLY_DEVICE_CPU);

  PolyTensor *tensor = poly_tensor_create(ctx, x, POLY_TENSOR_PLACE, POLY_DEVICE_CPU);
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

  PolyUOp *a = poly_buffer_f32(ctx, 3);
  float *da = malloc(3 * sizeof(float));
  ASSERT_NOT_NULL(da);
  da[0] = 1.0f;
  da[1] = 2.0f;
  da[2] = 3.0f;
  poly_buffer_set(ctx, a, da, 3 * sizeof(float), POLY_DEVICE_HOST);

  PolyUOp *x = poly_alu2(ctx, POLY_OP_ADD, a, poly_const_float(ctx, 1.0f));
  PolyTensor *tensor = poly_tensor_create(ctx, x, POLY_TENSOR_PLACE, POLY_DEVICE_CPU);
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

  PolyUOp *a = poly_buffer_f32(ctx, 4);
  float *da = malloc(4 * sizeof(float));
  ASSERT_NOT_NULL(da);
  da[0] = 1.0f;
  da[1] = 2.0f;
  da[2] = 3.0f;
  da[3] = 4.0f;
  poly_buffer_set(ctx, a, da, 4 * sizeof(float), POLY_DEVICE_HOST);

  int64_t dims[2] = {1, 4};
  PolyUOp *x = poly_reshape(ctx, a, dims, 2);
  PolyUOp *mean = poly_mean_reduce(ctx, x, 1, 1);
  PolyUOp *centered = poly_sub(ctx, x, mean);

  PolyTensor *tensor = poly_tensor_create(ctx, centered, POLY_TENSOR_PLACE, POLY_DEVICE_CPU);
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

TEST(realize, tensor_assign_host_buffer_updates_original_storage) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer_f32(ctx, 3);
  float *da = malloc(3 * sizeof(float));
  ASSERT_NOT_NULL(da);
  da[0] = 1.0f;
  da[1] = 2.0f;
  da[2] = 3.0f;
  poly_buffer_set(ctx, a, da, 3 * sizeof(float), POLY_DEVICE_HOST);

  PolyTensor *target = poly_tensor_create(ctx, a, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  PolyUOp *value_uop = poly_alu2(ctx, POLY_OP_ADD, a, poly_const_float(ctx, 10.0f));
  PolyTensor *value = poly_tensor_create(ctx, value_uop, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(target);
  ASSERT_NOT_NULL(value);

  PolyTensor *assigned = poly_tensor_assign(ctx, target, value);
  ASSERT_NOT_NULL(assigned);
  PolyTensor *out_tensor = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &assigned, 1, &out_tensor), 0);
  ASSERT_NOT_NULL(out_tensor);
  ASSERT_PTR_EQ(poly_uop_get_buffer_identity(poly_tensor_uop(out_tensor)), a);

  ASSERT_FLOAT_EQ(da[0], 11.0f, 1e-5f);
  ASSERT_FLOAT_EQ(da[1], 12.0f, 1e-5f);
  ASSERT_FLOAT_EQ(da[2], 13.0f, 1e-5f);

  poly_ctx_destroy(ctx);
  free(da);
  PASS();
}

TEST(realize, tensor_assign_shrink_view_updates_base_storage) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *base = poly_buffer_f32(ctx, 8);
  float data[8] = {0};
  poly_buffer_set(ctx, base, data, sizeof(data), POLY_DEVICE_CPU);
  PolyTensor *base_tensor = poly_tensor_create(ctx, base, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(base_tensor);

  PolyUOp *view_uop = poly_shrink(ctx, base, (int64_t[1][2]){{0, 4}}, 1);
  PolyTensor *view_tensor = poly_tensor_create(ctx, view_uop, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(view_tensor);

  PolyUOp *src = poly_buffer_f32(ctx, 4);
  float src_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  poly_buffer_set(ctx, src, src_data, sizeof(src_data), POLY_DEVICE_CPU);
  PolyTensor *src_tensor = poly_tensor_create(ctx, src, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(src_tensor);

  ASSERT_PTR_EQ(poly_tensor_assign(ctx, view_tensor, src_tensor), view_tensor);
  /* tinygrad Tensor.assign retargets live wrappers at the base-buffer level:
   * c[:4].assign(v) makes c.uop an AFTER over the original BUFFER, so realizing
   * c or the view executes a STORE into the base storage. */
  ASSERT_INT_EQ(poly_tensor_uop(base_tensor)->op, POLY_OP_AFTER);
  ASSERT_INT_EQ(poly_tensor_uop(view_tensor)->op, POLY_OP_SHRINK);

  PolyTensor *out = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &base_tensor, 1, &out), 0);
  ASSERT_PTR_EQ(out, base_tensor);
  ASSERT_PTR_EQ(poly_uop_get_buffer_identity(poly_tensor_uop(base_tensor)), base);
  ASSERT_FLOAT_EQ(data[0], 1.0f, 1e-5f);
  ASSERT_FLOAT_EQ(data[1], 2.0f, 1e-5f);
  ASSERT_FLOAT_EQ(data[2], 3.0f, 1e-5f);
  ASSERT_FLOAT_EQ(data[3], 4.0f, 1e-5f);
  ASSERT_FLOAT_EQ(data[4], 0.0f, 1e-5f);
  ASSERT_FLOAT_EQ(data[7], 0.0f, 1e-5f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, tensor_assign_shrink_view_realize_view_updates_base_storage) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *base = poly_buffer_f32(ctx, 8);
  float data[8] = {0};
  poly_buffer_set(ctx, base, data, sizeof(data), POLY_DEVICE_CPU);
  PolyTensor *base_tensor = poly_tensor_create(ctx, base, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(base_tensor);

  PolyUOp *view_uop = poly_shrink(ctx, base, (int64_t[1][2]){{0, 4}}, 1);
  PolyTensor *view_tensor = poly_tensor_create(ctx, view_uop, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(view_tensor);

  PolyUOp *src = poly_buffer_f32(ctx, 4);
  float src_data[4] = {9.0f, 8.0f, 7.0f, 6.0f};
  poly_buffer_set(ctx, src, src_data, sizeof(src_data), POLY_DEVICE_CPU);
  PolyTensor *src_tensor = poly_tensor_create(ctx, src, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(src_tensor);

  ASSERT_PTR_EQ(poly_tensor_assign(ctx, view_tensor, src_tensor), view_tensor);

  PolyTensor *out = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &view_tensor, 1, &out), 0);
  ASSERT_PTR_EQ(out, view_tensor);
  ASSERT_FLOAT_EQ(data[0], 9.0f, 1e-5f);
  ASSERT_FLOAT_EQ(data[1], 8.0f, 1e-5f);
  ASSERT_FLOAT_EQ(data[2], 7.0f, 1e-5f);
  ASSERT_FLOAT_EQ(data[3], 6.0f, 1e-5f);
  ASSERT_FLOAT_EQ(data[4], 0.0f, 1e-5f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, tensor_assign_permute_view_updates_base_storage) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *base = poly_buffer_f32(ctx, 6);
  float data[6] = {0};
  poly_buffer_set(ctx, base, data, sizeof(data), POLY_DEVICE_CPU);
  PolyUOp *matrix = poly_reshape(ctx, base, (int64_t[]){2, 3}, 2);
  PolyTensor *matrix_tensor = poly_tensor_create(ctx, matrix, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(matrix_tensor);

  PolyUOp *view_uop = poly_permute(ctx, matrix, (int64_t[]){1, 0}, 2);
  PolyTensor *view_tensor = poly_tensor_create(ctx, view_uop, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(view_tensor);

  PolyUOp *src_buf = poly_buffer_f32(ctx, 6);
  float src_data[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  poly_buffer_set(ctx, src_buf, src_data, sizeof(src_data), POLY_DEVICE_CPU);
  PolyUOp *src = poly_reshape(ctx, src_buf, (int64_t[]){3, 2}, 2);
  PolyTensor *src_tensor = poly_tensor_create(ctx, src, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(src_tensor);

  ASSERT_PTR_EQ(poly_tensor_assign(ctx, view_tensor, src_tensor), view_tensor);
  ASSERT_INT_EQ(poly_tensor_uop(matrix_tensor)->op, POLY_OP_AFTER);
  ASSERT_INT_EQ(poly_tensor_uop(view_tensor)->op, POLY_OP_PERMUTE);

  PolyTensor *out = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &matrix_tensor, 1, &out), 0);
  ASSERT_PTR_EQ(out, matrix_tensor);
  ASSERT_PTR_EQ(poly_uop_get_buffer_identity(poly_tensor_uop(matrix_tensor)), base);
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

TEST(realize, tensor_current_index_tracks_realize_and_assign) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer_f32(ctx, 1);
  float *da = malloc(sizeof(float));
  ASSERT_NOT_NULL(da);
  da[0] = 1.0f;
  poly_buffer_set(ctx, a, da, sizeof(float), POLY_DEVICE_CPU);

  PolyUOp *one = poly_const_float(ctx, 1.0f);
  PolyUOp *x_expr = poly_alu2(ctx, POLY_OP_ADD, a, one);
  PolyTensor *x = poly_tensor_create(ctx, x_expr, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(x);
  ASSERT_PTR_EQ(poly_tensor_find_current(ctx, x_expr, POLY_DEVICE_CPU, (PolyTensorRole)-1), x);

  PolyTensor *out = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &x, 1, &out), 0);
  ASSERT_PTR_EQ(out, x);
  PolyUOp *x_buf = poly_tensor_uop(x);
  ASSERT_PTR_NEQ(x_buf, x_expr);
  ASSERT_PTR_EQ(poly_tensor_uop_logical(x), x_expr);
  ASSERT_TRUE(poly_uop_has_buffer_identity(x_buf));
  ASSERT_TRUE(poly_tensor_find_current(ctx, x_expr, POLY_DEVICE_CPU, (PolyTensorRole)-1) == NULL);
  ASSERT_PTR_EQ(poly_tensor_find_current(ctx, x_buf, POLY_DEVICE_CPU, (PolyTensorRole)-1), x);

  PolyUOp *v = poly_buffer_f32(ctx, 1);
  float *dv = malloc(sizeof(float));
  ASSERT_NOT_NULL(dv);
  dv[0] = 5.0f;
  poly_buffer_set(ctx, v, dv, sizeof(float), POLY_DEVICE_CPU);
  PolyTensor *vt = poly_tensor_create(ctx, v, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(vt);

  ASSERT_PTR_EQ(poly_tensor_assign(ctx, x, vt), x);
  PolyUOp *after = poly_tensor_uop(x);
  ASSERT_NOT_NULL(after);
  ASSERT_INT_EQ(after->op, POLY_OP_AFTER);
  ASSERT_TRUE(poly_tensor_find_current(ctx, x_buf, POLY_DEVICE_CPU, (PolyTensorRole)-1) == NULL);
  ASSERT_PTR_EQ(poly_tensor_find_current(ctx, after, POLY_DEVICE_CPU, (PolyTensorRole)-1), x);

  out = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &x, 1, &out), 0);
  ASSERT_PTR_EQ(out, x);
  ASSERT_PTR_EQ(poly_tensor_uop(x), x_buf);
  PolyBuffer *x_storage = realized_buffer(ctx, poly_tensor_uop(x));
  ASSERT_NOT_NULL(x_storage);
  float x_val = 0.0f;
  READ_REALIZED_F32(ctx, poly_tensor_uop(x), &x_val, 1);
  ASSERT_FLOAT_EQ(x_val, 5.0f, 1e-5f);

  PolyUOp *fresh_inner = poly_alu2(ctx, POLY_OP_ADD, a, one);
  ASSERT_PTR_EQ(fresh_inner, x_expr);
  PolyUOp *z_expr = poly_alu2(ctx, POLY_OP_ADD, fresh_inner, one);
  PolyTensor *z = poly_tensor_create(ctx, z_expr, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
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

  PolyUOp *a = poly_buffer_f32(ctx, 1);
  float *da = malloc(sizeof(float));
  ASSERT_NOT_NULL(da);
  da[0] = 1.0f;
  poly_buffer_set(ctx, a, da, sizeof(float), POLY_DEVICE_CPU);
  PolyTensor *at = poly_tensor_create(ctx, a, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(at);

  PolyUOp *one = poly_const_float(ctx, 1.0f);
  PolyUOp *ar_expr = poly_alu2(ctx, POLY_OP_ADD, poly_tensor_uop(at), one);
  PolyTensor *ar = poly_tensor_create(ctx, ar_expr, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(ar);
  PolyTensor *out = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &ar, 1, &out), 0);
  ASSERT_PTR_EQ(out, ar);
  PolyBuffer *ar_storage = realized_buffer(ctx, poly_tensor_uop(ar));
  ASSERT_NOT_NULL(ar_storage);
  float ar_val = 0.0f;
  READ_REALIZED_F32(ctx, poly_tensor_uop(ar), &ar_val, 1);
  ASSERT_FLOAT_EQ(ar_val, 2.0f, 1e-5f);

  PolyUOp *ten = poly_buffer_f32(ctx, 1);
  float *dten = malloc(sizeof(float));
  ASSERT_NOT_NULL(dten);
  dten[0] = 10.0f;
  poly_buffer_set(ctx, ten, dten, sizeof(float), POLY_DEVICE_CPU);
  PolyTensor *tent = poly_tensor_create(ctx, ten, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(tent);
  ASSERT_PTR_EQ(poly_tensor_assign(ctx, at, tent), at);
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &at, 1, &out), 0);
  ASSERT_FLOAT_EQ(da[0], 10.0f, 1e-5f);

  PolyUOp *b_expr = poly_alu2(ctx, POLY_OP_ADD, poly_tensor_uop(ar), one);
  ASSERT_PTR_EQ(b_expr->src[0], poly_tensor_uop(ar));
  PolyTensor *b = poly_tensor_create(ctx, b_expr, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(b);
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

  PolyUOp *counter_buf = poly_buffer_f32(ctx, 1);
  float *counter_data = malloc(sizeof(float));
  ASSERT_NOT_NULL(counter_data);
  counter_data[0] = 0.0f;
  poly_buffer_set(ctx, counter_buf, counter_data, sizeof(float), POLY_DEVICE_CPU);
  PolyTensor *counter = poly_tensor_create(ctx, counter_buf, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(counter);

  PolyUOp *one = poly_const_float(ctx, 1.0f);
  PolyTensor *next = poly_tensor_create(
      ctx, poly_alu2(ctx, POLY_OP_ADD, poly_tensor_uop(counter), one), POLY_TENSOR_VALUE,
      POLY_DEVICE_CPU
  );
  ASSERT_NOT_NULL(next);
  ASSERT_PTR_EQ(poly_tensor_assign(ctx, counter, next), counter);
  PolyUOp *first_after = poly_tensor_uop(counter);
  ASSERT_INT_EQ(first_after->op, POLY_OP_AFTER);
  ASSERT_PTR_EQ(first_after->src[0], counter_buf);

  PolyTensor *version0 = poly_tensor_create(
      ctx, poly_contiguous(ctx, poly_sub(ctx, first_after, one)), POLY_TENSOR_VALUE, POLY_DEVICE_CPU
  );
  ASSERT_NOT_NULL(version0);

  next = poly_tensor_create(
      ctx, poly_alu2(ctx, POLY_OP_ADD, poly_tensor_uop(counter), one), POLY_TENSOR_VALUE,
      POLY_DEVICE_CPU
  );
  ASSERT_NOT_NULL(next);
  ASSERT_PTR_EQ(poly_tensor_assign(ctx, counter, next), counter);
  PolyUOp *second_after = poly_tensor_uop(counter);
  ASSERT_INT_EQ(second_after->op, POLY_OP_AFTER);
  ASSERT_PTR_EQ(second_after->src[0], first_after);

  PolyTensor *version1 = poly_tensor_create(
      ctx, poly_contiguous(ctx, poly_sub(ctx, second_after, one)), POLY_TENSOR_VALUE,
      POLY_DEVICE_CPU
  );
  ASSERT_NOT_NULL(version1);

  PolyUOp *versions[2] = {poly_tensor_uop_logical(version0), poly_tensor_uop_logical(version1)};
  PolyUOp *joined = poly_cat(ctx, versions, 2, 0);
  ASSERT_NOT_NULL(joined);
  PolyTensor *result = poly_tensor_create(ctx, joined, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(result);

  PolyTensor *realized = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &result, 1, &realized), 0);
  ASSERT_PTR_EQ(realized, result);
  float values[2] = {-1.0f, -1.0f};
  READ_REALIZED_F32(ctx, poly_tensor_uop(result), values, 2);
  ASSERT_FLOAT_EQ(values[0], 0.0f, 1e-5f);
  ASSERT_FLOAT_EQ(values[1], 1.0f, 1e-5f);

  ASSERT_PTR_EQ(poly_tensor_uop_logical(counter), second_after);
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

TEST(realize, tensor_shared_lazy_retarget_keeps_distinct_tensor_records) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer_f32(ctx, 1);
  float *da = malloc(sizeof(float));
  ASSERT_NOT_NULL(da);
  da[0] = 1.0f;
  poly_buffer_set(ctx, a, da, sizeof(float), POLY_DEVICE_CPU);

  PolyUOp *expr = poly_alu2(ctx, POLY_OP_ADD, a, poly_const_float(ctx, 1.0f));
  PolyTensor *x1 = poly_tensor_create(ctx, expr, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  PolyTensor *x2 = poly_tensor_create(ctx, expr, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(x1);
  ASSERT_NOT_NULL(x2);
  ASSERT_PTR_NEQ(x1, x2);
  ASSERT_PTR_EQ(poly_tensor_uop(x1), poly_tensor_uop(x2));

  PolyTensor *out = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &x1, 1, &out), 0);
  ASSERT_PTR_EQ(out, x1);
  PolyUOp *shared_buf = poly_tensor_uop(x1);
  ASSERT_TRUE(poly_uop_has_buffer_identity(shared_buf));
  ASSERT_INT_EQ(
      poly_tensor_update(ctx, x2, NULL, shared_buf, POLY_TENSOR_VALUE, POLY_DEVICE_CPU), 0
  );
  ASSERT_PTR_NEQ(x1, x2);
  ASSERT_PTR_EQ(poly_tensor_uop(x1), poly_tensor_uop(x2));

  PolyUOp *v = poly_buffer_f32(ctx, 1);
  float *dv = malloc(sizeof(float));
  ASSERT_NOT_NULL(dv);
  dv[0] = 5.0f;
  poly_buffer_set(ctx, v, dv, sizeof(float), POLY_DEVICE_CPU);
  PolyTensor *vt = poly_tensor_create(ctx, v, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
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

  PolySchedule *sched = poly_schedule_with_vars(ctx, targets, 1, realized);
  ASSERT_NOT_NULL(sched);
  ASSERT_NOT_NULL(sched->template);
  ASSERT_INT_EQ(sched->template->n_calls, 0);
  ASSERT_PTR_EQ(realized[0], buf);

  poly_schedule_free(sched);
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
  PolySchedule *sched = poly_schedule_with_vars(ctx, contig_targets, 1, contig_realized);
  ASSERT_NOT_NULL(sched);
  ASSERT_NOT_NULL(sched->template);
  ASSERT_INT_EQ(sched->template->n_calls, 0);
  ASSERT_PTR_EQ(contig_realized[0], contig);

  poly_schedule_free(sched);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, requested_plain_root_retargets_live_dependent_to_final_buffer) {
  for (int preplace_dependent = 0; preplace_dependent < 2; preplace_dependent++) {
    PolyCtx *ctx = poly_ctx_new();
    ASSERT_NOT_NULL(ctx);
    poly_ctx_set_preferred_device(ctx, POLY_DEVICE_CPU);

    float input_value = 1.0f;
    PolyUOp *input = poly_buffer_f32(ctx, 1);
    ASSERT_NOT_NULL(input);
    poly_buffer_set(ctx, input, &input_value, sizeof(input_value), POLY_DEVICE_HOST);
    PolyTensor *input_tensor = poly_tensor_create(ctx, input, POLY_TENSOR_VALUE, POLY_DEVICE_HOST);
    PolyUOp *requested_logical = poly_add(ctx, input, poly_const_float(ctx, 1.0));
    PolyTensor *requested =
        poly_tensor_create(ctx, requested_logical, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
    PolyUOp *dependent_logical = poly_mul(ctx, requested_logical, poly_const_float(ctx, 2.0));
    PolyTensor *dependent =
        poly_tensor_create(ctx, dependent_logical, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
    ASSERT_NOT_NULL(input_tensor);
    ASSERT_NOT_NULL(requested);
    ASSERT_NOT_NULL(dependent);

    PolyUOp *requested_placed = poly_tensor_physicalize(ctx, requested);
    PolyUOp *dependent_placed = poly_tensor_physicalize(ctx, dependent);
    ASSERT_NOT_NULL(requested_placed);
    ASSERT_NOT_NULL(dependent_placed);
    ASSERT_PTR_NEQ(requested_placed, requested_logical);
    if (preplace_dependent) {
      ASSERT_INT_EQ(
          poly_tensor_update(
              ctx, dependent, NULL, dependent_placed, POLY_TENSOR_VALUE, POLY_DEVICE_CPU
          ),
          0
      );
    }

    PolyTensor *realized = NULL;
    ASSERT_INT_EQ(poly_realize_tensors(ctx, &requested, 1, &realized), 0);
    ASSERT_PTR_EQ(realized, requested);
    ASSERT_PTR_EQ(poly_tensor_uop_logical(requested), requested_logical);
    ASSERT_PTR_EQ(poly_tensor_uop_logical(dependent), dependent_logical);

    PolyUOp *requested_physical = poly_tensor_uop_physical(requested);
    PolyUOp *dependent_physical = poly_tensor_uop_physical(dependent);
    ASSERT_NOT_NULL(requested_physical);
    ASSERT_NOT_NULL(dependent_physical);
    ASSERT_EQ(requested_physical->op, POLY_OP_BUFFER);
    ASSERT_EQ(dependent_physical->op, POLY_OP_MUL);
    /* Pinned transform_to_call returns the requested ADD in buffer_map and
     * Tensor.linear_with_vars applies it to every live Tensor. The dependent
     * must consume that final BUFFER, not retain/rebuild the placed ADD. */
    ASSERT_TRUE(poly_uop_reachable(ctx, dependent_physical, requested_physical));
    ASSERT_FALSE(poly_uop_reachable(ctx, dependent_physical, requested_placed));
    ASSERT_INT_EQ(count_root_ops(ctx, dependent_physical, POLY_OP_ADD), 0);

    input_value = 10.0f;
    ASSERT_INT_EQ(poly_buffer_write(ctx, input, &input_value, sizeof(input_value)), 0);
    ASSERT_INT_EQ(poly_realize_tensors(ctx, &dependent, 1, &realized), 0);
    ASSERT_PTR_EQ(realized, dependent);
    float dependent_value = 0.0f;
    READ_REALIZED_F32(ctx, poly_tensor_uop(dependent), &dependent_value, 1);
    ASSERT_FLOAT_EQ(dependent_value, 4.0f, 1e-6f);

    poly_ctx_destroy(ctx);
  }
  PASS();
}

TEST(realize, aggregate_map_preserves_opaque_bodies_and_places_external_args) {
  const PolyOps opaque_ops[] = {POLY_OP_CALL, POLY_OP_FUNCTION};
  for (int op_idx = 0; op_idx < 2; op_idx++) {
    for (int key_in_external = 0; key_in_external < 2; key_in_external++) {
      PolyCtx *ctx = poly_ctx_new();
      PolyMap *placement_memo[POLY_DEVICE_DISK + 1] = {0};
      ASSERT_NOT_NULL(ctx);

      float input_value = 1.0f;
      PolyUOp *input = poly_buffer_f32(ctx, 1);
      ASSERT_NOT_NULL(input);
      poly_buffer_set(ctx, input, &input_value, sizeof(input_value), POLY_DEVICE_HOST);
      PolyUOp *logical_key = poly_contiguous(ctx, poly_add(ctx, input, poly_const_float(ctx, 1.0)));
      PolyTensor *boundary =
          poly_tensor_create(ctx, logical_key, POLY_TENSOR_VALUE, POLY_DEVICE_CUDA);
      PolyTensor *boundary_roots[1] = {boundary};
      PolyUOp *placed_roots[1] = {NULL};
      ASSERT_NOT_NULL(logical_key);
      ASSERT_NOT_NULL(boundary);
      ASSERT_INT_EQ(
          poly_tensor_physicalize_many(ctx, boundary_roots, 1, placed_roots, placement_memo), 0
      );
      PolyUOp *placed_key = placed_roots[0];
      ASSERT_NOT_NULL(placed_key);

      PolyUOp *value = poly_buffer_on_device(ctx, POLY_FLOAT32, 1, POLY_DEVICE_CUDA);
      PolyUOp *body = poly_sink1(ctx, key_in_external ? value : logical_key);
      PolyUOp *external = key_in_external ? logical_key : poly_const_float(ctx, 3.0);
      PolyUOp *opaque_src[2] = {body, external};
      PolyUOp *opaque =
          poly_uop(ctx, opaque_ops[op_idx], POLY_VOID, opaque_src, 2, poly_arg_none());
      PolyUOp *after_src[2] = {value, opaque};
      PolyUOp *after = poly_uop(ctx, POLY_OP_AFTER, value->dtype, after_src, 2, poly_arg_none());
      PolyTensor *tensor = poly_tensor_create(ctx, after, POLY_TENSOR_VALUE, POLY_DEVICE_CUDA);
      PolyUOp *replacement = poly_buffer_on_device(ctx, POLY_FLOAT32, 1, POLY_DEVICE_CUDA);
      ASSERT_NOT_NULL(value);
      ASSERT_NOT_NULL(body);
      ASSERT_NOT_NULL(external);
      ASSERT_NOT_NULL(opaque);
      ASSERT_NOT_NULL(after);
      ASSERT_NOT_NULL(tensor);
      ASSERT_NOT_NULL(replacement);

      PolyUOp *from[1] = {placed_key};
      PolyUOp *to[1] = {replacement};
      PolyCtxStats before = {0}, after_stats = {0};
      ASSERT_INT_EQ(poly_ctx_stats(ctx, &before), 0);
      ASSERT_INT_EQ(
          poly_tensor_apply_realize_map(ctx, from, to, 1, POLY_DEVICE_CUDA, placement_memo), 0
      );
      ASSERT_INT_EQ(poly_ctx_stats(ctx, &after_stats), 0);
      ASSERT_PTR_EQ(poly_tensor_uop_logical(tensor), after);

      PolyUOp *physical = poly_tensor_uop_physical(tensor);
      if (!key_in_external) {
        /* Placed-only evidence inside an opaque body is inert and must not
         * retain a diagnostic physical graph in the ctx arena/CSE. */
        ASSERT_TRUE(physical == NULL);
        ASSERT_TRUE(after_stats.arena_bytes == before.arena_bytes);
        ASSERT_TRUE(after_stats.cse_entries == before.cse_entries);
      } else {
        ASSERT_NOT_NULL(physical);
        ASSERT_EQ(physical->op, POLY_OP_AFTER);
        ASSERT_PTR_EQ(physical->src[0], value);
        ASSERT_NOT_NULL(physical->src[1]);
        ASSERT_EQ(physical->src[1]->op, opaque_ops[op_idx]);
        ASSERT_PTR_EQ(physical->src[1]->src[0], body);
        ASSERT_PTR_EQ(physical->src[1]->src[1], replacement);
      }

      poly_tensor_physicalize_memo_destroy(placement_memo);
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

  PolyUOp *input = poly_buffer_f32(ctx, 1);
  float initial = 1.0f;
  poly_buffer_set(ctx, input, &initial, sizeof(initial), POLY_DEVICE_INTERP);
  PolyUOp *one = poly_const_float(ctx, 1.0);
  PolyUOp *two = poly_const_float(ctx, 2.0);
  PolyUOp *five = poly_const_float(ctx, 5.0);
  PolyUOp *inner = poly_contiguous(ctx, poly_add(ctx, input, one));
  PolyUOp *value = poly_mul(ctx, inner, two);
  PolyUOp *requested = poly_contiguous(ctx, value);
  PolyUOp *consumer = poly_add(ctx, inner, five);
  ASSERT_NOT_NULL(requested);
  ASSERT_NOT_NULL(consumer);

  PolyTensor *requested_tensor =
      poly_tensor_create(ctx, requested, POLY_TENSOR_VALUE, POLY_DEVICE_INTERP);
  PolyTensor *consumer_tensor =
      poly_tensor_create(ctx, consumer, POLY_TENSOR_VALUE, POLY_DEVICE_INTERP);
  ASSERT_NOT_NULL(requested_tensor);
  ASSERT_NOT_NULL(consumer_tensor);

  PolyTensor *realized = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &requested_tensor, 1, &realized), 0);
  ASSERT_PTR_EQ(realized, requested_tensor);
  ASSERT_PTR_EQ(poly_tensor_uop_logical(consumer_tensor), consumer);
  ASSERT_NOT_NULL(poly_tensor_uop_physical(consumer_tensor));

  float requested_out = 0.0f;
  READ_REALIZED_F32(ctx, poly_tensor_uop(requested_tensor), &requested_out, 1);
  ASSERT_FLOAT_EQ(requested_out, 4.0f, 1e-5f);

  float updated = 10.0f;
  ASSERT_INT_EQ(poly_buffer_write(ctx, input, &updated, sizeof(updated)), 0);
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

  PolyUOp *input = poly_buffer_f32(ctx, 1);
  float initial = 1.0f;
  poly_buffer_set(ctx, input, &initial, sizeof(initial), POLY_DEVICE_INTERP);
  PolyUOp *inner = poly_contiguous(ctx, poly_add(ctx, input, poly_const_float(ctx, 1.0)));
  PolyUOp *consumer = poly_add(ctx, inner, poly_const_float(ctx, 5.0));
  ASSERT_NOT_NULL(inner);
  ASSERT_NOT_NULL(consumer);

  PolyTensor *inner_tensor = poly_tensor_create(ctx, inner, POLY_TENSOR_VALUE, POLY_DEVICE_INTERP);
  PolyTensor *consumer_tensor =
      poly_tensor_create(ctx, consumer, POLY_TENSOR_VALUE, POLY_DEVICE_INTERP);
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

TEST(realize, placed_contiguous_map_retargets_prebuilt_logical_consumer) {
  PolyCtx *ctx = poly_ctx_new();
  PolyMap *placement_memo[POLY_DEVICE_DISK + 1] = {0};

  PolyUOp *input = poly_buffer_f32(ctx, 1);
  float initial = 1.0f;
  poly_buffer_set(ctx, input, &initial, sizeof(initial), POLY_DEVICE_CPU);
  PolyUOp *logical_contiguous =
      poly_contiguous(ctx, poly_add(ctx, input, poly_const_float(ctx, 1.0)));
  PolyTensor *boundary =
      poly_tensor_create(ctx, logical_contiguous, POLY_TENSOR_VALUE, POLY_DEVICE_CUDA);
  ASSERT_NOT_NULL(boundary);

  /* CUDA placement rebuilds the CONTIGUOUS around the input COPY, producing
   * the exact kind of placed-only becomes-map key seen in the HLB RNG crop. */
  PolyTensor *boundary_roots[1] = {boundary};
  PolyUOp *placed_roots[1] = {NULL};
  ASSERT_INT_EQ(
      poly_tensor_physicalize_many(ctx, boundary_roots, 1, placed_roots, placement_memo), 0
  );
  PolyUOp *placed_contiguous = placed_roots[0];
  ASSERT_NOT_NULL(placed_contiguous);
  ASSERT_INT_EQ(placed_contiguous->op, POLY_OP_CONTIGUOUS);
  ASSERT_PTR_NEQ(placed_contiguous, logical_contiguous);

  PolyUOp *consumer = poly_add(ctx, logical_contiguous, poly_const_float(ctx, 5.0));
  PolyTensor *consumer_tensor =
      poly_tensor_create(ctx, consumer, POLY_TENSOR_VALUE, POLY_DEVICE_CUDA);
  PolyUOp *replacement = poly_buffer_on_device(ctx, POLY_FLOAT32, 1, POLY_DEVICE_CUDA);
  ASSERT_NOT_NULL(consumer_tensor);
  ASSERT_NOT_NULL(replacement);

  PolyUOp *from[1] = {placed_contiguous};
  PolyUOp *to[1] = {replacement};
  ASSERT_INT_EQ(
      poly_tensor_apply_realize_map(ctx, from, to, 1, POLY_DEVICE_CUDA, placement_memo), 0
  );
  ASSERT_PTR_EQ(poly_tensor_uop_logical(boundary), logical_contiguous);
  ASSERT_PTR_EQ(poly_tensor_uop_logical(consumer_tensor), consumer);
  ASSERT_PTR_EQ(poly_tensor_uop_physical(boundary), replacement);
  ASSERT_NOT_NULL(poly_tensor_uop_physical(consumer_tensor));
  ASSERT_TRUE(poly_uop_reachable(ctx, poly_tensor_uop_physical(consumer_tensor), replacement));
  ASSERT_FALSE(poly_uop_reachable(ctx, poly_tensor_uop_physical(consumer_tensor), placed_contiguous)
  );

  poly_tensor_physicalize_memo_destroy(placement_memo);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, placed_after_map_retargets_live_assignment_across_copy) {
  PolyCtx *ctx = poly_ctx_new();
  PolyMap *placement_memo[POLY_DEVICE_DISK + 1] = {0};

  PolyUOp *logical_buffer = poly_buffer(ctx, POLY_UINT32, 2);
  PolyTensor *host_counter =
      poly_tensor_create(ctx, logical_buffer, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  PolyTensor *counter = poly_tensor_to_device(ctx, host_counter, POLY_DEVICE_CUDA);
  ASSERT_NOT_NULL(host_counter);
  ASSERT_NOT_NULL(counter);

  PolyUOp *incremented = poly_add(ctx, poly_tensor_uop(counter), poly_const_int(ctx, 1));
  PolyTensor *value = poly_tensor_create(ctx, incremented, POLY_TENSOR_VALUE, POLY_DEVICE_CUDA);
  ASSERT_NOT_NULL(value);
  ASSERT_PTR_EQ(poly_tensor_assign(ctx, counter, value), counter);

  PolyUOp *logical_after = poly_tensor_uop_logical(counter);
  PolyTensor *counter_roots[1] = {counter};
  PolyUOp *placed_roots[1] = {NULL};
  ASSERT_INT_EQ(
      poly_tensor_physicalize_many(ctx, counter_roots, 1, placed_roots, placement_memo), 0
  );
  PolyUOp *placed_after = placed_roots[0];
  ASSERT_NOT_NULL(logical_after);
  ASSERT_NOT_NULL(placed_after);
  ASSERT_INT_EQ(logical_after->op, POLY_OP_AFTER);
  ASSERT_INT_EQ(logical_after->src[0]->op, POLY_OP_BUFFER);
  ASSERT_INT_EQ(placed_after->op, POLY_OP_AFTER);
  ASSERT_INT_EQ(placed_after->src[0]->op, POLY_OP_COPY);
  ASSERT_TRUE(poly_uop_get_buffer_identity(placed_after->src[0]) == NULL);

  /* Pinned finalize_after maps the placed AFTER directly to its final storage.
   * The live logical tensor must retry against its exact placed graph because
   * the COPY is placement-only; broad shape/dtype resemblance is insufficient. */
  PolyUOp *realized = poly_buffer_on_device(ctx, POLY_UINT32, 2, POLY_DEVICE_CUDA);
  PolyUOp *from[1] = {placed_after};
  PolyUOp *to[1] = {realized};
  ASSERT_NOT_NULL(realized);
  ASSERT_INT_EQ(
      poly_tensor_apply_realize_map(ctx, from, to, 1, POLY_DEVICE_CUDA, placement_memo), 0
  );
  ASSERT_PTR_EQ(poly_tensor_uop_logical(counter), logical_after);
  ASSERT_PTR_EQ(poly_tensor_uop_physical(counter), realized);
  ASSERT_PTR_EQ(poly_tensor_uop(counter), realized);

  poly_tensor_physicalize_memo_destroy(placement_memo);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, transform_to_call_finalizes_creation_copy_assignment_after) {
  PolyCtx *ctx = poly_ctx_new();

  float initial = 2.0f;
  PolyUOp *host = poly_buffer_f32(ctx, 1);
  poly_buffer_set(ctx, host, &initial, sizeof(initial), POLY_DEVICE_HOST);
  PolyUOp *device = poly_uop0(ctx, POLY_OP_DEVICE, POLY_VOID, poly_arg_int(POLY_DEVICE_CPU));
  PolyUOp *copy_src[2] = {host, device};
  PolyUOp *copy = poly_uop(ctx, POLY_OP_COPY, POLY_FLOAT32, copy_src, 2, poly_arg_none());
  PolyUOp *value = poly_add(ctx, copy, poly_const_float(ctx, 1.0));
  PolyUOp *store = poly_store_val(ctx, copy, value);
  PolyUOp *after_src[2] = {copy, store};
  PolyUOp *after = poly_uop(ctx, POLY_OP_AFTER, POLY_FLOAT32, after_src, 2, poly_arg_none());
  ASSERT_NOT_NULL(after);

  /* Pinned add_tags merges a creation COPY tag into this assignment AFTER.
   * Keep live tensors for both original UOps to prove finalize_after's two
   * becomes-map entries, while the executable consumer reads the new value. */
  PolyTensor *counter = poly_tensor_create(ctx, after, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  PolyTensor *creation_copy = poly_tensor_create(ctx, copy, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
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
  ASSERT_INT_EQ(
      poly_tensor_apply_realize_map(ctx, map_orig, map_repl, map_n, POLY_DEVICE_AUTO, NULL), 0
  );

  const PolyUOp *realized_identity = poly_uop_get_buffer_identity(realized);
  const PolyUOp *counter_identity = poly_uop_get_buffer_identity(poly_tensor_uop_physical(counter));
  const PolyUOp *creation_copy_identity =
      poly_uop_get_buffer_identity(poly_tensor_uop_physical(creation_copy));
  ASSERT_PTR_EQ(poly_tensor_uop_logical(counter), after);
  ASSERT_NOT_NULL(counter_identity);
  ASSERT_PTR_EQ(poly_tensor_uop_logical(creation_copy), copy);
  ASSERT_PTR_EQ(creation_copy_identity, counter_identity);
  ASSERT_PTR_NEQ(counter_identity, realized_identity);
  ASSERT_TRUE(poly_uop_reachable(ctx, call, (PolyUOp *)counter_identity));

  free(map_repl);
  free(map_orig);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, nested_contiguous_inside_after_store_precedes_consumer) {
  PolyCtx *ctx = poly_ctx_new();
  poly_ctx_set_preferred_device(ctx, POLY_DEVICE_INTERP);

  PolyUOp *source = poly_buffer_f32(ctx, 1);
  PolyUOp *target = poly_buffer_f32(ctx, 1);
  float source_value = 1.0f, target_value = 0.0f;
  poly_buffer_set(ctx, source, &source_value, sizeof(source_value), POLY_DEVICE_INTERP);
  poly_buffer_set(ctx, target, &target_value, sizeof(target_value), POLY_DEVICE_INTERP);

  PolyUOp *inner = poly_contiguous(ctx, poly_add(ctx, source, poly_const_float(ctx, 1.0)));
  PolyUOp *store = poly_store_val(ctx, target, inner);
  PolyUOp *after_src[2] = {target, store};
  PolyUOp *after = poly_uop(ctx, POLY_OP_AFTER, target->dtype, after_src, 2, poly_arg_none());
  PolyUOp *requested = poly_contiguous(ctx, poly_add(ctx, after, poly_const_float(ctx, 5.0)));
  ASSERT_NOT_NULL(inner);
  ASSERT_NOT_NULL(store);
  ASSERT_NOT_NULL(after);
  ASSERT_NOT_NULL(requested);

  PolyTensor *tensor = poly_tensor_create(ctx, requested, POLY_TENSOR_VALUE, POLY_DEVICE_INTERP);
  ASSERT_NOT_NULL(tensor);
  PolyTensor *realized = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &tensor, 1, &realized), 0);
  ASSERT_PTR_EQ(realized, tensor);

  float requested_out = 0.0f, target_out = 0.0f;
  READ_REALIZED_F32(ctx, poly_tensor_uop(tensor), &requested_out, 1);
  READ_REALIZED_F32(ctx, target, &target_out, 1);
  ASSERT_FLOAT_EQ(requested_out, 7.0f, 1e-5f);
  ASSERT_FLOAT_EQ(target_out, 2.0f, 1e-5f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, nested_after_store_consumer_indexes_producer_buffer_once) {
  PolyCtx *ctx = poly_ctx_new();
  poly_ctx_set_preferred_device(ctx, POLY_DEVICE_CPU);

  PolyUOp *inner_buffer = poly_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CPU);
  PolyUOp *outer_buffer = poly_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(inner_buffer);
  ASSERT_NOT_NULL(outer_buffer);
  ASSERT_INT_EQ(poly_buffer_allocate(ctx, inner_buffer, POLY_DEVICE_CPU), 0);
  ASSERT_INT_EQ(poly_buffer_allocate(ctx, outer_buffer, POLY_DEVICE_CPU), 0);

  int64_t shape[1] = {4};
  int64_t singleton[1] = {1};
  PolyUOp *ones = poly_expand(
      ctx, poly_reshape(ctx, poly_const_float(ctx, 1.0), singleton, 1), shape, 1);
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
  PolySchedule *schedule = poly_schedule_with_vars(ctx, &outer_after, 1, &scheduled_out);
  ASSERT_NOT_NULL(schedule);
  ASSERT_NOT_NULL(scheduled_out);
  ASSERT_INT_EQ(schedule->template->n_calls, 2);
  ASSERT_INT_EQ(count_root_ops(ctx, poly_schedule_call_body(schedule, 1), POLY_OP_ADD), 0);
  ASSERT_INT_EQ(poly_run_schedule(ctx, schedule, NULL, 0), 0);

  float inner_values[4] = {0};
  float outer_values[4] = {0};
  ASSERT_INT_EQ(poly_buffer_read(ctx, inner_buffer, inner_values, sizeof(inner_values)), 0);
  ASSERT_INT_EQ(poly_buffer_read(ctx, outer_buffer, outer_values, sizeof(outer_values)), 0);
  for (int i = 0; i < 4; i++) {
    ASSERT_FLOAT_EQ(inner_values[i], 1.0f, 1e-6f);
    ASSERT_FLOAT_EQ(outer_values[i], 1.0f, 1e-6f);
  }

  poly_schedule_free(schedule);
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
  PolySchedule *schedule = poly_schedule_with_vars(ctx, &outer_after, 1, &scheduled_out);
  ASSERT_NOT_NULL(schedule);
  ASSERT_PTR_EQ(scheduled_out, outer_view);
  ASSERT_INT_EQ(schedule->template->n_calls, 2);
  ASSERT_INT_EQ(poly_run_schedule(ctx, schedule, NULL, 0), 0);

  float inner_out = 0.0f, outer_out = 0.0f;
  ASSERT_INT_EQ(poly_buffer_read(ctx, inner_buffer, &inner_out, sizeof(inner_out)), 0);
  ASSERT_INT_EQ(poly_buffer_read(ctx, outer_buffer, &outer_out, sizeof(outer_out)), 0);
  ASSERT_FLOAT_EQ(inner_out, 1.0f, 1e-6f);
  ASSERT_FLOAT_EQ(outer_out, 1.0f, 1e-6f);

  poly_schedule_free(schedule);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, batched_contiguous_inside_after_store_precedes_consumer) {
  uint64_t kernel_counts[2] = {0, 0};
  for (int reverse = 0; reverse < 2; reverse++) {
    PolyCtx *ctx = poly_ctx_new();
    poly_ctx_set_preferred_device(ctx, POLY_DEVICE_INTERP);

    PolyUOp *source = poly_buffer_f32(ctx, 1);
    PolyUOp *target = poly_buffer_f32(ctx, 1);
    float source_value = 1.0f, target_value = 0.0f;
    poly_buffer_set(ctx, source, &source_value, sizeof(source_value), POLY_DEVICE_INTERP);
    poly_buffer_set(ctx, target, &target_value, sizeof(target_value), POLY_DEVICE_INTERP);

    PolyUOp *inner = poly_contiguous(ctx, poly_add(ctx, source, poly_const_float(ctx, 1.0)));
    PolyUOp *store = poly_store_val(ctx, target, inner);
    PolyUOp *after_src[2] = {target, store};
    PolyUOp *after = poly_uop(ctx, POLY_OP_AFTER, target->dtype, after_src, 2, poly_arg_none());
    PolyUOp *requested = poly_contiguous(ctx, poly_add(ctx, after, poly_const_float(ctx, 5.0)));
    ASSERT_NOT_NULL(inner);
    ASSERT_NOT_NULL(store);
    ASSERT_NOT_NULL(after);
    ASSERT_NOT_NULL(requested);

    PolyTensor *inner_tensor =
        poly_tensor_create(ctx, inner, POLY_TENSOR_VALUE, POLY_DEVICE_INTERP);
    PolyTensor *requested_tensor =
        poly_tensor_create(ctx, requested, POLY_TENSOR_VALUE, POLY_DEVICE_INTERP);
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
    READ_REALIZED_F32(ctx, target, &target_out, 1);
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
  PolyUOp *materialized = poly_contiguous(ctx, poly_add(ctx, a, poly_const_float(ctx, 0.0f)));
  PolyUOp *selected = poly_shrink(ctx, materialized, selected_bounds, 2);
  PolyUOp *value = poly_div(ctx, b, selected);
  ASSERT_NOT_NULL(a);
  ASSERT_NOT_NULL(b);
  ASSERT_NOT_NULL(materialized);
  ASSERT_NOT_NULL(selected);
  ASSERT_NOT_NULL(value);
  PolyTensor *tensor = poly_tensor_create(ctx, value, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(tensor);
  PolyUOp *physical = poly_tensor_physicalize(ctx, tensor);
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
  PolySchedule *schedule = poly_schedule_with_vars(ctx, &physical, 1, &scheduled_out);
  ASSERT_NOT_NULL(schedule);
  ASSERT_NOT_NULL(scheduled_out);
  ASSERT_INT_EQ(schedule->template->n_calls, 4);

  PolyUOp *selected_index = NULL;
  for (int i = 0; i < schedule->template->n_calls; i++) {
    PolyUOp *body = poly_schedule_call_body(schedule, i);
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

  ASSERT_INT_EQ(poly_run_schedule(ctx, schedule, NULL, 0), 0);
  float result = 0.0f;
  READ_REALIZED_F32(ctx, scheduled_out, &result, 1);
  ASSERT_FLOAT_EQ(result, 1.4f, 1e-6f);

  poly_schedule_free(schedule);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, call_body_paramarg_survives_linear_then_resolves_concrete_args) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  poly_ctx_set_preferred_device(ctx, POLY_DEVICE_INTERP);

  PolyUOp *a = poly_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_INTERP);
  PolyUOp *b = poly_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_INTERP);
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
    ASSERT_TRUE(poly_dtype_eq(params[slot]->dtype, poly_dtype_scalar(concrete->dtype)));
    ASSERT_INT_EQ(params[slot]->arg.param->device, poly_uop_device(concrete));
    ASSERT_INT_EQ(params[slot]->src[0]->n_src, 1);
    ASSERT_INT_EQ(params[slot]->src[0]->src[0]->op, POLY_OP_CONST);
    ASSERT_INT_EQ(params[slot]->src[0]->src[0]->arg.i, 4);
  }

  PolyUOp *linear = poly_lower_sink_to_linear(ctx, outer->src[0], POLY_MODE_CALL);
  ASSERT_NOT_NULL(linear);
  ASSERT_INT_EQ(linear->op, POLY_OP_LINEAR);
  ASSERT_INT_EQ(linear->n_src, 2);
  ASSERT_INT_EQ(count_shaped_value_params(ctx, linear), 4);
  const int expected_outer_slots[2][3] = {{0, 1, 2}, {3, 1, 2}};
  for (int k = 0; k < linear->n_src; k++) {
    PolyUOp *call = linear->src[k];
    ASSERT_NOT_NULL(call);
    ASSERT_INT_EQ(call->op, POLY_OP_CALL);
    ASSERT_INT_EQ(call->n_src, 4);
    ASSERT_INT_EQ(count_shaped_value_params(ctx, call->src[0]), 0);
    for (int i = 1; i < call->n_src; i++) {
      ASSERT_TRUE(is_shaped_value_param(call->src[i]));
      int slot = (int)call->src[i]->arg.param->slot;
      ASSERT_TRUE(slot >= 0 && slot < 4);
      ASSERT_INT_EQ(slot, expected_outer_slots[k][i - 1]);
      ASSERT_PTR_EQ(call->src[i], params[slot]);
    }

    PolyUOp *program = call->src[0];
    ASSERT_INT_EQ(count_root_ops(ctx, program, POLY_OP_PARAM), 3);
    ASSERT_INT_EQ(count_root_ops(ctx, program, POLY_OP_INDEX), 3);
    ASSERT_INT_EQ(count_root_ops(ctx, program, POLY_OP_RANGE), 1);
    ASSERT_INT_EQ(count_root_ops(ctx, program, POLY_OP_STORE), 1);
    ASSERT_INT_EQ(count_root_ops(ctx, program, POLY_OP_END), 1);
    ASSERT_INT_EQ(count_root_ops(ctx, program, POLY_OP_STAGE), 0);

    PolyUOp *store = NULL;
    PolyUOp *alu = NULL;
    int n_program = 0;
    PolyUOp **program_topo = poly_toposort(ctx, program, &n_program);
    for (int i = 0; i < n_program; i++) {
      if (program_topo[i]->op == POLY_OP_STORE) store = program_topo[i];
      if (program_topo[i]->op == (k == 0 ? POLY_OP_ADD : POLY_OP_MUL)) alu = program_topo[i];
    }
    ASSERT_NOT_NULL(store);
    ASSERT_NOT_NULL(alu);
    ASSERT_INT_EQ(store->n_src, 2);
    ASSERT_INT_EQ(alu->n_src, 2);

    PolyUOp *dst_index = store->src[0];
    ASSERT_INT_EQ(dst_index->op, POLY_OP_INDEX);
    ASSERT_INT_EQ(dst_index->n_src, 2);
    ASSERT_INT_EQ(dst_index->src[0]->op, POLY_OP_PARAM);
    ASSERT_TRUE(dst_index->src[0]->dtype.is_ptr);
    ASSERT_INT_EQ(dst_index->src[0]->arg.kind, POLY_ARG_INT);
    ASSERT_INT_EQ(dst_index->src[0]->arg.i, 0);
    ASSERT_INT_EQ(dst_index->src[1]->op, POLY_OP_RANGE);
    ASSERT_PTR_EQ(store->src[1], alu);

    for (int i = 0; i < 2; i++) {
      PolyUOp *read_index = alu->src[i];
      ASSERT_INT_EQ(read_index->op, POLY_OP_INDEX);
      ASSERT_INT_EQ(read_index->n_src, 2);
      ASSERT_INT_EQ(read_index->src[0]->op, POLY_OP_PARAM);
      ASSERT_TRUE(read_index->src[0]->dtype.is_ptr);
      ASSERT_INT_EQ(read_index->src[0]->arg.kind, POLY_ARG_INT);
      ASSERT_INT_EQ(read_index->src[0]->arg.i, i + 1);
      ASSERT_PTR_EQ(read_index->src[1], dst_index->src[1]);
    }
    ASSERT_PTR_NEQ(dst_index->src[0], alu->src[0]->src[0]);
    ASSERT_PTR_NEQ(dst_index->src[0], alu->src[1]->src[0]);
    ASSERT_PTR_NEQ(alu->src[0]->src[0], alu->src[1]->src[0]);
  }

  PolyUOp *resolved_out[2] = {NULL, NULL};
  PolySchedule *schedule = poly_schedule_with_vars(ctx, targets, 2, resolved_out);
  ASSERT_NOT_NULL(schedule);
  ASSERT_INT_EQ(schedule->template->n_calls, 2);
  ASSERT_INT_EQ(poly_schedule_external_slot_count(schedule), 4);
  for (int k = 0; k < schedule->template->n_calls; k++) {
    PolyUOp *call = poly_schedule_call(schedule, k);
    ASSERT_NOT_NULL(call);
    ASSERT_INT_EQ(call->op, POLY_OP_CALL);
    ASSERT_INT_EQ(count_shaped_value_params(ctx, call), 0);
    for (int i = 1; i < call->n_src; i++)
      ASSERT_NOT_NULL(poly_uop_get_buffer_identity(call->src[i]));
  }
  ASSERT_INT_EQ(poly_run_schedule(ctx, schedule, NULL, 0), 0);

  float add_out[4] = {0};
  float mul_out[4] = {0};
  READ_REALIZED_F32(ctx, resolved_out[0], add_out, 4);
  READ_REALIZED_F32(ctx, resolved_out[1], mul_out, 4);
  for (int i = 0; i < 4; i++) {
    ASSERT_FLOAT_EQ(add_out[i], da[i] + db[i], 1e-6f);
    ASSERT_FLOAT_EQ(mul_out[i], da[i] * db[i], 1e-6f);
  }

  poly_schedule_free(schedule);
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
  PolySchedule *schedule = poly_schedule_with_vars(ctx, targets, 2, resolved);
  ASSERT_NOT_NULL(schedule);
  ASSERT_INT_EQ(schedule->template->n_calls, 2);
  const PolyUOp *parent_buffer = poly_uop_get_buffer_identity(resolved[0]);
  ASSERT_NOT_NULL(parent_buffer);
  for (int call_index = 0; call_index < 2; call_index++) {
    PolyUOp *call = poly_schedule_call(schedule, call_index);
    bool has_parent_buffer = false;
    ASSERT_NOT_NULL(call);
    for (int i = 1; i < call->n_src; i++)
      if (poly_uop_get_buffer_identity(call->src[i]) == parent_buffer)
        has_parent_buffer = true;
    ASSERT_TRUE(has_parent_buffer);
  }
  ASSERT_INT_EQ(poly_run_schedule(ctx, schedule, NULL, 0), 0);

  float parent_out[4] = {0};
  float descendant_out[4] = {0};
  READ_REALIZED_F32(ctx, resolved[0], parent_out, 4);
  READ_REALIZED_F32(ctx, resolved[1], descendant_out, 4);
  for (int i = 0; i < 4; i++) {
    ASSERT_FLOAT_EQ(parent_out[i], input_data[i] + bias_data[i], 1e-6f);
    ASSERT_FLOAT_EQ(descendant_out[i], parent_out[i] * scale_data[i], 1e-6f);
  }

  poly_schedule_free(schedule);
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
    ASSERT_INT_EQ(callified_out[1]->n_src, 1);
    ASSERT_PTR_EQ(callified_out[1]->src[0], callified_out[0]);

    PolyUOp *scheduled_out[] = {NULL, NULL};
    PolySchedule *schedule = poly_schedule_with_vars(ctx, targets, 2, scheduled_out);
    ASSERT_NOT_NULL(schedule);
    ASSERT_INT_EQ(schedule->template->n_calls, 1);
    ASSERT_NOT_NULL(scheduled_out[0]);
    ASSERT_NOT_NULL(scheduled_out[1]);
    ASSERT_EQ(scheduled_out[1]->op, expected_ops[i]);
    ASSERT_INT_EQ(scheduled_out[1]->n_src, 1);
    ASSERT_PTR_EQ(scheduled_out[1]->src[0], scheduled_out[0]);
    poly_schedule_free(schedule);
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
  PolyUOp *n = poly_define_var(ctx, "n", 1, 8);
  PolyUOp *two = poly_const_int(ctx, 2);
  PolyUOp *shape_srcs[] = {n, two};
  PolyUOp *shape = poly_uop(
      ctx, POLY_OP_STACK, poly_dtype_vec(POLY_INDEX, 2),
      shape_srcs, 2, poly_arg_none());
  PolyUOp *expand_srcs[] = {source, shape};
  PolyUOp *expanded = poly_uop(
      ctx, POLY_OP_EXPAND, POLY_FLOAT32,
      expand_srcs, 2, poly_arg_none());
  ASSERT_NOT_NULL(expanded);
  PolyUOp *value =
      poly_alu2(ctx, POLY_OP_ADD, expanded, poly_const_float(ctx, 1.0f));
  PolyUOp *reshape_srcs[] = {value, shape};
  PolyUOp *root = poly_uop(
      ctx, POLY_OP_RESHAPE, POLY_FLOAT32,
      reshape_srcs, 2, poly_arg_none());
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
  PolyUOp *n = poly_define_var(ctx, "n_nonmax", 1, 8);
  PolyUOp *shape_srcs[] = {n, poly_const_int(ctx, 2)};
  PolyUOp *shape = poly_uop(
      ctx, POLY_OP_STACK, poly_dtype_vec(POLY_INDEX, 2),
      shape_srcs, 2, poly_arg_none());
  PolyUOp *expand_srcs[] = {source, shape};
  PolyUOp *expanded = poly_uop(
      ctx, POLY_OP_EXPAND, POLY_FLOAT32,
      expand_srcs, 2, poly_arg_none());
  PolyUOp *value =
      poly_alu2(ctx, POLY_OP_ADD, expanded, poly_const_float(ctx, 1.0f));
  PolyUOp *reshape_srcs[] = {value, shape};
  PolyUOp *root = poly_uop(
      ctx, POLY_OP_RESHAPE, POLY_FLOAT32,
      reshape_srcs, 2, poly_arg_none());
  ASSERT_NOT_NULL(root);
  ASSERT_INT_EQ(poly_uop_max_shape_cached(ctx, root).ndim, 2);

  float input[2] = {1.0f, 2.0f};
  poly_buffer_set(ctx, base, input, sizeof(input), POLY_DEVICE_CPU);
  PolyUOp *scheduled_out = NULL;
  PolySchedule *schedule =
      poly_schedule_with_vars(ctx, &root, 1, &scheduled_out);
  ASSERT_NOT_NULL(schedule);
  ASSERT_NOT_NULL(scheduled_out);
  ASSERT_INT_EQ(poly_uop_max_shape_cached(ctx, scheduled_out).ndim, 2);
  ASSERT_INT_EQ(count_root_ops(ctx, scheduled_out, POLY_OP_SHRINK), 1);

  PolyVarBinding bind = {.var = n, .value = 4};
  ASSERT_INT_EQ(poly_run_schedule(ctx, schedule, &bind, 1), 0);
  /* Pinned UOp.has_buffer_identity (uop/ops.py:825-829) does not unwrap
   * SHRINK. Read the backing max-allocation buffer below the exact symbolic
   * view instead of broadening Polygrad's buffer-identity contract. */
  ASSERT_INT_EQ(scheduled_out->op, POLY_OP_RESHAPE);
  ASSERT_INT_EQ(scheduled_out->src[0]->op, POLY_OP_SHRINK);
  const PolyUOp *identity =
      poly_uop_get_buffer_identity(scheduled_out->src[0]->src[0]);
  ASSERT_NOT_NULL(identity);
  float output[8] = {0};
  ASSERT_INT_EQ(
      poly_buffer_read(ctx, (PolyUOp *)identity, output, sizeof(output)), 0);
  for (int i = 0; i < 4; i++) {
    ASSERT_FLOAT_EQ(output[2 * i], 2.0f, 1e-5f);
    ASSERT_FLOAT_EQ(output[2 * i + 1], 3.0f, 1e-5f);
  }

  poly_schedule_free(schedule);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, transform_to_call_preserves_scalar_movement_shape_sources) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *reshape_value =
      poly_alu2(ctx, POLY_OP_ADD, poly_buffer_f32(ctx, 8), poly_const_float(ctx, 1.0f));
  PolyUOp *n = poly_define_var(ctx, "n", 1, 8);
  PolyUOp *reshape_src[] = {reshape_value, n};
  PolyUOp *reshape = poly_uop(
      ctx, POLY_OP_RESHAPE, POLY_FLOAT32, reshape_src, 2, poly_arg_none());
  ASSERT_NOT_NULL(reshape);

  PolyUOp *realized_reshape = NULL;
  PolyUOp *reshape_call =
      poly_transform_to_call(ctx, &reshape, 1, &realized_reshape);
  ASSERT_NOT_NULL(reshape_call);
  ASSERT_NOT_NULL(realized_reshape);
  ASSERT_INT_EQ(realized_reshape->op, POLY_OP_RESHAPE);
  ASSERT_INT_EQ(realized_reshape->n_src, 2);
  ASSERT_PTR_EQ(realized_reshape->src[1], n);

  PolyUOp *unit = poly_reshape(ctx, poly_buffer_f32(ctx, 1), (int64_t[]){1}, 1);
  PolyUOp *expand_value =
      poly_alu2(ctx, POLY_OP_ADD, unit, poly_const_float(ctx, 1.0f));
  PolyUOp *expand_src[] = {expand_value, n};
  PolyUOp *expand = poly_uop(
      ctx, POLY_OP_EXPAND, POLY_FLOAT32, expand_src, 2, poly_arg_none());
  ASSERT_NOT_NULL(expand);

  PolyUOp *realized_expand = NULL;
  PolyUOp *expand_call =
      poly_transform_to_call(ctx, &expand, 1, &realized_expand);
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
  ASSERT_INT_EQ(count_root_ops(ctx, big_call->src[0], POLY_OP_REDUCE_AXIS), N_TERMS);

  PolyUOp *output_effect = big_call->src[0]->src[0];
  ASSERT_NOT_NULL(output_effect);
  ASSERT_INT_EQ(output_effect->op, POLY_OP_AFTER);
  ASSERT_INT_EQ(output_effect->n_src, 2);
  ASSERT_INT_EQ(output_effect->src[1]->op, POLY_OP_STORE);
  ASSERT_INT_EQ(count_root_ops(ctx, output_effect->src[1]->src[1], POLY_OP_REDUCE_AXIS), N_TERMS);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, transform_to_call_shared_reduce_matches_tinygrad_topology) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer_f32(ctx, 4);
  PolyUOp *c = poly_buffer_f32(ctx, 4);
  PolyUOp *e = poly_buffer_f32(ctx, 4);
  int64_t axis[] = {0};
  int64_t one[] = {1};
  int64_t four[] = {4};
  PolyUOp *reduced = poly_reduce_axis(ctx, POLY_OP_ADD, a, axis, 1);
  PolyUOp *scalar = poly_reshape(ctx, reduced, NULL, 0);
  PolyUOp *view = poly_reshape(ctx, scalar, one, 1);
  PolyUOp *expanded = poly_expand(ctx, view, four, 1);
  PolyUOp *targets[] = {
      poly_alu2(ctx, POLY_OP_ADD, expanded, c),
      poly_alu2(ctx, POLY_OP_MUL, expanded, e),
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
  ASSERT_INT_EQ(count_root_ops(ctx, big_call->src[0], POLY_OP_REDUCE_AXIS), 1);
  ASSERT_INT_EQ(count_root_ops(ctx, big_call->src[0], POLY_OP_RESHAPE), 2);
  ASSERT_INT_EQ(count_root_ops(ctx, big_call->src[0], POLY_OP_EXPAND), 1);
  ASSERT_INT_EQ(big_call->src[0]->src[0]->op, POLY_OP_AFTER);
  ASSERT_INT_EQ(big_call->src[0]->src[1]->op, POLY_OP_AFTER);
  ASSERT_INT_EQ(poly_arena_used(ctx->scratch), scratch_before);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, schedule_with_vars_then_run_vecadd) {
  PolyCtx *ctx = poly_ctx_new();

  /* tinygrad reference boundary:
   * Tensor.schedule_with_vars(*to_realize) -> run_schedule(schedule, vars). */
  PolyUOp *a = poly_buffer_f32(ctx, 4);
  PolyUOp *b = poly_buffer_f32(ctx, 4);
  PolyUOp *add = poly_alu2(ctx, POLY_OP_ADD, a, b);

  float da[] = {1.0f, 2.0f, 3.0f, 4.0f};
  float db[] = {10.0f, 20.0f, 30.0f, 40.0f};
  poly_buffer_set(ctx, a, da, sizeof(da), POLY_DEVICE_CPU);
  poly_buffer_set(ctx, b, db, sizeof(db), POLY_DEVICE_CPU);

  PolyUOp *targets[] = {add};
  PolyUOp *realized[] = {NULL};
  PolySchedule *sched = poly_schedule_with_vars(ctx, targets, 1, realized);
  ASSERT_NOT_NULL(sched);
  ASSERT_TRUE(realized[0] != NULL);
  ASSERT_INT_EQ(sched->template->n_calls, 1);

  ASSERT_INT_EQ(poly_run_schedule(ctx, sched, NULL, 0), 0);

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

  poly_schedule_free(sched);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, run_schedule_uses_bind_default_from_ctx_buffers) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *N = poly_define_var(ctx, "N", 1, 16);
  PolyUOp *bind_N = poly_bind_var(ctx, N, 4);

  PolyUOp *unique_a = poly_uop0(ctx, POLY_OP_UNIQUE, POLY_VOID, poly_arg_int(9100000));
  PolyUOp *unique_o = poly_uop0(ctx, POLY_OP_UNIQUE, POLY_VOID, poly_arg_int(9100001));
  PolyUOp *src_a[2] = {unique_a, bind_N};
  PolyUOp *src_o[2] = {unique_o, bind_N};
  PolyUOp *a = poly_uop(ctx, POLY_OP_BUFFER, POLY_FLOAT32, src_a, 2, poly_arg_int(16));
  PolyUOp *out = poly_uop(ctx, POLY_OP_BUFFER, POLY_FLOAT32, src_o, 2, poly_arg_int(16));
  PolyUOp *add = poly_alu2(ctx, POLY_OP_ADD, a, poly_const_float(ctx, 1.0));
  PolyUOp *store = poly_store_val(ctx, out, add);
  PolyUOp *sink = poly_sink1(ctx, store);

  float a_data[16];
  float out_data[16];
  for (int i = 0; i < 16; i++) {
    a_data[i] = (float)(i + 1);
    out_data[i] = -999.0f;
  }
  poly_buffer_set(ctx, a, a_data, sizeof(a_data), POLY_DEVICE_CPU);
  poly_buffer_set(ctx, out, out_data, sizeof(out_data), POLY_DEVICE_CPU);

  PolySchedule *sched = poly_schedule_effect_sink(ctx, sink);
  ASSERT_NOT_NULL(sched);
  ASSERT_INT_EQ(sched->template->n_calls, 1);
  ASSERT_INT_EQ(sched->template->n_default_vars, 1);
  ASSERT_PTR_EQ(sched->template->default_vars[0].var, N);
  ASSERT_INT_EQ(sched->template->default_vars[0].value, 4);
  ASSERT_INT_EQ(poly_run_schedule(ctx, sched, NULL, 0), 0);

  for (int i = 0; i < 4; i++)
    ASSERT_FLOAT_EQ(out_data[i], (float)(i + 2), 1e-5f);
  ASSERT_FLOAT_EQ(out_data[4], -999.0f, 1e-5f);

  for (int i = 0; i < 16; i++)
    out_data[i] = -999.0f;
  ASSERT_INT_EQ(poly_buffer_write(ctx, out, out_data, sizeof(out_data)), 0);
  PolyVarBinding bind_6 = {.var = N, .value = 6};
  ASSERT_INT_EQ(poly_run_schedule(ctx, sched, &bind_6, 1), 0);
  ASSERT_INT_EQ(poly_buffer_read(ctx, out, out_data, sizeof(out_data)), 0);
  for (int i = 0; i < 6; i++)
    ASSERT_FLOAT_EQ(out_data[i], (float)(i + 2), 1e-5f);
  ASSERT_FLOAT_EQ(out_data[6], -999.0f, 1e-5f);

  poly_schedule_free(sched);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, run_schedule_dynamic_var_override_from_ctx_buffers) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *N = poly_define_var(ctx, "N", 1, 16);
  PolyUOp *a = poly_buffer_var(ctx, POLY_FLOAT32, N, NULL, 0);
  PolyUOp *out = poly_buffer_var(ctx, POLY_FLOAT32, N, NULL, 0);
  PolyUOp *add = poly_alu2(ctx, POLY_OP_ADD, a, poly_const_float(ctx, 1.0));
  PolyUOp *store = poly_store_val(ctx, out, add);
  PolyUOp *sink = poly_sink1(ctx, store);

  float a_data[16];
  float out_data[16];
  for (int i = 0; i < 16; i++)
    a_data[i] = (float)(i + 1);
  for (int i = 0; i < 16; i++)
    out_data[i] = -999.0f;
  poly_buffer_set(ctx, a, a_data, sizeof(a_data), POLY_DEVICE_CPU);
  poly_buffer_set(ctx, out, out_data, sizeof(out_data), POLY_DEVICE_CPU);

  PolySchedule *sched = poly_schedule_effect_sink(ctx, sink);
  ASSERT_NOT_NULL(sched);
  ASSERT_INT_EQ(poly_run_schedule(ctx, sched, NULL, 0), -1);

  PolyVarBinding bind = {.var = N, .value = 6};
  ASSERT_INT_EQ(poly_run_schedule(ctx, sched, &bind, 1), 0);
  for (int i = 0; i < 6; i++)
    ASSERT_FLOAT_EQ(out_data[i], (float)(i + 2), 1e-5f);
  ASSERT_FLOAT_EQ(out_data[6], -999.0f, 1e-5f);

  poly_schedule_free(sched);
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
  PolySchedule *sched = poly_schedule_with_vars(ctx, targets, 1, realized);
  ASSERT_NOT_NULL(sched);
  ASSERT_INT_EQ(sched->template->n_calls, 1);
  ASSERT_INT_EQ(count_root_ops(ctx, poly_schedule_call_body(sched, 0), POLY_OP_REDUCE), 1);

  ASSERT_INT_EQ(poly_run_schedule(ctx, sched, NULL, 0), 0);
  ASSERT_NOT_NULL(realized[0]);
  const PolyUOp *out = poly_uop_get_buffer_identity(realized[0]);
  ASSERT_NOT_NULL(out);
  PolyBuffer *buf = poly_buffer_get(ctx, (PolyUOp *)out);
  ASSERT_NOT_NULL(buf);
  float reduced = 0.0f;
  READ_REALIZED_F32(ctx, realized[0], &reduced, 1);
  ASSERT_FLOAT_EQ(reduced, 10.0f, 1e-5f);

  poly_schedule_free(sched);
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
  PolySchedule *sched = poly_schedule_with_vars(ctx, targets, 1, realized);
  ASSERT_NOT_NULL(sched);
  ASSERT_INT_EQ(sched->template->n_calls, 1);
  ASSERT_NOT_NULL(realized[0]);

  PolyUOp *root = poly_schedule_call_body(sched, 0);
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_STORE), 1);
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_REDUCE), 1);
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_RANGE), 2);
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_END), 1);
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_INDEX), 2);
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_WHERE), 0);
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_REDUCE_AXIS), 0);

  poly_schedule_free(sched);
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
  PolySchedule *sched = poly_schedule_with_vars(ctx, targets, 1, realized);
  ASSERT_NOT_NULL(sched);
  ASSERT_INT_EQ(sched->template->n_calls, 1);
  ASSERT_NOT_NULL(realized[0]);

  PolyUOp *root = poly_schedule_call_body(sched, 0);
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_STORE), 1);
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_REDUCE), 1);
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_RANGE), 2);
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_END), 1);
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_INDEX), 2);
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_WHERE), 0);
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_REDUCE_AXIS), 0);

  poly_schedule_free(sched);
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
  PolySchedule *sched = poly_schedule_with_vars(ctx, targets, 1, realized);
  ASSERT_NOT_NULL(sched);
  ASSERT_INT_EQ(sched->template->n_calls, 1);
  ASSERT_NOT_NULL(realized[0]);

  PolyUOp *root = poly_schedule_call_body(sched, 0);
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_STORE), 1);
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_REDUCE), 0);
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_RANGE), 2);
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_END), 1);
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_INDEX), 3);
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_WHERE), 0);
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_LOAD), 0);
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_STAGE), 0);

  poly_schedule_free(sched);
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
  PolySchedule *sched = poly_schedule_with_vars(ctx, targets, 1, realized);
  ASSERT_NOT_NULL(sched);
  ASSERT_INT_EQ(sched->template->n_calls, 1);
  ASSERT_NOT_NULL(realized[0]);

  PolyUOp *root = poly_schedule_call_body(sched, 0);
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_STORE), 1);
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_REDUCE), 0);
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_RANGE), 1);
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_END), 1);
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_INDEX), 3);
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_WHERE), 0);
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_LOAD), 0);
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_STAGE), 0);
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_RESHAPE), 0);

  poly_schedule_free(sched);
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
  PolySchedule *sched = poly_schedule_with_vars(ctx, targets, 1, realized);
  ASSERT_NOT_NULL(sched);
  ASSERT_INT_EQ(sched->template->n_calls, 1);
  ASSERT_NOT_NULL(realized[0]);

  PolyUOp *root = poly_schedule_call_body(sched, 0);
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

  poly_schedule_free(sched);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, schedule_with_vars_expand_add_matches_tinygrad_counts) {
  PolyCtx *ctx = poly_ctx_new();

  /* tinygrad reference for (Tensor.empty(2,1) + Tensor.empty(2,1)).expand(2,4):
   * SCHEDULE_LEN=1 and the scheduled root contains exactly
   * STORE=1 REDUCE=0 RANGE=1 END=1 INDEX=3 WHERE=0 LOAD=0 BUFFERIZE=0.
   * The frontend-visible realized value remains EXPAND(RESHAPE(BUFFER), ...). */
  PolyUOp *a_buf = poly_buffer_f32(ctx, 2);
  PolyUOp *b_buf = poly_buffer_f32(ctx, 2);
  PolyUOp *a = poly_reshape(ctx, a_buf, (int64_t[]){2, 1}, 2);
  PolyUOp *b = poly_reshape(ctx, b_buf, (int64_t[]){2, 1}, 2);
  PolyUOp *x = poly_alu2(ctx, POLY_OP_ADD, a, b);
  PolyUOp *y = poly_expand(ctx, x, (int64_t[]){2, 4}, 2);

  PolyUOp *targets[] = {y};
  PolyUOp *realized[] = {NULL};
  PolySchedule *sched = poly_schedule_with_vars(ctx, targets, 1, realized);
  ASSERT_NOT_NULL(sched);
  ASSERT_INT_EQ(sched->template->n_calls, 1);
  ASSERT_NOT_NULL(realized[0]);

  PolyUOp *root = poly_schedule_call_body(sched, 0);
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_STORE), 1);
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_REDUCE), 0);
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_RANGE), 1);
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_END), 1);
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_INDEX), 3);
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_WHERE), 0);
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_LOAD), 0);
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_STAGE), 0);
  ASSERT_INT_EQ(realized[0]->op, POLY_OP_EXPAND);
  ASSERT_TRUE(realized[0]->n_src >= 1);
  ASSERT_INT_EQ(realized[0]->src[0]->op, POLY_OP_RESHAPE);
  ASSERT_TRUE(poly_uop_get_buffer_identity(realized[0]->src[0]) != NULL);

  poly_schedule_free(sched);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, schedule_with_vars_cross_entropy_sparse_last_axis_matches_tinygrad_kernel_count) {
  PolyCtx *ctx = poly_ctx_new();

  /* tinygrad reference for Tensor.zeros(2,3).cross_entropy(Tensor([0,2], dtype=int32)):
   * sched_len=3. The final *-1/N stays fused into the last reduction kernel. */
  PolyUOp *logits_buf = poly_buffer_f32(ctx, 6);
  PolyUOp *target_buf = poly_buffer(ctx, POLY_INT32, 2);
  PolyUOp *logits = poly_reshape(ctx, logits_buf, (int64_t[]){2, 3}, 2);
  PolyUOp *target = poly_reshape(ctx, target_buf, (int64_t[]){2}, 1);
  PolyUOp *loss = poly_cross_entropy(ctx, logits, target, 1);

  PolyUOp *targets[] = {loss};
  PolyUOp *realized[] = {NULL};
  PolySchedule *sched = poly_schedule_with_vars(ctx, targets, 1, realized);
  ASSERT_NOT_NULL(sched);
  ASSERT_NOT_NULL(realized[0]);
  ASSERT_INT_EQ(sched->template->n_calls, 3);

  poly_schedule_free(sched);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, schedule_with_vars_cross_entropy_sparse_non_last_axis_matches_tinygrad_kernel_count) {
  PolyCtx *ctx = poly_ctx_new();

  /* tinygrad reference for Tensor.zeros(2,3,2).cross_entropy(target int32[2,2]):
   * sched_len=3. The final *-1/N stays fused into the last reduction kernel. */
  PolyUOp *logits_buf = poly_buffer_f32(ctx, 12);
  PolyUOp *target_buf = poly_buffer(ctx, POLY_INT32, 4);
  PolyUOp *logits = poly_reshape(ctx, logits_buf, (int64_t[]){2, 3, 2}, 3);
  PolyUOp *target = poly_reshape(ctx, target_buf, (int64_t[]){2, 2}, 2);
  PolyUOp *loss = poly_cross_entropy(ctx, logits, target, -2);

  PolyUOp *targets[] = {loss};
  PolyUOp *realized[] = {NULL};
  PolySchedule *sched = poly_schedule_with_vars(ctx, targets, 1, realized);
  ASSERT_NOT_NULL(sched);
  ASSERT_NOT_NULL(realized[0]);
  ASSERT_INT_EQ(sched->template->n_calls, 3);

  poly_schedule_free(sched);
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
  PolySchedule *sched = poly_schedule_with_vars(ctx, targets, 2, realized);
  ASSERT_NOT_NULL(sched);
  ASSERT_PTR_EQ(realized[0], a);
  ASSERT_NOT_NULL(realized[1]);
  ASSERT_PTR_NEQ(realized[1], add);

  ASSERT_INT_EQ(poly_run_schedule(ctx, sched, NULL, 0), 0);

  PolyBuffer *buf = realized_buffer(ctx, realized[1]);
  ASSERT_NOT_NULL(buf);
  float dout[4];
  READ_REALIZED_F32(ctx, realized[1], dout, 4);
  ASSERT_FLOAT_EQ(dout[0], 11.0f, 1e-5f);
  ASSERT_FLOAT_EQ(dout[1], 22.0f, 1e-5f);
  ASSERT_FLOAT_EQ(dout[2], 33.0f, 1e-5f);
  ASSERT_FLOAT_EQ(dout[3], 44.0f, 1e-5f);

  poly_schedule_free(sched);
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
  PolySchedule *sched = poly_schedule_with_vars(ctx, targets, 2, realized);
  ASSERT_NOT_NULL(sched);
  ASSERT_NOT_NULL(realized[0]);
  ASSERT_NOT_NULL(realized[1]);

  ASSERT_INT_EQ(poly_run_schedule(ctx, sched, NULL, 0), 0);

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

  poly_schedule_free(sched);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, tensor_zero_size_placement_copy_adds_no_call) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *logical = poly_full(ctx, (int64_t[]){0}, 1, 0.0);
  PolyTensor *empty = poly_tensor_create(ctx, logical, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(logical);
  ASSERT_NOT_NULL(empty);

  PolyUOp *physical = poly_tensor_physicalize(ctx, empty);
  ASSERT_NOT_NULL(physical);
  ASSERT_EQ(physical->op, POLY_OP_COPY);

  PolyTensor *realized_tensor = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &empty, 1, &realized_tensor), 0);
  ASSERT_PTR_EQ(realized_tensor, empty);
  PolyUOp *realized = poly_tensor_uop(empty);
  ASSERT_NOT_NULL(realized);
  const PolyUOp *identity = poly_uop_get_buffer_identity(realized);
  ASSERT_NOT_NULL(identity);
  ASSERT_EQ(identity->op, POLY_OP_BUFFER);
  ASSERT_PTR_NEQ(identity, poly_uop_get_buffer_identity(logical));
  ASSERT_INT_EQ(identity->arg.i, 0);
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

TEST(realize, tensor_zero_axis_short_circuits_earlier_numel_overflow) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  int64_t dims[3] = {INT64_MAX, 2, 0};
  PolyUOp *logical = poly_full(ctx, dims, 3, 0.0);
  PolyTensor *empty = poly_tensor_create(ctx, logical, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(logical);
  ASSERT_NOT_NULL(empty);

  PolyTensor *realized_tensor = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &empty, 1, &realized_tensor), 0);
  ASSERT_PTR_EQ(realized_tensor, empty);
  PolyUOp *realized = poly_tensor_uop(empty);
  ASSERT_NOT_NULL(realized);
  ASSERT_EQ(realized->op, POLY_OP_RESHAPE);
  PolyShape shape = poly_uop_max_shape_cached(ctx, realized);
  ASSERT_INT_EQ(shape.ndim, 3);
  ASSERT_TRUE(shape.dims[0] == INT64_MAX);
  ASSERT_INT_EQ(shape.dims[1], 2);
  ASSERT_INT_EQ(shape.dims[2], 0);

  const PolyUOp *identity = poly_uop_get_buffer_identity(realized);
  ASSERT_NOT_NULL(identity);
  ASSERT_EQ(identity->op, POLY_OP_BUFFER);
  ASSERT_INT_EQ(identity->arg.i, 0);
  ASSERT_TRUE(poly_buffer_get(ctx, (PolyUOp *)identity) == NULL);
  ASSERT_FALSE(poly_buffer_is_allocated(ctx, (PolyUOp *)identity));
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
  PolySchedule *sched = poly_schedule_with_vars(ctx, targets, 1, realized);
  ASSERT_NOT_NULL(sched);
  ASSERT_NOT_NULL(realized[0]);
  ASSERT_EQ(realized[0]->op, POLY_OP_RESHAPE);
  PolyShape s = poly_uop_max_shape_cached(ctx, realized[0]);
  PolyShape expected = {dims, 2};
  ASSERT_TRUE(poly_shape_eq(s, expected));

  ASSERT_INT_EQ(poly_run_schedule(ctx, sched, NULL, 0), 0);
  PolyBuffer *buf = realized_buffer(ctx, realized[0]);
  ASSERT_NOT_NULL(buf);
  float dout[6];
  READ_REALIZED_F32(ctx, realized[0], dout, 6);
  ASSERT_FLOAT_EQ(dout[0], 11.0f, 1e-5f);
  ASSERT_FLOAT_EQ(dout[5], 66.0f, 1e-5f);

  poly_schedule_free(sched);
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
  PolySchedule *sched = poly_schedule_with_vars(ctx, targets, 1, realized);
  ASSERT_NOT_NULL(sched);
  ASSERT_PTR_EQ(realized[0], a);

  ASSERT_INT_EQ(poly_run_schedule(ctx, sched, NULL, 0), 0);
  ASSERT_FLOAT_EQ(da[0], 2.0f, 1e-5f);
  ASSERT_FLOAT_EQ(da[1], 4.0f, 1e-5f);
  ASSERT_FLOAT_EQ(da[2], 6.0f, 1e-5f);
  ASSERT_FLOAT_EQ(da[3], 8.0f, 1e-5f);

  poly_schedule_free(sched);
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
    PolySchedule *schedule = poly_schedule_with_vars(ctx, targets, 2, realized);
    ASSERT_NOT_NULL(schedule);
    ASSERT_INT_EQ(schedule->template->n_calls, 2);
    ASSERT_INT_EQ(poly_run_schedule(ctx, schedule, NULL, 0), 0);
    ASSERT_FLOAT_EQ(da[0], 1.0f, 1e-5f);
    ASSERT_FLOAT_EQ(db[0], 1.0f, 1e-5f);

    poly_schedule_free(schedule);
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
  PolySchedule *sched = poly_schedule_with_vars(ctx, targets, 1, realized);
  ASSERT_NOT_NULL(sched);
  ASSERT_INT_EQ(poly_run_schedule(ctx, sched, NULL, 0), 0);
  ASSERT_NOT_NULL(poly_buffer_get(ctx, b));
  ASSERT_NOT_NULL(realized[0]);
  ASSERT_NOT_NULL(realized_buffer(ctx, realized[0]));

  poly_schedule_free(sched);
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
  PolySchedule *sched = poly_schedule_with_vars(ctx, targets, 1, realized);
  ASSERT_NOT_NULL(sched);
  ASSERT_INT_EQ(sched->template->n_calls, 1);
  ASSERT_NOT_NULL(realized[0]);

  /* tinygrad schedule_with_vars boundary: triu root still has WHERE and no
   * LOAD. Late load insertion happens later in codegen pm_add_loads. */
  ASSERT_INT_EQ(count_root_ops(ctx, poly_schedule_call_body(sched, 0), POLY_OP_LOAD), 0);
  ASSERT_TRUE(count_root_ops(ctx, poly_schedule_call_body(sched, 0), POLY_OP_WHERE) > 0);

  ASSERT_INT_EQ(poly_run_schedule(ctx, sched, NULL, 0), 0);
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

  poly_schedule_free(sched);
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
  PolySchedule *sched = poly_schedule_with_vars(ctx, targets, 1, realized);
  ASSERT_NOT_NULL(sched);
  ASSERT_INT_EQ(sched->template->n_calls, 1);
  ASSERT_NOT_NULL(realized[0]);

  ASSERT_INT_EQ(count_root_ops(ctx, poly_schedule_call_body(sched, 0), POLY_OP_LOAD), 0);
  ASSERT_TRUE(count_root_ops(ctx, poly_schedule_call_body(sched, 0), POLY_OP_WHERE) > 0);

  ASSERT_INT_EQ(poly_run_schedule(ctx, sched, NULL, 0), 0);
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

  poly_schedule_free(sched);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, poly_jit_replays_raw_tensor_realize_with_new_input) {
  PolyCtx *ctx = poly_ctx_new();

  float a_data[3] = {1.0f, 2.0f, 3.0f};
  PolyUOp *a_buf = poly_buffer_f32(ctx, 3);
  poly_buffer_set(ctx, a_buf, a_data, sizeof(a_data), POLY_DEVICE_CPU);
  PolyTensor *a = poly_tensor_create(ctx, a_buf, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(a);

  PolyJit *jit = poly_jit_new(ctx);
  ASSERT_NOT_NULL(jit);
  ASSERT_INT_EQ(poly_jit_begin_capture(jit, &a, 1), 0);
  poly_ctx_reset_counters(ctx);

  PolyUOp *expr = poly_alu2(ctx, POLY_OP_ADD, poly_tensor_uop(a), poly_const_float(ctx, 1.0));
  PolyTensor *out = poly_tensor_create(ctx, expr, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
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

  ASSERT_INT_EQ(poly_jit_end_capture(jit), 0);
  ASSERT_TRUE(poly_jit_is_captured(jit));
  ASSERT_INT_EQ(poly_jit_schedule_count(jit), 1);
  PolyUOp *captured = poly_jit_captured_linear(jit);
  ASSERT_NOT_NULL(captured);
  ASSERT_INT_EQ(captured->op, POLY_OP_LINEAR);
  ASSERT_INT_EQ(count_shaped_value_params(ctx, captured), 1);
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
  PolyUOp *b_buf = poly_buffer_f32(ctx, 3);
  poly_buffer_set(ctx, b_buf, b_data, sizeof(b_data), POLY_DEVICE_CPU);
  PolyTensor *b = poly_tensor_create(ctx, b_buf, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
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

TEST(realize, poly_jit_capture_skips_already_current_tensor) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  float data[3] = {1.0f, 2.0f, 3.0f};
  PolyUOp *buf = poly_buffer_f32(ctx, 3);
  ASSERT_NOT_NULL(buf);
  poly_buffer_set(ctx, buf, data, sizeof(data), POLY_DEVICE_CPU);
  PolyTensor *input = poly_tensor_create(ctx, buf, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
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
  PolyUOp *buf = poly_buffer_f32(ctx, 3);
  ASSERT_NOT_NULL(buf);
  ASSERT_INT_EQ(poly_buffer_write(ctx, buf, data, sizeof(data)), 0);
  PolyTensor *input = poly_tensor_create(ctx, buf, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(input);

  PolyJit *outer = poly_jit_new(ctx);
  PolyJit *other = poly_jit_new(ctx);
  ASSERT_NOT_NULL(outer);
  ASSERT_NOT_NULL(other);
  ASSERT_INT_EQ(poly_jit_begin_capture(outer, &input, 1), 0);

  PolyTensor *out = poly_tensor_create(
      ctx, poly_alu2(ctx, POLY_OP_ADD, poly_tensor_uop(input), poly_const_float(ctx, 1.0)),
      POLY_TENSOR_VALUE, POLY_DEVICE_CPU
  );
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
  PolyUOp *a_buf = poly_buffer_f32(ctx, 5);
  ASSERT_INT_EQ(poly_buffer_write(ctx, a_buf, a_data, sizeof(a_data)), 0);
  int64_t a_bounds[1][2] = {{1, 4}};
  PolyUOp *a_view = poly_shrink(ctx, a_buf, a_bounds, 1);
  ASSERT_NOT_NULL(a_view);
  ASSERT_TRUE(poly_uop_get_buffer_identity(a_view) == NULL);
  PolyTensor *a = poly_tensor_create(ctx, a_view, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(a);

  PolyJit *jit = poly_jit_new(ctx);
  ASSERT_NOT_NULL(jit);
  ASSERT_INT_EQ(poly_jit_begin_capture(jit, &a, 1), 0);

  PolyUOp *expr = poly_alu2(ctx, POLY_OP_ADD, poly_tensor_uop(a), poly_const_float(ctx, 1.0));
  PolyTensor *out = poly_tensor_create(ctx, expr, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(out);
  PolyTensor *realized = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &out, 1, &realized), 0);
  ASSERT_PTR_EQ(realized, out);
  ASSERT_INT_EQ(poly_jit_end_capture(jit), 0);

  float b_data[5] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f};
  PolyUOp *b_buf = poly_buffer_f32(ctx, 5);
  ASSERT_INT_EQ(poly_buffer_write(ctx, b_buf, b_data, sizeof(b_data)), 0);
  int64_t b_bounds[1][2] = {{1, 4}};
  PolyUOp *b_view = poly_shrink(ctx, b_buf, b_bounds, 1);
  ASSERT_NOT_NULL(b_view);
  PolyTensor *b = poly_tensor_create(ctx, b_view, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
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
  PolyUOp *input_buf = poly_buffer_f32(ctx, 6);
  ASSERT_NOT_NULL(input_buf);
  ASSERT_INT_EQ(poly_buffer_write(ctx, input_buf, input_data, sizeof(input_data)), 0);

  PolyUOp *i = poly_define_var(ctx, "i", 0, 4);
  PolyUOp *size_2 = poly_const_int(ctx, 2);
  PolyUOp *capture_start = poly_bind_var(ctx, i, 2);
  PolyUOp *capture_starts[] = {capture_start};
  PolyUOp *capture_sizes[] = {size_2};
  PolyUOp *capture_view = poly_shrink_uop(ctx, input_buf, capture_starts, capture_sizes, 1);
  ASSERT_NOT_NULL(capture_view);
  ASSERT_INT_EQ(capture_view->op, POLY_OP_SHRINK);
  ASSERT_INT_EQ(count_root_ops(ctx, capture_view, POLY_OP_BIND), 1);
  PolyTensor *capture_input =
      poly_tensor_create(ctx, capture_view, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(capture_input);

  PolyJit *jit = poly_jit_new(ctx);
  ASSERT_NOT_NULL(jit);
  ASSERT_INT_EQ(poly_jit_begin_capture(jit, &capture_input, 1), 0);
  PolyTensor *out = poly_tensor_create(
      ctx,
      poly_alu2(ctx, POLY_OP_ADD, poly_tensor_uop(capture_input), poly_const_float(ctx, 100.0f)),
      POLY_TENSOR_VALUE, POLY_DEVICE_CPU
  );
  ASSERT_NOT_NULL(out);
  PolyTensor *realized = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &out, 1, &realized), 0);
  ASSERT_PTR_EQ(realized, out);
  ASSERT_INT_EQ(poly_jit_end_capture(jit), 0);
  ASSERT_TRUE(poly_jit_is_captured(jit));
  ASSERT_INT_EQ(count_root_ops(ctx, capture_view, POLY_OP_BIND), 1);

  PolyUOp *out_buf = (PolyUOp *)poly_uop_get_buffer_identity(poly_tensor_uop(out));
  ASSERT_NOT_NULL(out_buf);
  float got[2] = {0};
  ASSERT_INT_EQ(poly_buffer_read(ctx, out_buf, got, sizeof(got)), 0);
  ASSERT_FLOAT_EQ(got[0], 102.0f, 1e-5f);
  ASSERT_FLOAT_EQ(got[1], 103.0f, 1e-5f);

  PolyUOp *replay_start_4 = poly_bind_var(ctx, i, 4);
  PolyUOp *replay_starts_4[] = {replay_start_4};
  PolyUOp *replay_view_4 = poly_shrink_uop(ctx, input_buf, replay_starts_4, capture_sizes, 1);
  ASSERT_NOT_NULL(replay_view_4);
  ASSERT_INT_EQ(count_root_ops(ctx, replay_view_4, POLY_OP_BIND), 1);
  PolyTensor *replay_input_4 =
      poly_tensor_create(ctx, replay_view_4, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(replay_input_4);
  ASSERT_INT_EQ(poly_jit_run(jit, &replay_input_4, 1), 0);
  ASSERT_INT_EQ(poly_buffer_read(ctx, out_buf, got, sizeof(got)), 0);
  ASSERT_FLOAT_EQ(got[0], 104.0f, 1e-5f);
  ASSERT_FLOAT_EQ(got[1], 105.0f, 1e-5f);

  PolyUOp *replay_start_0 = poly_bind_var(ctx, i, 0);
  PolyUOp *replay_starts_0[] = {replay_start_0};
  PolyUOp *replay_view_0 = poly_shrink_uop(ctx, input_buf, replay_starts_0, capture_sizes, 1);
  ASSERT_NOT_NULL(replay_view_0);
  PolyTensor *replay_input_0 =
      poly_tensor_create(ctx, replay_view_0, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(replay_input_0);
  ASSERT_INT_EQ(poly_jit_run(jit, &replay_input_0, 1), 0);
  ASSERT_INT_EQ(poly_buffer_read(ctx, out_buf, got, sizeof(got)), 0);
  ASSERT_FLOAT_EQ(got[0], 100.0f, 1e-5f);
  ASSERT_FLOAT_EQ(got[1], 101.0f, 1e-5f);

  /* Pinned expected_input_info includes the unbound input view, so changing
   * the fixed slice size is an input-signature mismatch, not a new binding. */
  PolyUOp *size_3 = poly_const_int(ctx, 3);
  PolyUOp *mismatch_sizes[] = {size_3};
  PolyUOp *mismatch_view = poly_shrink_uop(ctx, input_buf, replay_starts_0, mismatch_sizes, 1);
  ASSERT_NOT_NULL(mismatch_view);
  PolyTensor *mismatch_input =
      poly_tensor_create(ctx, mismatch_view, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
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
  PolyUOp *a_buf = poly_buffer_f32(ctx, 4);
  PolyUOp *b_buf = poly_buffer_f32(ctx, 4);
  ASSERT_INT_EQ(poly_buffer_write(ctx, a_buf, a_data, sizeof(a_data)), 0);
  ASSERT_INT_EQ(poly_buffer_write(ctx, b_buf, b_data, sizeof(b_data)), 0);
  PolyTensor *a = poly_tensor_create(ctx, a_buf, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  PolyTensor *b = poly_tensor_create(ctx, b_buf, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
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
  ASSERT_INT_EQ(poly_jit_end_capture(jit), 0);
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
  float b_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  PolyUOp *a_buf = poly_buffer_f32(ctx, 8);
  PolyUOp *b_buf = poly_buffer_f32(ctx, 4);
  ASSERT_INT_EQ(poly_buffer_write(ctx, a_buf, a_data, sizeof(a_data)), 0);
  ASSERT_INT_EQ(poly_buffer_write(ctx, b_buf, b_data, sizeof(b_data)), 0);
  PolyTensor *a = poly_tensor_create(ctx, a_buf, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  PolyTensor *b = poly_tensor_create(ctx, b_buf, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(a);
  ASSERT_NOT_NULL(b);

  PolyJit *jit = poly_jit_new(ctx);
  ASSERT_NOT_NULL(jit);
  PolyTensor *inputs[2] = {a, b};
  ASSERT_INT_EQ(poly_jit_begin_capture(jit, inputs, 2), 0);

  PolyUOp *out_buf = NULL;
  PolyTensor *out = custom_summary_tensor(ctx, a, b, &out_buf);
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
  ASSERT_INT_EQ(poly_jit_end_capture(jit), 0);
  ASSERT_TRUE(poly_jit_is_captured(jit));

  float got[2] = {0};
  ASSERT_INT_EQ(poly_buffer_read(ctx, out_buf, got, sizeof(got)), 0);
  ASSERT_FLOAT_EQ(got[0], 30.0f, 1e-5f);
  ASSERT_FLOAT_EQ(got[1], 70.0f, 1e-5f);

  float a_update[8] = {2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
  ASSERT_INT_EQ(poly_buffer_write(ctx, a_buf, a_update, sizeof(a_update)), 0);
  ASSERT_INT_EQ(poly_jit_run(jit, inputs, 2), 0);
  memset(got, 0, sizeof(got));
  ASSERT_INT_EQ(poly_buffer_read(ctx, out_buf, got, sizeof(got)), 0);
  ASSERT_FLOAT_EQ(got[0], 40.0f, 1e-5f);
  ASSERT_FLOAT_EQ(got[1], 80.0f, 1e-5f);

  poly_jit_free(jit);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(
    realize,
    poly_jit_captures_custom_multi_output_reduction_call_and_replays_after_input_mutation
) {
  PolyCtx *ctx = poly_ctx_new();

  float a_data[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  float b_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  PolyUOp *a_buf = poly_buffer_f32(ctx, 8);
  PolyUOp *b_buf = poly_buffer_f32(ctx, 4);
  ASSERT_INT_EQ(poly_buffer_write(ctx, a_buf, a_data, sizeof(a_data)), 0);
  ASSERT_INT_EQ(poly_buffer_write(ctx, b_buf, b_data, sizeof(b_data)), 0);
  PolyTensor *a = poly_tensor_create(ctx, a_buf, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  PolyTensor *b = poly_tensor_create(ctx, b_buf, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
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
  ASSERT_INT_EQ(poly_jit_end_capture(jit), 0);
  ASSERT_TRUE(poly_jit_is_captured(jit));

  float got0[2] = {0};
  float got1[2] = {0};
  ASSERT_INT_EQ(poly_buffer_read(ctx, out0_buf, got0, sizeof(got0)), 0);
  ASSERT_INT_EQ(poly_buffer_read(ctx, out1_buf, got1, sizeof(got1)), 0);
  ASSERT_FLOAT_EQ(got0[0], 10.0f, 1e-5f);
  ASSERT_FLOAT_EQ(got0[1], 26.0f, 1e-5f);
  ASSERT_FLOAT_EQ(got1[0], 30.0f, 1e-5f);
  ASSERT_FLOAT_EQ(got1[1], 70.0f, 1e-5f);

  float a_update[8] = {2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
  ASSERT_INT_EQ(poly_buffer_write(ctx, a_buf, a_update, sizeof(a_update)), 0);
  ASSERT_INT_EQ(poly_jit_run(jit, inputs, 2), 0);
  memset(got0, 0, sizeof(got0));
  memset(got1, 0, sizeof(got1));
  ASSERT_INT_EQ(poly_buffer_read(ctx, out0_buf, got0, sizeof(got0)), 0);
  ASSERT_INT_EQ(poly_buffer_read(ctx, out1_buf, got1, sizeof(got1)), 0);
  ASSERT_FLOAT_EQ(got0[0], 14.0f, 1e-5f);
  ASSERT_FLOAT_EQ(got0[1], 30.0f, 1e-5f);
  ASSERT_FLOAT_EQ(got1[0], 40.0f, 1e-5f);
  ASSERT_FLOAT_EQ(got1[1], 80.0f, 1e-5f);

  poly_jit_free(jit);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, custom_grouped_compact_summary_keeps_independent_reduce_per_output) {
  PolyCtx *ctx = poly_ctx_new();

  float x_data[256];
  float y_data[128];
  for (int i = 0; i < 256; i++)
    x_data[i] = (float)(i + 1);
  for (int i = 0; i < 128; i++)
    y_data[i] = (float)(i + 1);
  PolyUOp *x_buf = poly_buffer_f32(ctx, 256);
  PolyUOp *y_buf = poly_buffer_f32(ctx, 128);
  ASSERT_INT_EQ(poly_buffer_write(ctx, x_buf, x_data, sizeof(x_data)), 0);
  ASSERT_INT_EQ(poly_buffer_write(ctx, y_buf, y_data, sizeof(y_data)), 0);
  PolyTensor *x = poly_tensor_create(ctx, x_buf, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  PolyTensor *y = poly_tensor_create(ctx, y_buf, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(x);
  ASSERT_NOT_NULL(y);

  PolyUOp *out_buf = NULL;
  PolyTensor *out = custom_grouped_intercept_summary_tensor(ctx, x, y, &out_buf);
  ASSERT_NOT_NULL(out);
  ASSERT_NOT_NULL(out_buf);
  PolyTensor *realized = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &out, 1, &realized), 0);
  ASSERT_PTR_EQ(realized, out);

  float got[10] = {0};
  ASSERT_INT_EQ(poly_buffer_read(ctx, out_buf, got, sizeof(got)), 0);
  float expected[10] = {128.0f, 128.0f, 0};
  for (int c = 0; c < 2; c++) {
    for (int r = 0; r < 128; r++) {
      float x = x_data[c * 128 + r];
      float y = y_data[r];
      expected[c + 2] += x;
      expected[c + 4] += x * x;
      expected[c + 6] += y;
      expected[c + 8] += x * y;
    }
  }
  for (int i = 0; i < 10; i++)
    ASSERT_FLOAT_EQ(got[i], expected[i], 1e-5f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, poly_jit_replays_assign_against_current_input) {
  PolyCtx *ctx = poly_ctx_new();

  float a_data[3] = {1.0f, 2.0f, 3.0f};
  PolyUOp *a_buf = poly_buffer_f32(ctx, 3);
  poly_buffer_set(ctx, a_buf, a_data, sizeof(a_data), POLY_DEVICE_CPU);
  PolyTensor *a = poly_tensor_create(ctx, a_buf, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(a);

  PolyJit *jit = poly_jit_new(ctx);
  ASSERT_NOT_NULL(jit);
  ASSERT_INT_EQ(poly_jit_begin_capture(jit, &a, 1), 0);

  PolyUOp *add = poly_alu2(ctx, POLY_OP_ADD, poly_tensor_uop(a), poly_const_float(ctx, 1.0));
  PolyTensor *value = poly_tensor_create(ctx, add, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(value);
  ASSERT_PTR_EQ(poly_tensor_assign(ctx, a, value), a);
  PolyTensor *realized = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &a, 1, &realized), 0);
  ASSERT_PTR_EQ(realized, a);
  ASSERT_INT_EQ(poly_jit_end_capture(jit), 0);

  float captured_after[3] = {0};
  ASSERT_INT_EQ(poly_buffer_read(ctx, a_buf, captured_after, sizeof(captured_after)), 0);
  ASSERT_FLOAT_EQ(captured_after[0], 2.0f, 1e-5f);
  ASSERT_FLOAT_EQ(captured_after[1], 3.0f, 1e-5f);
  ASSERT_FLOAT_EQ(captured_after[2], 4.0f, 1e-5f);

  float b_data[3] = {100.0f, 200.0f, 300.0f};
  PolyUOp *b_buf = poly_buffer_f32(ctx, 3);
  poly_buffer_set(ctx, b_buf, b_data, sizeof(b_data), POLY_DEVICE_CPU);
  PolyTensor *b = poly_tensor_create(ctx, b_buf, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
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

TEST(realize, poly_jit_combines_multiple_captured_realizes) {
  PolyCtx *ctx = poly_ctx_new();

  float a_data[3] = {1.0f, 2.0f, 3.0f};
  PolyUOp *a_buf = poly_buffer_f32(ctx, 3);
  poly_buffer_set(ctx, a_buf, a_data, sizeof(a_data), POLY_DEVICE_CPU);
  PolyTensor *a = poly_tensor_create(ctx, a_buf, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(a);

  PolyJit *jit = poly_jit_new(ctx);
  ASSERT_NOT_NULL(jit);
  ASSERT_INT_EQ(poly_jit_begin_capture(jit, &a, 1), 0);

  PolyTensor *y = poly_tensor_create(
      ctx, poly_alu2(ctx, POLY_OP_ADD, poly_tensor_uop(a), poly_const_float(ctx, 1.0)),
      POLY_TENSOR_VALUE, POLY_DEVICE_CPU
  );
  ASSERT_NOT_NULL(y);
  PolyTensor *realized_y = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &y, 1, &realized_y), 0);

  PolyTensor *z = poly_tensor_create(
      ctx, poly_alu2(ctx, POLY_OP_MUL, poly_tensor_uop(a), poly_const_float(ctx, 2.0)),
      POLY_TENSOR_VALUE, POLY_DEVICE_CPU
  );
  ASSERT_NOT_NULL(z);
  PolyTensor *realized_z = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &z, 1, &realized_z), 0);

  ASSERT_INT_EQ(poly_jit_end_capture(jit), 0);
  ASSERT_INT_EQ(poly_jit_schedule_count(jit), 2);

  float b_data[3] = {10.0f, 20.0f, 30.0f};
  PolyUOp *b_buf = poly_buffer_f32(ctx, 3);
  poly_buffer_set(ctx, b_buf, b_data, sizeof(b_data), POLY_DEVICE_CPU);
  PolyTensor *b = poly_tensor_create(ctx, b_buf, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
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
  PolyUOp *a_buf = poly_buffer_f32(ctx, 3);
  poly_buffer_set(ctx, a_buf, a_data, sizeof(a_data), POLY_DEVICE_CPU);
  PolyTensor *a = poly_tensor_create(ctx, a_buf, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(a);

  float side_data[3] = {-1.0f, -1.0f, -1.0f};
  float seed_data[3] = {7.0f, 8.0f, 9.0f};
  PolyUOp *side_buf = poly_buffer_f32(ctx, 3);
  PolyUOp *seed_buf = poly_buffer_f32(ctx, 3);
  poly_buffer_set(ctx, side_buf, side_data, sizeof(side_data), POLY_DEVICE_CPU);
  poly_buffer_set(ctx, seed_buf, seed_data, sizeof(seed_data), POLY_DEVICE_CPU);
  PolyTensor *side = poly_tensor_create(ctx, side_buf, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  PolyTensor *seed = poly_tensor_create(ctx, seed_buf, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
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

  PolyTensor *out = poly_tensor_create(
      ctx, poly_alu2(ctx, POLY_OP_ADD, poly_tensor_uop(a), poly_const_float(ctx, 1.0)),
      POLY_TENSOR_VALUE, POLY_DEVICE_CPU
  );
  ASSERT_NOT_NULL(out);
  PolyTensor *realized_out = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &out, 1, &realized_out), 0);
  ASSERT_INT_EQ(poly_jit_end_capture(jit), 0);
  ASSERT_INT_EQ(poly_jit_schedule_count(jit), 2);

  float side_after_capture[3] = {0};
  ASSERT_INT_EQ(poly_buffer_read(ctx, side_buf, side_after_capture, sizeof(side_after_capture)), 0);
  ASSERT_FLOAT_EQ(side_after_capture[0], 7.0f, 1e-5f);
  ASSERT_FLOAT_EQ(side_after_capture[1], 8.0f, 1e-5f);
  ASSERT_FLOAT_EQ(side_after_capture[2], 9.0f, 1e-5f);

  float reset_side[3] = {-9.0f, -9.0f, -9.0f};
  ASSERT_INT_EQ(poly_buffer_write(ctx, side_buf, reset_side, sizeof(reset_side)), 0);

  float b_data[3] = {100.0f, 200.0f, 300.0f};
  PolyUOp *b_buf = poly_buffer_f32(ctx, 3);
  poly_buffer_set(ctx, b_buf, b_data, sizeof(b_data), POLY_DEVICE_CPU);
  PolyTensor *b = poly_tensor_create(ctx, b_buf, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
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
  PolyUOp *a_buf = poly_buffer_f32(ctx, 3);
  poly_buffer_set(ctx, a_buf, a_data, sizeof(a_data), POLY_DEVICE_CPU);
  PolyTensor *a = poly_tensor_create(ctx, a_buf, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(a);

  PolyJit *jit = poly_jit_new(ctx);
  ASSERT_NOT_NULL(jit);
  ASSERT_INT_EQ(poly_jit_begin_capture(jit, &a, 1), 0);
  PolyTensor *out = poly_tensor_create(
      ctx, poly_alu2(ctx, POLY_OP_ADD, poly_tensor_uop(a), poly_const_float(ctx, 1.0)),
      POLY_TENSOR_VALUE, POLY_DEVICE_CPU
  );
  ASSERT_NOT_NULL(out);
  PolyTensor *realized = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &out, 1, &realized), 0);
  ASSERT_INT_EQ(poly_jit_end_capture(jit), 0);

  float b_data[4] = {10.0f, 20.0f, 30.0f, 40.0f};
  PolyUOp *b_buf = poly_buffer_f32(ctx, 4);
  poly_buffer_set(ctx, b_buf, b_data, sizeof(b_data), POLY_DEVICE_CPU);
  PolyTensor *b = poly_tensor_create(ctx, b_buf, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
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

TEST(realize, poly_jit_replays_with_bind_default_and_runtime_var_override) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *N = poly_define_var(ctx, "N", 1, 16);
  PolyUOp *bind_N = poly_bind_var(ctx, N, 4);
  ASSERT_NOT_NULL(N);
  ASSERT_NOT_NULL(bind_N);

  PolyUOp *unique_a = poly_uop0(ctx, POLY_OP_UNIQUE, POLY_VOID, poly_arg_int(9910000));
  PolyUOp *unique_out = poly_uop0(ctx, POLY_OP_UNIQUE, POLY_VOID, poly_arg_int(9910001));
  PolyUOp *a_src[2] = {unique_a, bind_N};
  PolyUOp *out_src[2] = {unique_out, bind_N};
  PolyUOp *a_buf = poly_uop(ctx, POLY_OP_BUFFER, POLY_FLOAT32, a_src, 2, poly_arg_int(16));
  PolyUOp *out_buf = poly_uop(ctx, POLY_OP_BUFFER, POLY_FLOAT32, out_src, 2, poly_arg_int(16));
  ASSERT_NOT_NULL(a_buf);
  ASSERT_NOT_NULL(out_buf);

  float a_data[16];
  float out_data[16];
  for (int i = 0; i < 16; i++) {
    a_data[i] = (float)(i + 1);
    out_data[i] = -999.0f;
  }
  poly_buffer_set(ctx, a_buf, a_data, sizeof(a_data), POLY_DEVICE_CPU);
  poly_buffer_set(ctx, out_buf, out_data, sizeof(out_data), POLY_DEVICE_CPU);

  PolyTensor *a = poly_tensor_create(ctx, a_buf, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(a);
  PolyJit *jit = poly_jit_new(ctx);
  ASSERT_NOT_NULL(jit);
  ASSERT_INT_EQ(poly_jit_begin_capture(jit, &a, 1), 0);

  PolyUOp *store = poly_store_val(
      ctx, out_buf, poly_alu2(ctx, POLY_OP_ADD, poly_tensor_uop(a), poly_const_float(ctx, 1.0f))
  );
  PolyUOp *after_src[2] = {out_buf, store};
  PolyUOp *assign = poly_uop(ctx, POLY_OP_AFTER, out_buf->dtype, after_src, 2, poly_arg_none());
  PolyUOp *realized[1] = {NULL};
  ASSERT_INT_EQ(poly_realize_uops(ctx, &assign, 1, realized), 0);
  ASSERT_PTR_EQ(realized[0], out_buf);
  ASSERT_INT_EQ(poly_jit_end_capture(jit), 0);

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
  PolyVarBinding bind_6 = {.var = N, .value = 6};
  ASSERT_INT_EQ(poly_jit_run_with_vars(jit, &a, 1, &bind_6, 1), 0);
  ASSERT_INT_EQ(poly_buffer_read(ctx, out_buf, out_data, sizeof(out_data)), 0);

  for (int i = 0; i < 6; i++)
    ASSERT_FLOAT_EQ(out_data[i], (float)(i + 2), 1e-5f);
  ASSERT_FLOAT_EQ(out_data[6], -999.0f, 1e-5f);

  poly_jit_free(jit);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, poly_jit_rejects_replay_dtype_mismatch) {
  PolyCtx *ctx = poly_ctx_new();

  float a_data[3] = {1.0f, 2.0f, 3.0f};
  PolyUOp *a_buf = poly_buffer_f32(ctx, 3);
  poly_buffer_set(ctx, a_buf, a_data, sizeof(a_data), POLY_DEVICE_CPU);
  PolyTensor *a = poly_tensor_create(ctx, a_buf, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(a);

  PolyJit *jit = poly_jit_new(ctx);
  ASSERT_NOT_NULL(jit);
  ASSERT_INT_EQ(poly_jit_begin_capture(jit, &a, 1), 0);
  PolyTensor *out = poly_tensor_create(
      ctx, poly_alu2(ctx, POLY_OP_ADD, poly_tensor_uop(a), poly_const_float(ctx, 1.0)),
      POLY_TENSOR_VALUE, POLY_DEVICE_CPU
  );
  ASSERT_NOT_NULL(out);
  PolyTensor *realized = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &out, 1, &realized), 0);
  ASSERT_INT_EQ(poly_jit_end_capture(jit), 0);

  int32_t b_data[3] = {10, 20, 30};
  PolyUOp *b_buf = poly_buffer(ctx, POLY_INT32, 3);
  poly_buffer_set(ctx, b_buf, b_data, sizeof(b_data), POLY_DEVICE_CPU);
  PolyTensor *b = poly_tensor_create(ctx, b_buf, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(b);

  ASSERT_INT_EQ(poly_jit_run(jit, &b, 1), -1);

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
  PolyUOp *x_buf = poly_buffer_f32(ctx, 8);
  PolyBuffer *x_storage = NULL;
  ASSERT_INT_EQ(poly_buffer_alloc_owned_host(ctx, x_buf, 8 * sizeof(float), true, &x_storage), 0);
  ASSERT_NOT_NULL(x_storage);
  for (int i = 0; i < 8; i++)
    ((float *)x_storage->ptr)[i] = (float)i;
  ASSERT_INT_EQ(poly_buffer_mark_host_written(ctx, x_buf), 0);

  PolyTensor *x = poly_tensor_create(ctx, x_buf, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  PolyTensor *out = poly_tensor_create(
      ctx, poly_alu2(ctx, POLY_OP_ADD, x_buf, poly_const_float(ctx, 1.0)), POLY_TENSOR_VALUE,
      POLY_DEVICE_CPU
  );
  ASSERT_NOT_NULL(x);
  ASSERT_NOT_NULL(out);

  PolyCtxStats stats = {0};
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &stats), 0);
  ASSERT_TRUE(stats.mem_used == 32);
  poly_ctx_reset_counters(ctx);
  PolyTensor *out_inputs[] = {out};
  PolyTensor *out_outputs[] = {NULL};
  ASSERT_INT_EQ(poly_realize_tensors_ex(ctx, out_inputs, 1, out_outputs, true), 0);
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &stats), 0);
  ASSERT_TRUE(stats.global_ops == 8);
  ASSERT_TRUE(stats.global_mem == 64);
  ASSERT_TRUE(stats.kernel_count == 1);
  ASSERT_TRUE(stats.mem_used == 64);

  poly_ctx_reset_counters(ctx);
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &stats), 0);
  ASSERT_TRUE(stats.global_ops == 0 && stats.global_mem == 0 && stats.kernel_count == 0);
  ASSERT_TRUE(stats.time_sum_s == 0.0);
  ASSERT_TRUE(stats.mem_used == 64);

  PolyTensor *suppressed = poly_tensor_create(
      ctx, poly_alu2(ctx, POLY_OP_ADD, x_buf, poly_const_float(ctx, 2.0)), POLY_TENSOR_VALUE,
      POLY_DEVICE_CPU
  );
  ASSERT_NOT_NULL(suppressed);
  PolyTensor *suppressed_inputs[] = {suppressed};
  PolyTensor *suppressed_outputs[] = {NULL};
  ASSERT_INT_EQ(poly_realize_tensors_ex(ctx, suppressed_inputs, 1, suppressed_outputs, false), 0);
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &stats), 0);
  ASSERT_TRUE(stats.global_ops == 0 && stats.global_mem == 0 && stats.kernel_count == 0);
  ASSERT_TRUE(stats.mem_used == 96);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, bound_view_materialization_executes_runtime_extent) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  float x_data[8] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
  PolyUOp *x_buf = poly_buffer_f32(ctx, 8);
  ASSERT_NOT_NULL(x_buf);
  poly_buffer_set(ctx, x_buf, x_data, sizeof(x_data), POLY_DEVICE_CPU);

  PolyUOp *n = poly_define_var(ctx, "n", 1, 8);
  PolyUOp *bound_n = poly_bind_var(ctx, n, 4);
  PolyUOp *starts[1] = {poly_const_int(ctx, 0)};
  PolyUOp *sizes[1] = {bound_n};
  PolyUOp *view = poly_shrink_uop(ctx, x_buf, starts, sizes, 1);
  PolyUOp *add = poly_alu2(ctx, POLY_OP_ADD, view, poly_const_float(ctx, 1.0));
  ASSERT_NOT_NULL(n);
  ASSERT_NOT_NULL(bound_n);
  ASSERT_NOT_NULL(view);
  ASSERT_NOT_NULL(add);
  ASSERT_PTR_EQ(poly_uop_shape_dim(ctx, add, 0), bound_n);

  PolyTensor *out = poly_tensor_create(ctx, add, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(out);
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
  ASSERT_INT_EQ(physical->src[1]->op, POLY_OP_STACK);
  ASSERT_INT_EQ(physical->src[1]->n_src, 1);
  ASSERT_INT_EQ(physical->src[1]->src[0]->op, POLY_OP_CONST);
  ASSERT_INT_EQ(physical->src[1]->src[0]->arg.i, 0);
  ASSERT_INT_EQ(physical->src[2]->op, POLY_OP_STACK);
  ASSERT_INT_EQ(physical->src[2]->n_src, 1);
  ASSERT_PTR_EQ(physical->src[2]->src[0], bound_n);
  ASSERT_TRUE(poly_uop_get_buffer_identity(physical) == NULL);
  const PolyUOp *physical_storage =
      poly_uop_get_buffer_identity(physical->src[0]);
  ASSERT_NOT_NULL(physical_storage);

  float values[4] = {0};
  ASSERT_INT_EQ(
      poly_buffer_read(
          ctx, (PolyUOp *)physical_storage, values, sizeof(values)),
      0);
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
  PolyUOp *base = poly_buffer_f32(ctx, 8);
  ASSERT_NOT_NULL(base);
  poly_buffer_set(ctx, base, data, sizeof(data), POLY_DEVICE_CPU);

  int64_t shape[2] = {2, 4};
  int64_t order[2] = {1, 0};
  PolyUOp *view = poly_permute(ctx, poly_reshape(ctx, base, shape, 2), order, 2);
  ASSERT_NOT_NULL(view);
  PolyTensor *tensor = poly_tensor_create(ctx, view, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
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
  PolyTensor *disk_tensor = poly_tensor_create(ctx, slice, POLY_TENSOR_VALUE, POLY_DEVICE_DISK);
  ASSERT_NOT_NULL(disk_tensor);

  poly_ctx_reset_counters(ctx);
  PolyTensor *disk_outputs[1] = {NULL};
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &disk_tensor, 1, disk_outputs), 0);
  ASSERT_NOT_NULL(disk_outputs[0]);
  PolyUOp *disk_realized = poly_tensor_uop(disk_outputs[0]);
  ASSERT_NOT_NULL(disk_realized);
  ASSERT_EQ(disk_realized->op, POLY_OP_BUFFER_VIEW);
  PolyBuffer *view_storage = poly_buffer_get(ctx, disk_realized);
  ASSERT_NOT_NULL(view_storage);
  ASSERT_INT_EQ(view_storage->device, POLY_DEVICE_DISK);
  ASSERT_TRUE(view_storage->nbytes == 6);
  ASSERT_PTR_EQ(view_storage->ptr, (uint8_t *)disk_storage->ptr + 3);
  uint8_t view_data[6] = {0};
  ASSERT_INT_EQ(poly_buffer_read(ctx, disk_realized, view_data, sizeof(view_data)), 0);
  for (int i = 0; i < 6; i++)
    ASSERT_INT_EQ(view_data[i], i + 3);
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &stats), 0);
  ASSERT_TRUE(stats.global_ops == 0 && stats.global_mem == 0 && stats.kernel_count == 0);
  ASSERT_TRUE(stats.mem_used == 0);

  PolyTensor *cpu_tensor = poly_tensor_to_device(ctx, disk_outputs[0], POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(cpu_tensor);
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
  ASSERT_TRUE(stats.kernel_count == 2);
  ASSERT_TRUE(stats.mem_used == 6);

  poly_ctx_destroy(ctx);
  ASSERT_INT_EQ(unlink(path), 0);
  PASS();
}

TEST(realize, contiguous_realized_view_is_only_physical_at_tensor_boundary) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  float data[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  PolyUOp *base = poly_buffer_f32(ctx, 8);
  ASSERT_NOT_NULL(base);
  PolyBuffer handle = poly_buffer_make_host_view(data, sizeof(data));
  poly_buffer_attach(ctx, base, &handle);

  int64_t nested_bounds[1][2] = {{1, 6}};
  PolyUOp *nested_view = poly_shrink(ctx, base, nested_bounds, 1);
  ASSERT_NOT_NULL(nested_view);
  PolyUOp *value = poly_alu2(ctx, POLY_OP_ADD, nested_view, poly_const_float(ctx, 1.0));
  ASSERT_NOT_NULL(value);
  PolyTensor *value_tensor = poly_tensor_create(ctx, value, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(value_tensor);
  PolyUOp *value_physical = poly_tensor_physicalize(ctx, value_tensor);
  ASSERT_NOT_NULL(value_physical);
  ASSERT_INT_EQ(count_root_ops(ctx, value_physical, POLY_OP_BUFFER_VIEW), 0);
  ASSERT_INT_EQ(count_root_ops(ctx, value_physical, POLY_OP_SHRINK), 1);

  int64_t root_bounds[1][2] = {{2, 5}};
  PolyUOp *root_view = poly_shrink(ctx, base, root_bounds, 1);
  ASSERT_NOT_NULL(root_view);
  PolyTensor *root_tensor = poly_tensor_create(ctx, root_view, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(root_tensor);
  PolyUOp *root_physical = poly_tensor_physicalize(ctx, root_tensor);
  ASSERT_NOT_NULL(root_physical);
  const PolyUOp *root_identity = poly_uop_get_buffer_identity(root_physical);
  ASSERT_NOT_NULL(root_identity);
  ASSERT_EQ(root_identity->op, POLY_OP_BUFFER_VIEW);
  PolyBuffer *alias = poly_buffer_get(ctx, (PolyUOp *)root_identity);
  ASSERT_NOT_NULL(alias);
  ASSERT_PTR_EQ(alias->ptr, data + 2);
  ASSERT_INT_EQ((int)alias->nbytes, 3 * (int)sizeof(float));

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
      poly_tensor_create(ctx, left, POLY_TENSOR_VALUE, POLY_DEVICE_DISK),
      poly_tensor_create(ctx, right, POLY_TENSOR_VALUE, POLY_DEVICE_DISK),
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

  const PolyUOp *disk_ids[2] = {
      poly_uop_get_buffer_identity(disk_roots[0]),
      poly_uop_get_buffer_identity(disk_roots[1]),
  };
  ASSERT_NOT_NULL(disk_ids[0]);
  ASSERT_NOT_NULL(disk_ids[1]);
  ASSERT_EQ(disk_ids[0]->op, POLY_OP_BUFFER_VIEW);
  ASSERT_EQ(disk_ids[1]->op, POLY_OP_BUFFER_VIEW);
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
  ASSERT_PTR_EQ(disk_views[0]->ptr, (uint8_t *)disk_storage->ptr + 4);
  ASSERT_PTR_EQ(disk_views[1]->ptr, (uint8_t *)disk_storage->ptr + 17);

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
  ASSERT_TRUE(stats.kernel_count == 4);
  ASSERT_TRUE(stats.mem_used == 14);

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

  PolyTensor *left_disk = poly_tensor_create(ctx, left, POLY_TENSOR_VALUE, POLY_DEVICE_DISK);
  PolyTensor *right_disk = poly_tensor_create(ctx, right, POLY_TENSOR_VALUE, POLY_DEVICE_DISK);
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

  PolyUOp *physical = poly_tensor_physicalize(ctx, result);
  PolyUOp *scheduled_out = NULL;
  PolySchedule *schedule = poly_schedule_with_vars(ctx, &physical, 1, &scheduled_out);
  ASSERT_NOT_NULL(schedule);
  ASSERT_NOT_NULL(scheduled_out);
  ASSERT_INT_EQ(schedule->template->n_calls, 5);
  int view_calls = 0, copy_calls = 0;
  for (int i = 0; i < 4; i++) {
    PolyUOp *body = poly_schedule_call_body(schedule, i);
    ASSERT_NOT_NULL(body);
    if (poly_schedule_call_is_copy(schedule, i))
      copy_calls++;
    else if (body->op == POLY_OP_BUFFER_VIEW)
      view_calls++;
  }
  ASSERT_INT_EQ(view_calls, 2);
  ASSERT_INT_EQ(copy_calls, 2);
  PolyUOp *compute = poly_schedule_call_body(schedule, 4);
  ASSERT_NOT_NULL(compute);
  ASSERT_INT_EQ(count_root_ops(ctx, compute, POLY_OP_RESHAPE), 0);
  ASSERT_INT_EQ(count_root_ops(ctx, compute, POLY_OP_SHRINK), 0);
  ASSERT_INT_EQ(count_root_ops(ctx, compute, POLY_OP_PAD), 0);
  ASSERT_INT_EQ(count_root_ops(ctx, compute, POLY_OP_BUFFER), 0);
  ASSERT_INT_EQ(count_root_ops(ctx, compute, POLY_OP_BUFFER_VIEW), 0);
  ASSERT_INT_EQ(count_root_ops(ctx, compute, POLY_OP_UNIQUE), 0);
  ASSERT_INT_EQ(count_root_ops(ctx, compute, POLY_OP_PARAM), 3);
  ASSERT_TRUE(count_root_ops(ctx, compute, POLY_OP_INDEX) > 0);
  ASSERT_TRUE(count_root_ops(ctx, compute, POLY_OP_STORE) > 0);
  poly_schedule_free(schedule);

  poly_ctx_reset_counters(ctx);
  PolyTensor *realized = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &result, 1, &realized), 0);
  ASSERT_NOT_NULL(realized);
  PolyCtxStats stats = {0};
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &stats), 0);
  ASSERT_TRUE(stats.kernel_count == 5);

  /* tinygrad Tensor._buffer and Polygrad Tensor._buffer both materialize a
   * non-contiguous realized view before flat byte readback. Keep that readback
   * call outside the five-call execution-topology assertion above. */
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

  PolyTensor *left_disk = poly_tensor_create(ctx, left, POLY_TENSOR_VALUE, POLY_DEVICE_DISK);
  PolyTensor *right_disk = poly_tensor_create(ctx, right, POLY_TENSOR_VALUE, POLY_DEVICE_DISK);
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

  PolyUOp *physical = poly_tensor_physicalize(ctx, result);
  PolyUOp *scheduled_out = NULL;
  PolySchedule *schedule = poly_schedule_with_vars(ctx, &physical, 1, &scheduled_out);
  ASSERT_NOT_NULL(schedule);
  ASSERT_NOT_NULL(scheduled_out);
  ASSERT_INT_EQ(schedule->template->n_calls, 6);
  /* Pinned tinygrad callify exposes the CONTIGUOUS buffer as a CALL argument;
   * create_linear_with_vars therefore holds it out of the memory planner. */
  ASSERT_INT_EQ(poly_schedule_external_slot_count(schedule), 7);

  int n_intermediates = 0;
  for (int i = 0; i < schedule->template->n_buf_slots; i++) {
    PolyScheduleBufSlot *slot = &schedule->template->buf_slots[i];
    if (slot->is_intermediate) n_intermediates++;
  }
  ASSERT_INT_EQ(n_intermediates, 0);
  ASSERT_TRUE(poly_schedule_runtime_intermediate_bytes(schedule) == 0);

  ASSERT_FALSE(poly_schedule_call_is_copy(schedule, 4));
  ASSERT_FALSE(poly_schedule_call_is_copy(schedule, 5));
  int reduction_slot = poly_schedule_call_buffer_slot(schedule, 4, 0);
  ASSERT_TRUE(reduction_slot >= 0);
  ASSERT_INT_EQ(poly_schedule_call_buffer_slot(schedule, 5, 1), reduction_slot);
  PolyScheduleBufSlot *reduction = &schedule->template->buf_slots[reduction_slot];
  ASSERT_FALSE(reduction->is_intermediate);
  ASSERT_TRUE(reduction->external_buf_idx >= 0);
  ASSERT_TRUE(poly_buffer_get(ctx, reduction->buf_uop) == NULL);
  ASSERT_FALSE(poly_buffer_is_allocated(ctx, reduction->buf_uop));

  int reduction_writes = 0;
  int reduction_reads = 0;
  for (int k = 0; k < schedule->template->n_calls; k++) {
    PolyCallIO *io = &schedule->run->call_io[k];
    ASSERT_NOT_NULL(io->access);
    for (int i = 0; i < io->n_args; i++) {
      if (io->arg_to_slot[i] != reduction_slot) continue;
      reduction_writes += io->access->outs[i] ? 1 : 0;
      reduction_reads += io->access->ins[i] ? 1 : 0;
    }
  }
  ASSERT_INT_EQ(reduction_writes, 1);
  ASSERT_INT_EQ(reduction_reads, 1);

  ASSERT_INT_EQ(poly_run_schedule(ctx, schedule, NULL, 0), 0);
  PolyBuffer *reduction_storage = poly_buffer_get(ctx, reduction->buf_uop);
  ASSERT_NOT_NULL(reduction_storage);
  ASSERT_TRUE(reduction_storage->valid);
  ASSERT_TRUE(poly_buffer_is_allocated(ctx, reduction->buf_uop));
  const PolyUOp *identity = poly_uop_get_buffer_identity(scheduled_out);
  ASSERT_NOT_NULL(identity);
  int32_t values[2] = {0};
  ASSERT_INT_EQ(poly_buffer_read(ctx, (PolyUOp *)identity, values, sizeof(values)), 0);
  ASSERT_INT_EQ(values[0], 35);
  ASSERT_INT_EQ(values[1], 66);

  poly_schedule_free(schedule);
  poly_ctx_destroy(ctx);
  ASSERT_INT_EQ(unlink(path), 0);
  PASS();
}
