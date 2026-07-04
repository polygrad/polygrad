/*
 * test_schedule_engine.c — Tests for the engine/schedule single-kernel path
 */

#include "test_harness.h"
#include "../src/engine/schedule.h"
#include "../src/engine/realize.h"
#include "../src/codegen.h"
#include "../src/schedule/rangeify.h"
#include "../src/frontend.h"
#include "../src/device.h"

/* Helper: build tensor-level graph, schedule, verify structure */

/* These scheduler tests linearize executable single-kernel roots. The public
 * poly_get_kernel_graph() boundary now stops earlier at the pre-codegen kernel
 * graph, so the tests map their local kernel lookups through the engine
 * schedule and pull back the one scheduled root explicitly. */
static PolyUOp *single_scheduled_root(PolyCtx *ctx, PolyUOp *sink) {
  PolySchedule *schedule = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  if (!schedule) return NULL;
  if (schedule->template->n_calls < 1 || !poly_schedule_call_body(schedule, schedule->template->n_calls - 1)) {
    poly_schedule_free(schedule);
    return NULL;
  }
  PolyUOp *root = poly_schedule_call_body(schedule, schedule->template->n_calls - 1);
  poly_schedule_free(schedule);
  return root;
}

#define poly_get_kernel_graph(ctx, sink) single_scheduled_root((ctx), (sink))

static PolyBuffer *test_realized_buffer(PolyCtx *ctx, PolyUOp *realized) {
  const PolyUOp *buf_uop = poly_uop_get_buffer_identity(realized);
  return buf_uop ? poly_buffer_get(ctx, (PolyUOp *)buf_uop) : NULL;
}

/* Count ops of a given type in a linearized graph */
static int count_ops(PolyUOp **lin, int n, PolyOps op) {
  int count = 0;
  for (int i = 0; i < n; i++)
    if (lin[i]->op == op) count++;
  return count;
}

static int count_root_ops(PolyCtx *ctx, PolyUOp *root, PolyOps op) {
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, root, &n_topo);
  int count = 0;
  for (int i = 0; i < n_topo; i++) {
    if (topo[i] && topo[i]->op == op) count++;
  }
  return count;
}

static int count_param_index_ptrs(PolyCtx *ctx, PolyUOp *root, bool is_ptr) {
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, root, &n_topo);
  int count = 0;
  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    if (!u || u->op != POLY_OP_INDEX || u->n_src < 1 || !u->src[0] ||
        u->src[0]->op != POLY_OP_PARAM)
      continue;
    if ((bool)u->dtype.is_ptr == is_ptr) count++;
  }
  return count;
}

/* IR structure tests */

TEST(sched, vecadd_ir) {
  /* c = a + b (1D, 10 elements): verify kernel IR structure */
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *b = poly_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *c = poly_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, a, b, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, c, add, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyUOp *kernel = poly_get_kernel_graph(ctx, sink);
  ASSERT_NOT_NULL(kernel);
  ASSERT_EQ(kernel->op, POLY_OP_SINK);

  /* Linearize and check structure */
  int n;
  PolyUOp **lin = poly_linearize(ctx, kernel, &n);
  ASSERT_TRUE(n > 0);

  /* Should have: 3 PARAMs, 1 RANGE, 1 END, 1 SINK, 2 LOADs, 1 ADD, 1 STORE */
  ASSERT_INT_EQ(count_ops(lin, n, POLY_OP_PARAM), 3);
  ASSERT_INT_EQ(count_ops(lin, n, POLY_OP_RANGE), 1);
  ASSERT_INT_EQ(count_ops(lin, n, POLY_OP_LOAD), 2);
  ASSERT_INT_EQ(count_ops(lin, n, POLY_OP_ADD), 1);
  ASSERT_INT_EQ(count_ops(lin, n, POLY_OP_STORE), 1);
  ASSERT_INT_EQ(count_ops(lin, n, POLY_OP_END), 1);

  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

/* End-to-end tests */

TEST_COMMON(sched, vecadd_e2e) {
  /* c = a + b: build tensor graph, schedule, compile, run, verify */
  int N = 16;
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *b = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *c = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, a, b, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, c, add, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyUOp *kernel = poly_get_kernel_graph(ctx, sink);
  ASSERT_NOT_NULL(kernel);

  int n_lin;
  PolyUOp **lin = poly_linearize(ctx, kernel, &n_lin);
  char *src = poly_render_c(lin, n_lin, "vecadd");
  ASSERT_NOT_NULL(src);

  PolyProgram *prog = poly_compile_c(src, "vecadd");
  ASSERT_NOT_NULL(prog);

  /* Prepare data */
  float a_data[16], b_data[16], c_data[16];
  for (int i = 0; i < N; i++) {
    a_data[i] = (float)(i + 1);
    b_data[i] = (float)(i + 1) * 0.5f;
    c_data[i] = 0.0f;
  }

  /* The scheduler assigns: param 0 = output (c), param 1 = a, param 2 = b */
  void *args[3] = {c_data, a_data, b_data};
  poly_program_call(prog, args, 3);

  for (int i = 0; i < N; i++)
    ASSERT_FLOAT_EQ(c_data[i], a_data[i] + b_data[i], 1e-6);

  poly_program_destroy(prog);
  free(src);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sched, direct_sink_copy_uses_copy_call) {
  int N = 8;
  float src_d[8], dst_d[8];
  for (int i = 0; i < N; i++) {
    src_d[i] = (float)(i + 1) * 1.25f;
    dst_d[i] = 0.0f;
  }

  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *src = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *dst = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, dst, src));

  PolySchedule *sched = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_NOT_NULL(sched);
  ASSERT_INT_EQ(sched->template->n_calls, 1);
  ASSERT_TRUE(poly_schedule_call_is_copy(sched, 0));
  ASSERT_TRUE(poly_schedule_call_body(sched, 0)->op != POLY_OP_SINK);
  ASSERT_TRUE(poly_schedule_call_body(sched, 0)->op != POLY_OP_STORE);

  poly_buffer_set(ctx, dst, dst_d, sizeof(dst_d), POLY_DEVICE_CPU);
  poly_buffer_set(ctx, src, src_d, sizeof(src_d), POLY_DEVICE_CPU);
  ASSERT_INT_EQ(poly_run_schedule(ctx, sched, NULL, 0), 0);
  for (int i = 0; i < N; i++)
    ASSERT_FLOAT_EQ(dst_d[i], src_d[i], 1e-6);

  poly_schedule_free(sched);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_COMMON(sched, realize_uops_reads_ctx_buffers) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, 4);
  PolyUOp *b = poly_buffer(ctx, POLY_FLOAT32, 4);
  PolyUOp *add = poly_alu2(ctx, POLY_OP_ADD, a, b);

  float a_d[] = {1.0f, 2.0f, 3.0f, 4.0f};
  float b_d[] = {10.0f, 20.0f, 30.0f, 40.0f};
  poly_buffer_set(ctx, a, a_d, sizeof(a_d), POLY_DEVICE_CPU);
  poly_buffer_set(ctx, b, b_d, sizeof(b_d), POLY_DEVICE_CPU);

  PolyUOp *out_uop = NULL;
  ASSERT_INT_EQ(poly_realize_uops(ctx, &add, 1, &out_uop), 0);
  ASSERT_NOT_NULL(out_uop);
  PolyBuffer *out_buf = test_realized_buffer(ctx, out_uop);
  ASSERT_NOT_NULL(out_buf);
  PolyUOp *out_storage = (PolyUOp *)poly_uop_get_buffer_identity(out_uop);
  ASSERT_NOT_NULL(out_storage);
  float out[4] = {0};
  ASSERT_INT_EQ(poly_buffer_read(ctx, out_storage, out, sizeof(out)), 0);
  for (int i = 0; i < 4; i++)
    ASSERT_FLOAT_EQ(out[i], a_d[i] + b_d[i], 1e-6f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST_COMMON(sched, ctx_buffer_rebinding_overrides_previous_storage) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, 4);
  PolyUOp *b = poly_buffer(ctx, POLY_FLOAT32, 4);
  PolyUOp *add = poly_alu2(ctx, POLY_OP_ADD, a, b);

  float a_ctx[] = {100.0f, 100.0f, 100.0f, 100.0f};
  float a_call[] = {1.0f, 2.0f, 3.0f, 4.0f};
  float b_d[] = {10.0f, 20.0f, 30.0f, 40.0f};
  poly_buffer_set(ctx, a, a_ctx, sizeof(a_ctx), POLY_DEVICE_CPU);
  poly_buffer_set(ctx, b, b_d, sizeof(b_d), POLY_DEVICE_CPU);

  poly_buffer_set(ctx, a, a_call, sizeof(a_call), POLY_DEVICE_CPU);

  PolyUOp *out_uop = NULL;
  ASSERT_INT_EQ(poly_realize_uops(ctx, &add, 1, &out_uop), 0);
  ASSERT_NOT_NULL(out_uop);
  PolyBuffer *out_buf = test_realized_buffer(ctx, out_uop);
  ASSERT_NOT_NULL(out_buf);
  PolyUOp *out_storage = (PolyUOp *)poly_uop_get_buffer_identity(out_uop);
  ASSERT_NOT_NULL(out_storage);
  float out[4] = {0};
  ASSERT_INT_EQ(poly_buffer_read(ctx, out_storage, out, sizeof(out)), 0);
  for (int i = 0; i < 4; i++)
    ASSERT_FLOAT_EQ(out[i], a_call[i] + b_d[i], 1e-6f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sched, placement_copy_item_uses_copy_root_and_two_slots) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, 3);
  float *data = malloc(3 * sizeof(float));
  ASSERT_NOT_NULL(data);
  data[0] = 1.0f;
  data[1] = 2.0f;
  data[2] = 3.0f;
  poly_buffer_set(ctx, a, data, 3 * sizeof(float), POLY_DEVICE_HOST);

  PolyUOp *x = poly_alu2(ctx, POLY_OP_ADD, a, poly_const_float(ctx, 1.0f));
  PolyTensor *tensor = poly_tensor_create(ctx, x, POLY_TENSOR_PLACE, POLY_DEVICE_CPU);
  PolyUOp *physical = poly_tensor_physicalize(ctx, tensor);
  ASSERT_NOT_NULL(physical);

  PolyUOp *out = poly_buffer_on_device(ctx, POLY_FLOAT32, 3, POLY_DEVICE_CPU);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, physical));
  PolySchedule *sched = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_NOT_NULL(sched);
  ASSERT_INT_EQ(sched->template->n_calls, 2);
  ASSERT_TRUE(poly_schedule_call_is_copy(sched, 0));
  ASSERT_NOT_NULL(poly_schedule_call_body(sched, 0));
  ASSERT_INT_EQ(poly_schedule_call_body(sched, 0)->op, POLY_OP_COPY);
  ASSERT_INT_EQ(poly_schedule_call_n_buffer_args(sched, 0), 2);

  poly_schedule_free(sched);
  poly_ctx_destroy(ctx);
  free(data);
  PASS();
}

TEST(sched, placement_computed_copy_source_materializes_before_copy) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, 3);
  float data[3] = {1.0f, 2.0f, 3.0f};
  poly_buffer_set(ctx, a, data, sizeof(data), POLY_DEVICE_CPU);

  PolyUOp *mul = poly_alu2(ctx, POLY_OP_MUL, a, poly_const_float(ctx, 2.0f));
  PolyTensor *mt = poly_tensor_create(ctx, mul, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(mt);
  PolyTensor *cuda_t = poly_tensor_to_device(ctx, mt, POLY_DEVICE_CUDA);
  ASSERT_NOT_NULL(cuda_t);

  PolyUOp *physical = poly_tensor_physicalize(ctx, cuda_t);
  ASSERT_NOT_NULL(physical);
  ASSERT_INT_EQ(physical->op, POLY_OP_COPY);

  PolyUOp *out = poly_buffer_on_device(ctx, POLY_FLOAT32, 3, POLY_DEVICE_CUDA);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, physical));
  PolySchedule *sched = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_NOT_NULL(sched);

  int n_copy = 0;
  int n_compute = 0;
  for (int i = 0; i < sched->template->n_calls; i++) {
    if (poly_schedule_call_is_copy(sched, i)) {
      n_copy++;
      ASSERT_INT_EQ(poly_schedule_call_body(sched, i)->op, POLY_OP_COPY);
      ASSERT_INT_EQ(poly_schedule_call_n_buffer_args(sched, i), 2);
    } else {
      n_compute++;
      ASSERT_INT_EQ(count_root_ops(ctx, poly_schedule_call_body(sched, i), POLY_OP_COPY), 0);
      ASSERT_TRUE(poly_schedule_call_n_buffer_args(sched, i) >= 1);
      int dst_slot = poly_schedule_call_buffer_slot(sched, i, 0);
      ASSERT_TRUE(dst_slot >= 0 && dst_slot < sched->template->n_buf_slots);
      ASSERT_INT_EQ(sched->template->buf_slots[dst_slot].device, POLY_DEVICE_CPU);
    }
  }
  ASSERT_INT_EQ(n_compute, 1);
  ASSERT_INT_EQ(n_copy, 1);

  poly_schedule_free(sched);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sched, placement_scalar_copy_operand_splits_before_compute) {
  PolyCtx *ctx = poly_ctx_new();
  float x_data[1] = {5.0f};
  float y_data[1] = {3.0f};

  PolyUOp *x = poly_buffer(ctx, POLY_FLOAT32, 1);
  PolyUOp *y = poly_buffer(ctx, POLY_FLOAT32, 1);
  poly_buffer_set(ctx, x, x_data, sizeof(x_data), POLY_DEVICE_CPU);
  poly_buffer_set(ctx, y, y_data, sizeof(y_data), POLY_DEVICE_HOST);

  PolyTensor *yt = poly_tensor_create(ctx, y, POLY_TENSOR_PLACE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(yt);
  PolyUOp *y_cpu = poly_tensor_physicalize(ctx, yt);
  ASSERT_NOT_NULL(y_cpu);

  PolyUOp *sub = poly_alu2(ctx, POLY_OP_SUB, x, y_cpu);
  PolyUOp *out = poly_buffer_on_device(ctx, POLY_FLOAT32, 1, POLY_DEVICE_CPU);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, sub));
  PolySchedule *sched = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_NOT_NULL(sched);

  ASSERT_INT_EQ(sched->template->n_calls, 2);
  ASSERT_TRUE(poly_schedule_call_is_copy(sched, 0));
  ASSERT_INT_EQ(poly_schedule_call_body(sched, 0)->op, POLY_OP_COPY);
  ASSERT_INT_EQ(poly_schedule_call_n_buffer_args(sched, 0), 2);
  ASSERT_FALSE(poly_schedule_call_is_copy(sched, 1));

  poly_schedule_free(sched);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sched, max_reduce_tail_keeps_only_store_index_ptr_typed) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_buffer(ctx, POLY_FLOAT32, 6);
  PolyUOp *xr = poly_reshape(ctx, x, (int64_t[]){2, 3}, 2);
  PolyUOp *m = poly_reduce_axis(ctx, POLY_OP_MAX, xr, (int64_t[]){1}, 1);
  PolyUOp *loss = poly_reduce_axis(ctx, POLY_OP_ADD, m, (int64_t[]){0}, 1);
  PolyUOp *gx = poly_grad(ctx, loss, xr);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, 6);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, gx));

  PolySchedule *sched = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_NOT_NULL(sched);
  ASSERT_TRUE(sched->template->n_calls >= 1);

  PolyUOp *tail = poly_schedule_call_body(sched, sched->template->n_calls - 1);
  ASSERT_NOT_NULL(tail);
  ASSERT_INT_EQ(count_param_index_ptrs(ctx, tail, true), 1);
  ASSERT_INT_EQ(count_param_index_ptrs(ctx, tail, false), 3);

  poly_schedule_free(sched);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_COMMON(sched, chain_e2e) {
  /* d = (a + b) * c: verify single kernel with correct results */
  int N = 8;
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *b = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *c = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *d = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, a, b, poly_arg_none());
  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, add, c, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, d, mul, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyUOp *kernel = poly_get_kernel_graph(ctx, sink);
  ASSERT_NOT_NULL(kernel);

  int n_lin;
  PolyUOp **lin = poly_linearize(ctx, kernel, &n_lin);
  char *src = poly_render_c(lin, n_lin, "chain");
  PolyProgram *prog = poly_compile_c(src, "chain");
  ASSERT_NOT_NULL(prog);

  float a_d[8], b_d[8], c_d[8], d_d[8];
  for (int i = 0; i < N; i++) {
    a_d[i] = (float)(i + 1);
    b_d[i] = 2.0f;
    c_d[i] = 0.5f;
    d_d[i] = 0.0f;
  }

  /* param 0 = output (d), then a, b, c in toposort order */
  void *args[4] = {d_d, a_d, b_d, c_d};
  poly_program_call(prog, args, 4);

  for (int i = 0; i < N; i++)
    ASSERT_FLOAT_EQ(d_d[i], (a_d[i] + b_d[i]) * c_d[i], 1e-6);

  poly_program_destroy(prog);
  free(src);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_COMMON(sched, broadcast_e2e) {
  /* c = a + scalar(2.0): scalar broadcast */
  int N = 8;
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *c = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *two = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(2.0));
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, a, two, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, c, add, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyUOp *kernel = poly_get_kernel_graph(ctx, sink);
  ASSERT_NOT_NULL(kernel);

  int n_lin;
  PolyUOp **lin = poly_linearize(ctx, kernel, &n_lin);
  char *src = poly_render_c(lin, n_lin, "bcast");
  PolyProgram *prog = poly_compile_c(src, "bcast");
  ASSERT_NOT_NULL(prog);

  float a_d[8], c_d[8];
  for (int i = 0; i < N; i++) {
    a_d[i] = (float)(i + 1);
    c_d[i] = 0.0f;
  }

  void *args[2] = {c_d, a_d};
  poly_program_call(prog, args, 2);

  for (int i = 0; i < N; i++)
    ASSERT_FLOAT_EQ(c_d[i], a_d[i] + 2.0f, 1e-6);

  poly_program_destroy(prog);
  free(src);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_COMMON(sched, unary_e2e) {
  /* b = neg(a) */
  int N = 8;
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *b = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *neg = poly_uop1(ctx, POLY_OP_NEG, POLY_FLOAT32, a, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, b, neg, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyUOp *kernel = poly_get_kernel_graph(ctx, sink);
  ASSERT_NOT_NULL(kernel);

  int n_lin;
  PolyUOp **lin = poly_linearize(ctx, kernel, &n_lin);
  char *src = poly_render_c(lin, n_lin, "neg_k");
  PolyProgram *prog = poly_compile_c(src, "neg_k");
  ASSERT_NOT_NULL(prog);

  float a_d[8], b_d[8];
  for (int i = 0; i < N; i++) {
    a_d[i] = (float)(i + 1);
    b_d[i] = 0.0f;
  }

  void *args[2] = {b_d, a_d};
  poly_program_call(prog, args, 2);

  for (int i = 0; i < N; i++)
    ASSERT_FLOAT_EQ(b_d[i], -a_d[i], 1e-6);

  poly_program_destroy(prog);
  free(src);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_COMMON(sched, 2d_e2e) {
  /* c = a + b where a, b are 4x8 (32 elements) */
  int N = 32;
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *b = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *c = poly_buffer(ctx, POLY_FLOAT32, N);

  /* Reshape to 2D */
  int64_t dims[] = {4, 8};
  PolyUOp *a2d = poly_reshape(ctx, a, dims, 2);
  PolyUOp *b2d = poly_reshape(ctx, b, dims, 2);

  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, a2d, b2d, poly_arg_none());

  /* Flatten back for storage */
  int64_t flat[] = {32};
  PolyUOp *flat_result = poly_reshape(ctx, add, flat, 1);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, c, flat_result, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyUOp *kernel = poly_get_kernel_graph(ctx, sink);
  ASSERT_NOT_NULL(kernel);

  int n_lin;
  PolyUOp **lin = poly_linearize(ctx, kernel, &n_lin);
  char *src = poly_render_c(lin, n_lin, "add2d");
  PolyProgram *prog = poly_compile_c(src, "add2d");
  ASSERT_NOT_NULL(prog);

  float a_d[32], b_d[32], c_d[32];
  for (int i = 0; i < N; i++) {
    a_d[i] = (float)(i + 1);
    b_d[i] = (float)(i + 1) * 0.1f;
    c_d[i] = 0.0f;
  }

  void *args[3] = {c_d, a_d, b_d};
  poly_program_call(prog, args, 3);

  for (int i = 0; i < N; i++)
    ASSERT_FLOAT_EQ(c_d[i], a_d[i] + b_d[i], 1e-5);

  poly_program_destroy(prog);
  free(src);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_COMMON(sched, expand_e2e) {
  /* Broadcast: a is (5, 4), b is (1, 4) expanded to (5, 4)
   * c[i,j] = a[i,j] + b[0,j] */
  int N = 20;
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a_buf = poly_buffer(ctx, POLY_FLOAT32, N); /* 20 elements */
  PolyUOp *b_buf = poly_buffer(ctx, POLY_FLOAT32, 4); /* 4 elements */
  PolyUOp *c_buf = poly_buffer(ctx, POLY_FLOAT32, N); /* 20 elements */

  /* Reshape a to (5,4) */
  int64_t a_dims[] = {5, 4};
  PolyUOp *a = poly_reshape(ctx, a_buf, a_dims, 2);

  /* Reshape b to (1,4), then expand to (5,4) */
  int64_t b_dims[] = {1, 4};
  PolyUOp *b_r = poly_reshape(ctx, b_buf, b_dims, 2);
  int64_t e_dims[] = {5, 4};
  PolyUOp *b = poly_expand(ctx, b_r, e_dims, 2);

  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, a, b, poly_arg_none());

  /* Flatten result for storage */
  int64_t flat[] = {20};
  PolyUOp *flat_result = poly_reshape(ctx, add, flat, 1);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, c_buf, flat_result, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyUOp *kernel = poly_get_kernel_graph(ctx, sink);
  ASSERT_NOT_NULL(kernel);

  int n_lin;
  PolyUOp **lin = poly_linearize(ctx, kernel, &n_lin);
  char *src = poly_render_c(lin, n_lin, "bcast2d");
  PolyProgram *prog = poly_compile_c(src, "bcast2d");
  ASSERT_NOT_NULL(prog);

  float a_d[20], b_d[4], c_d[20];
  for (int i = 0; i < N; i++) {
    a_d[i] = (float)(i + 1);
    c_d[i] = 0.0f;
  }
  for (int i = 0; i < 4; i++)
    b_d[i] = (float)(i + 1) * 10.0f;

  void *args[3] = {c_d, a_d, b_d};
  poly_program_call(prog, args, 3);

  /* Verify: c[i*4+j] = a[i*4+j] + b[j] */
  for (int i = 0; i < 5; i++)
    for (int j = 0; j < 4; j++)
      ASSERT_FLOAT_EQ(c_d[i * 4 + j], a_d[i * 4 + j] + b_d[j], 1e-5);

  poly_program_destroy(prog);
  free(src);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_COMMON(sched, reshape_e2e) {
  /* b = reshape(a, (2,4)) then flatten back — should be identity */
  int N = 8;
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a_buf = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *b_buf = poly_buffer(ctx, POLY_FLOAT32, N);

  /* Reshape a to (2,4), add 1.0, reshape back to (8) */
  int64_t dims[] = {2, 4};
  PolyUOp *a2d = poly_reshape(ctx, a_buf, dims, 2);
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(1.0));
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, a2d, one, poly_arg_none());
  int64_t flat[] = {8};
  PolyUOp *flat_result = poly_reshape(ctx, add, flat, 1);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, b_buf, flat_result, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyUOp *kernel = poly_get_kernel_graph(ctx, sink);
  ASSERT_NOT_NULL(kernel);

  int n_lin;
  PolyUOp **lin = poly_linearize(ctx, kernel, &n_lin);
  char *src = poly_render_c(lin, n_lin, "reshp");
  PolyProgram *prog = poly_compile_c(src, "reshp");
  ASSERT_NOT_NULL(prog);

  float a_d[8], b_d[8];
  for (int i = 0; i < N; i++) {
    a_d[i] = (float)(i + 1);
    b_d[i] = 0.0f;
  }

  void *args[2] = {b_d, a_d};
  poly_program_call(prog, args, 2);

  for (int i = 0; i < N; i++)
    ASSERT_FLOAT_EQ(b_d[i], a_d[i] + 1.0f, 1e-6);

  poly_program_destroy(prog);
  free(src);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

/* Reduce tests */

TEST(sched, reduce_sum_1d_ir) {
  /* sum(a) where a is 10 elements: verify kernel IR structure */
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *c = poly_buffer(ctx, POLY_FLOAT32, 1);
  int64_t axes[] = {0};
  PolyUOp *sum = poly_reduce_axis(ctx, POLY_OP_ADD, a, axes, 1);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, c, sum, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyUOp *kernel = poly_get_kernel_graph(ctx, sink);
  ASSERT_NOT_NULL(kernel);

  int n;
  PolyUOp **lin = poly_linearize(ctx, kernel, &n);
  ASSERT_TRUE(n > 0);

  /* Tinygrad parity on the optimized CPU path:
   * the scheduled root still has one RANGE, but full_rewrite_to_sink collapses
   * the 1D sum into scalarized LOAD/GEP/ADD without DEFINE_REG/RANGE/END in the
   * final linearized list. */
  ASSERT_INT_EQ(count_ops(lin, n, POLY_OP_PARAM), 2);
  ASSERT_INT_EQ(count_ops(lin, n, POLY_OP_DEFINE_REG), 0);
  ASSERT_INT_EQ(count_ops(lin, n, POLY_OP_RANGE), 0);
  ASSERT_INT_EQ(count_ops(lin, n, POLY_OP_LOAD), 3);
  ASSERT_INT_EQ(count_ops(lin, n, POLY_OP_ADD), 9);
  ASSERT_INT_EQ(count_ops(lin, n, POLY_OP_STORE), 1);
  ASSERT_INT_EQ(count_ops(lin, n, POLY_OP_END), 0);

  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_COMMON(sched, reduce_sum_1d_e2e) {
  /* sum([1, 2, ..., 10]) = 55 */
  int N = 10;
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *c = poly_buffer(ctx, POLY_FLOAT32, 1);
  int64_t axes[] = {0};
  PolyUOp *sum = poly_reduce_axis(ctx, POLY_OP_ADD, a, axes, 1);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, c, sum, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyUOp *kernel = poly_get_kernel_graph(ctx, sink);
  ASSERT_NOT_NULL(kernel);

  int n_lin;
  PolyUOp **lin = poly_linearize(ctx, kernel, &n_lin);
  char *src = poly_render_c(lin, n_lin, "sum1d");
  ASSERT_NOT_NULL(src);

  PolyProgram *prog = poly_compile_c(src, "sum1d");
  ASSERT_NOT_NULL(prog);

  float a_d[10], c_d[1] = {0.0f};
  float expected = 0.0f;
  for (int i = 0; i < N; i++) {
    a_d[i] = (float)(i + 1);
    expected += a_d[i];
  }

  void *args[2] = {c_d, a_d};
  poly_program_call(prog, args, 2);
  ASSERT_FLOAT_EQ(c_d[0], expected, 1e-4);

  poly_program_destroy(prog);
  free(src);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_COMMON(sched, reduce_sum_axis0_e2e) {
  /* Column sum: a(4,3) reduced on axis 0 -> 3 output elements */
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, 12);
  PolyUOp *c = poly_buffer(ctx, POLY_FLOAT32, 3);
  int64_t rdims[] = {4, 3};
  PolyUOp *a2d = poly_reshape(ctx, a, rdims, 2);
  int64_t axes[] = {0};
  PolyUOp *sum = poly_reduce_axis(ctx, POLY_OP_ADD, a2d, axes, 1);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, c, sum, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyUOp *kernel = poly_get_kernel_graph(ctx, sink);
  ASSERT_NOT_NULL(kernel);

  int n_lin;
  PolyUOp **lin = poly_linearize(ctx, kernel, &n_lin);
  char *src = poly_render_c(lin, n_lin, "colsum");
  PolyProgram *prog = poly_compile_c(src, "colsum");
  ASSERT_NOT_NULL(prog);

  float a_d[12], c_d[3] = {0};
  for (int i = 0; i < 12; i++)
    a_d[i] = (float)(i + 1);

  void *args[2] = {c_d, a_d};
  poly_program_call(prog, args, 2);

  for (int j = 0; j < 3; j++) {
    float exp = 0;
    for (int i = 0; i < 4; i++)
      exp += a_d[i * 3 + j];
    ASSERT_FLOAT_EQ(c_d[j], exp, 1e-4);
  }

  poly_program_destroy(prog);
  free(src);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_COMMON(sched, reduce_sum_axis1_e2e) {
  /* Row sum: a(4,3) reduced on axis 1 -> 4 output elements */
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, 12);
  PolyUOp *c = poly_buffer(ctx, POLY_FLOAT32, 4);
  int64_t rdims[] = {4, 3};
  PolyUOp *a2d = poly_reshape(ctx, a, rdims, 2);
  int64_t axes[] = {1};
  PolyUOp *sum = poly_reduce_axis(ctx, POLY_OP_ADD, a2d, axes, 1);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, c, sum, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyUOp *kernel = poly_get_kernel_graph(ctx, sink);
  ASSERT_NOT_NULL(kernel);

  int n_lin;
  PolyUOp **lin = poly_linearize(ctx, kernel, &n_lin);
  char *src = poly_render_c(lin, n_lin, "rowsum");
  PolyProgram *prog = poly_compile_c(src, "rowsum");
  ASSERT_NOT_NULL(prog);

  float a_d[12], c_d[4] = {0};
  for (int i = 0; i < 12; i++)
    a_d[i] = (float)(i + 1);

  void *args[2] = {c_d, a_d};
  poly_program_call(prog, args, 2);

  for (int i = 0; i < 4; i++) {
    float exp = 0;
    for (int j = 0; j < 3; j++)
      exp += a_d[i * 3 + j];
    ASSERT_FLOAT_EQ(c_d[i], exp, 1e-4);
  }

  poly_program_destroy(prog);
  free(src);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_COMMON(sched, reduce_sum_all_e2e) {
  /* Full reduction: a(4,3) reduced on both axes -> scalar */
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, 12);
  PolyUOp *c = poly_buffer(ctx, POLY_FLOAT32, 1);
  int64_t rdims[] = {4, 3};
  PolyUOp *a2d = poly_reshape(ctx, a, rdims, 2);
  int64_t axes[] = {0, 1};
  PolyUOp *sum = poly_reduce_axis(ctx, POLY_OP_ADD, a2d, axes, 2);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, c, sum, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyUOp *kernel = poly_get_kernel_graph(ctx, sink);
  ASSERT_NOT_NULL(kernel);

  int n_lin;
  PolyUOp **lin = poly_linearize(ctx, kernel, &n_lin);
  char *src = poly_render_c(lin, n_lin, "allsum");
  PolyProgram *prog = poly_compile_c(src, "allsum");
  ASSERT_NOT_NULL(prog);

  float a_d[12], c_d[1] = {0};
  float expected = 0;
  for (int i = 0; i < 12; i++) {
    a_d[i] = (float)(i + 1);
    expected += a_d[i];
  }

  void *args[2] = {c_d, a_d};
  poly_program_call(prog, args, 2);
  ASSERT_FLOAT_EQ(c_d[0], expected, 1e-4);

  poly_program_destroy(prog);
  free(src);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sched, reduce_max_e2e) {
  /* max([3, 1, 4, 1, 5, 9, 2, 6]) = 9 */
  int N = 8;
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *c = poly_buffer(ctx, POLY_FLOAT32, 1);
  int64_t axes[] = {0};
  PolyUOp *mx = poly_reduce_axis(ctx, POLY_OP_MAX, a, axes, 1);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, c, mx, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyUOp *kernel = poly_get_kernel_graph(ctx, sink);
  ASSERT_NOT_NULL(kernel);

  int n_lin;
  PolyUOp **lin = poly_linearize(ctx, kernel, &n_lin);
  char *src = poly_render_c(lin, n_lin, "maxred");
  PolyProgram *prog = poly_compile_c(src, "maxred");
  ASSERT_NOT_NULL(prog);

  float a_d[8] = {3, 1, 4, 1, 5, 9, 2, 6};
  float c_d[1] = {0};

  void *args[2] = {c_d, a_d};
  poly_program_call(prog, args, 2);
  ASSERT_FLOAT_EQ(c_d[0], 9.0f, 1e-6);

  poly_program_destroy(prog);
  free(src);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_COMMON(sched, reduce_scalar_chain_e2e) {
  /* tinygrad parity boundary: schedule_with_vars splits the reduced scalar
   * producer from the broadcasted consumer kernel. */
  int N = 8;
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *b = poly_buffer(ctx, POLY_FLOAT32, N);

  int64_t axes[] = {0};
  PolyUOp *sum = poly_reduce_axis(ctx, POLY_OP_ADD, a, axes, 1);
  PolyUOp *sum_scalar = poly_reshape(ctx, sum, NULL, 0);
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, sum_scalar, b, poly_arg_none());

  float a_d[8], b_d[8], c_d[8];
  float sumv = 0.0f;
  for (int i = 0; i < N; i++) {
    a_d[i] = (float)(i + 1);
    b_d[i] = (float)(10 + i);
    c_d[i] = 0.0f;
    sumv += a_d[i];
  }

  poly_buffer_set(ctx, a, a_d, sizeof(a_d), POLY_DEVICE_CPU);
  poly_buffer_set(ctx, b, b_d, sizeof(b_d), POLY_DEVICE_CPU);

  PolyUOp *targets[] = {add};
  PolyUOp *realized[] = {NULL};
  PolySchedule *sched = poly_schedule_with_vars(ctx, targets, 1, realized);
  ASSERT_NOT_NULL(sched);
  ASSERT_NOT_NULL(realized[0]);
  ASSERT_INT_EQ(sched->template->n_calls, 2);

  ASSERT_INT_EQ(count_root_ops(ctx, poly_schedule_call_body(sched, 0), POLY_OP_STORE), 1);
  ASSERT_INT_EQ(count_root_ops(ctx, poly_schedule_call_body(sched, 0), POLY_OP_REDUCE), 1);
  ASSERT_INT_EQ(count_root_ops(ctx, poly_schedule_call_body(sched, 0), POLY_OP_RANGE), 1);
  ASSERT_INT_EQ(count_root_ops(ctx, poly_schedule_call_body(sched, 0), POLY_OP_END), 0);
  ASSERT_INT_EQ(count_root_ops(ctx, poly_schedule_call_body(sched, 0), POLY_OP_INDEX), 2);

  ASSERT_INT_EQ(count_root_ops(ctx, poly_schedule_call_body(sched, 1), POLY_OP_STORE), 1);
  ASSERT_INT_EQ(count_root_ops(ctx, poly_schedule_call_body(sched, 1), POLY_OP_REDUCE), 0);
  ASSERT_INT_EQ(count_root_ops(ctx, poly_schedule_call_body(sched, 1), POLY_OP_RANGE), 1);
  ASSERT_INT_EQ(count_root_ops(ctx, poly_schedule_call_body(sched, 1), POLY_OP_END), 1);
  ASSERT_INT_EQ(count_root_ops(ctx, poly_schedule_call_body(sched, 1), POLY_OP_INDEX), 3);
  ASSERT_INT_EQ(count_root_ops(ctx, poly_schedule_call_body(sched, 1), POLY_OP_ADD), 1);

  ASSERT_INT_EQ(poly_run_schedule(ctx, sched, NULL, 0), 0);

  const PolyUOp *out = poly_uop_get_buffer_identity(realized[0]);
  ASSERT_TRUE(out != NULL);
  float dout[8] = {0};
  ASSERT_INT_EQ(poly_buffer_read(ctx, (PolyUOp *)out, dout, sizeof(dout)), 0);

  for (int i = 0; i < N; i++)
    ASSERT_FLOAT_EQ(dout[i], sumv + b_d[i], 1e-5);

  poly_schedule_free(sched);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_COMMON(sched, reduce_vector_chain_e2e) {
  /* tinygrad parity boundary: schedule_with_vars splits the reduced row sums
   * from the expanded consumer kernel. */
  int N = 12;
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *b = poly_buffer(ctx, POLY_FLOAT32, N);

  int64_t dims2d[] = {4, 3};
  PolyUOp *a2d = poly_reshape(ctx, a, dims2d, 2);
  PolyUOp *b2d = poly_reshape(ctx, b, dims2d, 2);

  int64_t axes[] = {1};
  PolyUOp *sum = poly_reduce_axis(ctx, POLY_OP_ADD, a2d, axes, 1); /* shape (4,1) */
  int64_t expd[] = {4, 3};
  PolyUOp *sum_exp = poly_expand(ctx, sum, expd, 2); /* shape (4,3) */

  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, sum_exp, b2d, poly_arg_none());

  float a_d[12], b_d[12], c_d[12];
  for (int i = 0; i < N; i++) {
    a_d[i] = (float)(i + 1); /* rows: [1 2 3], [4 5 6], ... */
    b_d[i] = (float)(100 + i);
    c_d[i] = 0.0f;
  }

  poly_buffer_set(ctx, a, a_d, sizeof(a_d), POLY_DEVICE_CPU);
  poly_buffer_set(ctx, b, b_d, sizeof(b_d), POLY_DEVICE_CPU);

  PolyUOp *targets[] = {add};
  PolyUOp *realized[] = {NULL};
  PolySchedule *sched = poly_schedule_with_vars(ctx, targets, 1, realized);
  ASSERT_NOT_NULL(sched);
  ASSERT_NOT_NULL(realized[0]);
  ASSERT_INT_EQ(sched->template->n_calls, 2);

  ASSERT_INT_EQ(count_root_ops(ctx, poly_schedule_call_body(sched, 0), POLY_OP_STORE), 1);
  ASSERT_INT_EQ(count_root_ops(ctx, poly_schedule_call_body(sched, 0), POLY_OP_REDUCE), 1);
  ASSERT_INT_EQ(count_root_ops(ctx, poly_schedule_call_body(sched, 0), POLY_OP_RANGE), 2);
  ASSERT_INT_EQ(count_root_ops(ctx, poly_schedule_call_body(sched, 0), POLY_OP_END), 1);
  ASSERT_INT_EQ(count_root_ops(ctx, poly_schedule_call_body(sched, 0), POLY_OP_INDEX), 2);
  ASSERT_INT_EQ(count_root_ops(ctx, poly_schedule_call_body(sched, 0), POLY_OP_ADD), 1);

  ASSERT_INT_EQ(count_root_ops(ctx, poly_schedule_call_body(sched, 1), POLY_OP_STORE), 1);
  ASSERT_INT_EQ(count_root_ops(ctx, poly_schedule_call_body(sched, 1), POLY_OP_REDUCE), 0);
  ASSERT_INT_EQ(count_root_ops(ctx, poly_schedule_call_body(sched, 1), POLY_OP_RANGE), 2);
  ASSERT_INT_EQ(count_root_ops(ctx, poly_schedule_call_body(sched, 1), POLY_OP_END), 1);
  ASSERT_INT_EQ(count_root_ops(ctx, poly_schedule_call_body(sched, 1), POLY_OP_INDEX), 3);
  ASSERT_INT_EQ(count_root_ops(ctx, poly_schedule_call_body(sched, 1), POLY_OP_ADD), 2);

  ASSERT_INT_EQ(poly_run_schedule(ctx, sched, NULL, 0), 0);

  const PolyUOp *out = poly_uop_get_buffer_identity(realized[0]);
  ASSERT_TRUE(out != NULL);
  float dout[12] = {0};
  ASSERT_INT_EQ(poly_buffer_read(ctx, (PolyUOp *)out, dout, sizeof(dout)), 0);

  for (int i = 0; i < 4; i++) {
    float row_sum = 0.0f;
    for (int j = 0; j < 3; j++)
      row_sum += a_d[i * 3 + j];
    for (int j = 0; j < 3; j++) {
      int idx = i * 3 + j;
      ASSERT_FLOAT_EQ(dout[idx], row_sum + b_d[idx], 1e-5);
    }
  }

  poly_schedule_free(sched);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_COMMON(sched, shared_scalar_reduce_two_stores_e2e) {
  /* tinygrad parity boundary: two realized targets that share the same scalar
   * reduction schedule as:
   *   1. the shared reduction producer
   *   2. the add consumer
   *   3. the mul consumer */
  int N = 8;
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *c = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *e = poly_buffer(ctx, POLY_FLOAT32, N);

  int64_t axes[] = {0};
  PolyUOp *sum = poly_reduce_axis(ctx, POLY_OP_ADD, a, axes, 1);
  PolyUOp *sum_scalar = poly_reshape(ctx, sum, NULL, 0);
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, sum_scalar, c, poly_arg_none());
  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, sum_scalar, e, poly_arg_none());

  /* E2E check via poly_test_realize_buffer_views */
  float a_d[8], c_d[8], e_d[8];
  float c0[8], e0[8];
  float sumv = 0.0f;
  for (int i = 0; i < N; i++) {
    a_d[i] = (float)(i + 1);
    c_d[i] = (float)(10 + i);
    e_d[i] = (float)(20 + i);
    c0[i] = c_d[i];
    e0[i] = e_d[i];
    sumv += a_d[i];
  }

  poly_buffer_set(ctx, a, a_d, sizeof(a_d), POLY_DEVICE_CPU);
  poly_buffer_set(ctx, c, c_d, sizeof(c_d), POLY_DEVICE_CPU);
  poly_buffer_set(ctx, e, e_d, sizeof(e_d), POLY_DEVICE_CPU);

  PolyUOp *targets[] = {add, mul};
  PolyUOp *realized[] = {NULL, NULL};
  PolySchedule *sched = poly_schedule_with_vars(ctx, targets, 2, realized);
  ASSERT_NOT_NULL(sched);
  ASSERT_NOT_NULL(realized[0]);
  ASSERT_NOT_NULL(realized[1]);
  ASSERT_INT_EQ(sched->template->n_calls, 3);

  ASSERT_INT_EQ(count_root_ops(ctx, poly_schedule_call_body(sched, 0), POLY_OP_STORE), 1);
  ASSERT_INT_EQ(count_root_ops(ctx, poly_schedule_call_body(sched, 0), POLY_OP_REDUCE), 1);
  ASSERT_INT_EQ(count_root_ops(ctx, poly_schedule_call_body(sched, 0), POLY_OP_RANGE), 1);
  ASSERT_INT_EQ(count_root_ops(ctx, poly_schedule_call_body(sched, 0), POLY_OP_END), 0);
  ASSERT_INT_EQ(count_root_ops(ctx, poly_schedule_call_body(sched, 0), POLY_OP_INDEX), 2);

  ASSERT_INT_EQ(count_root_ops(ctx, poly_schedule_call_body(sched, 1), POLY_OP_STORE), 1);
  ASSERT_INT_EQ(count_root_ops(ctx, poly_schedule_call_body(sched, 1), POLY_OP_REDUCE), 0);
  ASSERT_INT_EQ(count_root_ops(ctx, poly_schedule_call_body(sched, 1), POLY_OP_RANGE), 1);
  ASSERT_INT_EQ(count_root_ops(ctx, poly_schedule_call_body(sched, 1), POLY_OP_END), 1);
  ASSERT_INT_EQ(count_root_ops(ctx, poly_schedule_call_body(sched, 1), POLY_OP_INDEX), 3);
  ASSERT_INT_EQ(count_root_ops(ctx, poly_schedule_call_body(sched, 1), POLY_OP_ADD), 1);
  ASSERT_INT_EQ(count_root_ops(ctx, poly_schedule_call_body(sched, 1), POLY_OP_MUL), 0);

  ASSERT_INT_EQ(count_root_ops(ctx, poly_schedule_call_body(sched, 2), POLY_OP_STORE), 1);
  ASSERT_INT_EQ(count_root_ops(ctx, poly_schedule_call_body(sched, 2), POLY_OP_REDUCE), 0);
  ASSERT_INT_EQ(count_root_ops(ctx, poly_schedule_call_body(sched, 2), POLY_OP_RANGE), 1);
  ASSERT_INT_EQ(count_root_ops(ctx, poly_schedule_call_body(sched, 2), POLY_OP_END), 1);
  ASSERT_INT_EQ(count_root_ops(ctx, poly_schedule_call_body(sched, 2), POLY_OP_INDEX), 3);
  ASSERT_INT_EQ(count_root_ops(ctx, poly_schedule_call_body(sched, 2), POLY_OP_ADD), 0);
  ASSERT_INT_EQ(count_root_ops(ctx, poly_schedule_call_body(sched, 2), POLY_OP_MUL), 1);

  ASSERT_INT_EQ(poly_run_schedule(ctx, sched, NULL, 0), 0);

  const PolyUOp *c_out = poly_uop_get_buffer_identity(realized[0]);
  const PolyUOp *e_out = poly_uop_get_buffer_identity(realized[1]);
  ASSERT_TRUE(c_out != NULL);
  ASSERT_TRUE(e_out != NULL);
  float c_res[8] = {0};
  float e_res[8] = {0};
  ASSERT_INT_EQ(poly_buffer_read(ctx, (PolyUOp *)c_out, c_res, sizeof(c_res)), 0);
  ASSERT_INT_EQ(poly_buffer_read(ctx, (PolyUOp *)e_out, e_res, sizeof(e_res)), 0);

  for (int i = 0; i < N; i++) {
    ASSERT_FLOAT_EQ(c_res[i], c0[i] + sumv, 1e-5);
    ASSERT_FLOAT_EQ(e_res[i], e0[i] * sumv, 1e-5);
  }

  poly_schedule_free(sched);
  poly_ctx_destroy(ctx);
  PASS();
}

/* Movement op tests */

TEST(sched, permute_2d_e2e) {
  /* Transpose (3,4) → (4,3) */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, 12);
  PolyUOp *c = poly_buffer(ctx, POLY_FLOAT32, 12);

  int64_t rdims[] = {3, 4};
  PolyUOp *a2d = poly_reshape(ctx, a, rdims, 2);
  int64_t perm[] = {1, 0};
  PolyUOp *t = poly_permute(ctx, a2d, perm, 2);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, c, t, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyUOp *kernel = poly_get_kernel_graph(ctx, sink);
  ASSERT_NOT_NULL(kernel);
  int n_lin;
  PolyUOp **lin = poly_linearize(ctx, kernel, &n_lin);
  char *src = poly_render_c(lin, n_lin, "perm2d");
  PolyProgram *prog = poly_compile_c(src, "perm2d");
  ASSERT_NOT_NULL(prog);

  /* a = [[0,1,2,3],[4,5,6,7],[8,9,10,11]] (3x4, row-major) */
  float a_d[12], c_d[12];
  for (int i = 0; i < 12; i++) {
    a_d[i] = (float)i;
    c_d[i] = -1.0f;
  }

  void *args[2] = {c_d, a_d};
  poly_program_call(prog, args, 2);

  /* c = transpose -> (4x3): c[j][i] = a[i][j] */
  float expected[] = {0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11};
  for (int i = 0; i < 12; i++)
    ASSERT_FLOAT_EQ(c_d[i], expected[i], 1e-6);

  poly_program_destroy(prog);
  free(src);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sched, permute_3d_e2e) {
  /* Permute (2,3,4) -> (4,2,3) via perm=(2,0,1) */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, 24);
  PolyUOp *c = poly_buffer(ctx, POLY_FLOAT32, 24);

  int64_t rdims[] = {2, 3, 4};
  PolyUOp *a3d = poly_reshape(ctx, a, rdims, 3);
  int64_t perm[] = {2, 0, 1};
  PolyUOp *t = poly_permute(ctx, a3d, perm, 3);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, c, t, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyUOp *kernel = poly_get_kernel_graph(ctx, sink);
  ASSERT_NOT_NULL(kernel);
  int n_lin;
  PolyUOp **lin = poly_linearize(ctx, kernel, &n_lin);
  char *src = poly_render_c(lin, n_lin, "perm3d");
  PolyProgram *prog = poly_compile_c(src, "perm3d");
  ASSERT_NOT_NULL(prog);

  float a_d[24], c_d[24];
  for (int i = 0; i < 24; i++) {
    a_d[i] = (float)i;
    c_d[i] = -1.0f;
  }

  void *args[2] = {c_d, a_d};
  poly_program_call(prog, args, 2);

  /* out[k][i][j] = a[i][j][k], out shape (4,2,3) */
  for (int k = 0; k < 4; k++)
    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 3; j++)
        ASSERT_FLOAT_EQ(c_d[k * 6 + i * 3 + j], a_d[i * 12 + j * 4 + k], 1e-6);

  poly_program_destroy(prog);
  free(src);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sched, shrink_1d_e2e) {
  /* Slice [2:5] from 8-element vector -> 3 elements */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, 8);
  PolyUOp *c = poly_buffer(ctx, POLY_FLOAT32, 3);

  int64_t pairs[][2] = {{2, 5}};
  PolyUOp *s = poly_shrink(ctx, a, pairs, 1);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, c, s, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyUOp *kernel = poly_get_kernel_graph(ctx, sink);
  ASSERT_NOT_NULL(kernel);
  int n_lin;
  PolyUOp **lin = poly_linearize(ctx, kernel, &n_lin);
  char *src = poly_render_c(lin, n_lin, "shrk1d");
  PolyProgram *prog = poly_compile_c(src, "shrk1d");
  ASSERT_NOT_NULL(prog);

  float a_d[8], c_d[3];
  for (int i = 0; i < 8; i++)
    a_d[i] = (float)(i * 10);
  for (int i = 0; i < 3; i++)
    c_d[i] = -1.0f;

  void *args[2] = {c_d, a_d};
  poly_program_call(prog, args, 2);

  /* c = a[2:5] = {20, 30, 40} */
  ASSERT_FLOAT_EQ(c_d[0], 20.0f, 1e-6);
  ASSERT_FLOAT_EQ(c_d[1], 30.0f, 1e-6);
  ASSERT_FLOAT_EQ(c_d[2], 40.0f, 1e-6);

  poly_program_destroy(prog);
  free(src);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sched, shrink_2d_e2e) {
  /* Shrink (4,3) -> rows 1:3 -> (2,3) */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, 12);
  PolyUOp *c = poly_buffer(ctx, POLY_FLOAT32, 6);

  int64_t rdims[] = {4, 3};
  PolyUOp *a2d = poly_reshape(ctx, a, rdims, 2);
  int64_t pairs[][2] = {{1, 3}, {0, 3}};
  PolyUOp *s = poly_shrink(ctx, a2d, pairs, 2);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, c, s, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyUOp *kernel = poly_get_kernel_graph(ctx, sink);
  ASSERT_NOT_NULL(kernel);
  int n_lin;
  PolyUOp **lin = poly_linearize(ctx, kernel, &n_lin);
  char *src = poly_render_c(lin, n_lin, "shrk2d");
  PolyProgram *prog = poly_compile_c(src, "shrk2d");
  ASSERT_NOT_NULL(prog);

  /* a = [[0,1,2],[3,4,5],[6,7,8],[9,10,11]] (4x3) */
  float a_d[12], c_d[6];
  for (int i = 0; i < 12; i++)
    a_d[i] = (float)i;
  for (int i = 0; i < 6; i++)
    c_d[i] = -1.0f;

  void *args[2] = {c_d, a_d};
  poly_program_call(prog, args, 2);

  /* c = a[1:3, 0:3] = [[3,4,5],[6,7,8]] */
  float expected[] = {3, 4, 5, 6, 7, 8};
  for (int i = 0; i < 6; i++)
    ASSERT_FLOAT_EQ(c_d[i], expected[i], 1e-6);

  poly_program_destroy(prog);
  free(src);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sched, flip_1d_e2e) {
  /* Reverse 5-element vector */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, 5);
  PolyUOp *c = poly_buffer(ctx, POLY_FLOAT32, 5);

  int64_t axes[] = {0};
  PolyUOp *f = poly_flip(ctx, a, axes, 1);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, c, f, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyUOp *kernel = poly_get_kernel_graph(ctx, sink);
  ASSERT_NOT_NULL(kernel);
  int n_lin;
  PolyUOp **lin = poly_linearize(ctx, kernel, &n_lin);
  char *src = poly_render_c(lin, n_lin, "flip1d");
  PolyProgram *prog = poly_compile_c(src, "flip1d");
  ASSERT_NOT_NULL(prog);

  float a_d[] = {10, 20, 30, 40, 50};
  float c_d[5] = {0};

  void *args[2] = {c_d, a_d};
  poly_program_call(prog, args, 2);

  ASSERT_FLOAT_EQ(c_d[0], 50.0f, 1e-6);
  ASSERT_FLOAT_EQ(c_d[1], 40.0f, 1e-6);
  ASSERT_FLOAT_EQ(c_d[2], 30.0f, 1e-6);
  ASSERT_FLOAT_EQ(c_d[3], 20.0f, 1e-6);
  ASSERT_FLOAT_EQ(c_d[4], 10.0f, 1e-6);

  poly_program_destroy(prog);
  free(src);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sched, flip_2d_axis0_e2e) {
  /* Flip rows of (3,4) matrix */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, 12);
  PolyUOp *c = poly_buffer(ctx, POLY_FLOAT32, 12);

  int64_t rdims[] = {3, 4};
  PolyUOp *a2d = poly_reshape(ctx, a, rdims, 2);
  int64_t axes[] = {0};
  PolyUOp *f = poly_flip(ctx, a2d, axes, 1);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, c, f, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyUOp *kernel = poly_get_kernel_graph(ctx, sink);
  ASSERT_NOT_NULL(kernel);
  int n_lin;
  PolyUOp **lin = poly_linearize(ctx, kernel, &n_lin);
  char *src = poly_render_c(lin, n_lin, "flip2a");
  PolyProgram *prog = poly_compile_c(src, "flip2a");
  ASSERT_NOT_NULL(prog);

  float a_d[12], c_d[12];
  for (int i = 0; i < 12; i++) {
    a_d[i] = (float)i;
    c_d[i] = -1.0f;
  }

  void *args[2] = {c_d, a_d};
  poly_program_call(prog, args, 2);

  /* Flip axis 0: [[8,9,10,11],[4,5,6,7],[0,1,2,3]] */
  float expected[] = {8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3};
  for (int i = 0; i < 12; i++)
    ASSERT_FLOAT_EQ(c_d[i], expected[i], 1e-6);

  poly_program_destroy(prog);
  free(src);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sched, flip_2d_both_e2e) {
  /* Flip both axes of (3,4) matrix */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, 12);
  PolyUOp *c = poly_buffer(ctx, POLY_FLOAT32, 12);

  int64_t rdims[] = {3, 4};
  PolyUOp *a2d = poly_reshape(ctx, a, rdims, 2);
  int64_t axes[] = {0, 1};
  PolyUOp *f = poly_flip(ctx, a2d, axes, 2);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, c, f, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyUOp *kernel = poly_get_kernel_graph(ctx, sink);
  ASSERT_NOT_NULL(kernel);
  int n_lin;
  PolyUOp **lin = poly_linearize(ctx, kernel, &n_lin);
  char *src = poly_render_c(lin, n_lin, "flip2b");
  PolyProgram *prog = poly_compile_c(src, "flip2b");
  ASSERT_NOT_NULL(prog);

  float a_d[12], c_d[12];
  for (int i = 0; i < 12; i++) {
    a_d[i] = (float)i;
    c_d[i] = -1.0f;
  }

  void *args[2] = {c_d, a_d};
  poly_program_call(prog, args, 2);

  /* Flip both: [[11,10,9,8],[7,6,5,4],[3,2,1,0]] */
  float expected[] = {11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
  for (int i = 0; i < 12; i++)
    ASSERT_FLOAT_EQ(c_d[i], expected[i], 1e-6);

  poly_program_destroy(prog);
  free(src);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sched, pad_1d_e2e) {
  /* Pad 3-element vector: (2,2) -> 7 elements */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, 3);
  PolyUOp *c = poly_buffer(ctx, POLY_FLOAT32, 7);

  int64_t pairs[][2] = {{2, 2}};
  PolyUOp *p = poly_pad(ctx, a, pairs, 1);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, c, p, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyUOp *kernel = poly_get_kernel_graph(ctx, sink);
  ASSERT_NOT_NULL(kernel);
  int n_lin;
  PolyUOp **lin = poly_linearize(ctx, kernel, &n_lin);
  char *src = poly_render_c(lin, n_lin, "pad1d");
  PolyProgram *prog = poly_compile_c(src, "pad1d");
  ASSERT_NOT_NULL(prog);

  float a_d[] = {10, 20, 30};
  float c_d[7];
  for (int i = 0; i < 7; i++)
    c_d[i] = -1.0f;

  void *args[2] = {c_d, a_d};
  poly_program_call(prog, args, 2);

  /* c = [0, 0, 10, 20, 30, 0, 0] */
  float expected[] = {0, 0, 10, 20, 30, 0, 0};
  for (int i = 0; i < 7; i++)
    ASSERT_FLOAT_EQ(c_d[i], expected[i], 1e-6);

  poly_program_destroy(prog);
  free(src);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sched, pad_2d_e2e) {
  /* Pad (2,3) -> (4,5) with (1,1) before/after on each dim */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, 6);
  PolyUOp *c = poly_buffer(ctx, POLY_FLOAT32, 20);

  int64_t rdims[] = {2, 3};
  PolyUOp *a2d = poly_reshape(ctx, a, rdims, 2);
  int64_t pairs[][2] = {{1, 1}, {1, 1}};
  PolyUOp *p = poly_pad(ctx, a2d, pairs, 2);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, c, p, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyUOp *kernel = poly_get_kernel_graph(ctx, sink);
  ASSERT_NOT_NULL(kernel);
  int n_lin;
  PolyUOp **lin = poly_linearize(ctx, kernel, &n_lin);
  char *src = poly_render_c(lin, n_lin, "pad2d");
  PolyProgram *prog = poly_compile_c(src, "pad2d");
  ASSERT_NOT_NULL(prog);

  /* a = [[1,2,3],[4,5,6]] (2x3) */
  float a_d[] = {1, 2, 3, 4, 5, 6};
  float c_d[20];
  for (int i = 0; i < 20; i++)
    c_d[i] = -1.0f;

  void *args[2] = {c_d, a_d};
  poly_program_call(prog, args, 2);

  /* c (4x5): [[0,0,0,0,0], [0,1,2,3,0], [0,4,5,6,0], [0,0,0,0,0]] */
  float expected[] = {0, 0, 0, 0, 0, 0, 1, 2, 3, 0, 0, 4, 5, 6, 0, 0, 0, 0, 0, 0};
  for (int i = 0; i < 20; i++)
    ASSERT_FLOAT_EQ(c_d[i], expected[i], 1e-6);

  poly_program_destroy(prog);
  free(src);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sched, chain_permute_shrink_e2e) {
  /* Transpose (3,4) then shrink rows 0:2 -> (2,3) */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, 12);
  PolyUOp *c = poly_buffer(ctx, POLY_FLOAT32, 6);

  int64_t rdims[] = {3, 4};
  PolyUOp *a2d = poly_reshape(ctx, a, rdims, 2);
  int64_t perm[] = {1, 0};
  PolyUOp *t = poly_permute(ctx, a2d, perm, 2); /* (4,3) */
  int64_t pairs[][2] = {{0, 2}, {0, 3}};
  PolyUOp *s = poly_shrink(ctx, t, pairs, 2); /* (2,3) */
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, c, s, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyUOp *kernel = poly_get_kernel_graph(ctx, sink);
  ASSERT_NOT_NULL(kernel);
  int n_lin;
  PolyUOp **lin = poly_linearize(ctx, kernel, &n_lin);
  char *src = poly_render_c(lin, n_lin, "ps_ch");
  PolyProgram *prog = poly_compile_c(src, "ps_ch");
  ASSERT_NOT_NULL(prog);

  float a_d[12], c_d[6];
  for (int i = 0; i < 12; i++)
    a_d[i] = (float)i;
  for (int i = 0; i < 6; i++)
    c_d[i] = -1.0f;

  void *args[2] = {c_d, a_d};
  poly_program_call(prog, args, 2);

  /* transpose -> [[0,4,8],[1,5,9],[2,6,10],[3,7,11]] (4x3)
   * shrink rows 0:2 -> [[0,4,8],[1,5,9]] (2x3) */
  float expected[] = {0, 4, 8, 1, 5, 9};
  for (int i = 0; i < 6; i++)
    ASSERT_FLOAT_EQ(c_d[i], expected[i], 1e-6);

  poly_program_destroy(prog);
  free(src);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sched, chain_pad_flip_e2e) {
  /* Pad [1,2,3] to [0,1,2,3,0] then flip -> [0,3,2,1,0] */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, 3);
  PolyUOp *c = poly_buffer(ctx, POLY_FLOAT32, 5);

  int64_t pad_pairs[][2] = {{1, 1}};
  PolyUOp *p = poly_pad(ctx, a, pad_pairs, 1); /* 5 elements */
  int64_t axes[] = {0};
  PolyUOp *f = poly_flip(ctx, p, axes, 1);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, c, f, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyUOp *kernel = poly_get_kernel_graph(ctx, sink);
  ASSERT_NOT_NULL(kernel);
  int n_lin;
  PolyUOp **lin = poly_linearize(ctx, kernel, &n_lin);
  char *src = poly_render_c(lin, n_lin, "pf_ch");
  PolyProgram *prog = poly_compile_c(src, "pf_ch");
  ASSERT_NOT_NULL(prog);

  float a_d[] = {1, 2, 3};
  float c_d[5] = {-1, -1, -1, -1, -1};

  void *args[2] = {c_d, a_d};
  poly_program_call(prog, args, 2);

  /* pad -> [0,1,2,3,0], flip -> [0,3,2,1,0] */
  float expected[] = {0, 3, 2, 1, 0};
  for (int i = 0; i < 5; i++)
    ASSERT_FLOAT_EQ(c_d[i], expected[i], 1e-6);

  poly_program_destroy(prog);
  free(src);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sched, movement_alu_chain_e2e) {
  /* flip([1,2,3,4]) + [10,20,30,40] = [14,23,32,41] */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, 4);
  PolyUOp *b = poly_buffer(ctx, POLY_FLOAT32, 4);
  PolyUOp *c = poly_buffer(ctx, POLY_FLOAT32, 4);

  int64_t axes[] = {0};
  PolyUOp *f = poly_flip(ctx, a, axes, 1);
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, f, b, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, c, add, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyUOp *kernel = poly_get_kernel_graph(ctx, sink);
  ASSERT_NOT_NULL(kernel);
  int n_lin;
  PolyUOp **lin = poly_linearize(ctx, kernel, &n_lin);
  char *src = poly_render_c(lin, n_lin, "mvalu");
  PolyProgram *prog = poly_compile_c(src, "mvalu");
  ASSERT_NOT_NULL(prog);

  float a_d[] = {1, 2, 3, 4};
  float b_d[] = {10, 20, 30, 40};
  float c_d[4] = {0};

  void *args[3] = {c_d, a_d, b_d};
  poly_program_call(prog, args, 3);

  float expected[] = {14, 23, 32, 41};
  for (int i = 0; i < 4; i++)
    ASSERT_FLOAT_EQ(c_d[i], expected[i], 1e-6);

  poly_program_destroy(prog);
  free(src);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

/* poly_test_realize_buffer_views tests */

#include "../src/frontend.h"

TEST(sched, realize_vecadd) {
  int N = 16;
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer_f32(ctx, N);
  PolyUOp *b = poly_buffer_f32(ctx, N);
  PolyUOp *c = poly_buffer_f32(ctx, N);
  PolyUOp *add = poly_alu2(ctx, POLY_OP_ADD, a, b);
  PolyUOp *store = poly_store_val(ctx, c, add);
  PolyUOp *sink = poly_sink1(ctx, store);

  float a_d[16], b_d[16], c_d[16];
  for (int i = 0; i < N; i++) {
    a_d[i] = (float)(i + 1);
    b_d[i] = (float)(i + 1) * 0.5f;
    c_d[i] = 0;
  }

  PolyTestBufferView bindings[] = {
      POLY_TEST_HOST_VIEW(a, a_d), POLY_TEST_HOST_VIEW(b, b_d), POLY_TEST_HOST_VIEW(c, c_d)
  };
  int ret = poly_test_realize_buffer_views(ctx, sink, bindings, 3);
  ASSERT_INT_EQ(ret, 0);
  for (int i = 0; i < N; i++)
    ASSERT_FLOAT_EQ(c_d[i], a_d[i] + b_d[i], 1e-6);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sched, custom_kernel_call_after_executes) {
  int N = 4;
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *c = poly_buffer_f32(ctx, N);
  PolyUOp *a = poly_buffer_f32(ctx, N);
  PolyUOp *b = poly_buffer_f32(ctx, N);

  PolyUOp *pc = poly_uop_flatten(ctx, poly_uop_placeholder_like(ctx, c, 0));
  PolyUOp *pa = poly_uop_flatten(ctx, poly_uop_placeholder_like(ctx, a, 1));
  PolyUOp *pb = poly_uop_flatten(ctx, poly_uop_placeholder_like(ctx, b, 2));
  PolyUOp *i = poly_uop_range(ctx, N, 0, POLY_AXIS_LOOP);
  PolyUOp *idxs[1] = {i};
  PolyUOp *ci = poly_uop_index(ctx, pc, idxs, 1, 0);
  PolyUOp *ai = poly_uop_index(ctx, pa, idxs, 1, 0);
  PolyUOp *bi = poly_uop_index(ctx, pb, idxs, 1, 0);
  PolyUOp *sum = poly_alu2(ctx, POLY_OP_ADD, ai, bi);
  PolyUOp *store = poly_uop_store(ctx, ci, sum);
  PolyUOp *body = poly_uop_end(ctx, store, idxs, 1);
  PolyUOp *sink = poly_uop_sink(ctx, &body, 1);
  PolyUOp *args[3] = {c, a, b};
  PolyUOp *call = poly_uop_call(ctx, sink, args, 3);
  PolyUOp *out = poly_uop_after(ctx, c, call);

  float a_d[] = {1, 2, 3, 4};
  float b_d[] = {10, 20, 30, 40};
  float c_d[] = {0, 0, 0, 0};
  ASSERT_INT_EQ(poly_buffer_write(ctx, c, c_d, sizeof(c_d)), 0);
  ASSERT_INT_EQ(poly_buffer_write(ctx, a, a_d, sizeof(a_d)), 0);
  ASSERT_INT_EQ(poly_buffer_write(ctx, b, b_d, sizeof(b_d)), 0);

  PolyUOp *realized = NULL;
  ASSERT_INT_EQ(poly_realize_uops(ctx, &out, 1, &realized), 0);
  ASSERT_PTR_EQ(realized, c);
  ASSERT_INT_EQ(poly_buffer_read(ctx, c, c_d, sizeof(c_d)), 0);
  for (int j = 0; j < N; j++)
    ASSERT_FLOAT_EQ(c_d[j], a_d[j] + b_d[j], 1e-6);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sched, custom_kernel_call_replays_after_input_mutation) {
  int N = 4;
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *c = poly_buffer_f32(ctx, N);
  PolyUOp *a = poly_buffer_f32(ctx, N);
  PolyUOp *b = poly_buffer_f32(ctx, N);

  PolyUOp *pc = poly_uop_flatten(ctx, poly_uop_placeholder_like(ctx, c, 0));
  PolyUOp *pa = poly_uop_flatten(ctx, poly_uop_placeholder_like(ctx, a, 1));
  PolyUOp *pb = poly_uop_flatten(ctx, poly_uop_placeholder_like(ctx, b, 2));
  PolyUOp *i = poly_uop_range(ctx, N, 0, POLY_AXIS_LOOP);
  PolyUOp *idxs[1] = {i};
  PolyUOp *ci = poly_uop_index(ctx, pc, idxs, 1, 0);
  PolyUOp *ai = poly_uop_index(ctx, pa, idxs, 1, 0);
  PolyUOp *bi = poly_uop_index(ctx, pb, idxs, 1, 0);
  PolyUOp *sum = poly_alu2(ctx, POLY_OP_ADD, ai, bi);
  PolyUOp *store = poly_uop_store(ctx, ci, sum);
  PolyUOp *body = poly_uop_end(ctx, store, idxs, 1);
  PolyUOp *sink = poly_uop_sink(ctx, &body, 1);
  PolyUOp *args[3] = {c, a, b};
  PolyUOp *call = poly_uop_call(ctx, sink, args, 3);
  PolyUOp *out = poly_uop_after(ctx, c, call);

  float b_d[] = {10, 20, 30, 40};
  float c_d[] = {0, 0, 0, 0};
  ASSERT_INT_EQ(poly_buffer_write(ctx, b, b_d, sizeof(b_d)), 0);

  float runs[][4] = {{1, 2, 3, 4}, {5, 6, 7, 8}};
  float expected[][4] = {{11, 22, 33, 44}, {15, 26, 37, 48}};
  for (int pass = 0; pass < 2; pass++) {
    ASSERT_INT_EQ(poly_buffer_write(ctx, a, runs[pass], sizeof(runs[pass])), 0);
    ASSERT_INT_EQ(poly_buffer_write(ctx, c, c_d, sizeof(c_d)), 0);
    PolyUOp *realized = NULL;
    ASSERT_INT_EQ(poly_realize_uops(ctx, &out, 1, &realized), 0);
    ASSERT_PTR_EQ(realized, c);
    ASSERT_INT_EQ(poly_buffer_read(ctx, c, c_d, sizeof(c_d)), 0);
    for (int j = 0; j < N; j++)
      ASSERT_FLOAT_EQ(c_d[j], expected[pass][j], 1e-6);
  }

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sched, realize_reduce_sum) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_buffer_f32(ctx, 8);
  int64_t ax[] = {0};
  PolyUOp *s = poly_reduce_axis(ctx, POLY_OP_ADD, x, ax, 1);
  PolyUOp *out = poly_buffer_f32(ctx, 1);
  PolyUOp *store = poly_store_val(ctx, out, s);
  PolyUOp *sink = poly_sink1(ctx, store);

  float x_d[] = {1, 2, 3, 4, 5, 6, 7, 8};
  float out_d[] = {0};

  PolyTestBufferView bindings[] = {POLY_TEST_HOST_VIEW(x, x_d), POLY_TEST_HOST_VIEW(out, out_d)};
  int ret = poly_test_realize_buffer_views(ctx, sink, bindings, 2);
  ASSERT_INT_EQ(ret, 0);
  ASSERT_FLOAT_EQ(out_d[0], 36.0f, 1e-5);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sched, realize_grad_chain) {
  int N = 4;
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_buffer_f32(ctx, N);
  PolyUOp *sq = poly_alu2(ctx, POLY_OP_MUL, x, x);
  int64_t ax[] = {0};
  PolyUOp *loss = poly_reduce_axis(ctx, POLY_OP_ADD, sq, ax, 1);
  PolyUOp *gx = poly_grad(ctx, loss, x);
  ASSERT_NOT_NULL(gx);

  PolyUOp *out = poly_buffer_f32(ctx, N);
  PolyUOp *store = poly_store_val(ctx, out, gx);
  PolyUOp *sink = poly_sink1(ctx, store);

  float x_d[] = {1, 2, 3, 4};
  float gx_d[] = {0, 0, 0, 0};

  PolyTestBufferView bindings[] = {POLY_TEST_HOST_VIEW(x, x_d), POLY_TEST_HOST_VIEW(out, gx_d)};
  int ret = poly_test_realize_buffer_views(ctx, sink, bindings, 2);
  ASSERT_INT_EQ(ret, 0);
  for (int i = 0; i < N; i++)
    ASSERT_FLOAT_EQ(gx_d[i], 2.0f * x_d[i], 1e-5);

  poly_ctx_destroy(ctx);
  PASS();
}
