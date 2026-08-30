/*
 * test_schedule_engine.c — Tests for the engine/schedule single-kernel path
 */

#include "test_harness.h"
#include "../src/engine/schedule.h"
#include "../src/engine/realize.h"
#include "../src/codegen/codegen.h"
#include "../src/schedule/rangeify.h"
#include "../src/frontend.h"
#include "../src/device.h"

/* Helper: build tensor-level graph, schedule, verify structure */

/* These scheduler tests linearize executable single-kernel roots. The public
 * poly_get_kernel_graph() boundary now stops earlier at the pre-codegen kernel
 * graph, so the tests map their local kernel lookups through the engine
 * schedule and pull back the one scheduled root explicitly. */
static PolyUOp *single_scheduled_root(PolyCtx *ctx, PolyUOp *sink) {
  PolyUOp *linear = poly_test_create_linear(ctx, sink);
  return linear && linear->n_src > 0
             ? poly_test_linear_call_body(linear, linear->n_src - 1)
             : NULL;
}

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

static int count_param_indices(PolyCtx *ctx, PolyUOp *root) {
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, root, &n_topo);
  int count = 0;
  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    if (!u || u->op != POLY_OP_INDEX || u->n_src < 1 || !u->src[0] ||
        u->src[0]->op != POLY_OP_PARAM)
      continue;
    count++;
  }
  return count;
}

/* IR structure tests */

TEST(sched, vecadd_ir) {
  /* c = a + b (1D, 10 elements): verify kernel IR structure */
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_test_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *b = poly_test_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *c = poly_test_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, a, b, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, c, add, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyUOp *kernel = single_scheduled_root(ctx, sink);
  ASSERT_NOT_NULL(kernel);
  ASSERT_EQ(kernel->op, POLY_OP_SINK);

  /* Linearize and check structure */
  int n;
  PolyUOp **lin = poly_test_full_rewrite_and_linearize(ctx, kernel, &n);
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

  PolyUOp *a = poly_test_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *b = poly_test_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *c = poly_test_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, a, b, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, c, add, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyUOp *kernel = single_scheduled_root(ctx, sink);
  ASSERT_NOT_NULL(kernel);

  int n_lin;
  PolyUOp **lin = poly_test_full_rewrite_and_linearize(ctx, kernel, &n_lin);
  char *src = poly_render_c(ctx, lin, n_lin, "vecadd");
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

TEST(sched, direct_sink_store_uses_compute_call_like_tinygrad) {
  int N = 8;
  float src_d[8], dst_d[8];
  for (int i = 0; i < N; i++) {
    src_d[i] = (float)(i + 1) * 1.25f;
    dst_d[i] = 0.0f;
  }

  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *src = poly_test_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *dst = poly_test_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, dst, src));

  PolyUOp *linear = poly_test_create_linear(ctx, sink);
  ASSERT_NOT_NULL(linear);
  ASSERT_INT_EQ(linear->n_src, 1);
  ASSERT_FALSE(poly_test_linear_call_is_copy(linear, 0));
  PolyUOp *body = poly_test_linear_call_body(linear, 0);
  ASSERT_NOT_NULL(body);
  ASSERT_EQ(body->op, POLY_OP_SINK);
  ASSERT_INT_EQ(count_root_ops(ctx, body, POLY_OP_COPY), 0);
  ASSERT_INT_EQ(count_root_ops(ctx, body, POLY_OP_STORE), 1);
  ASSERT_INT_EQ(count_root_ops(ctx, body, POLY_OP_INDEX), 2);
  ASSERT_INT_EQ(poly_test_linear_call_n_buffers(linear, 0), 2);
  ASSERT_PTR_EQ(poly_test_linear_call_buffer(linear, 0, 0), dst);
  ASSERT_PTR_EQ(poly_test_linear_call_buffer(linear, 0, 1), src);

  poly_buffer_set(ctx, dst, dst_d, sizeof(dst_d), POLY_DEVICE_CPU);
  poly_buffer_set(ctx, src, src_d, sizeof(src_d), POLY_DEVICE_CPU);
  ASSERT_INT_EQ(poly_run_linear(ctx, linear, NULL, 0, NULL, 0, true, false, false), 0);
  ASSERT_INT_EQ(poly_buffer_read(ctx, dst, dst_d, sizeof(dst_d)), 0);
  for (int i = 0; i < N; i++)
    ASSERT_FLOAT_EQ(dst_d[i], src_d[i], 1e-6);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST_COMMON(sched, realize_uops_reads_ctx_buffers) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_test_buffer(ctx, POLY_FLOAT32, 4);
  PolyUOp *b = poly_test_buffer(ctx, POLY_FLOAT32, 4);
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

  PolyUOp *a = poly_test_buffer(ctx, POLY_FLOAT32, 4);
  PolyUOp *b = poly_test_buffer(ctx, POLY_FLOAT32, 4);
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

TEST(sched, from_host_copy_precedes_vector_compute) {
  PolyCtx *ctx = poly_ctx_new();

  float *data = malloc(3 * sizeof(float));
  ASSERT_NOT_NULL(data);
  data[0] = 1.0f;
  data[1] = 2.0f;
  data[2] = 3.0f;
  int64_t shape[] = {3};
  PolyTensor *a_source = poly_tensor_from_host(
      ctx, data, 3 * sizeof(float), POLY_FLOAT32, shape, 1
  );
  PolyTensor *a = poly_tensor_to_device(ctx, a_source, POLY_DEVICE_CPU);
  PolyTensor *one = poly_tensor_const_float_by_id(
      ctx, 1.0, poly_dtype_id_by_name("float32"), POLY_DEVICE_CPU
  );
  PolyTensor *x = poly_tensor_alu2(ctx, POLY_OP_ADD, a, one);
  ASSERT_NOT_NULL(a_source);
  ASSERT_NOT_NULL(a);
  ASSERT_NOT_NULL(one);
  ASSERT_NOT_NULL(x);
  PolyUOp *physical = poly_tensor_uop_physical(x);
  ASSERT_NOT_NULL(physical);

  PolyUOp *out = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 3, POLY_DEVICE_CPU);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, physical));
  PolyUOp *linear = poly_test_create_linear(ctx, sink);
  ASSERT_NOT_NULL(linear);
  ASSERT_INT_EQ(linear->n_src, 2);
  ASSERT_TRUE(poly_test_linear_call_is_copy(linear, 0));
  ASSERT_INT_EQ(poly_test_linear_call_body(linear, 0)->op, POLY_OP_COPY);
  ASSERT_INT_EQ(poly_test_linear_call_n_buffers(linear, 0), 2);
  ASSERT_INT_EQ(poly_run_linear(ctx, linear, NULL, 0, NULL, 0, true, false, false), 0);
  float got[3] = {0};
  ASSERT_INT_EQ(poly_buffer_read(ctx, out, got, sizeof(got)), 0);
  for (int i = 0; i < 3; i++) ASSERT_FLOAT_EQ(got[i], data[i] + 1.0f, 1e-6f);
  poly_ctx_destroy(ctx);
  free(data);
  PASS();
}

TEST(sched, placement_computed_copy_source_materializes_before_copy) {
  PolyCtx *ctx = poly_ctx_new();

  float data[3] = {1.0f, 2.0f, 3.0f};
  int64_t shape[1] = {3};
  int f32 = poly_dtype_id_by_name("float32");
  PolyTensor *a = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_CPU);
  PolyUOp *a_buffer =
      a ? (PolyUOp *)poly_uop_get_buffer_identity(poly_tensor_uop_physical(a)) : NULL;
  PolyTensor *two = poly_tensor_const_float_by_id(ctx, 2.0f, f32, POLY_DEVICE_CPU);
  PolyTensor *mt = poly_tensor_alu2(ctx, POLY_OP_MUL, a, two);
  ASSERT_NOT_NULL(a);
  ASSERT_NOT_NULL(a_buffer);
  poly_buffer_set(ctx, a_buffer, data, sizeof(data), POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(two);
  ASSERT_NOT_NULL(mt);
  PolyTensor *cuda_t = poly_tensor_to_device(ctx, mt, POLY_DEVICE_CUDA);
  ASSERT_NOT_NULL(cuda_t);

  PolyUOp *physical = poly_tensor_uop_physical(cuda_t);
  ASSERT_NOT_NULL(physical);
  ASSERT_INT_EQ(physical->op, POLY_OP_COPY);

  PolyUOp *out = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 3, POLY_DEVICE_CUDA);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, physical));
  PolyUOp *linear = poly_test_create_linear(ctx, sink);
  ASSERT_NOT_NULL(linear);

  int n_copy = 0;
  int n_compute = 0;
  for (int i = 0; i < linear->n_src; i++) {
    if (poly_test_linear_call_is_copy(linear, i)) {
      n_copy++;
      ASSERT_INT_EQ(poly_test_linear_call_body(linear, i)->op, POLY_OP_COPY);
      ASSERT_INT_EQ(poly_test_linear_call_n_buffers(linear, i), 2);
    } else {
      n_compute++;
      ASSERT_INT_EQ(count_root_ops(ctx, poly_test_linear_call_body(linear, i), POLY_OP_COPY), 0);
      ASSERT_TRUE(poly_test_linear_call_n_buffers(linear, i) >= 1);
      ASSERT_INT_EQ(
          poly_device_from_device_uop(poly_uop_device_uop_cached(
              ctx, poly_test_linear_call_buffer(linear, i, 0), NULL)),
          POLY_DEVICE_CPU
      );
    }
  }
  ASSERT_INT_EQ(n_compute, 1);
  ASSERT_INT_EQ(n_copy, 1);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sched, from_host_scalar_copy_precedes_compute_with_realized_peer) {
  PolyCtx *ctx = poly_ctx_new();
  float x_data[1] = {5.0f};
  float y_data[1] = {3.0f};
  int64_t shape[] = {1};

  PolyTensor *x = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_CPU);
  PolyUOp *x_buffer = x
                          ? (PolyUOp *)poly_uop_get_buffer_identity(
                                poly_tensor_uop_physical(x)
                            )
                          : NULL;
  PolyTensor *y_source = poly_tensor_from_host(
      ctx, y_data, sizeof(y_data), POLY_FLOAT32, shape, 1
  );
  PolyTensor *y = poly_tensor_to_device(ctx, y_source, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(x);
  ASSERT_NOT_NULL(x_buffer);
  ASSERT_NOT_NULL(y_source);
  ASSERT_NOT_NULL(y);
  poly_buffer_set(ctx, x_buffer, x_data, sizeof(x_data), POLY_DEVICE_CPU);

  PolyTensor *sub_tensor = poly_tensor_alu2(ctx, POLY_OP_SUB, x, y);
  ASSERT_NOT_NULL(sub_tensor);
  PolyUOp *sub = poly_tensor_uop_physical(sub_tensor);
  PolyUOp *out = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 1, POLY_DEVICE_CPU);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, sub));
  PolyUOp *linear = poly_test_create_linear(ctx, sink);
  ASSERT_NOT_NULL(linear);

  ASSERT_INT_EQ(linear->n_src, 2);
  ASSERT_TRUE(poly_test_linear_call_is_copy(linear, 0));
  ASSERT_INT_EQ(poly_test_linear_call_body(linear, 0)->op, POLY_OP_COPY);
  ASSERT_INT_EQ(poly_test_linear_call_n_buffers(linear, 0), 2);
  ASSERT_FALSE(poly_test_linear_call_is_copy(linear, 1));
  ASSERT_INT_EQ(poly_run_linear(ctx, linear, NULL, 0, NULL, 0, true, false, false), 0);
  float got = 0.0f;
  ASSERT_INT_EQ(poly_buffer_read(ctx, out, &got, sizeof(got)), 0);
  ASSERT_FLOAT_EQ(got, 2.0f, 1e-6f);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sched, max_reduce_tail_uses_value_typed_param_indices) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_test_buffer(ctx, POLY_FLOAT32, 6);
  PolyUOp *xr = poly_reshape(ctx, x, (int64_t[]){2, 3}, 2);
  PolyUOp *m = poly_reduce_axis(ctx, POLY_OP_MAX, xr, (int64_t[]){1}, 1);
  PolyUOp *loss = poly_reduce_axis(ctx, POLY_OP_ADD, m, (int64_t[]){0}, 1);
  PolyUOp *gx = poly_grad(ctx, loss, xr);
  PolyUOp *out = poly_test_buffer(ctx, POLY_FLOAT32, 6);
  PolyUOp *sink = poly_sink1(ctx, poly_test_store_to_buffer(ctx, out, gx));

  PolyUOp *linear = poly_test_create_linear(ctx, sink);
  ASSERT_NOT_NULL(linear);
  ASSERT_TRUE(linear->n_src >= 1);

  PolyUOp *tail = poly_test_linear_call_body(linear, linear->n_src - 1);
  ASSERT_NOT_NULL(tail);
  /* Current INDEX has the indexed value dtype; PARAM owns address metadata. */
  ASSERT_INT_EQ(count_param_indices(ctx, tail), 4);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST_COMMON(sched, chain_e2e) {
  /* d = (a + b) * c: verify single kernel with correct results */
  int N = 8;
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_test_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *b = poly_test_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *c = poly_test_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *d = poly_test_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, a, b, poly_arg_none());
  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, add, c, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, d, mul, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyUOp *kernel = single_scheduled_root(ctx, sink);
  ASSERT_NOT_NULL(kernel);

  int n_lin;
  PolyUOp **lin = poly_test_full_rewrite_and_linearize(ctx, kernel, &n_lin);
  char *src = poly_render_c(ctx, lin, n_lin, "chain");
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

  PolyUOp *a = poly_test_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *c = poly_test_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *two = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(2.0));
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, a, two, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, c, add, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyUOp *kernel = single_scheduled_root(ctx, sink);
  ASSERT_NOT_NULL(kernel);

  int n_lin;
  PolyUOp **lin = poly_test_full_rewrite_and_linearize(ctx, kernel, &n_lin);
  char *src = poly_render_c(ctx, lin, n_lin, "bcast");
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

  PolyUOp *a = poly_test_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *b = poly_test_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *neg = poly_uop1(ctx, POLY_OP_NEG, POLY_FLOAT32, a, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, b, neg, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyUOp *kernel = single_scheduled_root(ctx, sink);
  ASSERT_NOT_NULL(kernel);

  int n_lin;
  PolyUOp **lin = poly_test_full_rewrite_and_linearize(ctx, kernel, &n_lin);
  char *src = poly_render_c(ctx, lin, n_lin, "neg_k");
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

  PolyUOp *a = poly_test_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *b = poly_test_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *c = poly_test_buffer(ctx, POLY_FLOAT32, N);

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

  PolyUOp *kernel = single_scheduled_root(ctx, sink);
  ASSERT_NOT_NULL(kernel);

  int n_lin;
  PolyUOp **lin = poly_test_full_rewrite_and_linearize(ctx, kernel, &n_lin);
  char *src = poly_render_c(ctx, lin, n_lin, "add2d");
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

  PolyUOp *a_buf = poly_test_buffer(ctx, POLY_FLOAT32, N); /* 20 elements */
  PolyUOp *b_buf = poly_test_buffer(ctx, POLY_FLOAT32, 4); /* 4 elements */
  PolyUOp *c_buf = poly_test_buffer(ctx, POLY_FLOAT32, N); /* 20 elements */

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

  PolyUOp *kernel = single_scheduled_root(ctx, sink);
  ASSERT_NOT_NULL(kernel);

  int n_lin;
  PolyUOp **lin = poly_test_full_rewrite_and_linearize(ctx, kernel, &n_lin);
  char *src = poly_render_c(ctx, lin, n_lin, "bcast2d");
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

  PolyUOp *a_buf = poly_test_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *b_buf = poly_test_buffer(ctx, POLY_FLOAT32, N);

  /* Reshape a to (2,4), add 1.0, reshape back to (8) */
  int64_t dims[] = {2, 4};
  PolyUOp *a2d = poly_reshape(ctx, a_buf, dims, 2);
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(1.0));
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, a2d, one, poly_arg_none());
  int64_t flat[] = {8};
  PolyUOp *flat_result = poly_reshape(ctx, add, flat, 1);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, b_buf, flat_result, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyUOp *kernel = single_scheduled_root(ctx, sink);
  ASSERT_NOT_NULL(kernel);

  int n_lin;
  PolyUOp **lin = poly_test_full_rewrite_and_linearize(ctx, kernel, &n_lin);
  char *src = poly_render_c(ctx, lin, n_lin, "reshp");
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

  PolyUOp *a = poly_test_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *c = poly_test_buffer(ctx, POLY_FLOAT32, 1);
  int64_t axes[] = {0};
  PolyUOp *sum = poly_reduce_axis(ctx, POLY_OP_ADD, a, axes, 1);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, c, sum, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyUOp *kernel = single_scheduled_root(ctx, sink);
  ASSERT_NOT_NULL(kernel);

  int n;
  PolyUOp **lin = poly_test_full_rewrite_and_linearize(ctx, kernel, &n);
  ASSERT_TRUE(n > 0);

  /* Tinygrad parity on the optimized CPU path:
   * the scheduled root still has one RANGE, but full_rewrite_to_sink collapses
   * the 1D sum into scalarized LOAD/INDEX/ADD without BUFFER(REG)/RANGE/END in the
   * final linearized list. */
  ASSERT_INT_EQ(count_ops(lin, n, POLY_OP_PARAM), 2);
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

  PolyUOp *a = poly_test_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *c = poly_test_buffer(ctx, POLY_FLOAT32, 1);
  int64_t axes[] = {0};
  PolyUOp *sum = poly_reduce_axis(ctx, POLY_OP_ADD, a, axes, 1);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, c, sum, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyUOp *kernel = single_scheduled_root(ctx, sink);
  ASSERT_NOT_NULL(kernel);

  int n_lin;
  PolyUOp **lin = poly_test_full_rewrite_and_linearize(ctx, kernel, &n_lin);
  char *src = poly_render_c(ctx, lin, n_lin, "sum1d");
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

  PolyUOp *a = poly_test_buffer(ctx, POLY_FLOAT32, 12);
  PolyUOp *c = poly_test_buffer(ctx, POLY_FLOAT32, 3);
  int64_t rdims[] = {4, 3};
  PolyUOp *a2d = poly_reshape(ctx, a, rdims, 2);
  int64_t axes[] = {0};
  PolyUOp *sum = poly_reduce_axis(ctx, POLY_OP_ADD, a2d, axes, 1);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, c, sum, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyUOp *kernel = single_scheduled_root(ctx, sink);
  ASSERT_NOT_NULL(kernel);

  int n_lin;
  PolyUOp **lin = poly_test_full_rewrite_and_linearize(ctx, kernel, &n_lin);
  char *src = poly_render_c(ctx, lin, n_lin, "colsum");
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

  PolyUOp *a = poly_test_buffer(ctx, POLY_FLOAT32, 12);
  PolyUOp *c = poly_test_buffer(ctx, POLY_FLOAT32, 4);
  int64_t rdims[] = {4, 3};
  PolyUOp *a2d = poly_reshape(ctx, a, rdims, 2);
  int64_t axes[] = {1};
  PolyUOp *sum = poly_reduce_axis(ctx, POLY_OP_ADD, a2d, axes, 1);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, c, sum, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyUOp *kernel = single_scheduled_root(ctx, sink);
  ASSERT_NOT_NULL(kernel);

  int n_lin;
  PolyUOp **lin = poly_test_full_rewrite_and_linearize(ctx, kernel, &n_lin);
  char *src = poly_render_c(ctx, lin, n_lin, "rowsum");
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

  PolyUOp *a = poly_test_buffer(ctx, POLY_FLOAT32, 12);
  PolyUOp *c = poly_test_buffer(ctx, POLY_FLOAT32, 1);
  int64_t rdims[] = {4, 3};
  PolyUOp *a2d = poly_reshape(ctx, a, rdims, 2);
  int64_t axes[] = {0, 1};
  PolyUOp *sum = poly_reduce_axis(ctx, POLY_OP_ADD, a2d, axes, 2);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, c, sum, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyUOp *kernel = single_scheduled_root(ctx, sink);
  ASSERT_NOT_NULL(kernel);

  int n_lin;
  PolyUOp **lin = poly_test_full_rewrite_and_linearize(ctx, kernel, &n_lin);
  char *src = poly_render_c(ctx, lin, n_lin, "allsum");
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

  PolyUOp *a = poly_test_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *c = poly_test_buffer(ctx, POLY_FLOAT32, 1);
  int64_t axes[] = {0};
  PolyUOp *mx = poly_reduce_axis(ctx, POLY_OP_MAX, a, axes, 1);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, c, mx, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyUOp *kernel = single_scheduled_root(ctx, sink);
  ASSERT_NOT_NULL(kernel);

  int n_lin;
  PolyUOp **lin = poly_test_full_rewrite_and_linearize(ctx, kernel, &n_lin);
  char *src = poly_render_c(ctx, lin, n_lin, "maxred");
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

  PolyUOp *a = poly_test_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *b = poly_test_buffer(ctx, POLY_FLOAT32, N);

  int64_t axes[] = {0};
  PolyUOp *sum = poly_reduce_axis(ctx, POLY_OP_ADD, a, axes, 1);
  PolyUOp *sum_scalar = poly_reshape(ctx, sum, NULL, 0);
  ASSERT_PTR_EQ(sum_scalar, sum);
  PolyUOp *add = poly_add(ctx, sum_scalar, b);
  ASSERT_NOT_NULL(add);
  ASSERT_INT_EQ(count_root_ops(ctx, add, POLY_OP_RESHAPE), 0);
  ASSERT_INT_EQ(count_root_ops(ctx, add, POLY_OP_EXPAND), 0);

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
  PolyVarBinding *vars = NULL;
  int n_vars = 0;
  PolyUOp *linear = poly_linear_with_vars(ctx, targets, 1, realized, &vars, &n_vars);
  ASSERT_NOT_NULL(linear);
  ASSERT_NOT_NULL(realized[0]);
  /* Current tinygrad fuses a scalar reduction into its broadcast consumer:
   * one CALL with output LOOP and input REDUCE ranges. */
  ASSERT_INT_EQ(linear->n_src, 1);

  ASSERT_INT_EQ(count_root_ops(ctx, poly_test_linear_call_body(linear, 0), POLY_OP_STORE), 1);
  ASSERT_INT_EQ(count_root_ops(ctx, poly_test_linear_call_body(linear, 0), POLY_OP_REDUCE), 1);
  ASSERT_INT_EQ(count_root_ops(ctx, poly_test_linear_call_body(linear, 0), POLY_OP_RANGE), 2);
  ASSERT_INT_EQ(count_root_ops(ctx, poly_test_linear_call_body(linear, 0), POLY_OP_END), 1);
  ASSERT_INT_EQ(count_root_ops(ctx, poly_test_linear_call_body(linear, 0), POLY_OP_INDEX), 3);
  ASSERT_INT_EQ(count_root_ops(ctx, poly_test_linear_call_body(linear, 0), POLY_OP_ADD), 1);

  ASSERT_INT_EQ(poly_run_linear(ctx, linear, vars, n_vars, NULL, 0, true, false, false), 0);
  free(vars);

  const PolyUOp *out = poly_uop_get_buffer_identity(realized[0]);
  ASSERT_TRUE(out != NULL);
  float dout[8] = {0};
  ASSERT_INT_EQ(poly_buffer_read(ctx, (PolyUOp *)out, dout, sizeof(dout)), 0);

  for (int i = 0; i < N; i++)
    ASSERT_FLOAT_EQ(dout[i], sumv + b_d[i], 1e-5);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST_COMMON(sched, reduce_vector_chain_e2e) {
  /* tinygrad parity boundary: schedule_with_vars splits the reduced row sums
   * from the expanded consumer kernel. */
  int N = 12;
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_test_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *b = poly_test_buffer(ctx, POLY_FLOAT32, N);

  int64_t dims2d[] = {4, 3};
  PolyUOp *a2d = poly_reshape(ctx, a, dims2d, 2);
  PolyUOp *b2d = poly_reshape(ctx, b, dims2d, 2);

  int64_t axes[] = {1};
  PolyUOp *sum = poly_reduce_axis(ctx, POLY_OP_ADD, a2d, axes, 1); /* shape (4) */
  int64_t keepdim[] = {4, 1};
  PolyUOp *sum_keepdim = poly_reshape(ctx, sum, keepdim, 2);
  int64_t expd[] = {4, 3};
  PolyUOp *sum_exp = poly_expand(ctx, sum_keepdim, expd, 2); /* shape (4,3) */
  ASSERT_NOT_NULL(sum_exp);

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
  PolyVarBinding *vars = NULL;
  int n_vars = 0;
  PolyUOp *linear = poly_linear_with_vars(ctx, targets, 1, realized, &vars, &n_vars);
  ASSERT_NOT_NULL(linear);
  ASSERT_NOT_NULL(realized[0]);
  ASSERT_INT_EQ(linear->n_src, 2);

  ASSERT_INT_EQ(count_root_ops(ctx, poly_test_linear_call_body(linear, 0), POLY_OP_STORE), 1);
  ASSERT_INT_EQ(count_root_ops(ctx, poly_test_linear_call_body(linear, 0), POLY_OP_REDUCE), 1);
  ASSERT_INT_EQ(count_root_ops(ctx, poly_test_linear_call_body(linear, 0), POLY_OP_RANGE), 2);
  ASSERT_INT_EQ(count_root_ops(ctx, poly_test_linear_call_body(linear, 0), POLY_OP_END), 1);
  ASSERT_INT_EQ(count_root_ops(ctx, poly_test_linear_call_body(linear, 0), POLY_OP_INDEX), 2);
  ASSERT_INT_EQ(count_root_ops(ctx, poly_test_linear_call_body(linear, 0), POLY_OP_ADD), 1);

  ASSERT_INT_EQ(count_root_ops(ctx, poly_test_linear_call_body(linear, 1), POLY_OP_STORE), 1);
  ASSERT_INT_EQ(count_root_ops(ctx, poly_test_linear_call_body(linear, 1), POLY_OP_REDUCE), 0);
  ASSERT_INT_EQ(count_root_ops(ctx, poly_test_linear_call_body(linear, 1), POLY_OP_RANGE), 2);
  ASSERT_INT_EQ(count_root_ops(ctx, poly_test_linear_call_body(linear, 1), POLY_OP_END), 1);
  ASSERT_INT_EQ(count_root_ops(ctx, poly_test_linear_call_body(linear, 1), POLY_OP_INDEX), 3);
  ASSERT_INT_EQ(count_root_ops(ctx, poly_test_linear_call_body(linear, 1), POLY_OP_ADD), 2);

  ASSERT_INT_EQ(poly_run_linear(ctx, linear, vars, n_vars, NULL, 0, true, false, false), 0);
  free(vars);

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

  poly_ctx_destroy(ctx);
  PASS();
}

TEST_COMMON(sched, shared_scalar_reduce_two_stores_e2e) {
  /* Current tinygrad keeps the shared scalar REDUCE inline in both consumers,
   * producing one add CALL and one mul CALL. */
  int N = 8;
  PolyCtx *ctx = poly_ctx_new();
  PolyDevice device = poly_ctx_get_preferred_device(ctx);
  if (!poly_device_can_execute(device)) device = poly_device_default();

  PolyUOp *a = poly_test_buffer_on_device(ctx, POLY_FLOAT32, N, device);
  PolyUOp *c = poly_test_buffer_on_device(ctx, POLY_FLOAT32, N, device);
  PolyUOp *e = poly_test_buffer_on_device(ctx, POLY_FLOAT32, N, device);

  int64_t axes[] = {0};
  PolyUOp *sum = poly_reduce_axis(ctx, POLY_OP_ADD, a, axes, 1);
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, sum, c, poly_arg_none());
  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, sum, e, poly_arg_none());

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
  PolyVarBinding *vars = NULL;
  int n_vars = 0;
  PolyUOp *linear = poly_linear_with_vars(ctx, targets, 2, realized, &vars, &n_vars);
  ASSERT_NOT_NULL(linear);
  ASSERT_NOT_NULL(realized[0]);
  ASSERT_NOT_NULL(realized[1]);
  ASSERT_INT_EQ(linear->n_src, 2);

  for (int i = 0; i < 2; i++) {
    PolyUOp *body = poly_test_linear_call_body(linear, i);
    ASSERT_INT_EQ(count_root_ops(ctx, body, POLY_OP_STORE), 1);
    ASSERT_INT_EQ(count_root_ops(ctx, body, POLY_OP_REDUCE), 1);
    ASSERT_INT_EQ(count_root_ops(ctx, body, POLY_OP_RANGE), 2);
    ASSERT_INT_EQ(count_root_ops(ctx, body, POLY_OP_END), 1);
    ASSERT_INT_EQ(count_root_ops(ctx, body, POLY_OP_INDEX), 3);
  }
  ASSERT_INT_EQ(count_root_ops(ctx, poly_test_linear_call_body(linear, 0), POLY_OP_ADD), 1);
  ASSERT_INT_EQ(count_root_ops(ctx, poly_test_linear_call_body(linear, 0), POLY_OP_MUL), 0);
  ASSERT_INT_EQ(count_root_ops(ctx, poly_test_linear_call_body(linear, 1), POLY_OP_ADD), 0);
  ASSERT_INT_EQ(count_root_ops(ctx, poly_test_linear_call_body(linear, 1), POLY_OP_MUL), 1);

  ASSERT_INT_EQ(poly_run_linear(ctx, linear, vars, n_vars, NULL, 0, true, false, false), 0);
  free(vars);

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

  poly_ctx_destroy(ctx);
  PASS();
}

/* Movement op tests */

TEST(sched, permute_2d_e2e) {
  /* Transpose (3,4) → (4,3) */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_test_buffer(ctx, POLY_FLOAT32, 12);
  PolyUOp *c = poly_test_buffer(ctx, POLY_FLOAT32, 12);

  int64_t rdims[] = {3, 4};
  PolyUOp *a2d = poly_reshape(ctx, a, rdims, 2);
  int64_t perm[] = {1, 0};
  PolyUOp *t = poly_permute(ctx, a2d, perm, 2);
  PolyUOp *c2d = poly_reshape(ctx, c, (int64_t[]){4, 3}, 2);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, c2d, t, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyUOp *kernel = single_scheduled_root(ctx, sink);
  ASSERT_NOT_NULL(kernel);
  int n_lin;
  PolyUOp **lin = poly_test_full_rewrite_and_linearize(ctx, kernel, &n_lin);
  char *src = poly_render_c(ctx, lin, n_lin, "perm2d");
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
  PolyUOp *a = poly_test_buffer(ctx, POLY_FLOAT32, 24);
  PolyUOp *c = poly_test_buffer(ctx, POLY_FLOAT32, 24);

  int64_t rdims[] = {2, 3, 4};
  PolyUOp *a3d = poly_reshape(ctx, a, rdims, 3);
  int64_t perm[] = {2, 0, 1};
  PolyUOp *t = poly_permute(ctx, a3d, perm, 3);
  PolyUOp *c3d = poly_reshape(ctx, c, (int64_t[]){4, 2, 3}, 3);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, c3d, t, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyUOp *kernel = single_scheduled_root(ctx, sink);
  ASSERT_NOT_NULL(kernel);
  int n_lin;
  PolyUOp **lin = poly_test_full_rewrite_and_linearize(ctx, kernel, &n_lin);
  char *src = poly_render_c(ctx, lin, n_lin, "perm3d");
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
  PolyUOp *a = poly_test_buffer(ctx, POLY_FLOAT32, 8);
  PolyUOp *c = poly_test_buffer(ctx, POLY_FLOAT32, 3);

  int64_t pairs[][2] = {{2, 5}};
  PolyUOp *s = poly_shrink(ctx, a, pairs, 1);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, c, s, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyUOp *kernel = single_scheduled_root(ctx, sink);
  ASSERT_NOT_NULL(kernel);
  int n_lin;
  PolyUOp **lin = poly_test_full_rewrite_and_linearize(ctx, kernel, &n_lin);
  char *src = poly_render_c(ctx, lin, n_lin, "shrk1d");
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
  PolyUOp *a = poly_test_buffer(ctx, POLY_FLOAT32, 12);
  PolyUOp *c = poly_test_buffer(ctx, POLY_FLOAT32, 6);

  int64_t rdims[] = {4, 3};
  PolyUOp *a2d = poly_reshape(ctx, a, rdims, 2);
  int64_t pairs[][2] = {{1, 3}, {0, 3}};
  PolyUOp *s = poly_shrink(ctx, a2d, pairs, 2);
  PolyUOp *c2d = poly_reshape(ctx, c, (int64_t[]){2, 3}, 2);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, c2d, s, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyUOp *kernel = single_scheduled_root(ctx, sink);
  ASSERT_NOT_NULL(kernel);
  int n_lin;
  PolyUOp **lin = poly_test_full_rewrite_and_linearize(ctx, kernel, &n_lin);
  char *src = poly_render_c(ctx, lin, n_lin, "shrk2d");
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
  PolyUOp *a = poly_test_buffer(ctx, POLY_FLOAT32, 5);
  PolyUOp *c = poly_test_buffer(ctx, POLY_FLOAT32, 5);

  int64_t axes[] = {0};
  PolyUOp *f = poly_flip(ctx, a, axes, 1);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, c, f, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyUOp *kernel = single_scheduled_root(ctx, sink);
  ASSERT_NOT_NULL(kernel);
  int n_lin;
  PolyUOp **lin = poly_test_full_rewrite_and_linearize(ctx, kernel, &n_lin);
  char *src = poly_render_c(ctx, lin, n_lin, "flip1d");
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
  PolyUOp *a = poly_test_buffer(ctx, POLY_FLOAT32, 12);
  PolyUOp *c = poly_test_buffer(ctx, POLY_FLOAT32, 12);

  int64_t rdims[] = {3, 4};
  PolyUOp *a2d = poly_reshape(ctx, a, rdims, 2);
  int64_t axes[] = {0};
  PolyUOp *f = poly_flip(ctx, a2d, axes, 1);
  PolyUOp *c2d = poly_reshape(ctx, c, rdims, 2);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, c2d, f, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyUOp *kernel = single_scheduled_root(ctx, sink);
  ASSERT_NOT_NULL(kernel);
  int n_lin;
  PolyUOp **lin = poly_test_full_rewrite_and_linearize(ctx, kernel, &n_lin);
  char *src = poly_render_c(ctx, lin, n_lin, "flip2a");
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
  PolyUOp *a = poly_test_buffer(ctx, POLY_FLOAT32, 12);
  PolyUOp *c = poly_test_buffer(ctx, POLY_FLOAT32, 12);

  int64_t rdims[] = {3, 4};
  PolyUOp *a2d = poly_reshape(ctx, a, rdims, 2);
  int64_t axes[] = {0, 1};
  PolyUOp *f = poly_flip(ctx, a2d, axes, 2);
  PolyUOp *c2d = poly_reshape(ctx, c, rdims, 2);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, c2d, f, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyUOp *kernel = single_scheduled_root(ctx, sink);
  ASSERT_NOT_NULL(kernel);
  int n_lin;
  PolyUOp **lin = poly_test_full_rewrite_and_linearize(ctx, kernel, &n_lin);
  char *src = poly_render_c(ctx, lin, n_lin, "flip2b");
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
  PolyUOp *a = poly_test_buffer(ctx, POLY_FLOAT32, 3);
  PolyUOp *c = poly_test_buffer(ctx, POLY_FLOAT32, 7);

  int64_t pairs[][2] = {{2, 2}};
  PolyUOp *p = poly_pad(ctx, a, pairs, 1);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, c, p, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyUOp *kernel = single_scheduled_root(ctx, sink);
  ASSERT_NOT_NULL(kernel);
  int n_lin;
  PolyUOp **lin = poly_test_full_rewrite_and_linearize(ctx, kernel, &n_lin);
  char *src = poly_render_c(ctx, lin, n_lin, "pad1d");
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
  PolyUOp *a = poly_test_buffer(ctx, POLY_FLOAT32, 6);
  PolyUOp *c = poly_test_buffer(ctx, POLY_FLOAT32, 20);

  int64_t rdims[] = {2, 3};
  PolyUOp *a2d = poly_reshape(ctx, a, rdims, 2);
  int64_t pairs[][2] = {{1, 1}, {1, 1}};
  PolyUOp *p = poly_pad(ctx, a2d, pairs, 2);
  PolyUOp *c2d = poly_reshape(ctx, c, (int64_t[]){4, 5}, 2);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, c2d, p, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyUOp *kernel = single_scheduled_root(ctx, sink);
  ASSERT_NOT_NULL(kernel);
  int n_lin;
  PolyUOp **lin = poly_test_full_rewrite_and_linearize(ctx, kernel, &n_lin);
  char *src = poly_render_c(ctx, lin, n_lin, "pad2d");
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
  PolyUOp *a = poly_test_buffer(ctx, POLY_FLOAT32, 12);
  PolyUOp *c = poly_test_buffer(ctx, POLY_FLOAT32, 6);

  int64_t rdims[] = {3, 4};
  PolyUOp *a2d = poly_reshape(ctx, a, rdims, 2);
  int64_t perm[] = {1, 0};
  PolyUOp *t = poly_permute(ctx, a2d, perm, 2); /* (4,3) */
  int64_t pairs[][2] = {{0, 2}, {0, 3}};
  PolyUOp *s = poly_shrink(ctx, t, pairs, 2); /* (2,3) */
  PolyUOp *c2d = poly_reshape(ctx, c, (int64_t[]){2, 3}, 2);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, c2d, s, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyUOp *kernel = single_scheduled_root(ctx, sink);
  ASSERT_NOT_NULL(kernel);
  int n_lin;
  PolyUOp **lin = poly_test_full_rewrite_and_linearize(ctx, kernel, &n_lin);
  char *src = poly_render_c(ctx, lin, n_lin, "ps_ch");
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
  PolyUOp *a = poly_test_buffer(ctx, POLY_FLOAT32, 3);
  PolyUOp *c = poly_test_buffer(ctx, POLY_FLOAT32, 5);

  int64_t pad_pairs[][2] = {{1, 1}};
  PolyUOp *p = poly_pad(ctx, a, pad_pairs, 1); /* 5 elements */
  int64_t axes[] = {0};
  PolyUOp *f = poly_flip(ctx, p, axes, 1);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, c, f, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyUOp *kernel = single_scheduled_root(ctx, sink);
  ASSERT_NOT_NULL(kernel);
  int n_lin;
  PolyUOp **lin = poly_test_full_rewrite_and_linearize(ctx, kernel, &n_lin);
  char *src = poly_render_c(ctx, lin, n_lin, "pf_ch");
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
  PolyUOp *a = poly_test_buffer(ctx, POLY_FLOAT32, 4);
  PolyUOp *b = poly_test_buffer(ctx, POLY_FLOAT32, 4);
  PolyUOp *c = poly_test_buffer(ctx, POLY_FLOAT32, 4);

  int64_t axes[] = {0};
  PolyUOp *f = poly_flip(ctx, a, axes, 1);
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, f, b, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, c, add, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyUOp *kernel = single_scheduled_root(ctx, sink);
  ASSERT_NOT_NULL(kernel);
  int n_lin;
  PolyUOp **lin = poly_test_full_rewrite_and_linearize(ctx, kernel, &n_lin);
  char *src = poly_render_c(ctx, lin, n_lin, "mvalu");
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
  PolyUOp *ci = poly_uop_index(ctx, pc, idxs, 1);
  PolyUOp *ai = poly_uop_index(ctx, pa, idxs, 1);
  PolyUOp *bi = poly_uop_index(ctx, pb, idxs, 1);
  PolyUOp *sum = poly_alu2(ctx, POLY_OP_ADD, ai, bi);
  PolyUOp *store = poly_uop_store(ctx, ci, sum);
  PolyUOp *body = poly_uop_end(ctx, store, idxs, 1);
  PolyUOp *sink = poly_uop_sink_ex(ctx, &body, 1, "custom_add_4", 1);
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
  PolyUOp *ci = poly_uop_index(ctx, pc, idxs, 1);
  PolyUOp *ai = poly_uop_index(ctx, pa, idxs, 1);
  PolyUOp *bi = poly_uop_index(ctx, pb, idxs, 1);
  PolyUOp *sum = poly_alu2(ctx, POLY_OP_ADD, ai, bi);
  PolyUOp *store = poly_uop_store(ctx, ci, sum);
  PolyUOp *body = poly_uop_end(ctx, store, idxs, 1);
  PolyUOp *sink = poly_uop_sink_ex(ctx, &body, 1, "custom_add_replay_4", 1);
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

TEST(sched, custom_kernel_set_accumulator_noopt_executes) {
  const int candidates = 4;
  const int rows = 64;
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *out = poly_buffer_f32(ctx, candidates);
  PolyUOp *x = poly_buffer_f32(ctx, candidates * rows);

  PolyUOp *pout = poly_uop_flatten(ctx, poly_uop_placeholder_like(ctx, out, 0));
  PolyUOp *px = poly_uop_flatten(ctx, poly_uop_placeholder_like(ctx, x, 1));
  PolyUOp *c = poly_uop_range(ctx, candidates, 0, POLY_AXIS_LOOP);
  PolyUOp *r = poly_uop_range(ctx, rows, 1, POLY_AXIS_REDUCE);
  PolyUOp *out_idx[1] = {c};
  /* Pinned `c * 64` coerces the scalar through c.const_like, so the complete
   * pre-lowering expression remains weak-derived until pm_lower_index_dtype. */
  PolyUOp *rows_const = poly_const_typed(ctx, c->dtype, rows);
  PolyUOp *offset = poly_alu2(ctx, POLY_OP_ADD, poly_alu2(ctx, POLY_OP_MUL, c, rows_const), r);
  PolyUOp *x_idx[1] = {offset};
  ASSERT_TRUE(poly_dtype_eq(c->dtype, POLY_WEAKINT));
  ASSERT_TRUE(c->n_src == 1 && poly_dtype_eq(c->src[0]->dtype, POLY_WEAKINT));
  ASSERT_TRUE(poly_dtype_eq(rows_const->dtype, POLY_WEAKINT));
  ASSERT_TRUE(poly_dtype_eq(offset->dtype, POLY_WEAKINT));

  PolyUOp *acc = poly_uop_set(ctx, poly_uop_index(ctx, pout, out_idx, 1), poly_const_float(ctx, 0.0f), NULL, 0);
  ASSERT_NOT_NULL(acc);
  PolyUOp *acc_after_r = poly_uop_after(ctx, acc, r);
  PolyUOp *sum = poly_alu2(
      ctx, POLY_OP_ADD,
      poly_uop_index(ctx, acc_after_r, out_idx, 1),
      poly_uop_index(ctx, px, x_idx, 1)
  );
  acc = poly_uop_set(ctx, poly_uop_index(ctx, acc, out_idx, 1), sum, &r, 1);
  ASSERT_NOT_NULL(acc);
  PolyUOp *body = poly_uop_end(ctx, acc, &c, 1);
  PolyUOp *sink = poly_uop_sink_ex(ctx, &body, 1, "custom_sum_4_64", 0);
  PolyUOp *args[2] = {out, x};
  PolyUOp *call = poly_uop_call(ctx, sink, args, 2);
  PolyUOp *root = poly_uop_after(ctx, out, call);

  float x_data[candidates * rows];
  float expected[candidates];
  float out_data[candidates];
  for (int ci = 0; ci < candidates; ci++) {
    expected[ci] = 0.0f;
    out_data[ci] = 0.0f;
    for (int ri = 0; ri < rows; ri++) {
      float v = (float)(ci * rows + ri + 1);
      x_data[ci * rows + ri] = v;
      expected[ci] += v;
    }
  }
  ASSERT_INT_EQ(poly_buffer_write(ctx, x, x_data, sizeof(x_data)), 0);
  ASSERT_INT_EQ(poly_buffer_write(ctx, out, out_data, sizeof(out_data)), 0);

  PolyUOp *realized = NULL;
  ASSERT_INT_EQ(poly_realize_uops(ctx, &root, 1, &realized), 0);
  ASSERT_PTR_EQ(realized, out);
  ASSERT_INT_EQ(poly_buffer_read(ctx, out, out_data, sizeof(out_data)), 0);
  for (int ci = 0; ci < candidates; ci++)
    ASSERT_FLOAT_EQ(out_data[ci], expected[ci], 1e-4);

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

TEST(sched, estimates_match_tinygrad_edge_semantics) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  uint64_t ops = 0, lds = 0, mem = 0;
  PolyEstimates est = {0};

  PolyUOp *one = poly_const_float(ctx, 1.0);
  PolyUOp *two = poly_const_float(ctx, 2.0);
  PolyUOp *three = poly_const_float(ctx, 3.0);
  PolyUOp *add = poly_alu2(ctx, POLY_OP_ADD, one, two);
  PolyUOp *add_uops[] = {add};
  ASSERT_INT_EQ(poly_estimates_from_uops(ctx, add_uops, 1, false, &est), 0);
  ASSERT_INT_EQ(poly_estimates_infer(&est, NULL, 0, &ops, &lds, &mem), 0);
  ASSERT_TRUE(ops == 1 && lds == 0 && mem == 0);

  PolyUOp *mulacc = poly_alu3(ctx, POLY_OP_MULACC, one, two, three);
  PolyUOp *mulacc_uops[] = {mulacc};
  ASSERT_INT_EQ(poly_estimates_from_uops(ctx, mulacc_uops, 1, false, &est), 0);
  ASSERT_INT_EQ(poly_estimates_infer(&est, NULL, 0, &ops, &lds, &mem), 0);
  ASSERT_TRUE(ops == 2);

  /* Tinygrad 2026-08-22/a9069c177a9d renderer/__init__.py:54 counts
   * structural width through UOp.max_numel(), not a vector DType. */
  PolyUOp *va_src[] = {one, one, one, one};
  PolyUOp *vb_src[] = {two, two, two, two};
  PolyUOp *va = poly_uop_stack(ctx, va_src, 4);
  PolyUOp *vb = poly_uop_stack(ctx, vb_src, 4);
  PolyUOp *vadd = poly_alu2(ctx, POLY_OP_ADD, va, vb);
  PolyUOp *vadd_uops[] = {vadd};
  ASSERT_INT_EQ(poly_estimates_from_uops(ctx, vadd_uops, 1, false, &est), 0);
  ASSERT_INT_EQ(poly_estimates_infer(&est, NULL, 0, &ops, &lds, &mem), 0);
  ASSERT_TRUE(ops == 4);

  PolyUOp *four = poly_uop0(ctx, POLY_OP_CONST, POLY_INT64, poly_arg_int(4));
  PolyUOp *range4 = poly_uop1(
      ctx, POLY_OP_RANGE, POLY_INT64, four, poly_arg_range(0, POLY_AXIS_LOOP)
  );
  PolyUOp *range_end_src[] = {add, range4};
  PolyUOp *range_end =
      poly_uop(ctx, POLY_OP_END, POLY_VOID, range_end_src, 2, poly_arg_none());
  PolyUOp *range_uops[] = {range4, add, range_end};
  ASSERT_INT_EQ(poly_estimates_from_uops(ctx, range_uops, 3, false, &est), 0);
  ASSERT_INT_EQ(poly_estimates_infer(&est, NULL, 0, &ops, &lds, &mem), 0);
  ASSERT_TRUE(ops == 4);

  PolyUOp *eight = poly_uop0(ctx, POLY_OP_CONST, POLY_INT64, poly_arg_int(8));
  PolyUOp *special =
      poly_uop1(ctx, POLY_OP_SPECIAL, POLY_INT64, eight, poly_arg_str("gidx0"));
  PolyUOp *special_uops[] = {special, add};
  ASSERT_INT_EQ(poly_estimates_from_uops(ctx, special_uops, 2, false, &est), 0);
  ASSERT_INT_EQ(poly_estimates_infer(&est, NULL, 0, &ops, &lds, &mem), 0);
  ASSERT_TRUE(ops == 8);

  PolyUOp *param = poly_test_uop_param(ctx, POLY_FLOAT32, 8, 0, POLY_ADDR_GLOBAL);
  PolyUOp *idx_add = poly_alu2(ctx, POLY_OP_ADD, poly_const_int(ctx, 1), poly_const_int(ctx, 2));
  PolyUOp *index = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, param, idx_add, poly_arg_none());
  PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, index, poly_arg_none());
  PolyUOp *index_uops[] = {idx_add, index, load};
  ASSERT_INT_EQ(poly_estimates_from_uops(ctx, index_uops, 3, false, &est), 0);
  ASSERT_INT_EQ(poly_estimates_infer(&est, NULL, 0, &ops, &lds, &mem), 0);
  ASSERT_TRUE(ops == 1 && lds == 4 && mem == 4);
  ASSERT_INT_EQ(poly_estimates_from_uops(ctx, index_uops, 3, true, &est), 0);
  ASSERT_INT_EQ(poly_estimates_infer(&est, NULL, 0, &ops, &lds, &mem), 0);
  ASSERT_TRUE(ops == 0 && lds == 4 && mem == 4);

  PolyUOp *range8 = poly_uop1(
      ctx, POLY_OP_RANGE, POLY_INT64, eight, poly_arg_range(1, POLY_AXIS_LOOP)
  );
  PolyUOp *idx0 = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, param, range8, poly_arg_none());
  PolyUOp *idx1_expr = poly_alu2(ctx, POLY_OP_ADD, range8, poly_const_int(ctx, 0));
  PolyUOp *idx1 = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, param, idx1_expr, poly_arg_none());
  PolyUOp *load0 = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx0, poly_arg_none());
  PolyUOp *load1 = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx1, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx0, load0, poly_arg_none());
  PolyUOp *end_src[] = {store, range8};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *traffic_uops[] = {range8, idx0, load0, idx1_expr, idx1, load1, store, end};
  ASSERT_INT_EQ(poly_estimates_from_uops(ctx, traffic_uops, 8, true, &est), 0);
  ASSERT_INT_EQ(poly_estimates_infer(&est, NULL, 0, &ops, &lds, &mem), 0);
  ASSERT_TRUE(ops == 0 && lds == 96 && mem == 64);

  PolyUOp *n = poly_uop_variable(ctx, "n", 1, 8, POLY_WEAKINT, 1, false);
  PolyUOp *dynamic_range = poly_uop1(
      ctx, POLY_OP_RANGE, POLY_INT32, n, poly_arg_range(2, POLY_AXIS_LOOP)
  );
  PolyUOp *dynamic_end_src[] = {add, dynamic_range};
  PolyUOp *dynamic_end =
      poly_uop(ctx, POLY_OP_END, POLY_VOID, dynamic_end_src, 2, poly_arg_none());
  PolyUOp *dynamic_uops[] = {dynamic_range, add, dynamic_end};
  PolyVarBinding binding = {.var = n, .value = 4};
  ASSERT_INT_EQ(poly_estimates_from_uops(ctx, dynamic_uops, 3, false, &est), 0);
  ASSERT_INT_EQ(poly_estimates_infer(&est, &binding, 1, &ops, &lds, &mem), 0);
  ASSERT_TRUE(ops == 4);

  int tc_dims[3] = {16, 16, 16};
  PolyUOp *wmma = poly_uop(
      ctx, POLY_OP_WMMA, POLY_FLOAT32, (PolyUOp *[]){va, vb, va}, 3,
      poly_arg_tensor_core(tc_dims, POLY_FLOAT16, "AMD", 64, NULL, NULL, false)
  );
  PolyUOp *wmma_uops[] = {wmma};
  ASSERT_INT_EQ(poly_estimates_from_uops(ctx, wmma_uops, 1, false, &est), 0);
  ASSERT_INT_EQ(poly_estimates_infer(&est, NULL, 0, &ops, &lds, &mem), 0);
  ASSERT_TRUE(ops == 128);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sched, estimates_unbounded_range_does_not_scale_work) {
  /* Tinygrad 2026-08-22/a9069c177a9d renderer/__init__.py:40-46 treats
   * RANGE(void, NOOP) as an unbounded runtime loop with unknown trip count. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *noop = poly_uop0(ctx, POLY_OP_NOOP, POLY_VOID, poly_arg_none());
  PolyUOp *loop = poly_uop1(
      ctx, POLY_OP_RANGE, POLY_VOID, noop, poly_arg_range(0, POLY_AXIS_WEAK));
  PolyUOp *add = poly_alu2(ctx, POLY_OP_ADD, poly_const_float(ctx, 1.0),
                          poly_const_float(ctx, 2.0));
  PolyUOp *condition =
      poly_uop0(ctx, POLY_OP_CONST, POLY_BOOL, poly_arg_bool(true));
  PolyUOp *end_src[] = {add, loop, condition};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 3, poly_arg_none());
  PolyUOp *uops[] = {loop, add, end};

  PolyEstimates est = {0};
  uint64_t ops = 0, lds = 0, mem = 0;
  ASSERT_INT_EQ(poly_estimates_from_uops(ctx, uops, 3, false, &est), 0);
  ASSERT_INT_EQ(poly_estimates_infer(&est, NULL, 0, &ops, &lds, &mem), 0);
  ASSERT_TRUE(ops == 1 && lds == 0 && mem == 0);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sched, estimates_param_uses_uop_storage_extent) {
  /* Current UOp._shape reads PARAM storage extent through src[0].as_shape. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *extent = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(8));
  PolyParamArg arg = {.slot = 0, .dtype = POLY_FLOAT32, .addrspace = POLY_ADDR_GLOBAL};
  PolyUOp *param = poly_uop1(
      ctx, POLY_OP_PARAM, POLY_FLOAT32, extent, poly_arg_param(&arg));
  ASSERT_INT_EQ(param->n_src, 1);
  ASSERT_PTR_EQ(param->src[0], extent);
  ASSERT_INT_EQ(param->src[0]->op, POLY_OP_CONST);
  PolyAddrSpace addrspace = POLY_ADDR_ALU;
  ASSERT_TRUE(poly_uop_addrspace(param, &addrspace));
  ASSERT_INT_EQ(addrspace, POLY_ADDR_GLOBAL);
  PolyShape param_shape = poly_uop_max_shape_cached(ctx, param);
  ASSERT_INT_EQ(param_shape.ndim, 1);
  ASSERT_INT_EQ(param_shape.dims[0], 8);

  PolyUOp *sixteen = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(16));
  PolyUOp *range = poly_uop1(
      ctx, POLY_OP_RANGE, POLY_INT32, sixteen, poly_arg_range(0, POLY_AXIS_LOOP)
  );
  PolyUOp *index = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, param, range, poly_arg_none());
  PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, index, poly_arg_none());
  PolyUOp *end = poly_uop2(ctx, POLY_OP_END, POLY_VOID, load, range, poly_arg_none());
  PolyUOp *uops[] = {range, index, load, end};
  PolyEstimates est = {0};
  uint64_t ops = 0, lds = 0, mem = 0;
  ASSERT_INT_EQ(poly_estimates_from_uops(ctx, uops, 4, true, &est), 0);
  ASSERT_INT_EQ(poly_estimates_infer(&est, NULL, 0, &ops, &lds, &mem), 0);
  ASSERT_TRUE(ops == 0 && lds == 64 && mem == 32);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sched, estimate_inference_visits_shared_dag_once) {
  /* tinygrad uop/ops.py::UOp._sym_fxn simplifies then topologically renders
   * each unique symbolic UOp once. Repeated smin(accessed, cap) shares the
   * accessed subtree between the predicate and true arm; inference must stay
   * linear in unique UOps rather than recursive source occurrences. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *n = poly_uop_variable(ctx, "n", 0, 4, POLY_WEAKINT, 1, false);
  PolyUOp *expr = poly_uop0(ctx, POLY_OP_CONST, POLY_INT64, poly_arg_int(0));
  PolyUOp *cap = poly_uop0(ctx, POLY_OP_CONST, POLY_INT64, poly_arg_int(32));
  ASSERT_NOT_NULL(n);
  ASSERT_NOT_NULL(expr);
  ASSERT_NOT_NULL(cap);

  for (int i = 0; i < 80; i++) {
    PolyUOp *accessed = poly_uop2(ctx, POLY_OP_ADD, POLY_INT64, expr, n, poly_arg_none());
    PolyUOp *lt = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, accessed, cap, poly_arg_none());
    expr = poly_uop3(ctx, POLY_OP_WHERE, POLY_INT64, lt, accessed, cap, poly_arg_none());
  }
  ASSERT_INT_EQ(expr->op, POLY_OP_WHERE);

  PolyEstimates estimates = {.ops = expr, .lds = expr, .mem = expr};
  PolyVarBinding binding = {.var = n, .value = 1};
  uint64_t ops = 0, lds = 0, mem = 0;
  ASSERT_INT_EQ(poly_estimates_infer(&estimates, &binding, 1, &ops, &lds, &mem), 0);
  ASSERT_TRUE(ops == 32 && lds == 32 && mem == 32);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sched, estimate_inference_uses_unwrapped_host_symbolic_arithmetic) {
  /* Pinned tinygrad ops.py:1021-1035 renders sym_infer through host Python
   * arithmetic. int32 intermediates and CASTs are not width-wrapped. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *max =
      poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(INT32_MAX));
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(1));
  PolyUOp *sum =
      poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, max, one, poly_arg_none());
  PolyUOp *cast =
      poly_uop1(ctx, POLY_OP_CAST, POLY_INT32, sum, poly_arg_none());
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(0));
  PolyEstimates estimates = {.ops = sum, .lds = zero, .mem = zero};
  uint64_t ops = 0, lds = 0, mem = 0;

  ASSERT_INT_EQ(poly_estimates_infer(&estimates, NULL, 0, &ops, &lds, &mem), 0);
  ASSERT_TRUE(ops == UINT64_C(2147483648));
  ASSERT_TRUE(lds == 0 && mem == 0);

  estimates.ops = cast;
  ASSERT_INT_EQ(poly_estimates_infer(&estimates, NULL, 0, &ops, &lds, &mem), 0);
  ASSERT_TRUE(ops == UINT64_C(2147483648));

  poly_ctx_destroy(ctx);
  PASS();
}
