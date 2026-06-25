/*
 * test_realize.c — Tests for graph-driven poly_realize.
 *
 * Verifies that poly_realize materializes top-level value UOps through the
 * ctx->buffers side table with no external bindings array.
 */

#include "test_harness.h"
#include "../src/ctx.h"
#include "../src/engine/realize.h"
#include "../src/device.h"
#include "../src/frontend.h"
#include "../src/polygrad.h"
#include "../src/tensor.h"

static PolyBuffer *realized_buffer(PolyCtx *ctx, PolyUOp *realized) {
  const PolyUOp *buf_uop = poly_uop_get_buffer_identity(realized);
  return buf_uop ? poly_buffer_get(ctx, (PolyUOp *)buf_uop) : NULL;
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
  ASSERT_PTR_EQ(poly_tensor_uop(cuda_tensor), x);
  ASSERT_INT_EQ(cuda_tensor->role, POLY_TENSOR_PLACE);
  ASSERT_INT_EQ(poly_tensor_device(cuda_tensor), POLY_DEVICE_CUDA);

  PolyUOp *physical = poly_tensor_physicalize(ctx, cuda_tensor);
  ASSERT_NOT_NULL(physical);
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

  PolyUOp *y = poly_alu2(ctx, POLY_OP_ADD, poly_tensor_uop(x_cuda), poly_const_float(ctx, 2.0));
  PolyTensor *y_cuda = poly_tensor_create(ctx, y, POLY_TENSOR_VALUE, POLY_DEVICE_CUDA);
  ASSERT_NOT_NULL(y_cuda);

  PolyPlacementAudit audit;
  ASSERT_INT_EQ(poly_tensor_placement_audit(ctx, y_cuda, x, POLY_DEVICE_CUDA, &audit), 0);
  ASSERT_PTR_EQ(audit.selected, y_cuda);
  ASSERT_PTR_EQ(audit.selected_current, y);
  ASSERT_PTR_EQ(audit.selected_logical, y);
  ASSERT_TRUE(audit.selected_physical == NULL);
  ASSERT_INT_EQ(audit.selected_role, POLY_TENSOR_VALUE);
  ASSERT_INT_EQ(audit.selected_device, POLY_DEVICE_CUDA);
  ASSERT_PTR_EQ(audit.query_current, x);
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

  PolyUOp *y_expr =
      poly_alu2(ctx, POLY_OP_ADD, poly_tensor_uop(x_cpu_again), poly_const_float(ctx, 1.0f));
  PolyTensor *y_cpu = poly_tensor_create(ctx, y_expr, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(y_cpu);

  PolyPlacementAudit audit;
  ASSERT_INT_EQ(poly_tensor_placement_audit(ctx, y_cpu, x_buf, POLY_DEVICE_CPU, &audit), 0);
  ASSERT_PTR_EQ(audit.selected, y_cpu);
  ASSERT_PTR_EQ(audit.query_current, x_buf);
  ASSERT_PTR_EQ(audit.place_fact, x_cpu_again);
  ASSERT_PTR_EQ(audit.value_fact, x_cpu);
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

  PolyUOp *physical = poly_tensor_physicalize(ctx, x_cpu_again);
  ASSERT_NOT_NULL(physical);
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
  ASSERT_FLOAT_EQ(((float *)out->ptr)[0], 5.0f, 1e-5f);
  ASSERT_FLOAT_EQ(((float *)out->ptr)[1], 6.0f, 1e-5f);
  ASSERT_FLOAT_EQ(((float *)out->ptr)[2], 7.0f, 1e-5f);
  ASSERT_FLOAT_EQ(da[0], 1.0f, 1e-5f);
  ASSERT_FLOAT_EQ(da[1], 2.0f, 1e-5f);
  ASSERT_FLOAT_EQ(da[2], 3.0f, 1e-5f);

  poly_ctx_destroy(ctx);
  free(da);
  free(dv);
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

TEST(realize, tensor_realize_cpu_e2e) {
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
  float *out = (float *)buf->ptr;
  ASSERT_NOT_NULL(out);
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
  float *out = (float *)buf->ptr;
  ASSERT_NOT_NULL(out);
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
  float *out = (float *)buf->ptr;
  ASSERT_NOT_NULL(out);
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
  ASSERT_FLOAT_EQ(((float *)x_storage->ptr)[0], 5.0f, 1e-5f);

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
  ASSERT_FLOAT_EQ(((float *)z_storage->ptr)[0], 3.0f, 1e-5f);

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
  ASSERT_FLOAT_EQ(((float *)ar_storage->ptr)[0], 2.0f, 1e-5f);

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
  ASSERT_FLOAT_EQ(((float *)b_storage->ptr)[0], 3.0f, 1e-5f);

  poly_ctx_destroy(ctx);
  free(da);
  free(dten);
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
  ASSERT_FLOAT_EQ(((float *)storage->ptr)[0], 5.0f, 1e-5f);

  poly_ctx_destroy(ctx);
  free(da);
  free(dv);
  PASS();
}

TEST(realize, graph_vecadd) {
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
  float *dout = (float *)buf->ptr;
  ASSERT_TRUE(dout != NULL);
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
  ASSERT_TRUE(sched == NULL);
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
  PolySchedule *sched = poly_schedule_with_vars(ctx, contig_targets, 1, contig_realized);
  ASSERT_TRUE(sched == NULL);
  ASSERT_PTR_EQ(contig_realized[0], contig);

  poly_ctx_destroy(ctx);
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

TEST(realize, transform_to_call_many_reduce_view_sources_materialize_all) {
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
  ASSERT_TRUE(big_call->src[0]->n_src >= N_TERMS + 1);

  PolyUOp *final_store = big_call->src[0]->src[big_call->src[0]->n_src - 1];
  ASSERT_NOT_NULL(final_store);
  ASSERT_INT_EQ(final_store->op, POLY_OP_STORE);
  ASSERT_TRUE(final_store->n_src >= 2);
  ASSERT_INT_EQ(count_root_ops(ctx, final_store->src[1], POLY_OP_REDUCE_AXIS), 0);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, transform_to_call_reduce_dependency_scan_rewinds_scratch) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *buf = poly_buffer_f32(ctx, 4);
  PolyUOp *value = poly_alu2(ctx, POLY_OP_ADD, buf, poly_const_float(ctx, 1.0f));
  int64_t axis[] = {0};
  PolyUOp *reduced = poly_reduce_axis(ctx, POLY_OP_ADD, value, axis, 1);
  PolyUOp *view = poly_reshape(ctx, reduced, (int64_t[]){1}, 1);
  PolyUOp *target = poly_alu2(ctx, POLY_OP_ADD, view, poly_const_float(ctx, 2.0f));

  size_t scratch_before = poly_arena_used(ctx->scratch);
  PolyUOp *targets[] = {target};
  PolyUOp *realized[] = {NULL};
  PolyUOp *big_call = poly_transform_to_call(ctx, targets, 1, realized);
  ASSERT_NOT_NULL(big_call);
  ASSERT_NOT_NULL(realized[0]);
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
  float *dout = (float *)buf->ptr;
  ASSERT_NOT_NULL(dout);
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

  PolySchedule *sched = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_NOT_NULL(sched);
  ASSERT_INT_EQ(poly_run_schedule(ctx, sched, NULL, 0), 0);

  for (int i = 0; i < 4; i++)
    ASSERT_FLOAT_EQ(out_data[i], (float)(i + 2), 1e-5f);
  ASSERT_FLOAT_EQ(out_data[4], -999.0f, 1e-5f);

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

  PolySchedule *sched = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
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
  ASSERT_NOT_NULL(buf->ptr);
  ASSERT_FLOAT_EQ(((float *)buf->ptr)[0], 10.0f, 1e-5f);

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
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_BUFFERIZE), 0);

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
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_BUFFERIZE), 0);
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
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_BUFFERIZE), 0);
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
  ASSERT_INT_EQ(count_root_ops(ctx, root, POLY_OP_BUFFERIZE), 0);
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
  float *dout = (float *)buf->ptr;
  ASSERT_NOT_NULL(dout);
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
  float *add_out = (float *)add_buf->ptr;
  float *mul_out = (float *)mul_buf->ptr;
  ASSERT_NOT_NULL(add_out);
  ASSERT_NOT_NULL(mul_out);
  ASSERT_FLOAT_EQ(add_out[0], 11.0f, 1e-5f);
  ASSERT_FLOAT_EQ(add_out[3], 44.0f, 1e-5f);
  ASSERT_FLOAT_EQ(mul_out[0], 10.0f, 1e-5f);
  ASSERT_FLOAT_EQ(mul_out[3], 160.0f, 1e-5f);

  poly_schedule_free(sched);
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
  float *dout = (float *)buf->ptr;
  ASSERT_NOT_NULL(dout);
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
  float *dout = (float *)buf->ptr;
  ASSERT_NOT_NULL(dout);
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
  float *dout = (float *)buf->ptr;
  ASSERT_NOT_NULL(dout);
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

  PolyUOp *expr = poly_alu2(ctx, POLY_OP_ADD, poly_tensor_uop(a), poly_const_float(ctx, 1.0));
  PolyTensor *out = poly_tensor_create(ctx, expr, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(out);
  PolyTensor *realized = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &out, 1, &realized), 0);
  ASSERT_PTR_EQ(realized, out);
  ASSERT_INT_EQ(poly_jit_end_capture(jit), 0);
  ASSERT_TRUE(poly_jit_is_captured(jit));
  ASSERT_INT_EQ(poly_jit_schedule_count(jit), 1);

  float first[3] = {0};
  ASSERT_INT_EQ(
      poly_buffer_read(ctx, (PolyUOp *)poly_uop_get_buffer_identity(poly_tensor_uop(out)), first, sizeof(first)),
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

  ASSERT_INT_EQ(poly_jit_run(jit, &b, 1), 0);

  float second[3] = {0};
  ASSERT_INT_EQ(
      poly_buffer_read(ctx, (PolyUOp *)poly_uop_get_buffer_identity(poly_tensor_uop(out)), second, sizeof(second)),
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
      ctx,
      poly_alu2(ctx, POLY_OP_ADD, poly_tensor_uop(a), poly_const_float(ctx, 1.0)),
      POLY_TENSOR_VALUE,
      POLY_DEVICE_CPU
  );
  ASSERT_NOT_NULL(y);
  PolyTensor *realized_y = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &y, 1, &realized_y), 0);

  PolyTensor *z = poly_tensor_create(
      ctx,
      poly_alu2(ctx, POLY_OP_MUL, poly_tensor_uop(a), poly_const_float(ctx, 2.0)),
      POLY_TENSOR_VALUE,
      POLY_DEVICE_CPU
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
      poly_buffer_read(ctx, (PolyUOp *)poly_uop_get_buffer_identity(poly_tensor_uop(y)), y_out, sizeof(y_out)),
      0
  );
  ASSERT_FLOAT_EQ(y_out[0], 11.0f, 1e-5f);
  ASSERT_FLOAT_EQ(y_out[1], 21.0f, 1e-5f);
  ASSERT_FLOAT_EQ(y_out[2], 31.0f, 1e-5f);

  float z_out[3] = {0};
  ASSERT_INT_EQ(
      poly_buffer_read(ctx, (PolyUOp *)poly_uop_get_buffer_identity(poly_tensor_uop(z)), z_out, sizeof(z_out)),
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
      ctx,
      poly_alu2(ctx, POLY_OP_ADD, poly_tensor_uop(a), poly_const_float(ctx, 1.0)),
      POLY_TENSOR_VALUE,
      POLY_DEVICE_CPU
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
      poly_buffer_read(ctx, (PolyUOp *)poly_uop_get_buffer_identity(poly_tensor_uop(out)), out_data, sizeof(out_data)),
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
      ctx,
      poly_alu2(ctx, POLY_OP_ADD, poly_tensor_uop(a), poly_const_float(ctx, 1.0)),
      POLY_TENSOR_VALUE,
      POLY_DEVICE_CPU
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
      poly_buffer_read(ctx, (PolyUOp *)poly_uop_get_buffer_identity(poly_tensor_uop(out)), out_data, sizeof(out_data)),
      0
  );
  ASSERT_FLOAT_EQ(out_data[0], 11.0f, 1e-5f);
  ASSERT_FLOAT_EQ(out_data[1], 21.0f, 1e-5f);
  ASSERT_FLOAT_EQ(out_data[2], 31.0f, 1e-5f);

  poly_jit_free(jit);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(realize, poly_jit_replays_with_runtime_var_override) {
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
  PolyVarBinding bind_6 = {.var = N, .value = 6};
  ASSERT_INT_EQ(poly_jit_run_with_vars(jit, &a, 1, &bind_6, 1), 0);

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
      ctx,
      poly_alu2(ctx, POLY_OP_ADD, poly_tensor_uop(a), poly_const_float(ctx, 1.0)),
      POLY_TENSOR_VALUE,
      POLY_DEVICE_CPU
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
