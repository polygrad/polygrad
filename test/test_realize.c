/*
 * test_realize.c — Tests for graph-driven poly_realize.
 *
 * Verifies that poly_realize materializes top-level value UOps through the
 * ctx->buffers side table with no external bindings array.
 */

#include "test_harness.h"
#include "../src/engine/realize.h"
#include "../src/device.h"
#include "../src/frontend.h"
#include "../src/polygrad.h"

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
  ASSERT_INT_EQ(poly_realize(ctx, targets, 1, realized), 0);
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
  ASSERT_INT_EQ(poly_realize(ctx, targets, 1, realized), 0);
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
  ASSERT_INT_EQ(sched->n_items, 1);

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
  ASSERT_INT_EQ(sched->n_items, 1);
  ASSERT_INT_EQ(count_root_ops(ctx, sched->items[0].root, POLY_OP_REDUCE), 1);

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
  ASSERT_INT_EQ(sched->n_items, 1);
  ASSERT_NOT_NULL(realized[0]);

  PolyUOp *root = sched->items[0].root;
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
  ASSERT_INT_EQ(sched->n_items, 1);
  ASSERT_NOT_NULL(realized[0]);

  PolyUOp *root = sched->items[0].root;
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
  ASSERT_INT_EQ(sched->n_items, 1);
  ASSERT_NOT_NULL(realized[0]);

  PolyUOp *root = sched->items[0].root;
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
  ASSERT_INT_EQ(sched->n_items, 1);
  ASSERT_NOT_NULL(realized[0]);

  PolyUOp *root = sched->items[0].root;
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
  ASSERT_INT_EQ(sched->n_items, 1);
  ASSERT_NOT_NULL(realized[0]);

  PolyUOp *root = sched->items[0].root;
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
  ASSERT_INT_EQ(sched->n_items, 1);
  ASSERT_NOT_NULL(realized[0]);

  PolyUOp *root = sched->items[0].root;
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
  ASSERT_INT_EQ(sched->n_items, 3);

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
  ASSERT_INT_EQ(sched->n_items, 3);

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
  PolyShape s = poly_uop_shape_cached(ctx, realized[0]);
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
  PolyUOp *assign = poly_assign(ctx, a, mul);

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

TEST(realize, schedule_with_vars_missing_buffer_errors) {
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
  ASSERT_INT_EQ(poly_run_schedule(ctx, sched, NULL, 0), -1);

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
  ASSERT_INT_EQ(sched->n_items, 1);
  ASSERT_NOT_NULL(realized[0]);

  /* tinygrad schedule_with_vars boundary: triu root still has WHERE and no
   * LOAD. Late load insertion happens later in codegen pm_add_loads. */
  ASSERT_INT_EQ(count_root_ops(ctx, sched->items[0].root, POLY_OP_LOAD), 0);
  ASSERT_TRUE(count_root_ops(ctx, sched->items[0].root, POLY_OP_WHERE) > 0);

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
  ASSERT_INT_EQ(sched->n_items, 1);
  ASSERT_NOT_NULL(realized[0]);

  ASSERT_INT_EQ(count_root_ops(ctx, sched->items[0].root, POLY_OP_LOAD), 0);
  ASSERT_TRUE(count_root_ops(ctx, sched->items[0].root, POLY_OP_WHERE) > 0);

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

TEST(realize, graph_missing_buffer_errors) {
  /* Missing leaf data should still fail cleanly during schedule execution. */
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer_f32(ctx, 4);
  PolyUOp *b = poly_buffer_f32(ctx, 4);
  PolyUOp *add = poly_alu2(ctx, POLY_OP_ADD, a, b);

  /* Attach only a, leave b unattached. */
  float da[] = {1.0f, 2.0f, 3.0f, 4.0f};
  poly_buffer_set(ctx, a, da, sizeof(da), POLY_DEVICE_CPU);

  PolyUOp *targets[] = {add};
  PolyUOp *realized[] = {NULL};
  ASSERT_INT_EQ(poly_realize(ctx, targets, 1, realized), -1);

  poly_ctx_destroy(ctx);
  PASS();
}
