/*
 * test_tensor.c -- Tests for shape-on-UOp and v2 composed ops
 *
 * Reference values verified against tinygrad (conda env 'tiny').
 * All tests use the v2 API (shape read from UOp) or raw UOp construction.
 * No PolyExpr dependency.
 */

#include <math.h>
#include <stdlib.h>
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
#include "../src/optim.h"
#include "../src/tensor.h"
#include "../src/codegen/codegen.h"

TEST(tensor, execution_scalar_reduction_axes) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_const_typed(ctx, POLY_FLOAT32, 2.0);
  bool valid = true;
  for (int axis = -1; axis <= 0; axis++) {
    PolyUOp *sum = poly_sum_reduce(ctx, x, axis, 1);
    PolyUOp *soft = poly_softmax(ctx, x, axis);
    PolyUOp *logsoft = poly_log_softmax(ctx, x, axis);
    valid &= sum && soft && logsoft;
    if (sum && soft && logsoft)
      valid &= poly_uop_ndim(ctx, sum) == 0 && poly_uop_ndim(ctx, soft) == 0 &&
               poly_uop_ndim(ctx, logsoft) == 0;
  }
  valid &=
      !poly_sum_reduce(ctx, x, 1, 0) && !poly_softmax(ctx, x, 1) && !poly_log_softmax(ctx, x, 1);
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(valid);
  PASS();
}

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
  PolyUOp *store = poly_test_store_to_buffer(ctx, out_buf, val);
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
  for (int i = 0; i < n_topo; i++)
    count += topo[i]->op == op;
  poly_toposort_free(topo);
  return count;
}

static int uop_topo_index(PolyUOp **topo, int n, PolyUOp *needle) {
  for (int i = 0; i < n; i++)
    if (topo[i] == needle) return i;
  return -1;
}

static bool uop_graph_isomorphic(PolyCtx *a_ctx, PolyUOp *a, PolyCtx *b_ctx, PolyUOp *b) {
  int a_n = 0, b_n = 0;
  PolyUOp **a_topo = poly_toposort_alloc(a_ctx, a, &a_n);
  PolyUOp **b_topo = poly_toposort_alloc(b_ctx, b, &b_n);
  bool equal = a_topo && b_topo && a_n == b_n;
  for (int i = 0; equal && i < a_n; i++) {
    PolyUOp *x = a_topo[i], *y = b_topo[i];
    equal = x->op == y->op && poly_dtype_eq(x->dtype, y->dtype) && x->n_src == y->n_src &&
            poly_arg_eq(x->arg, y->arg) && x->tag == y->tag && poly_arg_eq(x->tag_arg, y->tag_arg);
    for (int j = 0; equal && j < x->n_src; j++)
      equal = uop_topo_index(a_topo, a_n, x->src[j]) == uop_topo_index(b_topo, b_n, y->src[j]);
  }
  poly_toposort_free(a_topo);
  poly_toposort_free(b_topo);
  return equal;
}

static int read_tensor_bytes(PolyCtx *ctx, PolyTensor *tensor, void *out, size_t nbytes) {
  PolyTensor *realized = NULL;
  if (!ctx || !tensor || !out || poly_realize_tensors(ctx, &tensor, 1, &realized) != 0 || !realized)
    return -1;
  const PolyUOp *buffer = poly_uop_get_buffer_identity(realized->uop_physical);
  return buffer ? poly_buffer_read(ctx, (PolyUOp *)buffer, out, nbytes) : -1;
}

static int read_tensor_f32(PolyCtx *ctx, PolyTensor *tensor, float *out, size_t n);

TEST(tensor, released_handles_retire_exact_buffer_residency) {
  /* Tinygrad 2026-08-22 UOp/Buffer destruction drops each allocation when
   * its final Tensor/UOp owner dies.  Polygrad exposes that lifetime
   * explicitly for C callers and collects only at a context safe point. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  int64_t shape[1] = {1024};
  float values[1024] = {0};
  PolyTensor *first = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_CPU);
  PolyTensor *second = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(first);
  ASSERT_NOT_NULL(second);
  ASSERT_INT_EQ(poly_buffer_write(ctx, first->uop_physical, values, sizeof(values)), 0);
  ASSERT_INT_EQ(poly_buffer_write(ctx, second->uop_physical, values, sizeof(values)), 0);

  PolyCtxStats stats = {0};
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &stats), 0);
  ASSERT_TRUE(stats.mem_used == 8192);

  poly_tensor_release(second);
  ASSERT_INT_EQ(poly_ctx_collect(ctx), 0);
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &stats), 0);
  ASSERT_TRUE(stats.mem_used == 4096);

  poly_tensor_release(first);
  ASSERT_INT_EQ(poly_ctx_collect(ctx), 0);
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &stats), 0);
  ASSERT_TRUE(stats.mem_used == 0);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(tensor, dirty_residency_is_collected_before_replacement_allocation) {
  /* Tinygrad 2026-08-22 device.py:163-191 deallocates dead Buffer storage
   * before a replacement Buffer allocates.  Polygrad defers the traversal to
   * this next allocation safe point. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  int64_t shape[1] = {1000000};
  float *values = calloc((size_t)shape[0], sizeof(*values));
  ASSERT_NOT_NULL(values);

  PolyTensor *first = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(first);
  ASSERT_INT_EQ(
      poly_buffer_write(ctx, first->uop_physical, values, (size_t)shape[0] * sizeof(*values)), 0
  );
  ASSERT_TRUE(poly_ctx_mem_used_for_device(ctx, POLY_DEVICE_CPU) == 4000000);
  poly_tensor_release(first);

  PolyTensor *second = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(second);
  ASSERT_INT_EQ(
      poly_buffer_write(ctx, second->uop_physical, values, (size_t)shape[0] * sizeof(*values)), 0
  );
  ASSERT_TRUE(poly_ctx_mem_used_for_device(ctx, POLY_DEVICE_CPU) == 4000000);

  poly_tensor_release(second);
  free(values);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(tensor, retained_physical_uop_owns_residency_but_logical_uop_does_not) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  int64_t shape[1] = {1024};
  float values[1024] = {0};

  PolyTensor *physical_owner = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(physical_owner);
  ASSERT_INT_EQ(poly_buffer_write(ctx, physical_owner->uop_physical, values, sizeof(values)), 0);
  PolyUOp *physical = physical_owner->uop_physical;
  ASSERT_INT_EQ(poly_uop_retain(ctx, physical), 0);
  poly_tensor_release(physical_owner);
  ASSERT_INT_EQ(poly_ctx_collect(ctx), 0);
  PolyCtxStats stats = {0};
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &stats), 0);
  ASSERT_TRUE(stats.mem_used == 4096);
  poly_uop_release(ctx, physical);
  ASSERT_INT_EQ(poly_ctx_collect(ctx), 0);
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &stats), 0);
  ASSERT_TRUE(stats.mem_used == 0);

  PolyTensor *logical_owner = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(logical_owner);
  ASSERT_INT_EQ(poly_buffer_write(ctx, logical_owner->uop_physical, values, sizeof(values)), 0);
  PolyUOp *logical = logical_owner->uop_logical;
  ASSERT_INT_EQ(poly_uop_retain(ctx, logical), 0);
  poly_tensor_release(logical_owner);
  ASSERT_INT_EQ(poly_ctx_collect(ctx), 0);
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &stats), 0);
  ASSERT_TRUE(stats.mem_used == 0);
  poly_uop_release(ctx, logical);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(tensor, movement_view_keeps_base_residency_until_last_handle_release) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  int64_t shape[1] = {1024};
  float values[1024] = {0};
  PolyTensor *base = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(base);
  ASSERT_INT_EQ(poly_buffer_write(ctx, base->uop_physical, values, sizeof(values)), 0);
  int64_t view_shape[2] = {256, 4};
  PolyTensor *view = poly_tensor_reshape(ctx, base, view_shape, 2);
  ASSERT_NOT_NULL(view);

  poly_tensor_release(base);
  ASSERT_INT_EQ(poly_ctx_collect(ctx), 0);
  PolyCtxStats stats = {0};
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &stats), 0);
  ASSERT_TRUE(stats.mem_used == 4096);

  poly_tensor_release(view);
  ASSERT_INT_EQ(poly_ctx_collect(ctx), 0);
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &stats), 0);
  ASSERT_TRUE(stats.mem_used == 0);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(tensor, downstream_graph_owns_input_residency_after_input_handle_release) {
  /* Tinygrad 2026-08-22 UOp src edges keep input Buffers live until the
   * downstream Tensor is realized; its realized BUFFER then replaces that
   * computation root (tensor.py:195-203). */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  int64_t shape[1] = {1024};
  float values[1024] = {0};
  PolyTensor *input = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_CPU);
  PolyTensor *one =
      poly_tensor_const_float_by_id(ctx, 1.0, poly_dtype_id_by_name("float32"), POLY_DEVICE_CPU);
  PolyTensor *out = poly_tensor_alu2(ctx, POLY_OP_ADD, input, one);
  ASSERT_NOT_NULL(input);
  ASSERT_NOT_NULL(one);
  ASSERT_NOT_NULL(out);
  ASSERT_INT_EQ(poly_buffer_write(ctx, input->uop_physical, values, sizeof(values)), 0);

  poly_tensor_release(input);
  poly_tensor_release(one);
  ASSERT_INT_EQ(poly_ctx_collect(ctx), 0);
  PolyCtxStats stats = {0};
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &stats), 0);
  ASSERT_TRUE(stats.mem_used == 4096);

  PolyTensor *realized = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &out, 1, &realized), 0);
  ASSERT_PTR_EQ(realized, out);
  ASSERT_TRUE(ctx->mem_used == 4096);
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &stats), 0);
  ASSERT_TRUE(stats.mem_used == 4096);

  poly_tensor_release(out);
  ASSERT_INT_EQ(poly_ctx_collect(ctx), 0);
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &stats), 0);
  ASSERT_TRUE(stats.mem_used == 0);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(tensor, logical_policy_always_preserves_producer) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  ASSERT_INT_EQ(poly_ctx_set_logical_policy(ctx, POLY_LOGICAL_ALWAYS), 0);
  int64_t shape[] = {4};
  float x_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
  float bias_data[] = {0.25f, 0.25f, 0.25f, 0.25f};
  PolyTensor *x = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_INTERP);
  PolyTensor *bias = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_INTERP);
  ASSERT_NOT_NULL(x);
  ASSERT_NOT_NULL(bias);
  ASSERT_INT_EQ(poly_buffer_write(ctx, x->uop_physical, x_data, sizeof(x_data)), 0);
  ASSERT_INT_EQ(poly_buffer_write(ctx, bias->uop_physical, bias_data, sizeof(bias_data)), 0);

  PolyTensor *out = poly_tensor_alu2(ctx, POLY_OP_ADD, x, bias);
  ASSERT_NOT_NULL(out);
  PolyUOp *logical_before = poly_tensor_uop_logical(out);
  ASSERT_NOT_NULL(logical_before);
  ASSERT_INT_EQ(poly_tensor_logical_policy(out), POLY_LOGICAL_ALWAYS);
  ASSERT_INT_EQ(poly_tensor_logical_state(out), POLY_LOGICAL_AVAILABLE);
  PolyTensor *realized = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &out, 1, &realized), 0);
  ASSERT_PTR_EQ(realized, out);
  ASSERT_PTR_EQ(poly_tensor_uop_logical(out), logical_before);
  ASSERT_NOT_NULL(poly_uop_get_buffer_identity(poly_tensor_uop_physical(out)));

  PolyTensor *next = poly_tensor_alu2(ctx, POLY_OP_ADD, out, bias);
  ASSERT_NOT_NULL(next);
  ASSERT_NOT_NULL(poly_tensor_uop_logical(next));
  ASSERT_INT_EQ(poly_tensor_logical_policy(next), POLY_LOGICAL_ALWAYS);
  float got[4] = {0};
  ASSERT_INT_EQ(read_tensor_f32(ctx, next, got, 4), 0);
  for (int i = 0; i < 4; i++)
    ASSERT_FLOAT_EQ(got[i], x_data[i] + 0.5f, 1e-6f);

  poly_tensor_release(next);
  poly_tensor_release(out);
  poly_tensor_release(bias);
  poly_tensor_release(x);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(tensor, logical_policy_is_copied_from_context_at_construction) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  ASSERT_INT_EQ(poly_ctx_set_logical_policy(ctx, POLY_LOGICAL_ALWAYS), 0);
  int64_t shape[] = {4};
  PolyTensor *before = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_INTERP);
  ASSERT_NOT_NULL(before);
  ASSERT_INT_EQ(poly_tensor_logical_policy(before), POLY_LOGICAL_ALWAYS);

  ASSERT_INT_EQ(poly_ctx_set_logical_policy(ctx, POLY_LOGICAL_UNTIL_REALIZE), 0);
  PolyTensor *after = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_INTERP);
  ASSERT_NOT_NULL(after);
  ASSERT_INT_EQ(poly_tensor_logical_policy(before), POLY_LOGICAL_ALWAYS);
  ASSERT_INT_EQ(poly_tensor_logical_policy(after), POLY_LOGICAL_UNTIL_REALIZE);
  ASSERT_INT_EQ(poly_tensor_logical_state(after), POLY_LOGICAL_AVAILABLE);

  poly_tensor_release(after);
  poly_tensor_release(before);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(tensor, logical_never_source_keeps_descendants_physical_only_after_scope) {
  /* Polygrad logical-lifetime divergence: omitting the portable twin must not
   * make later eager physical composition depend on the current ctx policy. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  ASSERT_INT_EQ(poly_ctx_set_logical_policy(ctx, POLY_LOGICAL_NEVER), 0);
  int64_t shape[] = {2};
  float values[] = {1.0f, 2.0f};
  PolyTensor *x = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_INTERP);
  PolyTensor *one = x ? poly_tensor_const_like_float(ctx, x, 1.0) : NULL;
  ASSERT_NOT_NULL(x);
  ASSERT_NOT_NULL(one);
  ASSERT_INT_EQ(poly_buffer_write(ctx, x->uop_physical, values, sizeof(values)), 0);

  ASSERT_INT_EQ(poly_ctx_set_logical_policy(ctx, POLY_LOGICAL_UNTIL_REALIZE), 0);
  PolyTensor *out = poly_tensor_alu2(ctx, POLY_OP_ADD, x, one);
  ASSERT_NOT_NULL(out);
  ASSERT_INT_EQ(poly_tensor_logical_policy(out), POLY_LOGICAL_NEVER);
  ASSERT_INT_EQ(poly_tensor_logical_state(out), POLY_LOGICAL_NEVER_CONSTRUCTED);
  ASSERT_PTR_EQ(poly_tensor_uop_logical(out), NULL);
  float got[2] = {0};
  ASSERT_INT_EQ(read_tensor_f32(ctx, out, got, 2), 0);
  ASSERT_FLOAT_EQ(got[0], 2.0f, 1e-6f);
  ASSERT_FLOAT_EQ(got[1], 3.0f, 1e-6f);

  poly_tensor_release(out);
  poly_tensor_release(one);
  poly_tensor_release(x);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(tensor, logical_never_clone_after_scope_keeps_exact_physical_clone) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  ASSERT_INT_EQ(poly_ctx_set_logical_policy(ctx, POLY_LOGICAL_NEVER), 0);
  int64_t shape[] = {2};
  float values[] = {1.0f, 2.0f};
  PolyTensor *source = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_INTERP);
  ASSERT_NOT_NULL(source);
  ASSERT_INT_EQ(poly_buffer_write(ctx, source->uop_physical, values, sizeof(values)), 0);

  ASSERT_INT_EQ(poly_ctx_set_logical_policy(ctx, POLY_LOGICAL_UNTIL_REALIZE), 0);
  PolyTensor *cloned = poly_tensor_clone(ctx, source, POLY_DEVICE_INTERP);
  ASSERT_NOT_NULL(cloned);
  ASSERT_INT_EQ(cloned->logical_policy, POLY_LOGICAL_NEVER);
  ASSERT_INT_EQ(cloned->logical_state, POLY_LOGICAL_NEVER_CONSTRUCTED);
  ASSERT_PTR_EQ(cloned->uop_logical, NULL);
  ASSERT_INT_EQ(cloned->uop_physical->op, POLY_OP_AFTER);
  ASSERT_INT_EQ(cloned->uop_physical->n_src, 2);
  ASSERT_INT_EQ(cloned->uop_physical->src[0]->op, POLY_OP_BUFFER);
  ASSERT_INT_EQ(cloned->uop_physical->src[1]->op, POLY_OP_STORE);
  ASSERT_PTR_EQ(cloned->uop_physical->src[1]->src[0], cloned->uop_physical->src[0]);
  ASSERT_PTR_EQ(cloned->uop_physical->src[1]->src[1], source->uop_physical);
  float got[2] = {0};
  ASSERT_INT_EQ(read_tensor_f32(ctx, cloned, got, 2), 0);
  ASSERT_FLOAT_EQ(got[0], values[0], 1e-6f);
  ASSERT_FLOAT_EQ(got[1], values[1], 1e-6f);

  poly_tensor_release(cloned);
  poly_tensor_release(source);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(tensor, logical_policy_default_is_until_realize) {
  const char *prior = getenv("POLY_LOGICAL");
  char *saved = prior ? strdup(prior) : NULL;
  ASSERT_INT_EQ(unsetenv("POLY_LOGICAL"), 0);
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  ASSERT_INT_EQ(poly_ctx_get_logical_policy(ctx), POLY_LOGICAL_UNTIL_REALIZE);
  poly_ctx_destroy(ctx);
  if (saved) {
    ASSERT_INT_EQ(setenv("POLY_LOGICAL", saved, 1), 0);
    free(saved);
  }
  PASS();
}

TEST(tensor, logical_policy_environment_is_strict_and_context_local) {
  const char *prior = getenv("POLY_LOGICAL");
  char *saved = prior ? strdup(prior) : NULL;
  const struct {
    const char *value;
    PolyLogicalPolicy policy;
  } cases[] = {
      {"0", POLY_LOGICAL_NEVER},
      {"1", POLY_LOGICAL_ALWAYS},
      {"2", POLY_LOGICAL_UNTIL_REALIZE},
  };
  for (int i = 0; i < 3; i++) {
    ASSERT_INT_EQ(setenv("POLY_LOGICAL", cases[i].value, 1), 0);
    PolyCtx *ctx = poly_ctx_new();
    ASSERT_NOT_NULL(ctx);
    ASSERT_INT_EQ(poly_ctx_get_logical_policy(ctx), cases[i].policy);
    poly_ctx_destroy(ctx);
  }
  ASSERT_INT_EQ(setenv("POLY_LOGICAL", "always", 1), 0);
  ASSERT_PTR_EQ(poly_ctx_new(), NULL);
  if (saved) {
    ASSERT_INT_EQ(setenv("POLY_LOGICAL", saved, 1), 0);
    free(saved);
  } else {
    ASSERT_INT_EQ(unsetenv("POLY_LOGICAL"), 0);
  }
  PASS();
}

TEST(tensor, logical_policy_explicit_tensor_transitions_are_non_retroactive) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  ASSERT_INT_EQ(poly_ctx_set_logical_policy(ctx, POLY_LOGICAL_ALWAYS), 0);
  int64_t shape[1] = {2};
  PolyTensor *x = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_INTERP);
  PolyTensor *pending =
      poly_tensor_alu2(ctx, POLY_OP_ADD, x, poly_tensor_const_like_float(ctx, x, 1.0));
  ASSERT_NOT_NULL(x);
  ASSERT_NOT_NULL(pending);
  ASSERT_INT_EQ(poly_tensor_set_logical_policy(ctx, pending, POLY_LOGICAL_UNTIL_REALIZE), 0);
  ASSERT_INT_EQ(pending->logical_policy, POLY_LOGICAL_UNTIL_REALIZE);
  ASSERT_INT_EQ(pending->logical_state, POLY_LOGICAL_AVAILABLE);
  PolyTensor *realized = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &pending, 1, &realized), 0);
  ASSERT_INT_EQ(pending->logical_state, POLY_LOGICAL_RETIRED);
  ASSERT_INT_EQ(poly_tensor_set_logical_policy(ctx, pending, POLY_LOGICAL_ALWAYS), -1);

  PolyTensor *drop =
      poly_tensor_alu2(ctx, POLY_OP_ADD, x, poly_tensor_const_like_float(ctx, x, 2.0));
  ASSERT_NOT_NULL(drop);
  ASSERT_INT_EQ(poly_tensor_set_logical_policy(ctx, drop, POLY_LOGICAL_NEVER), 0);
  ASSERT_INT_EQ(drop->logical_policy, POLY_LOGICAL_NEVER);
  ASSERT_INT_EQ(drop->logical_state, POLY_LOGICAL_NEVER_CONSTRUCTED);
  ASSERT_PTR_EQ(drop->uop_logical, NULL);
  ASSERT_INT_EQ(poly_tensor_set_logical_policy(ctx, drop, POLY_LOGICAL_ALWAYS), -1);

  PolyTensor *late =
      poly_tensor_alu2(ctx, POLY_OP_ADD, x, poly_tensor_const_like_float(ctx, x, 3.0));
  ASSERT_NOT_NULL(late);
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &late, 1, &realized), 0);
  ASSERT_INT_EQ(late->logical_state, POLY_LOGICAL_AVAILABLE);
  ASSERT_INT_EQ(poly_tensor_set_logical_policy(ctx, late, POLY_LOGICAL_UNTIL_REALIZE), 0);
  ASSERT_INT_EQ(late->logical_state, POLY_LOGICAL_RETIRED);

  poly_tensor_release(late);
  poly_tensor_release(drop);
  poly_tensor_release(pending);
  poly_tensor_release(x);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(tensor, logical_until_realize_retires_producer_to_exact_current_resource) {
  /* Tinygrad 2026-08-22 Tensor.realize retargets the current value to its
   * exact BUFFER. Polygrad additionally preserves a device-free resource leaf
   * with the same slot after retiring this wrapper's portable producer. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  ASSERT_INT_EQ(poly_ctx_set_logical_policy(ctx, POLY_LOGICAL_UNTIL_REALIZE), 0);
  int64_t shape[1] = {2};
  float values[2] = {1.0f, 2.0f};
  PolyTensor *x = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_INTERP);
  PolyTensor *one = poly_tensor_const_like_float(ctx, x, 1.0);
  PolyTensor *out = poly_tensor_alu2(ctx, POLY_OP_ADD, x, one);
  ASSERT_NOT_NULL(x);
  ASSERT_NOT_NULL(one);
  ASSERT_NOT_NULL(out);
  ASSERT_INT_EQ(poly_buffer_write(ctx, x->uop_physical, values, sizeof(values)), 0);

  PolyUOp *producer = out->uop_logical;
  ASSERT_NOT_NULL(producer);
  ASSERT_INT_EQ(out->logical_state, POLY_LOGICAL_AVAILABLE);
  PolyTensor *realized = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &out, 1, &realized), 0);
  ASSERT_PTR_EQ(realized, out);

  PolyUOp *physical = out->uop_physical;
  PolyUOp *resource = out->uop_logical;
  ASSERT_NOT_NULL(physical);
  ASSERT_NOT_NULL(resource);
  ASSERT_INT_EQ(out->logical_state, POLY_LOGICAL_RETIRED);
  ASSERT_PTR_NEQ(resource, producer);
  ASSERT_FALSE(poly_uop_reachable(ctx, resource, producer));
  ASSERT_INT_EQ(physical->op, POLY_OP_BUFFER);
  ASSERT_INT_EQ(physical->arg.kind, POLY_ARG_PARAM);
  ASSERT_INT_EQ(resource->op, POLY_OP_BUFFER);
  ASSERT_INT_EQ(resource->arg.kind, POLY_ARG_INT);
  ASSERT_INT_EQ(resource->n_src, 1);
  ASSERT_INT_EQ(resource->src[0]->op, POLY_OP_UNIQUE);
  ASSERT_TRUE(resource->src[0]->arg.i == physical->arg.param->slot);
  ASSERT_TRUE(resource->arg.i == physical->src[0]->arg.i);

  PolyTensor *two = poly_tensor_const_like_float(ctx, out, 2.0);
  PolyTensor *next = poly_tensor_alu2(ctx, POLY_OP_ADD, out, two);
  ASSERT_NOT_NULL(next);
  ASSERT_TRUE(poly_uop_reachable(ctx, next->uop_logical, resource));
  ASSERT_FALSE(poly_uop_reachable(ctx, next->uop_logical, producer));
  float got[2] = {0};
  ASSERT_INT_EQ(read_tensor_f32(ctx, next, got, 2), 0);
  ASSERT_FLOAT_EQ(got[0], 4.0f, 1e-6f);
  ASSERT_FLOAT_EQ(got[1], 5.0f, 1e-6f);

  poly_tensor_release(next);
  poly_tensor_release(two);
  poly_tensor_release(out);
  poly_tensor_release(one);
  poly_tensor_release(x);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(tensor, logical_until_realize_does_not_rewrite_always_sibling) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  ASSERT_INT_EQ(poly_ctx_set_logical_policy(ctx, POLY_LOGICAL_UNTIL_REALIZE), 0);
  int64_t shape[1] = {2};
  float values[2] = {1.0f, 2.0f};
  PolyTensor *x = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_INTERP);
  PolyTensor *one = poly_tensor_const_like_float(ctx, x, 1.0);
  PolyTensor *until = poly_tensor_alu2(ctx, POLY_OP_ADD, x, one);
  ASSERT_NOT_NULL(x);
  ASSERT_NOT_NULL(one);
  ASSERT_NOT_NULL(until);
  ASSERT_INT_EQ(poly_buffer_write(ctx, x->uop_physical, values, sizeof(values)), 0);
  PolyUOp *producer = until->uop_logical;
  PolyUOp *physical = until->uop_physical;

  ASSERT_INT_EQ(poly_ctx_set_logical_policy(ctx, POLY_LOGICAL_ALWAYS), 0);
  PolyTensor *sibling =
      poly_tensor_create_with_roots(ctx, producer, physical, POLY_TENSOR_VALUE, POLY_DEVICE_INTERP);
  ASSERT_NOT_NULL(sibling);
  ASSERT_INT_EQ(sibling->logical_policy, POLY_LOGICAL_ALWAYS);

  PolyTensor *realized = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &until, 1, &realized), 0);
  ASSERT_PTR_EQ(realized, until);
  ASSERT_INT_EQ(until->logical_state, POLY_LOGICAL_RETIRED);
  ASSERT_PTR_EQ(sibling->uop_logical, producer);
  ASSERT_INT_EQ(sibling->logical_state, POLY_LOGICAL_AVAILABLE);
  ASSERT_PTR_EQ(sibling->uop_physical, until->uop_physical);

  poly_tensor_release(sibling);
  poly_tensor_release(until);
  poly_tensor_release(one);
  poly_tensor_release(x);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(tensor, logical_until_realize_preserves_exact_reshape_resource_spine) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  ASSERT_INT_EQ(poly_ctx_set_logical_policy(ctx, POLY_LOGICAL_UNTIL_REALIZE), 0);
  int64_t flat_shape[1] = {4}, matrix_shape[2] = {2, 2};
  float values[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  PolyTensor *x = poly_tensor_empty(ctx, POLY_FLOAT32, flat_shape, 1, POLY_DEVICE_INTERP);
  PolyTensor *one = poly_tensor_const_like_float(ctx, x, 1.0);
  PolyTensor *sum = poly_tensor_alu2(ctx, POLY_OP_ADD, x, one);
  PolyTensor *matrix = poly_tensor_reshape(ctx, sum, matrix_shape, 2);
  ASSERT_NOT_NULL(x);
  ASSERT_NOT_NULL(one);
  ASSERT_NOT_NULL(sum);
  ASSERT_NOT_NULL(matrix);
  ASSERT_INT_EQ(poly_buffer_write(ctx, x->uop_physical, values, sizeof(values)), 0);
  PolyUOp *producer = matrix->uop_logical;

  PolyTensor *realized = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &matrix, 1, &realized), 0);
  ASSERT_PTR_EQ(realized, matrix);
  ASSERT_INT_EQ(matrix->uop_physical->op, POLY_OP_RESHAPE);
  ASSERT_INT_EQ(matrix->uop_physical->src[0]->op, POLY_OP_BUFFER);
  ASSERT_INT_EQ(matrix->logical_state, POLY_LOGICAL_RETIRED);
  ASSERT_PTR_NEQ(matrix->uop_logical, producer);
  ASSERT_INT_EQ(matrix->uop_logical->op, POLY_OP_RESHAPE);
  ASSERT_INT_EQ(matrix->uop_logical->src[0]->op, POLY_OP_BUFFER);
  ASSERT_INT_EQ(matrix->uop_logical->src[0]->arg.kind, POLY_ARG_INT);
  ASSERT_TRUE(
      matrix->uop_logical->src[0]->src[0]->arg.i == matrix->uop_physical->src[0]->arg.param->slot
  );
  ASSERT_FALSE(poly_uop_reachable(ctx, matrix->uop_logical, producer));

  float got[4] = {0};
  ASSERT_INT_EQ(read_tensor_f32(ctx, matrix, got, 4), 0);
  for (int i = 0; i < 4; i++)
    ASSERT_FLOAT_EQ(got[i], values[i] + 1.0f, 1e-6f);

  poly_tensor_release(matrix);
  poly_tensor_release(sum);
  poly_tensor_release(one);
  poly_tensor_release(x);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(tensor, logical_until_realize_preserves_canonical_shrink_resource_spine) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  ASSERT_INT_EQ(poly_ctx_set_logical_policy(ctx, POLY_LOGICAL_UNTIL_REALIZE), 0);
  int64_t shape[1] = {4}, pairs[1][2] = {{1, 3}};
  float values[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  PolyTensor *x = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_INTERP);
  PolyTensor *one = poly_tensor_const_like_float(ctx, x, 1.0);
  PolyTensor *sum = poly_tensor_alu2(ctx, POLY_OP_ADD, x, one);
  PolyTensor *view = poly_tensor_shrink(ctx, sum, pairs, 1);
  ASSERT_NOT_NULL(x);
  ASSERT_NOT_NULL(one);
  ASSERT_NOT_NULL(sum);
  ASSERT_NOT_NULL(view);
  ASSERT_INT_EQ(poly_buffer_write(ctx, x->uop_physical, values, sizeof(values)), 0);
  PolyUOp *producer = view->uop_logical;

  PolyTensor *realized = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &view, 1, &realized), 0);
  ASSERT_PTR_EQ(realized, view);
  ASSERT_INT_EQ(view->uop_physical->op, POLY_OP_SHRINK);
  ASSERT_INT_EQ(view->uop_physical->src[0]->op, POLY_OP_BUFFER);
  ASSERT_INT_EQ(view->logical_state, POLY_LOGICAL_RETIRED);
  ASSERT_PTR_NEQ(view->uop_logical, producer);
  ASSERT_INT_EQ(view->uop_logical->op, POLY_OP_SHRINK);
  ASSERT_INT_EQ(view->uop_logical->src[0]->op, POLY_OP_BUFFER);
  ASSERT_TRUE(
      view->uop_logical->src[0]->src[0]->arg.i == view->uop_physical->src[0]->arg.param->slot
  );
  ASSERT_FALSE(poly_uop_reachable(ctx, view->uop_logical, producer));

  PolyUOp *view_buffer = poly_uop_buffer(ctx, view->uop_physical);
  float got[2] = {0};
  ASSERT_NOT_NULL(view_buffer);
  ASSERT_INT_EQ(poly_buffer_read(ctx, view_buffer, got, sizeof(got)), 0);
  ASSERT_FLOAT_EQ(got[0], 3.0f, 1e-6f);
  ASSERT_FLOAT_EQ(got[1], 4.0f, 1e-6f);

  poly_tensor_release(view);
  poly_tensor_release(sum);
  poly_tensor_release(one);
  poly_tensor_release(x);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(tensor, logical_until_realize_preserves_exact_movement_resource_spine) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  ASSERT_INT_EQ(poly_ctx_set_logical_policy(ctx, POLY_LOGICAL_UNTIL_REALIZE), 0);
  int64_t flat_shape[1] = {6}, matrix_shape[2] = {2, 3}, order[2] = {1, 0};
  PolyTensor *x = poly_tensor_empty(ctx, POLY_FLOAT32, flat_shape, 1, POLY_DEVICE_INTERP);
  ASSERT_NOT_NULL(x);
  PolyUOp *physical_base = poly_tensor_uop_physical(x);
  PolyUOp *logical_base = poly_tensor_uop_logical(x);
  PolyUOp *one = poly_const_like_float(ctx, logical_base, 1.0);
  PolyUOp *producer = poly_permute(
      ctx, poly_reshape(ctx, poly_alu2(ctx, POLY_OP_ADD, logical_base, one), matrix_shape, 2),
      order, 2
  );
  PolyUOp *physical_view =
      poly_permute(ctx, poly_reshape(ctx, physical_base, matrix_shape, 2), order, 2);
  PolyTensor *view = poly_tensor_create_with_roots(
      ctx, producer, physical_view, POLY_TENSOR_VALUE, POLY_DEVICE_INTERP
  );
  ASSERT_NOT_NULL(view);
  ASSERT_NOT_NULL(physical_base);
  ASSERT_NOT_NULL(physical_view);
  ASSERT_NOT_NULL(producer);
  ASSERT_INT_EQ(poly_tensor_retire_logical_resources(ctx, &view, 1), 0);

  PolyUOp *resource = poly_tensor_uop_logical(view);
  ASSERT_NOT_NULL(resource);
  ASSERT_INT_EQ(poly_tensor_logical_state(view), POLY_LOGICAL_RETIRED);
  ASSERT_PTR_NEQ(resource, producer);
  ASSERT_INT_EQ(resource->op, POLY_OP_PERMUTE);
  ASSERT_INT_EQ(resource->src[0]->op, POLY_OP_RESHAPE);
  ASSERT_INT_EQ(resource->src[0]->src[0]->op, POLY_OP_BUFFER);
  ASSERT_INT_EQ(resource->src[0]->src[0]->arg.kind, POLY_ARG_INT);
  ASSERT_PTR_EQ(poly_tensor_uop_physical(view), physical_view);
  ASSERT_TRUE(resource->src[0]->src[0]->src[0]->arg.i == physical_base->arg.param->slot);
  ASSERT_FALSE(poly_uop_reachable(ctx, resource, producer));

  poly_tensor_release(view);
  poly_tensor_release(x);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(tensor, logical_until_realize_copy_roundtrip_adopts_each_destination_resource) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  ASSERT_INT_EQ(poly_ctx_set_logical_policy(ctx, POLY_LOGICAL_UNTIL_REALIZE), 0);
  int64_t shape[1] = {2};
  float values[2] = {3.0f, 4.0f};
  PolyTensor *x = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(x);
  ASSERT_INT_EQ(poly_buffer_write(ctx, x->uop_physical, values, sizeof(values)), 0);
  PolyUOp *x_logical = x->uop_logical;
  ASSERT_NOT_NULL(x_logical);

  PolyTensor *interp = poly_tensor_to_device(ctx, x, POLY_DEVICE_INTERP);
  ASSERT_NOT_NULL(interp);
  ASSERT_PTR_EQ(interp->uop_logical, x_logical);
  PolyTensor *realized = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &interp, 1, &realized), 0);
  ASSERT_PTR_EQ(realized, interp);
  ASSERT_INT_EQ(interp->logical_state, POLY_LOGICAL_RETIRED);
  PolyUOp *interp_resource = interp->uop_logical;
  ASSERT_INT_EQ(interp_resource->op, POLY_OP_BUFFER);
  ASSERT_PTR_NEQ(interp_resource, x_logical);
  ASSERT_TRUE(interp_resource->src[0]->arg.i != base_buf(x_logical)->src[0]->arg.i);

  PolyTensor *roundtrip = poly_tensor_to_device(ctx, interp, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(roundtrip);
  ASSERT_PTR_EQ(roundtrip->uop_logical, interp_resource);
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &roundtrip, 1, &realized), 0);
  ASSERT_PTR_EQ(realized, roundtrip);
  ASSERT_INT_EQ(roundtrip->logical_state, POLY_LOGICAL_RETIRED);
  PolyUOp *roundtrip_resource = roundtrip->uop_logical;
  ASSERT_INT_EQ(roundtrip_resource->op, POLY_OP_BUFFER);
  ASSERT_TRUE(roundtrip_resource->src[0]->arg.i != interp_resource->src[0]->arg.i);
  ASSERT_PTR_EQ(x->uop_logical, x_logical);
  ASSERT_INT_EQ(x->logical_state, POLY_LOGICAL_AVAILABLE);

  const PolyUOp *roundtrip_buffer = poly_uop_get_buffer_identity(roundtrip->uop_physical);
  float got[2] = {0};
  ASSERT_NOT_NULL(roundtrip_buffer);
  ASSERT_INT_EQ(poly_buffer_read(ctx, (PolyUOp *)roundtrip_buffer, got, sizeof(got)), 0);
  ASSERT_FLOAT_EQ(got[0], values[0], 1e-6f);
  ASSERT_FLOAT_EQ(got[1], values[1], 1e-6f);

  poly_tensor_release(roundtrip);
  poly_tensor_release(interp);
  poly_tensor_release(x);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(tensor, logical_until_realize_unsupported_multi_keeps_physical_composition) {
  /* Tinygrad 2026-08-22 can materialize a sharded Tensor as
   * UNSHARD(BUFFER(multi-device), RANGE). Polygrad cannot yet encode that
   * resource portably. Retirement bounds history, while later eager physical
   * composition remains valid and propagates the portability limitation. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  ASSERT_INT_EQ(poly_ctx_set_logical_policy(ctx, POLY_LOGICAL_UNTIL_REALIZE), 0);
  int64_t shape[1] = {2};
  PolyTensor *cpu = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_CPU);
  PolyTensor *interp = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_INTERP);
  ASSERT_NOT_NULL(cpu);
  ASSERT_NOT_NULL(interp);
  PolyUOp *producer = poly_alu2(ctx, POLY_OP_ADD, cpu->uop_logical, interp->uop_logical);
  PolyUOp *multi_src[2] = {cpu->uop_physical, interp->uop_physical};
  PolyUOp *multi = poly_uop(ctx, POLY_OP_MSTACK, POLY_FLOAT32, multi_src, 2, poly_arg_none());
  PolyTensor *out =
      poly_tensor_create_with_roots(ctx, producer, multi, POLY_TENSOR_VALUE, POLY_DEVICE_AUTO);
  ASSERT_NOT_NULL(producer);
  ASSERT_NOT_NULL(multi);
  ASSERT_NOT_NULL(out);

  ASSERT_INT_EQ(poly_tensor_retire_logical_resources(ctx, &out, 1), 0);
  ASSERT_INT_EQ(out->logical_state, POLY_LOGICAL_UNSUPPORTED_RESOURCE);
  ASSERT_PTR_EQ(out->uop_logical, NULL);
  ASSERT_PTR_EQ(out->uop_physical, multi);
  PolyTensor *composed = poly_tensor_alu2(ctx, POLY_OP_ADD, out, cpu);
  ASSERT_NOT_NULL(composed);
  ASSERT_INT_EQ(composed->logical_state, POLY_LOGICAL_UNSUPPORTED_RESOURCE);
  ASSERT_PTR_EQ(composed->uop_logical, NULL);
  ASSERT_INT_EQ(composed->uop_physical->op, POLY_OP_ADD);
  ASSERT_PTR_EQ(composed->uop_physical->src[0], multi);

  poly_tensor_release(composed);
  poly_tensor_release(out);
  poly_tensor_release(interp);
  poly_tensor_release(cpu);
  poly_ctx_destroy(ctx);
  PASS();
}

static PolyTensor *build_logical_policy_oracle(
    PolyCtx *ctx,
    PolyLogicalPolicy policy,
    float values[4]
) {
  int64_t shape[1] = {4};
  int64_t reshaped[2] = {2, 2};
  if (poly_ctx_set_logical_policy(ctx, policy) != 0) return NULL;
  PolyTensor *x = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_CPU);
  if (!x || poly_buffer_write(ctx, poly_tensor_uop_physical(x), values, sizeof(float) * 4) != 0)
    return NULL;
  PolyTensor *two = x ? poly_tensor_const_like_float(ctx, x, 2.0) : NULL;
  PolyTensor *sum = two ? poly_tensor_alu2(ctx, POLY_OP_ADD, x, two) : NULL;
  return sum ? poly_tensor_reshape(ctx, sum, reshaped, 2) : NULL;
}

TEST(tensor, logical_never_keeps_physical_graph_and_values_exact) {
  float values[4] = {1, 2, 3, 4};
  PolyCtx *always_ctx = poly_ctx_new(), *never_ctx = poly_ctx_new();
  ASSERT_NOT_NULL(always_ctx);
  ASSERT_NOT_NULL(never_ctx);
  PolyTensor *always = build_logical_policy_oracle(always_ctx, POLY_LOGICAL_ALWAYS, values);
  PolyTensor *never = build_logical_policy_oracle(never_ctx, POLY_LOGICAL_NEVER, values);
  ASSERT_NOT_NULL(always);
  ASSERT_NOT_NULL(never);
  ASSERT_NOT_NULL(poly_tensor_uop_logical(always));
  ASSERT_PTR_EQ(poly_tensor_uop_logical(never), NULL);
  ASSERT_INT_EQ(poly_tensor_logical_state(never), POLY_LOGICAL_NEVER_CONSTRUCTED);
  ASSERT_TRUE(uop_graph_isomorphic(
      always_ctx, poly_tensor_uop_physical(always), never_ctx, poly_tensor_uop_physical(never)
  ));

  float always_values[4] = {0}, never_values[4] = {0};
  ASSERT_INT_EQ(read_tensor_f32(always_ctx, always, always_values, 4), 0);
  ASSERT_INT_EQ(read_tensor_f32(never_ctx, never, never_values, 4), 0);
  for (int i = 0; i < 4; i++) {
    ASSERT_FLOAT_EQ(always_values[i], values[i] + 2.0f, 1e-6);
    ASSERT_FLOAT_EQ(never_values[i], always_values[i], 1e-6);
  }
  poly_ctx_destroy(always_ctx);
  poly_ctx_destroy(never_ctx);
  PASS();
}

static int run_logical_never_realized_steps(
    PolyCtx *ctx,
    PolyTensor **current,
    PolyTensor *bias,
    int steps
) {
  if (!ctx || !current || !*current || !bias || steps < 0) return -1;
  for (int i = 0; i < steps; i++) {
    PolyTensor *next = poly_tensor_alu2(ctx, POLY_OP_ADD, *current, bias);
    if (!next) return -1;
    poly_tensor_release(*current);
    *current = next;
    PolyTensor *realized = NULL;
    if (poly_realize_tensors(ctx, current, 1, &realized) != 0 || realized != *current) return -1;
  }
  return 0;
}

TEST(tensor, logical_never_realized_loop_reclaims_transient_ir_storage) {
  /* Tinygrad 2026-08-22/a9069c177a9d weak UOps, recursive_property shape
   * rows, and Buffer refcounts release each unreachable iteration. The C
   * collector may defer weak IR only within its fixed growth budget. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  ASSERT_INT_EQ(poly_ctx_set_logical_policy(ctx, POLY_LOGICAL_NEVER), 0);
  poly_ctx_set_preferred_device(ctx, POLY_DEVICE_INTERP);
  int64_t shape[1] = {4};
  float values[4] = {1, 2, 3, 4};
  float bias_values[4] = {.25f, .25f, .25f, .25f};
  PolyTensor *current = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_INTERP);
  PolyTensor *bias = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_INTERP);
  ASSERT_NOT_NULL(current);
  ASSERT_NOT_NULL(bias);
  ASSERT_INT_EQ(poly_buffer_write(ctx, current->uop_physical, values, sizeof(values)), 0);
  ASSERT_INT_EQ(poly_buffer_write(ctx, bias->uop_physical, bias_values, sizeof(bias_values)), 0);
  ASSERT_INT_EQ(run_logical_never_realized_steps(ctx, &current, bias, 32), 0);
  ASSERT_INT_EQ(poly_ctx_collect(ctx), 0);

  PolyCtxStats baseline = {0}, deferred = {0}, after = {0};
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &baseline), 0);
  ASSERT_INT_EQ(run_logical_never_realized_steps(ctx, &current, bias, 128), 0);
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &deferred), 0);
  ASSERT_TRUE(deferred.arena_bytes <= baseline.arena_bytes + POLY_IR_COLLECTION_MIN_GROWTH);
  ASSERT_INT_EQ(deferred.buffer_entries, baseline.buffer_entries);
  ASSERT_INT_EQ(poly_ctx_collect(ctx), 0);
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &after), 0);
  ASSERT_INT_EQ(after.arena_bytes, baseline.arena_bytes);
  ASSERT_INT_EQ(after.cse_entries, baseline.cse_entries);
  ASSERT_INT_EQ(after.shape_cache_entries, baseline.shape_cache_entries);
  ASSERT_INT_EQ(after.buffer_entries, baseline.buffer_entries);

  poly_tensor_release(current);
  poly_tensor_release(bias);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(tensor, realization_safe_point_bounds_transient_ir_without_stats) {
  /* Tinygrad 2026-08-22 weak UOps bound unreachable eager iterations. The C
   * safe point enforces the same bound without requiring a stats query. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  ASSERT_INT_EQ(poly_ctx_set_logical_policy(ctx, POLY_LOGICAL_NEVER), 0);
  poly_ctx_set_preferred_device(ctx, POLY_DEVICE_INTERP);
  int64_t shape[1] = {4};
  float values[4] = {1, 2, 3, 4};
  float bias_values[4] = {.25f, .25f, .25f, .25f};
  PolyTensor *current = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_INTERP);
  PolyTensor *bias = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_INTERP);
  ASSERT_NOT_NULL(current);
  ASSERT_NOT_NULL(bias);
  ASSERT_INT_EQ(poly_buffer_write(ctx, current->uop_physical, values, sizeof(values)), 0);
  ASSERT_INT_EQ(poly_buffer_write(ctx, bias->uop_physical, bias_values, sizeof(bias_values)), 0);
  ASSERT_INT_EQ(run_logical_never_realized_steps(ctx, &current, bias, 32), 0);
  ASSERT_INT_EQ(poly_ctx_collect(ctx), 0);
  size_t steady_uop_bytes = ctx->uop_storage_bytes;

  for (int i = 0; i < 512; i++) {
    ASSERT_INT_EQ(run_logical_never_realized_steps(ctx, &current, bias, 1), 0);
    ASSERT_TRUE(ctx->uop_storage_bytes <= steady_uop_bytes + POLY_IR_COLLECTION_MIN_GROWTH);
  }
  ASSERT_INT_EQ(poly_ctx_collect(ctx), 0);
  ASSERT_INT_EQ(ctx->uop_storage_bytes, steady_uop_bytes);

  poly_tensor_release(current);
  poly_tensor_release(bias);
  poly_ctx_destroy(ctx);
  PASS();
}

static PolyTensor *build_logical_policy_movement_oracle(
    PolyCtx *ctx,
    PolyLogicalPolicy policy,
    float values[4]
) {
  int64_t source_shape[1] = {4};
  int64_t matrix_shape[2] = {2, 2};
  int64_t flat_shape[2] = {1, 4};
  int64_t expanded_shape[2] = {2, 4};
  int64_t perm[2] = {1, 0};
  int64_t shrink[2][2] = {{1, 3}, {0, 2}};
  int64_t flip[1] = {0};
  int64_t pad[2][2] = {{1, 0}, {0, 1}};
  if (poly_ctx_set_logical_policy(ctx, policy) != 0) return NULL;
  PolyTensor *x = poly_tensor_empty(ctx, POLY_FLOAT32, source_shape, 1, POLY_DEVICE_CPU);
  if (!x || poly_buffer_write(ctx, poly_tensor_uop_physical(x), values, sizeof(float) * 4) != 0)
    return NULL;
  PolyTensor *matrix = poly_tensor_reshape(ctx, x, matrix_shape, 2);
  PolyTensor *flat = matrix ? poly_tensor_reshape(ctx, matrix, flat_shape, 2) : NULL;
  PolyTensor *expanded = flat ? poly_tensor_expand(ctx, flat, expanded_shape, 2) : NULL;
  PolyTensor *permuted = expanded ? poly_tensor_permute(ctx, expanded, perm, 2) : NULL;
  PolyTensor *shrunk = permuted ? poly_tensor_shrink(ctx, permuted, shrink, 2) : NULL;
  PolyTensor *flipped = shrunk ? poly_tensor_flip(ctx, shrunk, flip, 1) : NULL;
  PolyTensor *padded = flipped ? poly_tensor_pad_value_float(ctx, flipped, pad, 2, 0.0) : NULL;
  PolyTensor *wide =
      padded ? poly_tensor_cast_by_id(ctx, padded, poly_dtype_id_by_name("float64")) : NULL;
  PolyTensor *narrow =
      wide ? poly_tensor_cast_by_id(ctx, wide, poly_dtype_id_by_name("float32")) : NULL;
  PolyTensor *bits =
      narrow ? poly_tensor_bitcast_by_id(ctx, narrow, poly_dtype_id_by_name("uint32")) : NULL;
  PolyTensor *restored =
      bits ? poly_tensor_bitcast_by_id(ctx, bits, poly_dtype_id_by_name("float32")) : NULL;
  PolyTensor *contiguous = restored ? poly_tensor_contiguous(ctx, restored) : NULL;
  return contiguous ? poly_tensor_to_device(ctx, contiguous, POLY_DEVICE_INTERP) : NULL;
}

TEST(tensor, logical_never_movement_keeps_physical_graph_and_values_exact) {
  float values[4] = {1, 2, 3, 4};
  const float expected[9] = {0, 0, 0, 3, 3, 0, 2, 2, 0};
  PolyCtx *always_ctx = poly_ctx_new(), *never_ctx = poly_ctx_new();
  ASSERT_NOT_NULL(always_ctx);
  ASSERT_NOT_NULL(never_ctx);
  PolyTensor *always =
      build_logical_policy_movement_oracle(always_ctx, POLY_LOGICAL_ALWAYS, values);
  PolyTensor *never = build_logical_policy_movement_oracle(never_ctx, POLY_LOGICAL_NEVER, values);
  ASSERT_NOT_NULL(always);
  ASSERT_NOT_NULL(never);
  ASSERT_NOT_NULL(poly_tensor_uop_logical(always));
  ASSERT_PTR_EQ(poly_tensor_uop_logical(never), NULL);
  ASSERT_TRUE(uop_graph_isomorphic(
      always_ctx, poly_tensor_uop_physical(always), never_ctx, poly_tensor_uop_physical(never)
  ));

  float always_values[9] = {0}, never_values[9] = {0};
  ASSERT_INT_EQ(read_tensor_f32(always_ctx, always, always_values, 9), 0);
  ASSERT_INT_EQ(read_tensor_f32(never_ctx, never, never_values, 9), 0);
  for (int i = 0; i < 9; i++) {
    ASSERT_FLOAT_EQ(always_values[i], expected[i], 1e-6);
    ASSERT_FLOAT_EQ(never_values[i], always_values[i], 1e-6);
  }
  poly_ctx_destroy(always_ctx);
  poly_ctx_destroy(never_ctx);
  PASS();
}

static PolyTensor *build_logical_policy_effect_oracle(
    PolyCtx *ctx,
    PolyLogicalPolicy policy,
    float source_values[2],
    float target_values[2]
) {
  int64_t shape[1] = {2};
  if (poly_ctx_set_logical_policy(ctx, policy) != 0) return NULL;
  PolyTensor *source = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_CPU);
  PolyTensor *target = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_CPU);
  if (!source || !target ||
      poly_buffer_write(ctx, poly_tensor_uop_physical(source), source_values, sizeof(float) * 2) !=
          0 ||
      poly_buffer_write(ctx, poly_tensor_uop_physical(target), target_values, sizeof(float) * 2) !=
          0)
    return NULL;
  PolyTensor *moved_source = poly_tensor_to_device(ctx, source, POLY_DEVICE_INTERP);
  PolyTensor *moved_target = poly_tensor_to_device(ctx, target, POLY_DEVICE_INTERP);
  if (!moved_source || !moved_target || !poly_tensor_assign(ctx, moved_target, moved_source))
    return NULL;
  PolyTensor *clone_target = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_CPU);
  return clone_target ? poly_tensor_clone_into(ctx, clone_target, moved_target) : NULL;
}

TEST(tensor, logical_never_effects_keep_physical_graph_and_values_exact) {
  float source_values[2] = {5, 6}, target_values[2] = {1, 2};
  PolyCtx *always_ctx = poly_ctx_new(), *never_ctx = poly_ctx_new();
  ASSERT_NOT_NULL(always_ctx);
  ASSERT_NOT_NULL(never_ctx);
  PolyTensor *always = build_logical_policy_effect_oracle(
      always_ctx, POLY_LOGICAL_ALWAYS, source_values, target_values
  );
  PolyTensor *never = build_logical_policy_effect_oracle(
      never_ctx, POLY_LOGICAL_NEVER, source_values, target_values
  );
  ASSERT_NOT_NULL(always);
  ASSERT_NOT_NULL(never);
  ASSERT_NOT_NULL(poly_tensor_uop_logical(always));
  ASSERT_PTR_EQ(poly_tensor_uop_logical(never), NULL);
  ASSERT_TRUE(uop_graph_isomorphic(
      always_ctx, poly_tensor_uop_physical(always), never_ctx, poly_tensor_uop_physical(never)
  ));
  float always_values[2] = {0}, never_values[2] = {0};
  ASSERT_INT_EQ(read_tensor_f32(always_ctx, always, always_values, 2), 0);
  ASSERT_INT_EQ(read_tensor_f32(never_ctx, never, never_values, 2), 0);
  for (int i = 0; i < 2; i++) {
    ASSERT_FLOAT_EQ(always_values[i], source_values[i], 1e-6);
    ASSERT_FLOAT_EQ(never_values[i], always_values[i], 1e-6);
  }
  poly_ctx_destroy(always_ctx);
  poly_ctx_destroy(never_ctx);
  PASS();
}

static PolyTensor *build_logical_policy_view_assign_oracle(
    PolyCtx *ctx,
    PolyLogicalPolicy policy,
    float base_values[4],
    float update_values[2]
) {
  int64_t base_shape[1] = {4}, update_shape[1] = {2};
  int64_t bounds[1][2] = {{1, 3}};
  if (poly_ctx_set_logical_policy(ctx, policy) != 0) return NULL;
  PolyTensor *base = poly_tensor_empty(ctx, POLY_FLOAT32, base_shape, 1, POLY_DEVICE_CPU);
  PolyTensor *update = poly_tensor_empty(ctx, POLY_FLOAT32, update_shape, 1, POLY_DEVICE_CPU);
  if (!base || !update ||
      poly_buffer_write(ctx, poly_tensor_uop_physical(base), base_values, sizeof(float) * 4) != 0 ||
      poly_buffer_write(ctx, poly_tensor_uop_physical(update), update_values, sizeof(float) * 2) !=
          0)
    return NULL;
  PolyTensor *view = poly_tensor_shrink(ctx, base, bounds, 1);
  return view && poly_tensor_assign(ctx, view, update) ? base : NULL;
}

TEST(tensor, logical_never_view_assign_retargets_only_physical_graph) {
  float base_values[4] = {1, 2, 3, 4}, update_values[2] = {8, 9};
  const float expected[4] = {1, 8, 9, 4};
  PolyCtx *always_ctx = poly_ctx_new(), *never_ctx = poly_ctx_new();
  ASSERT_NOT_NULL(always_ctx);
  ASSERT_NOT_NULL(never_ctx);
  PolyTensor *always = build_logical_policy_view_assign_oracle(
      always_ctx, POLY_LOGICAL_ALWAYS, base_values, update_values
  );
  PolyTensor *never = build_logical_policy_view_assign_oracle(
      never_ctx, POLY_LOGICAL_NEVER, base_values, update_values
  );
  ASSERT_NOT_NULL(always);
  ASSERT_NOT_NULL(never);
  ASSERT_PTR_EQ(poly_tensor_uop_logical(never), NULL);
  ASSERT_TRUE(uop_graph_isomorphic(
      always_ctx, poly_tensor_uop_physical(always), never_ctx, poly_tensor_uop_physical(never)
  ));
  float always_values[4] = {0}, never_values[4] = {0};
  ASSERT_INT_EQ(read_tensor_f32(always_ctx, always, always_values, 4), 0);
  ASSERT_INT_EQ(read_tensor_f32(never_ctx, never, never_values, 4), 0);
  for (int i = 0; i < 4; i++) {
    ASSERT_FLOAT_EQ(always_values[i], expected[i], 1e-6);
    ASSERT_FLOAT_EQ(never_values[i], always_values[i], 1e-6);
  }
  poly_ctx_destroy(always_ctx);
  poly_ctx_destroy(never_ctx);
  PASS();
}

static PolyTensor *build_logical_policy_reduction_oracle(
    PolyCtx *ctx,
    PolyLogicalPolicy policy,
    float values[4],
    float weights[2]
) {
  int64_t flat_matrix_shape[1] = {4}, flat_weight_shape[1] = {2};
  int64_t matrix_shape[2] = {2, 2}, weight_shape[2] = {1, 2}, axis[1] = {1};
  if (poly_ctx_set_logical_policy(ctx, policy) != 0) return NULL;
  PolyTensor *x_flat = poly_tensor_empty(ctx, POLY_FLOAT32, flat_matrix_shape, 1, POLY_DEVICE_CPU);
  PolyTensor *w_flat = poly_tensor_empty(ctx, POLY_FLOAT32, flat_weight_shape, 1, POLY_DEVICE_CPU);
  if (!x_flat || !w_flat ||
      poly_buffer_write(ctx, poly_tensor_uop_physical(x_flat), values, sizeof(float) * 4) != 0 ||
      poly_buffer_write(ctx, poly_tensor_uop_physical(w_flat), weights, sizeof(float) * 2) != 0)
    return NULL;
  PolyTensor *x = poly_tensor_reshape(ctx, x_flat, matrix_shape, 2);
  PolyTensor *w = poly_tensor_reshape(ctx, w_flat, weight_shape, 2);
  if (!x || !w) return NULL;

  PolyTensor *u = poly_tensor_exp(ctx, x);
  u = u ? poly_tensor_log(ctx, u) : NULL;
  u = u ? poly_tensor_cos(ctx, u) : NULL;
  u = u ? poly_tensor_tan(ctx, u) : NULL;
  u = u ? poly_tensor_log1p(ctx, u) : NULL;
  u = u ? poly_tensor_expm1(ctx, u) : NULL;
  u = u ? poly_tensor_gelu(ctx, u) : NULL;
  u = u ? poly_tensor_relu(ctx, u) : NULL;
  u = u ? poly_tensor_sigmoid(ctx, u) : NULL;
  u = u ? poly_tensor_tanh(ctx, u) : NULL;
  u = u ? poly_tensor_silu(ctx, u) : NULL;
  u = u ? poly_tensor_quick_gelu(ctx, u) : NULL;
  PolyTensor *sum = u ? poly_tensor_sum(ctx, u, axis, 1, true) : NULL;
  PolyTensor *max = u ? poly_tensor_max(ctx, u, axis, 1, true) : NULL;
  PolyTensor *minimum = sum && max ? poly_tensor_minimum(ctx, sum, max) : NULL;
  PolyTensor *dot = minimum ? poly_tensor_dot(ctx, minimum, w) : NULL;
  PolyTensor *softmax = dot ? poly_tensor_softmax(ctx, dot, 1) : NULL;
  PolyTensor *log_softmax = softmax ? poly_tensor_log_softmax(ctx, softmax, 1) : NULL;
  return log_softmax ? poly_tensor_detach(ctx, log_softmax) : NULL;
}

TEST(tensor, logical_never_reductions_keep_physical_graph_and_values_exact) {
  float values[4] = {0.1f, 0.2f, 0.3f, 0.4f}, weights[2] = {1.0f, 2.0f};
  PolyCtx *always_ctx = poly_ctx_new(), *never_ctx = poly_ctx_new();
  ASSERT_NOT_NULL(always_ctx);
  ASSERT_NOT_NULL(never_ctx);
  PolyTensor *always =
      build_logical_policy_reduction_oracle(always_ctx, POLY_LOGICAL_ALWAYS, values, weights);
  PolyTensor *never =
      build_logical_policy_reduction_oracle(never_ctx, POLY_LOGICAL_NEVER, values, weights);
  ASSERT_NOT_NULL(always);
  ASSERT_NOT_NULL(never);
  ASSERT_PTR_EQ(poly_tensor_uop_logical(never), NULL);
  ASSERT_TRUE(uop_graph_isomorphic(
      always_ctx, poly_tensor_uop_physical(always), never_ctx, poly_tensor_uop_physical(never)
  ));
  float always_values[4] = {0}, never_values[4] = {0};
  ASSERT_INT_EQ(read_tensor_f32(always_ctx, always, always_values, 4), 0);
  ASSERT_INT_EQ(read_tensor_f32(never_ctx, never, never_values, 4), 0);
  for (int i = 0; i < 4; i++) {
    ASSERT_TRUE(isfinite(always_values[i]));
    ASSERT_FLOAT_EQ(never_values[i], always_values[i], 1e-6);
  }
  poly_ctx_destroy(always_ctx);
  poly_ctx_destroy(never_ctx);
  PASS();
}

static PolyTensor *build_logical_policy_custom_kernel_oracle(
    PolyCtx *ctx,
    PolyLogicalPolicy policy,
    float a_values[4],
    float b_values[4]
) {
  int64_t shape[1] = {4};
  if (poly_ctx_set_logical_policy(ctx, policy) != 0) return NULL;
  PolyTensor *c = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_CPU);
  PolyTensor *a = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_CPU);
  PolyTensor *b = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_CPU);
  if (!c || !a || !b ||
      poly_buffer_write(ctx, poly_tensor_uop_physical(a), a_values, sizeof(float) * 4) != 0 ||
      poly_buffer_write(ctx, poly_tensor_uop_physical(b), b_values, sizeof(float) * 4) != 0)
    return NULL;
  PolyUOp *pc = poly_uop_placeholder_like(ctx, c->uop_physical, 0);
  PolyUOp *pa = poly_uop_placeholder_like(ctx, a->uop_physical, 1);
  PolyUOp *pb = poly_uop_placeholder_like(ctx, b->uop_physical, 2);
  PolyUOp *r = poly_uop_range(ctx, 4, 0, POLY_AXIS_LOOP);
  PolyUOp *idxs[1] = {r};
  PolyUOp *ci = poly_uop_index(ctx, pc, idxs, 1);
  PolyUOp *ai = poly_uop_index(ctx, pa, idxs, 1);
  PolyUOp *bi = poly_uop_index(ctx, pb, idxs, 1);
  PolyUOp *sum = poly_alu2(ctx, POLY_OP_ADD, poly_uop_load(ctx, ai), poly_uop_load(ctx, bi));
  PolyUOp *store = poly_uop_store(ctx, ci, sum);
  PolyUOp *end = poly_uop_end(ctx, store, &r, 1);
  PolyUOp *body = poly_uop_sink_ex(ctx, &end, 1, "logical_policy_custom_add", 1);
  PolyTensor *inputs[3] = {c, a, b}, *outputs[3] = {0};
  return body && poly_tensor_custom_kernel(ctx, body, inputs, 3, 0, outputs) == 0 ? outputs[0]
                                                                                  : NULL;
}

TEST(tensor, logical_never_custom_kernel_keeps_physical_graph_and_values_exact) {
  float a_values[4] = {1, 2, 3, 4}, b_values[4] = {5, 6, 7, 8};
  const float expected[4] = {6, 8, 10, 12};
  PolyCtx *always_ctx = poly_ctx_new(), *never_ctx = poly_ctx_new();
  ASSERT_NOT_NULL(always_ctx);
  ASSERT_NOT_NULL(never_ctx);
  PolyTensor *always = build_logical_policy_custom_kernel_oracle(
      always_ctx, POLY_LOGICAL_ALWAYS, a_values, b_values
  );
  PolyTensor *never =
      build_logical_policy_custom_kernel_oracle(never_ctx, POLY_LOGICAL_NEVER, a_values, b_values);
  ASSERT_NOT_NULL(always);
  ASSERT_NOT_NULL(never);
  ASSERT_PTR_EQ(poly_tensor_uop_logical(never), NULL);
  ASSERT_TRUE(uop_graph_isomorphic(
      always_ctx, poly_tensor_uop_physical(always), never_ctx, poly_tensor_uop_physical(never)
  ));
  float always_values[4] = {0}, never_values[4] = {0};
  ASSERT_INT_EQ(read_tensor_f32(always_ctx, always, always_values, 4), 0);
  ASSERT_INT_EQ(read_tensor_f32(never_ctx, never, never_values, 4), 0);
  for (int i = 0; i < 4; i++) {
    ASSERT_FLOAT_EQ(always_values[i], expected[i], 1e-6);
    ASSERT_FLOAT_EQ(never_values[i], always_values[i], 1e-6);
  }
  poly_ctx_destroy(always_ctx);
  poly_ctx_destroy(never_ctx);
  PASS();
}

static PolyTensor *build_logical_policy_function_oracle(
    PolyCtx *ctx,
    PolyLogicalPolicy policy,
    float a_values[2],
    float b_values[2]
) {
  int64_t shape[1] = {2};
  if (poly_ctx_set_logical_policy(ctx, policy) != 0) return NULL;
  PolyTensor *a = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_CPU);
  PolyTensor *b = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_CPU);
  if (!a || !b ||
      poly_buffer_write(ctx, poly_tensor_uop_physical(a), a_values, sizeof(float) * 2) != 0 ||
      poly_buffer_write(ctx, poly_tensor_uop_physical(b), b_values, sizeof(float) * 2) != 0)
    return NULL;
  PolyTensor *sum = poly_tensor_alu2(ctx, POLY_OP_ADD, a, b);
  if (!sum) return NULL;
  PolyTensor *results[1] = {sum}, *outputs[1] = {NULL};
  PolyUOp *logical_inputs[2] = {a->uop_logical, b->uop_logical};
  PolyUOp *physical_inputs[2] = {a->uop_physical, b->uop_physical};
  PolyUOp **logical = policy == POLY_LOGICAL_NEVER ? NULL : logical_inputs;
  return poly_tensor_function(
             ctx, results, 1, logical, physical_inputs, 2, "logical_policy_add", false, false,
             false, outputs
         ) == 0
             ? outputs[0]
             : NULL;
}

TEST(tensor, logical_never_function_keeps_physical_graph_and_values_exact) {
  float a_values[2] = {1, 2}, b_values[2] = {3, 4};
  PolyCtx *always_ctx = poly_ctx_new(), *never_ctx = poly_ctx_new();
  ASSERT_NOT_NULL(always_ctx);
  ASSERT_NOT_NULL(never_ctx);
  PolyTensor *always =
      build_logical_policy_function_oracle(always_ctx, POLY_LOGICAL_ALWAYS, a_values, b_values);
  PolyTensor *never =
      build_logical_policy_function_oracle(never_ctx, POLY_LOGICAL_NEVER, a_values, b_values);
  ASSERT_NOT_NULL(always);
  ASSERT_NOT_NULL(never);
  ASSERT_PTR_EQ(poly_tensor_uop_logical(never), NULL);
  ASSERT_TRUE(uop_graph_isomorphic(
      always_ctx, poly_tensor_uop_physical(always), never_ctx, poly_tensor_uop_physical(never)
  ));
  float always_values[2] = {0}, never_values[2] = {0};
  ASSERT_INT_EQ(read_tensor_f32(always_ctx, always, always_values, 2), 0);
  ASSERT_INT_EQ(read_tensor_f32(never_ctx, never, never_values, 2), 0);
  ASSERT_FLOAT_EQ(always_values[0], 4.0f, 1e-6);
  ASSERT_FLOAT_EQ(always_values[1], 6.0f, 1e-6);
  ASSERT_FLOAT_EQ(never_values[0], always_values[0], 1e-6);
  ASSERT_FLOAT_EQ(never_values[1], always_values[1], 1e-6);
  poly_ctx_destroy(always_ctx);
  poly_ctx_destroy(never_ctx);
  PASS();
}

static PolyTensor *build_logical_policy_rng_oracle(PolyCtx *ctx, PolyLogicalPolicy policy) {
  int64_t shape[1] = {4};
  int f32 = poly_dtype_id_by_name("float32");
  if (poly_ctx_set_logical_policy(ctx, policy) != 0 || f32 < 0) return NULL;
  poly_tensor_manual_seed(ctx, 123);
  PolyTensor *first = poly_tensor_rand_by_id(ctx, shape, 1, f32, POLY_DEVICE_CPU, 1);
  PolyTensor *second = poly_tensor_rand_by_id(ctx, shape, 1, f32, POLY_DEVICE_CPU, 1);
  return first && second ? poly_tensor_alu2(ctx, POLY_OP_ADD, first, second) : NULL;
}

TEST(tensor, logical_never_rng_keeps_physical_graph_state_and_values_exact) {
  PolyCtx *always_ctx = poly_ctx_new(), *never_ctx = poly_ctx_new();
  ASSERT_NOT_NULL(always_ctx);
  ASSERT_NOT_NULL(never_ctx);
  PolyTensor *always = build_logical_policy_rng_oracle(always_ctx, POLY_LOGICAL_ALWAYS);
  PolyTensor *never = build_logical_policy_rng_oracle(never_ctx, POLY_LOGICAL_NEVER);
  ASSERT_NOT_NULL(always);
  ASSERT_NOT_NULL(never);
  ASSERT_PTR_EQ(poly_tensor_uop_logical(never), NULL);
  ASSERT_TRUE(uop_graph_isomorphic(
      always_ctx, poly_tensor_uop_physical(always), never_ctx, poly_tensor_uop_physical(never)
  ));
  float always_values[4] = {0}, never_values[4] = {0};
  ASSERT_INT_EQ(read_tensor_f32(always_ctx, always, always_values, 4), 0);
  ASSERT_INT_EQ(read_tensor_f32(never_ctx, never, never_values, 4), 0);
  for (int i = 0; i < 4; i++)
    ASSERT_FLOAT_EQ(never_values[i], always_values[i], 1e-6);
  poly_ctx_destroy(always_ctx);
  poly_ctx_destroy(never_ctx);
  PASS();
}

TEST(tensor, logical_until_realize_retires_effect_custom_function_and_rng_outputs) {
  float source_values[2] = {5, 6}, target_values[2] = {1, 2};
  float a4[4] = {1, 2, 3, 4}, b4[4] = {5, 6, 7, 8};
  float a2[2] = {1, 2}, b2[2] = {3, 4};
  PolyCtx *contexts[4] = {poly_ctx_new(), poly_ctx_new(), poly_ctx_new(), poly_ctx_new()};
  for (int i = 0; i < 4; i++)
    ASSERT_NOT_NULL(contexts[i]);
  PolyTensor *outputs[4] = {
      build_logical_policy_effect_oracle(
          contexts[0], POLY_LOGICAL_UNTIL_REALIZE, source_values, target_values
      ),
      build_logical_policy_custom_kernel_oracle(contexts[1], POLY_LOGICAL_UNTIL_REALIZE, a4, b4),
      build_logical_policy_function_oracle(contexts[2], POLY_LOGICAL_UNTIL_REALIZE, a2, b2),
      build_logical_policy_rng_oracle(contexts[3], POLY_LOGICAL_UNTIL_REALIZE),
  };

  for (int i = 0; i < 4; i++) {
    ASSERT_NOT_NULL(outputs[i]);
    PolyUOp *producer = outputs[i]->uop_logical;
    ASSERT_NOT_NULL(producer);
    PolyTensor *realized = NULL;
    ASSERT_INT_EQ(poly_realize_tensors(contexts[i], &outputs[i], 1, &realized), 0);
    ASSERT_PTR_EQ(realized, outputs[i]);
    ASSERT_INT_EQ(outputs[i]->logical_state, POLY_LOGICAL_RETIRED);
    ASSERT_PTR_NEQ(outputs[i]->uop_logical, producer);
    ASSERT_FALSE(poly_uop_reachable(contexts[i], outputs[i]->uop_logical, producer));
    ASSERT_NOT_NULL(poly_uop_get_buffer_identity(outputs[i]->uop_physical));
  }

  for (int i = 0; i < 4; i++)
    poly_ctx_destroy(contexts[i]);
  PASS();
}

static PolyTensor *logical_policy_tensor_from_f32(
    PolyCtx *ctx,
    const float *values,
    int64_t numel,
    int64_t *shape,
    int ndim
) {
  int64_t flat_shape[1] = {numel};
  PolyTensor *flat = poly_tensor_empty(ctx, POLY_FLOAT32, flat_shape, 1, POLY_DEVICE_CPU);
  if (!flat || poly_buffer_write(
                   ctx, poly_tensor_uop_physical(flat), values, (size_t)numel * sizeof(float)
               ) != 0)
    return NULL;
  return ndim == 1 && shape[0] == numel ? flat : poly_tensor_reshape(ctx, flat, shape, ndim);
}

static PolyTensor *build_logical_policy_nn_oracle(PolyCtx *ctx, PolyLogicalPolicy policy) {
  float x_values[4] = {1, 2, 3, 4}, weight_values[4] = {1, 0, 0, 1};
  float bias_values[2] = {0.5f, -0.5f}, scale_values[2] = {1.25f, 0.75f};
  int64_t matrix_shape[2] = {2, 2}, vector_shape[1] = {2};
  if (poly_ctx_set_logical_policy(ctx, policy) != 0) return NULL;
  PolyTensor *x = logical_policy_tensor_from_f32(ctx, x_values, 4, matrix_shape, 2);
  PolyTensor *weight = logical_policy_tensor_from_f32(ctx, weight_values, 4, matrix_shape, 2);
  PolyTensor *bias = logical_policy_tensor_from_f32(ctx, bias_values, 2, vector_shape, 1);
  PolyTensor *scale = logical_policy_tensor_from_f32(ctx, scale_values, 2, vector_shape, 1);
  PolyTensor *linear = x && weight && bias ? poly_tensor_linear_apply(ctx, x, weight, bias) : NULL;
  PolyTensor *layernorm = linear && scale && bias
                              ? poly_tensor_layernorm_apply(ctx, linear, scale, bias, -1, 1e-5)
                              : NULL;
  return layernorm && scale ? poly_tensor_rmsnorm_apply(ctx, layernorm, scale, 1e-5) : NULL;
}

TEST(tensor, logical_never_nn_layers_keep_physical_graph_and_values_exact) {
  PolyCtx *always_ctx = poly_ctx_new(), *never_ctx = poly_ctx_new();
  ASSERT_NOT_NULL(always_ctx);
  ASSERT_NOT_NULL(never_ctx);
  PolyTensor *always = build_logical_policy_nn_oracle(always_ctx, POLY_LOGICAL_ALWAYS);
  PolyTensor *never = build_logical_policy_nn_oracle(never_ctx, POLY_LOGICAL_NEVER);
  ASSERT_NOT_NULL(always);
  ASSERT_NOT_NULL(never);
  ASSERT_PTR_EQ(poly_tensor_uop_logical(never), NULL);
  ASSERT_TRUE(uop_graph_isomorphic(
      always_ctx, poly_tensor_uop_physical(always), never_ctx, poly_tensor_uop_physical(never)
  ));
  float always_values[4] = {0}, never_values[4] = {0};
  ASSERT_INT_EQ(read_tensor_f32(always_ctx, always, always_values, 4), 0);
  ASSERT_INT_EQ(read_tensor_f32(never_ctx, never, never_values, 4), 0);
  for (int i = 0; i < 4; i++) {
    ASSERT_TRUE(isfinite(always_values[i]));
    ASSERT_FLOAT_EQ(never_values[i], always_values[i], 1e-5);
  }
  poly_ctx_destroy(always_ctx);
  poly_ctx_destroy(never_ctx);
  PASS();
}

static PolyTensor *build_logical_policy_conv_oracle(PolyCtx *ctx, PolyLogicalPolicy policy) {
  float x_values[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  float weight_values[4] = {1, 1, 1, 1};
  float zero[1] = {0}, one[1] = {1};
  int64_t x_shape[4] = {1, 1, 3, 3}, weight_shape[4] = {1, 1, 2, 2};
  int64_t channel_shape[1] = {1}, channel_axis[1] = {1}, pool_kernel[2] = {2, 2};
  if (poly_ctx_set_logical_policy(ctx, policy) != 0) return NULL;
  PolyTensor *x = logical_policy_tensor_from_f32(ctx, x_values, 9, x_shape, 4);
  PolyTensor *weight = logical_policy_tensor_from_f32(ctx, weight_values, 4, weight_shape, 4);
  PolyTensor *bias = logical_policy_tensor_from_f32(ctx, zero, 1, channel_shape, 1);
  PolyTensor *mean = logical_policy_tensor_from_f32(ctx, zero, 1, channel_shape, 1);
  PolyTensor *invstd = logical_policy_tensor_from_f32(ctx, one, 1, channel_shape, 1);
  PolyTensor *scale = logical_policy_tensor_from_f32(ctx, one, 1, channel_shape, 1);
  PolyTensor *conv =
      x && weight && bias ? poly_tensor_conv2d(ctx, x, weight, bias, 1, NULL, NULL, NULL, 0) : NULL;
  PolyTensor *bn =
      conv && mean && invstd && scale && bias
          ? poly_tensor_batchnorm(ctx, conv, scale, bias, mean, invstd, channel_axis, 1)
          : NULL;
  return bn ? poly_tensor_max_pool2d(ctx, bn, pool_kernel, 2, NULL, NULL, NULL, 0, false, NULL)
            : NULL;
}

TEST(tensor, logical_never_conv_batchnorm_pool_keep_physical_graph_and_values_exact) {
  PolyCtx *always_ctx = poly_ctx_new(), *never_ctx = poly_ctx_new();
  ASSERT_NOT_NULL(always_ctx);
  ASSERT_NOT_NULL(never_ctx);
  PolyTensor *always = build_logical_policy_conv_oracle(always_ctx, POLY_LOGICAL_ALWAYS);
  PolyTensor *never = build_logical_policy_conv_oracle(never_ctx, POLY_LOGICAL_NEVER);
  ASSERT_NOT_NULL(always);
  ASSERT_NOT_NULL(never);
  ASSERT_PTR_EQ(poly_tensor_uop_logical(never), NULL);
  ASSERT_TRUE(uop_graph_isomorphic(
      always_ctx, poly_tensor_uop_physical(always), never_ctx, poly_tensor_uop_physical(never)
  ));
  float always_value[1] = {0}, never_value[1] = {0};
  ASSERT_INT_EQ(read_tensor_f32(always_ctx, always, always_value, 1), 0);
  ASSERT_INT_EQ(read_tensor_f32(never_ctx, never, never_value, 1), 0);
  ASSERT_FLOAT_EQ(always_value[0], 28.0f, 1e-5);
  ASSERT_FLOAT_EQ(never_value[0], always_value[0], 1e-5);
  poly_ctx_destroy(always_ctx);
  poly_ctx_destroy(never_ctx);
  PASS();
}

static PolyTensor *build_logical_policy_grad_oracle(PolyCtx *ctx, PolyLogicalPolicy policy) {
  float values[2] = {1, 2};
  int64_t shape[1] = {2}, axis[1] = {0};
  if (poly_ctx_set_logical_policy(ctx, policy) != 0) return NULL;
  PolyTensor *x = logical_policy_tensor_from_f32(ctx, values, 2, shape, 1);
  PolyTensor *square = x ? poly_tensor_alu2(ctx, POLY_OP_MUL, x, x) : NULL;
  PolyTensor *loss = square ? poly_tensor_sum(ctx, square, axis, 1, false) : NULL;
  if (!x || !loss) return NULL;
  PolyUOp *physical = poly_grad(ctx, loss->uop_physical, x->uop_physical);
  PolyUOp *logical =
      policy == POLY_LOGICAL_NEVER ? NULL : poly_grad(ctx, loss->uop_logical, x->uop_logical);
  return physical ? poly_tensor_create_with_roots(
                        ctx, logical, physical, POLY_TENSOR_VALUE, POLY_DEVICE_CPU
                    )
                  : NULL;
}

TEST(tensor, logical_never_autograd_keeps_physical_graph_and_values_exact) {
  PolyCtx *always_ctx = poly_ctx_new(), *never_ctx = poly_ctx_new();
  ASSERT_NOT_NULL(always_ctx);
  ASSERT_NOT_NULL(never_ctx);
  PolyTensor *always = build_logical_policy_grad_oracle(always_ctx, POLY_LOGICAL_ALWAYS);
  PolyTensor *never = build_logical_policy_grad_oracle(never_ctx, POLY_LOGICAL_NEVER);
  ASSERT_NOT_NULL(always);
  ASSERT_NOT_NULL(never);
  ASSERT_PTR_EQ(poly_tensor_uop_logical(never), NULL);
  ASSERT_TRUE(uop_graph_isomorphic(
      always_ctx, poly_tensor_uop_physical(always), never_ctx, poly_tensor_uop_physical(never)
  ));
  float always_values[2] = {0}, never_values[2] = {0};
  ASSERT_INT_EQ(read_tensor_f32(always_ctx, always, always_values, 2), 0);
  ASSERT_INT_EQ(read_tensor_f32(never_ctx, never, never_values, 2), 0);
  ASSERT_FLOAT_EQ(always_values[0], 2.0f, 1e-6);
  ASSERT_FLOAT_EQ(always_values[1], 4.0f, 1e-6);
  ASSERT_FLOAT_EQ(never_values[0], always_values[0], 1e-6);
  ASSERT_FLOAT_EQ(never_values[1], always_values[1], 1e-6);
  poly_ctx_destroy(always_ctx);
  poly_ctx_destroy(never_ctx);
  PASS();
}

static PolyTensor *build_logical_policy_optimizer_oracle(PolyCtx *ctx, PolyLogicalPolicy policy) {
  float param_values[2] = {1, 2}, grad_values[2] = {2, 4}, lr_value[1] = {0.1f};
  int64_t vector_shape[1] = {2}, scalar_shape[1] = {1};
  if (poly_ctx_set_logical_policy(ctx, policy) != 0) return NULL;
  PolyTensor *param = logical_policy_tensor_from_f32(ctx, param_values, 2, vector_shape, 1);
  PolyTensor *grad = logical_policy_tensor_from_f32(ctx, grad_values, 2, vector_shape, 1);
  PolyTensor *lr = logical_policy_tensor_from_f32(ctx, lr_value, 1, scalar_shape, 1);
  PolyOptimConfig cfg = {
      .kind = POLY_OPTIM_SGD,
      .weight_decay = 0.0f,
      .momentum = 0.0f,
      .nesterov = false,
      .classic = false,
  };
  PolyTensor *outputs[1] = {NULL};
  if (!param || !grad || !lr ||
      poly_optim_build_step(ctx, &cfg, lr, &param, &grad, 1, NULL, NULL, NULL, NULL, outputs, 1) !=
          1)
    return NULL;
  return outputs[0];
}

TEST(tensor, logical_never_optimizer_keeps_physical_graph_and_values_exact) {
  PolyCtx *always_ctx = poly_ctx_new(), *never_ctx = poly_ctx_new();
  ASSERT_NOT_NULL(always_ctx);
  ASSERT_NOT_NULL(never_ctx);
  PolyTensor *always = build_logical_policy_optimizer_oracle(always_ctx, POLY_LOGICAL_ALWAYS);
  PolyTensor *never = build_logical_policy_optimizer_oracle(never_ctx, POLY_LOGICAL_NEVER);
  ASSERT_NOT_NULL(always);
  ASSERT_NOT_NULL(never);
  ASSERT_PTR_EQ(poly_tensor_uop_logical(never), NULL);
  ASSERT_TRUE(uop_graph_isomorphic(
      always_ctx, poly_tensor_uop_physical(always), never_ctx, poly_tensor_uop_physical(never)
  ));
  float always_values[2] = {0}, never_values[2] = {0};
  ASSERT_INT_EQ(read_tensor_f32(always_ctx, always, always_values, 2), 0);
  ASSERT_INT_EQ(read_tensor_f32(never_ctx, never, never_values, 2), 0);
  ASSERT_FLOAT_EQ(always_values[0], 0.8f, 1e-6);
  ASSERT_FLOAT_EQ(always_values[1], 1.6f, 1e-6);
  ASSERT_FLOAT_EQ(never_values[0], always_values[0], 1e-6);
  ASSERT_FLOAT_EQ(never_values[1], always_values[1], 1e-6);
  poly_ctx_destroy(always_ctx);
  poly_ctx_destroy(never_ctx);
  PASS();
}

TEST(tensor, optimizer_mixed_logical_policies_update_each_physical_target) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  float p0_values[2] = {1, 2}, p1_values[2] = {3, 4};
  float g0_values[2] = {2, 4}, g1_values[2] = {6, 8}, lr_value[1] = {0.1f};
  int64_t vector_shape[1] = {2}, scalar_shape[1] = {1};

  ASSERT_INT_EQ(poly_ctx_set_logical_policy(ctx, POLY_LOGICAL_ALWAYS), 0);
  PolyTensor *p0 = logical_policy_tensor_from_f32(ctx, p0_values, 2, vector_shape, 1);
  PolyTensor *g0 = logical_policy_tensor_from_f32(ctx, g0_values, 2, vector_shape, 1);
  PolyTensor *lr = logical_policy_tensor_from_f32(ctx, lr_value, 1, scalar_shape, 1);
  ASSERT_INT_EQ(poly_ctx_set_logical_policy(ctx, POLY_LOGICAL_NEVER), 0);
  PolyTensor *p1 = logical_policy_tensor_from_f32(ctx, p1_values, 2, vector_shape, 1);
  PolyTensor *g1 = logical_policy_tensor_from_f32(ctx, g1_values, 2, vector_shape, 1);
  ASSERT_NOT_NULL(p0);
  ASSERT_NOT_NULL(g0);
  ASSERT_NOT_NULL(lr);
  ASSERT_NOT_NULL(p1);
  ASSERT_NOT_NULL(g1);

  PolyOptimConfig cfg = {
      .kind = POLY_OPTIM_SGD,
      .weight_decay = 0.0f,
      .momentum = 0.0f,
      .nesterov = false,
      .classic = false,
  };
  PolyTensor *params[2] = {p0, p1}, *grads[2] = {g0, g1}, *outputs[2] = {NULL, NULL};
  ASSERT_INT_EQ(
      poly_optim_build_step(ctx, &cfg, lr, params, grads, 2, NULL, NULL, NULL, NULL, outputs, 2), 2
  );
  ASSERT_PTR_EQ(outputs[0], p0);
  ASSERT_PTR_EQ(outputs[1], p1);
  ASSERT_NOT_NULL(p0->uop_logical);
  ASSERT_INT_EQ(p0->uop_logical->op, POLY_OP_AFTER);
  ASSERT_PTR_EQ(p1->uop_logical, NULL);
  ASSERT_INT_EQ(p1->logical_state, POLY_LOGICAL_NEVER_CONSTRUCTED);
  ASSERT_INT_EQ(p0->uop_physical->op, POLY_OP_AFTER);
  ASSERT_INT_EQ(p1->uop_physical->op, POLY_OP_AFTER);

  float p0_got[2] = {0}, p1_got[2] = {0};
  ASSERT_INT_EQ(read_tensor_f32(ctx, p0, p0_got, 2), 0);
  ASSERT_INT_EQ(read_tensor_f32(ctx, p1, p1_got, 2), 0);
  ASSERT_FLOAT_EQ(p0_got[0], 0.8f, 1e-6f);
  ASSERT_FLOAT_EQ(p0_got[1], 1.6f, 1e-6f);
  ASSERT_FLOAT_EQ(p1_got[0], 2.4f, 1e-6f);
  ASSERT_FLOAT_EQ(p1_got[1], 3.2f, 1e-6f);

  poly_tensor_release(g1);
  poly_tensor_release(p1);
  poly_tensor_release(lr);
  poly_tensor_release(g0);
  poly_tensor_release(p0);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(tensor, adam_mixed_logical_policies_do_not_depend_on_first_parameter) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  float p0_value[1] = {1}, p1_value[1] = {3};
  float g0_value[1] = {2}, g1_value[1] = {6};
  float zero[1] = {0}, one[1] = {1}, lr_value[1] = {0.1f};
  int64_t shape[1] = {1};

  ASSERT_INT_EQ(poly_ctx_set_logical_policy(ctx, POLY_LOGICAL_ALWAYS), 0);
  PolyTensor *lr = logical_policy_tensor_from_f32(ctx, lr_value, 1, shape, 1);
  PolyTensor *bc1 = logical_policy_tensor_from_f32(ctx, one, 1, shape, 1);
  PolyTensor *bc2 = logical_policy_tensor_from_f32(ctx, one, 1, shape, 1);
  PolyTensor *p1 = logical_policy_tensor_from_f32(ctx, p1_value, 1, shape, 1);
  PolyTensor *g1 = logical_policy_tensor_from_f32(ctx, g1_value, 1, shape, 1);
  PolyTensor *m1 = logical_policy_tensor_from_f32(ctx, zero, 1, shape, 1);
  PolyTensor *v1 = logical_policy_tensor_from_f32(ctx, zero, 1, shape, 1);
  ASSERT_INT_EQ(poly_ctx_set_logical_policy(ctx, POLY_LOGICAL_NEVER), 0);
  PolyTensor *p0 = logical_policy_tensor_from_f32(ctx, p0_value, 1, shape, 1);
  PolyTensor *g0 = logical_policy_tensor_from_f32(ctx, g0_value, 1, shape, 1);
  PolyTensor *m0 = logical_policy_tensor_from_f32(ctx, zero, 1, shape, 1);
  PolyTensor *v0 = logical_policy_tensor_from_f32(ctx, zero, 1, shape, 1);
  ASSERT_NOT_NULL(lr);
  ASSERT_NOT_NULL(bc1);
  ASSERT_NOT_NULL(bc2);
  ASSERT_NOT_NULL(p0);
  ASSERT_NOT_NULL(g0);
  ASSERT_NOT_NULL(m0);
  ASSERT_NOT_NULL(v0);
  ASSERT_NOT_NULL(p1);
  ASSERT_NOT_NULL(g1);
  ASSERT_NOT_NULL(m1);
  ASSERT_NOT_NULL(v1);

  PolyOptimConfig cfg = {
      .kind = POLY_OPTIM_ADAM,
      .beta1 = 0.9,
      .beta2 = 0.999,
      .eps = 1e-8,
      .weight_decay = 0.0,
  };
  PolyTensor *params[2] = {p0, p1}, *grads[2] = {g0, g1};
  PolyTensor *m[2] = {m0, m1}, *v[2] = {v0, v1}, *outputs[8] = {0};
  ASSERT_INT_EQ(
      poly_optim_build_step(ctx, &cfg, lr, params, grads, 2, m, v, bc1, bc2, outputs, 8), 8
  );
  ASSERT_PTR_EQ(outputs[0], bc1);
  ASSERT_PTR_EQ(outputs[1], bc2);
  ASSERT_PTR_EQ(outputs[6], p0);
  ASSERT_PTR_EQ(outputs[7], p1);
  ASSERT_PTR_EQ(p0->uop_logical, NULL);
  ASSERT_PTR_EQ(m0->uop_logical, NULL);
  ASSERT_PTR_EQ(v0->uop_logical, NULL);
  ASSERT_INT_EQ(p0->uop_physical->op, POLY_OP_AFTER);
  ASSERT_NOT_NULL(p1->uop_logical);
  ASSERT_NOT_NULL(m1->uop_logical);
  ASSERT_NOT_NULL(v1->uop_logical);
  ASSERT_INT_EQ(p1->uop_logical->op, POLY_OP_AFTER);
  ASSERT_INT_EQ(bc1->uop_logical->op, POLY_OP_AFTER);
  ASSERT_INT_EQ(bc2->uop_logical->op, POLY_OP_AFTER);

  float p0_got[1] = {0}, p1_got[1] = {0};
  ASSERT_INT_EQ(read_tensor_f32(ctx, p0, p0_got, 1), 0);
  ASSERT_INT_EQ(read_tensor_f32(ctx, p1, p1_got, 1), 0);
  ASSERT_FLOAT_EQ(p0_got[0], 0.9f, 1e-5f);
  ASSERT_FLOAT_EQ(p1_got[0], 2.9f, 1e-5f);

  PolyTensor *all[] = {v0, m0, g0, p0, v1, m1, g1, p1, bc2, bc1, lr};
  for (int i = 0; i < (int)(sizeof(all) / sizeof(all[0])); i++)
    poly_tensor_release(all[i]);
  poly_ctx_destroy(ctx);
  PASS();
}

static PolyTensor *build_logical_policy_host_oracle(
    PolyCtx *ctx,
    PolyLogicalPolicy policy,
    float values[4]
) {
  int64_t shape[2] = {2, 2};
  if (poly_ctx_set_logical_policy(ctx, policy) != 0) return NULL;
  PolyTensor *host =
      poly_tensor_from_host(ctx, values, 4 * sizeof(*values), POLY_FLOAT32, shape, 2);
  PolyTensor *cpu = host ? poly_tensor_to_device(ctx, host, POLY_DEVICE_CPU) : NULL;
  PolyTensor *one = cpu ? poly_tensor_const_like_int(ctx, cpu, 1) : NULL;
  return cpu && one ? poly_tensor_alu2(ctx, POLY_OP_ADD, cpu, one) : NULL;
}

TEST(tensor, host_write_after_realize_preserves_current_storage_and_policy) {
  /* Existing C owner recipe: pinned Tensor._buffer -> Buffer.copy_from.
   * Frontends must realize pending effects, then write current physical
   * storage without calling a graph-root setter. */
  for (int policy = POLY_LOGICAL_NEVER; policy <= POLY_LOGICAL_UNTIL_REALIZE; policy++) {
    PolyCtx *ctx = poly_ctx_new();
    ASSERT_NOT_NULL(ctx);
    ASSERT_INT_EQ(poly_ctx_set_logical_policy(ctx, (PolyLogicalPolicy)policy), 0);
    float initial[3] = {1, 2, 3}, values[3] = {4, 5, 6}, got[3] = {0};
    int64_t shape[1] = {3};
    PolyTensor *host = poly_tensor_from_host(ctx, initial, sizeof(initial), POLY_FLOAT32, shape, 1);
    PolyTensor *x = poly_tensor_to_device(ctx, host, POLY_DEVICE_CPU);
    ASSERT_NOT_NULL(x);
    PolyUOp *logical = x->uop_logical;
    PolyTensor *out = NULL;
    ASSERT_INT_EQ(poly_realize_tensors(ctx, &x, 1, &out), 0);
    ASSERT_PTR_EQ(out, x);
    PolyUOp *physical = x->uop_physical;
    ASSERT_INT_EQ(physical->op, POLY_OP_BUFFER);
    ASSERT_INT_EQ(poly_buffer_write(ctx, physical, values, sizeof(values)), 0);
    ASSERT_PTR_EQ(x->uop_physical, physical);
    if (policy == POLY_LOGICAL_NEVER) ASSERT_PTR_EQ(x->uop_logical, NULL);
    if (policy == POLY_LOGICAL_ALWAYS) ASSERT_PTR_EQ(x->uop_logical, logical);
    ASSERT_INT_EQ(poly_ctx_collect(ctx), 0);
    ASSERT_INT_EQ(read_tensor_f32(ctx, x, got, 3), 0);
    for (int i = 0; i < 3; i++)
      ASSERT_FLOAT_EQ(got[i], values[i], 0);
    ASSERT_PTR_EQ(x->uop_physical, physical);
    poly_tensor_release(host);
    poly_tensor_release(x);
    poly_ctx_destroy(ctx);
  }
  PASS();
}

TEST(tensor, logical_never_host_construction_keeps_physical_graph_and_values_exact) {
  float values[4] = {1, 2, 3, 4};
  PolyCtx *always_ctx = poly_ctx_new(), *never_ctx = poly_ctx_new();
  ASSERT_NOT_NULL(always_ctx);
  ASSERT_NOT_NULL(never_ctx);
  PolyTensor *always = build_logical_policy_host_oracle(always_ctx, POLY_LOGICAL_ALWAYS, values);
  PolyTensor *never = build_logical_policy_host_oracle(never_ctx, POLY_LOGICAL_NEVER, values);
  ASSERT_NOT_NULL(always);
  ASSERT_NOT_NULL(never);
  ASSERT_PTR_EQ(poly_tensor_uop_logical(never), NULL);
  ASSERT_TRUE(uop_graph_isomorphic(
      always_ctx, poly_tensor_uop_physical(always), never_ctx, poly_tensor_uop_physical(never)
  ));
  float always_values[4] = {0}, never_values[4] = {0};
  ASSERT_INT_EQ(read_tensor_f32(always_ctx, always, always_values, 4), 0);
  ASSERT_INT_EQ(read_tensor_f32(never_ctx, never, never_values, 4), 0);
  for (int i = 0; i < 4; i++) {
    ASSERT_FLOAT_EQ(always_values[i], (float)i + 2.0f, 1e-6);
    ASSERT_FLOAT_EQ(never_values[i], always_values[i], 1e-6);
  }
  poly_ctx_destroy(always_ctx);
  poly_ctx_destroy(never_ctx);
  PASS();
}

TEST(tensor, logical_never_host_construction_skips_portable_nodes) {
  float values[4] = {1, 2, 3, 4};
  int64_t shape[2] = {2, 2};
  PolyCtx *always_ctx = poly_ctx_new(), *never_ctx = poly_ctx_new();
  ASSERT_NOT_NULL(always_ctx);
  ASSERT_NOT_NULL(never_ctx);
  ASSERT_INT_EQ(poly_ctx_set_logical_policy(always_ctx, POLY_LOGICAL_ALWAYS), 0);
  ASSERT_INT_EQ(poly_ctx_set_logical_policy(never_ctx, POLY_LOGICAL_NEVER), 0);
  PolyTensor *always =
      poly_tensor_from_host(always_ctx, values, sizeof(values), POLY_FLOAT32, shape, 2);
  PolyTensor *never =
      poly_tensor_from_host(never_ctx, values, sizeof(values), POLY_FLOAT32, shape, 2);
  ASSERT_NOT_NULL(always);
  ASSERT_NOT_NULL(never);
  ASSERT_TRUE(uop_graph_isomorphic(
      always_ctx, poly_tensor_uop_physical(always), never_ctx, poly_tensor_uop_physical(never)
  ));
  PolyCtxStats always_stats = {0}, never_stats = {0};
  ASSERT_INT_EQ(poly_ctx_stats(always_ctx, &always_stats), 0);
  ASSERT_INT_EQ(poly_ctx_stats(never_ctx, &never_stats), 0);
  ASSERT_TRUE(never_stats.cse_entries < always_stats.cse_entries);
  poly_ctx_destroy(always_ctx);
  poly_ctx_destroy(never_ctx);
  PASS();
}

static bool append_logical_policy_output(
    PolyTensor **outputs,
    int *n_outputs,
    int capacity,
    PolyTensor *tensor
) {
  if (!outputs || !n_outputs || !tensor || *n_outputs >= capacity) return false;
  outputs[(*n_outputs)++] = tensor;
  return true;
}

static int build_logical_policy_remaining_tensor_oracle(
    PolyCtx *ctx,
    PolyLogicalPolicy policy,
    PolyTensor **outputs,
    int capacity
) {
  int n_outputs = 0;
  int64_t row_shape[2] = {1, 3}, matrix_shape[2] = {2, 2}, vector_shape[1] = {2};
  int64_t tall_shape[2] = {3, 2}, tall_rhs_shape[1] = {3};
  int64_t rope_shape[2] = {1, 4}, frequency_shape[2] = {1, 2};
  if (poly_ctx_set_logical_policy(ctx, policy) != 0) return -1;

  PolyTensor *x = poly_tensor_empty(ctx, POLY_FLOAT32, row_shape, 2, POLY_DEVICE_CPU);
  PolyTensor *index = poly_tensor_empty(ctx, POLY_INT32, row_shape, 2, POLY_DEVICE_CPU);
  PolyTensor *src = poly_tensor_empty(ctx, POLY_FLOAT32, row_shape, 2, POLY_DEVICE_CPU);
  PolyTensor *matrix = poly_tensor_empty(ctx, POLY_FLOAT32, matrix_shape, 2, POLY_DEVICE_CPU);
  PolyTensor *vector = poly_tensor_empty(ctx, POLY_FLOAT32, vector_shape, 1, POLY_DEVICE_CPU);
  PolyTensor *tall = poly_tensor_empty(ctx, POLY_FLOAT32, tall_shape, 2, POLY_DEVICE_CPU);
  PolyTensor *tall_rhs = poly_tensor_empty(ctx, POLY_FLOAT32, tall_rhs_shape, 1, POLY_DEVICE_CPU);
  PolyTensor *rope_x = poly_tensor_empty(ctx, POLY_FLOAT32, rope_shape, 2, POLY_DEVICE_CPU);
  PolyTensor *freqs_cos = poly_tensor_empty(ctx, POLY_FLOAT32, frequency_shape, 2, POLY_DEVICE_CPU);
  PolyTensor *freqs_sin = poly_tensor_empty(ctx, POLY_FLOAT32, frequency_shape, 2, POLY_DEVICE_CPU);
  if (!x || !index || !src || !matrix || !vector || !tall || !tall_rhs || !rope_x || !freqs_cos ||
      !freqs_sin)
    return -1;

  if (!append_logical_policy_output(
          outputs, &n_outputs, capacity, poly_tensor_one_hot(ctx, index, 3)
      ) ||
      !append_logical_policy_output(
          outputs, &n_outputs, capacity, poly_tensor_index_select(ctx, x, 1, index)
      ) ||
      !append_logical_policy_output(
          outputs, &n_outputs, capacity, poly_tensor_gather_dim(ctx, x, 1, index)
      ) ||
      !append_logical_policy_output(
          outputs, &n_outputs, capacity, poly_tensor_scatter(ctx, x, 1, index, src, NULL)
      ) ||
      !append_logical_policy_output(
          outputs, &n_outputs, capacity, poly_tensor_scatter_reduce(ctx, x, 1, index, src, "sum", 1)
      ))
    return -1;

  PolyTensor *sort_values = NULL, *sort_indices = NULL;
  PolyTensor *topk_values = NULL, *topk_indices = NULL;
  if (poly_tensor_sort(ctx, x, 1, 0, &sort_values, &sort_indices) != 0 ||
      poly_tensor_topk(ctx, x, 2, 1, 1, 1, &topk_values, &topk_indices) != 0 ||
      !append_logical_policy_output(outputs, &n_outputs, capacity, sort_values) ||
      !append_logical_policy_output(outputs, &n_outputs, capacity, sort_indices) ||
      !append_logical_policy_output(outputs, &n_outputs, capacity, topk_values) ||
      !append_logical_policy_output(outputs, &n_outputs, capacity, topk_indices))
    return -1;

  PolyTensor *einsum_inputs[2] = {matrix, matrix};
  if (!append_logical_policy_output(
          outputs, &n_outputs, capacity, poly_tensor_einsum(ctx, "ij,jk->ik", einsum_inputs, 2)
      ) ||
      !append_logical_policy_output(
          outputs, &n_outputs, capacity, poly_tensor_rearrange(ctx, "a b->(a b)", x, NULL, NULL, 0)
      ) ||
      !append_logical_policy_output(
          outputs, &n_outputs, capacity, poly_tensor_argmax(ctx, x, 1, true)
      ) ||
      !append_logical_policy_output(
          outputs, &n_outputs, capacity, poly_tensor_contiguous_backward(ctx, x)
      ) ||
      !append_logical_policy_output(
          outputs, &n_outputs, capacity, poly_tensor_rope(ctx, rope_x, freqs_cos, freqs_sin)
      ))
    return -1;

  PolyTensor *q = NULL, *r = NULL;
  if (poly_tensor_qr_ex(ctx, matrix, POLY_QR_COMPLETE, &q, &r) != 0 ||
      !append_logical_policy_output(outputs, &n_outputs, capacity, q) ||
      !append_logical_policy_output(outputs, &n_outputs, capacity, r))
    return -1;
  PolyTensor *chol = poly_tensor_cholesky(ctx, matrix, 0);
  if (!append_logical_policy_output(
          outputs, &n_outputs, capacity, poly_tensor_triangular_solve(ctx, matrix, vector, 0, 0, 0)
      ) ||
      !append_logical_policy_output(outputs, &n_outputs, capacity, chol) ||
      !append_logical_policy_output(
          outputs, &n_outputs, capacity, poly_tensor_cholesky_solve(ctx, chol, vector, 0)
      ) ||
      !append_logical_policy_output(
          outputs, &n_outputs, capacity, poly_tensor_solve(ctx, matrix, vector)
      ) ||
      !append_logical_policy_output(
          outputs, &n_outputs, capacity, poly_tensor_lstsq(ctx, tall, tall_rhs)
      ))
    return -1;
  return n_outputs;
}

TEST(tensor, logical_never_remaining_tensor_wrappers_keep_physical_graphs_exact) {
  enum { MAX_OUTPUTS = 24 };
  PolyCtx *always_ctx = poly_ctx_new(), *never_ctx = poly_ctx_new();
  PolyTensor *always[MAX_OUTPUTS] = {0}, *never[MAX_OUTPUTS] = {0};
  ASSERT_NOT_NULL(always_ctx);
  ASSERT_NOT_NULL(never_ctx);
  int n_always = build_logical_policy_remaining_tensor_oracle(
      always_ctx, POLY_LOGICAL_ALWAYS, always, MAX_OUTPUTS
  );
  int n_never = build_logical_policy_remaining_tensor_oracle(
      never_ctx, POLY_LOGICAL_NEVER, never, MAX_OUTPUTS
  );
  ASSERT_TRUE(n_always > 0);
  ASSERT_INT_EQ(n_never, n_always);
  for (int i = 0; i < n_always; i++) {
    ASSERT_PTR_EQ(poly_tensor_uop_logical(never[i]), NULL);
    ASSERT_TRUE(uop_graph_isomorphic(
        always_ctx, poly_tensor_uop_physical(always[i]), never_ctx,
        poly_tensor_uop_physical(never[i])
    ));
  }
  poly_ctx_destroy(always_ctx);
  poly_ctx_destroy(never_ctx);
  PASS();
}

static int build_logical_policy_frontend_creation_oracle(
    PolyCtx *ctx,
    PolyLogicalPolicy policy,
    PolyTensor **outputs,
    int capacity
) {
  int n_outputs = 0;
  int f32 = poly_dtype_id_by_name("float32"), i32 = poly_dtype_id_by_name("int32");
  int64_t shape[2] = {2, 2};
  if (capacity < 5 || poly_ctx_set_logical_policy(ctx, policy) != 0) return -1;
  if (!append_logical_policy_output(
          outputs, &n_outputs, capacity,
          poly_tensor_full_float_by_id(ctx, shape, 2, 2.5, f32, POLY_DEVICE_CPU, true, true)
      ) ||
      !append_logical_policy_output(
          outputs, &n_outputs, capacity,
          poly_tensor_full_float_by_id(ctx, shape, 2, 2.5, f32, POLY_DEVICE_CPU, true, false)
      ) ||
      !append_logical_policy_output(
          outputs, &n_outputs, capacity,
          poly_tensor_arange_int_by_id(ctx, 0, 4, 1, i32, POLY_DEVICE_CPU)
      ) ||
      !append_logical_policy_output(
          outputs, &n_outputs, capacity,
          poly_tensor_linspace_by_id(ctx, 0.0, 1.0, 4, f32, POLY_DEVICE_CPU)
      ) ||
      !append_logical_policy_output(
          outputs, &n_outputs, capacity, poly_tensor_eye_by_id(ctx, 3, 3, f32, POLY_DEVICE_CPU)
      ))
    return -1;
  return n_outputs;
}

TEST(tensor, logical_never_frontend_creation_keeps_physical_graphs_and_values_exact) {
  enum { OUTPUTS = 5 };
  PolyCtx *always_ctx = poly_ctx_new(), *never_ctx = poly_ctx_new();
  PolyTensor *always[OUTPUTS] = {0}, *never[OUTPUTS] = {0};
  ASSERT_NOT_NULL(always_ctx);
  ASSERT_NOT_NULL(never_ctx);
  ASSERT_INT_EQ(
      build_logical_policy_frontend_creation_oracle(
          always_ctx, POLY_LOGICAL_ALWAYS, always, OUTPUTS
      ),
      OUTPUTS
  );
  ASSERT_INT_EQ(
      build_logical_policy_frontend_creation_oracle(never_ctx, POLY_LOGICAL_NEVER, never, OUTPUTS),
      OUTPUTS
  );
  for (int i = 0; i < OUTPUTS; i++) {
    ASSERT_PTR_EQ(poly_tensor_uop_logical(never[i]), NULL);
    ASSERT_TRUE(uop_graph_isomorphic(
        always_ctx, poly_tensor_uop_physical(always[i]), never_ctx,
        poly_tensor_uop_physical(never[i])
    ));
  }
  float always_values[4] = {0}, never_values[4] = {0};
  ASSERT_INT_EQ(read_tensor_f32(always_ctx, always[0], always_values, 4), 0);
  ASSERT_INT_EQ(read_tensor_f32(never_ctx, never[0], never_values, 4), 0);
  for (int i = 0; i < 4; i++) {
    ASSERT_FLOAT_EQ(always_values[i], 2.5f, 1e-6);
    ASSERT_FLOAT_EQ(never_values[i], always_values[i], 1e-6);
  }
  poly_ctx_destroy(always_ctx);
  poly_ctx_destroy(never_ctx);
  PASS();
}

TEST(tensor, rng_state_owns_only_seed_and_counter_handles) {
  /* Tinygrad 2026-08-22 Tensor._next_counter keeps exactly the device seed and
   * counter wrappers after call-local RNG temporaries die. manual_seed drops
   * both dictionaries (tensor.py:621-653). */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  poly_tensor_manual_seed(ctx, 123);
  int64_t shape[] = {1024};
  int f32_id = poly_dtype_id_by_name("float32");
  PolyTensor *out = poly_tensor_rand_by_id(ctx, shape, 1, f32_id, POLY_DEVICE_CPU, 1);
  ASSERT_NOT_NULL(out);
  PolyTensor *realized = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &out, 1, &realized), 0);
  ASSERT_NOT_NULL(realized);

  poly_tensor_release(out);
  ASSERT_INT_EQ(poly_ctx_collect(ctx), 0);
  PolyCtxStats stats = {0};
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &stats), 0);
  ASSERT_INT_EQ(stats.tensor_records, 2);

  poly_tensor_manual_seed(ctx, 456);
  ASSERT_INT_EQ(poly_ctx_collect(ctx), 0);
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &stats), 0);
  ASSERT_INT_EQ(stats.tensor_records, 0);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(tensor, dtype_admission_rejects_float_shifts_and_range_overflow) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *src[] = {poly_uop_const(ctx, poly_arg_float(1.5), POLY_FLOAT32), poly_const_int(ctx, 1)};
  PolyDType dtype;
  ASSERT_FALSE(poly_dtype_from_uop(POLY_OP_SHL, src, 2, poly_arg_none(), POLY_VOID, &dtype));
  ASSERT_FALSE(poly_dtype_from_uop(POLY_OP_SHR, src, 2, poly_arg_none(), POLY_VOID, &dtype));
  ASSERT_TRUE(poly_arange_int_by_id(ctx, 0, 129, 1, poly_dtype_id_by_name("int8")) == NULL);
  /* Nonzero endpoints allocate both bounds; LSan covers their independent
   * scratch lifetimes even when the resulting graph is never executed. */
  ASSERT_NOT_NULL(poly_arange_int_by_id(ctx, 2, 5, 1, poly_dtype_id_by_name("int8")));
  ASSERT_NOT_NULL(poly_arange_int_by_id(ctx, -5, -2, 1, poly_dtype_id_by_name("int8")));
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(tensor, dtype_admission_randn_casts_integer_output) {
  PolyCtx *ctx = poly_ctx_new();
  int64_t shape[] = {3};
  PolyTensor *out =
      poly_tensor_randn_by_id(ctx, shape, 1, poly_dtype_id_by_name("int32"), POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(out);
  ASSERT_TRUE(poly_dtype_eq(poly_tensor_uop_physical(out)->dtype, POLY_INT32));
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(tensor, configured_seed_factory_defaults_and_integer_randn) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  int old = poly_get_default_float();
  poly_set_default_float(poly_dtype_id_by_name("float64"));
  int64_t shape[] = {3};
  PolyUOp *uniform = poly_rand(ctx, shape, 1, 42);
  PolyUOp *normal = poly_randn(ctx, shape, 1, 42);
  PolyUOp *integer = poly_randn_by_id(ctx, shape, 1, 42, poly_dtype_id_by_name("int32"));
  poly_set_default_float(old);
  ASSERT_NOT_NULL(uniform);
  ASSERT_NOT_NULL(normal);
  ASSERT_TRUE(poly_dtype_eq(uniform->dtype, POLY_FLOAT64));
  ASSERT_TRUE(poly_dtype_eq(normal->dtype, POLY_FLOAT64));
  ASSERT_NOT_NULL(integer);
  ASSERT_TRUE(poly_dtype_eq(integer->dtype, POLY_INT32));
  ASSERT_INT_EQ(integer->op, POLY_OP_CAST);
  ASSERT_TRUE(poly_dtype_eq(integer->src[0]->dtype, POLY_FLOAT32));
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(tensor, randn_reset_keeps_rng_source_owners_valid) {
  /* Tinygrad 2026-08-22 Tensor.manual_seed replaces the seed/counter maps;
   * nested randn construction must leave their source ownership valid until
   * that replacement, even after call-local Tensor wrappers retire. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  int64_t shape[] = {5};
  int f32_id = poly_dtype_id_by_name("float32");
  float first[5] = {0}, second[5] = {0}, reset[5] = {0};

  poly_tensor_manual_seed(ctx, 42);
  PolyTensor *a = poly_tensor_randn_by_id(ctx, shape, 1, f32_id, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(a);
  ASSERT_INT_EQ(read_tensor_bytes(ctx, a, first, sizeof(first)), 0);
  poly_tensor_release(a);

  PolyTensor *b = poly_tensor_randn_by_id(ctx, shape, 1, f32_id, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(b);
  ASSERT_INT_EQ(read_tensor_bytes(ctx, b, second, sizeof(second)), 0);
  poly_tensor_release(b);

  poly_tensor_manual_seed(ctx, 42);
  PolyTensor *c = poly_tensor_randn_by_id(ctx, shape, 1, f32_id, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(c);
  ASSERT_INT_EQ(read_tensor_bytes(ctx, c, reset, sizeof(reset)), 0);
  ASSERT_TRUE(memcmp(first, second, sizeof(first)) != 0);
  ASSERT_TRUE(memcmp(first, reset, sizeof(first)) == 0);
  poly_tensor_release(c);
  poly_ctx_destroy(ctx);
  PASS();
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
  PolyUOp *linear = poly_test_create_linear(ctx, sink);
  return linear && linear->n_src == 1 ? poly_test_linear_call_body(linear, 0) : NULL;
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
  PolyTensor *a_host = poly_tensor_from_host(ctx, av, sizeof(av), POLY_FLOAT32, shape, 1);
  PolyTensor *b_host = poly_tensor_from_host(ctx, bv, sizeof(bv), POLY_FLOAT32, shape, 1);
  PolyTensor *a = poly_tensor_to_device(ctx, a_host, POLY_DEVICE_CPU);
  PolyTensor *b = poly_tensor_to_device(ctx, b_host, POLY_DEVICE_CPU);
  PolyTensor *sum = poly_tensor_alu2(ctx, POLY_OP_ADD, a, b);
  ASSERT_NOT_NULL(sum);
  PolyTensor *results[1] = {sum};
  PolyUOp *logical_inputs[2] = {a->uop_logical, b->uop_logical};
  PolyUOp *physical_inputs[2] = {a->uop_physical, b->uop_physical};
  PolyTensor *outputs[1] = {NULL};
  ASSERT_INT_EQ(
      poly_tensor_function(
          ctx, results, 1, logical_inputs, physical_inputs, 2, "ordered_add", false, false, false,
          outputs
      ),
      0
  );
  ASSERT_NOT_NULL(outputs[0]);

  PolyUOp *surfaces[2] = {outputs[0]->uop_logical, outputs[0]->uop_physical};
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

  PolyUOp *grad = poly_grad(ctx, outputs[0]->uop_physical, a->uop_physical);
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

TEST(tensor, function_uses_pre_body_input_occurrences) {
  /* Tinygrad 2026-08-22/a9069c177a9d function.py:43-46 snapshots call_uops
   * before executing a body that may assign into a captured Tensor. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  float state_data[1] = {1.0f}, x_data[1] = {3.0f};
  int64_t shape[1] = {1};
  PolyTensor *state_host =
      poly_tensor_from_host(ctx, state_data, sizeof(state_data), POLY_FLOAT32, shape, 1);
  PolyTensor *x_host = poly_tensor_from_host(ctx, x_data, sizeof(x_data), POLY_FLOAT32, shape, 1);
  PolyTensor *state = poly_tensor_to_device(ctx, state_host, POLY_DEVICE_CPU);
  PolyTensor *x = poly_tensor_to_device(ctx, x_host, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(state);
  ASSERT_NOT_NULL(x);

  PolyUOp *before_logical = state->uop_logical;
  PolyUOp *before_physical = state->uop_physical;
  int f32 = poly_dtype_id_by_name("float32");
  PolyTensor *one = poly_tensor_const_float_by_id(ctx, 1.0, f32, POLY_DEVICE_CPU);
  PolyTensor *incremented = poly_tensor_alu2(ctx, POLY_OP_ADD, state, one);
  ASSERT_PTR_EQ(poly_tensor_assign(ctx, state, incremented), state);
  ASSERT_PTR_NEQ(state->uop_logical, before_logical);
  ASSERT_PTR_NEQ(state->uop_physical, before_physical);

  PolyTensor *two = poly_tensor_const_float_by_id(ctx, 2.0, f32, POLY_DEVICE_CPU);
  PolyTensor *result = poly_tensor_alu2(ctx, POLY_OP_ADD, x, two);
  PolyTensor *results[1] = {result};
  PolyUOp *logical_inputs[2] = {before_logical, x->uop_logical};
  PolyUOp *physical_inputs[2] = {before_physical, x->uop_physical};
  PolyTensor *output = NULL;
  ASSERT_INT_EQ(
      poly_tensor_function(
          ctx, results, 1, logical_inputs, physical_inputs, 2, "stateful", false, false, false,
          &output
      ),
      0
  );
  ASSERT_NOT_NULL(output);
  ASSERT_TRUE(poly_uop_reachable(ctx, output->uop_logical, before_logical));
  ASSERT_FALSE(poly_uop_reachable(ctx, output->uop_logical, state->uop_logical));
  ASSERT_TRUE(poly_uop_reachable(ctx, output->uop_physical, before_physical));
  ASSERT_FALSE(poly_uop_reachable(ctx, output->uop_physical, state->uop_physical));

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(tensor, static_empty_shares_unique_with_deviceful_physical_root) {
  PolyCtx *ctx = poly_ctx_new();
  int64_t shape[2] = {2, 3};
  ASSERT_EQ(poly_tensor_empty(ctx, POLY_FLOAT32, shape, 2, POLY_DEVICE_AUTO), NULL);
  ASSERT_EQ(poly_tensor_empty(ctx, POLY_FLOAT32, shape, 2, POLY_DEVICE_DISK), NULL);
  PolyTensor *host = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 2, POLY_DEVICE_HOST);
  ASSERT_NOT_NULL(host);
  ASSERT_STR_EQ(base_buf(host->uop_physical)->arg.param->device, "HOST");
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
  ASSERT_INT_EQ(physical_buffer->n_src, 1);
  ASSERT_INT_EQ(logical_buffer->src[0]->op, POLY_OP_UNIQUE);
  ASSERT_INT_EQ(physical_buffer->src[0]->op, POLY_OP_CONST);
  ASSERT_INT_EQ(physical_buffer->arg.kind, POLY_ARG_PARAM);
  ASSERT_INT_EQ(physical_buffer->arg.param->slot, logical_buffer->src[0]->arg.i);
  ASSERT_STR_EQ(physical_buffer->arg.param->device, "CPU");

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(tensor, root_mutators_require_complete_physical_root) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *logical = poly_buffer_f32(ctx, 2);
  PolyUOp *physical = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 2, POLY_DEVICE_CPU);
  PolyTensor *tensor =
      poly_tensor_create_with_roots(ctx, logical, physical, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(tensor);

  PolyUOp *realized = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 2, POLY_DEVICE_CPU);
  ASSERT_INT_EQ(
      poly_tensor_set_physical(ctx, tensor, realized, POLY_TENSOR_VALUE, POLY_DEVICE_CPU), 0
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

TEST(tensor, host_array_stages_creation_device_before_requested_copy) {
  PolyCtx *ctx = poly_ctx_new();
  float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  int64_t shape[2] = {2, 2};
  PolyTensor *source = poly_tensor_from_host(ctx, data, sizeof(data), POLY_FLOAT32, shape, 2);
  ASSERT_NOT_NULL(source);
  ASSERT_INT_EQ(source->device, POLY_DEVICE_HOST);
  ASSERT_INT_EQ(source->uop_logical->op, POLY_OP_RESHAPE);
  ASSERT_INT_EQ(source->uop_physical->op, POLY_OP_RESHAPE);
  ASSERT_INT_EQ(poly_uop_device(source->uop_physical), POLY_DEVICE_HOST);

  PolyUOp *logical_buffer = base_buf(source->uop_logical);
  PolyUOp *physical_buffer = base_buf(source->uop_physical);
  ASSERT_NOT_NULL(logical_buffer);
  ASSERT_NOT_NULL(physical_buffer);
  ASSERT_INT_EQ(logical_buffer->n_src, 1);
  ASSERT_INT_EQ(physical_buffer->n_src, 1);
  ASSERT_INT_EQ(physical_buffer->arg.kind, POLY_ARG_PARAM);
  ASSERT_STR_EQ(physical_buffer->arg.param->device, "HOST");
  ASSERT_EQ(poly_buffer_get_ptr(ctx, logical_buffer), NULL);
  ASSERT_PTR_EQ(poly_buffer_get_ptr(ctx, physical_buffer), data);

  PolyTensor *moved = poly_tensor_to_device(ctx, source, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(moved);
  ASSERT_PTR_NEQ(moved, source);
  ASSERT_INT_EQ(moved->uop_physical->op, POLY_OP_COPY);
  ASSERT_PTR_EQ(moved->uop_physical->src[0], source->uop_physical);

  PolyTensor *realized = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &moved, 1, &realized), 0);
  ASSERT_NOT_NULL(realized);
  PolyUOp *realized_buffer = (PolyUOp *)poly_uop_get_buffer_identity(realized->uop_physical);
  ASSERT_NOT_NULL(realized_buffer);
  ASSERT_PTR_NEQ(realized_buffer, physical_buffer);
  float out[4] = {0};
  ASSERT_INT_EQ(poly_buffer_read(ctx, realized_buffer, out, sizeof(out)), 0);
  for (int i = 0; i < 4; i++)
    ASSERT_FLOAT_NEAR(out[i], data[i], 4, 1e-6);

  ASSERT_EQ(poly_tensor_from_host(ctx, data, sizeof(data) - 1, POLY_FLOAT32, shape, 2), NULL);
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
  PolyTensor *first = poly_tensor_rand_by_id(ctx, shape, 1, f32_id, POLY_DEVICE_CPU, 1);
  ASSERT_NOT_NULL(first);
  ASSERT_INT_EQ(count_op_in_root(ctx, first->uop_physical, POLY_OP_AFTER), 1);
  ASSERT_INT_EQ(count_op_in_root(ctx, first->uop_physical, POLY_OP_STORE), 1);
  ASSERT_INT_EQ(count_op_in_root(ctx, first->uop_physical, POLY_OP_THREEFRY), 2);
  ASSERT_INT_EQ(count_op_in_root(ctx, first->uop_physical, POLY_OP_COPY), 2);
  ASSERT_INT_EQ(count_op_in_root(ctx, first->uop_logical, POLY_OP_COPY), 0);

  float first_values[8] = {0};
  ASSERT_INT_EQ(read_tensor_f32(ctx, first, first_values, 8), 0);
  const uint32_t expected_bits[8] = {
      UINT32_C(0x3efa31a0), UINT32_C(0x3eb22b7c), UINT32_C(0x3f28c97e), UINT32_C(0x3f22effe),
      UINT32_C(0x3ef13c94), UINT32_C(0x3e10dd30), UINT32_C(0x3e8e61ec), UINT32_C(0x3d4c9dc0),
  };
  ASSERT_TRUE(memcmp(first_values, expected_bits, sizeof(expected_bits)) == 0);

  poly_tensor_manual_seed(ctx, 1337);
  PolyTensor *reset = poly_tensor_rand_by_id(ctx, shape, 1, f32_id, POLY_DEVICE_CPU, 1);
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

TEST(tensor, stateful_rand_zero_extent_constructs_current_empty_graph) {
  /* Current RandMixin.random_bits returns counter[0:0] when num == 0; the
   * public Tensor.rand call still advances the counter by zero and composes
   * the ordinary bitcast/ALU graph (mixin/rand.py:17-42). */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  int f32_id = poly_dtype_id_by_name("float32");
  int64_t shape[1] = {0};
  poly_tensor_manual_seed(ctx, 123);
  PolyTensor *out = poly_tensor_rand_by_id(ctx, shape, 1, f32_id, POLY_DEVICE_CPU, 0);
  ASSERT_NOT_NULL(out);
  PolyShape out_shape = poly_uop_max_shape_cached(ctx, out->uop_physical);
  ASSERT_INT_EQ(out_shape.ndim, 1);
  ASSERT_INT_EQ(out_shape.dims[0], 0);
  ASSERT_INT_EQ(count_op_in_root(ctx, out->uop_physical, POLY_OP_AFTER), 1);
  ASSERT_INT_EQ(count_op_in_root(ctx, out->uop_physical, POLY_OP_STORE), 1);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(tensor, assign_accepts_deviceless_value_with_different_backend_label) {
  PolyCtx *ctx = poly_ctx_new();
  int64_t dims[] = {4};
  PolyTensor *target = poly_tensor_empty(ctx, POLY_FLOAT32, dims, 1, POLY_DEVICE_CPU);
  PolyTensor *value =
      poly_tensor_const_float_by_id(ctx, 1.25, poly_dtype_id_by_name("float32"), POLY_DEVICE_CUDA);
  PolyTensor *assigned = poly_tensor_assign(ctx, target, value);
  bool correct = assigned && assigned->uop_physical->op == POLY_OP_AFTER &&
                 assigned->uop_physical->n_src == 2 &&
                 assigned->uop_physical->src[1]->op == POLY_OP_STORE;
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(correct);
  PASS();
}

TEST(tensor, scalar_constructors_store_exact_const_as_both_roots) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  int int_id = poly_dtype_id_by_name("int32");
  int float_id = poly_dtype_id_by_name("float32");
  ASSERT_TRUE(int_id >= 0);
  ASSERT_TRUE(float_id >= 0);

  PolyTensor *i = poly_tensor_const_int_by_id(ctx, 7, int_id, POLY_DEVICE_CUDA);
  PolyTensor *f = poly_tensor_const_float_by_id(ctx, 1.5, float_id, POLY_DEVICE_CPU);
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

TEST(tensor, weak_const_promotion_retypes_the_const_argument) {
  /* Current ElementwiseMixin._broadcasted.promote rebuilds weak CONST values
   * through const_like(base.val, weak_dtype(out_dtype)). Thus weakint zero
   * promoted by weakfloat one becomes CONST weakfloat/ConstFloat(0.0), not a
   * weakfloat node retaining an integer argument (mixin/elementwise.py:22-29,
   * dtype.py:77-82). */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  int weakint = poly_dtype_id_by_name("weakint");
  int weakfloat = poly_dtype_id_by_name("weakfloat");
  PolyTensor *zero = poly_tensor_const_int_by_id(ctx, 0, weakint, POLY_DEVICE_CPU);
  PolyTensor *one = poly_tensor_const_float_by_id(ctx, 1.0, weakfloat, POLY_DEVICE_CPU);
  PolyTensor *sum = poly_tensor_alu2(ctx, POLY_OP_ADD, zero, one);
  ASSERT_NOT_NULL(sum);

  PolyUOp *roots[] = {sum->uop_logical, sum->uop_physical};
  for (size_t i = 0; i < sizeof(roots) / sizeof(roots[0]); i++) {
    ASSERT_NOT_NULL(roots[i]);
    ASSERT_EQ(roots[i]->op, POLY_OP_ADD);
    ASSERT_INT_EQ(roots[i]->n_src, 2);
    ASSERT_TRUE(poly_dtype_eq(roots[i]->src[0]->dtype, POLY_WEAKFLOAT));
    ASSERT_EQ(roots[i]->src[0]->arg.kind, POLY_ARG_FLOAT);
    ASSERT_FLOAT_EQ(roots[i]->src[0]->arg.f, 0.0, 0.0);
    ASSERT_TRUE(poly_dtype_eq(roots[i]->src[1]->dtype, POLY_WEAKFLOAT));
    ASSERT_EQ(roots[i]->src[1]->arg.kind, POLY_ARG_FLOAT);
    ASSERT_FLOAT_EQ(roots[i]->src[1]->arg.f, 1.0, 0.0);
  }

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(tensor, current_full_commits_weak_value_only_at_storage_boundary) {
  /* Current CreationMixin.full keeps the inferred weak EXPAND as the STORE
   * value, while empty_like(None) creates strong storage. buffer=False keeps
   * the weak value graph and explicit weak storage is rejected
   * (tinygrad/mixin/creation.py:61-85, tinygrad/uop/ops.py:814-827). */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  int weakfloat = poly_dtype_id_by_name("weakfloat");
  int64_t shape[] = {2};

  PolyTensor *buffered =
      poly_tensor_full_float_by_id(ctx, shape, 1, 0.0, weakfloat, POLY_DEVICE_CPU, false, true);
  ASSERT_NOT_NULL(buffered);
  PolyUOp *root = buffered->uop_physical;
  ASSERT_NOT_NULL(root);
  ASSERT_EQ(root->op, POLY_OP_AFTER);
  ASSERT_INT_EQ(root->n_src, 2);
  ASSERT_TRUE(poly_dtype_eq(root->dtype, POLY_FLOAT32));
  ASSERT_EQ(root->src[0]->op, POLY_OP_BUFFER);
  ASSERT_TRUE(poly_dtype_eq(root->src[0]->dtype, POLY_FLOAT32));
  ASSERT_EQ(root->src[1]->op, POLY_OP_STORE);
  ASSERT_INT_EQ(root->src[1]->n_src, 2);
  ASSERT_PTR_EQ(root->src[1]->src[0], root->src[0]);
  ASSERT_EQ(root->src[1]->src[1]->op, POLY_OP_EXPAND);
  ASSERT_TRUE(poly_dtype_eq(root->src[1]->src[1]->dtype, POLY_WEAKFLOAT));
  ASSERT_PTR_NEQ(buffered->uop_logical, buffered->uop_physical);

  PolyTensor *unbuffered =
      poly_tensor_full_float_by_id(ctx, shape, 1, 0.0, weakfloat, POLY_DEVICE_CPU, false, false);
  ASSERT_NOT_NULL(unbuffered);
  ASSERT_PTR_EQ(unbuffered->uop_logical, unbuffered->uop_physical);
  ASSERT_EQ(unbuffered->uop_physical->op, POLY_OP_EXPAND);
  ASSERT_TRUE(poly_dtype_eq(unbuffered->uop_physical->dtype, POLY_WEAKFLOAT));

  ASSERT_TRUE(
      poly_tensor_full_float_by_id(ctx, shape, 1, 0.0, weakfloat, POLY_DEVICE_CPU, true, true) ==
      NULL
  );
  ASSERT_TRUE(poly_tensor_empty(ctx, POLY_WEAKFLOAT, shape, 1, POLY_DEVICE_CPU) == NULL);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(tensor, pure_constructors_store_one_device_free_root) {
  /* Current full(buffer=False), arange, linspace, and eye keep their pure value
   * graph. The Tensor boundary stores that exact root as both twins; no
   * placement map is involved. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  int i32 = poly_dtype_id_by_name("int32");
  int f32 = poly_dtype_id_by_name("float32");
  int64_t shape[] = {2, 3};
  PolyTensor *values[] = {
      poly_tensor_full_int_by_id(ctx, shape, 2, 2, i32, POLY_DEVICE_CPU, true, false),
      poly_tensor_full_float_by_id(ctx, shape, 2, 2.0, f32, POLY_DEVICE_CPU, true, false),
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
      poly_tensor_full_int_by_id(ctx, singleton_shape, 1, 2, i32, POLY_DEVICE_CPU, true, false);
  PolyTensor *singleton_arange = poly_tensor_arange_int_by_id(ctx, 0, 1, 1, i32, POLY_DEVICE_CPU);
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
  PolyTensor *pad = poly_tensor_pad_value_float(ctx, source, pad_pairs, 2, 0.0);
  PolyTensor *flip = poly_tensor_flip(ctx, source, axes, 1);
  PolyTensor *results[6] = {reshape, expand, permute, shrink, pad, flip};
  PolyOps ops[6] = {
      POLY_OP_RESHAPE, POLY_OP_EXPAND, POLY_OP_PERMUTE, POLY_OP_SHRINK, POLY_OP_PAD, POLY_OP_FLIP,
  };

  for (int i = 0; i < 6; i++) {
    ASSERT_NOT_NULL(results[i]);
    ASSERT_INT_EQ(results[i]->uop_logical->op, ops[i]);
    ASSERT_INT_EQ(results[i]->uop_physical->op, ops[i]);
    PolyUOp *logical_source = results[i]->uop_logical->src[0];
    PolyUOp *physical_source = results[i]->uop_physical->src[0];
    if (ops[i] == POLY_OP_EXPAND) {
      ASSERT_INT_EQ(logical_source->op, POLY_OP_RESHAPE);
      ASSERT_INT_EQ(physical_source->op, POLY_OP_RESHAPE);
      ASSERT_PTR_EQ(logical_source->src[0], source->uop_logical);
      ASSERT_PTR_EQ(physical_source->src[0], source->uop_physical);
    } else {
      ASSERT_PTR_EQ(logical_source, source->uop_logical);
      ASSERT_PTR_EQ(physical_source, source->uop_physical);
    }
    ASSERT_PTR_NEQ(results[i]->uop_logical, results[i]->uop_physical);
  }

  /* Raw logical-only construction is inert import/re-placement input. Default
   * Tensor composition must reject it instead of reconstructing execution
   * state from the logical graph. */
  PolyUOp *legacy_uop = poly_buffer_f32(ctx, 3);
  PolyTensor *legacy =
      poly_tensor_create_with_roots(ctx, legacy_uop, NULL, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
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
  PolyUOp *physical_buffer = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 20, POLY_DEVICE_CPU);
  PolyUOp *physical = poly_reshape(ctx, physical_buffer, (int64_t[]){10, 2}, 2);
  PolyTensor *source =
      poly_tensor_create_with_roots(ctx, logical, physical, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(source);
  ASSERT_PTR_NEQ(source->uop_logical, source->uop_physical);

  float data[20];
  for (int i = 0; i < 20; i++)
    data[i] = (float)i;

  PolyUOp *i =
      poly_uop_variable(ctx, "i", poly_arg_int(0), poly_arg_int(8), POLY_WEAKINT, 1, false);
  PolyUOp *ib = poly_uop_bind(ctx, i, 4);
  PolyUOp *starts[2] = {ib, poly_const_int(ctx, 0)};
  PolyUOp *sizes[2] = {poly_const_int(ctx, 2), poly_const_int(ctx, 2)};
  PolyUOp *expected_logical = poly_shrink_uop(ctx, source->uop_logical, starts, sizes, 2);
  PolyUOp *expected_physical = poly_shrink_uop(ctx, source->uop_physical, starts, sizes, 2);

  PolyTensor *slice = poly_tensor_shrink_uop(ctx, source, starts, sizes, 2);
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
          ctx, slice->uop_physical,
          poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CPU), out, leaf_buffers,
          leaf_data, 1
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
  PolyUOp *physical_buffer = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 2, POLY_DEVICE_CPU);
  PolyUOp *physical = poly_reshape(ctx, physical_buffer, (int64_t[]){1, 2}, 2);
  PolyTensor *source =
      poly_tensor_create_with_roots(ctx, logical, physical, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(source);
  ASSERT_PTR_NEQ(source->uop_logical, source->uop_physical);

  PolyUOp *n =
      poly_uop_variable(ctx, "n", poly_arg_int(1), poly_arg_int(4), POLY_WEAKINT, 1, false);
  PolyUOp *nb = poly_uop_bind(ctx, n, 3);
  PolyUOp *dims[2] = {nb, poly_const_int(ctx, 2)};
  PolyUOp *expected_logical = poly_expand_uop(ctx, source->uop_logical, dims, 2);
  PolyUOp *expected_physical = poly_expand_uop(ctx, source->uop_physical, dims, 2);

  PolyTensor *expanded = poly_tensor_expand_uop(ctx, source, dims, 2);
  ASSERT_NOT_NULL(expanded);
  ASSERT_PTR_EQ(expanded->uop_logical, expected_logical);
  ASSERT_PTR_EQ(expanded->uop_physical, expected_physical);
  ASSERT_INT_EQ(expanded->uop_logical->op, POLY_OP_EXPAND);
  ASSERT_INT_EQ(expanded->uop_physical->op, POLY_OP_EXPAND);
  ASSERT_INT_EQ(expanded->uop_logical->src[0]->op, POLY_OP_RESHAPE);
  ASSERT_INT_EQ(expanded->uop_physical->src[0]->op, POLY_OP_RESHAPE);
  ASSERT_PTR_EQ(expanded->uop_logical->src[0]->src[0], source->uop_logical);
  ASSERT_PTR_EQ(expanded->uop_physical->src[0]->src[0], source->uop_physical);

  PolyUOp *starts[2] = {poly_const_int(ctx, 0), poly_const_int(ctx, 0)};
  PolyUOp *sizes[2] = {poly_const_int(ctx, 3), poly_const_int(ctx, 2)};
  PolyUOp *static_view = poly_shrink_uop(ctx, expanded->uop_physical, starts, sizes, 2);
  ASSERT_NOT_NULL(static_view);
  float data[2] = {1.0f, 2.0f};
  float out[6] = {0};
  PolyUOp *leaf_buffers[1] = {physical_buffer};
  float *leaf_data[1] = {data};
  ASSERT_INT_EQ(
      realize_uop(
          ctx, static_view, poly_test_buffer_on_device(ctx, POLY_FLOAT32, 6, POLY_DEVICE_CPU), out,
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
  PolyUOp *physical_buffer = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 64, POLY_DEVICE_CPU);
  PolyUOp *physical = poly_reshape(ctx, physical_buffer, (int64_t[]){1, 8, 1, 8}, 4);
  PolyTensor *source =
      poly_tensor_create_with_roots(ctx, logical, physical, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(source);

  PolyUOp *n =
      poly_uop_variable(ctx, "n", poly_arg_int(1), poly_arg_int(7), POLY_WEAKINT, 1, false);
  PolyUOp *nb = poly_uop_bind(ctx, n, 3);
  PolyUOp *starts[4] = {
      poly_const_int(ctx, 0),
      poly_const_int(ctx, 0),
      poly_const_int(ctx, 0),
      poly_const_int(ctx, 0),
  };
  PolyUOp *sizes[4] = {
      poly_const_int(ctx, 1),
      nb,
      poly_const_int(ctx, 1),
      poly_const_int(ctx, 8),
  };
  PolyTensor *slice = poly_tensor_shrink_uop(ctx, source, starts, sizes, 4);
  ASSERT_NOT_NULL(slice);

  PolyUOp *dims[4] = {
      poly_const_int(ctx, 1),
      nb,
      poly_const_int(ctx, 2),
      poly_const_int(ctx, 4),
  };
  PolyUOp *expected_logical = poly_reshape_uop(ctx, slice->uop_logical, dims, 4);
  PolyUOp *expected_physical = poly_reshape_uop(ctx, slice->uop_physical, dims, 4);
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
      poly_const_int(ctx, 1),
      nb,
      poly_const_int(ctx, 3),
      poly_const_int(ctx, 4),
  };
  ASSERT_EQ(poly_tensor_reshape_uop(ctx, slice, bad_dims, 4), NULL);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(tensor, symbolic_dot_preserves_noncontracted_bind_dimension) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *n =
      poly_uop_variable(ctx, "n", poly_arg_int(1), poly_arg_int(7), POLY_WEAKINT, 1, false);
  PolyUOp *nb = poly_uop_bind(ctx, n, 3);

  PolyUOp *q_logical = make_buf(ctx, (int64_t[]){1, 2, 1, 4}, 4);
  PolyUOp *q_buffer = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 8, POLY_DEVICE_CPU);
  PolyUOp *q_physical = poly_reshape(ctx, q_buffer, (int64_t[]){1, 2, 1, 4}, 4);
  PolyTensor *query =
      poly_tensor_create_with_roots(ctx, q_logical, q_physical, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);

  PolyUOp *k_logical_base = make_buf(ctx, (int64_t[]){1, 8, 2, 4}, 4);
  PolyUOp *k_buffer = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 64, POLY_DEVICE_CPU);
  PolyUOp *k_physical_base = poly_reshape(ctx, k_buffer, (int64_t[]){1, 8, 2, 4}, 4);
  PolyTensor *key_base = poly_tensor_create_with_roots(
      ctx, k_logical_base, k_physical_base, POLY_TENSOR_VALUE, POLY_DEVICE_CPU
  );
  PolyUOp *starts[4] = {
      poly_const_int(ctx, 0),
      poly_const_int(ctx, 0),
      poly_const_int(ctx, 0),
      poly_const_int(ctx, 0),
  };
  PolyUOp *sizes[4] = {
      poly_const_int(ctx, 1),
      nb,
      poly_const_int(ctx, 2),
      poly_const_int(ctx, 4),
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
  /* Current dot leaves implicit broadcasting on MUL and _rop removes the
   * reduced final axis directly (mixin/op.py:367-392). */
  ASSERT_INT_EQ(output->uop_logical->op, POLY_OP_REDUCE);
  ASSERT_INT_EQ(output->uop_physical->op, POLY_OP_REDUCE);
  ASSERT_INT_EQ(output->uop_physical->src[0]->op, POLY_OP_PERMUTE);
  ASSERT_INT_EQ(output->uop_physical->src[0]->src[0]->op, POLY_OP_MUL);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(tensor, symbolic_softmax_and_max_preserve_exact_bind_dimension) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *n =
      poly_uop_variable(ctx, "n", poly_arg_int(1), poly_arg_int(7), POLY_WEAKINT, 1, false);
  PolyUOp *nb = poly_uop_bind(ctx, n, 3);
  PolyUOp *base = make_buf(ctx, (int64_t[]){1, 2, 1, 7}, 4);
  PolyUOp *starts[4] = {
      poly_const_int(ctx, 0),
      poly_const_int(ctx, 0),
      poly_const_int(ctx, 0),
      poly_const_int(ctx, 0),
  };
  PolyUOp *sizes[4] = {
      poly_const_int(ctx, 1),
      poly_const_int(ctx, 2),
      poly_const_int(ctx, 1),
      nb,
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

  /* Current UOp._rop returns REDUCE directly when every requested axis is a
   * real reduction axis; the public non-keepdim path adds no reshape.  The
   * surviving symbolic dimension is still the exact AFTER/STORE binding
   * (uop/ops.py:629-638). */
  PolyUOp *other_axis = poly_max_reduce(ctx, x, 1, 0);
  ASSERT_NOT_NULL(other_axis);
  ASSERT_INT_EQ(other_axis->op, POLY_OP_REDUCE);
  ASSERT_INT_EQ(other_axis->src[0]->op, POLY_OP_PERMUTE);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, other_axis), 3);
  ASSERT_PTR_EQ(poly_uop_shape_dim(ctx, other_axis, 2), nb);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(tensor, custom_kernel_uses_ordered_roots_and_one_call_per_graph) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  int64_t shape[1] = {4};
  PolyTensor *c = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_CPU);
  PolyTensor *a = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_CPU);
  PolyTensor *b = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(c);
  ASSERT_NOT_NULL(a);
  ASSERT_NOT_NULL(b);

  PolyUOp *pc = poly_uop_placeholder_like(ctx, c->uop_physical, 0);
  PolyUOp *pa = poly_uop_placeholder_like(ctx, a->uop_physical, 1);
  PolyUOp *pb = poly_uop_placeholder_like(ctx, b->uop_physical, 2);
  PolyUOp *r = poly_uop_range(ctx, 4, 0, POLY_AXIS_LOOP);
  PolyUOp *idxs[1] = {r};
  PolyUOp *ci = poly_uop_index(ctx, pc, idxs, 1);
  PolyUOp *ai = poly_uop_index(ctx, pa, idxs, 1);
  PolyUOp *bi = poly_uop_index(ctx, pb, idxs, 1);
  PolyUOp *sum = poly_alu2(ctx, POLY_OP_ADD, poly_uop_load(ctx, ai), poly_uop_load(ctx, bi));
  PolyUOp *store = poly_uop_store(ctx, ci, sum);
  PolyUOp *end = poly_uop_end(ctx, store, &r, 1);
  PolyUOp *body = poly_uop_sink_ex(ctx, &end, 1, "custom_add_4", 1);
  ASSERT_NOT_NULL(body);

  PolyTensor *inputs[3] = {c, a, b};
  PolyTensor *outputs[3] = {0};
  ASSERT_INT_EQ(poly_tensor_custom_kernel(ctx, body, inputs, 3, 17, outputs), 0);
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
  ASSERT_INT_EQ(logical_call->arg.kind, POLY_ARG_CALL_INFO);
  ASSERT_INT_EQ(physical_call->arg.kind, POLY_ARG_CALL_INFO);
  ASSERT_TRUE(logical_call->arg.call_info->has_grad_fxn);
  ASSERT_TRUE(physical_call->arg.call_info->has_grad_fxn);
  ASSERT_INT_EQ(logical_call->arg.call_info->grad_fxn_key, 17);
  ASSERT_INT_EQ(physical_call->arg.call_info->grad_fxn_key, 17);
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
  PolyUOp *consumer = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CPU);
  PolyUOp *probe_store = poly_store_val(ctx, consumer, outputs[0]->uop_physical);
  PolyUOp *probe_sink = poly_sink1(ctx, probe_store);
  PolyVarBinding *probe_vars = NULL;
  int n_probe_vars = 0;
  PolyUOp *probe_linear = poly_linear_effect_sink(ctx, probe_sink, &probe_vars, &n_probe_vars);
  ASSERT_NOT_NULL(probe_linear);
  ASSERT_INT_EQ(n_probe_vars, 0);
  free(probe_vars);
  ASSERT_INT_EQ(probe_linear->n_src, 2);
  PolyUOp *custom_call = poly_test_linear_call(probe_linear, 0);
  ASSERT_NOT_NULL(custom_call);
  ASSERT_INT_EQ(custom_call->op, POLY_OP_CALL);
  ASSERT_INT_EQ(custom_call->n_src, 4);
  ASSERT_PTR_EQ(custom_call->src[0], body);
  ASSERT_INT_EQ(poly_test_linear_call(probe_linear, 1)->src[0]->op, POLY_OP_SINK);

  float c_data[4] = {0};
  float a_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float b_data[4] = {10.0f, 20.0f, 30.0f, 40.0f};
  PolyUOp *leaf_buffers[3] = {c->uop_physical, a->uop_physical, b->uop_physical};
  float *leaf_data[3] = {c_data, a_data, b_data};
  PolyTestBufferView views[3];
  for (int i = 0; i < 3; i++)
    views[i] = POLY_TEST_HOST_VIEW(leaf_buffers[i], leaf_data[i]);
  ASSERT_INT_EQ(
      poly_test_realize_buffer_views(ctx, poly_sink1(ctx, outputs[0]->uop_physical), views, 3), 0
  );
  ASSERT_FLOAT_EQ(c_data[0], 11.0f, 0.0);
  ASSERT_FLOAT_EQ(c_data[1], 22.0f, 0.0);
  ASSERT_FLOAT_EQ(c_data[2], 33.0f, 0.0);
  ASSERT_FLOAT_EQ(c_data[3], 44.0f, 0.0);

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

  /* Current DTypeMixin.bitcast emits one raw BITCAST for this ordinary CPU
   * width change; UOp shape inference scales the last dimension
   * (mixin/dtype.py:35-50, uop/ops.py:404-411). */
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

TEST(tensor, identity_constructors_return_owned_handles) {
  /* Tinygrad 2026-08-22/a9069c177a9d returns self for same-device/device-free
   * Tensor.to and identity DTypeMixin.bitcast. C preserves pointer identity,
   * but each constructor result must own one explicit reference. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  int f32 = poly_dtype_id_by_name("float32");
  int64_t shape[] = {2};
  PolyTensor *source = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_CPU);
  PolyTensor *scalar = poly_tensor_const_float_by_id(ctx, 1.0, f32, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(source);
  ASSERT_NOT_NULL(scalar);

  PolyTensor *same_device = poly_tensor_to_device(ctx, source, POLY_DEVICE_CPU);
  bool same_device_pointer = same_device == source;
  uint32_t same_device_refs = source->owner_refs;
  if (same_device_pointer && same_device_refs > 1) poly_tensor_release(same_device);

  PolyTensor *device_free = poly_tensor_to_device(ctx, scalar, POLY_DEVICE_INTERP);
  bool device_free_pointer = device_free == scalar;
  uint32_t device_free_refs = scalar->owner_refs;
  if (device_free_pointer && device_free_refs > 1) poly_tensor_release(device_free);

  PolyTensor *same_bitcast = poly_tensor_bitcast_by_id(ctx, source, f32);
  bool same_bitcast_pointer = same_bitcast == source;
  uint32_t same_bitcast_refs = source->owner_refs;
  if (same_bitcast_pointer && same_bitcast_refs > 1) poly_tensor_release(same_bitcast);

  poly_tensor_release(source);
  poly_tensor_release(scalar);
  poly_ctx_destroy(ctx);

  ASSERT_TRUE(same_device_pointer);
  ASSERT_INT_EQ(same_device_refs, 2);
  ASSERT_TRUE(device_free_pointer);
  ASSERT_INT_EQ(device_free_refs, 2);
  ASSERT_TRUE(same_bitcast_pointer);
  ASSERT_INT_EQ(same_bitcast_refs, 2);
  PASS();
}

TEST(tensor, unequal_width_bitcast_matches_pinned_tensor_topology_and_values) {
  /* Current DTypeMixin.bitcast represents non-identity width changes with
   * exactly one BITCAST. These assertions are paired with the direct
   * bitcast cases in the canonical graph corpus (mixin/dtype.py:35-50). */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  int u8_id = poly_dtype_id_by_name("uint8");
  int u32_id = poly_dtype_id_by_name("uint32");

  int64_t wide_source_shape[1] = {8};
  PolyTensor *wide_source =
      poly_tensor_empty(ctx, POLY_UINT8, wide_source_shape, 1, POLY_DEVICE_CPU);
  PolyUOp *wide_fill_uop = poly_full_int_by_id(ctx, wide_source_shape, 1, 1, u8_id);
  PolyTensor *wide_fill = poly_tensor_create_with_roots(
      ctx, wide_fill_uop, wide_fill_uop, POLY_TENSOR_VALUE, POLY_DEVICE_CPU
  );
  ASSERT_NOT_NULL(poly_tensor_assign(ctx, wide_source, wide_fill));
  PolyTensor *wide = poly_tensor_bitcast_by_id(ctx, wide_source, u32_id);
  ASSERT_NOT_NULL(wide);
  PolyShape wide_shape = poly_uop_max_shape_cached(ctx, wide->uop_physical);
  ASSERT_INT_EQ(wide_shape.ndim, 1);
  ASSERT_INT_EQ(wide_shape.dims[0], 2);
  ASSERT_INT_EQ(wide->uop_physical->op, POLY_OP_BITCAST);
  ASSERT_PTR_EQ(wide->uop_physical->src[0], wide_source->uop_physical);
  ASSERT_INT_EQ(count_op_in_root(ctx, wide->uop_physical, POLY_OP_BITCAST), 1);
  ASSERT_INT_EQ(count_op_in_root(ctx, wide->uop_physical, POLY_OP_CAST), 0);
  ASSERT_INT_EQ(count_op_in_root(ctx, wide->uop_physical, POLY_OP_SHL), 0);
  uint32_t wide_values[2] = {0};
  ASSERT_INT_EQ(read_tensor_bytes(ctx, wide, wide_values, sizeof(wide_values)), 0);
  ASSERT_INT_EQ(wide_values[0], UINT32_C(0x01010101));
  ASSERT_INT_EQ(wide_values[1], UINT32_C(0x01010101));

  int64_t narrow_source_shape[1] = {2};
  PolyTensor *narrow_source =
      poly_tensor_empty(ctx, POLY_UINT32, narrow_source_shape, 1, POLY_DEVICE_CPU);
  PolyUOp *narrow_fill_uop = poly_full_int_by_id(ctx, narrow_source_shape, 1, 1, u32_id);
  PolyTensor *narrow_fill = poly_tensor_create_with_roots(
      ctx, narrow_fill_uop, narrow_fill_uop, POLY_TENSOR_VALUE, POLY_DEVICE_CPU
  );
  ASSERT_NOT_NULL(poly_tensor_assign(ctx, narrow_source, narrow_fill));
  PolyTensor *narrow = poly_tensor_bitcast_by_id(ctx, narrow_source, u8_id);
  ASSERT_NOT_NULL(narrow);
  PolyShape narrow_shape = poly_uop_max_shape_cached(ctx, narrow->uop_physical);
  ASSERT_INT_EQ(narrow_shape.ndim, 1);
  ASSERT_INT_EQ(narrow_shape.dims[0], 8);
  ASSERT_INT_EQ(narrow->uop_physical->op, POLY_OP_BITCAST);
  ASSERT_PTR_EQ(narrow->uop_physical->src[0], narrow_source->uop_physical);
  ASSERT_INT_EQ(count_op_in_root(ctx, narrow->uop_physical, POLY_OP_BITCAST), 1);
  ASSERT_INT_EQ(count_op_in_root(ctx, narrow->uop_physical, POLY_OP_CAST), 0);
  ASSERT_INT_EQ(count_op_in_root(ctx, narrow->uop_physical, POLY_OP_SHR), 0);
  uint8_t narrow_values[8] = {0};
  ASSERT_INT_EQ(read_tensor_bytes(ctx, narrow, narrow_values, sizeof(narrow_values)), 0);
  const uint8_t expected_narrow[8] = {1, 0, 0, 0, 1, 0, 0, 0};
  ASSERT_TRUE(memcmp(narrow_values, expected_narrow, sizeof(expected_narrow)) == 0);

  int64_t invalid_shape[1] = {3};
  PolyTensor *invalid_source =
      poly_tensor_empty(ctx, POLY_UINT8, invalid_shape, 1, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(invalid_source);
  ASSERT_TRUE(poly_tensor_bitcast_by_id(ctx, invalid_source, u32_id) == NULL);

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

  PolyUOp *cuda = poly_device_uop(ctx, POLY_DEVICE_CUDA);
  PolyUOp *cpu = poly_device_uop(ctx, POLY_DEVICE_CPU);
  PolyUOp *to_cuda = poly_copy_to_device_uop(ctx, x, cuda);
  PolyUOp *physical = poly_copy_to_device_uop(ctx, to_cuda, cpu);
  PolyTensor *tensor =
      poly_tensor_create_with_roots(ctx, x, physical, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(tensor);
  PolyUOp *expected_logical = poly_rearrange(ctx, "(h w)->h w", x, "h w", axis_values, 2);
  PolyUOp *expected_physical = poly_rearrange(ctx, "(h w)->h w", physical, "h w", axis_values, 2);
  PolyTensor *tensor_valid =
      poly_tensor_rearrange(ctx, "(h w)->h w", tensor, "h w", axis_values, 2);
  ASSERT_NOT_NULL(tensor_valid);
  ASSERT_EQ(poly_tensor_uop_logical(tensor_valid), expected_logical);
  ASSERT_EQ(poly_tensor_uop_physical(tensor_valid), expected_physical);
  ASSERT_EQ(poly_tensor_rearrange(foreign_ctx, "(h w)->h w", tensor, "h w", axis_values, 2), NULL);

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
  ASSERT_EQ(poly_rearrange(ctx, "(h w)->h w", x, "h w", overflow_values, 2), NULL);

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

TEST(pe, tensor_min_builds_both_roots_from_exact_occurrences) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyDType dtypes[] = {POLY_INT8, POLY_UINT8, POLY_INT64, POLY_UINT64, POLY_BOOL, POLY_FLOAT32};
  for (size_t i = 0; i < sizeof(dtypes) / sizeof(dtypes[0]); i++) {
    PolyUOp *logical =
        poly_uop_new_buffer(ctx, poly_device_uop(ctx, POLY_DEVICE_CPU), 6, dtypes[i], (int)i);
    logical = poly_reshape(ctx, logical, (int64_t[]){2, 3}, 2);
    PolyUOp *physical = poly_copy_to_device_uop(
        ctx, poly_copy_to_device_uop(ctx, logical, poly_device_uop(ctx, POLY_DEVICE_CUDA)),
        poly_device_uop(ctx, POLY_DEVICE_CPU)
    );
    PolyTensor *x =
        poly_tensor_create_with_roots(ctx, logical, physical, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
    ASSERT_NOT_NULL(x);
    PolyTensor *out = poly_tensor_min(ctx, x, (int64_t[]){0, 1}, 2, false);
    ASSERT_NOT_NULL(out);
    PolyUOp *roots[] = {logical, physical};
    PolyUOp *outputs[] = {out->uop_logical, out->uop_physical};
    for (int domain = 0; domain < 2; domain++) {
      PolyUOp *mask;
      PolyOps op;
      if (poly_dtype_is_bool(dtypes[i])) {
        mask = poly_const_typed(ctx, POLY_BOOL, 1);
        op = POLY_OP_CMPNE;
      } else if (poly_dtype_is_float(dtypes[i])) {
        mask = poly_const_typed(ctx, POLY_WEAKFLOAT, -1.0);
        op = POLY_OP_MUL;
      } else {
        op = POLY_OP_XOR;
        if (poly_dtype_is_unsigned(dtypes[i])) {
          mask = poly_dtype_eq(dtypes[i], POLY_UINT64)
                     ? poly_uop_const(
                           ctx, poly_arg_bigint(1, (uint32_t[]){UINT32_MAX, UINT32_MAX}, 2),
                           POLY_WEAKINT
                       )
                     : poly_const_int(ctx, 255);
        } else
          mask = poly_const_int(ctx, -1);
      }
      PolyUOp *inverse = poly_alu2(ctx, op, roots[domain], mask);
      PolyUOp *reduced = poly_reduce_axis(ctx, POLY_OP_MAX, inverse, (int64_t[]){0, 1}, 2);
      ASSERT_PTR_EQ(outputs[domain], poly_alu2(ctx, op, reduced, mask));
      ASSERT_INT_EQ(poly_uop_ndim(ctx, outputs[domain]), 0);
    }
    ASSERT_EQ(poly_tensor_min(ctx, x, (int64_t[]){2}, 1, false), NULL);
    ASSERT_EQ(poly_tensor_min(ctx, x, NULL, 1, false), NULL);
  }
  PolyCtx *foreign = poly_ctx_new();
  PolyTensor *x = poly_tensor_empty(ctx, POLY_FLOAT32, (int64_t[]){2}, 1, POLY_DEVICE_CPU);
  ASSERT_EQ(poly_tensor_min(foreign, x, (int64_t[]){0}, 1, false), NULL);
  poly_ctx_destroy(foreign);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, tensor_minimum_bool_uses_xor_in_both_domains) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *logical =
      poly_uop_new_buffer(ctx, poly_device_uop(ctx, POLY_DEVICE_CPU), 4, POLY_BOOL, 0);
  PolyUOp *physical = poly_copy_to_device_uop(
      ctx, poly_copy_to_device_uop(ctx, logical, poly_device_uop(ctx, POLY_DEVICE_CUDA)),
      poly_device_uop(ctx, POLY_DEVICE_CPU)
  );
  PolyTensor *a =
      poly_tensor_create_with_roots(ctx, logical, logical, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  PolyTensor *b =
      poly_tensor_create_with_roots(ctx, logical, physical, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  PolyTensor *out = poly_tensor_minimum(ctx, a, b);
  ASSERT_NOT_NULL(out);
  PolyUOp *mask = poly_const_typed(ctx, POLY_BOOL, 1);
  PolyUOp *left = poly_alu2(ctx, POLY_OP_XOR, logical, mask);
  PolyUOp *right = poly_alu2(ctx, POLY_OP_XOR, physical, mask);
  bool logical_ok = out->uop_logical ==
                    poly_alu2(ctx, POLY_OP_XOR, poly_alu2(ctx, POLY_OP_MAX, left, left), mask);
  bool physical_ok = out->uop_physical ==
                     poly_alu2(ctx, POLY_OP_XOR, poly_alu2(ctx, POLY_OP_MAX, left, right), mask);
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(logical_ok && physical_ok);
  PASS();
}

TEST(pe, tensor_argmax_builds_both_roots_from_exact_occurrences) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *logical = make_buf(ctx, (int64_t[]){2, 3}, 2);
  PolyUOp *cuda = poly_device_uop(ctx, POLY_DEVICE_CUDA);
  PolyUOp *cpu = poly_device_uop(ctx, POLY_DEVICE_CPU);
  PolyUOp *to_cuda = poly_copy_to_device_uop(ctx, logical, cuda);
  PolyUOp *physical = poly_copy_to_device_uop(ctx, to_cuda, cpu);
  PolyTensor *src =
      poly_tensor_create_with_roots(ctx, logical, physical, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(src);

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

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, tensor_argmax_rejects_foreign_context) {
  PolyCtx *owner = poly_ctx_new();
  PolyCtx *foreign = poly_ctx_new();
  ASSERT_NOT_NULL(owner);
  ASSERT_NOT_NULL(foreign);
  PolyTensor *src = poly_tensor_empty(owner, POLY_FLOAT32, (int64_t[]){2, 3}, 2, POLY_DEVICE_CPU);
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
  PolyUOp *to_cuda = poly_copy_to_device_uop(ctx, logical, cuda);
  PolyUOp *physical = poly_copy_to_device_uop(ctx, to_cuda, cpu);
  PolyTensor *src =
      poly_tensor_create_with_roots(ctx, logical, physical, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(src);

  PolyUOp *expected_gelu_logical = poly_gelu(ctx, logical);
  PolyUOp *expected_gelu_physical = poly_gelu(ctx, physical);
  PolyTensor *gelu = poly_tensor_gelu(ctx, src);
  ASSERT_NOT_NULL(expected_gelu_logical);
  ASSERT_NOT_NULL(expected_gelu_physical);
  ASSERT_NOT_NULL(gelu);
  ASSERT_EQ(poly_tensor_uop_logical(gelu), expected_gelu_logical);
  ASSERT_EQ(poly_tensor_uop_physical(gelu), expected_gelu_physical);

  PolyUOp *expected_quick_logical = poly_quick_gelu(ctx, logical);
  PolyUOp *expected_quick_physical = poly_quick_gelu(ctx, physical);
  PolyTensor *quick = poly_tensor_quick_gelu(ctx, src);
  ASSERT_NOT_NULL(expected_quick_logical);
  ASSERT_NOT_NULL(expected_quick_physical);
  ASSERT_NOT_NULL(quick);
  ASSERT_EQ(poly_tensor_uop_logical(quick), expected_quick_logical);
  ASSERT_EQ(poly_tensor_uop_physical(quick), expected_quick_physical);

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
  PolyUOp *to_cuda = poly_copy_to_device_uop(ctx, logical, cuda);
  PolyUOp *physical = poly_copy_to_device_uop(ctx, to_cuda, cpu);
  PolyTensor *src =
      poly_tensor_create_with_roots(ctx, logical, physical, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(src);

  PolyUOp *expected_log1p_logical = poly_log1p(ctx, logical);
  PolyUOp *expected_log1p_physical = poly_log1p(ctx, physical);
  PolyTensor *log1p = poly_tensor_log1p(ctx, src);
  ASSERT_NOT_NULL(expected_log1p_logical);
  ASSERT_NOT_NULL(expected_log1p_physical);
  ASSERT_NOT_NULL(log1p);
  ASSERT_EQ(poly_tensor_uop_logical(log1p), expected_log1p_logical);
  ASSERT_EQ(poly_tensor_uop_physical(log1p), expected_log1p_physical);

  PolyUOp *expected_expm1_logical = poly_expm1(ctx, logical);
  PolyUOp *expected_expm1_physical = poly_expm1(ctx, physical);
  PolyTensor *expm1 = poly_tensor_expm1(ctx, src);
  ASSERT_NOT_NULL(expected_expm1_logical);
  ASSERT_NOT_NULL(expected_expm1_physical);
  ASSERT_NOT_NULL(expm1);
  ASSERT_EQ(poly_tensor_uop_logical(expm1), expected_expm1_logical);
  ASSERT_EQ(poly_tensor_uop_physical(expm1), expected_expm1_physical);

  ASSERT_EQ(poly_tensor_log1p(foreign, src), NULL);
  ASSERT_EQ(poly_tensor_expm1(foreign, src), NULL);

  poly_ctx_destroy(foreign);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, cast_same_dtype_elides_and_scalar_target_preserves_vector_lanes) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *scalar = poly_const_int(ctx, 3);
  PolyUOp *scalar_cast = poly_cast(ctx, scalar, POLY_INT32);
  ASSERT_NOT_NULL(scalar_cast);
  ASSERT_INT_EQ(scalar_cast->op, POLY_OP_CAST);
  ASSERT_PTR_EQ(scalar_cast->src[0], scalar);

  PolyUOp *lanes[2] = {poly_const_int(ctx, 1), poly_const_int(ctx, 2)};
  PolyUOp *vector = poly_uop(ctx, POLY_OP_STACK, POLY_INT32, lanes, 2, poly_arg_none());
  PolyUOp *cast = poly_cast(ctx, vector, POLY_FLOAT32);
  ASSERT_NOT_NULL(cast);
  ASSERT_INT_EQ(cast->op, POLY_OP_CAST);
  ASSERT_INT_EQ(poly_uop_max_numel(ctx, cast), 2);
  ASSERT_TRUE(poly_dtype_eq(cast->dtype, POLY_FLOAT32));
  ASSERT_EQ(cast->src[0], vector);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, gather_dim_e2e_matches_tinygrad_probe) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = make_buf(ctx, (int64_t[]){2, 3, 4}, 3);
  PolyUOp *idx = poly_reshape(ctx, poly_test_buffer(ctx, POLY_INT32, 8), (int64_t[]){2, 2, 2}, 3);
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
  PolyUOp *idx0 = poly_reshape(ctx, poly_test_buffer(ctx, POLY_INT32, 4), (int64_t[]){1, 4}, 2);
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
  PolyUOp *idx = poly_reshape(ctx, poly_test_buffer(ctx, POLY_INT32, 9), (int64_t[]){3, 3}, 2);
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
  PolyUOp *idx_dup = poly_reshape(ctx, poly_test_buffer(ctx, POLY_INT32, 3), (int64_t[]){1, 3}, 2);
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
  PolyUOp *idx = poly_reshape(ctx, poly_test_buffer(ctx, POLY_INT32, 10), (int64_t[]){1, 10}, 2);
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
  PolyTensor *self = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 2, POLY_DEVICE_CPU);
  PolyTensor *index = poly_tensor_empty(ctx, POLY_INT32, shape, 2, POLY_DEVICE_CPU);
  PolyTensor *src = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 2, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(self);
  ASSERT_NOT_NULL(index);
  ASSERT_NOT_NULL(src);

  PolyUOp *logical_scatter =
      poly_scatter(ctx, self->uop_logical, 1, index->uop_logical, src->uop_logical, NULL);
  PolyUOp *physical_scatter =
      poly_scatter(ctx, self->uop_physical, 1, index->uop_physical, src->uop_physical, NULL);
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
  PolyTensor *sum = poly_tensor_scatter_reduce(ctx, self, 1, index, src, "sum", 1);
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
          ctx, scatter->uop_physical,
          poly_test_buffer_on_device(ctx, POLY_FLOAT32, 5, POLY_DEVICE_CPU), scatter_values, leaves,
          leaf_data, 3
      ),
      0
  );
  const float expected_scatter[5] = {6, 8, 3, 9, 10};
  for (int i = 0; i < 5; i++)
    ASSERT_FLOAT_EQ(scatter_values[i], expected_scatter[i], 1e-5f);

  float sum_values[5] = {0};
  ASSERT_INT_EQ(
      realize_uop(
          ctx, sum->uop_physical, poly_test_buffer_on_device(ctx, POLY_FLOAT32, 5, POLY_DEVICE_CPU),
          sum_values, leaves, leaf_data, 3
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
  PolyTensor *a = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 2, POLY_DEVICE_CPU);
  PolyTensor *b = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 2, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(a);
  ASSERT_NOT_NULL(b);

  PolyUOp *logical_inputs[2] = {a->uop_logical, b->uop_logical};
  PolyUOp *physical_inputs[2] = {a->uop_physical, b->uop_physical};
  PolyUOp *logical = poly_einsum(ctx, "ij,jk->ik", logical_inputs, 2);
  PolyUOp *physical = poly_einsum(ctx, "ij,jk->ik", physical_inputs, 2);
  PolyTensor *inputs[2] = {a, b};
  PolyTensor *out = poly_tensor_einsum(ctx, "ij,jk->ik", inputs, 2);
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
          ctx, out->uop_physical, poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CPU),
          values, leaves, leaf_data, 2
      ),
      0
  );
  const float expected[4] = {19, 22, 43, 50};
  for (int i = 0; i < 4; i++)
    ASSERT_FLOAT_EQ(values[i], expected[i], 1e-5f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(tensor, surface_owners_lifetime_and_domains) {
  for (int logical = 0; logical < 2; logical++) {
    PolyCtx *ctx = poly_ctx_new(), *other = poly_ctx_new();
    poly_ctx_set_logical_policy(ctx, logical ? POLY_LOGICAL_ALWAYS : POLY_LOGICAL_NEVER);
    PolyTensor *x = poly_tensor_empty(ctx, POLY_FLOAT32, (int64_t[]){2, 3}, 2, POLY_DEVICE_INTERP);
    ASSERT_NOT_NULL(x);
    float data[] = {1, 2, 3, 4, 5, 6};
    ASSERT_INT_EQ(poly_buffer_write(ctx, poly_uop_base(x->uop_physical), data, sizeof(data)), 0);
    int64_t axis = 1;
    PolyTensor *product = poly_tensor_prod(ctx, x, &axis, 1, false);
    PolyTensor *scan = poly_tensor_logcumsumexp(ctx, x, 1);
    PolyTensor *stack = poly_tensor_stack(ctx, (PolyTensor *[]){x, x}, 2, 1);
    ASSERT_NOT_NULL(product);
    ASSERT_NOT_NULL(scan);
    ASSERT_NOT_NULL(stack);
    ASSERT_INT_EQ(scan->uop_logical != NULL, logical);
    ASSERT_PTR_EQ(poly_tensor_prod(other, x, &axis, 1, false), NULL);
    ASSERT_PTR_EQ(poly_tensor_unfold(ctx, x, 1, 2, 0), NULL);
    ASSERT_PTR_EQ(poly_tensor_diagonal(ctx, x, 0, 1, 1), NULL);
    ASSERT_PTR_EQ(poly_tensor_diagonal(ctx, x, INT64_MIN, 0, 1), NULL);
    ASSERT_PTR_EQ(poly_tensor_pad_mode(ctx, x, (int64_t[]){0, 0, INT64_MIN, 0}, 2, 1), NULL);
    poly_tensor_release(x);
    poly_ctx_collect(ctx);
    float got[12];
    ASSERT_INT_EQ(read_tensor_f32(ctx, product, got, 2), 0);
    ASSERT_FLOAT_EQ(got[0], 6, 1e-5);
    ASSERT_FLOAT_EQ(got[1], 120, 1e-5);
    ASSERT_INT_EQ(read_tensor_f32(ctx, scan, got, 6), 0);
    ASSERT_FLOAT_EQ(got[2], 3.407606, 1e-5);
    ASSERT_FLOAT_EQ(got[5], 6.407606, 1e-5);
    /* Like Tensor.numpy, pack the repeated view before reading storage bytes. */
    PolyTensor *packed = poly_tensor_contiguous(ctx, stack);
    ASSERT_NOT_NULL(packed);
    ASSERT_INT_EQ(read_tensor_f32(ctx, packed, got, 12), 0);
    ASSERT_FLOAT_EQ(got[3], 1, 0.0);
    ASSERT_FLOAT_EQ(got[11], 6, 0.0);
    poly_tensor_release(packed);
    poly_tensor_release(stack);
    poly_tensor_release(scan);
    poly_tensor_release(product);
    poly_ctx_collect(ctx);
    poly_ctx_destroy(other);
    poly_ctx_destroy(ctx);
  }
  PASS();
}

TEST(tensor, spatial_owners_pooling_lifetime_and_arguments) {
  for (int logical = 0; logical < 2; logical++) {
    PolyCtx *ctx = poly_ctx_new(), *other = poly_ctx_new();
    poly_ctx_set_logical_policy(ctx, logical ? POLY_LOGICAL_ALWAYS : POLY_LOGICAL_NEVER);
    PolyTensor *x =
        poly_tensor_empty(ctx, POLY_FLOAT32, (int64_t[]){1, 1, 3, 3}, 4, POLY_DEVICE_INTERP);
    float data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    ASSERT_INT_EQ(poly_buffer_write(ctx, poly_uop_base(x->uop_physical), data, sizeof(data)), 0);
    int64_t k[] = {2, 2};
    PolyTensor *indices = NULL;
    PolyTensor *max = poly_tensor_max_pool2d(ctx, x, k, 2, NULL, NULL, NULL, 0, true, &indices);
    PolyTensor *avg = poly_tensor_avg_pool2d(ctx, x, k, 2, NULL, NULL, NULL, 0, true, true);
    ASSERT_NOT_NULL(max);
    ASSERT_NOT_NULL(indices);
    ASSERT_NOT_NULL(avg);
    ASSERT_INT_EQ(max->uop_logical != NULL, logical);
    ASSERT_INT_EQ(indices->uop_logical != NULL, logical);
    ASSERT_PTR_EQ(poly_tensor_avg_pool2d(other, x, k, 2, NULL, NULL, NULL, 0, false, true), NULL);
    ASSERT_PTR_EQ(
        poly_tensor_avg_pool2d(ctx, x, (int64_t[]){0, 2}, 2, NULL, NULL, NULL, 0, false, true), NULL
    );
    poly_tensor_release(x);
    poly_ctx_collect(ctx);
    float got[4];
    ASSERT_INT_EQ(read_tensor_f32(ctx, max, got, 4), 0);
    ASSERT_FLOAT_EQ(got[0], 5, 0);
    ASSERT_FLOAT_EQ(got[3], 9, 0);
    ASSERT_INT_EQ(read_tensor_f32(ctx, avg, got, 4), 0);
    ASSERT_FLOAT_EQ(got[0], 3, 0);
    ASSERT_FLOAT_EQ(got[1], 4.5, 0);
    PolyTensor *idx_float = poly_tensor_cast_by_id(ctx, indices, 12);
    ASSERT_INT_EQ(read_tensor_f32(ctx, idx_float, got, 4), 0);
    ASSERT_FLOAT_EQ(got[0], 4, 0);
    ASSERT_FLOAT_EQ(got[3], 8, 0);
    poly_tensor_release(idx_float);
    poly_tensor_release(indices);
    poly_tensor_release(max);
    poly_tensor_release(avg);
    poly_ctx_collect(ctx);
    poly_ctx_destroy(other);
    poly_ctx_destroy(ctx);
  }
  PASS();
}

TEST(tensor, spatial_owners_invalid_const_and_storage_dtype) {
  PolyCtx *ctx = poly_ctx_new();
  PolyTensor *x =
      poly_tensor_full_invalid_by_id(ctx, (int64_t[]){2, 3}, 2, 12, POLY_DEVICE_INTERP, true);
  ASSERT_NOT_NULL(x);
  ASSERT_TRUE(poly_dtype_eq(x->uop_physical->dtype, POLY_FLOAT32));
  ASSERT_INT_EQ(x->uop_physical->op, POLY_OP_AFTER);
  PolyUOp *value = poly_uop_base(x->uop_physical->src[1]->src[1]);
  ASSERT_INT_EQ(value->op, POLY_OP_CONST);
  ASSERT_INT_EQ(value->arg.kind, POLY_ARG_INVALID);
  ASSERT_TRUE(poly_dtype_eq(value->dtype, POLY_BOOL));
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(tensor, spatial_owners_transpose_short_output_padding) {
  PolyCtx *ctx = poly_ctx_new();
  PolyTensor *x =
      poly_tensor_empty(ctx, POLY_FLOAT32, (int64_t[]){1, 1, 3, 3}, 4, POLY_DEVICE_INTERP);
  PolyTensor *w =
      poly_tensor_empty(ctx, POLY_FLOAT32, (int64_t[]){1, 1, 2, 2}, 4, POLY_DEVICE_INTERP);
  int64_t output_padding[] = {1};
  /* Only one readable element: ASan detects accidental spatial-rank reads. */
  PolyTensor *out =
      poly_tensor_conv_transpose2d(ctx, x, w, NULL, 1, NULL, NULL, NULL, 0, output_padding, 1);
  ASSERT_NOT_NULL(out);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, out->uop_physical), 4);
  const int64_t *shape = poly_uop_max_shape_dims(ctx, out->uop_physical);
  ASSERT_INT_EQ(shape[2], 4);
  ASSERT_INT_EQ(shape[3], 6);
  ASSERT_PTR_EQ(
      poly_tensor_conv_transpose2d(ctx, x, w, NULL, 1, NULL, NULL, NULL, 0, output_padding, 0), NULL
  );
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(tensor, indexed_owners_scalar_tensor_index) {
  PolyCtx *ctx = poly_ctx_new();
  PolyTensor *x = poly_tensor_empty(ctx, POLY_FLOAT32, (int64_t[]){2, 3}, 2, POLY_DEVICE_INTERP);
  PolyTensor *index = poly_tensor_empty(ctx, POLY_INT32, NULL, 0, POLY_DEVICE_INTERP);
  float data[] = {0, 1, 2, 3, 4, 5};
  int32_t idx = 1;
  /* Empty's shape view is not a storage identity (including scalar reshape). */
  ASSERT_INT_EQ(poly_buffer_write(ctx, poly_uop_base(x->uop_physical), data, sizeof(data)), 0);
  ASSERT_INT_EQ(poly_buffer_write(ctx, poly_uop_base(index->uop_physical), &idx, sizeof(idx)), 0);
  PolyTensor *out = poly_tensor_index_select(ctx, x, 0, index);
  float got[3];
  bool ok =
      out && read_tensor_f32(ctx, out, got, 3) == 0 && got[0] == 3 && got[1] == 4 && got[2] == 5;
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(ok);
  PASS();
}

TEST(tensor, indexed_owners_lifetime_and_admission) {
  for (int logical = 0; logical < 2; logical++) {
    PolyCtx *ctx = poly_ctx_new(), *other = poly_ctx_new();
    poly_ctx_set_logical_policy(ctx, logical ? POLY_LOGICAL_ALWAYS : POLY_LOGICAL_NEVER);
    PolyTensor *x = poly_tensor_empty(ctx, POLY_FLOAT32, (int64_t[]){4}, 1, POLY_DEVICE_INTERP);
    PolyTensor *index = poly_tensor_empty(ctx, POLY_INT32, (int64_t[]){3}, 1, POLY_DEVICE_INTERP);
    PolyTensor *v = poly_tensor_empty(ctx, POLY_FLOAT32, (int64_t[]){3}, 1, POLY_DEVICE_INTERP);
    ASSERT_INT_EQ(poly_buffer_write(ctx, x->uop_physical, (float[]){0, 1, 2, 3}, 16), 0);
    ASSERT_INT_EQ(poly_buffer_write(ctx, index->uop_physical, (int32_t[]){1, 1, 3}, 12), 0);
    ASSERT_INT_EQ(poly_buffer_write(ctx, v->uop_physical, (float[]){7, 8, 9}, 12), 0);
    int kinds[] = {POLY_INDEX_TENSOR};
    PolyUOp *starts[] = {poly_const_int(ctx, 0)}, *sizes[] = {poly_const_int(ctx, 4)};
    int64_t steps[] = {1};
    PolyTensor *indices[] = {index};
    ASSERT_INT_EQ(poly_tensor_setitem(other, x, kinds, starts, sizes, steps, indices, 1, v), -1);
    ASSERT_INT_EQ(poly_tensor_setitem(ctx, x, kinds, starts, sizes, steps, indices, 1, v), 0);
    ASSERT_INT_EQ(x->uop_logical != NULL, logical);
    PolyTensor *out = poly_tensor_getitem(ctx, x, kinds, starts, sizes, steps, indices, 1);
    ASSERT_NOT_NULL(out);
    ASSERT_INT_EQ(out->uop_logical != NULL, logical);
    ASSERT_PTR_EQ(poly_tensor_binary_crossentropy(other, x, x, 2), NULL);
    ASSERT_PTR_EQ(poly_tensor_binary_crossentropy(ctx, x, x, 3), NULL);
    poly_tensor_release(x);
    poly_tensor_release(index);
    poly_tensor_release(v);
    poly_ctx_collect(ctx);
    float got[3];
    ASSERT_INT_EQ(read_tensor_f32(ctx, out, got, 3), 0);
    ASSERT_FLOAT_EQ(got[0], 8, 0);
    ASSERT_FLOAT_EQ(got[1], 8, 0);
    ASSERT_FLOAT_EQ(got[2], 9, 0);
    poly_tensor_release(out);
    poly_ctx_collect(ctx);
    poly_ctx_destroy(other);
    poly_ctx_destroy(ctx);
  }
  PASS();
}

TEST(pe, pointwise_owners_sign_dtype_and_graph) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_test_buffer(ctx, POLY_INT32, 4);
  PolyUOp *out = poly_sign(ctx, x);
  bool correct = out && out->op == POLY_OP_WHERE && poly_dtype_eq(out->dtype, POLY_INT32);
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(correct);
  PASS();
}

TEST(tensor, pointwise_owners_lifetime_and_admission) {
  PolyTensor *(*unary[])(PolyCtx *, PolyTensor *) = {
      poly_tensor_log10, poly_tensor_atanh, poly_tensor_asinh, poly_tensor_acosh,
      poly_tensor_asin,  poly_tensor_acos,  poly_tensor_atan,  poly_tensor_logsigmoid,
      poly_tensor_sinh,  poly_tensor_cosh,  poly_tensor_erf,   poly_tensor_softsign};
  for (int logical = 0; logical < 2; logical++) {
    PolyCtx *ctx = poly_ctx_new(), *other = poly_ctx_new();
    poly_ctx_set_logical_policy(ctx, logical ? POLY_LOGICAL_ALWAYS : POLY_LOGICAL_NEVER);
    PolyTensor *x = poly_tensor_empty(ctx, POLY_FLOAT32, (int64_t[]){3}, 1, POLY_DEVICE_INTERP);
    float data[] = {0.15f, 0.4f, 0.7f};
    ASSERT_INT_EQ(poly_buffer_write(ctx, x->uop_physical, data, sizeof(data)), 0);
    for (size_t i = 0; i < sizeof(unary) / sizeof(unary[0]); i++) {
      ASSERT_PTR_EQ(unary[i](other, x), NULL);
      PolyTensor *out = unary[i](ctx, x);
      ASSERT_NOT_NULL(out);
      ASSERT_INT_EQ(out->uop_logical != NULL, logical);
      ASSERT_NOT_NULL(out->uop_physical);
      poly_tensor_release(out);
    }
    PolyTensor *out = poly_tensor_softsign(ctx, x);
    ASSERT_PTR_EQ(poly_tensor_binary_crossentropy_logits(ctx, x, x, NULL, 3), NULL);
    ASSERT_PTR_EQ(poly_tensor_binary_crossentropy_logits(other, x, x, NULL, 2), NULL);
    ASSERT_PTR_EQ(poly_tensor_nll_loss(ctx, x, x, NULL, NULL, 2), NULL);
    poly_tensor_release(x);
    poly_ctx_collect(ctx);
    float got[3];
    ASSERT_INT_EQ(read_tensor_f32(ctx, out, got, 3), 0);
    for (int i = 0; i < 3; i++)
      ASSERT_FLOAT_EQ(got[i], data[i] / (1 + data[i]), 1e-6);
    poly_tensor_release(out);
    poly_ctx_destroy(other);
    poly_ctx_destroy(ctx);
  }
  PASS();
}

TEST(pe, scan_owners_dtype_contract) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *u8 = poly_test_buffer(ctx, POLY_UINT8, 4);
  PolyUOp *i8 = poly_test_buffer(ctx, POLY_INT8, 4);
  PolyUOp *sum = poly_cumalu(ctx, u8, 0, POLY_OP_ADD);
  PolyUOp *max = poly_cumalu(ctx, i8, 0, POLY_OP_MAX);
  bool sum_ok = sum && poly_dtype_eq(sum->dtype, POLY_UINT32);
  bool max_ok = max && poly_dtype_eq(max->dtype, POLY_INT8);
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(sum_ok);
  ASSERT_TRUE(max_ok);
  PASS();
}

TEST(pe, scan_owners_nested_gradient_executes) {
  /* The pinned two-stage scan has nested reductions in its gradient.
   * Gated STORE must not end predicate ranges during linearization. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_reshape(ctx, poly_test_buffer(ctx, POLY_FLOAT32, 1026), (int64_t[]){513, 2}, 2);
  PolyUOp *scan = poly_split_cumalu(ctx, x, 0, POLY_OP_ADD);
  PolyUOp *loss = poly_reduce_axis(ctx, POLY_OP_ADD, scan, (int64_t[]){0, 1}, 2);
  PolyUOp *grad = poly_grad(ctx, loss, x);
  ASSERT_NOT_NULL(grad);
  float values[1026];
  int rc = realize_uop(ctx, grad, poly_test_buffer(ctx, POLY_FLOAT32, 1026), values, NULL, NULL, 0);
  bool correct = rc == 0;
  for (int i = 0; correct && i < 1026; i++)
    correct = values[i] == 513 - i / 2;
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(correct);
  PASS();
}

TEST(tensor, scan_owners_pair_lifetime_and_admission) {
  for (int keep_logical = 0; keep_logical < 2; keep_logical++) {
    PolyCtx *ctx = poly_ctx_new(), *other = poly_ctx_new();
    ASSERT_INT_EQ(
        poly_ctx_set_logical_policy(ctx, keep_logical ? POLY_LOGICAL_ALWAYS : POLY_LOGICAL_NEVER), 0
    );
    PolyTensor *x = poly_tensor_empty(ctx, POLY_FLOAT32, (int64_t[]){4}, 1, POLY_DEVICE_INTERP);
    float input[] = {-3, -1, -2, -1};
    ASSERT_INT_EQ(poly_buffer_write(ctx, x->uop_physical, input, sizeof(input)), 0);
    PolyTensor *values = NULL, *indices = NULL;
    ASSERT_INT_EQ(poly_tensor_cummax(other, x, 0, &values, &indices), -1);
    ASSERT_PTR_EQ(values, NULL);
    ASSERT_PTR_EQ(indices, NULL);
    ASSERT_INT_EQ(poly_tensor_cummax(ctx, x, 1, &values, &indices), -1);
    ASSERT_PTR_EQ(values, NULL);
    ASSERT_PTR_EQ(indices, NULL);
    ASSERT_INT_EQ(poly_tensor_cummax(ctx, x, 0, &values, &values), -1);
    ASSERT_INT_EQ(poly_tensor_cummax(ctx, x, -1, &values, &indices), 0);
    ASSERT_NOT_NULL(values->uop_physical);
    ASSERT_NOT_NULL(indices->uop_physical);
    ASSERT_INT_EQ(values->uop_logical != NULL, keep_logical);
    ASSERT_INT_EQ(indices->uop_logical != NULL, keep_logical);
    poly_tensor_release(x);
    float got[4];
    ASSERT_INT_EQ(read_tensor_f32(ctx, values, got, 4), 0);
    const float expected[] = {-3, -1, -1, -1};
    for (int i = 0; i < 4; i++)
      ASSERT_FLOAT_EQ(got[i], expected[i], 0);
    poly_tensor_release(values);
    PolyTensor *index_float =
        poly_tensor_cast_by_id(ctx, indices, poly_dtype_id_by_name("float32"));
    poly_tensor_release(indices);
    ASSERT_INT_EQ(read_tensor_f32(ctx, index_float, got, 4), 0);
    for (int i = 0; i < 4; i++)
      ASSERT_FLOAT_EQ(got[i], i ? 1 : 0, 0);
    poly_tensor_release(index_float);
    poly_ctx_destroy(other);
    poly_ctx_destroy(ctx);
  }
  PASS();
}

TEST(pe, unsigned_scatter_amin_matches_pinned_inverse_max_inverse) {
  /* Pinned scatter amin fills with the positive weak dtype.max literal and reduces through
   * Tensor.min's inverse/MAX/inverse program
   * (mixin/__init__.py:1206-1211, mixin/elementwise.py:379-393). */
  PolyDType dtypes[] = {POLY_UINT8, POLY_UINT16, POLY_UINT32, POLY_UINT64};
  const char *maxima[] = {
      "255",
      "65535",
      "4294967295",
      "18446744073709551615",
  };
  for (int d = 0; d < 4; d++) {
    PolyCtx *ctx = poly_ctx_new();
    PolyDType dtype = dtypes[d];
    PolyUOp *self = poly_reshape(ctx, poly_test_buffer(ctx, dtype, 2), (int64_t[]){1, 2}, 2);
    PolyUOp *idx = poly_reshape(ctx, poly_test_buffer(ctx, POLY_INT32, 2), (int64_t[]){1, 2}, 2);
    PolyUOp *src = poly_reshape(ctx, poly_test_buffer(ctx, dtype, 2), (int64_t[]){1, 2}, 2);
    PolyUOp *result = poly_scatter_reduce(ctx, self, 1, idx, src, "amin", 0);
    ASSERT_NOT_NULL(result);

    int n_topo = 0, neg_count = 0;
    bool saw_positive_max = false;
    PolyUOp **topo = poly_toposort_alloc(ctx, result, &n_topo);
    ASSERT_NOT_NULL(topo);
    for (int i = 0; i < n_topo; i++) {
      PolyUOp *u = topo[i];
      if (u->op == POLY_OP_NEG) neg_count++;
      if (u->op != POLY_OP_CONST || !poly_dtype_eq(u->dtype, POLY_WEAKINT) ||
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
        (float *)self_data.bytes,
        (float *)idx_data,
        (float *)src_data.bytes,
    };
    ASSERT_INT_EQ(
        realize_uop(ctx, result, poly_test_buffer(ctx, dtype, 2), output.bytes, leaves, data, 3), 0
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
  PolyUOp *out_idx_i32 = poly_test_buffer(ctx, POLY_INT32, 10);
  float got_vals[10] = {0}, got_idx[10] = {0};
  int32_t got_idx_i32[10] = {0};
  ASSERT_INT_EQ(realize_uop(ctx, vals, out_vals, got_vals, leaves, ld, 1), 0);
  ASSERT_INT_EQ(
      realize_uop(ctx, poly_cast(ctx, idx, POLY_FLOAT32), out_idx, got_idx, leaves, ld, 1), 0
  );
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
  PolyUOp *out_top_idx_i32 = poly_test_buffer(ctx, POLY_INT32, 4);
  float got_top_vals[4] = {0}, got_top_idx[4] = {0};
  int32_t got_top_idx_i32[4] = {0};
  ASSERT_INT_EQ(realize_uop(ctx, top_vals, out_top_vals, got_top_vals, leaves, ld, 1), 0);
  ASSERT_INT_EQ(
      realize_uop(
          ctx, poly_cast(ctx, top_idx, POLY_FLOAT32), out_top_idx, got_top_idx, leaves, ld, 1
      ),
      0
  );
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
  PolyUOp *to_cuda = poly_copy_to_device_uop(ctx, logical, cuda);
  PolyUOp *physical = poly_copy_to_device_uop(ctx, to_cuda, cpu);
  PolyTensor *src =
      poly_tensor_create_with_roots(ctx, logical, physical, POLY_TENSOR_VALUE, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(src);

  PolyUOp *expected_logical_values = NULL, *expected_logical_indices = NULL;
  PolyUOp *expected_physical_values = NULL, *expected_physical_indices = NULL;
  ASSERT_INT_EQ(
      poly_topk(ctx, logical, 2, 0, 1, 1, &expected_logical_values, &expected_logical_indices), 0
  );
  ASSERT_INT_EQ(
      poly_topk(ctx, physical, 2, 0, 1, 1, &expected_physical_values, &expected_physical_indices), 0
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
  PolyTensor *a = poly_tensor_empty(ctx, POLY_FLOAT32, matrix_shape, 2, POLY_DEVICE_CPU);
  PolyTensor *b = poly_tensor_empty(ctx, POLY_FLOAT32, vector_shape, 1, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(a);
  ASSERT_NOT_NULL(b);

  PolyUOp *logical_q = NULL, *logical_r = NULL;
  PolyUOp *physical_q = NULL, *physical_r = NULL;
  ASSERT_INT_EQ(poly_qr_ex(ctx, a->uop_logical, POLY_QR_COMPLETE, &logical_q, &logical_r), 0);
  ASSERT_INT_EQ(poly_qr_ex(ctx, a->uop_physical, POLY_QR_COMPLETE, &physical_q, &physical_r), 0);
  PolyTensor *q = NULL, *r = NULL;
  ASSERT_INT_EQ(poly_tensor_qr_ex(ctx, a, POLY_QR_COMPLETE, &q, &r), 0);
  ASSERT_NOT_NULL(q);
  ASSERT_NOT_NULL(r);
  ASSERT_PTR_EQ(q->uop_logical, logical_q);
  ASSERT_PTR_EQ(q->uop_physical, physical_q);
  ASSERT_PTR_EQ(r->uop_logical, logical_r);
  ASSERT_PTR_EQ(r->uop_physical, physical_r);

  PolyUOp *logical_tri = poly_triangular_solve(ctx, a->uop_logical, b->uop_logical, 0, 0, 0);
  PolyUOp *physical_tri = poly_triangular_solve(ctx, a->uop_physical, b->uop_physical, 0, 0, 0);
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

  PolyUOp *logical_chol_solve = poly_cholesky_solve(ctx, chol->uop_logical, b->uop_logical, 0);
  PolyUOp *physical_chol_solve = poly_cholesky_solve(ctx, chol->uop_physical, b->uop_physical, 0);
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
  PolyTensor *tall = poly_tensor_empty(ctx, POLY_FLOAT32, tall_shape, 2, POLY_DEVICE_CPU);
  PolyTensor *tall_rhs = poly_tensor_empty(ctx, POLY_FLOAT32, tall_rhs_shape, 1, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(tall);
  ASSERT_NOT_NULL(tall_rhs);
  PolyUOp *logical_lstsq = poly_lstsq(ctx, tall->uop_logical, tall_rhs->uop_logical);
  PolyUOp *physical_lstsq = poly_lstsq(ctx, tall->uop_physical, tall_rhs->uop_physical);
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
          ctx, q->uop_physical, poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CPU),
          q_values, a_leaves, a_leaf_data, 1
      ),
      0
  );
  ASSERT_INT_EQ(
      realize_uop(
          ctx, r->uop_physical, poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CPU),
          r_values, a_leaves, a_leaf_data, 1
      ),
      0
  );
  float expected_q[4] = {-0.89442706f, 0.44721359f, -0.44721359f, -0.89442718f};
  float expected_r[4] = {-4.47213554f, -4.02492189f, 5.4168083e-9f, -3.57770872f};
  for (int i = 0; i < 4; i++) {
    ASSERT_FLOAT_EQ(q_values[i], expected_q[i], 2e-5f);
    ASSERT_FLOAT_EQ(r_values[i], expected_r[i], 2e-5f);
  }

  float solve_values[2] = {0};
  PolyUOp *solve_leaves[2] = {a_leaf, b_leaf};
  float *solve_leaf_data[2] = {a_data, b_data};
  ASSERT_INT_EQ(
      realize_uop(
          ctx, solve->uop_physical,
          poly_test_buffer_on_device(ctx, POLY_FLOAT32, 2, POLY_DEVICE_CPU), solve_values,
          solve_leaves, solve_leaf_data, 2
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
  PolyTensor *src = poly_tensor_empty(owner, POLY_FLOAT32, (int64_t[]){4}, 1, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(src);

  PolyTensor *sort_values = NULL, *sort_indices = NULL;
  PolyTensor *topk_values = NULL, *topk_indices = NULL;
  int sort_rc = poly_tensor_sort(foreign, src, 0, 0, &sort_values, &sort_indices);
  int topk_rc = poly_tensor_topk(foreign, src, 2, 0, 1, 1, &topk_values, &topk_indices);

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
   * (mixin/reduce.py:13-44, mixin/op.py:367-392). */
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
  ASSERT_INT_EQ(sum_default_root->src[0]->op, POLY_OP_REDUCE);
  ASSERT_INT_EQ(sum_default_root->src[0]->src[0]->op, POLY_OP_PERMUTE);
  ASSERT_INT_EQ(sum_default_root->src[0]->src[0]->src[0]->op, POLY_OP_CAST);
  ASSERT_TRUE(poly_dtype_eq(sum_default_root->src[0]->src[0]->src[0]->dtype, POLY_FLOAT32));

  PolyTensor *sum_explicit = poly_tensor_sum_dtype_by_id(ctx, a, axis, 1, false, f32);
  PolyUOp *sum_explicit_root = poly_tensor_uop(sum_explicit);
  ASSERT_NOT_NULL(sum_explicit_root);
  ASSERT_INT_EQ(sum_explicit_root->op, POLY_OP_REDUCE);
  ASSERT_INT_EQ(sum_explicit_root->src[0]->op, POLY_OP_PERMUTE);
  ASSERT_INT_EQ(sum_explicit_root->src[0]->src[0]->op, POLY_OP_CAST);
  ASSERT_TRUE(poly_dtype_eq(sum_explicit_root->dtype, POLY_FLOAT32));

  PolyTensor *dot_default = poly_tensor_dot(ctx, a, b);
  PolyUOp *dot_default_root = poly_tensor_uop(dot_default);
  ASSERT_NOT_NULL(dot_default_root);
  ASSERT_INT_EQ(dot_default_root->op, POLY_OP_CAST);
  ASSERT_INT_EQ(dot_default_root->src[0]->op, POLY_OP_REDUCE);
  ASSERT_INT_EQ(dot_default_root->src[0]->src[0]->op, POLY_OP_PERMUTE);
  ASSERT_INT_EQ(dot_default_root->src[0]->src[0]->src[0]->op, POLY_OP_CAST);
  ASSERT_INT_EQ(dot_default_root->src[0]->src[0]->src[0]->src[0]->op, POLY_OP_MUL);

  PolyTensor *dot_explicit = poly_tensor_dot_dtype_by_id(ctx, a, b, f32);
  PolyUOp *dot_explicit_root = poly_tensor_uop(dot_explicit);
  ASSERT_NOT_NULL(dot_explicit_root);
  ASSERT_INT_EQ(dot_explicit_root->op, POLY_OP_REDUCE);
  ASSERT_INT_EQ(dot_explicit_root->src[0]->op, POLY_OP_PERMUTE);
  ASSERT_INT_EQ(dot_explicit_root->src[0]->src[0]->op, POLY_OP_CAST);
  ASSERT_INT_EQ(dot_explicit_root->src[0]->src[0]->src[0]->op, POLY_OP_MUL);
  ASSERT_TRUE(poly_dtype_eq(dot_explicit_root->dtype, POLY_FLOAT32));

  PolyTensor *b_f32 = poly_tensor_empty(ctx, POLY_FLOAT32, b_shape, 2, POLY_DEVICE_CPU);
  PolyTensor *dot_mixed = poly_tensor_dot(ctx, a, b_f32);
  PolyUOp *dot_mixed_root = poly_tensor_uop(dot_mixed);
  ASSERT_NOT_NULL(dot_mixed_root);
  ASSERT_INT_EQ(dot_mixed_root->op, POLY_OP_REDUCE);
  ASSERT_TRUE(poly_dtype_eq(dot_mixed_root->dtype, POLY_FLOAT32));
  ASSERT_INT_EQ(dot_mixed_root->src[0]->op, POLY_OP_PERMUTE);
  PolyUOp *mixed_mul = dot_mixed_root->src[0]->src[0];
  ASSERT_INT_EQ(mixed_mul->op, POLY_OP_MUL);
  ASSERT_TRUE(poly_dtype_eq(mixed_mul->dtype, POLY_FLOAT32));
  ASSERT_INT_EQ(mixed_mul->src[0]->op, POLY_OP_CAST);
  ASSERT_TRUE(poly_dtype_eq(mixed_mul->src[0]->dtype, POLY_FLOAT32));
  ASSERT_FALSE(mixed_mul->src[1]->op == POLY_OP_CAST);

  PolyTensor *v0 = poly_tensor_empty(ctx, POLY_FLOAT32, (int64_t[]){3}, 1, POLY_DEVICE_CPU);
  PolyTensor *v1 = poly_tensor_empty(ctx, POLY_FLOAT32, (int64_t[]){3}, 1, POLY_DEVICE_CPU);
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

  PolyTensor *mixed = poly_tensor_conv2d(ctx, x, w_float, b_float, 1, NULL, NULL, NULL, 0);
  PolyTensor *half_default = poly_tensor_conv2d(ctx, x, w_half, b_half, 1, NULL, NULL, NULL, 0);
  PolyTensor *half_explicit =
      poly_tensor_conv2d_dtype_by_id(ctx, x, w_half, b_half, 1, NULL, NULL, NULL, 0, f32);
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
      if (topo[i]->op == POLY_OP_REDUCE) ASSERT_TRUE(poly_dtype_eq(topo[i]->dtype, POLY_FLOAT32));
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
  /* Current Tensor.dot reaches ElementwiseMixin._broadcasted after reshaping
   * and transposing the operands.  Shape broadcasting is implicit in UOp
   * inference, so neither operand receives an explicit EXPAND. */
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
  ASSERT_INT_EQ(n_expand, 0);
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
      2, 0, 0, 1, 3, 0, -2, 0.5f, 4, 3, 0, 0, 1, 4, 0, -2, 0.5f, 5,
  };
  float b_batch[] = {
      2, 1, 7, 2, 9, 3, 3, 2, 8, 3, 10, 4,
  };

  float expect_lower_vec[] = {1, 2, 2.5f};
  float expect_lower_mat[] = {1, 0.5f, 2, 0.5f, 2.5f, 0.9375f};
  float expect_upper_mat[] = {0.8541666865f, 0.3958333433f, 0.8333333135f,
                              0.1666666716f, 2.25f,         0.75f};
  float expect_lower_trans[] = {2.2708332539f, 0.9791666865f, 1.9583333731f,
                                0.5416666865f, 2.25f,         0.75f};
  float expect_upper_trans[] = {
      1, 0.5f, 2.6666667461f, 0.8333333135f, 0.7916666865f, 0.2708333433f};
  float expect_lower_unit[] = {2, 1, 5, 1, 10.5f, 4.5f};
  float expect_batch[] = {
      1, 0.5f,          2,     0.5f,          2.5f,          0.9375f,
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
      2.4494898319f, 0, 0, 0.8164966106f, 2.0816659927f, 0, 0.4082483053f, 0.8006407619f,
      1.7867029905f,
  };
  float a4_eye[] = {
      4, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4,
  };
  float e4_eye[] = {
      2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2,
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
      {shape1, 2, a1, 0, e1, 1},          {shape2, 2, a2, 0, e2, 4},
      {shape2, 2, a2, 1, e2_upper, 4},    {shape3, 2, a3, 0, e3, 9},
      {shape4, 2, a4_eye, 0, e4_eye, 16}, {shape_batch, 3, abat, 0, ebat, 8},
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
  ASSERT_INT_EQ(
      realize_uop(ctx, x_vec, poly_buffer_f32(ctx, 2), got_vec, vec_leaves, vec_ld, 2), 0
  );
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
  ASSERT_INT_EQ(
      realize_uop(ctx, x_mat, poly_buffer_f32(ctx, 4), got_mat, mat_leaves, mat_ld, 2), 0
  );
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
      realize_uop(
          ctx, px_vec, poly_buffer_f32(ctx, 2), got_pivot_vec, pivot_vec_leaves, pivot_vec_ld, 2
      ),
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
      realize_uop(
          ctx, px_mat, poly_buffer_f32(ctx, 4), got_pivot_mat, pivot_mat_leaves, pivot_mat_ld, 2
      ),
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
  PolyTensor *public_ah =
      poly_tensor_from_host_by_id(ctx, public_a, sizeof(public_a), f32_id, public_a_shape, 2);
  PolyTensor *public_bh =
      poly_tensor_from_host_by_id(ctx, public_b, sizeof(public_b), f32_id, public_b_shape, 1);
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
      0, 0.4f, 1, 1.2f, 0.3636363745f, 0.6363636255f, 0.9090909362f, 1.0909091234f,
  };
  float expect_batch_vec[] = {-0.2f, 1.4f, 0.27272728f, 1.1818182f};
  PolyUOp *ab = make_buf(ctx, batch_shape, 3);
  PolyUOp *bb = make_buf(ctx, batch_shape, 3);
  PolyUOp *xb = poly_solve(ctx, ab, bb);
  ASSERT_NOT_NULL(xb);
  float got_batch[8] = {0};
  PolyUOp *batch_leaves[] = {base_buf(ab), base_buf(bb)};
  float *batch_ld[] = {a_batch, b_batch};
  ASSERT_INT_EQ(
      realize_uop(ctx, xb, poly_buffer_f32(ctx, 8), got_batch, batch_leaves, batch_ld, 2), 0
  );
  for (int i = 0; i < 8; i++)
    ASSERT_FLOAT_EQ(got_batch[i], expect_batch[i], 3e-4f);

  PolyUOp *bbv = make_buf(ctx, shape2, 2);
  PolyUOp *xbv = poly_solve(ctx, ab, bbv);
  ASSERT_NOT_NULL(xbv);
  float got_batch_vec[4] = {0};
  PolyUOp *batch_vec_leaves[] = {base_buf(ab), base_buf(bbv)};
  float *batch_vec_ld[] = {a_batch, b_batch_vec};
  ASSERT_INT_EQ(
      realize_uop(
          ctx, xbv, poly_buffer_f32(ctx, 4), got_batch_vec, batch_vec_leaves, batch_vec_ld, 2
      ),
      0
  );
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
      realize_uop(
          ctx, pxbv, poly_buffer_f32(ctx, 4), got_pivot_batch_vec, pivot_batch_leaves,
          pivot_batch_ld, 2
      ),
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
      realize_uop(
          ctx, xbv_single, poly_buffer_f32(ctx, 4), got_broadcast_vec, broadcast_vec_leaves,
          broadcast_vec_ld, 2
      ),
      0
  );
  for (int i = 0; i < 4; i++)
    ASSERT_FLOAT_EQ(got_broadcast_vec[i], expect_broadcast_vec[i], 5e-4f);

  const int64_t singleton_batch_shape[] = {1, 2, 2};
  PolyUOp *bb_single_batch = make_buf(ctx, singleton_batch_shape, 3);
  PolyUOp *xb_single_batch = poly_solve(ctx, ab, bb_single_batch);
  ASSERT_NOT_NULL(xb_single_batch);
  float expect_single_batch[] = {
      0, 0.4f, 1, 1.2f, 0.09090909f, 0.36363637f, 0.7272727f, 0.9090909f,
  };
  float got_single_batch[8] = {0};
  PolyUOp *single_batch_leaves[] = {base_buf(ab), base_buf(bb_single_batch)};
  float *single_batch_ld[] = {a_batch, rhs_mat};
  ASSERT_INT_EQ(
      realize_uop(
          ctx, xb_single_batch, poly_buffer_f32(ctx, 8), got_single_batch, single_batch_leaves,
          single_batch_ld, 2
      ),
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
      realize_uop(
          ctx, tri_x, poly_buffer_f32(ctx, 4), got_tri_broadcast_vec, tri_leaves, tri_ld, 2
      ),
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
      1, 0, 1, 1, 1, 2, 1, 0, 1, 1.5f, 1, 3,
  };
  float tall_b_batch[] = {
      1, 0.5f, 2, 1, 2.5f, 1.5f, 1.25f, 0.75f, 2.25f, 1.25f, 2.75f, 1.75f,
  };
  float tall_b_batch_vec[] = {1, 2, 2.5f, 1.25f, 2.25f, 2.75f};
  float expect_lstsq_vec[] = {1.0833334f, 0.75f};
  float expect_lstsq_mat[] = {1.0833334f, 0.5f, 0.75f, 0.5f};
  float expect_lstsq_batch[] = {
      1.0833334f, 0.5f, 0.75f, 0.5f, 1.3333334f, 0.75f, 0.5f, 0.33333334f,
  };
  float expect_lstsq_batch_vec[] = {1.0833334f, 0.75f, 1.3333334f, 0.5f};

  PolyUOp *la = make_buf(ctx, tall_shape, 2);
  PolyUOp *lbv = make_buf(ctx, tall_vec_shape, 1);
  PolyUOp *lxv = poly_lstsq(ctx, la, lbv);
  ASSERT_NOT_NULL(lxv);
  float got_lxv[2] = {0};
  PolyUOp *lvec_leaves[] = {base_buf(la), base_buf(lbv)};
  float *lvec_ld[] = {tall_a, tall_b_vec};
  ASSERT_INT_EQ(
      realize_uop(ctx, lxv, poly_buffer_f32(ctx, 2), got_lxv, lvec_leaves, lvec_ld, 2), 0
  );
  for (int i = 0; i < 2; i++)
    ASSERT_FLOAT_EQ(got_lxv[i], expect_lstsq_vec[i], 4e-4f);

  PolyUOp *lB = make_buf(ctx, tall_mat_shape, 2);
  PolyUOp *lxm = poly_lstsq(ctx, la, lB);
  ASSERT_NOT_NULL(lxm);
  float got_lxm[4] = {0};
  PolyUOp *lmat_leaves[] = {base_buf(la), base_buf(lB)};
  float *lmat_ld[] = {tall_a, tall_b_mat};
  ASSERT_INT_EQ(
      realize_uop(ctx, lxm, poly_buffer_f32(ctx, 4), got_lxm, lmat_leaves, lmat_ld, 2), 0
  );
  for (int i = 0; i < 4; i++)
    ASSERT_FLOAT_EQ(got_lxm[i], expect_lstsq_mat[i], 4e-4f);

  PolyUOp *lab = make_buf(ctx, tall_batch_shape, 3);
  PolyUOp *lbb = make_buf(ctx, tall_batch_rhs_shape, 3);
  PolyUOp *lxb = poly_lstsq(ctx, lab, lbb);
  ASSERT_NOT_NULL(lxb);
  float got_lxb[8] = {0};
  PolyUOp *lbatch_leaves[] = {base_buf(lab), base_buf(lbb)};
  float *lbatch_ld[] = {tall_a_batch, tall_b_batch};
  ASSERT_INT_EQ(
      realize_uop(ctx, lxb, poly_buffer_f32(ctx, 8), got_lxb, lbatch_leaves, lbatch_ld, 2), 0
  );
  for (int i = 0; i < 8; i++)
    ASSERT_FLOAT_EQ(got_lxb[i], expect_lstsq_batch[i], 5e-4f);

  PolyUOp *lbbv = make_buf(ctx, tall_batch_vec_shape, 2);
  PolyUOp *lxbv = poly_lstsq(ctx, lab, lbbv);
  ASSERT_NOT_NULL(lxbv);
  float got_lxbv[4] = {0};
  PolyUOp *lbatch_vec_leaves[] = {base_buf(lab), base_buf(lbbv)};
  float *lbatch_vec_ld[] = {tall_a_batch, tall_b_batch_vec};
  ASSERT_INT_EQ(
      realize_uop(
          ctx, lxbv, poly_buffer_f32(ctx, 4), got_lxbv, lbatch_vec_leaves, lbatch_vec_ld, 2
      ),
      0
  );
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
      realize_uop(
          ctx, lxbv_single, poly_buffer_f32(ctx, 4), got_lxbv_single, lbatch_broadcast_leaves,
          lbatch_broadcast_ld, 2
      ),
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
      -0.33333334f, -0.33333334f, 0.6666667f, 1.6666666f, 1.3333334f, 2.3333333f,
  };
  PolyUOp *lwa = make_buf(ctx, wide_shape, 2);
  PolyUOp *lwv = make_buf(ctx, wide_vec_shape, 1);
  PolyUOp *lwxv = poly_lstsq(ctx, lwa, lwv);
  ASSERT_NOT_NULL(lwxv);
  float got_lwxv[3] = {0};
  PolyUOp *lwide_vec_leaves[] = {base_buf(lwa), base_buf(lwv)};
  float *lwide_vec_ld[] = {wide_a, wide_b_vec};
  ASSERT_INT_EQ(
      realize_uop(ctx, lwxv, poly_buffer_f32(ctx, 3), got_lwxv, lwide_vec_leaves, lwide_vec_ld, 2),
      0
  );
  for (int i = 0; i < 3; i++)
    ASSERT_FLOAT_EQ(got_lwxv[i], expect_wide_vec[i], 6e-4f);

  PolyUOp *lwB = make_buf(ctx, wide_mat_shape, 2);
  PolyUOp *lwxm = poly_lstsq(ctx, lwa, lwB);
  ASSERT_NOT_NULL(lwxm);
  float got_lwxm[6] = {0};
  PolyUOp *lwide_mat_leaves[] = {base_buf(lwa), base_buf(lwB)};
  float *lwide_mat_ld[] = {wide_a, wide_b_mat};
  ASSERT_INT_EQ(
      realize_uop(ctx, lwxm, poly_buffer_f32(ctx, 6), got_lwxm, lwide_mat_leaves, lwide_mat_ld, 2),
      0
  );
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
  ASSERT_INT_EQ(
      realize_uop(
          ctx, rd_sq_xv, poly_buffer_f32(ctx, 2), got_rd_sq_v, rd_sq_v_leaves, rd_sq_v_ld, 2
      ),
      0
  );
  ASSERT_FLOAT_EQ(got_rd_sq_v[0], 1.5f, 1e-3f);
  ASSERT_FLOAT_EQ(got_rd_sq_v[1], 1.5f, 1e-3f);

  PolyUOp *rd_sq_bm = make_buf(ctx, wide_mat_shape, 2);
  PolyUOp *rd_sq_xm = poly_lstsq(ctx, rd_sq_a, rd_sq_bm);
  ASSERT_NOT_NULL(rd_sq_xm);
  float got_rd_sq_m[4] = {0};
  PolyUOp *rd_sq_m_leaves[] = {base_buf(rd_sq_a), base_buf(rd_sq_bm)};
  float *rd_sq_m_ld[] = {rankdef_square_a, rankdef_square_bm};
  ASSERT_INT_EQ(
      realize_uop(
          ctx, rd_sq_xm, poly_buffer_f32(ctx, 4), got_rd_sq_m, rd_sq_m_leaves, rd_sq_m_ld, 2
      ),
      0
  );
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
  ASSERT_INT_EQ(
      realize_uop(
          ctx, rd_tall_x, poly_buffer_f32(ctx, 2), got_rd_tall, rd_tall_leaves, rd_tall_ld, 2
      ),
      0
  );
  ASSERT_FLOAT_EQ(got_rd_tall[0], 0.5f, 1e-3f);
  ASSERT_FLOAT_EQ(got_rd_tall[1], 0.5f, 1e-3f);

  PolyUOp *rd_wide_a = make_buf(ctx, rankdef_wide_shape, 2);
  PolyUOp *rd_wide_bv = make_buf(ctx, wide_vec_shape, 1);
  PolyUOp *rd_wide_x = poly_lstsq(ctx, rd_wide_a, rd_wide_bv);
  ASSERT_NOT_NULL(rd_wide_x);
  float got_rd_wide[3] = {0};
  PolyUOp *rd_wide_leaves[] = {base_buf(rd_wide_a), base_buf(rd_wide_bv)};
  float *rd_wide_ld[] = {rankdef_wide_a, rankdef_square_bv};
  ASSERT_INT_EQ(
      realize_uop(
          ctx, rd_wide_x, poly_buffer_f32(ctx, 3), got_rd_wide, rd_wide_leaves, rd_wide_ld, 2
      ),
      0
  );
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
  PolyTensor *chol_ah = poly_tensor_from_host_by_id(ctx, spd, sizeof(spd), f32_id, shape2, 2);
  PolyTensor *chol_bh =
      poly_tensor_from_host_by_id(ctx, rhs_mat, sizeof(rhs_mat), f32_id, shape2, 2);
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
  PolyTensor *solve_ah = poly_tensor_from_host_by_id(ctx, a2, sizeof(a2), f32_id, shape2, 2);
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
  PolyUOp *b = poly_test_buffer(ctx, POLY_FLOAT32, 100);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, b), 1);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, b)[0], 100);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(shape_uop, buffer_dynamic) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *var =
      poly_uop_variable(ctx, "batch", poly_arg_int(1), poly_arg_int(32), POLY_WEAKINT, 1, false);
  PolyUOp *buf = poly_test_buffer_var(ctx, POLY_FLOAT32, var, (int64_t[]){10}, 1);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, buf), 2);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, buf)[0], 32);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, buf)[1], 10);
  ASSERT_PTR_EQ(poly_uop_shape_dim(ctx, buf, 0), var);
  ASSERT_INT_EQ(poly_uop_shape_dim(ctx, buf, 1)->op, POLY_OP_CONST);
  ASSERT_INT_EQ(poly_uop_shape_dim(ctx, buf, 1)->arg.i, 10);

  PolyUOp *bound = poly_uop_bind(ctx, var, 7);
  PolyUOp *bound_buf = poly_test_buffer_var(ctx, POLY_FLOAT32, bound, (int64_t[]){10}, 1);
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
   * ordered FUNCTION arguments, while AFTER._min_max retains the PARAM range
   * (tinygrad/uop/ops.py:242-253,628-630,1010-1012,1691). */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *empty = poly_uop(ctx, POLY_OP_STACK, POLY_VOID, NULL, 0, poly_arg_none());
  PolyParamArg dim_arg = {
      .slot = 1,
      .name = "size",
      .min_val = poly_arg_int(1),
      .max_val = poly_arg_int(8),
      .has_minmax = true,
      .addrspace = POLY_ADDR_GLOBAL,
  };
  PolyUOp *dim_param = poly_uop1(ctx, POLY_OP_PARAM, POLY_WEAKINT, empty, poly_arg_param(&dim_arg));
  PolyUOp *body_dims[2] = {dim_param, poly_const_int(ctx, 4)};
  PolyUOp *body_shape = poly_uop(ctx, POLY_OP_STACK, POLY_WEAKINT, body_dims, 2, poly_arg_none());
  PolyParamArg value_arg = {
      .slot = 0,
      .addrspace = POLY_ADDR_GLOBAL,
      .axis = 1,
      .has_axis = true,
  };
  PolyUOp *value =
      poly_uop1(ctx, POLY_OP_PARAM, POLY_FLOAT32, body_shape, poly_arg_param(&value_arg));
  PolyUOp *body = poly_uop1(ctx, POLY_OP_TUPLE, POLY_VOID, value, poly_arg_none());

  PolyUOp *actual = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 20, POLY_DEVICE_CPU);
  actual = poly_reshape(ctx, actual, (int64_t[]){5, 4}, 2);
  PolyUOp *size =
      poly_uop_variable(ctx, "size", poly_arg_int(1), poly_arg_int(8), POLY_WEAKINT, 1, false);
  PolyUOp *bound = poly_uop_bind(ctx, size, 5);
  PolyUOp *function_src[3] = {body, actual, bound};
  PolyUOp *function =
      poly_uop(ctx, POLY_OP_FUNCTION, POLY_VOID, function_src, 3, poly_arg_str("symbolic_shape"));
  PolyUOp *selected = poly_uop1(ctx, POLY_OP_GETTUPLE, POLY_FLOAT32, function, poly_arg_int(0));

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
  PolyUOp *i =
      poly_uop_variable(ctx, "i", poly_arg_int(0), poly_arg_int(8), POLY_WEAKINT, 1, false);
  PolyUOp *ib = poly_uop_bind(ctx, i, 4);
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
  for (int j = 0; j < 20; j++)
    data[j] = (float)j;
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
  PolyUOp *n =
      poly_uop_variable(ctx, "n", poly_arg_int(1), poly_arg_int(8), POLY_WEAKINT, 1, false);
  PolyUOp *nb = poly_uop_bind(ctx, n, 3);
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
  PolyUOp *n =
      poly_uop_variable(ctx, "n", poly_arg_int(1), poly_arg_int(4), POLY_WEAKINT, 1, false);
  PolyUOp *nb = poly_uop_bind(ctx, n, 3);
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

  PolyUOp *n =
      poly_uop_variable(ctx, "n", poly_arg_int(1), poly_arg_int(4), POLY_WEAKINT, 1, false);
  PolyUOp *nb = poly_uop_bind(ctx, n, 3);
  PolyUOp *lhs = poly_test_buffer_var(ctx, POLY_FLOAT32, nb, (int64_t[]){10}, 1);
  ASSERT_NOT_NULL(lhs);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, lhs), 2);
  ASSERT_PTR_EQ(poly_uop_unbind_var(poly_uop_shape_dim(ctx, lhs, 0)), n);

  PolyUOp *target[2] = {nb, poly_const_int(ctx, 10)};

  PolyUOp *rhs_1x10 = poly_reshape(ctx, poly_buffer_f32(ctx, 10), (int64_t[]){1, 10}, 2);
  PolyUOp *rhs_1x10_expanded = poly_expand_uop(ctx, rhs_1x10, target, 2);
  ASSERT_NOT_NULL(rhs_1x10_expanded);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, rhs_1x10_expanded), 2);
  ASSERT_PTR_EQ(poly_uop_unbind_var(poly_uop_shape_dim(ctx, rhs_1x10_expanded, 0)), n);
  ASSERT_PTR_EQ(poly_uop_shape_dim(ctx, lhs, 1), poly_uop_shape_dim(ctx, rhs_1x10_expanded, 1));

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
  ASSERT_PTR_EQ(poly_uop_shape_dim(ctx, lhs, 1), poly_uop_shape_dim(ctx, rhs_10_expanded, 1));

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
  PolyUOp *i =
      poly_uop_variable(ctx, "i", poly_arg_int(0), poly_arg_int(6), POLY_WEAKINT, 1, false);
  PolyUOp *ib = poly_uop_bind(ctx, i, 0);
  PolyUOp *starts[2] = {ib, poly_const_int(ctx, 0)};
  PolyUOp *sizes[2] = {poly_const_int(ctx, 2), poly_const_int(ctx, 10)};
  PolyUOp *labels = poly_shrink_uop(ctx, labels_all, starts, sizes, 2);
  ASSERT_NOT_NULL(labels);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, labels), 2);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, labels)[0], 2);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, labels)[1], 10);

  PolyUOp *smoothed = poly_add(
      ctx, poly_mul(ctx, labels, poly_const_float(ctx, 0.9f)), poly_const_float(ctx, 0.01f)
  );
  PolyUOp *static_logp = make_buf(ctx, (int64_t[]){2, 10}, 2);
  PolyUOp *prod = poly_mul(ctx, smoothed, static_logp);
  ASSERT_NOT_NULL(prod);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, prod), 2);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, prod)[0], 2);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, prod)[1], 10);

  PolyUOp *per_row = poly_reduce_axis(ctx, POLY_OP_ADD, prod, (int64_t[]){1}, 1);
  ASSERT_NOT_NULL(per_row);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, per_row), 1);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, per_row)[0], 2);
  PolyUOp *loss_rows = poly_reshape(ctx, per_row, (int64_t[]){2}, 1);
  ASSERT_NOT_NULL(loss_rows);
  ASSERT_PTR_EQ(loss_rows, per_row);
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

  PolyUOp *a0 = poly_test_buffer(ctx1, POLY_FLOAT32, 4);
  PolyUOp *a1 = poly_test_buffer(ctx1, POLY_FLOAT32, 4);
  PolyUOp *n =
      poly_uop_variable(ctx1, "N", poly_arg_int(1), poly_arg_int(8), POLY_WEAKINT, 1, false);
  PolyUOp *ad = poly_test_buffer_var(ctx1, POLY_FLOAT32, n, NULL, 0);
  PolyUOp *b0 = poly_test_buffer(ctx2, POLY_FLOAT32, 4);
  PolyUOp *m =
      poly_uop_variable(ctx2, "M", poly_arg_int(1), poly_arg_int(8), POLY_WEAKINT, 1, false);
  PolyUOp *bd = poly_test_buffer_var(ctx2, POLY_FLOAT32, m, NULL, 0);

  ASSERT_NOT_NULL(a0);
  ASSERT_NOT_NULL(a1);
  ASSERT_NOT_NULL(ad);
  ASSERT_NOT_NULL(b0);
  ASSERT_NOT_NULL(bd);
  PolyUOp *roots[] = {a0, a1, ad, b0, bd};
  PolyUOp *buffers[5];
  for (int i = 0; i < 5; i++) {
    buffers[i] = poly_uop_base(roots[i]);
    ASSERT_NOT_NULL(buffers[i]);
    ASSERT_INT_EQ(buffers[i]->arg.kind, POLY_ARG_PARAM);
    ASSERT_NOT_NULL(buffers[i]->arg.param);
    ASSERT_INT_EQ(buffers[i]->src[0]->op, POLY_OP_CONST);
  }
  ASSERT_INT_EQ(buffers[0]->arg.param->slot, 0);
  ASSERT_INT_EQ(buffers[1]->arg.param->slot, 1);
  ASSERT_INT_EQ(buffers[2]->arg.param->slot, 2);
  ASSERT_INT_EQ(buffers[3]->arg.param->slot, 0);
  ASSERT_INT_EQ(buffers[4]->arg.param->slot, 1);

  poly_ctx_destroy(ctx2);
  poly_ctx_destroy(ctx1);
  PASS();
}

TEST(shape_uop, reshape) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *r = poly_reshape(ctx, poly_test_buffer(ctx, POLY_FLOAT32, 24), (int64_t[]){2, 3, 4}, 3);
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
  ASSERT_INT_EQ(filled->src[0]->op, POLY_OP_CONST);
  ASSERT_TRUE(filled->src[0]->arg.b);
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
  /* Current UOp._rop permutes reduced axes to the prefix and stores only the
   * reduction op and prefix count (uop/ops.py:630-638). */
  ASSERT_EQ(r->op, POLY_OP_REDUCE);
  ASSERT_EQ(r->arg.kind, POLY_ARG_REDUCE);
  ASSERT_EQ(r->arg.reduce.op, POLY_OP_ADD);
  ASSERT_INT_EQ(r->arg.reduce.num_axes, 1);
  ASSERT_INT_EQ(r->n_src, 1);
  ASSERT_EQ(r->src[0]->op, POLY_OP_PERMUTE);
  ASSERT_PTR_EQ(r->src[0]->src[0], b);
  ASSERT_INT_EQ(r->src[0]->arg.int_tuple.vals[0], 1);
  ASSERT_INT_EQ(r->src[0]->arg.int_tuple.vals[1], 0);
  ASSERT_INT_EQ(r->src[0]->arg.int_tuple.vals[2], 2);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, r), 2);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, r)[0], 2);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, r)[1], 4);
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
  ASSERT_NOT_NULL(only_singleton);
  ASSERT_EQ(only_singleton->op, POLY_OP_RESHAPE);
  ASSERT_PTR_EQ(only_singleton->src[0], b);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, only_singleton), 2);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, only_singleton)[0], 4);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, only_singleton)[1], 3);

  PolyUOp *mixed = poly_reduce_axis(ctx, POLY_OP_ADD, b, (int64_t[]){2, 0}, 2);
  ASSERT_NOT_NULL(mixed);
  ASSERT_EQ(mixed->op, POLY_OP_RESHAPE);
  ASSERT_EQ(mixed->src[0]->op, POLY_OP_REDUCE);
  ASSERT_EQ(mixed->src[0]->arg.kind, POLY_ARG_REDUCE);
  ASSERT_EQ(mixed->src[0]->arg.reduce.op, POLY_OP_ADD);
  ASSERT_INT_EQ(mixed->src[0]->arg.reduce.num_axes, 1);
  ASSERT_EQ(mixed->src[0]->src[0]->op, POLY_OP_PERMUTE);
  ASSERT_PTR_EQ(mixed->src[0]->src[0]->src[0], b);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, mixed), 1);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, mixed)[0], 4);

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
  /* Current STORE follows its target shape; value broadcasting is lowered
   * from the STORE shape during codegen. */
  ASSERT_INT_EQ(poly_uop_ndim(ctx, st), 1);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, st)[0], 4);
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
  PolyUOp *idx = poly_reshape(ctx, poly_test_buffer(ctx, POLY_INT32, 3), (int64_t[]){3}, 1);
  PolyUOp *g = poly_gather(ctx, table, idx);
  ASSERT_INT_EQ(check_shape_parity(ctx, g), 0);
  PolyUOp *float_idx = poly_reshape(ctx, poly_test_buffer(ctx, POLY_FLOAT32, 3), (int64_t[]){3}, 1);
  ASSERT_TRUE(poly_gather(ctx, table, float_idx) == NULL);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(shape_uop, gather_integer_promotion_scalar_and_rank_match_tinygrad) {
  /* Pinned nn/__init__.py:371-392 and mixin/__init__.py:439-449 accept scalar
   * integer indices and promote the index/class-range pair before CMPNE. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *table = poly_reshape(ctx, poly_test_buffer(ctx, POLY_FLOAT32, 12), (int64_t[]){4, 3}, 2);
  ASSERT_NOT_NULL(table);

  struct {
    PolyDType input;
    PolyDType expected_cmp;
  } cases[] = {
      {POLY_INT8, POLY_INT32},  {POLY_UINT8, POLY_INT32},      {POLY_INT32, POLY_INT32},
      {POLY_INT64, POLY_INT64}, {POLY_UINT64, POLY_WEAKFLOAT},
  };
  for (int ci = 0; ci < (int)(sizeof(cases) / sizeof(cases[0])); ci++) {
    PolyUOp *idx = poly_reshape(ctx, poly_test_buffer(ctx, cases[ci].input, 2), (int64_t[]){2}, 1);
    PolyUOp *g = poly_gather(ctx, table, idx);
    ASSERT_NOT_NULL(g);
    PolyUOp *value_cmp = NULL;
    int n_topo = 0;
    PolyUOp **topo = poly_toposort(ctx, g, &n_topo);
    for (int i = 0; i < n_topo; i++) {
      PolyUOp *u = topo[i];
      if (u && u->op == POLY_OP_CMPNE && u->n_src == 2 && !poly_dtype_is_bool(u->src[0]->dtype)) {
        value_cmp = u;
        break;
      }
    }
    ASSERT_NOT_NULL(value_cmp);
    ASSERT_TRUE(poly_dtype_eq(value_cmp->src[0]->dtype, cases[ci].expected_cmp));
    ASSERT_TRUE(poly_dtype_eq(value_cmp->src[1]->dtype, cases[ci].expected_cmp));
  }

  PolyUOp *weak_values[] = {poly_const_int(ctx, 0), poly_const_int(ctx, 2)};
  PolyUOp *weak_idx = poly_stack(ctx, weak_values, 2, 0);
  PolyUOp *weak_gather = poly_gather(ctx, table, weak_idx);
  ASSERT_NOT_NULL(weak_gather);
  PolyUOp *weak_cmp = NULL;
  int n_weak_topo = 0;
  PolyUOp **weak_topo = poly_toposort(ctx, weak_gather, &n_weak_topo);
  for (int i = 0; i < n_weak_topo; i++) {
    PolyUOp *u = weak_topo[i];
    if (u && u->op == POLY_OP_CMPNE && u->n_src == 2 && !poly_dtype_is_bool(u->src[0]->dtype)) {
      weak_cmp = u;
      break;
    }
  }
  ASSERT_NOT_NULL(weak_cmp);
  ASSERT_TRUE(poly_dtype_eq(weak_cmp->src[0]->dtype, POLY_INT32));
  ASSERT_TRUE(poly_dtype_eq(weak_cmp->src[1]->dtype, POLY_INT32));

  PolyUOp *scalar = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(2));
  PolyUOp *scalar_gather = poly_gather(ctx, table, scalar);
  ASSERT_NOT_NULL(scalar_gather);
  PolyShape scalar_shape = poly_uop_max_shape(ctx, scalar_gather);
  ASSERT_INT_EQ(scalar_shape.ndim, 1);
  ASSERT_INT_EQ(scalar_shape.dims[0], 3);
  free(scalar_shape.dims);

  int64_t ones[POLY_MAX_DIMS];
  for (int i = 0; i < POLY_MAX_DIMS; i++)
    ones[i] = 1;
  PolyUOp *rank14 =
      poly_reshape(ctx, poly_test_buffer(ctx, POLY_INT32, 1), ones, POLY_MAX_DIMS - 2);
  PolyUOp *rank15 =
      poly_reshape(ctx, poly_test_buffer(ctx, POLY_INT32, 1), ones, POLY_MAX_DIMS - 1);
  PolyUOp *rank16 = poly_reshape(ctx, poly_test_buffer(ctx, POLY_INT32, 1), ones, POLY_MAX_DIMS);
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
  ASSERT_INT_EQ(sk->op, POLY_OP_RESHAPE);
  ASSERT_INT_EQ(sk->src[1]->op, POLY_OP_STACK);
  ASSERT_TRUE(poly_dtype_eq(sk->src[1]->src[1]->dtype, POLY_WEAKINT));
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
  ASSERT_PTR_EQ(poly_tensor_uop_logical(pure_contiguous), poly_tensor_uop_logical(pure));
  ASSERT_PTR_EQ(poly_tensor_uop_physical(pure_contiguous), pure_physical);
  ASSERT_INT_EQ(poly_tensor_uop_physical(pure_contiguous)->op, POLY_OP_ADD);

  int64_t shape[2] = {2, 3};
  int64_t order[2] = {1, 0};
  PolyTensor *storage = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 2, POLY_DEVICE_CPU);
  PolyTensor *permuted = poly_tensor_permute(ctx, storage, order, 2);
  PolyTensor *materialized = poly_tensor_contiguous(ctx, permuted);
  ASSERT_NOT_NULL(materialized);
  ASSERT_PTR_EQ(poly_tensor_uop_logical(materialized), poly_tensor_uop_logical(permuted));
  ASSERT_INT_EQ(poly_tensor_uop_physical(materialized)->op, POLY_OP_CONTIGUOUS);
  ASSERT_PTR_EQ(poly_tensor_uop_physical(materialized)->src[0], poly_tensor_uop_physical(permuted));

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(tensor, contiguous_backward_constructs_exact_barrier_and_gradient) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  int64_t shape[] = {3};
  PolyTensor *source = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_CPU);
  PolyTensor *result = poly_tensor_contiguous_backward(ctx, source);
  ASSERT_NOT_NULL(source);
  ASSERT_NOT_NULL(result);
  ASSERT_INT_EQ(poly_tensor_uop_logical(result)->op, POLY_OP_CONTIGUOUS_BACKWARD);
  ASSERT_PTR_EQ(poly_tensor_uop_logical(result)->src[0], poly_tensor_uop_logical(source));
  ASSERT_INT_EQ(poly_tensor_uop_physical(result)->op, POLY_OP_CONTIGUOUS_BACKWARD);
  ASSERT_PTR_EQ(poly_tensor_uop_physical(result)->src[0], poly_tensor_uop_physical(source));

  PolyUOp *grad =
      poly_grad(ctx, poly_tensor_uop_physical(result), poly_tensor_uop_physical(source));
  ASSERT_NOT_NULL(grad);
  int n_topo = 0, contiguous_count = 0;
  PolyUOp **topo = poly_toposort_alloc(ctx, grad, &n_topo);
  ASSERT_NOT_NULL(topo);
  for (int i = 0; i < n_topo; i++)
    contiguous_count += topo[i]->op == POLY_OP_CONTIGUOUS;
  ASSERT_INT_EQ(contiguous_count, 1);
  free(topo);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(tensor, binary_alu_uses_implicit_shape_broadcast_like_tinygrad) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  int64_t value_shape[] = {4, 3};
  int64_t bias_shape[] = {3};
  PolyTensor *value = poly_tensor_empty(ctx, POLY_FLOAT32, value_shape, 2, POLY_DEVICE_CPU);
  PolyTensor *bias = poly_tensor_empty(ctx, POLY_FLOAT32, bias_shape, 1, POLY_DEVICE_CPU);
  PolyTensor *sum = poly_tensor_alu2(ctx, POLY_OP_ADD, value, bias);
  ASSERT_NOT_NULL(value);
  ASSERT_NOT_NULL(bias);
  ASSERT_NOT_NULL(sum);

  /* Current UOp._shape owns ALU broadcasting. ElementwiseMixin._broadcasted
   * promotes dtype only, so the rank-one bias remains the ordered ADD source
   * without RESHAPE/EXPAND (uop/ops.py:456-461,
   * mixin/elementwise.py:19-29). */
  PolyUOp *physical = poly_tensor_uop_physical(sum);
  ASSERT_NOT_NULL(physical);
  ASSERT_INT_EQ(physical->op, POLY_OP_ADD);
  ASSERT_INT_EQ(physical->n_src, 2);
  ASSERT_PTR_EQ(physical->src[1], poly_tensor_uop_physical(bias));
  PolyShape physical_shape = poly_uop_max_shape(ctx, physical);
  ASSERT_INT_EQ(physical_shape.ndim, 2);
  ASSERT_INT_EQ(physical_shape.dims[0], 4);
  ASSERT_INT_EQ(physical_shape.dims[1], 3);
  free(physical_shape.dims);

  float value_data[12];
  for (int i = 0; i < 12; i++)
    value_data[i] = (float)i;
  float bias_data[] = {10.0f, 20.0f, 30.0f};
  PolyUOp *value_buffer = (PolyUOp *)poly_uop_get_buffer_identity(poly_tensor_uop_physical(value));
  PolyUOp *bias_buffer = (PolyUOp *)poly_uop_get_buffer_identity(poly_tensor_uop_physical(bias));
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
      ASSERT_FLOAT_NEAR(result[row * 3 + col], value_data[row * 3 + col] + bias_data[col], 4, 1e-6);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(tensor, realization_replaces_value_with_current_buffer_topology) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  int64_t shape[] = {2};
  PolyTensor *a = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_CPU);
  PolyTensor *b = poly_tensor_empty(ctx, POLY_FLOAT32, shape, 1, POLY_DEVICE_CPU);
  PolyTensor *sum = poly_tensor_alu2(ctx, POLY_OP_ADD, a, b);
  ASSERT_NOT_NULL(a);
  ASSERT_NOT_NULL(b);
  ASSERT_NOT_NULL(sum);

  float av[] = {1.0f, 2.0f}, bv[] = {3.0f, 4.0f};
  ASSERT_INT_EQ(
      poly_buffer_write(
          ctx, (PolyUOp *)poly_uop_get_buffer_identity(a->uop_physical), av, sizeof(av)
      ),
      0
  );
  ASSERT_INT_EQ(
      poly_buffer_write(
          ctx, (PolyUOp *)poly_uop_get_buffer_identity(b->uop_physical), bv, sizeof(bv)
      ),
      0
  );

  PolyTensor *realized = NULL;
  ASSERT_INT_EQ(poly_realize_tensors(ctx, &sum, 1, &realized), 0);
  ASSERT_NOT_NULL(realized);
  PolyUOp *buffer = poly_tensor_uop_physical(realized);

  /* Current Tensor.realize replaces the value with UOp.new_buffer's exact
   * one-source storage form (tensor.py:190-218; uop/ops.py:811-817). */
  ASSERT_NOT_NULL(buffer);
  ASSERT_INT_EQ(buffer->op, POLY_OP_BUFFER);
  ASSERT_INT_EQ(buffer->n_src, 1);
  ASSERT_INT_EQ(buffer->src[0]->op, POLY_OP_CONST);
  ASSERT_TRUE(poly_dtype_eq(buffer->src[0]->dtype, POLY_WEAKINT));
  ASSERT_INT_EQ(buffer->src[0]->arg.kind, POLY_ARG_INT);
  ASSERT_INT_EQ(buffer->src[0]->arg.i, 2);
  ASSERT_INT_EQ(buffer->arg.kind, POLY_ARG_PARAM);
  ASSERT_NOT_NULL(buffer->arg.param);
  ASSERT_TRUE(poly_dtype_eq(buffer->arg.param->dtype, POLY_FLOAT32));
  ASSERT_STR_EQ(buffer->arg.param->device, "CPU");

  float got[2] = {0};
  ASSERT_INT_EQ(poly_buffer_read(ctx, buffer, got, sizeof(got)), 0);
  ASSERT_FLOAT_NEAR(got[0], 4.0f, 4, 1e-6);
  ASSERT_FLOAT_NEAR(got[1], 6.0f, 4, 1e-6);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(tensor, raw_true_division_uses_mul_reciprocal_like_tinygrad) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *x = make_buf(ctx, (int64_t[]){5}, 1);
  PolyUOp *two = poly_const_typed(ctx, POLY_WEAKFLOAT, 2.0);
  PolyUOp *quotient = poly_div(ctx, x, two);
  ASSERT_NOT_NULL(quotient);

  /* Current ElementwiseMixin.div is MUL(a, RECIPROCAL(b)) at tensor/UOp
   * stage; FDIV is backend-decomposition vocabulary only
   * (mixin/elementwise.py:225-252, codegen/decomp/op.py:133-136). */
  ASSERT_INT_EQ(quotient->op, POLY_OP_MUL);
  ASSERT_PTR_EQ(quotient->src[0], x);
  ASSERT_INT_EQ(quotient->src[1]->op, POLY_OP_RECIPROCAL);
  ASSERT_PTR_EQ(quotient->src[1]->src[0], two);
  ASSERT_INT_EQ(count_op_in_root(ctx, quotient, POLY_OP_FDIV), 0);

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
  PolyUOp *store = poly_test_store_to_buffer(ctx, out_buf, result);
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
  PolyUOp *out_val = poly_cumalu(ctx, in, 0, POLY_OP_ADD);
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
  PolyUOp *out_val = poly_cumalu(ctx, in, 0, POLY_OP_ADD);
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

TEST(pe, cumalu_add_1d_explicit_prefix_shift) {
  PolyCtx *ctx = poly_ctx_new();

  /* tinygrad: Tensor([1..5]).pad(((1, -1),)).cumsum(0)
   *   -> [0, 1, 3, 6, 10] */
  PolyUOp *in = poly_buffer_f32(ctx, 5);
  PolyUOp *out_val =
      poly_cumalu(ctx, poly_pad_value(ctx, in, (int64_t[1][2]){{1, -1}}, 1, 0), 0, POLY_OP_ADD);
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
  PolyUOp *out_val = poly_cumalu(ctx, in, 1, POLY_OP_ADD);
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
  PolyUOp *out_val = poly_cumalu(ctx, in, 0, POLY_OP_ADD);
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

TEST(pe, cat_equal_width_uses_stack_flatten) {
  PolyCtx *ctx = poly_ctx_new();

  /* Current tinygrad mixin/op.py:750 takes the equal-width fast path:
   * stack(dim=1) followed by flatten(1, 2). */
  int64_t shape[2] = {2, 1};
  PolyUOp *a = make_buf(ctx, shape, 2);
  PolyUOp *b = make_buf(ctx, shape, 2);
  PolyUOp *parts[2] = {a, b};
  PolyUOp *out_val = poly_cat(ctx, parts, 2, 1);
  ASSERT_NOT_NULL(out_val);
  ASSERT_INT_EQ(out_val->op, POLY_OP_RESHAPE);
  ASSERT_INT_EQ(out_val->n_src, 2);
  ASSERT_INT_EQ(out_val->src[0]->op, POLY_OP_PERMUTE);
  ASSERT_INT_EQ(out_val->src[0]->n_src, 1);
  ASSERT_INT_EQ(out_val->src[0]->src[0]->op, POLY_OP_STACK);
  ASSERT_INT_EQ(out_val->src[0]->src[0]->n_src, 2);
  ASSERT_TRUE(out_val->src[0]->src[0]->src[0] == a);
  ASSERT_TRUE(out_val->src[0]->src[0]->src[1] == b);

  PolyShape out_shape = poly_uop_max_shape_cached(ctx, out_val);
  ASSERT_INT_EQ(out_shape.ndim, 2);
  ASSERT_INT_EQ(out_shape.dims[0], 2);
  ASSERT_INT_EQ(out_shape.dims[1], 2);

  PolyUOp *out_buf = poly_buffer_f32(ctx, 4);
  float a_d[2] = {1, 2}, b_d[2] = {7, 8}, out[4] = {0};
  PolyUOp *leaves[] = {base_buf(a), base_buf(b)};
  float *ld[] = {a_d, b_d};
  ASSERT_INT_EQ(realize_uop(ctx, out_val, out_buf, out, leaves, ld, 2), 0);
  float expected[4] = {1, 7, 2, 8};
  for (int i = 0; i < 4; i++)
    ASSERT_FLOAT_EQ(out[i], expected[i], 1e-5);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, cat_one_source_uses_stack_flatten) {
  PolyCtx *ctx = poly_ctx_new();

  /* Current OpMixin.cat still takes stack(...).flatten(...) when no extra
   * source is supplied (mixin/op.py:734-753).  The one-source STACK is an
   * occurrence in random_bits' one-chunk path, not an identity operation. */
  int64_t shape[1] = {4};
  PolyUOp *a = make_buf(ctx, shape, 1);
  PolyUOp *parts[1] = {a};
  PolyUOp *out_val = poly_cat(ctx, parts, 1, 0);
  ASSERT_NOT_NULL(out_val);
  ASSERT_INT_EQ(out_val->op, POLY_OP_RESHAPE);
  ASSERT_INT_EQ(out_val->n_src, 2);
  ASSERT_INT_EQ(out_val->src[0]->op, POLY_OP_STACK);
  ASSERT_INT_EQ(out_val->src[0]->n_src, 1);
  ASSERT_PTR_EQ(out_val->src[0]->src[0], a);

  PolyShape out_shape = poly_uop_max_shape_cached(ctx, out_val);
  ASSERT_INT_EQ(out_shape.ndim, 1);
  ASSERT_INT_EQ(out_shape.dims[0], 4);

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
  PolyUOp *out_val = poly_cumalu(ctx, in, 0, POLY_OP_MAX);
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
  PolyUOp *out_val = poly_cumalu(ctx, in, 0, POLY_OP_MAX);
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
  PolyUOp *out_val = poly_cumalu(ctx, in, 0, POLY_OP_MAX);
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

TEST(pe, cumalu_max_explicit_prefix_shift) {
  PolyCtx *ctx = poly_ctx_new();

  /* tinygrad: Tensor([1,3,2]).pad(((1,-1),), value=-inf).cummax(0)[0] -> [-inf, 1, 3] */
  PolyUOp *in = poly_buffer_f32(ctx, 3);
  PolyUOp *out_val = poly_cumalu(
      ctx, poly_pad_value(ctx, in, (int64_t[1][2]){{1, -1}}, 1, -INFINITY), 0, POLY_OP_MAX
  );
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
  PolyUOp *out_val = poly_cumalu(ctx, in, 0, POLY_OP_MUL);
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
  PolyUOp *out_val = poly_cumalu(ctx, in, 0, POLY_OP_MUL);
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

TEST(pe, cumalu_mul_explicit_prefix_shift) {
  PolyCtx *ctx = poly_ctx_new();

  /* tinygrad: Tensor([2,3,4]).pad(((1,-1),), value=1).cumprod(0) -> [1, 2, 6] */
  PolyUOp *in = poly_buffer_f32(ctx, 3);
  PolyUOp *out_val =
      poly_cumalu(ctx, poly_pad_value(ctx, in, (int64_t[1][2]){{1, -1}}, 1, 1), 0, POLY_OP_MUL);
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
    /* Current tinygrad full((), value).reshape((1,)).reshape(()).expand((0,)). */
    ASSERT_EQ(root->src[0]->src[0]->op, POLY_OP_RESHAPE);
    ASSERT_EQ(root->src[0]->src[0]->src[0]->op, POLY_OP_CONST);
    ASSERT_EQ(root->src[0]->src[0]->src[1]->op, POLY_OP_CONST);
    ASSERT_TRUE(poly_dtype_is_weak(root->src[0]->src[0]->src[1]->dtype));
    ASSERT_EQ(root->src[0]->src[1]->op, POLY_OP_STACK);
    ASSERT_EQ(root->src[1]->op, POLY_OP_CONST);
    ASSERT_TRUE(poly_dtype_is_weak(root->src[1]->dtype));
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
  ASSERT_INT_EQ(zero->src[0]->op, POLY_OP_CONST);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(pe, zero_dimension_broadcast_matches_tinygrad) {
  /* Current _broadcast_shape chooses zero when either aligned dimension is
   * zero. CreationMixin.full first reshapes the scalar to rank-many ones;
   * _broadcast_to then squeezes those expandable axes before raw EXPAND, so
   * the untouched full value retains two nested RESHAPEs
   * (uop/ops.py:63-71, mixin/creation.py:61-81,
   * mixin/movement.py:129-147). This is the core topology reached by
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
  ASSERT_INT_EQ(mask_value->src[0]->src[0]->op, POLY_OP_RESHAPE);
  ASSERT_INT_EQ(mask_value->src[0]->src[0]->src[0]->op, POLY_OP_CONST);
  ASSERT_INT_EQ(out->op, POLY_OP_WHERE);
  ASSERT_INT_EQ(poly_uop_ndim(ctx, out), 3);
  const int64_t *shape = poly_uop_max_shape_dims(ctx, out);
  ASSERT_NOT_NULL(shape);
  ASSERT_INT_EQ(shape[0], 5);
  ASSERT_INT_EQ(shape[1], 0);
  ASSERT_INT_EQ(shape[2], 3);
  /* Current ElementwiseMixin leaves broadcastable operands unexpanded and
   * UOp._shape infers WHERE's common shape (mixin/elementwise.py:19-29,
   * uop/ops.py:456-461). */
  ASSERT_PTR_EQ(out->src[0], mask);
  ASSERT_EQ(out->src[0]->op, POLY_OP_CMPNE);
  ASSERT_PTR_EQ(out->src[1], value);
  ASSERT_EQ(out->src[1]->op, POLY_OP_EXPAND);
  ASSERT_EQ(out->src[2]->op, POLY_OP_CONST);

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

  /* Current mixin/op.py:_tri keeps arange(c) rank one and unsqueezes only
   * arange(r). poly_le reverses the operands for CMPLT, so the column source
   * is direct while the row source is ADD(RESHAPE(...), 1). */
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, out_val, &n_topo);
  PolyUOp *coord_cmp = NULL;
  for (int i = 0; i < n_topo; i++)
    if (topo[i]->op == POLY_OP_CMPLT) coord_cmp = topo[i];
  ASSERT_NOT_NULL(coord_cmp);
  ASSERT_INT_EQ(coord_cmp->n_src, 2);
  ASSERT_TRUE(coord_cmp->src[0]->op != POLY_OP_RESHAPE);
  ASSERT_INT_EQ(coord_cmp->src[1]->op, POLY_OP_ADD);
  ASSERT_INT_EQ(coord_cmp->src[1]->src[0]->op, POLY_OP_RESHAPE);
  ASSERT_TRUE(poly_dtype_eq(coord_cmp->src[1]->src[1]->dtype, POLY_WEAKINT));
  PolyUOp *where = NULL;
  for (int i = 0; i < n_topo; i++)
    if (topo[i]->op == POLY_OP_WHERE) where = topo[i];
  ASSERT_NOT_NULL(where);
  ASSERT_INT_EQ(where->src[1]->op, POLY_OP_EXPAND);
  ASSERT_INT_EQ(where->src[1]->src[0]->op, POLY_OP_CONST);
  ASSERT_INT_EQ(where->src[1]->src[0]->arg.kind, POLY_ARG_FLOAT);
  ASSERT_FLOAT_EQ(where->src[1]->src[0]->arg.f, 0.0, 0.0);

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
      1, 2, 3, 0, 5, 6, 0, 0, 9, 10, 11, 12, 0, 14, 15, 0, 0, 18,
  };
  float expect_lower[18] = {
      1, 2, 0, 4, 5, 6, 7, 8, 9, 10, 11, 0, 13, 14, 15, 16, 17, 18,
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
    PolyUOp **lin = poly_test_full_rewrite_and_linearize(ctx, kernel, &n_lin);
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
