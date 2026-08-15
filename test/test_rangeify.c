/*
 * test_rangeify.c — Tests for the rangeify scheduling pipeline
 */

#define _POSIX_C_SOURCE 200809L
#include "test_harness.h"
#include "../src/schedule/rangeify.h"
#include "../src/schedule/rangeify.h"
#include "../src/engine/schedule.h"
#include "../src/engine/realize.h"
#include "../src/ctx.h"
#include "../src/schedule/indexing.h"
#include "../src/codegen.h"
#include "../src/frontend.h"
#include "../src/tensor.h"
#include "../src/utils.h"
#include <pthread.h>

/* Cleanup helper */

static void free_consumer_list_cb(const void *key, void *value, void *ud) {
  (void)key;
  (void)ud;
  PolyConsumerList *cl = value;
  free(cl->items);
  free(cl);
}

static void destroy_consumer_map(PolyMap *cmap) {
  poly_map_foreach(cmap, free_consumer_list_cb, NULL);
  poly_map_destroy(cmap);
}

static PolyRealizeInfo *get_realize_info(PolyIndexingCtx *ictx, PolyUOp *u) {
  return poly_map_get(ictx->realize_map, poly_ptr_hash(u), u, poly_ptr_eq);
}

static int count_ops(PolyCtx *ctx, PolyUOp *root, PolyOps op);
static PolyUOp *run_apply_rangeify(PolyIndexingCtx *ictx, PolyUOp *sink);

typedef struct {
  const char *key;
  char *value;
  bool had_value;
} RangeifyEnvSave;

static RangeifyEnvSave rangeify_save_env(const char *key) {
  const char *value = getenv(key);
  return (RangeifyEnvSave){
      .key = key,
      .value = value ? strdup(value) : NULL,
      .had_value = value != NULL,
  };
}

static void rangeify_restore_env(RangeifyEnvSave *saved) {
  if (saved->had_value)
    setenv(saved->key, saved->value ? saved->value : "", 1);
  else
    unsetenv(saved->key);
  free(saved->value);
  saved->value = NULL;
}

/* Consumer map tests */

TEST(rangeify, consumer_map_chain) {
  /* a → ADD(a,b) → STORE → SINK: verify consumer counts */
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *b = poly_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *c = poly_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, a, b, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, c, add, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyMap *cmap = poly_consumer_map_build(ctx, sink);
  ASSERT_NOT_NULL(cmap);

  /* a is consumed by ADD */
  PolyConsumerList *cl_a = poly_consumer_map_get(cmap, a);
  ASSERT_NOT_NULL(cl_a);
  ASSERT_INT_EQ(cl_a->count, 1);
  ASSERT_PTR_EQ(cl_a->items[0], add);

  /* b is consumed by ADD */
  PolyConsumerList *cl_b = poly_consumer_map_get(cmap, b);
  ASSERT_NOT_NULL(cl_b);
  ASSERT_INT_EQ(cl_b->count, 1);

  /* add is consumed by STORE */
  PolyConsumerList *cl_add = poly_consumer_map_get(cmap, add);
  ASSERT_NOT_NULL(cl_add);
  ASSERT_INT_EQ(cl_add->count, 1);
  ASSERT_PTR_EQ(cl_add->items[0], store);

  /* store is consumed by SINK */
  PolyConsumerList *cl_store = poly_consumer_map_get(cmap, store);
  ASSERT_NOT_NULL(cl_store);
  ASSERT_INT_EQ(cl_store->count, 1);
  ASSERT_PTR_EQ(cl_store->items[0], sink);

  /* sink has 0 consumers */
  PolyConsumerList *cl_sink = poly_consumer_map_get(cmap, sink);
  ASSERT_NOT_NULL(cl_sink);
  ASSERT_INT_EQ(cl_sink->count, 0);

  destroy_consumer_map(cmap);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, consumer_map_diamond) {
  /* a → {NEG(a), SQRT(a)} → ADD → STORE → SINK
   * a should have 2 consumers */
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *neg = poly_uop1(ctx, POLY_OP_NEG, POLY_FLOAT32, a, poly_arg_none());
  PolyUOp *sqrt_op = poly_uop1(ctx, POLY_OP_SQRT, POLY_FLOAT32, a, poly_arg_none());
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, neg, sqrt_op, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out, add, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyMap *cmap = poly_consumer_map_build(ctx, sink);

  /* a has 2 consumers: neg and sqrt */
  PolyConsumerList *cl_a = poly_consumer_map_get(cmap, a);
  ASSERT_NOT_NULL(cl_a);
  ASSERT_INT_EQ(cl_a->count, 2);

  /* neg and sqrt each have 1 consumer (add) */
  PolyConsumerList *cl_neg = poly_consumer_map_get(cmap, neg);
  ASSERT_INT_EQ(cl_neg->count, 1);
  PolyConsumerList *cl_sqrt = poly_consumer_map_get(cmap, sqrt_op);
  ASSERT_INT_EQ(cl_sqrt->count, 1);

  destroy_consumer_map(cmap);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, consumer_map_multi_output) {
  /* a used in two STORE paths: SINK(STORE(out1, a+b), STORE(out2, a*c))
   * a should have 2 consumers (ADD and MUL) */
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *b = poly_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *c = poly_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *out1 = poly_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *out2 = poly_buffer(ctx, POLY_FLOAT32, 10);

  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, a, b, poly_arg_none());
  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, a, c, poly_arg_none());
  PolyUOp *s1 = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out1, add, poly_arg_none());
  PolyUOp *s2 = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out2, mul, poly_arg_none());

  PolyUOp *stores[] = {s1, s2};
  PolyUOp *sink = poly_uop(ctx, POLY_OP_SINK, POLY_VOID, stores, 2, poly_arg_none());

  PolyMap *cmap = poly_consumer_map_build(ctx, sink);

  /* a has 2 consumers: add and mul */
  PolyConsumerList *cl_a = poly_consumer_map_get(cmap, a);
  ASSERT_NOT_NULL(cl_a);
  ASSERT_INT_EQ(cl_a->count, 2);

  /* sink has 0 consumers */
  PolyConsumerList *cl_sink = poly_consumer_map_get(cmap, sink);
  ASSERT_INT_EQ(cl_sink->count, 0);

  destroy_consumer_map(cmap);
  poly_ctx_destroy(ctx);
  PASS();
}

/* Realize map tests */

TEST(rangeify, realize_map_sink_sources) {
  /* SINK(STORE(out, a+b)): the STORE should be realized */
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *b = poly_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, a, b, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out, add, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyIndexingCtx *ictx = poly_indexing_ctx_new(ctx);
  poly_realize_map_build(ictx, sink);

  /* STORE is a SINK source → realized */
  ASSERT_TRUE(poly_is_realized(ictx, store));

  /* Interior ops (a, b, add) are NOT realized */
  ASSERT_FALSE(poly_is_realized(ictx, a));
  ASSERT_FALSE(poly_is_realized(ictx, b));
  ASSERT_FALSE(poly_is_realized(ictx, add));

  poly_indexing_ctx_destroy(ictx);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, realize_map_sink_after_is_already_contiguous) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *b = poly_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, a, b, poly_arg_none());
  PolyUOp *store = poly_store_val(ctx, out, add);
  PolyUOp *after_src[2] = {out, store};
  PolyUOp *after = poly_uop(ctx, POLY_OP_AFTER, out->dtype, after_src, 2, poly_arg_none());
  PolyUOp *sink = poly_sink_n(ctx, &after, 1);
  ASSERT_NOT_NULL(store);
  ASSERT_NOT_NULL(after);
  ASSERT_NOT_NULL(sink);

  PolyIndexingCtx *ictx = poly_indexing_ctx_new(ctx);
  poly_realize_map_build(ictx, sink);

  /* Pinned tinygrad's ALWAYS_CONTIGUOUS includes AFTER: its STORE is the
   * producer boundary, while re-realizing the AFTER creates a dead copy. */
  ASSERT_TRUE(poly_is_realized(ictx, store));
  ASSERT_FALSE(poly_is_realized(ictx, after));
  ASSERT_FALSE(poly_is_realized(ictx, add));

  poly_indexing_ctx_destroy(ictx);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, realize_map_elementwise_fused) {
  /* c = (a+b)*d: interior ALU ops should NOT be realized */
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *b = poly_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *d = poly_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, a, b, poly_arg_none());
  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, add, d, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out, mul, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyIndexingCtx *ictx = poly_indexing_ctx_new(ctx);
  poly_realize_map_build(ictx, sink);

  /* Only STORE is realized */
  ASSERT_TRUE(poly_is_realized(ictx, store));

  /* All ALU ops are fused (not realized) */
  ASSERT_FALSE(poly_is_realized(ictx, add));
  ASSERT_FALSE(poly_is_realized(ictx, mul));
  ASSERT_FALSE(poly_is_realized(ictx, a));
  ASSERT_FALSE(poly_is_realized(ictx, b));
  ASSERT_FALSE(poly_is_realized(ictx, d));

  poly_indexing_ctx_destroy(ictx);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, realize_map_reduce_not_auto_realized) {
  /* sum(a) → tensor REDUCE(a): REDUCE is NOT automatically realized.
   * Only SINK sources (the STORE wrapping it) are realized. */
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, 1);

  int64_t axes[] = {0};
  PolyUOp *reduce = poly_reduce_axis(ctx, POLY_OP_ADD, a, axes, 1);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out, reduce, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyIndexingCtx *ictx = poly_indexing_ctx_new(ctx);
  poly_realize_map_build(ictx, sink);

  /* STORE is realized (SINK source) */
  ASSERT_TRUE(poly_is_realized(ictx, store));

  /* Tensor REDUCE is NOT realized by default. */
  ASSERT_FALSE(poly_is_realized(ictx, reduce));

  poly_indexing_ctx_destroy(ictx);
  poly_ctx_destroy(ctx);
  PASS();
}

/* Range propagation tests */

TEST(rangeify, range_prop_after_carries_dependency_without_own_range) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *inner_buffer = poly_buffer(ctx, POLY_FLOAT32, 4);
  PolyUOp *outer_buffer = poly_buffer(ctx, POLY_FLOAT32, 4);
  int64_t shape[1] = {4};
  PolyUOp *ones = poly_expand(ctx, poly_const_float(ctx, 1.0), shape, 1);
  PolyUOp *inner_store = poly_store_val(ctx, inner_buffer, ones);
  PolyUOp *inner_after_src[2] = {inner_buffer, inner_store};
  PolyUOp *inner_after =
      poly_uop(ctx, POLY_OP_AFTER, inner_buffer->dtype, inner_after_src, 2, poly_arg_none());
  PolyUOp *outer_store = poly_store_val(ctx, outer_buffer, inner_after);
  PolyUOp *sink = poly_sink1(ctx, outer_store);
  ASSERT_NOT_NULL(inner_buffer);
  ASSERT_NOT_NULL(outer_buffer);
  ASSERT_NOT_NULL(ones);
  ASSERT_NOT_NULL(inner_store);
  ASSERT_NOT_NULL(inner_after);
  ASSERT_NOT_NULL(outer_store);
  ASSERT_NOT_NULL(sink);

  PolyIndexingCtx *ictx = poly_indexing_ctx_new(ctx);
  ictx->add_buffer_indices = true;
  poly_realize_map_build(ictx, sink);
  poly_range_propagate(ictx, sink);

  ASSERT_NOT_NULL(poly_range_map_get(ictx, inner_store));
  ASSERT_NOT_NULL(poly_range_map_get(ictx, outer_store));
  ASSERT_TRUE(poly_range_map_get(ictx, inner_after) == NULL);

  PolyUOp *rangeified = poly_run_rangeify(ictx, sink);
  ASSERT_NOT_NULL(rangeified);
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, rangeified, &n_topo);
  PolyUOp *mapped_after = NULL;
  for (int i = 0; i < n_topo; i++) {
    if (topo[i]->op == POLY_OP_AFTER && topo[i]->n_src >= 2) mapped_after = topo[i];
  }
  ASSERT_NOT_NULL(mapped_after);
  ASSERT_EQ(mapped_after->src[0]->op, POLY_OP_BUFFER);

  poly_indexing_ctx_destroy(ictx);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, range_prop_elementwise) {
  /* a + b → STORE → SINK: all fused, share same ranges */
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *b = poly_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, a, b, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out, add, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyIndexingCtx *ictx = poly_indexing_ctx_new(ctx);
  poly_realize_map_build(ictx, sink);
  poly_range_propagate(ictx, sink);

  /* STORE is realized → gets fresh RANGE */
  PolyRangeEntry *re_store = poly_range_map_get(ictx, store);
  ASSERT_NOT_NULL(re_store);
  ASSERT_INT_EQ(re_store->n_out, 1);
  ASSERT_EQ(re_store->out_rngs[0]->op, POLY_OP_RANGE);

  /* ADD inherits STORE's ranges (fused) — same range UOp pointer */
  PolyRangeEntry *re_add = poly_range_map_get(ictx, add);
  ASSERT_NOT_NULL(re_add);
  ASSERT_INT_EQ(re_add->n_out, 1);
  ASSERT_PTR_EQ(re_add->out_rngs[0], re_store->out_rngs[0]);

  /* a and b also inherit the same ranges */
  PolyRangeEntry *re_a = poly_range_map_get(ictx, a);
  ASSERT_NOT_NULL(re_a);
  ASSERT_PTR_EQ(re_a->out_rngs[0], re_store->out_rngs[0]);

  PolyRangeEntry *re_b = poly_range_map_get(ictx, b);
  ASSERT_NOT_NULL(re_b);
  ASSERT_PTR_EQ(re_b->out_rngs[0], re_store->out_rngs[0]);

  poly_indexing_ctx_destroy(ictx);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, range_prop_chain) {
  /* (a+b)*c → STORE → SINK: all ops share same ranges */
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, 8);
  PolyUOp *b = poly_buffer(ctx, POLY_FLOAT32, 8);
  PolyUOp *c = poly_buffer(ctx, POLY_FLOAT32, 8);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, 8);
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, a, b, poly_arg_none());
  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, add, c, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out, mul, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyIndexingCtx *ictx = poly_indexing_ctx_new(ctx);
  poly_realize_map_build(ictx, sink);
  poly_range_propagate(ictx, sink);

  PolyRangeEntry *re_store = poly_range_map_get(ictx, store);
  ASSERT_NOT_NULL(re_store);

  /* ADD, MUL, a, b, c all share STORE's range (all fused) */
  PolyRangeEntry *re_add = poly_range_map_get(ictx, add);
  PolyRangeEntry *re_mul = poly_range_map_get(ictx, mul);
  PolyRangeEntry *re_a = poly_range_map_get(ictx, a);
  PolyRangeEntry *re_c = poly_range_map_get(ictx, c);

  ASSERT_PTR_EQ(re_mul->out_rngs[0], re_store->out_rngs[0]);
  ASSERT_PTR_EQ(re_add->out_rngs[0], re_store->out_rngs[0]);
  ASSERT_PTR_EQ(re_a->out_rngs[0], re_store->out_rngs[0]);
  ASSERT_PTR_EQ(re_c->out_rngs[0], re_store->out_rngs[0]);

  poly_indexing_ctx_destroy(ictx);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, range_prop_reduce) {
  /* sum(a) where a has shape (10,): reduced axis gets new RANGE */
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, 1);
  int64_t axes[] = {0};
  PolyUOp *reduce = poly_reduce_axis(ctx, POLY_OP_ADD, a, axes, 1);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out, reduce, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyIndexingCtx *ictx = poly_indexing_ctx_new(ctx);
  poly_realize_map_build(ictx, sink);
  poly_range_propagate(ictx, sink);

  /* STORE: shape (1,), realized → range is CONST(0) since dim==1 */
  PolyRangeEntry *re_store = poly_range_map_get(ictx, store);
  ASSERT_NOT_NULL(re_store);
  ASSERT_INT_EQ(re_store->n_out, 1);
  ASSERT_EQ(re_store->out_rngs[0]->op, POLY_OP_CONST);

  /* Tensor REDUCE inherits STORE's scalar range as output,
   * but input gets a new RANGE for the reduced axis (dim=10) */
  PolyRangeEntry *re_reduce = poly_range_map_get(ictx, reduce);
  ASSERT_NOT_NULL(re_reduce);
  ASSERT_INT_EQ(re_reduce->n_in, 1);
  ASSERT_EQ(re_reduce->in_rngs[0]->op, POLY_OP_RANGE);

  /* a: inherits REDUCE's input range (the reduction loop variable) */
  PolyRangeEntry *re_a = poly_range_map_get(ictx, a);
  ASSERT_NOT_NULL(re_a);
  ASSERT_PTR_EQ(re_a->out_rngs[0], re_reduce->in_rngs[0]);

  poly_indexing_ctx_destroy(ictx);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, range_prop_reduce_chain) {
  /* sum(a) + b: reduce gets inner range, b gets outer (STORE) range */
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *b = poly_buffer(ctx, POLY_FLOAT32, 1);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, 1);
  int64_t axes[] = {0};
  PolyUOp *reduce = poly_reduce_axis(ctx, POLY_OP_ADD, a, axes, 1);
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, reduce, b, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out, add, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyIndexingCtx *ictx = poly_indexing_ctx_new(ctx);
  poly_realize_map_build(ictx, sink);
  poly_range_propagate(ictx, sink);

  /* STORE output range is CONST(0) since shape is (1,) */
  PolyRangeEntry *re_store = poly_range_map_get(ictx, store);
  ASSERT_NOT_NULL(re_store);

  /* ADD inherits STORE's range */
  PolyRangeEntry *re_add = poly_range_map_get(ictx, add);
  ASSERT_NOT_NULL(re_add);
  ASSERT_PTR_EQ(re_add->out_rngs[0], re_store->out_rngs[0]);

  /* b inherits ADD's range (outer/output range) */
  PolyRangeEntry *re_b = poly_range_map_get(ictx, b);
  ASSERT_NOT_NULL(re_b);
  ASSERT_PTR_EQ(re_b->out_rngs[0], re_store->out_rngs[0]);

  /* Tensor REDUCE gets an inner RANGE for the reduced axis (different from outer). */
  PolyRangeEntry *re_reduce = poly_range_map_get(ictx, reduce);
  ASSERT_NOT_NULL(re_reduce);
  ASSERT_INT_EQ(re_reduce->n_in, 1);
  ASSERT_EQ(re_reduce->in_rngs[0]->op, POLY_OP_RANGE);
  ASSERT_PTR_NEQ(re_reduce->in_rngs[0], re_store->out_rngs[0]);

  poly_indexing_ctx_destroy(ictx);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, range_prop_multi_consumer_fuse) {
  /* a → {NEG(a), SQRT(a)} → ADD → STORE → SINK
   * Both consumers of a see same ranges → a is fused (not realized) */
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *neg = poly_uop1(ctx, POLY_OP_NEG, POLY_FLOAT32, a, poly_arg_none());
  PolyUOp *sqrt_op = poly_uop1(ctx, POLY_OP_SQRT, POLY_FLOAT32, a, poly_arg_none());
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, neg, sqrt_op, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out, add, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyIndexingCtx *ictx = poly_indexing_ctx_new(ctx);
  poly_realize_map_build(ictx, sink);
  poly_range_propagate(ictx, sink);

  /* a has 2 consumers (neg, sqrt), but both see same ranges from STORE */
  ASSERT_FALSE(poly_is_realized(ictx, a));

  /* a should have range entry (fused, not skipped) */
  PolyRangeEntry *re_a = poly_range_map_get(ictx, a);
  ASSERT_NOT_NULL(re_a);

  /* a's ranges should match STORE's ranges */
  PolyRangeEntry *re_store = poly_range_map_get(ictx, store);
  ASSERT_PTR_EQ(re_a->out_rngs[0], re_store->out_rngs[0]);

  poly_indexing_ctx_destroy(ictx);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, range_prop_multi_consumer_realize) {
  /* a used in two STORE paths with different ranges:
   * SINK(STORE(out1, a+b), STORE(out2, a*c))
   * a's consumers (ADD, MUL) have different ranges → a must be realized */
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *b = poly_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *c = poly_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *out1 = poly_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *out2 = poly_buffer(ctx, POLY_FLOAT32, 10);

  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, a, b, poly_arg_none());
  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, a, c, poly_arg_none());
  PolyUOp *s1 = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out1, add, poly_arg_none());
  PolyUOp *s2 = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out2, mul, poly_arg_none());

  PolyUOp *stores[] = {s1, s2};
  PolyUOp *sink = poly_uop(ctx, POLY_OP_SINK, POLY_VOID, stores, 2, poly_arg_none());

  PolyIndexingCtx *ictx = poly_indexing_ctx_new(ctx);
  poly_realize_map_build(ictx, sink);
  poly_range_propagate(ictx, sink);

  /* s1 and s2 each get fresh ranges (both realized) */
  PolyRangeEntry *re_s1 = poly_range_map_get(ictx, s1);
  PolyRangeEntry *re_s2 = poly_range_map_get(ictx, s2);
  ASSERT_NOT_NULL(re_s1);
  ASSERT_NOT_NULL(re_s2);
  ASSERT_PTR_NEQ(re_s1->out_rngs[0], re_s2->out_rngs[0]);

  /* a has 2 consumers (ADD, MUL) with different ranges → realized */
  ASSERT_TRUE(poly_is_realized(ictx, a));

  /* a gets its own fresh ranges */
  PolyRangeEntry *re_a = poly_range_map_get(ictx, a);
  ASSERT_NOT_NULL(re_a);
  ASSERT_PTR_NEQ(re_a->out_rngs[0], re_s1->out_rngs[0]);
  ASSERT_PTR_NEQ(re_a->out_rngs[0], re_s2->out_rngs[0]);

  poly_indexing_ctx_destroy(ictx);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, range_prop_multi_consumer_overflow_realizes) {
  /* More than 16 consumers exceeds poly_range_propagate's fixed scratch
   * proof set. The first 16 consumers below all inherit the same final store
   * range, while the 17th is FLIP(x), whose input index differs. If the 17th
   * consumer is silently ignored, x incorrectly remains fused. */
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, 4);
  PolyUOp *b = poly_buffer(ctx, POLY_FLOAT32, 4);
  PolyUOp *x = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, a, b, poly_arg_none());

  PolyUOp *acc = NULL;
  for (int i = 0; i < 16; i++) {
    PolyUOp *c = poly_const_float(ctx, (double)(i + 1));
    PolyUOp *term = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, x, c, poly_arg_none());
    acc = acc ? poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, acc, term, poly_arg_none()) : term;
  }
  PolyUOp *flipped = poly_flip(ctx, x, (int64_t[]){0}, 1);
  acc = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, acc, flipped, poly_arg_none());

  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, 4);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out, acc, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyIndexingCtx *ictx = poly_indexing_ctx_new(ctx);
  poly_realize_map_build(ictx, sink);
  poly_range_propagate(ictx, sink);

  ASSERT_TRUE(poly_is_realized(ictx, x));

  PolyRangeEntry *re_x = poly_range_map_get(ictx, x);
  PolyRangeEntry *re_store = poly_range_map_get(ictx, store);
  ASSERT_NOT_NULL(re_x);
  ASSERT_NOT_NULL(re_store);
  ASSERT_INT_EQ(re_x->n_out, 1);
  ASSERT_PTR_NEQ(re_x->out_rngs[0], re_store->out_rngs[0]);

  poly_indexing_ctx_destroy(ictx);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, range_prop_expand_ending_realizes_elementwise_default_pcontig) {
  /* tinygrad indexing.py realizes ended ranges unconditionally when
   * PCONTIG <= 1. The default tinygrad setting is PCONTIG=0, so an
   * elementwise op feeding EXPAND must become a realize point here. */
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a_flat = poly_buffer(ctx, POLY_FLOAT32, 2);
  PolyUOp *b_flat = poly_buffer(ctx, POLY_FLOAT32, 2);
  PolyUOp *c_flat = poly_buffer(ctx, POLY_FLOAT32, 8);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, 8);

  int64_t sh21[] = {2, 1};
  int64_t sh24[] = {2, 4};
  PolyUOp *a = poly_reshape(ctx, a_flat, sh21, 2);
  PolyUOp *b = poly_reshape(ctx, b_flat, sh21, 2);
  PolyUOp *c = poly_reshape(ctx, c_flat, sh24, 2);
  PolyUOp *pre = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, a, b, poly_arg_none());
  PolyUOp *exp = poly_expand(ctx, pre, sh24, 2);
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, exp, c, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out, add, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyIndexingCtx *ictx = poly_indexing_ctx_new(ctx);
  poly_realize_map_build(ictx, sink);
  poly_range_propagate(ictx, sink);

  ASSERT_TRUE(poly_is_realized(ictx, pre));
  PolyRealizeInfo *ri = get_realize_info(ictx, pre);
  ASSERT_NOT_NULL(ri);
  ASSERT_INT_EQ(ri->n_axes, 2);
  ASSERT_INT_EQ(ri->axes[0], 0);
  ASSERT_INT_EQ(ri->axes[1], 1);

  PolyRangeEntry *re_pre = poly_range_map_get(ictx, pre);
  PolyRangeEntry *re_exp = poly_range_map_get(ictx, exp);
  ASSERT_NOT_NULL(re_pre);
  ASSERT_NOT_NULL(re_exp);
  ASSERT_INT_EQ(re_pre->n_out, 2);
  ASSERT_INT_EQ(re_exp->n_in, 2);
  ASSERT_PTR_NEQ(re_pre->out_rngs[0], re_exp->in_rngs[0]);

  poly_indexing_ctx_destroy(ictx);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, range_prop_multi_consumer_same_idx_different_valid_fuses) {
  /* tinygrad's multi-consumer merge compares local idx separately from valid.
   * Two consumers that access the same local idx but carry different valid
   * masks should stay fused at default PCONTIG=0. */
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, 4);
  PolyUOp *b = poly_buffer(ctx, POLY_FLOAT32, 4);
  PolyUOp *x = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, a, b, poly_arg_none());
  PolyUOp *c1 =
      poly_pad(ctx, poly_shrink(ctx, x, (int64_t[][2]){{0, 3}}, 1), (int64_t[][2]){{0, 2}}, 1);
  PolyUOp *c2 = poly_pad(ctx, x, (int64_t[][2]){{0, 1}}, 1);
  PolyUOp *y = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, c1, c2, poly_arg_none());
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, 5);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out, y, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyIndexingCtx *ictx = poly_indexing_ctx_new(ctx);
  poly_realize_map_build(ictx, sink);
  poly_range_propagate(ictx, sink);

  /* Match tinygrad's run_rangeify propagation boundary first: same local idx
   * plus different valid masks must not force a realize point here. */
  ASSERT_FALSE(poly_is_realized(ictx, x));
  PolyRangeEntry *re_x = poly_range_map_get(ictx, x);
  ASSERT_NOT_NULL(re_x);
  ASSERT_INT_EQ(re_x->n_out, 1);
  ASSERT_EQ(re_x->out_rngs[0]->op, POLY_OP_WHERE);

  PolyUOp *rangeified = run_apply_rangeify(ictx, sink);
  ASSERT_NOT_NULL(rangeified);

  ASSERT_FALSE(poly_is_realized(ictx, x));
  ASSERT_INT_EQ(count_ops(ctx, rangeified, POLY_OP_STAGE), 0);
  ASSERT_INT_EQ(count_ops(ctx, rangeified, POLY_OP_WHERE), 2);

  poly_indexing_ctx_destroy(ictx);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, range_prop_multi_consumer_same_idx_different_valid_fuses_2d) {
  /* Same rule as the 1D case above, but with a surviving leading axis.
   * This locks in that polygrad still fuses when only one axis carries
   * different valid masks and the local idx expressions stay identical. */
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a0 = poly_buffer(ctx, POLY_FLOAT32, 8);
  PolyUOp *b0 = poly_buffer(ctx, POLY_FLOAT32, 8);
  PolyUOp *a = poly_reshape(ctx, a0, (int64_t[]){2, 4}, 2);
  PolyUOp *b = poly_reshape(ctx, b0, (int64_t[]){2, 4}, 2);
  PolyUOp *x = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, a, b, poly_arg_none());
  PolyUOp *c1 = poly_pad(
      ctx, poly_shrink(ctx, x, (int64_t[][2]){{0, 2}, {0, 3}}, 2), (int64_t[][2]){{0, 0}, {0, 2}}, 2
  );
  PolyUOp *c2 = poly_pad(ctx, x, (int64_t[][2]){{0, 0}, {0, 1}}, 2);
  PolyUOp *y = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, c1, c2, poly_arg_none());
  PolyUOp *out0 = poly_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *out = poly_reshape(ctx, out0, (int64_t[]){2, 5}, 2);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out, y, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyIndexingCtx *ictx = poly_indexing_ctx_new(ctx);
  poly_realize_map_build(ictx, sink);
  poly_range_propagate(ictx, sink);

  ASSERT_FALSE(poly_is_realized(ictx, x));
  PolyRangeEntry *re_x = poly_range_map_get(ictx, x);
  ASSERT_NOT_NULL(re_x);
  ASSERT_INT_EQ(re_x->n_out, 2);
  /* Pinned schedule/indexing.py:136-141 leaves the unchanged leading PAD
   * dimension alone, while :201-215 merges the changed trailing dimension
   * and symbolically canonicalizes its inherited `r - 0` index. */
  ASSERT_EQ(re_x->out_rngs[0]->op, POLY_OP_RANGE);
  ASSERT_EQ(re_x->out_rngs[1]->op, POLY_OP_WHERE);
  ASSERT_EQ(poly_index_get_idx(ctx, re_x->out_rngs[0])->op, POLY_OP_RANGE);
  ASSERT_EQ(poly_index_get_idx(ctx, re_x->out_rngs[1])->op, POLY_OP_RANGE);
  ASSERT_EQ(poly_index_get_valid(ctx, re_x->out_rngs[0])->op, POLY_OP_CONST);
  ASSERT_EQ(poly_index_get_valid(ctx, re_x->out_rngs[1])->op, POLY_OP_OR);

  PolyUOp *rangeified = poly_run_rangeify(ictx, sink);
  ASSERT_NOT_NULL(rangeified);
  ASSERT_FALSE(poly_is_realized(ictx, x));
  ASSERT_INT_EQ(count_ops(ctx, rangeified, POLY_OP_STAGE), 0);
  ASSERT_INT_EQ(count_ops(ctx, rangeified, POLY_OP_WHERE), 2);

  poly_indexing_ctx_destroy(ictx);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, range_prop_reshape) {
  /* reshape(a, (2,3)) where a has shape (6,):
   * STORE gets 2D ranges, reshape transforms to 1D input range */
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, 6);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, 6);
  int64_t new_shape[] = {2, 3};
  PolyUOp *reshaped = poly_reshape(ctx, a, new_shape, 2);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out, reshaped, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyIndexingCtx *ictx = poly_indexing_ctx_new(ctx);
  poly_realize_map_build(ictx, sink);
  poly_range_propagate(ictx, sink);

  /* STORE: shape (2,3) → 2 output ranges */
  PolyRangeEntry *re_store = poly_range_map_get(ictx, store);
  ASSERT_NOT_NULL(re_store);
  ASSERT_INT_EQ(re_store->n_out, 2);

  /* RESHAPE: inherits STORE's 2D ranges as output,
   * but input is 1D (a's shape is (6,)) */
  PolyRangeEntry *re_reshape = poly_range_map_get(ictx, reshaped);
  ASSERT_NOT_NULL(re_reshape);
  ASSERT_INT_EQ(re_reshape->n_out, 2);
  ASSERT_INT_EQ(re_reshape->n_in, 1);

  /* a gets RESHAPE's 1D input range */
  PolyRangeEntry *re_a = poly_range_map_get(ictx, a);
  ASSERT_NOT_NULL(re_a);
  ASSERT_INT_EQ(re_a->n_out, 1);

  poly_indexing_ctx_destroy(ictx);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, reshape_indices_simplify_bounded_floor_ops_like_pinned) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *source = poly_reshape(ctx, poly_buffer_f32(ctx, 6), (int64_t[]){2, 3}, 2);
  PolyUOp *reshape = poly_reshape(ctx, source, (int64_t[]){6}, 1);
  ASSERT_NOT_NULL(reshape);
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(6));
  PolyUOp *range =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_INDEX, bound, poly_arg_range(0, POLY_AXIS_LOOP));
  PolyUOp *out_ranges[] = {range};
  PolyUOp *in_ranges[2] = {NULL, NULL};
  int n_in = 0;

  ASSERT_TRUE(poly_reshape_indices(ctx, reshape, out_ranges, 1, in_ranges, &n_in));
  ASSERT_INT_EQ(n_in, 2);

  ASSERT_NOT_NULL(in_ranges[0]);
  ASSERT_NOT_NULL(in_ranges[1]);
  /* Pinned _apply_reshape((2,3),(6,), RANGE(6)) returns RANGE//3 and RANGE%3:
   * the outer `%2` is an identity under the proved [0,1] quotient bound. */
  ASSERT_EQ(in_ranges[0]->op, POLY_OP_FLOORDIV);
  ASSERT_EQ(in_ranges[1]->op, POLY_OP_FLOORMOD);
  ASSERT_PTR_EQ(in_ranges[0]->src[0], range);
  ASSERT_PTR_EQ(in_ranges[1]->src[0], range);
  ASSERT_INT_EQ(count_ops(ctx, in_ranges[0], POLY_OP_CDIV), 0);
  ASSERT_INT_EQ(count_ops(ctx, in_ranges[0], POLY_OP_CMOD), 0);
  ASSERT_INT_EQ(count_ops(ctx, in_ranges[1], POLY_OP_CDIV), 0);
  ASSERT_INT_EQ(count_ops(ctx, in_ranges[1], POLY_OP_CMOD), 0);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, reshape_indices_simplify_floor_ops_under_pad_validity) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  /* Pinned schedule/indexing.py:113-127 rewrites reshape coordinates with
   * symbolic+pm_simplify_valid+pm_drop_and_clauses. A padded 2x4x4 input has
   * a flat coordinate in [0,31] whenever both pad guards hold, so `% 32` is
   * removed before the CALL body is formed. */
  PolyUOp *reshape =
      poly_reshape(ctx, poly_buffer_f32(ctx, 32), (int64_t[]){1, 2, 4, 4}, 4);
  ASSERT_NOT_NULL(reshape);
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(1));
  PolyUOp *five = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(5));
  PolyUOp *truev = poly_uop0(ctx, POLY_OP_CONST, POLY_BOOL, poly_arg_bool(true));
  PolyUOp *invalid = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_invalid());
  PolyUOp *channel = poly_uop_range(ctx, 2, 0, POLY_AXIS_REDUCE);
  PolyUOp *row = poly_uop_range(ctx, 6, 1, POLY_AXIS_REDUCE);
  PolyUOp *col = poly_uop_range(ctx, 6, 2, POLY_AXIS_REDUCE);
  PolyUOp *padded[2] = {NULL, NULL};
  PolyUOp *spatial[2] = {row, col};
  for (int i = 0; i < 2; i++) {
    PolyUOp *below =
        poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, spatial[i], one, poly_arg_none());
    PolyUOp *lower =
        poly_uop2(ctx, POLY_OP_CMPNE, POLY_BOOL, below, truev, poly_arg_none());
    PolyUOp *upper =
        poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, spatial[i], five, poly_arg_none());
    PolyUOp *valid =
        poly_uop2(ctx, POLY_OP_AND, POLY_BOOL, lower, upper, poly_arg_none());
    PolyUOp *shifted = poly_uop2(
        ctx, POLY_OP_ADD, POLY_INDEX, spatial[i],
        poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(-1)), poly_arg_none()
    );
    padded[i] =
        poly_uop3(ctx, POLY_OP_WHERE, POLY_INDEX, valid, shifted, invalid, poly_arg_none());
  }
  PolyUOp *out_ranges[4] = {
      poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(0)),
      channel,
      padded[0],
      padded[1],
  };
  PolyUOp *in_ranges[1] = {NULL};
  int n_in = 0;

  ASSERT_TRUE(poly_reshape_indices(ctx, reshape, out_ranges, 4, in_ranges, &n_in));
  ASSERT_INT_EQ(n_in, 1);
  ASSERT_NOT_NULL(in_ranges[0]);
  ASSERT_EQ(in_ranges[0]->op, POLY_OP_WHERE);
  ASSERT_INT_EQ(in_ranges[0]->n_src, 3);
  ASSERT_EQ(in_ranges[0]->src[2]->op, POLY_OP_CONST);
  ASSERT_EQ(in_ranges[0]->src[2]->arg.kind, POLY_ARG_INVALID);
  ASSERT_INT_EQ(count_ops(ctx, in_ranges[0], POLY_OP_FLOORMOD), 0);
  ASSERT_INT_EQ(count_ops(ctx, in_ranges[0], POLY_OP_FLOORDIV), 0);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, symbolic_reshape_indices_preserve_exact_dimensions) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  /* Pinned schedule/indexing.py:113-127,142-145 flattens and decomposes with
   * exact symbolic in_shape/marg, never their allocation maxima. */
  PolyUOp *n = poly_define_var(ctx, "n", 1, 8);
  PolyUOp *two = poly_const_int(ctx, 2);
  PolyUOp *input_shape_srcs[] = {two, n};
  PolyUOp *input_shape = poly_uop(
      ctx, POLY_OP_STACK, poly_dtype_vec(POLY_INDEX, 2), input_shape_srcs, 2, poly_arg_none()
  );
  PolyUOp *base = poly_reshape(ctx, poly_buffer_f32(ctx, 2), (int64_t[]){2, 1}, 2);
  PolyUOp *expand_srcs[] = {base, input_shape};
  PolyUOp *expanded = poly_uop(ctx, POLY_OP_EXPAND, POLY_FLOAT32, expand_srcs, 2, poly_arg_none());
  PolyUOp *output_shape_srcs[] = {n, two};
  PolyUOp *output_shape = poly_uop(
      ctx, POLY_OP_STACK, poly_dtype_vec(POLY_INDEX, 2), output_shape_srcs, 2, poly_arg_none()
  );
  PolyUOp *reshape_srcs[] = {expanded, output_shape};
  PolyUOp *reshape = poly_uop(ctx, POLY_OP_RESHAPE, POLY_FLOAT32, reshape_srcs, 2, poly_arg_none());
  ASSERT_NOT_NULL(reshape);

  PolyUOp *out_ranges[] = {poly_const_int(ctx, 2), poly_const_int(ctx, 0)};
  PolyUOp *in_ranges[2] = {NULL, NULL};
  int n_in = 0;
  ASSERT_TRUE(poly_reshape_indices(ctx, reshape, out_ranges, 2, in_ranges, &n_in));
  ASSERT_INT_EQ(n_in, 2);
  ASSERT_NOT_NULL(in_ranges[0]);
  ASSERT_NOT_NULL(in_ranges[1]);
  ASSERT_INT_EQ(in_ranges[0]->op, POLY_OP_FLOORMOD);
  ASSERT_INT_EQ(in_ranges[0]->src[0]->op, POLY_OP_FLOORDIV);
  ASSERT_INT_EQ(in_ranges[1]->op, POLY_OP_FLOORMOD);
  ASSERT_TRUE(count_ops(ctx, in_ranges[0], POLY_OP_DEFINE_VAR) > 0);
  ASSERT_TRUE(count_ops(ctx, in_ranges[1], POLY_OP_DEFINE_VAR) > 0);

  PolyUOp *four = poly_const_int(ctx, 4);
  PolyUOp *from[] = {n};
  PolyUOp *to[] = {four};
  for (int i = 0; i < 2; i++) {
    PolyUOp *bound_index = poly_uop_substitute(ctx, in_ranges[i], from, to, 1);
    bound_index = poly_graph_rewrite(ctx, bound_index, poly_symbolic());
    int64_t value = -1;
    ASSERT_INT_EQ(poly_uop_const_i64(bound_index, &value), 0);
    ASSERT_INT_EQ(value, i == 0 ? 1 : 0);
  }

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, partial_reshape_index_preserves_matching_suffix) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  /* Pinned schedule/rangeify.py:68-77 maps only the indexed prefixes when
   * the remaining RESHAPE output suffix equals an input suffix. */
  PolyUOp *source = poly_buffer_f32(ctx, 3);
  PolyUOp *reshape = poly_reshape(ctx, source, (int64_t[]){1, 3}, 2);
  PolyUOp *out_ranges[] = {poly_const_int(ctx, 0)};
  PolyUOp *in_ranges[1] = {NULL};
  int n_in = -1;
  ASSERT_TRUE(poly_reshape_indices(ctx, reshape, out_ranges, 1, in_ranges, &n_in));
  ASSERT_INT_EQ(n_in, 0);

  PolyUOp *bad = poly_reshape(ctx, poly_buffer_f32(ctx, 6), (int64_t[]){2, 3}, 2);
  ASSERT_NOT_NULL(bad);
  n_in = -1;
  ASSERT_FALSE(poly_reshape_indices(ctx, bad, out_ranges, 1, in_ranges, &n_in));

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, partial_reshape_matches_fully_canonical_symbolic_suffixes) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  /* Pinned tinygrad/schedule/rangeify.py:68-77 compares movement suffixes
   * after as_shape has run the complete symbolic matcher. */
  PolyUOp *n = poly_uop0(ctx, POLY_OP_DEFINE_VAR, POLY_INT32, poly_arg_define_var("n", 1, 8));
  PolyUOp *y = poly_uop0(ctx, POLY_OP_DEFINE_VAR, POLY_INT32, poly_arg_define_var("y", 1, 8));
  PolyUOp *one = poly_const_int(ctx, 1);
  PolyUOp *index_one = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(1));
  PolyUOp *two = poly_const_int(ctx, 2);
  PolyUOp *three = poly_const_int(ctx, 3);
  PolyUOp *five = poly_const_int(ctx, 5);
  PolyUOp *left[5] = {
      poly_uop2(ctx, POLY_OP_ADD, POLY_INDEX, n, n, poly_arg_none()),
      poly_uop2(
          ctx, POLY_OP_ADD, POLY_INDEX,
          poly_uop2(ctx, POLY_OP_ADD, POLY_INDEX, n, one, poly_arg_none()), two, poly_arg_none()
      ),
      poly_uop2(
          ctx, POLY_OP_MUL, POLY_INDEX, two,
          poly_uop2(ctx, POLY_OP_ADD, POLY_INDEX, n, one, poly_arg_none()), poly_arg_none()
      ),
      poly_uop2(
          ctx, POLY_OP_ADD, POLY_INDEX,
          poly_uop2(ctx, POLY_OP_MUL, POLY_INDEX, n, two, poly_arg_none()),
          poly_uop2(ctx, POLY_OP_MUL, POLY_INDEX, n, three, poly_arg_none()), poly_arg_none()
      ),
      poly_uop2(
          ctx, POLY_OP_ADD, POLY_INDEX,
          poly_uop2(ctx, POLY_OP_ADD, POLY_INDEX, y, n, poly_arg_none()), n, poly_arg_none()
      ),
  };
  PolyUOp *right[5] = {
      poly_uop2(ctx, POLY_OP_MUL, POLY_INDEX, n, two, poly_arg_none()),
      poly_uop2(ctx, POLY_OP_ADD, POLY_INDEX, n, three, poly_arg_none()),
      poly_uop2(
          ctx, POLY_OP_ADD, POLY_INDEX,
          poly_uop2(ctx, POLY_OP_MUL, POLY_INDEX, n, two, poly_arg_none()), two, poly_arg_none()
      ),
      poly_uop2(ctx, POLY_OP_MUL, POLY_INDEX, n, five, poly_arg_none()),
      poly_uop2(
          ctx, POLY_OP_ADD, POLY_INDEX, y,
          poly_uop2(ctx, POLY_OP_MUL, POLY_INDEX, n, two, poly_arg_none()), poly_arg_none()
      ),
  };
  PolyUOp *base = poly_reshape(ctx, poly_buffer_f32(ctx, 3), (int64_t[]){3, 1}, 2);
  ASSERT_NOT_NULL(base);

  for (int i = 0; i < 5; i++) {
    PolyUOp *expand_shape_src[] = {three, left[i]};
    PolyUOp *expand_shape = poly_uop(
        ctx, POLY_OP_STACK, poly_dtype_vec(POLY_INDEX, 2), expand_shape_src, 2, poly_arg_none()
    );
    PolyUOp *expand_src[] = {base, expand_shape};
    PolyUOp *expanded = poly_uop(ctx, POLY_OP_EXPAND, POLY_FLOAT32, expand_src, 2, poly_arg_none());
    PolyUOp *reshape_shape_src[] = {three, right[i]};
    PolyUOp *reshape_shape = poly_uop(
        ctx, POLY_OP_STACK, poly_dtype_vec(POLY_INDEX, 2), reshape_shape_src, 2, poly_arg_none()
    );
    PolyUOp *reshape_src[] = {expanded, reshape_shape};
    PolyUOp *reshaped =
        poly_uop(ctx, POLY_OP_RESHAPE, POLY_FLOAT32, reshape_src, 2, poly_arg_none());
    PolyUOp *out_ranges[] = {index_one};
    PolyUOp *in_ranges[POLY_MAX_DIMS] = {0};
    int n_in = -1;

    ASSERT_NOT_NULL(reshaped);
    ASSERT_TRUE(poly_reshape_indices(ctx, reshaped, out_ranges, 1, in_ranges, &n_in));
    ASSERT_INT_EQ(n_in, 1);
    ASSERT_PTR_EQ(poly_graph_rewrite(ctx, in_ranges[0], poly_symbolic()), index_one);
  }

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, range_prop_permute) {
  /* permute(a, [1,0]) where a has shape (3,4) → (4,3):
   * ranges should be reordered by inverse permutation */
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a_flat = poly_buffer(ctx, POLY_FLOAT32, 12);
  int64_t shape_2d[] = {3, 4};
  PolyUOp *a = poly_reshape(ctx, a_flat, shape_2d, 2);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, 12);
  int64_t perm[] = {1, 0};
  PolyUOp *permuted = poly_permute(ctx, a, perm, 2);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out, permuted, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyIndexingCtx *ictx = poly_indexing_ctx_new(ctx);
  poly_realize_map_build(ictx, sink);
  poly_range_propagate(ictx, sink);

  /* STORE: shape (4,3) → 2 ranges: r0=RANGE(4), r1=RANGE(3) */
  PolyRangeEntry *re_store = poly_range_map_get(ictx, store);
  ASSERT_NOT_NULL(re_store);
  ASSERT_INT_EQ(re_store->n_out, 2);

  /* PERMUTE: output is [r0, r1], input is [r1, r0] (swapped by perm [1,0]) */
  PolyRangeEntry *re_perm = poly_range_map_get(ictx, permuted);
  ASSERT_NOT_NULL(re_perm);
  ASSERT_INT_EQ(re_perm->n_in, 2);
  ASSERT_INT_EQ(re_perm->n_out, 2);
  /* in_rngs[0] should be out_rngs[1] (perm maps out[0]→in[1], out[1]→in[0]) */
  ASSERT_PTR_EQ(re_perm->in_rngs[0], re_perm->out_rngs[1]);
  ASSERT_PTR_EQ(re_perm->in_rngs[1], re_perm->out_rngs[0]);

  poly_indexing_ctx_destroy(ictx);
  poly_ctx_destroy(ctx);
  PASS();
}

/* apply_rangeify helper */

/* Count how many UOps of a given op type appear in the graph */
static int count_ops(PolyCtx *ctx, PolyUOp *root, PolyOps op) {
  int n;
  PolyUOp **topo = poly_toposort(ctx, root, &n);
  int count = 0;
  for (int i = 0; i < n; i++)
    if (topo[i]->op == op) count++;
  return count;
}

static int count_reduce_arg_kind(PolyCtx *ctx, PolyUOp *root, PolyArgKind kind) {
  int n;
  PolyUOp **topo = poly_toposort(ctx, root, &n);
  int count = 0;
  for (int i = 0; i < n; i++)
    if (topo[i]->op == POLY_OP_REDUCE && topo[i]->arg.kind == kind) count++;
  return count;
}

static bool is_direct_storage_source(PolyUOp *u) {
  return u && (u->op == POLY_OP_BUFFER || u->op == POLY_OP_PARAM || u->op == POLY_OP_BUFFER_VIEW);
}

static int count_alu_direct_storage_sources(PolyCtx *ctx, PolyUOp *root) {
  int n;
  PolyUOp **topo = poly_toposort(ctx, root, &n);
  int count = 0;
  for (int i = 0; i < n; i++) {
    PolyUOp *u = topo[i];
    if (!u || !poly_opset_has(POLY_GROUP_ALU, u->op)) continue;
    for (int j = 0; j < u->n_src; j++)
      if (is_direct_storage_source(u->src[j])) count++;
  }
  return count;
}

typedef struct {
  int store_target_index;
  int target_ptr_index;
  int read_ptr_index;
  int read_scalar_index;
} IndexKindCounts;

static IndexKindCounts count_index_kinds(PolyCtx *ctx, PolyUOp *root) {
  IndexKindCounts c = {0};
  int n;
  PolyUOp **topo = poly_toposort(ctx, root, &n);
  PolyUOp *store_targets[128] = {0};
  int n_store_targets = 0;

  for (int i = 0; i < n; i++) {
    PolyUOp *u = topo[i];
    if (!u || u->op != POLY_OP_STORE || u->n_src < 1) continue;
    if (u->src[0] && u->src[0]->op == POLY_OP_INDEX && n_store_targets < 128)
      store_targets[n_store_targets++] = u->src[0];
  }

  for (int i = 0; i < n; i++) {
    PolyUOp *u = topo[i];
    if (!u || u->op != POLY_OP_INDEX) continue;
    bool is_target = false;
    for (int j = 0; j < n_store_targets; j++) {
      if (store_targets[j] == u) {
        is_target = true;
        break;
      }
    }
    if (is_target) {
      c.store_target_index++;
      if (u->dtype.is_ptr) c.target_ptr_index++;
    } else if (u->dtype.is_ptr) {
      c.read_ptr_index++;
    } else {
      c.read_scalar_index++;
    }
  }

  return c;
}

/* Run full rangeify pipeline up to and including apply_rangeify */
static PolyUOp *run_apply_rangeify(PolyIndexingCtx *ictx, PolyUOp *sink) {
  poly_realize_map_build(ictx, sink);
  poly_range_propagate(ictx, sink);
  return poly_run_rangeify(ictx, sink);
}

/* Apply rangeify tests */

TEST(rangeify, apply_movement_removed) {
  /* reshape(a, (2,5)) + expand(b, (2,5)) → STORE → SINK
   * After apply_rangeify: no RESHAPE or EXPAND ops remain */
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *b = poly_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, 10);
  int64_t dims[] = {2, 5};
  PolyUOp *reshaped = poly_reshape(ctx, a, dims, 2);
  PolyUOp *expanded = poly_expand(ctx, b, dims, 2);
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, reshaped, expanded, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out, add, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  /* Before: RESHAPE and EXPAND exist */
  ASSERT_INT_EQ(count_ops(ctx, sink, POLY_OP_RESHAPE), 1);
  ASSERT_INT_EQ(count_ops(ctx, sink, POLY_OP_EXPAND), 1);

  PolyIndexingCtx *ictx = poly_indexing_ctx_new(ctx);
  PolyUOp *result = run_apply_rangeify(ictx, sink);
  ASSERT_NOT_NULL(result);

  /* After: no movement ops */
  ASSERT_INT_EQ(count_ops(ctx, result, POLY_OP_RESHAPE), 0);
  ASSERT_INT_EQ(count_ops(ctx, result, POLY_OP_EXPAND), 0);

  /* ADD still exists */
  ASSERT_INT_EQ(count_ops(ctx, result, POLY_OP_ADD), 1);

  poly_indexing_ctx_destroy(ictx);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, apply_reduce_to_reduce) {
  /* sum(a) → STORE → SINK
   * Pinned indexing.py:89-100 keeps the REDUCE op while replacing its
   * axis tuple with RANGE sources. */
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, 1);
  int64_t axes[] = {0};
  PolyUOp *reduce = poly_reduce_axis(ctx, POLY_OP_ADD, a, axes, 1);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out, reduce, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  /* Before: tensor REDUCE has (op, axes) and one value source. */
  ASSERT_INT_EQ(count_ops(ctx, sink, POLY_OP_REDUCE_AXIS), 0);
  ASSERT_INT_EQ(count_reduce_arg_kind(ctx, sink, POLY_ARG_REDUCE_AXIS), 1);
  ASSERT_INT_EQ(count_reduce_arg_kind(ctx, sink, POLY_ARG_OPS), 0);

  PolyIndexingCtx *ictx = poly_indexing_ctx_new(ctx);
  PolyUOp *result = run_apply_rangeify(ictx, sink);
  ASSERT_NOT_NULL(result);

  /* After: lowered REDUCE has an op arg and RANGE sources. */
  ASSERT_INT_EQ(count_ops(ctx, result, POLY_OP_REDUCE_AXIS), 0);
  ASSERT_INT_EQ(count_reduce_arg_kind(ctx, result, POLY_ARG_REDUCE_AXIS), 0);
  ASSERT_INT_EQ(count_reduce_arg_kind(ctx, result, POLY_ARG_OPS), 1);

  /* Find the REDUCE node and verify it has range sources */
  int n;
  PolyUOp **topo = poly_toposort(ctx, result, &n);
  for (int i = 0; i < n; i++) {
    if (topo[i]->op == POLY_OP_REDUCE) {
      /* src[0] = value, src[1+] = reduce ranges */
      ASSERT_TRUE(topo[i]->n_src >= 2);
      /* arg = the reduce op (ADD = POLY_OP_ADD) */
      ASSERT_INT_EQ(topo[i]->arg.i, (int64_t)POLY_OP_ADD);
      break;
    }
  }

  poly_indexing_ctx_destroy(ictx);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, reshape_indices_use_placeholder_ranges_before_valid_simplification) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  /* Pinned indexing.py:142-145 simplifies the coordinate SINK, temporarily
   * replaces active ranges with PLACEHOLDER ranges, applies the reshape, and
   * restores the originals. This exact ResNet split removes invalid-gated
   * remainders and leaves only the two quotient coordinates. */
  PolyUOp *flat = poly_buffer_f32(ctx, 1024);
  PolyUOp *input = poly_reshape(
      ctx, flat, (int64_t[]){2, 128, 1, 2, 1, 2}, 6
  );
  PolyUOp *reshape = poly_reshape(
      ctx, input, (int64_t[]){2, 128, 1, 2, 1, 1, 2, 1}, 8
  );
  PolyUOp *r0 = poly_uop_range(ctx, 2, 0, POLY_AXIS_LOOP);
  PolyUOp *r1 = poly_uop_range(ctx, 128, 1, POLY_AXIS_LOOP);
  PolyUOp *spatial[2] = {
      poly_uop_range(ctx, 4, 2, POLY_AXIS_LOOP),
      poly_uop_range(ctx, 4, 3, POLY_AXIS_LOOP),
  };
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(0));
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(1));
  PolyUOp *two = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(2));
  PolyUOp *minus_one = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(-1));
  PolyUOp *invalid = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_invalid());
  PolyUOp *quotient[2] = {NULL, NULL};
  PolyUOp *gated[2] = {NULL, NULL};
  for (int i = 0; i < 2; i++) {
    PolyUOp *mod = poly_uop2(
        ctx, POLY_OP_FLOORMOD, POLY_INDEX, spatial[i], two, poly_arg_none()
    );
    quotient[i] = poly_uop2(
        ctx, POLY_OP_FLOORDIV, POLY_INDEX, spatial[i], two, poly_arg_none()
    );
    PolyUOp *valid = poly_uop2(
        ctx, POLY_OP_CMPLT, POLY_BOOL, mod, one, poly_arg_none()
    );
    PolyUOp *negative_zero = poly_uop2(
        ctx, POLY_OP_MUL, POLY_INDEX, zero, minus_one, poly_arg_none()
    );
    PolyUOp *shifted = poly_uop2(
        ctx, POLY_OP_ADD, POLY_INDEX, mod, negative_zero, poly_arg_none()
    );
    gated[i] = poly_uop3(
        ctx, POLY_OP_WHERE, POLY_INDEX, valid, shifted, invalid, poly_arg_none()
    );
  }
  PolyUOp *out_ranges[8] = {
      r0, r1, zero, quotient[0], gated[0], zero, quotient[1], gated[1]
  };
  PolyUOp *in_ranges[POLY_MAX_DIMS] = {0};
  int n_in = 0;

  ASSERT_TRUE(poly_reshape_indices(ctx, reshape, out_ranges, 8, in_ranges, &n_in));
  ASSERT_INT_EQ(n_in, 6);
  PolyUOp *sink = poly_uop(ctx, POLY_OP_SINK, POLY_VOID, in_ranges, n_in, poly_arg_none());
  ASSERT_INT_EQ(count_ops(ctx, sink, POLY_OP_WHERE), 0);
  ASSERT_INT_EQ(count_ops(ctx, sink, POLY_OP_FLOORMOD), 0);
  ASSERT_INT_EQ(count_ops(ctx, sink, POLY_OP_FLOORDIV), 2);
  ASSERT_INT_EQ(count_ops(ctx, sink, POLY_OP_RANGE), 4);
  int n_nodes = 0;
  PolyUOp **nodes = poly_toposort_alloc(ctx, sink, &n_nodes);
  ASSERT_NOT_NULL(nodes);
  poly_toposort_free(nodes);
  ASSERT_INT_EQ(n_nodes, 11);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, apply_pad_to_where) {
  /* pad(a, ((1,1),)) → STORE → SINK
   * After apply: PAD becomes WHERE with valid mask */
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, 12);
  int64_t pairs[][2] = {{1, 1}};
  PolyUOp *padded = poly_pad(ctx, a, pairs, 1);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out, padded, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  /* Before: PAD exists, WHERE does not */
  ASSERT_INT_EQ(count_ops(ctx, sink, POLY_OP_PAD), 1);
  ASSERT_INT_EQ(count_ops(ctx, sink, POLY_OP_WHERE), 0);

  PolyIndexingCtx *ictx = poly_indexing_ctx_new(ctx);
  PolyUOp *result = run_apply_rangeify(ictx, sink);
  ASSERT_NOT_NULL(result);

  /* After: PAD gone, WHERE exists */
  ASSERT_INT_EQ(count_ops(ctx, result, POLY_OP_PAD), 0);
  ASSERT_INT_EQ(count_ops(ctx, result, POLY_OP_WHERE), 1);

  poly_indexing_ctx_destroy(ictx);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, apply_pad_coordinates_preserve_invalid_like_tinygrad) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  /* Pinned schedule/indexing.py:131-140 returns
   * valid.where(r-off, Invalid) for both the current movement vocabulary and
   * imported legacy PAD spelling. This keeps address and validity separable
   * for UOp.get_idx/get_valid. */
  PolyUOp *input = poly_buffer_f32(ctx, 3);
  int64_t pairs[1][2] = {{1, 2}};
  PolyUOp *current = poly_pad(ctx, input, pairs, 1);
  PolyArg legacy_arg = poly_arg_none();
  legacy_arg.kind = POLY_ARG_PAIR_TUPLE;
  legacy_arg.pair_tuple.pairs = pairs;
  legacy_arg.pair_tuple.n = 1;
  PolyUOp *legacy = poly_uop1(ctx, POLY_OP_PAD, POLY_FLOAT32, input, legacy_arg);
  PolyUOp *movements[2] = {current, legacy};
  int64_t dims[1] = {3};
  PolyShape input_shape = {.dims = dims, .ndim = 1};
  PolyUOp *output_range = poly_uop_range(ctx, 6, 0, POLY_AXIS_LOOP);

  for (int i = 0; i < 2; i++) {
    PolyUOp *input_ranges[POLY_MAX_DIMS] = {0};
    PolyUOp *valid = NULL;
    int n_input = 0;
    ASSERT_TRUE(poly_apply_movement_op(
        ctx, movements[i], POLY_OP_PAD, input_shape, movements[i]->arg, &output_range, 1,
        input_ranges, &n_input, &valid
    ));
    ASSERT_INT_EQ(n_input, 1);
    ASSERT_NOT_NULL(valid);
    ASSERT_NOT_NULL(input_ranges[0]);
    ASSERT_INT_EQ(input_ranges[0]->op, POLY_OP_WHERE);
    ASSERT_INT_EQ(input_ranges[0]->n_src, 3);
    ASSERT_PTR_EQ(input_ranges[0]->src[0], valid);
    ASSERT_INT_EQ(input_ranges[0]->src[2]->op, POLY_OP_CONST);
    ASSERT_INT_EQ(input_ranges[0]->src[2]->arg.kind, POLY_ARG_INVALID);
    ASSERT_TRUE(poly_dtype_eq(input_ranges[0]->dtype, POLY_INDEX));
    ASSERT_TRUE(poly_dtype_eq(input_ranges[0]->src[2]->dtype, POLY_INDEX));
  }

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, apply_pad_simplifies_new_valid_before_where_like_tinygrad) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  /* Pinned schedule/indexing.py:136-141 rewrites only the newly introduced
   * PAD validity before WHERE construction. For begin=0 and RANGE(6), the
   * lower bound is proved true while the explicit r-0 coordinate remains. */
  PolyUOp *input = poly_buffer_f32(ctx, 5);
  int64_t pairs[1][2] = {{0, 1}};
  PolyUOp *current = poly_pad(ctx, input, pairs, 1);
  PolyArg legacy_arg = poly_arg_none();
  legacy_arg.kind = POLY_ARG_PAIR_TUPLE;
  legacy_arg.pair_tuple.pairs = pairs;
  legacy_arg.pair_tuple.n = 1;
  PolyUOp *legacy = poly_uop1(ctx, POLY_OP_PAD, POLY_FLOAT32, input, legacy_arg);
  PolyUOp *movements[2] = {current, legacy};
  int64_t dims[1] = {5};
  PolyShape input_shape = {.dims = dims, .ndim = 1};
  PolyUOp *output_range = poly_uop_range(ctx, 6, 0, POLY_AXIS_LOOP);

  for (int i = 0; i < 2; i++) {
    PolyUOp *input_ranges[POLY_MAX_DIMS] = {0};
    PolyUOp *valid = NULL;
    int n_input = 0;
    ASSERT_TRUE(poly_apply_movement_op(
        ctx, movements[i], POLY_OP_PAD, input_shape, movements[i]->arg, &output_range, 1,
        input_ranges, &n_input, &valid
    ));
    ASSERT_INT_EQ(n_input, 1);
    ASSERT_NOT_NULL(input_ranges[0]);
    ASSERT_EQ(input_ranges[0]->op, POLY_OP_WHERE);
    ASSERT_PTR_EQ(input_ranges[0]->src[0], valid);
    ASSERT_EQ(valid->op, POLY_OP_CMPLT);
    ASSERT_PTR_EQ(valid->src[0], output_range);
    ASSERT_EQ(valid->src[1]->op, POLY_OP_CONST);
    ASSERT_INT_EQ(valid->src[1]->arg.i, 5);
    ASSERT_EQ(input_ranges[0]->src[1]->op, POLY_OP_ADD);
    ASSERT_PTR_EQ(input_ranges[0]->src[1]->src[0], output_range);
    ASSERT_EQ(input_ranges[0]->src[1]->src[1]->op, POLY_OP_MUL);
    ASSERT_INT_EQ(count_ops(ctx, input_ranges[0], POLY_OP_AND), 0);
    ASSERT_INT_EQ(count_ops(ctx, input_ranges[0], POLY_OP_CMPNE), 0);
    ASSERT_INT_EQ(count_ops(ctx, input_ranges[0], POLY_OP_CMPLT), 1);
    ASSERT_INT_EQ(count_ops(ctx, input_ranges[0], POLY_OP_WHERE), 1);
  }

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, apply_pad_leaves_unchanged_dimensions_as_original_ranges) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  /* Pinned indexing.py:137 returns r directly when offset=0 and the PAD
   * output size equals the input size. Do not add a true WHERE to that axis. */
  PolyUOp *flat = poly_buffer_f32(ctx, 35);
  PolyUOp *input = poly_reshape(ctx, flat, (int64_t[]){5, 7}, 2);
  int64_t pairs[2][2] = {{0, 0}, {0, 1}};
  PolyUOp *current = poly_pad(ctx, input, pairs, 2);
  PolyArg legacy_arg = poly_arg_none();
  legacy_arg.kind = POLY_ARG_PAIR_TUPLE;
  legacy_arg.pair_tuple.pairs = pairs;
  legacy_arg.pair_tuple.n = 2;
  PolyUOp *legacy = poly_uop1(ctx, POLY_OP_PAD, POLY_FLOAT32, input, legacy_arg);
  PolyUOp *movements[2] = {current, legacy};
  PolyShape input_shape = {.dims = (int64_t[]){5, 7}, .ndim = 2};
  PolyUOp *output_ranges[2] = {
      poly_uop_range(ctx, 5, 0, POLY_AXIS_LOOP),
      poly_uop_range(ctx, 8, 1, POLY_AXIS_LOOP),
  };

  for (int i = 0; i < 2; i++) {
    PolyUOp *input_ranges[POLY_MAX_DIMS] = {0};
    PolyUOp *valid = NULL;
    int n_input = 0;
    ASSERT_TRUE(poly_apply_movement_op(
        ctx, movements[i], POLY_OP_PAD, input_shape, movements[i]->arg, output_ranges, 2,
        input_ranges, &n_input, &valid
    ));
    ASSERT_INT_EQ(n_input, 2);
    ASSERT_PTR_EQ(input_ranges[0], output_ranges[0]);
    ASSERT_EQ(input_ranges[1]->op, POLY_OP_WHERE);
    ASSERT_NOT_NULL(valid);
    ASSERT_PTR_EQ(input_ranges[1]->src[0], valid);
  }

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, apply_pad_wrapper_keeps_inherited_validity_from_unchanged_axes) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  /* Pinned schedule/indexing.py:82-86 forms the PAD wrapper from get_valid()
   * of every transformed coordinate. Axis 0 is unchanged by this PAD but
   * already carries validity from an earlier movement; axis 1 gains a new
   * predicate. Both must appear, in axis order, in valid_out. */
  PolyUOp *flat = poly_buffer_f32(ctx, 2);
  PolyUOp *input = poly_reshape(ctx, flat, (int64_t[]){2, 1}, 2);
  int64_t pairs[2][2] = {{0, 0}, {0, 1}};
  PolyUOp *current = poly_pad(ctx, input, pairs, 2);
  PolyArg legacy_arg = poly_arg_none();
  legacy_arg.kind = POLY_ARG_PAIR_TUPLE;
  legacy_arg.pair_tuple.pairs = pairs;
  legacy_arg.pair_tuple.n = 2;
  PolyUOp *legacy = poly_uop1(ctx, POLY_OP_PAD, POLY_FLOAT32, input, legacy_arg);
  PolyUOp *movements[2] = {current, legacy};
  PolyShape input_shape = {.dims = (int64_t[]){2, 1}, .ndim = 2};
  PolyUOp *two = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(2));
  PolyUOp *r0 = poly_uop_range(ctx, 2, 0, POLY_AXIS_LOOP);
  PolyUOp *r1 = poly_uop_range(ctx, 2, 1, POLY_AXIS_LOOP);
  PolyUOp *old_valid = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, r0, two, poly_arg_none());
  PolyUOp *invalid = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_invalid());
  PolyUOp *output_ranges[2] = {
      poly_uop3(ctx, POLY_OP_WHERE, POLY_INDEX, old_valid, r0, invalid, poly_arg_none()),
      r1,
  };

  for (int i = 0; i < 2; i++) {
    PolyUOp *input_ranges[POLY_MAX_DIMS] = {0};
    PolyUOp *valid = NULL;
    int n_input = 0;
    ASSERT_TRUE(poly_apply_movement_op(
        ctx, movements[i], POLY_OP_PAD, input_shape, movements[i]->arg, output_ranges, 2,
        input_ranges, &n_input, &valid
    ));
    ASSERT_INT_EQ(n_input, 2);
    ASSERT_PTR_EQ(input_ranges[0], output_ranges[0]);
    ASSERT_EQ(input_ranges[1]->op, POLY_OP_WHERE);
    ASSERT_NOT_NULL(valid);
    ASSERT_EQ(valid->op, POLY_OP_AND);
    ASSERT_PTR_EQ(valid->src[0], old_valid);
    ASSERT_PTR_EQ(valid->src[1], input_ranges[1]->src[0]);
    ASSERT_INT_EQ(count_ops(ctx, valid, POLY_OP_AND), 1);
    ASSERT_INT_EQ(count_ops(ctx, valid, POLY_OP_CMPLT), 2);
  }

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, index_projection_matches_pinned_get_idx_get_valid) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  /* Pinned uop/ops.py:571-581 projects Invalid-bearing WHERE coordinates,
   * recurses through STACK, and leaves ordinary integer WHERE values intact. */
  PolyUOp *idx0 = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(3));
  PolyUOp *idx1 = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(7));
  PolyUOp *gate = poly_uop0(ctx, POLY_OP_PARAM, POLY_BOOL, poly_arg_int(0));
  PolyUOp *invalid = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_invalid());
  PolyUOp *gated = poly_uop3(ctx, POLY_OP_WHERE, POLY_INDEX, gate, idx0, invalid, poly_arg_none());
  PolyUOp *ordinary = poly_uop3(
      ctx, POLY_OP_WHERE, POLY_INDEX, gate, idx0,
      poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(0)), poly_arg_none()
  );
  PolyUOp *lanes[2] = {gated, idx1};
  PolyUOp *stack =
      poly_uop(ctx, POLY_OP_STACK, poly_dtype_vec(POLY_INDEX, 2), lanes, 2, poly_arg_none());

  ASSERT_PTR_EQ(poly_index_get_idx(ctx, gated), idx0);
  ASSERT_PTR_EQ(poly_index_get_valid(ctx, gated), gate);
  ASSERT_PTR_EQ(poly_index_get_idx(ctx, ordinary), ordinary);
  PolyUOp *ordinary_valid = poly_index_get_valid(ctx, ordinary);
  ASSERT_INT_EQ(ordinary_valid->op, POLY_OP_CONST);
  ASSERT_TRUE(ordinary_valid->arg.kind == POLY_ARG_BOOL && ordinary_valid->arg.b);
  PolyUOp *invalid_valid = poly_index_get_valid(ctx, invalid);
  ASSERT_INT_EQ(invalid_valid->op, POLY_OP_CONST);
  ASSERT_TRUE(invalid_valid->arg.kind == POLY_ARG_BOOL && !invalid_valid->arg.b);

  PolyUOp *stack_idx = poly_index_get_idx(ctx, stack);
  ASSERT_INT_EQ(stack_idx->op, POLY_OP_STACK);
  ASSERT_INT_EQ(stack_idx->n_src, 2);
  ASSERT_PTR_EQ(stack_idx->src[0], idx0);
  ASSERT_PTR_EQ(stack_idx->src[1], idx1);
  ASSERT_TRUE(poly_dtype_eq(stack_idx->dtype, poly_dtype_vec(POLY_INDEX, 2)));
  PolyUOp *stack_valid = poly_index_get_valid(ctx, stack);
  ASSERT_INT_EQ(stack_valid->op, POLY_OP_STACK);
  ASSERT_INT_EQ(stack_valid->n_src, 2);
  ASSERT_PTR_EQ(stack_valid->src[0], gate);
  ASSERT_TRUE(stack_valid->src[1]->arg.kind == POLY_ARG_BOOL && stack_valid->src[1]->arg.b);
  ASSERT_TRUE(poly_dtype_eq(stack_valid->dtype, poly_dtype_vec(POLY_BOOL, 2)));

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, flat_index_lifts_all_invalid_guards_like_tinygrad) {
  /* Pinned probe temp/tg_flat_invalid_probe_20260809.py and
   * uop/symbolic.py:60-86 produce
   * WHERE(g0 AND g1, r0*32 + r1, Invalid). */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(32));
  PolyUOp *r0 = poly_uop1(ctx, POLY_OP_RANGE, POLY_INDEX, bound, poly_arg_range(0, POLY_AXIS_LOOP));
  PolyUOp *r1 = poly_uop1(ctx, POLY_OP_RANGE, POLY_INDEX, bound, poly_arg_range(1, POLY_AXIS_LOOP));
  PolyUOp *g0 = poly_uop0(ctx, POLY_OP_PARAM, POLY_BOOL, poly_arg_int(0));
  PolyUOp *g1 = poly_uop0(ctx, POLY_OP_PARAM, POLY_BOOL, poly_arg_int(1));
  PolyUOp *invalid = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_invalid());
  PolyUOp *coords[2] = {
      poly_uop3(ctx, POLY_OP_WHERE, POLY_INDEX, g0, r0, invalid, poly_arg_none()),
      poly_uop3(ctx, POLY_OP_WHERE, POLY_INDEX, g1, r1, invalid, poly_arg_none()),
  };
  PolyUOp *bounds[2] = {bound, bound};

  int64_t dims[2] = {32, 32};
  PolyShape shape = {.dims = dims, .ndim = 2};
  PolyUOp *flats[2] = {
      poly_compute_flat_index(ctx, coords, 2, shape),
      poly_compute_flat_index_symbolic(ctx, coords, bounds, 2),
  };
  for (int i = 0; i < 2; i++) {
    PolyUOp *flat = flats[i];
    ASSERT_NOT_NULL(flat);
    ASSERT_INT_EQ(flat->op, POLY_OP_WHERE);
    ASSERT_INT_EQ(flat->n_src, 3);
    ASSERT_INT_EQ(flat->src[0]->op, POLY_OP_AND);
    ASSERT_TRUE(
        (flat->src[0]->src[0] == g0 && flat->src[0]->src[1] == g1) ||
        (flat->src[0]->src[0] == g1 && flat->src[0]->src[1] == g0)
    );
    ASSERT_INT_EQ(flat->src[1]->op, POLY_OP_ADD);
    ASSERT_INT_EQ(flat->src[2]->op, POLY_OP_CONST);
    ASSERT_INT_EQ(flat->src[2]->arg.kind, POLY_ARG_INVALID);
  }

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, hlb_reflect_input_index_reenters_symbolic_after_bufferize_removal) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  /* Exact pinned HLB reflect construction, examples/hlb_cifar10.py:189-192.
   * Pinned get_kernel_graph runs symbolic and remove_bufferize in one rewrite
   * fixed point (schedule/rangeify.py:598-607), so the substituted flat input
   * coordinate is canonicalized to WHERE(g0 AND g1, flat, Invalid). */
  PolyUOp *input =
      poly_reshape(ctx, poly_buffer_f32(ctx, 32 * 3 * 32 * 32), (int64_t[]){32, 3, 32, 32}, 4);
  ASSERT_NOT_NULL(input);

  PolyUOp *left = poly_flip(
      ctx, poly_shrink(ctx, input, (int64_t[][2]){{0, 32}, {0, 3}, {0, 32}, {1, 3}}, 4),
      (int64_t[]){3}, 1
  );
  PolyUOp *right = poly_flip(
      ctx, poly_shrink(ctx, input, (int64_t[][2]){{0, 32}, {0, 3}, {0, 32}, {29, 31}}, 4),
      (int64_t[]){3}, 1
  );
  PolyUOp *wide_parts[3] = {left, input, right};
  PolyUOp *wide = poly_cat(ctx, wide_parts, 3, 3);
  ASSERT_NOT_NULL(wide);

  PolyUOp *top = poly_flip(
      ctx, poly_shrink(ctx, wide, (int64_t[][2]){{0, 32}, {0, 3}, {1, 3}, {0, 36}}, 4),
      (int64_t[]){2}, 1
  );
  PolyUOp *bottom = poly_flip(
      ctx, poly_shrink(ctx, wide, (int64_t[][2]){{0, 32}, {0, 3}, {29, 31}, {0, 36}}, 4),
      (int64_t[]){2}, 1
  );
  PolyUOp *tall_parts[3] = {top, wide, bottom};
  PolyUOp *padded = poly_cat(ctx, tall_parts, 3, 2);
  ASSERT_NOT_NULL(padded);

  PolyUOp *realized = NULL;
  PolySchedule *schedule = poly_schedule_with_vars(ctx, &padded, 1, &realized);
  ASSERT_NOT_NULL(schedule);
  ASSERT_INT_EQ(schedule->template->n_calls, 1);
  PolyUOp *body = poly_schedule_call_body(schedule, 0);
  ASSERT_NOT_NULL(body);

  int n_body = 0;
  PolyUOp **body_topo = poly_toposort_alloc(ctx, body, &n_body);
  int invalid_indexes = 0;
  for (int i = 0; i < n_body; i++) {
    PolyUOp *index = body_topo[i];
    if (index->op != POLY_OP_INDEX || index->n_src != 2) continue;
    PolyUOp *coord = index->src[1];
    int n_coord = 0;
    PolyUOp **coord_topo = poly_toposort_alloc(ctx, coord, &n_coord);
    bool has_invalid = false;
    for (int j = 0; j < n_coord; j++)
      has_invalid |=
          coord_topo[j]->op == POLY_OP_CONST && coord_topo[j]->arg.kind == POLY_ARG_INVALID;
    poly_toposort_free(coord_topo);
    if (!has_invalid) continue;
    invalid_indexes++;
    ASSERT_INT_EQ(coord->op, POLY_OP_WHERE);
    ASSERT_INT_EQ(coord->n_src, 3);
    ASSERT_INT_EQ(coord->src[0]->op, POLY_OP_AND);
    ASSERT_INT_EQ(coord->src[2]->op, POLY_OP_CONST);
    ASSERT_INT_EQ(coord->src[2]->arg.kind, POLY_ARG_INVALID);
  }
  poly_toposort_free(body_topo);
  ASSERT_TRUE(invalid_indexes > 0);

  poly_schedule_free(schedule);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, apply_elementwise_passthrough) {
  /* a + b → STORE → SINK: ALU ops pass through unchanged */
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *b = poly_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, a, b, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out, add, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyIndexingCtx *ictx = poly_indexing_ctx_new(ctx);
  PolyUOp *result = run_apply_rangeify(ictx, sink);
  ASSERT_NOT_NULL(result);

  /* ADD, STORE, SINK all still present */
  ASSERT_INT_EQ(count_ops(ctx, result, POLY_OP_ADD), 1);
  ASSERT_INT_EQ(count_ops(ctx, result, POLY_OP_STORE), 1);
  ASSERT_INT_EQ(count_ops(ctx, result, POLY_OP_SINK), 1);

  /* No BUFFERIZE (single kernel, nothing needs intermediate buffer) */
  ASSERT_INT_EQ(count_ops(ctx, result, POLY_OP_STAGE), 0);

  poly_indexing_ctx_destroy(ictx);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, apply_const_passthrough) {
  /* const(1.0) + a → STORE → SINK: CONST passes through */
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *c = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(1.0));
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, a, c, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out, add, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyIndexingCtx *ictx = poly_indexing_ctx_new(ctx);
  PolyUOp *result = run_apply_rangeify(ictx, sink);
  ASSERT_NOT_NULL(result);

  /* CONST still present */
  ASSERT_TRUE(count_ops(ctx, result, POLY_OP_CONST) >= 1);

  poly_indexing_ctx_destroy(ictx);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, apply_realized_bufferize) {
  /* Two stores consuming same computed value with different ranges:
   * a → add(a,b) → STORE(out1, add)
   * a → mul(a,c) → STORE(out2, mul)
   * After range_prop, a is realized → but a is a BUFFER (already in memory),
   * so BUFFERIZE is not needed. Verify no BUFFERIZE for BUFFER nodes. */
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *b = poly_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *c = poly_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *out1 = poly_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *out2 = poly_buffer(ctx, POLY_FLOAT32, 10);

  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, a, b, poly_arg_none());
  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, a, c, poly_arg_none());
  PolyUOp *s1 = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out1, add, poly_arg_none());
  PolyUOp *s2 = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out2, mul, poly_arg_none());

  PolyUOp *stores[] = {s1, s2};
  PolyUOp *sink = poly_uop(ctx, POLY_OP_SINK, POLY_VOID, stores, 2, poly_arg_none());

  PolyIndexingCtx *ictx = poly_indexing_ctx_new(ctx);
  PolyUOp *result = run_apply_rangeify(ictx, sink);
  ASSERT_NOT_NULL(result);

  /* a is realized (different consumer ranges) */
  ASSERT_TRUE(poly_is_realized(ictx, a));

  /* But a is a BUFFER → no BUFFERIZE needed (already in memory) */
  ASSERT_INT_EQ(count_ops(ctx, result, POLY_OP_STAGE), 0);

  /* Both stores still present */
  ASSERT_INT_EQ(count_ops(ctx, result, POLY_OP_STORE), 2);

  poly_indexing_ctx_destroy(ictx);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, apply_vecadd_kernel_mode_read_indices_are_scalar) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, 4);
  PolyUOp *b = poly_buffer(ctx, POLY_FLOAT32, 4);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, 4);
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, a, b, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out, add, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyIndexingCtx *ictx = poly_indexing_ctx_new(ctx);
  ictx->add_buffer_indices = true;
  PolyUOp *result = run_apply_rangeify(ictx, sink);
  ASSERT_NOT_NULL(result);

  IndexKindCounts counts = count_index_kinds(ctx, result);
  ASSERT_INT_EQ(counts.store_target_index, 1);
  ASSERT_INT_EQ(counts.target_ptr_index, 1);
  ASSERT_INT_EQ(counts.read_ptr_index, 0);
  ASSERT_INT_EQ(counts.read_scalar_index, 2);

  poly_indexing_ctx_destroy(ictx);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, apply_triu_kernel_mode_read_indices_are_scalar) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *in = poly_buffer_f32(ctx, 9);
  PolyUOp *in2d = poly_reshape(ctx, in, (int64_t[]){3, 3}, 2);
  PolyUOp *tri = poly_triu(ctx, in2d, 0);
  PolyUOp *out = poly_buffer_f32(ctx, 9);
  PolyUOp *store = poly_store_val(ctx, out, tri);
  PolyUOp *sink = poly_sink1(ctx, store);

  PolyIndexingCtx *ictx = poly_indexing_ctx_new(ctx);
  ictx->add_buffer_indices = true;
  PolyUOp *result = run_apply_rangeify(ictx, sink);
  ASSERT_NOT_NULL(result);

  IndexKindCounts counts = count_index_kinds(ctx, result);
  ASSERT_INT_EQ(counts.store_target_index, 1);
  ASSERT_INT_EQ(counts.target_ptr_index, 1);
  ASSERT_INT_EQ(counts.read_ptr_index, 0);
  ASSERT_TRUE(counts.read_scalar_index >= 1);

  poly_indexing_ctx_destroy(ictx);
  poly_ctx_destroy(ctx);
  PASS();
}

/* schedule_v2 helper */

static int count_lin_ops(PolyUOp **lin, int n, PolyOps op) {
  int c = 0;
  for (int i = 0; i < n; i++)
    if (lin[i]->op == op) c++;
  return c;
}

/* schedule_v2 IR tests */

TEST(rangeify, schedule_v2_vecadd_ir) {
  /* c = a + b (1D, 10 elements): verify kernel IR structure matches v1 */
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *b = poly_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *c = poly_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, a, b, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, c, add, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyKernelScheduleResult sr = poly_build_kernel_schedule(ctx, sink);
  ASSERT_INT_EQ(sr.n_kernels, 1);
  ASSERT_NOT_NULL(sr.kernels[0]);
  ASSERT_EQ(sr.kernels[0]->op, POLY_OP_SINK);

  int n;
  PolyUOp **lin = poly_linearize(ctx, sr.kernels[0], &n);
  ASSERT_TRUE(n > 0);

  /* Same structure as v1: 3 PARAMs, 1 RANGE, 2 LOADs, 1 ADD, 1 STORE, 1 END */
  ASSERT_INT_EQ(count_lin_ops(lin, n, POLY_OP_PARAM), 3);
  ASSERT_INT_EQ(count_lin_ops(lin, n, POLY_OP_RANGE), 1);
  ASSERT_INT_EQ(count_lin_ops(lin, n, POLY_OP_LOAD), 2);
  ASSERT_INT_EQ(count_lin_ops(lin, n, POLY_OP_ADD), 1);
  ASSERT_INT_EQ(count_lin_ops(lin, n, POLY_OP_STORE), 1);
  ASSERT_INT_EQ(count_lin_ops(lin, n, POLY_OP_END), 1);

  free(lin);
  poly_kernel_schedule_result_free(&sr);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, schedule_v2_buffer_view_is_one_global_param) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *arena = poly_buffer(ctx, POLY_UINT8, 256);
  PolyUOp *old_buffer = poly_buffer(ctx, POLY_FLOAT32, 2);
  int64_t view_args[2] = {2, 16};
  PolyUOp *view_src[2] = {arena, old_buffer};
  PolyUOp *view = poly_uop(
      ctx, POLY_OP_BUFFER_VIEW, POLY_FLOAT32, view_src, 2,
      (PolyArg){.kind = POLY_ARG_INT_TUPLE, .int_tuple = {view_args, 2}}
  );
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, 2);
  PolyUOp *add = poly_alu2(ctx, POLY_OP_ADD, view, poly_const_float(ctx, 1.0));
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, add));

  PolyKernelScheduleResult sr = poly_build_kernel_schedule(ctx, sink);
  ASSERT_INT_EQ(sr.n_kernels, 1);
  ASSERT_INT_EQ(sr.kernel_n_params[0], 2);
  ASSERT_PTR_EQ(sr.param_to_buf[0][0], out);
  PolyUOp *scheduled_view = sr.param_to_buf[0][1];
  ASSERT_NOT_NULL(scheduled_view);
  ASSERT_EQ(scheduled_view->op, POLY_OP_BUFFER_VIEW);
  ASSERT_EQ(scheduled_view->arg.kind, POLY_ARG_INT_TUPLE);
  ASSERT_INT_EQ(scheduled_view->arg.int_tuple.n, 2);
  ASSERT_INT_EQ(scheduled_view->arg.int_tuple.vals[0], 2);
  ASSERT_INT_EQ(scheduled_view->arg.int_tuple.vals[1], 16);
  ASSERT_INT_EQ(scheduled_view->n_src, 2);
  ASSERT_EQ(scheduled_view->src[0]->op, POLY_OP_BUFFER);
  ASSERT_TRUE(poly_dtype_eq(scheduled_view->src[0]->dtype, POLY_UINT8));
  ASSERT_EQ(scheduled_view->src[0]->arg.kind, POLY_ARG_INT);
  ASSERT_INT_EQ(scheduled_view->src[0]->arg.i, 256);
  ASSERT_INT_EQ(count_ops(ctx, sr.kernels[0], POLY_OP_BUFFER_VIEW), 0);
  ASSERT_INT_EQ(count_ops(ctx, sr.kernels[0], POLY_OP_PARAM), 2);

  poly_kernel_schedule_result_free(&sr);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, schedule_v2_chain_ir) {
  /* d = (a + b) * c: single kernel, 4 PARAMs.
   *
   * On the optimized default CPU path, tinygrad upcasts this 8-wide chain into
   * one float4 loop. The linearized IR keeps one outer RANGE and one final
   * VECTORIZE store, but scalarizes the four lanes into 4 ADD + 4 MUL nodes. */
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, 8);
  PolyUOp *b = poly_buffer(ctx, POLY_FLOAT32, 8);
  PolyUOp *c = poly_buffer(ctx, POLY_FLOAT32, 8);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, 8);
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, a, b, poly_arg_none());
  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, add, c, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out, mul, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyKernelScheduleResult sr = poly_build_kernel_schedule(ctx, sink);
  ASSERT_INT_EQ(sr.n_kernels, 1);

  int n;
  PolyUOp **lin = poly_linearize(ctx, sr.kernels[0], &n);
  ASSERT_TRUE(n > 0);

  ASSERT_INT_EQ(count_lin_ops(lin, n, POLY_OP_PARAM), 4);
  ASSERT_INT_EQ(count_lin_ops(lin, n, POLY_OP_RANGE), 1);
  ASSERT_INT_EQ(count_lin_ops(lin, n, POLY_OP_LOAD), 3);
  ASSERT_INT_EQ(count_lin_ops(lin, n, POLY_OP_ADD), 4);
  ASSERT_INT_EQ(count_lin_ops(lin, n, POLY_OP_MUL), 4);
  ASSERT_INT_EQ(count_lin_ops(lin, n, POLY_OP_STORE), 1);
  ASSERT_INT_EQ(count_lin_ops(lin, n, POLY_OP_END), 1);
  ASSERT_INT_EQ(count_lin_ops(lin, n, POLY_OP_VECTORIZE), 1);

  free(lin);
  poly_kernel_schedule_result_free(&sr);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, schedule_v2_reduce_ir) {
  /* sum(a) where a is 10 elements: verify accumulator pattern */
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *c = poly_buffer(ctx, POLY_FLOAT32, 1);
  int64_t axes[] = {0};
  PolyUOp *sum = poly_reduce_axis(ctx, POLY_OP_ADD, a, axes, 1);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, c, sum, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyKernelScheduleResult sr = poly_build_kernel_schedule(ctx, sink);
  ASSERT_INT_EQ(sr.n_kernels, 1);

  int n;
  PolyUOp **lin = poly_linearize(ctx, sr.kernels[0], &n);
  ASSERT_TRUE(n > 0);

  /* Tinygrad parity on the optimized CPU path:
   * the scheduled kernel root still has one RANGE, but the final linearized
   * kernel is scalarized with PARAM/LOAD/GEP/ADD and no DEFINE_REG/RANGE. */
  ASSERT_INT_EQ(count_lin_ops(lin, n, POLY_OP_PARAM), 2);
  ASSERT_INT_EQ(count_lin_ops(lin, n, POLY_OP_DEFINE_REG), 0);
  ASSERT_INT_EQ(count_lin_ops(lin, n, POLY_OP_RANGE), 0);
  ASSERT_INT_EQ(count_lin_ops(lin, n, POLY_OP_LOAD), 3);
  ASSERT_INT_EQ(count_lin_ops(lin, n, POLY_OP_ADD), 9);
  ASSERT_INT_EQ(count_lin_ops(lin, n, POLY_OP_STORE), 1);
  ASSERT_INT_EQ(count_lin_ops(lin, n, POLY_OP_END), 0);

  free(lin);
  poly_kernel_schedule_result_free(&sr);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, schedule_v2_vecadd_e2e) {
  /* c = a + b: full compile + execute through v2 */
  int N = 16;
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *b = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *c = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, a, b, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, c, add, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyKernelScheduleResult sr = poly_build_kernel_schedule(ctx, sink);
  ASSERT_INT_EQ(sr.n_kernels, 1);

  int n_lin;
  PolyUOp **lin = poly_linearize(ctx, sr.kernels[0], &n_lin);
  char *src = poly_render_c(lin, n_lin, "v2_vecadd");
  ASSERT_NOT_NULL(src);

  PolyProgram *prog = poly_compile_c(src, "v2_vecadd");
  ASSERT_NOT_NULL(prog);

  float a_d[16], b_d[16], c_d[16];
  for (int i = 0; i < N; i++) {
    a_d[i] = (float)i;
    b_d[i] = (float)(i * 10);
  }

  void *args[3] = {c_d, a_d, b_d};
  poly_program_call(prog, args, 3);

  for (int i = 0; i < N; i++)
    ASSERT_FLOAT_EQ(c_d[i], a_d[i] + b_d[i], 1e-5);

  poly_program_destroy(prog);
  free(src);
  free(lin);
  poly_kernel_schedule_result_free(&sr);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, schedule_v2_reduce_chain_ir) {
  /* sum(a) + b: two loop nests in one kernel (reduce + elementwise) */
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *b = poly_buffer(ctx, POLY_FLOAT32, 1);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, 1);
  int64_t axes[] = {0};
  PolyUOp *sum = poly_reduce_axis(ctx, POLY_OP_ADD, a, axes, 1);
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, sum, b, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out, add, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyKernelScheduleResult sr = poly_build_kernel_schedule(ctx, sink);
  ASSERT_INT_EQ(sr.n_kernels, 1);

  int n;
  PolyUOp **lin = poly_linearize(ctx, sr.kernels[0], &n);
  ASSERT_TRUE(n > 0);

  /* Tinygrad parity on the optimized CPU path:
   * the fused scalar chain is fully scalarized after rewrite/linearize. */
  ASSERT_INT_EQ(count_lin_ops(lin, n, POLY_OP_PARAM), 3);
  ASSERT_INT_EQ(count_lin_ops(lin, n, POLY_OP_DEFINE_REG), 0);
  ASSERT_INT_EQ(count_lin_ops(lin, n, POLY_OP_RANGE), 0);
  ASSERT_INT_EQ(count_lin_ops(lin, n, POLY_OP_LOAD), 4);
  ASSERT_INT_EQ(count_lin_ops(lin, n, POLY_OP_ADD), 10);
  ASSERT_INT_EQ(count_lin_ops(lin, n, POLY_OP_STORE), 1);
  ASSERT_INT_EQ(count_lin_ops(lin, n, POLY_OP_END), 0);

  free(lin);
  poly_kernel_schedule_result_free(&sr);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, schedule_v2_reduce_chain_e2e) {
  /* sum([1..10]) + 5.0 = 60.0 */
  int N = 10;
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *b = poly_buffer(ctx, POLY_FLOAT32, 1);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, 1);
  int64_t axes[] = {0};
  PolyUOp *sum = poly_reduce_axis(ctx, POLY_OP_ADD, a, axes, 1);
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, sum, b, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out, add, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyKernelScheduleResult sr = poly_build_kernel_schedule(ctx, sink);
  ASSERT_INT_EQ(sr.n_kernels, 1);

  int n_lin;
  PolyUOp **lin = poly_linearize(ctx, sr.kernels[0], &n_lin);
  char *src = poly_render_c(lin, n_lin, "v2_sum_add");
  ASSERT_NOT_NULL(src);

  PolyProgram *prog = poly_compile_c(src, "v2_sum_add");
  ASSERT_NOT_NULL(prog);

  float a_d[10], b_d[1] = {5.0f}, out_d[1] = {0.0f};
  float expected = 5.0f;
  for (int i = 0; i < N; i++) {
    a_d[i] = (float)(i + 1);
    expected += a_d[i];
  }

  void *args[3] = {out_d, a_d, b_d};
  poly_program_call(prog, args, 3);
  ASSERT_FLOAT_EQ(out_d[0], expected, 1e-4);

  poly_program_destroy(prog);
  free(src);
  free(lin);
  poly_kernel_schedule_result_free(&sr);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, schedule_v2_reduce_e2e) {
  /* sum([1..10]) = 55: full compile + execute through v2 */
  int N = 10;
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *c = poly_buffer(ctx, POLY_FLOAT32, 1);
  int64_t axes[] = {0};
  PolyUOp *sum = poly_reduce_axis(ctx, POLY_OP_ADD, a, axes, 1);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, c, sum, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyKernelScheduleResult sr = poly_build_kernel_schedule(ctx, sink);
  ASSERT_INT_EQ(sr.n_kernels, 1);

  int n_lin;
  PolyUOp **lin = poly_linearize(ctx, sr.kernels[0], &n_lin);
  char *src = poly_render_c(lin, n_lin, "v2_sum");
  ASSERT_NOT_NULL(src);

  PolyProgram *prog = poly_compile_c(src, "v2_sum");
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
  poly_kernel_schedule_result_free(&sr);
  poly_ctx_destroy(ctx);
  PASS();
}

/* Parity helper: compare v1 and v2 numerical output */

static bool run_via_v1(PolyCtx *ctx, PolyUOp *sink, void **args, int n_args, const char *name) {
  PolySchedule *schedule = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  if (!schedule) return false;
  if (schedule->template->n_calls != 1 || !poly_schedule_call_body(schedule, 0)) {
    poly_schedule_free(schedule);
    return false;
  }
  PolyUOp *kernel = poly_schedule_call_body(schedule, 0);
  poly_schedule_free(schedule);
  int n_lin;
  PolyUOp **lin = poly_linearize(ctx, kernel, &n_lin);
  if (!lin || n_lin == 0) return false;
  char *src = poly_render_c(lin, n_lin, name);
  free(lin);
  if (!src) return false;
  PolyProgram *prog = poly_compile_c(src, name);
  free(src);
  if (!prog) return false;
  poly_program_call(prog, args, n_args);
  poly_program_destroy(prog);
  return true;
}

static bool run_via_v2(PolyCtx *ctx, PolyUOp *sink, void **args, int n_args, const char *name) {
  PolyKernelScheduleResult sr = poly_build_kernel_schedule(ctx, sink);
  if (sr.n_kernels != 1 || !sr.kernels[0]) {
    poly_kernel_schedule_result_free(&sr);
    return false;
  }
  int n_lin;
  PolyUOp **lin = poly_linearize(ctx, sr.kernels[0], &n_lin);
  poly_kernel_schedule_result_free(&sr);
  if (!lin || n_lin == 0) return false;
  char *src = poly_render_c(lin, n_lin, name);
  free(lin);
  if (!src) return false;
  PolyProgram *prog = poly_compile_c(src, name);
  free(src);
  if (!prog) return false;
  poly_program_call(prog, args, n_args);
  poly_program_destroy(prog);
  return true;
}

TEST(rangeify, schedule_v2_parity_vecadd) {
  /* Compare v1 vs v2 for a+b */
  int N = 16;
  float a[16], b[16], c_v1[16], c_v2[16];
  for (int i = 0; i < N; i++) {
    a[i] = (float)i * 0.5f;
    b[i] = (float)(N - i);
  }

  {
    PolyCtx *ctx = poly_ctx_new();
    PolyUOp *ua = poly_buffer(ctx, POLY_FLOAT32, N);
    PolyUOp *ub = poly_buffer(ctx, POLY_FLOAT32, N);
    PolyUOp *uc = poly_buffer(ctx, POLY_FLOAT32, N);
    PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, ua, ub, poly_arg_none());
    PolyUOp *st = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, uc, add, poly_arg_none());
    PolyUOp *sk = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, st, poly_arg_none());
    void *args[3] = {c_v1, a, b};
    ASSERT_TRUE(run_via_v1(ctx, sk, args, 3, "par_v1_add"));
    poly_ctx_destroy(ctx);
  }
  {
    PolyCtx *ctx = poly_ctx_new();
    PolyUOp *ua = poly_buffer(ctx, POLY_FLOAT32, N);
    PolyUOp *ub = poly_buffer(ctx, POLY_FLOAT32, N);
    PolyUOp *uc = poly_buffer(ctx, POLY_FLOAT32, N);
    PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, ua, ub, poly_arg_none());
    PolyUOp *st = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, uc, add, poly_arg_none());
    PolyUOp *sk = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, st, poly_arg_none());
    void *args[3] = {c_v2, a, b};
    ASSERT_TRUE(run_via_v2(ctx, sk, args, 3, "par_v2_add"));
    poly_ctx_destroy(ctx);
  }

  for (int i = 0; i < N; i++)
    ASSERT_FLOAT_EQ(c_v1[i], c_v2[i], 1e-6);
  PASS();
}

TEST(rangeify, schedule_v2_parity_reduce) {
  /* Compare v1 vs v2 for sum(a) */
  int N = 10;
  float a[10];
  float c_v1[1] = {0}, c_v2[1] = {0};
  for (int i = 0; i < N; i++)
    a[i] = (float)(i + 1) * 1.5f;

  {
    PolyCtx *ctx = poly_ctx_new();
    PolyUOp *ua = poly_buffer(ctx, POLY_FLOAT32, N);
    PolyUOp *uc = poly_buffer(ctx, POLY_FLOAT32, 1);
    int64_t axes[] = {0};
    PolyUOp *sum = poly_reduce_axis(ctx, POLY_OP_ADD, ua, axes, 1);
    PolyUOp *st = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, uc, sum, poly_arg_none());
    PolyUOp *sk = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, st, poly_arg_none());
    void *args[2] = {c_v1, a};
    ASSERT_TRUE(run_via_v1(ctx, sk, args, 2, "par_v1_sum"));
    poly_ctx_destroy(ctx);
  }
  {
    PolyCtx *ctx = poly_ctx_new();
    PolyUOp *ua = poly_buffer(ctx, POLY_FLOAT32, N);
    PolyUOp *uc = poly_buffer(ctx, POLY_FLOAT32, 1);
    int64_t axes[] = {0};
    PolyUOp *sum = poly_reduce_axis(ctx, POLY_OP_ADD, ua, axes, 1);
    PolyUOp *st = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, uc, sum, poly_arg_none());
    PolyUOp *sk = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, st, poly_arg_none());
    void *args[2] = {c_v2, a};
    ASSERT_TRUE(run_via_v2(ctx, sk, args, 2, "par_v2_sum"));
    poly_ctx_destroy(ctx);
  }

  ASSERT_FLOAT_EQ(c_v1[0], c_v2[0], 1e-4);
  PASS();
}

/* Multi-kernel tests */

TEST(rangeify, multi_kernel_shared_computed_ir) {
  /* d = neg(a), used by two stores:
   *   STORE(out1, d + b)
   *   STORE(out2, d * c)
   * pm_remove_bufferize inlines d (single BUFFER, no reduce) → 2 kernels. */
  int N = 8;
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *b = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *c = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *out1 = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *out2 = poly_buffer(ctx, POLY_FLOAT32, N);

  PolyUOp *d = poly_uop1(ctx, POLY_OP_NEG, POLY_FLOAT32, a, poly_arg_none());
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, d, b, poly_arg_none());
  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, d, c, poly_arg_none());
  PolyUOp *s1 = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out1, add, poly_arg_none());
  PolyUOp *s2 = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out2, mul, poly_arg_none());
  PolyUOp *stores[] = {s1, s2};
  PolyUOp *sink = poly_uop(ctx, POLY_OP_SINK, POLY_VOID, stores, 2, poly_arg_none());

  PolyKernelScheduleResult sr = poly_build_kernel_schedule(ctx, sink);
  ASSERT_INT_EQ(sr.n_kernels, 2); /* 2 fused kernels (d inlined) */
  ASSERT_INT_EQ(sr.n_intermediates, 0);

  /* All kernels should have SINK roots */
  for (int k = 0; k < sr.n_kernels; k++)
    ASSERT_EQ(sr.kernels[k]->op, POLY_OP_SINK);

  poly_kernel_schedule_result_free(&sr);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, multi_kernel_shared_computed_e2e) {
  /* Same pattern as IR test, full compile + execute via poly_test_realize_buffer_views().
   * d = neg(a)
   * out1 = d + b = -a + b
   * out2 = d * c = -a * c */
  int N = 8;
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *b = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *c = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *out1 = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *out2 = poly_buffer(ctx, POLY_FLOAT32, N);

  PolyUOp *d = poly_uop1(ctx, POLY_OP_NEG, POLY_FLOAT32, a, poly_arg_none());
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, d, b, poly_arg_none());
  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, d, c, poly_arg_none());
  PolyUOp *s1 = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out1, add, poly_arg_none());
  PolyUOp *s2 = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out2, mul, poly_arg_none());
  PolyUOp *stores[] = {s1, s2};
  PolyUOp *sink = poly_uop(ctx, POLY_OP_SINK, POLY_VOID, stores, 2, poly_arg_none());

  float a_d[8], b_d[8], c_d[8], o1_d[8], o2_d[8];
  for (int i = 0; i < N; i++) {
    a_d[i] = (float)(i + 1); /* 1..8 */
    b_d[i] = (float)(i * 10); /* 0,10,20,...,70 */
    c_d[i] = (float)(i + 1) * 0.5f; /* 0.5,1.0,...,4.0 */
  }

  PolyTestBufferView bindings[] = {
      POLY_TEST_HOST_VIEW(out1, o1_d), POLY_TEST_HOST_VIEW(out2, o2_d), POLY_TEST_HOST_VIEW(a, a_d),
      POLY_TEST_HOST_VIEW(b, b_d),     POLY_TEST_HOST_VIEW(c, c_d),
  };
  int ret = poly_test_realize_buffer_views(ctx, sink, bindings, 5);
  ASSERT_INT_EQ(ret, 0);

  for (int i = 0; i < N; i++) {
    float neg_a = -a_d[i];
    ASSERT_FLOAT_EQ(o1_d[i], neg_a + b_d[i], 1e-5);
    ASSERT_FLOAT_EQ(o2_d[i], neg_a * c_d[i], 1e-5);
  }

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, schedule_v2_switchover_parity) {
  /* Verify that poly_test_realize_buffer_views() produces correct results
   * for vecadd — the most basic pattern. */
  int N = 16;
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *b = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, a, b, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out, add, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  float a_d[16], b_d[16], o_d[16];
  for (int i = 0; i < N; i++) {
    a_d[i] = (float)i;
    b_d[i] = (float)(N - i);
  }

  PolyTestBufferView bindings[] = {
      POLY_TEST_HOST_VIEW(out, o_d),
      POLY_TEST_HOST_VIEW(a, a_d),
      POLY_TEST_HOST_VIEW(b, b_d),
  };
  int ret = poly_test_realize_buffer_views(ctx, sink, bindings, 3);
  ASSERT_INT_EQ(ret, 0);

  for (int i = 0; i < N; i++)
    ASSERT_FLOAT_EQ(o_d[i], a_d[i] + b_d[i], 1e-5);

  poly_ctx_destroy(ctx);
  PASS();
}

/* Rangeify stats tests */

static bool run_buffer_alt_stats_case(PolyRangeifyStats *out_stats) {
  PolyCtx *ctx = poly_ctx_new();
  if (!ctx) return false;

  PolyUOp *x = poly_buffer(ctx, POLY_FLOAT32, 32);
  if (!x) {
    poly_ctx_destroy(ctx);
    return false;
  }

  PolyUOp *terms[9];
  for (int i = 0; i < 9; i++) {
    int64_t pairs[][2] = {{i, i + 4}};
    terms[i] = poly_shrink(ctx, x, pairs, 1);
    if (!terms[i]) {
      poly_ctx_destroy(ctx);
      return false;
    }
  }

  PolyUOp *acc = terms[0];
  for (int i = 1; i < 9; i++) {
    acc = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, acc, terms[i], poly_arg_none());
    if (!acc) {
      poly_ctx_destroy(ctx);
      return false;
    }
  }

  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, 4);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out, acc, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());
  if (!out || !store || !sink) {
    poly_ctx_destroy(ctx);
    return false;
  }

  PolyIndexingCtx *ictx = poly_indexing_ctx_new(ctx);
  if (!ictx) {
    poly_ctx_destroy(ctx);
    return false;
  }

  poly_rangeify_stats_reset();
  PolyUOp *result = run_apply_rangeify(ictx, sink);
  if (out_stats) *out_stats = poly_rangeify_stats_get();

  poly_indexing_ctx_destroy(ictx);
  poly_ctx_destroy(ctx);
  return result != NULL;
}

typedef struct {
  PolyRangeifyStats before;
  PolyRangeifyStats after;
  bool ok;
} RangeifyStatsThreadResult;

static void *rangeify_stats_thread_main(void *opaque) {
  RangeifyStatsThreadResult *result = opaque;
  result->before = poly_rangeify_stats_get();
  result->ok = run_buffer_alt_stats_case(&result->after);
  return NULL;
}

TEST(rangeify, stats_clean_vecadd_no_workarounds) {
  int N = 16;
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *b = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, a, b, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out, add, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  float a_d[16], b_d[16], o_d[16];
  for (int i = 0; i < N; i++) {
    a_d[i] = (float)i;
    b_d[i] = (float)(N - i);
  }

  PolyTestBufferView bindings[] = {
      POLY_TEST_HOST_VIEW(out, o_d),
      POLY_TEST_HOST_VIEW(a, a_d),
      POLY_TEST_HOST_VIEW(b, b_d),
  };

  poly_rangeify_stats_reset();
  int ret = poly_test_realize_buffer_views(ctx, sink, bindings, 3);
  ASSERT_INT_EQ(ret, 0);

  PolyRangeifyStats stats = poly_rangeify_stats_get();
  ASSERT_INT_EQ(stats.remap_calls, 0);
  ASSERT_INT_EQ(stats.remap_id_matches, 0);
  ASSERT_INT_EQ(stats.remap_pos_matches, 0);
  ASSERT_INT_EQ(stats.remap_unique_bound_matches, 0);
  ASSERT_INT_EQ(stats.remap_bound_matches, 0);
  ASSERT_INT_EQ(stats.remap_failures, 0);
  ASSERT_INT_EQ(stats.orphan_top_level_hits, 0);
  ASSERT_INT_EQ(stats.deep_orphan_hits, 0);
  ASSERT_INT_EQ(stats.buffer_alt_created, 0);
  ASSERT_INT_EQ(stats.buffer_alt_used, 0);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, stats_clean_reduce_no_workarounds) {
  int R = 4, C = 3;
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a_flat = poly_buffer(ctx, POLY_FLOAT32, R * C);
  int64_t dims[] = {R, C};
  PolyUOp *a = poly_reshape(ctx, a_flat, dims, 2);
  int64_t axes[] = {1};
  PolyUOp *sum = poly_reduce_axis(ctx, POLY_OP_ADD, a, axes, 1);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, R);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out, sum, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  float a_d[12], o_d[4];
  for (int i = 0; i < R * C; i++)
    a_d[i] = (float)(i + 1);

  PolyTestBufferView bindings[] = {
      POLY_TEST_HOST_VIEW(out, o_d),
      POLY_TEST_HOST_VIEW(a_flat, a_d),
  };

  poly_rangeify_stats_reset();
  int ret = poly_test_realize_buffer_views(ctx, sink, bindings, 2);
  ASSERT_INT_EQ(ret, 0);

  PolyRangeifyStats stats = poly_rangeify_stats_get();
  ASSERT_INT_EQ(stats.remap_calls, 0);
  ASSERT_INT_EQ(stats.remap_id_matches, 0);
  ASSERT_INT_EQ(stats.remap_pos_matches, 0);
  ASSERT_INT_EQ(stats.remap_unique_bound_matches, 0);
  ASSERT_INT_EQ(stats.remap_bound_matches, 0);
  ASSERT_INT_EQ(stats.remap_failures, 0);
  ASSERT_INT_EQ(stats.orphan_top_level_hits, 0);
  ASSERT_INT_EQ(stats.deep_orphan_hits, 0);
  ASSERT_INT_EQ(stats.buffer_alt_created, 0);
  ASSERT_INT_EQ(stats.buffer_alt_used, 0);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, stats_are_thread_local) {
  PolyRangeifyStats main_stats = {0};
  ASSERT_TRUE(run_buffer_alt_stats_case(&main_stats));
  ASSERT_TRUE(main_stats.buffer_alt_max_count > 8);

  RangeifyStatsThreadResult thread_result = {0};
  pthread_t thread;
  ASSERT_INT_EQ(pthread_create(&thread, NULL, rangeify_stats_thread_main, &thread_result), 0);
  ASSERT_INT_EQ(pthread_join(thread, NULL), 0);

  ASSERT_TRUE(thread_result.ok);
  ASSERT_INT_EQ(thread_result.before.buffer_alt_created, 0);
  ASSERT_INT_EQ(thread_result.before.buffer_alt_max_count, 0);
  ASSERT_TRUE(thread_result.after.buffer_alt_max_count > 8);

  PolyRangeifyStats main_after = poly_rangeify_stats_get();
  ASSERT_INT_EQ(main_after.buffer_alt_created, main_stats.buffer_alt_created);
  ASSERT_INT_EQ(main_after.buffer_alt_max_count, main_stats.buffer_alt_max_count);
  PASS();
}

/* BUFFERIZE+INDEX structural tests */

TEST(rangeify, bufferize_same_size_dims_e2e) {
  /* Regression: 2x2 reduce-max → gradient requires BUFFERIZE with two
   * same-size dims (both size 2). Without structural INDEX wrapping, the
   * heuristic matcher can map both dims to the same context RANGE. */
  float x_d[4] = {1.0f, 3.0f, 2.0f, 4.0f};
  float gx_d[4] = {0};

  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_buffer_f32(ctx, 4);
  int64_t shape[] = {2, 2};
  PolyUOp *xr = poly_reshape(ctx, x, shape, 2);
  int64_t ax[] = {1};
  PolyUOp *m = poly_reduce_axis(ctx, POLY_OP_MAX, xr, ax, 1);
  int64_t ax2[] = {0};
  PolyUOp *loss = poly_reduce_axis(ctx, POLY_OP_ADD, m, ax2, 1);
  PolyUOp *gx = poly_grad(ctx, loss, xr);
  ASSERT_NOT_NULL(gx);

  PolyUOp *out = poly_buffer_f32(ctx, 4);
  PolyUOp *store = poly_store_val(ctx, out, gx);
  PolyUOp *sink = poly_sink1(ctx, store);

  PolyTestBufferView bindings[] = {POLY_TEST_HOST_VIEW(x, x_d), POLY_TEST_HOST_VIEW(out, gx_d)};
  int ret = poly_test_realize_buffer_views(ctx, sink, bindings, 2);
  ASSERT_INT_EQ(ret, 0);

  /* For [[1,3],[2,4]]: max over axis 1 = [3,4].
   * Gradient: 1 where value == max, 0 elsewhere → [[0,1],[0,1]]. */
  float expected[4] = {0.0f, 1.0f, 0.0f, 1.0f};
  for (int i = 0; i < 4; i++) {
    ASSERT_TRUE(!isnan(gx_d[i]));
    ASSERT_FLOAT_EQ(gx_d[i], expected[i], 1e-5);
  }

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, bufferize_index_wrapping_structural) {
  /* Structural: after apply_rangeify, every BUFFERIZE that has ranges
   * should appear as INDEX.src[0] in the consumer graph (never bare
   * in a consumer position). This verifies the INDEX wrapping at the
   * graph level, not just numeric correctness.
   *
   * Uses a shared computed op (neg(a)) consumed by two STOREs — this
   * forces range disagreement → BUFFERIZE creation. */
  int N = 8;
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *b = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *c = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *out1 = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *out2 = poly_buffer(ctx, POLY_FLOAT32, N);

  /* d = neg(a), consumed by both ADD and MUL → different stores → realizes */
  PolyUOp *d = poly_uop1(ctx, POLY_OP_NEG, POLY_FLOAT32, a, poly_arg_none());
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, d, b, poly_arg_none());
  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, d, c, poly_arg_none());
  PolyUOp *s1 = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out1, add, poly_arg_none());
  PolyUOp *s2 = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out2, mul, poly_arg_none());
  PolyUOp *stores[] = {s1, s2};
  PolyUOp *sink = poly_uop(ctx, POLY_OP_SINK, POLY_VOID, stores, 2, poly_arg_none());

  /* Run the full rangeify pipeline */
  PolyIndexingCtx *ictx = poly_indexing_ctx_new(ctx);
  PolyUOp *rangeified = run_apply_rangeify(ictx, sink);
  ASSERT_NOT_NULL(rangeified);

  /* Walk the rangeified graph and check: no BUFFERIZE appears bare
   * (i.e., as a source of a non-INDEX op) if it has ranges. */
  int n_topo;
  PolyUOp **topo = poly_toposort(ctx, rangeified, &n_topo);
  ASSERT_NOT_NULL(topo);

  int n_bufferize_refs = 0;
  int n_index_wrapped = 0;
  int n_bare_bufferize = 0;
  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    for (int j = 0; j < u->n_src; j++) {
      if (u->src[j]->op == POLY_OP_STAGE) {
        int n_rngs = u->src[j]->n_src - 1;
        if (n_rngs > 0) {
          if (u->op == POLY_OP_INDEX) {
            n_index_wrapped++;
          } else {
            n_bare_bufferize++;
          }
        }
        n_bufferize_refs++;
      }
    }
  }

  /* At least one BUFFERIZE reference should exist (shared d creates one) */
  ASSERT_TRUE(n_bufferize_refs > 0);
  /* All BUFFERIZE-with-ranges should be INDEX-wrapped */
  ASSERT_INT_EQ(n_bare_bufferize, 0);
  ASSERT_TRUE(n_index_wrapped > 0);

  poly_indexing_ctx_destroy(ictx);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, scalar_bufferize_read_indexes_singleton_intermediate) {
  /* Regression for tinygrad parity: new_range(1) is CONST(0), so a
   * singleton intermediate BUFFERIZE read must still become INDEX(..., 0).
   * Otherwise add_buffers can split it to a raw PARAM inside ALU, which
   * renders invalid C and bypasses the INDEX->LOAD late-codegen boundary. */
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *x = poly_buffer_f32(ctx, 1);
  PolyUOp *lg = poly_lgamma(ctx, x);
  int64_t axes[] = {0};
  PolyUOp *loss = poly_reduce_axis(ctx, POLY_OP_ADD, lg, axes, 1);
  PolyUOp *grad = poly_grad(ctx, loss, x);
  ASSERT_NOT_NULL(grad);

  PolyUOp *loss_out = poly_buffer_f32(ctx, 1);
  PolyUOp *grad_out = poly_buffer_f32(ctx, 1);
  PolyUOp *stores[] = {
      poly_store_val(ctx, loss_out, loss),
      poly_store_val(ctx, grad_out, grad),
  };
  PolyUOp *sink = poly_sink_n(ctx, stores, 2);

  PolySchedule *sched = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_NOT_NULL(sched);
  ASSERT_TRUE(sched->template->n_calls > 1);

  int bad = 0;
  for (int i = 0; i < sched->template->n_calls; i++)
    bad += count_alu_direct_storage_sources(ctx, poly_schedule_call_body(sched, i));
  ASSERT_INT_EQ(bad, 0);

  poly_schedule_free(sched);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, bufferize_foreign_range_same_size_dims_e2e) {
  /* Regression: multi-store kernel with shared 3x3 BUFFERIZE (both dims
   * size 3). With kernel-local split scheduling, this should lower
   * structurally with no remap fallback activity.
   *
   * Graph:
   *   a_flat(9) → reshape(3,3) → neg → d (shared, gets BUFFERIZE)
   *   out1 = d + b   (store 1)
   *   out2 = d * c   (store 2)
   *
   * Values chosen so row/col swap gives wrong results:
   *   a = [[1,2,3],[4,5,6],[7,8,9]]  → neg = [[-1,-2,-3],[-4,-5,-6],[-7,-8,-9]]
   *   b = [[10,20,30],[40,50,60],[70,80,90]]
   *   c = [[0.1,0.2,0.3],[0.4,0.5,0.6],[0.7,0.8,0.9]]
   *   out1[r][c] = -a[r][c] + b[r][c]
   *   out2[r][c] = -a[r][c] * c[r][c]
   *
   * If dims are swapped, out1[0][1] = -a[1][0] + b[0][1] = -4 + 20 = 16
   *   instead of correct -a[0][1] + b[0][1] = -2 + 20 = 18. */
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a_flat = poly_buffer(ctx, POLY_FLOAT32, 9);
  PolyUOp *b_flat = poly_buffer(ctx, POLY_FLOAT32, 9);
  PolyUOp *c_flat = poly_buffer(ctx, POLY_FLOAT32, 9);
  PolyUOp *out1_flat = poly_buffer(ctx, POLY_FLOAT32, 9);
  PolyUOp *out2_flat = poly_buffer(ctx, POLY_FLOAT32, 9);

  int64_t shape2d[] = {3, 3};
  PolyUOp *a = poly_reshape(ctx, a_flat, shape2d, 2);
  PolyUOp *b = poly_reshape(ctx, b_flat, shape2d, 2);
  PolyUOp *c = poly_reshape(ctx, c_flat, shape2d, 2);
  PolyUOp *out1 = poly_reshape(ctx, out1_flat, shape2d, 2);
  PolyUOp *out2 = poly_reshape(ctx, out2_flat, shape2d, 2);

  /* d = neg(a), shared by two stores → BUFFERIZE with 2 same-size dims */
  PolyUOp *d = poly_uop1(ctx, POLY_OP_NEG, POLY_FLOAT32, a, poly_arg_none());
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, d, b, poly_arg_none());
  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, d, c, poly_arg_none());
  PolyUOp *s1 = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out1, add, poly_arg_none());
  PolyUOp *s2 = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out2, mul, poly_arg_none());
  PolyUOp *stores[] = {s1, s2};
  PolyUOp *sink = poly_uop(ctx, POLY_OP_SINK, POLY_VOID, stores, 2, poly_arg_none());

  float a_d[9], b_d[9], c_d[9], o1_d[9], o2_d[9];
  for (int r = 0; r < 3; r++) {
    for (int col = 0; col < 3; col++) {
      int i = r * 3 + col;
      a_d[i] = (float)(i + 1); /* 1..9 */
      b_d[i] = (float)((col + 1) * 10 + r * 30); /* asymmetric */
      c_d[i] = (float)(i + 1) * 0.1f; /* 0.1..0.9 */
    }
  }

  PolyTestBufferView bindings[] = {
      POLY_TEST_HOST_VIEW(out1_flat, o1_d), POLY_TEST_HOST_VIEW(out2_flat, o2_d),
      POLY_TEST_HOST_VIEW(a_flat, a_d),     POLY_TEST_HOST_VIEW(b_flat, b_d),
      POLY_TEST_HOST_VIEW(c_flat, c_d),
  };
  poly_rangeify_stats_reset();
  int ret = poly_test_realize_buffer_views(ctx, sink, bindings, 5);
  ASSERT_INT_EQ(ret, 0);
  PolyRangeifyStats stats = poly_rangeify_stats_get();
  ASSERT_INT_EQ(stats.remap_calls, 0);
  ASSERT_INT_EQ(stats.remap_id_matches, 0);
  ASSERT_INT_EQ(stats.remap_pos_matches, 0);
  ASSERT_INT_EQ(stats.remap_unique_bound_matches, 0);
  ASSERT_INT_EQ(stats.remap_bound_matches, 0);
  ASSERT_INT_EQ(stats.remap_failures, 0);

  for (int i = 0; i < 9; i++) {
    float neg_a = -a_d[i];
    ASSERT_FLOAT_EQ(o1_d[i], neg_a + b_d[i], 1e-5);
    ASSERT_FLOAT_EQ(o2_d[i], neg_a * c_d[i], 1e-5);
  }

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, buffer_alt_ranges_grow_past_old_fixed_cap) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *x = poly_buffer(ctx, POLY_FLOAT32, 32);
  PolyUOp *terms[9];
  for (int i = 0; i < 9; i++) {
    int64_t pairs[][2] = {{i, i + 4}};
    terms[i] = poly_shrink(ctx, x, pairs, 1);
  }

  PolyUOp *acc = terms[0];
  for (int i = 1; i < 9; i++)
    acc = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, acc, terms[i], poly_arg_none());

  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, 4);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out, acc, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyIndexingCtx *ictx = poly_indexing_ctx_new(ctx);
  poly_rangeify_stats_reset();
  PolyUOp *result = run_apply_rangeify(ictx, sink);
  ASSERT_NOT_NULL(result);

  PolyRangeifyStats stats = poly_rangeify_stats_get();
  ASSERT_TRUE(stats.buffer_alt_max_count > 8);

  poly_indexing_ctx_destroy(ictx);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, bufferize_movement_chain_alt_ranges_e2e) {
  /* Regression: shared PAD source consumed by two shifted SHRINK paths
   * feeding an elementwise ADD, then reduced to scalar.
   *
   * This previously failed with:
   *   "BUFFER deep-foreign ranges failed remap"
   * because the BUFFER index expression carried foreign (5,7) ranges while
   * the reduction kernel context was (2,5,5).
   *
   * Structural fix: capture per-consumer BUFFER index mappings through
   * movement chains and use buffer_alt_rngs in lowering. */
  PolyCtx *ctx = poly_ctx_new();

  /* x: (1,3,5,5) flattened */
  PolyUOp *x_flat = poly_buffer(ctx, POLY_FLOAT32, 75);
  int64_t x_shape[] = {1, 3, 5, 5};
  PolyUOp *x = poly_reshape(ctx, x_flat, x_shape, 4);

  /* pad spatial dims: (1,3,7,7) */
  int64_t pad_pairs[][2] = {{0, 0}, {0, 0}, {1, 1}, {1, 1}};
  PolyUOp *xp = poly_pad(ctx, x, pad_pairs, 4);

  /* Two shifted windows from channel 0:
   * s1 = xp[:,0:1,0:5,0:5], s2 = xp[:,0:1,0:5,1:6] */
  int64_t s1_pairs[][2] = {{0, 1}, {0, 1}, {0, 5}, {0, 5}};
  int64_t s2_pairs[][2] = {{0, 1}, {0, 1}, {0, 5}, {1, 6}};
  PolyUOp *s1 = poly_shrink(ctx, xp, s1_pairs, 4);
  PolyUOp *s2 = poly_shrink(ctx, xp, s2_pairs, 4);

  /* Expand channel dim to 2 so reduction context includes (2,5,5) */
  int64_t out_shape[] = {1, 2, 5, 5};
  PolyUOp *e1 = poly_expand(ctx, s1, out_shape, 4);
  PolyUOp *e2 = poly_expand(ctx, s2, out_shape, 4);
  PolyUOp *sum = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, e1, e2, poly_arg_none());

  /* Reduce over channel+spatial dims to scalar-ish (shape [1]) */
  int64_t red_axes[] = {1, 2, 3};
  PolyUOp *loss = poly_reduce_axis(ctx, POLY_OP_ADD, sum, red_axes, 3);

  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, 1);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out, loss, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  float x_d[75], o_d[1];
  for (int i = 0; i < 75; i++)
    x_d[i] = (float)(i + 1);

  PolyTestBufferView bindings[] = {
      POLY_TEST_HOST_VIEW(out, o_d),
      POLY_TEST_HOST_VIEW(x_flat, x_d),
  };

  poly_rangeify_stats_reset();
  int ret = poly_test_realize_buffer_views(ctx, sink, bindings, 2);
  ASSERT_INT_EQ(ret, 0);

  /* Expected from numpy reference:
   * (broadcast(s1,(1,2,5,5)) + broadcast(s2,(1,2,5,5))).sum(axis=(1,2,3)) */
  ASSERT_FLOAT_EQ(o_d[0], 740.0f, 1e-5);

  PolyRangeifyStats stats = poly_rangeify_stats_get();
  ASSERT_INT_EQ(stats.remap_calls, 0);
  ASSERT_INT_EQ(stats.remap_id_matches, 0);
  ASSERT_INT_EQ(stats.remap_pos_matches, 0);
  ASSERT_INT_EQ(stats.remap_unique_bound_matches, 0);
  ASSERT_INT_EQ(stats.remap_bound_matches, 0);
  ASSERT_INT_EQ(stats.remap_failures, 0);
  /* Depending on scheduling order, this may avoid deep-orphan fallback
   * entirely (preferred) or use structural alt mappings. In both cases,
   * remap fallback counters must stay zero. */

  poly_ctx_destroy(ctx);
  PASS();
}

/* add_buffers tests */

TEST(rangeify, add_buffers_noop_single_kernel) {
  /* Simple vecadd: c = a + b. No BUFFERIZE nodes after apply_rangeify,
   * so add_buffers should be a no-op (no BUFFER/AFTER/END introduced). */
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *b = poly_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *c = poly_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, a, b, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, c, add, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyIndexingCtx *ictx = poly_indexing_ctx_new(ctx);
  PolyUOp *rangeified = run_apply_rangeify(ictx, sink);
  ASSERT_NOT_NULL(rangeified);

  /* No BUFFERIZE should exist (single kernel, no multi-consumer divergence) */
  ASSERT_INT_EQ(count_ops(ctx, rangeified, POLY_OP_STAGE), 0);

  /* Apply add_buffers — should be a no-op */
  PolyUOp *result = poly_apply_add_buffers(ctx, rangeified, NULL);
  ASSERT_NOT_NULL(result);

  /* No new BUFFER, AFTER, or END nodes introduced */
  ASSERT_INT_EQ(count_ops(ctx, result, POLY_OP_AFTER), 0);
  ASSERT_INT_EQ(count_ops(ctx, result, POLY_OP_LUNIQUE), 0);

  /* STORE and SINK still present */
  ASSERT_INT_EQ(count_ops(ctx, result, POLY_OP_STORE), 1);
  ASSERT_INT_EQ(count_ops(ctx, result, POLY_OP_SINK), 1);

  poly_indexing_ctx_destroy(ictx);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, add_buffers_multi_kernel) {
  /* d = neg(a), used by two stores with different ranges:
   *   STORE(out1, d + b)  [10 elements]
   *   STORE(out2, d * c)  [10 elements]
   * After apply_rangeify, d is realized → BUFFERIZE node.
   * After add_buffers, BUFFERIZE replaced by BUFFER+STORE+END+AFTER chain. */
  int N = 10;
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *b = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *c = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *out1 = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *out2 = poly_buffer(ctx, POLY_FLOAT32, N);

  PolyUOp *d = poly_uop1(ctx, POLY_OP_NEG, POLY_FLOAT32, a, poly_arg_none());
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, d, b, poly_arg_none());
  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, d, c, poly_arg_none());
  PolyUOp *s1 = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out1, add, poly_arg_none());
  PolyUOp *s2 = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out2, mul, poly_arg_none());
  PolyUOp *stores[] = {s1, s2};
  PolyUOp *sink = poly_uop(ctx, POLY_OP_SINK, POLY_VOID, stores, 2, poly_arg_none());

  PolyIndexingCtx *ictx = poly_indexing_ctx_new(ctx);
  PolyUOp *rangeified = run_apply_rangeify(ictx, sink);
  ASSERT_NOT_NULL(rangeified);

  /* Should have exactly 1 BUFFERIZE node (for d = neg(a)) */
  ASSERT_INT_EQ(count_ops(ctx, rangeified, POLY_OP_STAGE), 1);

  /* Apply add_buffers */
  PolyUOp *result = poly_apply_add_buffers(ctx, rangeified, NULL);
  ASSERT_NOT_NULL(result);

  /* No BUFFERIZE remains */
  ASSERT_INT_EQ(count_ops(ctx, result, POLY_OP_STAGE), 0);

  /* New intermediate buffer infrastructure present */
  ASSERT_INT_EQ(count_ops(ctx, result, POLY_OP_LUNIQUE), 1);
  ASSERT_TRUE(count_ops(ctx, result, POLY_OP_AFTER) >= 1);
  ASSERT_TRUE(count_ops(ctx, result, POLY_OP_END) >= 1);

  /* Original 2 consumer stores still present, plus 1 producer store */
  ASSERT_INT_EQ(count_ops(ctx, result, POLY_OP_STORE), 3);

  /* Verify BUFFER node has correct size in arg */
  int n_topo;
  PolyUOp **topo = poly_toposort(ctx, result, &n_topo);
  int found_buf = 0;
  for (int i = 0; i < n_topo; i++) {
    if (topo[i]->op == POLY_OP_BUFFER && topo[i]->n_src == 2 &&
        topo[i]->src[0]->op == POLY_OP_LUNIQUE) {
      /* Intermediate BUFFER — check size */
      ASSERT_INT_EQ(topo[i]->arg.i, N);
      found_buf = 1;
    }
  }
  ASSERT_TRUE(found_buf);

  poly_indexing_ctx_destroy(ictx);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, add_buffers_uses_bufferize_device_metadata) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, 8);
  PolyUOp *device = poly_device_uop_from_name(ctx, "CPU:1");
  PolyUOp *copy = poly_uop2(ctx, POLY_OP_COPY, POLY_FLOAT32, a, device, poly_arg_none());
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(8));
  PolyUOp *range =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_range(0, POLY_AXIS_LOOP));

  PolyUOp *src[] = {copy, range};
  PolyUOp *bufferize = poly_uop(
      ctx, POLY_OP_STAGE, POLY_FLOAT32, src, 2,
      poly_arg_bufferize_opts("CPU:1", POLY_ADDR_GLOBAL, false)
  );
  ASSERT_INT_EQ(bufferize->arg.kind, POLY_ARG_BUFFERIZE_OPTS);
  ASSERT_FALSE(poly_bufferize_arg_removable(bufferize->arg));
  ASSERT_STR_EQ(poly_bufferize_arg_device(bufferize->arg), "CPU:1");

  PolyUOp *result = poly_apply_add_buffers(ctx, bufferize, NULL);
  ASSERT_NOT_NULL(result);

  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, result, &n_topo);
  ASSERT_NOT_NULL(topo);
  bool found = false;
  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    if (u->op != POLY_OP_BUFFER || u->n_src < 2 || u->src[0]->op != POLY_OP_LUNIQUE) continue;
    ASSERT_TRUE(u->src[1]->op == POLY_OP_DEVICE);
    ASSERT_TRUE(u->src[1]->arg.kind == POLY_ARG_STRING);
    ASSERT_STR_EQ(u->src[1]->arg.str, "CPU:1");
    found = true;
  }
  ASSERT_TRUE(found);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, earliest_copy_equality_uses_exact_device_identity) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  /* Pinned earliest_rewrites compares exact UOp.device strings
   * (rangeify.py:184-187): CPU->CPU is NOOP, while CPU->CPU:1 remains COPY. */
  PolyUOp *source = poly_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CPU);
  PolyUOp *cpu = poly_device_uop(ctx, POLY_DEVICE_CPU);
  PolyUOp *cpu1 = poly_device_uop_from_name(ctx, "CPU:1");
  PolyUOp *same_src[2] = {source, cpu};
  PolyUOp *different_src[2] = {source, cpu1};
  PolyUOp *same = poly_uop(ctx, POLY_OP_COPY, POLY_FLOAT32, same_src, 2, poly_arg_none());
  PolyUOp *different = poly_uop(ctx, POLY_OP_COPY, POLY_FLOAT32, different_src, 2, poly_arg_none());

  /* Pinned's preceding COPY(MSELECT, same device) rule returns the exact
   * selected occurrence; the generic same-device rule below still emits a
   * NOOP for ordinary values (rangeify.py:183-187). */
  const char *names[2] = {"CPU", "CPU:1"};
  PolyUOp *tuple = poly_device_uop_from_names(ctx, names, 2);
  PolyUOp *tuple_unique = poly_uop0(ctx, POLY_OP_UNIQUE, POLY_VOID, poly_arg_int(7301));
  PolyUOp *tuple_src[2] = {tuple_unique, tuple};
  PolyUOp *tuple_buffer =
      poly_uop(ctx, POLY_OP_BUFFER, POLY_FLOAT32, tuple_src, 2, poly_arg_int(4));
  PolyUOp *selected = poly_uop1(ctx, POLY_OP_MSELECT, POLY_FLOAT32, tuple_buffer, poly_arg_int(0));
  PolyUOp *selected_copy_src[2] = {selected, cpu};
  PolyUOp *selected_copy = poly_uop(
      ctx, POLY_OP_COPY, POLY_FLOAT32, selected_copy_src, 2, poly_arg_none());

  PolyUOp *same_result = poly_apply_earliest_rewrites(ctx, poly_sink1(ctx, same));
  PolyUOp *different_result = poly_apply_earliest_rewrites(ctx, poly_sink1(ctx, different));
  PolyUOp *selected_result =
      poly_apply_earliest_rewrites(ctx, poly_sink1(ctx, selected_copy));
  ASSERT_NOT_NULL(same_result);
  ASSERT_NOT_NULL(different_result);
  ASSERT_NOT_NULL(selected_result);
  ASSERT_INT_EQ(same_result->n_src, 1);
  ASSERT_INT_EQ(different_result->n_src, 1);
  ASSERT_INT_EQ(same_result->src[0]->op, POLY_OP_NOOP);
  ASSERT_PTR_EQ(same_result->src[0]->src[0], source);
  ASSERT_PTR_EQ(different_result->src[0], different);
  ASSERT_STR_EQ(poly_uop_device_name(ctx, different_result->src[0]), "CPU:1");
  ASSERT_INT_EQ(selected_result->n_src, 1);
  ASSERT_PTR_EQ(selected_result->src[0], selected);

  poly_ctx_destroy(ctx);
  PASS();
}

static PolyUOp *multi_pm_test_source(
    PolyCtx *ctx,
    int unique_id,
    PolyDType dtype,
    const char *device_name,
    int64_t rows,
    int64_t cols
) {
  PolyUOp *unique = poly_uop0(ctx, POLY_OP_UNIQUE, POLY_VOID, poly_arg_int(unique_id));
  PolyUOp *device = poly_device_uop_from_name(ctx, device_name);
  PolyUOp *buffer_src[2] = {unique, device};
  PolyUOp *buffer = poly_uop(
      ctx, POLY_OP_BUFFER, dtype, buffer_src, 2, poly_arg_int(rows * cols));
  int64_t shape[2] = {rows, cols};
  return buffer ? poly_reshape(ctx, buffer, shape, 2) : NULL;
}

static PolyUOp *multi_pm_test_copy_tuple(
    PolyCtx *ctx,
    int unique_id,
    PolyDType dtype
) {
  const char *names[2] = {"CPU", "CPU:1"};
  PolyUOp *tuple = poly_device_uop_from_names(ctx, names, 2);
  PolyUOp *source = multi_pm_test_source(ctx, unique_id, dtype, "CPU", 4, 4);
  PolyUOp *copy_src[2] = {source, tuple};
  return source && tuple
             ? poly_uop(ctx, POLY_OP_COPY, dtype, copy_src, 2, poly_arg_none())
             : NULL;
}

static PolyUOp *multi_pm_test_shard(
    PolyCtx *ctx,
    int unique_id,
    PolyDType dtype,
    int axis
) {
  if (axis < 0 || axis > 1) return NULL;
  PolyUOp *copy = multi_pm_test_copy_tuple(ctx, unique_id, dtype);
  PolyUOp *device_num = poly_uop0(
      ctx, POLY_OP_DEFINE_VAR, POLY_INDEX,
      poly_arg_define_var("_device_num", 0, 1));
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(0));
  PolyUOp *two = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(2));
  PolyUOp *four = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(4));
  PolyUOp *start = poly_alu2(ctx, POLY_OP_MUL, device_num, two);
  PolyUOp *starts[2] = {axis == 0 ? start : zero, axis == 1 ? start : zero};
  PolyUOp *sizes[2] = {axis == 0 ? two : four, axis == 1 ? two : four};
  PolyUOp *local = copy ? poly_shrink_uop(ctx, copy, starts, sizes, 2) : NULL;
  return local ? poly_uop1(ctx, POLY_OP_MULTI, dtype, local, poly_arg_int(axis)) : NULL;
}

static PolyUOp *multi_pm_test_axis0_column_shard(
    PolyCtx *ctx,
    int unique_id,
    PolyDType dtype
) {
  const char *names[2] = {"CPU", "CPU:1"};
  PolyUOp *tuple = poly_device_uop_from_names(ctx, names, 2);
  PolyUOp *source = multi_pm_test_source(ctx, unique_id, dtype, "CPU", 4, 1);
  PolyUOp *copy_src[2] = {source, tuple};
  PolyUOp *copy = source && tuple
                      ? poly_uop(ctx, POLY_OP_COPY, dtype, copy_src, 2, poly_arg_none())
                      : NULL;
  PolyUOp *device_num = poly_uop0(
      ctx, POLY_OP_DEFINE_VAR, POLY_INDEX,
      poly_arg_define_var("_device_num", 0, 1));
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(0));
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(1));
  PolyUOp *two = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(2));
  PolyUOp *starts[2] = {poly_alu2(ctx, POLY_OP_MUL, device_num, two), zero};
  PolyUOp *sizes[2] = {two, one};
  PolyUOp *local = copy ? poly_shrink_uop(ctx, copy, starts, sizes, 2) : NULL;
  return local ? poly_uop1(ctx, POLY_OP_MULTI, dtype, local, poly_arg_int(0)) : NULL;
}

static PolyUOp *multi_pm_test_param(
    PolyCtx *ctx,
    int slot,
    const char *device,
    const char **devices,
    int n_devices,
    bool has_axis
) {
  PolyUOp *dims[2] = {
      poly_const_int(ctx, has_axis ? 4 : 2), poly_const_int(ctx, 4)};
  PolyUOp *shape = poly_uop(
      ctx, POLY_OP_STACK, poly_dtype_vec(POLY_INDEX, 2), dims, 2,
      poly_arg_none());
  PolyParamArg arg = {
      .slot = slot,
      .addrspace = POLY_ADDR_GLOBAL,
      .axis = 0,
      .has_axis = has_axis,
      .device = device,
      .devices = devices,
      .n_devices = n_devices,
      .device_is_tuple = devices != NULL,
  };
  return poly_uop1(ctx, POLY_OP_PARAM, POLY_FLOAT32, shape, poly_arg_param(&arg));
}

static PolyUOp *multi_pm_test_realized_multi(
    PolyCtx *ctx,
    int unique_base,
    const float values[16]
) {
  const char *devices[2] = {"CPU", "CPU:1"};
  PolyUOp *locals[2] = {0};
  for (int lane = 0; lane < 2; lane++) {
    PolyUOp *unique =
        poly_uop0(ctx, POLY_OP_UNIQUE, POLY_VOID, poly_arg_int(unique_base + lane));
    PolyUOp *device = poly_device_uop_from_name(ctx, devices[lane]);
    PolyUOp *buffer_src[2] = {unique, device};
    PolyUOp *buffer = poly_uop(
        ctx, POLY_OP_BUFFER, POLY_FLOAT32, buffer_src, 2, poly_arg_int(8));
    if (!buffer || poly_buffer_allocate(ctx, buffer, POLY_DEVICE_CPU) != 0 ||
        poly_buffer_copyin(ctx, buffer, values + lane * 8, 8 * sizeof(float)) != 0)
      return NULL;
    locals[lane] = poly_reshape(ctx, buffer, (int64_t[]){2, 4}, 2);
    if (!locals[lane]) return NULL;
  }
  PolyUOp *stack = poly_uop(
      ctx, POLY_OP_MSTACK, POLY_FLOAT32, locals, 2, poly_arg_none());
  return stack
             ? poly_uop1(ctx, POLY_OP_MULTI, POLY_FLOAT32, stack, poly_arg_int(0))
             : NULL;
}

static int multi_pm_test_node_count(PolyCtx *ctx, PolyUOp *root) {
  int n = 0;
  PolyUOp **topo = poly_toposort_alloc(ctx, root, &n);
  bool ok = topo != NULL;
  poly_toposort_free(topo);
  return ok ? n : -1;
}

static PolyUOp *multi_pm_test_find_named_call(
    PolyCtx *ctx,
    PolyUOp *root,
    const char *name
) {
  int n = 0;
  PolyUOp **topo = poly_toposort_ex_alloc(ctx, root, &n, NULL, true);
  PolyUOp *found = NULL;
  for (int i = 0; topo && i < n; i++) {
    PolyUOp *u = topo[i];
    if (u->op == POLY_OP_CALL && u->arg.kind == POLY_ARG_STRING &&
        strcmp(u->arg.str, name) == 0) {
      found = u;
      break;
    }
  }
  poly_toposort_free(topo);
  return found;
}

static PolyUOp *multi_pm_test_find_unique_buffer(
    PolyCtx *ctx,
    PolyUOp *root,
    int64_t unique_id
) {
  int n = 0;
  PolyUOp **topo = poly_toposort_ex_alloc(ctx, root, &n, NULL, true);
  PolyUOp *found = NULL;
  for (int i = 0; topo && i < n; i++) {
    PolyUOp *u = topo[i];
    if (u->op == POLY_OP_BUFFER && u->n_src >= 1 && u->src[0]->op == POLY_OP_UNIQUE &&
        u->src[0]->arg.kind == POLY_ARG_INT && u->src[0]->arg.i == unique_id) {
      found = u;
      break;
    }
  }
  poly_toposort_free(topo);
  return found;
}

TEST(rangeify, add_buffers_removes_invalid_clone_initialization_like_pinned) {
  /* Pinned pm_add_buffers turns STORE(dst, Invalid) into zero-source NOOP,
   * then removes that effect from AFTER (rangeify.py:476-479). The clone's
   * storage identity remains, but it schedules no initialization kernel. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  const char *names[2] = {"CPU", "CPU:1"};
  PolyUOp *tuple = poly_device_uop_from_names(ctx, names, 2);
  PolyUOp *unique = poly_uop0(ctx, POLY_OP_UNIQUE, POLY_VOID, poly_arg_int(9101));
  PolyUOp *buffer_src[2] = {unique, tuple};
  PolyUOp *buffer = poly_uop(
      ctx, POLY_OP_BUFFER, POLY_FLOAT32, buffer_src, 2, poly_arg_int(4));
  PolyUOp *invalid = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_invalid());
  PolyUOp *invalid_shaped = poly_reshape(ctx, invalid, (int64_t[]){1}, 1);
  invalid_shaped = poly_expand(ctx, invalid_shaped, (int64_t[]){4}, 1);
  PolyUOp *store = poly_uop2(
      ctx, POLY_OP_STORE, POLY_VOID, buffer, invalid_shaped, poly_arg_none());
  PolyUOp *after = poly_uop2(
      ctx, POLY_OP_AFTER, POLY_FLOAT32, buffer, store, poly_arg_none());
  PolyUOp *kernel_graph = poly_get_kernel_graph(ctx, poly_sink1(ctx, after));
  ASSERT_NOT_NULL(kernel_graph);
  ASSERT_INT_EQ(multi_pm_test_node_count(ctx, kernel_graph), 4);
  ASSERT_INT_EQ(count_ops(ctx, kernel_graph, POLY_OP_BUFFER), 1);
  ASSERT_INT_EQ(count_ops(ctx, kernel_graph, POLY_OP_AFTER), 0);
  ASSERT_INT_EQ(count_ops(ctx, kernel_graph, POLY_OP_STORE), 0);
  ASSERT_INT_EQ(count_ops(ctx, kernel_graph, POLY_OP_NOOP), 0);

  PolyKernelScheduleResult schedule =
      poly_build_kernel_schedule_from_kernel_graph(ctx, kernel_graph);
  ASSERT_INT_EQ(schedule.n_kernels, 0);
  poly_kernel_schedule_result_free(&schedule);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, multi_pm_alu_reduce_collective_match_pinned_topology) {
  /* Exact dependency group from pinned schedule/multi.py:44-78,113-120,
   * 145-158 and UOp._shard/_unshard (uop/ops.py:649-660). */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  const char *names[2] = {"CPU", "CPU:1"};
  PolyUOp *tuple = poly_device_uop_from_names(ctx, names, 2);
  ASSERT_NOT_NULL(tuple);

  PolyUOp *same = poly_alu2(
      ctx, POLY_OP_ADD, multi_pm_test_shard(ctx, 901, POLY_FLOAT32, 0),
      multi_pm_test_shard(ctx, 902, POLY_FLOAT32, 0));
  PolyUOp *same_result = poly_apply_multi_pm(ctx, same);
  ASSERT_NOT_NULL(same_result);
  ASSERT_INT_EQ(same_result->op, POLY_OP_MULTI);
  ASSERT_INT_EQ(same_result->arg.kind, POLY_ARG_INT);
  ASSERT_INT_EQ(same_result->arg.i, 0);
  ASSERT_INT_EQ(same_result->src[0]->op, POLY_OP_ADD);
  ASSERT_INT_EQ(same_result->src[0]->src[0]->op, POLY_OP_MSTACK);
  ASSERT_INT_EQ(same_result->src[0]->src[1]->op, POLY_OP_MSTACK);
  ASSERT_INT_EQ(count_ops(ctx, same_result, POLY_OP_MULTI), 1);
  ASSERT_INT_EQ(count_ops(ctx, same_result, POLY_OP_MSTACK), 2);
  ASSERT_INT_EQ(count_ops(ctx, same_result, POLY_OP_COPY), 4);
  ASSERT_INT_EQ(count_ops(ctx, same_result, POLY_OP_SHRINK), 4);
  ASSERT_INT_EQ(multi_pm_test_node_count(ctx, same_result), 27);

  PolyUOp *unsharded = poly_alu2(
      ctx, POLY_OP_ADD, multi_pm_test_shard(ctx, 903, POLY_FLOAT32, 0),
      multi_pm_test_copy_tuple(ctx, 904, POLY_FLOAT32));
  PolyUOp *unsharded_result = poly_apply_multi_pm(ctx, unsharded);
  ASSERT_NOT_NULL(unsharded_result);
  ASSERT_INT_EQ(unsharded_result->op, POLY_OP_MULTI);
  ASSERT_INT_EQ(unsharded_result->arg.i, 0);
  ASSERT_INT_EQ(unsharded_result->src[0]->op, POLY_OP_ADD);
  ASSERT_INT_EQ(unsharded_result->src[0]->src[0]->op, POLY_OP_MSTACK);
  ASSERT_INT_EQ(unsharded_result->src[0]->src[1]->op, POLY_OP_MSTACK);
  ASSERT_INT_EQ(count_ops(ctx, unsharded_result, POLY_OP_MULTI), 1);
  ASSERT_INT_EQ(count_ops(ctx, unsharded_result, POLY_OP_MSTACK), 2);
  ASSERT_INT_EQ(multi_pm_test_node_count(ctx, unsharded_result), 27);

  PolyUOp *mismatch = poly_alu2(
      ctx, POLY_OP_ADD, multi_pm_test_shard(ctx, 905, POLY_FLOAT32, 0),
      multi_pm_test_shard(ctx, 906, POLY_FLOAT32, 1));
  PolyUOp *mismatch_result = poly_apply_multi_pm(ctx, mismatch);
  ASSERT_NOT_NULL(mismatch_result);
  ASSERT_INT_EQ(mismatch_result->op, POLY_OP_MULTI);
  ASSERT_INT_EQ(mismatch_result->arg.i, 1);
  PolyUOp *mismatch_add = mismatch_result->src[0];
  ASSERT_INT_EQ(mismatch_add->op, POLY_OP_ADD);
  ASSERT_INT_EQ(mismatch_add->src[0]->op, POLY_OP_SHRINK);
  ASSERT_INT_EQ(mismatch_add->src[0]->src[0]->op, POLY_OP_ALLREDUCE);
  ASSERT_INT_EQ(mismatch_add->src[0]->src[0]->src[0]->op, POLY_OP_PAD);
  ASSERT_INT_EQ(mismatch_add->src[0]->src[0]->src[0]->src[0]->op, POLY_OP_MSTACK);
  ASSERT_INT_EQ(mismatch_add->src[1]->op, POLY_OP_MSTACK);
  ASSERT_INT_EQ(count_ops(ctx, mismatch_result, POLY_OP_MULTI), 1);
  ASSERT_INT_EQ(count_ops(ctx, mismatch_result, POLY_OP_ALLREDUCE), 1);
  ASSERT_INT_EQ(count_ops(ctx, mismatch_result, POLY_OP_PAD), 1);
  ASSERT_INT_EQ(count_ops(ctx, mismatch_result, POLY_OP_MSTACK), 2);
  /* Pinned uses PARAM(STACK(), name=_device_num); the registered
   * PG-PARITY-002 DEFINE_VAR spelling accounts for the sole node delta. */
  ASSERT_INT_EQ(multi_pm_test_node_count(ctx, mismatch_result), 37);

  int64_t axis0[1] = {0}, axis1[1] = {1};
  PolyUOp *reduce_shard = poly_reduce_axis(
      ctx, POLY_OP_ADD, multi_pm_test_shard(ctx, 907, POLY_FLOAT32, 0), axis0, 1);
  PolyUOp *reduce_shard_result = poly_apply_multi_pm(ctx, reduce_shard);
  ASSERT_NOT_NULL(reduce_shard_result);
  ASSERT_INT_EQ(reduce_shard_result->op, POLY_OP_ALLREDUCE);
  ASSERT_INT_EQ(reduce_shard_result->arg.kind, POLY_ARG_OPS);
  ASSERT_INT_EQ(reduce_shard_result->arg.ops, POLY_OP_ADD);
  ASSERT_INT_EQ(reduce_shard_result->src[0]->op, POLY_OP_REDUCE);
  ASSERT_INT_EQ(reduce_shard_result->src[0]->src[0]->op, POLY_OP_MSTACK);
  ASSERT_PTR_EQ(reduce_shard_result->src[1], tuple);
  ASSERT_INT_EQ(count_ops(ctx, reduce_shard_result, POLY_OP_MULTI), 0);
  ASSERT_INT_EQ(multi_pm_test_node_count(ctx, reduce_shard_result), 20);

  PolyUOp *reduce_other = poly_reduce_axis(
      ctx, POLY_OP_ADD, multi_pm_test_shard(ctx, 908, POLY_FLOAT32, 0), axis1, 1);
  PolyUOp *reduce_other_result = poly_apply_multi_pm(ctx, reduce_other);
  ASSERT_NOT_NULL(reduce_other_result);
  ASSERT_INT_EQ(reduce_other_result->op, POLY_OP_MULTI);
  ASSERT_INT_EQ(reduce_other_result->arg.i, 0);
  ASSERT_INT_EQ(reduce_other_result->src[0]->op, POLY_OP_REDUCE);
  ASSERT_INT_EQ(reduce_other_result->src[0]->src[0]->op, POLY_OP_MSTACK);
  ASSERT_INT_EQ(multi_pm_test_node_count(ctx, reduce_other_result), 19);

  PolyUOp *explicit_src[2] = {
      multi_pm_test_shard(ctx, 909, POLY_FLOAT32, 0), tuple};
  PolyUOp *explicit_allreduce = poly_uop(
      ctx, POLY_OP_ALLREDUCE, POLY_FLOAT32, explicit_src, 2,
      poly_arg_ops(POLY_OP_ADD));
  PolyUOp *explicit_result = poly_apply_multi_pm(ctx, explicit_allreduce);
  ASSERT_NOT_NULL(explicit_result);
  ASSERT_INT_EQ(explicit_result->op, POLY_OP_MULTI);
  ASSERT_INT_EQ(explicit_result->arg.i, 0);
  ASSERT_INT_EQ(explicit_result->src[0]->op, POLY_OP_ALLREDUCE);
  ASSERT_INT_EQ(explicit_result->src[0]->src[0]->op, POLY_OP_MSTACK);
  ASSERT_PTR_EQ(explicit_result->src[0]->src[1], tuple);
  ASSERT_INT_EQ(multi_pm_test_node_count(ctx, explicit_result), 20);

  /* COPY(MULTI -> scalar) uses every selected occurrence and concatenates
   * them; the older generic COPY_TO_ONE rule is only for axis-less MSTACK. */
  PolyUOp *copy_one_src[2] = {
      multi_pm_test_shard(ctx, 910, POLY_FLOAT32, 0),
      poly_device_uop_from_name(ctx, "CPU")};
  PolyUOp *copy_one = poly_uop(
      ctx, POLY_OP_COPY, POLY_FLOAT32, copy_one_src, 2, poly_arg_none());
  PolyUOp *copy_one_result = poly_apply_multi_pm(ctx, copy_one);
  ASSERT_NOT_NULL(copy_one_result);
  ASSERT_INT_EQ(copy_one_result->op, POLY_OP_ADD);
  ASSERT_INT_EQ(copy_one_result->src[0]->op, POLY_OP_PAD);
  ASSERT_INT_EQ(copy_one_result->src[1]->op, POLY_OP_PAD);
  ASSERT_INT_EQ(copy_one_result->src[0]->src[0]->op, POLY_OP_COPY);
  ASSERT_INT_EQ(copy_one_result->src[1]->src[0]->op, POLY_OP_COPY);
  ASSERT_INT_EQ(count_ops(ctx, copy_one_result, POLY_OP_MULTI), 0);
  ASSERT_INT_EQ(count_ops(ctx, copy_one_result, POLY_OP_MSELECT), 0);
  ASSERT_INT_EQ(count_ops(ctx, copy_one_result, POLY_OP_COPY), 4);
  ASSERT_INT_EQ(multi_pm_test_node_count(ctx, copy_one_result), 21);

  /* Pinned ALLREDUCE_CAST performs a sharded BF16 collective in BF16 while
   * retaining a float32 local reduction/result (multi.py:72-76). */
  RangeifyEnvSave allreduce_cast_env = rangeify_save_env("ALLREDUCE_CAST");
  setenv("ALLREDUCE_CAST", "1", 1);
  PolyUOp *bf16_children[2] = {
      multi_pm_test_source(ctx, 911, POLY_BFLOAT16, "CPU", 2, 4),
      multi_pm_test_source(ctx, 912, POLY_BFLOAT16, "CPU:1", 2, 4)};
  PolyUOp *bf16_stack = poly_uop(
      ctx, POLY_OP_MSTACK, POLY_BFLOAT16, bf16_children, 2, poly_arg_none());
  PolyUOp *bf16_cast =
      poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, bf16_stack, poly_arg_none());
  PolyUOp *bf16_multi =
      poly_uop1(ctx, POLY_OP_MULTI, POLY_FLOAT32, bf16_cast, poly_arg_int(0));
  PolyUOp *bf16_reduce =
      poly_reduce_axis(ctx, POLY_OP_ADD, bf16_multi, axis0, 1);
  PolyUOp *bf16_result = poly_apply_multi_pm(ctx, bf16_reduce);
  rangeify_restore_env(&allreduce_cast_env);
  ASSERT_NOT_NULL(bf16_result);
  ASSERT_INT_EQ(bf16_result->op, POLY_OP_CAST);
  ASSERT_TRUE(poly_dtype_eq(bf16_result->dtype, POLY_FLOAT32));
  ASSERT_INT_EQ(bf16_result->src[0]->op, POLY_OP_ALLREDUCE);
  ASSERT_TRUE(poly_dtype_eq(bf16_result->src[0]->dtype, POLY_BFLOAT16));
  ASSERT_INT_EQ(bf16_result->src[0]->src[0]->op, POLY_OP_CAST);
  ASSERT_TRUE(poly_dtype_eq(bf16_result->src[0]->src[0]->dtype, POLY_BFLOAT16));
  ASSERT_INT_EQ(bf16_result->src[0]->src[0]->src[0]->op, POLY_OP_REDUCE);
  ASSERT_INT_EQ(count_ops(ctx, bf16_result, POLY_OP_CAST), 3);
  ASSERT_INT_EQ(multi_pm_test_node_count(ctx, bf16_result), 18);

  /* graph_rewrite(..., enter_calls=False) keeps the body exact, while pinned
   * multi.py:167-169 strips a direct MULTI from a void CALL argument. */
  PolyUOp *opaque_body = poly_sink1(ctx, same);
  PolyUOp *call_src[2] = {opaque_body, same};
  PolyUOp *call = poly_uop(ctx, POLY_OP_CALL, POLY_VOID, call_src, 2, poly_arg_none());
  PolyUOp *call_result = poly_apply_multi_pm(ctx, call);
  ASSERT_NOT_NULL(call_result);
  ASSERT_PTR_EQ(call_result->src[0], opaque_body);
  ASSERT_INT_EQ(call_result->src[1]->op, POLY_OP_ADD);
  ASSERT_INT_EQ(call_result->src[1]->src[0]->op, POLY_OP_MSTACK);
  ASSERT_INT_EQ(call_result->src[1]->src[1]->op, POLY_OP_MSTACK);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, recursive_allreduce_schedule_matches_pinned_topology_and_values) {
  /* Pinned graph_rewrite keeps a nested CALL body opaque until recursive
   * scheduling, then lowers the naive two-device ALLREDUCE body to COPY,
   * COPY, compute. The resolved full LINEAR has eight calls and exec_copy
   * resolves each MSELECT before copying (schedule/__init__.py:94-105;
   * engine/realize.py:142-180). */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  poly_ctx_set_preferred_device(ctx, POLY_DEVICE_CPU);

  PolyUOp *root = poly_reduce_axis(
      ctx, POLY_OP_ADD, multi_pm_test_shard(ctx, 907, POLY_FLOAT32, 0),
      (int64_t[]){0}, 1);
  ASSERT_NOT_NULL(root);
  PolyUOp *source_buffer = multi_pm_test_find_unique_buffer(ctx, root, 907);
  ASSERT_NOT_NULL(source_buffer);
  float input[16];
  for (int i = 0; i < 16; i++) input[i] = (float)(700 + i);
  ASSERT_INT_EQ(poly_buffer_allocate(ctx, source_buffer, POLY_DEVICE_CPU), 0);
  ASSERT_INT_EQ(poly_buffer_copyin(ctx, source_buffer, input, sizeof(input)), 0);

  PolyUOp *callified_out = NULL;
  PolyUOp *callified = poly_transform_to_call(ctx, &root, 1, &callified_out);
  ASSERT_NOT_NULL(callified);
  ASSERT_INT_EQ(callified->op, POLY_OP_CALL);
  ASSERT_TRUE(callified->n_src >= 1);
  ASSERT_INT_EQ(callified->src[0]->op, POLY_OP_SINK);
  ASSERT_NOT_NULL(callified_out);

  PolyUOp *multi = poly_apply_multi_pm(ctx, callified->src[0]);
  PolyUOp *earliest = poly_apply_earliest_rewrites(ctx, multi);
  ASSERT_NOT_NULL(earliest);
  PolyUOp *nested_call = multi_pm_test_find_named_call(ctx, earliest, "allreduce");
  ASSERT_NOT_NULL(nested_call);
  ASSERT_INT_EQ(nested_call->n_src, 3);
  PolyUOp *nested_body = nested_call->src[0];
  ASSERT_NOT_NULL(nested_body);
  ASSERT_INT_EQ(nested_body->op, POLY_OP_SINK);
  ASSERT_INT_EQ(multi_pm_test_node_count(ctx, nested_body), 14);
  ASSERT_INT_EQ(count_ops(ctx, nested_body, POLY_OP_AFTER), 1);
  ASSERT_INT_EQ(count_ops(ctx, nested_body, POLY_OP_STORE), 1);
  ASSERT_INT_EQ(count_ops(ctx, nested_body, POLY_OP_CONTIGUOUS), 0);

  PolyUOp *outer_kernel_graph = poly_get_kernel_graph(ctx, earliest);
  ASSERT_NOT_NULL(outer_kernel_graph);
  PolyUOp *outer_nested_call =
      multi_pm_test_find_named_call(ctx, outer_kernel_graph, "allreduce");
  ASSERT_NOT_NULL(outer_nested_call);
  ASSERT_PTR_EQ(outer_nested_call->src[0], nested_body);
  ASSERT_INT_EQ(multi_pm_test_node_count(ctx, outer_nested_call->src[0]), 14);

  PolyUOp *nested_kernel_graph = poly_get_kernel_graph(ctx, nested_body);
  ASSERT_NOT_NULL(nested_kernel_graph);
  PolyKernelScheduleResult nested_schedule =
      poly_build_kernel_schedule_from_kernel_graph(ctx, nested_kernel_graph);
  ASSERT_INT_EQ(nested_schedule.n_kernels, 3);
  ASSERT_INT_EQ(nested_schedule.kernel_kinds[0], POLY_KERNEL_ITEM_COPY);
  ASSERT_INT_EQ(nested_schedule.kernel_kinds[1], POLY_KERNEL_ITEM_COPY);
  ASSERT_INT_EQ(nested_schedule.kernel_kinds[2], POLY_KERNEL_ITEM_COMPUTE);
  ASSERT_INT_EQ(nested_schedule.kernel_n_params[0], 2);
  ASSERT_INT_EQ(nested_schedule.kernel_n_params[1], 2);
  ASSERT_INT_EQ(nested_schedule.kernel_n_params[2], 3);
  poly_kernel_schedule_result_free(&nested_schedule);

  PolyUOp *scheduled_out = NULL;
  PolySchedule *schedule = poly_schedule_with_vars(ctx, &root, 1, &scheduled_out);
  ASSERT_NOT_NULL(schedule);
  ASSERT_NOT_NULL(scheduled_out);
  ASSERT_INT_EQ(schedule->template->n_calls, 8);
  PolyOps expected_bodies[8] = {
      POLY_OP_SINK, POLY_OP_SINK, POLY_OP_COPY, POLY_OP_SINK,
      POLY_OP_COPY, POLY_OP_COPY, POLY_OP_SINK, POLY_OP_SINK};
  for (int i = 0; i < 8; i++) {
    PolyUOp *call = poly_schedule_call(schedule, i);
    ASSERT_NOT_NULL(call);
    ASSERT_INT_EQ(call->op, POLY_OP_CALL);
    ASSERT_TRUE(call->n_src >= 1);
    ASSERT_INT_EQ(call->src[0]->op, expected_bodies[i]);
  }
  ASSERT_INT_EQ(poly_schedule_call(schedule, 4)->src[2]->op, POLY_OP_MSELECT);
  ASSERT_INT_EQ(poly_schedule_call(schedule, 5)->src[2]->op, POLY_OP_MSELECT);
  ASSERT_INT_EQ(poly_schedule_call(schedule, 6)->src[2]->op, POLY_OP_MSTACK);
  ASSERT_INT_EQ(poly_schedule_call(schedule, 6)->src[3]->op, POLY_OP_MSTACK);

  const float expected[4] = {2824, 2828, 2832, 2836};
  ASSERT_INT_EQ(poly_run_schedule(ctx, schedule, NULL, 0), 0);
  PolyBuffer *result = poly_uop_buffer_handle(ctx, scheduled_out);
  ASSERT_NOT_NULL(result);
  ASSERT_TRUE(poly_buffer_is_multi(result));
  ASSERT_INT_EQ(result->n_bufs, 2);
  for (int lane = 0; lane < 2; lane++) {
    PolyBuffer *child = poly_buffer_multi_child(result, lane);
    ASSERT_NOT_NULL(child);
    ASSERT_NOT_NULL(child->ptr);
    ASSERT_TRUE(child->valid);
    ASSERT_INT_EQ(child->nbytes, (int64_t)sizeof(expected));
    for (int i = 0; i < 4; i++)
      ASSERT_FLOAT_EQ(((float *)child->ptr)[i], expected[i], 0.0f);
  }

  PolyCompiledSchedule *compiled = poly_lower_schedule(ctx, schedule, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(compiled);
  ASSERT_INT_EQ(compiled->template->n_calls, 8);
  ASSERT_NOT_NULL(compiled->linear);
  PolyOps expected_compiled_bodies[8] = {
      POLY_OP_PROGRAM, POLY_OP_PROGRAM, POLY_OP_COPY, POLY_OP_SINK,
      POLY_OP_COPY, POLY_OP_COPY, POLY_OP_SINK, POLY_OP_SINK};
  for (int i = 0; i < 8; i++)
    ASSERT_INT_EQ(compiled->linear->src[i]->src[0]->op, expected_compiled_bodies[i]);
  for (int lane = 0; lane < 2; lane++) {
    PolyBuffer *child = poly_buffer_multi_child(result, lane);
    memset(child->ptr, 0, child->nbytes);
    child->valid = false;
  }
  ASSERT_INT_EQ(poly_run_compiled_schedule(compiled, NULL, 0, NULL, 0), 0);
  for (int lane = 0; lane < 2; lane++) {
    PolyBuffer *child = poly_buffer_multi_child(result, lane);
    ASSERT_NOT_NULL(child);
    ASSERT_TRUE(child->valid);
    for (int i = 0; i < 4; i++)
      ASSERT_FLOAT_EQ(((float *)child->ptr)[i], expected[i], 0.0f);
  }

  poly_compiled_schedule_free(compiled);
  poly_schedule_free(schedule);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, multi_pm_broadcast_and_select_match_pinned_topology) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *cpu = poly_device_uop_from_name(ctx, "CPU");
  PolyUOp *cpu1 = poly_device_uop_from_name(ctx, "CPU:1");
  const char *names[] = {"CPU", "CPU:1"};
  PolyUOp *tuple = poly_device_uop_from_names(ctx, names, 2);
  PolyUOp *unique = poly_uop0(ctx, POLY_OP_UNIQUE, POLY_VOID, poly_arg_int(301));
  PolyUOp *buffer_src[] = {unique, cpu};
  PolyUOp *buffer = poly_uop(ctx, POLY_OP_BUFFER, POLY_FLOAT32, buffer_src, 2, poly_arg_int(4));
  PolyUOp *copy_src[] = {buffer, tuple};
  PolyUOp *broadcast = poly_uop(ctx, POLY_OP_COPY, POLY_FLOAT32, copy_src, 2, poly_arg_none());

  PolyUOp *stack = poly_apply_multi_pm(ctx, broadcast);
  ASSERT_NOT_NULL(stack);
  ASSERT_INT_EQ(stack->op, POLY_OP_MSTACK);
  ASSERT_INT_EQ(stack->n_src, 2);
  ASSERT_INT_EQ(stack->src[0]->op, POLY_OP_COPY);
  ASSERT_INT_EQ(stack->src[1]->op, POLY_OP_COPY);
  ASSERT_PTR_EQ(stack->src[0]->src[0], buffer);
  ASSERT_PTR_EQ(stack->src[1]->src[0], buffer);
  ASSERT_PTR_EQ(stack->src[0]->src[1], cpu);
  ASSERT_PTR_EQ(stack->src[1]->src[1], cpu1);

  PolyUOp *select = poly_uop1(ctx, POLY_OP_MSELECT, POLY_FLOAT32, stack, poly_arg_int(1));
  ASSERT_PTR_EQ(poly_apply_multi_pm(ctx, select), stack->src[1]);

  /* Pinned COPY_TO_ONE creates MSELECT(source, 0), then the same fixed-point
   * pass resolves MSELECT(MSTACK) to the exact first occurrence. */
  PolyUOp *to_one_src[] = {stack, cpu1};
  PolyUOp *to_one = poly_uop(ctx, POLY_OP_COPY, POLY_FLOAT32, to_one_src, 2, poly_arg_none());
  PolyUOp *to_one_rewritten = poly_apply_multi_pm(ctx, to_one);
  ASSERT_NOT_NULL(to_one_rewritten);
  ASSERT_INT_EQ(to_one_rewritten->op, POLY_OP_COPY);
  ASSERT_PTR_EQ(to_one_rewritten->src[0], stack->src[0]);
  ASSERT_PTR_EQ(to_one_rewritten->src[1], cpu1);

  PolyUOp *bad_select = poly_uop1(ctx, POLY_OP_MSELECT, POLY_FLOAT32, stack, poly_arg_int(2));
  ASSERT_PTR_EQ(poly_apply_multi_pm(ctx, bad_select), bad_select);

  /* Pinned graph_rewrite(..., enter_calls=False) leaves the CALL body opaque
   * while rewriting caller-visible arguments. */
  PolyUOp *body = poly_sink1(ctx, broadcast);
  PolyUOp *call_src[] = {body, broadcast};
  PolyUOp *call = poly_uop(ctx, POLY_OP_CALL, POLY_VOID, call_src, 2, poly_arg_none());
  PolyUOp *rewritten_call = poly_apply_multi_pm(ctx, call);
  ASSERT_NOT_NULL(rewritten_call);
  ASSERT_PTR_EQ(rewritten_call->src[0], body);
  ASSERT_INT_EQ(rewritten_call->src[1]->op, POLY_OP_MSTACK);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, multi_pm_movement_param_and_nested_call_match_pinned) {
  /* Pinned schedule/multi.py:80-113,138-152 moves every supported movement
   * through MULTI and turns an axis-bearing public PARAM into a local PARAM
   * wrapped by MULTI. schedule/__init__.py:80-90 then resolves PARAM leaves
   * recursively inside an MSTACK CALL argument. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  poly_ctx_set_preferred_device(ctx, POLY_DEVICE_CPU);

  const char *devices[2] = {"CPU", "CPU:1"};
  PolyUOp *shape_src[2] = {poly_const_int(ctx, 4), poly_const_int(ctx, 4)};
  PolyUOp *param_shape = poly_uop(
      ctx, POLY_OP_STACK, poly_dtype_vec(POLY_INDEX, 2), shape_src, 2,
      poly_arg_none());
  PolyParamArg param_arg = {
      .slot = 1,
      .addrspace = POLY_ADDR_GLOBAL,
      .axis = 0,
      .has_axis = true,
      .devices = devices,
      .n_devices = 2,
      .device_is_tuple = true,
  };
  PolyUOp *public_param = poly_uop1(
      ctx, POLY_OP_PARAM, POLY_FLOAT32, param_shape, poly_arg_param(&param_arg));
  PolyUOp *local_param = poly_apply_multi_pm(ctx, public_param);
  ASSERT_NOT_NULL(local_param);
  ASSERT_INT_EQ(local_param->op, POLY_OP_MULTI);
  ASSERT_INT_EQ(local_param->arg.kind, POLY_ARG_INT);
  ASSERT_INT_EQ(local_param->arg.i, 0);
  ASSERT_INT_EQ(local_param->n_src, 1);
  ASSERT_INT_EQ(local_param->src[0]->op, POLY_OP_PARAM);
  ASSERT_TRUE(local_param->src[0]->arg.kind == POLY_ARG_PARAM &&
              local_param->src[0]->arg.param);
  ASSERT_FALSE(local_param->src[0]->arg.param->has_axis);
  PolyShape local_param_shape = poly_uop_max_shape_cached(ctx, local_param->src[0]);
  ASSERT_INT_EQ(local_param_shape.ndim, 2);
  ASSERT_INT_EQ(local_param_shape.dims[0], 2);
  ASSERT_INT_EQ(local_param_shape.dims[1], 4);

  PolyUOp *reshape = poly_reshape(
      ctx, multi_pm_test_shard(ctx, 920, POLY_FLOAT32, 0),
      (int64_t[]){2, 2, 4}, 3);
  PolyUOp *expand = poly_expand(
      ctx, multi_pm_test_axis0_column_shard(ctx, 921, POLY_FLOAT32),
      (int64_t[]){4, 3}, 2);
  PolyUOp *pad = poly_pad(
      ctx, multi_pm_test_shard(ctx, 922, POLY_FLOAT32, 0),
      (int64_t[][2]){{0, 0}, {1, 1}}, 2);
  PolyUOp *permute = poly_permute(
      ctx, multi_pm_test_shard(ctx, 923, POLY_FLOAT32, 0),
      (int64_t[]){1, 0}, 2);
  PolyUOp *shrink = poly_shrink(
      ctx, multi_pm_test_shard(ctx, 924, POLY_FLOAT32, 0),
      (int64_t[][2]){{0, 4}, {1, 3}}, 2);
  PolyUOp *flip = poly_flip(
      ctx, multi_pm_test_shard(ctx, 925, POLY_FLOAT32, 0),
      (int64_t[]){1}, 1);
  PolyUOp *movement_roots[6] = {reshape, expand, pad, permute, shrink, flip};
  PolyOps local_ops[6] = {
      POLY_OP_RESHAPE, POLY_OP_EXPAND, POLY_OP_PAD,
      POLY_OP_PERMUTE, POLY_OP_MSTACK, POLY_OP_FLIP};
  int64_t expected_axes[6] = {0, 0, 0, 1, 0, 0};
  for (int i = 0; i < 6; i++) {
    PolyUOp *moved = poly_apply_multi_pm(ctx, movement_roots[i]);
    ASSERT_NOT_NULL(moved);
    ASSERT_INT_EQ(moved->op, POLY_OP_MULTI);
    ASSERT_INT_EQ(moved->arg.kind, POLY_ARG_INT);
    ASSERT_INT_EQ(moved->arg.i, expected_axes[i]);
    ASSERT_INT_EQ(moved->n_src, 1);
    ASSERT_INT_EQ(moved->src[0]->op, local_ops[i]);
  }

  PolyUOp *partition = poly_shrink(
      ctx, multi_pm_test_shard(ctx, 926, POLY_FLOAT32, 0),
      (int64_t[][2]){{2, 4}, {0, 4}}, 2);
  PolyUOp *partition_result = poly_apply_multi_pm(ctx, partition);
  ASSERT_NOT_NULL(partition_result);
  ASSERT_INT_EQ(partition_result->op, POLY_OP_MSTACK);
  ASSERT_INT_EQ(count_ops(ctx, partition_result, POLY_OP_MSELECT), 0);

  PolyUOp *cross_partition = poly_shrink(
      ctx, multi_pm_test_shard(ctx, 927, POLY_FLOAT32, 0),
      (int64_t[][2]){{1, 3}, {0, 4}}, 2);
  PolyUOp *shard_pad = poly_pad(
      ctx, multi_pm_test_shard(ctx, 928, POLY_FLOAT32, 0),
      (int64_t[][2]){{1, 0}, {0, 0}}, 2);
  PolyUOp *shard_flip = poly_flip(
      ctx, multi_pm_test_shard(ctx, 929, POLY_FLOAT32, 0),
      (int64_t[]){0}, 1);
  ASSERT_TRUE(poly_apply_multi_pm(ctx, cross_partition) == NULL);
  ASSERT_TRUE(poly_apply_multi_pm(ctx, shard_pad) == NULL);
  ASSERT_TRUE(poly_apply_multi_pm(ctx, shard_flip) == NULL);

  /* The original indexed MSTACK shape, not its scalarized rangeify
   * replacement, supplies strides. This is the exact numerical canary for
   * `_apply_reshape((2,4),(1,2,4), ...)` in schedule/indexing.py:113-127. */
  PolyUOp *source = multi_pm_test_find_unique_buffer(ctx, reshape, 920);
  ASSERT_NOT_NULL(source);
  float input[16];
  for (int i = 0; i < 16; i++) input[i] = (float)i;
  ASSERT_INT_EQ(poly_buffer_allocate(ctx, source, POLY_DEVICE_CPU), 0);
  ASSERT_INT_EQ(poly_buffer_copyin(ctx, source, input, sizeof(input)), 0);
  PolyUOp *scheduled_root = poly_uop1(
      ctx, POLY_OP_CONTIGUOUS, reshape->dtype, reshape, poly_arg_none());
  PolyUOp *scheduled_out = NULL;
  PolySchedule *schedule = poly_schedule_with_vars(ctx, &scheduled_root, 1, &scheduled_out);
  ASSERT_NOT_NULL(schedule);
  ASSERT_NOT_NULL(scheduled_out);
  ASSERT_TRUE(schedule->template->n_calls >= 4);
  PolyUOp *final_call = poly_schedule_call(schedule, schedule->template->n_calls - 1);
  ASSERT_NOT_NULL(final_call);
  ASSERT_INT_EQ(final_call->n_src, 3);
  ASSERT_INT_EQ(final_call->src[2]->op, POLY_OP_MSTACK);
  ASSERT_INT_EQ(final_call->src[2]->n_src, 2);
  ASSERT_INT_EQ(count_ops(ctx, final_call->src[2], POLY_OP_PARAM), 0);
  ASSERT_INT_EQ(poly_run_schedule(ctx, schedule, NULL, 0), 0);
  PolyBuffer *output = poly_uop_buffer_handle(ctx, scheduled_out);
  ASSERT_NOT_NULL(output);
  ASSERT_TRUE(poly_buffer_is_multi(output));
  ASSERT_INT_EQ(output->n_bufs, 2);
  for (int lane = 0; lane < 2; lane++) {
    PolyBuffer *child = poly_buffer_multi_child(output, lane);
    ASSERT_NOT_NULL(child);
    ASSERT_NOT_NULL(child->ptr);
    ASSERT_INT_EQ(child->nbytes, 8 * (int)sizeof(float));
    for (int i = 0; i < 8; i++)
      ASSERT_FLOAT_EQ(((float *)child->ptr)[i], input[lane * 8 + i], 0.0f);
  }
  poly_schedule_free(schedule);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, multi_pm_function_gettuple_passthrough_matches_pinned) {
  /* Pinned schedule/multi.py:32-34,124-125,127-136,158-174 and
   * schedule/rangeify.py:138-155. This closes the value FUNCTION selector,
   * local passthrough, and opaque caller-argument dependency group with exact
   * topology plus an executed two-device numerical result. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  poly_ctx_set_preferred_device(ctx, POLY_DEVICE_CPU);
  const char *devices[2] = {"CPU", "CPU:1"};

  PolyUOp *local0 = multi_pm_test_param(ctx, 10, "CPU", NULL, 0, false);
  PolyUOp *local1 = multi_pm_test_param(ctx, 11, "CPU:1", NULL, 0, false);
  PolyUOp *stack_src[2] = {local0, local1};
  PolyUOp *stack = poly_uop(
      ctx, POLY_OP_MSTACK, POLY_FLOAT32, stack_src, 2, poly_arg_none());
  PolyUOp *multi =
      poly_uop1(ctx, POLY_OP_MULTI, POLY_FLOAT32, stack, poly_arg_int(0));

  PolyUOp *moved = poly_reshape(ctx, stack, (int64_t[]){4, 2}, 2);
  PolyUOp *selected = poly_uop1(
      ctx, POLY_OP_MSELECT, POLY_FLOAT32, moved, poly_arg_int(1));
  PolyUOp *selected_result = poly_apply_multi_pm(ctx, selected);
  ASSERT_NOT_NULL(selected_result);
  ASSERT_INT_EQ(selected_result->op, POLY_OP_RESHAPE);
  ASSERT_PTR_EQ(selected_result->src[0], local1);
  ASSERT_INT_EQ(count_ops(ctx, selected_result, POLY_OP_MSELECT), 0);
  ASSERT_INT_EQ(count_ops(ctx, selected_result, POLY_OP_MSTACK), 0);

  PolyUOp *plain_tuple_src[2] = {local0, local1};
  PolyUOp *plain_tuple = poly_uop(
      ctx, POLY_OP_TUPLE, POLY_VOID, plain_tuple_src, 2, poly_arg_none());
  PolyUOp *plain_get = poly_uop1(
      ctx, POLY_OP_GETTUPLE, POLY_FLOAT32, plain_tuple, poly_arg_int(1));
  ASSERT_PTR_EQ(poly_apply_multi_pm(ctx, plain_get), local1);

  PolyUOp *tuple_body = poly_uop1(
      ctx, POLY_OP_TUPLE, POLY_VOID, stack, poly_arg_none());
  PolyUOp *tuple_multi =
      poly_uop1(ctx, POLY_OP_MULTI, POLY_VOID, tuple_body, poly_arg_int(0));
  PolyUOp *tuple_get = poly_uop1(
      ctx, POLY_OP_GETTUPLE, POLY_FLOAT32, tuple_multi, poly_arg_int(0));
  PolyUOp *tuple_get_result = poly_apply_multi_pm(ctx, tuple_get);
  ASSERT_NOT_NULL(tuple_get_result);
  ASSERT_INT_EQ(tuple_get_result->op, POLY_OP_MULTI);
  ASSERT_INT_EQ(tuple_get_result->arg.i, 0);
  ASSERT_PTR_EQ(tuple_get_result->src[0], stack);

  PolyUOp *axis_param = multi_pm_test_param(ctx, 0, NULL, devices, 2, true);
  PolyUOp *body_values[2] = {
      poly_alu2(ctx, POLY_OP_ADD, axis_param,
                poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(1.0))),
      axis_param};
  PolyUOp *body = poly_uop(
      ctx, POLY_OP_TUPLE, POLY_VOID, body_values, 2, poly_arg_none());
  PolyUOp *function_src[2] = {body, axis_param};
  PolyUOp *function = poly_uop(
      ctx, POLY_OP_FUNCTION, POLY_VOID, function_src, 2,
      poly_arg_str("multi_value"));
  PolyUOp *function_result = poly_apply_multi_pm(ctx, function);
  ASSERT_NOT_NULL(function_result);
  ASSERT_INT_EQ(function_result->op, POLY_OP_TUPLE);
  ASSERT_INT_EQ(function_result->n_src, 2);
  for (int i = 0; i < 2; i++) {
    ASSERT_INT_EQ(function_result->src[i]->op, POLY_OP_MULTI);
    ASSERT_INT_EQ(function_result->src[i]->arg.i, 0);
    ASSERT_INT_EQ(function_result->src[i]->src[0]->op, POLY_OP_GETTUPLE);
    ASSERT_INT_EQ(function_result->src[i]->src[0]->arg.i, i);
    ASSERT_INT_EQ(function_result->src[i]->src[0]->src[0]->op, POLY_OP_FUNCTION);
  }
  ASSERT_INT_EQ(count_ops(ctx, function_result, POLY_OP_FUNCTION), 1);
  ASSERT_INT_EQ(count_ops(ctx, function_result, POLY_OP_MULTI), 2);

  PolyOps wrappers[3] = {
      POLY_OP_CAST, POLY_OP_CONTIGUOUS, POLY_OP_DETACH};
  for (int i = 0; i < 3; i++) {
    PolyUOp *wrapped = poly_uop1(
        ctx, wrappers[i], POLY_FLOAT32, multi, poly_arg_none());
    PolyUOp *wrapped_result = poly_apply_multi_pm(ctx, wrapped);
    ASSERT_NOT_NULL(wrapped_result);
    ASSERT_INT_EQ(wrapped_result->op, POLY_OP_MULTI);
    ASSERT_INT_EQ(wrapped_result->arg.i, 0);
    ASSERT_INT_EQ(wrapped_result->src[0]->op, wrappers[i]);
    ASSERT_PTR_EQ(wrapped_result->src[0]->src[0], stack);
  }

  PolyUOp *effect = poly_sink1(ctx, local0);
  PolyUOp *after = poly_uop2(
      ctx, POLY_OP_AFTER, POLY_FLOAT32, multi, effect, poly_arg_none());
  PolyUOp *after_result = poly_apply_multi_pm(ctx, after);
  ASSERT_NOT_NULL(after_result);
  ASSERT_INT_EQ(after_result->op, POLY_OP_MULTI);
  ASSERT_INT_EQ(after_result->src[0]->op, POLY_OP_AFTER);
  ASSERT_PTR_EQ(after_result->src[0]->src[0], stack);

  PolyUOp *call_src[3] = {effect, multi, local1};
  PolyUOp *call = poly_uop(
      ctx, POLY_OP_CALL, POLY_VOID, call_src, 3, poly_arg_str("void_multi"));
  PolyUOp *call_result = poly_apply_multi_pm(ctx, call);
  ASSERT_NOT_NULL(call_result);
  ASSERT_INT_EQ(call_result->op, POLY_OP_CALL);
  ASSERT_PTR_EQ(call_result->src[0], effect);
  ASSERT_PTR_EQ(call_result->src[1], stack);
  ASSERT_INT_EQ(count_ops(ctx, call_result, POLY_OP_MULTI), 0);

  PolyUOp *store = poly_uop2(
      ctx, POLY_OP_STORE, POLY_VOID, multi, multi, poly_arg_none());
  PolyUOp *store_result = poly_apply_multi_pm(ctx, store);
  ASSERT_NOT_NULL(store_result);
  ASSERT_INT_EQ(store_result->op, POLY_OP_STORE);
  ASSERT_PTR_EQ(store_result->src[0], stack);
  ASSERT_PTR_EQ(store_result->src[1], stack);
  ASSERT_INT_EQ(count_ops(ctx, store_result, POLY_OP_MULTI), 0);

  float lhs_values[16], rhs_values[16];
  for (int i = 0; i < 16; i++) {
    lhs_values[i] = (float)i;
    rhs_values[i] = (float)(i + 16);
  }
  PolyUOp *lhs = multi_pm_test_realized_multi(ctx, 1800, lhs_values);
  PolyUOp *rhs = multi_pm_test_realized_multi(ctx, 1810, rhs_values);
  ASSERT_NOT_NULL(lhs);
  ASSERT_NOT_NULL(rhs);
  PolyUOp *p0 = multi_pm_test_param(ctx, 0, NULL, devices, 2, true);
  PolyUOp *p1 = multi_pm_test_param(ctx, 1, NULL, devices, 2, true);
  PolyUOp *sum = poly_alu2(ctx, POLY_OP_ADD, p0, p1);
  PolyUOp *sum_body = poly_uop1(
      ctx, POLY_OP_TUPLE, POLY_VOID, sum, poly_arg_none());
  PolyUOp *value_src[3] = {sum_body, lhs, rhs};
  PolyUOp *value_function = poly_uop(
      ctx, POLY_OP_FUNCTION, POLY_VOID, value_src, 3,
      poly_arg_str("value_add"));
  PolyUOp *value = poly_uop1(
      ctx, POLY_OP_GETTUPLE, POLY_FLOAT32, value_function, poly_arg_int(0));
  PolyUOp *requested = poly_uop1(
      ctx, POLY_OP_CONTIGUOUS, POLY_FLOAT32, value, poly_arg_none());
  PolyUOp *post_multi = poly_apply_multi_pm(ctx, requested);
  ASSERT_NOT_NULL(post_multi);
  ASSERT_INT_EQ(post_multi->op, POLY_OP_MULTI);
  ASSERT_INT_EQ(post_multi->arg.i, 0);
  ASSERT_INT_EQ(count_ops(ctx, post_multi, POLY_OP_FUNCTION), 1);
  ASSERT_INT_EQ(count_ops(ctx, post_multi, POLY_OP_GETTUPLE), 1);
  ASSERT_INT_EQ(count_ops(ctx, post_multi, POLY_OP_MSTACK), 2);
  PolyShape post_shape = poly_uop_max_shape_cached(ctx, post_multi);
  ASSERT_INT_EQ(post_shape.ndim, 2);
  ASSERT_INT_EQ(post_shape.dims[0], 4);
  ASSERT_INT_EQ(post_shape.dims[1], 4);

  PolyUOp *scheduled_out = NULL;
  PolySchedule *schedule =
      poly_schedule_with_vars(ctx, &requested, 1, &scheduled_out);
  ASSERT_NOT_NULL(schedule);
  ASSERT_NOT_NULL(scheduled_out);
  ASSERT_INT_EQ(schedule->template->n_calls, 1);
  ASSERT_INT_EQ(poly_run_schedule(ctx, schedule, NULL, 0), 0);
  PolyBuffer *output = poly_uop_buffer_handle(ctx, scheduled_out);
  ASSERT_NOT_NULL(output);
  ASSERT_TRUE(poly_buffer_is_multi(output));
  ASSERT_INT_EQ(output->n_bufs, 2);
  for (int lane = 0; lane < 2; lane++) {
    PolyBuffer *child = poly_buffer_multi_child(output, lane);
    ASSERT_NOT_NULL(child);
    ASSERT_NOT_NULL(child->ptr);
    ASSERT_INT_EQ(child->nbytes, 8 * (int)sizeof(float));
    for (int i = 0; i < 8; i++) {
      float expected = lhs_values[lane * 8 + i] + rhs_values[lane * 8 + i];
      ASSERT_FLOAT_EQ(((float *)child->ptr)[i], expected, 0.0f);
    }
  }

  poly_schedule_free(schedule);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, multi_pm_moves_shrink_before_mstack_like_pinned) {
  /* Pinned schedule/multi.py:8-19,30-31 substitutes `_device_num` for each
   * ordered MSTACK occurrence, reconstructs a local SHRINK, and preserves a
   * COPY destination or appends CONTIGUOUS for a non-COPY child. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  poly_ctx_set_preferred_device(ctx, POLY_DEVICE_CPU);
  PolyUOp *cpu = poly_device_uop_from_name(ctx, "CPU");
  PolyUOp *cpu1 = poly_device_uop_from_name(ctx, "CPU:1");
  const char *names[] = {"CPU", "CPU:1"};
  PolyUOp *tuple = poly_device_uop_from_names(ctx, names, 2);
  PolyUOp *unique = poly_uop0(ctx, POLY_OP_UNIQUE, POLY_VOID, poly_arg_int(331));
  PolyUOp *buffer_src[] = {unique, cpu};
  PolyUOp *buffer = poly_uop(ctx, POLY_OP_BUFFER, POLY_INT32, buffer_src, 2, poly_arg_int(8));
  int32_t input_values[] = {0, 1, 2, 3, 4, 5, 6, 7};
  ASSERT_INT_EQ(poly_buffer_allocate(ctx, buffer, POLY_DEVICE_CPU), 0);
  ASSERT_INT_EQ(poly_buffer_copyin(ctx, buffer, input_values, sizeof(input_values)), 0);
  PolyUOp *value = poly_reshape(ctx, buffer, (int64_t[]){4, 2}, 2);
  PolyUOp *copy_src[] = {value, tuple};
  PolyUOp *broadcast = poly_uop(ctx, POLY_OP_COPY, POLY_INT32, copy_src, 2, poly_arg_none());
  PolyUOp *dvar = poly_uop0(
      ctx, POLY_OP_DEFINE_VAR, POLY_INDEX, poly_arg_define_var("_device_num", 0, 1));
  PolyUOp *two = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(2));
  PolyUOp *starts[] = {
      poly_alu2(ctx, POLY_OP_MUL, dvar, two),
      poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(0))};
  PolyUOp *sizes[] = {two, two};
  PolyUOp *local = poly_shrink_uop(ctx, broadcast, starts, sizes, 2);
  PolyUOp *multi = poly_uop1(ctx, POLY_OP_MULTI, POLY_INT32, local, poly_arg_int(0));
  PolyUOp *rewritten = poly_apply_multi_pm(ctx, multi);
  ASSERT_NOT_NULL(rewritten);
  ASSERT_INT_EQ(rewritten->op, POLY_OP_MULTI);
  ASSERT_INT_EQ(rewritten->arg.kind, POLY_ARG_INT);
  ASSERT_INT_EQ(rewritten->arg.i, 0);
  ASSERT_INT_EQ(rewritten->n_src, 1);
  PolyUOp *stack = rewritten->src[0];
  ASSERT_INT_EQ(stack->op, POLY_OP_MSTACK);
  ASSERT_INT_EQ(stack->n_src, 2);
  ASSERT_INT_EQ(count_ops(ctx, rewritten, POLY_OP_SHRINK), 2);
  ASSERT_INT_EQ(count_ops(ctx, rewritten, POLY_OP_COPY), 2);
  ASSERT_INT_EQ(count_ops(ctx, rewritten, POLY_OP_DEFINE_VAR), 0);
  ASSERT_INT_EQ(count_ops(ctx, rewritten, POLY_OP_MUL), 0);
  PolyUOp *expected_devices[] = {cpu, cpu1};
  int64_t expected_start0[] = {0, 2};
  for (int i = 0; i < 2; i++) {
    PolyUOp *copy = stack->src[i];
    ASSERT_INT_EQ(copy->op, POLY_OP_COPY);
    ASSERT_INT_EQ(copy->n_src, 2);
    ASSERT_PTR_EQ(copy->src[1], expected_devices[i]);
    PolyUOp *shrink = copy->src[0];
    ASSERT_INT_EQ(shrink->op, POLY_OP_SHRINK);
    ASSERT_INT_EQ(shrink->n_src, 3);
    ASSERT_PTR_EQ(shrink->src[0], value);
    ASSERT_INT_EQ(shrink->src[1]->op, POLY_OP_STACK);
    ASSERT_INT_EQ(shrink->src[2]->op, POLY_OP_STACK);
    ASSERT_INT_EQ(shrink->src[1]->n_src, 2);
    ASSERT_INT_EQ(shrink->src[2]->n_src, 2);
    ASSERT_INT_EQ(shrink->src[1]->src[0]->op, POLY_OP_CONST);
    ASSERT_INT_EQ(shrink->src[1]->src[0]->arg.kind, POLY_ARG_INT);
    ASSERT_INT_EQ(shrink->src[1]->src[0]->arg.i, expected_start0[i]);
    ASSERT_INT_EQ(shrink->src[1]->src[1]->arg.i, 0);
    ASSERT_INT_EQ(shrink->src[2]->src[0]->arg.i, 2);
    ASSERT_INT_EQ(shrink->src[2]->src[1]->arg.i, 2);
  }
  PolyShape global_shape = poly_uop_max_shape(ctx, rewritten);
  PolyShape local_shape = poly_uop_max_shape(ctx, stack);
  ASSERT_INT_EQ(global_shape.ndim, 2);
  ASSERT_INT_EQ(global_shape.dims[0], 4);
  ASSERT_INT_EQ(global_shape.dims[1], 2);
  ASSERT_INT_EQ(local_shape.ndim, 2);
  ASSERT_INT_EQ(local_shape.dims[0], 2);
  ASSERT_INT_EQ(local_shape.dims[1], 2);
  free(global_shape.dims);
  free(local_shape.dims);

  /* Pinned UOp.empty_like allocates one local shard per tuple device and
   * preserves MULTI(axis=0) around the callified output target. */
  PolyUOp *callified_out = NULL;
  PolyUOp *callified = poly_transform_to_call(ctx, &multi, 1, &callified_out);
  ASSERT_NOT_NULL(callified);
  ASSERT_INT_EQ(callified->op, POLY_OP_CALL);
  ASSERT_NOT_NULL(callified_out);
  ASSERT_INT_EQ(callified_out->op, POLY_OP_MULTI);
  ASSERT_INT_EQ(callified_out->arg.kind, POLY_ARG_INT);
  ASSERT_INT_EQ(callified_out->arg.i, 0);
  ASSERT_INT_EQ(callified_out->n_src, 1);
  ASSERT_INT_EQ(callified_out->src[0]->op, POLY_OP_RESHAPE);
  const PolyUOp *callified_identity = poly_uop_get_buffer_identity(callified_out);
  ASSERT_NOT_NULL(callified_identity);
  ASSERT_INT_EQ(callified_identity->op, POLY_OP_BUFFER);
  ASSERT_INT_EQ(callified_identity->arg.kind, POLY_ARG_INT);
  ASSERT_INT_EQ(callified_identity->arg.i, 4);
  ASSERT_PTR_EQ(callified_identity->src[1], tuple);

  /* Pinned schedule/multi.py:122,153 moves a sharded assignment inside the
   * value MULTI: MULTI(AFTER(local_dest, STORE(local_dest, local_value))). */
  PolyUOp *post_call_multi = poly_apply_multi_pm(ctx, callified->src[0]);
  ASSERT_NOT_NULL(post_call_multi);
  ASSERT_INT_EQ(post_call_multi->op, POLY_OP_SINK);
  ASSERT_INT_EQ(post_call_multi->n_src, 1);
  PolyUOp *assigned_multi = post_call_multi->src[0];
  ASSERT_INT_EQ(assigned_multi->op, POLY_OP_MULTI);
  ASSERT_INT_EQ(assigned_multi->arg.kind, POLY_ARG_INT);
  ASSERT_INT_EQ(assigned_multi->arg.i, 0);
  ASSERT_INT_EQ(assigned_multi->n_src, 1);
  PolyUOp *local_after = assigned_multi->src[0];
  ASSERT_INT_EQ(local_after->op, POLY_OP_AFTER);
  ASSERT_INT_EQ(local_after->n_src, 2);
  ASSERT_INT_EQ(local_after->src[0]->op, POLY_OP_RESHAPE);
  ASSERT_INT_EQ(local_after->src[1]->op, POLY_OP_STORE);
  ASSERT_INT_EQ(local_after->src[1]->n_src, 2);
  ASSERT_PTR_EQ(local_after->src[1]->src[0], local_after->src[0]);
  ASSERT_INT_EQ(local_after->src[1]->src[1]->op, POLY_OP_MSTACK);
  ASSERT_INT_EQ(count_ops(ctx, post_call_multi, POLY_OP_MULTI), 1);
  ASSERT_INT_EQ(count_ops(ctx, post_call_multi, POLY_OP_STORE), 1);
  int post_call_n = 0;
  ASSERT_NOT_NULL(poly_toposort(ctx, post_call_multi, &post_call_n));
  ASSERT_INT_EQ(post_call_n, 25);

  /* Pinned ALWAYS_RUN_OPS retains the same-device NOOP materialization made
   * by COPY(x, CPU) -> NOOP(x), so axis-0 sharding has two CPU slice kernels,
   * one CPU:1 transfer, and one tuple CALL (rangeify.py:184-187,216,250).
   * Inlining NOOP here silently changes the exact schedule from four CALLs
   * to three even though the final values can remain equal. */
  PolyUOp *multi_kernel_graph = poly_get_kernel_graph(ctx, post_call_multi);
  ASSERT_NOT_NULL(multi_kernel_graph);
  PolyKernelScheduleResult multi_schedule =
      poly_build_kernel_schedule_from_kernel_graph(ctx, multi_kernel_graph);
  ASSERT_INT_EQ(multi_schedule.n_kernels, 4);

  /* Pinned to_define_global debufs MSTACK/MSELECT as one buffer argument
   * (rangeify.py:497-503,528-535). The final tuple kernel therefore records
   * one output and one aggregate input; its body sees two PARAMs and no
   * MSTACK. CALL construction/runtime resolution is the following boundary. */
  ASSERT_INT_EQ(multi_schedule.kernel_n_params[3], 2);
  ASSERT_NOT_NULL(multi_schedule.param_to_buf[3]);
  ASSERT_INT_EQ(multi_schedule.param_to_buf[3][1]->op, POLY_OP_MSTACK);
  ASSERT_INT_EQ(count_ops(ctx, multi_schedule.kernels[3], POLY_OP_PARAM), 2);
  ASSERT_INT_EQ(count_ops(ctx, multi_schedule.kernels[3], POLY_OP_MSTACK), 0);
  poly_kernel_schedule_result_free(&multi_schedule);

  /* Pinned schedule/__init__.py:61-64 applies `_unwrap_src(s).buf_uop` to
   * CALL arguments, and UOp.buf_uop preserves MSTACK recursively
   * (uop/ops.py:800-807). The public LINEAR/replay boundary must retain that
   * aggregate argument instead of flattening it into slots or rejecting it. */
  PolyUOp *scheduled_out = NULL;
  PolySchedule *schedule = poly_schedule_with_vars(ctx, &multi, 1, &scheduled_out);
  ASSERT_NOT_NULL(schedule);
  ASSERT_NOT_NULL(scheduled_out);
  ASSERT_INT_EQ(scheduled_out->op, POLY_OP_MULTI);
  ASSERT_INT_EQ(schedule->template->n_calls, 4);
  PolyUOp *final_call = poly_schedule_call(schedule, 3);
  ASSERT_NOT_NULL(final_call);
  ASSERT_INT_EQ(final_call->op, POLY_OP_CALL);
  ASSERT_INT_EQ(final_call->n_src, 3);
  ASSERT_INT_EQ(final_call->src[1]->op, POLY_OP_BUFFER);
  ASSERT_PTR_EQ(final_call->src[1]->src[1], tuple);
  ASSERT_INT_EQ(final_call->src[2]->op, POLY_OP_MSTACK);
  ASSERT_INT_EQ(final_call->src[2]->n_src, 2);
  /* Pinned memory planning preserves MSTACK and replaces its scalar children
   * with planned SLICEs. Polygrad's reviewed BUFFER_VIEW spelling is the same
   * boundary: replacement must recurse into the aggregate instead of leaving
   * stale pre-plan BUFFER children. */
  ASSERT_INT_EQ(final_call->src[2]->src[0]->op, POLY_OP_BUFFER_VIEW);
  ASSERT_INT_EQ(final_call->src[2]->src[1]->op, POLY_OP_BUFFER_VIEW);
  ASSERT_STR_EQ(poly_uop_device_name(ctx, final_call->src[2]->src[0]), "CPU");
  ASSERT_STR_EQ(poly_uop_device_name(ctx, final_call->src[2]->src[1]), "CPU:1");
  PolyUOp *final_body = poly_schedule_call_body(schedule, 3);
  ASSERT_NOT_NULL(final_body);
  ASSERT_INT_EQ(count_ops(ctx, final_body, POLY_OP_PARAM), 2);
  ASSERT_INT_EQ(count_ops(ctx, final_body, POLY_OP_MSTACK), 0);

  /* Pinned exec_kernel resolves `[BUFFER, MSTACK]` into two ordered lanes and
   * executes one kernel per child (engine/realize.py:142-180). The first three
   * scalar CALLs plus the two final lanes are five executions. */
  ASSERT_INT_EQ(poly_run_schedule(ctx, schedule, NULL, 0), 0);
  ASSERT_INT_EQ(ctx->kernel_count, 5);
  PolyBuffer *scheduled_buffer = poly_uop_buffer_handle(ctx, scheduled_out);
  ASSERT_NOT_NULL(scheduled_buffer);
  ASSERT_TRUE(poly_buffer_is_multi(scheduled_buffer));
  ASSERT_INT_EQ(scheduled_buffer->n_bufs, 2);
  int32_t expected_lane_values[2][4] = {{0, 1, 2, 3}, {4, 5, 6, 7}};
  const char *expected_lane_devices[] = {"CPU", "CPU:1"};
  for (int lane = 0; lane < 2; lane++) {
    PolyBuffer *child = poly_buffer_multi_child(scheduled_buffer, lane);
    ASSERT_NOT_NULL(child);
    ASSERT_NOT_NULL(child->ptr);
    ASSERT_TRUE(child->valid);
    ASSERT_NOT_NULL(child->device_uop);
    ASSERT_INT_EQ(child->device_uop->arg.kind, POLY_ARG_STRING);
    ASSERT_STR_EQ(child->device_uop->arg.str, expected_lane_devices[lane]);
    ASSERT_INT_EQ(child->nbytes, 4 * (int)sizeof(int32_t));
    for (int i = 0; i < 4; i++)
      ASSERT_INT_EQ(((int32_t *)child->ptr)[i], expected_lane_values[lane][i]);
  }

  /* Pinned CapturedJit retains LINEAR and run_linear resolves MSTACK on each
   * invocation before runtime lookup (engine/jit.py:220-229;
   * engine/realize.py:142-180). Compiled replay must preserve the exact final
   * `[BUFFER, MSTACK(BUFFER_VIEW, BUFFER_VIEW)]` topology and execute both
   * lanes; scalar CALLs remain pre-lowered. */
  PolyCompiledSchedule *compiled = poly_lower_schedule(ctx, schedule, POLY_DEVICE_CPU);
  ASSERT_NOT_NULL(compiled);
  ASSERT_INT_EQ(compiled->template->n_calls, 4);
  PolyUOp *compiled_final = compiled->template->linear->src[3];
  ASSERT_INT_EQ(compiled_final->op, POLY_OP_CALL);
  ASSERT_INT_EQ(compiled_final->n_src, 3);
  ASSERT_INT_EQ(compiled_final->src[2]->op, POLY_OP_MSTACK);
  ASSERT_INT_EQ(compiled_final->src[2]->n_src, 2);
  ASSERT_INT_EQ(compiled_final->src[2]->src[0]->op, POLY_OP_BUFFER_VIEW);
  ASSERT_INT_EQ(compiled_final->src[2]->src[1]->op, POLY_OP_BUFFER_VIEW);
  poly_ctx_reset_counters(ctx);
  ASSERT_INT_EQ(poly_run_compiled_schedule(compiled, NULL, 0, NULL, 0), 0);
  ASSERT_INT_EQ(ctx->kernel_count, 5);
  for (int lane = 0; lane < 2; lane++) {
    PolyBuffer *child = poly_buffer_multi_child(scheduled_buffer, lane);
    ASSERT_NOT_NULL(child);
    ASSERT_TRUE(child->valid);
    ASSERT_STR_EQ(child->device_uop->arg.str, expected_lane_devices[lane]);
    for (int i = 0; i < 4; i++)
      ASSERT_INT_EQ(((int32_t *)child->ptr)[i], expected_lane_values[lane][i]);
  }
  poly_compiled_schedule_free(compiled);
  poly_schedule_free(schedule);

  /* The non-COPY branch materializes one local CONTIGUOUS per source and
   * ms.replace preserves both MSTACK metadata fields. */
  PolyUOp *unique1 = poly_uop0(ctx, POLY_OP_UNIQUE, POLY_VOID, poly_arg_int(332));
  PolyUOp *buffer1_src[] = {unique1, cpu1};
  PolyUOp *buffer1 = poly_uop(ctx, POLY_OP_BUFFER, POLY_INT32, buffer1_src, 2, poly_arg_int(8));
  PolyUOp *value1 = poly_reshape(ctx, buffer1, (int64_t[]){4, 2}, 2);
  PolyUOp *manual_src[] = {value, value1};
  PolyUOp *manual_stack = poly_uop_tagged_arg(
      ctx, POLY_OP_MSTACK, POLY_INT32, manual_src, 2, poly_arg_none(), 701,
      poly_arg_int(702));
  PolyUOp *manual_shrink = poly_shrink_uop(ctx, manual_stack, starts, sizes, 2);
  PolyUOp *manual_rewritten = poly_apply_multi_pm(ctx, manual_shrink);
  ASSERT_NOT_NULL(manual_rewritten);
  ASSERT_INT_EQ(manual_rewritten->op, POLY_OP_MSTACK);
  ASSERT_INT_EQ(manual_rewritten->tag, 701);
  ASSERT_INT_EQ(manual_rewritten->tag_arg.kind, POLY_ARG_INT);
  ASSERT_INT_EQ(manual_rewritten->tag_arg.i, 702);
  ASSERT_INT_EQ(manual_rewritten->n_src, 2);
  ASSERT_INT_EQ(count_ops(ctx, manual_rewritten, POLY_OP_SHRINK), 2);
  ASSERT_INT_EQ(count_ops(ctx, manual_rewritten, POLY_OP_CONTIGUOUS), 2);
  ASSERT_INT_EQ(count_ops(ctx, manual_rewritten, POLY_OP_DEFINE_VAR), 0);
  for (int i = 0; i < 2; i++) {
    ASSERT_INT_EQ(manual_rewritten->src[i]->op, POLY_OP_CONTIGUOUS);
    ASSERT_INT_EQ(manual_rewritten->src[i]->n_src, 1);
    ASSERT_INT_EQ(manual_rewritten->src[i]->src[0]->op, POLY_OP_SHRINK);
    ASSERT_PTR_EQ(manual_rewritten->src[i]->src[0]->src[0], manual_src[i]);
    ASSERT_INT_EQ(manual_rewritten->src[i]->src[0]->src[1]->src[0]->arg.i,
                  expected_start0[i]);
  }

  /* Without `_device_num`, pinned still duplicates the same static slice over
   * every occurrence. */
  PolyUOp *static_shrink =
      poly_shrink(ctx, manual_stack, (int64_t[][2]){{1, 3}, {0, 2}}, 2);
  PolyUOp *static_rewritten = poly_apply_multi_pm(ctx, static_shrink);
  ASSERT_NOT_NULL(static_rewritten);
  ASSERT_INT_EQ(static_rewritten->op, POLY_OP_MSTACK);
  ASSERT_INT_EQ(static_rewritten->n_src, 2);
  for (int i = 0; i < 2; i++) {
    PolyUOp *shrink = static_rewritten->src[i]->src[0];
    ASSERT_INT_EQ(static_rewritten->src[i]->op, POLY_OP_CONTIGUOUS);
    ASSERT_INT_EQ(shrink->op, POLY_OP_SHRINK);
    ASSERT_INT_EQ(shrink->src[1]->src[0]->arg.i, 1);
    ASSERT_INT_EQ(shrink->src[2]->src[0]->arg.i, 2);
  }

  /* graph_rewrite(..., enter_calls=False) keeps the function body opaque but
   * applies the same rule to a caller-visible argument. */
  PolyUOp *body = poly_sink1(ctx, manual_shrink);
  PolyUOp *call_src2[] = {body, manual_shrink};
  PolyUOp *call = poly_uop(ctx, POLY_OP_CALL, POLY_VOID, call_src2, 2, poly_arg_none());
  PolyUOp *call_rewritten = poly_apply_multi_pm(ctx, call);
  ASSERT_NOT_NULL(call_rewritten);
  ASSERT_PTR_EQ(call_rewritten->src[0], body);
  ASSERT_INT_EQ(call_rewritten->src[1]->op, POLY_OP_MSTACK);
  ASSERT_INT_EQ(count_ops(ctx, call_rewritten->src[1], POLY_OP_DEFINE_VAR), 0);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, multi_pm_cpu_broadcast_select_executes_like_pinned) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  float input[] = {1.0f, 2.0f, 3.0f, 4.0f};
  float output[] = {0.0f, 0.0f, 0.0f, 0.0f};
  PolyUOp *cpu = poly_device_uop_from_name(ctx, "CPU");
  PolyUOp *cpu1 = poly_device_uop_from_name(ctx, "CPU:1");
  PolyUOp *source_unique = poly_uop0(ctx, POLY_OP_UNIQUE, POLY_VOID, poly_arg_int(311));
  PolyUOp *dest_unique = poly_uop0(ctx, POLY_OP_UNIQUE, POLY_VOID, poly_arg_int(312));
  PolyUOp *source_src[] = {source_unique, cpu};
  PolyUOp *dest_src[] = {dest_unique, cpu1};
  PolyUOp *source = poly_uop(ctx, POLY_OP_BUFFER, POLY_FLOAT32, source_src, 2, poly_arg_int(4));
  PolyUOp *dest = poly_uop(ctx, POLY_OP_BUFFER, POLY_FLOAT32, dest_src, 2, poly_arg_int(4));
  ASSERT_INT_EQ(poly_buffer_allocate(ctx, source, POLY_DEVICE_CPU), 0);
  ASSERT_INT_EQ(poly_buffer_copyin(ctx, source, input, sizeof(input)), 0);

  const char *names[] = {"CPU", "CPU:1"};
  PolyUOp *tuple = poly_device_uop_from_names(ctx, names, 2);
  PolyUOp *copy_src[] = {source, tuple};
  PolyUOp *broadcast = poly_uop(ctx, POLY_OP_COPY, POLY_FLOAT32, copy_src, 2, poly_arg_none());
  PolyUOp *select = poly_uop1(ctx, POLY_OP_MSELECT, POLY_FLOAT32, broadcast, poly_arg_int(1));
  PolyUOp *store = poly_store_val(ctx, dest, select);
  PolyUOp *sink = poly_sink1(ctx, store);
  PolySchedule *schedule = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_NOT_NULL(schedule);
  ASSERT_INT_EQ(poly_run_schedule(ctx, schedule, NULL, 0), 0);
  ASSERT_INT_EQ(poly_buffer_copyout(ctx, dest, output, sizeof(output)), 0);
  for (int i = 0; i < 4; i++)
    ASSERT_FLOAT_EQ(output[i], input[i], 0.0f);

  poly_schedule_free(schedule);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, multi_pm_residual_tuple_const_copy_executes_like_pinned) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *cpu = poly_device_uop_from_name(ctx, "CPU");
  const char *names[] = {"CPU", "CPU:1"};
  PolyUOp *tuple = poly_device_uop_from_names(ctx, names, 2);
  PolyUOp *unique = poly_uop0(ctx, POLY_OP_UNIQUE, POLY_VOID, poly_arg_int(321));
  PolyUOp *dest_src[] = {unique, cpu};
  PolyUOp *dest = poly_uop(ctx, POLY_OP_BUFFER, POLY_FLOAT32, dest_src, 2, poly_arg_int(1));
  /* Pinned broadcast explicitly excludes CONST. Its tuple DEVICE survives
   * multi_pm, but callify schedules the enclosing STORE as one CPU CALL and
   * executes it successfully (schedule/multi.py:21-24). */
  PolyUOp *copy_src[] = {poly_const_float(ctx, 1.0), tuple};
  PolyUOp *copy = poly_uop(ctx, POLY_OP_COPY, POLY_FLOAT32, copy_src, 2, poly_arg_none());
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, dest, copy));
  PolySchedule *schedule = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_NOT_NULL(schedule);
  ASSERT_INT_EQ(schedule->template->n_calls, 1);
  ASSERT_INT_EQ(count_ops(ctx, poly_schedule_call_body(schedule, 0), POLY_OP_COPY), 0);
  ASSERT_INT_EQ(poly_run_schedule(ctx, schedule, NULL, 0), 0);
  float output = 0.0f;
  ASSERT_INT_EQ(poly_buffer_copyout(ctx, dest, &output, sizeof(output)), 0);
  ASSERT_FLOAT_EQ(output, 1.0f, 0.0f);
  poly_schedule_free(schedule);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, add_buffers_after_reuses_existing_assignment_buffer) {
  /* Pinned tinygrad rangeify.py:409-428 handles BUFFERIZE(AFTER(...)) as an
   * existing-buffer effect. It rebuilds and ends the owned STORE instead of
   * allocating a LUNIQUE materialization buffer. Keep two distinct ranges so
   * this also locks the deduplicated, axis-sorted range union. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *state = poly_buffer_on_device(ctx, POLY_FLOAT32, 6, POLY_DEVICE_CPU);
  PolyUOp *two = poly_const_int(ctx, 2);
  PolyUOp *three = poly_const_int(ctx, 3);
  PolyUOp *store_range =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_INDEX, two, poly_arg_range(7, POLY_AXIS_LOOP));
  PolyUOp *consumer_range =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_INDEX, three, poly_arg_range(3, POLY_AXIS_LOOP));
  PolyDType old_ptr = poly_dtype_ptr(POLY_FLOAT32, 6, POLY_ADDR_GLOBAL);
  PolyUOp *store_target =
      poly_uop2(ctx, POLY_OP_INDEX, old_ptr, state, store_range, poly_arg_none());
  PolyUOp *store = poly_uop2(
      ctx, POLY_OP_STORE, POLY_VOID, store_target, poly_const_float(ctx, 2.0), poly_arg_none()
  );
  PolyUOp *after_src[2] = {state, store};
  PolyUOp *after = poly_uop(ctx, POLY_OP_AFTER, POLY_FLOAT32, after_src, 2, poly_arg_none());
  PolyUOp *bufferize_src[2] = {after, consumer_range};
  PolyUOp *bufferize = poly_uop(
      ctx, POLY_OP_STAGE, POLY_FLOAT32, bufferize_src, 2,
      poly_arg_bufferize_opts("CPU", POLY_ADDR_GLOBAL, false)
  );

  PolyUOp *result = poly_apply_add_buffers(ctx, bufferize, NULL);
  ASSERT_NOT_NULL(result);
  ASSERT_EQ(result->op, POLY_OP_AFTER);
  ASSERT_PTR_EQ(result->src[0], state);
  ASSERT_INT_EQ(count_ops(ctx, result, POLY_OP_STAGE), 0);
  ASSERT_INT_EQ(count_ops(ctx, result, POLY_OP_LUNIQUE), 0);
  ASSERT_INT_EQ(count_ops(ctx, result, POLY_OP_AFTER), 1);
  ASSERT_INT_EQ(count_ops(ctx, result, POLY_OP_STORE), 1);
  ASSERT_INT_EQ(count_ops(ctx, result, POLY_OP_END), 1);

  PolyUOp *ended = result->src[1];
  ASSERT_EQ(ended->op, POLY_OP_END);
  ASSERT_INT_EQ(ended->n_src, 3);
  ASSERT_EQ(ended->src[0]->op, POLY_OP_STORE);
  ASSERT_EQ(ended->src[0]->src[0]->op, POLY_OP_INDEX);
  ASSERT_PTR_EQ(ended->src[0]->src[0]->src[0], state);
  ASSERT_INT_EQ(poly_range_axis_id(ended->src[1]->arg), 3);
  ASSERT_INT_EQ(poly_range_axis_id(ended->src[2]->arg), 7);

  PolyDType expected_ptr = poly_dtype_ptr(POLY_FLOAT32, 3, POLY_ADDR_GLOBAL);
  ASSERT_TRUE(poly_dtype_eq(ended->src[0]->src[0]->dtype, expected_ptr));

  poly_ctx_destroy(ctx);
  PASS();
}

/* Structural split parity tests */

/* Count how many RANGEs appear in a kernel's toposort that are NOT closed
 * by an END and are NOT reduce ranges (sources of REDUCE ops).
 *
 * Reduce ranges are expected orphans in the pre-codegen kernel: they're
 * handled by pm_reduce in the codegen pipeline, which creates inner
 * DEFINE_REG/END loops. Non-reduce orphan ranges indicate a real bug. */
static int count_orphan_ranges(PolyCtx *ctx, PolyUOp *kernel_sink) {
  int n_topo;
  PolyUOp **topo = poly_toposort(ctx, kernel_sink, &n_topo);

  /* Collect all RANGEs, END-closed RANGEs, and REDUCE-source RANGEs */
  PolyUOp *all_ranges[64];
  int n_ranges = 0;
  PolyUOp *closed_ranges[64];
  int n_closed = 0;
  PolyUOp *reduce_ranges[64];
  int n_reduce = 0;

  for (int i = 0; i < n_topo; i++) {
    if (topo[i]->op == POLY_OP_RANGE && n_ranges < 64) all_ranges[n_ranges++] = topo[i];
    if (topo[i]->op == POLY_OP_END) {
      for (int s = 1; s < topo[i]->n_src && n_closed < 64; s++) {
        if (topo[i]->src[s] && topo[i]->src[s]->op == POLY_OP_RANGE)
          closed_ranges[n_closed++] = topo[i]->src[s];
      }
    }
    if (topo[i]->op == POLY_OP_REDUCE) {
      for (int s = 1; s < topo[i]->n_src; s++) {
        if (topo[i]->src[s]->op == POLY_OP_RANGE && n_reduce < 64)
          reduce_ranges[n_reduce++] = topo[i]->src[s];
      }
    }
  }

  /* Count ranges not in closed set and not in reduce set */
  int orphans = 0;
  for (int i = 0; i < n_ranges; i++) {
    bool found = false;
    for (int j = 0; j < n_closed; j++) {
      if (all_ranges[i] == closed_ranges[j]) {
        found = true;
        break;
      }
    }
    if (found) continue;
    /* Reduce ranges are expected orphans (pm_reduce handles them) */
    bool is_reduce = false;
    for (int j = 0; j < n_reduce; j++) {
      if (all_ranges[i] == reduce_ranges[j]) {
        is_reduce = true;
        break;
      }
    }
    if (!is_reduce) orphans++;
  }
  return orphans;
}

/* Count occurrences of a specific op in a kernel's toposort. */
static int kernel_op_count(PolyCtx *ctx, PolyUOp *kernel_sink, PolyOps op) {
  int n_topo;
  PolyUOp **topo = poly_toposort(ctx, kernel_sink, &n_topo);
  int count = 0;
  for (int i = 0; i < n_topo; i++)
    if (topo[i]->op == op) count++;
  return count;
}

TEST(rangeify, split_after_index_target_does_not_leak_producer_ranges) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *inner_buf = poly_buffer(ctx, POLY_INT32, 15);
  PolyUOp *outer_buf = poly_buffer(ctx, POLY_INT32, 15);
  PolyUOp *three = poly_const_int(ctx, 3);
  PolyUOp *five = poly_const_int(ctx, 5);
  PolyDType ptr = poly_dtype_ptr(POLY_INT32, 15, POLY_ADDR_GLOBAL);

  PolyUOp *inner_r0 =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_INDEX, three, poly_arg_range(2, POLY_AXIS_LOOP));
  PolyUOp *inner_r1 =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_INDEX, five, poly_arg_range(3, POLY_AXIS_LOOP));
  PolyUOp *inner_mul = poly_uop2(ctx, POLY_OP_MUL, POLY_INDEX, inner_r0, five, poly_arg_none());
  PolyUOp *inner_flat =
      poly_uop2(ctx, POLY_OP_ADD, POLY_INDEX, inner_mul, inner_r1, poly_arg_none());
  PolyUOp *inner_idx = poly_uop2(ctx, POLY_OP_INDEX, ptr, inner_buf, inner_flat, poly_arg_none());
  PolyUOp *inner_store =
      poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, inner_idx, poly_const_int(ctx, 0), poly_arg_none());
  PolyUOp *inner_end_src[3] = {inner_store, inner_r0, inner_r1};
  PolyUOp *inner_end = poly_uop(ctx, POLY_OP_END, POLY_VOID, inner_end_src, 3, poly_arg_none());
  PolyUOp *inner_after_src[2] = {inner_idx, inner_end};
  PolyUOp *inner_after =
      poly_uop(ctx, POLY_OP_AFTER, POLY_INT32, inner_after_src, 2, poly_arg_none());

  PolyUOp *outer_r0 =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_INDEX, three, poly_arg_range(0, POLY_AXIS_LOOP));
  PolyUOp *outer_r1 =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_INDEX, five, poly_arg_range(1, POLY_AXIS_LOOP));
  PolyUOp *outer_mul = poly_uop2(ctx, POLY_OP_MUL, POLY_INDEX, outer_r0, five, poly_arg_none());
  PolyUOp *outer_flat =
      poly_uop2(ctx, POLY_OP_ADD, POLY_INDEX, outer_mul, outer_r1, poly_arg_none());
  PolyUOp *outer_idx = poly_uop2(ctx, POLY_OP_INDEX, ptr, outer_buf, outer_flat, poly_arg_none());
  PolyUOp *inner_read =
      poly_uop2(ctx, POLY_OP_INDEX, ptr, inner_after, outer_flat, poly_arg_none());
  PolyUOp *outer_store =
      poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, outer_idx, inner_read, poly_arg_none());
  PolyUOp *outer_end_src[3] = {outer_store, outer_r0, outer_r1};
  PolyUOp *outer_end = poly_uop(ctx, POLY_OP_END, POLY_VOID, outer_end_src, 3, poly_arg_none());
  PolyUOp *outer_after_src[2] = {outer_buf, outer_end};
  PolyUOp *outer_after =
      poly_uop(ctx, POLY_OP_AFTER, POLY_INT32, outer_after_src, 2, poly_arg_none());
  PolyUOp *kernel_graph = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, outer_after, poly_arg_none());

  PolyKernelScheduleResult sr = poly_build_kernel_schedule_from_kernel_graph(ctx, kernel_graph);
  ASSERT_INT_EQ(sr.n_kernels, 2);
  for (int k = 0; k < sr.n_kernels; k++) {
    ASSERT_INT_EQ(count_orphan_ranges(ctx, sr.kernels[k]), 0);
    ASSERT_INT_EQ(kernel_op_count(ctx, sr.kernels[k], POLY_OP_RANGE), 2);
    ASSERT_INT_EQ(kernel_op_count(ctx, sr.kernels[k], POLY_OP_END), 1);
  }

  poly_kernel_schedule_result_free(&sr);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, exec_order_keeps_versioned_read_after_same_buffer_writes) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *state = poly_buffer(ctx, POLY_INT32, 1);
  PolyUOp *zero = poly_const_int(ctx, 0);
  PolyDType ptr = poly_dtype_ptr(POLY_INT32, 1, POLY_ADDR_GLOBAL);
  PolyUOp *state_idx = poly_uop2(ctx, POLY_OP_INDEX, ptr, state, zero, poly_arg_none());

  PolyUOp *store0 =
      poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, state_idx, poly_const_int(ctx, 1), poly_arg_none());
  PolyUOp *after0_src[2] = {state, store0};
  PolyUOp *after0 = poly_uop(ctx, POLY_OP_AFTER, POLY_INT32, after0_src, 2, poly_arg_none());

  PolyUOp *store1 =
      poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, state_idx, poly_const_int(ctx, 2), poly_arg_none());
  PolyUOp *after1_src[2] = {state, store1};
  PolyUOp *after1 = poly_uop(ctx, POLY_OP_AFTER, POLY_INT32, after1_src, 2, poly_arg_none());

  PolyUOp *lunique = poly_uop0(ctx, POLY_OP_LUNIQUE, POLY_VOID, poly_arg_int(0));
  PolyUOp *device = poly_device_uop(ctx, POLY_DEVICE_CPU);
  PolyUOp *tmp_src[2] = {lunique, device};
  PolyUOp *tmp = poly_uop(ctx, POLY_OP_BUFFER, POLY_INT32, tmp_src, 2, poly_arg_int(1));
  PolyUOp *tmp_idx = poly_uop2(ctx, POLY_OP_INDEX, ptr, tmp, zero, poly_arg_none());
  PolyUOp *versioned_read = poly_uop2(ctx, POLY_OP_INDEX, ptr, after1, zero, poly_arg_none());
  PolyUOp *read_store =
      poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, tmp_idx, versioned_read, poly_arg_none());
  PolyUOp *read_after_src[2] = {tmp, read_store};
  PolyUOp *read_after =
      poly_uop(ctx, POLY_OP_AFTER, POLY_INT32, read_after_src, 2, poly_arg_none());

  PolyUOp *sink_src[2] = {after0, read_after};
  PolyUOp *kernel_graph = poly_uop(ctx, POLY_OP_SINK, POLY_VOID, sink_src, 2, poly_arg_none());
  PolyKernelScheduleResult sr = poly_build_kernel_schedule_from_kernel_graph(ctx, kernel_graph);

  ASSERT_INT_EQ(sr.n_kernels, 3);
  ASSERT_NOT_NULL(sr.exec_order);

  poly_kernel_schedule_result_free(&sr);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, exec_order_keeps_explicit_assign_after_before_fresh_consumer) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *state = poly_buffer_on_device(ctx, POLY_FLOAT32, 1, POLY_DEVICE_CPU);
  PolyUOp *out = poly_buffer_on_device(ctx, POLY_FLOAT32, 1, POLY_DEVICE_CPU);
  float initial = 0.0f;
  poly_buffer_set(ctx, state, &initial, sizeof(initial), POLY_DEVICE_CPU);

  PolyUOp *state_store = poly_store_val(ctx, state, poly_const_float(ctx, 2.0f));
  PolyUOp *state_after_src[2] = {state, state_store};
  PolyUOp *state_after =
      poly_uop(ctx, POLY_OP_AFTER, POLY_FLOAT32, state_after_src, 2, poly_arg_none());
  PolyUOp *consumer_value = poly_alu2(ctx, POLY_OP_ADD, state_after, poly_const_float(ctx, 1.0f));
  PolyUOp *consumer_store = poly_store_val(ctx, out, consumer_value);
  PolyUOp *sink_src[2] = {state_after, consumer_store};
  PolyUOp *sink = poly_uop(ctx, POLY_OP_SINK, POLY_VOID, sink_src, 2, poly_arg_none());
  ASSERT_NOT_NULL(state_store);
  ASSERT_NOT_NULL(state_after);
  ASSERT_NOT_NULL(consumer_value);
  ASSERT_NOT_NULL(consumer_store);
  ASSERT_NOT_NULL(sink);

  PolyUOp *kernel_graph = poly_get_kernel_graph(ctx, sink);
  ASSERT_NOT_NULL(kernel_graph);
  PolyKernelScheduleResult sr = poly_build_kernel_schedule_from_kernel_graph(ctx, kernel_graph);
  ASSERT_INT_EQ(sr.n_kernels, 2);
  ASSERT_NOT_NULL(sr.exec_order);
  ASSERT_INT_EQ(sr.exec_order[0], 0);
  ASSERT_INT_EQ(sr.exec_order[1], 1);

  poly_kernel_schedule_result_free(&sr);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, scalar_store_of_explicit_copy_is_a_copy_kernel) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *dst = poly_buffer_on_device(ctx, POLY_FLOAT32, 1, POLY_DEVICE_INTERP);
  PolyUOp *src = poly_buffer_on_device(ctx, POLY_FLOAT32, 1, POLY_DEVICE_CPU);
  PolyUOp *zero = poly_const_int(ctx, 0);
  PolyDType ptr = poly_dtype_ptr(POLY_FLOAT32, 1, POLY_ADDR_GLOBAL);
  PolyUOp *dst_idx = poly_uop2(ctx, POLY_OP_INDEX, ptr, dst, zero, poly_arg_none());
  PolyUOp *device = poly_device_uop(ctx, POLY_DEVICE_INTERP);
  PolyUOp *copy_src[2] = {src, device};
  PolyUOp *copy = poly_uop(ctx, POLY_OP_COPY, POLY_FLOAT32, copy_src, 2, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, dst_idx, copy, poly_arg_none());
  PolyUOp *kernel_graph = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyKernelScheduleResult sr = poly_build_kernel_schedule_from_kernel_graph(ctx, kernel_graph);
  ASSERT_INT_EQ(sr.n_kernels, 1);
  ASSERT_INT_EQ(sr.kernel_kinds[0], POLY_KERNEL_ITEM_COPY);
  ASSERT_INT_EQ(sr.copy_dst_params[0], 0);
  ASSERT_INT_EQ(sr.copy_src_params[0], 1);
  ASSERT_EQ(sr.kernels[0]->op, POLY_OP_COPY);

  poly_kernel_schedule_result_free(&sr);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, affine_indexed_store_of_copy_keeps_copy_kernel_and_ranges) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *dst = poly_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_INTERP);
  PolyUOp *src = poly_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CPU);
  PolyUOp *r0 = poly_uop_range(ctx, 2, 0, POLY_AXIS_LOOP);
  PolyUOp *r1 = poly_uop_range(ctx, 2, 1, POLY_AXIS_LOOP);
  PolyUOp *two = poly_const_int(ctx, 2);
  PolyUOp *idx = poly_add(ctx, poly_mul(ctx, r0, two), r1);
  PolyDType ptr = poly_dtype_ptr(POLY_FLOAT32, 4, POLY_ADDR_GLOBAL);
  PolyUOp *dst_idx = poly_uop2(ctx, POLY_OP_INDEX, ptr, dst, idx, poly_arg_none());
  PolyUOp *src_idx = poly_uop2(ctx, POLY_OP_INDEX, ptr, src, idx, poly_arg_none());
  PolyUOp *device = poly_device_uop(ctx, POLY_DEVICE_INTERP);
  PolyUOp *copy_src[2] = {src_idx, device};
  PolyUOp *copy = poly_uop(ctx, POLY_OP_COPY, POLY_FLOAT32, copy_src, 2, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, dst_idx, copy, poly_arg_none());
  PolyUOp *ranges[2] = {r0, r1};
  PolyUOp *end = poly_uop_end(ctx, store, ranges, 2);
  PolyUOp *kernel_graph = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());

  PolyKernelScheduleResult sr = poly_build_kernel_schedule_from_kernel_graph(ctx, kernel_graph);
  ASSERT_INT_EQ(sr.n_kernels, 1);
  ASSERT_INT_EQ(sr.kernel_kinds[0], POLY_KERNEL_ITEM_COPY);
  ASSERT_INT_EQ(sr.copy_dst_params[0], 0);
  ASSERT_INT_EQ(sr.copy_src_params[0], 1);
  ASSERT_EQ(sr.kernels[0]->op, POLY_OP_COPY);
  ASSERT_INT_EQ(sr.kernels[0]->n_src, 4);
  ASSERT_EQ(sr.kernels[0]->src[0]->op, POLY_OP_INDEX);
  ASSERT_EQ(sr.kernels[0]->src[1]->op, POLY_OP_DEVICE);
  ASSERT_EQ(sr.kernels[0]->src[2]->op, POLY_OP_RANGE);
  ASSERT_EQ(sr.kernels[0]->src[3]->op, POLY_OP_RANGE);

  poly_kernel_schedule_result_free(&sr);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, split_store_structural_parity) {
  /* Structural parity test for the kernel split pipeline.
   *
   * Graph designed to expose all historical hack vectors:
   *   - Same-size axes (3×3 matrix) → catches coefficient/stride confusion
   *   - Shared intermediate consumed by 2 stores → forces BUFFERIZE
   *   - Movement ops (RESHAPE + PERMUTE) before shared node → tests movement chain
   *   - One reduce consumer → tests REDUCE + END nesting
   *
   * Graph:
   *   a_flat(9) → reshape(3,3) → permute(1,0) → neg → d (shared, BUFFERIZE)
   *   out1 = d + b_2d                           (elementwise consumer)
   *   out2 = reduce_sum(d * c_2d, axis=1)       (reduce consumer)
   *
   * Expected kernels (new path, pm_remove_bufferize inlines d):
   *   K0 (elementwise): neg(perm(reshape(a))) + b → out1
   *   K1 (reduce):      reduce_sum(neg(perm(reshape(a))) * c) → out2
   *
   * Assertions:
   *   1. No BUFFERIZE after add_buffers
   *   2. n_kernels == 2, n_intermediates == 0
   *   3. No orphan RANGEs in any kernel
   *   4. Each kernel has correct structural op counts
   *   5. E2E values match expected
   */
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a_flat = poly_buffer(ctx, POLY_FLOAT32, 9);
  PolyUOp *b_flat = poly_buffer(ctx, POLY_FLOAT32, 9);
  PolyUOp *c_flat = poly_buffer(ctx, POLY_FLOAT32, 9);
  PolyUOp *out1_flat = poly_buffer(ctx, POLY_FLOAT32, 9);
  PolyUOp *out2_flat = poly_buffer(ctx, POLY_FLOAT32, 3);

  int64_t shape2d[] = {3, 3};
  PolyUOp *a = poly_reshape(ctx, a_flat, shape2d, 2);
  int64_t perm[] = {1, 0};
  PolyUOp *a_perm = poly_permute(ctx, a, perm, 2);
  PolyUOp *b = poly_reshape(ctx, b_flat, shape2d, 2);
  PolyUOp *c = poly_reshape(ctx, c_flat, shape2d, 2);
  PolyUOp *out1 = poly_reshape(ctx, out1_flat, shape2d, 2);

  /* d = neg(permute(reshape(a))), shared → BUFFERIZE */
  PolyUOp *d = poly_uop1(ctx, POLY_OP_NEG, POLY_FLOAT32, a_perm, poly_arg_none());

  /* out1 = d + b (elementwise) */
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, d, b, poly_arg_none());
  PolyUOp *s1 = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out1, add, poly_arg_none());

  /* out2 = reduce_sum(d * c, axis=1) */
  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, d, c, poly_arg_none());
  int64_t red_axes[] = {1};
  PolyUOp *red = poly_reduce_axis(ctx, POLY_OP_ADD, mul, red_axes, 1);
  PolyUOp *s2 = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out2_flat, red, poly_arg_none());

  PolyUOp *stores[] = {s1, s2};
  PolyUOp *sink = poly_uop(ctx, POLY_OP_SINK, POLY_VOID, stores, 2, poly_arg_none());

  /* IR-level assertions (before realize) */

  /* Run rangeify + add_buffers to inspect the intermediate graph */
  PolyIndexingCtx *ictx = poly_indexing_ctx_new(ctx);
  PolyUOp *rangeified = run_apply_rangeify(ictx, sink);
  ASSERT_NOT_NULL(rangeified);

  /* Shared d should be realized → exactly 1 BUFFERIZE */
  ASSERT_INT_EQ(count_ops(ctx, rangeified, POLY_OP_STAGE), 1);

  /* Apply add_buffers */
  PolyUOp *ab_result = poly_apply_add_buffers(ctx, rangeified, NULL);
  ASSERT_NOT_NULL(ab_result);

  /* Anti-hack assertion 1: No BUFFERIZE remains after add_buffers */
  ASSERT_INT_EQ(count_ops(ctx, ab_result, POLY_OP_STAGE), 0);

  /* Schedule and check kernel structure */

  PolyKernelScheduleResult sr = poly_build_kernel_schedule(ctx, sink);

  /* Assertion 2: correct kernel count */
  ASSERT_INT_EQ(sr.n_kernels, 2); /* 2 fused kernels (d inlined) */
  ASSERT_INT_EQ(sr.n_intermediates, 0);

  /* Assertion 3: no orphan RANGEs in any kernel */
  for (int k = 0; k < sr.n_kernels; k++) {
    int orphans = count_orphan_ranges(ctx, sr.kernels[k]);
    if (orphans > 0) {
      fprintf(stderr, "    kernel %d has %d orphan RANGE(s)\n", k, orphans);
    }
    ASSERT_INT_EQ(orphans, 0);
  }

  /* Assertion 4: structural op counts per kernel */
  /* K0 (elementwise): neg(perm(reshape(a))) + b → out1 */
  int k0_stores = kernel_op_count(ctx, sr.kernels[0], POLY_OP_STORE);
  int k0_ends = kernel_op_count(ctx, sr.kernels[0], POLY_OP_END);
  int k0_ranges = kernel_op_count(ctx, sr.kernels[0], POLY_OP_RANGE);
  ASSERT_INT_EQ(k0_stores, 1);
  ASSERT_TRUE(k0_ranges > 0);
  ASSERT_TRUE(k0_ends > 0);
  ASSERT_TRUE(k0_ranges >= k0_ends);

  /* K1 (reduce): reduce_sum(neg(perm(reshape(a))) * c) → out2 */
  int k1_stores = kernel_op_count(ctx, sr.kernels[1], POLY_OP_STORE);
  int k1_ranges = kernel_op_count(ctx, sr.kernels[1], POLY_OP_RANGE);
  int k1_ends = kernel_op_count(ctx, sr.kernels[1], POLY_OP_END);
  int k1_reduces = kernel_op_count(ctx, sr.kernels[1], POLY_OP_REDUCE);
  ASSERT_INT_EQ(k1_stores, 1);
  ASSERT_TRUE(k1_ranges > 0);
  ASSERT_TRUE(k1_ends > 0);
  ASSERT_TRUE(k1_ranges >= k1_ends); /* ranges >= ends (reduce range has no END) */
  ASSERT_INT_EQ(k1_reduces, 1);

  poly_kernel_schedule_result_free(&sr);
  poly_indexing_ctx_destroy(ictx);

  /* E2E value correctness */

  /* a_flat = [[1,2,3],[4,5,6],[7,8,9]] (asymmetric to detect transposes)
   * After permute(1,0): a_perm[r][c] = a[c][r]
   *   [[1,4,7],[2,5,8],[3,6,9]]
   * d = -a_perm: [[-1,-4,-7],[-2,-5,-8],[-3,-6,-9]]
   *
   * b = [[10,20,30],[40,50,60],[70,80,90]]
   * out1[r][c] = d[r][c] + b[r][c]
   *
   * c = [[0.1,0.2,0.3],[0.4,0.5,0.6],[0.7,0.8,0.9]]
   * out2[r] = sum_c(d[r][c] * c[r][c]) */
  float a_d[9], b_d[9], c_d[9], o1_d[9], o2_d[3];
  for (int r = 0; r < 3; r++)
    for (int col = 0; col < 3; col++) {
      int i = r * 3 + col;
      a_d[i] = (float)(i + 1); /* 1,2,3,4,5,6,7,8,9 */
      b_d[i] = (float)((col + 1) * 10 + r * 30); /* asymmetric */
      c_d[i] = (float)(i + 1) * 0.1f; /* 0.1..0.9 */
    }
  memset(o1_d, 0, sizeof(o1_d));
  memset(o2_d, 0, sizeof(o2_d));

  PolyTestBufferView bindings[] = {
      POLY_TEST_HOST_VIEW(out1_flat, o1_d), POLY_TEST_HOST_VIEW(out2_flat, o2_d),
      POLY_TEST_HOST_VIEW(a_flat, a_d),     POLY_TEST_HOST_VIEW(b_flat, b_d),
      POLY_TEST_HOST_VIEW(c_flat, c_d),
  };
  int ret = poly_test_realize_buffer_views(ctx, sink, bindings, 5);
  ASSERT_INT_EQ(ret, 0);

  /* Verify out1 = neg(permute(a)) + b */
  for (int r = 0; r < 3; r++)
    for (int col = 0; col < 3; col++) {
      int i = r * 3 + col;
      float neg_a_perm = -(float)(col * 3 + r + 1); /* -a[col][r] */
      ASSERT_FLOAT_EQ(o1_d[i], neg_a_perm + b_d[i], 1e-5);
    }

  /* Verify out2 = reduce_sum(neg(permute(a)) * c, axis=1) */
  for (int r = 0; r < 3; r++) {
    float expected = 0.0f;
    for (int col = 0; col < 3; col++) {
      float neg_a_perm = -(float)(col * 3 + r + 1);
      expected += neg_a_perm * c_d[r * 3 + col];
    }
    ASSERT_FLOAT_EQ(o2_d[r], expected, 1e-4);
  }

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, remove_bufferize_stops_at_after_effect) {
  /* Pinned tinygrad's remove_bufferize red_gate counts AFTER as exactly its
   * buffer identity and does not descend into the owned STORE. The shared NEG
   * below therefore has one accessed buffer and is inlined into both output
   * kernels. Descending through AFTER/STORE sees five creation inputs and
   * incorrectly creates a fourth, schedule-owned intermediate kernel. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  enum { N = 4 };
  PolyUOp *a = poly_buffer_f32(ctx, N);
  PolyUOp *b = poly_buffer_f32(ctx, N);
  PolyUOp *c = poly_buffer_f32(ctx, N);
  PolyUOp *d = poly_buffer_f32(ctx, N);
  PolyUOp *state = poly_buffer_f32(ctx, N);
  PolyUOp *out_add = poly_buffer_f32(ctx, N);
  PolyUOp *out_mul = poly_buffer_f32(ctx, N);
  PolyUOp *addend = poly_buffer_f32(ctx, N);
  PolyUOp *factor = poly_buffer_f32(ctx, N);

  PolyUOp *created = poly_add(ctx, poly_add(ctx, a, b), poly_add(ctx, c, d));
  PolyUOp *state_store = poly_store_val(ctx, state, created);
  PolyUOp *state_after_src[2] = {state, state_store};
  PolyUOp *state_after =
      poly_uop(ctx, POLY_OP_AFTER, state->dtype, state_after_src, 2, poly_arg_none());
  PolyUOp *shared = poly_uop1(ctx, POLY_OP_NEG, POLY_FLOAT32, state_after, poly_arg_none());
  PolyUOp *stores[2] = {
      poly_store_val(ctx, out_add, poly_add(ctx, shared, addend)),
      poly_store_val(ctx, out_mul, poly_mul(ctx, shared, factor)),
  };
  PolyUOp *sink = poly_sink_n(ctx, stores, 2);
  ASSERT_NOT_NULL(sink);

  PolyUOp *kernel_graph = poly_get_kernel_graph(ctx, sink);
  ASSERT_NOT_NULL(kernel_graph);
  ASSERT_INT_EQ(count_ops(ctx, kernel_graph, POLY_OP_STAGE), 0);
  ASSERT_INT_EQ(count_ops(ctx, kernel_graph, POLY_OP_AFTER), 1);
  ASSERT_INT_EQ(count_ops(ctx, kernel_graph, POLY_OP_STORE), 3);

  PolyKernelScheduleResult schedule =
      poly_build_kernel_schedule_from_kernel_graph(ctx, kernel_graph);
  ASSERT_INT_EQ(schedule.n_kernels, 3);
  ASSERT_INT_EQ(schedule.n_intermediates, 0);
  poly_kernel_schedule_result_free(&schedule);

  float a_data[N] = {1, 2, 3, 4};
  float b_data[N] = {10, 20, 30, 40};
  float c_data[N] = {100, 200, 300, 400};
  float d_data[N] = {1000, 2000, 3000, 4000};
  float state_data[N] = {0};
  float out_add_data[N] = {0};
  float out_mul_data[N] = {0};
  float addend_data[N] = {5, 5, 5, 5};
  float factor_data[N] = {2, 2, 2, 2};
  PolyTestBufferView bindings[] = {
      POLY_TEST_HOST_VIEW(a, a_data),
      POLY_TEST_HOST_VIEW(b, b_data),
      POLY_TEST_HOST_VIEW(c, c_data),
      POLY_TEST_HOST_VIEW(d, d_data),
      POLY_TEST_HOST_VIEW(state, state_data),
      POLY_TEST_HOST_VIEW(out_add, out_add_data),
      POLY_TEST_HOST_VIEW(out_mul, out_mul_data),
      POLY_TEST_HOST_VIEW(addend, addend_data),
      POLY_TEST_HOST_VIEW(factor, factor_data),
  };
  ASSERT_INT_EQ(poly_test_realize_buffer_views(ctx, sink, bindings, 9), 0);
  for (int i = 0; i < N; i++) {
    float expected_state = a_data[i] + b_data[i] + c_data[i] + d_data[i];
    ASSERT_FLOAT_EQ(state_data[i], expected_state, 1e-5f);
    ASSERT_FLOAT_EQ(out_add_data[i], -expected_state + addend_data[i], 1e-5f);
    ASSERT_FLOAT_EQ(out_mul_data[i], -expected_state * factor_data[i], 1e-5f);
  }

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, raw_buffer_shared_scalar_reduce_branches_ir) {
  /* a[8] → REDUCE(ADD, axis=0) → RESHAPE([1]) → EXPAND([8]) → sum_exp[8]
   * Branch 1: ADD(sum_exp, c0) → STORE(oc)
   * Branch 2: MUL(sum_exp, e0) → STORE(oe)
   * Pinned tinygrad raw-UOp parity: concrete BUFFER input is not a callified
   * PARAM boundary, so remove_bufferize fuses the shared expression into two
   * consumer CALL bodies. Normal Tensor callify is asserted separately and
   * creates two raw STAGEs, then the generic cleanup retains the shared
   * scalar-reduction STAGE and produces three ordered CALLs. */
  int N = 8;
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *c0 = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *e0 = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *oc = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *oe = poly_buffer(ctx, POLY_FLOAT32, N);

  int64_t axes[] = {0};
  int64_t one_sh[] = {1};
  int64_t exp_sh[] = {N};

  PolyUOp *sum = poly_reduce_axis(ctx, POLY_OP_ADD, a, axes, 1);
  PolyUOp *sum_1 = poly_reshape(ctx, sum, one_sh, 1);
  PolyUOp *sum_exp = poly_expand(ctx, sum_1, exp_sh, 1);

  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, sum_exp, c0, poly_arg_none());
  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, sum_exp, e0, poly_arg_none());
  PolyUOp *sc = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, oc, add, poly_arg_none());
  PolyUOp *se = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, oe, mul, poly_arg_none());
  PolyUOp *sink = poly_uop(ctx, POLY_OP_SINK, POLY_VOID, (PolyUOp *[]){sc, se}, 2, poly_arg_none());

  PolyKernelScheduleResult sr = poly_build_kernel_schedule(ctx, sink);
  /* Copy results before freeing so assertions don't leak on failure. */
  int n_kernels = sr.n_kernels;
  int n_intermediates = sr.n_intermediates;
  poly_kernel_schedule_result_free(&sr);
  poly_ctx_destroy(ctx);

  ASSERT_INT_EQ(n_kernels, 2);
  ASSERT_INT_EQ(n_intermediates, 0);
  PASS();
}

TEST(rangeify, callified_param_shared_scalar_reduce_matches_raw_stage_topology) {
  /* Exact pinned-tinygrad boundary from
   * temp/tg_shared_scalar_rangeify_trace.py: callify first replaces tensor
   * BUFFERs with shaped PARAMs. Raw run_rangeify creates a REDUCE STAGE and an
   * expanded-INDEX STAGE; the later symbolic/reduce/debufferize pass removes
   * the expanded-INDEX STAGE and retains the shared scalar producer. */
  int N = 8;
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *c = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *e = poly_buffer(ctx, POLY_FLOAT32, N);
  int64_t axes[] = {0};
  int64_t one[] = {1};
  int64_t eight[] = {N};
  PolyUOp *sum = poly_reduce_axis(ctx, POLY_OP_ADD, a, axes, 1);
  PolyUOp *sum_scalar = poly_reshape(ctx, sum, NULL, 0);
  PolyUOp *sum_one = poly_reshape(ctx, sum_scalar, one, 1);
  PolyUOp *expanded = poly_expand(ctx, sum_one, eight, 1);
  PolyUOp *targets[] = {
      poly_alu2(ctx, POLY_OP_ADD, expanded, c),
      poly_alu2(ctx, POLY_OP_MUL, expanded, e),
  };
  PolyUOp *realized[] = {NULL, NULL};
  PolyUOp *call = poly_transform_to_call(ctx, targets, 2, realized);
  ASSERT_NOT_NULL(call);
  ASSERT_INT_EQ(call->op, POLY_OP_CALL);
  ASSERT_NOT_NULL(call->src[0]);
  ASSERT_INT_EQ(call->src[0]->op, POLY_OP_SINK);
  ASSERT_INT_EQ(call->src[0]->n_src, 2);
  ASSERT_INT_EQ(count_ops(ctx, call->src[0], POLY_OP_AFTER), 2);
  ASSERT_INT_EQ(count_ops(ctx, call->src[0], POLY_OP_STORE), 2);
  ASSERT_INT_EQ(count_reduce_arg_kind(ctx, call->src[0], POLY_ARG_REDUCE_AXIS), 1);

  PolyUOp *function = poly_apply_earliest_rewrites(ctx, call->src[0]);
  ASSERT_NOT_NULL(function);
  PolyIndexingCtx *ictx = poly_indexing_ctx_new(ctx);
  ASSERT_NOT_NULL(ictx);
  ictx->add_buffer_indices = true;
  PolyUOp *rangeified = run_apply_rangeify(ictx, function);
  ASSERT_NOT_NULL(rangeified);
  ASSERT_INT_EQ(count_ops(ctx, rangeified, POLY_OP_STAGE), 2);
  ASSERT_INT_EQ(count_reduce_arg_kind(ctx, rangeified, POLY_ARG_OPS), 1);
  ASSERT_INT_EQ(count_ops(ctx, rangeified, POLY_OP_AFTER), 2);
  ASSERT_INT_EQ(count_ops(ctx, rangeified, POLY_OP_STORE), 2);
  ASSERT_INT_EQ(count_ops(ctx, rangeified, POLY_OP_INDEX), 8);
  ASSERT_INT_EQ(count_ops(ctx, rangeified, POLY_OP_RANGE), 4);

  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, rangeified, &n_topo);
  PolyUOp *reduce_stage = NULL;
  PolyUOp *index_stage = NULL;
  for (int i = 0; i < n_topo; i++) {
    if (topo[i]->op != POLY_OP_STAGE) continue;
    ASSERT_INT_EQ(topo[i]->n_src, 2);
    ASSERT_INT_EQ(topo[i]->arg.kind, POLY_ARG_BUFFERIZE_OPTS);
    ASSERT_INT_EQ(poly_bufferize_arg_addrspace(topo[i]->arg), POLY_ADDR_GLOBAL);
    ASSERT_TRUE(poly_bufferize_arg_removable(topo[i]->arg));
    if (topo[i]->src[0]->op == POLY_OP_REDUCE) {
      ASSERT_TRUE(reduce_stage == NULL);
      reduce_stage = topo[i];
    } else if (topo[i]->src[0]->op == POLY_OP_INDEX) {
      ASSERT_TRUE(index_stage == NULL);
      index_stage = topo[i];
    } else {
      ASSERT_TRUE(false);
    }
  }
  ASSERT_NOT_NULL(reduce_stage);
  ASSERT_NOT_NULL(index_stage);

  poly_indexing_ctx_destroy(ictx);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, shared_scalar_reduce_branches_e2e) {
  /* Same graph, full compile+execute via poly_test_realize_buffer_views().
   * a = [1..8] → sum = 36
   * c0 = [10,10,...] → oc[i] = 36 + 10 = 46
   * e0 = [2,2,...] → oe[i] = 36 * 2 = 72 */
  int N = 8;
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *c0 = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *e0 = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *oc = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *oe = poly_buffer(ctx, POLY_FLOAT32, N);

  int64_t axes[] = {0};
  int64_t one_sh[] = {1};
  int64_t exp_sh[] = {N};

  PolyUOp *sum = poly_reduce_axis(ctx, POLY_OP_ADD, a, axes, 1);
  PolyUOp *sum_1 = poly_reshape(ctx, sum, one_sh, 1);
  PolyUOp *sum_exp = poly_expand(ctx, sum_1, exp_sh, 1);

  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, sum_exp, c0, poly_arg_none());
  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, sum_exp, e0, poly_arg_none());
  PolyUOp *sc = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, oc, add, poly_arg_none());
  PolyUOp *se = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, oe, mul, poly_arg_none());
  PolyUOp *sink = poly_uop(ctx, POLY_OP_SINK, POLY_VOID, (PolyUOp *[]){sc, se}, 2, poly_arg_none());

  float a_d[8], c0_d[8], e0_d[8], oc_d[8], oe_d[8];
  for (int i = 0; i < N; i++) {
    a_d[i] = (float)(i + 1); /* 1..8, sum=36 */
    c0_d[i] = 10.0f;
    e0_d[i] = 2.0f;
    oc_d[i] = 0.0f;
    oe_d[i] = 0.0f;
  }

  PolyTestBufferView bindings[] = {
      POLY_TEST_HOST_VIEW(oc, oc_d), POLY_TEST_HOST_VIEW(oe, oe_d), POLY_TEST_HOST_VIEW(a, a_d),
      POLY_TEST_HOST_VIEW(c0, c0_d), POLY_TEST_HOST_VIEW(e0, e0_d),
  };
  int ret = poly_test_realize_buffer_views(ctx, sink, bindings, 5);
  poly_ctx_destroy(ctx);
  ASSERT_INT_EQ(ret, 0);

  for (int i = 0; i < N; i++) {
    ASSERT_FLOAT_EQ(oc_d[i], 46.0f, 1e-5);
    ASSERT_FLOAT_EQ(oe_d[i], 72.0f, 1e-5);
  }

  PASS();
}

/* Stage A: CONST-through-BUFFERIZE + noop removal */

TEST(rangeify, const_through_bufferize) {
  /* CONST(42.0) broadcasted via RESHAPE+EXPAND -> ADD with buffer.
   * The constant should fold through BUFFERIZE (no intermediate buffer). */
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, 4);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, 4);
  PolyUOp *c42 = poly_uop(ctx, POLY_OP_CONST, POLY_FLOAT32, NULL, 0, poly_arg_float(42.0));
  int64_t one_sh[] = {1};
  int64_t exp_sh[] = {4};
  PolyUOp *reshaped = poly_reshape(ctx, c42, one_sh, 1);
  PolyUOp *expanded = poly_expand(ctx, reshaped, exp_sh, 1);
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, a, expanded, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out, add, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  float a_d[] = {1, 2, 3, 4};
  float out_d[4] = {0};
  PolyTestBufferView bindings[] = {
      POLY_TEST_HOST_VIEW(out, out_d),
      POLY_TEST_HOST_VIEW(a, a_d),
  };
  int ret = poly_test_realize_buffer_views(ctx, sink, bindings, 2);
  poly_ctx_destroy(ctx);
  ASSERT_INT_EQ(ret, 0);

  for (int i = 0; i < 4; i++) {
    ASSERT_FLOAT_EQ(out_d[i], a_d[i] + 42.0f, 1e-5);
  }
  PASS();
}

TEST(rangeify, moved_const_folding_add_shrunk_zero_e2e) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, 4);
  PolyUOp *zero6 = poly_full(ctx, (int64_t[]){6}, 1, 0.0);
  PolyUOp *zero4 = poly_shrink(ctx, zero6, (int64_t[][2]){{1, 5}}, 1);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, 4);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, poly_alu2(ctx, POLY_OP_ADD, a, zero4)));

  /* Port of tinygrad test_const_folding.py::test_add_shrunk_zero. The
   * movement-wrapped zero must behave as an elementwise identity and stay in a
   * single output kernel, not force an intermediate materialization. */
  PolySchedule *sched = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_NOT_NULL(sched);
  ASSERT_INT_EQ(sched->template->n_calls, 1);
  poly_schedule_free(sched);

  float a_d[] = {1, 2, 3, 4};
  float out_d[4] = {0};
  PolyTestBufferView bindings[] = {
      POLY_TEST_HOST_VIEW(a, a_d),
      POLY_TEST_HOST_VIEW(out, out_d),
  };
  int ret = poly_test_realize_buffer_views(ctx, sink, bindings, 2);
  poly_ctx_destroy(ctx);
  ASSERT_INT_EQ(ret, 0);
  for (int i = 0; i < 4; i++)
    ASSERT_FLOAT_EQ(out_d[i], a_d[i], 1e-5);
  PASS();
}

TEST(rangeify, moved_const_folding_add_padded_zero_e2e) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, 4);
  PolyUOp *zero2 = poly_full(ctx, (int64_t[]){2}, 1, 0.0);
  PolyUOp *zero4 = poly_pad(ctx, zero2, (int64_t[][2]){{1, 1}}, 1);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, 4);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, poly_alu2(ctx, POLY_OP_ADD, a, zero4)));

  /* Port of tinygrad test_const_folding.py::test_add_padded_zero. Padded
   * zeros are valid/index expressions internally, so this catches regressions
   * where identity folding stops at movement boundaries. */
  PolySchedule *sched = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_NOT_NULL(sched);
  ASSERT_INT_EQ(sched->template->n_calls, 1);
  poly_schedule_free(sched);

  float a_d[] = {5, 6, 7, 8};
  float out_d[4] = {0};
  PolyTestBufferView bindings[] = {
      POLY_TEST_HOST_VIEW(a, a_d),
      POLY_TEST_HOST_VIEW(out, out_d),
  };
  int ret = poly_test_realize_buffer_views(ctx, sink, bindings, 2);
  poly_ctx_destroy(ctx);
  ASSERT_INT_EQ(ret, 0);
  for (int i = 0; i < 4; i++)
    ASSERT_FLOAT_EQ(out_d[i], a_d[i], 1e-5);
  PASS();
}

TEST(rangeify, moved_const_folding_mul_shrunk_one_e2e) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, 4);
  PolyUOp *one6 = poly_full(ctx, (int64_t[]){6}, 1, 1.0);
  PolyUOp *one4 = poly_shrink(ctx, one6, (int64_t[][2]){{1, 5}}, 1);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, 4);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, poly_alu2(ctx, POLY_OP_MUL, a, one4)));

  /* Port of tinygrad test_const_folding.py::test_mul_shrunk_one. */
  PolySchedule *sched = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_NOT_NULL(sched);
  ASSERT_INT_EQ(sched->template->n_calls, 1);
  poly_schedule_free(sched);

  float a_d[] = {-1, 2, -3, 4};
  float out_d[4] = {0};
  PolyTestBufferView bindings[] = {
      POLY_TEST_HOST_VIEW(a, a_d),
      POLY_TEST_HOST_VIEW(out, out_d),
  };
  int ret = poly_test_realize_buffer_views(ctx, sink, bindings, 2);
  poly_ctx_destroy(ctx);
  ASSERT_INT_EQ(ret, 0);
  for (int i = 0; i < 4; i++)
    ASSERT_FLOAT_EQ(out_d[i], a_d[i], 1e-5);
  PASS();
}

TEST(rangeify, zero_size_sum_folds_to_identity_e2e) {
  PolyCtx *ctx = poly_ctx_new();

  /* Pinned Tensor.empty is BUFFER(UNIQUE, DEVICE) followed by movement
   * (uop/ops.py:733-746); full(buffer=False) is a pure CONST graph and is not
   * bindable storage. */
  PolyUOp *empty = poly_reshape(
      ctx, poly_buffer_on_device(ctx, POLY_FLOAT32, 0, POLY_DEVICE_CPU), (int64_t[]){1, 0}, 2
  );
  PolyUOp *sum = poly_reduce_axis(ctx, POLY_OP_ADD, empty, (int64_t[]){0, 1}, 2);
  PolyUOp *scalar = poly_reshape(ctx, sum, NULL, 0);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, 1);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, scalar));

  /* Port of tinygrad TestReduceOpsConstFolding zero-size sum coverage. Empty
   * reductions should fold to the ADD identity instead of building an invalid
   * zero-iteration kernel. */
  float empty_storage[1] = {0.0f};
  float out_d[1] = {-1.0f};
  const PolyUOp *empty_buf = poly_uop_get_buffer_identity(empty);
  ASSERT_NOT_NULL(empty_buf);
  PolyTestBufferView bindings[] = {
      POLY_TEST_HOST_VIEW((PolyUOp *)empty_buf, empty_storage),
      POLY_TEST_HOST_VIEW(out, out_d),
  };
  int ret = poly_test_realize_buffer_views(ctx, sink, bindings, 2);
  poly_ctx_destroy(ctx);
  ASSERT_INT_EQ(ret, 0);
  ASSERT_FLOAT_EQ(out_d[0], 0.0f, 1e-5);
  PASS();
}

TEST(rangeify, zero_size_max_folds_to_typed_identity_e2e) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *empty = poly_reshape(
      ctx, poly_buffer_on_device(ctx, POLY_INT32, 0, POLY_DEVICE_CPU), (int64_t[]){1, 0}, 2
  );
  PolyUOp *maximum = poly_reduce_axis(ctx, POLY_OP_MAX, empty, (int64_t[]){0, 1}, 2);
  PolyUOp *scalar = poly_reshape(ctx, maximum, NULL, 0);
  PolyUOp *out = poly_buffer(ctx, POLY_INT32, 1);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, scalar));

  int32_t empty_storage[1] = {0};
  int32_t out_d[1] = {0};
  const PolyUOp *empty_buf = poly_uop_get_buffer_identity(empty);
  ASSERT_NOT_NULL(empty_buf);
  PolyTestBufferView bindings[] = {
      POLY_TEST_HOST_VIEW((PolyUOp *)empty_buf, empty_storage),
      POLY_TEST_HOST_VIEW(out, out_d),
  };
  int ret = poly_test_realize_buffer_views(ctx, sink, bindings, 2);
  poly_ctx_destroy(ctx);
  ASSERT_INT_EQ(ret, 0);
  ASSERT_INT_EQ(out_d[0], INT32_MIN);
  PASS();
}

TEST(rangeify, zero_size_max_fills_every_output_with_typed_identity_e2e) {
  /* Pinned rangeify.py:204-207 uses reduce.const_like(identity), not a scalar
   * CONST. A (2,0,3)->(2,1,3) MAX must therefore fill all six outputs. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *empty_i = poly_reshape(
      ctx, poly_buffer_on_device(ctx, POLY_INT32, 0, POLY_DEVICE_CPU), (int64_t[]){2, 0, 3}, 3
  );
  PolyUOp *max_i = poly_reduce_axis(ctx, POLY_OP_MAX, empty_i, (int64_t[]){1}, 1);
  PolyUOp *out_i = poly_buffer(ctx, POLY_INT32, 6);
  PolyUOp *store_i = poly_store_val(ctx, out_i, max_i);

  PolyUOp *empty_f = poly_reshape(
      ctx, poly_buffer_on_device(ctx, POLY_FLOAT32, 0, POLY_DEVICE_CPU), (int64_t[]){2, 0, 3}, 3
  );
  PolyUOp *max_f = poly_reduce_axis(ctx, POLY_OP_MAX, empty_f, (int64_t[]){1}, 1);
  PolyUOp *out_f = poly_buffer(ctx, POLY_FLOAT32, 6);
  PolyUOp *store_f = poly_store_val(ctx, out_f, max_f);
  PolyUOp *sink_src[] = {store_i, store_f};
  PolyUOp *sink = poly_sink_n(ctx, sink_src, 2);

  int32_t empty_i_storage[1] = {0}, out_i_data[6] = {0};
  float empty_f_storage[1] = {0.0f}, out_f_data[6] = {0.0f};
  const PolyUOp *empty_i_buf = poly_uop_get_buffer_identity(empty_i);
  const PolyUOp *empty_f_buf = poly_uop_get_buffer_identity(empty_f);
  ASSERT_NOT_NULL(empty_i_buf);
  ASSERT_NOT_NULL(empty_f_buf);
  PolyTestBufferView bindings[] = {
      POLY_TEST_HOST_VIEW((PolyUOp *)empty_i_buf, empty_i_storage),
      POLY_TEST_HOST_VIEW(out_i, out_i_data),
      POLY_TEST_HOST_VIEW((PolyUOp *)empty_f_buf, empty_f_storage),
      POLY_TEST_HOST_VIEW(out_f, out_f_data),
  };
  int ret = poly_test_realize_buffer_views(ctx, sink, bindings, 4);
  ASSERT_INT_EQ(ret, 0);
  for (int i = 0; i < 6; i++) {
    ASSERT_INT_EQ(out_i_data[i], INT32_MIN);
    ASSERT_TRUE(isinf(out_f_data[i]) && signbit(out_f_data[i]));
  }
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, zero_size_rewrites_preserve_const_like_shape_topology) {
  /* Exact pinned predicates:
   * - nonzero output: shaped typed identity;
   * - zero output: the general shaped-zero rule, not REDUCE identity. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *empty = poly_reshape(
      ctx, poly_buffer_on_device(ctx, POLY_INT32, 0, POLY_DEVICE_CPU), (int64_t[]){2, 0, 3}, 3
  );
  PolyUOp *maximum = poly_reduce_axis(ctx, POLY_OP_MAX, empty, (int64_t[]){1}, 1);
  PolyUOp *rewritten = poly_apply_earliest_rewrites(ctx, maximum);
  ASSERT_NOT_NULL(rewritten);
  ASSERT_EQ(rewritten->op, POLY_OP_EXPAND);
  ASSERT_EQ(rewritten->src[0]->op, POLY_OP_RESHAPE);
  ASSERT_EQ(rewritten->src[0]->src[0]->op, POLY_OP_CONST);
  ASSERT_INT_EQ(rewritten->src[0]->src[0]->arg.kind, POLY_ARG_INT);
  ASSERT_INT_EQ(rewritten->src[0]->src[0]->arg.i, INT32_MIN);
  PolyShape shape = poly_uop_max_shape_cached(ctx, rewritten);
  ASSERT_INT_EQ(shape.ndim, 3);
  ASSERT_INT_EQ(shape.dims[0], 2);
  ASSERT_INT_EQ(shape.dims[1], 1);
  ASSERT_INT_EQ(shape.dims[2], 3);

  PolyUOp *zero_output = poly_reshape(
      ctx, poly_buffer_on_device(ctx, POLY_INT32, 0, POLY_DEVICE_CPU), (int64_t[]){0, 0, 3}, 3
  );
  PolyUOp *zero_max = poly_reduce_axis(ctx, POLY_OP_MAX, zero_output, (int64_t[]){1}, 1);
  PolyUOp *zero_rewritten = poly_apply_earliest_rewrites(ctx, zero_max);
  ASSERT_NOT_NULL(zero_rewritten);
  ASSERT_EQ(zero_rewritten->op, POLY_OP_EXPAND);
  ASSERT_EQ(zero_rewritten->src[0]->op, POLY_OP_RESHAPE);
  ASSERT_EQ(zero_rewritten->src[0]->src[0]->op, POLY_OP_CONST);
  ASSERT_INT_EQ(zero_rewritten->src[0]->src[0]->arg.kind, POLY_ARG_INT);
  ASSERT_INT_EQ(zero_rewritten->src[0]->src[0]->arg.i, 0);
  shape = poly_uop_max_shape_cached(ctx, zero_rewritten);
  ASSERT_INT_EQ(shape.ndim, 3);
  ASSERT_INT_EQ(shape.dims[0], 0);
  ASSERT_INT_EQ(shape.dims[1], 1);
  ASSERT_INT_EQ(shape.dims[2], 3);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, zero_size_general_rule_strips_pointer_and_retags_only_root) {
  /* Pinned x.const_like(0).rtag(x.tag) uses dtype.base, preserves the
   * zero-extent shape, and applies metadata only to the replacement root. */
  PolyCtx *ctx = poly_ctx_new();
  PolyDType ptr0 = poly_dtype_ptr(POLY_FLOAT32, 0, POLY_ADDR_GLOBAL);
  PolyUOp *tagged = poly_uop_tagged(ctx, POLY_OP_PARAM, ptr0, NULL, 0, poly_arg_int(0), 77);
  PolyUOp *rewritten = poly_apply_earliest_rewrites(ctx, tagged);
  ASSERT_NOT_NULL(rewritten);
  ASSERT_EQ(rewritten->op, POLY_OP_EXPAND);
  ASSERT_INT_EQ(rewritten->tag, 77);
  ASSERT_TRUE(poly_dtype_eq(rewritten->dtype, POLY_FLOAT32));
  ASSERT_FALSE(rewritten->dtype.is_ptr);
  PolyShape shape = poly_uop_max_shape_cached(ctx, rewritten);
  ASSERT_INT_EQ(shape.ndim, 1);
  ASSERT_INT_EQ(shape.dims[0], 0);
  ASSERT_EQ(rewritten->src[0]->op, POLY_OP_RESHAPE);
  ASSERT_INT_EQ(rewritten->src[0]->tag, 0);
  ASSERT_EQ(rewritten->src[0]->src[0]->op, POLY_OP_CONST);
  ASSERT_INT_EQ(rewritten->src[0]->src[0]->tag, 0);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, zero_size_graph_eliminates_kernel) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *empty = poly_full(ctx, (int64_t[]){0}, 1, 0.0);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, 1);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, empty));

  /* Port of tinygrad rangeify.py's general "handle size 0" rule.  A
   * zero-shaped STORE body becomes SINK(CONST(0)), so schedule_linear emits
   * an empty LINEAR instead of a zero-trip kernel with an invalid value. */
  PolyUOp *kernel_graph = poly_get_kernel_graph(ctx, sink);
  ASSERT_NOT_NULL(kernel_graph);
  ASSERT_EQ(kernel_graph->op, POLY_OP_SINK);
  ASSERT_INT_EQ(kernel_graph->n_src, 1);
  ASSERT_EQ(kernel_graph->src[0]->op, POLY_OP_CONST);
  ASSERT_INT_EQ(count_ops(ctx, kernel_graph, POLY_OP_STORE), 0);
  ASSERT_INT_EQ(count_ops(ctx, kernel_graph, POLY_OP_RANGE), 0);

  PolySchedule *sched = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_NOT_NULL(sched);
  ASSERT_INT_EQ(sched->template->n_calls, 0);
  ASSERT_NOT_NULL(sched->template->linear);
  ASSERT_EQ(sched->template->linear->op, POLY_OP_LINEAR);
  ASSERT_INT_EQ(sched->template->linear->n_src, 0);
  ASSERT_INT_EQ(poly_run_schedule(ctx, sched, NULL, 0), 0);
  PolyCtxStats stats;
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &stats), 0);
  ASSERT_INT_EQ(stats.kernel_count, 0);
  ASSERT_INT_EQ(stats.global_ops, 0);

  poly_schedule_free(sched);
  poly_ctx_destroy(ctx);
  PASS();
}

/* Stage C: earliest_rewrites */

TEST(rangeify, earliest_pm_mops_moves_after_before_reshape_cleanup) {
  /* Pinned tinygrad pm_mops first moves the inner RESHAPE through AFTER,
   * preserving every effect. mop_cleanup can then merge the now-adjacent
   * reshapes. earliest_rewrites then makes the SINK reference only the base
   * AFTER while a real consumer retains the merged RESHAPE. This is the exact
   * topology needed by lazy shaped optimizer state before rangeify adds
   * executable INDEX nodes. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *base = poly_buffer(ctx, POLY_FLOAT32, 6);
  int64_t shaped_dims[] = {2, 3};
  int64_t flat_dims[] = {6};
  PolyUOp *shaped = poly_reshape(ctx, base, shaped_dims, 2);
  PolyUOp *shaped_src[POLY_MAX_DIMS + 1];
  for (int i = 0; i < shaped->n_src; i++)
    shaped_src[i] = shaped->src[i];
  shaped =
      poly_uop_tagged(ctx, shaped->op, shaped->dtype, shaped_src, shaped->n_src, shaped->arg, 701);
  PolyUOp *effect0 = poly_uop_tagged(ctx, POLY_OP_NOOP, POLY_VOID, NULL, 0, poly_arg_none(), 711);
  PolyUOp *effect1 = poly_uop_tagged(ctx, POLY_OP_NOOP, POLY_VOID, NULL, 0, poly_arg_none(), 712);
  PolyUOp *after_src[] = {shaped, effect0, effect1};
  PolyUOp *after =
      poly_uop_tagged(ctx, POLY_OP_AFTER, shaped->dtype, after_src, 3, poly_arg_none(), 702);
  PolyUOp *outer = poly_reshape(ctx, after, flat_dims, 1);
  PolyUOp *outer_src[POLY_MAX_DIMS + 1];
  for (int i = 0; i < outer->n_src; i++)
    outer_src[i] = outer->src[i];
  outer = poly_uop_tagged(ctx, outer->op, outer->dtype, outer_src, outer->n_src, outer->arg, 703);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, 6);
  PolyUOp *consumer_store = poly_store_val(ctx, out, outer);
  PolyUOp *sink_src[] = {outer, consumer_store};
  PolyUOp *sink = poly_uop(ctx, POLY_OP_SINK, POLY_VOID, sink_src, 2, poly_arg_none());

  PolyUOp *rewritten = poly_apply_earliest_rewrites(ctx, sink);
  ASSERT_NOT_NULL(rewritten);
  ASSERT_EQ(rewritten->op, POLY_OP_SINK);
  ASSERT_INT_EQ(rewritten->n_src, 2);
  PolyUOp *moved_after = rewritten->src[0];
  ASSERT_EQ(moved_after->op, POLY_OP_AFTER);
  ASSERT_INT_EQ(moved_after->tag, 702);
  ASSERT_INT_EQ(moved_after->n_src, 3);
  ASSERT_PTR_EQ(moved_after->src[0], base);
  ASSERT_PTR_EQ(moved_after->src[1], effect0);
  ASSERT_PTR_EQ(moved_after->src[2], effect1);

  PolyUOp *rewritten_store = rewritten->src[1];
  ASSERT_EQ(rewritten_store->op, POLY_OP_STORE);
  ASSERT_INT_EQ(rewritten_store->n_src, 2);
  PolyUOp *flat = rewritten_store->src[1];
  ASSERT_EQ(flat->op, POLY_OP_RESHAPE);
  ASSERT_INT_EQ(flat->tag, 703);
  ASSERT_INT_EQ(count_ops(ctx, rewritten, POLY_OP_RESHAPE), 1);
  ASSERT_INT_EQ(flat->arg.kind, POLY_ARG_NONE);
  ASSERT_INT_EQ(flat->n_src, 2);
  ASSERT_PTR_EQ(flat->src[0], moved_after);
  ASSERT_INT_EQ(flat->src[1]->op, POLY_OP_STACK);
  ASSERT_INT_EQ(flat->src[1]->n_src, 1);
  ASSERT_INT_EQ(flat->src[1]->src[0]->op, POLY_OP_CONST);
  ASSERT_INT_EQ(flat->src[1]->src[0]->arg.i, 6);
  PolyShape flat_shape = poly_uop_max_shape_cached(ctx, flat);
  ASSERT_INT_EQ(flat_shape.ndim, 1);
  ASSERT_INT_EQ(flat_shape.dims[0], 6);

  PolyIndexingCtx *ictx = poly_indexing_ctx_new(ctx);
  ASSERT_NOT_NULL(ictx);
  poly_realize_map_build(ictx, rewritten);
  ASSERT_FALSE(poly_is_realized(ictx, flat));
  ASSERT_FALSE(poly_is_realized(ictx, moved_after));
  ASSERT_TRUE(poly_is_realized(ictx, rewritten_store));
  poly_indexing_ctx_destroy(ictx);

  /* The combined pre-rangeify normalization is already at a fixpoint. */
  ASSERT_PTR_EQ(poly_apply_earliest_rewrites(ctx, rewritten), rewritten);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, earliest_pm_mops_removes_movement_from_end) {
  /* Pinned pm_mops retains END and all ended ranges while removing a movement
   * op from END.src[0]. END metadata is preserved by UOp.replace. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *base = poly_buffer(ctx, POLY_FLOAT32, 6);
  int64_t shaped_dims[] = {2, 3};
  PolyUOp *movement = poly_reshape(ctx, base, shaped_dims, 2);
  PolyUOp *movement_src[POLY_MAX_DIMS + 1];
  for (int i = 0; i < movement->n_src; i++)
    movement_src[i] = movement->src[i];
  movement = poly_uop_tagged(
      ctx, movement->op, movement->dtype, movement_src, movement->n_src, movement->arg, 721
  );
  PolyUOp *r0 = poly_uop_range(ctx, 2, 0, POLY_AXIS_LOOP);
  PolyUOp *r1 = poly_uop_range(ctx, 3, 1, POLY_AXIS_LOOP);
  PolyUOp *end_src[] = {movement, r0, r1};
  PolyUOp *end = poly_uop_tagged(ctx, POLY_OP_END, POLY_VOID, end_src, 3, poly_arg_none(), 722);
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());

  PolyUOp *rewritten = poly_apply_earliest_rewrites(ctx, sink);
  ASSERT_NOT_NULL(rewritten);
  ASSERT_EQ(rewritten->op, POLY_OP_SINK);
  ASSERT_INT_EQ(rewritten->n_src, 1);
  PolyUOp *rewritten_end = rewritten->src[0];
  ASSERT_EQ(rewritten_end->op, POLY_OP_END);
  ASSERT_INT_EQ(rewritten_end->tag, 722);
  ASSERT_INT_EQ(rewritten_end->n_src, 3);
  ASSERT_PTR_EQ(rewritten_end->src[0], base);
  ASSERT_PTR_EQ(rewritten_end->src[1], r0);
  ASSERT_PTR_EQ(rewritten_end->src[2], r1);
  ASSERT_INT_EQ(count_ops(ctx, rewritten, POLY_OP_RESHAPE), 0);
  ASSERT_PTR_EQ(poly_apply_earliest_rewrites(ctx, rewritten), rewritten);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, earliest_split_reduceop_static_threshold_topology) {
  /* Pinned rangeify.py:102-123:
   * REDUCE(32768, axis 0) becomes
   * RESHAPE(256,128) -> PERMUTE(128,256) -> REDUCE(axis 0) ->
   * CONTIGUOUS -> REDUCE(axis 1). SINK strips the final output RESHAPE to its
   * base, exactly like pinned earliest_rewrites. */
  RangeifyEnvSave split = rangeify_save_env("SPLIT_REDUCEOP");
  RangeifyEnvSave threshold = rangeify_save_env("REDUCEOP_SPLIT_THRESHOLD");
  RangeifyEnvSave size = rangeify_save_env("REDUCEOP_SPLIT_SIZE");
  setenv("SPLIT_REDUCEOP", "1", 1);
  setenv("REDUCEOP_SPLIT_THRESHOLD", "32768", 1);
  setenv("REDUCEOP_SPLIT_SIZE", "22", 1);

  PolyCtx *ctx = poly_ctx_new();
  int64_t axes[] = {0};
  PolyUOp *input = poly_buffer(ctx, POLY_FLOAT32, 32768);
  PolyUOp *reduce = poly_reduce_axis(ctx, POLY_OP_ADD, input, axes, 1);
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, reduce, poly_arg_none());
  PolyUOp *rewritten = poly_apply_earliest_rewrites(ctx, sink);
  PolyUOp *rerewritten = poly_apply_earliest_rewrites(ctx, rewritten);

  rangeify_restore_env(&size);
  rangeify_restore_env(&threshold);
  rangeify_restore_env(&split);

  ASSERT_NOT_NULL(rewritten);
  ASSERT_EQ(rewritten->op, POLY_OP_SINK);
  ASSERT_INT_EQ(count_reduce_arg_kind(ctx, rewritten, POLY_ARG_REDUCE_AXIS), 2);
  ASSERT_INT_EQ(count_ops(ctx, rewritten, POLY_OP_CONTIGUOUS), 1);
  ASSERT_INT_EQ(count_ops(ctx, rewritten, POLY_OP_PERMUTE), 1);
  ASSERT_INT_EQ(count_ops(ctx, rewritten, POLY_OP_RESHAPE), 1);

  PolyUOp *second_reduce = rewritten->src[0];
  ASSERT_EQ(second_reduce->op, POLY_OP_REDUCE);
  ASSERT_EQ(second_reduce->arg.kind, POLY_ARG_REDUCE_AXIS);
  ASSERT_INT_EQ(second_reduce->arg.reduce_axis.n, 1);
  ASSERT_INT_EQ(second_reduce->arg.reduce_axis.axes[0], 1);
  PolyUOp *contiguous = second_reduce->src[0];
  ASSERT_EQ(contiguous->op, POLY_OP_CONTIGUOUS);
  PolyUOp *first_reduce = contiguous->src[0];
  ASSERT_EQ(first_reduce->op, POLY_OP_REDUCE);
  ASSERT_INT_EQ(first_reduce->arg.reduce_axis.n, 1);
  ASSERT_INT_EQ(first_reduce->arg.reduce_axis.axes[0], 0);
  PolyUOp *permuted = first_reduce->src[0];
  ASSERT_EQ(permuted->op, POLY_OP_PERMUTE);
  ASSERT_EQ(permuted->arg.kind, POLY_ARG_INT_TUPLE);
  ASSERT_INT_EQ(permuted->arg.int_tuple.n, 2);
  ASSERT_INT_EQ(permuted->arg.int_tuple.vals[0], 1);
  ASSERT_INT_EQ(permuted->arg.int_tuple.vals[1], 0);
  PolyUOp *reshaped = permuted->src[0];
  ASSERT_EQ(reshaped->op, POLY_OP_RESHAPE);
  ASSERT_EQ(reshaped->arg.kind, POLY_ARG_NONE);
  ASSERT_INT_EQ(reshaped->n_src, 2);
  ASSERT_EQ(reshaped->src[1]->op, POLY_OP_STACK);
  ASSERT_INT_EQ(reshaped->src[1]->n_src, 2);
  ASSERT_EQ(reshaped->src[1]->src[0]->op, POLY_OP_CONST);
  ASSERT_EQ(reshaped->src[1]->src[1]->op, POLY_OP_CONST);
  ASSERT_INT_EQ(reshaped->src[1]->src[0]->arg.i, 256);
  ASSERT_INT_EQ(reshaped->src[1]->src[1]->arg.i, 128);
  ASSERT_PTR_EQ(reshaped->src[0], input);
  ASSERT_PTR_EQ(rerewritten, rewritten);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, earliest_split_reduceop_rejects_expanded_axis) {
  /* Pinned split_reduceop rangeifies x before selecting a split axis. An
   * EXPANDed axis has no surviving range marker and is never a candidate. */
  RangeifyEnvSave split = rangeify_save_env("SPLIT_REDUCEOP");
  RangeifyEnvSave threshold = rangeify_save_env("REDUCEOP_SPLIT_THRESHOLD");
  setenv("SPLIT_REDUCEOP", "1", 1);
  setenv("REDUCEOP_SPLIT_THRESHOLD", "32768", 1);

  PolyCtx *ctx = poly_ctx_new();
  int64_t expanded_shape[] = {32768};
  int64_t axes[] = {0};
  PolyUOp *input = poly_expand(ctx, poly_buffer(ctx, POLY_FLOAT32, 1), expanded_shape, 1);
  PolyUOp *reduce = poly_reduce_axis(ctx, POLY_OP_ADD, input, axes, 1);
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, reduce, poly_arg_none());
  PolyUOp *rewritten = poly_apply_earliest_rewrites(ctx, sink);

  rangeify_restore_env(&threshold);
  rangeify_restore_env(&split);

  ASSERT_INT_EQ(count_reduce_arg_kind(ctx, rewritten, POLY_ARG_REDUCE_AXIS), 1);
  ASSERT_INT_EQ(count_ops(ctx, rewritten, POLY_OP_CONTIGUOUS), 0);
  ASSERT_INT_EQ(count_ops(ctx, rewritten, POLY_OP_PERMUTE), 0);
  ASSERT_PTR_EQ(rewritten->src[0], reduce);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, earliest_split_reduceop_disabled_control) {
  /* Pinned SPLIT_REDUCEOP=0 leaves the original reduction untouched. */
  RangeifyEnvSave split = rangeify_save_env("SPLIT_REDUCEOP");
  setenv("SPLIT_REDUCEOP", "0", 1);

  PolyCtx *ctx = poly_ctx_new();
  int64_t axes[] = {0};
  PolyUOp *input = poly_buffer(ctx, POLY_FLOAT32, 32768);
  PolyUOp *reduce = poly_reduce_axis(ctx, POLY_OP_ADD, input, axes, 1);
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, reduce, poly_arg_none());
  PolyUOp *rewritten = poly_apply_earliest_rewrites(ctx, sink);

  rangeify_restore_env(&split);

  ASSERT_INT_EQ(count_reduce_arg_kind(ctx, rewritten, POLY_ARG_REDUCE_AXIS), 1);
  ASSERT_INT_EQ(count_ops(ctx, rewritten, POLY_OP_CONTIGUOUS), 0);
  ASSERT_PTR_EQ(rewritten->src[0], reduce);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, earliest_split_reduceop_descending_divisor_order) {
  /* Pinned candidates test every divisor from min(256,budget/output) down to
   * 8, not only powers of two. 384 first accepts 192. */
  RangeifyEnvSave split = rangeify_save_env("SPLIT_REDUCEOP");
  RangeifyEnvSave threshold = rangeify_save_env("REDUCEOP_SPLIT_THRESHOLD");
  RangeifyEnvSave size = rangeify_save_env("REDUCEOP_SPLIT_SIZE");
  setenv("SPLIT_REDUCEOP", "1", 1);
  setenv("REDUCEOP_SPLIT_THRESHOLD", "1", 1);
  setenv("REDUCEOP_SPLIT_SIZE", "22", 1);

  PolyCtx *ctx = poly_ctx_new();
  int64_t axes[] = {0};
  PolyUOp *input = poly_buffer(ctx, POLY_FLOAT32, 384);
  PolyUOp *reduce = poly_reduce_axis(ctx, POLY_OP_ADD, input, axes, 1);
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, reduce, poly_arg_none());
  PolyUOp *rewritten = poly_apply_earliest_rewrites(ctx, sink);

  rangeify_restore_env(&size);
  rangeify_restore_env(&threshold);
  rangeify_restore_env(&split);

  PolyUOp *reshaped = rewritten->src[0]->src[0]->src[0]->src[0]->src[0];
  ASSERT_EQ(reshaped->op, POLY_OP_RESHAPE);
  ASSERT_EQ(reshaped->arg.kind, POLY_ARG_NONE);
  ASSERT_INT_EQ(reshaped->n_src, 2);
  ASSERT_EQ(reshaped->src[1]->op, POLY_OP_STACK);
  ASSERT_INT_EQ(reshaped->src[1]->n_src, 2);
  ASSERT_EQ(reshaped->src[1]->src[0]->op, POLY_OP_CONST);
  ASSERT_EQ(reshaped->src[1]->src[1]->op, POLY_OP_CONST);
  ASSERT_INT_EQ(reshaped->src[1]->src[0]->arg.i, 192);
  ASSERT_INT_EQ(reshaped->src[1]->src[1]->arg.i, 2);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, earliest_split_reduceop_rejects_symbolic_shape) {
  /* Pinned all_int(x.shape) rejects even a bounded symbolic dimension. */
  RangeifyEnvSave split = rangeify_save_env("SPLIT_REDUCEOP");
  RangeifyEnvSave threshold = rangeify_save_env("REDUCEOP_SPLIT_THRESHOLD");
  setenv("SPLIT_REDUCEOP", "1", 1);
  setenv("REDUCEOP_SPLIT_THRESHOLD", "1", 1);

  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *n = poly_define_var(ctx, "split_n", 1, 65536);
  PolyUOp *input = poly_buffer_var(ctx, POLY_FLOAT32, n, NULL, 0);
  int64_t axes[] = {0};
  PolyUOp *reduce = poly_reduce_axis(ctx, POLY_OP_ADD, input, axes, 1);
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, reduce, poly_arg_none());
  PolyUOp *rewritten = poly_apply_earliest_rewrites(ctx, sink);

  rangeify_restore_env(&threshold);
  rangeify_restore_env(&split);

  ASSERT_INT_EQ(count_reduce_arg_kind(ctx, rewritten, POLY_ARG_REDUCE_AXIS), 1);
  ASSERT_INT_EQ(count_ops(ctx, rewritten, POLY_OP_CONTIGUOUS), 0);
  ASSERT_PTR_EQ(rewritten->src[0], reduce);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, earliest_reshape_merge) {
  /* RESHAPE(RESHAPE(x, [4,2]), [8]): verify correct output.
   * poly_earliest_rewrites should merge into RESHAPE(x, [8]). */
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, 8);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, 8);

  int64_t sh42[] = {4, 2};
  int64_t sh8[] = {8};
  PolyUOp *r1 = poly_reshape(ctx, a, sh42, 2);
  PolyUOp *r2 = poly_reshape(ctx, r1, sh8, 1);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out, r2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  float a_d[] = {1, 2, 3, 4, 5, 6, 7, 8};
  float out_d[8] = {0};
  PolyTestBufferView bindings[] = {
      POLY_TEST_HOST_VIEW(out, out_d),
      POLY_TEST_HOST_VIEW(a, a_d),
  };
  int ret = poly_test_realize_buffer_views(ctx, sink, bindings, 2);
  poly_ctx_destroy(ctx);
  ASSERT_INT_EQ(ret, 0);

  for (int i = 0; i < 8; i++) {
    ASSERT_FLOAT_EQ(out_d[i], a_d[i], 1e-5);
  }
  PASS();
}

TEST(rangeify, earliest_function_exposes_and_removes_detach) {
  /* Pinned schedule/rangeify.py:150-164 resolves FUNCTION and then traverses
   * the substituted body in the same bottom-up graph_rewrite.  A DETACH in
   * that newly exposed body must therefore be gone before LINEAR/rendering. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *eight = poly_const_int(ctx, 8);
  PolyUOp *shape_src[1] = {eight};
  PolyUOp *shape =
      poly_uop(ctx, POLY_OP_STACK, poly_dtype_vec(POLY_INDEX, 1), shape_src, 1, poly_arg_none());
  PolyParamArg lhs_arg = {.slot = 0, .addrspace = POLY_ADDR_GLOBAL, .device = "CPU"};
  PolyParamArg rhs_arg = {.slot = 1, .addrspace = POLY_ADDR_GLOBAL, .device = "CPU"};
  PolyUOp *lhs_param = poly_uop1(ctx, POLY_OP_PARAM, POLY_FLOAT32, shape, poly_arg_param(&lhs_arg));
  PolyUOp *rhs_param = poly_uop1(ctx, POLY_OP_PARAM, POLY_FLOAT32, shape, poly_arg_param(&rhs_arg));
  PolyUOp *detached = poly_uop1(ctx, POLY_OP_DETACH, POLY_FLOAT32, rhs_param, poly_arg_none());
  PolyUOp *sub = poly_alu2(ctx, POLY_OP_SUB, lhs_param, detached);
  PolyUOp *body = poly_uop1(ctx, POLY_OP_TUPLE, POLY_VOID, sub, poly_arg_none());

  PolyUOp *lhs = poly_buffer(ctx, POLY_FLOAT32, 8);
  PolyUOp *rhs = poly_buffer(ctx, POLY_FLOAT32, 8);
  PolyUOp *function_src[3] = {body, lhs, rhs};
  PolyCallInfo info = {.name = "centered", .precompile = false};
  PolyUOp *function =
      poly_uop(ctx, POLY_OP_FUNCTION, POLY_VOID, function_src, 3, poly_arg_call_info(&info));
  PolyUOp *selected = poly_uop1(ctx, POLY_OP_GETTUPLE, POLY_FLOAT32, function, poly_arg_int(0));
  ASSERT_NOT_NULL(selected);
  ASSERT_INT_EQ(count_ops(ctx, selected, POLY_OP_FUNCTION), 1);
  ASSERT_INT_EQ(count_ops(ctx, selected, POLY_OP_GETTUPLE), 1);
  ASSERT_INT_EQ(count_ops(ctx, selected, POLY_OP_DETACH), 1);

  PolyUOp *rewritten = poly_apply_earliest_rewrites(ctx, selected);
  ASSERT_NOT_NULL(rewritten);
  ASSERT_INT_EQ(rewritten->op, POLY_OP_SUB);
  ASSERT_INT_EQ(count_ops(ctx, rewritten, POLY_OP_FUNCTION), 0);
  ASSERT_INT_EQ(count_ops(ctx, rewritten, POLY_OP_GETTUPLE), 0);
  ASSERT_INT_EQ(count_ops(ctx, rewritten, POLY_OP_DETACH), 0);
  ASSERT_PTR_EQ(rewritten->src[0], lhs);
  ASSERT_PTR_EQ(rewritten->src[1], rhs);

  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, 8);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out, selected, poly_arg_none());
  PolyUOp *sink = poly_sink1(ctx, store);
  float lhs_data[8], rhs_data[8], out_data[8] = {0};
  for (int i = 0; i < 8; i++) {
    lhs_data[i] = (float)(i * 3 + 1);
    rhs_data[i] = (float)(i + 2);
  }
  PolyTestBufferView bindings[] = {
      POLY_TEST_HOST_VIEW(out, out_data),
      POLY_TEST_HOST_VIEW(lhs, lhs_data),
      POLY_TEST_HOST_VIEW(rhs, rhs_data),
  };
  ASSERT_INT_EQ(poly_test_realize_buffer_views(ctx, sink, bindings, 3), 0);
  for (int i = 0; i < 8; i++)
    ASSERT_FLOAT_EQ(out_data[i], lhs_data[i] - rhs_data[i], 0.0f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, earliest_detach_removal) {
  /* DETACH(a) + b -> STORE: verify DETACH is stripped and result is correct */
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, 4);
  PolyUOp *b = poly_buffer(ctx, POLY_FLOAT32, 4);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, 4);
  PolyUOp *detached = poly_uop1(ctx, POLY_OP_DETACH, POLY_FLOAT32, a, poly_arg_none());
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, detached, b, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out, add, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  float a_d[] = {1, 2, 3, 4};
  float b_d[] = {10, 20, 30, 40};
  float out_d[4] = {0};
  PolyTestBufferView bindings[] = {
      POLY_TEST_HOST_VIEW(out, out_d),
      POLY_TEST_HOST_VIEW(a, a_d),
      POLY_TEST_HOST_VIEW(b, b_d),
  };
  int ret = poly_test_realize_buffer_views(ctx, sink, bindings, 3);
  poly_ctx_destroy(ctx);
  ASSERT_INT_EQ(ret, 0);

  for (int i = 0; i < 4; i++) {
    ASSERT_FLOAT_EQ(out_d[i], a_d[i] + b_d[i], 1e-5);
  }
  PASS();
}

/* Stage 3.25: pm_limit_bufs */

TEST(rangeify, limit_bufs_ir) {
  /* 10 independent buffers chained: b0 + b1 + ... + b9 → STORE(out).
   * With POLY_MAX_KERNEL_BUFFERS=8, the scheduler must split so that
   * every kernel has <= 8 params (7 inputs + 1 output). */
  const int N = 16;
  const int N_BUFS = 10;

  /* Save and set env */
  const char *old = getenv("POLY_MAX_KERNEL_BUFFERS");
  char *olddup = old ? strdup(old) : NULL;
  setenv("POLY_MAX_KERNEL_BUFFERS", "8", 1);

  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *bufs[10];
  for (int i = 0; i < N_BUFS; i++)
    bufs[i] = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, N);

  /* Chain: bufs[0] + bufs[1] + ... + bufs[9] */
  PolyUOp *acc = bufs[0];
  for (int i = 1; i < N_BUFS; i++)
    acc = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, acc, bufs[i], poly_arg_none());

  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out, acc, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyKernelScheduleResult sr = poly_build_kernel_schedule(ctx, sink);

  /* Extract results before cleanup */
  int n_kernels = sr.n_kernels;
  int n_intermediates = sr.n_intermediates;
  bool all_under_limit = true;
  for (int k = 0; k < sr.n_kernels; k++) {
    if (sr.kernel_n_params[k] > 8) all_under_limit = false;
  }

  poly_kernel_schedule_result_free(&sr);
  poly_ctx_destroy(ctx);

  /* Restore env */
  if (olddup) {
    setenv("POLY_MAX_KERNEL_BUFFERS", olddup, 1);
    free(olddup);
  } else
    unsetenv("POLY_MAX_KERNEL_BUFFERS");

  ASSERT_TRUE(all_under_limit); /* primary: every kernel within limit */
  ASSERT_TRUE(n_kernels > 1); /* must split */
  ASSERT_TRUE(n_intermediates > 0); /* at least one intermediate */
  PASS();
}

TEST(rangeify, limit_bufs_e2e) {
  /* Same 10-buffer ADD chain, verify numerical correctness.
   * Each buffer has value (i+1), so out[j] = 1+2+...+10 = 55. */
  const int N = 16;
  const int N_BUFS = 10;

  const char *old = getenv("POLY_MAX_KERNEL_BUFFERS");
  char *olddup = old ? strdup(old) : NULL;
  setenv("POLY_MAX_KERNEL_BUFFERS", "8", 1);

  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *bufs[10];
  for (int i = 0; i < N_BUFS; i++)
    bufs[i] = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, N);

  PolyUOp *acc = bufs[0];
  for (int i = 1; i < N_BUFS; i++)
    acc = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, acc, bufs[i], poly_arg_none());

  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out, acc, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  float buf_data[10][16];
  for (int i = 0; i < N_BUFS; i++)
    for (int j = 0; j < N; j++)
      buf_data[i][j] = (float)(i + 1);
  float out_d[16] = {0};

  PolyTestBufferView bindings[12]; /* out + 10 inputs */
  bindings[0] = POLY_TEST_HOST_VIEW(out, out_d);
  for (int i = 0; i < N_BUFS; i++)
    bindings[1 + i] = POLY_TEST_HOST_VIEW(bufs[i], buf_data[i]);

  int ret = poly_test_realize_buffer_views(ctx, sink, bindings, 1 + N_BUFS);

  /* Copy before cleanup */
  float out_copy[16];
  for (int j = 0; j < N; j++)
    out_copy[j] = out_d[j];

  poly_ctx_destroy(ctx);

  /* Restore env */
  if (olddup) {
    setenv("POLY_MAX_KERNEL_BUFFERS", olddup, 1);
    free(olddup);
  } else
    unsetenv("POLY_MAX_KERNEL_BUFFERS");

  ASSERT_INT_EQ(ret, 0);
  /* Expected: 1+2+...+10 = 55 at every position */
  for (int j = 0; j < N; j++)
    ASSERT_FLOAT_EQ(out_copy[j], 55.0f, 1e-5);
  PASS();
}

TEST(rangeify, limit_bufs_noop) {
  /* 3 buffers (a + b + c), well under limit=8. Should produce 1 kernel. */
  const int N = 16;

  const char *old = getenv("POLY_MAX_KERNEL_BUFFERS");
  char *olddup = old ? strdup(old) : NULL;
  setenv("POLY_MAX_KERNEL_BUFFERS", "8", 1);

  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *b = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *c = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, N);

  PolyUOp *ab = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, a, b, poly_arg_none());
  PolyUOp *abc = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, ab, c, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out, abc, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyKernelScheduleResult sr = poly_build_kernel_schedule(ctx, sink);
  int n_kernels = sr.n_kernels;
  int n_intermediates = sr.n_intermediates;
  poly_kernel_schedule_result_free(&sr);
  poly_ctx_destroy(ctx);

  if (olddup) {
    setenv("POLY_MAX_KERNEL_BUFFERS", olddup, 1);
    free(olddup);
  } else
    unsetenv("POLY_MAX_KERNEL_BUFFERS");

  ASSERT_INT_EQ(n_kernels, 1);
  ASSERT_INT_EQ(n_intermediates, 0);
  PASS();
}

TEST(rangeify, limit_bufs_disabled) {
  /* 10 buffers but POLY_MAX_KERNEL_BUFFERS not set (default 0).
   * Pass is disabled, should produce 1 kernel (all fused). */
  const int N = 16;
  const int N_BUFS = 10;

  /* Ensure env var is unset */
  const char *old = getenv("POLY_MAX_KERNEL_BUFFERS");
  char *olddup = old ? strdup(old) : NULL;
  unsetenv("POLY_MAX_KERNEL_BUFFERS");

  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *bufs[10];
  for (int i = 0; i < N_BUFS; i++)
    bufs[i] = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, N);

  PolyUOp *acc = bufs[0];
  for (int i = 1; i < N_BUFS; i++)
    acc = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, acc, bufs[i], poly_arg_none());

  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out, acc, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyKernelScheduleResult sr = poly_build_kernel_schedule(ctx, sink);
  int n_kernels = sr.n_kernels;
  poly_kernel_schedule_result_free(&sr);
  poly_ctx_destroy(ctx);

  /* Restore env */
  if (olddup) {
    setenv("POLY_MAX_KERNEL_BUFFERS", olddup, 1);
    free(olddup);
  }

  ASSERT_INT_EQ(n_kernels, 1);
  PASS();
}

/* In-place update + WAR ordering tests */

/* assign_e2e: a.assign(a + b) — basic in-place update */
TEST(rangeify, assign_e2e) {
  const int N = 4;
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *buf_a = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *buf_b = poly_buffer(ctx, POLY_FLOAT32, N);

  /* value = a + b */
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, buf_a, buf_b, poly_arg_none());

  /* STORE(a, a + b) */
  PolyUOp *assign = poly_store_buffer_update(ctx, buf_a, add);

  /* SINK(STORE) -- optimizer/core in-place effect */
  PolyUOp *sink = poly_uop(ctx, POLY_OP_SINK, POLY_VOID, &assign, 1, poly_arg_none());

  float a_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float b_data[4] = {10.0f, 20.0f, 30.0f, 40.0f};

  PolyTestBufferView bindings[2] = {
      POLY_TEST_HOST_VIEW(buf_a, a_data),
      POLY_TEST_HOST_VIEW(buf_b, b_data),
  };

  int ret = poly_test_realize_buffer_views(ctx, sink, bindings, 2);
  float result[4];
  memcpy(result, a_data, sizeof(result));

  poly_ctx_destroy(ctx);

  ASSERT_INT_EQ(ret, 0);
  ASSERT_FLOAT_EQ(result[0], 11.0f, 1e-5);
  ASSERT_FLOAT_EQ(result[1], 22.0f, 1e-5);
  ASSERT_FLOAT_EQ(result[2], 33.0f, 1e-5);
  ASSERT_FLOAT_EQ(result[3], 44.0f, 1e-5);
  PASS();
}

/* assign_ir: verify in-place update scheduling produces correct kernel structure */
TEST(rangeify, assign_ir) {
  const int N = 4;
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *buf_a = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *buf_b = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, buf_a, buf_b, poly_arg_none());
  PolyUOp *assign = poly_store_buffer_update(ctx, buf_a, add);
  PolyUOp *sink = poly_uop(ctx, POLY_OP_SINK, POLY_VOID, &assign, 1, poly_arg_none());

  PolyKernelScheduleResult sr = poly_build_kernel_schedule(ctx, sink);

  int got_kernels = sr.n_kernels;
  int got_inter = sr.n_intermediates;

  /* Check that the update kernel writes to buf_a (existing buffer). */
  bool writes_buf_a = false;
  for (int k = 0; k < sr.n_kernels; k++) {
    for (int p = 0; p < sr.kernel_n_params[k]; p++) {
      if (sr.param_to_buf[k][p] == buf_a) writes_buf_a = true;
    }
  }

  poly_kernel_schedule_result_free(&sr);
  poly_ctx_destroy(ctx);

  /* 1 in-place update kernel, 0 consumer stores, 0 intermediates */
  ASSERT_INT_EQ(got_kernels, 1);
  ASSERT_INT_EQ(got_inter, 0);
  ASSERT_TRUE(writes_buf_a);
  PASS();
}

/* assign_war_ordering: reader kernel must complete before in-place writer.
 * Graph: a[4], out[4] = a + 10 (separate STORE kernel reading a),
 *        STORE(a, a * 2) (writes a).
 * WAR: out's kernel reads a, update writes a -> reader before writer.
 * Verify: out = {11, 12, 13, 14} (pre-update values), a = {2, 4, 6, 8}. */
TEST(rangeify, assign_war_ordering) {
  const int N = 4;
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *buf_a = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *buf_out = poly_buffer(ctx, POLY_FLOAT32, N);

  /* STORE: out = a + 10 */
  PolyUOp *ten = poly_const_float(ctx, 10.0);
  PolyUOp *a_plus_10 = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, buf_a, ten, poly_arg_none());
  PolyUOp *store_out =
      poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, buf_out, a_plus_10, poly_arg_none());

  /* In-place update: a = a * 2 */
  PolyUOp *two = poly_const_float(ctx, 2.0);
  PolyUOp *a_times_2 = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, buf_a, two, poly_arg_none());
  PolyUOp *assign = poly_store_buffer_update(ctx, buf_a, a_times_2);

  /* SINK(STORE(out, a+10), STORE(a, a*2)) */
  PolyUOp *sink_src[2] = {store_out, assign};
  PolyUOp *sink = poly_uop(ctx, POLY_OP_SINK, POLY_VOID, sink_src, 2, poly_arg_none());

  float a_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float out_data[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  PolyTestBufferView bindings[2] = {
      POLY_TEST_HOST_VIEW(buf_a, a_data),
      POLY_TEST_HOST_VIEW(buf_out, out_data),
  };

  int ret = poly_test_realize_buffer_views(ctx, sink, bindings, 2);
  float a_result[4], out_result[4];
  memcpy(a_result, a_data, sizeof(a_result));
  memcpy(out_result, out_data, sizeof(out_result));

  poly_ctx_destroy(ctx);

  ASSERT_INT_EQ(ret, 0);
  /* out should see pre-update values of a (WAR ordering) */
  ASSERT_FLOAT_EQ(out_result[0], 11.0f, 1e-5);
  ASSERT_FLOAT_EQ(out_result[1], 12.0f, 1e-5);
  ASSERT_FLOAT_EQ(out_result[2], 13.0f, 1e-5);
  ASSERT_FLOAT_EQ(out_result[3], 14.0f, 1e-5);
  /* a should be updated by the in-place effect */
  ASSERT_FLOAT_EQ(a_result[0], 2.0f, 1e-5);
  ASSERT_FLOAT_EQ(a_result[1], 4.0f, 1e-5);
  ASSERT_FLOAT_EQ(a_result[2], 6.0f, 1e-5);
  ASSERT_FLOAT_EQ(a_result[3], 8.0f, 1e-5);
  PASS();
}

/* assign_self_rhs: a.assign(a * 2) — value reads from same buffer as target.
 * This is the basic in-place elementwise update. Should work without
 * fix_assign_hazard because elementwise reads same index it writes. */
TEST(rangeify, assign_self_rhs) {
  const int N = 4;
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *buf_a = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *two = poly_const_float(ctx, 2.0);
  PolyUOp *a_times_2 = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, buf_a, two, poly_arg_none());
  PolyUOp *assign = poly_store_buffer_update(ctx, buf_a, a_times_2);
  PolyUOp *sink = poly_uop(ctx, POLY_OP_SINK, POLY_VOID, &assign, 1, poly_arg_none());

  float a_data[4] = {3.0f, 5.0f, 7.0f, 11.0f};

  PolyTestBufferView bindings[1] = {
      POLY_TEST_HOST_VIEW(buf_a, a_data),
  };

  int ret = poly_test_realize_buffer_views(ctx, sink, bindings, 1);
  float result[4];
  memcpy(result, a_data, sizeof(result));

  poly_ctx_destroy(ctx);

  ASSERT_INT_EQ(ret, 0);
  ASSERT_FLOAT_EQ(result[0], 6.0f, 1e-5);
  ASSERT_FLOAT_EQ(result[1], 10.0f, 1e-5);
  ASSERT_FLOAT_EQ(result[2], 14.0f, 1e-5);
  ASSERT_FLOAT_EQ(result[3], 22.0f, 1e-5);
  PASS();
}

/* Dynamic shapes (DEFINE_VAR) tests */

/* define_var_1d_e2e: a[N] + 1.0 -> out[N] with N=4 */
TEST(rangeify, define_var_1d_e2e) {
  PolyCtx *ctx = poly_ctx_new();

  /* Create symbolic variable N with bounds [1, 16] */
  PolyUOp *N = poly_define_var(ctx, "N", 1, 16);

  /* Create dynamic 1D buffers: a[N], out[N] */
  PolyUOp *buf_a = poly_buffer_var(ctx, POLY_FLOAT32, N, NULL, 0);
  PolyUOp *buf_out = poly_buffer_var(ctx, POLY_FLOAT32, N, NULL, 0);

  /* out = a + 1.0 */
  PolyUOp *one = poly_const_float(ctx, 1.0);
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, buf_a, one, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, buf_out, add, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  /* Execute with N=4 */
  float a_data[16] = {10.0f, 20.0f, 30.0f, 40.0f};
  float out_data[16];
  memset(out_data, 0, sizeof(out_data));

  PolyTestBufferView bindings[2] = {
      POLY_TEST_HOST_VIEW(buf_a, a_data),
      POLY_TEST_HOST_VIEW(buf_out, out_data),
  };
  PolyVarBinding var_bindings[1] = {
      {.var = N, .value = 4},
  };

  int ret = poly_test_realize_buffer_views_vars(ctx, sink, bindings, 2, var_bindings, 1);
  float result[16];
  memcpy(result, out_data, sizeof(result));

  poly_ctx_destroy(ctx);

  ASSERT_INT_EQ(ret, 0);
  /* First 4 elements should be a + 1.0 */
  ASSERT_FLOAT_EQ(result[0], 11.0f, 1e-5);
  ASSERT_FLOAT_EQ(result[1], 21.0f, 1e-5);
  ASSERT_FLOAT_EQ(result[2], 31.0f, 1e-5);
  ASSERT_FLOAT_EQ(result[3], 41.0f, 1e-5);
  /* Elements beyond the bound N are outside the runtime Tensor shape and are
   * not observable. Device allocators need not preserve their staging bytes. */
  PASS();
}

/* define_var_2d_e2e: a[N,4] + 1.0 -> out[N,4] with N=3, then N=2 */
TEST(rangeify, define_var_2d_e2e) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *N = poly_define_var(ctx, "N", 1, 8);

  /* Create 2D dynamic buffers: a[N,4], out[N,4] */
  int64_t inner_dim = 4;
  PolyUOp *buf_a = poly_buffer_var(ctx, POLY_FLOAT32, N, &inner_dim, 1);
  PolyUOp *buf_out = poly_buffer_var(ctx, POLY_FLOAT32, N, &inner_dim, 1);

  /* out = a + 1.0 */
  PolyUOp *one = poly_const_float(ctx, 1.0);
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, buf_a, one, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, buf_out, add, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  /* max alloc = 8*4 = 32 elements */
  float a_data[32];
  float out_data[32];
  for (int i = 0; i < 32; i++)
    a_data[i] = (float)(i + 1);
  memset(out_data, 0, sizeof(out_data));

  PolyTestBufferView bindings[2] = {
      POLY_TEST_HOST_VIEW(buf_a, a_data),
      POLY_TEST_HOST_VIEW(buf_out, out_data),
  };

  /* Execute with N=3 (12 elements) */
  PolyVarBinding var_bindings[1] = {
      {.var = N, .value = 3},
  };

  int ret = poly_test_realize_buffer_views_vars(ctx, sink, bindings, 2, var_bindings, 1);
  float result[32];
  memcpy(result, out_data, sizeof(result));

  poly_ctx_destroy(ctx);

  ASSERT_INT_EQ(ret, 0);
  /* First 12 elements (3 rows of 4) should be a + 1.0 */
  for (int i = 0; i < 12; i++) {
    ASSERT_FLOAT_EQ(result[i], (float)(i + 2), 1e-5);
  }
  /* Elements beyond the bound N are outside the runtime Tensor shape. */
  PASS();
}

/* bind_auto_extract: BIND(N, 4) in buffer source, no explicit var_bindings.
 * BIND stripping auto-extracts N=4 before scheduling. */
TEST(rangeify, bind_auto_extract) {
  PolyCtx *ctx = poly_ctx_new();

  /* Create symbolic variable N and a BIND node with value 4 */
  PolyUOp *N = poly_define_var(ctx, "N", 1, 16);
  PolyUOp *bind_N = poly_bind_var(ctx, N, 4);

  /* Build dynamic 1D buffers with BIND as source instead of bare DEFINE_VAR.
   * strip_bind_values will rewrite BIND(N,4) -> N before scheduling. */
  PolyUOp *unique_a = poly_uop0(ctx, POLY_OP_UNIQUE, POLY_VOID, poly_arg_int(2000000));
  PolyUOp *unique_o = poly_uop0(ctx, POLY_OP_UNIQUE, POLY_VOID, poly_arg_int(2000001));
  PolyUOp *src_a[2] = {unique_a, bind_N};
  PolyUOp *src_o[2] = {unique_o, bind_N};
  PolyUOp *buf_a = poly_uop(ctx, POLY_OP_BUFFER, POLY_FLOAT32, src_a, 2, poly_arg_int(16));
  PolyUOp *buf_out = poly_uop(ctx, POLY_OP_BUFFER, POLY_FLOAT32, src_o, 2, poly_arg_int(16));

  /* out = a + 1.0 */
  PolyUOp *one = poly_const_float(ctx, 1.0);
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, buf_a, one, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, buf_out, add, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  /* Execute with NO explicit var_bindings — BIND auto-extraction provides N=4 */
  float a_data[16] = {10.0f, 20.0f, 30.0f, 40.0f};
  float out_data[16];
  memset(out_data, 0, sizeof(out_data));

  PolyTestBufferView bindings[2] = {
      POLY_TEST_HOST_VIEW(buf_a, a_data),
      POLY_TEST_HOST_VIEW(buf_out, out_data),
  };

  int ret = poly_test_realize_buffer_views_vars(ctx, sink, bindings, 2, NULL, 0);
  float result[16];
  memcpy(result, out_data, sizeof(result));

  poly_ctx_destroy(ctx);

  ASSERT_INT_EQ(ret, 0);
  /* First 4 elements should be a + 1.0 */
  ASSERT_FLOAT_EQ(result[0], 11.0f, 1e-5);
  ASSERT_FLOAT_EQ(result[1], 21.0f, 1e-5);
  ASSERT_FLOAT_EQ(result[2], 31.0f, 1e-5);
  ASSERT_FLOAT_EQ(result[3], 41.0f, 1e-5);
  /* Elements beyond the bound N are outside the runtime Tensor shape. */
  PASS();
}

/* define_var_cache_hit: execute with N=4 then N=8 (reuses compiled kernel).
 * Second call should hit schedule cache and produce correct results. */
TEST(rangeify, define_var_cache_hit) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *N = poly_define_var(ctx, "N", 1, 16);
  PolyUOp *buf_a = poly_buffer_var(ctx, POLY_FLOAT32, N, NULL, 0);
  PolyUOp *buf_out = poly_buffer_var(ctx, POLY_FLOAT32, N, NULL, 0);

  PolyUOp *one = poly_const_float(ctx, 1.0);
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, buf_a, one, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, buf_out, add, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  float a_data[16];
  float out_data[16];
  for (int i = 0; i < 16; i++)
    a_data[i] = (float)(i * 10);

  PolyTestBufferView bindings[2] = {
      POLY_TEST_HOST_VIEW(buf_a, a_data),
      POLY_TEST_HOST_VIEW(buf_out, out_data),
  };

  /* First call: N=4 (compiles kernel) */
  memset(out_data, 0, sizeof(out_data));
  PolyVarBinding var4[1] = {{.var = N, .value = 4}};
  int ret1 = poly_test_realize_buffer_views_vars(ctx, sink, bindings, 2, var4, 1);
  float r1[16];
  memcpy(r1, out_data, sizeof(r1));

  /* Second call: N=8 (should hit cache, reuse compiled kernel) */
  memset(out_data, 0, sizeof(out_data));
  PolyVarBinding var8[1] = {{.var = N, .value = 8}};
  int ret2 = poly_test_realize_buffer_views_vars(ctx, sink, bindings, 2, var8, 1);
  float r2[16];
  memcpy(r2, out_data, sizeof(r2));

  poly_ctx_destroy(ctx);

  ASSERT_INT_EQ(ret1, 0);
  /* First 4 elements correct for N=4 */
  ASSERT_FLOAT_EQ(r1[0], 1.0f, 1e-5);
  ASSERT_FLOAT_EQ(r1[1], 11.0f, 1e-5);
  ASSERT_FLOAT_EQ(r1[2], 21.0f, 1e-5);
  ASSERT_FLOAT_EQ(r1[3], 31.0f, 1e-5);

  ASSERT_INT_EQ(ret2, 0);
  /* First 8 elements correct for N=8 */
  ASSERT_FLOAT_EQ(r2[0], 1.0f, 1e-5);
  ASSERT_FLOAT_EQ(r2[7], 71.0f, 1e-5);
  PASS();
}

/* Regression: chained reductions (singleton + real) */
/* Verifies that tensor REDUCE with a singleton dim (size 1) followed by
 * a real reduction (size > 1) compiles and executes correctly via
 * poly_test_realize_buffer_views. Regression for the bug where CONST(0) pseudo-ranges
 * from singleton dims entered REDUCE sources, producing END(CONST)
 * that corrupted scope depth in the C renderer.
 *
 * Graph shape: (1,4) * (1,4) → reduce axis 0 (singleton, keepdim) → (1,4)
 *              → reduce axis 1 (real sum of 4) → (1,1) → store
 *
 * tensor REDUCE keeps dims (sets reduced axis to 1), so reducing axis 0
 * of (1,4) gives (1,4) (no-op), and reducing axis 1 of (1,4) gives (1,1).
 */
TEST(rangeify, chained_singleton_reduce_e2e) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a_buf = poly_buffer_f32(ctx, 1);
  PolyUOp *w_buf = poly_buffer_f32(ctx, 4);
  PolyUOp *out_buf = poly_buffer_f32(ctx, 1);

  /* a reshaped to (1,4) via expand, w reshaped to (1,4) */
  int64_t sh14[] = {1, 4};
  PolyUOp *a_r = poly_reshape(ctx, a_buf, (int64_t[]){1, 1}, 2);
  PolyUOp *a_e = poly_expand(ctx, a_r, sh14, 2);
  PolyUOp *w_r = poly_reshape(ctx, w_buf, sh14, 2);

  /* Elementwise multiply: (1,4) * (1,4) -> (1,4) */
  PolyUOp *mul = poly_alu2(ctx, POLY_OP_MUL, a_e, w_r);

  /* Reduce axis 0 (size 1 -- singleton): (1,4) -> (1,4) [keepdim, no-op] */
  int64_t ax0[] = {0};
  PolyUOp *r0 = poly_reduce_axis(ctx, POLY_OP_ADD, mul, ax0, 1);

  /* Reduce axis 1 (size 4 -- real sum): (1,4) -> (1,1) */
  int64_t ax1[] = {1};
  PolyUOp *r1 = poly_reduce_axis(ctx, POLY_OP_ADD, r0, ax1, 1);

  /* Reshape to (1,) for store into scalar buffer */
  PolyUOp *r1_flat = poly_reshape(ctx, r1, (int64_t[]){1}, 1);

  /* Store + SINK */
  PolyUOp *store = poly_store_val(ctx, out_buf, r1_flat);
  PolyUOp *sink = poly_sink1(ctx, store);

  /* Execute */
  float a_data[] = {2.0f};
  float w_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
  float out_data[] = {0.0f};

  PolyTestBufferView bindings[] = {
      POLY_TEST_HOST_VIEW(a_buf, a_data),
      POLY_TEST_HOST_VIEW(w_buf, w_data),
      POLY_TEST_HOST_VIEW(out_buf, out_data),
  };

  int ret = poly_test_realize_buffer_views(ctx, sink, bindings, 3);
  poly_ctx_destroy(ctx);

  ASSERT_INT_EQ(ret, 0);
  /* Expected: 2*(1+2+3+4) = 20 */
  ASSERT_FLOAT_EQ(out_data[0], 20.0f, 1e-4);
  PASS();
}

/* Regression: mixed multi-axis reduction with singleton */
/* Reduce over axes [0,2] where axis 0 is singleton (size 1) and axis 2
 * is real (size 3).  Ensures:
 *   - singleton axes don't generate loop ranges
 *   - non-singleton axes still generate RANGE(3)
 *   - generated kernel accumulates correctly */
TEST(rangeify, mixed_multiaxis_singleton_reduce_e2e) {
  PolyCtx *ctx = poly_ctx_new();

  /* Input: (1, 2, 3) tensor stored flat in 6 elements */
  PolyUOp *in_buf = poly_buffer_f32(ctx, 6);
  PolyUOp *out_buf = poly_buffer_f32(ctx, 2); /* output: (1, 2, 1) → flat (2) */

  /* Reshape to (1, 2, 3) */
  int64_t sh123[] = {1, 2, 3};
  PolyUOp *inp = poly_reshape(ctx, in_buf, sh123, 3);

  /* Reduce axes 0 and 2: (1, 2, 3) → (1, 2, 1) */
  int64_t axes[] = {0, 2};
  PolyUOp *r = poly_reduce_axis(ctx, POLY_OP_ADD, inp, axes, 2);

  /* Reshape to (2) for store */
  PolyUOp *r_flat = poly_reshape(ctx, r, (int64_t[]){2}, 1);

  /* Store + SINK */
  PolyUOp *store = poly_store_val(ctx, out_buf, r_flat);
  PolyUOp *sink = poly_sink1(ctx, store);

  /* Execute: [[1,2,3],[4,5,6]] → row sums [6, 15] */
  float in_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  float out_data[] = {0.0f, 0.0f};

  PolyTestBufferView bindings[] = {
      POLY_TEST_HOST_VIEW(in_buf, in_data),
      POLY_TEST_HOST_VIEW(out_buf, out_data),
  };

  int ret = poly_test_realize_buffer_views(ctx, sink, bindings, 2);
  poly_ctx_destroy(ctx);

  ASSERT_INT_EQ(ret, 0);
  ASSERT_FLOAT_EQ(out_data[0], 6.0f, 1e-4); /* 1+2+3 */
  ASSERT_FLOAT_EQ(out_data[1], 15.0f, 1e-4); /* 4+5+6 */
  PASS();
}

/* C4 ASSIGN normalization in earliest_rewrites */

TEST(rangeify, earliest_assign_bitcast) {
  /* C4c: ASSIGN(BITCAST(buf, f16), value) → ASSIGN(buf, BITCAST(value, buf.dtype))
   * Test via e2e: build ASSIGN with BITCAST target, verify correct execution. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *buf = poly_buffer(ctx, POLY_FLOAT32, 4);
  PolyUOp *src = poly_buffer(ctx, POLY_FLOAT32, 4);

  /* BITCAST target to same type (simplest test — BITCAST f32→f32 is identity) */
  PolyUOp *bc_target = poly_uop1(ctx, POLY_OP_BITCAST, POLY_FLOAT32, buf, poly_arg_none());
  /* value = src * 2 */
  PolyUOp *two = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(2.0));
  int64_t sh4[] = {4};
  PolyUOp *two_exp = poly_expand(ctx, poly_reshape(ctx, two, sh4, 1), sh4, 1);
  PolyUOp *value = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, src, two_exp, poly_arg_none());

  /* ASSIGN(BITCAST(buf), value) */
  PolyUOp *assign_src[2] = {bc_target, value};
  PolyUOp *assign = poly_uop(ctx, POLY_OP_ASSIGN, POLY_FLOAT32, assign_src, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, assign, poly_arg_none());

  /* After earliest_rewrites, C4c should move BITCAST from target to source.
   * Execute: buf should become src * 2 */
  float buf_d[] = {1, 2, 3, 4};
  float src_d[] = {10, 20, 30, 40};
  PolyTestBufferView bindings[] = {
      POLY_TEST_HOST_VIEW(buf, buf_d),
      POLY_TEST_HOST_VIEW(src, src_d),
  };
  int ret = poly_test_realize_buffer_views(ctx, sink, bindings, 2);
  poly_ctx_destroy(ctx);
  ASSERT_INT_EQ(ret, 0);
  for (int i = 0; i < 4; i++)
    ASSERT_FLOAT_EQ(buf_d[i], src_d[i] * 2.0f, 1e-5);
  PASS();
}

TEST(rangeify, earliest_nested_assign_chain) {
  /* C4d: ASSIGN(ASSIGN(buf, v1), v2) → ASSIGN(buf, v2)
   * Chain of ASSIGNs collapses to root buffer target. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *buf = poly_buffer(ctx, POLY_FLOAT32, 4);
  PolyUOp *v1 = poly_buffer(ctx, POLY_FLOAT32, 4);
  PolyUOp *v2 = poly_buffer(ctx, POLY_FLOAT32, 4);

  /* Inner: ASSIGN(buf, v1) */
  PolyUOp *inner_src[2] = {buf, v1};
  PolyUOp *inner = poly_uop(ctx, POLY_OP_ASSIGN, POLY_FLOAT32, inner_src, 2, poly_arg_none());

  /* Outer: ASSIGN(inner_assign, v2) — target is ASSIGN, not PARAM */
  PolyUOp *outer_src[2] = {inner, v2};
  PolyUOp *outer = poly_uop(ctx, POLY_OP_ASSIGN, POLY_FLOAT32, outer_src, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, outer, poly_arg_none());

  /* Execute: C4d walks chain to buf. Result: buf = v2 data */
  float buf_d[] = {0, 0, 0, 0};
  float v1_d[] = {1, 2, 3, 4};
  float v2_d[] = {10, 20, 30, 40};
  PolyTestBufferView bindings[] = {
      POLY_TEST_HOST_VIEW(buf, buf_d),
      POLY_TEST_HOST_VIEW(v1, v1_d),
      POLY_TEST_HOST_VIEW(v2, v2_d),
  };
  int ret = poly_test_realize_buffer_views(ctx, sink, bindings, 3);
  poly_ctx_destroy(ctx);
  ASSERT_INT_EQ(ret, 0);
  for (int i = 0; i < 4; i++)
    ASSERT_FLOAT_EQ(buf_d[i], v2_d[i], 1e-5);
  PASS();
}

TEST(rangeify, earliest_assign_to_contiguous) {
  /* poly_store_buffer_update() normalizes RESHAPE(buf) target to base BUFFER and
   * reshapes value to flat shape. The STORE writes to buf in-place. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *buf = poly_buffer(ctx, POLY_FLOAT32, 8);
  PolyUOp *src = poly_buffer(ctx, POLY_FLOAT32, 8);

  /* Reshape target: view buf as [4,2] — base is BUFFER, safe */
  int64_t sh42[] = {4, 2};
  PolyUOp *reshaped = poly_reshape(ctx, buf, sh42, 2);

  /* value = src + 1 */
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(1.0));
  int64_t sh8[] = {8};
  PolyUOp *one_exp = poly_expand(ctx, poly_reshape(ctx, one, sh8, 1), sh8, 1);
  PolyUOp *value = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, src, one_exp, poly_arg_none());

  /* Use the whole-buffer helper, which normalizes RESHAPE(buf) -> BUFFER. */
  PolyUOp *assign = poly_store_buffer_update(ctx, reshaped, value);
  /* Verify normalization: target should be base BUFFER, not RESHAPE */
  ASSERT_TRUE(assign->src[0] == buf); /* target normalized to BUFFER */
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, assign, poly_arg_none());

  float buf_d[] = {99, 99, 99, 99, 99, 99, 99, 99};
  float src_d[] = {1, 2, 3, 4, 5, 6, 7, 8};
  PolyTestBufferView bindings[] = {
      POLY_TEST_HOST_VIEW(buf, buf_d),
      POLY_TEST_HOST_VIEW(src, src_d),
  };
  int ret = poly_test_realize_buffer_views(ctx, sink, bindings, 2);
  poly_ctx_destroy(ctx);
  /* poly_store_buffer_update normalizes to BUFFER target; STORE writes in-place */
  ASSERT_INT_EQ(ret, 0);
  for (int i = 0; i < 8; i++)
    ASSERT_FLOAT_EQ(buf_d[i], src_d[i] + 1.0f, 1e-5);
  PASS();
}

/* Item 5: range_start_for_op covers CALL/COPY/BUFFER_VIEW */

TEST(rangeify, range_start_for_op_all) {
  /* Verify range_start_for_op returns correct values for all 8 ops.
   * Matches tinygrad: {BUFFERIZE:1, REDUCE:1, STORE:2, WMMA:3, END:1,
   *                    CALL:1, COPY:2, BUFFER_VIEW:1} */
  ASSERT_INT_EQ(range_start_for_op(POLY_OP_STAGE), 1);
  ASSERT_INT_EQ(range_start_for_op(POLY_OP_REDUCE), 1);
  ASSERT_INT_EQ(range_start_for_op(POLY_OP_STORE), 2);
  ASSERT_INT_EQ(range_start_for_op(POLY_OP_WMMA), 3);
  ASSERT_INT_EQ(range_start_for_op(POLY_OP_END), 1);
  ASSERT_INT_EQ(range_start_for_op(POLY_OP_CALL), 1);
  ASSERT_INT_EQ(range_start_for_op(POLY_OP_COPY), 2);
  ASSERT_INT_EQ(range_start_for_op(POLY_OP_BUFFER_VIEW), 1);
  /* Unknown ops should return -1 */
  ASSERT_INT_EQ(range_start_for_op(POLY_OP_ADD), -1);
  ASSERT_INT_EQ(range_start_for_op(POLY_OP_CONST), -1);
  PASS();
}

/* Item 7: fix_assign_hazard SHRINK handling */

TEST(rangeify, assign_shrink_hazard) {
  /* a[:5].assign(a[3:8]) — overlapping SHRINK regions create write-before-read
   * aliasing. fix_assign_hazard should force materialization of the source
   * before writing. Build legacy ASSIGN manually (not via poly_store_buffer_update which
   * normalizes target) to preserve SHRINK on target.
   *
   * Verify: CONTIGUOUS insertion causes 2+ kernels (materialization + ASSIGN).
   * Without the SHRINK hazard check, source would fuse into one kernel and
   * reads would see partially-written data. */
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *buf_a = poly_buffer(ctx, POLY_FLOAT32, 8);

  /* a[3:8] — source SHRINK */
  int64_t src_pairs[][2] = {{3, 8}};
  PolyUOp *a_3_8 = poly_shrink(ctx, buf_a, src_pairs, 1);

  /* a[:5] — target SHRINK */
  int64_t tgt_pairs[][2] = {{0, 5}};
  PolyUOp *a_0_5 = poly_shrink(ctx, buf_a, tgt_pairs, 1);

  /* ASSIGN(a[:5], a[3:8]) — construct directly, preserving SHRINK on target */
  PolyUOp *assign_srcs[2] = {a_0_5, a_3_8};
  PolyUOp *assign = poly_uop(ctx, POLY_OP_ASSIGN, POLY_FLOAT32, assign_srcs, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, assign, poly_arg_none());

  PolyKernelScheduleResult sr = poly_build_kernel_schedule(ctx, sink);

  /* The CONTIGUOUS insertion should produce 2+ kernels:
   * one for materializing the source, one for the ASSIGN write. */
  ASSERT_TRUE(sr.n_kernels >= 2);

  poly_kernel_schedule_result_free(&sr);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, assign_shrink_no_hazard_different_buf) {
  /* a[:3].assign(b[:3]) — SHRINK on target but source is different buffer.
   * Source SHRINK does NOT reach target_base in backward slice.
   * No hazard, so CONTIGUOUS should not be inserted for hazard reasons.
   * (CONTIGUOUS may appear for other scheduling reasons like C4e.) */
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *buf_a = poly_buffer(ctx, POLY_FLOAT32, 8);
  PolyUOp *buf_b = poly_buffer(ctx, POLY_FLOAT32, 8);

  int64_t src_pairs[][2] = {{0, 3}};
  PolyUOp *b_0_3 = poly_shrink(ctx, buf_b, src_pairs, 1);

  int64_t tgt_pairs[][2] = {{0, 3}};
  PolyUOp *a_0_3 = poly_shrink(ctx, buf_a, tgt_pairs, 1);

  /* ASSIGN(a[:3], b[:3]) — construct directly */
  PolyUOp *assign_srcs[2] = {a_0_3, b_0_3};
  PolyUOp *assign = poly_uop(ctx, POLY_OP_ASSIGN, POLY_FLOAT32, assign_srcs, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, assign, poly_arg_none());

  /* This should schedule successfully */
  PolyKernelScheduleResult sr = poly_build_kernel_schedule(ctx, sink);
  ASSERT_TRUE(sr.n_kernels >= 1);

  poly_kernel_schedule_result_free(&sr);
  poly_ctx_destroy(ctx);
  PASS();
}

/* Item 6: realize_assign_src optimization */

TEST(rangeify, realize_assign_src_copy_unrealized) {
  /* ASSIGN(buf, COPY(src)) — COPY is the direct source of ASSIGN with
   * no movement ops on target. COPY should be un-realized (eliminated
   * as an intermediate). */
  PolyCtx *ctx = poly_ctx_new();
  PolyIndexingCtx *ictx = poly_indexing_ctx_new(ctx);

  PolyUOp *buf = poly_buffer(ctx, POLY_FLOAT32, 4);
  PolyUOp *src = poly_buffer(ctx, POLY_FLOAT32, 4);

  /* COPY(src) — a device-level copy */
  PolyUOp *copy = poly_uop1(ctx, POLY_OP_COPY, POLY_FLOAT32, src, poly_arg_none());

  /* ASSIGN(buf, COPY(src)) */
  PolyUOp *assign_src[2] = {buf, copy};
  PolyUOp *assign = poly_uop(ctx, POLY_OP_ASSIGN, POLY_FLOAT32, assign_src, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, assign, poly_arg_none());

  poly_realize_map_build(ictx, sink);

  /* COPY should NOT be realized (un-realized by realize_assign_src) */
  ASSERT_FALSE(poly_is_realized(ictx, copy));
  /* ASSIGN itself should still be realized */
  ASSERT_TRUE(poly_is_realized(ictx, assign));

  poly_indexing_ctx_destroy(ictx);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, realize_assign_src_war_forces_realize) {
  /* ASSIGN(buf, buf + 1) — target base appears in RHS backward slice.
   * WAR hazard: RHS should be force-realized. */
  PolyCtx *ctx = poly_ctx_new();
  PolyIndexingCtx *ictx = poly_indexing_ctx_new(ctx);

  PolyUOp *buf = poly_buffer(ctx, POLY_FLOAT32, 4);
  PolyUOp *one = poly_const_float(ctx, 1.0);
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, buf, one, poly_arg_none());

  /* ASSIGN(buf, buf + 1) */
  PolyUOp *assign_src[2] = {buf, add};
  PolyUOp *assign = poly_uop(ctx, POLY_OP_ASSIGN, POLY_FLOAT32, assign_src, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, assign, poly_arg_none());

  poly_realize_map_build(ictx, sink);

  /* RHS (add) should be realized due to WAR hazard */
  ASSERT_TRUE(poly_is_realized(ictx, add));

  poly_indexing_ctx_destroy(ictx);
  poly_ctx_destroy(ctx);
  PASS();
}
