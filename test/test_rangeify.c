/*
 * test_rangeify.c — Tests for the rangeify scheduling pipeline
 */

#define _POSIX_C_SOURCE 200809L
#include "test_harness.h"
#include "../src/rangeify.h"
#include "../src/sched.h"
#include "../src/indexing.h"
#include "../src/codegen.h"
#include "../src/frontend.h"

/* ── Cleanup helper ──────────────────────────────────────────────────── */

static void free_consumer_list_cb(const void *key, void *value, void *ud) {
  (void)key; (void)ud;
  PolyConsumerList *cl = value;
  free(cl->items);
  free(cl);
}

static void destroy_consumer_map(PolyMap *cmap) {
  poly_map_foreach(cmap, free_consumer_list_cb, NULL);
  poly_map_destroy(cmap);
}

/* ── Consumer map tests ──────────────────────────────────────────────── */

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

  PolyUOp *stores[] = { s1, s2 };
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

/* ── Realize map tests ───────────────────────────────────────────────── */

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
  /* sum(a) → REDUCE_AXIS(a): REDUCE_AXIS is NOT automatically realized.
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

  /* REDUCE_AXIS is NOT realized by default */
  ASSERT_FALSE(poly_is_realized(ictx, reduce));

  poly_indexing_ctx_destroy(ictx);
  poly_ctx_destroy(ctx);
  PASS();
}

/* ── Range propagation tests ─────────────────────────────────────────── */

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

  /* REDUCE_AXIS: inherits STORE's scalar range as output,
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

  /* REDUCE_AXIS gets inner RANGE for reduced axis (different from outer) */
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

  PolyUOp *stores[] = { s1, s2 };
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

/* ── apply_rangeify helper ───────────────────────────────────────────── */

/* Count how many UOps of a given op type appear in the graph */
static int count_ops(PolyCtx *ctx, PolyUOp *root, PolyOps op) {
  int n;
  PolyUOp **topo = poly_toposort(ctx, root, &n);
  int count = 0;
  for (int i = 0; i < n; i++)
    if (topo[i]->op == op) count++;
  return count;
}

/* Run full rangeify pipeline up to and including apply_rangeify */
static PolyUOp *run_apply_rangeify(PolyIndexingCtx *ictx, PolyUOp *sink) {
  poly_realize_map_build(ictx, sink);
  poly_range_propagate(ictx, sink);
  return poly_apply_rangeify(ictx, sink);
}

/* ── Apply rangeify tests ────────────────────────────────────────────── */

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
   * After apply: REDUCE_AXIS becomes REDUCE with range sources */
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, 1);
  int64_t axes[] = {0};
  PolyUOp *reduce = poly_reduce_axis(ctx, POLY_OP_ADD, a, axes, 1);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out, reduce, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  /* Before: REDUCE_AXIS exists, REDUCE does not */
  ASSERT_INT_EQ(count_ops(ctx, sink, POLY_OP_REDUCE_AXIS), 1);
  ASSERT_INT_EQ(count_ops(ctx, sink, POLY_OP_REDUCE), 0);

  PolyIndexingCtx *ictx = poly_indexing_ctx_new(ctx);
  PolyUOp *result = run_apply_rangeify(ictx, sink);
  ASSERT_NOT_NULL(result);

  /* After: REDUCE_AXIS gone, REDUCE exists */
  ASSERT_INT_EQ(count_ops(ctx, result, POLY_OP_REDUCE_AXIS), 0);
  ASSERT_INT_EQ(count_ops(ctx, result, POLY_OP_REDUCE), 1);

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
  ASSERT_INT_EQ(count_ops(ctx, result, POLY_OP_BUFFERIZE), 0);

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

  PolyUOp *stores[] = { s1, s2 };
  PolyUOp *sink = poly_uop(ctx, POLY_OP_SINK, POLY_VOID, stores, 2, poly_arg_none());

  PolyIndexingCtx *ictx = poly_indexing_ctx_new(ctx);
  PolyUOp *result = run_apply_rangeify(ictx, sink);
  ASSERT_NOT_NULL(result);

  /* a is realized (different consumer ranges) */
  ASSERT_TRUE(poly_is_realized(ictx, a));

  /* But a is a BUFFER → no BUFFERIZE needed (already in memory) */
  ASSERT_INT_EQ(count_ops(ctx, result, POLY_OP_BUFFERIZE), 0);

  /* Both stores still present */
  ASSERT_INT_EQ(count_ops(ctx, result, POLY_OP_STORE), 2);

  poly_indexing_ctx_destroy(ictx);
  poly_ctx_destroy(ctx);
  PASS();
}

/* ── schedule_v2 helper ─────────────────────────────────────────────── */

static int count_lin_ops(PolyUOp **lin, int n, PolyOps op) {
  int c = 0;
  for (int i = 0; i < n; i++)
    if (lin[i]->op == op) c++;
  return c;
}

/* ── schedule_v2 IR tests ───────────────────────────────────────────── */

TEST(rangeify, schedule_v2_vecadd_ir) {
  /* c = a + b (1D, 10 elements): verify kernel IR structure matches v1 */
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *b = poly_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *c = poly_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, a, b, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, c, add, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyScheduleResult sr = poly_schedule_v2(ctx, sink);
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
  poly_schedule_result_free(&sr);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, schedule_v2_chain_ir) {
  /* d = (a + b) * c: single kernel, 4 PARAMs */
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, 8);
  PolyUOp *b = poly_buffer(ctx, POLY_FLOAT32, 8);
  PolyUOp *c = poly_buffer(ctx, POLY_FLOAT32, 8);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, 8);
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, a, b, poly_arg_none());
  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, add, c, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out, mul, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyScheduleResult sr = poly_schedule_v2(ctx, sink);
  ASSERT_INT_EQ(sr.n_kernels, 1);

  int n;
  PolyUOp **lin = poly_linearize(ctx, sr.kernels[0], &n);
  ASSERT_TRUE(n > 0);

  ASSERT_INT_EQ(count_lin_ops(lin, n, POLY_OP_PARAM), 4);
  ASSERT_INT_EQ(count_lin_ops(lin, n, POLY_OP_RANGE), 1);
  ASSERT_INT_EQ(count_lin_ops(lin, n, POLY_OP_LOAD), 3);
  ASSERT_INT_EQ(count_lin_ops(lin, n, POLY_OP_ADD), 1);
  ASSERT_INT_EQ(count_lin_ops(lin, n, POLY_OP_MUL), 1);
  ASSERT_INT_EQ(count_lin_ops(lin, n, POLY_OP_STORE), 1);
  ASSERT_INT_EQ(count_lin_ops(lin, n, POLY_OP_END), 1);

  free(lin);
  poly_schedule_result_free(&sr);
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

  PolyScheduleResult sr = poly_schedule_v2(ctx, sink);
  ASSERT_INT_EQ(sr.n_kernels, 1);

  int n;
  PolyUOp **lin = poly_linearize(ctx, sr.kernels[0], &n);
  ASSERT_TRUE(n > 0);

  /* 2 PARAMs, 1 DEFINE_REG (accumulator), 1 RANGE (inner), 1+ ADD */
  ASSERT_INT_EQ(count_lin_ops(lin, n, POLY_OP_PARAM), 2);
  ASSERT_INT_EQ(count_lin_ops(lin, n, POLY_OP_DEFINE_REG), 1);
  ASSERT_INT_EQ(count_lin_ops(lin, n, POLY_OP_RANGE), 1);
  ASSERT_TRUE(count_lin_ops(lin, n, POLY_OP_ADD) >= 1);

  free(lin);
  poly_schedule_result_free(&sr);
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

  PolyScheduleResult sr = poly_schedule_v2(ctx, sink);
  ASSERT_INT_EQ(sr.n_kernels, 1);

  int n_lin;
  PolyUOp **lin = poly_linearize(ctx, sr.kernels[0], &n_lin);
  char *src = poly_render_c(lin, n_lin, "v2_vecadd");
  ASSERT_NOT_NULL(src);

  PolyProgram *prog = poly_compile_c(src, "v2_vecadd");
  ASSERT_NOT_NULL(prog);

  float a_d[16], b_d[16], c_d[16];
  for (int i = 0; i < N; i++) { a_d[i] = (float)i; b_d[i] = (float)(i * 10); }

  void *args[3] = { c_d, a_d, b_d };
  poly_program_call(prog, args, 3);

  for (int i = 0; i < N; i++)
    ASSERT_FLOAT_EQ(c_d[i], a_d[i] + b_d[i], 1e-5);

  poly_program_destroy(prog);
  free(src);
  free(lin);
  poly_schedule_result_free(&sr);
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

  PolyScheduleResult sr = poly_schedule_v2(ctx, sink);
  ASSERT_INT_EQ(sr.n_kernels, 1);

  int n;
  PolyUOp **lin = poly_linearize(ctx, sr.kernels[0], &n);
  ASSERT_TRUE(n > 0);

  /* Should have: 3 PARAMs, 1 DEFINE_REG (accumulator), 1 RANGE (reduce), 2+ ADD */
  ASSERT_INT_EQ(count_lin_ops(lin, n, POLY_OP_PARAM), 3);
  ASSERT_INT_EQ(count_lin_ops(lin, n, POLY_OP_DEFINE_REG), 1);
  ASSERT_TRUE(count_lin_ops(lin, n, POLY_OP_RANGE) >= 1);
  ASSERT_TRUE(count_lin_ops(lin, n, POLY_OP_ADD) >= 2);  /* 1 reduce + 1 ewise */

  free(lin);
  poly_schedule_result_free(&sr);
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

  PolyScheduleResult sr = poly_schedule_v2(ctx, sink);
  ASSERT_INT_EQ(sr.n_kernels, 1);

  int n_lin;
  PolyUOp **lin = poly_linearize(ctx, sr.kernels[0], &n_lin);
  char *src = poly_render_c(lin, n_lin, "v2_sum_add");
  ASSERT_NOT_NULL(src);

  PolyProgram *prog = poly_compile_c(src, "v2_sum_add");
  ASSERT_NOT_NULL(prog);

  float a_d[10], b_d[1] = {5.0f}, out_d[1] = {0.0f};
  float expected = 5.0f;
  for (int i = 0; i < N; i++) { a_d[i] = (float)(i + 1); expected += a_d[i]; }

  void *args[3] = { out_d, a_d, b_d };
  poly_program_call(prog, args, 3);
  ASSERT_FLOAT_EQ(out_d[0], expected, 1e-4);

  poly_program_destroy(prog);
  free(src);
  free(lin);
  poly_schedule_result_free(&sr);
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

  PolyScheduleResult sr = poly_schedule_v2(ctx, sink);
  ASSERT_INT_EQ(sr.n_kernels, 1);

  int n_lin;
  PolyUOp **lin = poly_linearize(ctx, sr.kernels[0], &n_lin);
  char *src = poly_render_c(lin, n_lin, "v2_sum");
  ASSERT_NOT_NULL(src);

  PolyProgram *prog = poly_compile_c(src, "v2_sum");
  ASSERT_NOT_NULL(prog);

  float a_d[10], c_d[1] = {0.0f};
  float expected = 0.0f;
  for (int i = 0; i < N; i++) { a_d[i] = (float)(i + 1); expected += a_d[i]; }

  void *args[2] = { c_d, a_d };
  poly_program_call(prog, args, 2);
  ASSERT_FLOAT_EQ(c_d[0], expected, 1e-4);

  poly_program_destroy(prog);
  free(src);
  free(lin);
  poly_schedule_result_free(&sr);
  poly_ctx_destroy(ctx);
  PASS();
}

/* ── Parity helper: compare v1 and v2 numerical output ────────────── */

static bool run_via_v1(PolyCtx *ctx, PolyUOp *sink,
                        void **args, int n_args, const char *name) {
  PolyUOp *kernel = poly_schedule(ctx, sink);
  if (!kernel) return false;
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

static bool run_via_v2(PolyCtx *ctx, PolyUOp *sink,
                        void **args, int n_args, const char *name) {
  PolyScheduleResult sr = poly_schedule_v2(ctx, sink);
  if (sr.n_kernels != 1 || !sr.kernels[0]) {
    poly_schedule_result_free(&sr);
    return false;
  }
  int n_lin;
  PolyUOp **lin = poly_linearize(ctx, sr.kernels[0], &n_lin);
  poly_schedule_result_free(&sr);
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
  for (int i = 0; i < N; i++) { a[i] = (float)i * 0.5f; b[i] = (float)(N - i); }

  {
    PolyCtx *ctx = poly_ctx_new();
    PolyUOp *ua = poly_buffer(ctx, POLY_FLOAT32, N);
    PolyUOp *ub = poly_buffer(ctx, POLY_FLOAT32, N);
    PolyUOp *uc = poly_buffer(ctx, POLY_FLOAT32, N);
    PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, ua, ub, poly_arg_none());
    PolyUOp *st = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, uc, add, poly_arg_none());
    PolyUOp *sk = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, st, poly_arg_none());
    void *args[3] = { c_v1, a, b };
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
    void *args[3] = { c_v2, a, b };
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
  for (int i = 0; i < N; i++) a[i] = (float)(i + 1) * 1.5f;

  {
    PolyCtx *ctx = poly_ctx_new();
    PolyUOp *ua = poly_buffer(ctx, POLY_FLOAT32, N);
    PolyUOp *uc = poly_buffer(ctx, POLY_FLOAT32, 1);
    int64_t axes[] = {0};
    PolyUOp *sum = poly_reduce_axis(ctx, POLY_OP_ADD, ua, axes, 1);
    PolyUOp *st = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, uc, sum, poly_arg_none());
    PolyUOp *sk = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, st, poly_arg_none());
    void *args[2] = { c_v1, a };
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
    void *args[2] = { c_v2, a };
    ASSERT_TRUE(run_via_v2(ctx, sk, args, 2, "par_v2_sum"));
    poly_ctx_destroy(ctx);
  }

  ASSERT_FLOAT_EQ(c_v1[0], c_v2[0], 1e-4);
  PASS();
}

/* ── Multi-kernel tests ─────────────────────────────────────────────── */

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
  PolyUOp *stores[] = { s1, s2 };
  PolyUOp *sink = poly_uop(ctx, POLY_OP_SINK, POLY_VOID, stores, 2, poly_arg_none());

  PolyScheduleResult sr = poly_schedule_v2(ctx, sink);
  ASSERT_INT_EQ(sr.n_kernels, 2);   /* 2 fused kernels (d inlined) */
  ASSERT_INT_EQ(sr.n_intermediates, 0);

  /* All kernels should have SINK roots */
  for (int k = 0; k < sr.n_kernels; k++)
    ASSERT_EQ(sr.kernels[k]->op, POLY_OP_SINK);

  poly_schedule_result_free(&sr);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, multi_kernel_shared_computed_e2e) {
  /* Same pattern as IR test, full compile + execute via poly_realize().
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
  PolyUOp *stores[] = { s1, s2 };
  PolyUOp *sink = poly_uop(ctx, POLY_OP_SINK, POLY_VOID, stores, 2, poly_arg_none());

  float a_d[8], b_d[8], c_d[8], o1_d[8], o2_d[8];
  for (int i = 0; i < N; i++) {
    a_d[i] = (float)(i + 1);       /* 1..8 */
    b_d[i] = (float)(i * 10);      /* 0,10,20,...,70 */
    c_d[i] = (float)(i + 1) * 0.5f; /* 0.5,1.0,...,4.0 */
  }

  PolyBufferBinding bindings[] = {
    { out1, o1_d }, { out2, o2_d },
    { a, a_d }, { b, b_d }, { c, c_d },
  };
  int ret = poly_realize(ctx, sink, bindings, 5);
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
  /* Verify that poly_realize() (now using v2) produces correct results
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
  for (int i = 0; i < N; i++) { a_d[i] = (float)i; b_d[i] = (float)(N - i); }

  PolyBufferBinding bindings[] = {
    { out, o_d }, { a, a_d }, { b, b_d },
  };
  int ret = poly_realize(ctx, sink, bindings, 3);
  ASSERT_INT_EQ(ret, 0);

  for (int i = 0; i < N; i++)
    ASSERT_FLOAT_EQ(o_d[i], a_d[i] + b_d[i], 1e-5);

  poly_ctx_destroy(ctx);
  PASS();
}

/* ── Rangeify stats tests ─────────────────────────────────────────────── */

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
  for (int i = 0; i < N; i++) { a_d[i] = (float)i; b_d[i] = (float)(N - i); }

  PolyBufferBinding bindings[] = {
    { out, o_d }, { a, a_d }, { b, b_d },
  };

  poly_rangeify_stats_reset();
  int ret = poly_realize(ctx, sink, bindings, 3);
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
  for (int i = 0; i < R * C; i++) a_d[i] = (float)(i + 1);

  PolyBufferBinding bindings[] = {
    { out, o_d }, { a_flat, a_d },
  };

  poly_rangeify_stats_reset();
  int ret = poly_realize(ctx, sink, bindings, 2);
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

/* ── BUFFERIZE+INDEX structural tests ─────────────────────────────────── */

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

  PolyBufferBinding bindings[] = { {x, x_d}, {out, gx_d} };
  int ret = poly_realize(ctx, sink, bindings, 2);
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
  PolyUOp *stores[] = { s1, s2 };
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
      if (u->src[j]->op == POLY_OP_BUFFERIZE) {
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
  PolyUOp *stores[] = { s1, s2 };
  PolyUOp *sink = poly_uop(ctx, POLY_OP_SINK, POLY_VOID, stores, 2, poly_arg_none());

  float a_d[9], b_d[9], c_d[9], o1_d[9], o2_d[9];
  for (int r = 0; r < 3; r++) {
    for (int col = 0; col < 3; col++) {
      int i = r * 3 + col;
      a_d[i] = (float)(i + 1);                    /* 1..9 */
      b_d[i] = (float)((col + 1) * 10 + r * 30);  /* asymmetric */
      c_d[i] = (float)(i + 1) * 0.1f;             /* 0.1..0.9 */
    }
  }

  PolyBufferBinding bindings[] = {
    { out1_flat, o1_d }, { out2_flat, o2_d },
    { a_flat, a_d }, { b_flat, b_d }, { c_flat, c_d },
  };
  poly_rangeify_stats_reset();
  int ret = poly_realize(ctx, sink, bindings, 5);
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
  for (int i = 0; i < 75; i++) x_d[i] = (float)(i + 1);

  PolyBufferBinding bindings[] = {
    { out, o_d }, { x_flat, x_d },
  };

  poly_rangeify_stats_reset();
  int ret = poly_realize(ctx, sink, bindings, 2);
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

/* ── add_buffers tests ─────────────────────────────────────────────── */

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
  ASSERT_INT_EQ(count_ops(ctx, rangeified, POLY_OP_BUFFERIZE), 0);

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
  PolyUOp *stores[] = { s1, s2 };
  PolyUOp *sink = poly_uop(ctx, POLY_OP_SINK, POLY_VOID, stores, 2, poly_arg_none());

  PolyIndexingCtx *ictx = poly_indexing_ctx_new(ctx);
  PolyUOp *rangeified = run_apply_rangeify(ictx, sink);
  ASSERT_NOT_NULL(rangeified);

  /* Should have exactly 1 BUFFERIZE node (for d = neg(a)) */
  ASSERT_INT_EQ(count_ops(ctx, rangeified, POLY_OP_BUFFERIZE), 1);

  /* Apply add_buffers */
  PolyUOp *result = poly_apply_add_buffers(ctx, rangeified, NULL);
  ASSERT_NOT_NULL(result);

  /* No BUFFERIZE remains */
  ASSERT_INT_EQ(count_ops(ctx, result, POLY_OP_BUFFERIZE), 0);

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

/* ── Structural split parity tests ───────────────────────────────────── */

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
    if (topo[i]->op == POLY_OP_RANGE && n_ranges < 64)
      all_ranges[n_ranges++] = topo[i];
    if (topo[i]->op == POLY_OP_END && topo[i]->n_src >= 2 &&
        topo[i]->src[1]->op == POLY_OP_RANGE && n_closed < 64)
      closed_ranges[n_closed++] = topo[i]->src[1];
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
      if (all_ranges[i] == closed_ranges[j]) { found = true; break; }
    }
    if (found) continue;
    /* Reduce ranges are expected orphans (pm_reduce handles them) */
    bool is_reduce = false;
    for (int j = 0; j < n_reduce; j++) {
      if (all_ranges[i] == reduce_ranges[j]) { is_reduce = true; break; }
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

  PolyUOp *stores[] = { s1, s2 };
  PolyUOp *sink = poly_uop(ctx, POLY_OP_SINK, POLY_VOID, stores, 2, poly_arg_none());

  /* ── IR-level assertions (before realize) ── */

  /* Run rangeify + add_buffers to inspect the intermediate graph */
  PolyIndexingCtx *ictx = poly_indexing_ctx_new(ctx);
  PolyUOp *rangeified = run_apply_rangeify(ictx, sink);
  ASSERT_NOT_NULL(rangeified);

  /* Shared d should be realized → exactly 1 BUFFERIZE */
  ASSERT_INT_EQ(count_ops(ctx, rangeified, POLY_OP_BUFFERIZE), 1);

  /* Apply add_buffers */
  PolyUOp *ab_result = poly_apply_add_buffers(ctx, rangeified, NULL);
  ASSERT_NOT_NULL(ab_result);

  /* Anti-hack assertion 1: No BUFFERIZE remains after add_buffers */
  ASSERT_INT_EQ(count_ops(ctx, ab_result, POLY_OP_BUFFERIZE), 0);

  /* ── Schedule and check kernel structure ── */

  PolyScheduleResult sr = poly_schedule_v2(ctx, sink);

  /* Assertion 2: correct kernel count */
  ASSERT_INT_EQ(sr.n_kernels, 2);       /* 2 fused kernels (d inlined) */
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
  ASSERT_INT_EQ(k0_ranges, k0_ends);

  /* K1 (reduce): reduce_sum(neg(perm(reshape(a))) * c) → out2 */
  int k1_stores = kernel_op_count(ctx, sr.kernels[1], POLY_OP_STORE);
  int k1_ranges = kernel_op_count(ctx, sr.kernels[1], POLY_OP_RANGE);
  int k1_ends = kernel_op_count(ctx, sr.kernels[1], POLY_OP_END);
  int k1_reduces = kernel_op_count(ctx, sr.kernels[1], POLY_OP_REDUCE);
  ASSERT_INT_EQ(k1_stores, 1);
  ASSERT_TRUE(k1_ranges > 0);
  ASSERT_TRUE(k1_ends > 0);
  ASSERT_TRUE(k1_ranges >= k1_ends);  /* ranges >= ends (reduce range has no END) */
  ASSERT_INT_EQ(k1_reduces, 1);

  poly_schedule_result_free(&sr);
  poly_indexing_ctx_destroy(ictx);

  /* ── E2E value correctness ── */

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
      a_d[i] = (float)(i + 1);                    /* 1,2,3,4,5,6,7,8,9 */
      b_d[i] = (float)((col + 1) * 10 + r * 30);  /* asymmetric */
      c_d[i] = (float)(i + 1) * 0.1f;             /* 0.1..0.9 */
    }
  memset(o1_d, 0, sizeof(o1_d));
  memset(o2_d, 0, sizeof(o2_d));

  PolyBufferBinding bindings[] = {
    { out1_flat, o1_d }, { out2_flat, o2_d },
    { a_flat, a_d }, { b_flat, b_d }, { c_flat, c_d },
  };
  int ret = poly_realize(ctx, sink, bindings, 5);
  ASSERT_INT_EQ(ret, 0);

  /* Verify out1 = neg(permute(a)) + b */
  for (int r = 0; r < 3; r++)
    for (int col = 0; col < 3; col++) {
      int i = r * 3 + col;
      float neg_a_perm = -(float)(col * 3 + r + 1);  /* -a[col][r] */
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

TEST(rangeify, shared_scalar_reduce_branches_ir) {
  /* a[8] → REDUCE_AXIS(ADD, axis=0) → RESHAPE([1]) → EXPAND([8]) → sum_exp[8]
   * Branch 1: ADD(sum_exp, c0) → STORE(oc)
   * Branch 2: MUL(sum_exp, e0) → STORE(oe)
   * Expected: 3 kernels (1 producer + 2 consumers), 1 intermediate (scalar, size=1). */
  int N = 8;
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a   = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *c0  = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *e0  = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *oc  = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *oe  = poly_buffer(ctx, POLY_FLOAT32, N);

  int64_t axes[]   = {0};
  int64_t one_sh[] = {1};
  int64_t exp_sh[] = {N};

  PolyUOp *sum     = poly_reduce_axis(ctx, POLY_OP_ADD, a, axes, 1);
  PolyUOp *sum_1   = poly_reshape(ctx, sum, one_sh, 1);
  PolyUOp *sum_exp = poly_expand(ctx, sum_1, exp_sh, 1);

  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, sum_exp, c0, poly_arg_none());
  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, sum_exp, e0, poly_arg_none());
  PolyUOp *sc  = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, oc, add, poly_arg_none());
  PolyUOp *se  = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, oe, mul, poly_arg_none());
  PolyUOp *sink = poly_uop(ctx, POLY_OP_SINK, POLY_VOID,
                            (PolyUOp *[]){sc, se}, 2, poly_arg_none());

  PolyScheduleResult sr = poly_schedule_v2(ctx, sink);
  /* Copy results before freeing so assertions don't leak on failure. */
  int n_kernels      = sr.n_kernels;
  int n_intermediates = sr.n_intermediates;
  int64_t size0      = (sr.n_intermediates > 0) ? sr.intermediate_sizes[0] : -1;
  poly_schedule_result_free(&sr);
  poly_ctx_destroy(ctx);

  ASSERT_INT_EQ(n_kernels, 3);
  ASSERT_INT_EQ(n_intermediates, 1);
  ASSERT_TRUE(size0 == 1); /* scalar reduce output */
  PASS();
}

TEST(rangeify, shared_scalar_reduce_branches_e2e) {
  /* Same graph, full compile+execute via poly_realize().
   * a = [1..8] → sum = 36
   * c0 = [10,10,...] → oc[i] = 36 + 10 = 46
   * e0 = [2,2,...] → oe[i] = 36 * 2 = 72 */
  int N = 8;
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a   = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *c0  = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *e0  = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *oc  = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *oe  = poly_buffer(ctx, POLY_FLOAT32, N);

  int64_t axes[]   = {0};
  int64_t one_sh[] = {1};
  int64_t exp_sh[] = {N};

  PolyUOp *sum     = poly_reduce_axis(ctx, POLY_OP_ADD, a, axes, 1);
  PolyUOp *sum_1   = poly_reshape(ctx, sum, one_sh, 1);
  PolyUOp *sum_exp = poly_expand(ctx, sum_1, exp_sh, 1);

  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, sum_exp, c0, poly_arg_none());
  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, sum_exp, e0, poly_arg_none());
  PolyUOp *sc  = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, oc, add, poly_arg_none());
  PolyUOp *se  = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, oe, mul, poly_arg_none());
  PolyUOp *sink = poly_uop(ctx, POLY_OP_SINK, POLY_VOID,
                            (PolyUOp *[]){sc, se}, 2, poly_arg_none());

  float a_d[8], c0_d[8], e0_d[8], oc_d[8], oe_d[8];
  for (int i = 0; i < N; i++) {
    a_d[i]  = (float)(i + 1);  /* 1..8, sum=36 */
    c0_d[i] = 10.0f;
    e0_d[i] = 2.0f;
    oc_d[i] = 0.0f;
    oe_d[i] = 0.0f;
  }

  PolyBufferBinding bindings[] = {
    { oc, oc_d }, { oe, oe_d },
    { a, a_d }, { c0, c0_d }, { e0, e0_d },
  };
  int ret = poly_realize(ctx, sink, bindings, 5);
  poly_ctx_destroy(ctx);
  ASSERT_INT_EQ(ret, 0);

  for (int i = 0; i < N; i++) {
    ASSERT_FLOAT_EQ(oc_d[i], 46.0f, 1e-5);
    ASSERT_FLOAT_EQ(oe_d[i], 72.0f, 1e-5);
  }

  PASS();
}

/* ── Stage A: CONST-through-BUFFERIZE + noop removal ────────────────── */

TEST(rangeify, const_through_bufferize) {
  /* CONST(42.0) broadcasted via RESHAPE+EXPAND -> ADD with buffer.
   * The constant should fold through BUFFERIZE (no intermediate buffer). */
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a   = poly_buffer(ctx, POLY_FLOAT32, 4);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, 4);
  PolyUOp *c42 = poly_uop(ctx, POLY_OP_CONST, POLY_FLOAT32, NULL, 0,
                            poly_arg_float(42.0));
  int64_t one_sh[] = {1};
  int64_t exp_sh[] = {4};
  PolyUOp *reshaped = poly_reshape(ctx, c42, one_sh, 1);
  PolyUOp *expanded = poly_expand(ctx, reshaped, exp_sh, 1);
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, a, expanded,
                             poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out, add,
                               poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store,
                              poly_arg_none());

  float a_d[] = {1, 2, 3, 4};
  float out_d[4] = {0};
  PolyBufferBinding bindings[] = {
    { out, out_d }, { a, a_d },
  };
  int ret = poly_realize(ctx, sink, bindings, 2);
  poly_ctx_destroy(ctx);
  ASSERT_INT_EQ(ret, 0);

  for (int i = 0; i < 4; i++) {
    ASSERT_FLOAT_EQ(out_d[i], a_d[i] + 42.0f, 1e-5);
  }
  PASS();
}

/* ── Stage C: earliest_rewrites ─────────────────────────────────────── */

TEST(rangeify, earliest_reshape_merge) {
  /* RESHAPE(RESHAPE(x, [4,2]), [8]): verify correct output.
   * poly_earliest_rewrites should merge into RESHAPE(x, [8]). */
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a   = poly_buffer(ctx, POLY_FLOAT32, 8);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, 8);

  int64_t sh42[] = {4, 2};
  int64_t sh8[]  = {8};
  PolyUOp *r1 = poly_reshape(ctx, a, sh42, 2);
  PolyUOp *r2 = poly_reshape(ctx, r1, sh8, 1);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out, r2,
                               poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store,
                              poly_arg_none());

  float a_d[] = {1, 2, 3, 4, 5, 6, 7, 8};
  float out_d[8] = {0};
  PolyBufferBinding bindings[] = {
    { out, out_d }, { a, a_d },
  };
  int ret = poly_realize(ctx, sink, bindings, 2);
  poly_ctx_destroy(ctx);
  ASSERT_INT_EQ(ret, 0);

  for (int i = 0; i < 8; i++) {
    ASSERT_FLOAT_EQ(out_d[i], a_d[i], 1e-5);
  }
  PASS();
}

TEST(rangeify, earliest_detach_removal) {
  /* DETACH(a) + b -> STORE: verify DETACH is stripped and result is correct */
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a   = poly_buffer(ctx, POLY_FLOAT32, 4);
  PolyUOp *b   = poly_buffer(ctx, POLY_FLOAT32, 4);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, 4);
  PolyUOp *detached = poly_uop1(ctx, POLY_OP_DETACH, POLY_FLOAT32, a,
                                  poly_arg_none());
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, detached, b,
                             poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out, add,
                               poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store,
                              poly_arg_none());

  float a_d[] = {1, 2, 3, 4};
  float b_d[] = {10, 20, 30, 40};
  float out_d[4] = {0};
  PolyBufferBinding bindings[] = {
    { out, out_d }, { a, a_d }, { b, b_d },
  };
  int ret = poly_realize(ctx, sink, bindings, 3);
  poly_ctx_destroy(ctx);
  ASSERT_INT_EQ(ret, 0);

  for (int i = 0; i < 4; i++) {
    ASSERT_FLOAT_EQ(out_d[i], a_d[i] + b_d[i], 1e-5);
  }
  PASS();
}

/* ── Stage 3.25: pm_limit_bufs ──────────────────────────────────────── */

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

  PolyScheduleResult sr = poly_schedule_v2(ctx, sink);

  /* Extract results before cleanup */
  int n_kernels = sr.n_kernels;
  int n_intermediates = sr.n_intermediates;
  bool all_under_limit = true;
  for (int k = 0; k < sr.n_kernels; k++) {
    if (sr.kernel_n_params[k] > 8) all_under_limit = false;
  }

  poly_schedule_result_free(&sr);
  poly_ctx_destroy(ctx);

  /* Restore env */
  if (olddup) { setenv("POLY_MAX_KERNEL_BUFFERS", olddup, 1); free(olddup); }
  else unsetenv("POLY_MAX_KERNEL_BUFFERS");

  ASSERT_TRUE(all_under_limit);     /* primary: every kernel within limit */
  ASSERT_TRUE(n_kernels > 1);       /* must split */
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

  PolyBufferBinding bindings[12]; /* out + 10 inputs */
  bindings[0] = (PolyBufferBinding){ out, out_d };
  for (int i = 0; i < N_BUFS; i++)
    bindings[1 + i] = (PolyBufferBinding){ bufs[i], buf_data[i] };

  int ret = poly_realize(ctx, sink, bindings, 1 + N_BUFS);

  /* Copy before cleanup */
  float out_copy[16];
  for (int j = 0; j < N; j++) out_copy[j] = out_d[j];

  poly_ctx_destroy(ctx);

  /* Restore env */
  if (olddup) { setenv("POLY_MAX_KERNEL_BUFFERS", olddup, 1); free(olddup); }
  else unsetenv("POLY_MAX_KERNEL_BUFFERS");

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

  PolyScheduleResult sr = poly_schedule_v2(ctx, sink);
  int n_kernels = sr.n_kernels;
  int n_intermediates = sr.n_intermediates;
  poly_schedule_result_free(&sr);
  poly_ctx_destroy(ctx);

  if (olddup) { setenv("POLY_MAX_KERNEL_BUFFERS", olddup, 1); free(olddup); }
  else unsetenv("POLY_MAX_KERNEL_BUFFERS");

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

  PolyScheduleResult sr = poly_schedule_v2(ctx, sink);
  int n_kernels = sr.n_kernels;
  poly_schedule_result_free(&sr);
  poly_ctx_destroy(ctx);

  /* Restore env */
  if (olddup) { setenv("POLY_MAX_KERNEL_BUFFERS", olddup, 1); free(olddup); }

  ASSERT_INT_EQ(n_kernels, 1);
  PASS();
}


/* ── ASSIGN + WAR ordering tests ───────────────────────────────────────── */

/* assign_e2e: a.assign(a + b) — basic in-place update */
TEST(rangeify, assign_e2e) {
  const int N = 4;
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *buf_a = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *buf_b = poly_buffer(ctx, POLY_FLOAT32, N);

  /* value = a + b */
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, buf_a, buf_b,
                             poly_arg_none());

  /* ASSIGN(a, a + b) */
  PolyUOp *assign = poly_assign(ctx, buf_a, add);

  /* SINK(ASSIGN) — ASSIGN goes directly in SINK */
  PolyUOp *sink = poly_uop(ctx, POLY_OP_SINK, POLY_VOID, &assign, 1,
                             poly_arg_none());

  float a_data[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
  float b_data[4] = { 10.0f, 20.0f, 30.0f, 40.0f };

  PolyBufferBinding bindings[2] = {
    { .buffer = buf_a, .data = a_data },
    { .buffer = buf_b, .data = b_data },
  };

  int ret = poly_realize(ctx, sink, bindings, 2);
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

/* assign_ir: verify ASSIGN scheduling produces correct kernel structure */
TEST(rangeify, assign_ir) {
  const int N = 4;
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *buf_a = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *buf_b = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, buf_a, buf_b,
                             poly_arg_none());
  PolyUOp *assign = poly_assign(ctx, buf_a, add);
  PolyUOp *sink = poly_uop(ctx, POLY_OP_SINK, POLY_VOID, &assign, 1,
                             poly_arg_none());

  PolyScheduleResult sr = poly_schedule_v2(ctx, sink);

  int got_kernels = sr.n_kernels;
  int got_inter = sr.n_intermediates;

  /* Check that the ASSIGN kernel writes to buf_a (existing buffer).
   * The ASSIGN AFTER's buffer should appear in param_to_buf for the kernel. */
  bool writes_buf_a = false;
  for (int k = 0; k < sr.n_kernels; k++) {
    for (int p = 0; p < sr.kernel_n_params[k]; p++) {
      if (sr.param_to_buf[k][p] == buf_a)
        writes_buf_a = true;
    }
  }

  poly_schedule_result_free(&sr);
  poly_ctx_destroy(ctx);

  /* 1 ASSIGN kernel (from AFTER), 0 consumer stores, 0 intermediates */
  ASSERT_INT_EQ(got_kernels, 1);
  ASSERT_INT_EQ(got_inter, 0);
  ASSERT_TRUE(writes_buf_a);
  PASS();
}

/* assign_war_ordering: reader kernel must complete before ASSIGN writer.
 * Graph: a[4], out[4] = a + 10 (separate STORE kernel reading a),
 *        ASSIGN(a, a * 2) (writes a).
 * WAR: out's kernel reads a, ASSIGN writes a → reader before writer.
 * Verify: out = {11, 12, 13, 14} (pre-ASSIGN values), a = {2, 4, 6, 8}. */
TEST(rangeify, assign_war_ordering) {
  const int N = 4;
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *buf_a = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *buf_out = poly_buffer(ctx, POLY_FLOAT32, N);

  /* STORE: out = a + 10 */
  PolyUOp *ten = poly_const_float(ctx, 10.0);
  PolyUOp *a_plus_10 = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, buf_a, ten,
                                   poly_arg_none());
  PolyUOp *store_out = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, buf_out,
                                   a_plus_10, poly_arg_none());

  /* ASSIGN: a = a * 2 */
  PolyUOp *two = poly_const_float(ctx, 2.0);
  PolyUOp *a_times_2 = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, buf_a, two,
                                   poly_arg_none());
  PolyUOp *assign = poly_assign(ctx, buf_a, a_times_2);

  /* SINK(STORE(out, a+10), ASSIGN(a, a*2)) */
  PolyUOp *sink_src[2] = { store_out, assign };
  PolyUOp *sink = poly_uop(ctx, POLY_OP_SINK, POLY_VOID, sink_src, 2,
                             poly_arg_none());

  float a_data[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
  float out_data[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

  PolyBufferBinding bindings[2] = {
    { .buffer = buf_a, .data = a_data },
    { .buffer = buf_out, .data = out_data },
  };

  int ret = poly_realize(ctx, sink, bindings, 2);
  float a_result[4], out_result[4];
  memcpy(a_result, a_data, sizeof(a_result));
  memcpy(out_result, out_data, sizeof(out_result));

  poly_ctx_destroy(ctx);

  ASSERT_INT_EQ(ret, 0);
  /* out should see pre-ASSIGN values of a (WAR ordering) */
  ASSERT_FLOAT_EQ(out_result[0], 11.0f, 1e-5);
  ASSERT_FLOAT_EQ(out_result[1], 12.0f, 1e-5);
  ASSERT_FLOAT_EQ(out_result[2], 13.0f, 1e-5);
  ASSERT_FLOAT_EQ(out_result[3], 14.0f, 1e-5);
  /* a should be updated by ASSIGN */
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
  PolyUOp *a_times_2 = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, buf_a, two,
                                   poly_arg_none());
  PolyUOp *assign = poly_assign(ctx, buf_a, a_times_2);
  PolyUOp *sink = poly_uop(ctx, POLY_OP_SINK, POLY_VOID, &assign, 1,
                             poly_arg_none());

  float a_data[4] = { 3.0f, 5.0f, 7.0f, 11.0f };

  PolyBufferBinding bindings[1] = {
    { .buffer = buf_a, .data = a_data },
  };

  int ret = poly_realize(ctx, sink, bindings, 1);
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

/* ── Dynamic shapes (DEFINE_VAR) tests ─────────────────────────────────── */

/* define_var_1d_e2e: a[N] + 1.0 -> out[N] with N=4 */
TEST(rangeify, define_var_1d_e2e) {
  PolyCtx *ctx = poly_ctx_new();

  /* Create symbolic variable N with bounds [1, 16] */
  PolyUOp *N = poly_define_var(ctx, "N", 1, 16);

  /* Create dynamic 1D buffers: a[N], out[N] */
  PolyUOp *buf_a   = poly_buffer_var(ctx, POLY_FLOAT32, N, NULL, 0);
  PolyUOp *buf_out  = poly_buffer_var(ctx, POLY_FLOAT32, N, NULL, 0);

  /* out = a + 1.0 */
  PolyUOp *one = poly_const_float(ctx, 1.0);
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, buf_a, one, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, buf_out, add, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  /* Execute with N=4 */
  float a_data[16] = { 10.0f, 20.0f, 30.0f, 40.0f };
  float out_data[16];
  memset(out_data, 0, sizeof(out_data));

  PolyBufferBinding bindings[2] = {
    { .buffer = buf_a,   .data = a_data },
    { .buffer = buf_out,  .data = out_data },
  };
  PolyVarBinding var_bindings[1] = {
    { .var = N, .value = 4 },
  };

  int ret = poly_realize_ex(ctx, sink, bindings, 2, var_bindings, 1);
  float result[16];
  memcpy(result, out_data, sizeof(result));

  poly_ctx_destroy(ctx);

  ASSERT_INT_EQ(ret, 0);
  /* First 4 elements should be a + 1.0 */
  ASSERT_FLOAT_EQ(result[0], 11.0f, 1e-5);
  ASSERT_FLOAT_EQ(result[1], 21.0f, 1e-5);
  ASSERT_FLOAT_EQ(result[2], 31.0f, 1e-5);
  ASSERT_FLOAT_EQ(result[3], 41.0f, 1e-5);
  /* Elements beyond N=4 should be untouched (still 0.0) */
  ASSERT_FLOAT_EQ(result[4], 0.0f, 1e-5);
  PASS();
}

/* define_var_2d_e2e: a[N,4] + 1.0 -> out[N,4] with N=3, then N=2 */
TEST(rangeify, define_var_2d_e2e) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *N = poly_define_var(ctx, "N", 1, 8);

  /* Create 2D dynamic buffers: a[N,4], out[N,4] */
  int64_t inner_dim = 4;
  PolyUOp *buf_a   = poly_buffer_var(ctx, POLY_FLOAT32, N, &inner_dim, 1);
  PolyUOp *buf_out  = poly_buffer_var(ctx, POLY_FLOAT32, N, &inner_dim, 1);

  /* out = a + 1.0 */
  PolyUOp *one = poly_const_float(ctx, 1.0);
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, buf_a, one, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, buf_out, add, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  /* max alloc = 8*4 = 32 elements */
  float a_data[32];
  float out_data[32];
  for (int i = 0; i < 32; i++) a_data[i] = (float)(i + 1);
  memset(out_data, 0, sizeof(out_data));

  PolyBufferBinding bindings[2] = {
    { .buffer = buf_a,   .data = a_data },
    { .buffer = buf_out,  .data = out_data },
  };

  /* Execute with N=3 (12 elements) */
  PolyVarBinding var_bindings[1] = {
    { .var = N, .value = 3 },
  };

  int ret = poly_realize_ex(ctx, sink, bindings, 2, var_bindings, 1);
  float result[32];
  memcpy(result, out_data, sizeof(result));

  poly_ctx_destroy(ctx);

  ASSERT_INT_EQ(ret, 0);
  /* First 12 elements (3 rows of 4) should be a + 1.0 */
  for (int i = 0; i < 12; i++) {
    ASSERT_FLOAT_EQ(result[i], (float)(i + 2), 1e-5);
  }
  /* Element 12 should be untouched */
  ASSERT_FLOAT_EQ(result[12], 0.0f, 1e-5);
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
  PolyUOp *src_a[2] = { unique_a, bind_N };
  PolyUOp *src_o[2] = { unique_o, bind_N };
  PolyUOp *buf_a   = poly_uop(ctx, POLY_OP_BUFFER, POLY_FLOAT32, src_a, 2, poly_arg_int(16));
  PolyUOp *buf_out  = poly_uop(ctx, POLY_OP_BUFFER, POLY_FLOAT32, src_o, 2, poly_arg_int(16));

  /* out = a + 1.0 */
  PolyUOp *one = poly_const_float(ctx, 1.0);
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, buf_a, one, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, buf_out, add, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  /* Execute with NO explicit var_bindings — BIND auto-extraction provides N=4 */
  float a_data[16] = { 10.0f, 20.0f, 30.0f, 40.0f };
  float out_data[16];
  memset(out_data, 0, sizeof(out_data));

  PolyBufferBinding bindings[2] = {
    { .buffer = buf_a,   .data = a_data },
    { .buffer = buf_out,  .data = out_data },
  };

  int ret = poly_realize_ex(ctx, sink, bindings, 2, NULL, 0);
  float result[16];
  memcpy(result, out_data, sizeof(result));

  poly_ctx_destroy(ctx);

  ASSERT_INT_EQ(ret, 0);
  /* First 4 elements should be a + 1.0 */
  ASSERT_FLOAT_EQ(result[0], 11.0f, 1e-5);
  ASSERT_FLOAT_EQ(result[1], 21.0f, 1e-5);
  ASSERT_FLOAT_EQ(result[2], 31.0f, 1e-5);
  ASSERT_FLOAT_EQ(result[3], 41.0f, 1e-5);
  /* Element 4 should be untouched */
  ASSERT_FLOAT_EQ(result[4], 0.0f, 1e-5);
  PASS();
}

/* define_var_cache_hit: execute with N=4 then N=8 (reuses compiled kernel).
 * Second call should hit schedule cache and produce correct results. */
TEST(rangeify, define_var_cache_hit) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *N = poly_define_var(ctx, "N", 1, 16);
  PolyUOp *buf_a   = poly_buffer_var(ctx, POLY_FLOAT32, N, NULL, 0);
  PolyUOp *buf_out  = poly_buffer_var(ctx, POLY_FLOAT32, N, NULL, 0);

  PolyUOp *one = poly_const_float(ctx, 1.0);
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, buf_a, one, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, buf_out, add, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  float a_data[16];
  float out_data[16];
  for (int i = 0; i < 16; i++) a_data[i] = (float)(i * 10);

  PolyBufferBinding bindings[2] = {
    { .buffer = buf_a,   .data = a_data },
    { .buffer = buf_out,  .data = out_data },
  };

  /* First call: N=4 (compiles kernel) */
  memset(out_data, 0, sizeof(out_data));
  PolyVarBinding var4[1] = {{ .var = N, .value = 4 }};
  int ret1 = poly_realize_ex(ctx, sink, bindings, 2, var4, 1);
  float r1[16];
  memcpy(r1, out_data, sizeof(r1));

  /* Second call: N=8 (should hit cache, reuse compiled kernel) */
  memset(out_data, 0, sizeof(out_data));
  PolyVarBinding var8[1] = {{ .var = N, .value = 8 }};
  int ret2 = poly_realize_ex(ctx, sink, bindings, 2, var8, 1);
  float r2[16];
  memcpy(r2, out_data, sizeof(r2));

  poly_ctx_destroy(ctx);

  ASSERT_INT_EQ(ret1, 0);
  /* First 4 elements correct for N=4 */
  ASSERT_FLOAT_EQ(r1[0], 1.0f, 1e-5);
  ASSERT_FLOAT_EQ(r1[1], 11.0f, 1e-5);
  ASSERT_FLOAT_EQ(r1[2], 21.0f, 1e-5);
  ASSERT_FLOAT_EQ(r1[3], 31.0f, 1e-5);
  ASSERT_FLOAT_EQ(r1[4], 0.0f, 1e-5);  /* untouched */

  ASSERT_INT_EQ(ret2, 0);
  /* First 8 elements correct for N=8 */
  ASSERT_FLOAT_EQ(r2[0], 1.0f, 1e-5);
  ASSERT_FLOAT_EQ(r2[7], 71.0f, 1e-5);
  ASSERT_FLOAT_EQ(r2[8], 0.0f, 1e-5);  /* untouched */
  PASS();
}

/* ── Regression: chained reductions (singleton + real) ─────────────── */
/* Verifies that REDUCE_AXIS with a singleton dim (size 1) followed by
 * a real reduction (size > 1) compiles and executes correctly via
 * poly_realize.  Regression for the bug where CONST(0) pseudo-ranges
 * from singleton dims entered REDUCE sources, producing END(CONST)
 * that corrupted scope depth in the C renderer.
 *
 * Graph shape: (1,4) * (1,4) → reduce axis 0 (singleton, keepdim) → (1,4)
 *              → reduce axis 1 (real sum of 4) → (1,1) → store
 *
 * REDUCE_AXIS keeps dims (sets reduced axis to 1), so reducing axis 0
 * of (1,4) gives (1,4) (no-op), and reducing axis 1 of (1,4) gives (1,1).
 */
TEST(rangeify, chained_singleton_reduce_e2e) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a_buf = poly_buffer_f32(ctx, 1);
  PolyUOp *w_buf = poly_buffer_f32(ctx, 4);
  PolyUOp *out_buf = poly_buffer_f32(ctx, 1);

  /* a reshaped to (1,4) via expand, w reshaped to (1,4) */
  int64_t sh14[] = { 1, 4 };
  PolyUOp *a_r = poly_reshape(ctx, a_buf, (int64_t[]){1, 1}, 2);
  PolyUOp *a_e = poly_expand(ctx, a_r, sh14, 2);
  PolyUOp *w_r = poly_reshape(ctx, w_buf, sh14, 2);

  /* Elementwise multiply: (1,4) * (1,4) -> (1,4) */
  PolyUOp *mul = poly_alu2(ctx, POLY_OP_MUL, a_e, w_r);

  /* Reduce axis 0 (size 1 -- singleton): (1,4) -> (1,4) [keepdim, no-op] */
  int64_t ax0[] = { 0 };
  PolyUOp *r0 = poly_reduce_axis(ctx, POLY_OP_ADD, mul, ax0, 1);

  /* Reduce axis 1 (size 4 -- real sum): (1,4) -> (1,1) */
  int64_t ax1[] = { 1 };
  PolyUOp *r1 = poly_reduce_axis(ctx, POLY_OP_ADD, r0, ax1, 1);

  /* Reshape to (1,) for store into scalar buffer */
  PolyUOp *r1_flat = poly_reshape(ctx, r1, (int64_t[]){1}, 1);

  /* Store + SINK */
  PolyUOp *store = poly_store_val(ctx, out_buf, r1_flat);
  PolyUOp *sink = poly_sink1(ctx, store);

  /* Execute */
  float a_data[] = { 2.0f };
  float w_data[] = { 1.0f, 2.0f, 3.0f, 4.0f };
  float out_data[] = { 0.0f };

  PolyBufferBinding bindings[] = {
    { a_buf, a_data },
    { w_buf, w_data },
    { out_buf, out_data },
  };

  int ret = poly_realize(ctx, sink, bindings, 3);
  poly_ctx_destroy(ctx);

  ASSERT_INT_EQ(ret, 0);
  /* Expected: 2*(1+2+3+4) = 20 */
  ASSERT_FLOAT_EQ(out_data[0], 20.0f, 1e-4);
  PASS();
}

/* ── Regression: mixed multi-axis reduction with singleton ─────────── */
/* Reduce over axes [0,2] where axis 0 is singleton (size 1) and axis 2
 * is real (size 3).  Ensures:
 *   - singleton axes don't generate loop ranges
 *   - non-singleton axes still generate RANGE(3)
 *   - generated kernel accumulates correctly */
TEST(rangeify, mixed_multiaxis_singleton_reduce_e2e) {
  PolyCtx *ctx = poly_ctx_new();

  /* Input: (1, 2, 3) tensor stored flat in 6 elements */
  PolyUOp *in_buf  = poly_buffer_f32(ctx, 6);
  PolyUOp *out_buf = poly_buffer_f32(ctx, 2);  /* output: (1, 2, 1) → flat (2) */

  /* Reshape to (1, 2, 3) */
  int64_t sh123[] = { 1, 2, 3 };
  PolyUOp *inp = poly_reshape(ctx, in_buf, sh123, 3);

  /* Reduce axes 0 and 2: (1, 2, 3) → (1, 2, 1) */
  int64_t axes[] = { 0, 2 };
  PolyUOp *r = poly_reduce_axis(ctx, POLY_OP_ADD, inp, axes, 2);

  /* Reshape to (2) for store */
  PolyUOp *r_flat = poly_reshape(ctx, r, (int64_t[]){2}, 1);

  /* Store + SINK */
  PolyUOp *store = poly_store_val(ctx, out_buf, r_flat);
  PolyUOp *sink = poly_sink1(ctx, store);

  /* Execute: [[1,2,3],[4,5,6]] → row sums [6, 15] */
  float in_data[] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
  float out_data[] = { 0.0f, 0.0f };

  PolyBufferBinding bindings[] = {
    { in_buf,  in_data },
    { out_buf, out_data },
  };

  int ret = poly_realize(ctx, sink, bindings, 2);
  poly_ctx_destroy(ctx);

  ASSERT_INT_EQ(ret, 0);
  ASSERT_FLOAT_EQ(out_data[0], 6.0f, 1e-4);   /* 1+2+3 */
  ASSERT_FLOAT_EQ(out_data[1], 15.0f, 1e-4);   /* 4+5+6 */
  PASS();
}
