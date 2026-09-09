/*
 * test_rangeify.c — Tests for the rangeify scheduling pipeline
 */

#define _POSIX_C_SOURCE 200809L
#include "test_harness.h"
#include "../src/schedule/rangeify.h"
#include "../src/schedule/schedule.h"
#include "../src/engine/schedule.h"
#include "../src/engine/realize.h"
#include "../src/ctx.h"
#include "../src/schedule/indexing.h"
#include "../src/schedule/multi.h"
#include "../src/codegen/codegen.h"
#include "../src/frontend.h"
#include "../src/tensor.h"
#include "../src/uop/spec.h"
#include "../src/utils.h"

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

/* Exercise current Tinygrad's pm_add_buffers matcher in isolation. Production
 * get_kernel_graph composes it with pm_add_param_range_tags in one rewrite. */
static PolyUOp *apply_add_buffers(PolyCtx *ctx, PolyUOp *sink) {
  int slot_counter = 0;
  return poly_graph_rewrite_ctx_ex2(ctx, sink, poly_pm_add_buffers(), &slot_counter, true, true);
}

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

#ifdef POLY_TESTING
extern void poly_test_range_scratch_fail_after(int count);

static bool range_scratch_failure_is_rejected(int fail_after) {
  RangeifyEnvSave pcontig = rangeify_save_env("PCONTIG");
  setenv("PCONTIG", "2", 1);
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *buf = poly_test_buffer(ctx, POLY_FLOAT32, 4);
  PolyUOp *ones = poly_expand(ctx, poly_const_float(ctx, 1.0), (int64_t[]){4}, 1);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, buf, ones));
  poly_test_range_scratch_fail_after(fail_after);
  bool rejected = poly_run_rangeify(ctx, sink, false) == NULL;
  poly_test_range_scratch_fail_after(-1);
  PolyUOp *retried = poly_run_rangeify(ctx, sink, false);
  int n = 0;
  PolyUOp **topo = retried ? poly_toposort_alloc(ctx, retried, &n) : NULL;
  int ranges = 0, stores = 0, ends = 0;
  for (int i = 0; topo && i < n; i++) {
    ranges += topo[i]->op == POLY_OP_RANGE;
    stores += topo[i]->op == POLY_OP_STORE;
    ends += topo[i]->op == POLY_OP_END;
  }
  bool ok = rejected && ranges == 1 && stores == 1 && ends == 1;
  poly_toposort_free(topo);
  poly_ctx_destroy(ctx);
  rangeify_restore_env(&pcontig);
  return ok;
}

TEST(rangeify, range_scratch_failure_does_not_downgrade_pcontig) {
  /* Python run_rangeify raises on list allocation failure; PCONTIG stays2. */
  ASSERT_TRUE(range_scratch_failure_is_rejected(0));
  PASS();
}

TEST(rangeify, range_scratch_failure_reclaims_consumer_ranges) {
  ASSERT_TRUE(range_scratch_failure_is_rejected(1));
  PASS();
}

TEST(rangeify, range_scratch_failure_reclaims_consumer_lengths) {
  ASSERT_TRUE(range_scratch_failure_is_rejected(2));
  PASS();
}

extern void poly_test_indexing_alloc_fail_after(const char *site, int count);
extern bool poly_test_indexing_alloc_failed(void);

static PolyUOp *metadata_test_sink(PolyCtx *ctx, int kind) {
  if (kind == 4) {
    PolyUOp *a = poly_reshape(ctx, poly_test_buffer(ctx, POLY_FLOAT32, 8), (int64_t[]){2, 4}, 2);
    PolyUOp *b = poly_reshape(ctx, poly_test_buffer(ctx, POLY_FLOAT32, 8), (int64_t[]){2, 4}, 2);
    PolyUOp *x = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, a, b, poly_arg_none());
    PolyUOp *y = poly_uop2(
        ctx, POLY_OP_ADD, POLY_FLOAT32, x, poly_flip(ctx, x, (int64_t[]){1}, 1), poly_arg_none()
    );
    PolyUOp *out = poly_reshape(ctx, poly_test_buffer(ctx, POLY_FLOAT32, 8), (int64_t[]){2, 4}, 2);
    return poly_sink1(ctx, poly_store_val(ctx, out, y));
  }
  if (kind == 3) {
    PolyUOp *stores[17];
    for (int i = 0; i < 17; i++)
      stores[i] = poly_store_val(
          ctx, poly_test_buffer(ctx, POLY_FLOAT32, 4),
          poly_expand(ctx, poly_const_float(ctx, i), (int64_t[]){4}, 1)
      );
    return poly_uop(ctx, POLY_OP_SINK, POLY_VOID, stores, 17, poly_arg_none());
  }
  if (kind == 1) {
    PolyUOp *x = poly_expand(ctx, poly_const_float(ctx, 1.0), (int64_t[]){2, 2, 2, 2, 2}, 5);
    PolyUOp *y = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, x, x, poly_arg_none());
    PolyUOp *out =
        poly_reshape(ctx, poly_test_buffer(ctx, POLY_FLOAT32, 32), (int64_t[]){2, 2, 2, 2, 2}, 5);
    return poly_sink1(ctx, poly_store_val(ctx, out, y));
  }
  PolyUOp *a = poly_test_buffer(ctx, POLY_FLOAT32, 4);
  PolyUOp *b = poly_test_buffer(ctx, POLY_FLOAT32, 4);
  PolyUOp *x = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, a, b, poly_arg_none());
  PolyUOp *y;
  int size = 4;
  if (kind == 0) {
    PolyUOp *c1 =
        poly_pad(ctx, poly_shrink(ctx, x, (int64_t[][2]){{0, 3}}, 1), (int64_t[][2]){{0, 2}}, 1);
    PolyUOp *c2 = poly_pad(ctx, x, (int64_t[][2]){{0, 1}}, 1);
    y = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, c1, c2, poly_arg_none());
    size = 5;
  } else {
    /* Force consumer-list growth beyond its first allocation. */
    y = x;
    for (int i = 1; i <= 6; i++) {
      PolyUOp *term =
          poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, x, poly_const_float(ctx, i), poly_arg_none());
      y = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, y, term, poly_arg_none());
    }
  }
  return poly_sink1(ctx, poly_store_val(ctx, poly_test_buffer(ctx, POLY_FLOAT32, size), y));
}

static bool metadata_failures_are_rejected(const char *site) {
  RangeifyEnvSave pcontig = rangeify_save_env("PCONTIG");
  setenv("PCONTIG", "2", 1);
  bool ok = true;
  int failures = 0;
  for (int kind = 0; kind < 5 && ok; kind++) {
    bool exhausted = false;
    for (int nth = 0; nth < 4096 && !exhausted && ok; nth++) {
      PolyCtx *ctx = poly_ctx_new();
      PolyUOp *sink = metadata_test_sink(ctx, kind);
      poly_test_indexing_alloc_fail_after(site, nth);
      PolyUOp *result = poly_run_rangeify(ctx, sink, false);
      bool failed = poly_test_indexing_alloc_failed();
      poly_test_indexing_alloc_fail_after(NULL, -1);
      ok = failed ? result == NULL : result != NULL;
      failures += failed;
      exhausted = !failed;
      /* A discarded private map must not poison a retry in the same context. */
      if (failed && ok) result = poly_run_rangeify(ctx, sink, false);
      if (ok) {
        int stores = kind == 3 ? 17 : 1;
        ok = result && count_ops(ctx, result, POLY_OP_STORE) == stores &&
             count_ops(ctx, result, POLY_OP_END) == stores;
        if (ok && kind == 0)
          ok = count_ops(ctx, result, POLY_OP_STAGE) == 0 &&
               count_ops(ctx, result, POLY_OP_WHERE) == 3;
      }
      poly_ctx_destroy(ctx);
    }
    ok = ok && exhausted;
  }
  rangeify_restore_env(&pcontig);
  printf("indexing %s: %d injected failures\n", site, failures);
  return ok && failures > 0;
}

TEST(rangeify, metadata_context_failure) {
  ASSERT_TRUE(metadata_failures_are_rejected("context"));
  PASS();
}
TEST(rangeify, metadata_consumer_failure) {
  ASSERT_TRUE(metadata_failures_are_rejected("consumer"));
  PASS();
}
TEST(rangeify, metadata_ending_failure) {
  ASSERT_TRUE(metadata_failures_are_rejected("ending"));
  PASS();
}
TEST(rangeify, metadata_realize_failure) {
  ASSERT_TRUE(metadata_failures_are_rejected("realize"));
  PASS();
}
TEST(rangeify, metadata_shape_failure) {
  ASSERT_TRUE(metadata_failures_are_rejected("shape"));
  PASS();
}
TEST(rangeify, metadata_range_failure) {
  ASSERT_TRUE(metadata_failures_are_rejected("range"));
  PASS();
}
TEST(rangeify, metadata_axes_failure) {
  ASSERT_TRUE(metadata_failures_are_rejected("axes"));
  PASS();
}
TEST(rangeify, metadata_valids_failure) {
  ASSERT_TRUE(metadata_failures_are_rejected("valids"));
  PASS();
}
TEST(rangeify, metadata_rewrite_failure) {
  ASSERT_TRUE(metadata_failures_are_rejected("rewrite"));
  PASS();
}

static bool metadata_replacement_preserves_owner(const char *site, int count) {
  bool ok = true;
  for (int nth = 0; nth < count; nth++) {
    PolyCtx *ctx = poly_ctx_new();
    PolyUOp *sink = poly_sink1(
        ctx, poly_store_val(
                 ctx, poly_test_buffer(ctx, POLY_FLOAT32, 4),
                 poly_expand(ctx, poly_const_float(ctx, 1.0), (int64_t[]){4}, 1)
             )
    );
    PolyUOp *store = sink->src[0];
    PolyIndexingCtx *ictx = poly_indexing_ctx_new(ctx);
    bool built = poly_realize_map_build(ictx, sink) && poly_range_propagate(ictx, sink);
    PolyRangeEntry *entry = poly_range_map_get(ictx, store);
    PolyRealizeInfo *ri = get_realize_info(ictx, store);
    PolyUOp **in = entry ? entry->in_rngs : NULL;
    PolyUOp **out = entry ? entry->out_rngs : NULL;
    int *axes = ri ? ri->axes : NULL;
    poly_test_indexing_alloc_fail_after(site, nth);
    bool propagated = poly_range_propagate(ictx, sink);
    bool injected = poly_test_indexing_alloc_failed();
    poly_test_indexing_alloc_fail_after(NULL, -1);
    ok = ok && built && injected && !propagated;
    if (!strcmp(site, "range"))
      ok = ok && entry == poly_range_map_get(ictx, store) && entry->in_rngs == in &&
           entry->out_rngs == out && entry->n_in == 1 && entry->n_out == 1;
    else
      ok = ok && ri == get_realize_info(ictx, store) && ri->axes == axes && ri->n_axes == 1 &&
           axes[0] == 0;
    poly_indexing_ctx_destroy(ictx);
    poly_ctx_destroy(ctx);
  }
  return ok;
}

TEST(rangeify, metadata_range_replacement_keeps_previous_entry) {
  ASSERT_TRUE(metadata_replacement_preserves_owner("range", 3));
  PASS();
}

TEST(rangeify, metadata_axes_replacement_keeps_previous_entry) {
  ASSERT_TRUE(metadata_replacement_preserves_owner("axes", 1));
  PASS();
}
#endif

/* Consumer map tests */

TEST(rangeify, consumer_map_chain) {
  /* a → ADD(a,b) → STORE → SINK: verify consumer counts */
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_test_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *b = poly_test_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *c = poly_test_buffer(ctx, POLY_FLOAT32, 10);
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

  PolyUOp *a = poly_test_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *out = poly_test_buffer(ctx, POLY_FLOAT32, 10);
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

  PolyUOp *a = poly_test_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *b = poly_test_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *c = poly_test_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *out1 = poly_test_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *out2 = poly_test_buffer(ctx, POLY_FLOAT32, 10);

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

TEST(rangeify, consumer_map_tracks_data_not_shape_sources) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  /* Current tinygrad schedule/indexing.py:data_srcs keeps RESHAPE src[0] as
   * data and excludes its shape STACK from the consumer map. A tensor STACK,
   * in contrast, consumes every value source. */
  PolyUOp *a = poly_test_buffer(ctx, POLY_FLOAT32, 2);
  PolyUOp *b = poly_test_buffer(ctx, POLY_FLOAT32, 2);
  PolyUOp *reshaped = poly_reshape(ctx, a, (int64_t[]){2, 1}, 2);
  ASSERT_NOT_NULL(reshaped);
  ASSERT_INT_EQ(reshaped->n_src, 2);
  ASSERT_INT_EQ(reshaped->src[1]->op, POLY_OP_STACK);
  PolyUOp *values[] = {reshaped, b};
  PolyUOp *stacked = poly_uop_stack(ctx, values, 2);
  PolyUOp *sink = poly_sink1(ctx, stacked);

  PolyMap *cmap = poly_consumer_map_build(ctx, sink);
  ASSERT_NOT_NULL(cmap);
  PolyConsumerList *a_consumers = poly_consumer_map_get(cmap, a);
  PolyConsumerList *shape_consumers = poly_consumer_map_get(cmap, reshaped->src[1]);
  PolyConsumerList *reshape_consumers = poly_consumer_map_get(cmap, reshaped);
  PolyConsumerList *b_consumers = poly_consumer_map_get(cmap, b);
  ASSERT_NOT_NULL(a_consumers);
  ASSERT_NOT_NULL(shape_consumers);
  ASSERT_NOT_NULL(reshape_consumers);
  ASSERT_NOT_NULL(b_consumers);
  ASSERT_INT_EQ(a_consumers->count, 1);
  ASSERT_PTR_EQ(a_consumers->items[0], reshaped);
  ASSERT_INT_EQ(shape_consumers->count, 0);
  ASSERT_INT_EQ(reshape_consumers->count, 1);
  ASSERT_PTR_EQ(reshape_consumers->items[0], stacked);
  ASSERT_INT_EQ(b_consumers->count, 1);
  ASSERT_PTR_EQ(b_consumers->items[0], stacked);

  destroy_consumer_map(cmap);
  poly_ctx_destroy(ctx);
  PASS();
}

/* Realize map tests */

TEST(rangeify, realize_map_sink_sources) {
  /* SINK(STORE(out, a+b)): the STORE should be realized */
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_test_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *b = poly_test_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *out = poly_test_buffer(ctx, POLY_FLOAT32, 10);
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

  PolyUOp *a = poly_test_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *b = poly_test_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *out = poly_test_buffer(ctx, POLY_FLOAT32, 10);
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

  PolyUOp *a = poly_test_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *b = poly_test_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *d = poly_test_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *out = poly_test_buffer(ctx, POLY_FLOAT32, 10);
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

  PolyUOp *a = poly_test_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *out = poly_reshape(ctx, poly_test_buffer(ctx, POLY_FLOAT32, 1), NULL, 0);

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

  PolyUOp *inner_buffer = poly_test_buffer(ctx, POLY_FLOAT32, 4);
  PolyUOp *outer_buffer = poly_test_buffer(ctx, POLY_FLOAT32, 4);
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
  poly_realize_map_build(ictx, sink);
  poly_range_propagate(ictx, sink);

  ASSERT_NOT_NULL(poly_range_map_get(ictx, inner_store));
  ASSERT_NOT_NULL(poly_range_map_get(ictx, outer_store));
  ASSERT_TRUE(poly_range_map_get(ictx, inner_after) == NULL);

  PolyUOp *rangeified = poly_apply_rangeify(ictx, sink);
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

TEST(rangeify, range_prop_reuses_existing_range_shape_bound) {
  /* Tinygrad 2026-08-22/a9069c177a9d IndexingContext.new_range returns an
   * existing RANGE unchanged; nesting it changes identity and axis semantics. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(8));
  PolyUOp *shape_range =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, bound, poly_arg_range(77, POLY_AXIS_WEAK));
  int float_id = poly_dtype_id_by_name("float");
  PolyUOp *buffer =
      poly_test_buffer_var_by_id(ctx, float_id, shape_range, NULL, 0, POLY_DEVICE_CPU);
  PolyUOp *contiguous = poly_contiguous(ctx, buffer);
  PolyUOp *sink = poly_sink1(ctx, contiguous);
  ASSERT_NOT_NULL(buffer);
  ASSERT_NOT_NULL(contiguous);
  ASSERT_NOT_NULL(sink);

  PolyIndexingCtx *ictx = poly_indexing_ctx_new(ctx);
  poly_realize_map_build(ictx, sink);
  poly_range_propagate(ictx, sink);
  PolyRangeEntry *entry = poly_range_map_get(ictx, contiguous);
  ASSERT_NOT_NULL(entry);
  ASSERT_INT_EQ(entry->n_out, 1);
  ASSERT_PTR_EQ(entry->out_rngs[0], shape_range);

  poly_indexing_ctx_destroy(ictx);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, range_prop_elementwise) {
  /* a + b → STORE → SINK: all fused, share same ranges */
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_test_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *b = poly_test_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *out = poly_test_buffer(ctx, POLY_FLOAT32, 10);
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

TEST(rangeify, broadcast_rngs_zero_expanded_axes_like_current_tinygrad) {
  /* Current tinygrad schedule/indexing.py:broadcast_rngs maps consumer ranges
   * per source: (2,1) sees (row,0), while (1,3) sees (0,col). */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a0 = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 2, POLY_DEVICE_CPU);
  PolyUOp *b0 = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 3, POLY_DEVICE_CPU);
  PolyUOp *out0 = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 6, POLY_DEVICE_CPU);
  PolyUOp *a = poly_reshape(ctx, a0, (int64_t[]){2, 1}, 2);
  PolyUOp *b = poly_reshape(ctx, b0, (int64_t[]){1, 3}, 2);
  PolyUOp *out = poly_reshape(ctx, out0, (int64_t[]){2, 3}, 2);
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, a, b, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out, add, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());
  ASSERT_NOT_NULL(a);
  ASSERT_NOT_NULL(b);
  ASSERT_NOT_NULL(add);
  ASSERT_NOT_NULL(sink);

  int axes[POLY_MAX_DIMS] = {-1, -1};
  ASSERT_INT_EQ(poly_broadcast_axes(ctx, a, add, axes, POLY_MAX_DIMS), 1);
  ASSERT_INT_EQ(axes[0], 1);
  ASSERT_INT_EQ(poly_broadcast_axes(ctx, b, add, axes, POLY_MAX_DIMS), 1);
  ASSERT_INT_EQ(axes[0], 0);

  PolyIndexingCtx *ictx = poly_indexing_ctx_new(ctx);
  poly_realize_map_build(ictx, sink);
  poly_range_propagate(ictx, sink);
  PolyRangeEntry *re_a = poly_range_map_get(ictx, a);
  PolyRangeEntry *re_b = poly_range_map_get(ictx, b);
  ASSERT_NOT_NULL(re_a);
  ASSERT_NOT_NULL(re_b);
  ASSERT_INT_EQ(re_a->n_out, 2);
  ASSERT_INT_EQ(re_b->n_out, 2);
  ASSERT_EQ(re_a->out_rngs[0]->op, POLY_OP_RANGE);
  ASSERT_EQ(re_a->out_rngs[1]->op, POLY_OP_CONST);
  ASSERT_INT_EQ(re_a->out_rngs[1]->arg.kind, POLY_ARG_INT);
  ASSERT_INT_EQ(re_a->out_rngs[1]->arg.i, 0);
  ASSERT_EQ(re_b->out_rngs[0]->op, POLY_OP_CONST);
  ASSERT_INT_EQ(re_b->out_rngs[0]->arg.kind, POLY_ARG_INT);
  ASSERT_INT_EQ(re_b->out_rngs[0]->arg.i, 0);
  ASSERT_EQ(re_b->out_rngs[1]->op, POLY_OP_RANGE);
  poly_indexing_ctx_destroy(ictx);

  float a_data[] = {1.0f, 2.0f};
  float b_data[] = {10.0f, 20.0f, 30.0f};
  float out_data[6] = {0};
  PolyTestBufferView bindings[] = {
      POLY_TEST_HOST_VIEW(a0, a_data),
      POLY_TEST_HOST_VIEW(b0, b_data),
      POLY_TEST_HOST_VIEW(out0, out_data),
  };
  ASSERT_INT_EQ(poly_test_realize_buffer_views(ctx, sink, bindings, 3), 0);
  const float expected[] = {11.0f, 21.0f, 31.0f, 12.0f, 22.0f, 32.0f};
  for (int i = 0; i < 6; i++)
    ASSERT_FLOAT_EQ(out_data[i], expected[i], 1e-6f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, range_prop_chain) {
  /* (a+b)*c → STORE → SINK: all ops share same ranges */
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_test_buffer(ctx, POLY_FLOAT32, 8);
  PolyUOp *b = poly_test_buffer(ctx, POLY_FLOAT32, 8);
  PolyUOp *c = poly_test_buffer(ctx, POLY_FLOAT32, 8);
  PolyUOp *out = poly_test_buffer(ctx, POLY_FLOAT32, 8);
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

  PolyUOp *a = poly_test_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *out = poly_reshape(ctx, poly_test_buffer(ctx, POLY_FLOAT32, 1), NULL, 0);
  int64_t axes[] = {0};
  PolyUOp *reduce = poly_reduce_axis(ctx, POLY_OP_ADD, a, axes, 1);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out, reduce, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyIndexingCtx *ictx = poly_indexing_ctx_new(ctx);
  poly_realize_map_build(ictx, sink);
  poly_range_propagate(ictx, sink);

  /* Current UOp._rop removes reduced axes. STORE therefore consumes a scalar
   * REDUCE and has no output ranges (uop/ops.py:629-638). */
  PolyRangeEntry *re_store = poly_range_map_get(ictx, store);
  ASSERT_NOT_NULL(re_store);
  ASSERT_INT_EQ(re_store->n_out, 0);

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

  PolyUOp *a = poly_test_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *b = poly_test_buffer(ctx, POLY_FLOAT32, 1);
  PolyUOp *out = poly_test_buffer(ctx, POLY_FLOAT32, 1);
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

  PolyUOp *a = poly_test_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *out = poly_test_buffer(ctx, POLY_FLOAT32, 10);
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

  PolyUOp *a = poly_test_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *b = poly_test_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *c = poly_test_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *out1 = poly_test_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *out2 = poly_test_buffer(ctx, POLY_FLOAT32, 10);

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

TEST(rangeify, range_prop_more_than_sixteen_includes_mismatched_last_consumer) {
  /* Current run_rangeify considers every consumer. The first 16 inherit the
   * final store range; the 17th FLIP has a different input index, so x realizes. */
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_test_buffer(ctx, POLY_FLOAT32, 4);
  PolyUOp *b = poly_test_buffer(ctx, POLY_FLOAT32, 4);
  PolyUOp *x = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, a, b, poly_arg_none());

  PolyUOp *acc = NULL;
  for (int i = 0; i < 16; i++) {
    PolyUOp *c = poly_const_float(ctx, (double)(i + 1));
    PolyUOp *term = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, x, c, poly_arg_none());
    acc = acc ? poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, acc, term, poly_arg_none()) : term;
  }
  PolyUOp *flipped = poly_flip(ctx, x, (int64_t[]){0}, 1);
  acc = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, acc, flipped, poly_arg_none());

  PolyUOp *out = poly_test_buffer(ctx, POLY_FLOAT32, 4);
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

TEST(rangeify, range_prop_more_than_sixteen_matching_consumers_stays_fused) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_test_buffer(ctx, POLY_FLOAT32, 4);
  PolyUOp *b = poly_test_buffer(ctx, POLY_FLOAT32, 4);
  PolyUOp *x = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, a, b, poly_arg_none());
  PolyUOp *acc = NULL;
  for (int i = 0; i < 17; i++) {
    PolyUOp *term = poly_uop2(
        ctx, POLY_OP_ADD, POLY_FLOAT32, x, poly_const_float(ctx, (double)(i + 1)), poly_arg_none()
    );
    acc = acc ? poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, acc, term, poly_arg_none()) : term;
  }
  PolyUOp *out = poly_test_buffer(ctx, POLY_FLOAT32, 4);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out, acc, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyIndexingCtx *ictx = poly_indexing_ctx_new(ctx);
  poly_realize_map_build(ictx, sink);
  poly_range_propagate(ictx, sink);

  ASSERT_FALSE(poly_is_realized(ictx, x));
  PolyRangeEntry *re_x = poly_range_map_get(ictx, x);
  PolyRangeEntry *re_store = poly_range_map_get(ictx, store);
  ASSERT_NOT_NULL(re_x);
  ASSERT_NOT_NULL(re_store);
  ASSERT_INT_EQ(re_x->n_out, 1);
  ASSERT_PTR_EQ(re_x->out_rngs[0], re_store->out_rngs[0]);

  poly_indexing_ctx_destroy(ictx);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, range_prop_expand_ending_realizes_elementwise_default_pcontig) {
  /* tinygrad indexing.py realizes ended ranges unconditionally when
   * PCONTIG <= 1. The default tinygrad setting is PCONTIG=0, so an
   * elementwise op feeding EXPAND must become a realize point here. */
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a_flat = poly_test_buffer(ctx, POLY_FLOAT32, 2);
  PolyUOp *b_flat = poly_test_buffer(ctx, POLY_FLOAT32, 2);
  PolyUOp *c_flat = poly_test_buffer(ctx, POLY_FLOAT32, 8);
  PolyUOp *out = poly_reshape(ctx, poly_test_buffer(ctx, POLY_FLOAT32, 8), (int64_t[]){2, 4}, 2);

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

TEST(rangeify, range_prop_pcontig_one_realizes_only_disagreeing_axis) {
  /* Tinygrad 2026-08-22/a9069c177a9d schedule/indexing.py:243-267:
   * PCONTIG=1 preserves axis 0 shared by x and x.flip(1), realizing axis 1. */
  RangeifyEnvSave pcontig = rangeify_save_env("PCONTIG");
  setenv("PCONTIG", "1", 1);

  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_reshape(ctx, poly_test_buffer(ctx, POLY_FLOAT32, 8), (int64_t[]){2, 4}, 2);
  PolyUOp *b = poly_reshape(ctx, poly_test_buffer(ctx, POLY_FLOAT32, 8), (int64_t[]){2, 4}, 2);
  PolyUOp *x = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, a, b, poly_arg_none());
  PolyUOp *flipped = poly_flip(ctx, x, (int64_t[]){1}, 1);
  PolyUOp *sum = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, x, flipped, poly_arg_none());
  PolyUOp *out = poly_reshape(ctx, poly_test_buffer(ctx, POLY_FLOAT32, 8), (int64_t[]){2, 4}, 2);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out, sum, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyIndexingCtx *ictx = poly_indexing_ctx_new(ctx);
  poly_realize_map_build(ictx, sink);
  poly_range_propagate(ictx, sink);

  PolyRealizeInfo *ri = get_realize_info(ictx, x);
  ASSERT_NOT_NULL(ri);
  ASSERT_INT_EQ(ri->n_axes, 1);
  ASSERT_INT_EQ(ri->axes[0], 1);
  PolyRangeEntry *re_x = poly_range_map_get(ictx, x);
  ASSERT_NOT_NULL(re_x);
  ASSERT_INT_EQ(re_x->n_out, 2);
  ASSERT_PTR_EQ(re_x->out_rngs[0], poly_range_map_get(ictx, sum)->in_rngs[0]);

  poly_indexing_ctx_destroy(ictx);
  poly_ctx_destroy(ctx);
  rangeify_restore_env(&pcontig);
  PASS();
}

TEST(rangeify, pcontig_two_keeps_outer_double_matmul_axis_local) {
  /* Tinygrad 2026-08-22/a9069c177a9d schedule/indexing.py:269-278:
   * PCONTIG=2 realizes only the inner first-matmul axis needed by the second
   * matmul, producing STAGE(addrspace=LOCAL, shape=(16,)). */
  RangeifyEnvSave pcontig = rangeify_save_env("PCONTIG");
  setenv("PCONTIG", "2", 1);

  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_reshape(ctx, poly_test_buffer(ctx, POLY_FLOAT32, 256), (int64_t[]){16, 16}, 2);
  PolyUOp *b = poly_reshape(ctx, poly_test_buffer(ctx, POLY_FLOAT32, 256), (int64_t[]){16, 16}, 2);
  PolyUOp *c = poly_reshape(ctx, poly_test_buffer(ctx, POLY_FLOAT32, 256), (int64_t[]){16, 16}, 2);
  PolyUOp *out =
      poly_reshape(ctx, poly_test_buffer(ctx, POLY_FLOAT32, 256), (int64_t[]){16, 16}, 2);
  PolyUOp *value = poly_dot(ctx, poly_dot(ctx, a, b), c);
  PolyUOp *rangeified =
      poly_run_rangeify(ctx, poly_sink1(ctx, poly_store_val(ctx, out, value)), false);
  ASSERT_NOT_NULL(rangeified);

  int n_topo = 0;
  PolyUOp **topo = poly_toposort_alloc(ctx, rangeified, &n_topo);
  int stages = 0;
  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    if (u->op != POLY_OP_STAGE) continue;
    stages++;
    ASSERT_INT_EQ(poly_bufferize_arg_addrspace(u->arg), POLY_ADDR_LOCAL);
    ASSERT_INT_EQ(poly_uop_ndim(ctx, u), 1);
    ASSERT_INT_EQ(poly_uop_shape_dim(ctx, u, 0)->arg.i, 16);
  }
  ASSERT_INT_EQ(stages, 1);

  poly_toposort_free(topo);
  poly_ctx_destroy(ctx);
  rangeify_restore_env(&pcontig);
  PASS();
}

TEST(rangeify, pcontig_three_inlines_more_than_three_accessed_params) {
  /* Tinygrad 2026-08-22/a9069c177a9d rangeify.py:221-289 permits removal
   * above the three-resource cost cap only when PCONTIG>2. */
  RangeifyEnvSave pcontig = rangeify_save_env("PCONTIG");
  setenv("PCONTIG", "3", 1);
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *four = poly_const_int(ctx, 4);
  PolyUOp *range =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, four, poly_arg_range(0, POLY_AXIS_WEAK));
  PolyParamArg args[4] = {0};
  PolyUOp *value = NULL;
  for (int i = 0; i < 4; i++) {
    args[i] = (PolyParamArg
    ){.slot = i, .dtype = POLY_FLOAT32, .addrspace = POLY_ADDR_GLOBAL, .device = "CPU"};
    PolyUOp *param = poly_uop1(ctx, POLY_OP_PARAM, POLY_FLOAT32, four, poly_arg_param(&args[i]));
    PolyUOp *indexed = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, param, range, poly_arg_none());
    value = value ? poly_alu2(ctx, POLY_OP_ADD, value, indexed) : indexed;
  }
  PolyUOp *stage_src[] = {value, range};
  PolyUOp *stage = poly_uop(
      ctx, POLY_OP_STAGE, POLY_FLOAT32, stage_src, 2,
      poly_arg_bufferize_opts("CPU", POLY_ADDR_GLOBAL, true)
  );
  PolyUOp *index = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, stage, range, poly_arg_none());
  PolyUOp *ret = poly_pm_rewrite(poly_pm_remove_bufferize(), ctx, index);
  ASSERT_NOT_NULL(ret);
  ASSERT_INT_EQ(ret->op, POLY_OP_ADD);
  ASSERT_INT_EQ(count_ops(ctx, ret, POLY_OP_STAGE), 0);
  poly_ctx_destroy(ctx);
  rangeify_restore_env(&pcontig);
  PASS();
}

TEST(rangeify, pcontig_three_inlines_profitable_reduce_stage) {
  /* Tinygrad rangeify.py:258-286 removes a buffer-in-reduce stage only for
   * PCONTIG>2 when output/input materialization exceeds the 10x threshold. */
  RangeifyEnvSave pcontig = rangeify_save_env("PCONTIG");
  setenv("PCONTIG", "3", 1);
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *outer = poly_uop1(
      ctx, POLY_OP_RANGE, POLY_WEAKINT, poly_const_int(ctx, 128), poly_arg_range(10, POLY_AXIS_WEAK)
  );
  PolyUOp *reduce_range = poly_uop1(
      ctx, POLY_OP_RANGE, POLY_WEAKINT, poly_const_int(ctx, 4), poly_arg_range(11, POLY_AXIS_REDUCE)
  );
  PolyUOp *replacement = poly_uop1(
      ctx, POLY_OP_RANGE, POLY_WEAKINT, poly_const_int(ctx, 128), poly_arg_range(12, POLY_AXIS_WEAK)
  );
  PolyParamArg arg = {
      .slot = 10, .dtype = POLY_FLOAT32, .addrspace = POLY_ADDR_GLOBAL, .device = "CPU"};
  PolyUOp *param =
      poly_uop1(ctx, POLY_OP_PARAM, POLY_FLOAT32, poly_const_int(ctx, 4), poly_arg_param(&arg));
  PolyUOp *param_index = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, param, outer, poly_arg_none());
  PolyUOp *reduce_src[] = {param_index, reduce_range};
  PolyUOp *reduce =
      poly_uop(ctx, POLY_OP_REDUCE, POLY_FLOAT32, reduce_src, 2, poly_arg_reduce(POLY_OP_ADD, 0));
  PolyUOp *stage_src[] = {reduce, outer};
  PolyUOp *stage = poly_uop(
      ctx, POLY_OP_STAGE, POLY_FLOAT32, stage_src, 2,
      poly_arg_bufferize_opts("CPU", POLY_ADDR_GLOBAL, true)
  );
  PolyUOp *index = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, stage, replacement, poly_arg_none());
  PolyUOp *ret = poly_pm_rewrite(poly_pm_remove_bufferize(), ctx, index);
  ASSERT_NOT_NULL(ret);
  ASSERT_INT_EQ(ret->op, POLY_OP_REDUCE);
  ASSERT_INT_EQ(count_ops(ctx, ret, POLY_OP_STAGE), 0);

  PolyUOp *small_outer = poly_uop1(
      ctx, POLY_OP_RANGE, POLY_WEAKINT, poly_const_int(ctx, 8), poly_arg_range(13, POLY_AXIS_WEAK)
  );
  PolyUOp *small_replacement = poly_uop1(
      ctx, POLY_OP_RANGE, POLY_WEAKINT, poly_const_int(ctx, 8), poly_arg_range(14, POLY_AXIS_WEAK)
  );
  PolyUOp *small_index =
      poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, param, small_outer, poly_arg_none());
  PolyUOp *small_reduce_src[] = {small_index, reduce_range};
  PolyUOp *small_reduce = poly_uop(
      ctx, POLY_OP_REDUCE, POLY_FLOAT32, small_reduce_src, 2, poly_arg_reduce(POLY_OP_ADD, 0)
  );
  PolyUOp *small_stage_src[] = {small_reduce, small_outer};
  PolyUOp *small_stage = poly_uop(
      ctx, POLY_OP_STAGE, POLY_FLOAT32, small_stage_src, 2,
      poly_arg_bufferize_opts("CPU", POLY_ADDR_GLOBAL, true)
  );
  PolyUOp *small_consumer =
      poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, small_stage, small_replacement, poly_arg_none());
  ASSERT_TRUE(poly_pm_rewrite(poly_pm_remove_bufferize(), ctx, small_consumer) == NULL);
  poly_ctx_destroy(ctx);
  rangeify_restore_env(&pcontig);
  PASS();
}

TEST(rangeify, pcontig_three_preserves_partial_local_stage) {
  /* Tinygrad rangeify.py:269-282 keeps local-index ranges staged while
   * substituting ordinary ranges through the surrounding reduce graph. */
  RangeifyEnvSave pcontig = rangeify_save_env("PCONTIG");
  setenv("PCONTIG", "3", 1);
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *bound128 = poly_const_int(ctx, 128);
  PolyUOp *bound4 = poly_const_int(ctx, 4);
  PolyUOp *k0 =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, bound128, poly_arg_range(20, POLY_AXIS_WEAK));
  PolyUOp *k1 =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, bound128, poly_arg_range(21, POLY_AXIS_WEAK));
  PolyUOp *v0 =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, bound128, poly_arg_range(22, POLY_AXIS_WEAK));
  PolyUOp *v1 =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, bound128, poly_arg_range(23, POLY_AXIS_WEAK));
  PolyUOp *reduce_range =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, bound4, poly_arg_range(24, POLY_AXIS_REDUCE));
  PolyParamArg args[2] = {
      {.slot = 20, .dtype = POLY_FLOAT32, .addrspace = POLY_ADDR_GLOBAL, .device = "CPU"},
      {.slot = 21, .dtype = POLY_FLOAT32, .addrspace = POLY_ADDR_GLOBAL, .device = "CPU"},
  };
  PolyUOp *p0 = poly_uop1(ctx, POLY_OP_PARAM, POLY_FLOAT32, bound4, poly_arg_param(&args[0]));
  PolyUOp *p1 = poly_uop1(ctx, POLY_OP_PARAM, POLY_FLOAT32, bound4, poly_arg_param(&args[1]));
  PolyUOp *local_value = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, p0, k0, poly_arg_none());
  PolyUOp *local_src[] = {local_value, k0};
  PolyUOp *local_stage = poly_uop(
      ctx, POLY_OP_STAGE, POLY_FLOAT32, local_src, 2,
      poly_arg_bufferize_opts(NULL, POLY_ADDR_LOCAL, true)
  );
  PolyUOp *local_index =
      poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, local_stage, k0, poly_arg_none());
  PolyUOp *p1_index = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, p1, k1, poly_arg_none());
  PolyUOp *reduce_src[] = {p1_index, reduce_range};
  PolyUOp *reduce =
      poly_uop(ctx, POLY_OP_REDUCE, POLY_FLOAT32, reduce_src, 2, poly_arg_reduce(POLY_OP_ADD, 0));
  PolyUOp *value = poly_alu2(ctx, POLY_OP_ADD, local_index, reduce);
  PolyUOp *stage_src[] = {value, k0, k1};
  PolyUOp *stage = poly_uop(
      ctx, POLY_OP_STAGE, POLY_FLOAT32, stage_src, 3,
      poly_arg_bufferize_opts("CPU", POLY_ADDR_GLOBAL, true)
  );
  PolyUOp *index_src[] = {stage, v0, v1};
  PolyUOp *index = poly_uop(ctx, POLY_OP_INDEX, POLY_FLOAT32, index_src, 3, poly_arg_none());
  /* Match Tinygrad's direct remove_bufferize probe. Full graph_rewrite removes
   * the nested LOCAL STAGE first and returns ADD in both implementations. */
  PolyUOp *ret = poly_pm_rewrite(poly_pm_remove_bufferize(), ctx, index);
  ASSERT_NOT_NULL(ret);
  ASSERT_INT_EQ(ret->op, POLY_OP_INDEX);
  ASSERT_INT_EQ(ret->src[0]->op, POLY_OP_STAGE);
  ASSERT_INT_EQ(poly_bufferize_arg_addrspace(ret->src[0]->arg), POLY_ADDR_LOCAL);
  ASSERT_INT_EQ(ret->src[0]->n_src, 2);
  PolyUOp *full = poly_graph_rewrite(ctx, index, poly_pm_remove_bufferize());
  ASSERT_NOT_NULL(full);
  ASSERT_INT_EQ(full->op, POLY_OP_ADD);
  ASSERT_INT_EQ(count_ops(ctx, full, POLY_OP_STAGE), 0);

  PolyUOp *p1_k0 = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, p1, k0, poly_arg_none());
  PolyUOp *all_partial_reduce_src[] = {p1_k0, reduce_range};
  PolyUOp *all_partial_reduce = poly_uop(
      ctx, POLY_OP_REDUCE, POLY_FLOAT32, all_partial_reduce_src, 2, poly_arg_reduce(POLY_OP_ADD, 0)
  );
  PolyUOp *all_partial_value = poly_alu2(ctx, POLY_OP_ADD, local_index, all_partial_reduce);
  PolyUOp *all_partial_stage_src[] = {all_partial_value, k0};
  PolyUOp *all_partial_stage = poly_uop(
      ctx, POLY_OP_STAGE, POLY_FLOAT32, all_partial_stage_src, 2,
      poly_arg_bufferize_opts("CPU", POLY_ADDR_GLOBAL, true)
  );
  PolyUOp *all_partial_index =
      poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, all_partial_stage, v0, poly_arg_none());
  ASSERT_TRUE(poly_pm_rewrite(poly_pm_remove_bufferize(), ctx, all_partial_index) == NULL);

  PolyUOp *q0 =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, bound128, poly_arg_range(25, POLY_AXIS_WEAK));
  PolyUOp *q1 =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, bound128, poly_arg_range(26, POLY_AXIS_WEAK));
  PolyUOp *q_reduce =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, bound4, poly_arg_range(27, POLY_AXIS_REDUCE));
  PolyUOp *reduce_replacement = poly_alu2(ctx, POLY_OP_ADD, q0, q_reduce);
  PolyUOp *p0_index = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, p0, k0, poly_arg_none());
  PolyUOp *replacement_reduce_src[] = {p0_index, reduce_range};
  PolyUOp *replacement_reduce = poly_uop(
      ctx, POLY_OP_REDUCE, POLY_FLOAT32, replacement_reduce_src, 2, poly_arg_reduce(POLY_OP_ADD, 0)
  );
  PolyUOp *replacement_value = poly_alu2(ctx, POLY_OP_ADD, replacement_reduce, p1_index);
  PolyUOp *replacement_stage_src[] = {replacement_value, k0, k1};
  PolyUOp *replacement_stage = poly_uop(
      ctx, POLY_OP_STAGE, POLY_FLOAT32, replacement_stage_src, 3,
      poly_arg_bufferize_opts("CPU", POLY_ADDR_GLOBAL, true)
  );
  PolyUOp *replacement_index_src[] = {replacement_stage, reduce_replacement, q1};
  PolyUOp *replacement_index =
      poly_uop(ctx, POLY_OP_INDEX, POLY_FLOAT32, replacement_index_src, 3, poly_arg_none());
  PolyUOp *replacement_ret = poly_pm_rewrite(poly_pm_remove_bufferize(), ctx, replacement_index);
  ASSERT_NOT_NULL(replacement_ret);
  ASSERT_INT_EQ(replacement_ret->op, POLY_OP_INDEX);
  ASSERT_INT_EQ(replacement_ret->src[0]->op, POLY_OP_STAGE);
  ASSERT_INT_EQ(poly_bufferize_arg_addrspace(replacement_ret->src[0]->arg), POLY_ADDR_LOCAL);
  poly_ctx_destroy(ctx);
  rangeify_restore_env(&pcontig);
  PASS();
}

TEST(rangeify, remove_bufferize_does_not_substitute_ended_effect_range) {
  /* Tinygrad rangeify.py:216-218 gates substitution by active ranges. The
   * range closed by END must not be replaced through AFTER's effect source. */
  RangeifyEnvSave pcontig = rangeify_save_env("PCONTIG");
  setenv("PCONTIG", "3", 1);
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *four = poly_const_int(ctx, 4);
  PolyUOp *k =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, four, poly_arg_range(30, POLY_AXIS_WEAK));
  PolyUOp *v =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, four, poly_arg_range(31, POLY_AXIS_WEAK));
  PolyParamArg arg = {
      .slot = 30, .dtype = POLY_FLOAT32, .addrspace = POLY_ADDR_GLOBAL, .device = "CPU"};
  PolyUOp *param = poly_uop1(ctx, POLY_OP_PARAM, POLY_FLOAT32, four, poly_arg_param(&arg));
  PolyUOp *target = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, param, k, poly_arg_none());
  PolyUOp *store =
      poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, target, poly_const_float(ctx, 1.0), poly_arg_none());
  PolyUOp *end_src[] = {store, k};
  PolyUOp *effect = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *after = poly_uop2(ctx, POLY_OP_AFTER, POLY_FLOAT32, param, effect, poly_arg_none());
  PolyUOp *stage_src[] = {after, k};
  PolyUOp *stage = poly_uop(
      ctx, POLY_OP_STAGE, POLY_FLOAT32, stage_src, 2,
      poly_arg_bufferize_opts("CPU", POLY_ADDR_GLOBAL, true)
  );
  PolyUOp *index = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, stage, v, poly_arg_none());
  PolyUOp *ret = poly_pm_rewrite(poly_pm_remove_bufferize(), ctx, index);
  ASSERT_PTR_EQ(ret, after);
  ASSERT_PTR_EQ(ret->src[1], effect);
  poly_ctx_destroy(ctx);
  rangeify_restore_env(&pcontig);
  PASS();
}

TEST(rangeify, range_prop_multi_consumer_same_idx_different_valid_fuses) {
  /* tinygrad's multi-consumer merge compares local idx separately from valid.
   * Two consumers that access the same local idx but carry different valid
   * masks should stay fused at default PCONTIG=0. */
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_test_buffer(ctx, POLY_FLOAT32, 4);
  PolyUOp *b = poly_test_buffer(ctx, POLY_FLOAT32, 4);
  PolyUOp *x = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, a, b, poly_arg_none());
  PolyUOp *c1 =
      poly_pad(ctx, poly_shrink(ctx, x, (int64_t[][2]){{0, 3}}, 1), (int64_t[][2]){{0, 2}}, 1);
  PolyUOp *c2 = poly_pad(ctx, x, (int64_t[][2]){{0, 1}}, 1);
  PolyUOp *y = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, c1, c2, poly_arg_none());
  PolyUOp *out = poly_test_buffer(ctx, POLY_FLOAT32, 5);
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
  ASSERT_EQ(poly_uop_get_idx(ctx, re_x->out_rngs[0])->op, POLY_OP_RANGE);
  ASSERT_EQ(poly_uop_get_valid(ctx, re_x->out_rngs[0])->op, POLY_OP_OR);

  PolyUOp *rangeified = run_apply_rangeify(ictx, sink);
  ASSERT_NOT_NULL(rangeified);

  ASSERT_FALSE(poly_is_realized(ictx, x));
  ASSERT_INT_EQ(count_ops(ctx, rangeified, POLY_OP_STAGE), 0);
  ASSERT_INT_EQ(count_ops(ctx, rangeified, POLY_OP_WHERE), 3);

  poly_indexing_ctx_destroy(ictx);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, range_prop_multi_consumer_same_idx_different_valid_fuses_2d) {
  /* Same rule as the 1D case above, but with a surviving leading axis.
   * This locks in that polygrad still fuses when only one axis carries
   * different valid masks and the local idx expressions stay identical. */
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a0 = poly_test_buffer(ctx, POLY_FLOAT32, 8);
  PolyUOp *b0 = poly_test_buffer(ctx, POLY_FLOAT32, 8);
  PolyUOp *a = poly_reshape(ctx, a0, (int64_t[]){2, 4}, 2);
  PolyUOp *b = poly_reshape(ctx, b0, (int64_t[]){2, 4}, 2);
  PolyUOp *x = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, a, b, poly_arg_none());
  PolyUOp *c1 = poly_pad(
      ctx, poly_shrink(ctx, x, (int64_t[][2]){{0, 2}, {0, 3}}, 2), (int64_t[][2]){{0, 0}, {0, 2}}, 2
  );
  PolyUOp *c2 = poly_pad(ctx, x, (int64_t[][2]){{0, 0}, {0, 1}}, 2);
  PolyUOp *y = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, c1, c2, poly_arg_none());
  PolyUOp *out0 = poly_test_buffer(ctx, POLY_FLOAT32, 10);
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
  ASSERT_EQ(poly_uop_get_idx(ctx, re_x->out_rngs[0])->op, POLY_OP_RANGE);
  ASSERT_EQ(poly_uop_get_idx(ctx, re_x->out_rngs[1])->op, POLY_OP_RANGE);
  ASSERT_EQ(poly_uop_get_valid(ctx, re_x->out_rngs[0])->op, POLY_OP_CONST);
  ASSERT_EQ(poly_uop_get_valid(ctx, re_x->out_rngs[1])->op, POLY_OP_OR);

  PolyUOp *rangeified = poly_apply_rangeify(ictx, sink);
  ASSERT_NOT_NULL(rangeified);
  ASSERT_FALSE(poly_is_realized(ictx, x));
  ASSERT_INT_EQ(count_ops(ctx, rangeified, POLY_OP_STAGE), 0);
  ASSERT_INT_EQ(count_ops(ctx, rangeified, POLY_OP_WHERE), 3);

  poly_indexing_ctx_destroy(ictx);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, range_prop_reshape) {
  /* reshape(a, (2,3)) where a has shape (6,):
   * STORE gets 2D ranges, reshape transforms to 1D input range */
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_test_buffer(ctx, POLY_FLOAT32, 6);
  PolyUOp *out = poly_reshape(ctx, poly_test_buffer(ctx, POLY_FLOAT32, 6), (int64_t[]){2, 3}, 2);
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
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(6));
  PolyUOp *range =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, bound, poly_arg_range(0, POLY_AXIS_LOOP));
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
  PolyUOp *reshape = poly_reshape(ctx, poly_buffer_f32(ctx, 32), (int64_t[]){1, 2, 4, 4}, 4);
  ASSERT_NOT_NULL(reshape);
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(1));
  PolyUOp *five = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(5));
  PolyUOp *truev = poly_uop0(ctx, POLY_OP_CONST, POLY_BOOL, poly_arg_bool(true));
  PolyUOp *invalid = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_invalid());
  PolyUOp *channel = poly_uop_range(ctx, 2, 0, POLY_AXIS_REDUCE);
  PolyUOp *row = poly_uop_range(ctx, 6, 1, POLY_AXIS_REDUCE);
  PolyUOp *col = poly_uop_range(ctx, 6, 2, POLY_AXIS_REDUCE);
  PolyUOp *padded[2] = {NULL, NULL};
  PolyUOp *spatial[2] = {row, col};
  for (int i = 0; i < 2; i++) {
    PolyUOp *below = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, spatial[i], one, poly_arg_none());
    PolyUOp *lower = poly_uop2(ctx, POLY_OP_CMPNE, POLY_BOOL, below, truev, poly_arg_none());
    PolyUOp *upper = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, spatial[i], five, poly_arg_none());
    PolyUOp *valid = poly_uop2(ctx, POLY_OP_AND, POLY_BOOL, lower, upper, poly_arg_none());
    PolyUOp *shifted = poly_uop2(
        ctx, POLY_OP_ADD, POLY_WEAKINT, spatial[i],
        poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(-1)), poly_arg_none()
    );
    padded[i] =
        poly_uop3(ctx, POLY_OP_WHERE, POLY_WEAKINT, valid, shifted, invalid, poly_arg_none());
  }
  PolyUOp *out_ranges[4] = {
      poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(0)),
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
  PolyUOp *n = poly_uop_variable(ctx, "n", 1, 8, POLY_WEAKINT, 1, false);
  PolyUOp *two = poly_const_int(ctx, 2);
  PolyUOp *input_shape_srcs[] = {two, n};
  PolyUOp *base = poly_reshape(ctx, poly_buffer_f32(ctx, 2), (int64_t[]){2, 1}, 2);
  PolyUOp *expanded = poly_expand_uop(ctx, base, input_shape_srcs, 2);
  PolyUOp *output_shape_srcs[] = {n, two};
  PolyUOp *output_shape =
      poly_uop(ctx, POLY_OP_STACK, POLY_WEAKINT, output_shape_srcs, 2, poly_arg_none());
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
  ASSERT_TRUE(count_ops(ctx, in_ranges[0], POLY_OP_BUFFER) > 0);
  ASSERT_TRUE(count_ops(ctx, in_ranges[1], POLY_OP_BUFFER) > 0);

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
  PolyUOp *n = poly_uop_variable(ctx, "n", 1, 8, POLY_INT32, 1, false);
  PolyUOp *y = poly_uop_variable(ctx, "y", 1, 8, POLY_INT32, 1, false);
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(1));
  PolyUOp *index_one = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(1));
  PolyUOp *two = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(2));
  PolyUOp *three = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(3));
  PolyUOp *five = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(5));
  PolyUOp *left[5] = {
      poly_binop(ctx, POLY_OP_ADD, n, n),
      poly_binop(ctx, POLY_OP_ADD, poly_binop(ctx, POLY_OP_ADD, n, one), two),
      poly_binop(ctx, POLY_OP_MUL, two, poly_binop(ctx, POLY_OP_ADD, n, one)),
      poly_binop(
          ctx, POLY_OP_ADD, poly_binop(ctx, POLY_OP_MUL, n, two),
          poly_binop(ctx, POLY_OP_MUL, n, three)
      ),
      poly_binop(ctx, POLY_OP_ADD, poly_binop(ctx, POLY_OP_ADD, y, n), n),
  };
  PolyUOp *right[5] = {
      poly_binop(ctx, POLY_OP_MUL, n, two),
      poly_binop(ctx, POLY_OP_ADD, n, three),
      poly_binop(ctx, POLY_OP_ADD, poly_binop(ctx, POLY_OP_MUL, n, two), two),
      poly_binop(ctx, POLY_OP_MUL, n, five),
      poly_binop(ctx, POLY_OP_ADD, y, poly_binop(ctx, POLY_OP_MUL, n, two)),
  };
  bool accepted[5] = {true, true, false, false, true};
  PolyUOp *base = poly_reshape(ctx, poly_buffer_f32(ctx, 3), (int64_t[]){3, 1}, 2);
  ASSERT_NOT_NULL(base);

  for (int i = 0; i < 5; i++) {
    PolyUOp *expand_shape_src[] = {three, left[i]};
    PolyUOp *expanded = poly_expand_uop(ctx, base, expand_shape_src, 2);
    PolyUOp *reshape_shape_src[] = {three, right[i]};
    PolyUOp *reshape_shape =
        poly_uop(ctx, POLY_OP_STACK, POLY_WEAKINT, reshape_shape_src, 2, poly_arg_none());
    PolyUOp *reshape_src[] = {expanded, reshape_shape};
    PolyUOp *reshaped =
        poly_uop(ctx, POLY_OP_RESHAPE, POLY_FLOAT32, reshape_src, 2, poly_arg_none());
    PolyUOp *out_ranges[] = {index_one};
    PolyUOp *in_ranges[POLY_MAX_DIMS] = {0};
    int n_in = -1;

    ASSERT_NOT_NULL(reshaped);
    if (accepted[i]) {
      ASSERT_INT_EQ(poly_uop_ndim(ctx, reshaped), 2);
      ASSERT_TRUE(poly_reshape_indices(ctx, reshaped, out_ranges, 1, in_ranges, &n_in));
      ASSERT_INT_EQ(n_in, 1);
      ASSERT_PTR_EQ(poly_graph_rewrite(ctx, in_ranges[0], poly_symbolic()), index_one);
    } else {
      ASSERT_INT_EQ(poly_uop_ndim(ctx, reshaped), -1);
      ASSERT_FALSE(poly_reshape_indices(ctx, reshaped, out_ranges, 1, in_ranges, &n_in));
    }
  }

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, range_prop_permute) {
  /* permute(a, [1,0]) where a has shape (3,4) → (4,3):
   * ranges should be reordered by inverse permutation */
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a_flat = poly_test_buffer(ctx, POLY_FLOAT32, 12);
  int64_t shape_2d[] = {3, 4};
  PolyUOp *a = poly_reshape(ctx, a_flat, shape_2d, 2);
  PolyUOp *out = poly_reshape(ctx, poly_test_buffer(ctx, POLY_FLOAT32, 12), (int64_t[]){4, 3}, 2);
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
  return u && (u->op == POLY_OP_BUFFER || u->op == POLY_OP_PARAM);
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
  int read_index;
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
    } else {
      c.read_index++;
    }
  }

  return c;
}

/* Run full rangeify pipeline up to and including apply_rangeify */
static PolyUOp *run_apply_rangeify(PolyIndexingCtx *ictx, PolyUOp *sink) {
  poly_realize_map_build(ictx, sink);
  poly_range_propagate(ictx, sink);
  return poly_apply_rangeify(ictx, sink);
}

/* Apply rangeify tests */

TEST(rangeify, deviceless_materialization_uses_sink_device) {
  /* Tinygrad 2026-08-22/a9069c177a9d indexing.py:147-150 binds an existing
   * global STAGE(device=None) to the enclosing physical sink device. Two
   * consumers make the device-free intermediate a real materialization. */
  PolyCtx *ctx = poly_ctx_new();
  int64_t shape[] = {4};
  PolyUOp *value = poly_expand(ctx, poly_const_float(ctx, 1.0), shape, 1);
  PolyUOp *shared = poly_uop1(ctx, POLY_OP_NEG, POLY_FLOAT32, value, poly_arg_none());
  PolyUOp *out0 = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CPU);
  PolyUOp *out1 = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CPU);
  PolyUOp *store0 = poly_store_val(
      ctx, out0, poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, shared, value, poly_arg_none())
  );
  PolyUOp *store1 = poly_store_val(
      ctx, out1, poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, shared, value, poly_arg_none())
  );
  PolyUOp *stores[] = {store0, store1};
  PolyUOp *sink = poly_uop(ctx, POLY_OP_SINK, POLY_VOID, stores, 2, poly_arg_none());
  PolyUOp *rangeified = poly_run_rangeify(ctx, sink, false);

  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, rangeified, &n_topo);
  int n_stage = 0;
  bool stages_are_cpu = true;
  for (int i = 0; i < n_topo; i++) {
    if (topo[i]->op != POLY_OP_STAGE) continue;
    n_stage++;
    const char *device = poly_bufferize_arg_device(topo[i]->arg);
    stages_are_cpu = stages_are_cpu && device && strcmp(device, "CPU") == 0;
  }
  poly_ctx_destroy(ctx);

  ASSERT_TRUE(n_stage > 0);
  ASSERT_TRUE(stages_are_cpu);
  PASS();
}

TEST(rangeify, apply_movement_removed) {
  /* reshape(a, (2,5)) + expand(b, (2,5)) → STORE → SINK
   * After apply_rangeify: no RESHAPE or EXPAND ops remain */
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_test_buffer(ctx, POLY_FLOAT32, 10);
  /* Current MovementMixin.expand rejects (10,) -> (2,5); use the valid
   * leading-axis broadcast (5,) -> (2,5). */
  PolyUOp *b = poly_test_buffer(ctx, POLY_FLOAT32, 5);
  PolyUOp *out = poly_test_buffer(ctx, POLY_FLOAT32, 10);
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

  PolyUOp *a = poly_test_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *out = poly_test_buffer(ctx, POLY_FLOAT32, 1);
  int64_t axes[] = {0};
  PolyUOp *reduce = poly_reduce_axis(ctx, POLY_OP_ADD, a, axes, 1);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out, reduce, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  /* Before: tensor REDUCE has (op, prefix-axis-count) and one value source. */
  ASSERT_INT_EQ(count_reduce_arg_kind(ctx, sink, POLY_ARG_REDUCE), 1);
  ASSERT_INT_EQ(count_reduce_arg_kind(ctx, sink, POLY_ARG_OPS), 0);
  ASSERT_INT_EQ(reduce->arg.reduce.num_axes, 1);

  PolyIndexingCtx *ictx = poly_indexing_ctx_new(ctx);
  PolyUOp *result = run_apply_rangeify(ictx, sink);
  ASSERT_NOT_NULL(result);

  /* After: lowered REDUCE retains the pair arg with zero tensor axes and
   * carries the concrete reduction RANGE sources. */
  ASSERT_INT_EQ(count_reduce_arg_kind(ctx, result, POLY_ARG_REDUCE), 1);
  ASSERT_INT_EQ(count_reduce_arg_kind(ctx, result, POLY_ARG_OPS), 0);

  /* Find the REDUCE node and verify it has range sources */
  int n;
  PolyUOp **topo = poly_toposort(ctx, result, &n);
  for (int i = 0; i < n; i++) {
    if (topo[i]->op == POLY_OP_REDUCE) {
      /* src[0] = value, src[1+] = reduce ranges */
      ASSERT_TRUE(topo[i]->n_src >= 2);
      ASSERT_INT_EQ(topo[i]->arg.reduce.op, POLY_OP_ADD);
      ASSERT_INT_EQ(topo[i]->arg.reduce.num_axes, 0);
      break;
    }
  }

  poly_indexing_ctx_destroy(ictx);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, apply_data_stack_to_selector_where) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  /* Current tinygrad schedule/indexing.py:121-129 converts a data STACK into
   * WHERE(selector == source_index, source, fallback). The STACK must not
   * survive as a scalar/vector constructor in the scheduled tensor kernel. */
  PolyUOp *a = poly_test_buffer(ctx, POLY_FLOAT32, 2);
  PolyUOp *b = poly_test_buffer(ctx, POLY_FLOAT32, 2);
  PolyUOp *values[] = {a, b};
  PolyUOp *stacked = poly_uop_stack(ctx, values, 2);
  ASSERT_NOT_NULL(stacked);
  PolyUOp *out = poly_reshape(ctx, poly_test_buffer(ctx, POLY_FLOAT32, 4), (int64_t[]){2, 2}, 2);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out, stacked, poly_arg_none());
  PolyUOp *sink = poly_sink1(ctx, store);

  PolyIndexingCtx *ictx = poly_indexing_ctx_new(ctx);
  PolyUOp *result = run_apply_rangeify(ictx, sink);
  ASSERT_NOT_NULL(result);
  ASSERT_INT_EQ(count_ops(ctx, result, POLY_OP_STACK), 0);
  ASSERT_INT_EQ(count_ops(ctx, result, POLY_OP_WHERE), 1);
  ASSERT_INT_EQ(count_ops(ctx, result, POLY_OP_CMPEQ), 1);

  PolyRangeEntry *re = poly_range_map_get(ictx, stacked);
  ASSERT_NOT_NULL(re);
  ASSERT_INT_EQ(re->n_out, 2);
  ASSERT_INT_EQ(re->n_in, 1);
  ASSERT_PTR_EQ(re->in_rngs[0], re->out_rngs[1]);

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
  PolyUOp *input = poly_reshape(ctx, flat, (int64_t[]){2, 128, 1, 2, 1, 2}, 6);
  PolyUOp *reshape = poly_reshape(ctx, input, (int64_t[]){2, 128, 1, 2, 1, 1, 2, 1}, 8);
  PolyUOp *r0 = poly_uop_range(ctx, 2, 0, POLY_AXIS_LOOP);
  PolyUOp *r1 = poly_uop_range(ctx, 128, 1, POLY_AXIS_LOOP);
  PolyUOp *spatial[2] = {
      poly_uop_range(ctx, 4, 2, POLY_AXIS_LOOP),
      poly_uop_range(ctx, 4, 3, POLY_AXIS_LOOP),
  };
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(0));
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(1));
  PolyUOp *two = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(2));
  PolyUOp *minus_one = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(-1));
  PolyUOp *invalid = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_invalid());
  PolyUOp *quotient[2] = {NULL, NULL};
  PolyUOp *gated[2] = {NULL, NULL};
  for (int i = 0; i < 2; i++) {
    PolyUOp *mod = poly_uop2(ctx, POLY_OP_FLOORMOD, POLY_WEAKINT, spatial[i], two, poly_arg_none());
    quotient[i] = poly_uop2(ctx, POLY_OP_FLOORDIV, POLY_WEAKINT, spatial[i], two, poly_arg_none());
    PolyUOp *valid = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, mod, one, poly_arg_none());
    PolyUOp *negative_zero =
        poly_uop2(ctx, POLY_OP_MUL, POLY_WEAKINT, zero, minus_one, poly_arg_none());
    PolyUOp *shifted =
        poly_uop2(ctx, POLY_OP_ADD, POLY_WEAKINT, mod, negative_zero, poly_arg_none());
    gated[i] =
        poly_uop3(ctx, POLY_OP_WHERE, POLY_WEAKINT, valid, shifted, invalid, poly_arg_none());
  }
  PolyUOp *out_ranges[8] = {r0, r1, zero, quotient[0], gated[0], zero, quotient[1], gated[1]};
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

  PolyUOp *a = poly_test_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *out = poly_test_buffer(ctx, POLY_FLOAT32, 12);
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

  /* Current rangeify keeps coordinate validity separate from value padding. */
  ASSERT_INT_EQ(count_ops(ctx, result, POLY_OP_PAD), 0);
  ASSERT_INT_EQ(count_ops(ctx, result, POLY_OP_WHERE), 2);

  poly_indexing_ctx_destroy(ictx);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, apply_pad_coordinates_preserve_invalid_like_tinygrad) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  /* Tinygrad 2026-08-22/a9069c177a9d schedule/indexing.py:131-140 returns
   * valid.where(r-off, Invalid), keeping address and validity separable. */
  PolyUOp *input = poly_buffer_f32(ctx, 3);
  int64_t pairs[1][2] = {{1, 2}};
  PolyUOp *current = poly_pad(ctx, input, pairs, 1);
  int64_t dims[1] = {3};
  PolyShape input_shape = {.dims = dims, .ndim = 1};
  PolyUOp *output_range = poly_uop_range(ctx, 6, 0, POLY_AXIS_LOOP);

  PolyUOp *input_ranges[POLY_MAX_DIMS] = {0};
  PolyUOp *valid = NULL;
  int n_input = 0;
  ASSERT_TRUE(poly_apply_movement_op(
      ctx, current, POLY_OP_PAD, input_shape, current->arg, &output_range, 1, input_ranges,
      &n_input, &valid
  ));
  ASSERT_INT_EQ(n_input, 1);
  ASSERT_NOT_NULL(valid);
  ASSERT_NOT_NULL(input_ranges[0]);
  ASSERT_INT_EQ(input_ranges[0]->op, POLY_OP_WHERE);
  ASSERT_INT_EQ(input_ranges[0]->n_src, 3);
  ASSERT_PTR_EQ(input_ranges[0]->src[0], valid);
  ASSERT_INT_EQ(input_ranges[0]->src[2]->op, POLY_OP_CONST);
  ASSERT_INT_EQ(input_ranges[0]->src[2]->arg.kind, POLY_ARG_INVALID);
  ASSERT_TRUE(poly_dtype_eq(input_ranges[0]->dtype, POLY_WEAKINT));
  ASSERT_TRUE(poly_dtype_eq(input_ranges[0]->src[2]->dtype, POLY_BOOL));

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
  int64_t dims[1] = {5};
  PolyShape input_shape = {.dims = dims, .ndim = 1};
  PolyUOp *output_range = poly_uop_range(ctx, 6, 0, POLY_AXIS_LOOP);

  PolyUOp *input_ranges[POLY_MAX_DIMS] = {0};
  PolyUOp *valid = NULL;
  int n_input = 0;
  ASSERT_TRUE(poly_apply_movement_op(
      ctx, current, POLY_OP_PAD, input_shape, current->arg, &output_range, 1, input_ranges,
      &n_input, &valid
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
  PolyShape input_shape = {.dims = (int64_t[]){5, 7}, .ndim = 2};
  PolyUOp *output_ranges[2] = {
      poly_uop_range(ctx, 5, 0, POLY_AXIS_LOOP),
      poly_uop_range(ctx, 8, 1, POLY_AXIS_LOOP),
  };

  PolyUOp *input_ranges[POLY_MAX_DIMS] = {0};
  PolyUOp *valid = NULL;
  int n_input = 0;
  ASSERT_TRUE(poly_apply_movement_op(
      ctx, current, POLY_OP_PAD, input_shape, current->arg, output_ranges, 2, input_ranges,
      &n_input, &valid
  ));
  ASSERT_INT_EQ(n_input, 2);
  ASSERT_PTR_EQ(input_ranges[0], output_ranges[0]);
  ASSERT_EQ(input_ranges[1]->op, POLY_OP_WHERE);
  ASSERT_NOT_NULL(valid);
  ASSERT_PTR_EQ(input_ranges[1]->src[0], valid);

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
  PolyShape input_shape = {.dims = (int64_t[]){2, 1}, .ndim = 2};
  PolyUOp *two = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(2));
  PolyUOp *r0 = poly_uop_range(ctx, 2, 0, POLY_AXIS_LOOP);
  PolyUOp *r1 = poly_uop_range(ctx, 2, 1, POLY_AXIS_LOOP);
  PolyUOp *old_valid = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, r0, two, poly_arg_none());
  PolyUOp *invalid = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_invalid());
  PolyUOp *output_ranges[2] = {
      poly_uop3(ctx, POLY_OP_WHERE, POLY_WEAKINT, old_valid, r0, invalid, poly_arg_none()),
      r1,
  };

  PolyUOp *input_ranges[POLY_MAX_DIMS] = {0};
  PolyUOp *valid = NULL;
  int n_input = 0;
  ASSERT_TRUE(poly_apply_movement_op(
      ctx, current, POLY_OP_PAD, input_shape, current->arg, output_ranges, 2, input_ranges,
      &n_input, &valid
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

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, index_projection_matches_pinned_get_idx_get_valid) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  /* Pinned uop/ops.py:571-581 projects Invalid-bearing WHERE coordinates,
   * recurses through STACK, and leaves ordinary integer WHERE values intact. */
  PolyUOp *idx0 = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(3));
  PolyUOp *idx1 = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(7));
  PolyUOp *concrete_i32 = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(5));
  PolyUOp *gate = poly_uop0(ctx, POLY_OP_PARAM, POLY_BOOL, poly_arg_int(0));
  PolyUOp *invalid = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_invalid());
  PolyUOp *gated =
      poly_uop3(ctx, POLY_OP_WHERE, POLY_WEAKINT, gate, idx0, invalid, poly_arg_none());
  PolyUOp *ordinary = poly_uop3(
      ctx, POLY_OP_WHERE, POLY_WEAKINT, gate, idx0,
      poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(0)), poly_arg_none()
  );
  PolyUOp *lanes[2] = {gated, idx1};
  PolyUOp *stack = poly_uop(ctx, POLY_OP_STACK, POLY_WEAKINT, lanes, 2, poly_arg_none());

  ASSERT_PTR_EQ(poly_uop_get_idx(ctx, gated), idx0);
  ASSERT_PTR_EQ(poly_uop_get_valid(ctx, gated), gate);
  ASSERT_PTR_EQ(poly_uop_get_idx(ctx, ordinary), ordinary);
  ASSERT_PTR_EQ(poly_uop_get_idx(ctx, concrete_i32), concrete_i32);
  PolyUOp *concrete_valid = poly_uop_get_valid(ctx, concrete_i32);
  ASSERT_INT_EQ(concrete_valid->op, POLY_OP_CONST);
  ASSERT_TRUE(concrete_valid->arg.kind == POLY_ARG_BOOL && concrete_valid->arg.b);
  PolyUOp *ordinary_valid = poly_uop_get_valid(ctx, ordinary);
  ASSERT_INT_EQ(ordinary_valid->op, POLY_OP_CONST);
  ASSERT_TRUE(ordinary_valid->arg.kind == POLY_ARG_BOOL && ordinary_valid->arg.b);
  PolyUOp *invalid_valid = poly_uop_get_valid(ctx, invalid);
  ASSERT_INT_EQ(invalid_valid->op, POLY_OP_CONST);
  ASSERT_TRUE(invalid_valid->arg.kind == POLY_ARG_BOOL && !invalid_valid->arg.b);

  PolyUOp *stack_idx = poly_uop_get_idx(ctx, stack);
  ASSERT_INT_EQ(stack_idx->op, POLY_OP_STACK);
  ASSERT_INT_EQ(stack_idx->n_src, 2);
  ASSERT_PTR_EQ(stack_idx->src[0], idx0);
  ASSERT_PTR_EQ(stack_idx->src[1], idx1);
  ASSERT_TRUE(poly_dtype_eq(stack_idx->dtype, POLY_WEAKINT));
  PolyUOp *stack_valid = poly_uop_get_valid(ctx, stack);
  ASSERT_INT_EQ(stack_valid->op, POLY_OP_STACK);
  ASSERT_INT_EQ(stack_valid->n_src, 2);
  ASSERT_PTR_EQ(stack_valid->src[0], gate);
  ASSERT_TRUE(stack_valid->src[1]->arg.kind == POLY_ARG_BOOL && stack_valid->src[1]->arg.b);
  ASSERT_TRUE(poly_dtype_eq(stack_valid->dtype, POLY_BOOL));

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
  PolyVarBinding *vars = NULL;
  int n_vars = 0;
  PolyUOp *linear = poly_linear_with_vars(ctx, &padded, 1, &realized, &vars, &n_vars);
  ASSERT_NOT_NULL(linear);
  ASSERT_INT_EQ(linear->n_src, 1);
  PolyUOp *body = poly_test_linear_call_body(linear, 0);
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

  free(vars);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, apply_elementwise_passthrough) {
  /* a + b → STORE → SINK: ALU ops pass through unchanged */
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_test_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *b = poly_test_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *out = poly_test_buffer(ctx, POLY_FLOAT32, 10);
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

  PolyUOp *a = poly_test_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *out = poly_test_buffer(ctx, POLY_FLOAT32, 10);
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

  PolyUOp *a = poly_test_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *b = poly_test_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *c = poly_test_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *out1 = poly_test_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *out2 = poly_test_buffer(ctx, POLY_FLOAT32, 10);

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

  PolyUOp *a = poly_test_buffer(ctx, POLY_FLOAT32, 4);
  PolyUOp *b = poly_test_buffer(ctx, POLY_FLOAT32, 4);
  PolyUOp *out = poly_test_buffer(ctx, POLY_FLOAT32, 4);
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, a, b, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out, add, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyIndexingCtx *ictx = poly_indexing_ctx_new(ctx);
  PolyUOp *result = run_apply_rangeify(ictx, sink);
  ASSERT_NOT_NULL(result);

  IndexKindCounts counts = count_index_kinds(ctx, result);
  ASSERT_INT_EQ(counts.store_target_index, 1);
  ASSERT_INT_EQ(counts.read_index, 2);

  poly_indexing_ctx_destroy(ictx);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, apply_triu_kernel_mode_read_indices_are_scalar) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *in = poly_buffer_f32(ctx, 9);
  PolyUOp *in2d = poly_reshape(ctx, in, (int64_t[]){3, 3}, 2);
  PolyUOp *tri = poly_triu(ctx, in2d, 0);
  PolyUOp *out = poly_reshape(ctx, poly_buffer_f32(ctx, 9), (int64_t[]){3, 3}, 2);
  PolyUOp *store = poly_store_val(ctx, out, tri);
  PolyUOp *sink = poly_sink1(ctx, store);

  PolyIndexingCtx *ictx = poly_indexing_ctx_new(ctx);
  PolyUOp *result = run_apply_rangeify(ictx, sink);
  ASSERT_NOT_NULL(result);

  IndexKindCounts counts = count_index_kinds(ctx, result);
  ASSERT_INT_EQ(counts.store_target_index, 1);
  ASSERT_TRUE(counts.read_index >= 1);

  poly_indexing_ctx_destroy(ictx);
  poly_ctx_destroy(ctx);
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

  PolyUOp *a = poly_test_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *b = poly_test_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *c = poly_test_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *out1 = poly_test_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *out2 = poly_test_buffer(ctx, POLY_FLOAT32, N);

  PolyUOp *d = poly_uop1(ctx, POLY_OP_NEG, POLY_FLOAT32, a, poly_arg_none());
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, d, b, poly_arg_none());
  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, d, c, poly_arg_none());
  PolyUOp *s1 = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out1, add, poly_arg_none());
  PolyUOp *s2 = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out2, mul, poly_arg_none());
  PolyUOp *stores[] = {s1, s2};
  PolyUOp *sink = poly_uop(ctx, POLY_OP_SINK, POLY_VOID, stores, 2, poly_arg_none());

  PolyUOp *scheduled_linear = poly_test_create_linear(ctx, sink);
  ASSERT_NOT_NULL(scheduled_linear);
  ASSERT_INT_EQ(scheduled_linear->n_src, 2); /* 2 fused kernels (d inlined) */

  /* All kernels should have SINK roots */
  for (int k = 0; k < scheduled_linear->n_src; k++)
    ASSERT_EQ(poly_test_linear_call_body(scheduled_linear, k)->op, POLY_OP_SINK);
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

  PolyUOp *a = poly_test_buffer_on_device(ctx, POLY_FLOAT32, N, POLY_DEVICE_CPU);
  PolyUOp *b = poly_test_buffer_on_device(ctx, POLY_FLOAT32, N, POLY_DEVICE_CPU);
  PolyUOp *c = poly_test_buffer_on_device(ctx, POLY_FLOAT32, N, POLY_DEVICE_CPU);
  PolyUOp *out1 = poly_test_buffer_on_device(ctx, POLY_FLOAT32, N, POLY_DEVICE_CPU);
  PolyUOp *out2 = poly_test_buffer_on_device(ctx, POLY_FLOAT32, N, POLY_DEVICE_CPU);

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

/* BUFFERIZE+INDEX structural tests */

TEST(rangeify, bufferize_same_size_dims_e2e) {
  /* Regression: 2x2 reduce-max → gradient requires BUFFERIZE with two
   * same-size dims (both size 2). Without structural INDEX wrapping, the
   * heuristic matcher can map both dims to the same context RANGE. */
  float x_d[4] = {1.0f, 3.0f, 2.0f, 4.0f};
  float gx_d[4] = {0};

  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CPU);
  int64_t shape[] = {2, 2};
  PolyUOp *xr = poly_reshape(ctx, x, shape, 2);
  int64_t ax[] = {1};
  PolyUOp *m = poly_reduce_axis(ctx, POLY_OP_MAX, xr, ax, 1);
  int64_t ax2[] = {0};
  PolyUOp *loss = poly_reduce_axis(ctx, POLY_OP_ADD, m, ax2, 1);
  PolyUOp *gx = poly_grad(ctx, loss, xr);
  ASSERT_NOT_NULL(gx);

  PolyUOp *out = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CPU);
  PolyUOp *store = poly_store_val(ctx, poly_reshape(ctx, out, shape, 2), gx);
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

  PolyUOp *a = poly_test_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *b = poly_test_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *c = poly_test_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *out1 = poly_test_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *out2 = poly_test_buffer(ctx, POLY_FLOAT32, N);

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

  PolyUOp *linear = poly_test_create_linear(ctx, sink);
  ASSERT_NOT_NULL(linear);
  ASSERT_TRUE(linear->n_src > 1);

  int bad = 0;
  for (int i = 0; i < linear->n_src; i++)
    bad += count_alu_direct_storage_sources(ctx, poly_test_linear_call_body(linear, i));
  ASSERT_INT_EQ(bad, 0);

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

  PolyUOp *a_flat = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 9, POLY_DEVICE_CPU);
  PolyUOp *b_flat = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 9, POLY_DEVICE_CPU);
  PolyUOp *c_flat = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 9, POLY_DEVICE_CPU);
  PolyUOp *out1_flat = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 9, POLY_DEVICE_CPU);
  PolyUOp *out2_flat = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 9, POLY_DEVICE_CPU);

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
  int ret = poly_test_realize_buffer_views(ctx, sink, bindings, 5);
  ASSERT_INT_EQ(ret, 0);

  for (int i = 0; i < 9; i++) {
    float neg_a = -a_d[i];
    ASSERT_FLOAT_EQ(o1_d[i], neg_a + b_d[i], 1e-5);
    ASSERT_FLOAT_EQ(o2_d[i], neg_a * c_d[i], 1e-5);
  }

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, bufferize_movement_chain_e2e) {
  /* Current rangeify must preserve two shifted PAD/SHRINK consumers through
   * one broadcast and reduction without any auxiliary range representation. */
  PolyCtx *ctx = poly_ctx_new();

  /* x: (1,3,5,5) flattened */
  PolyUOp *x_flat = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 75, POLY_DEVICE_CPU);
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

  PolyUOp *out = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 1, POLY_DEVICE_CPU);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out, loss, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  float x_d[75], o_d[1];
  for (int i = 0; i < 75; i++)
    x_d[i] = (float)(i + 1);

  PolyTestBufferView bindings[] = {
      POLY_TEST_HOST_VIEW(out, o_d),
      POLY_TEST_HOST_VIEW(x_flat, x_d),
  };

  int ret = poly_test_realize_buffer_views(ctx, sink, bindings, 2);
  ASSERT_INT_EQ(ret, 0);

  /* Expected from numpy reference:
   * (broadcast(s1,(1,2,5,5)) + broadcast(s2,(1,2,5,5))).sum(axis=(1,2,3)) */
  ASSERT_FLOAT_EQ(o_d[0], 740.0f, 1e-5);

  poly_ctx_destroy(ctx);
  PASS();
}

/* add_buffers tests */

TEST(rangeify, add_buffers_noop_single_kernel) {
  /* Simple vecadd: c = a + b. No BUFFERIZE nodes after apply_rangeify,
   * so add_buffers should be a no-op (no BUFFER/AFTER/END introduced). */
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_test_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *b = poly_test_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *c = poly_test_buffer(ctx, POLY_FLOAT32, 10);
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, a, b, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, c, add, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyIndexingCtx *ictx = poly_indexing_ctx_new(ctx);
  PolyUOp *rangeified = run_apply_rangeify(ictx, sink);
  ASSERT_NOT_NULL(rangeified);

  /* No BUFFERIZE should exist (single kernel, no multi-consumer divergence) */
  ASSERT_INT_EQ(count_ops(ctx, rangeified, POLY_OP_STAGE), 0);

  /* Apply add_buffers — should be a no-op */
  PolyUOp *result = apply_add_buffers(ctx, rangeified);
  ASSERT_NOT_NULL(result);

  /* No new BUFFER, AFTER, or END nodes introduced */
  ASSERT_INT_EQ(count_ops(ctx, result, POLY_OP_AFTER), 0);

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

  PolyUOp *a = poly_test_buffer_on_device(ctx, POLY_FLOAT32, N, POLY_DEVICE_CPU);
  PolyUOp *b = poly_test_buffer_on_device(ctx, POLY_FLOAT32, N, POLY_DEVICE_CPU);
  PolyUOp *c = poly_test_buffer_on_device(ctx, POLY_FLOAT32, N, POLY_DEVICE_CPU);
  PolyUOp *out1 = poly_test_buffer_on_device(ctx, POLY_FLOAT32, N, POLY_DEVICE_CPU);
  PolyUOp *out2 = poly_test_buffer_on_device(ctx, POLY_FLOAT32, N, POLY_DEVICE_CPU);

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
  PolyUOp *result = apply_add_buffers(ctx, rangeified);
  ASSERT_NOT_NULL(result);

  /* No BUFFERIZE remains */
  ASSERT_INT_EQ(count_ops(ctx, result, POLY_OP_STAGE), 0);

  /* Current bufferize_to_store creates BUFFER(shape, ParamArg). */
  ASSERT_TRUE(count_ops(ctx, result, POLY_OP_AFTER) >= 1);
  ASSERT_TRUE(count_ops(ctx, result, POLY_OP_END) >= 1);

  /* Original 2 consumer stores still present, plus 1 producer store */
  ASSERT_INT_EQ(count_ops(ctx, result, POLY_OP_STORE), 3);

  /* Verify the current ParamArg BUFFER has the flattened size. */
  int n_topo;
  PolyUOp **topo = poly_toposort(ctx, result, &n_topo);
  int found_buf = 0;
  for (int i = 0; i < n_topo; i++) {
    if (topo[i]->op == POLY_OP_BUFFER && topo[i]->n_src == 1 &&
        topo[i]->arg.kind == POLY_ARG_PARAM) {
      ASSERT_INT_EQ(topo[i]->src[0]->op, POLY_OP_CONST);
      ASSERT_INT_EQ(topo[i]->src[0]->arg.i, N);
      ASSERT_INT_EQ(topo[i]->arg.param->addrspace, POLY_ADDR_GLOBAL);
      found_buf = 1;
    }
  }
  ASSERT_TRUE(found_buf);

  poly_indexing_ctx_destroy(ictx);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, add_buffers_commits_storage_before_indexing_weak_value) {
  /* Tinygrad 2026-08-22/a9069c177a9d rangeify.py:392-452 stores weak values
   * strongly, indexes that storage as the strong dtype, then restores weak. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *source = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CPU);
  PolyUOp *four = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(4));
  PolyUOp *range =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, four, poly_arg_range(0, POLY_AXIS_WEAK));
  PolyUOp *value_index =
      poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, source, range, poly_arg_none());
  PolyUOp *weak_value =
      poly_uop1(ctx, POLY_OP_CAST, POLY_WEAKFLOAT, value_index, poly_arg_dtype(POLY_WEAKFLOAT));
  PolyUOp *stage_src[] = {weak_value, range};
  PolyUOp *stage = poly_uop(
      ctx, POLY_OP_STAGE, POLY_WEAKFLOAT, stage_src, 2,
      poly_arg_bufferize_opts("CPU", POLY_ADDR_GLOBAL, false)
  );
  PolyUOp *consumer = poly_uop2(ctx, POLY_OP_INDEX, POLY_WEAKFLOAT, stage, range, poly_arg_none());

  PolyUOp *result = apply_add_buffers(ctx, consumer);
  ASSERT_NOT_NULL(result);
  ASSERT_INT_EQ(result->op, POLY_OP_CAST);
  ASSERT_TRUE(poly_dtype_eq(result->dtype, POLY_WEAKFLOAT));
  ASSERT_INT_EQ(result->src[0]->op, POLY_OP_INDEX);
  ASSERT_TRUE(poly_dtype_eq(result->src[0]->dtype, POLY_FLOAT32));
  ASSERT_INT_EQ(result->src[0]->src[0]->op, POLY_OP_AFTER);
  ASSERT_TRUE(poly_dtype_eq(result->src[0]->src[0]->dtype, POLY_FLOAT32));

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, add_buffers_flattens_multirange_stage) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(1));
  PolyParamArg input_arg = {.slot = 7, .addrspace = POLY_ADDR_GLOBAL, .device = "CPU"};
  PolyUOp *input = poly_uop1(ctx, POLY_OP_PARAM, POLY_FLOAT32, one, poly_arg_param(&input_arg));
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(0));
  PolyUOp *value = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, input, zero, poly_arg_none());
  PolyUOp *two = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(2));
  PolyUOp *r0 =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, two, poly_arg_range(11, POLY_AXIS_WEAK));
  PolyUOp *r1 =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, two, poly_arg_range(12, POLY_AXIS_WEAK));
  PolyUOp *stage_src[3] = {value, r0, r1};
  PolyUOp *stage = poly_uop(
      ctx, POLY_OP_STAGE, POLY_FLOAT32, stage_src, 3,
      poly_arg_bufferize_opts("CPU", POLY_ADDR_GLOBAL, true)
  );

  PolyUOp *result = apply_add_buffers(ctx, stage);
  ASSERT_NOT_NULL(result);
  ASSERT_INT_EQ(result->op, POLY_OP_RESHAPE);
  ASSERT_INT_EQ(count_ops(ctx, result, POLY_OP_STAGE), 0);
  ASSERT_INT_EQ(count_ops(ctx, result, POLY_OP_BUFFER), 1);
  ASSERT_INT_EQ(count_ops(ctx, result, POLY_OP_STORE), 1);
  ASSERT_INT_EQ(count_ops(ctx, result, POLY_OP_END), 1);
  ASSERT_INT_EQ(count_ops(ctx, result, POLY_OP_AFTER), 1);

  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, result, &n_topo);
  ASSERT_NOT_NULL(topo);
  PolyUOp *buffer = NULL;
  for (int i = 0; i < n_topo; i++)
    if (topo[i]->op == POLY_OP_BUFFER) buffer = topo[i];
  ASSERT_NOT_NULL(buffer);
  ASSERT_INT_EQ(buffer->n_src, 1);
  ASSERT_INT_EQ(buffer->src[0]->op, POLY_OP_CONST);
  ASSERT_INT_EQ(buffer->src[0]->arg.i, 4);
  ASSERT_INT_EQ(buffer->arg.kind, POLY_ARG_PARAM);
  ASSERT_INT_EQ(buffer->arg.param->slot, 0);
  ASSERT_STR_EQ(buffer->arg.param->device, "CPU");

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, add_buffers_uses_bufferize_device_metadata) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *a = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 8, POLY_DEVICE_CPU);
  PolyUOp *device = poly_device_uop_from_name(ctx, "CPU:1");
  PolyUOp *copy = poly_copy_to_device_uop(ctx, a, device);
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

  PolyUOp *result = apply_add_buffers(ctx, bufferize);
  ASSERT_NOT_NULL(result);

  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, result, &n_topo);
  ASSERT_NOT_NULL(topo);
  int target_buffers = 0, source_buffers = 0;
  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    if (u->op != POLY_OP_BUFFER || u->n_src != 1 || u->arg.kind != POLY_ARG_PARAM) continue;
    if (strcmp(u->arg.param->device, "CPU:1") == 0)
      target_buffers++;
    else if (strcmp(u->arg.param->device, "CPU") == 0)
      source_buffers++;
    else
      FAIL("unexpected BUFFER device %s", u->arg.param->device);
  }
  /* Tinygrad 2026-08-22 a9069c17 bufferize_to_store uses the STAGE device
   * only for the new output BUFFER; its source BUFFER stays on CPU. */
  ASSERT_INT_EQ(target_buffers, 1);
  ASSERT_INT_EQ(source_buffers, 1);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, earliest_copy_equality_uses_exact_device_identity) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  /* Pinned earliest_rewrites compares exact UOp.device strings
   * (rangeify.py:184-187): CPU->CPU is NOOP, while CPU->CPU:1 remains COPY. */
  PolyUOp *source = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CPU);
  PolyUOp *cpu = poly_device_uop(ctx, POLY_DEVICE_CPU);
  PolyUOp *cpu1 = poly_device_uop_from_name(ctx, "CPU:1");
  PolyUOp *same = poly_copy_to_device_uop(ctx, source, cpu);
  PolyUOp *different = poly_copy_to_device_uop(ctx, source, cpu1);

  /* Current earliest_rewrites removes same-device COPY directly. */
  const char *names[2] = {"CPU", "CPU:1"};
  PolyUOp *tuple = poly_device_uop_from_names(ctx, names, 2);
  PolyUOp *tuple_buffer = poly_uop_new_buffer(ctx, tuple, 4, POLY_FLOAT32, 7301);
  PolyUOp *selected = poly_uop1(ctx, POLY_OP_MSELECT, POLY_FLOAT32, tuple_buffer, poly_arg_int(0));
  PolyUOp *selected_copy = poly_copy_to_device_uop(ctx, selected, cpu);

  PolyUOp *same_result = poly_apply_earliest_rewrites(ctx, poly_sink1(ctx, same));
  PolyUOp *different_result = poly_apply_earliest_rewrites(ctx, poly_sink1(ctx, different));
  PolyUOp *selected_result = poly_apply_earliest_rewrites(ctx, poly_sink1(ctx, selected_copy));
  ASSERT_NOT_NULL(same_result);
  ASSERT_NOT_NULL(different_result);
  ASSERT_NOT_NULL(selected_result);
  ASSERT_INT_EQ(same_result->n_src, 1);
  ASSERT_INT_EQ(different_result->n_src, 1);
  ASSERT_PTR_EQ(same_result->src[0], source);
  ASSERT_PTR_EQ(different_result->src[0], different);
  ASSERT_STR_EQ(poly_uop_device_name(ctx, different_result->src[0]), "CPU:1");
  ASSERT_INT_EQ(selected_result->n_src, 1);
  ASSERT_PTR_EQ(selected_result->src[0], selected);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, earliest_copy_materializes_reordered_movement) {
  /* Current schedule/rangeify.py:147-150 materializes reordered COPY input
   * even when the movement and its base have equal element counts. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *source = poly_reshape(
      ctx, poly_test_buffer_on_device(ctx, POLY_FLOAT32, 6, POLY_DEVICE_CPU), (int64_t[]){2, 3}, 2
  );
  PolyUOp *permuted = poly_permute(ctx, source, (int64_t[]){1, 0}, 2);
  PolyUOp *copy = poly_copy_to_device_uop(ctx, permuted, poly_device_uop_from_name(ctx, "PYTHON"));
  PolyUOp *rewritten = poly_apply_earliest_rewrites(ctx, poly_sink1(ctx, copy));

  ASSERT_NOT_NULL(rewritten);
  ASSERT_INT_EQ(count_ops(ctx, rewritten, POLY_OP_COPY), 1);
  ASSERT_INT_EQ(count_ops(ctx, rewritten, POLY_OP_CONTIGUOUS), 1);
  PolyUOp *rewritten_copy = rewritten->src[0];
  ASSERT_EQ(rewritten_copy->op, POLY_OP_COPY);
  ASSERT_EQ(rewritten_copy->src[0]->op, POLY_OP_CONTIGUOUS);
  ASSERT_PTR_EQ(rewritten_copy->src[0]->src[0], permuted);
  poly_ctx_destroy(ctx);
  PASS();
}

static PolyUOp *multi_pm_test_source(
    PolyCtx *ctx,
    int slot,
    PolyDType dtype,
    const char *device_name,
    int64_t rows,
    int64_t cols
) {
  PolyUOp *device = poly_device_uop_from_name(ctx, device_name);
  PolyUOp *buffer = poly_uop_new_buffer(ctx, device, rows * cols, dtype, slot);
  int64_t shape[2] = {rows, cols};
  return buffer ? poly_reshape(ctx, buffer, shape, 2) : NULL;
}

static PolyUOp *multi_pm_test_copy_tuple(PolyCtx *ctx, int unique_id, PolyDType dtype) {
  const char *names[2] = {"CPU", "CPU:1"};
  PolyUOp *tuple = poly_device_uop_from_names(ctx, names, 2);
  PolyUOp *source = multi_pm_test_source(ctx, unique_id, dtype, "CPU", 4, 4);
  return source && tuple ? poly_copy_to_device_uop(ctx, source, tuple) : NULL;
}

static PolyUOp *multi_pm_test_unshard(PolyCtx *ctx, PolyUOp *local, int axis, int count) {
  PolyUOp *range = poly_range(ctx, count, -1, POLY_AXIS_DEVICE);
  int64_t axis_value = axis;
  PolyUOp *ranges[] = {range};
  return local && range ? poly_unshard(ctx, local, &axis_value, ranges, 1) : NULL;
}

static PolyUOp *multi_pm_test_shard(PolyCtx *ctx, int unique_id, PolyDType dtype, int axis) {
  if (axis < 0 || axis > 1) return NULL;
  PolyUOp *copy = multi_pm_test_copy_tuple(ctx, unique_id, dtype);
  PolyUOp *device_num = poly_range(ctx, 2, -1, POLY_AXIS_DEVICE);
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(0));
  PolyUOp *two = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(2));
  PolyUOp *four = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(4));
  PolyUOp *start = poly_alu2(ctx, POLY_OP_MUL, device_num, two);
  PolyUOp *starts[2] = {axis == 0 ? start : zero, axis == 1 ? start : zero};
  PolyUOp *sizes[2] = {axis == 0 ? two : four, axis == 1 ? two : four};
  PolyUOp *local = copy ? poly_shrink_uop(ctx, copy, starts, sizes, 2) : NULL;
  return multi_pm_test_unshard(ctx, local, axis, 2);
}

static PolyUOp *multi_pm_test_axis0_column_shard(PolyCtx *ctx, int unique_id, PolyDType dtype) {
  const char *names[2] = {"CPU", "CPU:1"};
  PolyUOp *tuple = poly_device_uop_from_names(ctx, names, 2);
  PolyUOp *source = multi_pm_test_source(ctx, unique_id, dtype, "CPU", 4, 1);
  PolyUOp *copy = source && tuple ? poly_copy_to_device_uop(ctx, source, tuple) : NULL;
  PolyUOp *device_num = poly_range(ctx, 2, -1, POLY_AXIS_DEVICE);
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(0));
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(1));
  PolyUOp *two = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(2));
  PolyUOp *starts[2] = {poly_alu2(ctx, POLY_OP_MUL, device_num, two), zero};
  PolyUOp *sizes[2] = {two, one};
  PolyUOp *local = copy ? poly_shrink_uop(ctx, copy, starts, sizes, 2) : NULL;
  return multi_pm_test_unshard(ctx, local, 0, 2);
}

static PolyUOp *multi_pm_test_param(
    PolyCtx *ctx,
    int slot,
    const char *device,
    const char **devices,
    int n_devices,
    bool has_axis
) {
  PolyUOp *dims[2] = {poly_const_int(ctx, has_axis ? 4 : 2), poly_const_int(ctx, 4)};
  PolyUOp *shape = poly_uop(ctx, POLY_OP_STACK, POLY_WEAKINT, dims, 2, poly_arg_none());
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
    PolyUOp *device = poly_device_uop_from_name(ctx, devices[lane]);
    PolyUOp *buffer = poly_uop_new_buffer(ctx, device, 8, POLY_FLOAT32, unique_base + lane);
    if (!buffer || poly_buffer_allocate(ctx, buffer, POLY_DEVICE_CPU) != 0 ||
        poly_buffer_copyin(ctx, buffer, values + lane * 8, 8 * sizeof(float)) != 0)
      return NULL;
    locals[lane] = poly_reshape(ctx, buffer, (int64_t[]){2, 4}, 2);
    if (!locals[lane]) return NULL;
  }
  PolyUOp *stack = poly_uop(ctx, POLY_OP_MSTACK, POLY_FLOAT32, locals, 2, poly_arg_none());
  return multi_pm_test_unshard(ctx, stack, 0, 2);
}

static int multi_pm_test_node_count(PolyCtx *ctx, PolyUOp *root) {
  int n = 0;
  PolyUOp **topo = poly_toposort_alloc(ctx, root, &n);
  bool ok = topo != NULL;
  poly_toposort_free(topo);
  return ok ? n : -1;
}

static PolyUOp *multi_pm_test_find_named_call(PolyCtx *ctx, PolyUOp *root, const char *name) {
  int n = 0;
  PolyUOp **topo = poly_toposort_ex_alloc(ctx, root, &n, NULL, true);
  PolyUOp *found = NULL;
  for (int i = 0; topo && i < n; i++) {
    PolyUOp *u = topo[i];
    if (u->op == POLY_OP_CALL && u->arg.kind == POLY_ARG_STRING && strcmp(u->arg.str, name) == 0) {
      found = u;
      break;
    }
  }
  poly_toposort_free(topo);
  return found;
}

static PolyUOp *multi_pm_test_find_buffer_slot(PolyCtx *ctx, PolyUOp *root, int64_t slot) {
  int n = 0;
  PolyUOp **topo = poly_toposort_ex_alloc(ctx, root, &n, NULL, true);
  PolyUOp *found = NULL;
  for (int i = 0; topo && i < n; i++) {
    PolyUOp *u = topo[i];
    if (u->op == POLY_OP_BUFFER && u->arg.kind == POLY_ARG_PARAM && u->arg.param &&
        u->arg.param->slot == slot) {
      found = u;
      break;
    }
  }
  poly_toposort_free(topo);
  return found;
}

TEST(rangeify, add_buffers_removes_invalid_clone_initialization_like_pinned) {
  /* Current pm_add_buffers turns STORE(dst, Invalid) into zero-source NOOP,
   * then removes that effect from AFTER (rangeify.py:460-463). The raw buffer
   * identity remains as SINK(BUFFER(CONST(4))) and schedules no kernel. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  const char *names[2] = {"CPU", "CPU:1"};
  PolyUOp *tuple = poly_device_uop_from_names(ctx, names, 2);
  PolyUOp *buffer = poly_uop_new_buffer(ctx, tuple, 4, POLY_FLOAT32, 9101);
  PolyUOp *invalid = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_invalid());
  PolyUOp *invalid_shaped = poly_reshape(ctx, invalid, (int64_t[]){1}, 1);
  invalid_shaped = poly_expand(ctx, invalid_shaped, (int64_t[]){4}, 1);
  PolyUOp *store =
      poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, buffer, invalid_shaped, poly_arg_none());
  PolyUOp *after = poly_uop2(ctx, POLY_OP_AFTER, POLY_FLOAT32, buffer, store, poly_arg_none());
  PolyUOp *kernel_graph = poly_get_kernel_graph(ctx, poly_sink1(ctx, after));
  ASSERT_NOT_NULL(kernel_graph);
  ASSERT_INT_EQ(multi_pm_test_node_count(ctx, kernel_graph), 3);
  ASSERT_INT_EQ(count_ops(ctx, kernel_graph, POLY_OP_BUFFER), 1);
  ASSERT_INT_EQ(count_ops(ctx, kernel_graph, POLY_OP_AFTER), 0);
  ASSERT_INT_EQ(count_ops(ctx, kernel_graph, POLY_OP_STORE), 0);
  ASSERT_INT_EQ(count_ops(ctx, kernel_graph, POLY_OP_NOOP), 0);

  PolyUOp *linear = poly_create_schedule(ctx, kernel_graph);
  ASSERT_NOT_NULL(linear);
  ASSERT_INT_EQ(linear->n_src, 0);
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
      multi_pm_test_shard(ctx, 902, POLY_FLOAT32, 0)
  );
  PolyUOp *same_result = poly_apply_multi_pm(ctx, same);
  ASSERT_NOT_NULL(same_result);
  ASSERT_INT_EQ(same_result->op, POLY_OP_UNSHARD);
  ASSERT_INT_EQ(same_result->arg.kind, POLY_ARG_INT_TUPLE);
  ASSERT_INT_EQ(same_result->arg.int_tuple.n, 1);
  ASSERT_INT_EQ(same_result->arg.int_tuple.vals[0], 0);
  ASSERT_INT_EQ(same_result->src[0]->op, POLY_OP_ADD);
  ASSERT_INT_EQ(same_result->src[0]->src[0]->op, POLY_OP_MSTACK);
  ASSERT_INT_EQ(same_result->src[0]->src[1]->op, POLY_OP_MSTACK);
  ASSERT_INT_EQ(count_ops(ctx, same_result, POLY_OP_UNSHARD), 1);
  ASSERT_INT_EQ(count_ops(ctx, same_result, POLY_OP_MSTACK), 2);
  ASSERT_INT_EQ(count_ops(ctx, same_result, POLY_OP_COPY), 4);
  ASSERT_INT_EQ(count_ops(ctx, same_result, POLY_OP_SHRINK), 4);
  ASSERT_INT_EQ(multi_pm_test_node_count(ctx, same_result), 25);

  PolyUOp *unsharded = poly_alu2(
      ctx, POLY_OP_ADD, multi_pm_test_shard(ctx, 903, POLY_FLOAT32, 0),
      multi_pm_test_copy_tuple(ctx, 904, POLY_FLOAT32)
  );
  PolyUOp *unsharded_result = poly_apply_multi_pm(ctx, unsharded);
  ASSERT_NOT_NULL(unsharded_result);
  ASSERT_INT_EQ(unsharded_result->op, POLY_OP_UNSHARD);
  ASSERT_INT_EQ(unsharded_result->arg.int_tuple.vals[0], 0);
  ASSERT_INT_EQ(unsharded_result->src[0]->op, POLY_OP_ADD);
  ASSERT_INT_EQ(unsharded_result->src[0]->src[0]->op, POLY_OP_MSTACK);
  ASSERT_INT_EQ(unsharded_result->src[0]->src[1]->op, POLY_OP_MSTACK);
  ASSERT_INT_EQ(count_ops(ctx, unsharded_result, POLY_OP_UNSHARD), 1);
  ASSERT_INT_EQ(count_ops(ctx, unsharded_result, POLY_OP_MSTACK), 2);
  ASSERT_INT_EQ(multi_pm_test_node_count(ctx, unsharded_result), 25);

  PolyUOp *mismatch = poly_alu2(
      ctx, POLY_OP_ADD, multi_pm_test_shard(ctx, 905, POLY_FLOAT32, 0),
      multi_pm_test_shard(ctx, 906, POLY_FLOAT32, 1)
  );
  PolyUOp *mismatch_result = poly_apply_multi_pm(ctx, mismatch);
  ASSERT_NOT_NULL(mismatch_result);
  ASSERT_INT_EQ(mismatch_result->op, POLY_OP_UNSHARD);
  ASSERT_INT_EQ(mismatch_result->arg.int_tuple.vals[0], 1);
  PolyUOp *mismatch_add = mismatch_result->src[0];
  ASSERT_INT_EQ(mismatch_add->op, POLY_OP_ADD);
  ASSERT_INT_EQ(mismatch_add->src[0]->op, POLY_OP_SHRINK);
  ASSERT_INT_EQ(mismatch_add->src[0]->src[0]->op, POLY_OP_ALLREDUCE);
  ASSERT_INT_EQ(mismatch_add->src[0]->src[0]->src[0]->op, POLY_OP_PAD);
  ASSERT_INT_EQ(mismatch_add->src[0]->src[0]->src[0]->src[0]->op, POLY_OP_MSTACK);
  ASSERT_INT_EQ(mismatch_add->src[1]->op, POLY_OP_MSTACK);
  ASSERT_INT_EQ(count_ops(ctx, mismatch_result, POLY_OP_UNSHARD), 1);
  ASSERT_INT_EQ(count_ops(ctx, mismatch_result, POLY_OP_ALLREDUCE), 1);
  ASSERT_INT_EQ(count_ops(ctx, mismatch_result, POLY_OP_PAD), 1);
  ASSERT_INT_EQ(count_ops(ctx, mismatch_result, POLY_OP_MSTACK), 2);
  ASSERT_INT_EQ(multi_pm_test_node_count(ctx, mismatch_result), 33);

  int64_t axis0[1] = {0}, axis1[1] = {1};
  PolyUOp *reduce_shard =
      poly_reduce_axis(ctx, POLY_OP_ADD, multi_pm_test_shard(ctx, 907, POLY_FLOAT32, 0), axis0, 1);
  PolyUOp *reduce_shard_result = poly_apply_multi_pm(ctx, reduce_shard);
  ASSERT_NOT_NULL(reduce_shard_result);
  ASSERT_INT_EQ(reduce_shard_result->op, POLY_OP_ALLREDUCE);
  ASSERT_INT_EQ(reduce_shard_result->arg.kind, POLY_ARG_ALLREDUCE);
  ASSERT_INT_EQ(reduce_shard_result->arg.allreduce.op, POLY_OP_ADD);
  ASSERT_INT_EQ(reduce_shard_result->src[0]->op, POLY_OP_REDUCE);
  ASSERT_INT_EQ(reduce_shard_result->src[0]->src[0]->op, POLY_OP_MSTACK);
  ASSERT_TRUE(reduce_shard_result->arg.allreduce.device_is_tuple);
  ASSERT_INT_EQ(reduce_shard_result->arg.allreduce.n_devices, 2);
  ASSERT_INT_EQ(count_ops(ctx, reduce_shard_result, POLY_OP_UNSHARD), 0);
  ASSERT_INT_EQ(multi_pm_test_node_count(ctx, reduce_shard_result), 17);

  PolyUOp *reduce_other =
      poly_reduce_axis(ctx, POLY_OP_ADD, multi_pm_test_shard(ctx, 908, POLY_FLOAT32, 0), axis1, 1);
  PolyUOp *reduce_other_result = poly_apply_multi_pm(ctx, reduce_other);
  ASSERT_NOT_NULL(reduce_other_result);
  ASSERT_INT_EQ(reduce_other_result->op, POLY_OP_UNSHARD);
  ASSERT_INT_EQ(reduce_other_result->arg.int_tuple.vals[0], 0);
  ASSERT_INT_EQ(reduce_other_result->src[0]->op, POLY_OP_REDUCE);
  ASSERT_INT_EQ(reduce_other_result->src[0]->src[0]->op, POLY_OP_PERMUTE);
  ASSERT_INT_EQ(reduce_other_result->src[0]->src[0]->src[0]->op, POLY_OP_MSTACK);
  ASSERT_INT_EQ(multi_pm_test_node_count(ctx, reduce_other_result), 19);

  PolyUOp *explicit_allreduce =
      poly_allreduce(ctx, multi_pm_test_shard(ctx, 909, POLY_FLOAT32, 0), POLY_OP_ADD, tuple);
  PolyUOp *explicit_result = poly_apply_multi_pm(ctx, explicit_allreduce);
  ASSERT_NOT_NULL(explicit_result);
  ASSERT_INT_EQ(explicit_result->op, POLY_OP_UNSHARD);
  ASSERT_INT_EQ(explicit_result->arg.int_tuple.vals[0], 0);
  ASSERT_INT_EQ(explicit_result->src[0]->op, POLY_OP_ALLREDUCE);
  ASSERT_INT_EQ(explicit_result->src[0]->src[0]->op, POLY_OP_MSTACK);
  ASSERT_TRUE(explicit_result->src[0]->arg.allreduce.device_is_tuple);
  ASSERT_INT_EQ(multi_pm_test_node_count(ctx, explicit_result), 18);

  /* COPY(UNSHARD -> scalar) uses every selected occurrence and concatenates
   * them; the older generic COPY_TO_ONE rule is only for axis-less MSTACK. */
  PolyUOp *copy_one = poly_copy_to_device_uop(
      ctx, multi_pm_test_shard(ctx, 910, POLY_FLOAT32, 0), poly_device_uop_from_name(ctx, "CPU")
  );
  PolyUOp *copy_one_result = poly_apply_multi_pm(ctx, copy_one);
  ASSERT_NOT_NULL(copy_one_result);
  ASSERT_INT_EQ(copy_one_result->op, POLY_OP_RESHAPE);
  ASSERT_INT_EQ(copy_one_result->src[0]->op, POLY_OP_STACK);
  ASSERT_INT_EQ(copy_one_result->src[0]->n_src, 2);
  ASSERT_INT_EQ(copy_one_result->src[0]->src[0]->op, POLY_OP_COPY);
  ASSERT_INT_EQ(copy_one_result->src[0]->src[1]->op, POLY_OP_COPY);
  ASSERT_INT_EQ(count_ops(ctx, copy_one_result, POLY_OP_UNSHARD), 0);
  ASSERT_INT_EQ(count_ops(ctx, copy_one_result, POLY_OP_MSELECT), 0);
  ASSERT_INT_EQ(count_ops(ctx, copy_one_result, POLY_OP_COPY), 4);
  ASSERT_INT_EQ(multi_pm_test_node_count(ctx, copy_one_result), 18);

  /* Pinned ALLREDUCE_CAST performs a sharded BF16 collective in BF16 while
   * retaining a float32 local reduction/result (multi.py:72-76). */
  RangeifyEnvSave allreduce_cast_env = rangeify_save_env("ALLREDUCE_CAST");
  setenv("ALLREDUCE_CAST", "1", 1);
  PolyUOp *bf16_children[2] = {
      multi_pm_test_source(ctx, 911, POLY_BFLOAT16, "CPU", 2, 4),
      multi_pm_test_source(ctx, 912, POLY_BFLOAT16, "CPU:1", 2, 4)};
  PolyUOp *bf16_stack =
      poly_uop(ctx, POLY_OP_MSTACK, POLY_BFLOAT16, bf16_children, 2, poly_arg_none());
  PolyUOp *bf16_cast = poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, bf16_stack, poly_arg_none());
  PolyUOp *bf16_multi = multi_pm_test_unshard(ctx, bf16_cast, 0, 2);
  PolyUOp *bf16_reduce = poly_reduce_axis(ctx, POLY_OP_ADD, bf16_multi, axis0, 1);
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
  ASSERT_INT_EQ(multi_pm_test_node_count(ctx, bf16_result), 14);

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
      ctx, POLY_OP_ADD, multi_pm_test_shard(ctx, 907, POLY_FLOAT32, 0), (int64_t[]){0}, 1
  );
  ASSERT_NOT_NULL(root);
  PolyUOp *source_buffer = multi_pm_test_find_buffer_slot(ctx, root, 907);
  ASSERT_NOT_NULL(source_buffer);
  float input[16];
  for (int i = 0; i < 16; i++)
    input[i] = (float)(700 + i);
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
  ASSERT_INT_EQ(multi_pm_test_node_count(ctx, nested_body), 11);
  ASSERT_INT_EQ(count_ops(ctx, nested_body, POLY_OP_AFTER), 1);
  ASSERT_INT_EQ(count_ops(ctx, nested_body, POLY_OP_STORE), 1);
  ASSERT_INT_EQ(count_ops(ctx, nested_body, POLY_OP_CONTIGUOUS), 0);

  PolyUOp *outer_kernel_graph = poly_get_kernel_graph(ctx, earliest);
  ASSERT_NOT_NULL(outer_kernel_graph);
  PolyUOp *outer_nested_call = multi_pm_test_find_named_call(ctx, outer_kernel_graph, "allreduce");
  ASSERT_NOT_NULL(outer_nested_call);
  ASSERT_PTR_EQ(outer_nested_call->src[0], nested_body);
  ASSERT_INT_EQ(multi_pm_test_node_count(ctx, outer_nested_call->src[0]), 11);

  PolyUOp *nested_kernel_graph = poly_get_kernel_graph(ctx, nested_body);
  ASSERT_NOT_NULL(nested_kernel_graph);
  PolyUOp *nested_linear = poly_create_schedule(ctx, nested_kernel_graph);
  ASSERT_NOT_NULL(nested_linear);
  ASSERT_INT_EQ(nested_linear->n_src, 3);
  ASSERT_INT_EQ(nested_linear->src[0]->n_src, 3);
  ASSERT_INT_EQ(nested_linear->src[1]->n_src, 3);
  ASSERT_INT_EQ(nested_linear->src[2]->n_src, 4);

  PolyUOp *scheduled_out = NULL;
  PolyVarBinding *bindings = NULL;
  int n_bindings = 0;
  PolyUOp *linear = poly_linear_with_vars(ctx, &root, 1, &scheduled_out, &bindings, &n_bindings);
  ASSERT_NOT_NULL(linear);
  ASSERT_NOT_NULL(scheduled_out);
  ASSERT_INT_EQ(linear->op, POLY_OP_LINEAR);
  ASSERT_INT_EQ(linear->n_src, 8);
  ASSERT_INT_EQ(count_ops(ctx, linear, POLY_OP_LINEAR), 1);
  PolyOps expected_bodies[8] = {POLY_OP_SINK, POLY_OP_SINK, POLY_OP_COPY, POLY_OP_SINK,
                                POLY_OP_COPY, POLY_OP_COPY, POLY_OP_SINK, POLY_OP_SINK};
  for (int i = 0; i < 8; i++) {
    PolyUOp *call = linear->src[i];
    ASSERT_NOT_NULL(call);
    ASSERT_INT_EQ(call->op, POLY_OP_CALL);
    ASSERT_TRUE(call->n_src >= 1);
    ASSERT_INT_EQ(call->src[0]->op, expected_bodies[i]);
  }
  ASSERT_INT_EQ(linear->src[4]->src[2]->op, POLY_OP_MSELECT);
  ASSERT_INT_EQ(linear->src[5]->src[2]->op, POLY_OP_MSELECT);
  ASSERT_INT_EQ(linear->src[6]->src[2]->op, POLY_OP_MSTACK);
  ASSERT_INT_EQ(linear->src[6]->src[3]->op, POLY_OP_MSTACK);

  const float expected[4] = {2824, 2828, 2832, 2836};
  ASSERT_INT_EQ(poly_run_linear(ctx, linear, bindings, n_bindings, NULL, 0, true, false, false), 0);
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

  PolyUOp *compiled = poly_compile_linear(ctx, linear, -1);
  ASSERT_NOT_NULL(compiled);
  ASSERT_INT_EQ(compiled->n_src, 8);
  PolyOps expected_compiled_bodies[8] = {POLY_OP_PROGRAM, POLY_OP_PROGRAM, POLY_OP_COPY,
                                         POLY_OP_PROGRAM, POLY_OP_COPY,    POLY_OP_COPY,
                                         POLY_OP_PROGRAM, POLY_OP_PROGRAM};
  for (int i = 0; i < 8; i++)
    ASSERT_INT_EQ(compiled->src[i]->src[0]->op, expected_compiled_bodies[i]);
  for (int lane = 0; lane < 2; lane++) {
    PolyBuffer *child = poly_buffer_multi_child(result, lane);
    memset(child->ptr, 0, child->nbytes);
    child->valid = false;
  }
  ASSERT_INT_EQ(
      poly_run_linear(ctx, compiled, bindings, n_bindings, NULL, 0, true, true, false), 0
  );
  for (int lane = 0; lane < 2; lane++) {
    PolyBuffer *child = poly_buffer_multi_child(result, lane);
    ASSERT_NOT_NULL(child);
    ASSERT_TRUE(child->valid);
    for (int i = 0; i < 4; i++)
      ASSERT_FLOAT_EQ(((float *)child->ptr)[i], expected[i], 0.0f);
  }

  free(bindings);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, schedule_rejects_noncopy_exact_device_mismatch) {
  /* Current tinygrad schedule/__init__.py:147-178 validates every PARAM
   * device after COPY recognition, including ordinals beyond src[1:2]. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *shape = poly_const_int(ctx, 2);
  ASSERT_NOT_NULL(shape);
  PolyParamArg args[3] = {
      {.slot = 0, .addrspace = POLY_ADDR_GLOBAL, .device = "CPU"},
      {.slot = 1, .addrspace = POLY_ADDR_GLOBAL, .device = "CPU"},
      {.slot = 2, .addrspace = POLY_ADDR_GLOBAL, .device = "CPU:1"},
  };
  PolyUOp *params[3];
  for (int i = 0; i < 3; i++) {
    params[i] = poly_uop1(ctx, POLY_OP_PARAM, POLY_FLOAT32, shape, poly_arg_param(&args[i]));
    ASSERT_NOT_NULL(params[i]);
  }
  PolyUOp *sum = poly_alu2(ctx, POLY_OP_ADD, params[1], params[2]);
  PolyUOp *store = poly_store_val(ctx, params[0], sum);
  PolyUOp *sink = poly_sink1(ctx, store);
  ASSERT_NOT_NULL(sink);
  PolyUOp *call_src[4] = {sink, params[0], params[1], params[2]};
  PolyUOp *call = poly_uop(ctx, POLY_OP_CALL, POLY_VOID, call_src, 4, poly_arg_none());
  PolyUOp *linear = poly_uop1(ctx, POLY_OP_LINEAR, POLY_VOID, call, poly_arg_none());
  ASSERT_NOT_NULL(linear);
  ASSERT_PTR_EQ(poly_copy_from_store(ctx, linear), NULL);
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
  PolyUOp *buffer = poly_uop_new_buffer(ctx, cpu, 4, POLY_FLOAT32, 301);
  PolyUOp *broadcast = poly_copy_to_device_uop(ctx, buffer, tuple);

  PolyUOp *stack = poly_apply_multi_pm(ctx, broadcast);
  ASSERT_NOT_NULL(stack);
  ASSERT_INT_EQ(stack->op, POLY_OP_MSTACK);
  ASSERT_INT_EQ(stack->n_src, 2);
  ASSERT_INT_EQ(stack->src[0]->op, POLY_OP_COPY);
  ASSERT_INT_EQ(stack->src[1]->op, POLY_OP_COPY);
  ASSERT_PTR_EQ(stack->src[0]->src[0], buffer);
  ASSERT_PTR_EQ(stack->src[1]->src[0], buffer);
  ASSERT_INT_EQ(stack->src[0]->arg.kind, POLY_ARG_STRING);
  ASSERT_STR_EQ(stack->src[0]->arg.str, "CPU");
  ASSERT_STR_EQ(stack->src[1]->arg.str, "CPU:1");

  PolyUOp *select = poly_uop1(ctx, POLY_OP_MSELECT, POLY_FLOAT32, stack, poly_arg_int(1));
  ASSERT_PTR_EQ(poly_apply_multi_pm(ctx, select), stack->src[1]);

  /* Pinned COPY_TO_ONE creates MSELECT(source, 0), then the same fixed-point
   * pass resolves MSELECT(MSTACK) to the exact first occurrence. */
  PolyUOp *to_one = poly_copy_to_device_uop(ctx, stack, cpu1);
  PolyUOp *to_one_rewritten = poly_apply_multi_pm(ctx, to_one);
  ASSERT_NOT_NULL(to_one_rewritten);
  ASSERT_INT_EQ(to_one_rewritten->op, POLY_OP_COPY);
  ASSERT_PTR_EQ(to_one_rewritten->src[0], stack->src[0]);
  ASSERT_STR_EQ(to_one_rewritten->arg.str, "CPU:1");

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

TEST(rangeify, multi_pm_stack_index_and_store_match_current_tinygrad) {
  /* Current schedule/multi.py:stack_multi, index_multi, store_value_multi,
   * and store_dest_multi. Paired graph evidence:
   * temp/reference_upgrade_20260825/scheduler_vocabulary/
   * tg_multi_remaining_probe.out. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *a = multi_pm_test_shard(ctx, 340, POLY_FLOAT32, 0);
  PolyUOp *b = multi_pm_test_shard(ctx, 341, POLY_FLOAT32, 0);
  ASSERT_NOT_NULL(a);
  ASSERT_NOT_NULL(b);

  PolyUOp *stack_src[] = {a, b};
  PolyUOp *stack = poly_uop(ctx, POLY_OP_STACK, POLY_FLOAT32, stack_src, 2, poly_arg_none());
  PolyUOp *stack_result = poly_apply_multi_pm(ctx, stack);
  ASSERT_NOT_NULL(stack_result);
  ASSERT_INT_EQ(stack_result->op, POLY_OP_UNSHARD);
  ASSERT_INT_EQ(stack_result->arg.int_tuple.vals[0], 1);
  ASSERT_INT_EQ(stack_result->src[0]->op, POLY_OP_STACK);
  ASSERT_INT_EQ(stack_result->src[0]->src[0]->op, POLY_OP_MSTACK);
  ASSERT_INT_EQ(stack_result->src[0]->src[1]->op, POLY_OP_MSTACK);

  PolyUOp *two = poly_const_int(ctx, 2);
  PolyUOp *one = poly_const_int(ctx, 1);
  PolyUOp *row = poly_alu2(ctx, POLY_OP_MUL, a->src[1], two);
  PolyUOp *index_src[] = {a, row, one};
  PolyUOp *index = poly_uop(ctx, POLY_OP_INDEX, POLY_FLOAT32, index_src, 3, poly_arg_none());
  PolyUOp *index_result = poly_apply_multi_pm(ctx, index);
  ASSERT_NOT_NULL(index_result);
  ASSERT_INT_EQ(index_result->op, POLY_OP_INDEX);
  ASSERT_INT_EQ(index_result->src[0]->op, POLY_OP_MSTACK);
  ASSERT_INT_EQ(index_result->src[1]->op, POLY_OP_CONST);
  ASSERT_INT_EQ(index_result->src[1]->arg.i, 0);
  ASSERT_PTR_EQ(index_result->src[2], one);

  const char *devices[] = {"CPU", "CPU:1"};
  PolyUOp *tuple = poly_device_uop_from_names(ctx, devices, 2);
  PolyUOp *dest_buffer = poly_uop_new_buffer(ctx, tuple, 16, POLY_FLOAT32, 342);
  PolyUOp *dest = poly_reshape(ctx, dest_buffer, (int64_t[]){4, 4}, 2);
  PolyUOp *store_value = poly_store_val(ctx, dest, a);
  PolyUOp *store_value_result = poly_apply_multi_pm(ctx, store_value);
  ASSERT_NOT_NULL(store_value_result);
  ASSERT_INT_EQ(store_value_result->op, POLY_OP_STORE);
  ASSERT_INT_EQ(store_value_result->src[0]->op, POLY_OP_SHRINK);
  ASSERT_INT_EQ(store_value_result->src[1]->op, POLY_OP_MSTACK);

  PolyUOp *store_dest = poly_store_val(ctx, a, b);
  PolyUOp *store_dest_result = poly_apply_multi_pm(ctx, store_dest);
  ASSERT_NOT_NULL(store_dest_result);
  ASSERT_INT_EQ(store_dest_result->op, POLY_OP_STORE);
  ASSERT_INT_EQ(store_dest_result->src[0]->op, POLY_OP_MSTACK);
  ASSERT_INT_EQ(store_dest_result->src[1]->op, POLY_OP_MSTACK);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, multi_pm_multi_axis_sharding_matches_current_tinygrad) {
  /* Current UOp.unshard accepts sharding expressions, and copy_multi groups
   * their concrete device indices from last axis to first. Paired evidence:
   * temp/reference_upgrade_20260825/scheduler_vocabulary/
   * tg_multi_axis_probe.out. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  const char *names[4] = {"CPU", "CPU:1", "CPU:2", "CPU:3"};
  PolyUOp *a_local[4], *b_local[4];
  for (int i = 0; i < 4; i++) {
    PolyUOp *device = poly_device_uop_from_name(ctx, names[i]);
    PolyUOp *a_buffer = poly_uop_new_buffer(ctx, device, 4, POLY_FLOAT32, 350 + i);
    PolyUOp *b_buffer = poly_uop_new_buffer(ctx, device, 4, POLY_FLOAT32, 354 + i);
    a_local[i] = poly_reshape(ctx, a_buffer, (int64_t[]){2, 2}, 2);
    b_local[i] = poly_reshape(ctx, b_buffer, (int64_t[]){2, 2}, 2);
  }
  PolyUOp *a_stack = poly_uop(ctx, POLY_OP_MSTACK, POLY_FLOAT32, a_local, 4, poly_arg_none());
  PolyUOp *b_stack = poly_uop(ctx, POLY_OP_MSTACK, POLY_FLOAT32, b_local, 4, poly_arg_none());
  PolyUOp *device_range = poly_range(ctx, 4, -1, POLY_AXIS_DEVICE);
  PolyUOp *two = poly_const_int(ctx, 2);
  PolyUOp *ranges[2] = {
      poly_binop(ctx, POLY_OP_FLOORDIV, device_range, two),
      poly_binop(ctx, POLY_OP_FLOORMOD, device_range, two)};
  int64_t axes[2] = {0, 1};
  PolyUOp *a = poly_unshard(ctx, a_stack, axes, ranges, 2);
  PolyUOp *b = poly_unshard(ctx, b_stack, axes, ranges, 2);
  ASSERT_NOT_NULL(a);
  ASSERT_NOT_NULL(b);

  PolyUOp *add = poly_apply_multi_pm(ctx, poly_alu2(ctx, POLY_OP_ADD, a, b));
  ASSERT_NOT_NULL(add);
  ASSERT_INT_EQ(add->op, POLY_OP_UNSHARD);
  ASSERT_INT_EQ(add->arg.int_tuple.n, 2);
  ASSERT_INT_EQ(add->arg.int_tuple.vals[0], 0);
  ASSERT_INT_EQ(add->arg.int_tuple.vals[1], 1);
  ASSERT_INT_EQ(add->src[0]->op, POLY_OP_ADD);
  ASSERT_INT_EQ(add->src[0]->src[0]->op, POLY_OP_MSTACK);
  ASSERT_INT_EQ(add->src[0]->src[1]->op, POLY_OP_MSTACK);
  ASSERT_INT_EQ(multi_pm_test_node_count(ctx, add), 26);

  PolyUOp *copy = poly_apply_multi_pm(
      ctx, poly_copy_to_device_uop(ctx, a, poly_device_uop_from_name(ctx, "CPU"))
  );
  ASSERT_NOT_NULL(copy);
  ASSERT_INT_EQ(copy->op, POLY_OP_RESHAPE);
  ASSERT_INT_EQ(copy->src[0]->op, POLY_OP_STACK);
  ASSERT_INT_EQ(copy->src[0]->n_src, 2);
  PolyShape copy_shape = poly_uop_max_shape_cached(ctx, copy);
  ASSERT_INT_EQ(copy_shape.ndim, 2);
  ASSERT_INT_EQ(copy_shape.dims[0], 4);
  ASSERT_INT_EQ(copy_shape.dims[1], 4);
  ASSERT_INT_EQ(multi_pm_test_node_count(ctx, copy), 25);

  PolyUOp *permute = poly_apply_multi_pm(ctx, poly_permute(ctx, a, (int64_t[]){1, 0}, 2));
  ASSERT_NOT_NULL(permute);
  ASSERT_INT_EQ(permute->op, POLY_OP_UNSHARD);
  ASSERT_INT_EQ(permute->arg.int_tuple.n, 2);
  ASSERT_PTR_EQ(permute->src[1], ranges[1]);
  ASSERT_PTR_EQ(permute->src[2], ranges[0]);
  ASSERT_INT_EQ(permute->src[0]->op, POLY_OP_PERMUTE);
  ASSERT_INT_EQ(multi_pm_test_node_count(ctx, permute), 17);

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
  PolyUOp *param_shape = poly_uop(ctx, POLY_OP_STACK, POLY_WEAKINT, shape_src, 2, poly_arg_none());
  PolyParamArg param_arg = {
      .slot = 1,
      .addrspace = POLY_ADDR_GLOBAL,
      .axis = 0,
      .has_axis = true,
      .devices = devices,
      .n_devices = 2,
      .device_is_tuple = true,
  };
  PolyUOp *public_param =
      poly_uop1(ctx, POLY_OP_PARAM, POLY_FLOAT32, param_shape, poly_arg_param(&param_arg));
  PolyUOp *local_param = poly_apply_multi_pm(ctx, public_param);
  ASSERT_NOT_NULL(local_param);
  ASSERT_INT_EQ(local_param->op, POLY_OP_UNSHARD);
  ASSERT_INT_EQ(local_param->arg.kind, POLY_ARG_INT_TUPLE);
  ASSERT_INT_EQ(local_param->arg.int_tuple.vals[0], 0);
  ASSERT_INT_EQ(local_param->n_src, 2);
  ASSERT_INT_EQ(local_param->src[0]->op, POLY_OP_PARAM);
  ASSERT_TRUE(local_param->src[0]->arg.kind == POLY_ARG_PARAM && local_param->src[0]->arg.param);
  ASSERT_FALSE(local_param->src[0]->arg.param->has_axis);
  PolyShape local_param_shape = poly_uop_max_shape_cached(ctx, local_param->src[0]);
  ASSERT_INT_EQ(local_param_shape.ndim, 2);
  ASSERT_INT_EQ(local_param_shape.dims[0], 2);
  ASSERT_INT_EQ(local_param_shape.dims[1], 4);

  PolyUOp *reshape =
      poly_reshape(ctx, multi_pm_test_shard(ctx, 920, POLY_FLOAT32, 0), (int64_t[]){2, 2, 4}, 3);
  PolyUOp *expand = poly_expand(
      ctx, multi_pm_test_axis0_column_shard(ctx, 921, POLY_FLOAT32), (int64_t[]){4, 3}, 2
  );
  PolyUOp *pad = poly_pad(
      ctx, multi_pm_test_shard(ctx, 922, POLY_FLOAT32, 0), (int64_t[][2]){{0, 0}, {1, 1}}, 2
  );
  PolyUOp *permute =
      poly_permute(ctx, multi_pm_test_shard(ctx, 923, POLY_FLOAT32, 0), (int64_t[]){1, 0}, 2);
  PolyUOp *shrink = poly_shrink(
      ctx, multi_pm_test_shard(ctx, 924, POLY_FLOAT32, 0), (int64_t[][2]){{0, 4}, {1, 3}}, 2
  );
  PolyUOp *flip = poly_flip(ctx, multi_pm_test_shard(ctx, 925, POLY_FLOAT32, 0), (int64_t[]){1}, 1);
  PolyUOp *movement_roots[6] = {reshape, expand, pad, permute, shrink, flip};
  PolyOps local_ops[6] = {POLY_OP_RESHAPE, POLY_OP_PERMUTE, POLY_OP_PAD,
                          POLY_OP_PERMUTE, POLY_OP_MSTACK,  POLY_OP_FLIP};
  int64_t expected_axes[6] = {0, 0, 0, 1, 0, 0};
  for (int i = 0; i < 6; i++) {
    PolyUOp *moved = poly_apply_multi_pm(ctx, movement_roots[i]);
    ASSERT_NOT_NULL(moved);
    ASSERT_INT_EQ(moved->op, POLY_OP_UNSHARD);
    ASSERT_INT_EQ(moved->arg.kind, POLY_ARG_INT_TUPLE);
    ASSERT_INT_EQ(moved->arg.int_tuple.vals[0], expected_axes[i]);
    ASSERT_INT_EQ(moved->n_src, 2);
    ASSERT_INT_EQ(moved->src[0]->op, local_ops[i]);
  }

  PolyUOp *partition = poly_shrink(
      ctx, multi_pm_test_shard(ctx, 926, POLY_FLOAT32, 0), (int64_t[][2]){{2, 4}, {0, 4}}, 2
  );
  PolyUOp *partition_result = poly_apply_multi_pm(ctx, partition);
  ASSERT_NOT_NULL(partition_result);
  ASSERT_INT_EQ(partition_result->op, POLY_OP_MSTACK);
  ASSERT_INT_EQ(count_ops(ctx, partition_result, POLY_OP_MSELECT), 0);

  PolyUOp *cross_partition = poly_shrink(
      ctx, multi_pm_test_shard(ctx, 927, POLY_FLOAT32, 0), (int64_t[][2]){{1, 3}, {0, 4}}, 2
  );
  PolyUOp *shard_pad = poly_pad(
      ctx, multi_pm_test_shard(ctx, 928, POLY_FLOAT32, 0), (int64_t[][2]){{1, 0}, {0, 0}}, 2
  );
  PolyUOp *shard_flip =
      poly_flip(ctx, multi_pm_test_shard(ctx, 929, POLY_FLOAT32, 0), (int64_t[]){0}, 1);
  ASSERT_TRUE(poly_apply_multi_pm(ctx, cross_partition) == NULL);
  ASSERT_TRUE(poly_apply_multi_pm(ctx, shard_pad) == NULL);
  ASSERT_TRUE(poly_apply_multi_pm(ctx, shard_flip) == NULL);

  /* The original indexed MSTACK shape, not its scalarized rangeify
   * replacement, supplies strides. This is the exact numerical canary for
   * `_apply_reshape((2,4),(1,2,4), ...)` in schedule/indexing.py:113-127. */
  PolyUOp *source = multi_pm_test_find_buffer_slot(ctx, reshape, 920);
  ASSERT_NOT_NULL(source);
  float input[16];
  for (int i = 0; i < 16; i++)
    input[i] = (float)i;
  ASSERT_INT_EQ(poly_buffer_allocate(ctx, source, POLY_DEVICE_CPU), 0);
  ASSERT_INT_EQ(poly_buffer_copyin(ctx, source, input, sizeof(input)), 0);
  PolyUOp *scheduled_root =
      poly_uop1(ctx, POLY_OP_CONTIGUOUS, reshape->dtype, reshape, poly_arg_none());
  PolyUOp *scheduled_out = NULL;
  PolyVarBinding *vars = NULL;
  int n_vars = 0;
  PolyUOp *linear = poly_linear_with_vars(ctx, &scheduled_root, 1, &scheduled_out, &vars, &n_vars);
  ASSERT_NOT_NULL(linear);
  ASSERT_NOT_NULL(scheduled_out);
  ASSERT_TRUE(linear->n_src >= 4);
  PolyUOp *final_call = poly_test_linear_call(linear, linear->n_src - 1);
  ASSERT_NOT_NULL(final_call);
  ASSERT_INT_EQ(final_call->n_src, 3);
  ASSERT_INT_EQ(final_call->src[2]->op, POLY_OP_MSTACK);
  ASSERT_INT_EQ(final_call->src[2]->n_src, 2);
  ASSERT_INT_EQ(count_ops(ctx, final_call->src[2], POLY_OP_PARAM), 0);
  ASSERT_INT_EQ(poly_run_linear(ctx, linear, vars, n_vars, NULL, 0, true, false, false), 0);
  free(vars);
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
  PolyUOp *stack = poly_uop(ctx, POLY_OP_MSTACK, POLY_FLOAT32, stack_src, 2, poly_arg_none());
  PolyUOp *multi = multi_pm_test_unshard(ctx, stack, 0, 2);

  PolyUOp *moved = poly_reshape(ctx, stack, (int64_t[]){4, 2}, 2);
  PolyUOp *selected = poly_uop1(ctx, POLY_OP_MSELECT, POLY_FLOAT32, moved, poly_arg_int(1));
  PolyUOp *selected_result = poly_apply_multi_pm(ctx, selected);
  ASSERT_NOT_NULL(selected_result);
  ASSERT_INT_EQ(selected_result->op, POLY_OP_RESHAPE);
  ASSERT_PTR_EQ(selected_result->src[0], local1);
  ASSERT_INT_EQ(count_ops(ctx, selected_result, POLY_OP_MSELECT), 0);
  ASSERT_INT_EQ(count_ops(ctx, selected_result, POLY_OP_MSTACK), 0);

  PolyUOp *plain_tuple_src[2] = {local0, local1};
  PolyUOp *plain_tuple =
      poly_uop(ctx, POLY_OP_TUPLE, POLY_VOID, plain_tuple_src, 2, poly_arg_none());
  PolyUOp *plain_get = poly_uop1(ctx, POLY_OP_GETTUPLE, POLY_FLOAT32, plain_tuple, poly_arg_int(1));
  ASSERT_PTR_EQ(poly_apply_multi_pm(ctx, plain_get), local1);

  PolyUOp *tuple_body = poly_uop1(ctx, POLY_OP_TUPLE, POLY_VOID, stack, poly_arg_none());
  PolyUOp *tuple_multi = multi_pm_test_unshard(ctx, tuple_body, 0, 2);
  PolyUOp *tuple_get = poly_uop1(ctx, POLY_OP_GETTUPLE, POLY_FLOAT32, tuple_multi, poly_arg_int(0));
  PolyUOp *tuple_get_result = poly_apply_multi_pm(ctx, tuple_get);
  ASSERT_NOT_NULL(tuple_get_result);
  ASSERT_INT_EQ(tuple_get_result->op, POLY_OP_UNSHARD);
  ASSERT_INT_EQ(tuple_get_result->arg.kind, POLY_ARG_INT_TUPLE);
  ASSERT_INT_EQ(tuple_get_result->arg.int_tuple.n, 1);
  ASSERT_INT_EQ(tuple_get_result->arg.int_tuple.vals[0], 0);
  ASSERT_PTR_EQ(tuple_get_result->src[0], stack);

  PolyUOp *axis_param = multi_pm_test_param(ctx, 0, NULL, devices, 2, true);
  PolyUOp *body_values[2] = {
      poly_alu2(
          ctx, POLY_OP_ADD, axis_param,
          poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(1.0))
      ),
      axis_param};
  PolyUOp *body = poly_uop(ctx, POLY_OP_TUPLE, POLY_VOID, body_values, 2, poly_arg_none());
  PolyUOp *function_src[2] = {body, axis_param};
  PolyUOp *function =
      poly_uop(ctx, POLY_OP_FUNCTION, POLY_VOID, function_src, 2, poly_arg_str("multi_value"));
  PolyUOp *function_result = poly_apply_multi_pm(ctx, function);
  ASSERT_NOT_NULL(function_result);
  ASSERT_INT_EQ(function_result->op, POLY_OP_TUPLE);
  ASSERT_INT_EQ(function_result->n_src, 2);
  for (int i = 0; i < 2; i++) {
    ASSERT_INT_EQ(function_result->src[i]->op, POLY_OP_UNSHARD);
    ASSERT_INT_EQ(function_result->src[i]->arg.int_tuple.vals[0], 0);
    ASSERT_INT_EQ(function_result->src[i]->src[0]->op, POLY_OP_GETTUPLE);
    ASSERT_INT_EQ(function_result->src[i]->src[0]->arg.i, i);
    ASSERT_INT_EQ(function_result->src[i]->src[0]->src[0]->op, POLY_OP_FUNCTION);
  }
  ASSERT_INT_EQ(count_ops(ctx, function_result, POLY_OP_FUNCTION), 1);
  ASSERT_INT_EQ(count_ops(ctx, function_result, POLY_OP_UNSHARD), 2);

  PolyOps wrappers[3] = {POLY_OP_CAST, POLY_OP_CONTIGUOUS, POLY_OP_DETACH};
  for (int i = 0; i < 3; i++) {
    PolyUOp *wrapped = poly_uop1(ctx, wrappers[i], POLY_FLOAT32, multi, poly_arg_none());
    PolyUOp *wrapped_result = poly_apply_multi_pm(ctx, wrapped);
    ASSERT_NOT_NULL(wrapped_result);
    ASSERT_INT_EQ(wrapped_result->op, POLY_OP_UNSHARD);
    ASSERT_INT_EQ(wrapped_result->arg.int_tuple.vals[0], 0);
    ASSERT_INT_EQ(wrapped_result->src[0]->op, wrappers[i]);
    ASSERT_PTR_EQ(wrapped_result->src[0]->src[0], stack);
  }

  PolyUOp *effect = poly_sink1(ctx, local0);
  PolyUOp *after = poly_uop2(ctx, POLY_OP_AFTER, POLY_FLOAT32, multi, effect, poly_arg_none());
  PolyUOp *after_result = poly_apply_multi_pm(ctx, after);
  ASSERT_NOT_NULL(after_result);
  ASSERT_INT_EQ(after_result->op, POLY_OP_UNSHARD);
  ASSERT_INT_EQ(after_result->src[0]->op, POLY_OP_AFTER);
  ASSERT_PTR_EQ(after_result->src[0]->src[0], stack);

  PolyUOp *call_src[3] = {effect, multi, local1};
  PolyUOp *call = poly_uop(ctx, POLY_OP_CALL, POLY_VOID, call_src, 3, poly_arg_str("void_multi"));
  PolyUOp *call_result = poly_apply_multi_pm(ctx, call);
  ASSERT_NOT_NULL(call_result);
  ASSERT_INT_EQ(call_result->op, POLY_OP_CALL);
  ASSERT_PTR_EQ(call_result->src[0], effect);
  ASSERT_PTR_EQ(call_result->src[1], stack);
  ASSERT_INT_EQ(count_ops(ctx, call_result, POLY_OP_UNSHARD), 0);

  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, multi, multi, poly_arg_none());
  PolyUOp *store_result = poly_apply_multi_pm(ctx, store);
  ASSERT_NOT_NULL(store_result);
  ASSERT_INT_EQ(store_result->op, POLY_OP_STORE);
  /* Current store_value_multi runs before store_dest_multi. Each destination
   * lane is the contiguous shard subview; the value peels to the MSTACK. */
  ASSERT_INT_EQ(store_result->src[0]->op, POLY_OP_MSTACK);
  ASSERT_INT_EQ(store_result->src[0]->n_src, 2);
  for (int lane = 0; lane < 2; lane++) {
    PolyUOp *contiguous = store_result->src[0]->src[lane];
    ASSERT_INT_EQ(contiguous->op, POLY_OP_CONTIGUOUS);
    ASSERT_INT_EQ(contiguous->src[0]->op, POLY_OP_SHRINK);
    ASSERT_PTR_EQ(contiguous->src[0]->src[0], stack->src[lane]);
  }
  ASSERT_PTR_EQ(store_result->src[1], stack);
  ASSERT_INT_EQ(count_ops(ctx, store_result, POLY_OP_UNSHARD), 0);
  int store_result_n = 0;
  ASSERT_NOT_NULL(poly_toposort(ctx, store_result, &store_result_n));
  ASSERT_INT_EQ(store_result_n, 14);

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
  PolyUOp *sum_body = poly_uop1(ctx, POLY_OP_TUPLE, POLY_VOID, sum, poly_arg_none());
  PolyUOp *value_src[3] = {sum_body, lhs, rhs};
  PolyUOp *value_function =
      poly_uop(ctx, POLY_OP_FUNCTION, POLY_VOID, value_src, 3, poly_arg_str("value_add"));
  PolyUOp *value = poly_uop1(ctx, POLY_OP_GETTUPLE, POLY_FLOAT32, value_function, poly_arg_int(0));
  PolyUOp *requested = poly_uop1(ctx, POLY_OP_CONTIGUOUS, POLY_FLOAT32, value, poly_arg_none());
  PolyUOp *post_multi = poly_apply_multi_pm(ctx, requested);
  ASSERT_NOT_NULL(post_multi);
  ASSERT_INT_EQ(post_multi->op, POLY_OP_UNSHARD);
  ASSERT_INT_EQ(post_multi->arg.kind, POLY_ARG_INT_TUPLE);
  ASSERT_INT_EQ(post_multi->arg.int_tuple.n, 1);
  ASSERT_INT_EQ(post_multi->arg.int_tuple.vals[0], 0);
  ASSERT_INT_EQ(count_ops(ctx, post_multi, POLY_OP_FUNCTION), 1);
  ASSERT_INT_EQ(count_ops(ctx, post_multi, POLY_OP_GETTUPLE), 1);
  ASSERT_INT_EQ(count_ops(ctx, post_multi, POLY_OP_MSTACK), 2);
  PolyShape post_shape = poly_uop_max_shape_cached(ctx, post_multi);
  ASSERT_INT_EQ(post_shape.ndim, 2);
  ASSERT_INT_EQ(post_shape.dims[0], 4);
  ASSERT_INT_EQ(post_shape.dims[1], 4);

  PolyUOp *scheduled_out = NULL;
  PolyVarBinding *vars = NULL;
  int n_vars = 0;
  PolyUOp *linear = poly_linear_with_vars(ctx, &requested, 1, &scheduled_out, &vars, &n_vars);
  ASSERT_NOT_NULL(linear);
  ASSERT_NOT_NULL(scheduled_out);
  ASSERT_INT_EQ(linear->n_src, 1);
  ASSERT_INT_EQ(poly_run_linear(ctx, linear, vars, n_vars, NULL, 0, true, false, false), 0);
  free(vars);
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
  PolyUOp *buffer = poly_uop_new_buffer(ctx, cpu, 8, POLY_INT32, 331);
  int32_t input_values[] = {0, 1, 2, 3, 4, 5, 6, 7};
  ASSERT_INT_EQ(poly_buffer_allocate(ctx, buffer, POLY_DEVICE_CPU), 0);
  ASSERT_INT_EQ(poly_buffer_copyin(ctx, buffer, input_values, sizeof(input_values)), 0);
  PolyUOp *value = poly_reshape(ctx, buffer, (int64_t[]){4, 2}, 2);
  PolyUOp *broadcast = poly_copy_to_device_uop(ctx, value, tuple);
  PolyUOp *dvar = poly_range(ctx, 2, -1, POLY_AXIS_DEVICE);
  PolyUOp *two = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(2));
  PolyUOp *starts[] = {
      poly_alu2(ctx, POLY_OP_MUL, dvar, two),
      poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(0))};
  PolyUOp *sizes[] = {two, two};
  PolyUOp *local = poly_shrink_uop(ctx, broadcast, starts, sizes, 2);
  PolyUOp *multi = multi_pm_test_unshard(ctx, local, 0, 2);
  PolyUOp *rewritten = poly_apply_multi_pm(ctx, multi);
  ASSERT_NOT_NULL(rewritten);
  ASSERT_INT_EQ(rewritten->op, POLY_OP_UNSHARD);
  ASSERT_INT_EQ(rewritten->arg.kind, POLY_ARG_INT_TUPLE);
  ASSERT_INT_EQ(rewritten->arg.int_tuple.n, 1);
  ASSERT_INT_EQ(rewritten->arg.int_tuple.vals[0], 0);
  ASSERT_INT_EQ(rewritten->n_src, 2);
  PolyUOp *stack = rewritten->src[0];
  ASSERT_INT_EQ(stack->op, POLY_OP_MSTACK);
  ASSERT_INT_EQ(stack->n_src, 2);
  ASSERT_INT_EQ(count_ops(ctx, rewritten, POLY_OP_SHRINK), 2);
  ASSERT_INT_EQ(count_ops(ctx, rewritten, POLY_OP_COPY), 2);
  ASSERT_INT_EQ(count_ops(ctx, rewritten, POLY_OP_PARAM), 0);
  ASSERT_INT_EQ(count_ops(ctx, rewritten, POLY_OP_MUL), 0);
  PolyUOp *expected_devices[] = {cpu, cpu1};
  int64_t expected_start0[] = {0, 2};
  for (int i = 0; i < 2; i++) {
    PolyUOp *copy = stack->src[i];
    ASSERT_INT_EQ(copy->op, POLY_OP_COPY);
    ASSERT_INT_EQ(copy->n_src, 1);
    ASSERT_INT_EQ(copy->arg.kind, POLY_ARG_STRING);
    ASSERT_STR_EQ(copy->arg.str, expected_devices[i]->arg.str);
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
  ASSERT_INT_EQ(callified_out->op, POLY_OP_UNSHARD);
  ASSERT_INT_EQ(callified_out->arg.kind, POLY_ARG_INT_TUPLE);
  ASSERT_INT_EQ(callified_out->arg.int_tuple.vals[0], 0);
  ASSERT_INT_EQ(callified_out->n_src, 2);
  ASSERT_INT_EQ(callified_out->src[0]->op, POLY_OP_RESHAPE);
  const PolyUOp *callified_identity = poly_uop_get_buffer_identity(callified_out);
  ASSERT_NOT_NULL(callified_identity);
  ASSERT_INT_EQ(callified_identity->op, POLY_OP_BUFFER);
  ASSERT_INT_EQ(callified_identity->arg.kind, POLY_ARG_PARAM);
  ASSERT_NOT_NULL(callified_identity->arg.param);
  ASSERT_TRUE(callified_identity->arg.param->device_is_tuple);
  ASSERT_INT_EQ(callified_identity->arg.param->n_devices, 2);
  ASSERT_INT_EQ(callified_identity->n_src, 1);
  ASSERT_INT_EQ(callified_identity->src[0]->op, POLY_OP_CONST);
  ASSERT_INT_EQ(callified_identity->src[0]->arg.i, 4);

  /* Pinned schedule/multi.py:122,153 moves a sharded assignment inside the
   * value MULTI: MULTI(AFTER(local_dest, STORE(local_dest, local_value))). */
  PolyUOp *post_call_multi = poly_apply_multi_pm(ctx, callified->src[0]);
  ASSERT_NOT_NULL(post_call_multi);
  ASSERT_INT_EQ(post_call_multi->op, POLY_OP_SINK);
  ASSERT_INT_EQ(post_call_multi->n_src, 1);
  PolyUOp *assigned_multi = post_call_multi->src[0];
  ASSERT_INT_EQ(assigned_multi->op, POLY_OP_UNSHARD);
  ASSERT_INT_EQ(assigned_multi->arg.kind, POLY_ARG_INT_TUPLE);
  ASSERT_INT_EQ(assigned_multi->arg.int_tuple.vals[0], 0);
  ASSERT_INT_EQ(assigned_multi->n_src, 2);
  PolyUOp *local_after = assigned_multi->src[0];
  ASSERT_INT_EQ(local_after->op, POLY_OP_AFTER);
  ASSERT_INT_EQ(local_after->n_src, 2);
  ASSERT_INT_EQ(local_after->src[0]->op, POLY_OP_RESHAPE);
  /* Current schedule/multi.py:param_to_multi preserves p.device and clears
   * only p.axis; the local PARAM remains bound to the aggregate tuple. */
  PolyUOp *local_target_param = local_after->src[0]->src[0];
  ASSERT_INT_EQ(local_target_param->op, POLY_OP_PARAM);
  ASSERT_INT_EQ(local_target_param->arg.kind, POLY_ARG_PARAM);
  ASSERT_NOT_NULL(local_target_param->arg.param);
  ASSERT_TRUE(local_target_param->arg.param->device_is_tuple);
  ASSERT_INT_EQ(local_target_param->arg.param->n_devices, 2);
  ASSERT_STR_EQ(local_target_param->arg.param->devices[0], "CPU");
  ASSERT_STR_EQ(local_target_param->arg.param->devices[1], "CPU:1");
  ASSERT_INT_EQ(local_after->src[1]->op, POLY_OP_STORE);
  ASSERT_INT_EQ(local_after->src[1]->n_src, 2);
  /* Current store_value_multi runs before store_dest_multi even when both
   * STORE operands are UNSHARD, so the destination is first shard-subviewed. */
  ASSERT_INT_EQ(local_after->src[1]->src[0]->op, POLY_OP_SHRINK);
  ASSERT_PTR_EQ(local_after->src[1]->src[0]->src[0], local_after->src[0]);
  ASSERT_INT_EQ(local_after->src[1]->src[1]->op, POLY_OP_MSTACK);
  ASSERT_INT_EQ(count_ops(ctx, post_call_multi, POLY_OP_UNSHARD), 1);
  ASSERT_INT_EQ(count_ops(ctx, post_call_multi, POLY_OP_STORE), 1);
  int post_call_n = 0;
  ASSERT_NOT_NULL(poly_toposort(ctx, post_call_multi, &post_call_n));
  ASSERT_INT_EQ(post_call_n, 23);

  /* Pinned ALWAYS_RUN_OPS retains the same-device NOOP materialization made
   * by COPY(x, CPU) -> NOOP(x), so axis-0 sharding has two CPU slice kernels,
   * one CPU:1 transfer, and one tuple CALL (rangeify.py:184-187,216,250).
   * Inlining NOOP here silently changes the exact schedule from four CALLs
   * to three even though the final values can remain equal. */
  PolyUOp *multi_kernel_graph = poly_get_kernel_graph(ctx, post_call_multi);
  ASSERT_NOT_NULL(multi_kernel_graph);
  PolyUOp *multi_linear = poly_create_schedule(ctx, multi_kernel_graph);
  ASSERT_NOT_NULL(multi_linear);
  ASSERT_INT_EQ(multi_linear->n_src, 4);

  /* Pinned to_define_global debufs MSTACK/MSELECT as one buffer argument
   * (rangeify.py:497-503,528-535). The final tuple kernel therefore records
   * one output and one aggregate input; its body sees two PARAMs and no
   * MSTACK. CALL construction/runtime resolution is the following boundary. */
  ASSERT_INT_EQ(multi_linear->src[3]->n_src, 3);
  ASSERT_INT_EQ(multi_linear->src[3]->src[2]->op, POLY_OP_MSTACK);
  ASSERT_INT_EQ(count_ops(ctx, multi_linear->src[3]->src[0], POLY_OP_PARAM), 2);
  ASSERT_INT_EQ(count_ops(ctx, multi_linear->src[3]->src[0], POLY_OP_MSTACK), 0);

  /* Pinned schedule/__init__.py:61-64 applies `_unwrap_src(s).buf_uop` to
   * CALL arguments, and UOp.buf_uop preserves MSTACK recursively
   * (uop/ops.py:800-807). The public LINEAR/replay boundary must retain that
   * aggregate argument instead of flattening it into slots or rejecting it. */
  PolyUOp *scheduled_out = NULL;
  PolyVarBinding *vars = NULL;
  int n_vars = 0;
  PolyUOp *linear = poly_linear_with_vars(ctx, &multi, 1, &scheduled_out, &vars, &n_vars);
  ASSERT_NOT_NULL(linear);
  ASSERT_NOT_NULL(scheduled_out);
  ASSERT_INT_EQ(scheduled_out->op, POLY_OP_UNSHARD);
  ASSERT_INT_EQ(linear->n_src, 4);
  PolyUOp *final_call = poly_test_linear_call(linear, 3);
  ASSERT_NOT_NULL(final_call);
  ASSERT_INT_EQ(final_call->op, POLY_OP_CALL);
  ASSERT_INT_EQ(final_call->n_src, 3);
  ASSERT_INT_EQ(final_call->src[1]->op, POLY_OP_BUFFER);
  ASSERT_INT_EQ(final_call->src[1]->arg.kind, POLY_ARG_PARAM);
  ASSERT_NOT_NULL(final_call->src[1]->arg.param);
  ASSERT_TRUE(final_call->src[1]->arg.param->device_is_tuple);
  ASSERT_INT_EQ(final_call->src[1]->arg.param->n_devices, 2);
  ASSERT_STR_EQ(final_call->src[1]->arg.param->devices[0], "CPU");
  ASSERT_STR_EQ(final_call->src[1]->arg.param->devices[1], "CPU:1");
  ASSERT_INT_EQ(final_call->src[2]->op, POLY_OP_MSTACK);
  ASSERT_INT_EQ(final_call->src[2]->n_src, 2);
  /* Current schedule/memory.py:55-57 replaces each scalar child with
   * BITCAST(SHRINK(int8 arena)); planning must recurse through MSTACK. */
  for (int lane = 0; lane < 2; lane++) {
    PolyUOp *view = final_call->src[2]->src[lane];
    ASSERT_INT_EQ(view->op, POLY_OP_BITCAST);
    ASSERT_INT_EQ(view->n_src, 1);
    ASSERT_INT_EQ(view->src[0]->op, POLY_OP_SHRINK);
    ASSERT_INT_EQ(view->src[0]->src[0]->op, POLY_OP_BUFFER);
    ASSERT_TRUE(poly_dtype_eq(view->src[0]->src[0]->dtype, POLY_INT8));
  }
  ASSERT_STR_EQ(poly_uop_device_name(ctx, final_call->src[2]->src[0]), "CPU");
  ASSERT_STR_EQ(poly_uop_device_name(ctx, final_call->src[2]->src[1]), "CPU:1");
  PolyUOp *final_body = poly_test_linear_call_body(linear, 3);
  ASSERT_NOT_NULL(final_body);
  ASSERT_INT_EQ(count_ops(ctx, final_body, POLY_OP_PARAM), 2);
  ASSERT_INT_EQ(count_ops(ctx, final_body, POLY_OP_MSTACK), 0);

  /* Pinned exec_kernel resolves `[BUFFER, MSTACK]` into two ordered lanes and
   * executes one kernel per child (engine/realize.py:142-180). The first three
   * scalar CALLs plus the two final lanes are five executions. */
  ASSERT_INT_EQ(poly_run_linear(ctx, linear, vars, n_vars, NULL, 0, true, false, false), 0);
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
   * `[BUFFER, MSTACK(BITCAST(SHRINK), BITCAST(SHRINK))]` topology and execute both
   * lanes; scalar CALLs remain pre-lowered. */
  PolyUOp *compiled = poly_compile_linear(ctx, linear, -1);
  ASSERT_NOT_NULL(compiled);
  ASSERT_INT_EQ(compiled->n_src, 4);
  PolyUOp *compiled_final = compiled->src[3];
  ASSERT_INT_EQ(compiled_final->op, POLY_OP_CALL);
  ASSERT_INT_EQ(compiled_final->n_src, 3);
  ASSERT_INT_EQ(compiled_final->src[2]->op, POLY_OP_MSTACK);
  ASSERT_INT_EQ(compiled_final->src[2]->n_src, 2);
  for (int lane = 0; lane < 2; lane++) {
    PolyUOp *view = compiled_final->src[2]->src[lane];
    ASSERT_INT_EQ(view->op, POLY_OP_BITCAST);
    ASSERT_INT_EQ(view->src[0]->op, POLY_OP_SHRINK);
    ASSERT_TRUE(poly_dtype_eq(view->src[0]->src[0]->dtype, POLY_INT8));
  }
  poly_ctx_reset_counters(ctx);
  ASSERT_INT_EQ(poly_run_linear(ctx, compiled, vars, n_vars, NULL, 0, true, true, false), 0);
  ASSERT_INT_EQ(ctx->kernel_count, 5);
  for (int lane = 0; lane < 2; lane++) {
    PolyBuffer *child = poly_buffer_multi_child(scheduled_buffer, lane);
    ASSERT_NOT_NULL(child);
    ASSERT_TRUE(child->valid);
    ASSERT_STR_EQ(child->device_uop->arg.str, expected_lane_devices[lane]);
    for (int i = 0; i < 4; i++)
      ASSERT_INT_EQ(((int32_t *)child->ptr)[i], expected_lane_values[lane][i]);
  }
  free(vars);

  /* The non-COPY branch materializes one local CONTIGUOUS per source and
   * ms.replace preserves both MSTACK metadata fields. */
  PolyUOp *buffer1 = poly_uop_new_buffer(ctx, cpu1, 8, POLY_INT32, 332);
  PolyUOp *value1 = poly_reshape(ctx, buffer1, (int64_t[]){4, 2}, 2);
  PolyUOp *manual_src[] = {value, value1};
  PolyUOp *manual_stack = poly_uop_tagged_arg(
      ctx, POLY_OP_MSTACK, POLY_INT32, manual_src, 2, poly_arg_none(), 701, poly_arg_int(702)
  );
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
  ASSERT_INT_EQ(count_ops(ctx, manual_rewritten, POLY_OP_PARAM), 0);
  for (int i = 0; i < 2; i++) {
    ASSERT_INT_EQ(manual_rewritten->src[i]->op, POLY_OP_CONTIGUOUS);
    ASSERT_INT_EQ(manual_rewritten->src[i]->n_src, 1);
    ASSERT_INT_EQ(manual_rewritten->src[i]->src[0]->op, POLY_OP_SHRINK);
    ASSERT_PTR_EQ(manual_rewritten->src[i]->src[0]->src[0], manual_src[i]);
    ASSERT_INT_EQ(manual_rewritten->src[i]->src[0]->src[1]->src[0]->arg.i, expected_start0[i]);
  }

  /* Without `_device_num`, pinned still duplicates the same static slice over
   * every occurrence. */
  PolyUOp *static_shrink = poly_shrink(ctx, manual_stack, (int64_t[][2]){{1, 3}, {0, 2}}, 2);
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
  ASSERT_INT_EQ(count_ops(ctx, call_rewritten->src[1], POLY_OP_PARAM), 0);

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
  PolyUOp *source = poly_uop_new_buffer(ctx, cpu, 4, POLY_FLOAT32, 311);
  PolyUOp *dest = poly_uop_new_buffer(ctx, cpu1, 4, POLY_FLOAT32, 312);
  ASSERT_INT_EQ(poly_buffer_allocate(ctx, source, POLY_DEVICE_CPU), 0);
  ASSERT_INT_EQ(poly_buffer_copyin(ctx, source, input, sizeof(input)), 0);

  const char *names[] = {"CPU", "CPU:1"};
  PolyUOp *tuple = poly_device_uop_from_names(ctx, names, 2);
  PolyUOp *broadcast = poly_copy_to_device_uop(ctx, source, tuple);
  PolyUOp *select = poly_uop1(ctx, POLY_OP_MSELECT, POLY_FLOAT32, broadcast, poly_arg_int(1));
  PolyUOp *store = poly_store_val(ctx, dest, select);
  PolyUOp *sink = poly_sink1(ctx, store);
  ASSERT_INT_EQ(poly_realize_sink(ctx, sink), 0);
  ASSERT_INT_EQ(poly_buffer_copyout(ctx, dest, output, sizeof(output)), 0);
  for (int i = 0; i < 4; i++)
    ASSERT_FLOAT_EQ(output[i], input[i], 0.0f);

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

  PolyUOp *state = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 6, POLY_DEVICE_CPU);
  PolyUOp *two = poly_const_int(ctx, 2);
  PolyUOp *three = poly_const_int(ctx, 3);
  PolyUOp *store_range =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, two, poly_arg_range(7, POLY_AXIS_LOOP));
  PolyUOp *consumer_range =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, three, poly_arg_range(3, POLY_AXIS_LOOP));
  PolyUOp *store_target =
      poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, state, store_range, poly_arg_none());
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

  PolyUOp *result = apply_add_buffers(ctx, bufferize);
  ASSERT_NOT_NULL(result);
  ASSERT_EQ(result->op, POLY_OP_AFTER);
  ASSERT_PTR_EQ(result->src[0], state);
  ASSERT_INT_EQ(count_ops(ctx, result, POLY_OP_STAGE), 0);
  ASSERT_INT_EQ(count_ops(ctx, result, POLY_OP_AFTER), 1);
  ASSERT_INT_EQ(count_ops(ctx, result, POLY_OP_STORE), 1);
  ASSERT_INT_EQ(count_ops(ctx, result, POLY_OP_END), 1);

  PolyUOp *ended = result->src[1];
  ASSERT_EQ(ended->op, POLY_OP_END);
  ASSERT_INT_EQ(ended->n_src, 3);
  ASSERT_EQ(ended->src[0]->op, POLY_OP_STORE);
  ASSERT_EQ(ended->src[0]->src[0]->op, POLY_OP_INDEX);
  ASSERT_PTR_EQ(ended->src[0]->src[0], store_target);
  ASSERT_PTR_EQ(ended->src[0]->src[0]->src[0], state);
  ASSERT_INT_EQ(poly_range_axis_id(ended->src[1]->arg), 3);
  ASSERT_INT_EQ(poly_range_axis_id(ended->src[2]->arg), 7);

  ASSERT_TRUE(poly_dtype_eq(ended->src[0]->src[0]->dtype, POLY_FLOAT32));

  poly_ctx_destroy(ctx);
  PASS();
}

/* Structural split parity tests */

/* Count how many RANGEs appear in a kernel's toposort that are NOT closed
 * by an END and are NOT reduce ranges (sources of REDUCE ops).
 *
 * Reduce ranges are expected orphans in the pre-codegen kernel: they're
 * handled by pm_reduce in the codegen pipeline, which creates inner
 * BUFFER(REG)/END loops. Non-reduce orphan ranges indicate a real bug. */
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

  PolyUOp *a_flat = poly_test_buffer(ctx, POLY_FLOAT32, 9);
  PolyUOp *b_flat = poly_test_buffer(ctx, POLY_FLOAT32, 9);
  PolyUOp *c_flat = poly_test_buffer(ctx, POLY_FLOAT32, 9);

  int64_t shape2d[] = {3, 3};
  PolyUOp *a = poly_reshape(ctx, a_flat, shape2d, 2);
  int64_t perm[] = {1, 0};
  PolyUOp *a_perm = poly_permute(ctx, a, perm, 2);
  PolyUOp *b = poly_reshape(ctx, b_flat, shape2d, 2);
  PolyUOp *c = poly_reshape(ctx, c_flat, shape2d, 2);

  /* d = neg(permute(reshape(a))), shared → BUFFERIZE */
  PolyUOp *d = poly_uop1(ctx, POLY_OP_NEG, POLY_FLOAT32, a_perm, poly_arg_none());

  /* out1 = d + b (elementwise) */
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, d, b, poly_arg_none());

  /* out2 = reduce_sum(d * c, axis=1) */
  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, d, c, poly_arg_none());
  int64_t red_axes[] = {1};
  PolyUOp *red = poly_reduce_axis(ctx, POLY_OP_ADD, mul, red_axes, 1);

  /* Current Tinygrad tensor.py:transform_to_call -> rangeify.py:get_kernel_graph.
   * Raw STORE sinks are not a parity surface for the internal rangeify passes. */
  PolyUOp *requested[] = {add, red};
  PolyUOp *out_uops[2] = {NULL, NULL};
  PolyUOp *big_call = poly_transform_to_call(ctx, requested, 2, out_uops);
  ASSERT_NOT_NULL(big_call);
  ASSERT_EQ(big_call->op, POLY_OP_CALL);
  ASSERT_INT_EQ(count_ops(ctx, big_call, POLY_OP_AFTER), 2);
  ASSERT_INT_EQ(count_ops(ctx, big_call, POLY_OP_STORE), 2);
  ASSERT_INT_EQ(count_ops(ctx, big_call, POLY_OP_STAGE), 0);

  PolyUOp *kernel_graph = poly_get_kernel_graph(ctx, big_call->src[0]);
  ASSERT_NOT_NULL(kernel_graph);
  ASSERT_INT_EQ(count_ops(ctx, kernel_graph, POLY_OP_CALL), 2);
  ASSERT_INT_EQ(count_ops(ctx, kernel_graph, POLY_OP_AFTER), 2);
  ASSERT_INT_EQ(count_ops(ctx, kernel_graph, POLY_OP_STORE), 2);
  ASSERT_INT_EQ(count_ops(ctx, kernel_graph, POLY_OP_STAGE), 0);
  ASSERT_INT_EQ(count_ops(ctx, kernel_graph, POLY_OP_RANGE), 3);
  ASSERT_INT_EQ(count_ops(ctx, kernel_graph, POLY_OP_END), 2);
  ASSERT_INT_EQ(count_ops(ctx, kernel_graph, POLY_OP_REDUCE), 1);

  PolyVarBinding *vars = NULL;
  int n_vars = 0;
  PolyUOp *scheduled_linear = poly_create_linear_with_vars(ctx, big_call, &vars, &n_vars);
  ASSERT_NOT_NULL(scheduled_linear);

  /* Assertion 2: correct kernel count */
  ASSERT_INT_EQ(scheduled_linear->n_src, 2); /* 2 fused kernels (d inlined) */

  /* Assertion 3: no orphan RANGEs in any kernel */
  for (int k = 0; k < scheduled_linear->n_src; k++) {
    int orphans = count_orphan_ranges(ctx, poly_test_linear_call_body(scheduled_linear, k));
    if (orphans > 0) {
      fprintf(stderr, "    kernel %d has %d orphan RANGE(s)\n", k, orphans);
    }
    ASSERT_INT_EQ(orphans, 0);
  }

  /* Assertion 4: structural op counts per kernel */
  /* K0 (elementwise): neg(perm(reshape(a))) + b → out1 */
  int k0_stores =
      kernel_op_count(ctx, poly_test_linear_call_body(scheduled_linear, 0), POLY_OP_STORE);
  int k0_ends = kernel_op_count(ctx, poly_test_linear_call_body(scheduled_linear, 0), POLY_OP_END);
  int k0_ranges =
      kernel_op_count(ctx, poly_test_linear_call_body(scheduled_linear, 0), POLY_OP_RANGE);
  ASSERT_INT_EQ(k0_stores, 1);
  ASSERT_TRUE(k0_ranges > 0);
  ASSERT_TRUE(k0_ends > 0);
  ASSERT_TRUE(k0_ranges >= k0_ends);

  /* K1 (reduce): reduce_sum(neg(perm(reshape(a))) * c) → out2 */
  PolyUOp *k1 = poly_test_linear_call_body(scheduled_linear, 1);
  int k1_stores = kernel_op_count(ctx, k1, POLY_OP_STORE);
  int k1_ranges = kernel_op_count(ctx, k1, POLY_OP_RANGE);
  int k1_ends = kernel_op_count(ctx, k1, POLY_OP_END);
  int k1_reduces = kernel_op_count(ctx, k1, POLY_OP_REDUCE);
  ASSERT_INT_EQ(k1_stores, 1);
  ASSERT_TRUE(k1_ranges > 0);
  ASSERT_TRUE(k1_ends > 0);
  ASSERT_TRUE(k1_ranges >= k1_ends); /* ranges >= ends (reduce range has no END) */
  ASSERT_INT_EQ(k1_reduces, 1);
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

  PolyUOp *out1_buffer = poly_uop_get_buffer_identity(out_uops[0]);
  PolyUOp *out2_buffer = poly_uop_get_buffer_identity(out_uops[1]);
  ASSERT_NOT_NULL(out1_buffer);
  ASSERT_NOT_NULL(out2_buffer);
  PolyTestBufferView bindings[] = {
      POLY_TEST_HOST_VIEW(out1_buffer, o1_d), POLY_TEST_HOST_VIEW(out2_buffer, o2_d),
      POLY_TEST_HOST_VIEW(a_flat, a_d),       POLY_TEST_HOST_VIEW(b_flat, b_d),
      POLY_TEST_HOST_VIEW(c_flat, c_d),
  };
  int ret = poly_test_run_linear_buffer_views(ctx, scheduled_linear, bindings, 5, vars, n_vars);
  ASSERT_INT_EQ(ret, 0);
  free(vars);

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

  /* Current create_schedule consumes the callified AFTER-state graph.  The
   * raw STORE sink is callified by the public LINEAR boundary first. */
  PolyUOp *linear = poly_test_create_linear(ctx, sink);
  ASSERT_NOT_NULL(linear);
  ASSERT_INT_EQ(linear->n_src, 3);

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
  /* Current tinygrad implicit broadcast: the scalar REDUCE feeds both vector
   * consumers directly; run_rangeify leaves one REDUCE and zero STAGE per CALL. */
  int N = 8;
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_test_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *c0 = poly_test_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *e0 = poly_test_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *oc = poly_test_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *oe = poly_test_buffer(ctx, POLY_FLOAT32, N);

  int64_t axes[] = {0};
  PolyUOp *sum = poly_reduce_axis(ctx, POLY_OP_ADD, a, axes, 1);
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, sum, c0, poly_arg_none());
  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, sum, e0, poly_arg_none());
  PolyUOp *sc = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, oc, add, poly_arg_none());
  PolyUOp *se = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, oe, mul, poly_arg_none());
  PolyUOp *sink = poly_uop(ctx, POLY_OP_SINK, POLY_VOID, (PolyUOp *[]){sc, se}, 2, poly_arg_none());

  PolyUOp *scheduled_linear = poly_test_create_linear(ctx, sink);
  ASSERT_NOT_NULL(scheduled_linear);
  ASSERT_INT_EQ(scheduled_linear->op, POLY_OP_LINEAR);
  ASSERT_INT_EQ(scheduled_linear->n_src, 2);
  for (int i = 0; i < scheduled_linear->n_src; i++) {
    PolyUOp *call = scheduled_linear->src[i];
    ASSERT_INT_EQ(call->op, POLY_OP_CALL);
    ASSERT_INT_EQ(call->src[0]->op, POLY_OP_SINK);
    ASSERT_INT_EQ(count_ops(ctx, call->src[0], POLY_OP_STAGE), 0);
    ASSERT_INT_EQ(count_reduce_arg_kind(ctx, call->src[0], POLY_ARG_REDUCE), 1);
  }
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, callified_param_shared_scalar_reduce_matches_raw_stage_topology) {
  /* Current transform_to_call + run_rangeify keeps the shared scalar REDUCE
   * inline in both consumers: zero STAGE, five INDEX and three RANGE nodes.
   * See temp/reference_upgrade_20260823/tg_shared_scalar_rangeify_current.out. */
  int N = 8;
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_test_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *c = poly_test_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *e = poly_test_buffer(ctx, POLY_FLOAT32, N);
  int64_t axes[] = {0};
  PolyUOp *sum = poly_reduce_axis(ctx, POLY_OP_ADD, a, axes, 1);
  PolyUOp *targets[] = {
      poly_add(ctx, sum, c),
      poly_mul(ctx, sum, e),
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
  ASSERT_INT_EQ(count_reduce_arg_kind(ctx, call->src[0], POLY_ARG_REDUCE), 1);

  PolyUOp *function = poly_apply_earliest_rewrites(ctx, call->src[0]);
  ASSERT_NOT_NULL(function);
  PolyIndexingCtx *ictx = poly_indexing_ctx_new(ctx);
  ASSERT_NOT_NULL(ictx);
  PolyUOp *rangeified = run_apply_rangeify(ictx, function);
  ASSERT_NOT_NULL(rangeified);
  ASSERT_INT_EQ(count_ops(ctx, rangeified, POLY_OP_STAGE), 0);
  ASSERT_INT_EQ(count_reduce_arg_kind(ctx, rangeified, POLY_ARG_REDUCE), 1);
  ASSERT_INT_EQ(count_ops(ctx, rangeified, POLY_OP_AFTER), 2);
  ASSERT_INT_EQ(count_ops(ctx, rangeified, POLY_OP_STORE), 2);
  ASSERT_INT_EQ(count_ops(ctx, rangeified, POLY_OP_INDEX), 5);
  ASSERT_INT_EQ(count_ops(ctx, rangeified, POLY_OP_RANGE), 3);

  poly_indexing_ctx_destroy(ictx);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, get_kernel_graph_owns_kernel_splitting) {
  /* Current tinygrad get_kernel_graph runs split_kernels before returning;
   * create_schedule only orders its CALL/END/AFTER graph. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *a = poly_test_buffer(ctx, POLY_FLOAT32, 4);
  PolyUOp *b = poly_test_buffer(ctx, POLY_FLOAT32, 4);
  PolyUOp *mid = poly_contiguous(ctx, poly_add(ctx, a, b));
  PolyUOp *out = poly_mul(ctx, mid, poly_const_float(ctx, 2.0f));
  PolyUOp *realized = NULL;
  PolyUOp *call = poly_transform_to_call(ctx, &out, 1, &realized);
  ASSERT_NOT_NULL(call);
  ASSERT_INT_EQ(call->op, POLY_OP_CALL);
  ASSERT_INT_EQ(call->src[0]->op, POLY_OP_SINK);

  PolyUOp *kernel_graph = poly_get_kernel_graph(ctx, call->src[0]);
  ASSERT_NOT_NULL(kernel_graph);
  ASSERT_INT_EQ(count_ops(ctx, kernel_graph, POLY_OP_CALL), 2);
  ASSERT_INT_EQ(count_ops(ctx, kernel_graph, POLY_OP_END), 2);
  ASSERT_INT_EQ(count_ops(ctx, kernel_graph, POLY_OP_AFTER), 2);

  PolyUOp *linear = poly_create_schedule(ctx, kernel_graph);
  ASSERT_NOT_NULL(linear);
  ASSERT_INT_EQ(linear->op, POLY_OP_LINEAR);
  ASSERT_INT_EQ(linear->n_src, 2);
  ASSERT_INT_EQ(linear->src[0]->op, POLY_OP_CALL);
  ASSERT_INT_EQ(linear->src[1]->op, POLY_OP_CALL);

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
  PolyDevice device = poly_ctx_get_preferred_device(ctx);
  if (!poly_device_can_execute(device)) device = poly_device_default();

  PolyUOp *a = poly_test_buffer_on_device(ctx, POLY_FLOAT32, N, device);
  PolyUOp *c0 = poly_test_buffer_on_device(ctx, POLY_FLOAT32, N, device);
  PolyUOp *e0 = poly_test_buffer_on_device(ctx, POLY_FLOAT32, N, device);
  PolyUOp *oc = poly_test_buffer_on_device(ctx, POLY_FLOAT32, N, device);
  PolyUOp *oe = poly_test_buffer_on_device(ctx, POLY_FLOAT32, N, device);

  int64_t axes[] = {0};
  PolyUOp *sum = poly_reduce_axis(ctx, POLY_OP_ADD, a, axes, 1);
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, sum, c0, poly_arg_none());
  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, sum, e0, poly_arg_none());
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

  /* Current Tinygrad pm_const_buffer_folding rewrites STAGE(CONST, RANGE)
   * with STAGE.const_like, preserving the one-dimensional shape. */
  PolyUOp *stage_const = poly_uop_const(ctx, poly_arg_float(42.0), POLY_WEAKFLOAT);
  PolyUOp *stage_range = poly_uop_range(ctx, 4, 0, POLY_AXIS_WEAK);
  PolyUOp *stage_src[] = {stage_const, stage_range};
  PolyUOp *stage = poly_uop(
      ctx, POLY_OP_STAGE, POLY_WEAKFLOAT, stage_src, 2,
      poly_arg_bufferize_opts("CPU", POLY_ADDR_GLOBAL, true)
  );
  PolyUOp *folded =
      poly_graph_rewrite_ctx_ex2(ctx, stage, poly_pm_const_buffer_folding(), NULL, false, false);
  ASSERT_NOT_NULL(folded);
  ASSERT_EQ(folded->op, POLY_OP_EXPAND);
  ASSERT_INT_EQ(count_ops(ctx, folded, POLY_OP_CONST), 3);
  ASSERT_INT_EQ(count_ops(ctx, folded, POLY_OP_EXPAND), 1);
  ASSERT_INT_EQ(count_ops(ctx, folded, POLY_OP_RESHAPE), 2);
  ASSERT_INT_EQ(count_ops(ctx, folded, POLY_OP_STACK), 1);

  PolyUOp *a = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CPU);
  PolyUOp *out = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CPU);
  PolyUOp *c42 = poly_uop(ctx, POLY_OP_CONST, POLY_FLOAT32, NULL, 0, poly_arg_float(42.0));
  int64_t one_sh[] = {1};
  int64_t exp_sh[] = {4};
  PolyUOp *reshaped = poly_reshape(ctx, c42, one_sh, 1);
  PolyUOp *expanded = poly_expand(ctx, reshaped, exp_sh, 1);
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, a, expanded, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out, add, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  /* Current Tinygrad's full kernel graph has one fused CALL and no STAGE;
   * batch1_tg_const_stage_probe.py records the shared body vocabulary. */
  PolyUOp *linear = poly_test_create_linear(ctx, sink);
  ASSERT_NOT_NULL(linear);
  ASSERT_INT_EQ(linear->n_src, 1);
  PolyUOp *body = poly_test_linear_call_body(linear, 0);
  ASSERT_NOT_NULL(body);
  ASSERT_INT_EQ(count_ops(ctx, body, POLY_OP_ADD), 1);
  ASSERT_INT_EQ(count_ops(ctx, body, POLY_OP_CONST), 2);
  ASSERT_INT_EQ(count_ops(ctx, body, POLY_OP_END), 1);
  ASSERT_INT_EQ(count_ops(ctx, body, POLY_OP_INDEX), 2);
  ASSERT_INT_EQ(count_ops(ctx, body, POLY_OP_RANGE), 1);
  ASSERT_INT_EQ(count_ops(ctx, body, POLY_OP_STAGE), 0);
  ASSERT_INT_EQ(count_ops(ctx, body, POLY_OP_STORE), 1);

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

TEST(rangeify, after_all_invalid_matches_current_tinygrad) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *buf = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CPU);
  PolyUOp *range = poly_uop_range(ctx, 4, 0, POLY_AXIS_WEAK);
  PolyUOp *target = poly_uop_index(ctx, buf, &range, 1);
  PolyUOp *invalid = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_invalid());
  PolyUOp *after = poly_uop_set(ctx, target, invalid, &range, 1);
  PolyUOp *indexed = poly_uop_index(ctx, after, &range, 1);
  PolyUOp *folded =
      poly_graph_rewrite_ctx_ex2(ctx, indexed, poly_pm_const_buffer_folding(), NULL, false, false);
  ASSERT_NOT_NULL(folded);
  ASSERT_EQ(folded->op, POLY_OP_CONST);
  ASSERT_EQ(folded->arg.kind, POLY_ARG_INVALID);

  /* Partial coverage, another target buffer, and a valid stored value must
   * retain INDEX(AFTER); current Tinygrad rejects all three folds. */
  PolyUOp *partial_range = poly_uop_range(ctx, 2, 1, POLY_AXIS_WEAK);
  PolyUOp *partial_target = poly_uop_index(ctx, buf, &partial_range, 1);
  PolyUOp *partial_after = poly_uop_set(ctx, partial_target, invalid, &partial_range, 1);
  PolyUOp *partial_index = poly_uop_index(ctx, partial_after, &partial_range, 1);
  ASSERT_EQ(
      poly_graph_rewrite_ctx_ex2(
          ctx, partial_index, poly_pm_const_buffer_folding(), NULL, false, false
      )
          ->op,
      POLY_OP_INDEX
  );

  PolyUOp *other = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CPU);
  PolyUOp *other_target = poly_uop_index(ctx, other, &range, 1);
  PolyUOp *other_store = poly_uop_store(ctx, other_target, invalid);
  PolyUOp *other_end = poly_uop_end(ctx, other_store, &range, 1);
  PolyUOp *wrong_after = poly_uop_after(ctx, buf, other_end);
  PolyUOp *wrong_index = poly_uop_index(ctx, wrong_after, &range, 1);
  ASSERT_EQ(
      poly_graph_rewrite_ctx_ex2(
          ctx, wrong_index, poly_pm_const_buffer_folding(), NULL, false, false
      )
          ->op,
      POLY_OP_INDEX
  );

  PolyUOp *zero = poly_uop_const(ctx, poly_arg_float(0.0), POLY_FLOAT32);
  PolyUOp *valid_after = poly_uop_set(ctx, target, zero, &range, 1);
  PolyUOp *valid_index = poly_uop_index(ctx, valid_after, &range, 1);
  ASSERT_EQ(
      poly_graph_rewrite_ctx_ex2(
          ctx, valid_index, poly_pm_const_buffer_folding(), NULL, false, false
      )
          ->op,
      POLY_OP_INDEX
  );

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, moved_const_folding_add_shrunk_zero_e2e) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CPU);
  PolyUOp *zero6 = poly_full(ctx, (int64_t[]){6}, 1, 0.0);
  PolyUOp *zero4 = poly_shrink(ctx, zero6, (int64_t[][2]){{1, 5}}, 1);
  PolyUOp *out = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CPU);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, poly_alu2(ctx, POLY_OP_ADD, a, zero4)));

  /* Port of tinygrad test_const_folding.py::test_add_shrunk_zero. The
   * movement-wrapped zero must behave as an elementwise identity and stay in a
   * single output kernel, not force an intermediate materialization. */
  PolyUOp *linear = poly_test_create_linear(ctx, sink);
  ASSERT_NOT_NULL(linear);
  ASSERT_INT_EQ(linear->n_src, 1);

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

  PolyUOp *a = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CPU);
  PolyUOp *zero2 = poly_full(ctx, (int64_t[]){2}, 1, 0.0);
  PolyUOp *zero4 = poly_pad(ctx, zero2, (int64_t[][2]){{1, 1}}, 1);
  PolyUOp *out = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CPU);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, poly_alu2(ctx, POLY_OP_ADD, a, zero4)));

  /* Port of tinygrad test_const_folding.py::test_add_padded_zero. Padded
   * zeros are valid/index expressions internally, so this catches regressions
   * where identity folding stops at movement boundaries. */
  PolyUOp *linear = poly_test_create_linear(ctx, sink);
  ASSERT_NOT_NULL(linear);
  ASSERT_INT_EQ(linear->n_src, 1);

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

  PolyUOp *a = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CPU);
  PolyUOp *one6 = poly_full(ctx, (int64_t[]){6}, 1, 1.0);
  PolyUOp *one4 = poly_shrink(ctx, one6, (int64_t[][2]){{1, 5}}, 1);
  PolyUOp *out = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CPU);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, poly_alu2(ctx, POLY_OP_MUL, a, one4)));

  /* Port of tinygrad test_const_folding.py::test_mul_shrunk_one. */
  PolyUOp *linear = poly_test_create_linear(ctx, sink);
  ASSERT_NOT_NULL(linear);
  ASSERT_INT_EQ(linear->n_src, 1);

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

  /* Pinned Tensor.empty is BUFFER(shape, ParamArg(device)) followed by movement
   * (uop/ops.py:733-746); full(buffer=False) is a pure CONST graph and is not
   * bindable storage. */
  PolyUOp *empty = poly_reshape(
      ctx, poly_test_buffer_on_device(ctx, POLY_FLOAT32, 0, POLY_DEVICE_CPU), (int64_t[]){1, 0}, 2
  );
  PolyUOp *sum = poly_reduce_axis(ctx, POLY_OP_ADD, empty, (int64_t[]){0, 1}, 2);
  PolyUOp *scalar = poly_reshape(ctx, sum, NULL, 0);
  PolyUOp *out = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 1, POLY_DEVICE_CPU);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, poly_reshape(ctx, out, NULL, 0), scalar));

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
      ctx, poly_test_buffer_on_device(ctx, POLY_INT32, 0, POLY_DEVICE_CPU), (int64_t[]){1, 0}, 2
  );
  PolyUOp *maximum = poly_reduce_axis(ctx, POLY_OP_MAX, empty, (int64_t[]){0, 1}, 2);
  PolyUOp *scalar = poly_reshape(ctx, maximum, NULL, 0);
  PolyUOp *out = poly_test_buffer_on_device(ctx, POLY_INT32, 1, POLY_DEVICE_CPU);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, poly_reshape(ctx, out, NULL, 0), scalar));

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
      ctx, poly_test_buffer_on_device(ctx, POLY_INT32, 0, POLY_DEVICE_CPU), (int64_t[]){2, 0, 3}, 3
  );
  PolyUOp *max_i = poly_reduce_axis(ctx, POLY_OP_MAX, empty_i, (int64_t[]){1}, 1);
  PolyUOp *out_i = poly_test_buffer_on_device(ctx, POLY_INT32, 6, POLY_DEVICE_CPU);
  PolyUOp *store_i = poly_store_val(ctx, poly_reshape(ctx, out_i, (int64_t[]){2, 3}, 2), max_i);

  PolyUOp *empty_f = poly_reshape(
      ctx, poly_test_buffer_on_device(ctx, POLY_FLOAT32, 0, POLY_DEVICE_CPU), (int64_t[]){2, 0, 3},
      3
  );
  PolyUOp *max_f = poly_reduce_axis(ctx, POLY_OP_MAX, empty_f, (int64_t[]){1}, 1);
  PolyUOp *out_f = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 6, POLY_DEVICE_CPU);
  PolyUOp *store_f = poly_store_val(ctx, poly_reshape(ctx, out_f, (int64_t[]){2, 3}, 2), max_f);
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
      ctx, poly_test_buffer_on_device(ctx, POLY_INT32, 0, POLY_DEVICE_CPU), (int64_t[]){2, 0, 3}, 3
  );
  PolyUOp *maximum = poly_reduce_axis(ctx, POLY_OP_MAX, empty, (int64_t[]){1}, 1);
  PolyUOp *rewritten = poly_apply_earliest_rewrites(ctx, maximum);
  ASSERT_NOT_NULL(rewritten);
  ASSERT_EQ(rewritten->op, POLY_OP_EXPAND);
  ASSERT_EQ(rewritten->src[0]->op, POLY_OP_CONST);
  ASSERT_INT_EQ(rewritten->src[0]->arg.kind, POLY_ARG_INT);
  ASSERT_INT_EQ(rewritten->src[0]->arg.i, INT32_MIN);
  PolyShape shape = poly_uop_max_shape_cached(ctx, rewritten);
  ASSERT_INT_EQ(shape.ndim, 2);
  ASSERT_INT_EQ(shape.dims[0], 2);
  ASSERT_INT_EQ(shape.dims[1], 3);

  PolyUOp *zero_output = poly_reshape(
      ctx, poly_test_buffer_on_device(ctx, POLY_INT32, 0, POLY_DEVICE_CPU), (int64_t[]){0, 0, 3}, 3
  );
  PolyUOp *zero_max = poly_reduce_axis(ctx, POLY_OP_MAX, zero_output, (int64_t[]){1}, 1);
  PolyUOp *zero_rewritten = poly_apply_earliest_rewrites(ctx, zero_max);
  ASSERT_NOT_NULL(zero_rewritten);
  ASSERT_EQ(zero_rewritten->op, POLY_OP_EXPAND);
  ASSERT_EQ(zero_rewritten->src[0]->op, POLY_OP_CONST);
  ASSERT_INT_EQ(zero_rewritten->src[0]->arg.kind, POLY_ARG_INT);
  ASSERT_INT_EQ(zero_rewritten->src[0]->arg.i, 0);
  shape = poly_uop_max_shape_cached(ctx, zero_rewritten);
  ASSERT_INT_EQ(shape.ndim, 2);
  ASSERT_INT_EQ(shape.dims[0], 0);
  ASSERT_INT_EQ(shape.dims[1], 3);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, earliest_expand_bitcast_matches_current_lane_topology) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  /* Current expand_bitcast lives in earliest_rewrites, not Tensor
   * construction (schedule/rangeify.py:111-125,178). */
  PolyUOp *bytes = poly_test_buffer_on_device(ctx, POLY_UINT8, 8, POLY_DEVICE_CPU);
  PolyUOp *wide = poly_uop1(ctx, POLY_OP_BITCAST, POLY_UINT32, bytes, poly_arg_none());
  PolyUOp *wide_rewritten = poly_apply_earliest_rewrites(ctx, wide);
  ASSERT_NOT_NULL(wide_rewritten);
  ASSERT_INT_EQ(wide_rewritten->op, POLY_OP_BITCAST);
  ASSERT_INT_EQ(count_ops(ctx, wide_rewritten, POLY_OP_BITCAST), 1);
  ASSERT_INT_EQ(count_ops(ctx, wide_rewritten, POLY_OP_SHRINK), 4);
  ASSERT_INT_EQ(count_ops(ctx, wide_rewritten, POLY_OP_CAST), 4);
  ASSERT_INT_EQ(count_ops(ctx, wide_rewritten, POLY_OP_SHL), 4);
  ASSERT_INT_EQ(count_ops(ctx, wide_rewritten, POLY_OP_ADD), 3);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, wide_rewritten)[0], 2);

  PolyUOp *words = poly_test_buffer_on_device(ctx, POLY_UINT32, 2, POLY_DEVICE_CPU);
  PolyUOp *narrow = poly_uop1(ctx, POLY_OP_BITCAST, POLY_UINT8, words, poly_arg_none());
  PolyUOp *narrow_rewritten = poly_apply_earliest_rewrites(ctx, narrow);
  ASSERT_NOT_NULL(narrow_rewritten);
  ASSERT_INT_EQ(narrow_rewritten->op, POLY_OP_BITCAST);
  ASSERT_INT_EQ(count_ops(ctx, narrow_rewritten, POLY_OP_BITCAST), 1);
  ASSERT_INT_EQ(count_ops(ctx, narrow_rewritten, POLY_OP_SHR), 4);
  ASSERT_INT_EQ(count_ops(ctx, narrow_rewritten, POLY_OP_STACK), 1);
  ASSERT_INT_EQ(count_ops(ctx, narrow_rewritten, POLY_OP_CAST), 1);
  ASSERT_INT_EQ(poly_uop_max_shape_dims(ctx, narrow_rewritten)[0], 8);

  PolyUOp *same = poly_uop1(ctx, POLY_OP_BITCAST, POLY_INT32, words, poly_arg_none());
  ASSERT_PTR_EQ(poly_apply_earliest_rewrites(ctx, same), same);

  PolyUOp *disk = poly_test_buffer_on_device(ctx, POLY_UINT8, 8, POLY_DEVICE_DISK);
  PolyUOp *disk_wide = poly_uop1(ctx, POLY_OP_BITCAST, POLY_UINT32, disk, poly_arg_none());
  ASSERT_PTR_EQ(poly_apply_earliest_rewrites(ctx, disk_wide), disk_wide);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, zero_size_general_rule_preserves_shape_and_retags_only_root) {
  /* Current x.const_like(0).rtag(x.tag) preserves the zero-extent shape. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *shape = poly_const_int(ctx, 0);
  PolyParamArg arg = {.slot = 0, .dtype = POLY_FLOAT32, .addrspace = POLY_ADDR_GLOBAL};
  PolyUOp *tagged =
      poly_uop_tagged(ctx, POLY_OP_PARAM, POLY_FLOAT32, &shape, 1, poly_arg_param(&arg), 77);
  PolyUOp *rewritten = poly_apply_earliest_rewrites(ctx, tagged);
  ASSERT_NOT_NULL(rewritten);
  ASSERT_EQ(rewritten->op, POLY_OP_EXPAND);
  ASSERT_INT_EQ(rewritten->tag, 77);
  ASSERT_TRUE(poly_dtype_eq(rewritten->dtype, POLY_FLOAT32));
  PolyShape result_shape = poly_uop_max_shape_cached(ctx, rewritten);
  ASSERT_INT_EQ(result_shape.ndim, 1);
  ASSERT_INT_EQ(result_shape.dims[0], 0);
  ASSERT_EQ(rewritten->src[0]->op, POLY_OP_CONST);
  ASSERT_INT_EQ(rewritten->src[0]->tag, 0);

  poly_ctx_destroy(ctx);
  PASS();
}

/* Stage C: earliest_rewrites */

TEST(rangeify, earliest_pm_mops_moves_after_before_reshape_cleanup) {
  /* Pinned tinygrad pm_mops first moves the inner RESHAPE through AFTER,
   * preserving every effect. mop_cleanup can then merge the now-adjacent
   * reshapes and remove the merged no-op RESHAPE. Both SINK uses therefore
   * share the tagged base AFTER. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *base = poly_test_buffer(ctx, POLY_FLOAT32, 6);
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
  PolyUOp *out = poly_test_buffer(ctx, POLY_FLOAT32, 6);
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
  ASSERT_PTR_EQ(rewritten_store->src[1], moved_after);
  ASSERT_INT_EQ(count_ops(ctx, rewritten, POLY_OP_RESHAPE), 0);

  PolyIndexingCtx *ictx = poly_indexing_ctx_new(ctx);
  ASSERT_NOT_NULL(ictx);
  poly_realize_map_build(ictx, rewritten);
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

  PolyUOp *base = poly_test_buffer(ctx, POLY_FLOAT32, 6);
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
  PolyUOp *input = poly_test_buffer(ctx, POLY_FLOAT32, 32768);
  PolyUOp *reduce = poly_reduce_axis(ctx, POLY_OP_ADD, input, axes, 1);
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, reduce, poly_arg_none());
  PolyUOp *rewritten = poly_apply_earliest_rewrites(ctx, sink);
  PolyUOp *rerewritten = poly_apply_earliest_rewrites(ctx, rewritten);

  rangeify_restore_env(&size);
  rangeify_restore_env(&threshold);
  rangeify_restore_env(&split);

  ASSERT_NOT_NULL(rewritten);
  ASSERT_EQ(rewritten->op, POLY_OP_SINK);
  ASSERT_INT_EQ(count_reduce_arg_kind(ctx, rewritten, POLY_ARG_REDUCE), 2);
  ASSERT_INT_EQ(count_ops(ctx, rewritten, POLY_OP_CONTIGUOUS), 1);
  ASSERT_INT_EQ(count_ops(ctx, rewritten, POLY_OP_PERMUTE), 1);
  ASSERT_INT_EQ(count_ops(ctx, rewritten, POLY_OP_RESHAPE), 1);

  PolyUOp *second_reduce = rewritten->src[0];
  ASSERT_EQ(second_reduce->op, POLY_OP_REDUCE);
  ASSERT_EQ(second_reduce->arg.kind, POLY_ARG_REDUCE);
  ASSERT_INT_EQ(second_reduce->arg.reduce.num_axes, 1);
  PolyUOp *contiguous = second_reduce->src[0];
  ASSERT_EQ(contiguous->op, POLY_OP_CONTIGUOUS);
  PolyUOp *first_reduce = contiguous->src[0];
  ASSERT_EQ(first_reduce->op, POLY_OP_REDUCE);
  ASSERT_EQ(first_reduce->arg.kind, POLY_ARG_REDUCE);
  ASSERT_INT_EQ(first_reduce->arg.reduce.num_axes, 1);
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
  PolyUOp *input = poly_expand(ctx, poly_test_buffer(ctx, POLY_FLOAT32, 1), expanded_shape, 1);
  PolyUOp *reduce = poly_reduce_axis(ctx, POLY_OP_ADD, input, axes, 1);
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, reduce, poly_arg_none());
  PolyUOp *rewritten = poly_apply_earliest_rewrites(ctx, sink);

  rangeify_restore_env(&threshold);
  rangeify_restore_env(&split);

  ASSERT_INT_EQ(count_reduce_arg_kind(ctx, rewritten, POLY_ARG_REDUCE), 1);
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
  PolyUOp *input = poly_test_buffer(ctx, POLY_FLOAT32, 32768);
  PolyUOp *reduce = poly_reduce_axis(ctx, POLY_OP_ADD, input, axes, 1);
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, reduce, poly_arg_none());
  PolyUOp *rewritten = poly_apply_earliest_rewrites(ctx, sink);

  rangeify_restore_env(&split);

  ASSERT_INT_EQ(count_reduce_arg_kind(ctx, rewritten, POLY_ARG_REDUCE), 1);
  ASSERT_INT_EQ(count_ops(ctx, rewritten, POLY_OP_CONTIGUOUS), 0);
  ASSERT_PTR_EQ(rewritten->src[0], reduce);
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
  PolyUOp *n = poly_uop_variable(ctx, "split_n", 1, 65536, POLY_WEAKINT, 1, false);
  PolyUOp *input = poly_test_buffer_var(ctx, POLY_FLOAT32, n, NULL, 0);
  int64_t axes[] = {0};
  PolyUOp *reduce = poly_reduce_axis(ctx, POLY_OP_ADD, input, axes, 1);
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, reduce, poly_arg_none());
  PolyUOp *rewritten = poly_apply_earliest_rewrites(ctx, sink);

  rangeify_restore_env(&threshold);
  rangeify_restore_env(&split);

  ASSERT_INT_EQ(count_reduce_arg_kind(ctx, rewritten, POLY_ARG_REDUCE), 1);
  ASSERT_INT_EQ(count_ops(ctx, rewritten, POLY_OP_CONTIGUOUS), 0);
  ASSERT_PTR_EQ(rewritten->src[0], reduce);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, earliest_reshape_merge) {
  /* RESHAPE(RESHAPE(x, [4,2]), [8]): verify correct output.
   * poly_earliest_rewrites should merge into RESHAPE(x, [8]). */
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 8, POLY_DEVICE_CPU);
  PolyUOp *out = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 8, POLY_DEVICE_CPU);

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
  PolyUOp *shape = poly_uop(ctx, POLY_OP_STACK, POLY_WEAKINT, shape_src, 1, poly_arg_none());
  PolyParamArg lhs_arg = {.slot = 0, .addrspace = POLY_ADDR_GLOBAL, .device = "CPU"};
  PolyParamArg rhs_arg = {.slot = 1, .addrspace = POLY_ADDR_GLOBAL, .device = "CPU"};
  PolyUOp *lhs_param = poly_uop1(ctx, POLY_OP_PARAM, POLY_FLOAT32, shape, poly_arg_param(&lhs_arg));
  PolyUOp *rhs_param = poly_uop1(ctx, POLY_OP_PARAM, POLY_FLOAT32, shape, poly_arg_param(&rhs_arg));
  PolyUOp *detached = poly_uop1(ctx, POLY_OP_DETACH, POLY_FLOAT32, rhs_param, poly_arg_none());
  PolyUOp *sub = poly_alu2(ctx, POLY_OP_SUB, lhs_param, detached);
  PolyUOp *body = poly_uop1(ctx, POLY_OP_TUPLE, POLY_VOID, sub, poly_arg_none());

  PolyUOp *lhs = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 8, POLY_DEVICE_CPU);
  PolyUOp *rhs = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 8, POLY_DEVICE_CPU);
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

  PolyUOp *out = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 8, POLY_DEVICE_CPU);
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

  PolyUOp *a = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CPU);
  PolyUOp *b = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CPU);
  PolyUOp *out = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CPU);
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

TEST(rangeify, schedule_param_slot_does_not_truncate) {
  /* Pinned pm_post_sched_cache indexes with the full Python integer and
   * raises IndexError for every nonnegative slot outside the call args. */
  const int64_t slots[] = {INT64_C(4294967296), INT64_C(2147483648), INT64_MAX, 1, 0};
  for (int kind = 0; kind < 2; kind++) {
    for (size_t i = 0; i < sizeof(slots) / sizeof(*slots); i++) {
      PolyCtx *ctx = poly_ctx_new();
      ASSERT_NOT_NULL(ctx);
      PolyParamArg arg = {.slot = slots[i], .dtype = POLY_FLOAT32};
      PolyUOp *param = poly_uop0(
          ctx, POLY_OP_PARAM, POLY_FLOAT32, kind ? poly_arg_param(&arg) : poly_arg_int(slots[i])
      );
      PolyUOp *body = poly_uop0(ctx, POLY_OP_NOOP, POLY_VOID, poly_arg_none());
      PolyUOp *inner = poly_uop2(ctx, POLY_OP_CALL, POLY_VOID, body, param, poly_arg_none());
      PolyUOp *linear = poly_uop1(ctx, POLY_OP_LINEAR, POLY_VOID, inner, poly_arg_none());
      PolyUOp *value = poly_const_float(ctx, 7.0f);
      PolyUOp *call = poly_uop2(ctx, POLY_OP_CALL, POLY_VOID, linear, value, poly_arg_none());
      PolyVarBinding *bindings = NULL;
      int n_bindings = 0;
      PolyUOp *ret = poly_create_linear_with_vars(ctx, call, &bindings, &n_bindings);
      bool correct =
          slots[i] == 0 ? ret && ret->n_src == 1 && ret->src[0]->src[1] == value : ret == NULL;
      free(bindings);
      poly_ctx_destroy(ctx);
      ASSERT_TRUE(correct);
    }
  }
  PASS();
}

extern PolyUOp *poly_test_limit_bufs(PolyCtx *ctx, PolyUOp *sink, int fail_after);

TEST(rangeify, limit_bufs_scratch_failure_is_not_success) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *leaf = poly_uop0(ctx, POLY_OP_CUSTOM, POLY_VOID, poly_arg_none());
  PolyUOp *src[40];
  for (int i = 0; i < 40; i++)
    src[i] = leaf;
  PolyUOp *sink = poly_uop(ctx, POLY_OP_SINK, POLY_VOID, src, 40, poly_arg_none());
  ASSERT_NOT_NULL(sink);
  PolyUOp *failed = poly_test_limit_bufs(ctx, sink, 0);
  PolyUOp *retry = poly_test_limit_bufs(ctx, sink, -1);
  bool correct = failed == NULL && retry == sink && sink->n_src == 40;
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(correct);
  PASS();
}

TEST(rangeify, limit_bufs_count_failure_is_not_below_limit) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *a = multi_pm_test_source(ctx, 8001, POLY_FLOAT32, "CPU", 2, 4);
  PolyUOp *b = multi_pm_test_source(ctx, 8002, POLY_FLOAT32, "CPU", 2, 4);
  PolyUOp *sum = poly_add(ctx, a, b);
  ASSERT_NOT_NULL(sum);
  PolyUOp *failed = poly_test_limit_bufs(ctx, sum, 0);
  PolyUOp *retry = poly_test_limit_bufs(ctx, sum, -1);
  bool correct = failed == NULL && retry == sum;
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(correct);
  PASS();
}

TEST(rangeify, limit_bufs_range_id_exhaustion_fails_cleanly) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  RangeifyEnvSave limit = rangeify_save_env("POLY_MAX_KERNEL_BUFFERS");
  setenv("POLY_MAX_KERNEL_BUFFERS", "2", 1);
  PolyUOp *range = poly_uop1(
      ctx, POLY_OP_RANGE, POLY_WEAKINT, poly_const_int(ctx, 2),
      poly_arg_range(INT64_MAX - 1, POLY_AXIS_REDUCE)
  );
  PolyUOp *values[3];
  for (int i = 0; i < 3; i++) {
    PolyUOp *buffer = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 2, POLY_DEVICE_CPU);
    PolyUOp *param = poly_uop_param(ctx, i, buffer);
    values[i] = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, param, range, poly_arg_none());
  }
  PolyUOp *sum = poly_add(ctx, poly_add(ctx, values[0], values[1]), values[2]);
  ASSERT_NOT_NULL(sum);
  PolyUOp *ret = poly_test_limit_bufs(ctx, sum, -1);
  rangeify_restore_env(&limit);
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(ret == NULL);
  PASS();
}

/* Stage 3.25: pm_limit_bufs */

TEST(rangeify, limit_bufs_ir) {
  /* 10 independent buffers chained: b0 + b1 + ... + b9 → STORE(out).
   * With POLY_MAX_KERNEL_BUFFERS=8, the scheduler must split so that
   * every kernel has <= 8 params (7 inputs + 1 output). */
  const int N = 16;
  const int N_BUFS = 10;
  const int N_OUTPUT_BUFS = 1;

  /* Save and set env */
  const char *old = getenv("POLY_MAX_KERNEL_BUFFERS");
  char *olddup = old ? strdup(old) : NULL;
  setenv("POLY_MAX_KERNEL_BUFFERS", "8", 1);

  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *bufs[10];
  for (int i = 0; i < N_BUFS; i++)
    bufs[i] = poly_test_buffer_on_device(ctx, POLY_FLOAT32, N, POLY_DEVICE_CPU);
  PolyUOp *out = poly_test_buffer_on_device(ctx, POLY_FLOAT32, N, POLY_DEVICE_CPU);

  /* Chain: bufs[0] + bufs[1] + ... + bufs[9] */
  PolyUOp *acc = bufs[0];
  for (int i = 1; i < N_BUFS; i++)
    acc = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, acc, bufs[i], poly_arg_none());

  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out, acc, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyUOp *scheduled_linear = poly_test_create_linear(ctx, sink);
  ASSERT_NOT_NULL(scheduled_linear);

  /* Extract results before cleanup */
  int n_kernels = scheduled_linear->n_src;
  /* The fixture owns N_BUFS inputs and one output; any additional BUFFER is an intermediate. */
  int n_intermediates = count_ops(ctx, scheduled_linear, POLY_OP_BUFFER) - (N_BUFS + N_OUTPUT_BUFS);
  bool all_under_limit = true;
  for (int k = 0; k < scheduled_linear->n_src; k++) {
    if ((scheduled_linear->src[k]->n_src - 1) > 8) all_under_limit = false;
  }
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
    bufs[i] = poly_test_buffer_on_device(ctx, POLY_FLOAT32, N, POLY_DEVICE_CPU);
  PolyUOp *out = poly_test_buffer_on_device(ctx, POLY_FLOAT32, N, POLY_DEVICE_CPU);

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
  const int N_INPUT_BUFS = 3;
  const int N_OUTPUT_BUFS = 1;

  const char *old = getenv("POLY_MAX_KERNEL_BUFFERS");
  char *olddup = old ? strdup(old) : NULL;
  setenv("POLY_MAX_KERNEL_BUFFERS", "8", 1);

  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_test_buffer_on_device(ctx, POLY_FLOAT32, N, POLY_DEVICE_CPU);
  PolyUOp *b = poly_test_buffer_on_device(ctx, POLY_FLOAT32, N, POLY_DEVICE_CPU);
  PolyUOp *c = poly_test_buffer_on_device(ctx, POLY_FLOAT32, N, POLY_DEVICE_CPU);
  PolyUOp *out = poly_test_buffer_on_device(ctx, POLY_FLOAT32, N, POLY_DEVICE_CPU);

  PolyUOp *ab = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, a, b, poly_arg_none());
  PolyUOp *abc = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, ab, c, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out, abc, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyUOp *scheduled_linear = poly_test_create_linear(ctx, sink);
  ASSERT_NOT_NULL(scheduled_linear);
  int n_kernels = scheduled_linear->n_src;
  /* Only scheduler-created BUFFERs exceed the fixture's inputs and output. */
  int n_intermediates =
      count_ops(ctx, scheduled_linear, POLY_OP_BUFFER) - (N_INPUT_BUFS + N_OUTPUT_BUFS);
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

TEST(rangeify, limit_bufs_cpu_default_matches_current_tinygrad) {
  /* Current Tinygrad DEVICE_MAX_BUFS limits CPU calls to 31 buffers.  With
   * 33 inputs, the first call consumes 30 inputs plus its output and the
   * second consumes the intermediate, three remaining inputs, and output. */
  const int N_INPUTS = 33;
  const int N = 4;

  const char *old = getenv("POLY_MAX_KERNEL_BUFFERS");
  char *olddup = old ? strdup(old) : NULL;
  unsetenv("POLY_MAX_KERNEL_BUFFERS");

  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *inputs[N_INPUTS];
  for (int i = 0; i < N_INPUTS; i++)
    inputs[i] = poly_test_buffer_on_device(ctx, POLY_FLOAT32, N, POLY_DEVICE_CPU);
  PolyUOp *out = poly_test_buffer_on_device(ctx, POLY_FLOAT32, N, POLY_DEVICE_CPU);
  PolyUOp *acc = inputs[0];
  for (int i = 1; i < N_INPUTS; i++)
    acc = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, acc, inputs[i], poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out, acc, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyUOp *linear = poly_test_create_linear(ctx, sink);
  ASSERT_NOT_NULL(linear);
  ASSERT_INT_EQ(linear->op, POLY_OP_LINEAR);
  ASSERT_INT_EQ(linear->n_src, 2);
  ASSERT_INT_EQ(poly_test_linear_call_n_buffers(linear, 0), 31);
  ASSERT_INT_EQ(poly_test_linear_call_n_buffers(linear, 1), 5);
  ASSERT_INT_EQ(count_ops(ctx, linear, POLY_OP_ADD), 29);
  ASSERT_INT_EQ(count_ops(ctx, linear, POLY_OP_BITCAST), 1);
  ASSERT_INT_EQ(count_ops(ctx, linear, POLY_OP_BUFFER), 35);
  ASSERT_INT_EQ(count_ops(ctx, linear, POLY_OP_CALL), 2);
  ASSERT_INT_EQ(count_ops(ctx, linear, POLY_OP_CONST), 4);
  ASSERT_INT_EQ(count_ops(ctx, linear, POLY_OP_END), 2);
  ASSERT_INT_EQ(count_ops(ctx, linear, POLY_OP_INDEX), 31);
  ASSERT_INT_EQ(count_ops(ctx, linear, POLY_OP_PARAM), 31);
  ASSERT_INT_EQ(count_ops(ctx, linear, POLY_OP_RANGE), 1);
  ASSERT_INT_EQ(count_ops(ctx, linear, POLY_OP_SHRINK), 1);
  ASSERT_INT_EQ(count_ops(ctx, linear, POLY_OP_SINK), 2);
  ASSERT_INT_EQ(count_ops(ctx, linear, POLY_OP_STORE), 2);
  poly_ctx_destroy(ctx);

  if (olddup) {
    setenv("POLY_MAX_KERNEL_BUFFERS", olddup, 1);
    free(olddup);
  }
  PASS();
}

TEST(rangeify, limit_bufs_x86_uses_tinygrad_cpu_limit) {
  /* tinygrad CPU:X86 retains device CPU, so DEVICE_MAX_BUFS=31 applies. */
  const int N_INPUTS = 33;
  const int N = 4;
  const char *old = getenv("POLY_MAX_KERNEL_BUFFERS");
  char *olddup = old ? strdup(old) : NULL;
  unsetenv("POLY_MAX_KERNEL_BUFFERS");

  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *inputs[N_INPUTS];
  for (int i = 0; i < N_INPUTS; i++)
    inputs[i] = poly_test_buffer_on_device(ctx, POLY_FLOAT32, N, POLY_DEVICE_X86);
  PolyUOp *out = poly_test_buffer_on_device(ctx, POLY_FLOAT32, N, POLY_DEVICE_X86);
  PolyUOp *acc = inputs[0];
  for (int i = 1; i < N_INPUTS; i++)
    acc = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, acc, inputs[i], poly_arg_none());
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, acc));

  PolyUOp *linear = poly_test_create_linear(ctx, sink);
  ASSERT_NOT_NULL(linear);
  ASSERT_INT_EQ(linear->n_src, 2);
  ASSERT_INT_EQ(poly_test_linear_call_n_buffers(linear, 0), 31);
  ASSERT_INT_EQ(poly_test_linear_call_n_buffers(linear, 1), 5);
  poly_ctx_destroy(ctx);

  if (olddup) {
    setenv("POLY_MAX_KERNEL_BUFFERS", olddup, 1);
    free(olddup);
  }
  PASS();
}

TEST(rangeify, limit_bufs_cpu_default_below_limit) {
  /* Ten inputs plus one output stay below Tinygrad's CPU limit of 31. */
  const int N = 16;
  const int N_BUFS = 10;

  /* Ensure env var is unset */
  const char *old = getenv("POLY_MAX_KERNEL_BUFFERS");
  char *olddup = old ? strdup(old) : NULL;
  unsetenv("POLY_MAX_KERNEL_BUFFERS");

  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *bufs[10];
  for (int i = 0; i < N_BUFS; i++)
    bufs[i] = poly_test_buffer_on_device(ctx, POLY_FLOAT32, N, POLY_DEVICE_CPU);
  PolyUOp *out = poly_test_buffer_on_device(ctx, POLY_FLOAT32, N, POLY_DEVICE_CPU);

  PolyUOp *acc = bufs[0];
  for (int i = 1; i < N_BUFS; i++)
    acc = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, acc, bufs[i], poly_arg_none());

  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out, acc, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyUOp *scheduled_linear = poly_test_create_linear(ctx, sink);
  ASSERT_NOT_NULL(scheduled_linear);
  int n_kernels = scheduled_linear->n_src;
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

  PolyUOp *buf_a = poly_test_buffer_on_device(ctx, POLY_FLOAT32, N, POLY_DEVICE_CPU);
  PolyUOp *buf_b = poly_test_buffer_on_device(ctx, POLY_FLOAT32, N, POLY_DEVICE_CPU);

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

  PolyUOp *buf_a = poly_test_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *buf_b = poly_test_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, buf_a, buf_b, poly_arg_none());
  PolyUOp *assign = poly_store_buffer_update(ctx, buf_a, add);
  PolyUOp *sink = poly_uop(ctx, POLY_OP_SINK, POLY_VOID, &assign, 1, poly_arg_none());

  PolyUOp *scheduled_linear = poly_test_create_linear(ctx, sink);
  ASSERT_NOT_NULL(scheduled_linear);

  int got_kernels = scheduled_linear->n_src;

  /* Check that the update kernel writes to buf_a (existing buffer). */
  bool writes_buf_a = false;
  for (int k = 0; k < scheduled_linear->n_src; k++) {
    for (int p = 0; p < (scheduled_linear->src[k]->n_src - 1); p++) {
      if (scheduled_linear->src[k]->src[p + 1] == buf_a) writes_buf_a = true;
    }
  }
  poly_ctx_destroy(ctx);

  /* 1 in-place update kernel, 0 consumer stores, 0 intermediates */
  ASSERT_INT_EQ(got_kernels, 1);
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

  PolyUOp *buf_a = poly_test_buffer_on_device(ctx, POLY_FLOAT32, N, POLY_DEVICE_CPU);
  PolyUOp *buf_out = poly_test_buffer_on_device(ctx, POLY_FLOAT32, N, POLY_DEVICE_CPU);

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

  PolyUOp *buf_a = poly_test_buffer_on_device(ctx, POLY_FLOAT32, N, POLY_DEVICE_CPU);
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

/* Dynamic ALU BUFFER shape-variable tests */

/* define_var_1d_e2e: a[N] + 1.0 -> out[N] with N=4 */
TEST(rangeify, define_var_1d_e2e) {
  PolyCtx *ctx = poly_ctx_new();

  /* Create symbolic variable N with bounds [1, 16] */
  PolyUOp *N = poly_uop_variable(ctx, "N", 1, 16, POLY_WEAKINT, 1, false);

  /* Create dynamic 1D buffers: a[N], out[N] */
  int float_id = poly_dtype_id_by_name("float");
  PolyUOp *buf_a = poly_test_buffer_var_by_id(ctx, float_id, N, NULL, 0, POLY_DEVICE_CPU);
  PolyUOp *buf_out = poly_test_buffer_var_by_id(ctx, float_id, N, NULL, 0, POLY_DEVICE_CPU);

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
      POLY_TEST_HOST_VIEW(poly_uop_base(buf_a), a_data),
      POLY_TEST_HOST_VIEW(poly_uop_base(buf_out), out_data),
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

  PolyUOp *N = poly_uop_variable(ctx, "N", 1, 8, POLY_WEAKINT, 1, false);

  /* Create 2D dynamic buffers: a[N,4], out[N,4] */
  int64_t inner_dim = 4;
  int float_id = poly_dtype_id_by_name("float");
  PolyUOp *buf_a = poly_test_buffer_var_by_id(ctx, float_id, N, &inner_dim, 1, POLY_DEVICE_CPU);
  PolyUOp *buf_out = poly_test_buffer_var_by_id(ctx, float_id, N, &inner_dim, 1, POLY_DEVICE_CPU);

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
      POLY_TEST_HOST_VIEW(poly_uop_base(buf_a), a_data),
      POLY_TEST_HOST_VIEW(poly_uop_base(buf_out), out_data),
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

/* bind_auto_extract: AFTER(N, STORE(N, 4)) in a buffer source, with no explicit
 * var_bindings. Unbinding extracts N=4 before scheduling. */
TEST(rangeify, bind_auto_extract) {
  PolyCtx *ctx = poly_ctx_new();

  /* Create symbolic variable N and bind it to 4. */
  PolyUOp *N = poly_uop_variable(ctx, "N", 1, 16, POLY_WEAKINT, 1, false);
  PolyUOp *bind_N = poly_uop_bind(ctx, N, 4);

  /* Use the binding as the dynamic dimension. Scheduling unbinds it to N. */
  int float_id = poly_dtype_id_by_name("float");
  PolyUOp *buf_a = poly_test_buffer_var_by_id(ctx, float_id, bind_N, NULL, 0, POLY_DEVICE_CPU);
  PolyUOp *buf_out = poly_test_buffer_var_by_id(ctx, float_id, bind_N, NULL, 0, POLY_DEVICE_CPU);

  /* out = a + 1.0 */
  PolyUOp *one = poly_const_float(ctx, 1.0);
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, buf_a, one, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, buf_out, add, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyVarBinding *scheduled_vars = NULL;
  int n_scheduled_vars = 0;
  PolyUOp *linear = poly_linear_effect_sink(ctx, sink, &scheduled_vars, &n_scheduled_vars);
  bool saw_symbol = false;
  int64_t symbol_slot = INT64_MIN;
  const char *symbol_name = NULL;
  if (linear && linear->n_src == 1 && linear->src[0]->op == POLY_OP_CALL) {
    int n_topo = 0;
    PolyUOp **topo = poly_toposort_alloc(ctx, linear->src[0]->src[0], &n_topo);
    for (int i = 0; i < n_topo; i++) {
      PolyUOp *u = topo[i];
      if (poly_uop_is_alu_param(u)) {
        saw_symbol = true;
        symbol_slot = u->arg.param->slot;
        symbol_name = u->arg.param->name;
      }
    }
    poly_toposort_free(topo);
  }
  int scheduled_count = n_scheduled_vars;
  const char *scheduled_name = scheduled_count == 1 ? poly_uop_expr(scheduled_vars[0].var) : NULL;
  int64_t scheduled_value = scheduled_count == 1 ? scheduled_vars[0].value : INT64_MIN;
  free(scheduled_vars);
  ASSERT_NOT_NULL(linear);
  ASSERT_TRUE(saw_symbol);
  ASSERT_INT_EQ(symbol_slot, -1);
  ASSERT_STR_EQ(symbol_name, "N");
  ASSERT_INT_EQ(scheduled_count, 1);
  ASSERT_STR_EQ(scheduled_name, "N");
  ASSERT_INT_EQ(scheduled_value, 4);

  /* Execute without explicit var_bindings; auto-extraction provides N=4. */
  float a_data[16] = {10.0f, 20.0f, 30.0f, 40.0f};
  float out_data[16];
  memset(out_data, 0, sizeof(out_data));

  PolyTestBufferView bindings[2] = {
      POLY_TEST_HOST_VIEW(poly_uop_base(buf_a), a_data),
      POLY_TEST_HOST_VIEW(poly_uop_base(buf_out), out_data),
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

TEST(rangeify, find_bufs_rejects_mixed_index_modes) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *src = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CPU);
  PolyUOp *dst = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 1, POLY_DEVICE_CPU);
  PolyUOp *zero = poly_const_int(ctx, 0);
  PolyUOp *one = poly_const_int(ctx, 1);
  int64_t shape[] = {2, 2};
  PolyUOp *reshaped = poly_reshape(ctx, src, shape, 2);
  PolyUOp *direct = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, src, zero, poly_arg_none());
  PolyUOp *direct_one = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, src, one, poly_arg_none());
  PolyUOp *reshape_src[] = {reshaped, zero, zero};
  PolyUOp *through_reshape =
      poly_uop(ctx, POLY_OP_INDEX, POLY_FLOAT32, reshape_src, 3, poly_arg_none());
  PolyUOp *dst_index = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, dst, zero, poly_arg_none());
  PolyUOp *same_value = poly_add(ctx, direct, direct_one);
  PolyUOp *mixed_value = poly_add(ctx, direct, through_reshape);
  PolyUOp *same_store = poly_store_val(ctx, dst_index, same_value);
  PolyUOp *mixed_store = poly_store_val(ctx, dst_index, mixed_value);

  ASSERT_TRUE(poly_find_bufs(ctx, same_store));
  ASSERT_FALSE(poly_find_bufs(ctx, mixed_store));
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, find_bufs_keeps_opaque_call_bodies_separate) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *src = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CPU);
  PolyUOp *dst = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 1, POLY_DEVICE_CPU);
  PolyUOp *zero = poly_const_int(ctx, 0);
  int64_t shape[] = {2, 2};
  PolyUOp *reshaped = poly_reshape(ctx, src, shape, 2);
  PolyUOp *direct = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, src, zero, poly_arg_none());
  PolyUOp *reshape_src[] = {reshaped, zero, zero};
  PolyUOp *through_reshape =
      poly_uop(ctx, POLY_OP_INDEX, POLY_FLOAT32, reshape_src, 3, poly_arg_none());
  PolyUOp *dst_index = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, dst, zero, poly_arg_none());
  PolyUOp *inner_store = poly_store_val(ctx, dst_index, through_reshape);
  PolyUOp *body = poly_test_kernel_sink(ctx, &inner_store, 1, "opaque_index_mode");
  PolyUOp *call_src[] = {body, src, dst};
  PolyUOp *call = poly_uop(ctx, POLY_OP_CALL, POLY_VOID, call_src, 3, poly_arg_none());

  /* Tinygrad@2026-08-22/a9069c177a9d UOp.toposort defaults to
   * enter_calls=False: this consumer's direct INDEX cannot conflict with the
   * independent reshape INDEX inside the opaque kernel body. */
  PolyUOp *outer_src[] = {dst_index, direct, call};
  PolyUOp *outer_store = poly_uop(ctx, POLY_OP_STORE, POLY_VOID, outer_src, 3, poly_arg_none());
  ASSERT_TRUE(poly_find_bufs(ctx, outer_store));

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, spec_kernel_graph_rejects_unsupported_direct_node) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *src = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CPU);
  PolyUOp *dst = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CPU);
  PolyUOp *valid = poly_get_kernel_graph(ctx, poly_sink1(ctx, poly_store_val(ctx, dst, src)));
  ASSERT_NOT_NULL(valid);
  ASSERT_TRUE(poly_type_verify_kernel_graph(ctx, valid));

  PolyUOp *noop = poly_uop0(ctx, POLY_OP_NOOP, POLY_VOID, poly_arg_none());
  ASSERT_FALSE(poly_type_verify_kernel_graph(ctx, poly_sink1(ctx, noop)));

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, schedule_cache_reuses_normalized_sink) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *a = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CPU);
  PolyUOp *b = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CPU);
  PolyUOp *requested[] = {poly_add(ctx, a, b)};
  PolyUOp *outputs[1] = {NULL};
  PolyUOp *big_call = poly_transform_to_call(ctx, requested, 1, outputs);
  ASSERT_NOT_NULL(big_call);

  PolyVarBinding *first_vars = NULL, *second_vars = NULL;
  int n_first_vars = 0, n_second_vars = 0;
  ASSERT_NOT_NULL(poly_create_linear_with_vars(ctx, big_call, &first_vars, &n_first_vars));
  ASSERT_NOT_NULL(poly_create_linear_with_vars(ctx, big_call, &second_vars, &n_second_vars));
  ASSERT_INT_EQ((int)poly_schedule_cache_len(ctx), 1);

  free(first_vars);
  free(second_vars);
  poly_ctx_destroy(ctx);
  PASS();
}

/* define_var_cache_hit: execute with N=4 then N=8 (reuses compiled kernel).
 * Second call should hit schedule cache and produce correct results. */
TEST(rangeify, define_var_cache_hit) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *N = poly_uop_variable(ctx, "N", 1, 16, POLY_WEAKINT, 1, false);
  int float_id = poly_dtype_id_by_name("float");
  PolyUOp *buf_a = poly_test_buffer_var_by_id(ctx, float_id, N, NULL, 0, POLY_DEVICE_CPU);
  PolyUOp *buf_out = poly_test_buffer_var_by_id(ctx, float_id, N, NULL, 0, POLY_DEVICE_CPU);

  PolyUOp *one = poly_const_float(ctx, 1.0);
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, buf_a, one, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, buf_out, add, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  float a_data[16];
  float out_data[16];
  for (int i = 0; i < 16; i++)
    a_data[i] = (float)(i * 10);

  PolyTestBufferView bindings[2] = {
      POLY_TEST_HOST_VIEW(poly_uop_base(buf_a), a_data),
      POLY_TEST_HOST_VIEW(poly_uop_base(buf_out), out_data),
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

  /* Current raw UOp._rop removes requested singleton axes: (1,4) -> (4). */
  int64_t ax0[] = {0};
  PolyUOp *r0 = poly_reduce_axis(ctx, POLY_OP_ADD, mul, ax0, 1);
  ASSERT_NOT_NULL(r0);

  /* The remaining real axis is now axis 0: (4) -> scalar. */
  PolyUOp *r1 = poly_reduce_axis(ctx, POLY_OP_ADD, r0, ax0, 1);
  ASSERT_NOT_NULL(r1);

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

/* Current STORE normalization in earliest_rewrites. */

TEST(rangeify, earliest_store_bitcast_matches_current) {
  /* Current rangeify.py:175-176:
   * STORE(BITCAST(buf, uint32), value) → STORE(buf, BITCAST(value, float32)). */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *buf = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 4, POLY_DEVICE_CPU);
  PolyUOp *src = poly_test_buffer_on_device(ctx, POLY_UINT32, 4, POLY_DEVICE_CPU);
  PolyUOp *bc_target =
      poly_uop1(ctx, POLY_OP_BITCAST, POLY_UINT32, buf, poly_arg_dtype(POLY_UINT32));
  PolyUOp *store = poly_store_val(ctx, bc_target, src);
  PolyUOp *sink = poly_sink1(ctx, store);

  PolyUOp *rewritten = poly_apply_earliest_rewrites(ctx, sink);
  ASSERT_NOT_NULL(rewritten);
  int n_topo = 0;
  PolyUOp **topo = poly_toposort_alloc(ctx, rewritten, &n_topo);
  PolyUOp *rewritten_store = NULL;
  for (int i = 0; i < n_topo; i++)
    if (topo[i]->op == POLY_OP_STORE) rewritten_store = topo[i];
  ASSERT_NOT_NULL(rewritten_store);
  ASSERT_TRUE(rewritten_store->src[0] == buf);
  ASSERT_INT_EQ(rewritten_store->src[1]->op, POLY_OP_BITCAST);
  ASSERT_TRUE(poly_dtype_eq(rewritten_store->src[1]->dtype, POLY_FLOAT32));
  ASSERT_TRUE(rewritten_store->src[1]->src[0] == src);
  poly_toposort_free(topo);

  float buf_d[] = {1, 2, 3, 4};
  uint32_t src_d[] = {0x40800000u, 0x40400000u, 0x40000000u, 0x3f800000u};
  PolyTestBufferView bindings[] = {
      POLY_TEST_HOST_VIEW(buf, buf_d),
      POLY_TEST_HOST_VIEW(src, src_d),
  };
  int ret = poly_test_realize_buffer_views(ctx, sink, bindings, 2);
  poly_ctx_destroy(ctx);
  ASSERT_INT_EQ(ret, 0);
  ASSERT_FLOAT_EQ(buf_d[0], 4.0f, 0.0f);
  ASSERT_FLOAT_EQ(buf_d[1], 3.0f, 0.0f);
  ASSERT_FLOAT_EQ(buf_d[2], 2.0f, 0.0f);
  ASSERT_FLOAT_EQ(buf_d[3], 1.0f, 0.0f);
  PASS();
}

TEST(rangeify, earliest_assign_to_contiguous) {
  /* poly_store_buffer_update() normalizes RESHAPE(buf) target to base BUFFER and
   * reshapes value to flat shape. The STORE writes to buf in-place. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *buf = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 8, POLY_DEVICE_CPU);
  PolyUOp *src = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 8, POLY_DEVICE_CPU);

  /* Reshape target: view buf as [4,2] — base is BUFFER, safe */
  int64_t sh42[] = {4, 2};
  PolyUOp *reshaped = poly_reshape(ctx, buf, sh42, 2);

  /* Tinygrad 2026-08-22/a9069c177a9d broadcasts the scalar directly. A
   * scalar RESHAPE to (8,) is invalid because it changes cardinality. */
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(1.0));
  PolyUOp *value = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, src, one, poly_arg_none());

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

/* Current tinygrad uop/ops.py:range_start. */

TEST(rangeify, range_start_all) {
  ASSERT_INT_EQ(poly_range_start(POLY_OP_STAGE), 1);
  ASSERT_INT_EQ(poly_range_start(POLY_OP_REDUCE), 1);
  ASSERT_INT_EQ(poly_range_start(POLY_OP_WMMA), 3);
  ASSERT_INT_EQ(poly_range_start(POLY_OP_END), 1);
  ASSERT_INT_EQ(poly_range_start(POLY_OP_CALL), 1);
  ASSERT_INT_EQ(poly_range_start(POLY_OP_FUNCTION), 1);
  ASSERT_INT_EQ(poly_range_start(POLY_OP_LINEAR), 0);
  ASSERT_INT_EQ(poly_range_start(POLY_OP_STORE), -1);
  ASSERT_INT_EQ(poly_range_start(POLY_OP_COPY), -1);
  ASSERT_INT_EQ(poly_range_start(POLY_OP_ADD), -1);
  ASSERT_INT_EQ(poly_range_start(POLY_OP_CONST), -1);
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

  PolyUOp *buf_a = poly_test_buffer_on_device(ctx, POLY_FLOAT32, 8, POLY_DEVICE_CPU);

  /* a[3:8] — source SHRINK */
  int64_t src_pairs[][2] = {{3, 8}};
  PolyUOp *a_3_8 = poly_shrink(ctx, buf_a, src_pairs, 1);

  /* a[:5] — target SHRINK */
  int64_t tgt_pairs[][2] = {{0, 5}};
  PolyUOp *a_0_5 = poly_shrink(ctx, buf_a, tgt_pairs, 1);

  PolyUOp *store = poly_store_val(ctx, a_0_5, a_3_8);
  PolyUOp *after = poly_uop2(ctx, POLY_OP_AFTER, POLY_FLOAT32, a_0_5, store, poly_arg_none());
  PolyUOp *sink = poly_sink1(ctx, after);

  PolyUOp *scheduled_linear = poly_test_create_linear(ctx, sink);
  ASSERT_NOT_NULL(scheduled_linear);

  /* The CONTIGUOUS insertion should produce 2+ kernels:
   * one for materializing the source, one for the ASSIGN write. */
  ASSERT_TRUE(scheduled_linear->n_src >= 2);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, assign_shrink_no_hazard_different_buf) {
  /* a[:3].assign(b[:3]) — SHRINK on target but source is different buffer.
   * Source SHRINK does NOT reach target_base in backward slice.
   * No hazard, so CONTIGUOUS should not be inserted for hazard reasons.
   * (CONTIGUOUS may appear for other scheduling reasons like C4e.) */
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *buf_a = poly_test_buffer(ctx, POLY_FLOAT32, 8);
  PolyUOp *buf_b = poly_test_buffer(ctx, POLY_FLOAT32, 8);

  int64_t src_pairs[][2] = {{0, 3}};
  PolyUOp *b_0_3 = poly_shrink(ctx, buf_b, src_pairs, 1);

  int64_t tgt_pairs[][2] = {{0, 3}};
  PolyUOp *a_0_3 = poly_shrink(ctx, buf_a, tgt_pairs, 1);

  PolyUOp *store = poly_store_val(ctx, a_0_3, b_0_3);
  PolyUOp *after = poly_uop2(ctx, POLY_OP_AFTER, POLY_FLOAT32, a_0_3, store, poly_arg_none());
  PolyUOp *sink = poly_sink1(ctx, after);

  /* This should schedule successfully */
  PolyUOp *scheduled_linear = poly_test_create_linear(ctx, sink);
  ASSERT_NOT_NULL(scheduled_linear);
  ASSERT_TRUE(scheduled_linear->n_src >= 1);
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

  PolyUOp *buf = poly_test_buffer(ctx, POLY_FLOAT32, 4);
  PolyUOp *src = poly_test_buffer(ctx, POLY_FLOAT32, 4);

  PolyUOp *cpu = poly_device_uop_from_name(ctx, "CPU");
  PolyUOp *copy = poly_copy_to_device_uop(ctx, src, cpu);
  PolyUOp *store = poly_store_val(ctx, buf, copy);
  PolyUOp *after = poly_uop2(ctx, POLY_OP_AFTER, POLY_FLOAT32, buf, store, poly_arg_none());
  PolyUOp *sink = poly_sink1(ctx, after);

  poly_realize_map_build(ictx, sink);

  /* COPY should NOT be realized (un-realized by realize_assign_src) */
  ASSERT_FALSE(poly_is_realized(ictx, copy));
  ASSERT_TRUE(poly_is_realized(ictx, store));

  poly_indexing_ctx_destroy(ictx);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(rangeify, realize_assign_src_war_forces_realize) {
  /* ASSIGN(buf, buf + 1) — target base appears in RHS backward slice.
   * WAR hazard: RHS should be force-realized. */
  PolyCtx *ctx = poly_ctx_new();
  PolyIndexingCtx *ictx = poly_indexing_ctx_new(ctx);

  PolyUOp *buf = poly_test_buffer(ctx, POLY_FLOAT32, 4);
  PolyUOp *one = poly_const_float(ctx, 1.0);
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, buf, one, poly_arg_none());

  PolyUOp *store = poly_store_val(ctx, buf, add);
  PolyUOp *after = poly_uop2(ctx, POLY_OP_AFTER, POLY_FLOAT32, buf, store, poly_arg_none());
  PolyUOp *sink = poly_sink1(ctx, after);

  poly_realize_map_build(ictx, sink);

  /* RHS (add) should be realized due to WAR hazard */
  ASSERT_TRUE(poly_is_realized(ictx, add));

  poly_indexing_ctx_destroy(ictx);
  poly_ctx_destroy(ctx);
  PASS();
}
