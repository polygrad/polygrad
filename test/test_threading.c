/*
 * test_threading.c -- focused threading-contract smoke tests.
 */

#define _POSIX_C_SOURCE 200809L
#include "test_harness.h"
#include "../src/schedule/rangeify.h"
#include <pthread.h>

typedef struct {
  int tid;
  bool ok;
  int64_t min_v;
  int64_t max_v;
} ThreadingResult;

static void *threading_context_worker(void *opaque) {
  ThreadingResult *r = opaque;
  r->ok = false;

  PolyCtx *ctx = poly_ctx_new();
  if (!ctx) return NULL;

  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(8));
  PolyUOp *v = poly_uop1(
      ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_range(r->tid + 1, POLY_AXIS_LOOP)
  );
  PolyUOp *two = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(2));
  PolyUOp *three = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(3));
  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, POLY_INT32, v, two, poly_arg_none());
  PolyUOp *expr = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, mul, three, poly_arg_none());

  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, 16);
  PolyUOp *b = poly_buffer(ctx, POLY_FLOAT32, 16);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, 16);
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, a, b, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out, add, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  if (!bound || !v || !two || !three || !mul || !expr || !a || !b || !out || !add || !store ||
      !sink)
    goto done;

  poly_uop_minmax(ctx, expr, &r->min_v, &r->max_v);

  poly_rangeify_stats_reset();
  PolyIndexingCtx *ictx = poly_indexing_ctx_new(ctx);
  if (!ictx) goto done;
  PolyUOp *rangeified = poly_run_rangeify(ictx, sink);
  poly_indexing_ctx_destroy(ictx);
  if (!rangeified) goto done;

  r->ok = true;

done:
  poly_ctx_destroy(ctx);
  return NULL;
}

TEST(threading, independent_contexts_smoke) {
  PolyRangeifyStats main_before = poly_rangeify_stats_get();

  ThreadingResult a = {.tid = 0};
  ThreadingResult b = {.tid = 1};
  pthread_t ta, tb;
  ASSERT_INT_EQ(pthread_create(&ta, NULL, threading_context_worker, &a), 0);
  ASSERT_INT_EQ(pthread_create(&tb, NULL, threading_context_worker, &b), 0);
  ASSERT_INT_EQ(pthread_join(ta, NULL), 0);
  ASSERT_INT_EQ(pthread_join(tb, NULL), 0);

  ASSERT_TRUE(a.ok);
  ASSERT_TRUE(b.ok);
  ASSERT_INT_EQ(a.min_v, 3);
  ASSERT_INT_EQ(b.min_v, 3);

  PolyRangeifyStats main_after = poly_rangeify_stats_get();
  ASSERT_INT_EQ(main_after.buffer_alt_created, main_before.buffer_alt_created);
  ASSERT_INT_EQ(main_after.buffer_alt_max_count, main_before.buffer_alt_max_count);
  PASS();
}
