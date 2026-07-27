/*
 * test_threading.c -- focused threading-contract smoke tests.
 */

#define _POSIX_C_SOURCE 200809L
#include "test_harness.h"
#include "../src/pat.h"
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

typedef struct {
  pthread_mutex_t mutex;
  pthread_cond_t cond;
  int arrived;
  int generation;
  bool stopped;
} ThreadingLowerGate;

typedef struct {
  ThreadingLowerGate *gate;
  PolyOps expected_op;
  PolyOps other_op;
  bool ok;
} ThreadingLowerResult;

static bool threading_lower_gate_wait(ThreadingLowerGate *gate) {
  if (pthread_mutex_lock(&gate->mutex) != 0) return false;
  if (gate->stopped) {
    pthread_mutex_unlock(&gate->mutex);
    return false;
  }
  int generation = gate->generation;
  gate->arrived++;
  if (gate->arrived == 2) {
    gate->arrived = 0;
    gate->generation++;
    pthread_cond_broadcast(&gate->cond);
  } else {
    while (gate->generation == generation && !gate->stopped) {
      if (pthread_cond_wait(&gate->cond, &gate->mutex) != 0) {
        pthread_mutex_unlock(&gate->mutex);
        return false;
      }
    }
  }
  bool ok = !gate->stopped;
  return pthread_mutex_unlock(&gate->mutex) == 0 && ok;
}

static bool threading_linear_has_only_expected_op(
    PolyCtx *ctx,
    PolyUOp *linear,
    PolyOps expected_op,
    PolyOps other_op
) {
  int n_topo = 0;
  PolyUOp **topo = poly_toposort_ex_alloc(ctx, linear, &n_topo, NULL, true);
  if (!topo) return false;
  bool has_expected = false, has_other = false;
  for (int i = 0; i < n_topo; i++) {
    bool is_float32 = poly_dtype_eq(topo[i]->dtype, POLY_FLOAT32);
    has_expected |= is_float32 && topo[i]->op == expected_op;
    has_other |= is_float32 && topo[i]->op == other_op;
  }
  poly_toposort_free(topo);
  return has_expected && !has_other;
}

static void *threading_scache_disabled_worker(void *opaque) {
  ThreadingLowerResult *r = opaque;
  r->ok = false;

  PolyCtx *ctx = poly_ctx_new();
  if (!ctx) return NULL;
  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, 16);
  PolyUOp *b = poly_buffer(ctx, POLY_FLOAT32, 16);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, 16);
  PolyUOp *value = poly_uop2(ctx, r->expected_op, POLY_FLOAT32, a, b, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out, value, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());
  if (!a || !b || !out || !value || !store || !sink) goto done;

  bool all_ok = true;
  for (int i = 0; i < 5000; i++) {
    if (!threading_lower_gate_wait(r->gate)) goto done;
    PolyUOp *linear = poly_lower_sink_to_linear(ctx, sink, POLY_MODE_CALL);
    if (!linear || linear->op != POLY_OP_LINEAR ||
        !threading_linear_has_only_expected_op(ctx, linear, r->expected_op, r->other_op) ||
        poly_schedule_cache_len(ctx) != 0)
      all_ok = false;
  }
  r->ok = all_ok;

done:
  poly_ctx_destroy(ctx);
  return NULL;
}

TEST(threading, independent_contexts_scache_disabled_lowering) {
  const char *old_scache = getenv("POLY_SCACHE");
  char *saved_scache = old_scache ? strdup(old_scache) : NULL;
  if (old_scache && !saved_scache) FAIL("failed to save POLY_SCACHE");
  if (setenv("POLY_SCACHE", "0", 1) != 0) {
    free(saved_scache);
    FAIL("failed to set POLY_SCACHE=0");
  }

  ThreadingLowerGate gate = {.arrived = 0, .generation = 0, .stopped = false};
  int mutex_rc = pthread_mutex_init(&gate.mutex, NULL);
  int cond_rc = mutex_rc == 0 ? pthread_cond_init(&gate.cond, NULL) : -1;
  ThreadingLowerResult a = {
      .gate = &gate, .expected_op = POLY_OP_ADD, .other_op = POLY_OP_MUL,
  };
  ThreadingLowerResult b = {
      .gate = &gate, .expected_op = POLY_OP_MUL, .other_op = POLY_OP_ADD,
  };
  pthread_t ta, tb;
  int create_a = cond_rc == 0 ? pthread_create(&ta, NULL, threading_scache_disabled_worker, &a) : -1;
  int create_b = create_a == 0 ? pthread_create(&tb, NULL, threading_scache_disabled_worker, &b) : -1;
  if (create_a == 0 && create_b != 0) {
    pthread_mutex_lock(&gate.mutex);
    gate.stopped = true;
    gate.arrived = 0;
    gate.generation++;
    pthread_cond_broadcast(&gate.cond);
    pthread_mutex_unlock(&gate.mutex);
  }
  int join_a = create_a == 0 ? pthread_join(ta, NULL) : -1;
  int join_b = create_b == 0 ? pthread_join(tb, NULL) : -1;
  if (cond_rc == 0) pthread_cond_destroy(&gate.cond);
  if (mutex_rc == 0) pthread_mutex_destroy(&gate.mutex);

  int restore_rc = saved_scache ? setenv("POLY_SCACHE", saved_scache, 1)
                                : unsetenv("POLY_SCACHE");
  free(saved_scache);

  ASSERT_INT_EQ(mutex_rc, 0);
  ASSERT_INT_EQ(cond_rc, 0);
  ASSERT_INT_EQ(create_a, 0);
  ASSERT_INT_EQ(create_b, 0);
  ASSERT_INT_EQ(join_a, 0);
  ASSERT_INT_EQ(join_b, 0);
  ASSERT_INT_EQ(restore_rc, 0);
  ASSERT_TRUE(a.ok);
  ASSERT_TRUE(b.ok);
  PASS();
}

typedef struct {
  PolyPat *pat;
  bool creator_ok;
  bool consumer_ok;
} ThreadingPatHandoff;

static PolyUOp *threading_pat_unwrap(
    PolyCtx *ctx,
    PolyUOp *matched,
    const PolyBindings *bindings
) {
  (void)ctx;
  (void)matched;
  return poly_bind(bindings, "x");
}

static void *threading_pat_creator(void *opaque) {
  ThreadingPatHandoff *handoff = opaque;
  handoff->pat = poly_pat_op1(POLY_OP_NEG, poly_pat_any("x"), NULL);
  handoff->creator_ok = handoff->pat != NULL;
  return NULL;
}

static void *threading_pat_consumer(void *opaque) {
  ThreadingPatHandoff *handoff = opaque;
  PolyPat *pat = handoff->pat;
  PolyRule rule = {.pat = pat, .fn = threading_pat_unwrap};
  PolyPatternMatcher *pm = pat ? poly_pm_new(&rule, 1) : NULL;
  PolyCtx *ctx = pm ? poly_ctx_new() : NULL;
  PolyUOp *one = ctx ? poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(1.0)) : NULL;
  PolyUOp *neg = one ? poly_uop1(ctx, POLY_OP_NEG, POLY_FLOAT32, one, poly_arg_none()) : NULL;
  PolyUOp *rewritten = neg ? poly_pm_rewrite(pm, ctx, neg) : NULL;
  handoff->consumer_ok = rewritten == one;

  poly_pm_destroy(pm);
  poly_pat_free(pat);
  poly_ctx_destroy(ctx);
  return NULL;
}

TEST(threading, public_pattern_survives_creator_thread_exit) {
  ThreadingPatHandoff handoff = {0};
  pthread_t creator, consumer;
  int create_creator = pthread_create(&creator, NULL, threading_pat_creator, &handoff);
  int join_creator = create_creator == 0 ? pthread_join(creator, NULL) : -1;
  int create_consumer = join_creator == 0
                            ? pthread_create(&consumer, NULL, threading_pat_consumer, &handoff)
                            : -1;
  int join_consumer = create_consumer == 0 ? pthread_join(consumer, NULL) : -1;

  ASSERT_INT_EQ(create_creator, 0);
  ASSERT_INT_EQ(join_creator, 0);
  ASSERT_INT_EQ(create_consumer, 0);
  ASSERT_INT_EQ(join_consumer, 0);
  ASSERT_TRUE(handoff.creator_ok);
  ASSERT_TRUE(handoff.consumer_ok);
  PASS();
}

typedef struct {
  bool ok;
} ThreadingPatCacheResult;

static void *threading_pat_cache_worker(void *opaque) {
  ThreadingPatCacheResult *result = opaque;
  PolyPat *pat = poly_pat_op1(POLY_OP_NEG, poly_pat_any("x"), NULL);
  PolyRule rule = {.pat = pat, .fn = threading_pat_unwrap};
  PolyPatternMatcher *pm = pat ? poly_pm_new(&rule, 1) : NULL;
  if (!pm) {
    poly_pat_free(pat);
    return NULL;
  }
  result->ok = poly_pm_thread_cache(pm) == pm;
  return NULL;
}

TEST(threading, compiler_pattern_cache_releases_at_thread_exit) {
  ThreadingPatCacheResult result = {0};
  pthread_t worker;
  ASSERT_INT_EQ(pthread_create(&worker, NULL, threading_pat_cache_worker, &result), 0);
  ASSERT_INT_EQ(pthread_join(worker, NULL), 0);
  ASSERT_TRUE(result.ok);
  PASS();
}
