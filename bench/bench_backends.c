/*
 * bench_backends.c -- Compare CPU compiled vs interpreter vs WASM JIT execution
 *
 * Benchmarks the exec_plan pipeline (prepare + lower + run) across backends.
 * Tests: vecadd, chain (5-op fused), reduce_sum, matmul-like patterns.
 *
 * Usage: ./build/bench_backends [N]
 */

#define _POSIX_C_SOURCE 200809L
#include "../src/frontend.h"
#include "../src/exec_plan.h"
#include "../src/scheduler.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_us(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

/* ── Graph builders ──────────────────────────────────────────────────── */

/* out = a + b (elementwise) */
static PolyUOp *make_vecadd(PolyCtx *ctx, int n,
                             PolyUOp **a, PolyUOp **b, PolyUOp **out) {
  *a = poly_buffer_f32(ctx, n);
  *b = poly_buffer_f32(ctx, n);
  *out = poly_buffer_f32(ctx, n);
  PolyUOp *sum = poly_alu2(ctx, POLY_OP_ADD, *a, *b);
  PolyUOp *store = poly_store_val(ctx, *out, sum);
  return poly_sink1(ctx, store);
}

/* out = relu(sqrt(abs(a * b + a))) -- 5-op fused chain */
static PolyUOp *make_chain5(PolyCtx *ctx, int n,
                             PolyUOp **a, PolyUOp **b, PolyUOp **out) {
  *a = poly_buffer_f32(ctx, n);
  *b = poly_buffer_f32(ctx, n);
  *out = poly_buffer_f32(ctx, n);
  PolyUOp *prod = poly_alu2(ctx, POLY_OP_MUL, *a, *b);
  PolyUOp *sum = poly_alu2(ctx, POLY_OP_ADD, prod, *a);
  PolyUOp *ab = poly_abs(ctx, sum);
  PolyUOp *sq = poly_alu1(ctx, POLY_OP_SQRT, ab);
  PolyUOp *relu = poly_relu(ctx, sq);
  PolyUOp *store = poly_store_val(ctx, *out, relu);
  return poly_sink1(ctx, store);
}

/* out = sum(a) -- reduce */
static PolyUOp *make_reduce(PolyCtx *ctx, int n,
                             PolyUOp **a, PolyUOp **out) {
  *a = poly_buffer_f32(ctx, n);
  *out = poly_buffer_f32(ctx, 1);
  int64_t axes[] = {0};
  PolyUOp *reduced = poly_reduce_axis(ctx, POLY_OP_ADD, *a, axes, 1);
  PolyUOp *store = poly_store_val(ctx, *out, reduced);
  return poly_sink1(ctx, store);
}

/* ── Benchmark runner ────────────────────────────────────────────────── */

typedef struct {
  const char *name;
  double cpu_us;
  double interp_us;
  double cpu_lower_us;
  double interp_lower_us;
} BenchResult;

static BenchResult bench_graph(const char *name, PolyCtx *ctx, PolyUOp *sink,
                                void **slot_data, int n_slots, int iters) {
  BenchResult r = { .name = name };

  /* Prepare (shared, one-time) */
  PolyPreparedStep *prep = poly_prepare_step(ctx, sink, POLY_MODE_CALL);
  if (!prep) { fprintf(stderr, "prepare failed for %s\n", name); return r; }

  /* CPU: lower + warmup + bench */
  double t0 = now_us();
  PolyExecutableStep *cpu_exec = poly_lower_step(ctx, prep, POLY_DEVICE_CPU);
  r.cpu_lower_us = now_us() - t0;
  if (!cpu_exec) { fprintf(stderr, "CPU lower failed for %s\n", name); return r; }

  /* Warmup */
  poly_executable_step_run(cpu_exec, slot_data, n_slots, NULL, 0);

  t0 = now_us();
  for (int i = 0; i < iters; i++)
    poly_executable_step_run(cpu_exec, slot_data, n_slots, NULL, 0);
  r.cpu_us = (now_us() - t0) / iters;

  /* INTERP: lower + warmup + bench */
  t0 = now_us();
  PolyExecutableStep *interp_exec = poly_lower_step(ctx, prep, POLY_DEVICE_INTERP);
  r.interp_lower_us = now_us() - t0;
  if (!interp_exec) { fprintf(stderr, "INTERP lower failed for %s\n", name); return r; }

  poly_executable_step_run(interp_exec, slot_data, n_slots, NULL, 0);

  t0 = now_us();
  for (int i = 0; i < iters; i++)
    poly_executable_step_run(interp_exec, slot_data, n_slots, NULL, 0);
  r.interp_us = (now_us() - t0) / iters;

  poly_executable_step_free(cpu_exec);
  poly_executable_step_free(interp_exec);
  poly_prepared_step_free(prep);

  return r;
}

/* ── Main ────────────────────────────────────────────────────────────── */

int main(int argc, char **argv) {
  int N = (argc > 1) ? atoi(argv[1]) : 0;
  int iters = 200;

  int sizes[] = { 64, 1024, 16384, 262144, 1048576 };
  int n_sizes = sizeof(sizes) / sizeof(sizes[0]);

  printf("Backend Benchmark: CPU compiled vs Interpreter\n");
  printf("===============================================\n\n");

  if (N > 0) {
    sizes[0] = N;
    n_sizes = 1;
  }

  for (int si = 0; si < n_sizes; si++) {
    int n = sizes[si];
    printf("--- N = %d ---\n", n);
    printf("%-20s %12s %12s %8s   %12s %12s\n",
           "Kernel", "CPU (us)", "INTERP (us)", "Ratio",
           "CPU lower", "INTERP lower");

    /* Allocate test data */
    float *a = malloc((size_t)n * sizeof(float));
    float *b = malloc((size_t)n * sizeof(float));
    float *out = malloc((size_t)n * sizeof(float));
    for (int i = 0; i < n; i++) {
      a[i] = 1.0f + 0.001f * i;
      b[i] = 2.0f - 0.001f * i;
    }

    /* vecadd */
    {
      PolyCtx *ctx = poly_ctx_new();
      PolyUOp *ua, *ub, *uo;
      PolyUOp *sink = make_vecadd(ctx, n, &ua, &ub, &uo);
      void *slot_data[8] = {0};
      /* Find slots by matching buf_uop pointers */
      PolyPreparedStep *prep = poly_prepare_step(ctx, sink, POLY_MODE_CALL);
      for (int s = 0; s < prep->n_buf_slots; s++) {
        if (prep->buf_slots[s].buf_uop == ua) slot_data[s] = a;
        else if (prep->buf_slots[s].buf_uop == ub) slot_data[s] = b;
        else if (prep->buf_slots[s].buf_uop == uo) slot_data[s] = out;
      }
      poly_prepared_step_free(prep);

      BenchResult r = bench_graph("vecadd", ctx, sink,
                                   slot_data, 8, iters);
      printf("%-20s %12.1f %12.1f %7.1fx   %12.0f %12.0f\n",
             r.name, r.cpu_us, r.interp_us,
             r.interp_us / (r.cpu_us > 0 ? r.cpu_us : 1),
             r.cpu_lower_us, r.interp_lower_us);
      poly_ctx_destroy(ctx);
    }

    /* chain5 */
    {
      PolyCtx *ctx = poly_ctx_new();
      PolyUOp *ua, *ub, *uo;
      PolyUOp *sink = make_chain5(ctx, n, &ua, &ub, &uo);
      void *slot_data[8] = {0};
      PolyPreparedStep *prep = poly_prepare_step(ctx, sink, POLY_MODE_CALL);
      for (int s = 0; s < prep->n_buf_slots; s++) {
        if (prep->buf_slots[s].buf_uop == ua) slot_data[s] = a;
        else if (prep->buf_slots[s].buf_uop == ub) slot_data[s] = b;
        else if (prep->buf_slots[s].buf_uop == uo) slot_data[s] = out;
      }
      poly_prepared_step_free(prep);

      BenchResult r = bench_graph("chain5 (fused)", ctx, sink,
                                   slot_data, 8, iters);
      printf("%-20s %12.1f %12.1f %7.1fx   %12.0f %12.0f\n",
             r.name, r.cpu_us, r.interp_us,
             r.interp_us / (r.cpu_us > 0 ? r.cpu_us : 1),
             r.cpu_lower_us, r.interp_lower_us);
      poly_ctx_destroy(ctx);
    }

    /* reduce */
    {
      PolyCtx *ctx = poly_ctx_new();
      PolyUOp *ua, *uo;
      PolyUOp *sink = make_reduce(ctx, n, &ua, &uo);
      float reduce_out = 0;
      void *slot_data[8] = {0};
      PolyPreparedStep *prep = poly_prepare_step(ctx, sink, POLY_MODE_CALL);
      for (int s = 0; s < prep->n_buf_slots; s++) {
        if (prep->buf_slots[s].buf_uop == ua) slot_data[s] = a;
        else if (prep->buf_slots[s].buf_uop == uo) slot_data[s] = &reduce_out;
      }
      poly_prepared_step_free(prep);

      BenchResult r = bench_graph("reduce_sum", ctx, sink,
                                   slot_data, 8, n > 100000 ? 50 : iters);
      printf("%-20s %12.1f %12.1f %7.1fx   %12.0f %12.0f\n",
             r.name, r.cpu_us, r.interp_us,
             r.interp_us / (r.cpu_us > 0 ? r.cpu_us : 1),
             r.cpu_lower_us, r.interp_lower_us);
      poly_ctx_destroy(ctx);
    }

    free(a); free(b); free(out);
    printf("\n");
  }

  return 0;
}
