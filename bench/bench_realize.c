#define _POSIX_C_SOURCE 200809L
/*
 * bench_realize.c -- poly_realize performance benchmark
 *
 * Tests: single-kernel vecadd, multi-kernel reduce chain, and
 * repeated calls (cache hit path) at various sizes.
 */

#include "../src/polygrad.h"
#include "../src/frontend.h"
#include "../src/codegen.h"
#include "../src/scheduler.h"
#ifdef POLY_EXEC_PLAN_H
/* already pulled in by frontend.h on the branch */
#endif
/* If frontend.h doesn't pull in exec_plan.h (main), POLY_BIND_HOST won't
 * be defined and we fall back to the old PolyBufferBinding layout. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Compatibility: branch has PolyBufferBinding with .handle (PolyBufferHandle),
 * main has PolyBufferBinding with .data (void*).  Detect via POLY_BIND_HOST. */
#ifndef POLY_BIND_HOST
#define MAKE_BIND(buf, ptr) ((PolyBufferBinding){ (buf), (ptr) })
#else
#define MAKE_BIND(buf, ptr) POLY_BIND_HOST(buf, ptr)
#endif

static double now_us(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

/* ── Vecadd: single kernel, elementwise ─────────────────────────────── */

static void bench_vecadd(int n, int iters) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer_f32(ctx, n);
  PolyUOp *b = poly_buffer_f32(ctx, n);
  PolyUOp *c = poly_buffer_f32(ctx, n);
  PolyUOp *add = poly_alu2(ctx, POLY_OP_ADD, a, b);
  PolyUOp *store = poly_store_val(ctx, c, add);
  PolyUOp *sink = poly_sink1(ctx, store);

  float *ha = malloc(n * sizeof(float));
  float *hb = malloc(n * sizeof(float));
  float *hc = calloc(n, sizeof(float));
  for (int i = 0; i < n; i++) { ha[i] = (float)i * 0.001f; hb[i] = 1.0f; }

  PolyBufferBinding binds[] = {
    MAKE_BIND(c, hc), MAKE_BIND(a, ha), MAKE_BIND(b, hb)
  };

  /* Warmup (includes first compile) */
  poly_realize(ctx, sink, binds, 3);

  /* Timed: should hit cache */
  double t0 = now_us();
  for (int it = 0; it < iters; it++)
    poly_realize(ctx, sink, binds, 3);
  double us = (now_us() - t0) / iters;

  printf("  vecadd        N=%-8d  %8.1f us/call  (%d iters)\n", n, us, iters);

  free(ha); free(hb); free(hc);
  poly_ctx_destroy(ctx);
}

/* ── Reduce chain: multi-kernel ─────────────────────────────────────── */

static void bench_reduce_chain(int n, int iters) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer_f32(ctx, n);
  PolyUOp *b = poly_buffer_f32(ctx, n);
  PolyUOp *out = poly_buffer_f32(ctx, n);

  int64_t axes[] = {0};
  int64_t one_sh[] = {1};
  int64_t exp_sh[] = {n};
  PolyUOp *s = poly_reduce_axis(ctx, POLY_OP_ADD, a, axes, 1);
  PolyUOp *s1 = poly_reshape(ctx, s, one_sh, 1);
  PolyUOp *se = poly_expand(ctx, s1, exp_sh, 1);
  PolyUOp *c = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, se, b, poly_arg_none());
  PolyUOp *st = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out, c, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, st, poly_arg_none());

  float *ha = malloc(n * sizeof(float));
  float *hb = malloc(n * sizeof(float));
  float *hout = calloc(n, sizeof(float));
  for (int i = 0; i < n; i++) { ha[i] = (float)(i + 1); hb[i] = (float)(i * 10); }

  PolyBufferBinding binds[] = {
    MAKE_BIND(out, hout), MAKE_BIND(a, ha), MAKE_BIND(b, hb)
  };

  poly_realize(ctx, sink, binds, 3);

  double t0 = now_us();
  for (int it = 0; it < iters; it++)
    poly_realize(ctx, sink, binds, 3);
  double us = (now_us() - t0) / iters;

  printf("  reduce_chain  N=%-8d  %8.1f us/call  (%d iters)\n", n, us, iters);

  free(ha); free(hb); free(hout);
  poly_ctx_destroy(ctx);
}

/* ── First-call latency (cold compile) ──────────────────────────────── */

static void bench_cold_compile(int n) {
  double t0 = now_us();
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer_f32(ctx, n);
  PolyUOp *b = poly_buffer_f32(ctx, n);
  PolyUOp *c = poly_buffer_f32(ctx, n);
  PolyUOp *add = poly_alu2(ctx, POLY_OP_ADD, a, b);
  PolyUOp *store = poly_store_val(ctx, c, add);
  PolyUOp *sink = poly_sink1(ctx, store);

  float *ha = calloc(n, sizeof(float));
  float *hb = calloc(n, sizeof(float));
  float *hc = calloc(n, sizeof(float));

  PolyBufferBinding binds[] = {
    MAKE_BIND(c, hc), MAKE_BIND(a, ha), MAKE_BIND(b, hb)
  };
  poly_realize(ctx, sink, binds, 3);
  double us = now_us() - t0;

  printf("  cold_compile  N=%-8d  %8.1f us (graph build + schedule + compile + run)\n", n, us);

  free(ha); free(hb); free(hc);
  poly_ctx_destroy(ctx);
}

int main(void) {
  printf("\n  polygrad poly_realize benchmark\n");
  printf("  ================================\n\n");

  printf("  Cached (warm) calls:\n");
  bench_vecadd(64, 1000);
  bench_vecadd(1024, 500);
  bench_vecadd(65536, 100);
  bench_reduce_chain(64, 500);
  bench_reduce_chain(1024, 200);

  printf("\n  Cold compile latency:\n");
  bench_cold_compile(64);
  bench_cold_compile(1024);
  bench_cold_compile(65536);

  printf("\n");
  return 0;
}
