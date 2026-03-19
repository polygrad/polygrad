#define _POSIX_C_SOURCE 200809L
/*
 * bench_beam.c — Compare heuristic vs BEAM search optimization
 *
 * For each kernel: compile+time with heuristic, then compile+time with BEAM.
 * Reports execution time and speedup ratio.
 */

#include "../src/codegen.h"
#include "../src/frontend.h"
#include "../src/scheduler.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_us(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

/* Time a realize call (compile + execute). Returns median of `reps` runs.
 * First call includes compilation; subsequent calls use cached plan. */
static double bench_realize(PolyCtx *ctx, PolyUOp *sink,
                            PolyBufferBinding *bindings, int n_bind,
                            int warmup, int reps) {
  /* Warmup (includes compilation on first call) */
  for (int i = 0; i < warmup; i++)
    poly_realize(ctx, sink, bindings, n_bind);

  double *times = malloc((size_t)reps * sizeof(double));
  for (int i = 0; i < reps; i++) {
    double t0 = now_us();
    poly_realize(ctx, sink, bindings, n_bind);
    double t1 = now_us();
    times[i] = t1 - t0;
  }

  /* Sort, take median */
  for (int i = 0; i < reps - 1; i++)
    for (int j = i + 1; j < reps; j++)
      if (times[j] < times[i]) { double t = times[i]; times[i] = times[j]; times[j] = t; }
  double med = times[reps / 2];
  free(times);
  return med;
}

typedef struct {
  const char *name;
  double heuristic_us;
  double beam_us;
} BenchResult;

/* ── Benchmark kernels ─────────────────────────────────────────────── */

static BenchResult bench_vecadd(int N, int beam_width) {
  /* Heuristic */
  setenv("POLY_OPTIMIZE", "1", 1);
  setenv("POLY_DEVECTORIZE", "1", 1);
  unsetenv("POLY_BEAM");

  PolyCtx *ctx1 = poly_ctx_new();
  PolyUOp *a1 = poly_buffer_f32(ctx1, N);
  PolyUOp *b1 = poly_buffer_f32(ctx1, N);
  PolyUOp *c1 = poly_alu2(ctx1, POLY_OP_ADD, a1, b1);
  PolyUOp *o1 = poly_buffer_f32(ctx1, N);
  PolyUOp *s1 = poly_store_val(ctx1, o1, c1);
  PolyUOp *sk1 = poly_sink1(ctx1, s1);

  float *da = calloc(N, sizeof(float));
  float *db = calloc(N, sizeof(float));
  float *dout = calloc(N, sizeof(float));
  for (int i = 0; i < N; i++) { da[i] = (float)i; db[i] = 1.0f; }

  PolyBufferBinding bind1[] = {
    POLY_BIND_HOST(a1, da), POLY_BIND_HOST(b1, db), POLY_BIND_HOST(o1, dout)
  };
  double t_heur = bench_realize(ctx1, sk1, bind1, 3, 3, 20);
  poly_ctx_destroy(ctx1);

  /* BEAM */
  char bw[16];
  snprintf(bw, sizeof(bw), "%d", beam_width);
  setenv("POLY_BEAM", bw, 1);

  PolyCtx *ctx2 = poly_ctx_new();
  PolyUOp *a2 = poly_buffer_f32(ctx2, N);
  PolyUOp *b2 = poly_buffer_f32(ctx2, N);
  PolyUOp *c2 = poly_alu2(ctx2, POLY_OP_ADD, a2, b2);
  PolyUOp *o2 = poly_buffer_f32(ctx2, N);
  PolyUOp *s2 = poly_store_val(ctx2, o2, c2);
  PolyUOp *sk2 = poly_sink1(ctx2, s2);

  memset(dout, 0, N * sizeof(float));
  PolyBufferBinding bind2[] = {
    POLY_BIND_HOST(a2, da), POLY_BIND_HOST(b2, db), POLY_BIND_HOST(o2, dout)
  };
  double t_beam = bench_realize(ctx2, sk2, bind2, 3, 3, 20);
  poly_ctx_destroy(ctx2);
  unsetenv("POLY_BEAM");

  free(da); free(db); free(dout);
  return (BenchResult){ "vecadd", t_heur, t_beam };
}

static BenchResult bench_reduce_sum(int N, int beam_width) {
  setenv("POLY_OPTIMIZE", "1", 1);
  setenv("POLY_DEVECTORIZE", "1", 1);
  unsetenv("POLY_BEAM");

  PolyCtx *ctx1 = poly_ctx_new();
  PolyUOp *a1 = poly_buffer_f32(ctx1, N);
  int64_t axis = 0;
  PolyUOp *sum1 = poly_reduce_axis(ctx1, POLY_OP_ADD, a1, &axis, 1);
  PolyUOp *o1 = poly_buffer_f32(ctx1, 1);
  PolyUOp *s1 = poly_store_val(ctx1, o1, sum1);
  PolyUOp *sk1 = poly_sink1(ctx1, s1);

  float *da = calloc(N, sizeof(float));
  float dout = 0;
  for (int i = 0; i < N; i++) da[i] = 1.0f;

  PolyBufferBinding bind1[] = {
    POLY_BIND_HOST(a1, da), POLY_BIND_HOST(o1, &dout)
  };
  double t_heur = bench_realize(ctx1, sk1, bind1, 2, 3, 20);
  poly_ctx_destroy(ctx1);

  /* BEAM */
  char bw[16];
  snprintf(bw, sizeof(bw), "%d", beam_width);
  setenv("POLY_BEAM", bw, 1);

  PolyCtx *ctx2 = poly_ctx_new();
  PolyUOp *a2 = poly_buffer_f32(ctx2, N);
  PolyUOp *sum2 = poly_reduce_axis(ctx2, POLY_OP_ADD, a2, &axis, 1);
  PolyUOp *o2 = poly_buffer_f32(ctx2, 1);
  PolyUOp *s2 = poly_store_val(ctx2, o2, sum2);
  PolyUOp *sk2 = poly_sink1(ctx2, s2);

  dout = 0;
  PolyBufferBinding bind2[] = {
    POLY_BIND_HOST(a2, da), POLY_BIND_HOST(o2, &dout)
  };
  double t_beam = bench_realize(ctx2, sk2, bind2, 2, 3, 20);
  poly_ctx_destroy(ctx2);
  unsetenv("POLY_BEAM");

  free(da);
  return (BenchResult){ "reduce_sum", t_heur, t_beam };
}

static BenchResult bench_chain_fused(int N, int beam_width) {
  setenv("POLY_OPTIMIZE", "1", 1);
  setenv("POLY_DEVECTORIZE", "1", 1);
  unsetenv("POLY_BEAM");

  PolyCtx *ctx1 = poly_ctx_new();
  PolyUOp *a1 = poly_buffer_f32(ctx1, N);
  PolyUOp *b1 = poly_buffer_f32(ctx1, N);
  PolyUOp *ab = poly_alu2(ctx1, POLY_OP_MUL, a1, b1);
  PolyUOp *neg = poly_alu1(ctx1, POLY_OP_NEG, ab);
  PolyUOp *two = poly_const_float(ctx1, 2.0);
  PolyUOp *add2 = poly_alu2(ctx1, POLY_OP_ADD, neg, two);
  PolyUOp *o1 = poly_buffer_f32(ctx1, N);
  PolyUOp *s1 = poly_store_val(ctx1, o1, add2);
  PolyUOp *sk1 = poly_sink1(ctx1, s1);

  float *da = calloc(N, sizeof(float));
  float *db = calloc(N, sizeof(float));
  float *dout = calloc(N, sizeof(float));
  for (int i = 0; i < N; i++) { da[i] = 0.5f; db[i] = 0.3f; }

  PolyBufferBinding bind1[] = {
    POLY_BIND_HOST(a1, da), POLY_BIND_HOST(b1, db), POLY_BIND_HOST(o1, dout)
  };
  double t_heur = bench_realize(ctx1, sk1, bind1, 3, 3, 20);
  poly_ctx_destroy(ctx1);

  char bw[16];
  snprintf(bw, sizeof(bw), "%d", beam_width);
  setenv("POLY_BEAM", bw, 1);

  PolyCtx *ctx2 = poly_ctx_new();
  PolyUOp *a2 = poly_buffer_f32(ctx2, N);
  PolyUOp *b2 = poly_buffer_f32(ctx2, N);
  PolyUOp *ab2 = poly_alu2(ctx2, POLY_OP_MUL, a2, b2);
  PolyUOp *neg2 = poly_alu1(ctx2, POLY_OP_NEG, ab2);
  PolyUOp *two2 = poly_const_float(ctx2, 2.0);
  PolyUOp *add22 = poly_alu2(ctx2, POLY_OP_ADD, neg2, two2);
  PolyUOp *o2 = poly_buffer_f32(ctx2, N);
  PolyUOp *s2 = poly_store_val(ctx2, o2, add22);
  PolyUOp *sk2 = poly_sink1(ctx2, s2);

  memset(dout, 0, N * sizeof(float));
  PolyBufferBinding bind2[] = {
    POLY_BIND_HOST(a2, da), POLY_BIND_HOST(b2, db), POLY_BIND_HOST(o2, dout)
  };
  double t_beam = bench_realize(ctx2, sk2, bind2, 3, 3, 20);
  poly_ctx_destroy(ctx2);
  unsetenv("POLY_BEAM");

  free(da); free(db); free(dout);
  return (BenchResult){ "chain_fused", t_heur, t_beam };
}

static BenchResult bench_matmul(int M, int K, int N, int beam_width) {
  setenv("POLY_OPTIMIZE", "1", 1);
  setenv("POLY_DEVECTORIZE", "1", 1);
  unsetenv("POLY_BEAM");

  PolyCtx *ctx1 = poly_ctx_new();
  PolyUOp *a1 = poly_buffer_f32(ctx1, M * K);
  PolyUOp *b1 = poly_buffer_f32(ctx1, K * N);
  a1 = poly_reshape(ctx1, a1, (int64_t[]){M, K}, 2);
  b1 = poly_reshape(ctx1, b1, (int64_t[]){K, N}, 2);
  int64_t out_shape1[2]; int out_ndim1;
  PolyUOp *c1 = poly_dot(ctx1, a1, (int64_t[]){M, K}, 2, b1, (int64_t[]){K, N}, 2,
                          out_shape1, &out_ndim1);
  PolyUOp *o1 = poly_buffer_f32(ctx1, M * N);
  o1 = poly_reshape(ctx1, o1, (int64_t[]){M, N}, 2);
  PolyUOp *s1 = poly_store_val(ctx1, o1, c1);
  PolyUOp *sk1 = poly_sink1(ctx1, s1);

  float *da = calloc(M * K, sizeof(float));
  float *db = calloc(K * N, sizeof(float));
  float *dout = calloc(M * N, sizeof(float));
  for (int i = 0; i < M * K; i++) da[i] = 0.01f * (float)(i % 100);
  for (int i = 0; i < K * N; i++) db[i] = 0.01f * (float)(i % 100);

  PolyBufferBinding bind1[] = {
    POLY_BIND_HOST(a1, da), POLY_BIND_HOST(b1, db), POLY_BIND_HOST(o1, dout)
  };
  double t_heur = bench_realize(ctx1, sk1, bind1, 3, 2, 10);
  poly_ctx_destroy(ctx1);

  char bw[16];
  snprintf(bw, sizeof(bw), "%d", beam_width);
  setenv("POLY_BEAM", bw, 1);

  PolyCtx *ctx2 = poly_ctx_new();
  PolyUOp *a2 = poly_buffer_f32(ctx2, M * K);
  PolyUOp *b2 = poly_buffer_f32(ctx2, K * N);
  a2 = poly_reshape(ctx2, a2, (int64_t[]){M, K}, 2);
  b2 = poly_reshape(ctx2, b2, (int64_t[]){K, N}, 2);
  int64_t out_shape2[2]; int out_ndim2;
  PolyUOp *c2 = poly_dot(ctx2, a2, (int64_t[]){M, K}, 2, b2, (int64_t[]){K, N}, 2,
                          out_shape2, &out_ndim2);
  PolyUOp *o2 = poly_buffer_f32(ctx2, M * N);
  o2 = poly_reshape(ctx2, o2, (int64_t[]){M, N}, 2);
  PolyUOp *s2 = poly_store_val(ctx2, o2, c2);
  PolyUOp *sk2 = poly_sink1(ctx2, s2);

  memset(dout, 0, M * N * sizeof(float));
  PolyBufferBinding bind2[] = {
    POLY_BIND_HOST(a2, da), POLY_BIND_HOST(b2, db), POLY_BIND_HOST(o2, dout)
  };
  double t_beam = bench_realize(ctx2, sk2, bind2, 3, 2, 10);
  poly_ctx_destroy(ctx2);
  unsetenv("POLY_BEAM");

  free(da); free(db); free(dout);
  return (BenchResult){ "matmul", t_heur, t_beam };
}

int main(int argc, char **argv) {
  int beam_width = 4;
  if (argc > 1) beam_width = atoi(argv[1]);

  printf("BEAM search benchmark (beam_width=%d)\n", beam_width);
  printf("%-25s %12s %12s %10s\n", "kernel", "heuristic", "beam", "speedup");
  printf("%-25s %12s %12s %10s\n", "------", "---------", "----", "-------");

  int sizes[] = {256, 1024, 16384, 65536};
  int n_sizes = 4;

  for (int si = 0; si < n_sizes; si++) {
    int N = sizes[si];
    char name[64];

    snprintf(name, sizeof(name), "vecadd N=%d", N);
    BenchResult r = bench_vecadd(N, beam_width);
    printf("%-25s %10.1f us %10.1f us %9.2fx\n", name, r.heuristic_us, r.beam_us,
           r.beam_us > 0 ? r.heuristic_us / r.beam_us : 0.0);
  }

  for (int si = 0; si < n_sizes; si++) {
    int N = sizes[si];
    char name[64];

    snprintf(name, sizeof(name), "reduce_sum N=%d", N);
    BenchResult r = bench_reduce_sum(N, beam_width);
    printf("%-25s %10.1f us %10.1f us %9.2fx\n", name, r.heuristic_us, r.beam_us,
           r.beam_us > 0 ? r.heuristic_us / r.beam_us : 0.0);
  }

  for (int si = 0; si < n_sizes; si++) {
    int N = sizes[si];
    char name[64];

    snprintf(name, sizeof(name), "chain_fused N=%d", N);
    BenchResult r = bench_chain_fused(N, beam_width);
    printf("%-25s %10.1f us %10.1f us %9.2fx\n", name, r.heuristic_us, r.beam_us,
           r.beam_us > 0 ? r.heuristic_us / r.beam_us : 0.0);
  }

  /* Matmul (multi-kernel, separate concern -- skip for now) */

  return 0;
}
