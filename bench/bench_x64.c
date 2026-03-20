/*
 * bench_x64.c -- Compare x64 JIT (scalar + vec) vs CPU (C compiler)
 *
 * Tests: vecadd, neg, chain (mul+add+sqrt), reduce_sum
 */
#define _POSIX_C_SOURCE 200809L
#include "../src/codegen.h"
#include "../src/polygrad.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

static double now_us(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

typedef struct { PolyCtx *ctx; PolyUOp *sink; } Kernel;

/* c[i] = a[i] + b[i] */
static Kernel make_vecadd(int N) {
  PolyCtx *ctx = poly_ctx_new();
  PolyDType pf = poly_dtype_ptr(POLY_FLOAT32, -1, POLY_ADDR_GLOBAL);
  PolyUOp *p0 = poly_uop0(ctx, POLY_OP_PARAM, pf, poly_arg_int(0));
  PolyUOp *p1 = poly_uop0(ctx, POLY_OP_PARAM, pf, poly_arg_int(1));
  PolyUOp *p2 = poly_uop0(ctx, POLY_OP_PARAM, pf, poly_arg_int(2));
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(N));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));
  PolyUOp *i0 = poly_uop2(ctx, POLY_OP_INDEX, pf, p0, range, poly_arg_none());
  PolyUOp *i1 = poly_uop2(ctx, POLY_OP_INDEX, pf, p1, range, poly_arg_none());
  PolyUOp *i2 = poly_uop2(ctx, POLY_OP_INDEX, pf, p2, range, poly_arg_none());
  PolyUOp *l0 = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, i0, poly_arg_none());
  PolyUOp *l1 = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, i1, poly_arg_none());
  PolyUOp *alu = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, l0, l1, poly_arg_none());
  PolyUOp *st = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, i2, alu, poly_arg_none());
  PolyUOp *es[2] = { st, range };
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, es, 2, poly_arg_none());
  return (Kernel){ ctx, poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none()) };
}

/* b[i] = -a[i] */
static Kernel make_neg(int N) {
  PolyCtx *ctx = poly_ctx_new();
  PolyDType pf = poly_dtype_ptr(POLY_FLOAT32, -1, POLY_ADDR_GLOBAL);
  PolyUOp *p0 = poly_uop0(ctx, POLY_OP_PARAM, pf, poly_arg_int(0));
  PolyUOp *p1 = poly_uop0(ctx, POLY_OP_PARAM, pf, poly_arg_int(1));
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(N));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));
  PolyUOp *i0 = poly_uop2(ctx, POLY_OP_INDEX, pf, p0, range, poly_arg_none());
  PolyUOp *i1 = poly_uop2(ctx, POLY_OP_INDEX, pf, p1, range, poly_arg_none());
  PolyUOp *l0 = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, i0, poly_arg_none());
  PolyUOp *neg = poly_uop1(ctx, POLY_OP_NEG, POLY_FLOAT32, l0, poly_arg_none());
  PolyUOp *st = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, i1, neg, poly_arg_none());
  PolyUOp *es[2] = { st, range };
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, es, 2, poly_arg_none());
  return (Kernel){ ctx, poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none()) };
}

/* c[i] = sqrt(a[i] * b[i] + a[i]) — 3-op chain */
static Kernel make_chain(int N) {
  PolyCtx *ctx = poly_ctx_new();
  PolyDType pf = poly_dtype_ptr(POLY_FLOAT32, -1, POLY_ADDR_GLOBAL);
  PolyUOp *p0 = poly_uop0(ctx, POLY_OP_PARAM, pf, poly_arg_int(0));
  PolyUOp *p1 = poly_uop0(ctx, POLY_OP_PARAM, pf, poly_arg_int(1));
  PolyUOp *p2 = poly_uop0(ctx, POLY_OP_PARAM, pf, poly_arg_int(2));
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(N));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));
  PolyUOp *i0 = poly_uop2(ctx, POLY_OP_INDEX, pf, p0, range, poly_arg_none());
  PolyUOp *i1 = poly_uop2(ctx, POLY_OP_INDEX, pf, p1, range, poly_arg_none());
  PolyUOp *i2 = poly_uop2(ctx, POLY_OP_INDEX, pf, p2, range, poly_arg_none());
  PolyUOp *l0 = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, i0, poly_arg_none());
  PolyUOp *l1 = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, i1, poly_arg_none());
  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, l0, l1, poly_arg_none());
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, mul, l0, poly_arg_none());
  PolyUOp *sq = poly_uop1(ctx, POLY_OP_SQRT, POLY_FLOAT32, add, poly_arg_none());
  PolyUOp *st = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, i2, sq, poly_arg_none());
  PolyUOp *es[2] = { st, range };
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, es, 2, poly_arg_none());
  return (Kernel){ ctx, poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none()) };
}

typedef struct {
  const char *name;
  double cpu_exec, x64s_exec, x64v_exec;
  int x64s_bytes, x64v_bytes;
} Result;

static Result bench(const char *name, Kernel k, void **args, int n_args, int iters) {
  Result r = { .name = name };

  /* Scalar linearize (shared by CPU and x64 scalar) */
  int n_lin;
  PolyUOp **lin = poly_linearize(k.ctx, k.sink, &n_lin);
  if (!lin) { printf("  %-12s FAIL linearize\n", name); return r; }

  /* CPU (clang -O2) */
  char *src = poly_render_c(lin, n_lin, "bench_fn");
  if (!src) { free(lin); printf("  %-12s FAIL render_c\n", name); return r; }
  PolyProgram *cpu = poly_compile_c(src, "bench_fn");
  free(src);
  if (!cpu) { free(lin); printf("  %-12s FAIL compile_c\n", name); return r; }
  poly_program_call(cpu, args, n_args);
  double t0 = now_us();
  for (int i = 0; i < iters; i++) poly_program_call(cpu, args, n_args);
  r.cpu_exec = (now_us() - t0) / iters;

  /* x64 scalar (no optimize) */
  int sz_s;
  uint8_t *code_s = poly_render_x64(lin, n_lin, &sz_s);
  free(lin);
  if (code_s) {
    PolyX64Program *xs = poly_compile_x64(code_s, sz_s);
    free(code_s);
    if (xs) {
      r.x64s_bytes = sz_s;
      poly_x64_program_call(xs, args, n_args);
      t0 = now_us();
      for (int i = 0; i < iters; i++) poly_x64_program_call(xs, args, n_args);
      r.x64s_exec = (now_us() - t0) / iters;
      poly_x64_program_destroy(xs);
    }
  }

  /* x64 vec (optimize + UPCAST) */
  PolyRewriteOpts vopts = { .optimize = true, .devectorize = 0 };
  int n_v;
  PolyUOp **lin_v = poly_linearize_ex(k.ctx, k.sink, vopts, &n_v);
  if (lin_v) {
    int sz_v;
    uint8_t *code_v = poly_render_x64(lin_v, n_v, &sz_v);
    free(lin_v);
    if (code_v) {
      PolyX64Program *xv = poly_compile_x64(code_v, sz_v);
      free(code_v);
      if (xv) {
        r.x64v_bytes = sz_v;
        poly_x64_program_call(xv, args, n_args);
        t0 = now_us();
        for (int i = 0; i < iters; i++) poly_x64_program_call(xv, args, n_args);
        r.x64v_exec = (now_us() - t0) / iters;
        poly_x64_program_destroy(xv);
      }
    }
  }

  poly_program_destroy(cpu);
  return r;
}

static void print_result(Result r) {
  double s_ratio = r.x64s_exec > 0 ? r.cpu_exec / r.x64s_exec : 0;
  double v_ratio = r.x64v_exec > 0 ? r.cpu_exec / r.x64v_exec : 0;
  printf("  %-12s  cpu: %8.1f   x64s: %8.1f (%4.2fx, %3db)   x64v: %8.1f (%4.2fx, %3db)\n",
         r.name, r.cpu_exec, r.x64s_exec, s_ratio, r.x64s_bytes,
         r.x64v_exec, v_ratio, r.x64v_bytes);
}

int main(void) {
  int sizes[] = {1024, 16384, 262144};
  printf("\n  x64 JIT benchmark (us per call). ratio = cpu/x64 (>1 = x64 faster)\n");
  printf("  ===================================================================\n\n");

  for (int si = 0; si < 3; si++) {
    int N = sizes[si];
    int iters = N <= 1024 ? 100000 : (N <= 16384 ? 10000 : 1000);

    float *a = malloc((size_t)N * sizeof(float));
    float *b = malloc((size_t)N * sizeof(float));
    float *c = malloc((size_t)N * sizeof(float));
    for (int i = 0; i < N; i++) { a[i] = 1.0f + 0.001f * i; b[i] = 2.0f + 0.001f * i; }

    printf("  N = %d\n", N);

    void *args3[3] = { a, b, c };
    void *args2[2] = { a, b };

    Kernel k;

    k = make_vecadd(N);
    print_result(bench("vecadd", k, args3, 3, iters));
    poly_ctx_destroy(k.ctx);

    k = make_neg(N);
    print_result(bench("neg", k, args2, 2, iters));
    poly_ctx_destroy(k.ctx);

    k = make_chain(N);
    print_result(bench("chain", k, args3, 3, iters));
    poly_ctx_destroy(k.ctx);

    printf("\n");
    free(a); free(b); free(c);
  }
  return 0;
}
