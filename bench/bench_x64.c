/*
 * bench_x64.c -- x64 JIT benchmark: CPU vs scalar vs SSE4 vs AVX2
 *
 * Tests: vecadd, neg, chain, exp2, log2, sin, reduce_sum, fma_chain
 * Columns: CPU (clang -O2), x64 scalar, x64 SSE (vec4), x64 AVX2 (vec8+FMA)
 */
#define _POSIX_C_SOURCE 200809L
#include "../src/codegen.h"
#include "../src/polygrad.h"
#include <cpuid.h>
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

static bool has_avx2(void) {
  unsigned int eax, ebx, ecx, edx;
  if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx)) return false;
  if (!((ecx >> 27) & 1)) return false;
  unsigned int xcr0_lo, xcr0_hi;
  __asm__ volatile("xgetbv" : "=a"(xcr0_lo), "=d"(xcr0_hi) : "c"(0));
  if ((xcr0_lo & 0x6) != 0x6) return false;
  if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) return false;
  return (ebx >> 5) & 1;
}

static bool has_fma(void) {
  unsigned int eax, ebx, ecx, edx;
  if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx)) return false;
  return (ecx >> 12) & 1;
}

typedef struct { PolyCtx *ctx; PolyUOp *sink; } Kernel;

/* ── Kernel builders (raw UOp, not frontend) ────────────────────────── */

static Kernel make_binop(PolyOps op, int N) {
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
  PolyUOp *alu = poly_uop2(ctx, op, POLY_FLOAT32, l0, l1, poly_arg_none());
  PolyUOp *st = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, i2, alu, poly_arg_none());
  PolyUOp *es[2] = { st, range };
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, es, 2, poly_arg_none());
  return (Kernel){ ctx, poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none()) };
}

static Kernel make_unary(PolyOps op, int N) {
  PolyCtx *ctx = poly_ctx_new();
  PolyDType pf = poly_dtype_ptr(POLY_FLOAT32, -1, POLY_ADDR_GLOBAL);
  PolyUOp *p0 = poly_uop0(ctx, POLY_OP_PARAM, pf, poly_arg_int(0));
  PolyUOp *p1 = poly_uop0(ctx, POLY_OP_PARAM, pf, poly_arg_int(1));
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(N));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));
  PolyUOp *i0 = poly_uop2(ctx, POLY_OP_INDEX, pf, p0, range, poly_arg_none());
  PolyUOp *i1 = poly_uop2(ctx, POLY_OP_INDEX, pf, p1, range, poly_arg_none());
  PolyUOp *l0 = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, i0, poly_arg_none());
  PolyUOp *alu = poly_uop1(ctx, op, POLY_FLOAT32, l0, poly_arg_none());
  PolyUOp *st = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, i1, alu, poly_arg_none());
  PolyUOp *es[2] = { st, range };
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, es, 2, poly_arg_none());
  return (Kernel){ ctx, poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none()) };
}

/* c[i] = sqrt(a[i] * b[i] + a[i]) */
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

/* ── Benchmark runner ────────────────────────────────────────────────── */

typedef struct {
  const char *name;
  double cpu_us, scalar_us, sse4_us, avx2_us;
} Row;

static PolyX64Program *compile_x64(PolyCtx *ctx, PolyUOp *sink, PolyRewriteOpts opts) {
  int nl;
  PolyUOp **lin = poly_linearize_ex(ctx, sink, opts, &nl);
  if (!lin) return NULL;
  int sz;
  uint8_t *code = poly_render_x64(lin, nl, &sz);
  free(lin);
  if (!code) return NULL;
  PolyX64Program *p = poly_compile_x64(code, sz);
  free(code);
  return p;
}

static double time_x64(PolyX64Program *p, void **args, int n, int iters) {
  poly_x64_program_call(p, args, n); /* warmup */
  double t0 = now_us();
  for (int i = 0; i < iters; i++) poly_x64_program_call(p, args, n);
  return (now_us() - t0) / iters;
}

static Row bench(const char *name, Kernel k, void **args, int n_args, int iters) {
  Row r = { .name = name };

  /* CPU (clang -O2) */
  {
    int nl; PolyUOp **lin = poly_linearize(k.ctx, k.sink, &nl);
    if (!lin) return r;
    char *src = poly_render_c(lin, nl, "bench_fn");
    free(lin);
    if (!src) return r;
    PolyProgram *cpu = poly_compile_c(src, "bench_fn");
    free(src);
    if (!cpu) return r;
    poly_program_call(cpu, args, n_args);
    double t0 = now_us();
    for (int i = 0; i < iters; i++) poly_program_call(cpu, args, n_args);
    r.cpu_us = (now_us() - t0) / iters;
    poly_program_destroy(cpu);
  }

  /* x64 scalar (no optimize) */
  {
    PolyRewriteOpts opts = {0};
    PolyX64Program *p = compile_x64(k.ctx, k.sink, opts);
    if (p) { r.scalar_us = time_x64(p, args, n_args, iters); poly_x64_program_destroy(p); }
  }

  /* x64 SSE vec4 (optimize, devec=0, max_vec_width=4, has_simd_int) */
  {
    PolyRewriteOpts opts = { .optimize = true, .devectorize = 0 };
    opts.caps.max_vec_width = 4;
    opts.caps.has_simd_int = true;
    PolyX64Program *p = compile_x64(k.ctx, k.sink, opts);
    if (p) { r.sse4_us = time_x64(p, args, n_args, iters); poly_x64_program_destroy(p); }
  }

  /* x64 AVX2 vec8+FMA (optimize, devec=0, max_vec_width=8, has_simd_int) */
  if (has_avx2()) {
    PolyRewriteOpts opts = { .optimize = true, .devectorize = 0 };
    opts.caps.max_vec_width = 8;
    opts.caps.has_mulacc = has_fma();
    opts.caps.has_simd_int = true;
    PolyX64Program *p = compile_x64(k.ctx, k.sink, opts);
    if (p) { r.avx2_us = time_x64(p, args, n_args, iters); poly_x64_program_destroy(p); }
  }

  return r;
}

static void print_header(void) {
  printf("  %-14s %10s %10s %10s %10s   sse/cpu  avx/cpu  avx/sse\n",
         "kernel", "cpu(us)", "scalar", "sse4", "avx2");
  printf("  %-14s %10s %10s %10s %10s   -------  -------  -------\n",
         "------", "------", "------", "----", "----");
}

static void print_row(Row r) {
  double sse_cpu = (r.sse4_us > 0 && r.cpu_us > 0) ? r.cpu_us / r.sse4_us : 0;
  double avx_cpu = (r.avx2_us > 0 && r.cpu_us > 0) ? r.cpu_us / r.avx2_us : 0;
  double avx_sse = (r.avx2_us > 0 && r.sse4_us > 0) ? r.sse4_us / r.avx2_us : 0;
  printf("  %-14s %10.1f %10.1f %10.1f %10.1f   %5.2fx    %5.2fx    %5.2fx\n",
         r.name, r.cpu_us, r.scalar_us, r.sse4_us,
         r.avx2_us > 0 ? r.avx2_us : 0.0,
         sse_cpu, avx_cpu, avx_sse);
}

int main(void) {
  printf("\n  x64 JIT benchmark (us per call)\n");
  printf("  cpu = clang -O2, scalar = x64 no-opt, sse4 = vec4, avx2 = vec8+FMA\n");
  printf("  ratio > 1 means left column is faster than right\n");
  printf("  AVX2: %s  FMA: %s\n\n", has_avx2() ? "yes" : "no", has_fma() ? "yes" : "no");

  int sizes[] = {1024, 16384, 262144};

  for (int si = 0; si < 3; si++) {
    int N = sizes[si];
    int iters = N <= 1024 ? 100000 : (N <= 16384 ? 10000 : 1000);

    float *a = malloc((size_t)N * sizeof(float));
    float *b = malloc((size_t)N * sizeof(float));
    float *c = malloc((size_t)N * sizeof(float));
    for (int i = 0; i < N; i++) {
      a[i] = 1.0f + 0.001f * i;
      b[i] = 2.0f + 0.001f * i;
    }

    printf("  N = %d (%d iters)\n", N, iters);
    print_header();

    void *args3[3] = { a, b, c };
    void *args2[2] = { a, c };
    Kernel k;

    /* Elementwise */
    k = make_binop(POLY_OP_ADD, N);
    print_row(bench("vecadd", k, args3, 3, iters));
    poly_ctx_destroy(k.ctx);

    k = make_unary(POLY_OP_NEG, N);
    print_row(bench("neg", k, args2, 2, iters));
    poly_ctx_destroy(k.ctx);

    k = make_chain(N);
    print_row(bench("chain", k, args3, 3, iters));
    poly_ctx_destroy(k.ctx);

    /* Transcendentals */
    k = make_unary(POLY_OP_EXP2, N);
    print_row(bench("exp2", k, args2, 2, iters));
    poly_ctx_destroy(k.ctx);

    k = make_unary(POLY_OP_LOG2, N);
    print_row(bench("log2", k, args2, 2, iters));
    poly_ctx_destroy(k.ctx);

    k = make_unary(POLY_OP_SIN, N);
    print_row(bench("sin", k, args2, 2, iters));
    poly_ctx_destroy(k.ctx);

    printf("\n");
    free(a); free(b); free(c);
  }
  return 0;
}
