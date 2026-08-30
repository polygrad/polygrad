#define _POSIX_C_SOURCE 200809L
/*
 * bench_cuda.c -- CPU vs GPU benchmark
 *
 * Compares the unified ctx-buffer + poly_realize_sink path on CPU-domain vs
 * CUDA-domain buffer views.
 *
 * Usage: ./build/bench_cuda [max_size]
 */

#ifdef POLY_HAS_CUDA

#include "../src/codegen/codegen.h"
#include "../src/device.h"
#include "../src/engine/realize.h"
#include "../src/frontend.h"
#include "../src/engine/schedule.h"
#include "buffer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_us(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

typedef struct {
  PolyUOp *buffer;
  PolyBuffer handle;
} BenchBufferView;

static size_t bench_buffer_nbytes(PolyUOp *buf) {
  return buf && buf->n_src == 1 && buf->src[0] &&
                 buf->src[0]->op == POLY_OP_CONST &&
                 buf->src[0]->arg.kind == POLY_ARG_INT
             ? (size_t)buf->src[0]->arg.i *
                   (size_t)poly_dtype_itemsize(buf->dtype)
             : 0;
}

static void attach_views(PolyCtx *ctx, BenchBufferView *views, int n_views) {
  for (int i = 0; i < n_views; i++) {
    PolyBuffer h = views[i].handle;
    if (h.nbytes == 0) h.nbytes = bench_buffer_nbytes(views[i].buffer);
    h.valid = true;
    poly_buffer_attach(ctx, views[i].buffer, &h);
  }
}

static int run_with_views(PolyCtx *ctx, PolyUOp *sink, BenchBufferView *views, int n_views) {
  attach_views(ctx, views, n_views);
  return poly_realize_sink(ctx, sink);
}

static BenchBufferView host_view(PolyUOp *buf, void *ptr) {
  return (BenchBufferView){
      .buffer = buf,
      .handle =
          {
              .ptr = ptr,
              .nbytes = 0,
              .device = POLY_DEVICE_CPU,
              .owned = false,
              .allocator = NULL,
              .src = NULL,
              .valid = true,
          },
  };
}

/* Build CUDA-domain views (device-resident, persist across iterations) */
static int build_cuda_views(BenchBufferView *out, PolyUOp **bufs, float **host_ptrs, int n) {
  for (int i = 0; i < n; i++) {
    size_t nbytes = bench_buffer_nbytes(bufs[i]);
    unsigned long long dptr = poly_cuda_alloc(nbytes);
    if (!dptr) return -1;
    if (host_ptrs[i])
      poly_cuda_copy_htod(dptr, host_ptrs[i], nbytes);
    else
      poly_cuda_memset(dptr, 0, nbytes);
    out[i] = (BenchBufferView){
        .buffer = bufs[i],
        .handle =
            {
                .ptr = (void *)(uintptr_t)dptr,
                .nbytes = nbytes,
                .device = POLY_DEVICE_CUDA,
                .owned = true,
                .allocator = NULL,
                .src = NULL,
                .valid = true,
            },
    };
  }
  return 0;
}

static void free_cuda_views(BenchBufferView *views, int n) {
  for (int i = 0; i < n; i++)
    if (views[i].handle.owned) poly_cuda_free((unsigned long long)(uintptr_t)views[i].handle.ptr);
}

/* ── Bench: elementwise vecadd ───────────────────────────────────────── */

static void bench_vecadd(int n, int iters) {
  PolyCtx *cpu_ctx = poly_ctx_new(), *gpu_ctx = poly_ctx_new();
  PolyUOp *a = poly_bench_buffer(cpu_ctx, POLY_FLOAT32, n, POLY_DEVICE_CPU);
  PolyUOp *b = poly_bench_buffer(cpu_ctx, POLY_FLOAT32, n, POLY_DEVICE_CPU);
  PolyUOp *c = poly_bench_buffer(cpu_ctx, POLY_FLOAT32, n, POLY_DEVICE_CPU);
  PolyUOp *ga = poly_bench_buffer(gpu_ctx, POLY_FLOAT32, n, POLY_DEVICE_CUDA);
  PolyUOp *gb = poly_bench_buffer(gpu_ctx, POLY_FLOAT32, n, POLY_DEVICE_CUDA);
  PolyUOp *gc = poly_bench_buffer(gpu_ctx, POLY_FLOAT32, n, POLY_DEVICE_CUDA);
  PolyUOp *sink = poly_sink1(cpu_ctx, poly_store_val(cpu_ctx, c, poly_alu2(cpu_ctx, POLY_OP_ADD, a, b)));
  PolyUOp *gpu_sink = poly_sink1(gpu_ctx, poly_store_val(gpu_ctx, gc, poly_alu2(gpu_ctx, POLY_OP_ADD, ga, gb)));

  float *ha = malloc(n * sizeof(float));
  float *hb = malloc(n * sizeof(float));
  float *hc = calloc(n, sizeof(float));
  for (int i = 0; i < n; i++) { ha[i] = (float)i * 0.001f; hb[i] = 1.0f; }

  BenchBufferView cpu[] = {host_view(c, hc), host_view(a, ha), host_view(b, hb)};
  run_with_views(cpu_ctx, sink, cpu, 3); /* warmup */

  /* CUDA views (device-resident, reused across iterations) */
  PolyUOp *bufs[] = { gc, ga, gb };
  float *ptrs[] = { NULL, ha, hb };
  BenchBufferView gpu[3];
  build_cuda_views(gpu, bufs, ptrs, 3);
  run_with_views(gpu_ctx, gpu_sink, gpu, 3); /* warmup */

  double t0 = now_us();
  for (int it = 0; it < iters; it++) run_with_views(cpu_ctx, sink, cpu, 3);
  double cpu_us = (now_us() - t0) / iters;

  t0 = now_us();
  for (int it = 0; it < iters; it++) run_with_views(gpu_ctx, gpu_sink, gpu, 3);
  double gpu_us = (now_us() - t0) / iters;

  printf("  vecadd  N=%-8d  CPU: %8.0f us  GPU: %8.0f us  speedup: %.2fx\n",
         n, cpu_us, gpu_us, cpu_us / gpu_us);

  free_cuda_views(gpu, 3);
  free(ha); free(hb); free(hc);
  poly_ctx_destroy(gpu_ctx);
  poly_ctx_destroy(cpu_ctx);
}

/* ── Bench: elementwise mul ──────────────────────────────────────────── */

static void bench_mul(int n, int iters) {
  PolyCtx *cpu_ctx = poly_ctx_new(), *gpu_ctx = poly_ctx_new();
  PolyUOp *a = poly_bench_buffer(cpu_ctx, POLY_FLOAT32, n, POLY_DEVICE_CPU);
  PolyUOp *b = poly_bench_buffer(cpu_ctx, POLY_FLOAT32, n, POLY_DEVICE_CPU);
  PolyUOp *c = poly_bench_buffer(cpu_ctx, POLY_FLOAT32, n, POLY_DEVICE_CPU);
  PolyUOp *ga = poly_bench_buffer(gpu_ctx, POLY_FLOAT32, n, POLY_DEVICE_CUDA);
  PolyUOp *gb = poly_bench_buffer(gpu_ctx, POLY_FLOAT32, n, POLY_DEVICE_CUDA);
  PolyUOp *gc = poly_bench_buffer(gpu_ctx, POLY_FLOAT32, n, POLY_DEVICE_CUDA);
  PolyUOp *sink = poly_sink1(cpu_ctx, poly_store_val(cpu_ctx, c, poly_alu2(cpu_ctx, POLY_OP_MUL, a, b)));
  PolyUOp *gpu_sink = poly_sink1(gpu_ctx, poly_store_val(gpu_ctx, gc, poly_alu2(gpu_ctx, POLY_OP_MUL, ga, gb)));

  float *ha = malloc(n * sizeof(float));
  float *hb = malloc(n * sizeof(float));
  float *hc = calloc(n, sizeof(float));
  for (int i = 0; i < n; i++) { ha[i] = (float)i * 0.001f; hb[i] = 2.0f; }

  BenchBufferView cpu[] = {host_view(c, hc), host_view(a, ha), host_view(b, hb)};
  run_with_views(cpu_ctx, sink, cpu, 3);

  PolyUOp *bufs[] = { gc, ga, gb };
  float *ptrs[] = { NULL, ha, hb };
  BenchBufferView gpu[3];
  build_cuda_views(gpu, bufs, ptrs, 3);
  run_with_views(gpu_ctx, gpu_sink, gpu, 3);

  double t0 = now_us();
  for (int it = 0; it < iters; it++) run_with_views(cpu_ctx, sink, cpu, 3);
  double cpu_us = (now_us() - t0) / iters;

  t0 = now_us();
  for (int it = 0; it < iters; it++) run_with_views(gpu_ctx, gpu_sink, gpu, 3);
  double gpu_us = (now_us() - t0) / iters;

  printf("  mul     N=%-8d  CPU: %8.0f us  GPU: %8.0f us  speedup: %.2fx\n",
         n, cpu_us, gpu_us, cpu_us / gpu_us);

  free_cuda_views(gpu, 3);
  free(ha); free(hb); free(hc);
  poly_ctx_destroy(gpu_ctx);
  poly_ctx_destroy(cpu_ctx);
}

/* ── Bench: fused chain ((a + b) * a) ────────────────────────────────── */

static void bench_chain(int n, int iters) {
  PolyCtx *cpu_ctx = poly_ctx_new(), *gpu_ctx = poly_ctx_new();
  PolyUOp *a = poly_bench_buffer(cpu_ctx, POLY_FLOAT32, n, POLY_DEVICE_CPU);
  PolyUOp *b = poly_bench_buffer(cpu_ctx, POLY_FLOAT32, n, POLY_DEVICE_CPU);
  PolyUOp *c = poly_bench_buffer(cpu_ctx, POLY_FLOAT32, n, POLY_DEVICE_CPU);
  PolyUOp *ga = poly_bench_buffer(gpu_ctx, POLY_FLOAT32, n, POLY_DEVICE_CUDA);
  PolyUOp *gb = poly_bench_buffer(gpu_ctx, POLY_FLOAT32, n, POLY_DEVICE_CUDA);
  PolyUOp *gc = poly_bench_buffer(gpu_ctx, POLY_FLOAT32, n, POLY_DEVICE_CUDA);
  PolyUOp *add = poly_alu2(cpu_ctx, POLY_OP_ADD, a, b);
  PolyUOp *gadd = poly_alu2(gpu_ctx, POLY_OP_ADD, ga, gb);
  PolyUOp *sink = poly_sink1(cpu_ctx, poly_store_val(cpu_ctx, c, poly_alu2(cpu_ctx, POLY_OP_MUL, add, a)));
  PolyUOp *gpu_sink = poly_sink1(gpu_ctx, poly_store_val(gpu_ctx, gc, poly_alu2(gpu_ctx, POLY_OP_MUL, gadd, ga)));

  float *ha = malloc(n * sizeof(float));
  float *hb = malloc(n * sizeof(float));
  float *hc = calloc(n, sizeof(float));
  for (int i = 0; i < n; i++) { ha[i] = (float)i * 0.001f; hb[i] = 1.0f; }

  BenchBufferView cpu[] = {host_view(c, hc), host_view(a, ha), host_view(b, hb)};
  run_with_views(cpu_ctx, sink, cpu, 3);

  PolyUOp *bufs[] = {gc, ga, gb};
  float *ptrs[] = {NULL, ha, hb};
  BenchBufferView gpu[3];
  build_cuda_views(gpu, bufs, ptrs, 3);
  run_with_views(gpu_ctx, gpu_sink, gpu, 3);

  double t0 = now_us();
  for (int it = 0; it < iters; it++) run_with_views(cpu_ctx, sink, cpu, 3);
  double cpu_us = (now_us() - t0) / iters;

  t0 = now_us();
  for (int it = 0; it < iters; it++) run_with_views(gpu_ctx, gpu_sink, gpu, 3);
  double gpu_us = (now_us() - t0) / iters;

  printf("  chain   N=%-8d  CPU: %8.0f us  GPU: %8.0f us  speedup: %.2fx\n",
         n, cpu_us, gpu_us, cpu_us / gpu_us);

  free_cuda_views(gpu, 3);
  free(ha); free(hb); free(hc);
  poly_ctx_destroy(gpu_ctx);
  poly_ctx_destroy(cpu_ctx);
}

/* ── Bench: exp2 ─────────────────────────────────────────────────────── */

static void bench_exp2(int n, int iters) {
  PolyCtx *cpu_ctx = poly_ctx_new(), *gpu_ctx = poly_ctx_new();
  PolyUOp *a = poly_bench_buffer(cpu_ctx, POLY_FLOAT32, n, POLY_DEVICE_CPU);
  PolyUOp *c = poly_bench_buffer(cpu_ctx, POLY_FLOAT32, n, POLY_DEVICE_CPU);
  PolyUOp *ga = poly_bench_buffer(gpu_ctx, POLY_FLOAT32, n, POLY_DEVICE_CUDA);
  PolyUOp *gc = poly_bench_buffer(gpu_ctx, POLY_FLOAT32, n, POLY_DEVICE_CUDA);
  PolyUOp *sink = poly_sink1(cpu_ctx, poly_store_val(cpu_ctx, c, poly_alu1(cpu_ctx, POLY_OP_EXP2, a)));
  PolyUOp *gpu_sink = poly_sink1(gpu_ctx, poly_store_val(gpu_ctx, gc, poly_alu1(gpu_ctx, POLY_OP_EXP2, ga)));

  float *ha = malloc(n * sizeof(float));
  float *hc = calloc(n, sizeof(float));
  for (int i = 0; i < n; i++) ha[i] = ((float)i / (float)n) - 0.5f;

  BenchBufferView cpu[] = {host_view(c, hc), host_view(a, ha)};
  run_with_views(cpu_ctx, sink, cpu, 2);

  PolyUOp *bufs[] = {gc, ga};
  float *ptrs[] = {NULL, ha};
  BenchBufferView gpu[2];
  build_cuda_views(gpu, bufs, ptrs, 2);
  run_with_views(gpu_ctx, gpu_sink, gpu, 2);

  double t0 = now_us();
  for (int it = 0; it < iters; it++) run_with_views(cpu_ctx, sink, cpu, 2);
  double cpu_us = (now_us() - t0) / iters;

  t0 = now_us();
  for (int it = 0; it < iters; it++) run_with_views(gpu_ctx, gpu_sink, gpu, 2);
  double gpu_us = (now_us() - t0) / iters;

  printf("  exp2    N=%-8d  CPU: %8.0f us  GPU: %8.0f us  speedup: %.2fx\n",
         n, cpu_us, gpu_us, cpu_us / gpu_us);

  free_cuda_views(gpu, 2);
  free(ha); free(hc);
  poly_ctx_destroy(gpu_ctx);
  poly_ctx_destroy(cpu_ctx);
}

/* ── Bench: reduce sum ───────────────────────────────────────────────── */

static void bench_reduce_sum(int n, int iters) {
  PolyCtx *cpu_ctx = poly_ctx_new(), *gpu_ctx = poly_ctx_new();
  PolyUOp *a = poly_bench_buffer(cpu_ctx, POLY_FLOAT32, n, POLY_DEVICE_CPU);
  PolyUOp *c = poly_bench_buffer(cpu_ctx, POLY_FLOAT32, 1, POLY_DEVICE_CPU);
  PolyUOp *ga = poly_bench_buffer(gpu_ctx, POLY_FLOAT32, n, POLY_DEVICE_CUDA);
  PolyUOp *gc = poly_bench_buffer(gpu_ctx, POLY_FLOAT32, 1, POLY_DEVICE_CUDA);
  int64_t axes[] = { 0 };
  PolyUOp *sink = poly_sink1(cpu_ctx, poly_store_val(cpu_ctx, c, poly_reduce_axis(cpu_ctx, POLY_OP_ADD, a, axes, 1)));
  PolyUOp *gpu_sink = poly_sink1(gpu_ctx, poly_store_val(gpu_ctx, gc, poly_reduce_axis(gpu_ctx, POLY_OP_ADD, ga, axes, 1)));

  float *ha = malloc(n * sizeof(float));
  float hc = 0;
  for (int i = 0; i < n; i++) ha[i] = 1.0f;

  BenchBufferView cpu[] = {host_view(c, &hc), host_view(a, ha)};
  run_with_views(cpu_ctx, sink, cpu, 2);

  PolyUOp *bufs[] = { gc, ga };
  float *ptrs[] = { NULL, ha };
  BenchBufferView gpu[2];
  build_cuda_views(gpu, bufs, ptrs, 2);
  run_with_views(gpu_ctx, gpu_sink, gpu, 2);

  double t0 = now_us();
  for (int it = 0; it < iters; it++) run_with_views(cpu_ctx, sink, cpu, 2);
  double cpu_us = (now_us() - t0) / iters;

  t0 = now_us();
  for (int it = 0; it < iters; it++) run_with_views(gpu_ctx, gpu_sink, gpu, 2);
  double gpu_us = (now_us() - t0) / iters;

  printf("  reduce  N=%-8d  CPU: %8.0f us  GPU: %8.0f us  speedup: %.2fx\n",
         n, cpu_us, gpu_us, cpu_us / gpu_us);

  free_cuda_views(gpu, 2);
  free(ha);
  poly_ctx_destroy(gpu_ctx);
  poly_ctx_destroy(cpu_ctx);
}

int main(int argc, char **argv) {
  (void)argc; (void)argv;

  if (!poly_cuda_available()) {
    printf("CUDA not available -- skipping GPU benchmark\n");
    return 0;
  }

  printf("\n  polygrad CPU vs CUDA benchmark\n");
  printf("  ==========================\n\n");

  int sizes[] = { 1024, 10000, 100000, 1000000 };
  int n_sizes = 4;
  int iters_small = 20;
  int iters_large = 5;

  if (argc > 1) {
    int max_size = atoi(argv[1]);
    for (int i = 0; i < n_sizes; i++) {
      if (sizes[i] > max_size) { n_sizes = i; break; }
    }
  }

  printf("  Elementwise add:\n");
  for (int i = 0; i < n_sizes; i++)
    bench_vecadd(sizes[i], sizes[i] >= 100000 ? iters_large : iters_small);

  printf("\n  Elementwise mul:\n");
  for (int i = 0; i < n_sizes; i++)
    bench_mul(sizes[i], sizes[i] >= 100000 ? iters_large : iters_small);

  printf("\n  Fused chain:\n");
  for (int i = 0; i < n_sizes; i++)
    bench_chain(sizes[i], sizes[i] >= 100000 ? iters_large : iters_small);

  printf("\n  Exp2:\n");
  for (int i = 0; i < n_sizes; i++)
    bench_exp2(sizes[i], sizes[i] >= 100000 ? iters_large : iters_small);

  printf("\n  Reduce sum:\n");
  for (int i = 0; i < n_sizes; i++)
    bench_reduce_sum(sizes[i], sizes[i] >= 100000 ? iters_large : iters_small);

  printf("\n");
  return 0;
}

#else /* !POLY_HAS_CUDA */

#include <stdio.h>
int main(void) {
  printf("CUDA support not compiled in (POLY_HAS_CUDA not defined)\n");
  return 0;
}

#endif /* POLY_HAS_CUDA */
