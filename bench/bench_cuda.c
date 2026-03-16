#define _POSIX_C_SOURCE 200809L
/*
 * bench_cuda.c -- CPU vs GPU benchmark
 *
 * Compares poly_realize() on CPU-domain vs CUDA-domain bindings.
 *
 * Usage: ./build/bench_cuda [max_size]
 */

#ifdef POLY_HAS_CUDA

#include "../src/codegen.h"
#include "../src/frontend.h"
#include "../src/scheduler.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_us(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

/* Build CUDA-domain bindings (device-resident, persist across iterations) */
static int build_cuda_binds(PolyBufferBinding *out, PolyUOp **bufs,
                             float **host_ptrs, int n) {
  for (int i = 0; i < n; i++) {
    size_t nbytes = (size_t)bufs[i]->arg.i * poly_dtype_itemsize(
                      poly_dtype_scalar(bufs[i]->dtype));
    unsigned long long dptr = poly_cuda_alloc(nbytes);
    if (!dptr) return -1;
    if (host_ptrs[i])
      poly_cuda_copy_htod(dptr, host_ptrs[i], nbytes);
    else
      poly_cuda_memset(dptr, 0, nbytes);
    out[i].buffer = bufs[i];
    out[i].handle = (PolyBufferHandle){
      (void *)(uintptr_t)dptr, nbytes, POLY_DEVICE_CUDA, true
    };
  }
  return 0;
}

static void free_cuda_binds(PolyBufferBinding *bindings, int n) {
  for (int i = 0; i < n; i++)
    if (bindings[i].handle.owned)
      poly_cuda_free((unsigned long long)(uintptr_t)bindings[i].handle.ptr);
}

/* ── Bench: elementwise vecadd ───────────────────────────────────────── */

static void bench_vecadd(int n, int iters) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, n);
  PolyUOp *b = poly_buffer(ctx, POLY_FLOAT32, n);
  PolyUOp *c = poly_buffer(ctx, POLY_FLOAT32, n);
  PolyUOp *add = poly_alu2(ctx, POLY_OP_ADD, a, b);
  PolyUOp *store = poly_store_val(ctx, c, add);
  PolyUOp *sink = poly_sink1(ctx, store);

  float *ha = malloc(n * sizeof(float));
  float *hb = malloc(n * sizeof(float));
  float *hc = calloc(n, sizeof(float));
  for (int i = 0; i < n; i++) { ha[i] = (float)i * 0.001f; hb[i] = 1.0f; }

  /* CPU bindings */
  PolyBufferBinding cpu[] = { POLY_BIND_HOST(c, hc), POLY_BIND_HOST(a, ha), POLY_BIND_HOST(b, hb) };
  poly_realize(ctx, sink, cpu, 3); /* warmup */

  /* CUDA bindings (device-resident, reused across iterations) */
  PolyUOp *bufs[] = { c, a, b };
  float *ptrs[] = { NULL, ha, hb };
  PolyBufferBinding gpu[3];
  build_cuda_binds(gpu, bufs, ptrs, 3);
  poly_realize(ctx, sink, gpu, 3); /* warmup */

  double t0 = now_us();
  for (int it = 0; it < iters; it++) poly_realize(ctx, sink, cpu, 3);
  double cpu_us = (now_us() - t0) / iters;

  t0 = now_us();
  for (int it = 0; it < iters; it++) poly_realize(ctx, sink, gpu, 3);
  double gpu_us = (now_us() - t0) / iters;

  printf("  vecadd  N=%-8d  CPU: %8.0f us  GPU: %8.0f us  speedup: %.2fx\n",
         n, cpu_us, gpu_us, cpu_us / gpu_us);

  free_cuda_binds(gpu, 3);
  free(ha); free(hb); free(hc);
  poly_ctx_destroy(ctx);
}

/* ── Bench: elementwise mul ──────────────────────────────────────────── */

static void bench_mul(int n, int iters) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, n);
  PolyUOp *b = poly_buffer(ctx, POLY_FLOAT32, n);
  PolyUOp *c = poly_buffer(ctx, POLY_FLOAT32, n);
  PolyUOp *mul = poly_alu2(ctx, POLY_OP_MUL, a, b);
  PolyUOp *store = poly_store_val(ctx, c, mul);
  PolyUOp *sink = poly_sink1(ctx, store);

  float *ha = malloc(n * sizeof(float));
  float *hb = malloc(n * sizeof(float));
  float *hc = calloc(n, sizeof(float));
  for (int i = 0; i < n; i++) { ha[i] = (float)i * 0.001f; hb[i] = 2.0f; }

  PolyBufferBinding cpu[] = { POLY_BIND_HOST(c, hc), POLY_BIND_HOST(a, ha), POLY_BIND_HOST(b, hb) };
  poly_realize(ctx, sink, cpu, 3);

  PolyUOp *bufs[] = { c, a, b };
  float *ptrs[] = { NULL, ha, hb };
  PolyBufferBinding gpu[3];
  build_cuda_binds(gpu, bufs, ptrs, 3);
  poly_realize(ctx, sink, gpu, 3);

  double t0 = now_us();
  for (int it = 0; it < iters; it++) poly_realize(ctx, sink, cpu, 3);
  double cpu_us = (now_us() - t0) / iters;

  t0 = now_us();
  for (int it = 0; it < iters; it++) poly_realize(ctx, sink, gpu, 3);
  double gpu_us = (now_us() - t0) / iters;

  printf("  mul     N=%-8d  CPU: %8.0f us  GPU: %8.0f us  speedup: %.2fx\n",
         n, cpu_us, gpu_us, cpu_us / gpu_us);

  free_cuda_binds(gpu, 3);
  free(ha); free(hb); free(hc);
  poly_ctx_destroy(ctx);
}

/* ── Bench: reduce sum ───────────────────────────────────────────────── */

static void bench_reduce_sum(int n, int iters) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, n);
  PolyUOp *c = poly_buffer(ctx, POLY_FLOAT32, 1);
  int64_t axes[] = { 0 };
  PolyUOp *red = poly_reduce_axis(ctx, POLY_OP_ADD, a, axes, 1);
  PolyUOp *store = poly_store_val(ctx, c, red);
  PolyUOp *sink = poly_sink1(ctx, store);

  float *ha = malloc(n * sizeof(float));
  float hc = 0;
  for (int i = 0; i < n; i++) ha[i] = 1.0f;

  PolyBufferBinding cpu[] = { POLY_BIND_HOST(c, &hc), POLY_BIND_HOST(a, ha) };
  poly_realize(ctx, sink, cpu, 2);

  PolyUOp *bufs[] = { c, a };
  float *ptrs[] = { NULL, ha };
  PolyBufferBinding gpu[2];
  build_cuda_binds(gpu, bufs, ptrs, 2);
  poly_realize(ctx, sink, gpu, 2);

  double t0 = now_us();
  for (int it = 0; it < iters; it++) poly_realize(ctx, sink, cpu, 2);
  double cpu_us = (now_us() - t0) / iters;

  t0 = now_us();
  for (int it = 0; it < iters; it++) poly_realize(ctx, sink, gpu, 2);
  double gpu_us = (now_us() - t0) / iters;

  printf("  reduce  N=%-8d  CPU: %8.0f us  GPU: %8.0f us  speedup: %.2fx\n",
         n, cpu_us, gpu_us, cpu_us / gpu_us);

  free_cuda_binds(gpu, 2);
  free(ha);
  poly_ctx_destroy(ctx);
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
