#define _POSIX_C_SOURCE 200809L
/*
 * bench_hip.c -- CPU vs HIP GPU benchmark
 *
 * Compares poly_realize() on CPU-domain vs HIP-domain bindings.
 * Mirrors bench_cuda.c for direct CUDA/HIP comparison.
 *
 * Usage: ./build/bench_hip [max_size]
 */

#ifdef POLY_HAS_HIP

#include "../src/codegen/codegen.h"
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

/* Build HIP-domain bindings (device-resident, persist across iterations) */
static int build_hip_binds(PolyBufferBinding *out, PolyUOp **bufs,
                            float **host_ptrs, int n) {
  for (int i = 0; i < n; i++) {
    if (!bufs[i] || bufs[i]->n_src != 1 || !bufs[i]->src[0] ||
        bufs[i]->src[0]->op != POLY_OP_CONST ||
        bufs[i]->src[0]->arg.kind != POLY_ARG_INT)
      return -1;
    size_t nbytes =
        (size_t)bufs[i]->src[0]->arg.i * poly_dtype_itemsize(bufs[i]->dtype);
    void *dptr = poly_hip_alloc(nbytes);
    if (!dptr) return -1;
    if (host_ptrs[i])
      poly_hip_copy_htod(dptr, host_ptrs[i], nbytes);
    else
      poly_hip_memset(dptr, 0, nbytes);
    out[i].buffer = bufs[i];
    out[i].handle = (PolyBufferHandle){ dptr, nbytes, POLY_DEVICE_HIP, true };
  }
  return 0;
}

static void free_hip_binds(PolyBufferBinding *bindings, int n) {
  for (int i = 0; i < n; i++)
    if (bindings[i].handle.owned)
      poly_hip_free(bindings[i].handle.ptr);
}

/* ── Bench: elementwise vecadd ───────────────────────────────────────── */

static void bench_vecadd(int n, int iters) {
  PolyCtx *cpu_ctx = poly_ctx_new(), *gpu_ctx = poly_ctx_new();
  PolyUOp *a = poly_bench_buffer(cpu_ctx, POLY_FLOAT32, n, POLY_DEVICE_CPU);
  PolyUOp *b = poly_bench_buffer(cpu_ctx, POLY_FLOAT32, n, POLY_DEVICE_CPU);
  PolyUOp *c = poly_bench_buffer(cpu_ctx, POLY_FLOAT32, n, POLY_DEVICE_CPU);
  PolyUOp *ga = poly_bench_buffer(gpu_ctx, POLY_FLOAT32, n, POLY_DEVICE_HIP);
  PolyUOp *gb = poly_bench_buffer(gpu_ctx, POLY_FLOAT32, n, POLY_DEVICE_HIP);
  PolyUOp *gc = poly_bench_buffer(gpu_ctx, POLY_FLOAT32, n, POLY_DEVICE_HIP);
  PolyUOp *sink = poly_sink1(cpu_ctx, poly_store_val(cpu_ctx, c, poly_alu2(cpu_ctx, POLY_OP_ADD, a, b)));
  PolyUOp *gpu_sink = poly_sink1(gpu_ctx, poly_store_val(gpu_ctx, gc, poly_alu2(gpu_ctx, POLY_OP_ADD, ga, gb)));

  float *ha = malloc(n * sizeof(float));
  float *hb = malloc(n * sizeof(float));
  float *hc = calloc(n, sizeof(float));
  for (int i = 0; i < n; i++) { ha[i] = (float)i * 0.001f; hb[i] = 1.0f; }

  PolyBufferBinding cpu[] = { POLY_BIND_HOST(c, hc), POLY_BIND_HOST(a, ha), POLY_BIND_HOST(b, hb) };
  poly_realize_with_bindings(cpu_ctx, sink, cpu, 3);

  PolyUOp *bufs[] = { gc, ga, gb };
  float *ptrs[] = { NULL, ha, hb };
  PolyBufferBinding gpu[3];
  build_hip_binds(gpu, bufs, ptrs, 3);
  poly_realize_with_bindings(gpu_ctx, gpu_sink, gpu, 3);

  double t0 = now_us();
  for (int it = 0; it < iters; it++) poly_realize_with_bindings(cpu_ctx, sink, cpu, 3);
  double cpu_us = (now_us() - t0) / iters;

  t0 = now_us();
  for (int it = 0; it < iters; it++) poly_realize_with_bindings(gpu_ctx, gpu_sink, gpu, 3);
  double gpu_us = (now_us() - t0) / iters;

  printf("  vecadd  N=%-8d  CPU: %8.0f us  GPU: %8.0f us  speedup: %.2fx\n",
         n, cpu_us, gpu_us, cpu_us / gpu_us);

  free_hip_binds(gpu, 3);
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
  PolyUOp *ga = poly_bench_buffer(gpu_ctx, POLY_FLOAT32, n, POLY_DEVICE_HIP);
  PolyUOp *gb = poly_bench_buffer(gpu_ctx, POLY_FLOAT32, n, POLY_DEVICE_HIP);
  PolyUOp *gc = poly_bench_buffer(gpu_ctx, POLY_FLOAT32, n, POLY_DEVICE_HIP);
  PolyUOp *sink = poly_sink1(cpu_ctx, poly_store_val(cpu_ctx, c, poly_alu2(cpu_ctx, POLY_OP_MUL, a, b)));
  PolyUOp *gpu_sink = poly_sink1(gpu_ctx, poly_store_val(gpu_ctx, gc, poly_alu2(gpu_ctx, POLY_OP_MUL, ga, gb)));

  float *ha = malloc(n * sizeof(float));
  float *hb = malloc(n * sizeof(float));
  float *hc = calloc(n, sizeof(float));
  for (int i = 0; i < n; i++) { ha[i] = (float)i * 0.001f; hb[i] = 2.0f; }

  PolyBufferBinding cpu[] = { POLY_BIND_HOST(c, hc), POLY_BIND_HOST(a, ha), POLY_BIND_HOST(b, hb) };
  poly_realize_with_bindings(cpu_ctx, sink, cpu, 3);

  PolyUOp *bufs[] = { gc, ga, gb };
  float *ptrs[] = { NULL, ha, hb };
  PolyBufferBinding gpu[3];
  build_hip_binds(gpu, bufs, ptrs, 3);
  poly_realize_with_bindings(gpu_ctx, gpu_sink, gpu, 3);

  double t0 = now_us();
  for (int it = 0; it < iters; it++) poly_realize_with_bindings(cpu_ctx, sink, cpu, 3);
  double cpu_us = (now_us() - t0) / iters;

  t0 = now_us();
  for (int it = 0; it < iters; it++) poly_realize_with_bindings(gpu_ctx, gpu_sink, gpu, 3);
  double gpu_us = (now_us() - t0) / iters;

  printf("  mul     N=%-8d  CPU: %8.0f us  GPU: %8.0f us  speedup: %.2fx\n",
         n, cpu_us, gpu_us, cpu_us / gpu_us);

  free_hip_binds(gpu, 3);
  free(ha); free(hb); free(hc);
  poly_ctx_destroy(gpu_ctx);
  poly_ctx_destroy(cpu_ctx);
}

/* ── Bench: reduce sum ───────────────────────────────────────────────── */

static void bench_reduce_sum(int n, int iters) {
  PolyCtx *cpu_ctx = poly_ctx_new(), *gpu_ctx = poly_ctx_new();
  PolyUOp *a = poly_bench_buffer(cpu_ctx, POLY_FLOAT32, n, POLY_DEVICE_CPU);
  PolyUOp *c = poly_bench_buffer(cpu_ctx, POLY_FLOAT32, 1, POLY_DEVICE_CPU);
  PolyUOp *ga = poly_bench_buffer(gpu_ctx, POLY_FLOAT32, n, POLY_DEVICE_HIP);
  PolyUOp *gc = poly_bench_buffer(gpu_ctx, POLY_FLOAT32, 1, POLY_DEVICE_HIP);
  int64_t axes[] = { 0 };
  PolyUOp *sink = poly_sink1(cpu_ctx, poly_store_val(cpu_ctx, c, poly_reduce_axis(cpu_ctx, POLY_OP_ADD, a, axes, 1)));
  PolyUOp *gpu_sink = poly_sink1(gpu_ctx, poly_store_val(gpu_ctx, gc, poly_reduce_axis(gpu_ctx, POLY_OP_ADD, ga, axes, 1)));

  float *ha = malloc(n * sizeof(float));
  float hc = 0;
  for (int i = 0; i < n; i++) ha[i] = 1.0f;

  PolyBufferBinding cpu[] = { POLY_BIND_HOST(c, &hc), POLY_BIND_HOST(a, ha) };
  poly_realize_with_bindings(cpu_ctx, sink, cpu, 2);

  PolyUOp *bufs[] = { gc, ga };
  float *ptrs[] = { NULL, ha };
  PolyBufferBinding gpu[2];
  build_hip_binds(gpu, bufs, ptrs, 2);
  poly_realize_with_bindings(gpu_ctx, gpu_sink, gpu, 2);

  double t0 = now_us();
  for (int it = 0; it < iters; it++) poly_realize_with_bindings(cpu_ctx, sink, cpu, 2);
  double cpu_us = (now_us() - t0) / iters;

  t0 = now_us();
  for (int it = 0; it < iters; it++) poly_realize_with_bindings(gpu_ctx, gpu_sink, gpu, 2);
  double gpu_us = (now_us() - t0) / iters;

  printf("  reduce  N=%-8d  CPU: %8.0f us  GPU: %8.0f us  speedup: %.2fx\n",
         n, cpu_us, gpu_us, cpu_us / gpu_us);

  free_hip_binds(gpu, 2);
  free(ha);
  poly_ctx_destroy(gpu_ctx);
  poly_ctx_destroy(cpu_ctx);
}

int main(int argc, char **argv) {
  (void)argc; (void)argv;

  if (!poly_hip_available()) {
    printf("HIP not available -- skipping GPU benchmark\n");
    return 0;
  }

  printf("\n  polygrad CPU vs HIP benchmark (%s, wave_size=%d)\n",
         poly_hip_arch(), poly_hip_wave_size());
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

#else /* !POLY_HAS_HIP */

#include <stdio.h>
int main(void) {
  printf("HIP support not compiled in (POLY_HAS_HIP not defined)\n");
  return 0;
}

#endif /* POLY_HAS_HIP */
