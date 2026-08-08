#define _POSIX_C_SOURCE 200809L

/*
 * bench_smoke.c -- fast absolute Polygrad core regression benchmark.
 *
 * This intentionally times C tensor realization hot paths rather than broad
 * frontend ratios. It is small enough for commit-time smoke checks, while
 * still covering the scheduler/range/minmax/rewrite paths that have regressed
 * before.
 */

#include "polygrad.h"
#include "device.h"
#include "frontend.h"
#include "tensor.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef double (*BenchSampleFn)(int iters, int warmup);

typedef struct {
  const char *name;
  BenchSampleFn run;
  double threshold_pct;
  double threshold_abs_us;
} BenchSpec;

static double now_us(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1000000.0 + (double)ts.tv_nsec / 1000.0;
}

static int cmp_double(const void *a, const void *b) {
  double da = *(const double *)a;
  double db = *(const double *)b;
  return (da > db) - (da < db);
}

static double median_of(const double *vals, int n) {
  if (!vals || n <= 0) return NAN;
  double *tmp = malloc((size_t)n * sizeof(double));
  if (!tmp) return NAN;
  memcpy(tmp, vals, (size_t)n * sizeof(double));
  qsort(tmp, (size_t)n, sizeof(double), cmp_double);
  double ret = (n & 1) ? tmp[n / 2] : 0.5 * (tmp[n / 2 - 1] + tmp[n / 2]);
  free(tmp);
  return ret;
}

static double mad_of(const double *vals, int n, double median) {
  if (!vals || n <= 0 || isnan(median)) return NAN;
  double *dev = malloc((size_t)n * sizeof(double));
  if (!dev) return NAN;
  for (int i = 0; i < n; i++)
    dev[i] = fabs(vals[i] - median);
  double ret = median_of(dev, n);
  free(dev);
  return ret;
}

static PolyTensor *make_host_tensor(PolyCtx *ctx, float *data, size_t n, int64_t *shape, int ndim) {
  PolyTensor *source =
      poly_tensor_from_host(ctx, data, n * sizeof(float), POLY_FLOAT32, shape, ndim);
  return source ? poly_tensor_to_device(ctx, source, POLY_DEVICE_CPU) : NULL;
}

static void realize_or_die(PolyCtx *ctx, PolyTensor *tensor, const char *name) {
  PolyTensor *out = NULL;
  if (!tensor || poly_realize_tensors(ctx, &tensor, 1, &out) != 0) {
    fprintf(stderr, "bench_smoke: %s realize failed\n", name);
    exit(2);
  }
  (void)out;
}

static double bench_sum_1024(int iters, int warmup) {
  const int64_t n = 1024;
  int64_t shape[1] = {n};
  float *data = malloc((size_t)n * sizeof(float));
  if (!data) exit(2);
  for (int64_t i = 0; i < n; i++)
    data[i] = -1.0f + 2.0f * (float)i / (float)(n - 1);

  PolyCtx *ctx = poly_ctx_new();
  poly_ctx_set_preferred_device(ctx, POLY_DEVICE_CPU);
  PolyTensor *a = make_host_tensor(ctx, data, (size_t)n, shape, 1);
  int64_t axes[1] = {0};

  for (int i = 0; i < warmup; i++) {
    PolyTensor *sum = poly_tensor_sum(ctx, a, axes, 1, false);
    realize_or_die(ctx, sum, "sum_1024 warmup");
  }

  double t0 = now_us();
  for (int i = 0; i < iters; i++) {
    PolyTensor *sum = poly_tensor_sum(ctx, a, axes, 1, false);
    realize_or_die(ctx, sum, "sum_1024");
  }
  double ret = (now_us() - t0) / (double)iters;

  poly_ctx_destroy(ctx);
  free(data);
  return ret;
}

static double bench_movement_1024(int iters, int warmup) {
  const int64_t n = 1024;
  float *data = malloc((size_t)n * sizeof(float));
  if (!data) exit(2);
  for (int64_t i = 0; i < n; i++)
    data[i] = (float)i;

  PolyCtx *ctx = poly_ctx_new();
  poly_ctx_set_preferred_device(ctx, POLY_DEVICE_CPU);
  int64_t shape[2] = {32, 32};
  PolyTensor *a = make_host_tensor(ctx, data, (size_t)n, shape, 2);
  int64_t reshaped[2] = {16, 64};
  int64_t perm[2] = {1, 0};

  for (int i = 0; i < warmup; i++) {
    PolyTensor *r = poly_tensor_reshape(ctx, a, reshaped, 2);
    PolyTensor *p = poly_tensor_permute(ctx, r, perm, 2);
    PolyTensor *c = poly_tensor_contiguous(ctx, p);
    realize_or_die(ctx, c, "movement_1024 warmup");
  }

  double t0 = now_us();
  for (int i = 0; i < iters; i++) {
    PolyTensor *r = poly_tensor_reshape(ctx, a, reshaped, 2);
    PolyTensor *p = poly_tensor_permute(ctx, r, perm, 2);
    PolyTensor *c = poly_tensor_contiguous(ctx, p);
    realize_or_die(ctx, c, "movement_1024");
  }
  double ret = (now_us() - t0) / (double)iters;

  poly_ctx_destroy(ctx);
  free(data);
  return ret;
}

static double bench_chain_1024(int iters, int warmup) {
  const int64_t n = 1024;
  int64_t shape[1] = {n};
  float *a_data = malloc((size_t)n * sizeof(float));
  float *b_data = malloc((size_t)n * sizeof(float));
  if (!a_data || !b_data) exit(2);
  for (int64_t i = 0; i < n; i++) {
    a_data[i] = -0.75f + 2.0f * (float)i / (float)(n - 1);
    b_data[i] = 0.5f + (float)i / (float)(n - 1);
  }

  PolyCtx *ctx = poly_ctx_new();
  poly_ctx_set_preferred_device(ctx, POLY_DEVICE_CPU);
  PolyTensor *a = make_host_tensor(ctx, a_data, (size_t)n, shape, 1);
  PolyTensor *b = make_host_tensor(ctx, b_data, (size_t)n, shape, 1);
  int f32 = poly_dtype_id_by_name("float32");
  PolyTensor *two = poly_tensor_const_float_by_id(ctx, 2.0, f32, POLY_DEVICE_CPU);

  for (int i = 0; i < warmup; i++) {
    PolyTensor *sum = poly_tensor_alu2(ctx, POLY_OP_ADD, a, b);
    PolyTensor *scaled = poly_tensor_alu2(ctx, POLY_OP_MUL, sum, two);
    PolyTensor *expr = poly_tensor_alu2(ctx, POLY_OP_SUB, scaled, b);
    realize_or_die(ctx, expr, "chain_1024 warmup");
  }

  double t0 = now_us();
  for (int i = 0; i < iters; i++) {
    PolyTensor *sum = poly_tensor_alu2(ctx, POLY_OP_ADD, a, b);
    PolyTensor *scaled = poly_tensor_alu2(ctx, POLY_OP_MUL, sum, two);
    PolyTensor *expr = poly_tensor_alu2(ctx, POLY_OP_SUB, scaled, b);
    realize_or_die(ctx, expr, "chain_1024");
  }
  double ret = (now_us() - t0) / (double)iters;

  poly_ctx_destroy(ctx);
  free(a_data);
  free(b_data);
  return ret;
}

static double bench_matmul_16(int iters, int warmup) {
  const int n = 16;
  const size_t numel = (size_t)n * (size_t)n;
  float *a_data = malloc(numel * sizeof(float));
  float *b_data = malloc(numel * sizeof(float));
  if (!a_data || !b_data) exit(2);
  for (int r = 0; r < n; r++) {
    for (int c = 0; c < n; c++) {
      a_data[r * n + c] = ((float)(r * n + c) - 50.0f) / 64.0f;
      b_data[r * n + c] = ((float)(c * n + r) + 3.0f) / 32.0f;
    }
  }

  PolyCtx *ctx = poly_ctx_new();
  poly_ctx_set_preferred_device(ctx, POLY_DEVICE_CPU);
  int64_t shape[2] = {n, n};
  PolyTensor *a = make_host_tensor(ctx, a_data, numel, shape, 2);
  PolyTensor *b = make_host_tensor(ctx, b_data, numel, shape, 2);

  for (int i = 0; i < warmup; i++) {
    PolyTensor *dot = poly_tensor_dot(ctx, a, b);
    realize_or_die(ctx, dot, "matmul_16 warmup");
  }

  double t0 = now_us();
  for (int i = 0; i < iters; i++) {
    PolyTensor *dot = poly_tensor_dot(ctx, a, b);
    realize_or_die(ctx, dot, "matmul_16");
  }
  double ret = (now_us() - t0) / (double)iters;

  poly_ctx_destroy(ctx);
  free(a_data);
  free(b_data);
  return ret;
}

static const BenchSpec BENCHES[] = {
    {"sum_1024", bench_sum_1024, 0.10, 5.0},
    {"movement_1024", bench_movement_1024, 0.10, 5.0},
    {"chain_1024", bench_chain_1024, 0.10, 5.0},
    {"matmul_16", bench_matmul_16, 0.12, 8.0},
};

static int parse_int_arg(int argc, char **argv, const char *name, int def) {
  for (int i = 1; i + 1 < argc; i++) {
    if (strcmp(argv[i], name) == 0) return atoi(argv[i + 1]);
  }
  return def;
}

static bool has_arg(int argc, char **argv, const char *name) {
  for (int i = 1; i < argc; i++)
    if (strcmp(argv[i], name) == 0) return true;
  return false;
}

static void print_usage(const char *argv0) {
  fprintf(stderr, "Usage: %s [--samples N] [--iters N] [--warmup N]\n", argv0);
}

int main(int argc, char **argv) {
  if (has_arg(argc, argv, "--help") || has_arg(argc, argv, "-h")) {
    print_usage(argv[0]);
    return 0;
  }

  int samples = parse_int_arg(argc, argv, "--samples", 7);
  int iters = parse_int_arg(argc, argv, "--iters", 300);
  int warmup = parse_int_arg(argc, argv, "--warmup", 30);
  if (samples <= 0 || iters <= 0 || warmup < 0) {
    print_usage(argv[0]);
    return 2;
  }

  printf("{\n");
  printf("  \"version\": 1,\n");
  printf("  \"runner\": \"bench_smoke.c\",\n");
  printf("  \"samples\": %d,\n", samples);
  printf("  \"iters\": %d,\n", iters);
  printf("  \"warmup\": %d,\n", warmup);
  printf("  \"benchmarks\": {\n");

  int n_benches = (int)(sizeof(BENCHES) / sizeof(BENCHES[0]));
  for (int bi = 0; bi < n_benches; bi++) {
    const BenchSpec *spec = &BENCHES[bi];
    double *vals = calloc((size_t)samples, sizeof(double));
    if (!vals) return 2;
    for (int si = 0; si < samples; si++)
      vals[si] = spec->run(iters, warmup);

    double med = median_of(vals, samples);
    double mad = mad_of(vals, samples, med);
    double min_v = vals[0], max_v = vals[0];
    for (int si = 1; si < samples; si++) {
      if (vals[si] < min_v) min_v = vals[si];
      if (vals[si] > max_v) max_v = vals[si];
    }

    printf("    \"%s\": {\n", spec->name);
    printf("      \"median_us\": %.6f,\n", med);
    printf("      \"mad_us\": %.6f,\n", mad);
    printf("      \"min_us\": %.6f,\n", min_v);
    printf("      \"max_us\": %.6f,\n", max_v);
    printf("      \"threshold_pct\": %.6f,\n", spec->threshold_pct);
    printf("      \"threshold_abs_us\": %.6f,\n", spec->threshold_abs_us);
    printf("      \"samples_us\": [");
    for (int si = 0; si < samples; si++) {
      printf("%s%.6f", si ? ", " : "", vals[si]);
    }
    printf("]\n");
    printf("    }%s\n", bi == n_benches - 1 ? "" : ",");
    free(vals);
  }

  printf("  }\n");
  printf("}\n");
  return 0;
}
