/*
 * bench/train_mlp_c.c — C MLP training benchmark
 *
 * Matches bench/train_mlp_python.py for direct comparison:
 * same configurations, same loss function, same hyperparameters.
 *
 * Uses the recipe API (poly_mlp_create + poly_train_plan) to
 * measure training overhead with pre-compiled kernels.
 *
 * Build: make bench-train-c
 * Run:   ./build/bench_train_mlp_c
 */

#define _GNU_SOURCE
#include "../src/recipe.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ── Timer ──────────────────────────────────────────────────────────── */

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* ── Config ─────────────────────────────────────────────────────────── */

typedef struct {
  const char *name;
  int dims[8];
  int n_dims;
  int batch;
  int iters;
} Config;

static Config CONFIGS[] = {
  { "tiny",   {4, 4, 2},       3, 4,  100 },
  { "small",  {16, 8, 4},      3, 8,  50  },
  { "medium", {64, 32, 16},    3, 16, 20  },
  { "large",  {784, 128, 10},  3, 4,  5   },
};
#define N_CONFIGS 4

static float LR = 0.001f;
static int N_WARMUP = 1;

/* ── Build training graph ───────────────────────────────────────────── */

static PolyTrainPlan *build_plan(Config *cfg) {
  PolyCtx *ctx = poly_ctx_new();
  int batch = cfg->batch;
  int in_dim = cfg->dims[0];
  int out_dim = cfg->dims[cfg->n_dims - 1];

  /* Create model */
  PolyMLP *model = poly_mlp_create(ctx, cfg->dims, cfg->n_dims);
  if (!model) { fprintf(stderr, "mlp_create failed\n"); return NULL; }

  /* Input and target buffers (1D for data binding) */
  PolyUOp *x_buf = poly_buffer_f32(ctx, batch * in_dim);
  PolyUOp *y_buf = poly_buffer_f32(ctx, batch * out_dim);

  /* Reshape to 2D for shape-aware ops */
  int64_t x_shape[] = { batch, in_dim };
  PolyUOp *x_2d = poly_reshape(ctx, x_buf, x_shape, 2);

  int64_t y_shape[] = { batch, out_dim };
  PolyUOp *y_2d = poly_reshape(ctx, y_buf, y_shape, 2);

  /* Forward pass: output shape (batch, out_dim) */
  PolyUOp *pred = poly_mlp_forward(model, x_2d, batch);

  /* MSE loss: ((pred - y)^2).mean() */
  PolyUOp *diff = poly_alu2(ctx, POLY_OP_SUB, pred, y_2d);
  PolyUOp *sq = poly_alu2(ctx, POLY_OP_MUL, diff, diff);

  /* Flatten to 1D and reduce to scalar */
  int total = batch * out_dim;
  int64_t flat_shape[] = { total };
  PolyUOp *sq_flat = poly_reshape(ctx, sq, flat_shape, 1);
  int64_t loss_shape[8]; int loss_ndim;
  PolyUOp *loss = poly_mean_reduce(ctx, sq_flat, flat_shape, 1, 0, 0,
                                    loss_shape, &loss_ndim);

  /* Create training plan */
  PolyTrainPlan *plan = poly_train_plan_create(
      ctx, loss,
      x_buf, batch * in_dim,
      y_buf, batch * out_dim,
      model->param_bufs, model->param_datas, model->param_sizes,
      model->n_params, LR);

  if (!plan) {
    fprintf(stderr, "train_plan_create failed\n");
    poly_mlp_free(model);
    return NULL;
  }

  /* Don't free model params — plan holds references to them */
  free(model->param_bufs);
  free(model->param_sizes);
  free(model->layer_sizes);
  free(model);

  return plan;
}

/* ── Random data ────────────────────────────────────────────────────── */

static float randf(void) {
  return (float)rand() / (float)RAND_MAX;
}

/* Box-Muller for normal distribution */
static float randn(void) {
  float u1 = randf();
  float u2 = randf();
  if (u1 < 1e-10f) u1 = 1e-10f;
  return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
}

/* ── Main ───────────────────────────────────────────────────────────── */

int main(void) {
  printf("======================================================================\n");
  printf("C MLP Training Benchmark (recipe API)\n");
  printf("======================================================================\n\n");

  for (int c = 0; c < N_CONFIGS; c++) {
    Config *cfg = &CONFIGS[c];
    int batch = cfg->batch;
    int in_dim = cfg->dims[0];
    int out_dim = cfg->dims[cfg->n_dims - 1];
    int n_iters = cfg->iters;

    printf("--- %s: MLP(", cfg->name);
    for (int i = 0; i < cfg->n_dims; i++) {
      if (i > 0) printf(" -> ");
      printf("%d", cfg->dims[i]);
    }
    printf("), batch=%d ---\n", batch);

    srand(42);

    /* Build plan (includes model creation + graph build + gradient build) */
    double t_build_start = now_ms();
    PolyTrainPlan *plan = build_plan(cfg);
    double t_build_end = now_ms();

    if (!plan) {
      printf("  FAILED to build plan\n\n");
      continue;
    }

    printf("  Plan build:     %8.1f ms\n", t_build_end - t_build_start);

    /* Compile (first realize of all kernels) */
    double t_compile_start = now_ms();
    int ret = poly_train_plan_compile(plan);
    double t_compile_end = now_ms();

    if (ret != 0) {
      printf("  FAILED to compile plan\n\n");
      poly_train_plan_free(plan);
      continue;
    }

    printf("  Compile:        %8.1f ms\n", t_compile_end - t_compile_start);

    /* Generate random data (small * 0.01 to match Python benchmark) */
    float *x_data = malloc(batch * in_dim * sizeof(float));
    float *y_data = malloc(batch * out_dim * sizeof(float));
    for (int i = 0; i < batch * in_dim; i++) x_data[i] = randn() * 0.01f;
    for (int i = 0; i < batch * out_dim; i++) y_data[i] = randn() * 0.01f;

    /* Training loop */
    double cold_time = 0.0;
    double warm_total = 0.0;
    float first_loss = 0.0f, last_loss = 0.0f;

    for (int i = 0; i < N_WARMUP + n_iters; i++) {
      double t0 = now_ms();
      float loss = poly_train_step(plan, x_data, y_data);
      double t1 = now_ms();

      double elapsed = t1 - t0;

      if (i == 0) {
        cold_time = elapsed;
        first_loss = loss;
      }
      if (i >= N_WARMUP) {
        warm_total += elapsed;
      }
      last_loss = loss;
    }

    double warm_avg = warm_total / n_iters;
    double speedup = cold_time / warm_avg;

    printf("  Cold (iter 0):  %8.1f ms\n", cold_time);
    printf("  Warm (avg):     %8.1f ms\n", warm_avg);
    printf("  Speedup:        %.0fx (cold/warm)\n", speedup);
    printf("  Loss:           %.6f -> %.6f\n", first_loss, last_loss);
    if (warm_avg > 0)
      printf("  Throughput:     %.1f iter/sec (warm)\n", 1000.0 / warm_avg);
    printf("\n");

    free(x_data);
    free(y_data);
    poly_train_plan_free(plan);

    /* Flush caches between configs for clean measurement */
    poly_cpu_cache_flush();
    poly_sched_cache_flush();
  }

  printf("======================================================================\n");
  printf("C recipe: graph build + schedule + compile done ONCE.\n");
  printf("Warm iterations are pure kernel dispatch (schedule cache hit).\n");
  printf("======================================================================\n");

  return 0;
}
