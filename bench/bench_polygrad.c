#define _POSIX_C_SOURCE 200809L
/*
 * bench_polygrad.c — Benchmark: build graph → render C → compile → execute
 *
 * Extended benchmark scope:
 * - Forward elementwise kernels (legacy section)
 * - Graph kernels (reduce + movement chains)
 * - Autograd kernels
 *
 * Usage: ./build/bench_polygrad [N] [iters_elementwise] [iters_graph]
 */

#include "../src/codegen.h"
#include "../src/sched.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define LN2_F 0.69314718055994530942f

static double now_us(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

typedef struct {
  PolyCtx *ctx;
  PolyUOp *sink;
} Kernel;

static Kernel make_binop(PolyOps op, int n) {
  PolyCtx *ctx = poly_ctx_new();
  PolyDType ptr_f32 = poly_dtype_ptr(POLY_FLOAT32, -1, POLY_ADDR_GLOBAL);
  PolyUOp *p0 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(0));
  PolyUOp *p1 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(1));
  PolyUOp *p2 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(2));
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(n));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));
  PolyUOp *idx0 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p0, range, poly_arg_none());
  PolyUOp *idx1 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p1, range, poly_arg_none());
  PolyUOp *idx2 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p2, range, poly_arg_none());
  PolyUOp *ld0 = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx0, poly_arg_none());
  PolyUOp *ld1 = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx1, poly_arg_none());
  PolyUOp *alu = poly_uop2(ctx, op, POLY_FLOAT32, ld0, ld1, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx2, alu, poly_arg_none());
  PolyUOp *end_src[2] = { store, range };
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());
  return (Kernel){ ctx, sink };
}

static Kernel make_unop(PolyOps op, int n) {
  PolyCtx *ctx = poly_ctx_new();
  PolyDType ptr_f32 = poly_dtype_ptr(POLY_FLOAT32, -1, POLY_ADDR_GLOBAL);
  PolyUOp *p0 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(0));
  PolyUOp *p1 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(1));
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(n));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));
  PolyUOp *idx0 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p0, range, poly_arg_none());
  PolyUOp *idx1 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p1, range, poly_arg_none());
  PolyUOp *ld = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx0, poly_arg_none());
  PolyUOp *alu = poly_uop1(ctx, op, POLY_FLOAT32, ld, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx1, alu, poly_arg_none());
  PolyUOp *end_src[2] = { store, range };
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());
  return (Kernel){ ctx, sink };
}

static int compile_from_sink(PolyCtx *ctx, PolyUOp *sink, const char *fn_name, PolyProgram **prog_out) {
  int n_lin = 0;
  PolyUOp *kernel = poly_schedule(ctx, sink);
  if (!kernel) return 0;
  PolyUOp **lin = poly_linearize(ctx, kernel, &n_lin);
  if (!lin) return 0;
  char *src = poly_render_c(lin, n_lin, fn_name);
  if (!src) {
    free(lin);
    return 0;
  }
  PolyProgram *prog = poly_compile_c(src, fn_name);
  free(src);
  free(lin);
  if (!prog) return 0;
  *prog_out = prog;
  return 1;
}

static void print_row(const char *name, int iters, double compile_us, double exec_us, int ok) {
  printf("%-22s %8d %12.0f %12.1f %10s\n",
         name, iters, compile_us, exec_us, ok ? "PASS" : "FAIL");
}

static int run_binop_case(const char *name, PolyOps op, int n, int iters) {
  int ok = 1;
  char fn_name[64];
  snprintf(fn_name, sizeof(fn_name), "k_%s", name);
  float *a = malloc((size_t)n * sizeof(float));
  float *b = malloc((size_t)n * sizeof(float));
  float *c = malloc((size_t)n * sizeof(float));
  if (!a || !b || !c) {
    free(a); free(b); free(c);
    printf("%-22s %8d %12s %12s %10s\n", name, iters, "OOM", "-", "FAIL");
    return 0;
  }
  for (int i = 0; i < n; i++) {
    a[i] = (float)(i + 1);
    b[i] = (float)(i + 1) * 0.5f;
    c[i] = 0.0f;
  }

  double t0 = now_us();
  Kernel kern = make_binop(op, n);
  int n_lin = 0;
  PolyUOp **lin = poly_linearize(kern.ctx, kern.sink, &n_lin);
  char *src = lin ? poly_render_c(lin, n_lin, fn_name) : NULL;
  PolyProgram *prog = src ? poly_compile_c(src, fn_name) : NULL;
  double compile_us = now_us() - t0;

  if (!prog) {
    print_row(name, iters, compile_us, 0.0, 0);
    free(src);
    free(lin);
    poly_ctx_destroy(kern.ctx);
    free(a); free(b); free(c);
    return 0;
  }

  double t1 = now_us();
  for (int it = 0; it < iters; it++) {
    void *args[3] = { a, b, c };
    poly_program_call(prog, args, 3);
  }
  double exec_us = (now_us() - t1) / (double)iters;

  for (int i = 0; i < n && ok; i++) {
    float expected = 0.0f;
    switch (op) {
      case POLY_OP_ADD: expected = a[i] + b[i]; break;
      case POLY_OP_MUL: expected = a[i] * b[i]; break;
      case POLY_OP_SUB: expected = a[i] - b[i]; break;
      default: break;
    }
    if (fabsf(c[i] - expected) > 1e-3f) ok = 0;
  }

  print_row(name, iters, compile_us, exec_us, ok);

  poly_program_destroy(prog);
  free(src);
  free(lin);
  poly_ctx_destroy(kern.ctx);
  free(a); free(b); free(c);
  return ok;
}

static int run_unop_case(const char *name, PolyOps op, int n, int iters) {
  int ok = 1;
  char fn_name[64];
  snprintf(fn_name, sizeof(fn_name), "k_%s", name);
  float *a = malloc((size_t)n * sizeof(float));
  float *c = malloc((size_t)n * sizeof(float));
  if (!a || !c) {
    free(a); free(c);
    printf("%-22s %8d %12s %12s %10s\n", name, iters, "OOM", "-", "FAIL");
    return 0;
  }
  for (int i = 0; i < n; i++) {
    if (op == POLY_OP_EXP2) a[i] = (float)((i % 9) - 4) * 0.5f;
    else a[i] = (float)(i + 1);
    c[i] = 0.0f;
  }

  double t0 = now_us();
  Kernel kern = make_unop(op, n);
  int n_lin = 0;
  PolyUOp **lin = poly_linearize(kern.ctx, kern.sink, &n_lin);
  char *src = lin ? poly_render_c(lin, n_lin, fn_name) : NULL;
  PolyProgram *prog = src ? poly_compile_c(src, fn_name) : NULL;
  double compile_us = now_us() - t0;

  if (!prog) {
    print_row(name, iters, compile_us, 0.0, 0);
    free(src);
    free(lin);
    poly_ctx_destroy(kern.ctx);
    free(a); free(c);
    return 0;
  }

  double t1 = now_us();
  for (int it = 0; it < iters; it++) {
    void *args[2] = { a, c };
    poly_program_call(prog, args, 2);
  }
  double exec_us = (now_us() - t1) / (double)iters;

  for (int i = 0; i < n && ok; i++) {
    float expected = 0.0f;
    switch (op) {
      case POLY_OP_NEG:  expected = -a[i]; break;
      case POLY_OP_SQRT: expected = sqrtf(a[i]); break;
      case POLY_OP_EXP2: expected = exp2f(a[i]); break;
      default: break;
    }
    if (fabsf(c[i] - expected) > 1e-3f) ok = 0;
  }

  print_row(name, iters, compile_us, exec_us, ok);

  poly_program_destroy(prog);
  free(src);
  free(lin);
  poly_ctx_destroy(kern.ctx);
  free(a); free(c);
  return ok;
}

static int run_reduce_sum_axis1_case(int n, int iters) {
  int cols = 256;
  int rows = n / cols;
  if (rows < 1) rows = 1;
  int total = rows * cols;

  int ok = 1;
  float *a_d = malloc((size_t)total * sizeof(float));
  float *out_d = malloc((size_t)rows * sizeof(float));
  if (!a_d || !out_d) {
    free(a_d); free(out_d);
    printf("%-22s %8d %12s %12s %10s\n", "reduce_sum_axis1", iters, "OOM", "-", "FAIL");
    return 0;
  }
  for (int i = 0; i < total; i++) a_d[i] = (float)(i + 1);
  for (int i = 0; i < rows; i++) out_d[i] = 0.0f;

  double t0 = now_us();
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, total);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, rows);
  int64_t shape[] = {rows, cols};
  int64_t axes[] = {1};
  PolyUOp *a2d = poly_reshape(ctx, a, shape, 2);
  PolyUOp *sum = poly_reduce_axis(ctx, POLY_OP_ADD, a2d, axes, 1);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out, sum, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());
  PolyProgram *prog = NULL;
  if (!compile_from_sink(ctx, sink, "k_reduce_sum_axis1", &prog)) prog = NULL;
  double compile_us = now_us() - t0;
  if (!prog) {
    print_row("reduce_sum_axis1", iters, compile_us, 0.0, 0);
    poly_ctx_destroy(ctx);
    free(a_d); free(out_d);
    return 0;
  }

  double t1 = now_us();
  for (int it = 0; it < iters; it++) {
    void *args[2] = { out_d, a_d };
    poly_program_call(prog, args, 2);
  }
  double exec_us = (now_us() - t1) / (double)iters;

  for (int r = 0; r < rows && ok; r++) {
    float expected = 0.0f;
    for (int c = 0; c < cols; c++) expected += a_d[r * cols + c];
    if (fabsf(out_d[r] - expected) > 1e-3f) ok = 0;
  }

  print_row("reduce_sum_axis1", iters, compile_us, exec_us, ok);
  poly_program_destroy(prog);
  poly_ctx_destroy(ctx);
  free(a_d); free(out_d);
  return ok;
}

static int run_chain_pad_flip_case(int n, int iters) {
  int out_n = n + 2;
  int ok = 1;
  float *a_d = malloc((size_t)n * sizeof(float));
  float *out_d = malloc((size_t)out_n * sizeof(float));
  if (!a_d || !out_d) {
    free(a_d); free(out_d);
    printf("%-22s %8d %12s %12s %10s\n", "chain_pad_flip", iters, "OOM", "-", "FAIL");
    return 0;
  }
  for (int i = 0; i < n; i++) a_d[i] = (float)(i + 1);
  for (int i = 0; i < out_n; i++) out_d[i] = -1.0f;

  double t0 = now_us();
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, n);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, out_n);
  int64_t pad_pairs[][2] = {{1, 1}};
  int64_t axes[] = {0};
  PolyUOp *p = poly_pad(ctx, a, pad_pairs, 1);
  PolyUOp *f = poly_flip(ctx, p, axes, 1);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out, f, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());
  PolyProgram *prog = NULL;
  if (!compile_from_sink(ctx, sink, "k_chain_pad_flip", &prog)) prog = NULL;
  double compile_us = now_us() - t0;
  if (!prog) {
    print_row("chain_pad_flip", iters, compile_us, 0.0, 0);
    poly_ctx_destroy(ctx);
    free(a_d); free(out_d);
    return 0;
  }

  double t1 = now_us();
  for (int it = 0; it < iters; it++) {
    void *args[2] = { out_d, a_d };
    poly_program_call(prog, args, 2);
  }
  double exec_us = (now_us() - t1) / (double)iters;

  for (int i = 0; i < out_n && ok; i++) {
    float expected = 0.0f;
    if (i > 0 && i < out_n - 1) expected = (float)(n - (i - 1));
    if (fabsf(out_d[i] - expected) > 1e-3f) ok = 0;
  }

  print_row("chain_pad_flip", iters, compile_us, exec_us, ok);
  poly_program_destroy(prog);
  poly_ctx_destroy(ctx);
  free(a_d); free(out_d);
  return ok;
}

static int run_grad_mul_sum_case(int n, int iters) {
  int ok = 1;
  float *x_d = malloc((size_t)n * sizeof(float));
  float *gx_d = malloc((size_t)n * sizeof(float));
  if (!x_d || !gx_d) {
    free(x_d); free(gx_d);
    printf("%-22s %8d %12s %12s %10s\n", "grad_mul_sum", iters, "OOM", "-", "FAIL");
    return 0;
  }
  for (int i = 0; i < n; i++) {
    x_d[i] = (float)((i % 101) - 50);
    gx_d[i] = 0.0f;
  }

  double t0 = now_us();
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_buffer(ctx, POLY_FLOAT32, n);
  PolyUOp *xx = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, x, x, poly_arg_none());
  int64_t axes[] = {0};
  PolyUOp *loss = poly_reduce_axis(ctx, POLY_OP_ADD, xx, axes, 1);
  PolyUOp *gx = poly_grad(ctx, loss, x);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, n);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out, gx, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());
  PolyProgram *prog = NULL;
  if (!compile_from_sink(ctx, sink, "k_grad_mul_sum", &prog)) prog = NULL;
  double compile_us = now_us() - t0;
  if (!prog) {
    print_row("grad_mul_sum", iters, compile_us, 0.0, 0);
    poly_ctx_destroy(ctx);
    free(x_d); free(gx_d);
    return 0;
  }

  double t1 = now_us();
  for (int it = 0; it < iters; it++) {
    void *args[2] = { gx_d, x_d };
    poly_program_call(prog, args, 2);
  }
  double exec_us = (now_us() - t1) / (double)iters;

  for (int i = 0; i < n && ok; i++) {
    float expected = 2.0f * x_d[i];
    if (fabsf(gx_d[i] - expected) > 1e-3f) ok = 0;
  }

  print_row("grad_mul_sum", iters, compile_us, exec_us, ok);
  poly_program_destroy(prog);
  poly_ctx_destroy(ctx);
  free(x_d); free(gx_d);
  return ok;
}

static int run_grad_exp2_sum_case(int n, int iters) {
  int ok = 1;
  float *x_d = malloc((size_t)n * sizeof(float));
  float *gx_d = malloc((size_t)n * sizeof(float));
  if (!x_d || !gx_d) {
    free(x_d); free(gx_d);
    printf("%-22s %8d %12s %12s %10s\n", "grad_exp2_sum", iters, "OOM", "-", "FAIL");
    return 0;
  }
  for (int i = 0; i < n; i++) {
    x_d[i] = (float)((i % 9) - 4) * 0.5f;
    gx_d[i] = 0.0f;
  }

  double t0 = now_us();
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_buffer(ctx, POLY_FLOAT32, n);
  PolyUOp *e = poly_uop1(ctx, POLY_OP_EXP2, POLY_FLOAT32, x, poly_arg_none());
  int64_t axes[] = {0};
  PolyUOp *loss = poly_reduce_axis(ctx, POLY_OP_ADD, e, axes, 1);
  PolyUOp *gx = poly_grad(ctx, loss, x);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, n);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out, gx, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());
  PolyProgram *prog = NULL;
  if (!compile_from_sink(ctx, sink, "k_grad_exp2_sum", &prog)) prog = NULL;
  double compile_us = now_us() - t0;
  if (!prog) {
    print_row("grad_exp2_sum", iters, compile_us, 0.0, 0);
    poly_ctx_destroy(ctx);
    free(x_d); free(gx_d);
    return 0;
  }

  double t1 = now_us();
  for (int it = 0; it < iters; it++) {
    void *args[2] = { gx_d, x_d };
    poly_program_call(prog, args, 2);
  }
  double exec_us = (now_us() - t1) / (double)iters;

  for (int i = 0; i < n && ok; i++) {
    float expected = exp2f(x_d[i]) * LN2_F;
    if (fabsf(gx_d[i] - expected) > 2e-3f) ok = 0;
  }

  print_row("grad_exp2_sum", iters, compile_us, exec_us, ok);
  poly_program_destroy(prog);
  poly_ctx_destroy(ctx);
  free(x_d); free(gx_d);
  return ok;
}

static int run_grad_fdiv_sum_y_case(int n, int iters) {
  int ok = 1;
  float *x_d = malloc((size_t)n * sizeof(float));
  float *y_d = malloc((size_t)n * sizeof(float));
  float *gy_d = malloc((size_t)n * sizeof(float));
  if (!x_d || !y_d || !gy_d) {
    free(x_d); free(y_d); free(gy_d);
    printf("%-22s %8d %12s %12s %10s\n", "grad_fdiv_sum_y", iters, "OOM", "-", "FAIL");
    return 0;
  }
  for (int i = 0; i < n; i++) {
    int yi = (i % 11) - 5;
    if (yi == 0) yi = 1;
    x_d[i] = (float)((i % 17) - 8);
    y_d[i] = (float)yi * 0.5f;
    gy_d[i] = 0.0f;
  }

  double t0 = now_us();
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_buffer(ctx, POLY_FLOAT32, n);
  PolyUOp *y = poly_buffer(ctx, POLY_FLOAT32, n);
  PolyUOp *q = poly_uop2(ctx, POLY_OP_FDIV, POLY_FLOAT32, x, y, poly_arg_none());
  int64_t axes[] = {0};
  PolyUOp *loss = poly_reduce_axis(ctx, POLY_OP_ADD, q, axes, 1);
  PolyUOp *gy = poly_grad(ctx, loss, y);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, n);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out, gy, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());
  PolyProgram *prog = NULL;
  if (!compile_from_sink(ctx, sink, "k_grad_fdiv_sum_y", &prog)) prog = NULL;
  double compile_us = now_us() - t0;
  if (!prog) {
    print_row("grad_fdiv_sum_y", iters, compile_us, 0.0, 0);
    poly_ctx_destroy(ctx);
    free(x_d); free(y_d); free(gy_d);
    return 0;
  }

  double t1 = now_us();
  for (int it = 0; it < iters; it++) {
    void *args[3] = { gy_d, x_d, y_d };
    poly_program_call(prog, args, 3);
  }
  double exec_us = (now_us() - t1) / (double)iters;

  for (int i = 0; i < n && ok; i++) {
    float expected = -x_d[i] / (y_d[i] * y_d[i]);
    if (fabsf(gy_d[i] - expected) > 2e-3f) ok = 0;
  }

  print_row("grad_fdiv_sum_y", iters, compile_us, exec_us, ok);
  poly_program_destroy(prog);
  poly_ctx_destroy(ctx);
  free(x_d); free(y_d); free(gy_d);
  return ok;
}

static int run_grad_chain_movement_case(int n, int iters) {
  int cols = 3;
  int rows = n / cols;
  if (rows < 1) rows = 1;
  int total = rows * cols;

  int ok = 1;
  float *x_d = malloc((size_t)total * sizeof(float));
  float *gx_d = malloc((size_t)total * sizeof(float));
  if (!x_d || !gx_d) {
    free(x_d); free(gx_d);
    printf("%-22s %8d %12s %12s %10s\n", "grad_chain_movement", iters, "OOM", "-", "FAIL");
    return 0;
  }
  for (int i = 0; i < total; i++) {
    x_d[i] = (float)(i + 1);
    gx_d[i] = 0.0f;
  }

  double t0 = now_us();
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_buffer(ctx, POLY_FLOAT32, total);
  int64_t shape[] = {rows, cols};
  int64_t perm[] = {1, 0};
  int64_t axes[] = {0, 1};
  PolyUOp *xr = poly_reshape(ctx, x, shape, 2);
  PolyUOp *xp = poly_permute(ctx, xr, perm, 2);
  PolyUOp *loss = poly_reduce_axis(ctx, POLY_OP_ADD, xp, axes, 2);
  PolyUOp *gx = poly_grad(ctx, loss, x);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, total);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out, gx, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());
  PolyProgram *prog = NULL;
  if (!compile_from_sink(ctx, sink, "k_grad_chain_move", &prog)) prog = NULL;
  double compile_us = now_us() - t0;
  if (!prog) {
    print_row("grad_chain_movement", iters, compile_us, 0.0, 0);
    poly_ctx_destroy(ctx);
    free(x_d); free(gx_d);
    return 0;
  }

  double t1 = now_us();
  for (int it = 0; it < iters; it++) {
    void *args[2] = { gx_d, x_d };
    poly_program_call(prog, args, 2);
  }
  double exec_us = (now_us() - t1) / (double)iters;

  for (int i = 0; i < total && ok; i++) {
    if (fabsf(gx_d[i] - 1.0f) > 1e-3f) ok = 0;
  }

  print_row("grad_chain_movement", iters, compile_us, exec_us, ok);
  poly_program_destroy(prog);
  poly_ctx_destroy(ctx);
  free(x_d); free(gx_d);
  return ok;
}

int main(int argc, char **argv) {
  int n = (argc > 1) ? atoi(argv[1]) : 1024;
  int iters_elementwise = (argc > 2) ? atoi(argv[2]) : 100;
  int iters_graph = (argc > 3) ? atoi(argv[3]) : 20;

  if (n < 1) n = 1;
  if (iters_elementwise < 1) iters_elementwise = 1;
  if (iters_graph < 1) iters_graph = 1;

  printf("polygrad benchmark  N=%d  iters_elementwise=%d  iters_graph=%d\n",
         n, iters_elementwise, iters_graph);
  printf("%-22s %8s %12s %12s %10s\n",
         "case", "iters", "compile_us", "exec_us", "correct");
  printf("%-22s %8s %12s %12s %10s\n",
         "----------------------", "--------", "------------", "------------", "----------");

  printf("\n[forward_elementwise]\n");
  run_binop_case("add", POLY_OP_ADD, n, iters_elementwise);
  run_binop_case("mul", POLY_OP_MUL, n, iters_elementwise);
  run_binop_case("sub", POLY_OP_SUB, n, iters_elementwise);
  run_unop_case("neg", POLY_OP_NEG, n, iters_elementwise);
  run_unop_case("sqrt", POLY_OP_SQRT, n, iters_elementwise);
  run_unop_case("exp2", POLY_OP_EXP2, n, iters_elementwise);

  printf("\n[graph_and_autograd]\n");
  run_reduce_sum_axis1_case(n, iters_graph);
  run_chain_pad_flip_case(n, iters_graph);
  run_grad_mul_sum_case(n, iters_graph);
  run_grad_exp2_sum_case(n, iters_graph);
  run_grad_fdiv_sum_y_case(n, iters_graph);
  run_grad_chain_movement_case(n, iters_graph);

  return 0;
}
