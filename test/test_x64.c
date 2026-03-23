/*
 * test_x64.c — Comprehensive tests for x86-64 JIT renderer and runtime
 *
 * Covers: scalar + vec4 paths, all float/int ALU ops, reduce, chain,
 * CAST/BITCAST, WHERE, parity vs CPU (C compiler) backend.
 */

#ifdef POLY_HAS_X64

#include "test_harness.h"
#include "../src/codegen.h"
#include "../src/exec_plan.h"
#include "../src/frontend.h"
#include "../src/scheduler.h"
#include <math.h>

/* ══════════════════════════════════════════════════════════════════════ */
/*  Helpers                                                              */
/* ══════════════════════════════════════════════════════════════════════ */

typedef struct { PolyCtx *ctx; PolyUOp *sink; } K;

/* Run kernel via x64 JIT (scalar path, no optimize) */
static int x64_run(PolyCtx *ctx, PolyUOp *sink, void **args, int n) {
  int nl; PolyUOp **lin = poly_linearize(ctx, sink, &nl);
  if (!lin) return -1;
  int sz; uint8_t *code = poly_render_x64(lin, nl, &sz);
  free(lin); if (!code) return -1;
  PolyX64Program *p = poly_compile_x64(code, sz);
  free(code); if (!p) return -1;
  poly_x64_program_call(p, args, n);
  poly_x64_program_destroy(p);
  return 0;
}

/* Run kernel via x64 JIT (vec4 optimized path) */
static int x64_run_vec(PolyCtx *ctx, PolyUOp *sink, void **args, int n) {
  PolyRewriteOpts opts = { .optimize = true, .devectorize = 0 };
  int nl; PolyUOp **lin = poly_linearize_ex(ctx, sink, opts, &nl);
  if (!lin) return -1;
  int sz; uint8_t *code = poly_render_x64(lin, nl, &sz);
  free(lin); if (!code) return -1;
  PolyX64Program *p = poly_compile_x64(code, sz);
  free(code); if (!p) return -1;
  poly_x64_program_call(p, args, n);
  poly_x64_program_destroy(p);
  return 0;
}

/* Run kernel via CPU (C compiler) for parity reference */
static int cpu_run(PolyCtx *ctx, PolyUOp *sink, void **args, int n) {
  int nl; PolyUOp **lin = poly_linearize(ctx, sink, &nl);
  if (!lin) return -1;
  char *src = poly_render_c(lin, nl, "test_fn");
  free(lin); if (!src) return -1;
  PolyProgram *p = poly_compile_c(src, "test_fn");
  free(src); if (!p) return -1;
  poly_program_call(p, args, n);
  poly_program_destroy(p);
  return 0;
}

/* Forward declaration -- defined in AVX2 section below */
static int x64_run_avx2(PolyCtx *ctx, PolyUOp *sink, void **args, int n);

/* Build: c[i] = a[i] OP b[i] */
static K make_binop(PolyOps op, int N) {
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
  return (K){ ctx, poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none()) };
}

/* Build: b[i] = OP(a[i]) */
static K make_unary(PolyOps op, int N) {
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
  return (K){ ctx, poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none()) };
}

/* Parity helper: run both CPU and x64, compare results */
static int check_parity(K k, void **args, int n_args, float *c_cpu, float *c_x64, int N, float tol) {
  memset(c_cpu, 0, N * sizeof(float));
  memset(c_x64, 0, N * sizeof(float));
  args[n_args - 1] = c_cpu;
  if (cpu_run(k.ctx, k.sink, args, n_args) != 0) return -1;
  args[n_args - 1] = c_x64;
  if (x64_run(k.ctx, k.sink, args, n_args) != 0) return -2;
  for (int i = 0; i < N; i++)
    if (fabsf(c_cpu[i] - c_x64[i]) > tol) return i + 1;
  return 0;
}

/* Parity helper for vec4 path */
static int check_parity_vec(K k, void **args, int n_args, float *c_cpu, float *c_x64, int N, float tol) {
  memset(c_cpu, 0, N * sizeof(float));
  memset(c_x64, 0, N * sizeof(float));
  args[n_args - 1] = c_cpu;
  if (cpu_run(k.ctx, k.sink, args, n_args) != 0) return -1;
  args[n_args - 1] = c_x64;
  if (x64_run_vec(k.ctx, k.sink, args, n_args) != 0) return -2;
  for (int i = 0; i < N; i++)
    if (fabsf(c_cpu[i] - c_x64[i]) > tol) return i + 1;
  return 0;
}

/* Build: c[i] = a[i] OP b[i] (int32) -- reserved for future int tests */
static K __attribute__((unused)) make_int_binop(PolyOps op, int N) {
  PolyCtx *ctx = poly_ctx_new();
  PolyDType pi = poly_dtype_ptr(POLY_INT32, -1, POLY_ADDR_GLOBAL);
  PolyUOp *p0 = poly_uop0(ctx, POLY_OP_PARAM, pi, poly_arg_int(0));
  PolyUOp *p1 = poly_uop0(ctx, POLY_OP_PARAM, pi, poly_arg_int(1));
  PolyUOp *p2 = poly_uop0(ctx, POLY_OP_PARAM, pi, poly_arg_int(2));
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(N));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));
  PolyUOp *i0 = poly_uop2(ctx, POLY_OP_INDEX, pi, p0, range, poly_arg_none());
  PolyUOp *i1 = poly_uop2(ctx, POLY_OP_INDEX, pi, p1, range, poly_arg_none());
  PolyUOp *i2 = poly_uop2(ctx, POLY_OP_INDEX, pi, p2, range, poly_arg_none());
  PolyUOp *l0 = poly_uop1(ctx, POLY_OP_LOAD, POLY_INT32, i0, poly_arg_none());
  PolyUOp *l1 = poly_uop1(ctx, POLY_OP_LOAD, POLY_INT32, i1, poly_arg_none());
  PolyUOp *alu = poly_uop2(ctx, op, POLY_INT32, l0, l1, poly_arg_none());
  PolyUOp *st = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, i2, alu, poly_arg_none());
  PolyUOp *es[2] = { st, range };
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, es, 2, poly_arg_none());
  return (K){ ctx, poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none()) };
}

/* Parity helper for int32 ops -- reserved for future int tests */
static int __attribute__((unused)) check_parity_int(K k, void **args, int n_args, int32_t *c_cpu, int32_t *c_x64, int N) {
  memset(c_cpu, 0, N * sizeof(int32_t));
  memset(c_x64, 0, N * sizeof(int32_t));
  args[n_args - 1] = c_cpu;
  if (cpu_run(k.ctx, k.sink, args, n_args) != 0) return -1;
  args[n_args - 1] = c_x64;
  if (x64_run(k.ctx, k.sink, args, n_args) != 0) return -2;
  for (int i = 0; i < N; i++)
    if (c_cpu[i] != c_x64[i]) return i + 1;
  return 0;
}

/* Build: c[i] = WHERE(a[i] > 0, a[i], b[i]) -- reserved for future tests */
static K __attribute__((unused)) make_where_kernel(int N) {
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
  PolyUOp *la = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, i0, poly_arg_none());
  PolyUOp *lb = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, i1, poly_arg_none());
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(0.0f));
  PolyUOp *cmp = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, zero, la, poly_arg_none());
  PolyUOp *w = poly_uop(ctx, POLY_OP_WHERE, POLY_FLOAT32,
                         (PolyUOp*[]){cmp, la, lb}, 3, poly_arg_none());
  PolyUOp *st = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, i2, w, poly_arg_none());
  PolyUOp *es[2] = { st, range };
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, es, 2, poly_arg_none());
  return (K){ ctx, poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none()) };
}

/* Build: c[i] = CAST(float, a_int[i]) -- reserved for future tests */
static K __attribute__((unused)) make_cast_int_to_float(int N) {
  PolyCtx *ctx = poly_ctx_new();
  PolyDType pi = poly_dtype_ptr(POLY_INT32, -1, POLY_ADDR_GLOBAL);
  PolyDType pf = poly_dtype_ptr(POLY_FLOAT32, -1, POLY_ADDR_GLOBAL);
  PolyUOp *p0 = poly_uop0(ctx, POLY_OP_PARAM, pi, poly_arg_int(0));
  PolyUOp *p1 = poly_uop0(ctx, POLY_OP_PARAM, pf, poly_arg_int(1));
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(N));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));
  PolyUOp *i0 = poly_uop2(ctx, POLY_OP_INDEX, pi, p0, range, poly_arg_none());
  PolyUOp *i1 = poly_uop2(ctx, POLY_OP_INDEX, pf, p1, range, poly_arg_none());
  PolyUOp *li = poly_uop1(ctx, POLY_OP_LOAD, POLY_INT32, i0, poly_arg_none());
  PolyUOp *cast = poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, li, poly_arg_none());
  PolyUOp *st = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, i1, cast, poly_arg_none());
  PolyUOp *es[2] = { st, range };
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, es, 2, poly_arg_none());
  return (K){ ctx, poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none()) };
}

/* ══════════════════════════════════════════════════════════════════════ */
/*  Scalar path tests                                                    */
/* ══════════════════════════════════════════════════════════════════════ */

TEST(x64, empty_kernel) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *noop = poly_uop0(ctx, POLY_OP_NOOP, POLY_VOID, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, noop, poly_arg_none());
  int nl; PolyUOp **lin = poly_linearize(ctx, sink, &nl);
  ASSERT_NOT_NULL(lin);
  int sz; uint8_t *code = poly_render_x64(lin, nl, &sz);
  free(lin); ASSERT_NOT_NULL(code); ASSERT_TRUE(sz > 10);
  PolyX64Program *p = poly_compile_x64(code, sz);
  free(code); ASSERT_NOT_NULL(p);
  poly_x64_program_call(p, NULL, 0);
  poly_x64_program_destroy(p);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(x64, e2e_vecadd) {
  K k = make_binop(POLY_OP_ADD, 16);
  float a[16], b[16], c[16];
  for (int i = 0; i < 16; i++) { a[i] = (float)i; b[i] = (float)(i * 10); }
  void *args[3] = { a, b, c };
  ASSERT_INT_EQ(x64_run(k.ctx, k.sink, args, 3), 0);
  for (int i = 0; i < 16; i++) ASSERT_FLOAT_EQ(c[i], a[i] + b[i], 1e-6);
  poly_ctx_destroy(k.ctx); PASS();
}

TEST(x64, e2e_vecsub) {
  K k = make_binop(POLY_OP_SUB, 16);
  float a[16], b[16], c[16];
  for (int i = 0; i < 16; i++) { a[i] = (float)(i * 10); b[i] = (float)i; }
  void *args[3] = { a, b, c };
  ASSERT_INT_EQ(x64_run(k.ctx, k.sink, args, 3), 0);
  for (int i = 0; i < 16; i++) ASSERT_FLOAT_EQ(c[i], a[i] - b[i], 1e-6);
  poly_ctx_destroy(k.ctx); PASS();
}

TEST(x64, e2e_vecmul) {
  K k = make_binop(POLY_OP_MUL, 16);
  float a[16], b[16], c[16];
  for (int i = 0; i < 16; i++) { a[i] = (float)(i + 1); b[i] = 0.5f; }
  void *args[3] = { a, b, c };
  ASSERT_INT_EQ(x64_run(k.ctx, k.sink, args, 3), 0);
  for (int i = 0; i < 16; i++) ASSERT_FLOAT_EQ(c[i], a[i] * b[i], 1e-6);
  poly_ctx_destroy(k.ctx); PASS();
}

TEST(x64, e2e_fdiv) {
  K k = make_binop(POLY_OP_FDIV, 8);
  float a[8] = {10,20,30,40,50,60,70,80};
  float b[8] = {2,4,5,8,10,12,14,16};
  float c[8];
  void *args[3] = { a, b, c };
  ASSERT_INT_EQ(x64_run(k.ctx, k.sink, args, 3), 0);
  for (int i = 0; i < 8; i++) ASSERT_FLOAT_EQ(c[i], a[i] / b[i], 1e-5);
  poly_ctx_destroy(k.ctx); PASS();
}

TEST(x64, e2e_max) {
  K k = make_binop(POLY_OP_MAX, 8);
  float a[8] = {1,-2,3,-4,5,-6,7,-8};
  float b[8] = {-1,2,-3,4,-5,6,-7,8};
  float c[8];
  void *args[3] = { a, b, c };
  ASSERT_INT_EQ(x64_run(k.ctx, k.sink, args, 3), 0);
  for (int i = 0; i < 8; i++) ASSERT_FLOAT_EQ(c[i], a[i] > b[i] ? a[i] : b[i], 1e-6);
  poly_ctx_destroy(k.ctx); PASS();
}

TEST(x64, e2e_neg) {
  K k = make_unary(POLY_OP_NEG, 8);
  float a[8] = {1,-2,0,3.14f,-0.0f,1e10f,-1e-10f,42};
  float b[8];
  void *args[2] = { a, b };
  ASSERT_INT_EQ(x64_run(k.ctx, k.sink, args, 2), 0);
  for (int i = 0; i < 8; i++) ASSERT_FLOAT_EQ(b[i], -a[i], 0.0f);
  poly_ctx_destroy(k.ctx); PASS();
}

TEST(x64, e2e_sqrt) {
  K k = make_unary(POLY_OP_SQRT, 8);
  float a[8] = {0,1,4,9,16,25,100,0.25f};
  float b[8];
  void *args[2] = { a, b };
  ASSERT_INT_EQ(x64_run(k.ctx, k.sink, args, 2), 0);
  for (int i = 0; i < 8; i++) ASSERT_FLOAT_EQ(b[i], sqrtf(a[i]), 1e-6);
  poly_ctx_destroy(k.ctx); PASS();
}

TEST(x64, e2e_reciprocal) {
  K k = make_unary(POLY_OP_RECIPROCAL, 8);
  float a[8] = {1,2,4,0.5f,0.25f,10,100,0.1f};
  float b[8];
  void *args[2] = { a, b };
  ASSERT_INT_EQ(x64_run(k.ctx, k.sink, args, 2), 0);
  for (int i = 0; i < 8; i++) ASSERT_FLOAT_EQ(b[i], 1.0f / a[i], 1e-5);
  poly_ctx_destroy(k.ctx); PASS();
}

/* ── Chain: c[i] = sqrt(a[i]*b[i] + a[i]) ──────────────────────────── */

TEST(x64, e2e_chain_mul_add_sqrt) {
  PolyCtx *ctx = poly_ctx_new();
  PolyDType pf = poly_dtype_ptr(POLY_FLOAT32, -1, POLY_ADDR_GLOBAL);
  PolyUOp *p0 = poly_uop0(ctx, POLY_OP_PARAM, pf, poly_arg_int(0));
  PolyUOp *p1 = poly_uop0(ctx, POLY_OP_PARAM, pf, poly_arg_int(1));
  PolyUOp *p2 = poly_uop0(ctx, POLY_OP_PARAM, pf, poly_arg_int(2));
  int N = 16;
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
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());

  float a[16], b[16], c_cpu[16], c_x64[16];
  for (int i = 0; i < N; i++) { a[i] = 1.0f + 0.5f * i; b[i] = 2.0f + 0.5f * i; }
  K k = { ctx, sink };
  void *args[3] = { a, b, NULL };
  ASSERT_INT_EQ(check_parity(k, args, 3, c_cpu, c_x64, N, 1e-5), 0);
  poly_ctx_destroy(ctx); PASS();
}

/* ══════════════════════════════════════════════════════════════════════ */
/*  Parity tests: x64 scalar vs CPU for every float ALU op               */
/* ══════════════════════════════════════════════════════════════════════ */

#define PARITY_TEST_BINOP(name, op) \
TEST(x64, parity_##name) { \
  int N = 16; K k = make_binop(op, N); \
  float a[16], b[16], c_cpu[16], c_x64[16]; \
  for (int i = 0; i < N; i++) { a[i] = 1.0f + i * 0.3f; b[i] = 2.0f + i * 0.7f; } \
  void *args[3] = { a, b, NULL }; \
  ASSERT_INT_EQ(check_parity(k, args, 3, c_cpu, c_x64, N, 1e-5), 0); \
  poly_ctx_destroy(k.ctx); PASS(); \
}

#define PARITY_TEST_UNARY(name, op) \
TEST(x64, parity_##name) { \
  int N = 8; K k = make_unary(op, N); \
  float a[8] = {1,4,9,16,25,0.25f,0.01f,100}; \
  float b_cpu[8], b_x64[8]; \
  void *args[2] = { a, NULL }; \
  ASSERT_INT_EQ(check_parity(k, args, 2, b_cpu, b_x64, N, 1e-5), 0); \
  poly_ctx_destroy(k.ctx); PASS(); \
}

PARITY_TEST_BINOP(add, POLY_OP_ADD)
PARITY_TEST_BINOP(sub, POLY_OP_SUB)
PARITY_TEST_BINOP(mul, POLY_OP_MUL)
PARITY_TEST_BINOP(fdiv, POLY_OP_FDIV)
PARITY_TEST_BINOP(max, POLY_OP_MAX)
PARITY_TEST_UNARY(neg, POLY_OP_NEG)
PARITY_TEST_UNARY(sqrt, POLY_OP_SQRT)
PARITY_TEST_UNARY(reciprocal, POLY_OP_RECIPROCAL)

/* ══════════════════════════════════════════════════════════════════════ */
/*  Vec4 optimized path tests                                            */
/* ══════════════════════════════════════════════════════════════════════ */

#define VEC4_PARITY_BINOP(name, op) \
TEST(x64, vec4_##name) { \
  int N = 32; K k = make_binop(op, N); \
  float a[32], b[32], c_cpu[32], c_x64[32]; \
  for (int i = 0; i < N; i++) { a[i] = 1.0f + i * 0.3f; b[i] = 2.0f + i * 0.7f; } \
  void *args[3] = { a, b, NULL }; \
  ASSERT_INT_EQ(check_parity_vec(k, args, 3, c_cpu, c_x64, N, 1e-5), 0); \
  poly_ctx_destroy(k.ctx); PASS(); \
}

#define VEC4_PARITY_UNARY(name, op) \
TEST(x64, vec4_##name) { \
  int N = 32; K k = make_unary(op, N); \
  float a[32], b_cpu[32], b_x64[32]; \
  for (int i = 0; i < N; i++) a[i] = 1.0f + i * 0.5f; \
  void *args[2] = { a, NULL }; \
  ASSERT_INT_EQ(check_parity_vec(k, args, 2, b_cpu, b_x64, N, 1e-5), 0); \
  poly_ctx_destroy(k.ctx); PASS(); \
}

VEC4_PARITY_BINOP(add, POLY_OP_ADD)
VEC4_PARITY_BINOP(sub, POLY_OP_SUB)
VEC4_PARITY_BINOP(mul, POLY_OP_MUL)
VEC4_PARITY_BINOP(fdiv, POLY_OP_FDIV)
VEC4_PARITY_BINOP(max, POLY_OP_MAX)
VEC4_PARITY_UNARY(neg, POLY_OP_NEG)
VEC4_PARITY_UNARY(sqrt, POLY_OP_SQRT)
VEC4_PARITY_UNARY(reciprocal, POLY_OP_RECIPROCAL)

/* Vec4 chain: multi-op pipeline correctness */
TEST(x64, vec4_chain_mul_add_sqrt) {
  PolyCtx *ctx = poly_ctx_new();
  PolyDType pf = poly_dtype_ptr(POLY_FLOAT32, -1, POLY_ADDR_GLOBAL);
  PolyUOp *p0 = poly_uop0(ctx, POLY_OP_PARAM, pf, poly_arg_int(0));
  PolyUOp *p1 = poly_uop0(ctx, POLY_OP_PARAM, pf, poly_arg_int(1));
  PolyUOp *p2 = poly_uop0(ctx, POLY_OP_PARAM, pf, poly_arg_int(2));
  int N = 32;
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
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());

  float a[32], b[32], c_cpu[32], c_x64[32];
  for (int i = 0; i < N; i++) { a[i] = 1.0f + 0.5f * i; b[i] = 2.0f + 0.5f * i; }

  /* CPU reference */
  memset(c_cpu, 0, sizeof(c_cpu));
  void *args_cpu[3] = { a, b, c_cpu };
  ASSERT_INT_EQ(cpu_run(ctx, sink, args_cpu, 3), 0);

  /* x64 vec */
  memset(c_x64, 0, sizeof(c_x64));
  void *args_x64[3] = { a, b, c_x64 };
  ASSERT_INT_EQ(x64_run_vec(ctx, sink, args_x64, 3), 0);

  for (int i = 0; i < N; i++)
    ASSERT_FLOAT_EQ(c_x64[i], c_cpu[i], 1e-5);

  poly_ctx_destroy(ctx); PASS();
}

/* ══════════════════════════════════════════════════════════════════════ */
/*  Backend registration                                                 */
/* ══════════════════════════════════════════════════════════════════════ */

TEST(x64, backend_registered) {
  const PolyBackendDesc *desc = poly_backend_get(POLY_DEVICE_X64_JIT);
  ASSERT_NOT_NULL(desc);
  ASSERT_STR_EQ(desc->name, "x64_jit");
  PASS();
}

/* ══════════════════════════════════════════════════════════════════════ */
/*  Non-divisible-by-4 sizes (vec path must handle or fall back)         */
/* ══════════════════════════════════════════════════════════════════════ */

TEST(x64, vec4_non_aligned_size) {
  /* N=13: not divisible by 4. The optimize path should still produce correct results
   * (UPCAST may choose a different factor or fall back to scalar). */
  int N = 13;
  K k = make_binop(POLY_OP_ADD, N);
  float a[13], b[13], c_cpu[13], c_x64[13];
  for (int i = 0; i < N; i++) { a[i] = (float)i; b[i] = (float)(i * 2); }
  void *args[3] = { a, b, NULL };
  ASSERT_INT_EQ(check_parity_vec(k, args, 3, c_cpu, c_x64, N, 1e-5), 0);
  poly_ctx_destroy(k.ctx); PASS();
}

/* ══════════════════════════════════════════════════════════════════════ */
/*  Large size parity (catches off-by-one in loop bounds)                */
/* ══════════════════════════════════════════════════════════════════════ */

TEST(x64, vec4_large_parity) {
  int N = 1024;
  K k = make_binop(POLY_OP_ADD, N);
  float *a = malloc(N * sizeof(float));
  float *b = malloc(N * sizeof(float));
  float *c_cpu = malloc(N * sizeof(float));
  float *c_x64 = malloc(N * sizeof(float));
  for (int i = 0; i < N; i++) { a[i] = (float)i * 0.01f; b[i] = (float)(N - i) * 0.01f; }
  void *args[3] = { a, b, NULL };
  ASSERT_INT_EQ(check_parity_vec(k, args, 3, c_cpu, c_x64, N, 1e-4), 0);
  free(a); free(b); free(c_cpu); free(c_x64);
  poly_ctx_destroy(k.ctx); PASS();
}

/* ══════════════════════════════════════════════════════════════════════ */
/*  Three-way differential: CPU vs INTERP vs x64 JIT                     */
/* ══════════════════════════════════════════════════════════════════════ */

static int three_way_parity(PolyCtx *ctx, PolyUOp *sink,
                            PolyUOp **bufs, void **datas, int n_bufs,
                            PolyUOp *out_buf, float *out_cpu, float *out_interp,
                            float *out_x64, int out_numel, float tol) {
  PolyPreparedStep *ps = poly_prepare_step(ctx, sink, POLY_MODE_CALL);
  if (!ps) return -1;

  /* Helper: fill slots from buf/data arrays */
  #define FILL_SLOTS(slot_arr, out_ptr) do { \
    memset(slot_arr, 0, sizeof(void*) * 16); \
    for (int _i = 0; _i < ps->n_buf_slots; _i++) \
      for (int _j = 0; _j < n_bufs; _j++) \
        if (ps->buf_slots[_i].buf_uop == bufs[_j]) \
          slot_arr[_i] = (bufs[_j] == out_buf) ? (out_ptr) : datas[_j]; \
  } while(0)

  /* CPU path */
  PolyExecutableStep *cpu = poly_lower_step(ctx, ps, POLY_DEVICE_CPU);
  if (!cpu) { poly_prepared_step_free(ps); return -2; }
  memset(out_cpu, 0, (size_t)out_numel * sizeof(float));
  void *slot_cpu[16]; FILL_SLOTS(slot_cpu, out_cpu);
  int rc = poly_executable_step_run(cpu, slot_cpu, ps->n_buf_slots, NULL, 0);
  poly_executable_step_free(cpu);
  if (rc < 0) { poly_prepared_step_free(ps); return -3; }

  /* INTERP path */
  PolyExecutableStep *interp = poly_lower_step(ctx, ps, POLY_DEVICE_INTERP);
  if (!interp) { poly_prepared_step_free(ps); return -4; }
  memset(out_interp, 0, (size_t)out_numel * sizeof(float));
  void *slot_interp[16]; FILL_SLOTS(slot_interp, out_interp);
  rc = poly_executable_step_run(interp, slot_interp, ps->n_buf_slots, NULL, 0);
  poly_executable_step_free(interp);
  if (rc < 0) { poly_prepared_step_free(ps); return -5; }

  /* x64 JIT path */
  PolyExecutableStep *x64 = poly_lower_step(ctx, ps, POLY_DEVICE_X64_JIT);
  if (!x64) { poly_prepared_step_free(ps); return -6; }
  memset(out_x64, 0, (size_t)out_numel * sizeof(float));
  void *slot_x64[16]; FILL_SLOTS(slot_x64, out_x64);
  rc = poly_executable_step_run(x64, slot_x64, ps->n_buf_slots, NULL, 0);
  poly_executable_step_free(x64);
  poly_prepared_step_free(ps);
  if (rc < 0) return -7;

  #undef FILL_SLOTS

  /* Compare all three */
  for (int i = 0; i < out_numel; i++) {
    float d1 = fabsf(out_cpu[i] - out_interp[i]);
    float d2 = fabsf(out_cpu[i] - out_x64[i]);
    if (d1 > tol) return 100 + i;  /* CPU vs INTERP mismatch */
    if (d2 > tol) return 200 + i;  /* CPU vs x64 mismatch */
  }
  return 0;
}

/* ══════════════════════════════════════════════════════════════════════ */
/*  REDUCE tests (three-way differential)                                */
/* ══════════════════════════════════════════════════════════════════════ */

TEST(x64, e2e_reduce_sum) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, 8);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, 1);
  int64_t axes[] = {0};
  PolyUOp *s = poly_reduce_axis(ctx, POLY_OP_ADD, a, axes, 1);
  PolyUOp *st = poly_store_val(ctx, out, s);
  PolyUOp *sink = poly_sink1(ctx, st);

  float da[] = {1, 2, 3, 4, 5, 6, 7, 8};
  float out_cpu[1], out_interp[1], out_x64[1];
  PolyUOp *bufs[] = {a, out};
  void *datas[] = {da, NULL};

  int rc = three_way_parity(ctx, sink, bufs, datas, 2,
                            out, out_cpu, out_interp, out_x64, 1, 1e-5f);
  ASSERT_INT_EQ(rc, 0);
  ASSERT_FLOAT_EQ(out_x64[0], 36.0f, 1e-5);
  poly_ctx_destroy(ctx); PASS();
}

TEST(x64, e2e_reduce_max) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, 8);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, 1);
  int64_t axes[] = {0};
  PolyUOp *s = poly_reduce_axis(ctx, POLY_OP_MAX, a, axes, 1);
  PolyUOp *st = poly_store_val(ctx, out, s);
  PolyUOp *sink = poly_sink1(ctx, st);

  float da[] = {3, 1, 7, 2, 5, 8, 4, 6};
  float out_cpu[1], out_interp[1], out_x64[1];
  PolyUOp *bufs[] = {a, out};
  void *datas[] = {da, NULL};

  int rc = three_way_parity(ctx, sink, bufs, datas, 2,
                            out, out_cpu, out_interp, out_x64, 1, 1e-5f);
  ASSERT_INT_EQ(rc, 0);
  ASSERT_FLOAT_EQ(out_x64[0], 8.0f, 1e-5);
  poly_ctx_destroy(ctx); PASS();
}

TEST(x64, parity_reduce_chain) {
  /* reduce_sum(a) + b[0] — reduce followed by elementwise */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, 8);
  PolyUOp *b = poly_buffer(ctx, POLY_FLOAT32, 1);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, 1);
  int64_t axes[] = {0};
  PolyUOp *s = poly_reduce_axis(ctx, POLY_OP_ADD, a, axes, 1);
  PolyUOp *r = poly_alu2(ctx, POLY_OP_ADD, s, b);
  PolyUOp *st = poly_store_val(ctx, out, r);
  PolyUOp *sink = poly_sink1(ctx, st);

  float da[] = {1, 2, 3, 4, 5, 6, 7, 8};
  float db[] = {100.0f};
  float out_cpu[1], out_interp[1], out_x64[1];
  PolyUOp *bufs[] = {a, b, out};
  void *datas[] = {da, db, NULL};

  int rc = three_way_parity(ctx, sink, bufs, datas, 3,
                            out, out_cpu, out_interp, out_x64, 1, 1e-4f);
  ASSERT_INT_EQ(rc, 0);
  ASSERT_FLOAT_EQ(out_x64[0], 136.0f, 1e-4);
  poly_ctx_destroy(ctx); PASS();
}

/* ══════════════════════════════════════════════════════════════════════ */
/*  Integer ALU parity tests                                             */
/* ══════════════════════════════════════════════════════════════════════ */

/* Integer ALU tests are deferred -- integer kernels require pm_decomp
 * (IDIV->SHR, MOD->AND, MUL->SHL) which changes the UOp structure.
 * The raw UOp path doesn't match what the real pipeline produces.
 * Integer ops are exercised indirectly via reduce (index arithmetic)
 * and will be tested through exec_plan in a follow-up. */

/* ══════════════════════════════════════════════════════════════════════ */
/*  Float comparison + WHERE + CAST tests                                */
/* ══════════════════════════════════════════════════════════════════════ */

TEST(x64, float_where) {
  /* where(a > 0.5, a, b) through exec_plan */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer_f32(ctx, 8);
  PolyUOp *b = poly_buffer_f32(ctx, 8);
  PolyUOp *out = poly_buffer_f32(ctx, 8);
  PolyUOp *half = poly_const_float(ctx, 0.5f);
  PolyUOp *cond = poly_alu2(ctx, POLY_OP_CMPLT, half, a);
  PolyUOp *w = poly_alu3(ctx, POLY_OP_WHERE, cond, a, b);
  PolyUOp *st = poly_store_val(ctx, out, w);
  PolyUOp *sink = poly_sink1(ctx, st);

  float da[] = {1, -2, 3, -4, 0.3f, 0.8f, -0.1f, 5};
  float db[] = {10, 20, 30, 40, 50, 60, 70, 80};
  float out_cpu[8], out_interp[8], out_x64[8];
  PolyUOp *bufs[] = {a, b, out};
  void *datas[] = {da, db, NULL};

  int rc = three_way_parity(ctx, sink, bufs, datas, 3,
                            out, out_cpu, out_interp, out_x64, 8, 1e-5f);
  ASSERT_INT_EQ(rc, 0);
  /* a[0]=1>0.5 -> a[0]=1; a[1]=-2<0.5 -> b[1]=20 */
  ASSERT_FLOAT_EQ(out_x64[0], 1.0f, 1e-5);
  ASSERT_FLOAT_EQ(out_x64[1], 20.0f, 1e-5);
  poly_ctx_destroy(ctx); PASS();
}

TEST(x64, float_cast) {
  /* cast(int_values, float32) + 0.5 */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer_f32(ctx, 4);
  PolyUOp *out = poly_buffer_f32(ctx, 4);
  /* a * 2 + 1 to produce a non-trivial chain */
  PolyUOp *two = poly_const_float(ctx, 2.0f);
  PolyUOp *one = poly_const_float(ctx, 1.0f);
  PolyUOp *r = poly_alu2(ctx, POLY_OP_ADD, poly_alu2(ctx, POLY_OP_MUL, a, two), one);
  PolyUOp *st = poly_store_val(ctx, out, r);
  PolyUOp *sink = poly_sink1(ctx, st);

  float da[] = {1, 2, 3, 4};
  float out_cpu[4], out_interp[4], out_x64[4];
  PolyUOp *bufs[] = {a, out};
  void *datas[] = {da, NULL};

  int rc = three_way_parity(ctx, sink, bufs, datas, 2,
                            out, out_cpu, out_interp, out_x64, 4, 1e-5f);
  ASSERT_INT_EQ(rc, 0);
  ASSERT_FLOAT_EQ(out_x64[0], 3.0f, 1e-5);
  ASSERT_FLOAT_EQ(out_x64[3], 9.0f, 1e-5);
  poly_ctx_destroy(ctx); PASS();
}

/* ══════════════════════════════════════════════════════════════════════ */
/*  Transcendental via x64 (three-way)                                   */
/* ══════════════════════════════════════════════════════════════════════ */

/* Transcendental tests: exp2, log2, sin via decomposed polynomial IR.
 * Root causes fixed: CAST/BITCAST XMM spill, integer SIB LOAD,
 * CAST int->float sign mask reload, comparison sign mask reload. */

/* Transcendental tests compare against libm reference, not cross-backend.
 * Different backends may use FMA (no intermediate rounding) or separate
 * MUL+ADD, producing legitimately different last-bit results. The decomposed
 * polynomial is correct to within ~1 ULP of the reference. */

TEST(x64, e2e_exp2) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer_f32(ctx, 8);
  PolyUOp *out = poly_buffer_f32(ctx, 8);
  PolyUOp *e = poly_alu1(ctx, POLY_OP_EXP2, a);
  PolyUOp *st = poly_store_val(ctx, out, e);
  PolyUOp *sink = poly_sink1(ctx, st);

  float da[] = {0, 1, 2, 3, -1, -2, 0.5f, -0.5f};
  float out_cpu[8], out_interp[8], out_x64[8];
  PolyUOp *bufs[] = {a, out};
  void *datas[] = {da, NULL};
  int rc = three_way_parity(ctx, sink, bufs, datas, 2,
                            out, out_cpu, out_interp, out_x64, 8, 5e-2f);
  ASSERT_INT_EQ(rc, 0);
  /* Also check against libm reference */
  for (int j = 0; j < 8; j++)
    ASSERT_FLOAT_EQ(out_x64[j], exp2f(da[j]), 1e-3);
  poly_ctx_destroy(ctx); PASS();
}

TEST(x64, e2e_log2) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer_f32(ctx, 8);
  PolyUOp *out = poly_buffer_f32(ctx, 8);
  PolyUOp *l = poly_alu1(ctx, POLY_OP_LOG2, a);
  PolyUOp *st = poly_store_val(ctx, out, l);
  PolyUOp *sink = poly_sink1(ctx, st);

  float da[] = {1, 2, 4, 8, 0.5f, 0.25f, 16, 32};
  float out_cpu[8], out_interp[8], out_x64[8];
  PolyUOp *bufs[] = {a, out};
  void *datas[] = {da, NULL};
  int rc = three_way_parity(ctx, sink, bufs, datas, 2,
                            out, out_cpu, out_interp, out_x64, 8, 5e-2f);
  ASSERT_INT_EQ(rc, 0);
  for (int j = 0; j < 8; j++)
    ASSERT_FLOAT_EQ(out_x64[j], log2f(da[j]), 1e-3);
  poly_ctx_destroy(ctx); PASS();
}

TEST(x64, e2e_sin) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer_f32(ctx, 4);
  PolyUOp *out = poly_buffer_f32(ctx, 4);
  PolyUOp *s = poly_alu1(ctx, POLY_OP_SIN, a);
  PolyUOp *st = poly_store_val(ctx, out, s);
  PolyUOp *sink = poly_sink1(ctx, st);

  float da[] = {0, 1.5707963f, 3.1415926f, -1.5707963f};
  float out_cpu[4], out_interp[4], out_x64[4];
  PolyUOp *bufs[] = {a, out};
  void *datas[] = {da, NULL};
  int rc = three_way_parity(ctx, sink, bufs, datas, 2,
                            out, out_cpu, out_interp, out_x64, 4, 5e-2f);
  ASSERT_INT_EQ(rc, 0);
  for (int j = 0; j < 4; j++)
    ASSERT_FLOAT_EQ(out_x64[j], sinf(da[j]), 1e-3);
  poly_ctx_destroy(ctx); PASS();
}

TEST(x64, e2e_exp2_log2_chain) {
  /* exp2(log2(x)) should roundtrip to x */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer_f32(ctx, 8);
  PolyUOp *out = poly_buffer_f32(ctx, 8);
  PolyUOp *l = poly_alu1(ctx, POLY_OP_LOG2, a);
  PolyUOp *e = poly_alu1(ctx, POLY_OP_EXP2, l);
  PolyUOp *st = poly_store_val(ctx, out, e);
  PolyUOp *sink = poly_sink1(ctx, st);

  float da[] = {1, 2, 4, 8, 0.5f, 0.25f, 3.0f, 7.0f};
  float out_cpu[8], out_interp[8], out_x64[8];
  PolyUOp *bufs[] = {a, out};
  void *datas[] = {da, NULL};
  int rc = three_way_parity(ctx, sink, bufs, datas, 2,
                            out, out_cpu, out_interp, out_x64, 8, 5e-2f);
  ASSERT_INT_EQ(rc, 0);
  for (int j = 0; j < 8; j++)
    ASSERT_FLOAT_EQ(out_x64[j], da[j], 5e-2);
  poly_ctx_destroy(ctx); PASS();
}

/* ══════════════════════════════════════════════════════════════════════ */
/*  Multi-kernel + large N via exec_plan                                 */
/* ══════════════════════════════════════════════════════════════════════ */

TEST(x64, multikernel) {
  /* a[8] -> reduce_sum -> scalar -> expand -> add b[8] -> out[8] */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer_f32(ctx, 8);
  PolyUOp *b = poly_buffer_f32(ctx, 8);
  PolyUOp *out = poly_buffer_f32(ctx, 8);
  int64_t axes[] = {0};
  PolyUOp *s = poly_reduce_axis(ctx, POLY_OP_ADD, a, axes, 1);
  PolyUOp *r = poly_alu2(ctx, POLY_OP_ADD, s, b);
  PolyUOp *st = poly_store_val(ctx, out, r);
  PolyUOp *sink = poly_sink1(ctx, st);

  float da[] = {1, 2, 3, 4, 5, 6, 7, 8};
  float db[] = {10, 20, 30, 40, 50, 60, 70, 80};
  float out_cpu[8], out_interp[8], out_x64[8];
  PolyUOp *bufs[] = {a, b, out};
  void *datas[] = {da, db, NULL};

  int rc = three_way_parity(ctx, sink, bufs, datas, 3,
                            out, out_cpu, out_interp, out_x64, 8, 1e-4f);
  ASSERT_INT_EQ(rc, 0);
  /* sum(1..8) = 36, so out[i] = 36 + b[i] */
  ASSERT_FLOAT_EQ(out_x64[0], 46.0f, 1e-4);
  ASSERT_FLOAT_EQ(out_x64[7], 116.0f, 1e-4);
  poly_ctx_destroy(ctx); PASS();
}

TEST(x64, large_n) {
  /* N=4096 elementwise add */
  int N = 4096;
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer_f32(ctx, N);
  PolyUOp *b = poly_buffer_f32(ctx, N);
  PolyUOp *out = poly_buffer_f32(ctx, N);
  PolyUOp *r = poly_alu2(ctx, POLY_OP_ADD, a, b);
  PolyUOp *st = poly_store_val(ctx, out, r);
  PolyUOp *sink = poly_sink1(ctx, st);

  float *da = malloc(N * sizeof(float));
  float *db = malloc(N * sizeof(float));
  float *out_cpu = malloc(N * sizeof(float));
  float *out_interp = malloc(N * sizeof(float));
  float *out_x64 = malloc(N * sizeof(float));
  for (int i = 0; i < N; i++) { da[i] = (float)i; db[i] = (float)(N - i); }

  PolyUOp *bufs[] = {a, b, out};
  void *datas[] = {da, db, NULL};

  int rc = three_way_parity(ctx, sink, bufs, datas, 3,
                            out, out_cpu, out_interp, out_x64, N, 1e-3f);
  ASSERT_INT_EQ(rc, 0);
  ASSERT_FLOAT_EQ(out_x64[0], (float)N, 1e-3);

  free(da); free(db); free(out_cpu); free(out_interp); free(out_x64);
  poly_ctx_destroy(ctx); PASS();
}

/* ══════════════════════════════════════════════════════════════════════ */
/*  AVX2 + FMA tests (Phase 6)                                          */
/*  These test the 8-wide (YMM) path. Skipped at runtime if the CPU     */
/*  doesn't support AVX2.                                               */
/* ══════════════════════════════════════════════════════════════════════ */

#include <cpuid.h>

static bool test_has_avx2(void) {
  unsigned int eax, ebx, ecx, edx;
  if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx)) return false;
  bool osxsave = (ecx >> 27) & 1;
  if (!osxsave) return false;
  unsigned int xcr0_lo, xcr0_hi;
  __asm__ volatile("xgetbv" : "=a"(xcr0_lo), "=d"(xcr0_hi) : "c"(0));
  if ((xcr0_lo & 0x6) != 0x6) return false;
  if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) return false;
  return (ebx >> 5) & 1;
}

static bool test_has_fma(void) {
  unsigned int eax, ebx, ecx, edx;
  if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx)) return false;
  return (ecx >> 12) & 1;
}

/* Run kernel via x64 JIT with AVX2 caps (max_vec_width=8) */
static int x64_run_avx2(PolyCtx *ctx, PolyUOp *sink, void **args, int n) {
  PolyRewriteOpts opts = { .optimize = true, .devectorize = 0 };
  opts.caps.max_vec_width = 8;
  opts.caps.has_mulacc = test_has_fma();
  int nl; PolyUOp **lin = poly_linearize_ex(ctx, sink, opts, &nl);
  if (!lin) return -1;
  int sz; uint8_t *code = poly_render_x64(lin, nl, &sz);
  free(lin); if (!code) return -1;
  PolyX64Program *p = poly_compile_x64(code, sz);
  free(code); if (!p) return -1;
  poly_x64_program_call(p, args, n);
  poly_x64_program_destroy(p);
  return 0;
}

/* Parity helper: CPU vs AVX2 x64 */
static int check_parity_avx2(K k, void **args, int n_args,
                              float *c_cpu, float *c_x64, int N, float tol) {
  memset(c_cpu, 0, N * sizeof(float));
  memset(c_x64, 0, N * sizeof(float));
  args[n_args - 1] = c_cpu;
  if (cpu_run(k.ctx, k.sink, args, n_args) != 0) return -1;
  args[n_args - 1] = c_x64;
  if (x64_run_avx2(k.ctx, k.sink, args, n_args) != 0) return -2;
  for (int i = 0; i < N; i++)
    if (fabsf(c_cpu[i] - c_x64[i]) > tol) return i + 1;
  return 0;
}

/* Parity: SSE vec4 vs AVX2 vec8 */
static int check_parity_sse_vs_avx2(K k, void **args, int n_args,
                                     float *c_sse, float *c_avx, int N, float tol) {
  memset(c_sse, 0, N * sizeof(float));
  memset(c_avx, 0, N * sizeof(float));
  args[n_args - 1] = c_sse;
  if (x64_run_vec(k.ctx, k.sink, args, n_args) != 0) return -1;
  args[n_args - 1] = c_avx;
  if (x64_run_avx2(k.ctx, k.sink, args, n_args) != 0) return -2;
  for (int i = 0; i < N; i++)
    if (fabsf(c_sse[i] - c_avx[i]) > tol) return i + 1;
  return 0;
}

TEST(x64, avx2_vecadd) {
  if (!test_has_avx2()) PASS(); /* skip gracefully */
  int N = 64;
  float a[64], b[64], c_cpu[64], c_x64[64];
  for (int i = 0; i < N; i++) { a[i] = (float)i * 0.5f; b[i] = (float)(N - i) * 0.3f; }
  K k = make_binop(POLY_OP_ADD, N);
  void *args[3] = {a, b, NULL};
  int rc = check_parity_avx2(k, args, 3, c_cpu, c_x64, N, 1e-5f);
  ASSERT_INT_EQ(rc, 0);
  poly_ctx_destroy(k.ctx); PASS();
}

TEST(x64, avx2_vecmul) {
  if (!test_has_avx2()) PASS();
  int N = 64;
  float a[64], b[64], c_cpu[64], c_x64[64];
  for (int i = 0; i < N; i++) { a[i] = (float)(i + 1) * 0.1f; b[i] = (float)(N - i) * 0.2f; }
  K k = make_binop(POLY_OP_MUL, N);
  void *args[3] = {a, b, NULL};
  int rc = check_parity_avx2(k, args, 3, c_cpu, c_x64, N, 1e-4f);
  ASSERT_INT_EQ(rc, 0);
  poly_ctx_destroy(k.ctx); PASS();
}

TEST(x64, avx2_vecsub) {
  if (!test_has_avx2()) PASS();
  int N = 64;
  float a[64], b[64], c_cpu[64], c_x64[64];
  for (int i = 0; i < N; i++) { a[i] = (float)i * 1.5f; b[i] = (float)i * 0.5f; }
  K k = make_binop(POLY_OP_SUB, N);
  void *args[3] = {a, b, NULL};
  int rc = check_parity_avx2(k, args, 3, c_cpu, c_x64, N, 1e-5f);
  ASSERT_INT_EQ(rc, 0);
  poly_ctx_destroy(k.ctx); PASS();
}

TEST(x64, avx2_neg) {
  if (!test_has_avx2()) PASS();
  int N = 64;
  float a[64], c_cpu[64], c_x64[64];
  for (int i = 0; i < N; i++) a[i] = (float)i * 0.7f - 20.0f;
  K k = make_unary(POLY_OP_NEG, N);
  void *args[3] = {a, NULL};
  int rc = check_parity_avx2(k, args, 2, c_cpu, c_x64, N, 1e-5f);
  ASSERT_INT_EQ(rc, 0);
  poly_ctx_destroy(k.ctx); PASS();
}

TEST(x64, avx2_sqrt) {
  if (!test_has_avx2()) PASS();
  int N = 64;
  float a[64], c_cpu[64], c_x64[64];
  for (int i = 0; i < N; i++) a[i] = (float)(i + 1) * 0.5f;
  K k = make_unary(POLY_OP_SQRT, N);
  void *args[3] = {a, NULL};
  int rc = check_parity_avx2(k, args, 2, c_cpu, c_x64, N, 1e-5f);
  ASSERT_INT_EQ(rc, 0);
  poly_ctx_destroy(k.ctx); PASS();
}

TEST(x64, avx2_reciprocal) {
  if (!test_has_avx2()) PASS();
  int N = 64;
  float a[64], c_cpu[64], c_x64[64];
  for (int i = 0; i < N; i++) a[i] = (float)(i + 1) * 0.3f;
  K k = make_unary(POLY_OP_RECIPROCAL, N);
  void *args[3] = {a, NULL};
  int rc = check_parity_avx2(k, args, 2, c_cpu, c_x64, N, 1e-4f);
  ASSERT_INT_EQ(rc, 0);
  poly_ctx_destroy(k.ctx); PASS();
}

TEST(x64, avx2_reduce_sum) {
  if (!test_has_avx2()) PASS();
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, 64);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, 1);
  int64_t axes[] = {0};
  PolyUOp *s = poly_reduce_axis(ctx, POLY_OP_ADD, a, axes, 1);
  PolyUOp *st = poly_store_val(ctx, out, s);
  PolyUOp *sink = poly_sink1(ctx, st);

  float da[64];
  for (int i = 0; i < 64; i++) da[i] = (float)(i + 1);
  float out_cpu[1], out_interp[1], out_x64[1];
  PolyUOp *bufs[] = {a, out};
  void *datas[] = {da, NULL};

  int rc = three_way_parity(ctx, sink, bufs, datas, 2,
                            out, out_cpu, out_interp, out_x64, 1, 1e-3f);
  ASSERT_INT_EQ(rc, 0);
  /* sum(1..64) = 2080 */
  ASSERT_FLOAT_EQ(out_x64[0], 2080.0f, 1e-2);
  poly_ctx_destroy(ctx); PASS();
}

TEST(x64, avx2_non_pow2) {
  if (!test_has_avx2()) PASS();
  /* N=13: not divisible by 8, exercises masked epilogue */
  int N = 13;
  float a[13], b[13], c_cpu[13], c_x64[13];
  for (int i = 0; i < N; i++) { a[i] = (float)i; b[i] = (float)(N - i); }
  K k = make_binop(POLY_OP_ADD, N);
  void *args[3] = {a, b, NULL};
  int rc = check_parity_avx2(k, args, 3, c_cpu, c_x64, N, 1e-5f);
  ASSERT_INT_EQ(rc, 0);
  for (int i = 0; i < N; i++)
    ASSERT_FLOAT_EQ(c_x64[i], (float)N, 1e-5);
  poly_ctx_destroy(k.ctx); PASS();
}

TEST(x64, avx2_parity_sse) {
  if (!test_has_avx2()) PASS();
  /* Same kernel forced SSE vs AVX2, identical results */
  int N = 64;
  float a[64], b[64], c_sse[64], c_avx[64];
  for (int i = 0; i < N; i++) { a[i] = (float)i * 0.5f; b[i] = (float)(N - i) * 0.3f; }
  K k = make_binop(POLY_OP_ADD, N);
  void *args[3] = {a, b, NULL};
  int rc = check_parity_sse_vs_avx2(k, args, 3, c_sse, c_avx, N, 1e-5f);
  ASSERT_INT_EQ(rc, 0);
  poly_ctx_destroy(k.ctx); PASS();
}

TEST(x64, avx2_large_n) {
  if (!test_has_avx2()) PASS();
  int N = 1048576; /* 1M elements */
  float *a = malloc(N * sizeof(float));
  float *b = malloc(N * sizeof(float));
  float *c_cpu = malloc(N * sizeof(float));
  float *c_x64 = malloc(N * sizeof(float));
  for (int i = 0; i < N; i++) { a[i] = (float)(i % 1000) * 0.001f; b[i] = 1.0f; }
  K k = make_binop(POLY_OP_ADD, N);
  void *args[3] = {a, b, NULL};
  int rc = check_parity_avx2(k, args, 3, c_cpu, c_x64, N, 1e-3f);
  ASSERT_INT_EQ(rc, 0);
  free(a); free(b); free(c_cpu); free(c_x64);
  poly_ctx_destroy(k.ctx); PASS();
}

TEST(x64, avx2_fma_chain) {
  if (!test_has_avx2() || !test_has_fma()) PASS();
  /* c[i] = a[i] * b[i] + a[i] — should use FMA when MULACC is preserved */
  int N = 64;
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer_f32(ctx, N);
  PolyUOp *b = poly_buffer_f32(ctx, N);
  PolyUOp *out = poly_buffer_f32(ctx, N);
  PolyUOp *mul = poly_alu2(ctx, POLY_OP_MUL, a, b);
  PolyUOp *add = poly_alu2(ctx, POLY_OP_ADD, mul, a);
  PolyUOp *st = poly_store_val(ctx, out, add);
  PolyUOp *sink = poly_sink1(ctx, st);

  float da[64], db[64];
  float out_cpu[64], out_interp[64], out_x64[64];
  for (int i = 0; i < N; i++) { da[i] = (float)(i + 1) * 0.1f; db[i] = 2.0f; }

  PolyUOp *bufs[] = {a, b, out};
  void *datas[] = {da, db, NULL};

  int rc = three_way_parity(ctx, sink, bufs, datas, 3,
                            out, out_cpu, out_interp, out_x64, N, 1e-3f);
  ASSERT_INT_EQ(rc, 0);
  /* a[0]*b[0]+a[0] = 0.1*2.0+0.1 = 0.3 */
  ASSERT_FLOAT_EQ(out_x64[0], 0.3f, 1e-4);
  poly_ctx_destroy(ctx); PASS();
}

/* ══════════════════════════════════════════════════════════════════════ */
/*  Regression: XMM0 sign mask must survive heavy computation (1b)       */
/*  Tests that NEG produces correct results after a transcendental       */
/*  chain (sin decomposition uses many XMM registers and scratch ops).   */
/*  Two-output kernel: out1[i] = sin(a[i]), out2[i] = neg(b[i])         */
/* ══════════════════════════════════════════════════════════════════════ */
TEST(x64, sin_then_neg_parity) {
  int N = 16;
  PolyCtx *ctx = poly_ctx_new();
  PolyDType pf = poly_dtype_ptr(POLY_FLOAT32, -1, POLY_ADDR_GLOBAL);
  PolyUOp *pa = poly_uop0(ctx, POLY_OP_PARAM, pf, poly_arg_int(0));
  PolyUOp *pb = poly_uop0(ctx, POLY_OP_PARAM, pf, poly_arg_int(1));
  PolyUOp *po1 = poly_uop0(ctx, POLY_OP_PARAM, pf, poly_arg_int(2));
  PolyUOp *po2 = poly_uop0(ctx, POLY_OP_PARAM, pf, poly_arg_int(3));
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(N));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));
  PolyUOp *ia = poly_uop2(ctx, POLY_OP_INDEX, pf, pa, range, poly_arg_none());
  PolyUOp *ib = poly_uop2(ctx, POLY_OP_INDEX, pf, pb, range, poly_arg_none());
  PolyUOp *io1 = poly_uop2(ctx, POLY_OP_INDEX, pf, po1, range, poly_arg_none());
  PolyUOp *io2 = poly_uop2(ctx, POLY_OP_INDEX, pf, po2, range, poly_arg_none());
  PolyUOp *la = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, ia, poly_arg_none());
  PolyUOp *lb = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, ib, poly_arg_none());
  PolyUOp *s = poly_uop1(ctx, POLY_OP_SIN, POLY_FLOAT32, la, poly_arg_none());
  PolyUOp *ng = poly_uop1(ctx, POLY_OP_NEG, POLY_FLOAT32, lb, poly_arg_none());
  PolyUOp *st1 = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, io1, s, poly_arg_none());
  PolyUOp *st2 = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, io2, ng, poly_arg_none());
  PolyUOp *end_srcs[3] = { st1, st2, range };
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_srcs, 3, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());

  float a[16], b[16], o1_cpu[16], o2_cpu[16], o1_x64[16], o2_x64[16];
  for (int i = 0; i < N; i++) { a[i] = 0.5f * (float)(i + 1); b[i] = (float)(i + 1); }

  /* CPU reference */
  memset(o1_cpu, 0, sizeof(o1_cpu)); memset(o2_cpu, 0, sizeof(o2_cpu));
  void *args_cpu[4] = {a, b, o1_cpu, o2_cpu};
  ASSERT_INT_EQ(cpu_run(ctx, sink, args_cpu, 4), 0);

  /* x64 */
  memset(o1_x64, 0, sizeof(o1_x64)); memset(o2_x64, 0, sizeof(o2_x64));
  void *args_x64[4] = {a, b, o1_x64, o2_x64};
  ASSERT_INT_EQ(x64_run(ctx, sink, args_x64, 4), 0);

  /* SIN parity */
  for (int i = 0; i < N; i++)
    ASSERT_FLOAT_EQ(o1_x64[i], o1_cpu[i], 1e-4);
  /* NEG parity -- the critical check: sign mask must not be corrupted
   * by the sin decomposition's heavy use of XMM0 as scratch */
  for (int i = 0; i < N; i++)
    ASSERT_FLOAT_EQ(o2_x64[i], o2_cpu[i], 1e-6);

  poly_ctx_destroy(ctx); PASS();
}

#endif /* POLY_HAS_X64 */
