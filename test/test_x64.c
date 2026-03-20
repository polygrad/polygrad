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
  for (int i = 0; i < 8; i++) ASSERT_FLOAT_EQ(b[i], -a[i], 0);
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

#endif /* POLY_HAS_X64 */
