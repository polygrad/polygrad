/*
 * test_x86.c -- tinygrad-style x86 ISA backend tests.
 *
 * These tests exercise the new X86Renderer-shaped path:
 *   rewritten SINK -> INS -> linear-scan regalloc -> post-regalloc -> bytes.
 */

#ifdef POLY_HAS_X86

#include "test_harness.h"
#include "../src/codegen.h"
#include "../src/device.h"
#include "../src/engine/realize.h"
#include "../src/engine/schedule.h"
#include "../src/frontend.h"
#include "../src/models/tabm.h"
#include "../src/nn.h"
#include "../src/schedule/rangeify.h"
#include "../src/tensor.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static uint16_t x86_f32_to_f16_bits(float f) {
  union {
    float f;
    uint32_t u;
  } v = {f};
  uint32_t sign = (v.u >> 16) & 0x8000u;
  int32_t exp = (int32_t)((v.u >> 23) & 0xffu) - 127 + 15;
  uint32_t mant = v.u & 0x7fffffu;
  if (exp <= 0) return (uint16_t)sign;
  if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
  return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}

static float x86_f16_bits_to_f32(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1fu;
  uint32_t mant = h & 0x03ffu;
  uint32_t bits;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign;
    } else {
      exp = 1;
      while ((mant & 0x0400u) == 0) {
        mant <<= 1;
        exp--;
      }
      mant &= 0x03ffu;
      bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    bits = sign | 0x7f800000u | (mant << 13);
  } else {
    bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
  }
  union {
    uint32_t u;
    float f;
  } v = {bits};
  return v.f;
}

static PolyUOp *x86_make_vecadd(PolyCtx *ctx, int n) {
  PolyDType pf = poly_dtype_ptr(POLY_FLOAT32, -1, POLY_ADDR_GLOBAL);
  PolyUOp *p0 = poly_uop0(ctx, POLY_OP_PARAM, pf, poly_arg_int(0));
  PolyUOp *p1 = poly_uop0(ctx, POLY_OP_PARAM, pf, poly_arg_int(1));
  PolyUOp *p2 = poly_uop0(ctx, POLY_OP_PARAM, pf, poly_arg_int(2));
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(n));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));
  PolyUOp *i0 = poly_uop2(ctx, POLY_OP_INDEX, pf, p0, range, poly_arg_none());
  PolyUOp *i1 = poly_uop2(ctx, POLY_OP_INDEX, pf, p1, range, poly_arg_none());
  PolyUOp *i2 = poly_uop2(ctx, POLY_OP_INDEX, pf, p2, range, poly_arg_none());
  PolyUOp *l0 = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, i0, poly_arg_none());
  PolyUOp *l1 = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, i1, poly_arg_none());
  PolyUOp *sum = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, l0, l1, poly_arg_none());
  PolyUOp *st = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, i2, sum, poly_arg_none());
  PolyUOp *end_srcs[2] = {st, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_srcs, 2, poly_arg_none());
  return poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());
}

static PolyUOp *x86_make_abs_plus_bias(PolyCtx *ctx, int n) {
  PolyDType pf = poly_dtype_ptr(POLY_FLOAT32, -1, POLY_ADDR_GLOBAL);
  PolyUOp *p0 = poly_uop0(ctx, POLY_OP_PARAM, pf, poly_arg_int(0));
  PolyUOp *p1 = poly_uop0(ctx, POLY_OP_PARAM, pf, poly_arg_int(1));
  PolyUOp *p2 = poly_uop0(ctx, POLY_OP_PARAM, pf, poly_arg_int(2));
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(n));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));
  PolyUOp *i0 = poly_uop2(ctx, POLY_OP_INDEX, pf, p0, range, poly_arg_none());
  PolyUOp *i1 = poly_uop2(ctx, POLY_OP_INDEX, pf, p1, range, poly_arg_none());
  PolyUOp *i2 = poly_uop2(ctx, POLY_OP_INDEX, pf, p2, range, poly_arg_none());
  PolyUOp *x = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, i0, poly_arg_none());
  PolyUOp *bias = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, i1, poly_arg_none());
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(0.0));
  PolyUOp *mask = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, zero, x, poly_arg_none());
  PolyUOp *neg = poly_uop1(ctx, POLY_OP_NEG, POLY_FLOAT32, x, poly_arg_none());
  PolyUOp *sel_srcs[3] = {mask, x, neg};
  PolyUOp *sel = poly_uop(ctx, POLY_OP_WHERE, POLY_FLOAT32, sel_srcs, 3, poly_arg_none());
  PolyUOp *out = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, sel, bias, poly_arg_none());
  PolyUOp *st = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, i2, out, poly_arg_none());
  PolyUOp *end_srcs[2] = {st, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_srcs, 2, poly_arg_none());
  return poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());
}

static PolyUOp *x86_make_f32_add_const(PolyCtx *ctx, int n, float c) {
  PolyDType pf = poly_dtype_ptr(POLY_FLOAT32, -1, POLY_ADDR_GLOBAL);
  PolyUOp *p0 = poly_uop0(ctx, POLY_OP_PARAM, pf, poly_arg_int(0));
  PolyUOp *p1 = poly_uop0(ctx, POLY_OP_PARAM, pf, poly_arg_int(1));
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(n));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));
  PolyUOp *i0 = poly_uop2(ctx, POLY_OP_INDEX, pf, p0, range, poly_arg_none());
  PolyUOp *i1 = poly_uop2(ctx, POLY_OP_INDEX, pf, p1, range, poly_arg_none());
  PolyUOp *x = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, i0, poly_arg_none());
  PolyUOp *cv = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(c));
  PolyUOp *sum = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, x, cv, poly_arg_none());
  PolyUOp *st = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, i1, sum, poly_arg_none());
  PolyUOp *end_srcs[2] = {st, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_srcs, 2, poly_arg_none());
  return poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());
}

typedef PolyUOp *(*X86ExprFn)(PolyCtx *, PolyUOp *, PolyUOp *, PolyUOp *);
typedef PolyUOp *(*X86Expr3Fn)(PolyCtx *, PolyUOp *, PolyUOp *, PolyUOp *, PolyUOp *);

static PolyUOp *x86_make_one_range(
    PolyCtx *ctx,
    PolyDType ptr_dt,
    PolyDType val_dt,
    int n,
    X86ExprFn expr
) {
  PolyUOp *p0 = poly_uop0(ctx, POLY_OP_PARAM, ptr_dt, poly_arg_int(0));
  PolyUOp *p1 = poly_uop0(ctx, POLY_OP_PARAM, ptr_dt, poly_arg_int(1));
  PolyUOp *p2 = poly_uop0(ctx, POLY_OP_PARAM, ptr_dt, poly_arg_int(2));
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(n));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));
  PolyUOp *i0 = poly_uop2(ctx, POLY_OP_INDEX, ptr_dt, p0, range, poly_arg_none());
  PolyUOp *i1 = poly_uop2(ctx, POLY_OP_INDEX, ptr_dt, p1, range, poly_arg_none());
  PolyUOp *i2 = poly_uop2(ctx, POLY_OP_INDEX, ptr_dt, p2, range, poly_arg_none());
  PolyUOp *a = poly_uop1(ctx, POLY_OP_LOAD, val_dt, i0, poly_arg_none());
  PolyUOp *b = poly_uop1(ctx, POLY_OP_LOAD, val_dt, i1, poly_arg_none());
  PolyUOp *out = expr(ctx, a, b, range);
  PolyUOp *st = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, i2, out, poly_arg_none());
  PolyUOp *end_srcs[2] = {st, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_srcs, 2, poly_arg_none());
  return poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());
}

static PolyUOp *x86_make_three_input_range(
    PolyCtx *ctx,
    PolyDType ptr_dt,
    PolyDType val_dt,
    int n,
    X86Expr3Fn expr
) {
  PolyUOp *p0 = poly_uop0(ctx, POLY_OP_PARAM, ptr_dt, poly_arg_int(0));
  PolyUOp *p1 = poly_uop0(ctx, POLY_OP_PARAM, ptr_dt, poly_arg_int(1));
  PolyUOp *p2 = poly_uop0(ctx, POLY_OP_PARAM, ptr_dt, poly_arg_int(2));
  PolyUOp *p3 = poly_uop0(ctx, POLY_OP_PARAM, ptr_dt, poly_arg_int(3));
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(n));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));
  PolyUOp *i0 = poly_uop2(ctx, POLY_OP_INDEX, ptr_dt, p0, range, poly_arg_none());
  PolyUOp *i1 = poly_uop2(ctx, POLY_OP_INDEX, ptr_dt, p1, range, poly_arg_none());
  PolyUOp *i2 = poly_uop2(ctx, POLY_OP_INDEX, ptr_dt, p2, range, poly_arg_none());
  PolyUOp *i3 = poly_uop2(ctx, POLY_OP_INDEX, ptr_dt, p3, range, poly_arg_none());
  PolyUOp *a = poly_uop1(ctx, POLY_OP_LOAD, val_dt, i0, poly_arg_none());
  PolyUOp *b = poly_uop1(ctx, POLY_OP_LOAD, val_dt, i1, poly_arg_none());
  PolyUOp *c = poly_uop1(ctx, POLY_OP_LOAD, val_dt, i2, poly_arg_none());
  PolyUOp *out = expr(ctx, a, b, c, range);
  PolyUOp *st = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, i3, out, poly_arg_none());
  PolyUOp *end_srcs[2] = {st, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_srcs, 2, poly_arg_none());
  return poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());
}

static PolyUOp *x86_make_seven_input_sum(PolyCtx *ctx, int n) {
  PolyDType pf = poly_dtype_ptr(POLY_FLOAT32, -1, POLY_ADDR_GLOBAL);
  PolyUOp *params[8];
  for (int i = 0; i < 8; i++)
    params[i] = poly_uop0(ctx, POLY_OP_PARAM, pf, poly_arg_int(i));
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(n));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));
  PolyUOp *sum = NULL;
  for (int i = 0; i < 7; i++) {
    PolyUOp *idx = poly_uop2(ctx, POLY_OP_INDEX, pf, params[i], range, poly_arg_none());
    PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx, poly_arg_none());
    sum = sum ? poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, sum, load, poly_arg_none()) : load;
  }
  PolyUOp *out_idx = poly_uop2(ctx, POLY_OP_INDEX, pf, params[7], range, poly_arg_none());
  PolyUOp *st = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out_idx, sum, poly_arg_none());
  PolyUOp *end_srcs[2] = {st, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_srcs, 2, poly_arg_none());
  return poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());
}

static PolyUOp *x86_make_unary_cast_range(
    PolyCtx *ctx,
    PolyDType src_ptr_dt,
    PolyDType src_val_dt,
    PolyDType dst_ptr_dt,
    PolyDType dst_val_dt,
    PolyOps op,
    int n
) {
  PolyUOp *p0 = poly_uop0(ctx, POLY_OP_PARAM, src_ptr_dt, poly_arg_int(0));
  PolyUOp *p1 = poly_uop0(ctx, POLY_OP_PARAM, dst_ptr_dt, poly_arg_int(1));
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(n));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));
  PolyUOp *i0 = poly_uop2(ctx, POLY_OP_INDEX, src_ptr_dt, p0, range, poly_arg_none());
  PolyUOp *i1 = poly_uop2(ctx, POLY_OP_INDEX, dst_ptr_dt, p1, range, poly_arg_none());
  PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, src_val_dt, i0, poly_arg_none());
  PolyUOp *cast = poly_uop1(ctx, op, dst_val_dt, load, poly_arg_none());
  PolyUOp *st = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, i1, cast, poly_arg_none());
  PolyUOp *end_srcs[2] = {st, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_srcs, 2, poly_arg_none());
  return poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());
}

static PolyUOp *x86_expr_f32_unary_div(
    PolyCtx *ctx,
    PolyUOp *f,
    PolyUOp *g,
    PolyUOp *h,
    PolyUOp *range
) {
  (void)range;
  PolyUOp *sqrt_f = poly_uop1(ctx, POLY_OP_SQRT, POLY_FLOAT32, f, poly_arg_none());
  PolyUOp *div = poly_uop2(ctx, POLY_OP_FDIV, POLY_FLOAT32, h, g, poly_arg_none());
  PolyUOp *trunc = poly_uop1(ctx, POLY_OP_TRUNC, POLY_FLOAT32, div, poly_arg_none());
  return poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, sqrt_f, trunc, poly_arg_none());
}

static PolyUOp *x86_expr_f64_unary_div(PolyCtx *ctx, PolyUOp *d, PolyUOp *e, PolyUOp *range) {
  (void)range;
  PolyUOp *sqrt_d = poly_uop1(ctx, POLY_OP_SQRT, POLY_FLOAT64, d, poly_arg_none());
  PolyUOp *div = poly_uop2(ctx, POLY_OP_FDIV, POLY_FLOAT64, d, e, poly_arg_none());
  PolyUOp *trunc = poly_uop1(ctx, POLY_OP_TRUNC, POLY_FLOAT64, div, poly_arg_none());
  return poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT64, sqrt_d, trunc, poly_arg_none());
}

static PolyUOp *x86_expr_f64_where(PolyCtx *ctx, PolyUOp *a, PolyUOp *b, PolyUOp *range) {
  (void)range;
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT64, poly_arg_float(0.0));
  PolyUOp *mask = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, zero, a, poly_arg_none());
  PolyUOp *sum = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT64, a, b, poly_arg_none());
  PolyUOp *srcs[3] = {mask, sum, b};
  return poly_uop(ctx, POLY_OP_WHERE, POLY_FLOAT64, srcs, 3, poly_arg_none());
}

static PolyUOp *x86_expr_f32_max(PolyCtx *ctx, PolyUOp *a, PolyUOp *b, PolyUOp *range) {
  (void)range;
  return poly_uop2(ctx, POLY_OP_MAX, POLY_FLOAT32, a, b, poly_arg_none());
}

static PolyUOp *x86_expr_f32_min_where(PolyCtx *ctx, PolyUOp *a, PolyUOp *b, PolyUOp *range) {
  (void)range;
  PolyUOp *mask = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, a, b, poly_arg_none());
  PolyUOp *srcs[3] = {mask, a, b};
  return poly_uop(ctx, POLY_OP_WHERE, POLY_FLOAT32, srcs, 3, poly_arg_none());
}

static PolyUOp *x86_expr_i32_mix(PolyCtx *ctx, PolyUOp *a, PolyUOp *b, PolyUOp *range) {
  (void)range;
  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, POLY_INT32, a, b, poly_arg_none());
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, mul, a, poly_arg_none());
  return poly_uop2(ctx, POLY_OP_SUB, POLY_INT32, add, b, poly_arg_none());
}

static PolyUOp *x86_make_f32_reused_mul_add(PolyCtx *ctx) {
  PolyDType pf = poly_dtype_ptr(POLY_FLOAT32, -1, POLY_ADDR_GLOBAL);
  PolyUOp *a = poly_uop0(ctx, POLY_OP_PARAM, pf, poly_arg_int(0));
  PolyUOp *b = poly_uop0(ctx, POLY_OP_PARAM, pf, poly_arg_int(1));
  PolyUOp *out = poly_uop0(ctx, POLY_OP_PARAM, pf, poly_arg_int(2));
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *la = poly_uop1(
      ctx, POLY_OP_LOAD, POLY_FLOAT32, poly_uop2(ctx, POLY_OP_INDEX, pf, a, zero, poly_arg_none()),
      poly_arg_none()
  );
  PolyUOp *lb = poly_uop1(
      ctx, POLY_OP_LOAD, POLY_FLOAT32, poly_uop2(ctx, POLY_OP_INDEX, pf, b, zero, poly_arg_none()),
      poly_arg_none()
  );
  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, la, lb, poly_arg_none());
  PolyUOp *sum = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, mul, mul, poly_arg_none());
  PolyUOp *st = poly_uop2(
      ctx, POLY_OP_STORE, POLY_VOID, poly_uop2(ctx, POLY_OP_INDEX, pf, out, zero, poly_arg_none()),
      sum, poly_arg_none()
  );
  return poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, st, poly_arg_none());
}

static PolyUOp *x86_make_i32_complex_address_load(PolyCtx *ctx) {
  PolyDType pi = poly_dtype_ptr(POLY_INT32, -1, POLY_ADDR_GLOBAL);
  PolyUOp *in = poly_uop0(ctx, POLY_OP_PARAM, pi, poly_arg_int(0));
  PolyUOp *out = poly_uop0(ctx, POLY_OP_PARAM, pi, poly_arg_int(1));
  PolyUOp *i = poly_uop0(ctx, POLY_OP_PARAM, POLY_INT32, poly_arg_int(2));
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(1));
  PolyUOp *idx = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, i, one, poly_arg_none());
  PolyUOp *load = poly_uop1(
      ctx, POLY_OP_LOAD, POLY_INT32, poly_uop2(ctx, POLY_OP_INDEX, pi, in, idx, poly_arg_none()),
      poly_arg_none()
  );
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *st = poly_uop2(
      ctx, POLY_OP_STORE, POLY_VOID, poly_uop2(ctx, POLY_OP_INDEX, pi, out, zero, poly_arg_none()),
      load, poly_arg_none()
  );
  return poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, st, poly_arg_none());
}

static PolyUOp *x86_make_i32_fold_load_add(PolyCtx *ctx) {
  PolyDType pi = poly_dtype_ptr(POLY_INT32, -1, POLY_ADDR_GLOBAL);
  PolyUOp *in = poly_uop0(ctx, POLY_OP_PARAM, pi, poly_arg_int(0));
  PolyUOp *out = poly_uop0(ctx, POLY_OP_PARAM, pi, poly_arg_int(1));
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(1));
  PolyUOp *l0 = poly_uop1(
      ctx, POLY_OP_LOAD, POLY_INT32, poly_uop2(ctx, POLY_OP_INDEX, pi, in, zero, poly_arg_none()),
      poly_arg_none()
  );
  PolyUOp *l1 = poly_uop1(
      ctx, POLY_OP_LOAD, POLY_INT32, poly_uop2(ctx, POLY_OP_INDEX, pi, in, one, poly_arg_none()),
      poly_arg_none()
  );
  PolyUOp *sum = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, l0, l1, poly_arg_none());
  PolyUOp *st = poly_uop2(
      ctx, POLY_OP_STORE, POLY_VOID, poly_uop2(ctx, POLY_OP_INDEX, pi, out, zero, poly_arg_none()),
      sum, poly_arg_none()
  );
  return poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, st, poly_arg_none());
}

static PolyUOp *x86_make_i32_multiuse_load_add(PolyCtx *ctx) {
  PolyDType pi = poly_dtype_ptr(POLY_INT32, -1, POLY_ADDR_GLOBAL);
  PolyUOp *in = poly_uop0(ctx, POLY_OP_PARAM, pi, poly_arg_int(0));
  PolyUOp *out = poly_uop0(ctx, POLY_OP_PARAM, pi, poly_arg_int(1));
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *load = poly_uop1(
      ctx, POLY_OP_LOAD, POLY_INT32, poly_uop2(ctx, POLY_OP_INDEX, pi, in, zero, poly_arg_none()),
      poly_arg_none()
  );
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(1));
  PolyUOp *plus = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, load, one, poly_arg_none());
  PolyUOp *sum = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, plus, load, poly_arg_none());
  PolyUOp *st = poly_uop2(
      ctx, POLY_OP_STORE, POLY_VOID, poly_uop2(ctx, POLY_OP_INDEX, pi, out, zero, poly_arg_none()),
      sum, poly_arg_none()
  );
  return poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, st, poly_arg_none());
}

static PolyUOp *x86_make_i32_multiuse_load_mul(PolyCtx *ctx) {
  PolyDType pi = poly_dtype_ptr(POLY_INT32, -1, POLY_ADDR_GLOBAL);
  PolyUOp *in = poly_uop0(ctx, POLY_OP_PARAM, pi, poly_arg_int(0));
  PolyUOp *out = poly_uop0(ctx, POLY_OP_PARAM, pi, poly_arg_int(1));
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *load = poly_uop1(
      ctx, POLY_OP_LOAD, POLY_INT32, poly_uop2(ctx, POLY_OP_INDEX, pi, in, zero, poly_arg_none()),
      poly_arg_none()
  );
  PolyUOp *prod = poly_uop2(ctx, POLY_OP_MUL, POLY_INT32, load, load, poly_arg_none());
  PolyUOp *st = poly_uop2(
      ctx, POLY_OP_STORE, POLY_VOID, poly_uop2(ctx, POLY_OP_INDEX, pi, out, zero, poly_arg_none()),
      prod, poly_arg_none()
  );
  return poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, st, poly_arg_none());
}

static PolyUOp *x86_expr_i32_where(PolyCtx *ctx, PolyUOp *a, PolyUOp *b, PolyUOp *range) {
  (void)range;
  PolyUOp *mask = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, a, b, poly_arg_none());
  PolyUOp *srcs[3] = {mask, a, b};
  return poly_uop(ctx, POLY_OP_WHERE, POLY_INT32, srcs, 3, poly_arg_none());
}

static PolyUOp *x86_expr_i32_flag_clobber_where(
    PolyCtx *ctx,
    PolyUOp *a,
    PolyUOp *b,
    PolyUOp *range
) {
  (void)range;
  PolyUOp *mask = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, a, b, poly_arg_none());
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(1));
  PolyUOp *clobber = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, a, one, poly_arg_none());
  PolyUOp *lo_srcs[3] = {mask, a, b};
  PolyUOp *lo = poly_uop(ctx, POLY_OP_WHERE, POLY_INT32, lo_srcs, 3, poly_arg_none());
  PolyUOp *hi_srcs[3] = {mask, clobber, a};
  PolyUOp *hi = poly_uop(ctx, POLY_OP_WHERE, POLY_INT32, hi_srcs, 3, poly_arg_none());
  return poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, lo, hi, poly_arg_none());
}

static PolyUOp *x86_expr_i64_mix(PolyCtx *ctx, PolyUOp *a, PolyUOp *b, PolyUOp *range) {
  (void)range;
  PolyUOp *three = poly_uop0(ctx, POLY_OP_CONST, POLY_INT64, poly_arg_int(3));
  PolyUOp *two = poly_uop0(ctx, POLY_OP_CONST, POLY_INT64, poly_arg_int(2));
  PolyUOp *sum = poly_uop2(ctx, POLY_OP_ADD, POLY_INT64, a, b, poly_arg_none());
  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, POLY_INT64, sum, three, poly_arg_none());
  PolyUOp *shl = poly_uop2(ctx, POLY_OP_SHL, POLY_INT64, a, two, poly_arg_none());
  return poly_uop2(ctx, POLY_OP_XOR, POLY_INT64, mul, shl, poly_arg_none());
}

static PolyUOp *x86_expr_i64_where(PolyCtx *ctx, PolyUOp *a, PolyUOp *b, PolyUOp *range) {
  (void)range;
  PolyUOp *mask = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, a, b, poly_arg_none());
  PolyUOp *srcs[3] = {mask, a, b};
  return poly_uop(ctx, POLY_OP_WHERE, POLY_INT64, srcs, 3, poly_arg_none());
}

static PolyUOp *x86_expr_u32_where(PolyCtx *ctx, PolyUOp *a, PolyUOp *b, PolyUOp *range) {
  (void)range;
  PolyUOp *mask = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, a, b, poly_arg_none());
  PolyUOp *sum = poly_uop2(ctx, POLY_OP_ADD, POLY_UINT32, a, b, poly_arg_none());
  PolyUOp *diff = poly_uop2(ctx, POLY_OP_SUB, POLY_UINT32, a, b, poly_arg_none());
  PolyUOp *srcs[3] = {mask, sum, diff};
  return poly_uop(ctx, POLY_OP_WHERE, POLY_UINT32, srcs, 3, poly_arg_none());
}

static PolyUOp *x86_expr_u64_cdiv(PolyCtx *ctx, PolyUOp *a, PolyUOp *b, PolyUOp *range) {
  (void)range;
  return poly_uop2(ctx, POLY_OP_CDIV, POLY_UINT64, a, b, poly_arg_none());
}

static PolyUOp *x86_expr_u64_cdiv_all_ones(PolyCtx *ctx, PolyUOp *a, PolyUOp *b, PolyUOp *range) {
  (void)b;
  (void)range;
  PolyUOp *all_ones = poly_uop0(ctx, POLY_OP_CONST, POLY_UINT64, poly_arg_int(-1));
  return poly_uop2(ctx, POLY_OP_CDIV, POLY_UINT64, a, all_ones, poly_arg_none());
}

static PolyUOp *x86_expr_i64_cdiv(PolyCtx *ctx, PolyUOp *a, PolyUOp *b, PolyUOp *range) {
  (void)range;
  return poly_uop2(ctx, POLY_OP_CDIV, POLY_INT64, a, b, poly_arg_none());
}

static PolyUOp *x86_expr_u8_cdiv(PolyCtx *ctx, PolyUOp *a, PolyUOp *b, PolyUOp *range) {
  (void)range;
  return poly_uop2(ctx, POLY_OP_CDIV, POLY_UINT8, a, b, poly_arg_none());
}

static PolyUOp *x86_expr_i8_cdiv(PolyCtx *ctx, PolyUOp *a, PolyUOp *b, PolyUOp *range) {
  (void)range;
  return poly_uop2(ctx, POLY_OP_CDIV, POLY_INT8, a, b, poly_arg_none());
}

static PolyUOp *x86_expr_i8_mul(PolyCtx *ctx, PolyUOp *a, PolyUOp *b, PolyUOp *range) {
  (void)ctx;
  (void)range;
  return poly_uop2(ctx, POLY_OP_MUL, POLY_INT8, a, b, poly_arg_none());
}

static PolyUOp *x86_expr_i8_where(PolyCtx *ctx, PolyUOp *a, PolyUOp *b, PolyUOp *range) {
  (void)range;
  PolyUOp *mask = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, a, b, poly_arg_none());
  PolyUOp *srcs[3] = {mask, a, b};
  return poly_uop(ctx, POLY_OP_WHERE, POLY_INT8, srcs, 3, poly_arg_none());
}

static PolyUOp *x86_expr_f16_alu(PolyCtx *ctx, PolyUOp *a, PolyUOp *b, PolyUOp *range) {
  (void)b;
  (void)range;
  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT16, a, a, poly_arg_none());
  return poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT16, mul, a, poly_arg_none());
}

static PolyUOp *x86_expr_f16_exp2(PolyCtx *ctx, PolyUOp *a, PolyUOp *b, PolyUOp *range) {
  (void)b;
  (void)range;
  return poly_uop1(ctx, POLY_OP_EXP2, POLY_FLOAT16, a, poly_arg_none());
}

static PolyUOp *x86_expr_f16_where(PolyCtx *ctx, PolyUOp *a, PolyUOp *b, PolyUOp *range) {
  (void)b;
  (void)range;
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT16, poly_arg_float(0.0));
  PolyUOp *ten = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT16, poly_arg_float(10.0));
  PolyUOp *mask = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, a, zero, poly_arg_none());
  PolyUOp *plus = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT16, a, ten, poly_arg_none());
  PolyUOp *srcs[3] = {mask, a, plus};
  return poly_uop(ctx, POLY_OP_WHERE, POLY_FLOAT16, srcs, 3, poly_arg_none());
}

static PolyUOp *x86_expr_bf16_add(PolyCtx *ctx, PolyUOp *a, PolyUOp *b, PolyUOp *range) {
  (void)range;
  return poly_uop2(ctx, POLY_OP_ADD, POLY_BFLOAT16, a, b, poly_arg_none());
}

static PolyUOp *x86_make_cast_i32_f32(PolyCtx *ctx, int n) {
  PolyDType pi = poly_dtype_ptr(POLY_INT32, -1, POLY_ADDR_GLOBAL);
  PolyDType pf = poly_dtype_ptr(POLY_FLOAT32, -1, POLY_ADDR_GLOBAL);
  PolyUOp *p0 = poly_uop0(ctx, POLY_OP_PARAM, pi, poly_arg_int(0));
  PolyUOp *p1 = poly_uop0(ctx, POLY_OP_PARAM, pf, poly_arg_int(1));
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(n));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));
  PolyUOp *i0 = poly_uop2(ctx, POLY_OP_INDEX, pi, p0, range, poly_arg_none());
  PolyUOp *i1 = poly_uop2(ctx, POLY_OP_INDEX, pf, p1, range, poly_arg_none());
  PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, POLY_INT32, i0, poly_arg_none());
  PolyUOp *cast = poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, load, poly_arg_none());
  PolyUOp *st = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, i1, cast, poly_arg_none());
  PolyUOp *end_srcs[2] = {st, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_srcs, 2, poly_arg_none());
  return poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());
}

static PolyUOp *x86_make_shrink_load_width4(PolyCtx *ctx) {
  PolyDType pf = poly_dtype_ptr(POLY_FLOAT32, -1, POLY_ADDR_GLOBAL);
  PolyUOp *p0 = poly_uop0(ctx, POLY_OP_PARAM, pf, poly_arg_int(0));
  PolyUOp *p1 = poly_uop0(ctx, POLY_OP_PARAM, pf, poly_arg_int(1));
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *width = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(4));
  PolyUOp *shr_srcs[3] = {p0, zero, width};
  PolyUOp *shr =
      poly_uop(ctx, POLY_OP_SHRINK, poly_dtype_vec(POLY_FLOAT32, 4), shr_srcs, 3, poly_arg_none());
  PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, shr, poly_arg_none());
  PolyUOp *out_idx = poly_uop2(ctx, POLY_OP_INDEX, pf, p1, zero, poly_arg_none());
  PolyUOp *st = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out_idx, load, poly_arg_none());
  return poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, st, poly_arg_none());
}

static PolyUOp *x86_make_gated_load_false_uses_alt(PolyCtx *ctx) {
  PolyDType pf = poly_dtype_ptr(POLY_FLOAT32, -1, POLY_ADDR_GLOBAL);
  PolyUOp *p0 = poly_uop0(ctx, POLY_OP_PARAM, pf, poly_arg_int(0));
  PolyUOp *p1 = poly_uop0(ctx, POLY_OP_PARAM, pf, poly_arg_int(1));
  PolyUOp *bad = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(4096));
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *idx = poly_uop2(ctx, POLY_OP_INDEX, pf, p0, bad, poly_arg_none());
  PolyUOp *alt = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(7.5));
  PolyUOp *gate = poly_uop0(ctx, POLY_OP_CONST, POLY_BOOL, poly_arg_bool(false));
  PolyUOp *load_srcs[3] = {idx, alt, gate};
  PolyUOp *load = poly_uop(ctx, POLY_OP_LOAD, POLY_FLOAT32, load_srcs, 3, poly_arg_none());
  PolyUOp *out_idx = poly_uop2(ctx, POLY_OP_INDEX, pf, p1, zero, poly_arg_none());
  PolyUOp *st = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out_idx, load, poly_arg_none());
  return poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, st, poly_arg_none());
}

static PolyUOp *x86_make_gated_store_false_uses_scratch(PolyCtx *ctx) {
  PolyDType pf = poly_dtype_ptr(POLY_FLOAT32, -1, POLY_ADDR_GLOBAL);
  PolyUOp *p0 = poly_uop0(ctx, POLY_OP_PARAM, pf, poly_arg_int(0));
  PolyUOp *bad = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(4096));
  PolyUOp *idx = poly_uop2(ctx, POLY_OP_INDEX, pf, p0, bad, poly_arg_none());
  PolyUOp *val = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(42.0));
  PolyUOp *gate = poly_uop0(ctx, POLY_OP_CONST, POLY_BOOL, poly_arg_bool(false));
  PolyUOp *store_srcs[3] = {idx, val, gate};
  PolyUOp *st = poly_uop(ctx, POLY_OP_STORE, POLY_VOID, store_srcs, 3, poly_arg_none());
  return poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, st, poly_arg_none());
}

static PolyUOp *x86_make_cast_u32_i64(PolyCtx *ctx, int n) {
  PolyDType pu = poly_dtype_ptr(POLY_UINT32, -1, POLY_ADDR_GLOBAL);
  PolyDType pi64 = poly_dtype_ptr(POLY_INT64, -1, POLY_ADDR_GLOBAL);
  PolyUOp *p0 = poly_uop0(ctx, POLY_OP_PARAM, pu, poly_arg_int(0));
  PolyUOp *p1 = poly_uop0(ctx, POLY_OP_PARAM, pi64, poly_arg_int(1));
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(n));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));
  PolyUOp *i0 = poly_uop2(ctx, POLY_OP_INDEX, pu, p0, range, poly_arg_none());
  PolyUOp *i1 = poly_uop2(ctx, POLY_OP_INDEX, pi64, p1, range, poly_arg_none());
  PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, POLY_UINT32, i0, poly_arg_none());
  PolyUOp *cast = poly_uop1(ctx, POLY_OP_CAST, POLY_INT64, load, poly_arg_none());
  PolyUOp *st = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, i1, cast, poly_arg_none());
  PolyUOp *end_srcs[2] = {st, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_srcs, 2, poly_arg_none());
  return poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());
}

static PolyUOp *x86_make_cast_i64_i32(PolyCtx *ctx, int n) {
  PolyDType pi64 = poly_dtype_ptr(POLY_INT64, -1, POLY_ADDR_GLOBAL);
  PolyDType pi32 = poly_dtype_ptr(POLY_INT32, -1, POLY_ADDR_GLOBAL);
  PolyUOp *p0 = poly_uop0(ctx, POLY_OP_PARAM, pi64, poly_arg_int(0));
  PolyUOp *p1 = poly_uop0(ctx, POLY_OP_PARAM, pi32, poly_arg_int(1));
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(n));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));
  PolyUOp *i0 = poly_uop2(ctx, POLY_OP_INDEX, pi64, p0, range, poly_arg_none());
  PolyUOp *i1 = poly_uop2(ctx, POLY_OP_INDEX, pi32, p1, range, poly_arg_none());
  PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, POLY_INT64, i0, poly_arg_none());
  PolyUOp *cast = poly_uop1(ctx, POLY_OP_CAST, POLY_INT32, load, poly_arg_none());
  PolyUOp *st = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, i1, cast, poly_arg_none());
  PolyUOp *end_srcs[2] = {st, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_srcs, 2, poly_arg_none());
  return poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());
}

static PolyUOp *x86_make_cast_u8x4_i32x4(PolyCtx *ctx, int n_vec) {
  PolyDType pu8 = poly_dtype_ptr(POLY_UINT8, -1, POLY_ADDR_GLOBAL);
  PolyDType pi32 = poly_dtype_ptr(POLY_INT32, -1, POLY_ADDR_GLOBAL);
  PolyDType u8x4 = poly_dtype_vec(POLY_UINT8, 4);
  PolyDType i32x4 = poly_dtype_vec(POLY_INT32, 4);
  PolyUOp *p0 = poly_uop0(ctx, POLY_OP_PARAM, pu8, poly_arg_int(0));
  PolyUOp *p1 = poly_uop0(ctx, POLY_OP_PARAM, pi32, poly_arg_int(1));
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(n_vec));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));
  PolyUOp *four = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(4));
  PolyUOp *base = poly_uop2(ctx, POLY_OP_MUL, POLY_INT32, range, four, poly_arg_none());
  PolyUOp *i0 = poly_uop2(ctx, POLY_OP_INDEX, pu8, p0, base, poly_arg_none());
  PolyUOp *i1 = poly_uop2(ctx, POLY_OP_INDEX, pi32, p1, base, poly_arg_none());
  PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, u8x4, i0, poly_arg_none());
  PolyUOp *cast = poly_uop1(ctx, POLY_OP_CAST, i32x4, load, poly_arg_none());
  PolyUOp *st = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, i1, cast, poly_arg_none());
  PolyUOp *end_srcs[2] = {st, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_srcs, 2, poly_arg_none());
  return poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());
}

static PolyUOp *x86_make_vector_cast(
    PolyCtx *ctx,
    PolyDType src_scalar,
    int src_lanes,
    PolyDType dst_scalar,
    int dst_lanes,
    int n_vec
) {
  PolyDType ps = poly_dtype_ptr(src_scalar, -1, POLY_ADDR_GLOBAL);
  PolyDType pd = poly_dtype_ptr(dst_scalar, -1, POLY_ADDR_GLOBAL);
  PolyDType src_vec = poly_dtype_vec(src_scalar, src_lanes);
  PolyDType dst_vec = poly_dtype_vec(dst_scalar, dst_lanes);
  PolyUOp *p0 = poly_uop0(ctx, POLY_OP_PARAM, ps, poly_arg_int(0));
  PolyUOp *p1 = poly_uop0(ctx, POLY_OP_PARAM, pd, poly_arg_int(1));
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(n_vec));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));
  PolyUOp *src_stride = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(src_lanes));
  PolyUOp *dst_stride = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(dst_lanes));
  PolyUOp *src_base = poly_uop2(ctx, POLY_OP_MUL, POLY_INT32, range, src_stride, poly_arg_none());
  PolyUOp *dst_base = poly_uop2(ctx, POLY_OP_MUL, POLY_INT32, range, dst_stride, poly_arg_none());
  PolyUOp *i0 = poly_uop2(ctx, POLY_OP_INDEX, ps, p0, src_base, poly_arg_none());
  PolyUOp *i1 = poly_uop2(ctx, POLY_OP_INDEX, pd, p1, dst_base, poly_arg_none());
  PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, src_vec, i0, poly_arg_none());
  PolyUOp *cast = poly_uop1(ctx, POLY_OP_CAST, dst_vec, load, poly_arg_none());
  PolyUOp *st = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, i1, cast, poly_arg_none());
  PolyUOp *end_srcs[2] = {st, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_srcs, 2, poly_arg_none());
  return poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());
}

static PolyUOp *x86_make_i32x4_addsub(PolyCtx *ctx, int n_vec) {
  PolyDType pi = poly_dtype_ptr(POLY_INT32, -1, POLY_ADDR_GLOBAL);
  PolyDType i32x4 = poly_dtype_vec(POLY_INT32, 4);
  PolyUOp *p0 = poly_uop0(ctx, POLY_OP_PARAM, pi, poly_arg_int(0));
  PolyUOp *p1 = poly_uop0(ctx, POLY_OP_PARAM, pi, poly_arg_int(1));
  PolyUOp *p2 = poly_uop0(ctx, POLY_OP_PARAM, pi, poly_arg_int(2));
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(n_vec));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));
  PolyUOp *four = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(4));
  PolyUOp *base = poly_uop2(ctx, POLY_OP_MUL, POLY_INT32, range, four, poly_arg_none());
  PolyUOp *i0 = poly_uop2(ctx, POLY_OP_INDEX, pi, p0, base, poly_arg_none());
  PolyUOp *i1 = poly_uop2(ctx, POLY_OP_INDEX, pi, p1, base, poly_arg_none());
  PolyUOp *i2 = poly_uop2(ctx, POLY_OP_INDEX, pi, p2, base, poly_arg_none());
  PolyUOp *a = poly_uop1(ctx, POLY_OP_LOAD, i32x4, i0, poly_arg_none());
  PolyUOp *b = poly_uop1(ctx, POLY_OP_LOAD, i32x4, i1, poly_arg_none());
  PolyUOp *sum = poly_uop2(ctx, POLY_OP_ADD, i32x4, a, b, poly_arg_none());
  PolyUOp *diff = poly_uop2(ctx, POLY_OP_SUB, i32x4, sum, a, poly_arg_none());
  PolyUOp *st = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, i2, diff, poly_arg_none());
  PolyUOp *end_srcs[2] = {st, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_srcs, 2, poly_arg_none());
  return poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());
}

static PolyUOp *x86_make_int_vec_binary(
    PolyCtx *ctx,
    PolyDType scalar,
    int lanes,
    PolyOps op,
    int n_vec
) {
  PolyDType ptr = poly_dtype_ptr(scalar, -1, POLY_ADDR_GLOBAL);
  PolyDType vec = poly_dtype_vec(scalar, lanes);
  PolyUOp *p0 = poly_uop0(ctx, POLY_OP_PARAM, ptr, poly_arg_int(0));
  PolyUOp *p1 = poly_uop0(ctx, POLY_OP_PARAM, ptr, poly_arg_int(1));
  PolyUOp *p2 = poly_uop0(ctx, POLY_OP_PARAM, ptr, poly_arg_int(2));
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(n_vec));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));
  PolyUOp *stride = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(lanes));
  PolyUOp *base = poly_uop2(ctx, POLY_OP_MUL, POLY_INT32, range, stride, poly_arg_none());
  PolyUOp *i0 = poly_uop2(ctx, POLY_OP_INDEX, ptr, p0, base, poly_arg_none());
  PolyUOp *i1 = poly_uop2(ctx, POLY_OP_INDEX, ptr, p1, base, poly_arg_none());
  PolyUOp *i2 = poly_uop2(ctx, POLY_OP_INDEX, ptr, p2, base, poly_arg_none());
  PolyUOp *a = poly_uop1(ctx, POLY_OP_LOAD, vec, i0, poly_arg_none());
  PolyUOp *b = poly_uop1(ctx, POLY_OP_LOAD, vec, i1, poly_arg_none());
  PolyUOp *out = poly_uop2(ctx, op, vec, a, b, poly_arg_none());
  PolyUOp *st = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, i2, out, poly_arg_none());
  PolyUOp *end_srcs[2] = {st, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_srcs, 2, poly_arg_none());
  return poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());
}

static PolyUOp *x86_make_load_store(PolyCtx *ctx, PolyDType scalar, int lanes, int n_vec) {
  PolyDType ptr = poly_dtype_ptr(scalar, -1, POLY_ADDR_GLOBAL);
  PolyDType val_dt = lanes > 1 ? poly_dtype_vec(scalar, lanes) : scalar;
  PolyUOp *p0 = poly_uop0(ctx, POLY_OP_PARAM, ptr, poly_arg_int(0));
  PolyUOp *p1 = poly_uop0(ctx, POLY_OP_PARAM, ptr, poly_arg_int(1));
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(n_vec));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));
  PolyUOp *base = range;
  if (lanes > 1) {
    PolyUOp *stride = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(lanes));
    base = poly_uop2(ctx, POLY_OP_MUL, POLY_INT32, range, stride, poly_arg_none());
  }
  PolyUOp *i0 = poly_uop2(ctx, POLY_OP_INDEX, ptr, p0, base, poly_arg_none());
  PolyUOp *i1 = poly_uop2(ctx, POLY_OP_INDEX, ptr, p1, base, poly_arg_none());
  PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, val_dt, i0, poly_arg_none());
  PolyUOp *st = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, i1, load, poly_arg_none());
  PolyUOp *end_srcs[2] = {st, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_srcs, 2, poly_arg_none());
  return poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());
}

static PolyUOp *x86_make_i32_const_store(PolyCtx *ctx, int n, int value) {
  PolyDType pi = poly_dtype_ptr(POLY_INT32, -1, POLY_ADDR_GLOBAL);
  PolyUOp *p0 = poly_uop0(ctx, POLY_OP_PARAM, pi, poly_arg_int(0));
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(n));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));
  PolyUOp *idx = poly_uop2(ctx, POLY_OP_INDEX, pi, p0, range, poly_arg_none());
  PolyUOp *cv = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(value));
  PolyUOp *st = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx, cv, poly_arg_none());
  PolyUOp *end_srcs[2] = {st, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_srcs, 2, poly_arg_none());
  return poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());
}

static PolyUOp *x86_make_f32_compare_bool(PolyCtx *ctx, int n, PolyOps op) {
  PolyDType pf = poly_dtype_ptr(POLY_FLOAT32, -1, POLY_ADDR_GLOBAL);
  PolyDType pb = poly_dtype_ptr(POLY_BOOL, -1, POLY_ADDR_GLOBAL);
  PolyUOp *p0 = poly_uop0(ctx, POLY_OP_PARAM, pf, poly_arg_int(0));
  PolyUOp *p1 = poly_uop0(ctx, POLY_OP_PARAM, pf, poly_arg_int(1));
  PolyUOp *p2 = poly_uop0(ctx, POLY_OP_PARAM, pb, poly_arg_int(2));
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(n));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));
  PolyUOp *i0 = poly_uop2(ctx, POLY_OP_INDEX, pf, p0, range, poly_arg_none());
  PolyUOp *i1 = poly_uop2(ctx, POLY_OP_INDEX, pf, p1, range, poly_arg_none());
  PolyUOp *i2 = poly_uop2(ctx, POLY_OP_INDEX, pb, p2, range, poly_arg_none());
  PolyUOp *a = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, i0, poly_arg_none());
  PolyUOp *b = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, i1, poly_arg_none());
  PolyUOp *cmp = poly_uop2(ctx, op, POLY_BOOL, a, b, poly_arg_none());
  PolyUOp *st = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, i2, cmp, poly_arg_none());
  PolyUOp *end_srcs[2] = {st, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_srcs, 2, poly_arg_none());
  return poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());
}

static PolyUOp *x86_make_f64_compare_bool(PolyCtx *ctx, int n, PolyOps op) {
  PolyDType pd = poly_dtype_ptr(POLY_FLOAT64, -1, POLY_ADDR_GLOBAL);
  PolyDType pb = poly_dtype_ptr(POLY_BOOL, -1, POLY_ADDR_GLOBAL);
  PolyUOp *p0 = poly_uop0(ctx, POLY_OP_PARAM, pd, poly_arg_int(0));
  PolyUOp *p1 = poly_uop0(ctx, POLY_OP_PARAM, pd, poly_arg_int(1));
  PolyUOp *p2 = poly_uop0(ctx, POLY_OP_PARAM, pb, poly_arg_int(2));
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(n));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));
  PolyUOp *i0 = poly_uop2(ctx, POLY_OP_INDEX, pd, p0, range, poly_arg_none());
  PolyUOp *i1 = poly_uop2(ctx, POLY_OP_INDEX, pd, p1, range, poly_arg_none());
  PolyUOp *i2 = poly_uop2(ctx, POLY_OP_INDEX, pb, p2, range, poly_arg_none());
  PolyUOp *a = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT64, i0, poly_arg_none());
  PolyUOp *b = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT64, i1, poly_arg_none());
  PolyUOp *cmp = poly_uop2(ctx, op, POLY_BOOL, a, b, poly_arg_none());
  PolyUOp *st = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, i2, cmp, poly_arg_none());
  PolyUOp *end_srcs[2] = {st, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_srcs, 2, poly_arg_none());
  return poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());
}

static PolyUOp *x86_make_negative_step2(PolyCtx *ctx) {
  PolyDType pf = poly_dtype_ptr(POLY_FLOAT32, -1, POLY_ADDR_GLOBAL);
  PolyUOp *p0 = poly_uop0(ctx, POLY_OP_PARAM, pf, poly_arg_int(0));
  PolyUOp *p1 = poly_uop0(ctx, POLY_OP_PARAM, pf, poly_arg_int(1));
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(3));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));
  PolyUOp *neg_two = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(-2));
  PolyUOp *five = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(5));
  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, POLY_INT32, range, neg_two, poly_arg_none());
  PolyUOp *src_idx = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, mul, five, poly_arg_none());
  PolyUOp *load_idx = poly_uop2(ctx, POLY_OP_INDEX, pf, p0, src_idx, poly_arg_none());
  PolyUOp *store_idx = poly_uop2(ctx, POLY_OP_INDEX, pf, p1, range, poly_arg_none());
  PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, load_idx, poly_arg_none());
  PolyUOp *st = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, store_idx, load, poly_arg_none());
  PolyUOp *end_srcs[2] = {st, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_srcs, 2, poly_arg_none());
  return poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());
}

static PolyUOp *x86_make_f32_broadcast4(PolyCtx *ctx) {
  PolyDType pf = poly_dtype_ptr(POLY_FLOAT32, -1, POLY_ADDR_GLOBAL);
  PolyUOp *pin = poly_uop0(ctx, POLY_OP_PARAM, pf, poly_arg_int(0));
  PolyUOp *pout = poly_uop0(ctx, POLY_OP_PARAM, pf, poly_arg_int(1));
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *iin = poly_uop2(ctx, POLY_OP_INDEX, pf, pin, zero, poly_arg_none());
  PolyUOp *val = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, iin, poly_arg_none());
  PolyUOp *srcs[4] = {val, val, val, val};
  PolyUOp *vec =
      poly_uop(ctx, POLY_OP_STACK, poly_dtype_vec(POLY_FLOAT32, 4), srcs, 4, poly_arg_none());
  PolyUOp *iout = poly_uop2(ctx, POLY_OP_INDEX, pf, pout, zero, poly_arg_none());
  PolyUOp *st = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, iout, vec, poly_arg_none());
  return poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, st, poly_arg_none());
}

static PolyUOp *x86_make_f16_broadcast4(PolyCtx *ctx) {
  PolyDType pf = poly_dtype_ptr(POLY_FLOAT16, -1, POLY_ADDR_GLOBAL);
  PolyUOp *pin = poly_uop0(ctx, POLY_OP_PARAM, pf, poly_arg_int(0));
  PolyUOp *pout = poly_uop0(ctx, POLY_OP_PARAM, pf, poly_arg_int(1));
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *iin = poly_uop2(ctx, POLY_OP_INDEX, pf, pin, zero, poly_arg_none());
  PolyUOp *val = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT16, iin, poly_arg_none());
  PolyUOp *srcs[4] = {val, val, val, val};
  PolyUOp *vec =
      poly_uop(ctx, POLY_OP_STACK, poly_dtype_vec(POLY_FLOAT16, 4), srcs, 4, poly_arg_none());
  PolyUOp *iout = poly_uop2(ctx, POLY_OP_INDEX, pf, pout, zero, poly_arg_none());
  PolyUOp *st = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, iout, vec, poly_arg_none());
  return poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, st, poly_arg_none());
}

static PolyUOp *x86_make_i32_broadcast4(PolyCtx *ctx) {
  PolyDType pi = poly_dtype_ptr(POLY_INT32, -1, POLY_ADDR_GLOBAL);
  PolyUOp *pin = poly_uop0(ctx, POLY_OP_PARAM, pi, poly_arg_int(0));
  PolyUOp *pout = poly_uop0(ctx, POLY_OP_PARAM, pi, poly_arg_int(1));
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *iin = poly_uop2(ctx, POLY_OP_INDEX, pi, pin, zero, poly_arg_none());
  PolyUOp *val = poly_uop1(ctx, POLY_OP_LOAD, POLY_INT32, iin, poly_arg_none());
  PolyUOp *srcs[4] = {val, val, val, val};
  PolyUOp *vec =
      poly_uop(ctx, POLY_OP_STACK, poly_dtype_vec(POLY_INT32, 4), srcs, 4, poly_arg_none());
  PolyUOp *iout = poly_uop2(ctx, POLY_OP_INDEX, pi, pout, zero, poly_arg_none());
  PolyUOp *st = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, iout, vec, poly_arg_none());
  return poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, st, poly_arg_none());
}

static PolyUOp *x86_make_f32_shuffle4(PolyCtx *ctx) {
  PolyDType pf = poly_dtype_ptr(POLY_FLOAT32, -1, POLY_ADDR_GLOBAL);
  PolyDType v4 = poly_dtype_vec(POLY_FLOAT32, 4);
  PolyUOp *pa = poly_uop0(ctx, POLY_OP_PARAM, pf, poly_arg_int(0));
  PolyUOp *pb = poly_uop0(ctx, POLY_OP_PARAM, pf, poly_arg_int(1));
  PolyUOp *po = poly_uop0(ctx, POLY_OP_PARAM, pf, poly_arg_int(2));
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *la = poly_uop1(
      ctx, POLY_OP_LOAD, v4, poly_uop2(ctx, POLY_OP_INDEX, pf, pa, zero, poly_arg_none()),
      poly_arg_none()
  );
  PolyUOp *lb = poly_uop1(
      ctx, POLY_OP_LOAD, v4, poly_uop2(ctx, POLY_OP_INDEX, pf, pb, zero, poly_arg_none()),
      poly_arg_none()
  );
  PolyUOp *i0 = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *i1 = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(1));
  PolyUOp *srcs[4] = {
      poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, la, i0, poly_arg_none()),
      poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, la, i1, poly_arg_none()),
      poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, lb, i0, poly_arg_none()),
      poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, lb, i1, poly_arg_none()),
  };
  PolyUOp *vec = poly_uop(ctx, POLY_OP_STACK, v4, srcs, 4, poly_arg_none());
  PolyUOp *st = poly_uop2(
      ctx, POLY_OP_STORE, POLY_VOID, poly_uop2(ctx, POLY_OP_INDEX, pf, po, zero, poly_arg_none()),
      vec, poly_arg_none()
  );
  return poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, st, poly_arg_none());
}

static PolyUOp *x86_make_f64_shuffle2(PolyCtx *ctx) {
  PolyDType pd = poly_dtype_ptr(POLY_FLOAT64, -1, POLY_ADDR_GLOBAL);
  PolyDType v2 = poly_dtype_vec(POLY_FLOAT64, 2);
  PolyUOp *pa = poly_uop0(ctx, POLY_OP_PARAM, pd, poly_arg_int(0));
  PolyUOp *pb = poly_uop0(ctx, POLY_OP_PARAM, pd, poly_arg_int(1));
  PolyUOp *po = poly_uop0(ctx, POLY_OP_PARAM, pd, poly_arg_int(2));
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *la = poly_uop1(
      ctx, POLY_OP_LOAD, v2, poly_uop2(ctx, POLY_OP_INDEX, pd, pa, zero, poly_arg_none()),
      poly_arg_none()
  );
  PolyUOp *lb = poly_uop1(
      ctx, POLY_OP_LOAD, v2, poly_uop2(ctx, POLY_OP_INDEX, pd, pb, zero, poly_arg_none()),
      poly_arg_none()
  );
  PolyUOp *i0 = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *i1 = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(1));
  PolyUOp *srcs[2] = {
      poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT64, la, i1, poly_arg_none()),
      poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT64, lb, i0, poly_arg_none()),
  };
  PolyUOp *vec = poly_uop(ctx, POLY_OP_STACK, v2, srcs, 2, poly_arg_none());
  PolyUOp *st = poly_uop2(
      ctx, POLY_OP_STORE, POLY_VOID, poly_uop2(ctx, POLY_OP_INDEX, pd, po, zero, poly_arg_none()),
      vec, poly_arg_none()
  );
  return poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, st, poly_arg_none());
}

static PolyUOp *x86_make_f32_insert4(PolyCtx *ctx) {
  PolyDType pf = poly_dtype_ptr(POLY_FLOAT32, -1, POLY_ADDR_GLOBAL);
  PolyDType v4 = poly_dtype_vec(POLY_FLOAT32, 4);
  PolyUOp *pa = poly_uop0(ctx, POLY_OP_PARAM, pf, poly_arg_int(0));
  PolyUOp *pb = poly_uop0(ctx, POLY_OP_PARAM, pf, poly_arg_int(1));
  PolyUOp *pc = poly_uop0(ctx, POLY_OP_PARAM, pf, poly_arg_int(2));
  PolyUOp *pd = poly_uop0(ctx, POLY_OP_PARAM, pf, poly_arg_int(3));
  PolyUOp *po = poly_uop0(ctx, POLY_OP_PARAM, pf, poly_arg_int(4));
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *la = poly_uop1(
      ctx, POLY_OP_LOAD, v4, poly_uop2(ctx, POLY_OP_INDEX, pf, pa, zero, poly_arg_none()),
      poly_arg_none()
  );
  PolyUOp *lb = poly_uop1(
      ctx, POLY_OP_LOAD, v4, poly_uop2(ctx, POLY_OP_INDEX, pf, pb, zero, poly_arg_none()),
      poly_arg_none()
  );
  PolyUOp *lc = poly_uop1(
      ctx, POLY_OP_LOAD, v4, poly_uop2(ctx, POLY_OP_INDEX, pf, pc, zero, poly_arg_none()),
      poly_arg_none()
  );
  PolyUOp *ld = poly_uop1(
      ctx, POLY_OP_LOAD, v4, poly_uop2(ctx, POLY_OP_INDEX, pf, pd, zero, poly_arg_none()),
      poly_arg_none()
  );
  PolyUOp *i0 = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *i1 = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(1));
  PolyUOp *i2 = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(2));
  PolyUOp *i3 = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(3));
  PolyUOp *srcs[4] = {
      poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, la, i0, poly_arg_none()),
      poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, lb, i1, poly_arg_none()),
      poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, lc, i2, poly_arg_none()),
      poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, ld, i3, poly_arg_none()),
  };
  PolyUOp *vec = poly_uop(ctx, POLY_OP_STACK, v4, srcs, 4, poly_arg_none());
  PolyUOp *st = poly_uop2(
      ctx, POLY_OP_STORE, POLY_VOID, poly_uop2(ctx, POLY_OP_INDEX, pf, po, zero, poly_arg_none()),
      vec, poly_arg_none()
  );
  return poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, st, poly_arg_none());
}

static PolyUOp *x86_make_index_extract(PolyCtx *ctx, PolyDType scalar, int count, int lane) {
  PolyDType ptr = poly_dtype_ptr(scalar, -1, POLY_ADDR_GLOBAL);
  PolyDType vec = poly_dtype_vec(scalar, count);
  PolyUOp *pin = poly_uop0(ctx, POLY_OP_PARAM, ptr, poly_arg_int(0));
  PolyUOp *pout = poly_uop0(ctx, POLY_OP_PARAM, ptr, poly_arg_int(1));
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *load = poly_uop1(
      ctx, POLY_OP_LOAD, vec, poly_uop2(ctx, POLY_OP_INDEX, ptr, pin, zero, poly_arg_none()),
      poly_arg_none()
  );
  PolyUOp *idx = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(lane));
  PolyUOp *item = poly_uop2(ctx, POLY_OP_INDEX, scalar, load, idx, poly_arg_none());
  PolyUOp *st = poly_uop2(
      ctx, POLY_OP_STORE, POLY_VOID,
      poly_uop2(ctx, POLY_OP_INDEX, ptr, pout, zero, poly_arg_none()), item, poly_arg_none()
  );
  return poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, st, poly_arg_none());
}

static PolyUOp *x86_make_int_stack_from_scalar_lanes(PolyCtx *ctx, PolyDType scalar, int count) {
  PolyDType ptr = poly_dtype_ptr(scalar, -1, POLY_ADDR_GLOBAL);
  PolyDType vec = poly_dtype_vec(scalar, count);
  PolyUOp *pin = poly_uop0(ctx, POLY_OP_PARAM, ptr, poly_arg_int(0));
  PolyUOp *pout = poly_uop0(ctx, POLY_OP_PARAM, ptr, poly_arg_int(1));
  PolyUOp *srcs[16];
  for (int i = 0; i < count; i++) {
    PolyUOp *idx = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(i));
    PolyUOp *addr = poly_uop2(ctx, POLY_OP_INDEX, ptr, pin, idx, poly_arg_none());
    srcs[i] = poly_uop1(ctx, POLY_OP_LOAD, scalar, addr, poly_arg_none());
  }
  PolyUOp *stack = poly_uop(ctx, POLY_OP_STACK, vec, srcs, count, poly_arg_none());
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *st = poly_uop2(
      ctx, POLY_OP_STORE, POLY_VOID,
      poly_uop2(ctx, POLY_OP_INDEX, ptr, pout, zero, poly_arg_none()), stack, poly_arg_none()
  );
  return poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, st, poly_arg_none());
}

static int x86_run_direct(PolyCtx *ctx, PolyUOp *sink, void **args, int n_args) {
  int n_lin = 0;
  PolyUOp **lin = poly_linearize_x86(ctx, sink, &n_lin);
  if (!lin) return -1;
  int code_size = 0;
  uint8_t *code = poly_render_x86(lin, n_lin, &code_size);
  free(lin);
  if (!code) return -2;
  PolyX86Program *prog = poly_compile_x86(code, code_size);
  free(code);
  if (!prog) return -3;
  int rc = poly_x86_program_call(prog, args, n_args);
  poly_x86_program_destroy(prog);
  return rc;
}

static int x86_run_rewritten_direct(PolyCtx *ctx, PolyUOp *sink, void **args, int n_args) {
  int n_lin = 0;
  PolyUOp **lin = poly_linearize_x86_rewritten(ctx, sink, &n_lin);
  if (!lin) return -1;
  int code_size = 0;
  uint8_t *code = poly_render_x86(lin, n_lin, &code_size);
  free(lin);
  if (!code) return -2;
  PolyX86Program *prog = poly_compile_x86(code, code_size);
  free(code);
  if (!prog) return -3;
  int rc = poly_x86_program_call(prog, args, n_args);
  poly_x86_program_destroy(prog);
  return rc;
}

static uint64_t x86_topology_signature(PolyCtx *ctx, PolyUOp *root, int *n_out) {
  int n = 0;
  PolyUOp **topo = poly_toposort_alloc(ctx, root, &n);
  if (!topo) return 0;
  uint64_t signature = UINT64_C(1469598103934665603);
#define MIX(v)                                                                                      \
  do {                                                                                              \
    signature ^= (uint64_t)(v);                                                                     \
    signature *= UINT64_C(1099511628211);                                                           \
  } while (0)
  MIX(n);
  for (int i = 0; i < n; i++) {
    PolyUOp *u = topo[i];
    MIX(u->op);
    MIX((uint8_t)u->dtype.priority);
    MIX(u->dtype.bitsize);
    MIX((uint8_t)u->dtype.fmt);
    MIX(u->dtype.count);
    MIX(u->dtype.is_ptr);
    MIX(u->dtype.addrspace);
    MIX(u->dtype.vcount);
    MIX((uint64_t)u->dtype.ptr_size);
    MIX(poly_arg_hash(u->arg));
    MIX(u->n_src);
    for (int j = 0; j < u->n_src; j++) {
      int src_id = -1;
      for (int k = 0; k < n; k++)
        if (topo[k] == u->src[j]) {
          src_id = k;
          break;
        }
      MIX((uint32_t)src_id);
    }
  }
#undef MIX
  free(topo);
  if (n_out) *n_out = n;
  return signature;
}

static PolyUOp *x86_make_tagged_if_sink(
    PolyCtx *ctx,
    PolyDType dtype,
    PolyOps cmp_op,
    const char *label
) {
  PolyUOp *a = poly_uop0(ctx, POLY_OP_PARAM, dtype, poly_arg_int(0));
  PolyUOp *b = poly_uop0(ctx, POLY_OP_PARAM, dtype, poly_arg_int(1));
  PolyUOp *mask = poly_uop2(ctx, cmp_op, POLY_BOOL, a, b, poly_arg_none());
  PolyUOp *ifu = poly_uop_tagged_arg(
      ctx, POLY_OP_IF, POLY_VOID, &mask, 1, poly_arg_none(), 0, poly_arg_str(label)
  );
  return poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, ifu, poly_arg_none());
}

static int x86_find_tagged_ins_arg(PolyUOp **lin, int n, const char *label) {
  for (int i = 0; i < n; i++) {
    PolyUOp *u = lin[i];
    if (!u || u->op != POLY_OP_INS || u->arg.kind != POLY_ARG_INT) continue;
    if (u->tag_arg.kind == POLY_ARG_STRING && u->tag_arg.str && strcmp(u->tag_arg.str, label) == 0)
      return (int)u->arg.i;
  }
  return 0;
}

static int x86_linear_contains_op(PolyUOp **lin, int n, PolyOps op) {
  for (int i = 0; i < n; i++)
    if (lin[i] && lin[i]->op == op) return 1;
  return 0;
}

static int x86_linear_count_const_arg(PolyUOp **lin, int n, PolyArgKind kind) {
  int count = 0;
  for (int i = 0; i < n; i++)
    if (lin[i] && lin[i]->op == POLY_OP_CONST && lin[i]->arg.kind == kind) count++;
  return count;
}

static int x86_rewritten_linear_count_ins_arg(PolyCtx *ctx, PolyUOp *sink, int arg) {
  int n_lin = 0;
  PolyUOp **lin = poly_linearize_x86_rewritten(ctx, sink, &n_lin);
  if (!lin) return 0;
  int count = 0;
  for (int i = 0; i < n_lin; i++) {
    PolyUOp *u = lin[i];
    if (u && u->op == POLY_OP_INS && u->arg.kind == POLY_ARG_INT && (int)u->arg.i == arg) count++;
  }
  free(lin);
  return count;
}

static int x86_linear_all_range_bounds_are_consts(PolyUOp **lin, int n) {
  for (int i = 0; i < n; i++) {
    PolyUOp *u = lin[i];
    if (!u || u->op != POLY_OP_RANGE) continue;
    if (u->n_src < 1 || !u->src[0] || u->src[0]->op != POLY_OP_CONST) return 0;
    if (u->src[0]->tag_arg.kind != POLY_ARG_BOOL || !u->src[0]->tag_arg.b) return 0;
    if (u->tag == 0 && u->tag_arg.kind != POLY_ARG_INT_TUPLE) return 0;
  }
  return 1;
}

static int x86_hex_nibble(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

static int x86_rewritten_code_count_hex(PolyCtx *ctx, PolyUOp *sink, const char *hex) {
  int n_lin = 0;
  PolyUOp **lin = poly_linearize_x86_rewritten(ctx, sink, &n_lin);
  if (!lin) return 0;
  int code_size = 0;
  uint8_t *code = poly_render_x86(lin, n_lin, &code_size);
  free(lin);
  if (!code) return 0;
  int n_pat = 0;
  while (hex[n_pat * 2])
    n_pat++;
  uint8_t *pat = malloc((size_t)n_pat);
  if (!pat) {
    free(code);
    return 0;
  }
  for (int i = 0; i < n_pat; i++) {
    int hi = x86_hex_nibble(hex[i * 2]);
    int lo = x86_hex_nibble(hex[i * 2 + 1]);
    if (hi < 0 || lo < 0) {
      free(pat);
      free(code);
      return 0;
    }
    pat[i] = (uint8_t)((hi << 4) | lo);
  }
  int found = 0;
  for (int i = 0; i <= code_size - n_pat; i++) {
    if (memcmp(code + i, pat, (size_t)n_pat) == 0) {
      found++;
    }
  }
  free(pat);
  free(code);
  return found;
}

static int x86_rewritten_code_contains_hex(PolyCtx *ctx, PolyUOp *sink, const char *hex) {
  return x86_rewritten_code_count_hex(ctx, sink, hex) > 0;
}

enum {
  TX86_REG_RAX = 0,
  TX86_REG_RCX = 1,
  TX86_REG_RDX = 2,
  TX86_REG_RBX = 3,
  TX86_REG_RSP = 4,
  TX86_REG_RBP = 5,
  TX86_REG_RSI = 6,
  TX86_REG_RDI = 7,
  TX86_REG_R8 = 8,
  TX86_REG_R12 = 12,
};

enum {
  TX86_REG_CLASS_WGPR = 2,
  TX86_REG_CLASS_XMM = 3,
};

enum {
  TX86_OP_DEFINE = 3,
  TX86_OP_MOV = 5,
  TX86_OP_MOVm = 6,
  TX86_OP_MOVi = 7,
  TX86_OP_VMOVSS = 9,
  TX86_OP_VMOVSSm = 12,
  TX86_OP_VMOVSDm = 13,
  TX86_OP_VMOVUPSm = 14,
  TX86_OP_MOVSX = 16,
  TX86_OP_CMOVE = 67,
  TX86_OP_VBLENDVPS = 71,
  TX86_OP_VPEXTRW = 84,
  TX86_OP_VPEXTRD = 85,
  TX86_OP_VPINSRW = 88,
  TX86_OP_IMULi = 103,
  TX86_OP_CMP = 116,
  TX86_OP_VADDSS = 126,
  TX86_OP_VADDPS = 128,
  TX86_OP_VPADDB = 150,
  TX86_OP_VPSUBQ = 157,
  TX86_OP_VPMULLW = 158,
  TX86_OP_VPMULLD = 159,
  TX86_OP_VPAND = 160,
  TX86_OP_VPOR = 161,
  TX86_OP_VPXOR = 162,
  TX86_OP_VPSLLVD = 163,
  TX86_OP_VPSLLVQ = 164,
  TX86_OP_VPSRLVD = 165,
  TX86_OP_VPSRLVQ = 166,
  TX86_OP_VPSRAVD = 167,
};

#define TX86_TAG_REAL 0x40000000
#define TX86_TAG_CLASS_SHIFT 24
#define TX86_TAG_ID_MASK 0xFFFFFF

static int32_t tx86_tag_real(int cls, int reg) {
  return TX86_TAG_REAL | ((int32_t)cls << TX86_TAG_CLASS_SHIFT) | (reg & TX86_TAG_ID_MASK);
}

static int tx86_count_stack_memory_op(PolyUOp **lin, int n, int op, int memory_src, int width) {
  int count = 0;
  int32_t rsp = tx86_tag_real(TX86_REG_CLASS_WGPR, TX86_REG_RSP);
  for (int i = 0; i < n; i++) {
    PolyUOp *u = lin[i];
    if (!u || u->op != POLY_OP_INS || u->arg.kind != POLY_ARG_INT || u->arg.i != op ||
        memory_src < 0 || memory_src + 3 >= u->n_src)
      continue;
    PolyUOp *base = u->src[memory_src];
    PolyUOp *size = u->src[memory_src + 3];
    int32_t base_reg =
        base && base->tag_arg.kind == POLY_ARG_INT_TUPLE && base->tag_arg.int_tuple.n > 0
            ? (int32_t)base->tag_arg.int_tuple.vals[0]
        : base ? base->tag
               : 0;
    if (base_reg == rsp && size && size->op == POLY_OP_CONST && size->arg.kind == POLY_ARG_INT &&
        size->arg.i == width)
      count++;
  }
  return count;
}

static PolyArg tx86_arg_int_tuple(int64_t *vals, int n) {
  PolyArg a = {.kind = POLY_ARG_INT_TUPLE};
  a.int_tuple.vals = vals;
  a.int_tuple.n = n;
  return a;
}

static PolyUOp *tx86_const_i(PolyCtx *ctx, PolyDType dt, int64_t v) {
  return poly_uop_tagged_arg(
      ctx, POLY_OP_CONST, dt, NULL, 0, poly_arg_int(v), 0, poly_arg_bool(true)
  );
}

static PolyUOp *tx86_noop(PolyCtx *ctx) {
  return poly_uop0(ctx, POLY_OP_NOOP, POLY_VOID, poly_arg_none());
}

static int tx86_class_for_dtype(PolyDType dt) {
  if (dt.is_ptr || dt.count > 1) return dt.count > 1 ? TX86_REG_CLASS_XMM : TX86_REG_CLASS_WGPR;
  PolyDType s = poly_dtype_scalar(dt);
  return (poly_dtype_is_float(s) && !poly_dtype_is_bool(s)) ? TX86_REG_CLASS_XMM
                                                            : TX86_REG_CLASS_WGPR;
}

static PolyUOp *tx86_def_reg(PolyCtx *ctx, PolyDType dt, int reg) {
  int32_t tag = tx86_tag_real(tx86_class_for_dtype(dt), reg);
  int64_t vals[1] = {tag};
  return poly_uop_tagged_arg(
      ctx, POLY_OP_INS, dt, NULL, 0, poly_arg_int(TX86_OP_DEFINE), tag, tx86_arg_int_tuple(vals, 1)
  );
}

static PolyUOp *tx86_ins(
    PolyCtx *ctx,
    int op,
    PolyDType dt,
    PolyUOp **srcs,
    int n_src,
    int def_reg
) {
  int64_t vals[1];
  PolyArg tag_arg = poly_arg_none();
  int32_t tag = 0;
  if (def_reg >= 0) {
    tag = tx86_tag_real(tx86_class_for_dtype(dt), def_reg);
    vals[0] = tag;
    tag_arg = tx86_arg_int_tuple(vals, 1);
  }
  return poly_uop_tagged_arg(ctx, POLY_OP_INS, dt, srcs, n_src, poly_arg_int(op), tag, tag_arg);
}

static PolyUOp *tx86_ins_nodef(PolyCtx *ctx, int op, PolyDType dt, PolyUOp **srcs, int n_src) {
  return tx86_ins(ctx, op, dt, srcs, n_src, -1);
}

static int tx86_hex_to_bytes(const char *hex, uint8_t *out, int cap) {
  int n = 0;
  int hi = -1;
  for (const char *p = hex; *p; p++) {
    int v = x86_hex_nibble(*p);
    if (v < 0) continue;
    if (hi < 0) {
      hi = v;
    } else {
      if (n >= cap) return -1;
      out[n++] = (uint8_t)((hi << 4) | v);
      hi = -1;
    }
  }
  return hi < 0 ? n : -1;
}

static void tx86_assert_render_hex(
    PolyUOp *u,
    const char *expected_hex,
    int *_passed,
    int *_failed
) {
  int code_size = 0;
  PolyUOp *lin[1] = {u};
  uint8_t *code = poly_render_x86(lin, 1, &code_size);
  ASSERT_NOT_NULL(code);
  uint8_t expected[64];
  int n_expected = tx86_hex_to_bytes(expected_hex, expected, (int)sizeof(expected));
  ASSERT_TRUE(n_expected >= 0);
  if (code_size != n_expected || memcmp(code, expected, (size_t)n_expected) != 0) {
    fprintf(stderr, "    expected:");
    for (int i = 0; i < n_expected; i++)
      fprintf(stderr, " %02x", expected[i]);
    fprintf(stderr, "\n    actual:  ");
    for (int i = 0; i < code_size; i++)
      fprintf(stderr, " %02x", code[i]);
    fprintf(stderr, "\n");
    free(code);
    FAIL("%s", "x86 encoding mismatch");
  }
  free(code);
}

TEST_BACKEND(x86, direct_encoder_matches_tinygrad_addressing_and_legacy_bytes) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *load_base_srcs[] = {
      tx86_def_reg(ctx, POLY_UINT64, TX86_REG_RDI),
      tx86_noop(ctx),
      tx86_const_i(ctx, POLY_INT8, 0),
      tx86_const_i(ctx, POLY_UINT8, 4),
  };
  tx86_assert_render_hex(
      tx86_ins(ctx, TX86_OP_MOV, POLY_INT32, load_base_srcs, 4, TX86_REG_RDI), "8b3f", _passed,
      _failed
  );

  PolyUOp *load_rsp_srcs[] = {
      tx86_def_reg(ctx, POLY_UINT64, TX86_REG_RSP),
      tx86_noop(ctx),
      tx86_const_i(ctx, POLY_INT8, 0),
      tx86_const_i(ctx, POLY_UINT8, 4),
  };
  tx86_assert_render_hex(
      tx86_ins(ctx, TX86_OP_MOV, POLY_INT32, load_rsp_srcs, 4, TX86_REG_RSP), "8b2424", _passed,
      _failed
  );

  PolyUOp *load_rbp_srcs[] = {
      tx86_def_reg(ctx, POLY_UINT64, TX86_REG_RBP),
      tx86_noop(ctx),
      tx86_const_i(ctx, POLY_INT8, 0),
      tx86_const_i(ctx, POLY_UINT8, 4),
  };
  tx86_assert_render_hex(
      tx86_ins(ctx, TX86_OP_MOV, POLY_INT32, load_rbp_srcs, 4, TX86_REG_RBP), "8b6d00", _passed,
      _failed
  );

  PolyUOp *load_base_index_srcs[] = {
      tx86_def_reg(ctx, POLY_UINT64, TX86_REG_RAX),
      tx86_def_reg(ctx, POLY_INT32, TX86_REG_RDX),
      tx86_const_i(ctx, POLY_INT8, 0),
      tx86_const_i(ctx, POLY_UINT8, 4),
  };
  tx86_assert_render_hex(
      tx86_ins(ctx, TX86_OP_MOV, POLY_INT32, load_base_index_srcs, 4, TX86_REG_RAX), "8b0490",
      _passed, _failed
  );

  PolyUOp *load_rsp_index_srcs[] = {
      tx86_def_reg(ctx, POLY_UINT64, TX86_REG_RAX),
      tx86_def_reg(ctx, POLY_INT32, TX86_REG_RSP),
      tx86_const_i(ctx, POLY_INT8, 0),
      tx86_const_i(ctx, POLY_UINT8, 4),
  };
  tx86_assert_render_hex(
      tx86_ins(ctx, TX86_OP_MOV, POLY_INT32, load_rsp_index_srcs, 4, TX86_REG_RAX), "8b00", _passed,
      _failed
  );

  PolyUOp *load_r12_index_srcs[] = {
      tx86_def_reg(ctx, POLY_UINT64, TX86_REG_RAX),
      tx86_def_reg(ctx, POLY_INT32, TX86_REG_R12),
      tx86_const_i(ctx, POLY_INT8, 0),
      tx86_const_i(ctx, POLY_UINT8, 4),
  };
  tx86_assert_render_hex(
      tx86_ins(ctx, TX86_OP_MOV, POLY_INT32, load_r12_index_srcs, 4, TX86_REG_RAX), "428b04a0",
      _passed, _failed
  );

  PolyUOp *load_disp8_srcs[] = {
      tx86_def_reg(ctx, POLY_UINT64, TX86_REG_RDI),
      tx86_def_reg(ctx, POLY_INT32, TX86_REG_RSI),
      tx86_const_i(ctx, POLY_INT8, 10),
      tx86_const_i(ctx, POLY_UINT8, 4),
  };
  tx86_assert_render_hex(
      tx86_ins(ctx, TX86_OP_MOV, POLY_INT32, load_disp8_srcs, 4, TX86_REG_RDI), "8b7cb70a", _passed,
      _failed
  );

  PolyUOp *load_disp32_srcs[] = {
      tx86_def_reg(ctx, POLY_UINT64, TX86_REG_RDI),
      tx86_def_reg(ctx, POLY_INT32, TX86_REG_RSI),
      tx86_const_i(ctx, POLY_INT32, 10000),
      tx86_const_i(ctx, POLY_UINT8, 4),
  };
  tx86_assert_render_hex(
      tx86_ins(ctx, TX86_OP_MOV, POLY_INT32, load_disp32_srcs, 4, TX86_REG_RDI), "8bbcb710270000",
      _passed, _failed
  );

  PolyUOp *movsx_i8_srcs[] = {tx86_def_reg(ctx, POLY_INT8, TX86_REG_RDX)};
  tx86_assert_render_hex(
      tx86_ins(ctx, TX86_OP_MOVSX, POLY_INT32, movsx_i8_srcs, 1, TX86_REG_RAX), "0fbec2", _passed,
      _failed
  );
  PolyUOp *movsx_dil_srcs[] = {tx86_def_reg(ctx, POLY_INT8, TX86_REG_RDI)};
  tx86_assert_render_hex(
      tx86_ins(ctx, TX86_OP_MOVSX, POLY_INT32, movsx_dil_srcs, 1, TX86_REG_RAX), "400fbec7",
      _passed, _failed
  );
  tx86_assert_render_hex(
      tx86_ins(ctx, TX86_OP_MOVSX, POLY_INT16, movsx_i8_srcs, 1, TX86_REG_RAX), "660fbec2", _passed,
      _failed
  );
  tx86_assert_render_hex(
      tx86_ins(ctx, TX86_OP_MOVSX, POLY_INT64, movsx_i8_srcs, 1, TX86_REG_RAX), "480fbec2", _passed,
      _failed
  );

  PolyUOp *movi_addr[] = {
      tx86_def_reg(ctx, POLY_UINT64, TX86_REG_RDI),
      tx86_def_reg(ctx, POLY_INT8, TX86_REG_RSI),
      tx86_const_i(ctx, POLY_INT8, 10),
      tx86_const_i(ctx, POLY_UINT8, 1),
      tx86_const_i(ctx, POLY_INT8, 10),
  };
  tx86_assert_render_hex(
      tx86_ins_nodef(ctx, TX86_OP_MOVi, POLY_VOID, movi_addr, 5), "40c644370a0a", _passed, _failed
  );

  PolyUOp *imul_addr[] = {
      tx86_def_reg(ctx, POLY_UINT64, TX86_REG_RDI),
      tx86_def_reg(ctx, POLY_INT32, TX86_REG_RSI),
      tx86_const_i(ctx, POLY_INT32, 10),
      tx86_const_i(ctx, POLY_UINT8, 4),
      tx86_const_i(ctx, POLY_INT32, 10),
  };
  tx86_assert_render_hex(
      tx86_ins(ctx, TX86_OP_IMULi, POLY_INT32, imul_addr, 5, TX86_REG_RDI),
      "69bcb70a0000000a000000", _passed, _failed
  );

  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, direct_encoder_matches_tinygrad_vex_and_cmove_bytes) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *xmm0 = tx86_def_reg(ctx, POLY_FLOAT32, TX86_REG_RAX);
  PolyUOp *xmm1 = tx86_def_reg(ctx, POLY_FLOAT32, TX86_REG_RCX);
  PolyUOp *xmm8 = tx86_def_reg(ctx, POLY_FLOAT32, TX86_REG_R8);
  PolyUOp *vadd_srcs[] = {xmm0, xmm1};
  tx86_assert_render_hex(
      tx86_ins(ctx, TX86_OP_VADDSS, POLY_FLOAT32, vadd_srcs, 2, TX86_REG_RAX), "c5fa58c1", _passed,
      _failed
  );
  PolyUOp *vadd_long_srcs[] = {xmm0, xmm8};
  tx86_assert_render_hex(
      tx86_ins(ctx, TX86_OP_VADDSS, POLY_FLOAT32, vadd_long_srcs, 2, TX86_REG_RAX), "c4c17a58c0",
      _passed, _failed
  );

  PolyDType f32x8 = poly_dtype_vec(POLY_FLOAT32, 8);
  PolyUOp *ymm0 = tx86_def_reg(ctx, f32x8, TX86_REG_RAX);
  PolyUOp *ymm1 = tx86_def_reg(ctx, f32x8, TX86_REG_RCX);
  PolyUOp *vaddps_srcs[] = {ymm0, ymm1};
  tx86_assert_render_hex(
      tx86_ins(ctx, TX86_OP_VADDPS, f32x8, vaddps_srcs, 2, TX86_REG_RAX), "c5fc58c1", _passed,
      _failed
  );

  PolyUOp *xmm2 = tx86_def_reg(ctx, POLY_FLOAT32, TX86_REG_RDX);
  PolyUOp *blend_srcs[] = {xmm0, xmm1, xmm2};
  tx86_assert_render_hex(
      tx86_ins(ctx, TX86_OP_VBLENDVPS, POLY_FLOAT32, blend_srcs, 3, TX86_REG_RAX), "c4e3794ac120",
      _passed, _failed
  );

  PolyUOp *extr_srcs[] = {
      tx86_def_reg(ctx, POLY_UINT64, TX86_REG_RDI),
      tx86_def_reg(ctx, POLY_INT32, TX86_REG_RSI),
      tx86_const_i(ctx, POLY_INT8, 10),
      tx86_const_i(ctx, POLY_UINT8, 4),
      xmm0,
      tx86_const_i(ctx, POLY_UINT8, 0),
  };
  tx86_assert_render_hex(
      tx86_ins_nodef(ctx, TX86_OP_VPEXTRD, POLY_VOID, extr_srcs, 6), "c4e3791644b70a00", _passed,
      _failed
  );

  PolyUOp *cmove_mem_srcs[] = {
      tx86_def_reg(ctx, POLY_UINT64, TX86_REG_RDI),
      tx86_def_reg(ctx, POLY_INT32, TX86_REG_RSI),
      tx86_const_i(ctx, POLY_INT8, 10),
      tx86_const_i(ctx, POLY_UINT8, 4),
  };
  tx86_assert_render_hex(
      tx86_ins(ctx, TX86_OP_CMOVE, POLY_INT32, cmove_mem_srcs, 4, TX86_REG_RAX), "0f4444b70a",
      _passed, _failed
  );

  PolyUOp *cmp = tx86_ins_nodef(ctx, TX86_OP_CMP, POLY_VOID, NULL, 0);
  PolyUOp *cmove_reg_srcs[] = {tx86_def_reg(ctx, POLY_INT32, TX86_REG_RAX), cmp};
  tx86_assert_render_hex(
      tx86_ins(ctx, TX86_OP_CMOVE, POLY_INT32, cmove_reg_srcs, 2, TX86_REG_RDX), "0f44d0", _passed,
      _failed
  );

  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, feature_stamp_is_stable_for_runtime_cache_key) {
  uint32_t a = poly_x86_feature_stamp();
  uint32_t b = poly_x86_feature_stamp();
  ASSERT_INT_EQ((int)(a & 1u), 1);
  ASSERT_INT_EQ((int)a, (int)b);
  PASS();
}

TEST_BACKEND(x86, rewritten_const_isel_keeps_range_bounds_structural) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = x86_make_vecadd(ctx, 8);
  int n_lin = 0;
  PolyUOp **lin = poly_linearize_x86_rewritten(ctx, sink, &n_lin);
  ASSERT_NOT_NULL(lin);
  ASSERT_TRUE(x86_linear_all_range_bounds_are_consts(lin, n_lin));
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, rewritten_float_const_isel_matches_tinygrad_bitcast_path) {
  enum { N = 4 };
  float a[N], out[N];
  for (int i = 0; i < N; i++) {
    a[i] = (float)i * 0.5f;
    out[i] = -99.0f;
  }
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = x86_make_f32_add_const(ctx, N, 1.25f);
  int n_lin = 0;
  PolyUOp **lin = poly_linearize_x86_rewritten(ctx, sink, &n_lin);
  ASSERT_NOT_NULL(lin);
  ASSERT_INT_EQ(x86_linear_count_const_arg(lin, n_lin, POLY_ARG_FLOAT), 0);
  free(lin);

  void *args[2] = {a, out};
  ASSERT_INT_EQ(x86_run_rewritten_direct(ctx, sink, args, 2), 0);
  for (int i = 0; i < N; i++)
    ASSERT_FLOAT_EQ(out[i], a[i] + 1.25f, 0.0f);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, rewritten_f32_vector_index_lane0_noop_matches_tinygrad) {
  float in[4] = {3.5f, -2.0f, 7.0f, 8.0f};
  float out[1] = {-99.0f};
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = x86_make_index_extract(ctx, POLY_FLOAT32, 4, 0);
  void *args[2] = {in, out};
  ASSERT_INT_EQ(x86_run_rewritten_direct(ctx, sink, args, 2), 0);
  ASSERT_FLOAT_EQ(out[0], in[0], 0.0f);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, direct_vecadd_matches_tinygrad_probe_class) {
  enum { N = 8 };
  float a[N], b[N], out[N];
  for (int i = 0; i < N; i++) {
    a[i] = (float)i - 3.0f;
    b[i] = 0.25f * (float)i;
    out[i] = -99.0f;
  }
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = x86_make_vecadd(ctx, N);
  void *args[3] = {a, b, out};
  ASSERT_INT_EQ(x86_run_direct(ctx, sink, args, 3), 0);
  for (int i = 0; i < N; i++)
    ASSERT_FLOAT_EQ(out[i], a[i] + b[i], 1e-6);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, direct_scalar_vecadd_matches_tinygrad_readmem2nd_probe) {
  enum { N = 1 };
  float a[N] = {2.0f}, b[N] = {3.0f}, out[N] = {-99.0f};
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = x86_make_vecadd(ctx, N);
  void *args[3] = {a, b, out};
  ASSERT_INT_EQ(x86_run_direct(ctx, sink, args, 3), 0);
  ASSERT_FLOAT_EQ(out[0], 5.0f, 1e-6);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, direct_f32_broadcast4_matches_tinygrad_vbroadcastss_probe) {
  float x[1] = {2.0f};
  float out[4] = {-99.0f, -99.0f, -99.0f, -99.0f};
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = x86_make_f32_broadcast4(ctx);
  void *args[2] = {x, out};
  ASSERT_INT_EQ(x86_run_direct(ctx, sink, args, 2), 0);
  for (int i = 0; i < 4; i++)
    ASSERT_FLOAT_EQ(out[i], 2.0f, 1e-6);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, direct_f16_broadcast4_matches_tinygrad_vpinsrw_probe) {
  uint16_t x[1] = {x86_f32_to_f16_bits(2.0f)};
  uint16_t out[4] = {0, 0, 0, 0};
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = x86_make_f16_broadcast4(ctx);
  void *args[2] = {x, out};
  ASSERT_INT_EQ(x86_run_direct(ctx, sink, args, 2), 0);
  for (int i = 0; i < 4; i++)
    ASSERT_FLOAT_EQ(x86_f16_bits_to_f32(out[i]), 2.0f, 0.0f);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, direct_i32_broadcast4_matches_tinygrad_vpbroadcastd_probe) {
  int32_t x[1] = {7};
  int32_t out[4] = {0, 0, 0, 0};
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = x86_make_i32_broadcast4(ctx);
  void *args[2] = {x, out};
  ASSERT_INT_EQ(x86_run_direct(ctx, sink, args, 2), 0);
  for (int i = 0; i < 4; i++)
    ASSERT_INT_EQ(out[i], 7);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, rewritten_i32_broadcast4_isel_matches_tinygrad_vpbroadcastd_probe) {
  int32_t x[1] = {7};
  int32_t out[4] = {0, 0, 0, 0};
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = x86_make_i32_broadcast4(ctx);
  void *args[2] = {x, out};
  ASSERT_INT_EQ(x86_run_rewritten_direct(ctx, sink, args, 2), 0);
  for (int i = 0; i < 4; i++)
    ASSERT_INT_EQ(out[i], 7);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, direct_float_where_matches_tinygrad_probe_class) {
  enum { N = 8 };
  float a[N], b[N], out[N];
  for (int i = 0; i < N; i++) {
    a[i] = (float)i - 3.0f;
    b[i] = 0.25f * (float)i;
    out[i] = -99.0f;
  }
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = x86_make_abs_plus_bias(ctx, N);
  void *args[3] = {a, b, out};
  ASSERT_INT_EQ(x86_run_direct(ctx, sink, args, 3), 0);
  for (int i = 0; i < N; i++) {
    float expected = (a[i] > 0.0f ? a[i] : -a[i]) + b[i];
    ASSERT_FLOAT_EQ(out[i], expected, 1e-6);
  }
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, direct_scalar_float_where_uses_compare_mask_like_tinygrad) {
  float a[1] = {2.0f};
  float b[1] = {10.0f};
  float out[1] = {-99.0f};
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = x86_make_abs_plus_bias(ctx, 1);
  void *args[3] = {a, b, out};
  ASSERT_INT_EQ(x86_run_direct(ctx, sink, args, 3), 0);
  ASSERT_FLOAT_EQ(out[0], 12.0f, 1e-6);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, direct_float_compare_selects_integer_with_tinygrad_cmovb) {
  enum { N = 1 };
  float in[N] = {1.0f};
  int16_t out[N] = {0};
  PolyCtx *ctx = poly_ctx_new();
  PolyDType pf = poly_dtype_ptr(POLY_FLOAT32, -1, POLY_ADDR_GLOBAL);
  PolyDType pi = poly_dtype_ptr(POLY_INT16, -1, POLY_ADDR_GLOBAL);
  PolyUOp *pin = poly_uop0(ctx, POLY_OP_PARAM, pf, poly_arg_int(0));
  PolyUOp *pout = poly_uop0(ctx, POLY_OP_PARAM, pi, poly_arg_int(1));
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(N));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));
  PolyUOp *src_idx = poly_uop2(ctx, POLY_OP_INDEX, pf, pin, range, poly_arg_none());
  PolyUOp *dst_idx = poly_uop2(ctx, POLY_OP_INDEX, pi, pout, range, poly_arg_none());
  PolyUOp *value = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, src_idx, poly_arg_none());
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(0.0));
  PolyUOp *mask = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, zero, value, poly_arg_none());
  PolyUOp *select_srcs[3] = {
      mask,
      poly_uop0(ctx, POLY_OP_CONST, POLY_INT16, poly_arg_int(7)),
      poly_uop0(ctx, POLY_OP_CONST, POLY_INT16, poly_arg_int(3)),
  };
  PolyUOp *selected = poly_uop(ctx, POLY_OP_WHERE, POLY_INT16, select_srcs, 3, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, dst_idx, selected, poly_arg_none());
  PolyUOp *end_srcs[2] = {store, range};
  PolyUOp *sink = poly_uop1(
      ctx, POLY_OP_SINK, POLY_VOID,
      poly_uop(ctx, POLY_OP_END, POLY_VOID, end_srcs, 2, poly_arg_none()), poly_arg_none()
  );
  void *args[2] = {in, out};
  ASSERT_INT_EQ(x86_run_direct(ctx, sink, args, 2), 0);
  ASSERT_INT_EQ(out[0], 7);
  in[0] = -2.0f;
  out[0] = 0;
  ASSERT_INT_EQ(x86_run_direct(ctx, sink, args, 2), 0);
  ASSERT_INT_EQ(out[0], 3);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, rewritten_float_neg_pre_isel_matches_tinygrad_sub_zero_probe) {
  /* tinygrad CPU:X86 probe: NEG lowers before isel to SUB(0, x), emitted as VSUBSS. */
  float a[3] = {1.0f, -2.0f, 3.0f};
  float b[3] = {0.25f, 0.5f, 0.75f};
  float out[3] = {-99.0f, -99.0f, -99.0f};
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = x86_make_abs_plus_bias(ctx, 3);
  void *args[3] = {a, b, out};
  ASSERT_INT_EQ(x86_run_rewritten_direct(ctx, sink, args, 3), 0);
  for (int i = 0; i < 3; i++) {
    float expected = (a[i] > 0.0f ? a[i] : -a[i]) + b[i];
    ASSERT_FLOAT_EQ(out[i], expected, 1e-6);
  }
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, rewritten_vector_neg_const_like_uses_scalar_base_dtype) {
  /* tinygrad UOp.const_like uses dtype.base: vector NEG becomes
   * SUB(floatx4, CONST(float, 0), value), not CONST(floatx4, 0). */
  enum { N = 8 };
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *b = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *c = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *out1 = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *out2 = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *neg = poly_uop1(ctx, POLY_OP_NEG, POLY_FLOAT32, a, poly_arg_none());
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, neg, b, poly_arg_none());
  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, neg, c, poly_arg_none());
  PolyUOp *s1 = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out1, add, poly_arg_none());
  PolyUOp *s2 = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out2, mul, poly_arg_none());
  PolyUOp *stores[] = {s1, s2};
  PolyUOp *sink = poly_uop(ctx, POLY_OP_SINK, POLY_VOID, stores, 2, poly_arg_none());

  PolyKernelScheduleResult sr = poly_build_kernel_schedule(ctx, sink);
  ASSERT_INT_EQ(sr.n_kernels, 2);
  int found_vector_sub_zero = 0;
  for (int k = 0; k < sr.n_kernels; k++) {
    PolyUOp *rewritten = poly_rewrite_x86(ctx, sr.kernels[k]);
    ASSERT_NOT_NULL(rewritten);
    int n_topo = 0;
    PolyUOp **topo = poly_toposort_alloc(ctx, rewritten, &n_topo);
    ASSERT_NOT_NULL(topo);
    for (int i = 0; i < n_topo; i++) {
      PolyUOp *u = topo[i];
      if (!u || u->op != POLY_OP_SUB || u->dtype.count <= 1 || u->n_src != 2 ||
          !u->src[0] || u->src[0]->op != POLY_OP_CONST)
        continue;
      found_vector_sub_zero++;
      ASSERT_TRUE(poly_dtype_eq(u->src[0]->dtype, poly_dtype_scalar(u->dtype)));
      ASSERT_INT_EQ(u->src[0]->dtype.count, 1);
    }
    poly_toposort_free(topo);
  }
  ASSERT_TRUE(found_vector_sub_zero > 0);

  poly_kernel_schedule_result_free(&sr);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, direct_f32_bool_compare_store_matches_tinygrad_probe_class) {
  enum { N = 3 };
  float a[N] = {1.0f, 3.0f, 2.0f};
  float b[N] = {2.0f, 2.0f, 2.0f};
  uint8_t out[N] = {0xba, 0xba, 0xba};
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *lt_sink = x86_make_f32_compare_bool(ctx, N, POLY_OP_CMPLT);
  void *lt_args[3] = {a, b, out};
  ASSERT_INT_EQ(x86_run_direct(ctx, lt_sink, lt_args, 3), 0);
  ASSERT_INT_EQ(out[0], 1);
  ASSERT_INT_EQ(out[1], 0);
  ASSERT_INT_EQ(out[2], 0);

  memset(out, 0xba, sizeof(out));
  PolyUOp *gt_sink = x86_make_f32_compare_bool(ctx, N, POLY_OP_CMPLT);
  void *gt_args[3] = {b, a, out};
  ASSERT_INT_EQ(x86_run_direct(ctx, gt_sink, gt_args, 3), 0);
  ASSERT_INT_EQ(out[0], 0);
  ASSERT_INT_EQ(out[1], 1);
  ASSERT_INT_EQ(out[2], 0);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, direct_f64_bool_compare_uses_imm32_mask_like_tinygrad) {
  enum { N = 3 };
  double a[N] = {0.0, 3.0, -2.0};
  double b[N] = {1.0, 2.0, -2.0};
  uint8_t out[N] = {0xba, 0xba, 0xba};
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *lt_sink = x86_make_f64_compare_bool(ctx, N, POLY_OP_CMPLT);
  void *lt_args[3] = {a, b, out};
  ASSERT_INT_EQ(x86_run_direct(ctx, lt_sink, lt_args, 3), 0);
  ASSERT_INT_EQ(out[0], 1);
  ASSERT_INT_EQ(out[1], 0);
  ASSERT_INT_EQ(out[2], 0);

  memset(out, 0xba, sizeof(out));
  PolyUOp *ne_sink = x86_make_f64_compare_bool(ctx, N, POLY_OP_CMPNE);
  void *ne_args[3] = {a, b, out};
  ASSERT_INT_EQ(x86_run_direct(ctx, ne_sink, ne_args, 3), 0);
  ASSERT_INT_EQ(out[0], 1);
  ASSERT_INT_EQ(out[1], 1);
  ASSERT_INT_EQ(out[2], 0);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, direct_mixed_float_int_compare_matches_python_embedding_mask) {
  enum { N = 5 };
  float idx[1] = {2.0f};
  uint8_t out[N] = {0xba, 0xba, 0xba, 0xba, 0xba};
  PolyCtx *ctx = poly_ctx_new();
  PolyDType pf = poly_dtype_ptr(POLY_FLOAT32, -1, POLY_ADDR_GLOBAL);
  PolyDType pb = poly_dtype_ptr(POLY_BOOL, -1, POLY_ADDR_GLOBAL);
  PolyUOp *p_idx = poly_uop0(ctx, POLY_OP_PARAM, pf, poly_arg_int(0));
  PolyUOp *p_out = poly_uop0(ctx, POLY_OP_PARAM, pb, poly_arg_int(1));
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(N));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *idx_addr = poly_uop2(ctx, POLY_OP_INDEX, pf, p_idx, zero, poly_arg_none());
  PolyUOp *out_addr = poly_uop2(ctx, POLY_OP_INDEX, pb, p_out, range, poly_arg_none());
  PolyUOp *idx_val = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx_addr, poly_arg_none());
  PolyUOp *mask = poly_uop2(ctx, POLY_OP_CMPEQ, POLY_BOOL, idx_val, range, poly_arg_none());
  PolyUOp *st = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out_addr, mask, poly_arg_none());
  PolyUOp *end_srcs[2] = {st, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_srcs, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());

  void *args[2] = {idx, out};
  ASSERT_INT_EQ(x86_run_direct(ctx, sink, args, 2), 0);
  const uint8_t expected[N] = {0, 0, 1, 0, 0};
  for (int i = 0; i < N; i++)
    ASSERT_INT_EQ(out[i], expected[i]);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, direct_negative_step2_sign_extends_address_index_like_tinygrad) {
  float in[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  float out[3] = {0.0f, 0.0f, 0.0f};
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = x86_make_negative_step2(ctx);
  void *args[2] = {in, out};
  ASSERT_INT_EQ(x86_run_direct(ctx, sink, args, 2), 0);
  ASSERT_FLOAT_EQ(out[0], 6.0f, 0.0f);
  ASSERT_FLOAT_EQ(out[1], 4.0f, 0.0f);
  ASSERT_FLOAT_EQ(out[2], 2.0f, 0.0f);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, direct_stack_args_follow_tinygrad_sysv_x86_abi) {
  enum { N = 4, IN = 7 };
  float in[IN][N];
  float out[N] = {0};
  for (int j = 0; j < IN; j++)
    for (int i = 0; i < N; i++)
      in[j][i] = (float)(j + 1) + 0.25f * (float)i;
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = x86_make_seven_input_sum(ctx, N);
  void *args[8] = {in[0], in[1], in[2], in[3], in[4], in[5], in[6], out};
  ASSERT_INT_EQ(x86_run_direct(ctx, sink, args, 8), 0);
  for (int i = 0; i < N; i++) {
    float expected = 0.0f;
    for (int j = 0; j < IN; j++)
      expected += in[j][i];
    ASSERT_FLOAT_EQ(out[i], expected, 1e-6);
  }
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, schedule_runtime_tabm_forward_preserves_callee_saved_stack_args) {
  const char *spec = "{\"layers\":[2,4,1],\"activation\":\"relu\","
                     "\"loss\":\"mse\",\"batch_size\":1,\"seed\":42,\"n_ensemble\":4}";
  PolyInstance *inst = poly_tabm_instance(spec, (int)strlen(spec), POLY_DEVICE_AUTO);
  ASSERT_NOT_NULL(inst);
  ASSERT_INT_EQ(poly_instance_set_device(inst, POLY_DEVICE_X86), 0);

  float x[2] = {1.0f, 2.0f};
  PolyIOBinding io = POLY_IO_BINDING_ARRAY("x", x, POLY_FLOAT32);
  ASSERT_INT_EQ(poly_instance_forward(inst, &io, 1), 0);

  int64_t n = 0;
  float *out = poly_instance_buf_data_named(inst, "output", &n);
  ASSERT_NOT_NULL(out);
  ASSERT_TRUE(n == 1);
  ASSERT_TRUE(isfinite(out[0]));

  poly_instance_free(inst);
  PASS();
}

TEST_BACKEND(x86, direct_f64_where_matches_tinygrad_probe_class) {
  enum { N = 4 };
  double a[N] = {-1.0, 1.5, 2.0, -2.0};
  double b[N] = {0.5, 0.5, 4.0, 4.0};
  double out[N] = {0};
  const double expected[N] = {0.5, 2.0, 6.0, 4.0};
  PolyCtx *ctx = poly_ctx_new();
  PolyDType pd = poly_dtype_ptr(POLY_FLOAT64, -1, POLY_ADDR_GLOBAL);
  PolyUOp *sink = x86_make_one_range(ctx, pd, POLY_FLOAT64, N, x86_expr_f64_where);
  void *args[3] = {a, b, out};
  ASSERT_INT_EQ(x86_run_direct(ctx, sink, args, 3), 0);
  for (int i = 0; i < N; i++)
    ASSERT_FLOAT_EQ(out[i], expected[i], 1e-9);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, direct_f32_max_uses_tinygrad_vmax_pattern) {
  enum { N = 4 };
  float a[N] = {1.0f, -2.0f, 3.0f, -4.0f};
  float b[N] = {0.5f, 0.5f, 0.5f, 0.5f};
  float out[N] = {0};
  const float expected[N] = {1.0f, 0.5f, 3.0f, 0.5f};
  PolyCtx *ctx = poly_ctx_new();
  PolyDType pf = poly_dtype_ptr(POLY_FLOAT32, -1, POLY_ADDR_GLOBAL);
  PolyUOp *sink = x86_make_one_range(ctx, pf, POLY_FLOAT32, N, x86_expr_f32_max);
  void *args[3] = {a, b, out};
  ASSERT_INT_EQ(x86_run_direct(ctx, sink, args, 3), 0);
  for (int i = 0; i < N; i++)
    ASSERT_FLOAT_EQ(out[i], expected[i], 1e-6);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, rewritten_f32_min_where_uses_tinygrad_vmin_pattern) {
  enum { N = 4 };
  float a[N] = {1.0f, -2.0f, 3.0f, -4.0f};
  float b[N] = {0.5f, 0.5f, 0.5f, 0.5f};
  float out[N] = {0};
  const float expected[N] = {0.5f, -2.0f, 0.5f, -4.0f};
  PolyCtx *ctx = poly_ctx_new();
  PolyDType pf = poly_dtype_ptr(POLY_FLOAT32, -1, POLY_ADDR_GLOBAL);
  PolyUOp *sink = x86_make_one_range(ctx, pf, POLY_FLOAT32, N, x86_expr_f32_min_where);
  ASSERT_TRUE(x86_rewritten_code_contains_hex(ctx, sink, "c5fa5d"));
  void *args[3] = {a, b, out};
  ASSERT_INT_EQ(x86_run_rewritten_direct(ctx, sink, args, 3), 0);
  for (int i = 0; i < N; i++)
    ASSERT_FLOAT_EQ(out[i], expected[i], 1e-6);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, direct_i32_mix_matches_polygrad_probe_class) {
  enum { N = 8 };
  int32_t a[N] = {1, -2, 3, -4, 5, -6, 7, -8};
  int32_t b[N] = {6, 7, -6, -7, 2, -1, 1, -3};
  int32_t out[N] = {0};
  const int32_t expected[N] = {1, -23, -9, 31, 13, 1, 13, 19};
  PolyCtx *ctx = poly_ctx_new();
  PolyDType pi = poly_dtype_ptr(POLY_INT32, -1, POLY_ADDR_GLOBAL);
  PolyUOp *sink = x86_make_one_range(ctx, pi, POLY_INT32, N, x86_expr_i32_mix);
  void *args[3] = {a, b, out};
  ASSERT_INT_EQ(x86_run_direct(ctx, sink, args, 3), 0);
  for (int i = 0; i < N; i++)
    ASSERT_INT_EQ(out[i], expected[i]);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, rewritten_reused_mul_does_not_fuse_vfmadd_like_tinygrad) {
  float a[1] = {2.0f};
  float b[1] = {3.0f};
  float out[1] = {0};
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = x86_make_f32_reused_mul_add(ctx);
  /* tinygrad CPU:X86 probe for `a*b + a*b`:
   *   root VADDSS, sources VMULSS and the same VMULSS again.
   * Polygrad should therefore contain scalar VMULSS/VADDSS opcodes and no
   * VFMADD213SS opcode for this multi-use multiply.
   */
  ASSERT_TRUE(x86_rewritten_code_contains_hex(ctx, sink, "c5fa59"));
  ASSERT_TRUE(x86_rewritten_code_contains_hex(ctx, sink, "c5fa58"));
  ASSERT_INT_EQ(x86_rewritten_code_count_hex(ctx, sink, "a9"), 0);
  void *args[3] = {a, b, out};
  ASSERT_INT_EQ(x86_run_rewritten_direct(ctx, sink, args, 3), 0);
  ASSERT_FLOAT_EQ(out[0], 12.0f, 0.0f);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, rewritten_complex_address_folds_index_plus_const_like_tinygrad) {
  int32_t in[8] = {3, 5, 7, 11, 13, 17, 19, 23};
  int32_t out[1] = {0};
  int32_t i = 2;
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = x86_make_i32_complex_address_load(ctx);
  /* tinygrad CPU:X86 probe:
   *   MOV src tuple is base, index, CONST(int8, 4), CONST(uint8, 4).
   * The direct byte check is `mov 0x4(base,index,4), reg`.
   */
  ASSERT_TRUE(x86_rewritten_code_contains_hex(ctx, sink, "8b448804"));
  void *args[3] = {in, out, (void *)(intptr_t)i};
  ASSERT_INT_EQ(x86_run_rewritten_direct(ctx, sink, args, 3), 0);
  ASSERT_INT_EQ(out[0], in[i + 1]);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, rewritten_single_use_load_folds_into_integer_add_like_tinygrad) {
  int32_t in[2] = {10, 32};
  int32_t out[1] = {0};
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = x86_make_i32_fold_load_add(ctx);
  /* tinygrad CPU:X86 probe for `load0 + load1`:
   *   root ADD has five sources after ReadMem folding.
   * The direct byte check is `add 0x4(base), reg`.
   */
  ASSERT_TRUE(x86_rewritten_code_contains_hex(ctx, sink, "034804"));
  void *args[2] = {in, out};
  ASSERT_INT_EQ(x86_run_rewritten_direct(ctx, sink, args, 2), 0);
  ASSERT_INT_EQ(out[0], 42);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, rewritten_multiuse_load_does_not_fold_like_tinygrad) {
  int32_t in[1] = {7};
  int32_t out[1] = {0};
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *add_sink = x86_make_i32_multiuse_load_add(ctx);
  /* tinygrad CPU:X86 probe:
   *   `(load + 1) + load` remains ADD(ADDi, MOV), no ReadMem tuple.
   */
  ASSERT_TRUE(x86_rewritten_code_contains_hex(ctx, add_sink, "8b00"));
  ASSERT_TRUE(x86_rewritten_code_contains_hex(ctx, add_sink, "03d0"));
  ASSERT_INT_EQ(x86_rewritten_code_count_hex(ctx, add_sink, "0340"), 0);
  void *args[2] = {in, out};
  ASSERT_INT_EQ(x86_run_rewritten_direct(ctx, add_sink, args, 2), 0);
  ASSERT_INT_EQ(out[0], 15);

  out[0] = 0;
  PolyUOp *mul_sink = x86_make_i32_multiuse_load_mul(ctx);
  /* tinygrad CPU:X86 probe:
   *   `load * load` remains IMUL(MOV, MOV), no folded memory operand.
   */
  ASSERT_TRUE(x86_rewritten_code_contains_hex(ctx, mul_sink, "8b00"));
  ASSERT_TRUE(x86_rewritten_code_contains_hex(ctx, mul_sink, "0fafc0"));
  ASSERT_INT_EQ(x86_run_rewritten_direct(ctx, mul_sink, args, 2), 0);
  ASSERT_INT_EQ(out[0], 49);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, direct_i32_where_matches_tinygrad_probe_class) {
  enum { N = 8 };
  int32_t a[N] = {1, -2, 3, -4, 5, -6, 7, -8};
  int32_t b[N] = {8, 7, -6, -5, 4, 3, -2, -1};
  int32_t out[N] = {0};
  const int32_t expected[N] = {1, -2, -6, -5, 4, -6, -2, -8};
  PolyCtx *ctx = poly_ctx_new();
  PolyDType pi = poly_dtype_ptr(POLY_INT32, -1, POLY_ADDR_GLOBAL);
  PolyUOp *sink = x86_make_one_range(ctx, pi, POLY_INT32, N, x86_expr_i32_where);
  void *args[3] = {a, b, out};
  ASSERT_INT_EQ(x86_run_direct(ctx, sink, args, 3), 0);
  for (int i = 0; i < N; i++)
    ASSERT_INT_EQ(out[i], expected[i]);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, direct_i32_where_rematerializes_flags_after_clobber_like_tinygrad) {
  enum { N = 4 };
  int32_t a[N] = {1, 3, -2, 5};
  int32_t b[N] = {2, 2, -1, 5};
  int32_t out[N] = {0};
  const int32_t expected[N] = {3, 5, -3, 10};
  PolyCtx *ctx = poly_ctx_new();
  PolyDType pi = poly_dtype_ptr(POLY_INT32, -1, POLY_ADDR_GLOBAL);
  PolyUOp *sink = x86_make_one_range(ctx, pi, POLY_INT32, N, x86_expr_i32_flag_clobber_where);
  void *args[3] = {a, b, out};
  ASSERT_INT_EQ(x86_run_direct(ctx, sink, args, 3), 0);
  for (int i = 0; i < N; i++)
    ASSERT_INT_EQ(out[i], expected[i]);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, rewritten_if_compare_selects_tinygrad_jump_family) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *cases[4] = {
      x86_make_tagged_if_sink(ctx, POLY_UINT32, POLY_OP_CMPLT, ".IF_OUT_ult"),
      x86_make_tagged_if_sink(ctx, POLY_INT32, POLY_OP_CMPLT, ".IF_OUT_slt"),
      x86_make_tagged_if_sink(ctx, POLY_INT32, POLY_OP_CMPEQ, ".IF_OUT_eq"),
      x86_make_tagged_if_sink(ctx, POLY_INT32, POLY_OP_CMPNE, ".IF_OUT_ne"),
  };
  const char *labels[4] = {".IF_OUT_ult", ".IF_OUT_slt", ".IF_OUT_eq", ".IF_OUT_ne"};
  int jump_ops[4] = {0, 0, 0, 0};
  for (int i = 0; i < 4; i++) {
    int n_lin = 0;
    PolyUOp **lin = poly_linearize_x86_rewritten(ctx, cases[i], &n_lin);
    ASSERT_NOT_NULL(lin);
    ASSERT_INT_EQ(x86_linear_contains_op(lin, n_lin, POLY_OP_IF), 0);
    jump_ops[i] = x86_find_tagged_ins_arg(lin, n_lin, labels[i]);
    ASSERT_TRUE(jump_ops[i] != 0);
    free(lin);
  }
  ASSERT_TRUE(jump_ops[0] != jump_ops[1]);
  ASSERT_TRUE(jump_ops[1] != jump_ops[2]);
  ASSERT_TRUE(jump_ops[2] != jump_ops[3]);
  ASSERT_TRUE(jump_ops[0] != jump_ops[3]);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, direct_cast_i32_f32_matches_tinygrad_probe_class) {
  enum { N = 4 };
  int32_t a[N] = {1, 2, 3, 4};
  float out[N] = {0};
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = x86_make_cast_i32_f32(ctx, N);
  void *args[2] = {a, out};
  ASSERT_INT_EQ(x86_run_direct(ctx, sink, args, 2), 0);
  for (int i = 0; i < N; i++)
    ASSERT_FLOAT_EQ(out[i], (float)a[i], 1e-6);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, direct_vector_f32_to_i32_cast_uses_tinygrad_vcvttps2dq) {
  enum { V = 2, LANES = 4, N = V * LANES };
  float a[N] = {-2.8f, -1.1f, 0.0f, 1.9f, 2.2f, 3.8f, 127.9f, -128.4f};
  int32_t out[N] = {0};
  const int32_t expected[N] = {-2, -1, 0, 1, 2, 3, 127, -128};
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = x86_make_vector_cast(ctx, POLY_FLOAT32, LANES, POLY_INT32, LANES, V);
  void *args[2] = {a, out};
  ASSERT_INT_EQ(x86_run_rewritten_direct(ctx, sink, args, 2), 0);
  for (int i = 0; i < N; i++)
    ASSERT_INT_EQ(out[i], expected[i]);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, direct_vector_f64_to_i32_cast_uses_tinygrad_vcvttpd2dq) {
  enum { V = 2, LANES = 2, N = V * LANES };
  double a[N] = {-2.8, -1.1, 1.9, 127.9};
  int32_t out[N] = {0};
  const int32_t expected[N] = {-2, -1, 1, 127};
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = x86_make_vector_cast(ctx, POLY_FLOAT64, LANES, POLY_INT32, LANES, V);
  void *args[2] = {a, out};
  ASSERT_INT_EQ(x86_run_rewritten_direct(ctx, sink, args, 2), 0);
  for (int i = 0; i < N; i++)
    ASSERT_INT_EQ(out[i], expected[i]);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, direct_vector_f64_to_f32_cast_uses_tinygrad_vcvtpd2ps) {
  enum { V = 2, LANES = 2, N = V * LANES };
  double a[N] = {-2.5, -1.25, 1.75, 127.5};
  float out[N] = {0};
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = x86_make_vector_cast(ctx, POLY_FLOAT64, LANES, POLY_FLOAT32, LANES, V);
  void *args[2] = {a, out};
  ASSERT_INT_EQ(x86_run_rewritten_direct(ctx, sink, args, 2), 0);
  for (int i = 0; i < N; i++)
    ASSERT_FLOAT_EQ(out[i], (float)a[i], 0.0f);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, direct_vector_i32_to_f64_cast_uses_tinygrad_vcvtdq2pd) {
  enum { V = 2, IN_LANES = 4, OUT_LANES = 4, N = V * IN_LANES };
  int32_t a[N] = {-3, -1, 0, 2, 4, 8, 16, 32};
  double out[N] = {0};
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = x86_make_vector_cast(ctx, POLY_INT32, IN_LANES, POLY_FLOAT64, OUT_LANES, V);
  void *args[2] = {a, out};
  ASSERT_INT_EQ(x86_run_rewritten_direct(ctx, sink, args, 2), 0);
  for (int i = 0; i < N; i++)
    ASSERT_FLOAT_EQ(out[i], (double)a[i], 0.0);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, direct_vector_f32_to_f64_cast_uses_tinygrad_vcvtps2pd) {
  enum { V = 2, IN_LANES = 4, OUT_LANES = 4, N = V * IN_LANES };
  float a[N] = {-3.5f, -1.25f, 0.0f, 2.5f, 4.25f, 8.5f, 16.75f, 32.125f};
  double out[N] = {0};
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = x86_make_vector_cast(ctx, POLY_FLOAT32, IN_LANES, POLY_FLOAT64, OUT_LANES, V);
  void *args[2] = {a, out};
  ASSERT_INT_EQ(x86_run_rewritten_direct(ctx, sink, args, 2), 0);
  for (int i = 0; i < N; i++)
    ASSERT_FLOAT_EQ(out[i], (double)a[i], 0.0);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, direct_shrink_load_width_matches_tinygrad_pre_isel) {
  float a[4] = {1.25f, -2.5f, 3.75f, 9.0f};
  float out[4] = {0};
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = x86_make_shrink_load_width4(ctx);
  void *args[2] = {a, out};
  ASSERT_INT_EQ(x86_run_direct(ctx, sink, args, 2), 0);
  for (int i = 0; i < 4; i++)
    ASSERT_FLOAT_EQ(out[i], a[i], 0.0f);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, direct_gated_load_false_uses_scratch_alt_like_tinygrad) {
  float a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float out[1] = {0.0f};
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = x86_make_gated_load_false_uses_alt(ctx);
  void *args[2] = {a, out};
  ASSERT_INT_EQ(x86_run_direct(ctx, sink, args, 2), 0);
  ASSERT_FLOAT_EQ(out[0], 7.5f, 0.0f);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, direct_gated_store_false_uses_scratch_like_tinygrad) {
  float out[1] = {11.0f};
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = x86_make_gated_store_false_uses_scratch(ctx);
  void *args[1] = {out};
  ASSERT_INT_EQ(x86_run_direct(ctx, sink, args, 1), 0);
  ASSERT_FLOAT_EQ(out[0], 11.0f, 0.0f);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, direct_u32_to_i64_cast_matches_tinygrad_pre_isel_noop) {
  enum { N = 4 };
  uint32_t a[N] = {1u, 2147483651u, 17u, 123u};
  int64_t out[N] = {0};
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = x86_make_cast_u32_i64(ctx, N);
  void *args[2] = {a, out};
  ASSERT_INT_EQ(x86_run_direct(ctx, sink, args, 2), 0);
  for (int i = 0; i < N; i++)
    ASSERT_INT_EQ(out[i], (int64_t)(uint64_t)a[i]);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, direct_i64_to_i32_cast_matches_tinygrad_pre_isel_noop) {
  enum { N = 4 };
  int64_t a[N] = {1, -2, 3, -4};
  int32_t out[N] = {0};
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = x86_make_cast_i64_i32(ctx, N);
  void *args[2] = {a, out};
  ASSERT_INT_EQ(x86_run_direct(ctx, sink, args, 2), 0);
  for (int i = 0; i < N; i++)
    ASSERT_INT_EQ(out[i], (int32_t)a[i]);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, direct_vector_u8_to_i32_cast_uses_tinygrad_vpmov_rule) {
  enum { N = 8, V = 2 };
  uint8_t a[N] = {1, 2, 3, 4, 5, 6, 7, 255};
  int32_t out[N] = {0};
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = x86_make_cast_u8x4_i32x4(ctx, V);
  void *args[2] = {a, out};
  ASSERT_INT_EQ(x86_run_direct(ctx, sink, args, 2), 0);
  for (int i = 0; i < N; i++)
    ASSERT_INT_EQ(out[i], (int32_t)a[i]);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, direct_i32x4_addsub_matches_tinygrad_readmem2nd_group) {
  enum { N = 8, V = 2 };
  int32_t a[N] = {1, -2, 3, -4, 5, -6, 7, -8};
  int32_t b[N] = {8, 7, -6, -5, 4, 3, -2, -1};
  int32_t out[N] = {0};
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = x86_make_i32x4_addsub(ctx, V);
  void *args[3] = {a, b, out};
  ASSERT_INT_EQ(x86_run_direct(ctx, sink, args, 3), 0);
  for (int i = 0; i < N; i++)
    ASSERT_INT_EQ(out[i], b[i]);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, rewritten_i32x4_addsub_isel_uses_tinygrad_packed_ops) {
  enum { N = 8, V = 2 };
  int32_t a[N] = {1, -2, 3, -4, 5, -6, 7, -8};
  int32_t b[N] = {8, 7, -6, -5, 4, 3, -2, -1};
  int32_t out[N] = {0};
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = x86_make_i32x4_addsub(ctx, V);
  void *args[3] = {a, b, out};
  ASSERT_INT_EQ(x86_run_rewritten_direct(ctx, sink, args, 3), 0);
  for (int i = 0; i < N; i++)
    ASSERT_INT_EQ(out[i], b[i]);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, rewritten_store_family_uses_tinygrad_memory_ops) {
  enum { N = 8, V4 = 2 };

  float f32_in[N] = {1.0f, -2.0f, 3.5f, 4.25f, -5.0f, 6.0f, 7.75f, -8.5f};
  float f32_out[N] = {0};
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = x86_make_load_store(ctx, POLY_FLOAT32, 1, N);
  ASSERT_TRUE(x86_rewritten_linear_count_ins_arg(ctx, sink, TX86_OP_VMOVSSm) > 0);
  void *args_f32[2] = {f32_in, f32_out};
  ASSERT_INT_EQ(x86_run_rewritten_direct(ctx, sink, args_f32, 2), 0);
  for (int i = 0; i < N; i++)
    ASSERT_FLOAT_EQ(f32_out[i], f32_in[i], 0.0f);
  poly_ctx_destroy(ctx);

  double f64_in[N] = {1.0, -2.0, 3.5, 4.25, -5.0, 6.0, 7.75, -8.5};
  double f64_out[N] = {0};
  ctx = poly_ctx_new();
  sink = x86_make_load_store(ctx, POLY_FLOAT64, 1, N);
  ASSERT_TRUE(x86_rewritten_linear_count_ins_arg(ctx, sink, TX86_OP_VMOVSDm) > 0);
  void *args_f64[2] = {f64_in, f64_out};
  ASSERT_INT_EQ(x86_run_rewritten_direct(ctx, sink, args_f64, 2), 0);
  for (int i = 0; i < N; i++)
    ASSERT_FLOAT_EQ((float)f64_out[i], (float)f64_in[i], 0.0f);
  poly_ctx_destroy(ctx);

  int32_t i32_in[N] = {1, -2, 3, -4, 5, -6, 7, -8};
  int32_t i32_out[N] = {0};
  ctx = poly_ctx_new();
  sink = x86_make_load_store(ctx, POLY_INT32, 1, N);
  ASSERT_TRUE(x86_rewritten_linear_count_ins_arg(ctx, sink, TX86_OP_MOVm) > 0);
  void *args_i32[2] = {i32_in, i32_out};
  ASSERT_INT_EQ(x86_run_rewritten_direct(ctx, sink, args_i32, 2), 0);
  for (int i = 0; i < N; i++)
    ASSERT_INT_EQ(i32_out[i], i32_in[i]);
  poly_ctx_destroy(ctx);

  int32_t const_out[N] = {0};
  ctx = poly_ctx_new();
  sink = x86_make_i32_const_store(ctx, N, 17);
  ASSERT_TRUE(x86_rewritten_linear_count_ins_arg(ctx, sink, TX86_OP_MOVi) > 0);
  void *args_const[1] = {const_out};
  ASSERT_INT_EQ(x86_run_rewritten_direct(ctx, sink, args_const, 1), 0);
  for (int i = 0; i < N; i++)
    ASSERT_INT_EQ(const_out[i], 17);
  poly_ctx_destroy(ctx);

  ctx = poly_ctx_new();
  sink = x86_make_load_store(ctx, POLY_FLOAT16, 1, N);
  ASSERT_TRUE(x86_rewritten_linear_count_ins_arg(ctx, sink, TX86_OP_VPEXTRW) > 0);
  poly_ctx_destroy(ctx);

  memset(f32_out, 0, sizeof(f32_out));
  ctx = poly_ctx_new();
  sink = x86_make_load_store(ctx, POLY_FLOAT32, 4, V4);
  ASSERT_TRUE(x86_rewritten_linear_count_ins_arg(ctx, sink, TX86_OP_VMOVUPSm) > 0);
  ASSERT_INT_EQ(x86_run_rewritten_direct(ctx, sink, args_f32, 2), 0);
  for (int i = 0; i < N; i++)
    ASSERT_FLOAT_EQ(f32_out[i], f32_in[i], 0.0f);
  poly_ctx_destroy(ctx);

  PASS();
}

TEST_BACKEND(x86, rewritten_int_vector_bitwise_isel_uses_tinygrad_packed_ops) {
  enum { N = 8, V = 2 };
  int32_t a[N] = {0x13, 0x24, 0x35, 0x46, 0x57, 0x68, 0x79, 0x8a};
  int32_t b[N] = {0x0f, 0x33, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa};
  int32_t out[N] = {0};
  struct {
    PolyOps op;
    int x86_op;
  } cases[] = {
      {POLY_OP_AND, TX86_OP_VPAND},
      {POLY_OP_OR, TX86_OP_VPOR},
      {POLY_OP_XOR, TX86_OP_VPXOR},
  };
  for (int ci = 0; ci < (int)(sizeof(cases) / sizeof(cases[0])); ci++) {
    memset(out, 0, sizeof(out));
    PolyCtx *ctx = poly_ctx_new();
    PolyUOp *sink = x86_make_int_vec_binary(ctx, POLY_INT32, 4, cases[ci].op, V);
    ASSERT_TRUE(x86_rewritten_linear_count_ins_arg(ctx, sink, cases[ci].x86_op) > 0);
    void *args[3] = {a, b, out};
    ASSERT_INT_EQ(x86_run_rewritten_direct(ctx, sink, args, 3), 0);
    for (int i = 0; i < N; i++) {
      int32_t expected = cases[ci].op == POLY_OP_AND  ? (a[i] & b[i])
                         : cases[ci].op == POLY_OP_OR ? (a[i] | b[i])
                                                      : (a[i] ^ b[i]);
      ASSERT_INT_EQ(out[i], expected);
    }
    poly_ctx_destroy(ctx);
  }
  PASS();
}

TEST_BACKEND(x86, rewritten_int_vector_mul_shift_isel_uses_tinygrad_packed_ops) {
  enum { N32 = 8, V32 = 2, N16 = 16, V16 = 2, N64 = 4, V64 = 2 };

  int16_t a16[N16] = {1, 2, -3, 4, 5, -6, 7, 8, 2, -3, 4, 5, -6, 7, 8, 9};
  int16_t b16[N16] = {9, 8, 7, -6, 5, 4, -3, 2, 1, 2, -3, 4, 5, -6, 7, 8};
  int16_t out16[N16] = {0};
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = x86_make_int_vec_binary(ctx, POLY_INT16, 8, POLY_OP_MUL, V16);
  ASSERT_TRUE(x86_rewritten_linear_count_ins_arg(ctx, sink, TX86_OP_VPMULLW) > 0);
  void *args16[3] = {a16, b16, out16};
  ASSERT_INT_EQ(x86_run_rewritten_direct(ctx, sink, args16, 3), 0);
  for (int i = 0; i < N16; i++)
    ASSERT_INT_EQ(out16[i], (int16_t)(a16[i] * b16[i]));
  poly_ctx_destroy(ctx);

  int32_t a32[N32] = {1, 2, 3, 4, 17, 31, 64, 127};
  int32_t b32[N32] = {1, 2, 3, 0, 1, 2, 3, 1};
  int32_t out32[N32] = {0};
  struct {
    PolyOps op;
    int x86_op;
  } i32_cases[] = {
      {POLY_OP_MUL, TX86_OP_VPMULLD},
      {POLY_OP_SHL, TX86_OP_VPSLLVD},
      {POLY_OP_SHR, TX86_OP_VPSRAVD},
  };
  for (int ci = 0; ci < (int)(sizeof(i32_cases) / sizeof(i32_cases[0])); ci++) {
    memset(out32, 0, sizeof(out32));
    ctx = poly_ctx_new();
    sink = x86_make_int_vec_binary(ctx, POLY_INT32, 4, i32_cases[ci].op, V32);
    ASSERT_TRUE(x86_rewritten_linear_count_ins_arg(ctx, sink, i32_cases[ci].x86_op) > 0);
    void *args32[3] = {a32, b32, out32};
    ASSERT_INT_EQ(x86_run_rewritten_direct(ctx, sink, args32, 3), 0);
    for (int i = 0; i < N32; i++) {
      int32_t expected = i32_cases[ci].op == POLY_OP_MUL   ? (a32[i] * b32[i])
                         : i32_cases[ci].op == POLY_OP_SHL ? (a32[i] << b32[i])
                                                           : (a32[i] >> b32[i]);
      ASSERT_INT_EQ(out32[i], expected);
    }
    poly_ctx_destroy(ctx);
  }

  uint32_t au32[N32] = {0x80000000u, 0x40000000u, 0x7fffffffu, 16u, 31u, 64u, 127u, 255u};
  uint32_t bu32[N32] = {1u, 2u, 3u, 0u, 1u, 2u, 3u, 1u};
  uint32_t outu32[N32] = {0};
  ctx = poly_ctx_new();
  sink = x86_make_int_vec_binary(ctx, POLY_UINT32, 4, POLY_OP_SHR, V32);
  ASSERT_TRUE(x86_rewritten_linear_count_ins_arg(ctx, sink, TX86_OP_VPSRLVD) > 0);
  void *argsu32[3] = {au32, bu32, outu32};
  ASSERT_INT_EQ(x86_run_rewritten_direct(ctx, sink, argsu32, 3), 0);
  for (int i = 0; i < N32; i++)
    ASSERT_INT_EQ((int)outu32[i], (int)(au32[i] >> bu32[i]));
  poly_ctx_destroy(ctx);

  int64_t a64[N64] = {1, 2, 3, 4};
  int64_t b64[N64] = {1, 2, 3, 1};
  int64_t out64[N64] = {0};
  ctx = poly_ctx_new();
  sink = x86_make_int_vec_binary(ctx, POLY_INT64, 2, POLY_OP_SHL, V64);
  ASSERT_TRUE(x86_rewritten_linear_count_ins_arg(ctx, sink, TX86_OP_VPSLLVQ) > 0);
  void *args64[3] = {a64, b64, out64};
  ASSERT_INT_EQ(x86_run_rewritten_direct(ctx, sink, args64, 3), 0);
  for (int i = 0; i < N64; i++)
    ASSERT_INT_EQ((int)out64[i], (int)(a64[i] << b64[i]));
  poly_ctx_destroy(ctx);

  uint64_t au64[N64] = {0x8000000000000000ull, 0x4000000000000000ull, 16ull, 255ull};
  uint64_t bu64[N64] = {1ull, 2ull, 3ull, 1ull};
  uint64_t outu64[N64] = {0};
  ctx = poly_ctx_new();
  sink = x86_make_int_vec_binary(ctx, POLY_UINT64, 2, POLY_OP_SHR, V64);
  ASSERT_TRUE(x86_rewritten_linear_count_ins_arg(ctx, sink, TX86_OP_VPSRLVQ) > 0);
  void *argsu64[3] = {au64, bu64, outu64};
  ASSERT_INT_EQ(x86_run_rewritten_direct(ctx, sink, argsu64, 3), 0);
  for (int i = 0; i < N64; i++)
    ASSERT_INT_EQ((int)outu64[i], (int)(au64[i] >> bu64[i]));
  poly_ctx_destroy(ctx);

  PASS();
}

TEST_BACKEND(x86, rewritten_i8_add_and_i64_sub_use_tinygrad_packed_ops) {
  enum { N8 = 16, V8 = 1, N64 = 4, V64 = 2 };
  int8_t a8[N8] = {1, 2, 3, 4, 5, 6, 7, 8, -1, -2, -3, -4, 9, 10, 11, 12};
  int8_t b8[N8] = {12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, -1, -2, -3, -4};
  int8_t out8[N8] = {0};
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = x86_make_int_vec_binary(ctx, POLY_INT8, 16, POLY_OP_ADD, V8);
  ASSERT_TRUE(x86_rewritten_linear_count_ins_arg(ctx, sink, TX86_OP_VPADDB) > 0);
  void *args8[3] = {a8, b8, out8};
  ASSERT_INT_EQ(x86_run_rewritten_direct(ctx, sink, args8, 3), 0);
  for (int i = 0; i < N8; i++)
    ASSERT_INT_EQ(out8[i], (int8_t)(a8[i] + b8[i]));
  poly_ctx_destroy(ctx);

  int64_t a64[N64] = {100, 200, -300, 400};
  int64_t b64[N64] = {1, -2, 3, -4};
  int64_t out64[N64] = {0};
  ctx = poly_ctx_new();
  sink = x86_make_int_vec_binary(ctx, POLY_INT64, 2, POLY_OP_SUB, V64);
  ASSERT_TRUE(x86_rewritten_linear_count_ins_arg(ctx, sink, TX86_OP_VPSUBQ) > 0);
  void *args64[3] = {a64, b64, out64};
  ASSERT_INT_EQ(x86_run_rewritten_direct(ctx, sink, args64, 3), 0);
  for (int i = 0; i < N64; i++)
    ASSERT_INT_EQ((int)out64[i], (int)(a64[i] - b64[i]));
  poly_ctx_destroy(ctx);

  PASS();
}

TEST_BACKEND(x86, rewritten_f32_stack_index_lanes_uses_tinygrad_vshufps) {
  float a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float b[4] = {5.0f, 6.0f, 7.0f, 8.0f};
  float out[4] = {0};
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = x86_make_f32_shuffle4(ctx);
  void *args[3] = {a, b, out};
  ASSERT_INT_EQ(x86_run_rewritten_direct(ctx, sink, args, 3), 0);
  ASSERT_FLOAT_EQ(out[0], 1.0f, 0.0f);
  ASSERT_FLOAT_EQ(out[1], 2.0f, 0.0f);
  ASSERT_FLOAT_EQ(out[2], 5.0f, 0.0f);
  ASSERT_FLOAT_EQ(out[3], 6.0f, 0.0f);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, rewritten_f64_stack_index_lanes_uses_tinygrad_vshufpd) {
  double a[2] = {1.0, 2.0};
  double b[2] = {5.0, 6.0};
  double out[2] = {0};
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = x86_make_f64_shuffle2(ctx);
  void *args[3] = {a, b, out};
  ASSERT_INT_EQ(x86_run_rewritten_direct(ctx, sink, args, 3), 0);
  ASSERT_FLOAT_EQ((float)out[0], 2.0f, 0.0f);
  ASSERT_FLOAT_EQ((float)out[1], 5.0f, 0.0f);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, rewritten_f32_stack_unmatched_index_lanes_uses_tinygrad_vinsertps) {
  float a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float b[4] = {5.0f, 6.0f, 7.0f, 8.0f};
  float c[4] = {9.0f, 10.0f, 11.0f, 12.0f};
  float d[4] = {13.0f, 14.0f, 15.0f, 16.0f};
  float out[4] = {0};
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = x86_make_f32_insert4(ctx);
  void *args[5] = {a, b, c, d, out};
  ASSERT_INT_EQ(x86_run_rewritten_direct(ctx, sink, args, 5), 0);
  ASSERT_FLOAT_EQ(out[0], 1.0f, 0.0f);
  ASSERT_FLOAT_EQ(out[1], 6.0f, 0.0f);
  ASSERT_FLOAT_EQ(out[2], 11.0f, 0.0f);
  ASSERT_FLOAT_EQ(out[3], 16.0f, 0.0f);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, rewritten_f32_vector_index_extract_uses_tinygrad_vpsrldq) {
  float f32[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float f32_out[1] = {0};
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = x86_make_index_extract(ctx, POLY_FLOAT32, 4, 2);
  /* tinygrad CPU:X86 direct probe:
   *   X86Ops.VPSRLDQ(xmm0, imm8=8) -> c5f973d808
   */
  ASSERT_TRUE(x86_rewritten_code_contains_hex(ctx, sink, "c5f973d808"));
  void *args[2] = {f32, f32_out};
  ASSERT_INT_EQ(x86_run_rewritten_direct(ctx, sink, args, 2), 0);
  ASSERT_FLOAT_EQ(f32_out[0], 3.0f, 0.0f);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, rewritten_f64_vector_index_extract_uses_tinygrad_vpsrldq) {
  double f64[2] = {5.0, 6.0};
  double f64_out[1] = {0};
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = x86_make_index_extract(ctx, POLY_FLOAT64, 2, 1);
  void *args[2] = {f64, f64_out};
  ASSERT_INT_EQ(x86_run_rewritten_direct(ctx, sink, args, 2), 0);
  ASSERT_FLOAT_EQ((float)f64_out[0], 6.0f, 0.0f);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, rewritten_int_vector_index_extract_uses_tinygrad_vpextr_family) {
  int8_t i8[16] = {0, 1, 2, -3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  int8_t i8_out[1] = {0};
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = x86_make_index_extract(ctx, POLY_INT8, 16, 3);
  void *args[2] = {i8, i8_out};
  ASSERT_INT_EQ(x86_run_rewritten_direct(ctx, sink, args, 2), 0);
  ASSERT_INT_EQ(i8_out[0], -3);
  poly_ctx_destroy(ctx);

  int16_t i16[8] = {0, 1, -2, 3, 4, 5, 6, 7};
  int16_t i16_out[1] = {0};
  ctx = poly_ctx_new();
  sink = x86_make_index_extract(ctx, POLY_INT16, 8, 2);
  args[0] = i16;
  args[1] = i16_out;
  ASSERT_INT_EQ(x86_run_rewritten_direct(ctx, sink, args, 2), 0);
  ASSERT_INT_EQ(i16_out[0], -2);
  poly_ctx_destroy(ctx);

  int32_t i32[4] = {11, -22, 33, 44};
  int32_t i32_out[1] = {0};
  ctx = poly_ctx_new();
  sink = x86_make_index_extract(ctx, POLY_INT32, 4, 1);
  args[0] = i32;
  args[1] = i32_out;
  ASSERT_INT_EQ(x86_run_rewritten_direct(ctx, sink, args, 2), 0);
  ASSERT_INT_EQ(i32_out[0], -22);
  poly_ctx_destroy(ctx);

  int64_t i64[2] = {123, -456};
  int64_t i64_out[1] = {0};
  ctx = poly_ctx_new();
  sink = x86_make_index_extract(ctx, POLY_INT64, 2, 1);
  args[0] = i64;
  args[1] = i64_out;
  ASSERT_INT_EQ(x86_run_rewritten_direct(ctx, sink, args, 2), 0);
  ASSERT_INT_EQ((int)i64_out[0], -456);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, rewritten_int_stack_scalar_lanes_uses_tinygrad_vpins_family) {
  int8_t i8[4] = {1, -2, 3, -4};
  int8_t i8_out[4] = {0};
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sink = x86_make_int_stack_from_scalar_lanes(ctx, POLY_INT8, 4);
  void *args[2] = {i8, i8_out};
  ASSERT_INT_EQ(x86_run_rewritten_direct(ctx, sink, args, 2), 0);
  for (int i = 0; i < 4; i++)
    ASSERT_INT_EQ(i8_out[i], i8[i]);
  poly_ctx_destroy(ctx);

  int16_t i16[4] = {5, -6, 7, -8};
  int16_t i16_out[4] = {0};
  ctx = poly_ctx_new();
  sink = x86_make_int_stack_from_scalar_lanes(ctx, POLY_INT16, 4);
  args[0] = i16;
  args[1] = i16_out;
  ASSERT_INT_EQ(x86_run_rewritten_direct(ctx, sink, args, 2), 0);
  for (int i = 0; i < 4; i++)
    ASSERT_INT_EQ(i16_out[i], i16[i]);
  poly_ctx_destroy(ctx);

  int32_t i32[4] = {9, -10, 11, -12};
  int32_t i32_out[4] = {0};
  ctx = poly_ctx_new();
  sink = x86_make_int_stack_from_scalar_lanes(ctx, POLY_INT32, 4);
  args[0] = i32;
  args[1] = i32_out;
  ASSERT_INT_EQ(x86_run_rewritten_direct(ctx, sink, args, 2), 0);
  for (int i = 0; i < 4; i++)
    ASSERT_INT_EQ(i32_out[i], i32[i]);
  poly_ctx_destroy(ctx);

  int64_t i64[2] = {13, -14};
  int64_t i64_out[2] = {0};
  ctx = poly_ctx_new();
  sink = x86_make_int_stack_from_scalar_lanes(ctx, POLY_INT64, 2);
  args[0] = i64;
  args[1] = i64_out;
  ASSERT_INT_EQ(x86_run_rewritten_direct(ctx, sink, args, 2), 0);
  for (int i = 0; i < 2; i++)
    ASSERT_INT_EQ((int)i64_out[i], (int)i64[i]);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, direct_f32_unary_div_matches_tinygrad_probe_class) {
  enum { N = 8 };
  float f[N] = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f, 49.0f, 64.0f};
  float g[N] = {2.0f, -3.0f, 4.0f, -5.0f, 6.0f, -7.0f, 8.0f, -9.0f};
  float h[N] = {1.25f, -2.75f, 3.5f, -4.125f, 5.875f, -6.25f, 7.75f, -8.5f};
  float out[N] = {0};
  const float expected[N] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  PolyCtx *ctx = poly_ctx_new();
  PolyDType pf = poly_dtype_ptr(POLY_FLOAT32, -1, POLY_ADDR_GLOBAL);
  PolyUOp *sink = x86_make_three_input_range(ctx, pf, POLY_FLOAT32, N, x86_expr_f32_unary_div);
  void *args[4] = {f, g, h, out};
  ASSERT_INT_EQ(x86_run_direct(ctx, sink, args, 4), 0);
  for (int i = 0; i < N; i++)
    ASSERT_FLOAT_EQ(out[i], expected[i], 1e-6);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, direct_f64_unary_div_matches_tinygrad_probe_class) {
  enum { N = 4 };
  double d[N] = {1.5, 4.0, 9.0, 16.0};
  double e[N] = {0.5, -2.0, 3.0, -4.0};
  double out[N] = {0};
  const double expected[N] = {4.224744871391589, 0.0, 6.0, 0.0};
  PolyCtx *ctx = poly_ctx_new();
  PolyDType pd = poly_dtype_ptr(POLY_FLOAT64, -1, POLY_ADDR_GLOBAL);
  PolyUOp *sink = x86_make_one_range(ctx, pd, POLY_FLOAT64, N, x86_expr_f64_unary_div);
  void *args[3] = {d, e, out};
  ASSERT_INT_EQ(x86_run_direct(ctx, sink, args, 3), 0);
  for (int i = 0; i < N; i++)
    ASSERT_FLOAT_EQ(out[i], expected[i], 1e-9);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, direct_i64_mix_matches_tinygrad_probe_class) {
  enum { N = 4 };
  int64_t a[N] = {1, -2, 3, -4};
  int64_t b[N] = {8, 7, -6, -5};
  int64_t out[N] = {0};
  const int64_t expected[N] = {31, -9, -5, 21};
  PolyCtx *ctx = poly_ctx_new();
  PolyDType pi = poly_dtype_ptr(POLY_INT64, -1, POLY_ADDR_GLOBAL);
  PolyUOp *sink = x86_make_one_range(ctx, pi, POLY_INT64, N, x86_expr_i64_mix);
  void *args[3] = {a, b, out};
  ASSERT_INT_EQ(x86_run_direct(ctx, sink, args, 3), 0);
  for (int i = 0; i < N; i++)
    ASSERT_INT_EQ(out[i], expected[i]);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, direct_i64_where_matches_tinygrad_probe_class) {
  enum { N = 4 };
  int64_t a[N] = {1, -2, 3, -4};
  int64_t b[N] = {8, 7, -6, -5};
  int64_t out[N] = {0};
  const int64_t expected[N] = {1, -2, -6, -5};
  PolyCtx *ctx = poly_ctx_new();
  PolyDType pi = poly_dtype_ptr(POLY_INT64, -1, POLY_ADDR_GLOBAL);
  PolyUOp *sink = x86_make_one_range(ctx, pi, POLY_INT64, N, x86_expr_i64_where);
  void *args[3] = {a, b, out};
  ASSERT_INT_EQ(x86_run_direct(ctx, sink, args, 3), 0);
  for (int i = 0; i < N; i++)
    ASSERT_INT_EQ(out[i], expected[i]);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, direct_u32_where_matches_tinygrad_probe_class) {
  enum { N = 4 };
  uint32_t a[N] = {1, 2, 3, 4};
  uint32_t b[N] = {4, 3, 2, 1};
  uint32_t out[N] = {0};
  const uint32_t expected[N] = {5, 5, 1, 3};
  PolyCtx *ctx = poly_ctx_new();
  PolyDType pu = poly_dtype_ptr(POLY_UINT32, -1, POLY_ADDR_GLOBAL);
  PolyUOp *sink = x86_make_one_range(ctx, pu, POLY_UINT32, N, x86_expr_u32_where);
  void *args[3] = {a, b, out};
  ASSERT_INT_EQ(x86_run_direct(ctx, sink, args, 3), 0);
  for (int i = 0; i < N; i++)
    ASSERT_INT_EQ((int)out[i], (int)expected[i]);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, direct_u64_cdiv_matches_tinygrad_scalar_idiv_path) {
  enum { N = 4 };
  uint64_t a[N] = {10, 100, 0x123456789abcdef0ULL, 0xfffffffffffffff0ULL};
  uint64_t b[N] = {2, 7, 0x12345, 0x1000};
  uint64_t out[N] = {0};
  PolyCtx *ctx = poly_ctx_new();
  PolyDType pu = poly_dtype_ptr(POLY_UINT64, -1, POLY_ADDR_GLOBAL);
  PolyUOp *sink = x86_make_one_range(ctx, pu, POLY_UINT64, N, x86_expr_u64_cdiv);
  void *args[3] = {a, b, out};
  ASSERT_INT_EQ(x86_run_direct(ctx, sink, args, 3), 0);
  for (int i = 0; i < N; i++)
    ASSERT_INT_EQ((int64_t)out[i], (int64_t)(a[i] / b[i]));
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, rewritten_u64_cdiv_all_ones_matches_tinygrad_unsigned_semantics) {
  enum { N = 4 };
  uint64_t a[N] = {1, 2, 3, UINT64_MAX - 1};
  uint64_t unused[N] = {0};
  uint64_t out[N] = {UINT64_MAX, UINT64_MAX, UINT64_MAX, UINT64_MAX};
  PolyCtx *ctx = poly_ctx_new();
  PolyDType pu = poly_dtype_ptr(POLY_UINT64, -1, POLY_ADDR_GLOBAL);
  PolyUOp *sink = x86_make_one_range(ctx, pu, POLY_UINT64, N, x86_expr_u64_cdiv_all_ones);
  PolyUOp *rewritten = poly_rewrite_x86(ctx, sink);
  ASSERT_NOT_NULL(rewritten);
  void *args[3] = {a, unused, out};
  ASSERT_INT_EQ(x86_run_rewritten_direct(ctx, rewritten, args, 3), 0);
  for (int i = 0; i < N; i++)
    ASSERT_INT_EQ((int64_t)out[i], 0);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, direct_i64_cdiv_matches_tinygrad_scalar_idiv_path) {
  enum { N = 4 };
  int64_t a[N] = {10, -100, 0x123456789LL, -0x123456789LL};
  int64_t b[N] = {2, 7, -12345, -4096};
  int64_t out[N] = {0};
  PolyCtx *ctx = poly_ctx_new();
  PolyDType pi = poly_dtype_ptr(POLY_INT64, -1, POLY_ADDR_GLOBAL);
  PolyUOp *sink = x86_make_one_range(ctx, pi, POLY_INT64, N, x86_expr_i64_cdiv);
  void *args[3] = {a, b, out};
  ASSERT_INT_EQ(x86_run_direct(ctx, sink, args, 3), 0);
  for (int i = 0; i < N; i++)
    ASSERT_INT_EQ(out[i], a[i] / b[i]);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, direct_u8_cdiv_matches_tinygrad_movzx_div_path) {
  enum { N = 4 };
  uint8_t a[N] = {10, 100, 255, 128};
  uint8_t b[N] = {2, 7, 15, 3};
  uint8_t out[N] = {0};
  PolyCtx *ctx = poly_ctx_new();
  PolyDType pu = poly_dtype_ptr(POLY_UINT8, -1, POLY_ADDR_GLOBAL);
  PolyUOp *sink = x86_make_one_range(ctx, pu, POLY_UINT8, N, x86_expr_u8_cdiv);
  void *args[3] = {a, b, out};
  ASSERT_INT_EQ(x86_run_direct(ctx, sink, args, 3), 0);
  for (int i = 0; i < N; i++)
    ASSERT_INT_EQ((int)out[i], (int)(uint8_t)(a[i] / b[i]));
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, direct_i8_cdiv_matches_tinygrad_movsx_idiv_path) {
  enum { N = 4 };
  int8_t a[N] = {10, -100, 99, -128};
  int8_t b[N] = {2, 7, -9, -8};
  int8_t out[N] = {0};
  PolyCtx *ctx = poly_ctx_new();
  PolyDType pi = poly_dtype_ptr(POLY_INT8, -1, POLY_ADDR_GLOBAL);
  PolyUOp *sink = x86_make_one_range(ctx, pi, POLY_INT8, N, x86_expr_i8_cdiv);
  void *args[3] = {a, b, out};
  ASSERT_INT_EQ(x86_run_direct(ctx, sink, args, 3), 0);
  for (int i = 0; i < N; i++)
    ASSERT_INT_EQ(out[i], (int8_t)(a[i] / b[i]));
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, direct_i8_mul_and_where_match_tinygrad_extra_matcher) {
  enum { N = 4 };
  int8_t a[N] = {2, -3, 4, -5};
  int8_t b[N] = {3, 4, -5, -6};
  int8_t out_mul[N] = {0};
  int8_t out_where[N] = {0};
  const int8_t expected_mul[N] = {6, -12, -20, 30};
  const int8_t expected_where[N] = {2, -3, -5, -6};
  PolyCtx *ctx = poly_ctx_new();
  PolyDType pi8 = poly_dtype_ptr(POLY_INT8, -1, POLY_ADDR_GLOBAL);
  PolyUOp *mul_sink = x86_make_one_range(ctx, pi8, POLY_INT8, N, x86_expr_i8_mul);
  void *mul_args[3] = {a, b, out_mul};
  ASSERT_INT_EQ(x86_run_direct(ctx, mul_sink, mul_args, 3), 0);
  for (int i = 0; i < N; i++)
    ASSERT_INT_EQ(out_mul[i], expected_mul[i]);

  PolyUOp *where_sink = x86_make_one_range(ctx, pi8, POLY_INT8, N, x86_expr_i8_where);
  void *where_args[3] = {a, b, out_where};
  ASSERT_INT_EQ(x86_run_direct(ctx, where_sink, where_args, 3), 0);
  for (int i = 0; i < N; i++)
    ASSERT_INT_EQ(out_where[i], expected_where[i]);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, direct_f16_alu_and_where_match_tinygrad_extra_matcher) {
  enum { N = 4 };
  float vals[N] = {1.0f, -2.0f, 3.0f, -4.0f};
  uint16_t a[N], out_alu[N], out_where[N];
  for (int i = 0; i < N; i++) {
    a[i] = x86_f32_to_f16_bits(vals[i]);
    out_alu[i] = out_where[i] = 0;
  }
  const float expected_alu[N] = {2.0f, 2.0f, 12.0f, 12.0f};
  const float expected_where[N] = {11.0f, -2.0f, 13.0f, -4.0f};
  PolyCtx *ctx = poly_ctx_new();
  PolyDType pf16 = poly_dtype_ptr(POLY_FLOAT16, -1, POLY_ADDR_GLOBAL);
  PolyUOp *alu_sink = x86_make_one_range(ctx, pf16, POLY_FLOAT16, N, x86_expr_f16_alu);
  void *alu_args[3] = {a, a, out_alu};
  ASSERT_INT_EQ(x86_run_direct(ctx, alu_sink, alu_args, 3), 0);
  for (int i = 0; i < N; i++)
    ASSERT_FLOAT_EQ(x86_f16_bits_to_f32(out_alu[i]), expected_alu[i], 0.0f);

  PolyUOp *where_sink = x86_make_one_range(ctx, pf16, POLY_FLOAT16, N, x86_expr_f16_where);
  void *where_args[3] = {a, a, out_where};
  ASSERT_INT_EQ(x86_run_direct(ctx, where_sink, where_args, 3), 0);
  for (int i = 0; i < N; i++)
    ASSERT_FLOAT_EQ(x86_f16_bits_to_f32(out_where[i]), expected_where[i], 0.0f);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, f16_exp2_decomposes_before_extra_matcher_like_tinygrad) {
  /* Pinned tinygrad uop/decompositions.py:xexp2 accepts float16 and builds the
   * exponent with int16 before renderer/isa/x86.py promotes each half ALU to
   * float32. A raw EXP2 must therefore never reach X86 graph isel. */
  enum { N = 13 };
  const float values[N] = {
      -INFINITY, -23.0f, -22.5f, -22.0f, -21.5f, 0.0f, 15.0f,
      15.5f, 16.0f, 22.5f, 23.0f, INFINITY, NAN,
  };
  const float expected[N] = {
      0.0f, 0.0f, 0.0f, 2.384185791015625e-7f, 3.5762786865234375e-7f,
      1.0f, 32768.0f, 46336.0f, INFINITY, INFINITY, INFINITY, INFINITY, NAN,
  };
  uint16_t input[N], output[N];
  for (int i = 0; i < N; i++) {
    input[i] = x86_f32_to_f16_bits(values[i]);
    output[i] = 0;
  }
  input[N - 1] = 0x7e00u;

  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *param =
      poly_uop0(ctx, POLY_OP_PARAM, POLY_FLOAT16, poly_arg_int(0));
  PolyUOp *raw =
      poly_uop1(ctx, POLY_OP_EXP2, POLY_FLOAT16, param, poly_arg_none());
  PolyUOp *transcendental =
      poly_graph_rewrite(ctx, raw, poly_pm_transcendental_pass());
  ASSERT_NOT_NULL(transcendental);
  int n_transcendental = 0;
  ASSERT_TRUE(
      x86_topology_signature(ctx, transcendental, &n_transcendental) ==
      UINT64_C(0x293cde36cf9280e7)
  );
  ASSERT_INT_EQ(n_transcendental, 62);
  int n_transcendental_topo = 0;
  int floor_div_short = 0, early_cdiv = 0, early_cmod = 0;
  PolyUOp **transcendental_topo =
      poly_toposort_alloc(ctx, transcendental, &n_transcendental_topo);
  ASSERT_NOT_NULL(transcendental_topo);
  for (int i = 0; i < n_transcendental_topo; i++) {
    PolyUOp *u = transcendental_topo[i];
    if (u->op == POLY_OP_FLOORDIV && poly_dtype_eq(u->dtype, POLY_INT16))
      floor_div_short++;
    if (u->op == POLY_OP_CDIV) early_cdiv++;
    if (u->op == POLY_OP_CMOD) early_cmod++;
  }
  ASSERT_INT_EQ(floor_div_short, 1);
  ASSERT_INT_EQ(early_cdiv, 0);
  ASSERT_INT_EQ(early_cmod, 0);
  poly_toposort_free(transcendental_topo);

  PolyDType pf16 = poly_dtype_ptr(POLY_FLOAT16, -1, POLY_ADDR_GLOBAL);
  PolyUOp *sink = x86_make_one_range(ctx, pf16, POLY_FLOAT16, N, x86_expr_f16_exp2);
  PolyUOp *rewritten = poly_rewrite_x86(ctx, sink);
  ASSERT_NOT_NULL(rewritten);
  int n_rewritten = 0;
  ASSERT_TRUE(
      x86_topology_signature(ctx, rewritten, &n_rewritten) ==
      UINT64_C(0x03985a50393c78d6)
  );
  ASSERT_INT_EQ(n_rewritten, 133);

  int n_topo = 0, raw_exp2 = 0, half_from_short_bitcasts = 0, half_alu = 0;
  PolyUOp **topo = poly_toposort_alloc(ctx, rewritten, &n_topo);
  ASSERT_NOT_NULL(topo);
  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    if (u->op == POLY_OP_EXP2) raw_exp2++;
    if (poly_opset_has(POLY_GROUP_ALU, u->op) &&
        poly_dtype_eq(poly_dtype_scalar(u->dtype), POLY_FLOAT16))
      half_alu++;
    if (u->op == POLY_OP_BITCAST && poly_dtype_eq(u->dtype, POLY_FLOAT16) &&
        u->n_src == 1 && poly_dtype_eq(u->src[0]->dtype, POLY_INT16))
      half_from_short_bitcasts++;
  }
  ASSERT_INT_EQ(raw_exp2, 0);
  ASSERT_INT_EQ(half_from_short_bitcasts, 2);
  ASSERT_INT_EQ(half_alu, 0);
  poly_toposort_free(topo);

  void *args[3] = {input, input, output};
  ASSERT_INT_EQ(x86_run_rewritten_direct(ctx, rewritten, args, 3), 0);
  for (int i = 0; i < N; i++) {
    float got = x86_f16_bits_to_f32(output[i]);
    if (isnan(expected[i]))
      ASSERT_TRUE(isnan(got));
    else if (isinf(expected[i]))
      ASSERT_TRUE(isinf(got) && !signbit(got));
    else
      ASSERT_FLOAT_EQ(got, expected[i], 0.0f);
  }

  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, rewrite_emulates_unsupported_bf16_before_instruction_selection) {
  PolyCtx *ctx = poly_ctx_new();
  PolyDType pbf16 = poly_dtype_ptr(POLY_BFLOAT16, -1, POLY_ADDR_GLOBAL);
  PolyUOp *sink = x86_make_one_range(ctx, pbf16, POLY_BFLOAT16, 4, x86_expr_bf16_add);
  PolyUOp *rewritten = poly_rewrite_x86(ctx, sink);
  ASSERT_NOT_NULL(rewritten);

  int n_topo = 0, f32_adds = 0, bitwise_ops = 0;
  PolyUOp **topo = poly_toposort(ctx, rewritten, &n_topo);
  ASSERT_NOT_NULL(topo);
  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    PolyDType scalar = poly_dtype_scalar(u->dtype);
    if (poly_opset_has(POLY_GROUP_ALU, u->op))
      ASSERT_FALSE(poly_dtype_eq(scalar, POLY_BFLOAT16));
    if (u->op == POLY_OP_CAST) {
      ASSERT_FALSE(poly_dtype_eq(scalar, POLY_BFLOAT16));
      if (u->n_src > 0)
        ASSERT_FALSE(poly_dtype_eq(poly_dtype_scalar(u->src[0]->dtype), POLY_BFLOAT16));
    }
    if (u->op == POLY_OP_ADD && poly_dtype_eq(scalar, POLY_FLOAT32)) f32_adds++;
    if (u->op == POLY_OP_BITCAST || u->op == POLY_OP_SHL || u->op == POLY_OP_SHR) bitwise_ops++;
  }
  ASSERT_TRUE(f32_adds > 0);
  ASSERT_TRUE(bitwise_ops > 0);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, scalar_f16_spills_use_typed_16bit_memory_ops) {
  /* tinygrad x86.py:881-889 sends spills through typed STORE/LOAD isel;
   * scalar F16 therefore uses VPEXTRW/VPINSRW with a two-byte slot. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyDType pf16 = poly_dtype_ptr(POLY_FLOAT16, -1, POLY_ADDR_GLOBAL);
  PolyDType pbf16 = poly_dtype_ptr(POLY_BFLOAT16, -1, POLY_ADDR_GLOBAL);
  PolyUOp *f16_in = poly_uop0(ctx, POLY_OP_PARAM, pf16, poly_arg_int(0));
  PolyUOp *bf16_out = poly_uop0(ctx, POLY_OP_PARAM, pbf16, poly_arg_int(1));
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(5));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));
  PolyUOp *f16_index = poly_uop2(ctx, POLY_OP_INDEX, pf16, f16_in, range, poly_arg_none());
  PolyUOp *bf16_index = poly_uop2(ctx, POLY_OP_INDEX, pbf16, bf16_out, range, poly_arg_none());
  PolyUOp *f16_value = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT16, f16_index, poly_arg_none());
  PolyUOp *to_bf16 = poly_uop1(ctx, POLY_OP_BITCAST, POLY_BFLOAT16, f16_value, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, bf16_index, to_bf16, poly_arg_none());
  PolyUOp *end_srcs[2] = {store, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_srcs, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());

  int n_lin = 0;
  PolyUOp **lin = poly_linearize_x86_rewritten(ctx, sink, &n_lin);
  ASSERT_NOT_NULL(lin);
  ASSERT_TRUE(tx86_count_stack_memory_op(lin, n_lin, TX86_OP_VPEXTRW, 0, 2) > 0);
  ASSERT_TRUE(tx86_count_stack_memory_op(lin, n_lin, TX86_OP_VPINSRW, 1, 2) > 0);
  ASSERT_INT_EQ(tx86_count_stack_memory_op(lin, n_lin, TX86_OP_VMOVSSm, 0, 2), 0);
  ASSERT_INT_EQ(tx86_count_stack_memory_op(lin, n_lin, TX86_OP_VMOVSS, 0, 2), 0);

  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, direct_unsigned_to_float_casts_match_tinygrad_extra_matcher) {
  enum { N = 4 };
  uint32_t u32[N] = {1u, 0x00020003u, 0x7fffffffu, 0xfedcba98u};
  uint64_t u64[N] = {1, 2, 3, 4};
  float out32[N] = {0};
  float out64[N] = {0};
  PolyCtx *ctx = poly_ctx_new();
  PolyDType pu32 = poly_dtype_ptr(POLY_UINT32, -1, POLY_ADDR_GLOBAL);
  PolyDType pu64 = poly_dtype_ptr(POLY_UINT64, -1, POLY_ADDR_GLOBAL);
  PolyDType pf = poly_dtype_ptr(POLY_FLOAT32, -1, POLY_ADDR_GLOBAL);

  PolyUOp *u32_sink =
      x86_make_unary_cast_range(ctx, pu32, POLY_UINT32, pf, POLY_FLOAT32, POLY_OP_CAST, N);
  void *args32[2] = {u32, out32};
  ASSERT_INT_EQ(x86_run_direct(ctx, u32_sink, args32, 2), 0);
  for (int i = 0; i < N; i++)
    ASSERT_FLOAT_EQ(out32[i], (float)u32[i], 0.0f);

  PolyUOp *u64_sink =
      x86_make_unary_cast_range(ctx, pu64, POLY_UINT64, pf, POLY_FLOAT32, POLY_OP_CAST, N);
  void *args64[2] = {u64, out64};
  ASSERT_INT_EQ(x86_run_direct(ctx, u64_sink, args64, 2), 0);
  for (int i = 0; i < N; i++)
    ASSERT_FLOAT_EQ(out64[i], (float)u64[i], 0.0f);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, direct_bitcast_u32_f32_matches_tinygrad_probe_class) {
  enum { N = 4 };
  uint32_t bits[N] = {0x3f800000u, 0x40000000u, 0x40400000u, 0x40800000u};
  float out[N] = {0};
  const float expected[N] = {1.0f, 2.0f, 3.0f, 4.0f};
  PolyCtx *ctx = poly_ctx_new();
  PolyDType pu = poly_dtype_ptr(POLY_UINT32, -1, POLY_ADDR_GLOBAL);
  PolyDType pf = poly_dtype_ptr(POLY_FLOAT32, -1, POLY_ADDR_GLOBAL);
  PolyUOp *sink =
      x86_make_unary_cast_range(ctx, pu, POLY_UINT32, pf, POLY_FLOAT32, POLY_OP_BITCAST, N);
  void *args[2] = {bits, out};
  ASSERT_INT_EQ(x86_run_direct(ctx, sink, args, 2), 0);
  for (int i = 0; i < N; i++)
    ASSERT_FLOAT_EQ(out[i], expected[i], 0.0f);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, direct_bitcast_f32_i32_matches_tinygrad_probe_class) {
  enum { N = 4 };
  float vals[N] = {1.0f, -2.0f, 3.5f, -4.25f};
  int32_t out[N] = {0};
  const int32_t expected[N] = {1065353216, -1073741824, 1080033280, -1064828928};
  PolyCtx *ctx = poly_ctx_new();
  PolyDType pf = poly_dtype_ptr(POLY_FLOAT32, -1, POLY_ADDR_GLOBAL);
  PolyDType pi = poly_dtype_ptr(POLY_INT32, -1, POLY_ADDR_GLOBAL);
  PolyUOp *sink =
      x86_make_unary_cast_range(ctx, pf, POLY_FLOAT32, pi, POLY_INT32, POLY_OP_BITCAST, N);
  void *args[2] = {vals, out};
  ASSERT_INT_EQ(x86_run_direct(ctx, sink, args, 2), 0);
  for (int i = 0; i < N; i++)
    ASSERT_INT_EQ(out[i], expected[i]);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, to_program_attaches_linear_source_hex_and_binary_children) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  poly_program_source_render_count_reset();

  PolyUOp *a = poly_buffer_f32(ctx, 8);
  PolyUOp *b = poly_buffer_f32(ctx, 8);
  PolyUOp *out = poly_buffer_f32(ctx, 8);
  PolyUOp *sum = poly_alu2(ctx, POLY_OP_ADD, a, b);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, sum));

  PolySchedule *sched = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_NOT_NULL(sched);
  ASSERT_INT_EQ(sched->template->n_calls, 1);

  PolyUOp *program = poly_schedule_call_to_program(ctx, sched, 0, POLY_DEVICE_X86);
  ASSERT_NOT_NULL(program);
  ASSERT_INT_EQ(program->op, POLY_OP_PROGRAM);
  ASSERT_INT_EQ(program->arg.kind, POLY_ARG_PROGRAM_INFO);
  ASSERT_TRUE(program->n_src >= 5);
  ASSERT_NOT_NULL(poly_program_linear(program));
  ASSERT_INT_EQ(program->src[3]->op, POLY_OP_SOURCE);
  ASSERT_INT_EQ(program->src[3]->arg.kind, POLY_ARG_STRING);
  ASSERT_NOT_NULL(program->src[3]->arg.str);
  ASSERT_INT_EQ(program->src[4]->op, POLY_OP_BINARY);
  ASSERT_INT_EQ(program->src[4]->arg.kind, POLY_ARG_BYTES);
  ASSERT_NOT_NULL(program->src[4]->arg.bytes.data);
  ASSERT_TRUE(program->src[4]->arg.bytes.n > 0);

  const char *hex = program->src[3]->arg.str;
  int n_hex = (int)strlen(hex);
  ASSERT_INT_EQ(n_hex, program->src[4]->arg.bytes.n * 2);
  for (int i = 0; i < program->src[4]->arg.bytes.n; i++) {
    int hi = x86_hex_nibble(hex[2 * i]);
    int lo = x86_hex_nibble(hex[2 * i + 1]);
    ASSERT_TRUE(hi >= 0 && lo >= 0);
    ASSERT_INT_EQ((hi << 4) | lo, program->src[4]->arg.bytes.data[i]);
  }
  ASSERT_INT_EQ(poly_program_source_render_count(), 1);

  PolyUOp *again = poly_schedule_call_to_program(ctx, sched, 0, POLY_DEVICE_X86);
  ASSERT_PTR_EQ(again, program);
  ASSERT_INT_EQ((int)poly_to_program_cache_len(ctx), 1);
  ASSERT_INT_EQ(poly_program_source_render_count(), 1);

  poly_schedule_free(sched);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, to_program_allows_f16_after_x86_extra_legalization) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT16, 4);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT16, 4);
  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT16, a, a, poly_arg_none());
  PolyUOp *sum = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT16, mul, a, poly_arg_none());
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, sum));

  PolySchedule *sched = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_NOT_NULL(sched);
  ASSERT_INT_EQ(sched->template->n_calls, 1);
  PolyUOp *program = poly_schedule_call_to_program(ctx, sched, 0, POLY_DEVICE_X86);
  ASSERT_NOT_NULL(program);
  ASSERT_TRUE(program->n_src >= 5);
  ASSERT_INT_EQ(program->src[3]->op, POLY_OP_SOURCE);
  ASSERT_INT_EQ(program->src[4]->op, POLY_OP_BINARY);

  poly_schedule_free(sched);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, schedule_runtime_vecadd_uses_x86_device) {
  enum { N = 8 };
  float a_data[N], b_data[N], out[N];
  for (int i = 0; i < N; i++) {
    a_data[i] = (float)i - 3.0f;
    b_data[i] = 0.25f * (float)i;
  }
  PolyCtx *ctx = poly_ctx_new();
  poly_ctx_set_preferred_device(ctx, POLY_DEVICE_X86);
  PolyUOp *a = poly_buffer_f32(ctx, N);
  PolyUOp *b = poly_buffer_f32(ctx, N);
  poly_buffer_set(ctx, a, a_data, sizeof(a_data), POLY_DEVICE_CPU);
  poly_buffer_set(ctx, b, b_data, sizeof(b_data), POLY_DEVICE_CPU);
  PolyUOp *sum = poly_alu2(ctx, POLY_OP_ADD, a, b);
  PolyUOp *realized = NULL;
  ASSERT_INT_EQ(poly_realize_uops(ctx, &sum, 1, &realized), 0);
  ASSERT_NOT_NULL(realized);
  PolyUOp *buf = (PolyUOp *)poly_uop_get_buffer_identity(realized);
  ASSERT_NOT_NULL(buf);
  ASSERT_INT_EQ(poly_buffer_read(ctx, buf, out, sizeof(out)), 0);
  for (int i = 0; i < N; i++)
    ASSERT_FLOAT_EQ(out[i], a_data[i] + b_data[i], 1e-6);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, schedule_runtime_reduce_sum_axis1_matches_tinygrad_probe_class) {
  enum { N = 16, OUT = 4 };
  float a_data[N], out[OUT];
  for (int i = 0; i < N; i++)
    a_data[i] = (float)i;
  PolyCtx *ctx = poly_ctx_new();
  poly_ctx_set_preferred_device(ctx, POLY_DEVICE_X86);
  PolyUOp *a = poly_buffer_f32(ctx, N);
  poly_buffer_set(ctx, a, a_data, sizeof(a_data), POLY_DEVICE_CPU);
  int64_t shape[] = {4, 4};
  PolyUOp *a2d = poly_reshape(ctx, a, shape, 2);
  int64_t axes[] = {1};
  PolyUOp *sum = poly_reduce_axis(ctx, POLY_OP_ADD, a2d, axes, 1);
  PolyUOp *realized = NULL;
  ASSERT_INT_EQ(poly_realize_uops(ctx, &sum, 1, &realized), 0);
  ASSERT_NOT_NULL(realized);
  PolyUOp *buf = (PolyUOp *)poly_uop_get_buffer_identity(realized);
  ASSERT_NOT_NULL(buf);
  ASSERT_INT_EQ(poly_buffer_read(ctx, buf, out, sizeof(out)), 0);
  const float expected[OUT] = {6.0f, 22.0f, 38.0f, 54.0f};
  for (int i = 0; i < OUT; i++)
    ASSERT_FLOAT_EQ(out[i], expected[i], 1e-5);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, schedule_runtime_dot_matches_tinygrad_probe_class) {
  enum { XN = 8, WN = 8, OUT = 4 };
  float x_data[XN] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  float w_data[WN] = {0.5f, 1.0f, 1.5f, 2.0f, 2.0f, 1.5f, 1.0f, 0.5f};
  float out[OUT] = {0};
  PolyCtx *ctx = poly_ctx_new();
  poly_ctx_set_preferred_device(ctx, POLY_DEVICE_X86);
  PolyUOp *xb = poly_buffer_f32(ctx, XN);
  PolyUOp *wb = poly_buffer_f32(ctx, WN);
  poly_buffer_set(ctx, xb, x_data, sizeof(x_data), POLY_DEVICE_CPU);
  poly_buffer_set(ctx, wb, w_data, sizeof(w_data), POLY_DEVICE_CPU);
  PolyUOp *x = poly_reshape(ctx, xb, (int64_t[]){2, 4}, 2);
  PolyUOp *w = poly_reshape(ctx, wb, (int64_t[]){2, 4}, 2);
  PolyUOp *wt = poly_permute(ctx, w, (int64_t[]){1, 0}, 2);
  PolyUOp *dot = poly_dot(ctx, x, wt);
  ASSERT_NOT_NULL(dot);
  PolyUOp *realized = NULL;
  ASSERT_INT_EQ(poly_realize_uops(ctx, &dot, 1, &realized), 0);
  ASSERT_NOT_NULL(realized);
  PolyUOp *buf = (PolyUOp *)poly_uop_get_buffer_identity(realized);
  ASSERT_NOT_NULL(buf);
  ASSERT_INT_EQ(poly_buffer_read(ctx, buf, out, sizeof(out)), 0);
  const float expected[OUT] = {15.0f, 10.0f, 35.0f, 30.0f};
  for (int i = 0; i < OUT; i++)
    ASSERT_FLOAT_EQ(out[i], expected[i], 1e-4);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, threaded_vecadd_program_core_id_shards_match_tinygrad_cpu_x86) {
  enum { N = 262144 };
  float *a = malloc((size_t)N * sizeof(float));
  float *b = malloc((size_t)N * sizeof(float));
  float *out = calloc((size_t)N, sizeof(float));
  ASSERT_NOT_NULL(a);
  ASSERT_NOT_NULL(b);
  ASSERT_NOT_NULL(out);
  for (int i = 0; i < N; i++) {
    a[i] = (float)i;
    b[i] = (float)i * 0.25f;
  }

  setenv("CPU_COUNT", "2", 1);
  setenv("THREADS", "1", 1);

  PolyCtx *ctx = poly_ctx_new();
  poly_ctx_set_preferred_device(ctx, POLY_DEVICE_X86);
  PolyUOp *abuf = poly_buffer_f32(ctx, N);
  PolyUOp *bbuf = poly_buffer_f32(ctx, N);
  PolyUOp *obuf = poly_buffer_f32(ctx, N);
  PolyUOp *sum = poly_alu2(ctx, POLY_OP_ADD, abuf, bbuf);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, obuf, sum));
  PolySchedule *sched = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_NOT_NULL(sched);
  ASSERT_INT_EQ(sched->template->n_calls, 1);

  PolyUOp *program = poly_schedule_call_to_program(ctx, sched, 0, POLY_DEVICE_X86);
  ASSERT_NOT_NULL(program);
  const PolyProgramInfo *info = poly_program_info(ctx, program);
  ASSERT_NOT_NULL(info);
  ASSERT_INT_EQ(info->global_size[0], 2);
  ASSERT_TRUE(info->n_vars >= 1);

  ASSERT_TRUE(program->n_src >= 5);
  PolyUOp *binary = program->src[4];
  ASSERT_NOT_NULL(binary);
  ASSERT_INT_EQ(binary->op, POLY_OP_BINARY);
  ASSERT_INT_EQ(binary->arg.kind, POLY_ARG_BYTES);
  PolyX86Program *prog = poly_compile_x86(binary->arg.bytes.data, binary->arg.bytes.n);
  ASSERT_NOT_NULL(prog);

  void *args[3] = {out, a, b};
  ASSERT_INT_EQ(poly_x86_program_call_core(prog, args, 3, 0), 0);
  ASSERT_INT_EQ(poly_x86_program_call_core(prog, args, 3, 1), 0);
  for (int i = 0; i < N; i++)
    ASSERT_FLOAT_EQ(out[i], a[i] + b[i], 1e-6f);

  poly_x86_program_destroy(prog);
  poly_schedule_free(sched);
  poly_ctx_destroy(ctx);
  free(a);
  free(b);
  free(out);
  PASS();
}

TEST_BACKEND(x86, schedule_runtime_cross_entropy_dense_axis1_keeps_fifth_arg_live) {
  PolyCtx *ctx = poly_ctx_new();
  poly_ctx_set_preferred_device(ctx, POLY_DEVICE_X86);

  PolyUOp *logits_buf = poly_buffer_f32(ctx, 12);
  PolyUOp *target_buf = poly_buffer_f32(ctx, 12);
  PolyUOp *out_buf = poly_buffer_f32(ctx, 1);
  PolyUOp *logits = poly_reshape(ctx, logits_buf, (int64_t[]){2, 3, 2}, 3);
  PolyUOp *target = poly_reshape(ctx, target_buf, (int64_t[]){2, 3, 2}, 3);
  PolyUOp *loss = poly_cross_entropy(ctx, logits, target, 1);
  ASSERT_NOT_NULL(loss);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out_buf, loss));

  float logits_data[12] = {0};
  float target_data[12] = {
      1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0,
  };
  float out_data[1] = {0};
  PolyTestBufferView bindings[] = {
      POLY_TEST_HOST_VIEW(logits_buf, logits_data),
      POLY_TEST_HOST_VIEW(target_buf, target_data),
      POLY_TEST_HOST_VIEW(out_buf, out_data),
  };

  ASSERT_INT_EQ(poly_test_realize_buffer_views(ctx, sink, bindings, 3), 0);
  ASSERT_FLOAT_EQ(out_data[0], logf(3.0f), 1e-5f);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, schedule_runtime_computed_log2_keeps_loop_live_ins_like_tinygrad) {
  enum { N = 16 };
  float out[N] = {0};
  PolyCtx *ctx = poly_ctx_new();
  poly_ctx_set_preferred_device(ctx, POLY_DEVICE_X86);

  PolyUOp *obuf = poly_buffer_f32(ctx, N);
  PolyUOp *rf = poly_arange(ctx, 0.0, (double)N, 1.0);
  ASSERT_NOT_NULL(rf);
  PolyUOp *x = poly_alu2(
      ctx, POLY_OP_FDIV, poly_alu2(ctx, POLY_OP_ADD, rf, poly_const_float(ctx, 1.0)),
      poly_const_float(ctx, 17.0)
  );
  PolyUOp *y = poly_alu1(ctx, POLY_OP_LOG2, x);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, obuf, y));
  PolyTestBufferView view = POLY_TEST_HOST_VIEW(obuf, out);
  ASSERT_INT_EQ(poly_test_realize_buffer_views(ctx, sink, &view, 1), 0);

  for (int i = 0; i < N; i++) {
    float expected = log2f(((float)i + 1.0f) / 17.0f);
    ASSERT_TRUE(isfinite(out[i]));
    ASSERT_FLOAT_EQ(out[i], expected, 1e-4f);
  }
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, schedule_runtime_pow_const_exponents_match_tinygrad) {
  enum { N = 4 };
  float in[N] = {1.0f, 2.0f, 3.0f, 4.0f};
  float out_i[N] = {0}, out_h[N] = {0}, out_n[N] = {0};
  PolyCtx *ctx = poly_ctx_new();
  poly_ctx_set_preferred_device(ctx, POLY_DEVICE_X86);

  PolyUOp *a = poly_buffer_f32(ctx, N);
  PolyUOp *oi = poly_buffer_f32(ctx, N);
  PolyUOp *oh = poly_buffer_f32(ctx, N);
  PolyUOp *on = poly_buffer_f32(ctx, N);
  poly_buffer_set(ctx, a, in, sizeof(in), POLY_DEVICE_CPU);

  PolyUOp *pow_i = poly_alu2(ctx, POLY_OP_POW, a, poly_const_float(ctx, 2.0));
  PolyUOp *pow_h = poly_alu2(ctx, POLY_OP_POW, a, poly_const_float(ctx, 1.5));
  PolyUOp *pow_n = poly_alu2(ctx, POLY_OP_POW, a, poly_const_float(ctx, -1.0));
  PolyUOp *stores[3] = {
      poly_store_val(ctx, oi, pow_i),
      poly_store_val(ctx, oh, pow_h),
      poly_store_val(ctx, on, pow_n),
  };
  PolyUOp *sink = poly_sink_n(ctx, stores, 3);

  PolyTestBufferView views[] = {
      POLY_TEST_HOST_VIEW(oi, out_i),
      POLY_TEST_HOST_VIEW(oh, out_h),
      POLY_TEST_HOST_VIEW(on, out_n),
  };
  ASSERT_INT_EQ(poly_test_realize_buffer_views(ctx, sink, views, 3), 0);

  for (int i = 0; i < N; i++) {
    ASSERT_FLOAT_EQ(out_i[i], in[i] * in[i], 1e-5f);
    ASSERT_FLOAT_EQ(out_h[i], in[i] * sqrtf(in[i]), 1e-4f);
    ASSERT_FLOAT_EQ(out_n[i], 1.0f / in[i], 1e-5f);
  }
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, schedule_runtime_pow_dynamic_exponent_uses_xpow_like_tinygrad) {
  enum { N = 4 };
  float base[N] = {2.0f, 3.0f, 4.0f, 5.0f};
  float expv[N] = {3.0f, 2.0f, 0.5f, 1.0f};
  float out[N] = {0};
  PolyCtx *ctx = poly_ctx_new();
  poly_ctx_set_preferred_device(ctx, POLY_DEVICE_X86);

  PolyUOp *b = poly_buffer_f32(ctx, N);
  PolyUOp *e = poly_buffer_f32(ctx, N);
  PolyUOp *o = poly_buffer_f32(ctx, N);
  poly_buffer_set(ctx, b, base, sizeof(base), POLY_DEVICE_CPU);
  poly_buffer_set(ctx, e, expv, sizeof(expv), POLY_DEVICE_CPU);

  PolyUOp *pow = poly_alu2(ctx, POLY_OP_POW, b, e);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, o, pow));
  PolyTestBufferView view = POLY_TEST_HOST_VIEW(o, out);
  ASSERT_INT_EQ(poly_test_realize_buffer_views(ctx, sink, &view, 1), 0);

  const float expected[N] = {8.0f, 9.0f, 2.0f, 5.0f};
  for (int i = 0; i < N; i++)
    ASSERT_FLOAT_EQ(out[i], expected[i], 2e-3f);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(x86, schedule_runtime_integer_pow_is_exact_backend_superset) {
  enum { N = 10 };
  int32_t base[N] = {2, 3, -2, -1, 0, 1, 11, 0, -1, 2};
  int32_t expv[N] = {3, 2, 3, -3, -1, -2, 7, 0, INT32_MIN, INT32_MIN};
  int32_t out[N] = {0};
  const int32_t expected[N] = {8, 9, -8, -1, 0, 1, 19487171, 1, 1, 0};
  PolyCtx *ctx = poly_ctx_new();
  poly_ctx_set_preferred_device(ctx, POLY_DEVICE_X86);

  PolyUOp *b = poly_buffer(ctx, POLY_INT32, N);
  PolyUOp *e = poly_buffer(ctx, POLY_INT32, N);
  PolyUOp *o = poly_buffer(ctx, POLY_INT32, N);
  poly_buffer_set(ctx, b, base, sizeof(base), POLY_DEVICE_CPU);
  poly_buffer_set(ctx, e, expv, sizeof(expv), POLY_DEVICE_CPU);

  PolyUOp *pow = poly_alu2(ctx, POLY_OP_POW, b, e);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, o, pow));
  PolyTestBufferView view = POLY_TEST_HOST_VIEW(o, out);
  ASSERT_INT_EQ(poly_test_realize_buffer_views(ctx, sink, &view, 1), 0);

  for (int i = 0; i < N; i++)
    ASSERT_INT_EQ(out[i], expected[i]);
  poly_ctx_destroy(ctx);
  PASS();
}

static void x86_make_data(float *data, int n, uint32_t seed, float scale) {
  uint32_t x = seed;
  for (int i = 0; i < n; i++) {
    x = x * 1664525u + 1013904223u;
    data[i] = (float)((int)((x >> 8) % 1009u) - 504) * scale / 504.0f;
  }
}

TEST_BACKEND(x86, schedule_runtime_qwen_ffn_fused_large_matches_tinygrad_probe_class) {
  enum { D = 256, H = 1536 };
  setenv("CPU_COUNT", "1", 1);
  setenv("THREADS", "0", 1);

  float *x = malloc((size_t)D * sizeof(float));
  float *wg = malloc((size_t)H * D * sizeof(float));
  float *wu = malloc((size_t)H * D * sizeof(float));
  float *wd = malloc((size_t)D * H * sizeof(float));
  float *ref_prod = malloc((size_t)H * sizeof(float));
  float *ref = calloc((size_t)D, sizeof(float));
  float *out = calloc((size_t)D, sizeof(float));
  ASSERT_NOT_NULL(x);
  ASSERT_NOT_NULL(wg);
  ASSERT_NOT_NULL(wu);
  ASSERT_NOT_NULL(wd);
  ASSERT_NOT_NULL(ref_prod);
  ASSERT_NOT_NULL(ref);
  ASSERT_NOT_NULL(out);

  x86_make_data(x, D, 1, 0.02f);
  x86_make_data(wg, H * D, 6, 0.02f);
  x86_make_data(wu, H * D, 7, 0.02f);
  x86_make_data(wd, D * H, 8, 0.02f);

  for (int h = 0; h < H; h++) {
    float gate = 0.0f, up = 0.0f;
    for (int d = 0; d < D; d++) {
      gate += x[d] * wg[h * D + d];
      up += x[d] * wu[h * D + d];
    }
    gate = gate / (1.0f + expf(-gate));
    ref_prod[h] = gate * up;
  }
  for (int d = 0; d < D; d++)
    for (int h = 0; h < H; h++)
      ref[d] += ref_prod[h] * wd[d * H + h];

  PolyCtx *ctx = poly_ctx_new();
  poly_ctx_set_preferred_device(ctx, POLY_DEVICE_X86);
  PolyUOp *xb = poly_buffer_f32(ctx, D);
  PolyUOp *wgb = poly_buffer_f32(ctx, (int64_t)H * D);
  PolyUOp *wub = poly_buffer_f32(ctx, (int64_t)H * D);
  PolyUOp *wdb = poly_buffer_f32(ctx, (int64_t)D * H);
  poly_buffer_set(ctx, xb, x, (size_t)D * sizeof(float), POLY_DEVICE_CPU);
  poly_buffer_set(ctx, wgb, wg, (size_t)H * D * sizeof(float), POLY_DEVICE_CPU);
  poly_buffer_set(ctx, wub, wu, (size_t)H * D * sizeof(float), POLY_DEVICE_CPU);
  poly_buffer_set(ctx, wdb, wd, (size_t)D * H * sizeof(float), POLY_DEVICE_CPU);

  PolyUOp *x2 = poly_reshape(ctx, xb, (int64_t[]){1, D}, 2);
  PolyUOp *wg2 = poly_reshape(ctx, wgb, (int64_t[]){H, D}, 2);
  PolyUOp *wu2 = poly_reshape(ctx, wub, (int64_t[]){H, D}, 2);
  PolyUOp *wd2 = poly_reshape(ctx, wdb, (int64_t[]){D, H}, 2);
  PolyUOp *gate = poly_silu(ctx, poly_dot(ctx, x2, poly_permute(ctx, wg2, (int64_t[]){1, 0}, 2)));
  PolyUOp *up = poly_dot(ctx, x2, poly_permute(ctx, wu2, (int64_t[]){1, 0}, 2));
  PolyUOp *prod = poly_alu2(ctx, POLY_OP_MUL, gate, up);
  PolyUOp *res = poly_dot(ctx, prod, poly_permute(ctx, wd2, (int64_t[]){1, 0}, 2));
  PolyUOp *realized = NULL;
  ASSERT_INT_EQ(poly_realize_uops(ctx, &res, 1, &realized), 0);
  ASSERT_NOT_NULL(realized);
  PolyUOp *buf = (PolyUOp *)poly_uop_get_buffer_identity(realized);
  ASSERT_NOT_NULL(buf);
  ASSERT_INT_EQ(poly_buffer_read(ctx, buf, out, (size_t)D * sizeof(float)), 0);

  for (int i = 0; i < D; i++)
    ASSERT_FLOAT_ABS(out[i], ref[i], 2e-6f);

  poly_ctx_destroy(ctx);
  free(x);
  free(wg);
  free(wu);
  free(wd);
  free(ref_prod);
  free(ref);
  free(out);
  PASS();
}

#endif /* POLY_HAS_X86 */
