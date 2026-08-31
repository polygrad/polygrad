/* Tinygrad 2026-08-22/a9069c177a9d codegen/decomp/dtype.py. */

#include "codegen/decomp/dtype.h"
#include "bigint.h"
#include "uop/ops.h"
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ***** long as two ints *****
 * Tinygrad 2026-08-22/a9069c177a9d codegen/decomp/dtype.py:9-86,138-167. */

typedef struct {
  PolyUOp *lo;
  PolyUOp *hi;
} L2IPair;

static bool l2i_is_long(PolyDType dt) {
  return poly_dtype_eq(dt, POLY_INT64) || poly_dtype_eq(dt, POLY_UINT64);
}

static PolyDType l2i_dt(PolyDType dt) {
  return poly_dtype_eq(dt, POLY_UINT64) ? POLY_UINT32 : POLY_INT32;
}

static PolyUOp *l2i_clone(
    PolyCtx *ctx,
    PolyOps op,
    PolyDType dtype,
    PolyUOp **src,
    int n_src,
    PolyArg arg,
    int32_t tag,
    PolyArg tag_arg
) {
  return (tag != 0 || tag_arg.kind != POLY_ARG_NONE)
             ? poly_uop_tagged_arg(ctx, op, dtype, src, n_src, arg, tag, tag_arg)
             : poly_uop(ctx, op, dtype, src, n_src, arg);
}

static bool l2i_lane(PolyUOp *u, int *lane, PolyDType *dtype) {
  if (!u || u->tag_arg.kind != POLY_ARG_DTYPE ||
      (!poly_dtype_eq(u->tag_arg.dtype, POLY_INT32) && !poly_dtype_eq(u->tag_arg.dtype, POLY_UINT32)
      ) ||
      (u->tag != 0 && u->tag != 1))
    return false;
  if (lane) *lane = u->tag;
  if (dtype) *dtype = u->tag_arg.dtype;
  return true;
}

static PolyUOp *l2i_rtag(PolyCtx *ctx, PolyUOp *u, int lane, PolyDType dtype) {
  if (!u || (lane != 0 && lane != 1)) return NULL;
  return poly_uop_tagged_arg(
      ctx, u->op, u->dtype, u->src, u->n_src, u->arg, lane, poly_arg_dtype(dtype)
  );
}

static PolyUOp *l2i_const(PolyCtx *ctx, PolyDType dt, uint32_t bits) {
  int64_t value = poly_dtype_is_unsigned(dt) ? (int64_t)bits : (int64_t)(int32_t)bits;
  return poly_uop0(ctx, POLY_OP_CONST, dt, poly_arg_int(value));
}

static PolyUOp *l2i_cast(PolyCtx *ctx, PolyUOp *u, PolyDType dt) {
  if (!u) return NULL;
  return poly_dtype_eq(u->dtype, dt) ? u : poly_uop1(ctx, POLY_OP_CAST, dt, u, poly_arg_none());
}

static PolyUOp *l2i_bitcast(PolyCtx *ctx, PolyUOp *u, PolyDType dt) {
  if (!u) return NULL;
  return poly_dtype_eq(u->dtype, dt) ? u : poly_uop1(ctx, POLY_OP_BITCAST, dt, u, poly_arg_none());
}

static PolyUOp *l2i_binary(PolyCtx *ctx, PolyOps op, PolyDType dt, PolyUOp *a, PolyUOp *b) {
  if (!a || !b) return NULL;
  return poly_uop2(ctx, op, dt, a, b, poly_arg_none());
}

static PolyUOp *l2i_cmp(PolyCtx *ctx, PolyOps op, PolyUOp *a, PolyUOp *b) {
  if (!a || !b) return NULL;
  return poly_uop2(ctx, op, POLY_BOOL, a, b, poly_arg_none());
}

static PolyUOp *l2i_not(PolyCtx *ctx, PolyUOp *u) {
  PolyUOp *t = poly_uop0(ctx, POLY_OP_CONST, POLY_BOOL, poly_arg_bool(true));
  return l2i_cmp(ctx, POLY_OP_CMPNE, u, t);
}

static PolyUOp *l2i_eq(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  return l2i_not(ctx, l2i_cmp(ctx, POLY_OP_CMPNE, a, b));
}

static PolyUOp *l2i_where(PolyCtx *ctx, PolyDType dt, PolyUOp *cond, PolyUOp *t, PolyUOp *f) {
  if (!cond || !t || !f) return NULL;
  return poly_uop3(ctx, POLY_OP_WHERE, dt, cond, t, f, poly_arg_none());
}

static L2IPair l2i_add(
    PolyCtx *ctx,
    PolyDType dt,
    PolyUOp *a0,
    PolyUOp *a1,
    PolyUOp *b0,
    PolyUOp *b1
) {
  PolyUOp *low = l2i_binary(ctx, POLY_OP_ADD, dt, a0, b0);
  PolyUOp *carry = l2i_cmp(
      ctx, POLY_OP_CMPLT, l2i_bitcast(ctx, low, POLY_UINT32), l2i_bitcast(ctx, a0, POLY_UINT32)
  );
  PolyUOp *high = l2i_binary(
      ctx, POLY_OP_ADD, dt, l2i_binary(ctx, POLY_OP_ADD, dt, a1, b1), l2i_cast(ctx, carry, dt)
  );
  return (L2IPair){low, high};
}

static L2IPair l2i_sub(
    PolyCtx *ctx,
    PolyDType dt,
    PolyUOp *a0,
    PolyUOp *a1,
    PolyUOp *b0,
    PolyUOp *b1
) {
  PolyUOp *low = l2i_binary(ctx, POLY_OP_SUB, dt, a0, b0);
  PolyUOp *borrow = l2i_cmp(
      ctx, POLY_OP_CMPLT, l2i_bitcast(ctx, a0, POLY_UINT32), l2i_bitcast(ctx, b0, POLY_UINT32)
  );
  PolyUOp *high = l2i_binary(
      ctx, POLY_OP_SUB, dt, l2i_binary(ctx, POLY_OP_SUB, dt, a1, b1), l2i_cast(ctx, borrow, dt)
  );
  return (L2IPair){low, high};
}

static L2IPair l2i_shift(
    PolyCtx *ctx,
    PolyOps op,
    PolyDType dt,
    PolyUOp *a0,
    PolyUOp *a1,
    PolyUOp *b0
) {
  PolyUOp *zero = l2i_const(ctx, dt, 0);
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(1));
  PolyUOp *thirty_one = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(31));
  PolyUOp *thirty_two = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(32));
  PolyUOp *bmod = l2i_binary(ctx, POLY_OP_AND, b0->dtype, b0, thirty_one);
  PolyUOp *n = l2i_cast(ctx, bmod, POLY_UINT32);
  PolyUOp *inv = l2i_binary(ctx, POLY_OP_SUB, n->dtype, thirty_one, n);
  PolyUOp *wide = l2i_not(ctx, l2i_cmp(ctx, POLY_OP_CMPLT, b0, thirty_two));
  PolyUOp *a0u = l2i_bitcast(ctx, a0, POLY_UINT32);
  PolyUOp *a1u = l2i_bitcast(ctx, a1, POLY_UINT32);

  if (op == POLY_OP_SHL) {
    PolyUOp *low = l2i_bitcast(ctx, l2i_binary(ctx, POLY_OP_SHL, POLY_UINT32, a0u, n), dt);
    PolyUOp *carry = l2i_binary(
        ctx, POLY_OP_SHR, POLY_UINT32, l2i_binary(ctx, POLY_OP_SHR, POLY_UINT32, a0u, one), inv
    );
    PolyUOp *high = l2i_bitcast(
        ctx,
        l2i_binary(
            ctx, POLY_OP_OR, POLY_UINT32, l2i_binary(ctx, POLY_OP_SHL, POLY_UINT32, a1u, n), carry
        ),
        dt
    );
    return (L2IPair){l2i_where(ctx, dt, wide, zero, low), l2i_where(ctx, dt, wide, low, high)};
  }

  PolyUOp *carry = l2i_binary(
      ctx, POLY_OP_SHL, POLY_UINT32, l2i_binary(ctx, POLY_OP_SHL, POLY_UINT32, a1u, one), inv
  );
  PolyUOp *low = l2i_bitcast(
      ctx,
      l2i_binary(
          ctx, POLY_OP_OR, POLY_UINT32, l2i_binary(ctx, POLY_OP_SHR, POLY_UINT32, a0u, n), carry
      ),
      dt
  );
  PolyUOp *high = l2i_binary(ctx, POLY_OP_SHR, dt, a1, bmod);
  PolyUOp *sign =
      poly_dtype_is_unsigned(dt) ? zero : l2i_binary(ctx, POLY_OP_SHR, dt, a1, thirty_one);
  return (L2IPair){l2i_where(ctx, dt, wide, high, low), l2i_where(ctx, dt, wide, sign, high)};
}

static void l2i_unpack16(PolyCtx *ctx, PolyUOp *u, PolyUOp **lo, PolyUOp **hi) {
  PolyUOp *v = l2i_bitcast(ctx, u, POLY_UINT32);
  *lo = l2i_binary(ctx, POLY_OP_AND, POLY_UINT32, v, l2i_const(ctx, POLY_UINT32, 0xffff));
  *hi = l2i_binary(ctx, POLY_OP_SHR, POLY_UINT32, v, l2i_const(ctx, POLY_UINT32, 16));
}

static L2IPair l2i_mul(
    PolyCtx *ctx,
    PolyDType dt,
    PolyUOp *a0,
    PolyUOp *a1,
    PolyUOp *b0,
    PolyUOp *b1
) {
  PolyUOp *a00 = NULL, *a01 = NULL, *b00 = NULL, *b01 = NULL;
  l2i_unpack16(ctx, a0, &a00, &a01);
  l2i_unpack16(ctx, b0, &b00, &b01);
  PolyUOp *sixteen = l2i_const(ctx, POLY_UINT32, 16);
  PolyUOp *p01 = l2i_binary(ctx, POLY_OP_MUL, POLY_UINT32, a00, b01);
  PolyUOp *p10 = l2i_binary(ctx, POLY_OP_MUL, POLY_UINT32, a01, b00);
  L2IPair mid = l2i_add(
      ctx, dt, l2i_bitcast(ctx, l2i_binary(ctx, POLY_OP_SHL, POLY_UINT32, p01, sixteen), dt),
      l2i_bitcast(ctx, l2i_binary(ctx, POLY_OP_SHR, POLY_UINT32, p01, sixteen), dt),
      l2i_bitcast(ctx, l2i_binary(ctx, POLY_OP_SHL, POLY_UINT32, p10, sixteen), dt),
      l2i_bitcast(ctx, l2i_binary(ctx, POLY_OP_SHR, POLY_UINT32, p10, sixteen), dt)
  );
  PolyUOp *low_product = l2i_bitcast(ctx, l2i_binary(ctx, POLY_OP_MUL, POLY_UINT32, a00, b00), dt);
  PolyUOp *high_product = l2i_binary(
      ctx, POLY_OP_ADD, dt,
      l2i_binary(
          ctx, POLY_OP_ADD, dt,
          l2i_bitcast(ctx, l2i_binary(ctx, POLY_OP_MUL, POLY_UINT32, a01, b01), dt),
          l2i_binary(ctx, POLY_OP_MUL, dt, a0, b1)
      ),
      l2i_binary(ctx, POLY_OP_MUL, dt, a1, b0)
  );
  return l2i_add(ctx, dt, mid.lo, mid.hi, low_product, high_product);
}

static PolyUOp *l2i_compare(
    PolyCtx *ctx,
    PolyOps op,
    PolyUOp *a0,
    PolyUOp *a1,
    PolyUOp *b0,
    PolyUOp *b1
) {
  if (op == POLY_OP_CMPEQ) {
    return l2i_binary(ctx, POLY_OP_AND, POLY_BOOL, l2i_eq(ctx, a0, b0), l2i_eq(ctx, a1, b1));
  }
  if (op == POLY_OP_CMPNE) {
    return l2i_binary(
        ctx, POLY_OP_OR, POLY_BOOL, l2i_cmp(ctx, POLY_OP_CMPNE, a0, b0),
        l2i_cmp(ctx, POLY_OP_CMPNE, a1, b1)
    );
  }
  if (op != POLY_OP_CMPLT) return NULL;
  PolyUOp *high_lt = l2i_cmp(ctx, POLY_OP_CMPLT, a1, b1);
  PolyUOp *high_eq = l2i_eq(ctx, a1, b1);
  PolyUOp *low_lt = l2i_cmp(
      ctx, POLY_OP_CMPLT, l2i_bitcast(ctx, a0, POLY_UINT32), l2i_bitcast(ctx, b0, POLY_UINT32)
  );
  return l2i_binary(
      ctx, POLY_OP_OR, POLY_BOOL, high_lt, l2i_binary(ctx, POLY_OP_AND, POLY_BOOL, high_eq, low_lt)
  );
}

static L2IPair l2i_divmod(
    PolyCtx *ctx,
    PolyOps op,
    PolyDType dt,
    PolyUOp *a0,
    PolyUOp *a1,
    PolyUOp *b0,
    PolyUOp *b1
) {
  bool is_signed = !poly_dtype_is_unsigned(dt);
  PolyUOp *a_negative = NULL, *b_negative = NULL;
  if (is_signed) {
    PolyUOp *zero = l2i_const(ctx, dt, 0);
    a_negative = l2i_cmp(ctx, POLY_OP_CMPLT, a1, zero);
    b_negative = l2i_cmp(ctx, POLY_OP_CMPLT, b1, zero);
    a0 = l2i_bitcast(ctx, a0, POLY_UINT32);
    a1 = l2i_bitcast(ctx, a1, POLY_UINT32);
    b0 = l2i_bitcast(ctx, b0, POLY_UINT32);
    b1 = l2i_bitcast(ctx, b1, POLY_UINT32);
    L2IPair an = l2i_sub(
        ctx, POLY_UINT32, l2i_const(ctx, POLY_UINT32, 0), l2i_const(ctx, POLY_UINT32, 0), a0, a1
    );
    L2IPair bn = l2i_sub(
        ctx, POLY_UINT32, l2i_const(ctx, POLY_UINT32, 0), l2i_const(ctx, POLY_UINT32, 0), b0, b1
    );
    a0 = l2i_where(ctx, POLY_UINT32, a_negative, an.lo, a0);
    a1 = l2i_where(ctx, POLY_UINT32, a_negative, an.hi, a1);
    b0 = l2i_where(ctx, POLY_UINT32, b_negative, bn.lo, b0);
    b1 = l2i_where(ctx, POLY_UINT32, b_negative, bn.hi, b1);
  }

  PolyUOp *zero = l2i_const(ctx, POLY_UINT32, 0);
  PolyUOp *one = l2i_const(ctx, POLY_UINT32, 1);
  L2IPair q = {zero, zero}, r = {zero, zero};
  for (int i = 63; i >= 0; i--) {
    r = l2i_shift(ctx, POLY_OP_SHL, POLY_UINT32, r.lo, r.hi, one);
    L2IPair shifted =
        l2i_shift(ctx, POLY_OP_SHR, POLY_UINT32, a0, a1, l2i_const(ctx, POLY_UINT32, (uint32_t)i));
    PolyUOp *incoming = l2i_binary(ctx, POLY_OP_AND, POLY_UINT32, shifted.lo, one);
    r.lo = l2i_binary(ctx, POLY_OP_OR, POLY_UINT32, r.lo, incoming);
    PolyUOp *take = l2i_not(ctx, l2i_compare(ctx, POLY_OP_CMPLT, r.lo, r.hi, b0, b1));
    L2IPair diff = l2i_sub(ctx, POLY_UINT32, r.lo, r.hi, b0, b1);
    PolyUOp *qbit = l2i_binary(
        ctx, POLY_OP_SHL, POLY_UINT32, l2i_cast(ctx, take, POLY_UINT32),
        l2i_const(ctx, POLY_UINT32, (uint32_t)(i % 32))
    );
    if (i < 32)
      q.lo = l2i_binary(ctx, POLY_OP_OR, POLY_UINT32, q.lo, qbit);
    else
      q.hi = l2i_binary(ctx, POLY_OP_OR, POLY_UINT32, q.hi, qbit);
    r.lo = l2i_where(ctx, POLY_UINT32, take, diff.lo, r.lo);
    r.hi = l2i_where(ctx, POLY_UINT32, take, diff.hi, r.hi);
  }

  if (!is_signed) {
    L2IPair ret = op == POLY_OP_CMOD ? r : q;
    return (L2IPair){l2i_bitcast(ctx, ret.lo, dt), l2i_bitcast(ctx, ret.hi, dt)};
  }

  L2IPair nq = l2i_sub(ctx, POLY_UINT32, zero, zero, q.lo, q.hi);
  L2IPair nr = l2i_sub(ctx, POLY_UINT32, zero, zero, r.lo, r.hi);
  q = (L2IPair){l2i_bitcast(ctx, q.lo, dt), l2i_bitcast(ctx, q.hi, dt)};
  r = (L2IPair){l2i_bitcast(ctx, r.lo, dt), l2i_bitcast(ctx, r.hi, dt)};
  nq = (L2IPair){l2i_bitcast(ctx, nq.lo, dt), l2i_bitcast(ctx, nq.hi, dt)};
  nr = (L2IPair){l2i_bitcast(ctx, nr.lo, dt), l2i_bitcast(ctx, nr.hi, dt)};
  if (op == POLY_OP_CMOD)
    return (L2IPair
    ){l2i_where(ctx, dt, a_negative, nr.lo, r.lo), l2i_where(ctx, dt, a_negative, nr.hi, r.hi)};
  PolyUOp *quotient_negative = l2i_binary(ctx, POLY_OP_XOR, POLY_BOOL, a_negative, b_negative);
  return (L2IPair
  ){l2i_where(ctx, dt, quotient_negative, nq.lo, q.lo),
    l2i_where(ctx, dt, quotient_negative, nq.hi, q.hi)};
}

static L2IPair l2i_alu(PolyCtx *ctx, PolyOps op, PolyDType dt, PolyUOp **u, int n) {
  PolyUOp *zero = l2i_const(ctx, dt, 0);
  if (op == POLY_OP_NEG && n == 2) return l2i_sub(ctx, dt, zero, zero, u[0], u[1]);
  if (op == POLY_OP_SHL || op == POLY_OP_SHR) {
    if (n < 3) return (L2IPair){NULL, NULL};
    return l2i_shift(ctx, op, dt, u[0], u[1], u[2]);
  }
  if (op == POLY_OP_ADD && n == 4) return l2i_add(ctx, dt, u[0], u[1], u[2], u[3]);
  if (op == POLY_OP_SUB && n == 4) return l2i_sub(ctx, dt, u[0], u[1], u[2], u[3]);
  if (op == POLY_OP_MUL && n == 4) return l2i_mul(ctx, dt, u[0], u[1], u[2], u[3]);
  if ((op == POLY_OP_CDIV || op == POLY_OP_CMOD) && n == 4)
    return l2i_divmod(ctx, op, dt, u[0], u[1], u[2], u[3]);
  if ((op == POLY_OP_XOR || op == POLY_OP_OR || op == POLY_OP_AND) && n == 4) {
    return (L2IPair){l2i_binary(ctx, op, dt, u[0], u[2]), l2i_binary(ctx, op, dt, u[1], u[3])};
  }
  if (op == POLY_OP_WHERE && n == 5) {
    return (L2IPair){l2i_where(ctx, dt, u[0], u[1], u[3]), l2i_where(ctx, dt, u[0], u[2], u[4])};
  }
  if (op == POLY_OP_MAX && n == 4) {
    PolyUOp *cond = l2i_compare(ctx, POLY_OP_CMPLT, u[0], u[1], u[2], u[3]);
    return (L2IPair){l2i_where(ctx, dt, cond, u[2], u[0]), l2i_where(ctx, dt, cond, u[3], u[1])};
  }
  if (op == POLY_OP_BITCAST && n == 2) {
    return (L2IPair){l2i_bitcast(ctx, u[0], dt), l2i_bitcast(ctx, u[1], dt)};
  }
  return (L2IPair){NULL, NULL};
}

static L2IPair l2i_cast_to_long(PolyCtx *ctx, PolyDType target, PolyUOp *src) {
  PolyDType dt = l2i_dt(target);
  PolyUOp *lo = l2i_cast(ctx, src, dt);
  PolyUOp *zero_src = poly_dtype_is_float(src->dtype)
                          ? poly_uop0(ctx, POLY_OP_CONST, src->dtype, poly_arg_float(0.0))
                          : poly_uop0(ctx, POLY_OP_CONST, src->dtype, poly_arg_int(0));
  PolyUOp *negative = l2i_cmp(ctx, POLY_OP_CMPLT, src, zero_src);
  if (!poly_dtype_is_float(src->dtype)) {
    if (poly_dtype_is_bool(src->dtype) || poly_dtype_is_unsigned(src->dtype))
      return (L2IPair){lo, l2i_const(ctx, dt, 0)};
    return (L2IPair
    ){lo, l2i_where(ctx, dt, negative, l2i_const(ctx, dt, UINT32_MAX), l2i_const(ctx, dt, 0))};
  }

  PolyUOp *scale = poly_uop0(ctx, POLY_OP_CONST, src->dtype, poly_arg_float(4294967296.0));
  PolyUOp *high_float = l2i_binary(ctx, POLY_OP_FDIV, src->dtype, src, scale);
  PolyUOp *lo_nonzero = l2i_cmp(ctx, POLY_OP_CMPNE, lo, l2i_const(ctx, dt, 0));
  PolyUOp *adjust =
      l2i_cast(ctx, l2i_binary(ctx, POLY_OP_AND, POLY_BOOL, negative, lo_nonzero), dt);
  return (L2IPair){lo, l2i_binary(ctx, POLY_OP_SUB, dt, l2i_cast(ctx, high_float, dt), adjust)};
}

static PolyUOp *l2i_cast_from_long(PolyCtx *ctx, PolyDType target, PolyUOp *a0, PolyUOp *a1) {
  if (!a0 || !a1) return NULL;
  if (!poly_dtype_is_float(target)) return l2i_cast(ctx, l2i_bitcast(ctx, a0, POLY_UINT32), target);

  /* Tinygrad dtype.py:35-37 keeps Python literals weak through this graph. */
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(0));
  PolyUOp *minus_one = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(-1));
  PolyUOp *a0_nonnegative = l2i_not(ctx, l2i_cmp(ctx, POLY_OP_CMPLT, a0, zero));
  PolyUOp *a0_negative = l2i_cmp(ctx, POLY_OP_CMPLT, a0, zero);
  PolyUOp *small = l2i_binary(
      ctx, POLY_OP_OR, POLY_BOOL,
      l2i_binary(ctx, POLY_OP_AND, POLY_BOOL, l2i_eq(ctx, a1, zero), a0_nonnegative),
      l2i_binary(ctx, POLY_OP_AND, POLY_BOOL, l2i_eq(ctx, a1, minus_one), a0_negative)
  );
  PolyUOp *direct = l2i_cast(ctx, a0, target);
  PolyUOp *scale = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKFLOAT, poly_arg_float(4294967296.0));
  PolyUOp *wide = l2i_binary(
      ctx, POLY_OP_ADD, POLY_FLOAT32,
      l2i_binary(ctx, POLY_OP_MUL, POLY_FLOAT32, l2i_cast(ctx, a1, POLY_FLOAT32), scale),
      l2i_cast(ctx, l2i_bitcast(ctx, a0, POLY_UINT32), POLY_FLOAT32)
  );
  return l2i_where(ctx, target, small, direct, l2i_cast(ctx, wide, target));
}

static PolyUOp *split_l2i_inputs(PolyCtx *ctx, PolyUOp **uops, int n_uops) {
  PolyUOp *sink = poly_uop(ctx, POLY_OP_SINK, POLY_VOID, uops, n_uops, poly_arg_none());
  return sink ? poly_graph_rewrite_ctx_ex(
                    ctx, sink, poly_pm_long_decomp(), poly_graph_rewrite_userctx(), true
                )
              : NULL;
}

static PolyUOp *long_define(PolyCtx *ctx, PolyUOp *x, const PolyBindings *b) {
  PolyUOp *sz = poly_bind(b, "sz");
  if (!sz || !l2i_is_long(x->dtype) || x->arg.kind != POLY_ARG_PARAM || !x->arg.param) return NULL;
  PolyUOp *shape =
      l2i_binary(ctx, POLY_OP_MUL, sz->dtype, sz, poly_const_like(ctx, sz, poly_arg_int(2)));
  if (!shape) return NULL;
  PolyParamArg arg = *x->arg.param;
  arg.dtype = l2i_dt(x->dtype);
  PolyUOp *src[] = {shape};
  return l2i_clone(ctx, x->op, arg.dtype, src, 1, poly_arg_param(&arg), x->tag, x->tag_arg);
}

static PolyUOp *long_index(PolyCtx *ctx, PolyUOp *x, const PolyBindings *b) {
  (void)b;
  int lane = 0;
  PolyDType dtype;
  if (!l2i_is_long(x->dtype) || !l2i_lane(x, &lane, &dtype)) return NULL;
  PolyUOp *index = poly_reindex(ctx, x, lane, 2);
  return index ? l2i_clone(
                     ctx, index->op, dtype, index->src, index->n_src, index->arg, 0, poly_arg_none()
                 )
               : NULL;
}

static PolyUOp *long_store(PolyCtx *ctx, PolyUOp *st, const PolyBindings *b) {
  PolyUOp *idx = poly_bind(b, "idx"), *val = poly_bind(b, "val");
  if (!idx || !val || !l2i_is_long(idx->dtype) || val->tag != 0 ||
      val->tag_arg.kind != POLY_ARG_NONE)
    return NULL;
  PolyDType dtype = l2i_dt(idx->dtype);
  PolyUOp *src0[] = {l2i_rtag(ctx, idx, 0, dtype), l2i_rtag(ctx, val, 0, dtype)};
  PolyUOp *src1[] = {l2i_rtag(ctx, idx, 1, dtype), l2i_rtag(ctx, val, 1, dtype)};
  if (!src0[0] || !src0[1] || !src1[0] || !src1[1]) return NULL;
  PolyUOp *stores[] = {
      l2i_clone(ctx, st->op, st->dtype, src0, 2, st->arg, st->tag, st->tag_arg),
      l2i_clone(ctx, st->op, st->dtype, src1, 2, st->arg, st->tag, st->tag_arg),
  };
  return stores[0] && stores[1]
             ? poly_uop(ctx, POLY_OP_GROUP, POLY_VOID, stores, 2, poly_arg_none())
             : NULL;
}

static PolyUOp *long_comparison(PolyCtx *ctx, PolyUOp *x, const PolyBindings *b) {
  PolyUOp *a = poly_bind(b, "a");
  if (!a || !l2i_is_long(a->dtype) || x->n_src != 2) return NULL;
  PolyDType dtype = l2i_dt(a->dtype);
  PolyUOp *words[] = {
      l2i_rtag(ctx, x->src[0], 0, dtype),
      l2i_rtag(ctx, x->src[0], 1, dtype),
      l2i_rtag(ctx, x->src[1], 0, dtype),
      l2i_rtag(ctx, x->src[1], 1, dtype),
  };
  PolyUOp *split = split_l2i_inputs(ctx, words, 4);
  return split && split->n_src == 4
             ? l2i_compare(ctx, x->op, split->src[0], split->src[1], split->src[2], split->src[3])
             : NULL;
}

static PolyUOp *long_cast_long(PolyCtx *ctx, PolyUOp *x, const PolyBindings *b) {
  PolyUOp *a = poly_bind(b, "a");
  int lane = 0;
  if (!a || !l2i_is_long(x->dtype) || !l2i_is_long(a->dtype) || !l2i_lane(x, &lane, NULL))
    return NULL;
  PolyDType dtype = l2i_dt(x->dtype), source_dtype = l2i_dt(a->dtype);
  PolyUOp *words[] = {
      l2i_rtag(ctx, a, 0, source_dtype),
      l2i_rtag(ctx, a, 1, source_dtype),
  };
  PolyUOp *split = split_l2i_inputs(ctx, words, 2);
  L2IPair pair = split && split->n_src == 2 ? l2i_alu(ctx, POLY_OP_BITCAST, dtype, split->src, 2)
                                            : (L2IPair){NULL, NULL};
  return lane ? pair.hi : pair.lo;
}

static PolyUOp *long_cast_to(PolyCtx *ctx, PolyUOp *x, const PolyBindings *b) {
  PolyUOp *a = poly_bind(b, "a");
  int lane = 0;
  if (!a || !l2i_is_long(x->dtype) || l2i_is_long(a->dtype) || !l2i_lane(x, &lane, NULL))
    return NULL;
  PolyUOp *inputs[] = {a};
  PolyUOp *split = split_l2i_inputs(ctx, inputs, 1);
  L2IPair pair = split && split->n_src == 1 ? l2i_cast_to_long(ctx, x->dtype, split->src[0])
                                            : (L2IPair){NULL, NULL};
  return lane ? pair.hi : pair.lo;
}

static PolyUOp *long_cast_from(PolyCtx *ctx, PolyUOp *x, const PolyBindings *b) {
  PolyUOp *a = poly_bind(b, "a");
  if (!a || l2i_is_long(x->dtype) || !l2i_is_long(a->dtype) || a->tag != 0 ||
      a->tag_arg.kind != POLY_ARG_NONE)
    return NULL;
  PolyDType dtype = l2i_dt(a->dtype);
  PolyUOp *words[] = {l2i_rtag(ctx, a, 0, dtype), l2i_rtag(ctx, a, 1, dtype)};
  PolyUOp *split = split_l2i_inputs(ctx, words, 2);
  return split && split->n_src == 2
             ? l2i_cast_from_long(ctx, x->dtype, split->src[0], split->src[1])
             : NULL;
}

static PolyUOp *long_shift(PolyCtx *ctx, PolyUOp *x, const PolyBindings *b) {
  PolyUOp *a = poly_bind(b, "a"), *amount = poly_bind(b, "b");
  int lane = 0;
  if (!a || !amount || !l2i_is_long(x->dtype) || !l2i_lane(x, &lane, NULL)) return NULL;
  PolyDType dtype = l2i_dt(x->dtype);
  PolyUOp *words[] = {
      l2i_rtag(ctx, a, 0, dtype),
      l2i_rtag(ctx, a, 1, dtype),
      l2i_rtag(ctx, amount, 0, dtype),
  };
  PolyUOp *split = split_l2i_inputs(ctx, words, 3);
  L2IPair pair = split && split->n_src == 3 ? l2i_alu(ctx, x->op, dtype, split->src, 3)
                                            : (L2IPair){NULL, NULL};
  return lane ? pair.hi : pair.lo;
}

static PolyUOp *long_where(PolyCtx *ctx, PolyUOp *x, const PolyBindings *b) {
  PolyUOp *condition = poly_bind(b, "c"), *a = poly_bind(b, "a"), *other = poly_bind(b, "b");
  int lane = 0;
  if (!condition || !a || !other || !l2i_is_long(x->dtype) || !l2i_lane(x, &lane, NULL))
    return NULL;
  PolyDType dtype = l2i_dt(x->dtype);
  PolyUOp *words[] = {
      condition,
      l2i_rtag(ctx, a, 0, dtype),
      l2i_rtag(ctx, a, 1, dtype),
      l2i_rtag(ctx, other, 0, dtype),
      l2i_rtag(ctx, other, 1, dtype),
  };
  PolyUOp *split = split_l2i_inputs(ctx, words, 5);
  L2IPair pair = split && split->n_src == 5 ? l2i_alu(ctx, x->op, dtype, split->src, 5)
                                            : (L2IPair){NULL, NULL};
  return lane ? pair.hi : pair.lo;
}

static PolyUOp *long_alu(PolyCtx *ctx, PolyUOp *x, const PolyBindings *b) {
  (void)b;
  int lane = 0;
  if (!l2i_is_long(x->dtype) || !l2i_lane(x, &lane, NULL)) return NULL;
  PolyDType dtype = l2i_dt(x->dtype);
  int count = x->n_src * 2;
  PolyUOp **words = malloc((size_t)count * sizeof(*words));
  if (!words) return NULL;
  for (int i = 0; i < x->n_src; i++) {
    words[2 * i] = l2i_rtag(ctx, x->src[i], 0, dtype);
    words[2 * i + 1] = l2i_rtag(ctx, x->src[i], 1, dtype);
  }
  PolyUOp *split = split_l2i_inputs(ctx, words, count);
  free(words);
  L2IPair pair = split && split->n_src == count ? l2i_alu(ctx, x->op, dtype, split->src, count)
                                                : (L2IPair){NULL, NULL};
  return lane ? pair.hi : pair.lo;
}

static PolyUOp *long_load(PolyCtx *ctx, PolyUOp *x, const PolyBindings *b) {
  PolyUOp *idx = poly_bind(b, "idx");
  int lane = 0;
  PolyDType dtype;
  if (!idx || !l2i_is_long(x->dtype) || !l2i_lane(x, &lane, &dtype)) return NULL;
  PolyUOp *rewritten = poly_graph_rewrite_ctx_ex(
      ctx, idx, poly_pm_long_decomp(), poly_graph_rewrite_userctx(), true
  );
  PolyUOp *reindexed = rewritten ? poly_reindex(ctx, rewritten, lane, 2) : NULL;
  if (!reindexed) return NULL;
  reindexed = l2i_clone(
      ctx, reindexed->op, reindexed->dtype, reindexed->src, reindexed->n_src, reindexed->arg, 0,
      poly_arg_none()
  );
  return reindexed ? poly_uop1(ctx, POLY_OP_LOAD, dtype, reindexed, x->arg) : NULL;
}

static PolyUOp *long_const(PolyCtx *ctx, PolyUOp *x, const PolyBindings *b) {
  (void)b;
  int lane = 0;
  PolyDType dtype;
  if (!l2i_lane(x, &lane, &dtype) ||
      (x->arg.kind != POLY_ARG_INT && x->arg.kind != POLY_ARG_BIGINT))
    return NULL;
  uint64_t bits = poly_arg_integer_to_u64_mod(x->arg);
  return l2i_const(ctx, dtype, lane ? (uint32_t)(bits >> 32) : (uint32_t)bits);
}

static _Thread_local PolyPatternMatcher *g_pm_long_decomp = NULL;

PolyPatternMatcher *poly_pm_long_decomp(void) {
  if (g_pm_long_decomp) return g_pm_long_decomp;
  PolyDType longs[] = {POLY_INT64, POLY_UINT64};
  PolyOpSet shifts = poly_opset_add(poly_opset_add((PolyOpSet){{0, 0}}, POLY_OP_SHL), POLY_OP_SHR);
  PolyOpSet alu = POLY_GROUP_ALU;
  for (PolyOps op = 0; op < POLY_OP_COUNT; op++)
    if (poly_opset_has(POLY_GROUP_COMPARISON, op) || op == POLY_OP_SHL || op == POLY_OP_SHR ||
        op == POLY_OP_WHERE)
      alu.bits[op / 64] &= ~((uint64_t)1 << (op % 64));
  alu = poly_opset_add(alu, POLY_OP_BITCAST);

  PolyNamedRule rules[] = {
      POLY_RULE(
          poly_upat_set_dtype(
              poly_upat_ops1(POLY_GROUP_DEFINES, poly_upat_any("sz"), "x"), longs, 2
          ),
          long_define
      ),
      POLY_RULE(
          poly_upat_set_dtype(poly_upat_op(POLY_OP_INDEX, NULL, 0, "x"), longs, 2), long_index
      ),
      POLY_RULE(
          poly_upat_op2(
              POLY_OP_STORE, poly_upat_dtype("idx", longs, 2), poly_upat_any("val"), "st"
          ),
          long_store
      ),
      POLY_RULE(
          poly_upat_ops2(
              POLY_GROUP_COMPARISON, poly_upat_dtype("a", longs, 2), poly_upat_any(NULL), "x"
          ),
          long_comparison
      ),
      POLY_RULE(
          poly_upat_set_dtype(
              poly_upat_op1(POLY_OP_CAST, poly_upat_dtype("a", longs, 2), "x"), longs, 2
          ),
          long_cast_long
      ),
      POLY_RULE(
          poly_upat_set_dtype(poly_upat_op1(POLY_OP_CAST, poly_upat_any("a"), "x"), longs, 2),
          long_cast_to
      ),
      POLY_RULE(poly_upat_op1(POLY_OP_CAST, poly_upat_dtype("a", longs, 2), "x"), long_cast_from),
      POLY_RULE(
          poly_upat_set_dtype(
              poly_upat_ops2(shifts, poly_upat_any("a"), poly_upat_any("b"), "x"), longs, 2
          ),
          long_shift
      ),
      POLY_RULE(
          poly_upat_set_dtype(
              poly_upat_op3(
                  POLY_OP_WHERE, poly_upat_any("c"), poly_upat_any("a"), poly_upat_any("b"), "x"
              ),
              longs, 2
          ),
          long_where
      ),
      POLY_RULE(poly_upat_set_dtype(poly_upat_ops(alu, NULL, 0, "x"), longs, 2), long_alu),
      POLY_RULE(
          poly_upat_set_dtype(poly_upat_op1(POLY_OP_LOAD, poly_upat_any("idx"), "x"), longs, 2),
          long_load
      ),
      POLY_RULE(poly_upat_op(POLY_OP_CONST, NULL, 0, "x"), long_const),
  };
  g_pm_long_decomp =
      poly_pm_thread_cache(poly_pm_new_named(rules, (int)(sizeof(rules) / sizeof(rules[0]))));
  return g_pm_long_decomp;
}

static PolyFloatDecompContext *float_ctx(void) {
  return poly_graph_rewrite_userctx();
}

static bool dtype_is(PolyDType a, PolyDType b) {
  return poly_dtype_eq(a, b);
}

static PolyDType f2f_dt(PolyDType dt) {
  if (dt.bitsize == 8) return POLY_UINT8;
  if (dt.bitsize == 16) return POLY_UINT16;
  if (dt.bitsize == 32) return POLY_UINT32;
  return POLY_UINT64;
}

static bool finfo(PolyDType dt, int *exponent, int *mantissa) {
  if (dtype_is(dt, POLY_FP8E4M3) || dtype_is(dt, POLY_FP8E4M3FNUZ)) {
    *exponent = 4;
    *mantissa = 3;
    return true;
  }
  if (dtype_is(dt, POLY_FP8E5M2) || dtype_is(dt, POLY_FP8E5M2FNUZ)) {
    *exponent = 5;
    *mantissa = 2;
    return true;
  }
  if (dtype_is(dt, POLY_FLOAT16)) {
    *exponent = 5;
    *mantissa = 10;
    return true;
  }
  if (dtype_is(dt, POLY_BFLOAT16)) {
    *exponent = 8;
    *mantissa = 7;
    return true;
  }
  if (dtype_is(dt, POLY_FLOAT32)) {
    *exponent = 8;
    *mantissa = 23;
    return true;
  }
  if (dtype_is(dt, POLY_FLOAT64)) {
    *exponent = 11;
    *mantissa = 52;
    return true;
  }
  return false;
}

static int exponent_bias(PolyDType dt) {
  int exponent = 0, mantissa = 0;
  if (!finfo(dt, &exponent, &mantissa)) return 0;
  return (1 << (exponent - 1)) - 1 + (poly_dtype_is_fp8_fnuz(dt) ? 1 : 0);
}

static PolyUOp *weakint(PolyCtx *ctx, int64_t value) {
  return poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(value));
}

static PolyUOp *weakfloat(PolyCtx *ctx, double value) {
  return poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKFLOAT, poly_arg_float(value));
}

static PolyUOp *cast(PolyCtx *ctx, PolyUOp *x, PolyDType dtype) {
  return x && dtype_is(x->dtype, dtype) ? x
         : x ? poly_uop1(ctx, POLY_OP_CAST, dtype, x, poly_arg_dtype(dtype))
             : NULL;
}

static PolyUOp *bitcast(PolyCtx *ctx, PolyUOp *x, PolyDType dtype) {
  return x && dtype_is(x->dtype, dtype) ? x
         : x ? poly_uop1(ctx, POLY_OP_BITCAST, dtype, x, poly_arg_dtype(dtype))
             : NULL;
}

static PolyUOp *binary(PolyCtx *ctx, PolyOps op, PolyDType dtype, PolyUOp *a, PolyUOp *b) {
  return a && b ? poly_uop2(ctx, op, dtype, a, b, poly_arg_none()) : NULL;
}

static PolyUOp *cmp(PolyCtx *ctx, PolyOps op, PolyUOp *a, PolyUOp *b) {
  return binary(ctx, op, POLY_BOOL, a, b);
}

static PolyUOp *ne(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  return cmp(ctx, POLY_OP_CMPNE, a, b);
}

/* UOp.eq is expressed as logical_not(CMPNE), not a raw CMPEQ. */
static PolyUOp *eq(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  PolyUOp *different = ne(ctx, a, b);
  return different
             ? ne(ctx, different, poly_uop0(ctx, POLY_OP_CONST, POLY_BOOL, poly_arg_bool(true)))
             : NULL;
}

static PolyUOp *where(
    PolyCtx *ctx,
    PolyDType dtype,
    PolyUOp *condition,
    PolyUOp *yes,
    PolyUOp *no
) {
  return condition && yes && no
             ? poly_uop3(ctx, POLY_OP_WHERE, dtype, condition, yes, no, poly_arg_none())
             : NULL;
}

/* Tinygrad codegen/decomp/transcendental.py:19-20. */
static PolyUOp *shr(PolyCtx *ctx, PolyUOp *x, int amount) {
  return binary(ctx, POLY_OP_FLOORDIV, x->dtype, x, weakint(ctx, INT64_C(1) << amount));
}

static PolyUOp *shl(PolyCtx *ctx, PolyUOp *x, int amount) {
  return binary(ctx, POLY_OP_MUL, x->dtype, x, weakint(ctx, INT64_C(1) << amount));
}

static PolyUOp *clone_n(
    PolyCtx *ctx,
    PolyUOp *u,
    PolyOps op,
    PolyDType dtype,
    PolyUOp **src,
    int n_src,
    PolyArg arg,
    int32_t tag,
    PolyArg tag_arg
) {
  PolyUOp **use_src = src ? src : u->src;
  return tag != 0 || tag_arg.kind != POLY_ARG_NONE
             ? poly_uop_tagged_arg(ctx, op, dtype, use_src, n_src, arg, tag, tag_arg)
             : poly_uop(ctx, op, dtype, use_src, n_src, arg);
}

static PolyUOp *clone(
    PolyCtx *ctx,
    PolyUOp *u,
    PolyDType dtype,
    PolyUOp **src,
    PolyArg arg,
    int32_t tag,
    PolyArg tag_arg
) {
  return clone_n(ctx, u, u->op, dtype, src, u->n_src, arg, tag, tag_arg);
}

static PolyUOp *replace(PolyCtx *ctx, PolyUOp *u, PolyDType dtype, PolyUOp **src) {
  PolyArg arg =
      (u->op == POLY_OP_CAST || u->op == POLY_OP_BITCAST) ? poly_arg_dtype(dtype) : u->arg;
  return clone(ctx, u, dtype, src, arg, u->tag, u->tag_arg);
}

PolyUOp *poly_reindex(PolyCtx *ctx, PolyUOp *idx, int off, int mul) {
  if (!idx || (idx->op != POLY_OP_INDEX && idx->op != POLY_OP_SHRINK) || idx->n_src < 2 ||
      (idx->op == POLY_OP_SHRINK && mul != 1))
    return NULL;
  PolyUOp *offset = idx->src[1];
  PolyUOp *next = binary(
      ctx, POLY_OP_ADD, offset->dtype,
      binary(ctx, POLY_OP_MUL, offset->dtype, offset, weakint(ctx, mul)), weakint(ctx, off)
  );
  if (!next) return NULL;
  if (idx->op == POLY_OP_SHRINK) {
    PolyUOp *src[] = {idx->src[0], next};
    return clone_n(ctx, idx, POLY_OP_INDEX, idx->dtype, src, 2, idx->arg, idx->tag, idx->tag_arg);
  }
  PolyUOp **src = malloc((size_t)idx->n_src * sizeof(*src));
  if (!src) return NULL;
  memcpy(src, idx->src, (size_t)idx->n_src * sizeof(*src));
  src[1] = next;
  PolyUOp *ret = clone(ctx, idx, idx->dtype, src, idx->arg, idx->tag, idx->tag_arg);
  free(src);
  return ret;
}

static PolyUOp *rne(PolyCtx *ctx, PolyUOp *v, int shift) {
  PolyUOp *q = shr(ctx, v, shift);
  PolyUOp *round = binary(ctx, POLY_OP_AND, v->dtype, shr(ctx, v, shift - 1), weakint(ctx, 1));
  PolyUOp *tail =
      ne(ctx, binary(ctx, POLY_OP_AND, v->dtype, v, weakint(ctx, (INT64_C(1) << (shift - 1)) - 1)),
         weakint(ctx, 0));
  PolyUOp *sticky = binary(
      ctx, POLY_OP_OR, v->dtype, cast(ctx, tail, v->dtype),
      binary(ctx, POLY_OP_AND, v->dtype, q, weakint(ctx, 1))
  );
  return binary(ctx, POLY_OP_ADD, v->dtype, q, binary(ctx, POLY_OP_AND, v->dtype, round, sticky));
}

static PolyUOp *f2f_clamp(PolyCtx *ctx, PolyUOp *val, PolyDType dtype) {
  int exponent = 0, mantissa = 0;
  if (!finfo(dtype, &exponent, &mantissa)) return NULL;
  int max_exp, max_man;
  if (poly_dtype_is_fp8_fnuz(dtype)) {
    max_exp = (1 << exponent) - 1;
    max_man = (1 << mantissa) - 1;
  } else if (dtype_is(dtype, POLY_FP8E4M3)) {
    max_exp = (1 << exponent) - 1;
    max_man = (1 << mantissa) - 2;
  } else {
    max_exp = (1 << exponent) - 2;
    max_man = (1 << mantissa) - 1;
  }
  double mxv = ldexp(
      1.0 + (double)max_man / (double)(INT64_C(1) << mantissa), max_exp - exponent_bias(dtype)
  );
  PolyUOp *mx = poly_uop0(ctx, POLY_OP_CONST, val->dtype, poly_arg_float(mxv));
  PolyUOp *minus_one = weakfloat(ctx, -1.0);
  PolyUOp *neg_mx = binary(ctx, POLY_OP_MUL, val->dtype, mx, minus_one);
  PolyUOp *sat = poly_dtype_is_fp8(dtype)
                     ? mx
                     : poly_uop0(ctx, POLY_OP_CONST, val->dtype, poly_arg_float(INFINITY));
  PolyUOp *neg_sat = binary(ctx, POLY_OP_MUL, val->dtype, sat, minus_one);
  return where(
      ctx, val->dtype, ne(ctx, val, val), val,
      where(
          ctx, val->dtype, cmp(ctx, POLY_OP_CMPLT, val, neg_mx), neg_sat,
          where(ctx, val->dtype, cmp(ctx, POLY_OP_CMPLT, mx, val), sat, val)
      )
  );
}

/* Tinygrad 2026-08-22/a9069c177a9d codegen/decomp/dtype.py:f2f. */
static PolyUOp *f2f(PolyCtx *ctx, PolyUOp *v, PolyDType from, PolyDType to) {
  int fe = 0, fm = 0, te = 0, tm = 0;
  if (!finfo(from, &fe, &fm) || !finfo(to, &te, &tm)) return NULL;
  int fs = from.bitsize, ts = to.bitsize;
  int fb = exponent_bias(from), tb = exponent_bias(to);
  PolyDType from_uint = f2f_dt(from), to_uint = f2f_dt(to);

  if (fe <= te && fm < tm) {
    PolyUOp *sign =
        shl(ctx,
            cast(
                ctx, binary(ctx, POLY_OP_AND, from_uint, v, weakint(ctx, INT64_C(1) << (fs - 1))),
                to_uint
            ),
            ts - fs);
    PolyUOp *nosign = cast(
        ctx, binary(ctx, POLY_OP_AND, from_uint, v, weakint(ctx, (INT64_C(1) << (fs - 1)) - 1)),
        to_uint
    );
    PolyUOp *exp = shr(ctx, nosign, fm);
    PolyUOp *shifted = shl(ctx, nosign, tm - fm);
    PolyUOp *norm = binary(ctx, POLY_OP_ADD, to_uint, shifted, shl(ctx, weakint(ctx, tb - fb), tm));
    PolyUOp *nan = binary(
        ctx, POLY_OP_OR, to_uint, shifted, shl(ctx, weakint(ctx, (INT64_C(1) << te) - 1), tm)
    );
    if (poly_dtype_is_fp8_fnuz(from)) {
      PolyUOp *fnuz_nan = binary(
          ctx, POLY_OP_AND, POLY_BOOL, ne(ctx, sign, weakint(ctx, 0)),
          eq(ctx, nosign, weakint(ctx, 0))
      );
      PolyUOp *qnan = binary(
          ctx, POLY_OP_OR, to_uint, shl(ctx, weakint(ctx, (INT64_C(1) << te) - 1), tm),
          shl(ctx, weakint(ctx, 1), tm - 1)
      );
      int bias_floor = (fb > tb ? fb - tb : 0) + 1;
      PolyUOp *magnitude = where(
          ctx, to_uint, cmp(ctx, POLY_OP_CMPLT, exp, weakint(ctx, bias_floor)), weakint(ctx, 0),
          norm
      );
      return bitcast(
          ctx,
          where(ctx, to_uint, fnuz_nan, qnan, binary(ctx, POLY_OP_OR, to_uint, sign, magnitude)), to
      );
    }
    PolyUOp *is_nan = dtype_is(from, POLY_FP8E4M3)
                          ? eq(ctx, nosign, weakint(ctx, (INT64_C(1) << (fm + fe)) - 1))
                          : eq(ctx, exp, weakint(ctx, (INT64_C(1) << fe) - 1));
    PolyUOp *magnitude = where(
        ctx, to_uint, eq(ctx, exp, weakint(ctx, 0)), weakint(ctx, 0),
        where(ctx, to_uint, is_nan, nan, norm)
    );
    return bitcast(ctx, binary(ctx, POLY_OP_OR, to_uint, sign, magnitude), to);
  }

  if (fe >= te && fm > tm) {
    PolyUOp *bits = bitcast(ctx, f2f_clamp(ctx, bitcast(ctx, v, from), to), from_uint);
    PolyUOp *sign = binary(
        ctx, POLY_OP_AND, from_uint, shr(ctx, bits, fs - ts), weakint(ctx, INT64_C(1) << (ts - 1))
    );
    PolyUOp *nosign =
        binary(ctx, POLY_OP_AND, from_uint, bits, weakint(ctx, (INT64_C(1) << (fs - 1)) - 1));
    PolyUOp *norm = cast(
        ctx,
        binary(
            ctx, POLY_OP_SUB, from_uint, rne(ctx, nosign, fm - tm),
            weakint(ctx, (int64_t)(fb - tb) << tm)
        ),
        to_uint
    );
    PolyUOp *exp = binary(
        ctx, POLY_OP_AND, from_uint, shr(ctx, bits, fm), weakint(ctx, (INT64_C(1) << fe) - 1)
    );
    PolyUOp *underflow = cmp(ctx, POLY_OP_CMPLT, exp, weakint(ctx, 1 + fb - tb));
    PolyUOp *nan_mantissa = dtype_is(to, POLY_FP8E4M3)
                                ? weakint(ctx, (INT64_C(1) << tm) - 1)
                                : binary(
                                      ctx, POLY_OP_AND, from_uint, shr(ctx, nosign, fm - tm),
                                      weakint(ctx, (INT64_C(1) << tm) - 1)
                                  );
    PolyUOp *nan = cast(
        ctx,
        binary(
            ctx, POLY_OP_OR, from_uint, binary(ctx, POLY_OP_OR, from_uint, sign, nan_mantissa),
            shl(ctx, weakint(ctx, (INT64_C(1) << te) - 1), tm)
        ),
        to_uint
    );
    PolyUOp *finite = binary(
        ctx, POLY_OP_OR, to_uint, cast(ctx, sign, to_uint),
        where(ctx, to_uint, underflow, weakint(ctx, 0), norm)
    );
    PolyUOp *is_nan = eq(ctx, exp, weakint(ctx, (INT64_C(1) << fe) - 1));
    return poly_dtype_is_fp8_fnuz(to)
               ? where(ctx, to_uint, is_nan, shl(ctx, weakint(ctx, 1), ts - 1), finite)
               : where(ctx, to_uint, is_nan, nan, finite);
  }
  return NULL;
}

static PolyUOp *rewrite_index(PolyCtx *ctx, PolyUOp *idx, PolyFloatDecompContext *fctx) {
  return poly_graph_rewrite_ctx_ex(ctx, idx, poly_pm_float_decomp(), fctx, true);
}

static PolyUOp *f2f_load(PolyCtx *ctx, PolyUOp *load, PolyDType from, PolyDType to) {
  PolyFloatDecompContext fctx = {.from = from, .to = to};
  PolyUOp *idx = rewrite_index(ctx, load->src[0], &fctx);
  int64_t lanes = poly_uop_max_numel(ctx, load);
  if (!idx || lanes < 1 || lanes > INT32_MAX) return NULL;
  if (lanes == 1) {
    PolyUOp **src = malloc((size_t)load->n_src * sizeof(*src));
    if (!src) return NULL;
    memcpy(src, load->src, (size_t)load->n_src * sizeof(*src));
    src[0] = idx;
    PolyUOp *raw = replace(ctx, load, f2f_dt(from), src);
    free(src);
    return f2f(ctx, raw, from, to);
  }
  PolyUOp **values = malloc((size_t)lanes * sizeof(*values));
  PolyUOp **src = malloc((size_t)load->n_src * sizeof(*src));
  if (!values || !src) {
    free(values);
    free(src);
    return NULL;
  }
  memcpy(src, load->src, (size_t)load->n_src * sizeof(*src));
  for (int64_t i = 0; i < lanes; i++) {
    src[0] = poly_reindex(ctx, idx, (int)i, 1);
    PolyUOp *raw = src[0] ? replace(ctx, load, f2f_dt(from), src) : NULL;
    values[i] = raw ? f2f(ctx, raw, from, to) : NULL;
    if (!values[i]) {
      free(values);
      free(src);
      return NULL;
    }
  }
  PolyUOp *ret = poly_uop_stack(ctx, values, (int)lanes);
  free(values);
  free(src);
  return ret;
}

static PolyUOp *f2f_store(
    PolyCtx *ctx,
    PolyUOp *store,
    PolyUOp *idx,
    PolyUOp *val,
    PolyDType from,
    PolyDType to
) {
  int64_t lanes = poly_uop_max_numel(ctx, val);
  if (lanes < 1 || lanes > INT32_MAX) return NULL;
  if (lanes == 1) {
    PolyUOp **src = malloc((size_t)store->n_src * sizeof(*src));
    if (!src) return NULL;
    memcpy(src, store->src, (size_t)store->n_src * sizeof(*src));
    src[0] = idx;
    src[1] = f2f(ctx, bitcast(ctx, val, f2f_dt(to)), to, from);
    PolyUOp *ret = src[1] ? replace(ctx, store, store->dtype, src) : NULL;
    free(src);
    return ret;
  }
  PolyUOp **stores = malloc((size_t)lanes * sizeof(*stores));
  PolyUOp **src = malloc((size_t)store->n_src * sizeof(*src));
  if (!stores || !src) {
    free(stores);
    free(src);
    return NULL;
  }
  memcpy(src, store->src, (size_t)store->n_src * sizeof(*src));
  for (int64_t i = 0; i < lanes; i++) {
    PolyUOp *lane = weakint(ctx, i);
    PolyUOp *lane_val = poly_uop_index(ctx, val, &lane, 1);
    src[0] = poly_reindex(ctx, idx, (int)i, 1);
    src[1] = lane_val ? f2f(ctx, bitcast(ctx, lane_val, f2f_dt(to)), to, from) : NULL;
    stores[i] = src[0] && src[1] ? replace(ctx, store, store->dtype, src) : NULL;
    if (!stores[i]) {
      free(stores);
      free(src);
      return NULL;
    }
  }
  PolyUOp *ret = poly_uop(ctx, POLY_OP_GROUP, POLY_VOID, stores, (int)lanes, poly_arg_none());
  free(stores);
  free(src);
  return ret;
}

static PolyUOp *float_define(PolyCtx *ctx, PolyUOp *x, const PolyBindings *b) {
  (void)b;
  PolyFloatDecompContext *fctx = float_ctx();
  if (!fctx || !dtype_is(x->dtype, fctx->from)) return NULL;
  PolyArg arg = x->arg;
  PolyParamArg param;
  if (arg.kind == POLY_ARG_PARAM && arg.param) {
    param = *arg.param;
    param.dtype = f2f_dt(fctx->from);
    arg = poly_arg_param(&param);
  }
  return clone(ctx, x, f2f_dt(fctx->from), NULL, arg, x->tag, poly_arg_dtype(fctx->from));
}

static PolyUOp *float_index(PolyCtx *ctx, PolyUOp *x, const PolyBindings *b) {
  (void)b;
  PolyFloatDecompContext *fctx = float_ctx();
  if (!fctx || !dtype_is(x->dtype, fctx->from) || x->n_src < 1 ||
      (x->op == POLY_OP_INDEX && (x->src[0]->op == POLY_OP_LOAD || x->src[0]->op == POLY_OP_STACK)))
    return NULL;
  PolyUOp **src = malloc((size_t)x->n_src * sizeof(*src));
  if (!src) return NULL;
  memcpy(src, x->src, (size_t)x->n_src * sizeof(*src));
  src[0] = rewrite_index(ctx, x->src[0], fctx);
  PolyUOp *ret =
      src[0] ? clone(ctx, x, f2f_dt(fctx->from), src, x->arg, x->tag, poly_arg_dtype(fctx->from))
             : NULL;
  free(src);
  return ret;
}

static PolyUOp *float_load(PolyCtx *ctx, PolyUOp *x, const PolyBindings *b) {
  (void)b;
  PolyFloatDecompContext *fctx = float_ctx();
  return fctx && dtype_is(x->dtype, fctx->from) && x->n_src >= 1
             ? f2f_load(ctx, x, fctx->from, fctx->to)
             : NULL;
}

static PolyUOp *float_bitcast_load(PolyCtx *ctx, PolyUOp *bc, const PolyBindings *b) {
  PolyFloatDecompContext *fctx = float_ctx();
  PolyUOp *load = poly_bind(b, "ld");
  if (!fctx || !load || !dtype_is(load->dtype, fctx->from) || load->n_src < 1) return NULL;
  PolyUOp *idx = rewrite_index(ctx, load->src[0], fctx);
  PolyUOp **src = malloc((size_t)load->n_src * sizeof(*src));
  if (!idx || !src) {
    free(src);
    return NULL;
  }
  memcpy(src, load->src, (size_t)load->n_src * sizeof(*src));
  src[0] = idx;
  PolyUOp *raw = replace(ctx, load, f2f_dt(fctx->from), src);
  free(src);
  return raw ? bitcast(ctx, raw, bc->dtype) : NULL;
}

static PolyUOp *float_bitcast_from(PolyCtx *ctx, PolyUOp *bc, const PolyBindings *b) {
  PolyFloatDecompContext *fctx = float_ctx();
  PolyUOp *x = poly_bind(b, "x");
  if (!fctx || !x || !dtype_is(x->dtype, fctx->to) || bc->dtype.bitsize != fctx->from.bitsize)
    return NULL;
  PolyUOp *converted = f2f(ctx, bitcast(ctx, x, f2f_dt(fctx->to)), fctx->to, fctx->from);
  if (!converted) return NULL;
  PolyUOp *src[] = {converted};
  return replace(ctx, bc, bc->dtype, src);
}

static PolyUOp *float_bitcast_to(PolyCtx *ctx, PolyUOp *bc, const PolyBindings *b) {
  PolyFloatDecompContext *fctx = float_ctx();
  PolyUOp *x = poly_bind(b, "x");
  return fctx && x && dtype_is(bc->dtype, fctx->from)
             ? f2f(ctx, bitcast(ctx, x, f2f_dt(fctx->from)), fctx->from, fctx->to)
             : NULL;
}

static PolyUOp *float_cast(PolyCtx *ctx, PolyUOp *x, const PolyBindings *b) {
  PolyFloatDecompContext *fctx = float_ctx();
  PolyUOp *val = poly_bind(b, "val");
  return fctx && val && dtype_is(x->dtype, fctx->from)
             ? f2f_clamp(ctx, cast(ctx, val, fctx->to), fctx->from)
             : NULL;
}

static PolyUOp *float_const(PolyCtx *ctx, PolyUOp *x, const PolyBindings *b) {
  (void)b;
  PolyFloatDecompContext *fctx = float_ctx();
  if (!fctx || !dtype_is(x->dtype, fctx->from)) return NULL;
  return poly_uop0(ctx, POLY_OP_CONST, fctx->to, x->arg);
}

static PolyUOp *float_op(PolyCtx *ctx, PolyUOp *x, const PolyBindings *b) {
  (void)b;
  PolyFloatDecompContext *fctx = float_ctx();
  if (!fctx || !dtype_is(x->dtype, fctx->from)) return NULL;
  PolyUOp **src = x->n_src ? malloc((size_t)x->n_src * sizeof(*src)) : NULL;
  if (x->n_src && !src) return NULL;
  for (int i = 0; i < x->n_src; i++)
    src[i] = dtype_is(x->src[i]->dtype, fctx->from) ? cast(ctx, x->src[i], fctx->to) : x->src[i];
  PolyUOp *ret = replace(ctx, x, fctx->to, src);
  free(src);
  return ret;
}

static bool tagged_float(PolyUOp *idx, PolyDType dtype) {
  return idx && idx->tag_arg.kind == POLY_ARG_DTYPE && dtype_is(idx->tag_arg.dtype, dtype);
}

static PolyUOp *float_store_bitcast(PolyCtx *ctx, PolyUOp *st, const PolyBindings *b) {
  PolyFloatDecompContext *fctx = float_ctx();
  PolyUOp *idx = poly_bind(b, "idx"), *val = poly_bind(b, "val");
  if (!fctx || !idx || !val || val->n_src != 1 || !dtype_is(val->dtype, fctx->from) ||
      !tagged_float(idx, fctx->from))
    return NULL;
  PolyUOp **src = malloc((size_t)st->n_src * sizeof(*src));
  if (!src) return NULL;
  memcpy(src, st->src, (size_t)st->n_src * sizeof(*src));
  src[0] = idx;
  src[1] = bitcast(ctx, val->src[0], f2f_dt(fctx->from));
  PolyUOp *ret = src[1] ? replace(ctx, st, st->dtype, src) : NULL;
  free(src);
  return ret;
}

static PolyUOp *float_store(PolyCtx *ctx, PolyUOp *st, const PolyBindings *b) {
  PolyFloatDecompContext *fctx = float_ctx();
  PolyUOp *idx = poly_bind(b, "idx"), *val = poly_bind(b, "val");
  if (!fctx || !idx || !val || !dtype_is(val->dtype, fctx->to)) return NULL;
  if (idx->op == POLY_OP_CAST && idx->n_src == 1) idx = idx->src[0];
  return tagged_float(idx, fctx->from) ? f2f_store(ctx, st, idx, val, fctx->from, fctx->to) : NULL;
}

static _Thread_local PolyPatternMatcher *g_pm_float_decomp = NULL;

PolyPatternMatcher *poly_pm_float_decomp(void) {
  if (g_pm_float_decomp) return g_pm_float_decomp;
  PolyDType floats[] = {
      POLY_FP8E4M3, POLY_FP8E5M2,  POLY_FP8E4M3FNUZ, POLY_FP8E5M2FNUZ,
      POLY_FLOAT16, POLY_BFLOAT16, POLY_FLOAT32,     POLY_FLOAT64,
  };
  PolyOpSet float_ops = {{0, 0}};
  for (int op = 1; op < POLY_OP_COUNT; op++) {
    if (poly_opset_has(POLY_GROUP_DEFINES, (PolyOps)op) || op == POLY_OP_CAST ||
        op == POLY_OP_BITCAST || op == POLY_OP_CONST)
      continue;
    float_ops = poly_opset_add(float_ops, (PolyOps)op);
  }
  PolyUPat *load = poly_upat_allow_any_len(poly_upat_op(POLY_OP_LOAD, NULL, 0, "ld"));
  PolyUPat *bitcast_load_src[] = {load};
  PolyUPat *float_src = poly_upat_dtype("x", floats, 8);
  PolyUPat *any_src = poly_upat_any("x");
  PolyUPat *cast_src[] = {poly_upat_any("val")};
  PolyUPat *store_bitcast_src[] = {
      poly_upat_any("idx"),
      poly_upat_op1(POLY_OP_BITCAST, poly_upat_any(NULL), "val"),
  };
  PolyUPat *store_src[] = {poly_upat_any("idx"), poly_upat_dtype("val", floats, 8)};
  PolyOpSet index_ops = poly_opset_add((PolyOpSet){{0, 0}}, POLY_OP_INDEX);
  index_ops = poly_opset_add(index_ops, POLY_OP_SHRINK);
  PolyRule rules[] = {
      {poly_upat_allow_any_len(poly_upat_ops(POLY_GROUP_DEFINES, NULL, 0, "x")), float_define},
      {poly_upat_allow_any_len(poly_upat_ops(index_ops, NULL, 0, "x")), float_index},
      {poly_upat_allow_any_len(poly_upat_op(POLY_OP_LOAD, NULL, 0, "x")), float_load},
      {poly_upat_op(POLY_OP_BITCAST, bitcast_load_src, 1, "bc"), float_bitcast_load},
      {poly_upat_op1(POLY_OP_BITCAST, float_src, "bc"), float_bitcast_from},
      {poly_upat_op1(POLY_OP_BITCAST, any_src, "bc"), float_bitcast_to},
      {poly_upat_op(POLY_OP_CAST, cast_src, 1, "x"), float_cast},
      {poly_upat_set_dtype(poly_upat_op(POLY_OP_CONST, NULL, 0, "x"), floats, 8), float_const},
      {poly_upat_allow_any_len(poly_upat_ops(float_ops, NULL, 0, "x")), float_op},
      {poly_upat_op(POLY_OP_STORE, store_bitcast_src, 2, "st"), float_store_bitcast},
      {poly_upat_op(POLY_OP_STORE, store_src, 2, "st"), float_store},
  };
  g_pm_float_decomp =
      poly_pm_thread_cache(poly_pm_new(rules, (int)(sizeof(rules) / sizeof(rules[0]))));
  return g_pm_float_decomp;
}

static PolyUOp *detect_dtype_decomp(PolyCtx *ctx, PolyUOp *x, const PolyBindings *b) {
  (void)ctx;
  (void)b;
  PolyDTypeDecompsContext *dctx = poly_graph_rewrite_userctx();
  if (!dctx || !x) return NULL;
  if (dtype_is(x->dtype, POLY_INT64) || dtype_is(x->dtype, POLY_UINT64))
    dctx->seen_long = true;
  else if (dtype_is(x->dtype, POLY_FP8E4M3))
    dctx->seen_fp8 |= 1u << 0;
  else if (dtype_is(x->dtype, POLY_FP8E4M3FNUZ))
    dctx->seen_fp8 |= 1u << 1;
  else if (dtype_is(x->dtype, POLY_FP8E5M2))
    dctx->seen_fp8 |= 1u << 2;
  else if (dtype_is(x->dtype, POLY_FP8E5M2FNUZ))
    dctx->seen_fp8 |= 1u << 3;
  else if (dtype_is(x->dtype, POLY_FLOAT16))
    dctx->seen_float16 = true;
  else if (dtype_is(x->dtype, POLY_BFLOAT16))
    dctx->seen_bfloat16 = true;
  return NULL;
}

static PolyUOp *do_dtype_decomps(PolyCtx *ctx, PolyUOp *sink, const PolyBindings *b) {
  (void)b;
  PolyDTypeDecompsContext *dctx = poly_graph_rewrite_userctx();
  if (!dctx || !sink || sink->op != POLY_OP_SINK) return NULL;
  PolyUOp *ret = sink;
  unsigned char long_ctx = 0;
  if (dctx->seen_long && !poly_renderer_supports_dtype(dctx->caps, POLY_INT64))
    ret = poly_graph_rewrite_ctx_ex(ctx, ret, poly_pm_long_decomp(), &long_ctx, true);
  const PolyDType fp8s[] = {
      POLY_FP8E4M3,
      POLY_FP8E4M3FNUZ,
      POLY_FP8E5M2,
      POLY_FP8E5M2FNUZ,
  };
  for (int i = 0; ret && i < 4; i++) {
    if (!(dctx->seen_fp8 & (1u << i)) || poly_renderer_supports_dtype(dctx->caps, fp8s[i]))
      continue;
    PolyDType to =
        poly_renderer_supports_dtype(dctx->caps, POLY_FLOAT16) ? POLY_FLOAT16 : POLY_FLOAT32;
    PolyFloatDecompContext float_ctx = {.from = fp8s[i], .to = to};
    ret = poly_graph_rewrite_ctx_ex(ctx, ret, poly_pm_float_decomp(), &float_ctx, true);
  }
  if (ret && dctx->seen_float16 && !poly_renderer_supports_dtype(dctx->caps, POLY_FLOAT16)) {
    PolyFloatDecompContext float_ctx = {.from = POLY_FLOAT16, .to = POLY_FLOAT32};
    ret = poly_graph_rewrite_ctx_ex(ctx, ret, poly_pm_float_decomp(), &float_ctx, true);
  }
  if (ret && dctx->seen_bfloat16 && !poly_renderer_supports_dtype(dctx->caps, POLY_BFLOAT16)) {
    PolyFloatDecompContext float_ctx = {.from = POLY_BFLOAT16, .to = POLY_FLOAT32};
    ret = poly_graph_rewrite_ctx_ex(ctx, ret, poly_pm_float_decomp(), &float_ctx, true);
  }
  dctx->seen_long = dctx->seen_float16 = dctx->seen_bfloat16 = false;
  dctx->seen_fp8 = 0;
  return ret != sink ? ret : NULL;
}

static _Thread_local PolyPatternMatcher *g_pm_dtype_decomps = NULL;

PolyPatternMatcher *poly_pm_dtype_decomps(void) {
  if (g_pm_dtype_decomps) return g_pm_dtype_decomps;
  PolyDType emulated[] = {
      POLY_FP8E4M3, POLY_FP8E5M2,  POLY_FP8E4M3FNUZ, POLY_FP8E5M2FNUZ,
      POLY_FLOAT16, POLY_BFLOAT16, POLY_INT64,       POLY_UINT64,
  };
  PolyNamedRule rules[] = {
      POLY_RULE(poly_upat_dtype("x", emulated, 8), detect_dtype_decomp),
      POLY_RULE(
          poly_upat_allow_any_len(poly_upat_op(POLY_OP_SINK, NULL, 0, "sink")), do_dtype_decomps
      ),
  };
  g_pm_dtype_decomps =
      poly_pm_thread_cache(poly_pm_new_named(rules, (int)(sizeof(rules) / sizeof(rules[0]))));
  return g_pm_dtype_decomps;
}
