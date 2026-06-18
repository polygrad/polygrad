/*
 * alu.c — ALU constant-fold executor
 *
 * Mirrors tinygrad's python_alu / exec_alu: evaluates ALU ops on constants.
 * Used by symbolic simplification for constant folding.
 */

#include "pat.h"
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

/* Safe math helpers */

static double safe_exp2(double x) {
  if (x > 1023.0) return INFINITY;
  if (x < -1074.0) return 0.0;
  return exp2(x);
}

static double safe_log2(double x) {
  if (x > 0.0) return log2(x);
  if (x == 0.0) return -INFINITY;
  return NAN;
}

static double safe_sqrt(double x) {
  return x >= 0.0 ? sqrt(x) : NAN;
}

static double safe_recip(double x) {
  return x != 0.0 ? 1.0 / x : copysign(INFINITY, x);
}

static double safe_sin(double x) {
  return isinf(x) ? NAN : sin(x);
}

static double safe_pow(double x, double y) {
  double r = pow(x, y);
  if (isnan(r) && !isnan(x) && !isnan(y)) return INFINITY;
  return r;
}

/* C-style integer division (truncates toward zero) */
static int64_t cdiv(int64_t a, int64_t b) {
  if (b == 0) return 0;
  if (a == INT64_MIN && b == -1) return INT64_MIN;
  return a / b;
}

/* C-style modulo */
static int64_t cmod(int64_t a, int64_t b) {
  if (b == 0) return 0;
  if (a == INT64_MIN && b == -1) return 0;
  return a % b;
}

static int64_t i64_from_u64(uint64_t v) {
  int64_t out;
  memcpy(&out, &v, sizeof(out));
  return out;
}

/* Get numeric value from PolyArg */

static double arg_to_float(PolyArg a) {
  switch (a.kind) {
  case POLY_ARG_FLOAT:
    return a.f;
  case POLY_ARG_INT:
    return (double)a.i;
  case POLY_ARG_BOOL:
    return a.b ? 1.0 : 0.0;
  default:
    return 0.0;
  }
}

static int64_t arg_to_int(PolyArg a) {
  switch (a.kind) {
  case POLY_ARG_INT:
    return a.i;
  case POLY_ARG_FLOAT:
    return (int64_t)a.f;
  case POLY_ARG_BOOL:
    return a.b ? 1 : 0;
  default:
    return 0;
  }
}

static bool arg_to_bool(PolyArg a) {
  switch (a.kind) {
  case POLY_ARG_BOOL:
    return a.b;
  case POLY_ARG_INT:
    return a.i != 0;
  case POLY_ARG_FLOAT:
    return a.f != 0.0;
  default:
    return false;
  }
}

static bool arg_is_invalid(PolyArg a) {
  return a.kind == POLY_ARG_INVALID;
}

static bool is_cmp_op(PolyOps op) {
  return op == POLY_OP_CMPLT || op == POLY_OP_CMPNE || op == POLY_OP_CMPEQ;
}


static float f16_bits_to_f32(uint16_t h) {
  uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
  uint32_t exp = ((uint32_t)h >> 10) & 0x1fu;
  uint32_t frac = (uint32_t)h & 0x03ffu;
  uint32_t u;
  if (exp == 0) {
    if (frac == 0) {
      u = sign;
    } else {
      int e = -14;
      while ((frac & 0x0400u) == 0) {
        frac <<= 1;
        e--;
      }
      frac &= 0x03ffu;
      u = sign | (uint32_t)(e + 127) << 23 | (frac << 13);
    }
  } else if (exp == 31) {
    u = sign | 0x7f800000u | (frac << 13);
  } else {
    u = sign | ((exp + 127 - 15) << 23) | (frac << 13);
  }
  float out;
  memcpy(&out, &u, sizeof(out));
  return out;
}

static uint16_t f32_to_f16_bits_rne(float f) {
  uint32_t u;
  memcpy(&u, &f, sizeof(u));
  uint32_t sign = (u >> 16) & 0x8000u;
  uint32_t exp = (u >> 23) & 0xffu;
  uint32_t mant = u & 0x007fffffu;

  if (exp == 0xffu) return (uint16_t)(sign | 0x7c00u | (mant ? 0x0200u : 0));

  int32_t half_exp = (int32_t)exp - 127 + 15;
  if (half_exp >= 31) return (uint16_t)(sign | 0x7c00u);
  if (half_exp <= 0) {
    if (half_exp < -10) return (uint16_t)sign;
    mant |= 0x00800000u;
    uint32_t shift = (uint32_t)(14 - half_exp);
    uint32_t half_mant = mant >> shift;
    uint32_t round_bit = UINT32_C(1) << (shift - 1);
    uint32_t remainder = mant & (round_bit - 1);
    if ((mant & round_bit) && (remainder || (half_mant & 1u))) half_mant++;
    return (uint16_t)(sign | half_mant);
  }

  uint32_t half_mant = mant >> 13;
  uint32_t round_bit = 0x00001000u;
  uint32_t remainder = mant & (round_bit - 1);
  if ((mant & round_bit) && (remainder || (half_mant & 1u))) {
    half_mant++;
    if (half_mant == 0x0400u) {
      half_mant = 0;
      half_exp++;
      if (half_exp >= 31) return (uint16_t)(sign | 0x7c00u);
    }
  }
  return (uint16_t)(sign | ((uint32_t)half_exp << 10) | half_mant);
}

static double round_to_f16(double x) {
  float f = (float)x;
  return (double)f16_bits_to_f32(f32_to_f16_bits_rne(f));
}

static double round_to_bf16(double x) {
  float f = (float)x;
  if (!isfinite(f)) return (double)f;
  uint32_t u;
  memcpy(&u, &f, sizeof(u));
  u = (u + 0x7fffu + ((u >> 16) & 1u)) & 0xffff0000u;
  memcpy(&f, &u, sizeof(f));
  return (double)f;
}

/* Truncate result to dtype range */

static PolyArg truncate_result(PolyArg val, PolyDType dtype) {
  if (poly_dtype_is_bool(dtype)) return poly_arg_bool(arg_to_bool(val));

  if (poly_dtype_is_int(dtype)) {
    int64_t v = arg_to_int(val);
    int bits = dtype.bitsize;
    if (poly_dtype_is_unsigned(dtype)) {
      if (bits < 64) v &= ((int64_t)1 << bits) - 1;
    } else {
      if (bits < 64) {
        int64_t mask = ((int64_t)1 << bits) - 1;
        v &= mask;
        if (v & ((int64_t)1 << (bits - 1))) v |= ~mask; /* sign extend */
      }
    }
    return poly_arg_int(v);
  }

  if (poly_dtype_is_float(dtype)) {
    PolyDType sdt = poly_dtype_scalar(dtype);
    double v = arg_to_float(val);
    if (sdt.priority == POLY_FLOAT16.priority && sdt.bitsize == 16)
      v = round_to_f16(v);
    else if (sdt.priority == POLY_BFLOAT16.priority && sdt.bitsize == 16)
      v = round_to_bf16(v);
    else if (sdt.bitsize == 32)
      v = (double)(float)v;
    return poly_arg_float(v);
  }

  return val;
}

/* exec_alu: evaluate an ALU op on constant operands */

PolyArg poly_exec_alu(PolyOps op, PolyDType dtype, PolyArg *ops, int n_ops) {
  if (op == POLY_OP_CAST && n_ops == 1) {
    if (poly_dtype_is_bool(dtype)) return poly_arg_bool(arg_to_bool(ops[0]));
    if (poly_dtype_is_int(dtype)) return truncate_result(poly_arg_int(arg_to_int(ops[0])), dtype);
    if (poly_dtype_is_float(dtype))
      return truncate_result(poly_arg_float(arg_to_float(ops[0])), dtype);
    return ops[0];
  }

  /* Tinygrad exec_alu keeps WHERE branch values as-is, which matters for
   * Invalid-carrying index masks. Do this before any numeric coercion. */
  if (op == POLY_OP_WHERE && n_ops == 3) {
    return arg_to_bool(ops[0]) ? ops[1] : ops[2];
  }

  /* Tinygrad preserves Invalid through integer/index binary ALU instead of
   * coercing it to zero. Polygrad uses regular int dtypes in this domain, so
   * we gate on integer output dtype rather than weakint specifically. */
  if (poly_dtype_is_int(dtype) && poly_opset_has(POLY_GROUP_BINARY, op)) {
    for (int i = 0; i < n_ops; i++) {
      if (arg_is_invalid(ops[i])) return poly_arg_invalid();
    }
  }

  if (is_cmp_op(op) && n_ops >= 2) {
    PolyDType cmp_dtype = poly_dtype_scalar(dtype);
    bool use_float = poly_dtype_is_float(cmp_dtype) || ops[0].kind == POLY_ARG_FLOAT ||
                     ops[1].kind == POLY_ARG_FLOAT;
    if (use_float) {
      double a = arg_to_float(ops[0]);
      double b = arg_to_float(ops[1]);
      if (op == POLY_OP_CMPLT) return poly_arg_bool(a < b);
      if (op == POLY_OP_CMPNE) return poly_arg_bool(a != b);
      return poly_arg_bool(a == b);
    }
    if (poly_dtype_is_unsigned(cmp_dtype)) {
      uint64_t a = (uint64_t)truncate_result(ops[0], cmp_dtype).i;
      uint64_t b = (uint64_t)truncate_result(ops[1], cmp_dtype).i;
      if (op == POLY_OP_CMPLT) return poly_arg_bool(a < b);
      if (op == POLY_OP_CMPNE) return poly_arg_bool(a != b);
      return poly_arg_bool(a == b);
    }
    int64_t a = poly_dtype_is_bool(cmp_dtype) ? arg_to_int(ops[0]) : truncate_result(ops[0], cmp_dtype).i;
    int64_t b = poly_dtype_is_bool(cmp_dtype) ? arg_to_int(ops[1]) : truncate_result(ops[1], cmp_dtype).i;
    if (op == POLY_OP_CMPLT) return poly_arg_bool(a < b);
    if (op == POLY_OP_CMPNE) return poly_arg_bool(a != b);
    return poly_arg_bool(a == b);
  }

  /* Float path */
  if (poly_dtype_is_float(dtype)) {
    double a = n_ops > 0 ? arg_to_float(ops[0]) : 0.0;
    double b = n_ops > 1 ? arg_to_float(ops[1]) : 0.0;
    double c = n_ops > 2 ? arg_to_float(ops[2]) : 0.0;
    double r = 0.0;

    switch (op) {
    /* unary */
    case POLY_OP_NEG:
      r = -a;
      break;
    case POLY_OP_EXP2:
      r = safe_exp2(a);
      break;
    case POLY_OP_LOG2:
      r = safe_log2(a);
      break;
    case POLY_OP_SIN:
      r = safe_sin(a);
      break;
    case POLY_OP_SQRT:
      r = safe_sqrt(a);
      break;
    case POLY_OP_RECIPROCAL:
      r = safe_recip(a);
      break;
    case POLY_OP_TRUNC:
      r = trunc(a);
      break;
    /* binary */
    case POLY_OP_ADD:
      r = a + b;
      break;
    case POLY_OP_SUB:
      r = a - b;
      break;
    case POLY_OP_MUL:
      r = a * b;
      break;
    case POLY_OP_FDIV:
      r = b != 0.0 ? a / b : (a == 0.0 ? NAN : copysign(INFINITY, a * b));
      break;
    case POLY_OP_POW:
      r = safe_pow(a, b);
      break;
    case POLY_OP_MAX:
      r = fmax(a, b);
      break;
    case POLY_OP_CMPLT:
    case POLY_OP_CMPNE:
    case POLY_OP_CMPEQ:
      return poly_arg_bool(false);
    /* ternary */
    case POLY_OP_MULACC:
      r = a * b + c;
      break;
    default:
      return poly_arg_float(0.0);
    }

    return truncate_result(poly_arg_float(r), dtype);
  }

  /* Integer path */
  int64_t a = n_ops > 0 ? arg_to_int(ops[0]) : 0;
  int64_t b = n_ops > 1 ? arg_to_int(ops[1]) : 0;
  int64_t c = n_ops > 2 ? arg_to_int(ops[2]) : 0;
  int64_t r = 0;

  switch (op) {
  /* Bool NEG = logical NOT (matches C renderer's !x), not arithmetic -x.
   * Without this, NEG(false) folds to -0=0=false instead of true,
   * breaking PAD validity masks when a dimension has (0,0) padding. */
  case POLY_OP_NEG:
    r = poly_dtype_is_bool(dtype) ? !a : i64_from_u64(0u - (uint64_t)a);
    break;
  case POLY_OP_TRUNC:
    r = a;
    break;
  case POLY_OP_ADD:
    r = i64_from_u64((uint64_t)a + (uint64_t)b);
    break;
  case POLY_OP_SUB:
    r = i64_from_u64((uint64_t)a - (uint64_t)b);
    break;
  case POLY_OP_MUL:
    r = i64_from_u64((uint64_t)a * (uint64_t)b);
    break;
  case POLY_OP_IDIV:
    if (poly_dtype_is_unsigned(dtype)) {
      uint64_t ub = (uint64_t)b;
      r = i64_from_u64(ub == 0 ? 0 : (uint64_t)a / ub);
    } else {
      r = cdiv(a, b);
    }
    break;
  case POLY_OP_MOD:
    if (poly_dtype_is_unsigned(dtype)) {
      uint64_t ub = (uint64_t)b;
      r = i64_from_u64(ub == 0 ? 0 : (uint64_t)a % ub);
    } else {
      r = cmod(a, b);
    }
    break;
  case POLY_OP_MAX:
    r = a > b ? a : b;
    break;
  case POLY_OP_SHL:
    if (b < 0) return poly_arg_invalid();
    r = (b >= 64) ? 0 : i64_from_u64((uint64_t)a << b);
    break;
  case POLY_OP_SHR:
    if (b < 0) return poly_arg_invalid();
    if (poly_dtype_is_unsigned(dtype)) {
      r = (b >= 64) ? 0 : i64_from_u64((uint64_t)a >> b);
    } else if (b >= 64) {
      r = a < 0 ? -1 : 0;
    } else if (b == 0 || a >= 0) {
      r = i64_from_u64((uint64_t)a >> b);
    } else {
      uint64_t shifted = ((uint64_t)a >> b) | (~UINT64_C(0) << (64 - b));
      r = i64_from_u64(shifted);
    }
    break;
  case POLY_OP_XOR:
    r = a ^ b;
    break;
  case POLY_OP_OR:
    r = a | b;
    break;
  case POLY_OP_AND:
    r = a & b;
    break;
  case POLY_OP_CMPLT:
    return poly_arg_bool(a < b);
  case POLY_OP_CMPNE:
    return poly_arg_bool(a != b);
  case POLY_OP_CMPEQ:
    return poly_arg_bool(a == b);
  case POLY_OP_MULACC:
    r = i64_from_u64((uint64_t)a * (uint64_t)b + (uint64_t)c);
    break;
  default:
    return poly_arg_int(0);
  }

  return truncate_result(poly_arg_int(r), dtype);
}
