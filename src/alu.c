/*
 * alu.c — ALU constant-fold executor
 *
 * Mirrors tinygrad's python_alu / exec_alu: evaluates ALU ops on constants.
 * Used by symbolic simplification for constant folding.
 */

#include "uop/upat.h"
#include "bigint.h"
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
  if (b == 0) return a;
  if (a == INT64_MIN && b == -1) return 0;
  return a % b;
}

/* Python-style floor division and modulo. Mirrors tinygrad helpers.py. */
static int64_t floor_div_i64(int64_t a, int64_t b) {
  if (b == 0) return 0;
  if (a == INT64_MIN && b == -1) return INT64_MIN;
  int64_t q = a / b;
  int64_t r = a % b;
  if (r != 0 && ((a < 0) != (b < 0))) q--;
  return q;
}

static int64_t floor_mod_i64(int64_t a, int64_t b) {
  if (b == 0) return a;
  if (a == INT64_MIN && b == -1) return 0;
  int64_t r = a % b;
  return r != 0 && ((r < 0) != (b < 0)) ? r + b : r;
}

static int64_t i64_from_u64(uint64_t v) {
  int64_t out;
  memcpy(&out, &v, sizeof(out));
  return out;
}

static bool i64_add_checked(int64_t a, int64_t b, int64_t *out) {
  return !__builtin_add_overflow(a, b, out);
}

static bool i64_sub_checked(int64_t a, int64_t b, int64_t *out) {
  return !__builtin_sub_overflow(a, b, out);
}

static bool i64_mul_checked(int64_t a, int64_t b, int64_t *out) {
  return !__builtin_mul_overflow(a, b, out);
}

static bool i64_shl_checked(int64_t value, int64_t shift, int64_t *out) {
  if (shift < 0) return false;
  if (value == 0) {
    *out = 0;
    return true;
  }
  if (shift > 63) return false;
  for (int64_t i = 0; i < shift; i++) {
    if (!i64_add_checked(value, value, &value)) return false;
  }
  *out = value;
  return true;
}

static bool i64_pow_checked(int64_t base, int64_t exponent, int64_t *out) {
  if (exponent < 0) return false;
  int64_t result = 1;
  while (exponent) {
    if (exponent & 1) {
      if (!i64_mul_checked(result, base, &result)) return false;
    }
    exponent >>= 1;
    if (exponent && !i64_mul_checked(base, base, &base)) return false;
  }
  *out = result;
  return true;
}

static uint64_t u64_pow_wrapping(uint64_t base, uint64_t exponent) {
  uint64_t result = 1;
  while (exponent) {
    if (exponent & 1) result *= base;
    exponent >>= 1;
    if (exponent) base *= base;
  }
  return result;
}

/* Get numeric value from PolyArg */

static double arg_to_float(PolyArg a) {
  switch (a.kind) {
  case POLY_ARG_FLOAT:
    return a.f;
  case POLY_ARG_INT:
    return (double)a.i;
  case POLY_ARG_BIGINT:
    return poly_arg_integer_to_double(a);
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
  case POLY_ARG_BIGINT:
    return (int64_t)poly_arg_integer_to_u64_mod(a);
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
  case POLY_ARG_BIGINT:
    return a.bigint.n_limbs != 0;
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

/* Pinned tinygrad/uop/symbolic.py:19-24 fold_bitcast uses struct pack/unpack
 * over scalar dtype formats. Keep the same storage reinterpretation here;
 * memcpy avoids C aliasing and numeric-cast semantics. */
bool poly_exec_bitcast_const(PolyDType from, PolyDType to, PolyArg value, PolyArg *out) {
  if (!out || (!from.fmt && !poly_dtype_is_fp8(from)) ||
      (!to.fmt && !poly_dtype_is_fp8(to)) || from.bitsize != to.bitsize ||
      (from.bitsize != 8 && from.bitsize != 16 && from.bitsize != 32 && from.bitsize != 64))
    return false;

  uint64_t bits = 0;
  if (poly_dtype_is_bool(from)) {
    bits = arg_to_bool(value) ? 1u : 0u;
  } else if (poly_dtype_is_float(from)) {
    double v = arg_to_float(value);
    if (poly_dtype_is_fp8(from)) {
      bits = poly_float_to_fp8(v, from);
    } else if (from.bitsize == 16) {
      bits = f32_to_f16_bits_rne((float)v);
    } else if (from.bitsize == 32) {
      float v32 = (float)v;
      uint32_t raw = 0;
      memcpy(&raw, &v32, sizeof(raw));
      bits = raw;
    } else {
      memcpy(&bits, &v, sizeof(bits));
    }
  } else if (poly_dtype_is_int(from)) {
    bits = poly_arg_integer_to_u64_mod(value);
  } else {
    return false;
  }

  if (from.bitsize < 64) bits &= (UINT64_C(1) << from.bitsize) - 1;
  if (poly_dtype_is_bool(to)) {
    *out = poly_arg_bool(bits != 0);
    return true;
  }
  if (poly_dtype_is_float(to)) {
    if (poly_dtype_is_fp8(to)) {
      *out = poly_arg_float(poly_fp8_to_float((uint8_t)bits, to));
    } else if (to.bitsize == 16) {
      *out = poly_arg_float((double)f16_bits_to_f32((uint16_t)bits));
    } else if (to.bitsize == 32) {
      uint32_t raw = (uint32_t)bits;
      float v32 = 0.0f;
      memcpy(&v32, &raw, sizeof(v32));
      *out = poly_arg_float((double)v32);
    } else {
      double v64 = 0.0;
      memcpy(&v64, &bits, sizeof(v64));
      *out = poly_arg_float(v64);
    }
    return true;
  }
  if (!poly_dtype_is_int(to)) return false;

  if (!poly_dtype_is_unsigned(to) && to.bitsize < 64 &&
      (bits & (UINT64_C(1) << (to.bitsize - 1))))
    bits |= ~((UINT64_C(1) << to.bitsize) - 1);
  int64_t signed_bits = 0;
  memcpy(&signed_bits, &bits, sizeof(signed_bits));
  *out = poly_arg_int(signed_bits);
  return true;
}

/* Truncate result to dtype range */

static PolyArg truncate_result(PolyArg val, PolyDType dtype) {
  if (poly_dtype_is_bool(dtype)) return poly_arg_bool(arg_to_bool(val));
  /* Pinned weakint has no entry in dtype.truncate (dtype.py:351-355). */
  if (poly_dtype_is_index(dtype)) return val;

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
    PolyDType sdt = dtype;
    double v = arg_to_float(val);
    if (poly_dtype_is_fp8(sdt))
      v = poly_fp8_to_float(poly_float_to_fp8(v, sdt), sdt);
    else if (sdt.priority == POLY_FLOAT16.priority && sdt.bitsize == 16)
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

PolyArg poly_exec_alu(
    PolyOps op,
    PolyDType dtype,
    PolyArg *ops,
    int n_ops,
    bool truncate_output
) {
  if (op == POLY_OP_CAST && n_ops == 1) {
    if (poly_dtype_is_bool(dtype)) return poly_arg_bool(arg_to_bool(ops[0]));
    if (poly_dtype_is_int(dtype)) {
      PolyArg result = poly_arg_int(arg_to_int(ops[0]));
      return truncate_output ? truncate_result(result, dtype) : result;
    }
    if (poly_dtype_is_float(dtype))
      return truncate_output ? truncate_result(poly_arg_float(arg_to_float(ops[0])), dtype)
                             : poly_arg_float(arg_to_float(ops[0]));
    return ops[0];
  }

  /* Tinygrad exec_alu keeps WHERE branch values as-is, which matters for
   * Invalid-carrying index masks. Do this before any numeric coercion. */
  if (op == POLY_OP_WHERE && n_ops == 3) {
    PolyArg selected = arg_to_bool(ops[0]) ? ops[1] : ops[2];
    return truncate_output ? truncate_result(selected, dtype) : selected;
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
    bool use_float = ops[0].kind == POLY_ARG_FLOAT || ops[1].kind == POLY_ARG_FLOAT;
    if (use_float) {
      double a = arg_to_float(ops[0]);
      double b = arg_to_float(ops[1]);
      if (op == POLY_OP_CMPLT) return poly_arg_bool(a < b);
      if (op == POLY_OP_CMPNE) return poly_arg_bool(a != b);
      return poly_arg_bool(a == b);
    }
    if ((ops[0].kind == POLY_ARG_INT || ops[0].kind == POLY_ARG_BIGINT ||
         ops[0].kind == POLY_ARG_BOOL) &&
        (ops[1].kind == POLY_ARG_INT || ops[1].kind == POLY_ARG_BIGINT ||
         ops[1].kind == POLY_ARG_BOOL)) {
      bool valid = false;
      int cmp = poly_arg_integer_cmp(ops[0], ops[1], &valid);
      if (valid) {
        if (op == POLY_OP_CMPLT) return poly_arg_bool(cmp < 0);
        if (op == POLY_OP_CMPNE) return poly_arg_bool(cmp != 0);
        return poly_arg_bool(cmp == 0);
      }
    }
    int64_t a = arg_to_int(ops[0]);
    int64_t b = arg_to_int(ops[1]);
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

    PolyArg result = poly_arg_float(r);
    return truncate_output ? truncate_result(result, dtype) : result;
  }

  /* Integer path */
  int64_t a = n_ops > 0 ? arg_to_int(ops[0]) : 0;
  int64_t b = n_ops > 1 ? arg_to_int(ops[1]) : 0;
  int64_t c = n_ops > 2 ? arg_to_int(ops[2]) : 0;
  int64_t r = 0;

  switch (op) {
  case POLY_OP_NEG:
    if (!truncate_output && a == INT64_MIN) return poly_arg_invalid();
    r = i64_from_u64(0u - (uint64_t)a);
    break;
  case POLY_OP_TRUNC:
    r = a;
    break;
  case POLY_OP_ADD:
    if (!truncate_output && !i64_add_checked(a, b, &r)) return poly_arg_invalid();
    if (truncate_output) r = i64_from_u64((uint64_t)a + (uint64_t)b);
    break;
  case POLY_OP_SUB:
    if (!truncate_output && !i64_sub_checked(a, b, &r)) return poly_arg_invalid();
    if (truncate_output) r = i64_from_u64((uint64_t)a - (uint64_t)b);
    break;
  case POLY_OP_MUL:
    if (!truncate_output && !i64_mul_checked(a, b, &r)) return poly_arg_invalid();
    if (truncate_output) r = i64_from_u64((uint64_t)a * (uint64_t)b);
    break;
  case POLY_OP_IDIV:
    if (!truncate_output && a == INT64_MIN && b == -1) return poly_arg_invalid();
    r = cdiv(a, b);
    break;
  case POLY_OP_MOD:
    r = cmod(a, b);
    break;
  case POLY_OP_FLOORDIV:
    if (!truncate_output && a == INT64_MIN && b == -1) return poly_arg_invalid();
    r = floor_div_i64(a, b);
    break;
  case POLY_OP_FLOORMOD:
    r = floor_mod_i64(a, b);
    break;
  case POLY_OP_MAX:
    r = a > b ? a : b;
    break;
  case POLY_OP_SHL:
    if (b < 0) return poly_arg_invalid();
    if (!truncate_output) {
      if (!i64_shl_checked(a, b, &r)) return poly_arg_invalid();
    } else {
      r = (b >= 64) ? 0 : i64_from_u64((uint64_t)a << b);
    }
    break;
  case POLY_OP_SHR:
    if (b < 0) return poly_arg_invalid();
    if (b >= 64) {
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
    if (!truncate_output) {
      int64_t product;
      if (!i64_mul_checked(a, b, &product) || !i64_add_checked(product, c, &r))
        return poly_arg_invalid();
    } else {
      r = i64_from_u64((uint64_t)a * (uint64_t)b + (uint64_t)c);
    }
    break;
  case POLY_OP_POW:
    if (b < 0) {
      double value = safe_pow((double)a, (double)b);
      /* Pinned exec_alu first returns Python's float result. Fixed-width
       * integer truncation rejects it; weakint has no truncation function. */
      if (truncate_output && !poly_dtype_is_index(dtype)) return poly_arg_invalid();
      return poly_arg_float(value);
    }
    if (!truncate_output) {
      if (!i64_pow_checked(a, b, &r)) return poly_arg_invalid();
    } else {
      r = i64_from_u64(u64_pow_wrapping((uint64_t)a, (uint64_t)b));
    }
    break;
  default:
    return poly_arg_int(0);
  }

  PolyArg result = poly_arg_int(r);
  return truncate_output ? truncate_result(result, dtype) : result;
}
