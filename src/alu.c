/*
 * alu.c — ALU constant-fold executor
 *
 * Mirrors tinygrad's python_alu / exec_alu: evaluates ALU ops on constants.
 * Used by symbolic simplification for constant folding.
 */

#include "pat.h"
#include <math.h>
#include <stdint.h>
#include <string.h>

/* ── Safe math helpers ────────────────────────────────────────────────── */

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
  return a / b;
}

/* C-style modulo */
static int64_t cmod(int64_t a, int64_t b) {
  if (b == 0) return 0;
  return a % b;
}

/* ── Get numeric value from PolyArg ────────────────────────────────────── */

static double arg_to_float(PolyArg a) {
  switch (a.kind) {
    case POLY_ARG_FLOAT: return a.f;
    case POLY_ARG_INT:   return (double)a.i;
    case POLY_ARG_BOOL:  return a.b ? 1.0 : 0.0;
    default: return 0.0;
  }
}

static int64_t arg_to_int(PolyArg a) {
  switch (a.kind) {
    case POLY_ARG_INT:   return a.i;
    case POLY_ARG_FLOAT: return (int64_t)a.f;
    case POLY_ARG_BOOL:  return a.b ? 1 : 0;
    default: return 0;
  }
}

static bool arg_to_bool(PolyArg a) {
  switch (a.kind) {
    case POLY_ARG_BOOL:  return a.b;
    case POLY_ARG_INT:   return a.i != 0;
    case POLY_ARG_FLOAT: return a.f != 0.0;
    default: return false;
  }
}

/* ── Truncate result to dtype range ───────────────────────────────────── */

static PolyArg truncate_result(PolyArg val, PolyDType dtype) {
  if (poly_dtype_is_bool(dtype))
    return poly_arg_bool(arg_to_bool(val));

  if (poly_dtype_is_int(dtype)) {
    int64_t v = arg_to_int(val);
    int bits = dtype.bitsize;
    if (poly_dtype_is_unsigned(dtype)) {
      if (bits < 64) v &= ((int64_t)1 << bits) - 1;
    } else {
      if (bits < 64) {
        int64_t mask = ((int64_t)1 << bits) - 1;
        v &= mask;
        if (v & ((int64_t)1 << (bits - 1)))
          v |= ~mask; /* sign extend */
      }
    }
    return poly_arg_int(v);
  }

  if (poly_dtype_is_float(dtype)) {
    double v = arg_to_float(val);
    if (dtype.bitsize == 32)
      v = (double)(float)v;
    return poly_arg_float(v);
  }

  return val;
}

/* ── exec_alu: evaluate an ALU op on constant operands ────────────────── */

PolyArg poly_exec_alu(PolyOps op, PolyDType dtype, PolyArg *ops, int n_ops) {
  /* Float path */
  if (poly_dtype_is_float(dtype) || op == POLY_OP_CMPLT ||
      op == POLY_OP_CMPNE || op == POLY_OP_CMPEQ) {
    double a = n_ops > 0 ? arg_to_float(ops[0]) : 0.0;
    double b = n_ops > 1 ? arg_to_float(ops[1]) : 0.0;
    double c = n_ops > 2 ? arg_to_float(ops[2]) : 0.0;
    double r = 0.0;

    switch (op) {
      /* unary */
      case POLY_OP_NEG:        r = -a; break;
      case POLY_OP_EXP2:       r = safe_exp2(a); break;
      case POLY_OP_LOG2:       r = safe_log2(a); break;
      case POLY_OP_SIN:        r = safe_sin(a); break;
      case POLY_OP_SQRT:       r = safe_sqrt(a); break;
      case POLY_OP_RECIPROCAL: r = safe_recip(a); break;
      case POLY_OP_TRUNC:      r = trunc(a); break;
      /* binary */
      case POLY_OP_ADD:  r = a + b; break;
      case POLY_OP_SUB:  r = a - b; break;
      case POLY_OP_MUL:  r = a * b; break;
      case POLY_OP_FDIV: r = b != 0.0 ? a / b : copysign(INFINITY, a * b); break;
      case POLY_OP_POW:  r = safe_pow(a, b); break;
      case POLY_OP_MAX:  r = fmax(a, b); break;
      case POLY_OP_CMPLT: return poly_arg_bool(a < b);
      case POLY_OP_CMPNE: return poly_arg_bool(a != b);
      case POLY_OP_CMPEQ: return poly_arg_bool(a == b);
      /* ternary */
      case POLY_OP_WHERE:  r = arg_to_bool(ops[0]) ? b : c; break;
      case POLY_OP_MULACC: r = a * b + c; break;
      default: return poly_arg_float(0.0);
    }

    if (op == POLY_OP_CMPLT || op == POLY_OP_CMPNE || op == POLY_OP_CMPEQ)
      return poly_arg_bool(r != 0.0);
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
    case POLY_OP_NEG:   r = poly_dtype_is_bool(dtype) ? !a : -a; break;
    case POLY_OP_TRUNC: r = a; break;
    case POLY_OP_ADD:   r = a + b; break;
    case POLY_OP_SUB:   r = a - b; break;
    case POLY_OP_MUL:   r = a * b; break;
    case POLY_OP_IDIV:  r = cdiv(a, b); break;
    case POLY_OP_MOD:   r = cmod(a, b); break;
    case POLY_OP_MAX:   r = a > b ? a : b; break;
    case POLY_OP_SHL:   r = a << b; break;
    case POLY_OP_SHR:   r = (poly_dtype_is_unsigned(dtype)) ? (int64_t)((uint64_t)a >> b) : a >> b; break;
    case POLY_OP_XOR:   r = a ^ b; break;
    case POLY_OP_OR:    r = a | b; break;
    case POLY_OP_AND:   r = a & b; break;
    case POLY_OP_CMPLT: return poly_arg_bool(a < b);
    case POLY_OP_CMPNE: return poly_arg_bool(a != b);
    case POLY_OP_CMPEQ: return poly_arg_bool(a == b);
    case POLY_OP_WHERE:  r = arg_to_bool(ops[0]) ? b : c; break;
    case POLY_OP_MULACC: r = a * b + c; break;
    default: return poly_arg_int(0);
  }

  return truncate_result(poly_arg_int(r), dtype);
}
