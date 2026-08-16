/*
 * sym.c — Symbolic simplification rules (phase 1)
 *
 * Mirrors tinygrad's symbolic_simple: self-folding, zero-folding,
 * constant folding, cast folding, basic identities.
 */

#include "pat.h"
#include "bigint.h"
#include "ctx.h"
#include <math.h>
#include <stdint.h>
#include <limits.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "utils.h"

static int64_t floordiv(int64_t a, int64_t b);

/* Overflow-safe int64 helpers */

static bool i64_add_ok(int64_t a, int64_t b, int64_t *out) {
  return !__builtin_add_overflow(a, b, out);
}
static bool i64_sub_ok(int64_t a, int64_t b, int64_t *out) {
  return !__builtin_sub_overflow(a, b, out);
}
static bool i64_mul_ok(int64_t a, int64_t b, int64_t *out) {
  return !__builtin_mul_overflow(a, b, out);
}
static bool i64_neg_ok(int64_t a, int64_t *out) {
  if (a == INT64_MIN) return false;
  *out = -a;
  return true;
}
static bool i64_shl_ok(int64_t a, int64_t shift, int64_t *out) {
  if (shift < 0 || shift >= 63 || a < 0) return false;
  if (a > (INT64_MAX >> shift)) return false;
  *out = a << shift;
  return true;
}
static bool i64_cdiv_ok(int64_t a, int64_t b, int64_t *out) {
  if (b == 0) {
    *out = 0;
    return true;
  }
  if (a == INT64_MIN && b == -1) return false;
  *out = a / b;
  return true;
}
static bool i64_floordiv_ok(int64_t a, int64_t b, int64_t *out) {
  if (b == 0 || (a == INT64_MIN && b == -1)) return false;
  *out = floordiv(a, b);
  return true;
}

/* Pinned tinygrad symbolic.py:30-32 evaluates scalar constants with
 * exec_alu(..., truncate_output=False), so integer results are Python ints.
 * PolyArg stores signed int64_t only. Preserve every representable result and
 * refuse an unrepresentable fold instead of manufacturing a wrapped CONST. */
static bool arg_is_exact_integer(PolyArg arg) {
  return arg.kind == POLY_ARG_INT || arg.kind == POLY_ARG_BOOL || arg.kind == POLY_ARG_BIGINT;
}

static bool exact_integer_alu(
    PolyOps op,
    PolyDType dtype,
    PolyArg *operands,
    int n_operands,
    bool truncate_output,
    PolyArg *out,
    PolyInt *owned
) {
  if (!out || !owned || n_operands < 1 || n_operands > 3) return false;
  PolyInt values[3] = {{0}};
  PolyInt result = {0};
  bool ok = false;
  for (int i = 0; i < n_operands; i++) {
    if (!poly_int_from_arg(&values[i], operands[i])) goto done;
  }

  if (op == POLY_OP_CMPLT || op == POLY_OP_CMPNE || op == POLY_OP_CMPEQ) {
    int cmp = poly_int_cmp(&values[0], &values[1]);
    *out = poly_arg_bool(op == POLY_OP_CMPLT ? cmp < 0 : op == POLY_OP_CMPNE ? cmp != 0 : cmp == 0);
    ok = true;
    goto done;
  }
  if (op == POLY_OP_WHERE) {
    const PolyInt *selected = poly_int_is_zero(&values[0]) ? &values[2] : &values[1];
    if (!poly_int_copy(&result, selected)) goto done;
  } else if (op == POLY_OP_CAST || op == POLY_OP_TRUNC) {
    if (!poly_int_copy(&result, &values[0])) goto done;
  } else if (op == POLY_OP_NEG) {
    if (!poly_int_neg(&result, &values[0])) goto done;
  } else if (op == POLY_OP_ADD) {
    if (!poly_int_add(&result, &values[0], &values[1])) goto done;
  } else if (op == POLY_OP_SUB) {
    if (!poly_int_sub(&result, &values[0], &values[1])) goto done;
  } else if (op == POLY_OP_MUL) {
    if (!poly_int_mul(&result, &values[0], &values[1])) goto done;
  } else if (op == POLY_OP_MULACC) {
    PolyInt product = {0};
    if (!poly_int_mul(&product, &values[0], &values[1]) ||
        !poly_int_add(&result, &product, &values[2])) {
      poly_int_free(&product);
      goto done;
    }
    poly_int_free(&product);
  } else if (op == POLY_OP_MAX) {
    if (!poly_int_copy(
            &result, poly_int_cmp(&values[0], &values[1]) >= 0 ? &values[0] : &values[1]
        ))
      goto done;
  } else if (op == POLY_OP_AND || op == POLY_OP_OR || op == POLY_OP_XOR) {
    if (!poly_int_bitwise(&result, op, &values[0], &values[1])) goto done;
  } else if (op == POLY_OP_SHL || op == POLY_OP_SHR) {
    uint64_t shift = 0;
    if (poly_int_is_negative(&values[1]) || values[1].n_limbs > 2) goto done;
    shift = poly_int_to_u64_mod(&values[1]);
    if (!(op == POLY_OP_SHL ? poly_int_shl(&result, &values[0], shift)
                            : poly_int_shr(&result, &values[0], shift)))
      goto done;
  } else if (op == POLY_OP_IDIV || op == POLY_OP_MOD || op == POLY_OP_FLOORDIV || op == POLY_OP_FLOORMOD) {
    if (poly_int_is_zero(&values[1])) {
      if (op == POLY_OP_IDIV || op == POLY_OP_FLOORDIV) {
        if (!poly_int_from_i64(&result, 0)) goto done;
      } else if (!poly_int_copy(&result, &values[0])) {
        goto done;
      }
    } else {
      PolyInt quotient = {0}, remainder = {0};
      bool floor_mode = op == POLY_OP_FLOORDIV || op == POLY_OP_FLOORMOD;
      if (!poly_int_divmod(&quotient, &remainder, &values[0], &values[1], floor_mode)) goto done;
      result = op == POLY_OP_IDIV || op == POLY_OP_FLOORDIV ? quotient : remainder;
      if (op == POLY_OP_IDIV || op == POLY_OP_FLOORDIV)
        poly_int_free(&remainder);
      else
        poly_int_free(&quotient);
    }
  } else if (op == POLY_OP_POW) {
    if (poly_int_is_negative(&values[1])) goto done;
    if (!poly_int_pow(&result, &values[0], &values[1])) goto done;
  } else {
    goto done;
  }

  if (truncate_output && !poly_dtype_is_index(dtype)) {
    PolyDType scalar = poly_dtype_scalar(dtype);
    PolyInt truncated = {0};
    if (!poly_int_truncate(&truncated, &result, scalar.bitsize, poly_dtype_is_unsigned(scalar)))
      goto done;
    poly_int_free(&result);
    result = truncated;
  }
  *owned = result;
  poly_int_init(&result);
  *out = poly_int_as_arg(owned);
  ok = true;
done:
  for (int i = 0; i < n_operands; i++)
    poly_int_free(&values[i]);
  poly_int_free(&result);
  return ok;
}

static bool exec_symbolic_const_alu(
    PolyOps op,
    PolyDType dtype,
    PolyArg *operands,
    int n_operands,
    bool truncate_output,
    PolyArg *out,
    PolyInt *owned
) {
  if (!out || !owned) return false;
  poly_int_init(owned);
  bool all_integer = n_operands > 0;
  for (int i = 0; i < n_operands; i++)
    all_integer &= arg_is_exact_integer(operands[i]);
  if (all_integer && op == POLY_OP_POW && n_operands == 2) {
    PolyInt exponent = {0};
    bool negative = poly_int_from_arg(&exponent, operands[1]) && poly_int_is_negative(&exponent);
    poly_int_free(&exponent);
    if (negative) {
      if (operands[0].kind == POLY_ARG_BIGINT || operands[1].kind == POLY_ARG_BIGINT) return false;
      *out = poly_exec_alu(op, dtype, operands, n_operands, truncate_output);
      return true;
    }
  }
  if (all_integer &&
      (poly_dtype_is_int(dtype) || poly_dtype_is_bool(dtype) || op == POLY_OP_CMPLT ||
       op == POLY_OP_CMPNE || op == POLY_OP_CMPEQ || op == POLY_OP_WHERE))
    return exact_integer_alu(op, dtype, operands, n_operands, truncate_output, out, owned);

  /* Integer-to-float CAST is exact at the UOp argument boundary and rounds
   * only through the requested floating dtype, matching DType.const. */
  if (op == POLY_OP_CAST && n_operands == 1 && arg_is_exact_integer(operands[0]) &&
      poly_dtype_is_float(dtype)) {
    *out = poly_arg_float(poly_arg_integer_to_double(operands[0]));
    return true;
  }
  *out = poly_exec_alu(op, dtype, operands, n_operands, truncate_output);
  return true;
}

/* Pinned tinygrad/uop/symbolic.py:128-130 folds scalar constant THREEFRY
 * before late decomposition.  The folded argument is the exact mathematical
 * integer produced by decompositions.py:308-320 followed by simplify(), not a
 * prematurely wrapped uint64.  Keeping this phase ordering is material: the
 * parent uint32 mask/cast performs the intended truncation before float
 * conversion. */
static bool threefry_const_floordiv(PolyInt *out, const PolyInt *a, const PolyInt *b) {
  PolyInt remainder = {0};
  if (!poly_int_divmod(out, &remainder, a, b, true)) return false;
  poly_int_free(&remainder);
  return true;
}

static bool threefry_const_eval(
    PolyArg x_arg,
    PolyArg key_arg,
    PolyArg *out,
    PolyInt *owned
) {
  PolyInt x = {0}, key = {0}, mask = {0}, two32 = {0}, parity = {0};
  PolyInt x0 = {0}, x1 = {0}, key0 = {0}, key1 = {0};
  PolyInt xq = {0}, keyq = {0};
  PolyInt ks[3] = {{0}, {0}, {0}};
  PolyInt xr0 = {0}, xr1 = {0}, tmp0 = {0};
  PolyInt result = {0};
  bool ok = false;

  if (!poly_int_from_arg(&x, x_arg) || !poly_int_from_arg(&key, key_arg) ||
      !poly_int_from_i64(&mask, INT64_C(0xffffffff)) ||
      !poly_int_from_i64(&two32, INT64_C(1) << 32) ||
      !poly_int_from_i64(&parity, INT64_C(0x1BD11BDA)))
    goto done;
  if (!poly_int_bitwise(&x0, POLY_OP_AND, &x, &mask) ||
      !threefry_const_floordiv(&xq, &x, &two32) ||
      !poly_int_bitwise(&x1, POLY_OP_AND, &xq, &mask) ||
      !poly_int_bitwise(&key0, POLY_OP_AND, &key, &mask) ||
      !threefry_const_floordiv(&keyq, &key, &two32) ||
      !poly_int_bitwise(&key1, POLY_OP_AND, &keyq, &mask))
    goto done;

  if (!poly_int_copy(&ks[0], &key1) ||
      !poly_int_bitwise(&tmp0, POLY_OP_XOR, &key0, &key1) ||
      !poly_int_bitwise(&ks[1], POLY_OP_XOR, &tmp0, &parity) ||
      !poly_int_copy(&ks[2], &key0) || !poly_int_add(&xr0, &x0, &ks[2]) ||
      !poly_int_add(&xr1, &x1, &ks[0]))
    goto done;
  poly_int_free(&tmp0);

  static const int rotations[2][4] = {
      {13, 15, 26, 6},
      {17, 29, 16, 24},
  };
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 4; j++) {
      int r = rotations[i & 1][j];
      PolyInt sum = {0}, left_scale = {0}, right_scale = {0};
      PolyInt left = {0}, right = {0}, rotated = {0}, next1 = {0};
      bool round_ok =
          poly_int_add(&sum, &xr0, &xr1) &&
          poly_int_from_i64(&left_scale, INT64_C(1) << r) &&
          poly_int_from_i64(&right_scale, INT64_C(1) << (32 - r)) &&
          poly_int_mul(&left, &xr1, &left_scale) &&
          threefry_const_floordiv(&right, &xr1, &right_scale) &&
          poly_int_add(&rotated, &left, &right) &&
          poly_int_bitwise(&next1, POLY_OP_XOR, &sum, &rotated);
      poly_int_free(&left_scale);
      poly_int_free(&right_scale);
      poly_int_free(&left);
      poly_int_free(&right);
      poly_int_free(&rotated);
      if (!round_ok) {
        poly_int_free(&sum);
        poly_int_free(&next1);
        goto done;
      }
      poly_int_free(&xr0);
      poly_int_free(&xr1);
      xr0 = sum;
      xr1 = next1;
    }

    PolyInt round_add = {0}, next0 = {0}, keyed1 = {0}, next1 = {0};
    bool inject_ok = poly_int_from_i64(&round_add, i + 1) &&
                     poly_int_add(&next0, &xr0, &ks[i % 3]) &&
                     poly_int_add(&keyed1, &xr1, &ks[(i + 1) % 3]) &&
                     poly_int_add(&next1, &keyed1, &round_add);
    poly_int_free(&round_add);
    poly_int_free(&keyed1);
    if (!inject_ok) {
      poly_int_free(&next0);
      poly_int_free(&next1);
      goto done;
    }
    poly_int_free(&xr0);
    poly_int_free(&xr1);
    xr0 = next0;
    xr1 = next1;
  }

  if (!poly_int_mul(&tmp0, &xr1, &two32) ||
      !poly_int_bitwise(&result, POLY_OP_OR, &tmp0, &xr0))
    goto done;
  *owned = result;
  poly_int_init(&result);
  *out = poly_int_as_arg(owned);
  ok = true;

done:
  poly_int_free(&x);
  poly_int_free(&key);
  poly_int_free(&mask);
  poly_int_free(&two32);
  poly_int_free(&parity);
  poly_int_free(&x0);
  poly_int_free(&x1);
  poly_int_free(&key0);
  poly_int_free(&key1);
  poly_int_free(&xq);
  poly_int_free(&keyq);
  for (int i = 0; i < 3; i++)
    poly_int_free(&ks[i]);
  poly_int_free(&xr0);
  poly_int_free(&xr1);
  poly_int_free(&tmp0);
  poly_int_free(&result);
  return ok;
}

static PolyUOp *rule_threefry_const(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  if (!root || root->op != POLY_OP_THREEFRY || root->n_src != 2 ||
      root->dtype.count != 1 || !poly_dtype_is_int(root->dtype) ||
      root->src[0]->op != POLY_OP_CONST || root->src[1]->op != POLY_OP_CONST ||
      !arg_is_exact_integer(root->src[0]->arg) || !arg_is_exact_integer(root->src[1]->arg))
    return NULL;
  PolyArg value;
  PolyInt owned = {0};
  if (!threefry_const_eval(root->src[0]->arg, root->src[1]->arg, &value, &owned)) return NULL;
  PolyUOp *ret = poly_const_like(ctx, root, value);
  poly_int_free(&owned);
  return ret;
}

/* vmin/vmax bounds (port of tinygrad's UOp._min_max) *
 * Full port of the pinned tinygrad/uop/ops.py UOp._min_max. The switch is
 * gated on GroupOp.Binary with an outer `not dtypes.is_float(self.dtype)`.
 * Polygrad
 * mirrors the exact case ordering and guards; see inline tinygrad refs.
 *
 * Ops covered:
 *   Binary block (non-float dtype):
 *     ADD  line 860
 *     SUB  line 861
 *     AND  line 862 (int with non-neg const)
 *     MUL  line 864 (4-corner)
 *     SHL  line 866 (const rhs)
 *     SHR  line 867 (const rhs)
 *     MOD  lines 868-872 (three cases by divisor sign)
 *     IDIV lines 873-876 (sign-definite divisor only)
 *     XOR  line 877 (with -1 only: bitwise NOT)
 *     MAX  line 878
 *     CMPLT line 879
 *     CMPNE line 880
 *     OR   line 881 (bool dtype)
 *     AND  line 882 (bool dtype)
 *   Post-Binary:
 *     WHERE        line 884 (int dtype only)
 *     DEFINE_VAR   line 887
 *     RANGE/SPECIAL line 888 (src[0].vmax - 1 via identity (s-1).vmax)
 *     BIND         line 889 (passthrough src[0])
 *     UNROLL/STACK (min/max over srcs; VECTORIZE aliases STACK in Polygrad)
 *     CONST        line 891
 *     VCONST       line 892
 *     GEP          line 893 (passthrough src[0])
 *     CAST         lines 895-896 (monotone targets only: float/signed int)
 *
 * Not ported (deliberate divergences):
 *   - NEG: tinygrad's _min_max has no NEG case; falls through to dtype
 *     bounds. Polygrad does likewise for strict parity.
 *   - CMPEQ: in GroupOp.Binary but no dispatch case in tinygrad's switch
 *     (only CMPLT and CMPNE are handled). Falls through to dtype bounds.
 *   - BITCAST: not monotone, falls through to dtype bounds.
 *   - PARAM: tinygrad's rule reads src[2].arg/src[3].arg; polygrad's
 *     POLY_OP_PARAM (rangeify.c:1582) is a zero-src buffer index with
 *     no bounds info. Falls through to dtype bounds.
 *   - Float dtype: conservative int64 sentinel bounds. Float constants do not
 *     narrow those bounds because infinities, NaNs, and fractional values are
 *     not representable in this integer-only cache. Constant expressions are
 *     handled by the ordinary ALU constant-fold rules instead.
 *
 * Parity: test/parity_scripts/tg_minmax_gt.py captures the ground truth,
 * test/test_sym.c asserts every case verbatim. */

static int64_t dtype_min(PolyDType dt) {
  dt = poly_dtype_scalar(dt);
  if (poly_dtype_eq(dt, POLY_BOOL)) return 0;
  if (poly_dtype_eq(dt, POLY_INT8)) return INT8_MIN;
  if (poly_dtype_eq(dt, POLY_UINT8)) return 0;
  if (poly_dtype_eq(dt, POLY_INT16)) return INT16_MIN;
  if (poly_dtype_eq(dt, POLY_UINT16)) return 0;
  if (poly_dtype_eq(dt, POLY_INT32)) return INT32_MIN;
  if (poly_dtype_eq(dt, POLY_UINT32)) return 0;
  if (poly_dtype_eq(dt, POLY_INT64)) return INT64_MIN;
  if (poly_dtype_eq(dt, POLY_UINT64)) return 0;
  /* float / index / unknown: conservative */
  return INT64_MIN / 2;
}

static int64_t dtype_max(PolyDType dt) {
  dt = poly_dtype_scalar(dt);
  if (poly_dtype_eq(dt, POLY_BOOL)) return 1;
  if (poly_dtype_eq(dt, POLY_INT8)) return INT8_MAX;
  if (poly_dtype_eq(dt, POLY_UINT8)) return UINT8_MAX;
  if (poly_dtype_eq(dt, POLY_INT16)) return INT16_MAX;
  if (poly_dtype_eq(dt, POLY_UINT16)) return UINT16_MAX;
  if (poly_dtype_eq(dt, POLY_INT32)) return INT32_MAX;
  if (poly_dtype_eq(dt, POLY_UINT32)) return UINT32_MAX;
  if (poly_dtype_eq(dt, POLY_INT64)) return INT64_MAX;
  if (poly_dtype_eq(dt, POLY_UINT64)) return INT64_MAX; /* clamped */
  return INT64_MAX / 2;
}

static int64_t min4(int64_t a, int64_t b, int64_t c, int64_t d) {
  int64_t x = a < b ? a : b;
  int64_t y = c < d ? c : d;
  return x < y ? x : y;
}
static int64_t max4(int64_t a, int64_t b, int64_t c, int64_t d) {
  int64_t x = a > b ? a : b;
  int64_t y = c > d ? c : d;
  return x > y ? x : y;
}
static int64_t i64_min(int64_t a, int64_t b) {
  return a < b ? a : b;
}
static int64_t i64_max(int64_t a, int64_t b) {
  return a > b ? a : b;
}

static bool minmax_memo_get(PolyUOp *u, int64_t *lo, int64_t *hi) {
  if (!u) return false;
  if (u->minmax_cached) {
    *lo = u->minmax_vmin;
    *hi = u->minmax_vmax;
    return true;
  }
  return false;
}

static void minmax_memo_set(PolyUOp *u, int64_t lo, int64_t hi) {
  if (!u) return;
  u->minmax_vmin = lo;
  u->minmax_vmax = hi;
  u->minmax_cached = true;
}

static bool minmax_src(PolyUOp *s, int64_t *lo, int64_t *hi) {
  return minmax_memo_get(s, lo, hi);
}

static void minmax_default(PolyUOp *u, int64_t *vmin, int64_t *vmax) {
  if (!u) {
    *vmin = 0;
    *vmax = 0;
    return;
  }
  *vmin = dtype_min(u->dtype);
  *vmax = dtype_max(u->dtype);
}

static bool poly_uop_minmax_node(PolyCtx *ctx, PolyUOp *u, int64_t *vmin, int64_t *vmax) {
  (void)ctx;
  if (!u) {
    *vmin = 0;
    *vmax = 0;
    return true;
  }

  /* CONST */
  if (u->op == POLY_OP_CONST) {
    if (poly_dtype_is_float(u->dtype)) {
      /* This cache cannot represent tinygrad's float [-inf, +inf] bounds or
       * preserve NaN/fractional constants. Keep float min/max conservative;
       * casting those values to int64_t is both lossy and undefined for
       * infinities/out-of-range values. */
      *vmin = dtype_min(u->dtype);
      *vmax = dtype_max(u->dtype);
    } else if (poly_dtype_is_bool(u->dtype)) {
      *vmin = *vmax = u->arg.b ? 1 : 0;
    } else if (u->arg.kind == POLY_ARG_INT) {
      /* Pinned tinygrad ops.py:1010-1017 returns CONST.arg directly. */
      if (poly_dtype_is_unsigned(u->dtype) && poly_dtype_scalar(u->dtype).bitsize == 64 &&
          u->arg.i < 0) {
        minmax_default(u, vmin, vmax);
      } else {
        *vmin = *vmax = u->arg.i;
      }
    } else
      minmax_default(u, vmin, vmax);
    return true;
  }

  /* VCONST — child CONSTs in src[]. Min/max over lanes. */
  if (u->op == POLY_OP_VCONST && u->n_src > 0) {
    int64_t lo = INT64_MAX, hi = INT64_MIN;
    for (int i = 0; i < u->n_src; i++) {
      int64_t a, b;
      if (!minmax_src(u->src[i], &a, &b)) return false;
      if (a < lo) lo = a;
      if (b > hi) hi = b;
    }
    *vmin = lo;
    *vmax = hi;
    return true;
  }

  /* DEFINE_VAR: (name, min_val, max_val) */
  if (u->op == POLY_OP_DEFINE_VAR && u->arg.kind == POLY_ARG_DEFINE_VAR) {
    *vmin = u->arg.define_var.min_val;
    *vmax = u->arg.define_var.max_val;
    return true;
  }

  /* Pinned tinygrad UOp._min_max returns ParamArg.vmin_vmax directly
   * (tinygrad/uop/ops.py:1010). This is the ordinary int64 query equivalent
   * of exact_int_range_node's existing arbitrary-precision PARAM rule. */
  if (u->op == POLY_OP_PARAM && u->arg.kind == POLY_ARG_PARAM && u->arg.param &&
      u->arg.param->has_minmax) {
    *vmin = u->arg.param->min_val;
    *vmax = u->arg.param->max_val;
    return true;
  }

  /* RANGE / SPECIAL: tinygrad ops.py:888
   *   if self.op in (Ops.RANGE, Ops.SPECIAL): return 0, (self.src[0]-1).vmax
   * Tinygrad constructs (src[0] - 1) as a real UOp and recursively queries
   * its vmax. SUB's rule (line 861) gives `s0_vmax - s1_vmin`; with
   * s1 = CONST(1) (vmin=vmax=1), that simplifies to `src[0].vmax - 1`,
   * which is mathematically identical and avoids the allocation. */
  if ((u->op == POLY_OP_RANGE || u->op == POLY_OP_SPECIAL) && u->n_src >= 1) {
    int64_t lo, hi;
    if (!minmax_src(u->src[0], &lo, &hi)) return false;
    *vmin = 0;
    *vmax = hi - 1;
    return true;
  }

  /* BIND: passthrough src[0] (ignore bound value) */
  if (u->op == POLY_OP_BIND && u->n_src >= 1) {
    return minmax_src(u->src[0], vmin, vmax);
  }

  /* GEP: passthrough src[0] */
  if (u->op == POLY_OP_GEP && u->n_src >= 1) {
    return minmax_src(u->src[0], vmin, vmax);
  }

  /* UNROLL/VECTORIZE: min/max over all srcs */
  if ((u->op == POLY_OP_UNROLL || u->op == POLY_OP_VECTORIZE) && u->n_src > 0) {
    int64_t lo = INT64_MAX, hi = INT64_MIN;
    for (int i = 0; i < u->n_src; i++) {
      int64_t a, b;
      if (!minmax_src(u->src[i], &a, &b)) return false;
      if (a < lo) lo = a;
      if (b > hi) hi = b;
    }
    *vmin = lo;
    *vmax = hi;
    return true;
  }

  /* Binary ops — gated on `not is_float` to match tinygrad ops.py:858.
   * Float-dtype binary ops fall through to the dtype-bounds default; their
   * NaN handling makes interval arithmetic unsafe. CMPLT/CMPNE on float
   * operands are dispatched separately below since their result is bool. */
  if (u->n_src == 2 && !poly_dtype_is_float(u->dtype)) {
    int64_t a0, a1, b0, b1;
    if (!minmax_src(u->src[0], &a0, &a1)) return false;
    if (!minmax_src(u->src[1], &b0, &b1)) return false;

    if (u->op == POLY_OP_ADD) {
      int64_t lo, hi;
      /* Pinned tinygrad/uop/ops.py:858-861 tracks mathematical integer
       * intervals with Python ints; it does not clamp to the storage dtype. */
      if (i64_add_ok(a0, b0, &lo) && i64_add_ok(a1, b1, &hi)) {
        *vmin = lo;
        *vmax = hi;
        return true;
      }
      /* overflow: fall through */
    }
    if (u->op == POLY_OP_SUB) {
      int64_t lo, hi;
      if (i64_sub_ok(a0, b1, &lo) && i64_sub_ok(a1, b0, &hi)) {
        *vmin = lo;
        *vmax = hi;
        return true;
      }
      /* overflow: fall through */
    }
    if (u->op == POLY_OP_MUL) {
      int64_t v0, v1, v2, v3;
      if (i64_mul_ok(a0, b0, &v0) && i64_mul_ok(a0, b1, &v1) && i64_mul_ok(a1, b0, &v2) &&
          i64_mul_ok(a1, b1, &v3)) {
        int64_t lo = min4(v0, v1, v2, v3), hi = max4(v0, v1, v2, v3);
        *vmin = lo;
        *vmax = hi;
        return true;
      }
      /* overflow: fall through */
    }
    if (u->op == POLY_OP_MAX) {
      *vmin = i64_max(a0, b0);
      *vmax = i64_max(a1, b1);
      return true;
    }
    if (u->op == POLY_OP_MOD) {
      /* tinygrad ops.py:868-872 */
      if (b0 == b1 && b0 > 0) {
        int64_t c = b0;
        int64_t lo = (a0 > 0) ? 0 : (a0 >= -c + 1 && a0 <= 0 ? a0 : -(c - 1));
        int64_t hi = (a1 < 0) ? 0 : (a1 >= 0 && a1 < c ? a1 : c - 1);
        *vmin = lo;
        *vmax = hi;
        return true;
      }
      if (b0 > 0) {
        if (a0 >= 0) {
          *vmin = 0;
          *vmax = b1 - 1;
        } else if (a1 <= 0) {
          *vmin = -(b1 - 1);
          *vmax = 0;
        } else {
          *vmin = -(b1 - 1);
          *vmax = b1 - 1;
        }
        return true;
      }
      if (b1 < 0) {
        /* `-b0 - 1` is mathematically correct but negates INT64_MIN before
         * subtracting one. This equivalent ordering stays representable for
         * every strictly-negative int64 bound. */
        int64_t m = -(b0 + 1);
        if (a0 >= 0) {
          *vmin = 0;
          *vmax = m;
        } else if (a1 <= 0) {
          *vmin = -m;
          *vmax = 0;
        } else {
          *vmin = -m;
          *vmax = m;
        }
        return true;
      }
    }
    if (u->op == POLY_OP_FLOORMOD) {
      if (a0 > a1) {
        *vmin = 0;
        *vmax = 0;
        return true;
      }
      if (b0 == b1 && b0 > 0) {
        *vmin = 0;
        *vmax = b0 - 1;
        return true;
      }
      if (b0 > 0) {
        *vmin = 0;
        *vmax = b1 - 1;
        return true;
      }
      if (b1 < 0) {
        *vmin = b0 + 1;
        *vmax = 0;
        return true;
      }
    }
    if (u->op == POLY_OP_IDIV) {
      /* Only handle the case where the divisor sign is known */
      /* Tinygrad ops.py:875 uses `s1_vmin*s1_vmax>0` which can overflow
       * int64. The same-sign check below is equivalent and overflow-safe;
       * matches the idiom already used in fold_divmod_general. */
      if ((b0 > 0 && b1 > 0) || (b0 < 0 && b1 < 0)) {
        int64_t v0, v1, v2, v3;
        if (i64_cdiv_ok(a0, b0, &v0) && i64_cdiv_ok(a0, b1, &v1) && i64_cdiv_ok(a1, b0, &v2) &&
            i64_cdiv_ok(a1, b1, &v3)) {
          int64_t lo = min4(v0, v1, v2, v3), hi = max4(v0, v1, v2, v3);
          *vmin = lo;
          *vmax = hi;
          return true;
        }
      }
    }
    if (u->op == POLY_OP_FLOORDIV) {
      if (a0 > a1) {
        *vmin = 0;
        *vmax = 0;
        return true;
      }
      if ((b0 > 0 && b1 > 0) || (b0 < 0 && b1 < 0)) {
        int64_t v0, v1, v2, v3;
        if (i64_floordiv_ok(a0, b0, &v0) && i64_floordiv_ok(a0, b1, &v1) &&
            i64_floordiv_ok(a1, b0, &v2) && i64_floordiv_ok(a1, b1, &v3)) {
          int64_t lo = min4(v0, v1, v2, v3), hi = max4(v0, v1, v2, v3);
          *vmin = lo;
          *vmax = hi;
          return true;
        }
      }
    }
    if (u->op == POLY_OP_SHL && b0 == b1 && b0 >= 0 && b0 < 63) {
      int64_t v0, v1;
      if (i64_shl_ok(a0, b0, &v0) && i64_shl_ok(a1, b0, &v1)) {
        int64_t lo = i64_min(v0, v1), hi = i64_max(v0, v1);
        *vmin = lo;
        *vmax = hi;
        return true;
      }
      /* overflow or negative lhs: fall through */
    }
    if (u->op == POLY_OP_SHR && b0 == b1 && b0 >= 0 && b0 < 63) {
      *vmin = a0 >> b0;
      *vmax = a1 >> b0;
      return true;
    }
    if (u->op == POLY_OP_XOR && b0 == b1 && b0 == -1) {
      /* ~x: bitwise not */
      *vmin = ~a1;
      *vmax = ~a0;
      return true;
    }
    if (u->op == POLY_OP_AND && poly_dtype_is_int(u->dtype) && b0 == b1 && b0 >= 0) {
      /* tinygrad ops.py:862-863:
       *   if self.op is Ops.AND and dtypes.is_int(self.dtype)
       *      and s1_vmin == s1_vmax >= 0:
       *     return 0, s1_vmax if s0_vmin < 0 else min(s0_vmax, s1_vmax) */
      *vmin = 0;
      *vmax = (a0 < 0) ? b1 : i64_min(a1, b1);
      return true;
    }
  }

  /* Bool binary ops (AND / OR on bool dtype) */
  if (u->n_src == 2 && poly_dtype_eq(u->dtype, POLY_BOOL)) {
    int64_t a0, a1, b0, b1;
    if (!minmax_src(u->src[0], &a0, &a1)) return false;
    if (!minmax_src(u->src[1], &b0, &b1)) return false;
    if (u->op == POLY_OP_AND) {
      *vmin = (a0 && b0) ? 1 : 0;
      *vmax = (a1 && b1) ? 1 : 0;
      return true;
    }
    if (u->op == POLY_OP_OR) {
      *vmin = (a0 || b0) ? 1 : 0;
      *vmax = (a1 || b1) ? 1 : 0;
      return true;
    }
  }

  /* Comparisons: always bool, regardless of operand dtype */
  if (u->n_src == 2 && (u->op == POLY_OP_CMPLT || u->op == POLY_OP_CMPNE)) {
    int64_t a0, a1, b0, b1;
    if (!minmax_src(u->src[0], &a0, &a1)) return false;
    if (!minmax_src(u->src[1], &b0, &b1)) return false;
    if (u->op == POLY_OP_CMPLT) {
      *vmin = (a1 < b0) ? 1 : 0;
      *vmax = (a0 < b1) ? 1 : 0;
      return true;
    }
    /* CMPNE */
    bool def_ne = (a1 < b0) || (b1 < a0);
    bool all_eq = (a0 == a1) && (b0 == b1) && (a0 == b0);
    *vmin = def_ne ? 1 : 0;
    *vmax = all_eq ? 0 : 1;
    return true;
  }

  /* WHERE (int branches): min/max over both branches */
  if (u->op == POLY_OP_WHERE && u->n_src == 3 && poly_dtype_is_int(u->dtype)) {
    int64_t t0, t1, f0, f1;
    if (!minmax_src(u->src[1], &t0, &t1)) return false;
    if (!minmax_src(u->src[2], &f0, &f1)) return false;
    *vmin = i64_min(t0, f0);
    *vmax = i64_max(t1, f1);
    return true;
  }

  /* CAST: clamp src[0] bounds to dtype range. Matches tinygrad ops.py:895 —
   * only monotone casts. Cast to bool/unsigned is not necessarily monotone;
   * fall through to dtype bounds for those. */
  if (u->op == POLY_OP_CAST && u->n_src >= 1) {
    bool monotone = poly_dtype_is_float(u->dtype) ||
                    (poly_dtype_is_int(u->dtype) && !poly_dtype_is_unsigned(u->dtype) &&
                     !poly_dtype_eq(u->dtype, POLY_BOOL));
    if (monotone) {
      int64_t a0, a1;
      if (!minmax_src(u->src[0], &a0, &a1)) return false;
      *vmin = i64_max(dtype_min(u->dtype), a0);
      *vmax = i64_min(a1, dtype_max(u->dtype));
      return true;
    }
  }

  /* Fallback: dtype range */
  *vmin = dtype_min(u->dtype);
  *vmax = dtype_max(u->dtype);
  return true;
}

static bool poly_uop_minmax_rec_fast(
    PolyCtx *ctx,
    PolyUOp *u,
    int64_t *vmin,
    int64_t *vmax,
    int depth
) {
  if (!u) {
    *vmin = 0;
    *vmax = 0;
    return true;
  }
  if (minmax_memo_get(u, vmin, vmax)) return true;
  if (depth > 128) return false;

  for (int i = 0; i < u->n_src; i++) {
    int64_t lo = 0, hi = 0;
    if (!poly_uop_minmax_rec_fast(ctx, u->src[i], &lo, &hi, depth + 1)) return false;
  }

  int64_t lo = 0, hi = 0;
  if (!poly_uop_minmax_node(ctx, u, &lo, &hi)) minmax_default(u, &lo, &hi);
  minmax_memo_set(u, lo, hi);
  *vmin = lo;
  *vmax = hi;
  return true;
}

static void poly_uop_minmax_compute(PolyCtx *ctx, PolyUOp *u, int64_t *vmin, int64_t *vmax) {
  if (!u) {
    *vmin = 0;
    *vmax = 0;
    return;
  }

  if (minmax_memo_get(u, vmin, vmax)) return;

  if (poly_uop_minmax_rec_fast(ctx, u, vmin, vmax, 0)) return;

  typedef struct {
    PolyUOp *u;
    int state;
  } MinMaxFrame;
  MinMaxFrame stack_buf[128];
  int stack_cap = (int)(sizeof(stack_buf) / sizeof(stack_buf[0]));
  int stack_top = 0;
  MinMaxFrame *stack = stack_buf;

  stack[stack_top++] = (MinMaxFrame){u, 0};
  bool ok = true;
  while (ok && stack_top > 0) {
    MinMaxFrame *frame = &stack[stack_top - 1];
    PolyUOp *cur = frame->u;
    int64_t lo, hi;

    if (!cur || minmax_memo_get(cur, &lo, &hi)) {
      stack_top--;
      continue;
    }

    if (frame->state == 0) {
      frame->state = 1;
      for (int i = cur->n_src - 1; i >= 0; i--) {
        PolyUOp *src = cur->src[i];
        if (!src || minmax_memo_get(src, &lo, &hi)) continue;
        if (stack_top >= stack_cap) {
          int new_cap = stack_cap * 2;
          MinMaxFrame *new_stack = NULL;
          if (stack == stack_buf)
            new_stack = malloc((size_t)new_cap * sizeof(MinMaxFrame));
          else
            new_stack = realloc(stack, (size_t)new_cap * sizeof(MinMaxFrame));
          if (!new_stack) {
            ok = false;
            break;
          }
          if (stack == stack_buf)
            memcpy(new_stack, stack_buf, (size_t)stack_top * sizeof(MinMaxFrame));
          stack = new_stack;
          stack_cap = new_cap;
          frame = &stack[stack_top - 1];
        }
        stack[stack_top++] = (MinMaxFrame){src, 0};
      }
      continue;
    }

    if (!poly_uop_minmax_node(ctx, cur, &lo, &hi)) minmax_default(cur, &lo, &hi);
    minmax_memo_set(cur, lo, hi);
    stack_top--;
  }

  if (!ok || !minmax_memo_get(u, vmin, vmax)) minmax_default(u, vmin, vmax);
  if (stack != stack_buf) free(stack);
}

/* Public wrapper. Caches on the immutable UOp itself, matching tinygrad's
 * cached UOp properties. */
void poly_uop_minmax(PolyCtx *ctx, PolyUOp *u, int64_t *vmin, int64_t *vmax) {
  if (!vmin || !vmax) return;
  if (u && u->minmax_cached) {
    *vmin = u->minmax_vmin;
    *vmax = u->minmax_vmax;
    return;
  }
  poly_uop_minmax_compute(ctx, u, vmin, vmax);
}

void poly_uop_minmax_ex(
    PolyCtx *ctx,
    PolyUOp *u,
    PolyUOpCache *cache,
    int64_t *vmin,
    int64_t *vmax
) {
  (void)cache;
  if (!vmin || !vmax) return;
  poly_uop_minmax(ctx, u, vmin, vmax);
}

/* Rewrite callbacks */

/* Self-folding: return x */
static PolyUOp *rule_identity(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)ctx;
  (void)root;
  return poly_bind(b, "x");
}

/* x+0 -> x, x*1 -> x, x^0 -> x, x//1 -> x */
/* All use rule_identity — pattern does the matching */

/* x//x -> 1 */
static PolyUOp *rule_div_self(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  return poly_const_like_int(ctx, poly_bind(b, "x"), 1);
}

/* x//-1 -> -x for FLOORDIV, matching tinygrad's `//` rule. For unsigned
 * dtypes an arg of -1 is the all-ones value, not the numeric divisor -1. */
static PolyUOp *rule_div_neg1(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  if (poly_dtype_is_unsigned(root->dtype)) return NULL;
  PolyUOp *x = poly_bind(b, "x");
  int64_t x_min = 0, x_max = 0;
  poly_uop_minmax(ctx, x, &x_min, &x_max);
  (void)x_max;
  if (x_min <= dtype_min(x->dtype)) return NULL;
  return poly_uop1(ctx, POLY_OP_NEG, x->dtype, x, poly_arg_none());
}

/* Idempotent(x, x) -> x (OR, AND, MAX) */
static PolyUOp *rule_idempotent(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)ctx;
  (void)root;
  return poly_bind(b, "x");
}

/* TRUNC(x) -> x for integer, bool, and weak-index dtypes.
 * Pinned tinygrad/uop/symbolic.py:114 applies this identity before rendering;
 * integral truncation has no effect and must not reach backend float TRUNC
 * instructions. */
static PolyUOp *rule_trunc_integral_identity(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)ctx;
  (void)root;
  PolyUOp *x = poly_bind(b, "x");
  if (!x || (!poly_dtype_is_int(x->dtype) && !poly_dtype_is_bool(x->dtype))) return NULL;
  return x;
}

/* Zero-folding: x < x -> False */
static PolyUOp *false_comparison_like(PolyCtx *ctx, PolyUOp *x) {
  if (!ctx || !x) return NULL;
  /* Pinned tinygrad/uop/symbolic.py:107-108,120-121 first constructs the
   * false value like the operand, then casts it to the comparison's scalar or
   * vector bool dtype. Omitting the CAST changes CMPLT/CMPNE<bool> into an
   * integer CONST when x is integer. */
  PolyUOp *value = poly_const_like_bool(ctx, x, false);
  if (!value) return NULL;
  PolyDType bool_dtype =
      x->dtype.count > 1 ? poly_dtype_vec(POLY_BOOL, x->dtype.count) : POLY_BOOL;
  return poly_dtype_eq(value->dtype, bool_dtype)
             ? value
             : poly_uop1(ctx, POLY_OP_CAST, bool_dtype, value, poly_arg_none());
}

static PolyUOp *rule_lt_self(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  return false_comparison_like(ctx, poly_bind(b, "x"));
}

/* x != x -> false (int/bool only; float NaN!=NaN is true) */
static PolyUOp *rule_cmpne_self(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  PolyUOp *x = poly_bind(b, "x");
  if (poly_dtype_is_float(x->dtype)) return NULL; /* NaN != NaN */
  return false_comparison_like(ctx, x);
}

/* x % x -> 0 */
static PolyUOp *rule_mod_self(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  return poly_const_like_int(ctx, poly_bind(b, "x"), 0);
}

/* x ^ x -> 0 */
static PolyUOp *rule_xor_self(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  return poly_const_like_int(ctx, poly_bind(b, "x"), 0);
}

/* x * 0 -> 0 (simplified: ignore nan/inf edge cases for now) */
static PolyUOp *rule_mul_zero(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  PolyUOp *x = poly_bind(b, "x");
  /* If x is a float const that is nan or inf, result should be nan */
  if (x->op == POLY_OP_CONST && x->arg.kind == POLY_ARG_FLOAT &&
      (isnan(x->arg.f) || isinf(x->arg.f)))
    return poly_const_like_float(ctx, x, NAN);
  return poly_const_like_int(ctx, root, 0);
}

/* bool * bool -> AND */
static PolyUOp *rule_bool_mul_to_and(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  if (!poly_dtype_is_bool(root->dtype)) return NULL;
  PolyUOp *x = poly_bind(b, "x");
  PolyUOp *y = poly_bind(b, "y");
  return poly_uop2(ctx, POLY_OP_AND, root->dtype, x, y, poly_arg_none());
}

/* bool + bool -> OR */
static PolyUOp *rule_bool_add_to_or(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  if (!poly_dtype_is_bool(root->dtype)) return NULL;
  PolyUOp *x = poly_bind(b, "x");
  PolyUOp *y = poly_bind(b, "y");
  return poly_uop2(ctx, POLY_OP_OR, root->dtype, x, y, poly_arg_none());
}

/* AND(x, false) -> false */
static PolyUOp *rule_and_zero(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  return poly_const_like_bool(ctx, poly_bind(b, "x"), false);
}

/* OR(x, true) -> true */
static PolyUOp *rule_or_one(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  return poly_const_like_bool(ctx, poly_bind(b, "x"), true);
}

/* ALU/variable min==max -> CONST.
 * tinygrad symbolic.py:
 *   UPat({Ops.CMPLT, Ops.CMPNE, Ops.FLOORDIV, Ops.FLOORMOD,
 *         Ops.PARAM, Ops.BIND, Ops.SPECIAL})
 *     -> CONST when vmin == vmax.
 */
static PolyUOp *rule_const_when_minmax_point(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  int64_t vmin = 0, vmax = 0;
  poly_uop_minmax(ctx, root, &vmin, &vmax);
  if (vmin != vmax) return NULL;
  if (poly_dtype_eq(root->dtype, POLY_BOOL)) return poly_const_like_bool(ctx, root, vmin != 0);
  if (poly_dtype_is_float(root->dtype)) return poly_const_like_float(ctx, root, (double)vmin);
  return poly_const_like_int(ctx, root, vmin);
}

/* max folding
 *   maximum(x, y) -> x if x.vmin >= y.vmax else y if x.vmax <= y.vmin
 * tinygrad uop/symbolic.py:243 */
static PolyUOp *rule_max_fold(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  if (!root || root->op != POLY_OP_MAX || root->n_src != 2) return NULL;
  int64_t x_min = 0, x_max = 0, y_min = 0, y_max = 0;
  poly_uop_minmax(ctx, root->src[0], &x_min, &x_max);
  poly_uop_minmax(ctx, root->src[1], &y_min, &y_max);
  if (x_min >= y_max) return root->src[0];
  if (x_max <= y_min) return root->src[1];
  return NULL;
}

static bool const_lane_arg(PolyUOp *u, int lane, PolyArg *out) {
  if (!u || !out || lane < 0) return false;
  if (u->op == POLY_OP_CONST) {
    *out = u->arg;
    return true;
  }
  if (u->op != POLY_OP_VCONST && u->op != POLY_OP_STACK) return false;
  if (lane < u->n_src && u->src[lane] && u->src[lane]->op == POLY_OP_CONST) {
    *out = u->src[lane]->arg;
    return true;
  }
  if (u->arg.kind == POLY_ARG_INT_TUPLE && lane < u->arg.int_tuple.n) {
    *out = poly_arg_int(u->arg.int_tuple.vals[lane]);
    return true;
  }
  return false;
}

static bool numeric_negative_shift_arg(PolyArg a) {
  if (a.kind == POLY_ARG_INT) return a.i < 0;
  if (a.kind == POLY_ARG_BIGINT) return a.bigint.sign < 0;
  if (a.kind == POLY_ARG_FLOAT) return a.f < 0.0;
  return false;
}

/* Pinned UOp.const routes every folded lane through DType.const
 * (uop/ops.py:553-561, dtype.py:92-100). Keep that normalization local to
 * vector folding; non-finite floats cannot become integer CONST arguments. */
static bool normalize_vector_const_arg(PolyDType dtype, PolyArg val, PolyArg *out) {
  if (!out) return false;
  if (val.kind == POLY_ARG_INVALID) {
    *out = val;
    return true;
  }
  if (poly_dtype_is_float(dtype)) {
    if (val.kind == POLY_ARG_FLOAT)
      *out = val;
    else if (val.kind == POLY_ARG_INT)
      *out = poly_arg_float((double)val.i);
    else if (val.kind == POLY_ARG_BIGINT)
      *out = poly_arg_float(poly_arg_integer_to_double(val));
    else if (val.kind == POLY_ARG_BOOL)
      *out = poly_arg_float(val.b ? 1.0 : 0.0);
    else
      return false;
    return true;
  }
  if (poly_dtype_is_bool(dtype)) {
    bool b;
    if (val.kind == POLY_ARG_BOOL)
      b = val.b;
    else if (val.kind == POLY_ARG_INT)
      b = val.i != 0;
    else if (val.kind == POLY_ARG_BIGINT)
      b = val.bigint.n_limbs != 0;
    else if (val.kind == POLY_ARG_FLOAT)
      b = val.f != 0.0;
    else
      return false;
    *out = poly_arg_bool(b);
    return true;
  }
  if (poly_dtype_is_int(dtype)) {
    if (val.kind == POLY_ARG_BIGINT) {
      *out = val;
      return true;
    }
    int64_t i;
    if (val.kind == POLY_ARG_INT)
      i = val.i;
    else if (val.kind == POLY_ARG_BOOL)
      i = val.b ? 1 : 0;
    else if (val.kind == POLY_ARG_FLOAT) {
      if (!isfinite(val.f) || val.f < -0x1p63 || val.f >= 0x1p63) return false;
      i = (int64_t)val.f;
    } else {
      return false;
    }
    *out = poly_arg_int(i);
    return true;
  }
  *out = val;
  return true;
}

static PolyUOp *build_vector_const_fold(PolyCtx *ctx, PolyUOp *root, PolyOps op, int n_ops) {
  if (!root || root->dtype.count <= 1 || root->dtype.count > 128 || n_ops < 1 || n_ops > 3)
    return NULL;
  PolyUOp *elts[128];
  PolyDType lane_dt = poly_dtype_scalar(root->dtype);
  PolyDType exec_dt = lane_dt;
  if ((op == POLY_OP_CMPLT || op == POLY_OP_CMPNE || op == POLY_OP_CMPEQ) && root->n_src >= 1)
    exec_dt = poly_dtype_scalar(root->src[0]->dtype);
  for (int lane = 0; lane < root->dtype.count; lane++) {
    PolyArg lane_ops[3];
    for (int i = 0; i < n_ops; i++) {
      if (!const_lane_arg(root->src[i], lane, &lane_ops[i])) return NULL;
    }
    if ((op == POLY_OP_SHL || op == POLY_OP_SHR) && n_ops >= 2 &&
        numeric_negative_shift_arg(lane_ops[1]))
      return NULL;
    /* tinygrad ops.py:1192-1197 recursively defaults fixed-width vector lanes
     * to truncating execution. weakint has no truncate entry and remains
     * unbounded; refuse a lane fold that PolyArg cannot represent. */
    bool truncate_lane = !poly_dtype_is_index(exec_dt);
    PolyArg lane_result;
    PolyInt owned = {0};
    if (!exec_symbolic_const_alu(
            op, exec_dt, lane_ops, n_ops, truncate_lane, &lane_result, &owned
        )) {
      poly_int_free(&owned);
      return NULL;
    }
    if (op == POLY_OP_POW && truncate_lane && lane_result.kind == POLY_ARG_INVALID) goto lane_fail;
    PolyArg normalized;
    if (!normalize_vector_const_arg(lane_dt, lane_result, &normalized)) goto lane_fail;
    elts[lane] = poly_uop0(ctx, POLY_OP_CONST, lane_dt, normalized);
    poly_int_free(&owned);
    continue;
  lane_fail:
    poly_int_free(&owned);
    return NULL;
  }
  /* Current tinygrad represents lane-wise constants as STACK(CONST, ...).
   * Keeping child CONST lanes here also preserves Invalid lanes instead of
   * turning them into scalar zero. */
  return poly_uop(ctx, POLY_OP_STACK, root->dtype, elts, root->dtype.count, poly_arg_none());
}

/* Constant folding: Unary(CONST/STACK) -> CONST/STACK */
static PolyUOp *rule_const_fold_unary(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  PolyUOp *a = poly_bind(b, "a");
  if (a->dtype.count > 1) return build_vector_const_fold(ctx, a, a->op, 1);
  PolyArg operand = a->src[0]->arg;
  PolyArg result;
  PolyInt owned = {0};
  if (!exec_symbolic_const_alu(a->op, a->dtype, &operand, 1, false, &result, &owned)) return NULL;
  PolyUOp *ret = poly_const_like(ctx, a, result);
  poly_int_free(&owned);
  return ret;
}

/* Constant folding: Binary(CONST/STACK, CONST/STACK) -> CONST/STACK */
static PolyUOp *rule_const_fold_binary(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  PolyUOp *a = poly_bind(b, "a");
  if (a->dtype.count > 1) return build_vector_const_fold(ctx, a, a->op, 2);
  if ((a->op == POLY_OP_SHL || a->op == POLY_OP_SHR) && numeric_negative_shift_arg(a->src[1]->arg))
    return NULL;
  PolyArg operands[2] = {a->src[0]->arg, a->src[1]->arg};
  PolyDType exec_dtype = a->dtype;
  if ((a->op == POLY_OP_CMPLT || a->op == POLY_OP_CMPNE || a->op == POLY_OP_CMPEQ) && a->n_src >= 1)
    exec_dtype = poly_dtype_scalar(a->src[0]->dtype);
  PolyArg result;
  PolyInt owned = {0};
  if (!exec_symbolic_const_alu(a->op, exec_dtype, operands, 2, false, &result, &owned)) return NULL;
  PolyUOp *ret = poly_const_like(ctx, a, result);
  poly_int_free(&owned);
  return ret;
}

/* Constant folding: Ternary(CONST/STACK, ...) -> CONST/STACK */
static PolyUOp *rule_const_fold_ternary(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  PolyUOp *a = poly_bind(b, "a");
  if (a->dtype.count > 1) return build_vector_const_fold(ctx, a, a->op, 3);
  PolyArg operands[3] = {a->src[0]->arg, a->src[1]->arg, a->src[2]->arg};
  PolyArg result;
  PolyInt owned = {0};
  if (!exec_symbolic_const_alu(a->op, a->dtype, operands, 3, false, &result, &owned)) return NULL;
  PolyUOp *ret = poly_const_like(ctx, a, result);
  poly_int_free(&owned);
  return ret;
}

/* CAST(CONST) -> CONST with new dtype */
static PolyUOp *rule_cast_const(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  PolyUOp *c = root->src[0];
  return poly_const_like(ctx, root, c->arg);
}

/* Pinned tinygrad/uop/symbolic.py:19-24,150 reinterprets equal-size scalar
 * constant storage through dtype formats. */
static PolyUOp *rule_bitcast_const(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  if (!root || root->op != POLY_OP_BITCAST || root->n_src != 1 ||
      root->src[0]->op != POLY_OP_CONST)
    return NULL;
  PolyArg value;
  return poly_exec_bitcast_const(root->src[0]->dtype, root->dtype, root->src[0]->arg, &value)
             ? poly_const_like(ctx, root, value)
             : NULL;
}

static bool const_uop_integer_eq(PolyUOp *u, uint64_t value) {
  return u && u->op == POLY_OP_CONST && arg_is_exact_integer(u->arg) &&
         poly_arg_integer_to_u64_mod(u->arg) == value;
}

/* Pinned tinygrad/uop/symbolic.py:159:
 *   (x:uint64 & 0xffffffff).cast(uint32) -> x.cast(uint32). */
static PolyUOp *rule_threefry_low_lane_cast(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  if (!root || root->op != POLY_OP_CAST || root->n_src != 1 ||
      !poly_dtype_eq(root->dtype, POLY_UINT32))
    return NULL;
  PolyUOp *and = root->src[0];
  if (!and || and->op != POLY_OP_AND || and->n_src != 2 ||
      !poly_dtype_eq(and->dtype, POLY_UINT64))
    return NULL;
  PolyUOp *x = NULL;
  if (const_uop_integer_eq(and->src[1], UINT64_C(0xffffffff))) x = and->src[0];
  if (const_uop_integer_eq(and->src[0], UINT64_C(0xffffffff))) x = and->src[1];
  return x ? poly_uop1(ctx, POLY_OP_CAST, POLY_UINT32, x, poly_arg_none()) : NULL;
}

static PolyUOp *match_u64_cast_from_u32(PolyUOp *u) {
  return u && u->op == POLY_OP_CAST && u->n_src == 1 &&
                 poly_dtype_eq(u->dtype, POLY_UINT64) &&
                 poly_dtype_eq(u->src[0]->dtype, POLY_UINT32)
             ? u->src[0]
             : NULL;
}

static PolyUOp *match_u64_shift32_value(PolyUOp *u) {
  if (!u || u->n_src != 2 || !poly_dtype_eq(u->dtype, POLY_UINT64)) return NULL;
  if (u->op == POLY_OP_SHL && const_uop_integer_eq(u->src[1], 32)) return u->src[0];
  if (u->op == POLY_OP_MUL && const_uop_integer_eq(u->src[1], UINT64_C(1) << 32))
    return u->src[0];
  if (u->op == POLY_OP_MUL && const_uop_integer_eq(u->src[0], UINT64_C(1) << 32))
    return u->src[1];
  return NULL;
}

static bool match_u64_pack(PolyUOp *u, PolyUOp **high, PolyUOp **low) {
  if (!u || u->op != POLY_OP_OR || u->n_src != 2 || !poly_dtype_eq(u->dtype, POLY_UINT64))
    return false;
  for (int swap = 0; swap < 2; swap++) {
    PolyUOp *h = match_u64_shift32_value(u->src[swap]);
    PolyUOp *l = match_u64_cast_from_u32(u->src[swap ^ 1]);
    if (h && l) {
      *high = h;
      *low = l;
      return true;
    }
  }
  return false;
}

/* Pinned tinygrad/uop/symbolic.py:160,162: casting a packed uint64 to uint32
 * returns the exact low uint32 occurrence. */
static PolyUOp *rule_threefry_unpack_low(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)ctx;
  (void)b;
  if (!root || root->op != POLY_OP_CAST || root->n_src != 1 ||
      !poly_dtype_eq(root->dtype, POLY_UINT32))
    return NULL;
  PolyUOp *high = NULL, *low = NULL;
  return match_u64_pack(root->src[0], &high, &low) ? low : NULL;
}

/* Pinned tinygrad/uop/symbolic.py:161,163: dividing/shifting a packed uint64
 * by 2**32 returns the exact high occurrence. */
static PolyUOp *rule_threefry_unpack_high(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)ctx;
  (void)b;
  if (!root || root->n_src != 2 || !poly_dtype_eq(root->dtype, POLY_UINT64)) return NULL;
  bool high_extract =
      (root->op == POLY_OP_SHR && const_uop_integer_eq(root->src[1], 32)) ||
      (root->op == POLY_OP_FLOORDIV &&
       const_uop_integer_eq(root->src[1], UINT64_C(1) << 32));
  if (!high_extract) return NULL;
  PolyUOp *high = NULL, *low = NULL;
  return match_u64_pack(root->src[0], &high, &low) ? high : NULL;
}

/* CAST(x, bool) -> CMPNE(x, 0) */
static PolyUOp *rule_cast_bool(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  if (!poly_dtype_eq(root->dtype, POLY_BOOL)) return NULL;
  PolyUOp *x = root->src[0];
  if (poly_dtype_eq(x->dtype, POLY_BOOL)) return NULL; /* already bool, handled by noop */
  PolyUOp *zero = poly_const_like_int(ctx, x, 0);
  return poly_uop2(ctx, POLY_OP_CMPNE, POLY_BOOL, x, zero, poly_arg_none());
}

/* CAST/BITCAST same dtype -> identity */
static PolyUOp *rule_cast_noop(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)ctx;
  (void)b;
  if (poly_dtype_eq(root->dtype, root->src[0]->dtype)) return root->src[0];
  return NULL;
}

/* Pinned tinygrad symbolic.py:151-152:
 * b.cast(a).cast(b) -> b when a preserves every value in b. */
static PolyUOp *rule_cast_roundtrip_lossless(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)ctx;
  (void)b;
  PolyUOp *inner = root->src[0];
  if (!inner || inner->op != POLY_OP_CAST || inner->n_src != 1) return NULL;
  PolyUOp *x = inner->src[0];
  if (!x || !poly_dtype_eq(x->dtype, root->dtype)) return NULL;
  return poly_dtype_can_lossless_cast(root->dtype, inner->dtype) ? x : NULL;
}

typedef struct {
  PolyInt lo;
  PolyInt hi;
  bool valid;
} ExactIntRange;

static void exact_int_range_free(ExactIntRange *range) {
  if (!range) return;
  poly_int_free(&range->lo);
  poly_int_free(&range->hi);
  range->valid = false;
}

static bool exact_int_range_take(ExactIntRange *out, PolyInt *lo, PolyInt *hi) {
  if (!out || !lo || !hi) return false;
  out->lo = *lo;
  out->hi = *hi;
  out->valid = true;
  poly_int_init(lo);
  poly_int_init(hi);
  return true;
}

static bool exact_int_range_copy(ExactIntRange *out, const PolyInt *lo, const PolyInt *hi) {
  if (!out || !lo || !hi || !poly_int_copy(&out->lo, lo)) return false;
  if (!poly_int_copy(&out->hi, hi)) {
    poly_int_free(&out->lo);
    return false;
  }
  out->valid = true;
  return true;
}

static bool exact_int_range_i64(ExactIntRange *out, int64_t lo, int64_t hi) {
  if (!out || !poly_int_from_i64(&out->lo, lo)) return false;
  if (!poly_int_from_i64(&out->hi, hi)) {
    poly_int_free(&out->lo);
    return false;
  }
  out->valid = true;
  return true;
}

/* Pinned tinygrad dtype.py:84-91. Integer endpoints are Python integers,
 * including uint64 [0, 2**64-1] and weakint [-2**799, 2**799-1]. */
static bool exact_int_range_dtype(ExactIntRange *out, PolyDType dtype) {
  PolyDType scalar = poly_dtype_scalar(dtype);
  if (scalar.is_ptr || (!poly_dtype_is_int(scalar) && !poly_dtype_is_bool(scalar))) return false;
  if (poly_dtype_is_bool(scalar)) return exact_int_range_i64(out, 0, 1);

  int bits = scalar.bitsize;
  if (bits <= 0) return false;
  PolyInt one = {0}, power = {0}, lo = {0}, hi = {0};
  bool ok = false;
  if (!poly_int_from_i64(&one, 1)) goto done;
  if (poly_dtype_is_unsigned(scalar)) {
    if (!poly_int_from_i64(&lo, 0) || !poly_int_shl(&power, &one, (uint64_t)bits) ||
        !poly_int_sub(&hi, &power, &one))
      goto done;
  } else {
    if (!poly_int_shl(&power, &one, (uint64_t)(bits - 1)) || !poly_int_neg(&lo, &power) ||
        !poly_int_sub(&hi, &power, &one))
      goto done;
  }
  ok = exact_int_range_take(out, &lo, &hi);
done:
  poly_int_free(&one);
  poly_int_free(&power);
  poly_int_free(&lo);
  poly_int_free(&hi);
  return ok;
}

static bool exact_int_is_i64(const PolyInt *value, int64_t expected) {
  int64_t actual = 0;
  return poly_int_to_i64(value, &actual) && actual == expected;
}

static bool exact_int_to_nonnegative_u64(const PolyInt *value, uint64_t *out) {
  if (!value || !out || poly_int_is_negative(value) || value->n_limbs > 2) return false;
  *out = poly_int_to_u64_mod(value);
  return true;
}

static bool exact_int_add_i64(PolyInt *out, const PolyInt *value, int64_t addend) {
  PolyInt c = {0};
  if (!poly_int_from_i64(&c, addend)) return false;
  bool ok = poly_int_add(out, value, &c);
  poly_int_free(&c);
  return ok;
}

static bool exact_int_min_copy(PolyInt *out, const PolyInt *a, const PolyInt *b) {
  return poly_int_copy(out, poly_int_cmp(a, b) <= 0 ? a : b);
}

static bool exact_int_max_copy(PolyInt *out, const PolyInt *a, const PolyInt *b) {
  return poly_int_copy(out, poly_int_cmp(a, b) >= 0 ? a : b);
}

static ExactIntRange *exact_int_range_get(PolyMap *memo, PolyUOp *u) {
  return u ? poly_map_get(memo, poly_ptr_hash(u), u, poly_ptr_eq) : NULL;
}

static bool exact_int_div(
    PolyInt *out,
    const PolyInt *a,
    const PolyInt *b,
    bool floor_mode,
    bool remainder
) {
  PolyInt quotient = {0}, rem = {0};
  if (!poly_int_divmod(&quotient, &rem, a, b, floor_mode)) return false;
  *out = remainder ? rem : quotient;
  if (remainder)
    poly_int_free(&quotient);
  else
    poly_int_free(&rem);
  return true;
}

static bool exact_int_range_div_corners(
    ExactIntRange *out,
    const ExactIntRange *a,
    const ExactIntRange *b,
    bool floor_mode
) {
  PolyInt values[4] = {{0}}, lo = {0}, hi = {0};
  const PolyInt *lhs[4] = {&a->lo, &a->lo, &a->hi, &a->hi};
  const PolyInt *rhs[4] = {&b->lo, &b->hi, &b->lo, &b->hi};
  bool ok = false;
  for (int i = 0; i < 4; i++)
    if (!exact_int_div(&values[i], lhs[i], rhs[i], floor_mode, false)) goto done;
  if (!exact_int_min_copy(&lo, &values[0], &values[1]) ||
      !exact_int_min_copy(&hi, &values[2], &values[3]))
    goto done;
  for (int i = 0; i < 4; i++) {
    PolyInt next_lo = {0}, next_hi = {0};
    if (!exact_int_min_copy(&next_lo, &lo, &values[i]) ||
        !exact_int_max_copy(&next_hi, &hi, &values[i])) {
      poly_int_free(&next_lo);
      poly_int_free(&next_hi);
      goto done;
    }
    poly_int_free(&lo);
    poly_int_free(&hi);
    lo = next_lo;
    hi = next_hi;
  }
  ok = exact_int_range_take(out, &lo, &hi);
done:
  for (int i = 0; i < 4; i++)
    poly_int_free(&values[i]);
  poly_int_free(&lo);
  poly_int_free(&hi);
  return ok;
}

static bool exact_int_range_node(PolyUOp *u, PolyMap *memo, ExactIntRange *out) {
  if (!u || !out) return false;

  if (u->op == POLY_OP_CONST && (u->arg.kind == POLY_ARG_INT || u->arg.kind == POLY_ARG_BIGINT ||
                                 u->arg.kind == POLY_ARG_BOOL)) {
    if (!poly_int_from_arg(&out->lo, u->arg) || !poly_int_copy(&out->hi, &out->lo)) {
      exact_int_range_free(out);
      return false;
    }
    out->valid = true;
    return true;
  }
  if (u->op == POLY_OP_DEFINE_VAR && u->arg.kind == POLY_ARG_DEFINE_VAR)
    return exact_int_range_i64(out, u->arg.define_var.min_val, u->arg.define_var.max_val);
  if (u->op == POLY_OP_PARAM && u->arg.kind == POLY_ARG_PARAM && u->arg.param &&
      u->arg.param->has_minmax)
    return exact_int_range_i64(out, u->arg.param->min_val, u->arg.param->max_val);

  if ((u->op == POLY_OP_BIND || u->op == POLY_OP_GEP) && u->n_src >= 1) {
    ExactIntRange *src = exact_int_range_get(memo, u->src[0]);
    if (src && src->valid) return exact_int_range_copy(out, &src->lo, &src->hi);
  }

  if ((u->op == POLY_OP_RANGE || u->op == POLY_OP_SPECIAL) && u->n_src >= 1) {
    ExactIntRange *end = exact_int_range_get(memo, u->src[0]);
    PolyInt lo = {0}, hi = {0};
    bool ok = false;
    if (end && end->valid && poly_int_from_i64(&lo, 0) && exact_int_add_i64(&hi, &end->hi, -1))
      ok = exact_int_range_take(out, &lo, &hi);
    poly_int_free(&lo);
    poly_int_free(&hi);
    if (ok) return true;
  }

  if ((u->op == POLY_OP_UNROLL || u->op == POLY_OP_VECTORIZE || u->op == POLY_OP_VCONST) &&
      u->n_src > 0) {
    ExactIntRange *first = exact_int_range_get(memo, u->src[0]);
    if (first && first->valid && exact_int_range_copy(out, &first->lo, &first->hi)) {
      for (int i = 1; i < u->n_src; i++) {
        ExactIntRange *src = exact_int_range_get(memo, u->src[i]);
        PolyInt next_lo = {0}, next_hi = {0};
        if (!src || !src->valid || !exact_int_min_copy(&next_lo, &out->lo, &src->lo) ||
            !exact_int_max_copy(&next_hi, &out->hi, &src->hi)) {
          poly_int_free(&next_lo);
          poly_int_free(&next_hi);
          exact_int_range_free(out);
          break;
        }
        poly_int_free(&out->lo);
        poly_int_free(&out->hi);
        out->lo = next_lo;
        out->hi = next_hi;
      }
      if (out->valid) return true;
    }
  }

  if (u->op == POLY_OP_WHERE && u->n_src == 3 && poly_dtype_is_int(u->dtype)) {
    ExactIntRange *t = exact_int_range_get(memo, u->src[1]);
    ExactIntRange *f = exact_int_range_get(memo, u->src[2]);
    if (t && t->valid && f && f->valid && exact_int_min_copy(&out->lo, &t->lo, &f->lo) &&
        exact_int_max_copy(&out->hi, &t->hi, &f->hi)) {
      out->valid = true;
      return true;
    }
    exact_int_range_free(out);
  }

  if (u->op == POLY_OP_CAST && u->n_src >= 1) {
    PolyDType scalar = poly_dtype_scalar(u->dtype);
    /* Pinned tinygrad/uop/ops.py:1017-1018 matches exact scalar signed/weak
     * dtypes here. Vector CASTs fall back to their full dtype bounds. */
    bool monotone =
        u->dtype.count == 1 && (poly_dtype_is_index(scalar) ||
                                (poly_dtype_is_int(scalar) && !poly_dtype_is_unsigned(scalar) &&
                                 !poly_dtype_is_bool(scalar)));
    ExactIntRange *src = exact_int_range_get(memo, u->src[0]);
    ExactIntRange dtype_range = {0};
    if (monotone && src && src->valid && exact_int_range_dtype(&dtype_range, scalar) &&
        exact_int_max_copy(&out->lo, &dtype_range.lo, &src->lo) &&
        exact_int_min_copy(&out->hi, &src->hi, &dtype_range.hi)) {
      out->valid = true;
      exact_int_range_free(&dtype_range);
      return true;
    }
    exact_int_range_free(out);
    exact_int_range_free(&dtype_range);
  }

  if (u->n_src == 2 && !poly_dtype_is_float(u->dtype)) {
    ExactIntRange *a = exact_int_range_get(memo, u->src[0]);
    ExactIntRange *b = exact_int_range_get(memo, u->src[1]);
    if (a && a->valid && b && b->valid) {
      PolyInt lo = {0}, hi = {0};
      bool ok = false;
      /* Pinned tinygrad/uop/ops.py:989-999 maps only floor div/mod with an
       * empty numerator interval to the exact point [0,0]. */
      if ((u->op == POLY_OP_FLOORDIV || u->op == POLY_OP_FLOORMOD) &&
          poly_int_cmp(&a->lo, &a->hi) > 0)
        ok = poly_int_from_i64(&lo, 0) && poly_int_from_i64(&hi, 0);
      else if (u->op == POLY_OP_ADD)
        ok = poly_int_add(&lo, &a->lo, &b->lo) && poly_int_add(&hi, &a->hi, &b->hi);
      else if (u->op == POLY_OP_SUB)
        ok = poly_int_sub(&lo, &a->lo, &b->hi) && poly_int_sub(&hi, &a->hi, &b->lo);
      else if (u->op == POLY_OP_MUL) {
        PolyInt values[4] = {{0}};
        const PolyInt *lhs[4] = {&a->lo, &a->lo, &a->hi, &a->hi};
        const PolyInt *rhs[4] = {&b->lo, &b->hi, &b->lo, &b->hi};
        ok = true;
        for (int i = 0; ok && i < 4; i++)
          ok = poly_int_mul(&values[i], lhs[i], rhs[i]);
        if (ok) {
          ok = poly_int_copy(&lo, &values[0]) && poly_int_copy(&hi, &values[0]);
          for (int i = 1; ok && i < 4; i++) {
            PolyInt next_lo = {0}, next_hi = {0};
            ok = exact_int_min_copy(&next_lo, &lo, &values[i]) &&
                 exact_int_max_copy(&next_hi, &hi, &values[i]);
            if (ok) {
              poly_int_free(&lo);
              poly_int_free(&hi);
              lo = next_lo;
              hi = next_hi;
            } else {
              poly_int_free(&next_lo);
              poly_int_free(&next_hi);
            }
          }
        }
        for (int i = 0; i < 4; i++)
          poly_int_free(&values[i]);
      } else if ((u->op == POLY_OP_SHL || u->op == POLY_OP_SHR) && poly_int_cmp(&b->lo, &b->hi) == 0) {
        uint64_t shift = 0;
        if (exact_int_to_nonnegative_u64(&b->lo, &shift))
          ok = (u->op == POLY_OP_SHL ? poly_int_shl(&lo, &a->lo, shift)
                                     : poly_int_shr(&lo, &a->lo, shift)) &&
               (u->op == POLY_OP_SHL ? poly_int_shl(&hi, &a->hi, shift)
                                     : poly_int_shr(&hi, &a->hi, shift));
      } else if (u->op == POLY_OP_MAX)
        ok = exact_int_max_copy(&lo, &a->lo, &b->lo) && exact_int_max_copy(&hi, &a->hi, &b->hi);
      else if (u->op == POLY_OP_AND && poly_dtype_is_int(u->dtype) && poly_int_cmp(&b->lo, &b->hi) == 0 && !poly_int_is_negative(&b->lo)) {
        ok = poly_int_from_i64(&lo, 0) &&
             poly_int_copy(
                 &hi, poly_int_is_negative(&a->lo)
                          ? &b->hi
                          : (poly_int_cmp(&a->hi, &b->hi) <= 0 ? &a->hi : &b->hi)
             );
      } else if (u->op == POLY_OP_XOR && poly_int_cmp(&b->lo, &b->hi) == 0 && exact_int_is_i64(&b->lo, -1)) {
        ok = poly_int_bitwise(&lo, POLY_OP_XOR, &a->hi, &b->lo) &&
             poly_int_bitwise(&hi, POLY_OP_XOR, &a->lo, &b->lo);
      } else if (u->op == POLY_OP_CMPLT) {
        ok = poly_int_from_i64(&lo, poly_int_cmp(&a->hi, &b->lo) < 0 ? 1 : 0) &&
             poly_int_from_i64(&hi, poly_int_cmp(&a->lo, &b->hi) < 0 ? 1 : 0);
      } else if (u->op == POLY_OP_CMPNE) {
        bool definitely_ne = poly_int_cmp(&a->hi, &b->lo) < 0 || poly_int_cmp(&b->hi, &a->lo) < 0;
        bool all_equal = poly_int_cmp(&a->lo, &a->hi) == 0 && poly_int_cmp(&b->lo, &b->hi) == 0 &&
                         poly_int_cmp(&a->lo, &b->lo) == 0;
        ok = poly_int_from_i64(&lo, definitely_ne ? 1 : 0) &&
             poly_int_from_i64(&hi, all_equal ? 0 : 1);
      } else if ((u->op == POLY_OP_AND || u->op == POLY_OP_OR) && poly_dtype_eq(u->dtype, POLY_BOOL)) {
        bool is_and = u->op == POLY_OP_AND;
        ok = poly_int_from_i64(
                 &lo, is_and ? (!poly_int_is_zero(&a->lo) && !poly_int_is_zero(&b->lo))
                             : (!poly_int_is_zero(&a->lo) || !poly_int_is_zero(&b->lo))
             ) &&
             poly_int_from_i64(
                 &hi, is_and ? (!poly_int_is_zero(&a->hi) && !poly_int_is_zero(&b->hi))
                             : (!poly_int_is_zero(&a->hi) || !poly_int_is_zero(&b->hi))
             );
      } else if ((u->op == POLY_OP_CDIV || u->op == POLY_OP_FLOORDIV) && ((a->valid && b->lo.sign > 0 && b->hi.sign > 0) || (a->valid && b->lo.sign < 0 && b->hi.sign < 0))) {
        exact_int_range_free(out);
        if (exact_int_range_div_corners(out, a, b, u->op == POLY_OP_FLOORDIV)) return true;
      } else if (u->op == POLY_OP_CMOD) {
        PolyInt zero = {0}, one = {0};
        if (poly_int_from_i64(&zero, 0) && poly_int_from_i64(&one, 1)) {
          if (poly_int_cmp(&b->lo, &b->hi) == 0 && b->lo.sign > 0) {
            PolyInt neg_c = {0}, c_minus_one = {0}, neg_c_minus_one = {0};
            if (poly_int_neg(&neg_c, &b->lo) && poly_int_sub(&c_minus_one, &b->hi, &one) &&
                poly_int_neg(&neg_c_minus_one, &c_minus_one)) {
              const PolyInt *lo_src = a->lo.sign > 0 ? &zero
                                      : (a->lo.sign <= 0 && poly_int_cmp(&a->lo, &neg_c) > 0)
                                          ? &a->lo
                                          : &neg_c_minus_one;
              const PolyInt *hi_src = a->hi.sign < 0 ? &zero
                                      : (a->hi.sign >= 0 && poly_int_cmp(&a->hi, &b->lo) < 0)
                                          ? &a->hi
                                          : &c_minus_one;
              ok = poly_int_copy(&lo, lo_src) && poly_int_copy(&hi, hi_src);
            }
            poly_int_free(&neg_c);
            poly_int_free(&c_minus_one);
            poly_int_free(&neg_c_minus_one);
          } else if (b->lo.sign > 0) {
            PolyInt magnitude = {0}, negative = {0};
            if (poly_int_sub(&magnitude, &b->hi, &one) && poly_int_neg(&negative, &magnitude)) {
              const PolyInt *lo_src = a->lo.sign >= 0 ? &zero : &negative;
              const PolyInt *hi_src = a->hi.sign <= 0 ? &zero : &magnitude;
              ok = poly_int_copy(&lo, lo_src) && poly_int_copy(&hi, hi_src);
            }
            poly_int_free(&magnitude);
            poly_int_free(&negative);
          } else if (b->hi.sign < 0) {
            PolyInt neg_lo = {0}, magnitude = {0}, negative = {0};
            if (poly_int_neg(&neg_lo, &b->lo) && poly_int_sub(&magnitude, &neg_lo, &one) &&
                poly_int_neg(&negative, &magnitude)) {
              const PolyInt *lo_src = a->lo.sign >= 0 ? &zero : &negative;
              const PolyInt *hi_src = a->hi.sign <= 0 ? &zero : &magnitude;
              ok = poly_int_copy(&lo, lo_src) && poly_int_copy(&hi, hi_src);
            }
            poly_int_free(&neg_lo);
            poly_int_free(&magnitude);
            poly_int_free(&negative);
          }
        }
        poly_int_free(&zero);
        poly_int_free(&one);
      } else if (u->op == POLY_OP_FLOORMOD) {
        PolyInt zero = {0}, one = {0};
        if (poly_int_from_i64(&zero, 0) && poly_int_from_i64(&one, 1)) {
          if (poly_int_cmp(&b->lo, &b->hi) == 0 && b->lo.sign != 0) {
            PolyInt qlo = {0}, qhi = {0};
            if (exact_int_div(&qlo, &a->lo, &b->lo, true, false) &&
                exact_int_div(&qhi, &a->hi, &b->lo, true, false) && poly_int_cmp(&qlo, &qhi) == 0)
              ok = exact_int_div(&lo, &a->lo, &b->lo, true, true) &&
                   exact_int_div(&hi, &a->hi, &b->lo, true, true);
            else if (b->lo.sign > 0)
              ok = poly_int_from_i64(&lo, 0) && poly_int_sub(&hi, &b->lo, &one);
            else
              ok = poly_int_add(&lo, &b->lo, &one) && poly_int_from_i64(&hi, 0);
            poly_int_free(&qlo);
            poly_int_free(&qhi);
          } else if (b->lo.sign > 0)
            ok = poly_int_from_i64(&lo, 0) && poly_int_sub(&hi, &b->hi, &one);
          else if (b->hi.sign < 0)
            ok = poly_int_add(&lo, &b->lo, &one) && poly_int_from_i64(&hi, 0);
        }
        poly_int_free(&zero);
        poly_int_free(&one);
      }
      if (ok) {
        bool taken = exact_int_range_take(out, &lo, &hi);
        poly_int_free(&lo);
        poly_int_free(&hi);
        return taken;
      }
      poly_int_free(&lo);
      poly_int_free(&hi);
    }
  }

  PolyDType scalar = poly_dtype_scalar(u->dtype);
  if (scalar.is_ptr || (!poly_dtype_is_int(scalar) && !poly_dtype_is_bool(scalar))) return true;
  return exact_int_range_dtype(out, scalar);
}

/* Pinned tinygrad UOp._min_max uses arbitrary-precision Python integers.
 * Evaluate the exact integer interval only for the nested-CAST predicate;
 * this is pass-local evidence, not persistent UOp state or a second graph. */
static bool exact_int_range(PolyCtx *ctx, PolyUOp *root, ExactIntRange *out) {
  if (!ctx || !root || !out) return false;
  int n = 0;
  PolyUOp **topo = poly_toposort_alloc(ctx, root, &n);
  ExactIntRange *ranges = n > 0 ? calloc((size_t)n, sizeof(*ranges)) : NULL;
  PolyMap *memo = poly_map_new((size_t)n * 2 + 16);
  bool ok = topo && ranges && memo;
  for (int i = 0; ok && i < n; i++) {
    ok = exact_int_range_node(topo[i], memo, &ranges[i]);
    if (ok) poly_map_set(memo, poly_ptr_hash(topo[i]), topo[i], &ranges[i], poly_ptr_eq);
  }
  ExactIntRange *found = ok ? exact_int_range_get(memo, root) : NULL;
  if (!found || !found->valid || !exact_int_range_copy(out, &found->lo, &found->hi)) ok = false;
  for (int i = 0; i < n; i++)
    exact_int_range_free(&ranges[i]);
  poly_map_destroy(memo);
  free(ranges);
  free(topo);
  return ok;
}

static bool exact_int_range_fits_dtype(PolyCtx *ctx, PolyUOp *u, PolyDType dtype) {
  ExactIntRange value = {0}, limits = {0};
  bool fits = exact_int_range(ctx, u, &value) && exact_int_range_dtype(&limits, dtype) &&
              poly_int_cmp(&limits.lo, &value.lo) <= 0 &&
              poly_int_cmp(&value.hi, &limits.hi) <= 0;
  exact_int_range_free(&value);
  exact_int_range_free(&limits);
  return fits;
}

/* Pinned tinygrad/uop/symbolic.py:297-299 performs signed-long binary
 * arithmetic in int32 when the result and both operands are proven to fit,
 * then casts back to the original result dtype. This proof is what lets
 * uop_given_valid narrow gated gather addresses without changing genuinely
 * wide, ungated LOAD<int> + constant expressions. */
static PolyUOp *rule_narrow_proven_long_binary(
    PolyCtx *ctx,
    PolyUOp *root,
    const PolyBindings *b
) {
  (void)b;
  if (!root || root->n_src != 2 || !poly_opset_has(POLY_GROUP_BINARY, root->op)) return NULL;
  PolyUOp *x = root->src[0], *y = root->src[1];
  if (!x || !y || !poly_dtype_eq(poly_dtype_scalar(x->dtype), POLY_INT64) ||
      !poly_dtype_eq(poly_dtype_scalar(y->dtype), POLY_INT64))
    return NULL;
  if (!exact_int_range_fits_dtype(ctx, root, POLY_INT32) ||
      !exact_int_range_fits_dtype(ctx, x, POLY_INT32) ||
      !exact_int_range_fits_dtype(ctx, y, POLY_INT32))
    return NULL;

  PolyDType x_int = x->dtype.count > 1 ? poly_dtype_vec(POLY_INT32, x->dtype.count) : POLY_INT32;
  PolyDType y_int = y->dtype.count > 1 ? poly_dtype_vec(POLY_INT32, y->dtype.count) : POLY_INT32;
  PolyDType result_int = poly_opset_has(POLY_GROUP_COMPARISON, root->op)
                             ? root->dtype
                             : (root->dtype.count > 1
                                    ? poly_dtype_vec(POLY_INT32, root->dtype.count)
                                    : POLY_INT32);
  PolyUOp *x_cast = poly_uop1(ctx, POLY_OP_CAST, x_int, x, poly_arg_none());
  PolyUOp *y_cast = poly_uop1(ctx, POLY_OP_CAST, y_int, y, poly_arg_none());
  PolyUOp *narrow = poly_uop2(ctx, root->op, result_int, x_cast, y_cast, root->arg);
  return poly_dtype_eq(narrow->dtype, root->dtype)
             ? narrow
             : poly_uop1(ctx, POLY_OP_CAST, root->dtype, narrow, poly_arg_none());
}

/* Pinned tinygrad/uop/symbolic.py:297-301:
 *   x.cast(a).cast(b) -> x.cast(b)
 * when a preserves every x value, either by dtype or by the proved integer
 * bounds of this value. Unlike the earlier roundtrip rule, b may differ from
 * x.dtype. UOp.cast returns x when b already equals x.dtype. */
static PolyUOp *rule_compose_nested_casts(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  if (!root || root->op != POLY_OP_CAST || root->n_src != 1) return NULL;
  PolyUOp *inner = root->src[0];
  if (!inner || inner->op != POLY_OP_CAST || inner->n_src != 1) return NULL;
  PolyUOp *x = inner->src[0];
  if (!x) return NULL;

  bool intermediate_preserves_x = poly_dtype_can_lossless_cast(x->dtype, inner->dtype);
  if (!intermediate_preserves_x && !x->dtype.is_ptr && !inner->dtype.is_ptr &&
      poly_dtype_is_int(x->dtype) && poly_dtype_is_int(inner->dtype)) {
    ExactIntRange x_bounds = {0}, intermediate_bounds = {0};
    if (exact_int_range(ctx, x, &x_bounds) &&
        exact_int_range_dtype(&intermediate_bounds, inner->dtype))
      intermediate_preserves_x = poly_int_cmp(&intermediate_bounds.lo, &x_bounds.lo) <= 0 &&
                                 poly_int_cmp(&x_bounds.hi, &intermediate_bounds.hi) <= 0;
    exact_int_range_free(&x_bounds);
    exact_int_range_free(&intermediate_bounds);
  }
  if (!intermediate_preserves_x) return NULL;
  if (poly_dtype_eq(x->dtype, root->dtype)) return x;
  return poly_uop1(ctx, POLY_OP_CAST, root->dtype, x, poly_arg_none());
}

static bool after_dependency_is_effect(PolyOps op) {
  /* Pinned tinygrad/uop/symbolic.py:306-309 exact allowlist. */
  return op == POLY_OP_RANGE || op == POLY_OP_STORE || op == POLY_OP_CALL ||
         op == POLY_OP_FUNCTION || op == POLY_OP_BARRIER || op == POLY_OP_END ||
         op == POLY_OP_UNROLL || op == POLY_OP_LINEAR || op == POLY_OP_STAGE;
}

/* Pinned tinygrad/uop/symbolic.py:306-311. Preserve the AFTER value source,
 * retain effect dependencies, flatten every other dependency by exactly one
 * source level, and deduplicate that flattened dependency list in order. */
static PolyUOp *rule_after_canonicalize(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  if (!root || root->op != POLY_OP_AFTER || root->n_src < 1) return NULL;
  if (root->n_src == 1) return root->src[0];

  size_t cap = 1;
  for (int i = 1; i < root->n_src; i++) {
    PolyUOp *dependency = root->src[i];
    cap += after_dependency_is_effect(dependency->op) ? 1u : dependency->n_src;
    if (cap > UINT16_MAX) return NULL;
  }

  PolyUOp *inline_srcs[64];
  PolyUOp **srcs = cap <= 64 ? inline_srcs : malloc(cap * sizeof(*srcs));
  if (!srcs) return NULL;
  int n_src = 1;
  srcs[0] = root->src[0];
  for (int i = 1; i < root->n_src; i++) {
    PolyUOp *dependency = root->src[i];
    PolyUOp **candidates =
        after_dependency_is_effect(dependency->op) ? &root->src[i] : dependency->src;
    int n_candidates = after_dependency_is_effect(dependency->op) ? 1 : dependency->n_src;
    for (int j = 0; j < n_candidates; j++) {
      bool duplicate = false;
      for (int k = 1; k < n_src; k++) {
        if (srcs[k] == candidates[j]) {
          duplicate = true;
          break;
        }
      }
      if (!duplicate) srcs[n_src++] = candidates[j];
    }
  }

  bool changed = n_src != root->n_src;
  for (int i = 0; !changed && i < n_src; i++)
    changed = srcs[i] != root->src[i];
  PolyUOp *result = NULL;
  if (n_src == 1) {
    result = srcs[0];
  } else if (changed) {
    result = poly_uop_tagged_arg(
        ctx, root->op, root->dtype, srcs, n_src, root->arg, root->tag, root->tag_arg
    );
  }
  if (srcs != inline_srcs) free(srcs);
  return result;
}

/* NEG(NEG(x)) -> x */
static PolyUOp *rule_double_neg(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)ctx;
  (void)root;
  return poly_bind(b, "x");
}

/* x / x -> 1 (float division) */
static PolyUOp *rule_fdiv_self(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  return poly_const_like_float(ctx, poly_bind(b, "x"), 1.0);
}

/* Pinned ElementwiseMixin.div constructs true division as
 * a * b.reciprocal() (mixin/elementwise.py:219-241), so symbolic.py:136-145
 * x/x -> 1 matches MUL(x, RECIPROCAL(x)), not only raw FDIV(x,x). */
static PolyUOp *rule_reciprocal_self_product(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)root;
  return poly_const_like_float(ctx, poly_bind(b, "x"), 1.0);
}

static bool pow_const_double(PolyUOp *c, double *out) {
  if (!c || c->op != POLY_OP_CONST || !out) return false;
  switch (c->arg.kind) {
  case POLY_ARG_INT:
    *out = (double)c->arg.i;
    return true;
  case POLY_ARG_BIGINT:
    *out = poly_arg_integer_to_double(c->arg);
    return true;
  case POLY_ARG_FLOAT:
    *out = c->arg.f;
    return true;
  case POLY_ARG_BOOL:
    *out = c->arg.b ? 1.0 : 0.0;
    return true;
  default:
    return false;
  }
}

static bool const_is_numeric_one(PolyUOp *u) {
  double value = 0.0;
  return pow_const_double(u, &value) && value == 1.0;
}

static PolyUOp *reciprocal_one_minus(PolyCtx *ctx, PolyUOp *one, PolyUOp *d, PolyDType dtype) {
  PolyUOp *neg_one = poly_const_like_float(ctx, d, -1.0);
  PolyUOp *neg_d = neg_one ? poly_uop2(ctx, POLY_OP_MUL, dtype, d, neg_one, poly_arg_none()) : NULL;
  return neg_d ? poly_uop2(ctx, POLY_OP_ADD, dtype, one, neg_d, poly_arg_none()) : NULL;
}

static PolyUOp *rule_reciprocal_product_one_minus(
    PolyCtx *ctx,
    PolyUOp *root,
    const PolyBindings *b
) {
  PolyUOp *one = poly_bind(b, "one");
  PolyUOp *d = poly_bind(b, "d");
  if (!root || !one || !d || !const_is_numeric_one(one)) return NULL;
  return reciprocal_one_minus(ctx, one, d, root->dtype);
}

static PolyUOp *rule_reciprocal_product_scaled_one_minus(
    PolyCtx *ctx,
    PolyUOp *root,
    const PolyBindings *b
) {
  PolyUOp *one = poly_bind(b, "one");
  PolyUOp *d = poly_bind(b, "d");
  PolyUOp *y = poly_bind(b, "y");
  if (!root || !one || !d || !y || !const_is_numeric_one(one)) return NULL;
  PolyUOp *one_minus_d = reciprocal_one_minus(ctx, one, d, root->dtype);
  return one_minus_d ? poly_uop2(ctx, POLY_OP_MUL, root->dtype, y, one_minus_d, poly_arg_none())
                     : NULL;
}

static PolyUOp *rule_reciprocal_product_sum(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  PolyUOp *one = poly_bind(b, "one");
  PolyUOp *d = poly_bind(b, "d");
  PolyUOp *x = poly_bind(b, "x");
  PolyUOp *y = poly_bind(b, "y");
  if (!root || !one || !d || !x || !y || !const_is_numeric_one(one)) return NULL;
  PolyUOp *one_minus_d = reciprocal_one_minus(ctx, one, d, root->dtype);
  PolyUOp *xy = poly_uop2(ctx, POLY_OP_MUL, root->dtype, x, y, poly_arg_none());
  return one_minus_d && xy
             ? poly_uop2(ctx, POLY_OP_ADD, root->dtype, one_minus_d, xy, poly_arg_none())
             : NULL;
}

static bool pow_const_integer(double v, int64_t *out) {
  if (!isfinite(v) || v < (double)INT64_MIN || v > (double)INT64_MAX) return false;
  double ip = 0.0;
  if (modf(v, &ip) != 0.0) return false;
  if (out) *out = (int64_t)ip;
  return true;
}

static PolyUOp *pow_const_like_exp(PolyCtx *ctx, PolyUOp *exp_ref, double value) {
  if (exp_ref && !poly_dtype_is_float(exp_ref->dtype) && pow_const_integer(value, NULL))
    return poly_const_like_int(ctx, exp_ref, (int64_t)value);
  return poly_const_like_float(ctx, exp_ref, value);
}

/* tinygrad symbolic.py:simplify_pow:
 *   x**c, c const:
 *     c < 0      -> (1/x)**(-c)
 *     c == 0     -> 1
 *     c == n+.5  -> x**n * sqrt(x)
 *     c integer  -> square-and-multiply recursion
 */
static PolyUOp *rule_simplify_pow_const_exp(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)root;
  PolyUOp *x = poly_bind(b, "x");
  PolyUOp *c = poly_bind(b, "c");
  if (!x || !c) return NULL;
  double cv = 0.0;
  if (!pow_const_double(c, &cv) || !isfinite(cv)) return NULL;

  if (cv < 0.0) {
    if (!poly_dtype_is_float(x->dtype)) return NULL;
    PolyUOp *rec = poly_uop1(ctx, POLY_OP_RECIPROCAL, x->dtype, x, poly_arg_none());
    PolyUOp *pos = pow_const_like_exp(ctx, c, -cv);
    return poly_uop2(ctx, POLY_OP_POW, x->dtype, rec, pos, poly_arg_none());
  }
  if (cv == 0.0) return poly_const_like_int(ctx, x, 1);

  if (poly_dtype_is_float(x->dtype)) {
    double half_base = floor(cv - 0.5);
    if (half_base + 0.5 == cv) {
      PolyUOp *exp = pow_const_like_exp(ctx, c, cv - 0.5);
      PolyUOp *pow_part = poly_uop2(ctx, POLY_OP_POW, x->dtype, x, exp, poly_arg_none());
      PolyUOp *sqrt_part = poly_uop1(ctx, POLY_OP_SQRT, x->dtype, x, poly_arg_none());
      return poly_uop2(ctx, POLY_OP_MUL, x->dtype, pow_part, sqrt_part, poly_arg_none());
    }
  }

  int64_t ci = 0;
  if (!pow_const_integer(cv, &ci) || ci <= 0) return NULL;
  if (ci > 256) return NULL; /* avoid pathological rewrite expansion */
  PolyUOp *half = pow_const_like_exp(ctx, c, (double)(ci / 2));
  PolyUOp *y = poly_uop2(ctx, POLY_OP_POW, x->dtype, x, half, poly_arg_none());
  PolyUOp *yy = poly_uop2(ctx, POLY_OP_MUL, x->dtype, y, y, poly_arg_none());
  if (ci & 1) return poly_uop2(ctx, POLY_OP_MUL, x->dtype, yy, x, poly_arg_none());
  return yy;
}

/* tinygrad symbolic.py:
 *   c**x -> c                         if c == 1
 *   c**x -> exp2(x * log2(c))          if c > 0
 */
static PolyUOp *rule_simplify_pow_const_base(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  PolyUOp *c = poly_bind(b, "c");
  PolyUOp *x = poly_bind(b, "x");
  if (!c || !x) return NULL;
  double cv = 0.0;
  if (!pow_const_double(c, &cv) || !isfinite(cv)) return NULL;
  if (cv == 1.0) return c;
  if (cv <= 0.0 || !poly_dtype_is_float(root->dtype)) return NULL;
  PolyUOp *scale = poly_const_like_float(ctx, x, log2(cv));
  PolyUOp *arg = poly_uop2(ctx, POLY_OP_MUL, x->dtype, x, scale, poly_arg_none());
  return poly_uop1(ctx, POLY_OP_EXP2, root->dtype, arg, poly_arg_none());
}

/* tinygrad decompositions.py:xpow, invoked from symbolic.py:sym for remaining POW. */
static PolyUOp *rule_decomp_pow_generic(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  if (!root || root->op != POLY_OP_POW || root->n_src < 2) return NULL;
  if (!poly_dtype_is_float(root->dtype)) return NULL;
  PolyUOp *base = root->src[0];
  PolyUOp *exponent = root->src[1];
  PolyDType ft = root->dtype;
  PolyDType bt = (ft.count > 1) ? poly_dtype_vec(POLY_BOOL, ft.count) : POLY_BOOL;
  PolyDType it = (ft.count > 1) ? poly_dtype_vec(POLY_INT32, ft.count) : POLY_INT32;

  PolyUOp *base_zero = poly_const_like_float(ctx, base, 0.0);
  PolyUOp *exp_zero = poly_const_like_float(ctx, exponent, 0.0);
  PolyUOp *one = poly_const_like_float(ctx, root, 1.0);
  PolyUOp *nanv = poly_const_like_float(ctx, root, NAN);

  PolyUOp *base_lt0 = poly_uop2(ctx, POLY_OP_CMPLT, bt, base, base_zero, poly_arg_none());
  PolyUOp *abs_base = poly_uop3(
      ctx, POLY_OP_WHERE, base->dtype, base_lt0,
      poly_uop1(ctx, POLY_OP_NEG, base->dtype, base, poly_arg_none()), base, poly_arg_none()
  );

  PolyUOp *log_abs = poly_uop1(ctx, POLY_OP_LOG2, ft, abs_base, poly_arg_none());
  PolyUOp *scaled = poly_uop2(ctx, POLY_OP_MUL, ft, log_abs, exponent, poly_arg_none());
  PolyUOp *ret = poly_uop1(ctx, POLY_OP_EXP2, ft, scaled, poly_arg_none());

  PolyUOp *exp_int = poly_uop1(ctx, POLY_OP_CAST, it, exponent, poly_arg_none());
  PolyUOp *exp_back = poly_uop1(ctx, POLY_OP_CAST, ft, exp_int, poly_arg_none());
  PolyUOp *non_int = poly_uop2(ctx, POLY_OP_CMPNE, bt, exponent, exp_back, poly_arg_none());

  PolyUOp *exp_lt0 = poly_uop2(ctx, POLY_OP_CMPLT, bt, exponent, exp_zero, poly_arg_none());
  PolyUOp *abs_exp = poly_uop3(
      ctx, POLY_OP_WHERE, ft, exp_lt0, poly_uop1(ctx, POLY_OP_NEG, ft, exponent, poly_arg_none()),
      exponent, poly_arg_none()
  );
  PolyUOp *abs_exp_int = poly_uop1(ctx, POLY_OP_CAST, it, abs_exp, poly_arg_none());
  PolyUOp *rem = poly_uop2(
      ctx, POLY_OP_FLOORMOD, it, abs_exp_int, poly_const_like_int(ctx, abs_exp_int, 2),
      poly_arg_none()
  );
  PolyUOp *is_odd = poly_uop1(ctx, POLY_OP_CAST, bt, rem, poly_arg_none());

  PolyUOp *signed_ret = poly_uop3(
      ctx, POLY_OP_WHERE, ft, is_odd, poly_uop1(ctx, POLY_OP_NEG, ft, ret, poly_arg_none()), ret,
      poly_arg_none()
  );
  PolyUOp *neg_base_result =
      poly_uop3(ctx, POLY_OP_WHERE, ft, non_int, nanv, signed_ret, poly_arg_none());

  PolyUOp *base_eq0 = poly_uop2(ctx, POLY_OP_CMPEQ, bt, base, base_zero, poly_arg_none());
  PolyUOp *exp_eq0 = poly_uop2(ctx, POLY_OP_CMPEQ, bt, exponent, exp_zero, poly_arg_none());
  PolyUOp *zero_pow_zero = poly_uop2(ctx, POLY_OP_AND, bt, base_eq0, exp_eq0, poly_arg_none());
  PolyUOp *base_result =
      poly_uop3(ctx, POLY_OP_WHERE, ft, base_lt0, neg_base_result, ret, poly_arg_none());
  return poly_uop3(ctx, POLY_OP_WHERE, ft, zero_pow_zero, one, base_result, poly_arg_none());
}

/* WHERE(cond, val, val) -> val */
static PolyUOp *rule_where_same(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)ctx;
  (void)root;
  return poly_bind(b, "val");
}

/* WHERE(true/false_const, c0, c1) -> c0 or c1 */
static PolyUOp *rule_where_const_gate(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)ctx;
  PolyUOp *gate = poly_bind(b, "gate");
  PolyUOp *c0 = poly_bind(b, "c0");
  PolyUOp *c1 = poly_bind(b, "c1");
  if (gate->arg.kind == POLY_ARG_BOOL) return gate->arg.b ? c0 : c1;
  if (gate->arg.kind == POLY_ARG_INT) return gate->arg.i ? c0 : c1;
  return NULL;
}

static bool is_invalid_const_uop(PolyUOp *u) {
  return u && u->op == POLY_OP_CONST && u->arg.kind == POLY_ARG_INVALID;
}

static bool invalid_gate_parts(PolyUOp *u, PolyUOp **cond, PolyUOp **value, PolyUOp **invalid) {
  if (!u || u->op != POLY_OP_WHERE || u->n_src != 3 || !is_invalid_const_uop(u->src[2]))
    return false;
  if (cond) *cond = u->src[0];
  if (value) *value = u->src[1];
  if (invalid) *invalid = u->src[2];
  return true;
}

/* Pinned tinygrad/uop/symbolic.py:315-433 validity-aware simplification.
 * This belongs beside symbolic, not in late codegen: schedule/indexing.py's
 * _apply_reshape runs it before the CALL body is formed. */
typedef struct {
  PolyUOp *expr;
  int64_t lo;
  int64_t hi;
  bool has_lo;
  bool has_hi;
  PolyUOp *fake;
} PolyValidExprBound;

static int valid_split_op(PolyUOp *u, PolyOps op, PolyUOp **out, int max_n) {
  if (!u || !out || max_n <= 0) return 0;
  if (u->op != op) {
    out[0] = u;
    return 1;
  }
  int n = 0;
  for (int i = 0; i < u->n_src && n < max_n; i++)
    n += valid_split_op(u->src[i], op, out + n, max_n - n);
  return n;
}

static bool valid_true_const(PolyUOp *u) {
  return u && u->op == POLY_OP_CONST &&
         ((u->arg.kind == POLY_ARG_BOOL && u->arg.b) ||
          (u->arg.kind == POLY_ARG_INT && u->arg.i == 1));
}

static PolyUOp *valid_and_all(PolyCtx *ctx, PolyUOp **clauses, int n) {
  PolyUOp *acc = NULL;
  for (int i = 0; i < n; i++) {
    if (valid_true_const(clauses[i])) continue;
    acc = acc ? poly_uop2(ctx, POLY_OP_AND, POLY_BOOL, acc, clauses[i], poly_arg_none())
              : clauses[i];
  }
  return acc ? acc : poly_const_typed(ctx, POLY_BOOL, 1.0);
}

static bool valid_parse_clause(
    PolyCtx *ctx,
    PolyUOp *clause,
    PolyUOp **expr_out,
    bool *is_upper_out,
    int64_t *bound_out
) {
  if (!clause || !expr_out || !is_upper_out || !bound_out) return false;
  if (clause->op == POLY_OP_CMPNE && clause->n_src == 2 &&
      valid_true_const(clause->src[1])) {
    PolyUOp *lt = clause->src[0];
    if (!lt || lt->op != POLY_OP_CMPLT || lt->n_src != 2 ||
        !poly_dtype_is_int(lt->src[0]->dtype))
      return false;
    int64_t lo = 0, hi = 0;
    poly_uop_minmax(ctx, lt->src[1], &lo, &hi);
    *expr_out = lt->src[0];
    *is_upper_out = false;
    *bound_out = lo;
    return true;
  }
  if (clause->op == POLY_OP_CMPLT && clause->n_src == 2 &&
      poly_dtype_is_int(clause->src[0]->dtype)) {
    int64_t lo = 0, hi = 0;
    poly_uop_minmax(ctx, clause->src[1], &lo, &hi);
    if (hi == INT64_MIN) return false;
    *expr_out = clause->src[0];
    *is_upper_out = true;
    *bound_out = hi - 1;
    return true;
  }
  return false;
}

static PolyUOp *valid_simplify_substitution(
    PolyCtx *ctx,
    PolyUOp *u,
    PolyUOp **exprs,
    PolyUOp **fakes,
    int n
) {
  if (!ctx || !u || !exprs || !fakes || n <= 0) return u;
  PolyUOp *substituted = poly_uop_substitute(ctx, u, exprs, fakes, n);
  if (!substituted || substituted == u) return u;
  substituted = poly_graph_rewrite(ctx, substituted, poly_symbolic());
  if (!substituted) return u;
  PolyUOp *restored = poly_uop_substitute(ctx, substituted, fakes, exprs, n);
  if (!restored) return u;
  PolyUOp *simplified = poly_graph_rewrite(ctx, restored, poly_symbolic());
  return simplified ? simplified : u;
}

static bool valid_all_same(PolyUOp **uops, int n) {
  if (!uops || n <= 0) return false;
  for (int i = 1; i < n; i++)
    if (uops[i] != uops[0]) return false;
  return true;
}

PolyUOp *poly_uop_given_valid(
    PolyCtx *ctx,
    PolyUOp *valid,
    PolyUOp *uop,
    bool try_simplex
) {
  if (!ctx || !valid || !uop) return uop;

  PolyUOp *clauses[128];
  int n_clauses = valid_split_op(valid, POLY_OP_AND, clauses, 128);
  PolyValidExprBound bounds[128] = {0};
  int n_bounds = 0;
  for (int i = 0; i < n_clauses; i++) {
    PolyUOp *expr = NULL;
    bool is_upper = false;
    int64_t bound = 0;
    if (!valid_parse_clause(ctx, clauses[i], &expr, &is_upper, &bound)) continue;
    int at = -1;
    for (int j = 0; j < n_bounds; j++)
      if (bounds[j].expr == expr) {
        at = j;
        break;
      }
    if (at < 0) {
      if (n_bounds >= 128) break;
      at = n_bounds++;
      bounds[at].expr = expr;
    }
    if (is_upper) {
      bounds[at].hi = bound;
      bounds[at].has_hi = true;
    } else {
      bounds[at].lo = bound;
      bounds[at].has_lo = true;
    }
  }
  if (n_bounds == 0) return uop;

  for (int i = 0; i < n_bounds; i++) {
    int64_t vmin = 0, vmax = 0;
    poly_uop_minmax(ctx, bounds[i].expr, &vmin, &vmax);
    if (!bounds[i].has_lo) bounds[i].lo = vmin;
    if (!bounds[i].has_hi) bounds[i].hi = vmax;
    if (bounds[i].lo > bounds[i].hi) continue;
    char name[48];
    snprintf(name, sizeof(name), "valid_bound_%d", i);
    bounds[i].fake = poly_uop0(
        ctx, POLY_OP_DEFINE_VAR, bounds[i].expr->dtype,
        poly_arg_define_var(name, bounds[i].lo, bounds[i].hi)
    );
    if (!bounds[i].fake) continue;

    PolyUOp *exprs[1] = {bounds[i].expr};
    PolyUOp *fakes[1] = {bounds[i].fake};
    uop = valid_simplify_substitution(ctx, uop, exprs, fakes, 1);

    if (try_simplex && bounds[i].lo == 1 && bounds[i].expr->op == POLY_OP_ADD) {
      PolyUOp *terms[128];
      int n_terms = valid_split_op(bounds[i].expr, POLY_OP_ADD, terms, 128);
      bool irreducible = n_terms > 0;
      for (int j = 0; j < n_terms; j++)
        if (!poly_opset_has(POLY_GROUP_IRREDUCIBLE, terms[j]->op)) irreducible = false;
      if (irreducible) {
        PolyUOp *candidates[128];
        bool complete = true;
        for (int j = 0; j < n_terms; j++) {
          int64_t term_lo = 0, term_hi = 0;
          poly_uop_minmax(ctx, terms[j], &term_lo, &term_hi);
          (void)term_lo;
          char term_name[64];
          snprintf(term_name, sizeof(term_name), "valid_simplex_%d_%d", i, j);
          PolyUOp *fake = poly_uop0(
              ctx, POLY_OP_DEFINE_VAR, terms[j]->dtype,
              poly_arg_define_var(term_name, 1, term_hi)
          );
          if (!fake) {
            complete = false;
            break;
          }
          PolyUOp *term_exprs[1] = {terms[j]}, *term_fakes[1] = {fake};
          candidates[j] = valid_simplify_substitution(ctx, uop, term_exprs, term_fakes, 1);
        }
        if (complete && valid_all_same(candidates, n_terms)) {
          uop = candidates[0];
        } else if (complete && uop->op == POLY_OP_STACK && uop->n_src == 2) {
          bool first_same = true, second_same = true;
          for (int j = 0; j < n_terms; j++) {
            if (!candidates[j] || candidates[j]->op != POLY_OP_STACK ||
                candidates[j]->n_src != 2) {
              first_same = second_same = false;
              break;
            }
            if (j > 0 && candidates[j]->src[0] != candidates[0]->src[0]) first_same = false;
            if (j > 0 && candidates[j]->src[1] != candidates[0]->src[1]) second_same = false;
          }
          PolyUOp *srcs[2] = {uop->src[0], uop->src[1]};
          if (first_same) srcs[0] = candidates[0]->src[0];
          if (second_same) srcs[1] = candidates[0]->src[1];
          if (srcs[0] != uop->src[0] || srcs[1] != uop->src[1])
            uop = poly_uop_tagged_arg(
                ctx, uop->op, uop->dtype, srcs, 2, uop->arg, uop->tag, uop->tag_arg
            );
        }
      }
    }
  }

  PolyUOp *exprs[128], *fakes[128];
  int n_candidates = 0;
  for (int i = 0; i < n_bounds; i++) {
    if (!bounds[i].fake) continue;
    exprs[n_candidates] = bounds[i].expr;
    fakes[n_candidates] = bounds[i].fake;
    n_candidates++;
  }
  if (n_candidates > 0)
    uop = valid_simplify_substitution(ctx, uop, exprs, fakes, n_candidates);
  return uop;
}

static bool valid_contains_op(PolyUOp *u, PolyOps op) {
  if (!u) return false;
  if (u->op == op) return true;
  for (int i = 0; i < u->n_src; i++)
    if (valid_contains_op(u->src[i], op)) return true;
  return false;
}

static int valid_priority(PolyCtx *ctx, PolyUOp *clause, PolyUOp **clauses, int n) {
  PolyUOp *expr = NULL;
  bool upper = false;
  int64_t bound = 0;
  if (!valid_parse_clause(ctx, clause, &expr, &upper, &bound)) return 0;
  (void)upper;
  (void)bound;
  int score = 0;
  for (int i = 0; i < n; i++)
    if (poly_uop_reachable(ctx, clauses[i], expr)) score--;
  return score;
}

static PolyUOp *rule_simplify_valid(PolyCtx *ctx, PolyUOp *valid, const PolyBindings *b) {
  (void)b;
  if (!valid || valid->op != POLY_OP_AND || valid_contains_op(valid, POLY_OP_INDEX)) return NULL;
  PolyUOp *clauses[128];
  int n = valid_split_op(valid, POLY_OP_AND, clauses, 128);
  int priorities[128];
  for (int i = 0; i < n; i++) priorities[i] = valid_priority(ctx, clauses[i], clauses, n);
  for (int i = 1; i < n; i++) {
    PolyUOp *clause = clauses[i];
    int priority = priorities[i], j = i;
    while (j > 0 && priority < priorities[j - 1]) {
      clauses[j] = clauses[j - 1];
      priorities[j] = priorities[j - 1];
      j--;
    }
    clauses[j] = clause;
    priorities[j] = priority;
  }

  PolyUOp *ret[128];
  int n_ret = 0;
  bool changed = false;
  for (int i = 0; i < n; i++) {
    bool duplicate = false;
    for (int j = 0; j < n_ret; j++)
      if (ret[j] == clauses[i]) duplicate = true;
    if (duplicate) {
      changed = true;
      continue;
    }
    PolyUOp *stmt = clauses[i];
    if (n_ret > 0) {
      PolyUOp *known = valid_and_all(ctx, ret, n_ret);
      PolyUOp *simplified = poly_uop_given_valid(ctx, known, stmt, true);
      if (simplified != stmt) changed = true;
      stmt = simplified;
    }
    if (stmt != clauses[i]) changed = true;
    ret[n_ret++] = stmt;
  }
  return changed ? valid_and_all(ctx, ret, n_ret) : NULL;
}

static PolyUOp *rule_gated_given_valid(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  PolyUOp *cond = NULL, *value = NULL, *invalid = NULL;
  if (!invalid_gate_parts(root, &cond, &value, &invalid) ||
      !poly_dtype_eq(poly_dtype_scalar(value->dtype), POLY_INDEX))
    return NULL;
  PolyUOp *simplified = poly_uop_given_valid(ctx, cond, value, false);
  if (!simplified || simplified == value) return NULL;
  return poly_uop3(ctx, POLY_OP_WHERE, root->dtype, cond, simplified, invalid, poly_arg_none());
}

static bool valid_clause_reaches_value_range(PolyCtx *ctx, PolyUOp *clause, PolyUOp *value) {
  int n_topo = 0;
  PolyUOp **topo = poly_toposort_alloc(ctx, clause, &n_topo);
  PolyUOp **ranges = n_topo > 0 ? malloc((size_t)n_topo * sizeof(*ranges)) : NULL;
  int n_ranges = ranges ? poly_uop_ranges(ctx, clause, ranges, n_topo) : 0;
  bool reaches = false;
  for (int i = 0; i < n_ranges && !reaches; i++)
    reaches = poly_uop_in_ranges(ctx, value, ranges[i]);
  free(ranges);
  poly_toposort_free(topo);
  return reaches;
}

static PolyUOp *rule_drop_and_clauses(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  PolyUOp *cond = NULL, *value = NULL, *invalid = NULL;
  if (!invalid_gate_parts(root, &cond, &value, &invalid)) return NULL;
  PolyUOp *clauses[128], *keep[128];
  int n = valid_split_op(cond, POLY_OP_AND, clauses, 128), n_keep = 0;
  for (int i = 0; i < n; i++)
    if (valid_clause_reaches_value_range(ctx, clauses[i], value)) keep[n_keep++] = clauses[i];
  if (n_keep == n) return NULL;
  PolyUOp *new_cond = valid_and_all(ctx, keep, n_keep);
  return poly_uop3(ctx, POLY_OP_WHERE, root->dtype, new_cond, value, invalid, poly_arg_none());
}

static _Thread_local PolyPatternMatcher *g_pm_simplify_valid = NULL;
static _Thread_local PolyPatternMatcher *g_pm_drop_and_clauses = NULL;

PolyPatternMatcher *poly_pm_simplify_valid(void) {
  if (g_pm_simplify_valid) return g_pm_simplify_valid;
  PolyRule rules[] = {
      {poly_pat_op(POLY_OP_AND, NULL, 0, "valid"), rule_simplify_valid},
      {poly_pat_op(POLY_OP_WHERE, NULL, 0, "root"), rule_gated_given_valid},
  };
  g_pm_simplify_valid =
      poly_pm_thread_cache(poly_pm_new(rules, (int)(sizeof(rules) / sizeof(rules[0]))));
  return g_pm_simplify_valid;
}

PolyPatternMatcher *poly_pm_drop_and_clauses(void) {
  if (g_pm_drop_and_clauses) return g_pm_drop_and_clauses;
  PolyRule rules[] = {
      {poly_pat_op(POLY_OP_WHERE, NULL, 0, "root"), rule_drop_and_clauses},
  };
  g_pm_drop_and_clauses =
      poly_pm_thread_cache(poly_pm_new(rules, (int)(sizeof(rules) / sizeof(rules[0]))));
  return g_pm_drop_and_clauses;
}

static bool invalid_propagation_comparison(PolyOps op) {
  return op == POLY_OP_CMPLT || op == POLY_OP_CMPEQ || op == POLY_OP_CMPNE;
}

/*
 * Pinned tinygrad/uop/symbolic.py:60-86 `propagate_invalid`.
 *
 * Invalid is a masked-index sentinel, not an ordinary constant. Lift it
 * through ALU before identities such as x*0 -> 0 can erase the mask:
 *
 *   ALU(WHERE(gate, x, Invalid), y)
 *     -> WHERE(gate, ALU(x, y), Invalid)
 */
static PolyUOp *rule_propagate_invalid_cast(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  if (!root || root->op != POLY_OP_CAST || root->n_src != 1) return NULL;
  PolyUOp *value = NULL;
  PolyUOp *invalid = NULL;
  if (!invalid_gate_parts(root->src[0], NULL, &value, &invalid)) return NULL;
  if (!poly_dtype_is_index(poly_dtype_scalar(invalid->dtype))) return NULL;
  return poly_uop1(ctx, POLY_OP_CAST, root->dtype, value, root->arg);
}

static PolyUOp *rule_propagate_invalid_unary(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  if (!root || root->n_src != 1 || !poly_opset_has(POLY_GROUP_UNARY, root->op)) return NULL;
  PolyUOp *cond = NULL;
  PolyUOp *value = NULL;
  PolyUOp *invalid = NULL;
  if (invalid_gate_parts(root->src[0], &cond, &value, &invalid)) {
    PolyUOp *inner = poly_uop1(ctx, root->op, root->dtype, value, root->arg);
    return poly_uop3(ctx, POLY_OP_WHERE, root->dtype, cond, inner, invalid, poly_arg_none());
  }
  return is_invalid_const_uop(root->src[0]) ? root->src[0] : NULL;
}

static PolyUOp *rule_propagate_invalid_binary(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  if (!root || root->n_src != 2 || !poly_opset_has(POLY_GROUP_BINARY, root->op)) return NULL;

  bool comparison = invalid_propagation_comparison(root->op);
  for (int side = 0; side < 2; side++) {
    PolyUOp *cond = NULL;
    PolyUOp *value = NULL;
    PolyUOp *invalid = NULL;
    if (!invalid_gate_parts(root->src[side], &cond, &value, &invalid)) continue;
    PolyUOp *src[2] = {root->src[0], root->src[1]};
    src[side] = value;
    PolyUOp *inner = poly_uop2(ctx, root->op, root->dtype, src[0], src[1], root->arg);
    if (comparison && poly_dtype_is_index(poly_dtype_scalar(invalid->dtype))) return inner;
    PolyUOp *fallback =
        comparison ? poly_uop1(ctx, POLY_OP_CAST, root->dtype, invalid, poly_arg_none()) : invalid;
    return poly_uop3(ctx, POLY_OP_WHERE, root->dtype, cond, inner, fallback, poly_arg_none());
  }

  if (!comparison) {
    if (is_invalid_const_uop(root->src[0])) return root->src[0];
    if (is_invalid_const_uop(root->src[1])) return root->src[1];
  }
  return NULL;
}

static PolyUOp *rule_propagate_invalid_where(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  if (!root || root->op != POLY_OP_WHERE || root->n_src != 3) return NULL;
  PolyUOp *a = root->src[0];
  PolyUOp *bval = root->src[1];
  PolyUOp *cval = root->src[2];

  /* where(cond, Invalid, value) -> where(!cond, value, Invalid). */
  if (is_invalid_const_uop(bval)) {
    if (is_invalid_const_uop(cval)) return bval;
    PolyUOp *not_a = poly_uop2(
        ctx, POLY_OP_CMPNE, a->dtype, a, poly_const_like_bool(ctx, a, true), poly_arg_none()
    );
    return poly_uop3(ctx, POLY_OP_WHERE, root->dtype, not_a, cval, bval, poly_arg_none());
  }

  PolyUOp *cond = NULL;
  PolyUOp *value = NULL;
  PolyUOp *invalid = NULL;
  if (invalid_gate_parts(bval, &cond, &value, &invalid) && !is_invalid_const_uop(cval)) {
    PolyUOp *new_gate = cond;
    if (a != cond) {
      PolyUOp *not_a = poly_uop2(
          ctx, POLY_OP_CMPNE, a->dtype, a, poly_const_like_bool(ctx, a, true), poly_arg_none()
      );
      new_gate = poly_uop2(ctx, POLY_OP_OR, a->dtype, not_a, cond, poly_arg_none());
    }
    PolyUOp *inner = poly_uop3(ctx, POLY_OP_WHERE, root->dtype, a, value, cval, poly_arg_none());
    return poly_uop3(ctx, POLY_OP_WHERE, root->dtype, new_gate, inner, invalid, poly_arg_none());
  }

  if (invalid_gate_parts(cval, &cond, &value, &invalid) && !is_invalid_const_uop(bval)) {
    PolyUOp *new_gate = poly_uop2(ctx, POLY_OP_OR, a->dtype, a, cond, poly_arg_none());
    PolyUOp *inner = poly_uop3(ctx, POLY_OP_WHERE, root->dtype, a, bval, value, poly_arg_none());
    return poly_uop3(ctx, POLY_OP_WHERE, root->dtype, new_gate, inner, invalid, poly_arg_none());
  }
  return NULL;
}

static PolyUOp *rule_propagate_invalid_bitcast(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  if (!root || root->op != POLY_OP_BITCAST || root->n_src != 1) return NULL;
  if (is_invalid_const_uop(root->src[0]))
    return poly_uop1(ctx, POLY_OP_CAST, root->dtype, root->src[0], poly_arg_none());

  PolyUOp *cond = NULL;
  PolyUOp *value = NULL;
  PolyUOp *invalid = NULL;
  if (!invalid_gate_parts(root->src[0], &cond, &value, &invalid)) return NULL;
  PolyUOp *true_value = poly_uop1(ctx, POLY_OP_BITCAST, root->dtype, value, root->arg);
  PolyUOp *false_value = poly_uop1(ctx, POLY_OP_BITCAST, root->dtype, invalid, root->arg);
  return poly_uop3(ctx, POLY_OP_WHERE, root->dtype, cond, true_value, false_value, poly_arg_none());
}

static bool is_true_const_uop(PolyUOp *u) {
  if (!u || u->op != POLY_OP_CONST) return false;
  if (!poly_dtype_eq(u->dtype, POLY_BOOL)) return false;
  return (u->arg.kind == POLY_ARG_BOOL && u->arg.b) ||
         (u->arg.kind == POLY_ARG_INT && u->arg.i != 0);
}

static bool is_false_const_uop(PolyUOp *u) {
  if (!u || u->op != POLY_OP_CONST) return false;
  if (!poly_dtype_eq(u->dtype, POLY_BOOL)) return false;
  return (u->arg.kind == POLY_ARG_BOOL && !u->arg.b) ||
         (u->arg.kind == POLY_ARG_INT && u->arg.i == 0);
}

/* WHERE(cond, true, false) -> cond
 * WHERE(cond, false, true) -> cond.logical_not()
 *
 * tinygrad symbolic.py:
 *   UPat.var("x", dtype=dtypes.bool).where(True, False) -> x
 *   UPat.var("x", dtype=dtypes.bool).where(False, True) -> x.logical_not()
 */
static PolyUOp *rule_where_bool_identity(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  if (!root || root->n_src != 3) return NULL;
  PolyUOp *cond = root->src[0];
  if (!cond || !poly_dtype_eq(cond->dtype, POLY_BOOL) || !poly_dtype_eq(root->dtype, POLY_BOOL))
    return NULL;
  if (is_true_const_uop(root->src[1]) && is_false_const_uop(root->src[2])) return cond;
  if (is_false_const_uop(root->src[1]) && is_true_const_uop(root->src[2]))
    return poly_uop2(
        ctx, POLY_OP_CMPNE, POLY_BOOL, cond, poly_const_like_bool(ctx, cond, true), poly_arg_none()
    );
  return NULL;
}

/* WHERE(CMPNE(cond, true), t, f) -> WHERE(cond, f, t)
 * Tinygrad symbolic.py:230-231:
 *   cond.logical_not().where(t, f) -> cond.where(f, t)
 * Keep the Invalid guard from tinygrad so we don't move Invalid into the
 * taken branch and perturb validity semantics. */
static PolyUOp *rule_where_logical_not(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  PolyUOp *cond = poly_bind(b, "cond");
  PolyUOp *t = poly_bind(b, "t");
  PolyUOp *f = poly_bind(b, "f");
  if (!root || root->n_src != 3 || !is_true_const_uop(root->src[0]->src[1])) return NULL;
  if (is_invalid_const_uop(f)) return NULL;
  return poly_uop3(ctx, POLY_OP_WHERE, root->dtype, cond, f, t, poly_arg_none());
}

static PolyUOp *strip_casted_index_ptr(PolyUOp *u) {
  while (u && u->op == POLY_OP_CAST && u->n_src >= 1)
    u = u->src[0];
  return (u && u->op == POLY_OP_INDEX && u->n_src >= 2) ? u : NULL;
}

/* Tinygrad symbolic.py load/store folding:
 *   LOAD(INDEX(buf, Invalid)) -> const_like(0)
 *   STORE(INDEX(buf, Invalid), ...) -> NOOP
 * This is what removes dead masked lanes after devectorization. */
static PolyUOp *rule_fold_invalid_load_store(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  if (!root || (root->op != POLY_OP_LOAD && root->op != POLY_OP_STORE) || root->n_src < 1)
    return NULL;
  PolyUOp *idx = strip_casted_index_ptr(root->src[0]);
  if (!idx || !is_invalid_const_uop(idx->src[1])) return NULL;
  if (root->op == POLY_OP_STORE) return poly_uop0(ctx, POLY_OP_NOOP, POLY_VOID, poly_arg_none());
  if (poly_dtype_eq(root->dtype, POLY_BOOL)) return poly_const_like_bool(ctx, root, false);
  if (poly_dtype_is_float(root->dtype)) return poly_const_like_float(ctx, root, 0.0);
  return poly_const_like_int(ctx, root, 0);
}

/* Pinned tinygrad/uop/symbolic.py:247: x+x -> x*2. The replacement keeps the
 * matched ADD dtype; x can have a concrete integer dtype under a weak-index
 * result in Polygrad's current shape adapter. */
static PolyUOp *rule_add_self(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  PolyUOp *x = poly_bind(b, "x");
  PolyUOp *two = poly_const_like_int(ctx, x, 2);
  return poly_uop2(ctx, POLY_OP_MUL, root->dtype, x, two, poly_arg_none());
}

/* ADD(ADD(a, x), x) -> ADD(a, MUL(x, 2))
 * Associative grouping of identical addends — handles the gradient
 * accumulation pattern where the same variable contributes via different
 * paths and ends up in nested ADDs. Commutative matching on both the
 * inner and outer ADD covers all orderings. */
static PolyUOp *rule_add_assoc_self(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  PolyUOp *a = poly_bind(b, "a");
  PolyUOp *x = poly_bind(b, "x");
  PolyUOp *two = poly_const_like_int(ctx, x, 2);
  /* Pinned tinygrad/uop/symbolic.py:248 keeps y+x*2 in the matched weak-index
   * algebra even when Polygrad's current variable adapter is concrete int32. */
  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, root->dtype, x, two, poly_arg_none());
  return poly_uop2(ctx, POLY_OP_ADD, root->dtype, a, mul, poly_arg_none());
}

static bool is_scalar_const_uop(PolyUOp *u) {
  return u && u->op == POLY_OP_CONST;
}

static bool const_uop_is_negative_one(PolyUOp *u) {
  if (!is_scalar_const_uop(u)) return false;
  if (u->arg.kind == POLY_ARG_FLOAT) return u->arg.f == -1.0;
  if (!arg_is_exact_integer(u->arg)) return false;
  PolyInt value = {0};
  bool ret = poly_int_from_arg(&value, u->arg) && exact_int_is_i64(&value, -1);
  poly_int_free(&value);
  return ret;
}

static bool match_binary_one_const(PolyUOp *u, PolyUOp **out_x, PolyUOp **out_c) {
  if (!u || u->n_src != 2) return false;
  if (is_scalar_const_uop(u->src[0]) && !is_scalar_const_uop(u->src[1])) {
    *out_c = u->src[0];
    *out_x = u->src[1];
    return true;
  }
  if (is_scalar_const_uop(u->src[1]) && !is_scalar_const_uop(u->src[0])) {
    *out_c = u->src[1];
    *out_x = u->src[0];
    return true;
  }
  return false;
}

/* Pinned tinygrad/uop/symbolic.py:242:
 *   x*c0 + x*c1 -> x*(c0+c1)
 * Keep this scoped to weak-index shape algebra: reassociating float terms can
 * change rounding, and this dependency group proves only as_shape parity. */
static PolyUOp *rule_add_same_base_const_terms(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  if (!root || root->op != POLY_OP_ADD || root->n_src != 2 || !poly_dtype_is_index(root->dtype))
    return NULL;

  PolyUOp *x0 = NULL, *c0 = NULL, *x1 = NULL, *c1 = NULL;
  if (!match_binary_one_const(root->src[0], &x0, &c0) ||
      !match_binary_one_const(root->src[1], &x1, &c1) || root->src[0]->op != POLY_OP_MUL ||
      root->src[1]->op != POLY_OP_MUL || x0 != x1 || c0->arg.kind != POLY_ARG_INT ||
      c1->arg.kind != POLY_ARG_INT)
    return NULL;

  int64_t coefficient = 0;
  if (!i64_add_ok(c0->arg.i, c1->arg.i, &coefficient)) return NULL;
  PolyUOp *combined = poly_const_like_int(ctx, c0, coefficient);
  return combined ? poly_uop2(ctx, POLY_OP_MUL, root->dtype, x0, combined, poly_arg_none()) : NULL;
}

/* tinygrad symbolic.py:
 *   (x:weakint + c).cast(signed_int) -> x.cast(signed_int) + c.cast(signed_int)
 *
 * This turns Tensor.arange's `(range + 1).cast(int) + -1` class index into
 * `range.cast(int)`, exposing the sparse one-hot load-collapse rule. Keep it
 * scoped to POLY_INDEX sources and signed integer destinations, matching
 * dtypes.weakint -> dtypes.sints. */
static PolyUOp *rule_cast_index_add_const_to_add_casts(
    PolyCtx *ctx,
    PolyUOp *root,
    const PolyBindings *b
) {
  (void)b;
  if (!root || root->op != POLY_OP_CAST || root->n_src != 1) return NULL;
  if (!poly_dtype_is_int(root->dtype) || poly_dtype_is_unsigned(root->dtype) ||
      poly_dtype_is_bool(root->dtype))
    return NULL;

  PolyUOp *add = root->src[0];
  if (!add || add->op != POLY_OP_ADD || add->n_src != 2) return NULL;
  if (!poly_dtype_eq(poly_dtype_scalar(add->dtype), POLY_INDEX)) return NULL;

  PolyUOp *x = NULL;
  PolyUOp *c = NULL;
  if (!match_binary_one_const(add, &x, &c)) return NULL;

  PolyUOp *x_cast = poly_uop1(ctx, POLY_OP_CAST, root->dtype, x, poly_arg_none());
  PolyUOp *c_cast = poly_uop1(ctx, POLY_OP_CAST, root->dtype, c, poly_arg_none());
  return poly_uop2(ctx, POLY_OP_ADD, root->dtype, x_cast, c_cast, poly_arg_none());
}

/* tinygrad symbolic.py:260-262
 *   x.alu(op, c1).alu(op, c2) -> x.alu(op, c1.alu(op, c2))
 * Implemented structurally for binary associative ops with one const in the
 * inner node and one const at the root, regardless of src ordering. */
static PolyUOp *rule_assoc_fold_consts(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  if (!root || root->n_src != 2) return NULL;
  if (!poly_opset_has(POLY_GROUP_ASSOCIATIVE, root->op)) return NULL;

  PolyUOp *inner = NULL, *c2 = NULL;
  if (!match_binary_one_const(root, &inner, &c2)) return NULL;
  if (!inner || inner->op != root->op || inner->n_src != 2) return NULL;

  PolyUOp *x = NULL, *c1 = NULL;
  if (!match_binary_one_const(inner, &x, &c1)) return NULL;

  PolyArg operands[2] = {c1->arg, c2->arg};
  PolyArg folded;
  PolyInt owned = {0};
  if (!exec_symbolic_const_alu(root->op, root->dtype, operands, 2, false, &folded, &owned) ||
      folded.kind == POLY_ARG_INVALID) {
    poly_int_free(&owned);
    return NULL;
  }
  PolyUOp *fc = poly_const_like(ctx, c1, folded);
  PolyUOp *ret = poly_uop2(ctx, root->op, root->dtype, x, fc, poly_arg_none());
  poly_int_free(&owned);
  return ret;
}

/* Pinned tinygrad/uop/symbolic.py:261:
 *   (c0 + x) < c1 -> x < (c1 - c0)
 * UPat builds commutative ADD patterns from both source permutations. The
 * matcher below does the same; keep the callback limited to exact constants
 * and use the shared unbounded-integer constant evaluator. */
static PolyUOp *rule_cmplt_add_const_bound(
    PolyCtx *ctx,
    PolyUOp *root,
    const PolyBindings *b
) {
  PolyUOp *x = poly_bind(b, "x");
  PolyUOp *c0 = poly_bind(b, "c0");
  PolyUOp *c1 = poly_bind(b, "c1");
  if (!x || !c0 || !c1) return NULL;

  PolyArg operands[2] = {c1->arg, c0->arg};
  PolyArg folded;
  PolyInt owned = {0};
  if (!exec_symbolic_const_alu(
          POLY_OP_SUB, c1->dtype, operands, 2, false, &folded, &owned
      ) ||
      folded.kind == POLY_ARG_INVALID) {
    poly_int_free(&owned);
    return NULL;
  }
  PolyUOp *bound = poly_const_like(ctx, c1, folded);
  poly_int_free(&owned);
  return bound ? poly_uop2(ctx, POLY_OP_CMPLT, root->dtype, x, bound, poly_arg_none()) : NULL;
}

/* tinygrad symbolic.py:
 *   y * (x + c) -> (y * x) + (y * c)
 *
 * tinygrad gates this rule to weak integer/index expressions. Applying it to
 * float arithmetic changes rounding behavior and also breaks useful vector ALU
 * structure, so keep the Polygrad port integer-only instead of distributing
 * general f32/f64 math such as linspace scaling.
 * Support both src orderings so we do not depend on a prior commutative flip
 * just to expose the add+const shape. */
static PolyUOp *rule_distribute_const_mul_over_add(
    PolyCtx *ctx,
    PolyUOp *root,
    const PolyBindings *b
) {
  (void)b;
  if (!root || root->op != POLY_OP_MUL || root->n_src != 2) return NULL;
  if (!poly_dtype_eq(poly_dtype_scalar(root->dtype), POLY_INDEX)) return NULL;

  for (int swap = 0; swap < 2; swap++) {
    PolyUOp *c = root->src[swap];
    PolyUOp *add = root->src[swap ^ 1];
    if (!is_scalar_const_uop(c) || !add || add->op != POLY_OP_ADD || add->n_src != 2) continue;

    PolyUOp *x = NULL, *add_c = NULL;
    if (!match_binary_one_const(add, &x, &add_c)) continue;

    PolyArg operands[2] = {c->arg, add_c->arg};
    PolyArg folded;
    PolyInt owned = {0};
    if (!exec_symbolic_const_alu(POLY_OP_MUL, root->dtype, operands, 2, false, &folded, &owned) ||
        folded.kind == POLY_ARG_INVALID) {
      poly_int_free(&owned);
      continue;
    }

    PolyUOp *scaled_x = poly_uop2(ctx, POLY_OP_MUL, root->dtype, x, c, poly_arg_none());
    PolyUOp *scaled_c = poly_const_like(ctx, add_c, folded);
    PolyUOp *ret = poly_uop2(ctx, POLY_OP_ADD, root->dtype, scaled_x, scaled_c, poly_arg_none());
    poly_int_free(&owned);
    return ret;
  }

  return NULL;
}

/* Pinned tinygrad/uop/symbolic.py:250:
 *   -1 * (x + c) -> (-x) + (-c)
 *
 * This exact negation rule is intentionally separate from line 251's
 * weakint-only arbitrary-constant distribution. It applies to concrete ints
 * and floats too; the fixed -1 coefficient preserves floating-point
 * operation order while exposing the negated constant. */
static PolyUOp *rule_distribute_negated_add(
    PolyCtx *ctx,
    PolyUOp *root,
    const PolyBindings *b
) {
  (void)b;
  if (!root || root->op != POLY_OP_MUL || root->n_src != 2) return NULL;

  for (int swap = 0; swap < 2; swap++) {
    PolyUOp *neg_one = root->src[swap];
    PolyUOp *add = root->src[swap ^ 1];
    if (!const_uop_is_negative_one(neg_one) || !add || add->op != POLY_OP_ADD ||
        add->n_src != 2)
      continue;

    PolyUOp *x = NULL, *c = NULL;
    if (!match_binary_one_const(add, &x, &c)) continue;

    PolyArg operands[2] = {neg_one->arg, c->arg};
    PolyArg folded;
    PolyInt owned = {0};
    if (!exec_symbolic_const_alu(
            POLY_OP_MUL, root->dtype, operands, 2, false, &folded, &owned
        ) ||
        folded.kind == POLY_ARG_INVALID) {
      poly_int_free(&owned);
      continue;
    }

    PolyUOp *neg_x =
        poly_uop2(ctx, POLY_OP_MUL, root->dtype, x, neg_one, poly_arg_none());
    PolyUOp *neg_c = poly_const_like(ctx, c, folded);
    PolyUOp *ret =
        neg_x && neg_c
            ? poly_uop2(ctx, POLY_OP_ADD, root->dtype, neg_x, neg_c, poly_arg_none())
            : NULL;
    poly_int_free(&owned);
    return ret;
  }

  return NULL;
}

/* Pinned tinygrad/uop/symbolic.py:488, in the broader `sym` matcher:
 *   (x:weakint + y) * c -> x*c + y*c
 *
 * This is deliberately not part of poly_symbolic(): pinned `symbolic` only
 * distributes the narrower add-with-constant spelling at line 251. The
 * broader codegen-stage rule exposes split RANGE strides to the expander and
 * devectorizer without reassociating floating-point or concrete-int math. */
static PolyUOp *rule_sym_distribute_weak_mul_over_add(
    PolyCtx *ctx,
    PolyUOp *root,
    const PolyBindings *b
) {
  (void)b;
  if (!root || root->op != POLY_OP_MUL || root->n_src != 2) return NULL;

  for (int swap = 0; swap < 2; swap++) {
    PolyUOp *c = root->src[swap];
    PolyUOp *add = root->src[swap ^ 1];
    if (!is_scalar_const_uop(c) || !add || add->op != POLY_OP_ADD || add->n_src != 2)
      continue;
    if (!poly_dtype_eq(poly_dtype_scalar(add->src[0]->dtype), POLY_INDEX) &&
        !poly_dtype_eq(poly_dtype_scalar(add->src[1]->dtype), POLY_INDEX))
      continue;

    PolyUOp *lhs =
        poly_uop2(ctx, POLY_OP_MUL, root->dtype, add->src[0], c, poly_arg_none());
    PolyUOp *rhs =
        poly_uop2(ctx, POLY_OP_MUL, root->dtype, add->src[1], c, poly_arg_none());
    return poly_uop2(ctx, POLY_OP_ADD, root->dtype, lhs, rhs, poly_arg_none());
  }
  return NULL;
}

/* Pinned tinygrad symbolic.py:286-287
 *   (x + c) + y -> (x + y) + c
 *   (x * c) * y -> (x * y) * c
 * UPat tries both source permutations for commutative ADD/MUL, so the nested
 * same-op node may be either outer source. */
static PolyUOp *rule_move_const_to_end(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  if (!root || root->n_src != 2) return NULL;
  if (root->op != POLY_OP_ADD && root->op != POLY_OP_MUL) return NULL;

  for (int swap = 0; swap < 2; swap++) {
    PolyUOp *inner = root->src[swap];
    PolyUOp *y = root->src[swap ^ 1];
    if (!inner || inner->op != root->op || inner->n_src != 2 || is_scalar_const_uop(y)) continue;

    PolyUOp *x = NULL;
    PolyUOp *c = NULL;
    if (!match_binary_one_const(inner, &x, &c)) continue;
    PolyUOp *xy = poly_uop2(ctx, root->op, root->dtype, x, y, poly_arg_none());
    return poly_uop2(ctx, root->op, root->dtype, xy, c, poly_arg_none());
  }
  return NULL;
}

static int cmp_i64(int64_t a, int64_t b) {
  return (a > b) - (a < b);
}

static int cmp_u16(uint16_t a, uint16_t b) {
  return (a > b) - (a < b);
}

static int cmp_bool(bool a, bool b) {
  return (a > b) - (a < b);
}

static int cmp_cstr(const char *a, const char *b) {
  if (a == b) return 0;
  if (!a) return -1;
  if (!b) return 1;
  return strcmp(a, b);
}

static int dtype_tuplize_cmp(PolyDType a, PolyDType b) {
  if (poly_dtype_eq(a, b)) return 0;
  int ret;
  if ((ret = cmp_i64(a.priority, b.priority))) return ret;
  if ((ret = cmp_u16(a.bitsize, b.bitsize))) return ret;
  if ((ret = cmp_u16(a.count, b.count))) return ret;
  if ((ret = cmp_bool(a.is_ptr, b.is_ptr))) return ret;
  if ((ret = cmp_i64(a.addrspace, b.addrspace))) return ret;
  if ((ret = cmp_u16(a.vcount, b.vcount))) return ret;
  if ((ret = cmp_i64(a.ptr_size, b.ptr_size))) return ret;
  return cmp_cstr(a.name, b.name);
}

static int arg_int_tuple_cmp(const int64_t *a, int an, const int64_t *b, int bn) {
  int n = an < bn ? an : bn;
  for (int i = 0; i < n; i++) {
    int ret = cmp_i64(a[i], b[i]);
    if (ret) return ret;
  }
  return (an > bn) - (an < bn);
}

static int arg_pair_tuple_cmp(int64_t (*a)[2], int an, int64_t (*b)[2], int bn) {
  int n = an < bn ? an : bn;
  for (int i = 0; i < n; i++) {
    int ret = cmp_i64(a[i][0], b[i][0]);
    if (ret) return ret;
    ret = cmp_i64(a[i][1], b[i][1]);
    if (ret) return ret;
  }
  return (an > bn) - (an < bn);
}

static int arg_string_tuple_cmp(const char **a, int an, const char **b, int bn) {
  int n = an < bn ? an : bn;
  for (int i = 0; i < n; i++) {
    int ret = cmp_cstr(a[i], b[i]);
    if (ret) return ret;
  }
  return (an > bn) - (an < bn);
}

static int arg_tuplize_cmp(PolyArg a, PolyArg b) {
  if (poly_arg_eq(a, b)) return 0;
  if (a.kind != b.kind) return (a.kind > b.kind) - (a.kind < b.kind);
  switch (a.kind) {
  case POLY_ARG_NONE:
  case POLY_ARG_INVALID:
    return 0;
  case POLY_ARG_INT:
    return cmp_i64(a.i, b.i);
  case POLY_ARG_BIGINT: {
    bool ok = false;
    int ret = poly_arg_integer_cmp(a, b, &ok);
    return ok ? ret : 0;
  }
  case POLY_ARG_FLOAT: {
    uint64_t av, bv;
    memcpy(&av, &a.f, sizeof(av));
    memcpy(&bv, &b.f, sizeof(bv));
    return (av > bv) - (av < bv);
  }
  case POLY_ARG_BOOL:
    return cmp_bool(a.b, b.b);
  case POLY_ARG_INT_TUPLE:
    return arg_int_tuple_cmp(a.int_tuple.vals, a.int_tuple.n, b.int_tuple.vals, b.int_tuple.n);
  case POLY_ARG_PAIR_TUPLE:
    return arg_pair_tuple_cmp(
        a.pair_tuple.pairs, a.pair_tuple.n, b.pair_tuple.pairs, b.pair_tuple.n
    );
  case POLY_ARG_STRING:
    return cmp_cstr(a.str, b.str);
  case POLY_ARG_STRING_TUPLE:
    return arg_string_tuple_cmp(
        a.string_tuple.vals, a.string_tuple.n, b.string_tuple.vals, b.string_tuple.n
    );
  case POLY_ARG_OPS:
    return cmp_i64((int64_t)a.ops, (int64_t)b.ops);
  case POLY_ARG_REDUCE_AXIS: {
    int ret = cmp_i64((int64_t)a.reduce_axis.op, (int64_t)b.reduce_axis.op);
    if (ret) return ret;
    return arg_int_tuple_cmp(
        a.reduce_axis.axes, a.reduce_axis.n, b.reduce_axis.axes, b.reduce_axis.n
    );
  }
  case POLY_ARG_RANGE: {
    int ret = cmp_i64(a.range.axis_id, b.range.axis_id);
    if (ret) return ret;
    ret = cmp_i64((int64_t)a.range.axis_type, (int64_t)b.range.axis_type);
    if (ret) return ret;
    return arg_int_tuple_cmp(a.range.extra, a.range.n_extra, b.range.extra, b.range.n_extra);
  }
  case POLY_ARG_DEFINE_VAR: {
    int ret = cmp_cstr(a.define_var.name, b.define_var.name);
    if (ret) return ret;
    ret = cmp_i64(a.define_var.min_val, b.define_var.min_val);
    if (ret) return ret;
    return cmp_i64(a.define_var.max_val, b.define_var.max_val);
  }
  case POLY_ARG_BUFFERIZE_OPTS: {
    int ret = cmp_bool(a.bufferize_opts.device_is_tuple, b.bufferize_opts.device_is_tuple);
    if (ret) return ret;
    if (a.bufferize_opts.device_is_tuple)
      ret = arg_string_tuple_cmp(
          a.bufferize_opts.devices, a.bufferize_opts.n_devices,
          b.bufferize_opts.devices, b.bufferize_opts.n_devices
      );
    else
      ret = cmp_cstr(a.bufferize_opts.device, b.bufferize_opts.device);
    if (ret) return ret;
    ret = cmp_i64((int64_t)a.bufferize_opts.addrspace, (int64_t)b.bufferize_opts.addrspace);
    if (ret) return ret;
    return cmp_bool(a.bufferize_opts.removable, b.bufferize_opts.removable);
  }
  case POLY_ARG_TENSOR_CORE: {
    int ret = cmp_cstr(a.tensor_core.name, b.tensor_core.name);
    if (ret) return ret;
    for (int i = 0; i < 3; i++) {
      ret = cmp_i64(a.tensor_core.dims[i], b.tensor_core.dims[i]);
      if (ret) return ret;
    }
    return cmp_i64(a.tensor_core.threads, b.tensor_core.threads);
  }
  case POLY_ARG_PROGRAM_INFO: {
    uint32_t ah = poly_program_info_hash(a.program_info);
    uint32_t bh = poly_program_info_hash(b.program_info);
    if (ah != bh) return (ah > bh) - (ah < bh);
    return (a.program_info > b.program_info) - (a.program_info < b.program_info);
  }
  case POLY_ARG_BYTES: {
    int n = a.bytes.n < b.bytes.n ? a.bytes.n : b.bytes.n;
    for (int i = 0; i < n; i++) {
      uint8_t av = a.bytes.data ? a.bytes.data[i] : 0;
      uint8_t bv = b.bytes.data ? b.bytes.data[i] : 0;
      if (av != bv) return (av > bv) - (av < bv);
    }
    return (a.bytes.n > b.bytes.n) - (a.bytes.n < b.bytes.n);
  }
  case POLY_ARG_PARAM: {
    if (!a.param || !b.param) return (a.param != NULL) - (b.param != NULL);
    int ret = cmp_i64(a.param->slot, b.param->slot);
    if (ret) return ret;
    ret = cmp_bool(a.param->has_minmax, b.param->has_minmax);
    if (ret) return ret;
    if (a.param->has_minmax) {
      ret = cmp_i64(a.param->min_val, b.param->min_val);
      if (ret) return ret;
      ret = cmp_i64(a.param->max_val, b.param->max_val);
      if (ret) return ret;
    }
    ret = cmp_cstr(a.param->name, b.param->name);
    if (ret) return ret;
    ret = cmp_i64((int64_t)a.param->addrspace, (int64_t)b.param->addrspace);
    if (ret) return ret;
    ret = cmp_bool(a.param->has_axis, b.param->has_axis);
    if (ret) return ret;
    if (a.param->has_axis && (ret = cmp_i64(a.param->axis, b.param->axis))) return ret;
    ret = cmp_cstr(a.param->device, b.param->device);
    if (ret) return ret;
    ret = cmp_bool(a.param->device_is_tuple, b.param->device_is_tuple);
    if (ret) return ret;
    ret = cmp_i64(a.param->n_devices, b.param->n_devices);
    if (ret) return ret;
    for (int i = 0; i < a.param->n_devices; i++) {
      ret = cmp_cstr(
          a.param->devices ? a.param->devices[i] : NULL,
          b.param->devices ? b.param->devices[i] : NULL
      );
      if (ret) return ret;
    }
    return 0;
  }
  case POLY_ARG_CALL_INFO: {
    if (!a.call_info || !b.call_info)
      return (a.call_info != NULL) - (b.call_info != NULL);
    int ret = cmp_bool(a.call_info->has_grad_fxn, b.call_info->has_grad_fxn);
    if (ret) return ret;
    ret = cmp_bool(a.call_info->has_metadata, b.call_info->has_metadata);
    if (ret) return ret;
    ret = cmp_cstr(a.call_info->name, b.call_info->name);
    if (ret) return ret;
    ret = cmp_bool(a.call_info->precompile, b.call_info->precompile);
    if (ret) return ret;
    ret = cmp_bool(a.call_info->precompile_backward, b.call_info->precompile_backward);
    if (ret) return ret;
    return cmp_bool(a.call_info->has_aux, b.call_info->has_aux);
  }
  }
  return 0;
}

static int uop_tuplize_cmp(PolyUOp *a, PolyUOp *b) {
  if (a == b) return 0;
  if (!a) return -1;
  if (!b) return 1;

  /* Port of tinygrad UOp.tuplize tuple comparison:
   *   (op.value, arg, dtype) + tuple(src.tuplize for src in src)
   * This is used only by the weak-index commutative rule below, so it stays
   * local to symbolic rewriting instead of becoming a general identity rule. */
  int ret = cmp_i64((int64_t)a->op, (int64_t)b->op);
  if (ret) return ret;
  ret = arg_tuplize_cmp(a->arg, b->arg);
  if (ret) return ret;
  ret = dtype_tuplize_cmp(a->dtype, b->dtype);
  if (ret) return ret;

  int n = a->n_src < b->n_src ? a->n_src : b->n_src;
  for (int i = 0; i < n; i++) {
    ret = uop_tuplize_cmp(a->src[i], b->src[i]);
    if (ret) return ret;
  }
  return (a->n_src > b->n_src) - (a->n_src < b->n_src);
}

/* tinygrad symbolic.py:
 *   UPat(GroupOp.Commutative, dtype=dtypes.weakint)
 *     -> reverse srcs if src[1].tuplize < src[0].tuplize
 *
 * Polygrad's weak index dtype is POLY_INDEX. Keep the rule scoped there; doing
 * this for ordinary numeric ALU can disturb vector math merging, matching the
 * warning in tinygrad's own comment. */
static PolyUOp *rule_commutative_index_tuplize_order(
    PolyCtx *ctx,
    PolyUOp *root,
    const PolyBindings *b
) {
  (void)b;
  if (!root || root->n_src != 2) return NULL;
  if (!poly_dtype_eq(poly_dtype_scalar(root->dtype), POLY_INDEX)) return NULL;
  if (!poly_opset_has(POLY_GROUP_COMMUTATIVE, root->op)) return NULL;
  if (uop_tuplize_cmp(root->src[1], root->src[0]) >= 0) return NULL;
  return poly_uop2(ctx, root->op, root->dtype, root->src[1], root->src[0], root->arg);
}

static int64_t uop_const_factor(PolyUOp *u);

/* fold_divmod helpers (port of tinygrad divandmod.py) */

typedef struct {
  PolyUOp **items;
  int count;
  int cap;
  PolyUOp *inline_items[16];
} PolyAddTermList;

static void add_term_list_init(PolyAddTermList *list) {
  list->items = list->inline_items;
  list->count = 0;
  list->cap = (int)(sizeof(list->inline_items) / sizeof(list->inline_items[0]));
}

static void add_term_list_free(PolyAddTermList *list) {
  if (list->items != list->inline_items) free(list->items);
  list->items = NULL;
  list->count = list->cap = 0;
}

static bool add_term_list_push(PolyAddTermList *list, PolyUOp *u) {
  if (list->count == list->cap) {
    int new_cap = list->cap * 2;
    PolyUOp **new_items = NULL;
    if (list->items == list->inline_items) {
      new_items = malloc((size_t)new_cap * sizeof(*new_items));
      if (new_items) memcpy(new_items, list->items, (size_t)list->count * sizeof(*new_items));
    } else {
      new_items = realloc(list->items, (size_t)new_cap * sizeof(*new_items));
    }
    if (!new_items) return false;
    list->items = new_items;
    list->cap = new_cap;
  }
  list->items[list->count++] = u;
  return true;
}

/* Split ADD chain into a flat list of additive terms.
 * Mirrors tinygrad's unbounded `x.split_uop(Ops.ADD)` consumer model. */
static bool collect_add_terms(PolyUOp *u, PolyAddTermList *terms) {
  if (u->op == POLY_OP_ADD && u->n_src == 2)
    return collect_add_terms(u->src[0], terms) && collect_add_terms(u->src[1], terms);
  return add_term_list_push(terms, u);
}

/* Pinned UOp.split_uop(Ops.MUL), ops.py:598-601. Preserve source order and
 * multiplicity; this list is also used to rebuild the exact left-associated
 * products emitted by helpers.prod. */
static bool collect_mul_terms(PolyUOp *u, PolyAddTermList *terms) {
  if (u->op == POLY_OP_MUL && u->n_src == 2)
    return collect_mul_terms(u->src[0], terms) && collect_mul_terms(u->src[1], terms);
  return add_term_list_push(terms, u);
}

static PolyUOp *product_terms(
    PolyCtx *ctx, PolyDType dtype, PolyUOp **terms, int n_terms, PolyUOp *one_ref
) {
  if (n_terms == 0) return poly_const_like_int(ctx, one_ref, 1);
  PolyUOp *ret = terms[0];
  for (int i = 1; i < n_terms; i++)
    ret = poly_uop2(ctx, POLY_OP_MUL, dtype, ret, terms[i], poly_arg_none());
  return ret;
}

/* Pinned tinygrad/uop/symbolic.py:386-395,483-484 reduce_mul_chain.
 * RANGE-independent factors of additive reductions can be evaluated after
 * the REDUCE. MAX has the additional non-negative-factor condition because a
 * negative multiplier reverses the ordering. */
static PolyUOp *rule_sym_reduce_mul_chain(
    PolyCtx *ctx, PolyUOp *root, const PolyBindings *b
) {
  (void)b;
  if (!root || root->op != POLY_OP_REDUCE || root->n_src < 2 ||
      root->arg.kind != POLY_ARG_OPS ||
      (root->arg.ops != POLY_OP_ADD && root->arg.ops != POLY_OP_MAX) ||
      !poly_dtype_eq(root->dtype, root->src[0]->dtype) || root->src[0]->op != POLY_OP_MUL)
    return NULL;

  PolyAddTermList factors, inside, outside;
  add_term_list_init(&factors);
  add_term_list_init(&inside);
  add_term_list_init(&outside);
  if (!collect_mul_terms(root->src[0], &factors)) goto fail;

  for (int i = 0; i < factors.count; i++) {
    PolyUOp *factor = factors.items[i];
    bool depends_on_reduce = false;
    for (int j = 1; j < root->n_src && !depends_on_reduce; j++)
      depends_on_reduce = poly_uop_reachable(ctx, factor, root->src[j]);

    bool may_move = !depends_on_reduce;
    if (may_move && root->arg.ops == POLY_OP_MAX) {
      if (poly_dtype_is_float(poly_dtype_scalar(factor->dtype))) {
        may_move = factor->op == POLY_OP_CONST && factor->arg.kind == POLY_ARG_FLOAT &&
                   factor->arg.f >= 0.0;
      } else {
        int64_t vmin = 0, vmax = 0;
        poly_uop_minmax(ctx, factor, &vmin, &vmax);
        may_move = vmin >= 0;
      }
    }
    if (!add_term_list_push(may_move ? &outside : &inside, factor)) goto fail;
  }
  if (outside.count == 0) goto fail;

  PolyUOp *inside_value = product_terms(
      ctx, root->dtype, inside.items, inside.count, root->src[0]
  );
  if (!inside_value) goto fail;
  PolyUOp **reduce_src = malloc((size_t)root->n_src * sizeof(*reduce_src));
  if (!reduce_src) goto fail;
  reduce_src[0] = inside_value;
  for (int i = 1; i < root->n_src; i++) reduce_src[i] = root->src[i];
  PolyUOp *reduced = poly_uop_tagged_arg(
      ctx, root->op, root->dtype, reduce_src, root->n_src, root->arg, root->tag, root->tag_arg
  );
  free(reduce_src);
  PolyUOp *outside_value =
      product_terms(ctx, root->dtype, outside.items, outside.count, root->src[0]);
  PolyUOp *ret = reduced && outside_value
                     ? poly_uop2(
                           ctx, POLY_OP_MUL, root->dtype, reduced, outside_value, poly_arg_none()
                       )
                     : NULL;
  add_term_list_free(&factors);
  add_term_list_free(&inside);
  add_term_list_free(&outside);
  return ret;

fail:
  add_term_list_free(&factors);
  add_term_list_free(&inside);
  add_term_list_free(&outside);
  return NULL;
}

/* Get constant factor of a UOp (MUL(x,3)→3, CONST(5)→5, x→1) */
static int64_t uop_const_factor(PolyUOp *u) {
  if (u->op == POLY_OP_CONST && u->arg.kind == POLY_ARG_INT) return u->arg.i;
  if (u->op == POLY_OP_MUL && u->n_src == 2) {
    if (u->src[1]->op == POLY_OP_CONST && u->src[1]->arg.kind == POLY_ARG_INT)
      return u->src[1]->arg.i * uop_const_factor(u->src[0]);
    if (u->src[0]->op == POLY_OP_CONST && u->src[0]->arg.kind == POLY_ARG_INT)
      return u->src[0]->arg.i * uop_const_factor(u->src[1]);
  }
  return 1;
}

/* Divide UOp by constant factor: MUL(x,6)/3 → MUL(x,2) */
static PolyUOp *uop_divides(PolyCtx *ctx, PolyUOp *u, int64_t f) {
  if (f == 1) return u;
  if (f == 0) return NULL;
  if (u->op == POLY_OP_CONST && u->arg.kind == POLY_ARG_INT) {
    if (u->arg.i % f != 0) return NULL;
    return poly_uop0(ctx, POLY_OP_CONST, u->dtype, poly_arg_int(u->arg.i / f));
  }
  if (u->op == POLY_OP_MUL && u->n_src == 2) {
    if (u->src[1]->op == POLY_OP_CONST && u->src[1]->arg.kind == POLY_ARG_INT) {
      int64_t c = u->src[1]->arg.i;
      if (c % f == 0) {
        if (c / f == 1) return u->src[0];
        return poly_uop2(
            ctx, POLY_OP_MUL, u->dtype, u->src[0],
            poly_uop0(ctx, POLY_OP_CONST, u->dtype, poly_arg_int(c / f)), poly_arg_none()
        );
      }
    }
    if (u->src[0]->op == POLY_OP_CONST && u->src[0]->arg.kind == POLY_ARG_INT) {
      int64_t c = u->src[0]->arg.i;
      if (c % f == 0) {
        if (c / f == 1) return u->src[1];
        return poly_uop2(
            ctx, POLY_OP_MUL, u->dtype,
            poly_uop0(ctx, POLY_OP_CONST, u->dtype, poly_arg_int(c / f)), u->src[1], poly_arg_none()
        );
      }
    }
  }
  return NULL;
}

static PolyUOp *uop_mul_const_i64(PolyCtx *ctx, PolyDType dtype, PolyUOp *u, int64_t f) {
  if (f == 0) return poly_uop0(ctx, POLY_OP_CONST, dtype, poly_arg_int(0));
  if (f == 1) return u;
  PolyUOp *fc = poly_uop0(ctx, POLY_OP_CONST, dtype, poly_arg_int(f));
  return poly_uop2(ctx, POLY_OP_MUL, dtype, u, fc, poly_arg_none());
}

static PolyUOp *uop_sum_terms(PolyCtx *ctx, PolyDType dtype, PolyUOp **terms, int n_terms) {
  if (n_terms <= 0) return poly_uop0(ctx, POLY_OP_CONST, dtype, poly_arg_int(0));
  PolyUOp *ret = terms[0];
  for (int i = 1; i < n_terms; i++)
    ret = poly_uop2(ctx, POLY_OP_ADD, dtype, ret, terms[i], poly_arg_none());
  return ret;
}

/* Pinned tinygrad/uop/ops.py:928-945 const_factor/divides helpers used by
 * symbolic.py:174-178 lt_folding. Keep these separate from the div/mod
 * optimizer's recursive factor heuristic: tinygrad's MUL const_factor is the
 * immediate constant operand, not that constant times its other source's
 * factor. Constants are arbitrary-precision Python ints in the reference. */
static bool lt_poly_int_is_one(const PolyInt *value) {
  return value && value->sign == 1 && value->n_limbs == 1 && value->limbs[0] == 1;
}

static bool lt_poly_int_gcd(PolyInt *out, const PolyInt *a, const PolyInt *b) {
  PolyInt x = {0}, y = {0};
  if (!poly_int_copy(&x, a) || !poly_int_copy(&y, b)) goto fail;
  if (!poly_int_is_zero(&x)) x.sign = 1;
  if (!poly_int_is_zero(&y)) y.sign = 1;
  while (!poly_int_is_zero(&y)) {
    PolyInt q = {0}, r = {0};
    if (!poly_int_divmod(&q, &r, &x, &y, false)) {
      poly_int_free(&q);
      poly_int_free(&r);
      goto fail;
    }
    poly_int_free(&q);
    poly_int_free(&x);
    x = y;
    y = r;
  }
  *out = x;
  poly_int_init(&x);
  poly_int_free(&y);
  return true;
fail:
  poly_int_free(&x);
  poly_int_free(&y);
  return false;
}

static bool lt_uop_const_factor(PolyUOp *u, PolyInt *out) {
  if (!u || !out) return false;
  if (u->op == POLY_OP_CONST &&
      (u->arg.kind == POLY_ARG_INT || u->arg.kind == POLY_ARG_BIGINT))
    return poly_int_from_arg(out, u->arg);
  if (u->op == POLY_OP_STACK) {
    if (u->n_src == 0) return poly_int_from_i64(out, 0);
    PolyInt factor = {0};
    if (!lt_uop_const_factor(u->src[0], &factor)) return false;
    for (int i = 1; i < u->n_src; i++) {
      PolyInt next = {0}, gcd = {0};
      if (!lt_uop_const_factor(u->src[i], &next) ||
          !lt_poly_int_gcd(&gcd, &factor, &next)) {
        poly_int_free(&next);
        poly_int_free(&gcd);
        poly_int_free(&factor);
        return false;
      }
      poly_int_free(&next);
      poly_int_free(&factor);
      factor = gcd;
    }
    *out = factor;
    return true;
  }
  if (u->op == POLY_OP_ADD && u->n_src == 2) {
    PolyInt a = {0}, b = {0};
    bool ok = lt_uop_const_factor(u->src[0], &a) &&
              lt_uop_const_factor(u->src[1], &b) && lt_poly_int_gcd(out, &a, &b);
    poly_int_free(&a);
    poly_int_free(&b);
    return ok;
  }
  if (u->op == POLY_OP_MUL && u->n_src == 2) {
    for (int i = 0; i < 2; i++) {
      PolyUOp *source = u->src[i];
      if (source->op == POLY_OP_CONST &&
          (source->arg.kind == POLY_ARG_INT || source->arg.kind == POLY_ARG_BIGINT))
        return poly_int_from_arg(out, source->arg);
    }
  }
  return poly_int_from_i64(out, 1);
}

static PolyUOp *lt_uop_divides(PolyCtx *ctx, PolyUOp *u, const PolyInt *factor) {
  if (!ctx || !u || !factor || poly_int_is_zero(factor)) return NULL;
  if (lt_poly_int_is_one(factor)) return u;
  if (u->op == POLY_OP_CONST &&
      (u->arg.kind == POLY_ARG_INT || u->arg.kind == POLY_ARG_BIGINT)) {
    PolyInt value = {0}, quotient = {0}, remainder = {0};
    bool ok = poly_int_from_arg(&value, u->arg) &&
              poly_int_divmod(&quotient, &remainder, &value, factor, false) &&
              poly_int_is_zero(&remainder);
    PolyUOp *ret = ok ? poly_const_like(ctx, u, poly_int_as_arg(&quotient)) : NULL;
    poly_int_free(&value);
    poly_int_free(&quotient);
    poly_int_free(&remainder);
    return ret;
  }
  if (u->op == POLY_OP_STACK) {
    PolyUOp *inline_src[16];
    PolyUOp **src = u->n_src <= 16 ? inline_src : malloc((size_t)u->n_src * sizeof(*src));
    if (!src) return NULL;
    bool ok = true;
    for (int i = 0; i < u->n_src; i++)
      if (!(src[i] = lt_uop_divides(ctx, u->src[i], factor))) {
        ok = false;
        break;
      }
    PolyUOp *ret = ok ? poly_uop(ctx, POLY_OP_STACK, u->dtype, src, u->n_src, poly_arg_none()) : NULL;
    if (src != inline_src) free(src);
    return ret;
  }
  if (u->op == POLY_OP_ADD && u->n_src == 2) {
    PolyUOp *left = lt_uop_divides(ctx, u->src[0], factor);
    PolyUOp *right = lt_uop_divides(ctx, u->src[1], factor);
    return left && right
               ? poly_uop2(ctx, POLY_OP_ADD, u->dtype, left, right, poly_arg_none())
               : NULL;
  }
  if (u->op == POLY_OP_MUL && u->n_src == 2) {
    PolyUOp *left = lt_uop_divides(ctx, u->src[0], factor);
    if (left) return poly_uop2(ctx, POLY_OP_MUL, u->dtype, left, u->src[1], poly_arg_none());
    PolyUOp *right = lt_uop_divides(ctx, u->src[1], factor);
    if (right) return poly_uop2(ctx, POLY_OP_MUL, u->dtype, u->src[0], right, poly_arg_none());
  }
  return NULL;
}

/* Pinned tinygrad/uop/symbolic.py:174-178 lt_folding. For
 *   sum(non-unit-factor terms) + remainder < c
 * divide the non-unit terms and c by their GCD d when the exact remainder
 * range lies in [0,d). */
static PolyUOp *rule_lt_folding(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  if (!ctx || !root || root->op != POLY_OP_CMPLT || root->n_src != 2) return NULL;
  PolyUOp *x = root->src[0], *c = root->src[1];
  if (!poly_dtype_eq(poly_dtype_scalar(x->dtype), POLY_INDEX) ||
      c->op != POLY_OP_CONST ||
      (c->arg.kind != POLY_ARG_INT && c->arg.kind != POLY_ARG_BIGINT))
    return NULL;

  PolyInt c_value = {0}, divisor = {0}, one = {0}, zero = {0};
  if (!poly_int_from_arg(&c_value, c->arg) || c_value.sign <= 0 ||
      !poly_int_copy(&divisor, &c_value) || !poly_int_from_i64(&one, 1) ||
      !poly_int_from_i64(&zero, 0))
    goto cleanup;

  PolyAddTermList terms;
  add_term_list_init(&terms);
  if (!collect_add_terms(x, &terms) || terms.count == 0) {
    add_term_list_free(&terms);
    goto cleanup;
  }
  PolyUOp **unit = calloc((size_t)terms.count, sizeof(*unit));
  PolyUOp **non_unit = calloc((size_t)terms.count, sizeof(*non_unit));
  if (!unit || !non_unit) {
    free(unit);
    free(non_unit);
    add_term_list_free(&terms);
    goto cleanup;
  }
  int n_unit = 0, n_non_unit = 0;
  bool proved = true;
  for (int i = 0; i < terms.count; i++) {
    PolyInt factor = {0};
    if (!lt_uop_const_factor(terms.items[i], &factor)) {
      proved = false;
      poly_int_free(&factor);
      break;
    }
    if (lt_poly_int_is_one(&factor)) {
      unit[n_unit++] = terms.items[i];
    } else {
      PolyInt gcd = {0};
      non_unit[n_non_unit++] = terms.items[i];
      if (!lt_poly_int_gcd(&gcd, &divisor, &factor)) {
        proved = false;
        poly_int_free(&factor);
        break;
      }
      poly_int_free(&divisor);
      divisor = gcd;
    }
    poly_int_free(&factor);
  }

  PolyUOp *ret = NULL;
  if (proved && n_non_unit > 0 && poly_int_cmp(&divisor, &one) > 0) {
    PolyInt remainder_min = {0}, remainder_max = {0};
    proved = poly_int_from_i64(&remainder_min, 0) && poly_int_from_i64(&remainder_max, 0);
    for (int i = 0; proved && i < n_unit; i++) {
      ExactIntRange range = {0};
      PolyInt next_min = {0}, next_max = {0};
      proved = exact_int_range(ctx, unit[i], &range) &&
               poly_int_add(&next_min, &remainder_min, &range.lo) &&
               poly_int_add(&next_max, &remainder_max, &range.hi);
      exact_int_range_free(&range);
      if (proved) {
        poly_int_free(&remainder_min);
        poly_int_free(&remainder_max);
        remainder_min = next_min;
        remainder_max = next_max;
      } else {
        poly_int_free(&next_min);
        poly_int_free(&next_max);
      }
    }
    if (proved && poly_int_cmp(&remainder_min, &zero) >= 0 &&
        poly_int_cmp(&remainder_max, &divisor) < 0) {
      PolyUOp *non_unit_sum = uop_sum_terms(ctx, x->dtype, non_unit, n_non_unit);
      PolyUOp *divided = lt_uop_divides(ctx, non_unit_sum, &divisor);
      PolyInt quotient = {0}, rem = {0};
      if (divided && poly_int_divmod(&quotient, &rem, &c_value, &divisor, false) &&
          poly_int_is_zero(&rem)) {
        PolyUOp *bound = poly_const_like(ctx, c, poly_int_as_arg(&quotient));
        if (bound) ret = poly_uop2(ctx, POLY_OP_CMPLT, root->dtype, divided, bound, poly_arg_none());
      }
      poly_int_free(&quotient);
      poly_int_free(&rem);
    }
    poly_int_free(&remainder_min);
    poly_int_free(&remainder_max);
  }
  free(unit);
  free(non_unit);
  add_term_list_free(&terms);
  poly_int_free(&c_value);
  poly_int_free(&divisor);
  poly_int_free(&one);
  poly_int_free(&zero);
  return ret;

cleanup:
  poly_int_free(&c_value);
  poly_int_free(&divisor);
  poly_int_free(&one);
  poly_int_free(&zero);
  return NULL;
}

static PolyUOp *uop_sum_scaled_bases_raw(
    PolyCtx *ctx,
    PolyDType dtype,
    PolyUOp **bases,
    const int64_t *coeffs,
    int n_bases,
    int64_t const_part
) {
  PolyUOp *ret = poly_uop0(ctx, POLY_OP_CONST, dtype, poly_arg_int(0));
  for (int i = 0; i < n_bases; i++) {
    PolyUOp *coeff = poly_uop0(ctx, POLY_OP_CONST, dtype, poly_arg_int(coeffs[i]));
    PolyUOp *term = poly_uop2(ctx, POLY_OP_MUL, dtype, coeff, bases[i], poly_arg_none());
    ret = poly_uop2(ctx, POLY_OP_ADD, dtype, ret, term, poly_arg_none());
  }
  PolyUOp *constant = poly_uop0(ctx, POLY_OP_CONST, dtype, poly_arg_int(const_part));
  return poly_uop2(ctx, POLY_OP_ADD, dtype, ret, constant, poly_arg_none());
}

static int uop_backward_slice_size(PolyCtx *ctx, PolyUOp *u) {
  int n_topo = 0;
  PolyScratchMark scratch = poly_ctx_scratch_mark(ctx);
  (void)poly_toposort_scratch(ctx, u, &n_topo);
  poly_ctx_scratch_rewind(ctx, scratch);
  return n_topo;
}

/* Floor division (Python-style //) */
static int64_t floordiv(int64_t a, int64_t b) {
  int64_t q = a / b;
  if ((a ^ b) < 0 && q * b != a) q--;
  return q;
}

/* Floor mod (Python-style %) */
static int64_t poly_floormod(int64_t a, int64_t b) {
  return a - floordiv(a, b) * b;
}

/* GCD (Euclidean) */
static int64_t gcd64(int64_t a, int64_t b) {
  a = a < 0 ? -a : a;
  b = b < 0 ? -b : b;
  while (b) {
    int64_t t = b;
    b = a % b;
    a = t;
  }
  return a;
}

/* Check if a CONST UOp's value is exactly divisible by c. Returns quotient or 0. */
static int64_t uop_divides_const(PolyUOp *u, int64_t c) {
  if (c == 0) return 0;
  if (u->op == POLY_OP_CONST && u->arg.kind == POLY_ARG_INT && u->arg.i % c == 0)
    return u->arg.i / c;
  return 0;
}

static PolyUOp *fold_divmod_general(PolyCtx *ctx, PolyUOp *root);

static PolyUOp *try_fold_divmod_congruence(
    PolyCtx *ctx,
    PolyUOp *root,
    int64_t c,
    PolyUOp **bases,
    int64_t *base_mins,
    int64_t *base_maxs,
    int64_t *factors,
    int n_bases,
    int64_t additive_const
) {
  if (n_bases <= 0) return NULL;

  int64_t(*choices)[2] = calloc((size_t)n_bases, sizeof(*choices));
  int *n_choices = calloc((size_t)n_bases, sizeof(*n_choices));
  int *idx = calloc((size_t)n_bases, sizeof(*idx));
  int64_t *rems = calloc((size_t)n_bases, sizeof(*rems));
  int64_t *coeffs = calloc((size_t)n_bases, sizeof(*coeffs));
  if (!choices || !n_choices || !idx || !rems || !coeffs) {
    free(choices);
    free(n_choices);
    free(idx);
    free(rems);
    free(coeffs);
    return NULL;
  }

  PolyUOp *result = NULL;
  for (int i = 0; i < n_bases; i++) {
    int64_t r = poly_floormod(factors[i], c);
    int64_t r_neg, c_minus_r;
    if (!i64_sub_ok(r, c, &r_neg) || !i64_sub_ok(c, r, &c_minus_r)) goto cleanup;
    if (r == c_minus_r) {
      choices[i][0] = r;
      choices[i][1] = r_neg;
      n_choices[i] = 2;
    } else {
      choices[i][0] = (r < c_minus_r) ? r : r_neg;
      n_choices[i] = 1;
    }
  }

  while (true) {
    int64_t rem_lo = poly_floormod(additive_const, c);
    int64_t rem_hi = rem_lo;
    for (int i = 0; i < n_bases; i++) {
      rems[i] = choices[i][idx[i]];
      int64_t v0, v1;
      if (!i64_mul_ok(rems[i], base_mins[i], &v0) || !i64_mul_ok(rems[i], base_maxs[i], &v1))
        goto cleanup;
      int64_t lo = v0 < v1 ? v0 : v1, hi = v0 < v1 ? v1 : v0;
      if (!i64_add_ok(rem_lo, lo, &rem_lo) || !i64_add_ok(rem_hi, hi, &rem_hi)) goto cleanup;
    }

    int64_t interval, hi_interval;
    if (!i64_floordiv_ok(rem_lo, c, &interval) || !i64_floordiv_ok(rem_hi, c, &hi_interval))
      goto cleanup;
    if (interval == hi_interval) {
      if (root->op == POLY_OP_FLOORMOD) {
        int64_t interval_c, const_part;
        if (!i64_mul_ok(interval, c, &interval_c) ||
            !i64_sub_ok(poly_floormod(additive_const, c), interval_c, &const_part))
          goto cleanup;
        result = uop_sum_scaled_bases_raw(ctx, root->dtype, bases, rems, n_bases, const_part);
        break;
      }

      for (int i = 0; i < n_bases; i++) {
        int64_t factor_delta;
        if (!i64_sub_ok(factors[i], rems[i], &factor_delta) ||
            !i64_floordiv_ok(factor_delta, c, &coeffs[i]))
          goto cleanup;
      }
      int64_t additive_quotient, const_part;
      if (!i64_floordiv_ok(additive_const, c, &additive_quotient) ||
          !i64_add_ok(additive_quotient, interval, &const_part))
        goto cleanup;
      result = uop_sum_scaled_bases_raw(ctx, root->dtype, bases, coeffs, n_bases, const_part);
      break;
    }

    int pos = n_bases - 1;
    while (pos >= 0) {
      idx[pos]++;
      if (idx[pos] < n_choices[pos]) break;
      idx[pos] = 0;
      pos--;
    }
    if (pos < 0) break;
  }

cleanup:
  free(choices);
  free(n_choices);
  free(idx);
  free(rems);
  free(coeffs);
  return result;
}

static PolyUOp *try_nest_by_factor(
    PolyCtx *ctx,
    PolyUOp *root,
    PolyUOp *x,
    int64_t c,
    PolyUOp **terms,
    PolyUOp **bases,
    int64_t *factors,
    int n_terms,
    int64_t additive_const
) {
  int64_t *divs = calloc((size_t)(n_terms > 0 ? n_terms : 1), sizeof(*divs));
  if (!divs) return NULL;
  int n_divs = 0;
  for (int i = 0; i < n_terms; i++) {
    int64_t f = factors[i];
    if (f < 0 && !i64_neg_ok(f, &f)) continue;
    if (!(1 < f && f < c && c % f == 0)) continue;
    bool seen = false;
    for (int j = 0; j < n_divs; j++)
      if (divs[j] == f) seen = true;
    if (!seen) divs[n_divs++] = f;
  }

  PolyUOp *best = NULL;
  int best_size = 0;
  PolyUOp **b_parts = NULL;
  if (root->op == POLY_OP_FLOORMOD) {
    b_parts = calloc((size_t)n_terms + 1, sizeof(*b_parts));
    if (!b_parts) {
      free(divs);
      return NULL;
    }
  }

  for (int i = 0; i < n_divs; i++) {
    int64_t div = divs[i];
    PolyUOp *div_uop = poly_uop0(ctx, POLY_OP_CONST, root->dtype, poly_arg_int(div));
    PolyUOp *x_div = poly_uop2(ctx, POLY_OP_FLOORDIV, root->dtype, x, div_uop, poly_arg_none());
    PolyUOp *newxs = fold_divmod_general(ctx, x_div);
    if (!newxs) continue;
    int64_t nx_min, nx_max;
    poly_uop_minmax(ctx, newxs, &nx_min, &nx_max);
    if (nx_min < 0) continue;

    PolyUOp *candidate = NULL;
    if (root->op == POLY_OP_FLOORDIV) {
      PolyUOp *outer_den = poly_uop0(ctx, POLY_OP_CONST, root->dtype, poly_arg_int(c / div));
      candidate = poly_uop2(ctx, POLY_OP_FLOORDIV, root->dtype, newxs, outer_den, poly_arg_none());
      int size = uop_backward_slice_size(ctx, newxs);
      if (!best || size < best_size) {
        best = candidate;
        best_size = size;
      }
      continue;
    } else {
      int n_b = 0;
      for (int j = 0; j < n_terms; j++) {
        int64_t rem = factors[j] % div;
        if (rem != 0) b_parts[n_b++] = uop_mul_const_i64(ctx, root->dtype, bases[j], rem);
      }
      int64_t const_rem = poly_floormod(additive_const, div);
      if (const_rem != 0)
        b_parts[n_b++] = poly_uop0(ctx, POLY_OP_CONST, root->dtype, poly_arg_int(const_rem));
      PolyUOp *b = uop_sum_terms(ctx, root->dtype, b_parts, n_b);
      int64_t b_min, b_max;
      poly_uop_minmax(ctx, b, &b_min, &b_max);
      if (!(0 <= b_min && b_max < div)) continue;

      PolyUOp *outer_den = poly_uop0(ctx, POLY_OP_CONST, root->dtype, poly_arg_int(c / div));
      PolyUOp *inner =
          poly_uop2(ctx, POLY_OP_FLOORMOD, root->dtype, newxs, outer_den, poly_arg_none());
      PolyUOp *scaled = uop_mul_const_i64(ctx, root->dtype, inner, div);
      candidate = (n_b == 0) ? scaled
                             : poly_uop2(ctx, POLY_OP_ADD, root->dtype, scaled, b, poly_arg_none());
    }

    int size = uop_backward_slice_size(ctx, candidate);
    if (!best || size < best_size) {
      best = candidate;
      best_size = size;
    }
  }
  free(b_parts);
  free(divs);
  return best;
}

static PolyUOp *try_factor_remainder(
    PolyCtx *ctx,
    PolyUOp *root,
    PolyUOp *y,
    PolyUOp **all_terms,
    int n_terms,
    int64_t x_min,
    int64_t y_min
) {
  if (y_min < 0 || x_min < 0) return NULL;
  if (y->op != POLY_OP_CONST || y->arg.kind != POLY_ARG_INT || y->arg.i <= 0) return NULL;
  int64_t c = y->arg.i;

  PolyUOp **quo = calloc((size_t)(n_terms > 0 ? n_terms : 1), sizeof(*quo));
  PolyUOp **rem = calloc((size_t)(n_terms > 0 ? n_terms : 1), sizeof(*rem));
  if (!quo || !rem) {
    free(quo);
    free(rem);
    return NULL;
  }
  PolyUOp *result = NULL;
  int n_quo = 0, n_rem = 0;
  for (int i = 0; i < n_terms; i++) {
    PolyUOp *u = all_terms[i];
    PolyUOp *q = uop_divides(ctx, u, c);
    if (q) {
      quo[n_quo++] = q;
      continue;
    }

    int64_t f = uop_const_factor(u);
    int64_t f_rem = poly_floormod(f, c);
    if (f_rem != f) {
      PolyUOp *base = uop_divides(ctx, u, f);
      if (!base) goto cleanup;
      rem[n_rem++] = uop_mul_const_i64(ctx, root->dtype, base, f_rem);
      if (root->op == POLY_OP_FLOORDIV)
        quo[n_quo++] = uop_mul_const_i64(ctx, root->dtype, base, floordiv(f, c));
      else
        quo[n_quo++] = poly_uop0(ctx, POLY_OP_CONST, root->dtype, poly_arg_int(0));
    } else {
      rem[n_rem++] = u;
    }
  }

  if (n_quo == 0) goto cleanup;
  PolyUOp *new_x = uop_sum_terms(ctx, root->dtype, rem, n_rem);
  int64_t nx_min, nx_max;
  poly_uop_minmax(ctx, new_x, &nx_min, &nx_max);
  if (nx_min < 0) goto cleanup;

  if (root->op == POLY_OP_FLOORMOD) {
    result = poly_uop2(ctx, POLY_OP_FLOORMOD, root->dtype, new_x, y, poly_arg_none());
    goto cleanup;
  }

  PolyUOp *new_div = poly_uop2(ctx, POLY_OP_FLOORDIV, root->dtype, new_x, y, poly_arg_none());
  PolyUOp *quo_sum = uop_sum_terms(ctx, root->dtype, quo, n_quo);
  result = poly_uop2(ctx, POLY_OP_ADD, root->dtype, new_div, quo_sum, poly_arg_none());

cleanup:
  free(quo);
  free(rem);
  return result;
}

/* fold_divmod_general (port of tinygrad divandmod.py) */

static PolyUOp *fold_divmod_general(PolyCtx *ctx, PolyUOp *root) {
  if (root->n_src != 2 || !poly_dtype_is_index(root->dtype) ||
      (root->op != POLY_OP_FLOORDIV && root->op != POLY_OP_FLOORMOD))
    return NULL;
  PolyUOp *x = root->src[0], *y = root->src[1];
  int64_t x_min, x_max, y_min, y_max;
  poly_uop_minmax(ctx, x, &x_min, &x_max);
  poly_uop_minmax(ctx, y, &y_min, &y_max);

  /* 1. cancel_divmod: all corners give same quotient */
  /* Use same-sign check instead of y_min*y_max>0 to avoid int64 overflow */
  if ((y_min > 0 && y_max > 0) || (y_min < 0 && y_max < 0)) {
    int64_t q00, q01, q10, q11;
    if (i64_floordiv_ok(x_min, y_min, &q00) && i64_floordiv_ok(x_min, y_max, &q01) &&
        i64_floordiv_ok(x_max, y_min, &q10) && i64_floordiv_ok(x_max, y_max, &q11) && q00 == q01 &&
        q00 == q10 && q00 == q11) {
      int64_t q = q00;
      if (root->op == POLY_OP_FLOORMOD) {
        if (q == 0) return x;
        PolyUOp *qc = poly_uop0(ctx, POLY_OP_CONST, root->dtype, poly_arg_int(q));
        PolyUOp *qy = poly_uop2(ctx, POLY_OP_MUL, root->dtype, qc, y, poly_arg_none());
        return poly_uop2(
            ctx, POLY_OP_ADD, root->dtype, x,
            poly_uop1(ctx, POLY_OP_NEG, root->dtype, qy, poly_arg_none()), poly_arg_none()
        );
      }
      return poly_uop0(ctx, POLY_OP_CONST, root->dtype, poly_arg_int(q));
    }
  }

  /* Constant positive denominator required for remaining rules */
  if (y->op != POLY_OP_CONST || y->arg.i <= 0) return NULL;
  int64_t c = y->arg.i;

  /* 2. nested_div_mod: (x%(k*c))//c → (x//c)%k, (x%(k*c))%c → x%c */
  if (x->op == POLY_OP_FLOORMOD && x->n_src == 2) {
    int64_t k = uop_divides_const(x->src[1], c);
    if (k > 0) {
      if (root->op == POLY_OP_FLOORDIV) {
        PolyUOp *d = poly_uop2(ctx, POLY_OP_FLOORDIV, root->dtype, x->src[0], y, poly_arg_none());
        PolyUOp *kc = poly_uop0(ctx, POLY_OP_CONST, root->dtype, poly_arg_int(k));
        return poly_uop2(ctx, POLY_OP_FLOORMOD, root->dtype, d, kc, poly_arg_none());
      }
      return poly_uop2(ctx, POLY_OP_FLOORMOD, root->dtype, x->src[0], y, poly_arg_none());
    }
  }

  /* 3. remove_nested_mod: (a%4 + b)%2 → (a+b)%2 when x >= 0 */
  if (root->op == POLY_OP_FLOORMOD && x_min >= 0) {
    PolyAddTermList sum_terms;
    add_term_list_init(&sum_terms);
    if (collect_add_terms(x, &sum_terms) && sum_terms.count > 0) {
      bool changed = false;
      for (int i = 0; i < sum_terms.count; i++) {
        if (sum_terms.items[i]->op == POLY_OP_FLOORMOD && sum_terms.items[i]->n_src == 2 &&
            uop_divides_const(sum_terms.items[i]->src[1], c) > 0) {
          sum_terms.items[i] = sum_terms.items[i]->src[0];
          changed = true;
        }
      }
      if (changed) {
        PolyUOp *new_x = uop_sum_terms(ctx, root->dtype, sum_terms.items, sum_terms.count);
        int64_t nx_min, nx_max;
        poly_uop_minmax(ctx, new_x, &nx_min, &nx_max);
        if (nx_min >= 0) {
          PolyUOp *ret = poly_uop2(ctx, POLY_OP_FLOORMOD, root->dtype, new_x, y, poly_arg_none());
          add_term_list_free(&sum_terms);
          return ret;
        }
      }
    }
    add_term_list_free(&sum_terms);
  }

  if (x_min < 0) return NULL;

  /* Split x into additive terms */
  PolyAddTermList terms;
  add_term_list_init(&terms);
  if (!collect_add_terms(x, &terms) || terms.count == 0) {
    add_term_list_free(&terms);
    return NULL;
  }
  int n_terms = terms.count;

  /* Separate additive constant from non-constant terms */
  int64_t additive_const = 0;
  PolyUOp **nc_terms = calloc((size_t)n_terms, sizeof(*nc_terms));
  int64_t *nc_factors = calloc((size_t)n_terms, sizeof(*nc_factors));
  PolyUOp **bases = calloc((size_t)n_terms, sizeof(*bases));
  int64_t *base_mins = calloc((size_t)n_terms, sizeof(*base_mins));
  int64_t *base_maxs = calloc((size_t)n_terms, sizeof(*base_maxs));
  if (!nc_terms || !nc_factors || !bases || !base_mins || !base_maxs) {
    free(nc_terms);
    free(nc_factors);
    free(bases);
    free(base_mins);
    free(base_maxs);
    add_term_list_free(&terms);
    return NULL;
  }
  PolyUOp *result = NULL;
  int n_nc = 0;
  for (int i = 0; i < n_terms; i++) {
    if (terms.items[i]->op == POLY_OP_CONST && terms.items[i]->arg.kind == POLY_ARG_INT) {
      if (!i64_add_ok(additive_const, terms.items[i]->arg.i, &additive_const)) goto cleanup;
    } else {
      nc_terms[n_nc] = terms.items[i];
      nc_factors[n_nc] = uop_const_factor(nc_terms[n_nc]);
      n_nc++;
    }
  }

  /* Get base for each term: base[i] = nc_terms[i] / nc_factors[i] */
  for (int i = 0; i < n_nc; i++) {
    bases[i] = uop_divides(ctx, nc_terms[i], nc_factors[i]);
    if (!bases[i]) goto cleanup;
    poly_uop_minmax(ctx, bases[i], &base_mins[i], &base_maxs[i]);
  }

  /* 4. fold_binary_numerator: single non-const term with range of 2 */
  int64_t base_span = 0;
  if (n_nc == 1 && i64_sub_ok(base_maxs[0], base_mins[0], &base_span) && base_span == 1) {
    int64_t mul1, mul2, numerator1, numerator2;
    if (!i64_mul_ok(nc_factors[0], base_mins[0], &mul1) ||
        !i64_mul_ok(nc_factors[0], base_maxs[0], &mul2) ||
        !i64_add_ok(mul1, additive_const, &numerator1) ||
        !i64_add_ok(mul2, additive_const, &numerator2))
      goto cleanup;
    int64_t y1, y2;
    if (root->op == POLY_OP_FLOORMOD) {
      y1 = poly_floormod(numerator1, c);
      y2 = poly_floormod(numerator2, c);
    } else if (!i64_floordiv_ok(numerator1, c, &y1) || !i64_floordiv_ok(numerator2, c, &y2)) {
      goto cleanup;
    }
    /* result = (y2-y1)*(v-v_min) + y1 */
    int64_t slope, neg_base_min;
    if (!i64_sub_ok(y2, y1, &slope) || !i64_neg_ok(base_mins[0], &neg_base_min)) goto cleanup;
    PolyUOp *v_off = poly_uop2(
        ctx, POLY_OP_ADD, root->dtype, bases[0],
        poly_uop0(ctx, POLY_OP_CONST, root->dtype, poly_arg_int(neg_base_min)), poly_arg_none()
    );
    PolyUOp *r;
    if (slope == 0) {
      r = poly_uop0(ctx, POLY_OP_CONST, root->dtype, poly_arg_int(y1));
    } else if (slope == 1) {
      r = poly_uop2(
          ctx, POLY_OP_ADD, root->dtype, v_off,
          poly_uop0(ctx, POLY_OP_CONST, root->dtype, poly_arg_int(y1)), poly_arg_none()
      );
    } else {
      PolyUOp *sc = poly_uop0(ctx, POLY_OP_CONST, root->dtype, poly_arg_int(slope));
      r = poly_uop2(
          ctx, POLY_OP_ADD, root->dtype,
          poly_uop2(ctx, POLY_OP_MUL, root->dtype, sc, v_off, poly_arg_none()),
          poly_uop0(ctx, POLY_OP_CONST, root->dtype, poly_arg_int(y1)), poly_arg_none()
      );
    }
    result = r;
    goto cleanup;
  }

  result = try_fold_divmod_congruence(
      ctx, root, c, bases, base_mins, base_maxs, nc_factors, n_nc, additive_const
  );
  if (result) goto cleanup;

  /* 5. gcd_with_remainder: factor out GCD of all factors and c */
  if (x_min >= 0 && n_nc > 0) {
    int64_t g = c;
    for (int i = 0; i < n_nc; i++) {
      int64_t af = nc_factors[i];
      if (af < 0 && !i64_neg_ok(af, &af)) goto skip_gcd;
      g = gcd64(g, af);
    }
    if (g > 1) {
      /* Rebuild numerator / g */
      int64_t new_c = c / g;
      PolyUOp *new_x = NULL;
      for (int i = 0; i < n_nc; i++) {
        PolyUOp *divided = uop_divides(ctx, nc_terms[i], g);
        if (!divided) goto skip_gcd;
        new_x = new_x ? poly_uop2(ctx, POLY_OP_ADD, root->dtype, new_x, divided, poly_arg_none())
                      : divided;
      }
      /* Add (additive_const/g) % new_c */
      int64_t ac_g = floordiv(additive_const, g);
      int64_t ac_rem = poly_floormod(ac_g, new_c);
      if (ac_rem != 0 || !new_x) {
        PolyUOp *ac_uop = poly_uop0(ctx, POLY_OP_CONST, root->dtype, poly_arg_int(ac_rem));
        new_x = new_x ? poly_uop2(ctx, POLY_OP_ADD, root->dtype, new_x, ac_uop, poly_arg_none())
                      : ac_uop;
      }
      int64_t nx_min, nx_max;
      poly_uop_minmax(ctx, new_x, &nx_min, &nx_max);
      if (nx_min >= 0) {
        PolyUOp *new_y = poly_uop0(ctx, POLY_OP_CONST, root->dtype, poly_arg_int(new_c));
        if (root->op == POLY_OP_FLOORMOD) {
          PolyUOp *inner =
              poly_uop2(ctx, POLY_OP_FLOORMOD, root->dtype, new_x, new_y, poly_arg_none());
          PolyUOp *gc = poly_uop0(ctx, POLY_OP_CONST, root->dtype, poly_arg_int(g));
          PolyUOp *scaled = poly_uop2(ctx, POLY_OP_MUL, root->dtype, inner, gc, poly_arg_none());
          int64_t const_rem = poly_floormod(additive_const, g);
          if (const_rem != 0) {
            PolyUOp *cr = poly_uop0(ctx, POLY_OP_CONST, root->dtype, poly_arg_int(const_rem));
            result = poly_uop2(ctx, POLY_OP_ADD, root->dtype, scaled, cr, poly_arg_none());
            goto cleanup;
          }
          result = scaled;
          goto cleanup;
        }
        PolyUOp *inner =
            poly_uop2(ctx, POLY_OP_FLOORDIV, root->dtype, new_x, new_y, poly_arg_none());
        int64_t const_div = floordiv(additive_const, c);
        if (const_div != 0) {
          PolyUOp *cd = poly_uop0(ctx, POLY_OP_CONST, root->dtype, poly_arg_int(const_div));
          result = poly_uop2(ctx, POLY_OP_ADD, root->dtype, inner, cd, poly_arg_none());
          goto cleanup;
        }
        result = inner;
        goto cleanup;
      }
    }
  }
skip_gcd : {
  result = try_nest_by_factor(ctx, root, x, c, nc_terms, bases, nc_factors, n_nc, additive_const);
  if (result) goto cleanup;
}
  {
    PolyAddTermList all_terms;
    add_term_list_init(&all_terms);
    if (collect_add_terms(x, &all_terms) && all_terms.count > 0) {
      result = try_factor_remainder(ctx, root, y, all_terms.items, all_terms.count, x_min, y_min);
      if (result) {
        add_term_list_free(&all_terms);
        goto cleanup;
      }
    }
    add_term_list_free(&all_terms);
  }

cleanup:
  free(nc_terms);
  free(nc_factors);
  free(bases);
  free(base_mins);
  free(base_maxs);
  add_term_list_free(&terms);
  return result;
}

static PolyUOp *rule_cancel_divmod(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  return fold_divmod_general(ctx, root);
}

/* fold_add_divmod_recombine: dynamic divmod cancellation on ADD.
 * Port of tinygrad symbolic.py:28-48.  Replaces the old hardcoded pattern.
 * Handles:
 *   (base%div)*mul + (base//div)*(div*mul) -> base*mul
 *   ((base//d)%div)*mul + (base//(d*div))*(div*mul) -> (base//d)*mul
 *   ((base//div)%d)*div + base%div -> base%(div*d)
 */
static PolyUOp *rule_fold_add_divmod_recombine(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  if (root->op != POLY_OP_ADD) return NULL;
  if (!poly_dtype_is_index(root->dtype)) return NULL;

  PolyAddTermList terms;
  add_term_list_init(&terms);
  if (!collect_add_terms(root, &terms) || terms.count < 2) {
    add_term_list_free(&terms);
    return NULL;
  }
  int n = terms.count;
  PolyUOp *ret = NULL;

  for (int i = 0; i < n; i++) {
    PolyUOp *u = terms.items[i];
    PolyUOp *base = NULL;
    int64_t div_val = 0, mul_val = 0;

    /* Match: base%div (mul=1) */
    if (u->op == POLY_OP_FLOORMOD && u->n_src == 2 && u->src[1]->op == POLY_OP_CONST &&
        u->src[1]->arg.kind == POLY_ARG_INT) {
      base = u->src[0];
      div_val = u->src[1]->arg.i;
      mul_val = 1;
    }
    /* Match: (base%div)*mul */
    else if (u->op == POLY_OP_MUL && u->n_src == 2 && u->src[1]->op == POLY_OP_CONST &&
             u->src[1]->arg.kind == POLY_ARG_INT) {
      PolyUOp *m = u->src[0];
      if (m->op == POLY_OP_FLOORMOD && m->n_src == 2 && m->src[1]->op == POLY_OP_CONST &&
          m->src[1]->arg.kind == POLY_ARG_INT) {
        base = m->src[0];
        div_val = m->src[1]->arg.i;
        mul_val = u->src[1]->arg.i;
      }
    }
    if (!base || div_val == 0) continue;

    int64_t scaled_div;
    if (!i64_mul_ok(div_val, mul_val, &scaled_div)) continue;

    for (int j = 0; j < n; j++) {
      if (i == j) continue;
      PolyUOp *v = terms.items[j];
      /* v must be MUL(q, div*mul) */
      if (v->op != POLY_OP_MUL || v->n_src != 2 || v->src[1]->op != POLY_OP_CONST ||
          v->src[1]->arg.kind != POLY_ARG_INT || v->src[1]->arg.i != scaled_div)
        continue;
      PolyUOp *q = v->src[0];
      bool exact = false;

      /* (base%div)*mul + (base//div)*(div*mul) -> base*mul */
      if (q->op == POLY_OP_FLOORDIV && q->n_src == 2 && q->src[1]->op == POLY_OP_CONST &&
          q->src[1]->arg.kind == POLY_ARG_INT && q->src[1]->arg.i == div_val && q->src[0] == base) {
        exact = true;
      }
      /* ((base//d)%div)*mul + (base//(d*div))*(div*mul) -> (base//d)*mul */
      if (!exact && div_val > 0 && base->op == POLY_OP_FLOORDIV && base->n_src == 2 &&
          base->src[1]->op == POLY_OP_CONST && base->src[1]->arg.kind == POLY_ARG_INT) {
        int64_t nested_div;
        if (q->op == POLY_OP_FLOORDIV && q->n_src == 2 && q->src[1]->op == POLY_OP_CONST &&
            q->src[1]->arg.kind == POLY_ARG_INT && q->src[0] == base->src[0] &&
            i64_mul_ok(base->src[1]->arg.i, div_val, &nested_div) &&
            q->src[1]->arg.i == nested_div) {
          exact = true;
        }
      }
      if (exact) {
        /* result = base * mul + sum of remaining terms */
        PolyUOp *result;
        if (mul_val == 1) {
          result = base;
        } else {
          PolyUOp *mc = poly_const_like_int(ctx, root, mul_val);
          PolyUOp *ms[2] = {base, mc};
          result = poly_uop(ctx, POLY_OP_MUL, root->dtype, ms, 2, poly_arg_none());
        }
        for (int k = 0; k < n; k++) {
          if (k == i || k == j) continue;
          PolyUOp *as[2] = {result, terms.items[k]};
          result = poly_uop(ctx, POLY_OP_ADD, root->dtype, as, 2, poly_arg_none());
        }
        ret = result;
        goto cleanup;
      }

      /* ((base//div)%d)*div + base%div -> base%(div*d) */
      if (mul_val == 1 && div_val > 0 && q->op == POLY_OP_FLOORMOD && q->n_src == 2 &&
          q->src[1]->op == POLY_OP_CONST && q->src[1]->arg.kind == POLY_ARG_INT) {
        int64_t d = q->src[1]->arg.i;
        int64_t combined_modulus;
        if (d > 0 && q->src[0]->op == POLY_OP_FLOORDIV && q->src[0]->n_src == 2 &&
            q->src[0]->src[0] == base && q->src[0]->src[1]->op == POLY_OP_CONST &&
            q->src[0]->src[1]->arg.kind == POLY_ARG_INT && q->src[0]->src[1]->arg.i == div_val &&
            i64_mul_ok(div_val, d, &combined_modulus)) {
          PolyUOp *new_mod_c = poly_const_like_int(ctx, root, combined_modulus);
          PolyUOp *ms[2] = {base, new_mod_c};
          PolyUOp *result = poly_uop(ctx, POLY_OP_FLOORMOD, root->dtype, ms, 2, poly_arg_none());
          for (int k = 0; k < n; k++) {
            if (k == i || k == j) continue;
            PolyUOp *as[2] = {result, terms.items[k]};
            result = poly_uop(ctx, POLY_OP_ADD, root->dtype, as, 2, poly_arg_none());
          }
          ret = result;
          goto cleanup;
        }
      }
    }
  }

cleanup:
  add_term_list_free(&terms);
  return ret;
}

/* (x * y) / y → x.  Ref: tinygrad symbolic_simple line 90 */
static PolyUOp *rule_mul_fdiv_cancel(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  return poly_bind(b, "x");
}

/* WHERE(a, WHERE(b, c, d), d) → WHERE(AND(a, b), c, d).  Ref: tinygrad line 117 */
static PolyUOp *rule_nested_where(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  PolyUOp *a = poly_bind(b, "a");
  PolyUOp *bb = poly_bind(b, "b");
  PolyUOp *c = poly_bind(b, "c");
  PolyUOp *d = poly_bind(b, "d");
  PolyUOp *cond = poly_uop2(ctx, POLY_OP_AND, POLY_BOOL, a, bb, poly_arg_none());
  return poly_uop3(ctx, POLY_OP_WHERE, root->dtype, cond, c, d, poly_arg_none());
}

/* GEP(VECTORIZE(...)) -> VECTORIZE(select...) or scalar select */
static PolyUOp *rule_gep_vectorize(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  if (root->op != POLY_OP_GEP || root->n_src < 1) return NULL;
  PolyUOp *vec = root->src[0];
  if (!vec || vec->op != POLY_OP_VECTORIZE || vec->n_src <= 0) return NULL;

  if (root->arg.kind == POLY_ARG_INT) {
    int64_t idx = root->arg.i;
    if (idx < 0 || idx >= vec->n_src) return NULL;
    return vec->src[idx];
  }
  if (root->arg.kind != POLY_ARG_INT_TUPLE || root->arg.int_tuple.n <= 0) return NULL;
  int n = root->arg.int_tuple.n;
  if (n == 1) {
    int64_t idx = root->arg.int_tuple.vals[0];
    if (idx < 0 || idx >= vec->n_src) return NULL;
    return vec->src[idx];
  }
  PolyUOp **elts = calloc((size_t)n, sizeof(*elts));
  if (!elts) return NULL;
  for (int i = 0; i < n; i++) {
    int64_t idx = root->arg.int_tuple.vals[i];
    if (idx < 0 || idx >= vec->n_src) {
      free(elts);
      return NULL;
    }
    elts[i] = vec->src[idx];
  }
  PolyUOp *ret = poly_uop(ctx, POLY_OP_VECTORIZE, root->dtype, elts, n, poly_arg_none());
  free(elts);
  return ret;
}

/* GEP(CONST/VCONST) -> selected CONST(s). */
static PolyUOp *rule_gep_const(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  if (root->op != POLY_OP_GEP || root->n_src < 1) return NULL;
  PolyUOp *c = root->src[0];
  if (!c) return NULL;
  if (c->op == POLY_OP_CONST) {
    if (c->dtype.count <= 1) return c;
    if (root->arg.kind == POLY_ARG_INT)
      return poly_uop0(ctx, POLY_OP_CONST, poly_dtype_scalar(c->dtype), c->arg);
    if (root->arg.kind != POLY_ARG_INT_TUPLE || root->arg.int_tuple.n <= 0) return NULL;
    if (root->arg.int_tuple.n == 1)
      return poly_uop0(ctx, POLY_OP_CONST, poly_dtype_scalar(c->dtype), c->arg);
    return poly_uop0(ctx, POLY_OP_CONST, root->dtype, c->arg);
  }
  if (c->op != POLY_OP_VCONST) return NULL;

  if (root->arg.kind == POLY_ARG_INT) {
    int64_t idx = root->arg.i;
    if (idx < 0) return NULL;
    if (c->n_src > idx) return c->src[idx];
    if (c->arg.kind == POLY_ARG_INT_TUPLE && idx < c->arg.int_tuple.n)
      return poly_uop0(
          ctx, POLY_OP_CONST, poly_dtype_scalar(c->dtype), poly_arg_int(c->arg.int_tuple.vals[idx])
      );
    return NULL;
  }
  if (root->arg.kind != POLY_ARG_INT_TUPLE || root->arg.int_tuple.n <= 0) return NULL;
  int n = root->arg.int_tuple.n;
  if (n == 1) {
    int64_t idx = root->arg.int_tuple.vals[0];
    if (idx < 0) return NULL;
    if (c->n_src > idx) return c->src[idx];
    if (c->arg.kind == POLY_ARG_INT_TUPLE && idx < c->arg.int_tuple.n)
      return poly_uop0(
          ctx, POLY_OP_CONST, poly_dtype_scalar(c->dtype), poly_arg_int(c->arg.int_tuple.vals[idx])
      );
    return NULL;
  }
  PolyUOp **elts = calloc((size_t)n, sizeof(*elts));
  if (!elts) return NULL;
  for (int i = 0; i < n; i++) {
    int64_t idx = root->arg.int_tuple.vals[i];
    if (idx < 0) {
      free(elts);
      return NULL;
    }
    if (c->n_src > idx)
      elts[i] = c->src[idx];
    else if (c->arg.kind == POLY_ARG_INT_TUPLE && idx < c->arg.int_tuple.n)
      elts[i] = poly_uop0(
          ctx, POLY_OP_CONST, poly_dtype_scalar(c->dtype), poly_arg_int(c->arg.int_tuple.vals[idx])
      );
    else {
      free(elts);
      return NULL;
    }
  }
  PolyUOp *ret = poly_uop(ctx, POLY_OP_VECTORIZE, root->dtype, elts, n, poly_arg_none());
  free(elts);
  return ret;
}

/* Pinned tinygrad/uop/symbolic.py:213-214. GEP only selects value lanes;
 * a void effect has no lanes, so the wrapper is always the effect itself. */
static PolyUOp *rule_gep_void(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)ctx;
  (void)b;
  if (!root || root->op != POLY_OP_GEP || root->n_src != 1 || !root->src[0]) return NULL;
  return poly_dtype_eq(root->src[0]->dtype, POLY_VOID) ? root->src[0] : NULL;
}

/* GEP in natural order is identity. */
static PolyUOp *rule_gep_identity(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)ctx;
  (void)b;
  if (root->op != POLY_OP_GEP || root->n_src < 1) return NULL;
  PolyUOp *src = root->src[0];
  if (!src || src->dtype.is_ptr) return NULL;

  if (root->arg.kind == POLY_ARG_INT) {
    if (src->dtype.count == 1 && root->arg.i == 0) return src;
    return NULL;
  }
  if (root->arg.kind != POLY_ARG_INT_TUPLE || root->arg.int_tuple.n <= 0) return NULL;
  int n = root->arg.int_tuple.n;
  if (src->dtype.count == 1 && n == 1 && root->arg.int_tuple.vals[0] == 0) return src;
  if (n != src->dtype.count) return NULL;
  for (int i = 0; i < n; i++) {
    if (root->arg.int_tuple.vals[i] != i) return NULL;
  }
  return src;
}

/* GEP through ALU: GEP(ALU(a,b), i) → ALU(GEP(a,i), GEP(b,i))
 * (tinygrad symbolic.py:192-194, gep_pushing) */
static PolyUOp *rule_gep_through_alu(PolyCtx *ctx, PolyUOp *gep, const PolyBindings *b) {
  (void)b;
  if (gep->op != POLY_OP_GEP || gep->n_src < 1) return NULL;
  if (gep->dtype.is_ptr) return NULL;
  /* Only push GEP through integer ALU (index arithmetic).
   * tinygrad gates on dtype=dtypes.index; polygrad uses POLY_INT32 for indices. */
  if (!poly_dtype_is_int(poly_dtype_scalar(gep->dtype))) return NULL;
  PolyUOp *alu = gep->src[0];
  if (!alu || alu->dtype.is_ptr) return NULL;
  if (!poly_opset_has(POLY_GROUP_ALU, alu->op) && alu->op != POLY_OP_CAST &&
      alu->op != POLY_OP_BITCAST)
    return NULL;
  if (alu->dtype.count <= 1) return NULL; /* already scalar */
  /* Build new ALU with GEP pushed to each source */
  PolyUOp *srcs[8];
  if (alu->n_src > 8) return NULL;
  int gep_count = 1;
  if (gep->arg.kind == POLY_ARG_INT_TUPLE) gep_count = gep->arg.int_tuple.n;
  PolyDType new_dt = (gep_count > 1) ? poly_dtype_vec(poly_dtype_scalar(alu->dtype), gep_count)
                                     : poly_dtype_scalar(alu->dtype);
  for (int i = 0; i < alu->n_src; i++) {
    PolyUOp *s = alu->src[i];
    if (s->dtype.count > 1) {
      PolyDType s_new_dt = (gep_count > 1) ? poly_dtype_vec(poly_dtype_scalar(s->dtype), gep_count)
                                           : poly_dtype_scalar(s->dtype);
      srcs[i] = poly_uop1(ctx, POLY_OP_GEP, s_new_dt, s, gep->arg);
    } else {
      srcs[i] = s; /* scalar source passes through */
    }
  }
  if (alu->n_src == 1) return poly_uop1(ctx, alu->op, new_dt, srcs[0], alu->arg);
  if (alu->n_src == 2) return poly_uop2(ctx, alu->op, new_dt, srcs[0], srcs[1], alu->arg);
  if (alu->n_src == 3) return poly_uop3(ctx, alu->op, new_dt, srcs[0], srcs[1], srcs[2], alu->arg);
  return poly_uop(ctx, alu->op, new_dt, srcs, alu->n_src, alu->arg);
}

/* Tinygrad symbolic.py: VCAT cannot be rendered directly, so expand it into
 * VECTORIZE of per-lane GEPs early enough for later scalarization passes. */
static PolyUOp *rule_vcat_to_vectorize(PolyCtx *ctx, PolyUOp *x, const PolyBindings *b) {
  (void)b;
  if (!x || x->op != POLY_OP_VCAT || x->n_src <= 0 || x->dtype.is_ptr) return NULL;
  int total = 0;
  for (int i = 0; i < x->n_src; i++) {
    int cnt = x->src[i]->dtype.count > 0 ? x->src[i]->dtype.count : 1;
    if (cnt > INT32_MAX - total) return NULL;
    total += cnt;
  }
  if (total <= 0) return NULL;
  PolyUOp **elts = calloc((size_t)total, sizeof(*elts));
  if (!elts) return NULL;
  int p = 0;
  for (int i = 0; i < x->n_src; i++) {
    PolyUOp *src = x->src[i];
    int cnt = src->dtype.count > 0 ? src->dtype.count : 1;
    for (int j = 0; j < cnt; j++) {
      elts[p++] = poly_uop1(ctx, POLY_OP_GEP, poly_dtype_scalar(src->dtype), src, poly_arg_int(j));
    }
  }
  PolyUOp *ret = poly_uop(ctx, POLY_OP_VECTORIZE, x->dtype, elts, p, poly_arg_none());
  free(elts);
  return ret;
}

/* VECTORIZE(GEP(x, a0), GEP(x, a1), ...) where all GEPs share same source x
 * → x.gep((a0, a1, ...))  (collapse to single GEP with tuple arg)
 * Port of tinygrad symbolic.py:199:
 *   (UPat(Ops.VECTORIZE, src=UPat(Ops.GEP, src=(UPat.var("x"),))),
 *    lambda v,x: x.gep(tuple(get_single_element(i.arg) for i in v.src))) */
static PolyUOp *rule_vectorize_same_gep(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  if (root->op != POLY_OP_VECTORIZE || root->n_src < 2) return NULL;

  /* All sources must be single-lane GEP from the same source UOp */
  PolyUOp *base = NULL;
  for (int i = 0; i < root->n_src; i++) {
    PolyUOp *s = root->src[i];
    if (!s || s->op != POLY_OP_GEP || s->n_src < 1) return NULL;
    /* Accept both INT arg and INT_TUPLE with n=1 (tinygrad uses tuple form) */
    if (s->arg.kind != POLY_ARG_INT &&
        !(s->arg.kind == POLY_ARG_INT_TUPLE && s->arg.int_tuple.n == 1))
      return NULL;
    if (i == 0)
      base = s->src[0];
    else if (s->src[0] != base)
      return NULL;
  }
  if (!base) return NULL;

  /* Collect lane indices into tuple (stack — poly_uop copies to arena) */
  int n = root->n_src;
  int64_t *lanes = calloc((size_t)n, sizeof(*lanes));
  if (!lanes) return NULL;
  for (int i = 0; i < n; i++) {
    PolyUOp *s = root->src[i];
    lanes[i] = (s->arg.kind == POLY_ARG_INT) ? s->arg.i : s->arg.int_tuple.vals[0];
  }

  /* Create single GEP with tuple arg: base.gep((lane0, lane1, ...))
   * Then rule_gep_identity handles (0,1,...,N-1) → identity. */
  PolyArg tup;
  tup.kind = POLY_ARG_INT_TUPLE;
  tup.int_tuple.vals = lanes;
  tup.int_tuple.n = n;
  PolyUOp *ret = poly_uop1(ctx, POLY_OP_GEP, root->dtype, base, tup);
  free(lanes);
  return ret;
}

/* tinygrad symbolic.py: clean up singleton GROUP wrappers that appear after
 * store splitting/devectorization. Keeping GROUP(x) structurally distinct adds
 * a spurious late node compared to tinygrad's linearize path. */
static PolyUOp *rule_group_singleton(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)ctx;
  (void)b;
  if (!root || root->op != POLY_OP_GROUP || root->n_src != 1) return NULL;
  return root->src[0];
}

/* GEP pushing PM (for combined devec pass) */

static _Thread_local PolyPatternMatcher *g_pm_gep_pushing = NULL;

PolyPatternMatcher *poly_pm_gep_pushing(void) {
  if (g_pm_gep_pushing) return g_pm_gep_pushing;
  PolyRule rules[] = {
      {poly_pat_op(POLY_OP_GEP, NULL, 0, NULL), rule_gep_vectorize},
      {poly_pat_op(POLY_OP_GEP, NULL, 0, NULL), rule_gep_const},
      {poly_pat_op(POLY_OP_GEP, NULL, 0, NULL), rule_gep_void},
      {poly_pat_op(POLY_OP_GEP, NULL, 0, NULL), rule_gep_identity},
      {poly_pat_op(POLY_OP_GEP, NULL, 0, NULL), rule_gep_through_alu},
      /* VECTORIZE(GEP(x,a0), GEP(x,a1), ...) → x.gep((a0,a1,...))
       * tinygrad symbolic.py:199 */
      {poly_pat_op(POLY_OP_VECTORIZE, NULL, 0, NULL), rule_vectorize_same_gep},
  };
  g_pm_gep_pushing =
      poly_pm_thread_cache(poly_pm_new(rules, (int)(sizeof(rules) / sizeof(rules[0]))));
  return g_pm_gep_pushing;
}

/* Build the symbolic_simple PatternMatcher */

static _Thread_local PolyPatternMatcher *g_symbolic_simple = NULL;
static _Thread_local PolyPatternMatcher *g_symbolic = NULL;
static _Thread_local PolyPatternMatcher *g_sym = NULL;

PolyPatternMatcher *poly_symbolic_simple(void) {
  if (g_symbolic_simple) return g_symbolic_simple;

  /* Exclude THREEFRY from binary const fold */
  PolyOpSet binary_no_threefry = POLY_GROUP_BINARY;
  binary_no_threefry.bits[POLY_OP_THREEFRY / 64] &= ~((uint64_t)1 << (POLY_OP_THREEFRY % 64));

  /* CAST | BITCAST set */
  PolyOpSet cast_set =
      poly_opset_add(poly_opset_add((PolyOpSet){{0, 0}}, POLY_OP_CAST), POLY_OP_BITCAST);
  PolyOpSet constant_fold_value_set = poly_opset_add(
      poly_opset_add(poly_opset_add((PolyOpSet){{0, 0}}, POLY_OP_CONST), POLY_OP_VCONST),
      POLY_OP_STACK
  );

  PolyRule rules[] = {
      /*
       * Pinned tinygrad/uop/symbolic.py:60-86. These must precede ordinary
       * symbolic identities so Invalid remains a validity sentinel.
       */
      {poly_pat_op(POLY_OP_CAST, NULL, 0, "root"), rule_propagate_invalid_cast},
      {poly_pat_ops(POLY_GROUP_UNARY, NULL, 0, "root"), rule_propagate_invalid_unary},
      {poly_pat_ops(POLY_GROUP_BINARY, NULL, 0, "root"), rule_propagate_invalid_binary},
      {poly_pat_op(POLY_OP_WHERE, NULL, 0, "root"), rule_propagate_invalid_where},
      {poly_pat_op(POLY_OP_BITCAST, NULL, 0, "root"), rule_propagate_invalid_bitcast},

      /* -- Bool algebra (must come before generic ADD/MUL rules) -- */
      /* bool * bool -> AND */
      {poly_pat_op2(POLY_OP_MUL, poly_pat_any("x"), poly_pat_any("y"), NULL), rule_bool_mul_to_and},
      /* bool + bool -> OR */
      {poly_pat_op2(POLY_OP_ADD, poly_pat_any("x"), poly_pat_any("y"), NULL), rule_bool_add_to_or},

      /* -- Self-folding -- */
      /* x + 0 -> x */
      {poly_pat_op2c(POLY_OP_ADD, poly_pat_any("x"), poly_pat_const_val(poly_arg_int(0)), NULL),
       rule_identity},
      /* x + 0.0 -> x (float) */
      {poly_pat_op2c(POLY_OP_ADD, poly_pat_any("x"), poly_pat_const_val(poly_arg_float(0.0)), NULL),
       rule_identity},
      /* x * 1 -> x */
      {poly_pat_op2c(POLY_OP_MUL, poly_pat_any("x"), poly_pat_const_val(poly_arg_int(1)), NULL),
       rule_identity},
      /* x * 1.0 -> x (float) */
      {poly_pat_op2c(POLY_OP_MUL, poly_pat_any("x"), poly_pat_const_val(poly_arg_float(1.0)), NULL),
       rule_identity},
      /* AND(x, true) -> x */
      {poly_pat_op2c(POLY_OP_AND, poly_pat_any("x"), poly_pat_const_val(poly_arg_bool(true)), NULL),
       rule_identity},
      /* AND(x, false) -> false */
      {poly_pat_op2c(
           POLY_OP_AND, poly_pat_any("x"), poly_pat_const_val(poly_arg_bool(false)), NULL
       ),
       rule_and_zero},
      /* OR(x, false) -> x */
      {poly_pat_op2c(POLY_OP_OR, poly_pat_any("x"), poly_pat_const_val(poly_arg_bool(false)), NULL),
       rule_identity},
      /* OR(x, true) -> true */
      {poly_pat_op2c(POLY_OP_OR, poly_pat_any("x"), poly_pat_const_val(poly_arg_bool(true)), NULL),
       rule_or_one},
      /* x // x -> 1 */
      {poly_pat_op2(POLY_OP_FLOORDIV, poly_pat_any("x"), poly_pat_any("x"), NULL), rule_div_self},
      /* x // 1 -> x */
      {poly_pat_op2(POLY_OP_FLOORDIV, poly_pat_any("x"), poly_pat_const_val(poly_arg_int(1)), NULL),
       rule_identity},
      /* x // -1 -> -x */
      {poly_pat_op2(
           POLY_OP_FLOORDIV, poly_pat_any("x"), poly_pat_const_val(poly_arg_int(-1)), NULL
       ),
       rule_div_neg1},
      /* Idempotent(x, x) -> x */
      {poly_pat_ops2(POLY_GROUP_IDEMPOTENT, poly_pat_any("x"), poly_pat_any("x"), NULL),
       rule_idempotent},
      /* TRUNC(x) -> x for int/bool/weakint (tinygrad symbolic.py:114) */
      {poly_pat_op1(POLY_OP_TRUNC, poly_pat_any("x"), NULL), rule_trunc_integral_identity},

      /* -- Zero-folding -- */
      /* x < x -> False */
      {poly_pat_op2(POLY_OP_CMPLT, poly_pat_any("x"), poly_pat_any("x"), NULL), rule_lt_self},
      /* x != x -> False (int/bool only) */
      {poly_pat_op2(POLY_OP_CMPNE, poly_pat_any("x"), poly_pat_any("x"), NULL), rule_cmpne_self},
      /* x % x -> 0 */
      {poly_pat_op2(POLY_OP_FLOORMOD, poly_pat_any("x"), poly_pat_any("x"), NULL), rule_mod_self},
      /* x ^ 0 -> x */
      {poly_pat_op2c(POLY_OP_XOR, poly_pat_any("x"), poly_pat_const_val(poly_arg_int(0)), NULL),
       rule_identity},
      /* x ^ x -> 0 */
      {poly_pat_op2(POLY_OP_XOR, poly_pat_any("x"), poly_pat_any("x"), NULL), rule_xor_self},
      /* x & 0 -> 0 (tinygrad symbolic.py:98) */
      {poly_pat_op2c(POLY_OP_AND, poly_pat_any("x"), poly_pat_const_val(poly_arg_int(0)), NULL),
       rule_mul_zero},
      /* x * 0 -> 0 */
      {poly_pat_op2c(POLY_OP_MUL, poly_pat_any("x"), poly_pat_const_val(poly_arg_int(0)), NULL),
       rule_mul_zero},
      /* x * 0.0 -> 0 (float) */
      {poly_pat_op2c(POLY_OP_MUL, poly_pat_any("x"), poly_pat_const_val(poly_arg_float(0.0)), NULL),
       rule_mul_zero},

      /* -- Constant folding -- */
      /* Unary(CONST/STACK) -> CONST/STACK */
      {poly_pat_ops1(POLY_GROUP_UNARY, poly_pat_ops(constant_fold_value_set, NULL, 0, NULL), "a"),
       rule_const_fold_unary},
      /* Binary(CONST/STACK, CONST/STACK) -> CONST/STACK (excl. THREEFRY) */
      {poly_pat_ops2(
           binary_no_threefry, poly_pat_ops(constant_fold_value_set, NULL, 0, NULL),
           poly_pat_ops(constant_fold_value_set, NULL, 0, NULL), "a"
       ),
       rule_const_fold_binary},
      /* THREEFRY(CONST, CONST) -> exact threefry2x32(...).simplify().arg.
       * Pinned tinygrad/uop/symbolic.py:128-130. */
      {poly_pat_op2(
           POLY_OP_THREEFRY, poly_pat_cvar("x"), poly_pat_cvar("key"), "a"
       ),
       rule_threefry_const},
      /* Ternary(CONST/STACK, ...) -> CONST/STACK */
      {poly_pat_ops3(
           POLY_GROUP_TERNARY, poly_pat_ops(constant_fold_value_set, NULL, 0, NULL),
           poly_pat_ops(constant_fold_value_set, NULL, 0, NULL),
           poly_pat_ops(constant_fold_value_set, NULL, 0, NULL), "a"
       ),
       rule_const_fold_ternary},
      /* GEP simplifications */
      {poly_pat_op(POLY_OP_GEP, NULL, 0, NULL), rule_gep_vectorize},
      {poly_pat_op(POLY_OP_GEP, NULL, 0, NULL), rule_gep_const},
      {poly_pat_op(POLY_OP_GEP, NULL, 0, NULL), rule_gep_identity},
      /* VCAT -> VECTORIZE(GEP...) */
      {poly_pat_op(POLY_OP_VCAT, NULL, 0, "x"), rule_vcat_to_vectorize},
      /* GROUP(x) -> x */
      {poly_pat_op(POLY_OP_GROUP, NULL, 0, NULL), rule_group_singleton},
      /* VECTORIZE(GEP(x,a0), ...) → x.gep((a0,...)) is in gep_pushing (tinygrad symbolic.py:199).
       * Then rule_gep_identity handles the (0,1,...,N-1) → identity case. */

      /* -- POW folding (tinygrad symbolic.py:simplify_pow) -- */
      {poly_pat_op2(POLY_OP_POW, poly_pat_any("x"), poly_pat_cvar("c"), NULL),
       rule_simplify_pow_const_exp},
      {poly_pat_op2(POLY_OP_POW, poly_pat_cvar("c"), poly_pat_any("x"), NULL),
       rule_simplify_pow_const_base},

      /* -- Cast folding -- */
      /* Pinned symbolic.py:148 matches UPat.cvar, i.e. scalar CONST only.
       * Keep vector CAST(STACK) explicit for renderer and weak-index parity. */
      {poly_pat_op1(POLY_OP_CAST, poly_pat_op(POLY_OP_CONST, NULL, 0, "c"), NULL), rule_cast_const},
      {poly_pat_op1(POLY_OP_BITCAST, poly_pat_op(POLY_OP_CONST, NULL, 0, "c"), NULL),
       rule_bitcast_const},
      /* CAST/BITCAST same dtype -> identity */
      {poly_pat_ops(cast_set, NULL, 0, NULL), rule_cast_noop},
      /* b.cast(a).cast(b) -> b when a is lossless for b */
      {poly_pat_op1(POLY_OP_CAST, poly_pat_op1(POLY_OP_CAST, poly_pat_any("x"), NULL), NULL),
       rule_cast_roundtrip_lossless},
      /* CAST(x, bool) -> CMPNE(x, 0) */
      {poly_pat_op1(POLY_OP_CAST, poly_pat_any("x"), NULL), rule_cast_bool},
      /* Pinned symbolic.py:159-163 Threefry pack/unpack identities. */
      {poly_pat_op1(POLY_OP_CAST, poly_pat_any(NULL), NULL), rule_threefry_low_lane_cast},
      {poly_pat_op1(POLY_OP_CAST, poly_pat_any(NULL), NULL), rule_threefry_unpack_low},
      {poly_pat_ops2(
           poly_opset_add(
               poly_opset_add((PolyOpSet){{0, 0}}, POLY_OP_SHR), POLY_OP_FLOORDIV
           ),
           poly_pat_any(NULL), poly_pat_any(NULL), NULL
       ),
       rule_threefry_unpack_high},

      /* -- Double negation -- */
      /* NEG(NEG(x)) -> x */
      {poly_pat_op1(POLY_OP_NEG, poly_pat_op1(POLY_OP_NEG, poly_pat_any("x"), NULL), NULL),
       rule_double_neg},

      /* -- Division identities -- */
      /* Pinned symbolic.py:136-145, with ElementwiseMixin.div's ordinary
       * MUL(x, RECIPROCAL(x)) spelling. */
      {poly_pat_op2c(
           POLY_OP_MUL, poly_pat_any("x"),
           poly_pat_op1(POLY_OP_RECIPROCAL, poly_pat_any("x"), NULL), NULL
       ),
       rule_reciprocal_self_product},
      /* Keep raw FDIV self-cancellation for graphs introduced by an
       * FDIV-capable renderer's late decomposition. */
      {poly_pat_op2(POLY_OP_FDIV, poly_pat_any("x"), poly_pat_any("x"), NULL), rule_fdiv_self},
      /* (x * y) / y -> x */
      {poly_pat_op2(
           POLY_OP_FDIV, poly_pat_op2c(POLY_OP_MUL, poly_pat_any("x"), poly_pat_any("y"), NULL),
           poly_pat_any("y"), NULL
       ),
       rule_mul_fdiv_cancel},

      /* -- Where folding -- */
      /* WHERE(CMPNE(cond, true), t, f) -> WHERE(cond, f, t) */
      {poly_pat_op3(
           POLY_OP_WHERE,
           poly_pat_op2c(
               POLY_OP_CMPNE, poly_pat_dtype("cond", (PolyDType[]){POLY_BOOL}, 1),
               poly_pat_any("trueish"), NULL
           ),
           poly_pat_any("t"), poly_pat_any("f"), NULL
       ),
       rule_where_logical_not},
      /* WHERE(a, WHERE(b, c, d), d) -> WHERE(AND(a, b), c, d) */
      {poly_pat_op3(
           POLY_OP_WHERE, poly_pat_any("a"),
           poly_pat_op3(
               POLY_OP_WHERE, poly_pat_any("b"), poly_pat_any("c"), poly_pat_any("d"), NULL
           ),
           poly_pat_any("d"), NULL
       ),
       rule_nested_where},
      /* WHERE(cond, val, val) -> val */
      {poly_pat_op3(
           POLY_OP_WHERE, poly_pat_any(NULL), poly_pat_any("val"), poly_pat_any("val"), NULL
       ),
       rule_where_same},
      /* WHERE(cond, true, false) -> cond; WHERE(cond, false, true) -> cond.logical_not() */
      {poly_pat_op3(
           POLY_OP_WHERE, poly_pat_dtype("cond", (PolyDType[]){POLY_BOOL}, 1), poly_pat_cvar(NULL),
           poly_pat_cvar(NULL), NULL
       ),
       rule_where_bool_identity},
      /* LOAD/STORE with Invalid index folds away */
      {poly_pat_allow_any_len(poly_pat_op(POLY_OP_LOAD, NULL, 0, "x")),
       rule_fold_invalid_load_store},
      {poly_pat_allow_any_len(poly_pat_op(POLY_OP_STORE, NULL, 0, "x")),
       rule_fold_invalid_load_store},
      /* WHERE(const_gate, c0, c1) -> c0 or c1 */
      {poly_pat_op3(
           POLY_OP_WHERE, poly_pat_cvar("gate"), poly_pat_any("c0"), poly_pat_any("c1"), NULL
       ),
       rule_where_const_gate},

      /* -- vmin/vmax const folding -- */
      {poly_pat_ops(
           poly_opset_add(
               poly_opset_add(
                   poly_opset_add(
                       poly_opset_add(
                           poly_opset_add((PolyOpSet){{0, 0}}, POLY_OP_CMPLT), POLY_OP_CMPNE
                       ),
                       POLY_OP_FLOORDIV
                   ),
                   POLY_OP_FLOORMOD
               ),
               POLY_OP_DEFINE_VAR
           ),
           NULL, 0, "x"
       ),
       rule_const_when_minmax_point},
      {poly_pat_ops(
           poly_opset_add(poly_opset_add((PolyOpSet){{0, 0}}, POLY_OP_BIND), POLY_OP_SPECIAL), NULL,
           0, "x"
       ),
       rule_const_when_minmax_point},
      {poly_pat_op1(POLY_OP_RANGE, poly_pat_op(POLY_OP_CONST, NULL, 0, NULL), "x"),
       rule_const_when_minmax_point},
      {poly_pat_op2(POLY_OP_MAX, poly_pat_any("x"), poly_pat_any("y"), NULL), rule_max_fold},

      /* -- divmod recombine (dynamic): handles all divmod cancel patterns -- */
      /* (base%div)*mul + (base//div)*(div*mul) -> base*mul and variants */
      {poly_pat_op(POLY_OP_ADD, NULL, 0, NULL), rule_fold_add_divmod_recombine},

      /* -- cancel_divmod: weak-index FLOORMOD/FLOORDIV simplification via vmin/vmax -- */
      {poly_pat_ops2(
           poly_opset_add(poly_opset_add((PolyOpSet){{0, 0}}, POLY_OP_FLOORMOD), POLY_OP_FLOORDIV),
           poly_pat_any(NULL), poly_pat_any(NULL), NULL
       ),
       rule_cancel_divmod},
  };

  int n = sizeof(rules) / sizeof(rules[0]);
  g_symbolic_simple = poly_pm_thread_cache(poly_pm_new(rules, n));
  return g_symbolic_simple;
}

PolyPatternMatcher *poly_symbolic(void) {
  if (g_symbolic) return g_symbolic;
  PolyRule rules[] = {
      /* Pinned symbolic.py:297-301 general nested-CAST composition. */
      {poly_pat_op1(POLY_OP_CAST, poly_pat_op1(POLY_OP_CAST, poly_pat_any("x"), NULL), NULL),
       rule_compose_nested_casts},
      {poly_pat_ops2(POLY_GROUP_BINARY, poly_pat_any("x"), poly_pat_any("y"), "u"),
       rule_narrow_proven_long_binary},
      /* Pinned symbolic.py:306-311 generic AFTER effect canonicalization. */
      {poly_pat_op(POLY_OP_AFTER, NULL, 0, "x"), rule_after_canonicalize},
      {poly_pat_ops2(POLY_GROUP_COMMUTATIVE, poly_pat_any(NULL), poly_pat_any(NULL), "x"),
       rule_commutative_index_tuplize_order},
      {poly_pat_op(POLY_OP_ADD, NULL, 0, "add"), rule_add_same_base_const_terms},
      /* Pinned symbolic.py:237-248 keeps term-combination out of
       * symbolic_simple. Late renderer decomposition uses only the simple
       * matcher, while ordinary full symbolic retains these rewrites. */
      {poly_pat_op2c(POLY_OP_ADD, poly_pat_any("x"), poly_pat_any("x"), NULL), rule_add_self},
      {poly_pat_op2c(
           POLY_OP_ADD, poly_pat_op2c(POLY_OP_ADD, poly_pat_any("a"), poly_pat_any("x"), NULL),
           poly_pat_any("x"), NULL
       ),
       rule_add_assoc_self},
      {poly_pat_op1(POLY_OP_CAST, poly_pat_any("x"), "cast"),
       rule_cast_index_add_const_to_add_casts},
      {poly_pat_ops2(POLY_GROUP_ASSOCIATIVE, poly_pat_any(NULL), poly_pat_any(NULL), "alu"),
       rule_assoc_fold_consts},
      {poly_pat_op2(
           POLY_OP_CMPLT,
           poly_pat_op2c(
               POLY_OP_ADD, poly_pat_any("x"),
               poly_pat_op(POLY_OP_CONST, NULL, 0, "c0"), NULL
           ),
           poly_pat_op(POLY_OP_CONST, NULL, 0, "c1"), NULL
       ),
       rule_cmplt_add_const_bound},
      /* Pinned tinygrad/uop/symbolic.py:174-178,290 generic weak-index
       * less-than factor folding. The callback proves every precondition and
       * returns no rewrite for non-positive or non-divisible bounds. */
      {poly_pat_op2(
           POLY_OP_CMPLT, poly_pat_any(NULL),
           poly_pat_op(POLY_OP_CONST, NULL, 0, NULL), NULL
       ),
       rule_lt_folding},
      /* Remaining POW lowers through xpow, matching tinygrad symbolic.py:sym. */
      {poly_pat_op(POLY_OP_POW, NULL, 0, NULL), rule_decomp_pow_generic},
      /* Pinned tinygrad/uop/symbolic.py:478-480. d is exactly
       * reciprocal(1+x); retain the three source spellings and output order. */
      {poly_pat_op2c(
           POLY_OP_MUL, poly_pat_any("x"),
           poly_pat_op1(
               POLY_OP_RECIPROCAL,
               poly_pat_op2c(POLY_OP_ADD, poly_pat_cvar("one"), poly_pat_any("x"), NULL), "d"
           ),
           NULL
       ),
       rule_reciprocal_product_one_minus},
      {poly_pat_op2c(
           POLY_OP_MUL, poly_pat_any("x"),
           poly_pat_op2c(
               POLY_OP_MUL,
               poly_pat_op1(
                   POLY_OP_RECIPROCAL,
                   poly_pat_op2c(POLY_OP_ADD, poly_pat_cvar("one"), poly_pat_any("x"), NULL), "d"
               ),
               poly_pat_any("y"), NULL
           ),
           NULL
       ),
       rule_reciprocal_product_scaled_one_minus},
      {poly_pat_op2c(
           POLY_OP_MUL, poly_pat_any("x"),
           poly_pat_op2c(
               POLY_OP_ADD,
               poly_pat_op1(
                   POLY_OP_RECIPROCAL,
                   poly_pat_op2c(POLY_OP_ADD, poly_pat_cvar("one"), poly_pat_any("x"), NULL), "d"
               ),
               poly_pat_any("y"), NULL
           ),
           NULL
       ),
       rule_reciprocal_product_sum},
      {poly_pat_op(POLY_OP_MUL, NULL, 0, "mul"), rule_distribute_negated_add},
      {poly_pat_op(POLY_OP_MUL, NULL, 0, "mul"), rule_distribute_const_mul_over_add},
      {poly_pat_op(POLY_OP_ADD, NULL, 0, "add"), rule_move_const_to_end},
      {poly_pat_op(POLY_OP_MUL, NULL, 0, "mul"), rule_move_const_to_end},
  };
  PolyPatternMatcher *extra = poly_pm_new(rules, (int)(sizeof(rules) / sizeof(rules[0])));
  PolyPatternMatcher *with_extra = poly_pm_concat(poly_symbolic_simple(), extra);
  poly_pm_destroy(extra); /* concat copies rules; the temporary matcher is no longer needed. */
  g_symbolic = poly_pm_thread_cache(poly_pm_concat(with_extra, poly_pm_gep_pushing()));
  poly_pm_destroy(with_extra);
  return g_symbolic;
}

PolyPatternMatcher *poly_sym(void) {
  if (g_sym) return g_sym;
  PolyRule rules[] = {
      {poly_pat_op(POLY_OP_REDUCE, NULL, 0, "reduce"), rule_sym_reduce_mul_chain},
      {poly_pat_op(POLY_OP_MUL, NULL, 0, "mul"), rule_sym_distribute_weak_mul_over_add},
  };
  PolyPatternMatcher *extra = poly_pm_new(rules, (int)(sizeof(rules) / sizeof(rules[0])));
  PolyPatternMatcher *with_valid = poly_pm_concat(poly_symbolic(), poly_pm_simplify_valid());
  g_sym = poly_pm_thread_cache(poly_pm_concat(with_valid, extra));
  poly_pm_destroy(with_valid);
  poly_pm_destroy(extra);
  return g_sym;
}
