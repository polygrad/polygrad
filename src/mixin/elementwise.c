/*
 * mixin/elementwise.c -- ElementwiseMixin UOp composition
 *
 * Mirrors tinygrad/mixin/elementwise.py. Tensor dual-root construction calls
 * these same promotion rules independently for logical and physical roots.
 */

#include "mixin/elementwise.h"
#include "bigint.h"
#include <math.h>
#include <limits.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_LN2
#define M_LN2 0.693147180559945309417
#endif

/* C helper for current ElementwiseMixin._broadcasted.promote. It is public
 * inside the core because Polygrad applies one promotion to each retained
 * logical/physical root (mixin/elementwise.py:21-29). */
PolyUOp *poly_elementwise_promote(PolyCtx *ctx, PolyUOp *root, PolyDType common) {
  if (!ctx || !root) return NULL;
  PolyUOp *base = poly_uop_base(root);
  /* Invalid is a sentinel, not a bool value to cast to the common dtype. */
  if (base->op == POLY_OP_CONST && base->arg.kind == POLY_ARG_INVALID) return root;
  if (poly_dtype_is_weak(root->dtype) && base->op == POLY_OP_CONST) {
    return poly_const_like_dtype(ctx, root, base->arg, poly_dtype_weak(common));
  }
  return poly_dtype_eq(root->dtype, common) ? root : poly_cast(ctx, root, common);
}

/* Current ElementwiseMixin._broadcasted applies least_upper_dtype while
 * keeping bare weak CONSTs weak (mixin/elementwise.py:21-29). */
bool poly_broadcasted_pair(PolyCtx *ctx, PolyUOp **a, PolyUOp **b) {
  if (!ctx || !a || !b || !*a || !*b) return false;

  PolyDType common;
  if (!poly_dtype_least_upper((*a)->dtype, (*b)->dtype, &common)) return false;
  *a = poly_elementwise_promote(ctx, *a, common);
  *b = poly_elementwise_promote(ctx, *b, common);
  return *a != NULL && *b != NULL;
}

/* Current ElementwiseMixin._binop promotes once, then constructs the selected
 * ALU UOp (mixin/elementwise.py:32-34). */
PolyUOp *poly_binop(PolyCtx *ctx, PolyOps op, PolyUOp *a, PolyUOp *b) {
  if (!poly_broadcasted_pair(ctx, &a, &b)) return NULL;
  return poly_alu2(ctx, op, a, b);
}

/* Broadcasting binary ops */

PolyUOp *poly_add(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  if (!poly_broadcasted_pair(ctx, &a, &b)) return NULL;
  return poly_alu2(ctx, POLY_OP_ADD, a, b);
}

PolyUOp *poly_sub(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  if (!poly_broadcasted_pair(ctx, &a, &b)) return NULL;

  /* Current ElementwiseMixin.sub is a.alu(ADD, -b) after the one
   * _broadcasted promotion pass. Negation keeps its scalar -1 weak and UOp
   * shape inference owns any shape broadcast (mixin/elementwise.py:104-119). */
  PolyUOp *neg_b = NULL;
  if (poly_dtype_is_bool(b->dtype)) {
    neg_b = poly_logical_not(ctx, b);
  } else {
    PolyUOp *minus_one = poly_const_typed(ctx, poly_dtype_weak(b->dtype), -1.0);
    neg_b = minus_one ? poly_alu2(ctx, POLY_OP_MUL, b, minus_one) : NULL;
  }
  return neg_b ? poly_alu2(ctx, POLY_OP_ADD, a, neg_b) : NULL;
}

PolyUOp *poly_mul(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  if (!poly_broadcasted_pair(ctx, &a, &b)) return NULL;
  return poly_alu2(ctx, POLY_OP_MUL, a, b);
}

/* Comparisons (broadcasting) */

/* Logical NOT — `CMPNE(x, CONST(true))` for
 * bool inputs, matching tinygrad's `logical_not()` after CAST elision
 * (mixin/elementwise.py:39-47 + symbolic.py:93-131). Raw `NEG(bool)` retains
 * arithmetic NEG semantics and is not a second logical-NOT spelling. */
PolyUOp *poly_logical_not(PolyCtx *ctx, PolyUOp *x) {
  PolyUOp *t = poly_const_typed(ctx, POLY_BOOL, 1);
  return poly_alu2(ctx, POLY_OP_CMPNE, x, t);
}

/* All comparison helpers return BOOL, mirroring tinygrad's
 * mixin/elementwise.py:218-247:
 *   eq(a,b) = (a != b).logical_not()
 *   ne(a,b) = CMPNE(a,b)
 *   gt(a,b) = CMPLT(b,a)         (operand swap)
 *   lt(a,b) = CMPLT(a,b)
 *   ge(a,b) = (a < b).logical_not()
 *   le(a,b) = (a > b).logical_not() = (b < a).logical_not()
 *
 * Polygrad previously had ge/le returning float WHERE(0,1); fixed in P5
 * for tinygrad parity and to let Phase D's reduce_collapse Rule 4 match
 * polygrad's tril/triu masks. */
PolyUOp *poly_eq(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  /* Pinned Tensor.eq reaches _binop/_broadcasted, which broadcasts and then
   * promotes both operands with least_upper_dtype before CMPNE
   * (mixin/elementwise.py:324-325, mixin/__init__.py:439-449). */
  if (!poly_broadcasted_pair(ctx, &a, &b)) return NULL;
  PolyUOp *ne = poly_alu2(ctx, POLY_OP_CMPNE, a, b);
  return poly_logical_not(ctx, ne);
}

PolyUOp *poly_where_op(PolyCtx *ctx, PolyUOp *cond, PolyUOp *x, PolyUOp *y) {
  if (!cond || !poly_broadcasted_pair(ctx, &x, &y)) return NULL;
  /* tinygrad Tensor.where casts non-bool conditions to bool before building
   * Ops.WHERE. Keeping that in the core constructor preserves the expected
   * CMPNE(cond, 0) node in helper graphs such as nonzero-value padding. */
  if (!poly_dtype_is_bool(cond->dtype)) cond = poly_cast(ctx, cond, POLY_BOOL);
  return poly_alu3(ctx, POLY_OP_WHERE, cond, x, y);
}

/* Helper: float constant matching the dtype of a given UOp.
 * For float inputs: creates a constant with the same float dtype.
 * For non-float inputs (comparisons producing bool): defaults to float32. */
PolyUOp *poly_elementwise_float_const(PolyCtx *ctx, PolyUOp *ref, double v) {
  PolyDType dt = ref->dtype;
  if (poly_dtype_is_float(dt)) return poly_const_typed(ctx, dt, v);
  return poly_const_float(ctx, v);
}

/* Helper: const with explicit dtype -- use in special-math ops for dtype correctness */
static inline PolyUOp *cdt(PolyCtx *ctx, PolyDType dt, double v) {
  return poly_const_typed(ctx, dt, v);
}

PolyUOp *poly_const_exact_int(PolyCtx *ctx, PolyDType dt, int64_t value) {
  dt = dt;
  if (poly_dtype_is_bool(dt)) return poly_uop0(ctx, POLY_OP_CONST, dt, poly_arg_bool(value != 0));
  if (!poly_dtype_is_int(dt)) return NULL;
  return poly_uop0(ctx, POLY_OP_CONST, dt, poly_arg_int(value));
}

PolyUOp *poly_const_exact_float(PolyCtx *ctx, PolyDType dt, double value) {
  dt = dt;
  if (!poly_dtype_is_float(dt)) return NULL;
  return poly_uop0(ctx, POLY_OP_CONST, dt, poly_arg_float(value));
}

bool poly_dtype_bound_const(PolyCtx *ctx, PolyDType dt, bool use_min, PolyUOp **out) {
  if (!ctx || !out) return false;
  dt = dt;
  if (poly_dtype_is_float(dt)) {
    *out = poly_const_exact_float(ctx, dt, use_min ? -INFINITY : INFINITY);
    return *out != NULL;
  }
  if (poly_dtype_is_bool(dt)) {
    *out = poly_const_exact_int(ctx, dt, use_min ? 0 : 1);
    return *out != NULL;
  }
  if (!poly_dtype_is_int(dt)) return false;
  if (poly_dtype_is_unsigned(dt)) {
    if (use_min) {
      *out = poly_const_exact_int(ctx, dt, 0);
      return *out != NULL;
    }
    /* Pinned DType.max is the positive Python integer 2**bits-1
     * (dtype.py:84-100). Preserve that exact UOp arg; fixed-width renderers
     * apply the dtype bits only at their backend boundary. */
    PolyInt neg_one = {0}, maximum = {0};
    bool ok =
        poly_int_from_i64(&neg_one, -1) && poly_int_truncate(&maximum, &neg_one, dt.bitsize, true);
    if (ok) *out = poly_uop0(ctx, POLY_OP_CONST, dt, poly_int_as_arg(&maximum));
    poly_int_free(&maximum);
    poly_int_free(&neg_one);
    return ok && *out != NULL;
  }

  int64_t v = 0;
  if (dt.bitsize >= 64)
    v = use_min ? INT64_MIN : INT64_MAX;
  else {
    int bits = (int)dt.bitsize;
    v = use_min ? -(1LL << (bits - 1)) : ((1LL << (bits - 1)) - 1);
  }
  *out = poly_const_exact_int(ctx, dt, v);
  return *out != NULL;
}

/* erf tau helper */

/* A&S 7.1.26: tau(|x|) = t * P(t) * exp(-x^2) where t = 1/(1+p*|x|).
 * erf(x) = sign(x) * (1 - tau(|x|)).
 * erfc(x) = tau(x) for x >= 0, 2 - tau(|x|) for x < 0.
 * Computing tau directly avoids the 1-erf(x) cancellation in erfc. */
static PolyUOp *erf_tau(PolyCtx *ctx, PolyUOp *ax, PolyDType dt) {
  PolyUOp *t = poly_alu1(
      ctx, POLY_OP_RECIPROCAL,
      poly_alu2(
          ctx, POLY_OP_ADD, cdt(ctx, dt, 1.0),
          poly_alu2(ctx, POLY_OP_MUL, cdt(ctx, dt, 0.3275911), ax)
      )
  );
  PolyUOp *p = cdt(ctx, dt, 1.061405429);
  p = poly_alu2(ctx, POLY_OP_ADD, cdt(ctx, dt, -1.453152027), poly_alu2(ctx, POLY_OP_MUL, t, p));
  p = poly_alu2(ctx, POLY_OP_ADD, cdt(ctx, dt, 1.421413741), poly_alu2(ctx, POLY_OP_MUL, t, p));
  p = poly_alu2(ctx, POLY_OP_ADD, cdt(ctx, dt, -0.284496736), poly_alu2(ctx, POLY_OP_MUL, t, p));
  p = poly_alu2(ctx, POLY_OP_ADD, cdt(ctx, dt, 0.254829592), poly_alu2(ctx, POLY_OP_MUL, t, p));
  PolyUOp *x2 = poly_alu2(ctx, POLY_OP_MUL, ax, ax);
  PolyUOp *e = poly_exp(ctx, poly_alu1(ctx, POLY_OP_NEG, x2));
  return poly_alu2(ctx, POLY_OP_MUL, t, poly_alu2(ctx, POLY_OP_MUL, p, e));
}

/* lgamma Lanczos helper */

static PolyUOp *poly_lgamma_forward_lanczos(PolyCtx *ctx, PolyUOp *x, PolyDType dt) {
  /* Lanczos approximation with reflection. */
  const double g = 7.0;
  const double c0 = 0.99999999999980993;
  const double c[8] = {676.5203681218851,     -1259.1392167224028,  771.32342877765313,
                       -176.61502916214059,   12.507343278686905,   -0.13857109526572012,
                       9.9843695780195716e-6, 1.5056327351493116e-7};

  PolyUOp *xm1 = poly_alu2(ctx, POLY_OP_SUB, x, cdt(ctx, dt, 1.0));
  PolyUOp *a = cdt(ctx, dt, c0);
  for (int i = 0; i < 8; i++) {
    PolyUOp *den = poly_alu2(ctx, POLY_OP_ADD, xm1, cdt(ctx, dt, (double)(i + 1)));
    a = poly_alu2(ctx, POLY_OP_ADD, a, poly_alu2(ctx, POLY_OP_FDIV, cdt(ctx, dt, c[i]), den));
  }
  PolyUOp *t = poly_alu2(ctx, POLY_OP_ADD, xm1, cdt(ctx, dt, g + 0.5));
  PolyUOp *lg_pos = poly_alu2(
      ctx, POLY_OP_ADD, cdt(ctx, dt, 0.91893853320467274178),
      poly_alu2(
          ctx, POLY_OP_ADD,
          poly_alu2(
              ctx, POLY_OP_MUL, poly_alu2(ctx, POLY_OP_ADD, xm1, cdt(ctx, dt, 0.5)),
              poly_log(ctx, t)
          ),
          poly_alu2(ctx, POLY_OP_SUB, poly_log(ctx, a), t)
      )
  );

  PolyUOp *one_minus_x = poly_alu2(ctx, POLY_OP_SUB, cdt(ctx, dt, 1.0), x);
  PolyUOp *xm1r = poly_alu2(ctx, POLY_OP_SUB, one_minus_x, cdt(ctx, dt, 1.0));
  PolyUOp *ar = cdt(ctx, dt, c0);
  for (int i = 0; i < 8; i++) {
    PolyUOp *den = poly_alu2(ctx, POLY_OP_ADD, xm1r, cdt(ctx, dt, (double)(i + 1)));
    ar = poly_alu2(ctx, POLY_OP_ADD, ar, poly_alu2(ctx, POLY_OP_FDIV, cdt(ctx, dt, c[i]), den));
  }
  PolyUOp *tr = poly_alu2(ctx, POLY_OP_ADD, xm1r, cdt(ctx, dt, g + 0.5));
  PolyUOp *lg_ref_base = poly_alu2(
      ctx, POLY_OP_ADD, cdt(ctx, dt, 0.91893853320467274178),
      poly_alu2(
          ctx, POLY_OP_ADD,
          poly_alu2(
              ctx, POLY_OP_MUL, poly_alu2(ctx, POLY_OP_ADD, xm1r, cdt(ctx, dt, 0.5)),
              poly_log(ctx, tr)
          ),
          poly_alu2(ctx, POLY_OP_SUB, poly_log(ctx, ar), tr)
      )
  );
  PolyUOp *sinpix = poly_sin(ctx, poly_alu2(ctx, POLY_OP_MUL, cdt(ctx, dt, M_PI), x));
  PolyUOp *lg_ref = poly_alu2(
      ctx, POLY_OP_SUB,
      poly_alu2(ctx, POLY_OP_SUB, cdt(ctx, dt, log(M_PI)), poly_log(ctx, poly_abs(ctx, sinpix))),
      lg_ref_base
  );
  PolyUOp *cond = poly_alu2(ctx, POLY_OP_CMPLT, x, cdt(ctx, dt, 0.5));
  return poly_alu3(ctx, POLY_OP_WHERE, cond, lg_ref, lg_pos);
}

PolyUOp *poly_div(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  if (!poly_broadcasted_pair(ctx, &a, &b)) return NULL;
  /* Current ElementwiseMixin.div keeps true division at tensor/UOp stage as
   * MUL(a, RECIPROCAL(b)); FDIV is introduced only by backend decomposition
   * when supported (mixin/elementwise.py:225-252,
   * codegen/decomp/op.py:133-136). */
  if (poly_dtype_is_int(a->dtype) || poly_dtype_is_bool(a->dtype))
    a = poly_cast(ctx, a, poly_dtype_strong(POLY_WEAKFLOAT));
  PolyUOp *reciprocal = b ? poly_alu1(ctx, POLY_OP_RECIPROCAL, b) : NULL;
  return a && reciprocal ? poly_alu2(ctx, POLY_OP_MUL, a, reciprocal) : NULL;
}
/* Current UOp.ufix creates UOp.const(x), whose Python int/float dtype is weak.
 * _broadcasted performs the later promotion; it does not shape the scalar
 * (uop/ops.py:587-590, mixin/elementwise.py:19-29). */
static PolyUOp *poly_ufix_const(
    PolyCtx *ctx,
    PolyUOp *self,
    PolyDType from_py_dtype,
    double value
) {
  if (!ctx || !self) return NULL;
  return poly_const_typed(ctx, poly_dtype_weak(from_py_dtype), value);
}

PolyUOp *poly_elementwise_scalar_binop(
    PolyCtx *ctx,
    PolyOps op,
    PolyUOp *self,
    PolyDType from_py_dtype,
    double value,
    bool reverse
) {
  PolyUOp *scalar = poly_ufix_const(ctx, self, from_py_dtype, value);
  if (!scalar) return NULL;
  if (op == POLY_OP_SUB) return reverse ? poly_sub(ctx, scalar, self) : poly_sub(ctx, self, scalar);
  if (op == POLY_OP_FDIV)
    return reverse ? poly_div(ctx, scalar, self) : poly_div(ctx, self, scalar);
  return reverse ? poly_binop(ctx, op, scalar, self) : poly_binop(ctx, op, self, scalar);
}

/* Contiguous (realize barrier) */

PolyUOp *poly_contiguous(PolyCtx *ctx, PolyUOp *x) {
  if (!ctx || !x) return NULL;
  /* Tinygrad 2026-08-22/a9069c177a9d mixin/elementwise.py:55-61: weak and
   * device-free values have no storage materialization to request. */
  if (poly_dtype_is_weak(x->dtype)) return x;
  if (x->op == POLY_OP_CONTIGUOUS) return x;
  if (poly_uop_device(x) == POLY_DEVICE_AUTO) return x;
  if (poly_uop_has_buffer_identity(x)) return x;
  return poly_uop1(ctx, POLY_OP_CONTIGUOUS, x->dtype, x, poly_arg_none());
}

/* Math */

PolyUOp *poly_exp(PolyCtx *ctx, PolyUOp *x) {
  if (!ctx || !x) return NULL;
  /* Pinned exp promotes floating inputs to at least float32 and casts the
   * result back; non-floats stay at least default-float
   * (mixin/elementwise.py:491-503). */
  PolyDType input_dt;
  if (!poly_dtype_least_upper_float(x->dtype, &input_dt)) return NULL;
  x = poly_dtype_eq(x->dtype, input_dt) ? x : poly_cast(ctx, x, input_dt);
  if (!x) return NULL;
  PolyDType compute_dt;
  if (!poly_dtype_least_upper(input_dt, POLY_FLOAT32, &compute_dt)) return NULL;
  PolyUOp *compute_x = poly_dtype_eq(input_dt, compute_dt) ? x : poly_cast(ctx, x, compute_dt);
  PolyUOp *scaled = compute_x ? poly_elementwise_scalar_binop(
                                    ctx, POLY_OP_MUL, compute_x, POLY_FLOAT32, 1.0 / M_LN2, false
                                )
                              : NULL;
  PolyUOp *out = scaled ? poly_alu1(ctx, POLY_OP_EXP2, scaled) : NULL;
  if (out && poly_dtype_is_float(input_dt) && !poly_dtype_eq(input_dt, compute_dt))
    out = poly_cast(ctx, out, input_dt);
  return out;
}

PolyUOp *poly_log(PolyCtx *ctx, PolyUOp *x) {
  if (!ctx || !x) return NULL;
  /* Current log is exactly self.log2()*log(2). LOG2 owns the floating result
   * dtype and retains the original operand (mixin/elementwise.py:827-839,
   * uop/ops.py:144-145). */
  PolyUOp *log2_x = poly_alu1(ctx, POLY_OP_LOG2, x);
  return log2_x
             ? poly_elementwise_scalar_binop(ctx, POLY_OP_MUL, log2_x, POLY_FLOAT32, M_LN2, false)
             : NULL;
}

PolyUOp *poly_log1p(PolyCtx *ctx, PolyUOp *x) {
  /* Stable near zero: log(1+x) ~ x - x^2/2 + x^3/3 */
  PolyDType dt = x->dtype;
  PolyUOp *ax = poly_abs(ctx, x);
  PolyUOp *small = poly_alu2(ctx, POLY_OP_CMPLT, ax, cdt(ctx, dt, 1e-4));
  PolyUOp *x2 = poly_alu2(ctx, POLY_OP_MUL, x, x);
  PolyUOp *x3 = poly_alu2(ctx, POLY_OP_MUL, x2, x);
  PolyUOp *poly = poly_alu2(
      ctx, POLY_OP_ADD, x,
      poly_alu2(
          ctx, POLY_OP_ADD, poly_alu2(ctx, POLY_OP_MUL, cdt(ctx, dt, -0.5), x2),
          poly_alu2(ctx, POLY_OP_MUL, cdt(ctx, dt, 1.0 / 3.0), x3)
      )
  );
  PolyUOp *direct = poly_log(ctx, poly_alu2(ctx, POLY_OP_ADD, cdt(ctx, dt, 1.0), x));
  return poly_alu3(ctx, POLY_OP_WHERE, small, poly, direct);
}

PolyUOp *poly_expm1(PolyCtx *ctx, PolyUOp *x) {
  /* Stable near zero: expm1(x) ~ x + x^2/2 + x^3/6 */
  PolyDType dt = x->dtype;
  PolyUOp *ax = poly_abs(ctx, x);
  PolyUOp *small = poly_alu2(ctx, POLY_OP_CMPLT, ax, cdt(ctx, dt, 1e-4));
  PolyUOp *x2 = poly_alu2(ctx, POLY_OP_MUL, x, x);
  PolyUOp *x3 = poly_alu2(ctx, POLY_OP_MUL, x2, x);
  PolyUOp *poly = poly_alu2(
      ctx, POLY_OP_ADD, x,
      poly_alu2(
          ctx, POLY_OP_ADD, poly_alu2(ctx, POLY_OP_MUL, cdt(ctx, dt, 0.5), x2),
          poly_alu2(ctx, POLY_OP_MUL, cdt(ctx, dt, 1.0 / 6.0), x3)
      )
  );
  PolyUOp *direct = poly_alu2(ctx, POLY_OP_SUB, poly_exp(ctx, x), cdt(ctx, dt, 1.0));
  return poly_alu3(ctx, POLY_OP_WHERE, small, poly, direct);
}

PolyUOp *poly_sin(PolyCtx *ctx, PolyUOp *x) {
  return poly_alu1(ctx, POLY_OP_SIN, x);
}

PolyUOp *poly_cos(PolyCtx *ctx, PolyUOp *x) {
  if (!ctx || !x) return NULL;
  /* Current cos first casts with least_upper_float, computes pi/2-x at at
   * least float32, applies SIN, then casts back to the first dtype
   * (mixin/elementwise.py:480-489). */
  PolyDType result_dtype;
  if (!poly_dtype_least_upper_float(x->dtype, &result_dtype)) return NULL;
  PolyUOp *self = poly_dtype_eq(x->dtype, result_dtype) ? x : poly_cast(ctx, x, result_dtype);
  if (!self) return NULL;

  PolyDType compute_dtype;
  if (!poly_dtype_least_upper(self->dtype, POLY_FLOAT32, &compute_dtype)) return NULL;
  PolyUOp *work =
      poly_dtype_eq(self->dtype, compute_dtype) ? self : poly_cast(ctx, self, compute_dtype);
  PolyUOp *angle =
      work ? poly_elementwise_scalar_binop(ctx, POLY_OP_SUB, work, POLY_FLOAT32, M_PI / 2.0, true)
           : NULL;
  PolyUOp *out = angle ? poly_alu1(ctx, POLY_OP_SIN, angle) : NULL;
  return out && !poly_dtype_eq(out->dtype, result_dtype) ? poly_cast(ctx, out, result_dtype) : out;
}

PolyUOp *poly_tan(PolyCtx *ctx, PolyUOp *x) {
  /* Current tan is the ordinary high-level quotient self.sin()/self.cos()
   * (mixin/elementwise.py:917-927). */
  return poly_div(ctx, poly_sin(ctx, x), poly_cos(ctx, x));
}

/* Pinned ElementwiseMixin compositions. Keep scalar weakness and source order:
 * these are Tensor-level graphs, not algebraically equivalent backend rewrites. */
PolyUOp *poly_elementwise_neg(PolyCtx *ctx, PolyUOp *x) {
  if (!ctx || !x) return NULL;
  return poly_dtype_is_bool(x->dtype)
             ? poly_logical_not(ctx, x)
             : poly_elementwise_scalar_binop(ctx, POLY_OP_MUL, x, POLY_INT32, -1, false);
}

PolyUOp *poly_log10(PolyCtx *ctx, PolyUOp *x) {
  return poly_elementwise_scalar_binop(
      ctx, POLY_OP_MUL, poly_alu1(ctx, POLY_OP_LOG2, x), POLY_FLOAT32, log10(2.0), false
  );
}

PolyUOp *poly_atanh(PolyCtx *ctx, PolyUOp *x) {
  PolyUOp *a = poly_elementwise_scalar_binop(ctx, POLY_OP_ADD, x, POLY_INT32, 1, true);
  PolyUOp *b = poly_elementwise_scalar_binop(ctx, POLY_OP_SUB, x, POLY_INT32, 1, true);
  return poly_elementwise_scalar_binop(
      ctx, POLY_OP_FDIV, poly_log(ctx, poly_div(ctx, a, b)), POLY_INT32, 2, false
  );
}

PolyUOp *poly_asinh(PolyCtx *ctx, PolyUOp *x) {
  PolyUOp *a =
      poly_elementwise_scalar_binop(ctx, POLY_OP_ADD, poly_square(ctx, x), POLY_INT32, 1, false);
  return poly_log(ctx, poly_add(ctx, x, poly_alu1(ctx, POLY_OP_SQRT, a)));
}

PolyUOp *poly_acosh(PolyCtx *ctx, PolyUOp *x) {
  PolyUOp *a =
      poly_elementwise_scalar_binop(ctx, POLY_OP_SUB, poly_square(ctx, x), POLY_INT32, 1, false);
  return poly_log(ctx, poly_add(ctx, x, poly_alu1(ctx, POLY_OP_SQRT, a)));
}

/* helpers.polyN starts at 0.0, including the first multiply/add in the graph. */
static PolyUOp *pointwise_polyn(PolyCtx *ctx, PolyUOp *x, const double *coefficients, int n) {
  PolyUOp *p = poly_const_exact_float(ctx, POLY_WEAKFLOAT, 0.0);
  for (int i = 0; p && i < n; i++)
    p = poly_elementwise_scalar_binop(
        ctx, POLY_OP_ADD, poly_mul(ctx, p, x), POLY_FLOAT32, coefficients[i], false
    );
  return p;
}

PolyUOp *poly_asin(PolyCtx *ctx, PolyUOp *x) {
  const double coefficients[] = {-0.0012624911, 0.0066700901, -0.0170881256, 0.0308918810,
                                 -0.0501743046, 0.0889789874, -0.2145988016, 1.5707963050};
  PolyUOp *ax = poly_abs(ctx, x);
  PolyUOp *root = poly_alu1(
      ctx, POLY_OP_SQRT,
      poly_elementwise_scalar_binop(ctx, POLY_OP_SUB, ax, POLY_FLOAT32, 1.0, true)
  );
  PolyUOp *p = poly_mul(ctx, root, pointwise_polyn(ctx, ax, coefficients, 8));
  return poly_mul(
      ctx, poly_sign(ctx, x),
      poly_elementwise_scalar_binop(ctx, POLY_OP_SUB, p, POLY_FLOAT32, M_PI / 2, true)
  );
}

PolyUOp *poly_acos(PolyCtx *ctx, PolyUOp *x) {
  return poly_elementwise_scalar_binop(
      ctx, POLY_OP_SUB, poly_asin(ctx, x), POLY_FLOAT32, M_PI / 2, true
  );
}

PolyUOp *poly_atan(PolyCtx *ctx, PolyUOp *x) {
  PolyUOp *a =
      poly_elementwise_scalar_binop(ctx, POLY_OP_ADD, poly_mul(ctx, x, x), POLY_INT32, 1, true);
  return poly_asin(ctx, poly_div(ctx, x, poly_alu1(ctx, POLY_OP_SQRT, a)));
}

PolyUOp *poly_celu(PolyCtx *ctx, PolyUOp *x, PolyUOp *alpha) {
  PolyUOp *negative = poly_mul(
      ctx, alpha,
      poly_elementwise_scalar_binop(
          ctx, POLY_OP_SUB, poly_exp(ctx, poly_div(ctx, x, alpha)), POLY_INT32, 1, false
      )
  );
  PolyUOp *zero = poly_const_exact_int(ctx, POLY_WEAKINT, 0);
  return poly_add(ctx, poly_maximum(ctx, x, zero), poly_minimum(ctx, negative, zero));
}

PolyUOp *poly_selu(PolyCtx *ctx, PolyUOp *x, PolyUOp *alpha, PolyUOp *gamma) {
  PolyUOp *zero = poly_const_exact_int(ctx, POLY_WEAKINT, 0);
  PolyUOp *negative = poly_mul(
      ctx, alpha,
      poly_elementwise_scalar_binop(ctx, POLY_OP_SUB, poly_exp(ctx, x), POLY_INT32, 1, false)
  );
  return poly_mul(ctx, gamma, poly_where_op(ctx, poly_ge(ctx, x, zero), x, negative));
}

PolyUOp *poly_sinh(PolyCtx *ctx, PolyUOp *x) {
  PolyUOp *a = poly_sub(ctx, poly_exp(ctx, x), poly_exp(ctx, poly_elementwise_neg(ctx, x)));
  return poly_elementwise_scalar_binop(ctx, POLY_OP_FDIV, a, POLY_INT32, 2, false);
}

PolyUOp *poly_cosh(PolyCtx *ctx, PolyUOp *x) {
  PolyUOp *a = poly_add(ctx, poly_exp(ctx, x), poly_exp(ctx, poly_elementwise_neg(ctx, x)));
  return poly_elementwise_scalar_binop(ctx, POLY_OP_FDIV, a, POLY_INT32, 2, false);
}

PolyUOp *poly_softsign(PolyCtx *ctx, PolyUOp *x) {
  return poly_div(
      ctx, x, poly_elementwise_scalar_binop(ctx, POLY_OP_ADD, poly_abs(ctx, x), POLY_INT32, 1, true)
  );
}

PolyUOp *poly_erf(PolyCtx *ctx, PolyUOp *x) {
  const double coefficients[] = {1.061405429, -1.453152027, 1.421413741, -0.284496736, 0.254829592};
  PolyUOp *a = poly_elementwise_scalar_binop(
      ctx, POLY_OP_MUL, poly_abs(ctx, x), POLY_FLOAT32, 0.3275911, true
  );
  PolyUOp *t = poly_elementwise_scalar_binop(
      ctx, POLY_OP_FDIV,
      poly_elementwise_scalar_binop(ctx, POLY_OP_ADD, a, POLY_FLOAT32, 1.0, true), POLY_FLOAT32,
      1.0, true
  );
  PolyUOp *p = poly_mul(ctx, t, pointwise_polyn(ctx, t, coefficients, 5));
  PolyUOp *tail = poly_mul(ctx, p, poly_exp(ctx, poly_elementwise_neg(ctx, poly_square(ctx, x))));
  return poly_mul(
      ctx, poly_sign(ctx, x),
      poly_elementwise_scalar_binop(ctx, POLY_OP_SUB, tail, POLY_FLOAT32, 1.0, true)
  );
}

PolyUOp *poly_erfc(PolyCtx *ctx, PolyUOp *x) {
  /* erfc(x) = tau(|x|) for x >= 0, 2 - tau(|x|) for x < 0.
   * No 1-erf(x) cancellation -- tau is computed directly. */
  PolyDType dt = x->dtype;
  PolyUOp *ax = poly_abs(ctx, x);
  PolyUOp *tau = erf_tau(ctx, ax, dt);
  PolyUOp *neg = poly_alu2(ctx, POLY_OP_CMPLT, x, cdt(ctx, dt, 0.0));
  PolyUOp *erfc_neg = poly_alu2(ctx, POLY_OP_SUB, cdt(ctx, dt, 2.0), tau);
  return poly_alu3(ctx, POLY_OP_WHERE, neg, erfc_neg, tau);
}

PolyUOp *poly_erfinv(PolyCtx *ctx, PolyUOp *x) {
  /* Winitzki approximation (a=0.147). */
  PolyDType dt = x->dtype;
  PolyUOp *a = cdt(ctx, dt, 0.147);
  PolyUOp *one = cdt(ctx, dt, 1.0);
  PolyUOp *x2 = poly_alu2(ctx, POLY_OP_MUL, x, x);
  PolyUOp *ln = poly_log(ctx, poly_alu2(ctx, POLY_OP_SUB, one, x2));
  PolyUOp *term1 = poly_alu2(
      ctx, POLY_OP_ADD, poly_alu2(ctx, POLY_OP_FDIV, cdt(ctx, dt, 2.0 / (M_PI * 0.147)), one),
      poly_alu2(ctx, POLY_OP_MUL, cdt(ctx, dt, 0.5), ln)
  );
  PolyUOp *term2 = poly_alu2(ctx, POLY_OP_FDIV, ln, a);
  PolyUOp *inside = poly_alu2(ctx, POLY_OP_SUB, poly_alu2(ctx, POLY_OP_MUL, term1, term1), term2);
  PolyUOp *root = poly_alu1(
      ctx, POLY_OP_SQRT, poly_alu2(ctx, POLY_OP_SUB, poly_alu1(ctx, POLY_OP_SQRT, inside), term1)
  );
  return poly_alu2(ctx, POLY_OP_MUL, poly_sign(ctx, x), root);
}

PolyUOp *poly_ndtri(PolyCtx *ctx, PolyUOp *x) {
  /* ndtri(p) = sqrt(2) * erfinv(2p-1) */
  PolyDType dt = x->dtype;
  PolyUOp *arg = poly_alu2(
      ctx, POLY_OP_SUB, poly_alu2(ctx, POLY_OP_MUL, cdt(ctx, dt, 2.0), x), cdt(ctx, dt, 1.0)
  );
  return poly_alu2(ctx, POLY_OP_MUL, cdt(ctx, dt, sqrt(2.0)), poly_erfinv(ctx, arg));
}

PolyUOp *poly_digamma(PolyCtx *ctx, PolyUOp *x) {
  /* First-order asymptotic with recurrence to x>=6. */
  PolyDType dt = x->dtype;
  PolyUOp *acc = cdt(ctx, dt, 0.0);
  PolyUOp *xx = x;
  for (int i = 0; i < 6; i++) {
    PolyUOp *cond = poly_alu2(ctx, POLY_OP_CMPLT, xx, cdt(ctx, dt, 6.0));
    PolyUOp *inv = poly_alu1(ctx, POLY_OP_RECIPROCAL, xx);
    acc = poly_alu3(ctx, POLY_OP_WHERE, cond, poly_alu2(ctx, POLY_OP_SUB, acc, inv), acc);
    xx =
        poly_alu3(ctx, POLY_OP_WHERE, cond, poly_alu2(ctx, POLY_OP_ADD, xx, cdt(ctx, dt, 1.0)), xx);
  }
  PolyUOp *inv = poly_alu1(ctx, POLY_OP_RECIPROCAL, xx);
  PolyUOp *inv2 = poly_alu2(ctx, POLY_OP_MUL, inv, inv);
  PolyUOp *inv4 = poly_alu2(ctx, POLY_OP_MUL, inv2, inv2);
  PolyUOp *inv6 = poly_alu2(ctx, POLY_OP_MUL, inv4, inv2);
  PolyUOp *asym = poly_alu2(
      ctx, POLY_OP_ADD, poly_log(ctx, xx),
      poly_alu2(
          ctx, POLY_OP_ADD, poly_alu2(ctx, POLY_OP_MUL, cdt(ctx, dt, -0.5), inv),
          poly_alu2(
              ctx, POLY_OP_ADD, poly_alu2(ctx, POLY_OP_MUL, cdt(ctx, dt, -1.0 / 12.0), inv2),
              poly_alu2(
                  ctx, POLY_OP_ADD, poly_alu2(ctx, POLY_OP_MUL, cdt(ctx, dt, 1.0 / 120.0), inv4),
                  poly_alu2(ctx, POLY_OP_MUL, cdt(ctx, dt, -1.0 / 252.0), inv6)
              )
          )
      )
  );
  return poly_alu2(ctx, POLY_OP_ADD, acc, asym);
}

PolyUOp *poly_lgamma(PolyCtx *ctx, PolyUOp *x) {
  /* Explicit VJP override:
   * y = detach(f(x)) + (x - detach(x))*digamma(x) */
  PolyDType dt = x->dtype;
  PolyUOp *fwd = poly_lgamma_forward_lanczos(ctx, x, dt);
  PolyUOp *dx = poly_uop1(ctx, POLY_OP_DETACH, x->dtype, x, poly_arg_none());
  PolyUOp *df = poly_uop1(ctx, POLY_OP_DETACH, fwd->dtype, fwd, poly_arg_none());
  PolyUOp *delta = poly_alu2(ctx, POLY_OP_SUB, x, dx);
  PolyUOp *forced = poly_alu2(ctx, POLY_OP_MUL, delta, poly_digamma(ctx, x));
  return poly_alu2(ctx, POLY_OP_ADD, df, forced);
}

PolyUOp *poly_sigmoid(PolyCtx *ctx, PolyUOp *x) {
  /* Pinned sigmoid:
   * (1 + (x * (-1/log(2))).exp2()).reciprocal()
   * (mixin/elementwise.py:667-679). UOp.ufix keeps float16/float64 receivers
   * in their own dtype; integer/bool receivers promote at the multiplication. */
  if (!ctx || !x) return NULL;
  PolyUOp *scaled =
      poly_elementwise_scalar_binop(ctx, POLY_OP_MUL, x, POLY_FLOAT32, -1.0 / M_LN2, false);
  PolyUOp *e = scaled ? poly_alu1(ctx, POLY_OP_EXP2, scaled) : NULL;
  PolyUOp *denom =
      e ? poly_elementwise_scalar_binop(ctx, POLY_OP_ADD, e, POLY_INT32, 1.0, true) : NULL;
  return denom ? poly_alu1(ctx, POLY_OP_RECIPROCAL, denom) : NULL;
}

PolyUOp *poly_tanh_act(PolyCtx *ctx, PolyUOp *x) {
  /* Pinned tanh(x) = 2.0 * sigmoid(2.0 * x) - 1.0
   * (mixin/elementwise.py:739-749). */
  if (!ctx || !x) return NULL;
  PolyUOp *two_x = poly_elementwise_scalar_binop(ctx, POLY_OP_MUL, x, POLY_FLOAT32, 2.0, true);
  PolyUOp *sigmoid = two_x ? poly_sigmoid(ctx, two_x) : NULL;
  PolyUOp *twice =
      sigmoid ? poly_elementwise_scalar_binop(ctx, POLY_OP_MUL, sigmoid, POLY_FLOAT32, 2.0, true)
              : NULL;
  return twice ? poly_elementwise_scalar_binop(ctx, POLY_OP_SUB, twice, POLY_FLOAT32, 1.0, false)
               : NULL;
}

PolyUOp *poly_abs(PolyCtx *ctx, PolyUOp *x) {
  /* abs(x) = x * sign(x) */
  return poly_alu2(ctx, POLY_OP_MUL, x, poly_sign(ctx, x));
}

PolyUOp *poly_sign(PolyCtx *ctx, PolyUOp *x) {
  /* Pinned sign uses typed, shaped const_like branches; NaN selects +1. */
  PolyUOp *nonzero = poly_elementwise_scalar_binop(ctx, POLY_OP_CMPNE, x, POLY_INT32, 0, false);
  PolyUOp *negative = poly_elementwise_scalar_binop(ctx, POLY_OP_CMPLT, x, POLY_INT32, 0, false);
  PolyUOp *signed_one = poly_where_op(
      ctx, negative, poly_const_like(ctx, x, poly_arg_int(-1)),
      poly_const_like(ctx, x, poly_arg_int(1))
  );
  return poly_where_op(ctx, nonzero, signed_one, poly_const_like(ctx, x, poly_arg_int(0)));
}

PolyUOp *poly_square(PolyCtx *ctx, PolyUOp *x) {
  return poly_alu2(ctx, POLY_OP_MUL, x, x);
}

PolyUOp *poly_rsqrt(PolyCtx *ctx, PolyUOp *x) {
  return poly_alu1(ctx, POLY_OP_RECIPROCAL, poly_alu1(ctx, POLY_OP_SQRT, x));
}

PolyUOp *poly_ceil(PolyCtx *ctx, PolyUOp *x) {
  /* ceil(x) = (x > (b=trunc(x))).where(b+1, b) */
  PolyUOp *b = poly_alu1(ctx, POLY_OP_TRUNC, x);
  PolyUOp *cond = poly_alu2(ctx, POLY_OP_CMPLT, b, x); /* b < x = x > b */
  return poly_alu3(
      ctx, POLY_OP_WHERE, cond,
      poly_alu2(ctx, POLY_OP_ADD, b, poly_elementwise_float_const(ctx, x, 1.0)), b
  );
}

PolyUOp *poly_floor(PolyCtx *ctx, PolyUOp *x) {
  /* floor(x) = (x < (b=trunc(x))).where(b-1, b) */
  PolyUOp *b = poly_alu1(ctx, POLY_OP_TRUNC, x);
  PolyUOp *cond = poly_alu2(ctx, POLY_OP_CMPLT, x, b); /* x < b */
  return poly_alu3(
      ctx, POLY_OP_WHERE, cond,
      poly_alu2(ctx, POLY_OP_SUB, b, poly_elementwise_float_const(ctx, x, 1.0)), b
  );
}

PolyUOp *poly_round_f(PolyCtx *ctx, PolyUOp *x) {
  /* round(x) with banker's rounding (round half to even):
   * (x > 0) == (trunc(x/2) == trunc(trunc(x)/2)) ? ceil(x-0.5) : floor(x+0.5) */
  PolyUOp *half = poly_elementwise_float_const(ctx, x, 0.5);
  PolyUOp *two = poly_elementwise_float_const(ctx, x, 2.0);
  PolyUOp *b = poly_alu1(ctx, POLY_OP_TRUNC, x);
  PolyUOp *x_gt_0 = poly_alu2(ctx, POLY_OP_CMPLT, poly_elementwise_float_const(ctx, x, 0.0), x);
  PolyUOp *b_half = poly_alu2(ctx, POLY_OP_FDIV, b, two);
  PolyUOp *x_half = poly_alu2(ctx, POLY_OP_FDIV, x, two);
  PolyUOp *trunc_b_half = poly_alu1(ctx, POLY_OP_TRUNC, b_half);
  PolyUOp *trunc_x_half = poly_alu1(ctx, POLY_OP_TRUNC, x_half);
  PolyUOp *halves_eq = poly_eq(ctx, trunc_b_half, trunc_x_half);
  PolyUOp *cond = poly_eq(ctx, x_gt_0, halves_eq);
  return poly_alu3(
      ctx, POLY_OP_WHERE, cond, poly_ceil(ctx, poly_alu2(ctx, POLY_OP_SUB, x, half)),
      poly_floor(ctx, poly_alu2(ctx, POLY_OP_ADD, x, half))
  );
}

PolyUOp *poly_isinf(PolyCtx *ctx, PolyUOp *x) {
  /* ElementwiseMixin.isinf with both detection flags enabled. */
  PolyUOp *positive = poly_eq(ctx, x, poly_const_exact_float(ctx, POLY_WEAKFLOAT, INFINITY));
  PolyUOp *negative = poly_eq(ctx, x, poly_const_exact_float(ctx, POLY_WEAKFLOAT, -INFINITY));
  positive = poly_elementwise_scalar_binop(ctx, POLY_OP_MUL, positive, POLY_BOOL, 1, false);
  negative = poly_elementwise_scalar_binop(ctx, POLY_OP_MUL, negative, POLY_BOOL, 1, false);
  return poly_add(ctx, positive, negative);
}

PolyUOp *poly_isnan(PolyCtx *ctx, PolyUOp *x) {
  /* isnan(x) = (x != x) -- IEEE 754 */
  return poly_alu2(ctx, POLY_OP_CMPNE, x, x);
}

PolyUOp *poly_isfinite(PolyCtx *ctx, PolyUOp *x) {
  return poly_logical_not(ctx, poly_binop(ctx, POLY_OP_OR, poly_isinf(ctx, x), poly_isnan(ctx, x)));
}

PolyUOp *poly_isclose(
    PolyCtx *ctx,
    PolyUOp *a,
    PolyUOp *b,
    PolyUOp *rtol,
    PolyUOp *atol,
    bool equal_nan
) {
  PolyUOp *tolerance = poly_add(ctx, atol, poly_mul(ctx, rtol, poly_abs(ctx, b)));
  PolyUOp *finite = poly_binop(ctx, POLY_OP_AND, poly_isfinite(ctx, a), poly_isfinite(ctx, b));
  finite = poly_binop(
      ctx, POLY_OP_AND, finite, poly_le(ctx, poly_abs(ctx, poly_sub(ctx, a, b)), tolerance)
  );
  PolyUOp *infinite = poly_binop(ctx, POLY_OP_OR, poly_isinf(ctx, a), poly_isinf(ctx, b));
  infinite = poly_binop(ctx, POLY_OP_AND, infinite, poly_eq(ctx, a, b));
  PolyUOp *nan = poly_binop(ctx, POLY_OP_AND, poly_isnan(ctx, a), poly_isnan(ctx, b));
  nan = poly_elementwise_scalar_binop(ctx, POLY_OP_AND, nan, POLY_BOOL, equal_nan, false);
  return poly_binop(ctx, POLY_OP_OR, poly_binop(ctx, POLY_OP_OR, finite, infinite), nan);
}

PolyUOp *poly_copysign(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  if (!poly_broadcasted_pair(ctx, &a, &b)) return NULL;
  PolyUOp *negative = poly_elementwise_scalar_binop(ctx, POLY_OP_CMPLT, b, POLY_INT32, 0, false);
  PolyUOp *signbit = poly_elementwise_scalar_binop(
      ctx, POLY_OP_CMPLT, poly_alu1(ctx, POLY_OP_RECIPROCAL, b), POLY_INT32, 0, false
  );
  PolyUOp *magnitude = poly_abs(ctx, a);
  return poly_where_op(
      ctx, poly_binop(ctx, POLY_OP_OR, negative, signbit), poly_elementwise_neg(ctx, magnitude),
      magnitude
  );
}

PolyUOp *poly_lerp(PolyCtx *ctx, PolyUOp *x, PolyUOp *end, PolyUOp *weight, bool scalar_weight) {
  if (!ctx || !x || !end || !weight) return NULL;
  PolyUOp *difference = poly_sub(ctx, end, x);
  /* The pinned uint8 path is selected by host-scalar versus Tensor provenance,
   * not by weight dtype: a weak Tensor is still a Tensor. */
  if (poly_dtype_eq(x->dtype, POLY_UINT8) && !scalar_weight) {
    PolyUOp *scaled =
        poly_elementwise_scalar_binop(ctx, POLY_OP_MUL, weight, POLY_INT32, 128, false);
    PolyUOp *wi = poly_cast(
        ctx, poly_elementwise_scalar_binop(ctx, POLY_OP_ADD, scaled, POLY_FLOAT32, 0.5, false),
        POLY_INT16
    );
    PolyUOp *offset = poly_mul(ctx, poly_cast(ctx, difference, POLY_INT8), wi);
    offset = poly_cast(
        ctx, poly_elementwise_scalar_binop(ctx, POLY_OP_ADD, offset, POLY_INT32, 64, false),
        POLY_UINT16
    );
    offset = poly_elementwise_scalar_binop(ctx, POLY_OP_SHR, offset, POLY_INT32, 7, false);
    return poly_cast(ctx, poly_add(ctx, x, offset), POLY_UINT8);
  }
  return poly_add(ctx, x, poly_mul(ctx, difference, weight));
}

/* Activations */

PolyUOp *poly_relu(PolyCtx *ctx, PolyUOp *x) {
  /* relu(x) = where(0 < x, x, 0) */
  PolyUOp *zero = poly_elementwise_float_const(ctx, x, 0.0);
  PolyUOp *cond = poly_alu2(ctx, POLY_OP_CMPLT, zero, x);
  return poly_alu3(ctx, POLY_OP_WHERE, cond, x, zero);
}

PolyUOp *poly_relu6(PolyCtx *ctx, PolyUOp *x) {
  /* relu6(x) = relu(x) - relu(x - 6) */
  return poly_alu2(
      ctx, POLY_OP_SUB, poly_relu(ctx, x),
      poly_relu(ctx, poly_alu2(ctx, POLY_OP_SUB, x, poly_elementwise_float_const(ctx, x, 6.0)))
  );
}

PolyUOp *poly_leaky_relu(PolyCtx *ctx, PolyUOp *x, double neg_slope) {
  PolyUOp *cond = poly_alu2(ctx, POLY_OP_CMPLT, x, poly_elementwise_float_const(ctx, x, 0.0));
  return poly_alu3(
      ctx, POLY_OP_WHERE, cond,
      poly_alu2(ctx, POLY_OP_MUL, poly_elementwise_float_const(ctx, x, neg_slope), x), x
  );
}

PolyUOp *poly_gelu(PolyCtx *ctx, PolyUOp *x) {
  /* Pinned tanh GELU:
   * 0.5*x*(1 + (sqrt(2/pi)*(x + 0.044715*x**3)).tanh())
   * (mixin/elementwise.py:761-776). Keep every source operation's own ufix
   * and promotion boundary: integer x**3 remains integer until multiplied by
   * the floating coefficient. */
  if (!ctx || !x) return NULL;
  PolyUOp *half_x = poly_elementwise_scalar_binop(ctx, POLY_OP_MUL, x, POLY_FLOAT32, 0.5, true);
  PolyUOp *x3 = poly_elementwise_scalar_binop(ctx, POLY_OP_POW, x, POLY_INT32, 3.0, false);
  PolyUOp *cubic =
      x3 ? poly_elementwise_scalar_binop(ctx, POLY_OP_MUL, x3, POLY_FLOAT32, 0.044715, true) : NULL;
  PolyUOp *inner = cubic ? poly_binop(ctx, POLY_OP_ADD, x, cubic) : NULL;
  PolyUOp *scaled = inner ? poly_elementwise_scalar_binop(
                                ctx, POLY_OP_MUL, inner, POLY_FLOAT32, sqrt(2.0 / M_PI), true
                            )
                          : NULL;
  PolyUOp *tanh = scaled ? poly_tanh_act(ctx, scaled) : NULL;
  PolyUOp *one_plus =
      tanh ? poly_elementwise_scalar_binop(ctx, POLY_OP_ADD, tanh, POLY_INT32, 1.0, true) : NULL;
  return half_x && one_plus ? poly_binop(ctx, POLY_OP_MUL, half_x, one_plus) : NULL;
}

PolyUOp *poly_quick_gelu(PolyCtx *ctx, PolyUOp *x) {
  /* Pinned quick_gelu is self * (self * 1.702).sigmoid()
   * (mixin/elementwise.py:751-759). */
  if (!ctx || !x) return NULL;
  PolyUOp *inner = poly_elementwise_scalar_binop(ctx, POLY_OP_MUL, x, POLY_FLOAT32, 1.702, false);
  PolyUOp *sigmoid = inner ? poly_sigmoid(ctx, inner) : NULL;
  return sigmoid ? poly_binop(ctx, POLY_OP_MUL, x, sigmoid) : NULL;
}

PolyUOp *poly_silu(PolyCtx *ctx, PolyUOp *x) {
  PolyUOp *sigmoid = poly_sigmoid(ctx, x);
  return sigmoid ? poly_binop(ctx, POLY_OP_MUL, x, sigmoid) : NULL;
}

PolyUOp *poly_elu(PolyCtx *ctx, PolyUOp *x, double alpha) {
  return poly_alu2(
      ctx, POLY_OP_SUB, poly_relu(ctx, x),
      poly_alu2(
          ctx, POLY_OP_MUL, poly_elementwise_float_const(ctx, x, alpha),
          poly_relu(
              ctx, poly_alu2(
                       ctx, POLY_OP_SUB, poly_elementwise_float_const(ctx, x, 1.0), poly_exp(ctx, x)
                   )
          )
      )
  );
}

/* ElementwiseMixin.softplus/logaddexp, with the pinned floating beta. */
PolyUOp *poly_softplus(PolyCtx *ctx, PolyUOp *x, double beta) {
  if (!ctx || !x || beta == 0) return NULL;
  PolyUOp *a = poly_elementwise_scalar_binop(ctx, POLY_OP_MUL, x, POLY_FLOAT32, beta, false);
  PolyUOp *b = poly_const_exact_float(ctx, POLY_WEAKFLOAT, 0.0);
  if (!poly_broadcasted_pair(ctx, &a, &b)) return NULL;
  PolyUOp *m = poly_maximum(ctx, a, b);
  PolyUOp *sum =
      poly_add(ctx, poly_exp(ctx, poly_sub(ctx, a, m)), poly_exp(ctx, poly_sub(ctx, b, m)));
  PolyUOp *lae = poly_add(ctx, poly_log(ctx, sum), m);
  return poly_elementwise_scalar_binop(ctx, POLY_OP_MUL, lae, POLY_FLOAT32, 1.0 / beta, true);
}

PolyUOp *poly_logsigmoid(PolyCtx *ctx, PolyUOp *x) {
  PolyUOp *out = poly_softplus(ctx, poly_elementwise_neg(ctx, x), 1.0);
  return out ? poly_elementwise_neg(ctx, out) : NULL;
}

PolyUOp *poly_mish(PolyCtx *ctx, PolyUOp *x) {
  return poly_alu2(ctx, POLY_OP_MUL, x, poly_tanh_act(ctx, poly_softplus(ctx, x, 1.0)));
}

PolyUOp *poly_hardtanh(PolyCtx *ctx, PolyUOp *x, double min_val, double max_val) {
  return poly_clamp(ctx, x, min_val, max_val);
}

PolyUOp *poly_hardswish(PolyCtx *ctx, PolyUOp *x) {
  return poly_alu2(
      ctx, POLY_OP_MUL,
      poly_alu2(
          ctx, POLY_OP_MUL, x,
          poly_relu6(ctx, poly_alu2(ctx, POLY_OP_ADD, x, poly_elementwise_float_const(ctx, x, 3.0)))
      ),
      poly_elementwise_float_const(ctx, x, 1.0 / 6.0)
  );
}

PolyUOp *poly_hardsigmoid(PolyCtx *ctx, PolyUOp *x) {
  PolyUOp *t = poly_alu2(
      ctx, POLY_OP_ADD,
      poly_alu2(ctx, POLY_OP_MUL, poly_elementwise_float_const(ctx, x, 1.0 / 6.0), x),
      poly_elementwise_float_const(ctx, x, 0.5)
  );
  return poly_alu2(
      ctx, POLY_OP_SUB, poly_relu(ctx, t),
      poly_relu(ctx, poly_alu2(ctx, POLY_OP_SUB, t, poly_elementwise_float_const(ctx, x, 1.0)))
  );
}

PolyUOp *poly_ne(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  if (!poly_broadcasted_pair(ctx, &a, &b)) return NULL;
  return poly_alu2(ctx, POLY_OP_CMPNE, a, b);
}

PolyUOp *poly_gt(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  if (!poly_broadcasted_pair(ctx, &a, &b)) return NULL;
  return poly_alu2(ctx, POLY_OP_CMPLT, b, a);
}

PolyUOp *poly_ge(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  if (!poly_broadcasted_pair(ctx, &a, &b)) return NULL;
  PolyUOp *lt = poly_alu2(ctx, POLY_OP_CMPLT, a, b);
  return poly_logical_not(ctx, lt);
}

PolyUOp *poly_le(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  if (!poly_broadcasted_pair(ctx, &a, &b)) return NULL;
  PolyUOp *gt = poly_alu2(ctx, POLY_OP_CMPLT, b, a);
  return poly_logical_not(ctx, gt);
}

PolyUOp *poly_maximum(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  if (!poly_broadcasted_pair(ctx, &a, &b)) return NULL;
  return poly_alu2(ctx, POLY_OP_MAX, a, b);
}

/* Pinned ElementwiseMixin._inverse is ordinary broadcasted negation for
 * floating values and bitwise-not for integer/bool values
 * (mixin/elementwise.py:57-69,131-143,379-393). */
PolyUOp *poly_elementwise_inverse(PolyCtx *ctx, PolyUOp *x) {
  if (!ctx || !x) return NULL;
  PolyDType dt = x->dtype;
  if (poly_dtype_is_float(dt)) {
    PolyUOp *minus_one = poly_const_typed(ctx, POLY_WEAKFLOAT, -1.0);
    return minus_one ? poly_alu2(ctx, POLY_OP_MUL, x, minus_one) : NULL;
  }
  if (poly_dtype_is_bool(dt)) return poly_logical_not(ctx, x);
  if (!poly_dtype_is_int(dt)) return NULL;
  PolyUOp *mask = NULL;
  if (poly_dtype_is_unsigned(dt)) {
    if (!poly_dtype_bound_const(ctx, dt, false, &mask)) return NULL;
    /* _inverse uses Python dtype.max / -1 literals, not typed Tensor
     * constants. Preserve the exact uint64 bound while weakening its dtype. */
    mask = poly_uop_const(ctx, mask->arg, POLY_WEAKINT);
  } else {
    mask = poly_const_int(ctx, -1);
  }
  return mask ? poly_alu2(ctx, POLY_OP_XOR, x, mask) : NULL;
}

PolyUOp *poly_minimum(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  if (!poly_broadcasted_pair(ctx, &a, &b)) return NULL;
  /* ElementwiseMixin.minimum uses XOR with dtype.const(min + max), even
   * for bool. Unary min's _inverse instead uses logical_not (CMPNE). */
  if (poly_dtype_is_bool(a->dtype)) {
    PolyUOp *mask = poly_const_typed(ctx, POLY_BOOL, 1);
    if (!mask) return NULL;
    PolyUOp *left = poly_alu2(ctx, POLY_OP_XOR, a, mask);
    PolyUOp *right = poly_alu2(ctx, POLY_OP_XOR, b, mask);
    PolyUOp *maximum = left && right ? poly_alu2(ctx, POLY_OP_MAX, left, right) : NULL;
    return maximum ? poly_alu2(ctx, POLY_OP_XOR, maximum, mask) : NULL;
  }
  PolyUOp *inverse_a = poly_elementwise_inverse(ctx, a);
  PolyUOp *inverse_b = poly_elementwise_inverse(ctx, b);
  PolyUOp *maximum =
      (inverse_a && inverse_b) ? poly_alu2(ctx, POLY_OP_MAX, inverse_a, inverse_b) : NULL;
  return maximum ? poly_elementwise_inverse(ctx, maximum) : NULL;
}

PolyUOp *poly_clamp(PolyCtx *ctx, PolyUOp *x, double lo, double hi) {
  PolyUOp *lo_c = poly_elementwise_float_const(ctx, x, lo);
  PolyUOp *hi_c = poly_elementwise_float_const(ctx, x, hi);
  PolyUOp *lt_lo = poly_alu2(ctx, POLY_OP_CMPLT, x, lo_c);
  PolyUOp *clamped_lo = poly_alu3(ctx, POLY_OP_WHERE, lt_lo, lo_c, x);
  PolyUOp *gt_hi = poly_alu2(ctx, POLY_OP_CMPLT, hi_c, clamped_lo);
  return poly_alu3(ctx, POLY_OP_WHERE, gt_hi, hi_c, clamped_lo);
}

PolyUOp *poly_detach(PolyCtx *ctx, PolyUOp *x) {
  return poly_uop1(ctx, POLY_OP_DETACH, x->dtype, x, poly_arg_none());
}
