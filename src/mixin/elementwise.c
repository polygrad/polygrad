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
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_LN2
#define M_LN2 0.693147180559945309417
#endif

/* For a movement chain rooted at CONST, PAD is the only operation that can
 * add a different value. This is upstream's vmin == vmax proof specialized
 * to that chain, without projecting float bounds through the int64 cache. */
static bool weak_const_uniform(PolyUOp *root, PolyUOp *base) {
  if (base->arg.kind == POLY_ARG_FLOAT && isnan(base->arg.f)) return false;
  bool zero = (base->arg.kind == POLY_ARG_INT && base->arg.i == 0) ||
              (base->arg.kind == POLY_ARG_FLOAT && base->arg.f == 0.0);
  for (PolyUOp *u = root; u != base; u = u->src[0])
    if (u->op == POLY_OP_PAD && !zero) return false;
  return true;
}

static PolyUOp *remint_weak_const(PolyCtx *ctx, PolyUOp *u, PolyDType dt) {
  if (u->op == POLY_OP_CONST) return poly_uop_const_like_dtype(ctx, u, u->arg, dt);
  PolyUOp **src = malloc((size_t)u->n_src * sizeof(*src));
  if (!src) return NULL;
  memcpy(src, u->src, (size_t)u->n_src * sizeof(*src));
  src[0] = remint_weak_const(ctx, src[0], dt);
  PolyUOp *out =
      src[0] ? poly_uop_tagged_arg(ctx, u->op, dt, src, u->n_src, u->arg, u->tag, u->tag_arg)
             : NULL;
  free(src);
  return out;
}

/* C helper for current ElementwiseMixin._broadcasted.promote. It is public
 * inside the core because Polygrad applies one promotion to each retained
 * logical/physical root (mixin/elementwise.py:21-29). */
PolyUOp *poly_uop_elementwise_promote(PolyCtx *ctx, PolyUOp *root, PolyDType common) {
  if (!ctx || !root) return NULL;
  PolyUOp *base = poly_uop_base(root);
  /* Invalid is a sentinel, not a bool value to cast to the common dtype. */
  if (base->op == POLY_OP_CONST && base->arg.kind == POLY_ARG_INVALID) return root;
  if (poly_dtype_is_weak(root->dtype) && base->op == POLY_OP_CONST &&
      weak_const_uniform(root, base)) {
    return remint_weak_const(ctx, root, poly_dtype_weak(common));
  }
  return poly_dtype_eq(root->dtype, common) ? root : poly_uop_cast(ctx, root, common);
}

/* Current ElementwiseMixin._broadcasted applies least_upper_dtype while
 * keeping bare weak CONSTs weak (mixin/elementwise.py:21-29). */
bool poly_uop_broadcasted_pair(PolyCtx *ctx, PolyUOp **a, PolyUOp **b) {
  if (!ctx || !a || !b || !*a || !*b) return false;

  PolyDType common;
  if (!poly_dtype_least_upper((*a)->dtype, (*b)->dtype, &common)) return false;
  *a = poly_uop_elementwise_promote(ctx, *a, common);
  *b = poly_uop_elementwise_promote(ctx, *b, common);
  return *a != NULL && *b != NULL;
}

/* Current ElementwiseMixin._binop promotes once, then constructs the selected
 * ALU UOp (mixin/elementwise.py:32-34). */
PolyUOp *poly_uop_binop(PolyCtx *ctx, PolyOps op, PolyUOp *a, PolyUOp *b) {
  if (!poly_uop_broadcasted_pair(ctx, &a, &b)) return NULL;
  return poly_uop_alu2(ctx, op, a, b);
}

/* Broadcasting binary ops */

PolyUOp *poly_uop_add(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  if (!poly_uop_broadcasted_pair(ctx, &a, &b)) return NULL;
  return poly_uop_alu2(ctx, POLY_OP_ADD, a, b);
}

PolyUOp *poly_uop_sub(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  if (!poly_uop_broadcasted_pair(ctx, &a, &b)) return NULL;

  /* Current ElementwiseMixin.sub is a.alu(ADD, -b) after the one
   * _broadcasted promotion pass. Negation keeps its scalar -1 weak and UOp
   * shape inference owns any shape broadcast (mixin/elementwise.py:104-119). */
  PolyUOp *neg_b = NULL;
  if (poly_dtype_is_bool(b->dtype)) {
    neg_b = poly_uop_logical_not(ctx, b);
  } else {
    PolyUOp *minus_one = poly_uop_const_typed(ctx, poly_dtype_weak(b->dtype), -1.0);
    neg_b = minus_one ? poly_uop_alu2(ctx, POLY_OP_MUL, b, minus_one) : NULL;
  }
  return neg_b ? poly_uop_alu2(ctx, POLY_OP_ADD, a, neg_b) : NULL;
}

PolyUOp *poly_uop_mul(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  if (!poly_uop_broadcasted_pair(ctx, &a, &b)) return NULL;
  return poly_uop_alu2(ctx, POLY_OP_MUL, a, b);
}

/* Comparisons (broadcasting) */

/* Logical NOT — `CMPNE(x, CONST(true))` for
 * bool inputs, matching tinygrad's `logical_not()` after CAST elision
 * (mixin/elementwise.py:39-47 + symbolic.py:93-131). Raw `NEG(bool)` retains
 * arithmetic NEG semantics and is not a second logical-NOT spelling. */
PolyUOp *poly_uop_logical_not(PolyCtx *ctx, PolyUOp *x) {
  if (!x) return NULL;
  x = poly_uop_cast(ctx, x, POLY_BOOL);
  PolyUOp *t = poly_uop_const_typed(ctx, POLY_BOOL, 1);
  return poly_uop_alu2(ctx, POLY_OP_CMPNE, x, t);
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
PolyUOp *poly_uop_eq(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  /* Pinned Tensor.eq reaches _binop/_broadcasted, which broadcasts and then
   * promotes both operands with least_upper_dtype before CMPNE
   * (mixin/elementwise.py:324-325, mixin/__init__.py:439-449). */
  if (!poly_uop_broadcasted_pair(ctx, &a, &b)) return NULL;
  PolyUOp *ne = poly_uop_alu2(ctx, POLY_OP_CMPNE, a, b);
  return poly_uop_logical_not(ctx, ne);
}

PolyUOp *poly_uop_where(PolyCtx *ctx, PolyUOp *cond, PolyUOp *x, PolyUOp *y) {
  if (!cond || !poly_uop_broadcasted_pair(ctx, &x, &y)) return NULL;
  /* tinygrad Tensor.where casts non-bool conditions to bool before building
   * Ops.WHERE. Keeping that in the core constructor preserves the expected
   * CMPNE(cond, 0) node in helper graphs such as nonzero-value padding. */
  if (!poly_dtype_is_bool(cond->dtype)) cond = poly_uop_cast(ctx, cond, POLY_BOOL);
  return poly_uop_alu3(ctx, POLY_OP_WHERE, cond, x, y);
}

/* Helper: float constant matching the dtype of a given UOp.
 * For float inputs: creates a constant with the same float dtype.
 * For non-float inputs (comparisons producing bool): defaults to float32. */
PolyUOp *poly_uop_elementwise_float_const(PolyCtx *ctx, PolyUOp *ref, double v) {
  PolyDType dt = ref->dtype;
  if (poly_dtype_is_float(dt)) return poly_uop_const_typed(ctx, dt, v);
  return poly_uop_const_float(ctx, v);
}

/* Helper: const with explicit dtype -- use in special-math ops for dtype correctness */
static inline PolyUOp *cdt(PolyCtx *ctx, PolyDType dt, double v) {
  return poly_uop_const_typed(ctx, dt, v);
}

PolyUOp *poly_uop_const_exact_int(PolyCtx *ctx, PolyDType dt, int64_t value) {
  dt = dt;
  if (poly_dtype_is_bool(dt)) return poly_uop0(ctx, POLY_OP_CONST, dt, poly_arg_bool(value != 0));
  if (!poly_dtype_is_int(dt)) return NULL;
  return poly_uop0(ctx, POLY_OP_CONST, dt, poly_arg_int(value));
}

PolyUOp *poly_uop_const_exact_float(PolyCtx *ctx, PolyDType dt, double value) {
  dt = dt;
  if (!poly_dtype_is_float(dt)) return NULL;
  return poly_uop0(ctx, POLY_OP_CONST, dt, poly_arg_float(value));
}

bool poly_dtype_bound_const(PolyCtx *ctx, PolyDType dt, bool use_min, PolyUOp **out) {
  if (!ctx || !out) return false;
  dt = dt;
  if (poly_dtype_is_float(dt)) {
    double bound = poly_dtype_float_max(dt);
    *out = poly_uop_const_exact_float(ctx, dt, use_min ? -bound : bound);
    return *out != NULL;
  }
  if (poly_dtype_is_bool(dt)) {
    *out = poly_uop_const_exact_int(ctx, dt, use_min ? 0 : 1);
    return *out != NULL;
  }
  if (!poly_dtype_is_int(dt)) return false;
  if (poly_dtype_is_unsigned(dt)) {
    if (use_min) {
      *out = poly_uop_const_exact_int(ctx, dt, 0);
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
  *out = poly_uop_const_exact_int(ctx, dt, v);
  return *out != NULL;
}

/* Gaussian rational approximations: Cephes ndtr/ndtri, Stephen L. Moshier
 * (1984, 1987, 1989, 1992), SciPy/XSF (2024), BSD-3-Clause; see
 * special_license.h. Coefficients are evaluated as ordinary graph operations,
 * not host math calls, so forward and higher derivatives share all backends. */
#include "special_license.h"

static PolyUOp *special_polynomial(PolyCtx *ctx, PolyUOp *x, const double *c, int n, bool monic) {
  PolyUOp *p = cdt(ctx, x->dtype, monic ? 1 : c[0]);
  for (int i = monic ? 0 : 1; i < n; i++)
    p = poly_uop_add(ctx, poly_uop_mul(ctx, p, x), cdt(ctx, x->dtype, c[i]));
  return p;
}

static PolyUOp *special_float(PolyCtx *ctx, PolyUOp *x, PolyDType *result) {
  if (!ctx || !x || !poly_dtype_least_upper_float(x->dtype, result)) return NULL;
  PolyDType work = result->bitsize < 32 ? POLY_FLOAT32 : *result;
  return poly_dtype_eq(x->dtype, work) ? x : poly_uop_cast(ctx, x, work);
}

static PolyUOp *erf_center(PolyCtx *ctx, PolyUOp *x) {
  const double p[] = {
      9.60497373987051638749E0, 9.00260197203842689217E1, 2.23200534594684319226E3,
      7.00332514112805075473E3, 5.55923013010394962768E4};
  const double q[] = {
      3.35617141647503099647E1, 5.21357949780152679795E2, 4.59432382970980127987E3,
      2.26290000613890934246E4, 4.92673942608635921086E4};
  PolyUOp *z = poly_uop_mul(ctx, x, x);
  return poly_uop_mul(
      ctx, x,
      poly_uop_div(
          ctx, special_polynomial(ctx, z, p, 5, false), special_polynomial(ctx, z, q, 5, true)
      )
  );
}

static PolyUOp *erfc_tail_factor(PolyCtx *ctx, PolyUOp *a, bool logarithm) {
  const double p[] = {
      2.46196981473530512524E-10, 5.64189564831068821977E-1, 7.46321056442269912687E0,
      4.86371970985681366614E1,   1.96520832956077098242E2,  5.26445194995477358631E2,
      9.34528527171957607540E2,   1.02755188689515710272E3,  5.57535335369399327526E2};
  const double q[] = {1.32281951154744992508E1, 8.67072140885989742329E1, 3.54937778887819891062E2,
                      9.75708501743205489753E2, 1.82390916687909736289E3, 2.24633760818710981792E3,
                      1.65666309194161350182E3, 5.57535340817727675546E2};
  const double r[] = {5.64189583547755073984E-1, 1.27536670759978104416E0,
                      5.01905042251180477414E0,  6.16021097993053585195E0,
                      7.40974269950448939160E0,  2.97886665372100240670E0};
  const double s[] = {2.26052863220117276590E0, 9.39603524938001434673E0, 1.20489539808096656605E1,
                      1.70814450747565897222E1, 9.60896809063285878198E0, 3.36907645100081516050E0};
  PolyDType dt = a->dtype;
  PolyUOp *small = poly_uop_alu2(ctx, POLY_OP_CMPLT, a, cdt(ctx, dt, 8));
  if (logarithm) {
    /* Evaluate the large-argument rational in 1/a, factoring out 1/a.
     * Neither powers of a nor exp(-a*a) need be representable. Bound the
     * inactive middle branch too, so its derivative cannot inject NaNs. */
    double reverse_r[6], reverse_s[7];
    for (int i = 0; i < 6; i++) {
      reverse_r[i] = r[5 - i];
      reverse_s[i] = s[5 - i];
    }
    reverse_s[6] = 1;
    PolyUOp *mid = poly_uop_where(ctx, small, a, cdt(ctx, dt, 8));
    PolyUOp *inverse = poly_uop_div(ctx, cdt(ctx, dt, 1), a);
    PolyUOp *center = poly_uop_log(
        ctx,
        poly_uop_div(
            ctx, special_polynomial(ctx, mid, p, 9, false), special_polynomial(ctx, mid, q, 8, true)
        )
    );
    PolyUOp *tail = poly_uop_sub(
        ctx,
        poly_uop_log(
            ctx, poly_uop_div(
                     ctx, special_polynomial(ctx, inverse, reverse_r, 6, false),
                     special_polynomial(ctx, inverse, reverse_s, 7, false)
                 )
        ),
        poly_uop_log(ctx, a)
    );
    return poly_uop_where(ctx, small, center, tail);
  }
  return poly_uop_where(
      ctx, small,
      poly_uop_div(
          ctx, special_polynomial(ctx, a, p, 9, false), special_polynomial(ctx, a, q, 8, true)
      ),
      poly_uop_div(
          ctx, special_polynomial(ctx, a, r, 6, false), special_polynomial(ctx, a, s, 6, true)
      )
  );
}

static PolyUOp *erfc_tail(PolyCtx *ctx, PolyUOp *a) {
  PolyDType dt = a->dtype;
  /* erfc(28) is below the smallest F64 subnormal. Bound inactive branches
   * before evaluation, including infinity, to keep their VJPs finite. */
  PolyUOp *underflow = poly_uop_gt(ctx, a, cdt(ctx, dt, 28));
  a = poly_uop_where(ctx, underflow, cdt(ctx, dt, 28), a);
  PolyUOp *ratio = erfc_tail_factor(ctx, a, false);
  PolyUOp *tail = poly_uop_mul(
      ctx, ratio, poly_uop_exp(ctx, poly_uop_elementwise_neg(ctx, poly_uop_mul(ctx, a, a)))
  );
  return poly_uop_where(ctx, underflow, cdt(ctx, dt, 0), tail);
}

/* lgamma Lanczos helper */

static PolyUOp *gamma_fraction(PolyCtx *ctx, PolyUOp *x) {
  PolyDType dt = x->dtype;
  PolyUOp *r = poly_uop_sub(ctx, x, poly_uop_alu1(ctx, POLY_OP_TRUNC, x));
  /* Keep the sine argument near zero on either side of an integer pole. */
  r = poly_uop_where(
      ctx, poly_uop_gt(ctx, r, cdt(ctx, dt, 0.5)), poly_uop_sub(ctx, r, cdt(ctx, dt, 1)), r
  );
  return poly_uop_where(
      ctx, poly_uop_alu2(ctx, POLY_OP_CMPLT, r, cdt(ctx, dt, -0.5)),
      poly_uop_add(ctx, r, cdt(ctx, dt, 1)), r
  );
}

static PolyUOp *poly_lgamma_forward_lanczos(PolyCtx *ctx, PolyUOp *x, PolyDType dt) {
  /* Evaluate only positive Lanczos arguments. Masking an invalid branch's
   * result is insufficient: its NaN derivative can still poison WHERE's VJP. */
  const double g = 7.0;
  const double c0 = 0.99999999999980993;
  const double c[8] = {676.5203681218851,     -1259.1392167224028,  771.32342877765313,
                       -176.61502916214059,   12.507343278686905,   -0.13857109526572012,
                       9.9843695780195716e-6, 1.5056327351493116e-7};

  PolyUOp *cond = poly_uop_alu2(ctx, POLY_OP_CMPLT, x, cdt(ctx, dt, 0.5));
  PolyUOp *z = poly_uop_where(ctx, cond, poly_uop_sub(ctx, cdt(ctx, dt, 1), x), x);
  PolyUOp *xm1 = poly_uop_alu2(ctx, POLY_OP_SUB, z, cdt(ctx, dt, 1.0));
  PolyUOp *a = cdt(ctx, dt, c0);
  for (int i = 0; i < 8; i++) {
    PolyUOp *den = poly_uop_alu2(ctx, POLY_OP_ADD, xm1, cdt(ctx, dt, (double)(i + 1)));
    a = poly_uop_alu2(
        ctx, POLY_OP_ADD, a, poly_uop_alu2(ctx, POLY_OP_FDIV, cdt(ctx, dt, c[i]), den)
    );
  }
  PolyUOp *t = poly_uop_alu2(ctx, POLY_OP_ADD, xm1, cdt(ctx, dt, g + 0.5));
  PolyUOp *lg_pos = poly_uop_alu2(
      ctx, POLY_OP_ADD, cdt(ctx, dt, 0.91893853320467274178),
      poly_uop_alu2(
          ctx, POLY_OP_ADD,
          poly_uop_alu2(
              ctx, POLY_OP_MUL, poly_uop_alu2(ctx, POLY_OP_ADD, xm1, cdt(ctx, dt, 0.5)),
              poly_uop_log(ctx, t)
          ),
          poly_uop_alu2(ctx, POLY_OP_SUB, poly_uop_log(ctx, a), t)
      )
  );

  PolyUOp *r = poly_uop_where(ctx, cond, x, cdt(ctx, dt, 0.25));
  /* Reduce before multiplying by pi; integer poles are then exact. */
  r = gamma_fraction(ctx, r);
  PolyUOp *sinpix = poly_uop_sin(ctx, poly_uop_mul(ctx, cdt(ctx, dt, M_PI), r));
  PolyUOp *lg_ref = poly_uop_alu2(
      ctx, POLY_OP_SUB,
      poly_uop_alu2(
          ctx, POLY_OP_SUB, cdt(ctx, dt, log(M_PI)), poly_uop_log(ctx, poly_uop_abs(ctx, sinpix))
      ),
      lg_pos
  );
  PolyUOp *out = poly_uop_alu3(ctx, POLY_OP_WHERE, cond, lg_ref, lg_pos);
  return poly_uop_where(ctx, poly_uop_isinf(ctx, x), cdt(ctx, dt, INFINITY), out);
}

PolyUOp *poly_uop_div(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  if (!poly_uop_broadcasted_pair(ctx, &a, &b)) return NULL;
  /* Current ElementwiseMixin.div keeps true division at tensor/UOp stage as
   * MUL(a, RECIPROCAL(b)); FDIV is introduced only by backend decomposition
   * when supported (mixin/elementwise.py:225-252,
   * codegen/decomp/op.py:133-136). */
  if (poly_dtype_is_int(a->dtype) || poly_dtype_is_bool(a->dtype))
    a = poly_uop_cast(ctx, a, poly_dtype_strong(POLY_WEAKFLOAT));
  PolyUOp *reciprocal = b ? poly_uop_alu1(ctx, POLY_OP_RECIPROCAL, b) : NULL;
  return a && reciprocal ? poly_uop_alu2(ctx, POLY_OP_MUL, a, reciprocal) : NULL;
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
  return poly_uop_const_typed(ctx, poly_dtype_weak(from_py_dtype), value);
}

PolyUOp *poly_uop_elementwise_scalar_binop(
    PolyCtx *ctx,
    PolyOps op,
    PolyUOp *self,
    PolyDType from_py_dtype,
    double value,
    bool reverse
) {
  PolyUOp *scalar = poly_ufix_const(ctx, self, from_py_dtype, value);
  if (!scalar) return NULL;
  if (op == POLY_OP_SUB)
    return reverse ? poly_uop_sub(ctx, scalar, self) : poly_uop_sub(ctx, self, scalar);
  if (op == POLY_OP_FDIV)
    return reverse ? poly_uop_div(ctx, scalar, self) : poly_uop_div(ctx, self, scalar);
  return reverse ? poly_uop_binop(ctx, op, scalar, self) : poly_uop_binop(ctx, op, self, scalar);
}

/* Contiguous (realize barrier) */

PolyUOp *poly_uop_contiguous(PolyCtx *ctx, PolyUOp *x) {
  if (!ctx || !x) return NULL;
  /* Tinygrad mixin/elementwise.py:contiguous: weak and storage-free
   * expressions have no materialization to request. */
  if (poly_dtype_is_weak(x->dtype)) return x;
  if (x->op == POLY_OP_CONTIGUOUS) return x;
  if (poly_uop_has_buffer_identity(x)) return x;
  if (poly_uop_device(x) == POLY_DEVICE_AUTO) {
    /* Portable BUFFER(UNIQUE) is unplaced storage, not a virtual expression.
     * Preserve its materialization requests for Model placement/export;
     * genuinely storage-free values still follow tinygrad's device=None rule. */
    int n = 0;
    PolyUOp **topo = poly_uop_toposort_alloc(ctx, x, &n);
    if (!topo) return NULL;
    bool storage = false;
    for (int i = 0; i < n && !storage; i++)
      storage = topo[i]->op == POLY_OP_BUFFER && topo[i]->n_src == 1 &&
                topo[i]->src[0]->op == POLY_OP_UNIQUE;
    poly_uop_toposort_free(topo);
    if (!storage) return x;
  }
  return poly_uop1(ctx, POLY_OP_CONTIGUOUS, x->dtype, x, poly_arg_none());
}

/* Math */

PolyUOp *poly_uop_exp(PolyCtx *ctx, PolyUOp *x) {
  if (!ctx || !x) return NULL;
  /* Pinned exp promotes floating inputs to at least float32 and casts the
   * result back; non-floats stay at least default-float
   * (mixin/elementwise.py:491-503). */
  PolyDType input_dt;
  if (!poly_dtype_least_upper_float(x->dtype, &input_dt)) return NULL;
  x = poly_dtype_eq(x->dtype, input_dt) ? x : poly_uop_cast(ctx, x, input_dt);
  if (!x) return NULL;
  PolyDType compute_dt;
  if (!poly_dtype_least_upper(input_dt, POLY_FLOAT32, &compute_dt)) return NULL;
  PolyUOp *compute_x = poly_dtype_eq(input_dt, compute_dt) ? x : poly_uop_cast(ctx, x, compute_dt);
  PolyUOp *scaled = compute_x ? poly_uop_elementwise_scalar_binop(
                                    ctx, POLY_OP_MUL, compute_x, POLY_FLOAT32, 1.0 / M_LN2, false
                                )
                              : NULL;
  PolyUOp *out = scaled ? poly_uop_alu1(ctx, POLY_OP_EXP2, scaled) : NULL;
  if (out && poly_dtype_is_float(input_dt) && !poly_dtype_eq(input_dt, compute_dt))
    out = poly_uop_cast(ctx, out, input_dt);
  return out;
}

PolyUOp *poly_uop_log(PolyCtx *ctx, PolyUOp *x) {
  if (!ctx || !x) return NULL;
  /* Current log is exactly self.log2()*log(2). LOG2 owns the floating result
   * dtype and retains the original operand (mixin/elementwise.py:827-839,
   * uop/ops.py:144-145). */
  PolyUOp *log2_x = poly_uop_alu1(ctx, POLY_OP_LOG2, x);
  return log2_x ? poly_uop_elementwise_scalar_binop(
                      ctx, POLY_OP_MUL, log2_x, POLY_FLOAT32, M_LN2, false
                  )
                : NULL;
}

PolyUOp *poly_uop_log1p(PolyCtx *ctx, PolyUOp *x) {
  /* Stable near zero: log(1+x) ~ x - x^2/2 + x^3/3 */
  PolyDType dt = x->dtype;
  PolyUOp *ax = poly_uop_abs(ctx, x);
  PolyUOp *small = poly_uop_alu2(ctx, POLY_OP_CMPLT, ax, cdt(ctx, dt, 1e-4));
  PolyUOp *x2 = poly_uop_alu2(ctx, POLY_OP_MUL, x, x);
  PolyUOp *x3 = poly_uop_alu2(ctx, POLY_OP_MUL, x2, x);
  PolyUOp *poly = poly_uop_alu2(
      ctx, POLY_OP_ADD, x,
      poly_uop_alu2(
          ctx, POLY_OP_ADD, poly_uop_alu2(ctx, POLY_OP_MUL, cdt(ctx, dt, -0.5), x2),
          poly_uop_alu2(ctx, POLY_OP_MUL, cdt(ctx, dt, 1.0 / 3.0), x3)
      )
  );
  PolyUOp *direct = poly_uop_log(ctx, poly_uop_alu2(ctx, POLY_OP_ADD, cdt(ctx, dt, 1.0), x));
  return poly_uop_alu3(ctx, POLY_OP_WHERE, small, poly, direct);
}

PolyUOp *poly_uop_expm1(PolyCtx *ctx, PolyUOp *x) {
  /* Stable near zero: expm1(x) ~ x + x^2/2 + x^3/6 */
  PolyDType dt = x->dtype;
  PolyUOp *ax = poly_uop_abs(ctx, x);
  PolyUOp *small = poly_uop_alu2(ctx, POLY_OP_CMPLT, ax, cdt(ctx, dt, 1e-4));
  PolyUOp *x2 = poly_uop_alu2(ctx, POLY_OP_MUL, x, x);
  PolyUOp *x3 = poly_uop_alu2(ctx, POLY_OP_MUL, x2, x);
  PolyUOp *poly = poly_uop_alu2(
      ctx, POLY_OP_ADD, x,
      poly_uop_alu2(
          ctx, POLY_OP_ADD, poly_uop_alu2(ctx, POLY_OP_MUL, cdt(ctx, dt, 0.5), x2),
          poly_uop_alu2(ctx, POLY_OP_MUL, cdt(ctx, dt, 1.0 / 6.0), x3)
      )
  );
  PolyUOp *direct = poly_uop_alu2(ctx, POLY_OP_SUB, poly_uop_exp(ctx, x), cdt(ctx, dt, 1.0));
  return poly_uop_alu3(ctx, POLY_OP_WHERE, small, poly, direct);
}

PolyUOp *poly_uop_sin(PolyCtx *ctx, PolyUOp *x) {
  return poly_uop_alu1(ctx, POLY_OP_SIN, x);
}

PolyUOp *poly_uop_cos(PolyCtx *ctx, PolyUOp *x) {
  if (!ctx || !x) return NULL;
  /* Current cos first casts with least_upper_float, computes pi/2-x at at
   * least float32, applies SIN, then casts back to the first dtype
   * (mixin/elementwise.py:480-489). */
  PolyDType result_dtype;
  if (!poly_dtype_least_upper_float(x->dtype, &result_dtype)) return NULL;
  PolyUOp *self = poly_dtype_eq(x->dtype, result_dtype) ? x : poly_uop_cast(ctx, x, result_dtype);
  if (!self) return NULL;

  PolyDType compute_dtype;
  if (!poly_dtype_least_upper(self->dtype, POLY_FLOAT32, &compute_dtype)) return NULL;
  PolyUOp *work =
      poly_dtype_eq(self->dtype, compute_dtype) ? self : poly_uop_cast(ctx, self, compute_dtype);
  PolyUOp *angle = work ? poly_uop_elementwise_scalar_binop(
                              ctx, POLY_OP_SUB, work, POLY_FLOAT32, M_PI / 2.0, true
                          )
                        : NULL;
  PolyUOp *out = angle ? poly_uop_alu1(ctx, POLY_OP_SIN, angle) : NULL;
  return out && !poly_dtype_eq(out->dtype, result_dtype) ? poly_uop_cast(ctx, out, result_dtype)
                                                         : out;
}

PolyUOp *poly_uop_tan(PolyCtx *ctx, PolyUOp *x) {
  /* Current tan is the ordinary high-level quotient self.sin()/self.cos()
   * (mixin/elementwise.py:917-927). */
  return poly_uop_div(ctx, poly_uop_sin(ctx, x), poly_uop_cos(ctx, x));
}

/* Pinned ElementwiseMixin compositions. Keep scalar weakness and source order:
 * these are Tensor-level graphs, not algebraically equivalent backend rewrites. */
PolyUOp *poly_uop_elementwise_neg(PolyCtx *ctx, PolyUOp *x) {
  if (!ctx || !x) return NULL;
  return poly_dtype_is_bool(x->dtype)
             ? poly_uop_logical_not(ctx, x)
             : poly_uop_elementwise_scalar_binop(ctx, POLY_OP_MUL, x, POLY_INT32, -1, false);
}

PolyUOp *poly_uop_log10(PolyCtx *ctx, PolyUOp *x) {
  return poly_uop_elementwise_scalar_binop(
      ctx, POLY_OP_MUL, poly_uop_alu1(ctx, POLY_OP_LOG2, x), POLY_FLOAT32, log10(2.0), false
  );
}

PolyUOp *poly_uop_atanh(PolyCtx *ctx, PolyUOp *x) {
  PolyUOp *a = poly_uop_elementwise_scalar_binop(ctx, POLY_OP_ADD, x, POLY_INT32, 1, true);
  PolyUOp *b = poly_uop_elementwise_scalar_binop(ctx, POLY_OP_SUB, x, POLY_INT32, 1, true);
  return poly_uop_elementwise_scalar_binop(
      ctx, POLY_OP_FDIV, poly_uop_log(ctx, poly_uop_div(ctx, a, b)), POLY_INT32, 2, false
  );
}

PolyUOp *poly_uop_asinh(PolyCtx *ctx, PolyUOp *x) {
  PolyUOp *negative =
      poly_uop_elementwise_scalar_binop(ctx, POLY_OP_CMPLT, x, POLY_INT32, 0, false);
  PolyUOp *sign =
      poly_uop_where(ctx, negative, poly_uop_const_float(ctx, -1), poly_uop_const_float(ctx, 1));
  PolyUOp *a = poly_uop_elementwise_scalar_binop(
      ctx, POLY_OP_ADD, poly_uop_square(ctx, x), POLY_INT32, 1, false
  );
  return poly_uop_mul(
      ctx, sign,
      poly_uop_log(
          ctx, poly_uop_add(ctx, poly_uop_mul(ctx, x, sign), poly_uop_alu1(ctx, POLY_OP_SQRT, a))
      )
  );
}

PolyUOp *poly_uop_acosh(PolyCtx *ctx, PolyUOp *x) {
  PolyUOp *a = poly_uop_elementwise_scalar_binop(
      ctx, POLY_OP_SUB, poly_uop_square(ctx, x), POLY_INT32, 1, false
  );
  return poly_uop_log(ctx, poly_uop_add(ctx, x, poly_uop_alu1(ctx, POLY_OP_SQRT, a)));
}

/* helpers.polyN starts at 0.0, including the first multiply/add in the graph. */
static PolyUOp *pointwise_polyn(PolyCtx *ctx, PolyUOp *x, const double *coefficients, int n) {
  PolyUOp *p = poly_uop_const_exact_float(ctx, POLY_WEAKFLOAT, 0.0);
  for (int i = 0; p && i < n; i++)
    p = poly_uop_elementwise_scalar_binop(
        ctx, POLY_OP_ADD, poly_uop_mul(ctx, p, x), POLY_FLOAT32, coefficients[i], false
    );
  return p;
}

PolyUOp *poly_uop_asin(PolyCtx *ctx, PolyUOp *x) {
  const double coefficients[] = {-0.0012624911, 0.0066700901, -0.0170881256, 0.0308918810,
                                 -0.0501743046, 0.0889789874, -0.2145988016, M_PI / 2};
  PolyUOp *sign = poly_uop_where(
      ctx, poly_uop_ge(ctx, x, poly_uop_const_int(ctx, 0)), poly_uop_const_float(ctx, 1),
      poly_uop_const_float(ctx, -1)
  );
  PolyUOp *ax = poly_uop_mul(ctx, sign, x);
  PolyUOp *root = poly_uop_alu1(
      ctx, POLY_OP_SQRT,
      poly_uop_elementwise_scalar_binop(ctx, POLY_OP_SUB, ax, POLY_FLOAT32, 1.0, true)
  );
  PolyUOp *p = poly_uop_mul(ctx, root, pointwise_polyn(ctx, ax, coefficients, 8));
  return poly_uop_mul(
      ctx, sign,
      poly_uop_elementwise_scalar_binop(ctx, POLY_OP_SUB, p, POLY_FLOAT32, M_PI / 2, true)
  );
}

PolyUOp *poly_uop_acos(PolyCtx *ctx, PolyUOp *x) {
  return poly_uop_elementwise_scalar_binop(
      ctx, POLY_OP_SUB, poly_uop_asin(ctx, x), POLY_FLOAT32, M_PI / 2, true
  );
}

PolyUOp *poly_uop_atan(PolyCtx *ctx, PolyUOp *x) {
  PolyUOp *a = poly_uop_elementwise_scalar_binop(
      ctx, POLY_OP_ADD, poly_uop_mul(ctx, x, x), POLY_INT32, 1, true
  );
  return poly_uop_asin(ctx, poly_uop_div(ctx, x, poly_uop_alu1(ctx, POLY_OP_SQRT, a)));
}

PolyUOp *poly_uop_celu(PolyCtx *ctx, PolyUOp *x, PolyUOp *alpha) {
  return poly_uop_mul(ctx, alpha, poly_uop_elu(ctx, poly_uop_div(ctx, x, alpha), 1.0));
}

PolyUOp *poly_uop_selu(PolyCtx *ctx, PolyUOp *x, PolyUOp *alpha, PolyUOp *gamma) {
  return poly_uop_mul(ctx, gamma, poly_uop_elu_tensor(ctx, x, alpha));
}

PolyUOp *poly_uop_sinh(PolyCtx *ctx, PolyUOp *x) {
  PolyUOp *a =
      poly_uop_sub(ctx, poly_uop_exp(ctx, x), poly_uop_exp(ctx, poly_uop_elementwise_neg(ctx, x)));
  return poly_uop_elementwise_scalar_binop(ctx, POLY_OP_FDIV, a, POLY_INT32, 2, false);
}

PolyUOp *poly_uop_cosh(PolyCtx *ctx, PolyUOp *x) {
  PolyUOp *a =
      poly_uop_add(ctx, poly_uop_exp(ctx, x), poly_uop_exp(ctx, poly_uop_elementwise_neg(ctx, x)));
  return poly_uop_elementwise_scalar_binop(ctx, POLY_OP_FDIV, a, POLY_INT32, 2, false);
}

PolyUOp *poly_uop_softsign(PolyCtx *ctx, PolyUOp *x) {
  return poly_uop_div(
      ctx, x,
      poly_uop_elementwise_scalar_binop(ctx, POLY_OP_ADD, poly_uop_abs(ctx, x), POLY_INT32, 1, true)
  );
}

PolyUOp *poly_uop_erf(PolyCtx *ctx, PolyUOp *x) {
  PolyDType result;
  x = special_float(ctx, x, &result);
  if (!x) return NULL;
  PolyDType dt = x->dtype;
  PolyUOp *a = poly_uop_abs(ctx, x);
  PolyUOp *small = poly_uop_alu2(ctx, POLY_OP_CMPLT, a, cdt(ctx, dt, 1));
  PolyUOp *center = erf_center(ctx, poly_uop_where(ctx, small, x, cdt(ctx, dt, 0)));
  PolyUOp *tail = erfc_tail(ctx, poly_uop_where(ctx, small, cdt(ctx, dt, 1), a));
  PolyUOp *value = poly_uop_sub(ctx, cdt(ctx, dt, 1), tail);
  value = poly_uop_where(
      ctx, poly_uop_alu2(ctx, POLY_OP_CMPLT, x, cdt(ctx, dt, 0)),
      poly_uop_elementwise_neg(ctx, value), value
  );
  return poly_uop_cast(ctx, poly_uop_where(ctx, small, center, value), result);
}

PolyUOp *poly_uop_erfc(PolyCtx *ctx, PolyUOp *x) {
  PolyDType result;
  x = special_float(ctx, x, &result);
  if (!x) return NULL;
  PolyDType dt = x->dtype;
  PolyUOp *a = poly_uop_abs(ctx, x);
  PolyUOp *small = poly_uop_alu2(ctx, POLY_OP_CMPLT, a, cdt(ctx, dt, 1));
  PolyUOp *center = erf_center(ctx, poly_uop_where(ctx, small, x, cdt(ctx, dt, 0)));
  PolyUOp *tail = erfc_tail(ctx, poly_uop_where(ctx, small, cdt(ctx, dt, 1), a));
  tail = poly_uop_where(
      ctx, poly_uop_alu2(ctx, POLY_OP_CMPLT, x, cdt(ctx, dt, 0)),
      poly_uop_sub(ctx, cdt(ctx, dt, 2), tail), tail
  );
  return poly_uop_cast(
      ctx, poly_uop_where(ctx, small, poly_uop_sub(ctx, cdt(ctx, dt, 1), center), tail), result
  );
}

PolyUOp *poly_uop_log_ndtr(PolyCtx *ctx, PolyUOp *x) {
  PolyDType result;
  x = special_float(ctx, x, &result);
  if (!x) return NULL;
  PolyDType dt = x->dtype;
  /* log Phi(x) = log(erfcx(-x/sqrt(2))) - x*x/2 - log(2).
   * In the other tail use log1p(-erfc(x/sqrt(2))/2), not log(1-p).
   * Select safe arguments before evaluation to preserve interior autograd. */
  PolyUOp *negative = poly_uop_alu2(ctx, POLY_OP_CMPLT, x, cdt(ctx, dt, -1.4142135623730951));
  PolyUOp *lower = poly_uop_where(ctx, negative, x, cdt(ctx, dt, -1.4142135623730951));
  PolyUOp *a = poly_uop_mul(ctx, lower, cdt(ctx, dt, -0.70710678118654752440));
  PolyUOp *square = poly_uop_mul(ctx, poly_uop_mul(ctx, lower, cdt(ctx, dt, 0.5)), lower);
  PolyUOp *tail = poly_uop_sub(
      ctx, poly_uop_sub(ctx, erfc_tail_factor(ctx, a, true), square),
      cdt(ctx, dt, 0.69314718055994530942)
  );
  PolyUOp *upper = poly_uop_where(ctx, negative, cdt(ctx, dt, 0), x);
  PolyUOp *probability =
      poly_uop_erfc(ctx, poly_uop_mul(ctx, upper, cdt(ctx, dt, 0.70710678118654752440)));
  PolyUOp *regular = poly_uop_log1p(ctx, poly_uop_mul(ctx, probability, cdt(ctx, dt, -0.5)));
  return poly_uop_cast(ctx, poly_uop_where(ctx, negative, tail, regular), result);
}

/* Keep centered q and tail probability separate. Forming 2*p-1 loses small
 * ndtri probabilities; forming (1+x)/2 loses small erfinv arguments. */
static PolyUOp *normal_inverse(PolyCtx *ctx, PolyUOp *q, PolyUOp *tail) {
  const double p0[] = {
      -5.99633501014107895267E1, 9.80010754185999661536E1, -5.66762857469070293439E1,
      1.39312609387279679503E1, -1.23916583867381258016E0};
  const double q0[] = {1.95448858338141759834E0, 4.67627912898881538453E0,
                       8.63602421390890590575E1, -2.25462687854119370527E2,
                       2.00260212380060660359E2, -8.20372256168333339912E1,
                       1.59056225126211695515E1, -1.18331621121330003142E0};
  const double p1[] = {
      4.05544892305962419923E0,   3.15251094599893866154E1,   5.71628192246421288162E1,
      4.40805073893200834700E1,   1.46849561928858024014E1,   2.18663306850790267539E0,
      -1.40256079171354495875E-1, -3.50424626827848203418E-2, -8.57456785154685413611E-4};
  const double q1[] = {1.57799883256466749731E1,   4.53907635128879210584E1,
                       4.13172038254672030440E1,   1.50425385692907503408E1,
                       2.50464946208309415979E0,   -1.42182922854787788574E-1,
                       -3.80806407691578277194E-2, -9.33259480895457427372E-4};
  const double p2[] = {
      3.23774891776946035970E0,  6.91522889068984211695E0,  3.93881025292474443415E0,
      1.33303460815807542389E0,  2.01485389549179081538E-1, 1.23716634817820021358E-2,
      3.01581553508235416007E-4, 2.65806974686737550832E-6, 6.23974539184983293730E-9};
  const double q2[] = {6.02427039364742014255E0,  3.67983563856160859403E0,
                       1.37702099489081330271E0,  2.16236993594496635890E-1,
                       1.34204006088543189037E-2, 3.28014464682127739104E-4,
                       2.89247864745380683936E-6, 6.79019408009981274425E-9};
  PolyDType dt = q->dtype;
  PolyUOp *central = poly_uop_gt(ctx, tail, cdt(ctx, dt, 0.13533528323661269189));
  PolyUOp *y = poly_uop_where(ctx, central, q, cdt(ctx, dt, 0));
  PolyUOp *y2 = poly_uop_mul(ctx, y, y);
  PolyUOp *center = poly_uop_mul(
      ctx, cdt(ctx, dt, 2.50662827463100050242),
      poly_uop_add(
          ctx, y,
          poly_uop_mul(
              ctx, y,
              poly_uop_mul(
                  ctx, y2,
                  poly_uop_div(
                      ctx, special_polynomial(ctx, y2, p0, 5, false),
                      special_polynomial(ctx, y2, q0, 8, true)
                  )
              )
          )
      )
  );
  /* Ordered comparison: the shared <= composition negates >, so it also
   * accepts NaN. Do not replace a NaN probability with the safe tail input. */
  PolyUOp *endpoint = poly_uop_alu2(
      ctx, POLY_OP_OR, poly_uop_alu2(ctx, POLY_OP_CMPLT, tail, cdt(ctx, dt, 0)),
      poly_uop_eq(ctx, tail, cdt(ctx, dt, 0))
  );
  PolyUOp *p = poly_uop_where(ctx, central, cdt(ctx, dt, 0.1), tail);
  p = poly_uop_where(ctx, endpoint, cdt(ctx, dt, 0.1), p);
  PolyUOp *z =
      poly_uop_alu1(ctx, POLY_OP_SQRT, poly_uop_mul(ctx, cdt(ctx, dt, -2), poly_uop_log(ctx, p)));
  PolyUOp *r = poly_uop_div(ctx, cdt(ctx, dt, 1), z);
  PolyUOp *correction = poly_uop_where(
      ctx, poly_uop_alu2(ctx, POLY_OP_CMPLT, z, cdt(ctx, dt, 8)),
      poly_uop_div(
          ctx, special_polynomial(ctx, r, p1, 9, false), special_polynomial(ctx, r, q1, 8, true)
      ),
      poly_uop_div(
          ctx, special_polynomial(ctx, r, p2, 9, false), special_polynomial(ctx, r, q2, 8, true)
      )
  );
  PolyUOp *value = poly_uop_sub(
      ctx, z, poly_uop_mul(ctx, r, poly_uop_add(ctx, poly_uop_log(ctx, z), correction))
  );
  value =
      poly_uop_where(ctx, poly_uop_eq(ctx, tail, cdt(ctx, dt, 0)), cdt(ctx, dt, INFINITY), value);
  value = poly_uop_where(
      ctx, poly_uop_alu2(ctx, POLY_OP_CMPLT, q, cdt(ctx, dt, 0)),
      poly_uop_elementwise_neg(ctx, value), value
  );
  value = poly_uop_where(ctx, central, center, value);
  return poly_uop_where(
      ctx, poly_uop_alu2(ctx, POLY_OP_CMPLT, tail, cdt(ctx, dt, 0)), cdt(ctx, dt, NAN), value
  );
}

PolyUOp *poly_uop_erfinv(PolyCtx *ctx, PolyUOp *x) {
  PolyDType result;
  x = special_float(ctx, x, &result);
  if (!x) return NULL;
  PolyUOp *q = poly_uop_mul(ctx, x, cdt(ctx, x->dtype, 0.5));
  PolyUOp *tail = poly_uop_mul(
      ctx, cdt(ctx, x->dtype, 0.5), poly_uop_sub(ctx, cdt(ctx, x->dtype, 1), poly_uop_abs(ctx, x))
  );
  return poly_uop_cast(
      ctx,
      poly_uop_mul(ctx, cdt(ctx, x->dtype, 0.70710678118654752440), normal_inverse(ctx, q, tail)),
      result
  );
}

PolyUOp *poly_uop_ndtri(PolyCtx *ctx, PolyUOp *x) {
  PolyDType result;
  x = special_float(ctx, x, &result);
  if (!x) return NULL;
  PolyUOp *q = poly_uop_sub(ctx, x, cdt(ctx, x->dtype, 0.5));
  PolyUOp *tail = poly_uop_where(
      ctx, poly_uop_gt(ctx, x, cdt(ctx, x->dtype, 0.5)),
      poly_uop_sub(ctx, cdt(ctx, x->dtype, 1), x), x
  );
  return poly_uop_cast(ctx, normal_inverse(ctx, q, tail), result);
}

PolyUOp *poly_uop_digamma(PolyCtx *ctx, PolyUOp *x) {
  if (!ctx || !x) return NULL;
  if (!poly_dtype_is_float(x->dtype)) x = poly_uop_cast(ctx, x, POLY_FLOAT32);
  PolyDType dt = x->dtype;
  /* DLMF 5.5.4 reflection, 5.5.2 recurrence, 5.11.2 Bernoulli series.
   * Six terms at x>=10 leave a truncation error below 9e-16. */
  PolyUOp *reflect = poly_uop_alu2(ctx, POLY_OP_CMPLT, x, cdt(ctx, dt, 0));
  PolyUOp *xx = poly_uop_where(ctx, reflect, poly_uop_sub(ctx, cdt(ctx, dt, 1), x), x);
  PolyUOp *acc = cdt(ctx, dt, 0.0);
  for (int i = 0; i < 10; i++) {
    PolyUOp *cond = poly_uop_alu2(ctx, POLY_OP_CMPLT, xx, cdt(ctx, dt, 10.0));
    PolyUOp *inv = poly_uop_alu1(ctx, POLY_OP_RECIPROCAL, xx);
    acc = poly_uop_alu3(ctx, POLY_OP_WHERE, cond, poly_uop_alu2(ctx, POLY_OP_SUB, acc, inv), acc);
    xx = poly_uop_alu3(
        ctx, POLY_OP_WHERE, cond, poly_uop_alu2(ctx, POLY_OP_ADD, xx, cdt(ctx, dt, 1.0)), xx
    );
  }
  PolyUOp *inv = poly_uop_alu1(ctx, POLY_OP_RECIPROCAL, xx);
  PolyUOp *inv2 = poly_uop_alu2(ctx, POLY_OP_MUL, inv, inv);
  const double c[] = {691.0 / 32760, -1.0 / 132, 1.0 / 240, -1.0 / 252, 1.0 / 120, -1.0 / 12};
  PolyUOp *p = cdt(ctx, dt, c[0]);
  for (int i = 1; i < 6; i++)
    p = poly_uop_add(ctx, poly_uop_mul(ctx, p, inv2), cdt(ctx, dt, c[i]));
  PolyUOp *out = poly_uop_add(
      ctx, acc,
      poly_uop_add(
          ctx, poly_uop_log(ctx, xx),
          poly_uop_add(ctx, poly_uop_mul(ctx, cdt(ctx, dt, -0.5), inv), poly_uop_mul(ctx, inv2, p))
      )
  );
  PolyUOp *r = poly_uop_where(ctx, reflect, x, cdt(ctx, dt, 0.25));
  r = gamma_fraction(ctx, r);
  PolyUOp *angle = poly_uop_mul(ctx, cdt(ctx, dt, M_PI), r);
  PolyUOp *cot = poly_uop_div(ctx, poly_uop_cos(ctx, angle), poly_uop_sin(ctx, angle));
  PolyUOp *correction = poly_uop_where(
      ctx, poly_uop_eq(ctx, r, cdt(ctx, dt, 0)), cdt(ctx, dt, NAN),
      poly_uop_mul(ctx, cdt(ctx, dt, M_PI), cot)
  );
  return poly_uop_where(ctx, reflect, poly_uop_sub(ctx, out, correction), out);
}

PolyUOp *poly_uop_lgamma(PolyCtx *ctx, PolyUOp *x) {
  if (!ctx || !x) return NULL;
  if (!poly_dtype_is_float(x->dtype)) x = poly_uop_cast(ctx, x, POLY_FLOAT32);
  /* Differentiate the forward expression itself, including higher orders. */
  return poly_lgamma_forward_lanczos(ctx, x, x->dtype);
}

PolyUOp *poly_uop_sigmoid(PolyCtx *ctx, PolyUOp *x) {
  /* Pinned sigmoid:
   * (1 + (x * (-1/log(2))).exp2()).reciprocal()
   * (mixin/elementwise.py:667-679). UOp.ufix keeps float16/float64 receivers
   * in their own dtype; integer/bool receivers promote at the multiplication. */
  if (!ctx || !x) return NULL;
  PolyUOp *scaled =
      poly_uop_elementwise_scalar_binop(ctx, POLY_OP_MUL, x, POLY_FLOAT32, -1.0 / M_LN2, false);
  PolyUOp *e = scaled ? poly_uop_alu1(ctx, POLY_OP_EXP2, scaled) : NULL;
  PolyUOp *denom =
      e ? poly_uop_elementwise_scalar_binop(ctx, POLY_OP_ADD, e, POLY_INT32, 1.0, true) : NULL;
  return denom ? poly_uop_alu1(ctx, POLY_OP_RECIPROCAL, denom) : NULL;
}

PolyUOp *poly_uop_tanh(PolyCtx *ctx, PolyUOp *x) {
  if (!ctx || !x) return NULL;
  PolyDType result_dt, dt;
  if (!poly_dtype_least_upper_float(x->dtype, &result_dt) ||
      !poly_dtype_least_upper(result_dt, POLY_FLOAT32, &dt))
    return NULL;
  x = poly_dtype_eq(x->dtype, dt) ? x : poly_uop_cast(ctx, x, dt);
  if (!x) return NULL;
  PolyUOp *zero = cdt(ctx, dt, 0), *one = cdt(ctx, dt, 1);
  PolyUOp *negative = poly_uop_alu2(ctx, POLY_OP_CMPLT, x, zero);
  PolyUOp *a = poly_uop_where(ctx, negative, poly_uop_elementwise_neg(ctx, x), x);
  PolyUOp *small = poly_uop_alu2(ctx, POLY_OP_CMPLT, a, cdt(ctx, dt, 0.25));

  /* PG-DIV-018: 2*sigmoid(2*x)-1 cancels near zero. The odd Taylor
   * polynomial has truncation error <2.2e-9 (f32) / <5.9e-19 (f64)
   * on |x|<1/4. Coefficients follow y'=1-y*y, y(0)=0. Multiply by x
   * last to preserve relative accuracy, including subnormals and -0. */
  static const double coefficients[] = {
      -1.0 / 3,
      2.0 / 15,
      -17.0 / 315,
      62.0 / 2835,
      -1382.0 / 155925,
      21844.0 / 6081075,
      -929569.0 / 638512875,
      6404582.0 / 10854718875,
      -443861162.0 / 1856156927625,
      18888466084.0 / 194896477400625};
  int n = poly_dtype_eq(dt, POLY_FLOAT64) ? 10 : 4;
  /* WHERE is eager on some backends: bound both branches before arithmetic
   * so unselected infinities cannot overflow or contaminate gradients. */
  PolyUOp *s = poly_uop_where(ctx, small, x, zero);
  PolyUOp *z = poly_uop_mul(ctx, s, s), *p = cdt(ctx, dt, coefficients[n - 1]);
  for (int i = n - 2; i >= 0; i--)
    p = poly_uop_add(ctx, cdt(ctx, dt, coefficients[i]), poly_uop_mul(ctx, z, p));
  PolyUOp *near = poly_uop_mul(ctx, s, poly_uop_add(ctx, one, poly_uop_mul(ctx, z, p)));

  /* tanh(20) rounds to 1 even in f64. A negative exponential never
   * overflows; away from zero the subtraction is well conditioned. Select
   * the limit only above 20, so NaN still reaches the arithmetic branch. */
  PolyUOp *bounded = poly_uop_where(
      ctx, poly_uop_alu2(ctx, POLY_OP_CMPLT, cdt(ctx, dt, 20), a), cdt(ctx, dt, 20), a
  );
  PolyUOp *e = poly_uop_exp(ctx, poly_uop_mul(ctx, cdt(ctx, dt, -2), bounded));
  PolyUOp *far = poly_uop_div(ctx, poly_uop_sub(ctx, one, e), poly_uop_add(ctx, one, e));
  far = poly_uop_where(ctx, negative, poly_uop_elementwise_neg(ctx, far), far);
  PolyUOp *out = poly_uop_where(ctx, small, near, far);
  /* Shader compilers may fold x!=x under finite-math assumptions. Classify
   * the IEEE bits so saturation cannot silently turn an input NaN into 1. */
  bool f64 = poly_dtype_eq(dt, POLY_FLOAT64);
  PolyDType bits_dt = f64 ? POLY_UINT64 : POLY_UINT32;
  PolyUOp *bits = poly_uop_bitcast(ctx, x, bits_dt);
  PolyUOp *magnitude = poly_uop_alu2(
      ctx, POLY_OP_AND, bits,
      poly_uop_const(ctx, poly_arg_int(f64 ? INT64_MAX : INT32_MAX), bits_dt)
  );
  PolyUOp *nan = poly_uop_alu2(
      ctx, POLY_OP_CMPLT,
      poly_uop_const(
          ctx, poly_arg_int(f64 ? INT64_C(0x7ff0000000000000) : INT64_C(0x7f800000)), bits_dt
      ),
      magnitude
  );
  out = poly_uop_where(ctx, nan, x, out);
  return poly_dtype_eq(dt, result_dt) ? out : poly_uop_cast(ctx, out, result_dt);
}

PolyUOp *poly_uop_abs(PolyCtx *ctx, PolyUOp *x) {
  /* abs(x) = x * sign(x) */
  return poly_uop_alu2(ctx, POLY_OP_MUL, x, poly_uop_sign(ctx, x));
}

PolyUOp *poly_uop_sign(PolyCtx *ctx, PolyUOp *x) {
  /* Pinned sign uses typed, shaped const_like branches; NaN selects +1. */
  PolyUOp *nonzero = poly_uop_elementwise_scalar_binop(ctx, POLY_OP_CMPNE, x, POLY_INT32, 0, false);
  PolyUOp *negative =
      poly_uop_elementwise_scalar_binop(ctx, POLY_OP_CMPLT, x, POLY_INT32, 0, false);
  PolyUOp *signed_one = poly_uop_where(
      ctx, negative, poly_uop_const_like(ctx, x, poly_arg_int(-1)),
      poly_uop_const_like(ctx, x, poly_arg_int(1))
  );
  return poly_uop_where(ctx, nonzero, signed_one, poly_uop_const_like(ctx, x, poly_arg_int(0)));
}

PolyUOp *poly_uop_square(PolyCtx *ctx, PolyUOp *x) {
  return poly_uop_alu2(ctx, POLY_OP_MUL, x, x);
}

PolyUOp *poly_uop_rsqrt(PolyCtx *ctx, PolyUOp *x) {
  return poly_uop_alu1(ctx, POLY_OP_RECIPROCAL, poly_uop_alu1(ctx, POLY_OP_SQRT, x));
}

PolyUOp *poly_uop_ceil(PolyCtx *ctx, PolyUOp *x) {
  /* ceil(x) = (x > (b=trunc(x))).where(b+1, b) */
  PolyUOp *b = poly_uop_alu1(ctx, POLY_OP_TRUNC, x);
  PolyUOp *cond = poly_uop_binop(ctx, POLY_OP_CMPLT, b, x);
  return poly_uop_where(
      ctx, cond, poly_uop_elementwise_scalar_binop(ctx, POLY_OP_ADD, b, POLY_INT32, 1, false), b
  );
}

PolyUOp *poly_uop_floor(PolyCtx *ctx, PolyUOp *x) {
  /* floor(x) = (x < (b=trunc(x))).where(b-1, b) */
  PolyUOp *b = poly_uop_alu1(ctx, POLY_OP_TRUNC, x);
  PolyUOp *cond = poly_uop_binop(ctx, POLY_OP_CMPLT, x, b);
  return poly_uop_where(
      ctx, cond, poly_uop_elementwise_scalar_binop(ctx, POLY_OP_SUB, b, POLY_INT32, 1, false), b
  );
}

PolyUOp *poly_uop_round(PolyCtx *ctx, PolyUOp *x) {
  /* Pinned ElementwiseMixin.round: the truncated integer is even precisely
   * when b = trunc(x)/2 equals trunc(b). Comparing two truncations loses ties.
   * Scalar arithmetic must promote integer inputs, as _broadcasted does. */
  PolyUOp *half = poly_uop_const_exact_float(ctx, POLY_WEAKFLOAT, 0.5);
  PolyUOp *two = poly_uop_const_exact_float(ctx, POLY_WEAKFLOAT, 2.0);
  PolyUOp *b = poly_uop_alu1(ctx, POLY_OP_TRUNC, x);
  PolyUOp *x_gt_0 = poly_uop_elementwise_scalar_binop(ctx, POLY_OP_CMPLT, x, POLY_INT32, 0, true);
  PolyUOp *b_half = poly_uop_div(ctx, b, two);
  PolyUOp *trunc_b_half = poly_uop_alu1(ctx, POLY_OP_TRUNC, b_half);
  PolyUOp *halves_eq = poly_uop_eq(ctx, trunc_b_half, b_half);
  PolyUOp *cond = poly_uop_eq(ctx, x_gt_0, halves_eq);
  return poly_uop_where(
      ctx, cond, poly_uop_ceil(ctx, poly_uop_sub(ctx, x, half)),
      poly_uop_floor(ctx, poly_uop_add(ctx, x, half))
  );
}

PolyUOp *poly_uop_isinf(PolyCtx *ctx, PolyUOp *x) {
  return poly_uop_isinf_flags(ctx, x, true, true);
}

PolyUOp *poly_uop_isinf_flags(
    PolyCtx *ctx,
    PolyUOp *x,
    bool detect_positive,
    bool detect_negative
) {
  PolyUOp *positive =
      poly_uop_eq(ctx, x, poly_uop_const_exact_float(ctx, POLY_WEAKFLOAT, INFINITY));
  PolyUOp *negative =
      poly_uop_eq(ctx, x, poly_uop_const_exact_float(ctx, POLY_WEAKFLOAT, -INFINITY));
  positive = poly_uop_elementwise_scalar_binop(
      ctx, POLY_OP_MUL, positive, POLY_BOOL, detect_positive, false
  );
  negative = poly_uop_elementwise_scalar_binop(
      ctx, POLY_OP_MUL, negative, POLY_BOOL, detect_negative, false
  );
  return poly_uop_add(ctx, positive, negative);
}

PolyUOp *poly_uop_isnan(PolyCtx *ctx, PolyUOp *x) {
  /* isnan(x) = (x != x) -- IEEE 754 */
  return poly_uop_alu2(ctx, POLY_OP_CMPNE, x, x);
}

PolyUOp *poly_uop_isfinite(PolyCtx *ctx, PolyUOp *x) {
  return poly_uop_logical_not(
      ctx, poly_uop_binop(ctx, POLY_OP_OR, poly_uop_isinf(ctx, x), poly_uop_isnan(ctx, x))
  );
}

PolyUOp *poly_uop_isclose(
    PolyCtx *ctx,
    PolyUOp *a,
    PolyUOp *b,
    PolyUOp *rtol,
    PolyUOp *atol,
    bool equal_nan
) {
  PolyUOp *tolerance = poly_uop_add(ctx, atol, poly_uop_mul(ctx, rtol, poly_uop_abs(ctx, b)));
  PolyUOp *finite =
      poly_uop_binop(ctx, POLY_OP_AND, poly_uop_isfinite(ctx, a), poly_uop_isfinite(ctx, b));
  finite = poly_uop_binop(
      ctx, POLY_OP_AND, finite,
      poly_uop_le(ctx, poly_uop_abs(ctx, poly_uop_sub(ctx, a, b)), tolerance)
  );
  PolyUOp *infinite =
      poly_uop_binop(ctx, POLY_OP_OR, poly_uop_isinf(ctx, a), poly_uop_isinf(ctx, b));
  infinite = poly_uop_binop(ctx, POLY_OP_AND, infinite, poly_uop_eq(ctx, a, b));
  PolyUOp *nan = poly_uop_binop(ctx, POLY_OP_AND, poly_uop_isnan(ctx, a), poly_uop_isnan(ctx, b));
  nan = poly_uop_elementwise_scalar_binop(ctx, POLY_OP_AND, nan, POLY_BOOL, equal_nan, false);
  return poly_uop_binop(ctx, POLY_OP_OR, poly_uop_binop(ctx, POLY_OP_OR, finite, infinite), nan);
}

PolyUOp *poly_uop_copysign(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  if (!poly_uop_broadcasted_pair(ctx, &a, &b)) return NULL;
  PolyUOp *negative =
      poly_uop_elementwise_scalar_binop(ctx, POLY_OP_CMPLT, b, POLY_INT32, 0, false);
  PolyUOp *signbit = poly_uop_elementwise_scalar_binop(
      ctx, POLY_OP_CMPLT, poly_uop_alu1(ctx, POLY_OP_RECIPROCAL, b), POLY_INT32, 0, false
  );
  PolyUOp *magnitude = poly_uop_abs(ctx, a);
  return poly_uop_where(
      ctx, poly_uop_binop(ctx, POLY_OP_OR, negative, signbit),
      poly_uop_elementwise_neg(ctx, magnitude), magnitude
  );
}

PolyUOp *poly_uop_lerp(
    PolyCtx *ctx,
    PolyUOp *x,
    PolyUOp *end,
    PolyUOp *weight,
    bool scalar_weight
) {
  if (!ctx || !x || !end || !weight) return NULL;
  PolyUOp *difference = poly_uop_sub(ctx, end, x);
  /* The pinned uint8 path is selected by host-scalar versus Tensor provenance,
   * not by weight dtype: a weak Tensor is still a Tensor. */
  if (poly_dtype_eq(x->dtype, POLY_UINT8) && !scalar_weight) {
    PolyUOp *scaled =
        poly_uop_elementwise_scalar_binop(ctx, POLY_OP_MUL, weight, POLY_INT32, 128, false);
    PolyUOp *wi = poly_uop_cast(
        ctx, poly_uop_elementwise_scalar_binop(ctx, POLY_OP_ADD, scaled, POLY_FLOAT32, 0.5, false),
        POLY_INT16
    );
    PolyUOp *offset = poly_uop_mul(ctx, poly_uop_cast(ctx, difference, POLY_INT8), wi);
    offset = poly_uop_cast(
        ctx, poly_uop_elementwise_scalar_binop(ctx, POLY_OP_ADD, offset, POLY_INT32, 64, false),
        POLY_UINT16
    );
    offset = poly_uop_elementwise_scalar_binop(ctx, POLY_OP_SHR, offset, POLY_INT32, 7, false);
    return poly_uop_cast(ctx, poly_uop_add(ctx, x, offset), POLY_UINT8);
  }
  return poly_uop_add(ctx, x, poly_uop_mul(ctx, difference, weight));
}

/* Activations */

PolyUOp *poly_uop_relu(PolyCtx *ctx, PolyUOp *x) {
  /* Preserve Python integer scalar provenance; float zero would promote an
   * integer input and round values above the floating mantissa's precision. */
  PolyUOp *cond = poly_uop_elementwise_scalar_binop(ctx, POLY_OP_CMPLT, x, POLY_INT32, 0, true);
  return poly_uop_where(ctx, cond, x, poly_uop_const_exact_int(ctx, POLY_WEAKINT, 0));
}

PolyUOp *poly_uop_relu6(PolyCtx *ctx, PolyUOp *x) {
  PolyUOp *r = poly_uop_relu(ctx, x), *six = poly_uop_const_int(ctx, 6);
  return poly_uop_where(ctx, poly_uop_binop(ctx, POLY_OP_CMPLT, r, six), r, six);
}

PolyUOp *poly_uop_leaky_relu(PolyCtx *ctx, PolyUOp *x, double neg_slope) {
  return poly_uop_leaky_relu_tensor(
      ctx, x, poly_uop_const_exact_float(ctx, POLY_WEAKFLOAT, neg_slope)
  );
}

PolyUOp *poly_uop_leaky_relu_tensor(PolyCtx *ctx, PolyUOp *x, PolyUOp *slope) {
  PolyUOp *cond = poly_uop_elementwise_scalar_binop(ctx, POLY_OP_CMPLT, x, POLY_INT32, 0, false);
  PolyUOp *scaled = poly_uop_mul(ctx, slope, x);
  return poly_uop_where(ctx, cond, scaled, x);
}

PolyUOp *poly_uop_gelu(PolyCtx *ctx, PolyUOp *x) {
  /* Pinned tanh GELU:
   * 0.5*x*(1 + (sqrt(2/pi)*(x + 0.044715*x**3)).tanh())
   * (mixin/elementwise.py:761-776). Keep every source operation's own ufix
   * and promotion boundary: integer x**3 remains integer until multiplied by
   * the floating coefficient. */
  if (!ctx || !x) return NULL;
  PolyUOp *half_x = poly_uop_elementwise_scalar_binop(ctx, POLY_OP_MUL, x, POLY_FLOAT32, 0.5, true);
  PolyUOp *x3 = poly_uop_elementwise_scalar_binop(ctx, POLY_OP_POW, x, POLY_INT32, 3.0, false);
  PolyUOp *cubic =
      x3 ? poly_uop_elementwise_scalar_binop(ctx, POLY_OP_MUL, x3, POLY_FLOAT32, 0.044715, true)
         : NULL;
  PolyUOp *inner = cubic ? poly_uop_binop(ctx, POLY_OP_ADD, x, cubic) : NULL;
  PolyUOp *scaled = inner ? poly_uop_elementwise_scalar_binop(
                                ctx, POLY_OP_MUL, inner, POLY_FLOAT32, sqrt(2.0 / M_PI), true
                            )
                          : NULL;
  PolyUOp *tanh = scaled ? poly_uop_tanh(ctx, scaled) : NULL;
  PolyUOp *one_plus =
      tanh ? poly_uop_elementwise_scalar_binop(ctx, POLY_OP_ADD, tanh, POLY_INT32, 1.0, true)
           : NULL;
  return half_x && one_plus ? poly_uop_binop(ctx, POLY_OP_MUL, half_x, one_plus) : NULL;
}

PolyUOp *poly_uop_quick_gelu(PolyCtx *ctx, PolyUOp *x) {
  /* Pinned quick_gelu is self * (self * 1.702).sigmoid()
   * (mixin/elementwise.py:751-759). */
  if (!ctx || !x) return NULL;
  PolyUOp *inner =
      poly_uop_elementwise_scalar_binop(ctx, POLY_OP_MUL, x, POLY_FLOAT32, 1.702, false);
  PolyUOp *sigmoid = inner ? poly_uop_sigmoid(ctx, inner) : NULL;
  return sigmoid ? poly_uop_binop(ctx, POLY_OP_MUL, x, sigmoid) : NULL;
}

PolyUOp *poly_uop_silu(PolyCtx *ctx, PolyUOp *x) {
  PolyUOp *sigmoid = poly_uop_sigmoid(ctx, x);
  return sigmoid ? poly_uop_binop(ctx, POLY_OP_MUL, x, sigmoid) : NULL;
}

PolyUOp *poly_uop_elu(PolyCtx *ctx, PolyUOp *x, double alpha) {
  return poly_uop_elu_tensor(ctx, x, poly_uop_const_exact_float(ctx, POLY_WEAKFLOAT, alpha));
}

PolyUOp *poly_uop_elu_tensor(PolyCtx *ctx, PolyUOp *x, PolyUOp *alpha) {
  /* Bound the unselected exponential too: WHERE gradients multiply it by 0. */
  PolyUOp *bounded = poly_uop_sub(ctx, x, poly_uop_relu(ctx, x));
  PolyUOp *tail = poly_uop_elementwise_scalar_binop(
      ctx, POLY_OP_SUB, poly_uop_exp(ctx, bounded), POLY_INT32, 1, false
  );
  return poly_uop_where(
      ctx, poly_uop_gt(ctx, x, poly_uop_const_int(ctx, 0)), x, poly_uop_mul(ctx, alpha, tail)
  );
}

/* ElementwiseMixin.softplus/logaddexp, with the pinned floating beta. */
PolyUOp *poly_uop_softplus(PolyCtx *ctx, PolyUOp *x, double beta) {
  if (!ctx || !x || beta == 0) return NULL;
  return poly_uop_softplus_tensor(
      ctx, x, poly_uop_const_exact_float(ctx, POLY_WEAKFLOAT, beta),
      poly_uop_const_exact_float(ctx, POLY_WEAKFLOAT, 1.0 / beta)
  );
}

PolyUOp *poly_uop_logaddexp(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  if (!poly_uop_broadcasted_pair(ctx, &a, &b)) return NULL;
  PolyUOp *m = poly_uop_maximum(ctx, a, b);
  m = poly_uop_where(ctx, poly_uop_isfinite(ctx, m), m, poly_uop_const_int(ctx, 0));
  PolyUOp *sum = poly_uop_add(
      ctx, poly_uop_exp(ctx, poly_uop_sub(ctx, a, m)), poly_uop_exp(ctx, poly_uop_sub(ctx, b, m))
  );
  return poly_uop_add(ctx, poly_uop_log(ctx, sum), m);
}

/* Python computes scalar 1/beta before graph construction. Keep that host
 * constant distinct from Tensor beta's reciprocal program. */
PolyUOp *poly_uop_softplus_tensor(PolyCtx *ctx, PolyUOp *x, PolyUOp *beta, PolyUOp *inverse) {
  if (!inverse) inverse = poly_uop_div(ctx, poly_uop_const_int(ctx, 1), beta);
  return poly_uop_mul(
      ctx, inverse,
      poly_uop_logaddexp(
          ctx, poly_uop_mul(ctx, x, beta), poly_uop_const_exact_float(ctx, POLY_WEAKFLOAT, 0.0)
      )
  );
}

PolyUOp *poly_uop_logsigmoid(PolyCtx *ctx, PolyUOp *x) {
  PolyUOp *out = poly_uop_softplus(ctx, poly_uop_elementwise_neg(ctx, x), 1.0);
  return out ? poly_uop_elementwise_neg(ctx, out) : NULL;
}

PolyUOp *poly_uop_mish(PolyCtx *ctx, PolyUOp *x) {
  return poly_uop_mul(ctx, x, poly_uop_tanh(ctx, poly_uop_softplus(ctx, x, 1.0)));
}

PolyUOp *poly_uop_hardtanh(PolyCtx *ctx, PolyUOp *x, double min_val, double max_val) {
  return poly_uop_clamp(ctx, x, min_val, max_val);
}

PolyUOp *poly_uop_hardswish(PolyCtx *ctx, PolyUOp *x) {
  PolyUOp *shifted = poly_uop_elementwise_scalar_binop(ctx, POLY_OP_ADD, x, POLY_INT32, 3, false);
  PolyUOp *product = poly_uop_mul(ctx, x, poly_uop_relu6(ctx, shifted));
  return poly_uop_elementwise_scalar_binop(
      ctx, POLY_OP_MUL, product, POLY_FLOAT32, 1.0 / 6.0, false
  );
}

PolyUOp *poly_uop_hardsigmoid(PolyCtx *ctx, PolyUOp *x) {
  return poly_uop_hardsigmoid_tensor(
      ctx, x, poly_uop_const_exact_float(ctx, POLY_WEAKFLOAT, 1.0 / 6.0),
      poly_uop_const_exact_float(ctx, POLY_WEAKFLOAT, 0.5)
  );
}

PolyUOp *poly_uop_hardsigmoid_tensor(PolyCtx *ctx, PolyUOp *x, PolyUOp *alpha, PolyUOp *beta) {
  PolyUOp *y = poly_uop_relu(ctx, poly_uop_add(ctx, poly_uop_mul(ctx, alpha, x), beta));
  PolyUOp *one = poly_uop_const_int(ctx, 1);
  return poly_uop_where(ctx, poly_uop_binop(ctx, POLY_OP_CMPLT, y, one), y, one);
}

PolyUOp *poly_uop_ne(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  if (!poly_uop_broadcasted_pair(ctx, &a, &b)) return NULL;
  return poly_uop_alu2(ctx, POLY_OP_CMPNE, a, b);
}

PolyUOp *poly_uop_gt(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  if (!poly_uop_broadcasted_pair(ctx, &a, &b)) return NULL;
  return poly_uop_alu2(ctx, POLY_OP_CMPLT, b, a);
}

PolyUOp *poly_uop_ge(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  if (!poly_uop_broadcasted_pair(ctx, &a, &b)) return NULL;
  PolyUOp *lt = poly_uop_alu2(ctx, POLY_OP_CMPLT, a, b);
  return poly_uop_logical_not(ctx, lt);
}

PolyUOp *poly_uop_le(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  if (!poly_uop_broadcasted_pair(ctx, &a, &b)) return NULL;
  PolyUOp *gt = poly_uop_alu2(ctx, POLY_OP_CMPLT, b, a);
  return poly_uop_logical_not(ctx, gt);
}

PolyUOp *poly_uop_maximum(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  if (!poly_uop_broadcasted_pair(ctx, &a, &b)) return NULL;
  return poly_uop_alu2(ctx, POLY_OP_MAX, a, b);
}

/* Pinned ElementwiseMixin._inverse is ordinary broadcasted negation for
 * floating values and bitwise-not for integer/bool values
 * (mixin/elementwise.py:57-69,131-143,379-393). */
PolyUOp *poly_uop_elementwise_inverse(PolyCtx *ctx, PolyUOp *x) {
  if (!ctx || !x) return NULL;
  PolyDType dt = x->dtype;
  if (poly_dtype_is_float(dt)) {
    PolyUOp *minus_one = poly_uop_const_typed(ctx, POLY_WEAKFLOAT, -1.0);
    return minus_one ? poly_uop_alu2(ctx, POLY_OP_MUL, x, minus_one) : NULL;
  }
  if (poly_dtype_is_bool(dt)) return poly_uop_logical_not(ctx, x);
  if (!poly_dtype_is_int(dt)) return NULL;
  PolyUOp *mask = NULL;
  if (poly_dtype_is_unsigned(dt)) {
    if (!poly_dtype_bound_const(ctx, dt, false, &mask)) return NULL;
    /* _inverse uses Python dtype.max / -1 literals, not typed Tensor
     * constants. Preserve the exact uint64 bound while weakening its dtype. */
    mask = poly_uop_const(ctx, mask->arg, POLY_WEAKINT);
  } else {
    mask = poly_uop_const_int(ctx, -1);
  }
  return mask ? poly_uop_alu2(ctx, POLY_OP_XOR, x, mask) : NULL;
}

PolyUOp *poly_uop_minimum(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  if (!poly_uop_broadcasted_pair(ctx, &a, &b)) return NULL;
  /* ElementwiseMixin.minimum uses XOR with dtype.const(min + max), even
   * for bool. Unary min's _inverse instead uses logical_not (CMPNE). */
  if (poly_dtype_is_bool(a->dtype)) {
    PolyUOp *mask = poly_uop_const_typed(ctx, POLY_BOOL, 1);
    if (!mask) return NULL;
    PolyUOp *left = poly_uop_alu2(ctx, POLY_OP_XOR, a, mask);
    PolyUOp *right = poly_uop_alu2(ctx, POLY_OP_XOR, b, mask);
    PolyUOp *maximum = left && right ? poly_uop_alu2(ctx, POLY_OP_MAX, left, right) : NULL;
    return maximum ? poly_uop_alu2(ctx, POLY_OP_XOR, maximum, mask) : NULL;
  }
  PolyUOp *inverse_a = poly_uop_elementwise_inverse(ctx, a);
  PolyUOp *inverse_b = poly_uop_elementwise_inverse(ctx, b);
  PolyUOp *maximum =
      (inverse_a && inverse_b) ? poly_uop_alu2(ctx, POLY_OP_MAX, inverse_a, inverse_b) : NULL;
  return maximum ? poly_uop_elementwise_inverse(ctx, maximum) : NULL;
}

PolyUOp *poly_uop_clamp(PolyCtx *ctx, PolyUOp *x, double lo, double hi) {
  return poly_uop_clamp_tensor(
      ctx, x, poly_uop_const_exact_float(ctx, POLY_WEAKFLOAT, lo),
      poly_uop_const_exact_float(ctx, POLY_WEAKFLOAT, hi)
  );
}

PolyUOp *poly_uop_clamp_tensor(PolyCtx *ctx, PolyUOp *x, PolyUOp *lo, PolyUOp *hi) {
  if (!lo && !hi) return NULL;
  PolyUOp *ret = lo ? poly_uop_where(ctx, poly_uop_binop(ctx, POLY_OP_CMPLT, x, lo), lo, x) : x;
  return hi ? poly_uop_where(ctx, poly_uop_binop(ctx, POLY_OP_CMPLT, hi, ret), hi, ret) : ret;
}

PolyUOp *poly_uop_detach(PolyCtx *ctx, PolyUOp *x) {
  return poly_uop1(ctx, POLY_OP_DETACH, x->dtype, x, poly_arg_none());
}
