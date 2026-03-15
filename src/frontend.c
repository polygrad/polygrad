/*
 * frontend.c — FFI-friendly helpers for language bindings
 *
 * Thin wrappers around the core API that avoid passing PolyArg/PolyDType
 * across FFI boundaries, plus poly_realize() which wraps the full
 * schedule → linearize → render → compile → execute pipeline.
 */

#define _GNU_SOURCE
#include "frontend.h"
#include "exec_plan.h"
#include "scheduler.h"
#include "rangeify.h"
#include "codegen.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_LN2
#define M_LN2 0.693147180559945309417
#endif

/* ── Op enum helpers ──────────────────────────────────────────────────── */

int poly_op_count(void) { return (int)POLY_OP_COUNT; }

/* ── Constants ────────────────────────────────────────────────────────── */

PolyUOp *poly_const_float(PolyCtx *ctx, double value) {
  return poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(value));
}

PolyUOp *poly_const_double(PolyCtx *ctx, double value) {
  return poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT64, poly_arg_float(value));
}

PolyUOp *poly_const_int(PolyCtx *ctx, int64_t value) {
  return poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(value));
}

PolyUOp *poly_const_typed(PolyCtx *ctx, PolyDType dt, double value) {
  if (poly_dtype_is_float(dt))
    return poly_uop0(ctx, POLY_OP_CONST, dt, poly_arg_float(value));
  return poly_uop0(ctx, POLY_OP_CONST, dt, poly_arg_int((int64_t)value));
}

/* ── ALU ops ──────────────────────────────────────────────────────────── */

PolyUOp *poly_alu1(PolyCtx *ctx, PolyOps op, PolyUOp *src) {
  return poly_uop1(ctx, op, src->dtype, src, poly_arg_none());
}

PolyUOp *poly_alu2(PolyCtx *ctx, PolyOps op, PolyUOp *a, PolyUOp *b) {
  /* Result dtype: prefer float if either input is float, else use a's dtype.
   * For comparisons, result is always BOOL. */
  PolyDType dt;
  if (op == POLY_OP_CMPLT || op == POLY_OP_CMPNE || op == POLY_OP_CMPEQ) {
    dt = POLY_BOOL;
  } else if (poly_dtype_is_float(a->dtype)) {
    dt = a->dtype;
  } else if (poly_dtype_is_float(b->dtype)) {
    dt = b->dtype;
  } else {
    dt = a->dtype;
  }
  return poly_uop2(ctx, op, dt, a, b, poly_arg_none());
}

PolyUOp *poly_alu3(PolyCtx *ctx, PolyOps op, PolyUOp *a, PolyUOp *b, PolyUOp *c) {
  /* For WHERE: result dtype is b's dtype (true branch) */
  PolyDType dt = (op == POLY_OP_WHERE) ? b->dtype : a->dtype;
  return poly_uop3(ctx, op, dt, a, b, c, poly_arg_none());
}

/* ── Graph construction ───────────────────────────────────────────────── */

PolyUOp *poly_store_val(PolyCtx *ctx, PolyUOp *buf, PolyUOp *value) {
  return poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, buf, value, poly_arg_none());
}

PolyUOp *poly_sink1(PolyCtx *ctx, PolyUOp *store) {
  return poly_uop(ctx, POLY_OP_SINK, POLY_VOID, &store, 1, poly_arg_none());
}

PolyUOp *poly_sink_n(PolyCtx *ctx, PolyUOp **stores, int n) {
  return poly_uop(ctx, POLY_OP_SINK, POLY_VOID, stores, n, poly_arg_none());
}

/* ── In-place assignment ──────────────────────────────────────────────── */

PolyUOp *poly_assign(PolyCtx *ctx, PolyUOp *target, PolyUOp *value) {
  PolyUOp *srcs[2] = { target, value };
  return poly_uop(ctx, POLY_OP_ASSIGN, target->dtype, srcs, 2, poly_arg_none());
}

/* ── Dtype table for FFI (shared by poly_buffer_by_id and poly_cast_by_id) ─ */

static const PolyDType *_dtype_table_ffi[] = {
  &POLY_VOID, &POLY_BOOL, &POLY_INT8, &POLY_UINT8,
  &POLY_INT16, &POLY_UINT16, &POLY_INT32, &POLY_UINT32,
  &POLY_INT64, &POLY_UINT64, &POLY_FLOAT16, &POLY_BFLOAT16,
  &POLY_FLOAT32, &POLY_FLOAT64,
};
#define N_DTYPE_FFI ((int)(sizeof(_dtype_table_ffi) / sizeof(_dtype_table_ffi[0])))

/* ── Buffer shortcuts ─────────────────────────────────────────────────── */

PolyUOp *poly_buffer_f32(PolyCtx *ctx, int64_t size) {
  return poly_buffer(ctx, POLY_FLOAT32, size);
}

PolyUOp *poly_buffer_f64(PolyCtx *ctx, int64_t size) {
  return poly_buffer(ctx, POLY_FLOAT64, size);
}

PolyUOp *poly_buffer_by_id(PolyCtx *ctx, int64_t size, int dtype_id) {
  if (dtype_id < 0 || dtype_id >= N_DTYPE_FFI) return NULL;
  return poly_buffer(ctx, *_dtype_table_ffi[dtype_id], size);
}

/* ── Dynamic shapes (DEFINE_VAR / BIND) ──────────────────────────────── */

PolyUOp *poly_define_var(PolyCtx *ctx, const char *name, int64_t min_val, int64_t max_val) {
  /* Name string is copied into arena by poly_uop_create (POLY_ARG_DEFINE_VAR case) */
  return poly_uop0(ctx, POLY_OP_DEFINE_VAR, POLY_INT32,
                   poly_arg_define_var(name, min_val, max_val));
}

PolyUOp *poly_bind_var(PolyCtx *ctx, PolyUOp *var, int64_t value) {
  assert(var->op == POLY_OP_DEFINE_VAR);
  PolyUOp *val = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(value));
  return poly_uop2(ctx, POLY_OP_BIND, var->dtype, var, val, poly_arg_none());
}

static int poly_dyn_buffer_id = 1000000; /* separate range from sched.c's poly_buffer_id */

PolyUOp *poly_buffer_var(PolyCtx *ctx, PolyDType dt, PolyUOp *batch_var,
                          const int64_t *inner_dims, int n_inner) {
  assert(batch_var->op == POLY_OP_DEFINE_VAR);
  assert(n_inner >= 0 && n_inner < POLY_MAX_DIMS);
  int64_t max_val = batch_var->arg.define_var.max_val;
  int64_t alloc = max_val;
  /* src[0] = UNIQUE (prevent CSE), src[1] = DEFINE_VAR, src[2..] = CONST inner dims */
  int n_src = 2 + n_inner;
  PolyUOp *src[POLY_MAX_DIMS + 2];
  src[0] = poly_uop0(ctx, POLY_OP_UNIQUE, POLY_VOID, poly_arg_int(poly_dyn_buffer_id++));
  src[1] = batch_var;
  for (int i = 0; i < n_inner; i++) {
    alloc *= inner_dims[i];
    src[2 + i] = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(inner_dims[i]));
  }
  return poly_uop(ctx, POLY_OP_BUFFER, dt, src, n_src, poly_arg_int(alloc));
}

/* ── Composed elementwise ops (exact tinygrad decompositions) ─────────── */

/* Helper: float constant matching the dtype of a given UOp.
 * For float inputs: creates a constant with the same float dtype.
 * For non-float inputs (comparisons producing bool): defaults to float32. */
static inline PolyUOp *cf(PolyCtx *ctx, PolyUOp *ref, double v) {
  PolyDType dt = poly_dtype_scalar(ref->dtype);
  if (poly_dtype_is_float(dt))
    return poly_const_typed(ctx, dt, v);
  return poly_const_float(ctx, v);
}

/* Helper: const with explicit dtype — use in special-math ops for dtype correctness */
static inline PolyUOp *cdt(PolyCtx *ctx, PolyDType dt, double v) {
  return poly_const_typed(ctx, dt, v);
}

static void const_registry_add(PolyCtx *ctx, PolyUOp *buf, void *data);
static void *const_registry_lookup(PolyCtx *ctx, PolyUOp *buf);
static bool const_registry_has(PolyCtx *ctx, PolyUOp *buf);

static int64_t shape_numel_checked(const int64_t *shape, int ndim) {
  if (ndim < 0 || ndim > POLY_MAX_DIMS) return -1;
  if (ndim == 0) return 1;
  if (!shape) return -1;
  int64_t n = 1;
  for (int i = 0; i < ndim; i++) {
    if (shape[i] <= 0) return -1;
    if (n > INT64_MAX / shape[i]) return -1;
    n *= shape[i];
  }
  return n;
}

static bool shape_equal(const int64_t *a, int a_ndim, const int64_t *b, int b_ndim) {
  if (!a || !b || a_ndim != b_ndim) return false;
  for (int i = 0; i < a_ndim; i++) {
    if (a[i] != b[i]) return false;
  }
  return true;
}

static bool shape_equal_except_axis(const int64_t *full, int full_ndim,
                                    const int64_t *reduced, int reduced_ndim,
                                    int axis) {
  if (!full || !reduced || full_ndim != reduced_ndim + 1) return false;
  if (axis < 0) axis += full_ndim;
  if (axis < 0 || axis >= full_ndim) return false;
  for (int i = 0, j = 0; i < full_ndim; i++) {
    if (i == axis) continue;
    if (full[i] != reduced[j++]) return false;
  }
  return true;
}

static PolyUOp *make_const_buffer_tensor(PolyCtx *ctx, PolyDType dt,
                                         const void *src, size_t elem_size,
                                         const int64_t *shape, int ndim) {
  int64_t numel = shape_numel_checked(shape, ndim);
  if (numel <= 0 || !src || elem_size == 0) return NULL;
  if (numel > (1LL << 20)) {
    fprintf(stderr, "polygrad: large constant-backed tensor (%lld elems); "
            "prefer explicit bindings/device-generated paths for hot loops\n",
            (long long)numel);
  }
  if ((size_t)numel > SIZE_MAX / elem_size) return NULL;
  size_t nbytes = (size_t)numel * elem_size;
  void *copy = malloc(nbytes);
  if (!copy) return NULL;
  memcpy(copy, src, nbytes);

  PolyUOp *buf = poly_buffer(ctx, dt, numel);
  const_registry_add(ctx, buf, copy);
  if (ndim == 1) return buf;
  return poly_reshape(ctx, buf, (int64_t *)shape, ndim);
}

static PolyUOp *make_const_f32_tensor(PolyCtx *ctx, const float *src,
                                      const int64_t *shape, int ndim) {
  return make_const_buffer_tensor(ctx, POLY_FLOAT32, src, sizeof(float), shape, ndim);
}

static PolyUOp *make_const_u32_tensor(PolyCtx *ctx, const uint32_t *src,
                                      const int64_t *shape, int ndim) {
  return make_const_buffer_tensor(ctx, POLY_UINT32, src, sizeof(uint32_t), shape, ndim);
}

/* Math */

PolyUOp *poly_exp(PolyCtx *ctx, PolyUOp *x) {
  /* exp(x) = exp2(x * (1/ln2)) */
  PolyDType dt = poly_dtype_scalar(x->dtype);
  return poly_alu1(ctx, POLY_OP_EXP2,
    poly_alu2(ctx, POLY_OP_MUL, x, cdt(ctx, dt, 1.0 / M_LN2)));
}

PolyUOp *poly_log(PolyCtx *ctx, PolyUOp *x) {
  /* log(x) = log2(x) * ln2 */
  PolyDType dt = poly_dtype_scalar(x->dtype);
  return poly_alu2(ctx, POLY_OP_MUL,
    poly_alu1(ctx, POLY_OP_LOG2, x), cdt(ctx, dt, M_LN2));
}

PolyUOp *poly_log1p(PolyCtx *ctx, PolyUOp *x) {
  /* Stable near zero: log(1+x) ~ x - x^2/2 + x^3/3 */
  PolyDType dt = poly_dtype_scalar(x->dtype);
  PolyUOp *ax = poly_abs(ctx, x);
  PolyUOp *small = poly_alu2(ctx, POLY_OP_CMPLT, ax, cdt(ctx, dt, 1e-4));
  PolyUOp *x2 = poly_alu2(ctx, POLY_OP_MUL, x, x);
  PolyUOp *x3 = poly_alu2(ctx, POLY_OP_MUL, x2, x);
  PolyUOp *poly = poly_alu2(ctx, POLY_OP_ADD, x,
                 poly_alu2(ctx, POLY_OP_ADD,
                   poly_alu2(ctx, POLY_OP_MUL, cdt(ctx, dt, -0.5), x2),
                   poly_alu2(ctx, POLY_OP_MUL, cdt(ctx, dt, 1.0/3.0), x3)));
  PolyUOp *direct = poly_log(ctx, poly_alu2(ctx, POLY_OP_ADD, cdt(ctx, dt, 1.0), x));
  return poly_alu3(ctx, POLY_OP_WHERE, small, poly, direct);
}

PolyUOp *poly_expm1(PolyCtx *ctx, PolyUOp *x) {
  /* Stable near zero: expm1(x) ~ x + x^2/2 + x^3/6 */
  PolyDType dt = poly_dtype_scalar(x->dtype);
  PolyUOp *ax = poly_abs(ctx, x);
  PolyUOp *small = poly_alu2(ctx, POLY_OP_CMPLT, ax, cdt(ctx, dt, 1e-4));
  PolyUOp *x2 = poly_alu2(ctx, POLY_OP_MUL, x, x);
  PolyUOp *x3 = poly_alu2(ctx, POLY_OP_MUL, x2, x);
  PolyUOp *poly = poly_alu2(ctx, POLY_OP_ADD, x,
                 poly_alu2(ctx, POLY_OP_ADD,
                   poly_alu2(ctx, POLY_OP_MUL, cdt(ctx, dt, 0.5), x2),
                   poly_alu2(ctx, POLY_OP_MUL, cdt(ctx, dt, 1.0/6.0), x3)));
  PolyUOp *direct = poly_alu2(ctx, POLY_OP_SUB, poly_exp(ctx, x), cdt(ctx, dt, 1.0));
  return poly_alu3(ctx, POLY_OP_WHERE, small, poly, direct);
}

PolyUOp *poly_sin(PolyCtx *ctx, PolyUOp *x) {
  return poly_alu1(ctx, POLY_OP_SIN, x);
}

PolyUOp *poly_cos(PolyCtx *ctx, PolyUOp *x) {
  /* cos(x) = sin(pi/2 - x) */
  return poly_alu1(ctx, POLY_OP_SIN,
    poly_alu2(ctx, POLY_OP_SUB, cf(ctx, x, M_PI / 2.0), x));
}

PolyUOp *poly_tan(PolyCtx *ctx, PolyUOp *x) {
  /* tan(x) = sin(x) / cos(x) */
  return poly_alu2(ctx, POLY_OP_FDIV, poly_sin(ctx, x), poly_cos(ctx, x));
}

/* A&S 7.1.26: tau(|x|) = t * P(t) * exp(-x^2) where t = 1/(1+p*|x|).
 * erf(x) = sign(x) * (1 - tau(|x|)).
 * erfc(x) = tau(x) for x >= 0, 2 - tau(|x|) for x < 0.
 * Computing tau directly avoids the 1-erf(x) cancellation in erfc. */
static PolyUOp *erf_tau(PolyCtx *ctx, PolyUOp *ax, PolyDType dt) {
  PolyUOp *t = poly_alu1(ctx, POLY_OP_RECIPROCAL,
             poly_alu2(ctx, POLY_OP_ADD, cdt(ctx, dt, 1.0),
             poly_alu2(ctx, POLY_OP_MUL, cdt(ctx, dt, 0.3275911), ax)));
  PolyUOp *p = cdt(ctx, dt, 1.061405429);
  p = poly_alu2(ctx, POLY_OP_ADD, cdt(ctx, dt, -1.453152027), poly_alu2(ctx, POLY_OP_MUL, t, p));
  p = poly_alu2(ctx, POLY_OP_ADD, cdt(ctx, dt, 1.421413741), poly_alu2(ctx, POLY_OP_MUL, t, p));
  p = poly_alu2(ctx, POLY_OP_ADD, cdt(ctx, dt, -0.284496736), poly_alu2(ctx, POLY_OP_MUL, t, p));
  p = poly_alu2(ctx, POLY_OP_ADD, cdt(ctx, dt, 0.254829592), poly_alu2(ctx, POLY_OP_MUL, t, p));
  PolyUOp *x2 = poly_alu2(ctx, POLY_OP_MUL, ax, ax);
  PolyUOp *e = poly_exp(ctx, poly_alu1(ctx, POLY_OP_NEG, x2));
  return poly_alu2(ctx, POLY_OP_MUL, t, poly_alu2(ctx, POLY_OP_MUL, p, e));
}

PolyUOp *poly_erf(PolyCtx *ctx, PolyUOp *x) {
  PolyDType dt = poly_dtype_scalar(x->dtype);
  PolyUOp *sign = poly_sign(ctx, x);
  PolyUOp *ax = poly_abs(ctx, x);
  PolyUOp *tau = erf_tau(ctx, ax, dt);
  return poly_alu2(ctx, POLY_OP_MUL, sign, poly_alu2(ctx, POLY_OP_SUB, cdt(ctx, dt, 1.0), tau));
}

PolyUOp *poly_erfc(PolyCtx *ctx, PolyUOp *x) {
  /* erfc(x) = tau(|x|) for x >= 0, 2 - tau(|x|) for x < 0.
   * No 1-erf(x) cancellation -- tau is computed directly. */
  PolyDType dt = poly_dtype_scalar(x->dtype);
  PolyUOp *ax = poly_abs(ctx, x);
  PolyUOp *tau = erf_tau(ctx, ax, dt);
  PolyUOp *neg = poly_alu2(ctx, POLY_OP_CMPLT, x, cdt(ctx, dt, 0.0));
  PolyUOp *erfc_neg = poly_alu2(ctx, POLY_OP_SUB, cdt(ctx, dt, 2.0), tau);
  return poly_alu3(ctx, POLY_OP_WHERE, neg, erfc_neg, tau);
}

PolyUOp *poly_erfinv(PolyCtx *ctx, PolyUOp *x) {
  /* Winitzki approximation (a=0.147). */
  PolyDType dt = poly_dtype_scalar(x->dtype);
  PolyUOp *a = cdt(ctx, dt, 0.147);
  PolyUOp *one = cdt(ctx, dt, 1.0);
  PolyUOp *x2 = poly_alu2(ctx, POLY_OP_MUL, x, x);
  PolyUOp *ln = poly_log(ctx, poly_alu2(ctx, POLY_OP_SUB, one, x2));
  PolyUOp *term1 = poly_alu2(ctx, POLY_OP_ADD,
                   poly_alu2(ctx, POLY_OP_FDIV, cdt(ctx, dt, 2.0/(M_PI*0.147)), one),
                   poly_alu2(ctx, POLY_OP_MUL, cdt(ctx, dt, 0.5), ln));
  PolyUOp *term2 = poly_alu2(ctx, POLY_OP_FDIV, ln, a);
  PolyUOp *inside = poly_alu2(ctx, POLY_OP_SUB,
                    poly_alu2(ctx, POLY_OP_MUL, term1, term1), term2);
  PolyUOp *root = poly_alu1(ctx, POLY_OP_SQRT,
                  poly_alu2(ctx, POLY_OP_SUB, poly_alu1(ctx, POLY_OP_SQRT, inside), term1));
  return poly_alu2(ctx, POLY_OP_MUL, poly_sign(ctx, x), root);
}

PolyUOp *poly_ndtri(PolyCtx *ctx, PolyUOp *x) {
  /* ndtri(p) = sqrt(2) * erfinv(2p-1) */
  PolyDType dt = poly_dtype_scalar(x->dtype);
  PolyUOp *arg = poly_alu2(ctx, POLY_OP_SUB,
                poly_alu2(ctx, POLY_OP_MUL, cdt(ctx, dt, 2.0), x), cdt(ctx, dt, 1.0));
  return poly_alu2(ctx, POLY_OP_MUL, cdt(ctx, dt, sqrt(2.0)), poly_erfinv(ctx, arg));
}

PolyUOp *poly_digamma(PolyCtx *ctx, PolyUOp *x) {
  /* First-order asymptotic with recurrence to x>=6. */
  PolyDType dt = poly_dtype_scalar(x->dtype);
  PolyUOp *acc = cdt(ctx, dt, 0.0);
  PolyUOp *xx = x;
  for (int i = 0; i < 6; i++) {
    PolyUOp *cond = poly_alu2(ctx, POLY_OP_CMPLT, xx, cdt(ctx, dt, 6.0));
    PolyUOp *inv = poly_alu1(ctx, POLY_OP_RECIPROCAL, xx);
    acc = poly_alu3(ctx, POLY_OP_WHERE, cond, poly_alu2(ctx, POLY_OP_SUB, acc, inv), acc);
    xx = poly_alu3(ctx, POLY_OP_WHERE, cond, poly_alu2(ctx, POLY_OP_ADD, xx, cdt(ctx, dt, 1.0)), xx);
  }
  PolyUOp *inv = poly_alu1(ctx, POLY_OP_RECIPROCAL, xx);
  PolyUOp *inv2 = poly_alu2(ctx, POLY_OP_MUL, inv, inv);
  PolyUOp *inv4 = poly_alu2(ctx, POLY_OP_MUL, inv2, inv2);
  PolyUOp *inv6 = poly_alu2(ctx, POLY_OP_MUL, inv4, inv2);
  PolyUOp *asym = poly_alu2(ctx, POLY_OP_ADD, poly_log(ctx, xx),
                poly_alu2(ctx, POLY_OP_ADD,
                  poly_alu2(ctx, POLY_OP_MUL, cdt(ctx, dt, -0.5), inv),
                  poly_alu2(ctx, POLY_OP_ADD,
                    poly_alu2(ctx, POLY_OP_MUL, cdt(ctx, dt, -1.0/12.0), inv2),
                    poly_alu2(ctx, POLY_OP_ADD,
                      poly_alu2(ctx, POLY_OP_MUL, cdt(ctx, dt, 1.0/120.0), inv4),
                      poly_alu2(ctx, POLY_OP_MUL, cdt(ctx, dt, -1.0/252.0), inv6)))));
  return poly_alu2(ctx, POLY_OP_ADD, acc, asym);
}

static PolyUOp *poly_lgamma_forward_lanczos(PolyCtx *ctx, PolyUOp *x, PolyDType dt) {
  /* Lanczos approximation with reflection. */
  const double g = 7.0;
  const double c0 = 0.99999999999980993;
  const double c[8] = {
    676.5203681218851, -1259.1392167224028, 771.32342877765313,
    -176.61502916214059, 12.507343278686905, -0.13857109526572012,
    9.9843695780195716e-6, 1.5056327351493116e-7
  };

  PolyUOp *xm1 = poly_alu2(ctx, POLY_OP_SUB, x, cdt(ctx, dt, 1.0));
  PolyUOp *a = cdt(ctx, dt, c0);
  for (int i = 0; i < 8; i++) {
    PolyUOp *den = poly_alu2(ctx, POLY_OP_ADD, xm1, cdt(ctx, dt, (double)(i + 1)));
    a = poly_alu2(ctx, POLY_OP_ADD, a,
         poly_alu2(ctx, POLY_OP_FDIV, cdt(ctx, dt, c[i]), den));
  }
  PolyUOp *t = poly_alu2(ctx, POLY_OP_ADD, xm1, cdt(ctx, dt, g + 0.5));
  PolyUOp *lg_pos = poly_alu2(ctx, POLY_OP_ADD, cdt(ctx, dt, 0.91893853320467274178),
                 poly_alu2(ctx, POLY_OP_ADD,
                   poly_alu2(ctx, POLY_OP_MUL,
                     poly_alu2(ctx, POLY_OP_ADD, xm1, cdt(ctx, dt, 0.5)),
                     poly_log(ctx, t)),
                   poly_alu2(ctx, POLY_OP_SUB, poly_log(ctx, a), t)));

  PolyUOp *one_minus_x = poly_alu2(ctx, POLY_OP_SUB, cdt(ctx, dt, 1.0), x);
  PolyUOp *xm1r = poly_alu2(ctx, POLY_OP_SUB, one_minus_x, cdt(ctx, dt, 1.0));
  PolyUOp *ar = cdt(ctx, dt, c0);
  for (int i = 0; i < 8; i++) {
    PolyUOp *den = poly_alu2(ctx, POLY_OP_ADD, xm1r, cdt(ctx, dt, (double)(i + 1)));
    ar = poly_alu2(ctx, POLY_OP_ADD, ar,
         poly_alu2(ctx, POLY_OP_FDIV, cdt(ctx, dt, c[i]), den));
  }
  PolyUOp *tr = poly_alu2(ctx, POLY_OP_ADD, xm1r, cdt(ctx, dt, g + 0.5));
  PolyUOp *lg_ref_base = poly_alu2(ctx, POLY_OP_ADD, cdt(ctx, dt, 0.91893853320467274178),
                      poly_alu2(ctx, POLY_OP_ADD,
                        poly_alu2(ctx, POLY_OP_MUL,
                          poly_alu2(ctx, POLY_OP_ADD, xm1r, cdt(ctx, dt, 0.5)),
                          poly_log(ctx, tr)),
                        poly_alu2(ctx, POLY_OP_SUB, poly_log(ctx, ar), tr)));
  PolyUOp *sinpix = poly_sin(ctx, poly_alu2(ctx, POLY_OP_MUL, cdt(ctx, dt, M_PI), x));
  PolyUOp *lg_ref = poly_alu2(ctx, POLY_OP_SUB,
                  poly_alu2(ctx, POLY_OP_SUB, cdt(ctx, dt, log(M_PI)),
                    poly_log(ctx, poly_abs(ctx, sinpix))),
                  lg_ref_base);
  PolyUOp *cond = poly_alu2(ctx, POLY_OP_CMPLT, x, cdt(ctx, dt, 0.5));
  return poly_alu3(ctx, POLY_OP_WHERE, cond, lg_ref, lg_pos);
}

PolyUOp *poly_lgamma(PolyCtx *ctx, PolyUOp *x) {
  /* Explicit VJP override:
   * y = detach(f(x)) + (x - detach(x))*digamma(x) */
  PolyDType dt = poly_dtype_scalar(x->dtype);
  PolyUOp *fwd = poly_lgamma_forward_lanczos(ctx, x, dt);
  PolyUOp *dx = poly_uop1(ctx, POLY_OP_DETACH, x->dtype, x, poly_arg_none());
  PolyUOp *df = poly_uop1(ctx, POLY_OP_DETACH, fwd->dtype, fwd, poly_arg_none());
  PolyUOp *delta = poly_alu2(ctx, POLY_OP_SUB, x, dx);
  PolyUOp *forced = poly_alu2(ctx, POLY_OP_MUL, delta, poly_digamma(ctx, x));
  return poly_alu2(ctx, POLY_OP_ADD, df, forced);
}

PolyUOp *poly_sigmoid(PolyCtx *ctx, PolyUOp *x) {
  /* sigmoid(x) = (1 + exp2(x * (-1/ln2)))^-1 */
  PolyUOp *scaled = poly_alu2(ctx, POLY_OP_MUL, x, cf(ctx, x, -1.0 / M_LN2));
  PolyUOp *e = poly_alu1(ctx, POLY_OP_EXP2, scaled);
  return poly_alu1(ctx, POLY_OP_RECIPROCAL,
    poly_alu2(ctx, POLY_OP_ADD, cf(ctx, x, 1.0), e));
}

PolyUOp *poly_tanh_act(PolyCtx *ctx, PolyUOp *x) {
  /* tanh(x) = 2*sigmoid(2x) - 1 */
  PolyUOp *two_x = poly_alu2(ctx, POLY_OP_MUL, cf(ctx, x, 2.0), x);
  return poly_alu2(ctx, POLY_OP_SUB,
    poly_alu2(ctx, POLY_OP_MUL, cf(ctx, x, 2.0), poly_sigmoid(ctx, two_x)),
    cf(ctx, x, 1.0));
}

PolyUOp *poly_abs(PolyCtx *ctx, PolyUOp *x) {
  /* abs(x) = x * sign(x) */
  return poly_alu2(ctx, POLY_OP_MUL, x, poly_sign(ctx, x));
}

PolyUOp *poly_sign(PolyCtx *ctx, PolyUOp *x) {
  /* sign(x) = ne(x,0).where(lt(x,0).where(-1, 1), 0) + x*0
   * The +x*0 preserves NaN (NaN*0=NaN, NaN+0=NaN) */
  PolyUOp *zero = cf(ctx, x, 0.0);
  PolyUOp *is_nonzero = poly_alu2(ctx, POLY_OP_CMPNE, x, zero);
  PolyUOp *is_neg = poly_alu2(ctx, POLY_OP_CMPLT, x, zero);
  PolyUOp *neg_or_pos = poly_alu3(ctx, POLY_OP_WHERE, is_neg, cf(ctx, x, -1.0), cf(ctx, x, 1.0));
  PolyUOp *result = poly_alu3(ctx, POLY_OP_WHERE, is_nonzero, neg_or_pos, zero);
  /* +x*0 to propagate NaN */
  return poly_alu2(ctx, POLY_OP_ADD, result,
    poly_alu2(ctx, POLY_OP_MUL, x, zero));
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
  PolyUOp *cond = poly_alu2(ctx, POLY_OP_CMPLT, b, x); /* b < x ≡ x > b */
  return poly_alu3(ctx, POLY_OP_WHERE, cond,
    poly_alu2(ctx, POLY_OP_ADD, b, cf(ctx, x, 1.0)), b);
}

PolyUOp *poly_floor(PolyCtx *ctx, PolyUOp *x) {
  /* floor(x) = (x < (b=trunc(x))).where(b-1, b) */
  PolyUOp *b = poly_alu1(ctx, POLY_OP_TRUNC, x);
  PolyUOp *cond = poly_alu2(ctx, POLY_OP_CMPLT, x, b); /* x < b */
  return poly_alu3(ctx, POLY_OP_WHERE, cond,
    poly_alu2(ctx, POLY_OP_SUB, b, cf(ctx, x, 1.0)), b);
}

PolyUOp *poly_round_f(PolyCtx *ctx, PolyUOp *x) {
  /* round(x) with banker's rounding (round half to even):
   * (x > 0) == (trunc(x/2) == trunc(trunc(x)/2)) ? ceil(x-0.5) : floor(x+0.5) */
  PolyUOp *half = cf(ctx, x, 0.5);
  PolyUOp *two = cf(ctx, x, 2.0);
  PolyUOp *b = poly_alu1(ctx, POLY_OP_TRUNC, x);
  PolyUOp *x_gt_0 = poly_alu2(ctx, POLY_OP_CMPLT, cf(ctx, x, 0.0), x);
  PolyUOp *b_half = poly_alu2(ctx, POLY_OP_FDIV, b, two);
  PolyUOp *x_half = poly_alu2(ctx, POLY_OP_FDIV, x, two);
  PolyUOp *trunc_b_half = poly_alu1(ctx, POLY_OP_TRUNC, b_half);
  PolyUOp *trunc_x_half = poly_alu1(ctx, POLY_OP_TRUNC, x_half);
  PolyUOp *halves_eq = poly_eq(ctx, trunc_b_half, trunc_x_half);
  PolyUOp *cond = poly_eq(ctx, x_gt_0, halves_eq);
  return poly_alu3(ctx, POLY_OP_WHERE, cond,
    poly_ceil(ctx, poly_alu2(ctx, POLY_OP_SUB, x, half)),
    poly_floor(ctx, poly_alu2(ctx, POLY_OP_ADD, x, half)));
}

PolyUOp *poly_isinf(PolyCtx *ctx, PolyUOp *x) {
  /* isinf(x) = (x == inf) | (x == -inf)
   * But we don't have OR on bools easily. Use: abs(x) == inf
   * Actually: isinf = NOT(isnan(x)) AND NOT(isnan(x - x)) is wrong...
   * Simpler: isinf(x) = (x + x) == x AND x != 0 AND NOT(isnan(x)) */
  /* Simplest correct: x == inf OR x == -inf via (abs(x) == inf) without abs
   * Actually the simplest is: (x != x) is false and (x - x) != (x - x) is true
   * No -- let's just do: NOT isnan AND (x*0 != 0 is false for inf but...)
   * Cleanest: cmpne(x, x) is false for inf. (x - x) is NaN for inf.
   * So: isinf(x) = NOT(isnan(x)) AND isnan(x - x) AND cmpne(x, 0) */
  PolyUOp *zero = cf(ctx, x, 0.0);
  PolyUOp *x_minus_x = poly_alu2(ctx, POLY_OP_SUB, x, x);
  PolyUOp *not_nan = poly_eq(ctx, x, x); /* true if not NaN */
  PolyUOp *sub_nan = poly_ne(ctx, x_minus_x, x_minus_x); /* true if x-x is NaN (inf case) */
  PolyUOp *not_zero = poly_ne(ctx, x, zero);
  /* all three must be true: use AND via MUL on bool-like values */
  PolyUOp *t1 = poly_alu2(ctx, POLY_OP_MUL, not_nan, sub_nan);
  return poly_alu2(ctx, POLY_OP_MUL, t1, not_zero);
}

PolyUOp *poly_isnan(PolyCtx *ctx, PolyUOp *x) {
  /* isnan(x) = (x != x) — IEEE 754 */
  return poly_alu2(ctx, POLY_OP_CMPNE, x, x);
}

/* Activations */

PolyUOp *poly_relu(PolyCtx *ctx, PolyUOp *x) {
  /* relu(x) = where(0 < x, x, 0) */
  PolyUOp *zero = cf(ctx, x, 0.0);
  PolyUOp *cond = poly_alu2(ctx, POLY_OP_CMPLT, zero, x);
  return poly_alu3(ctx, POLY_OP_WHERE, cond, x, zero);
}

PolyUOp *poly_relu6(PolyCtx *ctx, PolyUOp *x) {
  /* relu6(x) = relu(x) - relu(x - 6) */
  return poly_alu2(ctx, POLY_OP_SUB,
    poly_relu(ctx, x),
    poly_relu(ctx, poly_alu2(ctx, POLY_OP_SUB, x, cf(ctx, x, 6.0))));
}

PolyUOp *poly_leaky_relu(PolyCtx *ctx, PolyUOp *x, double neg_slope) {
  /* leaky_relu(x) = where(x < 0, neg_slope * x, x) */
  PolyUOp *cond = poly_alu2(ctx, POLY_OP_CMPLT, x, cf(ctx, x, 0.0));
  return poly_alu3(ctx, POLY_OP_WHERE, cond,
    poly_alu2(ctx, POLY_OP_MUL, cf(ctx, x, neg_slope), x), x);
}

PolyUOp *poly_gelu(PolyCtx *ctx, PolyUOp *x) {
  /* gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))) */
  PolyUOp *x3 = poly_alu2(ctx, POLY_OP_MUL, x, poly_alu2(ctx, POLY_OP_MUL, x, x));
  PolyUOp *inner = poly_alu2(ctx, POLY_OP_ADD, x,
    poly_alu2(ctx, POLY_OP_MUL, cf(ctx, x, 0.044715), x3));
  PolyUOp *scaled = poly_alu2(ctx, POLY_OP_MUL,
    cf(ctx, x, sqrt(2.0 / M_PI)), inner);
  return poly_alu2(ctx, POLY_OP_MUL, cf(ctx, x, 0.5),
    poly_alu2(ctx, POLY_OP_MUL, x,
      poly_alu2(ctx, POLY_OP_ADD, cf(ctx, x, 1.0), poly_tanh_act(ctx, scaled))));
}

PolyUOp *poly_quick_gelu(PolyCtx *ctx, PolyUOp *x) {
  /* quick_gelu(x) = x * sigmoid(1.702 * x) */
  return poly_alu2(ctx, POLY_OP_MUL, x,
    poly_sigmoid(ctx, poly_alu2(ctx, POLY_OP_MUL, cf(ctx, x, 1.702), x)));
}

PolyUOp *poly_silu(PolyCtx *ctx, PolyUOp *x) {
  /* silu(x) = x * sigmoid(x) */
  return poly_alu2(ctx, POLY_OP_MUL, x, poly_sigmoid(ctx, x));
}

PolyUOp *poly_elu(PolyCtx *ctx, PolyUOp *x, double alpha) {
  /* elu(x, α) = relu(x) - α * relu(1 - exp(x)) */
  return poly_alu2(ctx, POLY_OP_SUB,
    poly_relu(ctx, x),
    poly_alu2(ctx, POLY_OP_MUL, cf(ctx, x, alpha),
      poly_relu(ctx,
        poly_alu2(ctx, POLY_OP_SUB, cf(ctx, x, 1.0), poly_exp(ctx, x)))));
}

PolyUOp *poly_softplus(PolyCtx *ctx, PolyUOp *x, double beta) {
  /* softplus(x, β) = (1/β) * log(1 + exp(β*x))
   * = (1/β) * logaddexp(β*x, 0)
   * logaddexp(a, b) = max(a,b) + log(exp(a-max) + exp(b-max)) */
  PolyUOp *bx = poly_alu2(ctx, POLY_OP_MUL, cf(ctx, x, beta), x);
  PolyUOp *zero = cf(ctx, x, 0.0);
  PolyUOp *m = poly_alu2(ctx, POLY_OP_MAX, bx, zero);
  PolyUOp *ea = poly_exp(ctx, poly_alu2(ctx, POLY_OP_SUB, bx, m));
  PolyUOp *eb = poly_exp(ctx, poly_alu2(ctx, POLY_OP_SUB, zero, m));
  PolyUOp *lae = poly_alu2(ctx, POLY_OP_ADD, m,
    poly_log(ctx, poly_alu2(ctx, POLY_OP_ADD, ea, eb)));
  return poly_alu2(ctx, POLY_OP_MUL, cf(ctx, x, 1.0 / beta), lae);
}

PolyUOp *poly_mish(PolyCtx *ctx, PolyUOp *x) {
  /* mish(x) = x * tanh(softplus(x, 1.0)) */
  return poly_alu2(ctx, POLY_OP_MUL, x,
    poly_tanh_act(ctx, poly_softplus(ctx, x, 1.0)));
}

PolyUOp *poly_hardtanh(PolyCtx *ctx, PolyUOp *x, double min_val, double max_val) {
  /* hardtanh = clamp(x, min, max) */
  return poly_clamp(ctx, x, min_val, max_val);
}

PolyUOp *poly_hardswish(PolyCtx *ctx, PolyUOp *x) {
  /* hardswish(x) = x * relu6(x + 3) * (1/6) */
  return poly_alu2(ctx, POLY_OP_MUL,
    poly_alu2(ctx, POLY_OP_MUL, x,
      poly_relu6(ctx, poly_alu2(ctx, POLY_OP_ADD, x, cf(ctx, x, 3.0)))),
    cf(ctx, x, 1.0 / 6.0));
}

PolyUOp *poly_hardsigmoid(PolyCtx *ctx, PolyUOp *x) {
  /* hardsigmoid(x) = relu(x/6 + 0.5) - relu(x/6 + 0.5 - 1) */
  PolyUOp *t = poly_alu2(ctx, POLY_OP_ADD,
    poly_alu2(ctx, POLY_OP_MUL, cf(ctx, x, 1.0/6.0), x), cf(ctx, x, 0.5));
  return poly_alu2(ctx, POLY_OP_SUB,
    poly_relu(ctx, t),
    poly_relu(ctx, poly_alu2(ctx, POLY_OP_SUB, t, cf(ctx, x, 1.0))));
}

/* Comparisons */

PolyUOp *poly_eq(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  /* eq(a,b) = ne(a,b).logical_not() = where(cmpne(a,b), 0, 1) */
  PolyUOp *ne = poly_alu2(ctx, POLY_OP_CMPNE, a, b);
  return poly_alu3(ctx, POLY_OP_WHERE, ne, cf(ctx, a, 0.0), cf(ctx, a, 1.0));
}

PolyUOp *poly_ne(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  return poly_alu2(ctx, POLY_OP_CMPNE, a, b);
}

PolyUOp *poly_gt(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  /* gt(a,b) = cmplt(b,a) */
  return poly_alu2(ctx, POLY_OP_CMPLT, b, a);
}

PolyUOp *poly_ge(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  /* ge(a,b) = NOT(cmplt(a,b)) = where(cmplt(a,b), 0, 1) */
  PolyUOp *lt = poly_alu2(ctx, POLY_OP_CMPLT, a, b);
  return poly_alu3(ctx, POLY_OP_WHERE, lt, cf(ctx, a, 0.0), cf(ctx, a, 1.0));
}

PolyUOp *poly_le(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  /* le(a,b) = NOT(cmplt(b,a)) = where(cmplt(b,a), 0, 1) */
  PolyUOp *gt_val = poly_alu2(ctx, POLY_OP_CMPLT, b, a);
  return poly_alu3(ctx, POLY_OP_WHERE, gt_val, cf(ctx, a, 0.0), cf(ctx, a, 1.0));
}

PolyUOp *poly_cast(PolyCtx *ctx, PolyUOp *x, PolyDType target) {
  return poly_uop1(ctx, POLY_OP_CAST, target, x, poly_arg_none());
}

PolyUOp *poly_cast_by_id(PolyCtx *ctx, PolyUOp *x, int dtype_id) {
  if (dtype_id < 0 || dtype_id >= N_DTYPE_FFI) return NULL;
  return poly_cast(ctx, x, *_dtype_table_ffi[dtype_id]);
}

PolyUOp *poly_where_op(PolyCtx *ctx, PolyUOp *cond, PolyUOp *x, PolyUOp *y) {
  return poly_alu3(ctx, POLY_OP_WHERE, cond, x, y);
}

PolyUOp *poly_maximum(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  return poly_alu2(ctx, POLY_OP_MAX, a, b);
}

PolyUOp *poly_minimum(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  /* minimum(a,b) = -max(-a, -b) */
  return poly_alu1(ctx, POLY_OP_NEG,
    poly_alu2(ctx, POLY_OP_MAX,
      poly_alu1(ctx, POLY_OP_NEG, a),
      poly_alu1(ctx, POLY_OP_NEG, b)));
}

PolyUOp *poly_clamp(PolyCtx *ctx, PolyUOp *x, double lo, double hi) {
  /* clamp(x, lo, hi) = where(x < lo, lo, where(x > hi, hi, x)) */
  PolyUOp *lo_c = cf(ctx, x, lo);
  PolyUOp *hi_c = cf(ctx, x, hi);
  PolyUOp *lt_lo = poly_alu2(ctx, POLY_OP_CMPLT, x, lo_c);
  PolyUOp *clamped_lo = poly_alu3(ctx, POLY_OP_WHERE, lt_lo, lo_c, x);
  PolyUOp *gt_hi = poly_alu2(ctx, POLY_OP_CMPLT, hi_c, clamped_lo);
  return poly_alu3(ctx, POLY_OP_WHERE, gt_hi, hi_c, clamped_lo);
}

PolyUOp *poly_detach(PolyCtx *ctx, PolyUOp *x) {
  return poly_uop1(ctx, POLY_OP_DETACH, x->dtype, x, poly_arg_none());
}

PolyUOp *poly_full(PolyCtx *ctx, const int64_t *shape, int ndim, double fill_value) {
  int64_t numel = shape_numel_checked(shape, ndim);
  if (numel < 0) return NULL;
  if (numel == 0) return poly_buffer(ctx, POLY_FLOAT32, 0);
  float *data = malloc((size_t)numel * sizeof(float));
  if (!data) return NULL;
  for (int64_t i = 0; i < numel; i++) data[i] = (float)fill_value;
  PolyUOp *u = make_const_f32_tensor(ctx, data, shape, ndim);
  free(data);
  return u;
}

PolyUOp *poly_arange(PolyCtx *ctx, double start, double stop, double step) {
  if (step == 0.0) {
    fprintf(stderr, "polygrad: arange: step must be non-zero\n");
    return NULL;
  }
  int64_t n = 0;
  if ((step > 0.0 && start < stop) || (step < 0.0 && start > stop)) {
    double span = (stop - start) / step;
    n = (int64_t)ceil(span - 1e-12);
    if (n < 0) n = 0;
  }
  if (n == 0) return poly_buffer(ctx, POLY_FLOAT32, 0);
  float *data = malloc((size_t)n * sizeof(float));
  if (!data) return NULL;
  for (int64_t i = 0; i < n; i++) data[i] = (float)(start + (double)i * step);
  int64_t shape[1] = { n };
  PolyUOp *u = make_const_f32_tensor(ctx, data, shape, 1);
  free(data);
  return u;
}

PolyUOp *poly_linspace(PolyCtx *ctx, double start, double stop, int64_t steps) {
  if (steps <= 0) return poly_buffer(ctx, POLY_FLOAT32, 0);
  float *data = malloc((size_t)steps * sizeof(float));
  if (!data) return NULL;
  if (steps == 1) data[0] = (float)start;
  else {
    for (int64_t i = 0; i < steps; i++)
      data[i] = (float)(start + (stop - start) * (double)i / (double)(steps - 1));
  }
  int64_t shape[1] = { steps };
  PolyUOp *u = make_const_f32_tensor(ctx, data, shape, 1);
  free(data);
  return u;
}

PolyUOp *poly_eye(PolyCtx *ctx, int64_t n) {
  if (n <= 0) return poly_buffer(ctx, POLY_FLOAT32, 0);
  if ((size_t)n > SIZE_MAX / (size_t)n) return NULL;
  size_t numel = (size_t)n * (size_t)n;
  float *data = calloc(numel, sizeof(float));
  if (!data) return NULL;
  for (int64_t i = 0; i < n; i++) data[(size_t)i * (size_t)n + (size_t)i] = 1.0f;
  int64_t shape[2] = { n, n };
  PolyUOp *u = make_const_f32_tensor(ctx, data, shape, 2);
  free(data);
  return u;
}

static PolyUOp *tri_mask(PolyCtx *ctx, const int64_t *shape, int diagonal, bool upper) {
  if (!shape || shape[0] <= 0 || shape[1] <= 0) return NULL;
  int64_t rows = shape[0], cols = shape[1];
  if ((size_t)rows > SIZE_MAX / (size_t)cols) return NULL;
  size_t numel = (size_t)rows * (size_t)cols;
  float *m = malloc(numel * sizeof(float));
  if (!m) return NULL;
  for (int64_t i = 0; i < rows; i++) {
    for (int64_t j = 0; j < cols; j++) {
      bool keep = upper ? (j >= i + diagonal) : (j <= i + diagonal);
      m[(size_t)i * (size_t)cols + (size_t)j] = keep ? 1.0f : 0.0f;
    }
  }
  PolyUOp *mask = make_const_f32_tensor(ctx, m, shape, 2);
  free(m);
  return mask;
}

PolyUOp *poly_tril(PolyCtx *ctx, PolyUOp *x, const int64_t *shape, int ndim, int diagonal) {
  if (!x || !shape || ndim != 2) {
    fprintf(stderr, "polygrad: tril: only 2D tensors are supported\n");
    return NULL;
  }
  PolyUOp *mask = tri_mask(ctx, shape, diagonal, false);
  if (!mask) return NULL;
  return poly_alu2(ctx, POLY_OP_MUL, x, mask);
}

PolyUOp *poly_triu(PolyCtx *ctx, PolyUOp *x, const int64_t *shape, int ndim, int diagonal) {
  if (!x || !shape || ndim != 2) {
    fprintf(stderr, "polygrad: triu: only 2D tensors are supported\n");
    return NULL;
  }
  PolyUOp *mask = tri_mask(ctx, shape, diagonal, true);
  if (!mask) return NULL;
  return poly_alu2(ctx, POLY_OP_MUL, x, mask);
}

PolyUOp *poly_rand(PolyCtx *ctx, const int64_t *shape, int ndim, uint64_t seed) {
  int64_t numel = shape_numel_checked(shape, ndim);
  if (numel <= 0) return NULL;
  if ((size_t)numel > SIZE_MAX / sizeof(uint32_t)) return NULL;
  uint32_t *counter = malloc((size_t)numel * sizeof(uint32_t));
  if (!counter) {
    return NULL;
  }

  uint32_t key_lo = (uint32_t)(seed & 0xffffffffu);
  uint32_t key_hi = (uint32_t)(seed >> 32);
  uint32_t mixed_key = key_lo ^ ((key_hi << 16) | (key_hi >> 16));
  for (int64_t i = 0; i < numel; i++) {
    counter[i] = (uint32_t)i;
  }

  PolyUOp *counter_t = make_const_u32_tensor(ctx, counter, shape, ndim);
  free(counter);
  if (!counter_t) return NULL;
  PolyUOp *key_t = poly_uop0(ctx, POLY_OP_CONST, POLY_UINT32, poly_arg_int((int64_t)mixed_key));

  PolyUOp *bits = poly_uop2(ctx, POLY_OP_THREEFRY, POLY_UINT32, counter_t, key_t, poly_arg_none());
  PolyUOp *sh = poly_uop0(ctx, POLY_OP_CONST, POLY_UINT32, poly_arg_int(8));
  PolyUOp *hi24 = poly_uop2(ctx, POLY_OP_SHR, POLY_UINT32, bits, sh, poly_arg_none());
  PolyUOp *as_f = poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, hi24, poly_arg_none());
  return poly_alu2(ctx, POLY_OP_MUL, as_f, cf(ctx, as_f, 1.0 / 16777216.0));
}

PolyUOp *poly_randn(PolyCtx *ctx, const int64_t *shape, int ndim, uint64_t seed) {
  PolyUOp *u1 = poly_rand(ctx, shape, ndim, seed);
  PolyUOp *u2 = poly_rand(ctx, shape, ndim, seed ^ 0x9E3779B97F4A7C15ull);
  if (!u1 || !u2) return NULL;
  PolyUOp *u1_safe = poly_maximum(ctx, u1, cf(ctx, u1, 1e-7));
  PolyUOp *r = poly_alu1(ctx, POLY_OP_SQRT,
              poly_alu2(ctx, POLY_OP_MUL, cf(ctx, u1, -2.0), poly_log(ctx, u1_safe)));
  PolyUOp *theta = poly_alu2(ctx, POLY_OP_MUL, cf(ctx, u2, 2.0 * M_PI), u2);
  return poly_alu2(ctx, POLY_OP_MUL, r, poly_cos(ctx, theta));
}

PolyUOp *poly_cholesky(PolyCtx *ctx, PolyUOp *x, const int64_t *shape, int ndim, int upper) {
  (void)ctx; (void)x; (void)shape; (void)ndim; (void)upper;
  fprintf(stderr, "polygrad: cholesky: Track C fallback not yet implemented in frontend graph builder\n");
  return NULL;
}

PolyUOp *poly_triangular_solve(PolyCtx *ctx,
                               PolyUOp *a, const int64_t *a_shape, int a_ndim,
                               PolyUOp *b, const int64_t *b_shape, int b_ndim,
                               int upper, int transpose_a, int unit_diagonal,
                               int64_t *out_shape, int *out_ndim) {
  (void)ctx; (void)a; (void)a_shape; (void)a_ndim; (void)b;
  (void)b_shape; (void)b_ndim; (void)upper; (void)transpose_a; (void)unit_diagonal;
  if (out_shape && b_shape && b_ndim > 0) memcpy(out_shape, b_shape, (size_t)b_ndim * sizeof(int64_t));
  if (out_ndim) *out_ndim = b_ndim;
  fprintf(stderr, "polygrad: triangular_solve: Track C fallback not yet implemented in frontend graph builder\n");
  return NULL;
}

/* ── Shape-aware composed ops ────────────────────────────────────────── */

/* Internal: compute output shape for a single-axis reduction */
static void reduce_output_shape(const int64_t *shape, int ndim, int axis,
                                int keepdim, int64_t *out_shape, int *out_ndim) {
  if (axis < 0) axis += ndim;
  int on = 0;
  for (int i = 0; i < ndim; i++) {
    if (i == axis) {
      if (keepdim) out_shape[on++] = 1;
    } else {
      out_shape[on++] = shape[i];
    }
  }
  if (on == 0) { out_shape[0] = 1; on = 1; }
  *out_ndim = on;
}

/* Internal: do a single-axis reduce and optionally reshape away the axis */
static PolyUOp *do_reduce(PolyCtx *ctx, PolyOps reduce_op, PolyUOp *x,
                           const int64_t *shape, int ndim, int axis, int keepdim,
                           int64_t *out_shape, int *out_ndim) {
  if (axis < 0) axis += ndim;
  int64_t axes[] = { axis };
  PolyUOp *r = poly_reduce_axis(ctx, reduce_op, x, axes, 1);
  reduce_output_shape(shape, ndim, axis, keepdim, out_shape, out_ndim);
  if (!keepdim) {
    r = poly_reshape(ctx, r, out_shape, *out_ndim);
  }
  return r;
}

PolyUOp *poly_sum_reduce(PolyCtx *ctx, PolyUOp *x,
                         const int64_t *shape, int ndim,
                         int axis, int keepdim,
                         int64_t *out_shape, int *out_ndim) {
  return do_reduce(ctx, POLY_OP_ADD, x, shape, ndim, axis, keepdim,
                   out_shape, out_ndim);
}

PolyUOp *poly_max_reduce(PolyCtx *ctx, PolyUOp *x,
                         const int64_t *shape, int ndim,
                         int axis, int keepdim,
                         int64_t *out_shape, int *out_ndim) {
  return do_reduce(ctx, POLY_OP_MAX, x, shape, ndim, axis, keepdim,
                   out_shape, out_ndim);
}

static PolyUOp *reshape_logical_input(PolyCtx *ctx, PolyUOp *x,
                                      const int64_t *shape, int ndim) {
  if (!ctx || !x || (ndim > 0 && !shape)) return NULL;
  if (ndim == 0) return poly_reshape(ctx, x, NULL, 0);
  return poly_reshape(ctx, x, (int64_t *)shape, ndim);
}

PolyUOp *poly_mean_reduce(PolyCtx *ctx, PolyUOp *x,
                          const int64_t *shape, int ndim,
                          int axis, int keepdim,
                          int64_t *out_shape, int *out_ndim) {
  if (axis < 0) axis += ndim;
  int64_t count = shape[axis];
  PolyUOp *s = do_reduce(ctx, POLY_OP_ADD, x, shape, ndim, axis, keepdim,
                          out_shape, out_ndim);
  return poly_alu2(ctx, POLY_OP_FDIV, s, cf(ctx, x, (double)count));
}

PolyUOp *poly_var_reduce(PolyCtx *ctx, PolyUOp *x,
                         const int64_t *shape, int ndim,
                         int axis, int keepdim, int correction,
                         int64_t *out_shape, int *out_ndim) {
  if (axis < 0) axis += ndim;
  int64_t count = shape[axis];
  PolyUOp *x_view = reshape_logical_input(ctx, x, shape, ndim);
  if (!x_view) return NULL;
  /* var(x) = mean((x - mean(x))^2) * count / (count - correction) */
  /* First get mean with keepdim=1 for broadcast */
  int64_t mean_shape[8]; int mean_ndim;
  PolyUOp *m = poly_mean_reduce(ctx, x_view, shape, ndim, axis, 1,
                                 mean_shape, &mean_ndim);
  /* Expand mean back to full shape for subtraction */
  PolyUOp *m_expanded = poly_expand(ctx, m, (int64_t *)shape, ndim);
  /* (x - mean)^2 */
  PolyUOp *diff = poly_alu2(ctx, POLY_OP_SUB, x_view, m_expanded);
  PolyUOp *sq = poly_alu2(ctx, POLY_OP_MUL, diff, diff);
  /* sum of squares / (count - correction) */
  PolyUOp *s = do_reduce(ctx, POLY_OP_ADD, sq, shape, ndim, axis, keepdim,
                          out_shape, out_ndim);
  double divisor = (double)(count - correction);
  if (divisor <= 0.0) divisor = 1.0;
  return poly_alu2(ctx, POLY_OP_FDIV, s, cf(ctx, x, divisor));
}

PolyUOp *poly_logsumexp(PolyCtx *ctx, PolyUOp *x,
                        const int64_t *shape, int ndim,
                        int axis, int keepdim,
                        int64_t *out_shape, int *out_ndim) {
  if (axis < 0) axis += ndim;
  PolyUOp *x_view = reshape_logical_input(ctx, x, shape, ndim);
  if (!x_view) return NULL;
  int64_t keep_shape[POLY_MAX_DIMS];
  int keep_ndim = 0;
  PolyUOp *m = poly_max_reduce(ctx, x_view, shape, ndim, axis, 1, keep_shape, &keep_ndim);
  PolyUOp *shifted = poly_alu2(ctx, POLY_OP_SUB, x_view, m); /* keepdim=1 broadcasts */
  PolyUOp *e = poly_exp(ctx, shifted);
  PolyUOp *s = poly_sum_reduce(ctx, e, shape, ndim, axis, 1, keep_shape, &keep_ndim);
  PolyUOp *lse_keep = poly_alu2(ctx, POLY_OP_ADD, poly_log(ctx, s), m);
  if (keepdim) {
    memcpy(out_shape, keep_shape, (size_t)keep_ndim * sizeof(int64_t));
    *out_ndim = keep_ndim;
    return lse_keep;
  }

  int64_t final_shape[POLY_MAX_DIMS];
  int fn = 0;
  for (int i = 0; i < ndim; i++) {
    if (i == axis) continue;
    final_shape[fn++] = shape[i];
  }
  if (fn == 0) {
    *out_ndim = 0;
    return poly_reshape(ctx, lse_keep, NULL, 0);
  }
  memcpy(out_shape, final_shape, (size_t)fn * sizeof(int64_t));
  *out_ndim = fn;
  return poly_reshape(ctx, lse_keep, final_shape, fn);
}

PolyUOp *poly_dot(PolyCtx *ctx,
                  PolyUOp *x, const int64_t *x_shape, int x_ndim,
                  PolyUOp *w, const int64_t *w_shape, int w_ndim,
                  int64_t *out_shape, int *out_ndim) {
  /* Port of tinygrad's dot: reshape+transpose+mul+sum.
   * x: (..., K), w: (..., K, N) or (K,) or (K, N) */
  if (!ctx || !x || !w || !x_shape || !w_shape || !out_shape || !out_ndim) return NULL;
  *out_ndim = 0;
  if (x_ndim < 1 || w_ndim < 1 || x_ndim > POLY_MAX_DIMS || w_ndim > POLY_MAX_DIMS) return NULL;

  int64_t K = x_shape[x_ndim - 1];
  int axis_w = w_ndim - (w_ndim >= 2 ? 2 : 1); /* -min(w_ndim, 2) */
  if (K != w_shape[axis_w]) return NULL;

  /* x_new = x.reshape(*x.shape[:-1], *[1]*min(dx-1,dw-1,1), x.shape[-1]) */
  int64_t xs[POLY_MAX_DIMS]; int xn = 0;
  for (int i = 0; i < x_ndim - 1; i++) xs[xn++] = x_shape[i];
  int n_ones_x;
  /* min(dx-1, dw-1, 1) */
  {
    int a = x_ndim - 1, b = w_ndim - 1;
    n_ones_x = a < b ? a : b;
    if (n_ones_x > 1) n_ones_x = 1;
  }
  for (int i = 0; i < n_ones_x; i++) xs[xn++] = 1;
  xs[xn++] = K;
  PolyUOp *xr = poly_reshape(ctx, x, xs, xn);

  /* w_new = w.reshape(*w.shape[:-2], *[1]*min(dx-1,dw-1,1), *w.shape[axis_w:]) */
  int64_t ws[POLY_MAX_DIMS]; int wn = 0;
  for (int i = 0; i < w_ndim - 2; i++) ws[wn++] = w_shape[i];
  for (int i = 0; i < n_ones_x; i++) ws[wn++] = 1;
  for (int i = axis_w; i < w_ndim; i++) ws[wn++] = w_shape[i];
  PolyUOp *wr = poly_reshape(ctx, w, ws, wn);

  /* w_new = w_new.transpose(-1, axis_w_in_new) */
  /* After reshape, axis_w in the new array is at position (wn - (w_ndim - axis_w)) */
  /* For simple 2D case (K, N), after reshape with 1 inserted it's (1, K, N)
   * and we want to transpose(-1, -2) = swap last two. */
  int new_axis_w = wn - 2; /* position of K in reshaped w */
  if (new_axis_w < 0) new_axis_w = 0;
  int64_t perm[POLY_MAX_DIMS];
  for (int i = 0; i < wn; i++) perm[i] = i;
  perm[wn - 1] = new_axis_w;
  perm[new_axis_w] = wn - 1;
  PolyUOp *wt = poly_permute(ctx, wr, perm, wn);
  int64_t wt_shape[POLY_MAX_DIMS];
  for (int i = 0; i < wn; i++) wt_shape[i] = ws[perm[i]];

  /* Broadcast x_reshaped and w_transposed, then multiply */
  /* Broadcast shapes: xs (xn dims) and wt_shape (wn dims) */
  int max_ndim = xn > wn ? xn : wn;
  int64_t bc_shape[POLY_MAX_DIMS];
  for (int i = 0; i < max_ndim; i++) {
    int xi = i - (max_ndim - xn);
    int wi = i - (max_ndim - wn);
    int64_t xd = (xi >= 0) ? xs[xi] : 1;
    int64_t wd = (wi >= 0) ? wt_shape[wi] : 1;
    if (xd != wd && xd != 1 && wd != 1) return NULL;
    bc_shape[i] = xd > wd ? xd : wd;
  }

  /* Left-pad smaller tensor with 1s before expand (not bc_shape!) */
  PolyUOp *x_exp, *w_exp;
  if (xn < max_ndim) {
    int64_t padded[POLY_MAX_DIMS];
    int pad = max_ndim - xn;
    for (int i = 0; i < pad; i++) padded[i] = 1;
    for (int i = 0; i < xn; i++) padded[pad + i] = xs[i];
    x_exp = poly_reshape(ctx, xr, padded, max_ndim);
  } else {
    x_exp = xr;
  }
  if (wn < max_ndim) {
    int64_t padded[POLY_MAX_DIMS];
    int pad = max_ndim - wn;
    for (int i = 0; i < pad; i++) padded[i] = 1;
    for (int i = 0; i < wn; i++) padded[pad + i] = wt_shape[i];
    w_exp = poly_reshape(ctx, wt, padded, max_ndim);
  } else {
    w_exp = wt;
  }
  x_exp = poly_expand(ctx, x_exp, bc_shape, max_ndim);
  w_exp = poly_expand(ctx, w_exp, bc_shape, max_ndim);

  PolyUOp *mul = poly_alu2(ctx, POLY_OP_MUL, x_exp, w_exp);

  /* Sum over last axis (the K dimension) */
  int64_t sum_axis[] = { max_ndim - 1 };
  PolyUOp *summed = poly_reduce_axis(ctx, POLY_OP_ADD, mul, sum_axis, 1);

  /* Output shape: remove last dim */
  int on = 0;
  for (int i = 0; i < max_ndim - 1; i++) {
    out_shape[on++] = bc_shape[i];
  }
  if (on == 0) { out_shape[0] = 1; on = 1; }
  *out_ndim = on;

  return poly_reshape(ctx, summed, out_shape, on);
}

PolyUOp *poly_softmax(PolyCtx *ctx, PolyUOp *x,
                      const int64_t *shape, int ndim, int axis) {
  /* softmax(x) = exp(x - max(x, keepdim=True)) / sum(exp(...), keepdim=True) */
  if (axis < 0) axis += ndim;
  PolyUOp *x_view = reshape_logical_input(ctx, x, shape, ndim);
  if (!x_view) return NULL;

  /* max with keepdim=1 */
  int64_t max_shape[8]; int max_ndim;
  PolyUOp *m = do_reduce(ctx, POLY_OP_MAX, x_view, shape, ndim, axis, 1,
                          max_shape, &max_ndim);
  /* Expand max back to full shape */
  PolyUOp *m_exp = poly_expand(ctx, m, (int64_t *)shape, ndim);

  /* x - max (numerically stable) */
  PolyUOp *shifted = poly_alu2(ctx, POLY_OP_SUB, x_view, m_exp);

  /* exp */
  PolyUOp *e = poly_exp(ctx, shifted);

  /* sum with keepdim=1 */
  int64_t sum_shape[8]; int sum_ndim;
  PolyUOp *s = do_reduce(ctx, POLY_OP_ADD, e, shape, ndim, axis, 1,
                          sum_shape, &sum_ndim);
  PolyUOp *s_exp = poly_expand(ctx, s, (int64_t *)shape, ndim);

  /* e / sum */
  return poly_alu2(ctx, POLY_OP_FDIV, e, s_exp);
}

PolyUOp *poly_log_softmax(PolyCtx *ctx, PolyUOp *x,
                          const int64_t *shape, int ndim, int axis) {
  /* log_softmax(x) = (x - max) - log(sum(exp(x - max), keepdim=True)) */
  if (axis < 0) axis += ndim;
  PolyUOp *x_view = reshape_logical_input(ctx, x, shape, ndim);
  if (!x_view) return NULL;

  int64_t max_shape[8]; int max_ndim;
  PolyUOp *m = do_reduce(ctx, POLY_OP_MAX, x_view, shape, ndim, axis, 1,
                          max_shape, &max_ndim);
  PolyUOp *m_exp = poly_expand(ctx, m, (int64_t *)shape, ndim);
  PolyUOp *shifted = poly_alu2(ctx, POLY_OP_SUB, x_view, m_exp);
  PolyUOp *e = poly_exp(ctx, shifted);

  int64_t sum_shape[8]; int sum_ndim;
  PolyUOp *s = do_reduce(ctx, POLY_OP_ADD, e, shape, ndim, axis, 1,
                          sum_shape, &sum_ndim);
  PolyUOp *log_s = poly_log(ctx, s);
  PolyUOp *log_s_exp = poly_expand(ctx, log_s, (int64_t *)shape, ndim);

  return poly_alu2(ctx, POLY_OP_SUB, shifted, log_s_exp);
}

PolyUOp *poly_cross_entropy(PolyCtx *ctx,
                            PolyUOp *logits, const int64_t *logits_shape, int logits_ndim,
                            PolyUOp *target, const int64_t *target_shape, int target_ndim,
                            int axis, int64_t *out_shape, int *out_ndim) {
  if (!ctx || !logits || !target || !logits_shape || !target_shape || !out_shape || !out_ndim) return NULL;
  *out_ndim = 0;
  if (logits_ndim < 1 || logits_ndim > POLY_MAX_DIMS || target_ndim < 0 || target_ndim > POLY_MAX_DIMS) return NULL;

  if (axis < 0) axis += logits_ndim;
  if (axis < 0 || axis >= logits_ndim) return NULL;

  const bool dense_targets = shape_equal(logits_shape, logits_ndim, target_shape, target_ndim);
  const bool sparse_targets = shape_equal_except_axis(logits_shape, logits_ndim, target_shape, target_ndim, axis);
  if (!dense_targets && !sparse_targets) return NULL;

  /* For dense targets, wrap in reshape so the scheduler treats it as a loadable
   * tensor (raw BUFFER in ALU tree renders as pointer, not loaded value). */
  PolyUOp *weights = dense_targets
    ? poly_reshape(ctx, target, (int64_t *)target_shape, target_ndim)
    : target;
  if (sparse_targets) {
    const int64_t classes = logits_shape[axis];
    int64_t target_us_shape[POLY_MAX_DIMS];
    for (int i = 0; i < axis; i++) target_us_shape[i] = target_shape[i];
    target_us_shape[axis] = 1;
    for (int i = axis; i < target_ndim; i++) target_us_shape[i + 1] = target_shape[i];

    PolyUOp *target_us = poly_reshape(ctx, target, target_us_shape, target_ndim + 1);
    PolyUOp *target_exp = poly_expand(ctx, target_us, (int64_t *)logits_shape, logits_ndim);

    PolyUOp *classes_uop = poly_arange(ctx, 0.0, (double)classes, 1.0);
    int64_t classes_shape[POLY_MAX_DIMS];
    for (int i = 0; i < logits_ndim; i++) classes_shape[i] = 1;
    classes_shape[axis] = classes;
    PolyUOp *classes_r = poly_reshape(ctx, classes_uop, classes_shape, logits_ndim);
    PolyUOp *classes_exp = poly_expand(ctx, classes_r, (int64_t *)logits_shape, logits_ndim);

    weights = poly_eq(ctx, target_exp, classes_exp);
  }

  PolyUOp *log_probs = poly_log_softmax(ctx, logits, logits_shape, logits_ndim, axis);
  PolyUOp *weighted = poly_alu2(ctx, POLY_OP_MUL, log_probs, weights);

  int64_t per_sample_shape[POLY_MAX_DIMS];
  int per_sample_ndim = 0;
  PolyUOp *per_sample = do_reduce(ctx, POLY_OP_ADD, weighted, logits_shape, logits_ndim, axis, 0,
                                  per_sample_shape, &per_sample_ndim);

  PolyUOp *total = per_sample;
  if (per_sample_ndim > 0) {
    int64_t axes[POLY_MAX_DIMS];
    for (int i = 0; i < per_sample_ndim; i++) axes[i] = i;
    total = poly_reduce_axis(ctx, POLY_OP_ADD, per_sample, axes, per_sample_ndim);
  }

  int64_t denom = shape_numel_checked(per_sample_shape, per_sample_ndim);
  if (denom <= 0) return NULL;

  PolyUOp *mean = poly_alu2(ctx, POLY_OP_FDIV, total, cf(ctx, log_probs, (double)denom));
  PolyUOp *loss = poly_alu1(ctx, POLY_OP_NEG, mean);
  *out_ndim = 0;
  return poly_reshape(ctx, loss, NULL, 0);
}

/* ── Shared helpers ──────────────────────────────────────────────────── */

#define MAX_REALIZE_BUFS 256

/* Reconstruct the buffer-to-PARAM ordering that poly_schedule() uses:
 * 1. Output buffers (STORE targets in SINK source order)
 * 2. Remaining input buffers (toposort encounter order) */
static int collect_ordered_buffers(PolyCtx *ctx, PolyUOp *tensor_sink,
                                    PolyUOp **ordered, int max_bufs) {
  int n = 0;

  /* Output buffers first */
  for (int i = 0; i < tensor_sink->n_src; i++) {
    PolyUOp *store = tensor_sink->src[i];
    if (store->op == POLY_OP_STORE && store->n_src >= 1 &&
        store->src[0]->op == POLY_OP_BUFFER) {
      PolyUOp *buf = store->src[0];
      /* Dedup */
      bool dup = false;
      for (int j = 0; j < n; j++) {
        if (ordered[j] == buf) { dup = true; break; }
      }
      if (!dup && n < max_bufs) ordered[n++] = buf;
    }
  }

  /* Input buffers in toposort order */
  int n_topo;
  PolyUOp **topo = poly_toposort(ctx, tensor_sink, &n_topo);
  for (int i = 0; i < n_topo; i++) {
    if (topo[i]->op == POLY_OP_BUFFER) {
      bool dup = false;
      for (int j = 0; j < n; j++) {
        if (ordered[j] == topo[i]) { dup = true; break; }
      }
      if (!dup && n < max_bufs) ordered[n++] = topo[i];
    }
  }

  return n;
}

/* ── Pointer hash/eq helpers (used by kernel cache and CPU realize) ──── */

static bool ptr_eq(const void *a, const void *b) { return a == b; }
static uint32_t ptr_hash(const void *p) {
  uintptr_t v = (uintptr_t)p;
  return (uint32_t)(v ^ (v >> 16) ^ (sizeof(v) > 4 ? (uint32_t)(v >> 32) : 0));
}

/* ── Constant buffer registry (ctx-scoped) ─────────────────────────────
 *
 * Some additive frontend creation helpers (arange/full/rand/...) return
 * BUFFER-backed tensors. We retain host data here and auto-bind it when a
 * caller omits bindings for those buffers.
 */

typedef struct {
  PolyCtx *ctx;
  PolyUOp *buf;
  void *data;
} PolyConstBindingEntry;

static PolyConstBindingEntry *g_const_bindings = NULL;
static int g_const_bindings_n = 0;
static int g_const_bindings_cap = 0;

static void const_registry_add(PolyCtx *ctx, PolyUOp *buf, void *data) {
  if (!ctx || !buf || !data) return;
  for (int i = 0; i < g_const_bindings_n; i++) {
    if (g_const_bindings[i].ctx == ctx && g_const_bindings[i].buf == buf) {
      free(g_const_bindings[i].data);
      g_const_bindings[i].data = data;
      return;
    }
  }
  if (g_const_bindings_n == g_const_bindings_cap) {
    int new_cap = (g_const_bindings_cap == 0) ? 64 : (g_const_bindings_cap * 2);
    PolyConstBindingEntry *nb = realloc(g_const_bindings, (size_t)new_cap * sizeof(*nb));
    if (!nb) {
      free(data);
      return;
    }
    g_const_bindings = nb;
    g_const_bindings_cap = new_cap;
  }
  g_const_bindings[g_const_bindings_n++] = (PolyConstBindingEntry){
    .ctx = ctx, .buf = buf, .data = data
  };
}

static void *const_registry_lookup(PolyCtx *ctx, PolyUOp *buf) {
  if (!ctx || !buf) return NULL;
  for (int i = 0; i < g_const_bindings_n; i++) {
    if (g_const_bindings[i].ctx == ctx && g_const_bindings[i].buf == buf)
      return g_const_bindings[i].data;
  }
  return NULL;
}

static bool const_registry_has(PolyCtx *ctx, PolyUOp *buf) {
  return const_registry_lookup(ctx, buf) != NULL;
}

void poly_frontend_ctx_cleanup(PolyCtx *ctx) {
  if (!ctx || g_const_bindings_n == 0) return;
  int wr = 0;
  for (int i = 0; i < g_const_bindings_n; i++) {
    if (g_const_bindings[i].ctx == ctx) {
      free(g_const_bindings[i].data);
      continue;
    }
    if (wr != i) g_const_bindings[wr] = g_const_bindings[i];
    wr++;
  }
  g_const_bindings_n = wr;
}

/* ── Structural hash/eq for kernel cache ─────────────────────────────
 *
 * The kernel cache needs to match computations that are structurally
 * identical but use different BUFFER UOp instances (e.g., each training
 * step creates new buffers).  We hash/compare the computation DAG
 * structure: ops, dtypes, args, and connectivity — treating BUFFER
 * nodes as positional placeholders (first encountered = 0, etc.).
 */

#define MAX_STRUCT_NODES 8192

typedef struct {
  PolyUOp *uop;
  uint32_t hash;
} StructVisited;

typedef struct {
  PolyUOp *uop;
  int id;
} StructBuf;

typedef struct {
  StructVisited visited[MAX_STRUCT_NODES];
  int n_visited;
  StructBuf bufs[MAX_REALIZE_BUFS];
  int n_bufs;
} StructHashCtx;

static uint32_t struct_hash_impl(PolyUOp *u, StructHashCtx *ctx) {
  /* Check if already visited (handles DAG sharing) */
  for (int i = 0; i < ctx->n_visited; i++) {
    if (ctx->visited[i].uop == u) return ctx->visited[i].hash;
  }

  uint32_t h = 0x811c9dc5; /* FNV-1a offset basis */

  if (u->op == POLY_OP_BUFFER) {
    /* BUFFER nodes: use positional ID instead of pointer identity */
    int buf_id = -1;
    for (int i = 0; i < ctx->n_bufs; i++) {
      if (ctx->bufs[i].uop == u) { buf_id = ctx->bufs[i].id; break; }
    }
    if (buf_id < 0 && ctx->n_bufs < MAX_REALIZE_BUFS) {
      buf_id = ctx->n_bufs;
      ctx->bufs[ctx->n_bufs].uop = u;
      ctx->bufs[ctx->n_bufs].id = buf_id;
      ctx->n_bufs++;
    }
    h ^= (uint32_t)u->op;        h *= 0x01000193;
    h ^= (uint32_t)u->dtype.priority; h *= 0x01000193;
    h ^= (uint32_t)u->dtype.bitsize;  h *= 0x01000193;
    h ^= (uint32_t)buf_id;       h *= 0x01000193;
    h ^= poly_arg_hash(u->arg);   h *= 0x01000193;
  } else {
    h ^= (uint32_t)u->op;        h *= 0x01000193;
    h ^= (uint32_t)u->dtype.priority; h *= 0x01000193;
    h ^= (uint32_t)u->dtype.bitsize;  h *= 0x01000193;
    for (int i = 0; i < u->n_src; i++) {
      h ^= struct_hash_impl(u->src[i], ctx); h *= 0x01000193;
    }
    h ^= poly_arg_hash(u->arg);   h *= 0x01000193;
  }

  /* Memoize */
  if (ctx->n_visited < MAX_STRUCT_NODES) {
    ctx->visited[ctx->n_visited].uop = u;
    ctx->visited[ctx->n_visited].hash = h;
    ctx->n_visited++;
  }
  return h;
}

static uint32_t structural_hash(PolyUOp *u) {
  StructHashCtx *ctx = calloc(1, sizeof(StructHashCtx));
  uint32_t h = struct_hash_impl(u, ctx);
  free(ctx);
  return h;
}

/* ── Structural equality ─────────────────────────────────────────────── */

typedef struct {
  PolyUOp *a[MAX_REALIZE_BUFS];
  PolyUOp *b[MAX_REALIZE_BUFS];
  int n;
} BufPairs;

typedef struct {
  PolyUOp *a[MAX_STRUCT_NODES];
  PolyUOp *b[MAX_STRUCT_NODES];
  int n;
} EqVisited;

static bool struct_eq_impl(PolyUOp *a, PolyUOp *b, BufPairs *bp, EqVisited *ev) {
  if (a == b) return true;
  if (!a || !b) return false;

  /* Check if this pair already visited (DAG sharing) */
  for (int i = 0; i < ev->n; i++) {
    if (ev->a[i] == a) return ev->b[i] == b;
    if (ev->b[i] == b) return false;
  }

  /* Mark visited */
  if (ev->n < MAX_STRUCT_NODES) {
    ev->a[ev->n] = a;
    ev->b[ev->n] = b;
    ev->n++;
  }

  /* Both BUFFER? Track correspondence */
  if (a->op == POLY_OP_BUFFER && b->op == POLY_OP_BUFFER) {
    if (!poly_dtype_eq(a->dtype, b->dtype)) return false;
    if (!poly_arg_eq(a->arg, b->arg)) return false;
    /* Check existing mapping */
    for (int i = 0; i < bp->n; i++) {
      if (bp->a[i] == a) return bp->b[i] == b;
      if (bp->b[i] == b) return false;
    }
    /* New pair */
    if (bp->n < MAX_REALIZE_BUFS) {
      bp->a[bp->n] = a;
      bp->b[bp->n] = b;
      bp->n++;
    }
    return true;
  }

  /* Same op, dtype, n_src, arg? */
  if (a->op != b->op) return false;
  if (!poly_dtype_eq(a->dtype, b->dtype)) return false;
  if (a->n_src != b->n_src) return false;
  if (!poly_arg_eq(a->arg, b->arg)) return false;

  /* Recursively compare sources */
  for (int i = 0; i < a->n_src; i++) {
    if (!struct_eq_impl(a->src[i], b->src[i], bp, ev)) return false;
  }
  return true;
}

static bool structural_eq(const void *a, const void *b) {
  BufPairs bp = { .n = 0 };
  EqVisited ev = { .n = 0 };
  return struct_eq_impl((PolyUOp *)a, (PolyUOp *)b, &bp, &ev);
}

/* DFS to assign positional IDs to BUFFER nodes, matching structural_hash order.
 * Children are visited left-to-right, same as struct_hash_impl().
 * n_bufs counts total BUFFERs found (may exceed buf_order capacity).
 * buf_order is only written up to MAX_REALIZE_BUFS entries.
 * Callers must check *n_bufs <= MAX_REALIZE_BUFS after the call. */
static void collect_buf_order(PolyUOp *u, PolyUOp **buf_order, int *n_bufs,
                               PolyUOp **visited, int *n_visited) {
  if (!u) return;
  for (int i = 0; i < *n_visited; i++)
    if (visited[i] == u) return;
  if (*n_visited < MAX_STRUCT_NODES) visited[(*n_visited)++] = u;

  if (u->op == POLY_OP_BUFFER) {
    if (*n_bufs < MAX_REALIZE_BUFS)
      buf_order[*n_bufs] = u;
    (*n_bufs)++;  /* always count, even past capacity */
    return;
  }
  for (int i = 0; i < u->n_src; i++)
    collect_buf_order(u->src[i], buf_order, n_bufs, visited, n_visited);
}

static int find_buf_position(PolyUOp *buf, PolyUOp **buf_order, int n_bufs) {
  for (int i = 0; i < n_bufs; i++)
    if (buf_order[i] == buf) return i;
  return -1;
}

/* ── CPU realize (not available in Emscripten) ───────────────────────── */

#ifndef __EMSCRIPTEN__

static int realize_counter = 0;

/* Defensive graph validator to avoid crashing in toposort/linearize when
 * a malformed kernel contains NULL sources. */
static bool validate_kernel_graph(PolyCtx *ctx, PolyUOp *root) {
  if (!root) return false;
  PolyMap *visited = poly_map_new(256);
  PolyUOp *stack[4096];
  PolyUOp *parent_stack[4096];
  int parent_src_idx[4096];
  int sp = 0;
  stack[sp++] = root;
  parent_stack[0] = NULL;
  parent_src_idx[0] = -1;

  while (sp > 0) {
    sp--;
    PolyUOp *u = stack[sp];
    PolyUOp *parent = parent_stack[sp];
    int src_idx = parent_src_idx[sp];
    if (!u) { poly_map_destroy(visited); return false; }
    if (!poly_ctx_owns_ptr(ctx, u)) {
      if (parent) {
        fprintf(stderr, "polygrad: realize: foreign/stale UOp pointer %p referenced by %s(%p) src[%d]\n",
                (void *)u, poly_op_name(parent->op), (void *)parent, src_idx);
        fprintf(stderr, "polygrad: realize: parent %s n_src=%d\n", poly_op_name(parent->op), parent->n_src);
        for (int si = 0; si < parent->n_src; si++) {
          PolyUOp *ps = parent->src[si];
          bool owned = poly_ctx_owns_ptr(ctx, ps);
          fprintf(stderr, "  parent.src[%d]=%p %s%s\n", si, (void *)ps,
                  owned ? "" : "[FOREIGN] ",
                  (owned && ps) ? poly_op_name(ps->op) : "");
        }
      } else {
        fprintf(stderr, "polygrad: realize: foreign/stale root UOp pointer %p in kernel graph\n", (void *)u);
      }
      poly_map_destroy(visited);
      return false;
    }
    if (poly_map_get(visited, ptr_hash(u), u, ptr_eq)) continue;
    poly_map_set(visited, ptr_hash(u), u, u, ptr_eq);

    if (u->n_src < 0 || u->n_src > 64) {
      fprintf(stderr, "polygrad: realize: invalid n_src=%d on %s(%p)\n",
              u->n_src, poly_op_name(u->op), (void *)u);
      poly_map_destroy(visited);
      return false;
    }
    for (int i = 0; i < u->n_src; i++) {
      if (!u->src[i]) {
        fprintf(stderr, "polygrad: realize: NULL src[%d] on %s(%p), n_src=%d\n",
                i, poly_op_name(u->op), (void *)u, u->n_src);
        poly_map_destroy(visited);
        return false;
      }
      if (sp < (int)(sizeof(stack) / sizeof(stack[0]))) {
        stack[sp++] = u->src[i];
        parent_stack[sp - 1] = u;
        parent_src_idx[sp - 1] = i;
      }
    }
  }
  poly_map_destroy(visited);
  return true;
}

/* ── Compiled program cache ───────────────────────────────────────────
 *
 * Caches compiled PolyProgram* (CPU) and PolyCudaProgram* (CUDA) keyed by
 * the structural hash of the tensor-level SINK.  Avoids re-running
 * cc/NVRTC on every realize() call for the same computation.
 */

#define PROG_CACHE_CAP 512

typedef struct {
  uint32_t hash;
  PolyProgram *prog;
} CpuCacheEntry;

static CpuCacheEntry cpu_prog_cache[PROG_CACHE_CAP];
static int cpu_prog_cache_n = 0;

static PolyProgram *cpu_cache_get(uint32_t h) {
  for (int i = 0; i < cpu_prog_cache_n; i++)
    if (cpu_prog_cache[i].hash == h) return cpu_prog_cache[i].prog;
  return NULL;
}

void poly_cpu_cache_flush(void) {
  for (int i = 0; i < cpu_prog_cache_n; i++)
    poly_program_destroy(cpu_prog_cache[i].prog);
  cpu_prog_cache_n = 0;
}

static void cpu_cache_put(uint32_t h, PolyProgram *prog) {
  if (cpu_prog_cache_n < PROG_CACHE_CAP) {
    cpu_prog_cache[cpu_prog_cache_n].hash = h;
    cpu_prog_cache[cpu_prog_cache_n].prog = prog;
    cpu_prog_cache_n++;
  } else {
    /* Cache full: destroy program immediately to avoid leak */
    poly_program_destroy(prog);
  }
}

/* ── Schedule cache ──────────────────────────────────────────────────
 *
 * Caches the scheduling result (param-to-buffer-position mappings,
 * intermediate buffer sizes) keyed by structural hash.  On cache hit,
 * poly_realize() skips poly_schedule_v2() entirely — the dominant
 * per-realize cost after compilation is cached.
 *
 * Each param in a kernel maps to either:
 *   - A BUFFER at a specific DFS position (matched to bindings at runtime), or
 *   - An intermediate buffer (by index into allocated intermediates).
 *
 * Buffer positions are assigned by DFS encounter order from the SINK,
 * matching the order used by structural_hash().  This makes the cache
 * independent of binding array ordering.
 */

#define SCHED_CACHE_CAP 512
#define SCHED_MAX_PARAMS 32
#define SCHED_MAX_KERNELS 16
#define SCHED_MAX_VARS 8
#define SCHED_CACHE_VERSION 3u

typedef struct {
  int buf_position;      /* DFS positional ID of BUFFER, or -1 if intermediate */
  int intermediate_idx;  /* index into intermediates[], or -1 if binding */
} ParamMapping;

typedef struct {
  uint32_t hash;
  int n_kernels;
  int n_bufs;  /* total BUFFER nodes in the graph */
  int kernel_n_params[SCHED_MAX_KERNELS];
  ParamMapping mappings[SCHED_MAX_KERNELS][SCHED_MAX_PARAMS];
  int n_intermediates;
  int intermediate_sizes[SCHED_MAX_KERNELS]; /* max n_intermediates */
  int intermediate_itemsizes[SCHED_MAX_KERNELS]; /* itemsize per element */
  int exec_order[SCHED_MAX_KERNELS]; /* kernel_index per step; identity if no deps */
  /* DEFINE_VAR support: per-kernel var params (appended after buffer params in args[]) */
  int kernel_n_vars[SCHED_MAX_KERNELS];
  PolyUOp *var_ptrs[SCHED_MAX_KERNELS][SCHED_MAX_VARS]; /* DEFINE_VAR UOp pointers */
} SchedCacheEntry;

static SchedCacheEntry sched_cache[SCHED_CACHE_CAP];
static int sched_cache_n = 0;

static SchedCacheEntry *sched_cache_get(uint32_t h) {
  for (int i = 0; i < sched_cache_n; i++)
    if (sched_cache[i].hash == h) return &sched_cache[i];
  return NULL;
}

static void sched_cache_put(uint32_t h, PolyScheduleResult *sr,
                             PolyUOp *tensor_sink) {
  if (sched_cache_n >= SCHED_CACHE_CAP) return;
  if (sr->n_kernels > SCHED_MAX_KERNELS) return;

  /* Compute DFS buffer ordering from the tensor SINK */
  PolyUOp *buf_order[MAX_REALIZE_BUFS];
  PolyUOp *dfs_visited[MAX_STRUCT_NODES];
  int n_bufs = 0, n_dfs = 0;
  collect_buf_order(tensor_sink, buf_order, &n_bufs, dfs_visited, &n_dfs);

  /* Refuse caching if graph exceeds fixed-size arrays */
  if (n_bufs > MAX_REALIZE_BUFS || n_dfs >= MAX_STRUCT_NODES) return;

  SchedCacheEntry *e = &sched_cache[sched_cache_n];
  e->hash = h;
  e->n_kernels = sr->n_kernels;
  e->n_bufs = n_bufs;
  e->n_intermediates = sr->n_intermediates;

  for (int b = 0; b < sr->n_intermediates && b < SCHED_MAX_KERNELS; b++) {
    e->intermediate_sizes[b] = sr->intermediate_sizes[b];
    e->intermediate_itemsizes[b] = (sr->intermediate_itemsizes && sr->intermediate_itemsizes[b] > 0)
                                     ? sr->intermediate_itemsizes[b] : (int)sizeof(float);
  }

  /* Store exec_order (identity permutation if NULL) */
  if (sr->exec_order) {
    memcpy(e->exec_order, sr->exec_order,
           sr->n_kernels * sizeof(int));
  } else {
    for (int k = 0; k < sr->n_kernels; k++) e->exec_order[k] = k;
  }

  bool cache_ok = true;
  for (int k = 0; k < sr->n_kernels; k++) {
    int np = sr->kernel_n_params[k];
    if (np > SCHED_MAX_PARAMS) return; /* too many params, skip caching */
    e->kernel_n_params[k] = np;

    for (int i = 0; i < np; i++) {
      PolyUOp *buf = sr->param_to_buf[k][i];
      e->mappings[k][i].buf_position = -1;
      e->mappings[k][i].intermediate_idx = -1;

      if (buf->op == POLY_OP_BUFFER) {
        int pos = find_buf_position(buf, buf_order, n_bufs);
        if (pos >= 0) {
          e->mappings[k][i].buf_position = pos;
        } else if (sr->intermediate_buf_uops) {
          /* BUFFER not in tensor sink DFS — must be an intermediate */
          for (int b = 0; b < sr->n_intermediates; b++) {
            if (sr->intermediate_buf_uops[b] == buf) {
              e->mappings[k][i].intermediate_idx = b;
              break;
            }
          }
        }
      } else {
        assert(buf->op != POLY_OP_BUFFERIZE &&
               "unexpected BUFFERIZE in param_to_buf (old split path removed)");
      }

      /* Safety: if param is neither bound nor intermediate, skip caching */
      if (e->mappings[k][i].buf_position < 0 &&
          e->mappings[k][i].intermediate_idx < 0) {
        cache_ok = false;
      }
    }

    /* Store DEFINE_VAR params for this kernel */
    int nv = (sr->kernel_n_vars ? sr->kernel_n_vars[k] : 0);
    if (nv > SCHED_MAX_VARS) { cache_ok = false; nv = 0; }
    e->kernel_n_vars[k] = nv;
    for (int v = 0; v < nv; v++) {
      e->var_ptrs[k][v] = (sr->var_to_buf && sr->var_to_buf[k])
                            ? sr->var_to_buf[k][v] : NULL;
      if (!e->var_ptrs[k][v]) cache_ok = false;
    }
  }
#ifndef NDEBUG
  if (cache_ok) {
    /* Group D: no param is both external buffer and intermediate */
    for (int _k = 0; _k < sr->n_kernels; _k++) {
      for (int _i = 0; _i < e->kernel_n_params[_k]; _i++) {
        ParamMapping *_m = &e->mappings[_k][_i];
        assert(!(_m->buf_position >= 0 && _m->intermediate_idx >= 0) &&
               "sched_cache: param resolved to both external and intermediate");
      }
    }
  }
#endif
  if (cache_ok) sched_cache_n++;
}

void poly_sched_cache_flush(void) {
  sched_cache_n = 0;
}

/* Execute from cached schedule: skip poly_schedule_v2(), dispatch kernels
 * directly using cached param-to-buffer-position mappings.
 * Returns 0 on success, -1 on error (caller should fall back to full path). */
static int realize_from_sched_cache(PolyCtx *ctx, SchedCacheEntry *se,
                                     uint32_t cache_hash,
                                     PolyUOp *tensor_sink,
                                     PolyBufferBinding *bindings, int n_bindings,
                                     PolyVarBinding *var_vals, int n_var_vals) {
  /* Compute DFS buffer ordering for the current tensor SINK */
  PolyUOp *buf_order[MAX_REALIZE_BUFS];
  PolyUOp *dfs_visited[MAX_STRUCT_NODES];
  int n_bufs = 0, n_dfs = 0;
  collect_buf_order(tensor_sink, buf_order, &n_bufs, dfs_visited, &n_dfs);

  if (n_bufs > MAX_REALIZE_BUFS || n_dfs >= MAX_STRUCT_NODES) return -1;
  if (n_bufs != se->n_bufs) return -1; /* mismatch, fall back */

  /* Build position → data pointer table from current bindings */
  void *pos_to_data[MAX_REALIZE_BUFS];
  memset(pos_to_data, 0, sizeof(pos_to_data));
  for (int j = 0; j < n_bindings; j++) {
    int pos = find_buf_position(bindings[j].buffer, buf_order, n_bufs);
    if (pos >= 0) pos_to_data[pos] = bindings[j].data;
  }

  int n_int = se->n_intermediates;
  void **intermediates = NULL;
  if (n_int > 0) {
    intermediates = calloc(n_int, sizeof(void *));
    for (int b = 0; b < n_int; b++) {
      int itemsize = se->intermediate_itemsizes[b] > 0
                       ? se->intermediate_itemsizes[b] : (int)sizeof(float);
      intermediates[b] = calloc(se->intermediate_sizes[b], itemsize);
    }
  }

  int ret = 0;
  /* Stack storage for DEFINE_VAR int values (persists through _call dispatch) */
  int var_int_storage[16];
  int n_var_ints = 0;
  for (int step = 0; step < se->n_kernels && ret == 0; step++) {
    int k = se->exec_order[step];
    int np = se->kernel_n_params[k];
    int nv = se->kernel_n_vars[k];
    int n_total = np + nv;
    uint32_t kh = cache_hash + (uint32_t)k;  /* key by kernel_index, not step */

    /* All kernels must be in the program cache for fast path */
    PolyProgram *prog = cpu_cache_get(kh);
    if (!prog) { ret = -1; break; }

    void **args = calloc((size_t)n_total, sizeof(void *));
    /* Fill buffer params */
    for (int i = 0; i < np; i++) {
      ParamMapping *m = &se->mappings[k][i];
      if (m->buf_position >= 0 && m->buf_position < n_bufs) {
        args[i] = pos_to_data[m->buf_position];
      /* Guard intermediates != NULL: when n_int == 0, intermediates is NULL
       * and the idx < n_int check would be false, but the analyzer can't
       * prove that across the two separate conditions. */
      } else if (intermediates && m->intermediate_idx >= 0 &&
                 m->intermediate_idx < n_int) {
        args[i] = intermediates[m->intermediate_idx];
      }
      if (!args[i]) { ret = -1; free(args); args = NULL; break; }
    }

    /* Fill var params: args[np..np+nv-1] = DEFINE_VAR int values */
    if (ret == 0 && args && nv > 0) {
      n_var_ints = 0;
      for (int v = 0; v < nv; v++) {
        PolyUOp *var = se->var_ptrs[k][v];
        bool found = false;
        for (int vb = 0; vb < n_var_vals; vb++) {
          if (var_vals[vb].var == var) {
            var_int_storage[n_var_ints] = (int)var_vals[vb].value;
            args[np + v] = &var_int_storage[n_var_ints];
            n_var_ints++;
            found = true;
            break;
          }
        }
        if (!found) { ret = -1; free(args); args = NULL; break; }
      }
    }

    if (ret == 0 && args) {
      poly_program_call(prog, args, n_total);
      free(args);
    }
  }

  if (intermediates) {
    for (int b = 0; b < n_int; b++) free(intermediates[b]);
    free(intermediates);
  }
  return ret;
}

/* Compile and execute a single kernel with the given args array.
 * Returns 0 on success, -1 on error.
 * cache_hash: structural hash of tensor_sink (0 to disable caching). */
static int compile_and_run(PolyCtx *ctx, PolyUOp *kernel,
                            void **args, int n_args, uint32_t cache_hash) {
  if (!kernel) {
    fprintf(stderr, "polygrad: realize: NULL kernel\n");
    return -1;
  }

  /* Check cache */
  if (cache_hash) {
    PolyProgram *cached = cpu_cache_get(cache_hash);
    if (cached) {
      poly_program_call(cached, args, n_args);
      return 0;
    }
  }

  if (!validate_kernel_graph(ctx, kernel)) {
    fprintf(stderr, "polygrad: realize: kernel graph validation failed\n");
    return -1;
  }
  int n_lin;
  PolyUOp **lin = poly_linearize(ctx, kernel, &n_lin);
  if (!lin) return -1;

  char fn_name[32];
  snprintf(fn_name, sizeof(fn_name), "k%d", realize_counter++);
  char *src = poly_render_c(lin, n_lin, fn_name);
  free(lin);
  if (!src) return -1;

  /* Debug: dump generated C source */
  if (getenv("POLY_DUMP_C"))
    fprintf(stderr, "=== C SOURCE (%s) ===\n%s\n=== END ===\n", fn_name, src);

  PolyProgram *prog = poly_compile_c(src, fn_name);
  if (!prog) {
    fprintf(stderr, "=== FAILED C SOURCE (%s) ===\n%s\n=== END ===\n", fn_name, src);
    free(src);
    return -1;
  }
  free(src);

  poly_program_call(prog, args, n_args);

  /* Cache the compiled program (don't destroy it) */
  if (cache_hash) {
    cpu_cache_put(cache_hash, prog);
  } else {
    poly_program_destroy(prog);
  }
  return 0;
}

/* Strip BIND values from graph, extracting {DEFINE_VAR → value} pairs.
 * Port of tinygrad's strip_bind in pm_pre_sched_cache.
 * Returns rewritten sink with BIND(DEFINE_VAR, CONST) → DEFINE_VAR.
 * Populates out_vals[0..n_out-1] with extracted bindings.
 * Also remaps buf_bindings[].buffer pointers to their rewritten equivalents
 * (so binding lookup works after graph rewrite). */
static PolyUOp *strip_bind_values(PolyCtx *ctx, PolyUOp *sink,
                                   PolyVarBinding *out_vals, int *n_out, int max_out,
                                   PolyBufferBinding *buf_bindings, int n_buf_bindings) {
  int n_uops;
  PolyUOp **topo = poly_toposort(ctx, sink, &n_uops);
  if (!topo) { *n_out = 0; return sink; }

  /* Check if any BIND nodes exist */
  bool has_bind = false;
  for (int i = 0; i < n_uops; i++) {
    if (topo[i]->op == POLY_OP_BIND) { has_bind = true; break; }
  }
  if (!has_bind) { *n_out = 0; return sink; }

  /* Bottom-up rewrite: BIND(DEFINE_VAR, CONST) → DEFINE_VAR */
  PolyMap *rmap = poly_map_new(n_uops * 2);
  int nv = 0;
  for (int i = 0; i < n_uops; i++) {
    PolyUOp *u = topo[i];
    if (u->op == POLY_OP_BIND && u->n_src >= 2 &&
        u->src[0]->op == POLY_OP_DEFINE_VAR &&
        u->src[1]->op == POLY_OP_CONST) {
      /* Extract var → value binding */
      if (nv < max_out) {
        out_vals[nv].var = u->src[0];
        out_vals[nv].value = (int32_t)u->src[1]->arg.i;
        nv++;
      }
      /* Replace BIND with DEFINE_VAR */
      poly_map_set(rmap, ptr_hash(u), u, u->src[0], ptr_eq);
      continue;
    }
    /* Rebuild node with remapped sources if any source changed */
    bool changed = false;
    PolyUOp *new_src[64];
    for (int s = 0; s < u->n_src && s < 64; s++) {
      PolyUOp *ms = poly_map_get(rmap, ptr_hash(u->src[s]), u->src[s], ptr_eq);
      new_src[s] = ms ? ms : u->src[s];
      if (new_src[s] != u->src[s]) changed = true;
    }
    if (changed) {
      PolyUOp *rebuilt = poly_uop(ctx, u->op, u->dtype, new_src, u->n_src, u->arg);
      poly_map_set(rmap, ptr_hash(u), u, rebuilt, ptr_eq);
    }
  }

  /* Get the rewritten sink */
  PolyUOp *result = poly_map_get(rmap, ptr_hash(sink), sink, ptr_eq);
  if (!result) result = sink;

  /* Remap buffer binding pointers to rewritten BUFFERs */
  for (int j = 0; j < n_buf_bindings; j++) {
    PolyUOp *remapped = poly_map_get(rmap, ptr_hash(buf_bindings[j].buffer),
                                      buf_bindings[j].buffer, ptr_eq);
    if (remapped) buf_bindings[j].buffer = remapped;
  }

  poly_map_destroy(rmap);
  /* topo is arena-allocated, no free needed */
  *n_out = nv;
  return result;
}

static int realize_impl(PolyCtx *ctx, PolyUOp *tensor_sink,
                        PolyBufferBinding *bindings, int n_bindings,
                        PolyVarBinding *var_bindings, int n_var_bindings) {
  if (!tensor_sink || tensor_sink->op != POLY_OP_SINK) {
    fprintf(stderr, "polygrad: realize: expected SINK\n");
    return -1;
  }

  /* Strip BIND values from graph (matches tinygrad's pm_pre_sched_cache).
   * BIND(DEFINE_VAR, CONST) → DEFINE_VAR, extracting var → value pairs.
   * Explicit var_bindings override extracted values. */
  PolyVarBinding bind_vals[16];
  int n_bind_vals = 0;
  tensor_sink = strip_bind_values(ctx, tensor_sink, bind_vals, &n_bind_vals, 16,
                                    bindings, n_bindings);

  /* Merge: explicit var_bindings override BIND-extracted values */
  PolyVarBinding all_var_vals[32];
  int n_all_vars = 0;
  /* Start with BIND-extracted values */
  for (int i = 0; i < n_bind_vals && n_all_vars < 32; i++)
    all_var_vals[n_all_vars++] = bind_vals[i];
  /* Add/override with explicit var_bindings */
  for (int i = 0; i < n_var_bindings && n_all_vars < 32; i++) {
    bool found = false;
    for (int j = 0; j < n_all_vars; j++) {
      if (all_var_vals[j].var == var_bindings[i].var) {
        all_var_vals[j].value = var_bindings[i].value;
        found = true;
        break;
      }
    }
    if (!found) all_var_vals[n_all_vars++] = var_bindings[i];
  }

  /* Structural hash for program + schedule cache */
  uint32_t cache_hash = structural_hash(tensor_sink) ^ (SCHED_CACHE_VERSION * 2654435761u);
  bool dbg_realize = getenv("POLY_DEBUG_REALIZE") != NULL;

  if (dbg_realize)
    fprintf(stderr, "DBG-REALIZE: n_src=%d hash=0x%08x\n", tensor_sink->n_src, cache_hash);

  /* Fast path: try schedule cache (skips poly_schedule_v2 entirely) */
  SchedCacheEntry *se = !getenv("POLY_NO_SCHED_CACHE") ? sched_cache_get(cache_hash) : NULL;
  if (se) {
    if (dbg_realize)
      fprintf(stderr, "DBG-REALIZE: CACHE HIT for hash 0x%08x\n", cache_hash);
    int fast_ret = realize_from_sched_cache(ctx, se, cache_hash, tensor_sink, bindings, n_bindings,
                                              all_var_vals, n_all_vars);
    if (fast_ret == 0) return 0;
    /* Fall through to slow path if fast path failed (e.g. program cache miss) */
  }

  /* Slow path: full schedule + compile */
  if (dbg_realize)
    fprintf(stderr, "DBG-REALIZE: SLOW PATH for hash 0x%08x\n", cache_hash);
  PolyScheduleResult sr = poly_schedule_v2(ctx, tensor_sink);
  if (sr.n_kernels < 1) { poly_schedule_result_free(&sr); return -1; }

  if (getenv("POLY_DEBUG_REALIZE"))
    fprintf(stderr, "REALIZE: %d kernels, %d intermediates, %d stores in SINK\n",
            sr.n_kernels, sr.n_intermediates, tensor_sink->n_src);

  /* 2. Execute kernels using schedule_v2 PARAM mapping
   * (works for both single- and multi-kernel graphs). */
  int n_int = sr.n_intermediates;

  /* Allocate intermediate buffers (dtype-aware: bool=1byte, float=4bytes, etc.) */
  void **intermediates = NULL;
  if (n_int > 0) {
    intermediates = calloc(n_int, sizeof(void *));
    for (int b = 0; b < n_int; b++) {
      int itemsize = (sr.intermediate_itemsizes && sr.intermediate_itemsizes[b] > 0)
                       ? sr.intermediate_itemsizes[b] : (int)sizeof(float);
      intermediates[b] = calloc(sr.intermediate_sizes[b], itemsize);
    }
  }

  /* Build intermediate buffer lookup: buf UOp → index+1 (0 = not found) */
  PolyMap *inter_set = NULL;
  if (n_int > 0 && sr.intermediate_buf_uops) {
    inter_set = poly_map_new(n_int < 4 ? 4 : (size_t)n_int);
    for (int b = 0; b < n_int; b++) {
      PolyUOp *ib = sr.intermediate_buf_uops[b];
      poly_map_set(inter_set, ptr_hash(ib), ib,
                   (PolyUOp *)(intptr_t)(b + 1), ptr_eq);
    }
  }

  int ret = 0;
  /* Stack storage for DEFINE_VAR integer values passed to kernels.
   * Each DEFINE_VAR param needs an int* that persists through the _call dispatch. */
  int var_int_storage[16];
  int n_var_ints = 0;
  for (int step = 0; step < sr.n_kernels && ret == 0; step++) {
    int k = sr.exec_order ? sr.exec_order[step] : step;
    n_var_ints = 0;  /* reset per kernel */
    if (!sr.kernel_n_params || !sr.param_to_buf || !sr.param_to_buf[k]) {
      fprintf(stderr, "polygrad: realize: missing PARAM mapping for kernel %d\n", k);
      ret = -1;
      break;
    }
    int n_params = sr.kernel_n_params[k];
    int n_vars = (sr.kernel_n_vars ? sr.kernel_n_vars[k] : 0);
    int n_total_args = n_params + n_vars;
    if (n_params <= 0 || n_total_args > MAX_REALIZE_BUFS) {
      fprintf(stderr, "polygrad: realize: invalid n_params=%d n_vars=%d for kernel %d\n", n_params, n_vars, k);
      ret = -1;
      break;
    }
    void **args = calloc((size_t)n_total_args, sizeof(void *));

    if (getenv("POLY_DEBUG_REALIZE"))
      fprintf(stderr, "  kernel[%d] n_params=%d n_vars=%d\n", k, n_params, n_vars);

    /* Fill buffer params: args[0..n_params-1] = buffer data pointers */
    for (int i = 0; i < n_params; i++) {
      PolyUOp *buf = sr.param_to_buf[k][i];
      if (!buf) {
        fprintf(stderr, "polygrad: realize: NULL param_to_buf[%d][%d]\n", k, i);
        ret = -1;
        free(args);
        args = NULL;
        break;
      }

      if (buf->op == POLY_OP_BUFFER) {
        for (int j = 0; j < n_bindings; j++) {
          if (bindings[j].buffer == buf) {
            args[i] = bindings[j].data;
            break;
          }
        }
        if (!args[i] && inter_set) {
          PolyUOp *v = poly_map_get(inter_set, ptr_hash(buf), buf, ptr_eq);
          if (v) args[i] = intermediates[(int)((intptr_t)v - 1)];
        }
        if (!args[i]) {
          args[i] = const_registry_lookup(ctx, buf);
        }
        if (getenv("POLY_DEBUG_REALIZE"))
          fprintf(stderr, "    param[%d] = BUFFER %p -> %p\n", i, (void*)buf, args[i]);
      } else {
        assert(buf->op != POLY_OP_BUFFERIZE &&
               "unexpected BUFFERIZE in param_to_buf (old split path removed)");
      }

      if (!args[i]) {
        fprintf(stderr, "polygrad: realize: no binding for param %d in kernel %d (expected buf=%p, op=%s)\n",
                i, k, (void*)buf, poly_op_name(buf->op));
        ret = -1;
        free(args);
        args = NULL;
        break;
      }
    }

    /* Fill var params: args[n_params..n_params+n_vars-1] = DEFINE_VAR int values.
     * Matches tinygrad convention: [buffer_ptrs...] + [var_vals...] */
    if (ret == 0 && args && n_vars > 0 && sr.var_to_buf && sr.var_to_buf[k]) {
      for (int v = 0; v < n_vars; v++) {
        PolyUOp *var = sr.var_to_buf[k][v];
        if (!var || var->op != POLY_OP_DEFINE_VAR) {
          fprintf(stderr, "polygrad: realize: invalid var_to_buf[%d][%d]\n", k, v);
          ret = -1; free(args); args = NULL; break;
        }
        bool found = false;
        for (int vb = 0; vb < n_all_vars; vb++) {
          if (all_var_vals[vb].var == var) {
            var_int_storage[n_var_ints] = (int)all_var_vals[vb].value;
            args[n_params + v] = &var_int_storage[n_var_ints];
            n_var_ints++;
            found = true;
            if (getenv("POLY_DEBUG_REALIZE"))
              fprintf(stderr, "    var[%d] = DEFINE_VAR \"%s\" -> %d\n", v,
                      var->arg.define_var.name ? var->arg.define_var.name : "?",
                      all_var_vals[vb].value);
            break;
          }
        }
        if (!found) {
          fprintf(stderr, "polygrad: realize: no binding for DEFINE_VAR \"%s\" in kernel %d\n",
                  var->arg.define_var.name ? var->arg.define_var.name : "?", k);
          ret = -1; free(args); args = NULL; break;
        }
      }
    }

    if (ret == 0 && args) {
      if (!sr.kernels[k]) {
        fprintf(stderr, "polygrad: realize: kernel %d is NULL\n", k);
        ret = -1;
        free(args);
        break;
      }
      /* If POLY_DEBUG_REALIZE and this is a multi-store SINK, dump kernel source */
      if (getenv("POLY_DEBUG_REALIZE") && tensor_sink->n_src > 2) {
        int dlin;
        PolyUOp **dlops = poly_linearize(ctx, sr.kernels[k], &dlin);
        if (dlops) {
          char dfn[32];
          snprintf(dfn, sizeof(dfn), "dbg_k%d", k);
          char *dsrc = poly_render_c(dlops, dlin, dfn);
          if (dsrc) {
            fprintf(stderr, "  === KERNEL[%d] (%d stores SINK) ===\n%s\n  === END ===\n", k, tensor_sink->n_src, dsrc);
            free(dsrc);
          }
          free(dlops);
        }
      }

      ret = compile_and_run(ctx, sr.kernels[k], args, n_params, cache_hash + k);

      /* Debug: dump intermediate buffer values after producer kernels */
      if (getenv("POLY_DEBUG_REALIZE") && k < n_int && intermediates) {
        float *ibuf = (float*)intermediates[k];
        int sz = sr.intermediate_sizes[k];
        float imin = ibuf[0], imax = ibuf[0], isum = 0;
        for (int ii = 0; ii < sz; ii++) {
          if (ibuf[ii] < imin) imin = ibuf[ii];
          if (ibuf[ii] > imax) imax = ibuf[ii];
          isum += ibuf[ii];
        }
        fprintf(stderr, "  INTERMEDIATE[%d] size=%d min=%.6f max=%.6f mean=%.6f\n",
                k, sz, imin, imax, isum / sz);
      }
      free(args);
    }
  }

  if (inter_set) poly_map_destroy(inter_set);

  /* Cache the schedule result for future fast-path */
  if (ret == 0 && cache_hash && !se) {
    sched_cache_put(cache_hash, &sr, tensor_sink);
  }

  if (intermediates) {
    for (int b = 0; b < n_int; b++) free(intermediates[b]);
    free(intermediates);
  }
  poly_schedule_result_free(&sr);
  return ret;
}

int poly_realize(PolyCtx *ctx, PolyUOp *tensor_sink,
                PolyBufferBinding *bindings, int n_bindings) {
  return realize_impl(ctx, tensor_sink, bindings, n_bindings, NULL, 0);
}

int poly_realize_ex(PolyCtx *ctx, PolyUOp *tensor_sink,
                    PolyBufferBinding *bindings, int n_bindings,
                    PolyVarBinding *var_bindings, int n_var_bindings) {
  return realize_impl(ctx, tensor_sink, bindings, n_bindings,
                      var_bindings, n_var_bindings);
}

/* ── Compiled Step (persistent sched-cache entry) ──────────────────────
 *
 * PolyStep is a first-class compiled execution step. It pre-compiles all
 * kernels, pre-allocates intermediates, and stores the binding layout so
 * poly_step_run() has zero scheduling overhead. Mirrors the existing
 * sched_cache_put + realize_from_sched_cache fast path, but made durable.
 */

typedef struct {
  PolyProgram *prog;       /* compiled kernel (owned by step) */
  int n_params;            /* buffer params */
  int n_vars;              /* DEFINE_VAR params */
  ParamMapping *mappings;  /* malloc'd [n_params] */
  PolyUOp **var_ptrs;      /* malloc'd [n_vars] */
} PolyStepKernel;

typedef struct {
  PolyStepBufRole role;
  PolyDType dtype;
  int64_t numel;
  int64_t nbytes;
} PolyStepBufMeta;

struct PolyStep {
  PolyCtx *ctx;                  /* owning context (for constant auto-bind lookup) */
  int n_kernels;
  PolyStepKernel *kernels;      /* malloc'd [n_kernels] */
  int *exec_order;              /* malloc'd [n_kernels] */
  int n_bufs;                   /* total BUFFER nodes in DFS order */
  PolyUOp **buf_order;          /* malloc'd [n_bufs] -- DFS buffer ordering */
  int n_output_bufs;            /* prefix of buf_order that are outputs */
  PolyStepBufMeta *buf_meta;    /* malloc'd [n_bufs + n_intermediates] */
  int n_total_buf_meta;         /* external + intermediates */

  int n_intermediates;
  void **intermediates;         /* malloc'd array of malloc'd buffers */
  size_t *intermediate_bytes;   /* precomputed: (size_t)size * itemsize */

  int n_default_vars;
  PolyVarBinding *default_vars; /* malloc'd [n_default_vars] -- BIND defaults */

  uint32_t graph_hash;
};

static int collect_output_buffers_in_sink(PolyUOp *tensor_sink, PolyUOp **out, int cap) {
  if (!tensor_sink || tensor_sink->op != POLY_OP_SINK) return 0;
  int n_seen = 0;
  for (int i = 0; i < tensor_sink->n_src; i++) {
    PolyUOp *store = tensor_sink->src[i];
    if (!store || store->op != POLY_OP_STORE || store->n_src < 1) continue;
    PolyUOp *buf = store->src[0];
    if (!buf || buf->op != POLY_OP_BUFFER) continue;
    bool dup = false;
    for (int j = 0; j < n_seen; j++) {
      if (out[j] == buf) { dup = true; break; }
    }
    if (!dup && n_seen < cap) out[n_seen++] = buf;
  }
  return n_seen;
}

PolyStep *poly_compile_step(PolyCtx *ctx, PolyUOp *tensor_sink) {
  if (!tensor_sink || tensor_sink->op != POLY_OP_SINK) {
    fprintf(stderr, "polygrad: compile_step: expected SINK\n");
    return NULL;
  }

  /* Collect DFS buffer ordering BEFORE strip_bind_values.
   * Callers know the original BUFFER UOp pointers (pre-strip).
   * strip_bind_values may rewrite BUFFERs that have BIND sources,
   * producing new UOp pointers. We store the pre-strip ordering in
   * the step so runtime find_buf_position matches caller pointers. */
  PolyUOp **buf_order_orig = calloc(MAX_REALIZE_BUFS, sizeof(PolyUOp *));
  PolyUOp **dfs_visited = calloc(MAX_STRUCT_NODES, sizeof(PolyUOp *));
  int n_bufs_orig = 0, n_dfs = 0;
  collect_buf_order(tensor_sink, buf_order_orig, &n_bufs_orig, dfs_visited, &n_dfs);
  PolyUOp *output_bufs[MAX_REALIZE_BUFS];
  int n_output_bufs = collect_output_buffers_in_sink(tensor_sink, output_bufs, MAX_REALIZE_BUFS);

  /* Strip BIND values, extract compile-time defaults (same as realize_impl) */
  PolyVarBinding bind_vals[16];
  int n_bind_vals = 0;
  tensor_sink = strip_bind_values(ctx, tensor_sink, bind_vals, &n_bind_vals, 16,
                                   NULL, 0);

  /* Collect DFS buffer ordering AFTER strip for mapping build.
   * The scheduler operates on the post-strip graph, so param_to_buf
   * references post-strip BUFFER pointers. */
  PolyUOp **buf_order_post = calloc(MAX_REALIZE_BUFS, sizeof(PolyUOp *));
  int n_bufs_post = 0;
  n_dfs = 0;
  collect_buf_order(tensor_sink, buf_order_post, &n_bufs_post, dfs_visited, &n_dfs);
  free(dfs_visited);

  uint32_t graph_hash = structural_hash(tensor_sink) ^ (SCHED_CACHE_VERSION * 2654435761u);

  /* Schedule */
  PolyScheduleResult sr = poly_schedule_v2(ctx, tensor_sink);
  if (sr.n_kernels < 1) {
    free(buf_order_orig); free(buf_order_post);
    poly_schedule_result_free(&sr);
    return NULL;
  }

  /* Allocate step */
  PolyStep *step = calloc(1, sizeof(PolyStep));
  if (!step) { poly_schedule_result_free(&sr); return NULL; }
  step->n_kernels = sr.n_kernels;
  step->ctx = ctx;
  step->kernels = calloc((size_t)sr.n_kernels, sizeof(PolyStepKernel));
  step->graph_hash = graph_hash;

  /* Copy BIND defaults */
  step->n_default_vars = n_bind_vals;
  if (n_bind_vals > 0) {
    step->default_vars = malloc((size_t)n_bind_vals * sizeof(PolyVarBinding));
    memcpy(step->default_vars, bind_vals, (size_t)n_bind_vals * sizeof(PolyVarBinding));
  }

  /* Store pre-strip buffer ordering (for runtime find_buf_position) */
  step->n_bufs = n_bufs_orig;
  step->n_output_bufs = n_output_bufs;
  if (step->n_output_bufs > step->n_bufs) step->n_output_bufs = step->n_bufs;
  if (n_bufs_orig > 0) {
    step->buf_order = malloc((size_t)n_bufs_orig * sizeof(PolyUOp *));
    memcpy(step->buf_order, buf_order_orig, (size_t)n_bufs_orig * sizeof(PolyUOp *));
  }

  /* Pre-allocate intermediates with overflow guard */
  step->n_intermediates = sr.n_intermediates;
  if (sr.n_intermediates > 0) {
    step->intermediate_bytes = malloc((size_t)sr.n_intermediates * sizeof(size_t));
    step->intermediates = calloc((size_t)sr.n_intermediates, sizeof(void *));
    for (int b = 0; b < sr.n_intermediates; b++) {
      int itemsize = (sr.intermediate_itemsizes && sr.intermediate_itemsizes[b] > 0)
                       ? sr.intermediate_itemsizes[b] : (int)sizeof(float);
      int64_t sz = sr.intermediate_sizes[b];
      if (sz > 0 && (size_t)sz > SIZE_MAX / (size_t)itemsize) {
        fprintf(stderr, "polygrad: compile_step: intermediate %d overflow (%lld * %d)\n",
                b, (long long)sz, itemsize);
        goto cleanup;
      }
      step->intermediate_bytes[b] = (size_t)sz * (size_t)itemsize;
      step->intermediates[b] = calloc((size_t)sz, (size_t)itemsize);
      if (!step->intermediates[b]) goto cleanup;
    }
  }

  /* Build stable buffer metadata table:
   *   [0..n_bufs)                    -> external buffers (input/output)
   *   [n_bufs..n_bufs+n_intermediates) -> temporaries */
  step->n_total_buf_meta = step->n_bufs + step->n_intermediates;
  if (step->n_total_buf_meta > 0) {
    step->buf_meta = calloc((size_t)step->n_total_buf_meta, sizeof(PolyStepBufMeta));
    if (!step->buf_meta) goto cleanup;
  }
  for (int i = 0; i < step->n_bufs; i++) {
    PolyUOp *buf = step->buf_order ? step->buf_order[i] : NULL;
    PolyStepBufMeta *m = &step->buf_meta[i];
    bool is_output = false;
    for (int j = 0; j < step->n_output_bufs; j++) {
      if (output_bufs[j] == buf) { is_output = true; break; }
    }
    if (is_output) m->role = POLY_STEP_BUF_OUTPUT;
    else if (const_registry_has(ctx, buf)) m->role = POLY_STEP_BUF_CONSTANT;
    else m->role = POLY_STEP_BUF_INPUT;
    m->dtype = buf ? poly_dtype_scalar(buf->dtype) : POLY_FLOAT32;
    m->numel = (buf && buf->arg.kind == POLY_ARG_INT) ? buf->arg.i : 0;
    if (m->numel > 0) m->nbytes = m->numel * poly_dtype_itemsize(m->dtype);
  }
  for (int b = 0; b < step->n_intermediates; b++) {
    int idx = step->n_bufs + b;
    PolyStepBufMeta *m = &step->buf_meta[idx];
    m->role = POLY_STEP_BUF_TEMP;
    if (sr.intermediate_buf_uops && sr.intermediate_buf_uops[b]) {
      m->dtype = poly_dtype_scalar(sr.intermediate_buf_uops[b]->dtype);
    } else {
      m->dtype = POLY_FLOAT32;
    }
    m->numel = sr.intermediate_sizes ? sr.intermediate_sizes[b] : 0;
    m->nbytes = (int64_t)(step->intermediate_bytes ? step->intermediate_bytes[b] : 0);
  }

  /* Build intermediate lookup (same as realize_impl's inter_set) */
  PolyMap *inter_set = NULL;
  if (sr.n_intermediates > 0 && sr.intermediate_buf_uops) {
    inter_set = poly_map_new((size_t)(sr.n_intermediates < 4 ? 4 : sr.n_intermediates));
    for (int b = 0; b < sr.n_intermediates; b++) {
      PolyUOp *ib = sr.intermediate_buf_uops[b];
      poly_map_set(inter_set, ptr_hash(ib), ib,
                   (PolyUOp *)(intptr_t)(b + 1), ptr_eq);
    }
  }

  /* For each kernel: validate, compile, build mapping */
  static int step_counter = 0;
  for (int k = 0; k < sr.n_kernels; k++) {
    PolyStepKernel *sk = &step->kernels[k];

    if (!validate_kernel_graph(ctx, sr.kernels[k])) {
      fprintf(stderr, "polygrad: compile_step: kernel %d validation failed\n", k);
      if (inter_set) poly_map_destroy(inter_set);
      goto cleanup;
    }

    int n_lin;
    PolyUOp **lin = poly_linearize(ctx, sr.kernels[k], &n_lin);
    if (!lin) { if (inter_set) poly_map_destroy(inter_set); goto cleanup; }

    char fn_name[64];
    snprintf(fn_name, sizeof(fn_name), "step%d_k%d", step_counter, k);
    char *src = poly_render_c(lin, n_lin, fn_name);
    free(lin);
    if (!src) { if (inter_set) poly_map_destroy(inter_set); goto cleanup; }

    if (getenv("POLY_DUMP_KERNELS"))
      fprintf(stderr, "=== STEP KERNEL %s ===\n%s\n=== END ===\n", fn_name, src);

    sk->prog = poly_compile_c(src, fn_name);
    if (!sk->prog) {
      fprintf(stderr, "=== FAILED STEP KERNEL %d ===\n%s\n=== END ===\n", k, src);
      free(src);
      if (inter_set) poly_map_destroy(inter_set);
      goto cleanup;
    }
    free(src);

    /* Build ParamMapping (same logic as sched_cache_put) */
    int np = sr.kernel_n_params[k];
    sk->n_params = np;
    sk->mappings = calloc((size_t)np, sizeof(ParamMapping));
    for (int i = 0; i < np; i++) {
      PolyUOp *buf = sr.param_to_buf[k][i];
      sk->mappings[i].buf_position = -1;
      sk->mappings[i].intermediate_idx = -1;

      if (buf->op == POLY_OP_BUFFER) {
        int pos = find_buf_position(buf, buf_order_post, n_bufs_post);
        if (pos >= 0) {
          sk->mappings[i].buf_position = pos;
        } else if (inter_set) {
          PolyUOp *v = poly_map_get(inter_set, ptr_hash(buf), buf, ptr_eq);
          if (v) sk->mappings[i].intermediate_idx = (int)((intptr_t)v - 1);
        }
      }
      if (sk->mappings[i].buf_position < 0 &&
          sk->mappings[i].intermediate_idx < 0) {
        fprintf(stderr, "polygrad: compile_step: unresolved param %d in kernel %d "
                "(buf=%p, op=%s)\n",
                i, k, (void*)buf, poly_op_name(buf->op));
        if (inter_set) poly_map_destroy(inter_set);
        goto cleanup;
      }
    }

    /* Build var mapping (malloc'd, no caps) */
    int nv = (sr.kernel_n_vars ? sr.kernel_n_vars[k] : 0);
    sk->n_vars = nv;
    if (nv > 0) {
      sk->var_ptrs = malloc((size_t)nv * sizeof(PolyUOp *));
      for (int v = 0; v < nv; v++) {
        sk->var_ptrs[v] = (sr.var_to_buf && sr.var_to_buf[k])
                            ? sr.var_to_buf[k][v] : NULL;
        if (!sk->var_ptrs[v]) {
          fprintf(stderr, "polygrad: compile_step: NULL var %d in kernel %d\n", v, k);
          if (inter_set) poly_map_destroy(inter_set);
          goto cleanup;
        }
      }
    }
  }
  step_counter++;

  if (inter_set) poly_map_destroy(inter_set);

  /* Deep-copy exec_order */
  step->exec_order = malloc((size_t)sr.n_kernels * sizeof(int));
  if (sr.exec_order)
    memcpy(step->exec_order, sr.exec_order, (size_t)sr.n_kernels * sizeof(int));
  else
    for (int k = 0; k < sr.n_kernels; k++) step->exec_order[k] = k;

  if (getenv("POLY_DEBUG_STEP")) {
    fprintf(stderr, "[compile_step] exec_order:");
    for (int k = 0; k < sr.n_kernels; k++)
      fprintf(stderr, " %d", step->exec_order[k]);
    fprintf(stderr, "\n");
  }

  free(buf_order_orig);
  free(buf_order_post);
  poly_schedule_result_free(&sr);
  return step;

cleanup:
  free(buf_order_orig);
  free(buf_order_post);
  if (step->kernels) {
    for (int k = 0; k < sr.n_kernels; k++) {
      PolyStepKernel *sk = &step->kernels[k];
      if (sk->prog) poly_program_destroy(sk->prog);
      free(sk->mappings);
      free(sk->var_ptrs);
    }
    free(step->kernels);
  }
  if (step->intermediates) {
    for (int b = 0; b < step->n_intermediates; b++)
      free(step->intermediates[b]);
    free(step->intermediates);
  }
  free(step->intermediate_bytes);
  free(step->default_vars);
  free(step->exec_order);
  free(step->buf_order);
  free(step->buf_meta);
  free(step);
  poly_schedule_result_free(&sr);
  return NULL;
}

static int64_t static_numel_of_uop(PolyCtx *ctx, PolyUOp *u) {
  PolyShape s = poly_uop_shape(ctx, u);
  if (s.ndim < 0) {
    if (s.dims) free(s.dims);
    return -1;
  }
  if (s.ndim == 0) return 1;
  int64_t n = 1;
  for (int i = 0; i < s.ndim; i++) {
    if (s.dims[i] <= 0) { free(s.dims); return -1; }
    if (n > INT64_MAX / s.dims[i]) { free(s.dims); return -1; }
    n *= s.dims[i];
  }
  free(s.dims);
  return n;
}

PolyStep *poly_compile_value_and_grad(PolyCtx *ctx, PolyUOp *loss,
                                      PolyUOp **params, int n_params,
                                      int *out_loss_buf_idx,
                                      int *out_grad_buf_idxs) {
  if (!ctx || !loss || !params || n_params <= 0 || !out_loss_buf_idx || !out_grad_buf_idxs) {
    fprintf(stderr, "polygrad: compile_value_and_grad: invalid arguments\n");
    return NULL;
  }

  /* Build gradients in one reverse pass. */
  PolyUOp **grads = calloc((size_t)n_params, sizeof(PolyUOp *));
  if (!grads) return NULL;
  if (poly_grad_many(ctx, loss, NULL, params, n_params, grads) != 0) {
    fprintf(stderr, "polygrad: compile_value_and_grad: poly_grad_many failed\n");
    free(grads);
    return NULL;
  }

  /* Build output stores:
   *   store[0]   -> scalar loss (as 1-element vector)
   *   store[i+1] -> flattened grad[i] */
  int n_stores = n_params + 1;
  PolyUOp **stores = calloc((size_t)n_stores, sizeof(PolyUOp *));
  PolyUOp **grad_out_bufs = calloc((size_t)n_params, sizeof(PolyUOp *));
  if (!stores || !grad_out_bufs) {
    free(stores);
    free(grad_out_bufs);
    free(grads);
    return NULL;
  }

  PolyDType out_dt = poly_dtype_scalar(loss->dtype);
  if (!poly_dtype_is_float(out_dt)) out_dt = POLY_FLOAT32;

  /* Loss output buffer (1 element). */
  PolyUOp *loss_buf = poly_buffer(ctx, out_dt, 1);
  PolyUOp *loss_flat = loss;
  if (static_numel_of_uop(ctx, loss) != 1) {
    int64_t one_shape[1] = {1};
    loss_flat = poly_reshape(ctx, loss, one_shape, 1);
  }
  stores[0] = poly_store_val(ctx, loss_buf, loss_flat);

  for (int i = 0; i < n_params; i++) {
    int64_t numel = static_numel_of_uop(ctx, grads[i]);
    if (numel <= 0) {
      fprintf(stderr, "polygrad: compile_value_and_grad: grad[%d] has unknown/invalid shape\n", i);
      free(stores);
      free(grad_out_bufs);
      free(grads);
      return NULL;
    }
    PolyDType gdt = poly_dtype_scalar(grads[i]->dtype);
    if (!poly_dtype_is_float(gdt)) gdt = POLY_FLOAT32;
    PolyUOp *gbuf = poly_buffer(ctx, gdt, numel);
    grad_out_bufs[i] = gbuf;
    PolyUOp *gflat = grads[i];
    PolyShape gs = poly_uop_shape(ctx, grads[i]);
    if (gs.ndim != 1 || (gs.ndim == 1 && gs.dims[0] != numel)) {
      int64_t flat_shape[1] = { numel };
      gflat = poly_reshape(ctx, grads[i], flat_shape, 1);
    }
    if (gs.dims) free(gs.dims);
    stores[i + 1] = poly_store_val(ctx, gbuf, gflat);
  }

  PolyUOp *sink = poly_sink_n(ctx, stores, n_stores);
  PolyStep *step = poly_compile_step(ctx, sink);

  free(stores);
  free(grads);

  if (!step) {
    free(grad_out_bufs);
    return NULL;
  }

  int loss_idx = find_buf_position(loss_buf, step->buf_order, step->n_bufs);
  if (loss_idx < 0) {
    fprintf(stderr, "polygrad: compile_value_and_grad: loss buffer index not found\n");
    free(grad_out_bufs);
    poly_step_destroy(step);
    return NULL;
  }
  *out_loss_buf_idx = loss_idx;
  for (int i = 0; i < n_params; i++) {
    int gi = find_buf_position(grad_out_bufs[i], step->buf_order, step->n_bufs);
    if (gi < 0) {
      fprintf(stderr, "polygrad: compile_value_and_grad: grad buffer %d index not found\n", i);
      free(grad_out_bufs);
      poly_step_destroy(step);
      return NULL;
    }
    out_grad_buf_idxs[i] = gi;
  }
  free(grad_out_bufs);
  return step;
}

int poly_step_run_ex(PolyStep *step,
                     PolyBufferBinding *bindings, int n_bindings,
                     PolyVarBinding *var_bindings, int n_var_bindings) {
  if (!step) return -1;

  /* Zero all intermediate buffers (needed for REDUCE accumulators) */
  for (int b = 0; b < step->n_intermediates; b++)
    memset(step->intermediates[b], 0, step->intermediate_bytes[b]);

  /* Merge default vars with runtime overrides (overrides win).
   * Same pattern as realize_impl's BIND merge. */
  PolyVarBinding all_var_vals[32];
  int n_all_vars = 0;
  for (int i = 0; i < step->n_default_vars && n_all_vars < 32; i++)
    all_var_vals[n_all_vars++] = step->default_vars[i];
  for (int i = 0; i < n_var_bindings && n_all_vars < 32; i++) {
    bool found = false;
    for (int j = 0; j < n_all_vars; j++) {
      if (all_var_vals[j].var == var_bindings[i].var) {
        all_var_vals[j].value = var_bindings[i].value;
        found = true;
        break;
      }
    }
    if (!found) all_var_vals[n_all_vars++] = var_bindings[i];
  }

  /* Build pos_to_data[] from bindings (same as realize_from_sched_cache) */
  void *pos_to_data[MAX_REALIZE_BUFS];
  memset(pos_to_data, 0, sizeof(pos_to_data));
  if (getenv("POLY_DEBUG_STEP")) {
    fprintf(stderr, "[step_run] n_bufs=%d n_bindings=%d\n", step->n_bufs, n_bindings);
    for (int p = 0; p < step->n_bufs; p++)
      fprintf(stderr, "  buf_order[%d] = %p (op=%s)\n", p, (void*)step->buf_order[p],
              poly_op_name(step->buf_order[p]->op));
    for (int j = 0; j < n_bindings; j++)
      fprintf(stderr, "  binding[%d] = buf=%p data=%p\n", j,
              (void*)bindings[j].buffer, bindings[j].data);
  }
  for (int j = 0; j < n_bindings; j++) {
    int pos = find_buf_position(bindings[j].buffer, step->buf_order, step->n_bufs);
    if (pos >= 0) pos_to_data[pos] = bindings[j].data;
    else if (getenv("POLY_DEBUG_STEP"))
      fprintf(stderr, "  binding[%d] NOT FOUND in buf_order!\n", j);
  }

  /* Execute kernels in exec_order */
  int ret = 0;
  int var_int_storage[16]; /* persists through _call dispatch */
  for (int s = 0; s < step->n_kernels && ret == 0; s++) {
    int k = step->exec_order[s];
    PolyStepKernel *sk = &step->kernels[k];
    int n_total = sk->n_params + sk->n_vars;
    int n_var_ints = 0;

    void **args = calloc((size_t)n_total, sizeof(void *));

    /* Fill buffer params via ParamMapping (O(1) per param) */
    for (int i = 0; i < sk->n_params; i++) {
      ParamMapping *m = &sk->mappings[i];
      if (m->buf_position >= 0 && m->buf_position < step->n_bufs) {
        args[i] = pos_to_data[m->buf_position];
      } else if (step->intermediates && m->intermediate_idx >= 0 &&
                 m->intermediate_idx < step->n_intermediates) {
        args[i] = step->intermediates[m->intermediate_idx];
      }
      if (!args[i] && m->buf_position >= 0 && m->buf_position < step->n_bufs) {
        PolyUOp *buf = step->buf_order[m->buf_position];
        args[i] = const_registry_lookup(step->ctx, buf);
      }
      if (!args[i]) {
        fprintf(stderr, "polygrad: step_run: missing binding for param %d kernel %d "
                "(pos=%d, inter=%d)\n",
                i, k, m->buf_position, m->intermediate_idx);
        ret = -1; free(args); args = NULL; break;
      }
    }

    /* Fill var params (same int* semantics as realize_from_sched_cache) */
    if (ret == 0 && args && sk->n_vars > 0) {
      for (int v = 0; v < sk->n_vars; v++) {
        PolyUOp *var = sk->var_ptrs[v];
        bool found = false;
        for (int vb = 0; vb < n_all_vars; vb++) {
          if (all_var_vals[vb].var == var) {
            var_int_storage[n_var_ints] = (int)all_var_vals[vb].value;
            args[sk->n_params + v] = &var_int_storage[n_var_ints];
            n_var_ints++;
            found = true;
            break;
          }
        }
        if (!found) {
          fprintf(stderr, "polygrad: step_run: no binding for DEFINE_VAR in kernel %d\n", k);
          ret = -1; free(args); args = NULL; break;
        }
      }
    }

    if (ret == 0 && args) {
      poly_program_call(sk->prog, args, n_total);
      free(args);
    }
  }

  return ret;
}

int poly_step_run(PolyStep *step,
                  PolyBufferBinding *bindings, int n_bindings) {
  return poly_step_run_ex(step, bindings, n_bindings, NULL, 0);
}

int poly_step_run_indexed_ex(PolyStep *step, void **buffer_data, int n_buffers,
                             PolyVarBinding *var_bindings, int n_var_bindings) {
  if (!step || !buffer_data || n_buffers <= 0) return -1;
  int n = (n_buffers < step->n_bufs) ? n_buffers : step->n_bufs;
  PolyBufferBinding *bindings = calloc((size_t)n, sizeof(PolyBufferBinding));
  if (!bindings) return -1;
  int nb = 0;
  for (int i = 0; i < n; i++) {
    if (!buffer_data[i]) continue;
    bindings[nb].buffer = step->buf_order[i];
    bindings[nb].data = buffer_data[i];
    nb++;
  }
  int ret = poly_step_run_ex(step, bindings, nb, var_bindings, n_var_bindings);
  free(bindings);
  return ret;
}

int poly_step_run_indexed(PolyStep *step, void **buffer_data, int n_buffers) {
  return poly_step_run_indexed_ex(step, buffer_data, n_buffers, NULL, 0);
}

void poly_step_destroy(PolyStep *step) {
  if (!step) return;
  for (int k = 0; k < step->n_kernels; k++) {
    PolyStepKernel *sk = &step->kernels[k];
    if (sk->prog) poly_program_destroy(sk->prog);
    free(sk->mappings);
    free(sk->var_ptrs);
  }
  free(step->kernels);
  free(step->exec_order);
  free(step->buf_order);
  if (step->intermediates) {
    for (int b = 0; b < step->n_intermediates; b++)
      free(step->intermediates[b]);
    free(step->intermediates);
  }
  free(step->intermediate_bytes);
  free(step->default_vars);
  free(step->buf_meta);
  free(step);
}

int poly_step_n_kernels(const PolyStep *step) { return step ? step->n_kernels : 0; }
int poly_step_n_intermediates(const PolyStep *step) { return step ? step->n_intermediates : 0; }
int poly_step_n_buffers(const PolyStep *step) { return step ? step->n_total_buf_meta : 0; }
int poly_step_n_bindable_buffers(const PolyStep *step) { return step ? step->n_bufs : 0; }

int poly_step_buffer_info(const PolyStep *step, int idx, PolyStepBufferInfo *out) {
  if (!step || !out || idx < 0 || idx >= step->n_total_buf_meta) return -1;
  PolyStepBufMeta *m = &step->buf_meta[idx];
  out->version = POLY_STEP_BUFFER_INFO_VERSION;
  out->index = idx;
  out->role = m->role;
  out->dtype = m->dtype;
  out->numel = m->numel;
  out->nbytes = m->nbytes;
  return 0;
}

PolyUOp *poly_step_buf_uop(const PolyStep *step, int idx) {
  if (!step || idx < 0 || idx >= step->n_bufs) return NULL;
  return step->buf_order[idx];
}

/* ── Prepared step construction ───────────────────────────────────────── */

PolyPreparedStep *poly_prepare_step(PolyCtx *ctx, PolyUOp *sink,
                                    PolyCompileMode mode) {
  if (!sink || sink->op != POLY_OP_SINK) {
    fprintf(stderr, "polygrad: prepare_step: expected SINK\n");
    return NULL;
  }

  /* --- Collect pre-strip buffer ordering -------------------------------- */
  PolyUOp **buf_order_orig = calloc(MAX_REALIZE_BUFS, sizeof(PolyUOp *));
  PolyUOp **dfs_visited = calloc(MAX_STRUCT_NODES, sizeof(PolyUOp *));
  int n_bufs_orig = 0, n_dfs = 0;
  collect_buf_order(sink, buf_order_orig, &n_bufs_orig, dfs_visited, &n_dfs);

  /* Identify output buffers */
  PolyUOp *output_bufs[MAX_REALIZE_BUFS];
  int n_output_bufs = collect_output_buffers_in_sink(sink, output_bufs, MAX_REALIZE_BUFS);

  /* --- Extract BIND defaults and strip ---------------------------------- */
  PolyVarBinding bind_vals[16];
  int n_bind_vals = 0;
  sink = strip_bind_values(ctx, sink, bind_vals, &n_bind_vals, 16, NULL, 0);

  /* Post-strip buffer ordering (scheduler sees these pointers) */
  PolyUOp **buf_order_post = calloc(MAX_REALIZE_BUFS, sizeof(PolyUOp *));
  int n_bufs_post = 0;
  n_dfs = 0;
  collect_buf_order(sink, buf_order_post, &n_bufs_post, dfs_visited, &n_dfs);
  free(dfs_visited);

  uint32_t ghash = structural_hash(sink) ^ (SCHED_CACHE_VERSION * 2654435761u);

  /* --- Schedule --------------------------------------------------------- */
  PolyScheduleResult sr = poly_schedule_v2(ctx, sink);
  if (sr.n_kernels < 1) {
    free(buf_order_orig); free(buf_order_post);
    poly_schedule_result_free(&sr);
    return NULL;
  }

  /* --- Allocate prepared step ------------------------------------------- */
  PolyPreparedStep *ps = calloc(1, sizeof(PolyPreparedStep));
  if (!ps) {
    free(buf_order_orig); free(buf_order_post);
    poly_schedule_result_free(&sr);
    return NULL;
  }
  ps->mode = mode;
  ps->graph_hash = ghash;
  ps->loss_buf_slot = -1;

  /* --- Default variable bindings ---------------------------------------- */
  ps->n_default_vars = n_bind_vals;
  if (n_bind_vals > 0) {
    ps->default_vars = malloc((size_t)n_bind_vals * sizeof(PolyVarBinding));
    memcpy(ps->default_vars, bind_vals, (size_t)n_bind_vals * sizeof(PolyVarBinding));
  }

  /* --- Build buffer slot table ------------------------------------------ */
  int n_external = n_bufs_orig;
  int n_intermediate = sr.n_intermediates;
  ps->n_buf_slots = n_external + n_intermediate;

  if (ps->n_buf_slots > 0) {
    ps->buf_slots = calloc((size_t)ps->n_buf_slots, sizeof(PolyPreparedBufSlot));

    /* External buffer slots */
    for (int i = 0; i < n_external; i++) {
      PolyPreparedBufSlot *slot = &ps->buf_slots[i];
      PolyUOp *buf = buf_order_orig[i];
      slot->buf_uop = buf;
      slot->is_intermediate = false;
      slot->external_buf_idx = i;
      slot->dtype = buf ? poly_dtype_scalar(buf->dtype) : POLY_FLOAT32;
      slot->numel = (buf && buf->arg.kind == POLY_ARG_INT) ? buf->arg.i : 0;
      if (slot->numel > 0)
        slot->nbytes = slot->numel * poly_dtype_itemsize(slot->dtype);
    }

    /* Intermediate buffer slots */
    for (int b = 0; b < n_intermediate; b++) {
      PolyPreparedBufSlot *slot = &ps->buf_slots[n_external + b];
      slot->is_intermediate = true;
      slot->external_buf_idx = -1;
      if (sr.intermediate_buf_uops && sr.intermediate_buf_uops[b]) {
        slot->buf_uop = sr.intermediate_buf_uops[b];
        slot->dtype = poly_dtype_scalar(sr.intermediate_buf_uops[b]->dtype);
      } else {
        slot->dtype = POLY_FLOAT32;
      }
      slot->numel = sr.intermediate_sizes ? sr.intermediate_sizes[b] : 0;
      int itemsize = (sr.intermediate_itemsizes && sr.intermediate_itemsizes[b] > 0)
                       ? sr.intermediate_itemsizes[b] : (int)sizeof(float);
      slot->nbytes = slot->numel * itemsize;
    }
  }

  /* --- Build intermediate UOp -> slot index lookup ---------------------- */
  PolyMap *inter_set = NULL;
  if (n_intermediate > 0 && sr.intermediate_buf_uops) {
    inter_set = poly_map_new((size_t)(n_intermediate < 4 ? 4 : n_intermediate));
    for (int b = 0; b < n_intermediate; b++) {
      PolyUOp *ib = sr.intermediate_buf_uops[b];
      /* Store 1-based index to distinguish from NULL */
      poly_map_set(inter_set, ptr_hash(ib), ib,
                   (PolyUOp *)(intptr_t)(n_external + b + 1), ptr_eq);
    }
  }

  /* --- Build exec items ------------------------------------------------- */
  ps->n_items = sr.n_kernels;
  ps->items = calloc((size_t)sr.n_kernels, sizeof(PolyExecItemSpec));

  for (int k = 0; k < sr.n_kernels; k++) {
    PolyExecItemSpec *item = &ps->items[k];
    item->kind = POLY_EXEC_COMPUTE;
    item->root = sr.kernels[k];

    /* Map each PARAM to a buffer slot index */
    int np = sr.kernel_n_params[k];
    item->n_buf_slots = np;
    item->buf_slot_indices = malloc((size_t)np * sizeof(int));

    for (int i = 0; i < np; i++) {
      PolyUOp *buf = sr.param_to_buf[k][i];
      item->buf_slot_indices[i] = -1;

      if (buf->op == POLY_OP_BUFFER) {
        /* Try external buffer lookup (post-strip ordering) */
        int pos = find_buf_position(buf, buf_order_post, n_bufs_post);
        if (pos >= 0) {
          item->buf_slot_indices[i] = pos;
        } else if (inter_set) {
          PolyUOp *v = poly_map_get(inter_set, ptr_hash(buf), buf, ptr_eq);
          if (v) item->buf_slot_indices[i] = (int)((intptr_t)v - 1);
        }
      }

      if (item->buf_slot_indices[i] < 0) {
        fprintf(stderr, "polygrad: prepare_step: unresolved param %d in kernel %d\n",
                i, k);
        if (inter_set) poly_map_destroy(inter_set);
        goto cleanup;
      }
    }
  }

  if (inter_set) poly_map_destroy(inter_set);

  /* --- Execution order -------------------------------------------------- */
  ps->exec_order = malloc((size_t)sr.n_kernels * sizeof(int));
  if (sr.exec_order)
    memcpy(ps->exec_order, sr.exec_order, (size_t)sr.n_kernels * sizeof(int));
  else
    for (int k = 0; k < sr.n_kernels; k++) ps->exec_order[k] = k;

  free(buf_order_orig);
  free(buf_order_post);
  poly_schedule_result_free(&sr);
  return ps;

cleanup:
  free(buf_order_orig);
  free(buf_order_post);
  poly_prepared_step_free(ps);
  poly_schedule_result_free(&sr);
  return NULL;
}

void poly_prepared_step_free(PolyPreparedStep *step) {
  if (!step) return;
  for (int i = 0; i < step->n_items; i++)
    free(step->items[i].buf_slot_indices);
  free(step->items);
  free(step->buf_slots);
  free(step->exec_order);
  free(step->default_vars);
  free(step->grad_buf_slots);
  free(step);
}

int poly_realize_flat(PolyCtx *ctx, PolyUOp *tensor_sink,
                     PolyUOp **buffers, void **datas, int n) {
  PolyBufferBinding bindings[MAX_REALIZE_BUFS];
  if (n > MAX_REALIZE_BUFS) n = MAX_REALIZE_BUFS;
  for (int i = 0; i < n; i++) {
    bindings[i].buffer = buffers[i];
    bindings[i].data = datas[i];
  }
  return poly_realize(ctx, tensor_sink, bindings, n);
}

/* ── Stateful realize builder ────────────────────────────────────────── */

static PolyBufferBinding g_realize_bindings[MAX_REALIZE_BUFS];
static int g_realize_n = 0;

void poly_realize_begin(PolyCtx *ctx) {
  (void)ctx;
  g_realize_n = 0;
}

void poly_realize_bind(PolyCtx *ctx, PolyUOp *buffer, void *data) {
  (void)ctx;
  if (g_realize_n >= MAX_REALIZE_BUFS) {
    fprintf(stderr, "polygrad: realize_bind: too many bindings (max %d)\n",
            MAX_REALIZE_BUFS);
    return;
  }
  g_realize_bindings[g_realize_n].buffer = buffer;
  g_realize_bindings[g_realize_n].data = data;
  g_realize_n++;
}

int poly_realize_exec(PolyCtx *ctx, PolyUOp *tensor_sink) {
  int ret = poly_realize(ctx, tensor_sink, g_realize_bindings, g_realize_n);
  g_realize_n = 0;
  return ret;
}

/* ── CUDA realize ────────────────────────────────────────────────────── */

#ifdef POLY_HAS_CUDA

static int cuda_realize_counter = 0;

/* CUDA compiled program cache */
typedef struct {
  uint32_t hash;
  PolyCudaProgram *prog;
  int grid_x;      /* grid blocks in x */
  int block_size;
} CudaCacheEntry;

static CudaCacheEntry cuda_prog_cache[PROG_CACHE_CAP];
static int cuda_prog_cache_n = 0;

/* GPU buffer cache: keeps GPU allocations alive between realize() calls.
 * Key: (host_ptr, bytes) → CUdeviceptr.
 * Eliminates repeated alloc + H2D overhead for unchanged inputs. */
#define GPU_BUF_CACHE_CAP 2048
typedef struct {
  void *host_ptr;
  size_t bytes;
  unsigned long long gpu_ptr;
} GpuBufCacheEntry;

static GpuBufCacheEntry gpu_buf_cache[GPU_BUF_CACHE_CAP];
static int gpu_buf_cache_n = 0;

static GpuBufCacheEntry *gpu_buf_lookup(void *host, size_t bytes) {
  for (int i = 0; i < gpu_buf_cache_n; i++)
    if (gpu_buf_cache[i].host_ptr == host && gpu_buf_cache[i].bytes == bytes)
      return &gpu_buf_cache[i];
  return NULL;
}

static void gpu_buf_insert(void *host, size_t bytes, unsigned long long gpu) {
  if (gpu_buf_cache_n < GPU_BUF_CACHE_CAP) {
    gpu_buf_cache[gpu_buf_cache_n].host_ptr = host;
    gpu_buf_cache[gpu_buf_cache_n].bytes = bytes;
    gpu_buf_cache[gpu_buf_cache_n].gpu_ptr = gpu;
    gpu_buf_cache_n++;
  }
}

static CudaCacheEntry *cuda_cache_get(uint32_t h) {
  for (int i = 0; i < cuda_prog_cache_n; i++)
    if (cuda_prog_cache[i].hash == h) return &cuda_prog_cache[i];
  return NULL;
}

static void cuda_cache_put(uint32_t h, PolyCudaProgram *prog, int gx, int bs) {
  if (cuda_prog_cache_n < PROG_CACHE_CAP) {
    cuda_prog_cache[cuda_prog_cache_n].hash = h;
    cuda_prog_cache[cuda_prog_cache_n].prog = prog;
    cuda_prog_cache[cuda_prog_cache_n].grid_x = gx;
    cuda_prog_cache[cuda_prog_cache_n].block_size = bs;
    cuda_prog_cache_n++;
  }
}

void poly_cuda_prog_cache_flush(void) {
  for (int i = 0; i < cuda_prog_cache_n; i++)
    poly_cuda_program_destroy(cuda_prog_cache[i].prog);
  cuda_prog_cache_n = 0;
}

int poly_realize_cuda(PolyCtx *ctx, PolyUOp *tensor_sink,
                     PolyBufferBinding *bindings, int n_bindings) {
  if (!tensor_sink || tensor_sink->op != POLY_OP_SINK) {
    fprintf(stderr, "polygrad: realize_cuda: expected SINK\n");
    return -1;
  }

  if (!poly_cuda_available()) {
    fprintf(stderr, "polygrad: realize_cuda: CUDA not available\n");
    return -1;
  }

  /* Structural hash for CUDA program cache */
  uint32_t cache_hash = structural_hash(tensor_sink) ^ (SCHED_CACHE_VERSION * 2654435761u);

  /* 1. Schedule */
  PolyScheduleResult sr = poly_schedule_v2(ctx, tensor_sink);
  if (sr.n_kernels < 1) { poly_schedule_result_free(&sr); return -1; }

  int n_int = sr.n_intermediates;

  /* GPU memory for intermediate buffers */
  unsigned long long *d_intermediates = NULL;
  if (n_int > 0) {
    d_intermediates = calloc(n_int, sizeof(unsigned long long));
    for (int b = 0; b < n_int; b++) {
      int itemsize = (sr.intermediate_itemsizes && sr.intermediate_itemsizes[b] > 0)
                       ? sr.intermediate_itemsizes[b] : (int)sizeof(float);
      d_intermediates[b] = poly_cuda_alloc(sr.intermediate_sizes[b] * itemsize);
    }
  }

  /* Track GPU allocations for user-bound buffers. Map BUFFER UOp → device ptr.
   * We use a flat array: up to MAX_REALIZE_BUFS entries. */
  PolyUOp *bound_bufs[MAX_REALIZE_BUFS];
  unsigned long long bound_dptrs[MAX_REALIZE_BUFS];
  int n_bound = 0;

  for (int i = 0; i < n_bindings && n_bound < MAX_REALIZE_BUFS; i++) {
    PolyUOp *buf = bindings[i].buffer;

    /* Determine buffer size from the shape (arg is int = number of elements) */
    int64_t n_elems = buf->arg.i;
    size_t bytes = (size_t)n_elems * poly_dtype_itemsize(poly_dtype_scalar(buf->dtype));

    /* Check GPU buffer cache — reuse allocation, always refresh data.
     * Host pointers can be reused by malloc after free, so cached GPU data
     * may be stale even when the host pointer matches. */
    unsigned long long dptr = 0;
    GpuBufCacheEntry *cached_buf = gpu_buf_lookup(bindings[i].data, bytes);
    if (cached_buf) {
      dptr = cached_buf->gpu_ptr;
    } else {
      dptr = poly_cuda_alloc(bytes);
      if (!dptr) {
        fprintf(stderr, "polygrad: realize_cuda: GPU alloc failed for buffer %d\n", i);
        if (d_intermediates) {
          for (int b = 0; b < n_int; b++) poly_cuda_free(d_intermediates[b]);
          free(d_intermediates);
        }
        poly_schedule_result_free(&sr);
        return -1;
      }
      /* Cache the allocation */
      gpu_buf_insert(bindings[i].data, bytes, dptr);
    }
    /* Always copy host → device (host data may have changed since last cache) */
    poly_cuda_copy_htod(dptr, bindings[i].data, bytes);

    bound_bufs[n_bound] = buf;
    bound_dptrs[n_bound] = dptr;
    n_bound++;
  }

  /* Build intermediate buffer lookup */
  PolyMap *inter_set = NULL;
  if (n_int > 0 && sr.intermediate_buf_uops) {
    inter_set = poly_map_new(n_int < 4 ? 4 : (size_t)n_int);
    for (int b = 0; b < n_int; b++) {
      PolyUOp *ib = sr.intermediate_buf_uops[b];
      poly_map_set(inter_set, ptr_hash(ib), ib,
                   (PolyUOp *)(intptr_t)(b + 1), ptr_eq);
    }
  }

  int ret = 0;
  for (int step = 0; step < sr.n_kernels && ret == 0; step++) {
    int k = sr.exec_order ? sr.exec_order[step] : step;
    if (!sr.kernel_n_params || !sr.param_to_buf || !sr.param_to_buf[k]) {
      fprintf(stderr, "polygrad: realize_cuda: missing PARAM mapping for kernel %d\n", k);
      ret = -1; break;
    }
    int n_params = sr.kernel_n_params[k];

    /* Build CUDA args: array of pointers to CUdeviceptr values */
    unsigned long long *dptrs = calloc(n_params, sizeof(unsigned long long));
    void **args = calloc(n_params, sizeof(void *));

    for (int i = 0; i < n_params; i++) {
      PolyUOp *buf = sr.param_to_buf[k][i];
      bool found = false;

      if (buf->op == POLY_OP_BUFFER) {
        for (int j = 0; j < n_bound; j++) {
          if (bound_bufs[j] == buf) {
            dptrs[i] = bound_dptrs[j];
            found = true;
            break;
          }
        }
        if (!found && inter_set) {
          PolyUOp *v = poly_map_get(inter_set, ptr_hash(buf), buf, ptr_eq);
          if (v) {
            dptrs[i] = d_intermediates[(int)((intptr_t)v - 1)];
            found = true;
          }
        }
      } else {
        assert(buf->op != POLY_OP_BUFFERIZE &&
               "unexpected BUFFERIZE in param_to_buf (old split path removed)");
      }

      if (!found) {
        fprintf(stderr, "polygrad: realize_cuda: no GPU buffer for param %d in kernel %d\n",
                i, k);
        ret = -1; break;
      }
      args[i] = &dptrs[i];
    }

    if (ret == 0) {
      uint32_t kern_hash = cache_hash + (uint32_t)k;
      CudaCacheEntry *cached = cuda_cache_get(kern_hash);

      if (cached) {
        /* Cache hit — just launch with stored grid/block dims */
        ret = poly_cuda_launch(cached->prog, args, n_params,
                               cached->grid_x, 1, 1, cached->block_size, 1, 1);
        if (ret == 0) ret = poly_cuda_sync();
      } else {
        /* Cache miss — full pipeline */
        int n_lin;
        PolyUOp **lin = poly_linearize_cuda(ctx, sr.kernels[k], &n_lin);
        if (!lin) { ret = -1; free(dptrs); free(args); break; }

        /* Extract grid/block size from SPECIAL ops.
         * gidx* → global parallelism (grid dimension)
         * lidx* → local parallelism (block dimension) */
        int grid_size = 0;
        int local_size = 0;
        for (int j = 0; j < n_lin; j++) {
          if (lin[j]->op == POLY_OP_SPECIAL && lin[j]->n_src > 0 &&
              lin[j]->src[0]->op == POLY_OP_CONST) {
            const char *sname = lin[j]->arg.str;
            if (sname && sname[0] == 'l') {
              local_size = (int)lin[j]->src[0]->arg.i;
            } else {
              grid_size = (int)lin[j]->src[0]->arg.i;
            }
          }
        }

        char fn_name[32];
        snprintf(fn_name, sizeof(fn_name), "k%d", cuda_realize_counter++);

        int block_size = local_size > 0 ? local_size : 256;
        char *src = poly_render_cuda(lin, n_lin, fn_name, block_size);
        free(lin);

        if (!src) { ret = -1; free(dptrs); free(args); break; }

        if (getenv("POLY_DUMP_CUDA"))
          fprintf(stderr, "=== CUDA SOURCE (%s) ===\n%s\n=== END ===\n", fn_name, src);

        PolyCudaProgram *prog = poly_compile_cuda(src, fn_name);
        if (!prog) {
          fprintf(stderr, "=== FAILED CUDA SOURCE (%s) ===\n%s\n=== END ===\n", fn_name, src);
          free(src); ret = -1; free(dptrs); free(args); break;
        }
        free(src);

        int gx;
        if (grid_size > 0 && local_size > 0) {
          /* Both gidx and lidx: grid = gidx_dim, block = lidx_dim */
          gx = grid_size;
        } else if (local_size > 0) {
          /* Only lidx (pure reduce): single block */
          gx = 1;
        } else if (grid_size > 0) {
          /* Only gidx (elementwise): current behavior */
          gx = (grid_size + block_size - 1) / block_size;
        } else {
          gx = 1;
        }
        ret = poly_cuda_launch(prog, args, n_params, gx, 1, 1, block_size, 1, 1);
        if (ret == 0) ret = poly_cuda_sync();

        /* Cache the compiled program (don't destroy it) */
        cuda_cache_put(kern_hash, prog, gx, block_size);
      }
    }

    free(dptrs);
    free(args);
  }

  /* GPU buffers stay resident — no automatic D2H copy.
   * Use poly_cuda_copyback() to explicitly read results.
   *
   * Don't free user-bound GPU memory — stays in cache for reuse.
   * Only free intermediates (not cached). */
  if (inter_set) poly_map_destroy(inter_set);
  if (d_intermediates) {
    for (int b = 0; b < n_int; b++) poly_cuda_free(d_intermediates[b]);
    free(d_intermediates);
  }
  poly_schedule_result_free(&sr);
  return ret;
}

void poly_cuda_flush_buffers(void) {
  for (int i = 0; i < gpu_buf_cache_n; i++)
    poly_cuda_free(gpu_buf_cache[i].gpu_ptr);
  gpu_buf_cache_n = 0;
}

int poly_cuda_copyback(PolyBufferBinding *bindings, int n_bindings) {
  for (int i = 0; i < n_bindings; i++) {
    int64_t n_elems = bindings[i].buffer->arg.i;
    size_t bytes = (size_t)n_elems * poly_dtype_itemsize(
        poly_dtype_scalar(bindings[i].buffer->dtype));
    GpuBufCacheEntry *cached = gpu_buf_lookup(bindings[i].data, bytes);
    if (cached) {
      poly_cuda_copy_dtoh(bindings[i].data, cached->gpu_ptr, bytes);
    }
  }
  return 0;
}

#endif /* POLY_HAS_CUDA */

#endif /* !__EMSCRIPTEN__ */

/* ── WASM kernel rendering ───────────────────────────────────────────── */

static PolyUOp *g_kernel_bufs[MAX_REALIZE_BUFS];
static int g_kernel_n_bufs = 0;

struct PolyWasmStepPlan {
  int n_kernels;
  uint8_t **kernel_bytes;
  int *kernel_lens;
  int *kernel_n_params;
  int **kernel_param_buf_idxs;
  int n_total_buffers;
  int n_bindable_buffers;
  int *exec_order;
  int64_t *buffer_sizes;  /* element count per buffer (bindable + intermediate) */
  int *buffer_itemsizes;  /* bytes per element per buffer (e.g. 4 for f32, 8 for f64) */
};

void poly_wasm_stepplan_destroy(PolyWasmStepPlan *p);

uint8_t *poly_render_kernel_wasm(PolyCtx *ctx, PolyUOp *tensor_sink,
                                 int *wasm_len, int *n_bufs_out) {
  if (!tensor_sink || tensor_sink->op != POLY_OP_SINK) {
    fprintf(stderr, "polygrad: render_kernel_wasm: expected SINK\n");
    *wasm_len = 0;
    *n_bufs_out = 0;
    return NULL;
  }

  /* Extract computation UOp (cache key): SINK → STORE → value */
  PolyUOp *comp = NULL;
  if (tensor_sink->n_src > 0 && tensor_sink->src[0]->op == POLY_OP_STORE &&
      tensor_sink->src[0]->n_src >= 2) {
    comp = tensor_sink->src[0]->src[1];
  }

  /* Check kernel cache (structural: matches even with different BUFFER UOps) */
  if (comp) {
    uint32_t h = structural_hash(comp);
    PolyCachedKernel *cached = poly_map_get(poly_ctx_kernel_cache(ctx), h, comp, structural_eq);
    if (cached) {
      /* Cache hit — return copy of cached bytes, collect NEW buffer ordering */
      uint8_t *copy = malloc(cached->len);
      if (copy) memcpy(copy, cached->bytes, cached->len);
      g_kernel_n_bufs = collect_ordered_buffers(ctx, tensor_sink,
                                                 g_kernel_bufs, MAX_REALIZE_BUFS);
      *wasm_len = cached->len;
      *n_bufs_out = g_kernel_n_bufs;
      return copy;
    }
  }

  /* Cache miss — full pipeline */

  /* 1. Reconstruct buffer ordering */
  g_kernel_n_bufs = collect_ordered_buffers(ctx, tensor_sink,
                                             g_kernel_bufs, MAX_REALIZE_BUFS);

  /* 2. Schedule */
  PolyUOp *kernel = poly_schedule(ctx, tensor_sink);
  if (!kernel) { *wasm_len = 0; *n_bufs_out = 0; return NULL; }

  /* 3. Linearize */
  int n_lin;
  PolyUOp **lin = poly_linearize(ctx, kernel, &n_lin);
  if (!lin) { *wasm_len = 0; *n_bufs_out = 0; return NULL; }

  /* 4. Render to WASM binary */
  int size;
  uint8_t *wasm = poly_render_wasm(lin, n_lin, &size, false /* scalar */);
  free(lin);

  /* Store in kernel cache */
  if (comp && wasm && size > 0) {
    PolyCachedKernel *ck = malloc(sizeof(PolyCachedKernel));
    if (ck) {
      ck->bytes = malloc(size);
      if (ck->bytes) {
        memcpy(ck->bytes, wasm, size);
        ck->len = size;
        ck->n_bufs = g_kernel_n_bufs;
        memcpy(ck->bufs, g_kernel_bufs, g_kernel_n_bufs * sizeof(PolyUOp *));
        poly_map_set(poly_ctx_kernel_cache(ctx), structural_hash(comp), comp, ck, structural_eq);
      } else {
        free(ck);
      }
    }
  }

  *wasm_len = size;
  *n_bufs_out = g_kernel_n_bufs;
  return wasm;
}

PolyUOp *poly_kernel_buf(PolyCtx *ctx, int index) {
  (void)ctx;
  if (index < 0 || index >= g_kernel_n_bufs) return NULL;
  return g_kernel_bufs[index];
}

PolyWasmStepPlan *poly_render_step_wasm_plan(PolyCtx *ctx, PolyUOp *tensor_sink) {
  if (!ctx || !tensor_sink || tensor_sink->op != POLY_OP_SINK) {
    fprintf(stderr, "polygrad: wasm_stepplan: expected SINK\n");
    return NULL;
  }

  PolyScheduleResult sr = poly_schedule_v2(ctx, tensor_sink);
  if (sr.n_kernels <= 0 || !sr.kernels) {
    poly_schedule_result_free(&sr);
    return NULL;
  }

  PolyWasmStepPlan *p = calloc(1, sizeof(*p));
  if (!p) {
    poly_schedule_result_free(&sr);
    return NULL;
  }
  p->n_kernels = sr.n_kernels;
  p->kernel_bytes = calloc((size_t)p->n_kernels, sizeof(uint8_t *));
  p->kernel_lens = calloc((size_t)p->n_kernels, sizeof(int));
  p->kernel_n_params = calloc((size_t)p->n_kernels, sizeof(int));
  p->kernel_param_buf_idxs = calloc((size_t)p->n_kernels, sizeof(int *));
  p->exec_order = calloc((size_t)p->n_kernels, sizeof(int));
  if (!p->kernel_bytes || !p->kernel_lens || !p->kernel_n_params ||
      !p->kernel_param_buf_idxs || !p->exec_order) {
    poly_wasm_stepplan_destroy(p);
    poly_schedule_result_free(&sr);
    return NULL;
  }

  PolyUOp *ext_bufs[MAX_REALIZE_BUFS];
  int n_ext = collect_ordered_buffers(ctx, tensor_sink, ext_bufs, MAX_REALIZE_BUFS);
  p->n_bindable_buffers = n_ext;
  p->n_total_buffers = n_ext + sr.n_intermediates;

  /* Populate global buffer list so poly_kernel_buf() works with step plan */
  g_kernel_n_bufs = n_ext;
  if (n_ext > 0) memcpy(g_kernel_bufs, ext_bufs, (size_t)n_ext * sizeof(PolyUOp *));

  /* Store per-buffer element counts (bindable from UOp arg, intermediate from schedule) */
  p->buffer_sizes = calloc((size_t)p->n_total_buffers, sizeof(int64_t));
  if (p->buffer_sizes) {
    for (int i = 0; i < n_ext; i++)
      p->buffer_sizes[i] = ext_bufs[i]->arg.i;
    for (int i = 0; i < sr.n_intermediates; i++)
      p->buffer_sizes[n_ext + i] = sr.intermediate_sizes ? sr.intermediate_sizes[i] : 0;
  }

  /* Store per-buffer itemsizes (bytes per element, dtype-aware) */
  p->buffer_itemsizes = calloc((size_t)p->n_total_buffers, sizeof(int));
  if (p->buffer_itemsizes) {
    for (int i = 0; i < n_ext; i++)
      p->buffer_itemsizes[i] = poly_dtype_itemsize(poly_dtype_scalar(ext_bufs[i]->dtype));
    for (int i = 0; i < sr.n_intermediates; i++)
      p->buffer_itemsizes[n_ext + i] = sr.intermediate_itemsizes ? sr.intermediate_itemsizes[i] : 4;
  }

  for (int k = 0; k < p->n_kernels; k++) {
    int n_lin = 0;
    PolyUOp **lin = poly_linearize(ctx, sr.kernels[k], &n_lin);
    if (!lin) {
      poly_wasm_stepplan_destroy(p);
      poly_schedule_result_free(&sr);
      return NULL;
    }
    int len = 0;
    uint8_t *bytes = poly_render_wasm(lin, n_lin, &len, false);
    free(lin);
    if (!bytes || len <= 0) {
      free(bytes);
      poly_wasm_stepplan_destroy(p);
      poly_schedule_result_free(&sr);
      return NULL;
    }
    p->kernel_bytes[k] = bytes;
    p->kernel_lens[k] = len;
    int n_params = (sr.kernel_n_params ? sr.kernel_n_params[k] : 0);
    p->kernel_n_params[k] = n_params;
    if (n_params > 0) {
      p->kernel_param_buf_idxs[k] = malloc((size_t)n_params * sizeof(int));
      if (!p->kernel_param_buf_idxs[k]) {
        poly_wasm_stepplan_destroy(p);
        poly_schedule_result_free(&sr);
        return NULL;
      }
      for (int i = 0; i < n_params; i++) p->kernel_param_buf_idxs[k][i] = -1;
      for (int i = 0; i < n_params; i++) {
        PolyUOp *pb = (sr.param_to_buf && sr.param_to_buf[k]) ? sr.param_to_buf[k][i] : NULL;
        int idx = -1;
        if (pb) idx = find_buf_position(pb, ext_bufs, n_ext);
        if (idx < 0 && pb && sr.intermediate_buf_uops && sr.n_intermediates > 0) {
          int ib = find_buf_position(pb, sr.intermediate_buf_uops, sr.n_intermediates);
          if (ib >= 0) idx = n_ext + ib;
        }
        if (idx < 0 && (!sr.param_to_buf || !sr.param_to_buf[k]) && i < n_ext) idx = i;
        p->kernel_param_buf_idxs[k][i] = idx;
      }
    }
  }

  if (sr.exec_order) memcpy(p->exec_order, sr.exec_order, (size_t)p->n_kernels * sizeof(int));
  else for (int i = 0; i < p->n_kernels; i++) p->exec_order[i] = i;

  poly_schedule_result_free(&sr);
  return p;
}

int poly_wasm_stepplan_n_kernels(const PolyWasmStepPlan *p) {
  return p ? p->n_kernels : 0;
}

const uint8_t *poly_wasm_stepplan_kernel_bytes(const PolyWasmStepPlan *p, int k, int *len) {
  if (!p || k < 0 || k >= p->n_kernels) return NULL;
  if (len) *len = p->kernel_lens[k];
  return p->kernel_bytes[k];
}

int poly_wasm_stepplan_kernel_n_params(const PolyWasmStepPlan *p, int k) {
  if (!p || k < 0 || k >= p->n_kernels) return 0;
  return p->kernel_n_params[k];
}

int poly_wasm_stepplan_n_buffers(const PolyWasmStepPlan *p) {
  return p ? p->n_total_buffers : 0;
}

int poly_wasm_stepplan_n_bindable_buffers(const PolyWasmStepPlan *p) {
  return p ? p->n_bindable_buffers : 0;
}

int poly_wasm_stepplan_kernel_param_buf_index(const PolyWasmStepPlan *p, int k, int param_idx) {
  if (!p || k < 0 || k >= p->n_kernels) return -1;
  if (param_idx < 0 || param_idx >= p->kernel_n_params[k]) return -1;
  return p->kernel_param_buf_idxs && p->kernel_param_buf_idxs[k]
    ? p->kernel_param_buf_idxs[k][param_idx] : -1;
}

const int *poly_wasm_stepplan_exec_order(const PolyWasmStepPlan *p, int *n) {
  if (!p) return NULL;
  if (n) *n = p->n_kernels;
  return p->exec_order;
}

int64_t poly_wasm_stepplan_buf_size(const PolyWasmStepPlan *p, int buf_idx) {
  if (!p || buf_idx < 0 || buf_idx >= p->n_total_buffers) return 0;
  return p->buffer_sizes ? p->buffer_sizes[buf_idx] : 0;
}

int64_t poly_wasm_stepplan_buf_nbytes(const PolyWasmStepPlan *p, int buf_idx) {
  if (!p || buf_idx < 0 || buf_idx >= p->n_total_buffers) return 0;
  int64_t elems = p->buffer_sizes ? p->buffer_sizes[buf_idx] : 0;
  int itemsize = (p->buffer_itemsizes && buf_idx < p->n_total_buffers)
                   ? p->buffer_itemsizes[buf_idx] : 4;
  return elems * itemsize;
}

int poly_wasm_stepplan_bindable_buf_index(const PolyWasmStepPlan *p, int bi) {
  if (!p || bi < 0 || bi >= p->n_bindable_buffers) return -1;
  return bi;  /* identity today; explicit contract for future indirection */
}

const void *poly_const_buffer_data(PolyCtx *ctx, PolyUOp *buf) {
  return const_registry_lookup(ctx, buf);
}

int poly_abi_version(void) {
  return POLYGRAD_ABI_VERSION;
}

void poly_wasm_stepplan_destroy(PolyWasmStepPlan *p) {
  if (!p) return;
  if (p->kernel_bytes) {
    for (int i = 0; i < p->n_kernels; i++) free(p->kernel_bytes[i]);
    free(p->kernel_bytes);
  }
  if (p->kernel_param_buf_idxs) {
    for (int i = 0; i < p->n_kernels; i++) free(p->kernel_param_buf_idxs[i]);
    free(p->kernel_param_buf_idxs);
  }
  free(p->kernel_lens);
  free(p->kernel_n_params);
  free(p->exec_order);
  free(p->buffer_sizes);
  free(p->buffer_itemsizes);
  free(p);
}

/* Debug helper — check opset state */
void poly_debug_opsets(void) {
  fprintf(stderr, "=== OpSet debug ===\n");
  fprintf(stderr, "POLY_GROUP_ALU.bits = [%llu, %llu]\n",
    (unsigned long long)POLY_GROUP_ALU.bits[0],
    (unsigned long long)POLY_GROUP_ALU.bits[1]);
  fprintf(stderr, "POLY_GROUP_UNARY.bits = [%llu, %llu]\n",
    (unsigned long long)POLY_GROUP_UNARY.bits[0],
    (unsigned long long)POLY_GROUP_UNARY.bits[1]);
  fprintf(stderr, "POLY_GROUP_BINARY.bits = [%llu, %llu]\n",
    (unsigned long long)POLY_GROUP_BINARY.bits[0],
    (unsigned long long)POLY_GROUP_BINARY.bits[1]);
  fprintf(stderr, "poly_opset_has(ALU, ADD=%d) = %d\n",
    POLY_OP_ADD, poly_opset_has(POLY_GROUP_ALU, POLY_OP_ADD));
  fprintf(stderr, "poly_opset_has(BINARY, ADD=%d) = %d\n",
    POLY_OP_ADD, poly_opset_has(POLY_GROUP_BINARY, POLY_OP_ADD));
  fprintf(stderr, "===================\n");
}

/* ── Einsum ──────────────────────────────────────────────────────────── */
/*
 * Port of tinygrad's Tensor.einsum (tensor.py:2133-2171).
 * Core algorithm: parse → size dict → align(permute+reshape+expand) → mul → sum → permute.
 * Supports lowercase a-z indices only. No ellipsis or trace (repeated indices) for v0.
 */

#define MAX_EINSUM_TENSORS 8

PolyUOp *poly_einsum(PolyCtx *ctx, const char *formula,
                     PolyUOp **tensors, const int64_t **shapes, const int *ndims,
                     int n_tensors, int64_t *out_shape, int *out_ndim) {
  if (!formula || n_tensors <= 0 || n_tensors > MAX_EINSUM_TENSORS) return NULL;

  /* Strip spaces */
  char clean[256];
  int ci = 0;
  for (const char *p = formula; *p && ci < 255; p++)
    if (*p != ' ') clean[ci++] = *p;
  clean[ci] = '\0';

  /* Split by "->" */
  char lhs_buf[256], rhs_buf[64];
  char *arrow = strstr(clean, "->");
  if (arrow) {
    int lhs_len = (int)(arrow - clean);
    memcpy(lhs_buf, clean, lhs_len);
    lhs_buf[lhs_len] = '\0';
    strcpy(rhs_buf, arrow + 2);
  } else {
    strcpy(lhs_buf, clean);
    int count[26] = {0};
    for (char *p2 = lhs_buf; *p2; p2++)
      if (*p2 >= 'a' && *p2 <= 'z') count[*p2 - 'a']++;
    int ri = 0;
    for (int i = 0; i < 26; i++)
      if (count[i] == 1) rhs_buf[ri++] = (char)('a' + i);
    rhs_buf[ri] = '\0';
  }

  /* Split lhs by comma */
  char *input_specs[MAX_EINSUM_TENSORS];
  int n_inputs = 0;
  char *pp = lhs_buf;
  while (*pp && n_inputs < MAX_EINSUM_TENSORS) {
    input_specs[n_inputs++] = pp;
    while (*pp && *pp != ',') pp++;
    if (*pp == ',') *pp++ = '\0';
  }
  if (n_inputs != n_tensors) return NULL;

  /* Build size dict (first pass — before trace reduction) */
  int64_t sz[26];
  bool has_letter[26];
  memset(has_letter, 0, sizeof(has_letter));
  for (int t = 0; t < n_tensors; t++) {
    const char *spec = input_specs[t];
    int spec_len = (int)strlen(spec);
    if (spec_len != ndims[t]) return NULL;
    for (int d = 0; d < spec_len; d++) {
      int li = spec[d] - 'a';
      if (li < 0 || li >= 26) return NULL;
      if (has_letter[li]) {
        if (sz[li] != shapes[t][d]) return NULL;
      } else {
        sz[li] = shapes[t][d];
        has_letter[li] = true;
      }
    }
  }

  /* Trace: extract diagonal when a letter repeats in a single input.
   * Matches tinygrad: for 2D, use diagonal(). For higher dims, permute
   * repeated dims to end, then flatten+pad+unflatten+shrink.
   * Diagonal trick: flatten last 2 dims (n,n)->(n*n), pad to (n*n+n),
   * reshape to (n, n+1), take column 0 via shrink -> (n,). */
  char trace_specs[MAX_EINSUM_TENSORS][64];
  PolyUOp *trace_tensors[MAX_EINSUM_TENSORS];
  int trace_ndims[MAX_EINSUM_TENSORS];
  int64_t trace_shapes[MAX_EINSUM_TENSORS][POLY_MAX_DIMS];

  for (int t = 0; t < n_tensors; t++) {
    strcpy(trace_specs[t], input_specs[t]);
    trace_tensors[t] = tensors[t];
    trace_ndims[t] = ndims[t];
    for (int d = 0; d < ndims[t]; d++)
      trace_shapes[t][d] = shapes[t][d];
  }

  for (int t = 0; t < n_tensors; t++) {
    char *s = trace_specs[t];
    int slen = (int)strlen(s);
    PolyUOp *x = trace_tensors[t];
    int x_ndim = trace_ndims[t];
    int64_t *x_shape = trace_shapes[t];

    /* For each unique letter, while it appears more than once, extract diagonal */
    for (int ci = 0; ci < slen; ci++) {
      char c = s[ci];
      /* Find second occurrence */
      int ki = -1;
      for (int k = ci + 1; k < slen; k++)
        if (s[k] == c) { ki = k; break; }
      if (ki < 0) continue;

      int64_t n = x_shape[ci];
      /* Dims ci and ki hold the repeated index; extract diagonal */

      /* Permute: move dims ci,ki to the end */
      int64_t perm[POLY_MAX_DIMS];
      int pi = 0;
      for (int d = 0; d < x_ndim; d++)
        if (d != ci && d != ki) perm[pi++] = d;
      perm[pi++] = ci;
      perm[pi++] = ki;
      x = poly_permute(ctx, x, perm, x_ndim);

      /* Update shape after permute */
      int64_t pshape[POLY_MAX_DIMS];
      for (int d = 0; d < x_ndim; d++)
        pshape[d] = x_shape[perm[d]];
      memcpy(x_shape, pshape, x_ndim * sizeof(int64_t));

      /* Flatten last 2 dims: (..., n, n) -> (..., n*n) */
      int64_t flat_shape[POLY_MAX_DIMS];
      int flat_ndim = x_ndim - 1;
      for (int d = 0; d < flat_ndim - 1; d++)
        flat_shape[d] = x_shape[d];
      flat_shape[flat_ndim - 1] = n * n;
      x = poly_reshape(ctx, x, flat_shape, flat_ndim);

      /* Pad last dim by n: (..., n*n) -> (..., n*n + n) */
      int64_t pad_pairs[POLY_MAX_DIMS][2];
      for (int d = 0; d < flat_ndim; d++) {
        pad_pairs[d][0] = 0;
        pad_pairs[d][1] = 0;
      }
      pad_pairs[flat_ndim - 1][1] = n;
      x = poly_pad(ctx, x, pad_pairs, flat_ndim);

      /* Reshape to (..., n, n+1) */
      int64_t uf_shape[POLY_MAX_DIMS];
      int uf_ndim = flat_ndim + 1;
      for (int d = 0; d < flat_ndim - 1; d++)
        uf_shape[d] = flat_shape[d];
      uf_shape[flat_ndim - 1] = n;
      uf_shape[flat_ndim] = n + 1;
      x = poly_reshape(ctx, x, uf_shape, uf_ndim);

      /* Shrink last dim to take column 0: (..., n, n+1) -> (..., n, 1) */
      int64_t shrink_pairs[POLY_MAX_DIMS][2];
      for (int d = 0; d < uf_ndim; d++) {
        shrink_pairs[d][0] = 0;
        shrink_pairs[d][1] = uf_shape[d];
      }
      shrink_pairs[uf_ndim - 1][0] = 0;
      shrink_pairs[uf_ndim - 1][1] = 1;
      x = poly_shrink(ctx, x, shrink_pairs, uf_ndim);

      /* Reshape to drop the trailing 1: (..., n, 1) -> (..., n) */
      int64_t final_shape[POLY_MAX_DIMS];
      int final_ndim = uf_ndim - 1;
      for (int d = 0; d < final_ndim; d++)
        final_shape[d] = uf_shape[d];
      x = poly_reshape(ctx, x, final_shape, final_ndim);

      /* Update spec: remove the second occurrence of c */
      for (int k = ki; k < slen - 1; k++)
        s[k] = s[k + 1];
      s[slen - 1] = '\0';
      slen--;

      /* Update shape array: the new shape is final_shape */
      x_ndim = final_ndim;
      memcpy(x_shape, final_shape, final_ndim * sizeof(int64_t));

      /* Re-check from the same position in case there are more repeats */
      ci--;
    }

    trace_tensors[t] = x;
    trace_ndims[t] = x_ndim;
    input_specs[t] = trace_specs[t];
  }

  /* Rebuild size dict after trace reduction */
  memset(has_letter, 0, sizeof(has_letter));
  for (int t = 0; t < n_tensors; t++) {
    const char *spec = input_specs[t];
    int spec_len = (int)strlen(spec);
    for (int d = 0; d < spec_len; d++) {
      int li = spec[d] - 'a';
      if (!has_letter[li]) {
        sz[li] = trace_shapes[t][d];
        has_letter[li] = true;
      }
    }
  }

  /* Sorted alphabet */
  char alpha[26];
  int n_alpha = 0;
  for (int i = 0; i < 26; i++)
    if (has_letter[i]) alpha[n_alpha++] = (char)('a' + i);

  /* Align each tensor: permute(sorted) → reshape(broadcast) → expand(full) */
  PolyUOp *aligned[MAX_EINSUM_TENSORS];
  for (int t = 0; t < n_tensors; t++) {
    const char *spec = input_specs[t];
    int spec_len = (int)strlen(spec);
    PolyUOp *x = trace_tensors[t];
    if (spec_len == 0) { aligned[t] = x; continue; }

    /* Sort spec chars */
    char sorted_spec[27];
    memcpy(sorted_spec, spec, spec_len);
    sorted_spec[spec_len] = '\0';
    for (int i = 0; i < spec_len - 1; i++)
      for (int j = i + 1; j < spec_len; j++)
        if (sorted_spec[i] > sorted_spec[j]) {
          char tmp = sorted_spec[i];
          sorted_spec[i] = sorted_spec[j];
          sorted_spec[j] = tmp;
        }

    /* Permute to sorted order */
    int64_t perm[POLY_MAX_DIMS];
    bool needs_perm = false;
    for (int i = 0; i < spec_len; i++) {
      for (int j = 0; j < spec_len; j++)
        if (spec[j] == sorted_spec[i]) { perm[i] = j; break; }
      if (perm[i] != i) needs_perm = true;
    }
    if (needs_perm)
      x = poly_permute(ctx, x, perm, spec_len);

    /* Reshape: 1 for missing alphabet letters, actual size for present */
    int64_t rshape[POLY_MAX_DIMS];
    for (int i = 0; i < n_alpha; i++) {
      bool found = false;
      for (int j = 0; j < spec_len; j++)
        if (sorted_spec[j] == alpha[i]) { found = true; break; }
      rshape[i] = found ? sz[(int)(alpha[i] - 'a')] : 1;
    }
    x = poly_reshape(ctx, x, rshape, n_alpha);

    /* Expand to full alphabet shape */
    int64_t full[POLY_MAX_DIMS];
    for (int i = 0; i < n_alpha; i++)
      full[i] = sz[(int)(alpha[i] - 'a')];
    x = poly_expand(ctx, x, full, n_alpha);

    aligned[t] = x;
  }

  /* Multiply all aligned tensors */
  PolyUOp *result = aligned[0];
  for (int t = 1; t < n_tensors; t++)
    result = poly_alu2(ctx, POLY_OP_MUL, result, aligned[t]);

  /* Sum over non-output indices */
  int64_t sum_axes[POLY_MAX_DIMS];
  int n_sum = 0;
  for (int i = 0; i < n_alpha; i++) {
    bool in_rhs = false;
    for (const char *r = rhs_buf; *r; r++)
      if (*r == alpha[i]) { in_rhs = true; break; }
    if (!in_rhs)
      sum_axes[n_sum++] = i;
  }
  if (n_sum > 0)
    result = poly_reduce_axis(ctx, POLY_OP_ADD, result, sum_axes, n_sum);

  /* Remaining alphabet after summing */
  char remaining[26];
  int n_remaining = 0;
  for (int i = 0; i < n_alpha; i++) {
    bool summed = false;
    for (int j = 0; j < n_sum; j++)
      if (sum_axes[j] == i) { summed = true; break; }
    if (!summed)
      remaining[n_remaining++] = alpha[i];
  }

  /* Permute to output order */
  int rhs_len = (int)strlen(rhs_buf);
  if (rhs_len != n_remaining) return NULL;

  int64_t out_perm[POLY_MAX_DIMS];
  bool needs_final_perm = false;
  for (int i = 0; i < rhs_len; i++) {
    for (int j = 0; j < n_remaining; j++)
      if (remaining[j] == rhs_buf[i]) {
        out_perm[i] = j;
        if (j != i) needs_final_perm = true;
        break;
      }
  }
  if (needs_final_perm)
    result = poly_permute(ctx, result, out_perm, rhs_len);

  for (int i = 0; i < rhs_len; i++)
    out_shape[i] = sz[(int)(rhs_buf[i] - 'a')];
  *out_ndim = rhs_len;
  return result;
}

/* ── Rearrange (einops) ─────────────────────────────────────────────── */

#define MAX_REARRANGE_TOKENS 32

static int parse_rearrange_side(const char *s,
                                 char tokens[][32], int *n_tokens,
                                 int groups[][2], int *n_groups) {
  *n_tokens = 0;
  *n_groups = 0;
  int paren_start = -1;

  const char *p = s;
  while (*p) {
    while (*p == ' ' || *p == '\t') p++;
    if (!*p) break;

    if (*p == '(') {
      paren_start = *n_tokens;
      p++;
      continue;
    }
    if (*p == ')') {
      if (paren_start >= 0) {
        groups[*n_groups][0] = paren_start;
        groups[*n_groups][1] = *n_tokens;
        (*n_groups)++;
      }
      paren_start = -1;
      p++;
      continue;
    }

    int ti = 0;
    while (*p && *p != ' ' && *p != '\t' && *p != '(' && *p != ')' && ti < 31)
      tokens[*n_tokens][ti++] = *p++;
    tokens[*n_tokens][ti] = '\0';
    (*n_tokens)++;
    if (*n_tokens >= MAX_REARRANGE_TOKENS) break;
  }
  return *n_tokens;
}

static int64_t find_axis_size(const char *name,
                               const char *axis_names, const int64_t *axis_values,
                               int n_axis_sizes) {
  if (!axis_names || n_axis_sizes <= 0) return -1;
  const char *p = axis_names;
  int idx = 0;
  while (*p && idx < n_axis_sizes) {
    while (*p == ' ') p++;
    if (!*p) break;
    const char *start = p;
    while (*p && *p != ' ') p++;
    int len = (int)(p - start);
    if ((int)strlen(name) == len && memcmp(start, name, len) == 0)
      return axis_values[idx];
    idx++;
  }
  return -1;
}

PolyUOp *poly_rearrange(PolyCtx *ctx, const char *formula,
                        PolyUOp *x, const int64_t *shape, int ndim,
                        const char *axis_names, const int64_t *axis_values,
                        int n_axis_sizes,
                        int64_t *out_shape, int *out_ndim) {
  if (!formula || !x) return NULL;

  const char *arrow_pos = strstr(formula, "->");
  if (!arrow_pos) return NULL;

  char lhs_str[256], rhs_str[256];
  int lhs_len = (int)(arrow_pos - formula);
  memcpy(lhs_str, formula, lhs_len);
  lhs_str[lhs_len] = '\0';
  strcpy(rhs_str, arrow_pos + 2);

  char lhs_tok[MAX_REARRANGE_TOKENS][32], rhs_tok[MAX_REARRANGE_TOKENS][32];
  int lhs_grp[8][2], rhs_grp[8][2];
  int n_lt = 0, n_rt = 0, n_lg = 0, n_rg = 0;

  parse_rearrange_side(lhs_str, lhs_tok, &n_lt, lhs_grp, &n_lg);
  parse_rearrange_side(rhs_str, rhs_tok, &n_rt, rhs_grp, &n_rg);

  PolyUOp *result = x;
  int64_t cur_shape[POLY_MAX_DIMS];
  int cur_ndim = ndim;
  memcpy(cur_shape, shape, ndim * sizeof(int64_t));

  /* Phase 1: Unflatten (lhs groups) */
  if (n_lg > 0) {
    bool in_group[MAX_REARRANGE_TOKENS];
    int g_id[MAX_REARRANGE_TOKENS];
    memset(in_group, 0, sizeof(in_group));
    for (int i = 0; i < MAX_REARRANGE_TOKENS; i++) g_id[i] = -1;
    for (int g = 0; g < n_lg; g++)
      for (int i = lhs_grp[g][0]; i < lhs_grp[g][1]; i++) {
        in_group[i] = true;
        g_id[i] = g;
      }

    int64_t new_shape[POLY_MAX_DIMS];
    int new_ndim = 0, input_dim = 0, ti = 0;

    while (ti < n_lt) {
      if (in_group[ti]) {
        int g = g_id[ti];
        int gs = lhs_grp[g][0], ge = lhs_grp[g][1];
        int gc = ge - gs;
        int64_t sub[POLY_MAX_DIMS];
        int64_t known = 1;
        int unk = -1;
        for (int i = 0; i < gc; i++) {
          const char *nm = lhs_tok[gs + i];
          if (strcmp(nm, "1") == 0) { sub[i] = 1; }
          else {
            int64_t v = find_axis_size(nm, axis_names, axis_values, n_axis_sizes);
            if (v > 0) sub[i] = v;
            else { if (unk >= 0) return NULL; unk = i; sub[i] = -1; }
          }
          if (sub[i] > 0) known *= sub[i];
        }
        if (unk >= 0) {
          if (input_dim >= cur_ndim) return NULL;
          sub[unk] = cur_shape[input_dim] / known;
        }
        for (int i = 0; i < gc; i++) new_shape[new_ndim++] = sub[i];
        input_dim++;
        ti = ge;
      } else {
        if (strcmp(lhs_tok[ti], "1") == 0) new_shape[new_ndim++] = 1;
        else {
          if (input_dim >= cur_ndim) return NULL;
          new_shape[new_ndim++] = cur_shape[input_dim];
        }
        input_dim++;
        ti++;
      }
    }

    if (new_ndim != cur_ndim || memcmp(new_shape, cur_shape, cur_ndim * sizeof(int64_t)) != 0) {
      result = poly_reshape(ctx, result, new_shape, new_ndim);
      memcpy(cur_shape, new_shape, new_ndim * sizeof(int64_t));
      cur_ndim = new_ndim;
    }
  }

  /* Phase 2: Permute (lhs order → rhs order) */
  int64_t perm[POLY_MAX_DIMS];
  bool need_perm = false;
  for (int i = 0; i < n_rt; i++) {
    perm[i] = -1;
    for (int j = 0; j < n_lt; j++)
      if (strcmp(rhs_tok[i], lhs_tok[j]) == 0) { perm[i] = j; break; }
    if (perm[i] < 0) return NULL;
    if (perm[i] != i) need_perm = true;
  }
  if (need_perm) {
    result = poly_permute(ctx, result, perm, n_rt);
    int64_t ps[POLY_MAX_DIMS];
    for (int i = 0; i < n_rt; i++) ps[i] = cur_shape[perm[i]];
    memcpy(cur_shape, ps, n_rt * sizeof(int64_t));
    cur_ndim = n_rt;
  }

  /* Phase 3: Flatten (rhs groups, process right to left) */
  for (int g = n_rg - 1; g >= 0; g--) {
    int gs = rhs_grp[g][0], ge = rhs_grp[g][1];
    if (ge - gs <= 1) continue;
    int64_t flat = 1;
    for (int i = gs; i < ge && i < cur_ndim; i++) flat *= cur_shape[i];
    int64_t ns[POLY_MAX_DIMS];
    int nn = 0;
    for (int i = 0; i < gs; i++) ns[nn++] = cur_shape[i];
    ns[nn++] = flat;
    for (int i = ge; i < cur_ndim; i++) ns[nn++] = cur_shape[i];
    result = poly_reshape(ctx, result, ns, nn);
    memcpy(cur_shape, ns, nn * sizeof(int64_t));
    cur_ndim = nn;
  }

  memcpy(out_shape, cur_shape, cur_ndim * sizeof(int64_t));
  *out_ndim = cur_ndim;
  return result;
}

/* ── Transformer building blocks ───────────────────────────────────── */

PolyUOp *poly_gather(PolyCtx *ctx,
                      PolyUOp *table, const int64_t *table_shape, int table_ndim,
                      PolyUOp *indices, const int64_t *idx_shape, int idx_ndim,
                      int64_t *out_shape, int *out_ndim) {
  if (!ctx || !table || !indices || table_ndim != 2 || idx_ndim < 1) return NULL;

  /*
   * Embedding lookup: table[indices, :] using the same arange==idx trick
   * as tinygrad's Tensor._embedding_fwd, but with a reshape that avoids
   * materializing a (vocab_size) one-hot per token.
   *
   * For table (V, D) and indices (...):
   *   arange(V) -> (V,)
   *   indices.unsqueeze(-1) -> (..., 1)
   *   mask = (indices.unsqueeze(-1) == arange) -> (..., V)
   *   mask.unsqueeze(-1) * table -> (..., V, D)
   *   sum over V axis -> (..., D)
   *
   * This is the standard tinygrad embedding approach.
   * A future GATHER UOp can replace this with O(1) per token.
   */
  int64_t V = table_shape[0];
  int64_t D = table_shape[1];

  /* arange(V): flat buffer of size V */
  PolyUOp *arange_buf = poly_buffer_f32(ctx, V);

  /* indices.unsqueeze(-1): add trailing dim of 1 */
  int64_t idx_us_shape[POLY_MAX_DIMS];
  int idx_us_ndim = idx_ndim + 1;
  for (int i = 0; i < idx_ndim; i++) idx_us_shape[i] = idx_shape[i];
  idx_us_shape[idx_ndim] = 1;
  PolyUOp *idx_us = poly_reshape(ctx, indices, idx_us_shape, idx_us_ndim);

  /* Broadcast idx_us to (..., V) */
  int64_t idx_bcast[POLY_MAX_DIMS];
  for (int i = 0; i < idx_ndim; i++) idx_bcast[i] = idx_shape[i];
  idx_bcast[idx_ndim] = V;
  PolyUOp *idx_exp = poly_expand(ctx, idx_us, idx_bcast, idx_us_ndim);

  /* Reshape arange to broadcastable (1,...,1, V) */
  int64_t arange_shape[POLY_MAX_DIMS];
  int arange_ndim = idx_us_ndim;
  for (int i = 0; i < idx_ndim; i++) arange_shape[i] = 1;
  arange_shape[idx_ndim] = V;
  PolyUOp *arange_r = poly_reshape(ctx, arange_buf, arange_shape, arange_ndim);
  PolyUOp *arange_exp = poly_expand(ctx, arange_r, idx_bcast, arange_ndim);

  /* mask = eq(idx_exp, arange_exp) -> (..., V) bool */
  PolyUOp *mask = poly_eq(ctx, idx_exp, arange_exp);

  /* mask.unsqueeze(-1) -> (..., V, 1) */
  int64_t mask_us_shape[POLY_MAX_DIMS];
  int mask_us_ndim = idx_us_ndim + 1;
  for (int i = 0; i < idx_us_ndim; i++) mask_us_shape[i] = idx_bcast[i];
  mask_us_shape[idx_us_ndim] = 1;
  PolyUOp *mask_us = poly_reshape(ctx, mask, mask_us_shape, mask_us_ndim);

  /* Broadcast mask to (..., V, D) */
  int64_t mask_bcast[POLY_MAX_DIMS];
  for (int i = 0; i < idx_us_ndim; i++) mask_bcast[i] = idx_bcast[i];
  mask_bcast[idx_us_ndim] = D;
  PolyUOp *mask_exp = poly_expand(ctx, mask_us, mask_bcast, mask_us_ndim);

  /* Reshape table to (1,...,1, V, D) and broadcast to (..., V, D) */
  int64_t tbl_shape[POLY_MAX_DIMS];
  int tbl_ndim = mask_us_ndim;
  for (int i = 0; i < idx_ndim; i++) tbl_shape[i] = 1;
  tbl_shape[idx_ndim] = V;
  tbl_shape[idx_ndim + 1] = D;
  PolyUOp *tbl_r = poly_reshape(ctx, table, tbl_shape, tbl_ndim);
  PolyUOp *tbl_exp = poly_expand(ctx, tbl_r, mask_bcast, tbl_ndim);

  /* where(mask, table, 0) -> (..., V, D) */
  PolyUOp *zero = cf(ctx, tbl_exp, 0.0);
  PolyUOp *selected = poly_where_op(ctx, mask_exp, tbl_exp, zero);

  /* Sum over the V axis (idx_ndim-th axis in the result, 0-indexed) */
  int64_t reduce_axes[] = { idx_ndim };
  PolyUOp *gathered = poly_reduce_axis(ctx, POLY_OP_ADD, selected,
                                        reduce_axes, 1);

  /* REDUCE_AXIS keeps dim (keepdim=true): (..., 1, D).
   * Reshape to squeeze the reduced V axis: (..., D). */
  *out_ndim = idx_ndim + 1;
  for (int i = 0; i < idx_ndim; i++) out_shape[i] = idx_shape[i];
  out_shape[idx_ndim] = D;

  return poly_reshape(ctx, gathered, out_shape, *out_ndim);
}

PolyUOp *poly_layernorm(PolyCtx *ctx, PolyUOp *x,
                         const int64_t *shape, int ndim,
                         int axis, double eps,
                         int64_t *out_shape, int *out_ndim) {
  if (!ctx || !x || ndim < 1) return NULL;
  if (axis < 0) axis += ndim;
  PolyUOp *x_view = reshape_logical_input(ctx, x, shape, ndim);
  if (!x_view) return NULL;

  /* mean = sum(x, axis) / N */
  int64_t mean_shape[POLY_MAX_DIMS];
  int mean_ndim;
  PolyUOp *mean = poly_mean_reduce(ctx, x_view, shape, ndim, axis, 1,
                                     mean_shape, &mean_ndim);

  /* Expand mean back to original shape for broadcast */
  PolyUOp *mean_exp = poly_expand(ctx, mean, (int64_t *)shape, ndim);

  /* x - mean */
  PolyUOp *centered = poly_alu2(ctx, POLY_OP_ADD, x_view,
                                 poly_alu1(ctx, POLY_OP_NEG, mean_exp));

  /* var = mean((x - mean)^2, axis) */
  PolyUOp *sq = poly_alu2(ctx, POLY_OP_MUL, centered, centered);
  int64_t var_shape[POLY_MAX_DIMS];
  int var_ndim;
  PolyUOp *var = poly_mean_reduce(ctx, sq, shape, ndim, axis, 1,
                                    var_shape, &var_ndim);
  PolyUOp *var_exp = poly_expand(ctx, var, (int64_t *)shape, ndim);

  /* (x - mean) / sqrt(var + eps) */
  PolyUOp *eps_c = cf(ctx, x, eps);
  PolyUOp *denom = poly_alu1(ctx, POLY_OP_SQRT,
                              poly_alu2(ctx, POLY_OP_ADD, var_exp, eps_c));
  PolyUOp *result = poly_alu2(ctx, POLY_OP_MUL, centered,
                               poly_alu1(ctx, POLY_OP_RECIPROCAL, denom));

  memcpy(out_shape, shape, ndim * sizeof(int64_t));
  *out_ndim = ndim;
  return result;
}

PolyUOp *poly_causal_mask(PolyCtx *ctx, int64_t T,
                           int64_t *out_shape, int *out_ndim) {
  if (!ctx || T <= 0) return NULL;

  /* arange(T) as row: (T, 1) */
  PolyUOp *arange_buf = poly_buffer_f32(ctx, T);
  int64_t row_shape[] = { T, 1 };
  PolyUOp *row = poly_reshape(ctx, arange_buf, row_shape, 2);
  int64_t row_exp_shape[] = { T, T };
  PolyUOp *row_exp = poly_expand(ctx, row, row_exp_shape, 2);

  /* arange(T) as col: (1, T) */
  int64_t col_shape[] = { 1, T };
  PolyUOp *col = poly_reshape(ctx, arange_buf, col_shape, 2);
  PolyUOp *col_exp = poly_expand(ctx, col, row_exp_shape, 2);

  /* mask = row < col (upper triangle = True where masked) */
  PolyUOp *mask = poly_alu2(ctx, POLY_OP_CMPLT, row_exp, col_exp);

  /* where(mask, -1e9, 0) */
  PolyUOp *neg_inf = poly_const_float(ctx, -1e9);
  PolyUOp *zero = poly_const_float(ctx, 0.0);
  PolyUOp *result = poly_where_op(ctx, mask, neg_inf, zero);

  out_shape[0] = T;
  out_shape[1] = T;
  *out_ndim = 2;
  return result;
}

PolyUOp *poly_linear(PolyCtx *ctx,
                      PolyUOp *x, const int64_t *x_shape, int x_ndim,
                      PolyUOp *weight, const int64_t *w_shape, int w_ndim,
                      PolyUOp *bias, const int64_t *bias_shape, int bias_ndim,
                      int64_t *out_shape, int *out_ndim) {
  if (!ctx || !x || !weight || x_ndim < 1 || w_ndim != 2) return NULL;

  /* weight.T: (out_features, in_features) -> (in_features, out_features) */
  int64_t perm[] = { 1, 0 };
  PolyUOp *wt = poly_permute(ctx, weight, perm, 2);
  int64_t wt_shape[] = { w_shape[1], w_shape[0] };

  /* x @ weight.T */
  int64_t dot_shape[POLY_MAX_DIMS];
  int dot_ndim;
  PolyUOp *result = poly_dot(ctx, x, x_shape, x_ndim, wt, wt_shape, 2,
                              dot_shape, &dot_ndim);

  /* + bias */
  if (bias) {
    /* Reshape bias to match output: (1,...,1, out_features) */
    int64_t b_shape[POLY_MAX_DIMS];
    for (int i = 0; i < dot_ndim - 1; i++) b_shape[i] = 1;
    b_shape[dot_ndim - 1] = w_shape[0];
    PolyUOp *b_r = poly_reshape(ctx, bias, b_shape, dot_ndim);
    PolyUOp *b_exp = poly_expand(ctx, b_r, dot_shape, dot_ndim);
    result = poly_alu2(ctx, POLY_OP_ADD, result, b_exp);
  }

  memcpy(out_shape, dot_shape, dot_ndim * sizeof(int64_t));
  *out_ndim = dot_ndim;
  return result;
}

/* Debug helper — print UOp info */
void poly_debug_uop(PolyCtx *ctx, PolyUOp *u) {
  if (!u) { fprintf(stderr, "poly_debug_uop: NULL\n"); return; }
  fprintf(stderr, "UOp@%p: op=%s(%d) n_src=%d arg.kind=%d",
          (void*)u, poly_op_name(u->op), u->op, u->n_src, u->arg.kind);
  if (u->arg.kind == POLY_ARG_INT)
    fprintf(stderr, " arg.i=%lld", (long long)u->arg.i);
  if (u->arg.kind == POLY_ARG_FLOAT)
    fprintf(stderr, " arg.f=%f", u->arg.f);
  fprintf(stderr, "\n");
  for (int i = 0; i < u->n_src; i++) {
    fprintf(stderr, "  src[%d]: @%p op=%s(%d)\n", i, (void*)u->src[i],
            poly_op_name(u->src[i]->op), u->src[i]->op);
  }
  /* Try shape */
  PolyShape s = poly_uop_shape(ctx, u);
  if (s.ndim >= 0) {
    fprintf(stderr, "  shape: (");
    for (int i = 0; i < s.ndim; i++) {
      if (i) fprintf(stderr, ", ");
      fprintf(stderr, "%lld", (long long)s.dims[i]);
    }
    fprintf(stderr, ")\n");
    if (s.ndim > 0 && s.dims) free(s.dims);
  } else {
    fprintf(stderr, "  shape: NONE\n");
  }
}
