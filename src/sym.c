/*
 * sym.c — Symbolic simplification rules (phase 1)
 *
 * Mirrors tinygrad's symbolic_simple: self-folding, zero-folding,
 * constant folding, cast folding, basic identities.
 */

#include "pat.h"
#include <math.h>

/* ── vmin/vmax bounds (port of tinygrad's UOp._min_max) ──────────────── */

static void poly_uop_minmax(PolyUOp *u, int64_t *vmin, int64_t *vmax) {
  if (u->op == POLY_OP_CONST) {
    if (poly_dtype_is_float(u->dtype)) {
      *vmin = *vmax = (int64_t)u->arg.f;
    } else {
      *vmin = *vmax = u->arg.i;
    }
    return;
  }
  if (u->op == POLY_OP_RANGE) {
    *vmin = 0;
    int64_t bmax;
    poly_uop_minmax(u->src[0], &bmax, &bmax);
    *vmax = bmax - 1;
    return;
  }
  if (u->op == POLY_OP_ADD && u->n_src == 2 && poly_dtype_is_int(u->dtype)) {
    int64_t a0, a1, b0, b1;
    poly_uop_minmax(u->src[0], &a0, &a1);
    poly_uop_minmax(u->src[1], &b0, &b1);
    *vmin = a0 + b0;
    *vmax = a1 + b1;
    return;
  }
  if (u->op == POLY_OP_MUL && u->n_src == 2 && poly_dtype_is_int(u->dtype)) {
    int64_t a0, a1, b0, b1;
    poly_uop_minmax(u->src[0], &a0, &a1);
    poly_uop_minmax(u->src[1], &b0, &b1);
    int64_t v[4] = { a0*b0, a0*b1, a1*b0, a1*b1 };
    *vmin = v[0]; *vmax = v[0];
    for (int i = 1; i < 4; i++) {
      if (v[i] < *vmin) *vmin = v[i];
      if (v[i] > *vmax) *vmax = v[i];
    }
    return;
  }
  if (u->op == POLY_OP_MOD && u->n_src == 2 && poly_dtype_is_int(u->dtype)) {
    int64_t a0, a1, b0, b1;
    poly_uop_minmax(u->src[0], &a0, &a1);
    poly_uop_minmax(u->src[1], &b0, &b1);
    if (b0 == b1 && b0 > 0) {
      *vmin = (a0 >= 0) ? 0 : a0;
      *vmax = (a0 >= 0) ? ((a1 < b0) ? a1 : b0 - 1) : b0 - 1;
      return;
    }
  }
  /* Fallback: dtype range */
  if (poly_dtype_is_int(u->dtype)) {
    if (poly_dtype_eq(u->dtype, POLY_INT32)) {
      *vmin = INT32_MIN; *vmax = INT32_MAX;
    } else {
      *vmin = INT64_MIN / 2; *vmax = INT64_MAX / 2;
    }
  } else {
    *vmin = INT64_MIN / 2; *vmax = INT64_MAX / 2;
  }
}

/* C-style integer division (truncates toward zero) */
static int64_t cdiv(int64_t x, int64_t y) {
  if (y == 0) return 0;
  int64_t ax = x < 0 ? -x : x;
  int64_t ay = y < 0 ? -y : y;
  int64_t q = ax / ay;
  return (x < 0) != (y < 0) ? -q : q;
}

/* ── Rewrite callbacks ────────────────────────────────────────────────── */

/* Self-folding: return x */
static PolyUOp *rule_identity(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)ctx; (void)root;
  return poly_bind(b, "x");
}

/* x+0 -> x, x*1 -> x, x^0 -> x, x//1 -> x */
/* All use rule_identity — pattern does the matching */

/* x//x -> 1 */
static PolyUOp *rule_div_self(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  return poly_const_like_int(ctx, poly_bind(b, "x"), 1);
}

/* x//-1 -> -x */
static PolyUOp *rule_div_neg1(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  PolyUOp *x = poly_bind(b, "x");
  return poly_uop1(ctx, POLY_OP_NEG, x->dtype, x, poly_arg_none());
}

/* Idempotent(x, x) -> x (OR, AND, MAX) */
static PolyUOp *rule_idempotent(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)ctx; (void)root;
  return poly_bind(b, "x");
}

/* Zero-folding: x < x -> False */
static PolyUOp *rule_lt_self(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  return poly_const_like_bool(ctx, poly_bind(b, "x"), false);
}

/* x != x -> false (int/bool only; float NaN!=NaN is true) */
static PolyUOp *rule_cmpne_self(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  PolyUOp *x = poly_bind(b, "x");
  if (poly_dtype_is_float(x->dtype)) return NULL;  /* NaN != NaN */
  return poly_const_like_bool(ctx, x, false);
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

/* Constant folding: Unary(CONST) -> CONST */
static PolyUOp *rule_const_fold_unary(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  PolyUOp *a = poly_bind(b, "a");
  PolyArg operand = a->src[0]->arg;
  PolyArg result = poly_exec_alu(a->op, a->dtype, &operand, 1);
  return poly_const_like(ctx, a, result);
}

/* Constant folding: Binary(CONST, CONST) -> CONST */
static PolyUOp *rule_const_fold_binary(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  PolyUOp *a = poly_bind(b, "a");
  PolyArg operands[2] = { a->src[0]->arg, a->src[1]->arg };
  PolyArg result = poly_exec_alu(a->op, a->dtype, operands, 2);
  return poly_const_like(ctx, a, result);
}

/* Constant folding: Ternary(CONST, CONST, CONST) -> CONST */
static PolyUOp *rule_const_fold_ternary(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  PolyUOp *a = poly_bind(b, "a");
  PolyArg operands[3] = { a->src[0]->arg, a->src[1]->arg, a->src[2]->arg };
  PolyArg result = poly_exec_alu(a->op, a->dtype, operands, 3);
  return poly_const_like(ctx, a, result);
}

/* CAST(CONST) -> CONST with new dtype */
static PolyUOp *rule_cast_const(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  PolyUOp *c = root->src[0];
  return poly_const_like(ctx, root, c->arg);
}

/* CAST/BITCAST same dtype -> identity */
static PolyUOp *rule_cast_noop(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)ctx; (void)b;
  if (poly_dtype_eq(root->dtype, root->src[0]->dtype))
    return root->src[0];
  return NULL;
}

/* NEG(NEG(x)) -> x */
static PolyUOp *rule_double_neg(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)ctx; (void)root;
  return poly_bind(b, "x");
}

/* x / x -> 1 (float division) */
static PolyUOp *rule_fdiv_self(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  return poly_const_like_float(ctx, poly_bind(b, "x"), 1.0);
}

/* WHERE(cond, val, val) -> val */
static PolyUOp *rule_where_same(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)ctx; (void)root;
  return poly_bind(b, "val");
}

/* WHERE(true/false_const, c0, c1) -> c0 or c1 */
static PolyUOp *rule_where_const_gate(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)ctx;
  PolyUOp *gate = poly_bind(b, "gate");
  PolyUOp *c0 = poly_bind(b, "c0");
  PolyUOp *c1 = poly_bind(b, "c1");
  if (gate->arg.kind == POLY_ARG_BOOL)
    return gate->arg.b ? c0 : c1;
  if (gate->arg.kind == POLY_ARG_INT)
    return gate->arg.i ? c0 : c1;
  return NULL;
}

/* x + x -> x * 2 */
static PolyUOp *rule_add_self(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  PolyUOp *x = poly_bind(b, "x");
  PolyUOp *two = poly_const_like_int(ctx, x, 2);
  return poly_uop2(ctx, POLY_OP_MUL, x->dtype, x, two, poly_arg_none());
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
  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, x->dtype, x, two, poly_arg_none());
  return poly_uop2(ctx, POLY_OP_ADD, root->dtype, a, mul, poly_arg_none());
}

/* ── fold_divmod helpers (port of tinygrad divandmod.py) ────────────── */

/* Split ADD chain into flat list of additive terms */
static int split_add_terms(PolyUOp *u, PolyUOp **terms, int max) {
  if (max <= 0) return 0;
  if (u->op == POLY_OP_ADD && u->n_src == 2) {
    int n = split_add_terms(u->src[0], terms, max);
    return n + split_add_terms(u->src[1], terms + n, max - n);
  }
  terms[0] = u;
  return 1;
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
  if (u->op == POLY_OP_CONST && u->arg.kind == POLY_ARG_INT)
    return poly_uop0(ctx, POLY_OP_CONST, u->dtype, poly_arg_int(u->arg.i / f));
  if (u->op == POLY_OP_MUL && u->n_src == 2) {
    if (u->src[1]->op == POLY_OP_CONST && u->src[1]->arg.kind == POLY_ARG_INT) {
      int64_t c = u->src[1]->arg.i;
      if (c % f == 0) {
        if (c / f == 1) return u->src[0];
        return poly_uop2(ctx, POLY_OP_MUL, u->dtype, u->src[0],
                          poly_uop0(ctx, POLY_OP_CONST, u->dtype, poly_arg_int(c / f)),
                          poly_arg_none());
      }
    }
    if (u->src[0]->op == POLY_OP_CONST && u->src[0]->arg.kind == POLY_ARG_INT) {
      int64_t c = u->src[0]->arg.i;
      if (c % f == 0) {
        if (c / f == 1) return u->src[1];
        return poly_uop2(ctx, POLY_OP_MUL, u->dtype,
                          poly_uop0(ctx, POLY_OP_CONST, u->dtype, poly_arg_int(c / f)),
                          u->src[1], poly_arg_none());
      }
    }
  }
  return NULL;
}

/* Floor division (Python-style //) */
static int64_t floordiv(int64_t a, int64_t b) {
  int64_t q = a / b;
  if ((a ^ b) < 0 && q * b != a) q--;
  return q;
}

/* ── fold_divmod_general (port of tinygrad divandmod.py) ───────────── */

static PolyUOp *fold_divmod_general(PolyCtx *ctx, PolyUOp *root) {
  if (root->n_src != 2 || !poly_dtype_is_int(root->dtype)) return NULL;
  PolyUOp *x = root->src[0], *y = root->src[1];
  int64_t x_min, x_max, y_min, y_max;
  poly_uop_minmax(x, &x_min, &x_max);
  poly_uop_minmax(y, &y_min, &y_max);

  /* 1. cancel_divmod: all corners give same quotient */
  if (y_min * y_max > 0) {
    int64_t q00 = cdiv(x_min, y_min), q01 = cdiv(x_min, y_max);
    int64_t q10 = cdiv(x_max, y_min), q11 = cdiv(x_max, y_max);
    if (q00 == q01 && q00 == q10 && q00 == q11) {
      int64_t q = q00;
      if (root->op == POLY_OP_MOD) {
        if (q == 0) return x;
        PolyUOp *qc = poly_uop0(ctx, POLY_OP_CONST, root->dtype, poly_arg_int(q));
        PolyUOp *qy = poly_uop2(ctx, POLY_OP_MUL, root->dtype, qc, y, poly_arg_none());
        return poly_uop2(ctx, POLY_OP_ADD, root->dtype, x,
                          poly_uop1(ctx, POLY_OP_NEG, root->dtype, qy, poly_arg_none()),
                          poly_arg_none());
      }
      return poly_uop0(ctx, POLY_OP_CONST, root->dtype, poly_arg_int(q));
    }
  }

  /* 2. fold_divmod_congruence: constant positive denominator */
  if (y->op != POLY_OP_CONST || y->arg.i <= 0) return NULL;
  if (x_min < 0) return NULL;
  int64_t c = y->arg.i;

  /* Split x into additive terms */
  PolyUOp *terms[16];
  int n_terms = split_add_terms(x, terms, 16);
  if (n_terms == 0 || n_terms > 15) return NULL;

  /* Separate additive constant from non-constant terms */
  int64_t additive_const = 0;
  PolyUOp *nc_terms[16];
  int64_t nc_factors[16];
  int n_nc = 0;
  for (int i = 0; i < n_terms; i++) {
    if (terms[i]->op == POLY_OP_CONST && terms[i]->arg.kind == POLY_ARG_INT) {
      additive_const += terms[i]->arg.i;
    } else {
      nc_terms[n_nc] = terms[i];
      nc_factors[n_nc] = uop_const_factor(nc_terms[n_nc]);
      n_nc++;
    }
  }

  /* Compute remainders: rems[i] = min(f%c, f%c-c, key=abs) */
  int64_t rems[16];
  for (int i = 0; i < n_nc; i++) {
    int64_t r = nc_factors[i] % c;
    int64_t r2 = r - c;
    int64_t a1 = r < 0 ? -r : r, a2 = r2 < 0 ? -r2 : r2;
    rems[i] = (a1 <= a2) ? r : r2;
  }

  /* Get base for each term: base[i] = nc_terms[i] / nc_factors[i] */
  PolyUOp *bases[16];
  int64_t base_mins[16], base_maxs[16];
  for (int i = 0; i < n_nc; i++) {
    bases[i] = uop_divides(ctx, nc_terms[i], nc_factors[i]);
    if (!bases[i]) return NULL;
    poly_uop_minmax(bases[i], &base_mins[i], &base_maxs[i]);
  }

  /* Compute bounds of rem = sum(rems[i]*base[i]) + additive_const%c */
  int64_t crem = additive_const % c;
  int64_t rem_lo = crem, rem_hi = crem;
  for (int i = 0; i < n_nc; i++) {
    int64_t v0 = rems[i] * base_mins[i], v1 = rems[i] * base_maxs[i];
    int64_t lo = v0 < v1 ? v0 : v1, hi = v0 < v1 ? v1 : v0;
    rem_lo += lo;
    rem_hi += hi;
  }

  /* Check: rem range fits in one c-interval */
  if (floordiv(rem_lo, c) != floordiv(rem_hi, c)) return NULL;

  if (root->op == POLY_OP_MOD) {
    /* return rem - floordiv(rem_lo, c) * c */
    int64_t offset = floordiv(rem_lo, c) * c;
    int64_t const_val = crem - offset;

    /* Build rem expression */
    PolyUOp *result = NULL;
    for (int i = 0; i < n_nc; i++) {
      if (rems[i] == 0) continue;
      PolyUOp *term;
      if (rems[i] == 1) {
        term = bases[i];
      } else {
        PolyUOp *rc = poly_uop0(ctx, POLY_OP_CONST, root->dtype, poly_arg_int(rems[i]));
        term = poly_uop2(ctx, POLY_OP_MUL, root->dtype, rc, bases[i], poly_arg_none());
      }
      result = result ? poly_uop2(ctx, POLY_OP_ADD, root->dtype, result, term, poly_arg_none()) : term;
    }
    if (const_val != 0 || !result) {
      PolyUOp *cc = poly_uop0(ctx, POLY_OP_CONST, root->dtype, poly_arg_int(const_val));
      result = result ? poly_uop2(ctx, POLY_OP_ADD, root->dtype, result, cc, poly_arg_none()) : cc;
    }
    return result;
  }

  /* IDIV: sum((f-r)//c * base[i]) + (additive_const - crem + floordiv(rem_lo,c)*c)//c */
  int64_t const_part = (additive_const - crem + floordiv(rem_lo, c) * c) / c;
  PolyUOp *result = NULL;
  for (int i = 0; i < n_nc; i++) {
    int64_t coeff = (nc_factors[i] - rems[i]) / c;
    if (coeff == 0) continue;
    PolyUOp *term;
    if (coeff == 1) {
      term = bases[i];
    } else {
      PolyUOp *cc = poly_uop0(ctx, POLY_OP_CONST, root->dtype, poly_arg_int(coeff));
      term = poly_uop2(ctx, POLY_OP_MUL, root->dtype, cc, bases[i], poly_arg_none());
    }
    result = result ? poly_uop2(ctx, POLY_OP_ADD, root->dtype, result, term, poly_arg_none()) : term;
  }
  if (const_part != 0 || !result) {
    PolyUOp *cc = poly_uop0(ctx, POLY_OP_CONST, root->dtype, poly_arg_int(const_part));
    result = result ? poly_uop2(ctx, POLY_OP_ADD, root->dtype, result, cc, poly_arg_none()) : cc;
  }
  return result;
}

static PolyUOp *rule_cancel_divmod(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  return fold_divmod_general(ctx, root);
}

/* (x % c) + (x // c) * c → x.  Ref: tinygrad symbolic_simple line 50 */
static PolyUOp *rule_divmod_cancel(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  return poly_bind(b, "x");
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

/* VECTORIZE(CONST...) -> VCONST(CONST...) */
static PolyUOp *rule_vectorize_const_fold(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  if (root->op != POLY_OP_VECTORIZE || root->n_src <= 0) return NULL;
  for (int i = 0; i < root->n_src; i++) {
    if (root->src[i]->op != POLY_OP_CONST && root->src[i]->op != POLY_OP_VCONST)
      return NULL;
  }
  return poly_uop(ctx, POLY_OP_VCONST, root->dtype, root->src, root->n_src, poly_arg_none());
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
  if (n > 128) return NULL;
  PolyUOp *elts[128];
  for (int i = 0; i < n; i++) {
    int64_t idx = root->arg.int_tuple.vals[i];
    if (idx < 0 || idx >= vec->n_src) return NULL;
    elts[i] = vec->src[idx];
  }
  return poly_uop(ctx, POLY_OP_VECTORIZE, root->dtype, elts, n, poly_arg_none());
}

/* GEP(CONST/VCONST) -> selected CONST(s). */
static PolyUOp *rule_gep_const(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  if (root->op != POLY_OP_GEP || root->n_src < 1) return NULL;
  PolyUOp *c = root->src[0];
  if (!c) return NULL;
  if (c->op == POLY_OP_CONST) return c;
  if (c->op != POLY_OP_VCONST) return NULL;

  if (root->arg.kind == POLY_ARG_INT) {
    int64_t idx = root->arg.i;
    if (idx < 0) return NULL;
    if (c->n_src > idx) return c->src[idx];
    if (c->arg.kind == POLY_ARG_INT_TUPLE && idx < c->arg.int_tuple.n)
      return poly_uop0(ctx, POLY_OP_CONST, poly_dtype_scalar(c->dtype),
                       poly_arg_int(c->arg.int_tuple.vals[idx]));
    return NULL;
  }
  if (root->arg.kind != POLY_ARG_INT_TUPLE || root->arg.int_tuple.n <= 0) return NULL;
  int n = root->arg.int_tuple.n;
  if (n == 1) {
    int64_t idx = root->arg.int_tuple.vals[0];
    if (idx < 0) return NULL;
    if (c->n_src > idx) return c->src[idx];
    if (c->arg.kind == POLY_ARG_INT_TUPLE && idx < c->arg.int_tuple.n)
      return poly_uop0(ctx, POLY_OP_CONST, poly_dtype_scalar(c->dtype),
                       poly_arg_int(c->arg.int_tuple.vals[idx]));
    return NULL;
  }
  if (n > 128) return NULL;
  PolyUOp *elts[128];
  for (int i = 0; i < n; i++) {
    int64_t idx = root->arg.int_tuple.vals[i];
    if (idx < 0) return NULL;
    if (c->n_src > idx) elts[i] = c->src[idx];
    else if (c->arg.kind == POLY_ARG_INT_TUPLE && idx < c->arg.int_tuple.n)
      elts[i] = poly_uop0(ctx, POLY_OP_CONST, poly_dtype_scalar(c->dtype),
                          poly_arg_int(c->arg.int_tuple.vals[idx]));
    else return NULL;
  }
  return poly_uop(ctx, POLY_OP_VECTORIZE, root->dtype, elts, n, poly_arg_none());
}

/* GEP in natural order is identity. */
static PolyUOp *rule_gep_identity(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)ctx; (void)b;
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

/* ── Build the symbolic_simple PatternMatcher ─────────────────────────── */

static PolyPatternMatcher *g_symbolic_simple = NULL;

PolyPatternMatcher *poly_symbolic_simple(void) {
  if (g_symbolic_simple) return g_symbolic_simple;

  /* Exclude THREEFRY from binary const fold */
  PolyOpSet binary_no_threefry = POLY_GROUP_BINARY;
  binary_no_threefry.bits[POLY_OP_THREEFRY / 64] &= ~((uint64_t)1 << (POLY_OP_THREEFRY % 64));

  /* CAST | BITCAST set */
  PolyOpSet cast_set = poly_opset_add(
    poly_opset_add((PolyOpSet){{0,0}}, POLY_OP_CAST), POLY_OP_BITCAST);

  PolyRule rules[] = {
    /* -- Bool algebra (must come before generic ADD/MUL rules) -- */
    /* bool * bool -> AND */
    { poly_pat_op2(POLY_OP_MUL, poly_pat_any("x"),
        poly_pat_any("y"), NULL), rule_bool_mul_to_and },
    /* bool + bool -> OR */
    { poly_pat_op2(POLY_OP_ADD, poly_pat_any("x"),
        poly_pat_any("y"), NULL), rule_bool_add_to_or },

    /* -- Self-folding -- */
    /* x + 0 -> x */
    { poly_pat_op2c(POLY_OP_ADD, poly_pat_any("x"),
        poly_pat_const_val(poly_arg_int(0)), NULL), rule_identity },
    /* x + 0.0 -> x (float) */
    { poly_pat_op2c(POLY_OP_ADD, poly_pat_any("x"),
        poly_pat_const_val(poly_arg_float(0.0)), NULL), rule_identity },
    /* x * 1 -> x */
    { poly_pat_op2c(POLY_OP_MUL, poly_pat_any("x"),
        poly_pat_const_val(poly_arg_int(1)), NULL), rule_identity },
    /* x * 1.0 -> x (float) */
    { poly_pat_op2c(POLY_OP_MUL, poly_pat_any("x"),
        poly_pat_const_val(poly_arg_float(1.0)), NULL), rule_identity },
    /* AND(x, true) -> x */
    { poly_pat_op2c(POLY_OP_AND, poly_pat_any("x"),
        poly_pat_const_val(poly_arg_bool(true)), NULL), rule_identity },
    /* AND(x, false) -> false */
    { poly_pat_op2c(POLY_OP_AND, poly_pat_any("x"),
        poly_pat_const_val(poly_arg_bool(false)), NULL), rule_and_zero },
    /* OR(x, false) -> x */
    { poly_pat_op2c(POLY_OP_OR, poly_pat_any("x"),
        poly_pat_const_val(poly_arg_bool(false)), NULL), rule_identity },
    /* OR(x, true) -> true */
    { poly_pat_op2c(POLY_OP_OR, poly_pat_any("x"),
        poly_pat_const_val(poly_arg_bool(true)), NULL), rule_or_one },
    /* x // x -> 1 */
    { poly_pat_op2(POLY_OP_IDIV, poly_pat_any("x"),
        poly_pat_any("x"), NULL), rule_div_self },
    /* x // 1 -> x */
    { poly_pat_op2(POLY_OP_IDIV, poly_pat_any("x"),
        poly_pat_const_val(poly_arg_int(1)), NULL), rule_identity },
    /* x // -1 -> -x */
    { poly_pat_op2(POLY_OP_IDIV, poly_pat_any("x"),
        poly_pat_const_val(poly_arg_int(-1)), NULL), rule_div_neg1 },
    /* Idempotent(x, x) -> x */
    { poly_pat_ops2(POLY_GROUP_IDEMPOTENT, poly_pat_any("x"),
        poly_pat_any("x"), NULL), rule_idempotent },

    /* -- Zero-folding -- */
    /* x < x -> False */
    { poly_pat_op2(POLY_OP_CMPLT, poly_pat_any("x"),
        poly_pat_any("x"), NULL), rule_lt_self },
    /* x != x -> False (int/bool only) */
    { poly_pat_op2(POLY_OP_CMPNE, poly_pat_any("x"),
        poly_pat_any("x"), NULL), rule_cmpne_self },
    /* x % x -> 0 */
    { poly_pat_op2(POLY_OP_MOD, poly_pat_any("x"),
        poly_pat_any("x"), NULL), rule_mod_self },
    /* x ^ 0 -> x */
    { poly_pat_op2c(POLY_OP_XOR, poly_pat_any("x"),
        poly_pat_const_val(poly_arg_int(0)), NULL), rule_identity },
    /* x ^ x -> 0 */
    { poly_pat_op2(POLY_OP_XOR, poly_pat_any("x"),
        poly_pat_any("x"), NULL), rule_xor_self },
    /* x * 0 -> 0 */
    { poly_pat_op2c(POLY_OP_MUL, poly_pat_any("x"),
        poly_pat_const_val(poly_arg_int(0)), NULL), rule_mul_zero },
    /* x * 0.0 -> 0 (float) */
    { poly_pat_op2c(POLY_OP_MUL, poly_pat_any("x"),
        poly_pat_const_val(poly_arg_float(0.0)), NULL), rule_mul_zero },

    /* -- Constant folding -- */
    /* Unary(CONST) -> CONST */
    { poly_pat_ops1(POLY_GROUP_UNARY, poly_pat_cvar(NULL), "a"),
      rule_const_fold_unary },
    /* Binary(CONST, CONST) -> CONST (excl. THREEFRY) */
    { poly_pat_ops2(binary_no_threefry, poly_pat_cvar(NULL),
        poly_pat_cvar(NULL), "a"), rule_const_fold_binary },
    /* Ternary(CONST, CONST, CONST) -> CONST */
    { poly_pat_ops3(POLY_GROUP_TERNARY, poly_pat_cvar(NULL),
        poly_pat_cvar(NULL), poly_pat_cvar(NULL), "a"),
      rule_const_fold_ternary },
    /* VECTORIZE(CONST...) -> VCONST */
    { poly_pat_op(POLY_OP_VECTORIZE, NULL, 0, NULL), rule_vectorize_const_fold },
    /* GEP simplifications */
    { poly_pat_op(POLY_OP_GEP, NULL, 0, NULL), rule_gep_vectorize },
    { poly_pat_op(POLY_OP_GEP, NULL, 0, NULL), rule_gep_const },
    { poly_pat_op(POLY_OP_GEP, NULL, 0, NULL), rule_gep_identity },

    /* -- Cast folding -- */
    /* CAST(CONST) -> CONST */
    { poly_pat_op1(POLY_OP_CAST, poly_pat_cvar("c"), NULL),
      rule_cast_const },
    /* CAST/BITCAST same dtype -> identity */
    { poly_pat_ops(cast_set, NULL, 0, NULL), rule_cast_noop },

    /* -- Double negation -- */
    /* NEG(NEG(x)) -> x */
    { poly_pat_op1(POLY_OP_NEG,
        poly_pat_op1(POLY_OP_NEG, poly_pat_any("x"), NULL), NULL),
      rule_double_neg },

    /* -- Division identities -- */
    /* x / x -> 1 (float) */
    { poly_pat_op2(POLY_OP_FDIV, poly_pat_any("x"),
        poly_pat_any("x"), NULL), rule_fdiv_self },
    /* (x * y) / y -> x */
    { poly_pat_op2(POLY_OP_FDIV,
        poly_pat_op2c(POLY_OP_MUL, poly_pat_any("x"), poly_pat_any("y"), NULL),
        poly_pat_any("y"), NULL), rule_mul_fdiv_cancel },

    /* -- Where folding -- */
    /* WHERE(a, WHERE(b, c, d), d) -> WHERE(AND(a, b), c, d) */
    { poly_pat_op3(POLY_OP_WHERE, poly_pat_any("a"),
        poly_pat_op3(POLY_OP_WHERE, poly_pat_any("b"),
          poly_pat_any("c"), poly_pat_any("d"), NULL),
        poly_pat_any("d"), NULL), rule_nested_where },
    /* WHERE(cond, val, val) -> val */
    { poly_pat_op3(POLY_OP_WHERE, poly_pat_any(NULL),
        poly_pat_any("val"), poly_pat_any("val"), NULL),
      rule_where_same },
    /* WHERE(const_gate, c0, c1) -> c0 or c1 */
    { poly_pat_op3(POLY_OP_WHERE, poly_pat_cvar("gate"),
        poly_pat_any("c0"), poly_pat_any("c1"), NULL),
      rule_where_const_gate },

    /* -- Combine terms -- */
    /* x + x -> x * 2 */
    { poly_pat_op2c(POLY_OP_ADD, poly_pat_any("x"),
        poly_pat_any("x"), NULL), rule_add_self },
    /* ADD(ADD(a, x), x) -> ADD(a, MUL(x, 2)) */
    { poly_pat_op2c(POLY_OP_ADD,
        poly_pat_op2c(POLY_OP_ADD, poly_pat_any("a"), poly_pat_any("x"), NULL),
        poly_pat_any("x"), NULL), rule_add_assoc_self },

    /* -- divmod cancel: (x%c) + (x//c)*c → x -- */
    { poly_pat_op2c(POLY_OP_ADD,
        poly_pat_op2(POLY_OP_MOD, poly_pat_any("x"), poly_pat_any("c"), NULL),
        poly_pat_op2(POLY_OP_MUL,
          poly_pat_op2(POLY_OP_IDIV, poly_pat_any("x"), poly_pat_any("c"), NULL),
          poly_pat_any("c"), NULL),
        NULL), rule_divmod_cancel },

    /* -- cancel_divmod: MOD/IDIV simplification via vmin/vmax -- */
    { poly_pat_ops2(poly_opset_add(poly_opset_add((PolyOpSet){{0,0}},
        POLY_OP_MOD), POLY_OP_IDIV),
        poly_pat_any(NULL), poly_pat_any(NULL), NULL), rule_cancel_divmod },
  };

  int n = sizeof(rules) / sizeof(rules[0]);
  g_symbolic_simple = poly_pm_new(rules, n);
  return g_symbolic_simple;
}
