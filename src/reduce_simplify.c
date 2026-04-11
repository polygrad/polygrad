/*
 * reduce_simplify.c -- Phase D port of tinygrad's pm_reduce_simplify.
 *
 * Tinygrad source: references/tinygrad_latest/tinygrad/codegen/simplify.py:73-149
 *
 * Sub-step status (PLAN_WEBGPU.md §7):
 *   D1 [done] : stub + wiring
 *   D2 [done] : pm_reduce_unparented        (rule_reduce_unparented)
 *   D3 [done] : 8 pm_reduce_collapse rules  (rule_collapse_*)
 *   D4 [done] : reduce_collapse driver      (reduce_collapse + entry rule)
 *   D5 [done] : pm_reduce_simplify combinator
 *
 * Patterns: tinygrad uses deeply nested UPat trees for the where-reduce
 * collapse rules. Polygrad's pat.h supports nested patterns but the rules
 * here are easier to express by matching the top-level op (REDUCE / CMPLT
 * / MUL) and inspecting the children manually inside each callback. This
 * is the same style fold_divmod_general (sym.c:692) uses.
 */

#include "reduce_simplify.h"

#include <stdio.h>
#include <stdlib.h>

#include "pat.h"
#include "polygrad.h"
#include "tensor.h"

/* helpers */

/* Test whether a UOp is one of the "external dependency" leaves that
 * reduce_collapse must NOT wrap in a DEFINE_VAR. Mirrors tinygrad's
 * exclusion list at simplify.py:136:
 *   {Ops.CONST, Ops.VCONST, Ops.PARAM, Ops.DEFINE_LOCAL, Ops.DEFINE_VAR}
 */
static bool is_external_leaf(PolyUOp *u) {
  switch (u->op) {
  case POLY_OP_CONST:
  case POLY_OP_VCONST:
  case POLY_OP_PARAM:
  case POLY_OP_DEFINE_LOCAL:
  case POLY_OP_DEFINE_VAR:
    return true;
  default:
    return false;
  }
}

/* Pointer-key map helpers (mirror src/uop.c:363-368). */
static bool ptr_eq_local(const void *a, const void *b) {
  return a == b;
}
static uint32_t ptr_hash_local(const void *p) {
  uintptr_t v = (uintptr_t)p;
  return (uint32_t)(v ^ (v >> 16) ^ (sizeof(v) > 4 ? (uint32_t)(v >> 32) : 0));
}

/* Construct a DEFINE_VAR with arbitrary dtype (frontend.c:poly_define_var
 * is INT32-only). Name is arena-copied by poly_uop_create. */
static PolyUOp *make_define_var(
    PolyCtx *ctx,
    const char *name,
    PolyDType dt,
    int64_t vmin,
    int64_t vmax
) {
  return poly_uop0(ctx, POLY_OP_DEFINE_VAR, dt, poly_arg_define_var(name, vmin, vmax));
}

/* CONST in a target dtype, value taken as a double and re-encoded into the
 * arg kind matching the dtype. Sole reason for existing: ensures we never
 * construct a CONST with mismatched dtype/arg.kind (the bug fixed in
 * pat.c:poly_const_like). */
static PolyUOp *typed_const(PolyCtx *ctx, PolyDType dt, double v) {
  if (poly_dtype_is_float(dt)) return poly_uop0(ctx, POLY_OP_CONST, dt, poly_arg_float(v));
  if (poly_dtype_is_bool(dt)) return poly_uop0(ctx, POLY_OP_CONST, dt, poly_arg_bool(v != 0.0));
  return poly_uop0(ctx, POLY_OP_CONST, dt, poly_arg_int((int64_t)v));
}

/* Cast that const-folds when src is a CONST in any kind, otherwise emits
 * a CAST UOp. Mirrors tinygrad's UOp.cast collapsing through symbolic. */
static PolyUOp *cast_to(PolyCtx *ctx, PolyUOp *x, PolyDType dt) {
  if (poly_dtype_eq(x->dtype, dt)) return x;
  if (x->op == POLY_OP_CONST) {
    double v = (x->arg.kind == POLY_ARG_INT)     ? (double)x->arg.i
               : (x->arg.kind == POLY_ARG_FLOAT) ? x->arg.f
               : (x->arg.kind == POLY_ARG_BOOL)  ? (x->arg.b ? 1.0 : 0.0)
                                                 : 0.0;
    return typed_const(ctx, dt, v);
  }
  return poly_cast(ctx, x, dt);
}

/* D2: reduce_unparented *
 * Tinygrad source (codegen/simplify.py:77-92):
 *
 *   def reduce_unparented(red):
 *     if red.arg not in {ADD, MAX, MUL}: return None
 *     parented, unparented = partition(red.src[1:], lambda x: x in red.src[0].ranges)
 *     if not unparented: return None
 *     ret = red.replace(src=(red.src[0],)+tuple(parented)) if parented or
 *           red.dtype != red.src[0].dtype else red.src[0]
 *     if red.arg is ADD: ret *= product(r.src[0].cast(...) for r in unparented)
 *     if red.arg is MUL: ret **= product(r.src[0].cast(...) for r in unparented)
 *     return ret
 *
 * Verified against tg_reduce_unparented_gt.py cases A-F.
 */
static PolyUOp *rule_reduce_unparented(PolyCtx *ctx, PolyUOp *red, const PolyBindings *b) {
  (void)b;
  if (red->arg.kind != POLY_ARG_OPS) return NULL;
  PolyOps rop = red->arg.ops;
  if (rop != POLY_OP_ADD && rop != POLY_OP_MAX && rop != POLY_OP_MUL) return NULL;
  if (red->n_src < 2) return NULL;
  for (uint16_t i = 1; i < red->n_src; i++)
    if (red->src[i]->op != POLY_OP_RANGE) return NULL;

  PolyUOp *value = red->src[0];
  PolyUOpCache *cache = poly_uop_cache_new();
  PolyUOp *parented[POLY_MAX_DIMS + 1];
  PolyUOp *unparented[POLY_MAX_DIMS + 1];
  int n_parented = 0, n_unparented = 0;
  for (uint16_t i = 1; i < red->n_src; i++) {
    PolyUOp *r = red->src[i];
    if (poly_uop_in_ranges_ex(ctx, value, r, cache))
      parented[n_parented++] = r;
    else
      unparented[n_unparented++] = r;
  }
  poly_uop_cache_destroy(cache);

  if (n_unparented == 0) return NULL;

  PolyUOp *ret;
  if (n_parented > 0 || !poly_dtype_eq(red->dtype, value->dtype)) {
    PolyUOp *new_srcs[POLY_MAX_DIMS + 2];
    new_srcs[0] = value;
    for (int i = 0; i < n_parented; i++)
      new_srcs[i + 1] = parented[i];
    ret = poly_uop(ctx, POLY_OP_REDUCE, red->dtype, new_srcs, 1 + n_parented, red->arg);
  } else {
    ret = value;
  }

  if (rop == POLY_OP_ADD || rop == POLY_OP_MUL) {
    PolyOps comb = (rop == POLY_OP_ADD) ? POLY_OP_MUL : POLY_OP_POW;
    for (int i = 0; i < n_unparented; i++) {
      PolyUOp *count = unparented[i]->src[0];
      ret = poly_alu2(ctx, comb, ret, cast_to(ctx, count, ret->dtype));
    }
  }
  /* MAX: drop unparented ranges with no multiplier */
  return ret;
}

/* D3: pm_reduce_collapse rules */

/* Rule 1: ((x+y).or_casted() < c) -> x < (c.cast(y.dtype)-y)  if no_range(y,c)
 * tinygrad simplify.py:96 */
static PolyUOp *rule_collapse_lift_add_from_cmplt(
    PolyCtx *ctx,
    PolyUOp *cmplt,
    const PolyBindings *b
) {
  (void)b;
  if (cmplt->op != POLY_OP_CMPLT || cmplt->n_src != 2) return NULL;
  PolyUOp *lhs = cmplt->src[0];
  PolyUOp *c = cmplt->src[1];
  /* or_casted: also accept CAST(ADD(x,y)) on lhs */
  if (lhs->op == POLY_OP_CAST && lhs->n_src == 1) lhs = lhs->src[0];
  if (lhs->op != POLY_OP_ADD || lhs->n_src != 2) return NULL;
  PolyUOp *x = lhs->src[0];
  PolyUOp *y = lhs->src[1];
  if (!poly_no_range(ctx, y) || !poly_no_range(ctx, c)) return NULL;
  /* x < (c.cast(y.dtype) - y) */
  PolyUOp *c_cast = cast_to(ctx, c, y->dtype);
  PolyUOp *rhs_new = poly_alu2(ctx, POLY_OP_SUB, c_cast, y);
  return poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, x, rhs_new, poly_arg_none());
}

/* Rule 2: ((x*y) < c) -> x < ((c+y-1)//y)  if no_range(y,c) and is_int(y) and y.vmin>0
 * tinygrad simplify.py:98-99 */
static PolyUOp *rule_collapse_lift_mul_from_cmplt(
    PolyCtx *ctx,
    PolyUOp *cmplt,
    const PolyBindings *b
) {
  (void)b;
  if (cmplt->op != POLY_OP_CMPLT || cmplt->n_src != 2) return NULL;
  PolyUOp *lhs = cmplt->src[0];
  PolyUOp *c = cmplt->src[1];
  if (lhs->op != POLY_OP_MUL || lhs->n_src != 2) return NULL;
  PolyUOp *x = lhs->src[0];
  PolyUOp *y = lhs->src[1];
  if (!poly_no_range(ctx, y) || !poly_no_range(ctx, c)) return NULL;
  if (!poly_dtype_is_int(y->dtype)) return NULL;
  int64_t y_vmin, y_vmax;
  poly_uop_minmax(ctx, y, &y_vmin, &y_vmax);
  if (y_vmin <= 0) return NULL;
  /* x < ((c + y - 1) // y) */
  PolyUOp *one = typed_const(ctx, c->dtype, 1);
  PolyUOp *cy1 = poly_alu2(ctx, POLY_OP_ADD, c, y);
  PolyUOp *cy1m1 = poly_alu2(ctx, POLY_OP_SUB, cy1, one);
  PolyUOp *div = poly_alu2(ctx, POLY_OP_IDIV, cy1m1, y);
  return poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, x, div, poly_arg_none());
}

/* Match patterns of the form REDUCE_ADD( WHERE( CMPLT(r, cut), tval, fval ), r ).
 * Used by rules 3 and 5 below. Returns true and fills out params on match. */
static bool match_where_cmplt_reduce(
    PolyUOp *red,
    PolyUOp **out_r,
    PolyUOp **out_cut,
    PolyUOp **out_tval,
    PolyUOp **out_fval
) {
  if (red->op != POLY_OP_REDUCE) return false;
  if (red->arg.kind != POLY_ARG_OPS || red->arg.ops != POLY_OP_ADD) return false;
  if (red->n_src != 2) return false; /* exactly one range */
  PolyUOp *r = red->src[1];
  if (r->op != POLY_OP_RANGE) return false;
  PolyUOp *value = red->src[0];
  if (value->op != POLY_OP_WHERE || value->n_src != 3) return false;
  PolyUOp *cmplt = value->src[0];
  if (cmplt->op != POLY_OP_CMPLT || cmplt->n_src != 2) return false;
  if (cmplt->src[0] != r) return false; /* CMPLT(r, cut) */
  *out_r = r;
  *out_cut = cmplt->src[1];
  *out_tval = value->src[1];
  *out_fval = value->src[2];
  return true;
}

/* Helper: build N.max(0).min(r.src[0]).cast(val.dtype) * val
 * where N is the integer expression "remaining count" the rule produces. */
static PolyUOp *build_count_mul_val(PolyCtx *ctx, PolyUOp *N, PolyUOp *r, PolyUOp *val) {
  /* N.maximum(0) -> MAX(N, 0) in N's dtype */
  PolyUOp *zero_n = typed_const(ctx, N->dtype, 0);
  PolyUOp *clamped_lo = poly_alu2(ctx, POLY_OP_MAX, N, zero_n);
  /* .minimum(r.src[0]) -> -MAX(-x, -count). Polygrad has no MIN op directly;
   * use ALU MIN if available. Check polygrad ops list — no POLY_OP_MIN. Use
   * the negation trick: MIN(a,b) = -MAX(-a,-b). */
  PolyUOp *count_in_n = cast_to(ctx, r->src[0], N->dtype);
  PolyUOp *neg_a = poly_alu1(ctx, POLY_OP_NEG, clamped_lo);
  PolyUOp *neg_b = poly_alu1(ctx, POLY_OP_NEG, count_in_n);
  PolyUOp *max_neg = poly_alu2(ctx, POLY_OP_MAX, neg_a, neg_b);
  PolyUOp *clamped_hi = poly_alu1(ctx, POLY_OP_NEG, max_neg);
  /* .cast(val.dtype) */
  PolyUOp *cast_n = cast_to(ctx, clamped_hi, val->dtype);
  /* * val */
  return poly_alu2(ctx, POLY_OP_MUL, cast_n, val);
}

/* Rule 3: fold_range_below
 *   ((r<cut).where(0, val)).reduce_add(r)
 *     -> (r.src[0]-cut).maximum(0).minimum(r.src[0]).cast(val.dtype) * val
 *  iff no_range(val).
 * tinygrad simplify.py:103 */
static PolyUOp *rule_collapse_fold_range_below(PolyCtx *ctx, PolyUOp *red, const PolyBindings *b) {
  (void)b;
  PolyUOp *r, *cut, *tval, *fval;
  if (!match_where_cmplt_reduce(red, &r, &cut, &tval, &fval)) return NULL;
  /* tval must be CONST(0); val = fval */
  if (tval->op != POLY_OP_CONST) return NULL;
  bool tval_is_zero = (tval->arg.kind == POLY_ARG_INT && tval->arg.i == 0) ||
                      (tval->arg.kind == POLY_ARG_FLOAT && tval->arg.f == 0.0) ||
                      (tval->arg.kind == POLY_ARG_BOOL && tval->arg.b == false);
  if (!tval_is_zero) return NULL;
  PolyUOp *val = fval;
  if (!poly_no_range(ctx, val)) return NULL;
  if (!poly_no_range(ctx, cut)) return NULL;
  /* N = r.src[0] - cut */
  PolyUOp *cut_in_r = cast_to(ctx, cut, r->src[0]->dtype);
  PolyUOp *N = poly_alu2(ctx, POLY_OP_SUB, r->src[0], cut_in_r);
  return build_count_mul_val(ctx, N, r, val);
}

/* Rule 5: fold_range_above
 *   ((r<cut).where(val, 0)).reduce_add(r)
 *     -> cut.maximum(0).minimum(r.src[0]).cast(val.dtype) * val
 *  iff no_range(val).
 * tinygrad simplify.py:109 */
static PolyUOp *rule_collapse_fold_range_above(PolyCtx *ctx, PolyUOp *red, const PolyBindings *b) {
  (void)b;
  PolyUOp *r, *cut, *tval, *fval;
  if (!match_where_cmplt_reduce(red, &r, &cut, &tval, &fval)) return NULL;
  /* fval must be CONST(0); val = tval */
  if (fval->op != POLY_OP_CONST) return NULL;
  bool fval_is_zero = (fval->arg.kind == POLY_ARG_INT && fval->arg.i == 0) ||
                      (fval->arg.kind == POLY_ARG_FLOAT && fval->arg.f == 0.0) ||
                      (fval->arg.kind == POLY_ARG_BOOL && fval->arg.b == false);
  if (!fval_is_zero) return NULL;
  PolyUOp *val = tval;
  if (!poly_no_range(ctx, val)) return NULL;
  if (!poly_no_range(ctx, cut)) return NULL;
  /* N = cut */
  PolyUOp *N = cast_to(ctx, cut, r->src[0]->dtype);
  return build_count_mul_val(ctx, N, r, val);
}

/* Rule 4: fold_range_two_sided
 *   (((r<lower).logical_not()) & (r<upper)).where(val,0).reduce_add(r)
 *     -> (upper.minimum(n) - lower.maximum(0)).maximum(0).minimum(n).cast(val.dtype) * val
 *  where n = r.src[0]. Polygrad's logical_not is CMPNE(x, true) (P5). The
 *  AND of two bool comparisons is POLY_OP_AND.
 * tinygrad simplify.py:105-107 */
static PolyUOp *rule_collapse_fold_range_two_sided(
    PolyCtx *ctx,
    PolyUOp *red,
    const PolyBindings *b
) {
  (void)b;
  if (red->op != POLY_OP_REDUCE) return NULL;
  if (red->arg.kind != POLY_ARG_OPS || red->arg.ops != POLY_OP_ADD) return NULL;
  if (red->n_src != 2) return NULL;
  PolyUOp *r = red->src[1];
  if (r->op != POLY_OP_RANGE) return NULL;
  PolyUOp *value = red->src[0];
  if (value->op != POLY_OP_WHERE || value->n_src != 3) return NULL;
  PolyUOp *fval = value->src[2];
  PolyUOp *val = value->src[1];
  if (fval->op != POLY_OP_CONST) return NULL;
  bool fval_is_zero = (fval->arg.kind == POLY_ARG_INT && fval->arg.i == 0) ||
                      (fval->arg.kind == POLY_ARG_FLOAT && fval->arg.f == 0.0);
  if (!fval_is_zero) return NULL;
  PolyUOp *cond = value->src[0];
  if (cond->op != POLY_OP_AND || cond->n_src != 2) return NULL;
  /* One of the AND operands is CMPNE(CMPLT(r, lower), CONST(true)),
   * the other is CMPLT(r, upper). Detect both. */
  PolyUOp *lo_form = cond->src[0];
  PolyUOp *hi_form = cond->src[1];
  PolyUOp *lower = NULL, *upper = NULL;
  for (int swap = 0; swap < 2 && (!lower || !upper); swap++) {
    PolyUOp *a = swap ? hi_form : lo_form;
    PolyUOp *b2 = swap ? lo_form : hi_form;
    /* a should be CMPNE(CMPLT(r, lower), CONST(true)) */
    if (a->op == POLY_OP_CMPNE && a->n_src == 2 && a->src[0]->op == POLY_OP_CMPLT &&
        a->src[0]->n_src == 2 && a->src[0]->src[0] == r && a->src[1]->op == POLY_OP_CONST &&
        ((a->src[1]->arg.kind == POLY_ARG_BOOL && a->src[1]->arg.b == true) ||
         (a->src[1]->arg.kind == POLY_ARG_INT && a->src[1]->arg.i != 0))) {
      PolyUOp *cand_lower = a->src[0]->src[1];
      /* b2 should be CMPLT(r, upper) */
      if (b2->op == POLY_OP_CMPLT && b2->n_src == 2 && b2->src[0] == r) {
        lower = cand_lower;
        upper = b2->src[1];
        break;
      }
    }
  }
  if (!lower || !upper) return NULL;
  if (!poly_no_range(ctx, val)) return NULL;
  if (!poly_no_range(ctx, lower) || !poly_no_range(ctx, upper)) return NULL;
  /* n = r.src[0] */
  PolyUOp *n = r->src[0];
  /* upper.minimum(n) = -MAX(-upper, -n) in n's dtype */
  PolyUOp *upper_in_n = cast_to(ctx, upper, n->dtype);
  PolyUOp *lower_in_n = cast_to(ctx, lower, n->dtype);
  PolyUOp *neg_u = poly_alu1(ctx, POLY_OP_NEG, upper_in_n);
  PolyUOp *neg_n = poly_alu1(ctx, POLY_OP_NEG, n);
  PolyUOp *upper_min_n = poly_alu1(ctx, POLY_OP_NEG, poly_alu2(ctx, POLY_OP_MAX, neg_u, neg_n));
  /* lower.maximum(0) = MAX(lower, 0) */
  PolyUOp *zero = typed_const(ctx, n->dtype, 0);
  PolyUOp *lower_max_0 = poly_alu2(ctx, POLY_OP_MAX, lower_in_n, zero);
  /* N = upper.min(n) - lower.max(0) */
  PolyUOp *N = poly_alu2(ctx, POLY_OP_SUB, upper_min_n, lower_max_0);
  return build_count_mul_val(ctx, N, r, val);
}

/* Rule 6: reduce_add_distribute
 *   (x+y).reduce_add(*ranges) -> x.reduce_add(*ranges) + y.reduce_add(*ranges)
 * tinygrad simplify.py:113 */
static PolyUOp *rule_collapse_reduce_add_distribute(
    PolyCtx *ctx,
    PolyUOp *red,
    const PolyBindings *b
) {
  (void)b;
  if (red->op != POLY_OP_REDUCE) return NULL;
  if (red->arg.kind != POLY_ARG_OPS || red->arg.ops != POLY_OP_ADD) return NULL;
  if (red->n_src < 2) return NULL;
  PolyUOp *value = red->src[0];
  if (value->op != POLY_OP_ADD || value->n_src != 2) return NULL;
  PolyUOp *x = value->src[0];
  PolyUOp *y = value->src[1];
  /* Build x.reduce_add(*ranges) and y.reduce_add(*ranges) */
  int n_extra = red->n_src - 1;
  PolyUOp *xs[POLY_MAX_DIMS + 2];
  PolyUOp *ys[POLY_MAX_DIMS + 2];
  xs[0] = x;
  ys[0] = y;
  for (int i = 0; i < n_extra; i++) {
    xs[1 + i] = red->src[1 + i];
    ys[1 + i] = red->src[1 + i];
  }
  PolyUOp *xred = poly_uop(ctx, POLY_OP_REDUCE, red->dtype, xs, 1 + n_extra, red->arg);
  PolyUOp *yred = poly_uop(ctx, POLY_OP_REDUCE, red->dtype, ys, 1 + n_extra, red->arg);
  return poly_alu2(ctx, POLY_OP_ADD, xred, yred);
}

/* Rule 7: and_on_where
 *   ((DEFINE_VAR & y).where(c, 0)).reduce_add(*ranges)
 *     -> y.where(c, 0).reduce_add(*ranges) * x.cast(c.dtype)
 * tinygrad simplify.py:115-116 */
static PolyUOp *rule_collapse_and_on_where(PolyCtx *ctx, PolyUOp *red, const PolyBindings *b) {
  (void)b;
  if (red->op != POLY_OP_REDUCE) return NULL;
  if (red->arg.kind != POLY_ARG_OPS || red->arg.ops != POLY_OP_ADD) return NULL;
  if (red->n_src < 2) return NULL;
  PolyUOp *where = red->src[0];
  if (where->op != POLY_OP_WHERE || where->n_src != 3) return NULL;
  PolyUOp *fval = where->src[2];
  if (fval->op != POLY_OP_CONST) return NULL;
  bool fval_is_zero = (fval->arg.kind == POLY_ARG_INT && fval->arg.i == 0) ||
                      (fval->arg.kind == POLY_ARG_FLOAT && fval->arg.f == 0.0);
  if (!fval_is_zero) return NULL;
  PolyUOp *and_op = where->src[0];
  if (and_op->op != POLY_OP_AND || and_op->n_src != 2) return NULL;
  /* One side must be DEFINE_VAR, the other is "y" */
  PolyUOp *x = NULL, *y = NULL;
  if (and_op->src[0]->op == POLY_OP_DEFINE_VAR) {
    x = and_op->src[0];
    y = and_op->src[1];
  } else if (and_op->src[1]->op == POLY_OP_DEFINE_VAR) {
    x = and_op->src[1];
    y = and_op->src[0];
  } else {
    return NULL;
  }
  PolyUOp *c = where->src[1];
  /* New WHERE: y.where(c, fval) */
  PolyUOp *new_where = poly_uop3(ctx, POLY_OP_WHERE, where->dtype, y, c, fval, poly_arg_none());
  /* New REDUCE with same ranges */
  int n_extra = red->n_src - 1;
  PolyUOp *new_srcs[POLY_MAX_DIMS + 2];
  new_srcs[0] = new_where;
  for (int i = 0; i < n_extra; i++)
    new_srcs[1 + i] = red->src[1 + i];
  PolyUOp *new_red = poly_uop(ctx, POLY_OP_REDUCE, red->dtype, new_srcs, 1 + n_extra, red->arg);
  /* * x.cast(c.dtype) */
  PolyUOp *x_cast = cast_to(ctx, x, c->dtype);
  return poly_alu2(ctx, POLY_OP_MUL, new_red, x_cast);
}

/* Rule 8: mul_casted_bool
 *   x * gate.cast()  ->  gate.where(x, 0)
 *  where gate.dtype is bool.
 * tinygrad simplify.py:118 */
static PolyUOp *rule_collapse_mul_casted_bool(PolyCtx *ctx, PolyUOp *mul, const PolyBindings *b) {
  (void)b;
  if (mul->op != POLY_OP_MUL || mul->n_src != 2) return NULL;
  /* Try both src orderings: src[0]=x, src[1]=CAST(gate); and the swap. */
  for (int swap = 0; swap < 2; swap++) {
    PolyUOp *x = mul->src[swap ? 1 : 0];
    PolyUOp *cast_node = mul->src[swap ? 0 : 1];
    if (cast_node->op != POLY_OP_CAST || cast_node->n_src != 1) continue;
    PolyUOp *gate = cast_node->src[0];
    if (!poly_dtype_is_bool(gate->dtype)) continue;
    /* gate.where(x, 0) — result dtype = x.dtype */
    PolyUOp *zero = typed_const(ctx, x->dtype, 0);
    return poly_uop3(ctx, POLY_OP_WHERE, x->dtype, gate, x, zero, poly_arg_none());
  }
  return NULL;
}

/* D4: reduce_collapse driver *
 * Tinygrad source (codegen/simplify.py:129-142):
 *
 *   def reduce_collapse(red, u, pm=pm_reduce_collapse):
 *     for r in red.src[1:]:
 *       included = u.toposort(gate=lambda x: r in x.ranges)
 *       if any(x.op in {STORE, REDUCE} for x in included): return None
 *       replaces = {}
 *       for u in included:
 *         for s in u.src:
 *           if s in included or s in replaces or s.op in
 *               {CONST, VCONST, PARAM, DEFINE_LOCAL, DEFINE_VAR}: continue
 *           replaces[s] = DEFINE_VAR(name=f'in{len(replaces)}', vmin, vmax)
 *       collapse_form = u.substitute(replaces).reduce(r, arg=ADD)
 *       sink = graph_rewrite(collapse_form, pm)
 *       if not no_range(sink): return None
 *       u = sink.substitute({v:k for k,v in replaces.items()})
 *     return u
 *
 * Polygrad uses one PolyUOpCache for the whole driver invocation; passes
 * include / replaces tracking via PolyMaps.
 */

static PolyPatternMatcher *pm_reduce_collapse_get(void);

/* The toposort gate needs to capture `r` and `cache`. We pass them via a
 * tiny struct stored in user_data. */
typedef struct {
  PolyCtx *ctx;
  PolyUOp *r;
  PolyUOpCache *cache;
} GateCtx;

static bool collapse_gate(PolyUOp *x, void *user_data) {
  GateCtx *g = (GateCtx *)user_data;
  return poly_uop_in_ranges_ex(g->ctx, x, g->r, g->cache);
}

static PolyUOp *reduce_collapse_drive(PolyCtx *ctx, PolyUOp *red, PolyUOp *u) {
  PolyUOpCache *cache = poly_uop_cache_new();
  PolyMap *included_map = NULL;
  PolyMap *replaces_map = NULL;
  bool dbg = getenv("POLY_DEBUG_REDUCE_SIMPLIFY") != NULL;

  /* Loop over each reduce range. */
  for (uint16_t ridx = 1; ridx < red->n_src; ridx++) {
    PolyUOp *r = red->src[ridx];
    if (r->op != POLY_OP_RANGE) goto fail;

    /* Toposort u, gated by "r in node.ranges" — yields "included" set. */
    GateCtx g = {.ctx = ctx, .r = r, .cache = cache};
    int n_inc = 0;
    PolyUOp **included = poly_toposort_ex_user(ctx, u, &n_inc, collapse_gate, &g, true);
    /* `included` is arena-allocated by toposort; do NOT free. */
    if (dbg) {
      fprintf(stderr, "  [reduce_collapse] r=%p included.n=%d value tree:\n", (void *)r, n_inc);
      poly_uop_dump_tree(stderr, u, 4, 12);
    }
    if (!included || n_inc == 0) goto fail;

    if (included_map) poly_map_destroy(included_map);
    included_map = poly_map_new(16);
    for (int i = 0; i < n_inc; i++) {
      poly_map_set(
          included_map, ptr_hash_local(included[i]), included[i], (void *)(uintptr_t)1, ptr_eq_local
      );
    }

    /* Bail if any included node is STORE or REDUCE (a nested reduce). */
    bool nested = false;
    for (int i = 0; i < n_inc; i++) {
      PolyOps op = included[i]->op;
      if (op == POLY_OP_STORE || op == POLY_OP_REDUCE) {
        nested = true;
        break;
      }
    }
    if (nested) goto fail;

    /* Build replaces: for each included node's external src that is not
     * itself included, not already replaced, and not an excluded leaf,
     * mint a fresh DEFINE_VAR carrying its current vmin/vmax. */
    if (replaces_map) poly_map_destroy(replaces_map);
    replaces_map = poly_map_new(16);
    PolyUOp *from_arr[256];
    PolyUOp *to_arr[256];
    int n_repl = 0;
    char namebuf[16];
    for (int i = 0; i < n_inc; i++) {
      PolyUOp *node = included[i];
      for (uint16_t k = 0; k < node->n_src; k++) {
        PolyUOp *s = node->src[k];
        if (poly_map_get(included_map, ptr_hash_local(s), s, ptr_eq_local)) continue;
        if (poly_map_get(replaces_map, ptr_hash_local(s), s, ptr_eq_local)) continue;
        if (is_external_leaf(s)) continue;
        int64_t vmin, vmax;
        poly_uop_minmax_ex(ctx, s, cache, &vmin, &vmax);
        snprintf(namebuf, sizeof(namebuf), "in%d", n_repl);
        PolyUOp *dv = make_define_var(ctx, namebuf, s->dtype, vmin, vmax);
        if (n_repl >= 256) goto fail;
        from_arr[n_repl] = s;
        to_arr[n_repl] = dv;
        poly_map_set(replaces_map, ptr_hash_local(s), s, dv, ptr_eq_local);
        n_repl++;
      }
    }

    /* Substitute, build collapse form, run pm_reduce_collapse, check
     * no_range, substitute back. */
    PolyUOp *substituted = poly_uop_substitute(ctx, u, from_arr, to_arr, n_repl);
    PolyUOp *one_range_srcs[2] = {substituted, r};
    PolyUOp *collapse_form =
        poly_uop(ctx, POLY_OP_REDUCE, red->dtype, one_range_srcs, 2, poly_arg_ops(POLY_OP_ADD));
    PolyUOp *sink = poly_graph_rewrite(ctx, collapse_form, pm_reduce_collapse_get());
    if (dbg)
      fprintf(
          stderr, "  [reduce_collapse] n_repl=%d sink_op=%s no_range=%d\n", n_repl,
          poly_op_name(sink->op), (int)poly_no_range_ex(ctx, sink, cache)
      );
    if (!poly_no_range_ex(ctx, sink, cache)) goto fail;
    /* Substitute back: from = to_arr (DEFINE_VARs), to = from_arr (originals) */
    u = poly_uop_substitute(ctx, sink, to_arr, from_arr, n_repl);
  }

  if (included_map) poly_map_destroy(included_map);
  if (replaces_map) poly_map_destroy(replaces_map);
  poly_uop_cache_destroy(cache);
  return u;

fail:
  if (included_map) poly_map_destroy(included_map);
  if (replaces_map) poly_map_destroy(replaces_map);
  poly_uop_cache_destroy(cache);
  return NULL;
}

/* Entry rule for pm_reduce_simplify (simplify.py:147-149):
 *   (UPat(REDUCE, src=(UPat.var("u"),), allow_any_len=True, arg=Ops.ADD,
 *       name="red"), reduce_collapse)
 */
static PolyUOp *rule_reduce_simplify_entry(PolyCtx *ctx, PolyUOp *red, const PolyBindings *b) {
  (void)b;
  if (red->op != POLY_OP_REDUCE) return NULL;
  if (red->arg.kind != POLY_ARG_OPS || red->arg.ops != POLY_OP_ADD) return NULL;
  if (red->n_src < 2) return NULL;
  return reduce_collapse_drive(ctx, red, red->src[0]);
}

/* D5: pm_reduce_simplify combinator *
 * Tinygrad source (codegen/simplify.py:147-149):
 *   pm_reduce_simplify = pm_reduce_unparented + PatternMatcher([
 *     (UPat(Ops.REDUCE, src=(UPat.var("u"),), allow_any_len=True,
 *           arg=Ops.ADD, name="red"), reduce_collapse),
 *   ])
 *
 * Tinygrad runs pm_reduce_simplify fused with `symbolic + pm_const_buffer
 * _folding + pm_remove_bufferize` in one graph_rewrite at rangeify.py:579.
 * Polygrad concatenates pm_reduce_unparented + entry rule + symbolic_simple
 * for that fusion. pm_remove_bufferize runs separately as a polygrad-side
 * stage before this pass — see src/rangeify.c.
 */
static PolyPatternMatcher *g_pm_reduce_unparented = NULL;
static PolyPatternMatcher *g_pm_reduce_collapse = NULL;
static PolyPatternMatcher *g_pm_reduce_simplify = NULL;

static PolyPatternMatcher *pm_reduce_unparented_get(void) {
  if (g_pm_reduce_unparented) return g_pm_reduce_unparented;
  PolyRule rules[] = {
      {poly_pat_op(POLY_OP_REDUCE, NULL, 0, "red"), rule_reduce_unparented},
  };
  g_pm_reduce_unparented = poly_pm_new(rules, (int)(sizeof(rules) / sizeof(rules[0])));
  return g_pm_reduce_unparented;
}

static PolyPatternMatcher *pm_reduce_collapse_get(void) {
  if (g_pm_reduce_collapse) return g_pm_reduce_collapse;
  PolyRule rules[] = {
      {poly_pat_op(POLY_OP_REDUCE, NULL, 0, "red"), rule_reduce_unparented},
      {poly_pat_op(POLY_OP_CMPLT, NULL, 0, "x"), rule_collapse_lift_add_from_cmplt},
      {poly_pat_op(POLY_OP_CMPLT, NULL, 0, "x"), rule_collapse_lift_mul_from_cmplt},
      {poly_pat_op(POLY_OP_REDUCE, NULL, 0, "red"), rule_collapse_fold_range_below},
      {poly_pat_op(POLY_OP_REDUCE, NULL, 0, "red"), rule_collapse_fold_range_two_sided},
      {poly_pat_op(POLY_OP_REDUCE, NULL, 0, "red"), rule_collapse_fold_range_above},
      {poly_pat_op(POLY_OP_REDUCE, NULL, 0, "red"), rule_collapse_reduce_add_distribute},
      {poly_pat_op(POLY_OP_REDUCE, NULL, 0, "red"), rule_collapse_and_on_where},
      {poly_pat_op(POLY_OP_MUL, NULL, 0, "mul"), rule_collapse_mul_casted_bool},
  };
  PolyPatternMatcher *base = poly_pm_new(rules, (int)(sizeof(rules) / sizeof(rules[0])));
  g_pm_reduce_collapse = poly_pm_concat(base, poly_symbolic_simple());
  return g_pm_reduce_collapse;
}

static PolyPatternMatcher *pm_reduce_simplify_get(void) {
  if (g_pm_reduce_simplify) return g_pm_reduce_simplify;
  PolyRule rules[] = {
      {poly_pat_op(POLY_OP_REDUCE, NULL, 0, "red"), rule_reduce_unparented},
      {poly_pat_op(POLY_OP_REDUCE, NULL, 0, "red"), rule_reduce_simplify_entry},
  };
  PolyPatternMatcher *base = poly_pm_new(rules, (int)(sizeof(rules) / sizeof(rules[0])));
  g_pm_reduce_simplify = poly_pm_concat(base, poly_symbolic_simple());
  return g_pm_reduce_simplify;
}

/* Public entries */

PolyUOp *poly_apply_reduce_unparented_only(PolyCtx *ctx, PolyUOp *sink) {
  /* Test-only entry: standalone pm_reduce_unparented (no symbolic concat).
   * Mirrors test/parity_scripts/tg_reduce_unparented_gt.py ground truth. */
  return poly_graph_rewrite(ctx, sink, pm_reduce_unparented_get());
}

PolyUOp *poly_apply_reduce_simplify(PolyCtx *ctx, PolyUOp *sink) {
  if (getenv("POLY_DISABLE_REDUCE_SIMPLIFY")) return sink;
  return poly_graph_rewrite(ctx, sink, pm_reduce_simplify_get());
}
