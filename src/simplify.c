/*
 * simplify.c -- port of tinygrad/codegen/simplify.py
 */

#include "simplify.h"

#include <stdio.h>
#include <stdlib.h>

#include "pat.h"
#include "polygrad.h"
#include "tensor.h"
#include "utils.h"

/* simplify.py: flatten_range */

int range_start_for_op(PolyOps op) {
  switch (op) {
  case POLY_OP_BUFFERIZE:
    return 1;
  case POLY_OP_REDUCE:
    return 1;
  case POLY_OP_STORE:
    return 2;
  case POLY_OP_WMMA:
    return 3;
  case POLY_OP_END:
    return 1;
  case POLY_OP_CALL:
    return 1;
  case POLY_OP_COPY:
    return 2;
  case POLY_OP_BUFFER_VIEW:
    return 1;
  default:
    return -1;
  }
}

static int collect_unique_ranges(
    PolyCtx *ctx,
    PolyUOp **rngs,
    int n_rngs,
    PolyUOp **out,
    int max_out
) {
  if (!rngs || n_rngs <= 0 || max_out <= 0) return 0;
  PolyUOp *tmp_sink = poly_uop(ctx, POLY_OP_SINK, POLY_VOID, rngs, n_rngs, poly_arg_none());
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, tmp_sink, &n_topo);
  int n_out = 0;
  for (int i = 0; i < n_topo && n_out < max_out; i++) {
    if (topo[i]->op != POLY_OP_RANGE) continue;
    bool dup = false;
    for (int j = 0; j < n_out; j++)
      if (out[j] == topo[i]) {
        dup = true;
        break;
      }
    if (!dup) out[n_out++] = topo[i];
  }
  return n_out;
}

static PolyUOp *flatten_range(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  int off = range_start_for_op(root->op);
  if (off < 0 || root->n_src <= off) return NULL;
  int n_rngs = root->n_src - off;
  if (n_rngs <= 0) return NULL;

  PolyUOp *flat_rngs[POLY_MAX_DIMS];
  int n_flat = collect_unique_ranges(ctx, &root->src[off], n_rngs, flat_rngs, POLY_MAX_DIMS);
  if (n_flat <= 0) return NULL;

  bool same = (n_flat == n_rngs);
  if (same) {
    for (int i = 0; i < n_flat; i++) {
      if (flat_rngs[i] != root->src[off + i]) {
        same = false;
        break;
      }
    }
  }
  if (same) return NULL;

  PolyUOp *new_src[POLY_MAX_DIMS + 8];
  int n_new = 0;
  for (int i = 0; i < off; i++)
    new_src[n_new++] = root->src[i];
  for (int i = 0; i < n_flat; i++)
    new_src[n_new++] = flat_rngs[i];
  return (root->tag != 0)
             ? poly_uop_tagged(ctx, root->op, root->dtype, new_src, n_new, root->arg, root->tag)
             : poly_uop(ctx, root->op, root->dtype, new_src, n_new, root->arg);
}

static PolyPatternMatcher *g_pm_flatten_range = NULL;
PolyPatternMatcher *poly_pm_flatten_range(void) {
  if (g_pm_flatten_range) return g_pm_flatten_range;
  PolyOpSet ops = {{0, 0}};
  ops = poly_opset_add(ops, POLY_OP_REDUCE);
  ops = poly_opset_add(ops, POLY_OP_STORE);
  ops = poly_opset_add(ops, POLY_OP_END);
  PolyRule rules[] = {
      {poly_pat_allow_any_len(poly_pat_ops(ops, NULL, 0, NULL)), flatten_range},
  };
  g_pm_flatten_range = poly_pm_new(rules, (int)(sizeof(rules) / sizeof(rules[0])));
  return g_pm_flatten_range;
}

/* simplify.py: count_divmod / simplify_merge_adjacent / pm_simplify_ranges */

static int count_divmod(PolyCtx *ctx, PolyUOp *u) {
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, u, &n_topo);
  int n = 0;
  for (int i = 0; i < n_topo; i++) {
    if (topo[i]->op == POLY_OP_IDIV || topo[i]->op == POLY_OP_MOD) n++;
  }
  return n;
}

static bool is_const_bound_range(PolyUOp *r, int64_t *bound) {
  if (!r || r->op != POLY_OP_RANGE || r->n_src <= 0 || r->src[0]->op != POLY_OP_CONST) return false;
  if (r->src[0]->arg.kind != POLY_ARG_INT) return false;
  if (bound) *bound = r->src[0]->arg.i;
  return true;
}

static PolyUOp *try_merge_two_ranges(PolyCtx *ctx, PolyUOp *root, PolyUOp *r0, PolyUOp *r1) {
  int64_t s0, s1;
  if (!is_const_bound_range(r0, &s0) || !is_const_bound_range(r1, &s1)) return NULL;
  if (poly_range_axis_type(r0->arg) != poly_range_axis_type(r1->arg)) return NULL;
  if (s0 <= 0 || s1 <= 0) return NULL;

  PolyUOp *prod = poly_uop0(ctx, POLY_OP_CONST, r0->dtype, poly_arg_int(s0 * s1));
  PolyUOp *new_range = poly_uop1(ctx, POLY_OP_RANGE, r0->dtype, prod, r0->arg);
  PolyUOp *s1c = poly_uop0(ctx, POLY_OP_CONST, r0->dtype, poly_arg_int(s1));
  PolyUOp *sub0 = poly_uop2(ctx, POLY_OP_IDIV, r0->dtype, new_range, s1c, poly_arg_none());
  PolyUOp *sub1 = poly_uop2(ctx, POLY_OP_MOD, r1->dtype, new_range, s1c, poly_arg_none());
  PolyUOp *from[2] = {r0, r1};
  PolyUOp *to[2] = {sub0, sub1};
  PolyUOp *cand = poly_uop_substitute(ctx, root, from, to, 2);
  cand = poly_graph_rewrite(ctx, cand, poly_symbolic());
  cand = poly_graph_rewrite(ctx, cand, poly_pm_flatten_range());
  return cand;
}

static PolyUOp *simplify_merge_adjacent(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  int off = range_start_for_op(root->op);
  if (off < 0 || root->n_src <= off + 1) return NULL;

  PolyUOp *best = root;
  int best_cost = count_divmod(ctx, root);
  int n_rng = root->n_src - off;

  for (int i = 0; i + 1 < n_rng; i++) {
    PolyUOp *r0 = root->src[off + i];
    PolyUOp *r1 = root->src[off + i + 1];
    if (r0->op != POLY_OP_RANGE || r1->op != POLY_OP_RANGE) continue;
    PolyUOp *cand = try_merge_two_ranges(ctx, best, r0, r1);
    if (!cand) continue;
    int c = count_divmod(ctx, cand);
    if (c <= best_cost) {
      best = cand;
      best_cost = c;
    }
  }
  return (best != root) ? best : NULL;
}

static PolyPatternMatcher *g_pm_simplify_ranges = NULL;
PolyPatternMatcher *poly_pm_simplify_ranges(void) {
  if (g_pm_simplify_ranges) return g_pm_simplify_ranges;
  PolyOpSet ops = {{0, 0}};
  ops = poly_opset_add(ops, POLY_OP_END);
  ops = poly_opset_add(ops, POLY_OP_REDUCE);
  PolyRule rules[] = {
      {poly_pat_allow_any_len(poly_pat_ops(ops, NULL, 0, NULL)), simplify_merge_adjacent},
  };
  g_pm_simplify_ranges = poly_pm_new(rules, (int)(sizeof(rules) / sizeof(rules[0])));
  return g_pm_simplify_ranges;
}

/* simplify.py: mark_range_mod / do_substitute / pm_split_ranges */

static SplitRangeCtx *current_split_ctx(void) {
  return (SplitRangeCtx *)poly_graph_rewrite_userctx();
}

static PolyUOp *mark_range_mod(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)ctx;
  PolyUOp *r = poly_bind(b, "r");
  PolyUOp *c = poly_bind(b, "c");
  if (!r || !c) return NULL;
  if (r->op != POLY_OP_RANGE || c->op != POLY_OP_CONST || c->arg.kind != POLY_ARG_INT) return NULL;
  if (!(r->n_src > 0 && r->src[0]->op == POLY_OP_CONST && r->src[0]->arg.kind == POLY_ARG_INT))
    return NULL;
  int64_t rv = r->src[0]->arg.i, cv = c->arg.i;
  if (cv <= 1 || rv <= 0) return NULL;
  if ((rv % cv) != 0) return NULL;

  SplitRangeCtx *sctx = current_split_ctx();
  if (!sctx) return NULL;
  for (int i = 0; i < sctx->n; i++)
    if (sctx->r[i] == r) return NULL;
  if (sctx->n < POLY_MAX_DIMS) {
    sctx->r[sctx->n] = r;
    sctx->c[sctx->n] = c;
    sctx->n++;
  }
  return NULL;
}

static PolyUOp *do_substitute(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  SplitRangeCtx *sctx = current_split_ctx();
  if (!sctx || sctx->n <= 0) return NULL;

  PolyUOp *from[POLY_MAX_DIMS];
  PolyUOp *to[POLY_MAX_DIMS];
  int n_sub = 0;

  for (int i = 0; i < sctx->n && n_sub < POLY_MAX_DIMS; i++) {
    PolyUOp *r = sctx->r[i], *v = sctx->c[i];
    if (!r || !v) continue;
    int n_extra = poly_range_n_extra(r->arg);
    if (n_extra + 1 > POLY_MAX_DIMS) n_extra = POLY_MAX_DIMS - 1;
    int64_t extra0[POLY_MAX_DIMS], extra1[POLY_MAX_DIMS];
    const int64_t *src_extra = poly_range_extra(r->arg);
    for (int j = 0; j < n_extra; j++) {
      extra0[j] = src_extra[j];
      extra1[j] = src_extra[j];
    }
    extra0[n_extra] = 0;
    extra1[n_extra] = 1;
    PolyArg k0_arg = poly_arg_range_ex(
        poly_range_axis_id(r->arg), poly_range_axis_type(r->arg), extra0, n_extra + 1
    );
    PolyArg k1_arg = poly_arg_range_ex(
        poly_range_axis_id(r->arg), poly_range_axis_type(r->arg), extra1, n_extra + 1
    );
    PolyUOp *k0_bound = poly_uop2(ctx, POLY_OP_IDIV, r->dtype, r->src[0], v, poly_arg_none());
    PolyUOp *k0 = poly_uop1(ctx, POLY_OP_RANGE, r->dtype, k0_bound, k0_arg);
    PolyUOp *k1 = poly_uop1(ctx, POLY_OP_RANGE, r->dtype, v, k1_arg);
    PolyUOp *k0_mul = poly_uop2(ctx, POLY_OP_MUL, r->dtype, k0, v, poly_arg_none());
    PolyUOp *expr = poly_uop2(ctx, POLY_OP_ADD, r->dtype, k0_mul, k1, poly_arg_none());
    from[n_sub] = r;
    to[n_sub] = expr;
    n_sub++;
  }
  sctx->n = 0;
  if (n_sub <= 0) return NULL;

  PolyUOp *out = poly_uop_substitute(ctx, root, from, to, n_sub);
  out = poly_graph_rewrite(ctx, out, poly_symbolic());
  out = poly_graph_rewrite(ctx, out, poly_pm_flatten_range());
  return out;
}

static PolyPatternMatcher *g_pm_split_ranges = NULL;
PolyPatternMatcher *poly_pm_split_ranges(void) {
  if (g_pm_split_ranges) return g_pm_split_ranges;
  PolyRule rules[] = {
      {poly_pat_op2(POLY_OP_MOD, poly_pat_any("r"), poly_pat_cvar("c"), NULL), mark_range_mod},
      {poly_pat_op(POLY_OP_SINK, NULL, 0, NULL), do_substitute},
  };
  g_pm_split_ranges = poly_pm_new(rules, (int)(sizeof(rules) / sizeof(rules[0])));
  return g_pm_split_ranges;
}

/* C helpers shared by reduce/load collapse ports. */

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
static PolyUOp *reduce_unparented(PolyCtx *ctx, PolyUOp *red, const PolyBindings *b) {
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

static PolyUOp *mul_neg_one(PolyCtx *ctx, PolyUOp *x);

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
  if (!poly_no_range(ctx, c)) return NULL;
  for (int swap = 0; swap < 2; swap++) {
    PolyUOp *x = lhs->src[swap];
    PolyUOp *y = lhs->src[swap ^ 1];
    /* UPat builds ADD as commutative and tries both source permutations.
     * Mirror that here so the range-bearing term can be either side. */
    if (!poly_no_range(ctx, y)) continue;
    /* tinygrad sub() lowers to add(-y), not a dedicated SUB op. */
    PolyUOp *c_cast = cast_to(ctx, c, y->dtype);
    PolyUOp *rhs_new = poly_alu2(ctx, POLY_OP_ADD, c_cast, mul_neg_one(ctx, y));
    return poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, x, rhs_new, poly_arg_none());
  }
  return NULL;
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
  if (!poly_no_range(ctx, c)) return NULL;
  for (int swap = 0; swap < 2; swap++) {
    PolyUOp *x = lhs->src[swap];
    PolyUOp *y = lhs->src[swap ^ 1];
    /* Tinygrad's commutative UPat tries both x/y bindings for MUL. */
    if (!poly_no_range(ctx, y)) continue;
    if (!poly_dtype_is_int(y->dtype)) continue;
    int64_t y_vmin, y_vmax;
    poly_uop_minmax(ctx, y, &y_vmin, &y_vmax);
    if (y_vmin <= 0) continue;
    /* x < ((c + y - 1) // y) */
    PolyUOp *one = typed_const(ctx, c->dtype, 1);
    PolyUOp *cy1 = poly_alu2(ctx, POLY_OP_ADD, c, y);
    PolyUOp *cy1m1 = poly_alu2(ctx, POLY_OP_SUB, cy1, one);
    PolyUOp *div = poly_alu2(ctx, POLY_OP_IDIV, cy1m1, y);
    return poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, x, div, poly_arg_none());
  }
  return NULL;
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

/* tinygrad builds minimum via _inverse().maximum(...)._inverse().
 * For signed ints/floats, _inverse is x * (-1), not a unary NEG op. */
static PolyUOp *mul_neg_one(PolyCtx *ctx, PolyUOp *x) {
  return poly_alu2(ctx, POLY_OP_MUL, x, typed_const(ctx, x->dtype, -1));
}

/* Helper: build N.max(0).min(r.src[0]).cast(val.dtype) * val
 * where N is the integer expression "remaining count" the rule produces. */
static PolyUOp *build_count_mul_val(PolyCtx *ctx, PolyUOp *N, PolyUOp *r, PolyUOp *val) {
  /* N.maximum(0) -> MAX(N, 0) in N's dtype */
  PolyUOp *zero_n = typed_const(ctx, N->dtype, 0);
  PolyUOp *clamped_lo = poly_alu2(ctx, POLY_OP_MAX, N, zero_n);
  /* .minimum(r.src[0]) -> (-x).maximum(-count) * (-1). */
  PolyUOp *count_in_n = cast_to(ctx, r->src[0], N->dtype);
  PolyUOp *neg_a = mul_neg_one(ctx, clamped_lo);
  PolyUOp *neg_b = mul_neg_one(ctx, count_in_n);
  PolyUOp *max_neg = poly_alu2(ctx, POLY_OP_MAX, neg_a, neg_b);
  PolyUOp *clamped_hi = mul_neg_one(ctx, max_neg);
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
  /* N = r.src[0] + (-cut) for tinygrad algebra parity. */
  PolyUOp *cut_in_r = cast_to(ctx, cut, r->src[0]->dtype);
  PolyUOp *N = poly_alu2(ctx, POLY_OP_ADD, r->src[0], mul_neg_one(ctx, cut_in_r));
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
  /* n = r.src[0] */
  PolyUOp *n = r->src[0];
  /* upper.minimum(n) = (-upper).maximum(-n) * (-1) in n's dtype */
  PolyUOp *upper_in_n = cast_to(ctx, upper, n->dtype);
  PolyUOp *lower_in_n = cast_to(ctx, lower, n->dtype);
  PolyUOp *neg_u = mul_neg_one(ctx, upper_in_n);
  PolyUOp *neg_n = mul_neg_one(ctx, n);
  PolyUOp *upper_min_n = mul_neg_one(ctx, poly_alu2(ctx, POLY_OP_MAX, neg_u, neg_n));
  /* lower.maximum(0) = MAX(lower, 0) */
  PolyUOp *zero = typed_const(ctx, n->dtype, 0);
  PolyUOp *lower_max_0 = poly_alu2(ctx, POLY_OP_MAX, lower_in_n, zero);
  /* N = upper.min(n) + (-lower.max(0)) for tinygrad algebra parity. */
  PolyUOp *N = poly_alu2(ctx, POLY_OP_ADD, upper_min_n, mul_neg_one(ctx, lower_max_0));
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
static PolyPatternMatcher *pm_reduce_load_collapse_get(void);

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

static PolyUOp *reduce_collapse(
    PolyCtx *ctx,
    PolyUOp *red,
    PolyUOp *u,
    PolyPatternMatcher *pm
) {
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
          included_map, poly_ptr_hash(included[i]), included[i], (void *)(uintptr_t)1, poly_ptr_eq
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
        if (poly_map_get(included_map, poly_ptr_hash(s), s, poly_ptr_eq)) continue;
        if (poly_map_get(replaces_map, poly_ptr_hash(s), s, poly_ptr_eq)) continue;
        if (is_external_leaf(s)) continue;
        int64_t vmin, vmax;
        poly_uop_minmax_ex(ctx, s, cache, &vmin, &vmax);
        snprintf(namebuf, sizeof(namebuf), "in%d", n_repl);
        PolyUOp *dv = make_define_var(ctx, namebuf, s->dtype, vmin, vmax);
        if (n_repl >= 256) goto fail;
        from_arr[n_repl] = s;
        to_arr[n_repl] = dv;
        poly_map_set(replaces_map, poly_ptr_hash(s), s, dv, poly_ptr_eq);
        n_repl++;
      }
    }

    /* Substitute, build collapse form, run pm_reduce_collapse, check
     * no_range, substitute back. */
    PolyUOp *substituted = poly_uop_substitute(ctx, u, from_arr, to_arr, n_repl);
    PolyUOp *one_range_srcs[2] = {substituted, r};
    PolyUOp *collapse_form =
        poly_uop(ctx, POLY_OP_REDUCE, red->dtype, one_range_srcs, 2, poly_arg_ops(POLY_OP_ADD));
    PolyUOp *sink = poly_graph_rewrite(ctx, collapse_form, pm);
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
static PolyUOp *reduce_simplify(PolyCtx *ctx, PolyUOp *red, const PolyBindings *b) {
  (void)b;
  if (red->op != POLY_OP_REDUCE) return NULL;
  if (red->arg.kind != POLY_ARG_OPS || red->arg.ops != POLY_OP_ADD) return NULL;
  if (red->n_src < 2) return NULL;
  return reduce_collapse(ctx, red, red->src[0], pm_reduce_collapse_get());
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
 * Polygrad concatenates pm_reduce_unparented + entry rule + symbolic
 * for that fusion. pm_remove_bufferize runs separately as a polygrad-side
 * stage before this pass — see src/rangeify.c.
 */
static PolyPatternMatcher *g_pm_reduce_unparented = NULL;
static PolyPatternMatcher *g_pm_reduce_collapse_base = NULL;
static PolyPatternMatcher *g_pm_reduce_collapse = NULL;
static PolyPatternMatcher *g_pm_reduce_load_collapse = NULL;
static PolyPatternMatcher *g_pm_reduce_simplify_base = NULL;
static PolyPatternMatcher *g_pm_reduce_simplify = NULL;
static PolyPatternMatcher *g_pm_symbolic_reduce_simplify = NULL;

static PolyPatternMatcher *pm_reduce_unparented_get(void) {
  if (g_pm_reduce_unparented) return g_pm_reduce_unparented;
  PolyRule rules[] = {
      {poly_pat_op(POLY_OP_REDUCE, NULL, 0, "red"), reduce_unparented},
  };
  g_pm_reduce_unparented = poly_pm_new(rules, (int)(sizeof(rules) / sizeof(rules[0])));
  return g_pm_reduce_unparented;
}

static PolyPatternMatcher *pm_reduce_collapse_base_get(void) {
  if (g_pm_reduce_collapse_base) return g_pm_reduce_collapse_base;
  PolyRule rules[] = {
      {poly_pat_op(POLY_OP_REDUCE, NULL, 0, "red"), reduce_unparented},
      {poly_pat_op(POLY_OP_CMPLT, NULL, 0, "x"), rule_collapse_lift_add_from_cmplt},
      {poly_pat_op(POLY_OP_CMPLT, NULL, 0, "x"), rule_collapse_lift_mul_from_cmplt},
      {poly_pat_op(POLY_OP_REDUCE, NULL, 0, "red"), rule_collapse_fold_range_below},
      {poly_pat_op(POLY_OP_REDUCE, NULL, 0, "red"), rule_collapse_fold_range_two_sided},
      {poly_pat_op(POLY_OP_REDUCE, NULL, 0, "red"), rule_collapse_fold_range_above},
      {poly_pat_op(POLY_OP_REDUCE, NULL, 0, "red"), rule_collapse_reduce_add_distribute},
      {poly_pat_op(POLY_OP_REDUCE, NULL, 0, "red"), rule_collapse_and_on_where},
      {poly_pat_op(POLY_OP_MUL, NULL, 0, "mul"), rule_collapse_mul_casted_bool},
  };
  g_pm_reduce_collapse_base = poly_pm_new(rules, (int)(sizeof(rules) / sizeof(rules[0])));
  return g_pm_reduce_collapse_base;
}

static PolyPatternMatcher *pm_reduce_collapse_get(void) {
  if (g_pm_reduce_collapse) return g_pm_reduce_collapse;
  g_pm_reduce_collapse = poly_pm_concat(pm_reduce_collapse_base_get(), poly_symbolic());
  return g_pm_reduce_collapse;
}

/* simplify.py: pm_reduce_load_collapse */

static bool is_zero_const(PolyUOp *u) {
  if (!u || u->op != POLY_OP_CONST) return false;
  return (u->arg.kind == POLY_ARG_INT && u->arg.i == 0) ||
         (u->arg.kind == POLY_ARG_FLOAT && u->arg.f == 0.0) ||
         (u->arg.kind == POLY_ARG_BOOL && u->arg.b == false);
}

static bool is_range_or_cast_of_range(PolyUOp *u, PolyUOp *r) {
  if (u == r) return true;
  return u && u->op == POLY_OP_CAST && u->n_src == 1 && u->src[0] == r;
}

static PolyUOp *rule_lift_add_from_cmpne(PolyCtx *ctx, PolyUOp *cmpne, const PolyBindings *b) {
  (void)b;
  if (!cmpne || cmpne->op != POLY_OP_CMPNE || cmpne->n_src != 2) return NULL;
  PolyUOp *lhs = cmpne->src[0];
  PolyUOp *c = cmpne->src[1];
  if (lhs->op == POLY_OP_CAST && lhs->n_src == 1) lhs = lhs->src[0];
  if (lhs->op != POLY_OP_ADD || lhs->n_src != 2) return NULL;
  if (!poly_no_range(ctx, c)) return NULL;
  for (int swap = 0; swap < 2; swap++) {
    PolyUOp *x = lhs->src[swap];
    PolyUOp *y = lhs->src[swap ^ 1];
    /* Same commutative UPat permutation behavior as tinygrad's load-collapse
     * `(x+y) != c` rule. */
    if (!poly_no_range(ctx, y)) continue;
    PolyUOp *rhs = poly_alu2(ctx, POLY_OP_ADD, cast_to(ctx, c, y->dtype), mul_neg_one(ctx, y));
    return poly_uop2(ctx, POLY_OP_CMPNE, POLY_BOOL, x, rhs, poly_arg_none());
  }
  return NULL;
}

static PolyUOp *rule_reduce_gated_load_collapse(
    PolyCtx *ctx,
    PolyUOp *red,
    const PolyBindings *b
) {
  (void)b;
  if (!red || red->op != POLY_OP_REDUCE) return NULL;
  if (red->arg.kind != POLY_ARG_OPS || red->arg.ops != POLY_OP_ADD) return NULL;
  if (red->n_src != 2 || red->src[1]->op != POLY_OP_RANGE) return NULL;

  PolyUOp *r = red->src[1];
  PolyUOp *where = red->src[0];
  if (!where || where->op != POLY_OP_WHERE || where->n_src != 3) return NULL;
  if (!is_zero_const(where->src[1])) return NULL;

  PolyUOp *cond = where->src[0];
  PolyUOp *expr = where->src[2];
  if (!cond || cond->op != POLY_OP_CMPNE || cond->n_src != 2) return NULL;

  PolyUOp *idx = NULL;
  if (is_range_or_cast_of_range(cond->src[0], r))
    idx = cond->src[1];
  else if (is_range_or_cast_of_range(cond->src[1], r))
    idx = cond->src[0];
  else
    return NULL;

  PolyUOp *idx_cast = cast_to(ctx, idx, r->dtype);
  PolyUOp *zero = typed_const(ctx, r->dtype, 0);
  PolyUOp *true_const = typed_const(ctx, POLY_BOOL, 1);
  PolyUOp *lt_zero = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, idx_cast, zero, poly_arg_none());
  PolyUOp *ge_zero = poly_uop2(ctx, POLY_OP_CMPNE, POLY_BOOL, lt_zero, true_const, poly_arg_none());
  PolyUOp *lt_dim =
      poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, idx_cast, cast_to(ctx, r->src[0], r->dtype), poly_arg_none());
  PolyUOp *valid = poly_uop2(ctx, POLY_OP_AND, POLY_BOOL, ge_zero, lt_dim, poly_arg_none());

  PolyUOp *invalid = poly_uop0(ctx, POLY_OP_CONST, r->dtype, poly_arg_invalid());
  PolyUOp *idx_valid = poly_uop3(ctx, POLY_OP_WHERE, r->dtype, valid, idx_cast, invalid, poly_arg_none());
  PolyUOp *from[1] = {r};
  PolyUOp *to[1] = {idx_valid};
  PolyUOp *sub_expr = poly_uop_substitute(ctx, expr, from, to, 1);
  PolyUOp *zero_expr = typed_const(ctx, sub_expr->dtype, 0);
  return poly_uop3(ctx, POLY_OP_WHERE, sub_expr->dtype, valid, sub_expr, zero_expr, poly_arg_none());
}

static PolyPatternMatcher *pm_reduce_load_collapse_get(void) {
  if (g_pm_reduce_load_collapse) return g_pm_reduce_load_collapse;
  PolyRule extra_rules[] = {
      {poly_pat_op(POLY_OP_CMPNE, NULL, 0, "cmpne"), rule_lift_add_from_cmpne},
      {poly_pat_op(POLY_OP_REDUCE, NULL, 0, "red"), rule_reduce_gated_load_collapse},
  };
  PolyPatternMatcher *extra =
      poly_pm_new(extra_rules, (int)(sizeof(extra_rules) / sizeof(extra_rules[0])));
  g_pm_reduce_load_collapse = poly_pm_concat(pm_reduce_collapse_get(), extra);
  poly_pm_destroy(extra); /* poly_pm_concat copies rules. */
  return g_pm_reduce_load_collapse;
}

static PolyUOp *reduce_load_collapse(PolyCtx *ctx, PolyUOp *red, const PolyBindings *b) {
  (void)b;
  if (red->op != POLY_OP_REDUCE) return NULL;
  if (red->arg.kind != POLY_ARG_OPS || red->arg.ops != POLY_OP_ADD) return NULL;
  if (red->n_src < 2) return NULL;
  return reduce_collapse(ctx, red, red->src[0], pm_reduce_load_collapse_get());
}

static PolyPatternMatcher *pm_reduce_simplify_base_get(void) {
  if (g_pm_reduce_simplify_base) return g_pm_reduce_simplify_base;
  PolyRule rules[] = {
      {poly_pat_op(POLY_OP_REDUCE, NULL, 0, "red"), reduce_unparented},
      {poly_pat_op(POLY_OP_REDUCE, NULL, 0, "red"), reduce_simplify},
  };
  g_pm_reduce_simplify_base = poly_pm_new(rules, (int)(sizeof(rules) / sizeof(rules[0])));
  return g_pm_reduce_simplify_base;
}

static PolyPatternMatcher *pm_reduce_simplify_get(void) {
  if (g_pm_reduce_simplify) return g_pm_reduce_simplify;
  g_pm_reduce_simplify = poly_pm_concat(pm_reduce_simplify_base_get(), poly_symbolic());
  return g_pm_reduce_simplify;
}

static PolyPatternMatcher *pm_symbolic_reduce_simplify_get(void) {
  if (g_pm_symbolic_reduce_simplify) return g_pm_symbolic_reduce_simplify;
  g_pm_symbolic_reduce_simplify =
      poly_pm_concat(poly_symbolic(), pm_reduce_simplify_base_get());
  return g_pm_symbolic_reduce_simplify;
}

/* simplify.py: no_load / pm_load_collapse */

static bool no_load(PolyCtx *ctx, PolyUOp *u) {
  int n = 0;
  PolyUOp **topo = poly_toposort(ctx, u, &n);
  if (!topo) return true;
  for (int i = 0; i < n; i++)
    if (topo[i] && topo[i]->op == POLY_OP_INDEX) return false;
  return true;
}

static PolyUOp *undo_loaded_index_math(PolyCtx *ctx, PolyUOp *cmplt, const PolyBindings *b) {
  (void)b;
  if (!cmplt || cmplt->op != POLY_OP_CMPLT || cmplt->n_src != 2) return NULL;
  if (!poly_dtype_is_int(cmplt->src[0]->dtype) || poly_dtype_is_bool(cmplt->src[0]->dtype)) return NULL;
  PolyUOp *lhs = cmplt->src[0];
  PolyUOp *c = cmplt->src[1];
  if (lhs->op != POLY_OP_ADD || lhs->n_src != 2) return NULL;
  PolyUOp *x = lhs->src[0];
  PolyUOp *y = lhs->src[1];
  if (no_load(ctx, x)) return NULL;
  if (!no_load(ctx, y) || !no_load(ctx, c)) return NULL;
  PolyUOp *rhs = poly_alu2(ctx, POLY_OP_SUB, cast_to(ctx, c, y->dtype), y);
  return poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, x, rhs, poly_arg_none());
}

static PolyPatternMatcher *g_pm_load_collapse = NULL;
PolyPatternMatcher *poly_pm_load_collapse(void) {
  if (g_pm_load_collapse) return g_pm_load_collapse;
  PolyRule rules[] = {
      {poly_pat_op(POLY_OP_REDUCE, NULL, 0, "red"), reduce_load_collapse},
      {poly_pat_op(POLY_OP_CMPLT, NULL, 0, "cmplt"), undo_loaded_index_math},
  };
  g_pm_load_collapse = poly_pm_new(rules, (int)(sizeof(rules) / sizeof(rules[0])));
  return g_pm_load_collapse;
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

PolyUOp *poly_apply_symbolic_reduce_simplify(PolyCtx *ctx, PolyUOp *sink) {
  if (getenv("POLY_DISABLE_REDUCE_SIMPLIFY")) return sink;
  return poly_graph_rewrite(ctx, sink, pm_symbolic_reduce_simplify_get());
}
