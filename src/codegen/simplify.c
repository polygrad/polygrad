/*
 * codegen/simplify.c -- port of tinygrad/codegen/simplify.py
 */

#include "simplify.h"

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#include "uop/upat.h"
#include "uop/symbolic.h"
#include "polygrad.h"
#include "bigint.h"
#include "ctx.h"
#include "schedule/indexing.h"
#include "tensor.h"
#include "utils.h"

#ifdef POLY_TESTING
static _Thread_local int simplify_fail_after = -1;
void poly_test_simplify_fail_after(int count) {
  simplify_fail_after = count;
}
#endif

static bool simplify_operation_allowed(void) {
#ifdef POLY_TESTING
  if (simplify_fail_after == 0) {
    simplify_fail_after = -1;
    return false;
  }
  if (simplify_fail_after > 0) --simplify_fail_after;
#endif
  return true;
}

/* A failed inverse substitution must not publish the temporary PARAM graph.
 * Use the shared status-returning owner, not input-on-failure convenience. */
static PolyUOp *collapse_substitute(PolyCtx *ctx, PolyUOp *u, PolyUOp **from, PolyUOp **to, int n) {
  PolyUOp *out = NULL;
  if (!simplify_operation_allowed() || poly_uop_substitute_many(ctx, &u, 1, from, to, n, &out) != 0)
    return NULL;
  return out;
}

/* tinygrad@2026-08-22/a9069c177a9d codegen/simplify.py:flatten_range. */
static PolyUOp *flatten_range(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  int off = poly_range_start(root->op);
  if (off < 0 || root->n_src <= off) return NULL;
  int n_rngs = root->n_src - off;
  PolyUOp **ordinary = malloc((size_t)n_rngs * sizeof(*ordinary));
  PolyUOp **backedge = malloc((size_t)n_rngs * sizeof(*backedge));
  if (!ordinary || !backedge) {
    free(ordinary);
    free(backedge);
    return NULL;
  }
  int n_ordinary = 0, n_backedge = 0;
  for (int i = off; i < root->n_src; i++) {
    PolyUOp *u = root->src[i];
    if (poly_dtype_eq(u->dtype, POLY_VOID) || poly_dtype_eq(u->dtype, POLY_BOOL))
      backedge[n_backedge++] = u;
    else
      ordinary[n_ordinary++] = u;
  }

  PolyUOp **flat = NULL;
  int n_flat = 0;
  if (n_ordinary > 0) {
    PolyUOp *sink = poly_uop(ctx, POLY_OP_SINK, POLY_VOID, ordinary, n_ordinary, poly_arg_none());
    int n_topo = 0;
    PolyUOp **topo = sink ? poly_toposort_alloc(ctx, sink, &n_topo) : NULL;
    flat = n_topo > 0 ? malloc((size_t)n_topo * sizeof(*flat)) : NULL;
    if (!topo || n_topo <= 0 || !flat) {
      poly_toposort_free(topo);
      free(flat);
      free(ordinary);
      free(backedge);
      return NULL;
    }
    n_flat = poly_uop_ranges(ctx, sink, flat, n_topo);
    poly_toposort_free(topo);
  }

  int n_new = off + n_flat + n_backedge;
  PolyUOp **new_src = malloc((size_t)n_new * sizeof(*new_src));
  if (!new_src) {
    free(flat);
    free(ordinary);
    free(backedge);
    return NULL;
  }
  int at = 0;
  for (int i = 0; i < off; i++)
    new_src[at++] = root->src[i];
  for (int i = 0; i < n_flat; i++)
    new_src[at++] = flat[i];
  for (int i = 0; i < n_backedge; i++)
    new_src[at++] = backedge[i];
  bool same = n_new == root->n_src;
  for (int i = 0; same && i < n_new; i++)
    same = new_src[i] == root->src[i];
  PolyUOp *ret = NULL;
  if (!same)
    ret = (root->tag != 0 || root->tag_arg.kind != POLY_ARG_NONE)
              ? poly_uop_tagged_arg(
                    ctx, root->op, root->dtype, new_src, n_new, root->arg, root->tag, root->tag_arg
                )
              : poly_uop(ctx, root->op, root->dtype, new_src, n_new, root->arg);
  free(new_src);
  free(flat);
  free(ordinary);
  free(backedge);
  return ret;
}

static _Thread_local PolyPatternMatcher *g_pm_flatten_range = NULL;
PolyPatternMatcher *poly_pm_flatten_range(void) {
  if (g_pm_flatten_range) return g_pm_flatten_range;
  PolyOpSet ops = poly_opset_add(poly_opset_add((PolyOpSet){{0, 0}}, POLY_OP_REDUCE), POLY_OP_END);
  PolyRule rules[] = {
      {poly_upat_allow_any_len(poly_upat_ops(ops, NULL, 0, NULL)), flatten_range},
  };
  g_pm_flatten_range =
      poly_pm_thread_cache(poly_pm_new(rules, (int)(sizeof(rules) / sizeof(rules[0]))));
  return g_pm_flatten_range;
}

/* tinygrad@2026-08-22/a9069c177a9d codegen/simplify.py:count_divmod. */

static int count_divmod(PolyCtx *ctx, PolyUOp *u) {
  int n_topo = 0;
  PolyUOp **topo = poly_toposort_alloc(ctx, u, &n_topo);
  if (!topo || n_topo <= 0) {
    poly_toposort_free(topo);
    return -1;
  }
  int n = 0;
  for (int i = 0; i < n_topo; i++) {
    if (topo[i]->op == POLY_OP_FLOORDIV || topo[i]->op == POLY_OP_FLOORMOD) n++;
  }
  poly_toposort_free(topo);
  return n;
}

/* C mechanics for simplify_merge_adjacent's substitution dictionary. */
static PolyUOp *try_merge_two_ranges(PolyCtx *ctx, PolyUOp *root, PolyUOp *r0, PolyUOp *r1) {
  if (!r0 || !r1 || r0->n_src < 1 || r1->n_src < 1) return NULL;
  if (poly_range_axis_type(r0->arg) != poly_range_axis_type(r1->arg)) return NULL;
  PolyUOp *prod = poly_uop2(ctx, POLY_OP_MUL, r0->dtype, r0->src[0], r1->src[0], poly_arg_none());
  PolyUOp *new_range =
      (r0->tag != 0 || r0->tag_arg.kind != POLY_ARG_NONE)
          ? poly_uop_tagged_arg(
                ctx, POLY_OP_RANGE, r0->dtype, &prod, 1, r0->arg, r0->tag, r0->tag_arg
            )
          : poly_uop1(ctx, POLY_OP_RANGE, r0->dtype, prod, r0->arg);
  if (!new_range) return NULL;
  PolyUOp *sub0 =
      poly_uop2(ctx, POLY_OP_FLOORDIV, r0->dtype, new_range, r1->src[0], poly_arg_none());
  PolyUOp *sub1 =
      poly_uop2(ctx, POLY_OP_FLOORMOD, r1->dtype, new_range, r1->src[0], poly_arg_none());
  if (!sub0 || !sub1) return NULL;
  PolyUOp *from[2] = {r0, r1};
  PolyUOp *to[2] = {sub0, sub1};
  PolyUOp *cand = poly_uop_substitute(ctx, root, from, to, 2);
  if (!cand) return NULL;
  PolyPatternMatcher *rewrite = poly_pm_concat(poly_symbolic(), poly_pm_flatten_range());
  if (!rewrite) return NULL;
  cand = poly_graph_rewrite(ctx, cand, rewrite);
  poly_pm_destroy(rewrite);
  return cand;
}

/* tinygrad@2026-08-22/a9069c177a9d
 * codegen/simplify.py:simplify_merge_adjacent. */
static PolyUOp *simplify_merge_adjacent(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  int off = poly_range_start(root->op);
  if (off < 0 || root->n_src <= off + 1) return NULL;
  for (int i = off; i < root->n_src; i++)
    if (root->src[i]->op != POLY_OP_RANGE) return NULL;
  PolyUOp *best = root;
  int best_cost = count_divmod(ctx, root);
  if (best_cost < 0) return NULL;
  int n_rng = root->n_src - off;
  int n_topo = 0;
  PolyUOp **topo = poly_toposort_alloc(ctx, root, &n_topo);
  if (!topo || n_topo <= 0) {
    poly_toposort_free(topo);
    return NULL;
  }
  for (int i = 0; i < n_rng; i++) {
    int j_begin = root->op == POLY_OP_END ? i + 1 : 0;
    int j_end = root->op == POLY_OP_END ? i + 2 : n_rng;
    for (int j = j_begin; j < j_end && j < n_rng; j++) {
      if (i == j) continue;
      PolyUOp *r0 = root->src[off + i], *r1 = root->src[off + j];
      if (poly_range_axis_type(r0->arg) != poly_range_axis_type(r1->arg)) continue;
      bool same_reduces = true;
      for (int k = 0; k < n_topo; k++) {
        if (topo[k]->op != POLY_OP_REDUCE) continue;
        if (poly_uop_in_ranges(ctx, topo[k], r0) != poly_uop_in_ranges(ctx, topo[k], r1)) {
          same_reduces = false;
          break;
        }
      }
      if (!same_reduces) continue;
      PolyUOp *cand = try_merge_two_ranges(ctx, best, r0, r1);
      if (!cand) continue;
      int cost = count_divmod(ctx, cand);
      if (cost >= 0 && cost <= best_cost) {
        best = cand;
        best_cost = cost;
      }
    }
  }
  poly_toposort_free(topo);
  return (best != root) ? best : NULL;
}

static PolyMap *current_range_ctx(void) {
  return (PolyMap *)poly_graph_rewrite_userctx();
}

static void range_ctx_set(PolyMap *state, PolyUOp *range, PolyUOp *bound) {
  if (state && range && bound) poly_map_set(state, poly_ptr_hash(range), range, bound, poly_ptr_eq);
}

static PolyUOp *range_ctx_get(PolyMap *state, PolyUOp *range) {
  return state ? poly_map_get(state, poly_ptr_hash(range), range, poly_ptr_eq) : NULL;
}

/* The dictionary's own address cannot be a UOp key. Reserve it to reject
 * the whole shrink decision after any failed INDEX inspection, including
 * failure after earlier indices already published smaller bounds. */
static bool range_ctx_failed(PolyMap *state) {
  return poly_map_get(state, poly_ptr_hash(state), state, poly_ptr_eq) != NULL;
}

static void range_ctx_fail(PolyMap *state) {
  poly_map_set(state, poly_ptr_hash(state), state, state, poly_ptr_eq);
}

typedef struct {
  PolyMap *state;
} MergeGuardCtx;

/* C callback for mark_gated's max-bound dictionary update. */
static void merge_guard_bound(const void *key, void *value, void *userdata) {
  PolyUOp *range = (PolyUOp *)key, *bound = (PolyUOp *)value;
  PolyMap *state = ((MergeGuardCtx *)userdata)->state;
  PolyUOp *current = range_ctx_get(state, range);
  if (!current) {
    range_ctx_set(state, range, bound);
    return;
  }
  if (current->op != POLY_OP_CONST || bound->op != POLY_OP_CONST) return;
  bool ok = false;
  if (poly_arg_integer_cmp(current->arg, bound->arg, &ok) < 0 && ok)
    range_ctx_set(state, range, bound);
}

/* tinygrad@2026-08-22/a9069c177a9d codegen/simplify.py:mark_gated. */
static PolyUOp *mark_gated(PolyCtx *ctx, PolyUOp *idx, const PolyBindings *b) {
  (void)b;
  PolyMap *state = current_range_ctx();
  if (!state || !idx || idx->op != POLY_OP_INDEX) return NULL;
  if (range_ctx_failed(state)) return NULL;
  if (!simplify_operation_allowed()) {
    range_ctx_fail(state);
    return NULL;
  }
  PolyMap *guards = poly_map_new(8);
  if (!guards) return NULL;
  PolyUOp *range_source = idx;
  if (idx->n_src > 1 && idx->src[1]->op == POLY_OP_WHERE) {
    PolyUOp *coord = idx->src[1];
    range_source = poly_uop_get_idx(ctx, coord);
    PolyUOp *cond = poly_uop_get_valid(ctx, coord);
    if (!range_source || !cond) {
      range_ctx_fail(state);
      poly_map_destroy(guards);
      return NULL;
    }
    int cap = 8, top = 0;
    PolyUOp **stack = malloc((size_t)cap * sizeof(*stack));
    if (cond && !stack) {
      range_ctx_fail(state);
      poly_map_destroy(guards);
      return NULL;
    }
    if (cond) stack[top++] = cond;
    bool failed = false;
    while (top > 0) {
      PolyUOp *term = stack[--top];
      if (term->op == POLY_OP_AND && term->n_src == 2) {
        if (top + 2 > cap) {
          cap *= 2;
          PolyUOp **grown = realloc(stack, (size_t)cap * sizeof(*stack));
          if (!grown) {
            failed = true;
            break;
          }
          stack = grown;
        }
        stack[top++] = term->src[1];
        stack[top++] = term->src[0];
      } else if (term->op == POLY_OP_CMPLT && term->n_src == 2 &&
                 term->src[0]->op == POLY_OP_RANGE && term->src[1]->op == POLY_OP_CONST) {
        range_ctx_set(guards, term->src[0], term->src[1]);
      }
    }
    free(stack);
    if (failed) {
      range_ctx_fail(state);
      poly_map_destroy(guards);
      return NULL;
    }
  }

  int n_topo = 0;
  PolyUOp **topo = poly_toposort_alloc(ctx, range_source, &n_topo);
  if (!topo || n_topo <= 0) {
    range_ctx_fail(state);
    poly_toposort_free(topo);
    poly_map_destroy(guards);
    return NULL;
  }
  PolyUOp **ranges = n_topo > 0 ? malloc((size_t)n_topo * sizeof(*ranges)) : NULL;
  if (!ranges) {
    range_ctx_fail(state);
    poly_toposort_free(topo);
    poly_map_destroy(guards);
    return NULL;
  }
  int n_ranges = poly_uop_ranges(ctx, range_source, ranges, n_topo);
  MergeGuardCtx merge = {.state = state};
  poly_map_foreach(guards, merge_guard_bound, &merge);
  for (int i = 0; i < n_ranges; i++)
    if (!range_ctx_get(guards, ranges[i]) && ranges[i]->n_src > 0)
      range_ctx_set(state, ranges[i], ranges[i]->src[0]);
  free(ranges);
  poly_toposort_free(topo);
  poly_map_destroy(guards);
  return NULL;
}

/* tinygrad@2026-08-22/a9069c177a9d pm_simplify_ranges REDUCE rule. */
static PolyUOp *mark_reduce_ranges(PolyCtx *ctx, PolyUOp *red, const PolyBindings *b) {
  (void)ctx;
  (void)b;
  PolyMap *state = current_range_ctx();
  if (!state || !red) return NULL;
  for (int i = 1; i < red->n_src; i++)
    if (red->src[i]->op == POLY_OP_RANGE && red->src[i]->n_src > 0)
      range_ctx_set(state, red->src[i], red->src[i]->src[0]);
  return NULL;
}

typedef struct {
  PolyCtx *ctx;
  PolyUOp **from;
  PolyUOp **to;
  int n;
  bool split;
  bool failed;
} RangeSubstitution;

/* C closure mechanics for simplify.py:do_substitute. */
static void build_range_substitution(const void *key, void *value, void *userdata) {
  RangeSubstitution *sub = userdata;
  PolyUOp *range = (PolyUOp *)key, *bound = (PolyUOp *)value;
  PolyUOp *replacement = NULL;
  if (!sub->split) {
    replacement = (range->tag != 0 || range->tag_arg.kind != POLY_ARG_NONE)
                      ? poly_uop_tagged_arg(
                            sub->ctx, POLY_OP_RANGE, range->dtype, &bound, 1, range->arg,
                            range->tag, range->tag_arg
                        )
                      : poly_uop1(sub->ctx, POLY_OP_RANGE, range->dtype, bound, range->arg);
  } else {
    int n_extra = poly_range_n_extra(range->arg);
    int64_t *extra0 = malloc((size_t)(n_extra + 1) * sizeof(*extra0));
    int64_t *extra1 = malloc((size_t)(n_extra + 1) * sizeof(*extra1));
    if (!extra0 || !extra1) {
      free(extra0);
      free(extra1);
      sub->failed = true;
      return;
    }
    const int64_t *old_extra = poly_range_extra(range->arg);
    for (int i = 0; i < n_extra; i++)
      extra0[i] = extra1[i] = old_extra[i];
    extra0[n_extra] = 0;
    extra1[n_extra] = 1;
    PolyArg outer_arg = poly_arg_range_ex(
        poly_range_axis_id(range->arg), poly_range_axis_type(range->arg), extra0, n_extra + 1
    );
    PolyArg inner_arg = poly_arg_range_ex(
        poly_range_axis_id(range->arg), poly_range_axis_type(range->arg), extra1, n_extra + 1
    );
    PolyUOp *outer_bound =
        poly_uop2(sub->ctx, POLY_OP_FLOORDIV, range->dtype, range->src[0], bound, poly_arg_none());
    PolyUOp *outer = poly_uop1(sub->ctx, POLY_OP_RANGE, range->dtype, outer_bound, outer_arg);
    PolyUOp *inner = poly_uop1(sub->ctx, POLY_OP_RANGE, range->dtype, bound, inner_arg);
    PolyUOp *scaled = poly_uop2(sub->ctx, POLY_OP_MUL, range->dtype, outer, bound, poly_arg_none());
    replacement = poly_uop2(sub->ctx, POLY_OP_ADD, range->dtype, scaled, inner, poly_arg_none());
    free(extra0);
    free(extra1);
  }
  if (!replacement) {
    sub->failed = true;
    return;
  }
  sub->from[sub->n] = range;
  sub->to[sub->n] = replacement;
  sub->n++;
}

static PolyUOp *do_range_substitute(PolyCtx *ctx, PolyUOp *root, bool split) {
  PolyMap *state = current_range_ctx();
  if (!state) return NULL;
  if (range_ctx_failed(state)) {
    poly_map_clear(state);
    return NULL;
  }
  size_t count = poly_map_len(state);
  if (count == 0) return NULL;
  PolyUOp **from = malloc(count * sizeof(*from));
  PolyUOp **to = malloc(count * sizeof(*to));
  if (!from || !to) {
    free(from);
    free(to);
    poly_map_clear(state);
    return NULL;
  }
  RangeSubstitution sub = {.ctx = ctx, .from = from, .to = to, .split = split};
  poly_map_foreach(state, build_range_substitution, &sub);
  poly_map_clear(state);
  if (sub.failed) {
    free(from);
    free(to);
    return NULL;
  }
  PolyUOp *ret = sub.n > 0 ? poly_uop_substitute(ctx, root, from, to, sub.n) : root;
  free(from);
  free(to);
  return ret != root ? poly_graph_rewrite(ctx, ret, poly_symbolic()) : NULL;
}

static PolyUOp *substitute_shrunk_ranges(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  return do_range_substitute(ctx, root, false);
}

static _Thread_local PolyPatternMatcher *g_pm_simplify_ranges = NULL;
PolyPatternMatcher *poly_pm_simplify_ranges(void) {
  if (g_pm_simplify_ranges) return g_pm_simplify_ranges;
  PolyOpSet ops = {{0, 0}};
  ops = poly_opset_add(ops, POLY_OP_END);
  ops = poly_opset_add(ops, POLY_OP_REDUCE);
  PolyRule rules[] = {
      {poly_upat_allow_any_len(poly_upat_ops(ops, NULL, 0, NULL)), simplify_merge_adjacent},
      {poly_upat_allow_any_len(poly_upat_op(POLY_OP_INDEX, NULL, 0, NULL)), mark_gated},
      {poly_upat_allow_any_len(poly_upat_op(POLY_OP_REDUCE, NULL, 0, NULL)), mark_reduce_ranges},
      {poly_upat_allow_any_len(poly_upat_op(POLY_OP_SINK, NULL, 0, NULL)),
       substitute_shrunk_ranges},
  };
  g_pm_simplify_ranges =
      poly_pm_thread_cache(poly_pm_new(rules, (int)(sizeof(rules) / sizeof(rules[0]))));
  return g_pm_simplify_ranges;
}

/* tinygrad@2026-08-22/a9069c177a9d codegen/simplify.py:mark_range_mod. */
static PolyUOp *mark_range_mod(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)ctx;
  PolyUOp *r = poly_bind(b, "r");
  PolyUOp *c = poly_bind(b, "c");
  PolyMap *state = current_range_ctx();
  if (!state || !r || !c || range_ctx_get(state, r)) return NULL;
  PolyAxisType axis_type = poly_range_axis_type(r->arg);
  if (axis_type == POLY_AXIS_WARP || axis_type == POLY_AXIS_DEVICE || r->op != POLY_OP_RANGE ||
      r->n_src < 1 || r->src[0]->op != POLY_OP_CONST || c->op != POLY_OP_CONST)
    return NULL;
  PolyInt dividend = {0}, divisor = {0}, quotient = {0}, remainder = {0};
  bool divides = poly_int_from_arg(&dividend, r->src[0]->arg) &&
                 poly_int_from_arg(&divisor, c->arg) && !poly_int_is_zero(&divisor) &&
                 poly_int_divmod(&quotient, &remainder, &dividend, &divisor, true) &&
                 poly_int_is_zero(&remainder);
  poly_int_free(&dividend);
  poly_int_free(&divisor);
  poly_int_free(&quotient);
  poly_int_free(&remainder);
  if (divides) range_ctx_set(state, r, c);
  return NULL;
}

static PolyUOp *substitute_split_ranges(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  return do_range_substitute(ctx, root, true);
}

static _Thread_local PolyPatternMatcher *g_pm_split_ranges = NULL;
PolyPatternMatcher *poly_pm_split_ranges(void) {
  if (g_pm_split_ranges) return g_pm_split_ranges;
  PolyRule rules[] = {
      {poly_upat_op2(POLY_OP_FLOORMOD, poly_upat_any("r"), poly_upat_cvar("c"), NULL),
       mark_range_mod},
      {poly_upat_op(POLY_OP_SINK, NULL, 0, NULL), substitute_split_ranges},
  };
  g_pm_split_ranges =
      poly_pm_thread_cache(poly_pm_new(rules, (int)(sizeof(rules) / sizeof(rules[0]))));
  return g_pm_split_ranges;
}

/* C helpers shared by reduce/load collapse ports. */

/* Current reduce_collapse leaves CONST, PARAM, and BUFFER dependencies
 * intact and replaces every other external expression with an ALU PARAM. */
static bool is_external_leaf(PolyUOp *u) {
  switch (u->op) {
  case POLY_OP_CONST:
  case POLY_OP_PARAM:
  case POLY_OP_BUFFER:
    return true;
  default:
    return false;
  }
}

/* Pointer-key map helpers (mirror src/uop/ops.c). */

/* CONST in a target dtype, value taken as a double and re-encoded into the
 * arg kind matching the dtype. Sole reason for existing: ensures we never
 * construct a CONST with mismatched dtype/arg.kind (the bug fixed in
 * upat.c:poly_const_like). */
static PolyUOp *typed_const(PolyCtx *ctx, PolyDType dt, double v) {
  if (poly_dtype_is_float(dt)) return poly_uop0(ctx, POLY_OP_CONST, dt, poly_arg_float(v));
  if (poly_dtype_is_bool(dt)) return poly_uop0(ctx, POLY_OP_CONST, dt, poly_arg_bool(v != 0.0));
  return poly_uop0(ctx, POLY_OP_CONST, dt, poly_arg_int((int64_t)v));
}

/* simplify.py:reduce_unparented preserves each original count and applies
 * ordinary UOp arithmetic, including weak/strong promotion. */
static PolyUOp *reduce_unparented(PolyCtx *ctx, PolyUOp *red, const PolyBindings *b) {
  (void)b;
  if (red->arg.kind != POLY_ARG_REDUCE) return NULL;
  PolyOps rop = red->arg.reduce.op;
  if (rop != POLY_OP_ADD && rop != POLY_OP_MAX && rop != POLY_OP_MUL) return NULL;
  if (red->n_src < 2) return NULL;
  for (uint16_t i = 1; i < red->n_src; i++)
    if (red->src[i]->op != POLY_OP_RANGE) return NULL;

  PolyUOp *value = red->src[0];
  PolyUOpCache *cache = poly_uop_cache_new();
  /* REDUCE arity is independent of Tensor rank. These are the two lists
   * returned by Tinygrad's partition, not fixed-rank shape storage. */
  PolyUOp **parented = malloc((size_t)red->n_src * sizeof(*parented));
  PolyUOp **unparented = malloc((size_t)red->n_src * sizeof(*unparented));
  if (!cache || !parented || !unparented) {
    poly_uop_cache_destroy(cache);
    free(parented);
    free(unparented);
    return NULL;
  }
  int n_parented = 0, n_unparented = 0;
  for (uint16_t i = 1; i < red->n_src; i++) {
    PolyUOp *r = red->src[i];
    if (poly_uop_in_ranges_ex(ctx, value, r, cache))
      parented[++n_parented] = r;
    else
      unparented[n_unparented++] = r;
  }
  poly_uop_cache_destroy(cache);

  if (n_unparented == 0) {
    free(parented);
    free(unparented);
    return NULL;
  }

  PolyUOp *ret;
  if (n_parented > 0 || !poly_dtype_eq(red->dtype, value->dtype)) {
    parented[0] = value;
    ret = poly_uop_tagged_arg(
        ctx, POLY_OP_REDUCE, red->dtype, parented, 1 + n_parented, red->arg, red->tag, red->tag_arg
    );
  } else {
    ret = value;
  }

  if (rop == POLY_OP_ADD || rop == POLY_OP_MUL) {
    PolyOps comb = (rop == POLY_OP_ADD) ? POLY_OP_MUL : POLY_OP_POW;
    for (int i = 0; ret && i < n_unparented; i++) {
      PolyUOp *count = unparented[i]->src[0];
      /* Pinned reduce_unparented multiplies by the original weak count.
       * Converting it through double loses integers above 2^53. */
      ret = poly_binop(ctx, comb, ret, count);
    }
  }
  /* MAX: drop unparented ranges with no multiplier */
  free(parented);
  free(unparented);
  return ret;
}

/* D3: pm_reduce_collapse rules */

/* UPat.reduce(arg=ADD) matches the complete (ADD, 0) metadata tuple.
 * reduce_unparented is deliberately broader and must not use this guard. */
static bool is_add_reduce(PolyUOp *red) {
  return red && red->op == POLY_OP_REDUCE && red->arg.kind == POLY_ARG_REDUCE &&
         red->arg.reduce.op == POLY_OP_ADD && red->arg.reduce.num_axes == 0;
}

static bool is_zero_const(PolyUOp *u) {
  if (!u || u->op != POLY_OP_CONST) return false;
  return (u->arg.kind == POLY_ARG_INT && u->arg.i == 0) ||
         (u->arg.kind == POLY_ARG_FLOAT && u->arg.f == 0.0) ||
         (u->arg.kind == POLY_ARG_BOOL && !u->arg.b);
}

/* Rule 1: ((x+y).or_casted() < c) -> x < (c-y)  if no_range(y,c)
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
    PolyUOp *rhs_new = poly_sub(ctx, c, y);
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
    PolyUOp *cy1 = poly_add(ctx, c, y);
    PolyUOp *cy1m1 = poly_sub(ctx, cy1, poly_const_like_int(ctx, cy1, 1));
    PolyUOp *div = poly_binop(ctx, POLY_OP_FLOORDIV, cy1m1, y);
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
  if (!is_add_reduce(red)) return false;
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

/* Pinned interval count: clamp in the count domain, then use ordinary
 * multiplication. Forcing counts through a value dtype can lose integers. */
static PolyUOp *build_count_mul_val(PolyCtx *ctx, PolyUOp *N, PolyUOp *r, PolyUOp *val) {
  PolyUOp *clamped =
      poly_minimum(ctx, poly_maximum(ctx, N, poly_const_like_int(ctx, N, 0)), r->src[0]);
  return poly_mul(ctx, clamped, val);
}

/* Rule 3: fold_range_below
 *   ((r<cut).where(0, val)).reduce_add(r)
 *     -> (r.src[0]-cut.maximum(0)).maximum(0).minimum(r.src[0]) * val
 *  iff no_range(val).
 * tinygrad simplify.py:103 */
static PolyUOp *rule_collapse_fold_range_below(PolyCtx *ctx, PolyUOp *red, const PolyBindings *b) {
  (void)b;
  PolyUOp *r, *cut, *tval, *fval;
  if (!match_where_cmplt_reduce(red, &r, &cut, &tval, &fval)) return NULL;
  /* tval must be CONST(0); val = fval */
  if (!is_zero_const(tval)) return NULL;
  PolyUOp *val = fval;
  if (!poly_no_range(ctx, val)) return NULL;
  PolyUOp *N = poly_sub(ctx, r->src[0], poly_maximum(ctx, cut, poly_const_like_int(ctx, cut, 0)));
  return build_count_mul_val(ctx, N, r, val);
}

/* Rule 5: fold_range_above
 *   ((r<cut).where(val, 0)).reduce_add(r)
 *     -> (cut.minimum(r.src[0])-r.const_like(0)).maximum(0).minimum(r.src[0]) * val
 *  iff no_range(val).
 * tinygrad simplify.py:109 */
static PolyUOp *rule_collapse_fold_range_above(PolyCtx *ctx, PolyUOp *red, const PolyBindings *b) {
  (void)b;
  PolyUOp *r, *cut, *tval, *fval;
  if (!match_where_cmplt_reduce(red, &r, &cut, &tval, &fval)) return NULL;
  /* fval must be CONST(0); val = tval */
  if (!is_zero_const(fval)) return NULL;
  PolyUOp *val = tval;
  if (!poly_no_range(ctx, val)) return NULL;
  PolyUOp *N = poly_sub(ctx, poly_minimum(ctx, cut, r->src[0]), poly_const_like_int(ctx, r, 0));
  return build_count_mul_val(ctx, N, r, val);
}

/* Rule 4: fold_range_two_sided
 *   (((r<lower).logical_not()) & (r<upper)).where(val,0).reduce_add(r)
 *     -> (upper.minimum(n) - lower.maximum(0)).maximum(0).minimum(n) * val
 *  where n = r.src[0]. Polygrad's logical_not is CMPNE(x, true) (P5). The
 *  AND of two bool comparisons is POLY_OP_AND.
 * tinygrad simplify.py:105-107 */
static PolyUOp *rule_collapse_fold_range_two_sided(
    PolyCtx *ctx,
    PolyUOp *red,
    const PolyBindings *b
) {
  (void)b;
  if (!is_add_reduce(red) || red->n_src != 2) return NULL;
  PolyUOp *r = red->src[1];
  if (r->op != POLY_OP_RANGE) return NULL;
  PolyUOp *value = red->src[0];
  if (value->op != POLY_OP_WHERE || value->n_src != 3) return NULL;
  PolyUOp *fval = value->src[2];
  PolyUOp *val = value->src[1];
  if (!is_zero_const(fval)) return NULL;
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
  PolyUOp *N = poly_sub(
      ctx, poly_minimum(ctx, upper, r->src[0]),
      poly_maximum(ctx, lower, poly_const_like_int(ctx, lower, 0))
  );
  return build_count_mul_val(ctx, N, r, val);
}

/* UOp.reduce builds a fresh node with the value's dtype and the supplied
 * range tuple. This compiler form must not run frontend axis admission. */
static PolyUOp *collapse_reduce_value(PolyCtx *ctx, PolyUOp *red, PolyUOp *value) {
  PolyUOp **src = malloc((size_t)red->n_src * sizeof(*src));
  if (!src) return NULL;
  src[0] = value;
  for (int i = 1; i < red->n_src; i++)
    src[i] = red->src[i];
  PolyUOp *out =
      poly_uop(ctx, POLY_OP_REDUCE, value->dtype, src, red->n_src, poly_arg_reduce(POLY_OP_ADD, 0));
  free(src);
  return out;
}

/* simplify.py: an independent invalid guard remains outside the reduction;
 * Invalid is not a numeric zero or an accumulated value. */
static PolyUOp *rule_collapse_invalid_guard(PolyCtx *ctx, PolyUOp *red, const PolyBindings *b) {
  (void)b;
  if (!is_add_reduce(red) || red->n_src < 1) return NULL;
  PolyUOp *value = red->src[0];
  if (value->op != POLY_OP_WHERE || value->n_src != 3 || value->src[2]->op != POLY_OP_CONST ||
      value->src[2]->arg.kind != POLY_ARG_INVALID || !poly_no_range(ctx, value->src[0]))
    return NULL;
  PolyUOp *reduced = collapse_reduce_value(ctx, red, value->src[1]);
  return reduced ? poly_uop3(
                       ctx, POLY_OP_WHERE, reduced->dtype, value->src[0], reduced, value->src[2],
                       poly_arg_none()
                   )
                 : NULL;
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
  if (!is_add_reduce(red) || red->n_src < 1) return NULL;
  PolyUOp *value = red->src[0];
  if (value->op != POLY_OP_ADD || value->n_src != 2) return NULL;
  PolyUOp *x = value->src[0];
  PolyUOp *y = value->src[1];
  PolyUOp *xred = collapse_reduce_value(ctx, red, x);
  PolyUOp *yred = collapse_reduce_value(ctx, red, y);
  return xred && yred ? poly_add(ctx, xred, yred) : NULL;
}

/* Rule 7: and_on_where
 *   ((PARAM & y).where(c, 0)).reduce_add(*ranges)
 *     -> y.where(c, 0).reduce_add(*ranges) * x
 * tinygrad simplify.py:115-116 */
static PolyUOp *rule_collapse_and_on_where(PolyCtx *ctx, PolyUOp *red, const PolyBindings *b) {
  (void)b;
  if (!is_add_reduce(red) || red->n_src < 1) return NULL;
  PolyUOp *where = red->src[0];
  if (where->op != POLY_OP_WHERE || where->n_src != 3) return NULL;
  PolyUOp *fval = where->src[2];
  if (!is_zero_const(fval)) return NULL;
  PolyUOp *and_op = where->src[0];
  if (and_op->op != POLY_OP_AND || and_op->n_src != 2) return NULL;
  /* One side must be the ALU PARAM introduced by reduce_collapse. */
  PolyUOp *x = NULL, *y = NULL;
  if (and_op->src[0]->op == POLY_OP_PARAM) {
    x = and_op->src[0];
    y = and_op->src[1];
  } else if (and_op->src[1]->op == POLY_OP_PARAM) {
    x = and_op->src[1];
    y = and_op->src[0];
  } else {
    return NULL;
  }
  PolyUOp *c = where->src[1];
  /* New WHERE: y.where(c, fval) */
  PolyUOp *new_where = poly_uop3(ctx, POLY_OP_WHERE, where->dtype, y, c, fval, poly_arg_none());
  PolyUOp *new_red = collapse_reduce_value(ctx, red, new_where);
  return new_red ? poly_mul(ctx, new_red, x) : NULL;
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
 *           if s in included or s in replaces or s.op in {CONST, PARAM, BUFFER}: continue
 *           replaces[s] = UOp.variable(..., param=True)
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

static PolyUOp *reduce_collapse(PolyCtx *ctx, PolyUOp *red, PolyUOp *u, PolyPatternMatcher *pm) {
  PolyUOpCache *cache = poly_uop_cache_new();
  if (!cache) return NULL;
  PolyMap *included_map = NULL;
  PolyMap *replaces_map = NULL;
  PolyUOp **from_arr = NULL, **to_arr = NULL;
  bool dbg = getenv("POLY_DEBUG_REDUCE_SIMPLIFY") != NULL;
  PolyScratchMark scratch = poly_ctx_scratch_mark(ctx);

  /* Loop over each reduce range. */
  for (uint16_t ridx = 1; ridx < red->n_src; ridx++) {
    PolyUOp *r = red->src[ridx];
    if (r->op != POLY_OP_RANGE) goto fail;

    /* Toposort u, gated by "r in node.ranges" — yields "included" set. */
    GateCtx g = {.ctx = ctx, .r = r, .cache = cache};
    int n_inc = 0;
    PolyUOp **included = poly_toposort_ex_user_scratch(ctx, u, &n_inc, collapse_gate, &g, true);
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
     * mint a fresh bounded ALU PARAM carrying its current vmin/vmax. */
    if (replaces_map) poly_map_destroy(replaces_map);
    replaces_map = poly_map_new(16);
    /* Each external dependency occurs on an included source edge. This is
     * an upper bound on Tinygrad's replaces dictionary, not a rank limit. */
    size_t capacity = 0;
    for (int i = 0; i < n_inc; i++) {
      if (capacity > INT_MAX - (size_t)included[i]->n_src) goto fail;
      capacity += included[i]->n_src;
    }
    if (capacity > SIZE_MAX / sizeof(*from_arr)) goto fail;
    free(from_arr);
    free(to_arr);
    from_arr = capacity ? malloc(capacity * sizeof(*from_arr)) : NULL;
    to_arr = capacity ? malloc(capacity * sizeof(*to_arr)) : NULL;
    if (capacity && (!from_arr || !to_arr)) goto fail;
    int n_repl = 0;
    char namebuf[32];
    for (int i = 0; i < n_inc; i++) {
      PolyUOp *node = included[i];
      for (uint16_t k = 0; k < node->n_src; k++) {
        PolyUOp *s = node->src[k];
        if (poly_map_get(included_map, poly_ptr_hash(s), s, poly_ptr_eq)) continue;
        if (poly_map_get(replaces_map, poly_ptr_hash(s), s, poly_ptr_eq)) continue;
        if (is_external_leaf(s)) continue;
        snprintf(namebuf, sizeof(namebuf), "in%d", n_repl);
        PolyUOp *dv = poly_uop_variable_like_bounds(ctx, namebuf, s);
        if (!dv) goto fail;
        from_arr[n_repl] = s;
        to_arr[n_repl] = dv;
        poly_map_set(replaces_map, poly_ptr_hash(s), s, dv, poly_ptr_eq);
        n_repl++;
      }
    }

    /* Substitute, build collapse form, run pm_reduce_collapse, check
     * no_range, substitute back. */
    PolyUOp *substituted = collapse_substitute(ctx, u, from_arr, to_arr, n_repl);
    if (!substituted) goto fail;
    PolyUOp *one_range_srcs[2] = {substituted, r};
    PolyUOp *collapse_form = poly_uop(
        ctx, POLY_OP_REDUCE, substituted->dtype, one_range_srcs, 2, poly_arg_reduce(POLY_OP_ADD, 0)
    );
    if (!collapse_form) goto fail;
    PolyUOp *sink = poly_graph_rewrite(ctx, collapse_form, pm);
    if (!sink) goto fail;
    if (dbg)
      fprintf(
          stderr, "  [reduce_collapse] n_repl=%d sink_op=%s no_range=%d\n", n_repl,
          poly_op_name(sink->op), (int)poly_no_range_ex(ctx, sink, cache)
      );
    if (!poly_no_range_ex(ctx, sink, cache)) goto fail;
    /* Substitute the original external expressions back after collapse. */
    u = collapse_substitute(ctx, sink, to_arr, from_arr, n_repl);
    if (!u) goto fail;
  }

  if (included_map) poly_map_destroy(included_map);
  if (replaces_map) poly_map_destroy(replaces_map);
  free(from_arr);
  free(to_arr);
  poly_ctx_scratch_rewind(ctx, scratch);
  poly_uop_cache_destroy(cache);
  return u;

fail:
  if (included_map) poly_map_destroy(included_map);
  if (replaces_map) poly_map_destroy(replaces_map);
  free(from_arr);
  free(to_arr);
  poly_ctx_scratch_rewind(ctx, scratch);
  poly_uop_cache_destroy(cache);
  return NULL;
}

/* Entry rule for pm_reduce_simplify (simplify.py:147-149):
 *   (UPat(REDUCE, src=(UPat.var("u"),), allow_any_len=True, arg=Ops.ADD,
 *       name="red"), reduce_collapse)
 */
static PolyUOp *reduce_simplify(PolyCtx *ctx, PolyUOp *red, const PolyBindings *b) {
  (void)b;
  if (!is_add_reduce(red) || red->n_src < 1) return NULL;
  return reduce_collapse(ctx, red, red->src[0], pm_reduce_collapse_get());
}

/* Current tinygrad codegen/simplify.py:pm_reduce_simplify.
 * Tinygrad source (codegen/simplify.py:147-149):
 *   pm_reduce_simplify = pm_reduce_unparented + PatternMatcher([
 *     (UPat(Ops.REDUCE, src=(UPat.var("u"),), allow_any_len=True,
 *           arg=Ops.ADD, name="red"), reduce_collapse),
 *   ])
 *
 * Rangeify concatenates this matcher with symbolic and its cleanup matchers,
 * matching schedule/rangeify.py:get_kernel_graph.
 */
static _Thread_local PolyPatternMatcher *g_pm_reduce_unparented = NULL;
static _Thread_local PolyPatternMatcher *g_pm_reduce_collapse_base = NULL;
static _Thread_local PolyPatternMatcher *g_pm_reduce_collapse = NULL;
static _Thread_local PolyPatternMatcher *g_pm_reduce_load_collapse = NULL;
static _Thread_local PolyPatternMatcher *g_pm_reduce_simplify_base = NULL;
static _Thread_local PolyPatternMatcher *g_pm_reduce_simplify = NULL;
static _Thread_local PolyPatternMatcher *g_pm_symbolic_reduce_simplify = NULL;

PolyPatternMatcher *poly_pm_reduce_unparented(void) {
  if (g_pm_reduce_unparented) return g_pm_reduce_unparented;
  PolyRule rules[] = {
      {poly_upat_op(POLY_OP_REDUCE, NULL, 0, "red"), reduce_unparented},
  };
  g_pm_reduce_unparented =
      poly_pm_thread_cache(poly_pm_new(rules, (int)(sizeof(rules) / sizeof(rules[0]))));
  return g_pm_reduce_unparented;
}

static PolyPatternMatcher *pm_reduce_collapse_base_get(void) {
  if (g_pm_reduce_collapse_base) return g_pm_reduce_collapse_base;
  PolyRule rules[] = {
      {poly_upat_op(POLY_OP_REDUCE, NULL, 0, "red"), reduce_unparented},
      {poly_upat_op(POLY_OP_CMPLT, NULL, 0, "x"), rule_collapse_lift_add_from_cmplt},
      {poly_upat_op(POLY_OP_CMPLT, NULL, 0, "x"), rule_collapse_lift_mul_from_cmplt},
      {poly_upat_op(POLY_OP_REDUCE, NULL, 0, "red"), rule_collapse_fold_range_below},
      {poly_upat_op(POLY_OP_REDUCE, NULL, 0, "red"), rule_collapse_fold_range_two_sided},
      {poly_upat_op(POLY_OP_REDUCE, NULL, 0, "red"), rule_collapse_fold_range_above},
      {poly_upat_op(POLY_OP_REDUCE, NULL, 0, "red"), rule_collapse_invalid_guard},
      {poly_upat_op(POLY_OP_REDUCE, NULL, 0, "red"), rule_collapse_reduce_add_distribute},
      {poly_upat_op(POLY_OP_REDUCE, NULL, 0, "red"), rule_collapse_and_on_where},
      {poly_upat_op(POLY_OP_MUL, NULL, 0, "mul"), rule_collapse_mul_casted_bool},
  };
  g_pm_reduce_collapse_base =
      poly_pm_thread_cache(poly_pm_new(rules, (int)(sizeof(rules) / sizeof(rules[0]))));
  return g_pm_reduce_collapse_base;
}

static PolyPatternMatcher *pm_reduce_collapse_get(void) {
  if (g_pm_reduce_collapse) return g_pm_reduce_collapse;
  g_pm_reduce_collapse =
      poly_pm_thread_cache(poly_pm_concat(pm_reduce_collapse_base_get(), poly_symbolic()));
  return g_pm_reduce_collapse;
}

#ifdef POLY_TESTING
PolyUOp *poly_test_reduce_collapse_rewrite(PolyCtx *ctx, PolyUOp *u) {
  return poly_pm_rewrite(pm_reduce_collapse_base_get(), ctx, u);
}
PolyUOp *poly_test_reduce_collapse(PolyCtx *ctx, PolyUOp *u) {
  return reduce_collapse(ctx, u, u->src[0], pm_reduce_collapse_get());
}
#endif

/* simplify.py: pm_reduce_load_collapse */

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
    PolyUOp *rhs = poly_sub(ctx, poly_cast(ctx, c, y->dtype), y);
    return poly_uop2(ctx, POLY_OP_CMPNE, POLY_BOOL, x, rhs, poly_arg_none());
  }
  return NULL;
}

static PolyUOp *rule_reduce_gated_load_collapse(PolyCtx *ctx, PolyUOp *red, const PolyBindings *b) {
  (void)b;
  if (!is_add_reduce(red)) return NULL;
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

  PolyUOp *idx_cast = poly_cast(ctx, idx, r->dtype);
  PolyUOp *zero = typed_const(ctx, r->dtype, 0);
  PolyUOp *true_const = typed_const(ctx, POLY_BOOL, 1);
  PolyUOp *lt_zero = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, idx_cast, zero, poly_arg_none());
  PolyUOp *ge_zero = poly_uop2(ctx, POLY_OP_CMPNE, POLY_BOOL, lt_zero, true_const, poly_arg_none());
  PolyUOp *lt_dim = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, idx_cast, r->src[0], poly_arg_none());
  PolyUOp *valid = poly_uop2(ctx, POLY_OP_AND, POLY_BOOL, ge_zero, lt_dim, poly_arg_none());

  PolyUOp *invalid = poly_uop_const(ctx, poly_arg_invalid(), r->dtype);
  PolyUOp *idx_valid =
      poly_uop3(ctx, POLY_OP_WHERE, r->dtype, valid, idx_cast, invalid, poly_arg_none());
  PolyUOp *from[1] = {r};
  PolyUOp *to[1] = {idx_valid};
  PolyUOp *sub_expr = poly_uop_substitute(ctx, expr, from, to, 1);
  PolyUOp *zero_expr = typed_const(ctx, sub_expr->dtype, 0);
  return poly_uop3(
      ctx, POLY_OP_WHERE, sub_expr->dtype, valid, sub_expr, zero_expr, poly_arg_none()
  );
}

static PolyPatternMatcher *pm_reduce_load_collapse_get(void) {
  if (g_pm_reduce_load_collapse) return g_pm_reduce_load_collapse;
  PolyRule extra_rules[] = {
      {poly_upat_op(POLY_OP_CMPNE, NULL, 0, "cmpne"), rule_lift_add_from_cmpne},
      {poly_upat_op(POLY_OP_REDUCE, NULL, 0, "red"), rule_reduce_gated_load_collapse},
  };
  PolyPatternMatcher *extra =
      poly_pm_new(extra_rules, (int)(sizeof(extra_rules) / sizeof(extra_rules[0])));
  g_pm_reduce_load_collapse = poly_pm_thread_cache(poly_pm_concat(pm_reduce_collapse_get(), extra));
  poly_pm_destroy(extra); /* poly_pm_concat copies rules. */
  return g_pm_reduce_load_collapse;
}

static PolyUOp *reduce_load_collapse(PolyCtx *ctx, PolyUOp *red, const PolyBindings *b) {
  (void)b;
  if (!is_add_reduce(red) || red->n_src != 2) return NULL;
  return reduce_collapse(ctx, red, red->src[0], pm_reduce_load_collapse_get());
}

PolyPatternMatcher *poly_pm_reduce_simplify(void) {
  if (g_pm_reduce_simplify_base) return g_pm_reduce_simplify_base;
  PolyRule rules[] = {
      {poly_upat_op(POLY_OP_REDUCE, NULL, 0, "red"), reduce_unparented},
      {poly_upat_op(POLY_OP_REDUCE, NULL, 0, "red"), reduce_simplify},
  };
  g_pm_reduce_simplify_base =
      poly_pm_thread_cache(poly_pm_new(rules, (int)(sizeof(rules) / sizeof(rules[0]))));
  return g_pm_reduce_simplify_base;
}

static PolyPatternMatcher *pm_reduce_simplify_get(void) {
  if (g_pm_reduce_simplify) return g_pm_reduce_simplify;
  g_pm_reduce_simplify =
      poly_pm_thread_cache(poly_pm_concat(poly_pm_reduce_simplify(), poly_symbolic()));
  return g_pm_reduce_simplify;
}

static PolyPatternMatcher *pm_symbolic_reduce_simplify_get(void) {
  if (g_pm_symbolic_reduce_simplify) return g_pm_symbolic_reduce_simplify;
  g_pm_symbolic_reduce_simplify =
      poly_pm_thread_cache(poly_pm_concat(poly_symbolic(), poly_pm_reduce_simplify()));
  return g_pm_symbolic_reduce_simplify;
}

/* simplify.py: no_load / pm_load_collapse */

/* C traversal can fail: neither absence nor presence is then established. */
static int no_load(PolyCtx *ctx, PolyUOp *u) {
  int n = 0;
  PolyUOp **topo = simplify_operation_allowed() ? poly_toposort_alloc(ctx, u, &n) : NULL;
  if (!topo) return -1;
  bool ret = true;
  for (int i = 0; i < n; i++)
    if (topo[i] && topo[i]->op == POLY_OP_INDEX) {
      ret = false;
      break;
    }
  poly_toposort_free(topo);
  return ret;
}

static PolyUOp *undo_loaded_index_math(PolyCtx *ctx, PolyUOp *cmplt, const PolyBindings *b) {
  (void)b;
  if (!cmplt || cmplt->op != POLY_OP_CMPLT || cmplt->n_src != 2) return NULL;
  PolyUOp *lhs = cmplt->src[0];
  PolyUOp *c = cmplt->src[1];
  if (lhs->op != POLY_OP_ADD || lhs->n_src != 2) return NULL;
  if (no_load(ctx, c) != 1) return NULL;
  for (int swap = 0; swap < 2; swap++) {
    PolyUOp *x = lhs->src[swap], *y = lhs->src[swap ^ 1];
    /* Pinned pm_load_collapse restricts x to weakint: moving arithmetic
     * across a fixed-width comparison would change overflow semantics. */
    if (!poly_dtype_eq(x->dtype, POLY_WEAKINT) || no_load(ctx, x) != 0 || no_load(ctx, y) != 1)
      continue;
    return poly_alu2(ctx, POLY_OP_CMPLT, x, poly_sub(ctx, c, y));
  }
  return NULL;
}

static _Thread_local PolyPatternMatcher *g_pm_load_collapse = NULL;
PolyPatternMatcher *poly_pm_load_collapse(void) {
  if (g_pm_load_collapse) return g_pm_load_collapse;
  PolyRule rules[] = {
      {poly_upat_op(POLY_OP_REDUCE, NULL, 0, "red"), reduce_load_collapse},
      {poly_upat_op(POLY_OP_CMPLT, NULL, 0, "cmplt"), undo_loaded_index_math},
  };
  g_pm_load_collapse =
      poly_pm_thread_cache(poly_pm_new(rules, (int)(sizeof(rules) / sizeof(rules[0]))));
  return g_pm_load_collapse;
}

/* Public entries */

PolyUOp *poly_apply_reduce_unparented_only(PolyCtx *ctx, PolyUOp *sink) {
  /* Test-only entry: standalone pm_reduce_unparented (no symbolic concat).
   * Mirrors test/parity_scripts/tg_reduce_unparented_gt.py ground truth. */
  return poly_graph_rewrite(ctx, sink, poly_pm_reduce_unparented());
}

PolyUOp *poly_apply_reduce_simplify(PolyCtx *ctx, PolyUOp *sink) {
  if (getenv("POLY_DISABLE_REDUCE_SIMPLIFY")) return sink;
  return poly_graph_rewrite(ctx, sink, pm_reduce_simplify_get());
}

PolyUOp *poly_apply_symbolic_reduce_simplify(PolyCtx *ctx, PolyUOp *sink) {
  if (getenv("POLY_DISABLE_REDUCE_SIMPLIFY")) return sink;
  /* Pinned graph_rewrite keeps CALL/FUNCTION bodies opaque by default; the
   * recursive scheduler rewrites each retained body separately
   * (uop/ops.py:1540-1649, schedule/__init__.py:94-105). */
  return poly_graph_rewrite_ctx_ex2(
      ctx, sink, pm_symbolic_reduce_simplify_get(), NULL, false, false
  );
}
