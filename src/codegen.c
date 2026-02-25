/*
 * codegen.c — Sequential rewrite passes (port of full_rewrite_to_sink)
 *
 * Mirrors tinygrad's codegen/__init__.py full_rewrite_to_sink pipeline.
 * Currently implements:
 *   - pm_reduce: REDUCE → DEFINE_REG + AFTER accumulation + END merge
 *   - pm_decomp: MAX→WHERE, MUL→SHL, IDIV→SHR (late decompositions)
 *   - pm_transcendental: EXP2 → polynomial approximation (xexp2)
 */

#include "codegen.h"
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <stdio.h>

static bool ptr_eq(const void *a, const void *b) { return a == b; }
static uint32_t ptr_hash(const void *p) {
  uintptr_t x = (uintptr_t)p;
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33;
  return (uint32_t)x;
}
static PolyArg poly_arg_int_tuple_local(int64_t *vals, int n);
static PolyUOp *scalarize_lane_expr(PolyCtx *ctx, PolyUOp *u, int lane);

/* ── Reduce identity helper ────────────────────────────────────────────── */

static double codegen_reduce_identity(PolyOps op) {
  switch (op) {
    case POLY_OP_ADD: return 0.0;
    case POLY_OP_MUL: return 1.0;
    case POLY_OP_MAX: return -__builtin_inf();
    default: return 0.0;
  }
}

static int horizontal_reduce_terms(PolyCtx *ctx, PolyUOp *inp, PolyDType out_dtype,
                                   PolyUOp **out_terms, int max_terms) {
  if (!inp || !out_terms || max_terms <= 0) return 0;
  if (poly_dtype_eq(inp->dtype, out_dtype)) {
    out_terms[0] = inp;
    return 1;
  }
  int in_cnt = inp->dtype.count;
  int out_cnt = out_dtype.count;
  if (in_cnt <= 0 || out_cnt <= 0 || (in_cnt % out_cnt) != 0) {
    out_terms[0] = inp;
    return 1;
  }
  int horizontal_amount = in_cnt / out_cnt;
  if (horizontal_amount <= 1) {
    out_terms[0] = inp;
    return 1;
  }

  int n_out = 0;
  PolyDType scalar = poly_dtype_scalar(inp->dtype);
  for (int i = 0; i < horizontal_amount && n_out < max_terms; i++) {
    int64_t idxs[128];
    int n_idxs = 0;
    for (int j = i; j < in_cnt && n_idxs < 128; j += horizontal_amount) idxs[n_idxs++] = j;
    PolyDType gep_dtype = (n_idxs == 1) ? scalar : poly_dtype_vec(scalar, n_idxs);
    out_terms[n_out++] = poly_uop1(ctx, POLY_OP_GEP, gep_dtype, inp, poly_arg_int_tuple_local(idxs, n_idxs));
  }
  return n_out > 0 ? n_out : 1;
}

/* ── pm_reduce: REDUCE → DEFINE_REG + END merge ───────────────────────── */

/* Forward declaration: shared substitute helper used by reduce END merge. */
static PolyUOp *substitute_node(PolyCtx *ctx, PolyUOp *node,
                                PolyUOp *old_node, PolyUOp *new_node,
                                PolyUOp **memo_old, PolyUOp **memo_new,
                                int *memo_n, int memo_cap);

/* ── Optimize preprocess (ports from tinygrad codegen/simplify.py) ───── */

static int range_start_for_op(PolyOps op) {
  switch (op) {
    case POLY_OP_BUFFERIZE: return 1;
    case POLY_OP_REDUCE: return 1;
    case POLY_OP_STORE: return 2;
    case POLY_OP_WMMA: return 3;
    case POLY_OP_END: return 1;
    default: return -1;
  }
}

static int collect_unique_ranges(PolyCtx *ctx, PolyUOp **rngs, int n_rngs,
                                 PolyUOp **out, int max_out) {
  if (!rngs || n_rngs <= 0 || max_out <= 0) return 0;
  PolyUOp *tmp_sink = poly_uop(ctx, POLY_OP_SINK, POLY_VOID, rngs, n_rngs, poly_arg_none());
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, tmp_sink, &n_topo);
  int n_out = 0;
  for (int i = 0; i < n_topo && n_out < max_out; i++) {
    if (topo[i]->op != POLY_OP_RANGE) continue;
    bool dup = false;
    for (int j = 0; j < n_out; j++) if (out[j] == topo[i]) { dup = true; break; }
    if (!dup) out[n_out++] = topo[i];
  }
  return n_out;
}

static PolyUOp *rule_flatten_range(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
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
      if (flat_rngs[i] != root->src[off + i]) { same = false; break; }
    }
  }
  if (same) return NULL;

  PolyUOp *new_src[POLY_MAX_DIMS + 8];
  int n_new = 0;
  for (int i = 0; i < off; i++) new_src[n_new++] = root->src[i];
  for (int i = 0; i < n_flat; i++) new_src[n_new++] = flat_rngs[i];
  return poly_uop(ctx, root->op, root->dtype, new_src, n_new, root->arg);
}

static PolyPatternMatcher *g_pm_flatten_range = NULL;
static PolyPatternMatcher *poly_pm_flatten_range(void) {
  if (g_pm_flatten_range) return g_pm_flatten_range;
  PolyOpSet ops = {{0, 0}};
  ops = poly_opset_add(ops, POLY_OP_REDUCE);
  ops = poly_opset_add(ops, POLY_OP_STORE);
  ops = poly_opset_add(ops, POLY_OP_END);
  PolyRule rules[] = {
    { poly_pat_allow_any_len(poly_pat_ops(ops, NULL, 0, NULL)), rule_flatten_range },
  };
  g_pm_flatten_range = poly_pm_new(rules, (int)(sizeof(rules) / sizeof(rules[0])));
  return g_pm_flatten_range;
}

typedef struct {
  PolyUOp *r[POLY_MAX_DIMS];
  PolyUOp *c[POLY_MAX_DIMS];
  int n;
} SplitRangeCtx;

static SplitRangeCtx *current_split_ctx(void) {
  SplitRangeCtx *sctx = (SplitRangeCtx *)poly_graph_rewrite_userctx();
  if (!sctx) {
    fprintf(stderr, "polygrad codegen: pm_split_ranges requires rewrite ctx\n");
    abort();
  }
  return sctx;
}

static PolyUOp *rule_mark_range_mod(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)ctx;
  PolyUOp *r = poly_bind(b, "r");
  PolyUOp *c = poly_bind(b, "c");
  if (!r || !c) return NULL;
  if (r->op != POLY_OP_RANGE || c->op != POLY_OP_CONST || c->arg.kind != POLY_ARG_INT) return NULL;
  if (!(r->n_src > 0 && r->src[0]->op == POLY_OP_CONST && r->src[0]->arg.kind == POLY_ARG_INT)) return NULL;
  int64_t rv = r->src[0]->arg.i, cv = c->arg.i;
  if (cv <= 1 || rv <= 0) return NULL;
  if ((rv % cv) != 0) return NULL;

  SplitRangeCtx *sctx = current_split_ctx();
  for (int i = 0; i < sctx->n; i++) if (sctx->r[i] == r) return NULL;
  if (sctx->n < POLY_MAX_DIMS) {
    sctx->r[sctx->n] = r;
    sctx->c[sctx->n] = c;
    sctx->n++;
  }
  return NULL;
}

static PolyUOp *rule_apply_split_ranges(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  SplitRangeCtx *sctx = current_split_ctx();
  if (sctx->n <= 0) return NULL;

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
    PolyArg k0_arg = poly_arg_range_ex(poly_range_axis_id(r->arg), poly_range_axis_type(r->arg),
                                       extra0, n_extra + 1);
    PolyArg k1_arg = poly_arg_range_ex(poly_range_axis_id(r->arg), poly_range_axis_type(r->arg),
                                       extra1, n_extra + 1);
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
  out = poly_graph_rewrite(ctx, out, poly_symbolic_simple());
  out = poly_graph_rewrite(ctx, out, poly_pm_flatten_range());
  return out;
}

static PolyPatternMatcher *g_pm_split_ranges = NULL;
static PolyPatternMatcher *poly_pm_split_ranges(void) {
  if (g_pm_split_ranges) return g_pm_split_ranges;
  PolyRule rules[] = {
    { poly_pat_op2(POLY_OP_MOD, poly_pat_any("r"), poly_pat_cvar("c"), NULL), rule_mark_range_mod },
    { poly_pat_op(POLY_OP_SINK, NULL, 0, NULL), rule_apply_split_ranges },
  };
  g_pm_split_ranges = poly_pm_new(rules, (int)(sizeof(rules) / sizeof(rules[0])));
  return g_pm_split_ranges;
}

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
  PolyUOp *from[2] = { r0, r1 };
  PolyUOp *to[2] = { sub0, sub1 };
  PolyUOp *cand = poly_uop_substitute(ctx, root, from, to, 2);
  cand = poly_graph_rewrite(ctx, cand, poly_symbolic_simple());
  cand = poly_graph_rewrite(ctx, cand, poly_pm_flatten_range());
  return cand;
}

static PolyUOp *rule_simplify_ranges(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
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
    if (c <= best_cost) { best = cand; best_cost = c; }
  }
  return (best != root) ? best : NULL;
}

static PolyPatternMatcher *g_pm_simplify_ranges = NULL;
static PolyPatternMatcher *poly_pm_simplify_ranges(void) {
  if (g_pm_simplify_ranges) return g_pm_simplify_ranges;
  PolyOpSet ops = {{0, 0}};
  ops = poly_opset_add(ops, POLY_OP_END);
  ops = poly_opset_add(ops, POLY_OP_REDUCE);
  PolyRule rules[] = {
    { poly_pat_allow_any_len(poly_pat_ops(ops, NULL, 0, NULL)), rule_simplify_ranges },
  };
  g_pm_simplify_ranges = poly_pm_new(rules, (int)(sizeof(rules) / sizeof(rules[0])));
  return g_pm_simplify_ranges;
}

/* ── apply_opts (BEAM=0, tinygrad-style baseline subset) ─────────────── */

typedef struct {
  int64_t axis_id;
  PolyAxisType axis_type;
  PolyUOp *r;
} AxisRangeRef;

static int axis_range_cmp(const void *ap, const void *bp) {
  const AxisRangeRef *a = (const AxisRangeRef *)ap;
  const AxisRangeRef *b = (const AxisRangeRef *)bp;
  if (a->axis_type != b->axis_type) return (int)a->axis_type - (int)b->axis_type;
  if (a->axis_id < b->axis_id) return -1;
  if (a->axis_id > b->axis_id) return 1;
  return (a->r < b->r) ? -1 : (a->r > b->r);
}

/* tinygrad heuristic baseline (hand_coded_optimizations):
 * if nothing is upcasted and the last upcastable dim can split by 4,
 * apply UPCAST by 4. This is enough to unlock expander/devectorizer path
 * on scalar elementwise kernels and preserves no-opt behavior on others. */
static PolyUOp *poly_unroll_reduce_ranges(PolyCtx *ctx, PolyUOp *sink) {
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, sink, &n_topo);
  int64_t reduce_axes[128];
  int n_reduce_axes = 0;

  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    if (u->op != POLY_OP_REDUCE || u->n_src <= 1) continue;
    for (int j = 1; j < u->n_src; j++) {
      PolyUOp *r = u->src[j];
      if (!r || r->op != POLY_OP_RANGE || !poly_arg_is_range(r->arg)) continue;
      int64_t axis_id = poly_range_axis_id(r->arg);
      bool dup = false;
      for (int k = 0; k < n_reduce_axes; k++) {
        if (reduce_axes[k] == axis_id) { dup = true; break; }
      }
      if (!dup && n_reduce_axes < 128) reduce_axes[n_reduce_axes++] = axis_id;
    }
  }

  PolyUOp *from[128];
  PolyUOp *to[128];
  int n_sub = 0;

  for (int i = 0; i < n_topo && n_sub < 128; i++) {
    PolyUOp *u = topo[i];
    if (u->op != POLY_OP_RANGE || !poly_arg_is_range(u->arg)) continue;
    bool is_reduce_axis = false;
    int64_t axis_id = poly_range_axis_id(u->arg);
    for (int j = 0; j < n_reduce_axes; j++) {
      if (reduce_axes[j] == axis_id) { is_reduce_axis = true; break; }
    }
    if (!is_reduce_axis) continue;
    if (!(u->n_src > 0 && u->src[0]->op == POLY_OP_CONST && u->src[0]->arg.kind == POLY_ARG_INT)) continue;
    int64_t bound = u->src[0]->arg.i;
    if (bound <= 1 || bound > 32) continue;

    int n_extra = poly_range_n_extra(u->arg);
    int64_t extra[POLY_MAX_DIMS];
    const int64_t *src_extra = poly_range_extra(u->arg);
    if (n_extra > POLY_MAX_DIMS) n_extra = POLY_MAX_DIMS;
    for (int j = 0; j < n_extra; j++) extra[j] = src_extra[j];
    PolyArg new_arg = poly_arg_range_ex(axis_id, POLY_AXIS_UNROLL, extra, n_extra);
    from[n_sub] = u;
    to[n_sub] = poly_uop1(ctx, POLY_OP_RANGE, u->dtype, u->src[0], new_arg);
    n_sub++;
  }
  if (n_sub == 0) return sink;

  PolyUOp *out = poly_uop_substitute(ctx, sink, from, to, n_sub);
  out = poly_graph_rewrite(ctx, out, poly_symbolic_simple());
  out = poly_graph_rewrite(ctx, out, poly_pm_flatten_range());
  return out;
}

static PolyUOp *poly_apply_opts_basic(PolyCtx *ctx, PolyUOp *sink) {
  sink = poly_unroll_reduce_ranges(ctx, sink);
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, sink, &n_topo);

  /* Keep non-reduce upcast conservative until reduce-side apply_opts parity is complete. */
  for (int i = 0; i < n_topo; i++) {
    if (topo[i]->op == POLY_OP_REDUCE || topo[i]->op == POLY_OP_REDUCE_AXIS)
      return sink;
  }

  AxisRangeRef rngs[POLY_MAX_DIMS * 8];
  int n_rngs = 0;
  int64_t max_axis_id = -1;
  bool has_upcast = false;
  int n_loop_axes = 0;

  for (int i = 0; i < n_topo && n_rngs < (int)(sizeof(rngs) / sizeof(rngs[0])); i++) {
    PolyUOp *u = topo[i];
    if (u->op != POLY_OP_RANGE || !poly_arg_is_range(u->arg)) continue;
    int64_t axis_id = poly_range_axis_id(u->arg);
    PolyAxisType axis_type = poly_range_axis_type(u->arg);
    if (axis_id > max_axis_id) max_axis_id = axis_id;
    if (axis_type == POLY_AXIS_UPCAST || axis_type == POLY_AXIS_UNROLL) has_upcast = true;
    if (axis_type == POLY_AXIS_GLOBAL || axis_type == POLY_AXIS_LOCAL || axis_type == POLY_AXIS_LOOP)
      n_loop_axes++;

    bool dup = false;
    for (int j = 0; j < n_rngs; j++) {
      if (rngs[j].r == u) { dup = true; break; }
    }
    if (!dup) rngs[n_rngs++] = (AxisRangeRef){ .axis_id = axis_id, .axis_type = axis_type, .r = u };
  }
  if (has_upcast || n_rngs == 0) return sink;
  if (n_loop_axes != 1) return sink;

  qsort(rngs, (size_t)n_rngs, sizeof(rngs[0]), axis_range_cmp);

  PolyUOp *target = NULL;
  int64_t target_bound = 0;
  for (int i = n_rngs - 1; i >= 0; i--) {
    PolyAxisType t = rngs[i].axis_type;
    if (!(t == POLY_AXIS_GLOBAL || t == POLY_AXIS_LOCAL || t == POLY_AXIS_LOOP)) continue;
    PolyUOp *r = rngs[i].r;
    if (!(r->n_src > 0 && r->src[0]->op == POLY_OP_CONST && r->src[0]->arg.kind == POLY_ARG_INT)) continue;
    int64_t bound = r->src[0]->arg.i;
    if (bound <= 1) continue;
    if ((bound % 4) != 0) continue;
    target = r;
    target_bound = bound;
    break;
  }
  if (!target) return sink;

  PolyUOp *old_sz = poly_uop0(ctx, POLY_OP_CONST, target->dtype, poly_arg_int(target_bound / 4));
  PolyUOp *replaced = poly_uop1(ctx, POLY_OP_RANGE, target->dtype, old_sz, target->arg);
  PolyUOp *up_sz = poly_uop0(ctx, POLY_OP_CONST, target->dtype, poly_arg_int(4));
  PolyUOp *up = poly_uop1(ctx, POLY_OP_RANGE, target->dtype, up_sz,
                          poly_arg_range(max_axis_id + 1, POLY_AXIS_UPCAST));
  PolyUOp *four = poly_uop0(ctx, POLY_OP_CONST, target->dtype, poly_arg_int(4));
  PolyUOp *sub_axis = poly_uop2(ctx, POLY_OP_ADD, target->dtype,
                                poly_uop2(ctx, POLY_OP_MUL, target->dtype, replaced, four, poly_arg_none()),
                                up, poly_arg_none());

  PolyUOp *from[1] = { target };
  PolyUOp *to[1] = { sub_axis };
  PolyUOp *out = poly_uop_substitute(ctx, sink, from, to, 1);
  out = poly_graph_rewrite(ctx, out, poly_symbolic_simple());
  out = poly_graph_rewrite(ctx, out, poly_pm_flatten_range());
  return out;
}

typedef struct {
  PolyUOp **ranges;
  int n_ranges;
  PolyUOp **ends;
  int n_ends;
  int cap_ends;
} ReduceEndGroup;

typedef struct {
  int acc_num;
  ReduceEndGroup *groups;
  int n_groups;
  int cap_groups;
} ReduceContext;

static ReduceContext *current_reduce_ctx(void) {
  ReduceContext *rctx = (ReduceContext *)poly_graph_rewrite_userctx();
  if (!rctx) {
    fprintf(stderr, "polygrad codegen: pm_reduce requires rewrite ctx (use poly_apply_pm_reduce)\n");
    abort();
  }
  return rctx;
}

static bool same_range_tuple(PolyUOp **a, int na, PolyUOp **b, int nb) {
  if (na != nb) return false;
  for (int i = 0; i < na; i++) if (a[i] != b[i]) return false;
  return true;
}

static void reduce_ctx_clear(ReduceContext *rctx) {
  if (!rctx) return;
  for (int i = 0; i < rctx->n_groups; i++) {
    free(rctx->groups[i].ranges);
    free(rctx->groups[i].ends);
  }
  free(rctx->groups);
  rctx->groups = NULL;
  rctx->n_groups = 0;
  rctx->cap_groups = 0;
  rctx->acc_num = 0;
}

static void reduce_ctx_add_end(ReduceContext *rctx, PolyUOp **ranges, int n_ranges, PolyUOp *end) {
  if (!rctx || !ranges || n_ranges <= 0 || !end) return;

  int gi = -1;
  for (int i = 0; i < rctx->n_groups; i++) {
    if (same_range_tuple(rctx->groups[i].ranges, rctx->groups[i].n_ranges, ranges, n_ranges)) {
      gi = i;
      break;
    }
  }

  if (gi < 0) {
    if (rctx->n_groups >= rctx->cap_groups) {
      rctx->cap_groups = rctx->cap_groups ? rctx->cap_groups * 2 : 8;
      rctx->groups = realloc(rctx->groups, (size_t)rctx->cap_groups * sizeof(ReduceEndGroup));
    }
    gi = rctx->n_groups++;
    rctx->groups[gi] = (ReduceEndGroup){0};
    rctx->groups[gi].ranges = malloc((size_t)n_ranges * sizeof(PolyUOp *));
    memcpy(rctx->groups[gi].ranges, ranges, (size_t)n_ranges * sizeof(PolyUOp *));
    rctx->groups[gi].n_ranges = n_ranges;
  }

  ReduceEndGroup *g = &rctx->groups[gi];
  if (g->n_ends >= g->cap_ends) {
    g->cap_ends = g->cap_ends ? g->cap_ends * 2 : 4;
    g->ends = realloc(g->ends, (size_t)g->cap_ends * sizeof(PolyUOp *));
  }
  g->ends[g->n_ends++] = end;
}

/*
 * rule_reduce_to_acc — Port of tinygrad's reduce_to_acc.
 *
 * Transforms REDUCE(reduce_op, value, reduce_range_0, ...) into:
 *   DEFINE_REG(acc_id)
 *   acc.after(input_ranges...).index(0).store(identity)   [init]
 *   acc.after(init, reduce_ranges...).index(0)             [loop read + LOAD]
 *   reduce_op(loop_read, value)                            [accumulate]
 *   acc.index(0).store(result).end(reduce_ranges...)       [finalize]
 *   acc.after(end).index(0)                                [final read + LOAD]
 */
static PolyUOp *rule_reduce_to_acc(PolyCtx *ctx, PolyUOp *root,
                                   const PolyBindings *b) {
  (void)b;
  PolyUOp *red = root;
  PolyUOp *inp = red->src[0];
  int n_reduce_range = red->n_src - 1;
  PolyOps reduce_op = red->arg.ops;
  PolyUOp *lst[128];
  int n_lst = horizontal_reduce_terms(ctx, inp, red->dtype, lst, 128);
  if (n_lst <= 0) return inp;

  /* Horizontal-only reduce (no loop ranges). */
  if (n_reduce_range == 0) {
    PolyUOp *ret = lst[0];
    for (int i = 1; i < n_lst; i++)
      ret = poly_uop2(ctx, reduce_op, red->dtype, ret, lst[i], poly_arg_none());
    return ret;
  }

  /* ── Find input_ranges (outer loops the value depends on) ──────────── */
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, inp, &n_topo);

  /* tinygrad parity: precompute ended ranges once from END nodes,
   * instead of rescanning all ENDs for every RANGE. */
  PolyMap *ended_ranges = poly_map_new((size_t)(n_topo > 0 ? n_topo * 2 : 16));
  if (ended_ranges) {
    for (int i = 0; i < n_topo; i++) {
      PolyUOp *u = topo[i];
      if (u->op != POLY_OP_END) continue;
      for (int s = 1; s < u->n_src; s++) {
        PolyUOp *r = u->src[s];
        if (!r || r->op != POLY_OP_RANGE) continue;
        poly_map_set(ended_ranges, ptr_hash(r), r, r, ptr_eq);
      }
    }
  }

  PolyUOp *input_ranges[POLY_MAX_DIMS];
  int n_input_ranges = 0;
  for (int i = 0; i < n_topo; i++) {
    if (topo[i]->op != POLY_OP_RANGE) continue;
    /* Skip reduce ranges */
    bool is_reduce = false;
    for (int j = 0; j < n_reduce_range; j++) {
      if (topo[i] == red->src[j + 1]) { is_reduce = true; break; }
    }
    /* Skip already-ended ranges */
    if (!is_reduce) {
      bool is_ended = false;
      if (ended_ranges) {
        is_ended = poly_map_get(ended_ranges, ptr_hash(topo[i]), topo[i], ptr_eq) != NULL;
      } else {
        for (int k = 0; k < n_topo; k++) {
          if (topo[k]->op != POLY_OP_END) continue;
          for (int s = 1; s < topo[k]->n_src; s++) {
            if (topo[k]->src[s] == topo[i]) { is_ended = true; break; }
          }
          if (is_ended) break;
        }
      }
      if (!is_ended && n_input_ranges < POLY_MAX_DIMS)
        input_ranges[n_input_ranges++] = topo[i];
    }
  }
  if (ended_ranges) poly_map_destroy(ended_ranges);
  /* topo is arena-allocated, no free needed */


  /* ── Identity element ──────────────────────────────────────────────── */
  double ident_val = codegen_reduce_identity(reduce_op);
  PolyUOp *identity;
  if (poly_dtype_is_float(red->dtype))
    identity = poly_uop0(ctx, POLY_OP_CONST, red->dtype,
                          poly_arg_float(ident_val));
  else
    identity = poly_uop0(ctx, POLY_OP_CONST, red->dtype,
                          poly_arg_int((int64_t)ident_val));

  ReduceContext *rctx = current_reduce_ctx();

  /* ── DEFINE_REG: accumulator register ──────────────────────────────── */
  int acc_id = rctx->acc_num++;
  PolyDType acc_ptr = poly_dtype_ptr(red->dtype, 1, POLY_ADDR_REG);
  PolyUOp *acc = poly_uop0(ctx, POLY_OP_DEFINE_REG, acc_ptr,
                           poly_arg_int(acc_id));

  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));

  /* ── Init: acc.after(input_ranges...).index(0).store(identity) ─────── */
  PolyUOp *acc_base;
  if (n_input_ranges > 0) {
    PolyUOp *after_srcs[POLY_MAX_DIMS + 1];
    after_srcs[0] = acc;
    for (int i = 0; i < n_input_ranges; i++)
      after_srcs[i + 1] = input_ranges[i];
    acc_base = poly_uop(ctx, POLY_OP_AFTER, acc_ptr,
                         after_srcs, n_input_ranges + 1, poly_arg_none());
  } else {
    acc_base = acc;
  }
  PolyUOp *init_idx = poly_uop2(ctx, POLY_OP_INDEX, acc_ptr,
                                acc_base, zero, poly_arg_none());
  PolyUOp *acc_init = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID,
                                init_idx, identity, poly_arg_none());

  /* ── Loop read: acc.after(init, reduce_ranges...).index(0) + LOAD ─── */
  PolyUOp *loop_srcs[POLY_MAX_DIMS + 2];
  loop_srcs[0] = acc;
  loop_srcs[1] = acc_init;
  for (int i = 0; i < n_reduce_range; i++)
    loop_srcs[i + 2] = red->src[i + 1];
  PolyUOp *loop_after = poly_uop(ctx, POLY_OP_AFTER, acc_ptr,
                                 loop_srcs, n_reduce_range + 2,
                                 poly_arg_none());
  PolyUOp *loop_idx = poly_uop2(ctx, POLY_OP_INDEX, acc_ptr,
                                loop_after, zero, poly_arg_none());
  PolyUOp *loop_load = poly_uop1(ctx, POLY_OP_LOAD, red->dtype,
                                 loop_idx, poly_arg_none());

  /* ── Accumulate: reduce_op(loop_load, horizontal_reduce(inp)) ──────── */
  PolyUOp *alu = loop_load;
  for (int i = 0; i < n_lst; i++)
    alu = poly_uop2(ctx, reduce_op, red->dtype, alu, lst[i], poly_arg_none());

  /* ── Store back + END: acc.index(0).store(alu).end(reduce_ranges...) ─ */
  PolyUOp *store_idx = poly_uop2(ctx, POLY_OP_INDEX, acc_ptr,
                                 acc, zero, poly_arg_none());
  PolyUOp *acc_store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID,
                                 store_idx, alu, poly_arg_none());

  /* Build END chain (innermost first to match tinygrad) */
  PolyUOp *chain = acc_store;
  for (int i = n_reduce_range - 1; i >= 0; i--) {
    PolyUOp *end_srcs[2] = { chain, red->src[i + 1] };
    chain = poly_uop(ctx, POLY_OP_END, POLY_VOID,
                      end_srcs, 2, poly_arg_none());
  }
  reduce_ctx_add_end(rctx, &red->src[1], n_reduce_range, chain);

  /* ── Final read: acc.after(end).index(0) + LOAD ────────────────────── */
  PolyUOp *final_srcs[2] = { acc, chain };
  PolyUOp *final_after = poly_uop(ctx, POLY_OP_AFTER, acc_ptr,
                                  final_srcs, 2, poly_arg_none());
  PolyUOp *final_idx = poly_uop2(ctx, POLY_OP_INDEX, acc_ptr,
                                 final_after, zero, poly_arg_none());
  PolyUOp *final_load = poly_uop1(ctx, POLY_OP_LOAD, red->dtype,
                                  final_idx, poly_arg_none());

  return final_load;
}

static PolyUOp *rule_merge_reduce_ends(PolyCtx *ctx, PolyUOp *root,
                                       const PolyBindings *b) {
  (void)b;
  if (!root || root->op != POLY_OP_SINK) return NULL;
  ReduceContext *rctx = current_reduce_ctx();

  int n_subs = 0;
  for (int i = 0; i < rctx->n_groups; i++)
    if (rctx->groups[i].n_ends > 1) n_subs += rctx->groups[i].n_ends;
  if (n_subs == 0) return NULL;

  PolyUOp **sub_old = malloc((size_t)n_subs * sizeof(PolyUOp *));
  PolyUOp **sub_new = malloc((size_t)n_subs * sizeof(PolyUOp *));
  int at = 0;

  for (int i = 0; i < rctx->n_groups; i++) {
    ReduceEndGroup *g = &rctx->groups[i];
    if (g->n_ends <= 1) continue;

    PolyUOp **group_srcs = malloc((size_t)g->n_ends * sizeof(PolyUOp *));
    for (int j = 0; j < g->n_ends; j++) group_srcs[j] = g->ends[j]->src[0];
    PolyUOp *chain = poly_uop(ctx, POLY_OP_GROUP, POLY_VOID,
                              group_srcs, g->n_ends, poly_arg_none());
    free(group_srcs);

    for (int r = g->n_ranges - 1; r >= 0; r--) {
      PolyUOp *esrc[2] = { chain, g->ranges[r] };
      chain = poly_uop(ctx, POLY_OP_END, POLY_VOID, esrc, 2, poly_arg_none());
    }

    for (int j = 0; j < g->n_ends; j++) {
      sub_old[at] = g->ends[j];
      sub_new[at] = chain;
      at++;
    }
  }

  PolyUOp *out = root;
  for (int i = 0; i < at; i++) {
    PolyUOp *memo_old[4096];
    PolyUOp *memo_new[4096];
    int memo_n = 0;
    out = substitute_node(ctx, out, sub_old[i], sub_new[i], memo_old, memo_new, &memo_n, 4096);
  }

  free(sub_new);
  free(sub_old);
  return out != root ? out : NULL;
}

/* ── Build the pm_reduce PatternMatcher ──────────────────────────────── */

static PolyPatternMatcher *g_pm_reduce = NULL;

static PolyPatternMatcher *poly_pm_reduce(void) {
  if (g_pm_reduce) return g_pm_reduce;

  PolyOpSet reduce_set = poly_opset_add((PolyOpSet){{0,0}}, POLY_OP_REDUCE);
  PolyOpSet sink_set = poly_opset_add((PolyOpSet){{0,0}}, POLY_OP_SINK);
  PolyRule rules[] = {
    { poly_pat_ops(reduce_set, NULL, 0, NULL), rule_reduce_to_acc },
    { poly_pat_ops(sink_set, NULL, 0, NULL), rule_merge_reduce_ends },
  };
  g_pm_reduce = poly_pm_new(rules, 2);
  return g_pm_reduce;
}

/* ── pm_decomp: late decompositions (MAX → CMPLT+WHERE, etc.) ───────── */

/*
 * rule_decomp_max — Port of tinygrad's get_late_rewrite_patterns MAX rule.
 * MAX(a, b) → WHERE(CMPLT(a, b), b, a)
 * ClangRenderer doesn't have native MAX, so decompose to CMPLT+WHERE.
 */
static PolyUOp *rule_decomp_max(PolyCtx *ctx, PolyUOp *root,
                                const PolyBindings *b) {
  (void)b;
  PolyUOp *a = root->src[0];
  PolyUOp *cmp = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, a, root->src[1],
                           poly_arg_none());
  return poly_uop3(ctx, POLY_OP_WHERE, root->dtype, cmp, root->src[1], a,
                    poly_arg_none());
}

/*
 * rule_mul_to_shl — Port of tinygrad's get_late_rewrite_patterns MUL→SHL rule.
 * x * c → SHL(x, log2(c))  when c is a power of 2 and x is integer type.
 */
static PolyUOp *rule_mul_to_shl(PolyCtx *ctx, PolyUOp *root,
                                const PolyBindings *b) {
  PolyUOp *c_node = poly_bind(b, "c");
  PolyUOp *x_node = poly_bind(b, "x");
  if (!c_node || !x_node) return NULL;
  if (!poly_dtype_is_int(root->dtype)) return NULL;
  if (c_node->arg.kind != POLY_ARG_INT) return NULL;
  int64_t c = c_node->arg.i;
  if (c <= 0 || (c & (c - 1)) != 0) return NULL;  /* not a power of 2 */
  int shift = 0;
  int64_t tmp = c;
  while (tmp > 1) { shift++; tmp >>= 1; }
  PolyUOp *shift_const = poly_uop0(ctx, POLY_OP_CONST, root->dtype,
                                   poly_arg_int(shift));
  return poly_uop2(ctx, POLY_OP_SHL, root->dtype, x_node, shift_const,
                    poly_arg_none());
}

/*
 * rule_idiv_to_shr — Port of tinygrad's get_late_rewrite_patterns IDIV→SHR rule.
 * x // c → SHR(x, log2(c))  when c is a power of 2.
 * For signed ints: (x + (x<0).where(c-1, 0)) >> log2(c)
 */
static PolyUOp *rule_idiv_to_shr(PolyCtx *ctx, PolyUOp *root,
                                  const PolyBindings *b) {
  PolyUOp *c_node = poly_bind(b, "c");
  PolyUOp *x_node = poly_bind(b, "x");
  if (!c_node || !x_node) return NULL;
  if (!poly_dtype_is_int(root->dtype)) return NULL;
  if (c_node->arg.kind != POLY_ARG_INT) return NULL;
  int64_t c = c_node->arg.i;
  if (c <= 0 || (c & (c - 1)) != 0) return NULL;  /* not a power of 2 */
  int shift = 0;
  int64_t tmp = c;
  while (tmp > 1) { shift++; tmp >>= 1; }
  PolyUOp *shift_const = poly_uop0(ctx, POLY_OP_CONST, root->dtype,
                                   poly_arg_int(shift));
  /* Unsigned: just shift right */
  if (poly_dtype_is_unsigned(root->dtype))
    return poly_uop2(ctx, POLY_OP_SHR, root->dtype, x_node, shift_const,
                      poly_arg_none());
  /* Signed: (x + (x<0).where(c-1, 0)) >> shift */
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, root->dtype, poly_arg_int(0));
  PolyUOp *cmplt = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL,
                              x_node, zero, poly_arg_none());
  PolyUOp *cm1 = poly_uop0(ctx, POLY_OP_CONST, root->dtype, poly_arg_int(c - 1));
  PolyUOp *correction = poly_uop3(ctx, POLY_OP_WHERE, root->dtype,
                                   cmplt, cm1, zero, poly_arg_none());
  PolyUOp *corrected = poly_uop2(ctx, POLY_OP_ADD, root->dtype,
                                  x_node, correction, poly_arg_none());
  return poly_uop2(ctx, POLY_OP_SHR, root->dtype, corrected, shift_const,
                    poly_arg_none());
}

/*
 * rule_mod_to_and — Port of tinygrad's MOD→AND rule.
 * x % c → x & (c-1)  when c is a power of 2 and x is integer type.
 */
static PolyUOp *rule_mod_to_and(PolyCtx *ctx, PolyUOp *root,
                                const PolyBindings *b) {
  PolyUOp *c_node = poly_bind(b, "c");
  PolyUOp *x_node = poly_bind(b, "x");
  if (!c_node || !x_node) return NULL;
  if (!poly_dtype_is_int(root->dtype)) return NULL;
  if (c_node->arg.kind != POLY_ARG_INT) return NULL;
  int64_t c = c_node->arg.i;
  if (c <= 0 || (c & (c - 1)) != 0) return NULL;  /* not a power of 2 */
  PolyUOp *mask = poly_uop0(ctx, POLY_OP_CONST, root->dtype,
                             poly_arg_int(c - 1));
  return poly_uop2(ctx, POLY_OP_AND, root->dtype, x_node, mask,
                    poly_arg_none());
}

/*
 * rule_mulacc_to_mul_add — MULACC(a, b, c) → ADD(MUL(a, b), c)
 * For renderers without native FMA.  Currently unconditional; CUDA should
 * keep MULACC once renderer capabilities are threaded through PolyRewriteOpts.
 */
static PolyUOp *rule_mulacc_to_mul_add(PolyCtx *ctx, PolyUOp *root,
                                        const PolyBindings *b) {
  (void)b;
  if (root->op != POLY_OP_MULACC || root->n_src != 3) return NULL;
  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, root->dtype,
                             root->src[0], root->src[1], poly_arg_none());
  return poly_uop2(ctx, POLY_OP_ADD, root->dtype,
                    mul, root->src[2], poly_arg_none());
}

/*
 * rule_mul_neg1_to_neg — Port of tinygrad late rewrite:
 * x * (-1) → NEG(x)
 */
static PolyUOp *rule_mul_neg1_to_neg(PolyCtx *ctx, PolyUOp *root,
                                     const PolyBindings *b) {
  PolyUOp *x = poly_bind(b, "x");
  PolyUOp *c = poly_bind(b, "c");
  if (!x || !c) return NULL;
  if (c->op != POLY_OP_CONST) return NULL;
  if (c->arg.kind == POLY_ARG_INT && c->arg.i == -1)
    return poly_uop1(ctx, POLY_OP_NEG, root->dtype, x, poly_arg_none());
  if (c->arg.kind == POLY_ARG_FLOAT && c->arg.f == -1.0)
    return poly_uop1(ctx, POLY_OP_NEG, root->dtype, x, poly_arg_none());
  return NULL;
}

/*
 * rule_add_neg_to_sub — Port of tinygrad late rewrite:
 * x + NEG(y) → SUB(x, y)
 */
static PolyUOp *rule_add_neg_to_sub(PolyCtx *ctx, PolyUOp *root,
                                    const PolyBindings *b) {
  PolyUOp *x = poly_bind(b, "x");
  PolyUOp *y = poly_bind(b, "y");
  if (!x || !y) return NULL;
  return poly_uop2(ctx, POLY_OP_SUB, root->dtype, x, y, poly_arg_none());
}

/*
 * rule_recip_to_fdiv — Port of tinygrad late rewrite:
 * RECIPROCAL(x) → FDIV(1, x)
 */
static PolyUOp *rule_recip_to_fdiv(PolyCtx *ctx, PolyUOp *root,
                                   const PolyBindings *b) {
  PolyUOp *x = poly_bind(b, "x");
  if (!x || !poly_dtype_is_float(root->dtype)) return NULL;
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, root->dtype, poly_arg_float(1.0));
  return poly_uop2(ctx, POLY_OP_FDIV, root->dtype, one, x, poly_arg_none());
}

/*
 * rule_mul_fdiv1_to_fdiv — Port of tinygrad late rewrite:
 * a * (1 / b) → a / b
 */
static PolyUOp *rule_mul_fdiv1_to_fdiv(PolyCtx *ctx, PolyUOp *root,
                                       const PolyBindings *b) {
  PolyUOp *a = poly_bind(b, "a");
  PolyUOp *bnode = poly_bind(b, "b");
  PolyUOp *one = poly_bind(b, "one");
  if (!a || !bnode || !one) return NULL;
  if (!poly_dtype_is_float(root->dtype)) return NULL;
  if (one->op != POLY_OP_CONST) return NULL;
  if (one->arg.kind == POLY_ARG_FLOAT && one->arg.f == 1.0)
    return poly_uop2(ctx, POLY_OP_FDIV, root->dtype, a, bnode, poly_arg_none());
  if (one->arg.kind == POLY_ARG_INT && one->arg.i == 1)
    return poly_uop2(ctx, POLY_OP_FDIV, root->dtype, a, bnode, poly_arg_none());
  return NULL;
}

static PolyPatternMatcher *g_pm_decomp = NULL;

static PolyPatternMatcher *poly_pm_decomp(void) {
  if (g_pm_decomp) return g_pm_decomp;

  PolyOpSet max_set = poly_opset_add((PolyOpSet){{0,0}}, POLY_OP_MAX);
  PolyRule rules[] = {
    { poly_pat_ops(max_set, NULL, 0, NULL), rule_decomp_max },
    /* MUL(x:int, c:const) → SHL(x, log2(c)) when c is power of 2 */
    { poly_pat_op2(POLY_OP_MUL, poly_pat_any("x"), poly_pat_cvar("c"), NULL),
      rule_mul_to_shl },
    /* x * (-1) → NEG(x) */
    { poly_pat_op2(POLY_OP_MUL, poly_pat_any("x"), poly_pat_cvar("c"), NULL),
      rule_mul_neg1_to_neg },
    /* IDIV(x:int, c:const) → SHR(x, log2(c)) when c is power of 2 */
    { poly_pat_op2(POLY_OP_IDIV, poly_pat_any("x"), poly_pat_cvar("c"), NULL),
      rule_idiv_to_shr },
    /* MOD(x:int, c:const) → AND(x, c-1) when c is power of 2 */
    { poly_pat_op2(POLY_OP_MOD, poly_pat_any("x"), poly_pat_cvar("c"), NULL),
      rule_mod_to_and },
    /* x + NEG(y) → SUB(x, y) */
    { poly_pat_op2(POLY_OP_ADD, poly_pat_any("x"),
        poly_pat_op1(POLY_OP_NEG, poly_pat_any("y"), NULL), NULL),
      rule_add_neg_to_sub },
    /* MULACC(a, b, c) → ADD(MUL(a, b), c) */
    { poly_pat_ops(poly_opset_add((PolyOpSet){{0,0}}, POLY_OP_MULACC),
        NULL, 0, NULL), rule_mulacc_to_mul_add },
    /* RECIPROCAL(x) → FDIV(1, x) */
    { poly_pat_op1(POLY_OP_RECIPROCAL, poly_pat_any("x"), NULL),
      rule_recip_to_fdiv },
    /* a * (1 / b) → a / b */
    { poly_pat_op2(POLY_OP_MUL, poly_pat_any("a"),
        poly_pat_op2(POLY_OP_FDIV, poly_pat_cvar("one"),
          poly_pat_any("b"), NULL), NULL),
      rule_mul_fdiv1_to_fdiv },
  };
  g_pm_decomp = poly_pm_new(rules, sizeof(rules) / sizeof(rules[0]));
  return g_pm_decomp;
}

/* ── pm_transcendental: EXP2 → polynomial approximation ──────────────── */

/*
 * rule_decomp_exp2 — Port of tinygrad's xexp2 from decompositions.py.
 *
 * EXP2(d) → polynomial approximation with IEEE 754 bit manipulation.
 * Only handles float32 (the only transcendental dtype in CPU parity tests).
 *
 * Algorithm (float32):
 *   1. _lazy_map_numbers: mask ±inf/NaN to 0
 *   2. rintk: round x to nearest integer q
 *   3. s = x - q (fractional part)
 *   4. polyN: 7-coeff Chebyshev polynomial on s ∈ [-0.5, 0.5]
 *   5. ldexp2k: multiply by 2^q via IEEE 754 exponent construction
 *   6. Edge cases: overflow→inf, underflow→0, NaN→NaN
 */
static PolyUOp *rule_decomp_exp2(PolyCtx *ctx, PolyUOp *root,
                                  const PolyBindings *b) {
  (void)b;
  PolyUOp *d = root->src[0];
  /* Only decompose float32 for now */
  if (root->dtype.bitsize != 32 || !poly_dtype_is_float(root->dtype))
    return NULL;

  PolyDType ft = root->dtype;  /* float32 */
  PolyDType it = POLY_INT32;
  PolyDType bt = POLY_BOOL;

  /* ── Constants ───────────────────────────────────────────────────── */
  PolyUOp *f_zero     = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(0.0));
  PolyUOp *f_neg_half = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(-0.5));
  PolyUOp *f_half     = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(0.5));
  PolyUOp *f_neg_inf  = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(-__builtin_inf()));
  PolyUOp *f_pos_inf  = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(__builtin_inf()));
  PolyUOp *f_nan      = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(__builtin_nan("")));
  PolyUOp *f_neg150   = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(-150.0));
  PolyUOp *f_128      = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(128.0));
  PolyUOp *b_true     = poly_uop0(ctx, POLY_OP_CONST, bt, poly_arg_bool(true));
  PolyUOp *i_23       = poly_uop0(ctx, POLY_OP_CONST, it, poly_arg_int(23));
  PolyUOp *i_127      = poly_uop0(ctx, POLY_OP_CONST, it, poly_arg_int(127));

  /* Float32 minimax polynomial coefficients */
  static const double coeffs[] = {
    0.1535920892e-3, 0.1339262701e-2, 0.9618384764e-2,
    0.5550347269e-1, 0.2402264476e+0, 0.6931471825e+0, 1.0
  };
  PolyUOp *f_c[7];
  for (int i = 0; i < 7; i++)
    f_c[i] = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(coeffs[i]));

  /* ── Step 1: _lazy_map_numbers — mask ±inf/NaN to 0 ────────────── */
  /* x = d.ne(inf).where(d.ne(d).where(0, d.ne(-inf).where(d, 0)), 0) */
  PolyUOp *nan_chk    = poly_uop2(ctx, POLY_OP_CMPNE, bt, d, d, poly_arg_none());
  PolyUOp *neginf_chk = poly_uop2(ctx, POLY_OP_CMPNE, bt, d, f_neg_inf,
                                   poly_arg_none());
  PolyUOp *posinf_chk = poly_uop2(ctx, POLY_OP_CMPNE, bt, d, f_pos_inf,
                                   poly_arg_none());
  PolyUOp *inner = poly_uop3(ctx, POLY_OP_WHERE, ft,
                              neginf_chk, d, f_zero, poly_arg_none());
  PolyUOp *mid   = poly_uop3(ctx, POLY_OP_WHERE, ft,
                              nan_chk, f_zero, inner, poly_arg_none());
  PolyUOp *x     = poly_uop3(ctx, POLY_OP_WHERE, ft,
                              posinf_chk, mid, f_zero, poly_arg_none());

  /* ── Step 2: rintk — round to nearest integer ──────────────────── */
  /* q = (x + (x<0).where(-0.5, 0.5)).cast(int32)                    */
  PolyUOp *x_lt0    = poly_uop2(ctx, POLY_OP_CMPLT, bt, x, f_zero,
                                 poly_arg_none());
  PolyUOp *rounding = poly_uop3(ctx, POLY_OP_WHERE, ft,
                                 x_lt0, f_neg_half, f_half, poly_arg_none());
  PolyUOp *rounded  = poly_uop2(ctx, POLY_OP_ADD, ft, x, rounding,
                                 poly_arg_none());
  PolyUOp *q        = poly_uop1(ctx, POLY_OP_CAST, it, rounded,
                                 poly_arg_none());

  /* ── Step 3: fractional part s = x - q.cast(float) ────────────── */
  PolyUOp *q_float = poly_uop1(ctx, POLY_OP_CAST, ft, q, poly_arg_none());
  PolyUOp *s       = poly_uop2(ctx, POLY_OP_SUB, ft, x, q_float,
                                poly_arg_none());

  /* ── Step 4: polyN — Horner's method (7 coefficients) ──────────── */
  /* reduce(lambda acc,c: acc*x+c, coeffs, 0.0)                       */
  /* First iter: 0.0*s + c0 = c0 (constant folded, so start at c0)    */
  PolyUOp *u = f_c[0];
  for (int i = 1; i < 7; i++) {
    u = poly_uop2(ctx, POLY_OP_MUL, ft, u, s, poly_arg_none());
    u = poly_uop2(ctx, POLY_OP_ADD, ft, u, f_c[i], poly_arg_none());
  }

  /* ── Step 5: ldexp2k — u * 2^q ─────────────────────────────────── */
  /* shr(q, 1) = q // 2. Emitted as IDIV; pm_decomp converts to SHR. */
  PolyUOp *i_two   = poly_uop0(ctx, POLY_OP_CONST, it, poly_arg_int(2));
  PolyUOp *half_q  = poly_uop2(ctx, POLY_OP_IDIV, it, q, i_two,
                                poly_arg_none());
  PolyUOp *other_q = poly_uop2(ctx, POLY_OP_SUB, it, q, half_q,
                                poly_arg_none());
  /* pow2if(half_q) = ((half_q + 127) << 23).bitcast(float) */
  PolyUOp *p1_add = poly_uop2(ctx, POLY_OP_ADD, it, half_q, i_127,
                               poly_arg_none());
  PolyUOp *p1_shl = poly_uop2(ctx, POLY_OP_SHL, it, p1_add, i_23,
                               poly_arg_none());
  PolyUOp *pow1   = poly_uop1(ctx, POLY_OP_BITCAST, ft, p1_shl,
                               poly_arg_none());
  /* pow2if(other_q) = ((other_q + 127) << 23).bitcast(float) */
  PolyUOp *p2_add = poly_uop2(ctx, POLY_OP_ADD, it, other_q, i_127,
                               poly_arg_none());
  PolyUOp *p2_shl = poly_uop2(ctx, POLY_OP_SHL, it, p2_add, i_23,
                               poly_arg_none());
  PolyUOp *pow2   = poly_uop1(ctx, POLY_OP_BITCAST, ft, p2_shl,
                               poly_arg_none());
  /* result = u * pow1 * pow2 */
  PolyUOp *result = poly_uop2(ctx, POLY_OP_MUL, ft, u, pow1, poly_arg_none());
  result = poly_uop2(ctx, POLY_OP_MUL, ft, result, pow2, poly_arg_none());

  /* ── Step 6: edge cases ─────────────────────────────────────────── */
  /* (d >= 128).where(inf, u) = CMPNE(CMPLT(d, 128), true).where(inf, u) */
  PolyUOp *cmp_hi = poly_uop2(ctx, POLY_OP_CMPLT, bt, d, f_128,
                               poly_arg_none());
  PolyUOp *ge_hi  = poly_uop2(ctx, POLY_OP_CMPNE, bt, cmp_hi, b_true,
                               poly_arg_none());
  result = poly_uop3(ctx, POLY_OP_WHERE, ft,
                       ge_hi, f_pos_inf, result, poly_arg_none());
  /* (d < -150).where(0, u) */
  PolyUOp *cmp_lo = poly_uop2(ctx, POLY_OP_CMPLT, bt, d, f_neg150,
                               poly_arg_none());
  result = poly_uop3(ctx, POLY_OP_WHERE, ft,
                       cmp_lo, f_zero, result, poly_arg_none());
  /* d.ne(d).where(nan, u) — nan_chk from step 1 reused via CSE */
  result = poly_uop3(ctx, POLY_OP_WHERE, ft,
                       nan_chk, f_nan, result, poly_arg_none());

  return result;
}

/*
 * rule_decomp_log2 — Port of tinygrad's xlog2 for float32.
 *
 * LOG2(d) → polynomial + IEEE754 exponent/mantissa manipulation.
 */
static PolyUOp *rule_decomp_log2(PolyCtx *ctx, PolyUOp *root,
                                  const PolyBindings *b) {
  (void)b;
  PolyUOp *d = root->src[0];
  if (root->dtype.bitsize != 32 || !poly_dtype_is_float(root->dtype))
    return NULL;

  PolyDType ft = root->dtype;
  PolyDType it = POLY_INT32;
  PolyDType bt = POLY_BOOL;

  /* Constants */
  PolyUOp *f_zero    = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(0.0));
  PolyUOp *f_neg_zero = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(-0.0));
  PolyUOp *f_one     = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(1.0));
  PolyUOp *f_neg_inf = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(-__builtin_inf()));
  PolyUOp *f_pos_inf = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(__builtin_inf()));
  PolyUOp *f_nan     = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(__builtin_nan("")));
  PolyUOp *f_1e4     = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(1e-4));
  PolyUOp *f_4_3     = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(1.0 / 0.75));
  PolyUOp *f_64      = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(64.0));
  PolyUOp *f_2p64    = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(18446744073709551616.0));
  PolyUOp *f_k1      = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(2.8853900432586669922));
  PolyUOp *f_k2      = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(3.2734474483568488616e-08));
  PolyUOp *i_23      = poly_uop0(ctx, POLY_OP_CONST, it, poly_arg_int(23));
  PolyUOp *i_255     = poly_uop0(ctx, POLY_OP_CONST, it, poly_arg_int(255));
  PolyUOp *i_127     = poly_uop0(ctx, POLY_OP_CONST, it, poly_arg_int(127));

  /* Denormal handling */
  PolyUOp *is_denormal = poly_uop2(ctx, POLY_OP_CMPLT, bt, d, f_1e4, poly_arg_none());
  PolyUOp *scaled = poly_uop2(ctx, POLY_OP_MUL, ft, d, f_2p64, poly_arg_none());
  PolyUOp *a = poly_uop3(ctx, POLY_OP_WHERE, ft, is_denormal, scaled, d, poly_arg_none());

  /* e = ilogb2k(a * (1/0.75)) */
  PolyUOp *a_scaled = poly_uop2(ctx, POLY_OP_MUL, ft, a, f_4_3, poly_arg_none());
  PolyUOp *a_bits = poly_uop1(ctx, POLY_OP_BITCAST, it, a_scaled, poly_arg_none());
  PolyUOp *exp_bits = poly_uop2(ctx, POLY_OP_SHR, it, a_bits, i_23, poly_arg_none());
  PolyUOp *exp_masked = poly_uop2(ctx, POLY_OP_AND, it, exp_bits, i_255, poly_arg_none());
  PolyUOp *e_int = poly_uop2(ctx, POLY_OP_SUB, it, exp_masked, i_127, poly_arg_none());
  PolyUOp *e = poly_uop1(ctx, POLY_OP_CAST, ft, e_int, poly_arg_none());

  /* m = ldexp3k(a, -e) */
  PolyUOp *neg_e = poly_uop1(ctx, POLY_OP_NEG, ft, e, poly_arg_none());
  PolyUOp *neg_e_i = poly_uop1(ctx, POLY_OP_CAST, it, neg_e, poly_arg_none());
  PolyUOp *e_shift = poly_uop2(ctx, POLY_OP_SHL, it, neg_e_i, i_23, poly_arg_none());
  PolyUOp *a_raw_bits = poly_uop1(ctx, POLY_OP_BITCAST, it, a, poly_arg_none());
  PolyUOp *m_bits = poly_uop2(ctx, POLY_OP_ADD, it, a_raw_bits, e_shift, poly_arg_none());
  PolyUOp *m = poly_uop1(ctx, POLY_OP_BITCAST, ft, m_bits, poly_arg_none());

  /* Denormal exponent correction */
  PolyUOp *e_minus64 = poly_uop2(ctx, POLY_OP_SUB, ft, e, f_64, poly_arg_none());
  PolyUOp *e_adj = poly_uop3(ctx, POLY_OP_WHERE, ft, is_denormal, e_minus64, e, poly_arg_none());

  /* x = (m - 1) / (m + 1) */
  PolyUOp *m_minus1 = poly_uop2(ctx, POLY_OP_SUB, ft, m, f_one, poly_arg_none());
  PolyUOp *m_plus1  = poly_uop2(ctx, POLY_OP_ADD, ft, m, f_one, poly_arg_none());
  PolyUOp *x = poly_uop2(ctx, POLY_OP_FDIV, ft, m_minus1, m_plus1, poly_arg_none());
  PolyUOp *x2 = poly_uop2(ctx, POLY_OP_MUL, ft, x, x, poly_arg_none());

  /* t = polyN(x2, [0.4374550283, 0.5764790177, 0.9618012905120]) */
  PolyUOp *t = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(0.4374550283));
  t = poly_uop2(ctx, POLY_OP_MUL, ft, t, x2, poly_arg_none());
  t = poly_uop2(ctx, POLY_OP_ADD, ft, t, poly_uop0(ctx, POLY_OP_CONST, ft,
                                                     poly_arg_float(0.5764790177)),
                poly_arg_none());
  t = poly_uop2(ctx, POLY_OP_MUL, ft, t, x2, poly_arg_none());
  t = poly_uop2(ctx, POLY_OP_ADD, ft, t, poly_uop0(ctx, POLY_OP_CONST, ft,
                                                     poly_arg_float(0.9618012905120)),
                poly_arg_none());

  PolyUOp *xx2 = poly_uop2(ctx, POLY_OP_MUL, ft, x, x2, poly_arg_none());
  PolyUOp *r = poly_uop2(ctx, POLY_OP_MUL, ft, t, xx2, poly_arg_none());
  r = poly_uop2(ctx, POLY_OP_ADD, ft, r, e_adj, poly_arg_none());
  r = poly_uop2(ctx, POLY_OP_ADD, ft, r,
                poly_uop2(ctx, POLY_OP_MUL, ft, x, f_k1, poly_arg_none()),
                poly_arg_none());
  r = poly_uop2(ctx, POLY_OP_ADD, ft, r,
                poly_uop2(ctx, POLY_OP_MUL, ft, x, f_k2, poly_arg_none()),
                poly_arg_none());

  /* Edge cases */
  PolyUOp *ne_inf = poly_uop2(ctx, POLY_OP_CMPNE, bt, d, f_pos_inf, poly_arg_none());
  r = poly_uop3(ctx, POLY_OP_WHERE, ft, ne_inf, r, f_pos_inf, poly_arg_none());
  PolyUOp *ne_zero = poly_uop2(ctx, POLY_OP_CMPNE, bt, d, f_zero, poly_arg_none());
  r = poly_uop3(ctx, POLY_OP_WHERE, ft, ne_zero, r, f_neg_inf, poly_arg_none());
  PolyUOp *lt_neg_zero = poly_uop2(ctx, POLY_OP_CMPLT, bt, d, f_neg_zero, poly_arg_none());
  r = poly_uop3(ctx, POLY_OP_WHERE, ft, lt_neg_zero, f_nan, r, poly_arg_none());
  PolyUOp *is_nan = poly_uop2(ctx, POLY_OP_CMPNE, bt, d, d, poly_arg_none());
  r = poly_uop3(ctx, POLY_OP_WHERE, ft, is_nan, f_nan, r, poly_arg_none());
  PolyUOp *rec = poly_uop1(ctx, POLY_OP_RECIPROCAL, ft, d, poly_arg_none());
  PolyUOp *rec_ne_ninf = poly_uop2(ctx, POLY_OP_CMPNE, bt, rec, f_neg_inf, poly_arg_none());
  r = poly_uop3(ctx, POLY_OP_WHERE, ft, rec_ne_ninf, r, f_neg_inf, poly_arg_none());

  return r;
}

/* sin_poly for float32, shared by tinygrad xsin small/large branches. */
static PolyUOp *sin_poly_f32(PolyCtx *ctx, PolyUOp *d) {
  PolyDType ft = d->dtype;
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(1.0));
  PolyUOp *d2 = poly_uop2(ctx, POLY_OP_MUL, ft, d, d, poly_arg_none());
  PolyUOp *t = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(2.6083159809786593541503e-06));
  t = poly_uop2(ctx, POLY_OP_MUL, ft, t, d2, poly_arg_none());
  t = poly_uop2(ctx, POLY_OP_ADD, ft, t,
                poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(-0.0001981069071916863322258)),
                poly_arg_none());
  t = poly_uop2(ctx, POLY_OP_MUL, ft, t, d2, poly_arg_none());
  t = poly_uop2(ctx, POLY_OP_ADD, ft, t,
                poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(0.00833307858556509017944336)),
                poly_arg_none());
  t = poly_uop2(ctx, POLY_OP_MUL, ft, t, d2, poly_arg_none());
  t = poly_uop2(ctx, POLY_OP_ADD, ft, t,
                poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(-0.166666597127914428710938)),
                poly_arg_none());
  t = poly_uop2(ctx, POLY_OP_MUL, ft, t, d2, poly_arg_none());
  t = poly_uop2(ctx, POLY_OP_ADD, ft, t, one, poly_arg_none());
  return poly_uop2(ctx, POLY_OP_MUL, ft, d, t, poly_arg_none());
}

/* tinygrad payne_hanek_reduction helper: select two_over_pi_f[i+offset]. */
static PolyUOp *take_two_over_pi_f32(PolyCtx *ctx, PolyUOp *i_u64, int offset) {
  static const uint32_t two_over_pi_f[] = {
    0x00000000u, 0x28be60dbu, 0x9391054au, 0x7f09d5f4u,
    0x7d4d3770u, 0x36d8a566u, 0x4f10e410u
  };
  const int len = (int)(sizeof(two_over_pi_f) / sizeof(two_over_pi_f[0]));
  const int max_count = len - 2 - offset;  /* tinygrad condition: count+offset < len-1 */
  PolyUOp *out = poly_uop0(ctx, POLY_OP_CONST, POLY_UINT32, poly_arg_int(0));
  for (int count = max_count; count >= 0; count--) {
    PolyUOp *cnt = poly_uop0(ctx, POLY_OP_CONST, POLY_UINT64, poly_arg_int((int64_t)count));
    PolyUOp *ne = poly_uop2(ctx, POLY_OP_CMPNE, POLY_BOOL, i_u64, cnt, poly_arg_none());
    PolyUOp *val = poly_uop0(ctx, POLY_OP_CONST, POLY_UINT32,
                             poly_arg_int((int64_t)two_over_pi_f[count + offset]));
    out = poly_uop3(ctx, POLY_OP_WHERE, POLY_UINT32, ne, out, val, poly_arg_none());
  }
  return out;
}

/*
 * rule_decomp_sin — Port of tinygrad's xsin for float32.
 *
 * SIN(d) → sign handling + Cody-Waite / Payne-Hanek reduction + polynomial.
 */
static PolyUOp *rule_decomp_sin(PolyCtx *ctx, PolyUOp *root,
                                 const PolyBindings *b) {
  (void)b;
  PolyUOp *d = root->src[0];
  if (root->dtype.bitsize != 32 || !poly_dtype_is_float(root->dtype))
    return NULL;

  PolyDType ft = root->dtype;
  PolyDType it = POLY_INT32;
  PolyDType ut32 = POLY_UINT32;
  PolyDType ut64 = POLY_UINT64;
  PolyDType bt = POLY_BOOL;

  /* Constants */
  PolyUOp *f_zero    = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(0.0));
  PolyUOp *f_one     = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(1.0));
  PolyUOp *f_neg_one = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(-1.0));
  PolyUOp *f_pi_2    = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(1.57079632679489661923));
  PolyUOp *f_half    = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(0.5));
  PolyUOp *f_neg_half= poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(-0.5));
  PolyUOp *f_switch  = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(30.0));
  PolyUOp *f_pos_inf = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(__builtin_inf()));
  PolyUOp *f_neg_inf = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(-__builtin_inf()));
  PolyUOp *f_nan     = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(__builtin_nan("")));
  PolyUOp *f_m_1_pi  = poly_uop0(ctx, POLY_OP_CONST, ft,
                                  poly_arg_float(0.318309886183790671537767526745028724));
  PolyUOp *f_2p32    = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(4294967296.0));
  PolyUOp *f_ph_mul  = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(3.4061215800865545e-19));
  PolyUOp *i_zero    = poly_uop0(ctx, POLY_OP_CONST, it, poly_arg_int(0));
  PolyUOp *i_one     = poly_uop0(ctx, POLY_OP_CONST, it, poly_arg_int(1));
  PolyUOp *i_two     = poly_uop0(ctx, POLY_OP_CONST, it, poly_arg_int(2));
  PolyUOp *i_23      = poly_uop0(ctx, POLY_OP_CONST, it, poly_arg_int(23));
  PolyUOp *i_31      = poly_uop0(ctx, POLY_OP_CONST, it, poly_arg_int(31));
  PolyUOp *i_32      = poly_uop0(ctx, POLY_OP_CONST, it, poly_arg_int(32));
  PolyUOp *i_126     = poly_uop0(ctx, POLY_OP_CONST, it, poly_arg_int(126));
  PolyUOp *i_255_u32 = poly_uop0(ctx, POLY_OP_CONST, ut32, poly_arg_int(255));
  PolyUOp *u_32      = poly_uop0(ctx, POLY_OP_CONST, ut64, poly_arg_int(32));
  PolyUOp *u_5       = poly_uop0(ctx, POLY_OP_CONST, ut64, poly_arg_int(5));
  PolyUOp *u_62      = poly_uop0(ctx, POLY_OP_CONST, ut64, poly_arg_int(62));
  PolyUOp *u_mask    = poly_uop0(ctx, POLY_OP_CONST, ut64, poly_arg_int(0x3fffffffffffffffull));
  PolyUOp *u_m1      = poly_uop0(ctx, POLY_OP_CONST, ut32, poly_arg_int(0x807fffffu));
  PolyUOp *u_m2      = poly_uop0(ctx, POLY_OP_CONST, ut32, poly_arg_int(0x3f000000u));

  /* _lazy_map_numbers(d, 0, 0, 0, d) */
  PolyUOp *d_ne_pos_inf = poly_uop2(ctx, POLY_OP_CMPNE, bt, d, f_pos_inf, poly_arg_none());
  PolyUOp *d_is_nan = poly_uop2(ctx, POLY_OP_CMPNE, bt, d, d, poly_arg_none());
  PolyUOp *d_ne_neg_inf = poly_uop2(ctx, POLY_OP_CMPNE, bt, d, f_neg_inf, poly_arg_none());
  PolyUOp *x_inner = poly_uop3(ctx, POLY_OP_WHERE, ft, d_ne_neg_inf, d, f_zero, poly_arg_none());
  PolyUOp *x_mid = poly_uop3(ctx, POLY_OP_WHERE, ft, d_is_nan, f_zero, x_inner, poly_arg_none());
  PolyUOp *x = poly_uop3(ctx, POLY_OP_WHERE, ft, d_ne_pos_inf, x_mid, f_zero, poly_arg_none());

  /* x_sign = x!=0 ? (x<0 ? -1 : 1) : 0 */
  PolyUOp *x_ne0 = poly_uop2(ctx, POLY_OP_CMPNE, bt, x, f_zero, poly_arg_none());
  PolyUOp *x_lt0 = poly_uop2(ctx, POLY_OP_CMPLT, bt, x, f_zero, poly_arg_none());
  PolyUOp *x_pm  = poly_uop3(ctx, POLY_OP_WHERE, ft, x_lt0, f_neg_one, f_one, poly_arg_none());
  PolyUOp *x_sign= poly_uop3(ctx, POLY_OP_WHERE, ft, x_ne0, x_pm, f_zero, poly_arg_none());
  PolyUOp *x_abs = poly_uop2(ctx, POLY_OP_MUL, ft, x, x_sign, poly_arg_none());

  /* Cody-Waite reduction (tinygrad cody_waite_reduction for float32). */
  PolyUOp *qf_raw = poly_uop2(ctx, POLY_OP_MUL, ft, x_abs, f_m_1_pi, poly_arg_none());
  PolyUOp *qf_lt0 = poly_uop2(ctx, POLY_OP_CMPLT, bt, qf_raw, f_zero, poly_arg_none());
  PolyUOp *qf_off = poly_uop3(ctx, POLY_OP_WHERE, ft, qf_lt0, f_neg_half, f_half, poly_arg_none());
  PolyUOp *qf_round = poly_uop2(ctx, POLY_OP_ADD, ft, qf_raw, qf_off, poly_arg_none());
  PolyUOp *q_small = poly_uop1(ctx, POLY_OP_CAST, it, qf_round, poly_arg_none());
  PolyUOp *qf = poly_uop1(ctx, POLY_OP_CAST, ft, q_small, poly_arg_none());
  PolyUOp *r_small = poly_uop2(ctx, POLY_OP_ADD, ft,
                               poly_uop2(ctx, POLY_OP_MUL, ft, qf,
                                         poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(-3.1414794921875)),
                                         poly_arg_none()),
                               x_abs, poly_arg_none());
  r_small = poly_uop2(ctx, POLY_OP_ADD, ft,
                      poly_uop2(ctx, POLY_OP_MUL, ft, qf,
                                poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(-0.00011315941810607910156)),
                                poly_arg_none()),
                      r_small, poly_arg_none());
  r_small = poly_uop2(ctx, POLY_OP_ADD, ft,
                      poly_uop2(ctx, POLY_OP_MUL, ft, qf,
                                poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(-1.9841872589410058936e-09)),
                                poly_arg_none()),
                      r_small, poly_arg_none());
  r_small = poly_uop2(ctx, POLY_OP_ADD, ft,
                      poly_uop2(ctx, POLY_OP_MUL, ft, qf,
                                poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(-1.2154201256553420762e-10)),
                                poly_arg_none()),
                      r_small, poly_arg_none());

  /* Payne-Hanek reduction (tinygrad payne_hanek_reduction). */
  PolyUOp *bits = poly_uop1(ctx, POLY_OP_BITCAST, ut32, x_abs, poly_arg_none());
  PolyUOp *exp_u32 = poly_uop2(ctx, POLY_OP_AND, ut32,
                               poly_uop2(ctx, POLY_OP_SHR, ut32, bits, i_23, poly_arg_none()),
                               i_255_u32, poly_arg_none());
  PolyUOp *f_bits = poly_uop2(ctx, POLY_OP_OR, ut32,
                              poly_uop2(ctx, POLY_OP_AND, ut32, bits, u_m1, poly_arg_none()),
                              u_m2, poly_arg_none());
  PolyUOp *f = poly_uop1(ctx, POLY_OP_BITCAST, ft, f_bits, poly_arg_none());
  PolyUOp *e_i = poly_uop2(ctx, POLY_OP_SUB, it,
                           poly_uop1(ctx, POLY_OP_CAST, it, exp_u32, poly_arg_none()),
                           i_126, poly_arg_none());
  PolyUOp *ia = poly_uop1(ctx, POLY_OP_CAST, ut64,
                          poly_uop2(ctx, POLY_OP_MUL, ft, f, f_2p32, poly_arg_none()),
                          poly_arg_none());
  PolyUOp *i_u64 = poly_uop2(ctx, POLY_OP_SHR, ut64,
                             poly_uop1(ctx, POLY_OP_CAST, ut64, e_i, poly_arg_none()),
                             u_5, poly_arg_none());
  PolyUOp *e_lo = poly_uop2(ctx, POLY_OP_AND, it, e_i, i_31, poly_arg_none());
  PolyUOp *offset = poly_uop2(ctx, POLY_OP_SUB, it, i_32, e_lo, poly_arg_none());

  PolyUOp *a0 = take_two_over_pi_f32(ctx, i_u64, 0);
  PolyUOp *a1 = take_two_over_pi_f32(ctx, i_u64, 1);
  PolyUOp *a2 = take_two_over_pi_f32(ctx, i_u64, 2);
  PolyUOp *a3 = take_two_over_pi_f32(ctx, i_u64, 3);
  PolyUOp *hi = poly_uop2(ctx, POLY_OP_OR, ut32,
                          poly_uop2(ctx, POLY_OP_SHL, ut32, a0, e_lo, poly_arg_none()),
                          poly_uop2(ctx, POLY_OP_SHR, ut32, a1, offset, poly_arg_none()),
                          poly_arg_none());
  PolyUOp *mi = poly_uop2(ctx, POLY_OP_OR, ut32,
                          poly_uop2(ctx, POLY_OP_SHL, ut32, a1, e_lo, poly_arg_none()),
                          poly_uop2(ctx, POLY_OP_SHR, ut32, a2, offset, poly_arg_none()),
                          poly_arg_none());
  PolyUOp *lo = poly_uop2(ctx, POLY_OP_OR, ut32,
                          poly_uop2(ctx, POLY_OP_SHL, ut32, a2, e_lo, poly_arg_none()),
                          poly_uop2(ctx, POLY_OP_SHR, ut32, a3, offset, poly_arg_none()),
                          poly_arg_none());

  PolyUOp *hp_hi = poly_uop2(ctx, POLY_OP_MUL, ut64, ia,
                             poly_uop1(ctx, POLY_OP_CAST, ut64, hi, poly_arg_none()),
                             poly_arg_none());
  PolyUOp *hp_mi = poly_uop2(ctx, POLY_OP_MUL, ut64, ia,
                             poly_uop1(ctx, POLY_OP_CAST, ut64, mi, poly_arg_none()),
                             poly_arg_none());
  PolyUOp *hp_lo = poly_uop2(ctx, POLY_OP_MUL, ut64, ia,
                             poly_uop1(ctx, POLY_OP_CAST, ut64, lo, poly_arg_none()),
                             poly_arg_none());
  PolyUOp *p = poly_uop2(ctx, POLY_OP_ADD, ut64,
                         poly_uop2(ctx, POLY_OP_ADD, ut64,
                                   poly_uop2(ctx, POLY_OP_SHL, ut64, hp_hi, u_32, poly_arg_none()),
                                   hp_mi, poly_arg_none()),
                         poly_uop2(ctx, POLY_OP_SHR, ut64, hp_lo, u_32, poly_arg_none()),
                         poly_arg_none());
  PolyUOp *q_ph = poly_uop1(ctx, POLY_OP_CAST, it,
                            poly_uop2(ctx, POLY_OP_SHR, ut64, p, u_62, poly_arg_none()),
                            poly_arg_none());
  PolyUOp *p_masked = poly_uop2(ctx, POLY_OP_AND, ut64, p, u_mask, poly_arg_none());
  PolyUOp *r_ph_base = poly_uop2(ctx, POLY_OP_MUL, ft,
                                 poly_uop1(ctx, POLY_OP_CAST, ft, p_masked, poly_arg_none()),
                                 f_ph_mul, poly_arg_none());
  PolyUOp *f_lt_half = poly_uop2(ctx, POLY_OP_CMPLT, bt, f, f_half, poly_arg_none());
  PolyUOp *r_ph = poly_uop3(ctx, POLY_OP_WHERE, ft, f_lt_half, r_ph_base,
                            poly_uop2(ctx, POLY_OP_SUB, ft, r_ph_base, f_pi_2, poly_arg_none()),
                            poly_arg_none());
  q_ph = poly_uop3(ctx, POLY_OP_WHERE, it, f_lt_half, q_ph,
                   poly_uop2(ctx, POLY_OP_ADD, it, q_ph, i_one, poly_arg_none()),
                   poly_arg_none());

  /* tinygrad xsin composition: sin_poly_small / sin_poly_large split at 30.0 */
  PolyUOp *q_small_odd = poly_uop2(ctx, POLY_OP_CMPNE, bt,
                                   poly_uop2(ctx, POLY_OP_AND, it, q_small, i_one, poly_arg_none()),
                                   i_zero, poly_arg_none());
  PolyUOp *small_sign = poly_uop3(ctx, POLY_OP_WHERE, ft, q_small_odd, f_neg_one, f_one, poly_arg_none());
  PolyUOp *result_small = poly_uop2(ctx, POLY_OP_MUL, ft, sin_poly_f32(ctx, r_small), small_sign, poly_arg_none());

  PolyUOp *q_ph_odd = poly_uop2(ctx, POLY_OP_CMPNE, bt,
                                poly_uop2(ctx, POLY_OP_AND, it, q_ph, i_one, poly_arg_none()),
                                i_zero, poly_arg_none());
  PolyUOp *large_arg = poly_uop2(ctx, POLY_OP_ADD, ft, r_ph,
                                 poly_uop3(ctx, POLY_OP_WHERE, ft, q_ph_odd, f_pi_2, f_zero, poly_arg_none()),
                                 poly_arg_none());
  PolyUOp *q_ph_bit2 = poly_uop2(ctx, POLY_OP_CMPNE, bt,
                                 poly_uop2(ctx, POLY_OP_AND, it, q_ph, i_two, poly_arg_none()),
                                 i_zero, poly_arg_none());
  PolyUOp *large_sign = poly_uop3(ctx, POLY_OP_WHERE, ft, q_ph_bit2, f_neg_one, f_one, poly_arg_none());
  PolyUOp *result_large = poly_uop2(ctx, POLY_OP_MUL, ft, sin_poly_f32(ctx, large_arg), large_sign, poly_arg_none());

  PolyUOp *use_small = poly_uop2(ctx, POLY_OP_CMPLT, bt, x_abs, f_switch, poly_arg_none());
  PolyUOp *result = poly_uop3(ctx, POLY_OP_WHERE, ft, use_small, result_small, result_large, poly_arg_none());

  /* Restore original sign */
  result = poly_uop2(ctx, POLY_OP_MUL, ft, result, x_sign, poly_arg_none());

  /* _lazy_map_numbers(d, nan, nan, nan, result) */
  PolyUOp *out_inner = poly_uop3(ctx, POLY_OP_WHERE, ft, d_ne_neg_inf, result, f_nan, poly_arg_none());
  PolyUOp *out_mid = poly_uop3(ctx, POLY_OP_WHERE, ft, d_is_nan, f_nan, out_inner, poly_arg_none());
  PolyUOp *out = poly_uop3(ctx, POLY_OP_WHERE, ft, d_ne_pos_inf, out_mid, f_nan, poly_arg_none());
  return out;
}

/* ── Build the pm_transcendental PatternMatcher ──────────────────────── */

static PolyPatternMatcher *g_pm_transcendental = NULL;

static PolyPatternMatcher *poly_pm_transcendental(void) {
  if (g_pm_transcendental) return g_pm_transcendental;

  PolyOpSet exp2_set = poly_opset_add((PolyOpSet){{0,0}}, POLY_OP_EXP2);
  PolyOpSet log2_set = poly_opset_add((PolyOpSet){{0,0}}, POLY_OP_LOG2);
  PolyOpSet sin_set  = poly_opset_add((PolyOpSet){{0,0}}, POLY_OP_SIN);
  PolyRule rules[] = {
    { poly_pat_ops(exp2_set, NULL, 0, NULL), rule_decomp_exp2 },
    { poly_pat_ops(log2_set, NULL, 0, NULL), rule_decomp_log2 },
    { poly_pat_ops(sin_set, NULL, 0, NULL),  rule_decomp_sin },
  };
  g_pm_transcendental = poly_pm_new(rules, sizeof(rules) / sizeof(rules[0]));
  return g_pm_transcendental;
}

/* ── Expander (tinygrad codegen/late/expander.py) ───────────────────── */

static int64_t pair_tuple_prod(PolyArg a) {
  if (a.kind != POLY_ARG_PAIR_TUPLE) return 1;
  int64_t p = 1;
  for (int i = 0; i < a.pair_tuple.n; i++) p *= a.pair_tuple.pairs[i][1];
  return p;
}

static bool pair_tuple_empty(PolyArg a) {
  return a.kind == POLY_ARG_NONE || (a.kind == POLY_ARG_PAIR_TUPLE && a.pair_tuple.n == 0);
}

static PolyArg poly_arg_pair_tuple(int64_t (*pairs)[2], int n) {
  PolyArg a = poly_arg_none();
  a.kind = POLY_ARG_PAIR_TUPLE;
  a.pair_tuple.pairs = pairs;
  a.pair_tuple.n = n;
  return a;
}

static PolyArg poly_arg_int_tuple_local(int64_t *vals, int n) {
  PolyArg a = poly_arg_none();
  a.kind = POLY_ARG_INT_TUPLE;
  a.int_tuple.vals = vals;
  a.int_tuple.n = n;
  return a;
}

static bool pair_list_contains(int64_t (*pairs)[2], int n, int64_t axis, int64_t sz) {
  for (int i = 0; i < n; i++)
    if (pairs[i][0] == axis && pairs[i][1] == sz) return true;
  return false;
}

static int pair_list_find_axis(int64_t (*pairs)[2], int n, int64_t axis) {
  for (int i = 0; i < n; i++) if (pairs[i][0] == axis) return i;
  return -1;
}

static int find_assignment(int64_t *ids, int64_t *vals, int n, int64_t axis, int64_t *out) {
  for (int i = 0; i < n; i++) {
    if (ids[i] == axis) { *out = vals[i]; return 1; }
  }
  return 0;
}

static int64_t compute_flat_from_assignment(int64_t (*eargs)[2], int n_eargs,
                                            int64_t *ids, int64_t *vals, int n_assign) {
  int64_t idx = 0, mul = 1;
  for (int i = n_eargs - 1; i >= 0; i--) {
    int64_t v = 0;
    (void)find_assignment(ids, vals, n_assign, eargs[i][0], &v);
    idx += v * mul;
    mul *= eargs[i][1];
  }
  return idx;
}

typedef struct {
  int64_t (*cargs)[2];
  int n_cargs;
  int64_t (*eargs)[2];
  int n_eargs;
  int64_t *vals;
  int64_t *out;
  int out_pos;
} SwizzleCtx;

static void swizzle_recur(SwizzleCtx *s, int dim) {
  if (dim == s->n_cargs) {
    int64_t ids[POLY_MAX_DIMS], v[POLY_MAX_DIMS];
    for (int i = 0; i < s->n_cargs; i++) { ids[i] = s->cargs[i][0]; v[i] = s->vals[i]; }
    s->out[s->out_pos++] = compute_flat_from_assignment(s->eargs, s->n_eargs, ids, v, s->n_cargs);
    return;
  }
  int64_t m = s->cargs[dim][1];
  for (int64_t i = 0; i < m; i++) {
    s->vals[dim] = i;
    swizzle_recur(s, dim + 1);
  }
}

static PolyUOp *make_gep(PolyCtx *ctx, PolyUOp *base, int64_t *idxs, int n_idxs) {
  return poly_uop1(ctx, POLY_OP_GEP, poly_dtype_vec(poly_dtype_scalar(base->dtype), n_idxs),
                   base, poly_arg_int_tuple_local(idxs, n_idxs));
}

static PolyUOp *do_expand(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  PolyUOp *expands[32];
  int n_expands = 0;
  for (int i = 0; i < root->n_src && n_expands < 32; i++) {
    if (root->src[i]->op == POLY_OP_UNROLL) expands[n_expands++] = root->src[i];
  }
  if (n_expands == 0) return NULL;

  int64_t expand_pairs[POLY_MAX_DIMS * 4][2];
  int n_expand_pairs = 0;
  for (int i = 0; i < n_expands; i++) {
    PolyArg a = expands[i]->arg;
    if (a.kind != POLY_ARG_PAIR_TUPLE) return NULL;
    for (int j = 0; j < a.pair_tuple.n; j++) {
      int64_t axis = a.pair_tuple.pairs[j][0], sz = a.pair_tuple.pairs[j][1];
      if (!pair_list_contains(expand_pairs, n_expand_pairs, axis, sz))
        expand_pairs[n_expand_pairs][0] = axis, expand_pairs[n_expand_pairs++][1] = sz;
    }
  }
  if (n_expand_pairs <= 0) return NULL;
  for (int i = 0; i < n_expand_pairs; i++) {
    for (int j = i + 1; j < n_expand_pairs; j++) {
      if (expand_pairs[j][0] < expand_pairs[i][0]) {
        int64_t t0 = expand_pairs[i][0], t1 = expand_pairs[i][1];
        expand_pairs[i][0] = expand_pairs[j][0]; expand_pairs[i][1] = expand_pairs[j][1];
        expand_pairs[j][0] = t0; expand_pairs[j][1] = t1;
      }
    }
  }
  int64_t expand_sz = 1;
  for (int i = 0; i < n_expand_pairs; i++) expand_sz *= expand_pairs[i][1];
  if (expand_sz <= 1) return NULL;

  PolyUOp *new_srcs[128];
  int n_new_srcs = 0;
  for (int i = 0; i < root->n_src && n_new_srcs < 128; i++) {
    PolyUOp *src = root->src[i];
    if (src->op == POLY_OP_UNROLL) {
      if (poly_arg_eq(src->arg, poly_arg_pair_tuple(expand_pairs, n_expand_pairs))) {
        new_srcs[n_new_srcs++] = src->src[0];
      } else {
        if (src->arg.kind != POLY_ARG_PAIR_TUPLE || src->n_src == 0) return NULL;
        int64_t n_swz = expand_sz;
        int64_t *swz = malloc((size_t)n_swz * sizeof(int64_t));
        int64_t vals[POLY_MAX_DIMS] = {0};
        SwizzleCtx swz_ctx = {
          .cargs = expand_pairs, .n_cargs = n_expand_pairs,
          .eargs = src->arg.pair_tuple.pairs, .n_eargs = src->arg.pair_tuple.n,
          .vals = vals, .out = swz, .out_pos = 0,
        };
        swizzle_recur(&swz_ctx, 0);
        int n_gep = (int)n_swz;
        if (src->dtype.count > 1) {
          int n2 = n_gep * src->dtype.count;
          int64_t *lst2 = malloc((size_t)n2 * sizeof(int64_t));
          int p = 0;
          for (int k = 0; k < n_gep; k++)
            for (int j = 0; j < src->dtype.count; j++) lst2[p++] = swz[k] * src->dtype.count + j;
          free(swz);
          swz = lst2;
          n_gep = n2;
        }
        new_srcs[n_new_srcs++] = make_gep(ctx, src->src[0], swz, n_gep);
        free(swz);
      }
      continue;
    }

    int off = range_start_for_op(root->op);
    if (off >= 0 && i >= off) {
      new_srcs[n_new_srcs++] = src;
      continue;
    }
    if (root->op == POLY_OP_INDEX && i >= 1 && !root->dtype.is_ptr) {
      new_srcs[n_new_srcs++] = src;
      continue;
    }

    if (src->dtype.count > 1) {
      PolyUOp *cat_srcs[128];
      int n_cat = (int)expand_sz;
      if (n_cat > 128) return NULL;
      for (int j = 0; j < n_cat; j++) cat_srcs[j] = src;
      new_srcs[n_new_srcs++] = poly_uop(ctx, POLY_OP_CAT,
                                        poly_dtype_vec(poly_dtype_scalar(src->dtype), n_cat * src->dtype.count),
                                        cat_srcs, n_cat, poly_arg_none());
    } else {
      PolyUOp *vec_srcs[128];
      int n_vec = (int)expand_sz;
      if (n_vec > 128) return NULL;
      for (int j = 0; j < n_vec; j++) vec_srcs[j] = src;
      new_srcs[n_new_srcs++] = poly_uop(ctx, POLY_OP_VECTORIZE,
                                        poly_dtype_vec(src->dtype, n_vec), vec_srcs, n_vec, poly_arg_none());
    }
  }

  PolyDType out_dt = poly_dtype_vec(poly_dtype_scalar(root->dtype), root->dtype.count * (int)expand_sz);
  PolyUOp *nsrc = poly_uop(ctx, root->op, out_dt, new_srcs, n_new_srcs, root->arg);
  return poly_uop1(ctx, POLY_OP_UNROLL, root->dtype, nsrc, poly_arg_pair_tuple(expand_pairs, n_expand_pairs));
}

static PolyUOp *do_contract(PolyCtx *ctx, PolyUOp *con, const PolyBindings *b) {
  (void)b;
  if (con->n_src < 1) return NULL;
  PolyUOp *ex = con->src[0];
  if (ex->op != POLY_OP_UNROLL) {
    if (con->dtype.count <= 1) return NULL;
    PolyUOp *srcs[128];
    if (con->dtype.count > 128) return NULL;
    for (int i = 0; i < con->dtype.count; i++) srcs[i] = ex;
    return poly_uop(ctx, POLY_OP_VECTORIZE, con->dtype, srcs, con->dtype.count, poly_arg_none());
  }
  if (con->arg.kind != POLY_ARG_PAIR_TUPLE || ex->arg.kind != POLY_ARG_PAIR_TUPLE || ex->n_src == 0) return NULL;

  int64_t new_pairs[POLY_MAX_DIMS][2];
  int n_new = 0;
  for (int i = 0; i < ex->arg.pair_tuple.n; i++) {
    int64_t axis = ex->arg.pair_tuple.pairs[i][0], sz = ex->arg.pair_tuple.pairs[i][1];
    if (pair_list_find_axis(con->arg.pair_tuple.pairs, con->arg.pair_tuple.n, axis) < 0) {
      new_pairs[n_new][0] = axis;
      new_pairs[n_new][1] = sz;
      n_new++;
    }
  }
  /* Common path (and tinygrad test case): CONTRACT removes all UNROLL axes. */
  if (n_new == 0) return ex->src[0];
  return poly_uop1(ctx, POLY_OP_UNROLL, con->dtype, ex->src[0], poly_arg_pair_tuple(new_pairs, n_new));
}

static PolyUOp *end_unrolls(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b) {
  (void)b;
  if (u->op != POLY_OP_END || u->n_src <= 1) return NULL;
  PolyUOp *unrolls[32], *others[64];
  int n_unrolls = 0, n_others = 0;
  for (int i = 1; i < u->n_src; i++) {
    if (u->src[i]->op == POLY_OP_UNROLL && n_unrolls < 32) unrolls[n_unrolls++] = u->src[i];
    else if (n_others < 64) others[n_others++] = u->src[i];
  }
  if (n_unrolls == 0) return NULL;

  int64_t pairs[POLY_MAX_DIMS * 4][2];
  int n_pairs = 0;
  for (int i = 0; i < n_unrolls; i++) {
    PolyArg a = unrolls[i]->arg;
    if (a.kind != POLY_ARG_PAIR_TUPLE) continue;
    for (int j = 0; j < a.pair_tuple.n; j++) {
      int64_t axis = a.pair_tuple.pairs[j][0], sz = a.pair_tuple.pairs[j][1];
      if (!pair_list_contains(pairs, n_pairs, axis, sz))
        pairs[n_pairs][0] = axis, pairs[n_pairs++][1] = sz;
    }
  }
  PolyUOp *ret = poly_uop1(ctx, POLY_OP_CONTRACT, POLY_VOID, u->src[0], poly_arg_pair_tuple(pairs, n_pairs));
  PolyUOp *new_src[65];
  int n_new = 0;
  new_src[n_new++] = ret;
  for (int i = 0; i < n_others; i++) new_src[n_new++] = others[i];
  return poly_uop(ctx, u->op, u->dtype, new_src, n_new, u->arg);
}

static PolyUOp *rule_empty_unroll(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b) {
  (void)ctx; (void)b;
  if (u->op != POLY_OP_UNROLL || u->n_src < 1) return NULL;
  return pair_tuple_empty(u->arg) ? u->src[0] : NULL;
}

static PolyUOp *rule_double_unroll(PolyCtx *ctx, PolyUOp *outer, const PolyBindings *b) {
  PolyUOp *inner = poly_bind(b, "inner");
  if (!inner || inner->op != POLY_OP_UNROLL || inner->n_src == 0) return NULL;
  if (outer->arg.kind != POLY_ARG_PAIR_TUPLE || inner->arg.kind != POLY_ARG_PAIR_TUPLE) return NULL;
  int n = inner->arg.pair_tuple.n + outer->arg.pair_tuple.n;
  int64_t pairs[POLY_MAX_DIMS * 4][2];
  if (n > (int)(sizeof(pairs) / sizeof(pairs[0]))) return NULL;
  int p = 0;
  for (int i = 0; i < inner->arg.pair_tuple.n; i++, p++) {
    pairs[p][0] = inner->arg.pair_tuple.pairs[i][0];
    pairs[p][1] = inner->arg.pair_tuple.pairs[i][1];
  }
  for (int i = 0; i < outer->arg.pair_tuple.n; i++, p++) {
    pairs[p][0] = outer->arg.pair_tuple.pairs[i][0];
    pairs[p][1] = outer->arg.pair_tuple.pairs[i][1];
  }
  return poly_uop1(ctx, POLY_OP_UNROLL, outer->dtype, inner->src[0], poly_arg_pair_tuple(pairs, p));
}

static PolyUOp *rule_pre_expand_range(PolyCtx *ctx, PolyUOp *r, const PolyBindings *b) {
  (void)b;
  if (r->op != POLY_OP_RANGE || !poly_arg_is_range(r->arg)) return NULL;
  PolyAxisType t = poly_range_axis_type(r->arg);
  if (!(t == POLY_AXIS_UPCAST || t == POLY_AXIS_UNROLL)) return NULL;
  if (!(r->n_src > 0 && r->src[0]->op == POLY_OP_CONST && r->src[0]->arg.kind == POLY_ARG_INT)) return NULL;
  int64_t s = r->src[0]->arg.i;
  if (s <= 0 || s > 128) return NULL;

  PolyUOp *vals[128];
  for (int64_t i = 0; i < s; i++)
    vals[i] = poly_uop0(ctx, POLY_OP_CONST, r->dtype, poly_arg_int(i));
  PolyUOp *vconst = poly_uop(ctx, POLY_OP_VCONST, poly_dtype_vec(r->dtype, (int)s), vals, (int)s, poly_arg_none());
  int64_t pairs[1][2] = {{ poly_range_axis_id(r->arg), s }};
  return poly_uop1(ctx, POLY_OP_UNROLL, r->dtype, vconst, poly_arg_pair_tuple(pairs, 1));
}

static PolyUOp *rule_fix_reduce_unroll(PolyCtx *ctx, PolyUOp *x, const PolyBindings *b) {
  (void)b;
  if (x->op != POLY_OP_REDUCE || x->n_src <= 1) return NULL;
  PolyUOp *reduce_range[32], *reduce_expand[32];
  int n_range = 0, n_expand = 0;
  for (int i = 1; i < x->n_src; i++) {
    if (x->src[i]->op == POLY_OP_RANGE && n_range < 32) reduce_range[n_range++] = x->src[i];
    else if (n_expand < 32) reduce_expand[n_expand++] = x->src[i];
  }
  if (n_expand == 0) return NULL;
  int64_t pairs[POLY_MAX_DIMS * 4][2];
  int n_pairs = 0;
  for (int i = 0; i < n_expand; i++) {
    if (reduce_expand[i]->op == POLY_OP_CONST) continue;
    if (reduce_expand[i]->op != POLY_OP_UNROLL || reduce_expand[i]->arg.kind != POLY_ARG_PAIR_TUPLE) return NULL;
    for (int j = 0; j < reduce_expand[i]->arg.pair_tuple.n; j++) {
      int64_t axis = reduce_expand[i]->arg.pair_tuple.pairs[j][0];
      int64_t sz = reduce_expand[i]->arg.pair_tuple.pairs[j][1];
      if (!pair_list_contains(pairs, n_pairs, axis, sz))
        pairs[n_pairs][0] = axis, pairs[n_pairs++][1] = sz;
    }
  }
  PolyUOp *ret = x->src[0];
  if (n_pairs > 0)
    ret = poly_uop1(ctx, POLY_OP_CONTRACT, poly_dtype_vec(x->dtype, (int)pair_tuple_prod(poly_arg_pair_tuple(pairs, n_pairs))),
                    ret, poly_arg_pair_tuple(pairs, n_pairs));

  PolyUOp *new_src[64];
  int n_new = 0;
  new_src[n_new++] = ret;
  for (int i = 0; i < n_range; i++) new_src[n_new++] = reduce_range[i];
  return poly_uop(ctx, POLY_OP_REDUCE, x->dtype, new_src, n_new, x->arg);
}

static PolyUOp *rule_fix_store_unroll(PolyCtx *ctx, PolyUOp *x, const PolyBindings *b) {
  (void)b;
  if (x->op != POLY_OP_STORE || x->n_src <= 2) return NULL;
  PolyUOp *store_expand[32], *store_range[32];
  int n_expand = 0, n_range = 0;
  for (int i = 2; i < x->n_src; i++) {
    if (x->src[i]->op == POLY_OP_UNROLL && n_expand < 32) store_expand[n_expand++] = x->src[i];
    else if (n_range < 32) store_range[n_range++] = x->src[i];
  }
  if (n_expand == 0) return NULL;

  PolyUOp *base_src[64];
  int n_base = 0;
  base_src[n_base++] = x->src[0];
  base_src[n_base++] = x->src[1];
  for (int i = 0; i < n_range; i++) base_src[n_base++] = store_range[i];
  PolyUOp *base_store = poly_uop(ctx, POLY_OP_STORE, x->dtype, base_src, n_base, x->arg);

  int64_t pairs[POLY_MAX_DIMS * 4][2];
  int n_pairs = 0;
  for (int i = 0; i < n_expand; i++) {
    if (store_expand[i]->arg.kind != POLY_ARG_PAIR_TUPLE) continue;
    for (int j = 0; j < store_expand[i]->arg.pair_tuple.n; j++) {
      int64_t axis = store_expand[i]->arg.pair_tuple.pairs[j][0];
      int64_t sz = store_expand[i]->arg.pair_tuple.pairs[j][1];
      if (!pair_list_contains(pairs, n_pairs, axis, sz))
        pairs[n_pairs][0] = axis, pairs[n_pairs++][1] = sz;
    }
  }
  return poly_uop1(ctx, POLY_OP_CONTRACT, POLY_VOID, base_store, poly_arg_pair_tuple(pairs, n_pairs));
}

static PolyPatternMatcher *g_pm_pre_expander = NULL;
static PolyPatternMatcher *poly_pm_pre_expander(void) {
  if (g_pm_pre_expander) return g_pm_pre_expander;
  PolyRule rules[] = {
    { poly_pat_op(POLY_OP_RANGE, NULL, 0, "r"), rule_pre_expand_range },
    { poly_pat_op(POLY_OP_REDUCE, NULL, 0, "x"), rule_fix_reduce_unroll },
    { poly_pat_op(POLY_OP_STORE, NULL, 0, "x"), rule_fix_store_unroll },
  };
  g_pm_pre_expander = poly_pm_new(rules, (int)(sizeof(rules) / sizeof(rules[0])));
  return g_pm_pre_expander;
}

static PolyPatternMatcher *g_pm_expander = NULL;
static PolyPatternMatcher *poly_pm_expander(void) {
  if (g_pm_expander) return g_pm_expander;
  PolyOpSet exp_ops = POLY_GROUP_ALU;
  exp_ops = poly_opset_add(exp_ops, POLY_OP_CAST);
  exp_ops = poly_opset_add(exp_ops, POLY_OP_BITCAST);
  exp_ops = poly_opset_add(exp_ops, POLY_OP_GEP);
  exp_ops = poly_opset_add(exp_ops, POLY_OP_WMMA);
  exp_ops = poly_opset_add(exp_ops, POLY_OP_LOAD);
  exp_ops = poly_opset_add(exp_ops, POLY_OP_STORE);
  exp_ops = poly_opset_add(exp_ops, POLY_OP_INDEX);
  exp_ops = poly_opset_add(exp_ops, POLY_OP_BUFFERIZE);
  exp_ops = poly_opset_add(exp_ops, POLY_OP_VECTORIZE);
  exp_ops = poly_opset_add(exp_ops, POLY_OP_REDUCE);
  exp_ops = poly_opset_add(exp_ops, POLY_OP_END);
  exp_ops = poly_opset_add(exp_ops, POLY_OP_AFTER);

  PolyOpSet rej = poly_opset_add((PolyOpSet){{0,0}}, POLY_OP_UNROLL);
  PolyRule rules[] = {
    { poly_pat_op(POLY_OP_END, NULL, 0, "u"), end_unrolls },
    { poly_pat_op1(POLY_OP_UNROLL, poly_pat_op(POLY_OP_UNROLL, NULL, 0, "inner"), "outer"), rule_double_unroll },
    { poly_pat_set_early_reject(poly_pat_allow_any_len(poly_pat_ops(exp_ops, NULL, 0, "root")), rej), do_expand },
    { poly_pat_op(POLY_OP_CONTRACT, NULL, 0, "con"), do_contract },
    { poly_pat_op(POLY_OP_UNROLL, NULL, 0, "u"), rule_empty_unroll },
  };
  g_pm_expander = poly_pm_new(rules, (int)(sizeof(rules) / sizeof(rules[0])));
  return g_pm_expander;
}

/* ── pm_add_loads (tinygrad codegen/late/devectorizer.py) ───────────── */

static PolyUOp *rule_add_load_to_index(PolyCtx *ctx, PolyUOp *idx, const PolyBindings *b) {
  (void)b;
  if (idx->op != POLY_OP_INDEX) return NULL;
  /* add LOAD only for non-pointer INDEX values */
  if (idx->dtype.is_ptr) return NULL;
  PolyDType ld_dtype = poly_dtype_scalar(idx->dtype);
  return poly_uop1(ctx, POLY_OP_LOAD, ld_dtype, idx, poly_arg_none());
}

static PolyUOp *rule_remove_load_from_store(PolyCtx *ctx, PolyUOp *s, const PolyBindings *b) {
  PolyUOp *ld = poly_bind(b, "ld");
  if (!ld || ld->op != POLY_OP_LOAD || ld->n_src < 1 || s->op != POLY_OP_STORE || s->n_src < 2) return NULL;
  PolyUOp *new_src[64];
  int n_new = 0;
  new_src[n_new++] = ld->src[0];
  new_src[n_new++] = s->src[1];
  for (int i = 2; i < s->n_src && n_new < 64; i++) new_src[n_new++] = s->src[i];
  return poly_uop(ctx, POLY_OP_STORE, s->dtype, new_src, n_new, s->arg);
}

static PolyPatternMatcher *g_pm_add_loads = NULL;
static PolyPatternMatcher *poly_pm_add_loads(void) {
  if (g_pm_add_loads) return g_pm_add_loads;
  PolyRule rules[] = {
    { poly_pat_op(POLY_OP_INDEX, NULL, 0, "idx"), rule_add_load_to_index },
    { poly_pat_allow_any_len(poly_pat_op2(POLY_OP_STORE, poly_pat_op(POLY_OP_LOAD, NULL, 0, "ld"), poly_pat_any("val"), "s")),
      rule_remove_load_from_store },
  };
  g_pm_add_loads = poly_pm_new(rules, (int)(sizeof(rules) / sizeof(rules[0])));
  return g_pm_add_loads;
}

/* ── load_store_folding (subset port for vectorized INDEX collapse) ─── */

static bool is_vectorize_of_same(PolyUOp *u, PolyUOp **scalar_out, int *n_out) {
  if (!u || u->op != POLY_OP_VECTORIZE || u->n_src <= 0) return false;
  PolyUOp *s0 = u->src[0];
  for (int i = 1; i < u->n_src; i++) if (u->src[i] != s0) return false;
  if (scalar_out) *scalar_out = s0;
  if (n_out) *n_out = u->n_src;
  return true;
}

static PolyUOp *make_gep_lane(PolyCtx *ctx, PolyUOp *src, int lane);

static PolyUOp *scalarize_lane_expr(PolyCtx *ctx, PolyUOp *u, int lane) {
  if (!u) return NULL;
  if (u->op == POLY_OP_UNROLL && u->n_src > 0) {
    PolyUOp *src = u->src[0];
    if (src && src->dtype.count > 1) {
      int cnt = src->dtype.count;
      int pick = (cnt > 0) ? (lane % cnt) : 0;
      if (pick < 0) pick = 0;
      return scalarize_lane_expr(ctx, src, pick);
    }
    return src;
  }
  if (u->dtype.count <= 1) return u;
  if (lane < 0) lane = 0;

  if (u->op == POLY_OP_VECTORIZE) {
    if (u->n_src <= 0) return NULL;
    int pick = lane;
    if (pick >= u->n_src) pick = u->n_src - 1;
    return scalarize_lane_expr(ctx, u->src[pick], 0);
  }
  if (u->op == POLY_OP_VCONST) {
    PolyDType sdt = poly_dtype_scalar(u->dtype);
    if (u->n_src > lane && u->src[lane] && u->src[lane]->op == POLY_OP_CONST)
      return u->src[lane];
    if (u->arg.kind == POLY_ARG_INT_TUPLE && lane < u->arg.int_tuple.n)
      return poly_uop0(ctx, POLY_OP_CONST, sdt, poly_arg_int(u->arg.int_tuple.vals[lane]));
    return NULL;
  }
  if (u->op == POLY_OP_GEP && u->n_src > 0) {
    if (u->arg.kind == POLY_ARG_INT) {
      return scalarize_lane_expr(ctx, u->src[0], (int)u->arg.i);
    }
    if (u->arg.kind == POLY_ARG_INT_TUPLE && u->arg.int_tuple.n > 0) {
      int pick = (u->arg.int_tuple.n == 1) ? (int)u->arg.int_tuple.vals[0] : lane;
      if (pick < 0) pick = 0;
      if (pick >= u->arg.int_tuple.n) pick = u->arg.int_tuple.n - 1;
      return scalarize_lane_expr(ctx, u->src[0], (int)u->arg.int_tuple.vals[pick]);
    }
  }

  bool can_scalarize = poly_opset_has(POLY_GROUP_ALU, u->op) ||
                       u->op == POLY_OP_CAST || u->op == POLY_OP_BITCAST;
  if (can_scalarize) {
    PolyUOp *srcs[8];
    if (u->n_src > 8) return make_gep_lane(ctx, u, lane);
    for (int i = 0; i < u->n_src; i++) {
      PolyUOp *s = u->src[i];
      srcs[i] = (s && s->dtype.count > 1) ? scalarize_lane_expr(ctx, s, lane) : s;
      if (!srcs[i]) return make_gep_lane(ctx, u, lane);
    }
    PolyDType sdt = poly_dtype_scalar(u->dtype);
    if (u->op == POLY_OP_CAST) {
      if (u->n_src != 1) return make_gep_lane(ctx, u, lane);
      return poly_uop1(ctx, POLY_OP_CAST, sdt, srcs[0], u->arg);
    }
    if (u->op == POLY_OP_BITCAST) {
      if (u->n_src != 1) return make_gep_lane(ctx, u, lane);
      return poly_uop1(ctx, POLY_OP_BITCAST, sdt, srcs[0], u->arg);
    }
    if (u->n_src == 1) return poly_uop1(ctx, u->op, sdt, srcs[0], u->arg);
    if (u->n_src == 2) return poly_uop2(ctx, u->op, sdt, srcs[0], srcs[1], u->arg);
    if (u->n_src == 3) return poly_uop3(ctx, u->op, sdt, srcs[0], srcs[1], srcs[2], u->arg);
  }
  return make_gep_lane(ctx, u, lane);
}

static int uop_ptr_cmp(const void *ap, const void *bp) {
  const PolyUOp *a = *(const PolyUOp *const *)ap;
  const PolyUOp *b = *(const PolyUOp *const *)bp;
  if (a < b) return -1;
  if (a > b) return 1;
  return 0;
}

static bool lanes_supported_for_fold(int lanes) {
  return lanes > 1 && lanes <= 4 && ((lanes & (lanes - 1)) == 0);
}

typedef struct {
  PolyUOp *terms[64];
  int n_terms;
  int64_t cst;
  bool ok;
} LaneAffineExpr;

static void collect_add_terms(PolyCtx *ctx, PolyUOp *u, LaneAffineExpr *out, bool negate) {
  (void)ctx;
  if (!u || !out || !out->ok) return;
  if (u->op == POLY_OP_ADD && u->n_src == 2) {
    collect_add_terms(ctx, u->src[0], out, negate);
    collect_add_terms(ctx, u->src[1], out, negate);
    return;
  }
  if (u->op == POLY_OP_SUB && u->n_src == 2) {
    collect_add_terms(ctx, u->src[0], out, negate);
    collect_add_terms(ctx, u->src[1], out, !negate);
    return;
  }
  if (u->op == POLY_OP_CONST && u->arg.kind == POLY_ARG_INT) {
    int64_t v = negate ? -u->arg.i : u->arg.i;
    out->cst += v;
    return;
  }
  if (negate) {
    if (out->n_terms >= 64) {
      out->ok = false;
      return;
    }
    out->terms[out->n_terms++] = poly_uop1(ctx, POLY_OP_NEG, u->dtype, u, poly_arg_none());
    return;
  }
  if (out->n_terms >= 64) {
    out->ok = false;
    return;
  }
  out->terms[out->n_terms++] = u;
}

static bool affine_terms_equal(PolyUOp **a, int n_a, PolyUOp **b, int n_b) {
  if (n_a != n_b) return false;
  for (int i = 0; i < n_a; i++) if (a[i] != b[i]) return false;
  return true;
}

static PolyUOp *build_add_expr(PolyCtx *ctx, PolyDType dt, PolyUOp **terms, int n_terms, int64_t cst) {
  PolyUOp *ret = NULL;
  if (n_terms > 0) {
    ret = terms[0];
    for (int i = 1; i < n_terms; i++)
      ret = poly_uop2(ctx, POLY_OP_ADD, dt, ret, terms[i], poly_arg_none());
  }
  if (cst != 0 || !ret) {
    PolyUOp *c = poly_uop0(ctx, POLY_OP_CONST, dt, poly_arg_int(cst));
    ret = ret ? poly_uop2(ctx, POLY_OP_ADD, dt, ret, c, poly_arg_none()) : c;
  }
  return ret;
}

static bool match_contiguous_lane_pattern(PolyCtx *ctx, PolyUOp *vidx, int lanes,
                                          PolyUOp **out_terms, int *out_n_terms,
                                          int64_t *out_base_const) {
  if (!vidx || lanes <= 0 || !out_terms || !out_n_terms || !out_base_const) return false;
  bool have_common = false;
  int n_common_terms = 0;
  int64_t base_const = 0;
  PolyUOp *common_terms[64];

  for (int i = 0; i < lanes; i++) {
    PolyUOp *lane_expr = scalarize_lane_expr(ctx, vidx, i);
    if (!lane_expr) return false;

    LaneAffineExpr ae = {.n_terms = 0, .cst = 0, .ok = true};
    collect_add_terms(ctx, lane_expr, &ae, false);
    if (!ae.ok) return false;
    qsort(ae.terms, (size_t)ae.n_terms, sizeof(ae.terms[0]), uop_ptr_cmp);

    if (!have_common) {
      have_common = true;
      n_common_terms = ae.n_terms;
      base_const = ae.cst;
      for (int j = 0; j < ae.n_terms; j++) common_terms[j] = ae.terms[j];
    } else {
      if (!affine_terms_equal(common_terms, n_common_terms, ae.terms, ae.n_terms)) return false;
      if (ae.cst != base_const + i) return false;
    }
  }
  if (!have_common) return false;
  for (int i = 0; i < n_common_terms; i++) out_terms[i] = common_terms[i];
  *out_n_terms = n_common_terms;
  *out_base_const = base_const;
  return true;
}

static PolyUOp *build_scalar_lane_index(PolyCtx *ctx, PolyUOp *idx, PolyUOp *buf_base, int lane) {
  if (!idx || idx->op != POLY_OP_INDEX || idx->n_src < 2 || !buf_base) return NULL;
  PolyUOp *srcs[64];
  int n_srcs = 0;
  srcs[n_srcs++] = buf_base;
  srcs[n_srcs++] = scalarize_lane_expr(ctx, idx->src[1], lane);
  if (!srcs[1]) return NULL;
  for (int i = 2; i < idx->n_src && n_srcs < 64; i++) {
    PolyUOp *s = idx->src[i];
    if (s && s->dtype.count > 1) s = scalarize_lane_expr(ctx, s, lane);
    if (!s) return NULL;
    srcs[n_srcs++] = s;
  }
  return poly_uop(ctx, POLY_OP_INDEX, buf_base->dtype, srcs, n_srcs, idx->arg);
}

/* tinygrad load_store_folding core behavior for this shape:
 * INDEX(VECTORIZE(buf,...), ADD(VECTORIZE(base,...), VCONST(0..N-1)))
 *   -> CAST(INDEX(buf, base), vec_ptr_dtype)
 */
static PolyUOp *rule_fold_vectorized_index(PolyCtx *ctx, PolyUOp *idx, const PolyBindings *b) {
  (void)b;
  if (!idx || idx->op != POLY_OP_INDEX || idx->n_src < 2) return NULL;
  PolyUOp *buf_vec = idx->src[0];
  PolyUOp *vidx = idx->src[1];
  PolyUOp *buf_base = NULL;
  int lanes = 0;
  if (!is_vectorize_of_same(buf_vec, &buf_base, &lanes) || !buf_base || lanes <= 1) return NULL;
  if (!lanes_supported_for_fold(lanes)) return NULL;
  if (!vidx || vidx->dtype.count != lanes) return NULL;

  PolyUOp *common_terms[64];
  int n_common_terms = 0;
  int64_t base_const = 0;
  if (!match_contiguous_lane_pattern(ctx, vidx, lanes,
                                     common_terms, &n_common_terms, &base_const)) return NULL;

  PolyDType idx_dt = poly_dtype_scalar(vidx->dtype);
  PolyUOp *base_scalar = build_add_expr(ctx, idx_dt, common_terms, n_common_terms, base_const);

  PolyUOp *new_src[64];
  int n_new = 0;
  new_src[n_new++] = buf_base;
  new_src[n_new++] = base_scalar;
  for (int i = 2; i < idx->n_src && n_new < 64; i++) new_src[n_new++] = idx->src[i];

  PolyUOp *base_idx = poly_uop(ctx, POLY_OP_INDEX, buf_base->dtype, new_src, n_new, idx->arg);
  PolyDType vec_ptr = poly_dtype_vec(buf_base->dtype, lanes);
  return poly_uop1(ctx, POLY_OP_CAST, vec_ptr, base_idx, poly_arg_none());
}

static PolyUOp *rule_split_vector_load(PolyCtx *ctx, PolyUOp *ld, const PolyBindings *b) {
  (void)b;
  if (!ld || ld->op != POLY_OP_LOAD || ld->n_src < 1) return NULL;
  PolyUOp *idx = ld->src[0];
  if (!idx || idx->op != POLY_OP_INDEX || idx->n_src < 2) return NULL;
  PolyUOp *buf_base = NULL;
  int lanes = 0;
  if (!is_vectorize_of_same(idx->src[0], &buf_base, &lanes) || !buf_base || lanes <= 1) return NULL;
  if (ld->dtype.count != lanes || idx->src[1]->dtype.count != lanes) return NULL;
  if (lanes > 128) return NULL;
  if (ld->n_src == 1 && lanes > 4) {
    PolyUOp *common_terms[64];
    int n_common_terms = 0;
    int64_t base_const = 0;
    if (match_contiguous_lane_pattern(ctx, idx->src[1], lanes,
                                      common_terms, &n_common_terms, &base_const)) {
      PolyUOp *elts[128];
      int p = 0;
      PolyDType sdt = poly_dtype_scalar(ld->dtype);
      PolyDType idx_dt = poly_dtype_scalar(idx->src[1]->dtype);
      for (int off = 0; off < lanes;) {
        int chunk = (lanes - off >= 4) ? 4 : 1;
        PolyUOp *base_scalar = build_add_expr(ctx, idx_dt, common_terms, n_common_terms, base_const + off);
        PolyUOp *chunk_idx = poly_uop2(ctx, POLY_OP_INDEX, buf_base->dtype, buf_base, base_scalar, idx->arg);
        if (chunk == 1) {
          elts[p++] = poly_uop1(ctx, POLY_OP_LOAD, sdt, chunk_idx, ld->arg);
        } else {
          PolyDType vec_ptr = poly_dtype_vec(buf_base->dtype, chunk);
          PolyUOp *cast_idx = poly_uop1(ctx, POLY_OP_CAST, vec_ptr, chunk_idx, poly_arg_none());
          PolyUOp *vload = poly_uop1(ctx, POLY_OP_LOAD, poly_dtype_vec(sdt, chunk), cast_idx, ld->arg);
          for (int i = 0; i < chunk; i++) {
            int64_t lane = i;
            elts[p++] = poly_uop1(ctx, POLY_OP_GEP, sdt, vload, poly_arg_int_tuple_local(&lane, 1));
          }
        }
        off += chunk;
      }
      return poly_uop(ctx, POLY_OP_VECTORIZE, ld->dtype, elts, p, poly_arg_none());
    }
  }

  PolyUOp *elts[128];
  PolyDType sdt = poly_dtype_scalar(ld->dtype);
  for (int i = 0; i < lanes; i++) {
    PolyUOp *lane_idx = build_scalar_lane_index(ctx, idx, buf_base, i);
    if (!lane_idx) return NULL;
    PolyUOp *lane_src[64];
    int n_lane_src = 0;
    lane_src[n_lane_src++] = lane_idx;
    for (int j = 1; j < ld->n_src && n_lane_src < 64; j++) {
      PolyUOp *s = ld->src[j];
      if (s && s->dtype.count > 1) s = scalarize_lane_expr(ctx, s, i);
      if (!s) return NULL;
      lane_src[n_lane_src++] = s;
    }
    elts[i] = poly_uop(ctx, POLY_OP_LOAD, sdt, lane_src, n_lane_src, ld->arg);
  }
  return poly_uop(ctx, POLY_OP_VECTORIZE, ld->dtype, elts, lanes, poly_arg_none());
}

static PolyUOp *rule_split_vector_store(PolyCtx *ctx, PolyUOp *st, const PolyBindings *b) {
  (void)b;
  if (!st || st->op != POLY_OP_STORE || st->n_src < 2) return NULL;
  PolyUOp *idx = st->src[0];
  if (!idx || idx->op != POLY_OP_INDEX || idx->n_src < 2) return NULL;
  PolyUOp *buf_base = NULL;
  int lanes = 0;
  if (!is_vectorize_of_same(idx->src[0], &buf_base, &lanes) || !buf_base || lanes <= 1) return NULL;
  if (idx->src[1]->dtype.count != lanes) return NULL;
  if (lanes > 128) return NULL;

  PolyUOp *stores[128];
  PolyDType val_sdt = poly_dtype_scalar(st->src[1]->dtype);
  for (int i = 0; i < lanes; i++) {
    PolyUOp *lane_idx = build_scalar_lane_index(ctx, idx, buf_base, i);
    if (!lane_idx) return NULL;
    PolyUOp *lane_val = st->src[1];
    if (lane_val && lane_val->dtype.count > 1) lane_val = scalarize_lane_expr(ctx, lane_val, i);
    if (!lane_val) return NULL;

    PolyUOp *lane_src[64];
    int n_lane_src = 0;
    lane_src[n_lane_src++] = lane_idx;
    if (val_sdt.count == 1 && lane_val->dtype.count > 1)
      lane_src[n_lane_src++] = poly_uop1(ctx, POLY_OP_CAST, val_sdt, lane_val, poly_arg_none());
    else
      lane_src[n_lane_src++] = lane_val;
    for (int j = 2; j < st->n_src && n_lane_src < 64; j++) lane_src[n_lane_src++] = st->src[j];
    stores[i] = poly_uop(ctx, POLY_OP_STORE, st->dtype, lane_src, n_lane_src, st->arg);
  }
  return poly_uop(ctx, POLY_OP_GROUP, POLY_VOID, stores, lanes, poly_arg_none());
}

static PolyPatternMatcher *g_pm_load_store_folding = NULL;
static PolyPatternMatcher *poly_pm_load_store_folding(void) {
  if (g_pm_load_store_folding) return g_pm_load_store_folding;
  PolyRule rules[] = {
    { poly_pat_op(POLY_OP_INDEX, NULL, 0, "idx"), rule_fold_vectorized_index },
    { poly_pat_allow_any_len(poly_pat_op(POLY_OP_LOAD, NULL, 0, "ld")), rule_split_vector_load },
    { poly_pat_allow_any_len(poly_pat_op(POLY_OP_STORE, NULL, 0, "st")), rule_split_vector_store },
  };
  g_pm_load_store_folding = poly_pm_new(rules, (int)(sizeof(rules) / sizeof(rules[0])));
  return g_pm_load_store_folding;
}

/* ── pm_render subset (constants + vector WHERE scalarization) ───────── */

static PolyUOp *rule_render_vconst(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b) {
  (void)b;
  if (u->op != POLY_OP_VCONST) return NULL;
  if (u->n_src > 0) {
    return poly_uop(ctx, POLY_OP_VECTORIZE, u->dtype, u->src, u->n_src, poly_arg_none());
  }
  if (u->arg.kind == POLY_ARG_INT_TUPLE && u->arg.int_tuple.n > 0) {
    PolyUOp *elts[128];
    int n = u->arg.int_tuple.n;
    if (n > 128) return NULL;
    PolyDType sdt = poly_dtype_scalar(u->dtype);
    for (int i = 0; i < n; i++) {
      elts[i] = poly_uop0(ctx, POLY_OP_CONST, sdt, poly_arg_int(u->arg.int_tuple.vals[i]));
    }
    return poly_uop(ctx, POLY_OP_VECTORIZE, u->dtype, elts, n, poly_arg_none());
  }
  return NULL;
}

static PolyUOp *rule_vectorize_single(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b) {
  (void)ctx; (void)b;
  if (u->op != POLY_OP_VECTORIZE || u->n_src != 1) return NULL;
  return u->src[0];
}

static PolyUOp *make_gep_lane(PolyCtx *ctx, PolyUOp *src, int lane) {
  int64_t idx = lane;
  return poly_uop1(ctx, POLY_OP_GEP, poly_dtype_scalar(src->dtype), src, poly_arg_int_tuple_local(&idx, 1));
}

static PolyUOp *lane_or_gep(PolyCtx *ctx, PolyUOp *src, int lane) {
  if (src->dtype.count <= 1) return src;
  if (src->op == POLY_OP_VECTORIZE && src->n_src > lane) return src->src[lane];
  return make_gep_lane(ctx, src, lane);
}

static PolyUOp *rule_vector_cmp_to_scalarized_vector(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b) {
  (void)b;
  if (!(u->op == POLY_OP_CMPLT || u->op == POLY_OP_CMPNE || u->op == POLY_OP_CMPEQ)) return NULL;
  if (u->n_src != 2 || u->dtype.count <= 1) return NULL;
  int lanes = u->dtype.count;
  if (lanes > 128) return NULL;
  PolyUOp *elts[128];
  for (int i = 0; i < lanes; i++) {
    PolyUOp *a = lane_or_gep(ctx, u->src[0], i);
    PolyUOp *b0 = lane_or_gep(ctx, u->src[1], i);
    elts[i] = poly_uop2(ctx, u->op, poly_dtype_scalar(u->dtype), a, b0, poly_arg_none());
  }
  return poly_uop(ctx, POLY_OP_VECTORIZE, u->dtype, elts, lanes, poly_arg_none());
}

static PolyUOp *rule_vector_where_to_scalar(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b) {
  (void)b;
  if (u->op != POLY_OP_WHERE || u->n_src != 3 || u->dtype.count <= 1) return NULL;
  int lanes = u->dtype.count;
  if (lanes > 128) return NULL;
  PolyUOp *elts[128];
  PolyDType sdt = poly_dtype_scalar(u->dtype);
  for (int i = 0; i < lanes; i++) {
    PolyUOp *c = lane_or_gep(ctx, u->src[0], i);
    PolyUOp *x = lane_or_gep(ctx, u->src[1], i);
    PolyUOp *y = lane_or_gep(ctx, u->src[2], i);
    elts[i] = poly_uop3(ctx, POLY_OP_WHERE, sdt, c, x, y, poly_arg_none());
  }
  return poly_uop(ctx, POLY_OP_VECTORIZE, u->dtype, elts, lanes, poly_arg_none());
}

static PolyUOp *rule_vector_bool_neg_to_scalarized_vector(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b) {
  (void)b;
  if (u->op != POLY_OP_NEG || u->n_src != 1 || u->dtype.count <= 1) return NULL;
  PolyDType sdt = poly_dtype_scalar(u->dtype);
  if (!poly_dtype_is_bool(sdt)) return NULL;
  int lanes = u->dtype.count;
  if (lanes > 128) return NULL;
  PolyUOp *elts[128];
  for (int i = 0; i < lanes; i++) {
    PolyUOp *x = lane_or_gep(ctx, u->src[0], i);
    elts[i] = poly_uop1(ctx, POLY_OP_NEG, sdt, x, poly_arg_none());
  }
  return poly_uop(ctx, POLY_OP_VECTORIZE, u->dtype, elts, lanes, poly_arg_none());
}

static PolyPatternMatcher *g_pm_render_subset = NULL;
static PolyPatternMatcher *poly_pm_render_subset(void) {
  if (g_pm_render_subset) return g_pm_render_subset;
  PolyRule rules[] = {
    { poly_pat_op(POLY_OP_VCONST, NULL, 0, "u"), rule_render_vconst },
    { poly_pat_op(POLY_OP_CMPLT, NULL, 0, "u"), rule_vector_cmp_to_scalarized_vector },
    { poly_pat_op(POLY_OP_CMPNE, NULL, 0, "u"), rule_vector_cmp_to_scalarized_vector },
    { poly_pat_op(POLY_OP_CMPEQ, NULL, 0, "u"), rule_vector_cmp_to_scalarized_vector },
    { poly_pat_op(POLY_OP_WHERE, NULL, 0, "u"), rule_vector_where_to_scalar },
    { poly_pat_op(POLY_OP_NEG, NULL, 0, "u"), rule_vector_bool_neg_to_scalarized_vector },
    { poly_pat_op(POLY_OP_VECTORIZE, NULL, 0, "u"), rule_vectorize_single },
  };
  g_pm_render_subset = poly_pm_new(rules, (int)(sizeof(rules) / sizeof(rules[0])));
  return g_pm_render_subset;
}

/* ── GPU dims: replace outermost RANGE with SPECIAL ─────────────────── */

/*
 * poly_add_gpudims — Port of tinygrad's pm_add_gpudims (simplified).
 *
 * After full_rewrite_to_sink, the graph has RANGE ops for data loops and
 * reduce loops.  This pass replaces the outermost non-reduce RANGE with
 * a SPECIAL("gidx0", N) op for GPU thread indexing.
 *
 * Reduce ranges (those that appear inside DEFINE_REG→AFTER→RANGE patterns)
 * are left as serial loops.
 *
 * Algorithm:
 * 1. Toposort to find all RANGE ops
 * 2. Identify reduce ranges: any RANGE that is src[2+] of an AFTER whose
 *    src[0] is a DEFINE_REG (these are the inner loops of accumulations)
 * 3. The first non-reduce RANGE → replace with SPECIAL
 * 4. Remove corresponding END ops for that RANGE
 */
PolyUOp *poly_add_gpudims(PolyCtx *ctx, PolyUOp *sink) {
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, sink, &n_topo);

  /* Find reduce ranges: RANGE nodes that appear as src[2+] of AFTER nodes
   * where src[0] traces back to DEFINE_REG. */
  PolyUOp *reduce_ranges[POLY_MAX_DIMS];
  int n_reduce = 0;

  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    if (u->op != POLY_OP_AFTER || u->n_src < 3) continue;
    /* Check if src[0] is DEFINE_REG (directly or via AFTER chain) */
    PolyUOp *base = u->src[0];
    while (base->op == POLY_OP_AFTER && base->n_src > 0) base = base->src[0];
    if (base->op != POLY_OP_DEFINE_REG) continue;
    /* src[2+] are reduce ranges */
    for (int j = 2; j < u->n_src; j++) {
      if (u->src[j]->op == POLY_OP_RANGE && n_reduce < POLY_MAX_DIMS) {
        bool dup = false;
        for (int k = 0; k < n_reduce; k++) {
          if (reduce_ranges[k] == u->src[j]) { dup = true; break; }
        }
        if (!dup) reduce_ranges[n_reduce++] = u->src[j];
      }
    }
  }

  /* Separate group ranges (GROUP_REDUCE axis type, from group_for_reduce) and
   * find the first non-reduce, non-group global RANGE. */
  PolyUOp *group_ranges[POLY_MAX_DIMS];
  int n_group = 0;
  PolyUOp *target_range = NULL;

  for (int i = 0; i < n_topo; i++) {
    if (topo[i]->op != POLY_OP_RANGE) continue;
    /* Group range (from group_for_reduce): axis type GROUP_REDUCE.
     * Migration fallback keeps old int-id heuristic for pre-range-typed IR. */
    bool is_group = (poly_arg_is_range(topo[i]->arg) &&
                     poly_range_axis_type(topo[i]->arg) == POLY_AXIS_GROUP_REDUCE) ||
                    (topo[i]->arg.kind == POLY_ARG_INT && topo[i]->arg.i >= 1000);
    if (is_group && n_group < POLY_MAX_DIMS) {
      group_ranges[n_group++] = topo[i];
      continue;
    }
    /* Reduce range: skip */
    bool is_reduce = false;
    for (int j = 0; j < n_reduce; j++) {
      if (reduce_ranges[j] == topo[i]) { is_reduce = true; break; }
    }
    if (!is_reduce && !target_range) { target_range = topo[i]; }
  }

  if (!target_range && n_group == 0) return sink;  /* nothing to parallelize */

  /* Substitute: replace target_range→gidx SPECIAL, group_ranges→lidx SPECIAL,
   * and remove corresponding END ops.
   * Rebuild graph bottom-up using a flat pointer-identity map. */
  PolyUOp **sub_old = (PolyUOp **)malloc((size_t)(n_topo + 64) * sizeof(PolyUOp *));
  PolyUOp **sub_new = (PolyUOp **)malloc((size_t)(n_topo + 64) * sizeof(PolyUOp *));
  int n_subs = 0;

  /* Seed: target_range → gidx SPECIAL */
  if (target_range) {
    PolyUOp *dim_size = target_range->src[0];
    PolyUOp *special = poly_uop1(ctx, POLY_OP_SPECIAL, POLY_INT32,
                                 dim_size, poly_arg_str("gidx0"));
    sub_old[n_subs] = target_range;
    sub_new[n_subs] = special;
    n_subs++;
  }

  /* Seed: group ranges → lidx SPECIAL */
  for (int g = 0; g < n_group; g++) {
    PolyUOp *gdim = group_ranges[g]->src[0];
    char lidx_name[16];
    snprintf(lidx_name, sizeof(lidx_name), "lidx%d", g);
    PolyUOp *lidx_special = poly_uop1(ctx, POLY_OP_SPECIAL, POLY_INT32,
                                      gdim, poly_arg_str(lidx_name));
    sub_old[n_subs] = group_ranges[g];
    sub_new[n_subs] = lidx_special;
    n_subs++;
  }

  PolyUOp *new_sink = sink;
  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    if (u == target_range) continue;
    /* Skip group ranges too (they've been substituted) */
    bool is_group = false;
    for (int g = 0; g < n_group; g++) {
      if (u == group_ranges[g]) { is_group = true; break; }
    }
    if (is_group) continue;

    /* END ops referencing the target range or group ranges → collapse to src[0] */
    if (u->op == POLY_OP_END) {
      bool refs_target = false;
      for (int j = 1; j < u->n_src; j++) {
        if (u->src[j] == target_range) { refs_target = true; break; }
        for (int g = 0; g < n_group; g++) {
          if (u->src[j] == group_ranges[g]) { refs_target = true; break; }
        }
        if (refs_target) break;
      }
      if (refs_target) {
        PolyUOp *repl = u->src[0];
        /* Lookup if src[0] was substituted */
        for (int k = 0; k < n_subs; k++) {
          if (sub_old[k] == repl) { repl = sub_new[k]; break; }
        }
        sub_old[n_subs] = u;
        sub_new[n_subs] = repl;
        n_subs++;
        continue;
      }
    }

    /* Check if any source was substituted */
    bool changed = false;
    PolyUOp *new_srcs[64];
    for (int j = 0; j < u->n_src && j < 64; j++) {
      PolyUOp *mapped = NULL;
      for (int k = 0; k < n_subs; k++) {
        if (sub_old[k] == u->src[j]) { mapped = sub_new[k]; break; }
      }
      if (mapped) { new_srcs[j] = mapped; changed = true; }
      else { new_srcs[j] = u->src[j]; }
    }

    if (changed) {
      PolyUOp *new_u = poly_uop(ctx, u->op, u->dtype, new_srcs, u->n_src, u->arg);
      sub_old[n_subs] = u;
      sub_new[n_subs] = new_u;
      n_subs++;
      if (u == sink) new_sink = new_u;
    }
  }

  free(sub_old);
  free(sub_new);
  return new_sink;
}

/* ── Group for reduce: parallel reduction via shared memory ──────────── */

/*
 * poly_group_for_reduce — Port of tinygrad's fix_group_for_reduce.
 *
 * Splits large REDUCE ops into block-level parallel reductions:
 *   REDUCE(op, val, [RANGE(N)]) →
 *     1. Per-thread partial: REDUCE(op, safe_val, [serial_range]) where each
 *        of block_size threads handles ceil(N/block_size) elements
 *     2. DEFINE_LOCAL shared memory + STORE + BARRIER
 *     3. Final serial: REDUCE(op, smem_load, [final_range]) over block_size
 *     4. IF guard: only thread 0 stores the output
 *
 * Uses axis ID conventions:
 *   - Group range (→ lidx):    arg = original_axis + 1000
 *   - Serial reduce range:     arg = original_axis + 100
 *   - Final reduce range:      arg = original_axis + 200
 */

/* Recursively clone a subtree, substituting old_node → new_node */
static PolyUOp *substitute_node(PolyCtx *ctx, PolyUOp *node,
                                PolyUOp *old_node, PolyUOp *new_node,
                                PolyUOp **memo_old, PolyUOp **memo_new,
                                int *memo_n, int memo_cap) {
  if (node == old_node) return new_node;
  /* Check memo */
  for (int i = 0; i < *memo_n; i++)
    if (memo_old[i] == node) return memo_new[i];

  /* Recurse on sources */
  bool changed = false;
  PolyUOp *new_srcs[64];
  int ns = node->n_src < 64 ? node->n_src : 64;
  for (int i = 0; i < ns; i++) {
    new_srcs[i] = substitute_node(ctx, node->src[i], old_node, new_node,
                                   memo_old, memo_new, memo_n, memo_cap);
    if (new_srcs[i] != node->src[i]) changed = true;
  }

  if (!changed) return node;  /* subtree unchanged */

  PolyUOp *result = poly_uop(ctx, node->op, node->dtype, new_srcs, ns, node->arg);
  if (*memo_n < memo_cap) {
    memo_old[*memo_n] = node;
    memo_new[*memo_n] = result;
    (*memo_n)++;
  }
  return result;
}

PolyUOp *poly_group_for_reduce(PolyCtx *ctx, PolyUOp *sink, int block_size) {
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, sink, &n_topo);

  /* Find all REDUCE ops */
  PolyUOp *reduces[32];
  int n_reduces = 0;
  for (int i = 0; i < n_topo; i++) {
    if (topo[i]->op == POLY_OP_REDUCE && n_reduces < 32)
      reduces[n_reduces++] = topo[i];
  }

  if (n_reduces == 0) return sink;

  /* Process each REDUCE: substitute in the full graph */
  PolyUOp **sub_old = malloc((size_t)(n_topo + 256) * sizeof(PolyUOp *));
  PolyUOp **sub_new = malloc((size_t)(n_topo + 256) * sizeof(PolyUOp *));
  int n_subs = 0;

  for (int r = 0; r < n_reduces; r++) {
    PolyUOp *red = reduces[r];
    PolyUOp *val = red->src[0];
    int n_reduce_range = red->n_src - 1;

    /* Only handle single-range reduces for now */
    if (n_reduce_range != 1) continue;

    PolyUOp *orig_range = red->src[1];
    if (orig_range->op != POLY_OP_RANGE) continue;
    if (orig_range->src[0]->op != POLY_OP_CONST) continue;

    int64_t N = orig_range->src[0]->arg.i;
    if (N <= block_size * 2) continue;  /* too small to parallelize */

    PolyOps reduce_op = red->arg.ops;
    int64_t orig_axis = poly_range_axis_id(orig_range->arg);
    int64_t serial_N = (N + block_size - 1) / block_size;

    /* Identity element for this reduce op */
    double ident_val = codegen_reduce_identity(reduce_op);

    /* ── Create group range (→ threadIdx.x / lidx0) ────────────────── */
    PolyUOp *group_bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32,
                                     poly_arg_int(block_size));
    PolyUOp *group_range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32,
                                     group_bound, poly_arg_range(orig_axis + 1000, POLY_AXIS_GROUP_REDUCE));

    /* ── Create serial range (per-thread iterations) ───────────────── */
    PolyUOp *serial_bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32,
                                      poly_arg_int(serial_N));
    PolyUOp *serial_range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32,
                                      serial_bound, poly_arg_range(orig_axis + 100, POLY_AXIS_REDUCE));

    /* ── Compound index: serial * block_size + group (coalesced) ───── */
    PolyUOp *bs_const = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32,
                                  poly_arg_int(block_size));
    PolyUOp *compound = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32,
                                  poly_uop2(ctx, POLY_OP_MUL, POLY_INT32,
                                            serial_range, bs_const, poly_arg_none()),
                                  group_range, poly_arg_none());

    /* ── Substitute original range → compound in value expression ──── */
    PolyUOp *memo_old_sub[4096];
    PolyUOp *memo_new_sub[4096];
    int memo_n = 0;
    PolyUOp *subst_val = substitute_node(ctx, val, orig_range, compound,
                                         memo_old_sub, memo_new_sub,
                                         &memo_n, 4096);

    /* ── Bounds check: WHERE(compound < N, substituted_val, identity) ─ */
    PolyUOp *n_const = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(N));
    PolyUOp *bounds_cmp = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL,
                                    compound, n_const, poly_arg_none());
    PolyUOp *identity;
    if (poly_dtype_is_float(red->dtype))
      identity = poly_uop0(ctx, POLY_OP_CONST, red->dtype,
                            poly_arg_float(ident_val));
    else
      identity = poly_uop0(ctx, POLY_OP_CONST, red->dtype,
                            poly_arg_int((int64_t)ident_val));
    PolyUOp *safe_val = poly_uop3(ctx, POLY_OP_WHERE, red->dtype,
                                  bounds_cmp, subst_val, identity,
                                  poly_arg_none());

    /* ── First REDUCE: per-thread partial ──────────────────────────── */
    PolyUOp *partial_srcs[2] = { safe_val, serial_range };
    PolyUOp *partial_reduce = poly_uop(ctx, POLY_OP_REDUCE, red->dtype,
                                       partial_srcs, 2, red->arg);

    /* ── DEFINE_LOCAL: shared memory buffer ────────────────────────── */
    PolyDType smem_ptr = poly_dtype_ptr(poly_dtype_scalar(red->dtype),
                                       block_size, POLY_ADDR_LOCAL);
    PolyUOp *smem = poly_uop0(ctx, POLY_OP_DEFINE_LOCAL, smem_ptr,
                              poly_arg_int(0));

    /* ── STORE partial → smem[group_range] ────────────────────────── */
    PolyUOp *smem_store_idx = poly_uop2(ctx, POLY_OP_INDEX, smem_ptr,
                                        smem, group_range, poly_arg_none());
    PolyUOp *smem_store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID,
                                    smem_store_idx, partial_reduce,
                                    poly_arg_none());

    /* ── BARRIER ───────────────────────────────────────────────────── */
    PolyUOp *barrier = poly_uop1(ctx, POLY_OP_BARRIER, POLY_VOID,
                                 smem_store, poly_arg_none());

    /* ── Final reduce range ────────────────────────────────────────── */
    PolyUOp *final_bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32,
                                     poly_arg_int(block_size));
    PolyUOp *final_range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32,
                                     final_bound, poly_arg_range(orig_axis + 200, POLY_AXIS_REDUCE));

    /* ── LOAD from smem after barrier ──────────────────────────────── */
    PolyUOp *smem_after_srcs[2] = { smem, barrier };
    PolyUOp *smem_after = poly_uop(ctx, POLY_OP_AFTER, smem_ptr,
                                   smem_after_srcs, 2, poly_arg_none());
    PolyUOp *final_load_idx = poly_uop2(ctx, POLY_OP_INDEX, smem_ptr,
                                        smem_after, final_range,
                                        poly_arg_none());
    PolyUOp *final_load = poly_uop1(ctx, POLY_OP_LOAD, red->dtype,
                                    final_load_idx, poly_arg_none());

    /* ── Second REDUCE: across shared memory ───────────────────────── */
    PolyUOp *final_red_srcs[2] = { final_load, final_range };
    PolyUOp *final_reduce = poly_uop(ctx, POLY_OP_REDUCE, red->dtype,
                                     final_red_srcs, 2, red->arg);

    /* Register substitution: original REDUCE → final_reduce.
     * The caller's STORE will now consume final_reduce.
     * We also need to add a guard (IF/ENDIF) around the STORE. */
    sub_old[n_subs] = red;
    sub_new[n_subs] = final_reduce;
    n_subs++;

    /* Find the STORE that consumes this REDUCE and wrap it with IF guard.
     * We mark the group_range for the guard. The guard is created here as
     * a new output that replaces the original STORE's value. */
  }

  if (n_subs == 0) {
    free(sub_old);
    free(sub_new);
    return sink;
  }

  /* Apply substitutions bottom-up through the graph.
   * Also wrap STORE ops that consumed a transformed REDUCE with IF guard. */

  /* Collect group ranges from substitutions to add IF guards */
  PolyUOp *group_range_for_guard = NULL;
  for (int i = 0; i < n_subs; i++) {
    /* The substituted reduce's final value traces back to a group range.
     * Find the group range by looking at the new reduce's ancestry. */
    PolyUOp *nr = sub_new[i];
    /* Walk the final reduce's load → after → smem → and find the group range
     * by looking for RANGE with arg >= 1000 in the toposort */
    int nt = 0;
    PolyUOp **sub_topo = poly_toposort(ctx, nr, &nt);
    for (int j = 0; j < nt; j++) {
      if (sub_topo[j]->op == POLY_OP_RANGE &&
          poly_arg_is_range(sub_topo[j]->arg) &&
          poly_range_axis_type(sub_topo[j]->arg) == POLY_AXIS_GROUP_REDUCE) {
        group_range_for_guard = sub_topo[j];
        break;
      }
    }
    break;  /* only handle first reduce for now */
  }

  /* Bottom-up graph rebuild with substitutions */
  PolyUOp *new_sink = sink;

  /* Re-toposort since we modified things */
  int n_topo2 = 0;
  PolyUOp **topo2 = poly_toposort(ctx, sink, &n_topo2);

  for (int i = 0; i < n_topo2; i++) {
    PolyUOp *u = topo2[i];

    /* Check if this node itself was substituted */
    bool is_subst = false;
    for (int k = 0; k < n_subs; k++) {
      if (sub_old[k] == u) { is_subst = true; break; }
    }
    if (is_subst) continue;

    /* Check if any source was substituted */
    bool changed = false;
    PolyUOp *new_srcs[64];
    int ns = u->n_src < 64 ? u->n_src : 64;
    for (int j = 0; j < ns; j++) {
      PolyUOp *mapped = NULL;
      for (int k = 0; k < n_subs; k++) {
        if (sub_old[k] == u->src[j]) { mapped = sub_new[k]; break; }
      }
      if (mapped) { new_srcs[j] = mapped; changed = true; }
      else { new_srcs[j] = u->src[j]; }
    }

    if (changed) {
      /* If this is a STORE consuming a transformed reduce, wrap with IF guard */
      if (u->op == POLY_OP_STORE && group_range_for_guard) {
        /* Check if value (src[1]) was a substituted reduce */
        bool val_subst = false;
        for (int k = 0; k < n_subs; k++) {
          if (sub_old[k] == u->src[1]) { val_subst = true; break; }
        }
        if (val_subst) {
          /* Create: IF(group_range < 1) { STORE(...) } ENDIF */
          PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(1));
          PolyUOp *guard = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL,
                                     group_range_for_guard, one, poly_arg_none());
          PolyUOp *if_op = poly_uop1(ctx, POLY_OP_IF, POLY_VOID,
                                     guard, poly_arg_none());
          PolyUOp *new_store = poly_uop(ctx, u->op, u->dtype,
                                        new_srcs, ns, u->arg);
          PolyUOp *endif_srcs[2] = { new_store, if_op };
          PolyUOp *endif_op = poly_uop(ctx, POLY_OP_ENDIF, POLY_VOID,
                                       endif_srcs, 2, poly_arg_none());
          sub_old[n_subs] = u;
          sub_new[n_subs] = endif_op;
          n_subs++;
          if (u == sink) new_sink = endif_op;
          continue;
        }
      }

      PolyUOp *new_u = poly_uop(ctx, u->op, u->dtype, new_srcs, ns, u->arg);
      sub_old[n_subs] = u;
      sub_new[n_subs] = new_u;
      n_subs++;
      if (u == sink) new_sink = new_u;
    }
  }

  free(sub_old);
  free(sub_new);
  return new_sink;
}

/* ── Public accessors for individual passes (used by CUDA linearizer) ── */

PolyPatternMatcher *poly_pm_reduce_pass(void)        { return poly_pm_reduce(); }
PolyPatternMatcher *poly_pm_decomp_pass(void)         { return poly_pm_decomp(); }
PolyPatternMatcher *poly_pm_transcendental_pass(void)  { return poly_pm_transcendental(); }
void poly_reset_acc_num(void)                         { }

PolyUOp *poly_apply_pm_reduce(PolyCtx *ctx, PolyUOp *sink) {
  ReduceContext local_ctx = {0};
  PolyUOp *out = poly_graph_rewrite_ctx(ctx, sink, poly_pm_reduce(), &local_ctx);
  reduce_ctx_clear(&local_ctx);
  return out;
}

/* ── Full rewrite-to-sink pipeline ───────────────────────────────────── */

PolyUOp *poly_full_rewrite_to_sink_ex(PolyCtx *ctx, PolyUOp *sink, PolyRewriteOpts opts) {
  if (opts.optimize && getenv("POLY_EXPERIMENTAL_LATE")) {
    /* tinygrad preprocess: split/flatten/simplify ranges before late lowering. */
    SplitRangeCtx srctx = {0};
    sink = poly_graph_rewrite_ctx(ctx, sink, poly_pm_split_ranges(), &srctx);
    sink = poly_graph_rewrite(ctx, sink, poly_pm_flatten_range());
    sink = poly_graph_rewrite(ctx, sink, poly_symbolic_simple());
    sink = poly_graph_rewrite(ctx, sink, poly_pm_flatten_range());
    sink = poly_graph_rewrite(ctx, sink, poly_pm_simplify_ranges());

    sink = poly_apply_opts_basic(ctx, sink);
    sink = poly_graph_rewrite(ctx, sink, poly_symbolic_simple());
    sink = poly_graph_rewrite(ctx, sink, poly_pm_pre_expander());
    sink = poly_graph_rewrite(ctx, sink, poly_pm_expander());
    sink = poly_graph_rewrite(ctx, sink, poly_pm_add_loads());
    sink = poly_graph_rewrite(ctx, sink, poly_pm_load_store_folding());
    sink = poly_graph_rewrite(ctx, sink, poly_pm_render_subset());
  }

  /* 1. Symbolic simplification */
  sink = poly_graph_rewrite(ctx, sink, poly_symbolic_simple());

  /* 2. pm_reduce: REDUCE → DEFINE_REG + END merge on SINK */
  sink = poly_apply_pm_reduce(ctx, sink);

  /* 3. Post-reduce symbolic simplification */
  sink = poly_graph_rewrite(ctx, sink, poly_symbolic_simple());
  if (opts.optimize && getenv("POLY_EXPERIMENTAL_LATE")) {
    sink = poly_graph_rewrite(ctx, sink, poly_pm_render_subset());
  }

  /* 4. pm_decomp: late decompositions (MAX→WHERE, MUL→SHL, IDIV→SHR) */
  sink = poly_graph_rewrite(ctx, sink, poly_pm_decomp());

  /* 5. pm_transcendental: EXP2 → polynomial (creates new IDIV ops) */
  sink = poly_graph_rewrite(ctx, sink, poly_pm_transcendental());

  /* 6. Final decomp: catch IDIV ops created by transcendental pass */
  sink = poly_graph_rewrite(ctx, sink, poly_pm_decomp());

  /* 7. Expander cleanup: remove identity UNROLL/CONTRACT nodes */
  sink = poly_graph_rewrite(ctx, sink, poly_pm_expander());
  if (opts.optimize && getenv("POLY_EXPERIMENTAL_LATE")) {
    sink = poly_graph_rewrite(ctx, sink, poly_pm_render_subset());
  }

  return sink;
}

PolyUOp *poly_full_rewrite_to_sink(PolyCtx *ctx, PolyUOp *sink) {
  PolyRewriteOpts opts = { .optimize = false, .devectorize = 0 };
  return poly_full_rewrite_to_sink_ex(ctx, sink, opts);
}
