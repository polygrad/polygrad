/*
 * codegen.c — Sequential rewrite passes (port of full_rewrite_to_sink)
 *
 * Mirrors tinygrad's codegen/__init__.py full_rewrite_to_sink pipeline.
 * Currently implements:
 *   - pm_reduce: REDUCE → DEFINE_REG + AFTER accumulation + END merge
 *   - pm_decomp: MAX→WHERE, MUL→SHL, IDIV→SHR (late decompositions)
 *   - pm_transcendental: EXP2 → polynomial approximation (xexp2)
 */

#define _POSIX_C_SOURCE 200809L

#include "codegen.h"
#include "exec_plan.h"
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#include <errno.h>

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

/* Max hardware vector fold width for load/store splitting.
 * Set by the pipeline before running correct_load_store pass.
 * Default 4 (SSE). Set to 8 for AVX2.
 * NOT THREAD-SAFE: concurrent codegen with different caps will race.
 * Acceptable for now (single-threaded); thread-safe fix requires passing
 * fold width through rewrite context instead of a mutable global. */
static int g_max_fold_width = 4;

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

int range_start_for_op(PolyOps op) {
  switch (op) {
    case POLY_OP_BUFFERIZE: return 1;
    case POLY_OP_REDUCE: return 1;
    case POLY_OP_STORE: return 2;
    case POLY_OP_WMMA: return 3;
    case POLY_OP_END: return 1;
    case POLY_OP_CALL: return 1;
    case POLY_OP_COPY: return 2;
    case POLY_OP_BUFFER_VIEW: return 1;
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
  /* Preserve tag: this is a replace(src=...) operation, matching tinygrad's
   * UOp.replace() which preserves tag (ops.py:142). */
  return (root->tag != 0)
    ? poly_uop_tagged(ctx, root->op, root->dtype, new_src, n_new, root->arg, root->tag)
    : poly_uop(ctx, root->op, root->dtype, new_src, n_new, root->arg);
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

/* ── Heuristic optimizer (port of tinygrad hand_coded_optimizations) ──── */

/* axis_to_pos ordering: matches tinygrad's axis_to_pos dict.
 * LOOP:-1, THREAD:0, GLOBAL:0, WARP:1, LOCAL:2, GROUP_REDUCE:2,
 * UPCAST:3, REDUCE:4, UNROLL:5 */
static int axis_to_pos(PolyAxisType t) {
  switch (t) {
    case POLY_AXIS_LOOP:          return -1;
    case POLY_AXIS_THREAD:        return 0;
    case POLY_AXIS_GLOBAL:        return 0;
    case POLY_AXIS_WARP:          return 1;
    case POLY_AXIS_LOCAL:         return 2;
    case POLY_AXIS_GROUP_REDUCE:  return 2;
    case POLY_AXIS_UPCAST:        return 3;
    case POLY_AXIS_REDUCE:        return 4;
    case POLY_AXIS_UNROLL:        return 5;
    default:                      return 6;
  }
}

/* Scheduler state for heuristic optimizer.
 * Matches tinygrad's Scheduler class (postrange.py:17-331). */
#define SCHED_MAX_RNGS 64
#define SCHED_MAX_BUFS 32

typedef struct {
  PolyCtx *ctx;
  PolyUOp *ast;               /* current kernel SINK */
  int64_t opt_range_next;      /* counter for new axis IDs */

  /* Sorted RANGE list (by axis_to_pos then axis_id) */
  PolyUOp *rngs[SCHED_MAX_RNGS];
  int n_rngs;

  /* Shape (bound of each range) */
  int64_t shape[SCHED_MAX_RNGS];

  /* Axis types */
  PolyAxisType types[SCHED_MAX_RNGS];

  /* INDEX ops (tinygrad k.bufs) - reversed toposort order */
  PolyUOp *bufs[SCHED_MAX_BUFS];
  int n_bufs;

  /* Reachability bitmask per buffer: buf_reach[bi] has bit j set if rngs[j]
   * is reachable from bufs[bi]'s source tree. n_rngs must be <= 64. */
  uint64_t buf_reach[SCHED_MAX_BUFS];
  bool has_reach;

  /* Has reduce op */
  bool has_reduce;
} OptScheduler;

static int sched_rng_cmp(const void *ap, const void *bp) {
  const PolyUOp *a = *(const PolyUOp *const *)ap;
  const PolyUOp *b = *(const PolyUOp *const *)bp;
  int pa = axis_to_pos(poly_range_axis_type(a->arg));
  int pb = axis_to_pos(poly_range_axis_type(b->arg));
  if (pa != pb) return pa - pb;
  int64_t ia = poly_range_axis_id(a->arg);
  int64_t ib = poly_range_axis_id(b->arg);
  if (ia != ib) return (ia < ib) ? -1 : 1;
  return 0;
}

/* Build reachability bitmask for all nodes in a toposort.
 * Uses a single forward pass: each node's bitmask = union of its sources' bitmasks.
 * RANGE nodes set their own bit. Result: reachable[i] has bit j set iff rngs[j]
 * is reachable from topo[i]'s source tree.
 *
 * n_rngs must be <= 64 (SCHED_MAX_RNGS). Returns malloc'd array (caller frees).
 * topo_map is used to look up topo index for a UOp pointer. */
static uint64_t *build_reachability_bitmask(PolyUOp **topo, int n_topo,
                                            PolyUOp **rngs, int n_rngs) {
  /* Build ptr→index map for O(1) source lookup */
  PolyMap *idx_map = poly_map_new((size_t)(n_topo < 64 ? 64 : (size_t)n_topo * 2));
  /* Store topo index + 1 (so 0 means "not found") */
  int *indices = (int *)malloc((size_t)n_topo * sizeof(int));
  for (int i = 0; i < n_topo; i++) {
    indices[i] = i + 1; /* 1-based so NULL means "not in map" */
    poly_map_set(idx_map, ptr_hash(topo[i]), topo[i], &indices[i], ptr_eq);
  }

  /* Build range_index: for each range UOp, which bit index */
  uint64_t *reach = (uint64_t *)calloc((size_t)n_topo, sizeof(uint64_t));

  /* Set bits for RANGE nodes */
  for (int i = 0; i < n_topo; i++) {
    if (topo[i]->op == POLY_OP_RANGE) {
      for (int ri = 0; ri < n_rngs; ri++) {
        if (rngs[ri] == topo[i]) reach[i] |= (1ULL << ri);
      }
    }
  }

  /* Forward pass: propagate bits from sources */
  for (int i = 0; i < n_topo; i++) {
    for (int j = 0; j < topo[i]->n_src; j++) {
      int *pidx = (int *)poly_map_get(idx_map, ptr_hash(topo[i]->src[j]),
                                       topo[i]->src[j], ptr_eq);
      if (pidx) reach[i] |= reach[*pidx - 1];
    }
  }

  poly_map_destroy(idx_map);
  free(indices);
  return reach;
}

static void sched_init(OptScheduler *s, PolyCtx *ctx, PolyUOp *sink) {
  s->ctx = ctx;
  s->ast = sink;
  s->n_rngs = 0;
  s->n_bufs = 0;
  s->has_reduce = false;
  s->has_reach = false;
  for (int i = 0; i < SCHED_MAX_BUFS; i++) s->buf_reach[i] = 0;

  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, sink, &n_topo);

  /* Collect unique RANGE ops with vmax > 0 and INDEX ops */
  int64_t max_id = -1;
  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    if (u->op == POLY_OP_REDUCE || u->op == POLY_OP_REDUCE_AXIS)
      s->has_reduce = true;
    if (u->op == POLY_OP_RANGE && poly_arg_is_range(u->arg)) {
      /* Check vmax > 0 (i.e., bound > 1 or bound > 0) */
      int64_t bound = 0;
      if (u->n_src > 0 && u->src[0]->op == POLY_OP_CONST && u->src[0]->arg.kind == POLY_ARG_INT)
        bound = u->src[0]->arg.i;
      if (bound <= 1) continue; /* vmax = bound - 1, so vmax > 0 means bound > 1 */
      bool dup = false;
      for (int j = 0; j < s->n_rngs; j++) {
        if (s->rngs[j] == u) { dup = true; break; }
      }
      if (!dup && s->n_rngs < SCHED_MAX_RNGS) {
        s->rngs[s->n_rngs] = u;
        s->n_rngs++;
      }
      int64_t aid = poly_range_axis_id(u->arg);
      if (aid > max_id) max_id = aid;
    }
    if (u->op == POLY_OP_INDEX && u->n_src > 0 && u->src[0]->op == POLY_OP_PARAM) {
      if (s->n_bufs < SCHED_MAX_BUFS) s->bufs[s->n_bufs++] = u;
    }
  }
  s->opt_range_next = max_id + 1;

  /* Sort ranges by axis_to_pos ordering */
  qsort(s->rngs, (size_t)s->n_rngs, sizeof(PolyUOp *), sched_rng_cmp);

  /* Reverse bufs order to match tinygrad ([::-1]) */
  for (int i = 0; i < s->n_bufs / 2; i++) {
    PolyUOp *tmp = s->bufs[i];
    s->bufs[i] = s->bufs[s->n_bufs - 1 - i];
    s->bufs[s->n_bufs - 1 - i] = tmp;
  }

  /* Extract shapes and types */
  for (int i = 0; i < s->n_rngs; i++) {
    PolyUOp *r = s->rngs[i];
    s->types[i] = poly_range_axis_type(r->arg);
    s->shape[i] = (r->n_src > 0 && r->src[0]->op == POLY_OP_CONST && r->src[0]->arg.kind == POLY_ARG_INT)
                    ? r->src[0]->arg.i : 0;
  }

  /* Build reachability bitmask: single forward pass over toposort */
  if (s->n_bufs > 0 && s->n_rngs > 0 && s->n_rngs <= 64) {
    uint64_t *reach = build_reachability_bitmask(topo, n_topo, s->rngs, s->n_rngs);
    /* Extract per-buffer bitmasks */
    PolyMap *idx_map = poly_map_new((size_t)(n_topo < 64 ? 64 : (size_t)n_topo * 2));
    int *indices = (int *)malloc((size_t)n_topo * sizeof(int));
    for (int i = 0; i < n_topo; i++) {
      indices[i] = i;
      poly_map_set(idx_map, ptr_hash(topo[i]), topo[i], &indices[i], ptr_eq);
    }
    for (int bi = 0; bi < s->n_bufs; bi++) {
      int *pidx = (int *)poly_map_get(idx_map, ptr_hash(s->bufs[bi]),
                                       s->bufs[bi], ptr_eq);
      if (pidx) s->buf_reach[bi] = reach[*pidx];
    }
    s->has_reach = true;
    poly_map_destroy(idx_map);
    free(indices);
    free(reach);
  }
}

/* Refresh rngs, shapes, types after a shift_to modifies the AST */
static void sched_refresh(OptScheduler *s) {
  s->n_rngs = 0;
  s->n_bufs = 0;
  s->has_reduce = false;

  int n_topo = 0;
  PolyUOp **topo = poly_toposort(s->ctx, s->ast, &n_topo);
  int64_t max_id = -1;

  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    if (u->op == POLY_OP_REDUCE || u->op == POLY_OP_REDUCE_AXIS)
      s->has_reduce = true;
    if (u->op == POLY_OP_RANGE && poly_arg_is_range(u->arg)) {
      int64_t bound = 0;
      if (u->n_src > 0 && u->src[0]->op == POLY_OP_CONST && u->src[0]->arg.kind == POLY_ARG_INT)
        bound = u->src[0]->arg.i;
      if (bound <= 1) continue;
      bool dup = false;
      for (int j = 0; j < s->n_rngs; j++) {
        if (s->rngs[j] == u) { dup = true; break; }
      }
      if (!dup && s->n_rngs < SCHED_MAX_RNGS) s->rngs[s->n_rngs++] = u;
      int64_t aid = poly_range_axis_id(u->arg);
      if (aid > max_id) max_id = aid;
    }
    if (u->op == POLY_OP_INDEX && u->n_src > 0 && u->src[0]->op == POLY_OP_PARAM) {
      if (s->n_bufs < SCHED_MAX_BUFS) s->bufs[s->n_bufs++] = u;
    }
  }
  if (max_id + 1 > s->opt_range_next) s->opt_range_next = max_id + 1;

  qsort(s->rngs, (size_t)s->n_rngs, sizeof(PolyUOp *), sched_rng_cmp);
  for (int i = 0; i < s->n_bufs / 2; i++) {
    PolyUOp *tmp = s->bufs[i];
    s->bufs[i] = s->bufs[s->n_bufs - 1 - i];
    s->bufs[s->n_bufs - 1 - i] = tmp;
  }
  for (int i = 0; i < s->n_rngs; i++) {
    PolyUOp *r = s->rngs[i];
    s->types[i] = poly_range_axis_type(r->arg);
    s->shape[i] = (r->n_src > 0 && r->src[0]->op == POLY_OP_CONST && r->src[0]->arg.kind == POLY_ARG_INT)
                    ? r->src[0]->arg.i : 0;
  }

  /* Rebuild reachability bitmask */
  for (int i = 0; i < SCHED_MAX_BUFS; i++) s->buf_reach[i] = 0;
  s->has_reach = false;
  if (s->n_bufs > 0 && s->n_rngs > 0 && s->n_rngs <= 64) {
    int n_topo2 = 0;
    PolyUOp **topo2 = poly_toposort(s->ctx, s->ast, &n_topo2);
    uint64_t *reach = build_reachability_bitmask(topo2, n_topo2, s->rngs, s->n_rngs);
    PolyMap *idx_map = poly_map_new((size_t)(n_topo2 < 64 ? 64 : (size_t)n_topo2 * 2));
    int *indices = (int *)malloc((size_t)n_topo2 * sizeof(int));
    for (int i = 0; i < n_topo2; i++) {
      indices[i] = i;
      poly_map_set(idx_map, ptr_hash(topo2[i]), topo2[i], &indices[i], ptr_eq);
    }
    for (int bi = 0; bi < s->n_bufs; bi++) {
      int *pidx = (int *)poly_map_get(idx_map, ptr_hash(s->bufs[bi]),
                                       s->bufs[bi], ptr_eq);
      if (pidx) s->buf_reach[bi] = reach[*pidx];
    }
    s->has_reach = true;
    poly_map_destroy(idx_map);
    free(indices);
    free(reach);
  }
}

/* shift_to_ex: split a RANGE into two. Port of tinygrad Scheduler.shift_to.
 * top=false: old_range = replaced * amount + new_rng
 * top=true:  old_range = new_rng * old_sz + replaced
 * input_new_rng: if non-NULL, use this expression instead of creating a fresh RANGE.
 *   This is used for TC WARP modular arithmetic (e.g. warp%2).
 * out_new_rng: if non-NULL, receives the new range expression (the split-off part).
 * Returns the replaced range UOp (the one that keeps the old axis type), or NULL on failure. */
/* Core shift_to: substitute only + refresh. Matches tinygrad's shift_to semantics.
 * Does NOT run graph_rewrite -- callers that need simplification do it themselves. */
static PolyUOp *sched_shift_to_core(OptScheduler *s, PolyUOp *rng, int64_t amount,
                                     PolyAxisType new_type, bool top,
                                     PolyUOp *input_new_rng, PolyUOp **out_new_rng) {
  int64_t bound = 0;
  if (rng->n_src > 0 && rng->src[0]->op == POLY_OP_CONST && rng->src[0]->arg.kind == POLY_ARG_INT)
    bound = rng->src[0]->arg.i;
  if (bound <= 0 || bound % amount != 0) return NULL;
  int64_t old_sz = bound / amount;

  PolyCtx *ctx = s->ctx;
  PolyDType dt = rng->dtype;

  /* Create or use provided new range */
  PolyUOp *new_rng;
  if (input_new_rng) {
    new_rng = input_new_rng;
  } else {
    PolyUOp *new_sz = poly_uop0(ctx, POLY_OP_CONST, dt, poly_arg_int(amount));
    new_rng = poly_uop1(ctx, POLY_OP_RANGE, dt, new_sz,
                         poly_arg_range(s->opt_range_next++, new_type));
  }

  /* Create complementary range with reduced bound */
  PolyUOp *rep_sz = poly_uop0(ctx, POLY_OP_CONST, dt, poly_arg_int(old_sz));
  PolyUOp *replaced = poly_uop1(ctx, POLY_OP_RANGE, dt, rep_sz, rng->arg);

  /* Compute substitution expression */
  PolyUOp *sub_axis;
  if (top) {
    PolyUOp *c = poly_uop0(ctx, POLY_OP_CONST, dt, poly_arg_int(old_sz));
    sub_axis = poly_uop2(ctx, POLY_OP_ADD, dt,
                         poly_uop2(ctx, POLY_OP_MUL, dt, new_rng, c, poly_arg_none()),
                         replaced, poly_arg_none());
  } else {
    PolyUOp *c = poly_uop0(ctx, POLY_OP_CONST, dt, poly_arg_int(amount));
    sub_axis = poly_uop2(ctx, POLY_OP_ADD, dt,
                         poly_uop2(ctx, POLY_OP_MUL, dt, replaced, c, poly_arg_none()),
                         new_rng, poly_arg_none());
  }

  PolyUOp *from[1] = { rng };
  PolyUOp *to[1] = { sub_axis };
  s->ast = poly_uop_substitute(ctx, s->ast, from, to, 1);
  sched_refresh(s);
  if (out_new_rng) *out_new_rng = new_rng;
  return replaced;
}

/* Convenience: shift_to + symbolic simplification + flatten.
 * Used by the CPU heuristic path where callers expect simplified state after each shift. */
static PolyUOp *sched_shift_to_ex(OptScheduler *s, PolyUOp *rng, int64_t amount,
                                   PolyAxisType new_type, bool top,
                                   PolyUOp *input_new_rng, PolyUOp **out_new_rng) {
  PolyUOp *result = sched_shift_to_core(s, rng, amount, new_type, top, input_new_rng, out_new_rng);
  if (result) {
    s->ast = poly_graph_rewrite(s->ctx, s->ast, poly_symbolic_simple());
    s->ast = poly_graph_rewrite(s->ctx, s->ast, poly_pm_flatten_range());
    sched_refresh(s);
  }
  return result;
}

static PolyUOp *sched_shift_to(OptScheduler *s, PolyUOp *rng, int64_t amount,
                                PolyAxisType new_type, bool top) {
  return sched_shift_to_ex(s, rng, amount, new_type, top, NULL, NULL);
}

/* Helper: get indices of upcastable dims (GLOBAL/LOCAL/LOOP with size > 1) */
static int sched_upcastable_dims(const OptScheduler *s, int *out, int max_n) {
  int n = 0;
  for (int i = 0; i < s->n_rngs && n < max_n; i++) {
    PolyAxisType t = s->types[i];
    if ((t == POLY_AXIS_GLOBAL || t == POLY_AXIS_LOCAL || t == POLY_AXIS_LOOP) && s->shape[i] > 1)
      out[n++] = i;
  }
  return n;
}

/* Helper: get indices of unrollable dims (GROUP_REDUCE/REDUCE with size > 1) */
static int sched_unrollable_dims(const OptScheduler *s, int *out, int max_n) {
  int n = 0;
  for (int i = 0; i < s->n_rngs && n < max_n; i++) {
    PolyAxisType t = s->types[i];
    if ((t == POLY_AXIS_GROUP_REDUCE || t == POLY_AXIS_REDUCE) && s->shape[i] > 1)
      out[n++] = i;
  }
  return n;
}

/* Helper: product of UPCAST/UNROLL dim sizes */
static int64_t sched_upcast_size(const OptScheduler *s) {
  int64_t prod = 1;
  for (int i = 0; i < s->n_rngs; i++) {
    if (s->types[i] == POLY_AXIS_UPCAST || s->types[i] == POLY_AXIS_UNROLL)
      prod *= s->shape[i];
  }
  return prod;
}

/* Helper: count of UPCAST/UNROLL axes */
static int sched_upcasted(const OptScheduler *s) {
  int n = 0;
  for (int i = 0; i < s->n_rngs; i++) {
    if (s->types[i] == POLY_AXIS_UPCAST || s->types[i] == POLY_AXIS_UNROLL) n++;
  }
  return n;
}

/* Helper: product of output shape (non-reduce dims) at upcastable indices */
static int64_t sched_output_prod_upcastable(const OptScheduler *s) {
  int up_dims[SCHED_MAX_RNGS];
  int n_up = sched_upcastable_dims(s, up_dims, SCHED_MAX_RNGS);
  int64_t prod = 1;
  for (int i = 0; i < n_up; i++) prod *= s->shape[up_dims[i]];
  return prod;
}

/* Flatten ADD tree into leaf addends (port of tinygrad split_uop(ADD)) */
static int split_uop_add(PolyUOp *u, PolyUOp **out, int max_n) {
  if (u->op == POLY_OP_ADD) {
    int n = 0;
    for (int i = 0; i < u->n_src && n < max_n; i++)
      n += split_uop_add(u->src[i], out + n, max_n - n);
    return n;
  }
  if (max_n > 0) { out[0] = u; return 1; }
  return 0;
}

/* ── TensorCore helpers (port of tc.py) ──────────────────────────────── */

/* tc.py:33 -- get_reduce_axes: returns [(0,2), (1,2), ...] for K dimension */
int tc_get_reduce_axes(const PolyTensorCore *tc, int out[][2]) {
  int k = tc->dims[2], n = 0;
  while (k > 1) { out[n][0] = n; out[n][1] = 2; n++; k /= 2; }
  return n;
}

/* tc.py:34-35 -- count local/upcast opts */
int tc_count_local(const PolyTensorCore *tc) {
  int n = 0;
  for (int i = 0; i < tc->n_opts; i++) if (tc->opts[i].type == 'l') n++;
  return n;
}
int tc_count_upcast(const PolyTensorCore *tc) {
  int n = 0;
  for (int i = 0; i < tc->n_opts; i++) if (tc->opts[i].type == 'u') n++;
  return n;
}

/* tc.py:25-32 -- base_shape_str: build axis name list from opts + reduce axes.
 * Returns count of entries written to out[]. Each entry is a string like "l0","u1","r3". */
int tc_base_shape_str(const PolyTensorCore *tc, const char *out[], int max_n) {
  int n = 0, cnt_l = 0, cnt_u = 0;
  for (int i = 0; i < tc->n_opts && n < max_n; i++) {
    static const char *l_names[] = {"l0","l1","l2","l3","l4","l5","l6","l7"};
    static const char *u_names[] = {"u0","u1","u2","u3","u4","u5","u6","u7"};
    if (tc->opts[i].type == 'l') out[n++] = l_names[cnt_l++];
    else                          out[n++] = u_names[cnt_u++];
  }
  /* Append reduce axes */
  int ra[16][2];
  int n_ra = tc_get_reduce_axes(tc, ra);
  static const char *r_names[] = {"r0","r1","r2","r3","r4","r5","r6","r7"};
  for (int i = 0; i < n_ra && n < max_n; i++) out[n++] = r_names[i];
  return n;
}

/* tc.py:36-38 -- base_upcast_axes: reversed list of reduce + upcast axis names */
int tc_base_upcast_axes(const PolyTensorCore *tc, const char *out[], int max_n) {
  int n_upcast = tc_count_upcast(tc);
  int ra[16][2];
  int n_ra = tc_get_reduce_axes(tc, ra);
  /* Build forward: [r0..rN, u0..uM] then reverse */
  const char *fwd[32];
  int n_fwd = 0;
  static const char *r_names[] = {"r0","r1","r2","r3","r4","r5","r6","r7"};
  static const char *u_names[] = {"u0","u1","u2","u3","u4","u5","u6","u7"};
  for (int i = 0; i < n_ra && n_fwd < 32; i++) fwd[n_fwd++] = r_names[i];
  for (int i = 0; i < n_upcast && n_fwd < 32; i++) fwd[n_fwd++] = u_names[i];
  int n = 0;
  for (int i = n_fwd - 1; i >= 0 && n < max_n; i--) out[n++] = fwd[i];
  return n;
}

/* tc.py:17-20 -- _remaps: build two remap dicts from swizzle.
 * fwd_st = base_shape_str, remap[i] maps fwd_st[j] -> swizzle[i] flattened.
 * Returns remap as parallel arrays: remap_from[k], remap_to[k] for k in 0..n-1. */
int tc_build_remap(const PolyTensorCore *tc, int swz_idx,
                          const char *remap_from[], const char *remap_to[], int max_n) {
  const char *fwd[32];
  int n_fwd = tc_base_shape_str(tc, fwd, 32);
  /* Flatten swizzle[swz_idx]: [local_axes] + [upcast_axes] + [reduce_axes] */
  const char *flat[32];
  int n_flat = 0;
  for (int g = 0; g < 3; g++)
    for (int j = 0; j < tc->swizzle_len[swz_idx][g] && n_flat < 32; j++)
      flat[n_flat++] = tc->swizzle[swz_idx][g][j];
  int n = (n_fwd < n_flat) ? n_fwd : n_flat;
  if (n > max_n) n = max_n;
  for (int i = 0; i < n; i++) { remap_from[i] = fwd[i]; remap_to[i] = flat[i]; }
  return n;
}

/* tc.py:21-23 -- permutes_for_shape_str: given shape_str, apply remap and return permutation.
 * shape_str[i] is an axis name. Output perm[i] = shape_str.index(remap[shape_str[i]]).
 * If shape_str[i] is not in remap, perm[i] = i. */
void tc_permute_for_shape_str(const PolyTensorCore *tc, int swz_idx,
                                      const char *shape_str[], int n_shape,
                                      int perm[], int max_n) {
  const char *rf[32], *rt[32];
  int n_remap = tc_build_remap(tc, swz_idx, rf, rt, 32);
  for (int i = 0; i < n_shape && i < max_n; i++) {
    /* Find shape_str[i] in remap_from -> get remap_to */
    const char *mapped = NULL;
    for (int r = 0; r < n_remap; r++) {
      if (strcmp(shape_str[i], rf[r]) == 0) { mapped = rt[r]; break; }
    }
    if (!mapped) { perm[i] = i; continue; }
    /* Find mapped in shape_str -> get index */
    perm[i] = i; /* default if not found */
    for (int j = 0; j < n_shape; j++) {
      if (strcmp(shape_str[j], mapped) == 0) { perm[i] = j; break; }
    }
  }
}

/* ── Axis letter mapping (ops.py:20-21) ──────────────────────────────── */
static const char *axis_letter(PolyAxisType t) {
  switch (t) {
    case POLY_AXIS_GLOBAL:       return "g";
    case POLY_AXIS_LOCAL:        return "l";
    case POLY_AXIS_WARP:         return "w";
    case POLY_AXIS_UPCAST:       return "u";
    case POLY_AXIS_GROUP_REDUCE: return "G";
    case POLY_AXIS_REDUCE:       return "R";
    case POLY_AXIS_UNROLL:       return "r";
    case POLY_AXIS_LOOP:         return "L";
    default:                     return "?";
  }
}

/* postrange.py:36-42 -- shape_str: build axis name array from scheduler state */
static int sched_shape_str(const OptScheduler *s, const char *out[], int max_n) {
  int n = 0;
  int cnt[16] = {0}; /* count per axis type */
  static char buf[64][8]; /* static buffer for generated strings */
  for (int i = 0; i < s->n_rngs && n < max_n && n < 64; i++) {
    int t = (int)s->types[i];
    snprintf(buf[n], 8, "%s%d", axis_letter(s->types[i]), cnt[t]++);
    out[n] = buf[n];
    n++;
  }
  return n;
}

/* Collect RANGE ops reachable from a UOp via backward walk.
 * Returns bitmask: bit j set if s->rngs[j] is reachable from u. */
static uint64_t collect_ranges_from(const OptScheduler *s, PolyUOp *u) {
  /* Simple DFS -- limited depth for scheduler-level graphs */
  uint64_t mask = 0;
  PolyUOp *stack[512];
  int sp = 0;
  stack[sp++] = u;
  /* Visited set using a simple pointer set (arena-allocated UOps have unique addresses) */
  PolyMap *visited = poly_map_new(256);
  while (sp > 0) {
    PolyUOp *cur = stack[--sp];
    if (poly_map_get(visited, ptr_hash(cur), cur, ptr_eq)) continue;
    poly_map_set(visited, ptr_hash(cur), cur, cur, ptr_eq);
    if (cur->op == POLY_OP_RANGE) {
      for (int j = 0; j < s->n_rngs; j++) {
        if (s->rngs[j] == cur) { mask |= (1ULL << j); break; }
      }
    }
    /* Don't recurse past RANGE (tinygrad: ended_ranges) */
    int rs = range_start_for_op(cur->op);
    int end = (rs >= 0) ? rs : cur->n_src;
    for (int i = 0; i < end && sp < 510; i++)
      stack[sp++] = cur->src[i];
  }
  poly_map_destroy(visited);
  return mask;
}

/* Forward declarations */
static PolyArg poly_arg_pair_tuple(int64_t (*pairs)[2], int n);
static void sched_copy(OptScheduler *dst, const OptScheduler *src);

/* ── sched_apply_tc_opt (port of postrange.py:221-314) ──────────────── */
#define TC_TAG 0x5443 /* 'TC' */

/* Sort helper: sort UOp* array by axis_id descending */
static int cmp_axis_id_desc(const void *a, const void *b) {
  int64_t ia = poly_range_axis_id((*(const PolyUOp *const *)a)->arg);
  int64_t ib = poly_range_axis_id((*(const PolyUOp *const *)b)->arg);
  return (ib > ia) - (ib < ia);
}

/* Returns true on success. On success, tc_axes_out[0..2] are the replaced N,M,K ranges.
 * use_tc: 1 = full WMMA construction, 2 = shape only (no WMMA UOps).
 * tc_select: -1 = try all, >=0 = specific TC index.
 * tc_opt: 0 = strict (single reduce axis, direct load->mul),
 *         1 = allow CAST'd buffers + multiple reduce axes.
 *         (2 = reserved for future PADTO; currently same as 1) */
static bool sched_apply_tc_opt(OptScheduler *s, int axis, int tc_select, int tc_opt,
                                int use_tc, const PolyTensorCore *tcs, int n_tcs,
                                PolyUOp *tc_axes_out[3]) {
  PolyCtx *ctx = s->ctx;

  /* 1. Find REDUCE(ADD) and its MUL (postrange.py:222-227) */
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, s->ast, &n_topo);
  PolyUOp *reduceop = NULL;
  for (int i = 0; i < n_topo; i++) {
    if (topo[i]->op == POLY_OP_REDUCE &&
        topo[i]->arg.kind == POLY_ARG_OPS && topo[i]->arg.ops == POLY_OP_ADD) {
      reduceop = topo[i];
      break;
    }
  }
  if (!reduceop || !use_tc) return false;

  PolyUOp *mul = reduceop->src[0];
  /* tc_opt >= 1 allows CAST'd buffers (postrange.py:225) */
  if (mul->op == POLY_OP_CAST && mul->n_src > 0) {
    if (tc_opt < 1) {
      return false;
    }
    mul = mul->src[0];
  }
  if (mul->op != POLY_OP_MUL || mul->n_src < 2) {
    return false;
  }

  PolyUOp *in0 = mul->src[0], *in1 = mul->src[1];

  /* 2. Try each TC spec (postrange.py:232-248) */
  const PolyTensorCore *tc_list = tcs;
  int tc_count = n_tcs;
  if (tc_select >= 0 && tc_select < n_tcs) { tc_list = &tcs[tc_select]; tc_count = 1; }
  else if (tc_select >= n_tcs) return false;

  for (int tci = 0; tci < tc_count; tci++) {
    const PolyTensorCore *tc = &tc_list[tci];

    /* Check dtype match */
    PolyDType in0_scalar = poly_dtype_scalar(in0->dtype);
    PolyDType in1_scalar = poly_dtype_scalar(in1->dtype);
    PolyDType red_scalar = poly_dtype_scalar(reduceop->dtype);
    if (!poly_dtype_eq(tc->dtype_in, in0_scalar) || !poly_dtype_eq(tc->dtype_in, in1_scalar)) continue;
    if (!poly_dtype_eq(tc->dtype_out, red_scalar)) continue;

    /* 3. Classify ranges (postrange.py:236-238) */
    uint64_t in0_reach = collect_ranges_from(s, in0);
    uint64_t in1_reach = collect_ranges_from(s, in1);

    PolyUOp *in0_ranges[SCHED_MAX_RNGS], *in1_ranges[SCHED_MAX_RNGS];
    int n_in0 = 0, n_in1 = 0;
    for (int i = 0; i < s->n_rngs; i++) {
      uint64_t bit = 1ULL << i;
      if ((in0_reach & bit) && !(in1_reach & bit) && n_in0 < SCHED_MAX_RNGS)
        in0_ranges[n_in0++] = s->rngs[i];
      if ((in1_reach & bit) && !(in0_reach & bit) && n_in1 < SCHED_MAX_RNGS)
        in1_ranges[n_in1++] = s->rngs[i];
    }

    /* red_ranges from REDUCE's trailing RANGE sources */
    PolyUOp *red_ranges[SCHED_MAX_RNGS];
    int n_red = 0;
    int rs = range_start_for_op(POLY_OP_REDUCE);
    for (int i = rs; i < reduceop->n_src && n_red < SCHED_MAX_RNGS; i++) {
      if (reduceop->src[i]->op == POLY_OP_RANGE)
        red_ranges[n_red++] = reduceop->src[i];
    }

    /* Sort all three by axis_id descending (postrange.py:236-238) */
    if (n_in0 > 1) qsort(in0_ranges, (size_t)n_in0, sizeof(PolyUOp *), cmp_axis_id_desc);
    if (n_in1 > 1) qsort(in1_ranges, (size_t)n_in1, sizeof(PolyUOp *), cmp_axis_id_desc);
    if (n_red > 1) qsort(red_ranges, (size_t)n_red, sizeof(PolyUOp *), cmp_axis_id_desc);

    if (n_in0 == 0 || n_in1 == 0 || n_red == 0) continue;

    /* tc_opt == 0: strict mode requires exactly one reduce axis (heuristic.py:28) */
    if (tc_opt == 0 && n_red > 1) continue;

    /* 4. Axis choices: product(in1_ranges, in0_ranges, red_ranges) -- note swap */
    int n_choices = n_in1 * n_in0 * n_red;
    if (axis >= n_choices) continue;
    int red_idx = axis % n_red;
    int in0_idx = (axis / n_red) % n_in0;
    int in1_idx = (axis / n_red / n_in0) % n_in1;

    PolyUOp *axes[3] = { in1_ranges[in1_idx], in0_ranges[in0_idx], red_ranges[red_idx] };

    /* 5. Tag reduceop via tagged clone + substitute (matches tinygrad's
     * self.ast.substitute({reduceop: reduceop.replace(tag="TC")})).
     * poly_uop_tagged includes tag in CSE key, creating a distinct node. */
    PolyUOp *tagged_red = poly_uop_tagged(ctx, reduceop->op, reduceop->dtype,
                                           reduceop->src, reduceop->n_src, reduceop->arg, TC_TAG);
    {
      PolyUOp *from_tag[1] = { reduceop };
      PolyUOp *to_tag[1] = { tagged_red };
      s->ast = poly_uop_substitute(ctx, s->ast, from_tag, to_tag, 1);
      sched_refresh(s);
    }

    /* 6. Check divisibility -- reject non-const bounds (postrange.py:254-262) */
    bool pad_ok = true;
    for (int i = 0; i < 3; i++) {
      if (axes[i]->n_src == 0 || axes[i]->src[0]->op != POLY_OP_CONST ||
          axes[i]->src[0]->arg.kind != POLY_ARG_INT) {
        pad_ok = false; break; /* non-const bound: hard reject */
      }
      int64_t sz = axes[i]->src[0]->arg.i;
      if (sz <= 0 || sz % tc->dims[i] != 0) {
        if (tc_opt < 2) { pad_ok = false; break; }
        /* TODO: PADTO support */
        pad_ok = false; break;
      }
    }
    if (!pad_ok) {
      continue;
    }
    /* Verify tag survived the substitute+refresh */

    /* 7. Create WARP range and apply opts (postrange.py:264-274) */
    PolyUOp *warp_sz = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(tc->threads));
    PolyUOp *warp = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, warp_sz,
                               poly_arg_range(-1, POLY_AXIS_WARP));

    PolyUOp *ne[32];
    int n_ne = 0;

    for (int oi = 0; oi < tc->n_opts; oi++) {
      char otype = tc->opts[oi].type;
      int odim = tc->opts[oi].dim;
      PolyUOp *new_rng = NULL;

      if (otype == 'l') {
        PolyUOp *two = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(2));
        PolyUOp *warp_mod2 = poly_uop2(ctx, POLY_OP_MOD, POLY_INT32, warp, two, poly_arg_none());
        axes[odim] = sched_shift_to_core(s, axes[odim], 2, POLY_AXIS_LOCAL, false, warp_mod2, &new_rng);
        warp = poly_uop2(ctx, POLY_OP_IDIV, POLY_INT32, warp, two, poly_arg_none());
      } else if (otype == 'u') {
        axes[odim] = sched_shift_to_core(s, axes[odim], 2, POLY_AXIS_UPCAST, false, NULL, &new_rng);
      }
      if (!axes[odim]) {
        return false;
      }
      if (new_rng) ne[n_ne++] = new_rng;
    }

    /* 8. Apply reduce axes (postrange.py:276-278) */
    int ra[16][2];
    int n_ra = tc_get_reduce_axes(tc, ra);
    for (int i = 0; i < n_ra; i++) {
      PolyUOp *new_rng = NULL;
      axes[2] = sched_shift_to_core(s, axes[2], ra[i][1], POLY_AXIS_UNROLL, false, NULL, &new_rng);
      if (!axes[2]) {
        return false;
      }
      if (new_rng) ne[n_ne++] = new_rng;
    }

    /* 9. Build WMMA UOps if use_tc == 1 (postrange.py:280-313) */
    if (use_tc == 1) {
      /* Re-find the tagged reduceop */
      int n_topo2 = 0;
      PolyUOp **topo2 = poly_toposort(ctx, s->ast, &n_topo2);

      /* Debug: check how many ne[] pointers are present in the current AST */
      PolyUOp *found_red = NULL;
      for (int i = 0; i < n_topo2; i++) {
        if (topo2[i]->op == POLY_OP_REDUCE && topo2[i]->tag == TC_TAG) {
          found_red = topo2[i]; break;
        }
      }
      if (!found_red) return false;

      /* Create tagged copies of ne[] (postrange.py:283: tne = [x.replace(tag=1) for x in ne]).
       * Tags ALL elements (RANGE, ALU expressions like warp%2, etc.) -- not just RANGEs. */
      PolyUOp *tne[32];
      for (int i = 0; i < n_ne; i++) {
        tne[i] = poly_uop_tagged(ctx, ne[i]->op, ne[i]->dtype,
                                  ne[i]->src, ne[i]->n_src, ne[i]->arg, 1);
      }

      /* Substitute ne -> tne in found_red to isolate the MUL operands (postrange.py:284-285) */
      PolyUOp *ret = poly_uop_substitute(ctx, found_red, ne, tne, n_ne);
      PolyUOp *ret_mul = ret->src[0];
      if (ret_mul->op == POLY_OP_CAST && ret_mul->n_src > 0) ret_mul = ret_mul->src[0];
      PolyUOp *srcs[2] = { ret_mul->src[0], ret_mul->src[1] };

      /* Apply swizzle permutations (postrange.py:286):
       * srcs[k] = x.substitute(dict(zip(tne, [ne[i] for i in argsort(p)])))
       * where p = tc.permutes_for_shape_str(tc.base_shape_str()) */
      const char *bss[32];
      int n_bss = tc_base_shape_str(tc, bss, 32);

      int perm0[32], perm1[32];
      tc_permute_for_shape_str(tc, 0, bss, n_bss, perm0, 32);
      tc_permute_for_shape_str(tc, 1, bss, n_bss, perm1, 32);

      /* argsort(perm): inverse permutation. argsort[j] = i where perm[i] = j */
      int argsort0[32], argsort1[32];
      for (int i = 0; i < n_bss; i++) { argsort0[i] = i; argsort1[i] = i; }
      for (int i = 0; i < n_bss; i++) {
        if (perm0[i] < n_bss) argsort0[perm0[i]] = i;
        if (perm1[i] < n_bss) argsort1[perm1[i]] = i;
      }

      /* Build reordered ne[] for each source (postrange.py:286):
       * ne_reordered[i] = ne[argsort[i]] -- permute the ne list itself */
      PolyUOp *ne_reordered0[32], *ne_reordered1[32];
      for (int i = 0; i < n_ne; i++) {
        int idx0 = (i < n_bss) ? argsort0[i] : i;
        int idx1 = (i < n_bss) ? argsort1[i] : i;
        ne_reordered0[i] = (idx0 >= 0 && idx0 < n_ne) ? ne[idx0] : ne[i];
        ne_reordered1[i] = (idx1 >= 0 && idx1 < n_ne) ? ne[idx1] : ne[i];
      }
      srcs[0] = poly_uop_substitute(ctx, srcs[0], tne, ne_reordered0, n_ne);
      srcs[1] = poly_uop_substitute(ctx, srcs[1], tne, ne_reordered1, n_ne);

      /* Compute tc_reduce_axes and tc_upcast_axes (postrange.py:289-295) */
      const char *shape_str[64];
      int n_ss = sched_shape_str(s, shape_str, 64);

      const char *bua[32];
      int n_bua = tc_base_upcast_axes(tc, bua, 32);

      /* tc_reduce_axes: axis ids for "r0","r1",... in scheduler shape_str */
      int tc_reduce_axis_ids[16];
      int n_tc_ra = 0;
      for (int ri = 0; ri < n_ra; ri++) {
        char rname[8];
        snprintf(rname, 8, "r%d", ri);
        for (int si = 0; si < n_ss; si++) {
          if (strcmp(shape_str[si], rname) == 0) {
            tc_reduce_axis_ids[n_tc_ra++] = (int)poly_range_axis_id(s->rngs[si]->arg);
            break;
          }
        }
      }

      /* tc_upcast_axes[dim]: first log2(ept[dim]) entries of base_upcast_axes, mapped to axis ids */
      int64_t upcast_pairs[3][16][2];
      int n_upcast[3] = {0, 0, 0};
      for (int dim = 0; dim < 3; dim++) {
        int need = 0;
        { int v = tc->elements_per_thread[dim]; while (v > 1) { need++; v /= 2; } }
        for (int ui = 0; ui < need && ui < n_bua; ui++) {
          for (int si = 0; si < n_ss; si++) {
            if (strcmp(shape_str[si], bua[ui]) == 0) {
              upcast_pairs[dim][n_upcast[dim]][0] = poly_range_axis_id(s->rngs[si]->arg);
              upcast_pairs[dim][n_upcast[dim]][1] = 2;
              n_upcast[dim]++;
              break;
            }
          }
        }
      }

      /* Build CONTRACT + WMMA + UNROLL (postrange.py:301-306) */
      PolyDType vec_in0 = poly_dtype_vec(tc->dtype_in, tc->elements_per_thread[0]);
      PolyDType vec_in1 = poly_dtype_vec(tc->dtype_in, tc->elements_per_thread[1]);
      PolyDType vec_out = poly_dtype_vec(tc->dtype_out, tc->elements_per_thread[2]);

      PolyUOp *ca_src[1] = { srcs[0] };
      PolyUOp *contract_a = poly_uop_tagged(ctx, POLY_OP_CONTRACT, vec_in0, ca_src, 1,
                                              poly_arg_pair_tuple(upcast_pairs[0], n_upcast[0]), 1);
      PolyUOp *cb_src[1] = { srcs[1] };
      PolyUOp *contract_b = poly_uop_tagged(ctx, POLY_OP_CONTRACT, vec_in1, cb_src, 1,
                                              poly_arg_pair_tuple(upcast_pairs[1], n_upcast[1]), 1);

      /* Zero accumulator */
      PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, tc->dtype_out, poly_arg_float(0.0));
      PolyUOp *zero_elems[16];
      for (int i = 0; i < tc->elements_per_thread[2] && i < 16; i++) zero_elems[i] = zero;
      PolyUOp *zero_vec = poly_uop(ctx, POLY_OP_VECTORIZE, vec_out, zero_elems,
                                     tc->elements_per_thread[2], poly_arg_none());

      PolyUOp *wmma_srcs[3] = { contract_a, contract_b, zero_vec };
      PolyUOp *wmma = poly_uop_tagged(ctx, POLY_OP_WMMA, vec_out, wmma_srcs, 3,
                                        poly_arg_str(tc->intrinsic_name), 1);

      PolyUOp *unroll_src[1] = { wmma };
      PolyUOp *tc_uop = poly_uop_tagged(ctx, POLY_OP_UNROLL, tc->dtype_out, unroll_src, 1,
                                          poly_arg_pair_tuple(upcast_pairs[2], n_upcast[2]), 1);

      /* Preserve extra reduce ranges not consumed by TC (postrange.py:309-310) */
      int rs2 = range_start_for_op(POLY_OP_REDUCE);
      PolyUOp *extra_rngs[16];
      int n_extra = 0;
      for (int i = rs2; i < found_red->n_src && n_extra < 16; i++) {
        if (found_red->src[i]->op != POLY_OP_RANGE) continue;
        int64_t aid = poly_range_axis_id(found_red->src[i]->arg);
        bool in_tc = false;
        for (int r = 0; r < n_tc_ra; r++)
          if (tc_reduce_axis_ids[r] == (int)aid) { in_tc = true; break; }
        if (!in_tc) extra_rngs[n_extra++] = found_red->src[i];
      }
      if (n_extra > 0) {
        PolyUOp *red_srcs[18];
        red_srcs[0] = tc_uop;
        for (int i = 0; i < n_extra; i++) red_srcs[i + 1] = extra_rngs[i];
        PolyArg red_arg = { .kind = POLY_ARG_OPS, .ops = POLY_OP_ADD };
        tc_uop = poly_uop(ctx, POLY_OP_REDUCE, tc_uop->dtype, red_srcs, n_extra + 1, red_arg);
      }

      /* Substitute found_red -> tc_uop in AST */
      PolyUOp *from_r[1] = { found_red };
      PolyUOp *to_r[1] = { tc_uop };
      s->ast = poly_uop_substitute(ctx, s->ast, from_r, to_r, 1);
    }

    if (tc_axes_out) {
      tc_axes_out[0] = axes[0];
      tc_axes_out[1] = axes[1];
      tc_axes_out[2] = axes[2];
    }
    sched_refresh(s);
    return true;
  }

  return false;
}

/* ── hand_coded_optimizations (heuristic.py:8-190, CPU-relevant subset) ── */
static PolyUOp *poly_apply_opts_heuristic(PolyCtx *ctx, PolyUOp *sink, PolyRendererCaps caps) {
  /* tinygrad apply_opts guard (postrange.py:352): skip heuristic for multi-block kernels. */
  {
    int n_topo = 0;
    PolyUOp **topo = poly_toposort(ctx, sink, &n_topo);
    for (int i = 0; i < n_topo; i++) {
      if (topo[i]->op == POLY_OP_BUFFERIZE) return sink;
    }
  }

  OptScheduler s;
  sched_init(&s, ctx, sink);
  if (s.n_rngs == 0) return sink;

  /* == Tensor core optimization (heuristic.py:28-46) ==
   * Try TC before other heuristics. On success, return immediately. */
  if (caps.n_tensor_cores > 0) {
    /* Count reduce axes */
    int n_reduce = 0;
    for (int i = 0; i < s.n_rngs; i++) {
      if (s.types[i] == POLY_AXIS_GROUP_REDUCE || s.types[i] == POLY_AXIS_REDUCE)
        n_reduce++;
    }
    int tc_opt_env = 0;
    { const char *e = getenv("POLY_TC_OPT"); if (e) tc_opt_env = atoi(e); }
    int use_tc_env = 1;
    { const char *e = getenv("POLY_USE_TC"); if (e) use_tc_env = atoi(e); }

    if (use_tc_env > 0 && (n_reduce == 1 || tc_opt_env >= 1)) {
      OptScheduler tk;
      sched_copy(&tk, &s);
      PolyUOp *tc_axes[3];
      bool tc_ok = sched_apply_tc_opt(&tk, 0, -1, tc_opt_env, use_tc_env,
                                       caps.tensor_cores, caps.n_tensor_cores, tc_axes);
      if (tc_ok) {
        /* Post-TC upcasts on M and N (heuristic.py:39-45) */
        for (int tc_dim = 1; tc_dim >= 0; tc_dim--) {
          int64_t bound = 0;
          if (tc_axes[tc_dim] && tc_axes[tc_dim]->n_src > 0 &&
              tc_axes[tc_dim]->src[0]->op == POLY_OP_CONST)
            bound = tc_axes[tc_dim]->src[0]->arg.i;
          if (bound <= 1) continue;
          int szs[] = {5, 4, 3, 2};
          for (int si = 0; si < 4; si++) {
            if (bound % szs[si] == 0) {
              int idx = -1;
              for (int ri = 0; ri < tk.n_rngs; ri++) {
                if (tk.rngs[ri] == tc_axes[tc_dim]) { idx = ri; break; }
              }
              if (idx >= 0)
                tc_axes[tc_dim] = sched_shift_to(&tk, tk.rngs[idx], szs[si], POLY_AXIS_UPCAST, false);
              break;
            }
          }
        }
        return tk.ast;
      }
    }
  }

  /* == Masked upcast (heuristic.py:96-105) ==
   * Upcast small dims (<=7) that appear in WHERE gates */
  {
    int up_dims[SCHED_MAX_RNGS];
    int n_up = sched_upcastable_dims(&s, up_dims, SCHED_MAX_RNGS);
    int to_upcast[SCHED_MAX_RNGS];
    int n_to_upcast = 0;

    /* Use the precomputed reachability bitmask to check WHERE gates.
     * Build a full reach array for this purpose (freed after masked upcast). */
    int n_topo = 0;
    PolyUOp **topo = poly_toposort(ctx, s.ast, &n_topo);
    uint64_t *full_reach = build_reachability_bitmask(topo, n_topo, s.rngs, s.n_rngs);

    /* Build ptr→index map for WHERE src[0] lookup */
    PolyMap *topo_idx = poly_map_new((size_t)(n_topo < 64 ? 64 : (size_t)n_topo * 2));
    int *tidx = (int *)malloc((size_t)n_topo * sizeof(int));
    for (int i = 0; i < n_topo; i++) {
      tidx[i] = i;
      poly_map_set(topo_idx, ptr_hash(topo[i]), topo[i], &tidx[i], ptr_eq);
    }

    for (int ui = 0; ui < n_up; ui++) {
      int axis = up_dims[ui];
      if (s.shape[axis] > 7) continue;
      uint64_t rng_bit = 1ULL << axis;
      bool is_masked = false;
      for (int ti = 0; ti < n_topo && !is_masked; ti++) {
        if (topo[ti]->op != POLY_OP_WHERE) continue;
        /* Check if rng is reachable from WHERE's src[0] (the condition) */
        int *pidx = (int *)poly_map_get(topo_idx, ptr_hash(topo[ti]->src[0]),
                                         topo[ti]->src[0], ptr_eq);
        if (pidx && (full_reach[*pidx] & rng_bit)) is_masked = true;
      }
      if (!is_masked) continue;
      /* Check total upcast product stays <= 49 (7*7) */
      int64_t prod = s.shape[axis];
      for (int j = 0; j < n_to_upcast; j++) prod *= s.shape[to_upcast[j]];
      if (prod <= 49 && n_to_upcast < SCHED_MAX_RNGS)
        to_upcast[n_to_upcast++] = axis;
    }
    poly_map_destroy(topo_idx);
    free(tidx);
    free(full_reach);

    /* Apply in reverse order (matching tinygrad) */
    for (int i = n_to_upcast - 1; i >= 0; i--) {
      int axis = to_upcast[i];
      if (axis < s.n_rngs && s.shape[axis] > 1)
        sched_shift_to(&s, s.rngs[axis], s.shape[axis], POLY_AXIS_UPCAST, false);
    }
  }

  /* == Multi-axis UPCAST with stride scoring (heuristic.py:107-133) == */
  {
    bool upcasted_axis[SCHED_MAX_RNGS] = {false};

    while (sched_output_prod_upcastable(&s) >= 1024 && sched_upcast_size(&s) < 32) {
      int up_dims[SCHED_MAX_RNGS];
      int n_up = sched_upcastable_dims(&s, up_dims, SCHED_MAX_RNGS);

      /* Score each candidate (num_strides, sum_strides, axis, amount) */
      typedef struct { int num_strides; int64_t sum_strides; int axis; int amount; } UpChoice;
      UpChoice choices[SCHED_MAX_RNGS * 2];
      int n_choices = 0;

      int amounts[] = {3, 4};
      for (int ui = 0; ui < n_up; ui++) {
        int axis = up_dims[ui];
        if (upcasted_axis[axis]) continue;

        for (int ai = 0; ai < 2; ai++) {
          int amount = amounts[ai];
          if (s.shape[axis] % amount != 0) continue;

          PolyUOp *rng = s.rngs[axis];

          /* Expanded axis check (heuristic.py:117-118):
           * Must have a buffer where rng is NOT in index but all UPCAST/UNROLL rngs ARE */
          bool has_expanded_buf = false;
          if (s.has_reach) {
            /* Build mask of all current UPCAST/UNROLL ranges */
            uint64_t upcast_mask = 0;
            for (int ri = 0; ri < s.n_rngs; ri++) {
              if (s.types[ri] == POLY_AXIS_UPCAST || s.types[ri] == POLY_AXIS_UNROLL)
                upcast_mask |= (1ULL << ri);
            }
            for (int bi = 0; bi < s.n_bufs && !has_expanded_buf; bi++) {
              if (s.buf_reach[bi] & (1ULL << axis)) continue; /* rng IS in this buf's index */
              /* Check all existing UPCAST/UNROLL ranges are in this buf's index */
              if ((s.buf_reach[bi] & upcast_mask) == upcast_mask)
                has_expanded_buf = true;
            }
          }
          if (!has_expanded_buf) continue;

          /* Count strides (heuristic.py:119-127) */
          int num_strides = 0;
          int64_t sum_strides = 0;
          for (int bi = 0; bi < s.n_bufs; bi++) {
            PolyUOp *idx_uop = s.bufs[bi];
            if (idx_uop->n_src < 2) continue;
            PolyUOp *idx_expr = idx_uop->src[1];

            /* Check if rng is in backward slice */
            if (s.has_reach && (s.buf_reach[bi] & (1ULL << axis)))
              num_strides++;

            /* Split on ADD and extract stride for this rng */
            PolyUOp *addends[256];
            int n_add = split_uop_add(idx_expr, addends, 256);
            for (int j = 0; j < n_add; j++) {
              PolyUOp *c = addends[j];
              if (c == rng) {
                sum_strides += 1;
              } else if (c->op == POLY_OP_MUL && c->n_src == 2) {
                if (c->src[0] == rng && c->src[1]->op == POLY_OP_CONST && c->src[1]->arg.kind == POLY_ARG_INT)
                  sum_strides += c->src[1]->arg.i;
                else if (c->src[1] == rng && c->src[0]->op == POLY_OP_CONST && c->src[0]->arg.kind == POLY_ARG_INT)
                  sum_strides += c->src[0]->arg.i;
              }
            }
          }

          if (n_choices < (int)(sizeof(choices) / sizeof(choices[0])))
            choices[n_choices++] = (UpChoice){ num_strides, sum_strides, axis, amount };
        }
      }

      if (n_choices == 0) break;

      /* Sort: lowest (num_strides, sum_strides) first */
      for (int i = 0; i < n_choices - 1; i++) {
        for (int j = i + 1; j < n_choices; j++) {
          bool swap = false;
          if (choices[j].num_strides < choices[i].num_strides) swap = true;
          else if (choices[j].num_strides == choices[i].num_strides &&
                   choices[j].sum_strides < choices[i].sum_strides) swap = true;
          if (swap) { UpChoice tmp = choices[i]; choices[i] = choices[j]; choices[j] = tmp; }
        }
      }

      int best_axis = choices[0].axis;
      int best_amount = choices[0].amount;
      if (best_axis < s.n_rngs && s.shape[best_axis] > 1)
        sched_shift_to(&s, s.rngs[best_axis], best_amount, POLY_AXIS_UPCAST, false);
      /* Mark the original axis as upcasted -- after refresh, find the axis by matching the rng */
      /* Since indices shift after refresh, we track by setting the flag before refresh */
      /* Actually, upcasted_axis tracks by index which changes after shift_to+refresh.
       * Use a simple counter limit instead (tinygrad limits upcast_size < 32) */
      (void)upcasted_axis; /* The while-loop condition handles termination */
      /* Prevent infinite loop: break if nothing changed */
      if (sched_upcast_size(&s) < 32 && n_choices > 0) {
        upcasted_axis[best_axis] = true;
      }
    }
  }

  /* == Reduce UNROLL (heuristic.py:135-149) == */
  if (s.has_reduce) {
    int unroll_dims[SCHED_MAX_RNGS];
    int n_unroll = sched_unrollable_dims(&s, unroll_dims, SCHED_MAX_RNGS);

    if (n_unroll > 0 && (sched_upcast_size(&s) <= 4 || !sched_upcasted(&s)) && sched_upcast_size(&s) < 64) {
      int last = unroll_dims[n_unroll - 1];
      int64_t last_sz = s.shape[last];

      if (last_sz <= 32) {
        /* Unroll fully (amount = full size) */
        if (last < s.n_rngs && last_sz > 1)
          sched_shift_to(&s, s.rngs[last], last_sz, POLY_AXIS_UNROLL, false);
        /* If small, try unrolling a second reduce dim */
        n_unroll = sched_unrollable_dims(&s, unroll_dims, SCHED_MAX_RNGS);
        if (n_unroll > 0 && last_sz <= 3 && s.shape[unroll_dims[n_unroll - 1]] <= 3) {
          int last2 = unroll_dims[n_unroll - 1];
          if (last2 < s.n_rngs && s.shape[last2] > 1)
            sched_shift_to(&s, s.rngs[last2], s.shape[last2], POLY_AXIS_UNROLL, false);
        }
      } else {
        /* Partial unroll by 4 if divisible */
        if (last_sz % 4 == 0 && last < s.n_rngs)
          sched_shift_to(&s, s.rngs[last], 4, POLY_AXIS_UNROLL, false);
      }
    }
  }

  /* == Default upcast fallback (heuristic.py:151-154) ==
   * If nothing upcasted and last upcastable dim % upcast_amount == 0,
   * upcast by that amount. Use max_vec_width from caps (4 for SSE, 8 for AVX2). */
  if (!sched_upcasted(&s)) {
    int upcast_amount = (caps.max_vec_width >= 8) ? 8 : 4;
    int up_dims[SCHED_MAX_RNGS];
    int n_up = sched_upcastable_dims(&s, up_dims, SCHED_MAX_RNGS);
    if (n_up > 0) {
      int last = up_dims[n_up - 1];
      /* Try preferred width first, fall back to 4 if not divisible */
      if (s.shape[last] % upcast_amount == 0 && last < s.n_rngs)
        sched_shift_to(&s, s.rngs[last], upcast_amount, POLY_AXIS_UPCAST, false);
      else if (upcast_amount > 4 && s.shape[last] % 4 == 0 && last < s.n_rngs)
        sched_shift_to(&s, s.rngs[last], 4, POLY_AXIS_UPCAST, false);
    }
  }

  return s.ast;
}

/* ── BEAM search optimizer ──────────────────────────────────────────────
 *
 * Explores the optimization space by trying many candidate optimizations,
 * compiling and timing each, and keeping the top-k. Finds better
 * optimizations than the heuristic for non-trivial kernels.
 *
 * Action space: UPCAST and UNROLL with various axis/amount combos.
 * For each beam member, enumerate all valid actions, compile candidates,
 * time them on actual hardware, sort by execution time, keep top beam_width.
 *
 * Requires native runtime (fork+clang for compilation, clock_gettime for
 * timing). Disabled in WASM builds. Future: use WASM JIT backend for
 * compile+time, IndexedDB for cache.
 * ────────────────────────────────────────────────────────────────────── */

#ifndef __EMSCRIPTEN__

typedef enum { POLY_OPT_UPCAST, POLY_OPT_UNROLL } PolyOptOp;

typedef struct {
  PolyOptOp op;
  int axis;
  int64_t amount;
} PolyBeamAction;

/* Static action table */
static const int64_t beam_upcast_amounts[] = {2, 3, 4, 5, 7, 8};
static const int beam_n_upcast_amounts = 6;
static const int64_t beam_unroll_amounts[] = {2, 3, 4, 7};
static const int beam_n_unroll_amounts = 4;
#define BEAM_MAX_AXIS 8
#define BEAM_MAX_ACTIONS ((6 * BEAM_MAX_AXIS) + (4 * 5))  /* 68 */
#define BEAM_MAX_BEAM 16
#define BEAM_MAX_ITERS 5
#define BEAM_MAX_CANDIDATES (BEAM_MAX_BEAM * BEAM_MAX_ACTIONS)

typedef struct {
  OptScheduler sched;
  double time_us;
  PolyBeamAction actions[BEAM_MAX_ITERS];
  int n_actions;
} BeamEntry;

typedef struct {
  OptScheduler sched;
  double time_us;
  PolyBeamAction actions[BEAM_MAX_ITERS];
  int n_actions;
} BeamCandidate;

/* Copy an OptScheduler. Shallow copy is sufficient because sched_shift_to
 * creates new UOps via poly_uop_substitute (the old AST is untouched). */
static void sched_copy(OptScheduler *dst, const OptScheduler *src) {
  *dst = *src;
}

/* Try to apply a single BEAM action to a scheduler. Returns true on success. */
static bool sched_apply_action(OptScheduler *s, PolyBeamAction act) {
  int dims[SCHED_MAX_RNGS];
  int n_dims;

  if (act.op == POLY_OPT_UPCAST) {
    n_dims = sched_upcastable_dims(s, dims, SCHED_MAX_RNGS);
  } else {
    n_dims = sched_unrollable_dims(s, dims, SCHED_MAX_RNGS);
  }

  if (act.axis >= n_dims) return false;
  int idx = dims[act.axis];
  if (idx >= s->n_rngs) return false;
  if (s->shape[idx] <= 1) return false;
  if (s->shape[idx] % act.amount != 0) return false;

  /* Limit total upcast+unroll product to prevent code explosion.
   * 64 is reasonable: e.g. UPCAST 4 on two axes = 16, or UPCAST 8 + UNROLL 4 = 32. */
  int64_t cur_prod = sched_upcast_size(s);
  if (cur_prod * act.amount > 64) return false;

  PolyAxisType new_type = (act.op == POLY_OPT_UPCAST) ? POLY_AXIS_UPCAST : POLY_AXIS_UNROLL;
  PolyUOp *result = sched_shift_to(s, s->rngs[idx], act.amount, new_type, false);
  return result != NULL;
}

/* Time a single kernel execution using clock_gettime (CLOCK_MONOTONIC). */
static double time_us_now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

/* Compile a kernel AST through the full post-optimization pipeline,
 * render to C, compile with clang, allocate test buffers, and time execution.
 * Returns median time in microseconds. Returns INFINITY on failure. */
static double beam_compile_and_time(PolyCtx *ctx, PolyUOp *sink,
                                    PolyRewriteOpts opts, int reps) {
  /* Run through the rest of the codegen pipeline (post-optimization stages).
   * Setting optimize=false skips the preprocessing+apply_opts pass since
   * opts have already been applied by the BEAM search. */
  PolyRewriteOpts post_opts = opts;
  post_opts.optimize = false;
  post_opts.beam_width = 0;
  sink = poly_full_rewrite_to_sink_ex(ctx, sink, post_opts);
  /* Control flow now runs inside poly_full_rewrite_to_sink_ex (tinygrad parity). */

  /* Linearize */
  int n_uops = 0;
  PolyUOp **uops = poly_linearize_rewritten(ctx, sink, &n_uops);
  if (!uops || n_uops == 0) return INFINITY;

  /* UOp count filter: skip huge kernels */
  if (n_uops > 3000) {
    free(uops);
    return INFINITY;
  }

  /* Render C */
  char fn_name[64];
  snprintf(fn_name, sizeof(fn_name), "beam_%d", (int)(uintptr_t)sink & 0xFFFF);
  char *source = poly_render_c(uops, n_uops, fn_name);
  free(uops);
  if (!source) return INFINITY;

  /* Compile */
  PolyProgram *prog = poly_compile_c(source, fn_name);
  free(source);
  if (!prog) return INFINITY;

  /* Collect PARAM count and buffer sizes from the sink's toposort */
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, sink, &n_topo);
  int n_params = 0;
  int64_t param_sizes[64];
  for (int i = 0; i < n_topo; i++) {
    if (topo[i]->op == POLY_OP_PARAM && n_params < 64) {
      /* Estimate buffer size from dtype ptr size field */
      int64_t sz = topo[i]->dtype.ptr_size;
      if (sz <= 0) sz = 1024;  /* default */
      param_sizes[n_params] = sz;
      n_params++;
    }
  }

  if (n_params == 0) {
    poly_program_destroy(prog);
    return INFINITY;
  }

  /* Allocate test buffers (random float32 data) */
  void *bufs[64];
  for (int i = 0; i < n_params; i++) {
    int64_t nbytes = param_sizes[i] * 4;  /* float32 */
    if (nbytes <= 0) nbytes = 4096;
    bufs[i] = calloc(1, (size_t)nbytes);
    /* Fill with small random values to avoid NaN/inf in transcendentals */
    float *fp = (float *)bufs[i];
    int n_elems = (int)(nbytes / 4);
    for (int j = 0; j < n_elems; j++)
      fp[j] = 0.1f + (float)(j % 100) * 0.01f;
  }

  /* Warm up */
  poly_program_call(prog, bufs, n_params);

  /* Time execution */
  double times[16];
  if (reps > 16) reps = 16;
  if (reps < 1) reps = 1;
  for (int r = 0; r < reps; r++) {
    double t0 = time_us_now();
    poly_program_call(prog, bufs, n_params);
    double t1 = time_us_now();
    times[r] = t1 - t0;
  }

  /* Sort times, take median */
  for (int i = 0; i < reps - 1; i++)
    for (int j = i + 1; j < reps; j++)
      if (times[j] < times[i]) { double t = times[i]; times[i] = times[j]; times[j] = t; }
  double median = times[reps / 2];

  /* Cleanup */
  for (int i = 0; i < n_params; i++) free(bufs[i]);
  poly_program_destroy(prog);

  return median;
}

/* ── Disk cache for BEAM results ─────────────────────────────────────── */

/* FNV-1a hash over the AST toposort (structural hash for cache key) */
static uint64_t beam_ast_hash(PolyCtx *ctx, PolyUOp *sink) {
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, sink, &n_topo);
  uint64_t h = 0xcbf29ce484222325ULL;
  for (int i = 0; i < n_topo; i++) {
    h ^= (uint64_t)topo[i]->op;
    h *= 0x100000001b3ULL;
    h ^= (uint64_t)topo[i]->dtype.bitsize;
    h *= 0x100000001b3ULL;
    h ^= poly_arg_hash(topo[i]->arg);
    h *= 0x100000001b3ULL;
    h ^= (uint64_t)topo[i]->n_src;
    h *= 0x100000001b3ULL;
  }
  return h;
}

static int beam_cache_dir(char *dir, int cap) {
  const char *xdg = getenv("XDG_CACHE_HOME");
  const char *home = getenv("HOME");
  if (xdg && xdg[0])
    snprintf(dir, cap, "%s/polygrad/beam", xdg);
  else if (home && home[0])
    snprintf(dir, cap, "%s/.cache/polygrad/beam", home);
  else
    return -1;

  /* mkdir -p: create parent dirs */
  char parent[512];
  snprintf(parent, sizeof(parent), "%s", dir);
  char *s = parent + 1;
  while (*s) {
    if (*s == '/') {
      *s = '\0';
      mkdir(parent, 0755);
      *s = '/';
    }
    s++;
  }
  if (mkdir(dir, 0755) == -1 && errno != EEXIST) return -1;
  return 0;
}

/* Cache entry: [n_actions:uint8][actions: n * (op:uint8, axis:uint8, amount:int64)] */
static bool beam_cache_load(uint64_t key, PolyBeamAction *actions, int *n_actions) {
  char dir[512], path[576];
  if (beam_cache_dir(dir, sizeof(dir)) != 0) return false;
  snprintf(path, sizeof(path), "%s/%016llx.bin", dir, (unsigned long long)key);

  FILE *f = fopen(path, "rb");
  if (!f) return false;

  uint8_t n;
  if (fread(&n, 1, 1, f) != 1 || n > BEAM_MAX_ITERS) { fclose(f); return false; }
  *n_actions = n;
  for (int i = 0; i < n; i++) {
    uint8_t op_byte, axis_byte;
    int64_t amount;
    if (fread(&op_byte, 1, 1, f) != 1) { fclose(f); return false; }
    if (fread(&axis_byte, 1, 1, f) != 1) { fclose(f); return false; }
    if (fread(&amount, sizeof(amount), 1, f) != 1) { fclose(f); return false; }
    actions[i] = (PolyBeamAction){ .op = (PolyOptOp)op_byte, .axis = axis_byte, .amount = amount };
  }
  fclose(f);
  return true;
}

static void beam_cache_save(uint64_t key, const PolyBeamAction *actions, int n_actions) {
  char dir[512], path[576];
  if (beam_cache_dir(dir, sizeof(dir)) != 0) return;
  snprintf(path, sizeof(path), "%s/%016llx.bin", dir, (unsigned long long)key);

  FILE *f = fopen(path, "wb");
  if (!f) return;
  uint8_t n = (uint8_t)n_actions;
  fwrite(&n, 1, 1, f);
  for (int i = 0; i < n_actions; i++) {
    uint8_t op_byte = (uint8_t)actions[i].op;
    uint8_t axis_byte = (uint8_t)actions[i].axis;
    fwrite(&op_byte, 1, 1, f);
    fwrite(&axis_byte, 1, 1, f);
    fwrite(&actions[i].amount, sizeof(actions[i].amount), 1, f);
  }
  fclose(f);
}

/* ── Main BEAM search loop ────────────────────────────────────────────── */

static int beam_candidate_cmp(const void *a, const void *b) {
  const BeamCandidate *ca = (const BeamCandidate *)a;
  const BeamCandidate *cb = (const BeamCandidate *)b;
  if (ca->time_us < cb->time_us) return -1;
  if (ca->time_us > cb->time_us) return 1;
  return 0;
}

static PolyUOp *poly_beam_search(PolyCtx *ctx, PolyUOp *sink,
                                  int beam_width, PolyRewriteOpts opts) {
  if (beam_width <= 0) return sink;
  if (beam_width > BEAM_MAX_BEAM) beam_width = BEAM_MAX_BEAM;

  /* Check disk cache */
  uint64_t cache_key = beam_ast_hash(ctx, sink);
  PolyBeamAction cached_actions[BEAM_MAX_ITERS];
  int cached_n = 0;
  if (beam_cache_load(cache_key, cached_actions, &cached_n) && cached_n > 0) {
    /* Replay cached actions */
    OptScheduler s;
    sched_init(&s, ctx, sink);
    for (int i = 0; i < cached_n; i++) {
      OptScheduler copy;
      sched_copy(&copy, &s);
      if (!sched_apply_action(&copy, cached_actions[i])) break;
      s = copy;
    }
    return s.ast;
  }

  /* Initialize beam with unoptimized baseline */
  BeamEntry *beam = (BeamEntry *)calloc(BEAM_MAX_BEAM, sizeof(BeamEntry));
  sched_init(&beam[0].sched, ctx, sink);
  beam[0].time_us = INFINITY;
  beam[0].n_actions = 0;
  int beam_size = 1;

  /* Time the baseline */
  beam[0].time_us = beam_compile_and_time(ctx, beam[0].sched.ast, opts, 3);

  /* Build action list */
  PolyBeamAction all_actions[BEAM_MAX_ACTIONS];
  int n_actions = 0;
  for (int axis = 0; axis < BEAM_MAX_AXIS; axis++) {
    for (int ai = 0; ai < beam_n_upcast_amounts; ai++) {
      all_actions[n_actions++] = (PolyBeamAction){
        .op = POLY_OPT_UPCAST, .axis = axis, .amount = beam_upcast_amounts[ai]
      };
    }
  }
  for (int axis = 0; axis < 5; axis++) {
    for (int ai = 0; ai < beam_n_unroll_amounts; ai++) {
      all_actions[n_actions++] = (PolyBeamAction){
        .op = POLY_OPT_UNROLL, .axis = axis, .amount = beam_unroll_amounts[ai]
      };
    }
  }

  BeamCandidate *candidates = (BeamCandidate *)calloc(BEAM_MAX_CANDIDATES, sizeof(BeamCandidate));

  for (int iter = 0; iter < BEAM_MAX_ITERS; iter++) {
    int n_cand = 0;

    /* Generate candidates from all beam members */
    for (int b = 0; b < beam_size; b++) {
      if (beam[b].n_actions >= BEAM_MAX_ITERS) continue;
      for (int a = 0; a < n_actions && n_cand < BEAM_MAX_CANDIDATES; a++) {
        OptScheduler copy;
        sched_copy(&copy, &beam[b].sched);
        if (!sched_apply_action(&copy, all_actions[a])) continue;

        candidates[n_cand].sched = copy;
        candidates[n_cand].n_actions = beam[b].n_actions + 1;
        memcpy(candidates[n_cand].actions, beam[b].actions,
               (size_t)beam[b].n_actions * sizeof(PolyBeamAction));
        candidates[n_cand].actions[beam[b].n_actions] = all_actions[a];
        candidates[n_cand].time_us = INFINITY;
        n_cand++;
      }
    }

    if (n_cand == 0) break;

    /* Compile and time each candidate */
    for (int i = 0; i < n_cand; i++) {
      candidates[i].time_us = beam_compile_and_time(
          ctx, candidates[i].sched.ast, opts, 3);

      /* Early stop: if > 3x slower than current best after timing, skip remaining reps */
      if (candidates[i].time_us > beam[0].time_us * 3.0 && beam[0].time_us < INFINITY)
        candidates[i].time_us = INFINITY;
    }

    /* Sort by time */
    qsort(candidates, (size_t)n_cand, sizeof(BeamCandidate), beam_candidate_cmp);

    /* Check convergence: best candidate not better than current best */
    if (n_cand > 0 && candidates[0].time_us >= beam[0].time_us - 0.01)
      break;

    /* Keep top beam_width */
    int new_size = n_cand < beam_width ? n_cand : beam_width;
    /* Filter out INF candidates */
    while (new_size > 0 && candidates[new_size - 1].time_us >= INFINITY)
      new_size--;
    if (new_size == 0) break;

    for (int i = 0; i < new_size; i++) {
      beam[i].sched = candidates[i].sched;
      beam[i].time_us = candidates[i].time_us;
      beam[i].n_actions = candidates[i].n_actions;
      memcpy(beam[i].actions, candidates[i].actions,
             (size_t)candidates[i].n_actions * sizeof(PolyBeamAction));
    }
    beam_size = new_size;
  }

  /* Save best result to disk cache */
  if (beam[0].n_actions > 0 && beam[0].time_us < INFINITY) {
    beam_cache_save(cache_key, beam[0].actions, beam[0].n_actions);
  }

  PolyUOp *result = beam[0].sched.ast;
  free(beam);
  free(candidates);
  return result;
}

#else /* __EMSCRIPTEN__ */

/* WASM stub: BEAM search requires native compilation (fork+clang).
 * Falls back to heuristic. Future: use WASM JIT backend for timing. */
static PolyUOp *poly_beam_search(PolyCtx *ctx, PolyUOp *sink,
                                  int beam_width, PolyRewriteOpts opts) {
  (void)beam_width;
  return poly_apply_opts_heuristic(ctx, sink, opts.caps);
}

#endif /* __EMSCRIPTEN__ */

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
  PolyOps reduce_op = red->arg.ops;

  /* Filter reduce ranges to actual RANGE nodes only.
   * Singleton dims may produce CONST(0) pseudo-ranges from rangeify;
   * these must not enter the AFTER/END chains (tinygrad invariant). */
  PolyUOp *reduce_ranges[POLY_MAX_DIMS];
  int n_reduce_range = 0;
  for (int j = 1; j < red->n_src; j++) {
    if (red->src[j]->op == POLY_OP_RANGE)
      reduce_ranges[n_reduce_range++] = red->src[j];
  }
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
      if (topo[i] == reduce_ranges[j]) { is_reduce = true; break; }
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
    loop_srcs[i + 2] = reduce_ranges[i];
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
    PolyUOp *end_srcs[2] = { chain, reduce_ranges[i] };
    chain = poly_uop(ctx, POLY_OP_END, POLY_VOID,
                      end_srcs, 2, poly_arg_none());
  }
  reduce_ctx_add_end(rctx, reduce_ranges, n_reduce_range, chain);

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
  PolyDType cmp_bt = (root->dtype.count > 1)
    ? poly_dtype_vec(POLY_BOOL, root->dtype.count) : POLY_BOOL;
  PolyUOp *cmp = poly_uop2(ctx, POLY_OP_CMPLT, cmp_bt, a, root->src[1],
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
  PolyDType cmp_bt = (root->dtype.count > 1)
    ? poly_dtype_vec(POLY_BOOL, root->dtype.count) : POLY_BOOL;
  PolyUOp *cmplt = poly_uop2(ctx, POLY_OP_CMPLT, cmp_bt,
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
 * For renderers without native FMA (CPU/ClangRenderer).
 * Gated on !caps.has_mulacc in poly_pm_decomp_with_caps().
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
 * rule_mul_add_to_mulacc — ADD(MUL(a, b), c) → MULACC(a, b, c)
 * Fusion rule for renderers with native FMA (CUDA).  Float scalars only.
 * Port of tinygrad: if Ops.MULACC in ops: a*b+c → MULACC(a,b,c)
 * Gated on caps.has_mulacc in poly_pm_decomp_with_caps().
 */
static PolyUOp *rule_mul_add_to_mulacc(PolyCtx *ctx, PolyUOp *root,
                                        const PolyBindings *b) {
  (void)b;
  if (root->op != POLY_OP_ADD || root->n_src != 2) return NULL;
  if (!poly_dtype_is_float(root->dtype)) return NULL;
  PolyUOp *mul = NULL, *add = NULL;
  if (root->src[0]->op == POLY_OP_MUL) {
    mul = root->src[0]; add = root->src[1];
  } else if (root->src[1]->op == POLY_OP_MUL) {
    mul = root->src[1]; add = root->src[0];
  } else return NULL;
  if (mul->n_src != 2) return NULL;
  if (!poly_dtype_eq(mul->dtype, root->dtype)) return NULL;
  if (!poly_dtype_eq(add->dtype, root->dtype)) return NULL;
  PolyUOp *srcs[3] = { mul->src[0], mul->src[1], add };
  return poly_uop(ctx, POLY_OP_MULACC, root->dtype, srcs, 3, poly_arg_none());
}

/*
 * rule_shl_add_to_mulacc — ADD(SHL(x, n), c) → MULACC(x, 2^n, c)
 * When MUL(x, pow2) was already decomposed to SHL by an earlier pass,
 * the ADD fusion needs to recognize the shifted form.
 * Ref: tinygrad decompositions.py:480
 */
static PolyUOp *rule_shl_add_to_mulacc(PolyCtx *ctx, PolyUOp *root,
                                         const PolyBindings *b) {
  (void)b;
  if (root->op != POLY_OP_ADD || root->n_src != 2) return NULL;
  if (!poly_dtype_is_int(root->dtype)) return NULL;
  PolyUOp *shl = NULL, *c = NULL;
  if (root->src[0]->op == POLY_OP_SHL) {
    shl = root->src[0]; c = root->src[1];
  } else if (root->src[1]->op == POLY_OP_SHL) {
    shl = root->src[1]; c = root->src[0];
  } else return NULL;
  if (shl->n_src != 2) return NULL;
  PolyUOp *n_const = shl->src[1];
  if (n_const->op != POLY_OP_CONST || n_const->arg.kind != POLY_ARG_INT) return NULL;
  int64_t shift = n_const->arg.i;
  if (shift < 0 || shift > 30) return NULL;
  PolyUOp *factor = poly_const_like_int(ctx, shl->src[0], 1LL << shift);
  PolyUOp *srcs[3] = { shl->src[0], factor, c };
  return poly_uop(ctx, POLY_OP_MULACC, root->dtype, srcs, 3, poly_arg_none());
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

static PolyUOp *u32_const(PolyCtx *ctx, uint32_t v) {
  return poly_uop0(ctx, POLY_OP_CONST, POLY_UINT32, poly_arg_int((int64_t)v));
}

static PolyUOp *u64_const(PolyCtx *ctx, uint64_t v) {
  return poly_uop0(ctx, POLY_OP_CONST, POLY_UINT64, poly_arg_int((int64_t)v));
}

static PolyUOp *u32_cast(PolyCtx *ctx, PolyUOp *u) {
  if (poly_dtype_eq(u->dtype, POLY_UINT32)) return u;
  return poly_uop1(ctx, POLY_OP_CAST, POLY_UINT32, u, poly_arg_none());
}

static PolyUOp *u64_cast(PolyCtx *ctx, PolyUOp *u) {
  if (poly_dtype_eq(u->dtype, POLY_UINT64)) return u;
  return poly_uop1(ctx, POLY_OP_CAST, POLY_UINT64, u, poly_arg_none());
}

static PolyUOp *u32_rol(PolyCtx *ctx, PolyUOp *x, int r) {
  PolyUOp *l = poly_uop2(ctx, POLY_OP_SHL, POLY_UINT32, x, u32_const(ctx, (uint32_t)r), poly_arg_none());
  PolyUOp *rr = poly_uop2(ctx, POLY_OP_SHR, POLY_UINT32, x, u32_const(ctx, (uint32_t)(32 - r)), poly_arg_none());
  return poly_uop2(ctx, POLY_OP_OR, POLY_UINT32, l, rr, poly_arg_none());
}

/*
 * rule_decomp_threefry32 — lower THREEFRY to pure integer ALU UOps.
 * This mirrors tinygrad's decomposition strategy (threefry2x32) but
 * emits a 32-bit lane value directly for current backends.
 */
static PolyUOp *rule_decomp_threefry32(PolyCtx *ctx, PolyUOp *root,
                                       const PolyBindings *b) {
  (void)b;
  if (root->op != POLY_OP_THREEFRY || root->n_src != 2) return NULL;

  /* Tinygrad-parity path: for uint64 THREEFRY, split into two uint32 lanes.
   * For uint32 input (legacy elementwise usage), high lanes are zero. */
  PolyUOp *x0, *x1, *key0, *key1;
  if (poly_dtype_eq(root->dtype, POLY_UINT64)) {
    PolyUOp *x64 = u64_cast(ctx, root->src[0]);
    PolyUOp *k64 = u64_cast(ctx, root->src[1]);
    PolyUOp *mask32 = u64_const(ctx, 0xFFFFFFFFull);
    PolyUOp *sh32 = u64_const(ctx, 32);
    x0 = u32_cast(ctx, poly_uop2(ctx, POLY_OP_AND, POLY_UINT64, x64, mask32, poly_arg_none()));
    x1 = u32_cast(ctx, poly_uop2(ctx, POLY_OP_AND, POLY_UINT64,
                                 poly_uop2(ctx, POLY_OP_SHR, POLY_UINT64, x64, sh32, poly_arg_none()),
                                 mask32, poly_arg_none()));
    key0 = u32_cast(ctx, poly_uop2(ctx, POLY_OP_AND, POLY_UINT64, k64, mask32, poly_arg_none()));
    key1 = u32_cast(ctx, poly_uop2(ctx, POLY_OP_AND, POLY_UINT64,
                                   poly_uop2(ctx, POLY_OP_SHR, POLY_UINT64, k64, sh32, poly_arg_none()),
                                   mask32, poly_arg_none()));
  } else {
    x0 = u32_cast(ctx, root->src[0]);
    x1 = u32_const(ctx, 0);
    key0 = u32_cast(ctx, root->src[1]);
    key1 = u32_const(ctx, 0);
  }

  PolyUOp *ks[3];
  ks[0] = key1;
  ks[1] = poly_uop2(ctx, POLY_OP_XOR, POLY_UINT32,
                    poly_uop2(ctx, POLY_OP_XOR, POLY_UINT32, key0, key1, poly_arg_none()),
                    u32_const(ctx, 0x1BD11BDAu), poly_arg_none());
  ks[2] = key0;

  PolyUOp *xr0 = poly_uop2(ctx, POLY_OP_ADD, POLY_UINT32, x0, ks[2], poly_arg_none());
  PolyUOp *xr1 = poly_uop2(ctx, POLY_OP_ADD, POLY_UINT32, x1, ks[0], poly_arg_none());

  static const int rotations[2][4] = {
    {13, 15, 26, 6},
    {17, 29, 16, 24},
  };

  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 4; j++) {
      int r = rotations[i & 1][j];
      PolyUOp *sum = poly_uop2(ctx, POLY_OP_ADD, POLY_UINT32, xr0, xr1, poly_arg_none());
      xr1 = poly_uop2(ctx, POLY_OP_XOR, POLY_UINT32, sum, u32_rol(ctx, xr1, r), poly_arg_none());
      xr0 = sum;
    }
    PolyUOp *round = u32_const(ctx, (uint32_t)(i + 1));
    xr0 = poly_uop2(ctx, POLY_OP_ADD, POLY_UINT32, xr0, ks[i % 3], poly_arg_none());
    xr1 = poly_uop2(ctx, POLY_OP_ADD, POLY_UINT32,
                    poly_uop2(ctx, POLY_OP_ADD, POLY_UINT32, xr1, ks[(i + 1) % 3], poly_arg_none()),
                    round, poly_arg_none());
  }

  if (poly_dtype_eq(root->dtype, POLY_UINT32)) return xr0;
  if (poly_dtype_eq(root->dtype, POLY_UINT64)) {
    PolyUOp *lo = u64_cast(ctx, xr0);
    PolyUOp *hi = poly_uop2(ctx, POLY_OP_SHL, POLY_UINT64, u64_cast(ctx, xr1),
                            u64_const(ctx, 32), poly_arg_none());
    return poly_uop2(ctx, POLY_OP_OR, POLY_UINT64, hi, lo, poly_arg_none());
  }
  return poly_uop1(ctx, POLY_OP_CAST, root->dtype, xr0, poly_arg_none());
}

/* Cached variants by (has_mulacc, has_threefry_native). */
static PolyPatternMatcher *g_pm_decomp_caps[2][2] = {{NULL, NULL}, {NULL, NULL}};

static PolyPatternMatcher *poly_pm_decomp_with_caps(bool has_mulacc, bool has_threefry) {
  PolyPatternMatcher **target = &g_pm_decomp_caps[has_mulacc ? 1 : 0][has_threefry ? 1 : 0];
  if (*target) return *target;

  PolyOpSet max_set = poly_opset_add((PolyOpSet){{0,0}}, POLY_OP_MAX);
  PolyRule rules[20];
  int n = 0;
  rules[n++] = (PolyRule){ poly_pat_ops(max_set, NULL, 0, NULL), rule_decomp_max };
  /* MUL(x:int, c:const) → SHL(x, log2(c)) when c is power of 2 */
  rules[n++] = (PolyRule){ poly_pat_op2(POLY_OP_MUL, poly_pat_any("x"),
      poly_pat_cvar("c"), NULL), rule_mul_to_shl };
  /* x * (-1) → NEG(x) */
  rules[n++] = (PolyRule){ poly_pat_op2(POLY_OP_MUL, poly_pat_any("x"),
      poly_pat_cvar("c"), NULL), rule_mul_neg1_to_neg };
  /* IDIV(x:int, c:const) → SHR(x, log2(c)) when c is power of 2 */
  rules[n++] = (PolyRule){ poly_pat_op2(POLY_OP_IDIV, poly_pat_any("x"),
      poly_pat_cvar("c"), NULL), rule_idiv_to_shr };
  /* MOD(x:int, c:const) → AND(x, c-1) when c is power of 2 */
  rules[n++] = (PolyRule){ poly_pat_op2(POLY_OP_MOD, poly_pat_any("x"),
      poly_pat_cvar("c"), NULL), rule_mod_to_and };
  /* x + NEG(y) → SUB(x, y) */
  rules[n++] = (PolyRule){ poly_pat_op2(POLY_OP_ADD, poly_pat_any("x"),
      poly_pat_op1(POLY_OP_NEG, poly_pat_any("y"), NULL), NULL),
    rule_add_neg_to_sub };

  if (!has_threefry) {
    PolyOpSet threefry_set = poly_opset_add((PolyOpSet){{0,0}}, POLY_OP_THREEFRY);
    rules[n++] = (PolyRule){ poly_pat_ops(threefry_set, NULL, 0, NULL), rule_decomp_threefry32 };
  }

  if (!has_mulacc) {
    /* CPU path: decompose MULACC → MUL+ADD */
    rules[n++] = (PolyRule){ poly_pat_ops(
        poly_opset_add((PolyOpSet){{0,0}}, POLY_OP_MULACC),
        NULL, 0, NULL), rule_mulacc_to_mul_add };
  } else {
    /* FMA path: fuse ADD(MUL(a,b), c) → MULACC(a,b,c) for floats (scalar + vector) */
    PolyOpSet add_set = poly_opset_add((PolyOpSet){{0,0}}, POLY_OP_ADD);
    rules[n++] = (PolyRule){ poly_pat_ops(add_set, NULL, 0, NULL),
      rule_mul_add_to_mulacc };
    /* SHL fusion: ADD(SHL(x,n), c) → MULACC(x, 2^n, c) for ints */
    /* SHL fusion: ADD(SHL(x,n), c) → MULACC(x, 2^n, c) for ints.
     * Renderer decomposes back to shl+add when profitable (x86: vpmulld is slow). */
    rules[n++] = (PolyRule){ poly_pat_ops(add_set, NULL, 0, NULL),
      rule_shl_add_to_mulacc };
  }

  /* RECIPROCAL(x) → FDIV(1, x) */
  rules[n++] = (PolyRule){ poly_pat_op1(POLY_OP_RECIPROCAL, poly_pat_any("x"),
      NULL), rule_recip_to_fdiv };
  /* a * (1 / b) → a / b */
  rules[n++] = (PolyRule){ poly_pat_op2(POLY_OP_MUL, poly_pat_any("a"),
      poly_pat_op2(POLY_OP_FDIV, poly_pat_cvar("one"),
        poly_pat_any("b"), NULL), NULL), rule_mul_fdiv1_to_fdiv };

  *target = poly_pm_new(rules, n);
  return *target;
}

/* Default: CPU decomp (no MULACC support). */
static PolyPatternMatcher *poly_pm_decomp(void) {
  return poly_pm_decomp_with_caps(false, false);
}

/* ── pm_transcendental: EXP2/LOG2/SIN → polynomial approximation ─────── */

/* Dtype-parametric helpers for IEEE 754 bit manipulation. */
static int xd_mantissa_bits(PolyDType dt) { return dt.bitsize == 64 ? 52 : 23; }
static int xd_exponent_bias(PolyDType dt) { return dt.bitsize == 64 ? 1023 : 127; }
static int64_t xd_exponent_mask(PolyDType dt) { return dt.bitsize == 64 ? 0x7FFLL : 0xFFLL; }
static PolyDType xd_int_for_float(PolyDType dt) {
  PolyDType sdt = poly_dtype_scalar(dt);
  PolyDType it = (sdt.bitsize == 64) ? POLY_INT64 : POLY_INT32;
  return (dt.count > 1) ? poly_dtype_vec(it, dt.count) : it;
}

/* Build a polyN Horner evaluation: acc = c[0]; for i in 1..n: acc = acc*x + c[i] */
static PolyUOp *xd_polyN(PolyCtx *ctx, PolyDType ft, PolyUOp *x,
                          const double *coeffs, int ncoeffs) {
  PolyUOp *u = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(coeffs[0]));
  for (int i = 1; i < ncoeffs; i++) {
    u = poly_uop2(ctx, POLY_OP_MUL, ft, u, x, poly_arg_none());
    u = poly_uop2(ctx, POLY_OP_ADD, ft, u,
                  poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(coeffs[i])),
                  poly_arg_none());
  }
  return u;
}

/* _lazy_map_numbers: mask +-inf/NaN to replacement values.
 * x.ne(inf).where(x.ne(x).where(nan_val, x.ne(-inf).where(ratio, ninf_val)), pinf_val) */
static PolyUOp *xd_lazy_map_numbers(PolyCtx *ctx, PolyDType ft, PolyUOp *d,
                                     PolyUOp *pinf_val, PolyUOp *ninf_val,
                                     PolyUOp *nan_val, PolyUOp *ratio) {
  PolyDType bt = (ft.count > 1) ? poly_dtype_vec(POLY_BOOL, ft.count) : POLY_BOOL;
  PolyUOp *f_neg_inf = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(-__builtin_inf()));
  PolyUOp *f_pos_inf = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(__builtin_inf()));
  PolyUOp *nan_chk    = poly_uop2(ctx, POLY_OP_CMPNE, bt, d, d, poly_arg_none());
  PolyUOp *neginf_chk = poly_uop2(ctx, POLY_OP_CMPNE, bt, d, f_neg_inf, poly_arg_none());
  PolyUOp *posinf_chk = poly_uop2(ctx, POLY_OP_CMPNE, bt, d, f_pos_inf, poly_arg_none());
  PolyUOp *inner = poly_uop3(ctx, POLY_OP_WHERE, ft, neginf_chk, ratio, ninf_val, poly_arg_none());
  PolyUOp *mid   = poly_uop3(ctx, POLY_OP_WHERE, ft, nan_chk, nan_val, inner, poly_arg_none());
  return poly_uop3(ctx, POLY_OP_WHERE, ft, posinf_chk, mid, pinf_val, poly_arg_none());
}

/* rintk: round float d to nearest integer (away from 0). */
static PolyUOp *xd_rintk(PolyCtx *ctx, PolyDType ft, PolyDType it, PolyUOp *d) {
  PolyDType bt = (ft.count > 1) ? poly_dtype_vec(POLY_BOOL, ft.count) : POLY_BOOL;
  PolyUOp *f_zero     = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(0.0));
  PolyUOp *f_neg_half = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(-0.5));
  PolyUOp *f_half     = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(0.5));
  PolyUOp *lt0    = poly_uop2(ctx, POLY_OP_CMPLT, bt, d, f_zero, poly_arg_none());
  PolyUOp *offset = poly_uop3(ctx, POLY_OP_WHERE, ft, lt0, f_neg_half, f_half, poly_arg_none());
  PolyUOp *rounded = poly_uop2(ctx, POLY_OP_ADD, ft, d, offset, poly_arg_none());
  return poly_uop1(ctx, POLY_OP_CAST, it, rounded, poly_arg_none());
}

/* pow2if: cast(2^q, float_dtype) via IEEE 754 exponent construction.
 * ((q + bias) << mantissa_bits).bitcast(float) */
static PolyUOp *xd_pow2if(PolyCtx *ctx, PolyDType ft, PolyDType it, PolyUOp *q) {
  int bias = xd_exponent_bias(ft);
  int mbits = xd_mantissa_bits(ft);
  PolyUOp *i_bias = poly_uop0(ctx, POLY_OP_CONST, it, poly_arg_int(bias));
  PolyUOp *i_mb   = poly_uop0(ctx, POLY_OP_CONST, it, poly_arg_int(mbits));
  PolyUOp *added  = poly_uop2(ctx, POLY_OP_ADD, it, q, i_bias, poly_arg_none());
  PolyUOp *shifted = poly_uop2(ctx, POLY_OP_SHL, it, added, i_mb, poly_arg_none());
  return poly_uop1(ctx, POLY_OP_BITCAST, ft, shifted, poly_arg_none());
}

/* ldexp2k: d * 2^e. Splits e into two halves to avoid overflow in pow2if. */
static PolyUOp *xd_ldexp2k(PolyCtx *ctx, PolyDType ft, PolyDType it, PolyUOp *d, PolyUOp *e) {
  PolyUOp *i_two = poly_uop0(ctx, POLY_OP_CONST, it, poly_arg_int(2));
  PolyUOp *half_e  = poly_uop2(ctx, POLY_OP_IDIV, it, e, i_two, poly_arg_none());
  PolyUOp *other_e = poly_uop2(ctx, POLY_OP_SUB, it, e, half_e, poly_arg_none());
  PolyUOp *pow1 = xd_pow2if(ctx, ft, it, half_e);
  PolyUOp *pow2 = xd_pow2if(ctx, ft, it, other_e);
  PolyUOp *r = poly_uop2(ctx, POLY_OP_MUL, ft, d, pow1, poly_arg_none());
  return poly_uop2(ctx, POLY_OP_MUL, ft, r, pow2, poly_arg_none());
}

/* ldexp3k: d * 2^e via bit manipulation.
 * (d.bitcast(int) + e.cast(int) << mantissa_bits).bitcast(float) */
static PolyUOp *xd_ldexp3k(PolyCtx *ctx, PolyDType ft, PolyDType it, PolyUOp *d, PolyUOp *e) {
  int mbits = xd_mantissa_bits(ft);
  PolyUOp *i_mb = poly_uop0(ctx, POLY_OP_CONST, it, poly_arg_int(mbits));
  PolyUOp *d_bits = poly_uop1(ctx, POLY_OP_BITCAST, it, d, poly_arg_none());
  PolyUOp *e_int  = poly_uop1(ctx, POLY_OP_CAST, it, e, poly_arg_none());
  PolyUOp *e_shift = poly_uop2(ctx, POLY_OP_SHL, it, e_int, i_mb, poly_arg_none());
  PolyUOp *m_bits = poly_uop2(ctx, POLY_OP_ADD, it, d_bits, e_shift, poly_arg_none());
  return poly_uop1(ctx, POLY_OP_BITCAST, ft, m_bits, poly_arg_none());
}

/* ilogb2k: integer part of log2(d) for normalized fp values.
 * (d.bitcast(int) >> mantissa_bits) & exponent_mask - exponent_bias */
static PolyUOp *xd_ilogb2k(PolyCtx *ctx, PolyDType ft, PolyDType it, PolyUOp *d) {
  int mbits = xd_mantissa_bits(ft);
  int64_t emask = xd_exponent_mask(ft);
  int bias = xd_exponent_bias(ft);
  PolyUOp *i_mb   = poly_uop0(ctx, POLY_OP_CONST, it, poly_arg_int(mbits));
  PolyUOp *i_mask = poly_uop0(ctx, POLY_OP_CONST, it, poly_arg_int(emask));
  PolyUOp *i_bias = poly_uop0(ctx, POLY_OP_CONST, it, poly_arg_int(bias));
  PolyUOp *d_bits = poly_uop1(ctx, POLY_OP_BITCAST, it, d, poly_arg_none());
  PolyUOp *exp_bits = poly_uop2(ctx, POLY_OP_SHR, it, d_bits, i_mb, poly_arg_none());
  PolyUOp *masked = poly_uop2(ctx, POLY_OP_AND, it, exp_bits, i_mask, poly_arg_none());
  return poly_uop2(ctx, POLY_OP_SUB, it, masked, i_bias, poly_arg_none());
}

/*
 * rule_decomp_exp2 — Port of tinygrad's xexp2 from decompositions.py.
 *
 * EXP2(d) → polynomial approximation with IEEE 754 bit manipulation.
 * Supports float32 (7 coefficients) and float64 (12 coefficients).
 *
 * Algorithm:
 *   1. _lazy_map_numbers: mask +-inf/NaN to 0
 *   2. rintk: round x to nearest integer q
 *   3. s = x - q (fractional part)
 *   4. polyN: Horner polynomial on s
 *   5. ldexp2k: multiply by 2^q via IEEE 754 exponent construction
 *   6. Edge cases: overflow->inf, underflow->0, NaN->NaN
 */
static PolyUOp *rule_decomp_exp2(PolyCtx *ctx, PolyUOp *root,
                                  const PolyBindings *b) {
  (void)b;
  PolyUOp *d = root->src[0];
  PolyDType sft = poly_dtype_scalar(root->dtype);
  if (!poly_dtype_is_float(sft) ||
      (sft.bitsize != 32 && sft.bitsize != 64))
    return NULL;

  PolyDType ft = root->dtype;  /* may be vec */
  PolyDType it = xd_int_for_float(ft);
  PolyDType bt = (ft.count > 1) ? poly_dtype_vec(POLY_BOOL, ft.count) : POLY_BOOL;
  bool is_f64 = (sft.bitsize == 64);

  /* ── Constants ───────────────────────────────────────────────────── */
  PolyUOp *f_zero    = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(0.0));
  PolyUOp *f_pos_inf = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(__builtin_inf()));
  PolyUOp *f_nan     = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(__builtin_nan("")));
  PolyUOp *b_true    = poly_uop0(ctx, POLY_OP_CONST, bt, poly_arg_bool(true));

  /* Dtype-specific overflow/underflow thresholds (from tinygrad) */
  double upper = is_f64 ? 1024.0 : 128.0;
  double lower = is_f64 ? -2000.0 : -150.0;
  PolyUOp *f_upper = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(upper));
  PolyUOp *f_lower = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(lower));

  /* Polynomial coefficients (from tinygrad decompositions.py) */
  static const double coeffs_f32[] = {
    0.1535920892e-3, 0.1339262701e-2, 0.9618384764e-2,
    0.5550347269e-1, 0.2402264476e+0, 0.6931471825e+0, 1.0
  };
  static const double coeffs_f64[] = {
    0.4434359082926529454e-9, 0.7073164598085707425e-8,
    0.1017819260921760451e-6, 0.1321543872511327615e-5,
    0.1525273353517584730e-4, 0.1540353045101147808e-3,
    0.1333355814670499073e-2, 0.9618129107597600536e-2,
    0.5550410866482046596e-1, 0.2402265069591012214e+0,
    0.6931471805599452862e+0, 0.1000000000000000000e+1
  };
  const double *coeffs = is_f64 ? coeffs_f64 : coeffs_f32;
  int ncoeffs = is_f64 ? 12 : 7;

  /* ── Step 1: _lazy_map_numbers — mask +-inf/NaN to 0 ────────────── */
  PolyUOp *x = xd_lazy_map_numbers(ctx, ft, d, f_zero, f_zero, f_zero, d);
  PolyUOp *nan_chk = poly_uop2(ctx, POLY_OP_CMPNE, bt, d, d, poly_arg_none());

  /* ── Step 2: rintk — round to nearest integer ──────────────────── */
  PolyUOp *q = xd_rintk(ctx, ft, it, x);

  /* ── Step 3: fractional part s = x - q.cast(float) ────────────── */
  PolyUOp *q_float = poly_uop1(ctx, POLY_OP_CAST, ft, q, poly_arg_none());
  PolyUOp *s = poly_uop2(ctx, POLY_OP_SUB, ft, x, q_float, poly_arg_none());

  /* ── Step 4: polyN — Horner's method ───────────────────────────── */
  PolyUOp *u = xd_polyN(ctx, ft, s, coeffs, ncoeffs);

  /* ── Step 5: ldexp2k — u * 2^q ─────────────────────────────────── */
  PolyUOp *result = xd_ldexp2k(ctx, ft, it, u, q);

  /* ── Step 6: edge cases ─────────────────────────────────────────── */
  /* (d >= upper).where(inf, result) */
  PolyUOp *cmp_hi = poly_uop2(ctx, POLY_OP_CMPLT, bt, d, f_upper, poly_arg_none());
  PolyUOp *ge_hi  = poly_uop2(ctx, POLY_OP_CMPNE, bt, cmp_hi, b_true, poly_arg_none());
  result = poly_uop3(ctx, POLY_OP_WHERE, ft, ge_hi, f_pos_inf, result, poly_arg_none());
  /* (d < lower).where(0, result) */
  PolyUOp *cmp_lo = poly_uop2(ctx, POLY_OP_CMPLT, bt, d, f_lower, poly_arg_none());
  result = poly_uop3(ctx, POLY_OP_WHERE, ft, cmp_lo, f_zero, result, poly_arg_none());
  /* d.ne(d).where(nan, result) — NaN propagation */
  result = poly_uop3(ctx, POLY_OP_WHERE, ft, nan_chk, f_nan, result, poly_arg_none());

  return result;
}

/*
 * rule_decomp_log2 — Port of tinygrad's xlog2.
 *
 * LOG2(d) → polynomial + IEEE754 exponent/mantissa manipulation.
 * Supports float32 (3 coefficients) and float64 (7 coefficients).
 */
static PolyUOp *rule_decomp_log2(PolyCtx *ctx, PolyUOp *root,
                                  const PolyBindings *b) {
  (void)b;
  PolyUOp *d = root->src[0];
  PolyDType sft = poly_dtype_scalar(root->dtype);
  if (!poly_dtype_is_float(sft) ||
      (sft.bitsize != 32 && sft.bitsize != 64))
    return NULL;

  PolyDType ft = root->dtype;  /* may be vec */
  PolyDType it = xd_int_for_float(ft);
  PolyDType bt = (ft.count > 1) ? poly_dtype_vec(POLY_BOOL, ft.count) : POLY_BOOL;
  bool is_f64 = (sft.bitsize == 64);

  /* Constants */
  PolyUOp *f_zero     = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(0.0));
  PolyUOp *f_neg_zero = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(-0.0));
  PolyUOp *f_one      = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(1.0));
  PolyUOp *f_neg_inf  = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(-__builtin_inf()));
  PolyUOp *f_pos_inf  = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(__builtin_inf()));
  PolyUOp *f_nan      = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(__builtin_nan("")));
  PolyUOp *f_1e4      = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(1e-4));
  PolyUOp *f_4_3      = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(1.0 / 0.75));
  PolyUOp *f_64       = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(64.0));
  PolyUOp *f_2p64     = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(18446744073709551616.0));

  /* Denormal handling: scale up subnormals by 2^64 */
  PolyUOp *is_denormal = poly_uop2(ctx, POLY_OP_CMPLT, bt, d, f_1e4, poly_arg_none());
  PolyUOp *scaled = poly_uop2(ctx, POLY_OP_MUL, ft, d, f_2p64, poly_arg_none());
  PolyUOp *a = poly_uop3(ctx, POLY_OP_WHERE, ft, is_denormal, scaled, d, poly_arg_none());

  /* e = ilogb2k(a * (1/0.75)), using shared helper */
  PolyUOp *a_scaled = poly_uop2(ctx, POLY_OP_MUL, ft, a, f_4_3, poly_arg_none());
  PolyUOp *e_int = xd_ilogb2k(ctx, ft, it, a_scaled);
  PolyUOp *e = poly_uop1(ctx, POLY_OP_CAST, ft, e_int, poly_arg_none());

  /* m = ldexp3k(a, -e), using shared helper */
  PolyUOp *neg_e = poly_uop1(ctx, POLY_OP_NEG, ft, e, poly_arg_none());
  PolyUOp *m = xd_ldexp3k(ctx, ft, it, a, neg_e);

  /* Denormal exponent correction: subtract the 2^64 scaling */
  PolyUOp *e_minus64 = poly_uop2(ctx, POLY_OP_SUB, ft, e, f_64, poly_arg_none());
  PolyUOp *e_adj = poly_uop3(ctx, POLY_OP_WHERE, ft, is_denormal, e_minus64, e, poly_arg_none());

  /* x = (m - 1) / (m + 1) */
  PolyUOp *m_minus1 = poly_uop2(ctx, POLY_OP_SUB, ft, m, f_one, poly_arg_none());
  PolyUOp *m_plus1  = poly_uop2(ctx, POLY_OP_ADD, ft, m, f_one, poly_arg_none());
  PolyUOp *x = poly_uop2(ctx, POLY_OP_FDIV, ft, m_minus1, m_plus1, poly_arg_none());
  PolyUOp *x2 = poly_uop2(ctx, POLY_OP_MUL, ft, x, x, poly_arg_none());

  /* Polynomial: dtype-specific coefficients */
  static const double coeffs_f32[] = {0.4374550283, 0.5764790177, 0.9618012905120};
  static const double coeffs_f64[] = {
    0.2211941750456081490e+0, 0.2200768693152277689e+0,
    0.2623708057488514656e+0, 0.3205977477944495502e+0,
    0.4121985945485324709e+0, 0.5770780162997058982e+0,
    0.96179669392608091449
  };
  const double *coeffs = is_f64 ? coeffs_f64 : coeffs_f32;
  int ncoeffs = is_f64 ? 7 : 3;
  PolyUOp *t = xd_polyN(ctx, ft, x2, coeffs, ncoeffs);

  /* Result assembly: r = t*(x*x2) + e_adj + x*k1 [+ x*k2 for f32] */
  PolyUOp *xx2 = poly_uop2(ctx, POLY_OP_MUL, ft, x, x2, poly_arg_none());
  PolyUOp *r = poly_uop2(ctx, POLY_OP_MUL, ft, t, xx2, poly_arg_none());
  r = poly_uop2(ctx, POLY_OP_ADD, ft, r, e_adj, poly_arg_none());

  if (is_f64) {
    /* f64: single multiplier constant, no s_lo term */
    PolyUOp *f_k1 = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(2.885390081777926774));
    r = poly_uop2(ctx, POLY_OP_ADD, ft, r,
                  poly_uop2(ctx, POLY_OP_MUL, ft, x, f_k1, poly_arg_none()),
                  poly_arg_none());
  } else {
    /* f32: k1 + s_lo term (x*k2) for extra precision */
    PolyUOp *f_k1 = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(2.8853900432586669922));
    PolyUOp *f_k2 = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(3.2734474483568488616e-08));
    r = poly_uop2(ctx, POLY_OP_ADD, ft, r,
                  poly_uop2(ctx, POLY_OP_MUL, ft, x, f_k1, poly_arg_none()),
                  poly_arg_none());
    r = poly_uop2(ctx, POLY_OP_ADD, ft, r,
                  poly_uop2(ctx, POLY_OP_MUL, ft, x, f_k2, poly_arg_none()),
                  poly_arg_none());
  }

  /* Edge cases (same for f32 and f64) */
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

/* sin_poly: trig_poly from tinygrad, dtype-aware.
 * Returns d * polyN(d*d, coeffs). Supports f32 (5 coeffs) and f64 (10 coeffs). */
static PolyUOp *sin_poly(PolyCtx *ctx, PolyUOp *d) {
  PolyDType ft = d->dtype;
  bool is_f64 = (ft.bitsize == 64);
  PolyUOp *d2 = poly_uop2(ctx, POLY_OP_MUL, ft, d, d, poly_arg_none());
  static const double coeffs_f32[] = {
    2.6083159809786593541503e-06, -0.0001981069071916863322258,
    0.00833307858556509017944336, -0.166666597127914428710938, 1.0
  };
  static const double coeffs_f64[] = {
    -7.97255955009037868891952e-18,  2.81009972710863200091251e-15,
    -7.64712219118158833288484e-13,  1.60590430605664501629054e-10,
    -2.50521083763502045810755e-08,  2.75573192239198747630416e-06,
    -0.000198412698412696162806809,  0.00833333333333332974823815,
    -0.166666666666666657414808,     1.0
  };
  const double *coeffs = is_f64 ? coeffs_f64 : coeffs_f32;
  int ncoeffs = is_f64 ? 10 : 5;
  PolyUOp *t = xd_polyN(ctx, ft, d2, coeffs, ncoeffs);
  return poly_uop2(ctx, POLY_OP_MUL, ft, d, t, poly_arg_none());
}

/* Payne-Hanek helper: select two_over_pi_f[i+offset] (f32 only). */
static PolyUOp *take_two_over_pi_f32(PolyCtx *ctx, PolyUOp *i_u64, int offset) {
  static const uint32_t two_over_pi_f[] = {
    0x00000000u, 0x28be60dbu, 0x9391054au, 0x7f09d5f4u,
    0x7d4d3770u, 0x36d8a566u, 0x4f10e410u
  };
  const int len = (int)(sizeof(two_over_pi_f) / sizeof(two_over_pi_f[0]));
  const int max_count = len - 2 - offset;
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

/* Cody-Waite _reduce_d for f32: 4-term PI subtraction. */
static PolyUOp *cody_waite_reduce_f32(PolyCtx *ctx, PolyDType ft,
                                       PolyUOp *x, PolyUOp *qf) {
  PolyUOp *d = poly_uop2(ctx, POLY_OP_ADD, ft,
      poly_uop2(ctx, POLY_OP_MUL, ft, qf,
        poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(-3.1414794921875)),
        poly_arg_none()), x, poly_arg_none());
  d = poly_uop2(ctx, POLY_OP_ADD, ft,
      poly_uop2(ctx, POLY_OP_MUL, ft, qf,
        poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(-0.00011315941810607910156)),
        poly_arg_none()), d, poly_arg_none());
  d = poly_uop2(ctx, POLY_OP_ADD, ft,
      poly_uop2(ctx, POLY_OP_MUL, ft, qf,
        poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(-1.9841872589410058936e-09)),
        poly_arg_none()), d, poly_arg_none());
  d = poly_uop2(ctx, POLY_OP_ADD, ft,
      poly_uop2(ctx, POLY_OP_MUL, ft, qf,
        poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(-1.2154201256553420762e-10)),
        poly_arg_none()), d, poly_arg_none());
  return d;
}

/* Cody-Waite _reduce_d for f64: qdh/q split with 4 PI constants. */
static PolyUOp *cody_waite_reduce_f64(PolyCtx *ctx, PolyDType ft,
                                       PolyUOp *x, PolyUOp *qdh, PolyUOp *qf) {
  /* PI_A..D from tinygrad sleef reference */
  static const double PI_A = 3.1415926218032836914;
  static const double PI_B = 3.1786509424591713469e-08;
  static const double PI_C = 1.2246467864107188502e-16;
  static const double PI_D = 1.2736634327021899816e-24;

  PolyUOp *pia = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(-PI_A));
  PolyUOp *pib = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(-PI_B));
  PolyUOp *pic = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(-PI_C));
  PolyUOp *pid = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(-PI_D));

  /* d = qdh * -PI_A + x */
  PolyUOp *d = poly_uop2(ctx, POLY_OP_ADD, ft,
      poly_uop2(ctx, POLY_OP_MUL, ft, qdh, pia, poly_arg_none()), x, poly_arg_none());
  /* d = q * -PI_A + d */
  d = poly_uop2(ctx, POLY_OP_ADD, ft,
      poly_uop2(ctx, POLY_OP_MUL, ft, qf, pia, poly_arg_none()), d, poly_arg_none());
  /* d = qdh * -PI_B + d */
  d = poly_uop2(ctx, POLY_OP_ADD, ft,
      poly_uop2(ctx, POLY_OP_MUL, ft, qdh, pib, poly_arg_none()), d, poly_arg_none());
  /* d = q * -PI_B + d */
  d = poly_uop2(ctx, POLY_OP_ADD, ft,
      poly_uop2(ctx, POLY_OP_MUL, ft, qf, pib, poly_arg_none()), d, poly_arg_none());
  /* d = qdh * -PI_C + d */
  d = poly_uop2(ctx, POLY_OP_ADD, ft,
      poly_uop2(ctx, POLY_OP_MUL, ft, qdh, pic, poly_arg_none()), d, poly_arg_none());
  /* d = q * -PI_C + d */
  d = poly_uop2(ctx, POLY_OP_ADD, ft,
      poly_uop2(ctx, POLY_OP_MUL, ft, qf, pic, poly_arg_none()), d, poly_arg_none());
  /* d = (qdh + q) * -PI_D + d */
  PolyUOp *qdh_plus_q = poly_uop2(ctx, POLY_OP_ADD, ft, qdh, qf, poly_arg_none());
  d = poly_uop2(ctx, POLY_OP_ADD, ft,
      poly_uop2(ctx, POLY_OP_MUL, ft, qdh_plus_q, pid, poly_arg_none()), d, poly_arg_none());
  return d;
}

/*
 * rule_decomp_sin — Port of tinygrad's xsin.
 *
 * SIN(d) → sign handling + Cody-Waite / Payne-Hanek reduction + polynomial.
 * Supports float32 and float64.
 *
 * f32: Cody-Waite (small) + Payne-Hanek (large), switchover at 30.0
 * f64: Cody-Waite with qdh precision split (small) + Payne-Hanek (large)
 */
static PolyUOp *rule_decomp_sin(PolyCtx *ctx, PolyUOp *root,
                                 const PolyBindings *b) {
  (void)b;
  PolyUOp *d = root->src[0];
  PolyDType sft = poly_dtype_scalar(root->dtype);
  if (!poly_dtype_is_float(sft) ||
      (sft.bitsize != 32 && sft.bitsize != 64))
    return NULL;

  PolyDType ft = root->dtype;  /* may be vec */
  int vc = ft.count;
  PolyDType it = (vc > 1) ? poly_dtype_vec(POLY_INT32, vc) : POLY_INT32;
  PolyDType ut32 = (vc > 1) ? poly_dtype_vec(POLY_UINT32, vc) : POLY_UINT32;
  PolyDType ut64 = (vc > 1) ? poly_dtype_vec(POLY_UINT64, vc) : POLY_UINT64;
  PolyDType bt = (vc > 1) ? poly_dtype_vec(POLY_BOOL, vc) : POLY_BOOL;
  bool is_f64 = (sft.bitsize == 64);

  /* Common constants */
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
  double m_1_pi = 0.318309886183790671537767526745028724;
  PolyUOp *f_m_1_pi  = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(m_1_pi));
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
  PolyUOp *u_mask    = poly_uop0(ctx, POLY_OP_CONST, ut64, poly_arg_int(0x3fffffffffffffffULL));
  PolyUOp *u_m1      = poly_uop0(ctx, POLY_OP_CONST, ut32, poly_arg_int(0x807fffffU));
  PolyUOp *u_m2      = poly_uop0(ctx, POLY_OP_CONST, ut32, poly_arg_int(0x3f000000U));

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

  /* ── Cody-Waite reduction (small branch) ─────────────────────────── */
  PolyUOp *q_small;
  PolyUOp *r_small;

  if (is_f64) {
    /* f64: qdh = (x_abs * (m_1_pi / 2^24)).cast(int64).cast(f64) * 2^24 */
    PolyUOp *f_m1pi_div2p24 = poly_uop0(ctx, POLY_OP_CONST, ft,
        poly_arg_float(m_1_pi / 16777216.0));  /* m_1_pi / 2^24 */
    PolyUOp *f_2p24 = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(16777216.0));
    PolyDType it64 = POLY_INT64;
    PolyUOp *qdh_raw = poly_uop2(ctx, POLY_OP_MUL, ft, x_abs, f_m1pi_div2p24, poly_arg_none());
    PolyUOp *qdh_int = poly_uop1(ctx, POLY_OP_CAST, it64, qdh_raw, poly_arg_none());
    PolyUOp *qdh = poly_uop2(ctx, POLY_OP_MUL, ft,
        poly_uop1(ctx, POLY_OP_CAST, ft, qdh_int, poly_arg_none()),
        f_2p24, poly_arg_none());

    /* quadrant = rintk(x_abs * m_1_pi - qdh) */
    PolyUOp *qf_raw = poly_uop2(ctx, POLY_OP_SUB, ft,
        poly_uop2(ctx, POLY_OP_MUL, ft, x_abs, f_m_1_pi, poly_arg_none()),
        qdh, poly_arg_none());
    q_small = xd_rintk(ctx, ft, it, qf_raw);
    PolyUOp *qf = poly_uop1(ctx, POLY_OP_CAST, ft, q_small, poly_arg_none());

    r_small = cody_waite_reduce_f64(ctx, ft, x_abs, qdh, qf);
  } else {
    /* f32: simple rintk(x_abs * m_1_pi) */
    PolyUOp *qf_raw = poly_uop2(ctx, POLY_OP_MUL, ft, x_abs, f_m_1_pi, poly_arg_none());
    q_small = xd_rintk(ctx, ft, it, qf_raw);
    PolyUOp *qf = poly_uop1(ctx, POLY_OP_CAST, ft, q_small, poly_arg_none());

    r_small = cody_waite_reduce_f32(ctx, ft, x_abs, qf);
  }

  /* ── Payne-Hanek reduction (large branch, same for f32/f64) ──────── */
  /* frexp via bit manipulation — always uses f32 intermediates for Payne-Hanek */
  PolyUOp *x_abs_f32;
  if (is_f64) {
    /* Demote to f32 for Payne-Hanek (tinygrad uses intermediate_dtype=d.dtype for non-f16) */
    /* Actually tinygrad uses d.dtype as intermediate for f64 too. But the two_over_pi_f table
     * is uint32-based. Let's keep the same Payne-Hanek as f32 since it operates on the
     * frexp decomposition which is dtype-independent for the bit table lookup. */
    x_abs_f32 = x_abs;  /* We'll use the same Payne-Hanek for both */
  } else {
    x_abs_f32 = x_abs;
  }

  PolyUOp *bits = poly_uop1(ctx, POLY_OP_BITCAST, ut32,
      is_f64 ? poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, x_abs, poly_arg_none()) : x_abs,
      poly_arg_none());
  PolyUOp *exp_u32 = poly_uop2(ctx, POLY_OP_AND, ut32,
                               poly_uop2(ctx, POLY_OP_SHR, ut32, bits, i_23, poly_arg_none()),
                               i_255_u32, poly_arg_none());
  PolyUOp *f_bits = poly_uop2(ctx, POLY_OP_OR, ut32,
                              poly_uop2(ctx, POLY_OP_AND, ut32, bits, u_m1, poly_arg_none()),
                              u_m2, poly_arg_none());
  PolyUOp *f_frexp = poly_uop1(ctx, POLY_OP_BITCAST, POLY_FLOAT32, f_bits, poly_arg_none());
  PolyUOp *e_i = poly_uop2(ctx, POLY_OP_SUB, it,
                           poly_uop1(ctx, POLY_OP_CAST, it, exp_u32, poly_arg_none()),
                           i_126, poly_arg_none());
  PolyUOp *ia = poly_uop1(ctx, POLY_OP_CAST, ut64,
                          poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, f_frexp,
                            poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(4294967296.0f)),
                            poly_arg_none()),
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
  /* Payne-Hanek word assembly: shift and OR adjacent 2/pi words.
   * When e_lo == 0, offset == 32 and (uint32 >> 32) is UB in C.
   * Fix: do shifts in uint64, then truncate to uint32. */
  PolyUOp *e_lo_64 = poly_uop1(ctx, POLY_OP_CAST, ut64, e_lo, poly_arg_none());
  PolyUOp *offset_64 = poly_uop1(ctx, POLY_OP_CAST, ut64, offset, poly_arg_none());
  PolyUOp *hi = poly_uop1(ctx, POLY_OP_CAST, ut32,
                  poly_uop2(ctx, POLY_OP_OR, ut64,
                    poly_uop2(ctx, POLY_OP_SHL, ut64,
                      poly_uop1(ctx, POLY_OP_CAST, ut64, a0, poly_arg_none()), e_lo_64, poly_arg_none()),
                    poly_uop2(ctx, POLY_OP_SHR, ut64,
                      poly_uop1(ctx, POLY_OP_CAST, ut64, a1, poly_arg_none()), offset_64, poly_arg_none()),
                    poly_arg_none()),
                  poly_arg_none());
  PolyUOp *mi = poly_uop1(ctx, POLY_OP_CAST, ut32,
                  poly_uop2(ctx, POLY_OP_OR, ut64,
                    poly_uop2(ctx, POLY_OP_SHL, ut64,
                      poly_uop1(ctx, POLY_OP_CAST, ut64, a1, poly_arg_none()), e_lo_64, poly_arg_none()),
                    poly_uop2(ctx, POLY_OP_SHR, ut64,
                      poly_uop1(ctx, POLY_OP_CAST, ut64, a2, poly_arg_none()), offset_64, poly_arg_none()),
                    poly_arg_none()),
                  poly_arg_none());
  PolyUOp *lo = poly_uop1(ctx, POLY_OP_CAST, ut32,
                  poly_uop2(ctx, POLY_OP_OR, ut64,
                    poly_uop2(ctx, POLY_OP_SHL, ut64,
                      poly_uop1(ctx, POLY_OP_CAST, ut64, a2, poly_arg_none()), e_lo_64, poly_arg_none()),
                    poly_uop2(ctx, POLY_OP_SHR, ut64,
                      poly_uop1(ctx, POLY_OP_CAST, ut64, a3, poly_arg_none()), offset_64, poly_arg_none()),
                    poly_arg_none()),
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
  PolyUOp *f_frexp_ft = is_f64
      ? poly_uop1(ctx, POLY_OP_CAST, ft, f_frexp, poly_arg_none())
      : f_frexp;
  PolyUOp *f_lt_half = poly_uop2(ctx, POLY_OP_CMPLT, bt, f_frexp_ft, f_half, poly_arg_none());
  PolyUOp *r_ph = poly_uop3(ctx, POLY_OP_WHERE, ft, f_lt_half, r_ph_base,
                            poly_uop2(ctx, POLY_OP_SUB, ft, r_ph_base, f_pi_2, poly_arg_none()),
                            poly_arg_none());
  q_ph = poly_uop3(ctx, POLY_OP_WHERE, it, f_lt_half, q_ph,
                   poly_uop2(ctx, POLY_OP_ADD, it, q_ph, i_one, poly_arg_none()),
                   poly_arg_none());

  /* ── sin_poly_small / sin_poly_large, split at switch_over ───────── */
  PolyUOp *q_small_odd = poly_uop2(ctx, POLY_OP_CMPNE, bt,
                                   poly_uop2(ctx, POLY_OP_AND, it, q_small, i_one, poly_arg_none()),
                                   i_zero, poly_arg_none());
  PolyUOp *small_sign = poly_uop3(ctx, POLY_OP_WHERE, ft, q_small_odd, f_neg_one, f_one, poly_arg_none());
  PolyUOp *result_small = poly_uop2(ctx, POLY_OP_MUL, ft, sin_poly(ctx, r_small), small_sign, poly_arg_none());

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
  PolyUOp *result_large = poly_uop2(ctx, POLY_OP_MUL, ft, sin_poly(ctx, large_arg), large_sign, poly_arg_none());

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

/* ── BF16 non-native type rewrites ────────────────────────────────────
 * Mirrors tinygrad's create_non_native_float_pats() + pm_manual_bf16_cast.
 * BF16 is stored as unsigned short on most targets (HIP, OpenCL, CPU).
 * ALU ops must be promoted to float32, and CAST bf16<->f32 uses bitwise ops.
 * ──────────────────────────────────────────────────────────────────────── */

static bool is_bf16(PolyDType dt) {
  PolyDType s = poly_dtype_scalar(dt);
  return s.priority == POLY_BFLOAT16.priority && s.bitsize == 16;
}

/* Rule: ALU(bf16, ...) -> CAST(ALU(CAST(src0, f32), ..., f32), bf16)
 * Applies to unary and binary float ALU ops with bf16 output. */
static PolyUOp *rule_bf16_alu_promote(PolyCtx *ctx, PolyUOp *root,
                                       const PolyBindings *b) {
  (void)b;
  if (!is_bf16(root->dtype)) return NULL;

  /* Cast all sources from bf16 to f32, preserving non-bf16 sources */
  PolyUOp *new_src[4];
  for (int i = 0; i < root->n_src && i < 4; i++) {
    if (is_bf16(root->src[i]->dtype))
      new_src[i] = poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, root->src[i], poly_arg_none());
    else
      new_src[i] = root->src[i];
  }

  /* Perform ALU in f32 */
  PolyUOp *f32_result = poly_uop(ctx, root->op, POLY_FLOAT32,
                                   new_src, root->n_src, root->arg);

  /* Cast result back to bf16 */
  return poly_uop1(ctx, POLY_OP_CAST, root->dtype, f32_result, poly_arg_none());
}

/* Rule: CMP(bf16, bf16) -> CMP(CAST(x, f32), CAST(y, f32))
 * Comparison ops with bf16 operands; result is bool, no cast back. */
static PolyUOp *rule_bf16_cmp_promote(PolyCtx *ctx, PolyUOp *root,
                                       const PolyBindings *b) {
  (void)b;
  if (root->n_src < 2) return NULL;
  if (!is_bf16(root->src[0]->dtype) && !is_bf16(root->src[1]->dtype)) return NULL;

  PolyUOp *s0 = is_bf16(root->src[0]->dtype)
    ? poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, root->src[0], poly_arg_none())
    : root->src[0];
  PolyUOp *s1 = is_bf16(root->src[1]->dtype)
    ? poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, root->src[1], poly_arg_none())
    : root->src[1];

  return poly_uop2(ctx, root->op, root->dtype, s0, s1, root->arg);
}

/* Rule: WHERE(cond, x:bf16, y:bf16) -> CAST(WHERE(cond, CAST(x,f32), CAST(y,f32)), bf16) */
static PolyUOp *rule_bf16_where_promote(PolyCtx *ctx, PolyUOp *root,
                                         const PolyBindings *b) {
  (void)b;
  if (!is_bf16(root->dtype)) return NULL;

  PolyUOp *cond = root->src[0];
  PolyUOp *x = poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, root->src[1], poly_arg_none());
  PolyUOp *y = poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, root->src[2], poly_arg_none());
  PolyUOp *where_f32 = poly_uop3(ctx, POLY_OP_WHERE, POLY_FLOAT32, cond, x, y, poly_arg_none());
  return poly_uop1(ctx, POLY_OP_CAST, root->dtype, where_f32, poly_arg_none());
}

/* Rule: CAST(x:bf16, f32) -> bitcast(CAST(bitcast(x, u16), u32) << 16, f32)
 * Tinygrad pm_manual_bf16_cast: (x.bitcast(ushort).cast(uint)<<16).bitcast(float) */
static PolyUOp *rule_bf16_to_f32_cast(PolyCtx *ctx, PolyUOp *root,
                                       const PolyBindings *b) {
  (void)b;
  if (!poly_dtype_eq(root->dtype, POLY_FLOAT32)) return NULL;
  if (root->n_src < 1 || !is_bf16(root->src[0]->dtype)) return NULL;

  PolyUOp *x = root->src[0];
  /* bitcast bf16 -> u16 */
  PolyUOp *u16 = poly_uop1(ctx, POLY_OP_BITCAST, POLY_UINT16, x, poly_arg_none());
  /* cast u16 -> u32 (zero extend) */
  PolyUOp *u32 = poly_uop1(ctx, POLY_OP_CAST, POLY_UINT32, u16, poly_arg_none());
  /* shift left 16 */
  PolyUOp *c16 = poly_uop0(ctx, POLY_OP_CONST, POLY_UINT32, poly_arg_int(16));
  PolyUOp *shifted = poly_uop2(ctx, POLY_OP_SHL, POLY_UINT32, u32, c16, poly_arg_none());
  /* bitcast u32 -> f32 */
  return poly_uop1(ctx, POLY_OP_BITCAST, POLY_FLOAT32, shifted, poly_arg_none());
}

/* Rule: CAST(x:f32, bf16) -> bitcast((round_to_bf16(bitcast(x, u32)) >> 16) as u16, bf16)
 * Tinygrad cast_float_to_bf16: handles rounding and special values. */
static PolyUOp *rule_f32_to_bf16_cast(PolyCtx *ctx, PolyUOp *root,
                                       const PolyBindings *b) {
  (void)b;
  if (!is_bf16(root->dtype)) return NULL;
  if (root->n_src < 1 || !poly_dtype_eq(root->src[0]->dtype, POLY_FLOAT32)) return NULL;

  PolyUOp *x = root->src[0];
  /* bitcast f32 -> u32 */
  PolyUOp *u = poly_uop1(ctx, POLY_OP_BITCAST, POLY_UINT32, x, poly_arg_none());

  /* Constants */
  PolyUOp *c0x7f800000 = poly_uop0(ctx, POLY_OP_CONST, POLY_UINT32, poly_arg_int(0x7f800000));
  PolyUOp *c0xffff     = poly_uop0(ctx, POLY_OP_CONST, POLY_UINT32, poly_arg_int(0xffff));
  PolyUOp *c0x10000    = poly_uop0(ctx, POLY_OP_CONST, POLY_UINT32, poly_arg_int(0x10000));
  PolyUOp *c0x7fff     = poly_uop0(ctx, POLY_OP_CONST, POLY_UINT32, poly_arg_int(0x7fff));
  PolyUOp *c16         = poly_uop0(ctx, POLY_OP_CONST, POLY_UINT32, poly_arg_int(16));
  PolyUOp *c1          = poly_uop0(ctx, POLY_OP_CONST, POLY_UINT32, poly_arg_int(1));
  PolyUOp *c0          = poly_uop0(ctx, POLY_OP_CONST, POLY_UINT32, poly_arg_int(0));

  /* neg_u = -u  (actually NEG for uint = two's complement negate) */
  PolyUOp *neg_u = poly_uop1(ctx, POLY_OP_NEG, POLY_UINT32, u, poly_arg_none());

  /* exponent check: (-u & 0x7f800000) != 0  (not zero/denorm) */
  PolyUOp *exp_masked = poly_uop2(ctx, POLY_OP_AND, POLY_UINT32, neg_u, c0x7f800000, poly_arg_none());
  PolyUOp *exp_nonzero = poly_uop2(ctx, POLY_OP_CMPNE, POLY_BOOL, exp_masked, c0, poly_arg_none());

  /* round: u + ((u >> 16) & 1) + 0x7fff */
  PolyUOp *u_shr16 = poly_uop2(ctx, POLY_OP_SHR, POLY_UINT32, u, c16, poly_arg_none());
  PolyUOp *lsb = poly_uop2(ctx, POLY_OP_AND, POLY_UINT32, u_shr16, c1, poly_arg_none());
  PolyUOp *rounded = poly_uop2(ctx, POLY_OP_ADD, POLY_UINT32, u, lsb, poly_arg_none());
  rounded = poly_uop2(ctx, POLY_OP_ADD, POLY_UINT32, rounded, c0x7fff, poly_arg_none());

  /* mantissa check: (u & 0xffff) != 0 */
  PolyUOp *mant_masked = poly_uop2(ctx, POLY_OP_AND, POLY_UINT32, u, c0xffff, poly_arg_none());
  PolyUOp *mant_nonzero = poly_uop2(ctx, POLY_OP_CMPNE, POLY_BOOL, mant_masked, c0, poly_arg_none());

  /* denorm/zero path: if mantissa != 0, set sticky bit; else keep u */
  PolyUOp *sticky = poly_uop2(ctx, POLY_OP_OR, POLY_UINT32, u, c0x10000, poly_arg_none());
  PolyUOp *denorm_result = poly_uop3(ctx, POLY_OP_WHERE, POLY_UINT32, mant_nonzero, sticky, u, poly_arg_none());

  /* final: if exponent nonzero -> rounded, else -> denorm_result */
  PolyUOp *final_u32 = poly_uop3(ctx, POLY_OP_WHERE, POLY_UINT32, exp_nonzero, rounded, denorm_result, poly_arg_none());

  /* shift right 16 -> u16 -> bitcast bf16 */
  PolyUOp *shifted = poly_uop2(ctx, POLY_OP_SHR, POLY_UINT32, final_u32, c16, poly_arg_none());
  PolyUOp *u16val = poly_uop1(ctx, POLY_OP_CAST, POLY_UINT16, shifted, poly_arg_none());
  return poly_uop1(ctx, POLY_OP_BITCAST, POLY_BFLOAT16, u16val, poly_arg_none());
}

/* Rule: CAST(x:bf16, non-f32) or CAST(x:non-f32, bf16) -> go through f32 */
static PolyUOp *rule_bf16_cast_via_f32(PolyCtx *ctx, PolyUOp *root,
                                        const PolyBindings *b) {
  (void)b;
  if (root->n_src < 1) return NULL;
  PolyDType src_dt = root->src[0]->dtype;
  PolyDType dst_dt = root->dtype;

  /* bf16 -> non-f32: go through f32 */
  if (is_bf16(src_dt) && !poly_dtype_eq(dst_dt, POLY_FLOAT32) && !is_bf16(dst_dt)) {
    PolyUOp *f32 = poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, root->src[0], poly_arg_none());
    return poly_uop1(ctx, POLY_OP_CAST, dst_dt, f32, poly_arg_none());
  }
  /* non-f32 -> bf16: go through f32 */
  if (is_bf16(dst_dt) && !poly_dtype_eq(src_dt, POLY_FLOAT32) && !is_bf16(src_dt)) {
    PolyUOp *f32 = poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, root->src[0], poly_arg_none());
    return poly_uop1(ctx, POLY_OP_CAST, dst_dt, f32, poly_arg_none());
  }
  return NULL;
}

/* Rule: CONST(bf16, val) -> cast_float_to_bf16(CONST(f32, val))
 * Tinygrad HIP extra_matcher: bf16 consts rendered as bit pattern via rounding. */
static PolyUOp *rule_bf16_const(PolyCtx *ctx, PolyUOp *root,
                                 const PolyBindings *b) {
  (void)b;
  if (!is_bf16(root->dtype)) return NULL;
  /* Create f32 CONST with same value, then apply f32->bf16 rounding */
  PolyUOp *f32_const = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(root->arg.f));
  return poly_uop1(ctx, POLY_OP_CAST, POLY_BFLOAT16, f32_const, poly_arg_none());
}

static PolyPatternMatcher *g_pm_bf16_non_native = NULL;

PolyPatternMatcher *poly_pm_bf16_non_native(void) {
  if (g_pm_bf16_non_native) return g_pm_bf16_non_native;

  /* ALU ops that need bf16->f32 promotion */
  PolyOpSet alu_set = {{0,0}};
  PolyOps alu_ops[] = {
    POLY_OP_ADD, POLY_OP_MUL, POLY_OP_SUB, POLY_OP_FDIV, POLY_OP_NEG,
    POLY_OP_SQRT, POLY_OP_RECIPROCAL, POLY_OP_EXP2, POLY_OP_LOG2,
    POLY_OP_SIN, POLY_OP_MAX, POLY_OP_TRUNC, POLY_OP_POW, POLY_OP_MULACC
  };
  for (int i = 0; i < (int)(sizeof(alu_ops)/sizeof(alu_ops[0])); i++)
    alu_set = poly_opset_add(alu_set, alu_ops[i]);

  /* Comparison ops that need bf16 operand promotion */
  PolyOpSet cmp_set = {{0,0}};
  cmp_set = poly_opset_add(cmp_set, POLY_OP_CMPLT);
  cmp_set = poly_opset_add(cmp_set, POLY_OP_CMPNE);
  cmp_set = poly_opset_add(cmp_set, POLY_OP_CMPEQ);

  PolyOpSet where_set = poly_opset_add((PolyOpSet){{0,0}}, POLY_OP_WHERE);
  PolyOpSet cast_set = poly_opset_add((PolyOpSet){{0,0}}, POLY_OP_CAST);
  PolyOpSet const_set = poly_opset_add((PolyOpSet){{0,0}}, POLY_OP_CONST);

  PolyRule rules[] = {
    /* 0. bf16 CONST: rewrite to f32 const + cast (renders as bit pattern) */
    { poly_pat_ops(const_set, NULL, 0, NULL), rule_bf16_const },
    /* 1. ALU ops: promote bf16 -> f32 -> bf16 */
    { poly_pat_ops(alu_set, NULL, 0, NULL), rule_bf16_alu_promote },
    /* 2. CMP ops: promote bf16 operands to f32 */
    { poly_pat_ops(cmp_set, NULL, 0, NULL), rule_bf16_cmp_promote },
    /* 3. WHERE: promote bf16 branches to f32 */
    { poly_pat_ops(where_set, NULL, 0, NULL), rule_bf16_where_promote },
    /* 4. CAST bf16->f32: bitwise expansion */
    { poly_pat_ops(cast_set, NULL, 0, NULL), rule_bf16_to_f32_cast },
    /* 5. CAST f32->bf16: bitwise with rounding */
    { poly_pat_ops(cast_set, NULL, 0, NULL), rule_f32_to_bf16_cast },
    /* 6. CAST bf16<->other: go through f32 */
    { poly_pat_ops(cast_set, NULL, 0, NULL), rule_bf16_cast_via_f32 },
  };

  g_pm_bf16_non_native = poly_pm_new(rules, (int)(sizeof(rules)/sizeof(rules[0])));
  return g_pm_bf16_non_native;
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
      new_srcs[n_new_srcs++] = poly_uop(ctx, POLY_OP_VCAT,
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

/* ── load_store_folding: PTRCAT pipeline (tinygrad devectorizer.py:63-136) ── */

/* expand_index (tinygrad devectorizer.py:63-66):
 * INDEX(VECTORIZE(buf,...), vec_idx) → VECTORIZE(INDEX(buf, GEP(vec,0)), ..., INDEX(buf, GEP(vec,N-1)))
 * Scatters a vectorized INDEX into per-element scalar INDEXes. */
static PolyUOp *rule_expand_index(PolyCtx *ctx, PolyUOp *idx, const PolyBindings *b) {
  (void)b;
  if (!idx || idx->op != POLY_OP_INDEX || idx->n_src < 2) return NULL;
  PolyUOp *buf_vec = idx->src[0];
  if (!buf_vec || buf_vec->op != POLY_OP_VECTORIZE || buf_vec->n_src <= 1) return NULL;
  /* All VECTORIZE sources must be Defines (PARAM) or AFTER */
  PolyUOp *buf = buf_vec->src[0];
  if (!buf) return NULL;
  if (buf->op != POLY_OP_PARAM && buf->op != POLY_OP_DEFINE_LOCAL &&
      buf->op != POLY_OP_DEFINE_REG && buf->op != POLY_OP_AFTER) return NULL;
  for (int i = 1; i < buf_vec->n_src; i++)
    if (buf_vec->src[i] != buf) return NULL;  /* all same buf */

  PolyUOp *vec = idx->src[1];
  int cnt = buf_vec->n_src;
  if (cnt > 128) return NULL;

  PolyUOp *elems[128];
  for (int i = 0; i < cnt; i++) {
    PolyUOp *gi = make_gep_lane(ctx, vec, i);
    /* Build INDEX(buf, gi) with ptr output (same as buf->dtype) */
    PolyUOp *idx_srcs[64];
    int ns = 0;
    idx_srcs[ns++] = buf;
    idx_srcs[ns++] = gi;
    for (int j = 2; j < idx->n_src && ns < 64; j++) idx_srcs[ns++] = idx->src[j];
    elems[i] = poly_uop(ctx, POLY_OP_INDEX, buf->dtype, idx_srcs, ns, idx->arg);
  }
  return poly_uop(ctx, POLY_OP_VECTORIZE, buf->dtype, elems, cnt, poly_arg_none());
}

/* fold_expanded_index (tinygrad devectorizer.py:68-104):
 * VECTORIZE(INDEX(buf,off0), INDEX(buf,off1), ...) → PTRCAT(...).gep(remap)
 * Groups contiguous offsets into vector pointer CASTs, wraps in PTRCAT. */
static PolyUOp *rule_fold_expanded_index(PolyCtx *ctx, PolyUOp *midx, const PolyBindings *b) {
  (void)b;
  if (!midx || midx->op != POLY_OP_VECTORIZE || midx->n_src <= 1) return NULL;
  /* All sources must be INDEX ops */
  for (int i = 0; i < midx->n_src; i++)
    if (!midx->src[i] || midx->src[i]->op != POLY_OP_INDEX) return NULL;
  /* All INDEX ops must reference the same buffer */
  PolyUOp *buf = midx->src[0]->src[0];
  if (!buf) return NULL;
  for (int i = 1; i < midx->n_src; i++)
    if (midx->src[i]->src[0] != buf) return NULL;
  /* All INDEX outputs must be pointer types */
  for (int i = 0; i < midx->n_src; i++)
    if (!midx->src[i]->dtype.is_ptr) return NULL;

  int n = midx->n_src;

  /* Extract offsets: decompose each INDEX's idx expr into (root_src, const_offset).
   * Polygrad simplified model: INDEX.src[1] is the offset directly (no get_idx/get_valid). */
  typedef struct { PolyUOp *root; int64_t offset; int orig_idx; } OffsetEntry;
  OffsetEntry entries[128];
  for (int i = 0; i < n; i++) {
    PolyUOp *idx_expr = midx->src[i]->src[1];
    entries[i].orig_idx = i;
    /* Decompose idx_expr into root + const offset */
    LaneAffineExpr ae = {.n_terms = 0, .cst = 0, .ok = true};
    collect_add_terms(ctx, idx_expr, &ae, false);
    if (!ae.ok) { entries[i].root = idx_expr; entries[i].offset = 0; continue; }
    qsort(ae.terms, (size_t)ae.n_terms, sizeof(ae.terms[0]), uop_ptr_cmp);
    entries[i].offset = ae.cst;
    entries[i].root = build_add_expr(ctx, poly_dtype_scalar(idx_expr->dtype), ae.terms, ae.n_terms, 0);
  }

  /* Group entries by root_src, then find contiguous offset sequences */
  /* Simple approach: group entries with same root pointer, sort by offset, find contiguous runs */
  PolyUOp *ret_srcs[128];
  int n_ret = 0;
  int64_t idxs[128]; /* remap: original lane → position in PTRCAT output */
  for (int i = 0; i < n; i++) idxs[i] = -1;
  int global_offset = 0;

  bool used[128];
  for (int i = 0; i < n; i++) used[i] = false;

  for (int i = 0; i < n; i++) {
    if (used[i]) continue;
    /* Collect all entries with same root */
    int group[128];
    int64_t group_offsets[128];
    int ng = 0;
    for (int j = i; j < n; j++) {
      if (used[j]) continue;
      if (entries[j].root == entries[i].root) {
        group[ng] = j;
        group_offsets[ng] = entries[j].offset;
        ng++;
      }
    }
    /* Sort group by offset */
    for (int a = 0; a < ng - 1; a++)
      for (int bb = a + 1; bb < ng; bb++)
        if (group_offsets[a] > group_offsets[bb]) {
          int64_t to = group_offsets[a]; group_offsets[a] = group_offsets[bb]; group_offsets[bb] = to;
          int ti = group[a]; group[a] = group[bb]; group[bb] = ti;
        }
    /* Find contiguous runs within the group */
    int run_start = 0;
    while (run_start < ng) {
      int run_end = run_start + 1;
      while (run_end < ng && group_offsets[run_end] == group_offsets[run_end-1] + 1) run_end++;
      int run_len = run_end - run_start;
      /* Use the first INDEX in the run as the base pointer */
      PolyUOp *lidx = midx->src[group[run_start]];
      if (run_len > 1) {
        /* CAST to vec pointer: CAST(INDEX(buf, base), vec_N_ptr) */
        PolyDType vec_ptr = buf->dtype;
        vec_ptr.vcount = (uint16_t)run_len;
        lidx = poly_uop1(ctx, POLY_OP_CAST, vec_ptr, lidx, poly_arg_none());
      }
      /* Map original lanes to PTRCAT positions */
      for (int k = run_start; k < run_end; k++) {
        idxs[entries[group[k]].orig_idx] = global_offset + (k - run_start);
        used[group[k]] = true;
      }
      ret_srcs[n_ret++] = lidx;
      global_offset += run_len;
      run_start = run_end;
    }
  }

  /* Verify all lanes mapped */
  for (int i = 0; i < n; i++) if (idxs[i] < 0) return NULL;

  /* Build PTRCAT */
  PolyDType ptrcat_dt = buf->dtype;
  ptrcat_dt.vcount = (uint16_t)global_offset;
  PolyUOp *ptrcat = poly_uop(ctx, POLY_OP_PTRCAT, ptrcat_dt, ret_srcs, n_ret, poly_arg_none());

  /* Apply GEP remap */
  return poly_uop1(ctx, POLY_OP_GEP, midx->dtype, ptrcat, poly_arg_int_tuple_local(idxs, n));
}

/* GEP after LOAD (tinygrad devectorizer.py:127-128):
 * LOAD(GEP(ptr, arg)) → LOAD(ptr, wider_dtype).gep(arg)
 * Pushes GEP through LOAD so the LOAD reads the full vector. */
static PolyUOp *rule_gep_after_load(PolyCtx *ctx, PolyUOp *ld, const PolyBindings *b) {
  (void)b;
  if (!ld || ld->op != POLY_OP_LOAD || ld->n_src < 1) return NULL;
  PolyUOp *gep = ld->src[0];
  if (!gep || gep->op != POLY_OP_GEP || gep->n_src < 1) return NULL;
  /* Build wider LOAD: dtype = scalar.vec(gep source vcount) */
  int src_count = gep->src[0]->dtype.is_ptr ? gep->src[0]->dtype.vcount : gep->src[0]->dtype.count;
  if (src_count <= 1) return NULL;
  PolyDType wider_dt = poly_dtype_vec(poly_dtype_scalar(ld->dtype), src_count);
  PolyUOp *ld_srcs[64];
  int ns = 0;
  ld_srcs[ns++] = gep->src[0];
  for (int i = 1; i < ld->n_src && ns < 64; i++) ld_srcs[ns++] = ld->src[i];
  PolyUOp *wider_load = poly_uop(ctx, POLY_OP_LOAD, wider_dt, ld_srcs, ns, ld->arg);
  return poly_uop1(ctx, POLY_OP_GEP, ld->dtype, wider_load, gep->arg);
}

/* GEP on STORE data (tinygrad devectorizer.py:115-121):
 * STORE(GEP(ptr, arg), data) → STORE(ptr.src[0], data.gep(inverted_arg)) */
static PolyUOp *rule_gep_on_store(PolyCtx *ctx, PolyUOp *sto, const PolyBindings *b) {
  (void)b;
  if (!sto || sto->op != POLY_OP_STORE || sto->n_src < 2) return NULL;
  PolyUOp *gep = sto->src[0];
  if (!gep || gep->op != POLY_OP_GEP || gep->n_src < 1) return NULL;
  if (gep->arg.kind != POLY_ARG_INT_TUPLE || gep->arg.int_tuple.n <= 0) return NULL;
  /* Invert the GEP permutation: argsort */
  int gn = gep->arg.int_tuple.n;
  int64_t new_arg[128];
  if (gn > 128) return NULL;
  /* Build inverse: for each position in gep->arg, map value→index */
  for (int i = 0; i < gn; i++) {
    int64_t v = gep->arg.int_tuple.vals[i];
    if (v < 0 || v >= gn) return NULL;
    new_arg[v] = i;
  }
  PolyUOp *st_data = sto->src[1];
  PolyUOp *reordered_data = poly_uop1(ctx, POLY_OP_GEP, st_data->dtype, st_data,
                                        poly_arg_int_tuple_local(new_arg, gn));
  PolyUOp *st_srcs[64];
  int ns = 0;
  st_srcs[ns++] = gep->src[0];
  st_srcs[ns++] = reordered_data;
  for (int i = 2; i < sto->n_src && ns < 64; i++) st_srcs[ns++] = sto->src[i];
  return poly_uop(ctx, POLY_OP_STORE, sto->dtype, st_srcs, ns, sto->arg);
}

/* PTRCAT after LOAD (tinygrad devectorizer.py:131-133):
 * LOAD(PTRCAT(ptr0, ptr1, ...)) → VCAT(LOAD(ptr0), LOAD(ptr1), ...)
 * Each LOAD reads the vector width of its pointer source. */
static PolyUOp *rule_ptrcat_after_load(PolyCtx *ctx, PolyUOp *ld, const PolyBindings *b) {
  (void)b;
  if (!ld || ld->op != POLY_OP_LOAD || ld->n_src < 1) return NULL;
  PolyUOp *cat = ld->src[0];
  if (!cat || cat->op != POLY_OP_PTRCAT || cat->n_src <= 0) return NULL;
  int total_count = 0;
  PolyUOp *loads[128];
  int nl = 0;
  PolyDType sdt = poly_dtype_scalar(ld->dtype);
  for (int i = 0; i < cat->n_src && nl < 128; i++) {
    PolyUOp *ptr = cat->src[i];
    int ptr_count = ptr->dtype.is_ptr ? ptr->dtype.vcount : 1;
    if (ptr_count <= 0) ptr_count = 1;
    PolyDType ld_dt = (ptr_count > 1) ? poly_dtype_vec(sdt, ptr_count) : sdt;
    PolyUOp *ld_srcs[64];
    int ns = 0;
    ld_srcs[ns++] = ptr;
    for (int j = 1; j < ld->n_src && ns < 64; j++) ld_srcs[ns++] = ld->src[j];
    loads[nl++] = poly_uop(ctx, POLY_OP_LOAD, ld_dt, ld_srcs, ns, ld->arg);
    total_count += ptr_count;
  }
  PolyDType cat_dt = poly_dtype_vec(sdt, total_count);
  return poly_uop(ctx, POLY_OP_VCAT, cat_dt, loads, nl, poly_arg_none());
}

/* PTRCAT after STORE (tinygrad devectorizer.py:106-113):
 * STORE(PTRCAT(ptr0, ptr1, ...), data) → GROUP(STORE(ptr0, GEP(data,0..n0)), ...) */
static PolyUOp *rule_ptrcat_after_store(PolyCtx *ctx, PolyUOp *sto, const PolyBindings *b) {
  (void)b;
  if (!sto || sto->op != POLY_OP_STORE || sto->n_src < 2) return NULL;
  PolyUOp *cat = sto->src[0];
  if (!cat || cat->op != POLY_OP_PTRCAT || cat->n_src <= 0) return NULL;
  PolyUOp *data = sto->src[1];
  int offset = 0;
  PolyUOp *stores[128];
  int ns_out = 0;
  for (int i = 0; i < cat->n_src && ns_out < 128; i++) {
    PolyUOp *ptr = cat->src[i];
    int ptr_count = ptr->dtype.is_ptr ? ptr->dtype.vcount : 1;
    if (ptr_count <= 0) ptr_count = 1;
    /* GEP to extract this slice of data */
    int64_t gep_args[128];
    for (int j = 0; j < ptr_count; j++) gep_args[j] = offset + j;
    PolyDType slice_dt = (ptr_count > 1)
      ? poly_dtype_vec(poly_dtype_scalar(data->dtype), ptr_count)
      : poly_dtype_scalar(data->dtype);
    PolyUOp *slice = poly_uop1(ctx, POLY_OP_GEP, slice_dt, data,
                                poly_arg_int_tuple_local(gep_args, ptr_count));
    PolyUOp *st_srcs[64];
    int ns = 0;
    st_srcs[ns++] = ptr;
    st_srcs[ns++] = slice;
    for (int j = 2; j < sto->n_src && ns < 64; j++) st_srcs[ns++] = sto->src[j];
    stores[ns_out++] = poly_uop(ctx, POLY_OP_STORE, sto->dtype, st_srcs, ns, sto->arg);
    offset += ptr_count;
  }
  return poly_uop(ctx, POLY_OP_GROUP, POLY_VOID, stores, ns_out, poly_arg_none());
}

/* split_load_store (tinygrad devectorizer.py:140-184):
 * LOAD/STORE(CAST(INDEX, vec_ptr)) → split to hardware-supported widths.
 * Matches correct_load_store pattern: LOAD/STORE with CAST(INDEX) source.
 * For CPU with supports_float4: fold_lengths = [4, 2, 1]. */
static PolyUOp *rule_split_load_store(PolyCtx *ctx, PolyUOp *ls, const PolyBindings *b) {
  (void)b;
  if (!ls) return NULL;
  bool is_load = (ls->op == POLY_OP_LOAD);
  bool is_store = (ls->op == POLY_OP_STORE);
  if (!is_load && !is_store) return NULL;
  if (ls->n_src < 1) return NULL;
  /* Match CAST(INDEX) source — the vec pointer form */
  PolyUOp *cast = ls->src[0];
  if (!cast || cast->op != POLY_OP_CAST) return NULL;
  if (cast->n_src < 1 || !cast->src[0] || cast->src[0]->op != POLY_OP_INDEX) return NULL;
  PolyUOp *idx = cast->src[0];
  int sz = cast->dtype.is_ptr ? cast->dtype.vcount : cast->dtype.count;
  if (sz <= 1) return NULL;  /* nothing to split */
  PolyUOp *buf = idx->src[0];
  if (!buf) return NULL;

  /* Determine fold lengths based on hardware vector width.
   * g_max_fold_width is set by the pipeline before running this pass. */
  int fold_lengths[4];
  int n_folds = 0;
  if (g_max_fold_width >= 8) fold_lengths[n_folds++] = 8;
  fold_lengths[n_folds++] = 4;
  fold_lengths[n_folds++] = 2;
  fold_lengths[n_folds++] = 1;

  /* Split into chunks */
  PolyUOp *ret[128];
  int n_ret = 0;
  int global_offset = 0;
  PolyDType sdt = is_load ? poly_dtype_scalar(ls->dtype) : poly_dtype_scalar(ls->src[1]->dtype);

  while (global_offset < sz) {
    int fold_length = 1;
    for (int f = 0; f < n_folds; f++) {
      if (global_offset + fold_lengths[f] <= sz) { fold_length = fold_lengths[f]; break; }
    }
    /* Build INDEX at (original_offset + global_offset) */
    PolyUOp *off_idx;
    if (global_offset == 0) {
      off_idx = idx;
    } else {
      PolyUOp *off_const = poly_uop0(ctx, POLY_OP_CONST, poly_dtype_scalar(idx->src[1]->dtype),
                                       poly_arg_int(global_offset));
      PolyUOp *new_offset = poly_uop2(ctx, POLY_OP_ADD, idx->src[1]->dtype,
                                        idx->src[1], off_const, poly_arg_none());
      PolyUOp *idx_srcs[64];
      int ins = 0;
      idx_srcs[ins++] = buf;
      idx_srcs[ins++] = new_offset;
      for (int j = 2; j < idx->n_src && ins < 64; j++) idx_srcs[ins++] = idx->src[j];
      off_idx = poly_uop(ctx, POLY_OP_INDEX, buf->dtype, idx_srcs, ins, idx->arg);
    }
    PolyUOp *src_ptr = off_idx;
    if (fold_length > 1) {
      PolyDType vec_ptr = buf->dtype;
      vec_ptr.vcount = (uint16_t)fold_length;
      src_ptr = poly_uop1(ctx, POLY_OP_CAST, vec_ptr, off_idx, poly_arg_none());
    }
    if (is_load) {
      PolyDType ld_dt = (fold_length > 1) ? poly_dtype_vec(sdt, fold_length) : sdt;
      PolyUOp *ld_srcs[64];
      int lns = 0;
      ld_srcs[lns++] = src_ptr;
      for (int j = 1; j < ls->n_src && lns < 64; j++) ld_srcs[lns++] = ls->src[j];
      ret[n_ret++] = poly_uop(ctx, POLY_OP_LOAD, ld_dt, ld_srcs, lns, ls->arg);
    } else {
      int64_t gep_args[128];
      for (int j = 0; j < fold_length; j++) gep_args[j] = global_offset + j;
      PolyDType slice_dt = (fold_length > 1) ? poly_dtype_vec(sdt, fold_length) : sdt;
      PolyUOp *slice = poly_uop1(ctx, POLY_OP_GEP, slice_dt, ls->src[1],
                                   poly_arg_int_tuple_local(gep_args, fold_length));
      PolyUOp *st_srcs[64];
      int sns = 0;
      st_srcs[sns++] = src_ptr;
      st_srcs[sns++] = slice;
      for (int j = 2; j < ls->n_src && sns < 64; j++) st_srcs[sns++] = ls->src[j];
      ret[n_ret++] = poly_uop(ctx, POLY_OP_STORE, ls->dtype, st_srcs, sns, ls->arg);
    }
    global_offset += fold_length;
  }
  if (n_ret <= 1) return NULL;  /* no split needed */
  if (is_load) {
    PolyDType cat_dt = poly_dtype_vec(sdt, sz);
    return poly_uop(ctx, POLY_OP_VCAT, cat_dt, ret, n_ret, poly_arg_none());
  } else {
    return poly_uop(ctx, POLY_OP_GROUP, POLY_VOID, ret, n_ret, poly_arg_none());
  }
}

static PolyPatternMatcher *g_pm_load_store_folding = NULL;
static PolyPatternMatcher *poly_pm_load_store_folding(void) {
  if (g_pm_load_store_folding) return g_pm_load_store_folding;
  PolyRule rules[] = {
    /* expand_index: INDEX(VECTORIZE(buf), vec) → VECTORIZE(INDEX(buf, gep(vec,i)), ...) */
    { poly_pat_op(POLY_OP_INDEX, NULL, 0, "idx"), rule_expand_index },
    /* fold_expanded_index: VECTORIZE(INDEX, INDEX, ...) → PTRCAT(...).gep(remap) */
    { poly_pat_op(POLY_OP_VECTORIZE, NULL, 0, "midx"), rule_fold_expanded_index },
    /* GEP after LOAD: LOAD(GEP(ptr)) → LOAD(ptr).gep(arg) */
    { poly_pat_allow_any_len(poly_pat_op(POLY_OP_LOAD, NULL, 0, "ld")), rule_gep_after_load },
    /* GEP on STORE: STORE(GEP(ptr), data) → STORE(ptr, data.gep(inv)) */
    { poly_pat_allow_any_len(poly_pat_op(POLY_OP_STORE, NULL, 0, "sto")), rule_gep_on_store },
    /* PTRCAT after LOAD: LOAD(PTRCAT) → VCAT(LOAD, LOAD, ...) */
    { poly_pat_allow_any_len(poly_pat_op(POLY_OP_LOAD, NULL, 0, "ld")), rule_ptrcat_after_load },
    /* PTRCAT after STORE: STORE(PTRCAT, data) → GROUP(STORE, STORE, ...) */
    { poly_pat_allow_any_len(poly_pat_op(POLY_OP_STORE, NULL, 0, "sto")), rule_ptrcat_after_store },
    /* correct_load_store: split oversized LOAD/STORE(CAST(INDEX)) */
    { poly_pat_allow_any_len(poly_pat_op(POLY_OP_LOAD, NULL, 0, "ls")), rule_split_load_store },
    { poly_pat_allow_any_len(poly_pat_op(POLY_OP_STORE, NULL, 0, "ls")), rule_split_load_store },
  };
  g_pm_load_store_folding = poly_pm_new(rules, (int)(sizeof(rules) / sizeof(rules[0])));
  return g_pm_load_store_folding;
}

/* ── pm_split_ends (tinygrad codegen/late/linearizer.py:88-96) ────────── */
/*
 * After range substitution (pm_split_ranges, apply_opts_basic), an END's
 * ended_ranges src[1:] may contain arithmetic expressions instead of RANGEs.
 * This pass walks backward from each src to find all actual RANGEs, then
 * rebuilds the END as a nested chain: END(END(...END(store, r_last)..., r1), r0).
 */

static void collect_ranges_backward(PolyCtx *ctx, PolyUOp *u,
                                     PolyUOp **out, int *n, int cap) {
  if (!u || *n >= cap) return;
  if (u->op == POLY_OP_RANGE) {
    /* Deduplicate */
    for (int i = 0; i < *n; i++) if (out[i] == u) return;
    out[(*n)++] = u;
    return;
  }
  for (int i = 0; i < u->n_src; i++)
    collect_ranges_backward(ctx, u->src[i], out, n, cap);
}

static int cmp_range_axis_id(const void *a, const void *b) {
  const PolyUOp *ra = *(const PolyUOp *const *)a;
  const PolyUOp *rb = *(const PolyUOp *const *)b;
  int64_t ia = poly_range_axis_id(ra->arg);
  int64_t ib = poly_range_axis_id(rb->arg);
  return (ia > ib) - (ia < ib);
}

static PolyUOp *rule_split_ends(PolyCtx *ctx, PolyUOp *end, const PolyBindings *b) {
  (void)b;
  if (end->op != POLY_OP_END || end->n_src < 2) return NULL;

  /* Check if any ended source is not a RANGE (broken by substitution) */
  bool needs_split = false;
  for (int j = 1; j < end->n_src; j++) {
    if (end->src[j]->op != POLY_OP_RANGE) { needs_split = true; break; }
  }
  /* Also split multi-RANGE ENDs into nested single-RANGE ENDs */
  if (!needs_split && end->n_src <= 2) return NULL;

  /* Collect all RANGE UOps reachable from src[1:] */
  PolyUOp *ranges[64];
  int n_ranges = 0;
  for (int j = 1; j < end->n_src; j++)
    collect_ranges_backward(ctx, end->src[j], ranges, &n_ranges, 64);

  if (n_ranges == 0) return NULL;

  /* Sort by axis_id (ascending = outermost first) */
  qsort(ranges, (size_t)n_ranges, sizeof(PolyUOp *), cmp_range_axis_id);

  /* Build nested END chain: END(END(...END(store, r_last)..., r1), r0)
   * Innermost (highest axis_id) is deepest in the chain. */
  PolyUOp *ret = end->src[0]; /* store or inner END */
  for (int i = n_ranges - 1; i >= 0; i--) {
    PolyUOp *end_srcs[2] = { ret, ranges[i] };
    ret = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_srcs, 2, poly_arg_none());
  }
  return ret;
}

static PolyPatternMatcher *g_pm_split_ends = NULL;
static PolyPatternMatcher *poly_pm_split_ends(void) {
  if (g_pm_split_ends) return g_pm_split_ends;
  PolyRule rules[] = {
    { poly_pat_allow_any_len(poly_pat_op(POLY_OP_END, NULL, 0, "end")), rule_split_ends },
  };
  g_pm_split_ends = poly_pm_new(rules, 1);
  return g_pm_split_ends;
}

/* Forward declarations for devectorize (defined in pm_render_subset section below) */
static PolyUOp *rule_vectorize_single(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b);
static PolyUOp *lane_or_gep(PolyCtx *ctx, PolyUOp *src, int lane);

/* ── devectorize (tinygrad codegen/late/devectorizer.py) ─────────────── */
/*
 * Scatters vectorized ALU/CAST/BITCAST ops into per-element scalar ops
 * wrapped in VECTORIZE. This ensures the renderer only sees scalar ALU.
 * Vector LOAD/STORE survive and are handled by load_store_folding.
 *
 * OP(vec_a, vec_b) → VECTORIZE(OP(GEP(a,0), GEP(b,0)), OP(GEP(a,1), GEP(b,1)), ...)
 */

static PolyUOp *rule_no_vectorized_alu(PolyCtx *ctx, PolyUOp *alu, const PolyBindings *b) {
  (void)b;
  if (alu->dtype.count <= 1) return NULL;
  /* Skip pointer types: vec_ptr CASTs from fold_vectorized_index are pointer
   * type conversions, not value ALU. tinygrad: PtrDType.vcount returns 1,
   * so no_vectorized_alu naturally skips them. */
  if (alu->dtype.is_ptr) return NULL;
  int lanes = alu->dtype.count;
  if (lanes > 128) return NULL;
  PolyDType sdt = poly_dtype_scalar(alu->dtype);
  PolyUOp *elts[128];
  for (int i = 0; i < lanes; i++) {
    PolyUOp *srcs[8];
    int ns = alu->n_src;
    if (ns > 8) return NULL;
    for (int j = 0; j < ns; j++)
      srcs[j] = lane_or_gep(ctx, alu->src[j], i);
    if (ns == 0)      elts[i] = poly_uop0(ctx, alu->op, sdt, alu->arg);
    else if (ns == 1) elts[i] = poly_uop1(ctx, alu->op, sdt, srcs[0], alu->arg);
    else if (ns == 2) elts[i] = poly_uop2(ctx, alu->op, sdt, srcs[0], srcs[1], alu->arg);
    else if (ns == 3) elts[i] = poly_uop3(ctx, alu->op, sdt, srcs[0], srcs[1], srcs[2], alu->arg);
    else              elts[i] = poly_uop(ctx, alu->op, sdt, srcs, ns, alu->arg);
  }
  return poly_uop(ctx, POLY_OP_VECTORIZE, alu->dtype, elts, lanes, poly_arg_none());
}

/* ── pm_move_where_on_load (tinygrad uop/symbolic.py line 375-390) ────
 * WHERE(cond, LOAD(INDEX(buf, idx)), CONST(0)) → LOAD(INDEX(buf, idx, cond))
 * Moves validity condition into INDEX gate so renderers can emit conditional loads.
 * Without this, unconditional LOAD on out-of-bounds PAD indices SEGVs in JIT.
 *
 * Guard: only move conditions that are pure index math (no LOAD in subtree).
 * PAD bounds checks (CMPLT/AND of RANGE/CONST) pass; data-dependent conditions
 * like WHERE(x>0, x, 0) are rejected because cond contains LOAD. */
static bool uop_tree_contains_load(PolyUOp *u) {
  if (!u) return false;
  if (u->op == POLY_OP_LOAD) return true;
  for (int i = 0; i < u->n_src; i++)
    if (uop_tree_contains_load(u->src[i])) return true;
  return false;
}

static PolyUOp *rule_where_on_load(PolyCtx *ctx, PolyUOp *w, const PolyBindings *b) {
  (void)b;
  if (w->op != POLY_OP_WHERE || w->n_src != 3) return NULL;
  PolyUOp *cond = w->src[0];
  PolyUOp *true_val = w->src[1];
  PolyUOp *false_val = w->src[2];

  /* Match: WHERE(cond, LOAD(INDEX(buf, idx)), CONST(0)) */
  if (true_val->op != POLY_OP_LOAD || true_val->n_src != 1) return NULL;
  PolyUOp *idx = true_val->src[0];
  if (idx->op != POLY_OP_INDEX || idx->n_src < 2) return NULL;
  if (false_val->op != POLY_OP_CONST) return NULL;
  /* Check false_val is zero */
  if (false_val->arg.kind == POLY_ARG_FLOAT && false_val->arg.f != 0.0) return NULL;
  if (false_val->arg.kind == POLY_ARG_INT && false_val->arg.i != 0) return NULL;

  /* Guard: reject if cond depends on loaded data (contains LOAD in subtree).
   * PAD bounds checks are pure index math (RANGE/CONST/ALU); data-dependent
   * conditions like WHERE(x>0, x, 0) contain LOAD and must stay as WHERE. */
  if (uop_tree_contains_load(cond)) return NULL;

  /* Merge cond with existing gate (AND them) if INDEX already has a gate */
  PolyUOp *gate = cond;
  if (idx->n_src >= 3) {
    gate = poly_uop2(ctx, POLY_OP_AND, POLY_BOOL, idx->src[2], cond, poly_arg_none());
  }

  /* Create gated INDEX: INDEX(buf, idx_expr, gate) */
  PolyUOp *new_srcs[3] = { idx->src[0], idx->src[1], gate };
  PolyUOp *gated_idx = poly_uop(ctx, POLY_OP_INDEX, idx->dtype, new_srcs, 3, idx->arg);

  /* 2-source LOAD: LOAD(gated_INDEX, alt_value) -- tinygrad parity.
   * Alt value is the original false_val (the zero constant). */
  PolyUOp *load_srcs[2] = { gated_idx, false_val };
  return poly_uop(ctx, POLY_OP_LOAD, true_val->dtype, load_srcs, 2, true_val->arg);
}

/* Also handle reversed: WHERE(cond, CONST(0), LOAD(INDEX(buf, idx)))
 * → LOAD(INDEX(buf, idx, NEG(cond))) */
static PolyUOp *rule_where_on_load_rev(PolyCtx *ctx, PolyUOp *w, const PolyBindings *b) {
  (void)b;
  if (w->op != POLY_OP_WHERE || w->n_src != 3) return NULL;
  PolyUOp *cond = w->src[0];
  PolyUOp *true_val = w->src[1];
  PolyUOp *false_val = w->src[2];

  /* Match: WHERE(cond, CONST(0), LOAD(INDEX(buf, idx))) */
  if (false_val->op != POLY_OP_LOAD || false_val->n_src != 1) return NULL;
  PolyUOp *idx = false_val->src[0];
  if (idx->op != POLY_OP_INDEX || idx->n_src < 2) return NULL;
  if (true_val->op != POLY_OP_CONST) return NULL;
  if (true_val->arg.kind == POLY_ARG_FLOAT && true_val->arg.f != 0.0) return NULL;
  if (true_val->arg.kind == POLY_ARG_INT && true_val->arg.i != 0) return NULL;

  /* Same guard as rule_where_on_load: reject data-dependent conditions */
  if (uop_tree_contains_load(cond)) return NULL;

  /* Gate is negated condition */
  PolyUOp *neg_cond = poly_uop1(ctx, POLY_OP_NEG, cond->dtype, cond, poly_arg_none());

  PolyUOp *gate = neg_cond;
  if (idx->n_src >= 3) {
    gate = poly_uop2(ctx, POLY_OP_AND, POLY_BOOL, idx->src[2], neg_cond, poly_arg_none());
  }

  PolyUOp *new_srcs[3] = { idx->src[0], idx->src[1], gate };
  PolyUOp *gated_idx = poly_uop(ctx, POLY_OP_INDEX, idx->dtype, new_srcs, 3, idx->arg);

  /* 2-source LOAD: LOAD(gated_INDEX, alt_value) -- tinygrad parity.
   * Alt value is the original true_val (the zero constant, since this is the reversed form). */
  PolyUOp *load_srcs[2] = { gated_idx, true_val };
  return poly_uop(ctx, POLY_OP_LOAD, false_val->dtype, load_srcs, 2, false_val->arg);
}

static PolyPatternMatcher *g_pm_move_where_on_load = NULL;
static PolyPatternMatcher *poly_pm_move_where_on_load(void) {
  if (g_pm_move_where_on_load) return g_pm_move_where_on_load;
  PolyRule rules[] = {
    { poly_pat_op(POLY_OP_WHERE, NULL, 0, "w"), rule_where_on_load },
    { poly_pat_op(POLY_OP_WHERE, NULL, 0, "w"), rule_where_on_load_rev },
  };
  g_pm_move_where_on_load = poly_pm_new(rules, 2);
  return g_pm_move_where_on_load;
}

/* INDEX(buf, idx, true) → INDEX(buf, idx) — drop unconditional validity gate */
static PolyUOp *rule_drop_true_gate(PolyCtx *ctx, PolyUOp *idx, const PolyBindings *b) {
  (void)b;
  if (idx->op != POLY_OP_INDEX || idx->n_src < 3) return NULL;
  PolyUOp *gate = idx->src[2];
  if (gate->op != POLY_OP_CONST || !poly_dtype_is_bool(gate->dtype)) return NULL;
  if (gate->arg.kind != POLY_ARG_INT || gate->arg.i != 1) return NULL;
  return poly_uop2(ctx, POLY_OP_INDEX, idx->dtype, idx->src[0], idx->src[1], idx->arg);
}

static PolyPatternMatcher *g_pm_devectorize = NULL;
static PolyPatternMatcher *poly_pm_devectorize(void) {
  if (g_pm_devectorize) return g_pm_devectorize;

  PolyRule rules[80];
  int n = 0;

  /* Scatter ALL elementwise ops (ALU + CAST + BITCAST) from vec to scalar.
   * Matches tinygrad devectorizer.py:283: UPat((*GroupOp.ALU, Ops.CAST, Ops.BITCAST), ...) */
  rules[n++] = (PolyRule){
    poly_pat_allow_any_len(poly_pat_ops(POLY_GROUP_ELEMENTWISE, NULL, 0, "alu")),
    rule_no_vectorized_alu
  };

  /* Drop true gate from INDEX */
  rules[n++] = (PolyRule){
    poly_pat_allow_any_len(poly_pat_op(POLY_OP_INDEX, NULL, 0, "idx")),
    rule_drop_true_gate
  };

  /* VECTORIZE(single) → unwrap */
  rules[n++] = (PolyRule){
    poly_pat_op(POLY_OP_VECTORIZE, NULL, 0, "u"),
    rule_vectorize_single
  };

  g_pm_devectorize = poly_pm_new(rules, n);
  return g_pm_devectorize;
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

/* VCAT → VECTORIZE(GEP, GEP, ...) lowering (tinygrad symbolic.py:196):
 * VCAT can't be rendered; expand to VECTORIZE of per-element GEPs. */
static PolyUOp *rule_cat_to_vectorize(PolyCtx *ctx, PolyUOp *x, const PolyBindings *b) {
  (void)b;
  if (!x || x->op != POLY_OP_VCAT || x->n_src <= 0) return NULL;
  if (x->dtype.is_ptr) return NULL;  /* don't expand pointer CATs */
  PolyUOp *elts[128];
  int p = 0;
  PolyDType sdt = poly_dtype_scalar(x->dtype);
  for (int i = 0; i < x->n_src; i++) {
    PolyUOp *src = x->src[i];
    int cnt = src->dtype.count;
    if (cnt <= 0) cnt = 1;
    for (int j = 0; j < cnt && p < 128; j++)
      elts[p++] = make_gep_lane(ctx, src, j);
  }
  return poly_uop(ctx, POLY_OP_VECTORIZE, x->dtype, elts, p, poly_arg_none());
}

/* Render subset: full (DEVECTORIZE>=1) scatters vec CMP/WHERE to scalar.
 * Minimal (DEVECTORIZE=0) keeps vec ALU, only lowers VCONST/VCAT. */
static PolyPatternMatcher *g_pm_render_subset = NULL;
static PolyPatternMatcher *poly_pm_render_subset(void) {
  if (g_pm_render_subset) return g_pm_render_subset;
  PolyRule rules[] = {
    { poly_pat_op(POLY_OP_VCONST, NULL, 0, "u"), rule_render_vconst },
    { poly_pat_op(POLY_OP_VCAT, NULL, 0, "x"), rule_cat_to_vectorize },
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

static PolyPatternMatcher *g_pm_render_subset_vec = NULL;
static PolyPatternMatcher *poly_pm_render_subset_vec(void) {
  if (g_pm_render_subset_vec) return g_pm_render_subset_vec;
  PolyRule rules[] = {
    { poly_pat_op(POLY_OP_VCONST, NULL, 0, "u"), rule_render_vconst },
    { poly_pat_op(POLY_OP_VCAT, NULL, 0, "x"), rule_cat_to_vectorize },
    /* Scatter vec CMP/WHERE to per-lane scalar (same as render_subset).
     * tinygrad does this even with DEVECTORIZE=0 — comparison semantics
     * require per-element evaluation, not packed SSE cmpps. */
    { poly_pat_op(POLY_OP_CMPLT, NULL, 0, "u"), rule_vector_cmp_to_scalarized_vector },
    { poly_pat_op(POLY_OP_CMPNE, NULL, 0, "u"), rule_vector_cmp_to_scalarized_vector },
    { poly_pat_op(POLY_OP_CMPEQ, NULL, 0, "u"), rule_vector_cmp_to_scalarized_vector },
    { poly_pat_op(POLY_OP_WHERE, NULL, 0, "u"), rule_vector_where_to_scalar },
    { poly_pat_op(POLY_OP_NEG, NULL, 0, "u"), rule_vector_bool_neg_to_scalarized_vector },
    { poly_pat_op(POLY_OP_VECTORIZE, NULL, 0, "u"), rule_vectorize_single },
  };
  /* Include gep_pushing so VECTORIZE(GEP(x,0),...) → x identity fires.
   * The expander at stage 8 creates GEP+VECTORIZE that need cleanup. */
  PolyPatternMatcher *base = poly_pm_new(rules, (int)(sizeof(rules) / sizeof(rules[0])));
  g_pm_render_subset_vec = poly_pm_concat(base, poly_pm_gep_pushing());
  poly_pm_destroy(base); /* concat copied rules; base infra no longer needed */
  return g_pm_render_subset_vec;
}

/* Render subset for x64 with SIMD integer: keep vector CMP/WHERE packed.
 * Unlike render_subset_vec, does NOT scatter CMP/WHERE to per-lane scalar. */
static PolyPatternMatcher *g_pm_render_subset_x64 = NULL;
static PolyPatternMatcher *poly_pm_render_subset_x64(void) {
  if (g_pm_render_subset_x64) return g_pm_render_subset_x64;
  PolyRule rules[] = {
    { poly_pat_op(POLY_OP_VCONST, NULL, 0, "u"), rule_render_vconst },
    { poly_pat_op(POLY_OP_VCAT, NULL, 0, "x"), rule_cat_to_vectorize },
    /* Keep vector CMP/WHERE packed -- x64 handles them natively */
    { poly_pat_op(POLY_OP_VECTORIZE, NULL, 0, "u"), rule_vectorize_single },
  };
  PolyPatternMatcher *base = poly_pm_new(rules, (int)(sizeof(rules) / sizeof(rules[0])));
  g_pm_render_subset_x64 = poly_pm_concat(base, poly_pm_gep_pushing());
  poly_pm_destroy(base);
  return g_pm_render_subset_x64;
}

/* ── Combined devectorize pass (cached) ─────────────────────────────── */
/*
 * Matches tinygrad codegen/__init__.py:79:
 *   pm_devectorize = sym+devectorize+load_store_folding+correct_load_store+load_store_indexing
 * All rules in ONE graph_rewrite so folding sees vectorized INDEX before devectorize scatters.
 */

static PolyPatternMatcher *g_combined_devec = NULL;
static PolyPatternMatcher *poly_pm_combined_devec(void) {
  if (g_combined_devec) return g_combined_devec;
  /* Matches tinygrad codegen/__init__.py:79:
   *   pm_devectorize = sym+devectorize+load_store_folding+correct_load_store+load_store_indexing
   * Include gep_pushing (from sym) so GEP(ALU) resolves between expand_index and fold.
   * Full sym excluded (render_subset's VCONST→VECTORIZE causes cycles). */
  static PolyPatternMatcher *gep_devec = NULL;
  if (!gep_devec) gep_devec = poly_pm_concat(poly_pm_gep_pushing(), poly_pm_devectorize());
  g_combined_devec = poly_pm_concat(gep_devec, poly_pm_load_store_folding());
  return g_combined_devec;
}

static PolyPatternMatcher *g_combined_nodevec = NULL;
static PolyPatternMatcher *poly_pm_combined_nodevec(void) {
  if (g_combined_nodevec) return g_combined_nodevec;
  /* Matches tinygrad pm_no_devec = sym + load_store_folding + correct_load_store + load_store_indexing.
   * gep_pushing (from sym) is required so GEP nodes simplify between
   * expand_index and fold_expanded_index for contiguity detection.
   * load_store_indexing: drop_true_gate is the only non-image rule. */
  PolyRule indexing_rules[] = {
    { poly_pat_allow_any_len(poly_pat_op(POLY_OP_INDEX, NULL, 0, "idx")), rule_drop_true_gate },
  };
  PolyPatternMatcher *pm_indexing = poly_pm_new(indexing_rules, 1);
  PolyPatternMatcher *base = poly_pm_concat(poly_pm_gep_pushing(), poly_pm_load_store_folding());
  g_combined_nodevec = poly_pm_concat(base, pm_indexing);
  poly_pm_destroy(base);       /* concat copied rules */
  poly_pm_destroy(pm_indexing); /* concat copied rules */
  return g_combined_nodevec;
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

/* ── Public wrapper for heuristic (used by tests) ──────────────────── */
PolyUOp *poly_apply_opts_heuristic_ex(PolyCtx *ctx, PolyUOp *sink, PolyRendererCaps caps) {
  return poly_apply_opts_heuristic(ctx, sink, caps);
}

/* ── Public accessors for individual passes (used by CUDA linearizer) ── */

PolyPatternMatcher *poly_pm_reduce_pass(void)        { return poly_pm_reduce(); }
PolyPatternMatcher *poly_pm_devectorize_pass(void)   { return poly_pm_devectorize(); }
PolyPatternMatcher *poly_pm_decomp_pass(void)         { return poly_pm_decomp(); }
PolyPatternMatcher *poly_pm_decomp_pass_caps(PolyRendererCaps caps) {
  return poly_pm_decomp_with_caps(caps.has_mulacc, caps.has_threefry);
}
PolyPatternMatcher *poly_pm_transcendental_pass(void)  { return poly_pm_transcendental(); }
PolyPatternMatcher *poly_pm_pre_expander_pass(void)    { return poly_pm_pre_expander(); }
PolyPatternMatcher *poly_pm_expander_pass(void)        { return poly_pm_expander(); }
void poly_reset_acc_num(void)                         { }

PolyUOp *poly_apply_tc_opt(PolyCtx *ctx, PolyUOp *sink, PolyRendererCaps caps) {
  if (caps.n_tensor_cores <= 0) return sink;
  OptScheduler s;
  sched_init(&s, ctx, sink);
  if (s.n_rngs == 0) return sink;

  int n_reduce = 0;
  for (int i = 0; i < s.n_rngs; i++)
    if (s.types[i] == POLY_AXIS_GROUP_REDUCE || s.types[i] == POLY_AXIS_REDUCE)
      n_reduce++;

  int tc_opt_env = 0;
  { const char *e = getenv("POLY_TC_OPT"); if (e) tc_opt_env = atoi(e); }
  int use_tc_env = 1;
  { const char *e = getenv("POLY_USE_TC"); if (e) use_tc_env = atoi(e); }

  if (use_tc_env > 0 && (n_reduce == 1 || tc_opt_env >= 1)) {
    OptScheduler tk;
    sched_copy(&tk, &s);
    PolyUOp *tc_axes[3];
    bool tc_ok = sched_apply_tc_opt(&tk, 0, -1, tc_opt_env, use_tc_env,
                                     caps.tensor_cores, caps.n_tensor_cores, tc_axes);
    if (tc_ok) {
      for (int tc_dim = 1; tc_dim >= 0; tc_dim--) {
        int64_t bound = 0;
        if (tc_axes[tc_dim] && tc_axes[tc_dim]->n_src > 0 &&
            tc_axes[tc_dim]->src[0]->op == POLY_OP_CONST)
          bound = tc_axes[tc_dim]->src[0]->arg.i;
        if (bound <= 1) continue;
        int szs[] = {5, 4, 3, 2};
        for (int si = 0; si < 4; si++) {
          if (bound % szs[si] == 0) {
            int idx = -1;
            for (int ri = 0; ri < tk.n_rngs; ri++)
              if (tk.rngs[ri] == tc_axes[tc_dim]) { idx = ri; break; }
            if (idx >= 0)
              tc_axes[tc_dim] = sched_shift_to(&tk, tk.rngs[idx], szs[si], POLY_AXIS_UPCAST, false);
            break;
          }
        }
      }
      return tk.ast;
    }
  }
  return sink;
}

PolyUOp *poly_apply_pm_reduce(PolyCtx *ctx, PolyUOp *sink) {
  ReduceContext local_ctx = {0};
  PolyUOp *out = poly_graph_rewrite_ctx(ctx, sink, poly_pm_reduce(), &local_ctx);
  reduce_ctx_clear(&local_ctx);
  return out;
}

/* ── Full rewrite-to-sink pipeline ───────────────────────────────────── */

static bool device_is_gpu(int device) {
  return device == POLY_DEVICE_CUDA || device == POLY_DEVICE_HIP ||
         device == POLY_DEVICE_WEBGPU;
}

PolyUOp *poly_full_rewrite_to_sink_ex(PolyCtx *ctx, PolyUOp *sink, PolyRewriteOpts opts) {
  /*
   * Unified pipeline matching tinygrad codegen/__init__.py full_rewrite_to_sink.
   * Backend-specific behavior is controlled by renderer config fields in opts,
   * not by if-device branches.
   *
   * New Phase 4 fields used here:
   *   opt_policy      — POLY_OPT_HEURISTIC (CPU) or POLY_OPT_TC_ONLY (GPU)
   *   gpu_block_size  — group_for_reduce block size (0 = skip)
   *   device          — PolyDeviceId, gates gpudims/control_flow
   *   extra_matcher   — renderer-specific final rewrite patterns (NULL = none)
   */

  /* ── 1. Preprocessing + optimization (gated by optimize) ─────────────
   * Matches tinygrad: optimize gates both preprocessing and apply_opts.
   * Backend differences come from opt_policy, not from skipping stages. */
  if (opts.optimize) {
    /* tinygrad lines 42-51: split ranges + flatten + sym + simplify */
    SplitRangeCtx srctx = {0};
    sink = poly_graph_rewrite_ctx(ctx, sink, poly_pm_split_ranges(), &srctx);
    sink = poly_graph_rewrite(ctx, sink, poly_pm_flatten_range());
    sink = poly_graph_rewrite(ctx, sink, poly_symbolic_simple());
    sink = poly_graph_rewrite(ctx, sink, poly_pm_flatten_range());
    sink = poly_graph_rewrite(ctx, sink, poly_pm_simplify_ranges());

    /* tinygrad line 54: apply_opts — renderer-aware optimization */
    if (opts.beam_width > 0) {
      sink = poly_beam_search(ctx, sink, opts.beam_width, opts);
    } else if (opts.opt_policy == POLY_OPT_TC_ONLY) {
      /* GPU: TC detection only, no CPU-oriented upcast/unroll */
      sink = poly_apply_tc_opt(ctx, sink, opts.caps);
    } else {
      /* CPU: full heuristic (includes TC when caps have tensor cores) */
      sink = poly_apply_opts_heuristic(ctx, sink, opts.caps);
    }
  }

  /* ── 2. Postopt symbolic + move WHERE on load + expander ────────────── */
  sink = poly_graph_rewrite(ctx, sink, poly_symbolic_simple());
  sink = poly_graph_rewrite(ctx, sink, poly_pm_move_where_on_load());
  sink = poly_graph_rewrite(ctx, sink, poly_pm_pre_expander());
  sink = poly_graph_rewrite(ctx, sink, poly_pm_expander());

  /* ── 3. Symbolic ────────────────────────────────────────────────────── */
  sink = poly_graph_rewrite(ctx, sink, poly_symbolic_simple());

  /* ── 4. GPU parallel reduction (gated by gpu_block_size > 0) ────────── */
  if (opts.gpu_block_size > 0)
    sink = poly_group_for_reduce(ctx, sink, opts.gpu_block_size);

  /* ── 5. pm_reduce + symbolic ────────────────────────────────────────── */
  sink = poly_apply_pm_reduce(ctx, sink);
  sink = poly_graph_rewrite(ctx, sink, poly_symbolic_simple());

  /* ── 6. Add loads + devectorize (gated by devectorize >= 0) ─────────── */
  g_max_fold_width = (opts.caps.max_vec_width >= 8) ? 8 : 4;
  if (opts.devectorize >= 0) {
    sink = poly_graph_rewrite(ctx, sink, poly_pm_add_loads());
    sink = poly_graph_rewrite(ctx, sink,
        (opts.devectorize >= 1) ? poly_pm_combined_devec() : poly_pm_combined_nodevec());

    if (opts.devectorize >= 1)
      sink = poly_graph_rewrite(ctx, sink, poly_pm_render_subset());
    else if (opts.caps.has_simd_int)
      sink = poly_graph_rewrite(ctx, sink, poly_pm_render_subset_x64());
    else
      sink = poly_graph_rewrite(ctx, sink, poly_pm_render_subset_vec());
    sink = poly_graph_rewrite(ctx, sink, poly_symbolic_simple());
  }

  /* ── 7. Decompositions ──────────────────────────────────────────────── */
  sink = poly_graph_rewrite(ctx, sink, poly_pm_decomp_with_caps(opts.caps.has_mulacc, opts.caps.has_threefry));
  sink = poly_graph_rewrite(ctx, sink, poly_pm_transcendental());
  sink = poly_graph_rewrite(ctx, sink, poly_pm_decomp_with_caps(opts.caps.has_mulacc, opts.caps.has_threefry));

  /* ── 8. Final rewrite: extra_matcher + expander + split_ends + render ── */
  if (opts.extra_matcher)
    sink = poly_graph_rewrite(ctx, sink, opts.extra_matcher);
  sink = poly_graph_rewrite(ctx, sink, poly_pm_expander());
  sink = poly_graph_rewrite(ctx, sink, poly_pm_split_ends());
  if (opts.devectorize >= 1 || opts.devectorize < 0)
    sink = poly_graph_rewrite(ctx, sink, poly_pm_render_subset());
  else if (opts.caps.has_simd_int)
    sink = poly_graph_rewrite(ctx, sink, poly_pm_render_subset_x64());
  else
    sink = poly_graph_rewrite(ctx, sink, poly_pm_render_subset_vec());

  /* ── 9. GPU dims (gated by device is GPU) ─────────────────────────── */
  if (device_is_gpu(opts.device))
    sink = poly_add_gpudims(ctx, sink);

  /* ── 10. Control flow (unconditional, tinygrad parity) ──────────────── */
  sink = poly_apply_control_flow(ctx, sink);

  return sink;
}

PolyUOp *poly_full_rewrite_to_sink(PolyCtx *ctx, PolyUOp *sink) {
  PolyRewriteOpts opts = { .optimize = false, .devectorize = 0 };
  return poly_full_rewrite_to_sink_ex(ctx, sink, opts);
}
