/*
 * pat.c — Pattern matcher: PolyPat, PolyPatternMatcher, graph_rewrite
 *
 * Mirrors tinygrad's UPat.match(), PatternMatcher.rewrite(), and
 * unified_rewrite (top-down mode).
 */

#include "pat.h"
#include "arena.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/* ── OpSet helpers ────────────────────────────────────────────────────── */

static int opset_popcount(PolyOpSet s) {
  return __builtin_popcountll(s.bits[0]) + __builtin_popcountll(s.bits[1]);
}

static PolyOps opset_first(PolyOpSet s) {
  if (s.bits[0]) return (PolyOps)__builtin_ctzll(s.bits[0]);
  if (s.bits[1]) return (PolyOps)(64 + __builtin_ctzll(s.bits[1]));
  return (PolyOps)0;
}

/* ── Pattern constructors ─────────────────────────────────────────────── */

static PolyPat *pat_alloc(void) {
  PolyPat *p = calloc(1, sizeof(PolyPat));
  return p;
}

static PolyOpSet compute_early_reject(PolyPat **src, int n_src) {
  PolyOpSet rej = {{ 0, 0 }};
  if (!src) return rej;
  for (int i = 0; i < n_src; i++) {
    if (src[i]->has_ops && opset_popcount(src[i]->ops) == 1)
      rej = poly_opset_add(rej, opset_first(src[i]->ops));
  }
  return rej;
}

static PolyPat **dup_src(PolyPat **src, int n) {
  if (!src || n == 0) return NULL;
  PolyPat **d = malloc(n * sizeof(PolyPat *));
  memcpy(d, src, n * sizeof(PolyPat *));
  return d;
}

PolyPat *poly_pat_any(const char *name) {
  PolyPat *p = pat_alloc();
  p->name = name;
  return p;
}

PolyPat *poly_pat_cvar(const char *name) {
  PolyPat *p = pat_alloc();
  p->has_ops = true;
  /* CONST | VCONST */
  p->ops = poly_opset_add(poly_opset_add((PolyOpSet){{0,0}}, POLY_OP_CONST), POLY_OP_VCONST);
  p->name = name;
  return p;
}

PolyPat *poly_pat_const_val(PolyArg val) {
  PolyPat *p = pat_alloc();
  p->has_ops = true;
  p->ops = poly_opset_add((PolyOpSet){{0,0}}, POLY_OP_CONST);
  p->match_arg = true;
  p->arg = val;
  return p;
}

PolyPat *poly_pat_op(PolyOps op, PolyPat **src, int n_src, const char *name) {
  PolyPat *p = pat_alloc();
  p->has_ops = true;
  p->ops = poly_opset_add((PolyOpSet){{0,0}}, op);
  p->src = dup_src(src, n_src);
  p->n_src = n_src;
  p->strict_length = (src != NULL);
  p->name = name;
  p->early_reject = compute_early_reject(p->src, p->n_src);
  return p;
}

PolyPat *poly_pat_ops(PolyOpSet ops, PolyPat **src, int n_src, const char *name) {
  PolyPat *p = pat_alloc();
  p->has_ops = true;
  p->ops = ops;
  p->src = dup_src(src, n_src);
  p->n_src = n_src;
  p->strict_length = (src != NULL);
  p->name = name;
  p->early_reject = compute_early_reject(p->src, p->n_src);
  return p;
}

PolyPat *poly_pat_op1(PolyOps op, PolyPat *s0, const char *name) {
  PolyPat *arr[] = { s0 };
  return poly_pat_op(op, arr, 1, name);
}

PolyPat *poly_pat_op2(PolyOps op, PolyPat *s0, PolyPat *s1, const char *name) {
  PolyPat *arr[] = { s0, s1 };
  return poly_pat_op(op, arr, 2, name);
}

PolyPat *poly_pat_op2c(PolyOps op, PolyPat *s0, PolyPat *s1, const char *name) {
  PolyPat *arr[] = { s0, s1 };
  PolyPat *p = poly_pat_op(op, arr, 2, name);
  p->commutative = true;
  return p;
}

PolyPat *poly_pat_op3(PolyOps op, PolyPat *s0, PolyPat *s1, PolyPat *s2, const char *name) {
  PolyPat *arr[] = { s0, s1, s2 };
  return poly_pat_op(op, arr, 3, name);
}

PolyPat *poly_pat_ops1(PolyOpSet ops, PolyPat *s0, const char *name) {
  PolyPat *arr[] = { s0 };
  return poly_pat_ops(ops, arr, 1, name);
}

PolyPat *poly_pat_ops2(PolyOpSet ops, PolyPat *s0, PolyPat *s1, const char *name) {
  PolyPat *arr[] = { s0, s1 };
  return poly_pat_ops(ops, arr, 2, name);
}

PolyPat *poly_pat_ops3(PolyOpSet ops, PolyPat *s0, PolyPat *s1, PolyPat *s2, const char *name) {
  PolyPat *arr[] = { s0, s1, s2 };
  return poly_pat_ops(ops, arr, 3, name);
}

PolyPat *poly_pat_dtype(const char *name, PolyDType *dtypes, int n) {
  PolyPat *p = pat_alloc();
  p->name = name;
  p->dtypes = malloc(n * sizeof(PolyDType));
  memcpy(p->dtypes, dtypes, n * sizeof(PolyDType));
  p->n_dtypes = n;
  return p;
}

PolyPat *poly_pat_allow_any_len(PolyPat *p) {
  if (!p) return NULL;
  p->strict_length = false;
  return p;
}

PolyPat *poly_pat_or_casted(PolyPat *p) {
  if (!p) return NULL;
  p->or_casted = true;
  return p;
}

PolyPat *poly_pat_set_early_reject(PolyPat *p, PolyOpSet early_reject) {
  if (!p) return NULL;
  p->early_reject = early_reject;
  return p;
}

void poly_pat_free(PolyPat *p) {
  if (!p) return;
  if (p->src) {
    for (int i = 0; i < p->n_src; i++)
      poly_pat_free(p->src[i]);
    free(p->src);
  }
  free(p->dtypes);
  free(p);
}

/* ── Pattern matching ─────────────────────────────────────────────────── */

static bool match_sources(const PolyPat *pat, PolyUOp *uop, PolyBindings *binds) {
  for (int i = 0; i < pat->n_src; i++) {
    if (!poly_pat_match(pat->src[i], uop->src[i], binds))
      return false;
  }
  return true;
}

bool poly_pat_match(const PolyPat *pat, PolyUOp *uop, PolyBindings *binds) {
  /* CAST-tolerant match: pattern or CAST(pattern). */
  if (pat->or_casted && uop->op == POLY_OP_CAST && uop->n_src == 1) {
    int saved = binds->n;
    if (poly_pat_match(pat, uop->src[0], binds))
      return true;
    binds->n = saved;
  }

  /* Op check */
  if (pat->has_ops && !poly_opset_has(pat->ops, uop->op))
    return false;

  /* Name binding: if already bound, must be same pointer */
  if (pat->name) {
    PolyUOp *existing = poly_bind(binds, pat->name);
    if (existing) {
      if (existing != uop) return false;
    } else {
      if (binds->n >= POLY_MAX_BINDINGS) return false;
      binds->names[binds->n] = pat->name;
      binds->uops[binds->n] = uop;
      binds->n++;
    }
  }

  /* DType check */
  if (pat->n_dtypes > 0) {
    bool found = false;
    PolyDType scalar = poly_dtype_scalar(uop->dtype);
    for (int i = 0; i < pat->n_dtypes; i++) {
      if (poly_dtype_eq(pat->dtypes[i], uop->dtype) ||
          poly_dtype_eq(pat->dtypes[i], scalar)) {
        found = true;
        break;
      }
    }
    if (!found) return false;
  }

  /* Arg check */
  if (pat->match_arg && !poly_arg_eq(pat->arg, uop->arg))
    return false;

  /* Source count */
  if (uop->n_src < pat->n_src) return false;
  if (pat->strict_length && uop->n_src != pat->n_src) return false;

  /* No source constraints = done */
  if (pat->src == NULL) return true;

  /* Commutative: try both orderings */
  if (pat->commutative && pat->n_src == 2) {
    int saved = binds->n;
    if (match_sources(pat, uop, binds))
      return true;
    binds->n = saved;
    /* Swap: match pat->src[0] against uop->src[1] and vice versa */
    if (poly_pat_match(pat->src[0], uop->src[1], binds) &&
        poly_pat_match(pat->src[1], uop->src[0], binds))
      return true;
    binds->n = saved;
    return false;
  }

  /* Fixed-order match */
  return match_sources(pat, uop, binds);
}

/* ── PatternMatcher ───────────────────────────────────────────────────── */

struct PolyPatternMatcher {
  PolyRule *rules;
  int n_rules;
  struct { int *indices; int n; int cap; } by_op[POLY_OP_COUNT];
};

static void pm_add_op(PolyPatternMatcher *pm, int op, int rule_idx) {
  if (op < 0 || op >= POLY_OP_COUNT) return;
  if (pm->by_op[op].n >= pm->by_op[op].cap) {
    pm->by_op[op].cap = pm->by_op[op].cap ? pm->by_op[op].cap * 2 : 8;
    pm->by_op[op].indices = realloc(pm->by_op[op].indices,
                                     pm->by_op[op].cap * sizeof(int));
  }
  pm->by_op[op].indices[pm->by_op[op].n++] = rule_idx;
}

PolyPatternMatcher *poly_pm_new(const PolyRule *rules, int n_rules) {
  PolyPatternMatcher *pm = calloc(1, sizeof(PolyPatternMatcher));
  pm->rules = malloc(n_rules * sizeof(PolyRule));
  memcpy(pm->rules, rules, n_rules * sizeof(PolyRule));
  pm->n_rules = n_rules;

  for (int i = 0; i < n_rules; i++) {
    PolyPat *p = rules[i].pat;
    if (!p->has_ops) {
      /* Pattern matches any op — add to all op lists */
      for (int j = 0; j < POLY_OP_COUNT; j++)
        pm_add_op(pm, j, i);
    } else {
      for (int j = 0; j < POLY_OP_COUNT; j++) {
        if (poly_opset_has(p->ops, (PolyOps)j))
          pm_add_op(pm, j, i);
      }
    }
  }
  return pm;
}

void poly_pm_destroy(PolyPatternMatcher *pm) {
  if (!pm) return;
  for (int i = 0; i < POLY_OP_COUNT; i++)
    free(pm->by_op[i].indices);
  free(pm->rules);
  free(pm);
}

PolyUOp *poly_pm_rewrite(PolyPatternMatcher *pm, PolyCtx *ctx, PolyUOp *uop) {
  int op = (int)uop->op;
  if (op < 0 || op >= POLY_OP_COUNT || pm->by_op[op].n == 0)
    return NULL;

  /* Compute src ops bitmask for early reject */
  PolyOpSet src_ops = {{ 0, 0 }};
  for (int i = 0; i < uop->n_src; i++)
    src_ops = poly_opset_add(src_ops, uop->src[i]->op);

  for (int i = 0; i < pm->by_op[op].n; i++) {
    int idx = pm->by_op[op].indices[i];
    PolyRule *rule = &pm->rules[idx];

    /* Early reject: required ops must appear in sources */
    if (!poly_opset_subset(rule->pat->early_reject, src_ops))
      continue;

    PolyBindings binds = { .n = 0 };
    if (poly_pat_match(rule->pat, uop, &binds)) {
      PolyUOp *result = rule->fn(ctx, uop, &binds);
      if (result != NULL && result != uop) return result;
    }
  }
  return NULL;
}

PolyPatternMatcher *poly_pm_concat(PolyPatternMatcher *a, PolyPatternMatcher *b) {
  int total = a->n_rules + b->n_rules;
  PolyRule *combined = malloc(total * sizeof(PolyRule));
  memcpy(combined, a->rules, a->n_rules * sizeof(PolyRule));
  memcpy(combined + a->n_rules, b->rules, b->n_rules * sizeof(PolyRule));
  PolyPatternMatcher *result = poly_pm_new(combined, total);
  free(combined);
  return result;
}

/* ── graph_rewrite (top-down unified_rewrite) ─────────────────────────── */

/* Pointer hash/eq for maps */
static bool ptr_eq(const void *a, const void *b) { return a == b; }
static uint32_t ptr_hash(const void *p) {
  uintptr_t v = (uintptr_t)p;
  return (uint32_t)(v ^ (v >> 16) ^ (sizeof(v) > 4 ? (uint32_t)(v >> 32) : 0));
}

/* Worklist entry */
typedef struct {
  PolyUOp *n;       /* original node */
  int stage;       /* 0, 1, or 2 */
  PolyUOp *new_n;   /* potentially rewritten node */
} WorkItem;

/* Dynamic stack */
typedef struct {
  WorkItem *items;
  int top, cap;
} WorkStack;

static void ws_push(WorkStack *ws, PolyUOp *n, int stage, PolyUOp *new_n) {
  if (ws->top >= ws->cap) {
    ws->cap = ws->cap ? ws->cap * 2 : 256;
    ws->items = realloc(ws->items, ws->cap * sizeof(WorkItem));
  }
  ws->items[ws->top++] = (WorkItem){ n, stage, new_n };
}

/* Waitlist: linked list of work items per UOp key */
typedef struct WaitNode {
  WorkItem item;
  struct WaitNode *next;
} WaitNode;

static void waitlist_add(PolyMap *wl, PolyUOp *key, WorkItem item) {
  uint32_t h = ptr_hash(key);
  WaitNode *node = malloc(sizeof(WaitNode));
  node->item = item;
  node->next = poly_map_get(wl, h, key, ptr_eq);
  poly_map_set(wl, h, key, node, ptr_eq);
}

static void waitlist_flush(PolyMap *wl, PolyUOp *key, WorkStack *ws) {
  uint32_t h = ptr_hash(key);
  WaitNode *chain = poly_map_get(wl, h, key, ptr_eq);
  while (chain) {
    ws_push(ws, chain->item.n, chain->item.stage, chain->item.new_n);
    WaitNode *next = chain->next;
    free(chain);
    chain = next;
  }
  poly_map_remove(wl, h, key, ptr_eq);
}

#define REWRITE_STACK_LIMIT 100000

static PolyUOp *replace_get(PolyMap *m, PolyUOp *key) {
  return poly_map_get(m, ptr_hash(key), key, ptr_eq);
}

static void replace_set(PolyMap *m, PolyUOp *key, PolyUOp *val) {
  poly_map_set(m, ptr_hash(key), key, val, ptr_eq);
}

/* Active graph_rewrite user context for callbacks that need pass-local state. */
static void *g_graph_rewrite_userctx = NULL;

void *poly_graph_rewrite_userctx(void) {
  return g_graph_rewrite_userctx;
}

PolyUOp *poly_graph_rewrite_ctx_ex(PolyCtx *ctx, PolyUOp *sink, PolyPatternMatcher *pm,
                                   void *user_ctx, bool bottom_up) {
  void *prev_userctx = g_graph_rewrite_userctx;
  g_graph_rewrite_userctx = user_ctx;

  PolyMap *replace = poly_map_new(256);
  PolyMap *on_stack = poly_map_new(256);
  PolyMap *waitlist = poly_map_new(64);
  WorkStack ws = { NULL, 0, 0 };

  /* Mark root as on_stack and push */
  poly_map_set(on_stack, ptr_hash(sink), sink, (void*)(uintptr_t)1, ptr_eq);
  ws_push(&ws, sink, 0, sink);

  while (ws.top > 0) {
    if (ws.top > REWRITE_STACK_LIMIT) {
      fprintf(stderr, "polygrad: graph_rewrite stack overflow\n");
      break;
    }
    WorkItem wi = ws.items[--ws.top];
    PolyUOp *n = wi.n, *new_n = wi.new_n;
    int stage = wi.stage;

    /* Skip if already done */
    if (replace_get(replace, n)) continue;

    if (stage == 0) {
      /* Bottom-up: apply matcher first to a fixed point, then descend. */
      if (bottom_up && pm) {
        PolyUOp *cur = new_n;
        for (int iter = 0; iter < 4096; iter++) {
          PolyUOp *next = poly_pm_rewrite(pm, ctx, cur);
          if (!next || next == cur) break;
          cur = next;
          if (iter == 4095)
            fprintf(stderr, "polygrad: graph_rewrite fixed-point iteration limit hit\n");
        }
        new_n = cur;
      }

      /* Stage 1 rebuilds from rewritten sources and applies top-down rewrite. */
      ws_push(&ws, n, 1, new_n);
      for (int i = new_n->n_src - 1; i >= 0; i--) {
        PolyUOp *x = new_n->src[i];
        if (poly_map_get(on_stack, ptr_hash(x), x, ptr_eq)) continue;
        poly_map_set(on_stack, ptr_hash(x), x, (void*)(uintptr_t)1, ptr_eq);
        ws_push(&ws, x, 0, x);
      }
    } else if (stage == 1) {
      /* All sources should be rewritten. Collect them. */
      bool all_ready = true;
      bool any_changed = false;
      bool heap_src = (new_n->n_src > 16);
      PolyUOp *new_src_buf[16];
      PolyUOp **new_src = heap_src ?
        malloc(new_n->n_src * sizeof(PolyUOp *)) : new_src_buf;

      for (int i = 0; i < new_n->n_src; i++) {
        PolyUOp *rx = replace_get(replace, new_n->src[i]);
        if (!rx) {
          waitlist_add(waitlist, new_n->src[i], (WorkItem){ n, 1, new_n });
          all_ready = false;
          break;
        }
        new_src[i] = rx;
        if (rx != new_n->src[i]) any_changed = true;
      }
      if (!all_ready) { if (heap_src) free(new_src); continue; }

      PolyUOp *new_src_n;
      if (!any_changed) {
        new_src_n = bottom_up ? NULL : poly_pm_rewrite(pm, ctx, new_n);
        if (!new_src_n || new_src_n == new_n) {
          replace_set(replace, n, new_n);
          waitlist_flush(waitlist, n, &ws);
          if (heap_src) free(new_src);
          continue;
        }
      } else {
        new_src_n = poly_uop(ctx, new_n->op, new_n->dtype,
                             new_src, new_n->n_src, new_n->arg);
      }
      if (heap_src) free(new_src);

      /* Push the new node for full rewrite, then link back in stage 2 */
      ws_push(&ws, n, 2, new_src_n);
      ws_push(&ws, new_src_n, 0, new_src_n);
    } else {
      /* Stage 2: link n -> result of new_n */
      PolyUOp *replaced = replace_get(replace, new_n);
      if (!replaced) {
        waitlist_add(waitlist, new_n, (WorkItem){ n, 2, new_n });
      } else {
        replace_set(replace, n, replaced);
        waitlist_flush(waitlist, n, &ws);
      }
    }
  }

  PolyUOp *result = replace_get(replace, sink);

  free(ws.items);
  poly_map_destroy(replace);
  poly_map_destroy(on_stack);
  /* Free any remaining waitlist nodes */
  poly_map_destroy(waitlist);

  g_graph_rewrite_userctx = prev_userctx;
  return result ? result : sink;
}

PolyUOp *poly_graph_rewrite_ctx(PolyCtx *ctx, PolyUOp *sink, PolyPatternMatcher *pm,
                                void *user_ctx) {
  return poly_graph_rewrite_ctx_ex(ctx, sink, pm, user_ctx, false);
}

PolyUOp *poly_graph_rewrite(PolyCtx *ctx, PolyUOp *sink, PolyPatternMatcher *pm) {
  return poly_graph_rewrite_ctx_ex(ctx, sink, pm, NULL, false);
}

PolyUOp *poly_graph_rewrite_ex(PolyCtx *ctx, PolyUOp *sink, PolyPatternMatcher *pm,
                               bool bottom_up) {
  return poly_graph_rewrite_ctx_ex(ctx, sink, pm, NULL, bottom_up);
}

/* ── UOp helpers for rewrite callbacks ────────────────────────────────── */

PolyUOp *poly_const_like(PolyCtx *ctx, PolyUOp *ref, PolyArg val) {
  return poly_uop0(ctx, POLY_OP_CONST, ref->dtype, val);
}

PolyUOp *poly_const_like_int(PolyCtx *ctx, PolyUOp *ref, int64_t val) {
  if (poly_dtype_is_float(ref->dtype))
    return poly_uop0(ctx, POLY_OP_CONST, ref->dtype, poly_arg_float((double)val));
  if (poly_dtype_is_bool(ref->dtype))
    return poly_uop0(ctx, POLY_OP_CONST, ref->dtype, poly_arg_bool(val != 0));
  return poly_uop0(ctx, POLY_OP_CONST, ref->dtype, poly_arg_int(val));
}

PolyUOp *poly_const_like_float(PolyCtx *ctx, PolyUOp *ref, double val) {
  if (poly_dtype_is_int(ref->dtype))
    return poly_uop0(ctx, POLY_OP_CONST, ref->dtype, poly_arg_int((int64_t)val));
  if (poly_dtype_is_bool(ref->dtype))
    return poly_uop0(ctx, POLY_OP_CONST, ref->dtype, poly_arg_bool(val != 0.0));
  return poly_uop0(ctx, POLY_OP_CONST, ref->dtype, poly_arg_float(val));
}

PolyUOp *poly_const_like_bool(PolyCtx *ctx, PolyUOp *ref, bool val) {
  return poly_uop0(ctx, POLY_OP_CONST, ref->dtype, poly_arg_bool(val));
}
