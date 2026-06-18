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
#include "utils.h"

/* OpSet helpers */

static int opset_popcount(PolyOpSet s) {
  return __builtin_popcountll(s.bits[0]) + __builtin_popcountll(s.bits[1]);
}

static PolyOps opset_first(PolyOpSet s) {
  if (s.bits[0]) return (PolyOps)__builtin_ctzll(s.bits[0]);
  if (s.bits[1]) return (PolyOps)(64 + __builtin_ctzll(s.bits[1]));
  return (PolyOps)0;
}

/* Pattern constructors */

static PolyPat *pat_alloc(void) {
  PolyPat *p = calloc(1, sizeof(PolyPat));
  return p;
}

static PolyOpSet compute_early_reject(PolyPat **src, int n_src) {
  PolyOpSet rej = {{0, 0}};
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
  p->ops = poly_opset_add(poly_opset_add((PolyOpSet){{0, 0}}, POLY_OP_CONST), POLY_OP_VCONST);
  p->name = name;
  return p;
}

PolyPat *poly_pat_const_val(PolyArg val) {
  PolyPat *p = pat_alloc();
  p->has_ops = true;
  p->ops = poly_opset_add((PolyOpSet){{0, 0}}, POLY_OP_CONST);
  p->match_arg = true;
  p->arg = val;
  return p;
}

PolyPat *poly_pat_op(PolyOps op, PolyPat **src, int n_src, const char *name) {
  PolyPat *p = pat_alloc();
  p->has_ops = true;
  p->ops = poly_opset_add((PolyOpSet){{0, 0}}, op);
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
  PolyPat *arr[] = {s0};
  return poly_pat_op(op, arr, 1, name);
}

PolyPat *poly_pat_op2(PolyOps op, PolyPat *s0, PolyPat *s1, const char *name) {
  PolyPat *arr[] = {s0, s1};
  return poly_pat_op(op, arr, 2, name);
}

PolyPat *poly_pat_op2c(PolyOps op, PolyPat *s0, PolyPat *s1, const char *name) {
  PolyPat *arr[] = {s0, s1};
  PolyPat *p = poly_pat_op(op, arr, 2, name);
  p->commutative = true;
  return p;
}

PolyPat *poly_pat_op3(PolyOps op, PolyPat *s0, PolyPat *s1, PolyPat *s2, const char *name) {
  PolyPat *arr[] = {s0, s1, s2};
  return poly_pat_op(op, arr, 3, name);
}

PolyPat *poly_pat_ops1(PolyOpSet ops, PolyPat *s0, const char *name) {
  PolyPat *arr[] = {s0};
  return poly_pat_ops(ops, arr, 1, name);
}

PolyPat *poly_pat_ops2(PolyOpSet ops, PolyPat *s0, PolyPat *s1, const char *name) {
  PolyPat *arr[] = {s0, s1};
  return poly_pat_ops(ops, arr, 2, name);
}

PolyPat *poly_pat_ops3(PolyOpSet ops, PolyPat *s0, PolyPat *s1, PolyPat *s2, const char *name) {
  PolyPat *arr[] = {s0, s1, s2};
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

/* Pattern matching */

static const char *bindings_name_at(const PolyBindings *b, int idx) {
  if (!b || idx < 0 || idx >= b->n) return NULL;
  if (idx < POLY_BINDINGS_INLINE) return b->names[idx];
  return b->extra[idx - POLY_BINDINGS_INLINE].name;
}

static PolyUOp *bindings_uop_at(const PolyBindings *b, int idx) {
  if (!b || idx < 0 || idx >= b->n) return NULL;
  if (idx < POLY_BINDINGS_INLINE) return b->uops[idx];
  return b->extra[idx - POLY_BINDINGS_INLINE].uop;
}

PolyUOp *poly_bind(const PolyBindings *b, const char *name) {
  if (!b || !name) return NULL;
  for (int i = 0; i < b->n; i++) {
    const char *binding_name = bindings_name_at(b, i);
    if (binding_name == name || (binding_name && strcmp(binding_name, name) == 0))
      return bindings_uop_at(b, i);
  }
  return NULL;
}

void poly_bindings_free(PolyBindings *b) {
  if (!b) return;
  free(b->extra);
  b->extra = NULL;
  b->extra_cap = 0;
  b->n = 0;
}

static bool bindings_add(PolyBindings *b, const char *name, PolyUOp *uop) {
  if (!b || !name || !uop) return false;
  if (b->n < POLY_BINDINGS_INLINE) {
    b->names[b->n] = name;
    b->uops[b->n] = uop;
    b->n++;
    return true;
  }

  int extra_idx = b->n - POLY_BINDINGS_INLINE;
  if (extra_idx >= b->extra_cap) {
    int new_cap = b->extra_cap ? b->extra_cap * 2 : 16;
    PolyBindingEntry *new_extra = realloc(b->extra, (size_t)new_cap * sizeof(PolyBindingEntry));
    if (!new_extra) return false;
    b->extra = new_extra;
    b->extra_cap = new_cap;
  }

  b->extra[extra_idx] = (PolyBindingEntry){.name = name, .uop = uop};
  b->n++;
  return true;
}

static bool match_sources(const PolyPat *pat, PolyUOp *uop, PolyBindings *binds) {
  for (int i = 0; i < pat->n_src; i++) {
    if (!poly_pat_match(pat->src[i], uop->src[i], binds)) return false;
  }
  return true;
}

bool poly_pat_match(const PolyPat *pat, PolyUOp *uop, PolyBindings *binds) {
  /* CAST-tolerant match: pattern or CAST(pattern). */
  if (pat->or_casted && uop->op == POLY_OP_CAST && uop->n_src == 1) {
    int saved = binds->n;
    if (poly_pat_match(pat, uop->src[0], binds)) return true;
    binds->n = saved;
  }

  /* Op check */
  if (pat->has_ops && !poly_opset_has(pat->ops, uop->op)) return false;

  /* Name binding: if already bound, must be same pointer */
  if (pat->name) {
    PolyUOp *existing = poly_bind(binds, pat->name);
    if (existing) {
      if (existing != uop) return false;
    } else {
      if (!bindings_add(binds, pat->name, uop)) return false;
    }
  }

  /* DType check */
  if (pat->n_dtypes > 0) {
    bool found = false;
    PolyDType scalar = poly_dtype_scalar(uop->dtype);
    for (int i = 0; i < pat->n_dtypes; i++) {
      if (poly_dtype_eq(pat->dtypes[i], uop->dtype) || poly_dtype_eq(pat->dtypes[i], scalar)) {
        found = true;
        break;
      }
    }
    if (!found) return false;
  }

  /* Arg check */
  if (pat->match_arg && !poly_arg_eq(pat->arg, uop->arg)) return false;

  /* Source count */
  if (uop->n_src < pat->n_src) return false;
  if (pat->strict_length && uop->n_src != pat->n_src) return false;

  /* No source constraints = done */
  if (pat->src == NULL) return true;

  /* Commutative: try both orderings */
  if (pat->commutative && pat->n_src == 2) {
    int saved = binds->n;
    if (match_sources(pat, uop, binds)) return true;
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

/* PatternMatcher */

struct PolyPatternMatcher {
  PolyRule *rules;
  int n_rules;
  struct {
    int *indices;
    int n;
    int cap;
  } by_op[POLY_OP_COUNT];
};

static void pm_add_op(PolyPatternMatcher *pm, int op, int rule_idx) {
  if (op < 0 || op >= POLY_OP_COUNT) return;
  if (pm->by_op[op].n >= pm->by_op[op].cap) {
    pm->by_op[op].cap = pm->by_op[op].cap ? pm->by_op[op].cap * 2 : 8;
    pm->by_op[op].indices = realloc(pm->by_op[op].indices, pm->by_op[op].cap * sizeof(int));
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
        if (poly_opset_has(p->ops, (PolyOps)j)) pm_add_op(pm, j, i);
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
  if (op < 0 || op >= POLY_OP_COUNT || pm->by_op[op].n == 0) return NULL;

  /* Compute src ops bitmask for early reject */
  PolyOpSet src_ops = {{0, 0}};
  for (int i = 0; i < uop->n_src; i++)
    src_ops = poly_opset_add(src_ops, uop->src[i]->op);

  for (int i = 0; i < pm->by_op[op].n; i++) {
    int idx = pm->by_op[op].indices[i];
    PolyRule *rule = &pm->rules[idx];

    /* Early reject: required ops must appear in sources */
    if (!poly_opset_subset(rule->pat->early_reject, src_ops)) continue;

    PolyBindings binds = {.n = 0};
    if (poly_pat_match(rule->pat, uop, &binds)) {
      PolyUOp *result = rule->fn(ctx, uop, &binds);
      poly_bindings_free(&binds);
      if (result != NULL && result != uop) return result;
    } else {
      poly_bindings_free(&binds);
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

/* graph_rewrite (top-down unified_rewrite) */

/* Pointer hash/eq for maps */

/* Worklist entry */
typedef struct {
  PolyUOp *n; /* original node */
  int stage; /* 0, 1, or 2 */
  PolyUOp *new_n; /* potentially rewritten node */
} WorkItem;

/* Dynamic stack */
typedef struct {
  WorkItem *items;
  int top, cap;
} WorkStack;

static bool ws_push(WorkStack *ws, PolyUOp *n, int stage, PolyUOp *new_n) {
  if (ws->top >= ws->cap) {
    int new_cap = ws->cap ? ws->cap * 2 : 256;
    WorkItem *new_items = realloc(ws->items, (size_t)new_cap * sizeof(WorkItem));
    if (!new_items) return false;
    ws->items = new_items;
    ws->cap = new_cap;
  }
  ws->items[ws->top++] = (WorkItem){n, stage, new_n};
  return true;
}

/* Waitlist: linked list of work items per UOp key */
typedef struct WaitNode {
  WorkItem item;
  struct WaitNode *next;
} WaitNode;

static bool waitlist_add(PolyMap *wl, PolyUOp *key, WorkItem item) {
  uint32_t h = poly_ptr_hash(key);
  WaitNode *node = malloc(sizeof(WaitNode));
  if (!node) return false;
  node->item = item;
  node->next = poly_map_get(wl, h, key, poly_ptr_eq);
  poly_map_set(wl, h, key, node, poly_ptr_eq);
  return true;
}

static void waitlist_free_chain(WaitNode *chain) {
  while (chain) {
    WaitNode *next = chain->next;
    free(chain);
    chain = next;
  }
}

static bool waitlist_flush(PolyMap *wl, PolyUOp *key, WorkStack *ws) {
  uint32_t h = poly_ptr_hash(key);
  WaitNode *chain = poly_map_get(wl, h, key, poly_ptr_eq);
  while (chain) {
    WaitNode *next = chain->next;
    bool ok = ws_push(ws, chain->item.n, chain->item.stage, chain->item.new_n);
    free(chain);
    if (!ok) {
      waitlist_free_chain(next);
      poly_map_remove(wl, h, key, poly_ptr_eq);
      return false;
    }
    chain = next;
  }
  poly_map_remove(wl, h, key, poly_ptr_eq);
  return true;
}

static void waitlist_free_entry(const void *key, void *value, void *userdata) {
  (void)key;
  (void)userdata;
  waitlist_free_chain((WaitNode *)value);
}

static void waitlist_destroy(PolyMap *wl) {
  if (!wl) return;
  poly_map_foreach(wl, waitlist_free_entry, NULL);
  poly_map_destroy(wl);
}

#define REWRITE_STACK_LIMIT 100000

static int rewrite_stack_limit(void) {
  int limit = poly_getenv_int("POLY_REWRITE_STACK_LIMIT", REWRITE_STACK_LIMIT);
  return limit > 0 ? limit : REWRITE_STACK_LIMIT;
}

static PolyUOp *replace_get(PolyMap *m, PolyUOp *key) {
  return poly_map_get(m, poly_ptr_hash(key), key, poly_ptr_eq);
}

static void replace_set(PolyMap *m, PolyUOp *key, PolyUOp *val) {
  poly_map_set(m, poly_ptr_hash(key), key, val, poly_ptr_eq);
}

/* Active graph_rewrite user context for callbacks that need pass-local state. */
static _Thread_local void *g_graph_rewrite_userctx = NULL;

void *poly_graph_rewrite_userctx(void) {
  return g_graph_rewrite_userctx;
}

PolyUOp *poly_graph_rewrite_ctx_ex2(
    PolyCtx *ctx,
    PolyUOp *sink,
    PolyPatternMatcher *pm,
    void *user_ctx,
    bool bottom_up,
    bool enter_calls
) {
  if (!ctx || !sink) return NULL;

  void *prev_userctx = g_graph_rewrite_userctx;
  g_graph_rewrite_userctx = user_ctx;

  PolyMap *replace = poly_map_new(256);
  PolyMap *on_stack = poly_map_new(256);
  PolyMap *waitlist = poly_map_new(64);
  WorkStack ws = {NULL, 0, 0};
  bool failed = false;
  PolyUOp *result = NULL;
  int stack_limit = rewrite_stack_limit();

  if (!replace || !on_stack || !waitlist) {
    fprintf(stderr, "polygrad: graph_rewrite allocation failure\n");
    failed = true;
    goto cleanup;
  }

  /* Mark root as on_stack and push. */
  poly_map_set(on_stack, poly_ptr_hash(sink), sink, (void *)(uintptr_t)1, poly_ptr_eq);
  if (!ws_push(&ws, sink, 0, sink)) {
    fprintf(stderr, "polygrad: graph_rewrite allocation failure\n");
    failed = true;
    goto cleanup;
  }

  while (ws.top > 0) {
    if (ws.top > stack_limit) {
      fprintf(stderr, "polygrad: graph_rewrite stack overflow\n");
      failed = true;
      break;
    }
    WorkItem wi = ws.items[--ws.top];
    PolyUOp *n = wi.n, *new_n = wi.new_n;
    int stage = wi.stage;

    /* Skip if already done. */
    if (replace_get(replace, n)) continue;

    if (stage == 0) {
      /* Bottom-up: apply matcher first to a fixed point, then descend.
       * tinygrad tracks seen UOps and raises on repeated fixed-point state;
       * in C the equivalent hard failure is returning NULL from this pass. */
      if (bottom_up && pm) {
        PolyUOp *cur = new_n;
        PolyMap *seen = poly_map_new(16);
        if (!seen) {
          fprintf(stderr, "polygrad: graph_rewrite allocation failure\n");
          failed = true;
          goto cleanup;
        }
        while (cur) {
          if (poly_map_get(seen, poly_ptr_hash(cur), cur, poly_ptr_eq)) {
            fprintf(stderr, "polygrad: graph_rewrite fixed-point cycle\n");
            poly_map_destroy(seen);
            failed = true;
            goto cleanup;
          }
          poly_map_set(seen, poly_ptr_hash(cur), cur, (void *)(uintptr_t)1, poly_ptr_eq);
          PolyUOp *next = poly_pm_rewrite(pm, ctx, cur);
          if (!next || next == cur) break;
          cur = next;
        }
        poly_map_destroy(seen);
        new_n = cur;
      }

      /* CALL gating: when enter_calls=false, set identity mappings for the
       * entire callee subgraph (src[0]) so stage 1 lookups resolve immediately
       * instead of stalling in the waitlist. */
      if (!enter_calls && new_n->op == POLY_OP_CALL && new_n->n_src > 0) {
        PolyUOp *callee = new_n->src[0];
        int n_callee = 0;
        PolyUOp **callee_topo = poly_toposort(ctx, callee, &n_callee);
        if (!callee_topo && n_callee > 0) {
          fprintf(stderr, "polygrad: graph_rewrite callee toposort failed\n");
          failed = true;
          goto cleanup;
        }
        for (int ci = 0; ci < n_callee; ci++)
          replace_set(replace, callee_topo[ci], callee_topo[ci]);
      }

      /* Stage 1 rebuilds from rewritten sources and applies top-down rewrite. */
      if (!ws_push(&ws, n, 1, new_n)) {
        fprintf(stderr, "polygrad: graph_rewrite allocation failure\n");
        failed = true;
        goto cleanup;
      }
      int src_start = (!enter_calls && new_n->op == POLY_OP_CALL && new_n->n_src > 1) ? 1 : 0;
      for (int i = new_n->n_src - 1; i >= src_start; i--) {
        PolyUOp *x = new_n->src[i];
        if (poly_map_get(on_stack, poly_ptr_hash(x), x, poly_ptr_eq)) continue;
        poly_map_set(on_stack, poly_ptr_hash(x), x, (void *)(uintptr_t)1, poly_ptr_eq);
        if (!ws_push(&ws, x, 0, x)) {
          fprintf(stderr, "polygrad: graph_rewrite allocation failure\n");
          failed = true;
          goto cleanup;
        }
      }
    } else if (stage == 1) {
      /* All sources should be rewritten. Collect them. */
      bool all_ready = true;
      bool any_changed = false;
      bool heap_src = (new_n->n_src > 16);
      PolyUOp *new_src_buf[16];
      PolyUOp **new_src = heap_src ? malloc((size_t)new_n->n_src * sizeof(PolyUOp *)) : new_src_buf;
      if (heap_src && !new_src) {
        fprintf(stderr, "polygrad: graph_rewrite allocation failure\n");
        failed = true;
        goto cleanup;
      }

      for (int i = 0; i < new_n->n_src; i++) {
        PolyUOp *rx = replace_get(replace, new_n->src[i]);
        if (!rx) {
          if (!waitlist_add(waitlist, new_n->src[i], (WorkItem){n, 1, new_n})) {
            fprintf(stderr, "polygrad: graph_rewrite allocation failure\n");
            if (heap_src) free(new_src);
            failed = true;
            goto cleanup;
          }
          all_ready = false;
          break;
        }
        new_src[i] = rx;
        if (rx != new_n->src[i]) any_changed = true;
      }
      if (!all_ready) {
        if (heap_src) free(new_src);
        continue;
      }

      PolyUOp *new_src_n;
      if (!any_changed) {
        new_src_n = bottom_up ? NULL : poly_pm_rewrite(pm, ctx, new_n);
        if (!new_src_n || new_src_n == new_n) {
          replace_set(replace, n, new_n);
          if (!waitlist_flush(waitlist, n, &ws)) {
            fprintf(stderr, "polygrad: graph_rewrite allocation failure\n");
            if (heap_src) free(new_src);
            failed = true;
            goto cleanup;
          }
          if (heap_src) free(new_src);
          continue;
        }
      } else {
        new_src_n =
            (new_n->tag != 0)
                ? poly_uop_tagged(
                      ctx, new_n->op, new_n->dtype, new_src, new_n->n_src, new_n->arg, new_n->tag
                  )
                : poly_uop(ctx, new_n->op, new_n->dtype, new_src, new_n->n_src, new_n->arg);
      }
      if (heap_src) free(new_src);

      /* Push the new node for full rewrite, then link back in stage 2. */
      if (!ws_push(&ws, n, 2, new_src_n) || !ws_push(&ws, new_src_n, 0, new_src_n)) {
        fprintf(stderr, "polygrad: graph_rewrite allocation failure\n");
        failed = true;
        goto cleanup;
      }
    } else {
      /* Stage 2: link n -> result of new_n. */
      PolyUOp *replaced = replace_get(replace, new_n);
      if (!replaced) {
        if (!waitlist_add(waitlist, new_n, (WorkItem){n, 2, new_n})) {
          fprintf(stderr, "polygrad: graph_rewrite allocation failure\n");
          failed = true;
          goto cleanup;
        }
      } else {
        replace_set(replace, n, replaced);
        if (!waitlist_flush(waitlist, n, &ws)) {
          fprintf(stderr, "polygrad: graph_rewrite allocation failure\n");
          failed = true;
          goto cleanup;
        }
      }
    }
  }

  if (!failed && waitlist && poly_map_len(waitlist) > 0) {
    fprintf(stderr, "polygrad: graph_rewrite unresolved waitlist\n");
    failed = true;
  }

  if (!failed) {
    result = replace_get(replace, sink);
    if (!result) {
      fprintf(stderr, "polygrad: graph_rewrite root was not rewritten\n");
      failed = true;
    }
  }

cleanup:
  free(ws.items);
  if (replace) poly_map_destroy(replace);
  if (on_stack) poly_map_destroy(on_stack);
  waitlist_destroy(waitlist);

  g_graph_rewrite_userctx = prev_userctx;
  return failed ? NULL : result;
}

/* walk_rewrite: MLIR-style single-pass, no re-traversal */

PolyUOp *poly_graph_walk_rewrite(
    PolyCtx *ctx,
    PolyUOp *sink,
    PolyPatternMatcher *pm,
    PolyPatternMatcher *bpm,
    void *user_ctx,
    bool enter_calls
) {
  void *prev_userctx = g_graph_rewrite_userctx;
  g_graph_rewrite_userctx = user_ctx;

  PolyMap *replace = poly_map_new(256);
  WorkStack ws = {NULL, 0, 0};

  if (!ws_push(&ws, sink, 0, sink)) {
    g_graph_rewrite_userctx = prev_userctx;
    free(ws.items);
    poly_map_destroy(replace);
    return NULL;
  }

  while (ws.top > 0) {
    WorkItem wi = ws.items[--ws.top];
    PolyUOp *n = wi.n;

    if (replace_get(replace, n)) continue;

    if (wi.stage == 0) {
      /* Bottom-up: try bpm first. If it rewrites, use result as-is
       * (no descent into replacement — this is the key walk_rewrite property). */
      if (bpm) {
        PolyUOp *r = poly_pm_rewrite(bpm, ctx, n);
        if (r && r != n) {
          replace_set(replace, n, r);
          continue;
        }
      }

      /* Push for rebuild, then push children */
      if (!ws_push(&ws, n, 1, n)) {
        free(ws.items);
        poly_map_destroy(replace);
        g_graph_rewrite_userctx = prev_userctx;
        return NULL;
      }

      /* CALL gating: identity-map entire callee subgraph */
      if (!enter_calls && n->op == POLY_OP_CALL && n->n_src > 0) {
        int n_callee = 0;
        PolyUOp **ct = poly_toposort(ctx, n->src[0], &n_callee);
        for (int i = 0; i < n_callee; i++)
          replace_set(replace, ct[i], ct[i]);
      }

      int start = (!enter_calls && n->op == POLY_OP_CALL && n->n_src > 1) ? 1 : 0;
      for (int i = n->n_src - 1; i >= start; i--) {
        if (!replace_get(replace, n->src[i]) && !ws_push(&ws, n->src[i], 0, n->src[i])) {
          free(ws.items);
          poly_map_destroy(replace);
          g_graph_rewrite_userctx = prev_userctx;
          return NULL;
        }
      }
    } else {
      /* Rebuild with rewritten sources */
      bool heap_src = (n->n_src > 16);
      PolyUOp *new_src_buf[16];
      PolyUOp **new_src = heap_src ? malloc(n->n_src * sizeof(PolyUOp *)) : new_src_buf;
      bool changed = false;

      for (int i = 0; i < n->n_src; i++) {
        PolyUOp *r = replace_get(replace, n->src[i]);
        new_src[i] = r ? r : n->src[i];
        if (new_src[i] != n->src[i]) changed = true;
      }

      PolyUOp *new_n;
      if (changed) {
        new_n = (n->tag != 0)
                    ? poly_uop_tagged(ctx, n->op, n->dtype, new_src, n->n_src, n->arg, n->tag)
                    : poly_uop(ctx, n->op, n->dtype, new_src, n->n_src, n->arg);
      } else {
        new_n = n;
      }

      /* Top-down: try pm on rebuilt node. Use result as-is (no re-traversal). */
      if (pm) {
        PolyUOp *r = poly_pm_rewrite(pm, ctx, new_n);
        if (r && r != new_n) new_n = r;
      }

      replace_set(replace, n, new_n);
      if (heap_src) free(new_src);
    }
  }

  PolyUOp *result = replace_get(replace, sink);

  free(ws.items);
  poly_map_destroy(replace);
  g_graph_rewrite_userctx = prev_userctx;
  return result ? result : sink;
}

PolyUOp *poly_graph_rewrite_ctx_ex(
    PolyCtx *ctx,
    PolyUOp *sink,
    PolyPatternMatcher *pm,
    void *user_ctx,
    bool bottom_up
) {
  return poly_graph_rewrite_ctx_ex2(ctx, sink, pm, user_ctx, bottom_up, true);
}

PolyUOp *poly_graph_rewrite_ctx(
    PolyCtx *ctx,
    PolyUOp *sink,
    PolyPatternMatcher *pm,
    void *user_ctx
) {
  return poly_graph_rewrite_ctx_ex2(ctx, sink, pm, user_ctx, false, true);
}

PolyUOp *poly_graph_rewrite(PolyCtx *ctx, PolyUOp *sink, PolyPatternMatcher *pm) {
  return poly_graph_rewrite_ctx_ex2(ctx, sink, pm, NULL, false, true);
}

PolyUOp *poly_graph_rewrite_ex(
    PolyCtx *ctx,
    PolyUOp *sink,
    PolyPatternMatcher *pm,
    bool bottom_up
) {
  return poly_graph_rewrite_ctx_ex2(ctx, sink, pm, NULL, bottom_up, true);
}

/* UOp helpers for rewrite callbacks */

PolyUOp *poly_const_like(PolyCtx *ctx, PolyUOp *ref, PolyArg val) {
  /* Normalise the arg kind to match ref->dtype.
   *
   * Tinygrad parity: every const construction goes through DType.const(b)
   * which converts to ConstFloat(float(b)) / bool(b) / int(b) based on the
   * target dtype (uop/ops.py:498-499 -> dtype.py:92-100). polygrad must do
   * the same here, otherwise rule_cast_const (sym.c:534) would propagate
   * a CONST_INT(5) into a float-tagged CONST and the codegen would mis-
   * interpret arg.i as arg.f, lowering it to a denormal/zero. Verified
   * against test/parity_scripts/tg_cast_const_fold_gt.py cases A-E. */
  if (val.kind == POLY_ARG_INT || val.kind == POLY_ARG_FLOAT || val.kind == POLY_ARG_BOOL) {
    if (poly_dtype_is_float(ref->dtype)) {
      double dval = (val.kind == POLY_ARG_INT)    ? (double)val.i
                    : (val.kind == POLY_ARG_BOOL) ? (val.b ? 1.0 : 0.0)
                                                  : val.f;
      return poly_uop0(ctx, POLY_OP_CONST, ref->dtype, poly_arg_float(dval));
    }
    if (poly_dtype_is_bool(ref->dtype)) {
      bool bval = (val.kind == POLY_ARG_INT)    ? (val.i != 0)
                  : (val.kind == POLY_ARG_BOOL) ? val.b
                                                : (val.f != 0.0);
      return poly_uop0(ctx, POLY_OP_CONST, ref->dtype, poly_arg_bool(bval));
    }
    int64_t ival = (val.kind == POLY_ARG_INT)    ? val.i
                   : (val.kind == POLY_ARG_BOOL) ? (val.b ? 1 : 0)
                                                 : (int64_t)val.f;
    return poly_uop0(ctx, POLY_OP_CONST, ref->dtype, poly_arg_int(ival));
  }
  /* Non-numeric arg kinds (NONE, OPS, RANGE, etc.) are passed through as-is
   * for callers like rule_const_fold_unary that synthesise the right kind
   * via poly_exec_alu. */
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
