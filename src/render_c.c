/*
 * render_c.c — Linearizer + C code renderer
 *
 * Linearizer: priority-based toposort matching tinygrad's linearizer.py.
 * Renderer: walks linearized UOps, emits C source (ClangRenderer port).
 */

#define _POSIX_C_SOURCE 200809L

#include "codegen.h"
#include "pat.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>
#include <assert.h>

/* ── String builder ──────────────────────────────────────────────────── */

typedef struct {
  char *buf;
  int len;
  int cap;
} StrBuf;

static void sb_init(StrBuf *sb) {
  sb->cap = 512;
  sb->buf = malloc(sb->cap);
  sb->buf[0] = '\0';
  sb->len = 0;
}

static void sb_printf(StrBuf *sb, const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  int need = vsnprintf(NULL, 0, fmt, ap);
  va_end(ap);

  while (sb->len + need + 1 > sb->cap) {
    sb->cap *= 2;
    sb->buf = realloc(sb->buf, sb->cap);
  }

  va_start(ap, fmt);
  sb->len += vsnprintf(sb->buf + sb->len, sb->cap - sb->len, fmt, ap);
  va_end(ap);
}

static void sb_puts(StrBuf *sb, const char *s) {
  sb_printf(sb, "%s", s);
}

/* ── Pointer → int hash map (for linearizer) ─────────────────────────── */

typedef struct {
  PolyUOp **keys;
  int *vals;
  int cap;
} IntMap;

static uint32_t ptr_hash_mix(const void *p) {
  uintptr_t v = (uintptr_t)p;
  v = ((v >> 16) ^ v) * 0x45d9f3bU;
  v = ((v >> 16) ^ v) * 0x45d9f3bU;
  return (uint32_t)((v >> 16) ^ v);
}

static void imap_init(IntMap *m, int n) {
  m->cap = (n < 4) ? 16 : n * 3;
  m->keys = calloc(m->cap, sizeof(PolyUOp *));
  m->vals = calloc(m->cap, sizeof(int));
}

static void imap_set(IntMap *m, PolyUOp *key, int val) {
  uint32_t h = ptr_hash_mix(key) % m->cap;
  while (m->keys[h] && m->keys[h] != key) h = (h + 1) % m->cap;
  m->keys[h] = key;
  m->vals[h] = val;
}

static int imap_get(IntMap *m, PolyUOp *key) {
  uint32_t h = ptr_hash_mix(key) % m->cap;
  while (m->keys[h] != key) h = (h + 1) % m->cap;
  return m->vals[h];
}

static int imap_try_get(IntMap *m, PolyUOp *key) {
  uint32_t h = ptr_hash_mix(key) % m->cap;
  while (m->keys[h]) {
    if (m->keys[h] == key) return m->vals[h];
    h = (h + 1) % m->cap;
  }
  return -1;
}

static void imap_destroy(IntMap *m) {
  free(m->keys);
  free(m->vals);
}

/* ── Pointer → string hash map (for renderer) ────────────────────────── */

typedef struct {
  PolyUOp **keys;
  char **vals;
  int cap;
} StrMap;

static void smap_init(StrMap *m, int n) {
  m->cap = (n < 4) ? 16 : n * 3;
  m->keys = calloc(m->cap, sizeof(PolyUOp *));
  m->vals = calloc(m->cap, sizeof(char *));
}

static void smap_set(StrMap *m, PolyUOp *key, char *val) {
  uint32_t h = ptr_hash_mix(key) % m->cap;
  while (m->keys[h] && m->keys[h] != key) h = (h + 1) % m->cap;
  if (m->keys[h] == key) free(m->vals[h]); /* replace existing */
  m->keys[h] = key;
  m->vals[h] = val;
}

static char *smap_get(StrMap *m, PolyUOp *key) {
  uint32_t h = ptr_hash_mix(key) % m->cap;
  while (m->keys[h]) {
    if (m->keys[h] == key) return m->vals[h];
    h = (h + 1) % m->cap;
  }
  return NULL;
}

static void smap_destroy(StrMap *m) {
  for (int i = 0; i < m->cap; i++)
    if (m->vals[i]) free(m->vals[i]);
  free(m->keys);
  free(m->vals);
}

/* ── Min-heap (for linearizer reverse toposort) ──────────────────────── */

typedef struct {
  int *keys;
  PolyUOp **vals;
  int len;
  int cap;
} Heap;

static void heap_init(Heap *h, int cap) {
  h->keys = malloc(cap * sizeof(int));
  h->vals = malloc(cap * sizeof(PolyUOp *));
  h->len = 0;
  h->cap = cap;
}

static void heap_push(Heap *h, int key, PolyUOp *val) {
  int i = h->len++;
  h->keys[i] = key;
  h->vals[i] = val;
  while (i > 0) {
    int p = (i - 1) / 2;
    if (h->keys[p] <= h->keys[i]) break;
    int tk = h->keys[i]; h->keys[i] = h->keys[p]; h->keys[p] = tk;
    PolyUOp *tv = h->vals[i]; h->vals[i] = h->vals[p]; h->vals[p] = tv;
    i = p;
  }
}

static PolyUOp *heap_pop(Heap *h) {
  PolyUOp *result = h->vals[0];
  h->len--;
  if (h->len > 0) {
    h->keys[0] = h->keys[h->len];
    h->vals[0] = h->vals[h->len];
    int i = 0;
    for (;;) {
      int l = 2*i + 1, r = 2*i + 2, s = i;
      if (l < h->len && h->keys[l] < h->keys[s]) s = l;
      if (r < h->len && h->keys[r] < h->keys[s]) s = r;
      if (s == i) break;
      int tk = h->keys[i]; h->keys[i] = h->keys[s]; h->keys[s] = tk;
      PolyUOp *tv = h->vals[i]; h->vals[i] = h->vals[s]; h->vals[s] = tv;
      i = s;
    }
  }
  return result;
}

static void heap_destroy(Heap *h) {
  free(h->keys);
  free(h->vals);
}

/* ── Linearizer ──────────────────────────────────────────────────────── */

static int op_priority(PolyOps op) {
  switch (op) {
    case POLY_OP_PARAM:       return -20;
    case POLY_OP_DEFINE_VAR:  return -19;
    case POLY_OP_DEFINE_LOCAL:return -18;
    case POLY_OP_DEFINE_REG:  return -17;
    /* NOTE: in the reference tinygrad commit, CONST has no special
     * priority — it falls through to default priority 0.  This is
     * critical for matching the tuplize-based ordering. */
    case POLY_OP_LOAD:        return -1;
    case POLY_OP_STORE:       return 1;
    case POLY_OP_RANGE:       return 5;
    case POLY_OP_END:         return -5;
    default:                 return 0;
  }
}

static inline void bitset_set(uint64_t *bs, int bit) {
  bs[(unsigned)bit >> 6] |= (uint64_t)1 << (bit & 63);
}

static inline bool bitset_has(const uint64_t *bs, int bit) {
  return ((bs[(unsigned)bit >> 6] >> (bit & 63)) & 1) != 0;
}

static inline void bitset_clear(uint64_t *bs, int bit) {
  bs[(unsigned)bit >> 6] &= ~((uint64_t)1 << (bit & 63));
}

/* Apply ended_ranges of a UOp to a ranges bitset (remove ended ranges).
 * Mirrors tinygrad's ended_ranges property + _ranges subtraction logic.
 * range_start = {BUFFERIZE:1, REDUCE:1, STORE:2, END:1}
 * AFTER: flatten([x.ended_ranges for x in src[1:]]) */
static void apply_uop_ended_ranges(uint64_t *r, PolyUOp *u, PolyUOp **topo,
                                    IntMap *idx, uint64_t *all_ranges, int words) {
  int rs = -1;
  switch (u->op) {
    case POLY_OP_STORE:  rs = 2; break;
    case POLY_OP_END:    rs = 1; break;
    case POLY_OP_REDUCE: rs = 1; break;
    case POLY_OP_AFTER:
      for (int j = 1; j < u->n_src; j++)
        apply_uop_ended_ranges(r, u->src[j], topo, idx, all_ranges, words);
      return;
    default: return;
  }
  for (int j = rs; j < u->n_src; j++) {
    int si = imap_try_get(idx, u->src[j]);
    if (si < 0) continue;
    if (topo[si]->op == POLY_OP_RANGE) {
      bitset_clear(r, si);
    } else {
      const uint64_t *er = all_ranges + (size_t)si * (size_t)words;
      for (int w = 0; w < words; w++) r[w] &= ~er[w];
    }
  }
}

static int dep_count_in_siblings(const uint64_t *deps, int words, int idx,
                                 const int *siblings, int n_siblings) {
  const uint64_t *d = deps + (size_t)idx * (size_t)words;
  int cnt = 0;
  for (int i = 0; i < n_siblings; i++)
    if (bitset_has(d, siblings[i])) cnt++;
  return cnt;
}

/* Build tinygrad-style control-flow edges:
 * when a RANGE has to be ordered after a sibling END, record extra_dep[idx].
 * The returned array is indexed by topo index; -1 means no extra dep. */
static int *build_control_edges(PolyUOp **topo, int n, IntMap *idx) {
  int *extra_dep = malloc((size_t)n * sizeof(int));
  if (!extra_dep) return NULL;
  for (int i = 0; i < n; i++) extra_dep[i] = -1;
  if (n == 0) return extra_dep;

  int words = (n + 63) / 64;
  uint64_t *deps = calloc((size_t)n * (size_t)words, sizeof(uint64_t));
  int *nest_parent = malloc((size_t)n * sizeof(int));
  int *siblings = malloc((size_t)n * sizeof(int));
  int *scores = malloc((size_t)n * sizeof(int));
  if (!deps || !nest_parent || !siblings || !scores) {
    free(deps);
    free(nest_parent);
    free(siblings);
    free(scores);
    return extra_dep;
  }
  for (int i = 0; i < n; i++) nest_parent[i] = -1;

  for (int i = 0; i < n; i++) {
    PolyUOp *u = topo[i];
    uint64_t *du = deps + (size_t)i * (size_t)words;

    /* deps[u] |= deps[src] */
    for (int j = 0; j < u->n_src; j++) {
      int si = imap_try_get(idx, u->src[j]);
      if (si < 0) continue;
      const uint64_t *ds = deps + (size_t)si * (size_t)words;
      for (int w = 0; w < words; w++) du[w] |= ds[w];
    }

    /* Build nesting map from END -> (END or SINK) parent. */
    if (u->op == POLY_OP_END || u->op == POLY_OP_SINK) {
      int parent_rng_idx = -1;
      if (u->op == POLY_OP_END && u->n_src > 1)
        parent_rng_idx = imap_try_get(idx, u->src[1]);

      for (int x = 0; x < n; x++) {
        if (nest_parent[x] != -1 || topo[x]->op != POLY_OP_END) continue;
        if (!bitset_has(du, x)) continue;

        bool ok = (u->op == POLY_OP_SINK);
        if (!ok && parent_rng_idx >= 0) {
          const uint64_t *dx = deps + (size_t)x * (size_t)words;
          ok = bitset_has(dx, parent_rng_idx);
        }
        if (ok) nest_parent[x] = i;
      }
    }

    if (u->op == POLY_OP_RANGE || u->op == POLY_OP_END) bitset_set(du, i);
  }

  /* For each parent, order sibling ENDs and create RANGE -> dependency edge. */
  for (int parent = 0; parent < n; parent++) {
    int n_siblings = 0;
    for (int j = 0; j < n; j++) {
      if (nest_parent[j] == parent) siblings[n_siblings++] = j;
    }
    if (n_siblings <= 0) continue;

    for (int i = 0; i < n_siblings; i++)
      scores[i] = dep_count_in_siblings(deps, words, siblings[i], siblings, n_siblings);

    /* Sort siblings by dependency count (stable insertion sort). */
    for (int i = 1; i < n_siblings; i++) {
      int k_idx = siblings[i];
      int k_score = scores[i];
      int j = i - 1;
      while (j >= 0 && scores[j] > k_score) {
        siblings[j + 1] = siblings[j];
        scores[j + 1] = scores[j];
        j--;
      }
      siblings[j + 1] = k_idx;
      scores[j + 1] = k_score;
    }

    if (topo[parent]->op == POLY_OP_SINK) {
      for (int i = 0; i + 1 < n_siblings; i++) {
        int x_idx = siblings[i];
        PolyUOp *y = topo[siblings[i + 1]];
        if (y->n_src <= 1) continue;
        int y_range = imap_try_get(idx, y->src[1]);
        if (y_range < 0 || topo[y_range]->op != POLY_OP_RANGE) continue;
        const uint64_t *dx = deps + (size_t)x_idx * (size_t)words;
        if (x_idx == y_range || bitset_has(dx, y_range)) continue;
        if (extra_dep[y_range] < 0) extra_dep[y_range] = x_idx;
      }
    } else if (topo[parent]->op == POLY_OP_END && topo[parent]->n_src > 1) {
      int first = imap_try_get(idx, topo[parent]->src[1]);
      if (first < 0) continue;
      for (int i = 0; i < n_siblings; i++) {
        int x_idx = (i == 0) ? first : siblings[i - 1];
        PolyUOp *y = topo[siblings[i]];
        if (y->n_src <= 1) continue;
        int y_range = imap_try_get(idx, y->src[1]);
        if (y_range < 0 || topo[y_range]->op != POLY_OP_RANGE) continue;
        const uint64_t *dx = deps + (size_t)x_idx * (size_t)words;
        if (x_idx == y_range || bitset_has(dx, y_range)) continue;
        if (extra_dep[y_range] < 0) extra_dep[y_range] = x_idx;
      }
    }
  }

  free(deps);
  free(nest_parent);
  free(siblings);
  free(scores);
  return extra_dep;
}

/* ── Port of tinygrad's pm_add_control_flow ─────────────────────────── */
/* Adds predecessor edges as real RANGE sources so loop nesting is
 * structural in the DAG.  Applied after full_rewrite_to_sink, before
 * linearize.  Uses memoized DFS to handle transitive chains correctly
 * (a topo-order loop would fail when ridx0 precedes ridx1 in topo). */

/* 3-state visit marker for cycle detection in cf_rewrite. */
enum { CF_UNVISITED = 0, CF_VISITING = 1, CF_DONE = 2 };

static PolyUOp *cf_rewrite(PolyCtx *ctx, PolyUOp *u,
                            PolyUOp **topo, IntMap *idx,
                            int *extra_dep, PolyUOp **memo,
                            uint8_t *visit) {
  int ui = imap_try_get(idx, u);
  if (ui < 0) return u;              /* shared constant not in topo */
  if (visit[ui] == CF_DONE) return memo[ui];

  /* Cycle detection: if we're already visiting this node, we have a cycle
   * in the control-flow edges. This shouldn't happen, but guard against it
   * rather than infinite-recursing. */
  if (visit[ui] == CF_VISITING) {
    fprintf(stderr, "cf_rewrite: cycle detected at topo[%d] op=%s\n",
            ui, poly_op_name(u->op));
    memo[ui] = u;
    visit[ui] = CF_DONE;
    return u;
  }
  visit[ui] = CF_VISITING;

  /* Rewrite sources first (recursive) */
  PolyUOp *src[64];
  int ns = u->n_src;
  bool changed = false;
  for (int j = 0; j < ns; j++) {
    src[j] = cf_rewrite(ctx, u->src[j], topo, idx, extra_dep, memo, visit);
    if (src[j] != u->src[j]) changed = true;
  }

  /* Append control-flow dep for RANGE nodes */
  if (u->op == POLY_OP_RANGE && extra_dep[ui] >= 0) {
    PolyUOp *dep = cf_rewrite(ctx, topo[extra_dep[ui]],
                               topo, idx, extra_dep, memo, visit);
    /* Dedup: skip if dep already a source (after rewrite) */
    bool dup = false;
    for (int j = 0; j < ns; j++) {
      if (src[j] == dep) { dup = true; break; }
    }
    if (!dup) {
      src[ns++] = dep;
      changed = true;
    }
  }

  if (changed)
    memo[ui] = poly_uop(ctx, u->op, u->dtype, src, ns, u->arg);
  else
    memo[ui] = u;

  visit[ui] = CF_DONE;
  return memo[ui];
}

PolyUOp *poly_apply_control_flow(PolyCtx *ctx, PolyUOp *sink) {
  int n;
  PolyUOp **topo = poly_toposort(ctx, sink, &n);

  IntMap idx;
  imap_init(&idx, n);
  for (int i = 0; i < n; i++) imap_set(&idx, topo[i], i);

  int *extra_dep = build_control_edges(topo, n, &idx);

  /* Early exit if no edges */
  bool has_edges = false;
  for (int i = 0; i < n; i++) {
    if (extra_dep[i] >= 0) { has_edges = true; break; }
  }
  if (!has_edges) {
    free(extra_dep);
    imap_destroy(&idx);
    return sink;
  }

  /* DFS rewrite from sink with 3-state cycle detection */
  PolyUOp **memo = calloc(n, sizeof(PolyUOp *));
  uint8_t *visit = calloc(n, sizeof(uint8_t));
  PolyUOp *result = cf_rewrite(ctx, sink, topo, &idx, extra_dep, memo, visit);

#ifndef NDEBUG
  /* Verify RANGE invariant: src[0] is always the bound (CONST or DEFINE_VAR),
   * additional sources from control-flow edges are ordering-only. */
  for (int i = 0; i < n; i++) {
    if (memo[i] && memo[i]->op == POLY_OP_RANGE && memo[i]->n_src > 0) {
      PolyUOp *bound = memo[i]->src[0];
      if (bound->op != POLY_OP_CONST && bound->op != POLY_OP_DEFINE_VAR) {
        fprintf(stderr, "cf_rewrite: RANGE[%d] src[0] is %s, expected CONST or DEFINE_VAR\n",
                i, poly_op_name(bound->op));
      }
    }
  }
#endif

  free(visit);
  free(memo);
  free(extra_dep);
  imap_destroy(&idx);
  return result;
}

/* ── Tuplize comparison (matches tinygrad's UOp.tuplize for TUPLE_ORDER) ── */

static int arg_cmp(PolyArg a, PolyArg b) {
  /* Match Python's comparison semantics for tinygrad arg types.
   * Python compares None < numbers, int/float cross-type works. */
  if (a.kind != b.kind) return a.kind < b.kind ? -1 : 1;
  switch (a.kind) {
    case POLY_ARG_NONE: return 0;
    case POLY_ARG_INT:
      return a.i < b.i ? -1 : (a.i > b.i ? 1 : 0);
    case POLY_ARG_FLOAT:
      return (a.f < b.f) ? -1 : (a.f > b.f ? 1 : 0);
    case POLY_ARG_RANGE:
      if (a.range.axis_id != b.range.axis_id)
        return a.range.axis_id < b.range.axis_id ? -1 : 1;
      if (a.range.axis_type != b.range.axis_type)
        return a.range.axis_type < b.range.axis_type ? -1 : 1;
      if (a.range.n_extra != b.range.n_extra)
        return a.range.n_extra < b.range.n_extra ? -1 : 1;
      for (int i = 0; i < a.range.n_extra; i++) {
        if (a.range.extra[i] != b.range.extra[i])
          return a.range.extra[i] < b.range.extra[i] ? -1 : 1;
      }
      return 0;
    case POLY_ARG_REDUCE_AXIS: {
      if (a.reduce_axis.op != b.reduce_axis.op)
        return a.reduce_axis.op < b.reduce_axis.op ? -1 : 1;
      if (a.reduce_axis.n != b.reduce_axis.n)
        return a.reduce_axis.n < b.reduce_axis.n ? -1 : 1;
      for (int i = 0; i < a.reduce_axis.n; i++) {
        if (a.reduce_axis.axes[i] != b.reduce_axis.axes[i])
          return a.reduce_axis.axes[i] < b.reduce_axis.axes[i] ? -1 : 1;
      }
      return 0;
    }
    default: return 0;
  }
}

static int dtype_lt_cmp(PolyDType a, PolyDType b) {
  /* Match Python's DType.__lt__: compares only (priority, bitsize, name, fmt, count).
   * Does NOT include ptr-specific fields (is_ptr, addrspace, ptr_size). */
  if (a.priority != b.priority) return a.priority < b.priority ? -1 : 1;
  if (a.bitsize != b.bitsize) return a.bitsize < b.bitsize ? -1 : 1;
  if (a.name && b.name) {
    int nc = strcmp(a.name, b.name);
    if (nc != 0) return nc;
  } else if (a.name != b.name) {
    return a.name ? 1 : -1;
  }
  if (a.fmt != b.fmt) return a.fmt < b.fmt ? -1 : 1;
  if (a.count != b.count) return a.count < b.count ? -1 : 1;
  return 0;
}

static int tuplize_cmp(PolyUOp *a, PolyUOp *b) {
  /* Recursive comparison matching Python's tuple comparison of
   * (op.value, arg, dtype) + tuple(src.tuplize for src in src).
   *
   * Python DType quirk: DType.__eq__ is not defined, so __ne__ uses identity.
   * When two PtrDTypes have the same __lt__ key (priority, bitsize, name, fmt,
   * count) but are different objects (e.g. float.ptr(1,REG) vs float.ptr(4)),
   * Python's tuple comparison sees a[i] != b[i] (identity), calls a[i] < b[i]
   * which returns False, and immediately returns False — without proceeding to
   * later elements. sorted() then preserves the original (toposort) order.
   *
   * To match: when dtypes differ structurally but __lt__ fields are equal,
   * return 0 to stop recursion. The insertion sort's topo-index tiebreaker
   * will preserve the original order, matching Python's stable sort. */
  if (a == b) return 0;
  if (a->op != b->op) return a->op < b->op ? -1 : 1;
  int ac = arg_cmp(a->arg, b->arg);
  if (ac != 0) return ac;
  /* DType comparison: match Python's identity + __lt__ semantics */
  if (!poly_dtype_eq(a->dtype, b->dtype)) {
    int dc = dtype_lt_cmp(a->dtype, b->dtype);
    if (dc != 0) return dc;
    /* __lt__ fields match but dtypes are different objects → "incomparable"
     * in Python. Return 0 to let stable sort preserve topo order. */
    return 0;
  }
  /* Dtypes structurally equal → proceed to source comparison */
  int min_src = a->n_src < b->n_src ? a->n_src : b->n_src;
  for (int i = 0; i < min_src; i++) {
    int sc = tuplize_cmp(a->src[i], b->src[i]);
    if (sc != 0) return sc;
  }
  return a->n_src < b->n_src ? -1 : (a->n_src > b->n_src ? 1 : 0);
}

PolyUOp **poly_linearize_rewritten(PolyCtx *ctx, PolyUOp *sink, int *n_out) {
  /* 1. Standard toposort */
  int n;
  PolyUOp **topo = poly_toposort(ctx, sink, &n);

  /* 2. Build UOp* → topo-index lookup */
  IntMap idx;
  imap_init(&idx, n);
  for (int i = 0; i < n; i++) imap_set(&idx, topo[i], i);

  /* 3. Compute ranges bitset per UOp (forward pass).
   * Mirrors tinygrad's UOp.ranges property: the set of RANGE ops
   * each UOp is "inside". Used to compute run_count. */
  int words = (n + 63) / 64;
  uint64_t *ranges = calloc((size_t)n * (size_t)words, sizeof(uint64_t));
  for (int i = 0; i < n; i++) {
    PolyUOp *u = topo[i];
    uint64_t *r = ranges + (size_t)i * (size_t)words;
    /* Union of source ranges */
    for (int j = 0; j < u->n_src; j++) {
      int si = imap_try_get(&idx, u->src[j]);
      if (si < 0) continue;
      const uint64_t *sr = ranges + (size_t)si * (size_t)words;
      for (int w = 0; w < words; w++) r[w] |= sr[w];
    }
    /* Remove ended ranges */
    apply_uop_ended_ranges(r, u, topo, &idx, ranges, words);
    /* RANGE: add self. Assert src[0] is the bound — additional sources
     * from poly_apply_control_flow are control-flow ordering only. */
    if (u->op == POLY_OP_RANGE) {
      assert(u->n_src >= 1 && "RANGE must have at least one source (bound)");
      assert((u->src[0]->op == POLY_OP_CONST || u->src[0]->op == POLY_OP_DEFINE_VAR) &&
             "RANGE.src[0] must be CONST or DEFINE_VAR (bound)");
      bitset_set(r, i);
    }
  }

  /* 4. Compute out_degree, run_count, priority, extra (reverse pass).
   * Sort key mirrors tinygrad: (run_count, priority, extra). */
  int *out_deg = calloc(n, sizeof(int));
  int64_t *run_count = malloc(n * sizeof(int64_t));
  int *prio = malloc(n * sizeof(int));
  int64_t *extra = malloc(n * sizeof(int64_t));

  for (int i = n - 1; i >= 0; i--) {
    PolyUOp *u = topo[i];
    for (int j = 0; j < u->n_src; j++)
      out_deg[imap_get(&idx, u->src[j])]++;

    /* run_count = prod([int(r.vmax)+1 for r in u.ranges]) */
    run_count[i] = 1;
    const uint64_t *r = ranges + (size_t)i * (size_t)words;
    for (int w = 0; w < words; w++) {
      uint64_t bits = r[w];
      while (bits) {
        int bit = __builtin_ctzll(bits);
        int b = w * 64 + bit;
        if (b < n && topo[b]->op == POLY_OP_RANGE &&
            topo[b]->n_src > 0 && topo[b]->src[0]->op == POLY_OP_CONST)
          run_count[i] *= topo[b]->src[0]->arg.i;
        bits &= bits - 1;
      }
    }

    prio[i] = op_priority(u->op);
    extra[i] = INT64_MIN; /* sentinel for None (sorts before any int) */
    if (u->op == POLY_OP_PARAM) extra[i] = u->arg.i;
  }

  /* 5. Build ideal order: sort by (run_count, priority, extra, topo_idx).
   * Matches tinygrad's sorted(lst, key=lambda x: priorities[x]). */
  int *ideal = malloc(n * sizeof(int));
  for (int i = 0; i < n; i++) ideal[i] = i;

  /* Insertion sort (stable, sufficient for kernel sizes).
   * Sort key: (run_count, priority, extra, tuplize).
   * Matches tinygrad's sorted(lst, key=lambda x: priorities[x]+x.tuplize). */
  for (int i = 1; i < n; i++) {
    int ki = ideal[i];
    int64_t kr = run_count[ki];
    int kp = prio[ki];
    int64_t ke = extra[ki];
    int j = i - 1;
    while (j >= 0) {
      int ji = ideal[j];
      if (run_count[ji] > kr) { ideal[j + 1] = ideal[j]; j--; continue; }
      if (run_count[ji] < kr) break;
      if (prio[ji] > kp) { ideal[j + 1] = ideal[j]; j--; continue; }
      if (prio[ji] < kp) break;
      if (extra[ji] > ke) { ideal[j + 1] = ideal[j]; j--; continue; }
      if (extra[ji] < ke) break;
      /* Tiebreak: tuplize comparison (matches TUPLE_ORDER=1 in tinygrad) */
      int tc = tuplize_cmp(topo[ji], topo[ki]);
      if (tc > 0) { ideal[j + 1] = ideal[j]; j--; continue; }
      if (tc < 0) break;
      /* Final tiebreak: topo index */
      if (ji > ki) { ideal[j + 1] = ideal[j]; j--; }
      else break;
    }
    ideal[j + 1] = ki;
  }

  /* nkey[i] = position of topo[i] in ideal order */
  int *nkey = malloc(n * sizeof(int));
  for (int i = 0; i < n; i++) nkey[ideal[i]] = i;

  if (getenv("POLY_DUMP_KERNELS")) {
    for (int i = 0; i < n; i++) {
      PolyUOp *u = topo[i];
      if (u->op == POLY_OP_RANGE || u->op == POLY_OP_STORE ||
          u->op == POLY_OP_DEFINE_REG || u->op == POLY_OP_AFTER) {
        fprintf(stderr, "[lin] topo[%d] %s run_count=%lld prio=%d nkey=%d out_deg=%d\n",
                i, poly_op_name(u->op), (long long)run_count[i], prio[i], nkey[i], out_deg[i]);
      }
    }
  }

  /* 6. Reverse Kahn's with min-heap (using -nkey for max priority).
   * Starts from SINK, pops highest-nkey node, releases sources. */
  Heap heap;
  heap_init(&heap, n);

  int sink_idx = imap_get(&idx, sink);
  heap_push(&heap, -nkey[sink_idx], sink);

  PolyUOp **result = malloc(n * sizeof(PolyUOp *));
  int rlen = 0;

  while (heap.len > 0) {
    PolyUOp *u = heap_pop(&heap);
    result[rlen++] = u;
    for (int j = 0; j < u->n_src; j++) {
      int si = imap_get(&idx, u->src[j]);
      if (--out_deg[si] == 0)
        heap_push(&heap, -nkey[si], u->src[j]);
    }
  }

  /* Reverse: heap output is SINK-first → forward order */
  for (int i = 0; i < rlen / 2; i++) {
    PolyUOp *tmp = result[i];
    result[i] = result[rlen - 1 - i];
    result[rlen - 1 - i] = tmp;
  }

  if (rlen != n) {
    free(result);
    result = malloc((size_t)n * sizeof(PolyUOp *));
    memcpy(result, topo, (size_t)n * sizeof(PolyUOp *));
    rlen = n;
  }

  imap_destroy(&idx);
  heap_destroy(&heap);
  free(ranges);
  free(out_deg);
  free(run_count);
  free(prio);
  free(extra);
  free(ideal);
  free(nkey);

  *n_out = rlen;
  return result;
}

static bool env_true(const char *name) {
  const char *v = getenv(name);
  return v && v[0] != '\0' && strcmp(v, "0") != 0 &&
         strcmp(v, "false") != 0 && strcmp(v, "False") != 0;
}

PolyUOp **poly_linearize(PolyCtx *ctx, PolyUOp *sink, int *n_out) {
  PolyRewriteOpts opts = { .optimize = env_true("POLY_OPTIMIZE"), .devectorize = 0 };
  const char *dev = getenv("POLY_DEVECTORIZE");
  if (dev && dev[0] != '\0') opts.devectorize = atoi(dev);
  sink = poly_full_rewrite_to_sink_ex(ctx, sink, opts);
  sink = poly_apply_control_flow(ctx, sink);
  return poly_linearize_rewritten(ctx, sink, n_out);
}

/* ── Render helpers ──────────────────────────────────────────────────── */

/* Render a float constant with f suffix, ensuring a decimal point. */
static char *render_float_const(double v, char *buf, int cap) {
  if (isinf(v)) {
    snprintf(buf, cap, v > 0 ? "__builtin_inff()" : "(-__builtin_inff())");
    return buf;
  }
  if (isnan(v)) {
    snprintf(buf, cap, "__builtin_nanf(\"\")");
    return buf;
  }
  /* Use enough digits to round-trip float32 constants through text. */
  snprintf(buf, cap, "%.9g", (double)(float)v);
  /* ensure decimal point for C float literal */
  if (!strchr(buf, '.') && !strchr(buf, 'e') && !strchr(buf, 'E')) {
    int len = (int)strlen(buf);
    if (len + 2 < cap) { buf[len] = '.'; buf[len+1] = '0'; buf[len+2] = '\0'; }
  }
  int len = (int)strlen(buf);
  if (len + 1 < cap) { buf[len] = 'f'; buf[len+1] = '\0'; }
  return buf;
}

/* Render a C type for a PolyDType, including vector and pointer forms. */
static void render_ctype_nonptr(PolyDType dt, char *buf, int cap) {
  if (dt.count <= 1) {
    snprintf(buf, cap, "%s", dt.name);
    return;
  }
  PolyDType s = poly_dtype_scalar(dt);
  /* Clang/GCC vector bool is problematic; model vector masks as int vectors. */
  if (poly_dtype_is_bool(s)) {
    int vec_sz = (int)sizeof(int) * dt.count;
    snprintf(buf, cap, "int __attribute__((vector_size(%d)))", vec_sz);
  } else {
    int vec_sz = poly_dtype_itemsize(s) * dt.count;
    snprintf(buf, cap, "%s __attribute__((vector_size(%d)))", s.name, vec_sz);
  }
}

static void render_ctype(PolyDType dt, char *buf, int cap) {
  if (!dt.is_ptr) {
    render_ctype_nonptr(dt, buf, cap);
    return;
  }
  PolyDType base = dt;
  base.is_ptr = false;
  base.addrspace = POLY_ADDR_GLOBAL;
  base.vcount = 1;
  base.ptr_size = 0;
  char bt[128];
  render_ctype_nonptr(base, bt, sizeof(bt));
  snprintf(buf, cap, "%s*", bt);
}

/* Render an ALU expression. */
static void render_alu(char *buf, int cap, PolyOps op, PolyDType dtype,
                       const char *s0, const char *s1, const char *s2) {
  switch (op) {
  /* unary */
  case POLY_OP_NEG:
    snprintf(buf, cap, poly_dtype_is_bool(dtype) ? "(!%s)" : "(-%s)", s0); break;
  case POLY_OP_SQRT:
    snprintf(buf, cap, poly_dtype_eq(dtype, POLY_FLOAT64)
      ? "__builtin_sqrt(%s)" : "__builtin_sqrtf(%s)", s0); break;
  case POLY_OP_TRUNC:
    snprintf(buf, cap, poly_dtype_eq(dtype, POLY_FLOAT64)
      ? "__builtin_trunc(%s)" : "__builtin_truncf(%s)", s0); break;
  case POLY_OP_EXP2:       snprintf(buf, cap, "exp2f(%s)", s0); break;
  case POLY_OP_LOG2:       snprintf(buf, cap, "log2f(%s)", s0); break;
  case POLY_OP_SIN:        snprintf(buf, cap, "sinf(%s)", s0); break;
  case POLY_OP_RECIPROCAL: snprintf(buf, cap, "(1/%s)", s0); break;
  /* binary */
  case POLY_OP_ADD:   snprintf(buf, cap, "(%s+%s)", s0, s1); break;
  case POLY_OP_SUB:   snprintf(buf, cap, "(%s-%s)", s0, s1); break;
  case POLY_OP_MUL:   snprintf(buf, cap, "(%s*%s)", s0, s1); break;
  case POLY_OP_FDIV:  snprintf(buf, cap, "(%s/%s)", s0, s1); break;
  case POLY_OP_IDIV:  snprintf(buf, cap, "(%s/%s)", s0, s1); break;
  case POLY_OP_MOD:   snprintf(buf, cap, "(%s%%%s)", s0, s1); break;
  case POLY_OP_SHL:   snprintf(buf, cap, "(%s<<%s)", s0, s1); break;
  case POLY_OP_SHR:   snprintf(buf, cap, "(%s>>%s)", s0, s1); break;
  case POLY_OP_AND:   snprintf(buf, cap, "(%s&%s)", s0, s1); break;
  case POLY_OP_OR:    snprintf(buf, cap, "(%s|%s)", s0, s1); break;
  case POLY_OP_XOR:   snprintf(buf, cap, "(%s^%s)", s0, s1); break;
  case POLY_OP_CMPLT: snprintf(buf, cap, "(%s<%s)", s0, s1); break;
  case POLY_OP_CMPNE: snprintf(buf, cap, "(%s!=%s)", s0, s1); break;
  case POLY_OP_CMPEQ: snprintf(buf, cap, "(%s==%s)", s0, s1); break;
  case POLY_OP_MAX:   snprintf(buf, cap, "((%s>%s)?%s:%s)", s0, s1, s0, s1); break;
  case POLY_OP_POW:
    snprintf(buf, cap, poly_dtype_eq(dtype, POLY_FLOAT64)
      ? "pow(%s, %s)" : "powf(%s, %s)", s0, s1); break;
  /* ternary */
  case POLY_OP_WHERE:  snprintf(buf, cap, "(%s?%s:%s)", s0, s1, s2); break;
  case POLY_OP_MULACC: snprintf(buf, cap, "((%s*%s)+%s)", s0, s1, s2); break;
  default: snprintf(buf, cap, "/* unknown op %d */0", op); break;
  }
}

static int range_slot(PolyUOp **ranges, int *n_ranges, PolyUOp *r, bool create) {
  if (!r) return -1;
  for (int i = 0; i < *n_ranges; i++) {
    if (ranges[i] == r) return i;
  }
  if (!create || *n_ranges >= 128) return -1;
  ranges[*n_ranges] = r;
  (*n_ranges)++;
  return *n_ranges - 1;
}

/* ── C Renderer ──────────────────────────────────────────────────────── */

char *poly_render_c(PolyUOp **uops, int n, const char *fn_name) {
  StrBuf decls;  /* variable declarations at function scope */
  StrBuf body;   /* function body with assignments */
  sb_init(&decls);
  sb_init(&body);

  StrMap names;
  smap_init(&names, n);

  /* function parameter entries: (type_str, name_str, sort_key) */
  char *param_types[64];
  char *param_names[64];
  int param_order[64]; /* maps param position → args[] index */
  int n_params = 0;
  int n_buffer_params = 0; /* count of PARAM (buffer) params, used for DEFINE_VAR offset */

  /* prefix counters */
  int c_val = 0, c_alu = 0, c_cast = 0, c_acc = 0;
  int depth = 1;

  /* Range liveness tracking: emit END only after last non-END use. */
  PolyUOp *live_ranges[128];
  int live_remaining[128];
  int n_live_ranges = 0;
  memset(live_ranges, 0, sizeof(live_ranges));
  memset(live_remaining, 0, sizeof(live_remaining));

  PolyUOp *open_ranges[128];
  int n_open_ranges = 0;
  memset(open_ranges, 0, sizeof(open_ranges));

  for (int i = 0; i < n; i++) {
    PolyUOp *u = uops[i];
    if (u->op == POLY_OP_RANGE)
      (void)range_slot(live_ranges, &n_live_ranges, u, true);
    if (u->op == POLY_OP_END) continue;
    for (int j = 0; j < u->n_src; j++) {
      if (u->src[j] && u->src[j]->op == POLY_OP_RANGE) {
        int ri = range_slot(live_ranges, &n_live_ranges, u->src[j], true);
        if (ri >= 0) live_remaining[ri]++;
      }
    }
  }

  for (int i = 0; i < n; i++) {
    PolyUOp *u = uops[i];

    /* --- SINK: skip ------------------------------------------------- */
    if (u->op == POLY_OP_SINK || u->op == POLY_OP_NOOP || u->op == POLY_OP_GROUP)
      continue;

    if (u->op != POLY_OP_END) {
      for (int j = 0; j < u->n_src; j++) {
        if (u->src[j] && u->src[j]->op == POLY_OP_RANGE) {
          int ri = range_slot(live_ranges, &n_live_ranges, u->src[j], false);
          if (ri >= 0 && live_remaining[ri] > 0) live_remaining[ri]--;
        }
      }
    }

    /* --- PARAM: buffer pointer parameter ---------------------------- */
    if (u->op == POLY_OP_PARAM) {
      char name[32];
      snprintf(name, sizeof(name), "data%lld", (long long)u->arg.i);
      smap_set(&names, u, strdup(name));

      /* type for signature: "float* restrict" */
      PolyDType base = poly_dtype_scalar(u->dtype);
      char type[64];
      snprintf(type, sizeof(type), "%s* restrict", base.name);
      param_types[n_params] = strdup(type);
      param_names[n_params] = strdup(name);
      param_order[n_params] = (int)u->arg.i;
      n_params++;
      n_buffer_params++;
      continue;
    }

    /* --- DEFINE_VAR: integer parameter ------------------------------ */
    if (u->op == POLY_OP_DEFINE_VAR) {
      const char *vname = u->arg.kind == POLY_ARG_DEFINE_VAR ? u->arg.define_var.name
                        : (u->arg.str ? u->arg.str : "var");
      smap_set(&names, u, strdup(vname));
      param_types[n_params] = strdup("const int");
      param_names[n_params] = strdup(vname);
      /* DEFINE_VAR args come after all buffer args in the args[] array.
       * n_buffer_params counts PARAMs seen so far (all PARAMs precede DEFINE_VARs
       * in linearized output due to priority -20 vs -19). var_idx counts
       * DEFINE_VARs within the var section. */
      param_order[n_params] = n_buffer_params + (n_params - n_buffer_params);
      n_params++;
      continue;
    }

    /* --- CONST: inline literal -------------------------------------- */
    if (u->op == POLY_OP_CONST) {
      char val[64];
      if (poly_dtype_is_float(u->dtype)) {
        render_float_const(u->arg.f, val, sizeof(val));
      } else if (poly_dtype_is_bool(u->dtype)) {
        snprintf(val, sizeof(val), "%d", u->arg.b ? 1 : 0);
      } else if (poly_dtype_eq(u->dtype, POLY_INT64)) {
        snprintf(val, sizeof(val), "%lldll", (long long)u->arg.i);
      } else if (poly_dtype_eq(u->dtype, POLY_UINT64)) {
        snprintf(val, sizeof(val), "%lluull", (unsigned long long)(uint64_t)u->arg.i);
      } else if (poly_dtype_eq(u->dtype, POLY_UINT32)) {
        snprintf(val, sizeof(val), "%uu", (unsigned)(uint32_t)u->arg.i);
      } else {
        snprintf(val, sizeof(val), "%lld", (long long)u->arg.i);
      }
      smap_set(&names, u, strdup(val));
      continue;
    }

    /* --- VECTORIZE / VCONST: vector literal ------------------------ */
    if (u->op == POLY_OP_VECTORIZE || u->op == POLY_OP_VCONST) {
      if (u->n_src == 1) {
        char *s = smap_get(&names, u->src[0]);
        smap_set(&names, u, strdup(s ? s : "0"));
        continue;
      }
      char dtype_s[128];
      render_ctype(u->dtype, dtype_s, sizeof(dtype_s));
      StrBuf vexpr;
      sb_init(&vexpr);
      sb_printf(&vexpr, "((%s){", dtype_s);
      if (u->n_src > 0) {
        for (int j = 0; j < u->n_src; j++) {
          if (j) sb_puts(&vexpr, ",");
          char *s = smap_get(&names, u->src[j]);
          sb_puts(&vexpr, s ? s : "0");
        }
      } else if (u->arg.kind == POLY_ARG_INT_TUPLE) {
        for (int j = 0; j < u->arg.int_tuple.n; j++) {
          if (j) sb_puts(&vexpr, ",");
          sb_printf(&vexpr, "%lld", (long long)u->arg.int_tuple.vals[j]);
        }
      }
      sb_puts(&vexpr, "})");
      smap_set(&names, u, vexpr.buf);  /* takes ownership */
      continue;
    }

    /* --- GEP: vector lane extract ---------------------------------- */
    if (u->op == POLY_OP_GEP) {
      if (u->n_src < 1) { smap_set(&names, u, strdup("0")); continue; }
      char *src_s = smap_get(&names, u->src[0]);
      if (u->arg.kind == POLY_ARG_INT_TUPLE && u->arg.int_tuple.n > 0) {
        if (u->arg.int_tuple.n == 1) {
          char expr[256];
          snprintf(expr, sizeof(expr), "(%s[%lld])", src_s ? src_s : "0",
                   (long long)u->arg.int_tuple.vals[0]);
          smap_set(&names, u, strdup(expr));
        } else {
          char dtype_s[128];
          render_ctype(u->dtype, dtype_s, sizeof(dtype_s));
          StrBuf vexpr;
          sb_init(&vexpr);
          sb_printf(&vexpr, "((%s){", dtype_s);
          for (int j = 0; j < u->arg.int_tuple.n; j++) {
            if (j) sb_puts(&vexpr, ",");
            sb_printf(&vexpr, "(%s[%lld])", src_s ? src_s : "0",
                      (long long)u->arg.int_tuple.vals[j]);
          }
          sb_puts(&vexpr, "})");
          smap_set(&names, u, vexpr.buf);
        }
      } else if (u->arg.kind == POLY_ARG_INT) {
        char expr[256];
        snprintf(expr, sizeof(expr), "(%s[%lld])", src_s ? src_s : "0", (long long)u->arg.i);
        smap_set(&names, u, strdup(expr));
      } else {
        smap_set(&names, u, strdup(src_s ? src_s : "0"));
      }
      continue;
    }

    /* --- INDEX: inline pointer arithmetic --------------------------- */
    if (u->op == POLY_OP_INDEX) {
      char *buf_s = smap_get(&names, u->src[0]);
      char *idx_s = smap_get(&names, u->src[1]);
      char expr[256];
      snprintf(expr, sizeof(expr), "(%s+%s)", buf_s, idx_s);
      smap_set(&names, u, strdup(expr));
      continue;
    }

    /* --- RANGE: for loop -------------------------------------------- */
    if (u->op == POLY_OP_RANGE) {
      char name[32];
      snprintf(name, sizeof(name), "ridx%lld", (long long)poly_range_axis_id(u->arg));
      smap_set(&names, u, strdup(name));

      char *bound = smap_get(&names, u->src[0]);
      for (int d = 0; d < depth; d++) sb_puts(&body, "  ");
      sb_printf(&body, "for (int %s = 0; %s < %s; %s++) {\n",
                name, name, bound, name);
      depth++;
      if (n_open_ranges < 128) open_ranges[n_open_ranges++] = u;
      continue;
    }

    /* --- END / ENDIF: close brace ----------------------------------- */
    if (u->op == POLY_OP_END || u->op == POLY_OP_ENDIF) {
      if (u->op == POLY_OP_END && u->n_src > 1 && u->src[1]->op == POLY_OP_RANGE) {
        PolyUOp *want = u->src[1];
        int wi = range_slot(live_ranges, &n_live_ranges, want, false);
        if (wi >= 0 && live_remaining[wi] > 0) continue;  /* too early */

        int pos = -1;
        for (int p = n_open_ranges - 1; p >= 0; p--) {
          if (open_ranges[p] == want) { pos = p; break; }
        }
        if (pos < 0) continue;  /* duplicate/stale END */

        bool can_close = true;
        for (int p = n_open_ranges - 1; p >= pos; p--) {
          int oi = range_slot(live_ranges, &n_live_ranges, open_ranges[p], false);
          if (oi >= 0 && live_remaining[oi] > 0 && open_ranges[p] != want) {
            can_close = false;
            break;
          }
        }
        if (!can_close) continue;

        while (n_open_ranges > pos) {
          depth--;
          for (int d = 0; d < depth; d++) sb_puts(&body, "  ");
          sb_puts(&body, "}\n");
          n_open_ranges--;
        }
        continue;
      }

      depth--;
      for (int d = 0; d < depth; d++) sb_puts(&body, "  ");
      sb_puts(&body, "}\n");
      if (u->op == POLY_OP_END && n_open_ranges > 0) n_open_ranges--;
      continue;
    }

    /* --- DEFINE_LOCAL: accumulator variable -------------------------- */
    if (u->op == POLY_OP_DEFINE_LOCAL) {
      char name[32];
      snprintf(name, sizeof(name), "acc%d", c_acc++);
      smap_set(&names, u, strdup(name));

      char initval[64];
      if (u->arg.kind == POLY_ARG_FLOAT)
        render_float_const(u->arg.f, initval, sizeof(initval));
      else
        snprintf(initval, sizeof(initval), "0.0f");

      char dtype_s[128];
      render_ctype(u->dtype, dtype_s, sizeof(dtype_s));
      sb_printf(&decls, "  %s %s;\n", dtype_s, name);
      for (int d = 0; d < depth; d++) sb_puts(&body, "  ");
      sb_printf(&body, "%s = %s;\n", name, initval);
      continue;
    }

    /* --- DEFINE_REG: register accumulator (float r0[1];) ------------ */
    if (u->op == POLY_OP_DEFINE_REG) {
      char name[32];
      snprintf(name, sizeof(name), "r%lld", (long long)u->arg.i);
      smap_set(&names, u, strdup(name));

      PolyDType base = poly_dtype_scalar(u->dtype);
      sb_printf(&decls, "  %s %s[1];\n", base.name, name);
      continue;
    }

    /* --- AFTER: pass-through (use src[0]'s name) -------------------- */
    if (u->op == POLY_OP_AFTER) {
      char *src_name = smap_get(&names, u->src[0]);
      if (src_name)
        smap_set(&names, u, strdup(src_name));
      continue;
    }

    /* --- LOAD: dereference indexed pointer -------------------------- */
    if (u->op == POLY_OP_LOAD) {
      char name[32];
      snprintf(name, sizeof(name), "val%d", c_val++);
      smap_set(&names, u, strdup(name));

      char *bidx = smap_get(&names, u->src[0]);
      char dtype_s[128];
      render_ctype(u->dtype, dtype_s, sizeof(dtype_s));
      sb_printf(&decls, "  %s %s;\n", dtype_s, name);
      for (int d = 0; d < depth; d++) sb_puts(&body, "  ");
      sb_printf(&body, "%s = (*%s);\n", name, bidx);
      continue;
    }

    /* --- STORE: write to indexed pointer or accumulator ------------- */
    if (u->op == POLY_OP_STORE) {
      char *target = smap_get(&names, u->src[0]);
      char *val    = smap_get(&names, u->src[1]);
      for (int d = 0; d < depth; d++) sb_puts(&body, "  ");
      /* Guard: STORE src[0] is always set by construction, but null-check
       * satisfies the analyzer's path-sensitive null-deref tracking. */
      if (u->src[0] && u->src[0]->op == POLY_OP_DEFINE_LOCAL)
        sb_printf(&body, "%s = %s;\n", target, val);
      else
        sb_printf(&body, "*%s = %s;\n", target, val);
      continue;
    }

    /* --- CAST: type conversion -------------------------------------- */
    if (u->op == POLY_OP_CAST) {
      char name[32];
      snprintf(name, sizeof(name), "cast%d", c_cast++);
      smap_set(&names, u, strdup(name));

      char *src_s = smap_get(&names, u->src[0]);
      char dtype_s[128];
      render_ctype(u->dtype, dtype_s, sizeof(dtype_s));
      sb_printf(&decls, "  %s %s;\n", dtype_s, name);
      for (int d = 0; d < depth; d++) sb_puts(&body, "  ");
      sb_printf(&body, "%s = (%s)(%s);\n", name, dtype_s, src_s);
      continue;
    }

    /* --- BITCAST: reinterpret bits (union punning, C11-legal) ------- */
    if (u->op == POLY_OP_BITCAST) {
      char name[32];
      snprintf(name, sizeof(name), "cast%d", c_cast++);
      smap_set(&names, u, strdup(name));

      char *src_s = smap_get(&names, u->src[0]);
      char src_type[128], dst_type[128];
      render_ctype(u->src[0]->dtype, src_type, sizeof(src_type));
      render_ctype(u->dtype, dst_type, sizeof(dst_type));
      sb_printf(&decls, "  %s %s;\n", dst_type, name);
      for (int d = 0; d < depth; d++) sb_puts(&body, "  ");
      sb_printf(&body, "{ %s _bc = %s; memcpy(&%s, &_bc, sizeof(%s)); }\n",
                src_type, src_s, name, name);
      continue;
    }

    /* --- ALU ops: arithmetic expressions ---------------------------- */
    if (poly_opset_has(POLY_GROUP_ALU, u->op)) {
      char expr[512];
      const char *s0 = (u->n_src > 0) ? smap_get(&names, u->src[0]) : "";
      const char *s1 = (u->n_src > 1) ? smap_get(&names, u->src[1]) : "";
      const char *s2 = (u->n_src > 2) ? smap_get(&names, u->src[2]) : "";
      render_alu(expr, sizeof(expr), u->op, u->dtype, s0, s1, s2);

      char name[32];
      snprintf(name, sizeof(name), "alu%d", c_alu++);
      smap_set(&names, u, strdup(name));

      char dtype_s[128];
      render_ctype(u->dtype, dtype_s, sizeof(dtype_s));
      sb_printf(&decls, "  %s %s;\n", dtype_s, name);
      for (int d = 0; d < depth; d++) sb_puts(&body, "  ");
      sb_printf(&body, "%s = %s;\n", name, expr);
      continue;
    }

    /* --- IF: conditional -------------------------------------------- */
    if (u->op == POLY_OP_IF) {
      char *cond_s = smap_get(&names, u->src[0]);
      for (int d = 0; d < depth; d++) sb_puts(&body, "  ");
      sb_printf(&body, "if (%s) {\n", cond_s);
      depth++;
      continue;
    }
  }

  /* Debug: check depth balance */
  if (depth != 1) {
    int n_ranges = 0, n_ends = 0, n_ends_norange = 0;
    PolyUOp *range_ptrs[64];
    int64_t range_sizes[64];
    int range_end_count[64];
    for (int i = 0; i < n; i++) {
      if (uops[i]->op == POLY_OP_RANGE && n_ranges < 64) {
        range_ptrs[n_ranges] = uops[i];
        range_sizes[n_ranges] = (uops[i]->n_src > 0 && uops[i]->src[0]->op == POLY_OP_CONST) ? uops[i]->src[0]->arg.i : -1;
        range_end_count[n_ranges] = 0;
        n_ranges++;
      }
      if (uops[i]->op == POLY_OP_END) {
        n_ends++;
        if (uops[i]->n_src > 1 && uops[i]->src[1]->op == POLY_OP_RANGE) {
          for (int r = 0; r < n_ranges; r++) {
            if (range_ptrs[r] == uops[i]->src[1]) range_end_count[r]++;
          }
        } else {
          n_ends_norange++;
        }
      }
    }
    fprintf(stderr, "polygrad render_c: DEPTH MISMATCH: depth=%d (expected 1) "
            "%d RANGEs, %d ENDs (%d without RANGE ref)\n",
            depth, n_ranges, n_ends, n_ends_norange);
    for (int r = 0; r < n_ranges; r++) {
      fprintf(stderr, "  RANGE[%d] %p size=%lld %s\n", r, (void*)range_ptrs[r],
              (long long)range_sizes[r],
              (range_end_count[r] > 0) ? "HAS_END" : "ORPHAN");
      fprintf(stderr, "    END count: %d\n", range_end_count[r]);
    }
  }

  /* ── Sort params by arg index (PARAM 0, 1, 2, ...) ─────────────── */
  for (int i = 1; i < n_params; i++) {
    int ko = param_order[i];
    char *kt = param_types[i], *kn = param_names[i];
    int j = i - 1;
    while (j >= 0 && param_order[j] > ko) {
      param_order[j+1] = param_order[j];
      param_types[j+1] = param_types[j];
      param_names[j+1] = param_names[j];
      j--;
    }
    param_order[j+1] = ko;
    param_types[j+1] = kt;
    param_names[j+1] = kn;
  }

  /* ── Build complete source ────────────────────────────────────────── */
  StrBuf out;
  sb_init(&out);
  sb_puts(&out, "#include <math.h>\n");
  sb_puts(&out, "#include <stdbool.h>\n");
  sb_puts(&out, "#include <string.h>\n");

  /* function signature */
  sb_printf(&out, "void %s(", fn_name);
  for (int i = 0; i < n_params; i++) {
    if (i > 0) sb_puts(&out, ", ");
    sb_printf(&out, "%s %s", param_types[i], param_names[i]);
  }
  sb_puts(&out, ") {\n");
  if (decls.len > 0) sb_puts(&out, decls.buf);
  sb_puts(&out, body.buf);
  sb_puts(&out, "}\n");

  /* _call wrapper: takes void** and dispatches to the typed function */
  sb_printf(&out, "void %s_call(void **args) {\n", fn_name);
  sb_printf(&out, "  %s(", fn_name);
  for (int i = 0; i < n_params; i++) {
    if (i > 0) sb_puts(&out, ", ");
    int arg_idx = param_order[i];
    /* pointer params: (type*)args[i]; int params: *(int*)args[i] */
    if (strchr(param_types[i], '*')) {
      /* extract base type (before '* restrict') */
      char base[64];
      const char *star = strchr(param_types[i], '*');
      int blen = (int)(star - param_types[i]);
      memcpy(base, param_types[i], blen);
      base[blen] = '\0';
      sb_printf(&out, "(%s*)args[%d]", base, arg_idx);
    } else {
      sb_printf(&out, "*(int*)args[%d]", arg_idx);
    }
  }
  sb_puts(&out, ");\n}\n");

  /* cleanup */
  for (int i = 0; i < n_params; i++) {
    free(param_types[i]);
    free(param_names[i]);
  }
  free(decls.buf);
  free(body.buf);
  smap_destroy(&names);

  return out.buf;
}
