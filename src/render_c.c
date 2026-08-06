/*
 * render_c.c — Linearizer + C code renderer
 *
 * Linearizer: priority-based toposort matching tinygrad's linearizer.py.
 * Renderer: walks linearized UOps, emits C source (ClangRenderer port).
 */

#define _POSIX_C_SOURCE 200809L

#include "codegen.h"
#include "bigint.h"
#include "ctx.h"
#include "utils.h"
#include "pat.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>

/* String builder */

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

/* Pointer → int hash map (for linearizer) */

typedef struct {
  PolyUOp **keys;
  int *vals;
  int cap;
} IntMap;


static void imap_init(IntMap *m, int n) {
  m->cap = (n < 4) ? 16 : n * 3;
  m->keys = calloc(m->cap, sizeof(PolyUOp *));
  m->vals = calloc(m->cap, sizeof(int));
}

static void imap_set(IntMap *m, PolyUOp *key, int val) {
  uint32_t h = poly_ptr_hash(key) % m->cap;
  while (m->keys[h] && m->keys[h] != key)
    h = (h + 1) % m->cap;
  m->keys[h] = key;
  m->vals[h] = val;
}

static int imap_get(IntMap *m, PolyUOp *key) {
  uint32_t h = poly_ptr_hash(key) % m->cap;
  while (m->keys[h] != key)
    h = (h + 1) % m->cap;
  return m->vals[h];
}

static int imap_try_get(IntMap *m, PolyUOp *key) {
  uint32_t h = poly_ptr_hash(key) % m->cap;
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

/* Pointer → string hash map (for renderer) */

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
  uint32_t h = poly_ptr_hash(key) % m->cap;
  while (m->keys[h] && m->keys[h] != key)
    h = (h + 1) % m->cap;
  if (m->keys[h] == key) free(m->vals[h]); /* replace existing */
  m->keys[h] = key;
  m->vals[h] = val;
}

static char *smap_get(StrMap *m, PolyUOp *key) {
  uint32_t h = poly_ptr_hash(key) % m->cap;
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

/* Min-heap (for linearizer reverse toposort) */

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
    int tk = h->keys[i];
    h->keys[i] = h->keys[p];
    h->keys[p] = tk;
    PolyUOp *tv = h->vals[i];
    h->vals[i] = h->vals[p];
    h->vals[p] = tv;
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
      int l = 2 * i + 1, r = 2 * i + 2, s = i;
      if (l < h->len && h->keys[l] < h->keys[s]) s = l;
      if (r < h->len && h->keys[r] < h->keys[s]) s = r;
      if (s == i) break;
      int tk = h->keys[i];
      h->keys[i] = h->keys[s];
      h->keys[s] = tk;
      PolyUOp *tv = h->vals[i];
      h->vals[i] = h->vals[s];
      h->vals[s] = tv;
      i = s;
    }
  }
  return result;
}

static void heap_destroy(Heap *h) {
  free(h->keys);
  free(h->vals);
}

/* Linearizer */

static int uop_priority(const PolyUOp *u) {
  if (!u) return 0;
  switch (u->op) {
  case POLY_OP_PARAM:
    return -20;
  case POLY_OP_DEFINE_VAR:
    return -19;
  case POLY_OP_BUFFER:
    return (u->dtype.is_ptr && u->dtype.addrspace == POLY_ADDR_LOCAL) ? -17 : -18;
  case POLY_OP_DEFINE_LOCAL:
    return -17;
  case POLY_OP_DEFINE_REG:
    return -18;
  /* NOTE: in the reference tinygrad commit, CONST has no special
   * priority — it falls through to default priority 0.  This is
   * critical for matching the tuplize-based ordering. */
  case POLY_OP_LOAD:
    return -1;
  case POLY_OP_STORE:
    return 1;
  case POLY_OP_RANGE:
    return 5;
  case POLY_OP_END:
    return -5;
  default:
    return 0;
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
static void apply_uop_ended_ranges(
    uint64_t *r,
    PolyUOp *u,
    PolyUOp **topo,
    IntMap *idx,
    uint64_t *all_ranges,
    int words
) {
  int rs = -1;
  switch (u->op) {
  case POLY_OP_STORE:
    rs = 2;
    break;
  case POLY_OP_END:
    rs = 1;
    break;
  case POLY_OP_REDUCE:
    rs = 1;
    break;
  case POLY_OP_AFTER:
    for (int j = 1; j < u->n_src; j++)
      apply_uop_ended_ranges(r, u->src[j], topo, idx, all_ranges, words);
    return;
  default:
    return;
  }
  for (int j = rs; j < u->n_src; j++) {
    int si = imap_try_get(idx, u->src[j]);
    if (si < 0) continue;
    if (topo[si]->op == POLY_OP_RANGE) {
      bitset_clear(r, si);
    } else {
      const uint64_t *er = all_ranges + (size_t)si * (size_t)words;
      for (int w = 0; w < words; w++)
        r[w] &= ~er[w];
    }
  }
}

static int dep_count_in_siblings(
    const uint64_t *deps,
    int words,
    int idx,
    const int *siblings,
    int n_siblings
) {
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
  for (int i = 0; i < n; i++)
    extra_dep[i] = -1;
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
  for (int i = 0; i < n; i++)
    nest_parent[i] = -1;

  for (int i = 0; i < n; i++) {
    PolyUOp *u = topo[i];
    uint64_t *du = deps + (size_t)i * (size_t)words;

    /* deps[u] |= deps[src] */
    for (int j = 0; j < u->n_src; j++) {
      int si = imap_try_get(idx, u->src[j]);
      if (si < 0) continue;
      const uint64_t *ds = deps + (size_t)si * (size_t)words;
      for (int w = 0; w < words; w++)
        du[w] |= ds[w];
    }

    /* Build nesting map from END -> (END or SINK) parent. */
    if (u->op == POLY_OP_END || u->op == POLY_OP_SINK) {
      int parent_rng_idx = -1;
      if (u->op == POLY_OP_END && u->n_src > 1) parent_rng_idx = imap_try_get(idx, u->src[1]);

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

/* Port of tinygrad's pm_add_control_flow */
/* Adds predecessor edges as real RANGE sources so loop nesting is
 * structural in the DAG.  Applied after full_rewrite_to_sink, before
 * linearize.  Uses memoized DFS to handle transitive chains correctly
 * (a topo-order loop would fail when ridx0 precedes ridx1 in topo). */

/* 3-state visit marker for cycle detection in cf_rewrite. */
enum { CF_UNVISITED = 0, CF_VISITING = 1, CF_DONE = 2 };

static PolyUOp *cf_rewrite(
    PolyCtx *ctx,
    PolyUOp *u,
    PolyUOp **topo,
    IntMap *idx,
    int *extra_dep,
    PolyUOp **memo,
    uint8_t *visit
) {
  int ui = imap_try_get(idx, u);
  if (ui < 0) return u; /* shared constant not in topo */
  if (visit[ui] == CF_DONE) return memo[ui];

  /* Cycle detection: if we're already visiting this node, we have a cycle
   * in the control-flow edges. This shouldn't happen, but guard against it
   * rather than infinite-recursing. */
  if (visit[ui] == CF_VISITING) {
    fprintf(stderr, "cf_rewrite: cycle detected at topo[%d] op=%s\n", ui, poly_op_name(u->op));
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
    PolyUOp *dep = cf_rewrite(ctx, topo[extra_dep[ui]], topo, idx, extra_dep, memo, visit);
    /* Dedup: skip if dep already a source (after rewrite) */
    bool dup = false;
    for (int j = 0; j < ns; j++) {
      if (src[j] == dep) {
        dup = true;
        break;
      }
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
  PolyScratchMark scratch = poly_ctx_scratch_mark(ctx);
  PolyUOp **topo = poly_toposort_scratch(ctx, sink, &n);
  if (!topo && n != 0) {
    poly_ctx_scratch_rewind(ctx, scratch);
    return sink;
  }

  IntMap idx;
  imap_init(&idx, n);
  for (int i = 0; i < n; i++)
    imap_set(&idx, topo[i], i);

  int *extra_dep = build_control_edges(topo, n, &idx);

  /* Early exit if no edges */
  bool has_edges = false;
  for (int i = 0; i < n; i++) {
    if (extra_dep[i] >= 0) {
      has_edges = true;
      break;
    }
  }
  if (!has_edges) {
    free(extra_dep);
    imap_destroy(&idx);
    poly_ctx_scratch_rewind(ctx, scratch);
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
        fprintf(
            stderr, "cf_rewrite: RANGE[%d] src[0] is %s, expected CONST or DEFINE_VAR\n", i,
            poly_op_name(bound->op)
        );
      }
    }
  }
#endif

  free(visit);
  free(memo);
  free(extra_dep);
  imap_destroy(&idx);
  poly_ctx_scratch_rewind(ctx, scratch);
  return result;
}

/* Tuplize ranking (matches tinygrad's cached UOp.tuplize for TUPLE_ORDER). */

static int arg_cmp(PolyArg a, PolyArg b) {
  /* Match Python's comparison semantics for tinygrad arg types.
   * Python compares None < numbers, int/float cross-type works. */
  if (a.kind != b.kind) return a.kind < b.kind ? -1 : 1;
  switch (a.kind) {
  case POLY_ARG_NONE:
    return 0;
  case POLY_ARG_INT:
    return a.i < b.i ? -1 : (a.i > b.i ? 1 : 0);
  case POLY_ARG_FLOAT:
    return (a.f < b.f) ? -1 : (a.f > b.f ? 1 : 0);
  case POLY_ARG_RANGE:
    if (a.range.axis_id != b.range.axis_id) return a.range.axis_id < b.range.axis_id ? -1 : 1;
    if (a.range.axis_type != b.range.axis_type)
      return a.range.axis_type < b.range.axis_type ? -1 : 1;
    if (a.range.n_extra != b.range.n_extra) return a.range.n_extra < b.range.n_extra ? -1 : 1;
    for (int i = 0; i < a.range.n_extra; i++) {
      if (a.range.extra[i] != b.range.extra[i]) return a.range.extra[i] < b.range.extra[i] ? -1 : 1;
    }
    return 0;
  case POLY_ARG_REDUCE_AXIS: {
    if (a.reduce_axis.op != b.reduce_axis.op) return a.reduce_axis.op < b.reduce_axis.op ? -1 : 1;
    if (a.reduce_axis.n != b.reduce_axis.n) return a.reduce_axis.n < b.reduce_axis.n ? -1 : 1;
    for (int i = 0; i < a.reduce_axis.n; i++) {
      if (a.reduce_axis.axes[i] != b.reduce_axis.axes[i])
        return a.reduce_axis.axes[i] < b.reduce_axis.axes[i] ? -1 : 1;
    }
    return 0;
  }
  default:
    return 0;
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

static PolyDType linearizer_tuplize_dtype(PolyUOp *u) {
  if (!u) return POLY_VOID;
  PolyDType dt = u->dtype;
  /* tinygrad runs pm_remove_vec_dtypes before linearize:
   *   PARAM/BUFFER ptrs become base dtype + size source
   *   every other op becomes dtype.base.scalar().base
   * Polygrad renderers still use vector/pointer dtype metadata internally, so
   * normalize only the linearizer tuplize key until all renderers consume the
   * new-style structural width representation directly. */
  if (u->op == POLY_OP_PARAM || u->op == POLY_OP_BUFFER) {
    if (dt.is_ptr) {
      dt.is_ptr = false;
      dt.addrspace = POLY_ADDR_GLOBAL;
      dt.ptr_size = 0;
      dt.vcount = 0;
      dt = poly_dtype_scalar(dt);
    }
    return dt;
  }
  dt = poly_dtype_scalar(dt);
  dt.is_ptr = false;
  dt.addrspace = POLY_ADDR_GLOBAL;
  dt.ptr_size = 0;
  dt.vcount = 0;
  return dt;
}

typedef struct {
  uint64_t *keys;
  int8_t *vals;
  int cap;
  int len;
} TuplizePairMemo;

static uint64_t tuplize_pair_key(int a, int b) {
  return (((uint64_t)(uint32_t)a) << 32) | (uint32_t)b;
}

static uint64_t tuplize_pair_hash(uint64_t x) {
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33;
  x *= 0xc4ceb9fe1a85ec53ULL;
  x ^= x >> 33;
  return x;
}

static bool tuplize_pair_memo_init(TuplizePairMemo *m, int n) {
  int cap = 1024;
  while (cap < n * 4) cap <<= 1;
  m->keys = calloc((size_t)cap, sizeof(uint64_t));
  m->vals = calloc((size_t)cap, sizeof(int8_t));
  if (!m->keys || !m->vals) {
    free(m->keys);
    free(m->vals);
    memset(m, 0, sizeof(*m));
    return false;
  }
  m->cap = cap;
  m->len = 0;
  return true;
}

static void tuplize_pair_memo_destroy(TuplizePairMemo *m) {
  free(m->keys);
  free(m->vals);
  memset(m, 0, sizeof(*m));
}

static bool tuplize_pair_memo_get(TuplizePairMemo *m, uint64_t key, int *out) {
  if (!m->cap) return false;
  uint64_t stored = key + 1;
  uint64_t mask = (uint64_t)m->cap - 1;
  uint64_t pos = tuplize_pair_hash(stored) & mask;
  while (m->keys[pos]) {
    if (m->keys[pos] == stored) {
      *out = (int)m->vals[pos];
      return true;
    }
    pos = (pos + 1) & mask;
  }
  return false;
}

static bool tuplize_pair_memo_grow(TuplizePairMemo *m) {
  TuplizePairMemo nm = {0};
  nm.cap = m->cap ? m->cap << 1 : 1024;
  nm.keys = calloc((size_t)nm.cap, sizeof(uint64_t));
  nm.vals = calloc((size_t)nm.cap, sizeof(int8_t));
  if (!nm.keys || !nm.vals) {
    free(nm.keys);
    free(nm.vals);
    return false;
  }

  for (int i = 0; i < m->cap; i++) {
    if (!m->keys[i]) continue;
    uint64_t mask = (uint64_t)nm.cap - 1;
    uint64_t pos = tuplize_pair_hash(m->keys[i]) & mask;
    while (nm.keys[pos]) pos = (pos + 1) & mask;
    nm.keys[pos] = m->keys[i];
    nm.vals[pos] = m->vals[i];
    nm.len++;
  }

  free(m->keys);
  free(m->vals);
  *m = nm;
  return true;
}

static bool tuplize_pair_memo_set(TuplizePairMemo *m, uint64_t key, int val) {
  if ((m->len + 1) * 2 >= m->cap && !tuplize_pair_memo_grow(m)) return false;
  uint64_t stored = key + 1;
  uint64_t mask = (uint64_t)m->cap - 1;
  uint64_t pos = tuplize_pair_hash(stored) & mask;
  while (m->keys[pos] && m->keys[pos] != stored) pos = (pos + 1) & mask;
  if (!m->keys[pos]) {
    m->keys[pos] = stored;
    m->len++;
  }
  m->vals[pos] = (int8_t)((val > 0) - (val < 0));
  return true;
}

typedef struct {
  PolyUOp **topo;
  IntMap *idx;
  TuplizePairMemo *memo;
} TuplizeCmpCtx;

static int tuplize_cmp_idx(TuplizeCmpCtx *tc, int ai, int bi) {
  if (ai == bi) return 0;

  uint64_t key = tuplize_pair_key(ai, bi);
  int cached = 0;
  if (tuplize_pair_memo_get(tc->memo, key, &cached)) return cached;

  PolyUOp *a = tc->topo[ai];
  PolyUOp *b = tc->topo[bi];
  int ret = 0;

  /* Port of tinygrad UOp.tuplize:
   *   (op.value, arg, dtype) + tuple(src.tuplize for src in src)
   *
   * tinygrad caches the recursive tuple on each UOp. Here the pair memo gives
   * the same effect for sorting: each structural pair comparison is computed at
   * most once, while sources are compared recursively in tuple order. */
  int ao = poly_op_value(a->op), bo = poly_op_value(b->op);
  if (ao != bo) {
    ret = ao < bo ? -1 : 1;
  } else {
    int ac = arg_cmp(a->arg, b->arg);
    if (ac != 0) {
      ret = ac;
    } else {
      int dc = dtype_lt_cmp(linearizer_tuplize_dtype(a), linearizer_tuplize_dtype(b));
      if (dc != 0) {
        ret = dc;
      } else {
        int min_src = a->n_src < b->n_src ? a->n_src : b->n_src;
        for (int i = 0; i < min_src && ret == 0; i++) {
          int as = imap_get(tc->idx, a->src[i]);
          int bs = imap_get(tc->idx, b->src[i]);
          ret = tuplize_cmp_idx(tc, as, bs);
        }
        if (ret == 0) ret = a->n_src < b->n_src ? -1 : (a->n_src > b->n_src ? 1 : 0);
      }
    }
  }

  ret = (ret > 0) - (ret < 0);
  tuplize_pair_memo_set(tc->memo, key, ret);
  tuplize_pair_memo_set(tc->memo, tuplize_pair_key(bi, ai), -ret);
  return ret;
}

static void tuplize_merge_sort_rec(TuplizeCmpCtx *tc, int *arr, int *tmp, int lo, int hi) {
  if (hi - lo <= 1) return;
  int mid = lo + (hi - lo) / 2;
  tuplize_merge_sort_rec(tc, arr, tmp, lo, mid);
  tuplize_merge_sort_rec(tc, arr, tmp, mid, hi);

  int i = lo, j = mid, k = lo;
  while (i < mid && j < hi) {
    /* Python's sorted is stable, so equal tuplize keys keep topo order. */
    if (tuplize_cmp_idx(tc, arr[i], arr[j]) <= 0)
      tmp[k++] = arr[i++];
    else
      tmp[k++] = arr[j++];
  }
  while (i < mid)
    tmp[k++] = arr[i++];
  while (j < hi)
    tmp[k++] = arr[j++];
  memcpy(arr + lo, tmp + lo, (size_t)(hi - lo) * sizeof(int));
}

static int *compute_tuplize_ranks(PolyUOp **topo, int n, IntMap *idx) {
  int *rank = malloc((size_t)n * sizeof(int));
  int *order = malloc((size_t)n * sizeof(int));
  int *tmp = malloc((size_t)n * sizeof(int));
  TuplizePairMemo memo = {0};
  if (!rank || !order || !tmp || !tuplize_pair_memo_init(&memo, n)) {
    free(rank);
    free(order);
    free(tmp);
    return NULL;
  }

  for (int i = 0; i < n; i++)
    order[i] = i;

  TuplizeCmpCtx tc = {.topo = topo, .idx = idx, .memo = &memo};
  tuplize_merge_sort_rec(&tc, order, tmp, 0, n);

  /* tinygrad enumerates the stable sorted list directly:
   *   nkey = {u:i for i,u in enumerate(sorted(... + x.tuplize))}
   * Equal tuplize keys therefore still get distinct positions according to
   * their stable topo order. Do not collapse equal tuplize keys into one rank. */
  for (int i = 0; i < n; i++)
    rank[order[i]] = i;

  tuplize_pair_memo_destroy(&memo);
  free(order);
  free(tmp);
  return rank;
}

static PolyUOp *linear_rebuild_preserve_metadata(
    PolyCtx *ctx,
    PolyUOp *u,
    PolyUOp **src,
    int n_src
) {
  return (u->tag != 0 || u->tag_arg.kind != POLY_ARG_NONE)
             ? poly_uop_tagged_arg(ctx, u->op, u->dtype, src, n_src, u->arg, u->tag, u->tag_arg)
             : poly_uop(ctx, u->op, u->dtype, src, n_src, u->arg);
}

static PolyUOp *linear_replacement(
    PolyUOp *u,
    PolyUOp **old_uops,
    PolyUOp **new_uops,
    int n
) {
  for (int i = n - 1; i >= 0; i--)
    if (old_uops[i] == u) return new_uops[i];
  return u;
}

static bool linear_is_gated_store(PolyUOp *u) {
  if (!u || u->op != POLY_OP_STORE || u->n_src != 3 ||
      !poly_dtype_eq(u->src[2]->dtype, POLY_BOOL))
    return false;
  PolyUOp *address = u->src[0];
  if (address && address->op == POLY_OP_CAST && address->n_src == 1)
    address = address->src[0];
  return address && (address->op == POLY_OP_INDEX || address->op == POLY_OP_SHRINK);
}

/* Pinned tinygrad/codegen/__init__.py:152-174 line-rewrites a gated STORE to
 * IF / ungated STORE / ENDIF after graph linearization. Rebuild later line
 * sources through the same replacement map so effect dependencies point at the
 * ungated STORE, matching tinygrad's line_rewrite contract. */
static PolyUOp **linearize_gated_store_cleanup(
    PolyCtx *ctx,
    PolyUOp **linear,
    int n,
    int *n_out
) {
  int n_gated = 0;
  for (int i = 0; i < n; i++)
    if (linear_is_gated_store(linear[i])) n_gated++;
  if (n_gated == 0) {
    if (n_out) *n_out = n;
    return linear;
  }

  PolyUOp **out = malloc((size_t)(n + 2 * n_gated) * sizeof(*out));
  PolyUOp **old_uops = malloc((size_t)n * sizeof(*old_uops));
  PolyUOp **new_uops = malloc((size_t)n * sizeof(*new_uops));
  if (!out || !old_uops || !new_uops) {
    free(out);
    free(old_uops);
    free(new_uops);
    return NULL;
  }

  int n_seen = 0, n_linear = 0;
  for (int i = 0; i < n; i++) {
    PolyUOp *u = linear[i];
    PolyUOp *stack_src[64];
    PolyUOp **src = u->n_src <= 64
                        ? stack_src
                        : malloc((size_t)u->n_src * sizeof(*src));
    if (!src) {
      free(out);
      free(old_uops);
      free(new_uops);
      return NULL;
    }
    bool changed = false;
    for (int j = 0; j < u->n_src; j++) {
      src[j] = linear_replacement(u->src[j], old_uops, new_uops, n_seen);
      if (src[j] != u->src[j]) changed = true;
    }
    PolyUOp *rewritten =
        changed ? linear_rebuild_preserve_metadata(ctx, u, src, u->n_src) : u;
    if (src != stack_src) free(src);

    old_uops[n_seen] = u;
    if (linear_is_gated_store(rewritten)) {
      PolyUOp *store_src[2] = {rewritten->src[0], rewritten->src[1]};
      PolyUOp *store =
          linear_rebuild_preserve_metadata(ctx, rewritten, store_src, 2);
      PolyUOp *if_src[2] = {rewritten->src[2], rewritten->src[0]};
      PolyUOp *ifu =
          poly_uop(ctx, POLY_OP_IF, POLY_VOID, if_src, 2, poly_arg_none());
      PolyUOp *endif =
          poly_uop1(ctx, POLY_OP_ENDIF, POLY_VOID, ifu, poly_arg_none());
      new_uops[n_seen++] = store;
      out[n_linear++] = ifu;
      out[n_linear++] = store;
      out[n_linear++] = endif;
    } else {
      new_uops[n_seen++] = rewritten;
      out[n_linear++] = rewritten;
    }
  }

  free(old_uops);
  free(new_uops);
  free(linear);
  if (n_out) *n_out = n_linear;
  return out;
}

PolyUOp **poly_linearize_rewritten(PolyCtx *ctx, PolyUOp *sink, int *n_out) {
  if (n_out) *n_out = 0;
  if (!ctx || !sink) return NULL;
  /* 1. Standard toposort */
  int n;
  PolyUOp **topo = poly_toposort(ctx, sink, &n);
  if (!topo || n <= 0) return NULL;

  /* 2. Build UOp* → topo-index lookup */
  IntMap idx;
  imap_init(&idx, n);
  for (int i = 0; i < n; i++)
    imap_set(&idx, topo[i], i);

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
      for (int w = 0; w < words; w++)
        r[w] |= sr[w];
    }
    /* Remove ended ranges */
    apply_uop_ended_ranges(r, u, topo, &idx, ranges, words);
    /* RANGE: add self. Assert src[0] is the bound — additional sources
     * from poly_apply_control_flow are control-flow ordering only. */
    if (u->op == POLY_OP_RANGE) {
      assert(u->n_src >= 1 && "RANGE must have at least one source (bound)");
      assert(
          (u->src[0]->op == POLY_OP_CONST || u->src[0]->op == POLY_OP_DEFINE_VAR) &&
          "RANGE.src[0] must be CONST or DEFINE_VAR (bound)"
      );
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
        if (b < n && topo[b]->op == POLY_OP_RANGE && topo[b]->n_src > 0 &&
            topo[b]->src[0]->op == POLY_OP_CONST)
          run_count[i] *= topo[b]->src[0]->arg.i;
        bits &= bits - 1;
      }
    }

    prio[i] = uop_priority(u);
    extra[i] = INT64_MIN; /* sentinel for None (sorts before any int) */
    if (u->op == POLY_OP_PARAM) extra[i] = u->arg.i;
  }

  int *tuplize_rank = compute_tuplize_ranks(topo, n, &idx);
  if (!tuplize_rank) {
    imap_destroy(&idx);
    free(ranges);
    free(out_deg);
    free(run_count);
    free(prio);
    free(extra);
    *n_out = n;
    return topo;
  }

  /* 6. Build ideal order: sort by (run_count, priority, extra, tuplize, topo_idx).
   * Matches tinygrad's sorted(lst, key=lambda x: priorities[x]+x.tuplize). */
  int *ideal = malloc(n * sizeof(int));
  for (int i = 0; i < n; i++)
    ideal[i] = i;

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
      if (run_count[ji] > kr) {
        ideal[j + 1] = ideal[j];
        j--;
        continue;
      }
      if (run_count[ji] < kr) break;
      if (prio[ji] > kp) {
        ideal[j + 1] = ideal[j];
        j--;
        continue;
      }
      if (prio[ji] < kp) break;
      if (extra[ji] > ke) {
        ideal[j + 1] = ideal[j];
        j--;
        continue;
      }
      if (extra[ji] < ke) break;
      if (tuplize_rank[ji] > tuplize_rank[ki]) {
        ideal[j + 1] = ideal[j];
        j--;
        continue;
      }
      if (tuplize_rank[ji] < tuplize_rank[ki]) break;
      /* Final tiebreak: topo index */
      if (ji > ki) {
        ideal[j + 1] = ideal[j];
        j--;
      } else
        break;
    }
    ideal[j + 1] = ki;
  }

  /* nkey[i] = position of topo[i] in ideal order */
  int *nkey = malloc(n * sizeof(int));
  for (int i = 0; i < n; i++)
    nkey[ideal[i]] = i;

  if (poly_dump_linear_enabled()) {
    for (int i = 0; i < n; i++) {
      PolyUOp *u = topo[i];
      PolyDType ldt = linearizer_tuplize_dtype(u);
      fprintf(
          stderr,
          "[lin] topo[%d] %s run_count=%lld prio=%d nkey=%d out_deg=%d arg=%d:%lld "
          "ldt=%s/%d/%d/%d src=[",
          i, poly_op_name(u->op), (long long)run_count[i], prio[i], nkey[i], out_deg[i],
          u->arg.kind, (long long)(u->arg.kind == POLY_ARG_INT ? u->arg.i : 0),
          ldt.name ? ldt.name : "?", ldt.priority, ldt.bitsize, ldt.count
      );
      for (int j = 0; j < u->n_src; j++) {
        int si = imap_try_get(&idx, u->src[j]);
        fprintf(stderr, "%s%d", j ? "," : "", si);
      }
      fprintf(stderr, "]\n");
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
      if (--out_deg[si] == 0) heap_push(&heap, -nkey[si], u->src[j]);
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
  free(tuplize_rank);
  free(ideal);
  free(nkey);

  int cleaned_n = 0;
  PolyUOp **cleaned = linearize_gated_store_cleanup(ctx, result, rlen, &cleaned_n);
  if (!cleaned) {
    free(result);
    if (n_out) *n_out = 0;
    return NULL;
  }
  result = cleaned;
  rlen = cleaned_n;

  if (n_out) *n_out = rlen;
  return result;
}

static int cpu_thread_count(void) {
  int n_env = poly_getenv_int("CPU_COUNT", 0);
  if (n_env > 0) return n_env;
#ifdef _SC_NPROCESSORS_ONLN
  long n = sysconf(_SC_NPROCESSORS_ONLN);
  if (n > 0 && n < INT32_MAX) return (int)n;
#endif
  return 1;
}

PolyRendererCaps poly_c_renderer_caps(void) {
  bool has_threads = poly_getenv_flag_default("THREADS", true);
  return (PolyRendererCaps){
      .has_mulacc = false,
      .has_threefry = false,
      /* Pinned ClangRenderer removes EXP2, LOG2, and SIN from code_for_op;
       * the shared codegen decomposition handles them before C rendering
       * (tinygrad/renderer/cstyle.py:246-269). */
      .has_exp2 = false,
      .has_log2 = false,
      .has_sin = false,
      .has_fdiv = true,
      .has_int64 = true,
      .has_local = false,
      .has_threads = has_threads,
      /* Clang/C rendering follows tinygrad's CStyle pm_render path, which
       * inserts masked-load alt values and scalarizes vector comparisons.
       * Direct ISA backends that keep packed integer masks set this capability
       * themselves. */
      .has_simd_int = false,
      .max_vec_width = 4,
      .max_threads = has_threads ? cpu_thread_count() : 0,
  };
}

static PolyRendererCaps poly_c_direct_call_caps(void) {
  PolyRendererCaps caps = poly_c_renderer_caps();
  /* poly_linearize() is used by low-level direct-call helpers that invoke
   * poly_program_call(), not the scheduled CPU runner. CPU THREAD axes require
   * poly_program_call_threaded() to execute every core_id shard, so keep direct
   * linearization single-core and enable THREAD axes explicitly in schedule.c. */
  caps.has_threads = false;
  caps.max_threads = 0;
  return caps;
}

PolyUOp **poly_linearize(PolyCtx *ctx, PolyUOp *sink, int *n_out) {
  PolyRewriteOpts opts = {
      .optimize = poly_kernel_optimize_enabled(sink),
      .devectorize = 1,
      .caps = poly_c_direct_call_caps(),
      .device = POLY_DEVICE_CPU,
      .opt_policy = POLY_OPT_HEURISTIC,
      .dtype_matcher = poly_pm_bf16_non_native(),
      .extra_matcher = poly_pm_c_renderer_extra(),
  };
  return poly_linearize_ex(ctx, sink, opts, n_out);
}

PolyUOp **poly_linearize_ex(PolyCtx *ctx, PolyUOp *sink, PolyRewriteOpts opts, int *n_out) {
  sink = poly_full_rewrite_to_sink_ex(ctx, sink, opts);
  if (!sink) {
    if (n_out) *n_out = 0;
    return NULL;
  }
  return poly_linearize_rewritten(ctx, sink, n_out);
}

PolyUOp **poly_linearize_env(PolyCtx *ctx, PolyUOp *sink, int *n_out) {
  bool opt = poly_getenv_flag("POLY_OPTIMIZE") && poly_kernel_optimize_enabled(sink);
  /* Default: OPTIMIZE=1 implies DEVECTORIZE=1 (vec load/store, scalar ALU — safe).
   * DEVECTORIZE=0 (full vec ALU) is opt-in only. */
  int devec = poly_getenv_int("POLY_DEVECTORIZE", opt ? 1 : 0);
  int beam = poly_getenv_int("POLY_BEAM", 0);
  PolyRewriteOpts opts = {
      .optimize = opt,
      .devectorize = devec,
      .beam_width = beam,
      .caps = poly_c_direct_call_caps(),
      .device = POLY_DEVICE_CPU,
      .opt_policy = POLY_OPT_HEURISTIC,
      .dtype_matcher = poly_pm_bf16_non_native(),
      .extra_matcher = poly_pm_c_renderer_extra(),
  };
  return poly_linearize_ex(ctx, sink, opts, n_out);
}

/* Render helpers */

/* Render INT64_MIN without spelling an out-of-range positive literal followed
 * by unary minus. */
static char *render_int64_const(int64_t v, char *buf, int cap) {
  if (v == INT64_MIN)
    snprintf(buf, cap, "(-9223372036854775807ll - 1ll)");
  else
    snprintf(buf, cap, "%lldll", (long long)v);
  return buf;
}

/* Render a float constant, dtype-aware. Pinned cstyle.py:40-43 renders half
 * constants through a larger float literal and explicitly casts to half. */
static char *render_float_const(double v, PolyDType dt, char *buf, int cap) {
  PolyDType scalar = poly_dtype_scalar(dt);
  bool is_f64 = poly_dtype_eq(scalar, POLY_FLOAT64);
  bool is_f16 = poly_dtype_eq(scalar, POLY_FLOAT16);
  char literal[96];
  if (isinf(v)) {
    if (is_f64)
      snprintf(buf, cap, v > 0 ? "__builtin_inf()" : "(-__builtin_inf())");
    else if (is_f16)
      snprintf(
          buf, cap, v > 0 ? "((__fp16)(__builtin_inff()))"
                          : "((__fp16)(-__builtin_inff()))"
      );
    else
      snprintf(buf, cap, v > 0 ? "__builtin_inff()" : "(-__builtin_inff())");
    return buf;
  }
  if (isnan(v)) {
    if (is_f64)
      snprintf(buf, cap, "__builtin_nan(\"\")");
    else if (is_f16)
      snprintf(buf, cap, "((__fp16)(__builtin_nanf(\"\")))");
    else
      snprintf(buf, cap, "__builtin_nanf(\"\")");
    return buf;
  }
  if (is_f64) {
    /* Full precision double literal: enough digits to round-trip, no suffix. */
    snprintf(buf, cap, "%.17g", v);
    if (!strchr(buf, '.') && !strchr(buf, 'e') && !strchr(buf, 'E')) {
      int len = (int)strlen(buf);
      if (len + 2 < cap) {
        buf[len] = '.';
        buf[len + 1] = '0';
        buf[len + 2] = '\0';
      }
    }
    return buf;
  }
  if (is_f16) {
    /* A double round-trip literal is harmlessly rounded to f32 by the suffix,
     * then to f16 by the explicit cast, matching tinygrad's emitted C type
     * boundary without pre-rounding the argument in the renderer. */
    snprintf(literal, sizeof(literal), "%.17g", v);
    if (!strchr(literal, '.') && !strchr(literal, 'e') && !strchr(literal, 'E')) {
      int len = (int)strlen(literal);
      if (len + 2 < (int)sizeof(literal)) {
        literal[len] = '.';
        literal[len + 1] = '0';
        literal[len + 2] = '\0';
      }
    }
    snprintf(buf, cap, "((__fp16)(%sf))", literal);
    return buf;
  }
  /* Float32: truncate to float32, add the 'f' suffix. */
  snprintf(literal, sizeof(literal), "%.9g", (double)(float)v);
  if (!strchr(literal, '.') && !strchr(literal, 'e') && !strchr(literal, 'E')) {
    int len = (int)strlen(literal);
    if (len + 2 < (int)sizeof(literal)) {
      literal[len] = '.';
      literal[len + 1] = '0';
      literal[len + 2] = '\0';
    }
  }
  int len = (int)strlen(literal);
  if (len + 1 < cap) {
    snprintf(buf, cap, "%sf", literal);
  }
  return buf;
}

/* Exact C equivalent of helpers.strip_parens used by pinned
 * cstyle.py:62-63: remove one balanced outer pair, not merely the first and
 * last characters. */
static char *render_strip_parens(const char *expr) {
  if (!expr) return strdup("");
  size_t len = strlen(expr);
  if (len < 2 || expr[0] != '(' || expr[len - 1] != ')') return strdup(expr);
  int depth = 0;
  for (size_t i = 1; i + 1 < len; i++) {
    if (expr[i] == '(')
      depth++;
    else if (expr[i] == ')' && --depth < 0)
      return strdup(expr);
  }
  if (depth != 0) return strdup(expr);
  char *stripped = malloc(len - 1);
  if (!stripped) return strdup(expr);
  memcpy(stripped, expr + 1, len - 2);
  stripped[len - 2] = '\0';
  return stripped;
}

/* Render a C type for a PolyDType, including vector and pointer forms. */
static bool render_is_bf16(PolyDType dt) {
  PolyDType s = poly_dtype_scalar(dt);
  return s.priority == POLY_BFLOAT16.priority && s.bitsize == 16;
}

static void render_ctype_nonptr(PolyDType dt, char *buf, int cap) {
  if (render_is_bf16(dt)) {
    /* CPU C follows tinygrad's non-native bf16 lowering: arithmetic is
     * promoted to f32 and bf16 storage is addressed as raw 16-bit lanes. */
    if (dt.count > 1)
      snprintf(
          buf, cap, "unsigned short __attribute__((vector_size(%d)))",
          poly_dtype_itemsize(poly_dtype_scalar(dt)) * dt.count
      );
    else
      snprintf(buf, cap, "unsigned short");
    return;
  }
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
  base.ptr_size = 0;
  /* tinygrad uses ptr.count for normal ptr-to-vector width. Polygrad still has
   * a few older late-pass sites that carried that width in ptr.vcount, so
   * accept either while the remaining producers are normalized. */
  int lanes = dt.count > 1 ? dt.count : dt.vcount;
  base = poly_dtype_scalar(base);
  if (lanes > 1) base = poly_dtype_vec(base, lanes);
  char bt[128];
  render_ctype_nonptr(base, bt, sizeof(bt));
  snprintf(buf, cap, "%s*", bt);
}

static bool render_index_lane_ptr(StrMap *names, PolyUOp *ptr_uop, int lane, char *buf, int cap) {
  PolyUOp *idx = poly_find_memory_slice_through_cast(ptr_uop);
  if (!idx || idx->n_src < 2) return false;
  char *base = smap_get(names, idx->src[0]);
  char *idx_s = smap_get(names, idx->src[1]);
  if (lane == 0)
    snprintf(buf, cap, "(%s+%s)", base ? base : "0", idx_s ? idx_s : "0");
  else
    snprintf(buf, cap, "(%s+(%s)+%d)", base ? base : "0", idx_s ? idx_s : "0", lane);
  return true;
}

static void render_vector_load_expr(
    StrBuf *body, StrMap *names, PolyUOp *ptr_uop, PolyDType dtype, const char *dtype_s
) {
  sb_printf(body, "((%s){", dtype_s);
  for (int lane = 0; lane < dtype.count; lane++) {
    char lane_ptr[512];
    if (lane) sb_puts(body, ",");
    if (render_index_lane_ptr(names, ptr_uop, lane, lane_ptr, sizeof(lane_ptr)))
      sb_printf(body, "(*%s)", lane_ptr);
    else
      sb_puts(body, "0");
  }
  sb_puts(body, "})");
}

/* Render an ALU expression.
 * For vec4 types, GCC vector extensions handle +, -, *, /, <<, >>, &, |, ^,
 * <, !=, == natively. WHERE/MAX need special handling. */
static void render_alu(
    char *buf,
    int cap,
    PolyOps op,
    PolyDType dtype,
    const char *s0,
    const char *s1,
    const char *s2
) {
  bool is_vec = (dtype.count > 1);
  PolyDType sdt = poly_dtype_scalar(dtype);
  switch (op) {
  /* unary */
  case POLY_OP_NEG:
    /* Pinned CStyleLanguage.code_for_op renders every NEG as arithmetic -x
     * (renderer/cstyle.py:128-130). Assignment to bool normalizes the result. */
    snprintf(buf, cap, "(-%s)", s0); /* works on vectors */
    break;
  case POLY_OP_SQRT:
    if (is_vec) {
      /* Vec SQRT: per-element via initializer (no vec builtin) */
      char vt[128];
      render_ctype(dtype, vt, sizeof(vt));
      const char *fn = poly_dtype_eq(sdt, POLY_FLOAT64) ? "__builtin_sqrt" : "__builtin_sqrtf";
      snprintf(
          buf, cap, "((%s){%s(%s[0]),%s(%s[1]),%s(%s[2]),%s(%s[3])})", vt, fn, s0, fn, s0, fn, s0,
          fn, s0
      );
    } else {
      snprintf(
          buf, cap, poly_dtype_eq(sdt, POLY_FLOAT64) ? "__builtin_sqrt(%s)" : "__builtin_sqrtf(%s)",
          s0
      );
    }
    break;
  case POLY_OP_TRUNC:
    snprintf(
        buf, cap, poly_dtype_eq(sdt, POLY_FLOAT64) ? "__builtin_trunc(%s)" : "__builtin_truncf(%s)",
        s0
    );
    break;
  case POLY_OP_EXP2:
    snprintf(buf, cap, poly_dtype_eq(sdt, POLY_FLOAT64) ? "exp2(%s)" : "exp2f(%s)", s0);
    break;
  case POLY_OP_LOG2:
    snprintf(buf, cap, poly_dtype_eq(sdt, POLY_FLOAT64) ? "log2(%s)" : "log2f(%s)", s0);
    break;
  case POLY_OP_SIN:
    snprintf(buf, cap, poly_dtype_eq(sdt, POLY_FLOAT64) ? "sin(%s)" : "sinf(%s)", s0);
    break;
  case POLY_OP_RECIPROCAL:
    if (is_vec) {
      /* Vec RECIPROCAL: (type){1.0f,...} / x */
      char vt[128];
      render_ctype(dtype, vt, sizeof(vt));
      char one[128];
      render_ctype(dtype, one, sizeof(one));
      snprintf(buf, cap, "((%s){1.0f,1.0f,1.0f,1.0f}/%s)", vt, s0);
    } else {
      snprintf(buf, cap, "(1/%s)", s0);
    }
    break;
  /* binary — +, -, *, /, <<, >>, &, |, ^, <, !=, == all work on GCC vectors */
  case POLY_OP_ADD:
    snprintf(buf, cap, "(%s+%s)", s0, s1);
    break;
  case POLY_OP_SUB:
    snprintf(buf, cap, "(%s-%s)", s0, s1);
    break;
  case POLY_OP_MUL:
    snprintf(buf, cap, "(%s*%s)", s0, s1);
    break;
  case POLY_OP_FDIV:
    snprintf(buf, cap, "(%s/%s)", s0, s1);
    break;
  case POLY_OP_IDIV:
    snprintf(buf, cap, "(%s/%s)", s0, s1);
    break;
  case POLY_OP_MOD:
    snprintf(buf, cap, "(%s%%%s)", s0, s1);
    break;
  case POLY_OP_SHL:
    snprintf(buf, cap, "(%s<<%s)", s0, s1);
    break;
  case POLY_OP_SHR:
    snprintf(buf, cap, "(%s>>%s)", s0, s1);
    break;
  case POLY_OP_AND:
    snprintf(buf, cap, "(%s&%s)", s0, s1);
    break;
  case POLY_OP_OR:
    snprintf(buf, cap, "(%s|%s)", s0, s1);
    break;
  case POLY_OP_XOR:
    snprintf(buf, cap, "(%s^%s)", s0, s1);
    break;
  case POLY_OP_CMPLT:
    snprintf(buf, cap, "(%s<%s)", s0, s1);
    break;
  case POLY_OP_CMPNE:
    snprintf(buf, cap, "(%s!=%s)", s0, s1);
    break;
  case POLY_OP_CMPEQ:
    snprintf(buf, cap, "(%s==%s)", s0, s1);
    break;
  case POLY_OP_MAX:
    if (is_vec) {
      /* vec MAX: bitwise select using comparison mask.
       * GCC vec comparison returns -1 (all bits set) or 0 per lane. */
      char int_type[128];
      PolyDType idt = poly_dtype_is_float(sdt) ? POLY_INT32 : sdt;
      render_ctype(poly_dtype_vec(idt, dtype.count), int_type, sizeof(int_type));
      char dst_type[128];
      render_ctype(dtype, dst_type, sizeof(dst_type));
      snprintf(
          buf, cap, "((%s)(((%s)(%s>%s) & (%s)%s) | (~(%s)(%s>%s) & (%s)%s)))", dst_type, int_type,
          s0, s1, int_type, s0, int_type, s0, s1, int_type, s1
      );
    } else {
      snprintf(buf, cap, "((%s>%s)?%s:%s)", s0, s1, s0, s1);
    }
    break;
  case POLY_OP_POW:
    snprintf(buf, cap, poly_dtype_eq(sdt, POLY_FLOAT64) ? "pow(%s, %s)" : "powf(%s, %s)", s0, s1);
    break;
  /* ternary */
  case POLY_OP_WHERE:
    if (is_vec) {
      /* vec WHERE(mask, a, b): bitwise select.
       * mask is int-typed (from CMPLT: -1 or 0 per lane). */
      char int_type[128];
      PolyDType idt = poly_dtype_is_float(sdt) ? POLY_INT32 : sdt;
      render_ctype(poly_dtype_vec(idt, dtype.count), int_type, sizeof(int_type));
      char dst_type[128];
      render_ctype(dtype, dst_type, sizeof(dst_type));
      snprintf(
          buf, cap, "((%s)((%s & (%s)%s) | (~%s & (%s)%s)))", dst_type, s0, int_type, s1, s0,
          int_type, s2
      );
    } else {
      snprintf(buf, cap, "(%s?%s:%s)", s0, s1, s2);
    }
    break;
  case POLY_OP_MULACC:
    snprintf(buf, cap, "((%s*%s)+%s)", s0, s1, s2);
    break;
  default:
    snprintf(buf, cap, "/* unknown op %d */0", op);
    break;
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

/* C Renderer */

#define POLY_RENDER_MAX_PARAMS 64
typedef struct {
  char type[256];
  char name[256];
  int order; /* maps param position -> args[] index */
  bool runtime_core_id; /* tinygrad CPU THREAD runtime variable */
} RenderParam;

char *poly_render_c(PolyUOp **uops, int n, const char *fn_name) {
  StrBuf decls; /* variable declarations at function scope */
  StrBuf body; /* function body with assignments */
  sb_init(&decls);
  sb_init(&body);

  StrMap names;
  smap_init(&names, n);

  /* Pinned cstyle.py:194 counts direct consumers once, then :232-237 uses
   * that count to inline single-consumer expressions unless EXPAND_SSA. */
  IntMap uop_indices;
  imap_init(&uop_indices, n);
  int *child_count = calloc((size_t)n, sizeof(*child_count));
  if (!child_count) goto fail;
  for (int i = 0; i < n; i++) imap_set(&uop_indices, uops[i], i);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < uops[i]->n_src; j++) {
      int source_index = imap_try_get(&uop_indices, uops[i]->src[j]);
      if (source_index >= 0) child_count[source_index]++;
    }
  }
  bool expand_ssa =
      poly_getenv_flag("EXPAND_SSA") || poly_getenv_flag("POLY_EXPAND_SSA");

  /* function parameter entries: (type_str, name_str, sort_key) */
  RenderParam params[POLY_RENDER_MAX_PARAMS];
  int n_params = 0;
  int n_buffer_params = 0; /* count of PARAM (buffer) params, used for DEFINE_VAR offset */
  int n_var_params = 0; /* DEFINE_VAR params excluding runtime core_id */

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
    if (u->op == POLY_OP_RANGE) (void)range_slot(live_ranges, &n_live_ranges, u, true);
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
    if (u->op == POLY_OP_SINK || u->op == POLY_OP_NOOP || u->op == POLY_OP_GROUP) continue;

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
      if (n_params >= POLY_RENDER_MAX_PARAMS) goto fail;
      char name[32];
      snprintf(name, sizeof(name), "data%lld", (long long)u->arg.i);
      smap_set(&names, u, strdup(name));

      /* Keep parameter storage type in sync with renderer dtype mapping. This
       * matters for non-native bf16, where the kernel ABI is raw u16 storage
       * even though the logical dtype remains POLY_BFLOAT16. */
      PolyDType base = poly_dtype_scalar(u->dtype);
      char base_type[128];
      render_ctype_nonptr(base, base_type, sizeof(base_type));
      snprintf(params[n_params].type, sizeof(params[n_params].type), "%s* restrict", base_type);
      snprintf(params[n_params].name, sizeof(params[n_params].name), "%s", name);
      params[n_params].order = (int)u->arg.i;
      params[n_params].runtime_core_id = false;
      n_params++;
      n_buffer_params++;
      continue;
    }

    /* --- DEFINE_VAR: integer parameter ------------------------------ */
    if (u->op == POLY_OP_DEFINE_VAR) {
      if (n_params >= POLY_RENDER_MAX_PARAMS) goto fail;
      const char *vname = u->arg.kind == POLY_ARG_DEFINE_VAR ? u->arg.define_var.name
                                                             : (u->arg.str ? u->arg.str : "var");
      smap_set(&names, u, strdup(vname));
      snprintf(params[n_params].type, sizeof(params[n_params].type), "const int");
      snprintf(params[n_params].name, sizeof(params[n_params].name), "%s", vname);
      params[n_params].runtime_core_id = strcmp(vname, "core_id") == 0;
      if (params[n_params].runtime_core_id) {
        /* tinygrad CPU threading injects core_id as a runtime variable. It is
         * not a user DEFINE_VAR binding and is supplied by _call_core(). */
        params[n_params].order = POLY_RENDER_MAX_PARAMS + n_params;
      } else {
        /* DEFINE_VAR args come after all buffer args in the args[] array.
         * n_buffer_params counts PARAMs seen so far (all PARAMs precede
         * DEFINE_VARs in linearized output due to priority -20 vs -19). */
        params[n_params].order = n_buffer_params + n_var_params++;
      }
      n_params++;
      continue;
    }

    /* --- CONST: inline literal -------------------------------------- */
    if (u->op == POLY_OP_CONST) {
      char val[64];
      char *wide = NULL;
      PolyDType sdt = poly_dtype_scalar(u->dtype);
      if (poly_dtype_is_float(sdt)) {
        render_float_const(u->arg.f, sdt, val, sizeof(val));
      } else if (poly_dtype_is_bool(sdt)) {
        /* Vec bool: GCC vec comparisons return -1 (all bits set) for true.
         * Scalar bool: standard C true = 1. */
        if (u->dtype.count > 1)
          snprintf(val, sizeof(val), "%d", u->arg.b ? -1 : 0);
        else
          snprintf(val, sizeof(val), "%d", u->arg.b ? 1 : 0);
      } else if (poly_dtype_eq(sdt, POLY_INT64)) {
        if (u->arg.kind == POLY_ARG_BIGINT) {
          char *decimal = poly_arg_integer_to_decimal(u->arg);
          if (!decimal) return NULL;
          size_t n = strlen(decimal) + 3;
          wide = malloc(n);
          if (!wide) {
            free(decimal);
            return NULL;
          }
          snprintf(wide, n, "%sll", decimal);
          free(decimal);
        } else {
          render_int64_const(u->arg.i, val, sizeof(val));
        }
      } else if (poly_dtype_eq(sdt, POLY_UINT64)) {
        snprintf(
            val, sizeof(val), "%lluull",
            (unsigned long long)poly_arg_integer_to_u64_mod(u->arg)
        );
      } else if (poly_dtype_eq(sdt, POLY_UINT32)) {
        snprintf(
            val, sizeof(val), "%uu",
            (unsigned)(uint32_t)poly_arg_integer_to_u64_mod(u->arg)
        );
      } else if (u->arg.kind == POLY_ARG_BIGINT) {
        wide = poly_arg_integer_to_decimal(u->arg);
        if (!wide) return NULL;
      } else {
        snprintf(val, sizeof(val), "%lld", (long long)u->arg.i);
      }
      const char *literal = wide ? wide : val;
      /* Vec CONST: broadcast scalar to all lanes */
      if (u->dtype.count > 1) {
        char dtype_s[128];
        render_ctype(u->dtype, dtype_s, sizeof(dtype_s));
        StrBuf vexpr;
        sb_init(&vexpr);
        sb_printf(&vexpr, "((%s){", dtype_s);
        for (int j = 0; j < u->dtype.count; j++)
          sb_printf(&vexpr, "%s%s", j > 0 ? "," : "", literal);
        sb_puts(&vexpr, "})");
        smap_set(&names, u, vexpr.buf);
      } else {
        smap_set(&names, u, strdup(literal));
      }
      free(wide);
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
      smap_set(&names, u, vexpr.buf); /* takes ownership */
      continue;
    }

    /* --- GEP: vector lane extract ---------------------------------- */
    if (u->op == POLY_OP_GEP) {
      if (u->n_src < 1) {
        smap_set(&names, u, strdup("0"));
        continue;
      }
      char *src_s = smap_get(&names, u->src[0]);
      if (u->arg.kind == POLY_ARG_INT_TUPLE && u->arg.int_tuple.n > 0) {
        if (u->arg.int_tuple.n == 1) {
          StrBuf expr;
          sb_init(&expr);
          sb_printf(
              &expr, "(%s[%lld])", src_s ? src_s : "0", (long long)u->arg.int_tuple.vals[0]
          );
          smap_set(&names, u, expr.buf);
        } else {
          char dtype_s[128];
          render_ctype(u->dtype, dtype_s, sizeof(dtype_s));
          StrBuf vexpr;
          sb_init(&vexpr);
          sb_printf(&vexpr, "((%s){", dtype_s);
          for (int j = 0; j < u->arg.int_tuple.n; j++) {
            if (j) sb_puts(&vexpr, ",");
            sb_printf(
                &vexpr, "(%s[%lld])", src_s ? src_s : "0", (long long)u->arg.int_tuple.vals[j]
            );
          }
          sb_puts(&vexpr, "})");
          smap_set(&names, u, vexpr.buf);
        }
      } else if (u->arg.kind == POLY_ARG_INT) {
        StrBuf expr;
        sb_init(&expr);
        sb_printf(&expr, "(%s[%lld])", src_s ? src_s : "0", (long long)u->arg.i);
        smap_set(&names, u, expr.buf);
      } else {
        smap_set(&names, u, strdup(src_s ? src_s : "0"));
      }
      continue;
    }

    /* --- INDEX: pointer arithmetic or vector lane extract ------------ */
    if (u->op == POLY_OP_INDEX) {
      char *buf_s = smap_get(&names, u->src[0]);
      char *idx_s = smap_get(&names, u->src[1]);
      StrBuf expr;
      sb_init(&expr);
      if (poly_is_program_memory_base(u->src[0]))
        sb_printf(&expr, "(%s+%s)", buf_s ? buf_s : "0", idx_s ? idx_s : "0");
      else
        sb_printf(&expr, "(%s[%s])", buf_s ? buf_s : "0", idx_s ? idx_s : "0");
      smap_set(&names, u, expr.buf);
      continue;
    }

    /* --- SHRINK: late codegen memory slice -------------------------- */
    if (u->op == POLY_OP_SHRINK) {
      char *buf_s = smap_get(&names, u->src[0]);
      char *idx_s = smap_get(&names, u->src[1]);
      StrBuf expr;
      sb_init(&expr);
      sb_printf(&expr, "(%s+%s)", buf_s ? buf_s : "0", idx_s ? idx_s : "0");
      smap_set(&names, u, expr.buf);
      continue;
    }

    /* --- RANGE: for loop -------------------------------------------- */
    if (u->op == POLY_OP_RANGE) {
      char name[32];
      int64_t aid = poly_range_axis_id(u->arg);
      int n_extra = poly_range_n_extra(u->arg);
      if (n_extra > 0) {
        const int64_t *extra = poly_range_extra(u->arg);
        snprintf(
            name, sizeof(name), "ridx%lld_%lld", (long long)aid, (long long)extra[n_extra - 1]
        );
      } else {
        snprintf(name, sizeof(name), "ridx%lld", (long long)aid);
      }
      smap_set(&names, u, strdup(name));

      char *bound = smap_get(&names, u->src[0]);
      for (int d = 0; d < depth; d++)
        sb_puts(&body, "  ");
      sb_printf(&body, "for (int %s = 0; %s < %s; %s++) {\n", name, name, bound, name);
      depth++;
      if (n_open_ranges < 128) open_ranges[n_open_ranges++] = u;
      continue;
    }

    /* --- END / ENDIF: close brace ----------------------------------- */
    if (u->op == POLY_OP_END || u->op == POLY_OP_ENDIF) {
      if (u->op == POLY_OP_END && u->n_src > 1 && u->src[1]->op == POLY_OP_RANGE) {
        PolyUOp *want = u->src[1];
        int wi = range_slot(live_ranges, &n_live_ranges, want, false);
        if (wi >= 0 && live_remaining[wi] > 0) continue; /* too early */

        int pos = -1;
        for (int p = n_open_ranges - 1; p >= 0; p--) {
          if (open_ranges[p] == want) {
            pos = p;
            break;
          }
        }
        if (pos < 0) continue; /* duplicate/stale END */

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
          for (int d = 0; d < depth; d++)
            sb_puts(&body, "  ");
          sb_puts(&body, "}\n");
          n_open_ranges--;
        }
        continue;
      }

      /* Defensive: END with non-RANGE source is a structural violation.
       * Upstream invariant (rangeify/codegen) should prevent this.
       * Debug: abort loudly.  Release: skip silently as safety belt. */
      if (u->op == POLY_OP_END && u->n_src > 1 && u->src[1]->op != POLY_OP_RANGE) {
#ifndef NDEBUG
        fprintf(
            stderr,
            "polygrad: render_c: END node references non-RANGE source "
            "(op=%s) -- structural invariant violation\n",
            poly_op_name(u->src[1]->op)
        );
        assert(0 && "END source must be RANGE");
#endif
        continue;
      }

      depth--;
      for (int d = 0; d < depth; d++)
        sb_puts(&body, "  ");
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
        render_float_const(u->arg.f, u->dtype, initval, sizeof(initval));
      else
        snprintf(
            initval, sizeof(initval),
            poly_dtype_eq(poly_dtype_scalar(u->dtype), POLY_FLOAT64) ? "0.0" : "0.0f"
        );

      char dtype_s[128];
      render_ctype(u->dtype, dtype_s, sizeof(dtype_s));
      sb_printf(&decls, "  %s %s;\n", dtype_s, name);
      for (int d = 0; d < depth; d++)
        sb_puts(&body, "  ");
      sb_printf(&body, "%s = %s;\n", name, initval);
      continue;
    }

    /* --- register accumulator (float r0[1];) ------------------------ */
    if (u->op == POLY_OP_DEFINE_REG ||
        (u->op == POLY_OP_BUFFER && u->dtype.is_ptr && u->dtype.addrspace == POLY_ADDR_REG)) {
      char name[32];
      snprintf(name, sizeof(name), "r%lld", (long long)u->arg.i);
      smap_set(&names, u, strdup(name));

      /* Extract the pointer's base type (preserves vec count for vec accumulators).
       * poly_dtype_scalar strips both ptr and vec; we need ptr stripped but vec kept.
       * PtrDType stores the pointee in the base dtype fields. For ptr(float.vec(4)),
       * is_ptr=true, count=4, bitsize=128. We need "float vec4" not "float". */
      PolyDType base = u->dtype;
      base.is_ptr = false;
      base.addrspace = 0;
      base.ptr_size = 0;
      int64_t reg_size = u->dtype.ptr_size > 0 ? u->dtype.ptr_size : 1;
      if (base.count > 1) {
        /* Vec accumulator: float __attribute__((vector_size(N))) r0[1]; */
        PolyDType elem = poly_dtype_scalar(u->dtype);
        int vbytes = (int)(elem.bitsize / 8) * base.count;
        sb_printf(
            &decls, "  %s __attribute__((vector_size(%d))) %s[%lld];\n", elem.name, vbytes, name,
            (long long)reg_size
        );
      } else {
        sb_printf(&decls, "  %s %s[%lld];\n", base.name, name, (long long)reg_size);
      }
      continue;
    }

    /* --- AFTER: pass-through (use src[0]'s name) -------------------- */
    if (u->op == POLY_OP_AFTER) {
      char *src_name = smap_get(&names, u->src[0]);
      if (src_name) smap_set(&names, u, strdup(src_name));
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
      for (int d = 0; d < depth; d++)
        sb_puts(&body, "  ");

      /* Pinned tinygrad final IR: LOAD(INDEX(buf, idx), alt, gate). */
      PolyUOp *idx_uop = poly_find_index_through_cast(u->src[0]);
      bool is_lane_load =
          idx_uop && idx_uop->n_src >= 1 && !poly_is_program_memory_base(idx_uop->src[0]);
      PolyUOp *gate_uop =
          (u->n_src >= 3 && poly_dtype_is_bool(poly_dtype_scalar(u->src[2]->dtype)))
              ? u->src[2]
              : NULL;
      if (gate_uop && u->n_src >= 2) {
        char *gate_s = smap_get(&names, gate_uop);
        char *alt_s = smap_get(&names, u->src[1]);
        if (is_lane_load) {
          sb_printf(&body, "%s = (%s?%s:%s);\n", name, gate_s, bidx, alt_s);
        } else if (u->dtype.count > 1) {
          sb_printf(&body, "if (%s) %s = ", gate_s, name);
          render_vector_load_expr(&body, &names, u->src[0], u->dtype, dtype_s);
          sb_printf(&body, "; else %s = %s;\n", name, alt_s);
        } else {
          sb_printf(&body, "%s = (%s?(*%s):%s);\n", name, gate_s, bidx, alt_s);
        }
      } else if (gate_uop) {
        char *gate_s = smap_get(&names, gate_uop);
        if (is_lane_load) {
          sb_printf(&body, "%s = (%s?%s:(%s)0);\n", name, gate_s, bidx, dtype_s);
        } else if (u->dtype.count > 1) {
          sb_printf(&body, "if (%s) %s = ", gate_s, name);
          render_vector_load_expr(&body, &names, u->src[0], u->dtype, dtype_s);
          sb_printf(&body, "; else memset(&%s, 0, sizeof(%s));\n", name, name);
        } else {
          sb_printf(&body, "%s = (%s?(*%s):(%s)0);\n", name, gate_s, bidx, dtype_s);
        }
      } else {
        if (is_lane_load) {
          sb_printf(&body, "%s = %s;\n", name, bidx);
        } else if (u->dtype.count > 1 && poly_find_memory_slice_through_cast(u->src[0])) {
          sb_printf(&body, "%s = ", name);
          render_vector_load_expr(&body, &names, u->src[0], u->dtype, dtype_s);
          sb_puts(&body, ";\n");
        } else {
          sb_printf(&body, "%s = (*%s);\n", name, bidx);
        }
      }
      continue;
    }

    /* --- STORE: write to indexed pointer or accumulator ------------- */
    if (u->op == POLY_OP_STORE) {
      char *target = smap_get(&names, u->src[0]);
      char *val = smap_get(&names, u->src[1]);
      for (int d = 0; d < depth; d++)
        sb_puts(&body, "  ");
      /* Guard: STORE src[0] is always set by construction, but null-check
       * satisfies the analyzer's path-sensitive null-deref tracking. */
      if (u->src[0] &&
          (u->src[0]->op == POLY_OP_DEFINE_LOCAL ||
           (u->src[0]->op == POLY_OP_BUFFER && u->src[0]->dtype.is_ptr &&
            u->src[0]->dtype.addrspace == POLY_ADDR_LOCAL)))
        sb_printf(&body, "%s = %s;\n", target, val);
      else if (u->src[1] && u->src[1]->dtype.count > 1 &&
               poly_find_memory_slice_through_cast(u->src[0])) {
        for (int lane = 0; lane < u->src[1]->dtype.count; lane++) {
          char lane_ptr[512];
          if (!render_index_lane_ptr(&names, u->src[0], lane, lane_ptr, sizeof(lane_ptr))) break;
          if (lane) {
            for (int d = 0; d < depth; d++)
              sb_puts(&body, "  ");
          }
          sb_printf(&body, "*%s = (%s)[%d];\n", lane_ptr, val, lane);
        }
      } else
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
      for (int d = 0; d < depth; d++)
        sb_puts(&body, "  ");

      /* Vector → vector CAST: __builtin_convertvector (tinygrad cstyle.py:24) */
      if (u->dtype.count > 1 && u->src[0]->dtype.count > 1 && !u->dtype.is_ptr) {
        sb_printf(&body, "%s = __builtin_convertvector(%s, %s);\n", name, src_s, dtype_s);
      }
      /* Scalar → vector CAST (non-pointer): broadcast via initializer */
      else if (u->dtype.count > 1 && u->src[0]->dtype.count <= 1 && !u->dtype.is_ptr) {
        char scalar_type[128];
        render_ctype(poly_dtype_scalar(u->dtype), scalar_type, sizeof(scalar_type));
        sb_printf(&body, "{ %s _sc = (%s)(%s); ", scalar_type, scalar_type, src_s);
        sb_printf(&body, "%s = (%s){", name, dtype_s);
        for (int vi = 0; vi < u->dtype.count; vi++)
          sb_printf(&body, "%s_sc", vi > 0 ? "," : "");
        sb_printf(&body, "}; }\n");
      }
      /* Vector → scalar: extract element 0, then cast */
      else if (u->dtype.count <= 1 && u->src[0]->dtype.count > 1) {
        sb_printf(&body, "%s = (%s)((%s)[0]);\n", name, dtype_s, src_s);
      } else {
        sb_printf(&body, "%s = (%s)(%s);\n", name, dtype_s, src_s);
      }
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
      for (int d = 0; d < depth; d++)
        sb_puts(&body, "  ");
      /* Vector → scalar bitcast: extract element 0, then reinterpret */
      if (u->dtype.count <= 1 && u->src[0]->dtype.count > 1) {
        char elem_type[128];
        render_ctype(poly_dtype_scalar(u->src[0]->dtype), elem_type, sizeof(elem_type));
        sb_printf(
            &body, "{ %s _bc = (%s)[0]; memcpy(&%s, &_bc, sizeof(%s)); }\n", elem_type, src_s, name,
            name
        );
      } else {
        sb_printf(
            &body, "{ %s _bc = %s; memcpy(&%s, &_bc, sizeof(%s)); }\n", src_type, src_s, name, name
        );
      }
      continue;
    }

    /* --- ALU ops: arithmetic expressions ---------------------------- */
    if (poly_opset_has(POLY_GROUP_ALU, u->op)) {
      const bool associative =
          u->op == POLY_OP_ADD || u->op == POLY_OP_MUL || u->op == POLY_OP_XOR ||
          u->op == POLY_OP_OR || u->op == POLY_OP_AND;
      char *stripped[3] = {NULL, NULL, NULL};
      const char *sources[3] = {"", "", ""};
      for (int j = 0; j < u->n_src && j < 3; j++) {
        const char *source = smap_get(&names, u->src[j]);
        if (associative && u->src[j]->op == u->op) {
          stripped[j] = render_strip_parens(source);
          sources[j] = stripped[j];
        } else {
          sources[j] = source ? source : "";
        }
      }
      const char *s0 = sources[0], *s1 = sources[1], *s2 = sources[2];
      size_t expr_cap = strlen(s0 ? s0 : "") + strlen(s1 ? s1 : "") + strlen(s2 ? s2 : "") + 1024;
      char *expr = malloc(expr_cap);
      if (!expr) expr = strdup("0");
      else render_alu(expr, (int)expr_cap, u->op, u->dtype, s0 ? s0 : "", s1 ? s1 : "", s2 ? s2 : "");
      for (int j = 0; j < 3; j++) free(stripped[j]);

      /* Pinned cstyle.py:232-237 keeps WHERE materialized but directly embeds
       * a one-use ALU expression in its consumer by default. */
      if (u->op != POLY_OP_WHERE && child_count[i] == 1 && !expand_ssa) {
        smap_set(&names, u, expr);
        continue;
      }

      char name[32];
      snprintf(name, sizeof(name), "alu%d", c_alu++);
      smap_set(&names, u, strdup(name));

      char dtype_s[128];
      render_ctype(u->dtype, dtype_s, sizeof(dtype_s));
      sb_printf(&decls, "  %s %s;\n", dtype_s, name);
      for (int d = 0; d < depth; d++)
        sb_puts(&body, "  ");
      sb_printf(&body, "%s = %s;\n", name, expr);
      free(expr);
      continue;
    }

    /* --- IF: conditional -------------------------------------------- */
    if (u->op == POLY_OP_IF) {
      char *cond_s = smap_get(&names, u->src[0]);
      for (int d = 0; d < depth; d++)
        sb_puts(&body, "  ");
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
        range_sizes[n_ranges] = (uops[i]->n_src > 0 && uops[i]->src[0]->op == POLY_OP_CONST)
                                    ? uops[i]->src[0]->arg.i
                                    : -1;
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
    fprintf(
        stderr,
        "polygrad render_c: DEPTH MISMATCH: depth=%d (expected 1) "
        "%d RANGEs, %d ENDs (%d without RANGE ref)\n",
        depth, n_ranges, n_ends, n_ends_norange
    );
    for (int r = 0; r < n_ranges; r++) {
      fprintf(
          stderr, "  RANGE[%d] %p size=%lld %s\n", r, (void *)range_ptrs[r],
          (long long)range_sizes[r], (range_end_count[r] > 0) ? "HAS_END" : "ORPHAN"
      );
      fprintf(stderr, "    END count: %d\n", range_end_count[r]);
    }
  }

  /* Sort params by arg index (PARAM 0, 1, 2, ...) */
  for (int i = 1; i < n_params; i++) {
    RenderParam kp = params[i];
    int j = i - 1;
    while (j >= 0 && params[j].order > kp.order) {
      params[j + 1] = params[j];
      j--;
    }
    params[j + 1] = kp;
  }

  /* Build complete source */
  StrBuf out;
  sb_init(&out);
  sb_puts(&out, "#include <math.h>\n");
  sb_puts(&out, "#include <stdbool.h>\n");
  sb_puts(&out, "#include <string.h>\n");

  /* function signature */
  sb_printf(&out, "void %s(", fn_name);
  for (int i = 0; i < n_params; i++) {
    if (i > 0) sb_puts(&out, ", ");
    sb_printf(&out, "%s %s", params[i].type, params[i].name);
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
    int arg_idx = params[i].order;
    if (params[i].runtime_core_id) {
      sb_puts(&out, "0");
      continue;
    }
    /* pointer params: (type*)args[i]; int params: *(int*)args[i] */
    if (strchr(params[i].type, '*')) {
      /* extract base type (before '* restrict') */
      char base[128];
      const char *star = strchr(params[i].type, '*');
      int blen = (int)(star - params[i].type);
      if (blen >= (int)sizeof(base)) blen = (int)sizeof(base) - 1;
      memcpy(base, params[i].type, blen);
      base[blen] = '\0';
      sb_printf(&out, "(%s*)args[%d]", base, arg_idx);
    } else {
      sb_printf(&out, "*(int*)args[%d]", arg_idx);
    }
  }
  sb_puts(&out, ");\n}\n");

  /* _call_core wrapper: same ABI as tinygrad's CPU runtimevars path. The
   * runtime calls the same compiled kernel once per worker with a different
   * core_id, while non-threaded kernels ignore the second argument. */
  sb_printf(&out, "void %s_call_core(void **args, int core_id) {\n", fn_name);
  sb_printf(&out, "  %s(", fn_name);
  for (int i = 0; i < n_params; i++) {
    if (i > 0) sb_puts(&out, ", ");
    int arg_idx = params[i].order;
    if (params[i].runtime_core_id) {
      sb_puts(&out, "core_id");
      continue;
    }
    if (strchr(params[i].type, '*')) {
      char base[128];
      const char *star = strchr(params[i].type, '*');
      int blen = (int)(star - params[i].type);
      if (blen >= (int)sizeof(base)) blen = (int)sizeof(base) - 1;
      memcpy(base, params[i].type, blen);
      base[blen] = '\0';
      sb_printf(&out, "(%s*)args[%d]", base, arg_idx);
    } else {
      sb_printf(&out, "*(int*)args[%d]", arg_idx);
    }
  }
  sb_puts(&out, ");\n}\n");

  /* cleanup */
  free(decls.buf);
  free(body.buf);
  free(child_count);
  imap_destroy(&uop_indices);
  smap_destroy(&names);

  return out.buf;

fail:
  free(decls.buf);
  free(body.buf);
  free(child_count);
  imap_destroy(&uop_indices);
  smap_destroy(&names);
  return NULL;
}
