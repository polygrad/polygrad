/*
 * Current Tinygrad 2026-08-22/a9069c177a9d codegen/late/linearizer.py.
 * C storage and tuple-comparison mechanics implement CFGContext and linearize.
 */

#define _POSIX_C_SOURCE 200809L

#include "codegen/late/linearizer.h"
#include "codegen/codegen.h"
#include "bigint.h"
#include "ctx.h"
#include "utils.h"
#include "uop/upat.h"
#include "uop/ops.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <limits.h>

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
  case POLY_OP_BUFFER:
    return poly_program_memory_is(u, POLY_ADDR_LOCAL) ? -17 : -18;
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
    free(extra_dep);
    return NULL;
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
    uint8_t *visit,
    bool *failed
) {
  if (*failed) return NULL;
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

  /* Pinned tinygrad/codegen/late/linearizer.py:83-85 replaces only a matched
   * RANGE and otherwise preserves arbitrary source-tuple arity. Allocate the
   * exact temporary width here: wide SINK/AFTER parents are legal and must not
   * inherit a renderer-local 64-source cap. */
  PolyUOp *src_stack[64];
  size_t src_cap = (size_t)u->n_src + 1;
  PolyUOp **src = src_cap <= 64 ? src_stack : malloc(src_cap * sizeof(PolyUOp *));
  if (!src) {
    *failed = true;
    return NULL;
  }
  int ns = u->n_src;
  bool changed = false;
  for (int j = 0; j < ns; j++) {
    src[j] = cf_rewrite(ctx, u->src[j], topo, idx, extra_dep, memo, visit, failed);
    if (*failed) {
      if (src != src_stack) free(src);
      return NULL;
    }
    if (src[j] != u->src[j]) changed = true;
  }

  /* Append control-flow dep for RANGE nodes */
  if (u->op == POLY_OP_RANGE && extra_dep[ui] >= 0) {
    PolyUOp *dep = cf_rewrite(ctx, topo[extra_dep[ui]], topo, idx, extra_dep, memo, visit, failed);
    if (*failed) {
      if (src != src_stack) free(src);
      return NULL;
    }
    /* Dedup: skip if dep already a source (after rewrite) */
    bool dup = false;
    for (int j = 0; j < ns; j++) {
      if (src[j] == dep) {
        dup = true;
        break;
      }
    }
    if (!dup) {
      /* PolyUOp currently stores n_src as uint16_t. Pinned tinygrad tuples can
       * grow beyond this, but returning a partial graph with a NULL RANGE is
       * never valid. Fail this pass cleanly until that representation debt is
       * explicitly approved and migrated. */
      if (ns == UINT16_MAX) {
        if (src != src_stack) free(src);
        *failed = true;
        return NULL;
      }
      src[ns++] = dep;
      changed = true;
    }
  }

  if (changed) {
    /* Pinned UOp.replace and GraphRewrite preserve metadata when sources are
     * rebuilt (uop/ops.py:156-161,1631-1633). */
    memo[ui] = (u->tag != 0 || u->tag_arg.kind != POLY_ARG_NONE)
                   ? poly_uop_tagged_arg(ctx, u->op, u->dtype, src, ns, u->arg, u->tag, u->tag_arg)
                   : poly_uop(ctx, u->op, u->dtype, src, ns, u->arg);
    if (!memo[ui]) *failed = true;
  } else {
    memo[ui] = u;
  }
  if (src != src_stack) free(src);

  if (*failed) return NULL;

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
  if (!extra_dep) {
    imap_destroy(&idx);
    poly_ctx_scratch_rewind(ctx, scratch);
    return NULL;
  }

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
  if (!memo || !visit) {
    free(visit);
    free(memo);
    free(extra_dep);
    imap_destroy(&idx);
    poly_ctx_scratch_rewind(ctx, scratch);
    return NULL;
  }
  bool failed = false;
  PolyUOp *result = cf_rewrite(ctx, sink, topo, &idx, extra_dep, memo, visit, &failed);
  if (failed) result = NULL;

#ifndef NDEBUG
  /* Pinned RANGE accepts any same-dtype sint bound expression; additional
   * sources from control-flow edges are ordering-only (uop/spec.py:73-76). */
  for (int i = 0; i < n; i++) {
    if (memo[i] && memo[i]->op == POLY_OP_RANGE && memo[i]->n_src > 0) {
      PolyUOp *bound = memo[i]->src[0];
      if (!poly_dtype_eq(bound->dtype, memo[i]->dtype)) {
        fprintf(stderr, "cf_rewrite: RANGE[%d] bound dtype differs from RANGE dtype\n", i);
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

static bool arg_is_python_int(PolyArg arg) {
  return arg.kind == POLY_ARG_BOOL || arg.kind == POLY_ARG_INT || arg.kind == POLY_ARG_BIGINT;
}

static int dtype_lt_cmp(PolyDType a, PolyDType b);

static int nullable_string_cmp(const char *a, const char *b) {
  if (a == b) return 0;
  if (!a || !b) return a ? 1 : -1;
  int ret = strcmp(a, b);
  return (ret > 0) - (ret < 0);
}

static int param_arg_cmp(const PolyParamArg *a, const PolyParamArg *b) {
  /* Current tinygrad ParamArg is an ordered dataclass (uop/ops.py:22-35).
   * Compare the C fields in the same declaration order. */
  if (a == b) return 0;
  if (!a || !b) return a ? 1 : -1;
  if (a->slot != b->slot) return a->slot < b->slot ? -1 : 1;
  int ret = dtype_lt_cmp(a->dtype, b->dtype);
  if (ret) return ret;
  if (a->has_minmax != b->has_minmax) return a->has_minmax ? 1 : -1;
  if (a->has_minmax) {
    if (a->min_val != b->min_val) return a->min_val < b->min_val ? -1 : 1;
    if (a->max_val != b->max_val) return a->max_val < b->max_val ? -1 : 1;
  }
  if (a->has_multiple_of != b->has_multiple_of) return a->has_multiple_of ? 1 : -1;
  if (a->has_multiple_of && a->multiple_of != b->multiple_of)
    return a->multiple_of < b->multiple_of ? -1 : 1;
  if ((ret = nullable_string_cmp(a->name, b->name))) return ret;
  if (a->addrspace != b->addrspace) return a->addrspace < b->addrspace ? -1 : 1;
  if (a->has_axis != b->has_axis) return a->has_axis ? 1 : -1;
  if (a->has_axis && a->axis != b->axis) return a->axis < b->axis ? -1 : 1;
  if (a->device_is_tuple != b->device_is_tuple) return a->device_is_tuple ? 1 : -1;
  if (a->device_is_tuple) {
    int n = a->n_devices < b->n_devices ? a->n_devices : b->n_devices;
    for (int i = 0; i < n; i++) {
      const char *ad = a->devices ? a->devices[i] : NULL;
      const char *bd = b->devices ? b->devices[i] : NULL;
      if ((ret = nullable_string_cmp(ad, bd))) return ret;
    }
    if (a->n_devices != b->n_devices) return a->n_devices < b->n_devices ? -1 : 1;
  } else if ((ret = nullable_string_cmp(a->device, b->device))) {
    return ret;
  }
  return a->volatile_ == b->volatile_ ? 0 : a->volatile_ ? 1 : -1;
}

static int arg_cmp(PolyArg a, PolyArg b, bool *unordered) {
  /* Current tinygrad UOp.tuplize uses Python tuple comparison. Numeric
   * bool/int/float arguments compare by value even when their types differ. */
  *unordered = false;
  if (a.kind != b.kind) {
    bool a_int = arg_is_python_int(a), b_int = arg_is_python_int(b);
    if (a_int && b_int) {
      bool ok = false;
      int ret = poly_arg_integer_cmp(a, b, &ok);
      if (ok) return ret;
    } else if (a_int && b.kind == POLY_ARG_FLOAT) {
      if (isnan(b.f)) {
        *unordered = true;
        return 0;
      }
      return poly_arg_integer_cmp_float(a, b.f);
    } else if (a.kind == POLY_ARG_FLOAT && b_int) {
      if (isnan(a.f)) {
        *unordered = true;
        return 0;
      }
      return -poly_arg_integer_cmp_float(b, a.f);
    }
    return a.kind < b.kind ? -1 : 1;
  }
  switch (a.kind) {
  case POLY_ARG_NONE:
    return 0;
  case POLY_ARG_INT:
    return a.i < b.i ? -1 : (a.i > b.i ? 1 : 0);
  case POLY_ARG_BIGINT: {
    bool ok = false;
    int ret = poly_arg_integer_cmp(a, b, &ok);
    return ok ? ret : 0;
  }
  case POLY_ARG_FLOAT:
    if (isnan(a.f) != isnan(b.f)) {
      *unordered = true;
      return 0;
    }
    return (a.f < b.f) ? -1 : (a.f > b.f ? 1 : 0);
  case POLY_ARG_BOOL:
    return a.b == b.b ? 0 : a.b ? 1 : -1;
  case POLY_ARG_PARAM:
    return param_arg_cmp(a.param, b.param);
  case POLY_ARG_RANGE:
    if (a.range.axis_id != b.range.axis_id) return a.range.axis_id < b.range.axis_id ? -1 : 1;
    if (a.range.axis_type != b.range.axis_type)
      return a.range.axis_type < b.range.axis_type ? -1 : 1;
    if (a.range.n_extra != b.range.n_extra) return a.range.n_extra < b.range.n_extra ? -1 : 1;
    for (int i = 0; i < a.range.n_extra; i++) {
      if (a.range.extra[i] != b.range.extra[i]) return a.range.extra[i] < b.range.extra[i] ? -1 : 1;
    }
    return 0;
  case POLY_ARG_REDUCE:
    if (a.reduce.op != b.reduce.op) return a.reduce.op < b.reduce.op ? -1 : 1;
    return a.reduce.num_axes < b.reduce.num_axes ? -1
                                                 : (a.reduce.num_axes > b.reduce.num_axes ? 1 : 0);
  case POLY_ARG_DTYPE:
    if (poly_dtype_eq(a.dtype, b.dtype)) return 0;
    if (a.dtype.priority != b.dtype.priority) return a.dtype.priority < b.dtype.priority ? -1 : 1;
    if (a.dtype.bitsize != b.dtype.bitsize) return a.dtype.bitsize < b.dtype.bitsize ? -1 : 1;
    return strcmp(a.dtype.name ? a.dtype.name : "", b.dtype.name ? b.dtype.name : "");
  default:
    return 0;
  }
}

static int dtype_lt_cmp(PolyDType a, PolyDType b) {
  /* Tinygrad 2026-08-22/a9069c177a9d DType ordering compares the scalar
   * dataclass fields (priority, bitsize, name, fmt). */
  if (a.priority != b.priority) return a.priority < b.priority ? -1 : 1;
  if (a.bitsize != b.bitsize) return a.bitsize < b.bitsize ? -1 : 1;
  if (a.name && b.name) {
    int nc = strcmp(a.name, b.name);
    if (nc != 0) return nc;
  } else if (a.name != b.name) {
    return a.name ? 1 : -1;
  }
  if (a.fmt != b.fmt) return a.fmt < b.fmt ? -1 : 1;
  return 0;
}

static PolyDType linearizer_tuplize_dtype(PolyUOp *u) {
  return u ? u->dtype : POLY_VOID;
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
  while (cap < n * 4)
    cap <<= 1;
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
    while (nm.keys[pos])
      pos = (pos + 1) & mask;
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
  while (m->keys[pos] && m->keys[pos] != stored)
    pos = (pos + 1) & mask;
  if (!m->keys[pos]) {
    m->keys[pos] = stored;
    m->len++;
  }
  m->vals[pos] = (int8_t)(val == 2 ? 2 : (val > 0) - (val < 0));
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
    bool arg_unordered = false;
    int ac = arg_cmp(a->arg, b->arg, &arg_unordered);
    if (arg_unordered) {
      ret = 2;
    } else if (ac != 0) {
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

  if (ret != 2) ret = (ret > 0) - (ret < 0);
  tuplize_pair_memo_set(tc->memo, key, ret);
  tuplize_pair_memo_set(tc->memo, tuplize_pair_key(bi, ai), ret == 2 ? 2 : -ret);
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
    if (tuplize_cmp_idx(tc, arr[i], arr[j]) != 1)
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

PolyUOp **poly_linearize(PolyCtx *ctx, PolyUOp *sink, int *n_out) {
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
    /* RANGE source 0 is any same-dtype sint bound expression; additional
     * sources are control-flow ordering only (uop/spec.py:73-76). */
    if (u->op == POLY_OP_RANGE) {
      assert(u->n_src >= 1 && "RANGE must have at least one source (bound)");
      assert(poly_dtype_eq(u->dtype, u->src[0]->dtype) && "RANGE bound dtype must match");
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
        if (b < n && topo[b]->op == POLY_OP_RANGE && topo[b]->n_src > 0) {
          /* Literal pinned priority is int(range.vmax)+1, not only a literal
           * CONST bound (codegen/late/linearizer.py:19-20). */
          int64_t rmin = 0, rmax = 0;
          poly_uop_minmax(ctx, topo[b], &rmin, &rmax);
          int64_t extent = rmax == INT64_MAX ? INT64_MAX : rmax + 1;
          int64_t product = 0;
          if (__builtin_mul_overflow(run_count[i], extent, &product))
            run_count[i] = INT64_MAX;
          else
            run_count[i] = product;
        }
        bits &= bits - 1;
      }
    }

    prio[i] = uop_priority(u);
    extra[i] = INT64_MIN; /* sentinel for None (sorts before any int) */
    if (u->op == POLY_OP_PARAM) extra[i] = poly_program_buffer_slot(u);
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
          "ldt=%s/%d/%d src=[",
          i, poly_op_name(u->op), (long long)run_count[i], prio[i], nkey[i], out_deg[i],
          u->arg.kind, (long long)(u->arg.kind == POLY_ARG_INT ? u->arg.i : 0),
          ldt.name ? ldt.name : "?", ldt.priority, ldt.bitsize
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

  if (n_out) *n_out = rlen;
  return result;
}
static int cmp_range_axis_id(const void *a, const void *b) {
  const PolyUOp *ra = *(const PolyUOp *const *)a;
  const PolyUOp *rb = *(const PolyUOp *const *)b;
  int64_t ia = poly_range_axis_id(ra->arg);
  int64_t ib = poly_range_axis_id(rb->arg);
  return (ia > ib) - (ia < ib);
}

/* Current Tinygrad 2026-08-22/a9069c177a9d
 * codegen/late/linearizer.py:do_split_ends. */
static PolyUOp *do_split_ends(PolyCtx *ctx, PolyUOp *end, const PolyBindings *b) {
  (void)b;
  if (end->op != POLY_OP_END || end->n_src == 0) return NULL;
  if (end->n_src == 1) return end->src[0];

  bool needs_split = false;
  for (int i = 1; i < end->n_src; i++) {
    if (end->src[i]->op != POLY_OP_RANGE) {
      needs_split = true;
      break;
    }
  }
  if (!needs_split && end->n_src <= 2) return NULL;

  PolyUOp *range_sink =
      poly_uop(ctx, POLY_OP_SINK, POLY_VOID, end->src + 1, end->n_src - 1, poly_arg_none());
  int n_topo = 0;
  PolyUOp **range_topo = poly_toposort_alloc(ctx, range_sink, &n_topo);
  if (!range_topo || n_topo <= 0) {
    free(range_topo);
    return NULL;
  }
  PolyUOp **ranges = malloc((size_t)n_topo * sizeof(*ranges));
  if (!ranges) {
    free(range_topo);
    return NULL;
  }
  int n_ranges = poly_uop_ranges(ctx, range_sink, ranges, n_topo);
  free(range_topo);
  if (n_ranges == 0) {
    free(ranges);
    return end->src[0];
  }

  qsort(ranges, (size_t)n_ranges, sizeof(*ranges), cmp_range_axis_id);
  PolyUOp *ret = end->src[0];
  for (int i = n_ranges - 1; i >= 0; i--) {
    PolyUOp *src[2] = {ret, ranges[i]};
    ret = poly_uop(ctx, POLY_OP_END, POLY_VOID, src, 2, poly_arg_none());
  }
  free(ranges);
  return ret;
}

static _Thread_local PolyPatternMatcher *g_pm_split_ends = NULL;
PolyPatternMatcher *poly_pm_split_ends(void) {
  if (g_pm_split_ends) return g_pm_split_ends;
  PolyRule rules[] = {
      {poly_upat_allow_any_len(poly_upat_op(POLY_OP_END, NULL, 0, "end")), do_split_ends},
  };
  g_pm_split_ends =
      poly_pm_thread_cache(poly_pm_new(rules, (int)(sizeof(rules) / sizeof(rules[0]))));
  return g_pm_split_ends;
}
