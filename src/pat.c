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
#ifndef __EMSCRIPTEN__
#include <pthread.h>
#endif

/* Cached compiler matchers are thread-local because rule statistics and rewrite
 * traces are mutable. POSIX destroys the TLS pointer slots when a user thread
 * exits, but C11 _Thread_local has no destructor for the matcher/pattern
 * allocations reachable from those slots. Explicitly registered compiler
 * caches own those allocations; ordinary public patterns and matchers retain
 * their caller-managed lifetime. The current Emscripten build is single-threaded
 * and keeps its caches for the lifetime of the module. */
typedef struct PolyPatThreadOwned {
  void *ptr;
  bool matcher;
  struct PolyPatThreadOwned *next;
} PolyPatThreadOwned;

typedef struct {
  PolyPatThreadOwned *head;
} PolyPatThreadRegistry;

static void pat_free_one(PolyPat *p);
static void pm_destroy_one(PolyPatternMatcher *pm);

#ifndef __EMSCRIPTEN__
static pthread_key_t g_pat_thread_key;
static pthread_once_t g_pat_thread_key_once = PTHREAD_ONCE_INIT;
static bool g_pat_thread_key_ready = false;

static void pat_thread_registry_destroy(void *opaque) {
  PolyPatThreadRegistry *registry = opaque;
  if (!registry) return;
  for (PolyPatThreadOwned *owned = registry->head; owned; owned = owned->next)
    if (owned->matcher) pm_destroy_one((PolyPatternMatcher *)owned->ptr);
  for (PolyPatThreadOwned *owned = registry->head; owned; owned = owned->next)
    if (!owned->matcher) pat_free_one((PolyPat *)owned->ptr);
  while (registry->head) {
    PolyPatThreadOwned *owned = registry->head;
    registry->head = owned->next;
    free(owned);
  }
  free(registry);
}

static void pat_thread_key_init(void) {
  g_pat_thread_key_ready = pthread_key_create(&g_pat_thread_key, pat_thread_registry_destroy) == 0;
}

static PolyPatThreadRegistry *pat_thread_registry_get(bool create) {
  if (pthread_once(&g_pat_thread_key_once, pat_thread_key_init) != 0 || !g_pat_thread_key_ready)
    return NULL;
  PolyPatThreadRegistry *registry = pthread_getspecific(g_pat_thread_key);
  if (!registry && create) {
    registry = calloc(1, sizeof(*registry));
    if (!registry) return NULL;
    if (pthread_setspecific(g_pat_thread_key, registry) != 0) {
      free(registry);
      return NULL;
    }
  }
  return registry;
}

static bool pat_thread_register(void *ptr, bool matcher) {
  if (!ptr) return false;
  PolyPatThreadRegistry *registry = pat_thread_registry_get(true);
  if (!registry) return false;
  for (PolyPatThreadOwned *owned = registry->head; owned; owned = owned->next)
    if (owned->ptr == ptr && owned->matcher == matcher) return true;
  PolyPatThreadOwned *owned = malloc(sizeof(*owned));
  if (!owned) return false;
  *owned = (PolyPatThreadOwned){.ptr = ptr, .matcher = matcher, .next = registry->head};
  registry->head = owned;
  return true;
}

static void pat_thread_unregister(void *ptr, bool matcher) {
  if (!ptr) return;
  PolyPatThreadRegistry *registry = pat_thread_registry_get(false);
  if (!registry) return;
  PolyPatThreadOwned **link = &registry->head;
  while (*link) {
    PolyPatThreadOwned *owned = *link;
    if (owned->ptr == ptr && owned->matcher == matcher) {
      *link = owned->next;
      free(owned);
      return;
    }
    link = &owned->next;
  }
}
#else
static bool pat_thread_register(void *ptr, bool matcher) {
  (void)matcher;
  return ptr != NULL;
}

static void pat_thread_unregister(void *ptr, bool matcher) {
  (void)ptr;
  (void)matcher;
}
#endif

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
  return calloc(1, sizeof(PolyPat));
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

static void pat_free_one(PolyPat *p) {
  if (!p) return;
  free(p->src);
  free(p->dtypes);
  free(p);
}

void poly_pat_free(PolyPat *p) {
  if (!p) return;
  for (int i = 0; p->src && i < p->n_src; i++)
    poly_pat_free(p->src[i]);
  pat_thread_unregister(p, false);
  pat_free_one(p);
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
  if (b->extra) free(b->extra);
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

typedef bool (*PatMatchContinuation)(PolyBindings *binds, void *opaque);

typedef struct {
  const PolyPat *pat;
  PolyUOp *uop;
  int src_index;
  bool swapped;
  PatMatchContinuation continuation;
  void *opaque;
} PatSourceMatch;

static bool pat_match_continue(
    const PolyPat *pat, PolyUOp *uop, PolyBindings *binds,
    PatMatchContinuation continuation, void *opaque, bool allow_cast
);

static bool pat_match_sources_continue(PolyBindings *binds, void *opaque) {
  PatSourceMatch *state = opaque;
  if (state->src_index == state->pat->n_src)
    return state->continuation(binds, state->opaque);

  int uop_index = state->swapped ? 1 - state->src_index : state->src_index;
  PatSourceMatch next = {
      .pat = state->pat,
      .uop = state->uop,
      .src_index = state->src_index + 1,
      .swapped = state->swapped,
      .continuation = state->continuation,
      .opaque = state->opaque,
  };
  return pat_match_continue(
      state->pat->src[state->src_index], state->uop->src[uop_index], binds,
      pat_match_sources_continue, &next, true
  );
}

static bool pat_match_continue(
    const PolyPat *pat, PolyUOp *uop, PolyBindings *binds,
    PatMatchContinuation continuation, void *opaque, bool allow_cast
) {
  int saved = binds->n;

  if (pat->has_ops && !poly_opset_has(pat->ops, uop->op)) goto no_match;

  /* Name binding: if already bound, it must be the same UOp occurrence. */
  if (pat->name) {
    PolyUOp *existing = poly_bind(binds, pat->name);
    if (existing) {
      if (existing != uop) goto no_match;
    } else if (!bindings_add(binds, pat->name, uop)) {
      goto no_match;
    }
  }

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
    if (!found) goto no_match;
  }

  if (pat->match_arg && !poly_arg_eq(pat->arg, uop->arg)) goto no_match;
  if (uop->n_src < pat->n_src) goto no_match;
  if (pat->strict_length && uop->n_src != pat->n_src) goto no_match;

  if (pat->src == NULL) {
    if (continuation(binds, opaque)) return true;
    goto no_match;
  }

  int source_saved = binds->n;
  PatSourceMatch sources = {
      .pat = pat,
      .uop = uop,
      .src_index = 0,
      .swapped = false,
      .continuation = continuation,
      .opaque = opaque,
  };
  if (pat_match_sources_continue(binds, &sources)) return true;
  binds->n = source_saved;

  /*
   * Pinned tinygrad UPat expands commutative source lists into permutations
   * (uop/ops.py:1237,1310). Keep the alternate ordering available until all
   * later sibling bindings have matched, rather than accepting a locally
   * successful nested ordering greedily.
   */
  if (pat->commutative && pat->n_src == 2) {
    sources.swapped = true;
    if (pat_match_sources_continue(binds, &sources)) return true;
  }

no_match:
  binds->n = saved;
  /* Pinned tinygrad UPat.or_casted is any(self, CAST(self)): direct-first and
   * exactly one CAST wrapper, not recursively CAST-tolerant. */
  if (allow_cast && pat->or_casted && uop->op == POLY_OP_CAST && uop->n_src == 1)
    return pat_match_continue(
        pat, uop->src[0], binds, continuation, opaque, false
    );
  return false;
}

static bool pat_match_accept(PolyBindings *binds, void *opaque) {
  (void)binds;
  (void)opaque;
  return true;
}

bool poly_pat_match(const PolyPat *pat, PolyUOp *uop, PolyBindings *binds) {
  return pat_match_continue(pat, uop, binds, pat_match_accept, NULL, true);
}

/* PatternMatcher */

struct PolyPatternMatcher {
  PolyRule *rules;
  PolyRuleStats *stats;
  int n_rules;
  bool stats_enabled;
  FILE *trace_fp;
  bool trace_owned;
  struct {
    int *indices;
    int n;
    int cap;
  } by_op[POLY_OP_COUNT];
};

static bool pm_stats_env_enabled(void) {
  int tracked = poly_getenv_int(
      "POLY_TRACK_MATCH_STATS",
      poly_getenv_int("TRACK_MATCH_STATS", 0)
  );
  return tracked > 0 || poly_getenv_flag("POLY_PRINT_MATCH_STATS") ||
         poly_getenv_flag("PRINT_MATCH_STATS");
}

static bool pm_trace_env_false(const char *v) {
  return !v || !v[0] || strcmp(v, "0") == 0 || strcmp(v, "false") == 0 ||
         strcmp(v, "False") == 0 || strcmp(v, "no") == 0 || strcmp(v, "NO") == 0;
}

static FILE *pm_trace_open(bool *owned) {
  if (owned) *owned = false;
  const char *v = getenv("POLY_REWRITE_TRACE_JSON");
  if (!v || !v[0]) v = getenv("REWRITE_TRACE_JSON");
  if (pm_trace_env_false(v)) return NULL;
  if (strcmp(v, "1") == 0 || strcmp(v, "true") == 0 || strcmp(v, "True") == 0 ||
      strcmp(v, "stderr") == 0 || strcmp(v, "-") == 0) {
    return stderr;
  }
  FILE *fp = fopen(v, "a");
  if (fp && owned) *owned = true;
  return fp;
}

static void pm_trace_json_string(FILE *fp, const char *s) {
  fputc('"', fp);
  if (s) {
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
      switch (*p) {
        case '\\': fputs("\\\\", fp); break;
        case '"': fputs("\\\"", fp); break;
        case '\b': fputs("\\b", fp); break;
        case '\f': fputs("\\f", fp); break;
        case '\n': fputs("\\n", fp); break;
        case '\r': fputs("\\r", fp); break;
        case '\t': fputs("\\t", fp); break;
        default:
          if (*p < 0x20) fprintf(fp, "\\u%04x", (unsigned)*p);
          else fputc((int)*p, fp);
          break;
      }
    }
  }
  fputc('"', fp);
}

static void pm_trace_emit_rewrite(
    PolyPatternMatcher *pm,
    const PolyRuleStats *st,
    PolyUOp *before,
    PolyUOp *after,
    double elapsed_ms
) {
  if (!pm || !pm->trace_fp || !before || !after) return;
  FILE *fp = pm->trace_fp;
  fputs("{\"event\":\"rewrite\",\"rule\":", fp);
  pm_trace_json_string(fp, st ? st->name : "<unnamed>");
  fputs(",\"before\":\"", fp);
  fprintf(fp, "%p", (void *)before);
  fputs("\",\"before_op\":", fp);
  pm_trace_json_string(fp, poly_op_name(before->op));
  fputs(",\"after\":\"", fp);
  fprintf(fp, "%p", (void *)after);
  fputs("\",\"after_op\":", fp);
  pm_trace_json_string(fp, poly_op_name(after->op));
  fprintf(fp, ",\"elapsed_ms\":%.6f}\n", elapsed_ms);
  fflush(fp);
}

static void pm_add_op(PolyPatternMatcher *pm, int op, int rule_idx) {
  if (op < 0 || op >= POLY_OP_COUNT) return;
  if (pm->by_op[op].n >= pm->by_op[op].cap) {
    pm->by_op[op].cap = pm->by_op[op].cap ? pm->by_op[op].cap * 2 : 8;
    pm->by_op[op].indices = realloc(pm->by_op[op].indices, pm->by_op[op].cap * sizeof(int));
  }
  pm->by_op[op].indices[pm->by_op[op].n++] = rule_idx;
}

static PolyPatternMatcher *poly_pm_new_impl(
    const PolyRule *rules,
    const char *const *names,
    int n_rules
) {
  if (n_rules < 0 || (n_rules > 0 && !rules)) return NULL;
  PolyPatternMatcher *pm = calloc(1, sizeof(PolyPatternMatcher));
  if (!pm) return NULL;
  if (n_rules > 0) {
    pm->rules = malloc((size_t)n_rules * sizeof(PolyRule));
    pm->stats = calloc((size_t)n_rules, sizeof(PolyRuleStats));
    if (!pm->rules || !pm->stats) {
      free(pm->rules);
      free(pm->stats);
      free(pm);
      return NULL;
    }
    memcpy(pm->rules, rules, (size_t)n_rules * sizeof(PolyRule));
  }
  pm->n_rules = n_rules;
  pm->stats_enabled = pm_stats_env_enabled();
  pm->trace_fp = pm_trace_open(&pm->trace_owned);
  for (int i = 0; i < n_rules; i++)
    pm->stats[i].name = (names && names[i]) ? names[i] : "<unnamed>";

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

PolyPatternMatcher *poly_pm_new(const PolyRule *rules, int n_rules) {
  return poly_pm_new_impl(rules, NULL, n_rules);
}

PolyPatternMatcher *poly_pm_new_named(const PolyNamedRule *rules, int n_rules) {
  if (n_rules < 0 || (n_rules > 0 && !rules)) return NULL;
  if (n_rules == 0) return poly_pm_new_impl(NULL, NULL, 0);

  PolyRule *plain = malloc((size_t)n_rules * sizeof(PolyRule));
  const char **names = malloc((size_t)n_rules * sizeof(const char *));
  if (!plain || !names) {
    free(plain);
    free(names);
    return NULL;
  }
  for (int i = 0; i < n_rules; i++) {
    plain[i] = (PolyRule){.pat = rules[i].pat, .fn = rules[i].fn};
    names[i] = rules[i].name;
  }
  PolyPatternMatcher *pm = poly_pm_new_impl(plain, names, n_rules);
  free(plain);
  free(names);
  return pm;
}

static void pat_thread_cache_register_pattern(PolyPat *pat) {
  if (!pat) return;
  (void)pat_thread_register(pat, false);
  for (int i = 0; pat->src && i < pat->n_src; i++)
    pat_thread_cache_register_pattern(pat->src[i]);
}

PolyPatternMatcher *poly_pm_thread_cache(PolyPatternMatcher *pm) {
  if (!pm) return NULL;
  for (int i = 0; i < pm->n_rules; i++)
    pat_thread_cache_register_pattern(pm->rules[i].pat);
  (void)pat_thread_register(pm, true);
  return pm;
}

static void pm_destroy_one(PolyPatternMatcher *pm) {
  if (!pm) return;
  for (int i = 0; i < POLY_OP_COUNT; i++)
    free(pm->by_op[i].indices);
  if (pm->trace_owned && pm->trace_fp) fclose(pm->trace_fp);
  free(pm->stats);
  free(pm->rules);
  free(pm);
}

void poly_pm_destroy(PolyPatternMatcher *pm) {
  if (!pm) return;
  pat_thread_unregister(pm, true);
  pm_destroy_one(pm);
}

typedef struct {
  PolyRule *rule;
  PolyCtx *ctx;
  PolyUOp *uop;
  PolyUOp *result;
  bool matched;
} PatRewriteContinuation;

static bool pat_rewrite_continue(PolyBindings *binds, void *opaque) {
  PatRewriteContinuation *state = opaque;
  state->matched = true;
  state->result = state->rule->fn(state->ctx, state->uop, binds);
  /* Pinned tinygrad/uop/ops.py:1345-1351 retries the next binding map only
   * when the callback returns None. */
  return state->result != NULL;
}

PolyUOp *poly_pm_rewrite(PolyPatternMatcher *pm, PolyCtx *ctx, PolyUOp *uop) {
  if (!pm || !uop) return NULL;
  int op = (int)uop->op;
  if (op < 0 || op >= POLY_OP_COUNT || pm->by_op[op].n == 0) return NULL;

  /* Compute src ops bitmask for early reject */
  PolyOpSet src_ops = {{0, 0}};
  for (int i = 0; i < uop->n_src; i++)
    src_ops = poly_opset_add(src_ops, uop->src[i]->op);

  for (int i = 0; i < pm->by_op[op].n; i++) {
    int idx = pm->by_op[op].indices[i];
    PolyRule *rule = &pm->rules[idx];
    PolyRuleStats *st = pm->stats_enabled ? &pm->stats[idx] : NULL;
    PolyRuleStats *trace_st = pm->trace_fp ? &pm->stats[idx] : st;
    double t0 = (st || pm->trace_fp) ? poly_now_ms() : 0.0;
    if (st) st->candidates++;

    /* Early reject: required ops must appear in sources */
    if (!poly_opset_subset(rule->pat->early_reject, src_ops)) {
      if (st) st->total_ms += poly_now_ms() - t0;
      continue;
    }
    if (st) st->attempts++;

    PolyBindings binds = {.n = 0};
    PatRewriteContinuation rewrite = {
        .rule = rule,
        .ctx = ctx,
        .uop = uop,
    };
    bool accepted = pat_match_continue(
        rule->pat, uop, &binds, pat_rewrite_continue, &rewrite, true
    );
    poly_bindings_free(&binds);
    if (rewrite.matched && st) st->pattern_matches++;
    if (accepted && rewrite.result != NULL) {
      double dt = (st || pm->trace_fp) ? poly_now_ms() - t0 : 0.0;
      if (st) {
        if (rewrite.result != uop) {
          st->rewrites++;
          st->rewrite_ms += dt;
        }
        st->total_ms += dt;
      }
      if (rewrite.result != uop)
        pm_trace_emit_rewrite(pm, trace_st, uop, rewrite.result, dt);
      return rewrite.result;
    }
    if (st) st->total_ms += poly_now_ms() - t0;
  }
  return NULL;
}

int poly_pm_rule_count(const PolyPatternMatcher *pm) {
  return pm ? pm->n_rules : 0;
}

int poly_pm_get_rule_stats(const PolyPatternMatcher *pm, int idx, PolyRuleStats *out) {
  if (!pm || !out || idx < 0 || idx >= pm->n_rules) return -1;
  *out = pm->stats[idx];
  return 0;
}

void poly_pm_reset_rule_stats(PolyPatternMatcher *pm) {
  if (!pm || !pm->stats) return;
  for (int i = 0; i < pm->n_rules; i++) {
    const char *name = pm->stats[i].name;
    pm->stats[i] = (PolyRuleStats){.name = name};
  }
}

PolyPatternMatcher *poly_pm_concat(PolyPatternMatcher *a, PolyPatternMatcher *b) {
  if (!a || !b) return NULL;
  int total = a->n_rules + b->n_rules;
  PolyRule *combined = total > 0 ? malloc((size_t)total * sizeof(PolyRule)) : NULL;
  const char **names = total > 0 ? malloc((size_t)total * sizeof(const char *)) : NULL;
  if (total > 0 && (!combined || !names)) {
    free(combined);
    free(names);
    return NULL;
  }
  memcpy(combined, a->rules, (size_t)a->n_rules * sizeof(PolyRule));
  memcpy(combined + a->n_rules, b->rules, (size_t)b->n_rules * sizeof(PolyRule));
  for (int i = 0; i < a->n_rules; i++)
    names[i] = (a->stats && a->stats[i].name) ? a->stats[i].name : NULL;
  for (int i = 0; i < b->n_rules; i++)
    names[a->n_rules + i] = (b->stats && b->stats[i].name) ? b->stats[i].name : NULL;
  PolyPatternMatcher *result = poly_pm_new_impl(combined, names, total);
  free(combined);
  free(names);
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
        PolyUOp *seen_inline[16];
        int n_seen_inline = 0;
        PolyMap *seen = NULL;
        while (cur) {
          bool already_seen = false;
          for (int si = 0; si < n_seen_inline; si++) {
            if (seen_inline[si] == cur) {
              already_seen = true;
              break;
            }
          }
          if (!already_seen && seen)
            already_seen = poly_map_get(seen, poly_ptr_hash(cur), cur, poly_ptr_eq) != NULL;
          if (already_seen) {
            fprintf(stderr, "polygrad: graph_rewrite fixed-point cycle\n");
            if (seen) poly_map_destroy(seen);
            failed = true;
            goto cleanup;
          }
          if (n_seen_inline < (int)(sizeof(seen_inline) / sizeof(seen_inline[0]))) {
            seen_inline[n_seen_inline++] = cur;
          } else {
            if (!seen) {
              seen = poly_map_new(32);
              if (!seen) {
                fprintf(stderr, "polygrad: graph_rewrite allocation failure\n");
                failed = true;
                goto cleanup;
              }
              for (int si = 0; si < n_seen_inline; si++)
                poly_map_set(
                    seen, poly_ptr_hash(seen_inline[si]), seen_inline[si],
                    (void *)(uintptr_t)1, poly_ptr_eq
                );
            }
            poly_map_set(seen, poly_ptr_hash(cur), cur, (void *)(uintptr_t)1, poly_ptr_eq);
          }
          PolyUOp *next = poly_pm_rewrite(pm, ctx, cur);
          if (!next || next == cur) break;
          cur = next;
        }
        if (seen) poly_map_destroy(seen);
        new_n = cur;
      }

      /* CALL gating: tinygrad's graph_rewrite treats CALL/FUNCTION bodies as
       * opaque when enter_calls=false by identity-mapping only src[0]. CALL
       * arguments remain normal graph inputs and must still be traversed. */
      bool opaque_body = new_n->op == POLY_OP_CALL || new_n->op == POLY_OP_FUNCTION;
      if (!enter_calls && opaque_body && new_n->n_src > 0) {
        replace_set(replace, new_n->src[0], new_n->src[0]);
      }

      /* Stage 1 rebuilds from rewritten sources and applies top-down rewrite. */
      if (!ws_push(&ws, n, 1, new_n)) {
        fprintf(stderr, "polygrad: graph_rewrite allocation failure\n");
        failed = true;
        goto cleanup;
      }
      int src_start = (!enter_calls && opaque_body && new_n->n_src > 0) ? 1 : 0;
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
        new_src_n = (new_n->tag != 0 || new_n->tag_arg.kind != POLY_ARG_NONE)
                        ? poly_uop_tagged_arg(
                              ctx,
                              new_n->op,
                              new_n->dtype,
                              new_src,
                              new_n->n_src,
                              new_n->arg,
                              new_n->tag,
                              new_n->tag_arg
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

      /* CALL gating: identity-map only the opaque callee root, matching
       * tinygrad RewriteContext.walk_rewrite. */
      bool opaque_body = n->op == POLY_OP_CALL || n->op == POLY_OP_FUNCTION;
      if (!enter_calls && opaque_body && n->n_src > 0) {
        replace_set(replace, n->src[0], n->src[0]);
      }

      int start = (!enter_calls && opaque_body && n->n_src > 0) ? 1 : 0;
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
        new_n = (n->tag != 0 || n->tag_arg.kind != POLY_ARG_NONE)
                    ? poly_uop_tagged_arg(
                          ctx, n->op, n->dtype, new_src, n->n_src, n->arg, n->tag, n->tag_arg
                      )
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
