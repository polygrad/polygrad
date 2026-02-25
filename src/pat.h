/*
 * pat.h — Pattern matcher for UOp graph rewriting
 *
 * Mirrors tinygrad's UPat + PatternMatcher + graph_rewrite.
 * Patterns are descriptor structs; callbacks are C function pointers.
 */

#ifndef POLY_PAT_H
#define POLY_PAT_H

#include "polygrad.h"
#include <string.h>

/* ── Named bindings from a match ──────────────────────────────────────── */

#define POLY_MAX_BINDINGS 16

typedef struct {
  const char *names[POLY_MAX_BINDINGS];
  PolyUOp *uops[POLY_MAX_BINDINGS];
  int n;
} PolyBindings;

static inline PolyUOp *poly_bind(const PolyBindings *b, const char *name) {
  for (int i = 0; i < b->n; i++)
    if (b->names[i] == name || strcmp(b->names[i], name) == 0)
      return b->uops[i];
  return NULL;
}

/* ── Pattern descriptor (mirrors UPat) ────────────────────────────────── */

typedef struct PolyPat PolyPat;
struct PolyPat {
  PolyOpSet ops;         /* acceptable ops (bitmask) */
  bool has_ops;         /* false = match any op */
  PolyDType *dtypes;     /* acceptable dtypes (NULL = any) */
  int n_dtypes;
  PolyArg arg;           /* arg to match */
  bool match_arg;       /* if true, check arg equality */
  const char *name;     /* binding name (NULL = don't bind) */
  PolyPat **src;         /* child patterns (NULL = don't check sources) */
  int n_src;
  bool strict_length;   /* require exact source count */
  bool commutative;     /* try both orderings for 2-src */
  bool or_casted;       /* match pattern or CAST(pattern) */
  PolyOpSet early_reject;
};

/* Pattern constructors (heap-allocated, caller frees with poly_pat_free) */
PolyPat *poly_pat_any(const char *name);
PolyPat *poly_pat_cvar(const char *name);
PolyPat *poly_pat_const_val(PolyArg val);
PolyPat *poly_pat_op(PolyOps op, PolyPat **src, int n_src, const char *name);
PolyPat *poly_pat_ops(PolyOpSet ops, PolyPat **src, int n_src, const char *name);
PolyPat *poly_pat_op1(PolyOps op, PolyPat *s0, const char *name);
PolyPat *poly_pat_op2(PolyOps op, PolyPat *s0, PolyPat *s1, const char *name);
PolyPat *poly_pat_op2c(PolyOps op, PolyPat *s0, PolyPat *s1, const char *name);
PolyPat *poly_pat_op3(PolyOps op, PolyPat *s0, PolyPat *s1, PolyPat *s2, const char *name);
PolyPat *poly_pat_ops1(PolyOpSet ops, PolyPat *s0, const char *name);
PolyPat *poly_pat_ops2(PolyOpSet ops, PolyPat *s0, PolyPat *s1, const char *name);
PolyPat *poly_pat_ops3(PolyOpSet ops, PolyPat *s0, PolyPat *s1, PolyPat *s2, const char *name);
PolyPat *poly_pat_dtype(const char *name, PolyDType *dtypes, int n);
PolyPat *poly_pat_allow_any_len(PolyPat *p);
PolyPat *poly_pat_or_casted(PolyPat *p);
PolyPat *poly_pat_set_early_reject(PolyPat *p, PolyOpSet early_reject);
void poly_pat_free(PolyPat *p);

/* ── Match ────────────────────────────────────────────────────────────── */

bool poly_pat_match(const PolyPat *pat, PolyUOp *uop, PolyBindings *binds);

/* ── Rewrite callback ─────────────────────────────────────────────────── */

typedef PolyUOp *(*PolyRewriteFn)(PolyCtx *ctx, PolyUOp *matched,
                                const PolyBindings *b);

typedef struct {
  PolyPat *pat;
  PolyRewriteFn fn;
} PolyRule;

/* ── PatternMatcher ───────────────────────────────────────────────────── */

typedef struct PolyPatternMatcher PolyPatternMatcher;

PolyPatternMatcher *poly_pm_new(const PolyRule *rules, int n_rules);
void poly_pm_destroy(PolyPatternMatcher *pm);
PolyUOp *poly_pm_rewrite(PolyPatternMatcher *pm, PolyCtx *ctx, PolyUOp *uop);
PolyPatternMatcher *poly_pm_concat(PolyPatternMatcher *a, PolyPatternMatcher *b);

/* ── graph_rewrite (top-down worklist engine) ─────────────────────────── */

/* Rewrites with optional pass-local user context (mirrors tinygrad's ctx=...). */
PolyUOp *poly_graph_rewrite_ctx(PolyCtx *ctx, PolyUOp *sink, PolyPatternMatcher *pm,
                                void *user_ctx);
PolyUOp *poly_graph_rewrite_ctx_ex(PolyCtx *ctx, PolyUOp *sink, PolyPatternMatcher *pm,
                                   void *user_ctx, bool bottom_up);
/* Legacy convenience wrapper (no user context). */
PolyUOp *poly_graph_rewrite(PolyCtx *ctx, PolyUOp *sink, PolyPatternMatcher *pm);
PolyUOp *poly_graph_rewrite_ex(PolyCtx *ctx, PolyUOp *sink, PolyPatternMatcher *pm,
                               bool bottom_up);
/* Returns current graph_rewrite user_ctx for callbacks in the active pass. */
void *poly_graph_rewrite_userctx(void);

/* ── UOp helpers used by rewrite callbacks ────────────────────────────── */

PolyUOp *poly_const_like(PolyCtx *ctx, PolyUOp *ref, PolyArg val);
PolyUOp *poly_const_like_int(PolyCtx *ctx, PolyUOp *ref, int64_t val);
PolyUOp *poly_const_like_float(PolyCtx *ctx, PolyUOp *ref, double val);
PolyUOp *poly_const_like_bool(PolyCtx *ctx, PolyUOp *ref, bool val);

/* ── ALU constant-fold executor ───────────────────────────────────────── */

PolyArg poly_exec_alu(PolyOps op, PolyDType dtype, PolyArg *operands, int n_ops);

/* ── Symbolic simplification rules ────────────────────────────────────── */

PolyPatternMatcher *poly_symbolic_simple(void);

/* ── Codegen pipeline (port of full_rewrite_to_sink) ─────────────────── */

PolyUOp *poly_full_rewrite_to_sink(PolyCtx *ctx, PolyUOp *sink);

#endif /* POLY_PAT_H */
