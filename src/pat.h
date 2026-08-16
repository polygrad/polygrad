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

/* Named bindings from a match */

#define POLY_BINDINGS_INLINE 16

typedef struct {
  const char *name;
  PolyUOp *uop;
} PolyBindingEntry;

typedef struct {
  const char *names[POLY_BINDINGS_INLINE];
  PolyUOp *uops[POLY_BINDINGS_INLINE];
  int n;
  PolyBindingEntry *extra;
  int extra_cap;
} PolyBindings;

PolyUOp *poly_bind(const PolyBindings *b, const char *name);
void poly_bindings_free(PolyBindings *b);

/* Pattern descriptor (mirrors UPat) */

typedef struct PolyPat PolyPat;
struct PolyPat {
  PolyOpSet ops; /* acceptable ops (bitmask) */
  bool has_ops; /* false = match any op */
  PolyDType *dtypes; /* acceptable dtypes (NULL = any) */
  int n_dtypes;
  PolyArg arg; /* arg to match */
  bool match_arg; /* if true, check arg equality */
  const char *name; /* binding name (NULL = don't bind) */
  PolyPat **src; /* child patterns (NULL = don't check sources) */
  int n_src;
  bool strict_length; /* require exact source count */
  bool commutative; /* try both orderings for 2-src */
  bool or_casted; /* match pattern or CAST(pattern) */
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

/* Match */

bool poly_pat_match(const PolyPat *pat, PolyUOp *uop, PolyBindings *binds);

/* Rewrite callback */

typedef PolyUOp *(*PolyRewriteFn)(PolyCtx *ctx, PolyUOp *matched, const PolyBindings *b);

typedef struct {
  PolyPat *pat;
  PolyRewriteFn fn;
} PolyRule;

typedef struct {
  PolyPat *pat;
  PolyRewriteFn fn;
  /* Borrowed diagnostic name. It must outlive the matcher. */
  const char *name;
} PolyNamedRule;

#define POLY_RULE(pat_, fn_) ((PolyNamedRule){.pat = (pat_), .fn = (fn_), .name = #fn_})

typedef struct {
  /* Borrowed diagnostic name. */
  const char *name;
  uint64_t candidates;       /* by-op rule candidates considered */
  uint64_t attempts;         /* candidates that pass early-reject */
  uint64_t pattern_matches;  /* pattern matched before callback */
  uint64_t rewrites;         /* callback produced a different UOp */
  double total_ms;           /* candidate handling time */
  double rewrite_ms;         /* callback-produced rewrite time */
} PolyRuleStats;

/* PatternMatcher */

typedef struct PolyPatternMatcher PolyPatternMatcher;

PolyPatternMatcher *poly_pm_new(const PolyRule *rules, int n_rules);
PolyPatternMatcher *poly_pm_new_named(const PolyNamedRule *rules, int n_rules);
/* Internal compiler-cache ownership: register a _Thread_local matcher and its
 * borrowed pattern tree for automatic cleanup when the constructing thread
 * exits. Ordinary public matchers retain caller-managed lifetime. */
PolyPatternMatcher *poly_pm_thread_cache(PolyPatternMatcher *pm);
void poly_pm_destroy(PolyPatternMatcher *pm);
PolyUOp *poly_pm_rewrite(PolyPatternMatcher *pm, PolyCtx *ctx, PolyUOp *uop);
PolyPatternMatcher *poly_pm_concat(PolyPatternMatcher *a, PolyPatternMatcher *b);
int poly_pm_rule_count(const PolyPatternMatcher *pm);
int poly_pm_get_rule_stats(const PolyPatternMatcher *pm, int idx, PolyRuleStats *out);
void poly_pm_reset_rule_stats(PolyPatternMatcher *pm);

/* graph_rewrite (top-down worklist engine) */

/* Rewrites with optional pass-local user context (mirrors tinygrad's ctx=...). */
PolyUOp *poly_graph_rewrite_ctx(
    PolyCtx *ctx,
    PolyUOp *sink,
    PolyPatternMatcher *pm,
    void *user_ctx
);
PolyUOp *poly_graph_rewrite_ctx_ex(
    PolyCtx *ctx,
    PolyUOp *sink,
    PolyPatternMatcher *pm,
    void *user_ctx,
    bool bottom_up
);
/* Full variant: enter_calls=false skips CALL/FUNCTION src[0] (callee body). */
PolyUOp *poly_graph_rewrite_ctx_ex2(
    PolyCtx *ctx,
    PolyUOp *sink,
    PolyPatternMatcher *pm,
    void *user_ctx,
    bool bottom_up,
    bool enter_calls
);
/* Legacy convenience wrapper (no user context). */
PolyUOp *poly_graph_rewrite(PolyCtx *ctx, PolyUOp *sink, PolyPatternMatcher *pm);
PolyUOp *poly_graph_rewrite_ex(PolyCtx *ctx, PolyUOp *sink, PolyPatternMatcher *pm, bool bottom_up);
/* MLIR-style walk_rewrite: single-pass, no re-traversal into rewritten subtrees.
 * pm = top-down (after rebuild), bpm = bottom-up (before descent, short-circuits).
 * Either can be NULL. */
PolyUOp *poly_graph_walk_rewrite(
    PolyCtx *ctx,
    PolyUOp *sink,
    PolyPatternMatcher *pm,
    PolyPatternMatcher *bpm,
    void *user_ctx,
    bool enter_calls
);
/* Returns current graph_rewrite user_ctx for callbacks in the active pass. */
void *poly_graph_rewrite_userctx(void);

/* UOp helpers used by rewrite callbacks */

PolyUOp *poly_const_like(PolyCtx *ctx, PolyUOp *ref, PolyArg val);
PolyUOp *poly_const_like_int(PolyCtx *ctx, PolyUOp *ref, int64_t val);
PolyUOp *poly_const_like_float(PolyCtx *ctx, PolyUOp *ref, double val);
PolyUOp *poly_const_like_bool(PolyCtx *ctx, PolyUOp *ref, bool val);

/* ALU constant-fold executor */

/* Pinned tinygrad exec_alu exposes truncate_output: symbolic constant folding
 * passes false, while execution-style callers pass true. */
PolyArg poly_exec_alu(
    PolyOps op,
    PolyDType dtype,
    PolyArg *operands,
    int n_ops,
    bool truncate_output
);
/* Pinned tinygrad symbolic.fold_bitcast: reinterpret one scalar CONST through
 * equal-width storage formats. Returns false for pointer/vector/formatless or
 * unequal-width dtypes. */
bool poly_exec_bitcast_const(
    PolyDType from,
    PolyDType to,
    PolyArg value,
    PolyArg *out
);

/* Symbolic simplification rules */

PolyPatternMatcher *poly_symbolic_simple(void);
PolyPatternMatcher *poly_symbolic(void);
/* Pinned tinygrad's broader codegen-stage `sym` matcher. */
PolyPatternMatcher *poly_sym(void);
PolyPatternMatcher *poly_pm_gep_pushing(void);
/* Pinned tinygrad/uop/symbolic.py validity-aware matcher components. */
PolyUOp *poly_uop_given_valid(
    PolyCtx *ctx,
    PolyUOp *valid,
    PolyUOp *uop,
    bool try_simplex
);
PolyPatternMatcher *poly_pm_simplify_valid(void);
PolyPatternMatcher *poly_pm_drop_and_clauses(void);

/* Codegen pipeline (port of full_rewrite_to_sink) */

PolyUOp *poly_full_rewrite_to_sink(PolyCtx *ctx, PolyUOp *sink);

#endif /* POLY_PAT_H */
