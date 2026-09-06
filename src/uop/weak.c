/*
 * uop/weak.c — weak dtype commit and lowering
 *
 * Mirrors tinygrad/uop/weak.py. Both weakint and weakfloat use this shared
 * commit/lowering machinery, including pm_lower_index_dtype.
 */

#include "uop/weak.h"
#include "utils.h"
#include <stdlib.h>
#include <string.h>

static PolyUOp *replace_uop(PolyCtx *ctx, PolyUOp *u, PolyDType dtype, PolyUOp **src, int n_src) {
  PolyArg arg =
      (u->op == POLY_OP_CAST || u->op == POLY_OP_BITCAST) ? poly_arg_dtype(dtype) : u->arg;
  return (u->tag != 0 || u->tag_arg.kind != POLY_ARG_NONE)
             ? poly_uop_tagged_arg(ctx, u->op, dtype, src, n_src, arg, u->tag, u->tag_arg)
             : poly_uop(ctx, u->op, dtype, src, n_src, arg);
}

static bool dtype_is_weakfloat(PolyDType dtype) {
  return poly_dtype_eq(dtype, POLY_WEAKFLOAT);
}

static bool has_weak_src(PolyUOp *u) {
  if (!u) return false;
  for (int i = 0; i < u->n_src; i++)
    if (poly_dtype_is_weak(u->src[i]->dtype)) return true;
  return false;
}

/* tinygrad/uop/weak.py:default_dtype. */
static PolyDType default_dtype(PolyCtx *ctx, PolyUOp *u) {
  PolyDType dtype;
  if (dtype_is_weakfloat(u->dtype)) {
    dtype = POLY_FLOAT32;
  } else {
    int64_t vmin = 0, vmax = 0;
    poly_uop_minmax(ctx, u, &vmin, &vmax);
    dtype = vmin >= INT32_MIN && vmax <= INT32_MAX ? POLY_INT32 : POLY_INT64;
  }
  return dtype;
}

/* tinygrad/uop/weak.py:commit_weak. */
PolyUOp *poly_commit_weak(PolyCtx *ctx, PolyUOp *u, PolyDType dtype) {
  if (poly_dtype_eq(u->dtype, dtype)) return u;
  if (u->op == POLY_OP_CONST) return poly_uop_const(ctx, u->arg, dtype);
  return poly_uop1(ctx, POLY_OP_CAST, dtype, u, poly_arg_none());
}

static bool least_upper_src_dtype(
    PolyUOp **src,
    int n_src,
    int start,
    PolyDType initial,
    PolyDType *out
) {
  PolyDType dtype = initial;
  for (int i = start; i < n_src; i++) {
    PolyDType next = POLY_VOID;
    if (!poly_dtype_least_upper(dtype, src[i]->dtype, &next)) return false;
    dtype = next;
  }
  *out = dtype;
  return true;
}

/* Current Polygrad equivalent of tinygrad/uop/ops.py:dtype_from_uop for the
 * broadcastable subset reachable from weak lowering. */
static PolyDType dtype_from_uop(PolyUOp *u, PolyUOp **src, int n_src) {
  PolyDType dtype = u->dtype;
  return poly_dtype_from_uop(u->op, src, n_src, u->arg, u->dtype, &dtype) ? dtype : u->dtype;
}

/* tinygrad/uop/weak.py:commit_srcs_at. */
static PolyUOp *commit_srcs_at(PolyCtx *ctx, PolyUOp *u, PolyDType dtype) {
  PolyUOp *src_inline[16];
  PolyUOp **src = u->n_src > (int)(sizeof(src_inline) / sizeof(src_inline[0]))
                      ? malloc((size_t)u->n_src * sizeof(*src))
                      : src_inline;
  if (!src) return NULL;
  bool changed = false;
  for (int i = 0; i < u->n_src; i++) {
    src[i] =
        poly_dtype_is_weak(u->src[i]->dtype) ? poly_commit_weak(ctx, u->src[i], dtype) : u->src[i];
    changed |= src[i] != u->src[i];
  }
  PolyUOp *result =
      changed ? replace_uop(ctx, u, dtype_from_uop(u, src, u->n_src), src, u->n_src) : NULL;
  if (src != src_inline) free(src);
  return result;
}

static PolyUOp *weak_cast_src(PolyUOp *u) {
  return u && u->op == POLY_OP_CAST && u->n_src == 1 && poly_dtype_is_weak(u->dtype) ? u->src[0]
                                                                                     : NULL;
}

/* Tinygrad 2026-08-22/a9069c177a9d uop/weak.py:52. */
static PolyUOp *lower_weak_const(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b) {
  (void)b;
  /* UOp.const creates a fresh literal; the weak occurrence's tag is not copied. */
  PolyUOp *concrete = poly_uop_const(ctx, u->arg, default_dtype(ctx, u));
  return poly_uop1(ctx, POLY_OP_CAST, u->dtype, concrete, poly_arg_none());
}

/* Tinygrad 2026-08-22/a9069c177a9d uop/weak.py:53-56. */
static PolyUOp *lower_stacked_weak_casts(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b) {
  (void)b;
  PolyUOp *x = u->src[0]->src[0];
  if (poly_dtype_is_weak(x->dtype)) return NULL;
  PolyDType inner_default = default_dtype(ctx, u->src[0]);
  PolyDType outer_default = default_dtype(ctx, u);
  PolyUOp *first = poly_uop1(ctx, POLY_OP_CAST, inner_default, x, poly_arg_none());
  PolyUOp *second = poly_uop1(ctx, POLY_OP_CAST, outer_default, first, poly_arg_none());
  return poly_uop1(ctx, POLY_OP_CAST, u->dtype, second, poly_arg_none());
}

/* Tinygrad 2026-08-22/a9069c177a9d uop/weak.py:57-58. */
static PolyUOp *lower_weak_alu_resource(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b) {
  (void)b;
  if (u->arg.kind != POLY_ARG_PARAM || !u->arg.param || u->arg.param->addrspace != POLY_ADDR_ALU)
    return NULL;
  PolyDType dtype = default_dtype(ctx, u);
  PolyParamArg arg = *u->arg.param;
  arg.dtype = dtype;
  /* UOp.replace(arg=...) preserves the resource's identity tag. */
  PolyUOp *concrete = poly_uop_tagged_arg(
      ctx, u->op, dtype, u->src, u->n_src, poly_arg_param(&arg), u->tag, u->tag_arg
  );
  return concrete ? poly_uop1(ctx, POLY_OP_CAST, u->dtype, concrete, poly_arg_none()) : NULL;
}

/* Tinygrad 2026-08-22/a9069c177a9d uop/weak.py:42-49 lower_weak_node. */
static PolyUOp *lower_weak_node(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b) {
  (void)b;
  PolyUOp *src_inline[16];
  PolyUOp **src = u->n_src > (int)(sizeof(src_inline) / sizeof(src_inline[0]))
                      ? malloc((size_t)u->n_src * sizeof(*src))
                      : src_inline;
  if (!src) return NULL;
  bool changed = false;
  int start = u->op == POLY_OP_WHERE ? 1 : 0;
  for (int i = 0; i < u->n_src; i++) {
    PolyUOp *inner = weak_cast_src(u->src[i]);
    src[i] = inner ? inner : u->src[i];
    changed |= src[i] != u->src[i];
  }
  for (int i = start; i < u->n_src; i++) {
    if (poly_dtype_is_weak(src[i]->dtype)) {
      if (src != src_inline) free(src);
      return NULL;
    }
  }
  if (!changed) {
    if (src != src_inline) free(src);
    return NULL;
  }

  PolyDType dtype = dtype_from_uop(u, src, u->n_src);
  if (poly_opset_has(POLY_GROUP_BINARY, u->op)) {
    dtype = default_dtype(ctx, u);
    if (!least_upper_src_dtype(src, u->n_src, 0, dtype, &dtype)) {
      if (src != src_inline) free(src);
      return NULL;
    }
  }
  dtype = poly_dtype_strong(dtype);
  for (int i = start; i < u->n_src; i++) {
    PolyUOp *base = poly_uop_base(src[i]);
    if (base->op == POLY_OP_CONST && base->arg.kind == POLY_ARG_INVALID) continue;
    src[i] = poly_commit_weak(ctx, src[i], dtype);
  }
  PolyUOp *concrete = replace_uop(ctx, u, dtype_from_uop(u, src, u->n_src), src, u->n_src);
  PolyUOp *result = poly_dtype_eq(concrete->dtype, u->dtype)
                        ? concrete
                        : poly_uop1(ctx, POLY_OP_CAST, u->dtype, concrete, poly_arg_none());
  if (src != src_inline) free(src);
  return result;
}

static _Thread_local PolyPatternMatcher *g_pm_lower_weak = NULL;
PolyPatternMatcher *poly_pm_lower_weak(void) {
  if (g_pm_lower_weak) return g_pm_lower_weak;
  PolyDType weaks[] = {POLY_WEAKINT, POLY_WEAKFLOAT};
  PolyDType weakint[] = {POLY_WEAKINT};
  PolyOpSet resources =
      poly_opset_add(poly_opset_add((PolyOpSet){{0, 0}}, POLY_OP_PARAM), POLY_OP_BUFFER);
  PolyOpSet lower_weak_ops = poly_opset_union(POLY_GROUP_BINARY, POLY_GROUP_UNARY);
  lower_weak_ops = poly_opset_add(lower_weak_ops, POLY_OP_WHERE);
  lower_weak_ops = poly_opset_add(lower_weak_ops, POLY_OP_RANGE);
  lower_weak_ops = poly_opset_add(lower_weak_ops, POLY_OP_STACK);
  lower_weak_ops = poly_opset_add(lower_weak_ops, POLY_OP_SPECIAL);
  PolyNamedRule rules[] = {
      POLY_RULE(
          poly_upat_set_dtype(poly_upat_op(POLY_OP_CONST, NULL, 0, "u"), weaks, 2), lower_weak_const
      ),
      POLY_RULE(
          poly_upat_set_dtype(
              poly_upat_op1(
                  POLY_OP_CAST,
                  poly_upat_set_dtype(
                      poly_upat_op1(POLY_OP_CAST, poly_upat_any("x"), NULL), weaks, 2
                  ),
                  "u"
              ),
              weaks, 2
          ),
          lower_stacked_weak_casts
      ),
      POLY_RULE(
          poly_upat_set_dtype(poly_upat_ops(resources, NULL, 0, "u"), weakint, 1),
          lower_weak_alu_resource
      ),
      POLY_RULE(poly_upat_ops(lower_weak_ops, NULL, 0, "u"), lower_weak_node),
  };
  g_pm_lower_weak =
      poly_pm_thread_cache(poly_pm_new_named(rules, (int)(sizeof(rules) / sizeof(rules[0]))));
  return g_pm_lower_weak;
}

static PolyUOp *lower_weak_source(PolyCtx *ctx, PolyUOp *u) {
  /* Tinygrad 2026-08-22/a9069c177a9d uop/weak.py:60-68 memoizes each
   * consumer-edge lowering in the final index rewrite's ctx dict. */
  PolyMap *cache = (PolyMap *)poly_graph_rewrite_userctx();
  PolyUOp *cached = cache ? poly_map_get(cache, poly_ptr_hash(u), u, poly_ptr_eq) : NULL;
  if (cached) return cached;
  PolyUOp *lowered = poly_graph_rewrite(ctx, u, poly_pm_lower_weak());
  PolyUOp *inner = weak_cast_src(lowered);
  PolyUOp *ret = inner ? inner : lowered;
  if (cache) poly_map_set(cache, poly_ptr_hash(u), u, ret, poly_ptr_eq);
  return ret;
}

/* Tinygrad 2026-08-22/a9069c177a9d uop/weak.py:18-20 commit_weak_srcs. */
static PolyUOp *commit_weak_srcs(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b) {
  (void)b;
  if (!u || !has_weak_src(u)) return NULL;
  PolyDType dtype = u->src[0]->dtype;
  if (!least_upper_src_dtype(u->src, u->n_src, 1, dtype, &dtype) || poly_dtype_is_weak(dtype))
    return NULL;
  return commit_srcs_at(ctx, u, dtype);
}

/* Tinygrad 2026-08-22/a9069c177a9d uop/weak.py:35-37. */
static PolyUOp *commit_weak_store(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b) {
  (void)b;
  PolyUOp *src_inline[16];
  PolyUOp **src = u->n_src > (int)(sizeof(src_inline) / sizeof(src_inline[0]))
                      ? malloc((size_t)u->n_src * sizeof(*src))
                      : src_inline;
  if (!src) return NULL;
  memcpy(src, u->src, (size_t)u->n_src * sizeof(*src));
  src[1] = poly_commit_weak(ctx, src[1], src[0]->dtype);
  PolyUOp *ret = replace_uop(ctx, u, u->dtype, src, u->n_src);
  if (src != src_inline) free(src);
  return ret;
}

static _Thread_local PolyPatternMatcher *g_pm_commit_weak = NULL;
PolyPatternMatcher *poly_pm_commit_weak(void) {
  if (g_pm_commit_weak) return g_pm_commit_weak;
  PolyDType weaks[] = {POLY_WEAKINT, POLY_WEAKFLOAT};
  PolyNamedRule rules[] = {
      POLY_RULE(poly_upat_ops(POLY_GROUP_BROADCASTABLE, NULL, 0, "u"), commit_weak_srcs),
      POLY_RULE(
          poly_upat_allow_any_len(poly_upat_op2(
              POLY_OP_STORE, poly_upat_any(NULL), poly_upat_dtype(NULL, weaks, 2), "u"
          )),
          commit_weak_store
      ),
  };
  g_pm_commit_weak =
      poly_pm_thread_cache(poly_pm_new_named(rules, (int)(sizeof(rules) / sizeof(rules[0]))));
  return g_pm_commit_weak;
}

/* tinygrad/uop/weak.py:cast_weak_srcs. */
static PolyUOp *cast_weak_srcs(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b) {
  (void)b;
  if (!u || u->op != POLY_OP_CAST || u->n_src != 1 || poly_dtype_is_weak(u->dtype) ||
      !poly_dtype_is_weak(u->src[0]->dtype))
    return NULL;

  PolyUOp *weak = u->src[0];
  if (!poly_dtype_eq(poly_dtype_weak(u->dtype), weak->dtype)) return NULL;

  PolyDType dtype = u->dtype, def = default_dtype(ctx, weak);
  if (!poly_dtype_least_upper(dtype, def, &dtype)) return NULL;
  PolyUOp *committed = commit_srcs_at(ctx, weak, dtype);
  if (!committed) return NULL;
  return poly_dtype_eq(committed->dtype, u->dtype)
             ? committed
             : poly_uop1(ctx, POLY_OP_CAST, u->dtype, committed, poly_arg_none());
}

/* Tinygrad 2026-08-22/a9069c177a9d uop/weak.py:29. */
static PolyUOp *commit_weak_cast_const(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b) {
  (void)b;
  return poly_commit_weak(ctx, u->src[0], u->dtype);
}

static _Thread_local PolyPatternMatcher *g_pm_cast_weak = NULL;
PolyPatternMatcher *poly_pm_cast_weak(void) {
  if (g_pm_cast_weak) return g_pm_cast_weak;
  PolyDType weaks[] = {POLY_WEAKINT, POLY_WEAKFLOAT};
  PolyNamedRule rules[] = {
      POLY_RULE(
          poly_upat_op1(
              POLY_OP_CAST,
              poly_upat_set_dtype(poly_upat_ops(POLY_GROUP_ALU, NULL, 0, "u"), weaks, 2), "c"
          ),
          cast_weak_srcs
      ),
      POLY_RULE(
          poly_upat_op1(
              POLY_OP_CAST,
              poly_upat_set_dtype(poly_upat_op(POLY_OP_CONST, NULL, 0, "u"), weaks, 2), "c"
          ),
          commit_weak_cast_const
      ),
  };
  g_pm_cast_weak =
      poly_pm_thread_cache(poly_pm_new_named(rules, (int)(sizeof(rules) / sizeof(rules[0]))));
  return g_pm_cast_weak;
}

/* tinygrad/uop/weak.py:lower_weak_srcs. */
PolyUOp *poly_lower_weak_srcs(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b) {
  (void)b;
  if (!u || poly_dtype_is_weak(u->dtype) || !has_weak_src(u)) return NULL;

  if (poly_opset_has(POLY_GROUP_COMPARISON, u->op)) {
    PolyUOp *ret = lower_weak_source(ctx, u);
    return ret != u ? ret : NULL;
  }

  PolyUOp *src_inline[16];
  PolyUOp **src = u->n_src > (int)(sizeof(src_inline) / sizeof(src_inline[0]))
                      ? malloc((size_t)u->n_src * sizeof(*src))
                      : src_inline;
  if (!src) return NULL;
  bool changed = false;
  for (int i = 0; i < u->n_src; i++) {
    src[i] = poly_dtype_is_weak(u->src[i]->dtype) ? lower_weak_source(ctx, u->src[i]) : u->src[i];
    changed |= src[i] != u->src[i];
  }
  PolyUOp *ret = changed ? replace_uop(ctx, u, u->dtype, src, u->n_src) : NULL;
  if (src != src_inline) free(src);
  return ret;
}

/* Tinygrad 2026-08-22/a9069c177a9d uop/weak.py:77-81. */
static PolyUOp *narrow_gated_long_index(PolyCtx *ctx, PolyUOp *idx, const PolyBindings *b) {
  (void)b;
  PolyUOp *coord = idx->src[1];
  int64_t max_numel = poly_uop_max_numel(ctx, idx->src[0]);
  if (max_numel < 0 || max_numel - 1 > INT32_MAX) return NULL;

  PolyUOp *narrow = poly_uop1(ctx, POLY_OP_CAST, POLY_INT32, coord->src[1], poly_arg_none());
  PolyUOp *invalid = poly_uop_const(ctx, poly_arg_invalid(), POLY_INT32);
  PolyUOp *valid =
      poly_uop3(ctx, POLY_OP_WHERE, POLY_INT32, coord->src[0], narrow, invalid, poly_arg_none());
  PolyUOp **src = malloc((size_t)idx->n_src * sizeof(*src));
  if (!src) return NULL;
  memcpy(src, idx->src, (size_t)idx->n_src * sizeof(*src));
  src[1] = valid;
  PolyUOp *ret = poly_uop_replace_src(ctx, idx, src);
  free(src);
  return ret;
}

static _Thread_local PolyPatternMatcher *g_pm_lower_index_dtype = NULL;
PolyPatternMatcher *poly_pm_lower_index_dtype(void) {
  if (g_pm_lower_index_dtype) return g_pm_lower_index_dtype;
  PolyDType int64_dtype[] = {POLY_INT64};
  PolyOpSet index_shrink =
      poly_opset_add(poly_opset_add((PolyOpSet){{0, 0}}, POLY_OP_INDEX), POLY_OP_SHRINK);
  PolyNamedRule rules[] = {
      POLY_RULE(poly_upat_any("u"), poly_lower_weak_srcs),
      POLY_RULE(
          poly_upat_allow_any_len(poly_upat_ops2(
              index_shrink, poly_upat_any("buf"),
              poly_upat_op3(
                  POLY_OP_WHERE, poly_upat_any("gate"), poly_upat_dtype("idx", int64_dtype, 1),
                  poly_upat_const_val(poly_arg_invalid()), NULL
              ),
              "u"
          )),
          narrow_gated_long_index
      ),
  };
  PolyPatternMatcher *local = poly_pm_new_named(rules, (int)(sizeof(rules) / sizeof(rules[0])));
  PolyPatternMatcher *weak = poly_pm_concat(poly_pm_commit_weak(), poly_pm_cast_weak());
  g_pm_lower_index_dtype = poly_pm_thread_cache(poly_pm_concat(weak, local));
  poly_pm_destroy(weak);
  poly_pm_destroy(local);
  return g_pm_lower_index_dtype;
}
