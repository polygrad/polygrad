/* Current Tinygrad codegen/gpudims.py. */

#include "codegen/codegen.h"
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Current Tinygrad codegen/gpudims.py. */
static int cmp_range_axis_id_ptr(const void *ap, const void *bp) {
  const PolyUOp *a = *(const PolyUOp *const *)ap;
  const PolyUOp *b = *(const PolyUOp *const *)bp;
  int64_t ia = poly_range_axis_id(a->arg);
  int64_t ib = poly_range_axis_id(b->arg);
  if (ia < ib) return -1;
  if (ia > ib) return 1;
  int na = poly_range_n_extra(a->arg), nb = poly_range_n_extra(b->arg);
  int n = na < nb ? na : nb;
  int64_t *ea = poly_range_extra(a->arg), *eb = poly_range_extra(b->arg);
  for (int i = 0; i < n; i++) {
    if (ea[i] < eb[i]) return -1;
    if (ea[i] > eb[i]) return 1;
  }
  if (na < nb) return -1;
  if (na > nb) return 1;
  return 0;
}

static bool range_same_axis_key(PolyUOp *a, PolyUOp *b) {
  if (!a || !b || a->op != POLY_OP_RANGE || b->op != POLY_OP_RANGE) return false;
  if (!poly_arg_is_range(a->arg) || !poly_arg_is_range(b->arg)) return false;
  if (poly_range_axis_id(a->arg) != poly_range_axis_id(b->arg)) return false;
  int na = poly_range_n_extra(a->arg), nb = poly_range_n_extra(b->arg);
  if (na != nb) return false;
  if (na == 0) return true;
  return memcmp(poly_range_extra(a->arg), poly_range_extra(b->arg), (size_t)na * sizeof(int64_t)) ==
         0;
}

static int find_range_axis_key(PolyUOp *u, PolyUOp **arr, int n) {
  for (int i = 0; i < n; i++)
    if (range_same_axis_key(u, arr[i])) return i;
  return -1;
}

static bool add_gpudim_sub(
    PolyUOp ***oldp,
    PolyUOp ***newp,
    int *n,
    int *cap,
    PolyUOp *old_u,
    PolyUOp *new_u
) {
  if (!oldp || !newp || !n || !cap || !old_u || !new_u) return false;
  if (*n >= *cap) {
    int new_cap = (*cap > 0) ? (*cap * 2) : 64;
    PolyUOp **new_old = malloc((size_t)new_cap * sizeof(PolyUOp *));
    PolyUOp **new_new = malloc((size_t)new_cap * sizeof(PolyUOp *));
    if (!new_old || !new_new) {
      free(new_old);
      free(new_new);
      return false;
    }
    if (*n > 0) {
      memcpy(new_old, *oldp, (size_t)*n * sizeof(PolyUOp *));
      memcpy(new_new, *newp, (size_t)*n * sizeof(PolyUOp *));
    }
    free(*oldp);
    free(*newp);
    *oldp = new_old;
    *newp = new_new;
    *cap = new_cap;
  }
  (*oldp)[*n] = old_u;
  (*newp)[*n] = new_u;
  (*n)++;
  return true;
}

static PolyUOp *gpudim_special_bound(PolyCtx *ctx, PolyUOp *bound) {
  if (!bound) return NULL;
  if (poly_dtype_is_index(bound->dtype)) return bound;
  if (bound->op == POLY_OP_CONST && bound->arg.kind == POLY_ARG_INT)
    return poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, bound->arg);
  return poly_uop1(ctx, POLY_OP_CAST, POLY_WEAKINT, bound, poly_arg_none());
}

typedef struct {
  PolyUOp *expr;
  int64_t max;
} GpuDimExpr;

static int64_t _dim_max(PolyCtx *ctx, PolyUOp *expr) {
  if (!expr) return 1;
  int64_t lo = 0, hi = 1;
  (void)lo;
  poly_uop_minmax(ctx, expr, &lo, &hi);
  return hi > 0 ? hi : 1;
}

static bool gpudim_caps_present(const int caps[3]) {
  return caps && caps[0] > 0 && caps[1] > 0 && caps[2] > 0;
}

static PolyUOp *gpudim_const(PolyCtx *ctx, int64_t v) {
  return poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(v));
}

static PolyUOp *gpudim_mul_const(PolyCtx *ctx, PolyUOp *x, int64_t v) {
  if (v == 1) return x;
  return poly_uop2(ctx, POLY_OP_MUL, POLY_WEAKINT, x, gpudim_const(ctx, v), poly_arg_none());
}

static PolyUOp *gpudim_floordiv_const(PolyCtx *ctx, PolyUOp *x, int64_t v) {
  if (v == 1) return x;
  return poly_uop2(ctx, POLY_OP_FLOORDIV, POLY_WEAKINT, x, gpudim_const(ctx, v), poly_arg_none());
}

static PolyUOp *gpudim_add(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  return poly_uop2(ctx, POLY_OP_ADD, POLY_WEAKINT, a, b, poly_arg_none());
}

static PolyUOp *gpudim_mul(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  return poly_uop2(ctx, POLY_OP_MUL, POLY_WEAKINT, a, b, poly_arg_none());
}

static bool gpudim_const_i64(PolyUOp *u, int64_t *out) {
  if (!u || u->op != POLY_OP_CONST || u->arg.kind != POLY_ARG_INT) return false;
  if (out) *out = u->arg.i;
  return true;
}

/* Tinygrad 2026-08-22/a9069c177a9d codegen/gpudims.py:11,16 preserves
 * Python-int products as constants while symbolic `sint` products remain ALU. */
static PolyUOp *gpudim_sint_mul(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  int64_t av = 0, bv = 0, product = 0;
  if (gpudim_const_i64(a, &av) && gpudim_const_i64(b, &bv)) {
    if (__builtin_mul_overflow(av, bv, &product)) return NULL;
    return gpudim_const(ctx, product);
  }
  if (gpudim_const_i64(a, &av) && av == 1) return b;
  if (gpudim_const_i64(b, &bv) && bv == 1) return a;
  return gpudim_mul(ctx, a, b);
}

static PolyUOp *gpudim_product(PolyCtx *ctx, const GpuDimExpr *dims, int begin, int end) {
  PolyUOp *product = gpudim_const(ctx, 1);
  for (int i = begin; i < end; i++) {
    product = gpudim_sint_mul(ctx, product, dims[i].expr);
    if (!product) return NULL;
  }
  return product;
}

static PolyUOp *gpudim_floormod(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  return poly_uop2(ctx, POLY_OP_FLOORMOD, POLY_WEAKINT, a, b, poly_arg_none());
}

static PolyUOp *gpudim_floordiv(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  return poly_uop2(ctx, POLY_OP_FLOORDIV, POLY_WEAKINT, a, b, poly_arg_none());
}

static bool gpudim_safe_mul_i64(int64_t a, int64_t b, int64_t *out) {
  if (__builtin_mul_overflow(a, b, out)) return false;
  return *out > 0;
}

static int64_t gpudim_smallest_divisor(int64_t x) {
  if (x <= 1) return 1;
  for (int64_t d = 2; d <= x / d; d++)
    if ((x % d) == 0) return d;
  return 1;
}

static bool _group_dims(
    PolyCtx *ctx,
    const GpuDimExpr *dims,
    int n_dims,
    const int caps[3],
    GpuDimExpr *out,
    int *n_out
) {
  if (!ctx || !dims || !out || !n_out || !gpudim_caps_present(caps) || n_dims <= 0 ||
      n_dims > POLY_MAX_DIMS)
    return false;

  int n = n_dims;
  for (int i = 0; i < n; i++)
    out[i] = dims[i];

  while (n > 3 || out[0].max > caps[0] || (n > 1 && out[1].max > caps[1]) ||
         (n > 2 && out[2].max > caps[2])) {
    bool grouped = false;
    for (int i = 0; i < 3 && i < n - 1; i++) {
      int64_t prod = 0;
      if (!gpudim_safe_mul_i64(out[i].max, out[i + 1].max, &prod) || prod > caps[i]) continue;
      out[i].expr = gpudim_sint_mul(ctx, out[i].expr, out[i + 1].expr);
      if (!out[i].expr) return false;
      out[i].max = prod;
      for (int j = i + 1; j < n - 1; j++)
        out[j] = out[j + 1];
      n--;
      grouped = true;
      break;
    }
    if (!grouped) return false;
  }

  *n_out = n;
  return true;
}

static bool _split_dims(
    PolyCtx *ctx,
    const GpuDimExpr *dims,
    int n_dims,
    const int caps[3],
    GpuDimExpr *out,
    int *n_out
) {
  if (!ctx || !dims || !out || !n_out || !gpudim_caps_present(caps) || n_dims <= 0 || n_dims > 3)
    return false;

  bool already_ok = true;
  for (int i = 0; i < n_dims; i++)
    if (dims[i].max > caps[i]) already_ok = false;
  if (already_ok) {
    for (int i = 0; i < n_dims; i++)
      out[i] = dims[i];
    *n_out = n_dims;
    return true;
  }

  for (int i = 0; i < 3; i++) {
    if (i < n_dims) {
      out[i] = dims[i];
    } else {
      out[i].expr = gpudim_const(ctx, 1);
      out[i].max = 1;
    }
  }

  for (int i = 0; i < 3; i++) {
    while (out[i].max > caps[i]) {
      int64_t div = gpudim_smallest_divisor(out[i].max);
      if (div == 1) return false;
      int next = (i + 1) % 3;
      int64_t next_max = 0;
      if (!gpudim_safe_mul_i64(out[next].max, div, &next_max)) return false;
      out[i].expr = gpudim_floordiv_const(ctx, out[i].expr, div);
      out[i].max /= div;
      out[next].expr = gpudim_mul_const(ctx, out[next].expr, div);
      out[next].max = next_max;
    }
  }

  *n_out = (out[2].max == 1) ? 2 : ((out[1].max == 1 && out[2].max == 1) ? 1 : 3);
  return true;
}

static bool gpudim_limited_dims(
    PolyCtx *ctx,
    const GpuDimExpr *dims,
    int n_dims,
    const int caps[3],
    GpuDimExpr *out,
    int *n_out
) {
  if (!gpudim_caps_present(caps)) {
    for (int i = 0; i < n_dims; i++)
      out[i] = dims[i];
    *n_out = n_dims;
    return true;
  }

  GpuDimExpr grouped[POLY_MAX_DIMS];
  int n_grouped = 0;
  if (_group_dims(ctx, dims, n_dims, caps, grouped, &n_grouped)) {
    for (int i = 0; i < n_grouped; i++)
      out[i] = grouped[i];
    *n_out = n_grouped;
    return true;
  }

  if (n_dims > 3) return false;
  return _split_dims(ctx, dims, n_dims, caps, out, n_out);
}

static bool get_grouped_dims(
    PolyCtx *ctx,
    const char *prefix,
    const GpuDimExpr *dims,
    int n_dims,
    const int caps[3],
    bool reverse,
    PolyUOp **out
) {
  if (!ctx || !prefix || !dims || !out || n_dims <= 0 || n_dims > POLY_MAX_DIMS) return false;

  GpuDimExpr ordered[POLY_MAX_DIMS];
  for (int i = 0; i < n_dims; i++)
    ordered[i] = reverse ? dims[n_dims - 1 - i] : dims[i];

  GpuDimExpr limited[POLY_MAX_DIMS];
  int n_limited = 0;
  if (!gpudim_limited_dims(ctx, ordered, n_dims, caps, limited, &n_limited)) return false;

  PolyUOp *raw[3] = {NULL, NULL, NULL};
  for (int i = 0; i < n_limited; i++) {
    char name[16];
    snprintf(name, sizeof(name), "%s%d", prefix, i);
    raw[i] = poly_uop1(
        ctx, POLY_OP_SPECIAL, POLY_WEAKINT, gpudim_special_bound(ctx, limited[i].expr),
        poly_arg_str(name)
    );
  }

  /* Tinygrad 2026-08-22/a9069c177a9d codegen/gpudims.py:51-52 flattens
   * hardware SPECIALs once, then reconstructs every logical dimension. This
   * also defines grouped-axis order; origin bookkeeping reverses that order. */
  PolyUOp *flat = NULL;
  for (int i = 0; i < n_limited; i++) {
    PolyUOp *stride = gpudim_product(ctx, limited, i + 1, n_limited);
    PolyUOp *term = stride ? gpudim_sint_mul(ctx, raw[i], stride) : NULL;
    if (!term) return false;
    flat = flat ? gpudim_add(ctx, flat, term) : term;
  }

  PolyUOp *ordered_out[POLY_MAX_DIMS] = {0};
  for (int i = 0; i < n_dims; i++) {
    PolyUOp *stride = gpudim_product(ctx, ordered, i + 1, n_dims);
    if (!stride) return false;
    PolyUOp *idx = gpudim_floordiv(ctx, flat, stride);
    if (i != 0) idx = gpudim_floormod(ctx, idx, ordered[i].expr);
    ordered_out[i] = poly_graph_rewrite(ctx, idx, poly_symbolic());
    if (!ordered_out[i]) return false;
  }
  for (int i = 0; i < n_dims; i++)
    out[i] = reverse ? ordered_out[n_dims - 1 - i] : ordered_out[i];
  return true;
}

static PolyUOp **uop_src_scratch_alloc(int n, PolyUOp **stack, int stack_cap) {
  if (n <= stack_cap) return stack;
  return malloc((size_t)n * sizeof(PolyUOp *));
}

static void uop_src_scratch_free(PolyUOp **buf, PolyUOp **stack) {
  if (buf && buf != stack) free(buf);
}

PolyUOp *poly_add_gpudims_ex(PolyCtx *ctx, PolyUOp *sink, PolyRendererCaps caps) {
  if (!ctx || !sink || sink->arg.kind == POLY_ARG_NONE) return sink;
  int n_topo = 0;
  PolyUOp **topo = poly_toposort_alloc(ctx, sink, &n_topo);
  for (int i = 0; i < n_topo; i++) {
    if (topo[i]->op == POLY_OP_SPECIAL) {
      poly_toposort_free(topo);
      return sink;
    }
  }

  /* Collect the non-reduce ranges that should become GPU SPECIALs.
   * tinygrad gpudims.py substitutes all global-like and local-like dims, not
   * just the outermost one. We keep the same grouping by axis kind here:
   *   global-like: GLOBAL, THREAD, LOOP fallback
   *   local-like:  WARP, LOCAL, GROUP_REDUCE */
  PolyUOp *global_ranges[POLY_MAX_DIMS];
  int n_global = 0;
  PolyUOp *local_ranges[POLY_MAX_DIMS];
  int n_local = 0;

  for (int i = 0; i < n_topo; i++) {
    if (topo[i]->op != POLY_OP_RANGE) continue;
    PolyAxisType t =
        poly_arg_is_range(topo[i]->arg) ? poly_range_axis_type(topo[i]->arg) : POLY_AXIS_WEAK;
    if ((t == POLY_AXIS_GLOBAL || t == POLY_AXIS_THREAD) && n_global < POLY_MAX_DIMS) {
      int existing = find_range_axis_key(topo[i], global_ranges, n_global);
      if (existing >= 0)
        global_ranges[existing] = topo[i];
      else
        global_ranges[n_global++] = topo[i];
      continue;
    }
    if ((t == POLY_AXIS_WARP || t == POLY_AXIS_LOCAL || t == POLY_AXIS_GROUP_REDUCE) &&
        n_local < POLY_MAX_DIMS) {
      int existing = find_range_axis_key(topo[i], local_ranges, n_local);
      if (existing >= 0)
        local_ranges[existing] = topo[i];
      else
        local_ranges[n_local++] = topo[i];
    }
  }

  if (n_global == 0 && n_local == 0) { /* nothing to parallelize */
    poly_toposort_free(topo);
    return sink;
  }

  qsort(global_ranges, (size_t)n_global, sizeof(PolyUOp *), cmp_range_axis_id_ptr);
  qsort(local_ranges, (size_t)n_local, sizeof(PolyUOp *), cmp_range_axis_id_ptr);

  GpuDimExpr global_dims[POLY_MAX_DIMS];
  PolyUOp *global_idxs[POLY_MAX_DIMS] = {0};
  for (int i = 0; i < n_global; i++) {
    global_dims[i].expr = gpudim_special_bound(ctx, global_ranges[i]->src[0]);
    global_dims[i].max = _dim_max(ctx, global_dims[i].expr);
  }
  GpuDimExpr local_dims[POLY_MAX_DIMS];
  PolyUOp *local_idxs[POLY_MAX_DIMS] = {0};
  for (int i = 0; i < n_local; i++) {
    local_dims[i].expr = gpudim_special_bound(ctx, local_ranges[i]->src[0]);
    local_dims[i].max = _dim_max(ctx, local_dims[i].expr);
  }
  if (caps.has_threads) {
    if (n_global != 1 || n_local != 0) {
      poly_toposort_free(topo);
      return NULL;
    }
    PolyUOp *core_id =
        poly_uop_variable(ctx, "core_id", 0, global_dims[0].max - 1, POLY_INT32, 1, true);
    global_idxs[0] =
        poly_uop1(ctx, POLY_OP_CAST, POLY_WEAKINT, core_id, poly_arg_dtype(POLY_WEAKINT));
  } else {
    if (n_global > 0 &&
        !get_grouped_dims(ctx, "gidx", global_dims, n_global, caps.global_max, true, global_idxs)) {
      poly_toposort_free(topo);
      return NULL;
    }
    if (n_local > 0 &&
        !get_grouped_dims(ctx, "lidx", local_dims, n_local, caps.local_max, false, local_idxs)) {
      poly_toposort_free(topo);
      return NULL;
    }
  }

  /* Substitute every global/local RANGE occurrence with its SPECIAL-derived
   * index. tinygrad/codegen/gpudims.py:58-105 uses s.substitute(subs), so END
   * sources are rebuilt with SPECIALs rather than deleted. */
  int sub_cap = n_topo * 2 + 64;
  PolyUOp **sub_old = (PolyUOp **)malloc((size_t)sub_cap * sizeof(PolyUOp *));
  PolyUOp **sub_new = (PolyUOp **)malloc((size_t)sub_cap * sizeof(PolyUOp *));
  if (!sub_old || !sub_new) {
    free(sub_old);
    free(sub_new);
    poly_toposort_free(topo);
    return sink;
  }
  int n_subs = 0;

  /* Seed global substitutions from tinygrad get_grouped_dims(..., reverse=True).
   * If a logical dimension was split, global_idxs[i] reconstructs the original
   * logical RANGE from multiple hardware SPECIALs. */
  for (int i = 0; i < n_global; i++) {
    for (int j = 0; j < n_topo; j++)
      if (topo[j]->op == POLY_OP_RANGE && range_same_axis_key(topo[j], global_ranges[i]))
        add_gpudim_sub(&sub_old, &sub_new, &n_subs, &sub_cap, topo[j], global_idxs[i]);
  }

  /* Seed local substitutions from tinygrad get_grouped_dims(...). */
  for (int g = 0; g < n_local; g++) {
    for (int j = 0; j < n_topo; j++)
      if (topo[j]->op == POLY_OP_RANGE && range_same_axis_key(topo[j], local_ranges[g]))
        add_gpudim_sub(&sub_old, &sub_new, &n_subs, &sub_cap, topo[j], local_idxs[g]);
  }

  PolyUOp *new_sink = sink;
  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    if (find_range_axis_key(u, global_ranges, n_global) >= 0 ||
        find_range_axis_key(u, local_ranges, n_local) >= 0)
      continue;

    /* Check if any source was substituted */
    bool changed = false;
    PolyUOp *stack_new_srcs[64];
    PolyUOp **new_srcs = uop_src_scratch_alloc(u->n_src, stack_new_srcs, 64);
    if (!new_srcs) continue;
    for (int j = 0; j < u->n_src; j++) {
      PolyUOp *mapped = NULL;
      for (int k = 0; k < n_subs; k++) {
        if (sub_old[k] == u->src[j]) {
          mapped = sub_new[k];
          break;
        }
      }
      if (mapped) {
        new_srcs[j] = mapped;
        changed = true;
      } else {
        new_srcs[j] = u->src[j];
      }
    }

    if (changed) {
      PolyUOp *new_u = poly_uop_replace_src(ctx, u, new_srcs);

      /* Gated STORE for GLOBAL buffers missing local dims.
       * Pinned tinygrad/codegen/gpudims.py:92-99 computes the exact set
       * difference `local_dims - idx.ranges`; a STORE may use one local axis
       * while still omitting a grouped-reduce axis. Gate only the omitted
       * occurrences and keep validity in the integer INDEX coordinate. */
      if (new_u->op == POLY_OP_STORE && n_local > 0 && new_u->n_src >= 2) {
        PolyUOp *idx = new_u->src[0];
        PolyUOp *original_idx = u->src[0];
        /* Walk through CASTs to find the INDEX and retain the wrapper chain. */
        PolyUOp *wrappers[16];
        int n_wrappers = 0;
        PolyUOp *raw_idx = idx;
        while (raw_idx->op == POLY_OP_CAST && raw_idx->n_src == 1 &&
               n_wrappers < (int)(sizeof(wrappers) / sizeof(wrappers[0]))) {
          wrappers[n_wrappers++] = raw_idx;
          raw_idx = raw_idx->src[0];
        }
        while (original_idx && original_idx->op == POLY_OP_CAST && original_idx->n_src == 1)
          original_idx = original_idx->src[0];

        /* Pinned tinygrad/codegen/gpudims.py:92 identifies a global STORE
         * from idx.src[0].addrspace. INDEX itself may already carry the
         * element dtype, as normal rangeified stores do. */
        if (raw_idx->op == POLY_OP_INDEX && raw_idx->n_src == 2 && raw_idx->src[0] &&
            poly_program_memory_is(raw_idx->src[0], POLY_ADDR_GLOBAL) && original_idx &&
            original_idx->op == POLY_OP_INDEX) {
          bool missing_local[POLY_MAX_DIMS] = {0};
          bool has_missing_local = false;
          for (int g = 0; g < n_local; g++) {
            missing_local[g] = !poly_uop_in_ranges(ctx, original_idx, local_ranges[g]);
            has_missing_local |= missing_local[g];
          }
          if (has_missing_local) {
            /* Pinned tinygrad/codegen/gpudims.py:96 builds
             * UOp.uprod(*[x.eq(0) ...]). Keep zero const-like with the weak
             * lidx so pm_lower_index_dtype can lower both operands together;
             * x.eq(0) is CMPNE(x, 0).logical_not(). */
            PolyUOp *gate = NULL;
            for (int g = 0; g < n_local; g++) {
              if (!missing_local[g]) continue;
              /* Find the lidx SPECIAL we created for this group range */
              PolyUOp *lidx = NULL;
              for (int k = 0; k < n_subs; k++) {
                if (sub_old[k] && sub_old[k]->op == POLY_OP_RANGE &&
                    range_same_axis_key(sub_old[k], local_ranges[g])) {
                  lidx = sub_new[k];
                  break;
                }
              }
              if (!lidx) continue;
              PolyUOp *not_zero = poly_uop2(
                  ctx, POLY_OP_CMPNE, POLY_BOOL, lidx, poly_const_like_int(ctx, lidx, 0),
                  poly_arg_none()
              );
              PolyUOp *eq_zero = poly_uop2(
                  ctx, POLY_OP_CMPNE, POLY_BOOL, not_zero,
                  poly_const_like_bool(ctx, not_zero, true), poly_arg_none()
              );
              gate = gate ? poly_uop2(ctx, POLY_OP_AND, POLY_BOOL, gate, eq_zero, poly_arg_none())
                          : eq_zero;
            }

            if (gate) {
              PolyUOp *coord = raw_idx->src[1];
              int64_t lanes = poly_uop_max_numel(ctx, coord);
              if (lanes < 1 || lanes > UINT16_MAX) {
                uop_src_scratch_free(new_srcs, stack_new_srcs);
                free(sub_old);
                free(sub_new);
                poly_toposort_free(topo);
                return NULL;
              }
              PolyUOp *coord_gate = gate;
              if (lanes > 1) {
                PolyUOp **gate_lanes = malloc((size_t)lanes * sizeof(*gate_lanes));
                if (!gate_lanes) {
                  uop_src_scratch_free(new_srcs, stack_new_srcs);
                  free(sub_old);
                  free(sub_new);
                  poly_toposort_free(topo);
                  return NULL;
                }
                for (int lane = 0; lane < lanes; lane++)
                  gate_lanes[lane] = gate;
                coord_gate = poly_uop_stack(ctx, gate_lanes, (int)lanes);
                free(gate_lanes);
              }
              PolyUOp *invalid = poly_uop_const(ctx, poly_arg_invalid(), coord->dtype);
              PolyUOp *where = poly_uop3(
                  ctx, POLY_OP_WHERE, coord->dtype, coord_gate, coord, invalid, poly_arg_none()
              );
              PolyUOp *gated_src[2] = {raw_idx->src[0], where};
              PolyUOp *final_idx = poly_uop_replace_src(ctx, raw_idx, gated_src);

              for (int w = n_wrappers - 1; w >= 0; w--) {
                PolyUOp *wrapper_src[1] = {final_idx};
                final_idx = poly_uop_replace_src(ctx, wrappers[w], wrapper_src);
              }

              /* Rebuild STORE with the Invalid-bearing integer coordinate. */
              PolyUOp *stack_store_srcs[64];
              PolyUOp **store_srcs = uop_src_scratch_alloc(new_u->n_src, stack_store_srcs, 64);
              if (store_srcs) {
                store_srcs[0] = final_idx;
                for (int j = 1; j < new_u->n_src; j++)
                  store_srcs[j] = new_u->src[j];
                new_u =
                    poly_uop(ctx, new_u->op, new_u->dtype, store_srcs, new_u->n_src, new_u->arg);
                uop_src_scratch_free(store_srcs, stack_store_srcs);
              }
            }
          }
        }
      }

      add_gpudim_sub(&sub_old, &sub_new, &n_subs, &sub_cap, u, new_u);
      if (u == sink) new_sink = new_u;
    }
    uop_src_scratch_free(new_srcs, stack_new_srcs);
  }

  free(sub_old);
  free(sub_new);
  poly_toposort_free(topo);
  return new_sink;
}

PolyUOp *poly_add_gpudims(PolyCtx *ctx, PolyUOp *sink) {
  PolyRendererCaps caps = {0};
  return poly_add_gpudims_ex(ctx, sink, caps);
}
