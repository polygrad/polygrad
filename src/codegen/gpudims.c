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
  /* Pinned _split_dims includes ceil(sqrt(x)); only x=2 has a divisor
   * above floor(sqrt(x)) in that interval. Avoid floating-point bounds. */
  if (x == 2) return 2;
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
  if (!ctx || !dims || !out || !n_out || !gpudim_caps_present(caps) || n_dims <= 0) return false;

  int n = n_dims;
  for (int i = 0; i < n; i++)
    out[i] = dims[i];

  while (true) {
    bool exceeds = n > 3;
    /* Python's any(d > cap) evaluates symbolic predicates, not vmax alone.
     * Preserve short-circuit order; an unresolved comparison is an error. */
    for (int i = 0; !exceeds && i < n; i++) {
      int64_t lo, hi;
      poly_uop_minmax(ctx, out[i].expr, &lo, &hi);
      if (lo <= caps[i] && hi > caps[i]) return false;
      exceeds = lo > caps[i];
    }
    if (!exceeds) break;
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
  for (int i = 0; already_ok && i < n_dims; i++) {
    int64_t lo, hi;
    poly_uop_minmax(ctx, dims[i].expr, &lo, &hi);
    if (lo <= caps[i] && hi > caps[i]) return false;
    already_ok = hi <= caps[i];
  }
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
      /* _split_dims factors an actual integer. A variable's vmax is not
       * its runtime extent: dividing it could silently omit workitems. */
      int64_t value;
      if (!gpudim_const_i64(out[i].expr, &value)) return false;
      int64_t div = gpudim_smallest_divisor(out[i].max);
      if (div == 1) return false;
      int next = (i + 1) % 3;
      int64_t next_max = 0;
      if (!gpudim_safe_mul_i64(out[next].max, div, &next_max)) return false;
      out[i].expr = gpudim_const(ctx, value / div);
      out[i].max /= div;
      out[next].expr = gpudim_sint_mul(ctx, out[next].expr, gpudim_const(ctx, div));
      if (!out[i].expr || !out[next].expr) return false;
      out[next].max = next_max;
    }
  }

  *n_out = (out[2].max == 1) ? 2 : 3;
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

  if (_group_dims(ctx, dims, n_dims, caps, out, n_out)) return true;

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
  if (!ctx || !prefix || !dims || !out || n_dims <= 0) return false;
  /* Uncapped renderer dimensions are a list, not a three-axis launch tuple.
   * Splitting a single capped axis can conversely require three slots. */
  size_t capacity = n_dims > 3 ? (size_t)n_dims : 3;
  if (capacity > SIZE_MAX / sizeof(GpuDimExpr)) return false;
  GpuDimExpr *ordered = malloc(capacity * sizeof(*ordered));
  GpuDimExpr *limited = malloc(capacity * sizeof(*limited));
  PolyUOp **raw = malloc(capacity * sizeof(*raw));
  bool ok = false;
  if (!ordered || !limited || !raw) goto done;
  for (int i = 0; i < n_dims; i++)
    ordered[i] = reverse ? dims[n_dims - 1 - i] : dims[i];

  int n_limited = 0;
  if (!gpudim_limited_dims(ctx, ordered, n_dims, caps, limited, &n_limited)) goto done;

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
    if (!term) goto done;
    flat = flat ? gpudim_add(ctx, flat, term) : term;
  }

  for (int i = 0; i < n_dims; i++) {
    PolyUOp *stride = gpudim_product(ctx, ordered, i + 1, n_dims);
    if (!stride) goto done;
    PolyUOp *idx = gpudim_floordiv(ctx, flat, stride);
    if (i != 0) idx = gpudim_floormod(ctx, idx, ordered[i].expr);
    out[reverse ? n_dims - 1 - i : i] = poly_graph_rewrite(ctx, idx, poly_symbolic());
    if (!out[reverse ? n_dims - 1 - i : i]) goto done;
  }
  ok = true;
done:
  free(ordered);
  free(limited);
  free(raw);
  return ok;
}

/* Tinygrad add_gpudims builds one substitution dictionary and applies it once.
 * In particular, a global STORE needs a local-zero gate even when none of its
 * existing inputs mention a parallel RANGE. Scratch is owned by this pass;
 * failures do not publish a partially substituted graph. */
PolyUOp *poly_add_gpudims_ex(PolyCtx *ctx, PolyUOp *sink, PolyRendererCaps caps) {
  if (!ctx || !sink || sink->arg.kind == POLY_ARG_NONE) return sink;
  int n_topo = 0;
  PolyUOp **topo = poly_toposort_alloc(ctx, sink, &n_topo);
  PolyUOp *ret = NULL;
  PolyUOp **storage = NULL, **subs = NULL;
  GpuDimExpr *dims = NULL;
  if (!topo || n_topo <= 0) goto done;
  for (int i = 0; i < n_topo; i++) {
    if (topo[i]->op == POLY_OP_SPECIAL) {
      ret = sink;
      goto done;
    }
  }
  size_t n = (size_t)n_topo;
  if (n > SIZE_MAX / (5 * sizeof(*storage)) || n > SIZE_MAX / (2 * sizeof(*dims))) goto done;
  storage = calloc(5 * n, sizeof(*storage));
  subs = malloc(2 * n * sizeof(*subs));
  dims = malloc(2 * n * sizeof(*dims));
  if (!storage || !subs || !dims) goto done;
  PolyUOp **global_ranges = storage, **local_ranges = storage + n;
  PolyUOp **global_idxs = storage + 2 * n, **local_idxs = storage + 3 * n;
  PolyUOp **all_ranges = storage + 4 * n;
  GpuDimExpr *global_dims = dims, *local_dims = dims + n;
  PolyUOp **from = subs, **to = subs + n;
  int n_global = 0, n_local = 0, n_sub = 0, n_ranges = 0;
  /* Tinygrad's all_ranges dictionary resolves the last occurrence of a key
   * before deciding whether that key denotes global, local or serial work. */
  for (int i = 0; i < n_topo; i++) {
    PolyUOp *r = topo[i];
    if (r->op != POLY_OP_RANGE || r->n_src != 1 || !poly_arg_is_range(r->arg)) continue;
    int existing = find_range_axis_key(r, all_ranges, n_ranges);
    all_ranges[existing >= 0 ? existing : n_ranges++] = r;
  }
  for (int i = 0; i < n_ranges; i++) {
    PolyUOp *r = all_ranges[i];
    PolyAxisType type = poly_range_axis_type(r->arg);
    if (type == POLY_AXIS_GLOBAL || type == POLY_AXIS_THREAD) {
      global_ranges[n_global++] = r;
    } else if (type == POLY_AXIS_WARP || type == POLY_AXIS_LOCAL || type == POLY_AXIS_GROUP_REDUCE) {
      local_ranges[n_local++] = r;
    }
  }
  if (n_global == 0 && n_local == 0) {
    ret = sink;
    goto done;
  }
  qsort(global_ranges, (size_t)n_global, sizeof(*global_ranges), cmp_range_axis_id_ptr);
  qsort(local_ranges, (size_t)n_local, sizeof(*local_ranges), cmp_range_axis_id_ptr);
  for (int i = 0; i < n_global; i++) {
    global_dims[i].expr = poly_graph_rewrite(ctx, global_ranges[i]->src[0], poly_symbolic());
    global_dims[i].expr = gpudim_special_bound(ctx, global_dims[i].expr);
    if (!global_dims[i].expr) goto done;
    global_dims[i].max = _dim_max(ctx, global_dims[i].expr);
  }
  for (int i = 0; i < n_local; i++) {
    local_dims[i].expr = poly_graph_rewrite(ctx, local_ranges[i]->src[0], poly_symbolic());
    local_dims[i].expr = gpudim_special_bound(ctx, local_dims[i].expr);
    if (!local_dims[i].expr) goto done;
    local_dims[i].max = _dim_max(ctx, local_dims[i].expr);
  }
  bool no_locals = sink->arg.kind == POLY_ARG_KERNEL_INFO && sink->arg.kernel_info &&
                   sink->arg.kernel_info->dont_use_locals;
  if (caps.has_threads) {
    if (n_global != 1 || n_local != 0) goto done;
    PolyUOp *core_id = poly_uop_variable(
        ctx, "core_id", poly_arg_int(0), poly_arg_int(global_dims[0].max - 1), POLY_INT32, 1, true
    );
    global_idxs[0] = poly_cast(ctx, core_id, POLY_WEAKINT);
    if (!global_idxs[0]) goto done;
  } else {
    if (no_locals && n_local) goto done;
    if (n_local &&
        !get_grouped_dims(ctx, "lidx", local_dims, n_local, caps.local_max, false, local_idxs))
      goto done;
    if (n_global && !get_grouped_dims(
                        ctx, no_locals ? "idx" : "gidx", global_dims, n_global, caps.global_max,
                        true, global_idxs
                    ))
      goto done;
  }
  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    if (u->op == POLY_OP_RANGE) {
      /* Pinned add_gpudims checks r.arg[1], not the trailing type field:
       * the guard applies to ordinary (axis, REDUCE) tuples. */
      if (poly_range_n_extra(u->arg) == 0 && poly_range_axis_type(u->arg) == POLY_AXIS_REDUCE)
        continue;
      int axis = find_range_axis_key(u, global_ranges, n_global);
      PolyUOp *replacement = axis >= 0 ? global_idxs[axis] : NULL;
      if (!replacement && (axis = find_range_axis_key(u, local_ranges, n_local)) >= 0)
        replacement = local_idxs[axis];
      if (replacement) {
        from[n_sub] = u;
        to[n_sub++] = replacement;
      }
    }
    if (u->op != POLY_OP_STORE || u->n_src < 2 || !n_local) continue;
    PolyUOp *idx = u->src[0];
    /* Preserve admitted CAST wrappers through the shared substitution owner. */
    while (idx->op == POLY_OP_CAST && idx->n_src == 1)
      idx = idx->src[0];
    if (idx->op != POLY_OP_INDEX || idx->n_src != 2 ||
        !poly_program_memory_is(idx->src[0], POLY_ADDR_GLOBAL))
      continue;
    PolyUOp *gate = NULL;
    for (int g = 0; g < n_local; g++) {
      if (poly_uop_in_ranges(ctx, idx, local_ranges[g])) continue;
      PolyUOp *eq = poly_eq(ctx, local_ranges[g], poly_const_like_int(ctx, local_ranges[g], 0));
      gate = gate ? poly_alu2(ctx, POLY_OP_AND, gate, eq) : eq;
      if (!gate) goto done;
    }
    if (!gate) continue;
    PolyUOp *coord = idx->src[1];
    int64_t lanes = poly_uop_max_numel(ctx, coord);
    if (lanes < 1 || lanes > UINT16_MAX) goto done;
    if (lanes > 1) {
      PolyUOp **lane_gates = malloc((size_t)lanes * sizeof(*lane_gates));
      if (!lane_gates) goto done;
      for (int lane = 0; lane < lanes; lane++)
        lane_gates[lane] = gate;
      gate = poly_uop_stack(ctx, lane_gates, (int)lanes);
      free(lane_gates);
      if (!gate) goto done;
    }
    PolyUOp *invalid = poly_uop_const(ctx, poly_arg_invalid(), coord->dtype);
    PolyUOp *gated =
        poly_uop3(ctx, POLY_OP_WHERE, coord->dtype, gate, coord, invalid, poly_arg_none());
    PolyUOp *src[2] = {idx->src[0], gated};
    PolyUOp *replacement = gated ? poly_uop_replace_src(ctx, idx, src) : NULL;
    if (!replacement) goto done;
    from[n_sub] = idx;
    to[n_sub++] = replacement;
  }
  /* RANGE and STORE are disjoint source nodes, so n_sub never exceeds n_topo.
   * Unlike the convenience substitute wrapper, _many reports scratch failure. */
  if (poly_uop_substitute_many(ctx, &sink, 1, from, to, n_sub, &ret) != 0) ret = NULL;
done:
  free(storage);
  free(subs);
  free(dims);
  poly_toposort_free(topo);
  return ret;
}

PolyUOp *poly_add_gpudims(PolyCtx *ctx, PolyUOp *sink) {
  PolyRendererCaps caps = {0};
  return poly_add_gpudims_ex(ctx, sink, caps);
}
