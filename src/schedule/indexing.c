/* C implementation of tinygrad schedule/indexing.py. */

#include "schedule/indexing.h"
#include "ctx.h"
#include "device.h"
#include "uop/upat.h"
#include "utils.h"
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef POLY_TESTING
static _Thread_local int range_scratch_fail_after = -1;
void poly_test_range_scratch_fail_after(int count) {
  range_scratch_fail_after = count;
}
#endif

static bool range_scratch_alloc_fails(void) {
#ifdef POLY_TESTING
  if (range_scratch_fail_after == 0) {
    range_scratch_fail_after = -1;
    return true;
  }
  if (range_scratch_fail_after > 0) range_scratch_fail_after--;
#endif
  return false;
}

#ifdef POLY_TESTING
static _Thread_local const char *indexing_fail_site;
static _Thread_local int indexing_fail_after = -1;
static _Thread_local bool indexing_alloc_failed;
void poly_test_indexing_alloc_fail_after(const char *site, int count) {
  indexing_fail_site = site;
  indexing_fail_after = count;
  indexing_alloc_failed = false;
}
bool poly_test_indexing_alloc_failed(void) {
  return indexing_alloc_failed;
}
#endif

static bool indexing_alloc_fails(const char *site) {
#ifdef POLY_TESTING
  if (indexing_fail_after >= 0 && indexing_fail_site && !strcmp(site, indexing_fail_site)) {
    if (indexing_fail_after-- == 0) return indexing_alloc_failed = true;
  }
#else
  (void)site;
#endif
  return false;
}

static void *indexing_alloc(size_t size, const char *site) {
  return indexing_alloc_fails(site) ? NULL : malloc(size);
}

static void *indexing_realloc(void *ptr, size_t size, const char *site) {
  return indexing_alloc_fails(site) ? NULL : realloc(ptr, size);
}

static PolyUOp *index_const(PolyCtx *ctx, int64_t value) {
  return poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(value));
}

static PolyUOp *to_index_dtype(PolyCtx *ctx, PolyUOp *u) {
  if (!u || poly_dtype_eq(u->dtype, POLY_WEAKINT)) return u;
  return poly_uop1(ctx, POLY_OP_CAST, POLY_WEAKINT, u, poly_arg_none());
}

/* Current rangeify.py:_mop_index accepts identical simplified suffix UOps and
 * otherwise requires their symbolic inequality to resolve false. */
static bool resolve_shape_equal(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  if (!a || !b) return false;
  a = poly_graph_rewrite(ctx, a, poly_symbolic());
  b = poly_graph_rewrite(ctx, b, poly_symbolic());
  if (!a || !b) return false;
  if (a == b) return true;
  PolyUOp *different = poly_binop(ctx, POLY_OP_CMPNE, a, b);
  return different && poly_uop_resolve(ctx, different, 1) == 0;
}

static PolyUOp *index_neg_like_tinygrad(PolyCtx *ctx, PolyUOp *u) {
  /* tinygrad ElementwiseMixin.neg() constructs x * -1, not a NEG UOp.
   * Movement indexes run through symbolic div/mod simplification before late
   * decompositions can introduce NEG, so keep this source shape identical. */
  return poly_uop2(
      ctx, POLY_OP_MUL, POLY_WEAKINT, to_index_dtype(ctx, u), index_const(ctx, -1), poly_arg_none()
  );
}

static _Thread_local PolyPatternMatcher *g_pm_reshape_indexing = NULL;
static _Thread_local PolyPatternMatcher *g_pm_pad_valid = NULL;

static PolyPatternMatcher *poly_pm_reshape_indexing(void) {
  if (g_pm_reshape_indexing) return g_pm_reshape_indexing;
  /* Pinned tinygrad/schedule/indexing.py:125-127. The three matchers run as
   * one fixed-point rewrite over the complete ordered coordinate SINK. */
  PolyPatternMatcher *symbolic_valid = poly_pm_concat(poly_symbolic(), poly_pm_simplify_valid());
  g_pm_reshape_indexing =
      poly_pm_thread_cache(poly_pm_concat(symbolic_valid, poly_pm_drop_and_clauses()));
  poly_pm_destroy(symbolic_valid);
  return g_pm_reshape_indexing;
}

static PolyPatternMatcher *poly_pm_pad_valid(void) {
  if (g_pm_pad_valid) return g_pm_pad_valid;
  /* Pinned tinygrad/schedule/indexing.py:136-141 simplifies the newly added
   * PAD validity before wrapping the shifted coordinate in WHERE(...,Invalid). */
  g_pm_pad_valid = poly_pm_thread_cache(poly_pm_concat(poly_symbolic(), poly_pm_simplify_valid()));
  return g_pm_pad_valid;
}

/* Pinned schedule/indexing.py:82-86 derives the PAD value wrapper from the
 * get_valid() projection of every transformed coordinate.  In particular, an
 * unchanged coordinate can already carry invalidity from an earlier movement
 * and must remain part of this PAD's wrapper predicate. */
static bool pad_wrapper_valid_from_coords(
    PolyCtx *ctx,
    PolyUOp **coords,
    int n_coords,
    int n_out,
    PolyUOp **valid_out
) {
  PolyUOp *valid = NULL;
  PolyUOp *falsev = poly_uop0(ctx, POLY_OP_CONST, POLY_BOOL, poly_arg_bool(false));
  for (int i = 0; i < n_coords; i++) {
    PolyUOp *dim_valid = i < n_out ? poly_uop_get_valid(ctx, coords[i]) : falsev;
    if (!dim_valid) return false;
    if (dim_valid->op == POLY_OP_CONST && dim_valid->arg.kind == POLY_ARG_BOOL && dim_valid->arg.b)
      continue;
    valid = valid ? poly_uop2(ctx, POLY_OP_AND, POLY_BOOL, valid, dim_valid, poly_arg_none())
                  : dim_valid;
  }
  *valid_out = valid;
  if (valid && poly_getenv_int("POLY_TRACE_PAD_VALID", 0)) {
    static _Thread_local int pad_valid_index = 0;
    int n_topo = 0, n_and = 0;
    PolyUOp **topo = poly_toposort_alloc(NULL, valid, &n_topo);
    for (int i = 0; topo && i < n_topo; i++)
      n_and += topo[i]->op == POLY_OP_AND;
    pad_valid_index++;
    if (n_and >= 3) {
      fprintf(
          stderr, "PG_PAD_VALID index=%d coords=%d nodes=%d and=%d\n", pad_valid_index, n_coords,
          n_topo, n_and
      );
      poly_uop_dump_tree(stderr, valid, 0, 24);
      if (n_and >= 10) {
        PolyUOp *simplified = poly_graph_rewrite(ctx, valid, poly_symbolic());
        int n_simplified = 0, n_simplified_and = 0;
        PolyUOp **simplified_topo = poly_toposort_alloc(NULL, simplified, &n_simplified);
        for (int i = 0; simplified_topo && i < n_simplified; i++)
          n_simplified_and += simplified_topo[i]->op == POLY_OP_AND;
        fprintf(
            stderr, "PG_PAD_VALID_SIMPLIFIED index=%d nodes=%d and=%d\n", pad_valid_index,
            n_simplified, n_simplified_and
        );
        poly_uop_dump_tree(stderr, simplified, 0, 24);
        poly_toposort_free(simplified_topo);
      }
    }
    poly_toposort_free(topo);
  }
  return true;
}

/* Reshape index transform */

bool poly_apply_reshape(
    PolyCtx *ctx,
    PolyUOp **in_shape,
    int n_in,
    PolyUOp **out_shape,
    int n_out,
    PolyUOp **out_ranges,
    int n_ranges,
    PolyUOp **in_ranges,
    int *n_in_out
) {
  if (!ctx || n_in < 0 || n_in > POLY_MAX_DIMS || n_out < 0 || n_out > POLY_MAX_DIMS ||
      n_ranges != n_out || (n_in > 0 && !in_shape) || (n_out > 0 && (!out_shape || !out_ranges)) ||
      !in_ranges || !n_in_out)
    return false;

  /* Pinned indexing.py:142-145 first simplifies the ordered coordinate SINK,
   * replaces its active ranges with PLACEHOLDER ranges, applies the cached
   * reshape, then restores the original ranges. This prevents existing range
   * bounds/validity from changing commutative rewrite order inside reshape. */
  PolyUOp *coordinate_sink =
      poly_uop(ctx, POLY_OP_SINK, POLY_VOID, out_ranges, n_ranges, poly_arg_none());
  coordinate_sink =
      coordinate_sink ? poly_graph_rewrite(ctx, coordinate_sink, poly_symbolic()) : NULL;
  if (!coordinate_sink || coordinate_sink->op != POLY_OP_SINK || coordinate_sink->n_src != n_ranges)
    return false;

  int range_cap = 0;
  PolyUOp **coordinate_topo = poly_toposort_alloc(ctx, coordinate_sink, &range_cap);
  if (!coordinate_topo && range_cap != 0) return false;
  poly_toposort_free(coordinate_topo);
  PolyUOp **original_ranges =
      range_cap > 0 ? malloc((size_t)range_cap * sizeof(*original_ranges)) : NULL;
  PolyUOp **placeholder_ranges =
      range_cap > 0 ? malloc((size_t)range_cap * sizeof(*placeholder_ranges)) : NULL;
  if (range_cap > 0 && (!original_ranges || !placeholder_ranges)) {
    free(original_ranges);
    free(placeholder_ranges);
    return false;
  }
  int n_replacements =
      range_cap > 0 ? poly_uop_ranges(ctx, coordinate_sink, original_ranges, range_cap) : 0;
  bool placeholder_ok = true;
  for (int i = 0; i < n_replacements; i++) {
    PolyUOp *range = original_ranges[i];
    if (!range || range->op != POLY_OP_RANGE || range->n_src != 1) {
      placeholder_ok = false;
      break;
    }
    placeholder_ranges[i] = poly_uop1(
        ctx, POLY_OP_RANGE, range->dtype, range->src[0], poly_arg_range(i, POLY_AXIS_PLACEHOLDER)
    );
    if (!placeholder_ranges[i]) {
      placeholder_ok = false;
      break;
    }
  }
  if (!placeholder_ok) {
    free(original_ranges);
    free(placeholder_ranges);
    return false;
  }
  PolyUOp *placeholder_sink = n_replacements > 0 ? poly_uop_substitute(
                                                       ctx, coordinate_sink, original_ranges,
                                                       placeholder_ranges, n_replacements
                                                   )
                                                 : coordinate_sink;
  if (!placeholder_sink || placeholder_sink->op != POLY_OP_SINK ||
      placeholder_sink->n_src != n_ranges) {
    free(original_ranges);
    free(placeholder_ranges);
    return false;
  }

  /* Pinned _apply_reshape uses the exact symbolic output shape to flatten,
   * then floor-mod/divides by each exact symbolic input dimension
   * (schedule/indexing.py:113-127). Cached PolyShape dimensions are allocation
   * maxima only, so use them solely for static dimensions. */
  /* Pinned _apply_reshape constructs every acc*src term and all div/mod nodes
   * before the one complete-SINK rewrite. In particular, do not call the
   * shared flat-index helper here: its eager symbolic rewrite combines
   * Invalid guards before reshape validity simplification can reason locally. */
  PolyUOp *acc = index_const(ctx, 1);
  PolyUOp *combined = index_const(ctx, 0);
  for (int i = n_out - 1; i >= 0; i--) {
    PolyUOp *term = poly_uop2(
        ctx, POLY_OP_MUL, POLY_WEAKINT, acc, to_index_dtype(ctx, placeholder_sink->src[i]),
        poly_arg_none()
    );
    combined =
        term ? poly_uop2(ctx, POLY_OP_ADD, POLY_WEAKINT, combined, term, poly_arg_none()) : NULL;
    acc = acc ? poly_uop2(
                    ctx, POLY_OP_MUL, POLY_WEAKINT, acc, to_index_dtype(ctx, out_shape[i]),
                    poly_arg_none()
                )
              : NULL;
    if (!combined || !acc) break;
  }
  if (!combined || !acc) {
    free(original_ranges);
    free(placeholder_ranges);
    return false;
  }

  for (int j = n_in - 1; j >= 0; j--) {
    PolyUOp *dim_val = to_index_dtype(ctx, in_shape[j]);
    in_ranges[j] =
        poly_uop2(ctx, POLY_OP_FLOORMOD, POLY_WEAKINT, combined, dim_val, poly_arg_none());
    combined = poly_uop2(ctx, POLY_OP_FLOORDIV, POLY_WEAKINT, combined, dim_val, poly_arg_none());
    if (!in_ranges[j] || !combined) {
      free(original_ranges);
      free(placeholder_ranges);
      return false;
    }
  }
  if (n_in > 0) {
    PolyUOp *sink = poly_uop(ctx, POLY_OP_SINK, POLY_VOID, in_ranges, n_in, poly_arg_none());
    PolyUOp *simplified_placeholder =
        sink ? poly_graph_rewrite(ctx, sink, poly_pm_reshape_indexing()) : NULL;
    PolyUOp *simplified =
        simplified_placeholder && n_replacements > 0
            ? poly_uop_substitute(
                  ctx, simplified_placeholder, placeholder_ranges, original_ranges, n_replacements
              )
            : simplified_placeholder;
    if (!simplified || simplified->op != POLY_OP_SINK || simplified->n_src != n_in) {
      free(original_ranges);
      free(placeholder_ranges);
      return false;
    }
    for (int j = 0; j < n_in; j++)
      in_ranges[j] = simplified->src[j];
  }
  free(original_ranges);
  free(placeholder_ranges);
  *n_in_out = n_in;
  return true;
}

bool poly_reshape_indices(
    PolyCtx *ctx,
    PolyUOp *reshape,
    PolyUOp **out_ranges,
    int n_out,
    PolyUOp **in_ranges,
    int *n_in_out
) {
  if (!ctx || !reshape || reshape->op != POLY_OP_RESHAPE || reshape->n_src != 2 || n_out < 0 ||
      (n_out > 0 && !out_ranges) || !in_ranges || !n_in_out)
    return false;
  int out_ndim = poly_uop_ndim(ctx, reshape);
  int in_ndim = poly_uop_ndim(ctx, reshape->src[0]);
  if (out_ndim < n_out || in_ndim < 0 || out_ndim > POLY_MAX_DIMS || in_ndim > POLY_MAX_DIMS)
    return false;

  /* Current rangeify.py:_mop_index keeps only matching unindexed suffixes. */
  int suffix_ndim = out_ndim - n_out;
  int n_in = in_ndim - suffix_ndim;
  if (n_in < 0) return false;
  PolyUOp *in_shape[POLY_MAX_DIMS], *out_shape[POLY_MAX_DIMS];
  for (int i = 0; i < in_ndim; i++) {
    in_shape[i] = poly_uop_shape_dim(ctx, reshape->src[0], i);
    if (!in_shape[i]) return false;
  }
  for (int i = 0; i < out_ndim; i++) {
    out_shape[i] = poly_uop_shape_dim(ctx, reshape, i);
    if (!out_shape[i]) return false;
  }
  for (int i = 0; i < suffix_ndim; i++) {
    if (!resolve_shape_equal(ctx, in_shape[n_in + i], out_shape[n_out + i])) return false;
  }
  return poly_apply_reshape(
      ctx, in_shape, n_in, out_shape, n_out, out_ranges, n_out, in_ranges, n_in_out
  );
}

/* apply_movement_op */

bool poly_apply_movement_op(
    PolyCtx *ctx,
    PolyUOp *movement,
    PolyOps op,
    PolyShape in_shape,
    PolyArg arg,
    PolyUOp **out_rngs,
    int n_out,
    PolyUOp **in_rngs,
    int *n_in_out,
    PolyUOp **valid_out
) {
  if (valid_out) *valid_out = NULL;

  switch (op) {

  /* Current raw EXPAND only injects leading axes. apply_movement_op drops
   * exactly those leading ranges (schedule/indexing.py:166-175). */
  case POLY_OP_EXPAND: {
    if (!movement || arg.kind != POLY_ARG_NONE || movement->n_src != 2) return false;
    PolyUOp *marg[POLY_MAX_DIMS];
    int n_marg = poly_uop_as_shape(ctx, movement->src[1], marg, POLY_MAX_DIMS);
    if (n_marg < 0 || n_marg > n_out || n_out - n_marg != in_shape.ndim) return false;
    for (int i = n_marg; i < n_out; i++)
      in_rngs[i - n_marg] = out_rngs[i];
    *n_in_out = n_out - n_marg;
    return true;
  }

  /* PERMUTE: reorder ranges by inverse permutation */
  case POLY_OP_PERMUTE: {
    if (arg.kind != POLY_ARG_INT_TUPLE) return false;
    int n = arg.int_tuple.n;
    PolyUOp *zero = index_const(ctx, 0);
    for (int i = 0; i < n; i++)
      in_rngs[i] = zero;
    for (int i = 0; i < n && i < n_out; i++) {
      int p = (int)arg.int_tuple.vals[i];
      if (p >= 0 && p < n) in_rngs[p] = out_rngs[i];
    }
    *n_in_out = n;
    return true;
  }

  /* SHRINK: offset range by start value */
  case POLY_OP_SHRINK: {
    if (movement && movement->arg.kind == POLY_ARG_NONE && movement->n_src >= 3) {
      int n = in_shape.ndim;
      PolyUOp *offsets[POLY_MAX_DIMS];
      if (poly_uop_as_shape(ctx, movement->src[1], offsets, POLY_MAX_DIMS) != n) return false;
      PolyUOp *zero = index_const(ctx, 0);
      for (int i = 0; i < n; i++) {
        if (i >= n_out) {
          in_rngs[i] = zero;
          continue;
        }
        PolyUOp *off = offsets[i];
        int64_t off_i = 0;
        if (poly_uop_const_i64(off, &off_i) == 0 && off_i == 0) {
          in_rngs[i] = out_rngs[i];
        } else {
          in_rngs[i] = poly_uop2(
              ctx, POLY_OP_ADD, POLY_WEAKINT, to_index_dtype(ctx, out_rngs[i]),
              to_index_dtype(ctx, off), poly_arg_none()
          );
        }
      }
      *n_in_out = n;
      return true;
    }
    return false;
  }

  /* FLIP: reverse indices along specified axes */
  case POLY_OP_FLIP: {
    if (arg.kind != POLY_ARG_INT_TUPLE) return false;
    if (arg.int_tuple.n != in_shape.ndim) return false;
    bool flipped[POLY_MAX_DIMS] = {false};
    for (int i = 0; i < arg.int_tuple.n; i++)
      flipped[i] = arg.int_tuple.vals[i] != 0;
    int n = in_shape.ndim;
    PolyUOp *zero = index_const(ctx, 0);
    for (int i = 0; i < n; i++) {
      if (i >= n_out) {
        in_rngs[i] = zero;
        continue;
      }
      if (flipped[i]) {
        PolyUOp *max_idx = index_const(ctx, in_shape.dims[i] - 1);
        in_rngs[i] = poly_uop2(
            ctx, POLY_OP_ADD, POLY_WEAKINT, max_idx, index_neg_like_tinygrad(ctx, out_rngs[i]),
            poly_arg_none()
        );
      } else {
        in_rngs[i] = out_rngs[i];
      }
    }
    *n_in_out = n;
    return true;
  }

  /* RESHAPE: flatten + decompose */
  case POLY_OP_RESHAPE: {
    if (!movement || movement->arg.kind != POLY_ARG_NONE || movement->n_src != 2) return false;
    if (!poly_reshape_indices(ctx, movement, out_rngs, n_out, in_rngs, n_in_out)) return false;
    return true;
  }

  /* PAD: offset + bounds check */
  case POLY_OP_PAD: {
    if (movement && movement->arg.kind == POLY_ARG_NONE && movement->n_src == 3) {
      int n = in_shape.ndim;
      PolyUOp *offsets[POLY_MAX_DIMS], *output_sizes[POLY_MAX_DIMS];
      if (poly_uop_as_shape(ctx, movement->src[1], offsets, POLY_MAX_DIMS) != n ||
          poly_uop_as_shape(ctx, movement->src[2], output_sizes, POLY_MAX_DIMS) != n)
        return false;
      PolyUOp *valid = NULL;
      PolyUOp *zero = index_const(ctx, 0);
      PolyUOp *invalid = poly_uop_const(ctx, poly_arg_invalid(), POLY_WEAKINT);
      PolyUOp *truev = poly_uop0(ctx, POLY_OP_CONST, POLY_BOOL, poly_arg_bool(true));

      for (int i = 0; i < n; i++) {
        if (i >= n_out) {
          in_rngs[i] = zero;
          continue;
        }
        PolyUOp *offset = offsets[i];
        PolyUOp *index = to_index_dtype(ctx, out_rngs[i]);
        PolyUOp *offset_index = to_index_dtype(ctx, offset);
        PolyUOp *input_size = poly_uop_shape_dim(ctx, movement->src[0], i);
        if (!input_size) input_size = index_const(ctx, in_shape.dims[i]);
        PolyUOp *output_size = output_sizes[i];
        int64_t offset_value = 0;
        if (output_size && poly_uop_const_i64(offset, &offset_value) == 0 && offset_value == 0 &&
            output_size == input_size) {
          /* Pinned indexing.py:137 leaves an unchanged PAD dimension alone. */
          in_rngs[i] = out_rngs[i];
          continue;
        }
        PolyUOp *shifted = poly_uop2(
            ctx, POLY_OP_ADD, POLY_WEAKINT, index, index_neg_like_tinygrad(ctx, offset_index),
            poly_arg_none()
        );
        input_size = to_index_dtype(ctx, input_size);
        PolyUOp *end =
            poly_uop2(ctx, POLY_OP_ADD, POLY_WEAKINT, input_size, offset_index, poly_arg_none());
        PolyUOp *lt_begin =
            poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, index, offset_index, poly_arg_none());
        PolyUOp *ge_begin =
            poly_uop2(ctx, POLY_OP_CMPNE, POLY_BOOL, lt_begin, truev, poly_arg_none());
        PolyUOp *lt_end = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, index, end, poly_arg_none());
        PolyUOp *dim_valid =
            poly_uop2(ctx, POLY_OP_AND, POLY_BOOL, ge_begin, lt_end, poly_arg_none());
        dim_valid = poly_graph_rewrite(ctx, dim_valid, poly_pm_pad_valid());
        if (!dim_valid) return false;
        /* Pinned indexing.py:137 keeps address and validity separable through
         * UOp.get_idx/get_valid: valid.where(r-off, Invalid). */
        in_rngs[i] = poly_uop3(
            ctx, POLY_OP_WHERE, POLY_WEAKINT, dim_valid, shifted, invalid, poly_arg_none()
        );
      }
      if (!pad_wrapper_valid_from_coords(ctx, in_rngs, n, n_out, &valid)) return false;
      if (valid_out) *valid_out = valid;
      *n_in_out = n;
      return true;
    }
    return false;
  }

  default:
    fprintf(stderr, "polygrad: indexing: unsupported movement op %s\n", poly_op_name(op));
    return false;
  }
}
/* Device metadata for BUFFERIZE creation.
 *
 * tinygrad stores BufferizeOpts(device=s.device, ...) when rangeify creates an
 * intermediate buffer. The later add_buffers pass then emits BUFFER(...,
 * DEVICE(device)) directly. Polygrad follows that flow: placement has already
 * made the graph physical, so we read the derived UOp device once with a shared
 * cache and store it in POLY_ARG_BUFFERIZE_OPTS. We do not infer placement
 * here or walk the value graph again during add_buffers.
 */
PolyUOp *poly_bufferize_device_hint(PolyCtx *ctx, PolyUOp *value, PolyMap *device_memo) {
  if (!value) return NULL;
  PolyUOp *result = poly_uop_device_uop_cached(ctx, value, device_memo);
  if (result) return result;

  /* Imported/raw PARAM storage can be supplied only through ctx residency.
   * Preserve its exact identity when available; the enum fallback is strictly
   * ordinal-zero compatibility. */
  if (ctx && (value->op == POLY_OP_BUFFER || value->op == POLY_OP_PARAM)) {
    PolyBuffer *buf = poly_buffer_get(ctx, value);
    if (buf && buf->device_uop) return buf->device_uop;
    if (buf && buf->device != POLY_DEVICE_AUTO) return poly_device_uop(ctx, buf->device);
  }
  return NULL;
}

/* Exact C spelling of pinned BufferizeOpts(device=s.device): scalar and
 * ordered tuple DEVICE identities are metadata, not a later placement guess
 * (schedule/indexing.py:70-77; schedule/rangeify.py:409-428). */
PolyArg poly_bufferize_opts_for_device(PolyUOp *device, PolyAddrSpace addrspace, bool removable) {
  if (device && device->op == POLY_OP_DEVICE && device->arg.kind == POLY_ARG_STRING_TUPLE)
    return poly_arg_bufferize_opts_tuple(
        device->arg.string_tuple.vals, device->arg.string_tuple.n, addrspace, removable
    );
  return poly_arg_bufferize_opts(
      device && device->op == POLY_OP_DEVICE && device->arg.kind == POLY_ARG_STRING
          ? device->arg.str
          : NULL,
      addrspace, removable
  );
}

static PolyUOp *rangeify_index_const(PolyCtx *ctx, int64_t value) {
  return poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(value));
}

static PolyUOp *rangeify_to_index_dtype(PolyCtx *ctx, PolyUOp *u) {
  if (!u || poly_dtype_eq(u->dtype, POLY_WEAKINT)) return u;
  return poly_uop1(ctx, POLY_OP_CAST, POLY_WEAKINT, u, poly_arg_none());
}

static bool rangeify_is_indexable_source(PolyUOp *u) {
  if (!u) return false;
  return u->op == POLY_OP_BUFFER || u->op == POLY_OP_PARAM || u->op == POLY_OP_MSTACK ||
         u->op == POLY_OP_MSELECT || u->op == POLY_OP_AFTER;
}

/* C storage for run_rangeify's Python dictionaries/lists. Grow before
 * publishing: failed realloc must leave the old owner available for cleanup. */
static bool indexing_list_grow(PolyUOp ***items, int *cap, const char *site) {
  if (*cap > INT_MAX / 2) return false;
  int next = *cap ? *cap * 2 : 4;
  if ((size_t)next > SIZE_MAX / sizeof(**items)) return false;
  PolyUOp **grown = indexing_realloc(*items, (size_t)next * sizeof(*grown), site);
  if (!grown) return false;
  *items = grown;
  *cap = next;
  return true;
}

static PolyConsumerList *consumer_list_new(void) {
  PolyConsumerList *cl = indexing_alloc(sizeof(PolyConsumerList), "consumer");
  if (!cl) return NULL;
  cl->items = NULL;
  cl->count = 0;
  cl->cap = 0;
  return cl;
}

static bool consumer_list_add(PolyConsumerList *cl, PolyUOp *consumer) {
  if (cl->count >= cl->cap && !indexing_list_grow(&cl->items, &cl->cap, "consumer")) return false;
  cl->items[cl->count++] = consumer;
  return true;
}

static void consumer_list_free(PolyConsumerList *cl) {
  free(cl->items);
  free(cl);
}

/* Consumer map */

/* Exact C port of current tinygrad schedule/indexing.py:data_srcs.
 * Shape, index, range, and ordering sources remain in the UOp graph but do
 * not carry tensor iteration ranges. */
static int rangeify_data_src_count(PolyUOp *u) {
  if (!u) return 0;
  /* A bound variable STORE carries its scalar value, not tensor data. */
  if (u->op == POLY_OP_STORE && u->n_src > 0 && poly_uop_is_variable(u->src[0])) return 0;
  switch (u->op) {
  case POLY_OP_PARAM:
  case POLY_OP_BUFFER:
  case POLY_OP_RANGE:
  case POLY_OP_SPECIAL:
    return 0;
  case POLY_OP_RESHAPE:
  case POLY_OP_EXPAND:
  case POLY_OP_PERMUTE:
  case POLY_OP_PAD:
  case POLY_OP_SHRINK:
  case POLY_OP_FLIP:
  case POLY_OP_INDEX:
  case POLY_OP_STAGE:
  case POLY_OP_REDUCE:
  case POLY_OP_AFTER:
  case POLY_OP_END:
    return u->n_src > 0 ? 1 : 0;
  default:
    return u->n_src;
  }
}

static void free_consumer_list_entry(const void *key, void *value, void *ud);

PolyMap *poly_consumer_map_build(PolyCtx *ctx, PolyUOp *sink) {
  PolyMap *cmap = poly_map_new(64);

  /* Toposort to get all UOps in dependency order */
  int n_uops;
  PolyUOp **topo = poly_toposort_ex_alloc(ctx, sink, &n_uops, NULL, false);
  if (!topo) goto fail;

  /* Ensure every UOp has an entry (even if 0 consumers) */
  for (int i = 0; i < n_uops; i++) {
    uint32_t h = poly_ptr_hash(topo[i]);
    if (!poly_map_get(cmap, h, topo[i], poly_ptr_eq)) {
      PolyConsumerList *cl = consumer_list_new();
      if (!cl) goto fail;
      poly_map_set(cmap, h, topo[i], cl, poly_ptr_eq);
    }
  }

  /* Register only data-source edges. Current tinygrad intentionally excludes
   * movement shape STACKs, INDEX coordinates, STAGE/RANGE metadata, and
   * AFTER/END ordering dependencies from range propagation. */
  for (int i = 0; i < n_uops; i++) {
    PolyUOp *u = topo[i];
    int n_data_src = rangeify_data_src_count(u);
    for (int j = 0; j < n_data_src; j++) {
      PolyUOp *src = u->src[j];
      uint32_t h = poly_ptr_hash(src);
      PolyConsumerList *cl = poly_map_get(cmap, h, src, poly_ptr_eq);
      if (cl && !consumer_list_add(cl, u)) goto fail;
    }
  }

  poly_toposort_free(topo);
  return cmap;
fail:
  poly_toposort_free(topo);
  poly_map_foreach(cmap, free_consumer_list_entry, NULL);
  poly_map_destroy(cmap);
  return NULL;
}

PolyConsumerList *poly_consumer_map_get(PolyMap *cmap, PolyUOp *u) {
  return poly_map_get(cmap, poly_ptr_hash(u), u, poly_ptr_eq);
}

/* Cleanup helper for consumer map */
static void free_consumer_list_entry(const void *key, void *value, void *ud) {
  (void)key;
  (void)ud;
  consumer_list_free((PolyConsumerList *)value);
}

/* Cleanup helper for realize map */
static void free_realize_entry(const void *key, void *value, void *ud) {
  (void)key;
  (void)ud;
  PolyRealizeInfo *ri = value;
  free(ri->axes);
  free(ri);
}

/* Cleanup helper for range map entries */
static void free_range_entry(const void *key, void *value, void *ud) {
  (void)key;
  (void)ud;
  PolyRangeEntry *re = value;
  free(re->in_rngs);
  free(re->out_rngs);
  free(re);
}

/* Cleanup helper for shape cache entries */
static void free_shape_entry(const void *key, void *value, void *ud) {
  (void)key;
  (void)ud;
  PolyShape *s = value;
  if (s->ndim > 0 && s->dims) free(s->dims);
  free(s);
}

static PolyUOp *merge_range_valids(PolyCtx *ctx, PolyUOp **valids, int n_valids) {
  if (!ctx || !valids || n_valids <= 0) return NULL;
  PolyUOp *acc = poly_uop0(ctx, POLY_OP_CONST, POLY_BOOL, poly_arg_bool(false));
  for (int i = 0; i < n_valids; i++) {
    if (!valids[i]) continue;
    acc = poly_uop2(ctx, POLY_OP_OR, POLY_BOOL, acc, valids[i], poly_arg_none());
  }
  return acc;
}

/* Per-UOp ending-ranges list (tinygrad run_rangeify parity helper). */
typedef struct {
  PolyUOp **items;
  int count;
  int cap;
} PolyEndingRanges;

static PolyEndingRanges *ending_new(void) {
  PolyEndingRanges *er = indexing_alloc(sizeof(PolyEndingRanges), "ending");
  if (!er) return NULL;
  er->count = 0;
  er->cap = 4;
  er->items = indexing_alloc(er->cap * sizeof(PolyUOp *), "ending");
  if (!er->items) {
    free(er);
    return NULL;
  }
  return er;
}

static void ending_destroy(const void *key, void *value, void *ud) {
  (void)key;
  (void)ud;
  PolyEndingRanges *er = value;
  if (!er) return;
  free(er->items);
  free(er);
}

static void ending_clear(PolyEndingRanges *er) {
  er->count = 0;
}

static bool ending_contains(PolyEndingRanges *er, PolyUOp *r) {
  for (int i = 0; i < er->count; i++)
    if (er->items[i] == r) return true;
  return false;
}

static bool ending_add(PolyEndingRanges *er, PolyUOp *r) {
  if (!r || ending_contains(er, r)) return true;
  if (er->count >= er->cap && !indexing_list_grow(&er->items, &er->cap, "ending")) return false;
  er->items[er->count++] = r;
  return true;
}

static PolyEndingRanges *ending_get_or_create(PolyMap *m, PolyUOp *key) {
  PolyEndingRanges *er = poly_map_get(m, poly_ptr_hash(key), key, poly_ptr_eq);
  if (er) return er;
  er = ending_new();
  if (!er) return NULL;
  poly_map_set(m, poly_ptr_hash(key), key, er, poly_ptr_eq);
  return er;
}

/* Indexing context */

PolyIndexingCtx *poly_indexing_ctx_new(PolyCtx *ctx) {
  PolyIndexingCtx *ictx = indexing_alloc(sizeof(PolyIndexingCtx), "context");
  if (!ictx) return NULL;
  ictx->ctx = ctx;
  ictx->consumer_map = NULL;
  ictx->realize_map = poly_map_new(32);
  ictx->non_removable = poly_map_new(16);
  ictx->range_map = poly_map_new(64);
  ictx->shape_cache = poly_map_new(64);
  ictx->next_range_id = 0;
  return ictx;
}

void poly_indexing_ctx_destroy(PolyIndexingCtx *ictx) {
  if (!ictx) return;
  if (ictx->consumer_map) {
    poly_map_foreach(ictx->consumer_map, free_consumer_list_entry, NULL);
    poly_map_destroy(ictx->consumer_map);
  }
  poly_map_foreach(ictx->range_map, free_range_entry, NULL);
  poly_map_destroy(ictx->range_map);
  poly_map_foreach(ictx->realize_map, free_realize_entry, NULL);
  poly_map_destroy(ictx->realize_map);
  poly_map_destroy(ictx->non_removable);
  poly_map_foreach(ictx->shape_cache, free_shape_entry, NULL);
  poly_map_destroy(ictx->shape_cache);
  free(ictx);
}

/* Realize map */

/*
 * Realize-point detection. Ports tinygrad's pm_generate_realize_map.
 *
 * An op is "realized" if it must become a kernel boundary — its output
 * goes to a buffer rather than being fused into a consumer's kernel.
 *
 * The pm_generate_realize_map rules below establish explicit boundaries;
 * consumer range propagation can introduce additional boundaries.
 */

/* Ops that are always contiguous / never need realization */
static bool is_always_contiguous(PolyOps op) {
  switch (op) {
  case POLY_OP_CONTIGUOUS:
  case POLY_OP_AFTER:
  case POLY_OP_BUFFER:
  case POLY_OP_CONST:
  case POLY_OP_MSELECT:
  case POLY_OP_MSTACK:
  case POLY_OP_PARAM:
  case POLY_OP_LOAD:
  case POLY_OP_CALL:
  case POLY_OP_FUNCTION:
    return true;
  default:
    return false;
  }
}

static bool realize_mark(PolyIndexingCtx *ictx, PolyUOp *u) {
  uint32_t h = poly_ptr_hash(u);
  if (poly_map_get(ictx->realize_map, h, u, poly_ptr_eq)) return true;
  PolyRealizeInfo *ri = indexing_alloc(sizeof(PolyRealizeInfo), "realize");
  if (!ri) return false;
  ri->axes = NULL;
  ri->n_axes = -1; /* -1 = all axes (not yet populated with specific list) */
  poly_map_set(ictx->realize_map, h, u, ri, poly_ptr_eq);
  return true;
}

/* Tinygrad 2026-08-22/a9069c177a9d schedule/indexing.py:realize. */
static bool poly_rangeify_realize(PolyIndexingCtx *ictx, PolyUOp *u) {
  return realize_mark(ictx, u);
}

/* Tinygrad 2026-08-22/a9069c177a9d schedule/indexing.py:realize_srcs. */
static bool poly_rangeify_realize_srcs(PolyIndexingCtx *ictx, PolyUOp *u) {
  for (int i = 0; i < u->n_src; i++) {
    PolyUOp *base = poly_uop_unsharded_base(u->src[i]);
    if (base && !is_always_contiguous(base->op) && !realize_mark(ictx, u->src[i])) return false;
  }
  return true;
}

/* Tinygrad 2026-08-22/a9069c177a9d
 * schedule/indexing.py:realize_custom_kernel_srcs. */
static bool poly_rangeify_realize_custom_kernel_srcs(PolyIndexingCtx *ictx, PolyUOp *call) {
  for (int i = 1; i < call->n_src; i++) {
    PolyUOp *src = call->src[i];
    while (src && src->op == POLY_OP_RESHAPE && src->n_src > 0)
      src = src->src[0];
    if (src && !is_always_contiguous(src->op)) {
      if (!realize_mark(ictx, src)) return false;
      poly_map_set(ictx->non_removable, poly_ptr_hash(src), src, src, poly_ptr_eq);
    }
  }
  return true;
}

/* Tinygrad 2026-08-22/a9069c177a9d
 * schedule/indexing.py:realize_store_after_src. */
static bool poly_rangeify_realize_store_after_src(
    PolyIndexingCtx *ictx,
    PolyUOp *dest,
    PolyUOp *src
) {
  PolyUOp *dest_base = poly_uop_base(dest);
  int n_topo = 0;
  PolyUOp **topo = poly_toposort_ex_alloc(ictx->ctx, src, &n_topo, NULL, false);
  if (!topo) return false;
  bool ok = true;
  for (int i = 0; i < n_topo; i++) {
    if (topo[i] == dest_base) {
      ok = realize_mark(ictx, src);
      break;
    }
  }
  poly_toposort_free(topo);
  return ok;
}

bool poly_realize_map_build(PolyIndexingCtx *ictx, PolyUOp *sink) {
  if (sink->op != POLY_OP_SINK) return true;

  /* Walk graph for the exact pm_generate_realize_map rules. Pinned tinygrad
   * does not blanket-realize SINK sources: earliest_rewrites has already made
   * the SINK reference each source's base, and explicit STORE/COPY/WAR rules
   * below define materialization boundaries. */
  int n_uops;
  PolyUOp **topo = poly_toposort_ex_alloc(ictx->ctx, sink, &n_uops, NULL, false);
  if (!topo) return false;

  for (int i = 0; i < n_uops; i++) {
    PolyUOp *u = topo[i];
    switch (u->op) {
    case POLY_OP_CONTIGUOUS:
    case POLY_OP_STORE:
      if (!poly_rangeify_realize(ictx, u)) goto fail;
      break;
    default:
      break;
    }

    if (u->op == POLY_OP_MSELECT || u->op == POLY_OP_MSTACK) {
      if (!poly_rangeify_realize_srcs(ictx, u)) goto fail;
    }

    /* Current realize_custom_kernel_srcs marks non-storage CALL inputs as
     * non-removable materializations after stripping leading RESHAPEs. */
    if (u->op == POLY_OP_CALL && u->n_src > 0 &&
        (u->src[0]->op == POLY_OP_SINK || u->src[0]->op == POLY_OP_PROGRAM)) {
      if (!poly_rangeify_realize_custom_kernel_srcs(ictx, u)) goto fail;
    }
  }

  /* Current realize_store_after_src adds only the WAR boundary. */
  for (int i = 0; i < n_uops; i++) {
    PolyUOp *u = topo[i];
    if (u->op != POLY_OP_STORE || u->n_src < 2) continue;
    if (!poly_rangeify_realize_store_after_src(ictx, u->src[0], u->src[1])) goto fail;
  }
  poly_toposort_free(topo);
  return true;
fail:
  poly_toposort_free(topo);
  return false;
}

bool poly_is_realized(PolyIndexingCtx *ictx, PolyUOp *u) {
  return poly_map_get(ictx->realize_map, poly_ptr_hash(u), u, poly_ptr_eq) != NULL;
}

/* Range map helpers */

/* Get cached shape, or compute and cache it */
static bool ictx_shape(PolyIndexingCtx *ictx, PolyUOp *u, PolyShape *out) {
  PolyShape *cached = poly_map_get(ictx->shape_cache, poly_ptr_hash(u), u, poly_ptr_eq);
  if (cached) {
    *out = *cached;
    return true;
  }
  PolyShape s = poly_uop_max_shape_cached(ictx->ctx, u);
  PolyShape *stored = indexing_alloc(sizeof(PolyShape), "shape");
  if (!stored) return false;
  *stored = (PolyShape){.ndim = s.ndim};
  if (s.ndim > 0) {
    stored->dims = indexing_alloc((size_t)s.ndim * sizeof(*stored->dims), "shape");
    if (!stored->dims) {
      free(stored);
      return false;
    }
    memcpy(stored->dims, s.dims, (size_t)s.ndim * sizeof(*stored->dims));
  }
  poly_map_set(ictx->shape_cache, poly_ptr_hash(u), u, stored, poly_ptr_eq);
  *out = *stored;
  return true;
}

/* Store a range entry in the range map */
static bool range_map_set_valid(
    PolyIndexingCtx *ictx,
    PolyUOp *u,
    PolyUOp **in_rngs,
    int n_in,
    PolyUOp **out_rngs,
    int n_out,
    PolyUOp *valid
) {
  /* Python builds the replacement tuple before assigning the dictionary
   * entry. Keep that ordering: even a failed refinement must be destructible. */
  PolyRangeEntry *re = indexing_alloc(sizeof(PolyRangeEntry), "range");
  if (!re) return false;
  re->in_rngs = n_in ? indexing_alloc((size_t)n_in * sizeof(PolyUOp *), "range") : NULL;
  re->out_rngs = n_out ? indexing_alloc((size_t)n_out * sizeof(PolyUOp *), "range") : NULL;
  if ((n_in && !re->in_rngs) || (n_out && !re->out_rngs)) {
    free_range_entry(u, re, NULL);
    return false;
  }
  if (n_in) memcpy(re->in_rngs, in_rngs, (size_t)n_in * sizeof(PolyUOp *));
  re->n_in = n_in;
  if (n_out) memcpy(re->out_rngs, out_rngs, (size_t)n_out * sizeof(PolyUOp *));
  re->n_out = n_out;
  re->valid = valid;
  PolyRangeEntry *old = poly_map_get(ictx->range_map, poly_ptr_hash(u), u, poly_ptr_eq);
  poly_map_set(ictx->range_map, poly_ptr_hash(u), u, re, poly_ptr_eq);
  if (old) free_range_entry(u, old, NULL);
  return true;
}

/* run_rangeify's realize_map[x] = axes: publish list and length together.
 * NULL axes denotes the complete [0, count) sequence, not missing storage. */
static bool realize_set_axes(PolyIndexingCtx *ictx, PolyUOp *u, const int *axes, int count) {
  int *next = count ? indexing_alloc((size_t)count * sizeof(*next), "axes") : NULL;
  if (count && !next) return false;
  for (int i = 0; i < count; i++)
    next[i] = axes ? axes[i] : i;
  if (!realize_mark(ictx, u)) {
    free(next);
    return false;
  }
  PolyRealizeInfo *ri = poly_map_get(ictx->realize_map, poly_ptr_hash(u), u, poly_ptr_eq);
  free(ri->axes);
  ri->axes = next;
  ri->n_axes = count;
  return true;
}

PolyRangeEntry *poly_range_map_get(PolyIndexingCtx *ictx, PolyUOp *u) {
  return poly_map_get(ictx->range_map, poly_ptr_hash(u), u, poly_ptr_eq);
}

/* Create a new RANGE UOp for a given dimension size. Current tinygrad's
 * IndexingContext.new_range defaults ordinary elementwise ranges to WEAK;
 * reduction construction promotes exactly those ranges to REDUCE. */
static PolyUOp *new_range_ex(PolyIndexingCtx *ictx, int64_t dim_size, PolyAxisType axis_type) {
  PolyCtx *ctx = ictx->ctx;
  /* If dim_size == 1, this range is trivially 0 */
  if (dim_size == 1) return rangeify_index_const(ctx, 0);
  PolyUOp *bound = rangeify_index_const(ctx, dim_size);
  return poly_uop1(
      ctx, POLY_OP_RANGE, POLY_WEAKINT, bound, poly_arg_range(ictx->next_range_id++, axis_type)
  );
}

/* Convenience: create WEAK range (default for elementwise iteration). */
static PolyUOp *new_range(PolyIndexingCtx *ictx, int64_t dim_size) {
  return new_range_ex(ictx, dim_size, POLY_AXIS_WEAK);
}

/* Create a RANGE with a UOp bound (for dynamic shapes).
 * If bound is CONST(1), returns CONST(0) (singleton dim).
 * If bound is CONST(N), delegates to new_range_ex.
 * A symbolic ALU value remains the RANGE bound. */
static PolyUOp *new_range_uop(PolyIndexingCtx *ictx, PolyUOp *bound, PolyAxisType axis_type) {
  PolyCtx *ctx = ictx->ctx;
  /* Tinygrad 2026-08-22/a9069c177a9d IndexingContext.new_range preserves an
   * existing RANGE: it already carries the bound, axis identity and type. */
  if (bound->op == POLY_OP_RANGE) return bound;
  if (bound->op == POLY_OP_CONST) {
    /* Static bound: use existing helper */
    return new_range_ex(ictx, bound->arg.i, axis_type);
  }
  /* Preserve a symbolic UOp bound. */
  return poly_uop1(
      ctx, POLY_OP_RANGE, POLY_WEAKINT, rangeify_to_index_dtype(ctx, bound),
      poly_arg_range(ictx->next_range_id++, axis_type)
  );
}

/* Exact C port of tinygrad/schedule/indexing.py:broadcast_rngs.  A
 * Broadcastable consumer indexes an added axis by dropping it and an expanded
 * singleton axis by zeroing it. */
static bool broadcast_rngs(
    PolyIndexingCtx *ictx,
    PolyUOp *consumer,
    PolyUOp *src,
    PolyUOp **rngs,
    int n_rngs,
    PolyUOp **out,
    int *n_out
) {
  if (!ictx || !consumer || !src || n_rngs < 0 || (n_rngs > 0 && !rngs) || !out || !n_out)
    return false;
  if (!poly_opset_has(POLY_GROUP_BROADCASTABLE, consumer->op)) {
    if (n_rngs > 0) memcpy(out, rngs, (size_t)n_rngs * sizeof(*out));
    *n_out = n_rngs;
    return true;
  }

  int src_ndim = poly_uop_ndim(ictx->ctx, src);
  int out_ndim = poly_uop_ndim(ictx->ctx, consumer);
  if (src_ndim < 0 || out_ndim < src_ndim || out_ndim > POLY_MAX_DIMS) return false;
  int nleft = out_ndim - src_ndim;
  int axes[POLY_MAX_DIMS];
  int n_axes = poly_broadcast_axes(ictx->ctx, src, consumer, axes, POLY_MAX_DIMS);
  if (n_axes < 0) return false;

  int count = 0;
  for (int j = 0; j < n_rngs; j++) {
    if (j < nleft) continue;
    bool broadcast = false;
    for (int i = 0; i < n_axes; i++)
      if (axes[i] == j) {
        broadcast = true;
        break;
      }
    out[count++] = broadcast ? rangeify_index_const(ictx->ctx, 0) : rngs[j];
  }
  *n_out = count;
  return true;
}

/* Range propagation */
/*
 * Port of tinygrad's run_rangeify() (indexing.py:158-276).
 *
 * Walks the tensor graph in reverse topological order. For each UOp:
 *  1. If realized: create fresh RANGE UOps (new kernel boundary)
 *  2. If single consumer: inherit consumer's ranges (fusion)
 *  3. If multi-consumer: merge if same, or realize if different
 *  4. Apply movement op transforms to compute input ranges
 *  5. Add reduction ranges for tensor REDUCE
 */

static bool is_tensor_reduce(PolyUOp *u) {
  return u && u->op == POLY_OP_REDUCE && u->arg.kind == POLY_ARG_REDUCE &&
         u->arg.reduce.num_axes > 0 && u->n_src == 1;
}

/* C tuple comparison for Tinygrad 2026-08-22/a9069c177a9d
 * schedule/indexing.py:274 (`rr.arg > e.arg`). Polygrad stores the trailing
 * AxisType separately from the integer axis-id/split-path tuple. */
static int range_arg_cmp(PolyArg a, PolyArg b) {
  int64_t aid = poly_range_axis_id(a), bid = poly_range_axis_id(b);
  if (aid != bid) return aid < bid ? -1 : 1;
  int an = poly_range_n_extra(a), bn = poly_range_n_extra(b);
  int n = an < bn ? an : bn;
  const int64_t *ae = poly_range_extra(a), *be = poly_range_extra(b);
  for (int i = 0; i < n; i++) {
    if (ae[i] != be[i]) return ae[i] < be[i] ? -1 : 1;
  }
  if (an != bn) return an < bn ? -1 : 1;
  PolyAxisType at = poly_range_axis_type(a), bt = poly_range_axis_type(b);
  return at == bt ? 0 : at < bt ? -1 : 1;
}

/* Tinygrad schedule/indexing.py:269-278 keeps an axis under PCONTIG>1 unless
 * one of its active RANGE keys sorts after a propagated ending-range key. */
static bool pcontig_realizes_ended_axis(
    PolyCtx *ctx,
    PolyUOp *range,
    PolyEndingRanges *ending,
    PolyUOp **scratch,
    int scratch_cap
) {
  int n_ranges = poly_uop_ranges(ctx, range, scratch, scratch_cap);
  for (int i = 0; i < n_ranges; i++)
    for (int j = 0; j < ending->count; j++)
      if (range_arg_cmp(scratch[i]->arg, ending->items[j]->arg) > 0) return true;
  return false;
}

bool poly_range_propagate(PolyIndexingCtx *ictx, PolyUOp *sink) {
  PolyCtx *ctx = ictx->ctx;
  /* Tinygrad 2026-08-22/a9069c177a9d helpers.py:PCONTIG controls partial-axis
   * realization in schedule/indexing.py. Like other Tinygrad ContextVars in
   * the C core, the process environment supplies the current scalar value. */
  int pcontig = poly_getenv_int("PCONTIG", 0);

  /* Get toposort (we'll walk in reverse) */
  int n_uops;
  PolyUOp **topo = poly_toposort_ex_alloc(ctx, sink, &n_uops, NULL, false);
  if (!topo) return false;
  PolyUOp **range_scratch = pcontig > 1 && n_uops > 0 && !range_scratch_alloc_fails()
                                ? malloc((size_t)n_uops * sizeof(*range_scratch))
                                : NULL;
  /* Python run_rangeify raises on allocation failure. Do not silently select
   * a different PCONTIG policy or publish a partially propagated range map. */
  if (pcontig > 1 && n_uops > 0 && !range_scratch) {
    poly_toposort_free(topo);
    return false;
  }

  /* Build consumer map if not already built */
  if (!ictx->consumer_map) ictx->consumer_map = poly_consumer_map_build(ctx, sink);
  if (!ictx->consumer_map) {
    free(range_scratch);
    poly_toposort_free(topo);
    return false;
  }

  /* tinygrad parity: per-node propagated ending ranges */
  PolyMap *ending_map = poly_map_new(n_uops * 2);
  bool ok = true;

  /* Reverse topological traversal */
  for (int ti = n_uops - 1; ti >= 0; ti--) {
    PolyUOp *x = topo[ti];

    /* Current gate_kernel_sink keeps nested kernels out of range propagation;
     * MSTACK/MSELECT are sink-like and do not own ranges. */
    if (x->op == POLY_OP_DEVICE || x->op == POLY_OP_UNIQUE || x->op == POLY_OP_CALL ||
        x->op == POLY_OP_FUNCTION || x->op == POLY_OP_LINEAR || x->op == POLY_OP_MSTACK ||
        x->op == POLY_OP_MSELECT)
      continue;

    /* Pinned tinygrad's run_rangeify gives AFTER no range entry. AFTER carries
     * producer ordering for src[0]; the consumer applies its own range when it
     * indexes the AFTER value. Giving AFTER an independent range indexes the
     * underlying buffer twice, so INDEX(INDEX(buf, r), r) becomes buf[2*r]. */
    if (x->op == POLY_OP_AFTER) continue;

    /* Get shape of this UOp */
    PolyShape shape;
    if (!ictx_shape(ictx, x, &shape)) {
      ok = false;
      break;
    }

    /* Collect consumer ranges: for each consumer that has ranges,
     * get the input ranges that consumer assigned to this UOp */
    PolyConsumerList *consumers = poly_consumer_map_get(ictx->consumer_map, x);
    int consumer_cap = consumers ? consumers->count : 0;
    PolyUOp *(*consumer_rngs_buf)[POLY_MAX_DIMS] =
        consumer_cap > 0 && !range_scratch_alloc_fails()
            ? calloc((size_t)consumer_cap, sizeof(*consumer_rngs_buf))
            : NULL;
    int *consumer_rngs_lens = consumer_cap > 0 && !range_scratch_alloc_fails()
                                  ? calloc((size_t)consumer_cap, sizeof(*consumer_rngs_lens))
                                  : NULL;
    if (consumer_cap > 0 && (!consumer_rngs_buf || !consumer_rngs_lens)) {
      free(consumer_rngs_buf);
      free(consumer_rngs_lens);
      ok = false;
      break;
    }
    int n_consumer_rngs = 0;

    PolyEndingRanges *broadcast_ending = NULL;
    PolyEndingRanges *ending = ending_get_or_create(ending_map, x);
    if (!ending) goto node_fail;
    ending_clear(ending);
    if (consumers) {
      for (int ci = 0; ci < consumers->count; ci++) {
        PolyUOp *consumer = consumers->items[ci];
        PolyRangeEntry *cre = poly_range_map_get(ictx, consumer);
        if (!cre) continue;
        int n_broadcast = 0;
        if (!broadcast_rngs(
                ictx, consumer, x, cre->in_rngs, cre->n_in, consumer_rngs_buf[n_consumer_rngs],
                &n_broadcast
            )) {
          if (!realize_mark(ictx, x)) goto node_fail;
          continue;
        }
        consumer_rngs_lens[n_consumer_rngs] = n_broadcast;
        n_consumer_rngs++;
      }
      /* ending_ranges[x] = concat(ending_ranges[consumer]) */
      for (int ci = 0; ci < consumers->count; ci++) {
        PolyUOp *consumer = consumers->items[ci];
        PolyEndingRanges *ec =
            poly_map_get(ending_map, poly_ptr_hash(consumer), consumer, poly_ptr_eq);
        if (!ec) continue;
        for (int ei = 0; ei < ec->count; ei++)
          if (!ending_add(ending, ec->items[ei])) goto node_fail;
      }
    }

    /* Current broadcast_ending_ranges records consumer axes that x broadcasts
     * over. REDUCE sees them before fusion; every op propagates them after its
     * own ended-range decision. */
    broadcast_ending = ending_new();
    if (!broadcast_ending) goto node_fail;
    if (consumers) {
      for (int ci = 0; ci < consumers->count; ci++) {
        PolyUOp *consumer = consumers->items[ci];
        PolyRangeEntry *cre = poly_range_map_get(ictx, consumer);
        if (!cre || !poly_opset_has(POLY_GROUP_BROADCASTABLE, consumer->op)) continue;
        int axes[POLY_MAX_DIMS];
        int n_axes = poly_broadcast_axes(ctx, x, consumer, axes, POLY_MAX_DIMS);
        if (n_axes < 0) continue;
        for (int ai = 0; ai < n_axes; ai++) {
          int axis = axes[ai];
          if (axis < 0 || axis >= cre->n_in) continue;
          PolyUOp *ranges[POLY_MAX_DIMS];
          int n_ranges = poly_uop_ranges(ctx, cre->in_rngs[axis], ranges, POLY_MAX_DIMS);
          for (int ri = 0; ri < n_ranges; ri++)
            if (!ending_add(broadcast_ending, ranges[ri])) goto node_fail;
        }
      }
    }
    if (is_tensor_reduce(x))
      for (int i = 0; i < broadcast_ending->count; i++)
        if (!ending_add(ending, broadcast_ending->items[i])) goto node_fail;
    /* Determine output ranges for x */
    PolyUOp *out_rngs[POLY_MAX_DIMS];
    memset(out_rngs, 0, sizeof(out_rngs));
    int n_out = 0;

    if (poly_is_realized(ictx, x)) {
      /* Case 1: Realized — create fresh ranges */
      if (shape.ndim <= 0) {
        /* Scalar or no shape — no ranges needed */
        n_out = 0;
      } else {
        for (int i = 0; i < shape.ndim; i++) {
          /* tinygrad schedule/indexing.py:184-187 creates realized ranges from
           * every x.shape element, not from the backing buffer's max allocation.
           * Keep the AFTER/STORE binding here so schedule construction can collect its concrete
           * value; kernel extraction later lowers it to an ALU PARAM. */
          PolyUOp *dim = poly_uop_shape_dim(ctx, x, i);
          out_rngs[i] =
              dim ? new_range_uop(ictx, dim, POLY_AXIS_WEAK) : new_range(ictx, shape.dims[i]);
        }
        n_out = shape.ndim;
      }

      /* Update realize map with specific axes */
      if (n_out > 0 && !realize_set_axes(ictx, x, NULL, n_out)) goto node_fail;
      ending_clear(ending);
    } else if (n_consumer_rngs == 0) {
      /* Case 2: No consumers with ranges — skip */
      free(consumer_rngs_buf);
      free(consumer_rngs_lens);
      ending_destroy(NULL, broadcast_ending, NULL);
      continue;
    } else if (n_consumer_rngs == 1) {
      /* Case 3: Single consumer — inherit (fusion!) */
      memcpy(out_rngs, consumer_rngs_buf[0], consumer_rngs_lens[0] * sizeof(PolyUOp *));
      n_out = consumer_rngs_lens[0];
    } else {
      /* Case 4: Multiple consumers — tinygrad compares local idx and valid
       * separately here. With the default PCONTIG=0 setting, we only keep the
       * merged path when all local idx expressions match across every axis;
       * otherwise we realize. */
      bool same_shape = true;
      int ref_len = consumer_rngs_lens[0];
      for (int ci = 1; ci < n_consumer_rngs; ci++) {
        if (consumer_rngs_lens[ci] != ref_len) {
          same_shape = false;
          break;
        }
      }

      int realize_axes[POLY_MAX_DIMS];
      int n_realize_axes = 0;
      if (same_shape) {
        bool axis_same[POLY_MAX_DIMS];
        bool all_all_same = true;
        for (int d = 0; d < ref_len; d++) {
          PolyUOp *base_idx = poly_uop_get_idx(ctx, consumer_rngs_buf[0][d]);
          axis_same[d] = true;
          for (int ci = 1; ci < n_consumer_rngs; ci++) {
            if (poly_uop_get_idx(ctx, consumer_rngs_buf[ci][d]) != base_idx) {
              axis_same[d] = false;
              all_all_same = false;
              break;
            }
          }
        }

        for (int d = 0; d < ref_len; d++) {
          /* Tinygrad schedule/indexing.py:257-264 preserves a matching axis
           * under PCONTIG, even when another axis forces materialization. */
          if (!all_all_same && !(pcontig && axis_same[d])) {
            PolyUOp *dim = d < shape.ndim ? poly_uop_shape_dim(ctx, x, d) : NULL;
            if (!dim) {
              same_shape = false;
              break;
            }
            out_rngs[d] = new_range_uop(ictx, dim, POLY_AXIS_WEAK);
            if (!out_rngs[d]) {
              same_shape = false;
              break;
            }
            realize_axes[n_realize_axes++] = d;
            continue;
          }

          bool all_same_range_ptr = true;
          for (int ci = 1; ci < n_consumer_rngs; ci++) {
            if (consumer_rngs_buf[ci][d] != consumer_rngs_buf[0][d]) {
              all_same_range_ptr = false;
              break;
            }
          }
          if (all_same_range_ptr) {
            /* Tinygrad keeps the inherited range object when every consumer
             * already shares the exact same range. Reuse that pointer instead
             * of rebuilding an equivalent WHERE wrapper. */
            out_rngs[d] = consumer_rngs_buf[0][d];
            continue;
          }

          PolyUOp **valids = indexing_alloc((size_t)n_consumer_rngs * sizeof(*valids), "valids");
          if (!valids) goto node_fail;
          for (int ci = 0; ci < n_consumer_rngs; ci++)
            valids[ci] = poly_uop_get_valid(ctx, consumer_rngs_buf[ci][d]);
          PolyUOp *merged_valid = merge_range_valids(ctx, valids, n_consumer_rngs);
          free(valids);
          PolyUOp *merged_idx = poly_uop_get_idx(ctx, consumer_rngs_buf[0][d]);
          if (merged_valid->op == POLY_OP_CONST && merged_valid->arg.kind == POLY_ARG_BOOL &&
              merged_valid->arg.b) {
            out_rngs[d] = merged_idx;
            continue;
          }
          PolyUOp *invalid = poly_uop_const(ctx, poly_arg_invalid(), merged_idx->dtype);
          PolyUOp *merged_range = poly_uop3(
              ctx, POLY_OP_WHERE, merged_idx->dtype, merged_valid, merged_idx, invalid,
              poly_arg_none()
          );
          /* Pinned tinygrad/schedule/indexing.py:214-215 rewrites the full
           * merged valid.where(local_idx, Invalid) with symbolic. This is
           * distinct from PAD's local-valid rewrite: it also canonicalizes
           * the inherited local index (for example RANGE + 0 -> RANGE). */
          out_rngs[d] =
              merged_range ? poly_graph_rewrite(ctx, merged_range, poly_symbolic()) : NULL;
          if (!out_rngs[d]) {
            same_shape = false;
            break;
          }
        }
        if (same_shape) {
          n_out = ref_len;
          if (n_realize_axes > 0 && !realize_set_axes(ictx, x, realize_axes, n_realize_axes))
            goto node_fail;
        }
      }

      if (!same_shape) {
        /* Consumers disagree — must realize this op */
        if (shape.ndim <= 0) {
          /* Scalar with disagreeing consumers: propagate empty ranges
           * so sources (e.g. tensor REDUCE below) still get range entries */
          n_out = 0;
        } else {
          if (!realize_mark(ictx, x)) goto node_fail;
          for (int i = 0; i < shape.ndim; i++) {
            PolyUOp *dim = poly_uop_shape_dim(ctx, x, i);
            out_rngs[i] =
                dim ? new_range_uop(ictx, dim, POLY_AXIS_WEAK) : new_range(ictx, shape.dims[i]);
          }
          n_out = shape.ndim;

          if (n_out > 0 && !realize_set_axes(ictx, x, NULL, n_out)) goto node_fail;
        }
      }
    }

    free(consumer_rngs_buf);
    free(consumer_rngs_lens);
    consumer_rngs_buf = NULL;
    consumer_rngs_lens = NULL;

    /* tinygrad parity: if ended ranges flow into elementwise/reduce, realize axes */
    if (ending->count > 0 &&
        (poly_opset_has(POLY_GROUP_ELEMENTWISE, x->op) || is_tensor_reduce(x))) {
      int realize_axes[POLY_MAX_DIMS];
      int n_realize_axes = 0;
      PolyRealizeInfo *ri = poly_map_get(ictx->realize_map, poly_ptr_hash(x), x, poly_ptr_eq);
      if (ri && ri->axes && ri->n_axes > 0) {
        for (int i = 0; i < ri->n_axes && n_realize_axes < POLY_MAX_DIMS; i++) {
          int ax = ri->axes[i];
          bool seen = false;
          for (int j = 0; j < n_realize_axes; j++) {
            if (realize_axes[j] == ax) {
              seen = true;
              break;
            }
          }
          if (!seen) realize_axes[n_realize_axes++] = ax;
        }
      }

      for (int i = 0; i < n_out && n_realize_axes < POLY_MAX_DIMS; i++) {
        bool seen = false;
        for (int j = 0; j < n_realize_axes; j++) {
          if (realize_axes[j] == i) {
            seen = true;
            break;
          }
        }
        if (!seen && (pcontig <= 1 ||
                      pcontig_realizes_ended_axis(ctx, out_rngs[i], ending, range_scratch, n_uops)))
          realize_axes[n_realize_axes++] = i;
      }

      ending_clear(ending);
      if (n_realize_axes > 0) {
        if (!realize_set_axes(ictx, x, realize_axes, n_realize_axes)) goto node_fail;
        for (int i = 0; i < n_realize_axes; i++) {
          int ax = realize_axes[i];
          if (ax >= 0 && ax < n_out && ax < shape.ndim) {
            PolyUOp *dim = poly_uop_shape_dim(ctx, x, ax);
            out_rngs[ax] =
                dim ? new_range_uop(ictx, dim, POLY_AXIS_WEAK) : new_range(ictx, shape.dims[ax]);
          }
        }
      }
    }

    for (int i = 0; i < broadcast_ending->count; i++)
      if (!ending_add(ending, broadcast_ending->items[i])) goto node_fail;
    ending_destroy(NULL, broadcast_ending, NULL);
    broadcast_ending = NULL;

    if (n_out == 0 && !poly_is_realized(ictx, x) && x->n_src == 0) continue;

    /* Compute input ranges (what this op's sources see) */
    PolyUOp *in_rngs[POLY_MAX_DIMS];
    memset(in_rngs, 0, sizeof(in_rngs));
    int n_in = n_out;
    memcpy(in_rngs, out_rngs, n_out * sizeof(PolyUOp *));

    /* Apply movement op transforms */
    PolyUOp *valid_mask = NULL;
    if (poly_opset_has(POLY_GROUP_MOVEMENT, x->op) && x->n_src > 0) {
      PolyShape src_shape;
      if (!ictx_shape(ictx, x->src[0], &src_shape)) goto node_fail;
      if (src_shape.ndim >= 0) {
        PolyUOp *transformed[POLY_MAX_DIMS];
        int n_transformed = 0;
        if (poly_apply_movement_op(
                ctx, x, x->op, src_shape, x->arg, out_rngs, n_out, transformed, &n_transformed,
                &valid_mask
            )) {
          memcpy(in_rngs, transformed, n_transformed * sizeof(PolyUOp *));
          n_in = n_transformed;
        }
      }
    }

    /* Current data STACK adds one leading selector dimension to its output;
     * every source sees only the trailing ranges (indexing.py:295-296). */
    if (x->op == POLY_OP_STACK) {
      if (n_out > 0) {
        for (int i = 1; i < n_out; i++)
          in_rngs[i - 1] = out_rngs[i];
        n_in = n_out - 1;
      } else {
        n_in = 0;
      }
    }

    /* Current EXPAND injects only its leading marg axes. Those ranges end at
     * this movement unless the public shape itself contains a RANGE
     * (schedule/indexing.py:297-300). */
    if (x->op == POLY_OP_EXPAND) {
      bool mark_prefix = true;
      int x_ndim = poly_uop_ndim(ctx, x);
      for (int i = 0; i < x_ndim; i++) {
        PolyUOp *dim = poly_uop_shape_dim(ctx, x, i);
        if (dim && dim->op == POLY_OP_RANGE) {
          mark_prefix = false;
          break;
        }
      }
      PolyUOp *marg[POLY_MAX_DIMS];
      int n_marg = x->n_src == 2 ? poly_uop_as_shape(ctx, x->src[1], marg, POLY_MAX_DIMS) : -1;
      if (mark_prefix && n_marg >= 0 && n_marg <= n_out) {
        for (int i = 0; i < n_marg; i++) {
          PolyUOp *ranges[POLY_MAX_DIMS];
          int n_ranges = poly_uop_ranges(ctx, out_rngs[i], ranges, POLY_MAX_DIMS);
          for (int j = 0; j < n_ranges; j++)
            if (!ending_add(ending, ranges[j])) goto node_fail;
        }
      }
    }

    /* Current tensor REDUCE creates new ranges for its reduced source prefix
     * (tinygrad/schedule/indexing.py:303-304). */
    if (is_tensor_reduce(x)) {
      PolyShape src_shape;
      if (!ictx_shape(ictx, x->src[0], &src_shape)) goto node_fail;
      if (src_shape.ndim > 0) {
        int n_axes = x->arg.reduce.num_axes;
        PolyUOp *new_in[POLY_MAX_DIMS];
        for (int i = 0; i < src_shape.ndim; i++) {
          if (i < n_axes) {
            new_in[i] = new_range_ex(ictx, src_shape.dims[i], POLY_AXIS_REDUCE);
          } else if (i - n_axes < n_out) {
            new_in[i] = out_rngs[i - n_axes];
          } else {
            new_in[i] = rangeify_index_const(ctx, 0);
          }
        }
        memcpy(in_rngs, new_in, src_shape.ndim * sizeof(PolyUOp *));
        n_in = src_shape.ndim;
      }
    }

    /* Store in range map (with valid mask for PAD ops) */
    if (!range_map_set_valid(ictx, x, in_rngs, n_in, out_rngs, n_out, valid_mask)) goto node_fail;
    continue;

  node_fail:
    free(consumer_rngs_buf);
    free(consumer_rngs_lens);
    ending_destroy(NULL, broadcast_ending, NULL);
    ok = false;
    break;
  }

  poly_map_foreach(ending_map, ending_destroy, NULL);
  poly_map_destroy(ending_map);
  free(range_scratch);
  poly_toposort_free(topo);
  return ok;
}

/* Apply rangeify graph rewrite */
/*
 * Transforms tensor-level IR to kernel-level IR using range annotations.
 *
 * Walks toposort order, applying:
 *  1. Movement ops (RESHAPE, EXPAND, PERMUTE, SHRINK, FLIP) → src[0]
 *  2. PAD → WHERE(valid_mask, src[0], 0.0)
 *  3. tensor REDUCE(op, axes) → REDUCE(value, reduce_ranges...) with arg = reduce_op
 *  4. Realized computed ops → BUFFERIZE(op, out_ranges...)
 *  5. Everything else → pass through (rebuild if sources changed)
 */

static PolyUOp *rmap_get(PolyMap *m, PolyUOp *key) {
  return poly_map_get(m, poly_ptr_hash(key), key, poly_ptr_eq);
}

static void rmap_set(PolyMap *m, PolyUOp *key, PolyUOp *val) {
  poly_map_set(m, poly_ptr_hash(key), key, val, poly_ptr_eq);
}

/* Tinygrad 2026-08-22/a9069c177a9d
 * schedule/indexing.py:create_bufferize_and_index_srcs. `rmap` supplies the
 * bottom-up child replacements that Python's graph_rewrite owns implicitly. */
static void poly_create_bufferize_and_index_srcs(
    PolyIndexingCtx *ictx,
    PolyUOp *u,
    PolyMap *rmap,
    PolyMap *device_memo,
    PolyUOp **new_src,
    bool *src_changed
) {
  PolyCtx *ctx = ictx->ctx;
  PolyRangeEntry *u_re = poly_range_map_get(ictx, u);
  int n_data_src = rangeify_data_src_count(u);
  *src_changed = false;

  for (int j = 0; j < u->n_src; j++) {
    PolyUOp *orig = u->src[j];
    PolyUOp *mapped = rmap_get(rmap, orig);
    new_src[j] = mapped ? mapped : orig;
    if (new_src[j] != orig) *src_changed = true;

    PolyUOp *src_rngs[POLY_MAX_DIMS];
    int n_src_rngs = 0;
    if (u_re && !broadcast_rngs(ictx, u, orig, u_re->in_rngs, u_re->n_in, src_rngs, &n_src_rngs))
      continue;

    if (rangeify_is_indexable_source(orig)) {
      if (u_re && j < n_data_src) {
        PolyUOp *idx_src[POLY_MAX_DIMS + 1];
        idx_src[0] = new_src[j];
        for (int d = 0; d < n_src_rngs; d++)
          idx_src[d + 1] = src_rngs[d];
        new_src[j] = poly_uop(
            ctx, POLY_OP_INDEX, new_src[j]->dtype, idx_src, n_src_rngs + 1, poly_arg_none()
        );
        *src_changed = true;
      }
      continue;
    }

    PolyRealizeInfo *ri = poly_map_get(ictx->realize_map, poly_ptr_hash(orig), orig, poly_ptr_eq);
    PolyRangeEntry *src_re = poly_range_map_get(ictx, orig);
    if (!ri || !src_re || ri->n_axes < 0 || (ri->n_axes > 0 && !ri->axes)) continue;

    PolyUOp *closed[POLY_MAX_DIMS];
    int n_closed = 0;
    for (int ai = 0; ai < ri->n_axes; ai++) {
      int axis = ri->axes[ai];
      if (axis >= 0 && axis < src_re->n_out) closed[n_closed++] = src_re->out_rngs[axis];
    }

    if (orig->op == POLY_OP_STORE) {
      PolyUOp *end_src[POLY_MAX_DIMS + 1];
      int n_end = 1;
      end_src[0] = new_src[j];
      for (int d = 0; d < n_closed; d++)
        if (closed[d]->op == POLY_OP_RANGE) end_src[n_end++] = closed[d];
      if (n_end > 1)
        new_src[j] = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, n_end, poly_arg_none());
      *src_changed = true;
      continue;
    }

    PolyUOp *stage_src[POLY_MAX_DIMS + 1];
    stage_src[0] = new_src[j];
    for (int d = 0; d < n_closed; d++)
      stage_src[d + 1] = closed[d];
    bool removable = !is_always_contiguous(orig->op) &&
                     !poly_map_get(ictx->non_removable, poly_ptr_hash(orig), orig, poly_ptr_eq);
    PolyAddrSpace addrspace = src_re->n_out == ri->n_axes ? POLY_ADDR_GLOBAL : POLY_ADDR_LOCAL;
    PolyUOp *device = poly_bufferize_device_hint(ctx, orig, device_memo);
    PolyUOp *stage = poly_uop(
        ctx, POLY_OP_STAGE, orig->dtype, stage_src, n_closed + 1,
        poly_bufferize_opts_for_device(device, addrspace, removable)
    );
    if (!stage) continue;
    new_src[j] = stage;

    if (u_re) {
      PolyUOp *idx_src[POLY_MAX_DIMS + 1];
      int n_idx = 0;
      idx_src[n_idx++] = stage;
      for (int ai = 0; ai < ri->n_axes; ai++) {
        int axis = ri->axes[ai];
        if (axis >= 0 && axis < n_src_rngs) idx_src[n_idx++] = src_rngs[axis];
      }
      new_src[j] = poly_uop(ctx, POLY_OP_INDEX, stage->dtype, idx_src, n_idx, poly_arg_none());
    }
    *src_changed = true;
  }
}

/* Tinygrad 2026-08-22/a9069c177a9d
 * schedule/indexing.py:convert_pad_to_where_to_keep_behavior_local. */
static PolyUOp *poly_convert_pad_to_where_to_keep_behavior_local(
    PolyIndexingCtx *ictx,
    PolyUOp *u,
    PolyUOp **new_src
) {
  PolyRangeEntry *re = poly_range_map_get(ictx, u);
  if (!re || u->n_src == 0) return NULL;
  PolyUOp *valid = re->valid;
  if (!valid) valid = poly_uop0(ictx->ctx, POLY_OP_CONST, POLY_BOOL, poly_arg_bool(true));
  PolyUOp *zero = poly_const_like_int(ictx->ctx, new_src[0], 0);
  return poly_uop3(ictx->ctx, POLY_OP_WHERE, u->dtype, valid, new_src[0], zero, poly_arg_none());
}

/* Tinygrad 2026-08-22/a9069c177a9d
 * schedule/indexing.py:convert_reduce_to_reduce_with_ranges. */
static PolyUOp *poly_convert_reduce_to_reduce_with_ranges(
    PolyIndexingCtx *ictx,
    PolyUOp *u,
    PolyUOp **new_src
) {
  if (u->arg.kind != POLY_ARG_REDUCE || u->arg.reduce.num_axes == 0 || u->n_src == 0) return NULL;
  PolyRangeEntry *re = poly_range_map_get(ictx, u);
  if (!re) return NULL;
  PolyUOp *src[POLY_MAX_DIMS + 1] = {new_src[0]};
  int n_src = 1;
  for (int i = 0; i < u->arg.reduce.num_axes && i < re->n_in; i++)
    if (re->in_rngs[i]->op == POLY_OP_RANGE) src[n_src++] = re->in_rngs[i];
  return poly_uop(
      ictx->ctx, POLY_OP_REDUCE, u->dtype, src, n_src, poly_arg_reduce(u->arg.reduce.op, 0)
  );
}

/* Tinygrad 2026-08-22/a9069c177a9d
 * schedule/indexing.py:convert_stack_to_where. */
static PolyUOp *poly_convert_stack_to_where(PolyIndexingCtx *ictx, PolyUOp *u, PolyUOp **new_src) {
  PolyRangeEntry *re = poly_range_map_get(ictx, u);
  if (!re || re->n_out == 0 || poly_dtype_eq(u->dtype, POLY_VOID) || u->n_src == 0) return NULL;
  PolyUOp *selector = re->out_rngs[0];
  PolyUOp *ret = new_src[u->n_src - 1];
  for (int i = u->n_src - 2; i >= 0; i--) {
    PolyUOp *choice = rangeify_index_const(ictx->ctx, i);
    PolyUOp *cond =
        poly_uop2(ictx->ctx, POLY_OP_CMPEQ, POLY_BOOL, selector, choice, poly_arg_none());
    ret = poly_uop3(ictx->ctx, POLY_OP_WHERE, u->dtype, cond, new_src[i], ret, poly_arg_none());
  }
  return ret;
}

/* Tinygrad 2026-08-22/a9069c177a9d
 * schedule/indexing.py:remove_movement_op_after_rangeify. */
static PolyUOp *poly_remove_movement_op_after_rangeify(
    PolyIndexingCtx *ictx,
    PolyUOp *u,
    PolyUOp **new_src
) {
  if (!poly_opset_has(POLY_GROUP_MOVEMENT, u->op) || u->n_src == 0) return NULL;
  return poly_range_map_get(ictx, u) || new_src[0]->op == POLY_OP_INDEX ? new_src[0] : NULL;
}

/* Tinygrad 2026-08-22/a9069c177a9d
 * schedule/indexing.py:create_bufferize_and_index_based_on_ranges. */
static PolyUOp *poly_create_bufferize_and_index_based_on_ranges(
    PolyIndexingCtx *ictx,
    PolyUOp *u,
    PolyUOp **new_src,
    bool src_changed
) {
  return src_changed ? poly_uop(ictx->ctx, u->op, u->dtype, new_src, u->n_src, u->arg) : NULL;
}

/* Current Tinygrad schedule/indexing.py:pm_apply_rangeify.  The caller owns
 * IndexingContext construction and range propagation for focused probes. */
PolyUOp *poly_apply_rangeify(PolyIndexingCtx *ictx, PolyUOp *sink) {
  PolyCtx *ctx = ictx->ctx;
  int n_uops = 0;
  PolyUOp **topo = poly_toposort_ex_alloc(ctx, sink, &n_uops, NULL, false);
  if (!topo) return NULL;

  PolyMap *rmap = poly_map_new(n_uops * 2);
  PolyMap *device_memo = poly_map_new(n_uops * 2);
  bool ok = true;

  for (int i = 0; i < n_uops; i++) {
    PolyUOp *u = topo[i];
    bool src_changed = false;
    PolyUOp *new_src_buf[16] = {0};
    PolyUOp **new_src =
        u->n_src > (int)(sizeof(new_src_buf) / sizeof(new_src_buf[0]))
            ? (indexing_alloc_fails("rewrite") ? NULL : calloc((size_t)u->n_src, sizeof(*new_src)))
            : new_src_buf;
    if (!new_src) {
      ok = false;
      break;
    }

    poly_create_bufferize_and_index_srcs(ictx, u, rmap, device_memo, new_src, &src_changed);

    PolyUOp *result = NULL;
    switch (u->op) {
    case POLY_OP_PAD:
      result = poly_convert_pad_to_where_to_keep_behavior_local(ictx, u, new_src);
      break;

    case POLY_OP_STACK:
      result = poly_convert_stack_to_where(ictx, u, new_src);
      break;

    case POLY_OP_REDUCE:
      result = poly_convert_reduce_to_reduce_with_ranges(ictx, u, new_src);
      break;

    default:
      break;
    }

    if (!result) result = poly_remove_movement_op_after_rangeify(ictx, u, new_src);
    if (!result)
      result = poly_create_bufferize_and_index_based_on_ranges(ictx, u, new_src, src_changed);
    if (result && result != u) rmap_set(rmap, u, result);
    if (new_src != new_src_buf) free(new_src);
  }

  PolyUOp *new_sink = rmap_get(rmap, sink);
  poly_map_destroy(device_memo);
  poly_map_destroy(rmap);
  poly_toposort_free(topo);
  return ok ? (new_sink ? new_sink : sink) : NULL;
}

static PolyUOp *rule_fix_deviceless(PolyCtx *ctx, PolyUOp *root, const PolyBindings *bindings) {
  (void)bindings;
  if (!root || root->op != POLY_OP_STAGE || root->arg.kind != POLY_ARG_BUFFERIZE_OPTS ||
      poly_bufferize_arg_addrspace(root->arg) != POLY_ADDR_GLOBAL ||
      root->arg.bufferize_opts.device_is_int || poly_bufferize_arg_device(root->arg) ||
      poly_bufferize_arg_device_is_tuple(root->arg))
    return NULL;
  PolyUOp *device = poly_graph_rewrite_userctx();
  if (!device || device->op != POLY_OP_DEVICE) return NULL;
  PolyArg arg = poly_bufferize_opts_for_device(
      device, POLY_ADDR_GLOBAL, poly_bufferize_arg_removable(root->arg)
  );
  return poly_uop_tagged_arg(
      ctx, root->op, root->dtype, root->src, root->n_src, arg, root->tag, root->tag_arg
  );
}

static PolyPatternMatcher *poly_pm_fix_deviceless(void) {
  static _Thread_local PolyPatternMatcher *pm = NULL;
  if (pm) return pm;
  PolyRule rules[] = {
      {poly_upat_allow_any_len(poly_upat_op(POLY_OP_STAGE, NULL, 0, "stage")), rule_fix_deviceless},
  };
  pm = poly_pm_thread_cache(poly_pm_new(rules, 1));
  return pm;
}

/* Tinygrad 2026-08-22/a9069c177a9d schedule/indexing.py:render_ranges.
 * C prints the same input/output range boundary for rangeify diagnostics. */
static void poly_render_ranges(
    FILE *fp,
    const PolyRangeEntry *entry,
    const PolyRealizeInfo *realized
) {
  if (!fp || !entry) return;
  int n = entry->n_in > entry->n_out ? entry->n_in : entry->n_out;
  for (int i = 0; i < n; i++) {
    bool is_realized = false;
    if (realized && realized->axes)
      for (int j = 0; j < realized->n_axes; j++)
        is_realized |= realized->axes[j] == i;
    char *in = i < entry->n_in ? poly_uop_str(entry->in_rngs[i]) : NULL;
    char *out = i < entry->n_out ? poly_uop_str(entry->out_rngs[i]) : NULL;
    fprintf(fp, "%s[%s", is_realized ? "*" : "", in ? in : "-");
    if ((!in && out) || (in && out && entry->in_rngs[i] != entry->out_rngs[i]))
      fprintf(fp, " -> %s", out ? out : "-");
    fputc(']', fp);
    free(in);
    free(out);
  }
}

/* Current Tinygrad schedule/indexing.py:run_rangeify owns realization
 * discovery, consumer/range propagation, pm_apply_rangeify, and the final
 * pm_fix_deviceless rewrite. */
PolyUOp *poly_run_rangeify(PolyCtx *ctx, PolyUOp *sink, bool debug) {
  (void)debug;
  if (!ctx || !sink) return NULL;
  PolyIndexingCtx *ictx = poly_indexing_ctx_new(ctx);
  if (!ictx) return NULL;
  if (!poly_realize_map_build(ictx, sink) || !poly_range_propagate(ictx, sink)) {
    poly_indexing_ctx_destroy(ictx);
    return NULL;
  }
  PolyUOp *rangeified = poly_apply_rangeify(ictx, sink);
  if (debug) {
    int n_topo = 0;
    PolyUOp **topo = poly_toposort_ex_alloc(ctx, sink, &n_topo, NULL, false);
    for (int i = 0; topo && i < n_topo; i++) {
      PolyRangeEntry *entry = poly_range_map_get(ictx, topo[i]);
      if (!entry) continue;
      PolyRealizeInfo *realized =
          poly_map_get(ictx->realize_map, poly_ptr_hash(topo[i]), topo[i], poly_ptr_eq);
      fprintf(stderr, "%s: ", poly_op_name(topo[i]->op));
      poly_render_ranges(stderr, entry, realized);
      fputc('\n', stderr);
    }
    poly_toposort_free(topo);
  }
  PolyUOp *device = poly_uop_device_uop_cached(ctx, sink, NULL);
  if (rangeified && device)
    rangeified =
        poly_graph_rewrite_ctx_ex2(ctx, rangeified, poly_pm_fix_deviceless(), device, false, false);
  poly_indexing_ctx_destroy(ictx);
  return rangeified;
}

/* Current Tinygrad schedule/rangeify.py:426-434. */
