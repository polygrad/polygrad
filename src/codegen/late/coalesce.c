/* Tinygrad 2026-08-22/a9069c177a9d codegen/late/coalesce.py. */

#include "codegen/late/coalesce.h"
#include "codegen/codegen.h"
#include "uop/ops.h"
#include "uop/symbolic.h"
#include "utils.h"

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  PolyUOp *u;
  PolyUOp *buf;
  PolyUOp *base;
  PolyUOp *valid;
  PolyDType idx_dtype;
  PolyOps op;
  int base_kind; /* 0=UOp, 1=CONST, 2=Invalid */
  int64_t offset;
} MemoryCoalescingRecord;

/* Implements Tinygrad's defaultdict key and sorted offset traversal. */
static int memory_coalescing_record_cmp(const void *ap, const void *bp) {
  const MemoryCoalescingRecord *a = ap, *b = bp;
#define CMP_FIELD(x, y)                                                                            \
  do {                                                                                             \
    if ((x) < (y)) return -1;                                                                      \
    if ((x) > (y)) return 1;                                                                       \
  } while (0)
  CMP_FIELD(a->op, b->op);
  CMP_FIELD((uintptr_t)a->buf, (uintptr_t)b->buf);
  CMP_FIELD(a->base_kind, b->base_kind);
  CMP_FIELD((uintptr_t)a->base, (uintptr_t)b->base);
  CMP_FIELD((uintptr_t)a->valid, (uintptr_t)b->valid);
  CMP_FIELD(a->offset, b->offset);
  CMP_FIELD((uintptr_t)a->u, (uintptr_t)b->u);
#undef CMP_FIELD
  return 0;
}

static bool memory_coalescing_same_key(
    const MemoryCoalescingRecord *a,
    const MemoryCoalescingRecord *b
) {
  return a->op == b->op && a->buf == b->buf && a->base_kind == b->base_kind && a->base == b->base &&
         a->valid == b->valid;
}

/* Current coalesce.py's ordinary scalar storage classes. */
static bool memory_coalescing_dtype(PolyDType dtype) {
  PolyDType scalar = dtype;
  return poly_dtype_eq(scalar, POLY_FLOAT32) || poly_dtype_eq(scalar, POLY_FLOAT16) ||
         poly_dtype_eq(scalar, POLY_INT32) || poly_dtype_eq(scalar, POLY_UINT32) ||
         poly_dtype_is_fp8(scalar);
}

static bool const_true(PolyUOp *u) {
  return u && u->op == POLY_OP_CONST && u->arg.kind == POLY_ARG_BOOL && u->arg.b;
}

/* Rebuild base+offset without validity, as coalesce.py does before divides. */
static PolyUOp *memory_coalescing_offset(
    PolyCtx *ctx,
    const MemoryCoalescingRecord *record,
    int64_t delta
) {
  PolyUOp *offset = NULL;
  if (record->base_kind == 0) {
    offset = record->base;
    if (delta != 0) {
      PolyUOp *c = poly_uop0(ctx, POLY_OP_CONST, offset->dtype, poly_arg_int(delta));
      offset = poly_uop2(ctx, POLY_OP_ADD, offset->dtype, offset, c, poly_arg_none());
    }
  } else if (record->base_kind == 1) {
    offset = poly_uop0(ctx, POLY_OP_CONST, record->idx_dtype, poly_arg_int(delta));
  } else {
    offset = poly_uop_const(ctx, poly_arg_invalid(), record->idx_dtype);
  }
  return offset;
}

/* Current tinygrad codegen/late/coalesce.py::memory_coalescing.  This runs
 * after devectorizer2 has made each memory occurrence scalar, groups adjacent
 * symbolic-base-plus-constant offsets globally, and rebuilds shaped SHRINK
 * loads/stores. */
PolyUOp *poly_memory_coalescing(PolyCtx *ctx, PolyUOp *sink, PolyRendererCaps caps) {
  if (!ctx || !sink || poly_getenv_flag("DMC")) return sink;
  int n_topo = 0;
  PolyUOp **topo = poly_toposort_alloc(ctx, sink, &n_topo);
  if (!topo || n_topo <= 0) {
    if (topo) poly_toposort_free(topo);
    return sink;
  }

  MemoryCoalescingRecord *records = malloc((size_t)n_topo * sizeof(*records));
  PolyUOp **from = malloc((size_t)n_topo * sizeof(*from));
  PolyUOp **to = malloc((size_t)n_topo * sizeof(*to));
  if (!records || !from || !to) {
    free(records);
    free(from);
    free(to);
    poly_toposort_free(topo);
    return NULL;
  }

  int n_records = 0;
  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    if (u->op != POLY_OP_LOAD && u->op != POLY_OP_STORE) continue;
    int expected = u->op == POLY_OP_STORE ? 2 : 1;
    if (u->n_src != expected || !u->src[0] || u->src[0]->op != POLY_OP_INDEX ||
        u->src[0]->n_src != 2) {
      fprintf(stderr, "polygrad: memory_coalescing requires ungated INDEX loads/stores\n");
      free(records);
      free(from);
      free(to);
      poly_toposort_free(topo);
      return NULL;
    }
    PolyUOp *buf = u->src[0]->src[0], *coord = u->src[0]->src[1];
    if (poly_program_memory_is(buf, POLY_ADDR_REG)) continue;
    PolyUOp *idx = poly_uop_get_idx(ctx, coord);
    PolyUOp *valid = poly_uop_get_valid(ctx, coord);
    if (!idx || !valid) continue;

    MemoryCoalescingRecord r = {
        .u = u,
        .buf = buf,
        .base = idx,
        .valid = valid,
        .idx_dtype = idx->dtype,
        .op = u->op,
        .base_kind = 0,
        .offset = 0,
    };
    if (idx->op == POLY_OP_ADD && idx->n_src == 2 && idx->src[1]->op == POLY_OP_CONST &&
        idx->src[1]->arg.kind == POLY_ARG_INT) {
      r.base = idx->src[0];
      r.offset = idx->src[1]->arg.i;
    } else if (idx->op == POLY_OP_ADD && idx->n_src == 2 &&
               idx->src[0]->op == POLY_OP_CONST && idx->src[0]->arg.kind == POLY_ARG_INT) {
      r.base = idx->src[1];
      r.offset = idx->src[0]->arg.i;
    } else if (idx->op == POLY_OP_CONST && idx->arg.kind == POLY_ARG_INVALID) {
      r.base = NULL;
      r.base_kind = 2;
    } else if (idx->op == POLY_OP_CONST && idx->arg.kind == POLY_ARG_INT) {
      r.base = NULL;
      r.base_kind = 1;
      r.offset = idx->arg.i;
    }
    records[n_records++] = r;
  }
  poly_toposort_free(topo);
  if (n_records == 0) {
    free(records);
    free(from);
    free(to);
    return sink;
  }
  qsort(records, (size_t)n_records, sizeof(*records), memory_coalescing_record_cmp);

  int n_sub = 0;
  for (int key_start = 0; key_start < n_records;) {
    int key_end = key_start + 1;
    while (key_end < n_records && memory_coalescing_same_key(&records[key_start], &records[key_end])
    )
      key_end++;

    int n_key = key_end - key_start;
    int64_t *offsets = malloc((size_t)n_key * sizeof(*offsets));
    int *starts = malloc((size_t)n_key * sizeof(*starts));
    int *counts = malloc((size_t)n_key * sizeof(*counts));
    if (!offsets || !starts || !counts) {
      free(offsets);
      free(starts);
      free(counts);
      free(records);
      free(from);
      free(to);
      return NULL;
    }
    int n_unique = 0;
    for (int i = key_start; i < key_end;) {
      int j = i + 1;
      while (j < key_end && records[j].offset == records[i].offset)
        j++;
      offsets[n_unique] = records[i].offset;
      starts[n_unique] = i;
      counts[n_unique++] = j - i;
      i = j;
    }

    for (int run_start = 0; run_start < n_unique;) {
      int run_end = run_start + 1;
      while (run_end < n_unique && offsets[run_end] == offsets[run_end - 1] + 1)
        run_end++;
      for (int pos = run_start; pos < run_end;) {
        MemoryCoalescingRecord *r = &records[starts[pos]];
        int remaining = run_end - pos;
        int candidates[4] = {1, 1, 1, 1};
        int n_candidates = 1;
        if (poly_uop_is_image_shape(ctx, r->buf)) {
          candidates[0] = 4;
          candidates[1] = 1;
          n_candidates = 2;
        } else if (memory_coalescing_dtype(r->buf->dtype) && caps.max_vec_width >= 2) {
          if (caps.max_vec_width >= 4) {
            candidates[0] = 4;
            candidates[1] = 2;
            candidates[2] = 1;
            n_candidates = 3;
          } else {
            candidates[0] = 2;
            candidates[1] = 1;
            n_candidates = 2;
          }
          if (poly_dtype_eq(r->buf->dtype, POLY_FLOAT16) && caps.max_vec_width >= 8 &&
              poly_getenv_flag("ALLOW_HALF8")) {
            candidates[0] = 8;
            candidates[1] = 4;
            candidates[2] = 2;
            candidates[3] = 1;
            n_candidates = 4;
          }
        }
        int length = 1;
        for (int c = 0; c < n_candidates; c++) {
          if (candidates[c] <= remaining) {
            PolyUOp *test_offset = memory_coalescing_offset(ctx, r, offsets[pos]);
            if (candidates[c] == 1 || poly_uop_divides(ctx, test_offset, candidates[c])) {
              length = candidates[c];
              break;
            }
          }
        }

        PolyUOp *offset = memory_coalescing_offset(ctx, r, offsets[pos]);
        /* A predicate constrains execution, not base-address divisibility. */
        if (offset && !const_true(r->valid)) {
          PolyUOp *invalid = poly_uop_const(ctx, poly_arg_invalid(), offset->dtype);
          offset = poly_uop3(
              ctx, POLY_OP_WHERE, offset->dtype, r->valid, offset, invalid, poly_arg_none()
          );
        }
        PolyUOp *address = NULL;
        if (length > 1) {
          PolyUOp *len = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(length));
          PolyUOp *src[3] = {r->buf, offset, len};
          address = poly_uop(ctx, POLY_OP_SHRINK, r->buf->dtype, src, 3, poly_arg_none());
        } else {
          address = poly_uop_index(ctx, r->buf, &offset, 1);
        }
        if (!address) {
          free(offsets);
          free(starts);
          free(counts);
          free(records);
          free(from);
          free(to);
          return NULL;
        }

        if (r->op == POLY_OP_STORE) {
          PolyUOp **data = malloc((size_t)length * sizeof(*data));
          if (!data) {
            free(offsets);
            free(starts);
            free(counts);
            free(records);
            free(from);
            free(to);
            return NULL;
          }
          bool valid_store = true;
          for (int lane = 0; lane < length; lane++) {
            if (counts[pos + lane] != 1) {
              valid_store = false;
              break;
            }
            data[lane] = records[starts[pos + lane]].u->src[1];
          }
          if (!valid_store) {
            fprintf(stderr, "polygrad: memory_coalescing found multiple stores to one offset\n");
            free(data);
            free(offsets);
            free(starts);
            free(counts);
            free(records);
            free(from);
            free(to);
            return NULL;
          }
          PolyUOp *value = length > 1 ? poly_uop_stack(ctx, data, length) : data[0];
          free(data);
          PolyUOp *store =
              poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, address, value, poly_arg_none());
          for (int lane = 0; lane < length; lane++) {
            from[n_sub] = records[starts[pos + lane]].u;
            to[n_sub++] = store;
          }
        } else {
          PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, r->u->dtype, address, poly_arg_none());
          for (int lane = 0; lane < length; lane++) {
            PolyUOp *value = load;
            if (length > 1) {
              PolyUOp *lane_uop = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(lane));
              value = poly_uop_index(ctx, load, &lane_uop, 1);
            }
            for (int k = 0; k < counts[pos + lane]; k++) {
              from[n_sub] = records[starts[pos + lane] + k].u;
              to[n_sub++] = value;
            }
          }
        }
        pos += length;
      }
      run_start = run_end;
    }
    free(offsets);
    free(starts);
    free(counts);
    key_start = key_end;
  }

  PolyUOp *ret = n_sub ? poly_uop_substitute(ctx, sink, from, to, n_sub) : sink;
  free(records);
  free(from);
  free(to);
  return ret;
}

/* Polygrad tags are C metadata around Tinygrad's UOp.replace semantics. */
static PolyUOp *rebuild_preserve_tag(
    PolyCtx *ctx,
    PolyUOp *u,
    PolyDType dtype,
    PolyUOp **srcs,
    int n_src
) {
  PolyArg arg =
      (u->op == POLY_OP_CAST || u->op == POLY_OP_BITCAST) ? poly_arg_dtype(dtype) : u->arg;
  return (u->tag != 0 || u->tag_arg.kind != POLY_ARG_NONE)
             ? poly_uop_tagged_arg(ctx, u->op, dtype, srcs, n_src, arg, u->tag, u->tag_arg)
             : poly_uop(ctx, u->op, dtype, srcs, n_src, arg);
}

static bool is_invalid_const(PolyUOp *u) {
  return u && u->op == POLY_OP_CONST && u->arg.kind == POLY_ARG_INVALID;
}

/* Current tinygrad/codegen/late/coalesce.py:43-46,57-61. */
static PolyUOp *rule_simplify_valid_index(PolyCtx *ctx, PolyUOp *index, const PolyBindings *b) {
  (void)b;
  if (!index || index->op != POLY_OP_INDEX || index->n_src != 2) return NULL;
  PolyUOp *coord = index->src[1];
  if (!coord || coord->op != POLY_OP_WHERE || coord->n_src != 3 || !is_invalid_const(coord->src[2]))
    return NULL;
  PolyUOp *simplified = poly_uop_given_valid(ctx, coord->src[0], coord->src[1], true);
  if (!simplified || simplified == coord->src[1]) return NULL;
  /* Current Tinygrad commit 6b8b2f5a rejects validity specialization that is
   * merely the coordinate's ordinary symbolic simplification. Rewriting it
   * anyway can alternate with that same simplification in a combined matcher. */
  PolyUOp *ordinary = poly_graph_rewrite(ctx, coord->src[1], poly_symbolic());
  if (!ordinary || simplified == ordinary) return NULL;
  PolyUOp *coord_srcs[3] = {coord->src[0], simplified, coord->src[2]};
  PolyUOp *new_coord = rebuild_preserve_tag(ctx, coord, coord->dtype, coord_srcs, 3);
  PolyUOp *index_srcs[2] = {index->src[0], new_coord};
  return poly_uop(ctx, POLY_OP_INDEX, index->dtype, index_srcs, 2, index->arg);
}

static PolyUOp *poly_simplify_valid_image_load(
    PolyCtx *ctx,
    PolyUOp *index,
    const PolyBindings *bindings
);

static _Thread_local PolyPatternMatcher *g_indexing_simplify = NULL;
PolyPatternMatcher *poly_indexing_simplify(void) {
  if (g_indexing_simplify) return g_indexing_simplify;
  /* Tinygrad 2026-08-22/a9069c177a9d codegen/late/coalesce.py:57-61.
   * The common scalar INDEX rule is shared by both current rewrite stages. */
  PolyNamedRule rules[] = {
      POLY_RULE(poly_upat_op(POLY_OP_INDEX, NULL, 0, "idx"), rule_simplify_valid_index),
      POLY_RULE(poly_upat_op(POLY_OP_INDEX, NULL, 0, "idx"), poly_simplify_valid_image_load),
  };
  g_indexing_simplify =
      poly_pm_thread_cache(poly_pm_new_named(rules, (int)(sizeof(rules) / sizeof(rules[0]))));
  return g_indexing_simplify;
}

static bool image_ctx_lookup(
    const PolyImageRewriteCtx *ctx,
    int64_t slot,
    int64_t *height,
    int64_t *width
) {
  if (!ctx) return false;
  for (int i = 0; i < ctx->count; i++) {
    if (ctx->slots[i] != slot) continue;
    if (height) *height = ctx->heights[i];
    if (width) *width = ctx->widths[i];
    return true;
  }
  return false;
}

static bool image_ctx_store(PolyImageRewriteCtx *ctx, int64_t slot, int64_t height, int64_t width) {
  if (!ctx) return false;
  for (int i = 0; i < ctx->count; i++) {
    if (ctx->slots[i] != slot) continue;
    ctx->heights[i] = height;
    ctx->widths[i] = width;
    return true;
  }
  if (ctx->count == ctx->capacity) {
    int next = ctx->capacity ? ctx->capacity * 2 : 8;
    int64_t *slots = malloc((size_t)next * sizeof(*slots));
    int64_t *heights = malloc((size_t)next * sizeof(*heights));
    int64_t *widths = malloc((size_t)next * sizeof(*widths));
    if (!slots || !heights || !widths) {
      free(slots);
      free(heights);
      free(widths);
      return false;
    }
    if (ctx->count) {
      memcpy(slots, ctx->slots, (size_t)ctx->count * sizeof(*slots));
      memcpy(heights, ctx->heights, (size_t)ctx->count * sizeof(*heights));
      memcpy(widths, ctx->widths, (size_t)ctx->count * sizeof(*widths));
    }
    free(ctx->slots);
    free(ctx->heights);
    free(ctx->widths);
    ctx->slots = slots;
    ctx->heights = heights;
    ctx->widths = widths;
    ctx->capacity = next;
  }
  ctx->slots[ctx->count] = slot;
  ctx->heights[ctx->count] = height;
  ctx->widths[ctx->count] = width;
  ctx->count++;
  return true;
}

void poly_image_rewrite_ctx_destroy(PolyImageRewriteCtx *ctx) {
  if (!ctx) return;
  free(ctx->slots);
  free(ctx->heights);
  free(ctx->widths);
  ctx->slots = ctx->heights = ctx->widths = NULL;
  ctx->count = ctx->capacity = 0;
}

static bool image_coord_out_of_bounds(
    PolyCtx *ctx,
    PolyUOp *coord,
    PolyUOp *from,
    PolyUOp *to,
    int64_t bound
) {
  PolyUOp *test = from ? poly_uop_substitute(ctx, coord, &from, &to, 1) : coord;
  if (!test) return false;
  test = poly_graph_rewrite(ctx, test, poly_symbolic());
  if (!test) return false;
  int64_t lo = 0, hi = 0;
  poly_uop_minmax(ctx, test, &lo, &hi);
  return lo >= bound || hi < 0;
}

static PolyUOp *sum_uops(PolyCtx *ctx, PolyUOp **items, int count) {
  if (!items || count <= 0) return NULL;
  PolyUOp *ret = items[0];
  for (int i = 1; i < count; i++)
    ret = poly_uop2(ctx, POLY_OP_ADD, ret->dtype, ret, items[i], poly_arg_none());
  return ret;
}

/* Tinygrad 2026-08-22/a9069c177a9d
 * codegen/late/coalesce.py:_drop_valid_stmts. */
static int drop_valid_stmts(
    PolyCtx *ctx,
    PolyUOp *valid,
    PolyUOp *idx,
    int64_t height,
    int64_t width,
    PolyUOp ***clauses_out,
    bool **dropped_out,
    int *count_out
) {
  int count = 0;
  PolyUOp **clauses = poly_uop_split(valid, POLY_OP_AND, &count);
  bool *dropped = count > 0 ? calloc((size_t)count, sizeof(*dropped)) : NULL;
  if (!clauses || (count > 0 && !dropped)) {
    free(clauses);
    free(dropped);
    return -1;
  }

  int dropped_count = 0;
  for (int i = 0; i < count; i++) {
    PolyUOp *expr = NULL;
    bool is_upper = false;
    int64_t c = 0;
    if (!poly_parse_valid(ctx, clauses[i], &expr, &is_upper, &c)) continue;

    int n_terms = 0;
    PolyUOp **terms = poly_uop_split(expr, POLY_OP_ADD, &n_terms);
    if (!is_upper && c == 1 && terms && n_terms > 0) {
      bool simplex = true;
      PolyUOp **zeros = malloc((size_t)n_terms * sizeof(*zeros));
      for (int j = 0; j < n_terms; j++) {
        int64_t lo = 0, hi = 0;
        poly_uop_minmax(ctx, terms[j], &lo, &hi);
        if (!poly_opset_has(POLY_GROUP_IRREDUCIBLE, terms[j]->op) || lo != 0) simplex = false;
        if (zeros) zeros[j] = poly_const_like_int(ctx, terms[j], 0);
      }
      if (simplex && zeros) {
        PolyUOp *test = poly_uop_substitute(ctx, idx, terms, zeros, n_terms);
        test = test ? poly_graph_rewrite(ctx, test, poly_symbolic()) : NULL;
        PolyUOp *zero = test ? poly_const_int(ctx, 0) : NULL;
        PolyUOp *one = test ? poly_const_int(ctx, 1) : NULL;
        PolyUOp *x = test ? poly_uop_index(ctx, test, &zero, 1) : NULL;
        PolyUOp *y = test ? poly_uop_index(ctx, test, &one, 1) : NULL;
        int64_t xlo = 0, xhi = 0, ylo = 0, yhi = 0;
        if (x && y) {
          poly_uop_minmax(ctx, x, &xlo, &xhi);
          poly_uop_minmax(ctx, y, &ylo, &yhi);
          if (xhi < 0 || yhi < 0) dropped[i] = true;
        }
      }
      free(zeros);
    }

    int64_t expr_lo = 0, expr_hi = 0;
    poly_uop_minmax(ctx, expr, &expr_lo, &expr_hi);
    int64_t lo = is_upper ? (c == INT64_MAX ? INT64_MAX : c + 1) : expr_lo;
    int64_t hi = is_upper ? expr_hi : (c == INT64_MIN ? INT64_MIN : c - 1);
    if (!dropped[i] && lo <= hi) {
      char name[48];
      snprintf(name, sizeof(name), "fake%d", i);
      PolyUOp *fake = poly_uop_variable(ctx, name, lo, hi, expr->dtype, 1, true);
      if (fake) {
        PolyUOp *zero = poly_const_int(ctx, 0), *one = poly_const_int(ctx, 1);
        PolyUOp *x = poly_uop_index(ctx, idx, &zero, 1);
        PolyUOp *y = poly_uop_index(ctx, idx, &one, 1);
        if (image_coord_out_of_bounds(ctx, x, expr, fake, width) ||
            image_coord_out_of_bounds(ctx, y, expr, fake, height))
          dropped[i] = true;

        PolyUOp *variable = NULL;
        for (int j = 0; !dropped[i] && j < n_terms; j++) {
          if (terms[j]->op != POLY_OP_CONST &&
              poly_opset_has(POLY_GROUP_IRREDUCIBLE, terms[j]->op)) {
            variable = terms[j];
            break;
          }
        }
        if (!dropped[i] && variable && n_terms > 1) {
          PolyUOp **rest = malloc((size_t)(n_terms - 1) * sizeof(*rest));
          int n_rest = 0;
          for (int j = 0; rest && j < n_terms; j++)
            if (terms[j] != variable) rest[n_rest++] = terms[j];
          PolyUOp *rest_sum = rest ? sum_uops(ctx, rest, n_rest) : NULL;
          PolyUOp *replacement =
              rest_sum ? poly_uop2(ctx, POLY_OP_SUB, fake->dtype, fake, rest_sum, poly_arg_none())
                       : NULL;
          if (replacement && (image_coord_out_of_bounds(ctx, x, variable, replacement, width) ||
                              image_coord_out_of_bounds(ctx, y, variable, replacement, height)))
            dropped[i] = true;
          free(rest);
        }
      }
    }
    if (dropped[i]) dropped_count++;
    free(terms);
  }

  if (clauses_out)
    *clauses_out = clauses;
  else
    free(clauses);
  if (dropped_out)
    *dropped_out = dropped;
  else
    free(dropped);
  if (count_out) *count_out = count;
  return dropped_count;
}

static PolyUOp *valid_and(PolyCtx *ctx, PolyUOp **clauses, bool *dropped, int count) {
  PolyUOp *ret = NULL;
  for (int i = 0; i < count; i++) {
    if (dropped[i]) continue;
    ret =
        ret ? poly_uop2(ctx, POLY_OP_AND, POLY_BOOL, ret, clauses[i], poly_arg_none()) : clauses[i];
  }
  return ret;
}

static PolyUOp *valid_coord(PolyCtx *ctx, PolyUOp *coord, PolyUOp *valid) {
  if (!valid) return coord;
  PolyUOp *invalid = poly_uop_const(ctx, poly_arg_invalid(), POLY_BOOL);
  return poly_uop3(ctx, POLY_OP_WHERE, coord->dtype, valid, coord, invalid, poly_arg_none());
}

/* Tinygrad 2026-08-22/a9069c177a9d
 * codegen/late/coalesce.py:simplify_valid_image_load. */
static PolyUOp *poly_simplify_valid_image_load(
    PolyCtx *ctx,
    PolyUOp *index,
    const PolyBindings *bindings
) {
  (void)bindings;
  if (!index || index->op != POLY_OP_INDEX || index->n_src != 3 ||
      !poly_uop_is_image_shape(ctx, index->src[0]))
    return NULL;
  PolyUOp *gated_y = index->src[1], *gated_x = index->src[2];
  if (!gated_y || !gated_x || gated_y->op != POLY_OP_WHERE || gated_x->op != POLY_OP_WHERE ||
      gated_y->n_src != 3 || gated_x->n_src != 3 || gated_y->src[0] != gated_x->src[0] ||
      !is_invalid_const(gated_y->src[2]) || !is_invalid_const(gated_x->src[2]))
    return NULL;

  PolyUOp *valid = gated_y->src[0];
  PolyUOp *idx_y = gated_y->src[1], *idx_x = gated_x->src[1];
  if (!poly_dtype_eq(idx_x->dtype, idx_y->dtype)) {
    idx_x = poly_cast(ctx, idx_x, POLY_INT32);
    idx_y = poly_cast(ctx, idx_y, POLY_INT32);
  }
  PolyUOp *start_src[2] = {idx_x, idx_y};
  PolyUOp *start = poly_uop_stack(ctx, start_src, 2);
  PolyUOp *simplified = poly_uop_given_valid(ctx, valid, start, true);
  PolyShape shape = poly_uop_max_shape_cached(ctx, index->src[0]);
  if (!simplified || shape.ndim != 3) return NULL;

  PolyUOp **clauses = NULL;
  bool *dropped = NULL;
  int n_clauses = 0;
  int n_dropped = drop_valid_stmts(
      ctx, valid, simplified, shape.dims[0], shape.dims[1], &clauses, &dropped, &n_clauses
  );
  if (n_dropped < 0) return NULL;
  if (n_dropped == 0 && simplified == start) {
    free(clauses);
    free(dropped);
    return NULL;
  }

  PolyUOp *new_valid = valid_and(ctx, clauses, dropped, n_clauses);
  PolyUOp *zero = poly_const_int(ctx, 0), *one = poly_const_int(ctx, 1);
  idx_x = poly_uop_index(ctx, simplified, &zero, 1);
  idx_y = poly_uop_index(ctx, simplified, &one, 1);
  PolyUOp *coords[2] = {valid_coord(ctx, idx_y, new_valid), valid_coord(ctx, idx_x, new_valid)};
  PolyUOp *ret = poly_uop_index(ctx, index->src[0], coords, 2);
  free(clauses);
  free(dropped);
  return ret;
}

/* Tinygrad 2026-08-22/a9069c177a9d
 * codegen/late/coalesce.py:image_valid_dims. */
PolyImageDim *poly_image_valid_dims(
    PolyDType base,
    int64_t size,
    const char *arch,
    int *count_out
) {
  if (count_out) *count_out = 0;
  const char *key = arch ? strstr(arch, "IMAGE_PITCH_ALIGNMENT=") : NULL;
  if (!key || size < 0) return NULL;
  char *end = NULL;
  long alignment = strtol(key + strlen("IMAGE_PITCH_ALIGNMENT="), &end, 10);
  if (alignment <= 0 || alignment > INT_MAX) return NULL;
  const int64_t max_width = 16384;
  int64_t pixels = size / 4;
  if ((!poly_dtype_eq(base, POLY_FLOAT16) && !poly_dtype_eq(base, POLY_FLOAT32)) ||
      size > 4 * max_width * max_width)
    return NULL;

  PolyImageDim *dims = NULL;
  int count = 0, capacity = 0;
#define APPEND_IMAGE_DIM(h_, w_)                                                                   \
  do {                                                                                             \
    if (count == capacity) {                                                                       \
      int next = capacity ? capacity * 2 : 8;                                                      \
      PolyImageDim *grown = realloc(dims, (size_t)next * sizeof(*grown));                          \
      if (!grown) {                                                                                \
        free(dims);                                                                                \
        return NULL;                                                                               \
      }                                                                                            \
      dims = grown;                                                                                \
      capacity = next;                                                                             \
    }                                                                                              \
    dims[count++] = (PolyImageDim){(h_), (w_)};                                                    \
  } while (0)

  if (size % ((int64_t)alignment * 4) != 0) {
    int itemsize = poly_dtype_itemsize(base);
#ifdef __APPLE__
    int64_t byte_alignment = 64;
#else
    int64_t byte_alignment = alignment;
#endif
    if (itemsize > 0 && size <= INT64_MAX / itemsize && (itemsize * size) % byte_alignment == 0 &&
        pixels <= max_width)
      APPEND_IMAGE_DIM(1, pixels);
  } else {
    int64_t units = pixels / alignment;
    int64_t first = (units + max_width - 1) / max_width;
    int64_t last = units < max_width / alignment ? units : max_width / alignment;
    for (int64_t k = first; k <= last; k++)
      if (k > 0 && units % k == 0) APPEND_IMAGE_DIM(units / k, alignment * k);
  }
#undef APPEND_IMAGE_DIM
  if (count_out) *count_out = count;
  return dims;
}

static bool image_target_supported(const char *device) {
  return device && (!strcmp(device, "QCOM") || !strcmp(device, "CL") || !strcmp(device, "PYTHON") ||
                    !strcmp(device, "NULL"));
}

static PolyUOp *image_coordinate(PolyCtx *ctx, PolyUOp *x, int64_t width) {
  if (!x || width <= 0 || width > INT64_MAX / 4) return NULL;
  PolyUOp *four = poly_const_like_int(ctx, x, 4);
  PolyUOp *row_width = poly_const_like_int(ctx, x, 4 * width);
  PolyUOp *width_uop = poly_const_like_int(ctx, x, width);
  PolyUOp *x_div_four = poly_uop2(ctx, POLY_OP_FLOORDIV, x->dtype, x, four, poly_arg_none());
  PolyUOp *coord_x =
      poly_uop2(ctx, POLY_OP_FLOORMOD, x->dtype, x_div_four, width_uop, poly_arg_none());
  PolyUOp *coord_y = poly_uop2(ctx, POLY_OP_FLOORDIV, x->dtype, x, row_width, poly_arg_none());
  PolyUOp *src[2] = {coord_x, coord_y};
  return poly_uop_stack(ctx, src, 2);
}

static int image_index_complexity(PolyCtx *ctx, PolyUOp *idx) {
  PolyUOp *one = poly_const_int(ctx, 1);
  PolyUOp *y = poly_uop_index(ctx, idx, &one, 1);
  y = y ? poly_graph_rewrite(ctx, y, poly_symbolic()) : NULL;
  int count = 0;
  PolyUOp **topo = y ? poly_toposort_alloc(ctx, y, &count) : NULL;
  bool ok = topo != NULL;
  poly_toposort_free(topo);
  return ok ? count : INT_MAX;
}

/* Tinygrad 2026-08-22/a9069c177a9d
 * codegen/late/coalesce.py:transform_to_image. */
static PolyUOp *poly_transform_to_image(
    PolyCtx *ctx,
    PolyUOp *shrink,
    const PolyBindings *bindings
) {
  (void)bindings;
  PolyImageRewriteCtx *image_ctx = poly_graph_rewrite_userctx();
  if (!image_ctx || !poly_getenv_flag("IMAGE") || !image_target_supported(image_ctx->caps.device) ||
      !shrink || shrink->op != POLY_OP_SHRINK || shrink->n_src != 3 ||
      shrink->src[0]->op != POLY_OP_PARAM)
    return NULL;
  int64_t lanes = 0;
  if (poly_uop_const_i64(shrink->src[2], &lanes) != 0 || lanes != 4) return NULL;

  PolyUOp *buf = shrink->src[0];
  int64_t slot = poly_program_buffer_slot(buf);
  PolyUOp *valid = poly_uop_get_valid(ctx, shrink->src[1]);
  PolyUOp *x = poly_uop_get_idx(ctx, shrink->src[1]);
  if (!valid || !x) return NULL;

  PolyImageDim *dims = NULL;
  int n_dims = 0;
  int64_t known_height = 0, known_width = 0;
  if (image_ctx_lookup(image_ctx, slot, &known_height, &known_width)) {
    dims = malloc(sizeof(*dims));
    if (!dims) return NULL;
    dims[0] = (PolyImageDim){known_height, known_width};
    n_dims = 1;
  } else {
    dims = poly_image_valid_dims(
        buf->dtype, poly_uop_max_numel(ctx, buf), image_ctx->caps.arch, &n_dims
    );
  }
  if (!dims || n_dims == 0) {
    free(dims);
    return NULL;
  }

  int best_drop = -1, best_complexity = INT_MAX;
  int64_t best_height = 0, best_width = 0;
  PolyUOp *best_idx = NULL;
  for (int i = 0; i < n_dims; i++) {
    PolyUOp *candidate = image_coordinate(ctx, x, dims[i].width);
    candidate = candidate ? poly_uop_given_valid(ctx, valid, candidate, true) : NULL;
    if (!candidate) continue;
    int dropped =
        drop_valid_stmts(ctx, valid, candidate, dims[i].height, dims[i].width, NULL, NULL, NULL);
    if (dropped < 0) continue;
    int complexity = image_index_complexity(ctx, candidate);
    if (dropped > best_drop || (dropped == best_drop && complexity < best_complexity)) {
      best_drop = dropped;
      best_complexity = complexity;
      best_height = dims[i].height;
      best_width = dims[i].width;
      best_idx = candidate;
    }
  }
  free(dims);
  if (!best_idx || !image_ctx_store(image_ctx, slot, best_height, best_width)) return NULL;

  PolyUOp *shape_src[3] = {
      poly_const_int(ctx, best_height), poly_const_int(ctx, best_width), poly_const_int(ctx, 4)};
  PolyUOp *shape = poly_shape_to_shape_arg(ctx, shape_src, 3);
  PolyUOp *param_src[1] = {shape};
  PolyUOp *image_buf = rebuild_preserve_tag(ctx, buf, buf->dtype, param_src, 1);
  PolyUOp *zero = poly_const_int(ctx, 0), *one = poly_const_int(ctx, 1);
  PolyUOp *idx_x = poly_uop_index(ctx, best_idx, &zero, 1);
  PolyUOp *idx_y = poly_uop_index(ctx, best_idx, &one, 1);
  if (!const_true(valid)) {
    idx_y = valid_coord(ctx, idx_y, valid);
    idx_x = valid_coord(ctx, idx_x, valid);
  }
  PolyUOp *coords[2] = {idx_y, idx_x};
  return image_buf ? poly_uop_index(ctx, image_buf, coords, 2) : NULL;
}

static PolyUOp *poly_image_load_to_float(
    PolyCtx *ctx,
    PolyUOp *load,
    const PolyBindings *bindings
) {
  (void)bindings;
  if (!load || load->op != POLY_OP_LOAD || load->n_src != 1 ||
      !poly_dtype_eq(load->dtype, POLY_FLOAT16) || load->src[0]->op != POLY_OP_INDEX ||
      !poly_dtype_eq(load->src[0]->dtype, POLY_FLOAT32))
    return NULL;
  PolyUOp *float_load = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, load->src[0], poly_arg_none());
  return float_load ? poly_cast(ctx, float_load, POLY_FLOAT16) : NULL;
}

static PolyUOp *poly_image_store_to_float(
    PolyCtx *ctx,
    PolyUOp *store,
    const PolyBindings *bindings
) {
  (void)bindings;
  if (!store || store->op != POLY_OP_STORE || store->n_src != 2 ||
      store->src[0]->op != POLY_OP_INDEX || !poly_dtype_eq(store->src[0]->dtype, POLY_FLOAT32) ||
      !poly_dtype_eq(store->src[1]->dtype, POLY_FLOAT16))
    return NULL;
  PolyUOp *value = poly_cast(ctx, store->src[1], POLY_FLOAT32);
  return value ? poly_store_val(ctx, store->src[0], value) : NULL;
}

static PolyUOp *poly_remove_image_cast_roundtrip(
    PolyCtx *ctx,
    PolyUOp *cast,
    const PolyBindings *bindings
) {
  (void)ctx;
  (void)bindings;
  if (!cast || cast->op != POLY_OP_CAST || cast->n_src != 1 ||
      !poly_dtype_eq(cast->dtype, POLY_FLOAT32))
    return NULL;
  PolyUOp *inner = cast->src[0];
  if (!inner || inner->op != POLY_OP_CAST || inner->n_src != 1 ||
      !poly_dtype_eq(inner->dtype, POLY_FLOAT16) ||
      !poly_dtype_eq(inner->src[0]->dtype, POLY_FLOAT32))
    return NULL;
  return inner->src[0];
}

static _Thread_local PolyPatternMatcher *g_pm_simplify_add_image = NULL;
PolyPatternMatcher *poly_pm_simplify_add_image(void) {
  if (g_pm_simplify_add_image) return g_pm_simplify_add_image;
  /* Tinygrad 2026-08-22/a9069c177a9d
   * codegen/late/coalesce.py:95-101, in source order. */
  PolyNamedRule rules[] = {
      POLY_RULE(poly_upat_op(POLY_OP_SHRINK, NULL, 0, "shrink"), poly_transform_to_image),
      POLY_RULE(poly_upat_op(POLY_OP_LOAD, NULL, 0, "load"), poly_image_load_to_float),
      POLY_RULE(poly_upat_op(POLY_OP_STORE, NULL, 0, "store"), poly_image_store_to_float),
      POLY_RULE(poly_upat_op(POLY_OP_CAST, NULL, 0, "cast"), poly_remove_image_cast_roundtrip),
  };
  g_pm_simplify_add_image =
      poly_pm_thread_cache(poly_pm_new_named(rules, (int)(sizeof(rules) / sizeof(rules[0]))));
  return g_pm_simplify_add_image;
}
