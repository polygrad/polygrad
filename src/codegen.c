/*
 * codegen.c — Sequential rewrite passes (port of full_rewrite_to_sink)
 *
 * Mirrors tinygrad's codegen/__init__.py full_rewrite_to_sink pipeline.
 * Currently implements:
 *   - pm_reduce: REDUCE → DEFINE_REG + AFTER accumulation + END merge
 *   - pm_decomp: MAX→WHERE, MUL→SHL, IDIV→SHR (late decompositions)
 *   - pm_transcendental: EXP2 → polynomial approximation (xexp2)
 */

#define _POSIX_C_SOURCE 200809L

#include "codegen.h"
#include "bigint.h"
#include "engine/schedule.h"
#include "frontend_internal.h"
#include "schedule/indexing.h"
#include "simplify.h"
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#include <errno.h>
#include "utils.h"

static PolyArg poly_arg_int_tuple_local(int64_t *vals, int n);
static PolyUOp *scalarize_lane_expr(PolyCtx *ctx, PolyUOp *u, int lane);
static bool poly_is_reg_or_local_buffer_codegen(PolyUOp *u);

/* Max hardware vector fold width for load/store splitting.
 * Set by the pipeline before running correct_load_store pass.
 * Default 4 (SSE). Set to 8 for AVX2.
 * Thread-local because concurrent codegen with different backend caps must not
 * share vector width or renderer capability state. */
static _Thread_local int g_max_fold_width = 4;
static _Thread_local PolyRendererCaps g_render_caps = {0};

static bool codegen_is_bool_uop(PolyUOp *u) {
  return u && poly_dtype_is_bool(poly_dtype_scalar(u->dtype));
}

static void poly_debug_print_graph(FILE *fp, PolyUOp *u, const char *tag) {
  if (!fp || !u) return;
  fprintf(fp, "=== GRAPH %s ===\n", tag ? tag : "sink");
  poly_uop_dump_tree(fp, u, 0, 24);
  fprintf(fp, "=== END GRAPH %s ===\n", tag ? tag : "sink");
}

static void poly_debug_stage_graph(const char *tag, PolyUOp *u) {
  if (poly_debug_at_least(3) && u) {
    PolyMap *seen = poly_map_new(256);
    PolyUOp **stack = malloc(1024 * sizeof(PolyUOp *));
    int stack_cap = stack ? 1024 : 0;
    int stack_n = 0;
    int n = 0;
    int n_index = 0, n_weak_range = 0, n_weak_alu = 0;
    int max_count = 1;
    PolyUOp *max_count_uop = NULL;
    if (seen && stack) stack[stack_n++] = u;
    while (stack_n > 0) {
      PolyUOp *x = stack[--stack_n];
      if (!x || poly_map_get(seen, poly_ptr_hash(x), x, poly_ptr_eq)) continue;
      poly_map_set(seen, poly_ptr_hash(x), x, x, poly_ptr_eq);
      n++;
      if (poly_dtype_is_index(x->dtype)) {
        n_index++;
        if (x->op == POLY_OP_RANGE)
          n_weak_range++;
        else if (poly_opset_has(POLY_GROUP_ALU, x->op))
          n_weak_alu++;
      }
      if (x->dtype.count > max_count) {
        max_count = x->dtype.count;
        max_count_uop = x;
      }
      for (int i = 0; i < x->n_src; i++) {
        if (stack_n >= stack_cap) {
          int new_cap = stack_cap ? stack_cap * 2 : 1024;
          PolyUOp **new_stack = realloc(stack, (size_t)new_cap * sizeof(PolyUOp *));
          if (!new_stack) break;
          stack = new_stack;
          stack_cap = new_cap;
        }
        stack[stack_n++] = x->src[i];
      }
    }
    if (seen) poly_map_destroy(seen);
    free(stack);
    fprintf(
        stderr, "[polygrad:codegen_stage] %s nodes=%d weakint=%d weak_range=%d weak_alu=%d\n",
        tag ? tag : "sink", n, n_index, n_weak_range, n_weak_alu
    );
    if (max_count_uop && max_count > 1) {
      fprintf(
          stderr, "[polygrad:codegen_stage] %s max_count=%d max_op=%s\n", tag ? tag : "sink",
          max_count, poly_op_name(max_count_uop->op)
      );
    }
    fflush(stderr);
  }
  if (poly_dump_graph_enabled()) poly_debug_print_graph(stderr, u, tag);
}

static int horizontal_reduce_terms(
    PolyCtx *ctx,
    PolyUOp *inp,
    PolyDType out_dtype,
    PolyUOp **out_terms,
    int max_terms
) {
  if (!inp || !out_terms || max_terms <= 0) return 0;
  if (poly_dtype_eq(inp->dtype, out_dtype)) {
    out_terms[0] = inp;
    return 1;
  }
  int in_cnt = inp->dtype.count;
  int out_cnt = out_dtype.count;
  if (in_cnt <= 0 || out_cnt <= 0 || (in_cnt % out_cnt) != 0) {
    out_terms[0] = inp;
    return 1;
  }
  int horizontal_amount = in_cnt / out_cnt;
  if (horizontal_amount <= 1) {
    out_terms[0] = inp;
    return 1;
  }

  int n_out = 0;
  PolyDType scalar = poly_dtype_scalar(inp->dtype);
  for (int i = 0; i < horizontal_amount && n_out < max_terms; i++) {
    int64_t *idxs = calloc((size_t)out_cnt, sizeof(*idxs));
    if (!idxs) return n_out > 0 ? n_out : 1;
    int n_idxs = 0;
    for (int j = i; j < in_cnt && n_idxs < out_cnt; j += horizontal_amount)
      idxs[n_idxs++] = j;
    PolyDType gep_dtype = (n_idxs == 1) ? scalar : poly_dtype_vec(scalar, n_idxs);
    out_terms[n_out++] =
        poly_uop1(ctx, POLY_OP_GEP, gep_dtype, inp, poly_arg_int_tuple_local(idxs, n_idxs));
    free(idxs);
  }
  return n_out > 0 ? n_out : 1;
}

/* pm_reduce: REDUCE → DEFINE_REG + END merge */

/* Forward declaration: shared substitute helper used by reduce END merge. */
static PolyUOp *substitute_node(
    PolyCtx *ctx,
    PolyUOp *node,
    PolyUOp *old_node,
    PolyUOp *new_node,
    PolyUOp **memo_old,
    PolyUOp **memo_new,
    int *memo_n,
    int memo_cap
);

/* Heuristic optimizer (port of tinygrad hand_coded_optimizations) */

/* axis_to_pos ordering: matches tinygrad's axis_to_pos dict.
 * LOOP:-1, THREAD:0, GLOBAL:0, WARP:1, LOCAL:2, GROUP_REDUCE:2,
 * UPCAST:3, REDUCE:4, UNROLL:5 */
static int axis_to_pos(PolyAxisType t) {
  switch (t) {
  case POLY_AXIS_LOOP:
    return -1;
  case POLY_AXIS_THREAD:
    return 0;
  case POLY_AXIS_GLOBAL:
    return 0;
  case POLY_AXIS_WARP:
    return 1;
  case POLY_AXIS_LOCAL:
    return 2;
  case POLY_AXIS_GROUP_REDUCE:
    return 2;
  case POLY_AXIS_UPCAST:
    return 3;
  case POLY_AXIS_REDUCE:
    return 4;
  case POLY_AXIS_UNROLL:
    return 5;
  default:
    return 6;
  }
}

/* Scheduler state for heuristic optimizer.
 * Matches tinygrad's Scheduler class (postrange.py:17-331). */
#define SCHED_MAX_RNGS 64
#define SCHED_MAX_BUFS 32

typedef struct {
  PolyCtx *ctx;
  PolyUOp *ast; /* current kernel SINK */
  int64_t opt_range_next; /* counter for new axis IDs */

  /* Sorted RANGE list (by axis_to_pos then axis_id) */
  PolyUOp *rngs[SCHED_MAX_RNGS];
  int n_rngs;

  /* Shape (bound of each range) */
  int64_t shape[SCHED_MAX_RNGS];

  /* Axis types */
  PolyAxisType types[SCHED_MAX_RNGS];

  /* INDEX ops (tinygrad k.bufs) - reversed toposort order */
  PolyUOp *bufs[SCHED_MAX_BUFS];
  int n_bufs;

  /* Reachability bitmask per buffer: buf_reach[bi] has bit j set if rngs[j]
   * is reachable from bufs[bi]'s source tree. n_rngs must be <= 64. */
  uint64_t buf_reach[SCHED_MAX_BUFS];
  bool has_reach;

  /* Has reduce op */
  bool has_reduce;

  /* The real tinygrad Scheduler uses dynamic Python lists for rngs/bufs.
   * If the C scratch caps would truncate those lists, skip optional
   * optimization rather than optimizing a partial scheduler view. */
  bool overflow;
} OptScheduler;

static bool sched_can_optimize(const OptScheduler *s) {
  return s && !s->overflow && s->n_rngs > 0;
}

static int sched_rng_cmp(const void *ap, const void *bp) {
  const PolyUOp *a = *(const PolyUOp *const *)ap;
  const PolyUOp *b = *(const PolyUOp *const *)bp;
  int pa = axis_to_pos(poly_range_axis_type(a->arg));
  int pb = axis_to_pos(poly_range_axis_type(b->arg));
  if (pa != pb) return pa - pb;
  int64_t ia = poly_range_axis_id(a->arg);
  int64_t ib = poly_range_axis_id(b->arg);
  if (ia != ib) return (ia < ib) ? -1 : 1;
  return 0;
}

static bool contains_uop(PolyUOp **arr, int n, PolyUOp *u) {
  for (int i = 0; i < n; i++)
    if (arr[i] == u) return true;
  return false;
}

/* tinygrad apply_opts starts by converting eligible LOOP output ranges into
 * GLOBAL ranges (postrange.py:340). WebGPU/CUDA scheduling depends on that
 * boundary before the later upcast heuristics run. */
static PolyUOp *convert_loop_output_ranges_to_global(PolyCtx *ctx, PolyUOp *ast) {
  if (!ctx || !ast) return ast;

  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, ast, &n_topo);
  if (!topo || n_topo <= 0) return ast;

  PolyUOp *output_rngs[64];
  int n_output_rngs = 0;
  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    if (!u || u->op != POLY_OP_END) continue;
    for (int j = 1; j < u->n_src; j++) {
      PolyUOp *ranges[64];
      int n_ranges = poly_uop_ranges(ctx, u->src[j], ranges, 64);
      for (int k = 0; k < n_ranges; k++) {
        PolyUOp *r = ranges[k];
        if (!r || r->op != POLY_OP_RANGE) continue;
        if (poly_range_axis_type(r->arg) == POLY_AXIS_REDUCE) continue;
        if (!contains_uop(output_rngs, n_output_rngs, r) && n_output_rngs < 64)
          output_rngs[n_output_rngs++] = r;
      }
    }
  }

  PolyUOp *from[64];
  PolyUOp *to[64];
  int n_sub = 0;
  for (int i = 0; i < n_output_rngs; i++) {
    PolyUOp *r = output_rngs[i];
    if (poly_range_axis_type(r->arg) != POLY_AXIS_LOOP) continue;

    bool globalizable = true;
    for (int ti = 0; ti < n_topo; ti++) {
      PolyUOp *u = topo[ti];
      if (!u || u->op != POLY_OP_STAGE) continue;
      if (!poly_uop_in_ranges(ctx, u, r)) {
        globalizable = false;
        break;
      }
    }
    if (!globalizable) continue;

    int64_t extra_local[16];
    int n_extra = poly_range_n_extra(r->arg);
    if (n_extra > 16) n_extra = 16;
    int64_t *src_extra = poly_range_extra(r->arg);
    for (int k = 0; k < n_extra; k++)
      extra_local[k] = src_extra[k];

    PolyArg g_arg =
        (n_extra > 0)
            ? poly_arg_range_ex(poly_range_axis_id(r->arg), POLY_AXIS_GLOBAL, extra_local, n_extra)
            : poly_arg_range(poly_range_axis_id(r->arg), POLY_AXIS_GLOBAL);
    PolyUOp *g_rng =
        (r->tag != 0)
            ? poly_uop_tagged(ctx, POLY_OP_RANGE, r->dtype, r->src, r->n_src, g_arg, r->tag)
            : poly_uop(ctx, POLY_OP_RANGE, r->dtype, r->src, r->n_src, g_arg);
    from[n_sub] = r;
    to[n_sub] = g_rng;
    n_sub++;
  }

  return (n_sub > 0) ? poly_uop_substitute(ctx, ast, from, to, n_sub) : ast;
}

/* Build reachability bitmask for all nodes in a toposort.
 * Uses a single forward pass: each node's bitmask = union of its sources' bitmasks.
 * RANGE nodes set their own bit. Result: reachable[i] has bit j set iff rngs[j]
 * is reachable from topo[i]'s source tree.
 *
 * n_rngs must be <= 64 (SCHED_MAX_RNGS). Returns malloc'd array (caller frees).
 * topo_map is used to look up topo index for a UOp pointer. */
static uint64_t *build_reachability_bitmask(
    PolyUOp **topo,
    int n_topo,
    PolyUOp **rngs,
    int n_rngs
) {
  /* Build ptr→index map for O(1) source lookup */
  PolyMap *idx_map = poly_map_new((size_t)(n_topo < 64 ? 64 : (size_t)n_topo * 2));
  /* Store topo index + 1 (so 0 means "not found") */
  int *indices = (int *)malloc((size_t)n_topo * sizeof(int));
  for (int i = 0; i < n_topo; i++) {
    indices[i] = i + 1; /* 1-based so NULL means "not in map" */
    poly_map_set(idx_map, poly_ptr_hash(topo[i]), topo[i], &indices[i], poly_ptr_eq);
  }

  /* Build range_index: for each range UOp, which bit index */
  uint64_t *reach = (uint64_t *)calloc((size_t)n_topo, sizeof(uint64_t));

  /* Set bits for RANGE nodes */
  for (int i = 0; i < n_topo; i++) {
    if (topo[i]->op == POLY_OP_RANGE) {
      for (int ri = 0; ri < n_rngs; ri++) {
        if (rngs[ri] == topo[i]) reach[i] |= (1ULL << ri);
      }
    }
  }

  /* Forward pass: propagate bits from sources */
  for (int i = 0; i < n_topo; i++) {
    for (int j = 0; j < topo[i]->n_src; j++) {
      int *pidx = (int *)poly_map_get(
          idx_map, poly_ptr_hash(topo[i]->src[j]), topo[i]->src[j], poly_ptr_eq
      );
      if (pidx) reach[i] |= reach[*pidx - 1];
    }
  }

  poly_map_destroy(idx_map);
  free(indices);
  return reach;
}

static uint64_t projected_node_reachability(PolyUOp *u, PolyMap *idx_map, uint64_t *reach) {
  if (!u) return 0;
  int *pidx = (int *)poly_map_get(idx_map, poly_ptr_hash(u), u, poly_ptr_eq);
  if (pidx) return reach[*pidx];
  if (u->op != POLY_OP_STACK) return 0;
  uint64_t mask = 0;
  for (int i = 0; i < u->n_src; i++)
    mask |= projected_node_reachability(u->src[i], idx_map, reach);
  return mask;
}

static uint64_t projected_index_reachability(
    PolyCtx *ctx,
    PolyUOp *coord,
    PolyMap *idx_map,
    uint64_t *reach
) {
  PolyUOp *idx = poly_index_get_idx(ctx, coord);
  if (!idx) return 0;
  /* Pinned heuristic.py:118-128,160-175 asks membership in
   * `get_idx().backward_slice`; UOp.backward_slice explicitly excludes the
   * projected coordinate root (uop/ops.py:177-183). Union the source closures
   * rather than returning the root's own RANGE bit. This matters for direct
   * coordinates such as INDEX(mean, r_channel). */
  uint64_t mask = 0;
  for (int i = 0; i < idx->n_src; i++)
    mask |= projected_node_reachability(idx->src[i], idx_map, reach);
  return mask;
}

static void sched_init(OptScheduler *s, PolyCtx *ctx, PolyUOp *sink) {
  s->ctx = ctx;
  s->ast = sink;
  s->n_rngs = 0;
  s->n_bufs = 0;
  s->has_reduce = false;
  s->has_reach = false;
  s->overflow = false;
  for (int i = 0; i < SCHED_MAX_BUFS; i++)
    s->buf_reach[i] = 0;

  int n_topo = 0;
  PolyUOp **topo = poly_toposort_alloc(ctx, sink, &n_topo);

  /* Collect unique RANGE ops with vmax > 0 and INDEX ops */
  int64_t max_id = -1;
  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    if (u->op == POLY_OP_REDUCE || u->op == POLY_OP_REDUCE_AXIS) s->has_reduce = true;
    if (u->op == POLY_OP_RANGE && poly_arg_is_range(u->arg)) {
      /* Check vmax > 0 (i.e., bound > 1 or bound > 0) */
      int64_t bound = 0;
      if (u->n_src > 0 && u->src[0]->op == POLY_OP_CONST && u->src[0]->arg.kind == POLY_ARG_INT)
        bound = u->src[0]->arg.i;
      if (bound <= 1) continue; /* vmax = bound - 1, so vmax > 0 means bound > 1 */
      bool dup = false;
      for (int j = 0; j < s->n_rngs; j++) {
        if (s->rngs[j] == u) {
          dup = true;
          break;
        }
      }
      if (!dup) {
        if (s->n_rngs < SCHED_MAX_RNGS) {
          s->rngs[s->n_rngs] = u;
          s->n_rngs++;
        } else {
          s->overflow = true;
        }
      }
      int64_t aid = poly_range_axis_id(u->arg);
      if (aid > max_id) max_id = aid;
    }
    if (u->op == POLY_OP_INDEX && u->n_src > 0 && u->src[0]->op == POLY_OP_PARAM) {
      if (s->n_bufs < SCHED_MAX_BUFS)
        s->bufs[s->n_bufs++] = u;
      else
        s->overflow = true;
    }
  }
  s->opt_range_next = max_id + 1;

  /* Sort ranges by axis_to_pos ordering */
  qsort(s->rngs, (size_t)s->n_rngs, sizeof(PolyUOp *), sched_rng_cmp);

  /* Reverse bufs order to match tinygrad ([::-1]) */
  for (int i = 0; i < s->n_bufs / 2; i++) {
    PolyUOp *tmp = s->bufs[i];
    s->bufs[i] = s->bufs[s->n_bufs - 1 - i];
    s->bufs[s->n_bufs - 1 - i] = tmp;
  }

  /* Extract shapes and types */
  for (int i = 0; i < s->n_rngs; i++) {
    PolyUOp *r = s->rngs[i];
    s->types[i] = poly_range_axis_type(r->arg);
    s->shape[i] =
        (r->n_src > 0 && r->src[0]->op == POLY_OP_CONST && r->src[0]->arg.kind == POLY_ARG_INT)
            ? r->src[0]->arg.i
            : 0;
  }

  /* Build reachability bitmask: single forward pass over toposort */
  if (!s->overflow && s->n_bufs > 0 && s->n_rngs > 0 && s->n_rngs <= 64) {
    uint64_t *reach = build_reachability_bitmask(topo, n_topo, s->rngs, s->n_rngs);
    /* Extract per-buffer bitmasks */
    PolyMap *idx_map = poly_map_new((size_t)(n_topo < 64 ? 64 : (size_t)n_topo * 2));
    int *indices = (int *)malloc((size_t)n_topo * sizeof(int));
    for (int i = 0; i < n_topo; i++) {
      indices[i] = i;
      poly_map_set(idx_map, poly_ptr_hash(topo[i]), topo[i], &indices[i], poly_ptr_eq);
    }
    for (int bi = 0; bi < s->n_bufs; bi++) {
      if (s->bufs[bi]->n_src >= 2)
        s->buf_reach[bi] =
            projected_index_reachability(ctx, s->bufs[bi]->src[1], idx_map, reach);
    }
    s->has_reach = true;
    poly_map_destroy(idx_map);
    free(indices);
    free(reach);
  }
  poly_toposort_free(topo);
}

/* Refresh rngs, shapes, types after a shift_to modifies the AST */
static void sched_refresh(OptScheduler *s) {
  s->n_rngs = 0;
  s->n_bufs = 0;
  s->has_reduce = false;
  s->has_reach = false;
  s->overflow = false;

  int n_topo = 0;
  PolyUOp **topo = poly_toposort_alloc(s->ctx, s->ast, &n_topo);
  int64_t max_id = -1;

  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    if (u->op == POLY_OP_REDUCE || u->op == POLY_OP_REDUCE_AXIS) s->has_reduce = true;
    if (u->op == POLY_OP_RANGE && poly_arg_is_range(u->arg)) {
      int64_t bound = 0;
      if (u->n_src > 0 && u->src[0]->op == POLY_OP_CONST && u->src[0]->arg.kind == POLY_ARG_INT)
        bound = u->src[0]->arg.i;
      if (bound <= 1) continue;
      bool dup = false;
      for (int j = 0; j < s->n_rngs; j++) {
        if (s->rngs[j] == u) {
          dup = true;
          break;
        }
      }
      if (!dup) {
        if (s->n_rngs < SCHED_MAX_RNGS)
          s->rngs[s->n_rngs++] = u;
        else
          s->overflow = true;
      }
      int64_t aid = poly_range_axis_id(u->arg);
      if (aid > max_id) max_id = aid;
    }
    if (u->op == POLY_OP_INDEX && u->n_src > 0 && u->src[0]->op == POLY_OP_PARAM) {
      if (s->n_bufs < SCHED_MAX_BUFS)
        s->bufs[s->n_bufs++] = u;
      else
        s->overflow = true;
    }
  }
  if (max_id + 1 > s->opt_range_next) s->opt_range_next = max_id + 1;

  qsort(s->rngs, (size_t)s->n_rngs, sizeof(PolyUOp *), sched_rng_cmp);
  for (int i = 0; i < s->n_bufs / 2; i++) {
    PolyUOp *tmp = s->bufs[i];
    s->bufs[i] = s->bufs[s->n_bufs - 1 - i];
    s->bufs[s->n_bufs - 1 - i] = tmp;
  }
  for (int i = 0; i < s->n_rngs; i++) {
    PolyUOp *r = s->rngs[i];
    s->types[i] = poly_range_axis_type(r->arg);
    s->shape[i] =
        (r->n_src > 0 && r->src[0]->op == POLY_OP_CONST && r->src[0]->arg.kind == POLY_ARG_INT)
            ? r->src[0]->arg.i
            : 0;
  }

  /* Rebuild reachability bitmask */
  for (int i = 0; i < SCHED_MAX_BUFS; i++)
    s->buf_reach[i] = 0;
  if (!s->overflow && s->n_bufs > 0 && s->n_rngs > 0 && s->n_rngs <= 64) {
    uint64_t *reach = build_reachability_bitmask(topo, n_topo, s->rngs, s->n_rngs);
    PolyMap *idx_map = poly_map_new((size_t)(n_topo < 64 ? 64 : (size_t)n_topo * 2));
    int *indices = (int *)malloc((size_t)n_topo * sizeof(int));
    for (int i = 0; i < n_topo; i++) {
      indices[i] = i;
      poly_map_set(idx_map, poly_ptr_hash(topo[i]), topo[i], &indices[i], poly_ptr_eq);
    }
    for (int bi = 0; bi < s->n_bufs; bi++) {
      if (s->bufs[bi]->n_src >= 2)
        s->buf_reach[bi] = projected_index_reachability(
            s->ctx, s->bufs[bi]->src[1], idx_map, reach
        );
    }
    s->has_reach = true;
    poly_map_destroy(idx_map);
    free(indices);
    free(reach);
  }
  poly_toposort_free(topo);
}

/* shift_to_ex: split a RANGE into two. Port of tinygrad Scheduler.shift_to.
 * top=false: old_range = replaced * amount + new_rng
 * top=true:  old_range = new_rng * old_sz + replaced
 * input_new_rng: if non-NULL, use this expression instead of creating a fresh RANGE.
 *   This is used for TC WARP modular arithmetic (e.g. warp%2).
 * out_new_rng: if non-NULL, receives the new range expression (the split-off part).
 * Returns the replaced range UOp (the one that keeps the old axis type), or NULL on failure. */
/* Core shift_to: substitute only + refresh. Matches tinygrad's shift_to semantics.
 * Does NOT run graph_rewrite -- callers that need simplification do it themselves. */
static PolyUOp *sched_shift_to_core(
    OptScheduler *s,
    PolyUOp *rng,
    int64_t amount,
    PolyAxisType new_type,
    bool top,
    PolyUOp *input_new_rng,
    PolyUOp **out_new_rng
) {
  if (!s || s->overflow) return NULL;
  if (!input_new_rng && s->n_rngs >= SCHED_MAX_RNGS) return NULL;

  int64_t bound = 0;
  if (rng->n_src > 0 && rng->src[0]->op == POLY_OP_CONST && rng->src[0]->arg.kind == POLY_ARG_INT)
    bound = rng->src[0]->arg.i;
  if (bound <= 0 || bound % amount != 0) return NULL;
  int64_t old_sz = bound / amount;

  PolyCtx *ctx = s->ctx;
  PolyDType dt = rng->dtype;

  /* Create or use provided new range */
  PolyUOp *new_rng;
  if (input_new_rng) {
    new_rng = input_new_rng;
  } else {
    PolyUOp *new_sz = poly_uop0(ctx, POLY_OP_CONST, dt, poly_arg_int(amount));
    new_rng =
        poly_uop1(ctx, POLY_OP_RANGE, dt, new_sz, poly_arg_range(s->opt_range_next++, new_type));
  }

  /* Create complementary range with reduced bound */
  PolyUOp *rep_sz = poly_uop0(ctx, POLY_OP_CONST, dt, poly_arg_int(old_sz));
  PolyUOp *replaced = poly_uop1(ctx, POLY_OP_RANGE, dt, rep_sz, rng->arg);

  /* Compute substitution expression */
  PolyUOp *sub_axis;
  if (top) {
    PolyUOp *c = poly_uop0(ctx, POLY_OP_CONST, dt, poly_arg_int(old_sz));
    sub_axis = poly_uop2(
        ctx, POLY_OP_ADD, dt, poly_uop2(ctx, POLY_OP_MUL, dt, new_rng, c, poly_arg_none()),
        replaced, poly_arg_none()
    );
  } else {
    PolyUOp *c = poly_uop0(ctx, POLY_OP_CONST, dt, poly_arg_int(amount));
    sub_axis = poly_uop2(
        ctx, POLY_OP_ADD, dt, poly_uop2(ctx, POLY_OP_MUL, dt, replaced, c, poly_arg_none()),
        new_rng, poly_arg_none()
    );
  }

  PolyUOp *from[1] = {rng};
  PolyUOp *to[1] = {sub_axis};
  OptScheduler old = *s;
  s->ast = poly_uop_substitute(ctx, s->ast, from, to, 1);
  sched_refresh(s);
  if (s->overflow) {
    *s = old;
    return NULL;
  }
  if (out_new_rng) *out_new_rng = new_rng;
  return replaced;
}

static PolyUOp *sched_shift_to(
    OptScheduler *s,
    PolyUOp *rng,
    int64_t amount,
    PolyAxisType new_type,
    bool top
) {
  /* Tinygrad Scheduler.apply_opt leaves shift_to substitutions structural.
   * In particular, full-size UPCAST keeps the complementary size-1 GLOBAL
   * range alive until later passes. Running symbolic cleanup here deletes
   * those ranges too early and changes downstream move_where/expander shape. */
  return sched_shift_to_core(s, rng, amount, new_type, top, NULL, NULL);
}

/* Helper: get indices of upcastable dims (GLOBAL/LOCAL/LOOP with size > 1) */
static int sched_upcastable_dims(const OptScheduler *s, int *out, int max_n) {
  int n = 0;
  for (int i = 0; i < s->n_rngs && n < max_n; i++) {
    PolyAxisType t = s->types[i];
    if ((t == POLY_AXIS_GLOBAL || t == POLY_AXIS_LOCAL || t == POLY_AXIS_LOOP) && s->shape[i] > 1)
      out[n++] = i;
  }
  return n;
}

/* Helper: get indices of unrollable dims (GROUP_REDUCE/REDUCE with size > 1) */
static int sched_unrollable_dims(const OptScheduler *s, int *out, int max_n) {
  int n = 0;
  for (int i = 0; i < s->n_rngs && n < max_n; i++) {
    PolyAxisType t = s->types[i];
    if ((t == POLY_AXIS_GROUP_REDUCE || t == POLY_AXIS_REDUCE) && s->shape[i] > 1) out[n++] = i;
  }
  return n;
}

/* Helper: product of UPCAST/UNROLL dim sizes */
static int64_t sched_upcast_size(const OptScheduler *s) {
  int64_t prod = 1;
  for (int i = 0; i < s->n_rngs; i++) {
    if (s->types[i] == POLY_AXIS_UPCAST || s->types[i] == POLY_AXIS_UNROLL) prod *= s->shape[i];
  }
  return prod;
}

static int64_t sched_full_shape_prod(const OptScheduler *s) {
  int64_t prod = 1;
  for (int i = 0; i < s->n_rngs; i++) {
    if (s->shape[i] <= 0) continue;
    if (prod > INT64_MAX / s->shape[i]) return INT64_MAX;
    prod *= s->shape[i];
  }
  return prod;
}

/* Helper: count of UPCAST/UNROLL axes */
static int sched_upcasted(const OptScheduler *s) {
  int n = 0;
  for (int i = 0; i < s->n_rngs; i++) {
    if (s->types[i] == POLY_AXIS_UPCAST || s->types[i] == POLY_AXIS_UNROLL) n++;
  }
  return n;
}

static bool sched_has_axis_type(const OptScheduler *s, PolyAxisType t) {
  for (int i = 0; i < s->n_rngs; i++) {
    if (s->types[i] == t) return true;
  }
  return false;
}

/* Helper: product of output shape (non-reduce dims) at upcastable indices */
static int64_t sched_output_prod_upcastable(const OptScheduler *s) {
  int up_dims[SCHED_MAX_RNGS];
  int n_up = sched_upcastable_dims(s, up_dims, SCHED_MAX_RNGS);
  int64_t prod = 1;
  for (int i = 0; i < n_up; i++)
    prod *= s->shape[up_dims[i]];
  return prod;
}

typedef struct {
  bool expanded;
  int axis;
} LocalAxisRank;

typedef struct {
  int axis;
  int size;
} LocalChoice;

static int cmp_local_axis_rank(const void *ap, const void *bp) {
  const LocalAxisRank *a = (const LocalAxisRank *)ap;
  const LocalAxisRank *b = (const LocalAxisRank *)bp;
  if (a->expanded != b->expanded) return a->expanded ? -1 : 1;
  return b->axis - a->axis;
}

static int cmp_local_choice_axis(const void *ap, const void *bp) {
  const LocalChoice *a = (const LocalChoice *)ap;
  const LocalChoice *b = (const LocalChoice *)bp;
  return a->axis - b->axis;
}

/* Flatten ADD tree into leaf addends (port of tinygrad split_uop(ADD)) */
static int split_uop_add(PolyUOp *u, PolyUOp **out, int max_n) {
  if (u->op == POLY_OP_ADD) {
    int n = 0;
    for (int i = 0; i < u->n_src && n < max_n; i++)
      n += split_uop_add(u->src[i], out + n, max_n - n);
    return n;
  }
  if (max_n > 0) {
    out[0] = u;
    return 1;
  }
  return 0;
}

/* Flatten AND tree into leaf clauses (port of tinygrad split_uop(AND)). */
static int split_uop_and(PolyUOp *u, PolyUOp **out, int max_n) {
  if (u->op == POLY_OP_AND) {
    int n = 0;
    for (int i = 0; i < u->n_src && n < max_n; i++)
      n += split_uop_and(u->src[i], out + n, max_n - n);
    return n;
  }
  if (max_n > 0) {
    out[0] = u;
    return 1;
  }
  return 0;
}

static bool uop_ptr_in_list(PolyUOp *u, PolyUOp **list, int n) {
  for (int i = 0; i < n; i++)
    if (list[i] == u) return true;
  return false;
}

static bool is_true_clause_const(PolyUOp *u) {
  return u && u->op == POLY_OP_CONST &&
         ((u->arg.kind == POLY_ARG_BOOL && u->arg.b) ||
          (u->arg.kind == POLY_ARG_INT && u->arg.i == 1));
}

/* Tinygrad's UOp.uprod(*clauses) is bool-AND here. Skip literal `true`
 * identity clauses locally so a LOAD gate does not keep `true AND mask`
 * after pm_move_where_on_load. */
static PolyUOp *and_all_clauses(PolyCtx *ctx, PolyUOp **clauses, int n, PolyUOp *init) {
  PolyUOp *acc = (init && !is_true_clause_const(init)) ? init : NULL;
  for (int i = 0; i < n; i++) {
    if (is_true_clause_const(clauses[i])) continue;
    if (!acc) {
      acc = clauses[i];
      continue;
    }
    acc = poly_uop2(ctx, POLY_OP_AND, POLY_BOOL, acc, clauses[i], poly_arg_none());
  }
  return acc ? acc : poly_const_typed(ctx, POLY_BOOL, 1.0);
}

/* TensorCore helpers (port of tc.py) */

/* tc.py:33 -- get_reduce_axes: returns [(0,2), (1,2), ...] for K dimension */
int poly_tc_get_reduce_axes(const PolyTensorCore *tc, int out[][2]) {
  int k = tc->dims[2], n = 0;
  while (k > 1) {
    out[n][0] = n;
    out[n][1] = 2;
    n++;
    k /= 2;
  }
  return n;
}

/* tc.py:34-35 -- count local/upcast opts */
int poly_tc_count_local(const PolyTensorCore *tc) {
  int n = 0;
  for (int i = 0; i < tc->n_opts; i++)
    if (tc->opts[i].type == 'l') n++;
  return n;
}
int poly_tc_count_upcast(const PolyTensorCore *tc) {
  int n = 0;
  for (int i = 0; i < tc->n_opts; i++)
    if (tc->opts[i].type == 'u') n++;
  return n;
}

/* tc.py:25-32 -- base_shape_str: build axis name list from opts + reduce axes.
 * Returns count of entries written to out[]. Each entry is a string like "l0","u1","r3". */
int poly_tc_base_shape_str(const PolyTensorCore *tc, const char *out[], int max_n) {
  int n = 0, cnt_l = 0, cnt_u = 0;
  for (int i = 0; i < tc->n_opts && n < max_n; i++) {
    static const char *l_names[] = {"l0", "l1", "l2", "l3", "l4", "l5", "l6", "l7"};
    static const char *u_names[] = {"u0", "u1", "u2", "u3", "u4", "u5", "u6", "u7"};
    if (tc->opts[i].type == 'l')
      out[n++] = l_names[cnt_l++];
    else
      out[n++] = u_names[cnt_u++];
  }
  /* Append reduce axes */
  int ra[16][2];
  int n_ra = poly_tc_get_reduce_axes(tc, ra);
  static const char *r_names[] = {"r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7"};
  for (int i = 0; i < n_ra && n < max_n; i++)
    out[n++] = r_names[i];
  return n;
}

/* tc.py:36-38 -- base_upcast_axes: reversed list of reduce + upcast axis names */
int poly_tc_base_upcast_axes(const PolyTensorCore *tc, const char *out[], int max_n) {
  int n_upcast = poly_tc_count_upcast(tc);
  int ra[16][2];
  int n_ra = poly_tc_get_reduce_axes(tc, ra);
  /* Build forward: [r0..rN, u0..uM] then reverse */
  const char *fwd[32];
  int n_fwd = 0;
  static const char *r_names[] = {"r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7"};
  static const char *u_names[] = {"u0", "u1", "u2", "u3", "u4", "u5", "u6", "u7"};
  for (int i = 0; i < n_ra && n_fwd < 32; i++)
    fwd[n_fwd++] = r_names[i];
  for (int i = 0; i < n_upcast && n_fwd < 32; i++)
    fwd[n_fwd++] = u_names[i];
  int n = 0;
  for (int i = n_fwd - 1; i >= 0 && n < max_n; i--)
    out[n++] = fwd[i];
  return n;
}

/* tc.py:17-20 -- _remaps: build two remap dicts from swizzle.
 * fwd_st = base_shape_str, remap[i] maps fwd_st[j] -> swizzle[i] flattened.
 * Returns remap as parallel arrays: remap_from[k], remap_to[k] for k in 0..n-1. */
static int poly_tc_build_remap(
    const PolyTensorCore *tc,
    int swz_idx,
    const char *remap_from[],
    const char *remap_to[],
    int max_n
) {
  const char *fwd[32];
  int n_fwd = poly_tc_base_shape_str(tc, fwd, 32);
  /* Flatten swizzle[swz_idx]: [local_axes] + [upcast_axes] + [reduce_axes] */
  const char *flat[32];
  int n_flat = 0;
  for (int g = 0; g < 3; g++)
    for (int j = 0; j < tc->swizzle_len[swz_idx][g] && n_flat < 32; j++)
      flat[n_flat++] = tc->swizzle[swz_idx][g][j];
  int n = (n_fwd < n_flat) ? n_fwd : n_flat;
  if (n > max_n) n = max_n;
  for (int i = 0; i < n; i++) {
    remap_from[i] = fwd[i];
    remap_to[i] = flat[i];
  }
  return n;
}

/* tc.py:21-23 -- permutes_for_shape_str: given shape_str, apply remap and return permutation.
 * shape_str[i] is an axis name. Output perm[i] = shape_str.index(remap[shape_str[i]]).
 * If shape_str[i] is not in remap, perm[i] = i. */
void poly_tc_permute_for_shape_str(
    const PolyTensorCore *tc,
    int swz_idx,
    const char *shape_str[],
    int n_shape,
    int perm[],
    int max_n
) {
  const char *rf[32], *rt[32];
  int n_remap = poly_tc_build_remap(tc, swz_idx, rf, rt, 32);
  for (int i = 0; i < n_shape && i < max_n; i++) {
    /* Find shape_str[i] in remap_from -> get remap_to */
    const char *mapped = NULL;
    for (int r = 0; r < n_remap; r++) {
      if (strcmp(shape_str[i], rf[r]) == 0) {
        mapped = rt[r];
        break;
      }
    }
    if (!mapped) {
      perm[i] = i;
      continue;
    }
    /* Find mapped in shape_str -> get index */
    perm[i] = i; /* default if not found */
    for (int j = 0; j < n_shape; j++) {
      if (strcmp(shape_str[j], mapped) == 0) {
        perm[i] = j;
        break;
      }
    }
  }
}

/* Axis letter mapping (ops.py:20-21) */
static const char *axis_letter(PolyAxisType t) {
  switch (t) {
  case POLY_AXIS_GLOBAL:
    return "g";
  case POLY_AXIS_LOCAL:
    return "l";
  case POLY_AXIS_WARP:
    return "w";
  case POLY_AXIS_UPCAST:
    return "u";
  case POLY_AXIS_GROUP_REDUCE:
    return "G";
  case POLY_AXIS_REDUCE:
    return "R";
  case POLY_AXIS_UNROLL:
    return "r";
  case POLY_AXIS_LOOP:
    return "L";
  default:
    return "?";
  }
}

/* postrange.py:36-42 -- shape_str: build axis name array from scheduler state */
static int sched_shape_str(const OptScheduler *s, const char *out[], int max_n) {
  int n = 0;
  int cnt[16] = {0}; /* count per axis type */
  static char buf[64][8]; /* static buffer for generated strings */
  for (int i = 0; i < s->n_rngs && n < max_n && n < 64; i++) {
    int t = (int)s->types[i];
    snprintf(buf[n], 8, "%s%d", axis_letter(s->types[i]), cnt[t]++);
    out[n] = buf[n];
    n++;
  }
  return n;
}

/* Collect RANGE ops reachable from a UOp via backward walk.
 * Returns bitmask: bit j set if s->rngs[j] is reachable from u. */
static uint64_t collect_ranges_from(const OptScheduler *s, PolyUOp *u) {
  /* Simple DFS -- limited depth for scheduler-level graphs */
  uint64_t mask = 0;
  PolyUOp *stack[512];
  int sp = 0;
  stack[sp++] = u;
  /* Visited set using a simple pointer set (arena-allocated UOps have unique addresses) */
  PolyMap *visited = poly_map_new(256);
  while (sp > 0) {
    PolyUOp *cur = stack[--sp];
    if (poly_map_get(visited, poly_ptr_hash(cur), cur, poly_ptr_eq)) continue;
    poly_map_set(visited, poly_ptr_hash(cur), cur, cur, poly_ptr_eq);
    if (cur->op == POLY_OP_RANGE) {
      for (int j = 0; j < s->n_rngs; j++) {
        if (s->rngs[j] == cur) {
          mask |= (1ULL << j);
          break;
        }
      }
    }
    /* Don't recurse past RANGE (tinygrad: ended_ranges) */
    int rs = range_start_for_op(cur->op);
    int end = (rs >= 0) ? rs : cur->n_src;
    for (int i = 0; i < end && sp < 510; i++)
      stack[sp++] = cur->src[i];
  }
  poly_map_destroy(visited);
  return mask;
}

/* Forward declarations */
static PolyArg poly_arg_pair_tuple(int64_t (*pairs)[2], int n);
static void sched_copy(OptScheduler *dst, const OptScheduler *src);

/* sched_apply_tc_opt (port of postrange.py:221-314) */
#define TC_TAG 0x5443 /* 'TC' */

/* Sort helper: sort UOp* array by axis_id descending */
static int cmp_axis_id_desc(const void *a, const void *b) {
  int64_t ia = poly_range_axis_id((*(const PolyUOp *const *)a)->arg);
  int64_t ib = poly_range_axis_id((*(const PolyUOp *const *)b)->arg);
  return (ib > ia) - (ib < ia);
}

/* Returns true on success. On success, tc_axes_out[0..2] are the replaced N,M,K ranges.
 * use_tc: 1 = full WMMA construction, 2 = shape only (no WMMA UOps).
 * tc_select: -1 = try all, >=0 = specific TC index.
 * tc_opt: 0 = strict (single reduce axis, direct load->mul),
 *         1 = allow CAST'd buffers + multiple reduce axes.
 *         (2 = reserved for future PADTO; currently same as 1) */
static bool sched_apply_tc_opt(
    OptScheduler *s,
    int axis,
    int tc_select,
    int tc_opt,
    int use_tc,
    const PolyTensorCore *tcs,
    int n_tcs,
    PolyUOp *tc_axes_out[3]
) {
  PolyCtx *ctx = s->ctx;

  /* 1. Find REDUCE(ADD) and its MUL (postrange.py:222-227) */
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, s->ast, &n_topo);
  PolyUOp *reduceop = NULL;
  for (int i = 0; i < n_topo; i++) {
    if (topo[i]->op == POLY_OP_REDUCE && topo[i]->arg.kind == POLY_ARG_OPS &&
        topo[i]->arg.ops == POLY_OP_ADD) {
      reduceop = topo[i];
      break;
    }
  }
  if (!reduceop || !use_tc) return false;

  PolyUOp *mul = reduceop->src[0];
  /* tc_opt >= 1 allows CAST'd buffers (postrange.py:225) */
  if (mul->op == POLY_OP_CAST && mul->n_src > 0) {
    if (tc_opt < 1) {
      return false;
    }
    mul = mul->src[0];
  }
  if (mul->op != POLY_OP_MUL || mul->n_src < 2) {
    return false;
  }

  PolyUOp *in0 = mul->src[0], *in1 = mul->src[1];

  /* 2. Try each TC spec (postrange.py:232-248) */
  const PolyTensorCore *tc_list = tcs;
  int tc_count = n_tcs;
  if (tc_select >= 0 && tc_select < n_tcs) {
    tc_list = &tcs[tc_select];
    tc_count = 1;
  } else if (tc_select >= n_tcs)
    return false;

  for (int tci = 0; tci < tc_count; tci++) {
    const PolyTensorCore *tc = &tc_list[tci];

    /* Check dtype match */
    PolyDType in0_scalar = poly_dtype_scalar(in0->dtype);
    PolyDType in1_scalar = poly_dtype_scalar(in1->dtype);
    PolyDType red_scalar = poly_dtype_scalar(reduceop->dtype);
    if (!poly_dtype_eq(tc->dtype_in, in0_scalar) || !poly_dtype_eq(tc->dtype_in, in1_scalar))
      continue;
    if (!poly_dtype_eq(tc->dtype_out, red_scalar)) continue;

    /* 3. Classify ranges (postrange.py:236-238) */
    uint64_t in0_reach = collect_ranges_from(s, in0);
    uint64_t in1_reach = collect_ranges_from(s, in1);

    PolyUOp *in0_ranges[SCHED_MAX_RNGS], *in1_ranges[SCHED_MAX_RNGS];
    int n_in0 = 0, n_in1 = 0;
    for (int i = 0; i < s->n_rngs; i++) {
      uint64_t bit = 1ULL << i;
      if ((in0_reach & bit) && !(in1_reach & bit) && n_in0 < SCHED_MAX_RNGS)
        in0_ranges[n_in0++] = s->rngs[i];
      if ((in1_reach & bit) && !(in0_reach & bit) && n_in1 < SCHED_MAX_RNGS)
        in1_ranges[n_in1++] = s->rngs[i];
    }

    /* red_ranges from REDUCE's trailing RANGE sources */
    PolyUOp *red_ranges[SCHED_MAX_RNGS];
    int n_red = 0;
    int rs = range_start_for_op(POLY_OP_REDUCE);
    for (int i = rs; i < reduceop->n_src && n_red < SCHED_MAX_RNGS; i++) {
      if (reduceop->src[i]->op == POLY_OP_RANGE) red_ranges[n_red++] = reduceop->src[i];
    }

    /* Sort all three by axis_id descending (postrange.py:236-238) */
    if (n_in0 > 1) qsort(in0_ranges, (size_t)n_in0, sizeof(PolyUOp *), cmp_axis_id_desc);
    if (n_in1 > 1) qsort(in1_ranges, (size_t)n_in1, sizeof(PolyUOp *), cmp_axis_id_desc);
    if (n_red > 1) qsort(red_ranges, (size_t)n_red, sizeof(PolyUOp *), cmp_axis_id_desc);

    if (n_in0 == 0 || n_in1 == 0 || n_red == 0) continue;

    /* tc_opt == 0: strict mode requires exactly one reduce axis (heuristic.py:28) */
    if (tc_opt == 0 && n_red > 1) continue;

    /* 4. Axis choices: product(in1_ranges, in0_ranges, red_ranges) -- note swap */
    int n_choices = n_in1 * n_in0 * n_red;
    if (axis >= n_choices) continue;
    int red_idx = axis % n_red;
    int in0_idx = (axis / n_red) % n_in0;
    int in1_idx = (axis / n_red / n_in0) % n_in1;

    PolyUOp *axes[3] = {in1_ranges[in1_idx], in0_ranges[in0_idx], red_ranges[red_idx]};

    /* 5. Tag reduceop via tagged clone + substitute (matches tinygrad's
     * self.ast.substitute({reduceop: reduceop.replace(tag="TC")})).
     * poly_uop_tagged includes tag in CSE key, creating a distinct node. */
    PolyUOp *tagged_red = poly_uop_tagged(
        ctx, reduceop->op, reduceop->dtype, reduceop->src, reduceop->n_src, reduceop->arg, TC_TAG
    );
    {
      PolyUOp *from_tag[1] = {reduceop};
      PolyUOp *to_tag[1] = {tagged_red};
      s->ast = poly_uop_substitute(ctx, s->ast, from_tag, to_tag, 1);
      sched_refresh(s);
    }

    /* 6. Check divisibility -- reject non-const bounds (postrange.py:254-262) */
    bool pad_ok = true;
    for (int i = 0; i < 3; i++) {
      if (axes[i]->n_src == 0 || axes[i]->src[0]->op != POLY_OP_CONST ||
          axes[i]->src[0]->arg.kind != POLY_ARG_INT) {
        pad_ok = false;
        break; /* non-const bound: hard reject */
      }
      int64_t sz = axes[i]->src[0]->arg.i;
      if (sz <= 0 || sz % tc->dims[i] != 0) {
        if (tc_opt < 2) {
          pad_ok = false;
          break;
        }
        /* TODO: PADTO support */
        pad_ok = false;
        break;
      }
    }
    if (!pad_ok) {
      continue;
    }
    /* Verify tag survived the substitute+refresh */

    /* 7. Create WARP range and apply opts (postrange.py:264-274) */
    PolyUOp *warp_sz = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(tc->threads));
    PolyUOp *warp =
        poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, warp_sz, poly_arg_range(-1, POLY_AXIS_WARP));

    PolyUOp *ne[32];
    int n_ne = 0;

    for (int oi = 0; oi < tc->n_opts; oi++) {
      char otype = tc->opts[oi].type;
      int odim = tc->opts[oi].dim;
      PolyUOp *new_rng = NULL;

      if (otype == 'l') {
        PolyUOp *two = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(2));
        PolyUOp *warp_mod2 =
            poly_uop2(ctx, POLY_OP_FLOORMOD, POLY_INT32, warp, two, poly_arg_none());
        axes[odim] =
            sched_shift_to_core(s, axes[odim], 2, POLY_AXIS_LOCAL, false, warp_mod2, &new_rng);
        warp = poly_uop2(ctx, POLY_OP_FLOORDIV, POLY_INT32, warp, two, poly_arg_none());
      } else if (otype == 'u') {
        axes[odim] = sched_shift_to_core(s, axes[odim], 2, POLY_AXIS_UPCAST, false, NULL, &new_rng);
      }
      if (!axes[odim]) {
        return false;
      }
      if (new_rng) ne[n_ne++] = new_rng;
    }

    /* 8. Apply reduce axes (postrange.py:276-278) */
    int ra[16][2];
    int n_ra = poly_tc_get_reduce_axes(tc, ra);
    for (int i = 0; i < n_ra; i++) {
      PolyUOp *new_rng = NULL;
      axes[2] = sched_shift_to_core(s, axes[2], ra[i][1], POLY_AXIS_UNROLL, false, NULL, &new_rng);
      if (!axes[2]) {
        return false;
      }
      if (new_rng) ne[n_ne++] = new_rng;
    }

    /* 9. Build WMMA UOps if use_tc == 1 (postrange.py:280-313) */
    if (use_tc == 1) {
      /* Re-find the tagged reduceop */
      int n_topo2 = 0;
      PolyUOp **topo2 = poly_toposort(ctx, s->ast, &n_topo2);

      /* Debug: check how many ne[] pointers are present in the current AST */
      PolyUOp *found_red = NULL;
      for (int i = 0; i < n_topo2; i++) {
        if (topo2[i]->op == POLY_OP_REDUCE && topo2[i]->tag == TC_TAG) {
          found_red = topo2[i];
          break;
        }
      }
      if (!found_red) return false;

      /* Create tagged copies of ne[] (postrange.py:283: tne = [x.replace(tag=1) for x in ne]).
       * Tags ALL elements (RANGE, ALU expressions like warp%2, etc.) -- not just RANGEs. */
      PolyUOp *tne[32];
      for (int i = 0; i < n_ne; i++) {
        tne[i] =
            poly_uop_tagged(ctx, ne[i]->op, ne[i]->dtype, ne[i]->src, ne[i]->n_src, ne[i]->arg, 1);
      }

      /* Substitute ne -> tne in found_red to isolate the MUL operands (postrange.py:284-285) */
      PolyUOp *ret = poly_uop_substitute(ctx, found_red, ne, tne, n_ne);
      PolyUOp *ret_mul = ret->src[0];
      if (ret_mul->op == POLY_OP_CAST && ret_mul->n_src > 0) ret_mul = ret_mul->src[0];
      PolyUOp *srcs[2] = {ret_mul->src[0], ret_mul->src[1]};

      /* Apply swizzle permutations (postrange.py:286):
       * srcs[k] = x.substitute(dict(zip(tne, [ne[i] for i in argsort(p)])))
       * where p = tc.permutes_for_shape_str(tc.base_shape_str()) */
      const char *bss[32];
      int n_bss = poly_tc_base_shape_str(tc, bss, 32);

      int perm0[32], perm1[32];
      poly_tc_permute_for_shape_str(tc, 0, bss, n_bss, perm0, 32);
      poly_tc_permute_for_shape_str(tc, 1, bss, n_bss, perm1, 32);

      /* argsort(perm): inverse permutation. argsort[j] = i where perm[i] = j */
      int argsort0[32], argsort1[32];
      for (int i = 0; i < n_bss; i++) {
        argsort0[i] = i;
        argsort1[i] = i;
      }
      for (int i = 0; i < n_bss; i++) {
        if (perm0[i] < n_bss) argsort0[perm0[i]] = i;
        if (perm1[i] < n_bss) argsort1[perm1[i]] = i;
      }

      /* Build reordered ne[] for each source (postrange.py:286):
       * ne_reordered[i] = ne[argsort[i]] -- permute the ne list itself */
      PolyUOp *ne_reordered0[32], *ne_reordered1[32];
      for (int i = 0; i < n_ne; i++) {
        int idx0 = (i < n_bss) ? argsort0[i] : i;
        int idx1 = (i < n_bss) ? argsort1[i] : i;
        ne_reordered0[i] = (idx0 >= 0 && idx0 < n_ne) ? ne[idx0] : ne[i];
        ne_reordered1[i] = (idx1 >= 0 && idx1 < n_ne) ? ne[idx1] : ne[i];
      }
      srcs[0] = poly_uop_substitute(ctx, srcs[0], tne, ne_reordered0, n_ne);
      srcs[1] = poly_uop_substitute(ctx, srcs[1], tne, ne_reordered1, n_ne);

      /* Compute tc_reduce_axes and tc_upcast_axes (postrange.py:289-295) */
      const char *shape_str[64];
      int n_ss = sched_shape_str(s, shape_str, 64);

      const char *bua[32];
      int n_bua = poly_tc_base_upcast_axes(tc, bua, 32);

      /* tc_reduce_axes: axis ids for "r0","r1",... in scheduler shape_str */
      int tc_reduce_axis_ids[16];
      int n_tc_ra = 0;
      for (int ri = 0; ri < n_ra; ri++) {
        char rname[32];
        int rname_len = snprintf(rname, sizeof(rname), "r%d", ri);
        if (rname_len < 0 || rname_len >= (int)sizeof(rname)) continue;
        for (int si = 0; si < n_ss; si++) {
          if (strcmp(shape_str[si], rname) == 0) {
            tc_reduce_axis_ids[n_tc_ra++] = (int)poly_range_axis_id(s->rngs[si]->arg);
            break;
          }
        }
      }

      /* tc_upcast_axes[dim]: first log2(ept[dim]) entries of base_upcast_axes, mapped to axis ids
       */
      int64_t upcast_pairs[3][16][2];
      int n_upcast[3] = {0, 0, 0};
      for (int dim = 0; dim < 3; dim++) {
        int need = 0;
        {
          int v = tc->elements_per_thread[dim];
          while (v > 1) {
            need++;
            v /= 2;
          }
        }
        for (int ui = 0; ui < need && ui < n_bua; ui++) {
          for (int si = 0; si < n_ss; si++) {
            if (strcmp(shape_str[si], bua[ui]) == 0) {
              upcast_pairs[dim][n_upcast[dim]][0] = poly_range_axis_id(s->rngs[si]->arg);
              upcast_pairs[dim][n_upcast[dim]][1] = 2;
              n_upcast[dim]++;
              break;
            }
          }
        }
      }

      /* Build CONTRACT + WMMA + UNROLL (postrange.py:301-306) */
      PolyDType vec_in0 = poly_dtype_vec(tc->dtype_in, tc->elements_per_thread[0]);
      PolyDType vec_in1 = poly_dtype_vec(tc->dtype_in, tc->elements_per_thread[1]);
      PolyDType vec_out = poly_dtype_vec(tc->dtype_out, tc->elements_per_thread[2]);

      PolyUOp *ca_src[1] = {srcs[0]};
      PolyUOp *contract_a = poly_uop_tagged(
          ctx, POLY_OP_CONTRACT, vec_in0, ca_src, 1,
          poly_arg_pair_tuple(upcast_pairs[0], n_upcast[0]), 1
      );
      PolyUOp *cb_src[1] = {srcs[1]};
      PolyUOp *contract_b = poly_uop_tagged(
          ctx, POLY_OP_CONTRACT, vec_in1, cb_src, 1,
          poly_arg_pair_tuple(upcast_pairs[1], n_upcast[1]), 1
      );

      /* Zero accumulator */
      PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, tc->dtype_out, poly_arg_float(0.0));
      PolyUOp *zero_elems[16];
      for (int i = 0; i < tc->elements_per_thread[2] && i < 16; i++)
        zero_elems[i] = zero;
      PolyUOp *zero_vec = poly_uop(
          ctx, POLY_OP_VECTORIZE, vec_out, zero_elems, tc->elements_per_thread[2], poly_arg_none()
      );

      PolyUOp *wmma_srcs[3] = {contract_a, contract_b, zero_vec};
      PolyUOp *wmma = poly_uop_tagged(
          ctx, POLY_OP_WMMA, vec_out, wmma_srcs, 3,
          poly_arg_tensor_core(tc->intrinsic_name, tc->dims, tc->threads), 1
      );

      PolyUOp *unroll_src[1] = {wmma};
      PolyUOp *tc_uop = poly_uop_tagged(
          ctx, POLY_OP_UNROLL, tc->dtype_out, unroll_src, 1,
          poly_arg_pair_tuple(upcast_pairs[2], n_upcast[2]), 1
      );

      /* Preserve extra reduce ranges not consumed by TC (postrange.py:309-310) */
      int rs2 = range_start_for_op(POLY_OP_REDUCE);
      PolyUOp *extra_rngs[16];
      int n_extra = 0;
      for (int i = rs2; i < found_red->n_src && n_extra < 16; i++) {
        if (found_red->src[i]->op != POLY_OP_RANGE) continue;
        int64_t aid = poly_range_axis_id(found_red->src[i]->arg);
        bool in_tc = false;
        for (int r = 0; r < n_tc_ra; r++)
          if (tc_reduce_axis_ids[r] == (int)aid) {
            in_tc = true;
            break;
          }
        if (!in_tc) extra_rngs[n_extra++] = found_red->src[i];
      }
      if (n_extra > 0) {
        PolyUOp *red_srcs[18];
        red_srcs[0] = tc_uop;
        for (int i = 0; i < n_extra; i++)
          red_srcs[i + 1] = extra_rngs[i];
        PolyArg red_arg = {.kind = POLY_ARG_OPS, .ops = POLY_OP_ADD};
        tc_uop = poly_uop(ctx, POLY_OP_REDUCE, tc_uop->dtype, red_srcs, n_extra + 1, red_arg);
      }

      /* Substitute found_red -> tc_uop in AST */
      PolyUOp *from_r[1] = {found_red};
      PolyUOp *to_r[1] = {tc_uop};
      s->ast = poly_uop_substitute(ctx, s->ast, from_r, to_r, 1);
    }

    if (tc_axes_out) {
      tc_axes_out[0] = axes[0];
      tc_axes_out[1] = axes[1];
      tc_axes_out[2] = axes[2];
    }
    sched_refresh(s);
    return true;
  }

  return false;
}

/* hand_coded_optimizations (heuristic.py:8-190, CPU-relevant subset) */
static PolyUOp *poly_apply_opts_heuristic(PolyCtx *ctx, PolyUOp *sink, PolyRendererCaps caps) {
  /* tinygrad apply_opts guard (postrange.py:352): skip heuristic for multi-block kernels. */
  {
    int n_topo = 0;
    PolyUOp **topo = poly_toposort(ctx, sink, &n_topo);
    for (int i = 0; i < n_topo; i++) {
      if (topo[i]->op == POLY_OP_STAGE) return sink;
    }
  }

  OptScheduler s;
  sched_init(&s, ctx, sink);
  if (!sched_can_optimize(&s)) return sink;

  /* == Tensor core optimization (heuristic.py:28-46) ==
   * Try TC before other heuristics. On success, return immediately. */
  if (caps.n_tensor_cores > 0) {
    /* Count reduce axes */
    int n_reduce = 0;
    for (int i = 0; i < s.n_rngs; i++) {
      if (s.types[i] == POLY_AXIS_GROUP_REDUCE || s.types[i] == POLY_AXIS_REDUCE) n_reduce++;
    }
    int tc_opt_env = poly_getenv_int("POLY_TC_OPT", 0);
    int use_tc_env = poly_getenv_int("POLY_USE_TC", 1);

    if (use_tc_env > 0 && (n_reduce == 1 || tc_opt_env >= 1)) {
      OptScheduler tk;
      sched_copy(&tk, &s);
      PolyUOp *tc_axes[3];
      bool tc_ok = sched_apply_tc_opt(
          &tk, 0, -1, tc_opt_env, use_tc_env, caps.tensor_cores, caps.n_tensor_cores, tc_axes
      );
      if (tc_ok) {
        /* Post-TC upcasts on M and N (heuristic.py:39-45) */
        for (int tc_dim = 1; tc_dim >= 0; tc_dim--) {
          int64_t bound = 0;
          if (tc_axes[tc_dim] && tc_axes[tc_dim]->n_src > 0 &&
              tc_axes[tc_dim]->src[0]->op == POLY_OP_CONST)
            bound = tc_axes[tc_dim]->src[0]->arg.i;
          if (bound <= 1) continue;
          int szs[] = {5, 4, 3, 2};
          for (int si = 0; si < 4; si++) {
            if (bound % szs[si] == 0) {
              int idx = -1;
              for (int ri = 0; ri < tk.n_rngs; ri++) {
                if (tk.rngs[ri] == tc_axes[tc_dim]) {
                  idx = ri;
                  break;
                }
              }
              if (idx >= 0)
                tc_axes[tc_dim] =
                    sched_shift_to(&tk, tk.rngs[idx], szs[si], POLY_AXIS_UPCAST, false);
              break;
            }
          }
        }
        return tk.ast;
      }
    }
  }

  /* == Group for reduces (tinygrad heuristic.py:101-110) ==
   * Try GROUPTOP(16) on the first few REDUCE axes when the output footprint
   * is small enough. If grouping succeeds, stop here like tinygrad and do not
   * fall through into the later reduce-unroll heuristic. */
  if (caps.has_local && sched_output_prod_upcastable(&s) <= 2048) {
    for (int axis = 0; axis < 3; axis++) {
      int reduce_axes[SCHED_MAX_RNGS];
      int n_reduce_axes = 0;
      for (int i = 0; i < s.n_rngs && n_reduce_axes < SCHED_MAX_RNGS; i++) {
        if (s.types[i] == POLY_AXIS_REDUCE) reduce_axes[n_reduce_axes++] = i;
      }
      if (axis >= n_reduce_axes) break;
      int ridx = reduce_axes[axis];
      if (ridx < s.n_rngs && s.shape[ridx] > 1 &&
          sched_shift_to(&s, s.rngs[ridx], 16, POLY_AXIS_GROUP_REDUCE, true)) {
        break;
      }
    }
  }

  if (sched_has_axis_type(&s, POLY_AXIS_GROUP_REDUCE)) return s.ast;

  /* == Masked upcast (heuristic.py:96-105) ==
   * Upcast small dims (<=7) that appear in WHERE gates */
  {
    int up_dims[SCHED_MAX_RNGS];
    int n_up = sched_upcastable_dims(&s, up_dims, SCHED_MAX_RNGS);
    int to_upcast[SCHED_MAX_RNGS];
    int n_to_upcast = 0;
    int n_topo = 0;
    PolyUOp **topo = poly_toposort(ctx, s.ast, &n_topo);

    for (int ui = 0; ui < n_up; ui++) {
      int axis = up_dims[ui];
      if (s.shape[axis] > 7) continue;
      bool is_masked = false;
      for (int ti = 0; ti < n_topo && !is_masked; ti++) {
        if (topo[ti]->op != POLY_OP_WHERE) continue;
        /* Match tinygrad heuristic.py: the axis is "masked" when the active
         * range appears anywhere in the WHERE condition backward slice. */
        if (poly_uop_in_ranges(ctx, topo[ti]->src[0], s.rngs[axis])) is_masked = true;
      }
      if (!is_masked) continue;
      /* Check total upcast product stays <= 49 (7*7) */
      int64_t prod = s.shape[axis];
      for (int j = 0; j < n_to_upcast; j++)
        prod *= s.shape[to_upcast[j]];
      if (prod <= 49 && n_to_upcast < SCHED_MAX_RNGS) to_upcast[n_to_upcast++] = axis;
    }

    /* Apply in reverse order (matching tinygrad) */
    for (int i = n_to_upcast - 1; i >= 0; i--) {
      int axis = to_upcast[i];
      if (axis < s.n_rngs && s.shape[axis] > 1)
        sched_shift_to(&s, s.rngs[axis], s.shape[axis], POLY_AXIS_UPCAST, false);
    }
  }

  /* == Multi-axis UPCAST with stride scoring (heuristic.py:107-133) == */
  {
    bool upcasted_axis[SCHED_MAX_RNGS] = {false};

    while (sched_output_prod_upcastable(&s) >= 1024 && sched_upcast_size(&s) < 32) {
      int up_dims[SCHED_MAX_RNGS];
      int n_up = sched_upcastable_dims(&s, up_dims, SCHED_MAX_RNGS);

      /* Score each candidate (num_strides, sum_strides, axis, amount) */
      typedef struct {
        int num_strides;
        int64_t sum_strides;
        int axis;
        int amount;
      } UpChoice;
      UpChoice choices[SCHED_MAX_RNGS * 2];
      int n_choices = 0;

      int amounts[] = {3, 4};
      for (int ui = 0; ui < n_up; ui++) {
        int axis = up_dims[ui];
        if (upcasted_axis[axis]) continue;

        for (int ai = 0; ai < 2; ai++) {
          int amount = amounts[ai];
          if (s.shape[axis] % amount != 0) continue;

          PolyUOp *rng = s.rngs[axis];

          /* Expanded axis check (heuristic.py:117-118):
           * Must have a buffer where rng is NOT in index but all UPCAST/UNROLL rngs ARE */
          bool has_expanded_buf = false;
          if (s.has_reach) {
            /* Build mask of all current UPCAST/UNROLL ranges */
            uint64_t upcast_mask = 0;
            for (int ri = 0; ri < s.n_rngs; ri++) {
              if (s.types[ri] == POLY_AXIS_UPCAST || s.types[ri] == POLY_AXIS_UNROLL)
                upcast_mask |= (1ULL << ri);
            }
            for (int bi = 0; bi < s.n_bufs && !has_expanded_buf; bi++) {
              if (s.buf_reach[bi] & (1ULL << axis)) continue; /* rng IS in this buf's index */
              /* Check all existing UPCAST/UNROLL ranges are in this buf's index */
              if ((s.buf_reach[bi] & upcast_mask) == upcast_mask) has_expanded_buf = true;
            }
          }
          if (!has_expanded_buf) continue;

          /* Count strides (heuristic.py:119-127) */
          int num_strides = 0;
          int64_t sum_strides = 0;
          for (int bi = 0; bi < s.n_bufs; bi++) {
            PolyUOp *idx_uop = s.bufs[bi];
            if (idx_uop->n_src < 2) continue;
            /* Pinned heuristic.py:118-128 scores the address projection, not
             * the validity predicate carried by an Invalid-bearing WHERE. */
            PolyUOp *idx_expr = poly_index_get_idx(ctx, idx_uop->src[1]);
            if (!idx_expr) continue;

            /* Check if rng is in backward slice */
            if (s.has_reach && (s.buf_reach[bi] & (1ULL << axis))) num_strides++;

            /* Split on ADD and extract stride for this rng */
            PolyUOp *addends[256];
            int n_add = split_uop_add(idx_expr, addends, 256);
            for (int j = 0; j < n_add; j++) {
              PolyUOp *c = addends[j];
              if (c == rng) {
                sum_strides += 1;
              } else if (c->op == POLY_OP_MUL && c->n_src == 2) {
                if (c->src[0] == rng && c->src[1]->op == POLY_OP_CONST &&
                    c->src[1]->arg.kind == POLY_ARG_INT)
                  sum_strides += c->src[1]->arg.i;
                else if (c->src[1] == rng && c->src[0]->op == POLY_OP_CONST && c->src[0]->arg.kind == POLY_ARG_INT)
                  sum_strides += c->src[0]->arg.i;
              }
            }
          }

          if (n_choices < (int)(sizeof(choices) / sizeof(choices[0])))
            choices[n_choices++] = (UpChoice){num_strides, sum_strides, axis, amount};
        }
      }

      if (n_choices == 0) break;

      /* Sort: lowest (num_strides, sum_strides) first */
      for (int i = 0; i < n_choices - 1; i++) {
        for (int j = i + 1; j < n_choices; j++) {
          bool swap = false;
          if (choices[j].num_strides < choices[i].num_strides)
            swap = true;
          else if (choices[j].num_strides == choices[i].num_strides &&
                   choices[j].sum_strides < choices[i].sum_strides)
            swap = true;
          if (swap) {
            UpChoice tmp = choices[i];
            choices[i] = choices[j];
            choices[j] = tmp;
          }
        }
      }

      int best_axis = choices[0].axis;
      int best_amount = choices[0].amount;
      if (best_axis < s.n_rngs && s.shape[best_axis] > 1)
        sched_shift_to(&s, s.rngs[best_axis], best_amount, POLY_AXIS_UPCAST, false);
      /* Mark the original axis as upcasted -- after refresh, find the axis by matching the rng */
      /* Since indices shift after refresh, we track by setting the flag before refresh */
      /* Actually, upcasted_axis tracks by index which changes after shift_to+refresh.
       * Use a simple counter limit instead (tinygrad limits upcast_size < 32) */
      (void)upcasted_axis; /* The while-loop condition handles termination */
      /* Prevent infinite loop: break if nothing changed */
      if (sched_upcast_size(&s) < 32 && n_choices > 0) {
        upcasted_axis[best_axis] = true;
      }
    }
  }

  /* == Reduce UNROLL (heuristic.py:135-149) == */
  if (s.has_reduce) {
    int unroll_dims[SCHED_MAX_RNGS];
    int n_unroll = sched_unrollable_dims(&s, unroll_dims, SCHED_MAX_RNGS);

    if (n_unroll > 0 &&
        (sched_upcast_size(&s) <= 4 || !sched_has_axis_type(&s, POLY_AXIS_UNROLL)) &&
        sched_upcast_size(&s) < 64) {
      int last = unroll_dims[n_unroll - 1];
      int64_t last_sz = s.shape[last];

      if (last_sz <= 32) {
        /* Unroll fully (amount = full size) */
        if (last < s.n_rngs && last_sz > 1)
          sched_shift_to(&s, s.rngs[last], last_sz, POLY_AXIS_UNROLL, false);
        /* If small, try unrolling a second reduce dim */
        n_unroll = sched_unrollable_dims(&s, unroll_dims, SCHED_MAX_RNGS);
        if (n_unroll > 0 && last_sz <= 3 && s.shape[unroll_dims[n_unroll - 1]] <= 3) {
          int last2 = unroll_dims[n_unroll - 1];
          if (last2 < s.n_rngs && s.shape[last2] > 1)
            sched_shift_to(&s, s.rngs[last2], s.shape[last2], POLY_AXIS_UNROLL, false);
        }
      } else {
        /* Partial unroll by 4 if divisible */
        if (last_sz % 4 == 0 && last < s.n_rngs)
          sched_shift_to(&s, s.rngs[last], 4, POLY_AXIS_UNROLL, false);
      }
    }
  }

  /* == Default upcast fallback (heuristic.py:151-154) ==
   * If nothing upcasted and last upcastable dim % upcast_amount == 0,
   * upcast by that amount. Use max_vec_width from caps (4 for SSE, 8 for AVX2). */
  if (!sched_upcasted(&s)) {
    int upcast_amount = (caps.max_vec_width >= 8) ? 8 : 4;
    int up_dims[SCHED_MAX_RNGS];
    int n_up = sched_upcastable_dims(&s, up_dims, SCHED_MAX_RNGS);
    if (n_up > 0) {
      int last = up_dims[n_up - 1];
      /* Try preferred width first, fall back to 4 if not divisible */
      if (s.shape[last] % upcast_amount == 0 && last < s.n_rngs)
        sched_shift_to(&s, s.rngs[last], upcast_amount, POLY_AXIS_UPCAST, false);
      else if (upcast_amount > 4 && s.shape[last] % 4 == 0 && last < s.n_rngs)
        sched_shift_to(&s, s.rngs[last], 4, POLY_AXIS_UPCAST, false);
    }
  }

  /* == Local groups (heuristic.py:160-175 subset) ==
   * Port the tinygrad local scheduling block for backends with workgroup
   * locals. This is what splits large LOOP/GLOBAL axes into LOOP x LOCAL for
   * WebGPU masked kernels like triu(9,9). */
  if (caps.has_local) {
    LocalAxisRank ranked[SCHED_MAX_RNGS];
    int n_ranked = 0;
    for (int axis = 0; axis < s.n_rngs && n_ranked < SCHED_MAX_RNGS; axis++) {
      if (!(s.types[axis] == POLY_AXIS_GLOBAL || s.types[axis] == POLY_AXIS_LOOP)) continue;
      if (s.shape[axis] <= 1) continue;
      bool expanded = false;
      if (s.has_reach) {
        for (int bi = 0; bi < s.n_bufs; bi++) {
          if ((s.buf_reach[bi] & (1ULL << axis)) == 0) {
            expanded = true;
            break;
          }
        }
      }
      ranked[n_ranked++] = (LocalAxisRank){.expanded = expanded, .axis = axis};
    }
    qsort(ranked, (size_t)n_ranked, sizeof(LocalAxisRank), cmp_local_axis_rank);

    LocalChoice to_local[SCHED_MAX_RNGS];
    int n_to_local = 0;
    for (int ri = 0; ri < n_ranked && n_to_local < SCHED_MAX_RNGS; ri++) {
      int axis = ranked[ri].axis;
      int64_t local_size = 1;
      for (int i = 0; i < n_to_local; i++)
        local_size *= to_local[i].size;

      int candidates[6];
      int n_candidates = 0;
      if (axis == 0) candidates[n_candidates++] = 32;
      candidates[n_candidates++] = 16;
      candidates[n_candidates++] = 8;
      candidates[n_candidates++] = 4;
      candidates[n_candidates++] = 3;
      candidates[n_candidates++] = 2;

      int chosen = 0;
      for (int ci = 0; ci < n_candidates; ci++) {
        int cand = candidates[ci];
        if ((s.shape[axis] % cand) == 0 && local_size * cand <= 128) {
          chosen = cand;
          break;
        }
      }
      if (chosen > 0) to_local[n_to_local++] = (LocalChoice){.axis = axis, .size = chosen};
    }

    if (n_to_local > 0) {
      /* Pinned tinygrad heuristic.py:174-175 applies sorted(to_local[:3]):
       * preserve ranked selection before sorting the chosen axes. */
      int n_apply = n_to_local < 3 ? n_to_local : 3;
      qsort(to_local, (size_t)n_apply, sizeof(LocalChoice), cmp_local_choice_axis);
      int deleted_shape = 0;
      for (int i = 0; i < n_apply; i++) {
        int axis = to_local[i].axis - deleted_shape;
        if (axis < 0 || axis >= s.n_rngs) continue;
        bool will_delete_shape = to_local[i].size == s.shape[axis];
        if ((s.types[axis] == POLY_AXIS_GLOBAL || s.types[axis] == POLY_AXIS_LOOP) &&
            s.shape[axis] > 1)
          sched_shift_to(&s, s.rngs[axis], to_local[i].size, POLY_AXIS_LOCAL, false);
        if (will_delete_shape) deleted_shape++;
      }
    }
  }

  /* == CPU THREAD axis (tinygrad heuristic.py:180-190) ==
   * ClangRenderer has has_threads=true, then gpudims.py replaces the THREAD
   * axis with a runtime DEFINE_VAR("core_id"). Keep this after local grouping
   * just like tinygrad's final heuristic block. */
  if (caps.has_threads && caps.max_threads > 1 && !sched_has_axis_type(&s, POLY_AXIS_THREAD)) {
    int candidates[] = {32, 16, 12, 8, 6, 5, 4, 3, 2};
    int64_t full_prod = sched_full_shape_prod(&s);
    for (int ci = 0; ci < (int)(sizeof(candidates) / sizeof(candidates[0])); ci++) {
      int threads = candidates[ci];
      if (threads > caps.max_threads) continue;
      if (full_prod / (128LL << 10) < threads) continue;
      for (int axis = 0; axis < s.n_rngs; axis++) {
        if (s.types[axis] != POLY_AXIS_LOOP) continue;
        if (s.shape[axis] <= 1 || (s.shape[axis] % threads) != 0) continue;
        if (sched_shift_to(&s, s.rngs[axis], threads, POLY_AXIS_THREAD, true)) {
          ci = (int)(sizeof(candidates) / sizeof(candidates[0]));
          break;
        }
      }
    }
  }

  return s.ast;
}

/* BEAM search optimizer *
 * Explores the optimization space by trying many candidate optimizations,
 * compiling and timing each, and keeping the top-k. Finds better
 * optimizations than the heuristic for non-trivial kernels.
 *
 * Action space: UPCAST and UNROLL with various axis/amount combos.
 * For each beam member, enumerate all valid actions, compile candidates,
 * time them on actual hardware, sort by execution time, keep top beam_width.
 *
 * Requires native runtime (fork+clang for compilation, clock_gettime for
 * timing). Disabled in WASM builds. Future: use WASM JIT backend for
 * compile+time, IndexedDB for cache.
 */
/* Shallow-copy an OptScheduler. Used by TC optimization and BEAM search.
 * Safe because sched_shift_to creates new UOps via poly_uop_substitute. */
static void sched_copy(OptScheduler *dst, const OptScheduler *src) {
  *dst = *src;
}

#ifndef __EMSCRIPTEN__

typedef enum { POLY_OPT_UPCAST, POLY_OPT_UNROLL } PolyOptOp;

typedef struct {
  PolyOptOp op;
  int axis;
  int64_t amount;
} PolyBeamAction;

/* Static action table */
static const int64_t beam_upcast_amounts[] = {2, 3, 4, 5, 7, 8};
static const int beam_n_upcast_amounts = 6;
static const int64_t beam_unroll_amounts[] = {2, 3, 4, 7};
static const int beam_n_unroll_amounts = 4;
#define BEAM_MAX_AXIS 8
#define BEAM_MAX_ACTIONS ((6 * BEAM_MAX_AXIS) + (4 * 5)) /* 68 */
#define BEAM_MAX_BEAM 16
#define BEAM_MAX_ITERS 5
#define BEAM_MAX_CANDIDATES (BEAM_MAX_BEAM * BEAM_MAX_ACTIONS)

typedef struct {
  OptScheduler sched;
  double time_us;
  PolyBeamAction actions[BEAM_MAX_ITERS];
  int n_actions;
} BeamEntry;

typedef struct {
  OptScheduler sched;
  double time_us;
  PolyBeamAction actions[BEAM_MAX_ITERS];
  int n_actions;
} BeamCandidate;

/* Try to apply a single BEAM action to a scheduler. Returns true on success. */
static bool sched_apply_action(OptScheduler *s, PolyBeamAction act) {
  if (!sched_can_optimize(s)) return false;
  int dims[SCHED_MAX_RNGS];
  int n_dims;

  if (act.op == POLY_OPT_UPCAST) {
    n_dims = sched_upcastable_dims(s, dims, SCHED_MAX_RNGS);
  } else {
    n_dims = sched_unrollable_dims(s, dims, SCHED_MAX_RNGS);
  }

  if (act.axis >= n_dims) return false;
  int idx = dims[act.axis];
  if (idx >= s->n_rngs) return false;
  if (s->shape[idx] <= 1) return false;
  if (s->shape[idx] % act.amount != 0) return false;

  /* Limit total upcast+unroll product to prevent code explosion.
   * 64 is reasonable: e.g. UPCAST 4 on two axes = 16, or UPCAST 8 + UNROLL 4 = 32. */
  int64_t cur_prod = sched_upcast_size(s);
  if (cur_prod * act.amount > 64) return false;

  PolyAxisType new_type = (act.op == POLY_OPT_UPCAST) ? POLY_AXIS_UPCAST : POLY_AXIS_UNROLL;
  PolyUOp *result = sched_shift_to(s, s->rngs[idx], act.amount, new_type, false);
  return result != NULL;
}

/* Time a single kernel execution using clock_gettime (CLOCK_MONOTONIC). */
static double time_us_now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

/* Compile a kernel AST through the full post-optimization pipeline,
 * render to C, compile with clang, allocate test buffers, and time execution.
 * Returns median time in microseconds. Returns INFINITY on failure. */
static double beam_compile_and_time(PolyCtx *ctx, PolyUOp *sink, PolyRewriteOpts opts, int reps) {
  /* Run through the rest of the codegen pipeline (post-optimization stages).
   * Setting optimize=false skips the preprocessing+apply_opts pass since
   * opts have already been applied by the BEAM search. */
  PolyRewriteOpts post_opts = opts;
  post_opts.optimize = false;
  post_opts.beam_width = 0;
  sink = poly_full_rewrite_to_sink_ex(ctx, sink, post_opts);
  /* Control flow now runs inside poly_full_rewrite_to_sink_ex (tinygrad parity). */

  /* Linearize */
  int n_uops = 0;
  PolyUOp **uops = poly_linearize_rewritten(ctx, sink, &n_uops);
  if (!uops || n_uops == 0) return INFINITY;

  /* UOp count filter: skip huge kernels */
  if (n_uops > 3000) {
    free(uops);
    return INFINITY;
  }

  /* Render C */
  char fn_name[64];
  snprintf(fn_name, sizeof(fn_name), "beam_%d", (int)(uintptr_t)sink & 0xFFFF);
  char *source = poly_render_c(uops, n_uops, fn_name);
  free(uops);
  if (!source) return INFINITY;

  /* Compile */
  PolyProgram *prog = poly_compile_c(source, fn_name);
  free(source);
  if (!prog) return INFINITY;

  /* Collect PARAM count and buffer sizes from the sink's toposort */
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, sink, &n_topo);
  int n_params = 0;
  int64_t param_sizes[64];
  for (int i = 0; i < n_topo; i++) {
    if (topo[i]->op == POLY_OP_PARAM && n_params < 64) {
      /* Estimate buffer size from dtype ptr size field */
      int64_t sz = topo[i]->dtype.ptr_size;
      if (sz <= 0) sz = 1024; /* default */
      param_sizes[n_params] = sz;
      n_params++;
    }
  }

  if (n_params == 0) {
    poly_program_destroy(prog);
    return INFINITY;
  }

  /* Allocate test buffers (random float32 data) */
  void *bufs[64];
  for (int i = 0; i < n_params; i++) {
    int64_t nbytes = param_sizes[i] * 4; /* float32 */
    if (nbytes <= 0) nbytes = 4096;
    bufs[i] = calloc(1, (size_t)nbytes);
    /* Fill with small random values to avoid NaN/inf in transcendentals */
    float *fp = (float *)bufs[i];
    int n_elems = (int)(nbytes / 4);
    for (int j = 0; j < n_elems; j++)
      fp[j] = 0.1f + (float)(j % 100) * 0.01f;
  }

  /* Warm up */
  poly_program_call(prog, bufs, n_params);

  /* Time execution */
  double times[16];
  if (reps > 16) reps = 16;
  if (reps < 1) reps = 1;
  for (int r = 0; r < reps; r++) {
    double t0 = time_us_now();
    poly_program_call(prog, bufs, n_params);
    double t1 = time_us_now();
    times[r] = t1 - t0;
  }

  /* Sort times, take median */
  for (int i = 0; i < reps - 1; i++)
    for (int j = i + 1; j < reps; j++)
      if (times[j] < times[i]) {
        double t = times[i];
        times[i] = times[j];
        times[j] = t;
      }
  double median = times[reps / 2];

  /* Cleanup */
  for (int i = 0; i < n_params; i++)
    free(bufs[i]);
  poly_program_destroy(prog);

  return median;
}

/* Disk cache for BEAM results */

/* FNV-1a hash over the AST toposort (structural hash for cache key) */
static uint64_t beam_ast_hash(PolyCtx *ctx, PolyUOp *sink) {
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, sink, &n_topo);
  uint64_t h = 0xcbf29ce484222325ULL;
  for (int i = 0; i < n_topo; i++) {
    h ^= (uint64_t)topo[i]->op;
    h *= 0x100000001b3ULL;
    h ^= (uint64_t)topo[i]->dtype.bitsize;
    h *= 0x100000001b3ULL;
    h ^= poly_arg_hash(topo[i]->arg);
    h *= 0x100000001b3ULL;
    h ^= (uint64_t)topo[i]->n_src;
    h *= 0x100000001b3ULL;
  }
  return h;
}

static int beam_cache_dir(char *dir, int cap) {
  const char *xdg = getenv("XDG_CACHE_HOME");
  const char *home = getenv("HOME");
  if (xdg && xdg[0])
    snprintf(dir, cap, "%s/polygrad/beam", xdg);
  else if (home && home[0])
    snprintf(dir, cap, "%s/.cache/polygrad/beam", home);
  else
    return -1;

  /* mkdir -p: create parent dirs */
  char parent[512];
  snprintf(parent, sizeof(parent), "%s", dir);
  char *s = parent + 1;
  while (*s) {
    if (*s == '/') {
      *s = '\0';
      mkdir(parent, 0755);
      *s = '/';
    }
    s++;
  }
  if (mkdir(dir, 0755) == -1 && errno != EEXIST) return -1;
  return 0;
}

/* Cache entry: [n_actions:uint8][actions: n * (op:uint8, axis:uint8, amount:int64)] */
static bool beam_cache_load(uint64_t key, PolyBeamAction *actions, int *n_actions) {
  char dir[512], path[576];
  if (beam_cache_dir(dir, sizeof(dir)) != 0) return false;
  snprintf(path, sizeof(path), "%s/%016llx.bin", dir, (unsigned long long)key);

  FILE *f = fopen(path, "rb");
  if (!f) return false;

  uint8_t n;
  if (fread(&n, 1, 1, f) != 1 || n > BEAM_MAX_ITERS) {
    fclose(f);
    return false;
  }
  *n_actions = n;
  for (int i = 0; i < n; i++) {
    uint8_t op_byte, axis_byte;
    int64_t amount;
    if (fread(&op_byte, 1, 1, f) != 1) {
      fclose(f);
      return false;
    }
    if (fread(&axis_byte, 1, 1, f) != 1) {
      fclose(f);
      return false;
    }
    if (fread(&amount, sizeof(amount), 1, f) != 1) {
      fclose(f);
      return false;
    }
    actions[i] = (PolyBeamAction){.op = (PolyOptOp)op_byte, .axis = axis_byte, .amount = amount};
  }
  fclose(f);
  return true;
}

static void beam_cache_save(uint64_t key, const PolyBeamAction *actions, int n_actions) {
  char dir[512], path[576];
  if (beam_cache_dir(dir, sizeof(dir)) != 0) return;
  snprintf(path, sizeof(path), "%s/%016llx.bin", dir, (unsigned long long)key);

  FILE *f = fopen(path, "wb");
  if (!f) return;
  uint8_t n = (uint8_t)n_actions;
  fwrite(&n, 1, 1, f);
  for (int i = 0; i < n_actions; i++) {
    uint8_t op_byte = (uint8_t)actions[i].op;
    uint8_t axis_byte = (uint8_t)actions[i].axis;
    fwrite(&op_byte, 1, 1, f);
    fwrite(&axis_byte, 1, 1, f);
    fwrite(&actions[i].amount, sizeof(actions[i].amount), 1, f);
  }
  fclose(f);
}

/* Main BEAM search loop */

static int beam_candidate_cmp(const void *a, const void *b) {
  const BeamCandidate *ca = (const BeamCandidate *)a;
  const BeamCandidate *cb = (const BeamCandidate *)b;
  if (ca->time_us < cb->time_us) return -1;
  if (ca->time_us > cb->time_us) return 1;
  return 0;
}

static PolyUOp *poly_beam_search(
    PolyCtx *ctx,
    PolyUOp *sink,
    int beam_width,
    PolyRewriteOpts opts
) {
  if (beam_width <= 0) return sink;
  if (beam_width > BEAM_MAX_BEAM) beam_width = BEAM_MAX_BEAM;

  /* Check disk cache */
  uint64_t cache_key = beam_ast_hash(ctx, sink);
  PolyBeamAction cached_actions[BEAM_MAX_ITERS];
  int cached_n = 0;
  if (beam_cache_load(cache_key, cached_actions, &cached_n) && cached_n > 0) {
    /* Replay cached actions */
    OptScheduler s;
    sched_init(&s, ctx, sink);
    if (!sched_can_optimize(&s)) return sink;
    for (int i = 0; i < cached_n; i++) {
      OptScheduler copy;
      sched_copy(&copy, &s);
      if (!sched_apply_action(&copy, cached_actions[i])) break;
      s = copy;
    }
    return s.ast;
  }

  /* Initialize beam with unoptimized baseline */
  BeamEntry *beam = (BeamEntry *)calloc(BEAM_MAX_BEAM, sizeof(BeamEntry));
  sched_init(&beam[0].sched, ctx, sink);
  if (!sched_can_optimize(&beam[0].sched)) {
    free(beam);
    return sink;
  }
  beam[0].time_us = INFINITY;
  beam[0].n_actions = 0;
  int beam_size = 1;

  /* Time the baseline */
  beam[0].time_us = beam_compile_and_time(ctx, beam[0].sched.ast, opts, 3);

  /* Build action list */
  PolyBeamAction all_actions[BEAM_MAX_ACTIONS];
  int n_actions = 0;
  for (int axis = 0; axis < BEAM_MAX_AXIS; axis++) {
    for (int ai = 0; ai < beam_n_upcast_amounts; ai++) {
      all_actions[n_actions++] =
          (PolyBeamAction){.op = POLY_OPT_UPCAST, .axis = axis, .amount = beam_upcast_amounts[ai]};
    }
  }
  for (int axis = 0; axis < 5; axis++) {
    for (int ai = 0; ai < beam_n_unroll_amounts; ai++) {
      all_actions[n_actions++] =
          (PolyBeamAction){.op = POLY_OPT_UNROLL, .axis = axis, .amount = beam_unroll_amounts[ai]};
    }
  }

  BeamCandidate *candidates = (BeamCandidate *)calloc(BEAM_MAX_CANDIDATES, sizeof(BeamCandidate));

  for (int iter = 0; iter < BEAM_MAX_ITERS; iter++) {
    int n_cand = 0;

    /* Generate candidates from all beam members */
    for (int b = 0; b < beam_size; b++) {
      if (beam[b].n_actions >= BEAM_MAX_ITERS) continue;
      for (int a = 0; a < n_actions && n_cand < BEAM_MAX_CANDIDATES; a++) {
        OptScheduler copy;
        sched_copy(&copy, &beam[b].sched);
        if (!sched_apply_action(&copy, all_actions[a])) continue;

        candidates[n_cand].sched = copy;
        candidates[n_cand].n_actions = beam[b].n_actions + 1;
        memcpy(
            candidates[n_cand].actions, beam[b].actions,
            (size_t)beam[b].n_actions * sizeof(PolyBeamAction)
        );
        candidates[n_cand].actions[beam[b].n_actions] = all_actions[a];
        candidates[n_cand].time_us = INFINITY;
        n_cand++;
      }
    }

    if (n_cand == 0) break;

    /* Compile and time each candidate */
    for (int i = 0; i < n_cand; i++) {
      candidates[i].time_us = beam_compile_and_time(ctx, candidates[i].sched.ast, opts, 3);

      /* Early stop: if > 3x slower than current best after timing, skip remaining reps */
      if (candidates[i].time_us > beam[0].time_us * 3.0 && beam[0].time_us < INFINITY)
        candidates[i].time_us = INFINITY;
    }

    /* Sort by time */
    qsort(candidates, (size_t)n_cand, sizeof(BeamCandidate), beam_candidate_cmp);

    /* Check convergence: best candidate not better than current best */
    if (n_cand > 0 && candidates[0].time_us >= beam[0].time_us - 0.01) break;

    /* Keep top beam_width */
    int new_size = n_cand < beam_width ? n_cand : beam_width;
    /* Filter out INF candidates */
    while (new_size > 0 && candidates[new_size - 1].time_us >= INFINITY)
      new_size--;
    if (new_size == 0) break;

    for (int i = 0; i < new_size; i++) {
      beam[i].sched = candidates[i].sched;
      beam[i].time_us = candidates[i].time_us;
      beam[i].n_actions = candidates[i].n_actions;
      memcpy(
          beam[i].actions, candidates[i].actions,
          (size_t)candidates[i].n_actions * sizeof(PolyBeamAction)
      );
    }
    beam_size = new_size;
  }

  /* Save best result to disk cache */
  if (beam[0].n_actions > 0 && beam[0].time_us < INFINITY) {
    beam_cache_save(cache_key, beam[0].actions, beam[0].n_actions);
  }

  PolyUOp *result = beam[0].sched.ast;
  free(beam);
  free(candidates);
  return result;
}

#else /* __EMSCRIPTEN__ */

/* WASM stub: BEAM search requires native compilation (fork+clang).
 * Falls back to heuristic. Future: use WASM JIT backend for timing. */
static PolyUOp *poly_beam_search(
    PolyCtx *ctx,
    PolyUOp *sink,
    int beam_width,
    PolyRewriteOpts opts
) {
  (void)beam_width;
  return poly_apply_opts_heuristic(ctx, sink, opts.caps);
}

#endif /* __EMSCRIPTEN__ */

typedef struct {
  PolyUOp **ranges;
  int n_ranges;
  PolyUOp **ends;
  int n_ends;
  int cap_ends;
} ReduceEndGroup;

typedef struct {
  int acc_num;
  ReduceEndGroup *groups;
  int n_groups;
  int cap_groups;
} ReduceContext;

static ReduceContext *current_reduce_ctx(void) {
  return (ReduceContext *)poly_graph_rewrite_userctx();
}

static bool same_range_tuple(PolyUOp **a, int na, PolyUOp **b, int nb) {
  if (na != nb) return false;
  for (int i = 0; i < na; i++)
    if (a[i] != b[i]) return false;
  return true;
}

static bool same_range_set(PolyUOp **a, int na, PolyUOp **b, int nb) {
  if (na != nb) return false;
  for (int i = 0; i < na; i++) {
    bool found = false;
    for (int j = 0; j < nb; j++) {
      if (a[i] == b[j]) {
        found = true;
        break;
      }
    }
    if (!found) return false;
  }
  return true;
}

static PolyUOp *clone_range_axis(PolyCtx *ctx, PolyUOp *r, int64_t axis_id) {
  if (!ctx || !r || r->op != POLY_OP_RANGE || r->arg.kind != POLY_ARG_RANGE) return NULL;
  PolyArg a = r->arg;
  PolyArg new_arg;
  if (poly_range_n_extra(a) > 0)
    new_arg = poly_arg_range_ex(
        axis_id, poly_range_axis_type(a), poly_range_extra(a), poly_range_n_extra(a)
    );
  else
    new_arg = poly_arg_range(axis_id, poly_range_axis_type(a));
  return poly_uop(ctx, POLY_OP_RANGE, r->dtype, r->src, r->n_src, new_arg);
}

static void reduce_ctx_clear(ReduceContext *rctx) {
  if (!rctx) return;
  for (int i = 0; i < rctx->n_groups; i++) {
    free(rctx->groups[i].ranges);
    free(rctx->groups[i].ends);
  }
  free(rctx->groups);
  rctx->groups = NULL;
  rctx->n_groups = 0;
  rctx->cap_groups = 0;
  rctx->acc_num = 0;
}

static void reduce_ctx_add_end(ReduceContext *rctx, PolyUOp **ranges, int n_ranges, PolyUOp *end) {
  if (!rctx || !ranges || n_ranges <= 0 || !end) return;

  int gi = -1;
  for (int i = 0; i < rctx->n_groups; i++) {
    if (same_range_tuple(rctx->groups[i].ranges, rctx->groups[i].n_ranges, ranges, n_ranges)) {
      gi = i;
      break;
    }
  }

  if (gi < 0) {
    if (rctx->n_groups >= rctx->cap_groups) {
      rctx->cap_groups = rctx->cap_groups ? rctx->cap_groups * 2 : 8;
      rctx->groups = realloc(rctx->groups, (size_t)rctx->cap_groups * sizeof(ReduceEndGroup));
    }
    gi = rctx->n_groups++;
    rctx->groups[gi] = (ReduceEndGroup){0};
    rctx->groups[gi].ranges = malloc((size_t)n_ranges * sizeof(PolyUOp *));
    memcpy(rctx->groups[gi].ranges, ranges, (size_t)n_ranges * sizeof(PolyUOp *));
    rctx->groups[gi].n_ranges = n_ranges;
  }

  ReduceEndGroup *g = &rctx->groups[gi];
  if (g->n_ends >= g->cap_ends) {
    g->cap_ends = g->cap_ends ? g->cap_ends * 2 : 4;
    g->ends = realloc(g->ends, (size_t)g->cap_ends * sizeof(PolyUOp *));
  }
  g->ends[g->n_ends++] = end;
}

/*
 * rule_reduce_to_acc — Port of tinygrad's reduce_to_acc.
 *
 * Transforms REDUCE(reduce_op, value, reduce_range_0, ...) into:
 *   DEFINE_REG(acc_id)
 *   acc.after(input_ranges...).index(0).store(identity)   [init]
 *   acc.after(init, reduce_ranges...).index(0)             [loop read; LOAD added later]
 *   reduce_op(loop_read, value)                            [accumulate]
 *   acc.index(0).store(result).end(reduce_ranges...)       [finalize]
 *   acc.after(end).index(0)                                [final read; LOAD added later]
 */
static PolyUOp *rule_reduce_to_acc(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  PolyUOp *red = root;
  PolyUOp *inp = red->src[0];
  PolyOps reduce_op = red->arg.ops;

  /* Filter reduce ranges to actual RANGE nodes only.
   * Singleton dims may produce CONST(0) pseudo-ranges from rangeify;
   * these must not enter the AFTER/END chains (tinygrad invariant). */
  PolyUOp *reduce_ranges[POLY_MAX_DIMS];
  int n_reduce_range = 0;
  for (int j = 1; j < red->n_src; j++) {
    if (red->src[j]->op == POLY_OP_RANGE) reduce_ranges[n_reduce_range++] = red->src[j];
  }
  int in_cnt = inp->dtype.count > 0 ? inp->dtype.count : 1;
  int out_cnt = red->dtype.count > 0 ? red->dtype.count : 1;
  int max_terms = (in_cnt > out_cnt && (in_cnt % out_cnt) == 0) ? (in_cnt / out_cnt) : 1;
  PolyUOp **lst = calloc((size_t)max_terms, sizeof(*lst));
  if (!lst) return NULL;
  int n_lst = horizontal_reduce_terms(ctx, inp, red->dtype, lst, max_terms);
  if (n_lst <= 0) {
    free(lst);
    return inp;
  }

  /* Horizontal-only reduce (no loop ranges). */
  if (n_reduce_range == 0) {
    PolyUOp *ret = lst[0];
    for (int i = 1; i < n_lst; i++)
      ret = poly_uop2(ctx, reduce_op, red->dtype, ret, lst[i], poly_arg_none());
    free(lst);
    return ret;
  }

  /* Find input_ranges (outer loops the value depends on) */
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, inp, &n_topo);

  /* tinygrad parity: precompute ended ranges once from END nodes,
   * instead of rescanning all ENDs for every RANGE. */
  PolyMap *ended_ranges = poly_map_new((size_t)(n_topo > 0 ? n_topo * 2 : 16));
  if (ended_ranges) {
    for (int i = 0; i < n_topo; i++) {
      PolyUOp *u = topo[i];
      if (u->op != POLY_OP_END) continue;
      for (int s = 1; s < u->n_src; s++) {
        PolyUOp *r = u->src[s];
        if (!r || r->op != POLY_OP_RANGE) continue;
        poly_map_set(ended_ranges, poly_ptr_hash(r), r, r, poly_ptr_eq);
      }
    }
  }

  PolyUOp *input_ranges[POLY_MAX_DIMS];
  int n_input_ranges = 0;
  for (int i = 0; i < n_topo; i++) {
    if (topo[i]->op != POLY_OP_RANGE) continue;
    /* Skip reduce ranges */
    bool is_reduce = false;
    for (int j = 0; j < n_reduce_range; j++) {
      if (topo[i] == reduce_ranges[j]) {
        is_reduce = true;
        break;
      }
    }
    /* Skip already-ended ranges */
    if (!is_reduce) {
      bool is_ended = false;
      if (ended_ranges) {
        is_ended = poly_map_get(ended_ranges, poly_ptr_hash(topo[i]), topo[i], poly_ptr_eq) != NULL;
      } else {
        for (int k = 0; k < n_topo; k++) {
          if (topo[k]->op != POLY_OP_END) continue;
          for (int s = 1; s < topo[k]->n_src; s++) {
            if (topo[k]->src[s] == topo[i]) {
              is_ended = true;
              break;
            }
          }
          if (is_ended) break;
        }
      }
      if (!is_ended && n_input_ranges < POLY_MAX_DIMS) input_ranges[n_input_ranges++] = topo[i];
    }
  }
  if (ended_ranges) poly_map_destroy(ended_ranges);
  /* topo is arena-allocated, no free needed */

  /* Identity element */
  PolyUOp *identity = poly_identity_element(ctx, reduce_op, red->dtype);
  if (!identity) {
    free(lst);
    return NULL;
  }

  ReduceContext *rctx = current_reduce_ctx();
  if (!rctx) {
    free(lst);
    return NULL;
  }

  /* tinygrad reduce_to_acc uses UOp.placeholder(..., AddrSpace.REG), which is
   * represented as a BUFFER in register address space with a size source. Keep
   * the acc slot in arg.i for C-side uniqueness. */
  int acc_id = rctx->acc_num++;
  PolyDType acc_ptr = poly_dtype_ptr(red->dtype, 1, POLY_ADDR_REG);
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(1));
  PolyUOp *acc = poly_uop1(ctx, POLY_OP_BUFFER, acc_ptr, one, poly_arg_int(acc_id));
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));

  /* Init: acc.after(input_ranges...).index(0).store(identity) */
  PolyUOp *acc_base;
  if (n_input_ranges > 0) {
    PolyUOp *after_srcs[POLY_MAX_DIMS + 1];
    after_srcs[0] = acc;
    for (int i = 0; i < n_input_ranges; i++)
      after_srcs[i + 1] = input_ranges[i];
    acc_base =
        poly_uop(ctx, POLY_OP_AFTER, acc_ptr, after_srcs, n_input_ranges + 1, poly_arg_none());
  } else {
    acc_base = acc;
  }
  PolyUOp *init_idx = poly_uop2(ctx, POLY_OP_INDEX, red->dtype, acc_base, zero, poly_arg_none());
  PolyUOp *acc_init = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, init_idx, identity, poly_arg_none());

  /* Loop read: acc.after(init, reduce_ranges...).index(0).
   * tinygrad adds LOAD in pm_add_loads, after thread/global dimensions are
   * inserted. Keep the same boundary so x86 isel sees the same REG-backed
   * address forms that tinygrad folds into memory operands. */
  PolyUOp *loop_srcs[POLY_MAX_DIMS + 2];
  loop_srcs[0] = acc;
  loop_srcs[1] = acc_init;
  for (int i = 0; i < n_reduce_range; i++)
    loop_srcs[i + 2] = reduce_ranges[i];
  PolyUOp *loop_after =
      poly_uop(ctx, POLY_OP_AFTER, acc_ptr, loop_srcs, n_reduce_range + 2, poly_arg_none());
  PolyUOp *loop_idx = poly_uop2(ctx, POLY_OP_INDEX, red->dtype, loop_after, zero, poly_arg_none());

  /* Accumulate: reduce_op(loop_load, horizontal_reduce(inp)) */
  PolyUOp *alu = loop_idx;
  for (int i = 0; i < n_lst; i++)
    alu = poly_uop2(ctx, reduce_op, red->dtype, alu, lst[i], poly_arg_none());

  /* Store back + END: acc.index(0).store(alu).end(reduce_ranges...) */
  PolyUOp *store_idx = poly_uop2(ctx, POLY_OP_INDEX, red->dtype, acc, zero, poly_arg_none());
  PolyUOp *acc_store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, store_idx, alu, poly_arg_none());

  /* Build END chain (innermost first to match tinygrad) */
  PolyUOp *chain = acc_store;
  for (int i = n_reduce_range - 1; i >= 0; i--) {
    PolyUOp *end_srcs[2] = {chain, reduce_ranges[i]};
    chain = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_srcs, 2, poly_arg_none());
  }
  reduce_ctx_add_end(rctx, reduce_ranges, n_reduce_range, chain);

  /* Final read: acc.after(end).index(0); LOAD is added by pm_add_loads. */
  PolyUOp *final_srcs[2] = {acc, chain};
  PolyUOp *final_after = poly_uop(ctx, POLY_OP_AFTER, acc_ptr, final_srcs, 2, poly_arg_none());
  PolyUOp *final_idx =
      poly_uop2(ctx, POLY_OP_INDEX, red->dtype, final_after, zero, poly_arg_none());

  free(lst);
  return final_idx;
}

static PolyUOp *rule_merge_reduce_ends(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  if (!root || root->op != POLY_OP_SINK) return NULL;
  ReduceContext *rctx = current_reduce_ctx();
  if (!rctx) return NULL;

  int n_subs_cap = 0;
  for (int i = 0; i < rctx->n_groups; i++)
    if (rctx->groups[i].n_ends > 1) n_subs_cap += rctx->groups[i].n_ends;
  if (n_subs_cap == 0) return NULL;

  PolyUOp **sub_old = malloc((size_t)n_subs_cap * sizeof(PolyUOp *));
  PolyUOp **sub_new = malloc((size_t)n_subs_cap * sizeof(PolyUOp *));
  if (!sub_old || !sub_new) {
    free(sub_old);
    free(sub_new);
    return NULL;
  }
  int at = 0;

  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, root, &n_topo);
  int64_t next_axis = 0;
  for (int i = 0; i < n_topo; i++) {
    if (topo[i] && topo[i]->op == POLY_OP_RANGE && topo[i]->arg.kind == POLY_ARG_RANGE) {
      int64_t axis = poly_range_axis_id(topo[i]->arg);
      if (axis >= next_axis) next_axis = axis + 1;
    }
  }

  for (int i = 0; i < rctx->n_groups; i++) {
    ReduceEndGroup *g = &rctx->groups[i];
    if (g->n_ends <= 1) continue;

    PolyUOp **ctx_ranges = calloc((size_t)g->n_ends * 64, sizeof(PolyUOp *));
    int *ctx_n = calloc((size_t)g->n_ends, sizeof(int));
    int *ctx_group = calloc((size_t)g->n_ends, sizeof(int));
    int n_ctx_groups = 0;
    if (!ctx_ranges || !ctx_n || !ctx_group) {
      free(ctx_ranges);
      free(ctx_n);
      free(ctx_group);
      continue;
    }

    for (int j = 0; j < g->n_ends; j++) {
      PolyUOp **jranges = &ctx_ranges[(size_t)j * 64];
      int n_j = poly_uop_ranges(ctx, g->ends[j], jranges, 64);
      if (n_j < 0) n_j = 0;
      ctx_n[j] = n_j;
      int cg = -1;
      for (int k = 0; k < n_ctx_groups; k++) {
        PolyUOp **kranges = &ctx_ranges[(size_t)k * 64];
        if (same_range_set(jranges, n_j, kranges, ctx_n[k])) {
          cg = k;
          break;
        }
      }
      if (cg < 0) {
        cg = n_ctx_groups++;
        if (cg != j) {
          memcpy(&ctx_ranges[(size_t)cg * 64], jranges, (size_t)n_j * sizeof(PolyUOp *));
          ctx_n[cg] = n_j;
        }
      }
      ctx_group[j] = cg;
    }

    for (int cg = 0; cg < n_ctx_groups; cg++) {
      int count = 0;
      for (int j = 0; j < g->n_ends; j++)
        if (ctx_group[j] == cg) count++;
      if (count <= 0) continue;

      PolyUOp *mapped_ranges[POLY_MAX_DIMS];
      if (cg == 0) {
        for (int r = 0; r < g->n_ranges; r++)
          mapped_ranges[r] = g->ranges[r];
      } else {
        for (int r = 0; r < g->n_ranges; r++) {
          mapped_ranges[r] = clone_range_axis(ctx, g->ranges[r], next_axis + r);
          if (!mapped_ranges[r]) mapped_ranges[r] = g->ranges[r];
        }
        next_axis += g->n_ranges;
      }

      PolyUOp **mapped_ends = malloc((size_t)count * sizeof(PolyUOp *));
      int m = 0;
      for (int j = 0; j < g->n_ends; j++) {
        if (ctx_group[j] != cg) continue;
        PolyUOp *mapped = g->ends[j];
        if (cg != 0) {
          for (int r = 0; r < g->n_ranges; r++) {
            PolyUOp *memo_old[4096];
            PolyUOp *memo_new[4096];
            int memo_n = 0;
            mapped = substitute_node(
                ctx, mapped, g->ranges[r], mapped_ranges[r], memo_old, memo_new, &memo_n, 4096
            );
          }
        }
        mapped_ends[m++] = mapped;
      }

      PolyUOp *chain = mapped_ends[0];
      if (count > 1) {
        PolyUOp **group_srcs = malloc((size_t)count * sizeof(PolyUOp *));
        for (int j = 0; j < count; j++)
          group_srcs[j] = mapped_ends[j]->src[0];
        chain = poly_uop(ctx, POLY_OP_GROUP, POLY_VOID, group_srcs, count, poly_arg_none());
        free(group_srcs);

        for (int r = g->n_ranges - 1; r >= 0; r--) {
          PolyUOp *esrc[2] = {chain, mapped_ranges[r]};
          chain = poly_uop(ctx, POLY_OP_END, POLY_VOID, esrc, 2, poly_arg_none());
        }
      }

      m = 0;
      for (int j = 0; j < g->n_ends; j++) {
        if (ctx_group[j] != cg) continue;
        if (chain != g->ends[j]) {
          sub_old[at] = g->ends[j];
          sub_new[at] = count > 1 ? chain : mapped_ends[m];
          at++;
        }
        m++;
      }
      free(mapped_ends);
    }

    free(ctx_ranges);
    free(ctx_n);
    free(ctx_group);
  }

  if (at == 0) {
    free(sub_new);
    free(sub_old);
    return NULL;
  }

  PolyUOp *out = root;
  for (int i = 0; i < at; i++) {
    PolyUOp *memo_old[4096];
    PolyUOp *memo_new[4096];
    int memo_n = 0;
    out = substitute_node(ctx, out, sub_old[i], sub_new[i], memo_old, memo_new, &memo_n, 4096);
  }

  free(sub_new);
  free(sub_old);
  return out != root ? out : NULL;
}

/* Build the pm_reduce PatternMatcher */

static _Thread_local PolyPatternMatcher *g_pm_reduce = NULL;

static PolyPatternMatcher *poly_pm_reduce(void) {
  if (g_pm_reduce) return g_pm_reduce;

  PolyOpSet reduce_set = poly_opset_add((PolyOpSet){{0, 0}}, POLY_OP_REDUCE);
  PolyOpSet sink_set = poly_opset_add((PolyOpSet){{0, 0}}, POLY_OP_SINK);
  PolyRule rules[] = {
      {poly_pat_ops(reduce_set, NULL, 0, NULL), rule_reduce_to_acc},
      {poly_pat_ops(sink_set, NULL, 0, NULL), rule_merge_reduce_ends},
  };
  g_pm_reduce = poly_pm_thread_cache(poly_pm_new(rules, 2));
  return g_pm_reduce;
}

/* pm_decomp: late decompositions (MAX → CMPLT+WHERE, etc.) */

/*
 * rule_decomp_max — Port of tinygrad's get_late_rewrite_patterns MAX rule.
 * MAX(a, b) → WHERE(CMPLT(a, b), b, a)
 * ClangRenderer doesn't have native MAX, so decompose to CMPLT+WHERE.
 */
static PolyUOp *rule_decomp_max(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  PolyUOp *a = root->src[0];
  PolyDType cmp_bt =
      (root->dtype.count > 1) ? poly_dtype_vec(POLY_BOOL, root->dtype.count) : POLY_BOOL;
  PolyUOp *cmp = poly_uop2(ctx, POLY_OP_CMPLT, cmp_bt, a, root->src[1], poly_arg_none());
  return poly_uop3(ctx, POLY_OP_WHERE, root->dtype, cmp, root->src[1], a, poly_arg_none());
}

/*
 * rule_mul_to_shl — Port of tinygrad's get_late_rewrite_patterns MUL→SHL rule.
 * x * c → SHL(x, log2(c))  when c is a power of 2 and x is integer type.
 */
static PolyUOp *rule_mul_to_shl(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  PolyUOp *c_node = poly_bind(b, "c");
  PolyUOp *x_node = poly_bind(b, "x");
  if (!c_node || !x_node) return NULL;
  /* tinygrad's final render loop leaves vector MUL-by-constant as vector ALU
   * plus STACK constants; this scalar strength reduction must not fire on
   * vector lanes. */
  if (root->dtype.count > 1) return NULL;
  if (!poly_dtype_is_int(root->dtype)) return NULL;
  if (c_node->arg.kind != POLY_ARG_INT) return NULL;
  int64_t c = c_node->arg.i;
  if (c <= 0 || (c & (c - 1)) != 0) return NULL; /* not a power of 2 */
  int shift = 0;
  int64_t tmp = c;
  while (tmp > 1) {
    shift++;
    tmp >>= 1;
  }
  PolyUOp *shift_const = poly_uop0(ctx, POLY_OP_CONST, root->dtype, poly_arg_int(shift));
  return poly_uop2(ctx, POLY_OP_SHL, root->dtype, x_node, shift_const, poly_arg_none());
}

/*
 * rule_idiv_to_shr — Port of tinygrad's get_late_rewrite_patterns IDIV→SHR rule.
 * x // c → SHR(x, log2(c))  when c is a power of 2.
 * For signed ints: (x + (x<0).where(c-1, 0)) >> log2(c)
 */
static PolyUOp *rule_idiv_to_shr(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  PolyUOp *c_node = poly_bind(b, "c");
  PolyUOp *x_node = poly_bind(b, "x");
  if (!c_node || !x_node) return NULL;
  /* Match tinygrad's vector path: vector IDIV by a constant is rendered as
   * IDIV(..., STACK(...)), not decomposed into per-lane shift/correction here. */
  if (root->dtype.count > 1) return NULL;
  if (!poly_dtype_is_int(root->dtype)) return NULL;
  if (c_node->arg.kind != POLY_ARG_INT) return NULL;
  int64_t c = c_node->arg.i;
  if (c <= 0 || (c & (c - 1)) != 0) return NULL; /* not a power of 2 */
  int shift = 0;
  int64_t tmp = c;
  while (tmp > 1) {
    shift++;
    tmp >>= 1;
  }
  PolyUOp *shift_const = poly_uop0(ctx, POLY_OP_CONST, root->dtype, poly_arg_int(shift));
  /* Unsigned: just shift right */
  if (poly_dtype_is_unsigned(root->dtype))
    return poly_uop2(ctx, POLY_OP_SHR, root->dtype, x_node, shift_const, poly_arg_none());
  /* Signed: (x + (x<0).where(c-1, 0)) >> shift */
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, root->dtype, poly_arg_int(0));
  PolyDType cmp_bt =
      (root->dtype.count > 1) ? poly_dtype_vec(POLY_BOOL, root->dtype.count) : POLY_BOOL;
  PolyUOp *cmplt = poly_uop2(ctx, POLY_OP_CMPLT, cmp_bt, x_node, zero, poly_arg_none());
  PolyUOp *cm1 = poly_uop0(ctx, POLY_OP_CONST, root->dtype, poly_arg_int(c - 1));
  PolyUOp *correction =
      poly_uop3(ctx, POLY_OP_WHERE, root->dtype, cmplt, cm1, zero, poly_arg_none());
  PolyUOp *corrected =
      poly_uop2(ctx, POLY_OP_ADD, root->dtype, x_node, correction, poly_arg_none());
  return poly_uop2(ctx, POLY_OP_SHR, root->dtype, corrected, shift_const, poly_arg_none());
}

static bool divmod_floor_same_as_c(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  int64_t a_min = 0, a_max = 0, b_min = 0, b_max = 0;
  poly_uop_minmax(ctx, a, &a_min, &a_max);
  poly_uop_minmax(ctx, b, &b_min, &b_max);
  return (a_min >= 0 && b_min > 0) || (a_max <= 0 && b_max < 0);
}

static PolyUOp *rule_floordiv_to_cdiv(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  if (!root || root->n_src != 2 || !poly_dtype_is_int(poly_dtype_scalar(root->dtype))) return NULL;
  PolyUOp *a = root->src[0];
  PolyUOp *den = root->src[1];
  if (poly_dtype_is_unsigned(poly_dtype_scalar(root->dtype)) || divmod_floor_same_as_c(ctx, a, den))
    return poly_uop2(ctx, POLY_OP_CDIV, root->dtype, a, den, poly_arg_none());

  PolyUOp *trunc = poly_uop2(ctx, POLY_OP_CDIV, root->dtype, a, den, poly_arg_none());
  PolyUOp *rem = poly_uop2(ctx, POLY_OP_CMOD, root->dtype, a, den, poly_arg_none());
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, root->dtype, poly_arg_int(0));
  PolyDType bt = (root->dtype.count > 1) ? poly_dtype_vec(POLY_BOOL, root->dtype.count) : POLY_BOOL;
  PolyUOp *rem_ne_zero = poly_uop2(ctx, POLY_OP_CMPNE, bt, rem, zero, poly_arg_none());
  PolyUOp *a_lt_zero = poly_uop2(ctx, POLY_OP_CMPLT, bt, a, zero, poly_arg_none());
  PolyUOp *b_lt_zero = poly_uop2(ctx, POLY_OP_CMPLT, bt, den, zero, poly_arg_none());
  PolyUOp *sign_mismatch = poly_uop2(ctx, POLY_OP_CMPNE, bt, a_lt_zero, b_lt_zero, poly_arg_none());
  PolyUOp *needs_adjust =
      poly_uop2(ctx, POLY_OP_AND, bt, rem_ne_zero, sign_mismatch, poly_arg_none());
  PolyUOp *adjust = poly_uop1(ctx, POLY_OP_CAST, root->dtype, needs_adjust, poly_arg_none());
  return poly_uop2(ctx, POLY_OP_SUB, root->dtype, trunc, adjust, poly_arg_none());
}

static PolyUOp *rule_floormod_to_cmod(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  if (!root || root->n_src != 2 || !poly_dtype_is_int(poly_dtype_scalar(root->dtype))) return NULL;
  PolyUOp *a = root->src[0];
  PolyUOp *den = root->src[1];
  if (poly_dtype_is_unsigned(poly_dtype_scalar(root->dtype)) || divmod_floor_same_as_c(ctx, a, den))
    return poly_uop2(ctx, POLY_OP_CMOD, root->dtype, a, den, poly_arg_none());

  PolyUOp *rem = poly_uop2(ctx, POLY_OP_CMOD, root->dtype, a, den, poly_arg_none());
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, root->dtype, poly_arg_int(0));
  PolyDType bt = (root->dtype.count > 1) ? poly_dtype_vec(POLY_BOOL, root->dtype.count) : POLY_BOOL;
  PolyUOp *rem_ne_zero = poly_uop2(ctx, POLY_OP_CMPNE, bt, rem, zero, poly_arg_none());
  PolyUOp *a_lt_zero = poly_uop2(ctx, POLY_OP_CMPLT, bt, a, zero, poly_arg_none());
  PolyUOp *b_lt_zero = poly_uop2(ctx, POLY_OP_CMPLT, bt, den, zero, poly_arg_none());
  PolyUOp *sign_mismatch = poly_uop2(ctx, POLY_OP_CMPNE, bt, a_lt_zero, b_lt_zero, poly_arg_none());
  PolyUOp *needs_adjust =
      poly_uop2(ctx, POLY_OP_AND, bt, rem_ne_zero, sign_mismatch, poly_arg_none());
  PolyUOp *correction =
      poly_uop3(ctx, POLY_OP_WHERE, root->dtype, needs_adjust, den, zero, poly_arg_none());
  return poly_uop2(ctx, POLY_OP_ADD, root->dtype, rem, correction, poly_arg_none());
}

/*
 * rule_mulacc_to_mul_add — MULACC(a, b, c) → ADD(MUL(a, b), c)
 * For renderers without native FMA (CPU/ClangRenderer).
 * Gated on !caps.has_mulacc in poly_pm_decomp_with_caps().
 */
static PolyUOp *rule_mulacc_to_mul_add(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  if (root->op != POLY_OP_MULACC || root->n_src != 3) return NULL;
  PolyUOp *mul =
      poly_uop2(ctx, POLY_OP_MUL, root->dtype, root->src[0], root->src[1], poly_arg_none());
  return poly_uop2(ctx, POLY_OP_ADD, root->dtype, mul, root->src[2], poly_arg_none());
}

/*
 * rule_mul_add_to_mulacc — ADD(MUL(a, b), c) → MULACC(a, b, c)
 * Fusion rule for renderers with native FMA (CUDA).  Float scalars only.
 * Port of tinygrad: if Ops.MULACC in ops: a*b+c → MULACC(a,b,c)
 * Gated on caps.has_mulacc in poly_pm_decomp_with_caps().
 */
static PolyUOp *rule_mul_add_to_mulacc(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  if (root->op != POLY_OP_ADD || root->n_src != 2) return NULL;
  if (!poly_dtype_is_float(root->dtype)) return NULL;
  PolyUOp *mul = NULL, *add = NULL;
  if (root->src[0]->op == POLY_OP_MUL) {
    mul = root->src[0];
    add = root->src[1];
  } else if (root->src[1]->op == POLY_OP_MUL) {
    mul = root->src[1];
    add = root->src[0];
  } else
    return NULL;
  if (mul->n_src != 2) return NULL;
  if (!poly_dtype_eq(mul->dtype, root->dtype)) return NULL;
  if (!poly_dtype_eq(add->dtype, root->dtype)) return NULL;
  PolyUOp *srcs[3] = {mul->src[0], mul->src[1], add};
  return poly_uop(ctx, POLY_OP_MULACC, root->dtype, srcs, 3, poly_arg_none());
}

/*
 * rule_shl_add_to_mulacc — ADD(SHL(x, n), c) → MULACC(x, 2^n, c)
 * When MUL(x, pow2) was already decomposed to SHL by an earlier pass,
 * the ADD fusion needs to recognize the shifted form.
 * Ref: tinygrad decompositions.py:480
 */
static PolyUOp *rule_shl_add_to_mulacc(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  if (root->op != POLY_OP_ADD || root->n_src != 2) return NULL;
  if (!poly_dtype_is_int(root->dtype)) return NULL;
  PolyUOp *shl = NULL, *c = NULL;
  if (root->src[0]->op == POLY_OP_SHL) {
    shl = root->src[0];
    c = root->src[1];
  } else if (root->src[1]->op == POLY_OP_SHL) {
    shl = root->src[1];
    c = root->src[0];
  } else
    return NULL;
  if (shl->n_src != 2) return NULL;
  PolyUOp *n_const = shl->src[1];
  if (n_const->op != POLY_OP_CONST || n_const->arg.kind != POLY_ARG_INT) return NULL;
  int64_t shift = n_const->arg.i;
  if (shift < 0 || shift > 30) return NULL;
  PolyUOp *factor = poly_const_like_int(ctx, shl->src[0], 1LL << shift);
  PolyUOp *srcs[3] = {shl->src[0], factor, c};
  return poly_uop(ctx, POLY_OP_MULACC, root->dtype, srcs, 3, poly_arg_none());
}

/*
 * rule_mul_neg1_to_neg — Port of tinygrad late rewrite:
 * x * (-1) → NEG(x)
 */
static PolyUOp *rule_mul_neg1_to_neg(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  PolyUOp *x = poly_bind(b, "x");
  PolyUOp *c = poly_bind(b, "c");
  if (!x || !c) return NULL;
  /* Vector x * -1 stays as vector MUL with rendered STACK constants in
   * tinygrad, so only scalar MUL is folded to NEG here. */
  if (root->dtype.count > 1) return NULL;
  if (c->op != POLY_OP_CONST) return NULL;
  if (c->arg.kind == POLY_ARG_INT && c->arg.i == -1)
    return poly_uop1(ctx, POLY_OP_NEG, root->dtype, x, poly_arg_none());
  if (c->arg.kind == POLY_ARG_FLOAT && c->arg.f == -1.0)
    return poly_uop1(ctx, POLY_OP_NEG, root->dtype, x, poly_arg_none());
  return NULL;
}

/*
 * rule_add_neg_to_sub — Port of tinygrad late rewrite:
 * x + NEG(y) → SUB(x, y)
 */
static PolyUOp *rule_add_neg_to_sub(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  PolyUOp *x = poly_bind(b, "x");
  PolyUOp *y = poly_bind(b, "y");
  if (!x || !y) return NULL;
  return poly_uop2(ctx, POLY_OP_SUB, root->dtype, x, y, poly_arg_none());
}

static PolyUOp *rule_neg_add_to_sub(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  PolyUOp *x = poly_bind(b, "x");
  PolyUOp *y = poly_bind(b, "y");
  if (!x || !y) return NULL;
  return poly_uop2(ctx, POLY_OP_SUB, root->dtype, x, y, poly_arg_none());
}

/*
 * rule_recip_to_fdiv — Port of tinygrad late rewrite:
 * RECIPROCAL(x) → FDIV(1, x)
 */
static PolyUOp *rule_recip_to_fdiv(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  PolyUOp *x = poly_bind(b, "x");
  if (!x || !poly_dtype_is_float(root->dtype)) return NULL;
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, root->dtype, poly_arg_float(1.0));
  return poly_uop2(ctx, POLY_OP_FDIV, root->dtype, one, x, poly_arg_none());
}

/*
 * rule_mul_fdiv1_to_fdiv — Port of tinygrad late rewrite:
 * a * (1 / b) → a / b
 */
static PolyUOp *rule_mul_fdiv1_to_fdiv(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  PolyUOp *a = poly_bind(b, "a");
  PolyUOp *bnode = poly_bind(b, "b");
  PolyUOp *one = poly_bind(b, "one");
  if (!a || !bnode || !one) return NULL;
  if (!poly_dtype_is_float(root->dtype)) return NULL;
  if (one->op != POLY_OP_CONST) return NULL;
  if (one->arg.kind == POLY_ARG_FLOAT && one->arg.f == 1.0)
    return poly_uop2(ctx, POLY_OP_FDIV, root->dtype, a, bnode, poly_arg_none());
  if (one->arg.kind == POLY_ARG_INT && one->arg.i == 1)
    return poly_uop2(ctx, POLY_OP_FDIV, root->dtype, a, bnode, poly_arg_none());
  return NULL;
}

static bool is_true_const_late(PolyUOp *u) {
  return u && u->op == POLY_OP_CONST &&
         ((u->arg.kind == POLY_ARG_BOOL && u->arg.b) ||
          (u->arg.kind == POLY_ARG_INT && u->arg.i == 1));
}

static bool int_const_value_codegen(PolyUOp *u, int64_t *out);

/* tinygrad decompositions.py:get_late_rewrite_patterns:
 *   (x < c).logical_not() -> c-1 < x
 *   (c < x).logical_not() -> x < c+1
 *
 * This is deliberately a late decomposition, not a symbolic construction rule.
 * Earlier rewrites rely on the not-CMPLT shape for simplex/valid reasoning. */
static PolyUOp *rule_not_cmplt_to_bound(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  if (!root || root->op != POLY_OP_CMPNE || root->n_src != 2) return NULL;

  PolyUOp *lt = NULL;
  if (root->src[0]->op == POLY_OP_CMPLT && is_true_const_late(root->src[1]))
    lt = root->src[0];
  else if (root->src[1]->op == POLY_OP_CMPLT && is_true_const_late(root->src[0]))
    lt = root->src[1];
  else
    return NULL;
  if (!lt || lt->n_src != 2) return NULL;

  int64_t cval = 0, adjusted = 0;
  if (int_const_value_codegen(lt->src[1], &cval) && poly_dtype_is_int(lt->src[0]->dtype) &&
      !poly_dtype_is_unsigned(lt->src[0]->dtype) && !poly_dtype_is_bool(lt->src[0]->dtype)) {
    if (__builtin_sub_overflow(cval, 1, &adjusted)) return NULL;
    return poly_uop2(
        ctx, POLY_OP_CMPLT, POLY_BOOL, poly_const_like_int(ctx, lt->src[0], adjusted), lt->src[0],
        poly_arg_none()
    );
  }
  if (int_const_value_codegen(lt->src[0], &cval) && poly_dtype_is_int(lt->src[1]->dtype) &&
      !poly_dtype_is_unsigned(lt->src[1]->dtype) && !poly_dtype_is_bool(lt->src[1]->dtype)) {
    if (__builtin_add_overflow(cval, 1, &adjusted)) return NULL;
    return poly_uop2(
        ctx, POLY_OP_CMPLT, POLY_BOOL, lt->src[1], poly_const_like_int(ctx, lt->src[1], adjusted),
        poly_arg_none()
    );
  }
  return NULL;
}

/* tinygrad decompositions.py:
 *   x.ne(y).logical_not() -> x.alu(Ops.CMPEQ, y)
 *
 * Tensor-level eq stays as CMPNE(CMPNE(x,y), true) for frontend parity with
 * tinygrad's elementwise helper. Renderers that support CMPEQ should see the
 * direct comparison in late IR, which removes an extra bool compare and matches
 * current tinygrad linearized kernels such as Tensor.eye. */
static PolyUOp *rule_cmpne_not_to_cmpeq(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  if (!root || root->op != POLY_OP_CMPNE || root->n_src != 2) return NULL;

  PolyUOp *cmpne = NULL;
  if (root->src[0]->op == POLY_OP_CMPNE && is_true_const_late(root->src[1])) {
    cmpne = root->src[0];
  } else if (root->src[1]->op == POLY_OP_CMPNE && is_true_const_late(root->src[0])) {
    cmpne = root->src[1];
  } else {
    return NULL;
  }
  if (!cmpne || cmpne->n_src != 2) return NULL;
  return poly_uop2(ctx, POLY_OP_CMPEQ, root->dtype, cmpne->src[0], cmpne->src[1], poly_arg_none());
}

static PolyUOp *u32_const(PolyCtx *ctx, uint32_t v) {
  return poly_uop0(ctx, POLY_OP_CONST, POLY_UINT32, poly_arg_int((int64_t)v));
}

static PolyUOp *u64_const(PolyCtx *ctx, uint64_t v) {
  return poly_uop0(ctx, POLY_OP_CONST, POLY_UINT64, poly_arg_int((int64_t)v));
}

static PolyUOp *u32_cast(PolyCtx *ctx, PolyUOp *u) {
  if (poly_dtype_eq(u->dtype, POLY_UINT32)) return u;
  return poly_uop1(ctx, POLY_OP_CAST, POLY_UINT32, u, poly_arg_none());
}

static PolyUOp *u64_cast(PolyCtx *ctx, PolyUOp *u) {
  if (poly_dtype_eq(u->dtype, POLY_UINT64)) return u;
  return poly_uop1(ctx, POLY_OP_CAST, POLY_UINT64, u, poly_arg_none());
}

static PolyUOp *u32_rol(PolyCtx *ctx, PolyUOp *x, int r) {
  PolyUOp *l =
      poly_uop2(ctx, POLY_OP_SHL, POLY_UINT32, x, u32_const(ctx, (uint32_t)r), poly_arg_none());
  PolyUOp *rr = poly_uop2(
      ctx, POLY_OP_SHR, POLY_UINT32, x, u32_const(ctx, (uint32_t)(32 - r)), poly_arg_none()
  );
  return poly_uop2(ctx, POLY_OP_OR, POLY_UINT32, l, rr, poly_arg_none());
}

/*
 * rule_decomp_threefry32 — lower THREEFRY to pure integer ALU UOps.
 * This mirrors tinygrad's decomposition strategy (threefry2x32) but
 * emits a 32-bit lane value directly for current backends.
 */
static PolyUOp *rule_decomp_threefry32(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  if (root->op != POLY_OP_THREEFRY || root->n_src != 2) return NULL;

  /* Tinygrad-parity path: for uint64 THREEFRY, split into two uint32 lanes.
   * For uint32 input (legacy elementwise usage), high lanes are zero. */
  PolyUOp *x0, *x1, *key0, *key1;
  if (poly_dtype_eq(root->dtype, POLY_UINT64)) {
    PolyUOp *x64 = u64_cast(ctx, root->src[0]);
    PolyUOp *k64 = u64_cast(ctx, root->src[1]);
    PolyUOp *mask32 = u64_const(ctx, 0xFFFFFFFFull);
    PolyUOp *sh32 = u64_const(ctx, 32);
    x0 = u32_cast(ctx, poly_uop2(ctx, POLY_OP_AND, POLY_UINT64, x64, mask32, poly_arg_none()));
    x1 = u32_cast(
        ctx, poly_uop2(
                 ctx, POLY_OP_AND, POLY_UINT64,
                 poly_uop2(ctx, POLY_OP_SHR, POLY_UINT64, x64, sh32, poly_arg_none()), mask32,
                 poly_arg_none()
             )
    );
    key0 = u32_cast(ctx, poly_uop2(ctx, POLY_OP_AND, POLY_UINT64, k64, mask32, poly_arg_none()));
    key1 = u32_cast(
        ctx, poly_uop2(
                 ctx, POLY_OP_AND, POLY_UINT64,
                 poly_uop2(ctx, POLY_OP_SHR, POLY_UINT64, k64, sh32, poly_arg_none()), mask32,
                 poly_arg_none()
             )
    );
  } else {
    x0 = u32_cast(ctx, root->src[0]);
    x1 = u32_const(ctx, 0);
    key0 = u32_cast(ctx, root->src[1]);
    key1 = u32_const(ctx, 0);
  }

  PolyUOp *ks[3];
  ks[0] = key1;
  ks[1] = poly_uop2(
      ctx, POLY_OP_XOR, POLY_UINT32,
      poly_uop2(ctx, POLY_OP_XOR, POLY_UINT32, key0, key1, poly_arg_none()),
      u32_const(ctx, 0x1BD11BDAu), poly_arg_none()
  );
  ks[2] = key0;

  PolyUOp *xr0 = poly_uop2(ctx, POLY_OP_ADD, POLY_UINT32, x0, ks[2], poly_arg_none());
  PolyUOp *xr1 = poly_uop2(ctx, POLY_OP_ADD, POLY_UINT32, x1, ks[0], poly_arg_none());

  static const int rotations[2][4] = {
      {13, 15, 26, 6},
      {17, 29, 16, 24},
  };

  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 4; j++) {
      int r = rotations[i & 1][j];
      PolyUOp *sum = poly_uop2(ctx, POLY_OP_ADD, POLY_UINT32, xr0, xr1, poly_arg_none());
      xr1 = poly_uop2(ctx, POLY_OP_XOR, POLY_UINT32, sum, u32_rol(ctx, xr1, r), poly_arg_none());
      xr0 = sum;
    }
    PolyUOp *round = u32_const(ctx, (uint32_t)(i + 1));
    xr0 = poly_uop2(ctx, POLY_OP_ADD, POLY_UINT32, xr0, ks[i % 3], poly_arg_none());
    xr1 = poly_uop2(
        ctx, POLY_OP_ADD, POLY_UINT32,
        poly_uop2(ctx, POLY_OP_ADD, POLY_UINT32, xr1, ks[(i + 1) % 3], poly_arg_none()), round,
        poly_arg_none()
    );
  }

  if (poly_dtype_eq(root->dtype, POLY_UINT32)) return xr0;
  if (poly_dtype_eq(root->dtype, POLY_UINT64)) {
    PolyUOp *lo = u64_cast(ctx, xr0);
    PolyUOp *hi = poly_uop2(
        ctx, POLY_OP_SHL, POLY_UINT64, u64_cast(ctx, xr1), u64_const(ctx, 32), poly_arg_none()
    );
    return poly_uop2(ctx, POLY_OP_OR, POLY_UINT64, hi, lo, poly_arg_none());
  }
  return poly_uop1(ctx, POLY_OP_CAST, root->dtype, xr0, poly_arg_none());
}

/* ***** long as two ints *****
 *
 * Port of tinygrad decompositions.py::l2i/pm_long_decomp. Renderer capability
 * controls whether this pass runs: WGSL has only i32/u32, while C, CUDA, HIP,
 * WASM, and x86 retain native int64 values. A pass-private tag_arg marker
 * carries low/high lane requests while preserving the complete public tag and
 * tag_arg; no lane marker may survive this pass. */

typedef struct {
  PolyUOp *lo;
  PolyUOp *hi;
} L2IPair;

typedef struct {
  uint64_t magic;
  uintptr_t token;
  uint64_t generation;
  int32_t original_tag;
  int32_t lane;
  PolyArg original_tag_arg;
} L2ILaneMarker;

#define L2I_LANE_MAGIC UINT64_C(0x6c32692d6c616e65)

static _Thread_local uintptr_t g_l2i_active_token = 0;
static _Thread_local uint64_t g_l2i_active_generation = 0;
static _Thread_local uint64_t g_l2i_generation_counter = 0;

static PolyDType l2i_value_dtype(PolyDType dt) {
  PolyDType base = poly_dtype_scalar(dt);
  base.is_ptr = false;
  base.addrspace = 0;
  base.vcount = 0;
  base.ptr_size = 0;
  return base;
}

static bool l2i_is_long(PolyDType dt) {
  if (dt.is_ptr) return false;
  PolyDType base = l2i_value_dtype(dt);
  return poly_dtype_is_int(base) && !poly_dtype_is_bool(base) && base.bitsize == 64;
}

static bool l2i_is_long_ptr(PolyDType dt) {
  if (!dt.is_ptr) return false;
  PolyDType base = l2i_value_dtype(dt);
  return poly_dtype_is_int(base) && !poly_dtype_is_bool(base) && base.bitsize == 64;
}

static PolyDType l2i_dt(PolyDType dt) {
  PolyDType base = l2i_value_dtype(dt);
  PolyDType out = poly_dtype_is_unsigned(base) ? POLY_UINT32 : POLY_INT32;
  if (!dt.is_ptr && dt.count > 1) out = poly_dtype_vec(out, dt.count);
  return out;
}

static PolyUOp *l2i_clone(
    PolyCtx *ctx,
    PolyOps op,
    PolyDType dtype,
    PolyUOp **src,
    int n_src,
    PolyArg arg,
    int32_t tag,
    PolyArg tag_arg
) {
  return (tag != 0 || tag_arg.kind != POLY_ARG_NONE)
             ? poly_uop_tagged_arg(ctx, op, dtype, src, n_src, arg, tag, tag_arg)
             : poly_uop(ctx, op, dtype, src, n_src, arg);
}

static PolyUOp *l2i_replace_dtype(PolyCtx *ctx, PolyUOp *u, PolyDType dtype) {
  return l2i_clone(ctx, u->op, dtype, u->src, u->n_src, u->arg, u->tag, u->tag_arg);
}

static bool l2i_lane(PolyUOp *u, int *lane, L2ILaneMarker *decoded) {
  if (!u || !g_l2i_active_token || !g_l2i_active_generation || u->tag_arg.kind != POLY_ARG_BYTES ||
      u->tag_arg.bytes.n != (int)sizeof(L2ILaneMarker) || !u->tag_arg.bytes.data)
    return false;
  L2ILaneMarker marker = {0};
  memcpy(&marker, u->tag_arg.bytes.data, sizeof(marker));
  if (marker.magic != L2I_LANE_MAGIC || marker.token != g_l2i_active_token ||
      marker.generation != g_l2i_active_generation || (marker.lane != 0 && marker.lane != 1))
    return false;
  if (lane) *lane = marker.lane;
  if (decoded) *decoded = marker;
  return true;
}

static PolyUOp *l2i_rtag(PolyCtx *ctx, PolyUOp *u, int lane) {
  int existing_lane = 0;
  if (l2i_lane(u, &existing_lane, NULL)) return existing_lane == lane ? u : NULL;
  if (!u || !g_l2i_active_token || !g_l2i_active_generation || (lane != 0 && lane != 1))
    return NULL;
  L2ILaneMarker marker = {
      .magic = L2I_LANE_MAGIC,
      .token = g_l2i_active_token,
      .generation = g_l2i_active_generation,
      .original_tag = u->tag,
      .lane = lane,
      .original_tag_arg = u->tag_arg,
  };
  return poly_uop_tagged_arg(
      ctx, u->op, u->dtype, u->src, u->n_src, u->arg, u->tag,
      poly_arg_bytes((const uint8_t *)&marker, (int)sizeof(marker))
  );
}

static PolyUOp *l2i_finish_lane(PolyCtx *ctx, PolyUOp *request, PolyUOp *value) {
  int lane = 0;
  L2ILaneMarker marker = {0};
  if (!value || !l2i_lane(request, &lane, &marker)) return value;
  (void)lane;
  return l2i_clone(
      ctx, value->op, value->dtype, value->src, value->n_src, value->arg, marker.original_tag,
      marker.original_tag_arg
  );
}

static PolyUOp *l2i_finish_value(PolyCtx *ctx, PolyUOp *original, PolyUOp *value) {
  if (!value) return NULL;
  return l2i_clone(
      ctx, value->op, value->dtype, value->src, value->n_src, value->arg, original->tag,
      original->tag_arg
  );
}

static PolyUOp *l2i_const(PolyCtx *ctx, PolyDType dt, uint32_t bits) {
  int64_t value = poly_dtype_is_unsigned(dt) ? (int64_t)bits : (int64_t)(int32_t)bits;
  return poly_uop0(ctx, POLY_OP_CONST, dt, poly_arg_int(value));
}

static PolyUOp *l2i_cast(PolyCtx *ctx, PolyUOp *u, PolyDType dt) {
  if (!u) return NULL;
  return poly_dtype_eq(u->dtype, dt) ? u : poly_uop1(ctx, POLY_OP_CAST, dt, u, poly_arg_none());
}

static PolyUOp *l2i_bitcast(PolyCtx *ctx, PolyUOp *u, PolyDType dt) {
  if (!u) return NULL;
  return poly_dtype_eq(u->dtype, dt) ? u : poly_uop1(ctx, POLY_OP_BITCAST, dt, u, poly_arg_none());
}

static PolyUOp *l2i_binary(PolyCtx *ctx, PolyOps op, PolyDType dt, PolyUOp *a, PolyUOp *b) {
  if (!a || !b) return NULL;
  return poly_uop2(ctx, op, dt, a, b, poly_arg_none());
}

static PolyUOp *l2i_cmp(PolyCtx *ctx, PolyOps op, PolyUOp *a, PolyUOp *b) {
  if (!a || !b) return NULL;
  return poly_uop2(ctx, op, POLY_BOOL, a, b, poly_arg_none());
}

static PolyUOp *l2i_not(PolyCtx *ctx, PolyUOp *u) {
  PolyUOp *t = poly_uop0(ctx, POLY_OP_CONST, POLY_BOOL, poly_arg_bool(true));
  return l2i_cmp(ctx, POLY_OP_CMPNE, u, t);
}

static PolyUOp *l2i_where(PolyCtx *ctx, PolyDType dt, PolyUOp *cond, PolyUOp *t, PolyUOp *f) {
  if (!cond || !t || !f) return NULL;
  return poly_uop3(ctx, POLY_OP_WHERE, dt, cond, t, f, poly_arg_none());
}

/* Pinned tinygrad uop/decompositions.py:326 reindex(idx, off, mul):
 * rewrite the scalar INDEX offset for storage-lane decomposition. */
static PolyUOp *codegen_reindex(PolyCtx *ctx, PolyUOp *idx, int lane, int mul) {
  if (!idx || idx->op != POLY_OP_INDEX || idx->n_src < 2 || idx->n_src > 64) return NULL;
  PolyUOp *offset = idx->src[1];
  PolyUOp *scale = poly_const_like(ctx, offset, poly_arg_int(mul));
  PolyUOp *off = poly_const_like(ctx, offset, poly_arg_int(lane));
  PolyUOp *scaled = l2i_binary(ctx, POLY_OP_MUL, offset->dtype, offset, scale);
  PolyUOp *lane_offset = l2i_binary(ctx, POLY_OP_ADD, offset->dtype, scaled, off);
  PolyUOp *src[64];
  for (int i = 0; i < idx->n_src; i++)
    src[i] = idx->src[i];
  src[1] = lane_offset;
  return l2i_clone(ctx, idx->op, idx->dtype, src, idx->n_src, idx->arg, idx->tag, idx->tag_arg);
}

static PolyUOp *l2i_reindex(PolyCtx *ctx, PolyUOp *idx, int lane) {
  return codegen_reindex(ctx, idx, lane, 2);
}

static L2IPair l2i_add(
    PolyCtx *ctx,
    PolyDType dt,
    PolyUOp *a0,
    PolyUOp *a1,
    PolyUOp *b0,
    PolyUOp *b1
) {
  PolyUOp *low = l2i_binary(ctx, POLY_OP_ADD, dt, a0, b0);
  PolyUOp *carry = l2i_cmp(
      ctx, POLY_OP_CMPLT, l2i_bitcast(ctx, low, POLY_UINT32), l2i_bitcast(ctx, a0, POLY_UINT32)
  );
  PolyUOp *high = l2i_binary(
      ctx, POLY_OP_ADD, dt, l2i_binary(ctx, POLY_OP_ADD, dt, a1, b1), l2i_cast(ctx, carry, dt)
  );
  return (L2IPair){low, high};
}

static L2IPair l2i_sub(
    PolyCtx *ctx,
    PolyDType dt,
    PolyUOp *a0,
    PolyUOp *a1,
    PolyUOp *b0,
    PolyUOp *b1
) {
  PolyUOp *low = l2i_binary(ctx, POLY_OP_SUB, dt, a0, b0);
  PolyUOp *borrow = l2i_cmp(
      ctx, POLY_OP_CMPLT, l2i_bitcast(ctx, a0, POLY_UINT32), l2i_bitcast(ctx, b0, POLY_UINT32)
  );
  PolyUOp *high = l2i_binary(
      ctx, POLY_OP_SUB, dt, l2i_binary(ctx, POLY_OP_SUB, dt, a1, b1), l2i_cast(ctx, borrow, dt)
  );
  return (L2IPair){low, high};
}

static L2IPair l2i_shift(
    PolyCtx *ctx,
    PolyOps op,
    PolyDType dt,
    PolyUOp *a0,
    PolyUOp *a1,
    PolyUOp *b0,
    PolyUOp *b1
) {
  PolyUOp *zero = l2i_const(ctx, dt, 0);
  PolyUOp *one = l2i_const(ctx, dt, 1);
  PolyUOp *thirty_one = l2i_const(ctx, dt, 31);
  PolyUOp *thirty_two = l2i_const(ctx, dt, 32);
  PolyUOp *sixty_four = l2i_const(ctx, POLY_UINT32, 64);
  PolyUOp *bmod = l2i_binary(ctx, POLY_OP_AND, dt, b0, thirty_one);
  PolyUOp *inv = l2i_binary(ctx, POLY_OP_SUB, dt, thirty_one, bmod);
  PolyUOp *b0u = l2i_bitcast(ctx, b0, POLY_UINT32);
  PolyUOp *b1u = l2i_bitcast(ctx, b1, POLY_UINT32);
  PolyUOp *high_nonzero = l2i_cmp(ctx, POLY_OP_CMPNE, b1u, l2i_const(ctx, POLY_UINT32, 0));
  PolyUOp *wide = l2i_binary(
      ctx, POLY_OP_OR, POLY_BOOL, high_nonzero,
      l2i_not(ctx, l2i_cmp(ctx, POLY_OP_CMPLT, b0u, l2i_bitcast(ctx, thirty_two, POLY_UINT32)))
  );
  PolyUOp *too_wide = l2i_binary(
      ctx, POLY_OP_OR, POLY_BOOL, high_nonzero,
      l2i_not(ctx, l2i_cmp(ctx, POLY_OP_CMPLT, b0u, sixty_four))
  );
  PolyUOp *a0u = l2i_bitcast(ctx, a0, POLY_UINT32);
  PolyUOp *a1u = l2i_bitcast(ctx, a1, POLY_UINT32);
  PolyUOp *bmodu = l2i_bitcast(ctx, bmod, POLY_UINT32);
  PolyUOp *invu = l2i_bitcast(ctx, inv, POLY_UINT32);
  PolyUOp *oneu = l2i_bitcast(ctx, one, POLY_UINT32);

  if (op == POLY_OP_SHL) {
    PolyUOp *low = l2i_bitcast(ctx, l2i_binary(ctx, POLY_OP_SHL, POLY_UINT32, a0u, bmodu), dt);
    PolyUOp *carry = l2i_binary(
        ctx, POLY_OP_SHR, POLY_UINT32, l2i_binary(ctx, POLY_OP_SHR, POLY_UINT32, a0u, oneu), invu
    );
    PolyUOp *high = l2i_bitcast(
        ctx,
        l2i_binary(
            ctx, POLY_OP_OR, POLY_UINT32, l2i_binary(ctx, POLY_OP_SHL, POLY_UINT32, a1u, bmodu),
            carry
        ),
        dt
    );
    return (L2IPair
    ){l2i_where(ctx, dt, too_wide, zero, l2i_where(ctx, dt, wide, zero, low)),
      l2i_where(ctx, dt, too_wide, zero, l2i_where(ctx, dt, wide, low, high))};
  }

  PolyUOp *carry = l2i_binary(
      ctx, POLY_OP_SHL, POLY_UINT32, l2i_binary(ctx, POLY_OP_SHL, POLY_UINT32, a1u, oneu), invu
  );
  PolyUOp *low = l2i_bitcast(
      ctx,
      l2i_binary(
          ctx, POLY_OP_OR, POLY_UINT32, l2i_binary(ctx, POLY_OP_SHR, POLY_UINT32, a0u, bmodu), carry
      ),
      dt
  );
  PolyUOp *high = l2i_binary(ctx, POLY_OP_SHR, dt, a1, bmod);
  PolyUOp *sign =
      poly_dtype_is_unsigned(dt) ? zero : l2i_binary(ctx, POLY_OP_SHR, dt, a1, thirty_one);
  return (L2IPair
  ){l2i_where(ctx, dt, too_wide, sign, l2i_where(ctx, dt, wide, high, low)),
    l2i_where(ctx, dt, too_wide, sign, l2i_where(ctx, dt, wide, sign, high))};
}

static void l2i_unpack16(PolyCtx *ctx, PolyUOp *u, PolyUOp **lo, PolyUOp **hi) {
  PolyUOp *v = l2i_bitcast(ctx, u, POLY_UINT32);
  *lo = l2i_binary(ctx, POLY_OP_AND, POLY_UINT32, v, l2i_const(ctx, POLY_UINT32, 0xffff));
  *hi = l2i_binary(ctx, POLY_OP_SHR, POLY_UINT32, v, l2i_const(ctx, POLY_UINT32, 16));
}

static L2IPair l2i_mul(
    PolyCtx *ctx,
    PolyDType dt,
    PolyUOp *a0,
    PolyUOp *a1,
    PolyUOp *b0,
    PolyUOp *b1
) {
  PolyUOp *a00 = NULL, *a01 = NULL, *b00 = NULL, *b01 = NULL;
  l2i_unpack16(ctx, a0, &a00, &a01);
  l2i_unpack16(ctx, b0, &b00, &b01);
  PolyUOp *sixteen = l2i_const(ctx, POLY_UINT32, 16);
  PolyUOp *p01 = l2i_binary(ctx, POLY_OP_MUL, POLY_UINT32, a00, b01);
  PolyUOp *p10 = l2i_binary(ctx, POLY_OP_MUL, POLY_UINT32, a01, b00);
  L2IPair mid = l2i_add(
      ctx, dt, l2i_bitcast(ctx, l2i_binary(ctx, POLY_OP_SHL, POLY_UINT32, p01, sixteen), dt),
      l2i_bitcast(ctx, l2i_binary(ctx, POLY_OP_SHR, POLY_UINT32, p01, sixteen), dt),
      l2i_bitcast(ctx, l2i_binary(ctx, POLY_OP_SHL, POLY_UINT32, p10, sixteen), dt),
      l2i_bitcast(ctx, l2i_binary(ctx, POLY_OP_SHR, POLY_UINT32, p10, sixteen), dt)
  );
  PolyUOp *low_product = l2i_bitcast(ctx, l2i_binary(ctx, POLY_OP_MUL, POLY_UINT32, a00, b00), dt);
  PolyUOp *high_product = l2i_binary(
      ctx, POLY_OP_ADD, dt,
      l2i_binary(
          ctx, POLY_OP_ADD, dt,
          l2i_bitcast(ctx, l2i_binary(ctx, POLY_OP_MUL, POLY_UINT32, a01, b01), dt),
          l2i_binary(ctx, POLY_OP_MUL, dt, a0, b1)
      ),
      l2i_binary(ctx, POLY_OP_MUL, dt, a1, b0)
  );
  return l2i_add(ctx, dt, mid.lo, mid.hi, low_product, high_product);
}

static PolyUOp *l2i_compare(
    PolyCtx *ctx,
    PolyOps op,
    PolyUOp *a0,
    PolyUOp *a1,
    PolyUOp *b0,
    PolyUOp *b1
) {
  if (op == POLY_OP_CMPEQ) {
    return l2i_binary(
        ctx, POLY_OP_AND, POLY_BOOL, l2i_cmp(ctx, POLY_OP_CMPEQ, a0, b0),
        l2i_cmp(ctx, POLY_OP_CMPEQ, a1, b1)
    );
  }
  if (op == POLY_OP_CMPNE) {
    return l2i_binary(
        ctx, POLY_OP_OR, POLY_BOOL, l2i_cmp(ctx, POLY_OP_CMPNE, a0, b0),
        l2i_cmp(ctx, POLY_OP_CMPNE, a1, b1)
    );
  }
  if (op != POLY_OP_CMPLT) return NULL;
  PolyUOp *high_lt = l2i_cmp(ctx, POLY_OP_CMPLT, a1, b1);
  PolyUOp *high_eq = l2i_cmp(ctx, POLY_OP_CMPEQ, a1, b1);
  PolyUOp *low_lt = l2i_cmp(
      ctx, POLY_OP_CMPLT, l2i_bitcast(ctx, a0, POLY_UINT32), l2i_bitcast(ctx, b0, POLY_UINT32)
  );
  return l2i_binary(
      ctx, POLY_OP_OR, POLY_BOOL, high_lt, l2i_binary(ctx, POLY_OP_AND, POLY_BOOL, high_eq, low_lt)
  );
}

static L2IPair l2i_divmod(
    PolyCtx *ctx,
    PolyOps op,
    PolyDType dt,
    PolyUOp *a0,
    PolyUOp *a1,
    PolyUOp *b0,
    PolyUOp *b1
) {
  bool is_signed = !poly_dtype_is_unsigned(dt);
  PolyUOp *a_negative = NULL, *b_negative = NULL;
  if (is_signed) {
    PolyUOp *zero = l2i_const(ctx, dt, 0);
    a_negative = l2i_cmp(ctx, POLY_OP_CMPLT, a1, zero);
    b_negative = l2i_cmp(ctx, POLY_OP_CMPLT, b1, zero);
    a0 = l2i_bitcast(ctx, a0, POLY_UINT32);
    a1 = l2i_bitcast(ctx, a1, POLY_UINT32);
    b0 = l2i_bitcast(ctx, b0, POLY_UINT32);
    b1 = l2i_bitcast(ctx, b1, POLY_UINT32);
    L2IPair an = l2i_sub(
        ctx, POLY_UINT32, l2i_const(ctx, POLY_UINT32, 0), l2i_const(ctx, POLY_UINT32, 0), a0, a1
    );
    L2IPair bn = l2i_sub(
        ctx, POLY_UINT32, l2i_const(ctx, POLY_UINT32, 0), l2i_const(ctx, POLY_UINT32, 0), b0, b1
    );
    a0 = l2i_where(ctx, POLY_UINT32, a_negative, an.lo, a0);
    a1 = l2i_where(ctx, POLY_UINT32, a_negative, an.hi, a1);
    b0 = l2i_where(ctx, POLY_UINT32, b_negative, bn.lo, b0);
    b1 = l2i_where(ctx, POLY_UINT32, b_negative, bn.hi, b1);
  }

  PolyUOp *zero = l2i_const(ctx, POLY_UINT32, 0);
  PolyUOp *one = l2i_const(ctx, POLY_UINT32, 1);
  L2IPair q = {zero, zero}, r = {zero, zero};
  for (int i = 63; i >= 0; i--) {
    r = l2i_shift(ctx, POLY_OP_SHL, POLY_UINT32, r.lo, r.hi, one, zero);
    L2IPair shifted = l2i_shift(
        ctx, POLY_OP_SHR, POLY_UINT32, a0, a1, l2i_const(ctx, POLY_UINT32, (uint32_t)i), zero
    );
    PolyUOp *incoming = l2i_binary(ctx, POLY_OP_AND, POLY_UINT32, shifted.lo, one);
    r.lo = l2i_binary(ctx, POLY_OP_OR, POLY_UINT32, r.lo, incoming);
    PolyUOp *take = l2i_not(ctx, l2i_compare(ctx, POLY_OP_CMPLT, r.lo, r.hi, b0, b1));
    L2IPair diff = l2i_sub(ctx, POLY_UINT32, r.lo, r.hi, b0, b1);
    PolyUOp *qbit = l2i_binary(
        ctx, POLY_OP_SHL, POLY_UINT32, l2i_cast(ctx, take, POLY_UINT32),
        l2i_const(ctx, POLY_UINT32, (uint32_t)(i % 32))
    );
    if (i < 32)
      q.lo = l2i_binary(ctx, POLY_OP_OR, POLY_UINT32, q.lo, qbit);
    else
      q.hi = l2i_binary(ctx, POLY_OP_OR, POLY_UINT32, q.hi, qbit);
    r.lo = l2i_where(ctx, POLY_UINT32, take, diff.lo, r.lo);
    r.hi = l2i_where(ctx, POLY_UINT32, take, diff.hi, r.hi);
  }

  if (!is_signed) {
    L2IPair ret = op == POLY_OP_CMOD ? r : q;
    return (L2IPair){l2i_bitcast(ctx, ret.lo, dt), l2i_bitcast(ctx, ret.hi, dt)};
  }

  L2IPair nq = l2i_sub(ctx, POLY_UINT32, zero, zero, q.lo, q.hi);
  L2IPair nr = l2i_sub(ctx, POLY_UINT32, zero, zero, r.lo, r.hi);
  q = (L2IPair){l2i_bitcast(ctx, q.lo, dt), l2i_bitcast(ctx, q.hi, dt)};
  r = (L2IPair){l2i_bitcast(ctx, r.lo, dt), l2i_bitcast(ctx, r.hi, dt)};
  nq = (L2IPair){l2i_bitcast(ctx, nq.lo, dt), l2i_bitcast(ctx, nq.hi, dt)};
  nr = (L2IPair){l2i_bitcast(ctx, nr.lo, dt), l2i_bitcast(ctx, nr.hi, dt)};
  if (op == POLY_OP_CMOD)
    return (L2IPair
    ){l2i_where(ctx, dt, a_negative, nr.lo, r.lo), l2i_where(ctx, dt, a_negative, nr.hi, r.hi)};
  PolyUOp *quotient_negative = l2i_binary(ctx, POLY_OP_XOR, POLY_BOOL, a_negative, b_negative);
  return (L2IPair
  ){l2i_where(ctx, dt, quotient_negative, nq.lo, q.lo),
    l2i_where(ctx, dt, quotient_negative, nq.hi, q.hi)};
}

static L2IPair l2i_alu(PolyCtx *ctx, PolyOps op, PolyDType dt, PolyUOp **u, int n) {
  PolyUOp *zero = l2i_const(ctx, dt, 0);
  if (op == POLY_OP_NEG && n == 2) return l2i_sub(ctx, dt, zero, zero, u[0], u[1]);
  if (op == POLY_OP_SHL || op == POLY_OP_SHR) {
    if (n < 4) return (L2IPair){NULL, NULL};
    return l2i_shift(ctx, op, dt, u[0], u[1], u[2], u[3]);
  }
  if (op == POLY_OP_ADD && n == 4) return l2i_add(ctx, dt, u[0], u[1], u[2], u[3]);
  if (op == POLY_OP_SUB && n == 4) return l2i_sub(ctx, dt, u[0], u[1], u[2], u[3]);
  if (op == POLY_OP_MUL && n == 4) return l2i_mul(ctx, dt, u[0], u[1], u[2], u[3]);
  if ((op == POLY_OP_CDIV || op == POLY_OP_CMOD) && n == 4)
    return l2i_divmod(ctx, op, dt, u[0], u[1], u[2], u[3]);
  if ((op == POLY_OP_XOR || op == POLY_OP_OR || op == POLY_OP_AND) && n == 4) {
    return (L2IPair){l2i_binary(ctx, op, dt, u[0], u[2]), l2i_binary(ctx, op, dt, u[1], u[3])};
  }
  if (op == POLY_OP_WHERE && n == 5) {
    return (L2IPair){l2i_where(ctx, dt, u[0], u[1], u[3]), l2i_where(ctx, dt, u[0], u[2], u[4])};
  }
  if (op == POLY_OP_MAX && n == 4) {
    PolyUOp *cond = l2i_compare(ctx, POLY_OP_CMPLT, u[0], u[1], u[2], u[3]);
    return (L2IPair){l2i_where(ctx, dt, cond, u[2], u[0]), l2i_where(ctx, dt, cond, u[3], u[1])};
  }
  if (op == POLY_OP_BITCAST && n == 2) {
    return (L2IPair){l2i_bitcast(ctx, u[0], dt), l2i_bitcast(ctx, u[1], dt)};
  }
  return (L2IPair){NULL, NULL};
}

static L2IPair l2i_cast_to_long(PolyCtx *ctx, PolyDType target, PolyUOp *src) {
  PolyDType dt = l2i_dt(target);
  PolyUOp *lo = l2i_cast(ctx, src, dt);
  PolyUOp *zero_src = poly_dtype_is_float(src->dtype)
                          ? poly_uop0(ctx, POLY_OP_CONST, src->dtype, poly_arg_float(0.0))
                          : poly_uop0(ctx, POLY_OP_CONST, src->dtype, poly_arg_int(0));
  PolyUOp *negative = l2i_cmp(ctx, POLY_OP_CMPLT, src, zero_src);
  if (!poly_dtype_is_float(src->dtype)) {
    return (L2IPair
    ){lo, l2i_where(ctx, dt, negative, l2i_const(ctx, dt, UINT32_MAX), l2i_const(ctx, dt, 0))};
  }

  PolyUOp *scale = poly_uop0(ctx, POLY_OP_CONST, src->dtype, poly_arg_float(4294967296.0));
  PolyUOp *high_float = l2i_binary(ctx, POLY_OP_FDIV, src->dtype, src, scale);
  PolyUOp *lo_nonzero = l2i_cmp(ctx, POLY_OP_CMPNE, lo, l2i_const(ctx, dt, 0));
  PolyUOp *adjust =
      l2i_cast(ctx, l2i_binary(ctx, POLY_OP_AND, POLY_BOOL, negative, lo_nonzero), dt);
  return (L2IPair){lo, l2i_binary(ctx, POLY_OP_SUB, dt, l2i_cast(ctx, high_float, dt), adjust)};
}

static PolyUOp *l2i_cast_from_long(PolyCtx *ctx, PolyDType target, PolyUOp *a0, PolyUOp *a1) {
  if (!a0 || !a1) return NULL;
  PolyDType dt = a0->dtype;
  if (!poly_dtype_is_float(target)) return l2i_cast(ctx, l2i_bitcast(ctx, a0, POLY_UINT32), target);

  PolyUOp *zero = l2i_const(ctx, dt, 0);
  PolyUOp *minus_one = l2i_const(ctx, dt, UINT32_MAX);
  PolyUOp *a0_nonnegative = l2i_not(ctx, l2i_cmp(ctx, POLY_OP_CMPLT, a0, zero));
  PolyUOp *a0_negative = l2i_cmp(ctx, POLY_OP_CMPLT, a0, zero);
  PolyUOp *small = l2i_binary(
      ctx, POLY_OP_OR, POLY_BOOL,
      l2i_binary(
          ctx, POLY_OP_AND, POLY_BOOL, l2i_cmp(ctx, POLY_OP_CMPEQ, a1, zero), a0_nonnegative
      ),
      l2i_binary(
          ctx, POLY_OP_AND, POLY_BOOL, l2i_cmp(ctx, POLY_OP_CMPEQ, a1, minus_one), a0_negative
      )
  );
  PolyUOp *direct = l2i_cast(ctx, a0, target);
  PolyUOp *scale = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(4294967296.0));
  PolyUOp *wide = l2i_binary(
      ctx, POLY_OP_ADD, POLY_FLOAT32,
      l2i_binary(ctx, POLY_OP_MUL, POLY_FLOAT32, l2i_cast(ctx, a1, POLY_FLOAT32), scale),
      l2i_cast(ctx, l2i_bitcast(ctx, a0, POLY_UINT32), POLY_FLOAT32)
  );
  return l2i_where(ctx, target, small, direct, l2i_cast(ctx, wide, target));
}

static PolyUOp *rule_long_decomp(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;

  /* INDEX consumes integer values even when the indexed buffer itself is not
   * long.  Demand the representable 32-bit address value here so the ordinary
   * lane-tagged long rules can lower its producer.  WGSL accepts i32/u32
   * indexes; an address outside both domains remains an unsupported graph. */
  if (root->op == POLY_OP_INDEX && root->n_src >= 2 && root->n_src <= 64) {
    PolyUOp *src[64];
    bool changed = false;
    for (int i = 0; i < root->n_src; i++)
      src[i] = root->src[i];
    for (int i = 1; i < root->n_src; i++) {
      PolyUOp *offset = root->src[i];
      if (!offset || !l2i_is_long(offset->dtype)) continue;
      int64_t vmin = 0, vmax = 0;
      poly_uop_minmax(ctx, offset, &vmin, &vmax);
      PolyDType target;
      if (vmin >= INT32_MIN && vmax <= INT32_MAX)
        target = POLY_INT32;
      else if (vmin >= 0 && (uint64_t)vmax <= UINT32_MAX)
        target = POLY_UINT32;
      else
        return NULL;
      PolyDType lane_dtype = l2i_dt(offset->dtype);
      PolyUOp *lo = l2i_cast(ctx, l2i_rtag(ctx, offset, 0), lane_dtype);
      PolyUOp *hi = l2i_cast(ctx, l2i_rtag(ctx, offset, 1), lane_dtype);
      src[i] = l2i_finish_value(ctx, offset, l2i_cast_from_long(ctx, target, lo, hi));
      if (!src[i]) return NULL;
      changed = true;
    }
    if (changed)
      return l2i_clone(
          ctx, root->op, root->dtype, src, root->n_src, root->arg, root->tag, root->tag_arg
      );
  }

  if (l2i_is_long_ptr(root->dtype) &&
      (root->op == POLY_OP_PARAM || root->op == POLY_OP_BUFFER || root->op == POLY_OP_INDEX)) {
    int64_t size = root->dtype.ptr_size;
    if (size > 0) {
      if (size > INT64_MAX / 2) return NULL;
      size *= 2;
    }
    PolyDType dtype = poly_dtype_ptr(l2i_dt(root->dtype), size, root->dtype.addrspace);
    dtype.vcount = root->dtype.vcount;
    return l2i_replace_dtype(ctx, root, dtype);
  }

  if (root->op == POLY_OP_STORE && root->n_src >= 2 && l2i_is_long(root->src[1]->dtype) &&
      !l2i_lane(root->src[1], NULL, NULL)) {
    if (root->n_src > 64) return NULL;
    PolyUOp *idx0 = l2i_reindex(ctx, root->src[0], 0);
    PolyUOp *idx1 = l2i_reindex(ctx, root->src[0], 1);
    PolyUOp *val0 = l2i_rtag(ctx, root->src[1], 0);
    PolyUOp *val1 = l2i_rtag(ctx, root->src[1], 1);
    if (!idx0 || !idx1 || !val0 || !val1) return NULL;
    PolyUOp *src0[64], *src1[64];
    for (int i = 0; i < root->n_src; i++)
      src0[i] = src1[i] = root->src[i];
    src0[0] = idx0;
    src0[1] = val0;
    src1[0] = idx1;
    src1[1] = val1;
    PolyUOp *stores[2] = {
        l2i_clone(
            ctx, root->op, root->dtype, src0, root->n_src, root->arg, root->tag, root->tag_arg
        ),
        l2i_clone(
            ctx, root->op, root->dtype, src1, root->n_src, root->arg, root->tag, root->tag_arg
        ),
    };
    return poly_uop(ctx, POLY_OP_GROUP, POLY_VOID, stores, 2, poly_arg_none());
  }

  if (poly_opset_has(POLY_GROUP_COMPARISON, root->op) && root->n_src == 2 &&
      l2i_is_long(root->src[0]->dtype) && l2i_is_long(root->src[1]->dtype)) {
    PolyDType dt = l2i_dt(root->src[0]->dtype);
    PolyUOp *a0 = l2i_cast(ctx, l2i_rtag(ctx, root->src[0], 0), dt);
    PolyUOp *a1 = l2i_cast(ctx, l2i_rtag(ctx, root->src[0], 1), dt);
    PolyUOp *b0 = l2i_cast(ctx, l2i_rtag(ctx, root->src[1], 0), dt);
    PolyUOp *b1 = l2i_cast(ctx, l2i_rtag(ctx, root->src[1], 1), dt);
    return l2i_finish_value(ctx, root, l2i_compare(ctx, root->op, a0, a1, b0, b1));
  }

  if (root->op == POLY_OP_CAST && root->n_src == 1 && !l2i_is_long(root->dtype) &&
      l2i_is_long(root->src[0]->dtype) && !l2i_lane(root->src[0], NULL, NULL)) {
    PolyDType dt = l2i_dt(root->src[0]->dtype);
    PolyUOp *a0 = l2i_cast(ctx, l2i_rtag(ctx, root->src[0], 0), dt);
    PolyUOp *a1 = l2i_cast(ctx, l2i_rtag(ctx, root->src[0], 1), dt);
    return l2i_finish_value(ctx, root, l2i_cast_from_long(ctx, root->dtype, a0, a1));
  }

  int lane = 0;
  if (!l2i_is_long(root->dtype) || !l2i_lane(root, &lane, NULL)) return NULL;
  PolyDType dt = l2i_dt(root->dtype);

  /* Pinned tinygrad/uop/decompositions.py:528-529 selects each 32-bit lane
   * from the exact Python integer before truncating to the lane dtype. */
  if (root->op == POLY_OP_CONST &&
      (root->arg.kind == POLY_ARG_INT || root->arg.kind == POLY_ARG_BIGINT)) {
    uint64_t bits = poly_arg_integer_to_u64_mod(root->arg);
    PolyUOp *value = l2i_const(ctx, dt, lane ? (uint32_t)(bits >> 32) : (uint32_t)bits);
    return l2i_finish_lane(ctx, root, value);
  }

  if (root->op == POLY_OP_LOAD && root->n_src >= 1 && root->n_src <= 64) {
    PolyUOp *src[64];
    for (int i = 0; i < root->n_src; i++) {
      src[i] = (i > 0 && l2i_is_long(root->src[i]->dtype)) ? l2i_rtag(ctx, root->src[i], lane)
                                                           : root->src[i];
    }
    src[0] = l2i_reindex(ctx, root->src[0], lane);
    if (!src[0]) return NULL;
    PolyUOp *value = poly_uop(ctx, POLY_OP_LOAD, dt, src, root->n_src, root->arg);
    return l2i_finish_lane(ctx, root, value);
  }

  if (root->op == POLY_OP_INDEX) {
    PolyUOp *value = l2i_reindex(ctx, root, lane);
    if (!value) return NULL;
    value = l2i_replace_dtype(ctx, value, dt);
    return l2i_finish_lane(ctx, root, value);
  }

  if (root->op == POLY_OP_CAST && root->n_src == 1) {
    L2IPair pair = {NULL, NULL};
    if (l2i_is_long(root->src[0]->dtype)) {
      PolyDType src_dt = l2i_dt(root->src[0]->dtype);
      pair.lo = l2i_bitcast(ctx, l2i_cast(ctx, l2i_rtag(ctx, root->src[0], 0), src_dt), dt);
      pair.hi = l2i_bitcast(ctx, l2i_cast(ctx, l2i_rtag(ctx, root->src[0], 1), src_dt), dt);
    } else {
      pair = l2i_cast_to_long(ctx, root->dtype, root->src[0]);
    }
    return l2i_finish_lane(ctx, root, lane ? pair.hi : pair.lo);
  }

  if ((root->op == POLY_OP_SHL || root->op == POLY_OP_SHR) && root->n_src == 2) {
    L2IPair value = {NULL, NULL}, shift = {NULL, NULL};
    if (l2i_is_long(root->src[0]->dtype)) {
      value.lo = l2i_cast(ctx, l2i_rtag(ctx, root->src[0], 0), dt);
      value.hi = l2i_cast(ctx, l2i_rtag(ctx, root->src[0], 1), dt);
    } else {
      value = l2i_cast_to_long(ctx, root->dtype, root->src[0]);
    }
    if (l2i_is_long(root->src[1]->dtype)) {
      shift.lo = l2i_cast(ctx, l2i_rtag(ctx, root->src[1], 0), dt);
      shift.hi = l2i_cast(ctx, l2i_rtag(ctx, root->src[1], 1), dt);
    } else {
      shift.lo = l2i_cast(ctx, root->src[1], dt);
      shift.hi = l2i_const(ctx, dt, 0);
    }
    L2IPair pair = l2i_shift(ctx, root->op, dt, value.lo, value.hi, shift.lo, shift.hi);
    return l2i_finish_lane(ctx, root, lane ? pair.hi : pair.lo);
  }

  if (poly_opset_has(POLY_GROUP_ALU, root->op) || root->op == POLY_OP_BITCAST) {
    PolyUOp *flat[16];
    int n_flat = 0;
    for (int i = 0; i < root->n_src; i++) {
      if (l2i_is_long(root->src[i]->dtype)) {
        if (n_flat + 2 > 16) return NULL;
        flat[n_flat++] = l2i_cast(ctx, l2i_rtag(ctx, root->src[i], 0), dt);
        flat[n_flat++] = l2i_cast(ctx, l2i_rtag(ctx, root->src[i], 1), dt);
      } else {
        if (n_flat + 1 > 16) return NULL;
        flat[n_flat++] = root->src[i];
      }
    }
    L2IPair pair = l2i_alu(ctx, root->op, dt, flat, n_flat);
    return l2i_finish_lane(ctx, root, lane ? pair.hi : pair.lo);
  }

  return NULL;
}

static _Thread_local PolyPatternMatcher *g_pm_long_decomp = NULL;

static PolyPatternMatcher *poly_pm_long_decomp(void) {
  if (g_pm_long_decomp) return g_pm_long_decomp;
  PolyRule rule = {poly_pat_any("x"), rule_long_decomp};
  g_pm_long_decomp = poly_pm_thread_cache(poly_pm_new(&rule, 1));
  return g_pm_long_decomp;
}

static PolyUOp *poly_decompose_int64(PolyCtx *ctx, PolyUOp *sink) {
  uintptr_t previous_token = g_l2i_active_token;
  uint64_t previous_generation = g_l2i_active_generation;
  unsigned char token = 0;
  uint64_t generation = ++g_l2i_generation_counter;
  if (!generation) generation = ++g_l2i_generation_counter;
  g_l2i_active_token = (uintptr_t)&token;
  g_l2i_active_generation = generation;
  PolyUOp *rewritten = poly_graph_rewrite_ex(ctx, sink, poly_pm_long_decomp(), true);
  int n = 0;
  PolyUOp **topo = rewritten ? poly_toposort(ctx, rewritten, &n) : NULL;
  for (int i = 0; topo && i < n; i++) {
    if (l2i_lane(topo[i], NULL, NULL)) {
      rewritten = NULL;
      break;
    }
  }
  g_l2i_active_token = previous_token;
  g_l2i_active_generation = previous_generation;
  return rewritten;
}

/*
 * rule_store_dtype_cast — Insert CAST when STORE value dtype mismatches buffer dtype.
 * STORE(INDEX(ptr<T>), value<U>) → STORE(INDEX(ptr<T>), CAST<T>(value)) when T != U.
 * This is a safety net: frontends should match dtypes, but if they don't, the codegen
 * pipeline normalizes it here so renderers never see cross-type stores.
 */
static PolyUOp *rule_store_dtype_cast(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  if (root->n_src < 2) return NULL;
  PolyUOp *idx = root->src[0]; /* INDEX node */
  PolyUOp *val = root->src[1]; /* value to store */
  if (!idx->dtype.is_ptr) return NULL;
  /* Extract the pointed-to value type from the pointer dtype.
   * poly_dtype_scalar only strips vector count, not is_ptr/addrspace.
   * We need a clean non-pointer scalar for the CAST target. */
  PolyDType buf_scalar = poly_dtype_scalar(idx->dtype);
  buf_scalar.is_ptr = false;
  buf_scalar.addrspace = 0;
  buf_scalar.ptr_size = 0;
  buf_scalar.vcount = 0;
  PolyDType val_scalar = poly_dtype_scalar(val->dtype);
  /* Match by priority+bitsize (not poly_dtype_eq, which checks ptr metadata) */
  if (buf_scalar.priority == val_scalar.priority && buf_scalar.bitsize == val_scalar.bitsize)
    return NULL;
  /* Insert CAST: value → buffer's scalar type (respecting vector width) */
  PolyDType cast_dt =
      (val->dtype.count > 1) ? poly_dtype_vec(buf_scalar, val->dtype.count) : buf_scalar;
  PolyUOp *casted = poly_uop1(ctx, POLY_OP_CAST, cast_dt, val, poly_arg_none());
  PolyUOp *st_srcs[64];
  int ns = 0;
  st_srcs[ns++] = idx;
  st_srcs[ns++] = casted;
  for (int i = 2; i < root->n_src && ns < 64; i++)
    st_srcs[ns++] = root->src[i];
  return poly_uop(ctx, POLY_OP_STORE, root->dtype, st_srcs, ns, root->arg);
}

/* Cached variants by renderer-supported late ops. Pinned tinygrad derives
 * this from Renderer.code_for_op before get_late_rewrite_patterns. */
static _Thread_local PolyPatternMatcher *g_pm_decomp_caps[2][2][2][2] = {{{{NULL}}}};

static PolyPatternMatcher *
poly_pm_decomp_with_caps(bool has_mulacc, bool has_max, bool has_threefry, bool has_fdiv) {
  PolyPatternMatcher **target =
      &g_pm_decomp_caps[has_mulacc ? 1 : 0][has_max ? 1 : 0][has_threefry ? 1 : 0]
                       [has_fdiv ? 1 : 0];
  if (*target) return *target;

  PolyOpSet max_set = poly_opset_add((PolyOpSet){{0, 0}}, POLY_OP_MAX);
  PolyRule rules[24];
  int n = 0;
  if (!has_max)
    rules[n++] = (PolyRule){poly_pat_ops(max_set, NULL, 0, NULL), rule_decomp_max};
  /* MUL(x:int, c:const) → SHL(x, log2(c)) when c is power of 2 */
  rules[n++] = (PolyRule
  ){poly_pat_op2(POLY_OP_MUL, poly_pat_any("x"), poly_pat_cvar("c"), NULL), rule_mul_to_shl};
  /* x * (-1) → NEG(x) */
  rules[n++] = (PolyRule
  ){poly_pat_op2(POLY_OP_MUL, poly_pat_any("x"), poly_pat_cvar("c"), NULL), rule_mul_neg1_to_neg};
  rules[n++] = (PolyRule){poly_pat_op(POLY_OP_FLOORDIV, NULL, 0, NULL), rule_floordiv_to_cdiv};
  rules[n++] = (PolyRule){poly_pat_op(POLY_OP_FLOORMOD, NULL, 0, NULL), rule_floormod_to_cmod};
  /* IDIV(x:int, c:const) → SHR(x, log2(c)) when c is power of 2 */
  rules[n++] = (PolyRule
  ){poly_pat_op2(POLY_OP_IDIV, poly_pat_any("x"), poly_pat_cvar("c"), NULL), rule_idiv_to_shr};
  /* x + NEG(y) → SUB(x, y) */
  rules[n++] = (PolyRule
  ){poly_pat_op2(
        POLY_OP_ADD, poly_pat_any("x"), poly_pat_op1(POLY_OP_NEG, poly_pat_any("y"), NULL), NULL
    ),
    rule_add_neg_to_sub};
  /* NEG(y) + x → SUB(x, y). tinygrad's UPat ADD is commutative; mirror the
   * reversed shape explicitly in C. */
  rules[n++] = (PolyRule
  ){poly_pat_op2(
        POLY_OP_ADD, poly_pat_op1(POLY_OP_NEG, poly_pat_any("y"), NULL), poly_pat_any("x"), NULL
    ),
    rule_neg_add_to_sub};

  if (!has_threefry) {
    PolyOpSet threefry_set = poly_opset_add((PolyOpSet){{0, 0}}, POLY_OP_THREEFRY);
    rules[n++] = (PolyRule){poly_pat_ops(threefry_set, NULL, 0, NULL), rule_decomp_threefry32};
  }

  if (!has_mulacc) {
    /* CPU path: decompose MULACC → MUL+ADD */
    rules[n++] = (PolyRule
    ){poly_pat_ops(poly_opset_add((PolyOpSet){{0, 0}}, POLY_OP_MULACC), NULL, 0, NULL),
      rule_mulacc_to_mul_add};
  } else {
    /* FMA path: fuse ADD(MUL(a,b), c) → MULACC(a,b,c) for floats (scalar + vector) */
    PolyOpSet add_set = poly_opset_add((PolyOpSet){{0, 0}}, POLY_OP_ADD);
    rules[n++] = (PolyRule){poly_pat_ops(add_set, NULL, 0, NULL), rule_mul_add_to_mulacc};
    /* SHL fusion: ADD(SHL(x,n), c) → MULACC(x, 2^n, c) for ints */
    /* SHL fusion: ADD(SHL(x,n), c) → MULACC(x, 2^n, c) for ints.
     * Renderer decomposes back to shl+add when profitable (x86: vpmulld is slow). */
    rules[n++] = (PolyRule){poly_pat_ops(add_set, NULL, 0, NULL), rule_shl_add_to_mulacc};
  }

  /* Pinned get_late_rewrite_patterns lowers RECIPROCAL to FDIV when FDIV is
   * advertised by the selected renderer (decompositions.py:500-503). */
  if (has_fdiv) {
    rules[n++] =
        (PolyRule){poly_pat_op1(POLY_OP_RECIPROCAL, poly_pat_any("x"), NULL), rule_recip_to_fdiv};
    /* Pinned decompositions.py:500-503 also gates a * (1 / b) → a / b on
     * FDIV renderer support. */
    rules[n++] = (PolyRule
    ){poly_pat_op2(
          POLY_OP_MUL, poly_pat_any("a"),
          poly_pat_op2(POLY_OP_FDIV, poly_pat_cvar("one"), poly_pat_any("b"), NULL), NULL
      ),
      rule_mul_fdiv1_to_fdiv};
  }
  /* Late not-CMPLT bound rewrite must run before generic not-CMPNE → CMPEQ. */
  rules[n++] = (PolyRule){poly_pat_op(POLY_OP_CMPNE, NULL, 0, NULL), rule_not_cmplt_to_bound};
  /* CMPNE(CMPNE(x,y), true) → CMPEQ(x,y), when the renderer supports CMPEQ. */
  rules[n++] = (PolyRule){poly_pat_op(POLY_OP_CMPNE, NULL, 0, NULL), rule_cmpne_not_to_cmpeq};

  /* STORE(ptr<T>, value<U>) → STORE(ptr<T>, CAST<T>(value)) when T != U */
  {
    PolyOpSet store_set = poly_opset_add((PolyOpSet){{0, 0}}, POLY_OP_STORE);
    rules[n++] = (PolyRule){poly_pat_ops(store_set, NULL, 0, NULL), rule_store_dtype_cast};
  }

  PolyPatternMatcher *late = poly_pm_new(rules, n);
  *target = poly_pm_thread_cache(poly_pm_concat(poly_symbolic_simple(), late));
  poly_pm_destroy(late);
  return *target;
}

/* Default: CPU decomp (no MULACC support). */
static PolyPatternMatcher *poly_pm_decomp(void) {
  return poly_pm_decomp_with_caps(false, false, false, true);
}

/* pm_transcendental: EXP2/LOG2/SIN → polynomial approximation */

/* Dtype-parametric helpers for IEEE 754 bit manipulation. */
static int xd_mantissa_bits(PolyDType dt) {
  int bits = poly_dtype_scalar(dt).bitsize;
  return bits == 16 ? 10 : bits == 64 ? 52 : 23;
}
static int xd_exponent_bias(PolyDType dt) {
  int bits = poly_dtype_scalar(dt).bitsize;
  return bits == 16 ? 15 : bits == 64 ? 1023 : 127;
}
static int64_t xd_exponent_mask(PolyDType dt) {
  int bits = poly_dtype_scalar(dt).bitsize;
  return bits == 16 ? 0x1FLL : bits == 64 ? 0x7FFLL : 0xFFLL;
}
static PolyDType xd_int_for_float(PolyDType dt) {
  PolyDType sdt = poly_dtype_scalar(dt);
  PolyDType it =
      sdt.bitsize == 16 ? POLY_INT16 : sdt.bitsize == 64 ? POLY_INT64 : POLY_INT32;
  return (dt.count > 1) ? poly_dtype_vec(it, dt.count) : it;
}

/* Build a polyN Horner evaluation: acc = c[0]; for i in 1..n: acc = acc*x + c[i] */
static PolyUOp *xd_polyN(
    PolyCtx *ctx,
    PolyDType ft,
    PolyUOp *x,
    const double *coeffs,
    int ncoeffs
) {
  PolyUOp *u = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(coeffs[0]));
  for (int i = 1; i < ncoeffs; i++) {
    u = poly_uop2(ctx, POLY_OP_MUL, ft, u, x, poly_arg_none());
    u = poly_uop2(
        ctx, POLY_OP_ADD, ft, u, poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(coeffs[i])),
        poly_arg_none()
    );
  }
  return u;
}

/* _lazy_map_numbers: mask +-inf/NaN to replacement values.
 * x.ne(inf).where(x.ne(x).where(nan_val, x.ne(-inf).where(ratio, ninf_val)), pinf_val) */
static PolyUOp *xd_lazy_map_numbers(
    PolyCtx *ctx,
    PolyDType ft,
    PolyUOp *d,
    PolyUOp *pinf_val,
    PolyUOp *ninf_val,
    PolyUOp *nan_val,
    PolyUOp *ratio
) {
  PolyDType bt = (ft.count > 1) ? poly_dtype_vec(POLY_BOOL, ft.count) : POLY_BOOL;
  PolyUOp *f_neg_inf = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(-__builtin_inf()));
  PolyUOp *f_pos_inf = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(__builtin_inf()));
  PolyUOp *nan_chk = poly_uop2(ctx, POLY_OP_CMPNE, bt, d, d, poly_arg_none());
  PolyUOp *neginf_chk = poly_uop2(ctx, POLY_OP_CMPNE, bt, d, f_neg_inf, poly_arg_none());
  PolyUOp *posinf_chk = poly_uop2(ctx, POLY_OP_CMPNE, bt, d, f_pos_inf, poly_arg_none());
  PolyUOp *inner = poly_uop3(ctx, POLY_OP_WHERE, ft, neginf_chk, ratio, ninf_val, poly_arg_none());
  PolyUOp *mid = poly_uop3(ctx, POLY_OP_WHERE, ft, nan_chk, nan_val, inner, poly_arg_none());
  return poly_uop3(ctx, POLY_OP_WHERE, ft, posinf_chk, mid, pinf_val, poly_arg_none());
}

/* rintk: round float d to nearest integer (away from 0). */
static PolyUOp *xd_rintk(PolyCtx *ctx, PolyDType ft, PolyDType it, PolyUOp *d) {
  PolyDType bt = (ft.count > 1) ? poly_dtype_vec(POLY_BOOL, ft.count) : POLY_BOOL;
  PolyUOp *f_zero = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(0.0));
  PolyUOp *f_neg_half = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(-0.5));
  PolyUOp *f_half = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(0.5));
  PolyUOp *lt0 = poly_uop2(ctx, POLY_OP_CMPLT, bt, d, f_zero, poly_arg_none());
  PolyUOp *offset = poly_uop3(ctx, POLY_OP_WHERE, ft, lt0, f_neg_half, f_half, poly_arg_none());
  PolyUOp *rounded = poly_uop2(ctx, POLY_OP_ADD, ft, d, offset, poly_arg_none());
  return poly_uop1(ctx, POLY_OP_CAST, it, rounded, poly_arg_none());
}

/* Pinned tinygrad uop/decompositions.py:29-32:
 * pow2if chooses its output dtype from q.dtype (int32 -> float32,
 * int64 -> float64); float_dtype is only the int16 fallback. */
static PolyUOp *xd_pow2if(PolyCtx *ctx, PolyDType ft, PolyDType it, PolyUOp *q) {
  if (!q || !poly_dtype_eq(q->dtype, it)) return NULL;
  PolyDType q_scalar = poly_dtype_scalar(q->dtype);
  PolyDType out_scalar;
  if (poly_dtype_eq(q_scalar, POLY_INT64))
    out_scalar = POLY_FLOAT64;
  else if (poly_dtype_eq(q_scalar, POLY_INT32))
    out_scalar = POLY_FLOAT32;
  else if (poly_dtype_eq(q_scalar, POLY_INT16))
    out_scalar = poly_dtype_scalar(ft);
  else
    return NULL;
  PolyDType out_ft = q->dtype.count > 1 ? poly_dtype_vec(out_scalar, q->dtype.count) : out_scalar;
  int bias = xd_exponent_bias(out_ft);
  int mbits = xd_mantissa_bits(out_ft);
  PolyUOp *i_bias = poly_uop0(ctx, POLY_OP_CONST, it, poly_arg_int(bias));
  PolyUOp *i_factor = poly_uop0(ctx, POLY_OP_CONST, it, poly_arg_int(1LL << mbits));
  PolyUOp *added = poly_uop2(ctx, POLY_OP_ADD, it, q, i_bias, poly_arg_none());
  /* tinygrad's decompositions.shl helper creates x * (2**n). Scalar lanes may
   * later fold to SHL, while vector lanes keep MUL with rendered STACK consts. */
  PolyUOp *shifted = poly_uop2(ctx, POLY_OP_MUL, it, added, i_factor, poly_arg_none());
  return poly_uop1(ctx, POLY_OP_BITCAST, out_ft, shifted, poly_arg_none());
}

static PolyUOp *xd_sub_like_tinygrad(PolyCtx *ctx, PolyDType dt, PolyUOp *a, PolyUOp *b);
static PolyUOp *xd_floordiv_positive_const(
    PolyCtx *ctx,
    PolyDType it,
    PolyUOp *x,
    int64_t divisor
);

/* Pinned tinygrad uop/decompositions.py:54-65. This retains f64 for f64
 * Payne-Hanek inputs; only f16 callers widen their intermediate to f32. */
static bool xd_frexp(
    PolyCtx *ctx,
    PolyDType ft,
    PolyUOp *v,
    PolyUOp **mantissa_out,
    PolyUOp **exponent_out
) {
  if (!v || !mantissa_out || !exponent_out) return false;
  PolyDType scalar = poly_dtype_scalar(ft);
  PolyDType bit_scalar;
  int64_t m1, m2;
  if (poly_dtype_eq(scalar, POLY_FLOAT64)) {
    bit_scalar = POLY_UINT64;
    m1 = INT64_C(0x000fffffffffffff);
    m2 = INT64_C(0x3fe0000000000000);
  } else if (poly_dtype_eq(scalar, POLY_FLOAT32)) {
    bit_scalar = POLY_UINT32;
    m1 = INT64_C(0x807fffff);
    m2 = INT64_C(0x3f000000);
  } else {
    return false;
  }
  PolyDType bit_dt = ft.count > 1 ? poly_dtype_vec(bit_scalar, ft.count) : bit_scalar;
  PolyUOp *bits = poly_uop1(ctx, POLY_OP_BITCAST, bit_dt, v, poly_arg_none());
  PolyUOp *mask = poly_uop0(ctx, POLY_OP_CONST, bit_dt, poly_arg_int(xd_exponent_mask(ft)));
  PolyUOp *exponent = poly_uop2(
      ctx, POLY_OP_AND, bit_dt,
      xd_floordiv_positive_const(ctx, bit_dt, bits, INT64_C(1) << xd_mantissa_bits(ft)), mask,
      poly_arg_none()
  );
  PolyUOp *mantissa_bits = poly_uop2(
      ctx, POLY_OP_OR, bit_dt,
      poly_uop2(
          ctx, POLY_OP_AND, bit_dt, bits, poly_uop0(ctx, POLY_OP_CONST, bit_dt, poly_arg_int(m1)),
          poly_arg_none()
      ),
      poly_uop0(ctx, POLY_OP_CONST, bit_dt, poly_arg_int(m2)), poly_arg_none()
  );
  *mantissa_out = poly_uop1(ctx, POLY_OP_BITCAST, ft, mantissa_bits, poly_arg_none());
  PolyUOp *bias =
      poly_uop0(ctx, POLY_OP_CONST, bit_dt, poly_arg_int(xd_exponent_bias(ft)));
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, bit_dt, poly_arg_int(1));
  *exponent_out = poly_uop2(
      ctx, POLY_OP_ADD, bit_dt, xd_sub_like_tinygrad(ctx, bit_dt, exponent, bias), one,
      poly_arg_none()
  );
  return true;
}

static PolyUOp *xd_sub_like_tinygrad(PolyCtx *ctx, PolyDType dt, PolyUOp *a, PolyUOp *b) {
  /* tinygrad's elementwise subtraction is built as a + (b * -1). The scalar
   * late rewrite can still collapse this to SUB, but vector paths keep the
   * MUL/ADD shape used by the reference linearizer. */
  PolyArg neg_arg = poly_dtype_is_float(dt) ? poly_arg_float(-1.0) : poly_arg_int(-1);
  PolyUOp *neg_one = poly_uop0(ctx, POLY_OP_CONST, dt, neg_arg);
  PolyUOp *neg_b = poly_uop2(ctx, POLY_OP_MUL, dt, b, neg_one, poly_arg_none());
  return poly_uop2(ctx, POLY_OP_ADD, dt, a, neg_b, poly_arg_none());
}

/* Pinned tinygrad uop/decompositions.py:18,49-52 keeps shr(e, 1) as
 * FLOORDIV at the transcendental stage. The later decomp matcher owns its
 * target-specific CDIV/CMOD lowering. */
static PolyUOp *xd_floordiv_positive_const(
    PolyCtx *ctx,
    PolyDType it,
    PolyUOp *x,
    int64_t divisor
) {
  PolyUOp *den = poly_uop0(ctx, POLY_OP_CONST, it, poly_arg_int(divisor));
  return poly_uop2(ctx, POLY_OP_FLOORDIV, it, x, den, poly_arg_none());
}

/* ldexp2k: d * 2^e. Splits e into two halves to avoid overflow in pow2if. */
static PolyUOp *xd_ldexp2k(PolyCtx *ctx, PolyDType ft, PolyDType it, PolyUOp *d, PolyUOp *e) {
  PolyUOp *half_e = xd_floordiv_positive_const(ctx, it, e, 2);
  PolyUOp *other_e = xd_sub_like_tinygrad(ctx, it, e, half_e);
  PolyUOp *pow1 = xd_pow2if(ctx, ft, it, half_e);
  PolyUOp *pow2 = xd_pow2if(ctx, ft, it, other_e);
  PolyUOp *r = poly_uop2(ctx, POLY_OP_MUL, ft, d, pow1, poly_arg_none());
  return poly_uop2(ctx, POLY_OP_MUL, ft, r, pow2, poly_arg_none());
}

/* ldexp3k: d * 2^e via bit manipulation.
 * tinygrad's helper uses shl(x, n) = x * 2**n here, not a raw Ops.SHL. The
 * late MUL->SHL rewrite may still form an immediate shift for scalar lanes. */
static PolyUOp *xd_ldexp3k(PolyCtx *ctx, PolyDType ft, PolyDType it, PolyUOp *d, PolyUOp *e) {
  int mbits = xd_mantissa_bits(ft);
  PolyUOp *factor = poly_uop0(ctx, POLY_OP_CONST, it, poly_arg_int(1LL << mbits));
  PolyUOp *d_bits = poly_uop1(ctx, POLY_OP_BITCAST, it, d, poly_arg_none());
  PolyUOp *e_int = poly_uop1(ctx, POLY_OP_CAST, it, e, poly_arg_none());
  PolyUOp *e_shift = poly_uop2(ctx, POLY_OP_MUL, it, e_int, factor, poly_arg_none());
  PolyUOp *m_bits = poly_uop2(ctx, POLY_OP_ADD, it, d_bits, e_shift, poly_arg_none());
  return poly_uop1(ctx, POLY_OP_BITCAST, ft, m_bits, poly_arg_none());
}

/* ilogb2k: integer part of log2(d) for normalized fp values.
 * tinygrad decompositions.py uses `shr(dint, mantissa_bits)`, and that helper
 * is `dint // 2**mantissa_bits`, not a raw signed right shift or truncating
 * CDIV. Build the floor-division correction explicitly, matching
 * floordiv_to_idiv before the power-of-two CDIV->SHR rewrite. */
static PolyUOp *xd_ilogb2k(PolyCtx *ctx, PolyDType ft, PolyDType it, PolyUOp *d) {
  int mbits = xd_mantissa_bits(ft);
  int64_t emask = xd_exponent_mask(ft);
  int bias = xd_exponent_bias(ft);
  PolyUOp *i_mask = poly_uop0(ctx, POLY_OP_CONST, it, poly_arg_int(emask));
  PolyUOp *i_neg_bias = poly_uop0(ctx, POLY_OP_CONST, it, poly_arg_int(-bias));
  PolyUOp *d_bits = poly_uop1(ctx, POLY_OP_BITCAST, it, d, poly_arg_none());
  PolyUOp *exp_bits = xd_floordiv_positive_const(ctx, it, d_bits, 1LL << mbits);
  PolyUOp *masked = poly_uop2(ctx, POLY_OP_AND, it, exp_bits, i_mask, poly_arg_none());
  return poly_uop2(ctx, POLY_OP_ADD, it, masked, i_neg_bias, poly_arg_none());
}

/*
 * rule_decomp_exp2 — Port of tinygrad's xexp2 from decompositions.py.
 *
 * EXP2(d) → polynomial approximation with IEEE 754 bit manipulation.
 * Supports float16/float32 (7 coefficients) and float64 (12 coefficients).
 *
 * Algorithm:
 *   1. _lazy_map_numbers: mask +-inf/NaN to 0
 *   2. rintk: round x to nearest integer q
 *   3. s = x - q (fractional part)
 *   4. polyN: Horner polynomial on s
 *   5. ldexp2k: multiply by 2^q via IEEE 754 exponent construction
 *   6. Edge cases: overflow->inf, underflow->0, NaN->NaN
 */
static PolyUOp *rule_decomp_exp2(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  PolyUOp *d = root->src[0];
  PolyDType sft = poly_dtype_scalar(root->dtype);
  bool is_f16 = poly_dtype_eq(sft, POLY_FLOAT16);
  bool is_f32 = poly_dtype_eq(sft, POLY_FLOAT32);
  bool is_f64 = poly_dtype_eq(sft, POLY_FLOAT64);
  if (!is_f16 && !is_f32 && !is_f64)
    return NULL;

  PolyDType ft = root->dtype; /* may be vec */
  PolyDType it = xd_int_for_float(ft);
  PolyDType bt = (ft.count > 1) ? poly_dtype_vec(POLY_BOOL, ft.count) : POLY_BOOL;

  /* Constants */
  PolyUOp *f_zero = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(0.0));
  PolyUOp *f_pos_inf = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(__builtin_inf()));
  PolyUOp *f_nan = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(__builtin_nan("")));
  PolyUOp *b_true = poly_uop0(ctx, POLY_OP_CONST, bt, poly_arg_bool(true));

  /* Dtype-specific overflow/underflow thresholds (from tinygrad) */
  double upper = is_f16 ? 23.0 : is_f64 ? 1024.0 : 128.0;
  double lower = is_f16 ? -22.0 : is_f64 ? -2000.0 : -150.0;
  PolyUOp *f_upper = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(upper));
  PolyUOp *f_lower = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(lower));

  /* Polynomial coefficients (from tinygrad decompositions.py) */
  static const double coeffs_f32[] = {
      0.1535920892e-3,
      0.1339262701e-2,
      0.9618384764e-2,
      0.5550347269e-1,
      0.2402264476e+0,
      0.6931471825e+0,
      1.0};
  static const double coeffs_f64[] = {
      0.4434359082926529454e-9, 0.7073164598085707425e-8, 0.1017819260921760451e-6,
      0.1321543872511327615e-5, 0.1525273353517584730e-4, 0.1540353045101147808e-3,
      0.1333355814670499073e-2, 0.9618129107597600536e-2, 0.5550410866482046596e-1,
      0.2402265069591012214e+0, 0.6931471805599452862e+0, 0.1000000000000000000e+1};
  const double *coeffs = is_f64 ? coeffs_f64 : coeffs_f32;
  int ncoeffs = is_f64 ? 12 : 7;

  /* Step 1: _lazy_map_numbers — mask +-inf/NaN to 0 */
  PolyUOp *x = xd_lazy_map_numbers(ctx, ft, d, f_zero, f_zero, f_zero, d);
  PolyUOp *nan_chk = poly_uop2(ctx, POLY_OP_CMPNE, bt, d, d, poly_arg_none());

  /* Step 2: rintk — round to nearest integer */
  PolyUOp *q = xd_rintk(ctx, ft, it, x);

  /* Step 3: fractional part s = x - q.cast(float) */
  PolyUOp *q_float = poly_uop1(ctx, POLY_OP_CAST, ft, q, poly_arg_none());
  PolyUOp *s = xd_sub_like_tinygrad(ctx, ft, x, q_float);

  /* Step 4: polyN — Horner's method */
  PolyUOp *u = xd_polyN(ctx, ft, s, coeffs, ncoeffs);

  /* Step 5: ldexp2k — u * 2^q */
  PolyUOp *result = xd_ldexp2k(ctx, ft, it, u, q);

  /* Step 6: edge cases */
  /* (d >= upper).where(inf, result) */
  PolyUOp *cmp_hi = poly_uop2(ctx, POLY_OP_CMPLT, bt, d, f_upper, poly_arg_none());
  PolyUOp *ge_hi = poly_uop2(ctx, POLY_OP_CMPNE, bt, cmp_hi, b_true, poly_arg_none());
  result = poly_uop3(ctx, POLY_OP_WHERE, ft, ge_hi, f_pos_inf, result, poly_arg_none());
  /* (d < lower).where(0, result) */
  PolyUOp *cmp_lo = poly_uop2(ctx, POLY_OP_CMPLT, bt, d, f_lower, poly_arg_none());
  result = poly_uop3(ctx, POLY_OP_WHERE, ft, cmp_lo, f_zero, result, poly_arg_none());
  /* d.ne(d).where(nan, result) — NaN propagation */
  result = poly_uop3(ctx, POLY_OP_WHERE, ft, nan_chk, f_nan, result, poly_arg_none());

  return result;
}

/* Pinned tinygrad uop/decompositions.py:get_transcendental_patterns widens
 * every float outside TRANSCENDENTAL_DTYPES (float16/32/64) to float32,
 * applies the same transcendental op, then casts back. In particular BF16 is
 * not IEEE binary16 and must never enter the half mantissa/bias path above. */
static PolyUOp *
rule_decomp_transcendental_other_float(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  if (!root || root->n_src != 1) return NULL;
  PolyDType scalar = poly_dtype_scalar(root->dtype);
  if (!poly_dtype_is_float(scalar) || poly_dtype_eq(scalar, POLY_FLOAT16) ||
      poly_dtype_eq(scalar, POLY_FLOAT32) || poly_dtype_eq(scalar, POLY_FLOAT64))
    return NULL;
  PolyDType f32 =
      root->dtype.count > 1 ? poly_dtype_vec(POLY_FLOAT32, root->dtype.count) : POLY_FLOAT32;
  PolyUOp *wide =
      poly_uop1(ctx, POLY_OP_CAST, f32, root->src[0], poly_arg_none());
  PolyUOp *transcendental =
      poly_uop1(ctx, root->op, f32, wide, poly_arg_none());
  return poly_uop1(ctx, POLY_OP_CAST, root->dtype, transcendental, poly_arg_none());
}

/*
 * rule_decomp_log2 — Port of tinygrad's xlog2.
 *
 * LOG2(d) → polynomial + IEEE754 exponent/mantissa manipulation.
 * Supports float32 (3 coefficients) and float64 (7 coefficients).
 */
static PolyUOp *rule_decomp_log2(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  PolyUOp *d = root->src[0];
  PolyDType sft = poly_dtype_scalar(root->dtype);
  if (!poly_dtype_is_float(sft) || (sft.bitsize != 32 && sft.bitsize != 64)) return NULL;

  PolyDType ft = root->dtype; /* may be vec */
  PolyDType it = xd_int_for_float(ft);
  PolyDType bt = (ft.count > 1) ? poly_dtype_vec(POLY_BOOL, ft.count) : POLY_BOOL;
  bool is_f64 = (sft.bitsize == 64);

  /* Constants */
  PolyUOp *f_zero = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(0.0));
  PolyUOp *f_neg_zero = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(-0.0));
  PolyUOp *f_one = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(1.0));
  PolyUOp *f_neg_one = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(-1.0));
  PolyUOp *f_neg_inf = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(-__builtin_inf()));
  PolyUOp *f_pos_inf = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(__builtin_inf()));
  PolyUOp *f_nan = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(__builtin_nan("")));
  PolyUOp *f_1e4 = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(1e-4));
  PolyUOp *f_4_3 = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(1.0 / 0.75));
  PolyUOp *f_neg_64 = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(-64.0));
  PolyUOp *f_2p64 = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(18446744073709551616.0));

  /* Denormal handling: scale up subnormals by 2^64 */
  PolyUOp *is_denormal = poly_uop2(ctx, POLY_OP_CMPLT, bt, d, f_1e4, poly_arg_none());
  PolyUOp *scaled = poly_uop2(ctx, POLY_OP_MUL, ft, d, f_2p64, poly_arg_none());
  PolyUOp *a = poly_uop3(ctx, POLY_OP_WHERE, ft, is_denormal, scaled, d, poly_arg_none());

  /* e = ilogb2k(a * (1/0.75)), using shared helper */
  PolyUOp *a_scaled = poly_uop2(ctx, POLY_OP_MUL, ft, a, f_4_3, poly_arg_none());
  PolyUOp *e_int = xd_ilogb2k(ctx, ft, it, a_scaled);
  PolyUOp *e = poly_uop1(ctx, POLY_OP_CAST, ft, e_int, poly_arg_none());

  /* m = ldexp3k(a, -e), using shared helper */
  PolyUOp *neg_e = poly_uop1(ctx, POLY_OP_NEG, ft, e, poly_arg_none());
  PolyUOp *m = xd_ldexp3k(ctx, ft, it, a, neg_e);

  /* Denormal exponent correction: subtract the 2^64 scaling */
  PolyUOp *e_minus64 = poly_uop2(ctx, POLY_OP_ADD, ft, e, f_neg_64, poly_arg_none());
  PolyUOp *e_adj = poly_uop3(ctx, POLY_OP_WHERE, ft, is_denormal, e_minus64, e, poly_arg_none());

  /* x = (m - 1) / (m + 1) */
  PolyUOp *m_minus1 = poly_uop2(ctx, POLY_OP_ADD, ft, m, f_neg_one, poly_arg_none());
  PolyUOp *m_plus1 = poly_uop2(ctx, POLY_OP_ADD, ft, m, f_one, poly_arg_none());
  PolyUOp *x = poly_uop2(ctx, POLY_OP_FDIV, ft, m_minus1, m_plus1, poly_arg_none());
  PolyUOp *x2 = poly_uop2(ctx, POLY_OP_MUL, ft, x, x, poly_arg_none());

  /* Polynomial: dtype-specific coefficients */
  static const double coeffs_f32[] = {0.4374550283, 0.5764790177, 0.9618012905120};
  static const double coeffs_f64[] = {0.2211941750456081490e+0, 0.2200768693152277689e+0,
                                      0.2623708057488514656e+0, 0.3205977477944495502e+0,
                                      0.4121985945485324709e+0, 0.5770780162997058982e+0,
                                      0.96179669392608091449};
  const double *coeffs = is_f64 ? coeffs_f64 : coeffs_f32;
  int ncoeffs = is_f64 ? 7 : 3;
  PolyUOp *t = xd_polyN(ctx, ft, x2, coeffs, ncoeffs);

  /* Result assembly: r = t*(x*x2) + e_adj + x*k1 [+ x*k2 for f32] */
  PolyUOp *xx2 = poly_uop2(ctx, POLY_OP_MUL, ft, x, x2, poly_arg_none());
  PolyUOp *r = poly_uop2(ctx, POLY_OP_MUL, ft, t, xx2, poly_arg_none());
  r = poly_uop2(ctx, POLY_OP_ADD, ft, r, e_adj, poly_arg_none());

  if (is_f64) {
    /* f64: single multiplier constant, no s_lo term */
    PolyUOp *f_k1 = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(2.885390081777926774));
    r = poly_uop2(
        ctx, POLY_OP_ADD, ft, r, poly_uop2(ctx, POLY_OP_MUL, ft, x, f_k1, poly_arg_none()),
        poly_arg_none()
    );
  } else {
    /* f32: k1 + s_lo term (x*k2) for extra precision */
    PolyUOp *f_k1 = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(2.8853900432586669922));
    PolyUOp *f_k2 = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(3.2734474483568488616e-08));
    r = poly_uop2(
        ctx, POLY_OP_ADD, ft, r, poly_uop2(ctx, POLY_OP_MUL, ft, x, f_k1, poly_arg_none()),
        poly_arg_none()
    );
    r = poly_uop2(
        ctx, POLY_OP_ADD, ft, r, poly_uop2(ctx, POLY_OP_MUL, ft, x, f_k2, poly_arg_none()),
        poly_arg_none()
    );
  }

  /* Edge cases (same for f32 and f64) */
  PolyUOp *ne_inf = poly_uop2(ctx, POLY_OP_CMPNE, bt, d, f_pos_inf, poly_arg_none());
  r = poly_uop3(ctx, POLY_OP_WHERE, ft, ne_inf, r, f_pos_inf, poly_arg_none());
  PolyUOp *ne_zero = poly_uop2(ctx, POLY_OP_CMPNE, bt, d, f_zero, poly_arg_none());
  r = poly_uop3(ctx, POLY_OP_WHERE, ft, ne_zero, r, f_neg_inf, poly_arg_none());
  PolyUOp *lt_neg_zero = poly_uop2(ctx, POLY_OP_CMPLT, bt, d, f_neg_zero, poly_arg_none());
  r = poly_uop3(ctx, POLY_OP_WHERE, ft, lt_neg_zero, f_nan, r, poly_arg_none());
  PolyUOp *is_nan = poly_uop2(ctx, POLY_OP_CMPNE, bt, d, d, poly_arg_none());
  r = poly_uop3(ctx, POLY_OP_WHERE, ft, is_nan, f_nan, r, poly_arg_none());
  PolyUOp *rec = poly_uop1(ctx, POLY_OP_RECIPROCAL, ft, d, poly_arg_none());
  PolyUOp *rec_ne_ninf = poly_uop2(ctx, POLY_OP_CMPNE, bt, rec, f_neg_inf, poly_arg_none());
  r = poly_uop3(ctx, POLY_OP_WHERE, ft, rec_ne_ninf, r, f_neg_inf, poly_arg_none());

  return r;
}

/* sin_poly: trig_poly from tinygrad, dtype-aware.
 * Returns d * polyN(d*d, coeffs). Supports f32 (5 coeffs) and f64 (10 coeffs). */
static PolyUOp *sin_poly(PolyCtx *ctx, PolyUOp *d) {
  PolyDType ft = d->dtype;
  /* Pinned tinygrad uop/decompositions.py:152 dispatches on
   * d.dtype.scalar(), not aggregate vector width. */
  bool is_f64 = poly_dtype_eq(poly_dtype_scalar(ft), POLY_FLOAT64);
  PolyUOp *d2 = poly_uop2(ctx, POLY_OP_MUL, ft, d, d, poly_arg_none());
  static const double coeffs_f32[] = {
      2.6083159809786593541503e-06, -0.0001981069071916863322258, 0.00833307858556509017944336,
      -0.166666597127914428710938, 1.0};
  static const double coeffs_f64[] = {-7.97255955009037868891952e-18, 2.81009972710863200091251e-15,
                                      -7.64712219118158833288484e-13, 1.60590430605664501629054e-10,
                                      -2.50521083763502045810755e-08, 2.75573192239198747630416e-06,
                                      -0.000198412698412696162806809, 0.00833333333333332974823815,
                                      -0.166666666666666657414808,    1.0};
  const double *coeffs = is_f64 ? coeffs_f64 : coeffs_f32;
  int ncoeffs = is_f64 ? 10 : 5;
  PolyUOp *t = xd_polyN(ctx, ft, d2, coeffs, ncoeffs);
  return poly_uop2(ctx, POLY_OP_MUL, ft, d, t, poly_arg_none());
}

/* Pinned tinygrad uop/decompositions.py:91-98 starts _take from
 * uint32.vec(d.dtype.count), so its constants, comparisons, and WHEREs retain
 * the input lane count. */
static PolyUOp *take_two_over_pi_f32(PolyCtx *ctx, PolyUOp *i_u64, int offset) {
  static const uint32_t two_over_pi_f[] = {0x00000000u, 0x28be60dbu, 0x9391054au, 0x7f09d5f4u,
                                           0x7d4d3770u, 0x36d8a566u, 0x4f10e410u};
  const int len = (int)(sizeof(two_over_pi_f) / sizeof(two_over_pi_f[0]));
  const int max_count = len - 2 - offset;
  int lanes = i_u64->dtype.count;
  PolyDType u64 = lanes > 1 ? poly_dtype_vec(POLY_UINT64, lanes) : POLY_UINT64;
  PolyDType u32 = lanes > 1 ? poly_dtype_vec(POLY_UINT32, lanes) : POLY_UINT32;
  PolyDType bt = lanes > 1 ? poly_dtype_vec(POLY_BOOL, lanes) : POLY_BOOL;
  PolyUOp *out = poly_uop0(ctx, POLY_OP_CONST, u32, poly_arg_int(0));
  for (int count = max_count; count >= 0; count--) {
    PolyUOp *cnt = poly_uop0(ctx, POLY_OP_CONST, u64, poly_arg_int((int64_t)count));
    PolyUOp *ne = poly_uop2(ctx, POLY_OP_CMPNE, bt, i_u64, cnt, poly_arg_none());
    PolyUOp *val = poly_uop0(
        ctx, POLY_OP_CONST, u32, poly_arg_int((int64_t)two_over_pi_f[count + offset])
    );
    out = poly_uop3(ctx, POLY_OP_WHERE, u32, ne, out, val, poly_arg_none());
  }
  return out;
}

/* tinygrad's Payne-Hanek _shl_lazy/_shr_lazy use pow2if multiply/divide
 * instead of raw dynamic SHL/SHR. This keeps the generated x86 shape aligned
 * with tinygrad's X86Renderer, which only encodes immediate scalar shifts. */
static PolyUOp *xd_lazy_shl_u32(
    PolyCtx *ctx,
    PolyDType ft,
    PolyDType it,
    PolyDType ut64,
    PolyDType ut32,
    PolyUOp *x,
    PolyUOp *y
) {
  PolyUOp *x64 = poly_uop1(ctx, POLY_OP_CAST, ut64, x, poly_arg_none());
  PolyUOp *pow = poly_uop1(ctx, POLY_OP_CAST, ut64, xd_pow2if(ctx, ft, it, y), poly_arg_none());
  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, ut64, x64, pow, poly_arg_none());
  return poly_uop1(ctx, POLY_OP_CAST, ut32, mul, poly_arg_none());
}

static PolyUOp *xd_lazy_shr_u32(
    PolyCtx *ctx,
    PolyDType ft,
    PolyDType it,
    PolyDType ut64,
    PolyDType ut32,
    PolyUOp *x,
    PolyUOp *y
) {
  PolyUOp *x64 = poly_uop1(ctx, POLY_OP_CAST, ut64, x, poly_arg_none());
  PolyUOp *pow = poly_uop1(ctx, POLY_OP_CAST, ut64, xd_pow2if(ctx, ft, it, y), poly_arg_none());
  /* Pinned tinygrad decompositions.py:104 uses `//`, retaining FLOORDIV at
   * the transcendental stage. */
  PolyUOp *div = poly_uop2(ctx, POLY_OP_FLOORDIV, ut64, x64, pow, poly_arg_none());
  return poly_uop1(ctx, POLY_OP_CAST, ut32, div, poly_arg_none());
}

/* Cody-Waite _reduce_d for f32: 4-term PI subtraction. */
static PolyUOp *cody_waite_reduce_f32(PolyCtx *ctx, PolyDType ft, PolyUOp *x, PolyUOp *qf) {
  PolyUOp *d = poly_uop2(
      ctx, POLY_OP_ADD, ft,
      poly_uop2(
          ctx, POLY_OP_MUL, ft, qf,
          poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(-3.1414794921875)), poly_arg_none()
      ),
      x, poly_arg_none()
  );
  d = poly_uop2(
      ctx, POLY_OP_ADD, ft,
      poly_uop2(
          ctx, POLY_OP_MUL, ft, qf,
          poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(-0.00011315941810607910156)),
          poly_arg_none()
      ),
      d, poly_arg_none()
  );
  d = poly_uop2(
      ctx, POLY_OP_ADD, ft,
      poly_uop2(
          ctx, POLY_OP_MUL, ft, qf,
          poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(-1.9841872589410058936e-09)),
          poly_arg_none()
      ),
      d, poly_arg_none()
  );
  d = poly_uop2(
      ctx, POLY_OP_ADD, ft,
      poly_uop2(
          ctx, POLY_OP_MUL, ft, qf,
          poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(-1.2154201256553420762e-10)),
          poly_arg_none()
      ),
      d, poly_arg_none()
  );
  return d;
}

/* Cody-Waite _reduce_d for f64: qdh/q split with 4 PI constants. */
static PolyUOp *cody_waite_reduce_f64(
    PolyCtx *ctx,
    PolyDType ft,
    PolyUOp *x,
    PolyUOp *qdh,
    PolyUOp *qf
) {
  /* PI_A..D from tinygrad sleef reference */
  static const double PI_A = 3.1415926218032836914;
  static const double PI_B = 3.1786509424591713469e-08;
  static const double PI_C = 1.2246467864107188502e-16;
  static const double PI_D = 1.2736634327021899816e-24;

  PolyUOp *pia = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(-PI_A));
  PolyUOp *pib = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(-PI_B));
  PolyUOp *pic = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(-PI_C));
  PolyUOp *pid = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(-PI_D));

  /* d = qdh * -PI_A + x */
  PolyUOp *d = poly_uop2(
      ctx, POLY_OP_ADD, ft, poly_uop2(ctx, POLY_OP_MUL, ft, qdh, pia, poly_arg_none()), x,
      poly_arg_none()
  );
  /* d = q * -PI_A + d */
  d = poly_uop2(
      ctx, POLY_OP_ADD, ft, poly_uop2(ctx, POLY_OP_MUL, ft, qf, pia, poly_arg_none()), d,
      poly_arg_none()
  );
  /* d = qdh * -PI_B + d */
  d = poly_uop2(
      ctx, POLY_OP_ADD, ft, poly_uop2(ctx, POLY_OP_MUL, ft, qdh, pib, poly_arg_none()), d,
      poly_arg_none()
  );
  /* d = q * -PI_B + d */
  d = poly_uop2(
      ctx, POLY_OP_ADD, ft, poly_uop2(ctx, POLY_OP_MUL, ft, qf, pib, poly_arg_none()), d,
      poly_arg_none()
  );
  /* d = qdh * -PI_C + d */
  d = poly_uop2(
      ctx, POLY_OP_ADD, ft, poly_uop2(ctx, POLY_OP_MUL, ft, qdh, pic, poly_arg_none()), d,
      poly_arg_none()
  );
  /* d = q * -PI_C + d */
  d = poly_uop2(
      ctx, POLY_OP_ADD, ft, poly_uop2(ctx, POLY_OP_MUL, ft, qf, pic, poly_arg_none()), d,
      poly_arg_none()
  );
  /* d = (qdh + q) * -PI_D + d */
  PolyUOp *qdh_plus_q = poly_uop2(ctx, POLY_OP_ADD, ft, qdh, qf, poly_arg_none());
  d = poly_uop2(
      ctx, POLY_OP_ADD, ft, poly_uop2(ctx, POLY_OP_MUL, ft, qdh_plus_q, pid, poly_arg_none()), d,
      poly_arg_none()
  );
  return d;
}

/*
 * rule_decomp_sin — Port of tinygrad's xsin.
 *
 * SIN(d) → sign handling + Cody-Waite / Payne-Hanek reduction + polynomial.
 * Supports float32 and float64.
 *
 * f32: Cody-Waite (small) + Payne-Hanek (large), switchover at 30.0
 * f64: Cody-Waite with qdh precision split (small) + Payne-Hanek (large)
 */
static PolyUOp *rule_decomp_sin(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  PolyUOp *d = root->src[0];
  PolyDType sft = poly_dtype_scalar(root->dtype);
  if (!poly_dtype_is_float(sft) || (sft.bitsize != 32 && sft.bitsize != 64)) return NULL;

  PolyDType ft = root->dtype; /* may be vec */
  int vc = ft.count;
  PolyDType it = (vc > 1) ? poly_dtype_vec(POLY_INT32, vc) : POLY_INT32;
  PolyDType ut32 = (vc > 1) ? poly_dtype_vec(POLY_UINT32, vc) : POLY_UINT32;
  PolyDType ut64 = (vc > 1) ? poly_dtype_vec(POLY_UINT64, vc) : POLY_UINT64;
  PolyDType bt = (vc > 1) ? poly_dtype_vec(POLY_BOOL, vc) : POLY_BOOL;
  bool is_f64 = (sft.bitsize == 64);

  /* Common constants */
  PolyUOp *f_zero = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(0.0));
  PolyUOp *f_one = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(1.0));
  PolyUOp *f_neg_one = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(-1.0));
  PolyUOp *f_pi_2 = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(1.57079632679489661923));
  PolyUOp *f_half = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(0.5));
  PolyUOp *f_switch = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(30.0));
  PolyUOp *f_pos_inf = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(__builtin_inf()));
  PolyUOp *f_neg_inf = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(-__builtin_inf()));
  PolyUOp *f_nan = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(__builtin_nan("")));
  double m_1_pi = 0.318309886183790671537767526745028724;
  PolyUOp *f_m_1_pi = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(m_1_pi));
  PolyUOp *f_ph_mul = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(3.4061215800865545e-19));
  PolyUOp *i_zero = poly_uop0(ctx, POLY_OP_CONST, it, poly_arg_int(0));
  PolyUOp *i_one = poly_uop0(ctx, POLY_OP_CONST, it, poly_arg_int(1));
  PolyUOp *i_two = poly_uop0(ctx, POLY_OP_CONST, it, poly_arg_int(2));
  PolyUOp *i_31 = poly_uop0(ctx, POLY_OP_CONST, it, poly_arg_int(31));
  PolyUOp *i_32 = poly_uop0(ctx, POLY_OP_CONST, it, poly_arg_int(32));
  PolyUOp *u_mask = poly_uop0(ctx, POLY_OP_CONST, ut64, poly_arg_int(0x3fffffffffffffffULL));

  /* _lazy_map_numbers(d, 0, 0, 0, d) */
  PolyUOp *d_ne_pos_inf = poly_uop2(ctx, POLY_OP_CMPNE, bt, d, f_pos_inf, poly_arg_none());
  PolyUOp *d_is_nan = poly_uop2(ctx, POLY_OP_CMPNE, bt, d, d, poly_arg_none());
  PolyUOp *d_ne_neg_inf = poly_uop2(ctx, POLY_OP_CMPNE, bt, d, f_neg_inf, poly_arg_none());
  PolyUOp *x_inner = poly_uop3(ctx, POLY_OP_WHERE, ft, d_ne_neg_inf, d, f_zero, poly_arg_none());
  PolyUOp *x_mid = poly_uop3(ctx, POLY_OP_WHERE, ft, d_is_nan, f_zero, x_inner, poly_arg_none());
  PolyUOp *x = poly_uop3(ctx, POLY_OP_WHERE, ft, d_ne_pos_inf, x_mid, f_zero, poly_arg_none());

  /* x_sign = x!=0 ? (x<0 ? -1 : 1) : 0 */
  PolyUOp *x_ne0 = poly_uop2(ctx, POLY_OP_CMPNE, bt, x, f_zero, poly_arg_none());
  PolyUOp *x_lt0 = poly_uop2(ctx, POLY_OP_CMPLT, bt, x, f_zero, poly_arg_none());
  PolyUOp *x_pm = poly_uop3(ctx, POLY_OP_WHERE, ft, x_lt0, f_neg_one, f_one, poly_arg_none());
  PolyUOp *x_sign = poly_uop3(ctx, POLY_OP_WHERE, ft, x_ne0, x_pm, f_zero, poly_arg_none());
  PolyUOp *x_abs = poly_uop2(ctx, POLY_OP_MUL, ft, x, x_sign, poly_arg_none());

  /* Cody-Waite reduction (small branch) */
  PolyUOp *q_small;
  PolyUOp *r_small;

  if (is_f64) {
    /* f64: qdh = (x_abs * (m_1_pi / 2^24)).cast(int64).cast(f64) * 2^24 */
    PolyUOp *f_m1pi_div2p24 =
        poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(m_1_pi / 16777216.0)); /* m_1_pi / 2^24 */
    PolyUOp *f_2p24 = poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(16777216.0));
    /* Pinned tinygrad UOp.cast (uop/ops.py:513-516) preserves the source
     * vector count when the requested cast dtype is scalar. */
    PolyDType it64 = vc > 1 ? poly_dtype_vec(POLY_INT64, vc) : POLY_INT64;
    PolyUOp *qdh_raw = poly_uop2(ctx, POLY_OP_MUL, ft, x_abs, f_m1pi_div2p24, poly_arg_none());
    PolyUOp *qdh_int = poly_uop1(ctx, POLY_OP_CAST, it64, qdh_raw, poly_arg_none());
    PolyUOp *qdh = poly_uop2(
        ctx, POLY_OP_MUL, ft, poly_uop1(ctx, POLY_OP_CAST, ft, qdh_int, poly_arg_none()), f_2p24,
        poly_arg_none()
    );

    /* quadrant = rintk(x_abs * m_1_pi - qdh) */
    PolyUOp *qf_raw = xd_sub_like_tinygrad(
        ctx, ft, poly_uop2(ctx, POLY_OP_MUL, ft, x_abs, f_m_1_pi, poly_arg_none()), qdh
    );
    /* Pinned tinygrad decompositions.py:147-150: rintk(float64) returns
     * int64; only the returned quadrant is narrowed to int32. */
    PolyUOp *q_small_i64 = xd_rintk(ctx, ft, it64, qf_raw);
    PolyUOp *qf = poly_uop1(ctx, POLY_OP_CAST, ft, q_small_i64, poly_arg_none());

    r_small = cody_waite_reduce_f64(ctx, ft, x_abs, qdh, qf);
    q_small = poly_uop1(ctx, POLY_OP_CAST, it, q_small_i64, poly_arg_none());
  } else {
    /* f32: simple rintk(x_abs * m_1_pi) */
    PolyUOp *qf_raw = poly_uop2(ctx, POLY_OP_MUL, ft, x_abs, f_m_1_pi, poly_arg_none());
    q_small = xd_rintk(ctx, ft, it, qf_raw);
    PolyUOp *qf = poly_uop1(ctx, POLY_OP_CAST, ft, q_small, poly_arg_none());

    r_small = cody_waite_reduce_f32(ctx, ft, x_abs, qf);
  }

  /* Payne-Hanek reduction. Pinned tinygrad keeps f64 intermediates for f64
   * inputs; only f16 widens to f32 (unsupported by this rule today). */
  PolyUOp *f_frexp = NULL, *e_raw = NULL;
  if (!xd_frexp(ctx, ft, x_abs, &f_frexp, &e_raw)) return NULL;
  PolyUOp *e_i = poly_uop2(
      ctx, POLY_OP_AND, it, poly_uop1(ctx, POLY_OP_CAST, it, e_raw, poly_arg_none()), i_31,
      poly_arg_none()
  );
  PolyUOp *ia = poly_uop1(
      ctx, POLY_OP_CAST, ut64,
      poly_uop2(
          ctx, POLY_OP_MUL, ft, f_frexp,
          poly_uop0(ctx, POLY_OP_CONST, ft, poly_arg_float(4294967296.0)), poly_arg_none()
      ),
      poly_arg_none()
  );
  PolyUOp *i_u64 = xd_floordiv_positive_const(
      ctx, ut64, poly_uop1(ctx, POLY_OP_CAST, ut64, e_raw, poly_arg_none()), INT64_C(1) << 5
  );
  PolyUOp *offset = xd_sub_like_tinygrad(ctx, it, i_32, e_i);

  PolyUOp *a0 = take_two_over_pi_f32(ctx, i_u64, 0);
  PolyUOp *a1 = take_two_over_pi_f32(ctx, i_u64, 1);
  PolyUOp *a2 = take_two_over_pi_f32(ctx, i_u64, 2);
  PolyUOp *a3 = take_two_over_pi_f32(ctx, i_u64, 3);
  PolyUOp *hi = poly_uop2(
      ctx, POLY_OP_OR, ut32, xd_lazy_shl_u32(ctx, ft, it, ut64, ut32, a0, e_i),
      xd_lazy_shr_u32(ctx, ft, it, ut64, ut32, a1, offset), poly_arg_none()
  );
  PolyUOp *mi = poly_uop2(
      ctx, POLY_OP_OR, ut32, xd_lazy_shl_u32(ctx, ft, it, ut64, ut32, a1, e_i),
      xd_lazy_shr_u32(ctx, ft, it, ut64, ut32, a2, offset), poly_arg_none()
  );
  PolyUOp *lo = poly_uop2(
      ctx, POLY_OP_OR, ut32, xd_lazy_shl_u32(ctx, ft, it, ut64, ut32, a2, e_i),
      xd_lazy_shr_u32(ctx, ft, it, ut64, ut32, a3, offset), poly_arg_none()
  );

  PolyUOp *hp_hi = poly_uop2(
      ctx, POLY_OP_MUL, ut64, ia, poly_uop1(ctx, POLY_OP_CAST, ut64, hi, poly_arg_none()),
      poly_arg_none()
  );
  PolyUOp *hp_mi = poly_uop2(
      ctx, POLY_OP_MUL, ut64, ia, poly_uop1(ctx, POLY_OP_CAST, ut64, mi, poly_arg_none()),
      poly_arg_none()
  );
  PolyUOp *hp_lo = poly_uop2(
      ctx, POLY_OP_MUL, ut64, ia, poly_uop1(ctx, POLY_OP_CAST, ut64, lo, poly_arg_none()),
      poly_arg_none()
  );
  PolyUOp *p = poly_uop2(
      ctx, POLY_OP_ADD, ut64,
      poly_uop2(
          ctx, POLY_OP_ADD, ut64,
          poly_uop2(
              ctx, POLY_OP_MUL, ut64, hp_hi,
              poly_uop0(ctx, POLY_OP_CONST, ut64, poly_arg_int(INT64_C(1) << 32)),
              poly_arg_none()
          ),
          hp_mi, poly_arg_none()
      ),
      xd_floordiv_positive_const(ctx, ut64, hp_lo, INT64_C(1) << 32), poly_arg_none()
  );
  PolyUOp *q_ph = poly_uop1(
      ctx, POLY_OP_CAST, it,
      xd_floordiv_positive_const(ctx, ut64, p, INT64_C(1) << 62), poly_arg_none()
  );
  PolyUOp *p_masked = poly_uop2(ctx, POLY_OP_AND, ut64, p, u_mask, poly_arg_none());
  PolyUOp *r_ph_base = poly_uop2(
      ctx, POLY_OP_MUL, ft, poly_uop1(ctx, POLY_OP_CAST, ft, p_masked, poly_arg_none()), f_ph_mul,
      poly_arg_none()
  );
  PolyUOp *f_lt_half = poly_uop2(ctx, POLY_OP_CMPLT, bt, f_frexp, f_half, poly_arg_none());
  PolyUOp *r_ph = poly_uop3(
      ctx, POLY_OP_WHERE, ft, f_lt_half, r_ph_base,
      xd_sub_like_tinygrad(ctx, ft, r_ph_base, f_pi_2), poly_arg_none()
  );
  q_ph = poly_uop3(
      ctx, POLY_OP_WHERE, it, f_lt_half, q_ph,
      poly_uop2(ctx, POLY_OP_ADD, it, q_ph, i_one, poly_arg_none()), poly_arg_none()
  );

  /* sin_poly_small / sin_poly_large, split at switch_over */
  PolyUOp *q_small_odd = poly_uop2(
      ctx, POLY_OP_CMPNE, bt, poly_uop2(ctx, POLY_OP_AND, it, q_small, i_one, poly_arg_none()),
      i_zero, poly_arg_none()
  );
  PolyUOp *small_sign =
      poly_uop3(ctx, POLY_OP_WHERE, ft, q_small_odd, f_neg_one, f_one, poly_arg_none());
  PolyUOp *result_small =
      poly_uop2(ctx, POLY_OP_MUL, ft, sin_poly(ctx, r_small), small_sign, poly_arg_none());

  PolyUOp *q_ph_odd = poly_uop2(
      ctx, POLY_OP_CMPNE, bt, poly_uop2(ctx, POLY_OP_AND, it, q_ph, i_one, poly_arg_none()), i_zero,
      poly_arg_none()
  );
  PolyUOp *large_arg = poly_uop2(
      ctx, POLY_OP_ADD, ft, r_ph,
      poly_uop3(ctx, POLY_OP_WHERE, ft, q_ph_odd, f_pi_2, f_zero, poly_arg_none()), poly_arg_none()
  );
  PolyUOp *q_ph_bit2 = poly_uop2(
      ctx, POLY_OP_CMPNE, bt, poly_uop2(ctx, POLY_OP_AND, it, q_ph, i_two, poly_arg_none()), i_zero,
      poly_arg_none()
  );
  PolyUOp *large_sign =
      poly_uop3(ctx, POLY_OP_WHERE, ft, q_ph_bit2, f_neg_one, f_one, poly_arg_none());
  PolyUOp *result_large =
      poly_uop2(ctx, POLY_OP_MUL, ft, sin_poly(ctx, large_arg), large_sign, poly_arg_none());

  PolyUOp *use_small = poly_uop2(ctx, POLY_OP_CMPLT, bt, x_abs, f_switch, poly_arg_none());
  PolyUOp *result =
      poly_uop3(ctx, POLY_OP_WHERE, ft, use_small, result_small, result_large, poly_arg_none());

  /* Restore original sign */
  result = poly_uop2(ctx, POLY_OP_MUL, ft, result, x_sign, poly_arg_none());

  /* _lazy_map_numbers(d, nan, nan, nan, result) */
  PolyUOp *out_inner =
      poly_uop3(ctx, POLY_OP_WHERE, ft, d_ne_neg_inf, result, f_nan, poly_arg_none());
  PolyUOp *out_mid = poly_uop3(ctx, POLY_OP_WHERE, ft, d_is_nan, f_nan, out_inner, poly_arg_none());
  PolyUOp *out = poly_uop3(ctx, POLY_OP_WHERE, ft, d_ne_pos_inf, out_mid, f_nan, poly_arg_none());
  return out;
}

/* BF16 non-native type rewrites * Mirrors tinygrad's create_non_native_float_pats() +
 * pm_manual_bf16_cast. BF16 is stored as unsigned short on most targets (HIP, OpenCL, CPU). ALU ops
 * must be promoted to float32, and CAST bf16<->f32 uses bitwise ops.
 */
static bool is_bf16(PolyDType dt) {
  if (dt.is_ptr) return false;
  PolyDType s = poly_dtype_scalar(dt);
  return s.priority == POLY_BFLOAT16.priority && s.bitsize == 16;
}

static bool is_bf16_ptr(PolyDType dt) {
  if (!dt.is_ptr) return false;
  /* Vectorized pointer dtypes retain the element identity but multiply
   * bitsize/count. Match the scalar storage element, as tinygrad's
   * f2f_store checks the INDEX pointer tag (uop/decompositions.py:423-429). */
  PolyDType s = poly_dtype_scalar(dt);
  s.is_ptr = false;
  s.addrspace = 0;
  s.vcount = 0;
  s.ptr_size = 0;
  return s.priority == POLY_BFLOAT16.priority && s.bitsize == 16;
}

static bool is_f16(PolyDType dt) {
  if (dt.is_ptr) return false;
  PolyDType s = poly_dtype_scalar(dt);
  return s.priority == POLY_FLOAT16.priority && s.bitsize == 16;
}

static bool is_f64(PolyDType dt) {
  if (dt.is_ptr) return false;
  PolyDType s = poly_dtype_scalar(dt);
  return s.priority == POLY_FLOAT64.priority && s.bitsize == 64;
}

static PolyDType dtype_count_like(PolyDType scalar, PolyDType dt) {
  return dt.count > 1 ? poly_dtype_vec(scalar, dt.count) : scalar;
}

static PolyDType float32_like(PolyDType dt) {
  return dtype_count_like(POLY_FLOAT32, dt);
}

static PolyUOp *float_decomp_clone(PolyCtx *ctx, PolyUOp *u, PolyDType dtype, PolyUOp **src) {
  return (u->tag != 0 || u->tag_arg.kind != POLY_ARG_NONE)
             ? poly_uop_tagged_arg(
                   ctx, u->op, dtype, src ? src : u->src, u->n_src, u->arg, u->tag, u->tag_arg
               )
             : poly_uop(ctx, u->op, dtype, src ? src : u->src, u->n_src, u->arg);
}

/* Pinned tinygrad uop/decompositions.py:f2f_load/f2f_store retag BF16
 * storage as uint16 before emulating the value in float32. */
static PolyUOp *bf16_storage_ref_as_u16(PolyCtx *ctx, PolyUOp *u) {
  if (!u || (!is_bf16_ptr(u->dtype) && !is_bf16(u->dtype))) return u;
  PolyUOp **src = NULL;
  if (u->n_src > 0) {
    src = malloc((size_t)u->n_src * sizeof(PolyUOp *));
    if (!src) return NULL;
    for (int i = 0; i < u->n_src; i++)
      src[i] = (is_bf16_ptr(u->src[i]->dtype) || is_bf16(u->src[i]->dtype))
                   ? bf16_storage_ref_as_u16(ctx, u->src[i])
                   : u->src[i];
  }
  PolyDType dtype;
  if (u->dtype.is_ptr) {
    PolyDType storage =
        u->dtype.count > 1 ? poly_dtype_vec(POLY_UINT16, u->dtype.count) : POLY_UINT16;
    dtype = poly_dtype_ptr(storage, u->dtype.ptr_size, u->dtype.addrspace);
    dtype.vcount = u->dtype.vcount;
  } else {
    dtype = u->dtype.count > 1 ? poly_dtype_vec(POLY_UINT16, u->dtype.count) : POLY_UINT16;
  }
  PolyUOp *out = float_decomp_clone(ctx, u, dtype, src);
  free(src);
  return out;
}

/* Pinned tinygrad uop/decompositions.py:f2f for BF16 -> float32. BF16
 * denormals flush to zero; finite values and NaN payload bits are expanded
 * through the uint16 storage representation. */
static PolyUOp *bf16_bits_to_f32(PolyCtx *ctx, PolyUOp *raw_u16) {
  PolyDType u32dt = dtype_count_like(POLY_UINT32, raw_u16->dtype);
  PolyDType booldt = dtype_count_like(POLY_BOOL, raw_u16->dtype);
  PolyDType f32dt = float32_like(raw_u16->dtype);
  PolyUOp *u32 = poly_uop1(ctx, POLY_OP_CAST, u32dt, raw_u16, poly_arg_none());
  PolyUOp *c0 = poly_uop0(ctx, POLY_OP_CONST, u32dt, poly_arg_int(0));
  PolyUOp *c7 = poly_uop0(ctx, POLY_OP_CONST, u32dt, poly_arg_int(7));
  PolyUOp *c16 = poly_uop0(ctx, POLY_OP_CONST, u32dt, poly_arg_int(16));
  PolyUOp *c255 = poly_uop0(ctx, POLY_OP_CONST, u32dt, poly_arg_int(255));
  PolyUOp *sign_mask = poly_uop0(ctx, POLY_OP_CONST, u32dt, poly_arg_int(0x8000));
  PolyUOp *value_mask = poly_uop0(ctx, POLY_OP_CONST, u32dt, poly_arg_int(0x7fff));
  PolyUOp *inf_bits = poly_uop0(ctx, POLY_OP_CONST, u32dt, poly_arg_int(0x7f800000));

  PolyUOp *sign = poly_uop2(
      ctx, POLY_OP_SHL, u32dt, poly_uop2(ctx, POLY_OP_AND, u32dt, u32, sign_mask, poly_arg_none()),
      c16, poly_arg_none()
  );
  PolyUOp *nosign = poly_uop2(ctx, POLY_OP_AND, u32dt, u32, value_mask, poly_arg_none());
  PolyUOp *exp = poly_uop2(ctx, POLY_OP_SHR, u32dt, nosign, c7, poly_arg_none());
  PolyUOp *norm = poly_uop2(ctx, POLY_OP_SHL, u32dt, nosign, c16, poly_arg_none());
  PolyUOp *nan_bits = poly_uop2(ctx, POLY_OP_OR, u32dt, norm, inf_bits, poly_arg_none());
  PolyUOp *exp_zero = poly_uop2(ctx, POLY_OP_CMPEQ, booldt, exp, c0, poly_arg_none());
  PolyUOp *exp_nan = poly_uop2(ctx, POLY_OP_CMPEQ, booldt, exp, c255, poly_arg_none());
  PolyUOp *magnitude = poly_uop3(
      ctx, POLY_OP_WHERE, u32dt, exp_zero, c0,
      poly_uop3(ctx, POLY_OP_WHERE, u32dt, exp_nan, nan_bits, norm, poly_arg_none()),
      poly_arg_none()
  );
  return poly_uop1(
      ctx, POLY_OP_BITCAST, f32dt,
      poly_uop2(ctx, POLY_OP_OR, u32dt, sign, magnitude, poly_arg_none()), poly_arg_none()
  );
}

/* Pinned tinygrad uop/decompositions.py:f2f_clamp for float32 -> BF16. */
static PolyUOp *bf16_clamp_f32(PolyCtx *ctx, PolyUOp *x) {
  PolyDType f32dt = float32_like(x->dtype);
  PolyDType booldt = dtype_count_like(POLY_BOOL, x->dtype);
  PolyUOp *max = poly_uop0(ctx, POLY_OP_CONST, f32dt, poly_arg_float(3.3895313892515355e38));
  PolyUOp *neg_max = poly_uop1(ctx, POLY_OP_NEG, f32dt, max, poly_arg_none());
  PolyUOp *pos_inf = poly_uop0(ctx, POLY_OP_CONST, f32dt, poly_arg_float(INFINITY));
  PolyUOp *neg_inf = poly_uop0(ctx, POLY_OP_CONST, f32dt, poly_arg_float(-INFINITY));
  PolyUOp *is_nan = poly_uop2(ctx, POLY_OP_CMPNE, booldt, x, x, poly_arg_none());
  PolyUOp *below = poly_uop2(ctx, POLY_OP_CMPLT, booldt, x, neg_max, poly_arg_none());
  PolyUOp *above = poly_uop2(ctx, POLY_OP_CMPLT, booldt, max, x, poly_arg_none());
  PolyUOp *bounded = poly_uop3(
      ctx, POLY_OP_WHERE, f32dt, below, neg_inf,
      poly_uop3(ctx, POLY_OP_WHERE, f32dt, above, pos_inf, x, poly_arg_none()), poly_arg_none()
  );
  return poly_uop3(ctx, POLY_OP_WHERE, f32dt, is_nan, x, bounded, poly_arg_none());
}

/* Pinned tinygrad uop/decompositions.py:f2f for float32 -> BF16 storage
 * bits, including round-to-nearest-even and NaN payload preservation. */
static PolyUOp *f32_to_bf16_bits(PolyCtx *ctx, PolyUOp *x) {
  x = bf16_clamp_f32(ctx, x);
  PolyDType u32dt = dtype_count_like(POLY_UINT32, x->dtype);
  PolyDType u16dt = dtype_count_like(POLY_UINT16, x->dtype);
  PolyDType booldt = dtype_count_like(POLY_BOOL, x->dtype);
  PolyUOp *v = poly_uop1(ctx, POLY_OP_BITCAST, u32dt, x, poly_arg_none());
  PolyUOp *c0 = poly_uop0(ctx, POLY_OP_CONST, u32dt, poly_arg_int(0));
  PolyUOp *c1 = poly_uop0(ctx, POLY_OP_CONST, u32dt, poly_arg_int(1));
  PolyUOp *c7fff = poly_uop0(ctx, POLY_OP_CONST, u32dt, poly_arg_int(0x7fff));
  PolyUOp *c7fffffff = poly_uop0(ctx, POLY_OP_CONST, u32dt, poly_arg_int(0x7fffffff));
  PolyUOp *c7f80 = poly_uop0(ctx, POLY_OP_CONST, u32dt, poly_arg_int(0x7f80));
  PolyUOp *c7f = poly_uop0(ctx, POLY_OP_CONST, u32dt, poly_arg_int(0x7f));
  PolyUOp *c8000 = poly_uop0(ctx, POLY_OP_CONST, u32dt, poly_arg_int(0x8000));
  PolyUOp *c15 = poly_uop0(ctx, POLY_OP_CONST, u32dt, poly_arg_int(15));
  PolyUOp *c16 = poly_uop0(ctx, POLY_OP_CONST, u32dt, poly_arg_int(16));
  PolyUOp *c23 = poly_uop0(ctx, POLY_OP_CONST, u32dt, poly_arg_int(23));
  PolyUOp *c255 = poly_uop0(ctx, POLY_OP_CONST, u32dt, poly_arg_int(255));

  PolyUOp *sign = poly_uop2(
      ctx, POLY_OP_AND, u32dt, poly_uop2(ctx, POLY_OP_SHR, u32dt, v, c16, poly_arg_none()), c8000,
      poly_arg_none()
  );
  PolyUOp *nosign = poly_uop2(ctx, POLY_OP_AND, u32dt, v, c7fffffff, poly_arg_none());
  PolyUOp *q = poly_uop2(ctx, POLY_OP_SHR, u32dt, nosign, c16, poly_arg_none());
  PolyUOp *round_bit = poly_uop2(
      ctx, POLY_OP_AND, u32dt, poly_uop2(ctx, POLY_OP_SHR, u32dt, nosign, c15, poly_arg_none()), c1,
      poly_arg_none()
  );
  PolyUOp *tail = poly_uop2(
      ctx, POLY_OP_CMPNE, booldt,
      poly_uop2(ctx, POLY_OP_AND, u32dt, nosign, c7fff, poly_arg_none()), c0, poly_arg_none()
  );
  PolyUOp *q_odd = poly_uop2(ctx, POLY_OP_AND, u32dt, q, c1, poly_arg_none());
  PolyUOp *sticky = poly_uop2(
      ctx, POLY_OP_OR, u32dt, poly_uop1(ctx, POLY_OP_CAST, u32dt, tail, poly_arg_none()), q_odd,
      poly_arg_none()
  );
  PolyUOp *rounded = poly_uop2(
      ctx, POLY_OP_ADD, u32dt, q,
      poly_uop2(ctx, POLY_OP_AND, u32dt, round_bit, sticky, poly_arg_none()), poly_arg_none()
  );
  PolyUOp *norm = poly_uop1(ctx, POLY_OP_CAST, u16dt, rounded, poly_arg_none());
  PolyUOp *exp = poly_uop2(
      ctx, POLY_OP_AND, u32dt, poly_uop2(ctx, POLY_OP_SHR, u32dt, v, c23, poly_arg_none()), c255,
      poly_arg_none()
  );
  PolyUOp *underflow = poly_uop2(ctx, POLY_OP_CMPLT, booldt, exp, c1, poly_arg_none());
  PolyUOp *is_nan = poly_uop2(ctx, POLY_OP_CMPEQ, booldt, exp, c255, poly_arg_none());
  PolyUOp *nan_bits = poly_uop1(
      ctx, POLY_OP_CAST, u16dt,
      poly_uop2(
          ctx, POLY_OP_OR, u32dt,
          poly_uop2(
              ctx, POLY_OP_OR, u32dt, sign,
              poly_uop2(
                  ctx, POLY_OP_AND, u32dt,
                  poly_uop2(ctx, POLY_OP_SHR, u32dt, nosign, c16, poly_arg_none()), c7f,
                  poly_arg_none()
              ),
              poly_arg_none()
          ),
          c7f80, poly_arg_none()
      ),
      poly_arg_none()
  );
  PolyUOp *sign_u16 = poly_uop1(ctx, POLY_OP_CAST, u16dt, sign, poly_arg_none());
  PolyUOp *zero_u16 = poly_uop0(ctx, POLY_OP_CONST, u16dt, poly_arg_int(0));
  PolyUOp *finite = poly_uop2(
      ctx, POLY_OP_OR, u16dt, sign_u16,
      poly_uop3(ctx, POLY_OP_WHERE, u16dt, underflow, zero_u16, norm, poly_arg_none()),
      poly_arg_none()
  );
  return poly_uop3(ctx, POLY_OP_WHERE, u16dt, is_nan, nan_bits, finite, poly_arg_none());
}

/* CPU C renderer parity with tinygrad ClangRenderer.extra_matcher:
 * avoid backend compiler-rt/runtime helper calls for double->half/bf16 and
 * bf16->half by lowering through float32 before final render. */
static PolyUOp *rule_c_renderer_cast_via_f32(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  if (root->op != POLY_OP_CAST || root->n_src < 1) return NULL;
  PolyDType src_dt = root->src[0]->dtype;
  PolyDType dst_dt = root->dtype;
  if (!((is_f64(src_dt) && (is_f16(dst_dt) || is_bf16(dst_dt))) ||
        (is_bf16(src_dt) && is_f16(dst_dt))))
    return NULL;
  PolyUOp *f32 = poly_uop1(ctx, POLY_OP_CAST, float32_like(src_dt), root->src[0], poly_arg_none());
  return poly_uop1(ctx, POLY_OP_CAST, dst_dt, f32, poly_arg_none());
}

/* Pinned tinygrad pm_float_decomp GroupOp.All-{BITCAST}: every BF16-valued
 * producer executes as float32 with the same lane count and source order.
 * LOAD/STORE/BITCAST/CAST have earlier specialized rules in this matcher. */
static PolyUOp *rule_bf16_float_producer(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  if (root->op == POLY_OP_BITCAST || !is_bf16(root->dtype)) return NULL;

  PolyUOp **new_src = NULL;
  if (root->n_src > 0) {
    new_src = malloc((size_t)root->n_src * sizeof(*new_src));
    if (!new_src) return NULL;
    for (int i = 0; i < root->n_src; i++) {
      /* Matches tinygrad's exact `s.dtype == ctx[0]` scalar check. Vector
       * children have already been promoted by the bottom-up traversal. */
      new_src[i] = poly_dtype_eq(root->src[i]->dtype, POLY_BFLOAT16)
                       ? poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, root->src[i], poly_arg_none())
                       : root->src[i];
    }
  }
  PolyUOp *out = float_decomp_clone(ctx, root, float32_like(root->dtype), new_src);
  free(new_src);
  return out;
}

/* Pinned tinygrad pm_float_decomp keeps a cast-to-BF16 value in float32 and
 * performs the lossy conversion only when materializing BF16 storage. */
static PolyUOp *rule_f32_to_bf16_cast(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  if (!is_bf16(root->dtype)) return NULL;
  /* Pinned pm_manual_bf16_cast only matches CAST(float32 -> bfloat16)
   * (renderer/cstyle.py:91-95). Matching a same-dtype/index CAST here changes
   * the BF16 storage reference to float32 before f2f_store can recognize it. */
  if (root->n_src < 1 || !poly_dtype_eq(poly_dtype_scalar(root->src[0]->dtype), POLY_FLOAT32))
    return NULL;
  return bf16_clamp_f32(ctx, root->src[0]);
}

/* Rule: CAST(x:bf16, non-f32) or CAST(x:non-f32, bf16) -> go through f32 */
static PolyUOp *rule_bf16_cast_via_f32(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  if (root->n_src < 1) return NULL;
  PolyDType src_dt = root->src[0]->dtype;
  PolyDType dst_dt = root->dtype;
  PolyDType src_scalar = poly_dtype_scalar(src_dt);
  PolyDType dst_scalar = poly_dtype_scalar(dst_dt);

  /* bf16 -> non-f32: go through f32 */
  if (is_bf16(src_dt) && !poly_dtype_eq(dst_scalar, POLY_FLOAT32) && !is_bf16(dst_dt)) {
    PolyUOp *f32 =
        poly_uop1(ctx, POLY_OP_CAST, float32_like(src_dt), root->src[0], poly_arg_none());
    return poly_uop1(ctx, POLY_OP_CAST, dst_dt, f32, poly_arg_none());
  }
  /* non-f32 -> bf16: go through f32 */
  if (is_bf16(dst_dt) && !poly_dtype_eq(src_scalar, POLY_FLOAT32) && !is_bf16(src_dt)) {
    PolyUOp *f32 =
        poly_uop1(ctx, POLY_OP_CAST, float32_like(dst_dt), root->src[0], poly_arg_none());
    return poly_uop1(ctx, POLY_OP_CAST, dst_dt, f32, poly_arg_none());
  }
  return NULL;
}

/* Pinned tinygrad pm_float_decomp f2f_load: storage remains 16-bit while the
 * loaded execution value becomes float32. */
static PolyUOp *rule_bf16_load(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  if (root->op != POLY_OP_LOAD || !is_bf16(root->dtype) || root->n_src < 1) return NULL;
  int lanes = root->dtype.count;
  if (lanes > 1) {
    /* Pinned f2f_load vectorizes converted scalar loads and reindexes the
     * INDEX below the address CAST (uop/decompositions.py:423-425). */
    PolyUOp *addr = root->src[0];
    if (root->n_src != 1 || !addr || addr->op != POLY_OP_CAST || addr->n_src != 1 ||
        !addr->src[0] || addr->src[0]->op != POLY_OP_INDEX)
      return NULL;
    PolyUOp **values = calloc((size_t)lanes, sizeof(*values));
    if (!values) return NULL;
    for (int lane = 0; lane < lanes; lane++) {
      PolyUOp *idx = codegen_reindex(ctx, addr->src[0], lane, 1);
      idx = bf16_storage_ref_as_u16(ctx, idx);
      if (!idx) {
        free(values);
        return NULL;
      }
      PolyUOp *src[1] = {idx};
      PolyUOp *raw = float_decomp_clone(ctx, root, POLY_UINT16, src);
      values[lane] = raw ? bf16_bits_to_f32(ctx, raw) : NULL;
      if (!values[lane]) {
        free(values);
        return NULL;
      }
    }
    PolyUOp *out = poly_uop(
        ctx, POLY_OP_STACK, poly_dtype_vec(POLY_FLOAT32, lanes), values, lanes, poly_arg_none()
    );
    free(values);
    return out;
  }
  PolyUOp *idx = bf16_storage_ref_as_u16(ctx, root->src[0]);
  if (!idx) return NULL;
  PolyUOp **src = malloc((size_t)root->n_src * sizeof(PolyUOp *));
  if (!src) return NULL;
  memcpy(src, root->src, (size_t)root->n_src * sizeof(PolyUOp *));
  src[0] = idx;
  PolyUOp *raw = float_decomp_clone(ctx, root, POLY_UINT16, src);
  free(src);
  return bf16_bits_to_f32(ctx, raw);
}

static PolyUOp *bf16_extract_lane(PolyCtx *ctx, PolyUOp *u, int lane) {
  if (!u || lane < 0) return NULL;
  if (u->op == POLY_OP_STACK && lane < u->n_src) return u->src[lane];
  if (u->dtype.count <= 1) return lane == 0 ? u : NULL;
  return poly_uop1(ctx, POLY_OP_GEP, poly_dtype_scalar(u->dtype), u, poly_arg_int((int64_t)lane));
}

/* Pinned tinygrad pm_float_decomp f2f_store: convert the float32 execution
 * value once, at the BF16 storage boundary. Vector stores are decomposed into
 * scalar lanes exactly like f2f_store (uop/decompositions.py:423-429). */
static PolyUOp *rule_bf16_store(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  if (root->op != POLY_OP_STORE || root->n_src != 2 ||
      (!is_bf16_ptr(root->src[0]->dtype) && !is_bf16(root->src[0]->dtype)))
    return NULL;
  /* Pinned pm_float_decomp unwraps the CAST inserted around an INDEX before
   * testing its storage tag and calling f2f_store
   * (uop/decompositions.py:551-554). Retagging that pointer CAST itself is not
   * a valid storage reference. */
  PolyUOp *storage_idx = (root->src[0]->op == POLY_OP_CAST && root->src[0]->n_src == 1)
                             ? root->src[0]->src[0]
                             : root->src[0];
  PolyUOp *idx = bf16_storage_ref_as_u16(ctx, storage_idx);
  if (!idx) return NULL;
  int lanes = root->src[1]->dtype.count;
  if (lanes > 1) {
    PolyUOp **stores = calloc((size_t)lanes, sizeof(*stores));
    if (!stores) return NULL;
    for (int lane = 0; lane < lanes; lane++) {
      PolyUOp *lane_idx = codegen_reindex(ctx, idx, lane, 1);
      PolyUOp *lane_value = bf16_extract_lane(ctx, root->src[1], lane);
      if (lane_value && is_bf16(lane_value->dtype))
        lane_value = poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, lane_value, poly_arg_none());
      if (!lane_idx || !lane_value || !poly_dtype_eq(lane_value->dtype, POLY_FLOAT32)) {
        free(stores);
        return NULL;
      }
      PolyUOp **src = malloc((size_t)root->n_src * sizeof(*src));
      if (!src) {
        free(stores);
        return NULL;
      }
      memcpy(src, root->src, (size_t)root->n_src * sizeof(*src));
      src[0] = lane_idx;
      src[1] = f32_to_bf16_bits(ctx, lane_value);
      stores[lane] = src[1] ? float_decomp_clone(ctx, root, root->dtype, src) : NULL;
      free(src);
      if (!stores[lane]) {
        free(stores);
        return NULL;
      }
    }
    PolyUOp *group = poly_uop(ctx, POLY_OP_GROUP, POLY_VOID, stores, lanes, poly_arg_none());
    free(stores);
    return group;
  }
  PolyUOp *value = poly_dtype_eq(root->src[1]->dtype, POLY_FLOAT32)
                       ? root->src[1]
                       : poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, root->src[1], poly_arg_none());
  PolyUOp **src = malloc((size_t)root->n_src * sizeof(PolyUOp *));
  if (!src) return NULL;
  memcpy(src, root->src, (size_t)root->n_src * sizeof(PolyUOp *));
  src[0] = idx;
  src[1] = f32_to_bf16_bits(ctx, value);
  PolyUOp *out = float_decomp_clone(ctx, root, root->dtype, src);
  free(src);
  return out;
}

/* Pinned tinygrad pm_float_decomp BITCAST-to/from rules after child values
 * have been emulated as float32. */
static PolyUOp *rule_bf16_bitcast(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  if (root->op != POLY_OP_BITCAST || root->n_src < 1) return NULL;
  PolyDType src = poly_dtype_scalar(root->src[0]->dtype);
  PolyDType dst = poly_dtype_scalar(root->dtype);
  /* Pinned pm_float_decomp rewrites BITCAST(LOAD(BF16), dst16) by retagging
   * the LOAD to uint16 before any numeric BF16 conversion
   * (uop/decompositions.py:537-539). */
  PolyUOp *load = root->src[0];
  if (load->op == POLY_OP_LOAD && poly_dtype_eq(load->dtype, POLY_BFLOAT16) && load->n_src >= 1 &&
      root->dtype.count == 1 && dst.bitsize == POLY_BFLOAT16.bitsize) {
    PolyUOp *idx = bf16_storage_ref_as_u16(ctx, load->src[0]);
    PolyUOp **load_src = malloc((size_t)load->n_src * sizeof(*load_src));
    if (!idx || !load_src) {
      free(load_src);
      return NULL;
    }
    memcpy(load_src, load->src, (size_t)load->n_src * sizeof(*load_src));
    load_src[0] = idx;
    PolyUOp *raw_load = float_decomp_clone(ctx, load, POLY_UINT16, load_src);
    free(load_src);
    if (!raw_load) return NULL;
    return poly_dtype_eq(root->dtype, POLY_UINT16)
               ? raw_load
               : poly_uop1(ctx, POLY_OP_BITCAST, root->dtype, raw_load, root->arg);
  }
  /* Pinned pm_float_decomp "bitcast to": every same-width 16-bit source is
   * first reinterpreted as BF16's uint16 storage representation
   * (uop/decompositions.py:544-545). */
  if (poly_dtype_eq(root->dtype, POLY_BFLOAT16) && root->src[0]->dtype.count == 1 &&
      src.bitsize == POLY_BFLOAT16.bitsize) {
    PolyDType raw_dtype = POLY_UINT16;
    PolyUOp *raw = poly_dtype_eq(root->src[0]->dtype, raw_dtype)
                       ? root->src[0]
                       : poly_uop1(ctx, POLY_OP_BITCAST, raw_dtype, root->src[0], poly_arg_none());
    return bf16_bits_to_f32(ctx, raw);
  }
  /* Pinned pm_float_decomp "bitcast from": after the BF16 child is promoted
   * to float32, restore its raw uint16 bits and retain BITCAST to every
   * same-width destination (uop/decompositions.py:541-542). */
  if (poly_dtype_eq(root->src[0]->dtype, POLY_FLOAT32) && root->dtype.count == 1 &&
      dst.bitsize == POLY_BFLOAT16.bitsize) {
    PolyUOp *raw = f32_to_bf16_bits(ctx, root->src[0]);
    return raw ? poly_uop1(ctx, POLY_OP_BITCAST, root->dtype, raw, root->arg) : NULL;
  }
  return NULL;
}

static _Thread_local PolyPatternMatcher *g_pm_bf16_non_native = NULL;

PolyPatternMatcher *poly_pm_bf16_non_native(void) {
  if (g_pm_bf16_non_native) return g_pm_bf16_non_native;

  /* Pinned GroupOp.All-{BITCAST}; include Polygrad's reviewed superset ops so
   * unsupported BF16 producers cannot escape merely because the op is new. */
  PolyOpSet all_no_bitcast = {{0, 0}};
  for (int op = 1; op < POLY_OP_COUNT; op++)
    if (op != POLY_OP_BITCAST) all_no_bitcast = poly_opset_add(all_no_bitcast, (PolyOps)op);

  PolyOpSet cast_set = poly_opset_add((PolyOpSet){{0, 0}}, POLY_OP_CAST);
  PolyOpSet bitcast_set = poly_opset_add((PolyOpSet){{0, 0}}, POLY_OP_BITCAST);
  PolyOpSet load_set = poly_opset_add((PolyOpSet){{0, 0}}, POLY_OP_LOAD);
  PolyOpSet store_set = poly_opset_add((PolyOpSet){{0, 0}}, POLY_OP_STORE);

  PolyRule rules[] = {
      /* 0. BF16 storage boundaries become uint16 loads/stores. */
      {poly_pat_ops(load_set, NULL, 0, NULL), rule_bf16_load},
      {poly_pat_ops(store_set, NULL, 0, NULL), rule_bf16_store},
      {poly_pat_ops(bitcast_set, NULL, 0, NULL), rule_bf16_bitcast},
      /* 3. CAST f32->bf16: retained in f32 until STORE. Pinned
       * pm_float_decomp has no general bf16->f32 CAST fallback; producer
       * rules legalize LOAD/ALU/CONST values exactly once
       * (uop/decompositions.py:533-562). */
      {poly_pat_ops(cast_set, NULL, 0, NULL), rule_f32_to_bf16_cast},
      /* 4. CAST bf16<->other: go through f32. */
      {poly_pat_ops(cast_set, NULL, 0, NULL), rule_bf16_cast_via_f32},
      /* 5. Every other BF16-valued producer executes in float32, matching
       * pm_float_decomp's GroupOp.All-{BITCAST} rule. */
      {poly_pat_ops(all_no_bitcast, NULL, 0, NULL), rule_bf16_float_producer},
  };

  g_pm_bf16_non_native =
      poly_pm_thread_cache(poly_pm_new(rules, (int)(sizeof(rules) / sizeof(rules[0]))));
  return g_pm_bf16_non_native;
}

/* Pinned tinygrad pm_float_decomp for unsupported IEEE float16. Unlike BF16,
 * Python 3.11 executes these values as float32 and materializes uint16 storage
 * only at LOAD/STORE/BITCAST boundaries. Denormals flush to signed zero
 * exactly as decompositions.py:f2f specifies. */
static bool is_f16_ptr(PolyDType dt) {
  if (!dt.is_ptr) return false;
  PolyDType s = poly_dtype_scalar(dt);
  s.is_ptr = false;
  s.addrspace = 0;
  s.vcount = 0;
  s.ptr_size = 0;
  return s.priority == POLY_FLOAT16.priority && s.bitsize == 16;
}

static PolyUOp *f16_storage_ref_as_u16(PolyCtx *ctx, PolyUOp *u) {
  if (!u || (!is_f16_ptr(u->dtype) && !is_f16(u->dtype))) return u;
  PolyUOp **src = NULL;
  if (u->n_src > 0) {
    src = malloc((size_t)u->n_src * sizeof(*src));
    if (!src) return NULL;
    for (int i = 0; i < u->n_src; i++)
      src[i] = (is_f16_ptr(u->src[i]->dtype) || is_f16(u->src[i]->dtype))
                   ? f16_storage_ref_as_u16(ctx, u->src[i])
                   : u->src[i];
  }
  PolyDType dtype;
  if (u->dtype.is_ptr) {
    PolyDType storage =
        u->dtype.count > 1 ? poly_dtype_vec(POLY_UINT16, u->dtype.count) : POLY_UINT16;
    dtype = poly_dtype_ptr(storage, u->dtype.ptr_size, u->dtype.addrspace);
    dtype.vcount = u->dtype.vcount;
  } else {
    dtype = u->dtype.count > 1 ? poly_dtype_vec(POLY_UINT16, u->dtype.count) : POLY_UINT16;
  }
  PolyUOp *out = float_decomp_clone(ctx, u, dtype, src);
  free(src);
  return out;
}

static PolyUOp *f16_bits_to_f32(PolyCtx *ctx, PolyUOp *raw_u16) {
  PolyDType u32dt = dtype_count_like(POLY_UINT32, raw_u16->dtype);
  PolyDType booldt = dtype_count_like(POLY_BOOL, raw_u16->dtype);
  PolyDType f32dt = float32_like(raw_u16->dtype);
  PolyUOp *u32 = poly_uop1(ctx, POLY_OP_CAST, u32dt, raw_u16, poly_arg_none());
  PolyUOp *c0 = poly_uop0(ctx, POLY_OP_CONST, u32dt, poly_arg_int(0));
  PolyUOp *c10 = poly_uop0(ctx, POLY_OP_CONST, u32dt, poly_arg_int(10));
  PolyUOp *c13 = poly_uop0(ctx, POLY_OP_CONST, u32dt, poly_arg_int(13));
  PolyUOp *c16 = poly_uop0(ctx, POLY_OP_CONST, u32dt, poly_arg_int(16));
  PolyUOp *c31 = poly_uop0(ctx, POLY_OP_CONST, u32dt, poly_arg_int(31));
  PolyUOp *sign_mask = poly_uop0(ctx, POLY_OP_CONST, u32dt, poly_arg_int(0x8000));
  PolyUOp *value_mask = poly_uop0(ctx, POLY_OP_CONST, u32dt, poly_arg_int(0x7fff));
  PolyUOp *bias = poly_uop0(ctx, POLY_OP_CONST, u32dt, poly_arg_int(0x38000000));
  PolyUOp *inf_bits = poly_uop0(ctx, POLY_OP_CONST, u32dt, poly_arg_int(0x7f800000));

  PolyUOp *sign = poly_uop2(
      ctx, POLY_OP_SHL, u32dt,
      poly_uop2(ctx, POLY_OP_AND, u32dt, u32, sign_mask, poly_arg_none()), c16,
      poly_arg_none()
  );
  PolyUOp *nosign = poly_uop2(ctx, POLY_OP_AND, u32dt, u32, value_mask, poly_arg_none());
  PolyUOp *exp = poly_uop2(ctx, POLY_OP_SHR, u32dt, nosign, c10, poly_arg_none());
  PolyUOp *norm = poly_uop2(
      ctx, POLY_OP_ADD, u32dt,
      poly_uop2(ctx, POLY_OP_SHL, u32dt, nosign, c13, poly_arg_none()), bias,
      poly_arg_none()
  );
  PolyUOp *nan_bits = poly_uop2(
      ctx, POLY_OP_OR, u32dt,
      poly_uop2(ctx, POLY_OP_SHL, u32dt, nosign, c13, poly_arg_none()), inf_bits,
      poly_arg_none()
  );
  PolyUOp *exp_zero = poly_uop2(ctx, POLY_OP_CMPEQ, booldt, exp, c0, poly_arg_none());
  PolyUOp *exp_nan = poly_uop2(ctx, POLY_OP_CMPEQ, booldt, exp, c31, poly_arg_none());
  PolyUOp *magnitude = poly_uop3(
      ctx, POLY_OP_WHERE, u32dt, exp_zero, c0,
      poly_uop3(ctx, POLY_OP_WHERE, u32dt, exp_nan, nan_bits, norm, poly_arg_none()),
      poly_arg_none()
  );
  return poly_uop1(
      ctx, POLY_OP_BITCAST, f32dt,
      poly_uop2(ctx, POLY_OP_OR, u32dt, sign, magnitude, poly_arg_none()), poly_arg_none()
  );
}

static PolyUOp *f16_clamp_f32(PolyCtx *ctx, PolyUOp *x) {
  PolyDType f32dt = float32_like(x->dtype);
  PolyDType booldt = dtype_count_like(POLY_BOOL, x->dtype);
  PolyUOp *max = poly_uop0(ctx, POLY_OP_CONST, f32dt, poly_arg_float(65504.0));
  PolyUOp *neg_max = poly_uop1(ctx, POLY_OP_NEG, f32dt, max, poly_arg_none());
  PolyUOp *pos_inf = poly_uop0(ctx, POLY_OP_CONST, f32dt, poly_arg_float(INFINITY));
  PolyUOp *neg_inf = poly_uop0(ctx, POLY_OP_CONST, f32dt, poly_arg_float(-INFINITY));
  PolyUOp *is_nan = poly_uop2(ctx, POLY_OP_CMPNE, booldt, x, x, poly_arg_none());
  PolyUOp *below = poly_uop2(ctx, POLY_OP_CMPLT, booldt, x, neg_max, poly_arg_none());
  PolyUOp *above = poly_uop2(ctx, POLY_OP_CMPLT, booldt, max, x, poly_arg_none());
  PolyUOp *bounded = poly_uop3(
      ctx, POLY_OP_WHERE, f32dt, below, neg_inf,
      poly_uop3(ctx, POLY_OP_WHERE, f32dt, above, pos_inf, x, poly_arg_none()), poly_arg_none()
  );
  return poly_uop3(ctx, POLY_OP_WHERE, f32dt, is_nan, x, bounded, poly_arg_none());
}

static PolyUOp *f32_to_f16_bits(PolyCtx *ctx, PolyUOp *x) {
  x = f16_clamp_f32(ctx, x);
  PolyDType u32dt = dtype_count_like(POLY_UINT32, x->dtype);
  PolyDType u16dt = dtype_count_like(POLY_UINT16, x->dtype);
  PolyDType booldt = dtype_count_like(POLY_BOOL, x->dtype);
  PolyUOp *v = poly_uop1(ctx, POLY_OP_BITCAST, u32dt, x, poly_arg_none());
  PolyUOp *c0 = poly_uop0(ctx, POLY_OP_CONST, u32dt, poly_arg_int(0));
  PolyUOp *c1 = poly_uop0(ctx, POLY_OP_CONST, u32dt, poly_arg_int(1));
  PolyUOp *c12 = poly_uop0(ctx, POLY_OP_CONST, u32dt, poly_arg_int(12));
  PolyUOp *c13 = poly_uop0(ctx, POLY_OP_CONST, u32dt, poly_arg_int(13));
  PolyUOp *c16 = poly_uop0(ctx, POLY_OP_CONST, u32dt, poly_arg_int(16));
  PolyUOp *c23 = poly_uop0(ctx, POLY_OP_CONST, u32dt, poly_arg_int(23));
  PolyUOp *c113 = poly_uop0(ctx, POLY_OP_CONST, u32dt, poly_arg_int(113));
  PolyUOp *c255 = poly_uop0(ctx, POLY_OP_CONST, u32dt, poly_arg_int(255));
  PolyUOp *c3ff = poly_uop0(ctx, POLY_OP_CONST, u32dt, poly_arg_int(0x3ff));
  PolyUOp *cfff = poly_uop0(ctx, POLY_OP_CONST, u32dt, poly_arg_int(0xfff));
  PolyUOp *c7fffffff = poly_uop0(ctx, POLY_OP_CONST, u32dt, poly_arg_int(0x7fffffff));
  PolyUOp *c8000 = poly_uop0(ctx, POLY_OP_CONST, u32dt, poly_arg_int(0x8000));
  PolyUOp *c7c00 = poly_uop0(ctx, POLY_OP_CONST, u32dt, poly_arg_int(0x7c00));
  PolyUOp *bias = poly_uop0(ctx, POLY_OP_CONST, u32dt, poly_arg_int(0x1c000));

  PolyUOp *sign = poly_uop2(
      ctx, POLY_OP_AND, u32dt,
      poly_uop2(ctx, POLY_OP_SHR, u32dt, v, c16, poly_arg_none()), c8000,
      poly_arg_none()
  );
  PolyUOp *nosign = poly_uop2(ctx, POLY_OP_AND, u32dt, v, c7fffffff, poly_arg_none());
  PolyUOp *q = poly_uop2(ctx, POLY_OP_SHR, u32dt, nosign, c13, poly_arg_none());
  PolyUOp *round_bit = poly_uop2(
      ctx, POLY_OP_AND, u32dt,
      poly_uop2(ctx, POLY_OP_SHR, u32dt, nosign, c12, poly_arg_none()), c1,
      poly_arg_none()
  );
  PolyUOp *tail = poly_uop2(
      ctx, POLY_OP_CMPNE, booldt,
      poly_uop2(ctx, POLY_OP_AND, u32dt, nosign, cfff, poly_arg_none()), c0,
      poly_arg_none()
  );
  PolyUOp *q_odd = poly_uop2(ctx, POLY_OP_AND, u32dt, q, c1, poly_arg_none());
  PolyUOp *sticky = poly_uop2(
      ctx, POLY_OP_OR, u32dt, poly_uop1(ctx, POLY_OP_CAST, u32dt, tail, poly_arg_none()),
      q_odd, poly_arg_none()
  );
  PolyUOp *rounded = poly_uop2(
      ctx, POLY_OP_ADD, u32dt, q,
      poly_uop2(ctx, POLY_OP_AND, u32dt, round_bit, sticky, poly_arg_none()),
      poly_arg_none()
  );
  PolyUOp *norm = poly_uop1(
      ctx, POLY_OP_CAST, u16dt,
      poly_uop2(ctx, POLY_OP_SUB, u32dt, rounded, bias, poly_arg_none()), poly_arg_none()
  );
  PolyUOp *exp = poly_uop2(
      ctx, POLY_OP_AND, u32dt,
      poly_uop2(ctx, POLY_OP_SHR, u32dt, v, c23, poly_arg_none()), c255,
      poly_arg_none()
  );
  PolyUOp *underflow = poly_uop2(ctx, POLY_OP_CMPLT, booldt, exp, c113, poly_arg_none());
  PolyUOp *is_nan = poly_uop2(ctx, POLY_OP_CMPEQ, booldt, exp, c255, poly_arg_none());
  PolyUOp *nan_mantissa = poly_uop2(
      ctx, POLY_OP_AND, u32dt,
      poly_uop2(ctx, POLY_OP_SHR, u32dt, nosign, c13, poly_arg_none()), c3ff,
      poly_arg_none()
  );
  PolyUOp *nan_bits = poly_uop1(
      ctx, POLY_OP_CAST, u16dt,
      poly_uop2(
          ctx, POLY_OP_OR, u32dt,
          poly_uop2(ctx, POLY_OP_OR, u32dt, sign, nan_mantissa, poly_arg_none()), c7c00,
          poly_arg_none()
      ),
      poly_arg_none()
  );
  PolyUOp *sign_u16 = poly_uop1(ctx, POLY_OP_CAST, u16dt, sign, poly_arg_none());
  PolyUOp *zero_u16 = poly_uop0(ctx, POLY_OP_CONST, u16dt, poly_arg_int(0));
  PolyUOp *finite = poly_uop2(
      ctx, POLY_OP_OR, u16dt, sign_u16,
      poly_uop3(ctx, POLY_OP_WHERE, u16dt, underflow, zero_u16, norm, poly_arg_none()),
      poly_arg_none()
  );
  return poly_uop3(ctx, POLY_OP_WHERE, u16dt, is_nan, nan_bits, finite, poly_arg_none());
}

static PolyUOp *rule_f16_float_producer(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  if (root->op == POLY_OP_BITCAST || !is_f16(root->dtype)) return NULL;
  PolyUOp **new_src = NULL;
  if (root->n_src > 0) {
    new_src = malloc((size_t)root->n_src * sizeof(*new_src));
    if (!new_src) return NULL;
    for (int i = 0; i < root->n_src; i++)
      new_src[i] = poly_dtype_eq(root->src[i]->dtype, POLY_FLOAT16)
                       ? poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, root->src[i], poly_arg_none())
                       : root->src[i];
  }
  PolyUOp *out = float_decomp_clone(ctx, root, float32_like(root->dtype), new_src);
  free(new_src);
  return out;
}

static PolyUOp *rule_f32_to_f16_cast(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  if (!is_f16(root->dtype) || root->n_src < 1 ||
      !poly_dtype_eq(poly_dtype_scalar(root->src[0]->dtype), POLY_FLOAT32))
    return NULL;
  return f16_clamp_f32(ctx, root->src[0]);
}

static PolyUOp *rule_f16_cast_via_f32(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  if (root->n_src < 1) return NULL;
  PolyDType src_dt = root->src[0]->dtype;
  PolyDType dst_dt = root->dtype;
  PolyDType src_scalar = poly_dtype_scalar(src_dt);
  PolyDType dst_scalar = poly_dtype_scalar(dst_dt);
  if (is_f16(src_dt) && !poly_dtype_eq(dst_scalar, POLY_FLOAT32) && !is_f16(dst_dt)) {
    PolyUOp *f32 =
        poly_uop1(ctx, POLY_OP_CAST, float32_like(src_dt), root->src[0], poly_arg_none());
    return poly_uop1(ctx, POLY_OP_CAST, dst_dt, f32, poly_arg_none());
  }
  if (is_f16(dst_dt) && !poly_dtype_eq(src_scalar, POLY_FLOAT32) && !is_f16(src_dt)) {
    PolyUOp *f32 =
        poly_uop1(ctx, POLY_OP_CAST, float32_like(dst_dt), root->src[0], poly_arg_none());
    return poly_uop1(ctx, POLY_OP_CAST, dst_dt, f32, poly_arg_none());
  }
  return NULL;
}

static PolyUOp *rule_f16_load(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  if (root->op != POLY_OP_LOAD || !is_f16(root->dtype) || root->n_src < 1) return NULL;
  int lanes = root->dtype.count;
  if (lanes > 1) {
    PolyUOp *addr = root->src[0];
    if (root->n_src != 1 || !addr || addr->op != POLY_OP_CAST || addr->n_src != 1 ||
        !addr->src[0] || addr->src[0]->op != POLY_OP_INDEX)
      return NULL;
    PolyUOp **values = calloc((size_t)lanes, sizeof(*values));
    if (!values) return NULL;
    for (int lane = 0; lane < lanes; lane++) {
      PolyUOp *idx = codegen_reindex(ctx, addr->src[0], lane, 1);
      idx = f16_storage_ref_as_u16(ctx, idx);
      if (!idx) {
        free(values);
        return NULL;
      }
      PolyUOp *src[1] = {idx};
      PolyUOp *raw = float_decomp_clone(ctx, root, POLY_UINT16, src);
      values[lane] = raw ? f16_bits_to_f32(ctx, raw) : NULL;
      if (!values[lane]) {
        free(values);
        return NULL;
      }
    }
    PolyUOp *out = poly_uop(
        ctx, POLY_OP_STACK, poly_dtype_vec(POLY_FLOAT32, lanes), values, lanes, poly_arg_none()
    );
    free(values);
    return out;
  }
  PolyUOp *idx = f16_storage_ref_as_u16(ctx, root->src[0]);
  if (!idx) return NULL;
  PolyUOp **src = malloc((size_t)root->n_src * sizeof(*src));
  if (!src) return NULL;
  memcpy(src, root->src, (size_t)root->n_src * sizeof(*src));
  src[0] = idx;
  PolyUOp *raw = float_decomp_clone(ctx, root, POLY_UINT16, src);
  free(src);
  return raw ? f16_bits_to_f32(ctx, raw) : NULL;
}

static PolyUOp *f16_extract_lane(PolyCtx *ctx, PolyUOp *u, int lane) {
  if (!u || lane < 0) return NULL;
  if (u->op == POLY_OP_STACK && lane < u->n_src) return u->src[lane];
  if (u->dtype.count <= 1) return lane == 0 ? u : NULL;
  return poly_uop1(ctx, POLY_OP_GEP, poly_dtype_scalar(u->dtype), u, poly_arg_int(lane));
}

static PolyUOp *rule_f16_store(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  if (root->op != POLY_OP_STORE || root->n_src != 2 ||
      (!is_f16_ptr(root->src[0]->dtype) && !is_f16(root->src[0]->dtype)))
    return NULL;
  PolyUOp *storage_idx = (root->src[0]->op == POLY_OP_CAST && root->src[0]->n_src == 1)
                             ? root->src[0]->src[0]
                             : root->src[0];
  PolyUOp *idx = f16_storage_ref_as_u16(ctx, storage_idx);
  if (!idx) return NULL;
  int lanes = root->src[1]->dtype.count;
  if (lanes > 1) {
    PolyUOp **stores = calloc((size_t)lanes, sizeof(*stores));
    if (!stores) return NULL;
    for (int lane = 0; lane < lanes; lane++) {
      PolyUOp *lane_idx = codegen_reindex(ctx, idx, lane, 1);
      PolyUOp *lane_value = f16_extract_lane(ctx, root->src[1], lane);
      if (lane_value && is_f16(lane_value->dtype))
        lane_value = poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, lane_value, poly_arg_none());
      if (!lane_idx || !lane_value || !poly_dtype_eq(lane_value->dtype, POLY_FLOAT32)) {
        free(stores);
        return NULL;
      }
      PolyUOp **src = malloc((size_t)root->n_src * sizeof(*src));
      if (!src) {
        free(stores);
        return NULL;
      }
      memcpy(src, root->src, (size_t)root->n_src * sizeof(*src));
      src[0] = lane_idx;
      src[1] = f32_to_f16_bits(ctx, lane_value);
      stores[lane] = src[1] ? float_decomp_clone(ctx, root, root->dtype, src) : NULL;
      free(src);
      if (!stores[lane]) {
        free(stores);
        return NULL;
      }
    }
    PolyUOp *group = poly_uop(ctx, POLY_OP_GROUP, POLY_VOID, stores, lanes, poly_arg_none());
    free(stores);
    return group;
  }
  PolyUOp *value = poly_dtype_eq(root->src[1]->dtype, POLY_FLOAT32)
                       ? root->src[1]
                       : poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, root->src[1], poly_arg_none());
  PolyUOp **src = malloc((size_t)root->n_src * sizeof(*src));
  if (!src) return NULL;
  memcpy(src, root->src, (size_t)root->n_src * sizeof(*src));
  src[0] = idx;
  src[1] = f32_to_f16_bits(ctx, value);
  PolyUOp *out = float_decomp_clone(ctx, root, root->dtype, src);
  free(src);
  return out;
}

static PolyUOp *rule_f16_bitcast(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  if (root->op != POLY_OP_BITCAST || root->n_src < 1) return NULL;
  PolyDType src = poly_dtype_scalar(root->src[0]->dtype);
  PolyDType dst = poly_dtype_scalar(root->dtype);
  PolyUOp *load = root->src[0];
  if (load->op == POLY_OP_LOAD && poly_dtype_eq(load->dtype, POLY_FLOAT16) &&
      load->n_src >= 1 && root->dtype.count == 1 && dst.bitsize == POLY_FLOAT16.bitsize) {
    PolyUOp *idx = f16_storage_ref_as_u16(ctx, load->src[0]);
    PolyUOp **load_src = malloc((size_t)load->n_src * sizeof(*load_src));
    if (!idx || !load_src) {
      free(load_src);
      return NULL;
    }
    memcpy(load_src, load->src, (size_t)load->n_src * sizeof(*load_src));
    load_src[0] = idx;
    PolyUOp *raw_load = float_decomp_clone(ctx, load, POLY_UINT16, load_src);
    free(load_src);
    if (!raw_load) return NULL;
    return poly_dtype_eq(root->dtype, POLY_UINT16)
               ? raw_load
               : poly_uop1(ctx, POLY_OP_BITCAST, root->dtype, raw_load, root->arg);
  }
  if (poly_dtype_eq(root->dtype, POLY_FLOAT16) && root->src[0]->dtype.count == 1 &&
      src.bitsize == POLY_FLOAT16.bitsize) {
    PolyUOp *raw = poly_dtype_eq(root->src[0]->dtype, POLY_UINT16)
                       ? root->src[0]
                       : poly_uop1(ctx, POLY_OP_BITCAST, POLY_UINT16, root->src[0], poly_arg_none());
    return f16_bits_to_f32(ctx, raw);
  }
  if (poly_dtype_eq(root->src[0]->dtype, POLY_FLOAT32) && root->dtype.count == 1 &&
      dst.bitsize == POLY_FLOAT16.bitsize) {
    PolyUOp *raw = f32_to_f16_bits(ctx, root->src[0]);
    return raw ? poly_uop1(ctx, POLY_OP_BITCAST, root->dtype, raw, root->arg) : NULL;
  }
  return NULL;
}

static _Thread_local PolyPatternMatcher *g_pm_f16_non_native = NULL;

PolyPatternMatcher *poly_pm_f16_non_native(void) {
  if (g_pm_f16_non_native) return g_pm_f16_non_native;
  PolyOpSet all_no_bitcast = {{0, 0}};
  for (int op = 1; op < POLY_OP_COUNT; op++)
    if (op != POLY_OP_BITCAST) all_no_bitcast = poly_opset_add(all_no_bitcast, (PolyOps)op);
  PolyOpSet cast_set = poly_opset_add((PolyOpSet){{0, 0}}, POLY_OP_CAST);
  PolyOpSet bitcast_set = poly_opset_add((PolyOpSet){{0, 0}}, POLY_OP_BITCAST);
  PolyOpSet load_set = poly_opset_add((PolyOpSet){{0, 0}}, POLY_OP_LOAD);
  PolyOpSet store_set = poly_opset_add((PolyOpSet){{0, 0}}, POLY_OP_STORE);
  PolyRule rules[] = {
      {poly_pat_ops(load_set, NULL, 0, NULL), rule_f16_load},
      {poly_pat_ops(store_set, NULL, 0, NULL), rule_f16_store},
      {poly_pat_ops(bitcast_set, NULL, 0, NULL), rule_f16_bitcast},
      {poly_pat_ops(cast_set, NULL, 0, NULL), rule_f32_to_f16_cast},
      {poly_pat_ops(cast_set, NULL, 0, NULL), rule_f16_cast_via_f32},
      {poly_pat_ops(all_no_bitcast, NULL, 0, NULL), rule_f16_float_producer},
  };
  g_pm_f16_non_native =
      poly_pm_thread_cache(poly_pm_new(rules, (int)(sizeof(rules) / sizeof(rules[0]))));
  return g_pm_f16_non_native;
}

/* Pinned tinygrad renderer/cstyle.py:create_non_native_float_pats for a
 * renderer which supports BF16 storage/WMMA but executes ordinary BF16 ALU in
 * float32. This is a renderer-final matcher, not pm_dtype_decomps: it must not
 * rewrite BF16 CONTRACT/WMMA fragments or storage definitions. */
static PolyUOp *rule_bf16_renderer_alu(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  if (!root || !poly_opset_has(POLY_GROUP_ALU, root->op)) return NULL;

  if (root->op == POLY_OP_WHERE && root->n_src == 3 && is_bf16(root->src[1]->dtype) &&
      is_bf16(root->src[2]->dtype)) {
    PolyDType f32 = float32_like(root->dtype);
    PolyUOp *src[] = {
        root->src[0],
        poly_uop1(ctx, POLY_OP_CAST, f32, root->src[1], poly_arg_none()),
        poly_uop1(ctx, POLY_OP_CAST, f32, root->src[2], poly_arg_none()),
    };
    PolyUOp *alu = float_decomp_clone(ctx, root, f32, src);
    return alu ? poly_uop1(ctx, POLY_OP_CAST, root->dtype, alu, poly_arg_none()) : NULL;
  }

  if (is_bf16(root->dtype)) {
    PolyUOp **src = calloc((size_t)root->n_src, sizeof(*src));
    if (!src) return NULL;
    for (int i = 0; i < root->n_src; i++) {
      PolyDType f32 = float32_like(root->src[i]->dtype);
      src[i] = poly_uop1(ctx, POLY_OP_CAST, f32, root->src[i], poly_arg_none());
    }
    PolyUOp *alu = float_decomp_clone(ctx, root, float32_like(root->dtype), src);
    free(src);
    return alu ? poly_uop1(ctx, POLY_OP_CAST, root->dtype, alu, poly_arg_none()) : NULL;
  }

  if (poly_dtype_is_bool(root->dtype) && root->n_src == 2 && is_bf16(root->src[0]->dtype) &&
      is_bf16(root->src[1]->dtype)) {
    PolyUOp *src[] = {
        poly_uop1(
            ctx, POLY_OP_CAST, float32_like(root->src[0]->dtype), root->src[0], poly_arg_none()
        ),
        poly_uop1(
            ctx, POLY_OP_CAST, float32_like(root->src[1]->dtype), root->src[1], poly_arg_none()
        ),
    };
    return float_decomp_clone(ctx, root, root->dtype, src);
  }
  return NULL;
}

/* Pinned renderer/cstyle.py:cast_float_to_bf16. This is deliberately
 * separate from f32_to_bf16_bits: renderer-final native BF16 casting retains
 * a BF16 BITCAST, whereas unsupported-dtype decomposition eliminates BF16. */
static PolyUOp *renderer_f32_to_bf16(PolyCtx *ctx, PolyUOp *x, PolyDType bf16_dtype) {
  PolyDType u32dt = dtype_count_like(POLY_UINT32, x->dtype);
  PolyDType u16dt = dtype_count_like(POLY_UINT16, x->dtype);
  PolyDType booldt = dtype_count_like(POLY_BOOL, x->dtype);
  PolyUOp *bits = poly_uop1(ctx, POLY_OP_BITCAST, u32dt, x, poly_arg_none());
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, u32dt, poly_arg_int(0));
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, u32dt, poly_arg_int(1));
  PolyUOp *c16 = poly_uop0(ctx, POLY_OP_CONST, u32dt, poly_arg_int(16));
  PolyUOp *c7fff = poly_uop0(ctx, POLY_OP_CONST, u32dt, poly_arg_int(0x7fff));
  PolyUOp *cffff = poly_uop0(ctx, POLY_OP_CONST, u32dt, poly_arg_int(0xffff));
  PolyUOp *c10000 = poly_uop0(ctx, POLY_OP_CONST, u32dt, poly_arg_int(0x10000));
  PolyUOp *c7f800000 = poly_uop0(ctx, POLY_OP_CONST, u32dt, poly_arg_int(0x7f800000));
  PolyUOp *neg_bits = poly_uop1(ctx, POLY_OP_NEG, u32dt, bits, poly_arg_none());
  PolyUOp *is_finite = poly_uop2(
      ctx, POLY_OP_CMPNE, booldt,
      poly_uop2(ctx, POLY_OP_AND, u32dt, neg_bits, c7f800000, poly_arg_none()), zero,
      poly_arg_none()
  );
  PolyUOp *rounded = poly_uop2(
      ctx, POLY_OP_ADD, u32dt,
      poly_uop2(
          ctx, POLY_OP_ADD, u32dt, bits,
          poly_uop2(
              ctx, POLY_OP_AND, u32dt,
              poly_uop2(ctx, POLY_OP_SHR, u32dt, bits, c16, poly_arg_none()), one, poly_arg_none()
          ),
          poly_arg_none()
      ),
      c7fff, poly_arg_none()
  );
  PolyUOp *low_nonzero = poly_uop2(
      ctx, POLY_OP_CMPNE, booldt, poly_uop2(ctx, POLY_OP_AND, u32dt, bits, cffff, poly_arg_none()),
      zero, poly_arg_none()
  );
  PolyUOp *nan_adjusted = poly_uop3(
      ctx, POLY_OP_WHERE, u32dt, low_nonzero,
      poly_uop2(ctx, POLY_OP_OR, u32dt, bits, c10000, poly_arg_none()), bits, poly_arg_none()
  );
  PolyUOp *selected =
      poly_uop3(ctx, POLY_OP_WHERE, u32dt, is_finite, rounded, nan_adjusted, poly_arg_none());
  PolyUOp *raw = poly_uop1(
      ctx, POLY_OP_CAST, u16dt, poly_uop2(ctx, POLY_OP_SHR, u32dt, selected, c16, poly_arg_none()),
      poly_arg_none()
  );
  return poly_uop1(ctx, POLY_OP_BITCAST, bf16_dtype, raw, poly_arg_none());
}

/* Pinned renderer/cstyle.py:pm_manual_bf16_cast. */
static PolyUOp *rule_bf16_renderer_manual_cast(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  if (!root || root->op != POLY_OP_CAST || root->n_src != 1) return NULL;
  PolyDType src = poly_dtype_scalar(root->src[0]->dtype);
  PolyDType dst = poly_dtype_scalar(root->dtype);
  if (poly_dtype_eq(src, POLY_BFLOAT16) && poly_dtype_eq(dst, POLY_FLOAT32)) {
    PolyDType u16dt = dtype_count_like(POLY_UINT16, root->src[0]->dtype);
    PolyDType u32dt = dtype_count_like(POLY_UINT32, root->src[0]->dtype);
    PolyDType f32dt = float32_like(root->src[0]->dtype);
    PolyUOp *raw = poly_uop1(ctx, POLY_OP_BITCAST, u16dt, root->src[0], poly_arg_none());
    PolyUOp *wide = poly_uop1(ctx, POLY_OP_CAST, u32dt, raw, poly_arg_none());
    PolyUOp *shift = poly_uop0(ctx, POLY_OP_CONST, u32dt, poly_arg_int(16));
    return poly_uop1(
        ctx, POLY_OP_BITCAST, f32dt,
        poly_uop2(ctx, POLY_OP_SHL, u32dt, wide, shift, poly_arg_none()), poly_arg_none()
    );
  }
  if (poly_dtype_eq(src, POLY_FLOAT32) && poly_dtype_eq(dst, POLY_BFLOAT16))
    return renderer_f32_to_bf16(ctx, root->src[0], root->dtype);
  return NULL;
}

/* Pinned HIPRenderer.extra_matcher converts a scalar BF16 constant from a
 * float32 constant through the same manual final cast
 * (renderer/cstyle.py:519-520). Rendering the original BF16 CONST directly
 * would assign a numeric float literal to HIP's raw uint16 storage. */
static PolyUOp *rule_bf16_renderer_const(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  if (!root || root->op != POLY_OP_CONST || root->dtype.count != 1 ||
      !poly_dtype_eq(root->dtype, POLY_BFLOAT16))
    return NULL;
  PolyUOp *f32 = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, root->arg);
  return renderer_f32_to_bf16(ctx, f32, POLY_BFLOAT16);
}

static _Thread_local PolyPatternMatcher *g_pm_bf16_renderer_extra = NULL;

PolyPatternMatcher *poly_pm_bf16_renderer_extra(void) {
  if (g_pm_bf16_renderer_extra) return g_pm_bf16_renderer_extra;
  PolyOpSet cast_set = poly_opset_add((PolyOpSet){{0, 0}}, POLY_OP_CAST);
  PolyOpSet const_set = poly_opset_add((PolyOpSet){{0, 0}}, POLY_OP_CONST);
  PolyRule rules[] = {
      {poly_pat_ops(POLY_GROUP_ALU, NULL, 0, NULL), rule_bf16_renderer_alu},
      {poly_pat_ops(cast_set, NULL, 0, NULL), rule_bf16_cast_via_f32},
      {poly_pat_ops(cast_set, NULL, 0, NULL), rule_bf16_renderer_manual_cast},
      {poly_pat_ops(const_set, NULL, 0, NULL), rule_bf16_renderer_const},
  };
  g_pm_bf16_renderer_extra =
      poly_pm_thread_cache(poly_pm_new(rules, (int)(sizeof(rules) / sizeof(rules[0]))));
  return g_pm_bf16_renderer_extra;
}

static _Thread_local PolyPatternMatcher *g_pm_c_renderer_extra = NULL;

PolyPatternMatcher *poly_pm_c_renderer_extra(void) {
  if (g_pm_c_renderer_extra) return g_pm_c_renderer_extra;

  PolyOpSet cast_set = poly_opset_add((PolyOpSet){{0, 0}}, POLY_OP_CAST);
  PolyRule c_rules[] = {
      {poly_pat_ops(cast_set, NULL, 0, NULL), rule_c_renderer_cast_via_f32},
  };
  g_pm_c_renderer_extra =
      poly_pm_thread_cache(poly_pm_new(c_rules, (int)(sizeof(c_rules) / sizeof(c_rules[0]))));
  return g_pm_c_renderer_extra;
}

/* Build the renderer-capability-specific transcendental matcher. Pinned
 * tinygrad only decomposes an operation absent from renderer.code_for_op. */
static _Thread_local PolyPatternMatcher *g_pm_transcendental_caps[2][2][2] = {0};

static PolyPatternMatcher *poly_pm_transcendental(PolyRendererCaps caps) {
  PolyPatternMatcher **target =
      &g_pm_transcendental_caps[caps.has_exp2 ? 1 : 0][caps.has_log2 ? 1 : 0][caps.has_sin ? 1 : 0];
  if (*target) return *target;

  PolyOpSet exp2_set = poly_opset_add((PolyOpSet){{0, 0}}, POLY_OP_EXP2);
  PolyOpSet log2_set = poly_opset_add((PolyOpSet){{0, 0}}, POLY_OP_LOG2);
  PolyOpSet sin_set = poly_opset_add((PolyOpSet){{0, 0}}, POLY_OP_SIN);
  PolyRule rules[6];
  int n = 0;
  if (!caps.has_exp2) {
    rules[n++] = (PolyRule){poly_pat_ops(exp2_set, NULL, 0, NULL), rule_decomp_exp2};
    rules[n++] =
        (PolyRule){poly_pat_ops(exp2_set, NULL, 0, NULL), rule_decomp_transcendental_other_float};
  }
  if (!caps.has_log2) {
    rules[n++] = (PolyRule){poly_pat_ops(log2_set, NULL, 0, NULL), rule_decomp_log2};
    rules[n++] =
        (PolyRule){poly_pat_ops(log2_set, NULL, 0, NULL), rule_decomp_transcendental_other_float};
  }
  if (!caps.has_sin) {
    rules[n++] = (PolyRule){poly_pat_ops(sin_set, NULL, 0, NULL), rule_decomp_sin};
    rules[n++] =
        (PolyRule){poly_pat_ops(sin_set, NULL, 0, NULL), rule_decomp_transcendental_other_float};
  }

  if (n == 0) {
    *target = poly_symbolic_simple();
    return *target;
  }
  PolyPatternMatcher *transcendental = poly_pm_new(rules, n);
  *target = poly_pm_thread_cache(poly_pm_concat(poly_symbolic_simple(), transcendental));
  poly_pm_destroy(transcendental);
  return *target;
}

/* Expander (tinygrad codegen/late/expander.py) */

static int64_t pair_tuple_prod(PolyArg a) {
  if (a.kind != POLY_ARG_PAIR_TUPLE) return 1;
  int64_t p = 1;
  for (int i = 0; i < a.pair_tuple.n; i++)
    p *= a.pair_tuple.pairs[i][1];
  return p;
}

static bool pair_tuple_empty(PolyArg a) {
  return a.kind == POLY_ARG_NONE || (a.kind == POLY_ARG_PAIR_TUPLE && a.pair_tuple.n == 0);
}

static PolyArg poly_arg_pair_tuple(int64_t (*pairs)[2], int n) {
  PolyArg a = poly_arg_none();
  a.kind = POLY_ARG_PAIR_TUPLE;
  a.pair_tuple.pairs = pairs;
  a.pair_tuple.n = n;
  return a;
}

static PolyArg poly_arg_int_tuple_local(int64_t *vals, int n) {
  PolyArg a = poly_arg_none();
  a.kind = POLY_ARG_INT_TUPLE;
  a.int_tuple.vals = vals;
  a.int_tuple.n = n;
  return a;
}

static bool pair_list_contains(int64_t (*pairs)[2], int n, int64_t axis, int64_t sz) {
  for (int i = 0; i < n; i++)
    if (pairs[i][0] == axis && pairs[i][1] == sz) return true;
  return false;
}

static int find_assignment(int64_t *ids, int64_t *vals, int n, int64_t axis, int64_t *out) {
  for (int i = 0; i < n; i++) {
    if (ids[i] == axis) {
      *out = vals[i];
      return 1;
    }
  }
  return 0;
}

static int64_t compute_flat_from_assignment(
    int64_t (*eargs)[2],
    int n_eargs,
    int64_t *ids,
    int64_t *vals,
    int n_assign
) {
  int64_t idx = 0, mul = 1;
  for (int i = n_eargs - 1; i >= 0; i--) {
    int64_t v = 0;
    (void)find_assignment(ids, vals, n_assign, eargs[i][0], &v);
    idx += v * mul;
    mul *= eargs[i][1];
  }
  return idx;
}

typedef struct {
  int64_t (*cargs)[2];
  int n_cargs;
  int64_t (*eargs)[2];
  int n_eargs;
  int64_t *vals;
  int64_t *out;
  int out_pos;
} SwizzleCtx;

static void swizzle_recur(SwizzleCtx *s, int dim) {
  if (dim == s->n_cargs) {
    int64_t ids[POLY_MAX_DIMS], v[POLY_MAX_DIMS];
    for (int i = 0; i < s->n_cargs; i++) {
      ids[i] = s->cargs[i][0];
      v[i] = s->vals[i];
    }
    s->out[s->out_pos++] = compute_flat_from_assignment(s->eargs, s->n_eargs, ids, v, s->n_cargs);
    return;
  }
  int64_t m = s->cargs[dim][1];
  for (int64_t i = 0; i < m; i++) {
    s->vals[dim] = i;
    swizzle_recur(s, dim + 1);
  }
}

static void decode_choice_index(
    int64_t idx,
    int64_t (*pairs)[2],
    int n,
    int64_t *ids,
    int64_t *vals
) {
  for (int i = n - 1; i >= 0; i--) {
    int64_t m = pairs[i][1];
    ids[i] = pairs[i][0];
    vals[i] = (m > 0) ? (idx % m) : 0;
    if (m > 0) idx /= m;
  }
}

static PolyDType dtype_for_gep_result(PolyDType src_dt, int n_idxs) {
  /* tinygrad's PtrDType has two independent lane concepts:
   *   count  = lanes in the pointed-to value, e.g. ptr(float4)
   *   vcount = lanes in a vector of pointers, e.g. ptr(float4).vec(4)
   *
   * Polygrad stores both fields too, but older late-codegen paths used count
   * for both. GEP(tuple) is the place where this distinction matters for
   * upcasted reductions: acc.cast(ptr(float4)).gep((...4 lanes...)) must stay
   * ptr(float4).vec(4), not collapse to ptr(float16). */
  if (n_idxs <= 1) {
    if (src_dt.is_ptr) {
      PolyDType r = src_dt;
      r.vcount = 1;
      return r;
    }
    return poly_dtype_scalar(src_dt);
  }
  if (src_dt.is_ptr) {
    PolyDType r = src_dt;
    r.vcount = (uint16_t)n_idxs;
    return r;
  }
  return poly_dtype_vec(poly_dtype_scalar(src_dt), n_idxs);
}

static PolyUOp *make_gep(PolyCtx *ctx, PolyUOp *base, int64_t *idxs, int n_idxs) {
  return poly_uop1(
      ctx, POLY_OP_GEP, dtype_for_gep_result(base->dtype, n_idxs), base,
      poly_arg_int_tuple_local(idxs, n_idxs)
  );
}

static PolyDType dtype_for_vectorize_result(PolyDType elem_dt, int lanes) {
  /* VECTORIZE/BROADCAST of a pointer is a vector of pointers in tinygrad
   * (PtrDType.vcount), not a pointer to a wider value (PtrDType.count). The
   * latter is produced explicitly by CAST to a vector pointer. */
  if (lanes <= 1) return elem_dt;
  if (elem_dt.is_ptr) {
    PolyDType r = elem_dt;
    int base_vcount = r.vcount > 1 ? r.vcount : 1;
    r.vcount = (uint16_t)(base_vcount * lanes);
    return r;
  }
  return poly_dtype_vec(elem_dt, lanes);
}

static PolyDType dtype_for_expand_result(PolyDType root_dt, int64_t expand_sz) {
  /* Expanding a pointer UOp follows the same pointer-vector rule as
   * VECTORIZE. This keeps upcasted register accumulators shaped as
   * ptr(floatN).vec(M), so no_vectorized_index can assign every output lane a
   * unique scalar register slot. */
  if (expand_sz <= 1) return root_dt;
  if (root_dt.is_ptr) {
    PolyDType r = root_dt;
    int base_vcount = r.vcount > 1 ? r.vcount : 1;
    r.vcount = (uint16_t)(base_vcount * expand_sz);
    return r;
  }
  return poly_dtype_vec(poly_dtype_scalar(root_dt), root_dt.count * (int)expand_sz);
}

static PolyUOp *do_expand(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  PolyUOp *expands[32];
  int n_expands = 0;
  for (int i = 0; i < root->n_src && n_expands < 32; i++) {
    if (root->src[i]->op == POLY_OP_UNROLL) expands[n_expands++] = root->src[i];
  }
  if (n_expands == 0) return NULL;

  int64_t expand_pairs[POLY_MAX_DIMS * 4][2];
  int n_expand_pairs = 0;
  for (int i = 0; i < n_expands; i++) {
    PolyArg a = expands[i]->arg;
    if (a.kind != POLY_ARG_PAIR_TUPLE) return NULL;
    for (int j = 0; j < a.pair_tuple.n; j++) {
      int64_t axis = a.pair_tuple.pairs[j][0], sz = a.pair_tuple.pairs[j][1];
      if (!pair_list_contains(expand_pairs, n_expand_pairs, axis, sz))
        expand_pairs[n_expand_pairs][0] = axis, expand_pairs[n_expand_pairs++][1] = sz;
    }
  }
  if (n_expand_pairs <= 0) return NULL;
  for (int i = 0; i < n_expand_pairs; i++) {
    for (int j = i + 1; j < n_expand_pairs; j++) {
      if (expand_pairs[j][0] < expand_pairs[i][0]) {
        int64_t t0 = expand_pairs[i][0], t1 = expand_pairs[i][1];
        expand_pairs[i][0] = expand_pairs[j][0];
        expand_pairs[i][1] = expand_pairs[j][1];
        expand_pairs[j][0] = t0;
        expand_pairs[j][1] = t1;
      }
    }
  }
  int64_t expand_sz = 1;
  for (int i = 0; i < n_expand_pairs; i++)
    expand_sz *= expand_pairs[i][1];
  if (expand_sz <= 1) return NULL;

  PolyUOp **new_srcs = calloc((size_t)root->n_src, sizeof(*new_srcs));
  if (!new_srcs) return NULL;
  int n_new_srcs = 0;
  for (int i = 0; i < root->n_src; i++) {
    PolyUOp *src = root->src[i];
    if (src->op == POLY_OP_UNROLL) {
      if (poly_arg_eq(src->arg, poly_arg_pair_tuple(expand_pairs, n_expand_pairs))) {
        new_srcs[n_new_srcs++] = src->src[0];
      } else {
        if (src->arg.kind != POLY_ARG_PAIR_TUPLE || src->n_src == 0) {
          free(new_srcs);
          return NULL;
        }
        int64_t n_swz = expand_sz;
        int64_t *swz = malloc((size_t)n_swz * sizeof(int64_t));
        if (!swz) {
          free(new_srcs);
          return NULL;
        }
        int64_t vals[POLY_MAX_DIMS] = {0};
        SwizzleCtx swz_ctx = {
            .cargs = expand_pairs,
            .n_cargs = n_expand_pairs,
            .eargs = src->arg.pair_tuple.pairs,
            .n_eargs = src->arg.pair_tuple.n,
            .vals = vals,
            .out = swz,
            .out_pos = 0,
        };
        swizzle_recur(&swz_ctx, 0);
        int n_gep = (int)n_swz;
        if (src->dtype.count > 1) {
          int n2 = n_gep * src->dtype.count;
          int64_t *lst2 = malloc((size_t)n2 * sizeof(int64_t));
          int p = 0;
          for (int k = 0; k < n_gep; k++)
            for (int j = 0; j < src->dtype.count; j++)
              lst2[p++] = swz[k] * src->dtype.count + j;
          free(swz);
          swz = lst2;
          n_gep = n2;
        }
        new_srcs[n_new_srcs++] = make_gep(ctx, src->src[0], swz, n_gep);
        free(swz);
      }
      continue;
    }

    int off = range_start_for_op(root->op);
    if (off >= 0 && i >= off) {
      new_srcs[n_new_srcs++] = src;
      continue;
    }
    if (root->op == POLY_OP_INDEX && i >= 1 && !root->dtype.is_ptr) {
      new_srcs[n_new_srcs++] = src;
      continue;
    }

    if (src->dtype.count > 1) {
      int n_cat = (int)expand_sz;
      PolyUOp **cat_srcs = calloc((size_t)n_cat, sizeof(*cat_srcs));
      if (!cat_srcs) {
        free(new_srcs);
        return NULL;
      }
      for (int j = 0; j < n_cat; j++)
        cat_srcs[j] = src;
      new_srcs[n_new_srcs++] = poly_uop(
          ctx, POLY_OP_VCAT,
          poly_dtype_vec(poly_dtype_scalar(src->dtype), n_cat * src->dtype.count), cat_srcs, n_cat,
          poly_arg_none()
      );
      free(cat_srcs);
    } else {
      int n_vec = (int)expand_sz;
      PolyUOp **vec_srcs = calloc((size_t)n_vec, sizeof(*vec_srcs));
      if (!vec_srcs) {
        free(new_srcs);
        return NULL;
      }
      for (int j = 0; j < n_vec; j++)
        vec_srcs[j] = src;
      new_srcs[n_new_srcs++] = poly_uop(
          ctx, POLY_OP_VECTORIZE, dtype_for_vectorize_result(src->dtype, n_vec), vec_srcs, n_vec,
          poly_arg_none()
      );
      free(vec_srcs);
    }
  }

  PolyDType out_dt = dtype_for_expand_result(root->dtype, expand_sz);
  PolyUOp *nsrc = poly_uop(ctx, root->op, out_dt, new_srcs, n_new_srcs, root->arg);
  PolyUOp *ret = poly_uop1(
      ctx, POLY_OP_UNROLL, root->dtype, nsrc, poly_arg_pair_tuple(expand_pairs, n_expand_pairs)
  );
  free(new_srcs);
  return ret;
}

static PolyUOp *do_contract(PolyCtx *ctx, PolyUOp *con, const PolyBindings *b) {
  (void)b;
  if (con->n_src < 1) return NULL;
  PolyUOp *ex = con->src[0];
  if (ex->op != POLY_OP_UNROLL) {
    if (con->dtype.count <= 1) return NULL;
    PolyUOp **srcs = calloc((size_t)con->dtype.count, sizeof(*srcs));
    if (!srcs) return NULL;
    for (int i = 0; i < con->dtype.count; i++)
      srcs[i] = ex;
    PolyUOp *ret =
        poly_uop(ctx, POLY_OP_VECTORIZE, con->dtype, srcs, con->dtype.count, poly_arg_none());
    free(srcs);
    return ret;
  }
  if (con->arg.kind != POLY_ARG_PAIR_TUPLE || ex->arg.kind != POLY_ARG_PAIR_TUPLE || ex->n_src == 0)
    return NULL;

  int64_t new_pairs[POLY_MAX_DIMS * 4][2];
  int n_new = 0;
  for (int i = 0; i < ex->arg.pair_tuple.n; i++) {
    int64_t axis = ex->arg.pair_tuple.pairs[i][0], sz = ex->arg.pair_tuple.pairs[i][1];
    if (!pair_list_contains(con->arg.pair_tuple.pairs, con->arg.pair_tuple.n, axis, sz)) {
      new_pairs[n_new][0] = axis;
      new_pairs[n_new][1] = sz;
      n_new++;
    }
  }

  int64_t new_prod = 1;
  for (int i = 0; i < n_new; i++)
    new_prod *= new_pairs[i][1];
  int64_t con_prod = 1;
  for (int i = 0; i < con->arg.pair_tuple.n; i++)
    con_prod *= con->arg.pair_tuple.pairs[i][1];
  if (new_prod <= 0 || con_prod <= 0 || new_prod > INT32_MAX / con_prod) return NULL;
  int n_idxs = (int)(new_prod * con_prod);

  int64_t *idxs = calloc((size_t)n_idxs, sizeof(*idxs));
  int64_t *ids = calloc((size_t)(n_new + con->arg.pair_tuple.n), sizeof(*ids));
  int64_t *vals = calloc((size_t)(n_new + con->arg.pair_tuple.n), sizeof(*vals));
  if (!idxs || !ids || !vals) {
    free(idxs);
    free(ids);
    free(vals);
    return NULL;
  }

  int p = 0;
  for (int64_t rpk = 0; rpk < new_prod; rpk++) {
    decode_choice_index(rpk, new_pairs, n_new, ids, vals);
    for (int64_t lrpk = 0; lrpk < con_prod; lrpk++) {
      decode_choice_index(
          lrpk, con->arg.pair_tuple.pairs, con->arg.pair_tuple.n, ids + n_new, vals + n_new
      );
      idxs[p++] = compute_flat_from_assignment(
          ex->arg.pair_tuple.pairs, ex->arg.pair_tuple.n, ids, vals, n_new + con->arg.pair_tuple.n
      );
    }
  }

  PolyUOp *gep = make_gep(ctx, ex->src[0], idxs, n_idxs);
  free(idxs);
  free(ids);
  free(vals);

  return poly_uop1(ctx, POLY_OP_UNROLL, con->dtype, gep, poly_arg_pair_tuple(new_pairs, n_new));
}

static PolyUOp *end_unrolls(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b) {
  (void)b;
  if (u->op != POLY_OP_END || u->n_src <= 1) return NULL;
  PolyUOp *unrolls[32], *others[64];
  int n_unrolls = 0, n_others = 0;
  for (int i = 1; i < u->n_src; i++) {
    if (u->src[i]->op == POLY_OP_UNROLL && n_unrolls < 32)
      unrolls[n_unrolls++] = u->src[i];
    else if (n_others < 64)
      others[n_others++] = u->src[i];
  }
  if (n_unrolls == 0) return NULL;

  int64_t pairs[POLY_MAX_DIMS * 4][2];
  int n_pairs = 0;
  for (int i = 0; i < n_unrolls; i++) {
    PolyArg a = unrolls[i]->arg;
    if (a.kind != POLY_ARG_PAIR_TUPLE) continue;
    for (int j = 0; j < a.pair_tuple.n; j++) {
      int64_t axis = a.pair_tuple.pairs[j][0], sz = a.pair_tuple.pairs[j][1];
      if (!pair_list_contains(pairs, n_pairs, axis, sz))
        pairs[n_pairs][0] = axis, pairs[n_pairs++][1] = sz;
    }
  }
  PolyUOp *ret =
      poly_uop1(ctx, POLY_OP_CONTRACT, POLY_VOID, u->src[0], poly_arg_pair_tuple(pairs, n_pairs));
  PolyUOp *new_src[65];
  int n_new = 0;
  new_src[n_new++] = ret;
  for (int i = 0; i < n_others; i++)
    new_src[n_new++] = others[i];
  return poly_uop(ctx, u->op, u->dtype, new_src, n_new, u->arg);
}

static PolyUOp *rule_empty_unroll(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b) {
  (void)ctx;
  (void)b;
  if (u->op != POLY_OP_UNROLL || u->n_src < 1) return NULL;
  return pair_tuple_empty(u->arg) ? u->src[0] : NULL;
}

static PolyUOp *rule_double_unroll(PolyCtx *ctx, PolyUOp *outer, const PolyBindings *b) {
  PolyUOp *inner = poly_bind(b, "inner");
  if (!inner || inner->op != POLY_OP_UNROLL || inner->n_src == 0) return NULL;
  if (outer->arg.kind != POLY_ARG_PAIR_TUPLE || inner->arg.kind != POLY_ARG_PAIR_TUPLE) return NULL;
  int n = inner->arg.pair_tuple.n + outer->arg.pair_tuple.n;
  int64_t pairs[POLY_MAX_DIMS * 4][2];
  if (n > (int)(sizeof(pairs) / sizeof(pairs[0]))) return NULL;
  int p = 0;
  for (int i = 0; i < inner->arg.pair_tuple.n; i++, p++) {
    pairs[p][0] = inner->arg.pair_tuple.pairs[i][0];
    pairs[p][1] = inner->arg.pair_tuple.pairs[i][1];
  }
  for (int i = 0; i < outer->arg.pair_tuple.n; i++, p++) {
    pairs[p][0] = outer->arg.pair_tuple.pairs[i][0];
    pairs[p][1] = outer->arg.pair_tuple.pairs[i][1];
  }
  return poly_uop1(ctx, POLY_OP_UNROLL, outer->dtype, inner->src[0], poly_arg_pair_tuple(pairs, p));
}

static PolyUOp *rule_pre_expand_range(PolyCtx *ctx, PolyUOp *r, const PolyBindings *b) {
  (void)b;
  if (r->op != POLY_OP_RANGE || !poly_arg_is_range(r->arg)) return NULL;
  PolyAxisType t = poly_range_axis_type(r->arg);
  if (!(t == POLY_AXIS_UPCAST || t == POLY_AXIS_UNROLL)) return NULL;
  if (!(r->n_src > 0 && r->src[0]->op == POLY_OP_CONST && r->src[0]->arg.kind == POLY_ARG_INT))
    return NULL;
  int64_t s = r->src[0]->arg.i;
  if (s <= 0 || s > 128) return NULL;

  PolyUOp *vals[128];
  for (int64_t i = 0; i < s; i++)
    vals[i] = poly_uop0(ctx, POLY_OP_CONST, r->dtype, poly_arg_int(i));
  PolyUOp *vconst = poly_uop(
      ctx, POLY_OP_VCONST, poly_dtype_vec(r->dtype, (int)s), vals, (int)s, poly_arg_none()
  );
  int64_t pairs[1][2] = {{poly_range_axis_id(r->arg), s}};
  return poly_uop1(ctx, POLY_OP_UNROLL, r->dtype, vconst, poly_arg_pair_tuple(pairs, 1));
}

static PolyUOp *rule_fix_reduce_unroll(PolyCtx *ctx, PolyUOp *x, const PolyBindings *b) {
  (void)b;
  if (x->op != POLY_OP_REDUCE || x->n_src <= 1) return NULL;
  PolyUOp *reduce_range[32], *reduce_expand[32];
  int n_range = 0, n_expand = 0;
  for (int i = 1; i < x->n_src; i++) {
    if (x->src[i]->op == POLY_OP_RANGE && n_range < 32)
      reduce_range[n_range++] = x->src[i];
    else if (n_expand < 32)
      reduce_expand[n_expand++] = x->src[i];
  }
  if (n_expand == 0) return NULL;
  int64_t pairs[POLY_MAX_DIMS * 4][2];
  int n_pairs = 0;
  for (int i = 0; i < n_expand; i++) {
    if (reduce_expand[i]->op == POLY_OP_CONST) continue;
    if (reduce_expand[i]->op != POLY_OP_UNROLL || reduce_expand[i]->arg.kind != POLY_ARG_PAIR_TUPLE)
      return NULL;
    for (int j = 0; j < reduce_expand[i]->arg.pair_tuple.n; j++) {
      int64_t axis = reduce_expand[i]->arg.pair_tuple.pairs[j][0];
      int64_t sz = reduce_expand[i]->arg.pair_tuple.pairs[j][1];
      if (!pair_list_contains(pairs, n_pairs, axis, sz))
        pairs[n_pairs][0] = axis, pairs[n_pairs++][1] = sz;
    }
  }
  PolyUOp *ret = x->src[0];
  if (n_pairs > 0)
    ret = poly_uop1(
        ctx, POLY_OP_CONTRACT,
        poly_dtype_vec(x->dtype, (int)pair_tuple_prod(poly_arg_pair_tuple(pairs, n_pairs))), ret,
        poly_arg_pair_tuple(pairs, n_pairs)
    );

  PolyUOp *new_src[64];
  int n_new = 0;
  new_src[n_new++] = ret;
  for (int i = 0; i < n_range; i++)
    new_src[n_new++] = reduce_range[i];
  return poly_uop(ctx, POLY_OP_REDUCE, x->dtype, new_src, n_new, x->arg);
}

static PolyUOp *rule_fix_store_unroll(PolyCtx *ctx, PolyUOp *x, const PolyBindings *b) {
  (void)b;
  if (x->op != POLY_OP_STORE || x->n_src <= 2) return NULL;
  PolyUOp *store_expand[32], *store_range[32];
  int n_expand = 0, n_range = 0;
  for (int i = 2; i < x->n_src; i++) {
    if (x->src[i]->op == POLY_OP_UNROLL && n_expand < 32)
      store_expand[n_expand++] = x->src[i];
    else if (n_range < 32)
      store_range[n_range++] = x->src[i];
  }
  if (n_expand == 0) return NULL;

  PolyUOp *base_src[64];
  int n_base = 0;
  base_src[n_base++] = x->src[0];
  base_src[n_base++] = x->src[1];
  for (int i = 0; i < n_range; i++)
    base_src[n_base++] = store_range[i];
  PolyUOp *base_store = poly_uop(ctx, POLY_OP_STORE, x->dtype, base_src, n_base, x->arg);

  int64_t pairs[POLY_MAX_DIMS * 4][2];
  int n_pairs = 0;
  for (int i = 0; i < n_expand; i++) {
    if (store_expand[i]->arg.kind != POLY_ARG_PAIR_TUPLE) continue;
    for (int j = 0; j < store_expand[i]->arg.pair_tuple.n; j++) {
      int64_t axis = store_expand[i]->arg.pair_tuple.pairs[j][0];
      int64_t sz = store_expand[i]->arg.pair_tuple.pairs[j][1];
      if (!pair_list_contains(pairs, n_pairs, axis, sz))
        pairs[n_pairs][0] = axis, pairs[n_pairs++][1] = sz;
    }
  }
  return poly_uop1(
      ctx, POLY_OP_CONTRACT, POLY_VOID, base_store, poly_arg_pair_tuple(pairs, n_pairs)
  );
}

static _Thread_local PolyPatternMatcher *g_pm_pre_expander = NULL;
static PolyPatternMatcher *poly_pm_pre_expander(void) {
  if (g_pm_pre_expander) return g_pm_pre_expander;
  PolyRule rules[] = {
      {poly_pat_op(POLY_OP_RANGE, NULL, 0, "r"), rule_pre_expand_range},
      {poly_pat_op(POLY_OP_REDUCE, NULL, 0, "x"), rule_fix_reduce_unroll},
      {poly_pat_op(POLY_OP_STORE, NULL, 0, "x"), rule_fix_store_unroll},
  };
  g_pm_pre_expander =
      poly_pm_thread_cache(poly_pm_new(rules, (int)(sizeof(rules) / sizeof(rules[0]))));
  return g_pm_pre_expander;
}

static _Thread_local PolyPatternMatcher *g_pm_expander = NULL;
static PolyPatternMatcher *poly_pm_expander(void) {
  if (g_pm_expander) return g_pm_expander;
  PolyOpSet exp_ops = POLY_GROUP_ALU;
  exp_ops = poly_opset_add(exp_ops, POLY_OP_CAST);
  exp_ops = poly_opset_add(exp_ops, POLY_OP_BITCAST);
  exp_ops = poly_opset_add(exp_ops, POLY_OP_GEP);
  exp_ops = poly_opset_add(exp_ops, POLY_OP_WMMA);
  exp_ops = poly_opset_add(exp_ops, POLY_OP_LOAD);
  exp_ops = poly_opset_add(exp_ops, POLY_OP_STORE);
  exp_ops = poly_opset_add(exp_ops, POLY_OP_INDEX);
  exp_ops = poly_opset_add(exp_ops, POLY_OP_STAGE);
  exp_ops = poly_opset_add(exp_ops, POLY_OP_VECTORIZE);
  exp_ops = poly_opset_add(exp_ops, POLY_OP_REDUCE);
  exp_ops = poly_opset_add(exp_ops, POLY_OP_END);
  exp_ops = poly_opset_add(exp_ops, POLY_OP_AFTER);

  PolyOpSet rej = poly_opset_add((PolyOpSet){{0, 0}}, POLY_OP_UNROLL);
  PolyRule rules[] = {
      {poly_pat_op(POLY_OP_END, NULL, 0, "u"), end_unrolls},
      {poly_pat_op1(POLY_OP_UNROLL, poly_pat_op(POLY_OP_UNROLL, NULL, 0, "inner"), "outer"),
       rule_double_unroll},
      {poly_pat_set_early_reject(
           poly_pat_allow_any_len(poly_pat_ops(exp_ops, NULL, 0, "root")), rej
       ),
       do_expand},
      {poly_pat_op(POLY_OP_CONTRACT, NULL, 0, "con"), do_contract},
      {poly_pat_op(POLY_OP_UNROLL, NULL, 0, "u"), rule_empty_unroll},
  };
  g_pm_expander = poly_pm_thread_cache(poly_pm_new(rules, (int)(sizeof(rules) / sizeof(rules[0]))));
  return g_pm_expander;
}

/* pm_add_loads (tinygrad codegen/late/devectorizer.py) */

static PolyUOp *rule_add_load_to_index(PolyCtx *ctx, PolyUOp *idx, const PolyBindings *b) {
  (void)b;
  if (idx->op != POLY_OP_INDEX) return NULL;
  /* tinygrad pm_add_loads only lowers scalar read INDEX nodes here.
   * ptr-typed INDEX is already the lowered store-target form and must stay as
   * INDEX until STORE rendering. */
  if (idx->dtype.is_ptr) return NULL;
  if (idx->n_src < 1 || !idx->src[0]) return NULL;

  PolyDType ptr_dt = idx->src[0]->dtype;
  if (!ptr_dt.is_ptr) return NULL;

  /* Rebuild the same INDEX with a ptr dtype, then LOAD the original scalar
   * element dtype. This matches tinygrad's buf.index(..., ptr=True).load(). */
  PolyUOp *ptr_idx = poly_uop(ctx, POLY_OP_INDEX, ptr_dt, idx->src, idx->n_src, idx->arg);
  return poly_uop1(ctx, POLY_OP_LOAD, idx->dtype, ptr_idx, poly_arg_none());
}

static PolyUOp *rule_remove_load_from_store(PolyCtx *ctx, PolyUOp *s, const PolyBindings *b) {
  PolyUOp *ld = poly_bind(b, "ld");
  if (!ld || ld->op != POLY_OP_LOAD || ld->n_src < 1 || s->op != POLY_OP_STORE || s->n_src < 2)
    return NULL;
  PolyUOp *new_src[64];
  int n_new = 0;
  new_src[n_new++] = ld->src[0];
  new_src[n_new++] = s->src[1];
  for (int i = 2; i < s->n_src && n_new < 64; i++)
    new_src[n_new++] = s->src[i];
  return poly_uop(ctx, POLY_OP_STORE, s->dtype, new_src, n_new, s->arg);
}

static _Thread_local PolyPatternMatcher *g_pm_add_loads = NULL;
static PolyPatternMatcher *poly_pm_add_loads(void) {
  if (g_pm_add_loads) return g_pm_add_loads;
  PolyRule rules[] = {
      {poly_pat_op(POLY_OP_INDEX, NULL, 0, "idx"), rule_add_load_to_index},
      {poly_pat_allow_any_len(poly_pat_op2(
           POLY_OP_STORE, poly_pat_op(POLY_OP_LOAD, NULL, 0, "ld"), poly_pat_any("val"), "s"
       )),
       rule_remove_load_from_store},
  };
  g_pm_add_loads =
      poly_pm_thread_cache(poly_pm_new(rules, (int)(sizeof(rules) / sizeof(rules[0]))));
  return g_pm_add_loads;
}

PolyPatternMatcher *poly_pm_add_loads_pass(void) {
  return poly_pm_add_loads();
}

/* load_store_folding (subset port for vectorized INDEX collapse) */

static PolyUOp *make_gep_lane(PolyCtx *ctx, PolyUOp *src, int lane);
static PolyUOp *build_scalar_lane_index(PolyCtx *ctx, PolyUOp *idx, PolyUOp *buf_base, int lane);

static PolyUOp *scalarize_lane_expr(PolyCtx *ctx, PolyUOp *u, int lane) {
  if (!u) return NULL;
  if (u->op == POLY_OP_UNROLL && u->n_src > 0) {
    PolyUOp *src = u->src[0];
    if (src && src->dtype.count > 1) {
      int cnt = src->dtype.count;
      int pick = (cnt > 0) ? (lane % cnt) : 0;
      if (pick < 0) pick = 0;
      return scalarize_lane_expr(ctx, src, pick);
    }
    return src;
  }
  if (u->dtype.count <= 1) return u;
  if (lane < 0) lane = 0;

  if (u->op == POLY_OP_VECTORIZE) {
    if (u->n_src <= 0) return NULL;
    int pick = lane;
    if (pick >= u->n_src) pick = u->n_src - 1;
    return scalarize_lane_expr(ctx, u->src[pick], 0);
  }
  if (u->op == POLY_OP_VCAT) {
    int pos = lane;
    for (int i = 0; i < u->n_src; i++) {
      PolyUOp *src = u->src[i];
      int cnt = src->dtype.count > 1 ? src->dtype.count : 1;
      if (pos < cnt) return scalarize_lane_expr(ctx, src, pos);
      pos -= cnt;
    }
    return NULL;
  }
  if (u->op == POLY_OP_VCONST) {
    PolyDType sdt = poly_dtype_scalar(u->dtype);
    if (u->n_src > lane && u->src[lane] && u->src[lane]->op == POLY_OP_CONST) return u->src[lane];
    if (u->arg.kind == POLY_ARG_INT_TUPLE && lane < u->arg.int_tuple.n)
      return poly_uop0(ctx, POLY_OP_CONST, sdt, poly_arg_int(u->arg.int_tuple.vals[lane]));
    return NULL;
  }
  if (u->op == POLY_OP_GEP && u->n_src > 0) {
    if (u->arg.kind == POLY_ARG_INT) {
      return scalarize_lane_expr(ctx, u->src[0], (int)u->arg.i);
    }
    if (u->arg.kind == POLY_ARG_INT_TUPLE && u->arg.int_tuple.n > 0) {
      int pick = (u->arg.int_tuple.n == 1) ? (int)u->arg.int_tuple.vals[0] : lane;
      if (pick < 0) pick = 0;
      if (pick >= u->arg.int_tuple.n) pick = u->arg.int_tuple.n - 1;
      return scalarize_lane_expr(ctx, u->src[0], (int)u->arg.int_tuple.vals[pick]);
    }
  }

  bool can_scalarize =
      poly_opset_has(POLY_GROUP_ALU, u->op) || u->op == POLY_OP_CAST || u->op == POLY_OP_BITCAST;
  if (can_scalarize) {
    PolyUOp *srcs[8];
    if (u->n_src > 8) return make_gep_lane(ctx, u, lane);
    for (int i = 0; i < u->n_src; i++) {
      PolyUOp *s = u->src[i];
      srcs[i] = (s && s->dtype.count > 1) ? scalarize_lane_expr(ctx, s, lane) : s;
      if (!srcs[i]) return make_gep_lane(ctx, u, lane);
    }
    PolyDType sdt = poly_dtype_scalar(u->dtype);
    if (u->op == POLY_OP_CAST) {
      if (u->n_src != 1) return make_gep_lane(ctx, u, lane);
      return poly_uop1(ctx, POLY_OP_CAST, sdt, srcs[0], u->arg);
    }
    if (u->op == POLY_OP_BITCAST) {
      if (u->n_src != 1) return make_gep_lane(ctx, u, lane);
      return poly_uop1(ctx, POLY_OP_BITCAST, sdt, srcs[0], u->arg);
    }
    if (u->n_src == 1) return poly_uop1(ctx, u->op, sdt, srcs[0], u->arg);
    if (u->n_src == 2) return poly_uop2(ctx, u->op, sdt, srcs[0], srcs[1], u->arg);
    if (u->n_src == 3) return poly_uop3(ctx, u->op, sdt, srcs[0], srcs[1], srcs[2], u->arg);
  }
  return make_gep_lane(ctx, u, lane);
}

static int uop_ptr_cmp(const void *ap, const void *bp) {
  const PolyUOp *a = *(const PolyUOp *const *)ap;
  const PolyUOp *b = *(const PolyUOp *const *)bp;
  if (a < b) return -1;
  if (a > b) return 1;
  return 0;
}

static bool expr_divides_const_codegen(PolyUOp *u, int64_t v) {
  if (!u || v == 0) return false;
  if (v == 1) return true;
  if (u->op == POLY_OP_CONST && u->arg.kind == POLY_ARG_INT) return (u->arg.i % v) == 0;
  if (u->op == POLY_OP_ADD && u->n_src == 2)
    return expr_divides_const_codegen(u->src[0], v) && expr_divides_const_codegen(u->src[1], v);
  if (u->op == POLY_OP_MUL && u->n_src == 2)
    return expr_divides_const_codegen(u->src[0], v) || expr_divides_const_codegen(u->src[1], v);
  return false;
}

typedef struct {
  PolyUOp *terms[64];
  int n_terms;
  int64_t cst;
  bool ok;
} LaneAffineExpr;

static void collect_add_terms(PolyCtx *ctx, PolyUOp *u, LaneAffineExpr *out, bool negate) {
  (void)ctx;
  if (!u || !out || !out->ok) return;
  if (u->op == POLY_OP_ADD && u->n_src == 2) {
    collect_add_terms(ctx, u->src[0], out, negate);
    collect_add_terms(ctx, u->src[1], out, negate);
    return;
  }
  if (u->op == POLY_OP_SUB && u->n_src == 2) {
    collect_add_terms(ctx, u->src[0], out, negate);
    collect_add_terms(ctx, u->src[1], out, !negate);
    return;
  }
  if (u->op == POLY_OP_CONST && u->arg.kind == POLY_ARG_INT) {
    int64_t v = negate ? -u->arg.i : u->arg.i;
    out->cst += v;
    return;
  }
  if (negate) {
    if (out->n_terms >= 64) {
      out->ok = false;
      return;
    }
    out->terms[out->n_terms++] = poly_uop1(ctx, POLY_OP_NEG, u->dtype, u, poly_arg_none());
    return;
  }
  if (out->n_terms >= 64) {
    out->ok = false;
    return;
  }
  out->terms[out->n_terms++] = u;
}

static PolyUOp *build_add_expr(
    PolyCtx *ctx,
    PolyDType dt,
    PolyUOp **terms,
    int n_terms,
    int64_t cst
) {
  PolyUOp *ret = NULL;
  if (n_terms > 0) {
    ret = terms[0];
    for (int i = 1; i < n_terms; i++)
      ret = poly_uop2(ctx, POLY_OP_ADD, dt, ret, terms[i], poly_arg_none());
  }
  if (cst != 0 || !ret) {
    PolyUOp *c = poly_uop0(ctx, POLY_OP_CONST, dt, poly_arg_int(cst));
    ret = ret ? poly_uop2(ctx, POLY_OP_ADD, dt, ret, c, poly_arg_none()) : c;
  }
  return ret;
}

static PolyUOp *build_scalar_lane_index(PolyCtx *ctx, PolyUOp *idx, PolyUOp *buf_base, int lane) {
  if (!idx || idx->op != POLY_OP_INDEX || idx->n_src < 2 || !buf_base) return NULL;
  PolyUOp *srcs[64];
  int n_srcs = 0;
  srcs[n_srcs++] = buf_base;
  srcs[n_srcs++] = scalarize_lane_expr(ctx, idx->src[1], lane);
  if (!srcs[1]) return NULL;
  for (int i = 2; i < idx->n_src && n_srcs < 64; i++) {
    PolyUOp *s = idx->src[i];
    if (s && s->dtype.count > 1) s = scalarize_lane_expr(ctx, s, lane);
    if (!s) return NULL;
    srcs[n_srcs++] = s;
  }
  return poly_uop(ctx, POLY_OP_INDEX, buf_base->dtype, srcs, n_srcs, idx->arg);
}

/* load_store_folding: PTRCAT pipeline (tinygrad devectorizer.py:63-136) */
static int poly_ptr_lane_count(PolyDType dt);
static PolyDType poly_dtype_ptr_vec(PolyDType ptr, int lanes);

/* expand_index (tinygrad devectorizer.py:63-66):
 * INDEX(STACK(buf,...), vec_idx) → STACK(INDEX(buf, GEP(vec,0)), ..., INDEX(buf,
 * GEP(vec,N-1))) Scatters a vectorized INDEX into per-element scalar INDEXes. */
static PolyUOp *rule_expand_index(PolyCtx *ctx, PolyUOp *idx, const PolyBindings *b) {
  (void)b;
  if (!idx || idx->op != POLY_OP_INDEX || idx->n_src < 2) return NULL;
  PolyUOp *buf_vec = idx->src[0];
  if (!buf_vec || (buf_vec->op != POLY_OP_STACK && buf_vec->op != POLY_OP_VECTORIZE) ||
      buf_vec->n_src <= 1)
    return NULL;
  /* tinygrad load_store_folding expands INDEX(STACK(buf,...), vec) before
   * pm_add_loads. The repeated base can be a shaped pointer view such as
   * RESHAPE(PARAM), not only a raw PARAM/AFTER node. */
  PolyUOp *buf = buf_vec->src[0];
  if (!buf || !buf->dtype.is_ptr) return NULL;
  for (int i = 1; i < buf_vec->n_src; i++)
    if (buf_vec->src[i] != buf) return NULL; /* all same buf */

  PolyUOp *vec = idx->src[1];
  int cnt = buf_vec->n_src;
  PolyUOp **elems = calloc((size_t)cnt, sizeof(*elems));
  if (!elems) return NULL;
  for (int i = 0; i < cnt; i++) {
    PolyUOp *gi = make_gep_lane(ctx, vec, i);
    /* Build INDEX(buf, gi) with ptr output (same as buf->dtype) */
    PolyUOp *idx_srcs[64];
    int ns = 0;
    idx_srcs[ns++] = buf;
    idx_srcs[ns++] = gi;
    for (int j = 2; j < idx->n_src && ns < 64; j++)
      idx_srcs[ns++] = idx->src[j];
    elems[i] = poly_uop(ctx, POLY_OP_INDEX, buf->dtype, idx_srcs, ns, idx->arg);
  }
  PolyUOp *ret = poly_uop(ctx, POLY_OP_STACK, buf->dtype, elems, cnt, poly_arg_none());
  free(elems);
  return ret;
}

/* fold_expanded_index (tinygrad devectorizer.py:68-104):
 * STACK(INDEX(buf,off0), INDEX(buf,off1), ...) → PTRCAT(...).gep(remap)
 * Groups contiguous offsets into vector pointer CASTs, wraps in PTRCAT. */
static PolyUOp *rule_fold_expanded_index(PolyCtx *ctx, PolyUOp *midx, const PolyBindings *b) {
  (void)b;
  if (!midx || (midx->op != POLY_OP_STACK && midx->op != POLY_OP_VECTORIZE) || midx->n_src <= 1)
    return NULL;
  /* All sources must be INDEX ops */
  for (int i = 0; i < midx->n_src; i++)
    if (!midx->src[i] || midx->src[i]->op != POLY_OP_INDEX) return NULL;
  /* All INDEX ops must reference the same buffer */
  PolyUOp *buf = midx->src[0]->src[0];
  if (!buf) return NULL;
  for (int i = 1; i < midx->n_src; i++)
    if (midx->src[i]->src[0] != buf) return NULL;
  /* All INDEX outputs must be pointer types */
  for (int i = 0; i < midx->n_src; i++)
    if (!midx->src[i]->dtype.is_ptr) return NULL;

  int n = midx->n_src;

  /* Extract offsets: decompose each INDEX's idx expr into (root_src, const_offset).
   * Polygrad simplified model: INDEX.src[1] is the offset directly (no get_idx/get_valid). */
  typedef struct {
    PolyUOp *root;
    int64_t offset;
    int orig_idx;
  } OffsetEntry;
  OffsetEntry *entries = calloc((size_t)n, sizeof(*entries));
  int64_t *idxs = malloc((size_t)n * sizeof(*idxs));
  bool *used = calloc((size_t)n, sizeof(*used));
  PolyUOp **ret_srcs = calloc((size_t)n, sizeof(*ret_srcs));
  int *group = malloc((size_t)n * sizeof(*group));
  int64_t *group_offsets = malloc((size_t)n * sizeof(*group_offsets));
  if (!entries || !idxs || !used || !ret_srcs || !group || !group_offsets) {
    free(entries);
    free(idxs);
    free(used);
    free(ret_srcs);
    free(group);
    free(group_offsets);
    return NULL;
  }
  for (int i = 0; i < n; i++) {
    PolyUOp *idx_expr = midx->src[i]->src[1];
    entries[i].orig_idx = i;
    /* Decompose idx_expr into root + const offset */
    LaneAffineExpr ae = {.n_terms = 0, .cst = 0, .ok = true};
    collect_add_terms(ctx, idx_expr, &ae, false);
    if (!ae.ok) {
      entries[i].root = idx_expr;
      entries[i].offset = 0;
      continue;
    }
    qsort(ae.terms, (size_t)ae.n_terms, sizeof(ae.terms[0]), uop_ptr_cmp);
    entries[i].offset = ae.cst;
    entries[i].root =
        build_add_expr(ctx, poly_dtype_scalar(idx_expr->dtype), ae.terms, ae.n_terms, 0);
  }

  /* Group entries by root_src, then find contiguous offset sequences */
  /* Simple approach: group entries with same root pointer, sort by offset, find contiguous runs */
  int n_ret = 0;
  for (int i = 0; i < n; i++)
    idxs[i] = -1;
  int global_offset = 0;

  for (int i = 0; i < n; i++)
    used[i] = false;

  for (int i = 0; i < n; i++) {
    if (used[i]) continue;
    /* Collect all entries with same root */
    int ng = 0;
    for (int j = i; j < n; j++) {
      if (used[j]) continue;
      if (entries[j].root == entries[i].root) {
        group[ng] = j;
        group_offsets[ng] = entries[j].offset;
        ng++;
      }
    }
    /* Sort group by offset */
    for (int a = 0; a < ng - 1; a++)
      for (int bb = a + 1; bb < ng; bb++)
        if (group_offsets[a] > group_offsets[bb]) {
          int64_t to = group_offsets[a];
          group_offsets[a] = group_offsets[bb];
          group_offsets[bb] = to;
          int ti = group[a];
          group[a] = group[bb];
          group[bb] = ti;
        }
    /* Find contiguous runs within the group */
    int run_start = 0;
    while (run_start < ng) {
      int run_end = run_start + 1;
      while (run_end < ng && group_offsets[run_end] == group_offsets[run_end - 1] + 1)
        run_end++;
      int run_len = run_end - run_start;
      /* Use the first INDEX in the run as the base pointer */
      PolyUOp *lidx = midx->src[group[run_start]];
      if (run_len > 1) {
        /* CAST to vec pointer: CAST(INDEX(buf, base), vec_N_ptr) */
        PolyDType vec_ptr = poly_dtype_ptr_vec(buf->dtype, run_len);
        lidx = poly_uop1(ctx, POLY_OP_CAST, vec_ptr, lidx, poly_arg_none());
      }
      /* Map original lanes to PTRCAT positions */
      for (int k = run_start; k < run_end; k++) {
        idxs[entries[group[k]].orig_idx] = global_offset + (k - run_start);
        used[group[k]] = true;
      }
      ret_srcs[n_ret++] = lidx;
      global_offset += run_len;
      run_start = run_end;
    }
  }

  /* Verify all lanes mapped */
  for (int i = 0; i < n; i++)
    if (idxs[i] < 0) {
      free(entries);
      free(idxs);
      free(used);
      free(ret_srcs);
      free(group);
      free(group_offsets);
      return NULL;
    }

  /* Build PTRCAT */
  PolyUOp *out = NULL;
  if (n_ret == 1) {
    out = poly_uop1(ctx, POLY_OP_GEP, midx->dtype, ret_srcs[0], poly_arg_int_tuple_local(idxs, n));
    free(entries);
    free(idxs);
    free(used);
    free(ret_srcs);
    free(group);
    free(group_offsets);
    return out;
  }
  PolyDType ptrcat_dt = buf->dtype;
  ptrcat_dt.vcount = (uint16_t)global_offset;
  PolyUOp *ptrcat = poly_uop(ctx, POLY_OP_PTRCAT, ptrcat_dt, ret_srcs, n_ret, poly_arg_none());

  /* Apply GEP remap */
  out = poly_uop1(ctx, POLY_OP_GEP, midx->dtype, ptrcat, poly_arg_int_tuple_local(idxs, n));
  free(entries);
  free(idxs);
  free(used);
  free(ret_srcs);
  free(group);
  free(group_offsets);
  return out;
}

/* GEP after LOAD (tinygrad devectorizer.py:127-128):
 * LOAD(GEP(ptr, arg)) → LOAD(ptr, wider_dtype).gep(arg)
 * Pushes GEP through LOAD so the LOAD reads the full vector. */
static PolyUOp *rule_gep_after_load(PolyCtx *ctx, PolyUOp *ld, const PolyBindings *b) {
  (void)b;
  if (!ld || ld->op != POLY_OP_LOAD || ld->n_src < 1) return NULL;
  PolyUOp *gep = ld->src[0];
  if (!gep || gep->op != POLY_OP_GEP || gep->n_src < 1) return NULL;
  /* Build wider LOAD: dtype = scalar.vec(gep source lane count) */
  int src_count = poly_ptr_lane_count(gep->src[0]->dtype);
  if (src_count <= 1) return NULL;
  PolyDType wider_dt = poly_dtype_vec(poly_dtype_scalar(ld->dtype), src_count);
  PolyUOp *ld_srcs[64];
  int ns = 0;
  ld_srcs[ns++] = gep->src[0];
  for (int i = 1; i < ld->n_src && ns < 64; i++)
    ld_srcs[ns++] = ld->src[i];
  PolyUOp *wider_load = poly_uop(ctx, POLY_OP_LOAD, wider_dt, ld_srcs, ns, ld->arg);
  return poly_uop1(ctx, POLY_OP_GEP, ld->dtype, wider_load, gep->arg);
}

/* GEP on STORE data (tinygrad devectorizer.py:115-121):
 * STORE(GEP(ptr, arg), data) → STORE(ptr.src[0], data.gep(inverted_arg)) */
static PolyUOp *rule_gep_on_store(PolyCtx *ctx, PolyUOp *sto, const PolyBindings *b) {
  (void)b;
  if (!sto || sto->op != POLY_OP_STORE || sto->n_src < 2) return NULL;
  PolyUOp *gep = sto->src[0];
  if (!gep || gep->op != POLY_OP_GEP || gep->n_src < 1) return NULL;
  if (gep->arg.kind != POLY_ARG_INT_TUPLE || gep->arg.int_tuple.n <= 0) return NULL;
  int gn = gep->arg.int_tuple.n;
  /* Tinygrad's gep_on_store sorts by the original GEP offsets and keeps the
   * source positions, which handles arbitrary offsets like (4,) on tail stores. */
  typedef struct {
    int64_t key;
    int64_t pos;
  } GEPInvPair;
  GEPInvPair *pairs = malloc((size_t)gn * sizeof(*pairs));
  int64_t *new_arg = malloc((size_t)gn * sizeof(*new_arg));
  if (!pairs || !new_arg) {
    free(pairs);
    free(new_arg);
    return NULL;
  }
  for (int i = 0; i < gn; i++) {
    pairs[i].key = gep->arg.int_tuple.vals[i];
    pairs[i].pos = i;
  }
  for (int i = 0; i < gn; i++) {
    for (int j = i + 1; j < gn; j++) {
      if (pairs[j].key < pairs[i].key) {
        GEPInvPair tmp = pairs[i];
        pairs[i] = pairs[j];
        pairs[j] = tmp;
      }
    }
  }
  for (int i = 0; i < gn; i++)
    new_arg[i] = pairs[i].pos;
  PolyUOp *st_data = sto->src[1];
  PolyUOp *reordered_data = poly_uop1(
      ctx, POLY_OP_GEP, dtype_for_gep_result(st_data->dtype, gn), st_data,
      poly_arg_int_tuple_local(new_arg, gn)
  );
  free(pairs);
  free(new_arg);
  PolyUOp *st_srcs[64];
  int ns = 0;
  st_srcs[ns++] = gep->src[0];
  st_srcs[ns++] = reordered_data;
  for (int i = 2; i < sto->n_src && ns < 64; i++)
    st_srcs[ns++] = sto->src[i];
  return poly_uop(ctx, POLY_OP_STORE, sto->dtype, st_srcs, ns, sto->arg);
}

/* PTRCAT after LOAD (tinygrad devectorizer.py:131-133):
 * LOAD(PTRCAT(ptr0, ptr1, ...)) → VCAT(LOAD(ptr0), LOAD(ptr1), ...)
 * Each LOAD reads the vector width of its pointer source. */
static PolyUOp *rule_ptrcat_after_load(PolyCtx *ctx, PolyUOp *ld, const PolyBindings *b) {
  (void)b;
  if (!ld || ld->op != POLY_OP_LOAD || ld->n_src < 1) return NULL;
  PolyUOp *cat = ld->src[0];
  if (!cat || cat->op != POLY_OP_PTRCAT || cat->n_src <= 0) return NULL;
  int total_count = 0;
  PolyUOp **loads = calloc((size_t)cat->n_src, sizeof(*loads));
  if (!loads) return NULL;
  int nl = 0;
  PolyDType sdt = poly_dtype_scalar(ld->dtype);
  for (int i = 0; i < cat->n_src; i++) {
    PolyUOp *ptr = cat->src[i];
    int ptr_count = poly_ptr_lane_count(ptr->dtype);
    if (ptr_count <= 0) ptr_count = 1;
    PolyDType ld_dt = (ptr_count > 1) ? poly_dtype_vec(sdt, ptr_count) : sdt;
    PolyUOp *ld_srcs[64];
    int ns = 0;
    ld_srcs[ns++] = ptr;
    for (int j = 1; j < ld->n_src && ns < 64; j++)
      ld_srcs[ns++] = ld->src[j];
    loads[nl++] = poly_uop(ctx, POLY_OP_LOAD, ld_dt, ld_srcs, ns, ld->arg);
    total_count += ptr_count;
  }
  PolyDType cat_dt = poly_dtype_vec(sdt, total_count);
  PolyUOp *ret = poly_uop(ctx, POLY_OP_VCAT, cat_dt, loads, nl, poly_arg_none());
  free(loads);
  return ret;
}

/* PTRCAT after STORE (tinygrad devectorizer.py:106-113):
 * STORE(PTRCAT(ptr0, ptr1, ...), data) → GROUP(STORE(ptr0, GEP(data,0..n0)), ...) */
static PolyUOp *rule_ptrcat_after_store(PolyCtx *ctx, PolyUOp *sto, const PolyBindings *b) {
  (void)b;
  if (!sto || sto->op != POLY_OP_STORE || sto->n_src < 2) return NULL;
  PolyUOp *cat = sto->src[0];
  if (!cat || cat->op != POLY_OP_PTRCAT || cat->n_src <= 0) return NULL;
  PolyUOp *data = sto->src[1];
  int offset = 0;
  PolyUOp **stores = calloc((size_t)cat->n_src, sizeof(*stores));
  if (!stores) return NULL;
  int ns_out = 0;
  for (int i = 0; i < cat->n_src; i++) {
    PolyUOp *ptr = cat->src[i];
    int ptr_count = poly_ptr_lane_count(ptr->dtype);
    if (ptr_count <= 0) ptr_count = 1;
    /* GEP to extract this slice of data */
    int64_t gep_args[128];
    for (int j = 0; j < ptr_count; j++)
      gep_args[j] = offset + j;
    PolyDType slice_dt = dtype_for_gep_result(data->dtype, ptr_count);
    PolyUOp *slice =
        poly_uop1(ctx, POLY_OP_GEP, slice_dt, data, poly_arg_int_tuple_local(gep_args, ptr_count));
    PolyUOp *st_srcs[64];
    int ns = 0;
    st_srcs[ns++] = ptr;
    st_srcs[ns++] = slice;
    for (int j = 2; j < sto->n_src && ns < 64; j++)
      st_srcs[ns++] = sto->src[j];
    stores[ns_out++] = poly_uop(ctx, POLY_OP_STORE, sto->dtype, st_srcs, ns, sto->arg);
    offset += ptr_count;
  }
  PolyUOp *ret = poly_uop(ctx, POLY_OP_GROUP, POLY_VOID, stores, ns_out, poly_arg_none());
  free(stores);
  return ret;
}

static bool codegen_foldable_buffer_dtype(PolyUOp *buf) {
  if (!buf || !buf->dtype.is_ptr) return false;
  /* Pinned tinygrad devectorizer.py:163-175 only vector-folds normal buffers
   * whose base is float32, float16, or FP8. Polygrad has no FP8 vocabulary
   * yet; BF16 and float64 must retain the scalar fallback even on renderers
   * which support float4 memory operations. */
  PolyDType base = buf->dtype;
  base.is_ptr = false;
  base.addrspace = POLY_ADDR_GLOBAL;
  base.vcount = 0;
  base.ptr_size = 0;
  base = poly_dtype_scalar(base);
  return poly_dtype_eq(base, POLY_FLOAT32) || poly_dtype_eq(base, POLY_FLOAT16);
}

/* split_load_store (tinygrad devectorizer.py:140-184):
 * LOAD/STORE(CAST(INDEX, vec_ptr)) → split to hardware-supported widths.
 * Matches correct_load_store pattern: LOAD/STORE with CAST(INDEX) source.
 * For CPU with supports_float4: fold_lengths = [4, 2, 1]. */
static PolyUOp *rule_split_load_store(PolyCtx *ctx, PolyUOp *ls, const PolyBindings *b) {
  (void)b;
  if (!ls) return NULL;
  bool is_load = (ls->op == POLY_OP_LOAD);
  bool is_store = (ls->op == POLY_OP_STORE);
  if (!is_load && !is_store) return NULL;
  if (ls->n_src < 1) return NULL;
  /* Match tinygrad's UPat(Ops.INDEX).cast(): the split width is the pointer
   * vector lane count, not the LOAD/STORE value dtype. Splitting a scalar
   * pointer only because the result is vector-typed recreates LOAD -> VCAT
   * indefinitely for accumulator AFTER graphs. */
  PolyUOp *ptr = ls->src[0];
  PolyUOp *idx = ptr;
  if (ptr && ptr->op == POLY_OP_CAST && ptr->n_src >= 1) idx = ptr->src[0];
  if (!idx || (idx->op != POLY_OP_INDEX && idx->op != POLY_OP_SHRINK)) return NULL;
  if (idx->n_src < 2) return NULL;

  int sz = poly_ptr_lane_count(ptr->dtype);
  if (sz <= 1) return NULL; /* nothing to split */
  PolyUOp *buf = idx->src[0];
  if (!buf) return NULL;

  /* Determine fold lengths based on hardware vector width.
   * g_max_fold_width is set by the pipeline before running this pass.
   * Tinygrad devectorizer.py:161-175:
   *   supports_float4=true  → lengths = [4, 2, 1] (or [8,4,2] for half+AMX)
   *   supports_float4=false → lengths = [1] */
  int fold_lengths[5];
  int n_folds = 0;
  if (buf->dtype.is_ptr && buf->dtype.addrspace == POLY_ADDR_REG) {
    /* tinygrad split_load_store keeps register-backed loads/stores scalar. */
  } else if (!codegen_foldable_buffer_dtype(buf)) {
    /* Non-float normal buffers keep the fallback width [1], matching
     * tinygrad's dtype filter in split_load_store. */
  } else {
    if (g_max_fold_width >= 8) fold_lengths[n_folds++] = 8;
    if (g_max_fold_width >= 4) fold_lengths[n_folds++] = 4;
    if (g_max_fold_width >= 2) fold_lengths[n_folds++] = 2;
  }
  fold_lengths[n_folds++] = 1;

  /* tinygrad devectorizer.py: split_load_store filters candidate vector widths
   * with offset.divides(width). Without that guard the CPU C renderer can emit
   * alignment-unsafe casts like float4* at offsets such as ridx0*6. */
  PolyUOp *offset = idx->src[1];
  int filtered_lengths[5];
  int n_filtered = 0;
  for (int i = 0; i < n_folds; i++) {
    int width = fold_lengths[i];
    if (width == 1 || expr_divides_const_codegen(offset, width))
      filtered_lengths[n_filtered++] = width;
  }
  if (n_filtered == 0) {
    filtered_lengths[n_filtered++] = 1;
  }
  for (int i = 0; i < n_filtered; i++)
    fold_lengths[i] = filtered_lengths[i];
  n_folds = n_filtered;

  /* Split into chunks. tinygrad accumulates this in a Python list, so large
   * upcasted vectors (for example 512 scalar lanes after a fallback split)
   * must not be capped by a fixed scratch array. */
  PolyUOp **ret = calloc((size_t)sz, sizeof(*ret));
  if (!ret) return NULL;
  int n_ret = 0;
  int global_offset = 0;
  PolyDType sdt = is_load ? poly_dtype_scalar(ls->dtype) : poly_dtype_scalar(ls->src[1]->dtype);

  while (global_offset < sz) {
    int fold_length = 1;
    for (int f = 0; f < n_folds; f++) {
      if (global_offset + fold_lengths[f] <= sz) {
        fold_length = fold_lengths[f];
        break;
      }
    }
    /* Build INDEX at (original_offset + global_offset) */
    PolyUOp *off_idx;
    if (global_offset == 0) {
      off_idx = idx;
    } else {
      PolyUOp *off_const = poly_uop0(
          ctx, POLY_OP_CONST, poly_dtype_scalar(idx->src[1]->dtype), poly_arg_int(global_offset)
      );
      PolyUOp *new_offset =
          poly_uop2(ctx, POLY_OP_ADD, idx->src[1]->dtype, idx->src[1], off_const, poly_arg_none());
      PolyUOp *idx_srcs[64];
      int ins = 0;
      idx_srcs[ins++] = buf;
      idx_srcs[ins++] = new_offset;
      for (int j = 2; j < idx->n_src && ins < 64; j++)
        idx_srcs[ins++] = idx->src[j];
      off_idx = poly_uop(ctx, POLY_OP_INDEX, buf->dtype, idx_srcs, ins, idx->arg);
    }
    PolyUOp *src_ptr = off_idx;
    if (fold_length > 1) {
      PolyDType vec_ptr = poly_dtype_ptr_vec(buf->dtype, fold_length);
      src_ptr = poly_uop1(ctx, POLY_OP_CAST, vec_ptr, off_idx, poly_arg_none());
    }
    if (is_load) {
      PolyDType ld_dt = (fold_length > 1) ? poly_dtype_vec(sdt, fold_length) : sdt;
      PolyUOp *ld_srcs[64];
      int lns = 0;
      ld_srcs[lns++] = src_ptr;
      for (int j = 1; j < ls->n_src && lns < 64; j++)
        ld_srcs[lns++] = ls->src[j];
      ret[n_ret++] = poly_uop(ctx, POLY_OP_LOAD, ld_dt, ld_srcs, lns, ls->arg);
    } else {
      int64_t gep_args[128];
      for (int j = 0; j < fold_length; j++)
        gep_args[j] = global_offset + j;
      PolyDType slice_dt = dtype_for_gep_result(ls->src[1]->dtype, fold_length);
      PolyUOp *slice = poly_uop1(
          ctx, POLY_OP_GEP, slice_dt, ls->src[1], poly_arg_int_tuple_local(gep_args, fold_length)
      );
      PolyUOp *st_srcs[64];
      int sns = 0;
      st_srcs[sns++] = src_ptr;
      st_srcs[sns++] = slice;
      for (int j = 2; j < ls->n_src && sns < 64; j++)
        st_srcs[sns++] = ls->src[j];
      ret[n_ret++] = poly_uop(ctx, POLY_OP_STORE, ls->dtype, st_srcs, sns, ls->arg);
    }
    global_offset += fold_length;
  }
  if (n_ret <= 1) {
    free(ret);
    return NULL; /* no split needed */
  }
  PolyUOp *out = NULL;
  if (is_load) {
    PolyDType cat_dt = poly_dtype_vec(sdt, sz);
    out = poly_uop(ctx, POLY_OP_VCAT, cat_dt, ret, n_ret, poly_arg_none());
  } else {
    out = poly_uop(ctx, POLY_OP_GROUP, POLY_VOID, ret, n_ret, poly_arg_none());
  }
  free(ret);
  return out;
}

static _Thread_local PolyPatternMatcher *g_pm_load_store_folding = NULL;
static PolyPatternMatcher *poly_pm_load_store_folding(void) {
  if (g_pm_load_store_folding) return g_pm_load_store_folding;
  PolyOpSet stack_or_vectorize = {{0, 0}};
  stack_or_vectorize = poly_opset_add(stack_or_vectorize, POLY_OP_STACK);
  stack_or_vectorize = poly_opset_add(stack_or_vectorize, POLY_OP_VECTORIZE);
  PolyNamedRule rules[] = {
      /* expand_index: INDEX(VECTORIZE(buf), vec) → VECTORIZE(INDEX(buf, gep(vec,i)), ...) */
      {poly_pat_op(POLY_OP_INDEX, NULL, 0, "idx"), rule_expand_index, "expand_index"},
      /* fold_expanded_index: STACK(INDEX, INDEX, ...) → PTRCAT(...).gep(remap) */
      {poly_pat_ops(stack_or_vectorize, NULL, 0, "midx"), rule_fold_expanded_index,
       "fold_expanded_index"},
      /* GEP after LOAD: LOAD(GEP(ptr)) → LOAD(ptr).gep(arg) */
      {poly_pat_allow_any_len(poly_pat_op(POLY_OP_LOAD, NULL, 0, "ld")), rule_gep_after_load,
       "gep_after_load"},
      /* GEP on STORE: STORE(GEP(ptr), data) → STORE(ptr, data.gep(inv)) */
      {poly_pat_allow_any_len(poly_pat_op(POLY_OP_STORE, NULL, 0, "sto")), rule_gep_on_store,
       "gep_on_store"},
      /* PTRCAT after LOAD: LOAD(PTRCAT) → VCAT(LOAD, LOAD, ...) */
      {poly_pat_allow_any_len(poly_pat_op(POLY_OP_LOAD, NULL, 0, "ld")), rule_ptrcat_after_load,
       "ptrcat_after_load"},
      /* PTRCAT after STORE: STORE(PTRCAT, data) → GROUP(STORE, STORE, ...) */
      {poly_pat_allow_any_len(poly_pat_op(POLY_OP_STORE, NULL, 0, "sto")), rule_ptrcat_after_store,
       "ptrcat_after_store"},
      /* correct_load_store: split oversized LOAD/STORE(CAST(INDEX)) */
      {poly_pat_allow_any_len(poly_pat_op(POLY_OP_LOAD, NULL, 0, "ls")), rule_split_load_store,
       "split_load_store_load"},
      {poly_pat_allow_any_len(poly_pat_op(POLY_OP_STORE, NULL, 0, "ls")), rule_split_load_store,
       "split_load_store_store"},
  };
  g_pm_load_store_folding =
      poly_pm_thread_cache(poly_pm_new_named(rules, (int)(sizeof(rules) / sizeof(rules[0]))));
  return g_pm_load_store_folding;
}

/* pm_split_ends (tinygrad codegen/late/linearizer.py:88-96) */
/*
 * After range substitution (pm_split_ranges, apply_opts_basic), an END's
 * ended_ranges src[1:] may contain arithmetic expressions instead of RANGEs.
 * This pass walks backward from each src to find all actual RANGEs, then
 * rebuilds the END as a nested chain: END(END(...END(store, r_last)..., r1), r0).
 */

static void collect_ranges_backward(PolyCtx *ctx, PolyUOp *u, PolyUOp **out, int *n, int cap) {
  if (!u || *n >= cap) return;
  if (u->op == POLY_OP_RANGE) {
    /* Deduplicate */
    for (int i = 0; i < *n; i++)
      if (out[i] == u) return;
    out[(*n)++] = u;
    return;
  }
  for (int i = 0; i < u->n_src; i++)
    collect_ranges_backward(ctx, u->src[i], out, n, cap);
}

static int cmp_range_axis_id(const void *a, const void *b) {
  const PolyUOp *ra = *(const PolyUOp *const *)a;
  const PolyUOp *rb = *(const PolyUOp *const *)b;
  int64_t ia = poly_range_axis_id(ra->arg);
  int64_t ib = poly_range_axis_id(rb->arg);
  return (ia > ib) - (ia < ib);
}

static PolyUOp *rule_split_ends(PolyCtx *ctx, PolyUOp *end, const PolyBindings *b) {
  (void)b;
  if (end->op != POLY_OP_END) return NULL;
  if (end->n_src == 0) return NULL;
  if (end->n_src == 1) return end->src[0];

  /* Check if any ended source is not a RANGE (broken by substitution) */
  bool needs_split = false;
  for (int j = 1; j < end->n_src; j++) {
    if (end->src[j]->op != POLY_OP_RANGE) {
      needs_split = true;
      break;
    }
  }
  /* Also split multi-RANGE ENDs into nested single-RANGE ENDs */
  if (!needs_split && end->n_src <= 2) return NULL;

  /* Collect all RANGE UOps reachable from src[1:] */
  PolyUOp *ranges[64];
  int n_ranges = 0;
  for (int j = 1; j < end->n_src; j++)
    collect_ranges_backward(ctx, end->src[j], ranges, &n_ranges, 64);

  /* tinygrad do_split_ends returns src[0] when no ranges remain reachable from
   * the END sources. Keeping an orphan END produces renderer depth mismatch. */
  if (n_ranges == 0) return end->src[0];

  /* Sort by axis_id (ascending = outermost first) */
  qsort(ranges, (size_t)n_ranges, sizeof(PolyUOp *), cmp_range_axis_id);

  /* Build nested END chain: END(END(...END(store, r_last)..., r1), r0)
   * Innermost (highest axis_id) is deepest in the chain. */
  PolyUOp *ret = end->src[0]; /* store or inner END */
  for (int i = n_ranges - 1; i >= 0; i--) {
    PolyUOp *end_srcs[2] = {ret, ranges[i]};
    ret = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_srcs, 2, poly_arg_none());
  }
  return ret;
}

static bool end_has_same_range_tuple(PolyUOp *a, PolyUOp *b) {
  if (!a || !b || a->op != POLY_OP_END || b->op != POLY_OP_END) return false;
  if (a->n_src < 2 || a->n_src != b->n_src) return false;
  for (int i = 1; i < a->n_src; i++) {
    if (!a->src[i] || a->src[i]->op != POLY_OP_RANGE) return false;
    if (a->src[i] != b->src[i]) return false;
  }
  return true;
}

static PolyUOp *rule_merge_sibling_ends(PolyCtx *ctx, PolyUOp *sink, const PolyBindings *b) {
  (void)b;
  if (!sink || sink->op != POLY_OP_SINK || sink->n_src < 2) return NULL;
  bool *used = calloc((size_t)sink->n_src, sizeof(bool));
  PolyUOp **new_srcs = malloc((size_t)sink->n_src * sizeof(PolyUOp *));
  if (!used || !new_srcs) {
    free(used);
    free(new_srcs);
    return NULL;
  }

  bool changed = false;
  int n_new = 0;
  for (int i = 0; i < sink->n_src; i++) {
    if (used[i]) continue;
    PolyUOp *end = sink->src[i];
    if (!end || end->op != POLY_OP_END || end->n_src < 2) {
      new_srcs[n_new++] = end;
      continue;
    }

    int n_group = 1;
    for (int j = i + 1; j < sink->n_src; j++)
      if (!used[j] && end_has_same_range_tuple(end, sink->src[j])) n_group++;

    if (n_group <= 1) {
      new_srcs[n_new++] = end;
      continue;
    }

    PolyUOp **group_srcs = malloc((size_t)n_group * sizeof(PolyUOp *));
    PolyUOp **end_srcs = malloc((size_t)end->n_src * sizeof(PolyUOp *));
    if (!group_srcs || !end_srcs) {
      free(group_srcs);
      free(end_srcs);
      free(used);
      free(new_srcs);
      return NULL;
    }

    int at = 0;
    group_srcs[at++] = end->src[0];
    used[i] = true;
    for (int j = i + 1; j < sink->n_src; j++) {
      if (used[j] || !end_has_same_range_tuple(end, sink->src[j])) continue;
      group_srcs[at++] = sink->src[j]->src[0];
      used[j] = true;
    }

    PolyUOp *group = poly_uop(ctx, POLY_OP_GROUP, POLY_VOID, group_srcs, n_group, poly_arg_none());
    end_srcs[0] = group;
    for (int r = 1; r < end->n_src; r++)
      end_srcs[r] = end->src[r];
    new_srcs[n_new++] =
        poly_uop(ctx, POLY_OP_END, POLY_VOID, end_srcs, end->n_src, poly_arg_none());
    changed = true;
    free(group_srcs);
    free(end_srcs);
  }

  PolyUOp *ret = NULL;
  if (changed) {
    if (sink->tag != 0 || sink->tag_arg.kind != POLY_ARG_NONE)
      ret = poly_uop_tagged_arg(
          ctx, POLY_OP_SINK, POLY_VOID, new_srcs, n_new, sink->arg, sink->tag, sink->tag_arg
      );
    else
      ret = poly_uop(ctx, POLY_OP_SINK, POLY_VOID, new_srcs, n_new, sink->arg);
  }
  free(used);
  free(new_srcs);
  return ret;
}

static _Thread_local PolyPatternMatcher *g_pm_split_ends = NULL;
static PolyPatternMatcher *poly_pm_split_ends(void) {
  if (g_pm_split_ends) return g_pm_split_ends;
  PolyRule rules[] = {
      {poly_pat_op(POLY_OP_SINK, NULL, 0, "sink"), rule_merge_sibling_ends},
      {poly_pat_allow_any_len(poly_pat_op(POLY_OP_END, NULL, 0, "end")), rule_split_ends},
  };
  g_pm_split_ends =
      poly_pm_thread_cache(poly_pm_new(rules, (int)(sizeof(rules) / sizeof(rules[0]))));
  return g_pm_split_ends;
}

/* Forward declarations for devectorize (defined in pm_render_subset section below) */
static PolyUOp *rule_vectorize_single(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b);
static PolyUOp *lane_or_gep(PolyCtx *ctx, PolyUOp *src, int lane);

/* tinygrad uses ptr.count for normal ptr-to-vector width. Polygrad still has a
 * few older late-pass sites that carried that width in ptr.vcount instead.
 * Treat count as canonical and vcount as a compatibility fallback until those
 * remaining sites are normalized. */
static int poly_ptr_lane_count(PolyDType dt) {
  if (!dt.is_ptr) return dt.count;
  if (dt.count > 1) return dt.count;
  if (dt.vcount > 1) return dt.vcount;
  return 1;
}

static PolyDType poly_dtype_ptr_vec(PolyDType ptr, int lanes) {
  if (!ptr.is_ptr || lanes <= 1) return ptr;
  PolyDType base = ptr;
  base.is_ptr = false;
  base.addrspace = POLY_ADDR_GLOBAL;
  base.ptr_size = 0;
  base.vcount = 1;
  base = poly_dtype_vec(poly_dtype_scalar(base), lanes);
  return poly_dtype_ptr(base, ptr.ptr_size, ptr.addrspace);
}

static bool poly_vector_compare_op(PolyOps op) {
  return op == POLY_OP_CMPLT || op == POLY_OP_CMPEQ || op == POLY_OP_CMPNE;
}

static bool poly_caps_packs_vector_alu(PolyRendererCaps caps, PolyOps op, PolyDType dt) {
  if (!caps.has_simd_float || dt.is_ptr || dt.count <= 1) return false;
  PolyDType s = poly_dtype_scalar(dt);
  if (!poly_dtype_is_float(s) || s.bitsize != 32 || dt.count != 4 || caps.max_vec_width < 4)
    return false;

  switch (op) {
  case POLY_OP_NEG:
  case POLY_OP_SQRT:
  case POLY_OP_ADD:
  case POLY_OP_SUB:
  case POLY_OP_MUL:
  case POLY_OP_FDIV:
  case POLY_OP_MAX:
  case POLY_OP_MULACC:
  case POLY_OP_CMPLT:
  case POLY_OP_CMPEQ:
  case POLY_OP_CMPNE:
    return true;
  default:
    return false;
  }
}

static bool poly_caps_packs_vector_compare(PolyRendererCaps caps, PolyUOp *u) {
  if (!u || !poly_vector_compare_op(u->op) || u->n_src < 2) return false;
  if (!caps.has_simd_float || u->dtype.count != 4 || caps.max_vec_width < 4) return false;
  PolyDType a = poly_dtype_scalar(u->src[0]->dtype);
  PolyDType b = poly_dtype_scalar(u->src[1]->dtype);
  return poly_dtype_is_float(a) && poly_dtype_is_float(b) && a.bitsize == 32 && b.bitsize == 32;
}

static bool poly_caps_packs_vector_where(PolyRendererCaps caps, PolyUOp *w) {
  if (!w || w->op != POLY_OP_WHERE || w->n_src < 3) return false;
  if (!caps.has_simd_float || w->dtype.is_ptr || w->dtype.count != 4 || caps.max_vec_width < 4)
    return false;
  PolyDType s = poly_dtype_scalar(w->dtype);
  if (!poly_dtype_is_float(s) || s.bitsize != 32) return false;
  return poly_caps_packs_vector_compare(caps, w->src[0]);
}

static bool poly_is_reg_or_local_buffer_codegen(PolyUOp *u) {
  if (!u) return false;
  if (u->op == POLY_OP_DEFINE_REG || u->op == POLY_OP_DEFINE_LOCAL) return true;
  return u->op == POLY_OP_BUFFER && u->dtype.is_ptr &&
         (u->dtype.addrspace == POLY_ADDR_REG || u->dtype.addrspace == POLY_ADDR_LOCAL);
}

/* devectorize (tinygrad codegen/late/devectorizer.py) */
/*
 * Scatters vectorized ALU/CAST/BITCAST ops into per-element scalar ops
 * wrapped in VECTORIZE. Renderers that directly emit a native vector value can
 * preserve a small, explicit subset through caps; vector LOAD/STORE survive and
 * are handled by load_store_folding.
 *
 * OP(vec_a, vec_b) → VECTORIZE(OP(GEP(a,0), GEP(b,0)), OP(GEP(a,1), GEP(b,1)), ...)
 */

static PolyUOp *rule_no_vectorized_alu(PolyCtx *ctx, PolyUOp *alu, const PolyBindings *b) {
  (void)b;
  if (alu->dtype.count <= 1) return NULL;
  /* Skip pointer types: vec_ptr CASTs from fold_vectorized_index are pointer
   * type conversions, not value ALU. tinygrad: PtrDType.vcount returns 1,
   * so no_vectorized_alu naturally skips them. */
  if (alu->dtype.is_ptr) return NULL;
  if (alu->op == POLY_OP_WHERE) {
    if (poly_caps_packs_vector_where(g_render_caps, alu)) return NULL;
  } else if (poly_vector_compare_op(alu->op)) {
    if (poly_caps_packs_vector_compare(g_render_caps, alu)) return NULL;
  } else if (poly_caps_packs_vector_alu(g_render_caps, alu->op, alu->dtype)) {
    return NULL;
  }
  int lanes = alu->dtype.count;
  PolyDType sdt = poly_dtype_scalar(alu->dtype);
  PolyUOp **elts = calloc((size_t)lanes, sizeof(*elts));
  if (!elts) return NULL;
  for (int i = 0; i < lanes; i++) {
    PolyUOp *srcs[8];
    int ns = alu->n_src;
    if (ns > 8) {
      free(elts);
      return NULL;
    }
    for (int j = 0; j < ns; j++)
      srcs[j] = lane_or_gep(ctx, alu->src[j], i);
    if (ns == 0)
      elts[i] = poly_uop0(ctx, alu->op, sdt, alu->arg);
    else if (ns == 1)
      elts[i] = poly_uop1(ctx, alu->op, sdt, srcs[0], alu->arg);
    else if (ns == 2)
      elts[i] = poly_uop2(ctx, alu->op, sdt, srcs[0], srcs[1], alu->arg);
    else if (ns == 3)
      elts[i] = poly_uop3(ctx, alu->op, sdt, srcs[0], srcs[1], srcs[2], alu->arg);
    else
      elts[i] = poly_uop(ctx, alu->op, sdt, srcs, ns, alu->arg);
  }
  PolyUOp *ret = poly_uop(ctx, POLY_OP_VECTORIZE, alu->dtype, elts, lanes, poly_arg_none());
  free(elts);
  return ret;
}

static PolyUOp *rule_bool_and_to_scalarized_vector(
    PolyCtx *ctx,
    PolyUOp *alu,
    const PolyBindings *b
) {
  if (!alu || alu->op != POLY_OP_AND || !poly_dtype_is_bool(poly_dtype_scalar(alu->dtype)))
    return NULL;
  return rule_no_vectorized_alu(ctx, alu, b);
}

/* pm_move_where_on_load (tinygrad uop/symbolic.py:where_on_load).
 * WHERE(cond, LOAD(INDEX(buf, idx)), CONST(0))
 *   -> LOAD(INDEX(buf, idx), CONST(0), cond)
 * Moves validity to the LOAD gate. Without this, an unconditional LOAD on an
 * out-of-bounds PAD index can fault in a native backend.
 *
 * Guard: only move conditions that are pure index math (no LOAD in subtree).
 * PAD bounds checks (CMPLT/AND of RANGE/CONST) pass; data-dependent conditions
 * like WHERE(x>0, x, 0) are rejected because cond contains LOAD. */
static bool uop_tree_contains_load(PolyUOp *u) {
  if (!u) return false;
  if (u->op == POLY_OP_LOAD) return true;
  for (int i = 0; i < u->n_src; i++)
    if (uop_tree_contains_load(u->src[i])) return true;
  return false;
}

static bool uop_tree_indexes_buffer(PolyUOp *u, PolyUOp *buf) {
  if (!u || !buf) return false;
  if (u->op == POLY_OP_INDEX && u->n_src >= 1 && u->src[0] == buf) return true;
  for (int i = 0; i < u->n_src; i++)
    if (uop_tree_indexes_buffer(u->src[i], buf)) return true;
  return false;
}

static bool clause_ranges_subset_of_index(
    PolyCtx *ctx,
    PolyUOp *clause,
    PolyUOp **idx_ranges,
    int n_idx_ranges,
    PolyUOpCache *cache
) {
  PolyUOp *clause_ranges[64];
  int n_clause_ranges = poly_uop_ranges_ex(ctx, clause, clause_ranges, 64, cache);
  for (int i = 0; i < n_clause_ranges; i++) {
    if (!uop_ptr_in_list(clause_ranges[i], idx_ranges, n_idx_ranges)) return false;
  }
  return true;
}

static bool clause_indexes_subset_of_index_graph(PolyCtx *ctx, PolyUOp *clause, PolyUOp *idx) {
  PolyUOp *idx_indexes[128];
  int n_idx_indexes = 0;
  int n_idx_topo = 0;
  PolyUOp **idx_topo = poly_toposort(ctx, idx, &n_idx_topo);
  for (int i = 0; i < n_idx_topo && n_idx_indexes < 128; i++) {
    if (idx_topo[i] && idx_topo[i]->op == POLY_OP_INDEX) idx_indexes[n_idx_indexes++] = idx_topo[i];
  }

  int n_clause_topo = 0;
  PolyUOp **clause_topo = poly_toposort(ctx, clause, &n_clause_topo);
  for (int i = 0; i < n_clause_topo; i++) {
    if (!clause_topo[i] || clause_topo[i]->op != POLY_OP_INDEX) continue;
    if (!uop_ptr_in_list(clause_topo[i], idx_indexes, n_idx_indexes)) return false;
  }
  return true;
}

/* Port of tinygrad uop/symbolic.py:where_on_load. Move only the safe AND
 * clauses into LOAD.valid and keep the residual clauses outside as WHERE. */
static PolyUOp *move_where_clauses_into_index(
    PolyCtx *ctx,
    PolyUOp *cond,
    PolyUOp *idx,
    PolyUOp *existing_load_gate,
    bool has_load,
    PolyUOp *zero,
    PolyDType out_dtype,
    PolyArg load_arg
) {
  PolyUOp *where_clauses[128];
  PolyUOp *load_clauses[128];
  PolyUOp *moved[128];
  PolyUOp *keep[128];
  PolyUOp *idx_ranges[64];

  PolyUOp *load_valid =
      existing_load_gate ? existing_load_gate : poly_const_typed(ctx, POLY_BOOL, 1.0);
  int n_where = split_uop_and(cond, where_clauses, 128);
  int n_load = split_uop_and(load_valid, load_clauses, 128);
  PolyUOpCache *cache = poly_uop_cache_new();
  int n_idx_ranges = poly_uop_ranges_ex(ctx, idx, idx_ranges, 64, cache);

  int n_moved = 0;
  int n_keep = 0;
  for (int i = 0; i < n_where; i++) {
    PolyUOp *clause = where_clauses[i];
    if (uop_ptr_in_list(clause, load_clauses, n_load)) continue;
    bool can_move = clause_ranges_subset_of_index(ctx, clause, idx_ranges, n_idx_ranges, cache) &&
                    clause_indexes_subset_of_index_graph(ctx, clause, idx);
    if (can_move)
      moved[n_moved++] = clause;
    else
      keep[n_keep++] = clause;
  }
  poly_uop_cache_destroy(cache);

  /* Match tinygrad exactly: if every original where clause would still remain
   * outside, there is nothing to rewrite. Clauses already present in the load
   * gate count as removed even when no new clause moves. */
  if (n_keep == n_where) return NULL;

  PolyUOp *new_valid = and_all_clauses(ctx, moved, n_moved, load_valid);
  PolyUOp *base = NULL;
  if (has_load) {
    /* Pinned tinygrad/uop/spec.py:73-77 keeps every INDEX coordinate integer;
     * LOAD carries its validity predicate as src[2]. */
    PolyUOp *load_srcs[3] = {idx, zero, new_valid};
    base = poly_uop(ctx, POLY_OP_LOAD, out_dtype, load_srcs, 3, load_arg);
  } else {
    /* Tinygrad keeps pm_move_where_on_load in the weakint index domain:
     *   INDEX(buf, WHERE(valid, idx, Invalid))
     * The late gater moves validity to LOAD/STORE after index-dtype lowering. */
    PolyUOp *invalid = poly_uop0(ctx, POLY_OP_CONST, idx->src[1]->dtype, poly_arg_invalid());
    PolyUOp *index_expr = poly_uop3(
        ctx, POLY_OP_WHERE, idx->src[1]->dtype, new_valid, idx->src[1], invalid, poly_arg_none()
    );
    base = poly_uop2(ctx, POLY_OP_INDEX, idx->dtype, idx->src[0], index_expr, idx->arg);
  }
  if (n_keep == 0) return base;

  PolyUOp *keep_cond = and_all_clauses(ctx, keep, n_keep, NULL);
  return poly_uop3(ctx, POLY_OP_WHERE, out_dtype, keep_cond, base, zero, poly_arg_none());
}

static PolyUOp *rule_where_on_load(PolyCtx *ctx, PolyUOp *w, const PolyBindings *b) {
  (void)b;
  if (w->op != POLY_OP_WHERE || w->n_src != 3) return NULL;
  PolyUOp *cond = w->src[0];
  PolyUOp *true_val = w->src[1];
  PolyUOp *false_val = w->src[2];

  /* Match two patterns:
   * A) WHERE(cond, LOAD(INDEX(buf, idx)), CONST(0))  -- post pm_add_loads
   * B) WHERE(cond, INDEX(buf, idx), CONST(0))         -- pre pm_add_loads (PAD path) */
  PolyUOp *idx = NULL;
  bool has_load = false;
  if (true_val->op == POLY_OP_LOAD && true_val->n_src >= 1) {
    idx = true_val->src[0];
    has_load = true;
  } else if (true_val->op == POLY_OP_INDEX) {
    idx = true_val;
  }
  if (!idx || idx->op != POLY_OP_INDEX || idx->n_src < 2) return NULL;
  if (false_val->op != POLY_OP_CONST) return NULL;
  if (false_val->arg.kind == POLY_ARG_FLOAT && false_val->arg.f != 0.0) return NULL;
  if (false_val->arg.kind == POLY_ARG_INT && false_val->arg.i != 0) return NULL;

  /* Guard: reject value-dependent selects. In Polygrad this pass runs before
   * pm_add_loads, so ReLU-like conditions are still CMPLT(..., INDEX(buf, i))
   * rather than CMPLT(..., LOAD(INDEX(buf, i))). Moving those into the index
   * turns WHERE(x>0, x, 0) into LOAD(x[WHERE(x>0, i, Invalid)]) and loses the
   * false branch. PAD/triu masks use range/index math and still move. */
  if (uop_tree_contains_load(cond) || uop_tree_indexes_buffer(cond, idx->src[0])) return NULL;
  return move_where_clauses_into_index(
      ctx, cond, idx, has_load && true_val->n_src >= 3 ? true_val->src[2] : NULL, has_load,
      false_val, has_load ? true_val->dtype : idx->dtype, has_load ? true_val->arg : poly_arg_none()
  );
}

/* Also handle reversed: WHERE(cond, CONST(0), LOAD(INDEX(buf, idx)))
 * → LOAD(INDEX(buf, idx, NEG(cond))) */
static PolyUOp *rule_where_on_load_rev(PolyCtx *ctx, PolyUOp *w, const PolyBindings *b) {
  (void)b;
  if (w->op != POLY_OP_WHERE || w->n_src != 3) return NULL;
  PolyUOp *cond = w->src[0];
  PolyUOp *true_val = w->src[1];
  PolyUOp *false_val = w->src[2];

  /* Match: WHERE(cond, CONST(0), LOAD(INDEX(buf, idx))) or WHERE(cond, CONST(0), INDEX(buf, idx))
   */
  PolyUOp *idx = NULL;
  bool has_load = false;
  if (false_val->op == POLY_OP_LOAD && false_val->n_src >= 1) {
    idx = false_val->src[0];
    has_load = true;
  } else if (false_val->op == POLY_OP_INDEX) {
    idx = false_val;
  }
  if (!idx || idx->op != POLY_OP_INDEX || idx->n_src < 2) return NULL;
  if (true_val->op != POLY_OP_CONST) return NULL;
  if (true_val->arg.kind == POLY_ARG_FLOAT && true_val->arg.f != 0.0) return NULL;
  if (true_val->arg.kind == POLY_ARG_INT && true_val->arg.i != 0) return NULL;

  if (uop_tree_contains_load(cond) || uop_tree_indexes_buffer(cond, idx->src[0])) return NULL;
  PolyUOp *neg_cond = poly_uop2(
      ctx, POLY_OP_CMPNE, cond->dtype, cond, poly_const_like_bool(ctx, cond, true), poly_arg_none()
  );
  return move_where_clauses_into_index(
      ctx, neg_cond, idx, has_load && false_val->n_src >= 3 ? false_val->src[2] : NULL, has_load,
      true_val, has_load ? false_val->dtype : idx->dtype,
      has_load ? false_val->arg : poly_arg_none()
  );
}

static _Thread_local PolyPatternMatcher *g_pm_move_where_on_load = NULL;
static PolyPatternMatcher *poly_pm_move_where_on_load(void) {
  if (g_pm_move_where_on_load) return g_pm_move_where_on_load;
  PolyRule rules[] = {
      {poly_pat_op(POLY_OP_WHERE, NULL, 0, "w"), rule_where_on_load},
      {poly_pat_op(POLY_OP_WHERE, NULL, 0, "w"), rule_where_on_load_rev},
  };
  g_pm_move_where_on_load = poly_pm_thread_cache(poly_pm_new(rules, 2));
  return g_pm_move_where_on_load;
}

/* no_vectorized_buf: scalarize register/local buffers with vector dtype.
 * Tinygrad devectorizer.py:241-242. */
static PolyUOp *rule_no_vectorized_buf(PolyCtx *ctx, PolyUOp *buf, const PolyBindings *b) {
  (void)b;
  PolyDType dt = buf->dtype;
  if (!dt.is_ptr || dt.count <= 1) return NULL;
  PolyDType base = poly_dtype_scalar(dt);
  PolyDType scalar_ptr = poly_dtype_ptr(base, dt.ptr_size * dt.count, dt.addrspace);
  PolyUOp *scalar_buf = poly_uop(ctx, buf->op, scalar_ptr, buf->src, buf->n_src, buf->arg);
  return poly_uop1(ctx, POLY_OP_CAST, dt, scalar_buf, poly_arg_none());
}

/* CAST-after-AFTER canonicalization (tinygrad devectorizer.py:269).
 * AFTER(CAST(x), deps) → CAST(AFTER(x, deps)) */
static PolyUOp *rule_cast_after_after(PolyCtx *ctx, PolyUOp *after, const PolyBindings *b) {
  (void)b;
  if (after->op != POLY_OP_AFTER || after->n_src < 1) return NULL;
  if (after->src[0]->op != POLY_OP_CAST || after->src[0]->n_src < 1) return NULL;
  PolyUOp *cast = after->src[0];
  PolyUOp *new_srcs[128];
  new_srcs[0] = cast->src[0];
  int nd = after->n_src - 1;
  for (int i = 0; i < nd && i < 127; i++)
    new_srcs[1 + i] = after->src[1 + i];
  PolyUOp *new_after =
      poly_uop(ctx, POLY_OP_AFTER, cast->src[0]->dtype, new_srcs, 1 + nd, poly_arg_none());
  return poly_uop1(ctx, POLY_OP_CAST, cast->dtype, new_after, poly_arg_none());
}

/* no_vectorized_index: adjust INDEX on scalarized registers.
 * Tinygrad devectorizer.py:244-256.
 * Match: INDEX( CAST(buf_or_after) , idx ) where CAST has vec dtype, buf is scalar.
 * Output: INDEX( VECTORIZE(buf*count), VECTORIZE(idx*count+0, ..., idx*count+(count-1)) ) */
static PolyUOp *rule_no_vectorized_index(PolyCtx *ctx, PolyUOp *idx_uop, const PolyBindings *b) {
  (void)b;
  if (idx_uop->op != POLY_OP_INDEX || idx_uop->n_src < 2) return NULL;
  PolyUOp *src0 = idx_uop->src[0];
  PolyUOp *bcast = NULL;
  PolyUOp *cast_node = src0;
  if (cast_node->op != POLY_OP_CAST) {
    if ((src0->op != POLY_OP_VECTORIZE && src0->op != POLY_OP_GEP) || src0->n_src < 1) return NULL;
    bcast = src0;
    cast_node = src0->src[0];
  }
  if (cast_node->op != POLY_OP_CAST || cast_node->n_src < 1) return NULL;
  PolyUOp *buf = cast_node->src[0];
  PolyUOp *def = buf;
  if (def->op == POLY_OP_AFTER && def->n_src > 0) def = def->src[0];
  if (!poly_is_reg_or_local_buffer_codegen(def)) return NULL;
  int count = cast_node->dtype.count;
  if (count <= 1 || def->dtype.count > 1) return NULL;
  int groups = cast_node->dtype.is_ptr && cast_node->dtype.vcount > 1 ? cast_node->dtype.vcount : 1;

  PolyUOp *orig_idx = idx_uop->src[1];
  int64_t n_pairs64 = 0;
  if (bcast && bcast->op == POLY_OP_GEP && bcast->arg.kind == POLY_ARG_INT_TUPLE) {
    n_pairs64 = (int64_t)groups * bcast->arg.int_tuple.n;
  } else if (bcast) {
    int bcast_v =
        (bcast->dtype.is_ptr && bcast->dtype.vcount > 1) ? bcast->dtype.vcount : bcast->dtype.count;
    n_pairs64 = (int64_t)count * bcast_v;
  } else {
    n_pairs64 = count;
  }
  if (n_pairs64 <= 0 || n_pairs64 > INT32_MAX) return NULL;

  int n_pairs = (int)n_pairs64;
  int *idx_lanes = calloc((size_t)n_pairs, sizeof(*idx_lanes));
  int *offsets = calloc((size_t)n_pairs, sizeof(*offsets));
  if (!idx_lanes || !offsets) {
    free(idx_lanes);
    free(offsets);
    return NULL;
  }
  int p_pair = 0;

  if (bcast && bcast->op == POLY_OP_GEP && bcast->arg.kind == POLY_ARG_INT_TUPLE) {
    int n_gep = bcast->arg.int_tuple.n;
    for (int g = 0; g < groups; g++) {
      for (int k = 0; k < n_gep; k++) {
        idx_lanes[p_pair] = k;
        offsets[p_pair] = g + (int)bcast->arg.int_tuple.vals[k];
        p_pair++;
      }
    }
  } else if (bcast) {
    int bcast_v =
        (bcast->dtype.is_ptr && bcast->dtype.vcount > 1) ? bcast->dtype.vcount : bcast->dtype.count;
    for (int c = 0; c < count; c++) {
      for (int j = 0; j < bcast_v; j++) {
        idx_lanes[p_pair] = j;
        offsets[p_pair] = c;
        p_pair++;
      }
    }
  } else {
    for (int c = 0; c < count; c++) {
      idx_lanes[p_pair] = 0;
      offsets[p_pair] = c;
      p_pair++;
    }
  }
  n_pairs = p_pair;
  if (n_pairs <= 0) {
    free(idx_lanes);
    free(offsets);
    return NULL;
  }

  /* Build real vector dtypes, not just count-mutated structs. PolyDType.bitsize
   * is total vector width, so mutating count alone turns int32.vec(16) into a
   * bogus 2-bit scalar after poly_dtype_scalar(). That aliases register lanes
   * in upcasted reductions. */
  PolyDType vec_buf_dt = dtype_for_vectorize_result(buf->dtype, n_pairs);
  PolyDType vec_idx_dt = poly_dtype_vec(poly_dtype_scalar(orig_idx->dtype), n_pairs);

  PolyUOp **bsrcs = calloc((size_t)n_pairs, sizeof(*bsrcs));
  PolyUOp **isrcs = calloc((size_t)n_pairs, sizeof(*isrcs));
  PolyUOp **csrcs = calloc((size_t)n_pairs, sizeof(*csrcs));
  PolyUOp **osrcs = calloc((size_t)n_pairs, sizeof(*osrcs));
  if (!bsrcs || !isrcs || !csrcs || !osrcs) {
    free(idx_lanes);
    free(offsets);
    free(bsrcs);
    free(isrcs);
    free(csrcs);
    free(osrcs);
    return NULL;
  }
  for (int i = 0; i < n_pairs; i++)
    bsrcs[i] = buf;
  PolyUOp *bcast_buf =
      poly_uop(ctx, POLY_OP_VECTORIZE, vec_buf_dt, bsrcs, n_pairs, poly_arg_none());

  /* tinygrad:
   *   buf.broadcast(len(pairs)).index(idx.gep(idx_lanes)*count + const(offsets), ptr=True) */
  PolyUOp *cnt = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(count));
  for (int i = 0; i < n_pairs; i++) {
    isrcs[i] = scalarize_lane_expr(ctx, orig_idx, idx_lanes[i]);
    if (!isrcs[i]) {
      free(idx_lanes);
      free(offsets);
      free(bsrcs);
      free(isrcs);
      free(csrcs);
      free(osrcs);
      return NULL;
    }
    csrcs[i] = cnt;
    osrcs[i] = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(offsets[i]));
  }
  PolyUOp *idx_v = poly_uop(ctx, POLY_OP_VECTORIZE, vec_idx_dt, isrcs, n_pairs, poly_arg_none());
  PolyUOp *cnt_v = poly_uop(ctx, POLY_OP_VECTORIZE, vec_idx_dt, csrcs, n_pairs, poly_arg_none());
  PolyUOp *off_v = poly_uop(ctx, POLY_OP_VECTORIZE, vec_idx_dt, osrcs, n_pairs, poly_arg_none());
  PolyUOp *scaled = poly_uop2(ctx, POLY_OP_MUL, vec_idx_dt, idx_v, cnt_v, poly_arg_none());
  PolyUOp *final_idx = poly_uop2(ctx, POLY_OP_ADD, vec_idx_dt, scaled, off_v, poly_arg_none());
  PolyUOp *ret =
      poly_uop2(ctx, POLY_OP_INDEX, idx_uop->dtype, bcast_buf, final_idx, poly_arg_none());

  free(bsrcs);
  free(isrcs);
  free(csrcs);
  free(osrcs);
  free(idx_lanes);
  free(offsets);
  return ret;
}

static _Thread_local PolyPatternMatcher *g_pm_devectorize = NULL;
static PolyPatternMatcher *poly_pm_devectorize(void) {
  if (g_pm_devectorize) return g_pm_devectorize;

  PolyRule rules[80];
  int n = 0;

  /* 0. CAST-after-AFTER canonicalization (must fire before index rules) */
  rules[n++] = (PolyRule
  ){poly_pat_allow_any_len(poly_pat_op(POLY_OP_AFTER, NULL, 0, "a")), rule_cast_after_after};

  /* 1. Scatter ALL elementwise ops (ALU + CAST + BITCAST) from vec to scalar */
  rules[n++] = (PolyRule
  ){poly_pat_allow_any_len(poly_pat_ops(POLY_GROUP_ELEMENTWISE, NULL, 0, "alu")),
    rule_no_vectorized_alu};

  /* 2. Scalarize register/local buffers with vector dtype */
  PolyOpSet buf_set = {{0, 0}};
  buf_set = poly_opset_add(buf_set, POLY_OP_DEFINE_REG);
  buf_set = poly_opset_add(buf_set, POLY_OP_DEFINE_LOCAL);
  buf_set = poly_opset_add(buf_set, POLY_OP_BUFFER);
  rules[n++] = (PolyRule){poly_pat_ops(buf_set, NULL, 0, "buf"), rule_no_vectorized_buf};

  /* 3. Adjust INDEX on scalarized registers */
  rules[n++] = (PolyRule
  ){poly_pat_allow_any_len(poly_pat_op(POLY_OP_INDEX, NULL, 0, "idx")), rule_no_vectorized_index};

  /* 4. VECTORIZE(single) → unwrap */
  rules[n++] = (PolyRule){poly_pat_op(POLY_OP_VECTORIZE, NULL, 0, "u"), rule_vectorize_single};

  g_pm_devectorize = poly_pm_thread_cache(poly_pm_new(rules, n));
  return g_pm_devectorize;
}

/* pm_render subset (constants + vector WHERE scalarization) */

static PolyUOp *rule_render_vector_const(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b) {
  (void)b;
  if (!u || u->op != POLY_OP_CONST || u->dtype.count <= 1) return NULL;

  /* tinygrad pm_render lowers vector CONST to explicit STACK before final
   * rendering. This lets later GEP-of-constant lanes collapse to scalar
   * constants instead of surviving as extra lane extract nodes. */
  int lanes = u->dtype.count;
  PolyDType sdt = poly_dtype_scalar(u->dtype);
  PolyUOp **elts = calloc((size_t)lanes, sizeof(*elts));
  if (!elts) return NULL;
  for (int i = 0; i < lanes; i++)
    elts[i] = poly_uop0(ctx, POLY_OP_CONST, sdt, u->arg);
  PolyUOp *ret = poly_uop(ctx, POLY_OP_VECTORIZE, u->dtype, elts, lanes, poly_arg_none());
  free(elts);
  return ret;
}

static PolyUOp *rule_render_vconst(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b) {
  (void)b;
  if (u->op != POLY_OP_VCONST) return NULL;
  if (u->n_src > 0) {
    return poly_uop(ctx, POLY_OP_VECTORIZE, u->dtype, u->src, u->n_src, poly_arg_none());
  }
  if (u->arg.kind == POLY_ARG_INT_TUPLE && u->arg.int_tuple.n > 0) {
    int n = u->arg.int_tuple.n;
    PolyUOp **elts = calloc((size_t)n, sizeof(*elts));
    if (!elts) return NULL;
    PolyDType sdt = poly_dtype_scalar(u->dtype);
    for (int i = 0; i < n; i++) {
      elts[i] = poly_uop0(ctx, POLY_OP_CONST, sdt, poly_arg_int(u->arg.int_tuple.vals[i]));
    }
    PolyUOp *ret = poly_uop(ctx, POLY_OP_VECTORIZE, u->dtype, elts, n, poly_arg_none());
    free(elts);
    return ret;
  }
  return NULL;
}

static PolyUOp *rule_vectorize_single(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b) {
  (void)ctx;
  (void)b;
  if (u->op != POLY_OP_VECTORIZE || u->n_src != 1) return NULL;
  return u->src[0];
}

static PolyUOp *rule_vectorize_duplicate_void_effect(
    PolyCtx *ctx,
    PolyUOp *u,
    const PolyBindings *b
) {
  (void)ctx;
  (void)b;
  if (u->op != POLY_OP_VECTORIZE || !poly_dtype_eq(u->dtype, POLY_VOID) || u->n_src <= 1)
    return NULL;
  PolyUOp *first = u->src[0];
  for (int i = 1; i < u->n_src; i++)
    if (u->src[i] != first) return NULL;
  return first;
}

static PolyUOp *make_gep_lane(PolyCtx *ctx, PolyUOp *src, int lane) {
  int64_t idx = lane;
  return poly_uop1(
      ctx, POLY_OP_GEP, dtype_for_gep_result(src->dtype, 1), src, poly_arg_int_tuple_local(&idx, 1)
  );
}

static PolyUOp *rule_render_gep_multi(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b) {
  (void)b;
  if (!u || u->op != POLY_OP_GEP || u->n_src < 1 || u->arg.kind != POLY_ARG_INT_TUPLE) return NULL;
  int n = u->arg.int_tuple.n;
  if (n <= 1) return NULL;

  /* tinygrad's final pm_render lowers GEP(tuple) into STACK(GEP(...), ...).
   * This is intentionally later than gep_pushing: final render is an explicit
   * renderer-facing form, not another symbolic cleanup pass. */
  PolyUOp **elts = calloc((size_t)n, sizeof(*elts));
  if (!elts) return NULL;
  for (int i = 0; i < n; i++) {
    /* tinygrad calls gep.src[0].gep(x) here. UOp.gep(int) shortcuts STACK
     * sources to their selected lane, so preserve that renderer-facing shape
     * instead of creating extra GEP nodes over a STACK. */
    elts[i] = lane_or_gep(ctx, u->src[0], (int)u->arg.int_tuple.vals[i]);
  }
  PolyUOp *ret = poly_uop(ctx, POLY_OP_VECTORIZE, u->dtype, elts, n, poly_arg_none());
  free(elts);
  return ret;
}

static PolyUOp *lane_or_gep(PolyCtx *ctx, PolyUOp *src, int lane) {
  if (src->dtype.count <= 1) return src;
  if (src->op == POLY_OP_VECTORIZE && src->n_src > lane) return src->src[lane];
  if (src->op == POLY_OP_VCAT) {
    PolyUOp *lane_src = scalarize_lane_expr(ctx, src, lane);
    if (lane_src) return lane_src;
  }
  return make_gep_lane(ctx, src, lane);
}

static PolyUOp *rule_render_gep_single_shortcut(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b) {
  (void)ctx;
  (void)b;
  if (!u || u->op != POLY_OP_GEP || u->n_src < 1) return NULL;
  int lane = -1;
  if (u->arg.kind == POLY_ARG_INT) {
    lane = (int)u->arg.i;
  } else if (u->arg.kind == POLY_ARG_INT_TUPLE && u->arg.int_tuple.n == 1) {
    lane = (int)u->arg.int_tuple.vals[0];
  }
  if (lane < 0) return NULL;

  PolyUOp *src = u->src[0];
  if (src->op == POLY_OP_VECTORIZE && lane < src->n_src) return src->src[lane];
  return (src->dtype.count <= 1 && lane == 0) ? src : NULL;
}

static PolyUOp *rule_vector_cmp_to_scalarized_vector(
    PolyCtx *ctx,
    PolyUOp *u,
    const PolyBindings *b
) {
  (void)b;
  if (!(u->op == POLY_OP_CMPLT || u->op == POLY_OP_CMPNE || u->op == POLY_OP_CMPEQ)) return NULL;
  if (u->n_src != 2 || u->dtype.count <= 1) return NULL;
  if (poly_caps_packs_vector_compare(g_render_caps, u)) return NULL;
  int lanes = u->dtype.count;
  PolyUOp **elts = calloc((size_t)lanes, sizeof(*elts));
  if (!elts) return NULL;
  for (int i = 0; i < lanes; i++) {
    PolyUOp *a = lane_or_gep(ctx, u->src[0], i);
    PolyUOp *b0 = lane_or_gep(ctx, u->src[1], i);
    elts[i] = poly_uop2(ctx, u->op, poly_dtype_scalar(u->dtype), a, b0, poly_arg_none());
  }
  PolyUOp *ret = poly_uop(ctx, POLY_OP_VECTORIZE, u->dtype, elts, lanes, poly_arg_none());
  free(elts);
  return ret;
}

static PolyUOp *rule_vector_where_to_scalar(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b) {
  (void)b;
  if (u->op != POLY_OP_WHERE || u->n_src != 3 || u->dtype.count <= 1) return NULL;
  if (poly_caps_packs_vector_where(g_render_caps, u)) return NULL;
  int lanes = u->dtype.count;
  PolyUOp **elts = calloc((size_t)lanes, sizeof(*elts));
  if (!elts) return NULL;
  PolyDType sdt = poly_dtype_scalar(u->dtype);
  for (int i = 0; i < lanes; i++) {
    PolyUOp *c = lane_or_gep(ctx, u->src[0], i);
    PolyUOp *x = lane_or_gep(ctx, u->src[1], i);
    PolyUOp *y = lane_or_gep(ctx, u->src[2], i);
    elts[i] = poly_uop3(ctx, POLY_OP_WHERE, sdt, c, x, y, poly_arg_none());
  }
  PolyUOp *ret = poly_uop(ctx, POLY_OP_VECTORIZE, u->dtype, elts, lanes, poly_arg_none());
  free(elts);
  return ret;
}

static PolyUOp *rule_vector_const_where_to_stack(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b) {
  (void)b;
  if (!u || u->op != POLY_OP_WHERE || u->n_src != 3 || u->dtype.count <= 1) return NULL;
  if (poly_caps_packs_vector_where(g_render_caps, u)) return NULL;
  int lanes = u->dtype.count;
  if (lanes <= 0) return NULL;

  /* tinygrad folds vector WHEREs with compile-time lane masks into the chosen
   * lane values before rendering. This avoids producing scalar WHERE nodes for
   * upcasted pad/replicate masks such as [1,1,0,0,...]. */
  PolyUOp **elts = calloc((size_t)lanes, sizeof(*elts));
  if (!elts) return NULL;
  for (int i = 0; i < lanes; i++) {
    PolyUOp *gate = scalarize_lane_expr(ctx, u->src[0], i);
    if (!gate || gate->op != POLY_OP_CONST || gate->arg.kind != POLY_ARG_BOOL) {
      free(elts);
      return NULL;
    }
    PolyUOp *branch = gate->arg.b ? u->src[1] : u->src[2];
    elts[i] = scalarize_lane_expr(ctx, branch, i);
    if (!elts[i]) {
      free(elts);
      return NULL;
    }
  }
  PolyUOp *ret = poly_uop(ctx, POLY_OP_VECTORIZE, u->dtype, elts, lanes, poly_arg_none());
  free(elts);
  return ret;
}

static PolyUOp *rule_vector_sub_same_stack_rhs_to_add_neg(
    PolyCtx *ctx,
    PolyUOp *u,
    const PolyBindings *b
) {
  (void)b;
  if (!u || u->op != POLY_OP_SUB || u->n_src != 2 || u->dtype.count <= 1) return NULL;
  PolyUOp *rhs = u->src[1];
  if (!rhs || rhs->op != POLY_OP_VECTORIZE || rhs->n_src <= 1) return NULL;

  PolyUOp *same = rhs->src[0];
  for (int i = 1; i < rhs->n_src; i++)
    if (rhs->src[i] != same) return NULL;

  /* tinygrad's high-level Tensor.sub is `a + (b * -1)`. For broadcasted
   * scalar RHS values this reaches render as STACK(NEG(scalar), ...), not as a
   * vector SUB. Keep ordinary vector SUB intact; only rewrite repeated scalar
   * RHS stacks that came from broadcast subtraction. */
  PolyUOp *neg = poly_uop1(ctx, POLY_OP_NEG, same->dtype, same, poly_arg_none());
  PolyUOp **elts = calloc((size_t)rhs->n_src, sizeof(*elts));
  if (!elts) return NULL;
  for (int i = 0; i < rhs->n_src; i++)
    elts[i] = neg;
  PolyUOp *neg_stack =
      poly_uop(ctx, POLY_OP_VECTORIZE, rhs->dtype, elts, rhs->n_src, poly_arg_none());
  free(elts);
  return poly_uop2(ctx, POLY_OP_ADD, u->dtype, u->src[0], neg_stack, poly_arg_none());
}

typedef struct {
  bool ok;
  bool from_vector_load;
  PolyUOp *buf;
  PolyUOp *idx;
  int64_t offset;
  int64_t base;
  int width;
} ScalarLoadRun;

typedef struct {
  PolyUOp *buf;
  int64_t base;
  int width;
  uint64_t mask;
} ScalarLoadGroup;

static _Thread_local ScalarLoadGroup *g_scalar_load_groups = NULL;
static _Thread_local int g_n_scalar_load_groups = 0;

static uint64_t scalar_load_group_visible_mask(ScalarLoadRun info) {
  for (int i = 0; i < g_n_scalar_load_groups; i++) {
    ScalarLoadGroup *g = &g_scalar_load_groups[i];
    if (g->buf == info.buf && g->base == info.base && g->width == info.width) return g->mask;
  }
  return 0;
}

static bool scalar_const_load_info(PolyUOp *u, ScalarLoadRun *out) {
  if (out) memset(out, 0, sizeof(*out));
  if (!u || u->op != POLY_OP_LOAD || u->n_src != 1) return false;
  PolyUOp *idx = u->src[0];
  if (!idx || idx->op != POLY_OP_INDEX || idx->n_src != 2) return false;
  PolyUOp *buf = idx->src[0];
  PolyUOp *off = idx->src[1];
  if (!buf || !buf->dtype.is_ptr || !off || off->op != POLY_OP_CONST ||
      off->arg.kind != POLY_ARG_INT)
    return false;
  if (!codegen_foldable_buffer_dtype(buf)) return false;
  if (buf->dtype.ptr_size <= 0) return false;

  int width = g_max_fold_width;
  if (width < 2) return false;
  if (width > 4) width = 4; /* CStyle parity: tinygrad folds to float4 here. */
  int64_t offset = off->arg.i;
  if (offset < 0 || offset >= buf->dtype.ptr_size) return false;
  int64_t base = (offset / width) * width;
  if (base + width > buf->dtype.ptr_size) return false;

  if (out) {
    out->ok = true;
    out->from_vector_load = false;
    out->buf = buf;
    out->idx = idx;
    out->offset = offset;
    out->base = base;
    out->width = width;
  }
  return true;
}

static bool vector_load_lane_info(PolyUOp *u, ScalarLoadRun *out) {
  if (out) memset(out, 0, sizeof(*out));
  if (!u) return false;
  int64_t lane = 0;
  PolyUOp *load = NULL;
  if (u->op == POLY_OP_GEP && u->n_src == 1) {
    if (u->arg.kind == POLY_ARG_INT) {
      lane = u->arg.i;
    } else if (u->arg.kind == POLY_ARG_INT_TUPLE && u->arg.int_tuple.n == 1) {
      lane = u->arg.int_tuple.vals[0];
    } else {
      return false;
    }
    load = u->src[0];
  } else if (u->op == POLY_OP_INDEX && u->n_src == 2 && u->src[1] &&
             u->src[1]->op == POLY_OP_CONST && u->src[1]->arg.kind == POLY_ARG_INT) {
    lane = u->src[1]->arg.i;
    load = u->src[0];
  } else {
    return false;
  }

  if (!load || load->op != POLY_OP_LOAD || load->n_src != 1 || load->dtype.count <= 1) return false;
  PolyUOp *cast = load->src[0];
  if (!cast || cast->op != POLY_OP_CAST || cast->n_src != 1 || !cast->dtype.is_ptr ||
      cast->dtype.count <= 1)
    return false;
  PolyUOp *idx = cast->src[0];
  if (!idx || idx->op != POLY_OP_INDEX || idx->n_src != 2) return false;
  PolyUOp *buf = idx->src[0];
  PolyUOp *off = idx->src[1];
  if (!buf || !buf->dtype.is_ptr || !off || off->op != POLY_OP_CONST ||
      off->arg.kind != POLY_ARG_INT)
    return false;
  if (!codegen_foldable_buffer_dtype(buf)) return false;
  if (buf->dtype.ptr_size <= 0) return false;
  if (lane < 0 || lane >= cast->dtype.count) return false;

  int width = g_max_fold_width;
  if (width < 2) return false;
  if (width > 4) width = 4;
  int64_t offset = off->arg.i + lane;
  if (offset < 0 || offset >= buf->dtype.ptr_size) return false;
  int64_t base = (offset / width) * width;
  if (base + width > buf->dtype.ptr_size) return false;

  if (out) {
    out->ok = true;
    out->from_vector_load = true;
    out->buf = buf;
    out->idx = idx;
    out->offset = offset;
    out->base = base;
    out->width = width;
  }
  return true;
}

static PolyUOp *vector_load_lane_for_scalar_load(PolyCtx *ctx, ScalarLoadRun info) {
  PolyUOp *base_const =
      poly_uop0(ctx, POLY_OP_CONST, info.idx->src[1]->dtype, poly_arg_int(info.base));
  PolyUOp *idx_srcs[2] = {info.buf, base_const};
  PolyUOp *base_idx = poly_uop(ctx, POLY_OP_INDEX, info.buf->dtype, idx_srcs, 2, info.idx->arg);
  PolyDType vec_ptr = poly_dtype_ptr_vec(info.buf->dtype, info.width);
  PolyUOp *cast = poly_uop1(ctx, POLY_OP_CAST, vec_ptr, base_idx, poly_arg_none());
  PolyDType elem_dt = poly_dtype_scalar(info.buf->dtype);
  elem_dt.is_ptr = false;
  elem_dt.addrspace = 0;
  elem_dt.ptr_size = 0;
  elem_dt.vcount = 0;
  PolyDType vec_dt = poly_dtype_vec(elem_dt, info.width);
  PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, vec_dt, cast, poly_arg_none());
  return make_gep_lane(ctx, load, (int)(info.offset - info.base));
}

static PolyUOp *rule_stack_scalar_load_runs(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b) {
  (void)b;
  if (!u || u->op != POLY_OP_VECTORIZE || u->n_src <= 1) return NULL;

  ScalarLoadRun *info = calloc((size_t)u->n_src, sizeof(*info));
  PolyUOp **elts = calloc((size_t)u->n_src, sizeof(*elts));
  if (!info || !elts) {
    free(info);
    free(elts);
    return NULL;
  }
  bool any = false;
  for (int i = 0; i < u->n_src; i++) {
    if (scalar_const_load_info(u->src[i], &info[i]) || vector_load_lane_info(u->src[i], &info[i]))
      any = true;
  }
  if (!any) {
    free(info);
    free(elts);
    return NULL;
  }

  bool changed = false;
  for (int i = 0; i < u->n_src; i++) {
    elts[i] = u->src[i];
    if (!info[i].ok) continue;

    int run_uses = 0;
    int run_first_offset = 0;
    int run_last_offset = 0;
    bool run_seen = false;
    bool run_is_source_order_contiguous = true;
    bool run_has_all_lanes = true;
    uint64_t run_mask = 0;
    for (int j = 0; j < u->n_src; j++) {
      if (info[j].ok && info[j].buf == info[i].buf && info[j].base == info[i].base &&
          info[j].width == info[i].width) {
        run_uses++;
        int lane = (int)(info[j].offset - info[i].base);
        if (lane >= 0 && lane < 64) run_mask |= 1ULL << lane;
        if (!run_seen) {
          run_first_offset = (int)info[j].offset;
          run_last_offset = (int)info[j].offset;
          run_seen = true;
        } else {
          if (info[j].offset != (int64_t)run_last_offset + 1)
            run_is_source_order_contiguous = false;
          run_last_offset = (int)info[j].offset;
        }
      }
    }
    for (int lane = 0; lane < info[i].width; lane++) {
      bool found_lane = false;
      for (int j = 0; j < u->n_src; j++) {
        if (info[j].ok && info[j].buf == info[i].buf && info[j].base == info[i].base &&
            info[j].width == info[i].width && info[j].offset == info[i].base + lane) {
          found_lane = true;
          break;
        }
      }
      if (!found_lane) {
        run_has_all_lanes = false;
        break;
      }
    }
    /* tinygrad's load_store_folding groups indexes before renderer STACKs are
     * considered, so a lane from a smaller vector LOAD can still canonicalize
     * to the wider folded LOAD even when this particular STACK only uses one
     * lane. Fresh scalar LOADs need a source-order contiguous ascending run;
     * repeated or reversed pad/tril loads stay scalar in tinygrad. */
    if (!info[i].from_vector_load && (u->dtype.count <= g_max_fold_width || run_uses < 2 ||
                                      (!run_is_source_order_contiguous && !run_has_all_lanes)))
      continue;

    ScalarLoadRun load_info = info[i];
    int narrow_width = 1;
    if (g_max_fold_width >= 8 && run_uses >= 8)
      narrow_width = 8;
    else if (g_max_fold_width >= 4 && run_uses >= 4)
      narrow_width = 4;
    else if (g_max_fold_width >= 2 && run_uses >= 2)
      narrow_width = 2;
    uint64_t visible_mask = info[i].from_vector_load ? scalar_load_group_visible_mask(info[i]) : 0;
    bool has_external_visible_lanes = visible_mask != 0 && ((visible_mask & ~run_mask) != 0);
    if (info[i].from_vector_load && !has_external_visible_lanes &&
        u->dtype.count <= g_max_fold_width && narrow_width > 1 && narrow_width < info[i].width &&
        run_is_source_order_contiguous && (run_first_offset % narrow_width) == 0) {
      /* tinygrad's fold_expanded_index groups contiguous suffixes by their
       * first actual offset. For triu tails this means LOAD vec2 at offset 10,
       * not LOAD vec4 at offset 8 followed by GEP(2,3). */
      load_info.base = run_first_offset;
      load_info.width = narrow_width;
    }

    /* tinygrad's load_store_folding represents repeated contiguous scalar
     * loads as one vector LOAD plus GEP lane extracts. Apply the same shape
     * inside renderer-facing STACKs so cumalu windows do not carry many
     * redundant scalar LOAD/INDEX pairs. */
    elts[i] = vector_load_lane_for_scalar_load(ctx, load_info);
    changed = true;
  }

  if (!changed) {
    free(info);
    free(elts);
    return NULL;
  }
  PolyUOp *ret = poly_uop(ctx, POLY_OP_VECTORIZE, u->dtype, elts, u->n_src, poly_arg_none());
  free(info);
  free(elts);
  return ret;
}

static _Thread_local PolyMap *g_scalar_load_run_replace = NULL;

static PolyUOp *rule_replace_full_scalar_load_run(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b) {
  (void)ctx;
  (void)b;
  if (!g_scalar_load_run_replace) return NULL;
  return poly_map_get(g_scalar_load_run_replace, poly_ptr_hash(u), u, poly_ptr_eq);
}

static _Thread_local PolyPatternMatcher *g_pm_full_scalar_load_runs = NULL;
static PolyPatternMatcher *poly_pm_full_scalar_load_runs(void) {
  if (g_pm_full_scalar_load_runs) return g_pm_full_scalar_load_runs;
  PolyRule rules[] = {
      {poly_pat_op(POLY_OP_LOAD, NULL, 0, "u"), rule_replace_full_scalar_load_run},
      {poly_pat_op(POLY_OP_GEP, NULL, 0, "u"), rule_replace_full_scalar_load_run},
  };
  g_pm_full_scalar_load_runs =
      poly_pm_thread_cache(poly_pm_new(rules, (int)(sizeof(rules) / sizeof(rules[0]))));
  return g_pm_full_scalar_load_runs;
}

static PolyUOp *poly_fold_full_scalar_load_runs(PolyCtx *ctx, PolyUOp *sink) {
  if (!ctx || !sink) return sink;
  int n = 0;
  PolyUOp **topo = poly_toposort_alloc(ctx, sink, &n);
  if (!topo || n <= 0) {
    if (topo) poly_toposort_free(topo);
    return sink;
  }

  ScalarLoadRun *infos = calloc((size_t)n, sizeof(*infos));
  ScalarLoadGroup *groups = calloc((size_t)n, sizeof(*groups));
  if (!infos || !groups) {
    free(infos);
    free(groups);
    poly_toposort_free(topo);
    return sink;
  }

  int n_groups = 0;
  for (int i = 0; i < n; i++) {
    if (!scalar_const_load_info(topo[i], &infos[i]) && !vector_load_lane_info(topo[i], &infos[i]))
      continue;
    if (infos[i].width <= 1 || infos[i].width >= 64) continue;
    int gi = -1;
    for (int j = 0; j < n_groups; j++) {
      if (groups[j].buf == infos[i].buf && groups[j].base == infos[i].base &&
          groups[j].width == infos[i].width) {
        gi = j;
        break;
      }
    }
    if (gi < 0) {
      gi = n_groups++;
      groups[gi].buf = infos[i].buf;
      groups[gi].base = infos[i].base;
      groups[gi].width = infos[i].width;
      groups[gi].mask = 0;
    }
    int lane = (int)(infos[i].offset - infos[i].base);
    if (lane >= 0 && lane < infos[i].width) groups[gi].mask |= 1ULL << lane;
  }

  PolyMap *replace = poly_map_new(64);
  int n_replace = 0;
  for (int i = 0; i < n; i++) {
    if (!infos[i].ok || infos[i].width <= 1 || infos[i].width >= 64) continue;
    bool full = false;
    for (int j = 0; j < n_groups; j++) {
      if (groups[j].buf == infos[i].buf && groups[j].base == infos[i].base &&
          groups[j].width == infos[i].width) {
        uint64_t want = (1ULL << infos[i].width) - 1ULL;
        full = (groups[j].mask & want) == want;
        break;
      }
    }
    if (!full) continue;
    PolyUOp *repl = vector_load_lane_for_scalar_load(ctx, infos[i]);
    poly_map_set(replace, poly_ptr_hash(topo[i]), topo[i], repl, poly_ptr_eq);
    n_replace++;
  }

  PolyUOp *ret = sink;
  if (n_replace > 0) {
    PolyMap *prev = g_scalar_load_run_replace;
    g_scalar_load_run_replace = replace;
    ret = poly_graph_rewrite(ctx, sink, poly_pm_full_scalar_load_runs());
    g_scalar_load_run_replace = prev;
  }

  poly_map_destroy(replace);
  free(infos);
  free(groups);
  poly_toposort_free(topo);
  return ret;
}

static ScalarLoadGroup *poly_collect_scalar_load_groups(PolyCtx *ctx, PolyUOp *sink, int *out_n) {
  if (out_n) *out_n = 0;
  if (!ctx || !sink) return NULL;
  int n = 0;
  PolyUOp **topo = poly_toposort_alloc(ctx, sink, &n);
  if (!topo || n <= 0) {
    if (topo) poly_toposort_free(topo);
    return NULL;
  }

  ScalarLoadRun *infos = calloc((size_t)n, sizeof(*infos));
  ScalarLoadGroup *groups = calloc((size_t)n, sizeof(*groups));
  if (!infos || !groups) {
    free(infos);
    free(groups);
    poly_toposort_free(topo);
    return NULL;
  }

  int n_groups = 0;
  for (int i = 0; i < n; i++) {
    if (!scalar_const_load_info(topo[i], &infos[i]) && !vector_load_lane_info(topo[i], &infos[i]))
      continue;
    if (infos[i].width <= 1 || infos[i].width >= 64) continue;
    int gi = -1;
    for (int j = 0; j < n_groups; j++) {
      if (groups[j].buf == infos[i].buf && groups[j].base == infos[i].base &&
          groups[j].width == infos[i].width) {
        gi = j;
        break;
      }
    }
    if (gi < 0) {
      gi = n_groups++;
      groups[gi].buf = infos[i].buf;
      groups[gi].base = infos[i].base;
      groups[gi].width = infos[i].width;
      groups[gi].mask = 0;
    }
    int lane = (int)(infos[i].offset - infos[i].base);
    if (lane >= 0 && lane < infos[i].width) groups[gi].mask |= 1ULL << lane;
  }

  ScalarLoadGroup *ret = NULL;
  if (n_groups > 0) {
    ret = malloc((size_t)n_groups * sizeof(*ret));
    if (ret) memcpy(ret, groups, (size_t)n_groups * sizeof(*ret));
  }

  free(infos);
  free(groups);
  poly_toposort_free(topo);
  if (out_n) *out_n = ret ? n_groups : 0;
  if (!ret) {
    return NULL;
  }
  return ret;
}

static PolyUOp *unwrap_casted_load(PolyUOp *u, PolyUOp **cast_out) {
  if (!u) return NULL;
  if (u->op == POLY_OP_LOAD) {
    if (cast_out) *cast_out = NULL;
    return u;
  }
  if (u->op == POLY_OP_CAST && u->n_src == 1 && u->src[0] && u->src[0]->op == POLY_OP_LOAD) {
    if (cast_out) *cast_out = u;
    return u->src[0];
  }
  return NULL;
}

static bool is_true_const_codegen(PolyUOp *u) {
  return u && u->op == POLY_OP_CONST &&
         ((u->arg.kind == POLY_ARG_BOOL && u->arg.b) ||
          (u->arg.kind == POLY_ARG_INT && u->arg.i == 1));
}

static bool gated_load_matches_cond(PolyUOp *load, PolyUOp *cond, bool negated) {
  if (!load || load->op != POLY_OP_LOAD || load->n_src < 1 || !cond) return false;
  PolyUOp *gate = (load->n_src >= 3) ? load->src[2] : NULL;
  if (!gate) return false;
  if (!negated) return gate == cond;
  return gate->op == POLY_OP_CMPNE && gate->n_src == 2 && gate->src[0] == cond &&
         is_true_const_codegen(gate->src[1]);
}

static PolyUOp *cast_alt_for_load(PolyCtx *ctx, PolyUOp *alt, PolyDType load_dtype) {
  if (!alt) return NULL;
  if (alt->op == POLY_OP_CAST && alt->n_src == 1 && poly_dtype_eq(alt->src[0]->dtype, load_dtype))
    return alt->src[0];
  if (poly_dtype_eq(alt->dtype, load_dtype)) return alt;
  return poly_uop1(ctx, POLY_OP_CAST, load_dtype, alt, poly_arg_none());
}

/* tinygrad pm_render: give masked loads an explicit alt value before render. */
static PolyUOp *rule_masked_load_add_alt(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b) {
  (void)b;
  if (!u || u->op != POLY_OP_LOAD || u->n_src < 1) return NULL;
  bool has_gate = u->n_src >= 3 && codegen_is_bool_uop(u->src[2]);
  if (!has_gate) return NULL;

  bool needs_alt = (u->n_src == 1);
  if (!needs_alt && u->n_src >= 2 && u->src[1] &&
      (u->src[1]->op == POLY_OP_CUSTOM || u->src[1]->op == POLY_OP_STORE ||
       u->src[1]->op == POLY_OP_BARRIER))
    needs_alt = true;
  if (!needs_alt) return NULL;

  PolyUOp *zero = poly_const_like_int(ctx, u, 0);
  PolyUOp *srcs[64];
  int ns = 0;
  srcs[ns++] = u->src[0];
  srcs[ns++] = zero;
  for (int i = 2; i < u->n_src && ns < 64; i++)
    srcs[ns++] = u->src[i];
  return poly_uop(ctx, POLY_OP_LOAD, u->dtype, srcs, ns, u->arg);
}

/* tinygrad pm_render: WHERE(cond, gated_load(cond), alt) -> gated_load(alt). */
static PolyUOp *rule_where_after_gated_load(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b) {
  (void)b;
  if (!u || u->op != POLY_OP_WHERE || u->n_src != 3) return NULL;

  PolyUOp *cond = u->src[0];
  PolyUOp *load_cast = NULL;
  PolyUOp *load = unwrap_casted_load(u->src[1], &load_cast);
  if (!load || !gated_load_matches_cond(load, cond, false)) return NULL;

  PolyUOp *new_alt = cast_alt_for_load(ctx, u->src[2], load->dtype);
  if (!new_alt) return NULL;
  PolyUOp *srcs[64];
  int ns = 0;
  srcs[ns++] = load->src[0];
  srcs[ns++] = new_alt;
  for (int i = 2; i < load->n_src && ns < 64; i++)
    srcs[ns++] = load->src[i];
  PolyUOp *new_load = poly_uop(ctx, POLY_OP_LOAD, load->dtype, srcs, ns, load->arg);
  if (load_cast && !poly_dtype_eq(load_cast->dtype, new_load->dtype))
    new_load = poly_uop1(ctx, POLY_OP_CAST, load_cast->dtype, new_load, load_cast->arg);
  if (!poly_dtype_eq(u->dtype, new_load->dtype))
    new_load = poly_uop1(ctx, POLY_OP_CAST, u->dtype, new_load, poly_arg_none());
  return new_load;
}

/* tinygrad pm_render reverse form:
 * WHERE(cond, alt, gated_load(!cond)) -> gated_load(alt). */
static PolyUOp *rule_where_after_gated_load_rev(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b) {
  (void)b;
  if (!u || u->op != POLY_OP_WHERE || u->n_src != 3) return NULL;

  PolyUOp *cond = u->src[0];
  PolyUOp *load_cast = NULL;
  PolyUOp *load = unwrap_casted_load(u->src[2], &load_cast);
  if (!load || !gated_load_matches_cond(load, cond, true)) return NULL;

  PolyUOp *new_alt = cast_alt_for_load(ctx, u->src[1], load->dtype);
  if (!new_alt) return NULL;
  PolyUOp *srcs[64];
  int ns = 0;
  srcs[ns++] = load->src[0];
  srcs[ns++] = new_alt;
  for (int i = 2; i < load->n_src && ns < 64; i++)
    srcs[ns++] = load->src[i];
  PolyUOp *new_load = poly_uop(ctx, POLY_OP_LOAD, load->dtype, srcs, ns, load->arg);
  if (load_cast && !poly_dtype_eq(load_cast->dtype, new_load->dtype))
    new_load = poly_uop1(ctx, POLY_OP_CAST, load_cast->dtype, new_load, load_cast->arg);
  if (!poly_dtype_eq(u->dtype, new_load->dtype))
    new_load = poly_uop1(ctx, POLY_OP_CAST, u->dtype, new_load, poly_arg_none());
  return new_load;
}

static bool uop_tree_contains_target(PolyUOp *u, PolyUOp *target) {
  if (!u || !target) return false;
  if (u == target) return true;
  for (int i = 0; i < u->n_src; i++)
    if (uop_tree_contains_target(u->src[i], target)) return true;
  return false;
}

static int infer_scalar_load_lane(PolyCtx *ctx, PolyUOp *idx_expr, PolyUOp *vec_gate) {
  if (!ctx || !idx_expr || !vec_gate || vec_gate->dtype.count <= 1) return -1;
  int found = -1;
  for (int lane = 0; lane < vec_gate->dtype.count; lane++) {
    PolyUOp *gate_lane = scalarize_lane_expr(ctx, vec_gate, lane);
    if (!gate_lane) continue;
    if (!uop_tree_contains_target(idx_expr, gate_lane)) continue;
    if (found != -1 && found != lane) return -1;
    found = lane;
  }
  return found;
}

/* Scalar gated LOAD can still carry vector alt/gate operands after
 * devectorization. WGSL scalar loads need lane-specific scalar operands. */
static PolyUOp *rule_scalar_gated_load_operands(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b) {
  (void)b;
  if (!u || u->op != POLY_OP_LOAD || u->dtype.count != 1 || u->n_src < 1) return NULL;

  PolyUOp *idx = poly_find_index_through_cast(u->src[0]);
  if (!idx || idx->op != POLY_OP_INDEX) return NULL;

  PolyUOp *gate = (u->n_src >= 3 && codegen_is_bool_uop(u->src[2])) ? u->src[2] : NULL;
  if (!gate) return NULL;
  PolyUOp *alt = (u->n_src >= 2) ? u->src[1] : NULL;
  bool gate_is_vec = gate && gate->dtype.count > 1;
  bool alt_is_vec = alt && alt->dtype.count > 1;
  if (!gate_is_vec && !alt_is_vec) return NULL;

  int lane = gate_is_vec ? infer_scalar_load_lane(ctx, idx->src[1], gate) : -1;
  if (lane < 0) return NULL;

  PolyUOp *new_idx = build_scalar_lane_index(ctx, idx, idx->src[0], lane);
  if (!new_idx) return NULL;

  if (u->n_src >= 2) {
    PolyUOp *new_alt = alt_is_vec ? scalarize_lane_expr(ctx, alt, lane) : alt;
    if (!new_alt) return NULL;
    PolyUOp *load_srcs[3] = {
        new_idx, new_alt, gate_is_vec ? scalarize_lane_expr(ctx, gate, lane) : gate};
    if (!load_srcs[2]) return NULL;
    return poly_uop(ctx, POLY_OP_LOAD, u->dtype, load_srcs, 3, u->arg);
  }

  PolyUOp *zero = poly_const_like_int(ctx, u, 0);
  PolyUOp *load_srcs[3] = {
      new_idx, zero, gate_is_vec ? scalarize_lane_expr(ctx, gate, lane) : gate};
  if (!load_srcs[2]) return NULL;
  return poly_uop(ctx, POLY_OP_LOAD, u->dtype, load_srcs, 3, u->arg);
}

/* VCAT → VECTORIZE(GEP, GEP, ...) lowering (tinygrad symbolic.py:196):
 * VCAT can't be rendered; expand to VECTORIZE of per-element GEPs. */
static PolyUOp *rule_cat_to_vectorize(PolyCtx *ctx, PolyUOp *x, const PolyBindings *b) {
  (void)b;
  if (!x || x->op != POLY_OP_VCAT || x->n_src <= 0) return NULL;
  if (x->dtype.is_ptr) return NULL; /* don't expand pointer CATs */
  int total = 0;
  for (int i = 0; i < x->n_src; i++) {
    int cnt = x->src[i]->dtype.count;
    total += cnt > 0 ? cnt : 1;
  }
  if (total <= 0) return NULL;
  PolyUOp **elts = calloc((size_t)total, sizeof(*elts));
  if (!elts) return NULL;
  int p = 0;
  for (int i = 0; i < x->n_src; i++) {
    PolyUOp *src = x->src[i];
    int cnt = src->dtype.count;
    if (cnt <= 0) cnt = 1;
    for (int j = 0; j < cnt; j++)
      elts[p++] = make_gep_lane(ctx, src, j);
  }
  PolyUOp *ret = poly_uop(ctx, POLY_OP_VECTORIZE, x->dtype, elts, p, poly_arg_none());
  free(elts);
  return ret;
}

/* Render subset: full (DEVECTORIZE>=1) scatters unsupported vec CMP/WHERE to
 * scalar. A renderer capability can preserve native f32x4 compare/where ops. */
static _Thread_local PolyPatternMatcher *g_pm_render_subset = NULL;
static PolyPatternMatcher *poly_pm_render_subset(void) {
  if (g_pm_render_subset) return g_pm_render_subset;
  PolyRule rules[] = {
      {poly_pat_allow_any_len(poly_pat_op(POLY_OP_LOAD, NULL, 0, "u")), rule_masked_load_add_alt},
      {poly_pat_op(POLY_OP_WHERE, NULL, 0, "u"), rule_where_after_gated_load},
      {poly_pat_op(POLY_OP_WHERE, NULL, 0, "u"), rule_where_after_gated_load_rev},
      {poly_pat_allow_any_len(poly_pat_op(POLY_OP_LOAD, NULL, 0, "u")),
       rule_scalar_gated_load_operands},
      {poly_pat_op(POLY_OP_CONST, NULL, 0, "u"), rule_render_vector_const},
      {poly_pat_op(POLY_OP_VCONST, NULL, 0, "u"), rule_render_vconst},
      {poly_pat_op(POLY_OP_VCAT, NULL, 0, "x"), rule_cat_to_vectorize},
      {poly_pat_op(POLY_OP_VECTORIZE, NULL, 0, "u"), rule_vectorize_duplicate_void_effect},
      {poly_pat_op(POLY_OP_VECTORIZE, NULL, 0, "u"), rule_stack_scalar_load_runs},
      {poly_pat_op(POLY_OP_SUB, NULL, 0, "u"), rule_vector_sub_same_stack_rhs_to_add_neg},
      {poly_pat_op(POLY_OP_GEP, NULL, 0, "u"), rule_render_gep_single_shortcut},
      {poly_pat_op(POLY_OP_GEP, NULL, 0, "u"), rule_render_gep_multi},
      /* tinygrad's final rewrite still runs late scalar decompositions after
       * vector comparisons have been split into lane expressions. This turns
       * `(idx < 0) != true` gates into `-1 < idx` before the gates are stacked. */
      {poly_pat_op(POLY_OP_CMPNE, NULL, 0, "u"), rule_not_cmplt_to_bound},
      {poly_pat_op(POLY_OP_CMPLT, NULL, 0, "u"), rule_vector_cmp_to_scalarized_vector},
      {poly_pat_op(POLY_OP_CMPNE, NULL, 0, "u"), rule_vector_cmp_to_scalarized_vector},
      {poly_pat_op(POLY_OP_CMPEQ, NULL, 0, "u"), rule_vector_cmp_to_scalarized_vector},
      /* DEVECTORIZE=0 keeps most vector ALU, but bool masks used as INDEX
       * valids must become lane-wise scalar predicates for renderer parity with
       * tinygrad's cross-entropy gather kernels. */
      {poly_pat_op(POLY_OP_AND, NULL, 0, "u"), rule_bool_and_to_scalarized_vector},
      {poly_pat_op(POLY_OP_WHERE, NULL, 0, "u"), rule_vector_const_where_to_stack},
      {poly_pat_op(POLY_OP_WHERE, NULL, 0, "u"), rule_vector_where_to_scalar},
      {poly_pat_op(POLY_OP_VECTORIZE, NULL, 0, "u"), rule_vectorize_single},
  };
  g_pm_render_subset =
      poly_pm_thread_cache(poly_pm_new(rules, (int)(sizeof(rules) / sizeof(rules[0]))));
  return g_pm_render_subset;
}

static _Thread_local PolyPatternMatcher *g_pm_render_subset_vec = NULL;
static PolyPatternMatcher *poly_pm_render_subset_vec(void) {
  if (g_pm_render_subset_vec) return g_pm_render_subset_vec;
  PolyRule rules[] = {
      {poly_pat_allow_any_len(poly_pat_op(POLY_OP_LOAD, NULL, 0, "u")), rule_masked_load_add_alt},
      {poly_pat_op(POLY_OP_WHERE, NULL, 0, "u"), rule_where_after_gated_load},
      {poly_pat_op(POLY_OP_WHERE, NULL, 0, "u"), rule_where_after_gated_load_rev},
      {poly_pat_allow_any_len(poly_pat_op(POLY_OP_LOAD, NULL, 0, "u")),
       rule_scalar_gated_load_operands},
      {poly_pat_op(POLY_OP_CONST, NULL, 0, "u"), rule_render_vector_const},
      {poly_pat_op(POLY_OP_VCONST, NULL, 0, "u"), rule_render_vconst},
      {poly_pat_op(POLY_OP_VCAT, NULL, 0, "x"), rule_cat_to_vectorize},
      {poly_pat_op(POLY_OP_VECTORIZE, NULL, 0, "u"), rule_vectorize_duplicate_void_effect},
      {poly_pat_op(POLY_OP_VECTORIZE, NULL, 0, "u"), rule_stack_scalar_load_runs},
      {poly_pat_op(POLY_OP_SUB, NULL, 0, "u"), rule_vector_sub_same_stack_rhs_to_add_neg},
      {poly_pat_op(POLY_OP_GEP, NULL, 0, "u"), rule_render_gep_single_shortcut},
      {poly_pat_op(POLY_OP_GEP, NULL, 0, "u"), rule_render_gep_multi},
      /* See the full render subset: scalarized valid masks need the late
       * not-CMPLT bound form that tinygrad emits before stacking gates. */
      {poly_pat_op(POLY_OP_CMPNE, NULL, 0, "u"), rule_not_cmplt_to_bound},
      /* Scatter unsupported vec CMP/WHERE to per-lane scalar (same as
       * render_subset). Native vector compare/where backends opt out through
       * the caps-gated rules. */
      {poly_pat_op(POLY_OP_CMPLT, NULL, 0, "u"), rule_vector_cmp_to_scalarized_vector},
      {poly_pat_op(POLY_OP_CMPNE, NULL, 0, "u"), rule_vector_cmp_to_scalarized_vector},
      {poly_pat_op(POLY_OP_CMPEQ, NULL, 0, "u"), rule_vector_cmp_to_scalarized_vector},
      /* Keep this scoped to boolean AND masks. Integer vector AND can represent
       * arithmetic decompositions and should stay packed under DEVECTORIZE=0. */
      {poly_pat_op(POLY_OP_AND, NULL, 0, "u"), rule_bool_and_to_scalarized_vector},
      {poly_pat_op(POLY_OP_WHERE, NULL, 0, "u"), rule_vector_const_where_to_stack},
      {poly_pat_op(POLY_OP_WHERE, NULL, 0, "u"), rule_vector_where_to_scalar},
      {poly_pat_op(POLY_OP_VECTORIZE, NULL, 0, "u"), rule_vectorize_single},
  };
  g_pm_render_subset_vec =
      poly_pm_thread_cache(poly_pm_new(rules, (int)(sizeof(rules) / sizeof(rules[0]))));
  return g_pm_render_subset_vec;
}

/* Render subset for direct backends with packed integer masks: keep vector
 * CMP/WHERE packed. Unlike render_subset_vec, this does not scatter CMP/WHERE
 * to per-lane scalar. */
static _Thread_local PolyPatternMatcher *g_pm_render_subset_packed_int = NULL;
static PolyPatternMatcher *poly_pm_render_subset_packed_int(void) {
  if (g_pm_render_subset_packed_int) return g_pm_render_subset_packed_int;
  PolyRule rules[] = {
      {poly_pat_op(POLY_OP_CONST, NULL, 0, "u"), rule_render_vector_const},
      {poly_pat_op(POLY_OP_VCONST, NULL, 0, "u"), rule_render_vconst},
      {poly_pat_op(POLY_OP_VCAT, NULL, 0, "x"), rule_cat_to_vectorize},
      {poly_pat_op(POLY_OP_VECTORIZE, NULL, 0, "u"), rule_vectorize_duplicate_void_effect},
      {poly_pat_op(POLY_OP_VECTORIZE, NULL, 0, "u"), rule_stack_scalar_load_runs},
      {poly_pat_op(POLY_OP_SUB, NULL, 0, "u"), rule_vector_sub_same_stack_rhs_to_add_neg},
      {poly_pat_op(POLY_OP_GEP, NULL, 0, "u"), rule_render_gep_single_shortcut},
      {poly_pat_op(POLY_OP_GEP, NULL, 0, "u"), rule_render_gep_multi},
      /* Keep vector CMP/WHERE packed: the backend handles masks natively. */
      {poly_pat_op(POLY_OP_WHERE, NULL, 0, "u"), rule_vector_const_where_to_stack},
      {poly_pat_op(POLY_OP_VECTORIZE, NULL, 0, "u"), rule_vectorize_single},
  };
  g_pm_render_subset_packed_int =
      poly_pm_thread_cache(poly_pm_new(rules, (int)(sizeof(rules) / sizeof(rules[0]))));
  return g_pm_render_subset_packed_int;
}

/* tinygrad codegen/__init__.py::pm_remove_vec_dtypes, pointer storage part:
 * PARAM/BUFFER pointer UOps become their base dtype and retain the storage
 * extent as a CONST source. This final representation is what the linearizer
 * sorts and renders. */
static PolyDType poly_ptr_base_dtype_codegen(PolyDType dt) {
  PolyDType base = dt;
  base.is_ptr = false;
  base.addrspace = POLY_ADDR_GLOBAL;
  base.vcount = 0;
  base.ptr_size = 0;
  return base;
}

static PolyUOp *rule_remove_vec_dtype_param_buffer(
    PolyCtx *ctx,
    PolyUOp *buf,
    const PolyBindings *b
) {
  (void)b;
  if (!buf || (buf->op != POLY_OP_PARAM && buf->op != POLY_OP_BUFFER)) return NULL;
  if (!buf->dtype.is_ptr) return NULL;
  if (buf->op == POLY_OP_BUFFER && buf->dtype.addrspace != POLY_ADDR_GLOBAL) return NULL;
  int64_t size = buf->dtype.ptr_size;
  if (size < 0) size = 0;
  PolyUOp *src[1] = {poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(size))};
  return poly_uop(ctx, buf->op, poly_ptr_base_dtype_codegen(buf->dtype), src, 1, buf->arg);
}

static _Thread_local PolyPatternMatcher *g_pm_remove_vec_dtypes = NULL;
static PolyPatternMatcher *poly_pm_remove_vec_dtypes(void) {
  if (g_pm_remove_vec_dtypes) return g_pm_remove_vec_dtypes;
  PolyOpSet storage_set = {{0, 0}};
  storage_set = poly_opset_add(storage_set, POLY_OP_PARAM);
  storage_set = poly_opset_add(storage_set, POLY_OP_BUFFER);
  PolyRule rules[] = {
      {poly_pat_ops(storage_set, NULL, 0, "buf"), rule_remove_vec_dtype_param_buffer},
  };
  g_pm_remove_vec_dtypes =
      poly_pm_thread_cache(poly_pm_new(rules, (int)(sizeof(rules) / sizeof(rules[0]))));
  return g_pm_remove_vec_dtypes;
}

static PolyUOp *gater_rebuild(PolyCtx *ctx, PolyUOp *u, PolyUOp **src, int n_src) {
  return (u->tag != 0 || u->tag_arg.kind != POLY_ARG_NONE)
             ? poly_uop_tagged_arg(ctx, u->op, u->dtype, src, n_src, u->arg, u->tag, u->tag_arg)
             : poly_uop(ctx, u->op, u->dtype, src, n_src, u->arg);
}

static bool gater_invalid_where(PolyUOp *u, PolyUOp **gate, PolyUOp **index) {
  if (!u || u->op != POLY_OP_WHERE || u->n_src != 3 || !u->src[2] ||
      u->src[2]->op != POLY_OP_CONST || u->src[2]->arg.kind != POLY_ARG_INVALID ||
      !codegen_is_bool_uop(u->src[0]) || !poly_dtype_is_int(u->src[1]->dtype))
    return false;
  if (gate) *gate = u->src[0];
  if (index) *index = u->src[1];
  return true;
}

/* Pinned tinygrad/codegen/late/gater.py:5-17 moves Invalid-carrying integer
 * coordinates to LOAD/STORE validity sources. */
static PolyUOp *gater_ungated_index(PolyCtx *ctx, PolyUOp *mop, PolyUOp **gate_out) {
  if (gate_out) *gate_out = NULL;
  if (!mop || (mop->op != POLY_OP_INDEX && mop->op != POLY_OP_SHRINK) || mop->n_src < 2 ||
      mop->n_src > 64)
    return NULL;

  PolyUOp *src[64];
  int n_src = 0;
  src[n_src++] = mop->src[0];
  PolyUOp *gate = NULL;
  bool changed = false;

  for (int i = 1; i < mop->n_src; i++) {
    PolyUOp *coord_gate = NULL;
    PolyUOp *coord = NULL;
    if (gater_invalid_where(mop->src[i], &coord_gate, &coord)) {
      if (gate && gate != coord_gate) return NULL;
      gate = coord_gate;
      src[n_src++] = coord;
      changed = true;
      continue;
    }
    src[n_src++] = mop->src[i];
  }

  if (!changed || !gate) return NULL;
  if (gate_out) *gate_out = gate;
  return gater_rebuild(ctx, mop, src, n_src);
}

static PolyUOp *rule_move_gated_index_to_load(PolyCtx *ctx, PolyUOp *load, const PolyBindings *b) {
  (void)b;
  if (!load || load->op != POLY_OP_LOAD || load->n_src < 1 || load->n_src >= 3) return NULL;
  PolyUOp *gate = NULL;
  PolyUOp *ungated = gater_ungated_index(ctx, load->src[0], &gate);
  if (!ungated || !gate) return NULL;

  PolyUOp *alt = (load->n_src >= 2) ? load->src[1] : poly_const_like_int(ctx, load, 0);
  PolyUOp *src[3] = {ungated, alt, gate};
  return gater_rebuild(ctx, load, src, 3);
}

static PolyUOp *rule_move_gated_index_to_store(
    PolyCtx *ctx,
    PolyUOp *store,
    const PolyBindings *b
) {
  (void)b;
  if (!store || store->op != POLY_OP_STORE || store->n_src != 2) return NULL;
  PolyUOp *gate = NULL;
  PolyUOp *ungated = gater_ungated_index(ctx, store->src[0], &gate);
  if (!ungated || !gate) return NULL;

  PolyUOp *src[3] = {ungated, store->src[1], gate};
  return gater_rebuild(ctx, store, src, 3);
}

static _Thread_local PolyPatternMatcher *g_pm_move_gates_from_index = NULL;
static PolyPatternMatcher *poly_pm_move_gates_from_index(void) {
  if (g_pm_move_gates_from_index) return g_pm_move_gates_from_index;
  PolyRule rules[] = {
      {poly_pat_allow_any_len(poly_pat_op(POLY_OP_LOAD, NULL, 0, "load")),
       rule_move_gated_index_to_load},
      {poly_pat_allow_any_len(poly_pat_op(POLY_OP_STORE, NULL, 0, "store")),
       rule_move_gated_index_to_store},
  };
  g_pm_move_gates_from_index =
      poly_pm_thread_cache(poly_pm_new(rules, (int)(sizeof(rules) / sizeof(rules[0]))));
  return g_pm_move_gates_from_index;
}

static PolyDType codegen_shrink_dtype_from_ptr_cast(PolyDType dt) {
  int lanes = dt.count > 1 ? dt.count : dt.vcount;
  PolyDType base = dt;
  base.is_ptr = false;
  base.addrspace = POLY_ADDR_GLOBAL;
  base.ptr_size = 0;
  base.vcount = 1;
  base = poly_dtype_scalar(base);
  return lanes > 1 ? poly_dtype_vec(base, lanes) : base;
}

/* tinygrad pm_index_is_shrink:
 *   CAST(INDEX(buf, idx)) -> SHRINK(buf, idx, width)
 *
 * This is a late codegen memory-slice form, not frontend movement SHRINK. */
static PolyUOp *rule_index_cast_to_shrink(PolyCtx *ctx, PolyUOp *cast, const PolyBindings *b) {
  (void)b;
  if (!cast || cast->op != POLY_OP_CAST || cast->n_src != 1 || !cast->dtype.is_ptr) return NULL;
  int width = cast->dtype.count > 1 ? cast->dtype.count : cast->dtype.vcount;
  if (width <= 1) return NULL;
  PolyUOp *idx = cast->src[0];
  if (!idx || idx->op != POLY_OP_INDEX || idx->n_src != 2) return NULL;
  PolyUOp *buf = idx->src[0];
  if (!buf || !buf->dtype.is_ptr) return NULL;
  PolyUOp *width_uop = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(width));
  PolyUOp *srcs[3] = {buf, idx->src[1], width_uop};
  return poly_uop(
      ctx, POLY_OP_SHRINK, codegen_shrink_dtype_from_ptr_cast(cast->dtype), srcs, 3, poly_arg_none()
  );
}

/* tinygrad pm_index_is_shrink: final GEP becomes INDEX. */
static PolyUOp *rule_gep_to_index_codegen(PolyCtx *ctx, PolyUOp *gep, const PolyBindings *b) {
  (void)b;
  if (!gep || gep->op != POLY_OP_GEP || gep->n_src < 1) return NULL;
  PolyUOp *idx = NULL;
  if (gep->arg.kind == POLY_ARG_INT) {
    idx = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(gep->arg.i));
  } else if (gep->arg.kind == POLY_ARG_INT_TUPLE && gep->arg.int_tuple.n == 1) {
    idx = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(gep->arg.int_tuple.vals[0]));
  } else if (gep->arg.kind == POLY_ARG_INT_TUPLE && gep->arg.int_tuple.n > 1) {
    int n = gep->arg.int_tuple.n;
    PolyUOp **elts = calloc((size_t)n, sizeof(*elts));
    if (!elts) return NULL;
    for (int i = 0; i < n; i++)
      elts[i] = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(gep->arg.int_tuple.vals[i]));
    idx = poly_uop(ctx, POLY_OP_VECTORIZE, poly_dtype_vec(POLY_INT32, n), elts, n, poly_arg_none());
    free(elts);
  }
  if (!idx) return NULL;
  return poly_uop2(ctx, POLY_OP_INDEX, gep->dtype, gep->src[0], idx, poly_arg_none());
}

static _Thread_local PolyPatternMatcher *g_pm_index_is_shrink = NULL;
static PolyPatternMatcher *poly_pm_index_is_shrink(void) {
  if (g_pm_index_is_shrink) return g_pm_index_is_shrink;
  PolyRule rules[] = {
      {poly_pat_op1(POLY_OP_CAST, poly_pat_op(POLY_OP_INDEX, NULL, 0, NULL), "x"),
       rule_index_cast_to_shrink},
      {poly_pat_op(POLY_OP_GEP, NULL, 0, "x"), rule_gep_to_index_codegen},
  };
  g_pm_index_is_shrink =
      poly_pm_thread_cache(poly_pm_new(rules, (int)(sizeof(rules) / sizeof(rules[0]))));
  return g_pm_index_is_shrink;
}

/* Combined devectorize pass (cached) */
/*
 * Matches tinygrad codegen/__init__.py:79:
 *   pm_devectorize = sym+devectorize+load_store_folding+correct_load_store+load_store_indexing
 * All rules in ONE graph_rewrite so folding sees vectorized INDEX before devectorize scatters.
 */

static _Thread_local PolyPatternMatcher *g_combined_devec = NULL;
static PolyPatternMatcher *poly_pm_combined_devec(void) {
  if (g_combined_devec) return g_combined_devec;
  /* Matches tinygrad codegen/__init__.py:79:
   *   pm_devectorize = sym+devectorize+load_store_folding+correct_load_store+load_store_indexing
   * Run the shared symbolic matcher here so newly created invalid LOAD/STORE
   * nodes fold inside the same late devectorize stage, like tinygrad. */
  PolyPatternMatcher *sym_devec = poly_pm_concat(poly_symbolic(), poly_pm_devectorize());
  g_combined_devec = poly_pm_thread_cache(poly_pm_concat(sym_devec, poly_pm_load_store_folding()));
  poly_pm_destroy(sym_devec); /* g_combined_devec owns copied rules. */
  return g_combined_devec;
}

static _Thread_local PolyPatternMatcher *g_combined_nodevec = NULL;
static PolyPatternMatcher *poly_pm_combined_nodevec(void) {
  if (g_combined_nodevec) return g_combined_nodevec;
  /* Matches tinygrad pm_no_devec = sym + load_store_folding + correct_load_store +
   * load_store_indexing.
   *
   * The full lower-Invalid rewrite belongs to the post-index stage below, after
   * index dtype narrowing has matched tinygrad's pm_lower_index_dtype boundary. */
  PolyPatternMatcher *base = poly_pm_concat(poly_symbolic(), poly_pm_load_store_folding());
  g_combined_nodevec = poly_pm_thread_cache(base);
  return g_combined_nodevec;
}

/* Post-devectorize index dtype lowering (tinygrad pm_lower_index_dtype parity)
 *
 * Tinygrad carries symbolic index expressions as dtypes.weakint, then runs a
 * dedicated "lower all index dtypes" pass after devectorization. Polygrad's
 * matching dtype is POLY_INDEX. This pass narrows only POLY_INDEX subtrees to
 * concrete integer dtypes while preserving ordinary tensor integer ALU.
 */

static PolyUOp *rebuild_preserve_tag(
    PolyCtx *ctx,
    PolyUOp *u,
    PolyDType dtype,
    PolyUOp **srcs,
    int n_src
) {
  return (u->tag != 0 || u->tag_arg.kind != POLY_ARG_NONE)
             ? poly_uop_tagged_arg(ctx, u->op, dtype, srcs, n_src, u->arg, u->tag, u->tag_arg)
             : poly_uop(ctx, u->op, dtype, srcs, n_src, u->arg);
}

static PolyDType select_index_dtype(PolyCtx *ctx, PolyUOp *u) {
  /* Pinned tinygrad uop/ops.py:1655 selects solely from cached bounds:
   * long iff the expression overflows int32. A subtree-contains-LOAD shortcut
   * is both quadratic when repeated at every node and wrong for loaded index
   * expressions whose surrounding arithmetic genuinely exceeds int32. */
  int64_t vmin = 0, vmax = 0;
  poly_uop_minmax(ctx, u, &vmin, &vmax);
  if (vmin >= INT32_MIN && vmax <= INT32_MAX) {
    PolyDType lowered = POLY_INT32;
    if (u->dtype.count > 1) lowered = poly_dtype_vec(lowered, u->dtype.count);
    return lowered;
  }
  PolyDType lowered = POLY_INT64;
  if (u->dtype.count > 1) lowered = poly_dtype_vec(lowered, u->dtype.count);
  return lowered;
}

static bool weak_index_cast_inner(PolyUOp *u, PolyUOp **inner) {
  if (!u || u->op != POLY_OP_CAST || u->n_src != 1 || !poly_dtype_is_index(u->dtype)) return false;
  if (inner) *inner = u->src[0];
  return true;
}

static bool concrete_index_cast_inner(PolyUOp *u, PolyUOp **inner) {
  if (!u || u->op != POLY_OP_CAST || u->n_src != 1 || !u->src[0] ||
      !poly_dtype_is_int(u->src[0]->dtype) || poly_dtype_is_index(u->src[0]->dtype) ||
      poly_dtype_is_bool(u->src[0]->dtype))
    return false;
  if (inner) *inner = u->src[0];
  return true;
}

static PolyUOp *cast_index_value(PolyCtx *ctx, PolyUOp *u, PolyDType dtype) {
  return poly_dtype_eq(u->dtype, dtype) ? u
                                        : poly_uop1(ctx, POLY_OP_CAST, dtype, u, poly_arg_none());
}

static PolyUOp *wrap_index_value(PolyCtx *ctx, PolyUOp *original, PolyUOp *concrete) {
  return poly_dtype_eq(original->dtype, concrete->dtype)
             ? concrete
             : poly_uop1(ctx, POLY_OP_CAST, original->dtype, concrete, poly_arg_none());
}

static PolyUOp *wrap_weak_index_value(PolyCtx *ctx, PolyUOp *concrete) {
  PolyDType weak =
      concrete->dtype.count > 1 ? poly_dtype_vec(POLY_INDEX, concrete->dtype.count) : POLY_INDEX;
  return poly_uop1(ctx, POLY_OP_CAST, weak, concrete, poly_arg_none());
}

/* Pinned tinygrad uop/ops.py:1655-1686 registers local
 * pm_lower_index_dtype rules and lets graph_rewrite own graph traversal and
 * memoization. The old Polygrad callback recursively rewrote the complete
 * subtree of every matched UOp, then poly_graph_rewrite traversed those same
 * descendants again. Keep this callback node-local: its sources have already
 * been rewritten by the shared graph-rewrite context. */
static PolyUOp *lower_index_node(PolyCtx *ctx, PolyUOp *u) {
  if (!u) return NULL;

  /* Pinned tinygrad uop/ops.py:1684-1685 removes the final weak wrapper only
   * at graph roots. Keeping it on intermediate values is what lets parent
   * Binary/WHERE/RANGE rules distinguish weak-derived IR from raw concrete IR. */
  if (u->op == POLY_OP_SINK || u->op == POLY_OP_NOOP || u->op == POLY_OP_END) {
    PolyUOp *src_stack[16];
    PolyUOp **new_srcs = u->n_src > (int)(sizeof(src_stack) / sizeof(src_stack[0]))
                             ? malloc((size_t)u->n_src * sizeof(*new_srcs))
                             : src_stack;
    if (!new_srcs) return NULL;
    bool changed = false;
    for (int i = 0; i < u->n_src; i++) {
      PolyUOp *inner = NULL;
      new_srcs[i] = u->src[i]->dtype.count == 1 && weak_index_cast_inner(u->src[i], &inner)
                        ? inner
                        : u->src[i];
      if (new_srcs[i] != u->src[i]) changed = true;
    }
    PolyUOp *result = changed ? rebuild_preserve_tag(ctx, u, u->dtype, new_srcs, u->n_src) : NULL;
    if (new_srcs != src_stack) free(new_srcs);
    return result;
  }

  if (u->op == POLY_OP_CONST && poly_dtype_is_index(poly_dtype_scalar(u->dtype))) {
    if (u->arg.kind == POLY_ARG_INVALID) return NULL;
    PolyDType concrete_dtype = select_index_dtype(ctx, u);
    PolyUOp *concrete = rebuild_preserve_tag(ctx, u, concrete_dtype, u->src, u->n_src);
    return wrap_index_value(ctx, u, concrete);
  }

  /* Pinned tinygrad uop/ops.py:1657-1659. Both weak wrappers are part of the
   * pattern; raw mixed-width concrete comparisons are intentionally untouched
   * and remain validation errors rather than being silently normalized. */
  PolyUOp *x = NULL, *y = NULL;
  if (u->n_src == 2 && poly_opset_has(POLY_GROUP_BINARY, u->op) &&
      weak_index_cast_inner(u->src[0], &x) && weak_index_cast_inner(u->src[1], &y)) {
    PolyDType common = select_index_dtype(ctx, u);
    PolyDType common_scalar = poly_dtype_scalar(common);
    if (!poly_dtype_least_upper(common_scalar, x->dtype, &common_scalar) ||
        !poly_dtype_least_upper(common_scalar, y->dtype, &common_scalar))
      return NULL;
    common = u->dtype.count > 1 ? poly_dtype_vec(common_scalar, u->dtype.count) : common_scalar;
    PolyUOp *new_srcs[2] = {cast_index_value(ctx, x, common), cast_index_value(ctx, y, common)};
    PolyDType out_dtype = poly_opset_has(POLY_GROUP_COMPARISON, u->op) ? u->dtype : common;
    PolyUOp *concrete = poly_uop(ctx, u->op, out_dtype, new_srcs, 2, u->arg);
    return wrap_index_value(ctx, u, concrete);
  }

  if (u->op == POLY_OP_RANGE && u->n_src == 1) {
    /* Pinned tinygrad uop/ops.py:1663 takes RANGE dtype directly from the
     * already-lowered weak-wrapped end and does not constrain the RANGE's
     * current dtype. Selecting from RANGE's [0,end-1] bounds is wrong at
     * end=2^31: the values fit int32, but the bound itself requires int64. */
    PolyUOp *end = NULL;
    if (!weak_index_cast_inner(u->src[0], &end)) return NULL;
    PolyDType range_dtype = end->dtype;
    PolyUOp *src_stack[16];
    PolyUOp **new_srcs = u->n_src > (int)(sizeof(src_stack) / sizeof(src_stack[0]))
                             ? malloc((size_t)u->n_src * sizeof(*new_srcs))
                             : src_stack;
    if (!new_srcs) return NULL;
    for (int i = 0; i < u->n_src; i++)
      new_srcs[i] = u->src[i];
    new_srcs[0] = end;
    PolyUOp *concrete = rebuild_preserve_tag(ctx, u, range_dtype, new_srcs, u->n_src);
    if (new_srcs != src_stack) free(new_srcs);
    return wrap_weak_index_value(ctx, concrete);
  }

  if (u->op == POLY_OP_SPECIAL && u->n_src == 1 &&
      poly_dtype_is_index(poly_dtype_scalar(u->dtype))) {
    /* tinygrad pm_lower_index_dtype:
     *   SPECIAL(var.cast(weakint)) -> SPECIAL(int, var).cast(weakint)
     *
     * Workgroup IDs are backend/runtime values and WGSL declares each textual
     * id once. If widened address expressions rebuild SPECIAL itself as long,
     * the renderer sees two declarations for the same gidx/lidx name. Keep the
     * SPECIAL node int32 and put any required widening on the use-site cast. */
    PolyUOp *bound = NULL;
    if (!weak_index_cast_inner(u->src[0], &bound)) return NULL;
    PolyUOp *srcs[1] = {bound};
    PolyUOp *special = rebuild_preserve_tag(ctx, u, POLY_INT32, srcs, 1);
    return wrap_weak_index_value(ctx, special);
  }

  /* Pinned tinygrad pm_lower_index_dtype unifies two weak-wrapped concrete
   * WHERE branches (uop/ops.py:1661-1662). Invalid-bearing WHERE coordinates
   * are handled only by the gated INDEX rule below. */
  if (u->op == POLY_OP_WHERE && u->n_src == 3 && poly_dtype_is_index(poly_dtype_scalar(u->dtype))) {
    PolyUOp *new_srcs[3] = {u->src[0], u->src[1], u->src[2]};
    PolyUOp *x_value = NULL, *y_value = NULL;
    if (!weak_index_cast_inner(new_srcs[1], &x_value) ||
        !weak_index_cast_inner(new_srcs[2], &y_value))
      return NULL;
    PolyDType common = POLY_VOID;
    bool have_common = poly_dtype_least_upper(x_value->dtype, y_value->dtype, &common);
    int count = u->dtype.count;
    if (count > 1 && have_common) common = poly_dtype_vec(common, count);
    if (have_common && poly_dtype_is_int(common) && !poly_dtype_is_index(common) &&
        !poly_dtype_is_bool(common)) {
      new_srcs[1] = cast_index_value(ctx, x_value, common);
      new_srcs[2] = cast_index_value(ctx, y_value, common);
      PolyUOp *concrete = poly_uop(ctx, u->op, common, new_srcs, 3, u->arg);
      return wrap_weak_index_value(ctx, concrete);
    }
  }

  if (u->op == POLY_OP_STACK && poly_dtype_is_index(poly_dtype_scalar(u->dtype))) {
    PolyDType concrete_dtype = select_index_dtype(ctx, u);
    PolyDType lane_dtype = poly_dtype_scalar(concrete_dtype);
    PolyUOp *src_stack[16];
    PolyUOp **new_srcs = u->n_src > (int)(sizeof(src_stack) / sizeof(src_stack[0]))
                             ? malloc((size_t)u->n_src * sizeof(*new_srcs))
                             : src_stack;
    if (!new_srcs) return NULL;
    bool matched = true;
    for (int i = 0; i < u->n_src; i++) {
      PolyUOp *inner = NULL;
      if (!weak_index_cast_inner(u->src[i], &inner)) {
        matched = false;
        break;
      }
      new_srcs[i] = cast_index_value(ctx, inner, lane_dtype);
    }
    PolyUOp *result = NULL;
    if (matched) {
      PolyUOp *concrete = rebuild_preserve_tag(ctx, u, concrete_dtype, new_srcs, u->n_src);
      result = wrap_weak_index_value(ctx, concrete);
    }
    if (new_srcs != src_stack) free(new_srcs);
    return result;
  }

  /* POLY_OP_DEFINE_VAR is Polygrad's documented PG-PARITY-002 adaptation for
   * pinned PARAM(..., AddrSpace.ALU). Do not admit arbitrary PARAM as ALU until
   * that vocabulary debt is closed. */
  if (u->op == POLY_OP_DEFINE_VAR && poly_dtype_is_index(u->dtype)) {
    PolyUOp *concrete = rebuild_preserve_tag(ctx, u, POLY_INT32, u->src, u->n_src);
    return wrap_weak_index_value(ctx, concrete);
  }

  if (u->op == POLY_OP_BIND && u->n_src == 2 && poly_dtype_is_index(poly_dtype_scalar(u->dtype))) {
    PolyUOp *var = NULL, *value = NULL;
    if (!weak_index_cast_inner(u->src[0], &var) || !weak_index_cast_inner(u->src[1], &value))
      return NULL;
    if (value->op != POLY_OP_CONST) return NULL;
    PolyUOp *srcs[2] = {var, value};
    PolyUOp *concrete = poly_uop(ctx, u->op, var->dtype, srcs, 2, u->arg);
    return wrap_weak_index_value(ctx, concrete);
  }

  /* CAST is deliberately not rewritten here. It carries the weak provenance
   * to its parent, matching pinned pm_lower_index_dtype. */
  return NULL;
}

static PolyUOp *lower_gated_index_coord(PolyCtx *ctx, PolyUOp *coord, PolyUOp **gate_out) {
  if (!coord || coord->op != POLY_OP_WHERE || coord->n_src != 3 || !coord->src[2] ||
      coord->src[2]->op != POLY_OP_CONST || coord->src[2]->arg.kind != POLY_ARG_INVALID)
    return NULL;
  PolyUOp *idx = NULL;
  if (!concrete_index_cast_inner(coord->src[1], &idx)) return NULL;
  if (gate_out) *gate_out = coord->src[0];
  PolyUOp *invalid = poly_uop0(ctx, POLY_OP_CONST, idx->dtype, poly_arg_invalid());
  return poly_uop3(ctx, POLY_OP_WHERE, idx->dtype, coord->src[0], idx, invalid, coord->arg);
}

static PolyUOp *rule_lower_index_dtype(PolyCtx *ctx, PolyUOp *idx, const PolyBindings *b) {
  (void)b;
  if (!idx) return NULL;

  if (idx->op != POLY_OP_INDEX) return lower_index_node(ctx, idx);

  if (idx->n_src < 2) return NULL;
  PolyUOp *new_srcs[POLY_MAX_DIMS + 1];
  if (idx->n_src > (int)(sizeof(new_srcs) / sizeof(new_srcs[0]))) return NULL;
  bool changed = false;
  new_srcs[0] = idx->src[0];
  for (int i = 1; i < idx->n_src; i++)
    new_srcs[i] = idx->src[i];

  if (idx->n_src == 2) {
    PolyUOp *plain = NULL;
    PolyUOp *gated = lower_gated_index_coord(ctx, idx->src[1], NULL);
    if (gated)
      new_srcs[1] = gated;
    else if (concrete_index_cast_inner(idx->src[1], &plain))
      new_srcs[1] = plain;
  } else if (idx->n_src == 3) {
    /* Pinned image rules match the complete coordinate tuple: either two
     * plain casts, or two gated casts sharing the exact same gate. */
    PolyUOp *gate_y = NULL, *gate_x = NULL;
    PolyUOp *gated_y = lower_gated_index_coord(ctx, idx->src[1], &gate_y);
    PolyUOp *gated_x = lower_gated_index_coord(ctx, idx->src[2], &gate_x);
    if (gated_y && gated_x && gate_y == gate_x) {
      new_srcs[1] = gated_y;
      new_srcs[2] = gated_x;
    } else {
      PolyUOp *plain_y = NULL, *plain_x = NULL;
      if (concrete_index_cast_inner(idx->src[1], &plain_y) &&
          concrete_index_cast_inner(idx->src[2], &plain_x)) {
        new_srcs[1] = plain_y;
        new_srcs[2] = plain_x;
      }
    }
  } else {
    /* Polygrad's raw multidimensional INDEX extension has no pinned image
     * equivalent. Preserve its existing plain-coordinate legalization, but
     * do not generalize gated-image matching beyond tinygrad's arities. */
    for (int i = 1; i < idx->n_src; i++) {
      PolyUOp *plain = NULL;
      if (concrete_index_cast_inner(idx->src[i], &plain)) new_srcs[i] = plain;
    }
  }
  for (int i = 1; i < idx->n_src; i++)
    if (new_srcs[i] != idx->src[i]) changed = true;
  if (!changed) return NULL;
  /* Pinned buf.index(...) constructs a fresh INDEX, so matched tags do not
   * survive this rule. */
  return poly_uop(ctx, idx->op, idx->dtype, new_srcs, idx->n_src, idx->arg);
}

static _Thread_local PolyPatternMatcher *g_pm_lower_index_dtype = NULL;
static PolyPatternMatcher *poly_pm_lower_index_dtype(void) {
  if (g_pm_lower_index_dtype) return g_pm_lower_index_dtype;
  PolyRule rules[] = {
      /* tinygrad runs pm_lower_index_dtype on the whole kernel sink, not just
       * INDEX nodes. That matters once RANGE/END sources are real weakint
       * nodes: the loop owner and every address expression must be narrowed
       * together to the same concrete integer dtype. */
      {poly_pat_any("idx"), rule_lower_index_dtype},
  };
  g_pm_lower_index_dtype =
      poly_pm_thread_cache(poly_pm_new(rules, (int)(sizeof(rules) / sizeof(rules[0]))));
  return g_pm_lower_index_dtype;
}

static bool is_invalid_const_codegen(PolyUOp *u) {
  return u && u->op == POLY_OP_CONST && u->arg.kind == POLY_ARG_INVALID;
}

static bool uop_tree_contains_invalid_const_codegen(PolyUOp *u) {
  if (!u) return false;
  if (is_invalid_const_codegen(u)) return true;
  for (int i = 0; i < u->n_src; i++)
    if (uop_tree_contains_invalid_const_codegen(u->src[i])) return true;
  return false;
}

static bool int_const_value_codegen(PolyUOp *u, int64_t *out) {
  if (!u || u->op != POLY_OP_CONST || u->arg.kind != POLY_ARG_INT) return false;
  if (out) *out = u->arg.i;
  return true;
}

static bool split_add_const_codegen(PolyUOp *u, PolyUOp **base_out, int64_t *offset_out) {
  if (!u) return false;
  if (u->op == POLY_OP_ADD && u->n_src == 2) {
    int64_t k = 0;
    if (int_const_value_codegen(u->src[1], &k)) {
      if (base_out) *base_out = u->src[0];
      if (offset_out) *offset_out = k;
      return true;
    }
    if (int_const_value_codegen(u->src[0], &k)) {
      if (base_out) *base_out = u->src[1];
      if (offset_out) *offset_out = k;
      return true;
    }
  }
  if (base_out) *base_out = u;
  if (offset_out) *offset_out = 0;
  return true;
}

/* tinygrad uop/symbolic.py parses valid masks as integer bounds before late
 * rendering. Keep this codegen-scoped helper out of early symbolic passes:
 * helper construction and rangeify still need the original frontend graph, but
 * late LOAD.valid and padding-value masks should share the canonical bounds. */
static PolyUOp *canonicalize_valid_bound_clause(PolyCtx *ctx, PolyUOp *gate) {
  if (!gate) return gate;

  if (gate->op == POLY_OP_CMPLT && gate->n_src == 2) {
    PolyUOp *x = gate->src[0];
    PolyUOp *c = gate->src[1];
    int64_t cval = 0, offset = 0, adjusted = 0;
    PolyUOp *base = NULL;
    if (!poly_dtype_is_int(x->dtype) || poly_dtype_is_bool(x->dtype)) return gate;
    if (!int_const_value_codegen(c, &cval)) return gate;
    split_add_const_codegen(x, &base, &offset);
    if (offset == 0) return gate;
    if (!base || !poly_dtype_is_int(base->dtype) || poly_dtype_is_bool(base->dtype)) return gate;
    if (__builtin_sub_overflow(cval, offset, &adjusted)) return gate;
    PolyUOp *upper = poly_const_like_int(ctx, base, adjusted);
    return poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, base, upper, poly_arg_none());
  }

  if (gate->op != POLY_OP_CMPNE || gate->n_src != 2) return gate;
  PolyUOp *lt = NULL;
  if (gate->src[0]->op == POLY_OP_CMPLT && is_true_clause_const(gate->src[1]))
    lt = gate->src[0];
  else if (gate->src[1]->op == POLY_OP_CMPLT && is_true_clause_const(gate->src[0]))
    lt = gate->src[1];
  if (!lt || lt->n_src != 2) return gate;

  PolyUOp *x = lt->src[0];
  PolyUOp *c = lt->src[1];
  int64_t cval = 0, offset = 0, adjusted = 0;
  PolyUOp *base = NULL;
  if (!poly_dtype_is_int(x->dtype) || poly_dtype_is_bool(x->dtype)) return gate;
  if (!int_const_value_codegen(c, &cval)) return gate;
  split_add_const_codegen(x, &base, &offset);
  if (!base || !poly_dtype_is_int(base->dtype) || poly_dtype_is_bool(base->dtype)) return gate;
  if (__builtin_sub_overflow(cval, offset, &adjusted)) return gate;
  if (adjusted == INT64_MIN) return poly_const_like_bool(ctx, gate, true);
  adjusted -= 1;
  PolyUOp *lower = poly_const_like_int(ctx, base, adjusted);
  return poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, lower, base, poly_arg_none());
}

static PolyUOp *canonicalize_valid_gate_bounds(PolyCtx *ctx, PolyUOp *gate) {
  if (!gate) return gate;
  if (gate->op != POLY_OP_AND) return canonicalize_valid_bound_clause(ctx, gate);

  PolyUOp *clauses[128];
  int n_clauses = split_uop_and(gate, clauses, 128);
  bool changed = false;
  for (int i = 0; i < n_clauses; i++) {
    PolyUOp *new_clause = canonicalize_valid_bound_clause(ctx, clauses[i]);
    if (new_clause != clauses[i]) {
      clauses[i] = new_clause;
      changed = true;
    }
  }
  return changed ? and_all_clauses(ctx, clauses, n_clauses, NULL) : gate;
}

static bool uop_tree_contains_op_codegen(PolyUOp *u, PolyOps op) {
  if (!u) return false;
  if (u->op == op) return true;
  for (int i = 0; i < u->n_src; i++)
    if (uop_tree_contains_op_codegen(u->src[i], op)) return true;
  return false;
}

static PolyUOp *rule_canonicalize_and_valid_bounds(
    PolyCtx *ctx,
    PolyUOp *root,
    const PolyBindings *b
) {
  (void)b;
  if (!root || root->op != POLY_OP_AND) return NULL;
  /* tinygrad's simplify_valid skips masks that contain INDEX nodes; those are
   * data-dependent conditions, not pure bounds facts. */
  if (uop_tree_contains_op_codegen(root, POLY_OP_INDEX)) return NULL;

  PolyUOp *ret = canonicalize_valid_gate_bounds(ctx, root);
  return ret == root ? NULL : ret;
}

static PolyUOp *simplify_index_expr_given_gate(
    PolyCtx *ctx,
    PolyUOp *u,
    PolyUOp *gate,
    PolyUOp **memo_old,
    PolyUOp **memo_new,
    int *memo_n,
    int memo_cap
);

static bool uop_tree_contains_op_codegen(PolyUOp *u, PolyOps op);

static PolyUOp *rule_where_branch_given_gate(PolyCtx *ctx, PolyUOp *w, const PolyBindings *b) {
  (void)b;
  if (!w || w->op != POLY_OP_WHERE || w->n_src != 3) return NULL;

  PolyUOp *new_true = w->src[1];
  PolyUOp *new_false = w->src[2];
  bool changed = false;

  /* tinygrad's symbolic valid simplification (`uop_given_valid`) removes
   * Invalid-bearing expressions once the surrounding gate proves the branch.
   * This catches value-producing index kernels such as gather's intermediate
   * offset program, where Invalid is not an INDEX coordinate. */
  if (uop_tree_contains_invalid_const_codegen(new_true)) {
    PolyUOp *memo_old[256], *memo_new[256];
    int memo_n = 0;
    PolyUOp *simplified =
        simplify_index_expr_given_gate(ctx, new_true, w->src[0], memo_old, memo_new, &memo_n, 256);
    if (simplified != new_true) {
      new_true = simplified;
      changed = true;
    }
  }

  if (uop_tree_contains_invalid_const_codegen(new_false)) {
    PolyUOp *true_uop = poly_const_like_bool(ctx, w->src[0], true);
    PolyUOp *not_gate =
        poly_uop2(ctx, POLY_OP_CMPNE, w->src[0]->dtype, w->src[0], true_uop, poly_arg_none());
    PolyUOp *memo_old[256], *memo_new[256];
    int memo_n = 0;
    PolyUOp *simplified =
        simplify_index_expr_given_gate(ctx, new_false, not_gate, memo_old, memo_new, &memo_n, 256);
    if (simplified != new_false) {
      new_false = simplified;
      changed = true;
    }
  }

  if (!changed) return NULL;
  PolyUOp *new_srcs[3] = {w->src[0], new_true, new_false};
  return rebuild_preserve_tag(ctx, w, w->dtype, new_srcs, 3);
}

/* Return:
 *   1  -> gate implies cond is true
 *  -1  -> gate implies cond is false
 *   0  -> no implication recognized
 */
static int gate_implies_cond_branch(PolyUOp *gate, PolyUOp *cond) {
  if (!gate || !cond) return 0;
  if (gate == cond) return 1;
  if (gate->op == POLY_OP_CMPNE && gate->n_src == 2 && gate->src[0] == cond &&
      is_true_const_codegen(gate->src[1]))
    return -1;
  if (gate->op == POLY_OP_AND && gate->n_src == 2) {
    int left = gate_implies_cond_branch(gate->src[0], cond);
    if (left != 0) return left;
    return gate_implies_cond_branch(gate->src[1], cond);
  }
  return 0;
}

static bool gate_implies_range_point(PolyCtx *ctx, PolyUOp *gate, PolyUOp *expr, int64_t *out) {
  if (!gate || !expr) return false;
  if (gate->op == POLY_OP_AND && gate->n_src == 2) {
    return gate_implies_range_point(ctx, gate->src[0], expr, out) ||
           gate_implies_range_point(ctx, gate->src[1], expr, out);
  }
  if (gate->op != POLY_OP_CMPLT || gate->n_src != 2) return false;

  int64_t c = 0, lo = 0, hi = 0;
  poly_uop_minmax(ctx, expr, &lo, &hi);
  if (gate->src[1] == expr && int_const_value_codegen(gate->src[0], &c)) {
    /* c < expr and expr.max == c+1 proves expr == c+1. This is the common
     * replicate/right-pad endpoint after late valid-bound canonicalization. */
    int64_t upper_point = 0;
    if (!__builtin_add_overflow(c, 1, &upper_point) && hi == upper_point) {
      if (out) *out = hi;
      return true;
    }
  }
  if (gate->src[0] == expr && int_const_value_codegen(gate->src[1], &c)) {
    /* expr < c and expr.min == c-1 proves expr == c-1. */
    int64_t lower_point = 0;
    if (!__builtin_sub_overflow(c, 1, &lower_point) && lo == lower_point) {
      if (out) *out = lo;
      return true;
    }
  }
  return false;
}

static bool expr_point_given_gate(PolyCtx *ctx, PolyUOp *gate, PolyUOp *u, int64_t *out) {
  if (!u || !poly_dtype_is_int(u->dtype) || poly_dtype_is_bool(u->dtype)) return false;
  if (u->op == POLY_OP_CONST && u->arg.kind == POLY_ARG_INT) {
    if (out) *out = u->arg.i;
    return true;
  }
  int64_t base_val = 0, k = 0;
  PolyUOp *base = NULL;
  if (split_add_const_codegen(u, &base, &k) && base != u &&
      gate_implies_range_point(ctx, gate, base, &base_val)) {
    int64_t folded = 0;
    if (!__builtin_add_overflow(base_val, k, &folded)) {
      if (out) *out = folded;
      return true;
    }
  }
  return gate_implies_range_point(ctx, gate, u, out);
}

static PolyUOp *simplify_index_expr_given_gate(
    PolyCtx *ctx,
    PolyUOp *u,
    PolyUOp *gate,
    PolyUOp **memo_old,
    PolyUOp **memo_new,
    int *memo_n,
    int memo_cap
) {
  if (!u) return NULL;
  for (int i = 0; i < *memo_n; i++)
    if (memo_old[i] == u) return memo_new[i];

  PolyUOp *result = u;
  int64_t point = 0;
  if (expr_point_given_gate(ctx, gate, u, &point)) {
    result = poly_const_like_int(ctx, u, point);
    if (*memo_n < memo_cap) {
      memo_old[*memo_n] = u;
      memo_new[*memo_n] = result;
      (*memo_n)++;
    }
    return result;
  }

  if (u->op == POLY_OP_WHERE && u->n_src == 3) {
    int branch = gate_implies_cond_branch(gate, u->src[0]);
    if (branch != 0) {
      result = simplify_index_expr_given_gate(
          ctx, (branch > 0) ? u->src[1] : u->src[2], gate, memo_old, memo_new, memo_n, memo_cap
      );
      if (*memo_n < memo_cap) {
        memo_old[*memo_n] = u;
        memo_new[*memo_n] = result;
        (*memo_n)++;
      }
      return result;
    }
  }

  PolyUOp *new_srcs[64];
  bool changed = false;
  int ns = u->n_src < 64 ? u->n_src : 64;
  for (int i = 0; i < ns; i++) {
    new_srcs[i] =
        simplify_index_expr_given_gate(ctx, u->src[i], gate, memo_old, memo_new, memo_n, memo_cap);
    if (new_srcs[i] != u->src[i]) changed = true;
  }
  if (changed) result = rebuild_preserve_tag(ctx, u, u->dtype, new_srcs, ns);

  if (*memo_n < memo_cap) {
    memo_old[*memo_n] = u;
    memo_new[*memo_n] = result;
    (*memo_n)++;
  }
  return result;
}

typedef struct {
  PolyUOp *expr;
  int64_t lo;
  int64_t hi;
  bool has_lo;
  bool has_hi;
  PolyUOp *fake;
} ValidExprBound;

/* Pinned tinygrad/uop/symbolic.py:315-328. */
static bool parse_valid_bound_clause(
    PolyCtx *ctx,
    PolyUOp *clause,
    PolyUOp **expr_out,
    bool *is_upper_out,
    int64_t *bound_out
) {
  if (!clause || !expr_out || !is_upper_out || !bound_out) return false;
  if (clause->op == POLY_OP_CMPNE && clause->n_src == 2 &&
      is_true_clause_const(clause->src[1])) {
    PolyUOp *lt = clause->src[0];
    if (!lt || lt->op != POLY_OP_CMPLT || lt->n_src != 2 ||
        !poly_dtype_is_int(lt->src[0]->dtype))
      return false;
    int64_t lo = 0, hi = 0;
    poly_uop_minmax(ctx, lt->src[1], &lo, &hi);
    *expr_out = lt->src[0];
    *is_upper_out = false;
    *bound_out = lo;
    return true;
  }
  if (clause->op == POLY_OP_CMPLT && clause->n_src == 2 &&
      poly_dtype_is_int(clause->src[0]->dtype)) {
    int64_t lo = 0, hi = 0;
    poly_uop_minmax(ctx, clause->src[1], &lo, &hi);
    if (hi == INT64_MIN) return false;
    *expr_out = clause->src[0];
    *is_upper_out = true;
    *bound_out = hi - 1;
    return true;
  }
  return false;
}

static PolyUOp *simplify_with_valid_substitution(
    PolyCtx *ctx,
    PolyUOp *u,
    PolyUOp **exprs,
    PolyUOp **fakes,
    int n
) {
  if (!ctx || !u || !exprs || !fakes || n <= 0) return u;
  PolyUOp *substituted = poly_uop_substitute(ctx, u, exprs, fakes, n);
  if (!substituted || substituted == u) return u;
  substituted = poly_graph_rewrite(ctx, substituted, poly_symbolic());
  if (!substituted) return u;
  PolyUOp *restored = poly_uop_substitute(ctx, substituted, fakes, exprs, n);
  if (!restored) return u;
  PolyUOp *simplified = poly_graph_rewrite(ctx, restored, poly_symbolic());
  return simplified ? simplified : u;
}

static bool all_same_uops(PolyUOp **uops, int n) {
  if (!uops || n <= 0) return false;
  for (int i = 1; i < n; i++)
    if (uops[i] != uops[0]) return false;
  return true;
}

/* Pinned tinygrad/uop/symbolic.py:331-356. Polygrad's temporary DEFINE_VAR is
 * the registered PG-PARITY-002 spelling of tinygrad's ALU PARAM. It exists
 * only while proving a validity-constrained simplification and is substituted
 * back out before this helper returns. */
static PolyUOp *uop_given_valid(PolyCtx *ctx, PolyUOp *valid, PolyUOp *uop) {
  if (!ctx || !valid || !uop) return uop;

  PolyUOp *clauses[128];
  int n_clauses = split_uop_and(valid, clauses, 128);
  ValidExprBound bounds[128] = {0};
  int n_bounds = 0;
  for (int i = 0; i < n_clauses; i++) {
    PolyUOp *expr = NULL;
    bool is_upper = false;
    int64_t bound = 0;
    if (!parse_valid_bound_clause(ctx, clauses[i], &expr, &is_upper, &bound)) continue;
    int at = -1;
    for (int j = 0; j < n_bounds; j++)
      if (bounds[j].expr == expr) {
        at = j;
        break;
      }
    if (at < 0) {
      if (n_bounds >= 128) break;
      at = n_bounds++;
      bounds[at].expr = expr;
    }
    if (is_upper) {
      bounds[at].hi = bound;
      bounds[at].has_hi = true;
    } else {
      bounds[at].lo = bound;
      bounds[at].has_lo = true;
    }
  }
  if (n_bounds == 0) return uop;

  for (int i = 0; i < n_bounds; i++) {
    int64_t vmin = 0, vmax = 0;
    poly_uop_minmax(ctx, bounds[i].expr, &vmin, &vmax);
    if (!bounds[i].has_lo) bounds[i].lo = vmin;
    if (!bounds[i].has_hi) bounds[i].hi = vmax;
    if (bounds[i].lo > bounds[i].hi) continue;
    char name[48];
    snprintf(name, sizeof(name), "valid_bound_%d", i);
    bounds[i].fake = poly_uop0(
        ctx, POLY_OP_DEFINE_VAR, bounds[i].expr->dtype,
        poly_arg_define_var(name, bounds[i].lo, bounds[i].hi)
    );
    if (!bounds[i].fake) continue;

    PolyUOp *exprs[1] = {bounds[i].expr};
    PolyUOp *fakes[1] = {bounds[i].fake};
    uop = simplify_with_valid_substitution(ctx, uop, exprs, fakes, 1);

    /* Pinned try_simplex branch: X0+...+Xn >= 1 can prove the same rewrite
     * independently from each irreducible Xi >= 1. */
    if (bounds[i].lo == 1 && bounds[i].expr->op == POLY_OP_ADD) {
      PolyUOp *terms[128];
      int n_terms = split_uop_add(bounds[i].expr, terms, 128);
      bool irreducible = n_terms > 0;
      for (int j = 0; j < n_terms; j++)
        if (!poly_opset_has(POLY_GROUP_IRREDUCIBLE, terms[j]->op)) irreducible = false;
      if (irreducible) {
        PolyUOp *candidates[128];
        bool complete = true;
        for (int j = 0; j < n_terms; j++) {
          int64_t term_lo = 0, term_hi = 0;
          poly_uop_minmax(ctx, terms[j], &term_lo, &term_hi);
          (void)term_lo;
          char term_name[64];
          snprintf(term_name, sizeof(term_name), "valid_simplex_%d_%d", i, j);
          PolyUOp *fake = poly_uop0(
              ctx, POLY_OP_DEFINE_VAR, terms[j]->dtype,
              poly_arg_define_var(term_name, 1, term_hi)
          );
          if (!fake) {
            complete = false;
            break;
          }
          PolyUOp *term_exprs[1] = {terms[j]}, *term_fakes[1] = {fake};
          candidates[j] = simplify_with_valid_substitution(ctx, uop, term_exprs, term_fakes, 1);
        }
        if (complete && all_same_uops(candidates, n_terms)) {
          uop = candidates[0];
        } else if (complete && uop->op == POLY_OP_STACK && uop->n_src == 2) {
          bool first_same = true, second_same = true;
          for (int j = 0; j < n_terms; j++) {
            if (!candidates[j] || candidates[j]->op != POLY_OP_STACK ||
                candidates[j]->n_src != 2) {
              first_same = second_same = false;
              break;
            }
            if (j > 0 && candidates[j]->src[0] != candidates[0]->src[0]) first_same = false;
            if (j > 0 && candidates[j]->src[1] != candidates[0]->src[1]) second_same = false;
          }
          PolyUOp *srcs[2] = {uop->src[0], uop->src[1]};
          if (first_same) srcs[0] = candidates[0]->src[0];
          if (second_same) srcs[1] = candidates[0]->src[1];
          if (srcs[0] != uop->src[0] || srcs[1] != uop->src[1])
            uop = rebuild_preserve_tag(ctx, uop, uop->dtype, srcs, 2);
        }
      }
    }
  }

  PolyUOp *exprs[128], *fakes[128];
  int n_candidates = 0;
  for (int i = 0; i < n_bounds; i++) {
    if (!bounds[i].fake) continue;
    exprs[n_candidates] = bounds[i].expr;
    fakes[n_candidates] = bounds[i].fake;
    n_candidates++;
  }
  if (n_candidates > 0)
    uop = simplify_with_valid_substitution(ctx, uop, exprs, fakes, n_candidates);
  return uop;
}

/* Pinned tinygrad/codegen/late/devectorizer.py:39-42,54-56. */
static PolyUOp *rule_simplify_valid_index(
    PolyCtx *ctx,
    PolyUOp *index,
    const PolyBindings *b
) {
  (void)b;
  if (!index || index->op != POLY_OP_INDEX || index->n_src != 2) return NULL;
  PolyUOp *coord = index->src[1];
  if (!coord || coord->op != POLY_OP_WHERE || coord->n_src != 3 ||
      !is_invalid_const_codegen(coord->src[2]))
    return NULL;
  PolyUOp *simplified = uop_given_valid(ctx, coord->src[0], coord->src[1]);
  if (!simplified || simplified == coord->src[1]) return NULL;
  PolyUOp *coord_srcs[3] = {coord->src[0], simplified, coord->src[2]};
  PolyUOp *new_coord = rebuild_preserve_tag(ctx, coord, coord->dtype, coord_srcs, 3);
  PolyUOp *index_srcs[2] = {index->src[0], new_coord};
  return poly_uop(ctx, POLY_OP_INDEX, index->dtype, index_srcs, 2, index->arg);
}

static _Thread_local PolyPatternMatcher *g_pm_post_index_lower = NULL;
static PolyPatternMatcher *poly_pm_post_index_lower(void) {
  if (g_pm_post_index_lower) return g_pm_post_index_lower;
  PolyRule indexing_rules[] = {
      {poly_pat_op(POLY_OP_INDEX, NULL, 0, "idx"), rule_simplify_valid_index},
      {poly_pat_op(POLY_OP_WHERE, NULL, 0, "w"), rule_where_branch_given_gate},
      {poly_pat_op(POLY_OP_AND, NULL, 0, "valid"), rule_canonicalize_and_valid_bounds},
      {poly_pat_op(POLY_OP_WHERE, NULL, 0, "w"), rule_where_after_gated_load},
      {poly_pat_op(POLY_OP_WHERE, NULL, 0, "w"), rule_where_after_gated_load_rev},
  };
  /* Match tinygrad's post-index cleanup shape: lower dtype, push GEP, then only
   * the checked load/store indexing cleanup needed for Invalid-carrying indexes. */
  PolyPatternMatcher *pm_indexing =
      poly_pm_new(indexing_rules, (int)(sizeof(indexing_rules) / sizeof(indexing_rules[0])));
  PolyPatternMatcher *base = poly_pm_concat(poly_pm_lower_index_dtype(), poly_pm_gep_pushing());
  g_pm_post_index_lower = poly_pm_thread_cache(poly_pm_concat(base, pm_indexing));
  poly_pm_destroy(base);
  poly_pm_destroy(pm_indexing);
  return g_pm_post_index_lower;
}

/* Match the same fold-width selection that full_rewrite_to_sink_ex uses before
 * the combined devectorize stage. Public stage probes must see the same
 * load/store split behavior as the real backend pipeline. */
static int fold_width_from_caps(PolyRendererCaps caps) {
  return (caps.max_vec_width >= 8) ? 8 : (caps.max_vec_width >= 2) ? 4 : 1;
}

PolyUOp *poly_apply_devectorize_stage(
    PolyCtx *ctx,
    PolyUOp *sink,
    int devectorize,
    PolyRendererCaps caps
) {
  if (!ctx || !sink || devectorize < 0) return sink;
  g_max_fold_width = fold_width_from_caps(caps);
  g_render_caps = caps;
  return poly_graph_rewrite(
      ctx, sink, (devectorize >= 1) ? poly_pm_combined_devec() : poly_pm_combined_nodevec()
  );
}

PolyUOp *poly_apply_post_index_symbolic_stage(PolyCtx *ctx, PolyUOp *sink, int devectorize) {
  (void)devectorize;
  if (!ctx || !sink) return sink;
  sink = poly_graph_rewrite(ctx, sink, poly_pm_post_index_lower());
  return poly_graph_rewrite(ctx, sink, poly_symbolic());
}

/* GPU dims: replace outermost RANGE with SPECIAL */

/*
 * poly_add_gpudims — Port of tinygrad's pm_add_gpudims (simplified).
 *
 * After full_rewrite_to_sink, the graph has RANGE ops for data loops and
 * reduce loops.  This pass replaces the outermost non-reduce RANGE with
 * a SPECIAL("gidx0", N) op for GPU thread indexing.
 *
 * Reduce ranges (those that appear inside DEFINE_REG→AFTER→RANGE patterns)
 * are left as serial loops.
 *
 * Algorithm:
 * 1. Toposort to find all RANGE ops
 * 2. Identify reduce ranges: any RANGE that is src[2+] of an AFTER whose
 *    src[0] is a DEFINE_REG (these are the inner loops of accumulations)
 * 3. The first non-reduce RANGE → replace with SPECIAL
 * 4. Remove corresponding END ops for that RANGE
 */
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
    return poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, bound->arg);
  return poly_uop1(ctx, POLY_OP_CAST, POLY_INDEX, bound, poly_arg_none());
}

typedef struct {
  PolyUOp *expr;
  int64_t max;
  int orig[POLY_MAX_DIMS];
  int n_orig;
} GpuDimExpr;

static int64_t gpudim_expr_max(PolyCtx *ctx, PolyUOp *expr) {
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
  return poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(v));
}

static PolyUOp *gpudim_mul_const(PolyCtx *ctx, PolyUOp *x, int64_t v) {
  if (v == 1) return x;
  return poly_uop2(ctx, POLY_OP_MUL, POLY_INDEX, x, gpudim_const(ctx, v), poly_arg_none());
}

static PolyUOp *gpudim_floordiv_const(PolyCtx *ctx, PolyUOp *x, int64_t v) {
  if (v == 1) return x;
  return poly_uop2(ctx, POLY_OP_FLOORDIV, POLY_INDEX, x, gpudim_const(ctx, v), poly_arg_none());
}

static PolyUOp *gpudim_add(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  return poly_uop2(ctx, POLY_OP_ADD, POLY_INDEX, a, b, poly_arg_none());
}

static PolyUOp *gpudim_mul(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  return poly_uop2(ctx, POLY_OP_MUL, POLY_INDEX, a, b, poly_arg_none());
}

static PolyUOp *gpudim_floormod(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  return poly_uop2(ctx, POLY_OP_FLOORMOD, POLY_INDEX, a, b, poly_arg_none());
}

static PolyUOp *gpudim_floordiv(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  return poly_uop2(ctx, POLY_OP_FLOORDIV, POLY_INDEX, a, b, poly_arg_none());
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

static bool gpudim_group_dims(
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
      out[i].expr = gpudim_mul(ctx, out[i].expr, out[i + 1].expr);
      out[i].max = prod;
      if (out[i].n_orig + out[i + 1].n_orig > POLY_MAX_DIMS) return false;
      memcpy(
          out[i].orig + out[i].n_orig, out[i + 1].orig,
          (size_t)out[i + 1].n_orig * sizeof(out[i].orig[0])
      );
      out[i].n_orig += out[i + 1].n_orig;
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

static bool gpudim_split_dims(
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
      out[i].n_orig = 0;
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
  if (gpudim_group_dims(ctx, dims, n_dims, caps, grouped, &n_grouped)) {
    for (int i = 0; i < n_grouped; i++)
      out[i] = grouped[i];
    *n_out = n_grouped;
    return true;
  }

  if (n_dims > 3) return false;
  return gpudim_split_dims(ctx, dims, n_dims, caps, out, n_out);
}

static bool gpudim_build_indices(
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
  for (int i = 0; i < n_dims; i++) {
    ordered[i] = reverse ? dims[n_dims - 1 - i] : dims[i];
    /* tinygrad gpudims.py:28-29 recursively reconstructs dims[::-1], then
     * reverses the completed result. Contraction origins therefore index this
     * ordered domain, not the caller's pre-reversal dimension positions. */
    if (reverse) {
      ordered[i].orig[0] = i;
      ordered[i].n_orig = 1;
    }
  }

  GpuDimExpr limited[POLY_MAX_DIMS];
  int n_limited = 0;
  if (!gpudim_limited_dims(ctx, ordered, n_dims, caps, limited, &n_limited)) return false;

  PolyUOp *raw[3] = {NULL, NULL, NULL};
  for (int i = 0; i < n_limited; i++) {
    char name[16];
    snprintf(name, sizeof(name), "%s%d", prefix, i);
    raw[i] = poly_uop1(
        ctx, POLY_OP_SPECIAL, POLY_INDEX, gpudim_special_bound(ctx, limited[i].expr),
        poly_arg_str(name)
    );
  }

  PolyUOp *ordered_out[POLY_MAX_DIMS] = {0};
  if (n_limited < n_dims) {
    for (int i = 0; i < n_limited; i++) {
      PolyUOp *idx = raw[i];
      for (int j = 0; j < limited[i].n_orig; j++) {
        int orig = limited[i].orig[j];
        if (orig < 0 || orig >= n_dims) return false;
        if (j + 1 < limited[i].n_orig) {
          ordered_out[orig] = gpudim_floormod(ctx, idx, ordered[orig].expr);
          idx = gpudim_floordiv(ctx, idx, ordered[orig].expr);
        } else {
          ordered_out[orig] = idx;
        }
      }
    }
  } else if (n_limited > n_dims) {
    if (n_limited == 2 && n_dims == 1) {
      ordered_out[0] = gpudim_add(ctx, gpudim_mul(ctx, raw[0], limited[1].expr), raw[1]);
    } else if (n_limited == 3 && n_dims == 1) {
      PolyUOp *inner = gpudim_add(ctx, gpudim_mul(ctx, raw[0], limited[1].expr), raw[1]);
      ordered_out[0] = gpudim_add(ctx, gpudim_mul(ctx, inner, limited[2].expr), raw[2]);
    } else {
      return false;
    }
  } else {
    bool same = true;
    for (int i = 0; i < n_dims; i++)
      if (limited[i].max != ordered[i].max) same = false;
    if (same) {
      for (int i = 0; i < n_dims; i++)
        ordered_out[i] = raw[i];
    } else if (n_dims == 2) {
      PolyUOp *flat = gpudim_add(ctx, gpudim_mul(ctx, raw[0], limited[1].expr), raw[1]);
      ordered_out[0] = gpudim_floordiv(ctx, flat, ordered[1].expr);
      ordered_out[1] = gpudim_floormod(ctx, flat, ordered[1].expr);
    } else if (n_dims == 3) {
      PolyUOp *mul12 = gpudim_mul(ctx, ordered[2].expr, ordered[1].expr);
      PolyUOp *flat01 = gpudim_add(ctx, gpudim_mul(ctx, raw[0], limited[1].expr), raw[1]);
      PolyUOp *flat = gpudim_add(ctx, gpudim_mul(ctx, flat01, limited[2].expr), raw[2]);
      ordered_out[0] = gpudim_floordiv(ctx, flat, mul12);
      ordered_out[1] =
          gpudim_floormod(ctx, gpudim_floordiv(ctx, flat, ordered[2].expr), ordered[1].expr);
      ordered_out[2] = gpudim_floormod(ctx, flat, ordered[2].expr);
    } else {
      return false;
    }
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
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, sink, &n_topo);

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
        poly_arg_is_range(topo[i]->arg) ? poly_range_axis_type(topo[i]->arg) : POLY_AXIS_LOOP;
    if ((t == POLY_AXIS_GLOBAL || t == POLY_AXIS_THREAD || t == POLY_AXIS_LOOP) &&
        n_global < POLY_MAX_DIMS) {
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

  if (n_global == 0 && n_local == 0) return sink; /* nothing to parallelize */

  qsort(global_ranges, (size_t)n_global, sizeof(PolyUOp *), cmp_range_axis_id_ptr);
  qsort(local_ranges, (size_t)n_local, sizeof(PolyUOp *), cmp_range_axis_id_ptr);

  GpuDimExpr global_dims[POLY_MAX_DIMS];
  PolyUOp *global_idxs[POLY_MAX_DIMS] = {0};
  for (int i = 0; i < n_global; i++) {
    global_dims[i].expr = gpudim_special_bound(ctx, global_ranges[i]->src[0]);
    global_dims[i].max = gpudim_expr_max(ctx, global_dims[i].expr);
    global_dims[i].orig[0] = i;
    global_dims[i].n_orig = 1;
  }
  if (n_global > 0 &&
      !gpudim_build_indices(ctx, "gidx", global_dims, n_global, caps.global_max, true, global_idxs))
    return sink;

  GpuDimExpr local_dims[POLY_MAX_DIMS];
  PolyUOp *local_idxs[POLY_MAX_DIMS] = {0};
  for (int i = 0; i < n_local; i++) {
    local_dims[i].expr = gpudim_special_bound(ctx, local_ranges[i]->src[0]);
    local_dims[i].max = gpudim_expr_max(ctx, local_dims[i].expr);
    local_dims[i].orig[0] = i;
    local_dims[i].n_orig = 1;
  }
  if (n_local > 0 &&
      !gpudim_build_indices(ctx, "lidx", local_dims, n_local, caps.local_max, false, local_idxs))
    return sink;

  /* Substitute: replace all global/local ranges with SPECIALs and remove the
   * matching END nodes.
   * Rebuild graph bottom-up using a flat pointer-identity map. */
  int sub_cap = n_topo * 2 + 64;
  PolyUOp **sub_old = (PolyUOp **)malloc((size_t)sub_cap * sizeof(PolyUOp *));
  PolyUOp **sub_new = (PolyUOp **)malloc((size_t)sub_cap * sizeof(PolyUOp *));
  if (!sub_old || !sub_new) {
    free(sub_old);
    free(sub_new);
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

    /* END ops referencing substituted ranges must preserve any non-substituted
     * RANGE sources. tinygrad only drops the ended ranges that became SPECIAL;
     * it does not collapse a mixed END(store, specialized_range, serial_range)
     * to just store. */
    if (u->op == POLY_OP_END) {
      bool refs_target = false;
      PolyUOp *stack_end_srcs[64];
      PolyUOp **end_srcs = uop_src_scratch_alloc(u->n_src, stack_end_srcs, 64);
      if (!end_srcs) continue;
      int n_end_srcs = 0;
      if (u->n_src > 0) end_srcs[n_end_srcs++] = u->src[0];
      for (int j = 1; j < u->n_src; j++) {
        if (find_range_axis_key(u->src[j], global_ranges, n_global) >= 0 ||
            find_range_axis_key(u->src[j], local_ranges, n_local) >= 0) {
          refs_target = true;
          continue;
        }
        end_srcs[n_end_srcs++] = u->src[j];
      }
      if (refs_target) {
        PolyUOp *repl = NULL;
        if (n_end_srcs <= 1) {
          repl = u->src[0];
          /* Lookup if src[0] was substituted */
          for (int k = 0; k < n_subs; k++) {
            if (sub_old[k] == repl) {
              repl = sub_new[k];
              break;
            }
          }
        } else {
          PolyUOp *stack_mapped_srcs[64];
          PolyUOp **mapped_srcs = uop_src_scratch_alloc(n_end_srcs, stack_mapped_srcs, 64);
          if (!mapped_srcs) {
            uop_src_scratch_free(end_srcs, stack_end_srcs);
            continue;
          }
          for (int j = 0; j < n_end_srcs; j++) {
            mapped_srcs[j] = end_srcs[j];
            for (int k = 0; k < n_subs; k++) {
              if (sub_old[k] == mapped_srcs[j]) {
                mapped_srcs[j] = sub_new[k];
                break;
              }
            }
          }
          repl = poly_uop(ctx, POLY_OP_END, u->dtype, mapped_srcs, n_end_srcs, u->arg);
          uop_src_scratch_free(mapped_srcs, stack_mapped_srcs);
        }
        add_gpudim_sub(&sub_old, &sub_new, &n_subs, &sub_cap, u, repl);
        uop_src_scratch_free(end_srcs, stack_end_srcs);
        continue;
      }
      uop_src_scratch_free(end_srcs, stack_end_srcs);
    }

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
      PolyUOp *new_u = poly_uop(ctx, u->op, u->dtype, new_srcs, u->n_src, u->arg);

      /* Gated STORE for GLOBAL buffers missing local dims.
       * Pinned tinygrad/codegen/gpudims.py:92-99. After group_for_reduce, all
       * threads participate in the per-thread accumulation + shared-memory
       * reduction.  But only thread 0 should write the final result to the
       * global output buffer.  Without a guard, every thread STOREs the
       * same scalar → a benign-but-incorrect write race.
       *
       * Tinygrad keeps INDEX coordinates integer by replacing the offset with
       * WHERE(gate, offset, Invalid). The late gater moves that predicate to
       * STORE.src[2], and linearize emits IF/STORE/ENDIF.
       *
       * We detect: STORE whose src[0] is INDEX with a GLOBAL-addrspace
       * pointer, where the INDEX subtree contains no lidx SPECIAL.  For
       * those, add gate = CMPLT(lidx0, 1) to the integer coordinate. */
      if (new_u->op == POLY_OP_STORE && n_local > 0 && new_u->n_src >= 2) {
        PolyUOp *idx = new_u->src[0];
        /* Walk through CASTs to find the INDEX and retain the wrapper chain. */
        PolyUOp *wrappers[16];
        int n_wrappers = 0;
        PolyUOp *raw_idx = idx;
        while (raw_idx->op == POLY_OP_CAST && raw_idx->n_src == 1 &&
               n_wrappers < (int)(sizeof(wrappers) / sizeof(wrappers[0]))) {
          wrappers[n_wrappers++] = raw_idx;
          raw_idx = raw_idx->src[0];
        }

        if (raw_idx->op == POLY_OP_INDEX && raw_idx->n_src == 2 && raw_idx->dtype.is_ptr &&
            raw_idx->dtype.addrspace == POLY_ADDR_GLOBAL) {
          /* Check if any lidx SPECIAL appears in the INDEX subtree */
          int idx_n = 0;
          PolyUOp **idx_topo = poly_toposort(ctx, raw_idx, &idx_n);
          bool has_lidx = false;
          for (int j = 0; j < idx_n; j++) {
            if (idx_topo[j]->op == POLY_OP_SPECIAL && idx_topo[j]->arg.kind == POLY_ARG_STRING &&
                idx_topo[j]->arg.str && strncmp(idx_topo[j]->arg.str, "lidx", 4) == 0) {
              has_lidx = true;
              break;
            }
          }

          if (!has_lidx) {
            /* Pinned tinygrad/codegen/gpudims.py:96 builds
             * UOp.uprod(*[x.eq(0) ...]). Keep zero const-like with the weak
             * lidx so pm_lower_index_dtype can lower both operands together;
             * x.eq(0) is CMPNE(x, 0).logical_not(). */
            PolyUOp *gate = NULL;
            for (int g = 0; g < n_local; g++) {
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
              int lanes = coord->dtype.count > 1 ? coord->dtype.count : 1;
              PolyUOp *coord_gate = gate;
              if (lanes > 1) {
                if (lanes > 64) {
                  fprintf(
                      stderr, "polygrad: gpudims gate width %d exceeds UOp source cap\n", lanes
                  );
                  uop_src_scratch_free(new_srcs, stack_new_srcs);
                  free(sub_old);
                  free(sub_new);
                  return NULL;
                }
                PolyUOp *gate_lanes[64];
                for (int lane = 0; lane < lanes; lane++)
                  gate_lanes[lane] = gate;
                coord_gate = poly_uop(
                    ctx, POLY_OP_STACK, poly_dtype_vec(POLY_BOOL, lanes), gate_lanes, lanes,
                    poly_arg_none()
                );
              }
              PolyUOp *invalid = poly_uop0(ctx, POLY_OP_CONST, coord->dtype, poly_arg_invalid());
              PolyUOp *where = poly_uop3(
                  ctx, POLY_OP_WHERE, coord->dtype, coord_gate, coord, invalid, poly_arg_none()
              );
              PolyUOp *gated_src[2] = {raw_idx->src[0], where};
              PolyUOp *final_idx = gater_rebuild(ctx, raw_idx, gated_src, 2);

              for (int w = n_wrappers - 1; w >= 0; w--) {
                PolyUOp *wrapper_src[1] = {final_idx};
                final_idx = gater_rebuild(ctx, wrappers[w], wrapper_src, 1);
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
  return new_sink;
}

PolyUOp *poly_add_gpudims(PolyCtx *ctx, PolyUOp *sink) {
  PolyRendererCaps caps = {0};
  return poly_add_gpudims_ex(ctx, sink, caps);
}

static PolyUOp *poly_add_cpu_thread_dims(PolyCtx *ctx, PolyUOp *sink) {
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, sink, &n_topo);

  PolyUOp *thread_ranges[1];
  int n_thread = 0;
  for (int i = 0; i < n_topo; i++) {
    if (topo[i]->op != POLY_OP_RANGE) continue;
    PolyAxisType t =
        poly_arg_is_range(topo[i]->arg) ? poly_range_axis_type(topo[i]->arg) : POLY_AXIS_LOOP;
    if (t == POLY_AXIS_THREAD && n_thread < 1) thread_ranges[n_thread++] = topo[i];
  }
  if (n_thread == 0) return sink;

  PolyUOp *thread_range = thread_ranges[0];
  int64_t threads = 1;
  if (thread_range->n_src > 0 && thread_range->src[0]->op == POLY_OP_CONST &&
      thread_range->src[0]->arg.kind == POLY_ARG_INT)
    threads = thread_range->src[0]->arg.i;
  if (threads <= 1 || threads > INT32_MAX) return sink;

  PolyUOp *core_id = poly_uop0(
      ctx, POLY_OP_DEFINE_VAR, POLY_INT32, poly_arg_define_var("core_id", 0, threads - 1)
  );
  PolyUOp *thread_sub =
      poly_dtype_eq(thread_range->dtype, POLY_INT32)
          ? core_id
          : poly_uop1(ctx, POLY_OP_CAST, thread_range->dtype, core_id, poly_arg_none());

  int sub_cap = n_topo * 2 + 16;
  PolyUOp **sub_old = (PolyUOp **)malloc((size_t)sub_cap * sizeof(PolyUOp *));
  PolyUOp **sub_new = (PolyUOp **)malloc((size_t)sub_cap * sizeof(PolyUOp *));
  if (!sub_old || !sub_new) {
    free(sub_old);
    free(sub_new);
    return sink;
  }
  int n_subs = 0;
  add_gpudim_sub(&sub_old, &sub_new, &n_subs, &sub_cap, thread_range, thread_sub);

  PolyUOp *new_sink = sink;
  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    if (range_same_axis_key(u, thread_range)) continue;

    if (u->op == POLY_OP_END) {
      bool refs_thread = false;
      PolyUOp *stack_end_srcs[64];
      PolyUOp **end_srcs = uop_src_scratch_alloc(u->n_src, stack_end_srcs, 64);
      if (!end_srcs) continue;
      int n_end_srcs = 0;
      if (u->n_src > 0) end_srcs[n_end_srcs++] = u->src[0];
      for (int j = 1; j < u->n_src; j++) {
        if (range_same_axis_key(u->src[j], thread_range)) {
          refs_thread = true;
          continue;
        }
        end_srcs[n_end_srcs++] = u->src[j];
      }
      if (refs_thread) {
        PolyUOp *repl = NULL;
        if (n_end_srcs <= 1) {
          repl = end_srcs[0];
          for (int k = 0; k < n_subs; k++) {
            if (sub_old[k] == repl) {
              repl = sub_new[k];
              break;
            }
          }
        } else {
          PolyUOp *stack_mapped_srcs[64];
          PolyUOp **mapped_srcs = uop_src_scratch_alloc(n_end_srcs, stack_mapped_srcs, 64);
          if (!mapped_srcs) {
            uop_src_scratch_free(end_srcs, stack_end_srcs);
            continue;
          }
          for (int j = 0; j < n_end_srcs; j++) {
            mapped_srcs[j] = end_srcs[j];
            for (int k = 0; k < n_subs; k++) {
              if (sub_old[k] == mapped_srcs[j]) {
                mapped_srcs[j] = sub_new[k];
                break;
              }
            }
          }
          repl = poly_uop(ctx, POLY_OP_END, u->dtype, mapped_srcs, n_end_srcs, u->arg);
          uop_src_scratch_free(mapped_srcs, stack_mapped_srcs);
        }
        add_gpudim_sub(&sub_old, &sub_new, &n_subs, &sub_cap, u, repl);
        uop_src_scratch_free(end_srcs, stack_end_srcs);
        continue;
      }
      uop_src_scratch_free(end_srcs, stack_end_srcs);
    }

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
      PolyUOp *new_u = poly_uop(ctx, u->op, u->dtype, new_srcs, u->n_src, u->arg);
      add_gpudim_sub(&sub_old, &sub_new, &n_subs, &sub_cap, u, new_u);
      if (u == sink) new_sink = new_u;
    }
    uop_src_scratch_free(new_srcs, stack_new_srcs);
  }

  free(sub_old);
  free(sub_new);
  return new_sink;
}

/* Group for reduce: parallel reduction via shared memory */

/*
 * poly_group_for_reduce — Port of tinygrad's fix_group_for_reduce.
 *
 * Lowers REDUCE ops whose ranges were already tagged GROUP_REDUCE by
 * apply_opts/search. This mirrors tinygrad's pm_group_for_reduce:
 *   reduce_gfr, reduce_r = partition(...)
 *   if len(reduce_gfr) == 0: return None
 *
 * Selection of GROUP_REDUCE axes is an optimizer decision. This pass must not
 * invent new grouped-reduce axes for large serial reductions.
 */

/* Recursively clone a subtree, substituting old_node → new_node */
static PolyUOp *substitute_node(
    PolyCtx *ctx,
    PolyUOp *node,
    PolyUOp *old_node,
    PolyUOp *new_node,
    PolyUOp **memo_old,
    PolyUOp **memo_new,
    int *memo_n,
    int memo_cap
) {
  if (node == old_node) return new_node;
  /* Check memo */
  for (int i = 0; i < *memo_n; i++)
    if (memo_old[i] == node) return memo_new[i];

  /* Recurse on sources */
  bool changed = false;
  PolyUOp *new_srcs[64];
  int ns = node->n_src < 64 ? node->n_src : 64;
  for (int i = 0; i < ns; i++) {
    new_srcs[i] = substitute_node(
        ctx, node->src[i], old_node, new_node, memo_old, memo_new, memo_n, memo_cap
    );
    if (new_srcs[i] != node->src[i]) changed = true;
  }

  if (!changed) return node; /* subtree unchanged */

  PolyUOp *result = poly_uop(ctx, node->op, node->dtype, new_srcs, ns, node->arg);
  if (*memo_n < memo_cap) {
    memo_old[*memo_n] = node;
    memo_new[*memo_n] = result;
    (*memo_n)++;
  }
  return result;
}

PolyUOp *poly_group_for_reduce(PolyCtx *ctx, PolyUOp *sink, int block_size) {
  (void)block_size;
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, sink, &n_topo);

  /* Find all REDUCE ops */
  PolyUOp *reduces[32];
  int n_reduces = 0;
  for (int i = 0; i < n_topo; i++) {
    if (topo[i]->op == POLY_OP_REDUCE && n_reduces < 32) reduces[n_reduces++] = topo[i];
  }

  if (n_reduces == 0) return sink;

  /* Process each REDUCE: substitute in the full graph */
  PolyUOp **sub_old = malloc((size_t)(n_topo + 256) * sizeof(PolyUOp *));
  PolyUOp **sub_new = malloc((size_t)(n_topo + 256) * sizeof(PolyUOp *));
  int n_subs = 0;

  for (int r = 0; r < n_reduces; r++) {
    PolyUOp *red = reduces[r];
    PolyUOp *val = red->src[0];

    /* tinygrad fix_group_for_reduce:
     * REDUCE(val, ..., GROUP_REDUCE, REDUCE...) ->
     *   BUFFERIZE(partial over non-grouped ranges).INDEX(reduce_loop).REDUCE(reduce_loop)
     *
     * Polygrad still lowers local buffers manually (Phase 5 is not ported yet),
     * so keep the same semantic split but build DEFINE_LOCAL/STORE/BARRIER/LOAD
     * directly here when the heuristic has already retagged a range as
     * GROUP_REDUCE. */
    {
      PolyUOp *group_ranges[POLY_MAX_DIMS];
      PolyUOp *other_ranges[POLY_MAX_DIMS];
      int n_group = 0, n_other = 0;
      for (int i = 1; i < red->n_src; i++) {
        PolyUOp *rng = red->src[i];
        if (rng->op == POLY_OP_RANGE && poly_range_axis_type(rng->arg) == POLY_AXIS_GROUP_REDUCE) {
          if (n_group < POLY_MAX_DIMS) group_ranges[n_group++] = rng;
        } else if (rng->op == POLY_OP_RANGE) {
          if (n_other < POLY_MAX_DIMS) other_ranges[n_other++] = rng;
        }
      }

      if (n_group > 0) {
        int red_n_topo = 0;
        PolyUOp **tmp_topo = poly_toposort(ctx, red, &red_n_topo);
        if (!tmp_topo) continue;

        PolyUOp *upstream_locals[POLY_MAX_DIMS];
        int n_upstream = 0;
        for (int i = 0; i < red_n_topo; i++) {
          PolyUOp *u = tmp_topo[i];
          if (u->op != POLY_OP_RANGE || poly_range_axis_type(u->arg) != POLY_AXIS_LOCAL) continue;
          bool dup = false;
          for (int j = 0; j < n_upstream; j++) {
            if (upstream_locals[j] == u) {
              dup = true;
              break;
            }
          }
          if (!dup && n_upstream < POLY_MAX_DIMS) upstream_locals[n_upstream++] = u;
        }

        int64_t smem_size = 1;
        bool static_sizes = true;
        for (int i = 0; i < n_upstream; i++) {
          if (upstream_locals[i]->src[0]->op != POLY_OP_CONST) {
            static_sizes = false;
            break;
          }
          smem_size *= upstream_locals[i]->src[0]->arg.i;
        }
        for (int i = 0; i < n_group; i++) {
          if (group_ranges[i]->src[0]->op != POLY_OP_CONST) {
            static_sizes = false;
            break;
          }
          smem_size *= group_ranges[i]->src[0]->arg.i;
        }
        if (!static_sizes || smem_size <= 0 || smem_size > INT32_MAX) continue;

        PolyUOp *partial_srcs[1 + POLY_MAX_DIMS];
        partial_srcs[0] = val;
        for (int i = 0; i < n_other; i++)
          partial_srcs[1 + i] = other_ranges[i];
        /* tinygrad's fix_group_for_reduce keeps this as
         * x.replace(src=(x.src[0],)+reduce_r). With no remaining range axes
         * this still performs the horizontal lane reduction, which prevents a
         * vector partial from being stored into a scalar shared-memory slot. */
        PolyUOp *partial =
            poly_uop(ctx, POLY_OP_REDUCE, red->dtype, partial_srcs, 1 + n_other, red->arg);

        /* tinygrad's fix_group_for_reduce bufferizes the reduced value itself.
         * If the partial is vector-typed, the local buffer must keep that vector
         * dtype so devectorize can scalarize it into distinct shared-memory
         * lanes. Scalarizing here aliases vector lanes to one smem slot and can
         * leave STORE(STACK(LOAD(...)), vec) for renderers. */
        PolyDType smem_ptr = poly_dtype_ptr(red->dtype, smem_size, POLY_ADDR_LOCAL);
        PolyUOp *smem = poly_uop0(ctx, POLY_OP_DEFINE_LOCAL, smem_ptr, poly_arg_int(0));

        PolyUOp *store_rngs[POLY_MAX_DIMS];
        PolyUOp *store_bounds[POLY_MAX_DIMS];
        int n_store_dims = 0;
        for (int i = 0; i < n_upstream && n_store_dims < POLY_MAX_DIMS; i++) {
          store_rngs[n_store_dims] = upstream_locals[i];
          store_bounds[n_store_dims++] = upstream_locals[i]->src[0];
        }
        for (int i = 0; i < n_group && n_store_dims < POLY_MAX_DIMS; i++) {
          store_rngs[n_store_dims] = group_ranges[i];
          store_bounds[n_store_dims++] = group_ranges[i]->src[0];
        }
        PolyUOp *store_idx =
            poly_compute_flat_index_symbolic(ctx, store_rngs, store_bounds, n_store_dims);
        if (!store_idx) continue;

        PolyUOp *smem_store_idx =
            poly_uop2(ctx, POLY_OP_INDEX, smem_ptr, smem, store_idx, poly_arg_none());
        PolyUOp *smem_store =
            poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, smem_store_idx, partial, poly_arg_none());
        PolyUOp *barrier = poly_uop1(ctx, POLY_OP_BARRIER, POLY_VOID, smem_store, poly_arg_none());

        PolyUOp *final_ranges[POLY_MAX_DIMS];
        PolyUOp *final_bounds[POLY_MAX_DIMS];
        for (int i = 0; i < n_group; i++) {
          int64_t axis = poly_range_axis_id(group_ranges[i]->arg);
          /* These ranges are private to the synthetic final reduction created
           * for one GROUP_REDUCE.  Keep them distinct across source REDUCE ops:
           * pm_reduce merges END chains by identical range UOps, and merging
           * independent shared-memory finalizations nests the final reductions
           * and resets later accumulators inside the first loop. */
          int64_t final_axis = axis + 100 + (int64_t)r * 1024;
          /* Pinned expander.py:139 uses x.replace(arg=...), preserving the
           * GROUP_REDUCE RANGE dtype, sources, tag, and tag_arg.  Rebuilding
           * this as RANGE<int> makes flat-index construction add a weak CAST
           * that survives into HLB's grouped-reduction AFTER chain. */
          final_ranges[i] = poly_uop_tagged_arg(
              ctx, group_ranges[i]->op, group_ranges[i]->dtype, group_ranges[i]->src,
              group_ranges[i]->n_src, poly_arg_range(final_axis, POLY_AXIS_REDUCE),
              group_ranges[i]->tag, group_ranges[i]->tag_arg
          );
          final_bounds[i] = group_ranges[i]->src[0];
        }

        PolyUOp *load_rngs[POLY_MAX_DIMS];
        PolyUOp *load_bounds[POLY_MAX_DIMS];
        int n_load_dims = 0;
        for (int i = 0; i < n_upstream && n_load_dims < POLY_MAX_DIMS; i++) {
          load_rngs[n_load_dims] = upstream_locals[i];
          load_bounds[n_load_dims++] = upstream_locals[i]->src[0];
        }
        for (int i = 0; i < n_group && n_load_dims < POLY_MAX_DIMS; i++) {
          load_rngs[n_load_dims] = final_ranges[i];
          load_bounds[n_load_dims++] = final_bounds[i];
        }
        PolyUOp *load_idx =
            poly_compute_flat_index_symbolic(ctx, load_rngs, load_bounds, n_load_dims);
        if (!load_idx) continue;

        PolyUOp *smem_after_srcs[2] = {smem, barrier};
        PolyUOp *smem_after =
            poly_uop(ctx, POLY_OP_AFTER, smem_ptr, smem_after_srcs, 2, poly_arg_none());
        PolyUOp *final_load_idx =
            poly_uop2(ctx, POLY_OP_INDEX, smem_ptr, smem_after, load_idx, poly_arg_none());
        PolyUOp *final_load =
            poly_uop1(ctx, POLY_OP_LOAD, red->dtype, final_load_idx, poly_arg_none());

        PolyUOp *final_red_srcs[1 + POLY_MAX_DIMS];
        final_red_srcs[0] = final_load;
        for (int i = 0; i < n_group; i++)
          final_red_srcs[1 + i] = final_ranges[i];
        PolyUOp *final_reduce =
            poly_uop(ctx, POLY_OP_REDUCE, red->dtype, final_red_srcs, 1 + n_group, red->arg);

        sub_old[n_subs] = red;
        sub_new[n_subs] = final_reduce;
        n_subs++;
        continue;
      }
    }
  }

  if (n_subs == 0) {
    free(sub_old);
    free(sub_new);
    return sink;
  }

  /* Bottom-up graph rebuild: substitute original REDUCE → final_reduce.
   * No IF/ENDIF here: pinned gpudims first attaches Invalid-bearing integer
   * index validity, and the late gater/linearizer emits IF/STORE/ENDIF. */
  PolyUOp *new_sink = sink;

  /* Re-toposort since we modified things */
  int n_topo2 = 0;
  PolyUOp **topo2 = poly_toposort(ctx, sink, &n_topo2);

  for (int i = 0; i < n_topo2; i++) {
    PolyUOp *u = topo2[i];

    /* Check if this node itself was substituted */
    bool is_subst = false;
    for (int k = 0; k < n_subs; k++) {
      if (sub_old[k] == u) {
        is_subst = true;
        break;
      }
    }
    if (is_subst) continue;

    /* Check if any source was substituted */
    bool changed = false;
    PolyUOp *new_srcs[64];
    int ns = u->n_src < 64 ? u->n_src : 64;
    for (int j = 0; j < ns; j++) {
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
      /* Do NOT create IF/ENDIF here. Tinygrad's fix_group_for_reduce
       * (expander.py) never injects IF; add_gpudims attaches an
       * Invalid-bearing WHERE coordinate and the late gater/linearizer
       * converts it to IF/STORE/ENDIF.
       *
       * Creating IF/ENDIF in the DAG causes the IF (linearizer priority 0)
       * to float above the accumulation RANGE (priority 5), wrapping the
       * entire kernel body in `if (lidx0 < 1)`.  Only thread 0 executes,
       * so the inner loop checks vocab indices at stride 256 (0, 256, 512,
       * ...) and misses all non-stride-aligned indices -- e.g. token 785
       * is between 768 and 1024 and never gets checked. */
      PolyUOp *new_u = poly_uop(ctx, u->op, u->dtype, new_srcs, ns, u->arg);
      sub_old[n_subs] = u;
      sub_new[n_subs] = new_u;
      n_subs++;
      if (u == sink) new_sink = new_u;
    }
  }

  free(sub_old);
  free(sub_new);
  return new_sink;
}

/* Public wrapper for heuristic (used by tests) */
PolyUOp *poly_apply_opts_heuristic_ex(PolyCtx *ctx, PolyUOp *sink, PolyRendererCaps caps) {
  return poly_apply_opts_heuristic(ctx, sink, caps);
}

/* Public accessors for individual passes (used by CUDA linearizer) */

PolyPatternMatcher *poly_pm_reduce_pass(void) {
  return poly_pm_reduce();
}
PolyPatternMatcher *poly_pm_devectorize_pass(void) {
  return poly_pm_devectorize();
}
PolyPatternMatcher *poly_pm_decomp_pass(void) {
  return poly_pm_decomp();
}
PolyPatternMatcher *poly_pm_decomp_pass_caps(PolyRendererCaps caps) {
  return poly_pm_decomp_with_caps(caps.has_mulacc, caps.has_max, caps.has_threefry, caps.has_fdiv);
}
PolyPatternMatcher *poly_pm_transcendental_pass(void) {
  return poly_pm_transcendental((PolyRendererCaps){0});
}
PolyPatternMatcher *poly_pm_pre_expander_pass(void) {
  return poly_pm_pre_expander();
}
PolyPatternMatcher *poly_pm_move_where_on_load_pass(void) {
  return poly_pm_move_where_on_load();
}
PolyPatternMatcher *poly_pm_expander_pass(void) {
  return poly_pm_expander();
}
PolyPatternMatcher *poly_pm_render_subset_pass(void) {
  return poly_pm_render_subset();
}
PolyPatternMatcher *poly_pm_split_ends_pass(void) {
  return poly_pm_split_ends();
}
void poly_reset_acc_num(void) {}

PolyUOp *poly_apply_tc_opt(PolyCtx *ctx, PolyUOp *sink, PolyRendererCaps caps) {
  if (caps.n_tensor_cores <= 0) return sink;
  OptScheduler s;
  sched_init(&s, ctx, sink);
  if (!sched_can_optimize(&s)) return sink;

  int n_reduce = 0;
  for (int i = 0; i < s.n_rngs; i++)
    if (s.types[i] == POLY_AXIS_GROUP_REDUCE || s.types[i] == POLY_AXIS_REDUCE) n_reduce++;

  int tc_opt_env = poly_getenv_int("POLY_TC_OPT", 0);
  int use_tc_env = poly_getenv_int("POLY_USE_TC", 1);

  if (use_tc_env > 0 && (n_reduce == 1 || tc_opt_env >= 1)) {
    OptScheduler tk;
    sched_copy(&tk, &s);
    PolyUOp *tc_axes[3];
    bool tc_ok = sched_apply_tc_opt(
        &tk, 0, -1, tc_opt_env, use_tc_env, caps.tensor_cores, caps.n_tensor_cores, tc_axes
    );
    if (tc_ok) {
      for (int tc_dim = 1; tc_dim >= 0; tc_dim--) {
        int64_t bound = 0;
        if (tc_axes[tc_dim] && tc_axes[tc_dim]->n_src > 0 &&
            tc_axes[tc_dim]->src[0]->op == POLY_OP_CONST)
          bound = tc_axes[tc_dim]->src[0]->arg.i;
        if (bound <= 1) continue;
        int szs[] = {5, 4, 3, 2};
        for (int si = 0; si < 4; si++) {
          if (bound % szs[si] == 0) {
            int idx = -1;
            for (int ri = 0; ri < tk.n_rngs; ri++)
              if (tk.rngs[ri] == tc_axes[tc_dim]) {
                idx = ri;
                break;
              }
            if (idx >= 0)
              tc_axes[tc_dim] = sched_shift_to(&tk, tk.rngs[idx], szs[si], POLY_AXIS_UPCAST, false);
            break;
          }
        }
      }
      return tk.ast;
    }
  }
  return sink;
}

PolyUOp *poly_apply_pm_reduce(PolyCtx *ctx, PolyUOp *sink) {
  ReduceContext local_ctx = {0};
  PolyUOp *out = poly_graph_rewrite_ctx(ctx, sink, poly_pm_reduce(), &local_ctx);
  reduce_ctx_clear(&local_ctx);
  return out;
}

/* tinygrad schedule/rangeify.py::pm_mops, codegen preprocess subset:
 *   RESHAPE(...).INDEX(i, j, ...) -> source.INDEX(flattened_source_indices...)
 *
 * Custom CALL bodies enter full_rewrite_to_sink directly, bypassing the
 * schedule/rangeify movement-index path. Keep the same tinygrad rewrite here
 * so multidimensional custom-kernel placeholder indexes are flattened before
 * late LOAD/gated-load handling. */
static PolyUOp *rule_codegen_reshape_index(PolyCtx *ctx, PolyUOp *idx, const PolyBindings *b) {
  (void)b;
  if (!ctx || !idx || idx->op != POLY_OP_INDEX || idx->n_src < 2) return NULL;
  PolyUOp *reshape = idx->src[0];
  if (!reshape || reshape->op != POLY_OP_RESHAPE || reshape->n_src < 1) return NULL;

  int n_idx = idx->n_src - 1;
  if (n_idx <= 0 || n_idx > POLY_MAX_DIMS) return NULL;
  PolyShape in_shape = poly_uop_max_shape_cached(ctx, reshape->src[0]);
  if (in_shape.ndim < 0 || in_shape.ndim > POLY_MAX_DIMS) return NULL;

  PolyUOp *out_rngs[POLY_MAX_DIMS];
  for (int i = 0; i < n_idx; i++)
    out_rngs[i] = idx->src[1 + i];
  PolyUOp *in_rngs[POLY_MAX_DIMS];
  int n_in = 0;
  if (!poly_reshape_indices(ctx, reshape, out_rngs, n_idx, in_rngs, &n_in)) return NULL;

  /* Pinned schedule/rangeify.py:68-77 removes a partial RESHAPE INDEX
   * entirely when the indexed output prefix maps to an empty input prefix. */
  if (n_in == 0) return poly_dtype_eq(reshape->src[0]->dtype, idx->dtype) ? reshape->src[0] : NULL;

  PolyUOp *srcs[POLY_MAX_DIMS + 1];
  srcs[0] = reshape->src[0];
  for (int i = 0; i < n_in; i++)
    srcs[1 + i] = in_rngs[i];
  PolyUOp *ret = poly_uop(ctx, POLY_OP_INDEX, idx->dtype, srcs, n_in + 1, idx->arg);
  if (!ret) return NULL;

  /* Pinned returns the partial rewrite only when its remaining Tensor shape
   * exactly matches the original INDEX shape. Canonicalize dimension UOps
   * before comparing so static dtype spelling and symbolic maxima are not
   * mistaken for semantic equality. */
  PolyShape ret_shape = poly_uop_max_shape_cached(ctx, ret);
  PolyShape idx_shape = poly_uop_max_shape_cached(ctx, idx);
  if (ret_shape.ndim != idx_shape.ndim || ret_shape.ndim < 0) return NULL;
  for (int i = 0; i < ret_shape.ndim; i++) {
    PolyUOp *ret_dim = poly_uop_shape_dim(ctx, ret, i);
    PolyUOp *idx_dim = poly_uop_shape_dim(ctx, idx, i);
    if (!ret_dim)
      ret_dim = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(ret_shape.dims[i]));
    if (!idx_dim)
      idx_dim = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(idx_shape.dims[i]));
    if (!ret_dim || !idx_dim) return NULL;
    if (!poly_dtype_eq(poly_dtype_scalar(ret_dim->dtype), POLY_INDEX))
      ret_dim = poly_uop1(ctx, POLY_OP_CAST, POLY_INDEX, ret_dim, poly_arg_none());
    if (!poly_dtype_eq(poly_dtype_scalar(idx_dim->dtype), POLY_INDEX))
      idx_dim = poly_uop1(ctx, POLY_OP_CAST, POLY_INDEX, idx_dim, poly_arg_none());
    ret_dim = poly_graph_rewrite(ctx, ret_dim, poly_symbolic_simple());
    idx_dim = poly_graph_rewrite(ctx, idx_dim, poly_symbolic_simple());
    if (!ret_dim || !idx_dim || ret_dim != idx_dim) return NULL;
  }
  return ret;
}

/* Pinned schedule/rangeify.py:41-47 pm_syntactic_sugar concatenates the
 * coordinates of INDEX(INDEX(ptr, ...), ...). This lets the adjacent pm_mops
 * rules see the complete coordinate tuple and remove any remaining movement
 * op; it is source concatenation, not flat-offset addition. */
static PolyUOp *rule_codegen_ptr_index_concat(PolyCtx *ctx, PolyUOp *idx, const PolyBindings *b) {
  (void)b;
  if (!ctx || !idx || idx->op != POLY_OP_INDEX || idx->n_src < 2 || idx->dtype.is_ptr) return NULL;
  PolyUOp *inner = idx->src[0];
  if (!inner || inner->op != POLY_OP_INDEX || inner->n_src < 2 || !inner->dtype.is_ptr) return NULL;
  /* Pinned tinygrad/uop/spec.py:73-77 requires every INDEX coordinate to be
   * integer. The full-rewrite validator is the failure boundary; this guard
   * keeps the syntactic concat literal if an unsupported caller bypasses it. */
  for (int i = 1; i < inner->n_src; i++)
    if (!inner->src[i] || !poly_dtype_is_int(inner->src[i]->dtype)) return NULL;
  for (int i = 1; i < idx->n_src; i++)
    if (!idx->src[i] || !poly_dtype_is_int(idx->src[i]->dtype)) return NULL;
  if (inner->n_src > UINT16_MAX - (idx->n_src - 1)) return NULL;

  int n_src = inner->n_src + idx->n_src - 1;
  PolyUOp **srcs = malloc((size_t)n_src * sizeof(*srcs));
  if (!srcs) return NULL;
  for (int i = 0; i < inner->n_src; i++)
    srcs[i] = inner->src[i];
  for (int i = 1; i < idx->n_src; i++)
    srcs[inner->n_src + i - 1] = idx->src[i];
  PolyUOp *ret =
      poly_uop_tagged_arg(ctx, idx->op, idx->dtype, srcs, n_src, idx->arg, idx->tag, idx->tag_arg);
  free(srcs);
  return ret;
}

static _Thread_local PolyPatternMatcher *g_pm_codegen_mops = NULL;
static PolyPatternMatcher *poly_pm_codegen_mops(void) {
  if (g_pm_codegen_mops) return g_pm_codegen_mops;
  PolyRule rules[] = {
      {poly_pat_op(POLY_OP_INDEX, NULL, 0, "idx"), rule_codegen_reshape_index},
      {poly_pat_op(POLY_OP_INDEX, NULL, 0, "idx"), rule_codegen_ptr_index_concat},
  };
  g_pm_codegen_mops =
      poly_pm_thread_cache(poly_pm_new(rules, (int)(sizeof(rules) / sizeof(rules[0]))));
  return g_pm_codegen_mops;
}

/* Full rewrite-to-sink pipeline */

static bool device_is_gpu(int device) {
  return device == POLY_DEVICE_CUDA || device == POLY_DEVICE_HIP || device == POLY_DEVICE_WEBGPU;
}

static bool graph_has_int64(PolyCtx *ctx, PolyUOp *sink, PolyUOp **first) {
  int n = 0;
  PolyUOp **topo = poly_toposort(ctx, sink, &n);
  if (!topo) return true;
  for (int i = 0; i < n; i++) {
    PolyDType scalar = poly_dtype_scalar(topo[i]->dtype);
    if (poly_dtype_is_int(scalar) && scalar.bitsize == 64) {
      if (first) *first = topo[i];
      return true;
    }
  }
  return false;
}

PolyUOp *poly_full_rewrite_to_sink_ex(PolyCtx *ctx, PolyUOp *sink, PolyRewriteOpts opts) {
  /*
   * Unified pipeline matching tinygrad codegen/__init__.py full_rewrite_to_sink.
   * Backend-specific behavior is controlled by renderer config fields in opts,
   * not by if-device branches.
   *
   * New Phase 4 fields used here:
   *   opt_policy      — POLY_OPT_HEURISTIC (CPU) or POLY_OPT_TC_ONLY (GPU)
   *   gpu_block_size  — group_for_reduce block size (0 = skip)
   *   device          — PolyDevice, gates gpudims/control_flow
   *   dtype_matcher   — pinned pm_dtype_decomps bottom-up legalization
   *   extra_matcher   — renderer-specific final rewrite patterns (NULL = none)
   */

  /* Pinned tinygrad/codegen/__init__.py:55-61 runs type_verify(spec_tensor)
   * before pm_mops+pm_syntactic_sugar. Enforce its integer-INDEX-coordinate
   * predicate here, together with Polygrad's structural checks. */
  if (!ctx || !poly_validate_kernel_graph(ctx, sink)) return NULL;
  poly_debug_stage_graph("input", sink);

#define POLY_REWRITE_CHECK(stage_name)                                                             \
  do {                                                                                             \
    if (!sink) {                                                                                   \
      fprintf(stderr, "polygrad: full_rewrite_to_sink failed at %s\n", (stage_name));              \
      return NULL;                                                                                 \
    }                                                                                              \
  } while (0)

  POLY_REWRITE_CHECK("input");

  /* 1. Preprocess
   *
   * tinygrad runs pm_mops + pm_syntactic_sugar + pm_store_ranges here. Keep the
   * movement-on-INDEX subset here for custom CALL bodies, which bypass the
   * schedule/rangeify movement-index path. */
  sink = poly_graph_rewrite(ctx, sink, poly_pm_codegen_mops());
  poly_debug_stage_graph("preprocess", sink);
  POLY_REWRITE_CHECK("preprocess");

  /* 2. Optimization block (gated by optimize).
   * Matches tinygrad stage boundaries even where individual subpasses are still
   * incomplete on the polygrad side. */
  if (opts.optimize) {
    /* tinygrad: pm_load_collapse */
    sink = poly_graph_rewrite(ctx, sink, poly_pm_load_collapse());
    poly_debug_stage_graph("load collapse", sink);
    POLY_REWRITE_CHECK("load collapse");

    /* tinygrad: pm_split_ranges + pm_flatten_range */
    SplitRangeCtx srctx = {0};
    sink = poly_graph_rewrite_ctx(ctx, sink, poly_pm_split_ranges(), &srctx);
    sink = poly_graph_rewrite(ctx, sink, poly_pm_flatten_range());
    poly_debug_stage_graph("split ranges", sink);
    POLY_REWRITE_CHECK("split ranges");

    /* tinygrad: sym + pm_flatten_range */
    sink = poly_graph_rewrite(ctx, sink, poly_symbolic());
    sink = poly_graph_rewrite(ctx, sink, poly_pm_flatten_range());
    poly_debug_stage_graph("initial symbolic", sink);
    POLY_REWRITE_CHECK("initial symbolic");

    /* tinygrad: pm_flatten_range + pm_simplify_ranges */
    sink = poly_graph_rewrite(ctx, sink, poly_pm_simplify_ranges());
    poly_debug_stage_graph("simplify ranges", sink);
    POLY_REWRITE_CHECK("simplify ranges");

    /* tinygrad apply_opts prelude: convert eligible LOOP output ranges to
     * GLOBAL before running heuristic/beam/tensor-core scheduling. */
    if (device_is_gpu(opts.device)) {
      sink = convert_loop_output_ranges_to_global(ctx, sink);
      POLY_REWRITE_CHECK("convert loop output ranges");
    }

    /* tinygrad: apply_opts
     *
     * tinygrad apply_opts(...) returns Scheduler.get_optimized_ast(), which
     * runs pm_flatten_range before the pipeline moves on. Without that cleanup
     * END sources can remain as arithmetic expressions after shift_to/upcast,
     * and later expander passes drop the ended RANGE structure entirely. */
    if (opts.beam_width > 0) {
      sink = poly_beam_search(ctx, sink, opts.beam_width, opts);
    } else if (opts.opt_policy == POLY_OPT_TC_ONLY) {
      sink = poly_apply_tc_opt(ctx, sink, opts.caps);
    } else {
      sink = poly_apply_opts_heuristic(ctx, sink, opts.caps);
    }
    sink = poly_graph_rewrite(ctx, sink, poly_pm_flatten_range());
    poly_debug_stage_graph("apply opts", sink);
    POLY_REWRITE_CHECK("apply opts");
  }

  /* 3. Postopt symbolic */
  sink = poly_graph_rewrite(ctx, sink, poly_symbolic());
  sink = poly_graph_rewrite(ctx, sink, poly_pm_move_where_on_load());
  poly_debug_stage_graph("postopt symbolic", sink);
  POLY_REWRITE_CHECK("postopt symbolic");

  /* 4. Expander.
   * tinygrad groups sym + pm_pre_expander + pm_group_for_reduce + expander. */
  sink = poly_graph_rewrite(ctx, sink, poly_symbolic());
  /* Normal tensor kernels pass through schedule/rangeify, which already runs
   * symbolic + pm_reduce_simplify before this backend rewrite. Public custom
   * CALL bodies enter here directly, so run the same cleanup to preserve
   * tinygrad's unparented-reduce semantics, e.g. REDUCE_ADD(1, r) -> bound(r). */
  sink = poly_apply_symbolic_reduce_simplify(ctx, sink);
  POLY_REWRITE_CHECK("reduce simplify");
  sink = poly_graph_rewrite(ctx, sink, poly_pm_pre_expander());
  if (opts.gpu_block_size > 0) sink = poly_group_for_reduce(ctx, sink, opts.gpu_block_size);
  sink = poly_graph_rewrite(ctx, sink, poly_pm_expander());
  /* tinygrad runs sym in the same combined matcher as expander, so symbolic
   * folds on freshly expanded small masked paths happen in this stage, not much
   * later in lower_index_dtype. Keep that stage boundary aligned. */
  sink = poly_graph_rewrite(ctx, sink, poly_symbolic());
  poly_debug_stage_graph("expander", sink);
  POLY_REWRITE_CHECK("expander");

  /* 5. Add local buffers.
   * tinygrad runs pm_add_buffers_local + rangeify_codegen here. Polygrad keeps
   * this stage explicit in traces while still using the shared prepared-step
   * scheduling path outside full_rewrite_to_sink_ex. */
  poly_debug_stage_graph("add local buffers", sink);
  POLY_REWRITE_CHECK("add local buffers");

  /* 6. Remove reduce */
  sink = poly_apply_pm_reduce(ctx, sink);
  sink = poly_graph_rewrite(ctx, sink, poly_pm_gep_pushing());
  poly_debug_stage_graph("remove reduce", sink);
  POLY_REWRITE_CHECK("remove reduce");

  /* 7. Add gpudims / CPU thread dims */
  if (opts.caps.has_threads)
    sink = poly_add_cpu_thread_dims(ctx, sink);
  else if (device_is_gpu(opts.device))
    sink = poly_add_gpudims_ex(ctx, sink, opts.caps);
  poly_debug_stage_graph("add gpudims", sink);
  POLY_REWRITE_CHECK("add gpudims");

  /* 8. Add loads */
  sink = poly_graph_rewrite(ctx, sink, poly_pm_add_loads());
  poly_debug_stage_graph("add loads", sink);
  POLY_REWRITE_CHECK("add loads");

  /* 9. Devectorize */
  g_max_fold_width = fold_width_from_caps(opts.caps);
  g_render_caps = opts.caps;
  if (opts.devectorize >= 0) {
    sink = poly_graph_rewrite(
        ctx, sink, (opts.devectorize >= 1) ? poly_pm_combined_devec() : poly_pm_combined_nodevec()
    );
  }
  poly_debug_stage_graph("devectorize", sink);
  POLY_REWRITE_CHECK("devectorize");

  /* 10. Lower index dtype */
  sink = poly_graph_rewrite(ctx, sink, poly_pm_post_index_lower());
  sink = poly_graph_rewrite(ctx, sink, poly_symbolic());
  poly_debug_stage_graph("lower index dtype", sink);
  POLY_REWRITE_CHECK("lower index dtype");

  /* 11. Decompositions */
  sink = poly_graph_rewrite(
      ctx, sink,
      poly_pm_decomp_with_caps(
          opts.caps.has_mulacc, opts.caps.has_max, opts.caps.has_threefry, opts.caps.has_fdiv
      )
  );
  if (!opts.caps.has_int64) {
    sink = poly_decompose_int64(ctx, sink);
    POLY_REWRITE_CHECK("decomp int64");
  }
  /* Pinned tinygrad uop/decompositions.py:557-562 and
   * codegen/__init__.py:116-140: unsupported dtype producer rules run
   * bottom-up after normal decompositions and before transcendental/final
   * renderer rewrites. Keep this distinct from extra_matcher. */
  if (opts.dtype_matcher) {
    sink = poly_graph_rewrite_ex(ctx, sink, opts.dtype_matcher, true);
    POLY_REWRITE_CHECK("dtype decompositions");
  }
  sink = poly_graph_rewrite(ctx, sink, poly_pm_transcendental(opts.caps));
  /* Pinned decompositions.py:445-458 lowers FLOORDIV to CDIV before
   * decompositions.py:329-380 emulates long arithmetic; l2i intentionally
   * accepts CDIV/CMOD rather than floor-semantics ops. Transcendental
   * expansion creates new FLOORDIV nodes, so rerun the existing late matcher
   * before optional long emulation. test_future_passes.c records the required
   * decomp -> transcendental -> decomp lifecycle. */
  sink = poly_graph_rewrite(
      ctx, sink,
      poly_pm_decomp_with_caps(
          opts.caps.has_mulacc, opts.caps.has_max, opts.caps.has_threefry, opts.caps.has_fdiv
      )
  );
  POLY_REWRITE_CHECK("decomp post-transcendental");
  if (!opts.caps.has_int64) {
    sink = poly_decompose_int64(ctx, sink);
    POLY_REWRITE_CHECK("decomp post-transcendental int64");
    PolyUOp *unsupported = NULL;
    if (graph_has_int64(ctx, sink, &unsupported)) {
      fprintf(
          stderr, "polygrad: int64 decomposition left unsupported %s for renderer\n",
          unsupported ? poly_op_name(unsupported->op) : "graph"
      );
      return NULL;
    }
  }
  /* Long decomposition can itself create ordinary late-rewrite candidates;
   * preserve the existing final cleanup pass. */
  sink = poly_graph_rewrite(
      ctx, sink,
      poly_pm_decomp_with_caps(
          opts.caps.has_mulacc, opts.caps.has_max, opts.caps.has_threefry, opts.caps.has_fdiv
      )
  );
  poly_debug_stage_graph("decompositions", sink);
  POLY_REWRITE_CHECK("decompositions");

  /* 12. Final rewrite */
  sink = poly_fold_full_scalar_load_runs(ctx, sink);
  ScalarLoadGroup *prev_scalar_load_groups = g_scalar_load_groups;
  int prev_n_scalar_load_groups = g_n_scalar_load_groups;
  g_scalar_load_groups = poly_collect_scalar_load_groups(ctx, sink, &g_n_scalar_load_groups);
  if (opts.devectorize >= 1 || opts.devectorize < 0)
    sink = poly_graph_rewrite(ctx, sink, poly_pm_render_subset());
  else if (opts.caps.has_simd_int)
    sink = poly_graph_rewrite(ctx, sink, poly_pm_render_subset_packed_int());
  else
    sink = poly_graph_rewrite(ctx, sink, poly_pm_render_subset_vec());
  free(g_scalar_load_groups);
  g_scalar_load_groups = prev_scalar_load_groups;
  g_n_scalar_load_groups = prev_n_scalar_load_groups;
  sink = poly_graph_rewrite(ctx, sink, poly_pm_index_is_shrink());
  sink = poly_graph_rewrite(ctx, sink, poly_pm_remove_vec_dtypes());
  sink = poly_graph_rewrite(ctx, sink, poly_pm_move_gates_from_index());
  /* Pinned tinygrad codegen/__init__.py:124-137 runs pm_render, new-style
   * index/vector rewrites, and gate movement before the final matcher
   * pm_decomp + renderer.extra_matcher + pm_split_ends. Gate movement may
   * introduce dtype-sensitive LOAD alternatives, so renderer legalization
   * cannot run before it. */
  sink = poly_graph_rewrite(
      ctx, sink,
      poly_pm_decomp_with_caps(
          opts.caps.has_mulacc, opts.caps.has_max, opts.caps.has_threefry, opts.caps.has_fdiv
      )
  );
  if (opts.extra_matcher) sink = poly_graph_rewrite(ctx, sink, opts.extra_matcher);
  /* A combined tinygrad matcher revisits nodes exposed by renderer rules.
   * Repeat the shared decompositions before splitting ENDs to preserve that
   * fixed-point behavior with Polygrad's separate matchers. */
  sink = poly_graph_rewrite(
      ctx, sink,
      poly_pm_decomp_with_caps(
          opts.caps.has_mulacc, opts.caps.has_max, opts.caps.has_threefry, opts.caps.has_fdiv
      )
  );
  sink = poly_graph_rewrite(ctx, sink, poly_pm_split_ends());
  poly_debug_stage_graph("final rewrite", sink);
  POLY_REWRITE_CHECK("final rewrite");

  /* 13. Control flow */
  sink = poly_apply_control_flow(ctx, sink);
  poly_debug_stage_graph("control flow", sink);
  POLY_REWRITE_CHECK("control flow");
  poly_debug_stage_graph("rewritten", sink);
  POLY_REWRITE_CHECK("rewritten");

#undef POLY_REWRITE_CHECK
  return sink;
}

PolyUOp *poly_full_rewrite_to_sink(PolyCtx *ctx, PolyUOp *sink) {
  PolyRewriteOpts opts = {
      .optimize = poly_kernel_optimize_enabled(sink),
      .devectorize = 1,
      .caps = poly_c_renderer_caps(),
      .device = POLY_DEVICE_CPU,
      .opt_policy = POLY_OPT_HEURISTIC,
      .dtype_matcher = poly_pm_bf16_non_native(),
      .extra_matcher = poly_pm_c_renderer_extra(),
  };
  return poly_full_rewrite_to_sink_ex(ctx, sink, opts);
}
