/*
 * codegen/codegen.c — current Tinygrad codegen pipeline
 *
 * Mirrors tinygrad's codegen/__init__.py full_rewrite_to_sink pipeline.
 * Currently implements:
 *   - pm_reduce: REDUCE → BUFFER(REG) + AFTER accumulation + END merge
 *   - pm_decomp: MAX→WHERE, MUL→SHL, IDIV→SHR (late decompositions)
 *   - pm_transcendental: EXP2 → polynomial approximation (xexp2)
 */

#define _POSIX_C_SOURCE 200809L

#include "codegen.h"
#include "codegen/decomp/dtype.h"
#include "codegen/late/coalesce.h"
#include "codegen/late/linearizer.h"
#include "codegen/late/gater.h"
#include "renderer/cstyle.h"
#include "bigint.h"
#include "engine/schedule.h"
#include "frontend_internal.h"
#include "schedule/indexing.h"
#include "schedule/multi.h"
#include "schedule/rangeify.h"
#include "codegen/simplify.h"
#include "uop/movement.h"
#include "uop/spec.h"
#include "uop/symbolic.h"
#include "uop/weak.h"
#include <limits.h>
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#include <errno.h>
#include "utils.h"

/* C storage for Tinygrad's NOOPT ContextVar. A Wasm module has its own copy;
 * native contexts share it, just as Tinygrad contexts share compiler policy. */
static int noopt_value;
static bool noopt_initialized;

int poly_get_noopt(void) {
  if (!noopt_initialized) {
    noopt_value = poly_getenv_int("NOOPT", 0);
    noopt_initialized = true;
  }
  return noopt_value;
}

void poly_set_noopt(int value) {
  noopt_value = value;
  noopt_initialized = true;
}
/* Max hardware vector fold width for load/store splitting.
 * Set by the pipeline before running correct_load_store pass.
 * Default 4 (SSE). Set to 8 for AVX2.
 * Thread-local because concurrent codegen with different backend caps must not
 * share vector width or renderer capability state. */

static _Thread_local int poly_trace_codegen_kernel = 0;

static void poly_debug_print_graph(FILE *fp, PolyUOp *u, const char *tag) {
  if (!fp || !u) return;
  fprintf(fp, "=== GRAPH %s ===\n", tag ? tag : "sink");
  poly_uop_dump_tree(fp, u, 0, 24);
  fprintf(fp, "=== END GRAPH %s ===\n", tag ? tag : "sink");
}

static void poly_debug_invalid_parents(const char *tag, PolyUOp *root) {
  int n_topo = 0;
  PolyUOp **topo = poly_toposort_alloc(NULL, root, &n_topo);
  for (int i = 0; topo && i < n_topo; i++) {
    if (topo[i]->op != POLY_OP_CONST || topo[i]->arg.kind != POLY_ARG_INVALID) continue;
    fprintf(stderr, "[polygrad:invalid_stage] %s", tag ? tag : "sink");
    for (int j = 0; j < n_topo; j++)
      for (int k = 0; k < topo[j]->n_src; k++)
        if (topo[j]->src[k] == topo[i]) {
          fprintf(stderr, " parent=%s[%d]", poly_op_name(topo[j]->op), k);
          for (int p = 0; p < n_topo; p++)
            for (int q = 0; q < topo[p]->n_src; q++)
              if (topo[p]->src[q] == topo[j])
                fprintf(stderr, "<-%s[%d]", poly_op_name(topo[p]->op), q);
        }
    fputc('\n', stderr);
  }
  poly_toposort_free(topo);
}

static void poly_debug_stage_json(const char *tag, PolyUOp *root) {
  int n_topo = 0;
  PolyUOp **topo = poly_toposort_alloc(NULL, root, &n_topo);
  fprintf(stderr, "PG_STAGE_JSON {\"name\":\"%s\",\"rows\":[", tag ? tag : "sink");
  for (int i = 0; topo && i < n_topo; i++) {
    PolyUOp *u = topo[i];
    fprintf(
        stderr, "%s{\"op\":\"%s\",\"dtype\":\"%s\",\"arg_kind\":%d,\"arg_hash\":%u,\"src\":[",
        i ? "," : "", poly_op_name(u->op), poly_dtype_name(u->dtype), (int)u->arg.kind,
        poly_arg_hash(u->arg)
    );
    for (int j = 0; j < u->n_src; j++) {
      int src_index = -1;
      for (int k = 0; k < n_topo; k++)
        if (topo[k] == u->src[j]) {
          src_index = k;
          break;
        }
      fprintf(stderr, "%s%d", j ? "," : "", src_index);
    }
    fputs("]}", stderr);
  }
  fputs("]}\n", stderr);
  poly_toposort_free(topo);
}

static void poly_debug_stage_graph(PolyCtx *ctx, const char *tag, PolyUOp *u) {
  if (tag && strcmp(tag, "input") == 0) poly_trace_codegen_kernel++;
  int trace_target = poly_getenv_int("POLY_TRACE_CODEGEN_KERNEL", 0);
  bool trace_this = trace_target == poly_trace_codegen_kernel;
  if (trace_this && u) poly_debug_invalid_parents(tag, u);
  if (trace_this && u && poly_getenv_int("POLY_TRACE_CODEGEN_JSON", 0))
    poly_debug_stage_json(tag, u);
  if (trace_this && tag &&
      (strcmp(tag, "input") == 0 || strcmp(tag, "load collapse") == 0 ||
       strcmp(tag, "split ranges") == 0 || strcmp(tag, "initial symbolic") == 0 ||
       strcmp(tag, "simplify ranges") == 0 || strcmp(tag, "apply opts") == 0 ||
       strcmp(tag, "postopt symbolic") == 0 || strcmp(tag, "add gpudims") == 0 ||
       strcmp(tag, "add loads") == 0 || strcmp(tag, "devectorize2") == 0 ||
       strcmp(tag, "lower all index dtypes") == 0 || strcmp(tag, "late decompositions") == 0 ||
       strcmp(tag, "move gates from index") == 0 || strcmp(tag, "final rewrite") == 0))
    poly_debug_print_graph(stderr, u, tag);
  if ((poly_debug_at_least(3) || trace_this) && u) {
    PolyMap *seen = poly_map_new(256);
    PolyUOp **stack = malloc(1024 * sizeof(PolyUOp *));
    int stack_cap = stack ? 1024 : 0;
    int stack_n = 0;
    int n = 0;
    int n_index = 0, n_weak_range = 0, n_weak_alu = 0;
    int n_where = 0, n_load = 0, n_store = 0, n_reduce = 0, n_range = 0;
    int64_t max_numel = 1;
    PolyUOp *max_numel_uop = NULL;
    if (seen && stack) stack[stack_n++] = u;
    while (stack_n > 0) {
      PolyUOp *x = stack[--stack_n];
      if (!x || poly_map_get(seen, poly_ptr_hash(x), x, poly_ptr_eq)) continue;
      poly_map_set(seen, poly_ptr_hash(x), x, x, poly_ptr_eq);
      n++;
      n_where += x->op == POLY_OP_WHERE;
      n_load += x->op == POLY_OP_LOAD;
      n_store += x->op == POLY_OP_STORE;
      n_reduce += x->op == POLY_OP_REDUCE;
      n_range += x->op == POLY_OP_RANGE;
      if (poly_debug_at_least(6)) {
        fprintf(
            stderr, "[polygrad:codegen_node] %s node=%p op=%s dtype=%s numel=%lld",
            tag ? tag : "sink", (void *)x, poly_op_name(x->op), poly_dtype_name(x->dtype),
            (long long)poly_uop_max_numel(ctx, x)
        );
        for (int i = 0; i < x->n_src; i++)
          fprintf(stderr, " src%d=%p", i, (void *)x->src[i]);
        if (x->arg.kind == POLY_ARG_INT)
          fprintf(stderr, " arg=%lld", (long long)x->arg.i);
        else if (x->arg.kind == POLY_ARG_INT_TUPLE) {
          fputs(" arg=(", stderr);
          for (int i = 0; i < x->arg.int_tuple.n; i++)
            fprintf(stderr, "%s%lld", i ? "," : "", (long long)x->arg.int_tuple.vals[i]);
          fputc(')', stderr);
        }
        fprintf(stderr, "\n");
      }
      if (poly_dtype_is_index(x->dtype)) {
        n_index++;
        if (poly_debug_at_least(4)) {
          fprintf(
              stderr, "[polygrad:codegen_stage] %s weak=%p op=%s numel=%lld n_src=%d",
              tag ? tag : "sink", (void *)x, poly_op_name(x->op),
              (long long)poly_uop_max_numel(ctx, x), x->n_src
          );
          for (int i = 0; i < x->n_src; i++)
            fprintf(
                stderr, " src%d=%s/%s/%lld", i, poly_op_name(x->src[i]->op),
                poly_dtype_name(x->src[i]->dtype), (long long)poly_uop_max_numel(ctx, x->src[i])
            );
          fprintf(stderr, "\n");
        }
        if (x->op == POLY_OP_RANGE)
          n_weak_range++;
        else if (poly_opset_has(POLY_GROUP_ALU, x->op))
          n_weak_alu++;
      }
      int64_t x_numel = poly_uop_max_numel(ctx, x);
      if (x_numel > max_numel) {
        max_numel = x_numel;
        max_numel_uop = x;
      }
      for (int i = 0; i < x->n_src; i++) {
        if (poly_debug_at_least(4) && poly_dtype_is_index(x->src[i]->dtype))
          fprintf(
              stderr, "[polygrad:codegen_stage] %s weak_src=%p parent=%p/%s/%s/%lld edge=%d\n",
              tag ? tag : "sink", (void *)x->src[i], (void *)x, poly_op_name(x->op),
              poly_dtype_name(x->dtype), (long long)poly_uop_max_numel(ctx, x), i
          );
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
        stderr,
        "[polygrad:codegen_stage] %s nodes=%d where=%d load=%d store=%d reduce=%d range=%d "
        "weakint=%d weak_range=%d weak_alu=%d\n",
        tag ? tag : "sink", n, n_where, n_load, n_store, n_reduce, n_range, n_index, n_weak_range,
        n_weak_alu
    );
    if (max_numel_uop && max_numel > 1) {
      fprintf(
          stderr, "[polygrad:codegen_stage] %s max_numel=%lld max_op=%s\n", tag ? tag : "sink",
          (long long)max_numel, poly_op_name(max_numel_uop->op)
      );
    }
    fflush(stderr);
  }
  if (poly_dump_graph_enabled()) poly_debug_print_graph(stderr, u, tag);
}

/* Tinygrad 2026-08-22/a9069c177a9d codegen/__init__.py:222-225. */
static PolyUOp *poly_expand_horizontal_reduce(
    PolyCtx *ctx,
    PolyUOp *r,
    const PolyBindings *bindings
) {
  (void)bindings;
  if (!ctx || !r || r->op != POLY_OP_REDUCE || r->n_src < 1 || r->arg.kind != POLY_ARG_REDUCE)
    return NULL;
  PolyUOp *inp = r->src[0];
  int n_axes = r->arg.reduce.num_axes;
  PolyShape shape = poly_uop_max_shape_cached(ctx, inp);
  if (n_axes < 0 || shape.ndim < 0 || n_axes > shape.ndim || n_axes > POLY_MAX_DIMS) return NULL;
  if (n_axes == 0) return inp;

  size_t n_terms = 1;
  for (int axis = 0; axis < n_axes; axis++) {
    if (shape.dims[axis] <= 0 || (size_t)shape.dims[axis] > SIZE_MAX / n_terms) return NULL;
    n_terms *= (size_t)shape.dims[axis];
  }

  PolyUOp *ret = NULL;
  for (size_t linear = 0; linear < n_terms; linear++) {
    size_t rem = linear;
    PolyUOp *indices[POLY_MAX_DIMS];
    for (int axis = n_axes - 1; axis >= 0; axis--) {
      int64_t coord = (int64_t)(rem % (size_t)shape.dims[axis]);
      rem /= (size_t)shape.dims[axis];
      indices[axis] = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(coord));
      if (!indices[axis]) return NULL;
    }
    PolyUOp *term = poly_uop_index(ctx, inp, indices, n_axes);
    if (!term) return NULL;
    ret = ret ? poly_uop2(ctx, r->arg.reduce.op, r->dtype, ret, term, poly_arg_none()) : term;
    if (!ret) return NULL;
  }
  return ret;
}

/* pm_reduce: REDUCE → BUFFER(REG) + END merge */

/* Heuristic optimizer (port of tinygrad hand_coded_optimizations) */

/* axis_to_pos ordering: matches tinygrad's axis_to_pos dict.
 * DEVICE:-2, WEAK/LOOP:-1, THREAD/GLOBAL:0, WARP:1, LOCAL/GROUP_REDUCE:2,
 * UPCAST:3, REDUCE:4, UNROLL:5 */
static int axis_to_pos(PolyAxisType t) {
  switch (t) {
  case POLY_AXIS_DEVICE:
    return -2;
  case POLY_AXIS_WEAK:
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
  /* Tinygrad codegen/opt/postrange.py:Scheduler.rngs sorts by
   * (axis_to_pos(axis_type), axis_id, *split_path). */
  int pa = axis_to_pos(poly_range_axis_type(a->arg));
  int pb = axis_to_pos(poly_range_axis_type(b->arg));
  if (pa != pb) return pa - pb;
  int64_t ia = poly_range_axis_id(a->arg);
  int64_t ib = poly_range_axis_id(b->arg);
  if (ia != ib) return (ia < ib) ? -1 : 1;
  int na = poly_range_n_extra(a->arg), nb = poly_range_n_extra(b->arg);
  int n = na < nb ? na : nb;
  const int64_t *ea = poly_range_extra(a->arg), *eb = poly_range_extra(b->arg);
  for (int i = 0; i < n; i++) {
    if (ea[i] != eb[i]) return ea[i] < eb[i] ? -1 : 1;
  }
  if (na != nb) return na < nb ? -1 : 1;
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
static PolyUOp *convert_loop_to_global(PolyCtx *ctx, PolyUOp *ast) {
  if (!ctx || !ast) return ast;

  int n_topo = 0;
  PolyUOp **topo = poly_toposort_alloc(ctx, ast, &n_topo);
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
    if (poly_range_axis_type(r->arg) != POLY_AXIS_WEAK) continue;

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

  PolyUOp *out = (n_sub > 0) ? poly_uop_substitute(ctx, ast, from, to, n_sub) : ast;
  poly_toposort_free(topo);
  return out;
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
  PolyUOp *idx = poly_uop_get_idx(ctx, coord);
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
    if (u->op == POLY_OP_REDUCE) s->has_reduce = true;
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
        s->buf_reach[bi] = projected_index_reachability(ctx, s->bufs[bi]->src[1], idx_map, reach);
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
    if (u->op == POLY_OP_REDUCE) s->has_reduce = true;
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
        s->buf_reach[bi] =
            projected_index_reachability(s->ctx, s->bufs[bi]->src[1], idx_map, reach);
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

/* Helper: get indices of upcastable dims (GLOBAL/LOCAL/WEAK with size > 1) */
static int sched_upcastable_dims(const OptScheduler *s, int *out, int max_n) {
  int n = 0;
  for (int i = 0; i < s->n_rngs && n < max_n; i++) {
    PolyAxisType t = s->types[i];
    if ((t == POLY_AXIS_GLOBAL || t == POLY_AXIS_LOCAL || t == POLY_AXIS_WEAK) && s->shape[i] > 1)
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
  case POLY_AXIS_WEAK:
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
    int rs = poly_range_start(cur->op);
    int end = (rs >= 0) ? rs : cur->n_src;
    for (int i = 0; i < end && sp < 510; i++)
      stack[sp++] = cur->src[i];
  }
  poly_map_destroy(visited);
  return mask;
}

/* Forward declarations */
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
 * tc_opt: 0 = one reduce axis, 1 = multiple reduce axes,
 *         2 = allow PADTO (not yet implemented). */
static bool sched_apply_tc_opt(
    OptScheduler *s,
    int axis,
    int tc_select,
    int tc_opt,
    int use_tc,
    const char *device,
    const PolyTensorCore *tcs,
    int n_tcs,
    PolyUOp *tc_axes_out[3]
) {
  PolyCtx *ctx = s->ctx;

  /* 1. Find REDUCE(ADD) and its MUL (postrange.py:222-227) */
  int n_topo = 0;
  PolyUOp **topo = poly_toposort_alloc(ctx, s->ast, &n_topo);
  PolyUOp *reduceop = NULL;
  for (int i = 0; i < n_topo; i++) {
    if (topo[i]->op == POLY_OP_REDUCE && topo[i]->arg.kind == POLY_ARG_REDUCE &&
        topo[i]->arg.reduce.op == POLY_OP_ADD) {
      reduceop = topo[i];
      break;
    }
  }
  poly_toposort_free(topo);
  if (!reduceop || !use_tc) return false;

  PolyUOp *mul = reduceop->src[0];
  /* Tinygrad 2026-08-22/a9069c177a9d postrange.py:217-220 unwraps the
   * reduction product CAST independently of TC_OPT. */
  if (mul->op == POLY_OP_CAST && mul->n_src > 0) mul = mul->src[0];
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

    /* Tinygrad 2026-08-22/a9069c177a9d codegen/opt/postrange.py:226 keeps
     * CUDA TF32 tensor cores disabled unless ALLOW_TF32 is explicit. */
    if (device && (strcmp(device, "CUDA") == 0 || strcmp(device, "NV") == 0) &&
        poly_dtype_eq(tc->dtype_in, POLY_FLOAT32) && !poly_getenv_flag("ALLOW_TF32"))
      continue;

    /* Check dtype match */
    PolyDType in0_scalar = in0->dtype;
    PolyDType in1_scalar = in1->dtype;
    PolyDType red_scalar = reduceop->dtype;
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
    int rs = poly_range_start(POLY_OP_REDUCE);
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
      PolyUOp **topo2 = poly_toposort_alloc(ctx, s->ast, &n_topo2);

      /* Debug: check how many ne[] pointers are present in the current AST */
      PolyUOp *found_red = NULL;
      for (int i = 0; i < n_topo2; i++) {
        if (topo2[i]->op == POLY_OP_REDUCE && topo2[i]->tag == TC_TAG) {
          found_red = topo2[i];
          break;
        }
      }
      poly_toposort_free(topo2);
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

      /* Current postrange.py:297-301 keeps every input fragment axis in all
       * three metadata rows. Missing input axes are broadcast with size one. */
      for (int src_dim = 0; src_dim < 2; src_dim++) {
        for (int i = 0; i < n_upcast[src_dim]; i++) {
          int64_t axis_id = upcast_pairs[src_dim][i][0];
          for (int dim = 0; dim < 3; dim++) {
            bool found = false;
            for (int j = 0; j < n_upcast[dim]; j++)
              if (upcast_pairs[dim][j][0] == axis_id) {
                found = true;
                break;
              }
            if (!found && n_upcast[dim] < 16) {
              upcast_pairs[dim][n_upcast[dim]][0] = axis_id;
              upcast_pairs[dim][n_upcast[dim]][1] = 1;
              n_upcast[dim]++;
            }
          }
        }
      }

      /* Zero accumulator */
      PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, tc->dtype_out, poly_arg_float(0.0));
      PolyUOp *zero_elems[16];
      for (int i = 0; i < tc->elements_per_thread[2] && i < 16; i++)
        zero_elems[i] = zero;
      PolyUOp *zero_stack = poly_uop_stack(ctx, zero_elems, tc->elements_per_thread[2]);
      int64_t(*upcast_ptrs[3])[2] = {upcast_pairs[0], upcast_pairs[1], upcast_pairs[2]};
      PolyUOp *wmma_srcs[3] = {srcs[0], srcs[1], zero_stack};
      PolyUOp *tc_uop = poly_uop(
          ctx, POLY_OP_WMMA, tc->dtype_out, wmma_srcs, 3,
          poly_arg_tensor_core(
              tc->dims, tc->dtype_in, device ? device : "?", tc->threads, upcast_ptrs, n_upcast,
              true
          )
      );

      /* Preserve extra reduce ranges not consumed by TC (postrange.py:309-310) */
      int rs2 = poly_range_start(POLY_OP_REDUCE);
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
        PolyArg red_arg = poly_arg_reduce(POLY_OP_ADD, 0);
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
    PolyUOp **topo = poly_toposort_alloc(ctx, sink, &n_topo);
    bool has_stage = false;
    for (int i = 0; i < n_topo; i++) {
      if (topo[i]->op == POLY_OP_STAGE) {
        has_stage = true;
        break;
      }
    }
    poly_toposort_free(topo);
    if (has_stage) return sink;
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
    int tc_opt_env = poly_getenv_int("TC_OPT", 0);
    int use_tc_env = poly_getenv_int("TC", 1);

    if (use_tc_env > 0 && (n_reduce == 1 || tc_opt_env >= 1)) {
      OptScheduler tk;
      sched_copy(&tk, &s);
      PolyUOp *tc_axes[3];
      bool tc_ok = sched_apply_tc_opt(
          &tk, 0, -1, tc_opt_env, use_tc_env, caps.device, caps.tensor_cores, caps.n_tensor_cores,
          tc_axes
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

  /* == Matvec reduction (pinned tinygrad heuristic.py:60-80) ==
   * Polygrad's has_local capability covers both workgroup axes and the local
   * storage used by GROUP_REDUCE; current renderers do not expose those
   * capabilities independently. */
  int mv_blocksize = poly_getenv_int("MV_BLOCKSIZE", 4);
  int mv_threads_per_row = poly_getenv_int("MV_THREADS_PER_ROW", 8);
  int mv_rows_per_thread = poly_getenv_int("MV_ROWS_PER_THREAD", 4);
  if (caps.has_local && poly_getenv_int("MV", 1) != 0 &&
      (mv_blocksize > 1 || mv_threads_per_row > 1 || mv_rows_per_thread > 1) && s.n_rngs >= 2) {
    int n_topo = 0;
    PolyUOp **topo = poly_toposort_alloc(ctx, s.ast, &n_topo);
    PolyUOp *reduceop = NULL;
    for (int i = 0; i < n_topo; i++) {
      PolyUOp *u = topo[i];
      if (u && u->op == POLY_OP_REDUCE && u->arg.kind == POLY_ARG_REDUCE &&
          u->arg.reduce.op == POLY_OP_ADD) {
        reduceop = u;
        break;
      }
    }

    PolyUOp *mul = reduceop && reduceop->n_src > 0 && reduceop->src[0]->op == POLY_OP_MUL &&
                           reduceop->src[0]->n_src == 2
                       ? reduceop->src[0]
                       : NULL;
    PolyUOp *idx0 = mul && mul->src[0]->op == POLY_OP_INDEX && mul->src[0]->n_src >= 2
                        ? poly_uop_get_idx(ctx, mul->src[0]->src[1])
                        : NULL;
    PolyUOp *idx1 = mul && mul->src[1]->op == POLY_OP_INDEX && mul->src[1]->n_src >= 2
                        ? poly_uop_get_idx(ctx, mul->src[1]->src[1])
                        : NULL;
    int first_reduce = -1;
    for (int i = 0; i < s.n_rngs; i++) {
      if (s.types[i] == POLY_AXIS_REDUCE) {
        first_reduce = i;
        break;
      }
    }

    bool reduce_is_addend = false;
    if (idx0 && first_reduce >= 0) {
      PolyUOp *terms[256];
      int n_terms = split_uop_add(idx0, terms, 256);
      for (int i = 0; i < n_terms; i++) {
        if (terms[i] == s.rngs[first_reduce]) {
          reduce_is_addend = true;
          break;
        }
      }
    }
    bool second_covers_first = idx0 && idx1;
    for (int i = 0; second_covers_first && i < s.n_rngs; i++) {
      if (poly_uop_in_ranges(ctx, idx0, s.rngs[i]) && !poly_uop_in_ranges(ctx, idx1, s.rngs[i]))
        second_covers_first = false;
    }

    if (reduce_is_addend && second_covers_first) {
      for (int global_idx = 0; global_idx < s.n_rngs; global_idx++) {
        if (s.types[global_idx] != POLY_AXIS_GLOBAL) continue;
        int64_t reduce_size = s.shape[first_reduce];
        int64_t global_size = s.shape[global_idx];
        if (mv_threads_per_row <= 0 || mv_blocksize <= 0 || mv_rows_per_thread <= 0 ||
            reduce_size % mv_threads_per_row != 0 ||
            global_size % ((int64_t)mv_blocksize * mv_rows_per_thread) != 0)
          continue;

        if (mv_threads_per_row > 1)
          sched_shift_to(
              &s, s.rngs[first_reduce], mv_threads_per_row, POLY_AXIS_GROUP_REDUCE, false
          );
        if (mv_blocksize > 1)
          sched_shift_to(&s, s.rngs[global_idx], mv_blocksize, POLY_AXIS_LOCAL, false);
        if (mv_rows_per_thread > 1)
          sched_shift_to(&s, s.rngs[global_idx], mv_rows_per_thread, POLY_AXIS_UPCAST, false);
        poly_toposort_free(topo);
        return s.ast;
      }
    }
    poly_toposort_free(topo);
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
    PolyUOp **topo = poly_toposort_alloc(ctx, s.ast, &n_topo);

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
    poly_toposort_free(topo);

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
            PolyUOp *idx_expr = poly_uop_get_idx(ctx, idx_uop->src[1]);
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
      if (!(s.types[axis] == POLY_AXIS_GLOBAL || s.types[axis] == POLY_AXIS_WEAK)) continue;
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
        if ((s.types[axis] == POLY_AXIS_GLOBAL || s.types[axis] == POLY_AXIS_WEAK) &&
            s.shape[axis] > 1)
          sched_shift_to(&s, s.rngs[axis], to_local[i].size, POLY_AXIS_LOCAL, false);
        if (will_delete_shape) deleted_shape++;
      }
    }
  }

  /* == CPU THREAD axis (tinygrad heuristic.py:180-190) ==
   * ClangRenderer has has_threads=true, then gpudims.py replaces the THREAD
   * axis with the runtime ALU PARAM "core_id". Keep this after local grouping
   * just like tinygrad's final heuristic block. */
  if (caps.has_threads && caps.max_threads > 1 && !sched_has_axis_type(&s, POLY_AXIS_THREAD)) {
    int candidates[] = {32, 16, 12, 8, 6, 5, 4, 3, 2};
    int64_t full_prod = sched_full_shape_prod(&s);
    for (int ci = 0; ci < (int)(sizeof(candidates) / sizeof(candidates[0])); ci++) {
      int threads = candidates[ci];
      if (threads > caps.max_threads) continue;
      if (full_prod / (128LL << 10) < threads) continue;
      for (int axis = 0; axis < s.n_rngs; axis++) {
        if (s.types[axis] != POLY_AXIS_WEAK) continue;
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

typedef struct {
  PolyOptOps op;
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

static void beam_free_args(void **bufs, int n_params) {
  if (bufs)
    for (int i = 0; i < n_params; i++)
      free(bufs[i]);
  free(bufs);
}

static void **beam_args_from_ast(PolyCtx *ctx, PolyUOp *sink, int *n_args) {
  *n_args = 0;
  /* opt/postrange.py:args_from_ast, adapted to the existing C wrapper's
   * compact buffer-then-scalar arguments and implicit core_id. */
  int n_topo = 0;
  PolyUOp **topo = poly_toposort_alloc(ctx, sink, &n_topo);
  int n_params = 0;
  void **bufs = NULL;
  if (!topo) goto cleanup;
  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    if (u->op != POLY_OP_PARAM) continue;
    if (u->arg.kind != POLY_ARG_PARAM || !u->arg.param) goto cleanup;
    const char *name = poly_uop_expr(u);
    if (poly_uop_is_alu_param(u) && name && strcmp(name, "core_id") == 0) continue;
    topo[n_params++] = u;
  }
  for (int i = 1; i < n_params; i++) {
    PolyUOp *param = topo[i];
    bool scalar = poly_uop_is_alu_param(param);
    int j = i;
    while (j > 0) {
      bool prior_scalar = poly_uop_is_alu_param(topo[j - 1]);
      if (prior_scalar < scalar || (prior_scalar == scalar && poly_program_buffer_slot(topo[j - 1]
                                                              ) <= poly_program_buffer_slot(param)))
        break;
      topo[j] = topo[j - 1];
      j--;
    }
    topo[j] = param;
  }
  /* search._time_program passes the complete rawbufs list. The rendered
   * wrapper reads every argument; silently truncating it corrupts memory. */
  bufs = calloc((size_t)(n_params ? n_params : 1), sizeof(*bufs));
  if (!bufs) goto cleanup;
  for (int i = 0; i < n_params; i++) {
    PolyUOp *param = topo[i];
    bool scalar = poly_uop_is_alu_param(param);
    /* The existing native wrapper takes all ALU values through int*, even
     * when its typed function subsequently converts to another dtype. */
    int itemsize = scalar ? (int)sizeof(int) : poly_dtype_itemsize(param->dtype);
    int64_t sz = scalar ? 1 : poly_uop_max_numel(ctx, param);
    if (sz < 0 || itemsize <= 0 || (uint64_t)sz > SIZE_MAX / (size_t)itemsize) goto cleanup;
    bufs[i] = calloc((size_t)(sz ? sz : 1), (size_t)itemsize);
    if (!bufs[i]) goto cleanup;
    if (scalar) {
      const PolyParamArg *arg = param->arg.param;
      if (!arg->has_minmax || arg->min_val > arg->max_val ||
          (!poly_dtype_is_int(param->dtype) && !poly_dtype_eq(param->dtype, POLY_BOOL)))
        goto cleanup;
      /* Python's (lo+hi)//2, without overflowing the C sum. */
      int64_t value =
          arg->min_val + (int64_t)(((uint64_t)arg->max_val - (uint64_t)arg->min_val) / 2);
      if (value < INT_MIN || value > INT_MAX) goto cleanup;
      *(int *)bufs[i] = (int)value;
    } else if (poly_dtype_eq(param->dtype, POLY_FLOAT32)) {
      /* Preserve existing finite f32 timing inputs; other storage is zeroed. */
      float *fp = bufs[i];
      for (int64_t j = 0; j < sz; j++)
        fp[j] = 0.1f + (float)(j % 100) * 0.01f;
    }
  }
  poly_toposort_free(topo);

  *n_args = n_params;
  return bufs;
cleanup:
  poly_toposort_free(topo);
  beam_free_args(bufs, n_params);
  return NULL;
}

/* Compile a kernel AST through the full post-optimization pipeline,
 * render to C, compile with clang, allocate test buffers, and time execution.
 * Returns median time in microseconds. Returns INFINITY on failure. */
#ifdef POLY_TESTING
static int beam_test_parameter_count;
#endif
static double beam_compile_and_time(PolyCtx *ctx, PolyUOp *sink, PolyRewriteOpts opts, int reps) {
  /* search._try_compile calls Scheduler.get_optimized_ast: shift_to can
   * leave END operands as expressions, which must become ranges before
   * the post-optimization pipeline verifies its input. */
  sink = poly_graph_rewrite(ctx, sink, poly_pm_flatten_range());
  if (!sink) return INFINITY;
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
  PolyUOp **uops = poly_do_linearize(ctx, sink, &n_uops);
  if (!uops || n_uops == 0) return INFINITY;

  /* UOp count filter: skip huge kernels */
  if (n_uops > 3000) {
    free(uops);
    return INFINITY;
  }

  /* Render C */
  char fn_name[64];
  snprintf(fn_name, sizeof(fn_name), "beam_%d", (int)(uintptr_t)sink & 0xFFFF);
  char *source = poly_render_c(ctx, uops, n_uops, fn_name);
  free(uops);
  if (!source) return INFINITY;

  /* Compile */
  PolyProgram *prog = poly_compile_c(source, fn_name);
  free(source);
  if (!prog) return INFINITY;

  int n_params = 0;
  void **bufs = beam_args_from_ast(ctx, sink, &n_params);
  if (!bufs) {
    poly_program_destroy(prog);
    return INFINITY;
  }
  double median;

  /* Warm up */
#ifdef POLY_TESTING
  beam_test_parameter_count = n_params;
#endif
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
  median = times[reps / 2];

  beam_free_args(bufs, n_params);
  poly_program_destroy(prog);

  return median;
}

/* Disk cache for BEAM results */
#ifdef POLY_TESTING
void **poly_test_beam_args_from_ast(PolyCtx *ctx, PolyUOp *sink, int *n_args) {
  return beam_args_from_ast(ctx, sink, n_args);
}

double poly_test_beam_compile_and_time(
    PolyCtx *ctx,
    PolyUOp *sink,
    PolyRewriteOpts opts,
    int reps,
    int *n_args
) {
  beam_test_parameter_count = 0;
  double elapsed = beam_compile_and_time(ctx, sink, opts, reps);
  *n_args = beam_test_parameter_count;
  return elapsed;
}
#endif

/* FNV-1a hash over the AST toposort (structural hash for cache key) */
static uint64_t beam_ast_hash(PolyCtx *ctx, PolyUOp *sink) {
  int n_topo = 0;
  PolyUOp **topo = poly_toposort_alloc(ctx, sink, &n_topo);
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
  poly_toposort_free(topo);
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
    if (op_byte != POLY_OPT_UPCAST && op_byte != POLY_OPT_UNROLL) {
      fclose(f);
      return false;
    }
    actions[i] = (PolyBeamAction){.op = (PolyOptOps)op_byte, .axis = axis_byte, .amount = amount};
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

/* Current tinygrad codegen/__init__.py::reduce_ranges_to_acc. */
static PolyUOp *rule_reduce_to_acc(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  PolyUOp *red = root;
  if (!red || red->arg.kind != POLY_ARG_REDUCE || red->n_src < 1) return NULL;
  PolyUOp *inp = red->src[0];
  PolyOps reduce_op = red->arg.reduce.op;

  /* Filter reduce ranges to actual RANGE nodes only.
   * Singleton dims may produce CONST(0) pseudo-ranges from rangeify;
   * these must not enter the AFTER/END chains (tinygrad invariant). */
  PolyUOp *reduce_ranges[POLY_MAX_DIMS];
  int n_reduce_range = 0;
  for (int j = 1; j < red->n_src; j++) {
    if (red->src[j]->op == POLY_OP_RANGE) reduce_ranges[n_reduce_range++] = red->src[j];
  }

  /* Horizontal-only reduce (no loop ranges). */
  if (n_reduce_range == 0) return NULL;

  /* A range reduction can also retain a leading shaped prefix after
   * expander2. Tinygrad reduces that prefix before the loop accumulator. */
  PolyUOp *reduced_inp =
      red->arg.reduce.num_axes ? poly_expand_horizontal_reduce(ctx, red, NULL) : inp;
  if (!reduced_inp) return NULL;

  /* Find input_ranges (outer loops the value depends on) */
  int n_topo = 0;
  PolyUOp **topo = poly_toposort_alloc(ctx, inp, &n_topo);

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
  poly_toposort_free(topo);

  /* Identity element */
  PolyUOp *identity = poly_identity_element(ctx, reduce_op, red->dtype);
  if (!identity) return NULL;

  ReduceContext *rctx = current_reduce_ctx();
  if (!rctx) return NULL;

  /* placeholder_like flattens the REDUCE result shape into one strong-dtype
   * REG buffer. Scalar results therefore own one element. */
  int acc_id = rctx->acc_num++;
  PolyShape red_shape = poly_uop_max_shape_cached(ctx, red);
  int64_t acc_size = poly_shape_numel(red_shape);
  if (acc_size < 0) return NULL;
  PolyDType acc_dtype = poly_dtype_strong(red->dtype);
  PolyUOp *shape = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(acc_size));
  PolyParamArg acc_param = {.slot = acc_id, .addrspace = POLY_ADDR_REG};
  PolyUOp *acc = poly_uop1(ctx, POLY_OP_BUFFER, acc_dtype, shape, poly_arg_param(&acc_param));
  if (acc && red_shape.ndim > 1) acc = poly_reshape(ctx, acc, red_shape.dims, red_shape.ndim);
  if (!acc) return NULL;

  /* Init: acc.after(input_ranges...).store(identity) */
  PolyUOp *acc_base;
  if (n_input_ranges > 0) {
    PolyUOp *after_srcs[POLY_MAX_DIMS + 1];
    after_srcs[0] = acc;
    for (int i = 0; i < n_input_ranges; i++)
      after_srcs[i + 1] = input_ranges[i];
    acc_base =
        poly_uop(ctx, POLY_OP_AFTER, acc_dtype, after_srcs, n_input_ranges + 1, poly_arg_none());
  } else {
    acc_base = acc;
  }
  PolyUOp *acc_init = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, acc_base, identity, poly_arg_none());

  /* pm_add_loads turns this REG-backed value into LOAD after dimensions exist. */
  PolyUOp *loop_srcs[POLY_MAX_DIMS + 2];
  loop_srcs[0] = acc;
  loop_srcs[1] = acc_init;
  for (int i = 0; i < n_reduce_range; i++)
    loop_srcs[i + 2] = reduce_ranges[i];
  PolyUOp *loop_after =
      poly_uop(ctx, POLY_OP_AFTER, acc_dtype, loop_srcs, n_reduce_range + 2, poly_arg_none());

  /* Accumulate one shaped/horizontal-reduced value per loop iteration. */
  PolyUOp *alu = poly_uop2(ctx, reduce_op, red->dtype, loop_after, reduced_inp, poly_arg_none());

  PolyUOp *acc_store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, loop_after, alu, poly_arg_none());

  /* Build END chain (innermost first to match tinygrad) */
  PolyUOp *chain = acc_store;
  for (int i = n_reduce_range - 1; i >= 0; i--) {
    PolyUOp *end_srcs[2] = {chain, reduce_ranges[i]};
    chain = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_srcs, 2, poly_arg_none());
  }
  reduce_ctx_add_end(rctx, reduce_ranges, n_reduce_range, chain);

  /* Final read: acc.after(end); LOAD is added by pm_add_loads. */
  PolyUOp *final_srcs[2] = {acc, chain};
  PolyUOp *final_after = poly_uop(ctx, POLY_OP_AFTER, acc_dtype, final_srcs, 2, poly_arg_none());

  return final_after;
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
  PolyUOp **topo = poly_toposort_alloc(ctx, root, &n_topo);
  int64_t next_axis = 0;
  for (int i = 0; i < n_topo; i++) {
    if (topo[i] && topo[i]->op == POLY_OP_RANGE && topo[i]->arg.kind == POLY_ARG_RANGE) {
      int64_t axis = poly_range_axis_id(topo[i]->arg);
      if (axis >= next_axis) next_axis = axis + 1;
    }
  }
  poly_toposort_free(topo);

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
          /* Pinned devectorizer.py:338 applies the complete r -> tr map in
           * one e.substitute(dict(zip(r, tr))) traversal.  The shared
           * substitution API is likewise unbounded and preserves metadata. */
          PolyUOp *mapped_out = NULL;
          if (poly_uop_substitute_many(
                  ctx, &mapped, 1, g->ranges, mapped_ranges, g->n_ranges, &mapped_out
              ) != 0 ||
              !mapped_out) {
            free(mapped_ends);
            free(ctx_ranges);
            free(ctx_n);
            free(ctx_group);
            free(sub_new);
            free(sub_old);
            return NULL;
          }
          mapped = mapped_out;
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

  /* Tinygrad 2026-08-22/a9069c177a9d codegen/__init__.py:190-208 builds one complete
   * replacement dictionary and calls sink.substitute(subs) once. Sequential
   * substitution is not equivalent: the first traversal rebuilds sibling END
   * nodes, invalidating the original pointer keys of later rows. */
  PolyUOp *out = poly_uop_substitute(ctx, root, sub_old, sub_new, at);

  free(sub_new);
  free(sub_old);
  return out != root ? out : NULL;
}

/* Current Tinygrad codegen/decomp/op.py decomposition callbacks. */

/*
 * rule_decomp_max — get_simplifying_rewrite_patterns MAX rule.
 * MAX(a, b) → WHERE(CMPLT(a, b), b, a)
 * ClangRenderer doesn't have native MAX, so decompose to CMPLT+WHERE.
 */
static PolyUOp *rule_decomp_max(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  PolyUOp *a = root->src[0];
  PolyUOp *cmp = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, a, root->src[1], poly_arg_none());
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
  PolyUOp *shift_const = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(shift));
  return poly_uop2(ctx, POLY_OP_SHL, root->dtype, x_node, shift_const, poly_arg_none());
}

/*
 * rule_cdiv_to_shr — Current get_late_rewrite_patterns CDIV→SHR rule.
 * CDIV(x, c) → SHR(x, log2(c)) when c is a power of two.
 * For signed ints: (x + (x<0).where(c-1, 0)) >> log2(c)
 */
static PolyUOp *rule_cdiv_to_shr(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  PolyUOp *c_node = poly_bind(b, "c");
  PolyUOp *x_node = poly_bind(b, "x");
  if (!c_node || !x_node) return NULL;
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
  PolyUOp *shift_const = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(shift));
  /* Unsigned: just shift right */
  if (poly_dtype_is_unsigned(root->dtype))
    return poly_uop2(ctx, POLY_OP_SHR, root->dtype, x_node, shift_const, poly_arg_none());
  /* Signed: (x + (x<0).where(c-1, 0)) >> shift */
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, root->dtype, poly_arg_int(0));
  PolyUOp *cmplt = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, x_node, zero, poly_arg_none());
  PolyUOp *cm1 = poly_uop0(ctx, POLY_OP_CONST, root->dtype, poly_arg_int(c - 1));
  PolyUOp *correction =
      poly_uop3(ctx, POLY_OP_WHERE, root->dtype, cmplt, cm1, zero, poly_arg_none());
  PolyUOp *corrected =
      poly_uop2(ctx, POLY_OP_ADD, root->dtype, x_node, correction, poly_arg_none());
  return poly_uop2(ctx, POLY_OP_SHR, root->dtype, corrected, shift_const, poly_arg_none());
}

static int power_of_two_shift(int64_t value) {
  if (value <= 0 || (value & (value - 1)) != 0) return -1;
  int shift = 0;
  while (value > 1) {
    value >>= 1;
    shift++;
  }
  return shift;
}

/* Current get_simplifying_rewrite_patterns runs this before generic floor
 * decomposition: arithmetic right shift is signed floor division by 2**n. */
static PolyUOp *floordiv_pow2_to_shr(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  if (!root || root->n_src != 2) return NULL;
  PolyDType scalar = root->dtype;
  if (!poly_dtype_is_int(scalar) || poly_dtype_is_index(scalar) || poly_dtype_is_bool(scalar))
    return NULL;
  PolyUOp *den = root->src[1];
  if (den->op != POLY_OP_CONST || den->arg.kind != POLY_ARG_INT) return NULL;
  int shift = power_of_two_shift(den->arg.i);
  if (shift <= 0) return NULL;
  PolyUOp *amount = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(shift));
  return poly_uop2(ctx, POLY_OP_SHR, root->dtype, root->src[0], amount, poly_arg_none());
}

/* Current get_simplifying_rewrite_patterns lowers signed floor modulo by a
 * power of two to the exact two's-complement mask. */
static PolyUOp *floormod_pow2_to_and(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  if (!root || root->n_src != 2) return NULL;
  PolyDType scalar = root->dtype;
  if (!poly_dtype_is_int(scalar) || poly_dtype_is_index(scalar) || poly_dtype_is_bool(scalar))
    return NULL;
  PolyUOp *den = root->src[1];
  if (den->op != POLY_OP_CONST || den->arg.kind != POLY_ARG_INT ||
      power_of_two_shift(den->arg.i) < 0)
    return NULL;
  PolyUOp *mask = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(den->arg.i - 1));
  return poly_uop2(ctx, POLY_OP_AND, root->dtype, root->src[0], mask, poly_arg_none());
}

static bool divmod_floor_same_as_c(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  int64_t a_min = 0, a_max = 0, b_min = 0, b_max = 0;
  poly_uop_minmax(ctx, a, &a_min, &a_max);
  poly_uop_minmax(ctx, b, &b_min, &b_max);
  return (a_min >= 0 && b_min > 0) || (a_max <= 0 && b_max < 0);
}

static PolyUOp *rule_floordiv_to_cdiv(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  if (!root || root->n_src != 2 || !poly_dtype_is_int(root->dtype)) return NULL;
  PolyUOp *a = root->src[0];
  PolyUOp *den = root->src[1];
  /* Pinned tinygrad/uop/decompositions.py:445-448 selects CDIV only from
   * expression bounds. Unsigned UOps can still carry mathematically inferred
   * ranges outside their storage dtype after wrapping ALU, so dtype alone is
   * not a proof that floor and truncating division agree. */
  if (divmod_floor_same_as_c(ctx, a, den))
    return poly_uop2(ctx, POLY_OP_CDIV, root->dtype, a, den, poly_arg_none());

  PolyUOp *trunc = poly_uop2(ctx, POLY_OP_CDIV, root->dtype, a, den, poly_arg_none());
  PolyUOp *rem = poly_uop2(ctx, POLY_OP_CMOD, root->dtype, a, den, poly_arg_none());
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, root->dtype, poly_arg_int(0));
  PolyDType bt = POLY_BOOL;
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
  if (!root || root->n_src != 2 || !poly_dtype_is_int(root->dtype)) return NULL;
  PolyUOp *a = root->src[0];
  PolyUOp *den = root->src[1];
  /* Pinned tinygrad/uop/decompositions.py:450-456 uses the same bounds-only
   * proof for FLOORMOD; do not infer sign from the unsigned storage dtype. */
  if (divmod_floor_same_as_c(ctx, a, den))
    return poly_uop2(ctx, POLY_OP_CMOD, root->dtype, a, den, poly_arg_none());

  PolyUOp *rem = poly_uop2(ctx, POLY_OP_CMOD, root->dtype, a, den, poly_arg_none());
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, root->dtype, poly_arg_int(0));
  PolyDType bt = POLY_BOOL;
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
 * rule_mul_add_to_mulacc — ADD(MUL(a, b), c) → MULACC(a, b, c)
 * Current tinygrad codegen/decomp/op.py:get_late_rewrite_patterns emits this
 * only when the renderer's supported-op set contains MULACC.
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
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKFLOAT, poly_arg_float(1.0));
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

static PolyUOp *logical_not_value(PolyUOp *u, PolyUOp **true_const) {
  if (!u || u->op != POLY_OP_CMPNE || u->n_src != 2) return NULL;
  if (is_true_const_late(u->src[1])) {
    if (true_const) *true_const = u->src[1];
    return u->src[0];
  }
  if (is_true_const_late(u->src[0])) {
    if (true_const) *true_const = u->src[0];
    return u->src[1];
  }
  return NULL;
}

/* Current get_late_rewrite_patterns applies De Morgan when OR is renderable. */
static PolyUOp *rule_demorgan_and(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  if (!root || root->op != POLY_OP_AND || root->n_src != 2 || !poly_dtype_is_bool(root->dtype))
    return NULL;
  PolyUOp *true_const = NULL;
  PolyUOp *x = logical_not_value(root->src[0], &true_const);
  PolyUOp *y = logical_not_value(root->src[1], NULL);
  if (!x || !y || !true_const) return NULL;
  PolyUOp *either = poly_uop2(ctx, POLY_OP_OR, root->dtype, x, y, poly_arg_none());
  return poly_uop2(ctx, POLY_OP_CMPNE, root->dtype, either, true_const, poly_arg_none());
}

static bool int_const_value_codegen(PolyUOp *u, int64_t *out) {
  if (!u || u->op != POLY_OP_CONST || u->arg.kind != POLY_ARG_INT) return false;
  if (out) *out = u->arg.i;
  return true;
}

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

static PolyUOp *u32_cast(PolyCtx *ctx, PolyUOp *u) {
  if (poly_dtype_eq(u->dtype, POLY_UINT32)) return u;
  return poly_uop1(ctx, POLY_OP_CAST, POLY_UINT32, u, poly_arg_none());
}

static PolyUOp *u64_cast(PolyCtx *ctx, PolyUOp *u) {
  if (poly_dtype_eq(u->dtype, POLY_UINT64)) return u;
  return poly_uop1(ctx, POLY_OP_CAST, POLY_UINT64, u, poly_arg_none());
}

static PolyUOp *u32_rol(PolyCtx *ctx, PolyUOp *x, int r) {
  /* tinygrad@2026-08-22/a9069c177a9d codegen/decomp/op.py:60. */
  PolyUOp *l = poly_uop2(ctx, POLY_OP_SHL, POLY_UINT32, x, poly_const_int(ctx, r), poly_arg_none());
  PolyUOp *rr =
      poly_uop2(ctx, POLY_OP_SHR, POLY_UINT32, x, poly_const_int(ctx, 32 - r), poly_arg_none());
  return poly_uop2(ctx, POLY_OP_ADD, POLY_UINT32, l, rr, poly_arg_none());
}

/* tinygrad@2026-08-22/a9069c177a9d codegen/decomp/op.py:48-63 threefry2x32. */
static PolyUOp *threefry2x32(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  if (root->op != POLY_OP_THREEFRY || root->n_src != 2) return NULL;

  PolyUOp *x0, *x1, *key0, *key1;
  if (poly_dtype_eq(root->dtype, POLY_UINT64)) {
    PolyUOp *x64 = u64_cast(ctx, root->src[0]);
    PolyUOp *k64 = u64_cast(ctx, root->src[1]);
    PolyUOp *shift32 = poly_const_int(ctx, 32);
    x0 = u32_cast(ctx, x64);
    x1 = u32_cast(ctx, poly_uop2(ctx, POLY_OP_SHR, POLY_UINT64, x64, shift32, poly_arg_none()));
    key0 = u32_cast(ctx, k64);
    key1 = u32_cast(ctx, poly_uop2(ctx, POLY_OP_SHR, POLY_UINT64, k64, shift32, poly_arg_none()));
  } else {
    x0 = u32_cast(ctx, root->src[0]);
    x1 = poly_const_int(ctx, 0);
    key0 = u32_cast(ctx, root->src[1]);
    key1 = poly_const_int(ctx, 0);
  }

  PolyUOp *ks[3];
  ks[0] = key1;
  ks[1] = poly_uop2(
      ctx, POLY_OP_XOR, POLY_UINT32,
      poly_uop2(ctx, POLY_OP_XOR, POLY_UINT32, key0, key1, poly_arg_none()),
      poly_const_int(ctx, 0x1BD11BDAu), poly_arg_none()
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
    PolyUOp *round_index = poly_const_int(ctx, i);
    PolyUOp *one = poly_const_int(ctx, 1);
    xr0 = poly_uop2(ctx, POLY_OP_ADD, POLY_UINT32, xr0, ks[i % 3], poly_arg_none());
    xr1 = poly_uop2(
        ctx, POLY_OP_ADD, POLY_UINT32,
        poly_uop2(
            ctx, POLY_OP_ADD, POLY_UINT32,
            poly_uop2(ctx, POLY_OP_ADD, POLY_UINT32, xr1, ks[(i + 1) % 3], poly_arg_none()),
            round_index, poly_arg_none()
        ),
        one, poly_arg_none()
    );
  }

  if (poly_dtype_eq(root->dtype, POLY_UINT32)) return xr0;
  if (poly_dtype_eq(root->dtype, POLY_UINT64)) {
    PolyUOp *lo = u64_cast(ctx, xr0);
    PolyUOp *hi = poly_uop2(
        ctx, POLY_OP_SHL, POLY_UINT64, u64_cast(ctx, xr1), poly_const_int(ctx, 32), poly_arg_none()
    );
    return poly_uop2(ctx, POLY_OP_OR, POLY_UINT64, hi, lo, poly_arg_none());
  }
  return poly_uop1(ctx, POLY_OP_CAST, root->dtype, xr0, poly_arg_none());
}

/* Current Tinygrad codegen/decomp/op.py:get_simplifying_rewrite_patterns.
 * Renderer switches expose the same relevant op support through these caps. */
static _Thread_local PolyPatternMatcher *g_pm_simplifying_decomp[2][2] = {{NULL}};

PolyPatternMatcher *poly_get_simplifying_rewrite_patterns(PolyRendererCaps caps) {
  PolyPatternMatcher **target =
      &g_pm_simplifying_decomp[caps.has_max ? 1 : 0][caps.has_threefry ? 1 : 0];
  if (*target) return *target;

  PolyRule rules[6];
  int n = 0;
  rules[n++] = (PolyRule){poly_upat_op(POLY_OP_FLOORDIV, NULL, 0, NULL), floordiv_pow2_to_shr};
  rules[n++] = (PolyRule){poly_upat_op(POLY_OP_FLOORDIV, NULL, 0, NULL), rule_floordiv_to_cdiv};
  rules[n++] = (PolyRule){poly_upat_op(POLY_OP_FLOORMOD, NULL, 0, NULL), floormod_pow2_to_and};
  rules[n++] = (PolyRule){poly_upat_op(POLY_OP_FLOORMOD, NULL, 0, NULL), rule_floormod_to_cmod};
  if (!caps.has_threefry)
    rules[n++] = (PolyRule){poly_upat_op(POLY_OP_THREEFRY, NULL, 0, NULL), threefry2x32};
  if (!caps.has_max)
    rules[n++] = (PolyRule){poly_upat_op(POLY_OP_MAX, NULL, 0, NULL), rule_decomp_max};
  *target = poly_pm_thread_cache(poly_pm_new(rules, n));
  return *target;
}

/* Current Tinygrad codegen/decomp/op.py:get_late_rewrite_patterns. Fast IDIV
 * remains disabled, matching the current DISABLE_FAST_IDIV=1 default. */
static _Thread_local PolyPatternMatcher *g_pm_late_decomp[2][2] = {{NULL}};

PolyPatternMatcher *poly_get_late_rewrite_patterns(PolyRendererCaps caps) {
  PolyPatternMatcher **target = &g_pm_late_decomp[caps.has_mulacc ? 1 : 0][caps.has_fdiv ? 1 : 0];
  if (*target) return *target;

  PolyRule rules[16];
  int n = 0;
  rules[n++] = (PolyRule){poly_upat_op(POLY_OP_AND, NULL, 0, NULL), rule_demorgan_and};
  rules[n++] = (PolyRule
  ){poly_upat_op2(POLY_OP_MUL, poly_upat_any("x"), poly_upat_cvar("c"), NULL), rule_mul_to_shl};
  rules[n++] = (PolyRule
  ){poly_upat_op2(POLY_OP_CDIV, poly_upat_any("x"), poly_upat_cvar("c"), NULL), rule_cdiv_to_shr};
  rules[n++] = (PolyRule
  ){poly_upat_op2(POLY_OP_MUL, poly_upat_any("x"), poly_upat_cvar("c"), NULL),
    rule_mul_neg1_to_neg};
  rules[n++] = (PolyRule
  ){poly_upat_op2(
        POLY_OP_ADD, poly_upat_any("x"), poly_upat_op1(POLY_OP_NEG, poly_upat_any("y"), NULL), NULL
    ),
    rule_add_neg_to_sub};
  /* UPat treats ADD as commutative; C spells the second ordering explicitly. */
  rules[n++] = (PolyRule
  ){poly_upat_op2(
        POLY_OP_ADD, poly_upat_op1(POLY_OP_NEG, poly_upat_any("y"), NULL), poly_upat_any("x"), NULL
    ),
    rule_neg_add_to_sub};
  if (caps.has_mulacc) {
    rules[n++] = (PolyRule){poly_upat_op(POLY_OP_ADD, NULL, 0, NULL), rule_mul_add_to_mulacc};
    rules[n++] = (PolyRule){poly_upat_op(POLY_OP_ADD, NULL, 0, NULL), rule_shl_add_to_mulacc};
  }
  if (caps.has_fdiv) {
    rules[n++] =
        (PolyRule){poly_upat_op1(POLY_OP_RECIPROCAL, poly_upat_any("x"), NULL), rule_recip_to_fdiv};
    rules[n++] = (PolyRule
    ){poly_upat_op2c(
          POLY_OP_MUL, poly_upat_any("a"),
          poly_upat_op2(POLY_OP_FDIV, poly_upat_cvar("one"), poly_upat_any("b"), NULL), NULL
      ),
      rule_mul_fdiv1_to_fdiv};
  }
  rules[n++] = (PolyRule){poly_upat_op(POLY_OP_CMPNE, NULL, 0, NULL), rule_not_cmplt_to_bound};
  rules[n++] = (PolyRule){poly_upat_op(POLY_OP_CMPNE, NULL, 0, NULL), rule_cmpne_not_to_cmpeq};

  *target = poly_pm_thread_cache(poly_pm_new(rules, n));
  return *target;
}

/* pm_transcendental: EXP2/LOG2/SIN → polynomial approximation */

/* Dtype-parametric helpers for IEEE 754 bit manipulation. */
static int xd_mantissa_bits(PolyDType dt) {
  int bits = dt.bitsize;
  return bits == 16 ? 10 : bits == 64 ? 52 : 23;
}
static int xd_exponent_bias(PolyDType dt) {
  int bits = dt.bitsize;
  return bits == 16 ? 15 : bits == 64 ? 1023 : 127;
}
static int64_t xd_exponent_mask(PolyDType dt) {
  int bits = dt.bitsize;
  return bits == 16 ? 0x1FLL : bits == 64 ? 0x7FFLL : 0xFFLL;
}
static PolyDType xd_int_for_float(PolyDType dt) {
  return dt.bitsize == 16 ? POLY_INT16 : dt.bitsize == 64 ? POLY_INT64 : POLY_INT32;
}

/* helpers.polyN receives Python floats, so coefficients stay weak until the
 * later commit_weak stage. */
static PolyUOp *xd_polyN(
    PolyCtx *ctx,
    PolyDType ft,
    PolyUOp *x,
    const double *coeffs,
    int ncoeffs
) {
  PolyUOp *u = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKFLOAT, poly_arg_float(coeffs[0]));
  for (int i = 1; i < ncoeffs; i++) {
    u = poly_uop2(ctx, POLY_OP_MUL, ft, u, x, poly_arg_none());
    u = poly_uop2(
        ctx, POLY_OP_ADD, ft, u,
        poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKFLOAT, poly_arg_float(coeffs[i])), poly_arg_none()
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
  PolyDType bt = POLY_BOOL;
  PolyUOp *f_neg_inf =
      poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKFLOAT, poly_arg_float(-__builtin_inf()));
  PolyUOp *f_pos_inf =
      poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKFLOAT, poly_arg_float(__builtin_inf()));
  PolyUOp *nan_chk = poly_uop2(ctx, POLY_OP_CMPNE, bt, d, d, poly_arg_none());
  PolyUOp *neginf_chk = poly_uop2(ctx, POLY_OP_CMPNE, bt, d, f_neg_inf, poly_arg_none());
  PolyUOp *posinf_chk = poly_uop2(ctx, POLY_OP_CMPNE, bt, d, f_pos_inf, poly_arg_none());
  PolyUOp *inner = poly_uop3(ctx, POLY_OP_WHERE, ft, neginf_chk, ratio, ninf_val, poly_arg_none());
  PolyUOp *mid = poly_uop3(ctx, POLY_OP_WHERE, ft, nan_chk, nan_val, inner, poly_arg_none());
  return poly_uop3(ctx, POLY_OP_WHERE, ft, posinf_chk, mid, pinf_val, poly_arg_none());
}

/* rintk: round float d to nearest integer (away from 0). */
static PolyUOp *xd_rintk(PolyCtx *ctx, PolyDType ft, PolyDType it, PolyUOp *d) {
  PolyDType bt = POLY_BOOL;
  PolyUOp *f_zero = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKFLOAT, poly_arg_float(0.0));
  PolyUOp *f_neg_half = poly_const_like_float(ctx, d, -0.5);
  PolyUOp *f_half = poly_const_like_float(ctx, d, 0.5);
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
  PolyDType out_scalar;
  if (poly_dtype_eq(q->dtype, POLY_INT64))
    out_scalar = POLY_FLOAT64;
  else if (poly_dtype_eq(q->dtype, POLY_INT32))
    out_scalar = POLY_FLOAT32;
  else if (poly_dtype_eq(q->dtype, POLY_INT16))
    out_scalar = ft;
  else
    return NULL;
  PolyDType out_ft = out_scalar;
  int bias = xd_exponent_bias(out_ft);
  int mbits = xd_mantissa_bits(out_ft);
  PolyUOp *i_bias = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(bias));
  PolyUOp *i_factor = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(1LL << mbits));
  PolyUOp *added = poly_uop2(ctx, POLY_OP_ADD, it, q, i_bias, poly_arg_none());
  /* decompositions.shl creates x * (2**n); a later scalar rewrite may form SHL. */
  PolyUOp *shifted = poly_uop2(ctx, POLY_OP_MUL, it, added, i_factor, poly_arg_none());
  return poly_uop1(ctx, POLY_OP_BITCAST, out_ft, shifted, poly_arg_none());
}

static PolyUOp *xd_sub_like_tinygrad(PolyCtx *ctx, PolyDType dt, PolyUOp *a, PolyUOp *b);
static PolyUOp *xd_floordiv_positive_const(PolyCtx *ctx, PolyDType it, PolyUOp *x, int64_t divisor);

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
  PolyDType scalar = ft;
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
  PolyDType bit_dt = bit_scalar;
  PolyUOp *bits = poly_uop1(ctx, POLY_OP_BITCAST, bit_dt, v, poly_arg_none());
  PolyUOp *mask = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(xd_exponent_mask(ft)));
  PolyUOp *exponent = poly_uop2(
      ctx, POLY_OP_AND, bit_dt,
      xd_floordiv_positive_const(ctx, bit_dt, bits, INT64_C(1) << xd_mantissa_bits(ft)), mask,
      poly_arg_none()
  );
  PolyUOp *mantissa_bits = poly_uop2(
      ctx, POLY_OP_OR, bit_dt,
      poly_uop2(
          ctx, POLY_OP_AND, bit_dt, bits,
          poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(m1)), poly_arg_none()
      ),
      poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(m2)), poly_arg_none()
  );
  *mantissa_out = poly_uop1(ctx, POLY_OP_BITCAST, ft, mantissa_bits, poly_arg_none());
  PolyUOp *bias = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(xd_exponent_bias(ft)));
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(1));
  *exponent_out = poly_uop2(
      ctx, POLY_OP_ADD, bit_dt, xd_sub_like_tinygrad(ctx, bit_dt, exponent, bias), one,
      poly_arg_none()
  );
  return true;
}

static PolyUOp *xd_sub_like_tinygrad(PolyCtx *ctx, PolyDType dt, PolyUOp *a, PolyUOp *b) {
  /* Elementwise subtraction is built as a + (b * -1). */
  PolyArg neg_arg = poly_dtype_is_float(dt) ? poly_arg_float(-1.0) : poly_arg_int(-1);
  PolyDType neg_dtype = poly_dtype_is_float(dt) ? POLY_WEAKFLOAT : POLY_WEAKINT;
  PolyUOp *neg_one = poly_uop0(ctx, POLY_OP_CONST, neg_dtype, neg_arg);
  PolyUOp *neg_b = poly_uop2(ctx, POLY_OP_MUL, b->dtype, b, neg_one, poly_arg_none());
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
  PolyUOp *den = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(divisor));
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
  PolyUOp *factor = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(1LL << mbits));
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
  PolyUOp *i_mask = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(emask));
  PolyUOp *i_neg_bias = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(-bias));
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
  PolyDType sft = root->dtype;
  bool is_f16 = poly_dtype_eq(sft, POLY_FLOAT16);
  bool is_f32 = poly_dtype_eq(sft, POLY_FLOAT32);
  bool is_f64 = poly_dtype_eq(sft, POLY_FLOAT64);
  if (!is_f16 && !is_f32 && !is_f64) return NULL;

  PolyDType ft = root->dtype;
  PolyDType it = xd_int_for_float(ft);
  PolyDType bt = POLY_BOOL;

  /* const_like values are strong; numeric comparison operands stay weak. */
  PolyUOp *f_zero = poly_const_like_float(ctx, d, 0.0);
  PolyUOp *f_pos_inf = poly_const_like_float(ctx, d, __builtin_inf());
  PolyUOp *f_nan = poly_const_like_float(ctx, d, __builtin_nan(""));
  PolyUOp *b_true = poly_uop0(ctx, POLY_OP_CONST, bt, poly_arg_bool(true));

  /* Dtype-specific overflow/underflow thresholds (from tinygrad) */
  double upper = is_f16 ? 23.0 : is_f64 ? 1024.0 : 128.0;
  double lower = is_f16 ? -22.0 : is_f64 ? -2000.0 : -150.0;
  PolyUOp *f_upper = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKFLOAT, poly_arg_float(upper));
  PolyUOp *f_lower = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKFLOAT, poly_arg_float(lower));

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
static PolyUOp *rule_decomp_transcendental_other_float(
    PolyCtx *ctx,
    PolyUOp *root,
    const PolyBindings *b
) {
  (void)b;
  if (!root || root->n_src != 1) return NULL;
  PolyDType scalar = root->dtype;
  if (!poly_dtype_is_float(scalar) || poly_dtype_eq(scalar, POLY_FLOAT16) ||
      poly_dtype_eq(scalar, POLY_FLOAT32) || poly_dtype_eq(scalar, POLY_FLOAT64))
    return NULL;
  PolyDType f32 = POLY_FLOAT32;
  PolyUOp *wide = poly_uop1(ctx, POLY_OP_CAST, f32, root->src[0], poly_arg_none());
  PolyUOp *transcendental = poly_uop1(ctx, root->op, f32, wide, poly_arg_none());
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
  PolyDType sft = root->dtype;
  if (!poly_dtype_is_float(sft) || (sft.bitsize != 32 && sft.bitsize != 64)) return NULL;

  PolyDType ft = root->dtype;
  PolyDType it = xd_int_for_float(ft);
  PolyDType bt = POLY_BOOL;
  bool is_f64 = (sft.bitsize == 64);

  PolyUOp *f_1e4 = poly_const_like_float(ctx, d, 1e-4);
  PolyUOp *f_4_3 = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKFLOAT, poly_arg_float(1.0 / 0.75));
  PolyUOp *f_neg_64 = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKFLOAT, poly_arg_float(-64.0));
  PolyUOp *f_2p64 =
      poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKFLOAT, poly_arg_float(18446744073709551616.0));

  /* Denormal handling: scale up subnormals by 2^64 */
  PolyUOp *is_denormal = poly_uop2(ctx, POLY_OP_CMPLT, bt, d, f_1e4, poly_arg_none());
  PolyUOp *scaled = poly_uop2(ctx, POLY_OP_MUL, ft, d, f_2p64, poly_arg_none());
  PolyUOp *a = poly_uop3(ctx, POLY_OP_WHERE, ft, is_denormal, scaled, d, poly_arg_none());

  /* e = ilogb2k(a * (1/0.75)), using shared helper */
  PolyUOp *a_scaled = poly_uop2(ctx, POLY_OP_MUL, ft, a, f_4_3, poly_arg_none());
  PolyUOp *e_int = xd_ilogb2k(ctx, ft, it, a_scaled);
  PolyUOp *e = poly_uop1(ctx, POLY_OP_CAST, ft, e_int, poly_arg_none());

  /* m = ldexp3k(a, -e), using shared helper */
  PolyUOp *neg_e = poly_uop2(
      ctx, POLY_OP_MUL, ft, e, poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKFLOAT, poly_arg_float(-1.0)),
      poly_arg_none()
  );
  PolyUOp *m = xd_ldexp3k(ctx, ft, it, a, neg_e);

  /* Denormal exponent correction: subtract the 2^64 scaling */
  PolyUOp *e_minus64 = poly_uop2(ctx, POLY_OP_ADD, ft, e, f_neg_64, poly_arg_none());
  PolyUOp *e_adj = poly_uop3(ctx, POLY_OP_WHERE, ft, is_denormal, e_minus64, e, poly_arg_none());

  /* x = (m - 1) / (m + 1) */
  PolyUOp *m_minus1 = poly_uop2(
      ctx, POLY_OP_ADD, ft, m, poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKFLOAT, poly_arg_float(-1.0)),
      poly_arg_none()
  );
  PolyUOp *m_plus1 = poly_uop2(
      ctx, POLY_OP_ADD, ft, m, poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKFLOAT, poly_arg_float(1.0)),
      poly_arg_none()
  );
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
    PolyUOp *f_k1 =
        poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKFLOAT, poly_arg_float(2.885390081777926774));
    r = poly_uop2(
        ctx, POLY_OP_ADD, ft, r, poly_uop2(ctx, POLY_OP_MUL, ft, x, f_k1, poly_arg_none()),
        poly_arg_none()
    );
  } else {
    /* f32: k1 + s_lo term (x*k2) for extra precision */
    PolyUOp *f_k1 =
        poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKFLOAT, poly_arg_float(2.8853900432586669922));
    PolyUOp *f_k2 =
        poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKFLOAT, poly_arg_float(3.2734474483568488616e-08));
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
  PolyUOp *ne_inf = poly_uop2(
      ctx, POLY_OP_CMPNE, bt, d,
      poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKFLOAT, poly_arg_float(__builtin_inf())),
      poly_arg_none()
  );
  r = poly_uop3(
      ctx, POLY_OP_WHERE, ft, ne_inf, r, poly_const_like_float(ctx, r, __builtin_inf()),
      poly_arg_none()
  );
  PolyUOp *ne_zero = poly_uop2(
      ctx, POLY_OP_CMPNE, bt, d, poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKFLOAT, poly_arg_float(0.0)),
      poly_arg_none()
  );
  r = poly_uop3(
      ctx, POLY_OP_WHERE, ft, ne_zero, r, poly_const_like_float(ctx, r, -__builtin_inf()),
      poly_arg_none()
  );
  PolyUOp *lt_neg_zero = poly_uop2(
      ctx, POLY_OP_CMPLT, bt, d,
      poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKFLOAT, poly_arg_float(-0.0)), poly_arg_none()
  );
  r = poly_uop3(
      ctx, POLY_OP_WHERE, ft, lt_neg_zero, poly_const_like_float(ctx, r, __builtin_nan("")), r,
      poly_arg_none()
  );
  PolyUOp *is_nan = poly_uop2(ctx, POLY_OP_CMPNE, bt, d, d, poly_arg_none());
  r = poly_uop3(
      ctx, POLY_OP_WHERE, ft, is_nan, poly_const_like_float(ctx, r, __builtin_nan("")), r,
      poly_arg_none()
  );
  PolyUOp *rec = poly_uop1(ctx, POLY_OP_RECIPROCAL, ft, d, poly_arg_none());
  PolyUOp *rec_ne_ninf = poly_uop2(
      ctx, POLY_OP_CMPNE, bt, rec,
      poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKFLOAT, poly_arg_float(-__builtin_inf())),
      poly_arg_none()
  );
  r = poly_uop3(
      ctx, POLY_OP_WHERE, ft, rec_ne_ninf, r, poly_const_like_float(ctx, r, -__builtin_inf()),
      poly_arg_none()
  );

  return r;
}

/* sin_poly: trig_poly from tinygrad, dtype-aware.
 * Returns d * polyN(d*d, coeffs). Supports f32 (5 coeffs) and f64 (10 coeffs). */
static PolyUOp *sin_poly(PolyCtx *ctx, PolyUOp *d) {
  PolyDType ft = d->dtype;
  /* tinygrad@2026-08-22/a9069c177a9d uop/decompositions.py:152 dispatches
   * on the scalar DType; UOp shape owns aggregate width. */
  bool is_f64 = poly_dtype_eq(ft, POLY_FLOAT64);
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

/* _take starts from typed uint32 zero. Table values use an.const_like, while
 * Python loop counters remain weak integers. */
static PolyUOp *take_two_over_pi_f32(PolyCtx *ctx, PolyUOp *i_u64, int offset) {
  static const uint32_t two_over_pi_f[] = {0x00000000u, 0x28be60dbu, 0x9391054au, 0x7f09d5f4u,
                                           0x7d4d3770u, 0x36d8a566u, 0x4f10e410u};
  const int len = (int)(sizeof(two_over_pi_f) / sizeof(two_over_pi_f[0]));
  const int max_count = len - 2 - offset;
  PolyDType u32 = POLY_UINT32;
  PolyDType bt = POLY_BOOL;
  PolyUOp *out = poly_uop0(ctx, POLY_OP_CONST, u32, poly_arg_int(0));
  for (int count = max_count; count >= 0; count--) {
    PolyUOp *cnt = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int((int64_t)count));
    PolyUOp *ne = poly_uop2(ctx, POLY_OP_CMPNE, bt, i_u64, cnt, poly_arg_none());
    PolyUOp *val =
        poly_uop0(ctx, POLY_OP_CONST, u32, poly_arg_int((int64_t)two_over_pi_f[count + offset]));
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
          poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKFLOAT, poly_arg_float(-3.1414794921875)),
          poly_arg_none()
      ),
      x, poly_arg_none()
  );
  d = poly_uop2(
      ctx, POLY_OP_ADD, ft,
      poly_uop2(
          ctx, POLY_OP_MUL, ft, qf,
          poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKFLOAT, poly_arg_float(-0.00011315941810607910156)),
          poly_arg_none()
      ),
      d, poly_arg_none()
  );
  d = poly_uop2(
      ctx, POLY_OP_ADD, ft,
      poly_uop2(
          ctx, POLY_OP_MUL, ft, qf,
          poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKFLOAT, poly_arg_float(-1.9841872589410058936e-09)),
          poly_arg_none()
      ),
      d, poly_arg_none()
  );
  d = poly_uop2(
      ctx, POLY_OP_ADD, ft,
      poly_uop2(
          ctx, POLY_OP_MUL, ft, qf,
          poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKFLOAT, poly_arg_float(-1.2154201256553420762e-10)),
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

  PolyUOp *pia = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKFLOAT, poly_arg_float(-PI_A));
  PolyUOp *pib = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKFLOAT, poly_arg_float(-PI_B));
  PolyUOp *pic = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKFLOAT, poly_arg_float(-PI_C));
  PolyUOp *pid = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKFLOAT, poly_arg_float(-PI_D));

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
  PolyDType sft = root->dtype;
  if (!poly_dtype_is_float(sft) || (sft.bitsize != 32 && sft.bitsize != 64)) return NULL;

  PolyDType ft = root->dtype;
  PolyDType it = POLY_INT32;
  PolyDType ut32 = POLY_UINT32;
  PolyDType ut64 = POLY_UINT64;
  PolyDType bt = POLY_BOOL;
  bool is_f64 = (sft.bitsize == 64);

  PolyUOp *f_zero = poly_const_like_float(ctx, d, 0.0);
  PolyUOp *f_one = poly_const_like_float(ctx, d, 1.0);
  PolyUOp *f_neg_one = poly_const_like_float(ctx, d, -1.0);
  PolyUOp *f_pi_2 = poly_const_like_float(ctx, d, 1.57079632679489661923);
  PolyUOp *f_nan = poly_const_like_float(ctx, d, __builtin_nan(""));
  PolyUOp *w_zero = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKFLOAT, poly_arg_float(0.0));
  PolyUOp *w_half = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKFLOAT, poly_arg_float(0.5));
  PolyUOp *w_pi_2 =
      poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKFLOAT, poly_arg_float(1.57079632679489661923));
  PolyUOp *f_switch = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKFLOAT, poly_arg_float(30.0));
  double m_1_pi = 0.318309886183790671537767526745028724;
  PolyUOp *f_m_1_pi = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKFLOAT, poly_arg_float(m_1_pi));
  PolyUOp *f_ph_mul =
      poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKFLOAT, poly_arg_float(3.4061215800865545e-19));
  PolyUOp *i_zero = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(0));
  PolyUOp *i_one = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(1));
  PolyUOp *i_two = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(2));
  PolyUOp *i_31 = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(31));
  PolyUOp *i_32 = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(32));
  PolyUOp *u_mask =
      poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(0x3fffffffffffffffULL));

  PolyUOp *x = xd_lazy_map_numbers(ctx, ft, d, f_zero, f_zero, f_zero, d);

  /* x_sign = x!=0 ? (x<0 ? -1 : 1) : 0 */
  PolyUOp *x_ne0 = poly_uop2(ctx, POLY_OP_CMPNE, bt, x, w_zero, poly_arg_none());
  PolyUOp *x_lt0 = poly_uop2(ctx, POLY_OP_CMPLT, bt, x, w_zero, poly_arg_none());
  PolyUOp *x_pm = poly_uop3(ctx, POLY_OP_WHERE, ft, x_lt0, f_neg_one, f_one, poly_arg_none());
  PolyUOp *x_sign = poly_uop3(ctx, POLY_OP_WHERE, ft, x_ne0, x_pm, f_zero, poly_arg_none());
  PolyUOp *x_abs = poly_uop2(ctx, POLY_OP_MUL, ft, x, x_sign, poly_arg_none());

  /* Cody-Waite reduction (small branch) */
  PolyUOp *q_small;
  PolyUOp *r_small;

  if (is_f64) {
    /* f64: qdh = (x_abs * (m_1_pi / 2^24)).cast(int64).cast(f64) * 2^24 */
    PolyUOp *f_m1pi_div2p24 =
        poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKFLOAT, poly_arg_float(m_1_pi / 16777216.0));
    PolyUOp *f_2p24 = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKFLOAT, poly_arg_float(16777216.0));
    PolyDType it64 = POLY_INT64;
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
          poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKFLOAT, poly_arg_float(4294967296.0)),
          poly_arg_none()
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
              poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(INT64_C(1) << 32)),
              poly_arg_none()
          ),
          hp_mi, poly_arg_none()
      ),
      xd_floordiv_positive_const(ctx, ut64, hp_lo, INT64_C(1) << 32), poly_arg_none()
  );
  PolyUOp *q_ph = poly_uop1(
      ctx, POLY_OP_CAST, it, xd_floordiv_positive_const(ctx, ut64, p, INT64_C(1) << 62),
      poly_arg_none()
  );
  PolyUOp *p_masked = poly_uop2(ctx, POLY_OP_AND, ut64, p, u_mask, poly_arg_none());
  PolyUOp *r_ph_base = poly_uop2(
      ctx, POLY_OP_MUL, ft, poly_uop1(ctx, POLY_OP_CAST, ft, p_masked, poly_arg_none()), f_ph_mul,
      poly_arg_none()
  );
  PolyUOp *f_lt_half = poly_uop2(ctx, POLY_OP_CMPLT, bt, f_frexp, w_half, poly_arg_none());
  PolyUOp *r_ph = poly_uop3(
      ctx, POLY_OP_WHERE, ft, f_lt_half, r_ph_base,
      xd_sub_like_tinygrad(ctx, ft, r_ph_base, w_pi_2), poly_arg_none()
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

  return xd_lazy_map_numbers(ctx, ft, d, f_nan, f_nan, f_nan, result);
}

/* Build the renderer-capability-specific transcendental matcher. Pinned
 * tinygrad only decomposes an operation absent from renderer.code_for_op. */
static _Thread_local PolyPatternMatcher *g_pm_transcendental_caps[2][2][2] = {0};

PolyPatternMatcher *poly_get_transcendental_patterns(PolyRendererCaps caps) {
  PolyPatternMatcher **target =
      &g_pm_transcendental_caps[caps.has_exp2 ? 1 : 0][caps.has_log2 ? 1 : 0][caps.has_sin ? 1 : 0];
  if (*target) return *target;

  PolyOpSet exp2_set = poly_opset_add((PolyOpSet){{0, 0}}, POLY_OP_EXP2);
  PolyOpSet log2_set = poly_opset_add((PolyOpSet){{0, 0}}, POLY_OP_LOG2);
  PolyOpSet sin_set = poly_opset_add((PolyOpSet){{0, 0}}, POLY_OP_SIN);
  PolyRule rules[6];
  int n = 0;
  if (!caps.has_exp2) {
    rules[n++] = (PolyRule){poly_upat_ops(exp2_set, NULL, 0, NULL), rule_decomp_exp2};
    rules[n++] =
        (PolyRule){poly_upat_ops(exp2_set, NULL, 0, NULL), rule_decomp_transcendental_other_float};
  }
  if (!caps.has_log2) {
    rules[n++] = (PolyRule){poly_upat_ops(log2_set, NULL, 0, NULL), rule_decomp_log2};
    rules[n++] =
        (PolyRule){poly_upat_ops(log2_set, NULL, 0, NULL), rule_decomp_transcendental_other_float};
  }
  if (!caps.has_sin) {
    rules[n++] = (PolyRule){poly_upat_ops(sin_set, NULL, 0, NULL), rule_decomp_sin};
    rules[n++] =
        (PolyRule){poly_upat_ops(sin_set, NULL, 0, NULL), rule_decomp_transcendental_other_float};
  }

  if (n == 0) {
    *target = poly_pm_thread_cache(poly_pm_new(NULL, 0));
    return *target;
  }
  *target = poly_pm_thread_cache(poly_pm_new(rules, n));
  return *target;
}

/* Current expander2 (tinygrad/codegen/__init__.py:42-90).
 *
 * Current Tinygrad gives each UPCAST/UNROLL RANGE its own tensor-shape axis.
 * RANGE expansion therefore produces a scalar-dtype STACK reshaped onto that
 * axis. */

typedef struct {
  int64_t *axis_ids;
  int *shape_axes;
  int count;
} Expander2Context;

static int expander2_axis(const Expander2Context *ectx, int64_t axis_id) {
  if (!ectx) return -1;
  for (int i = 0; i < ectx->count; i++)
    if (ectx->axis_ids[i] == axis_id) return ectx->shape_axes[i];
  return -1;
}

static PolyUOp *poly_expand_range(PolyCtx *ctx, PolyUOp *r, const PolyBindings *b) {
  (void)b;
  if (!r || r->op != POLY_OP_RANGE || !poly_arg_is_range(r->arg)) return NULL;
  Expander2Context *ectx = (Expander2Context *)poly_graph_rewrite_userctx();
  int axis = expander2_axis(ectx, poly_range_axis_id(r->arg));
  if (axis < 0 || !ectx || axis >= ectx->count) return NULL;

  int64_t vmin = 0, vmax = 0;
  poly_uop_minmax(ctx, r, &vmin, &vmax);
  if (vmin != 0 || vmax < 0 || vmax == INT64_MAX) return NULL;
  int64_t size = vmax + 1;
  if (size <= 0 || size > UINT16_MAX) return NULL;

  PolyUOp **values = malloc((size_t)size * sizeof(*values));
  PolyUOp **shape = malloc((size_t)ectx->count * sizeof(*shape));
  if (!values || !shape) {
    free(values);
    free(shape);
    return NULL;
  }
  for (int64_t i = 0; i < size; i++)
    values[i] = poly_uop0(ctx, POLY_OP_CONST, r->dtype, poly_arg_int(i));
  for (int i = 0; i < ectx->count; i++)
    shape[i] = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(i == axis ? size : 1));

  PolyUOp *stack = poly_uop_stack(ctx, values, (int)size);
  PolyUOp *ret = stack ? poly_reshape_uop(ctx, stack, shape, ectx->count) : NULL;
  free(shape);
  free(values);
  return ret;
}

static PolyUOp *poly_expand_reduce(PolyCtx *ctx, PolyUOp *r, const PolyBindings *b) {
  (void)b;
  if (!r || r->op != POLY_OP_REDUCE || r->n_src < 1 || r->arg.kind != POLY_ARG_REDUCE ||
      r->arg.reduce.num_axes != 0)
    return NULL;

  PolyUOp *value = r->src[0];
  int ndim = poly_uop_ndim(ctx, value);
  if (ndim < 0 || ndim > POLY_MAX_DIMS) return NULL;

  bool new_axis[POLY_MAX_DIMS] = {0};
  int n_new_axes = 0, n_ranges = 0;
  for (int i = 1; i < r->n_src; i++) {
    PolyUOp *src = r->src[i];
    if (src->op == POLY_OP_RANGE) {
      n_ranges++;
      continue;
    }
    int src_ndim = poly_uop_ndim(ctx, src);
    if (src_ndim < 0 || src_ndim > ndim) return NULL;
    for (int axis = 0; axis < src_ndim; axis++) {
      PolyUOp *dim = poly_uop_shape_dim(ctx, src, axis);
      int64_t size = 0;
      if (!dim || poly_uop_const_i64(dim, &size) != 0) return NULL;
      if (size > 1 && !new_axis[axis]) {
        new_axis[axis] = true;
        n_new_axes++;
      }
    }
  }
  if (n_new_axes == 0) return NULL;

  int64_t perm[POLY_MAX_DIMS];
  PolyUOp *out_shape[POLY_MAX_DIMS];
  int at = 0;
  for (int axis = 0; axis < ndim; axis++)
    if (new_axis[axis]) perm[at++] = axis;
  for (int axis = 0; axis < ndim; axis++)
    if (!new_axis[axis]) perm[at++] = axis;
  for (int axis = 0; axis < ndim; axis++) {
    out_shape[axis] = new_axis[axis] ? poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(1))
                                     : poly_uop_shape_dim(ctx, value, axis);
    if (!out_shape[axis]) return NULL;
  }

  PolyUOp *ordered = poly_permute(ctx, value, perm, ndim);
  PolyUOp **reduce_src = malloc((size_t)(1 + n_ranges) * sizeof(*reduce_src));
  if (!ordered || !reduce_src) {
    free(reduce_src);
    return NULL;
  }
  reduce_src[0] = ordered;
  at = 1;
  for (int i = 1; i < r->n_src; i++)
    if (r->src[i]->op == POLY_OP_RANGE) reduce_src[at++] = r->src[i];
  PolyUOp *reduced = poly_uop_tagged_arg(
      ctx, POLY_OP_REDUCE, r->dtype, reduce_src, 1 + n_ranges,
      poly_arg_reduce(r->arg.reduce.op, n_new_axes), r->tag, r->tag_arg
  );
  free(reduce_src);
  return reduced ? poly_reshape_uop(ctx, reduced, out_shape, ndim) : NULL;
}

/* Current codegen/__init__.py:64-82 contracts input fragment axes, clears
 * temporary WMMA metadata, then restores output fragment axes. */
static PolyUOp *poly_contract_axis(
    PolyCtx *ctx,
    Expander2Context *ectx,
    PolyUOp *u,
    int64_t (*pairs)[2],
    int n_pairs
) {
  int ndim = poly_uop_ndim(ctx, u);
  if (ndim < 0 || ndim > POLY_MAX_DIMS || n_pairs <= 0 || n_pairs > ndim) return NULL;
  bool tail[POLY_MAX_DIMS] = {0};
  int64_t perm[POLY_MAX_DIMS];
  int n_head = 0;
  for (int i = 0; i < n_pairs; i++) {
    int axis = expander2_axis(ectx, pairs[i][0]);
    if (axis < 0 || axis >= ndim || tail[axis]) return NULL;
    tail[axis] = true;
  }
  for (int axis = 0; axis < ndim; axis++)
    if (!tail[axis]) perm[n_head++] = axis;
  for (int i = 0; i < n_pairs; i++)
    perm[n_head + i] = expander2_axis(ectx, pairs[i][0]);
  PolyUOp *permuted = poly_permute(ctx, u, perm, ndim);
  if (!permuted) return NULL;

  PolyUOp *shape[POLY_MAX_DIMS];
  for (int i = 0; i < n_head; i++)
    shape[i] = poly_uop_shape_dim(ctx, permuted, i);
  PolyUOp *flat = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(1));
  for (int i = n_head; i < ndim; i++) {
    PolyUOp *dim = poly_uop_shape_dim(ctx, permuted, i);
    if (!dim) return NULL;
    flat = poly_uop2(ctx, POLY_OP_MUL, POLY_WEAKINT, flat, dim, poly_arg_none());
  }
  flat = poly_graph_rewrite(ctx, flat, poly_symbolic());
  if (!flat) return NULL;
  shape[n_head] = flat;
  return poly_reshape_uop(ctx, permuted, shape, n_head + 1);
}

static PolyUOp *poly_unroll_axis(
    PolyCtx *ctx,
    Expander2Context *ectx,
    PolyUOp *u,
    int64_t (*pairs)[2],
    int n_pairs
) {
  int ndim = poly_uop_ndim(ctx, u);
  if (ndim <= 0 || ndim > POLY_MAX_DIMS || n_pairs <= 0 || ndim - 1 + n_pairs > POLY_MAX_DIMS)
    return NULL;
  int out_ndim = ndim - 1 + n_pairs;
  PolyUOp *shape[POLY_MAX_DIMS];
  for (int i = 0; i < ndim - 1; i++)
    shape[i] = poly_uop_shape_dim(ctx, u, i);
  for (int i = 0; i < n_pairs; i++)
    shape[ndim - 1 + i] = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(pairs[i][1]));
  PolyUOp *reshaped = poly_reshape_uop(ctx, u, shape, out_ndim);
  if (!reshaped) return NULL;

  bool tail[POLY_MAX_DIMS] = {0};
  int ordered[POLY_MAX_DIMS], inverse[POLY_MAX_DIMS];
  int at = 0;
  for (int i = 0; i < n_pairs; i++) {
    int axis = expander2_axis(ectx, pairs[i][0]);
    if (axis < 0 || axis >= out_ndim || tail[axis]) return NULL;
    tail[axis] = true;
  }
  for (int axis = 0; axis < out_ndim; axis++)
    if (!tail[axis]) ordered[at++] = axis;
  for (int i = 0; i < n_pairs; i++)
    ordered[at++] = expander2_axis(ectx, pairs[i][0]);
  if (at != out_ndim) return NULL;
  for (int i = 0; i < out_ndim; i++)
    inverse[ordered[i]] = i;
  int64_t perm[POLY_MAX_DIMS];
  for (int i = 0; i < out_ndim; i++)
    perm[i] = inverse[i];
  return poly_permute(ctx, reshaped, perm, out_ndim);
}

static PolyUOp *poly_expand_wmma(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b) {
  (void)b;
  if (!u || u->op != POLY_OP_WMMA || u->n_src != 3 || u->arg.kind != POLY_ARG_TENSOR_CORE ||
      !u->arg.tensor_core.has_upcast_axes)
    return NULL;
  Expander2Context *ectx = (Expander2Context *)poly_graph_rewrite_userctx();
  if (!ectx) return NULL;
  PolyUOp *in0 = poly_contract_axis(
      ctx, ectx, u->src[0], u->arg.tensor_core.upcast_axes[0], u->arg.tensor_core.n_upcast_axes[0]
  );
  PolyUOp *in1 = poly_contract_axis(
      ctx, ectx, u->src[1], u->arg.tensor_core.upcast_axes[1], u->arg.tensor_core.n_upcast_axes[1]
  );
  if (!in0 || !in1) return NULL;
  PolyUOp *src[3] = {in0, in1, u->src[2]};
  PolyArg arg = u->arg;
  arg.tensor_core.has_upcast_axes = false;
  for (int d = 0; d < 3; d++) {
    arg.tensor_core.upcast_axes[d] = NULL;
    arg.tensor_core.n_upcast_axes[d] = 0;
  }
  PolyUOp *wmma = poly_uop_tagged_arg(ctx, POLY_OP_WMMA, u->dtype, src, 3, arg, u->tag, u->tag_arg);
  return wmma ? poly_unroll_axis(
                    ctx, ectx, wmma, u->arg.tensor_core.upcast_axes[2],
                    u->arg.tensor_core.n_upcast_axes[2]
                )
              : NULL;
}

static _Thread_local PolyPatternMatcher *g_pm_expander2 = NULL;
static PolyPatternMatcher *poly_pm_expander2(void) {
  if (g_pm_expander2) return g_pm_expander2;
  PolyRule rules[] = {
      {poly_upat_allow_any_len(poly_upat_op(POLY_OP_REDUCE, NULL, 0, "r")), poly_expand_reduce},
      {poly_upat_allow_any_len(poly_upat_op(POLY_OP_RANGE, NULL, 0, "r")), poly_expand_range},
      {poly_upat_allow_any_len(poly_upat_op(POLY_OP_WMMA, NULL, 0, "u")), poly_expand_wmma},
  };
  g_pm_expander2 =
      poly_pm_thread_cache(poly_pm_new(rules, (int)(sizeof(rules) / sizeof(rules[0]))));
  return g_pm_expander2;
}

static _Thread_local PolyPatternMatcher *g_expander2 = NULL;
static PolyPatternMatcher *poly_expander2(void) {
  if (g_expander2) return g_expander2;
  PolyPatternMatcher *expanded = poly_pm_concat(poly_pm_expander2(), poly_pm_flatten_range());
  g_expander2 = poly_pm_thread_cache(poly_pm_concat(expanded, poly_mop_cleanup()));
  poly_pm_destroy(expanded);
  return g_expander2;
}

PolyUOp *poly_apply_expander2(PolyCtx *ctx, PolyUOp *sink) {
  if (!ctx || !sink) return NULL;
  int n_topo = 0;
  PolyUOp **topo = poly_toposort_alloc(ctx, sink, &n_topo);
  if (!topo && n_topo != 0) return NULL;

  Expander2Context ectx = {0};
  ectx.axis_ids = n_topo ? malloc((size_t)n_topo * sizeof(*ectx.axis_ids)) : NULL;
  ectx.shape_axes = n_topo ? malloc((size_t)n_topo * sizeof(*ectx.shape_axes)) : NULL;
  if (n_topo && (!ectx.axis_ids || !ectx.shape_axes)) {
    free(ectx.axis_ids);
    free(ectx.shape_axes);
    poly_toposort_free(topo);
    return NULL;
  }
  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    if (!u || u->op != POLY_OP_RANGE || !poly_arg_is_range(u->arg)) continue;
    PolyAxisType type = poly_range_axis_type(u->arg);
    if (type != POLY_AXIS_UPCAST && type != POLY_AXIS_UNROLL) continue;
    int64_t id = poly_range_axis_id(u->arg);
    int found = -1;
    for (int j = 0; j < ectx.count; j++)
      if (ectx.axis_ids[j] == id) {
        found = j;
        break;
      }
    if (found >= 0)
      ectx.shape_axes[found] = ectx.count;
    else {
      ectx.axis_ids[ectx.count] = id;
      ectx.shape_axes[ectx.count] = ectx.count;
      ectx.count++;
    }
  }
  poly_toposort_free(topo);

  PolyUOp *ret = poly_graph_rewrite_ctx(ctx, sink, poly_expander2(), &ectx);
  free(ectx.axis_ids);
  free(ectx.shape_axes);
  return ret;
}

/* pm_add_loads (current tinygrad codegen/__init__.py:235-242) */

static bool codegen_shape_equal(PolyCtx *ctx, PolyUOp *a, PolyUOp *b);

static bool codegen_value_addrspace(PolyUOp *u, PolyAddrSpace *out) {
  if (!u) return false;
  if (u->op == POLY_OP_PARAM || u->op == POLY_OP_BUFFER) {
    if (out) *out = poly_program_memory_addrspace(u);
    return true;
  }
  if (u->op == POLY_OP_LOAD || u->op == POLY_OP_RANGE || u->op == POLY_OP_SPECIAL) return false;
  if ((u->op == POLY_OP_INDEX || u->op == POLY_OP_CAST || u->op == POLY_OP_AFTER ||
       u->op == POLY_OP_REDUCE || u->op == POLY_OP_STORE || u->op == POLY_OP_MSTACK ||
       u->op == POLY_OP_MSELECT || u->op == POLY_OP_END ||
       poly_opset_has(POLY_GROUP_MOVEMENT, u->op)) &&
      u->n_src > 0)
    return codegen_value_addrspace(u->src[0], out);
  if (u->op == POLY_OP_STACK || u->op == POLY_OP_WMMA || u->op == POLY_OP_GROUP ||
      poly_opset_has(POLY_GROUP_ELEMENTWISE, u->op)) {
    bool found = false;
    PolyAddrSpace addrspace = POLY_ADDR_GLOBAL;
    for (int i = 0; i < u->n_src; i++) {
      PolyAddrSpace src_addrspace;
      if (!codegen_value_addrspace(u->src[i], &src_addrspace)) continue;
      if (found && src_addrspace != addrspace) return false;
      found = true;
      addrspace = src_addrspace;
    }
    if (found && out) *out = addrspace;
    return found;
  }
  return false;
}

/* Tinygrad 2026-08-22/a9069c177a9d codegen/__init__.py:236-237. */
static bool is_shape_changing_bitcast(PolyCtx *ctx, PolyUOp *u) {
  return u && u->op == POLY_OP_BITCAST && u->n_src == 1 && !codegen_shape_equal(ctx, u, u->src[0]);
}

static PolyUOp *maybe_load(PolyCtx *ctx, PolyUOp *u) {
  PolyAddrSpace addrspace;
  if (!codegen_value_addrspace(u, &addrspace)) return u;
  if (addrspace != POLY_ADDR_GLOBAL && addrspace != POLY_ADDR_LOCAL && addrspace != POLY_ADDR_REG)
    return u;
  return poly_uop1(ctx, POLY_OP_LOAD, u->dtype, u, poly_arg_none());
}

static PolyUOp *add_value_loads(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b) {
  (void)b;
  if (!u || u->n_src <= 0 || is_shape_changing_bitcast(ctx, u)) return NULL;

  PolyUOp **src = malloc((size_t)u->n_src * sizeof(*src));
  if (!src) return NULL;
  bool changed = false;
  for (int i = 0; i < u->n_src; i++) {
    src[i] = maybe_load(ctx, u->src[i]);
    if (!src[i]) {
      free(src);
      return NULL;
    }
    changed |= src[i] != u->src[i];
  }
  PolyUOp *ret =
      changed ? poly_uop_tagged_arg(ctx, u->op, u->dtype, src, u->n_src, u->arg, u->tag, u->tag_arg)
              : NULL;
  free(src);
  return ret;
}

static PolyUOp *add_store_load(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b) {
  (void)b;
  if (!u || u->op != POLY_OP_STORE || u->n_src < 2) return NULL;
  PolyUOp *value = maybe_load(ctx, u->src[1]);
  if (!value || value == u->src[1]) return NULL;
  PolyUOp **src = malloc((size_t)u->n_src * sizeof(*src));
  if (!src) return NULL;
  memcpy(src, u->src, (size_t)u->n_src * sizeof(*src));
  src[1] = value;
  PolyUOp *ret =
      poly_uop_tagged_arg(ctx, u->op, u->dtype, src, u->n_src, u->arg, u->tag, u->tag_arg);
  free(src);
  return ret;
}

static _Thread_local PolyPatternMatcher *g_pm_add_loads = NULL;
static PolyPatternMatcher *poly_pm_add_loads(void) {
  if (g_pm_add_loads) return g_pm_add_loads;
  PolyOpSet ops = poly_opset_add(POLY_GROUP_ELEMENTWISE, POLY_OP_REDUCE);
  ops = poly_opset_add(ops, POLY_OP_WMMA);
  ops = poly_opset_add(ops, POLY_OP_STACK);
  PolyRule rules[] = {
      {poly_upat_allow_any_len(poly_upat_ops(ops, NULL, 0, "x")), add_value_loads},
      {poly_upat_allow_any_len(poly_upat_op(POLY_OP_STORE, NULL, 0, "x")), add_store_load},
  };
  g_pm_add_loads =
      poly_pm_thread_cache(poly_pm_new(rules, (int)(sizeof(rules) / sizeof(rules[0]))));
  return g_pm_add_loads;
}

/* Current tinygrad codegen/__init__.py:92-149 shape-based broadcast and
 * devectorization. Shapes, rather than renderer vector dtypes, own this stage. */

static bool codegen_shape_equal(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  int a_ndim = poly_uop_ndim(ctx, a), b_ndim = poly_uop_ndim(ctx, b);
  if (a_ndim < 0 || a_ndim != b_ndim) return false;
  for (int i = 0; i < a_ndim; i++) {
    PolyUOp *ad = poly_uop_shape_dim(ctx, a, i);
    PolyUOp *bd = poly_uop_shape_dim(ctx, b, i);
    if (ad == bd) continue;
    int64_t av = 0, bv = 0;
    if (!ad || !bd || poly_uop_const_i64(ad, &av) != 0 || poly_uop_const_i64(bd, &bv) != 0 ||
        av != bv)
      return false;
  }
  return true;
}

static PolyUOp *codegen_base(PolyUOp *u) {
  while (u && u->n_src > 0 &&
         (poly_opset_has(POLY_GROUP_MOVEMENT, u->op) || u->op == POLY_OP_DETACH))
    u = u->src[0];
  return u;
}

static bool codegen_base_is_invalid(PolyUOp *u) {
  PolyUOp *base = codegen_base(u);
  return base && base->op == POLY_OP_CONST && base->arg.kind == POLY_ARG_INVALID;
}

/* Tinygrad 2026-08-22/a9069c177a9d codegen/__init__.py:108-115. */
static PolyUOp *wmma_add(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)root;
  PolyUOp *wmma = poly_bind(b, "wmma"), *add = poly_bind(b, "add");
  if (!wmma || wmma->op != POLY_OP_WMMA || wmma->n_src != 3 || !add) return NULL;
  PolyUOp *acc =
      poly_uop2(ctx, POLY_OP_ADD, wmma->src[2]->dtype, wmma->src[2], add, poly_arg_none());
  if (!acc) return NULL;
  PolyUOp *src[3] = {wmma->src[0], wmma->src[1], acc};
  return poly_uop(ctx, POLY_OP_WMMA, wmma->dtype, src, 3, wmma->arg);
}

static bool argsort_permutation(PolyUOp *permute, int64_t *inverse, int *ndim) {
  if (!permute || permute->op != POLY_OP_PERMUTE || permute->arg.kind != POLY_ARG_INT_TUPLE ||
      permute->arg.int_tuple.n < 0 || permute->arg.int_tuple.n > POLY_MAX_DIMS)
    return false;
  int n = permute->arg.int_tuple.n;
  bool seen[POLY_MAX_DIMS] = {0};
  for (int i = 0; i < n; i++) {
    int64_t axis = permute->arg.int_tuple.vals[i];
    if (axis < 0 || axis >= n || seen[axis]) return false;
    seen[axis] = true;
    inverse[axis] = i;
  }
  *ndim = n;
  return true;
}

static PolyUOp *reshape_like(PolyCtx *ctx, PolyUOp *value, PolyUOp *shape_owner) {
  int ndim = poly_uop_ndim(ctx, shape_owner);
  if (ndim < 0 || ndim > POLY_MAX_DIMS) return NULL;
  PolyUOp *shape[POLY_MAX_DIMS];
  for (int i = 0; i < ndim; i++) {
    shape[i] = poly_uop_shape_dim(ctx, shape_owner, i);
    if (!shape[i]) return NULL;
  }
  return poly_reshape_uop(ctx, value, shape, ndim);
}

static PolyUOp *permute_wmma_add(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)root;
  PolyUOp *wmma = poly_bind(b, "wmma"), *permute = poly_bind(b, "permute");
  PolyUOp *add = poly_bind(b, "add");
  int64_t inverse[POLY_MAX_DIMS];
  int ndim = 0;
  if (!wmma || !permute || !add || !argsort_permutation(permute, inverse, &ndim)) return NULL;
  PolyUOp *moved_add = poly_permute(ctx, add, inverse, ndim);
  PolyUOp *sum =
      moved_add ? poly_uop2(ctx, POLY_OP_ADD, wmma->dtype, wmma, moved_add, poly_arg_none()) : NULL;
  return sum ? poly_permute(ctx, sum, permute->arg.int_tuple.vals, ndim) : NULL;
}

static PolyUOp *permute_reshape_wmma_add(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)root;
  PolyUOp *wmma = poly_bind(b, "wmma"), *reshape = poly_bind(b, "reshape");
  PolyUOp *permute = poly_bind(b, "permute"), *add = poly_bind(b, "add");
  int64_t inverse[POLY_MAX_DIMS];
  int ndim = 0;
  if (!wmma || !reshape || !permute || !add || !argsort_permutation(permute, inverse, &ndim))
    return NULL;
  PolyUOp *moved_add = poly_permute(ctx, add, inverse, ndim);
  moved_add = moved_add ? reshape_like(ctx, moved_add, wmma) : NULL;
  PolyUOp *sum =
      moved_add ? poly_uop2(ctx, POLY_OP_ADD, wmma->dtype, wmma, moved_add, poly_arg_none()) : NULL;
  PolyUOp *reshaped = sum ? reshape_like(ctx, sum, reshape) : NULL;
  return reshaped ? poly_permute(ctx, reshaped, permute->arg.int_tuple.vals, ndim) : NULL;
}

static _Thread_local PolyPatternMatcher *g_pm_wmma_add = NULL;
static PolyPatternMatcher *poly_pm_wmma_add(void) {
  if (g_pm_wmma_add) return g_pm_wmma_add;
  PolyUPat *wmma = poly_upat_allow_any_len(poly_upat_op(POLY_OP_WMMA, NULL, 0, "wmma"));
  PolyUPat *permuted_wmma = poly_upat_op1(
      POLY_OP_PERMUTE, poly_upat_allow_any_len(poly_upat_op(POLY_OP_WMMA, NULL, 0, "wmma")),
      "permute"
  );
  PolyUPat *reshape_src[2] = {
      poly_upat_allow_any_len(poly_upat_op(POLY_OP_WMMA, NULL, 0, "wmma")),
      poly_upat_any(NULL),
  };
  PolyUPat *permuted_reshape_wmma = poly_upat_op1(
      POLY_OP_PERMUTE, poly_upat_op(POLY_OP_RESHAPE, reshape_src, 2, "reshape"), "permute"
  );
  PolyRule rules[] = {
      {poly_upat_op2c(POLY_OP_ADD, wmma, poly_upat_any("add"), NULL), wmma_add},
      {poly_upat_op2c(POLY_OP_ADD, permuted_wmma, poly_upat_any("add"), NULL), permute_wmma_add},
      {poly_upat_op2c(POLY_OP_ADD, permuted_reshape_wmma, poly_upat_any("add"), NULL),
       permute_reshape_wmma_add},
  };
  g_pm_wmma_add = poly_pm_thread_cache(poly_pm_new(rules, (int)(sizeof(rules) / sizeof(rules[0]))));
  return g_pm_wmma_add;
}

static bool shape_dim_equal(PolyUOp *a, PolyUOp *b) {
  if (a == b) return true;
  int64_t av = 0, bv = 0;
  return a && b && poly_uop_const_i64(a, &av) == 0 && poly_uop_const_i64(b, &bv) == 0 && av == bv;
}

static int broadcast_wmma_outer_shape(PolyCtx *ctx, PolyUOp *wmma, PolyUOp **out_shape) {
  int outer_ndim = 0;
  for (int i = 0; i < wmma->n_src; i++) {
    int ndim = poly_uop_ndim(ctx, wmma->src[i]);
    if (ndim <= 0 || ndim - 1 > POLY_MAX_DIMS) return -1;
    if (ndim - 1 > outer_ndim) outer_ndim = ndim - 1;
  }
  for (int axis = 0; axis < outer_ndim; axis++) {
    PolyUOp *selected = NULL;
    for (int i = 0; i < wmma->n_src; i++) {
      int ndim = poly_uop_ndim(ctx, wmma->src[i]);
      int src_axis = axis - (outer_ndim - (ndim - 1));
      PolyUOp *dim = src_axis >= 0 ? poly_uop_shape_dim(ctx, wmma->src[i], src_axis)
                                   : poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(1));
      int64_t value = 0;
      if (!dim) return -1;
      if (poly_uop_const_i64(dim, &value) == 0 && value == 1) continue;
      if (!selected)
        selected = dim;
      else if (!shape_dim_equal(selected, dim))
        return -1;
    }
    out_shape[axis] =
        selected ? selected : poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(1));
  }
  return outer_ndim;
}

static PolyUOp *poly_broadcast_and_devec_wmma(
    PolyCtx *ctx,
    PolyUOp *wmma,
    const PolyBindings *bindings
) {
  (void)bindings;
  if (!wmma || wmma->op != POLY_OP_WMMA || wmma->n_src != 3) return NULL;
  PolyUOp *outer_shape[POLY_MAX_DIMS];
  int outer_ndim = broadcast_wmma_outer_shape(ctx, wmma, outer_shape);
  if (outer_ndim < 0) return NULL;

  bool all_same = true;
  for (int i = 0; i < wmma->n_src && all_same; i++) {
    int ndim = poly_uop_ndim(ctx, wmma->src[i]);
    if (ndim - 1 != outer_ndim) {
      all_same = false;
      break;
    }
    for (int axis = 0; axis < outer_ndim; axis++)
      if (!shape_dim_equal(poly_uop_shape_dim(ctx, wmma->src[i], axis), outer_shape[axis])) {
        all_same = false;
        break;
      }
  }
  if (all_same) return NULL;

  PolyUOp *expanded[3];
  for (int i = 0; i < 3; i++) {
    int ndim = poly_uop_ndim(ctx, wmma->src[i]);
    PolyUOp *shape[POLY_MAX_DIMS];
    for (int axis = 0; axis < outer_ndim; axis++)
      shape[axis] = outer_shape[axis];
    shape[outer_ndim] = poly_uop_shape_dim(ctx, wmma->src[i], ndim - 1);
    expanded[i] =
        shape[outer_ndim] ? poly_expand_uop(ctx, wmma->src[i], shape, outer_ndim + 1) : NULL;
    if (!expanded[i]) return NULL;
  }

  int64_t dims[POLY_MAX_DIMS];
  size_t n_fragments = 1;
  for (int axis = 0; axis < outer_ndim; axis++) {
    if (poly_uop_const_i64(outer_shape[axis], &dims[axis]) != 0 || dims[axis] < 0 ||
        (dims[axis] && n_fragments > SIZE_MAX / (size_t)dims[axis]))
      return NULL;
    n_fragments *= (size_t)dims[axis];
  }
  if (n_fragments == 0 || n_fragments > UINT16_MAX) return NULL;
  PolyUOp **fragments = malloc(n_fragments * sizeof(*fragments));
  if (!fragments) return NULL;
  for (size_t lane = 0; lane < n_fragments; lane++) {
    size_t rem = lane;
    PolyUOp *coords[POLY_MAX_DIMS];
    for (int axis = outer_ndim - 1; axis >= 0; axis--) {
      int64_t coord = dims[axis] ? (int64_t)(rem % (size_t)dims[axis]) : 0;
      if (dims[axis]) rem /= (size_t)dims[axis];
      coords[axis] = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(coord));
    }
    PolyUOp *src[3];
    for (int i = 0; i < 3; i++) {
      src[i] = poly_uop_index(ctx, expanded[i], coords, outer_ndim);
      if (!src[i]) {
        free(fragments);
        return NULL;
      }
    }
    fragments[lane] = poly_uop(ctx, POLY_OP_WMMA, wmma->dtype, src, 3, wmma->arg);
    if (!fragments[lane]) {
      free(fragments);
      return NULL;
    }
  }
  PolyUOp *stack = poly_uop_stack(ctx, fragments, (int)n_fragments);
  free(fragments);
  return stack ? reshape_like(ctx, stack, wmma) : NULL;
}

static PolyUOp *poly_expand_broadcast(PolyCtx *ctx, PolyUOp *x, const PolyBindings *b) {
  (void)b;
  if (!x || (!poly_opset_has(POLY_GROUP_BROADCASTABLE, x->op) && x->op != POLY_OP_STORE) ||
      x->n_src <= 0)
    return NULL;
  PolyUOp *shape[POLY_MAX_DIMS];
  int ndim = poly_broadcast_shape(ctx, x->src, x->n_src, shape, POLY_MAX_DIMS);
  if (ndim < 0) return NULL;
  PolyUOp **src = malloc((size_t)x->n_src * sizeof(*src));
  if (!src) return NULL;
  bool changed = false;
  for (int i = 0; i < x->n_src; i++) {
    src[i] = poly_expand_uop(ctx, x->src[i], shape, ndim);
    if (!src[i]) {
      free(src);
      return NULL;
    }
    changed |= src[i] != x->src[i];
  }
  PolyUOp *ret =
      changed ? poly_uop_tagged_arg(ctx, x->op, x->dtype, src, x->n_src, x->arg, x->tag, x->tag_arg)
              : NULL;
  free(src);
  return ret;
}

static PolyUOp *poly_empty_index(PolyCtx *ctx, PolyUOp *idx, const PolyBindings *b) {
  (void)ctx;
  (void)b;
  return idx && idx->op == POLY_OP_INDEX && idx->n_src == 1 ? idx->src[0] : NULL;
}

/* Current tinygrad codegen/__init__.py::devectorizer2.  Once a shaped
 * coordinate has been expanded, INDEX owns one scalar memory occurrence per
 * STACK lane.  Keeping the STACK inside INDEX leaves a shaped address for the
 * old dtype-vector pass and aliases every lane to zero. */
static PolyUOp *poly_index_buffer_stack(PolyCtx *ctx, PolyUOp *idx, const PolyBindings *b) {
  (void)b;
  if (!idx || idx->op != POLY_OP_INDEX || idx->n_src != 2) return NULL;
  PolyUOp *buf = idx->src[0], *stack = idx->src[1];
  if (!buf || (buf->op != POLY_OP_PARAM && buf->op != POLY_OP_BUFFER) || !stack ||
      stack->op != POLY_OP_STACK)
    return NULL;
  PolyUOp **lanes = stack->n_src ? malloc((size_t)stack->n_src * sizeof(*lanes)) : NULL;
  if (stack->n_src && !lanes) return NULL;
  for (int i = 0; i < stack->n_src; i++) {
    lanes[i] = poly_uop_index(ctx, buf, &stack->src[i], 1);
    if (!lanes[i]) {
      free(lanes);
      return NULL;
    }
  }
  PolyUOp *ret = poly_uop_stack(ctx, lanes, stack->n_src);
  free(lanes);
  return ret;
}

static PolyUOp *poly_index_buffer_reshape(PolyCtx *ctx, PolyUOp *idx, const PolyBindings *b) {
  (void)b;
  if (!idx || idx->op != POLY_OP_INDEX || idx->n_src != 2) return NULL;
  PolyUOp *buf = idx->src[0], *reshape = idx->src[1];
  if (!buf || (buf->op != POLY_OP_PARAM && buf->op != POLY_OP_BUFFER) || !reshape ||
      reshape->op != POLY_OP_RESHAPE || reshape->n_src < 1)
    return NULL;
  PolyUOp *indexed = poly_uop_index(ctx, buf, &reshape->src[0], 1);
  return indexed ? poly_reshape_uop(ctx, indexed, reshape->src + 1, reshape->n_src - 1) : NULL;
}

static PolyUOp *poly_reshape_void(PolyCtx *ctx, PolyUOp *reshape, const PolyBindings *b) {
  (void)ctx;
  (void)b;
  return reshape && reshape->op == POLY_OP_RESHAPE && poly_dtype_eq(reshape->dtype, POLY_VOID) &&
                 reshape->n_src > 0
             ? reshape->src[0]
             : NULL;
}

static PolyUOp *poly_reshape_single_to_scalar(
    PolyCtx *ctx,
    PolyUOp *reshape,
    const PolyBindings *b
) {
  (void)b;
  if (!reshape || reshape->op != POLY_OP_RESHAPE || reshape->n_src < 1 ||
      poly_uop_ndim(ctx, reshape) != 0)
    return NULL;
  PolyShape in_shape = poly_uop_max_shape_cached(ctx, reshape->src[0]);
  if (in_shape.ndim != 1 || in_shape.dims[0] != 1) return NULL;
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(0));
  return poly_uop_index(ctx, reshape->src[0], &zero, 1);
}

static PolyUOp *poly_expand_scalar_to_stack(PolyCtx *ctx, PolyUOp *expand, const PolyBindings *b) {
  (void)b;
  if (!expand || expand->op != POLY_OP_EXPAND || expand->n_src < 1 ||
      poly_uop_ndim(ctx, expand->src[0]) != 0)
    return NULL;
  PolyShape out_shape = poly_uop_max_shape_cached(ctx, expand);
  int64_t n = poly_shape_numel(out_shape);
  if (out_shape.ndim != 1 || n < 0 || n > UINT16_MAX || out_shape.dims[0] != n) return NULL;
  PolyUOp **src = n ? malloc((size_t)n * sizeof(*src)) : NULL;
  if (n && !src) return NULL;
  for (int64_t i = 0; i < n; i++)
    src[i] = expand->src[0];
  PolyUOp *ret = poly_uop_stack(ctx, src, (int)n);
  free(src);
  return ret;
}

static PolyUOp *poly_do_devectorize(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b) {
  (void)b;
  if (!u || (!poly_opset_has(POLY_GROUP_ELEMENTWISE, u->op) && u->op != POLY_OP_LOAD &&
             u->op != POLY_OP_STORE))
    return NULL;
  int ndim = poly_uop_ndim(ctx, u);
  if (ndim <= 0 || ndim > POLY_MAX_DIMS) return NULL;
  for (int i = 0; i < u->n_src; i++)
    if (!codegen_base_is_invalid(u->src[i]) && !codegen_shape_equal(ctx, u->src[i], u)) return NULL;

  int64_t dims[POLY_MAX_DIMS];
  PolyUOp *shape[POLY_MAX_DIMS];
  size_t n_values = 1;
  for (int i = 0; i < ndim; i++) {
    shape[i] = poly_uop_shape_dim(ctx, u, i);
    if (!shape[i] || poly_uop_const_i64(shape[i], &dims[i]) != 0 || dims[i] < 0) return NULL;
    if (dims[i] != 0 && n_values > SIZE_MAX / (size_t)dims[i]) return NULL;
    n_values *= (size_t)dims[i];
  }
  if (n_values > (size_t)INT_MAX || n_values > UINT16_MAX) return NULL;

  PolyUOp **values = n_values ? calloc(n_values, sizeof(*values)) : NULL;
  PolyUOp **src = u->n_src ? malloc((size_t)u->n_src * sizeof(*src)) : NULL;
  if ((n_values && !values) || (u->n_src && !src)) {
    free(values);
    free(src);
    return NULL;
  }
  for (size_t lane = 0; lane < n_values; lane++) {
    size_t rem = lane;
    PolyUOp *coords[POLY_MAX_DIMS];
    for (int axis = ndim - 1; axis >= 0; axis--) {
      int64_t coord = dims[axis] ? (int64_t)(rem % (size_t)dims[axis]) : 0;
      if (dims[axis]) rem /= (size_t)dims[axis];
      coords[axis] = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(coord));
    }
    for (int i = 0; i < u->n_src; i++) {
      src[i] = codegen_base_is_invalid(u->src[i]) ? codegen_base(u->src[i])
                                                  : poly_uop_index(ctx, u->src[i], coords, ndim);
      if (!src[i]) goto fail;
    }
    PolyDType dtype = u->dtype;
    if (u->op != POLY_OP_LOAD && u->op != POLY_OP_STORE)
      (void)poly_dtype_from_uop(u->op, src, u->n_src, u->arg, u->dtype, &dtype);
    values[lane] =
        poly_uop_tagged_arg(ctx, u->op, dtype, src, u->n_src, u->arg, u->tag, u->tag_arg);
    if (!values[lane]) goto fail;
  }

  PolyUOp *ret = NULL;
  if (u->op == POLY_OP_STORE) {
    ret = n_values == 1
              ? values[0]
              : poly_uop(ctx, POLY_OP_GROUP, POLY_VOID, values, (int)n_values, poly_arg_none());
  } else {
    PolyUOp *stack = poly_uop_stack(ctx, values, (int)n_values);
    ret = stack ? poly_reshape_uop(ctx, stack, shape, ndim) : NULL;
  }
  free(src);
  free(values);
  return ret;

fail:
  free(src);
  free(values);
  return NULL;
}

static PolyUOp *poly_do_stack_wmma(PolyCtx *ctx, PolyUOp *wmma, const PolyBindings *b) {
  (void)b;
  if (!wmma || wmma->op != POLY_OP_WMMA || poly_uop_ndim(ctx, wmma) != 1) return NULL;
  bool ready = true;
  for (int i = 0; i < wmma->n_src; i++)
    ready &= wmma->src[i]->op == POLY_OP_STACK || wmma->src[i]->op == POLY_OP_WMMA;
  if (ready) return NULL;

  PolyUOp **src = malloc((size_t)wmma->n_src * sizeof(*src));
  if (!src) return NULL;
  for (int i = 0; i < wmma->n_src; i++) {
    if (wmma->src[i]->op == POLY_OP_STACK) {
      src[i] = wmma->src[i];
      continue;
    }
    int64_t lanes = poly_uop_max_numel(ctx, wmma->src[i]);
    if (lanes < 0 || lanes > UINT16_MAX) goto fail;
    PolyUOp **values = lanes ? malloc((size_t)lanes * sizeof(*values)) : NULL;
    if (lanes && !values) goto fail;
    for (int64_t lane = 0; lane < lanes; lane++) {
      PolyUOp *idx = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(lane));
      values[lane] = poly_uop_index(ctx, wmma->src[i], &idx, 1);
      if (!values[lane]) {
        free(values);
        goto fail;
      }
    }
    src[i] = poly_uop_stack(ctx, values, (int)lanes);
    free(values);
    if (!src[i]) goto fail;
  }
  PolyUOp *ret = poly_uop_tagged_arg(
      ctx, wmma->op, wmma->dtype, src, wmma->n_src, wmma->arg, wmma->tag, wmma->tag_arg
  );
  free(src);
  return ret;

fail:
  free(src);
  return NULL;
}

static _Thread_local PolyPatternMatcher *g_pm_expand_broadcast = NULL;

/* Current Tinygrad codegen/__init__.py:add_local_buffer. */
static PolyUOp *poly_add_local_buffer(PolyCtx *ctx, PolyUOp *stage, const PolyBindings *bindings) {
  (void)bindings;
  if (!stage || stage->op != POLY_OP_STAGE || stage->n_src < 1 ||
      stage->arg.kind != POLY_ARG_BUFFERIZE_OPTS)
    return NULL;
  PolyShape max_shape = poly_uop_max_shape_cached(ctx, stage);
  if (max_shape.ndim < 0 || max_shape.ndim > POLY_MAX_DIMS) return NULL;
  int *next_local_slot = (int *)poly_graph_rewrite_userctx();
  if (!next_local_slot) return NULL;
  PolyUOp *buf = poly_uop_placeholder(
      ctx, max_shape.dims, max_shape.ndim, stage->dtype, (*next_local_slot)++,
      poly_bufferize_arg_addrspace(stage->arg), NULL, false
  );
  if (!buf) return NULL;
  PolyUOp **index_src = malloc((size_t)stage->n_src * sizeof(*index_src));
  if (!index_src) return NULL;
  index_src[0] = buf;
  for (int i = 1; i < stage->n_src; i++)
    index_src[i] = stage->src[i];
  PolyUOp *index =
      poly_uop(ctx, POLY_OP_INDEX, stage->dtype, index_src, stage->n_src, poly_arg_none());
  free(index_src);
  PolyUOp *store = index ? poly_store_val(ctx, index, stage->src[0]) : NULL;
  if (!store) return NULL;
  if (stage->n_src > 1) {
    PolyUOp **end_src = malloc((size_t)stage->n_src * sizeof(*end_src));
    if (!end_src) return NULL;
    end_src[0] = store;
    for (int i = 1; i < stage->n_src; i++)
      end_src[i] = stage->src[i];
    store = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, stage->n_src, poly_arg_none());
    free(end_src);
  }
  return store ? poly_uop2(ctx, POLY_OP_AFTER, stage->dtype, buf, store, poly_arg_none()) : NULL;
}

static PolyPatternMatcher *poly_pm_add_local_buffers(void) {
  static _Thread_local PolyPatternMatcher *pm = NULL;
  if (pm) return pm;
  PolyRule rules[] = {
      {poly_upat_allow_any_len(poly_upat_op(POLY_OP_STAGE, NULL, 0, "x")), poly_add_local_buffer},
  };
  PolyPatternMatcher *add = poly_pm_new(rules, (int)(sizeof(rules) / sizeof(rules[0])));
  pm = poly_pm_thread_cache(poly_pm_concat(add, poly_pm_mops()));
  poly_pm_destroy(add);
  return pm;
}

static PolyPatternMatcher *poly_pm_expand_broadcast(void) {
  if (g_pm_expand_broadcast) return g_pm_expand_broadcast;
  PolyOpSet ops = poly_opset_add(POLY_GROUP_BROADCASTABLE, POLY_OP_STORE);
  PolyRule rules[] = {
      {poly_upat_allow_any_len(poly_upat_ops(ops, NULL, 0, "x")), poly_expand_broadcast},
      {poly_upat_allow_any_len(poly_upat_op(POLY_OP_WMMA, NULL, 0, "b")),
       poly_broadcast_and_devec_wmma},
  };
  PolyPatternMatcher *broadcast = poly_pm_new(rules, (int)(sizeof(rules) / sizeof(rules[0])));
  g_pm_expand_broadcast = poly_pm_thread_cache(poly_pm_concat(poly_pm_wmma_add(), broadcast));
  poly_pm_destroy(broadcast);
  return g_pm_expand_broadcast;
}

static _Thread_local PolyPatternMatcher *g_pm_devectorizer2 = NULL;
static PolyPatternMatcher *poly_devectorizer2(void) {
  if (g_pm_devectorizer2) return g_pm_devectorizer2;
  PolyOpSet shaped =
      poly_opset_add(poly_opset_add(POLY_GROUP_ELEMENTWISE, POLY_OP_LOAD), POLY_OP_STORE);
  PolyRule rules[] = {
      {poly_upat_allow_any_len(poly_upat_ops(shaped, NULL, 0, "u")), poly_do_devectorize},
      {poly_upat_allow_any_len(poly_upat_op(POLY_OP_INDEX, NULL, 0, "idx")), poly_empty_index},
      {poly_upat_allow_any_len(poly_upat_op(POLY_OP_WMMA, NULL, 0, "u")), poly_do_stack_wmma},
      {poly_upat_allow_any_len(poly_upat_op(POLY_OP_INDEX, NULL, 0, "idx")),
       poly_index_buffer_stack},
      {poly_upat_allow_any_len(poly_upat_op(POLY_OP_INDEX, NULL, 0, "idx")),
       poly_index_buffer_reshape},
      {poly_upat_allow_any_len(poly_upat_op(POLY_OP_RESHAPE, NULL, 0, "reshape")),
       poly_reshape_void},
      {poly_upat_allow_any_len(poly_upat_op(POLY_OP_RESHAPE, NULL, 0, "reshape")),
       poly_reshape_single_to_scalar},
      {poly_upat_allow_any_len(poly_upat_op(POLY_OP_EXPAND, NULL, 0, "expand")),
       poly_expand_scalar_to_stack},
  };
  PolyPatternMatcher *specific = poly_pm_new(rules, (int)(sizeof(rules) / sizeof(rules[0])));
  PolyPatternMatcher *movement = poly_pm_concat(poly_mop_cleanup(), poly_pm_mops());
  g_pm_devectorizer2 = poly_pm_thread_cache(poly_pm_concat(movement, specific));
  poly_pm_destroy(movement);
  poly_pm_destroy(specific);
  return g_pm_devectorizer2;
}

/* Current tinygrad codegen/__init__.py:134-137 ew_devectorizer applies the
 * same shape-based expansion only to remaining elementwise UOps after memory
 * coalescing. LOAD/STORE must stay shaped so the renderer performs the
 * coalesced access. */
static _Thread_local PolyPatternMatcher *g_ew_devectorizer = NULL;
static PolyPatternMatcher *poly_ew_devectorizer(void) {
  if (g_ew_devectorizer) return g_ew_devectorizer;
  PolyRule rules[] = {
      {poly_upat_allow_any_len(poly_upat_ops(POLY_GROUP_ELEMENTWISE, NULL, 0, "b")),
       poly_do_devectorize},
  };
  g_ew_devectorizer =
      poly_pm_thread_cache(poly_pm_new(rules, (int)(sizeof(rules) / sizeof(rules[0]))));
  return g_ew_devectorizer;
}

static _Thread_local PolyPatternMatcher *g_devectorizer2_stage = NULL;
static PolyPatternMatcher *poly_devectorizer2_stage(void) {
  if (g_devectorizer2_stage) return g_devectorizer2_stage;
  /* Current tinygrad codegen/__init__.py composes these three matchers in one
   * graph_rewrite fixed point.  In particular indexing_simplify must see an
   * INDEX produced by devectorizer2 before the rewrite leaves this stage. */
  PolyPatternMatcher *symbolic_devectorizer =
      poly_pm_concat(poly_symbolic_simple(), poly_devectorizer2());
  g_devectorizer2_stage =
      poly_pm_thread_cache(poly_pm_concat(symbolic_devectorizer, poly_indexing_simplify()));
  poly_pm_destroy(symbolic_devectorizer);
  return g_devectorizer2_stage;
}

PolyUOp *poly_apply_devectorizer2_stage(PolyCtx *ctx, PolyUOp *sink, PolyRendererCaps caps) {
  if (!ctx || !sink) return sink;
  (void)caps;
  return poly_graph_rewrite(ctx, sink, poly_devectorizer2_stage());
}

PolyUOp *poly_apply_expand_broadcast_stage(PolyCtx *ctx, PolyUOp *sink) {
  if (!ctx || !sink) return sink;
  return poly_graph_rewrite(ctx, sink, poly_pm_expand_broadcast());
}

PolyUOp *poly_apply_post_index_symbolic_stage(PolyCtx *ctx, PolyUOp *sink) {
  if (!ctx || !sink) return sink;
  PolyPatternMatcher *simple_lower =
      poly_pm_concat(poly_symbolic_simple(), poly_pm_lower_index_dtype());
  PolyPatternMatcher *lower_index =
      simple_lower ? poly_pm_concat(simple_lower, poly_indexing_simplify()) : NULL;
  poly_pm_destroy(simple_lower);
  if (!lower_index) return NULL;
  /* Tinygrad 2026-08-22/a9069c177a9d codegen/__init__.py:354 supplies
   * one ctx dict for weak-source lowering across this complete rewrite. */
  PolyMap *lower_cache = poly_map_new(256);
  if (!lower_cache) {
    poly_pm_destroy(lower_index);
    return NULL;
  }
  sink = poly_graph_rewrite_ctx(ctx, sink, lower_index, lower_cache);
  poly_map_destroy(lower_cache);
  poly_pm_destroy(lower_index);
  return poly_graph_rewrite(ctx, sink, poly_symbolic());
}

/* Tinygrad 2026-08-22/a9069c177a9d codegen/__init__.py:170-184. */
static PolyUOp *poly_fix_group_for_reduce(
    PolyCtx *ctx,
    PolyUOp *red,
    const PolyBindings *bindings
) {
  (void)bindings;
  if (!red || red->op != POLY_OP_REDUCE || red->n_src < 2) return NULL;

  PolyUOp *group_ranges[POLY_MAX_DIMS], *other_ranges[POLY_MAX_DIMS];
  int n_group = 0, n_other = 0;
  for (int i = 1; i < red->n_src; i++) {
    PolyUOp *range = red->src[i];
    if (!range || range->op != POLY_OP_RANGE) continue;
    if (poly_range_axis_type(range->arg) == POLY_AXIS_GROUP_REDUCE) {
      if (n_group == POLY_MAX_DIMS) return NULL;
      group_ranges[n_group++] = range;
    } else {
      if (n_other == POLY_MAX_DIMS) return NULL;
      other_ranges[n_other++] = range;
    }
  }
  if (n_group == 0) return NULL;

  int n_topo = 0;
  PolyUOp **topo = poly_toposort_alloc(ctx, red, &n_topo);
  if (!topo) return NULL;
  PolyUOp *upstream_locals[POLY_MAX_DIMS];
  int n_upstream = 0;
  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    if (!u || u->op != POLY_OP_RANGE || poly_range_axis_type(u->arg) != POLY_AXIS_LOCAL) continue;
    bool duplicate = false;
    for (int j = 0; j < n_upstream; j++)
      duplicate |= upstream_locals[j] == u;
    if (!duplicate) {
      if (n_upstream == POLY_MAX_DIMS) {
        poly_toposort_free(topo);
        return NULL;
      }
      upstream_locals[n_upstream++] = u;
    }
  }
  poly_toposort_free(topo);

  PolyUOp *partial_src[1 + POLY_MAX_DIMS] = {red->src[0]};
  for (int i = 0; i < n_other; i++)
    partial_src[i + 1] = other_ranges[i];
  PolyUOp *partial = poly_uop_tagged_arg(
      ctx, red->op, red->dtype, partial_src, n_other + 1, red->arg, red->tag, red->tag_arg
  );

  PolyUOp *stage_src[1 + 2 * POLY_MAX_DIMS] = {partial};
  int n_stage = 1;
  for (int i = 0; i < n_upstream; i++)
    stage_src[n_stage++] = upstream_locals[i];
  for (int i = 0; i < n_group; i++)
    stage_src[n_stage++] = group_ranges[i];
  PolyUOp *stage = poly_uop(
      ctx, POLY_OP_STAGE, red->dtype, stage_src, n_stage,
      poly_arg_bufferize_opts(NULL, POLY_ADDR_LOCAL, true)
  );

  PolyUOp *reduce_loop[POLY_MAX_DIMS];
  for (int i = 0; i < n_group; i++) {
    PolyUOp *range = group_ranges[i];
    reduce_loop[i] = poly_uop_tagged_arg(
        ctx, range->op, range->dtype, range->src, range->n_src,
        poly_arg_range(poly_range_axis_id(range->arg) + 100, POLY_AXIS_REDUCE), range->tag,
        range->tag_arg
    );
  }

  PolyUOp *index_src[1 + 2 * POLY_MAX_DIMS] = {stage};
  int n_index = 1;
  for (int i = 0; i < n_upstream; i++)
    index_src[n_index++] = upstream_locals[i];
  for (int i = 0; i < n_group; i++)
    index_src[n_index++] = reduce_loop[i];
  PolyUOp *index = poly_uop(ctx, POLY_OP_INDEX, red->dtype, index_src, n_index, poly_arg_none());

  PolyUOp *final_src[1 + POLY_MAX_DIMS] = {index};
  for (int i = 0; i < n_group; i++)
    final_src[i + 1] = reduce_loop[i];
  return poly_uop_tagged_arg(
      ctx, red->op, red->dtype, final_src, n_group + 1, red->arg, red->tag, red->tag_arg
  );
}

/* Public wrapper for heuristic (used by tests) */
PolyUOp *poly_apply_opts_heuristic_ex(PolyCtx *ctx, PolyUOp *sink, PolyRendererCaps caps) {
  return poly_apply_opts_heuristic(ctx, sink, caps);
}

/* Public accessors for individual passes (used by CUDA linearizer) */

PolyUOp *poly_apply_tc_opt(PolyCtx *ctx, PolyUOp *sink, PolyRendererCaps caps) {
  if (caps.n_tensor_cores <= 0) return sink;
  OptScheduler s;
  sched_init(&s, ctx, sink);
  if (!sched_can_optimize(&s)) return sink;

  int n_reduce = 0;
  for (int i = 0; i < s.n_rngs; i++)
    if (s.types[i] == POLY_AXIS_GROUP_REDUCE || s.types[i] == POLY_AXIS_REDUCE) n_reduce++;

  int tc_opt_env = poly_getenv_int("TC_OPT", 0);
  int use_tc_env = poly_getenv_int("TC", 1);

  if (use_tc_env > 0 && (n_reduce == 1 || tc_opt_env >= 1)) {
    OptScheduler tk;
    sched_copy(&tk, &s);
    PolyUOp *tc_axes[3];
    bool tc_ok = sched_apply_tc_opt(
        &tk, 0, -1, tc_opt_env, use_tc_env, caps.device, caps.tensor_cores, caps.n_tensor_cores,
        tc_axes
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

static PolyPatternMatcher *poly_pm_reduce_local(void) {
  static _Thread_local PolyPatternMatcher *pm = NULL;
  if (pm) return pm;
  PolyRule rules[] = {
      {poly_upat_allow_any_len(poly_upat_op(POLY_OP_REDUCE, NULL, 0, "x")),
       poly_fix_group_for_reduce},
      {poly_upat_allow_any_len(
           poly_upat_op2c(POLY_OP_REDUCE, poly_upat_any(NULL), poly_upat_any(NULL), "r")
       ),
       rule_reduce_to_acc},
      {poly_upat_op1(POLY_OP_REDUCE, poly_upat_any(NULL), "r"), poly_expand_horizontal_reduce},
      {poly_upat_allow_any_len(poly_upat_op(POLY_OP_SINK, NULL, 0, "sink")),
       rule_merge_reduce_ends},
  };
  PolyPatternMatcher *local = poly_pm_new(rules, (int)(sizeof(rules) / sizeof(rules[0])));
  PolyPatternMatcher *with_wmma = poly_pm_concat(poly_pm_wmma_add(), local);
  pm = poly_pm_thread_cache(poly_pm_concat(with_wmma, poly_pm_clean_up_group_sink()));
  poly_pm_destroy(with_wmma);
  poly_pm_destroy(local);
  return pm;
}

PolyUOp *poly_apply_pm_reduce(PolyCtx *ctx, PolyUOp *sink) {
  static _Thread_local PolyPatternMatcher *remove_reduce = NULL;
  if (!remove_reduce)
    remove_reduce =
        poly_pm_thread_cache(poly_pm_concat(poly_mop_cleanup(), poly_pm_reduce_local()));
  ReduceContext local_ctx = {0};
  PolyUOp *out = poly_graph_rewrite_ctx(ctx, sink, remove_reduce, &local_ctx);
  reduce_ctx_clear(&local_ctx);
  return out;
}

/* Current tinygrad codegen/__init__.py:252-257 makes the operand conversion
 * explicit at the final-program boundary, after index dtype lowering and
 * before decompositions.  Tensor construction intentionally retains the raw
 * integer/bool operand while the transcendental result carries its promoted
 * float dtype. */
static PolyUOp *rule_cast_float_alu_operand(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b) {
  PolyUOp *x = poly_bind(b, "x");
  if (!u || !x || u->n_src != 1 || poly_dtype_eq(x->dtype, u->dtype)) return NULL;
  PolyUOp *cast = poly_uop1(ctx, POLY_OP_CAST, u->dtype, x, poly_arg_none());
  if (!cast) return NULL;
  PolyUOp *src[1] = {cast};
  return poly_uop_tagged_arg(ctx, u->op, u->dtype, src, 1, u->arg, u->tag, u->tag_arg);
}

static _Thread_local PolyPatternMatcher *g_pm_cast_float_alu = NULL;
static PolyPatternMatcher *poly_pm_cast_float_alu(void) {
  if (g_pm_cast_float_alu) return g_pm_cast_float_alu;
  PolyOpSet ops = {{0, 0}};
  ops = poly_opset_add(ops, POLY_OP_SIN);
  ops = poly_opset_add(ops, POLY_OP_LOG2);
  ops = poly_opset_add(ops, POLY_OP_EXP2);
  ops = poly_opset_add(ops, POLY_OP_SQRT);
  ops = poly_opset_add(ops, POLY_OP_RECIPROCAL);
  PolyRule rules[] = {
      {poly_upat_ops1(ops, poly_upat_any("x"), "u"), rule_cast_float_alu_operand},
  };
  g_pm_cast_float_alu =
      poly_pm_thread_cache(poly_pm_new(rules, (int)(sizeof(rules) / sizeof(rules[0]))));
  return g_pm_cast_float_alu;
}

/* Current tinygrad/codegen/__init__.py:285-287.  Program literals are
 * spelled CAST(strong_dtype, CONST(weak_kind, value)); the walk is single-pass
 * so the newly created weak literal is not visited again. */
static PolyUOp *rule_casted_const(PolyCtx *ctx, PolyUOp *c, const PolyBindings *b) {
  (void)b;
  if (!c || c->op != POLY_OP_CONST || poly_dtype_is_weak(c->dtype)) return NULL;
  PolyDType weak = poly_dtype_weak(c->dtype);
  PolyUOp *literal = poly_uop0(ctx, POLY_OP_CONST, weak, c->arg);
  return literal ? poly_uop1(ctx, POLY_OP_CAST, c->dtype, literal, poly_arg_dtype(c->dtype)) : NULL;
}

static _Thread_local PolyPatternMatcher *g_pm_casted_consts = NULL;
static PolyPatternMatcher *poly_pm_casted_consts(void) {
  if (g_pm_casted_consts) return g_pm_casted_consts;
  PolyRule rules[] = {
      {poly_upat_op(POLY_OP_CONST, NULL, 0, "c"), rule_casted_const},
  };
  g_pm_casted_consts =
      poly_pm_thread_cache(poly_pm_new(rules, (int)(sizeof(rules) / sizeof(rules[0]))));
  return g_pm_casted_consts;
}

static bool codegen_barrier_gate(PolyUOp *u) {
  return u && u->op != POLY_OP_BARRIER;
}

static bool codegen_is_local_store(PolyUOp *u) {
  return u && u->op == POLY_OP_STORE && u->n_src > 0 &&
         poly_program_memory_is(u->src[0], POLY_ADDR_LOCAL);
}

/* Current tinygrad/codegen/__init__.py:261-267. */
static PolyUOp *rule_add_raw_barrier(PolyCtx *ctx, PolyUOp *after, const PolyBindings *b) {
  (void)b;
  if (!after || after->op != POLY_OP_AFTER || after->n_src < 2 ||
      !poly_program_memory_is(after, POLY_ADDR_LOCAL))
    return NULL;
  PolyUOp *deps =
      poly_uop(ctx, POLY_OP_SINK, POLY_VOID, after->src + 1, after->n_src - 1, poly_arg_none());
  if (!deps) return NULL;
  int n_topo = 0;
  PolyUOp **topo = poly_toposort_ex_alloc(ctx, deps, &n_topo, codegen_barrier_gate, true);
  bool has_local_store = false;
  for (int i = 0; topo && i < n_topo; i++)
    has_local_store |= codegen_is_local_store(topo[i]);
  poly_toposort_free(topo);
  if (!has_local_store) return NULL;
  PolyUOp *barrier =
      poly_uop(ctx, POLY_OP_BARRIER, POLY_VOID, after->src + 1, after->n_src - 1, poly_arg_none());
  return barrier ? poly_uop2(ctx, POLY_OP_AFTER, after->dtype, after->src[0], barrier, after->arg)
                 : NULL;
}

static bool codegen_ptr_in(PolyUOp **items, int n, PolyUOp *item) {
  for (int i = 0; i < n; i++)
    if (items[i] == item) return true;
  return false;
}

/* Current tinygrad/codegen/__init__.py:269-278. */
static PolyUOp *rule_add_war_barrier(PolyCtx *ctx, PolyUOp *end, const PolyBindings *b) {
  (void)b;
  if (!end || end->op != POLY_OP_END || end->n_src < 2 ||
      (end->src[0] && end->src[0]->op == POLY_OP_BARRIER))
    return NULL;

  PolyUOp **ranges = malloc((size_t)(end->n_src - 1) * sizeof(*ranges));
  if (!ranges) return NULL;
  int n_ranges = 0;
  for (int i = 1; i < end->n_src; i++) {
    PolyUOp *r = end->src[i];
    if (!r || r->op != POLY_OP_RANGE || r->arg.kind != POLY_ARG_RANGE) continue;
    PolyAxisType axis = poly_range_axis_type(r->arg);
    int64_t lo = 0, hi = 0;
    poly_uop_minmax(ctx, r, &lo, &hi);
    if ((axis == POLY_AXIS_REDUCE || axis == POLY_AXIS_WEAK || axis == POLY_AXIS_LOOP) && hi > 0)
      ranges[n_ranges++] = r;
  }
  if (n_ranges == 0) {
    free(ranges);
    return NULL;
  }

  int n_topo = 0;
  PolyUOp **topo = poly_toposort_alloc(ctx, end->src[0], &n_topo);
  if (!topo || n_topo <= 0) {
    free(ranges);
    return NULL;
  }
  PolyUOp **store_bufs = malloc((size_t)n_topo * sizeof(*store_bufs));
  PolyUOp **loads = malloc((size_t)n_topo * sizeof(*loads));
  PolyUOp **active = malloc((size_t)n_topo * sizeof(*active));
  if (!store_bufs || !loads || !active) {
    free(active);
    free(loads);
    free(store_bufs);
    free(ranges);
    poly_toposort_free(topo);
    return NULL;
  }
  int n_store_bufs = 0, n_loads = 0;
  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    if (!codegen_is_local_store(u)) continue;
    int n_active = poly_uop_ranges(ctx, u, active, n_topo);
    bool inside = false;
    for (int ri = 0; ri < n_ranges && !inside; ri++)
      inside = codegen_ptr_in(active, n_active, ranges[ri]);
    PolyUOp *buf = inside ? poly_uop_buf_uop(ctx, u->src[0]) : NULL;
    if (buf && !codegen_ptr_in(store_bufs, n_store_bufs, buf)) store_bufs[n_store_bufs++] = buf;
  }
  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    if (!u || u->op != POLY_OP_LOAD || u->n_src < 1) continue;
    PolyUOp *buf = poly_uop_buf_uop(ctx, u->src[0]);
    if (buf && codegen_ptr_in(store_bufs, n_store_bufs, buf)) loads[n_loads++] = u;
  }
  poly_toposort_free(topo);
  if (n_loads == 0) {
    free(active);
    free(loads);
    free(store_bufs);
    free(ranges);
    return NULL;
  }

  PolyUOp **barrier_src = malloc((size_t)(n_loads + 1) * sizeof(*barrier_src));
  PolyUOp **end_src = malloc((size_t)end->n_src * sizeof(*end_src));
  if (!barrier_src || !end_src) {
    free(end_src);
    free(barrier_src);
    free(active);
    free(loads);
    free(store_bufs);
    free(ranges);
    return NULL;
  }
  barrier_src[0] = end->src[0];
  for (int i = 0; i < n_loads; i++)
    barrier_src[i + 1] = loads[i];
  PolyUOp *barrier =
      poly_uop(ctx, POLY_OP_BARRIER, POLY_VOID, barrier_src, n_loads + 1, poly_arg_none());
  end_src[0] = barrier;
  for (int i = 1; i < end->n_src; i++)
    end_src[i] = end->src[i];
  PolyUOp *ret = barrier ? poly_uop_tagged_arg(
                               ctx, POLY_OP_END, end->dtype, end_src, end->n_src, end->arg,
                               end->tag, end->tag_arg
                           )
                         : NULL;
  free(end_src);
  free(barrier_src);
  free(active);
  free(loads);
  free(store_bufs);
  free(ranges);
  return ret;
}

static _Thread_local PolyPatternMatcher *g_pm_implicit_barriers = NULL;
static PolyPatternMatcher *poly_pm_implicit_barriers(void) {
  if (g_pm_implicit_barriers) return g_pm_implicit_barriers;
  PolyRule rules[] = {
      {poly_upat_allow_any_len(poly_upat_op(POLY_OP_AFTER, NULL, 0, "after")),
       rule_add_raw_barrier},
      {poly_upat_allow_any_len(poly_upat_op(POLY_OP_END, NULL, 0, "end")), rule_add_war_barrier},
  };
  g_pm_implicit_barriers =
      poly_pm_thread_cache(poly_pm_new(rules, (int)(sizeof(rules) / sizeof(rules[0]))));
  return g_pm_implicit_barriers;
}

typedef struct {
  int64_t next;
} NumberParamsContext;

/* Current tinygrad/codegen/__init__.py:31-41. */
static PolyUOp *rule_number_param(PolyCtx *ctx, PolyUOp *param, const PolyBindings *b) {
  (void)b;
  NumberParamsContext *numbering = poly_graph_rewrite_userctx();
  if (!param || param->op != POLY_OP_PARAM || param->arg.kind != POLY_ARG_PARAM ||
      !param->arg.param || param->arg.param->slot != -1 || !numbering)
    return NULL;
  PolyParamArg arg = *param->arg.param;
  arg.slot = numbering->next++;
  return poly_uop_tagged_arg(
      ctx, param->op, param->dtype, param->src, param->n_src, poly_arg_param(&arg), param->tag,
      param->tag_arg
  );
}

static _Thread_local PolyPatternMatcher *g_pm_number_params = NULL;
static PolyPatternMatcher *poly_pm_number_params(void) {
  if (g_pm_number_params) return g_pm_number_params;
  PolyRule rules[] = {
      {poly_upat_allow_any_len(poly_upat_op(POLY_OP_PARAM, NULL, 0, "param")), rule_number_param},
  };
  g_pm_number_params =
      poly_pm_thread_cache(poly_pm_new(rules, (int)(sizeof(rules) / sizeof(rules[0]))));
  return g_pm_number_params;
}

static PolyUOp *poly_number_params(PolyCtx *ctx, PolyUOp *sink) {
  int n_topo = 0;
  PolyUOp **topo = poly_toposort_alloc(ctx, sink, &n_topo);
  if (!topo) return NULL;
  NumberParamsContext numbering = {0};
  for (int i = 0; i < n_topo; i++)
    if (topo[i]->op == POLY_OP_PARAM && topo[i]->arg.kind == POLY_ARG_PARAM && topo[i]->arg.param &&
        topo[i]->arg.param->slot != -1)
      numbering.next++;
  poly_toposort_free(topo);
  return poly_graph_walk_rewrite(ctx, sink, poly_pm_number_params(), NULL, &numbering, true);
}

/* Full rewrite-to-sink pipeline */

static bool device_is_gpu(int device) {
  return device == POLY_DEVICE_CUDA || device == POLY_DEVICE_HIP || device == POLY_DEVICE_WEBGPU;
}

PolyUOp *poly_full_rewrite_to_sink_ex(PolyCtx *ctx, PolyUOp *sink, PolyRewriteOpts opts) {
  /*
   * Unified pipeline matching tinygrad codegen/__init__.py full_rewrite_to_sink.
   * Backend-specific behavior is controlled by renderer config fields in opts,
   * not by if-device branches.
   *
   * New Phase 4 fields used here:
   *   opt_policy      — POLY_OPT_HEURISTIC (CPU) or POLY_OPT_TC_ONLY (GPU)
   *   device          — PolyDevice, gates gpudims/control_flow
   *   extra_matcher   — renderer-specific final rewrite patterns (NULL = none)
   */

  /* tinygrad@2026-08-22/a9069c177a9d codegen/__init__.py:292-296 verifies
   * spec_tensor before resolving multi-device and movement operations. */
  if (!ctx || !poly_type_verify_tensor(ctx, sink)) return NULL;
  poly_debug_stage_graph(ctx, "input", sink);

#define POLY_REWRITE_CHECK(stage_name)                                                             \
  do {                                                                                             \
    if (!sink) {                                                                                   \
      fprintf(stderr, "polygrad: full_rewrite_to_sink failed at %s\n", (stage_name));              \
      return NULL;                                                                                 \
    }                                                                                              \
  } while (0)

  POLY_REWRITE_CHECK("input");

  /* Current tinygrad codegen/__init__.py:295-299 resolves in-kernel shards,
   * then runs the shared movement matcher bottom-up. */
  sink = poly_apply_multi_pm(ctx, sink);
  POLY_REWRITE_CHECK("multi_pm");
  sink = poly_graph_rewrite_ex(ctx, sink, poly_pm_mops(), true);
  poly_debug_stage_graph(ctx, "preprocess", sink);
  POLY_REWRITE_CHECK("preprocess");

  /* 2. Optimization block (gated by optimize).
   * Matches tinygrad stage boundaries even where individual subpasses are still
   * incomplete on the polygrad side. */
  if (opts.optimize) {
    /* tinygrad: pm_load_collapse */
    sink = poly_graph_rewrite(ctx, sink, poly_pm_load_collapse());
    poly_debug_stage_graph(ctx, "load collapse", sink);
    POLY_REWRITE_CHECK("load collapse");

    /* tinygrad: pm_split_ranges + pm_flatten_range */
    PolyMap *split_range_ctx = poly_map_new(16);
    PolyPatternMatcher *split_ranges =
        poly_pm_concat(poly_pm_split_ranges(), poly_pm_flatten_range());
    if (!split_ranges) return NULL;
    sink = poly_graph_rewrite_ctx(ctx, sink, split_ranges, split_range_ctx);
    poly_pm_destroy(split_ranges);
    poly_map_destroy(split_range_ctx);
    poly_debug_stage_graph(ctx, "split ranges", sink);
    POLY_REWRITE_CHECK("split ranges");

    /* tinygrad: sym + pm_flatten_range */
    PolyPatternMatcher *initial_symbolic = poly_pm_concat(poly_sym(), poly_pm_flatten_range());
    if (!initial_symbolic) return NULL;
    sink = poly_graph_rewrite(ctx, sink, initial_symbolic);
    poly_pm_destroy(initial_symbolic);
    poly_debug_stage_graph(ctx, "initial symbolic", sink);
    POLY_REWRITE_CHECK("initial symbolic");

    /* tinygrad: pm_flatten_range + pm_simplify_ranges */
    PolyPatternMatcher *simplify_ranges =
        poly_pm_concat(poly_pm_flatten_range(), poly_pm_simplify_ranges());
    if (!simplify_ranges) return NULL;
    PolyMap *simplify_range_ctx = poly_map_new(16);
    sink = poly_graph_rewrite_ctx(ctx, sink, simplify_ranges, simplify_range_ctx);
    poly_map_destroy(simplify_range_ctx);
    poly_pm_destroy(simplify_ranges);
    poly_debug_stage_graph(ctx, "simplify ranges", sink);
    POLY_REWRITE_CHECK("simplify ranges");

    /* tinygrad apply_opts prelude: convert eligible WEAK output ranges to
     * GLOBAL before running heuristic/beam/tensor-core scheduling. */
    if (opts.caps.has_local) {
      sink = convert_loop_to_global(ctx, sink);
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
    } else if (!poly_get_noopt() && opts.opt_policy == POLY_OPT_TC_ONLY) {
      sink = poly_apply_tc_opt(ctx, sink, opts.caps);
    } else if (!poly_get_noopt()) {
      sink = poly_apply_opts_heuristic(ctx, sink, opts.caps);
    }
    sink = poly_graph_rewrite(ctx, sink, poly_pm_flatten_range());
    poly_debug_stage_graph(ctx, "apply opts", sink);
    POLY_REWRITE_CHECK("apply opts");
  }

  /* Current tinygrad codegen/__init__.py:319 composes this closure so a rule
   * minted by any member immediately re-enters every preceding member. */
  PolyPatternMatcher *postopt_a = poly_pm_concat(poly_sym(), poly_pm_move_where_on_load());
  PolyPatternMatcher *postopt_b =
      postopt_a ? poly_pm_concat(postopt_a, poly_pm_flatten_range()) : NULL;
  PolyPatternMatcher *postopt =
      postopt_b ? poly_pm_concat(postopt_b, poly_pm_reduce_unparented()) : NULL;
  poly_pm_destroy(postopt_a);
  poly_pm_destroy(postopt_b);
  if (!postopt) return NULL;
  sink = poly_graph_rewrite(ctx, sink, postopt);
  poly_pm_destroy(postopt);
  poly_debug_stage_graph(ctx, "postopt symbolic", sink);
  POLY_REWRITE_CHECK("postopt symbolic");

  /* Current expander2 owns RANGE/REDUCE/WMMA expansion. Grouped reduction is
   * encoded by apply_opts and consumed by pm_reduce_local; there is no extra
   * graph-wide group-for-reduce pass here. */
  sink = poly_apply_expander2(ctx, sink);
  poly_debug_stage_graph(ctx, "expander", sink);
  POLY_REWRITE_CHECK("expander");

  /* Current tinygrad: mop_cleanup+pm_reduce_local. */
  sink = poly_apply_pm_reduce(ctx, sink);
  poly_debug_stage_graph(ctx, "remove reduce", sink);
  POLY_REWRITE_CHECK("remove reduce");

  /* Current Tinygrad codegen/__init__.py:332. */
  int next_local_slot = 0;
  sink = poly_graph_rewrite_ctx(ctx, sink, poly_pm_add_local_buffers(), &next_local_slot);
  poly_debug_stage_graph(ctx, "add local buffers", sink);
  POLY_REWRITE_CHECK("add local buffers");

  /* Current Tinygrad codegen/__init__.py:330-331 uses one matcher for GPU
   * SPECIAL axes, CPU core_id, and DEVICE launch parameters. */
  if (opts.caps.has_threads || device_is_gpu(opts.device))
    sink = poly_add_gpudims_ex(ctx, sink, opts.caps);
  poly_debug_stage_graph(ctx, "add gpudims", sink);
  POLY_REWRITE_CHECK("add gpudims");

  /* 8. Broadcast and add loads.
   * Current tinygrad codegen/__init__.py runs symbolic_simple +
   * pm_expand_broadcast + pm_add_loads as one rewrite stage. */
  PolyPatternMatcher *symbolic_broadcast =
      poly_pm_concat(poly_symbolic_simple(), poly_pm_expand_broadcast());
  PolyPatternMatcher *expand_and_load =
      symbolic_broadcast ? poly_pm_concat(symbolic_broadcast, poly_pm_add_loads()) : NULL;
  poly_pm_destroy(symbolic_broadcast);
  if (!expand_and_load) return NULL;
  sink = poly_graph_rewrite(ctx, sink, expand_and_load);
  poly_pm_destroy(expand_and_load);
  poly_debug_stage_graph(ctx, "add loads", sink);
  POLY_REWRITE_CHECK("add loads");

  /* 9. Current shape devectorization.  Do not run the June dtype-vector
   * folding matcher here: current Tinygrad keeps scalar lanes until the
   * following global memory_coalescing pass. */
  sink = poly_graph_rewrite(ctx, sink, poly_devectorizer2_stage());
  poly_debug_stage_graph(ctx, "devectorize2", sink);
  POLY_REWRITE_CHECK("devectorize2");

  /* Current tinygrad codegen/__init__.py:340-346 simplifies scalar address
   * lanes, globally coalesces adjacent LOAD/STORE occurrences, then performs
   * the remaining elementwise/image devectorization. */
  sink = poly_graph_rewrite(ctx, sink, poly_sym());
  poly_debug_stage_graph(ctx, "early symbolic", sink);
  POLY_REWRITE_CHECK("early symbolic");
  sink = poly_memory_coalescing(ctx, sink, opts.caps);
  poly_debug_stage_graph(ctx, "memory coalescing", sink);
  POLY_REWRITE_CHECK("memory coalescing");

  PolyPatternMatcher *symbolic_ew = poly_pm_concat(poly_symbolic_simple(), poly_ew_devectorizer());
  PolyPatternMatcher *add_images =
      symbolic_ew ? poly_pm_concat(symbolic_ew, poly_pm_simplify_add_image()) : NULL;
  poly_pm_destroy(symbolic_ew);
  if (!add_images) return NULL;
  /* Tinygrad 2026-08-22/a9069c177a9d codegen/__init__.py:344-346 passes
   * `ctx=({}, ren)` through this combined bottom-up fixed point. */
  PolyImageRewriteCtx image_ctx = {.caps = opts.caps};
  sink = poly_graph_rewrite_ctx_ex(ctx, sink, add_images, &image_ctx, true);
  poly_image_rewrite_ctx_destroy(&image_ctx);
  poly_pm_destroy(add_images);
  poly_debug_stage_graph(ctx, "add images", sink);
  POLY_REWRITE_CHECK("remaining elementwise devectorization");

  /* Current tinygrad codegen/__init__.py:349-359 keeps indexing_simplify in
   * both the weak-index symbolic cleanup and the lowering fixed point. */
  PolyPatternMatcher *symbolic_index = poly_pm_concat(poly_sym(), poly_indexing_simplify());
  if (!symbolic_index) return NULL;
  sink = poly_graph_rewrite(ctx, sink, symbolic_index);
  poly_pm_destroy(symbolic_index);
  poly_debug_stage_graph(ctx, "extra symbolic", sink);
  POLY_REWRITE_CHECK("extra symbolic");

  PolyPatternMatcher *symbolic_lower =
      poly_pm_concat(poly_symbolic_simple(), poly_pm_lower_index_dtype());
  PolyPatternMatcher *lower_index =
      symbolic_lower ? poly_pm_concat(symbolic_lower, poly_indexing_simplify()) : NULL;
  poly_pm_destroy(symbolic_lower);
  if (!lower_index) return NULL;
  /* Tinygrad 2026-08-22/a9069c177a9d codegen/__init__.py:354 supplies
   * one ctx dict for weak-source lowering across this complete rewrite. */
  PolyMap *lower_cache = poly_map_new(256);
  if (!lower_cache) {
    poly_pm_destroy(lower_index);
    return NULL;
  }
  sink = poly_graph_rewrite_ctx(ctx, sink, lower_index, lower_cache);
  poly_map_destroy(lower_cache);
  poly_pm_destroy(lower_index);
  poly_debug_stage_graph(ctx, "lower all index dtypes", sink);
  POLY_REWRITE_CHECK("lower all index dtypes");

  sink = poly_graph_rewrite(ctx, sink, poly_symbolic());
  poly_debug_stage_graph(ctx, "final symbolic", sink);
  POLY_REWRITE_CHECK("final symbolic");

  /* Current tinygrad codegen/__init__.py:365 runs pm_cast_float_alu here,
   * immediately before early decompositions.  This is final-program operand
   * legality, not Tensor-level dtype promotion. */
  sink = poly_graph_rewrite(ctx, sink, poly_pm_cast_float_alu());
  poly_debug_stage_graph(ctx, "cast float alu operands", sink);
  POLY_REWRITE_CHECK("cast float alu operands");

  /* Current tinygrad codegen/__init__.py:364-366 composes symbolic_simple
   * with only the simplifying decomposition rules at this boundary. */
  PolyPatternMatcher *early_decomp =
      poly_pm_concat(poly_symbolic_simple(), poly_get_simplifying_rewrite_patterns(opts.caps));
  if (!early_decomp) return NULL;
  sink = poly_graph_rewrite(ctx, sink, early_decomp);
  poly_pm_destroy(early_decomp);
  poly_debug_stage_graph(ctx, "early decompositions", sink);
  POLY_REWRITE_CHECK("early decompositions");
  /* Tinygrad 2026-08-22/a9069c177a9d codegen/__init__.py:369 and
   * codegen/decomp/dtype.py:206-212 discover unsupported dtypes from the
   * graph, then lower them from lowest priority to highest. */
  PolyPatternMatcher *dtype_and_weak =
      poly_pm_concat(poly_pm_dtype_decomps(), poly_pm_commit_weak());
  if (!dtype_and_weak) return NULL;
  PolyDTypeDecompsContext dtype_ctx = {.caps = opts.caps};
  sink = poly_graph_rewrite_ctx(ctx, sink, dtype_and_weak, &dtype_ctx);
  poly_pm_destroy(dtype_and_weak);
  poly_debug_stage_graph(ctx, "decomp dtypes", sink);
  POLY_REWRITE_CHECK("decomp dtypes");

  /* Current tinygrad codegen/__init__.py:370-374 retains the early matcher,
   * then appends late op and transcendental rules for one rewrite closure. */
  PolyPatternMatcher *late_decomp =
      poly_pm_concat(poly_symbolic_simple(), poly_get_simplifying_rewrite_patterns(opts.caps));
  PolyPatternMatcher *with_late =
      late_decomp ? poly_pm_concat(late_decomp, poly_get_late_rewrite_patterns(opts.caps)) : NULL;
  poly_pm_destroy(late_decomp);
  late_decomp =
      with_late ? poly_pm_concat(with_late, poly_get_transcendental_patterns(opts.caps)) : NULL;
  poly_pm_destroy(with_late);
  if (!late_decomp) return NULL;
  sink = poly_graph_rewrite_ctx(ctx, sink, late_decomp, &opts.caps);
  poly_debug_stage_graph(ctx, "late decompositions", sink);
  POLY_REWRITE_CHECK("late decompositions");

  /* 12. Final rewrite. Current shaped STACK/INDEX/SHRINK is renderer IR. */
  sink = poly_graph_rewrite(ctx, sink, poly_pm_move_gates_from_index());
  poly_debug_stage_graph(ctx, "move gates from index", sink);
  /* Current tinygrad codegen/__init__.py:378-379 composes the complete final
   * closure, including Invalid removal. */
  PolyPatternMatcher *weak_final = poly_pm_concat(poly_pm_commit_weak(), poly_pm_cast_weak());
  PolyPatternMatcher *final_matcher = weak_final ? poly_pm_concat(weak_final, late_decomp) : NULL;
  poly_pm_destroy(weak_final);
  if (opts.extra_matcher && final_matcher) {
    PolyPatternMatcher *with_extra = poly_pm_concat(final_matcher, opts.extra_matcher);
    poly_pm_destroy(final_matcher);
    final_matcher = with_extra;
  }
  if (final_matcher) {
    PolyPatternMatcher *with_split = poly_pm_concat(final_matcher, poly_pm_split_ends());
    poly_pm_destroy(final_matcher);
    final_matcher = with_split;
  }
  if (final_matcher) {
    PolyPatternMatcher *with_invalid = poly_pm_concat(final_matcher, poly_pm_remove_invalid());
    poly_pm_destroy(final_matcher);
    final_matcher = with_invalid;
  }
  if (!final_matcher) return NULL;
  sink = poly_graph_rewrite(ctx, sink, final_matcher);
  poly_pm_destroy(final_matcher);
  poly_pm_destroy(late_decomp);
  poly_debug_stage_graph(ctx, "final rewrite", sink);
  POLY_REWRITE_CHECK("final rewrite");

  /* Current tinygrad/codegen/__init__.py:381-393: freeze literal spelling,
   * materialize implicit local-memory ordering, inject control flow, and only
   * then allocate slots for unnamed PARAMs. */
  sink = poly_graph_walk_rewrite(ctx, sink, poly_pm_casted_consts(), NULL, NULL, true);
  poly_debug_stage_graph(ctx, "casted consts", sink);
  POLY_REWRITE_CHECK("casted consts");

  sink = poly_graph_rewrite(ctx, sink, poly_pm_implicit_barriers());
  poly_debug_stage_graph(ctx, "add implicit barriers", sink);
  POLY_REWRITE_CHECK("add implicit barriers");

  /* 13. Control flow */
  sink = poly_apply_control_flow(ctx, sink);
  poly_debug_stage_graph(ctx, "control flow", sink);
  POLY_REWRITE_CHECK("control flow");

  sink = poly_number_params(ctx, sink);
  poly_debug_stage_graph(ctx, "number params with -1", sink);
  POLY_REWRITE_CHECK("number params with -1");
  /* Current Tinygrad codegen/__init__.py:389-394 validates the numbered final
   * SINK against spec_program before it can enter linearization. */
  if (!poly_type_verify_program(ctx, sink)) {
    if (poly_getenv_int("POLY_TRACE_CODEGEN_FAILURE", 0)) {
      fprintf(
          stderr, "polygrad: codegen kernel %d failed program verification\n",
          poly_trace_codegen_kernel
      );
      int n_topo = 0;
      PolyUOp **topo = poly_toposort_alloc(ctx, sink, &n_topo);
      for (int i = 0; topo && i < n_topo; i++) {
        if (topo[i]->op != POLY_OP_CONST || topo[i]->arg.kind != POLY_ARG_INVALID) continue;
        fprintf(stderr, "polygrad: invalid program node %p", (void *)topo[i]);
        for (int j = 0; j < n_topo; j++) {
          for (int k = 0; k < topo[j]->n_src; k++) {
            if (topo[j]->src[k] == topo[i]) {
              fprintf(stderr, " parent=%s[%d]", poly_op_name(topo[j]->op), k);
              for (int p = 0; p < n_topo; p++)
                for (int q = 0; q < topo[p]->n_src; q++)
                  if (topo[p]->src[q] == topo[j])
                    fprintf(stderr, " grandparent=%s[%d]", poly_op_name(topo[p]->op), q);
            }
          }
        }
        fputc('\n', stderr);
      }
      poly_toposort_free(topo);
    }
    return NULL;
  }
  poly_debug_stage_graph(ctx, "rewritten", sink);
  POLY_REWRITE_CHECK("rewritten");

#undef POLY_REWRITE_CHECK
  return sink;
}

PolyUOp *poly_full_rewrite_to_sink(PolyCtx *ctx, PolyUOp *sink) {
  PolyRewriteOpts opts = {
      .optimize = poly_kernel_optimize_enabled(sink),
      .caps = poly_c_renderer_caps(),
      .device = POLY_DEVICE_CPU,
      .opt_policy = POLY_OPT_HEURISTIC,
      .extra_matcher = poly_clang_renderer_extra_matcher(),
  };
  return poly_full_rewrite_to_sink_ex(ctx, sink, opts);
}

static PolyUOp *line_rebuild(PolyCtx *ctx, PolyUOp *u, PolyUOp **src, int n_src) {
  return (u->tag != 0 || u->tag_arg.kind != POLY_ARG_NONE)
             ? poly_uop_tagged_arg(ctx, u->op, u->dtype, src, n_src, u->arg, u->tag, u->tag_arg)
             : poly_uop(ctx, u->op, u->dtype, src, n_src, u->arg);
}

static PolyUOp *line_replacement(PolyUOp *u, PolyUOp **old_uops, PolyUOp **new_uops, int n) {
  for (int i = n - 1; i >= 0; i--)
    if (old_uops[i] == u) return new_uops[i];
  return u;
}

static bool line_is_gated_store(PolyUOp *u) {
  if (!u || u->op != POLY_OP_STORE || u->n_src != 3 || !poly_dtype_eq(u->src[2]->dtype, POLY_BOOL))
    return false;
  PolyUOp *address = u->src[0];
  if (address && address->op == POLY_OP_CAST && address->n_src == 1) address = address->src[0];
  return address && (address->op == POLY_OP_INDEX || address->op == POLY_OP_SHRINK);
}

/* Tinygrad 2026-08-22/a9069c177a9d codegen/__init__.py:397-419
 * line-rewrites gated STORE into IF, STORE, ENDIF and redirects later uses. */
static PolyUOp **line_rewrite_cleanups(PolyCtx *ctx, PolyUOp **linear, int n, int *n_out) {
  int n_gated = 0;
  for (int i = 0; i < n; i++)
    if (line_is_gated_store(linear[i])) n_gated++;
  if (n_gated == 0) {
    if (n_out) *n_out = n;
    return linear;
  }

  PolyUOp **out = malloc((size_t)(n + 2 * n_gated) * sizeof(*out));
  PolyUOp **old_uops = malloc((size_t)n * sizeof(*old_uops));
  PolyUOp **new_uops = malloc((size_t)n * sizeof(*new_uops));
  if (!out || !old_uops || !new_uops) {
    free(out);
    free(old_uops);
    free(new_uops);
    return NULL;
  }

  int n_seen = 0, n_linear = 0;
  for (int i = 0; i < n; i++) {
    PolyUOp *u = linear[i];
    PolyUOp *stack_src[64];
    PolyUOp **src = u->n_src <= 64 ? stack_src : malloc((size_t)u->n_src * sizeof(*src));
    if (!src) {
      free(out);
      free(old_uops);
      free(new_uops);
      return NULL;
    }
    bool changed = false;
    for (int j = 0; j < u->n_src; j++) {
      src[j] = line_replacement(u->src[j], old_uops, new_uops, n_seen);
      if (src[j] != u->src[j]) changed = true;
    }
    PolyUOp *rewritten = changed ? line_rebuild(ctx, u, src, u->n_src) : u;
    if (src != stack_src) free(src);

    old_uops[n_seen] = u;
    if (line_is_gated_store(rewritten)) {
      PolyUOp *store_src[2] = {rewritten->src[0], rewritten->src[1]};
      PolyUOp *store = line_rebuild(ctx, rewritten, store_src, 2);
      PolyUOp *if_src[2] = {rewritten->src[2], rewritten->src[0]};
      PolyUOp *ifu = poly_uop(ctx, POLY_OP_IF, POLY_VOID, if_src, 2, poly_arg_none());
      PolyUOp *endif = poly_uop1(ctx, POLY_OP_ENDIF, POLY_VOID, ifu, poly_arg_none());
      new_uops[n_seen++] = store;
      out[n_linear++] = ifu;
      out[n_linear++] = store;
      out[n_linear++] = endif;
    } else {
      new_uops[n_seen++] = rewritten;
      out[n_linear++] = rewritten;
    }
  }

  free(old_uops);
  free(new_uops);
  free(linear);
  if (n_out) *n_out = n_linear;
  return out;
}

PolyUOp **poly_do_linearize(PolyCtx *ctx, PolyUOp *sink, int *n_out) {
  int n = 0;
  PolyUOp **linear = poly_linearize(ctx, sink, &n);
  if (!linear) {
    if (n_out) *n_out = 0;
    return NULL;
  }
  PolyUOp **cleaned = line_rewrite_cleanups(ctx, linear, n, n_out);
  if (!cleaned) free(linear);
  return cleaned;
}
