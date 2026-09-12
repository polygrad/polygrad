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
#include "ir.h"
#include "engine/schedule.h"
#include "schedule/indexing.h"
#include "schedule/multi.h"
#include "schedule/rangeify.h"
#include "codegen/simplify.h"
#include "uop/movement.h"
#include "uop/ops.h"
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
#include <unistd.h>
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

/* helpers.BEAM: explicit frontend scopes override the environment. Native
 * contexts share this policy; each Wasm module has independent C storage. */
static int beam_value;
static bool beam_initialized;
int poly_get_beam(void) {
  return beam_initialized ? beam_value : poly_getenv_int("BEAM", 0);
}
void poly_set_beam(int value) {
  beam_value = value;
  beam_initialized = true;
}

/* helpers.IGNORE_BEAM_CACHE shares BEAM's frontend-independent scope. */
static int ignore_beam_cache_value;
static bool ignore_beam_cache_initialized;
int poly_get_ignore_beam_cache(void) {
  return ignore_beam_cache_initialized ? ignore_beam_cache_value
                                       : poly_getenv_int("IGNORE_BEAM_CACHE", 0);
}
void poly_set_ignore_beam_cache(int value) {
  ignore_beam_cache_value = value;
  ignore_beam_cache_initialized = true;
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
typedef struct {
  PolyCtx *ctx;
  PolyUOp *ast; /* current kernel SINK */
  int64_t opt_range_next; /* counter for new axis IDs */

  /* Sorted RANGE list (by axis_to_pos then axis_id) */
  PolyUOp **rngs;
  int n_rngs;

  /* Shape (bound of each range) */
  int64_t *shape;

  /* Axis types */
  PolyAxisType *types;

  /* INDEX ops (tinygrad k.bufs) - reversed toposort order */
  PolyUOp **bufs;
  int n_bufs;

  /* Immutable metadata shared by Scheduler.copy; refresh publishes a new
   * snapshot. This owns C scratch, not UOps or residency. */
  size_t *references;
  size_t reach_words;
  uint64_t *buf_reach;
  bool has_reach;

  /* Has reduce op */
  bool has_reduce;

  /* Failed metadata construction must never expose a partial scheduler view. */
  bool failed;
  const PolyTensorCore *tensor_core;
} OptScheduler;

static void sched_destroy(OptScheduler *s) {
  if (s->references && --*s->references == 0) {
    free(s->rngs);
    free(s->shape);
    free(s->types);
    free(s->bufs);
    free(s->buf_reach);
    free(s->references);
  }
  s->references = NULL;
}

static void sched_copy(OptScheduler *dst, const OptScheduler *src) {
  *dst = *src;
  if (dst->references) ++*dst->references;
}

#ifdef POLY_TESTING
static _Thread_local int sched_alloc_fail_after = -1;
#endif

static void *sched_calloc(size_t count, size_t size) {
#ifdef POLY_TESTING
  if (sched_alloc_fail_after == 0) return NULL;
  if (sched_alloc_fail_after > 0) --sched_alloc_fail_after;
#endif
  return calloc(count ? count : 1, size);
}

static bool sched_buf_reaches(const OptScheduler *s, int bi, int ri) {
  return s->has_reach &&
         (s->buf_reach[(size_t)bi * s->reach_words + (size_t)ri / 64] & (UINT64_C(1) << (ri % 64)));
}

static bool sched_can_optimize(const OptScheduler *s) {
  return s && !s->failed && s->n_rngs > 0;
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

/* Scheduler._globalizable_rngs: only immediate SINK END operands define
 * outputs. A nested END closes a loop inside an output, not a launch axis. */
static bool sched_is_globalizable(
    PolyCtx *ctx,
    PolyUOp *ast,
    PolyUOp *rng,
    PolyUOp **topo,
    int n_topo
) {
  if (poly_range_axis_type(rng->arg) != POLY_AXIS_WEAK) return false;
  bool output = false;
  for (int i = 0; i < ast->n_src; i++) {
    PolyUOp *end = ast->src[i];
    if (end->op != POLY_OP_END) continue;
    for (int j = 1; j < end->n_src; j++)
      output |= poly_uop_in_ranges(ctx, end->src[j], rng);
  }
  if (!output) return false;
  for (int i = 0; i < n_topo; i++)
    if (topo[i]->op == POLY_OP_STAGE && !poly_uop_in_ranges(ctx, topo[i], rng)) return false;
  return true;
}

/* tinygrad apply_opts starts by converting eligible LOOP output ranges into
 * GLOBAL ranges (postrange.py:340). WebGPU/CUDA scheduling depends on that
 * boundary before the later upcast heuristics run. */
static PolyUOp *convert_loop_to_global(PolyCtx *ctx, PolyUOp *ast) {
  if (!ctx || !ast) return ast;

  int n_topo = 0;
  PolyUOp **topo = poly_toposort_alloc(ctx, ast, &n_topo);
  if (!topo) return NULL;
  PolyUOp **from = calloc((size_t)n_topo, sizeof(*from));
  PolyUOp **to = calloc((size_t)n_topo, sizeof(*to));
  if (!from || !to) {
    free(from);
    free(to);
    poly_toposort_free(topo);
    return NULL;
  }
  int n_sub = 0;
  for (int i = 0; i < n_topo; i++) {
    PolyUOp *r = topo[i];
    if (r->op != POLY_OP_RANGE || !poly_arg_is_range(r->arg) || poly_dtype_eq(r->dtype, POLY_VOID))
      continue;
    int64_t lo, hi;
    poly_uop_minmax(ctx, r, &lo, &hi);
    if (hi <= 0 || !sched_is_globalizable(ctx, ast, r, topo, n_topo)) continue;
    PolyArg g_arg = poly_arg_range_ex(
        poly_range_axis_id(r->arg), POLY_AXIS_GLOBAL, poly_range_extra(r->arg),
        poly_range_n_extra(r->arg)
    );
    PolyUOp *g_rng =
        poly_uop_tagged_arg(ctx, r->op, r->dtype, r->src, r->n_src, g_arg, r->tag, r->tag_arg);
    from[n_sub] = r;
    to[n_sub] = g_rng;
    n_sub++;
  }

  PolyUOp *out = (n_sub > 0) ? poly_uop_substitute(ctx, ast, from, to, n_sub) : ast;
  free(from);
  free(to);
  poly_toposort_free(topo);
  return out;
}

/* Build reachability bitmask for all nodes in a toposort.
 * Uses a single forward pass: each node's bitmask = union of its sources' bitmasks.
 * RANGE nodes set their own bit. Result: reachable[i] has bit j set iff rngs[j]
 * is reachable from topo[i]'s source tree.
 *
 * Each row contains ceildiv(n_rngs,64) words. Caller owns the result. */
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
  size_t words = ((size_t)n_rngs + 63) / 64;
  uint64_t *reach = NULL;
  if (!idx_map || !indices || !words || (size_t)n_topo > SIZE_MAX / sizeof(*reach) / words)
    goto done;
  reach = calloc((size_t)n_topo * words, sizeof(*reach));
  if (!reach) goto done;
  for (int i = 0; i < n_topo; i++) {
    indices[i] = i + 1; /* 1-based so NULL means "not in map" */
    poly_map_set(idx_map, poly_ptr_hash(topo[i]), topo[i], &indices[i], poly_ptr_eq);
  }

  /* Set bits for RANGE nodes */
  for (int i = 0; i < n_topo; i++) {
    if (topo[i]->op == POLY_OP_RANGE) {
      for (int ri = 0; ri < n_rngs; ri++) {
        if (rngs[ri] == topo[i])
          reach[(size_t)i * words + (size_t)ri / 64] |= UINT64_C(1) << (ri % 64);
      }
    }
  }

  /* Forward pass: propagate bits from sources */
  for (int i = 0; i < n_topo; i++) {
    for (int j = 0; j < topo[i]->n_src; j++) {
      int *pidx = (int *)poly_map_get(
          idx_map, poly_ptr_hash(topo[i]->src[j]), topo[i]->src[j], poly_ptr_eq
      );
      if (pidx)
        for (size_t w = 0; w < words; w++)
          reach[(size_t)i * words + w] |= reach[(size_t)(*pidx - 1) * words + w];
    }
  }

done:
  poly_map_destroy(idx_map);
  free(indices);
  return reach;
}

static void projected_node_reachability(
    PolyUOp *u,
    PolyMap *idx_map,
    uint64_t *reach,
    size_t words,
    uint64_t *out
) {
  if (!u) return;
  int *pidx = (int *)poly_map_get(idx_map, poly_ptr_hash(u), u, poly_ptr_eq);
  if (pidx) {
    for (size_t w = 0; w < words; w++)
      out[w] |= reach[(size_t)*pidx * words + w];
    return;
  }
  if (u->op != POLY_OP_STACK) return;
  for (int i = 0; i < u->n_src; i++)
    projected_node_reachability(u->src[i], idx_map, reach, words, out);
}

static void projected_index_reachability(
    PolyCtx *ctx,
    PolyUOp *coord,
    PolyMap *idx_map,
    uint64_t *reach,
    size_t words,
    uint64_t *out
) {
  PolyUOp *idx = poly_uop_get_idx(ctx, coord);
  if (!idx) return;
  /* Pinned heuristic.py:118-128,160-175 asks membership in
   * `get_idx().backward_slice`; UOp.backward_slice explicitly excludes the
   * projected coordinate root (uop/ops.py:177-183). Union the source closures
   * rather than returning the root's own RANGE bit. This matters for direct
   * coordinates such as INDEX(mean, r_channel). */
  for (int i = 0; i < idx->n_src; i++)
    projected_node_reachability(idx->src[i], idx_map, reach, words, out);
}

static void sched_refresh(OptScheduler *s);

static void sched_init(OptScheduler *s, PolyCtx *ctx, PolyUOp *sink) {
  *s = (OptScheduler){.ctx = ctx, .ast = sink, .opt_range_next = 1};
  sched_refresh(s);
}

/* Refresh rngs, shapes, types after a shift_to modifies the AST */
static void sched_refresh(OptScheduler *owner) {
  OptScheduler next = {
      .ctx = owner->ctx,
      .ast = owner->ast,
      .opt_range_next = owner->opt_range_next,
      .tensor_core = owner->tensor_core};
  OptScheduler *s = &next;
  int n_topo = 0;
  PolyUOp **topo = poly_toposort_alloc(s->ctx, s->ast, &n_topo);
  if (!topo) goto failed;
  /* Size metadata from the graph; toposort already deduplicates identities. */
  s->references = sched_calloc(1, sizeof(*s->references));
  if (!s->references) goto failed;
  *s->references = 1;
  int range_capacity = 0, buffer_capacity = 0;
  for (int i = 0; i < n_topo; i++) {
    range_capacity += topo[i]->op == POLY_OP_RANGE;
    buffer_capacity += topo[i]->op == POLY_OP_INDEX;
  }
  s->rngs = sched_calloc((size_t)range_capacity, sizeof(*s->rngs));
  s->shape = sched_calloc((size_t)range_capacity, sizeof(*s->shape));
  s->types = sched_calloc((size_t)range_capacity, sizeof(*s->types));
  s->bufs = sched_calloc((size_t)buffer_capacity, sizeof(*s->bufs));
  if (!s->rngs || !s->shape || !s->types || !s->bufs) goto failed;
  int64_t max_id = -1;

  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    if (u->op == POLY_OP_REDUCE) s->has_reduce = true;
    if (u->op == POLY_OP_RANGE && poly_arg_is_range(u->arg) &&
        !poly_dtype_eq(u->dtype, POLY_VOID) && poly_range_axis_type(u->arg) != POLY_AXIS_DEVICE) {
      int64_t lo, hi;
      poly_uop_minmax(s->ctx, u, &lo, &hi);
      if (hi <= 0) continue;
      s->rngs[s->n_rngs++] = u;
      int64_t aid = poly_range_axis_id(u->arg);
      if (aid > max_id) max_id = aid;
    }
    if (u->op == POLY_OP_INDEX) {
      s->bufs[s->n_bufs++] = u;
    }
  }
  if (max_id == INT64_MAX) goto failed;
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
    PolyUOp *bound =
        r->n_src ? poly_graph_rewrite(s->ctx, r->src[0], poly_symbolic_simple()) : NULL;
    /* full_shape may be symbolic; zero is only the static-size sentinel, not
     * an execution bound. Static-only heuristics must not specialize it. */
    s->shape[i] =
        bound && bound->op == POLY_OP_CONST && bound->arg.kind == POLY_ARG_INT ? bound->arg.i : 0;
  }

  /* Rebuild reachability bitmask */
  s->reach_words = ((size_t)s->n_rngs + 63) / 64;
  if (s->n_bufs > 0 && s->n_rngs > 0) {
    if ((size_t)s->n_bufs > SIZE_MAX / sizeof(*s->buf_reach) / s->reach_words) goto failed;
    s->buf_reach = sched_calloc((size_t)s->n_bufs * s->reach_words, sizeof(*s->buf_reach));
    uint64_t *reach = build_reachability_bitmask(topo, n_topo, s->rngs, s->n_rngs);
    PolyMap *idx_map = poly_map_new((size_t)(n_topo < 64 ? 64 : (size_t)n_topo * 2));
    int *indices = (int *)malloc((size_t)n_topo * sizeof(int));
    if (!s->buf_reach || !reach || !idx_map || !indices) {
      free(reach);
      free(indices);
      poly_map_destroy(idx_map);
      goto failed;
    }
    for (int i = 0; i < n_topo; i++) {
      indices[i] = i;
      poly_map_set(idx_map, poly_ptr_hash(topo[i]), topo[i], &indices[i], poly_ptr_eq);
    }
    for (int bi = 0; bi < s->n_bufs; bi++) {
      if (s->bufs[bi]->n_src >= 2)
        projected_index_reachability(
            s->ctx, s->bufs[bi]->src[1], idx_map, reach, s->reach_words,
            s->buf_reach + (size_t)bi * s->reach_words
        );
    }
    s->has_reach = true;
    poly_map_destroy(idx_map);
    free(indices);
    free(reach);
  }
  poly_toposort_free(topo);
  sched_destroy(owner);
  *owner = next;
  return;
failed:
  poly_toposort_free(topo);
  sched_destroy(&next);
  owner->failed = true;
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
  if (!s || s->failed) return NULL;
  if (!input_new_rng && s->opt_range_next == INT64_MAX) return NULL;

  if (amount <= 0 || rng->n_src != 1) return NULL;
  PolyUOp *old_sz = poly_uop_divides(s->ctx, rng->src[0], amount);
  if (!old_sz) return NULL;

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
  PolyUOp *replaced = poly_uop1(ctx, POLY_OP_RANGE, dt, old_sz, rng->arg);

  /* Compute substitution expression */
  PolyUOp *sub_axis;
  if (top) {
    sub_axis = poly_uop2(
        ctx, POLY_OP_ADD, dt, poly_uop2(ctx, POLY_OP_MUL, dt, new_rng, old_sz, poly_arg_none()),
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
  OptScheduler old;
  sched_copy(&old, s);
  s->ast = poly_uop_substitute(ctx, s->ast, from, to, 1);
  sched_refresh(s);
  if (s->failed) {
    sched_destroy(s);
    *s = old;
    return NULL;
  }
  sched_destroy(&old);
  if (out_new_rng) *out_new_rng = new_rng;
  return replaced;
}

static bool sched_apply_opt(OptScheduler *s, PolyRendererCaps caps, PolyOpt opt, PolyUOp **result);

/* Integer Opt construction and tuple result unpacking for the C heuristic.
 * All legality, history and transactional publication stay in apply_opt. */
static PolyUOp *sched_apply_int_opt(
    OptScheduler *s,
    PolyRendererCaps caps,
    PolyOptOps op,
    int axis,
    int64_t amount
) {
  PolyUOp *result[3] = {0};
  PolyOpt opt = {
      .op = op, .has_axis = true, .axis = axis, .arg_kind = POLY_OPT_ARG_INT, .arg = amount};
  return sched_apply_opt(s, caps, opt, result) ? result[0] : NULL;
}

/* Outside heuristic.py's explicit KernelOptError catches, a selected option
 * must succeed. A failed mandatory choice cannot publish a partial policy. */
static PolyUOp *sched_require_int_opt(
    OptScheduler *s,
    PolyRendererCaps caps,
    PolyOptOps op,
    int axis,
    int64_t amount
) {
  PolyUOp *result = sched_apply_int_opt(s, caps, op, axis, amount);
  if (!result) s->failed = true;
  return result;
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

/* UOp.__bool__/_eval requires a proven truth value, unlike resolve(default).
 * Indeterminate predicates reject the candidate instead of specializing bounds. */
static int sched_eval_lt(PolyCtx *ctx, PolyUOp *value, int64_t limit) {
  PolyUOp *pred = poly_alu2(
      ctx, POLY_OP_CMPLT, value, poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(limit))
  );
  pred = poly_graph_rewrite(ctx, pred, poly_symbolic());
  if (!pred) return -1;
  int64_t lo, hi;
  poly_uop_minmax(ctx, pred, &lo, &hi);
  return lo == hi ? lo != 0 : -1;
}

/* Scheduler.upcast_size plus the caller's comparison. Keep constant tiles on
 * the integer fast path; symbolic extents are values, not the shape-cache0. */
static int sched_upcast_size_lt(const OptScheduler *s, int64_t limit) {
  int64_t prod = 1;
  bool symbolic = false;
  for (int i = 0; i < s->n_rngs; i++) {
    if (s->types[i] == POLY_AXIS_UPCAST || s->types[i] == POLY_AXIS_UNROLL) {
      if (s->shape[i] <= 0)
        symbolic = true;
      else if (prod > INT64_MAX / s->shape[i])
        prod = INT64_MAX;
      else
        prod *= s->shape[i];
    }
  }
  if (!symbolic) return prod < limit;
  PolyUOp *value = poly_uop0(s->ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(1));
  for (int i = 0; i < s->n_rngs; i++)
    if (s->types[i] == POLY_AXIS_UPCAST || s->types[i] == POLY_AXIS_UNROLL)
      value = poly_alu2(
          s->ctx, POLY_OP_MUL, value, poly_cast(s->ctx, s->rngs[i]->src[0], POLY_WEAKINT)
      );
  return sched_eval_lt(s->ctx, value, limit);
}

static PolyUOp *sched_full_shape_prod(const OptScheduler *s) {
  /* heuristic.py resolves the complete symbolic product. Unknown dimensions
   * must not become factors of one; weak arithmetic avoids host overflow. */
  PolyUOp *prod = poly_uop0(s->ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(1));
  for (int i = 0; i < s->n_rngs; i++) {
    prod =
        poly_alu2(s->ctx, POLY_OP_MUL, prod, poly_cast(s->ctx, s->rngs[i]->src[0], POLY_WEAKINT));
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
  int64_t prod = 1;
  for (int i = 0; i < s->n_rngs; i++) {
    PolyAxisType t = s->types[i];
    if ((t != POLY_AXIS_GLOBAL && t != POLY_AXIS_LOCAL && t != POLY_AXIS_WEAK) || s->shape[i] <= 1)
      continue;
    if (prod > INT64_MAX / s->shape[i]) return INT64_MAX;
    prod *= s->shape[i];
  }
  return prod;
}

/* heuristic.py sum_strides is a Python integer even when index terms are
 * large and cancel. Use the shared integer owner, not int64 or float scores. */
static bool sched_stride_score(const OptScheduler *s, int axis, int *num_strides, PolyInt *sum) {
  PolyUOp *rng = s->rngs[axis];
  *num_strides = 0;
  for (int bi = 0; bi < s->n_bufs; bi++) {
    if (s->bufs[bi]->n_src < 2) continue;
    PolyUOp *idx = poly_uop_get_idx(s->ctx, s->bufs[bi]->src[1]);
    if (!idx) return false;
    if (sched_buf_reaches(s, bi, axis)) ++*num_strides;
    int n_add = 0;
    PolyUOp **addends = poly_uop_split(idx, POLY_OP_ADD, &n_add);
    if (!addends) return false;
    for (int j = 0; j < n_add; j++) {
      PolyUOp *c = addends[j], *coefficient = NULL;
      if (c->op == POLY_OP_MUL && c->n_src == 2) {
        if (c->src[0] == rng && c->src[1]->op == POLY_OP_CONST)
          coefficient = c->src[1];
        else if (c->src[1] == rng && c->src[0]->op == POLY_OP_CONST)
          coefficient = c->src[0];
      }
      if (c != rng && !coefficient) continue;
      PolyInt term = {0}, next = {0};
      bool ok =
          coefficient ? poly_int_from_arg(&term, coefficient->arg) : poly_int_from_i64(&term, 1);
      ok = ok && poly_int_add(&next, sum, &term);
      poly_int_free(&term);
      if (!ok) {
        poly_int_free(&next);
        free(addends);
        return false;
      }
      poly_int_free(sum);
      *sum = next;
    }
    free(addends);
  }
  return true;
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
static int sched_shape_str_to_axis(const OptScheduler *s, const char *name) {
  int cnt[16] = {0}; /* count per axis type */
  for (int i = 0; i < s->n_rngs; i++) {
    char buf[32];
    int t = (int)s->types[i];
    snprintf(buf, sizeof(buf), "%s%d", axis_letter(s->types[i]), cnt[t]++);
    if (!strcmp(buf, name)) return i;
  }
  return -1;
}

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
/* Scheduler.apply_opt(PADTO): pad addresses, never the underlying allocation.
 * Load pointers need a second validity guard; a store's invalid index already
 * controls its effect. Keep both gates until normal symbolic lowering. */
static PolyUOp *sched_padto(OptScheduler *s, PolyUOp *rng, int64_t amount) {
  if (!rng || amount <= 0 || rng->n_src != 1 || rng->src[0]->op != POLY_OP_CONST ||
      rng->src[0]->arg.kind != POLY_ARG_INT)
    return NULL;
  PolyAxisType type = poly_range_axis_type(rng->arg);
  if (type == POLY_AXIS_UPCAST || type == POLY_AXIS_UNROLL || type == POLY_AXIS_THREAD) return NULL;
  int64_t size = rng->src[0]->arg.i;
  if (size <= 0 || size > INT64_MAX - (amount - 1)) return NULL;
  int64_t padded = ((size + amount - 1) / amount) * amount;
  if (size <= padded / 4) return NULL;
  PolyCtx *ctx = s->ctx;
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, rng->dtype, poly_arg_int(padded));
  PolyUOp *replacement = poly_uop1(ctx, POLY_OP_RANGE, rng->dtype, bound, rng->arg);
  PolyUOp *valid =
      poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, replacement, rng->src[0], poly_arg_none());
  int n = 0;
  PolyUOp **topo = poly_toposort_alloc(ctx, s->ast, &n);
  if (!topo) return NULL;
  PolyUOp **from = malloc(((size_t)n + 1) * sizeof(*from));
  PolyUOp **to = malloc(((size_t)n + 1) * sizeof(*to));
  if (!from || !to) {
    free(from);
    free(to);
    poly_toposort_free(topo);
    return NULL;
  }
  int count = 1;
  from[0] = rng;
  to[0] = replacement;
  for (int i = n - 1; i >= 0; i--) {
    PolyUOp *b = topo[i];
    if (b->op != POLY_OP_INDEX || b->n_src != 2) continue;
    PolyUOp *index = poly_uop_get_idx(ctx, b->src[1]);
    PolyUOp *old_valid = poly_uop_get_valid(ctx, b->src[1]);
    if (!index || !old_valid) goto failed;
    int ni = 0;
    PolyUOp **indices = poly_toposort_alloc(ctx, index, &ni);
    if (!indices) goto failed;
    bool depends = contains_uop(indices, ni, rng);
    poly_toposort_free(indices);
    if (!depends) continue;
    PolyUOp *mask = poly_uop2(ctx, POLY_OP_AND, POLY_BOOL, valid, old_valid, poly_arg_none());
    PolyUOp *invalid = poly_uop_const(ctx, poly_arg_invalid(), b->dtype);
    PolyUOp *gated =
        poly_uop3(ctx, POLY_OP_WHERE, index->dtype, mask, index, invalid, poly_arg_none());
    PolyUOp *src[] = {b->src[0], gated};
    PolyUOp *nb = poly_uop_replace_src(ctx, b, src);
    bool store_target = false;
    for (int j = 0; j < n; j++)
      if (topo[j]->op == POLY_OP_STORE && topo[j]->n_src && topo[j]->src[0] == b)
        store_target = true;
    from[count] = b;
    to[count++] =
        store_target ? nb
                     : poly_uop3(ctx, POLY_OP_WHERE, b->dtype, valid, nb, invalid, poly_arg_none());
  }
  /* Replacement values contain the old range. Substitute it explicitly:
   * UOp.substitute does not recursively apply its map to replacement values. */
  for (int i = 1; i < count; i++)
    to[i] = poly_uop_substitute(ctx, to[i], &rng, &replacement, 1);
  s->ast = poly_uop_substitute(ctx, s->ast, from, to, count);
  free(from);
  free(to);
  poly_toposort_free(topo);
  sched_refresh(s);
  return s->failed ? NULL : replacement;
failed:
  free(from);
  free(to);
  poly_toposort_free(topo);
  return NULL;
}

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

  /* Scheduler._apply_tc_opt selects reduceops[0], not the first ADD. */
  int n_topo = 0;
  PolyUOp **topo = poly_toposort_alloc(ctx, s->ast, &n_topo);
  PolyUOp *reduceop = NULL;
  for (int i = 0; i < n_topo; i++) {
    if (topo[i]->op == POLY_OP_REDUCE) {
      reduceop = topo[i];
      break;
    }
  }
  poly_toposort_free(topo);
  if (!reduceop || !use_tc || reduceop->arg.kind != POLY_ARG_REDUCE ||
      reduceop->arg.reduce.op != POLY_OP_ADD)
    return false;

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
    PolyUOp **in0_ranges = calloc((size_t)s->n_rngs, sizeof(*in0_ranges));
    PolyUOp **in1_ranges = calloc((size_t)s->n_rngs, sizeof(*in1_ranges));
    PolyUOp **red_ranges = calloc((size_t)reduceop->n_src, sizeof(*red_ranges));
    if (!in0_ranges || !in1_ranges || !red_ranges) {
      free(in0_ranges);
      free(in1_ranges);
      free(red_ranges);
      return false;
    }
    int n_in0 = 0, n_in1 = 0;
    for (int i = 0; i < s->n_rngs; i++) {
      bool r0 = poly_uop_in_ranges(ctx, in0, s->rngs[i]);
      bool r1 = poly_uop_in_ranges(ctx, in1, s->rngs[i]);
      if (r0 && !r1) in0_ranges[n_in0++] = s->rngs[i];
      if (r1 && !r0) in1_ranges[n_in1++] = s->rngs[i];
    }

    /* red_ranges from REDUCE's trailing RANGE sources */
    int n_red = 0;
    int rs = poly_range_start(POLY_OP_REDUCE);
    for (int i = rs; i < reduceop->n_src; i++) {
      if (reduceop->src[i]->op == POLY_OP_RANGE) red_ranges[n_red++] = reduceop->src[i];
    }

    /* Sort all three by axis_id descending (postrange.py:236-238) */
    if (n_in0 > 1) qsort(in0_ranges, (size_t)n_in0, sizeof(PolyUOp *), cmp_axis_id_desc);
    if (n_in1 > 1) qsort(in1_ranges, (size_t)n_in1, sizeof(PolyUOp *), cmp_axis_id_desc);
    if (n_red > 1) qsort(red_ranges, (size_t)n_red, sizeof(PolyUOp *), cmp_axis_id_desc);

    /* 4. Axis choices: product(in1_ranges, in0_ranges, red_ranges) -- note swap */
    /* Index the Cartesian product from either end without materializing it
     * or overflowing the product of its three axis counts. */
    int64_t choice = axis < 0 ? -(int64_t)axis - 1 : axis;
    if (!n_in0 || !n_in1 || !n_red || choice / n_red / n_in0 >= n_in1) {
      free(in0_ranges);
      free(in1_ranges);
      free(red_ranges);
      continue;
    }
    int red_idx = (int)(choice % n_red);
    int in0_idx = (int)((choice / n_red) % n_in0);
    int in1_idx = (int)(choice / n_red / n_in0);
    if (axis < 0) {
      red_idx = n_red - 1 - red_idx;
      in0_idx = n_in0 - 1 - in0_idx;
      in1_idx = n_in1 - 1 - in1_idx;
    }

    PolyUOp *axes[3] = {in1_ranges[in1_idx], in0_ranges[in0_idx], red_ranges[red_idx]};
    free(in0_ranges);
    free(in1_ranges);
    free(red_ranges);

    /* Tensor-core X/Y axes are output dimensions, not reductions. */
    if (poly_range_axis_type(axes[0]->arg) == POLY_AXIS_REDUCE ||
        poly_range_axis_type(axes[1]->arg) == POLY_AXIS_REDUCE)
      return false;

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

    /* postrange._apply_tc_opt tests vmax+1 before shift_to proves division.
     * Only PADTO requires a literal bound; divisible expressions need no pad. */
    bool pad_ok = true;
    for (int i = 0; i < 3; i++) {
      int64_t lo, sz;
      if (axes[i]->n_src == 0) {
        pad_ok = false;
        break;
      }
      poly_uop_minmax(ctx, axes[i]->src[0], &lo, &sz);
      if (sz <= 0 || sz % tc->dims[i] != 0) {
        if (tc_opt < 2) {
          pad_ok = false;
          break;
        }
        axes[i] = sched_padto(s, axes[i], tc->dims[i]);
        if (!axes[i]) {
          pad_ok = false;
          break;
        }
      }
    }
    if (!pad_ok) {
      continue;
    }
    /* Verify tag survived the substitute+refresh */

    /* 7. Create WARP range and apply opts (postrange.py:264-274) */
    /* UOp.range and its warp-bit arithmetic remain weak until index lowering;
     * an int32 fragment creates mixed-dtype ADDs in candidate verification. */
    PolyUOp *warp_sz = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(tc->threads));
    PolyUOp *warp =
        poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, warp_sz, poly_arg_range(-1, POLY_AXIS_WARP));

    PolyUOp *ne[32];
    int n_ne = 0;

    for (int oi = 0; oi < tc->n_opts; oi++) {
      char otype = tc->opts[oi].type;
      int odim = tc->opts[oi].dim;
      PolyUOp *new_rng = NULL;

      if (otype == 'l') {
        PolyUOp *two = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(2));
        PolyUOp *warp_mod2 =
            poly_uop2(ctx, POLY_OP_FLOORMOD, POLY_WEAKINT, warp, two, poly_arg_none());
        axes[odim] =
            sched_shift_to_core(s, axes[odim], 2, POLY_AXIS_LOCAL, false, warp_mod2, &new_rng);
        warp = poly_uop2(ctx, POLY_OP_FLOORDIV, POLY_WEAKINT, warp, two, poly_arg_none());
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

      const char *bua[32];
      int n_bua = poly_tc_base_upcast_axes(tc, bua, 32);

      /* tc_reduce_axes: axis ids for "r0","r1",... in scheduler shape_str */
      int tc_reduce_axis_ids[16];
      int n_tc_ra = 0;
      for (int ri = 0; ri < n_ra; ri++) {
        char rname[32];
        int rname_len = snprintf(rname, sizeof(rname), "r%d", ri);
        if (rname_len < 0 || rname_len >= (int)sizeof(rname)) continue;
        int si = sched_shape_str_to_axis(s, rname);
        if (si >= 0) tc_reduce_axis_ids[n_tc_ra++] = (int)poly_range_axis_id(s->rngs[si]->arg);
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
          int si = sched_shape_str_to_axis(s, bua[ui]);
          if (si >= 0) {
            upcast_pairs[dim][n_upcast[dim]][0] = poly_range_axis_id(s->rngs[si]->arg);
            upcast_pairs[dim][n_upcast[dim]][1] = 2;
            n_upcast[dim]++;
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

      /* postrange._apply_tc_opt traverses the complete trailing expressions:
       * shift_to leaves ADD/MUL expressions here, not only direct RANGEs. */
      int rs2 = poly_range_start(POLY_OP_REDUCE);
      PolyUOp *range_sink = poly_uop(
          ctx, POLY_OP_SINK, POLY_VOID, found_red->src + rs2, found_red->n_src - rs2,
          poly_arg_none()
      );
      int n_ranges_topo = 0;
      PolyUOp **ranges_topo = poly_toposort_alloc(ctx, range_sink, &n_ranges_topo);
      PolyUOp **red_srcs =
          ranges_topo ? calloc((size_t)n_ranges_topo + 1, sizeof(*red_srcs)) : NULL;
      if (!red_srcs) {
        poly_toposort_free(ranges_topo);
        return false;
      }
      red_srcs[0] = tc_uop;
      int n_extra = 0;
      for (int i = 0; i < n_ranges_topo; i++) {
        if (ranges_topo[i]->op != POLY_OP_RANGE) continue;
        int64_t aid = poly_range_axis_id(ranges_topo[i]->arg);
        bool in_tc = false;
        for (int r = 0; r < n_tc_ra; r++)
          if (tc_reduce_axis_ids[r] == (int)aid) {
            in_tc = true;
            break;
          }
        if (!in_tc) red_srcs[++n_extra] = ranges_topo[i];
      }
      poly_toposort_free(ranges_topo);
      if (n_extra >= UINT16_MAX) {
        free(red_srcs);
        return false;
      }
      if (n_extra > 0) {
        PolyArg red_arg = poly_arg_reduce(POLY_OP_ADD, 0);
        tc_uop = poly_uop(ctx, POLY_OP_REDUCE, tc_uop->dtype, red_srcs, n_extra + 1, red_arg);
      }
      free(red_srcs);

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
    s->tensor_core = tc;
    sched_refresh(s);
    return true;
  }

  return false;
}

static int sched_range_index(const OptScheduler *s, PolyUOp *range) {
  for (int i = 0; i < s->n_rngs; i++)
    if (s->rngs[i] == range) return i;
  return -1;
}

/* heuristic.hand_coded_optimizations: TC selection and post-TC M/N choices.
 * The heuristic and TC-only policy share this sequence and its Opt history. */
static bool sched_hand_coded_tensor_cores(OptScheduler *s, PolyRendererCaps caps) {
  if (caps.n_tensor_cores <= 0) return false;
  int n_reduce = 0;
  for (int i = 0; i < s->n_rngs; i++)
    n_reduce += s->types[i] == POLY_AXIS_GROUP_REDUCE || s->types[i] == POLY_AXIS_REDUCE;
  int64_t args[] = {
      poly_getenv_int("TC_SELECT", -1), poly_getenv_int("TC_OPT", 0), poly_getenv_int("TC", 1)};
  if (args[2] <= 0 || (n_reduce != 1 && args[1] < 1)) return false;
  for (int axis = 0; axis < 3; axis++) {
    OptScheduler candidate;
    sched_copy(&candidate, s);
    PolyUOp *ranges[3] = {0};
    bool ok = sched_apply_opt(
        &candidate, caps,
        (PolyOpt
        ){.op = POLY_OPT_TC,
          .has_axis = true,
          .axis = axis,
          .arg_kind = POLY_OPT_ARG_INT_TUPLE,
          .arg_tuple = args,
          .n_arg_tuple = 3},
        ranges
    );
    if (!ok) {
      sched_destroy(&candidate);
      continue;
    }
    for (int dim = 1; ok && dim >= 0; dim--) {
      for (int amount = 5; amount >= 2; amount--) {
        if (!poly_uop_divides(s->ctx, ranges[dim]->src[0], amount)) continue;
        ranges[dim] = sched_require_int_opt(
            &candidate, caps, POLY_OPT_UPCAST, sched_range_index(&candidate, ranges[dim]), amount
        );
        ok = ranges[dim] != NULL;
        break;
      }
    }
    if (ok) {
      for (int amount = 4; amount >= 2; amount -= 2) {
        if (!poly_uop_divides(s->ctx, ranges[0]->src[0], amount)) continue;
        ok = sched_require_int_opt(
                 &candidate, caps, POLY_OPT_LOCAL, sched_range_index(&candidate, ranges[0]), amount
             ) != NULL;
        break;
      }
    }
    if (ok) {
      sched_destroy(s);
      *s = candidate;
      return true;
    }
    sched_destroy(&candidate);
    s->failed = true;
    return false;
  }
  return false;
}

/* hand_coded_optimizations (heuristic.py:8-190) */
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
  if (!sched_can_optimize(&s)) {
    sched_destroy(&s);
    return sink;
  }

  if (sched_hand_coded_tensor_cores(&s, caps) || s.failed) goto done;

  /* heuristic.py: image lanes precede local/group and masked-axis decisions. */
  if (poly_getenv_flag("IMAGE")) {
    for (int bi = 0; bi < s.n_bufs; bi++) {
      PolyUOp *buf = s.bufs[bi];
      if (buf->n_src < 2) continue;
      int n_dims = 0;
      PolyImageDim *dims = poly_image_valid_dims(
          buf->src[0]->dtype, poly_uop_max_numel(ctx, buf->src[0]), caps.arch, &n_dims
      );
      free(dims);
      if (!n_dims) continue;
      PolyUOp *idx = poly_uop_get_idx(ctx, buf->src[1]);
      PolyUOp *valid = poly_uop_get_valid(ctx, buf->src[1]);
      int n_add = 0, n_valid = 0;
      PolyUOp **add = idx ? poly_uop_split(idx, POLY_OP_ADD, &n_add) : NULL;
      PolyUOp **valid_topo = valid ? poly_toposort_alloc(ctx, valid, &n_valid) : NULL;
      if (!add || !valid_topo) {
        free(add);
        poly_toposort_free(valid_topo);
        goto done;
      }
      int axis = -1;
      for (int i = 0; i < n_add; i++) {
        PolyUOp *r = add[i];
        if (r->op != POLY_OP_RANGE) continue;
        bool gated = false;
        for (int j = 0; j < n_valid; j++)
          gated |= valid_topo[j] == r;
        if (gated) continue;
        for (int j = 0; j < s.n_rngs; j++)
          if (s.rngs[j] == r && s.shape[j] > 1 && s.shape[j] % 4 == 0) axis = j;
        if (axis >= 0) break;
      }
      free(add);
      poly_toposort_free(valid_topo);
      if (axis < 0) continue;
      PolyAxisType type = s.types[axis];
      if (type == POLY_AXIS_GLOBAL || type == POLY_AXIS_LOCAL || type == POLY_AXIS_WEAK)
        sched_require_int_opt(&s, caps, POLY_OPT_UPCAST, axis, 4);
      else if (type == POLY_AXIS_REDUCE || type == POLY_AXIS_GROUP_REDUCE) {
        int ordinal = 0;
        for (int j = 0; j < axis; j++)
          ordinal += (s.types[j] == POLY_AXIS_REDUCE || s.types[j] == POLY_AXIS_GROUP_REDUCE) &&
                     s.shape[j] > 1;
        sched_require_int_opt(&s, caps, POLY_OPT_UNROLL, ordinal, 4);
      }
      if (s.failed) goto done;
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
      /* Scheduler.reduceop selects the first reduction, not the first ADD. */
      if (u->op == POLY_OP_REDUCE) {
        reduceop = u;
        break;
      }
    }

    PolyUOp *mul = reduceop && reduceop->arg.kind == POLY_ARG_REDUCE &&
                           reduceop->arg.reduce.op == POLY_OP_ADD && reduceop->n_src > 0 &&
                           reduceop->src[0]->op == POLY_OP_MUL && reduceop->src[0]->n_src == 2
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
      int n_terms = 0;
      PolyUOp **terms = poly_uop_split(idx0, POLY_OP_ADD, &n_terms);
      if (!terms) {
        poly_toposort_free(topo);
        goto done;
      }
      for (int i = 0; i < n_terms; i++) {
        if (terms[i] == s.rngs[first_reduce]) {
          reduce_is_addend = true;
          break;
        }
      }
      free(terms);
    }
    bool second_covers_first = idx0 && idx1;
    for (int i = 0; second_covers_first && i < s.n_rngs; i++) {
      if (poly_uop_in_ranges(ctx, idx0, s.rngs[i]) && !poly_uop_in_ranges(ctx, idx1, s.rngs[i]))
        second_covers_first = false;
    }

    if (reduce_is_addend && second_covers_first) {
      for (int global_idx = 0; global_idx < s.n_rngs; global_idx++) {
        if (s.types[global_idx] != POLY_AXIS_GLOBAL) continue;
        int64_t global_size = s.shape[global_idx];
        if (mv_threads_per_row <= 0 || mv_blocksize <= 0 || mv_rows_per_thread <= 0 ||
            !poly_uop_divides(ctx, s.rngs[first_reduce]->src[0], mv_threads_per_row))
          continue;
        int64_t block = (int64_t)mv_blocksize * mv_rows_per_thread;
        /* heuristic.py compares full_shape % block to the Python integer0.
         * UOp equality is identity, not a symbolic divisibility predicate. */
        if (global_size <= 0 || global_size % block != 0) continue;

        if (mv_threads_per_row > 1)
          sched_apply_int_opt(&s, caps, POLY_OPT_GROUP, 0, mv_threads_per_row);
        if (mv_blocksize > 1)
          sched_require_int_opt(&s, caps, POLY_OPT_LOCAL, global_idx, mv_blocksize);
        if (mv_rows_per_thread > 1)
          sched_require_int_opt(&s, caps, POLY_OPT_UPCAST, global_idx, mv_rows_per_thread);
        poly_toposort_free(topo);
        PolyUOp *out = s.failed ? NULL : s.ast;
        sched_destroy(&s);
        return out;
      }
    }
    poly_toposort_free(topo);
  }

  /* == Group for reduces (tinygrad heuristic.py:101-110) ==
   * Try GROUPTOP(16) on the first few REDUCE axes when the output footprint
   * is small enough. If grouping succeeds, stop here like tinygrad and do not
   * fall through into the later reduce-unroll heuristic. */
  if (caps.has_local &&
      sched_output_prod_upcastable(&s) <= (poly_getenv_flag("NOLOCALS") ? 240 : 2048)) {
    for (int axis = 0; axis < 3; axis++) {
      int ridx = -1, remaining = axis;
      for (int i = 0; i < s.n_rngs; i++) {
        if (s.types[i] == POLY_AXIS_REDUCE && remaining-- == 0) {
          ridx = i;
          break;
        }
      }
      if (ridx < 0) break;
      /* apply_opt/shift_to proves symbolic divisibility; no static-size guard. */
      if (sched_apply_int_opt(&s, caps, POLY_OPT_GROUPTOP, axis, 16)) {
        break;
      }
    }
  }

  if (sched_has_axis_type(&s, POLY_AXIS_GROUP_REDUCE)) goto done;

  /* == Masked upcast (heuristic.py:96-105) ==
   * Upcast small dims (<=7) that appear in WHERE gates */
  {
    int *up_dims = calloc((size_t)s.n_rngs, sizeof(*up_dims));
    int *to_upcast = calloc((size_t)s.n_rngs, sizeof(*to_upcast));
    if (!up_dims || !to_upcast) {
      free(up_dims);
      free(to_upcast);
      goto done;
    }
    int n_up = sched_upcastable_dims(&s, up_dims, s.n_rngs);
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
      if (prod > 49) continue;
      if (poly_getenv_flag("IMAGE") && s.types[axis] == POLY_AXIS_GLOBAL) {
        int64_t global_upcast = s.shape[axis];
        for (int j = 0; j < n_to_upcast; j++)
          if (s.types[to_upcast[j]] == POLY_AXIS_GLOBAL) global_upcast *= s.shape[to_upcast[j]];
        PolyUOp *items = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(1));
        for (int j = 0; j < s.n_rngs; j++)
          if (s.types[j] == POLY_AXIS_GLOBAL)
            items =
                poly_alu2(ctx, POLY_OP_MUL, items, poly_cast(ctx, s.rngs[j]->src[0], POLY_WEAKINT));
        items = poly_alu2(
            ctx, POLY_OP_IDIV, items,
            poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(global_upcast))
        );
        PolyUOp *too_small = poly_alu2(
            ctx, POLY_OP_CMPLT, items,
            poly_uop0(
                ctx, POLY_OP_CONST, POLY_WEAKINT,
                poly_arg_int(poly_getenv_int("OCCUPANCY_FLOOR", 4096))
            )
        );
        if (poly_uop_resolve(ctx, too_small, 0) == 1) continue;
      }
      to_upcast[n_to_upcast++] = axis;
    }
    poly_toposort_free(topo);

    /* Apply in reverse order (matching tinygrad) */
    for (int i = n_to_upcast - 1; i >= 0; i--) {
      int axis = to_upcast[i];
      if (axis < s.n_rngs && s.shape[axis] > 1)
        sched_require_int_opt(&s, caps, POLY_OPT_UPCAST, axis, 0);
      if (s.failed) break;
    }
    free(up_dims);
    free(to_upcast);
  }
  if (s.failed) goto done;

  /* == Multi-axis UPCAST with stride scoring (heuristic.py:107-133) == */
  {
    int n_axis_flags = s.n_rngs;
    bool *upcasted_axis = calloc((size_t)n_axis_flags, sizeof(*upcasted_axis));
    if (!upcasted_axis) goto done;

    while (sched_output_prod_upcastable(&s) >= 1024) {
      int below = sched_upcast_size_lt(&s, 32);
      if (below < 0) {
        free(upcasted_axis);
        s.failed = true;
        goto done;
      }
      if (!below) break;

      /* Score each candidate (num_strides, sum_strides, axis, amount) */
      typedef struct {
        int num_strides;
        PolyInt sum_strides;
        int axis;
        int amount;
      } UpChoice;
      UpChoice best = {0};
      int n_choices = 0;

      bool is_dsp = caps.device && !strcmp(caps.device, "DSP");
      int amounts[] = {is_dsp ? 128 : 3, 4};
      int n_amounts = is_dsp ? 1 : 2;
      if (is_dsp) {
        for (int i = 0; i < n_axis_flags; i++)
          if (upcasted_axis[i]) n_amounts = 0;
      }
      for (int axis = 0; axis < s.n_rngs; axis++) {
        PolyAxisType t = s.types[axis];
        if ((t != POLY_AXIS_GLOBAL && t != POLY_AXIS_LOCAL && t != POLY_AXIS_WEAK) ||
            s.shape[axis] <= 1)
          continue;
        if (axis < n_axis_flags && upcasted_axis[axis]) continue;

        for (int ai = 0; ai < n_amounts; ai++) {
          int amount = amounts[ai];
          if (s.shape[axis] % amount != 0) continue;

          /* Expanded axis check (heuristic.py:117-118):
           * Must have a buffer where rng is NOT in index but all UPCAST/UNROLL rngs ARE */
          bool has_expanded_buf = false;
          if (s.has_reach) {
            for (int bi = 0; bi < s.n_bufs && !has_expanded_buf; bi++) {
              if (sched_buf_reaches(&s, bi, axis)) continue;
              /* Check all existing UPCAST/UNROLL ranges are in this buf's index */
              bool all = true;
              for (int ri = 0; ri < s.n_rngs; ri++)
                if ((s.types[ri] == POLY_AXIS_UPCAST || s.types[ri] == POLY_AXIS_UNROLL) &&
                    !sched_buf_reaches(&s, bi, ri)) {
                  all = false;
                  break;
                }
              if (all) has_expanded_buf = true;
            }
          }
          if (!has_expanded_buf) continue;

          /* Count strides (heuristic.py:119-127) */
          int num_strides = 0;
          PolyInt sum_strides = {0};
          if (!sched_stride_score(&s, axis, &num_strides, &sum_strides)) {
            poly_int_free(&sum_strides);
            poly_int_free(&best.sum_strides);
            free(upcasted_axis);
            goto done;
          }

          if (!n_choices || num_strides < best.num_strides ||
              (num_strides == best.num_strides && poly_int_cmp(&sum_strides, &best.sum_strides) < 0
              )) {
            poly_int_free(&best.sum_strides);
            best = (UpChoice){num_strides, sum_strides, axis, amount};
          } else
            poly_int_free(&sum_strides);
          n_choices++;
        }
      }

      if (n_choices == 0) break;

      /* Enumeration already supplies the axis/amount tie-break order. */
      int best_axis = best.axis;
      int best_amount = best.amount;
      poly_int_free(&best.sum_strides);
      if (best_axis < s.n_rngs && s.shape[best_axis] > 1)
        sched_require_int_opt(&s, caps, POLY_OPT_UPCAST, best_axis, best_amount);
      if (s.failed) break;
      /* Pinned heuristic tracks the selected index, not UOp identity. */
      if (best_axis < n_axis_flags) upcasted_axis[best_axis] = true;
    }
    free(upcasted_axis);
  }
  if (s.failed) goto done;

  /* == Reduce UNROLL (heuristic.py:135-149) == */
  if (s.has_reduce) {
    int capacity = s.n_rngs;
    int *unroll_dims = calloc((size_t)capacity, sizeof(*unroll_dims));
    if (!unroll_dims) goto done;
    int n_unroll = sched_unrollable_dims(&s, unroll_dims, capacity);

    int can_unroll = 0;
    if (n_unroll > 0) {
      can_unroll = sched_upcast_size_lt(&s, 5);
      if (can_unroll >= 0 && (can_unroll || !sched_has_axis_type(&s, POLY_AXIS_UNROLL)))
        can_unroll = sched_upcast_size_lt(&s, 64);
      if (can_unroll < 0) {
        free(unroll_dims);
        s.failed = true;
        goto done;
      }
    }
    if (can_unroll) {
      int last = unroll_dims[n_unroll - 1];
      int64_t last_sz = s.shape[last];

      if (last_sz <= 32) {
        /* Unroll fully (amount = full size) */
        if (last < s.n_rngs && last_sz > 1)
          sched_apply_int_opt(&s, caps, POLY_OPT_UNROLL, n_unroll - 1, 0);
        /* If small, try unrolling a second reduce dim */
        n_unroll = sched_unrollable_dims(&s, unroll_dims, capacity);
        if (n_unroll > 0 && last_sz <= 3 && s.shape[unroll_dims[n_unroll - 1]] <= 3) {
          int last2 = unroll_dims[n_unroll - 1];
          if (last2 < s.n_rngs && s.shape[last2] > 1)
            sched_apply_int_opt(&s, caps, POLY_OPT_UNROLL, n_unroll - 1, 0);
        }
      } else {
        /* Partial unroll by 4 if divisible */
        if (last_sz % 4 == 0 && last < s.n_rngs)
          sched_apply_int_opt(&s, caps, POLY_OPT_UNROLL, n_unroll - 1, 4);
      }
    }
    free(unroll_dims);
  }

  /* heuristic.py's fallback is four lanes, independent of renderer width. */
  if (!sched_upcasted(&s)) {
    int *up_dims = calloc((size_t)s.n_rngs, sizeof(*up_dims));
    if (!up_dims) goto done;
    int n_up = sched_upcastable_dims(&s, up_dims, s.n_rngs);
    if (n_up > 0) {
      int last = up_dims[n_up - 1];
      if (s.shape[last] % 4 == 0) sched_require_int_opt(&s, caps, POLY_OPT_UPCAST, last, 4);
    }
    free(up_dims);
  }
  if (s.failed) goto done;

  /* == Local groups (heuristic.py:160-175 subset) ==
   * Port the tinygrad local scheduling block for backends with workgroup
   * locals. This is what splits large LOOP/GLOBAL axes into LOOP x LOCAL for
   * WebGPU masked kernels like triu(9,9). */
  if (caps.has_local && poly_getenv_flag("NOLOCALS")) {
    if (!sched_apply_opt(&s, caps, (PolyOpt){.op = POLY_OPT_NOLOCALS}, NULL)) {
      s.failed = true;
      goto done;
    }
  } else if (caps.has_local) {
    LocalAxisRank *ranked = calloc((size_t)s.n_rngs, sizeof(*ranked));
    LocalChoice *to_local = calloc((size_t)s.n_rngs, sizeof(*to_local));
    if (!ranked || !to_local) {
      free(ranked);
      free(to_local);
      goto done;
    }
    int n_ranked = 0;
    for (int axis = 0; axis < s.n_rngs; axis++) {
      if (!(s.types[axis] == POLY_AXIS_GLOBAL || s.types[axis] == POLY_AXIS_WEAK)) continue;
      /* heuristic.py selects LOCAL only for an original literal extent. */
      if (s.rngs[axis]->src[0]->op != POLY_OP_CONST) continue;
      if (s.shape[axis] <= 1) continue;
      bool expanded = false;
      if (s.has_reach) {
        for (int bi = 0; bi < s.n_bufs; bi++) {
          if (!sched_buf_reaches(&s, bi, axis)) {
            expanded = true;
            break;
          }
        }
      }
      ranked[n_ranked++] = (LocalAxisRank){.expanded = expanded, .axis = axis};
    }
    qsort(ranked, (size_t)n_ranked, sizeof(LocalAxisRank), cmp_local_axis_rank);

    int n_to_local = 0;
    for (int ri = 0; ri < n_ranked; ri++) {
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
          sched_require_int_opt(&s, caps, POLY_OPT_LOCAL, axis, to_local[i].size);
        if (s.failed) break;
        if (will_delete_shape) deleted_shape++;
      }
    }
    free(ranked);
    free(to_local);
  }
  if (s.failed) goto done;

  /* == CPU THREAD axis (tinygrad heuristic.py:180-190) ==
   * ClangRenderer has has_threads=true, then gpudims.py replaces the THREAD
   * axis with the runtime ALU PARAM "core_id". Keep this after local grouping
   * just like tinygrad's final heuristic block. */
  if (caps.has_threads && caps.global_max[0] > 1 && !sched_has_axis_type(&s, POLY_AXIS_THREAD)) {
    int candidates[] = {32, 16, 12, 8, 6, 5, 4, 3, 2};
    PolyUOp *work = poly_alu2(
        ctx, POLY_OP_IDIV, sched_full_shape_prod(&s),
        poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(128LL << 10))
    );
    for (int ci = 0; ci < (int)(sizeof(candidates) / sizeof(candidates[0])); ci++) {
      int threads = candidates[ci];
      if (threads > caps.global_max[0]) continue;
      PolyUOp *too_small = poly_alu2(
          ctx, POLY_OP_CMPLT, work,
          poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(threads))
      );
      if (poly_uop_resolve(ctx, too_small, 1) != 0) continue;
      for (int axis = 0; axis < s.n_rngs; axis++) {
        if (s.types[axis] != POLY_AXIS_WEAK) continue;
        if (s.shape[axis] <= 1 || (s.shape[axis] % threads) != 0) continue;
        if (sched_apply_int_opt(&s, caps, POLY_OPT_THREAD, axis, threads)) {
          ci = (int)(sizeof(candidates) / sizeof(candidates[0]));
          break;
        }
      }
    }
  }

done:;
  PolyUOp *out = s.failed ? NULL : s.ast;
  sched_destroy(&s);
  return out;
}

/* Scheduler.apply_opt (pinned codegen/opt/postrange.py). Action history is
 * immutable KernelInfo data, so failed candidates cannot mutate a parent. */
static int sched_real_axis(const OptScheduler *s, PolyOpt opt) {
  if (!opt.has_axis || opt.op == POLY_OPT_TC) return -1;
  if (opt.op == POLY_OPT_UNROLL || opt.op == POLY_OPT_GROUP || opt.op == POLY_OPT_GROUPTOP) {
    int remaining = opt.axis;
    /* These options index a filtered Python list, unlike absolute axes. */
    if (remaining < 0) {
      for (int i = 0; i < s->n_rngs; i++) {
        bool match = s->types[i] == POLY_AXIS_REDUCE ||
                     (opt.op == POLY_OPT_UNROLL && s->types[i] == POLY_AXIS_GROUP_REDUCE);
        remaining += match && (opt.op != POLY_OPT_UNROLL || s->shape[i] > 1);
      }
      if (remaining < 0) return -1;
    }
    for (int i = 0; i < s->n_rngs; i++) {
      bool match = s->types[i] == POLY_AXIS_REDUCE ||
                   (opt.op == POLY_OPT_UNROLL && s->types[i] == POLY_AXIS_GROUP_REDUCE);
      if (match && (opt.op != POLY_OPT_UNROLL || s->shape[i] > 1) && remaining-- == 0) return i;
    }
    return -1;
  }
  return opt.axis < s->n_rngs ? opt.axis : -1;
}

static bool sched_globalizable(OptScheduler *s, PolyUOp *rng) {
  int n = 0;
  PolyUOp **topo = poly_toposort_alloc(s->ctx, s->ast, &n);
  if (!topo) return false;
  bool output = sched_is_globalizable(s->ctx, s->ast, rng, topo, n);
  poly_toposort_free(topo);
  return output;
}

static bool sched_apply_opt_impl(
    OptScheduler *s,
    PolyRendererCaps caps,
    PolyOpt opt,
    PolyUOp **result
) {
  if (!s || s->failed) return false;
  PolyKernelInfo info = s->ast->arg.kind == POLY_ARG_KERNEL_INFO && s->ast->arg.kernel_info
                            ? *s->ast->arg.kernel_info
                            : (PolyKernelInfo){0};
  if (!info.name) info.name = "test";
  int axis = sched_real_axis(s, opt);
  PolyUOp *rng = axis >= 0 ? s->rngs[axis] : NULL;
  bool local = false, grouped = false, threaded = false;
  for (int i = 0; i < s->n_rngs; i++) {
    local |= s->types[i] == POLY_AXIS_WARP || s->types[i] == POLY_AXIS_LOCAL ||
             s->types[i] == POLY_AXIS_GROUP_REDUCE;
    grouped |= s->types[i] == POLY_AXIS_GROUP_REDUCE;
    threaded |= s->types[i] == POLY_AXIS_THREAD;
  }
  if (opt.op == POLY_OPT_NOLOCALS) {
    if (local) return false;
    info.dont_use_locals = true;
  } else if (opt.op == POLY_OPT_TC) {
    if (info.n_applied_opts || !opt.has_axis || opt.arg_kind != POLY_OPT_ARG_INT_TUPLE ||
        opt.n_arg_tuple != 3 || !opt.arg_tuple || opt.arg_tuple[0] < -1 ||
        opt.arg_tuple[0] >= caps.n_tensor_cores || opt.arg_tuple[1] < 0 || opt.arg_tuple[1] > 2 ||
        opt.arg_tuple[2] <= 0 || opt.arg_tuple[2] > 2)
      return false;
    if (!sched_apply_tc_opt(
            s, opt.axis, (int)opt.arg_tuple[0], (int)opt.arg_tuple[1], (int)opt.arg_tuple[2],
            caps.device, caps.tensor_cores, caps.n_tensor_cores, result
        ))
      goto failed;
  } else if (opt.op == POLY_OPT_PADTO) {
    if (opt.arg_kind != POLY_OPT_ARG_INT || !sched_padto(s, rng, opt.arg)) goto failed;
  } else if (opt.op == POLY_OPT_SWAP) {
    if (!rng || opt.arg_kind != POLY_OPT_ARG_INT || opt.arg < -s->n_rngs || opt.arg >= s->n_rngs)
      return false;
    PolyUOp *other = s->rngs[opt.arg < 0 ? s->n_rngs + opt.arg : opt.arg];
    if (poly_range_axis_type(rng->arg) != POLY_AXIS_GLOBAL ||
        poly_range_axis_type(other->arg) != POLY_AXIS_GLOBAL)
      return false;
    PolyUOp *from[] = {rng, other};
    PolyUOp *to[] = {
        poly_uop_tagged(s->ctx, rng->op, rng->dtype, rng->src, rng->n_src, other->arg, 1),
        poly_uop_tagged(s->ctx, other->op, other->dtype, other->src, other->n_src, rng->arg, 1)};
    s->ast = poly_uop_substitute(s->ctx, s->ast, from, to, 2);
    int n = 0;
    PolyUOp **topo = poly_toposort_alloc(s->ctx, s->ast, &n);
    PolyUOp **untagged = topo ? malloc((size_t)n * sizeof(*untagged)) : NULL;
    if (!topo || !untagged) {
      poly_toposort_free(topo);
      free(untagged);
      goto failed;
    }
    int count = 0;
    for (int i = 0; i < n; i++) {
      PolyUOp *u = topo[i];
      if (!u->tag && u->tag_arg.kind == POLY_ARG_NONE) continue;
      topo[count] = u;
      untagged[count++] = poly_uop(s->ctx, u->op, u->dtype, u->src, u->n_src, u->arg);
    }
    s->ast = poly_uop_substitute(s->ctx, s->ast, topo, untagged, count);
    poly_toposort_free(topo);
    free(untagged);
    sched_refresh(s);
  } else {
    if (!rng || opt.arg_kind != POLY_OPT_ARG_INT || opt.arg < 0) return false;
    PolyAxisType type = s->types[axis], to_type;
    int64_t amount = opt.arg;
    if (!amount) {
      /* Scheduler.apply_opt uses rng.vmax+1; shift_to still proves divisibility. */
      int64_t lo, hi;
      poly_uop_minmax(s->ctx, rng, &lo, &hi);
      if (hi == INT64_MAX) return false;
      amount = hi + 1;
    }
    if (amount <= 0) return false;
    switch (opt.op) {
    case POLY_OPT_UPCAST:
      if ((amount > 16 && (!caps.device || strcmp(caps.device, "DSP"))) ||
          (type != POLY_AXIS_GLOBAL && type != POLY_AXIS_LOCAL && type != POLY_AXIS_WEAK))
        return false;
      to_type = POLY_AXIS_UPCAST;
      break;
    case POLY_OPT_UNROLL:
      if (amount > 32 || (type != POLY_AXIS_REDUCE && type != POLY_AXIS_GROUP_REDUCE)) return false;
      to_type = POLY_AXIS_UNROLL;
      break;
    case POLY_OPT_LOCAL:
      if (!caps.has_local || info.dont_use_locals ||
          (type != POLY_AXIS_GLOBAL && type != POLY_AXIS_WEAK))
        return false;
      to_type = POLY_AXIS_LOCAL;
      break;
    case POLY_OPT_THREAD:
      if (!caps.has_threads || amount > caps.global_max[0] || threaded ||
          !sched_globalizable(s, rng))
        return false;
      to_type = POLY_AXIS_THREAD;
      break;
    case POLY_OPT_GROUP:
    case POLY_OPT_GROUPTOP:
      if (!caps.has_local || info.dont_use_locals || type != POLY_AXIS_REDUCE) return false;
      for (int i = 0; i < info.n_applied_opts; i++)
        if (info.applied_opts[i].op == POLY_OPT_TC) return false;
      to_type = POLY_AXIS_GROUP_REDUCE;
      break;
    default:
      return false;
    }
    if (s->has_reduce && (grouped || to_type == POLY_AXIS_GROUP_REDUCE)) {
      int n = 0;
      PolyUOp **topo = poly_toposort_alloc(s->ctx, s->ast, &n);
      if (!topo) return false;
      int itemsize = 0;
      bool nested = false;
      for (int i = 0; i < n; i++) {
        PolyUOp *u = topo[i];
        if (u->op != POLY_OP_REDUCE) continue;
        if (!itemsize) itemsize = poly_dtype_itemsize(u->dtype);
        if (to_type != POLY_AXIS_GROUP_REDUCE) continue;
        bool ends = false;
        for (int j = 1; j < u->n_src; j++)
          ends |= poly_uop_in_ranges(s->ctx, u->src[j], rng);
        if (ends)
          for (int j = 0; j < s->n_rngs; j++)
            if ((s->types[j] == POLY_AXIS_REDUCE || s->types[j] == POLY_AXIS_UNROLL ||
                 s->types[j] == POLY_AXIS_GROUP_REDUCE) &&
                poly_uop_in_ranges(s->ctx, u, s->rngs[j]))
              nested = true;
      }
      poly_toposort_free(topo);
      if (nested) return false;
      int64_t limit = caps.shared_max > 0 ? caps.shared_max : 32768;
      if (itemsize <= 0) return false;
      PolyUOp *size = poly_alu2(
          s->ctx, POLY_OP_MUL,
          poly_uop0(s->ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(itemsize)),
          poly_uop0(s->ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(amount))
      );
      for (int i = 0; i < s->n_rngs; i++) {
        PolyAxisType t = s->types[i];
        if (t != POLY_AXIS_UPCAST && t != POLY_AXIS_WARP && t != POLY_AXIS_LOCAL &&
            t != POLY_AXIS_GROUP_REDUCE)
          continue;
        size = poly_alu2(
            s->ctx, POLY_OP_MUL, size, poly_cast(s->ctx, s->rngs[i]->src[0], POLY_WEAKINT)
        );
      }
      if (sched_eval_lt(s->ctx, size, limit + 1) != 1) return false;
    }
    PolyUOp *replaced = sched_shift_to_core(
        s, rng, amount, to_type, opt.op == POLY_OPT_GROUPTOP || opt.op == POLY_OPT_THREAD, NULL,
        result ? &result[1] : NULL
    );
    if (!replaced) goto failed;
    if (result) result[0] = replaced;
  }
  if (info.n_applied_opts == INT_MAX) goto failed;
  PolyOpt *history = malloc(((size_t)info.n_applied_opts + 1) * sizeof(*history));
  if (!history) goto failed;
  if (info.n_applied_opts)
    memcpy(history, info.applied_opts, (size_t)info.n_applied_opts * sizeof(*history));
  history[info.n_applied_opts++] = opt;
  info.applied_opts = history;
  s->ast = poly_uop_tagged_arg(
      s->ctx, s->ast->op, s->ast->dtype, s->ast->src, s->ast->n_src, poly_arg_kernel_info(&info),
      s->ast->tag, s->ast->tag_arg
  );
  free(history);
  if (!s->ast) goto failed;
  return true;
failed:
  return false;
}

static bool sched_apply_opt(OptScheduler *s, PolyRendererCaps caps, PolyOpt opt, PolyUOp **result) {
  if (!s || s->failed) return false;
  OptScheduler candidate;
  sched_copy(&candidate, s);
  /* An unchanged candidate owns a second reference; refresh instead gives it
   * a new snapshot. Releasing either owner cannot invalidate the other. */
  if (!sched_apply_opt_impl(&candidate, caps, opt, result) || candidate.failed) {
    sched_destroy(&candidate);
    return false;
  }
  sched_destroy(s);
  *s = candidate;
  return true;
}

/* search.py:actions. Tuple storage belongs to the catalogue; KernelInfo copies
 * each applied Opt, including its tuple, when publishing scheduler history. */
typedef struct {
  PolyOpt opts[201];
  int64_t tc_args[10][3];
  int count;
} BeamActions;

static void beam_add_action(BeamActions *a, PolyOptOps op, int axis, int64_t amount) {
  a->opts[a->count++] = (PolyOpt
  ){.op = op, .has_axis = true, .axis = axis, .arg_kind = POLY_OPT_ARG_INT, .arg = amount};
}

static void beam_actions(BeamActions *a) {
  *a = (BeamActions){0};
  static const struct {
    PolyOptOps op;
    int axes, count;
    int amounts[10];
  } groups[] = {
      {POLY_OPT_UPCAST, 8, 6, {0, 2, 3, 4, 5, 7}},
      {POLY_OPT_UNROLL, 5, 3, {0, 4, 7}},
      {POLY_OPT_LOCAL, 6, 7, {2, 3, 4, 8, 13, 16, 29}},
      {POLY_OPT_GROUPTOP, 3, 8, {13, 16, 28, 29, 32, 49, 64, 256}},
      {POLY_OPT_GROUP, 3, 4, {0, 4, 8, 16}},
  };
  for (size_t g = 0; g < sizeof(groups) / sizeof(groups[0]); g++)
    for (int i = 0; i < groups[g].count; i++)
      for (int axis = 0; axis < groups[g].axes; axis++)
        beam_add_action(a, groups[g].op, axis, groups[g].amounts[i]);
  if (poly_getenv_flag("BEAM_PADTO"))
    for (int axis = 0; axis < 7; axis++)
      beam_add_action(a, POLY_OPT_PADTO, axis, 32);
  beam_add_action(a, POLY_OPT_LOCAL, 0, 32);
  beam_add_action(a, POLY_OPT_LOCAL, 6, 2);
  for (int i = 0; i < 10; i++) {
    a->tc_args[i][0] = -1;
    a->tc_args[i][1] = i == 0 ? 0 : poly_getenv_int("TC_OPT", 2);
    a->tc_args[i][2] = poly_getenv_int("TC", 1);
    a->opts[a->count++] = (PolyOpt
    ){.op = POLY_OPT_TC,
      .has_axis = true,
      .axis = i ? i - 1 : 0,
      .arg_kind = POLY_OPT_ARG_INT_TUPLE,
      .arg_tuple = a->tc_args[i],
      .n_arg_tuple = 3};
  }
  for (int i = 0; i < 5; i++)
    for (int j = i + 1; j < 5; j++)
      beam_add_action(a, POLY_OPT_SWAP, i, j);
  static const int threads[] = {2, 3, 4, 5, 8, 12, 16, 24, 32, 64};
  for (int i = 0; i < 10; i++)
    for (int axis = 0; axis < 3; axis++)
      beam_add_action(a, POLY_OPT_THREAD, axis, threads[i]);
  if (poly_getenv_flag("NOLOCALS")) a->opts[a->count++] = (PolyOpt){.op = POLY_OPT_NOLOCALS};
}

typedef struct {
  OptScheduler sched;
  double time_us;
  int order; /* Python's sorted preserves enumeration order for equal times. */
} BeamEntry;

/* search.get_kernel_actions: apply_opt owns legality; these are search budgets
 * and duplicate whole-axis spellings, not alternate scheduling semantics. */
static bool beam_get_kernel_action(
    OptScheduler *s,
    PolyRendererCaps caps,
    PolyOpt opt,
    const BeamActions *actions
) {
  if (opt.has_axis && opt.op != POLY_OPT_TC) {
    int axis = sched_real_axis(s, opt);
    if (axis < 0) return false;
    if (opt.arg_kind == POLY_OPT_ARG_INT && s->shape[axis] > 0 && s->shape[axis] == opt.arg) {
      for (int i = 0; i < actions->count; i++) {
        PolyOpt zero = actions->opts[i];
        if (zero.op == opt.op && zero.has_axis && zero.axis == opt.axis &&
            zero.arg_kind == POLY_OPT_ARG_INT && zero.arg == 0)
          return false;
      }
    }
  }
  if (!sched_apply_opt(s, caps, opt, NULL)) return false;
  long double up = 1, local = 1, tc_up = 1;
  bool symbolic = false;
  if (s->tensor_core)
    tc_up = (long double)s->tensor_core->dims[0] * s->tensor_core->dims[1] *
            s->tensor_core->dims[2] / s->tensor_core->threads;
  for (int i = 0; i < s->n_rngs; i++) {
    PolyAxisType t = s->types[i];
    if (t == POLY_AXIS_UPCAST || t == POLY_AXIS_UNROLL || t == POLY_AXIS_WARP ||
        t == POLY_AXIS_LOCAL || t == POLY_AXIS_GROUP_REDUCE)
      symbolic |= s->shape[i] <= 0;
    if (t == POLY_AXIS_UPCAST || t == POLY_AXIS_UNROLL)
      up *= s->shape[i];
    else if (t == POLY_AXIS_WARP || t == POLY_AXIS_LOCAL || t == POLY_AXIS_GROUP_REDUCE)
      local *= s->shape[i];
  }
  if (symbolic) {
    /* search.get_kernel_actions uses Python integer products and strict UOp
     * truth. Unknown resource bounds are not zero-sized candidates. */
    PolyUOp *up_size = poly_uop0(s->ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(1));
    PolyUOp *local_size = up_size;
    for (int i = 0; i < s->n_rngs; i++) {
      PolyAxisType t = s->types[i];
      PolyUOp **size = t == POLY_AXIS_UPCAST || t == POLY_AXIS_UNROLL ? &up_size
                       : t == POLY_AXIS_WARP || t == POLY_AXIS_LOCAL || t == POLY_AXIS_GROUP_REDUCE
                           ? &local_size
                           : NULL;
      if (size)
        *size = poly_alu2(
            s->ctx, POLY_OP_MUL, *size, poly_cast(s->ctx, s->rngs[i]->src[0], POLY_WEAKINT)
        );
    }
    up_size = poly_alu2(
        s->ctx, POLY_OP_FLOORDIV, up_size,
        poly_uop0(s->ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int((int64_t)tc_up))
    );
    int admitted =
        sched_eval_lt(s->ctx, up_size, (int64_t)poly_getenv_int("BEAM_UPCAST_MAX", 256) + 1);
    if (admitted == 1)
      admitted =
          sched_eval_lt(s->ctx, local_size, (int64_t)poly_getenv_int("BEAM_LOCAL_MAX", 1024) + 1);
    if (admitted < 0) s->failed = true;
    return admitted == 1;
  }
  return floorl(up / tc_up) <= poly_getenv_int("BEAM_UPCAST_MAX", 256) &&
         local <= poly_getenv_int("BEAM_LOCAL_MAX", 1024);
}

/* Time a single kernel execution using clock_gettime (CLOCK_MONOTONIC). */
#ifdef POLY_TESTING
int poly_test_beam_actions(PolyOpt *out, int capacity) {
  static _Thread_local BeamActions actions;
  beam_actions(&actions);
  int count = capacity < actions.count ? capacity : actions.count;
  if (count > 0) memcpy(out, actions.opts, (size_t)count * sizeof(*out));
  return actions.count;
}

PolyUOp *poly_test_apply_opt(PolyCtx *ctx, PolyUOp *sink, PolyRendererCaps caps, PolyOpt opt) {
  OptScheduler s;
  sched_init(&s, ctx, sink);
  PolyUOp *out = sched_apply_opt(&s, caps, opt, NULL) ? s.ast : NULL;
  sched_destroy(&s);
  return out;
}

int poly_test_beam_kernel_action(PolyCtx *ctx, PolyUOp *sink, PolyRendererCaps caps, PolyOpt opt) {
  OptScheduler s;
  sched_init(&s, ctx, sink);
  BeamActions actions = {.opts = {opt}, .count = 1};
  bool admitted = beam_get_kernel_action(&s, caps, opt, &actions);
  int result = s.failed ? -1 : admitted;
  sched_destroy(&s);
  return result;
}

bool poly_test_scheduler_copy_rollback(PolyCtx *ctx, PolyUOp *sink, PolyOpt opt) {
  OptScheduler parent;
  sched_init(&parent, ctx, sink);
  bool ok = !parent.failed;
  for (int fail = 0; ok && fail < 6; fail++) {
    OptScheduler copy;
    sched_copy(&copy, &parent);
    sched_alloc_fail_after = fail;
    bool applied = sched_apply_opt(&copy, (PolyRendererCaps){.device = "CPU"}, opt, NULL);
    sched_alloc_fail_after = -1;
    ok &= !applied && copy.ast == parent.ast && copy.rngs == parent.rngs &&
          copy.opt_range_next == parent.opt_range_next && !copy.failed;
    ok &= sched_apply_opt(&copy, (PolyRendererCaps){.device = "CPU"}, opt, NULL) &&
          copy.ast != parent.ast && copy.rngs != parent.rngs && parent.ast == sink;
    sched_destroy(&copy);
  }
  sched_destroy(&parent);
  return ok;
}

bool poly_test_scheduler_copy_lifetime(PolyCtx *ctx, PolyUOp *sink) {
  OptScheduler parent, copies[4];
  sched_init(&parent, ctx, sink);
  if (!sched_can_optimize(&parent)) {
    sched_destroy(&parent);
    return false;
  }
  for (int i = 0; i < 4; i++)
    sched_copy(&copies[i], &parent);
  bool ok = *parent.references == 5;
  PolyRendererCaps caps = {.device = "CPU"};
  PolyOpt split = {
      .op = POLY_OPT_UPCAST, .has_axis = true, .axis = 0, .arg_kind = POLY_OPT_ARG_INT, .arg = 4};
  /* Metadata-preserving success, rejected mutation, then detached success. */
  ok &= sched_apply_opt(&copies[0], caps, (PolyOpt){.op = POLY_OPT_NOLOCALS}, NULL);
  PolyOpt invalid = split;
  invalid.arg = 3;
  ok &= !sched_apply_opt(&copies[1], caps, invalid, NULL);
  ok &= sched_apply_opt(&copies[2], caps, split, NULL);
  ok &= copies[0].references == parent.references && copies[1].references == parent.references &&
        copies[2].references != parent.references && *parent.references == 4;
  sched_destroy(&parent);
  ok &= *copies[0].references == 3;
  sched_destroy(&copies[1]);
  sched_destroy(&copies[3]);
  ok &= *copies[0].references == 1 && copies[0].shape[0] == 8 &&
        copies[0].types[0] == POLY_AXIS_GLOBAL;
  OptScheduler last;
  sched_copy(&last, &copies[0]);
  sched_destroy(&copies[0]);
  ok &= sched_apply_opt(&last, caps, split, NULL) && last.shape[0] == 2 &&
        copies[2].shape[0] == 2 && last.references != copies[2].references;
  sched_destroy(&last);
  sched_destroy(&copies[2]);
  return ok;
}

bool poly_test_scheduler_reaches(PolyCtx *ctx, PolyUOp *sink, PolyUOp *index, PolyUOp *range) {
  OptScheduler s;
  sched_init(&s, ctx, sink);
  bool found = false;
  for (int b = 0; !s.failed && b < s.n_bufs; b++)
    for (int r = 0; r < s.n_rngs; r++)
      if (s.bufs[b] == index && s.rngs[r] == range) found = sched_buf_reaches(&s, b, r);
  sched_destroy(&s);
  return found;
}
#endif

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
      int64_t lo = 0, hi = 0;
      if (!arg->has_minmax || !poly_arg_integer_to_i64(arg->min_val, &lo) ||
          !poly_arg_integer_to_i64(arg->max_val, &hi) || lo > hi ||
          (!poly_dtype_is_int(param->dtype) && !poly_dtype_eq(param->dtype, POLY_BOOL)))
        goto cleanup;
      /* Python's (lo+hi)//2, without overflowing the C sum. */
      int64_t value = lo + (int64_t)(((uint64_t)hi - (uint64_t)lo) / 2);
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

/* Uncached selected-backend compilation and minimum waited time, in us. */
#ifdef POLY_TESTING
static int beam_test_parameter_count;
static int beam_test_device;
int poly_test_beam_last_device(void) {
  return beam_test_device;
}
#endif
static PolyDevice beam_device(PolyRewriteOpts opts) {
  if (opts.device > POLY_DEVICE_AUTO) return (PolyDevice)opts.device;
  if (opts.caps.device && !strcmp(opts.caps.device, "PYTHON")) return POLY_DEVICE_INTERP;
  PolyDevice device = opts.caps.device ? poly_device_by_name(opts.caps.device) : POLY_DEVICE_CPU;
  return device;
}

/* search._try_compile: use the same PROGRAM builder as normal execution.
 * A tagged optimized SINK suppresses recursive apply_opts, not required lowering. */
static int beam_prepare_candidate_impl(
    PolyCtx *ctx,
    PolyUOp *sink,
    PolyRewriteOpts opts,
    PolyRunner *runner
) {
  sink = poly_graph_rewrite(ctx, sink, poly_pm_flatten_range());
  if (!sink) return -1;
  PolyKernelInfo info = sink->arg.kind == POLY_ARG_KERNEL_INFO && sink->arg.kernel_info
                            ? *sink->arg.kernel_info
                            : (PolyKernelInfo){0};
  info.name = "test";
  info.beam = 0;
  sink = poly_uop_tagged(
      ctx, sink->op, sink->dtype, sink->src, sink->n_src, poly_arg_kernel_info(&info), 1
  );
  int n = 0;
  PolyUOp **topo = poly_toposort_alloc(ctx, sink, &n);
  PolyUOp **params = topo ? calloc((size_t)n + 1, sizeof(*params)) : NULL;
  if (!params) {
    poly_toposort_free(topo);
    return -1;
  }
  int n_params = 0;
  for (int i = 0; i < n; i++) {
    PolyUOp *u = topo[i];
    if (u->op != POLY_OP_PARAM || poly_uop_is_alu_param(u)) continue;
    int slot = poly_program_buffer_slot(u);
    if (slot < 0 || slot >= n || (params[slot + 1] && params[slot + 1] != u)) goto failed;
    params[slot + 1] = u;
    if (slot >= n_params) n_params = slot + 1;
  }
  for (int i = 1; i <= n_params; i++)
    if (!params[i]) goto failed;
  params[0] = sink;
  PolyUOp *call = poly_uop(ctx, POLY_OP_CALL, POLY_VOID, params, n_params + 1, poly_arg_none());
  free(params);
  poly_toposort_free(topo);
  return poly_time_call_prepare(ctx, call, beam_device(opts), runner);
failed:
  free(params);
  poly_toposort_free(topo);
  return -1;
}

static int beam_prepare_candidate(
    PolyCtx *ctx,
    PolyUOp *sink,
    PolyRewriteOpts opts,
    PolyRunner *runner
) {
  double previous = poly_compile_deadline_ms;
  int seconds = poly_getenv_int("BEAM_TIMEOUT_SEC", 10);
  if (seconds > 0) {
    double deadline = poly_now_ms() + (double)seconds * 1000;
    poly_compile_deadline_ms = previous > 0 && previous < deadline ? previous : deadline;
  }
  int rc = beam_prepare_candidate_impl(ctx, sink, opts, runner);
  bool timed_out = poly_compile_timed_out();
  poly_compile_deadline_ms = previous;
  if (timed_out) {
    if (rc == 0) poly_time_call_finish(ctx, runner, beam_device(opts));
    if (poly_debug_at_least(2)) fprintf(stderr, "polygrad: BEAM compile timeout\n");
    return -2;
  }
  return rc;
}

/* args_from_ast/_ensure_buffer_alloc: one raw-buffer set for the whole search.
 * Candidate ProgramInfo may eliminate or reorder globals; bind by original slot. */
typedef struct {
  void **host;
  int n_host;
  PolyBuffer *buffers;
  int *slots;
  int count;
  const PolyAllocator *allocator;
} BeamBuffers;

static void beam_buffers_free(BeamBuffers *raw) {
  if (raw->buffers && raw->allocator && !raw->allocator->host_addressable)
    for (int i = 0; i < raw->count; i++)
      if (raw->buffers[i].ptr) raw->allocator->free(&raw->buffers[i], raw->allocator->dev_ctx);
  free(raw->buffers);
  free(raw->slots);
  beam_free_args(raw->host, raw->n_host);
  *raw = (BeamBuffers){0};
}

static bool beam_buffers_init(PolyCtx *ctx, PolyUOp *sink, PolyDevice device, BeamBuffers *raw) {
  *raw = (BeamBuffers){0};
  const PolyBackendDesc *backend = poly_backend_get(device);
  if (!backend || poly_backend_ensure_open(device) != 0 || !backend->get_allocator) return false;
  raw->allocator = backend->get_allocator();
  raw->host = beam_args_from_ast(ctx, sink, &raw->n_host);
  int n = 0;
  PolyUOp **topo = poly_toposort_alloc(ctx, sink, &n);
  if (!raw->host || !raw->allocator || !topo) goto failed;
  for (int i = 0; i < n; i++)
    if (topo[i]->op == POLY_OP_PARAM && !poly_uop_is_alu_param(topo[i]))
      topo[raw->count++] = topo[i];
  for (int i = 1; i < raw->count; i++) {
    PolyUOp *param = topo[i];
    int j = i;
    while (j && poly_program_buffer_slot(topo[j - 1]) > poly_program_buffer_slot(param)) {
      topo[j] = topo[j - 1];
      j--;
    }
    topo[j] = param;
  }
  if (raw->count > raw->n_host) goto failed;
  raw->buffers = calloc((size_t)(raw->count ? raw->count : 1), sizeof(*raw->buffers));
  raw->slots = calloc((size_t)(raw->count ? raw->count : 1), sizeof(*raw->slots));
  if (!raw->buffers || !raw->slots) goto failed;
  for (int i = 0; i < raw->count; i++) {
    int itemsize = poly_dtype_itemsize(topo[i]->dtype);
    int64_t count = poly_uop_max_numel(ctx, topo[i]);
    raw->slots[i] = poly_program_buffer_slot(topo[i]);
    if (raw->slots[i] < 0 || (i && raw->slots[i] == raw->slots[i - 1]) || count < 0 ||
        itemsize <= 0 || (uint64_t)(count ? count : 1) > SIZE_MAX / (size_t)itemsize)
      goto failed;
    size_t nbytes = (size_t)(count ? count : 1) * (size_t)itemsize;
    raw->buffers[i] = (PolyBuffer
    ){.ptr = raw->host[i],
      .nbytes = nbytes,
      .device = device,
      .allocator = raw->allocator,
      .valid = true};
    if (raw->allocator->host_addressable) continue;
    raw->buffers[i].ptr = raw->allocator->alloc(nbytes, raw->allocator->dev_ctx);
    if (!raw->buffers[i].ptr) goto failed;
    raw->buffers[i].owned = true;
    PolyBuffer host = {
        .ptr = raw->host[i], .nbytes = nbytes, .device = POLY_DEVICE_CPU, .valid = true};
    if (!raw->allocator->copy_in ||
        raw->allocator->copy_in(&raw->buffers[i], &host, nbytes, raw->allocator->dev_ctx) != 0)
      goto failed;
  }
  poly_toposort_free(topo);
  return true;
failed:
  poly_toposort_free(topo);
  beam_buffers_free(raw);
  return false;
}

static double beam_time_candidate(
    PolyCtx *ctx,
    PolyRunner *runner,
    PolyDevice device,
    const BeamBuffers *raw,
    int reps,
    double early_stop_us,
    int max_global_size
) {
  int n_args = runner->n_params + runner->n_vars;
  void **args = calloc((size_t)(n_args ? n_args : 1), sizeof(*args));
  PolyVarBinding *bindings =
      calloc((size_t)(runner->n_vars ? runner->n_vars : 1), sizeof(*bindings));
  int *values = calloc((size_t)(runner->n_vars ? runner->n_vars : 1), sizeof(*values));
  const PolyProgramInfo *info = poly_program_info(ctx, runner->program);
  double elapsed = INFINITY;
  if (!args || !bindings || !values || !info) goto cleanup;
  for (int i = 0; i < runner->n_params; i++) {
    for (int j = 0; j < raw->count; j++)
      if (raw->slots[j] == info->globals[i]) {
        args[i] = raw->buffers[j].ptr;
        break;
      }
    if (!args[i]) goto cleanup;
  }
  for (int i = 0; i < runner->n_vars; i++) {
    PolyUOp *var = info->vars[runner->var_indices[i]];
    int64_t lo, hi;
    poly_uop_minmax(ctx, var, &lo, &hi);
    if (lo > hi) goto cleanup;
    int64_t value = lo + (int64_t)(((uint64_t)hi - (uint64_t)lo) / 2);
    if (value < INT_MIN || value > INT_MAX) goto cleanup;
    values[i] = (int)value;
    bindings[i] = (PolyVarBinding){.var = var, .value = value};
    args[runner->n_params + i] = &values[i];
  }
#ifdef POLY_TESTING
  beam_test_parameter_count = n_args;
  beam_test_device = device;
#endif
  elapsed = poly_time_call(
      runner, device, args, n_args, bindings, runner->n_vars, reps, early_stop_us, max_global_size
  );
cleanup:
  free(bindings);
  free(values);
  free(args);
  return elapsed;
}

static double beam_compile_and_time(PolyCtx *ctx, PolyUOp *sink, PolyRewriteOpts opts, int reps) {
  PolyRunner runner;
  if (beam_prepare_candidate(ctx, sink, opts, &runner) != 0) return INFINITY;
  PolyDevice device = beam_device(opts);
  BeamBuffers raw;
  double elapsed = INFINITY;
  if (beam_buffers_init(ctx, runner.program, device, &raw)) {
    elapsed = beam_time_candidate(ctx, &runner, device, &raw, reps, INFINITY, 65536);
    beam_buffers_free(&raw);
  }
  poly_time_call_finish(ctx, &runner, device);
  return elapsed;
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
/* Cache identity reuses the existing UOp encoder on the pre-lowering SINK. These bytes are
 * compared, never imported or executed: cache values are only Opt sequences. */
typedef struct {
  uint8_t *data;
  size_t size;
  uint64_t hash;
} BeamCacheKey;

static bool beam_key_append(BeamCacheKey *key, const void *data, size_t size) {
  if (size > SIZE_MAX - key->size) return false;
  uint8_t *next = realloc(key->data, key->size + size);
  if (!next && size) return false;
  key->data = next;
  if (size) memcpy(next + key->size, data, size);
  key->size += size;
  return true;
}

static bool beam_key_string(BeamCacheKey *key, const char *text) {
  return beam_key_append(key, text ? text : "", strlen(text ? text : "") + 1);
}

static bool beam_cache_key(
    PolyCtx *ctx,
    PolyUOp *sink,
    int width,
    PolyRewriteOpts opts,
    BeamCacheKey *key
) {
  *key = (BeamCacheKey){0};
  PolyIrEntrypoint entry = {.name = "beam", .sink = sink};
  PolyIrSpec spec = {.ctx = ctx, .entrypoints = &entry, .n_entrypoints = 1};
  int length = 0;
  key->data = poly_ir_export(&spec, &length);
  if (!key->data || length <= 0) goto failed;
  key->size = (size_t)length;
  PolyRendererCaps c = opts.caps;
  int fields[] = {
      POLYGRAD_ABI_VERSION,
      width,
      poly_getenv_flag_default("BEAM_ESTIMATE", true),
      beam_device(opts),
      c.has_mulacc,
      c.has_max,
      c.has_threefry,
      c.has_exp2,
      c.has_log2,
      c.has_sin,
      c.has_fdiv,
      c.supports_float16,
      c.supports_bfloat16,
      c.supports_fp8e4m3,
      c.supports_fp8e5m2,
      c.supports_fp8e4m3fnuz,
      c.supports_fp8e5m2fnuz,
      c.has_int64,
      c.has_local,
      c.has_threads,
      c.has_simd_int,
      c.has_simd_float,
      c.max_vec_width,
      c.max_threads,
      c.global_max[0],
      c.global_max[1],
      c.global_max[2],
      c.local_max[0],
      c.local_max[1],
      c.local_max[2],
      c.shared_max,
      c.n_tensor_cores};
  if (!beam_key_append(key, fields, sizeof(fields)) || !beam_key_string(key, c.device) ||
      !beam_key_string(key, c.arch))
    goto failed;
  const char *envs[] = {"CC", "POLY_OPT", "POLY_CPU_ARCH"};
  for (size_t i = 0; i < sizeof(envs) / sizeof(envs[0]); i++)
    if (!beam_key_string(key, getenv(envs[i]))) goto failed;
  for (int i = 0; i < c.n_tensor_cores; i++) {
    const PolyTensorCore *tc = &c.tensor_cores[i];
    int tc_fields[] = {
        tc->dims[0],
        tc->dims[1],
        tc->dims[2],
        tc->threads,
        tc->elements_per_thread[0],
        tc->elements_per_thread[1],
        tc->elements_per_thread[2],
        tc->dtype_in.priority,
        tc->dtype_in.bitsize,
        tc->dtype_out.priority,
        tc->dtype_out.bitsize,
        tc->n_opts};
    if (!beam_key_append(key, tc_fields, sizeof(tc_fields))) goto failed;
    for (int j = 0; j < tc->n_opts; j++) {
      int pair[] = {tc->opts[j].type, tc->opts[j].dim};
      if (!beam_key_append(key, pair, sizeof(pair))) goto failed;
    }
    for (int a = 0; a < 2; a++)
      for (int b = 0; b < 3; b++) {
        if (!beam_key_append(key, &tc->swizzle_len[a][b], sizeof(int))) goto failed;
        for (int j = 0; j < tc->swizzle_len[a][b]; j++)
          if (!beam_key_string(key, tc->swizzle[a][b][j])) goto failed;
      }
  }
  key->hash = UINT64_C(0xcbf29ce484222325);
  for (size_t i = 0; i < key->size; i++) {
    key->hash ^= key->data[i];
    key->hash *= UINT64_C(0x100000001b3);
  }
  return true;
failed:
  free(key->data);
  *key = (BeamCacheKey){0};
  return false;
}

static int beam_cache_path(const BeamCacheKey *key, char *path, size_t capacity) {
  const char *xdg = getenv("XDG_CACHE_HOME"), *home = getenv("HOME");
  char dir[512];
  int n = xdg && *xdg     ? snprintf(dir, sizeof(dir), "%s/polygrad/beam", xdg)
          : home && *home ? snprintf(dir, sizeof(dir), "%s/.cache/polygrad/beam", home)
                          : -1;
  if (n < 0 || (size_t)n >= sizeof(dir)) return -1;
  for (char *p = dir + 1; *p; p++)
    if (*p == '/') {
      *p = '\0';
      int rc = mkdir(dir, 0755);
      *p = '/';
      if (rc != 0 && errno != EEXIST) return -1;
    }
  if (mkdir(dir, 0755) != 0 && errno != EEXIST) return -1;
  n = snprintf(path, capacity, "%s/v2-%016llx.bin", dir, (unsigned long long)key->hash);
  return n < 0 || (size_t)n >= capacity ? -1 : 0;
}

typedef struct {
  PolyOpt opt;
  int64_t tuple[3];
} BeamCachedOpt;

/* Local cache format, not an artifact ABI. Endianness/version and exact key
 * bytes are checked before any Opt is applied. Replays publish all or nothing. */
static bool beam_cache_load(const BeamCacheKey *key, OptScheduler *s, PolyRendererCaps caps) {
  char path[600];
  if (!key->data || beam_cache_path(key, path, sizeof(path)) != 0) return false;
  FILE *f = fopen(path, "rb");
  if (!f) return false;
  uint32_t header[5];
  BeamCachedOpt *cached = NULL;
  uint8_t *stored_key = NULL;
  OptScheduler candidate = {0};
  bool ok = false;
  if (fread(header, sizeof(header), 1, f) != 1 || header[0] != UINT32_C(0x50474232) ||
      header[1] != POLYGRAD_ABI_VERSION || header[2] != UINT32_C(0x01020304) ||
      header[3] != key->size || header[4] > INT_MAX)
    goto done;
  long payload = ftell(f);
  if (payload < 0 || fseek(f, 0, SEEK_END) != 0) goto done;
  long end = ftell(f);
  uint64_t expected = (uint64_t)payload + key->size + (uint64_t)header[4] * 8 * sizeof(int64_t);
  if (end < 0 || (uint64_t)end != expected || fseek(f, payload, SEEK_SET) != 0) goto done;
  stored_key = malloc(key->size);
  cached = calloc(header[4] ? header[4] : 1, sizeof(*cached));
  if (!stored_key || !cached || fread(stored_key, key->size, 1, f) != 1 ||
      memcmp(stored_key, key->data, key->size))
    goto done;
  for (uint32_t i = 0; i < header[4]; i++) {
    int64_t v[8];
    if (fread(v, sizeof(v), 1, f) != 1 || v[0] < POLY_OPT_TC || v[0] > POLY_OPT_SWAP ||
        (v[1] != 0 && v[1] != 1) || v[2] < 0 || v[2] > INT_MAX || v[3] < POLY_OPT_ARG_NONE ||
        v[3] > POLY_OPT_ARG_INT_TUPLE)
      goto done;
    cached[i].opt = (PolyOpt
    ){.op = (PolyOptOps)v[0],
      .has_axis = v[1],
      .axis = (int)v[2],
      .arg_kind = (int)v[3],
      .arg = v[4]};
    if (v[3] == POLY_OPT_ARG_INT_TUPLE) {
      memcpy(cached[i].tuple, &v[5], sizeof(cached[i].tuple));
      cached[i].opt.arg_tuple = cached[i].tuple;
      cached[i].opt.n_arg_tuple = 3;
    }
  }
  sched_copy(&candidate, s);
  int prefix = s->ast->arg.kind == POLY_ARG_KERNEL_INFO && s->ast->arg.kernel_info
                   ? s->ast->arg.kernel_info->n_applied_opts
                   : 0;
  if ((uint32_t)prefix > header[4]) goto done;
  for (uint32_t i = (uint32_t)prefix; i < header[4]; i++)
    if (!sched_apply_opt(&candidate, caps, cached[i].opt, NULL)) goto done;
  sched_destroy(s);
  sched_copy(s, &candidate);
  ok = true;
done:
  sched_destroy(&candidate);
  free(cached);
  free(stored_key);
  fclose(f);
  return ok;
}

static void beam_cache_save(const BeamCacheKey *key, PolyUOp *sink) {
  if (!key->data || key->size > UINT32_MAX) return;
  const PolyKernelInfo *info =
      sink->arg.kind == POLY_ARG_KERNEL_INFO ? sink->arg.kernel_info : NULL;
  int count = info ? info->n_applied_opts : 0;
  char path[600], temporary[620];
  if (beam_cache_path(key, path, sizeof(path)) != 0) return;
  int n = snprintf(temporary, sizeof(temporary), "%s.XXXXXX", path);
  if (n < 0 || (size_t)n >= sizeof(temporary)) return;
  int fd = mkstemp(temporary);
  if (fd < 0) return;
  FILE *f = fdopen(fd, "wb");
  if (!f) {
    close(fd);
    remove(temporary);
    return;
  }
  uint32_t header[] = {
      UINT32_C(0x50474232), POLYGRAD_ABI_VERSION, UINT32_C(0x01020304), (uint32_t)key->size,
      (uint32_t)count};
  bool ok = fwrite(header, sizeof(header), 1, f) == 1 && fwrite(key->data, key->size, 1, f) == 1;
  for (int i = 0; ok && i < count; i++) {
    PolyOpt a = info->applied_opts[i];
    int64_t v[] = {a.op, a.has_axis, a.axis, a.arg_kind, a.arg, 0, 0, 0};
    if (a.arg_kind == POLY_OPT_ARG_INT_TUPLE) {
      if (a.n_arg_tuple != 3 || !a.arg_tuple) {
        ok = false;
        break;
      }
      memcpy(&v[5], a.arg_tuple, 3 * sizeof(int64_t));
    }
    ok = fwrite(v, sizeof(v), 1, f) == 1;
  }
  if (fclose(f) != 0) ok = false;
  if (!ok || rename(temporary, path) != 0) remove(temporary);
}

#ifdef POLY_TESTING
PolyUOp *poly_test_convert_loop_to_global(PolyCtx *ctx, PolyUOp *sink) {
  return convert_loop_to_global(ctx, sink);
}

uint64_t poly_test_beam_cache_key(PolyCtx *ctx, PolyUOp *sink, int width, PolyDevice device) {
  BeamCacheKey key;
  PolyRewriteOpts opts = {.device = device, .caps = {.device = poly_device_name(device)}};
  if (!beam_cache_key(ctx, sink, width, opts, &key)) return 0;
  uint64_t hash = key.hash;
  free(key.data);
  return hash;
}

int poly_test_beam_cache_write(
    PolyCtx *ctx,
    PolyUOp *sink,
    PolyUOp *result,
    int width,
    char *path,
    size_t capacity
) {
  BeamCacheKey key;
  PolyRewriteOpts opts = {.device = POLY_DEVICE_CPU, .caps = {.device = "CPU"}};
  if (!beam_cache_key(ctx, sink, width, opts, &key)) return -1;
  beam_cache_save(&key, result);
  int rc = beam_cache_path(&key, path, capacity);
  free(key.data);
  return rc;
}

PolyUOp *poly_test_beam_cache_read(PolyCtx *ctx, PolyUOp *sink, int width) {
  BeamCacheKey key;
  PolyRewriteOpts opts = {.device = POLY_DEVICE_CPU, .caps = {.device = "CPU"}};
  if (!beam_cache_key(ctx, sink, width, opts, &key)) return NULL;
  OptScheduler s;
  sched_init(&s, ctx, sink);
  bool ok = beam_cache_load(&key, &s, opts.caps);
  free(key.data);
  PolyUOp *out = ok ? s.ast : NULL;
  sched_destroy(&s);
  return out;
}
#endif

static int beam_candidate_cmp(const void *a, const void *b) {
  const BeamEntry *x = a, *y = b;
  return x->time_us < y->time_us   ? -1
         : x->time_us > y->time_us ? 1
         : x->order < y->order     ? -1
                                   : x->order > y->order;
}

static bool beam_compute_ops(PolyCtx *ctx, PolyUOp *program, uint64_t *ops) {
  *ops = 0;
  PolyUOp *sink = program->src[0];
  if (sink->arg.kind != POLY_ARG_KERNEL_INFO || !sink->arg.kernel_info ||
      !sink->arg.kernel_info->estimates)
    return true;
  const PolyProgramInfo *info = poly_program_info(ctx, program);
  PolyVarBinding *vars = calloc((size_t)(info->n_vars ? info->n_vars : 1), sizeof(*vars));
  if (!vars) return false;
  for (int i = 0; i < info->n_vars; i++) {
    int64_t lo, hi;
    poly_uop_minmax(ctx, info->vars[i], &lo, &hi);
    if (lo > hi) {
      free(vars);
      return false;
    }
    vars[i] = (PolyVarBinding
    ){.var = info->vars[i], .value = lo + (int64_t)(((uint64_t)hi - (uint64_t)lo) / 2)};
  }
  uint64_t lds, mem;
  bool ok =
      poly_estimates_infer(sink->arg.kernel_info->estimates, vars, info->n_vars, ops, &lds, &mem) ==
      0;
  free(vars);
  return ok;
}

/* search.py diagnostics use the existing C UOp printer, once per DAG node.
 * No graph traversal, formatting allocation or diagnostic timing when disabled. */
static void beam_debug_graph(PolyCtx *ctx, PolyUOp *sink) {
  int n = 0;
  PolyUOp **topo = poly_toposort_alloc(ctx, sink, &n);
  for (int i = 0; i < n; i++) {
    fprintf(stderr, "%p <-", (void *)topo[i]);
    for (int j = 0; j < topo[i]->n_src; j++)
      fprintf(stderr, " %p", (void *)topo[i]->src[j]);
    fputc(' ', stderr);
    poly_uop_dump_tree(stderr, topo[i], 0, 0);
  }
  poly_toposort_free(topo);
}

static void beam_debug_opts(const OptScheduler *s) {
  static const char *names[] = {"?",     "TC",       "UPCAST",   "UNROLL", "LOCAL", "THREAD",
                                "GROUP", "GROUPTOP", "NOLOCALS", "PADTO",  "SWAP"};
  const PolyKernelInfo *info =
      s->ast->arg.kind == POLY_ARG_KERNEL_INFO ? s->ast->arg.kernel_info : NULL;
  fputc('[', stderr);
  for (int i = 0; info && i < info->n_applied_opts; i++) {
    PolyOpt o = info->applied_opts[i];
    fprintf(
        stderr, "%sOpt(%s", i ? ", " : "",
        o.op >= POLY_OPT_TC && o.op <= POLY_OPT_SWAP ? names[o.op] : "?"
    );
    if (o.has_axis) fprintf(stderr, ",axis=%d", o.axis);
    if (o.arg_kind == POLY_OPT_ARG_INT)
      fprintf(stderr, ",arg=%lld", (long long)o.arg);
    else if (o.arg_kind == POLY_OPT_ARG_INT_TUPLE) {
      fputs(",arg=(", stderr);
      for (int j = 0; j < o.n_arg_tuple; j++)
        fprintf(stderr, "%s%lld", j ? "," : "", (long long)o.arg_tuple[j]);
      fputc(')', stderr);
    }
    fputc(')', stderr);
  }
  fputc(']', stderr);
}

static PolyUOp *poly_beam_search(PolyCtx *ctx, PolyUOp *sink, int width, PolyRewriteOpts opts) {
  if (width <= 0) return sink;
  int beam_debug = poly_getenv_int("BEAM_DEBUG", 0);
  bool search_started = false;
  BeamActions actions;
  beam_actions(&actions);
  if ((size_t)width > SIZE_MAX / sizeof(BeamEntry) / (size_t)actions.count ||
      width > INT_MAX / actions.count)
    return NULL;
  BeamEntry *beam = calloc((size_t)width, sizeof(*beam));
  BeamEntry *candidates = calloc((size_t)width * (size_t)actions.count, sizeof(*candidates));
  PolyUOp **seen = NULL;
  PolyUOp **candidate_roots =
      calloc((size_t)width * (size_t)actions.count, sizeof(*candidate_roots));
  int n_candidate_roots = 0, size = 1, count = 0;
  bool fatal = false;
  BeamBuffers raw = {0};
  size_t n_seen = 0, seen_capacity = 0;
  if (!beam || !candidates || !candidate_roots) {
    free(beam);
    free(candidates);
    free(candidate_roots);
    return NULL;
  }
  sched_init(&beam[0].sched, ctx, sink);
  if (poly_uop_retain(ctx, sink) != 0) {
    sched_destroy(&beam[0].sched);
    free(beam);
    free(candidates);
    free(candidate_roots);
    return NULL;
  }
  beam[0].time_us = INFINITY;
  BeamCacheKey key;
  beam_cache_key(ctx, sink, width, opts, &key);
  if (!poly_get_ignore_beam_cache() && poly_getenv_int("CACHELEVEL", 2) >= 1 &&
      beam_cache_load(&key, &beam[0].sched, opts.caps)) {
    if (poly_uop_retain(ctx, beam[0].sched.ast) != 0)
      beam[0].sched.ast = sink;
    else
      poly_uop_release(ctx, sink);
    goto done;
  }
  if (!sched_can_optimize(&beam[0].sched)) goto done;
  search_started = true;
  if (beam_debug) {
    fputs("BEAM_SEARCH:\n", stderr);
    beam_debug_graph(ctx, sink);
  }
  PolyDevice device = beam_device(opts);
  if (!beam_buffers_init(ctx, sink, device, &raw)) goto done;
  double min_progress = 0.01;
  const char *progress = getenv("BEAM_MIN_PROGRESS");
  if (progress) {
    char *end;
    double parsed = strtod(progress, &end);
    if (end != progress && !*end && isfinite(parsed) && parsed >= 0) min_progress = parsed;
  }
  for (;;) {
    for (int i = 0; i < count; i++)
      sched_destroy(&candidates[i].sched);
    for (int i = 0; i < n_candidate_roots; i++)
      poly_uop_release(ctx, candidate_roots[i]);
    n_candidate_roots = 0;
    count = 0;
    for (int b = 0; b < size; b++)
      for (int a = 0; a < actions.count; a++) {
        OptScheduler copy;
        sched_copy(&copy, &beam[b].sched);
        if (!beam_get_kernel_action(&copy, opts.caps, actions.opts[a], &actions)) {
          bool failed = copy.failed;
          sched_destroy(&copy);
          if (failed) {
            fatal = true;
            goto done;
          }
          continue;
        }
        if (poly_uop_retain(ctx, copy.ast) != 0) {
          sched_destroy(&copy);
          goto done;
        }
        candidate_roots[n_candidate_roots++] = copy.ast;
        candidates[count] = (BeamEntry){.sched = copy, .time_us = INFINITY, .order = count};
        count++;
      }
    if (!count) break;
    uint64_t least = UINT64_MAX;
    int timed = 0;
    for (int i = 0; i < count; i++) {
      PolyRunner runner;
      double compile_start = beam_debug > 1 ? poly_now_ms() : 0;
      int prepared = beam_prepare_candidate(ctx, candidates[i].sched.ast, opts, &runner);
      double compile_ms = beam_debug > 1 ? poly_now_ms() - compile_start : 0;
      if (prepared == -2 && poly_getenv_flag("BEAM_STRICT_MODE")) {
        fatal = true;
        goto done;
      }
      if (prepared != 0) {
        if (beam_debug) {
          fprintf(stderr, "BEAM rejected compile status=%d opts=", prepared);
          beam_debug_opts(&candidates[i].sched);
          fputc('\n', stderr);
        }
        continue;
      }
      PolyUOp *program = runner.program;
      PolyUOp *lib = runner.compiled_binary ? runner.compiled_binary
                     : program->n_src >= 4  ? program->src[3]
                     : program->n_src >= 3  ? program->src[2]
                                            : poly_program_linear(program);
      bool duplicate = false;
      for (size_t j = 0; j < n_seen; j++)
        if (seen[j] == lib) {
          duplicate = true;
          break;
        }
      uint64_t compute;
      if (!lib || duplicate || !beam_compute_ops(ctx, program, &compute)) {
        poly_time_call_finish(ctx, &runner, device);
        continue;
      }
      if (compute < least) least = compute;
      if ((long double)compute > (long double)least * 1000) {
        poly_time_call_finish(ctx, &runner, device);
        continue;
      }
      if (n_seen == seen_capacity) {
        size_t next = seen_capacity ? seen_capacity * 2 : 64;
        PolyUOp **grown = next > seen_capacity && next <= SIZE_MAX / sizeof(*seen)
                              ? realloc(seen, next * sizeof(*seen))
                              : NULL;
        if (!grown) {
          poly_time_call_finish(ctx, &runner, device);
          goto done;
        }
        seen = grown;
        seen_capacity = next;
      }
      seen[n_seen++] = lib;
      poly_uop_retain(ctx, lib);
      double time = beam_time_candidate(
          ctx, &runner, device, &raw, 3, beam[0].time_us * 3,
          poly_getenv_flag_default("BEAM_ESTIMATE", true) ? 65536 : 0
      );
      if (beam_debug > 1) {
        PolyUOp *linear = poly_program_linear(runner.program);
        fprintf(
            stderr, "%d %u uops %.3fms compile/%.3fus run opts=", i,
            linear ? (unsigned)linear->n_src : 0, compile_ms, time
        );
        beam_debug_opts(&candidates[i].sched);
        fputc('\n', stderr);
      } else if (beam_debug && !isfinite(time)) {
        fputs("BEAM failed timing opts=", stderr);
        beam_debug_opts(&candidates[i].sched);
        fputc('\n', stderr);
      }
      poly_time_call_finish(ctx, &runner, device);
      if (!isfinite(time)) continue;
      candidates[i].time_us = time;
      timed++;
    }
    if (!timed) break;
    qsort(candidates, (size_t)count, sizeof(*candidates), beam_candidate_cmp);
    bool exiting = candidates[0].time_us < min_progress ||
                   beam[0].time_us - candidates[0].time_us < min_progress;
    if (!exiting || candidates[0].time_us < beam[0].time_us) {
      int next_size = exiting ? 1 : timed < width ? timed : width;
      int retained = 0;
      for (; retained < next_size; retained++)
        if (poly_uop_retain(ctx, candidates[retained].sched.ast) != 0) break;
      if (retained != next_size) {
        for (int i = 0; i < retained; i++)
          poly_uop_release(ctx, candidates[i].sched.ast);
        goto done;
      }
      for (int i = 0; i < size; i++) {
        poly_uop_release(ctx, beam[i].sched.ast);
        sched_destroy(&beam[i].sched);
      }
      size = next_size;
      for (int i = 0; i < size; i++) {
        beam[i] = candidates[i];
        sched_copy(&beam[i].sched, &candidates[i].sched);
      }
    }
    if (exiting) break;
  }
  if (poly_getenv_int("CACHELEVEL", 2) >= 1) beam_cache_save(&key, beam[0].sched.ast);
done:
  if (beam_debug && search_started) {
    fprintf(stderr, "BEAM_SEARCH: final tm=%.3fus, applied_opts=", beam[0].time_us);
    beam_debug_opts(&beam[0].sched);
    if (fatal) fputs(" (failed)", stderr);
    fputc('\n', stderr);
  }
  for (int i = 0; i < count; i++)
    sched_destroy(&candidates[i].sched);
  beam_buffers_free(&raw);
  for (int i = 0; i < n_candidate_roots; i++)
    poly_uop_release(ctx, candidate_roots[i]);
  free(candidate_roots);
  for (size_t i = 0; i < n_seen; i++)
    poly_uop_release(ctx, seen[i]);
  free(seen);
  free(key.data);
  PolyUOp *result = fatal ? NULL : beam[0].sched.ast;
  for (int i = 0; i < size; i++) {
    poly_uop_release(ctx, beam[i].sched.ast);
    sched_destroy(&beam[i].sched);
  }
  free(beam);
  free(candidates);
  return result;
}

/* tinygrad/codegen/__init__.py:ReduceContext. Failure is C-only: matcher
 * callbacks use NULL for no rewrite, so allocation failure needs a separate
 * pass result rather than publishing a partially lowered graph. */
typedef struct {
  int acc_num;
  bool failed;
} ReduceContext;

static ReduceContext *current_reduce_ctx(void) {
  return (ReduceContext *)poly_graph_rewrite_userctx();
}

#ifdef POLY_TESTING
static _Thread_local int reduce_alloc_fail_after = -1;
void poly_test_reduce_alloc_fail_after(int count) {
  reduce_alloc_fail_after = count;
}
#endif

static void *reduce_alloc(size_t count, size_t size) {
#ifdef POLY_TESTING
  if (reduce_alloc_fail_after == 0) return NULL;
  if (reduce_alloc_fail_after > 0) --reduce_alloc_fail_after;
#endif
  return calloc(count, size);
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
    for (int j = 0; j < nb; j++)
      if (a[i] == b[j]) {
        found = true;
        break;
      }
    if (!found) return false;
  }
  return true;
}

/* UOp.replace(arg=(axis_id, *r.arg[1:])) preserves dtype, sources and tags. */
static PolyUOp *clone_range_axis(PolyCtx *ctx, PolyUOp *r, int64_t axis_id) {
  if (!r || r->op != POLY_OP_RANGE || r->arg.kind != POLY_ARG_RANGE) return NULL;
  PolyArg arg = poly_arg_range_ex(
      axis_id, poly_range_axis_type(r->arg), poly_range_extra(r->arg), poly_range_n_extra(r->arg)
  );
  return poly_uop_tagged_arg(ctx, r->op, r->dtype, r->src, r->n_src, arg, r->tag, r->tag_arg);
}

/* tinygrad/codegen/__init__.py:reduce_ranges_to_acc. Loop arity is not Tensor
 * rank. Keep the complete range tuple until pm_add_control_flow splits ENDs. */
static PolyUOp *reduce_ranges_to_acc(PolyCtx *ctx, PolyUOp *red, const PolyBindings *b) {
  (void)b;
  ReduceContext *rctx = current_reduce_ctx();
  if (!rctx || rctx->failed || red->arg.kind != POLY_ARG_REDUCE || red->n_src < 2) return NULL;
  int n_topo = 0;
  PolyUOp **topo = poly_toposort_alloc(ctx, red->src[0], &n_topo);
  PolyUOp **src = NULL, *out = NULL;
  PolyMap *excluded = NULL;
  if (!topo || rctx->acc_num == INT_MAX || red->n_src == UINT16_MAX) goto fail;
  size_t capacity = (size_t)n_topo + 1;
  if (capacity < (size_t)red->n_src + 1) capacity = (size_t)red->n_src + 1;
  src = reduce_alloc(capacity, sizeof(*src));
  excluded = poly_map_new(((size_t)n_topo + red->n_src) * 2);
  if (!src || !excluded) goto fail;
  for (int i = 1; i < red->n_src; i++)
    poly_map_set(excluded, poly_ptr_hash(red->src[i]), red->src[i], red->src[i], poly_ptr_eq);
  for (int i = 0; i < n_topo; i++) {
    if (topo[i]->op != POLY_OP_END) continue;
    /* END.ended_ranges is exactly src[1:], not its active-range closure. */
    for (int j = 1; j < topo[i]->n_src; j++) {
      PolyUOp *r = topo[i]->src[j];
      poly_map_set(excluded, poly_ptr_hash(r), r, r, poly_ptr_eq);
    }
  }
  int n_input = 0;
  for (int i = 0; i < n_topo; i++)
    if (topo[i]->op == POLY_OP_RANGE &&
        !poly_map_get(excluded, poly_ptr_hash(topo[i]), topo[i], poly_ptr_eq))
      src[++n_input] = topo[i];
  if (n_input >= UINT16_MAX) goto fail;

  /* placeholder_like flattens storage, then restores a multidimensional
   * result shape. The accumulator value keeps the strong storage dtype. */
  PolyShape shape = poly_uop_max_shape_cached(ctx, red);
  int64_t size = poly_shape_numel(shape);
  if (size < 0) goto fail;
  PolyDType dtype = poly_dtype_strong(red->dtype);
  PolyUOp *dim = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(size));
  PolyParamArg param = {.slot = rctx->acc_num++, .addrspace = POLY_ADDR_REG};
  PolyUOp *acc = poly_uop1(ctx, POLY_OP_BUFFER, dtype, dim, poly_arg_param(&param));
  if (acc && shape.ndim > 1) acc = poly_reshape(ctx, acc, shape.dims, shape.ndim);
  if (!acc) goto fail;
  src[0] = acc;
  PolyUOp *base =
      n_input ? poly_uop(ctx, POLY_OP_AFTER, dtype, src, n_input + 1, poly_arg_none()) : acc;
  PolyUOp *identity = poly_identity_element(ctx, red->arg.reduce.op, red->dtype);
  PolyUOp *init = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, base, identity, poly_arg_none());
  src[1] = init;
  memcpy(src + 2, red->src + 1, (size_t)(red->n_src - 1) * sizeof(*src));
  PolyUOp *initted = poly_uop(ctx, POLY_OP_AFTER, dtype, src, red->n_src + 1, poly_arg_none());
  PolyUOp *inp =
      red->arg.reduce.num_axes ? poly_expand_horizontal_reduce(ctx, red, NULL) : red->src[0];
  /* UOp.alu promotes the weak input against the strong REG-backed value. */
  PolyUOp *value = poly_uop2(ctx, red->arg.reduce.op, dtype, initted, inp, poly_arg_none());
  src[0] = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, initted, value, poly_arg_none());
  memcpy(src + 1, red->src + 1, (size_t)(red->n_src - 1) * sizeof(*src));
  PolyUOp *end = poly_uop_tagged_arg(
      ctx, POLY_OP_END, POLY_VOID, src, red->n_src, poly_arg_none(), 0, poly_arg_str("mergeable")
  );
  out = poly_uop2(ctx, POLY_OP_AFTER, dtype, acc, end, poly_arg_none());
  if (!out) goto fail;
  goto done;
fail:
  rctx->failed = true;
done:
  if (excluded) poly_map_destroy(excluded);
  free(src);
  poly_toposort_free(topo);
  return out;
}

/* Scratch rows for merge_reduce_ends' range_to_ends/by_ctx dictionaries.
 * They never own graph roots or survive this single callback. */
typedef struct {
  PolyUOp *end;
  PolyUOp **ranges;
  int n_ranges;
  bool used;
} ReduceEnd;

/* tinygrad/codegen/__init__.py:merge_reduce_ends. Discover the tagged ENDs
 * from sink.backward_slice, including graphs lowered in an earlier pass. */
static PolyUOp *merge_reduce_ends(PolyCtx *ctx, PolyUOp *sink, const PolyBindings *b) {
  (void)b;
  ReduceContext *rctx = current_reduce_ctx();
  if (!rctx || rctx->failed) return NULL;
  int n_topo = 0, n_ends = 0, n_ranges = 0, max_src = 1;
  PolyUOp **topo = poly_toposort_alloc(ctx, sink, &n_topo);
  ReduceEnd *ends = NULL;
  PolyUOp **old = NULL, **replacement = NULL, **tr = NULL, **body = NULL, **src = NULL;
  PolyUOp *out = NULL;
  int64_t max_axis = -1;
  if (!topo) goto fail;
  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    if (u->op == POLY_OP_RANGE) {
      n_ranges++;
      if (poly_range_axis_id(u->arg) > max_axis) max_axis = poly_range_axis_id(u->arg);
    }
    if (u->op != POLY_OP_END || u->tag_arg.kind != POLY_ARG_STRING ||
        strcmp(u->tag_arg.str, "mergeable"))
      continue;
    n_ends++;
    if (u->n_src > max_src) max_src = u->n_src;
  }
  if (n_ends < 2) goto done;
  ends = reduce_alloc((size_t)n_ends, sizeof(*ends));
  old = reduce_alloc((size_t)n_ends, sizeof(*old));
  replacement = reduce_alloc((size_t)n_ends, sizeof(*replacement));
  tr = reduce_alloc((size_t)max_src, sizeof(*tr));
  body = reduce_alloc((size_t)n_ends, sizeof(*body));
  src = reduce_alloc((size_t)max_src, sizeof(*src));
  if (!ends || !old || !replacement || !tr || !body || !src) goto fail;
  int at = 0;
  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    if (u->op != POLY_OP_END || u->tag_arg.kind != POLY_ARG_STRING ||
        strcmp(u->tag_arg.str, "mergeable"))
      continue;
    ReduceEnd *e = &ends[at++];
    e->end = u;
    if (n_ranges) {
      e->ranges = reduce_alloc((size_t)n_ranges, sizeof(*e->ranges));
      if (!e->ranges) goto fail;
      e->n_ranges = poly_uop_ranges(ctx, u, e->ranges, n_ranges);
    }
  }
  int n_subs = 0;
  for (int i = 0; i < n_ends; i++) {
    if (ends[i].used) continue;
    PolyUOp *first = ends[i].end;
    PolyUOp **r = first->src + 1;
    int nr = first->n_src - 1, count = 0;
    for (int j = i; j < n_ends; j++)
      if (same_range_tuple(r, nr, ends[j].end->src + 1, ends[j].end->n_src - 1)) count++;
    if (count < 2) {
      ends[i].used = true;
      continue;
    }
    int context = 0;
    for (int j = i; j < n_ends; j++) {
      if (ends[j].used || !same_range_tuple(r, nr, ends[j].end->src + 1, ends[j].end->n_src - 1))
        continue;
      if (context && max_axis > INT64_MAX - nr) goto fail;
      for (int k = 0; k < nr; k++) {
        tr[k] = context ? clone_range_axis(ctx, r[k], max_axis + 1 + k) : r[k];
        if (!tr[k]) goto fail;
      }
      if (context) max_axis += nr;
      int start = n_subs, n_body = 0;
      PolyUOp *mapped = NULL;
      for (int k = j; k < n_ends; k++) {
        ReduceEnd *e = &ends[k];
        if (e->used || !same_range_tuple(r, nr, e->end->src + 1, e->end->n_src - 1) ||
            !same_range_set(ends[j].ranges, ends[j].n_ranges, e->ranges, e->n_ranges))
          continue;
        mapped = context ? poly_uop_substitute(ctx, e->end, r, tr, nr) : e->end;
        if (!mapped) goto fail;
        body[n_body++] = mapped->src[0];
        old[n_subs++] = e->end;
        e->used = true;
      }
      if (n_body > 1) {
        src[0] = poly_uop(ctx, POLY_OP_GROUP, POLY_VOID, body, n_body, poly_arg_none());
        memcpy(src + 1, tr, (size_t)nr * sizeof(*src));
        mapped = nr ? poly_uop(ctx, POLY_OP_END, POLY_VOID, src, nr + 1, poly_arg_none()) : src[0];
      }
      if (!mapped) goto fail;
      for (int k = start; k < n_subs; k++)
        replacement[k] = mapped;
      context++;
    }
  }
  /* One complete map: sequential substitutions invalidate sibling keys. */
  if (n_subs) {
    out = poly_uop_substitute(ctx, sink, old, replacement, n_subs);
    if (!out) goto fail;
  }
  goto done;
fail:
  rctx->failed = true;
done:
  if (ends)
    for (int i = 0; i < n_ends; i++)
      free(ends[i].ranges);
  free(ends);
  free(old);
  free(replacement);
  free(tr);
  free(body);
  free(src);
  poly_toposort_free(topo);
  return out != sink ? out : NULL;
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
  if (c <= 1 || (c & (c - 1)) != 0) return NULL; /* pinned rule excludes shift zero */
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
  if (c <= 1 || (c & (c - 1)) != 0) return NULL; /* pinned rule excludes shift zero */
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
  PolyUOp *zero = poly_const_int(ctx, 0);
  PolyUOp *cmplt = poly_binop(ctx, POLY_OP_CMPLT, x_node, zero);
  int64_t lo = 0, hi = 1;
  poly_uop_minmax(ctx, cmplt, &lo, &hi);
  if (lo == hi) cmplt = poly_const_like_bool(ctx, cmplt, lo != 0);
  PolyUOp *cm1 = poly_sub(ctx, c_node, poly_const_int(ctx, 1));
  PolyUOp *correction = poly_where_op(ctx, cmplt, cm1, zero);
  return poly_binop(ctx, POLY_OP_SHR, poly_add(ctx, x_node, correction), shift_const);
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
  /* Pinned codegen/decomp/op.py:floordiv_to_idiv selects CDIV only from
   * expression bounds. Unsigned UOps can still carry mathematically inferred
   * ranges outside their storage dtype after wrapping ALU, so dtype alone is
   * not a proof that floor and truncating division agree. */
  if (divmod_floor_same_as_c(ctx, a, den))
    return poly_uop2(ctx, POLY_OP_CDIV, root->dtype, a, den, poly_arg_none());

  PolyUOp *trunc = poly_uop2(ctx, POLY_OP_CDIV, root->dtype, a, den, poly_arg_none());
  PolyUOp *rem = poly_uop2(ctx, POLY_OP_CMOD, root->dtype, a, den, poly_arg_none());
  PolyUOp *zero = poly_const_int(ctx, 0);
  PolyDType bt = POLY_BOOL;
  PolyUOp *rem_ne_zero = poly_uop2(ctx, POLY_OP_CMPNE, bt, rem, zero, poly_arg_none());
  PolyUOp *a_lt_zero = poly_uop2(ctx, POLY_OP_CMPLT, bt, a, zero, poly_arg_none());
  PolyUOp *b_lt_zero = poly_uop2(ctx, POLY_OP_CMPLT, bt, den, zero, poly_arg_none());
  PolyUOp *sign_mismatch = poly_uop2(ctx, POLY_OP_CMPNE, bt, a_lt_zero, b_lt_zero, poly_arg_none());
  PolyUOp *needs_adjust =
      poly_uop2(ctx, POLY_OP_AND, bt, rem_ne_zero, sign_mismatch, poly_arg_none());
  /* Use ElementwiseMixin.sub promotion/negation, not an early backend SUB. */
  return poly_sub(ctx, trunc, needs_adjust);
}

static PolyUOp *rule_floormod_to_cmod(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  if (!root || root->n_src != 2 || !poly_dtype_is_int(root->dtype)) return NULL;
  PolyUOp *a = root->src[0];
  PolyUOp *den = root->src[1];
  /* Pinned codegen/decomp/op.py:floormod_to_mod uses the same bounds-only
   * proof for FLOORMOD; do not infer sign from the unsigned storage dtype. */
  if (divmod_floor_same_as_c(ctx, a, den))
    return poly_uop2(ctx, POLY_OP_CMOD, root->dtype, a, den, poly_arg_none());

  PolyUOp *rem = poly_uop2(ctx, POLY_OP_CMOD, root->dtype, a, den, poly_arg_none());
  PolyUOp *zero = poly_const_int(ctx, 0);
  PolyDType bt = POLY_BOOL;
  PolyUOp *rem_ne_zero = poly_uop2(ctx, POLY_OP_CMPNE, bt, rem, zero, poly_arg_none());
  PolyUOp *a_lt_zero = poly_uop2(ctx, POLY_OP_CMPLT, bt, a, zero, poly_arg_none());
  PolyUOp *b_lt_zero = poly_uop2(ctx, POLY_OP_CMPLT, bt, den, zero, poly_arg_none());
  PolyUOp *sign_mismatch = poly_uop2(ctx, POLY_OP_CMPNE, bt, a_lt_zero, b_lt_zero, poly_arg_none());
  PolyUOp *needs_adjust =
      poly_uop2(ctx, POLY_OP_AND, bt, rem_ne_zero, sign_mismatch, poly_arg_none());
  PolyUOp *correction = poly_where_op(ctx, needs_adjust, den, poly_const_like_int(ctx, den, 0));
  return poly_add(ctx, rem, correction);
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

static bool late_signed_value(PolyUOp *u) {
  return u && poly_dtype_is_int(u->dtype) && !poly_dtype_is_unsigned(u->dtype) &&
         !poly_dtype_is_index(u->dtype) && !poly_dtype_is_bool(u->dtype);
}

/* tinygrad codegen/decomp/op.py:get_late_rewrite_patterns:
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

  if (lt->src[1]->op == POLY_OP_CONST && late_signed_value(lt->src[0])) {
    return poly_binop(
        ctx, POLY_OP_CMPLT, poly_sub(ctx, lt->src[1], poly_const_int(ctx, 1)), lt->src[0]
    );
  }
  if (lt->src[0]->op == POLY_OP_CONST && late_signed_value(lt->src[1])) {
    return poly_binop(
        ctx, POLY_OP_CMPLT, lt->src[1], poly_add(ctx, lt->src[0], poly_const_int(ctx, 1))
    );
  }
  return NULL;
}

static PolyUOp *late_signed_product(PolyUOp *u, PolyUOp **constant) {
  if (!u || u->op != POLY_OP_MUL || u->n_src != 2) return NULL;
  for (int i = 0; i < 2; i++) {
    if (late_signed_value(u->src[i]) && u->src[1 - i]->op == POLY_OP_CONST) {
      *constant = u->src[1 - i];
      return u->src[i];
    }
  }
  return NULL;
}

/* These comparisons stay late: simplex expects the original inequality. */
static PolyUOp *rule_negative_cmplt(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  if (!root || root->op != POLY_OP_CMPLT || root->n_src != 2) return NULL;
  PolyUOp *factor = NULL, *x = late_signed_product(root->src[0], &factor);
  int64_t value = 0;
  if (!x || !int_const_value_codegen(factor, &value) || value != -1) return NULL;
  PolyUOp *c = NULL, *y = late_signed_product(root->src[1], &c);
  if (!y) {
    c = root->src[1];
    if (c->op != POLY_OP_CONST) return NULL;
  }
  PolyUOp *negative = poly_mul(ctx, c, poly_const_int(ctx, -1));
  return poly_binop(ctx, POLY_OP_CMPLT, y ? poly_mul(ctx, y, negative) : negative, x);
}

static PolyUOp *rule_singleton_interval(PolyCtx *ctx, PolyUOp *root, const PolyBindings *b) {
  (void)b;
  if (!root || root->op != POLY_OP_AND || root->n_src != 2) return NULL;
  for (int i = 0; i < 2; i++) {
    PolyUOp *lower = root->src[i], *upper = root->src[1 - i];
    if (lower->op != POLY_OP_CMPLT || upper->op != POLY_OP_CMPLT || lower->n_src != 2 ||
        upper->n_src != 2 || lower->src[1] != upper->src[0] || !late_signed_value(lower->src[1]))
      continue;
    int64_t lo = 0, hi = 0, midpoint = 0;
    if (!int_const_value_codegen(lower->src[0], &lo) ||
        !int_const_value_codegen(upper->src[1], &hi) || __builtin_add_overflow(lo, 1, &midpoint) ||
        midpoint == INT64_MAX || hi != midpoint + 1)
      continue;
    return poly_eq(ctx, lower->src[1], poly_add(ctx, lower->src[0], poly_const_int(ctx, 1)));
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
  rules[n++] = (PolyRule){poly_upat_op(POLY_OP_CMPLT, NULL, 0, NULL), rule_negative_cmplt};
  rules[n++] = (PolyRule){poly_upat_op(POLY_OP_AND, NULL, 0, NULL), rule_singleton_interval};
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
  int64_t perm[POLY_MAX_DIMS];
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
      if (size > 1) {
        /* expand_reduce preserves source/axis encounter order. Sorting here
         * changes the horizontal reduction order (and floating-point sums). */
        if (new_axis[axis]) return NULL;
        new_axis[axis] = true;
        perm[n_new_axes++] = axis;
      }
    }
  }
  if (n_new_axes == 0) return NULL;

  PolyUOp *out_shape[POLY_MAX_DIMS];
  int at = n_new_axes;
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
  if (u->op == POLY_OP_LOAD || u->op == POLY_OP_RANGE || u->op == POLY_OP_SPECIAL) {
    /* UOp.addrspace: ALU is a real space, not the absent-space sentinel.
     * It must participate in the common-space check below. */
    if (out) *out = POLY_ADDR_ALU;
    return true;
  }
  if ((u->op == POLY_OP_INDEX || u->op == POLY_OP_CAST || u->op == POLY_OP_AFTER ||
       u->op == POLY_OP_REDUCE || u->op == POLY_OP_STORE || u->op == POLY_OP_MSTACK ||
       u->op == POLY_OP_MSELECT || u->op == POLY_OP_END || u->op == POLY_OP_UNSHARD ||
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
    /* Tinygrad uses x.replace: lane expansion must retain INDEX metadata. */
    PolyUOp *src[] = {buf, stack->src[i]};
    lanes[i] = poly_uop_replace_src(ctx, idx, src);
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
  /* placeholder commits a strong storage dtype, independent of STAGE's value. */
  PolyUOp *index = poly_uop_index(ctx, buf, stage->src + 1, stage->n_src - 1);
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
  return store ? poly_uop2(ctx, POLY_OP_AFTER, buf->dtype, buf, store, poly_arg_none()) : NULL;
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
  ReduceContext *rctx = current_reduce_ctx();
  if (!rctx || rctx->failed) return NULL;
  /* GROUP/LOCAL ranges become STAGE dimensions and retain the existing shape
   * rank limit. Other reduction loops do not become dimensions. */
  PolyUOp *group_ranges[POLY_MAX_DIMS];
  PolyUOp **partial_src = NULL, **topo = NULL, *out = NULL;
  int n_group = 0, n_other = 0;
  for (int i = 1; i < red->n_src; i++) {
    PolyUOp *range = red->src[i];
    if (range->op == POLY_OP_RANGE && poly_range_axis_type(range->arg) == POLY_AXIS_GROUP_REDUCE) {
      if (n_group == POLY_MAX_DIMS) goto fail;
      group_ranges[n_group++] = range;
    }
  }
  if (n_group == 0) return NULL;

  partial_src = reduce_alloc((size_t)red->n_src, sizeof(*partial_src));
  if (!partial_src) goto fail;
  partial_src[0] = red->src[0];
  for (int i = 1; i < red->n_src; i++) {
    PolyUOp *range = red->src[i];
    if (range->op != POLY_OP_RANGE || poly_range_axis_type(range->arg) != POLY_AXIS_GROUP_REDUCE)
      partial_src[++n_other] = range;
  }

  int n_topo = 0;
  topo = poly_toposort_alloc(ctx, red, &n_topo);
  if (!topo) goto fail;
  PolyUOp *upstream_locals[POLY_MAX_DIMS];
  int n_upstream = 0;
  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    if (!u || u->op != POLY_OP_RANGE || poly_range_axis_type(u->arg) != POLY_AXIS_LOCAL) continue;
    bool duplicate = false;
    for (int j = 0; j < n_upstream; j++)
      duplicate |= upstream_locals[j] == u;
    if (!duplicate) {
      if (n_upstream + n_group == POLY_MAX_DIMS) goto fail;
      upstream_locals[n_upstream++] = u;
    }
  }
  PolyUOp *partial = poly_uop_tagged_arg(
      ctx, red->op, red->dtype, partial_src, n_other + 1, red->arg, red->tag, red->tag_arg
  );

  PolyUOp *stage_src[1 + 2 * POLY_MAX_DIMS] = {partial};
  int n_stage = 1;
  for (int i = 0; i < n_upstream; i++)
    stage_src[n_stage++] = upstream_locals[i];
  for (int i = 0; i < n_group; i++)
    stage_src[n_stage++] = group_ranges[i];
  /* The first grouped range identifies LOCAL storage in BufferizeOpts.device. */
  PolyUOp *stage = poly_uop(
      ctx, POLY_OP_STAGE, red->dtype, stage_src, n_stage,
      poly_arg_bufferize_opts_int(poly_range_axis_id(group_ranges[0]->arg), POLY_ADDR_LOCAL, true)
  );

  PolyUOp *reduce_loop[POLY_MAX_DIMS];
  for (int i = 0; i < n_group; i++) {
    PolyUOp *range = group_ranges[i];
    if (poly_range_axis_id(range->arg) > INT64_MAX - 100) goto fail;
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
  out = poly_uop_tagged_arg(
      /* fix_group_for_reduce consumes horizontal axes in the partial reduction. */
      ctx, red->op, red->dtype, final_src, n_group + 1, poly_arg_reduce(red->arg.reduce.op, 0),
      red->tag, red->tag_arg
  );
  if (!out) goto fail;
  goto done;
fail:
  rctx->failed = true;
done:
  free(partial_src);
  poly_toposort_free(topo);
  return out;
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
  PolyUOp *out = sched_can_optimize(&s) && sched_hand_coded_tensor_cores(&s, caps) ? s.ast : sink;
  if (s.failed) out = NULL;
  sched_destroy(&s);
  return out;
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
       reduce_ranges_to_acc},
      {poly_upat_op1(POLY_OP_REDUCE, poly_upat_any(NULL), "r"), poly_expand_horizontal_reduce},
      {poly_upat_allow_any_len(poly_upat_op(POLY_OP_SINK, NULL, 0, "sink")), merge_reduce_ends},
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
  return local_ctx.failed ? NULL : out;
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
    /* backward_slice_with_self is root first, then the remaining toposort.
     * A body-root LOAD is also an explicit barrier dependency in Tinygrad. */
    PolyUOp *u = i == 0 ? topo[n_topo - 1] : topo[i - 1];
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

#ifdef POLY_TESTING
PolyUOp *poly_test_add_loads(PolyCtx *ctx, PolyUOp *u) {
  return poly_pm_rewrite(poly_pm_add_loads(), ctx, u);
}

PolyUOp *poly_test_implicit_barriers(PolyCtx *ctx, PolyUOp *u) {
  return poly_pm_rewrite(poly_pm_implicit_barriers(), ctx, u);
}

PolyUOp *poly_test_devectorizer2(PolyCtx *ctx, PolyUOp *u) {
  return poly_pm_rewrite(poly_devectorizer2(), ctx, u);
}

PolyUOp *poly_test_add_local_buffers(PolyCtx *ctx, PolyUOp *u, int *next_slot) {
  return poly_graph_rewrite_ctx(ctx, u, poly_pm_add_local_buffers(), next_slot);
}
#endif

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
   * opt_policy selects scheduling policy; extra_matcher supplies the
   * renderer's final rewrite rules. CUDA uses the shared heuristic policy.
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

    /* postrange.apply_opts: an explicit list (even empty) takes precedence
     * over BEAM and NOOPT. Tagged kernels have already been scheduled. */
    if (!sink->tag && sink->tag_arg.kind == POLY_ARG_NONE) {
      if (opts.caps.has_local) {
        sink = convert_loop_to_global(ctx, sink);
        POLY_REWRITE_CHECK("convert loop output ranges");
      }
      const PolyKernelInfo *info =
          sink->arg.kind == POLY_ARG_KERNEL_INFO ? sink->arg.kernel_info : NULL;
      if (info && info->has_opts_to_apply) {
        OptScheduler scheduler;
        sched_init(&scheduler, ctx, sink);
        for (int i = 0; i < info->n_opts_to_apply; i++) {
          if (!sched_apply_opt(&scheduler, opts.caps, info->opts_to_apply[i], NULL)) {
            fprintf(stderr, "polygrad: explicit kernel option %d is invalid\n", i);
            sched_destroy(&scheduler);
            return NULL;
          }
        }
        sink = scheduler.ast;
        sched_destroy(&scheduler);
      } else if (opts.beam_width > 0) {
        sink = poly_beam_search(ctx, sink, opts.beam_width, opts);
      } else if (!poly_get_noopt() && (!info || !info->n_applied_opts)) {
        int n = 0;
        PolyUOp **topo = poly_toposort_alloc(ctx, sink, &n);
        if (!topo) return NULL;
        bool staged = false;
        for (int i = 0; i < n; i++)
          staged |= topo[i]->op == POLY_OP_STAGE;
        poly_toposort_free(topo);
        if (!staged)
          sink = opts.opt_policy == POLY_OPT_TC_ONLY
                     ? poly_apply_tc_opt(ctx, sink, opts.caps)
                     : poly_apply_opts_heuristic(ctx, sink, opts.caps);
      }
      POLY_REWRITE_CHECK("schedule options");
      /* get_optimized_ast consumes construction options and seals scheduling;
       * flatten END range expressions before expander consumes them. */
      sink = poly_graph_rewrite(ctx, sink, poly_pm_flatten_range());
      POLY_REWRITE_CHECK("flatten scheduled ranges");
      PolyKernelInfo scheduled = sink->arg.kind == POLY_ARG_KERNEL_INFO
                                     ? *sink->arg.kernel_info
                                     : (PolyKernelInfo){.name = "test"};
      scheduled.has_opts_to_apply = false;
      scheduled.opts_to_apply = NULL;
      scheduled.n_opts_to_apply = 0;
      sink = poly_uop_tagged(
          ctx, sink->op, sink->dtype, sink->src, sink->n_src, poly_arg_kernel_info(&scheduled), 1
      );
    }
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
  for (int i = 0; i < n; i++) {
    /* pm_linearize_cleanups rejects graph IF/ENDIF before introducing its
     * own balanced conditional around a gated STORE. */
    if (linear[i]->op == POLY_OP_IF || linear[i]->op == POLY_OP_ENDIF) {
      fprintf(stderr, "polygrad: if not allowed in graph\n");
      return NULL;
    }
    if (line_is_gated_store(linear[i])) n_gated++;
  }
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
  if (n_out) *n_out = 0;
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
