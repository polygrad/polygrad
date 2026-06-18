/*
 * frontend.c — FFI-friendly helpers for language bindings
 *
 * Thin wrappers around the core API that avoid passing PolyArg/PolyDType
 * across FFI boundaries. Execution uses graph/schedule entrypoints backed by
 * ctx->buffers; there is no separate runtime buffer-binding graph.
 */

#define _GNU_SOURCE
#include "frontend.h"
#include "frontend_internal.h"
#include "engine/schedule.h"
#include "schedule/rangeify.h"
#include "codegen.h"
#include "interp.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "utils.h"

/* Dtype table for FFI (shared by buffer/dtype convenience helpers) */

static const PolyDType *_dtype_table_ffi[] = {
    &POLY_VOID,    &POLY_BOOL,     &POLY_INT8,    &POLY_UINT8,   &POLY_INT16,
    &POLY_UINT16,  &POLY_INT32,    &POLY_UINT32,  &POLY_INT64,   &POLY_UINT64,
    &POLY_FLOAT16, &POLY_BFLOAT16, &POLY_FLOAT32, &POLY_FLOAT64,
};
#define N_DTYPE_FFI ((int)(sizeof(_dtype_table_ffi) / sizeof(_dtype_table_ffi[0])))

PolyUOp *poly_buffer_by_id(PolyCtx *ctx, int dtype_id, int64_t size) {
  PolyDType dt;
  if (!poly_dtype_by_id(dtype_id, &dt)) return NULL;
  return poly_buffer(ctx, poly_dtype_scalar(dt), size);
}

PolyUOp *poly_buffer_f32(PolyCtx *ctx, int64_t size) {
  return poly_buffer(ctx, POLY_FLOAT32, size);
}

PolyUOp *poly_buffer_f64(PolyCtx *ctx, int64_t size) {
  return poly_buffer(ctx, POLY_FLOAT64, size);
}

int poly_uop_dtype_id(PolyCtx *ctx, PolyUOp *u) {
  (void)ctx;
  if (!u) return 0;
  PolyDType sdt = poly_dtype_scalar(u->dtype);
  for (int i = 0; i < N_DTYPE_FFI; i++) {
    if (poly_dtype_eq(sdt, *_dtype_table_ffi[i])) return i;
  }
  return 0;
}

/* Dynamic shapes (DEFINE_VAR / BIND) */

PolyUOp *poly_define_var(PolyCtx *ctx, const char *name, int64_t min_val, int64_t max_val) {
  /* Name string is copied into arena by poly_uop_create (POLY_ARG_DEFINE_VAR case) */
  return poly_uop0(
      ctx, POLY_OP_DEFINE_VAR, POLY_INT32, poly_arg_define_var(name, min_val, max_val)
  );
}

PolyUOp *poly_bind_var(PolyCtx *ctx, PolyUOp *var, int64_t value) {
  assert(var->op == POLY_OP_DEFINE_VAR);
  PolyUOp *val = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(value));
  return poly_uop2(ctx, POLY_OP_BIND, var->dtype, var, val, poly_arg_none());
}

/* POLY_MAX_REALIZE_BUFS defined in frontend_internal.h */

static bool uop_vec_append(PolyUOp ***items, int *count, int *cap, PolyUOp *u) {
  if (!items || !count || !cap) return false;
  if (*count >= *cap) {
    int new_cap = *cap ? *cap * 2 : 16;
    PolyUOp **tmp = realloc(*items, (size_t)new_cap * sizeof(PolyUOp *));
    if (!tmp) return false;
    *items = tmp;
    *cap = new_cap;
  }
  (*items)[(*count)++] = u;
  return true;
}

static bool uop_vec_contains(PolyUOp **items, int count, PolyUOp *u) {
  for (int i = 0; i < count; i++)
    if (items[i] == u) return true;
  return false;
}

/* Reconstruct the buffer-to-PARAM ordering used by kernel-graph scheduling:
 * 1. Output buffers (STORE targets in SINK source order)
 * 2. Remaining input buffers (toposort encounter order) */
bool poly_collect_ordered_buffers_alloc(
    PolyCtx *ctx,
    PolyUOp *tensor_sink,
    PolyUOp ***out_ordered,
    int *out_n_ordered
) {
  if (!ctx || !tensor_sink || !out_ordered || !out_n_ordered) return false;
  *out_ordered = NULL;
  *out_n_ordered = 0;

  PolyUOp **ordered = NULL;
  int n = 0, cap = 0;

  /* Output buffers first */
  for (int i = 0; i < tensor_sink->n_src; i++) {
    PolyUOp *store = tensor_sink->src[i];
    if (store && store->op == POLY_OP_STORE && store->n_src >= 1 &&
        store->src[0]->op == POLY_OP_BUFFER) {
      PolyUOp *buf = store->src[0];
      if (!uop_vec_contains(ordered, n, buf) && !uop_vec_append(&ordered, &n, &cap, buf)) {
        free(ordered);
        return false;
      }
    }
  }

  /* Input buffers in toposort order */
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, tensor_sink, &n_topo);
  if (!topo && n_topo > 0) {
    free(ordered);
    return false;
  }
  for (int i = 0; i < n_topo; i++) {
    if (topo[i]->op == POLY_OP_BUFFER && !uop_vec_contains(ordered, n, topo[i]) &&
        !uop_vec_append(&ordered, &n, &cap, topo[i])) {
      free(ordered);
      return false;
    }
  }

  *out_ordered = ordered;
  *out_n_ordered = n;
  return true;
}

int poly_collect_ordered_buffers(
    PolyCtx *ctx,
    PolyUOp *tensor_sink,
    PolyUOp **ordered,
    int max_bufs
) {
  PolyUOp **all = NULL;
  int n_all = 0;
  if (!poly_collect_ordered_buffers_alloc(ctx, tensor_sink, &all, &n_all)) return 0;
  int n_copy = n_all < max_bufs ? n_all : max_bufs;
  for (int i = 0; i < n_copy; i++)
    ordered[i] = all[i];
  free(all);
  return n_copy;
}

/* Weak context cleanup hook called from ctx.c when this translation unit is
 * linked. Frontend-global caches were removed; per-context caches are owned by
 * ctx/schedule/program-cache teardown. */
void poly_frontend_ctx_cleanup(PolyCtx *ctx) {
  (void)ctx;
}

/* Structural hash/eq for graph caches.
 * Cached schedules/programs need to match computations that are structurally
 * identical but use different BUFFER UOp instances, such as fresh training
 * step buffers. We hash/compare the computation DAG structure: ops, dtypes,
 * args, and connectivity, treating BUFFER
 * nodes as positional placeholders (first encountered = 0, etc.).
 */

/* POLY_MAX_STRUCT_NODES defined in frontend_internal.h */

typedef struct {
  PolyMap *visited; /* UOp* -> 1-based index into hashes. Dynamic for model-scale DAGs. */
  uint32_t *hashes;
  int n_hashes;
  int cap_hashes;
  PolyUOp **bufs;
  int n_bufs;
  int cap_bufs;
} StructHashCtx;

static uint32_t struct_hash_impl(PolyUOp *u, StructHashCtx *ctx) {
  /* Check if already visited */
  void *memo_val = poly_map_get(ctx->visited, poly_ptr_hash(u), u, poly_ptr_eq);
  if (memo_val) return ctx->hashes[(int)((intptr_t)memo_val - 1)];

  uint32_t h = 0x811c9dc5; /* FNV-1a offset basis */

  if (u->op == POLY_OP_BUFFER) {
    /* BUFFER nodes: use positional ID instead of pointer identity */
    int buf_id = -1;
    for (int i = 0; i < ctx->n_bufs; i++) {
      if (ctx->bufs[i] == u) {
        buf_id = i;
        break;
      }
    }
    if (buf_id < 0) {
      if (ctx->n_bufs >= ctx->cap_bufs) {
        int new_cap = ctx->cap_bufs ? ctx->cap_bufs * 2 : 64;
        PolyUOp **new_bufs = realloc(ctx->bufs, (size_t)new_cap * sizeof(PolyUOp *));
        if (!new_bufs) return h;
        ctx->bufs = new_bufs;
        ctx->cap_bufs = new_cap;
      }
      buf_id = ctx->n_bufs;
      ctx->bufs[ctx->n_bufs++] = u;
    }
    h ^= (uint32_t)u->op;
    h *= 0x01000193;
    h ^= (uint32_t)u->dtype.priority;
    h *= 0x01000193;
    h ^= (uint32_t)u->dtype.bitsize;
    h *= 0x01000193;
    h ^= (uint32_t)buf_id;
    h *= 0x01000193;
    h ^= poly_arg_hash(u->arg);
    h *= 0x01000193;
  } else {
    h ^= (uint32_t)u->op;
    h *= 0x01000193;
    h ^= (uint32_t)u->dtype.priority;
    h *= 0x01000193;
    h ^= (uint32_t)u->dtype.bitsize;
    h *= 0x01000193;
    for (int i = 0; i < u->n_src; i++) {
      h ^= struct_hash_impl(u->src[i], ctx);
      h *= 0x01000193;
    }
    h ^= poly_arg_hash(u->arg);
    h *= 0x01000193;
  }

  if (ctx->n_hashes >= ctx->cap_hashes) {
    int new_cap = ctx->cap_hashes ? ctx->cap_hashes * 2 : 1024;
    uint32_t *new_hashes = realloc(ctx->hashes, (size_t)new_cap * sizeof(uint32_t));
    if (!new_hashes) return h;
    ctx->hashes = new_hashes;
    ctx->cap_hashes = new_cap;
  }
  int idx = ctx->n_hashes++;
  ctx->hashes[idx] = h;
  poly_map_set(ctx->visited, poly_ptr_hash(u), u, (void *)(intptr_t)(idx + 1), poly_ptr_eq);
  return h;
}

uint32_t poly_structural_hash(PolyUOp *u) {
  if (!u) return 0;
  StructHashCtx ctx;
  memset(&ctx, 0, sizeof(ctx));
  ctx.visited = poly_map_new(1024);
  if (!ctx.visited) return 0;
  uint32_t h = struct_hash_impl(u, &ctx);
  poly_map_destroy(ctx.visited);
  free(ctx.hashes);
  free(ctx.bufs);
  return h;
}

/* Structural equality */

typedef struct {
  PolyMap *a_to_b;
  PolyMap *b_to_a;
} BufPairs;

typedef struct {
  PolyMap *a_to_b;
  PolyMap *b_to_a;
} EqVisited;

static bool struct_eq_impl(PolyUOp *a, PolyUOp *b, BufPairs *bp, EqVisited *ev) {
  if (a == b) return true;
  if (!a || !b) return false;

  /* Check if this pair already visited (DAG sharing) */
  void *seen_b = poly_map_get(ev->a_to_b, poly_ptr_hash(a), a, poly_ptr_eq);
  if (seen_b) return seen_b == b;
  if (poly_map_get(ev->b_to_a, poly_ptr_hash(b), b, poly_ptr_eq)) return false;

  poly_map_set(ev->a_to_b, poly_ptr_hash(a), a, b, poly_ptr_eq);
  poly_map_set(ev->b_to_a, poly_ptr_hash(b), b, a, poly_ptr_eq);

  /* Both BUFFER? Track correspondence */
  if (a->op == POLY_OP_BUFFER && b->op == POLY_OP_BUFFER) {
    if (!poly_dtype_eq(a->dtype, b->dtype)) return false;
    if (!poly_arg_eq(a->arg, b->arg)) return false;
    /* Check existing mapping */
    void *mapped_b = poly_map_get(bp->a_to_b, poly_ptr_hash(a), a, poly_ptr_eq);
    if (mapped_b) return mapped_b == b;
    if (poly_map_get(bp->b_to_a, poly_ptr_hash(b), b, poly_ptr_eq)) return false;
    poly_map_set(bp->a_to_b, poly_ptr_hash(a), a, b, poly_ptr_eq);
    poly_map_set(bp->b_to_a, poly_ptr_hash(b), b, a, poly_ptr_eq);
    return true;
  }

  /* Same op, dtype, n_src, arg? */
  if (a->op != b->op) return false;
  if (!poly_dtype_eq(a->dtype, b->dtype)) return false;
  if (a->n_src != b->n_src) return false;
  if (!poly_arg_eq(a->arg, b->arg)) return false;

  /* Recursively compare sources */
  for (int i = 0; i < a->n_src; i++) {
    if (!struct_eq_impl(a->src[i], b->src[i], bp, ev)) return false;
  }
  return true;
}

bool poly_structural_eq(const void *a, const void *b) {
  BufPairs bp = {.a_to_b = poly_map_new(64), .b_to_a = poly_map_new(64)};
  EqVisited ev = {.a_to_b = poly_map_new(1024), .b_to_a = poly_map_new(1024)};
  if (!bp.a_to_b || !bp.b_to_a || !ev.a_to_b || !ev.b_to_a) {
    if (bp.a_to_b) poly_map_destroy(bp.a_to_b);
    if (bp.b_to_a) poly_map_destroy(bp.b_to_a);
    if (ev.a_to_b) poly_map_destroy(ev.a_to_b);
    if (ev.b_to_a) poly_map_destroy(ev.b_to_a);
    return false;
  }
  bool ok = struct_eq_impl((PolyUOp *)a, (PolyUOp *)b, &bp, &ev);
  poly_map_destroy(bp.a_to_b);
  poly_map_destroy(bp.b_to_a);
  poly_map_destroy(ev.a_to_b);
  poly_map_destroy(ev.b_to_a);
  return ok;
}

/* DFS to assign positional IDs to BUFFER nodes, matching poly_structural_hash order.
 * Children are visited left-to-right, same as struct_hash_impl().
 * n_bufs counts total BUFFERs found (may exceed buf_order capacity).
 * buf_order is only written up to POLY_MAX_REALIZE_BUFS entries.
 * Callers must check *n_bufs <= POLY_MAX_REALIZE_BUFS after the call. */
void poly_collect_buf_order(
    PolyUOp *u,
    PolyUOp **buf_order,
    int *n_bufs,
    PolyUOp **visited,
    int *n_visited
) {
  if (!u || !n_bufs || !n_visited) return;

  /* Model-scale graphs exceed POLY_MAX_STRUCT_NODES. Keep the public scratch
   * arrays for ABI compatibility, but use a dynamic visited map so traversal
   * stays O(nodes) instead of revisiting shared DAG tails after the cap. */
  PolyMap *seen = poly_map_new(1024);
  if (!seen) return;
  int initial_visited = *n_visited;
  for (int i = 0; visited && i < initial_visited && i < POLY_MAX_STRUCT_NODES; i++) {
    if (visited[i])
      poly_map_set(seen, poly_ptr_hash(visited[i]), visited[i], visited[i], poly_ptr_eq);
  }

  int cap = 1024;
  int sp = 0;
  PolyUOp **stack = malloc((size_t)cap * sizeof(PolyUOp *));
  if (!stack) {
    poly_map_destroy(seen);
    return;
  }
  stack[sp++] = u;

  while (sp > 0) {
    PolyUOp *cur = stack[--sp];
    if (!cur) continue;
    if (poly_map_get(seen, poly_ptr_hash(cur), cur, poly_ptr_eq)) continue;
    poly_map_set(seen, poly_ptr_hash(cur), cur, cur, poly_ptr_eq);
    if (visited && *n_visited < POLY_MAX_STRUCT_NODES) visited[*n_visited] = cur;
    (*n_visited)++;

    if (cur->op == POLY_OP_BUFFER) {
      if (buf_order && *n_bufs < POLY_MAX_REALIZE_BUFS) buf_order[*n_bufs] = cur;
      (*n_bufs)++; /* always count, even past capacity */
      continue;
    }

    if (sp + cur->n_src > cap) {
      int new_cap = cap;
      while (sp + cur->n_src > new_cap)
        new_cap *= 2;
      PolyUOp **new_stack = realloc(stack, (size_t)new_cap * sizeof(PolyUOp *));
      if (!new_stack) break;
      stack = new_stack;
      cap = new_cap;
    }
    for (int i = cur->n_src - 1; i >= 0; i--)
      stack[sp++] = cur->src[i];
  }

  free(stack);
  poly_map_destroy(seen);
}

bool poly_collect_buf_order_alloc(
    PolyUOp *u,
    PolyUOp ***out_buf_order,
    int *out_n_bufs,
    int *out_n_visited
) {
  if (!u || !out_buf_order || !out_n_bufs || !out_n_visited) return false;
  *out_buf_order = NULL;
  *out_n_bufs = 0;
  *out_n_visited = 0;

  PolyMap *seen = poly_map_new(1024);
  if (!seen) return false;

  int stack_cap = 1024;
  int sp = 0;
  PolyUOp **stack = malloc((size_t)stack_cap * sizeof(PolyUOp *));
  if (!stack) {
    poly_map_destroy(seen);
    return false;
  }
  stack[sp++] = u;

  PolyUOp **buf_order = NULL;
  int n_bufs = 0, buf_cap = 0;
  bool ok = true;

  while (ok && sp > 0) {
    PolyUOp *cur = stack[--sp];
    if (!cur) continue;
    if (poly_map_get(seen, poly_ptr_hash(cur), cur, poly_ptr_eq)) continue;
    poly_map_set(seen, poly_ptr_hash(cur), cur, cur, poly_ptr_eq);
    (*out_n_visited)++;

    if (cur->op == POLY_OP_BUFFER) {
      ok = uop_vec_append(&buf_order, &n_bufs, &buf_cap, cur);
      continue;
    }

    if (sp + cur->n_src > stack_cap) {
      int new_cap = stack_cap;
      while (sp + cur->n_src > new_cap)
        new_cap *= 2;
      PolyUOp **new_stack = realloc(stack, (size_t)new_cap * sizeof(PolyUOp *));
      if (!new_stack) {
        ok = false;
        break;
      }
      stack = new_stack;
      stack_cap = new_cap;
    }
    for (int i = cur->n_src - 1; i >= 0; i--)
      stack[sp++] = cur->src[i];
  }

  free(stack);
  poly_map_destroy(seen);
  if (!ok) {
    free(buf_order);
    return false;
  }
  *out_buf_order = buf_order;
  *out_n_bufs = n_bufs;
  return true;
}

int poly_find_buf_position(PolyUOp *buf, PolyUOp **buf_order, int n_bufs) {
  for (int i = 0; i < n_bufs; i++)
    if (buf_order[i] == buf) return i;
  return -1;
}

/* Helpers shared by all builds (exec_plan + realize) */

/* Defensive graph validator to avoid crashing in toposort/linearize when
 * a malformed kernel contains NULL sources. */
bool poly_validate_kernel_graph(PolyCtx *ctx, PolyUOp *root) {
  if (!root) return false;
  PolyMap *visited = poly_map_new(256);
  PolyUOp *stack[4096];
  PolyUOp *parent_stack[4096];
  int parent_src_idx[4096];
  int sp = 0;
  stack[sp++] = root;
  parent_stack[0] = NULL;
  parent_src_idx[0] = -1;

  while (sp > 0) {
    sp--;
    PolyUOp *u = stack[sp];
    PolyUOp *parent = parent_stack[sp];
    int src_idx = parent_src_idx[sp];
    if (!u) {
      poly_map_destroy(visited);
      return false;
    }
    if (!poly_ctx_owns_ptr(ctx, u)) {
      if (parent) {
        fprintf(
            stderr,
            "polygrad: realize: foreign/stale UOp pointer %p referenced by %s(%p) src[%d]\n",
            (void *)u, poly_op_name(parent->op), (void *)parent, src_idx
        );
        fprintf(
            stderr, "polygrad: realize: parent %s n_src=%d\n", poly_op_name(parent->op),
            parent->n_src
        );
        for (int si = 0; si < parent->n_src; si++) {
          PolyUOp *ps = parent->src[si];
          bool owned = poly_ctx_owns_ptr(ctx, ps);
          fprintf(
              stderr, "  parent.src[%d]=%p %s%s\n", si, (void *)ps, owned ? "" : "[FOREIGN] ",
              (owned && ps) ? poly_op_name(ps->op) : ""
          );
        }
      } else {
        fprintf(
            stderr, "polygrad: realize: foreign/stale root UOp pointer %p in kernel graph\n",
            (void *)u
        );
      }
      poly_map_destroy(visited);
      return false;
    }
    if (poly_map_get(visited, poly_ptr_hash(u), u, poly_ptr_eq)) continue;
    poly_map_set(visited, poly_ptr_hash(u), u, u, poly_ptr_eq);

    if (u->n_src > 64) {
      fprintf(
          stderr, "polygrad: realize: invalid n_src=%d on %s(%p)\n", u->n_src, poly_op_name(u->op),
          (void *)u
      );
      poly_map_destroy(visited);
      return false;
    }
    for (int i = 0; i < u->n_src; i++) {
      if (!u->src[i]) {
        fprintf(
            stderr, "polygrad: realize: NULL src[%d] on %s(%p), n_src=%d\n", i, poly_op_name(u->op),
            (void *)u, u->n_src
        );
        poly_map_destroy(visited);
        return false;
      }
      if (sp < (int)(sizeof(stack) / sizeof(stack[0]))) {
        stack[sp++] = u->src[i];
        parent_stack[sp - 1] = u;
        parent_src_idx[sp - 1] = i;
      }
    }
  }
  poly_map_destroy(visited);
  return true;
}

/* POLY_SCHED_CACHE_VERSION defined in frontend_internal.h */

/* CPU realize (not available in Emscripten) */

int poly_collect_output_buffers_in_sink(PolyUOp *tensor_sink, PolyUOp **out, int cap) {
  if (!tensor_sink || tensor_sink->op != POLY_OP_SINK) return 0;
  int n_seen = 0;
  for (int i = 0; i < tensor_sink->n_src; i++) {
    PolyUOp *store = tensor_sink->src[i];
    if (!store || store->op != POLY_OP_STORE || store->n_src < 1) continue;
    PolyUOp *buf = store->src[0];
    if (!buf || buf->op != POLY_OP_BUFFER) continue;
    bool dup = false;
    for (int j = 0; j < n_seen; j++) {
      if (out[j] == buf) {
        dup = true;
        break;
      }
    }
    if (!dup && n_seen < cap) out[n_seen++] = buf;
  }
  return n_seen;
}

void poly_cpu_cache_flush(void) {
  /* Retained for ABI/frontend cleanup paths. CPU program caches are per-context
   * now and are released through poly_ctx_destroy(). */
}

/* Exec plan functions (poly_complete_create_schedule_with_vars, poly_run_schedule,
 * backend lowering, etc.) live in engine/schedule.c. Frontend execution now
 * reaches them only through graph/tensor realize entrypoints. */

int poly_abi_version(void) {
  return POLYGRAD_ABI_VERSION;
}

/* Debug helper — check opset state */
void poly_debug_opsets(void) {
  fprintf(stderr, "=== OpSet debug ===\n");
  fprintf(
      stderr, "POLY_GROUP_ALU.bits = [%llu, %llu]\n", (unsigned long long)POLY_GROUP_ALU.bits[0],
      (unsigned long long)POLY_GROUP_ALU.bits[1]
  );
  fprintf(
      stderr, "POLY_GROUP_UNARY.bits = [%llu, %llu]\n",
      (unsigned long long)POLY_GROUP_UNARY.bits[0], (unsigned long long)POLY_GROUP_UNARY.bits[1]
  );
  fprintf(
      stderr, "POLY_GROUP_BINARY.bits = [%llu, %llu]\n",
      (unsigned long long)POLY_GROUP_BINARY.bits[0], (unsigned long long)POLY_GROUP_BINARY.bits[1]
  );
  fprintf(
      stderr, "poly_opset_has(ALU, ADD=%d) = %d\n", POLY_OP_ADD,
      poly_opset_has(POLY_GROUP_ALU, POLY_OP_ADD)
  );
  fprintf(
      stderr, "poly_opset_has(BINARY, ADD=%d) = %d\n", POLY_OP_ADD,
      poly_opset_has(POLY_GROUP_BINARY, POLY_OP_ADD)
  );
  fprintf(stderr, "===================\n");
}

/* Debug helper — print UOp info */
void poly_debug_uop(PolyCtx *ctx, PolyUOp *u) {
  if (!u) {
    fprintf(stderr, "poly_debug_uop: NULL\n");
    return;
  }
  fprintf(
      stderr, "UOp@%p: op=%s(%d) n_src=%d arg.kind=%d", (void *)u, poly_op_name(u->op), u->op,
      u->n_src, u->arg.kind
  );
  if (u->arg.kind == POLY_ARG_INT) fprintf(stderr, " arg.i=%lld", (long long)u->arg.i);
  if (u->arg.kind == POLY_ARG_FLOAT) fprintf(stderr, " arg.f=%f", u->arg.f);
  fprintf(stderr, "\n");
  for (int i = 0; i < u->n_src; i++) {
    fprintf(
        stderr, "  src[%d]: @%p op=%s(%d)\n", i, (void *)u->src[i], poly_op_name(u->src[i]->op),
        u->src[i]->op
    );
  }
  /* Try shape */
  PolyShape s = poly_uop_shape(ctx, u);
  if (s.ndim >= 0) {
    fprintf(stderr, "  shape: (");
    for (int i = 0; i < s.ndim; i++) {
      if (i) fprintf(stderr, ", ");
      fprintf(stderr, "%lld", (long long)s.dims[i]);
    }
    fprintf(stderr, ")\n");
    if (s.ndim > 0 && s.dims) free(s.dims);
  } else {
    fprintf(stderr, "  shape: NONE\n");
  }
}
