/*
 * frontend.c — FFI-friendly helpers for language bindings
 *
 * Thin wrappers around the core API that avoid passing PolyArg/PolyDType
 * across FFI boundaries, plus poly_realize() which wraps the full
 * schedule → linearize → render → compile → execute pipeline.
 */

#define _GNU_SOURCE
#include "frontend.h"
#include "tensor.h"
#include "frontend_internal.h"
#include "exec_plan.h"
#include "scheduler.h"
#include "rangeify.h"
#include "codegen.h"
#include "interp.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* In-place assignment (stays here: depends on scheduler.h) */

PolyUOp *poly_assign(PolyCtx *ctx, PolyUOp *target, PolyUOp *value) {
  /* Normalize: walk target through movement ops to the base BUFFER.
   * Reshape value to the base buffer's flat shape so the ASSIGN target
   * is always a raw BUFFER. This ensures in-place writes go directly to
   * the buffer data without the scheduler needing to handle movement ops
   * on ASSIGN targets. */
  PolyUOp *base = target;
  while (poly_opset_has(POLY_GROUP_MOVEMENT, base->op) && base->n_src > 0)
    base = base->src[0];

  if (base != target && base->op == POLY_OP_BUFFER) {
    /* Base is a BUFFER with a flat shape. Reshape value to match. */
    int64_t numel = (base->arg.kind == POLY_ARG_INT) ? base->arg.i : 0;
    if (numel > 0) {
      int64_t flat_shape[1] = {numel};
      value = poly_reshape(ctx, value, flat_shape, 1);
    }
    target = base;
  }

  PolyUOp *srcs[2] = {target, value};
  return poly_uop(ctx, POLY_OP_ASSIGN, target->dtype, srcs, 2, poly_arg_none());
}

/* Dtype table for FFI (shared by poly_buffer_by_id and poly_cast_by_id) */

static const PolyDType *_dtype_table_ffi[] = {
    &POLY_VOID,    &POLY_BOOL,     &POLY_INT8,    &POLY_UINT8,   &POLY_INT16,
    &POLY_UINT16,  &POLY_INT32,    &POLY_UINT32,  &POLY_INT64,   &POLY_UINT64,
    &POLY_FLOAT16, &POLY_BFLOAT16, &POLY_FLOAT32, &POLY_FLOAT64,
};
#define N_DTYPE_FFI ((int)(sizeof(_dtype_table_ffi) / sizeof(_dtype_table_ffi[0])))

/* Buffer shortcuts */

PolyUOp *poly_buffer_f32(PolyCtx *ctx, int64_t size) {
  return poly_buffer(ctx, POLY_FLOAT32, size);
}

PolyUOp *poly_buffer_f64(PolyCtx *ctx, int64_t size) {
  return poly_buffer(ctx, POLY_FLOAT64, size);
}

PolyUOp *poly_buffer_by_id(PolyCtx *ctx, int64_t size, int dtype_id) {
  if (dtype_id < 0 || dtype_id >= N_DTYPE_FFI) return NULL;
  return poly_buffer(ctx, *_dtype_table_ffi[dtype_id], size);
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

static int poly_dyn_buffer_id = 1000000; /* separate range from sched.c's poly_buffer_id */

PolyUOp *poly_buffer_var(
    PolyCtx *ctx,
    PolyDType dt,
    PolyUOp *batch_var,
    const int64_t *inner_dims,
    int n_inner
) {
  assert(batch_var->op == POLY_OP_DEFINE_VAR);
  assert(n_inner >= 0 && n_inner < POLY_MAX_DIMS);
  int64_t max_val = batch_var->arg.define_var.max_val;
  int64_t alloc = max_val;
  /* src[0] = UNIQUE (prevent CSE), src[1] = DEFINE_VAR, src[2..] = CONST inner dims */
  int n_src = 2 + n_inner;
  PolyUOp *src[POLY_MAX_DIMS + 2];
  src[0] = poly_uop0(ctx, POLY_OP_UNIQUE, POLY_VOID, poly_arg_int(poly_dyn_buffer_id++));
  src[1] = batch_var;
  for (int i = 0; i < n_inner; i++) {
    alloc *= inner_dims[i];
    src[2 + i] = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(inner_dims[i]));
  }
  return poly_uop(ctx, POLY_OP_BUFFER, dt, src, n_src, poly_arg_int(alloc));
}

/* Forward declarations removed -- now in tensor.h/tensor.c */

/* shape helpers moved to tensor.c */

/* make_const helpers moved to tensor.c */

/* All composed elementwise/reduction/creation ops moved to tensor.c */

/* Shared helpers */

/* (Remaining code: collect_ordered_buffers, ptr_hash/eq, structural hash/eq,
 * realize pipeline, compiled step, WASM rendering, debug helpers,
 * causal_mask -- all stay here) */

/* --- All composed ops (poly_exp through poly_cross_entropy, poly_einsum,
 *     poly_rearrange, poly_gather, and all v2 wrappers) have been moved
 *     to tensor.c. See tensor.h for declarations. --- */

/* Shared helpers */

/* POLY_MAX_REALIZE_BUFS defined in frontend_internal.h */

/* Reconstruct the buffer-to-PARAM ordering that poly_schedule() uses:
 * 1. Output buffers (STORE targets in SINK source order)
 * 2. Remaining input buffers (toposort encounter order) */
int poly_collect_ordered_buffers(
    PolyCtx *ctx,
    PolyUOp *tensor_sink,
    PolyUOp **ordered,
    int max_bufs
) {
  int n = 0;

  /* Output buffers first */
  for (int i = 0; i < tensor_sink->n_src; i++) {
    PolyUOp *store = tensor_sink->src[i];
    if (store->op == POLY_OP_STORE && store->n_src >= 1 && store->src[0]->op == POLY_OP_BUFFER) {
      PolyUOp *buf = store->src[0];
      /* Dedup */
      bool dup = false;
      for (int j = 0; j < n; j++) {
        if (ordered[j] == buf) {
          dup = true;
          break;
        }
      }
      if (!dup && n < max_bufs) ordered[n++] = buf;
    }
  }

  /* Input buffers in toposort order */
  int n_topo;
  PolyUOp **topo = poly_toposort(ctx, tensor_sink, &n_topo);
  for (int i = 0; i < n_topo; i++) {
    if (topo[i]->op == POLY_OP_BUFFER) {
      bool dup = false;
      for (int j = 0; j < n; j++) {
        if (ordered[j] == topo[i]) {
          dup = true;
          break;
        }
      }
      if (!dup && n < max_bufs) ordered[n++] = topo[i];
    }
  }

  return n;
}

/* Pointer hash/eq helpers (used by kernel cache and CPU realize) */

bool poly_ptr_eq(const void *a, const void *b) {
  return a == b;
}
uint32_t poly_ptr_hash(const void *p) {
  uintptr_t v = (uintptr_t)p;
  return (uint32_t)(v ^ (v >> 16) ^ (sizeof(v) > 4 ? (uint32_t)(v >> 32) : 0));
}

/* Phase E: const-registry was deleted from tensor.c, so the corresponding
 * cleanup hook here no longer needs to drain g_const_bindings. */

#ifndef __EMSCRIPTEN__
static void realize_cache_purge(PolyCtx *ctx); /* defined below */

void poly_frontend_ctx_cleanup(PolyCtx *ctx) {
  if (!ctx) return;
  realize_cache_purge(ctx);
}
#endif

/* Structural hash/eq for kernel cache *
 * The kernel cache needs to match computations that are structurally
 * identical but use different BUFFER UOp instances (e.g., each training
 * step creates new buffers).  We hash/compare the computation DAG
 * structure: ops, dtypes, args, and connectivity — treating BUFFER
 * nodes as positional placeholders (first encountered = 0, etc.).
 */

/* POLY_MAX_STRUCT_NODES defined in frontend_internal.h */

typedef struct {
  PolyUOp *uop;
  uint32_t hash;
} StructVisited;

typedef struct {
  PolyUOp *uop;
  int id;
} StructBuf;

typedef struct {
  StructVisited visited[POLY_MAX_STRUCT_NODES];
  int n_visited;
  StructBuf bufs[POLY_MAX_REALIZE_BUFS];
  int n_bufs;
} StructHashCtx;

static uint32_t struct_hash_impl(PolyUOp *u, StructHashCtx *ctx) {
  /* Check if already visited (handles DAG sharing) */
  for (int i = 0; i < ctx->n_visited; i++) {
    if (ctx->visited[i].uop == u) return ctx->visited[i].hash;
  }

  uint32_t h = 0x811c9dc5; /* FNV-1a offset basis */

  if (u->op == POLY_OP_BUFFER) {
    /* BUFFER nodes: use positional ID instead of pointer identity */
    int buf_id = -1;
    for (int i = 0; i < ctx->n_bufs; i++) {
      if (ctx->bufs[i].uop == u) {
        buf_id = ctx->bufs[i].id;
        break;
      }
    }
    if (buf_id < 0 && ctx->n_bufs < POLY_MAX_REALIZE_BUFS) {
      buf_id = ctx->n_bufs;
      ctx->bufs[ctx->n_bufs].uop = u;
      ctx->bufs[ctx->n_bufs].id = buf_id;
      ctx->n_bufs++;
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

  /* Memoize */
  if (ctx->n_visited < POLY_MAX_STRUCT_NODES) {
    ctx->visited[ctx->n_visited].uop = u;
    ctx->visited[ctx->n_visited].hash = h;
    ctx->n_visited++;
  }
  return h;
}

uint32_t poly_structural_hash(PolyUOp *u) {
  StructHashCtx *ctx = calloc(1, sizeof(StructHashCtx));
  uint32_t h = struct_hash_impl(u, ctx);
  free(ctx);
  return h;
}

/* Structural equality */

typedef struct {
  PolyUOp *a[POLY_MAX_REALIZE_BUFS];
  PolyUOp *b[POLY_MAX_REALIZE_BUFS];
  int n;
} BufPairs;

typedef struct {
  PolyUOp *a[POLY_MAX_STRUCT_NODES];
  PolyUOp *b[POLY_MAX_STRUCT_NODES];
  int n;
} EqVisited;

static bool struct_eq_impl(PolyUOp *a, PolyUOp *b, BufPairs *bp, EqVisited *ev) {
  if (a == b) return true;
  if (!a || !b) return false;

  /* Check if this pair already visited (DAG sharing) */
  for (int i = 0; i < ev->n; i++) {
    if (ev->a[i] == a) return ev->b[i] == b;
    if (ev->b[i] == b) return false;
  }

  /* Mark visited */
  if (ev->n < POLY_MAX_STRUCT_NODES) {
    ev->a[ev->n] = a;
    ev->b[ev->n] = b;
    ev->n++;
  }

  /* Both BUFFER? Track correspondence */
  if (a->op == POLY_OP_BUFFER && b->op == POLY_OP_BUFFER) {
    if (!poly_dtype_eq(a->dtype, b->dtype)) return false;
    if (!poly_arg_eq(a->arg, b->arg)) return false;
    /* Check existing mapping */
    for (int i = 0; i < bp->n; i++) {
      if (bp->a[i] == a) return bp->b[i] == b;
      if (bp->b[i] == b) return false;
    }
    /* New pair */
    if (bp->n < POLY_MAX_REALIZE_BUFS) {
      bp->a[bp->n] = a;
      bp->b[bp->n] = b;
      bp->n++;
    }
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
  BufPairs bp = {.n = 0};
  EqVisited ev = {.n = 0};
  return struct_eq_impl((PolyUOp *)a, (PolyUOp *)b, &bp, &ev);
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
  if (!u) return;
  for (int i = 0; i < *n_visited; i++)
    if (visited[i] == u) return;
  if (*n_visited < POLY_MAX_STRUCT_NODES) visited[(*n_visited)++] = u;

  if (u->op == POLY_OP_BUFFER) {
    if (*n_bufs < POLY_MAX_REALIZE_BUFS) buf_order[*n_bufs] = u;
    (*n_bufs)++; /* always count, even past capacity */
    return;
  }
  for (int i = 0; i < u->n_src; i++)
    poly_collect_buf_order(u->src[i], buf_order, n_bufs, visited, n_visited);
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

    if (u->n_src < 0 || u->n_src > 64) {
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

/* Strip BIND values from graph, extracting {DEFINE_VAR → value} pairs.
 * Port of tinygrad's strip_bind in pm_pre_sched_cache.
 * Returns rewritten sink with BIND(DEFINE_VAR, CONST) → DEFINE_VAR.
 * Populates out_vals[0..n_out-1] with extracted bindings.
 * Also remaps buf_bindings[].buffer pointers to their rewritten equivalents
 * (so binding lookup works after graph rewrite). */
PolyUOp *poly_strip_bind_values(
    PolyCtx *ctx,
    PolyUOp *sink,
    PolyVarBinding *out_vals,
    int *n_out,
    int max_out,
    PolyBufferBinding *buf_bindings,
    int n_buf_bindings
) {
  int n_uops;
  PolyUOp **topo = poly_toposort(ctx, sink, &n_uops);
  if (!topo) {
    *n_out = 0;
    return sink;
  }

  /* Check if any BIND nodes exist */
  bool has_bind = false;
  for (int i = 0; i < n_uops; i++) {
    if (topo[i]->op == POLY_OP_BIND) {
      has_bind = true;
      break;
    }
  }
  if (!has_bind) {
    *n_out = 0;
    return sink;
  }

  /* Bottom-up rewrite: BIND(DEFINE_VAR, CONST) → DEFINE_VAR */
  PolyMap *rmap = poly_map_new(n_uops * 2);
  int nv = 0;
  for (int i = 0; i < n_uops; i++) {
    PolyUOp *u = topo[i];
    if (u->op == POLY_OP_BIND && u->n_src >= 2 && u->src[0]->op == POLY_OP_DEFINE_VAR &&
        u->src[1]->op == POLY_OP_CONST) {
      /* Extract var → value binding */
      if (nv < max_out) {
        out_vals[nv].var = u->src[0];
        out_vals[nv].value = (int32_t)u->src[1]->arg.i;
        nv++;
      }
      /* Replace BIND with DEFINE_VAR */
      poly_map_set(rmap, poly_ptr_hash(u), u, u->src[0], poly_ptr_eq);
      continue;
    }
    /* Rebuild node with remapped sources if any source changed */
    bool changed = false;
    PolyUOp *new_src[64];
    for (int s = 0; s < u->n_src && s < 64; s++) {
      PolyUOp *ms = poly_map_get(rmap, poly_ptr_hash(u->src[s]), u->src[s], poly_ptr_eq);
      new_src[s] = ms ? ms : u->src[s];
      if (new_src[s] != u->src[s]) changed = true;
    }
    if (changed) {
      PolyUOp *rebuilt = poly_uop(ctx, u->op, u->dtype, new_src, u->n_src, u->arg);
      poly_map_set(rmap, poly_ptr_hash(u), u, rebuilt, poly_ptr_eq);
    }
  }

  /* Get the rewritten sink */
  PolyUOp *result = poly_map_get(rmap, poly_ptr_hash(sink), sink, poly_ptr_eq);
  if (!result) result = sink;

  /* Remap buffer binding pointers to rewritten BUFFERs */
  for (int j = 0; j < n_buf_bindings; j++) {
    PolyUOp *remapped = poly_map_get(
        rmap, poly_ptr_hash(buf_bindings[j].buffer), buf_bindings[j].buffer, poly_ptr_eq
    );
    if (remapped) buf_bindings[j].buffer = remapped;
  }

  poly_map_destroy(rmap);
  /* topo is arena-allocated, no free needed */
  *n_out = nv;
  return result;
}

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

#ifndef __EMSCRIPTEN__

static int realize_counter = 0;

/* Compiled program cache *
 * Caches compiled PolyProgram* (CPU) and PolyCudaProgram* (CUDA) keyed by
 * the structural hash of the tensor-level SINK.  Avoids re-running
 * cc/NVRTC on every realize() call for the same computation.
 */

#define PROG_CACHE_CAP 512

typedef struct {
  uint32_t hash;
  PolyProgram *prog;
} CpuCacheEntry;

static CpuCacheEntry cpu_prog_cache[PROG_CACHE_CAP];
static int cpu_prog_cache_n = 0;

static PolyProgram *cpu_cache_get(uint32_t h) {
  for (int i = 0; i < cpu_prog_cache_n; i++)
    if (cpu_prog_cache[i].hash == h) return cpu_prog_cache[i].prog;
  return NULL;
}

void poly_cpu_cache_flush(void) {
  for (int i = 0; i < cpu_prog_cache_n; i++)
    poly_program_destroy(cpu_prog_cache[i].prog);
  cpu_prog_cache_n = 0;
}

static void cpu_cache_put(uint32_t h, PolyProgram *prog) {
  if (cpu_prog_cache_n < PROG_CACHE_CAP) {
    cpu_prog_cache[cpu_prog_cache_n].hash = h;
    cpu_prog_cache[cpu_prog_cache_n].prog = prog;
    cpu_prog_cache_n++;
  } else {
    /* Cache full: destroy program immediately to avoid leak */
    poly_program_destroy(prog);
  }
}

/* Legacy realize_impl, sched_cache, and compile_and_run were here.
 * Removed: all execution now routes through exec_plan
 * (poly_schedule_for → poly_compile_schedule → poly_compiled_plan_run).
 * DEFINE_VAR support is handled by exec_plan's var_uops mechanism. */

/* poly_realize and helpers moved below #endif to be available in all builds */

/* Compiled Step *
 * PolyStep is a thin wrapper over the exec_plan infrastructure.
 * poly_compile_step() schedules and compiles via poly_schedule_for() +
 * poly_compile_schedule(). poly_step_run() delegates to
 * poly_compiled_plan_run(). Buffer metadata and pre-strip buf_order
 * are kept for backward-compatible query APIs.
 */

typedef struct {
  PolyStepBufRole role;
  PolyDType dtype;
  int64_t numel;
  int64_t nbytes;
} PolyStepBufMeta;

struct PolyStep {
  PolyCtx *ctx;
  PolySchedule *schedule; /* owned, from poly_schedule_for */
  PolyCompiledPlan *plan; /* owned, from poly_compile_schedule */

  /* Pre-strip buffer ordering for callers that bind by original UOp pointer */
  int n_bufs;
  PolyUOp **buf_order; /* malloc'd [n_bufs] */

  /* Query API metadata */
  PolyStepBufMeta *buf_meta; /* malloc'd [n_buf_slots] */
  int n_total_buf_meta;

  uint32_t graph_hash;
};

PolyStep *poly_compile_step(PolyCtx *ctx, PolyUOp *tensor_sink) {
  if (!tensor_sink || tensor_sink->op != POLY_OP_SINK) {
    fprintf(stderr, "polygrad: compile_step: expected SINK\n");
    return NULL;
  }

  /* Schedule via exec_plan (handles BIND stripping, buffer ordering,
   * intermediate allocation, kernel scheduling internally) */
  PolySchedule *sched = poly_schedule_for(ctx, tensor_sink, POLY_MODE_CALL);
  if (!sched) return NULL;

  /* Compile for CPU via exec_plan backend vtable */
  PolyCompiledPlan *plan = poly_compile_schedule(ctx, sched, POLY_DEVICE_CPU);
  if (!plan) {
    poly_schedule_free(sched);
    return NULL;
  }

  /* Allocate step wrapper */
  PolyStep *step = calloc(1, sizeof(PolyStep));
  if (!step) {
    poly_compiled_plan_free(plan);
    poly_schedule_free(sched);
    return NULL;
  }
  step->ctx = ctx;
  step->schedule = sched;
  step->plan = plan;
  step->graph_hash = sched->graph_hash;

  /* Build pre-strip buf_order from schedule's external buf_slots.
   * poly_schedule_for stores pre-strip buf_uop pointers in external slots. */
  int n_external = 0;
  for (int i = 0; i < sched->n_buf_slots; i++)
    if (!sched->buf_slots[i].is_intermediate) n_external++;

  step->n_bufs = n_external;
  if (n_external > 0) {
    step->buf_order = malloc((size_t)n_external * sizeof(PolyUOp *));
    int idx = 0;
    for (int i = 0; i < sched->n_buf_slots; i++)
      if (!sched->buf_slots[i].is_intermediate)
        step->buf_order[idx++] = sched->buf_slots[i].buf_uop;
  }

  /* Build buffer metadata from schedule buf_slots */
  step->n_total_buf_meta = sched->n_buf_slots;
  if (sched->n_buf_slots > 0) {
    step->buf_meta = calloc((size_t)sched->n_buf_slots, sizeof(PolyStepBufMeta));
    for (int i = 0; i < sched->n_buf_slots; i++) {
      PolyScheduleBufSlot *slot = &sched->buf_slots[i];
      PolyStepBufMeta *m = &step->buf_meta[i];
      if (slot->is_intermediate) {
        m->role = POLY_STEP_BUF_TEMP;
      } else {
        /* Determine role: output if it's a SINK source's STORE target */
        PolyUOp *buf = slot->buf_uop;
        bool is_output = false;
        if (tensor_sink->op == POLY_OP_SINK) {
          for (int s = 0; s < tensor_sink->n_src; s++) {
            PolyUOp *st = tensor_sink->src[s];
            if (st && st->op == POLY_OP_STORE && st->n_src > 0 && st->src[0] == buf) {
              is_output = true;
              break;
            }
          }
        }
        if (is_output)
          m->role = POLY_STEP_BUF_OUTPUT;
        else
          m->role = POLY_STEP_BUF_INPUT;
        /* Phase E: POLY_STEP_BUF_CONSTANT removed; const-registry deleted. */
      }
      m->dtype = slot->dtype;
      m->numel = slot->numel;
      m->nbytes = slot->nbytes;
    }
  }

  return step;
}

static int64_t static_numel_of_uop(PolyCtx *ctx, PolyUOp *u) {
  PolyShape s = poly_uop_shape(ctx, u);
  if (s.ndim < 0) {
    if (s.dims) free(s.dims);
    return -1;
  }
  if (s.ndim == 0) return 1;
  int64_t n = 1;
  for (int i = 0; i < s.ndim; i++) {
    if (s.dims[i] <= 0) {
      free(s.dims);
      return -1;
    }
    if (n > INT64_MAX / s.dims[i]) {
      free(s.dims);
      return -1;
    }
    n *= s.dims[i];
  }
  free(s.dims);
  return n;
}

PolyStep *poly_compile_value_and_grad(
    PolyCtx *ctx,
    PolyUOp *loss,
    PolyUOp **params,
    int n_params,
    int *out_loss_buf_idx,
    int *out_grad_buf_idxs
) {
  if (!ctx || !loss || !params || n_params <= 0 || !out_loss_buf_idx || !out_grad_buf_idxs) {
    fprintf(stderr, "polygrad: compile_value_and_grad: invalid arguments\n");
    return NULL;
  }

  /* Build gradients in one reverse pass. */
  PolyUOp **grads = calloc((size_t)n_params, sizeof(PolyUOp *));
  if (!grads) return NULL;
  if (poly_grad_many(ctx, loss, NULL, params, n_params, grads) != 0) {
    fprintf(stderr, "polygrad: compile_value_and_grad: poly_grad_many failed\n");
    free(grads);
    return NULL;
  }

  /* Build output stores:
   *   store[0]   -> scalar loss (as 1-element vector)
   *   store[i+1] -> flattened grad[i] */
  int n_stores = n_params + 1;
  PolyUOp **stores = calloc((size_t)n_stores, sizeof(PolyUOp *));
  PolyUOp **grad_out_bufs = calloc((size_t)n_params, sizeof(PolyUOp *));
  if (!stores || !grad_out_bufs) {
    free(stores);
    free(grad_out_bufs);
    free(grads);
    return NULL;
  }

  PolyDType out_dt = poly_dtype_scalar(loss->dtype);
  if (!poly_dtype_is_float(out_dt)) out_dt = POLY_FLOAT32;

  /* Loss output buffer (1 element). */
  PolyUOp *loss_buf = poly_buffer(ctx, out_dt, 1);
  PolyUOp *loss_flat = loss;
  if (static_numel_of_uop(ctx, loss) != 1) {
    int64_t one_shape[1] = {1};
    loss_flat = poly_reshape(ctx, loss, one_shape, 1);
  }
  stores[0] = poly_store_val(ctx, loss_buf, loss_flat);

  for (int i = 0; i < n_params; i++) {
    int64_t numel = static_numel_of_uop(ctx, grads[i]);
    if (numel <= 0) {
      fprintf(stderr, "polygrad: compile_value_and_grad: grad[%d] has unknown/invalid shape\n", i);
      free(stores);
      free(grad_out_bufs);
      free(grads);
      return NULL;
    }
    PolyDType gdt = poly_dtype_scalar(grads[i]->dtype);
    if (!poly_dtype_is_float(gdt)) gdt = POLY_FLOAT32;
    PolyUOp *gbuf = poly_buffer(ctx, gdt, numel);
    grad_out_bufs[i] = gbuf;
    PolyUOp *gflat = grads[i];
    PolyShape gs = poly_uop_shape(ctx, grads[i]);
    if (gs.ndim != 1 || (gs.ndim == 1 && gs.dims[0] != numel)) {
      int64_t flat_shape[1] = {numel};
      gflat = poly_reshape(ctx, grads[i], flat_shape, 1);
    }
    if (gs.dims) free(gs.dims);
    stores[i + 1] = poly_store_val(ctx, gbuf, gflat);
  }

  PolyUOp *sink = poly_sink_n(ctx, stores, n_stores);
  PolyStep *step = poly_compile_step(ctx, sink);

  free(stores);
  free(grads);

  if (!step) {
    free(grad_out_bufs);
    return NULL;
  }

  int loss_idx = poly_find_buf_position(loss_buf, step->buf_order, step->n_bufs);
  if (loss_idx < 0) {
    fprintf(stderr, "polygrad: compile_value_and_grad: loss buffer index not found\n");
    free(grad_out_bufs);
    poly_step_destroy(step);
    return NULL;
  }
  *out_loss_buf_idx = loss_idx;
  for (int i = 0; i < n_params; i++) {
    int gi = poly_find_buf_position(grad_out_bufs[i], step->buf_order, step->n_bufs);
    if (gi < 0) {
      fprintf(stderr, "polygrad: compile_value_and_grad: grad buffer %d index not found\n", i);
      free(grad_out_bufs);
      poly_step_destroy(step);
      return NULL;
    }
    out_grad_buf_idxs[i] = gi;
  }
  free(grad_out_bufs);
  return step;
}

int poly_step_run_ex(
    PolyStep *step,
    PolyBufferBinding *bindings,
    int n_bindings,
    PolyVarBinding *var_bindings,
    int n_var_bindings
) {
  if (!step || !step->schedule || !step->plan) return -1;

  /* Map caller's bindings (keyed by pre-strip buf_uop pointers) to
   * schedule buf_slot indices, then delegate to poly_compiled_plan_run. */
  PolySchedule *sched = step->schedule;
  int n_slots = sched->n_buf_slots;
  void **slot_data = calloc((size_t)(n_slots > 0 ? n_slots : 1), sizeof(void *));
  if (!slot_data) return -1;

  /* Fill external slots from bindings */
  for (int j = 0; j < n_bindings; j++) {
    int pos = poly_find_buf_position(bindings[j].buffer, step->buf_order, step->n_bufs);
    if (pos >= 0 && pos < n_slots) slot_data[pos] = bindings[j].handle.ptr;
  }

  /* Phase E: const-registry autobind path removed. Unbound external slots
   * are now an explicit caller error rather than being silently filled
   * from g_const_bindings. */

  int ret = poly_compiled_plan_run(step->plan, slot_data, n_slots, var_bindings, n_var_bindings);
  free(slot_data);
  return ret;
}

int poly_step_run(PolyStep *step, PolyBufferBinding *bindings, int n_bindings) {
  return poly_step_run_ex(step, bindings, n_bindings, NULL, 0);
}

int poly_step_run_indexed_ex(
    PolyStep *step,
    void **buffer_data,
    int n_buffers,
    PolyVarBinding *var_bindings,
    int n_var_bindings
) {
  if (!step || !buffer_data || n_buffers <= 0) return -1;
  int n = (n_buffers < step->n_bufs) ? n_buffers : step->n_bufs;
  PolyBufferBinding *bindings = calloc((size_t)n, sizeof(PolyBufferBinding));
  if (!bindings) return -1;
  int nb = 0;
  for (int i = 0; i < n; i++) {
    if (!buffer_data[i]) continue;
    bindings[nb].buffer = step->buf_order[i];
    bindings[nb].handle = (PolyBufferHandle){buffer_data[i], 0, POLY_DEVICE_CPU, false};
    nb++;
  }
  int ret = poly_step_run_ex(step, bindings, nb, var_bindings, n_var_bindings);
  free(bindings);
  return ret;
}

int poly_step_run_indexed(PolyStep *step, void **buffer_data, int n_buffers) {
  return poly_step_run_indexed_ex(step, buffer_data, n_buffers, NULL, 0);
}

void poly_step_destroy(PolyStep *step) {
  if (!step) return;
  if (step->plan) poly_compiled_plan_free(step->plan);
  if (step->schedule) poly_schedule_free(step->schedule);
  free(step->buf_order);
  free(step->buf_meta);
  free(step);
}

int poly_step_n_kernels(const PolyStep *step) {
  return (step && step->schedule) ? step->schedule->n_items : 0;
}
int poly_step_n_intermediates(const PolyStep *step) {
  if (!step || !step->schedule) return 0;
  int n = 0;
  for (int i = 0; i < step->schedule->n_buf_slots; i++)
    if (step->schedule->buf_slots[i].is_intermediate) n++;
  return n;
}
int poly_step_n_buffers(const PolyStep *step) {
  return step ? step->n_total_buf_meta : 0;
}
int poly_step_n_bindable_buffers(const PolyStep *step) {
  return step ? step->n_bufs : 0;
}

int poly_step_buffer_info(const PolyStep *step, int idx, PolyStepBufferInfo *out) {
  if (!step || !out || idx < 0 || idx >= step->n_total_buf_meta) return -1;
  PolyStepBufMeta *m = &step->buf_meta[idx];
  out->version = POLY_STEP_BUFFER_INFO_VERSION;
  out->index = idx;
  out->role = m->role;
  out->dtype = m->dtype;
  out->numel = m->numel;
  out->nbytes = m->nbytes;
  return 0;
}

PolyUOp *poly_step_buf_uop(const PolyStep *step, int idx) {
  if (!step || idx < 0 || idx >= step->n_bufs) return NULL;
  return step->buf_order[idx];
}

#endif /* !__EMSCRIPTEN__ -- end CPU-only realize/compile block */

/* Exec plan functions (poly_schedule_for, poly_compile_schedule, etc.)
 * have been moved to exec_plan.c for cross-build compilation. */

/* ══════════════════════════════════════════════════════════════════════ */
/*  Per-context caches for poly_realize                                  */
/*  Schedule cache: keyed by (ctx, graph_hash, mode)                     */
/*  Compiled plan cache: keyed by (ctx, graph_hash, mode, device)        */
/* ══════════════════════════════════════════════════════════════════════ */

#define REALIZE_SCHED_CACHE_CAP 128
#define REALIZE_PLAN_CACHE_CAP 256

typedef struct {
  PolyCtx *ctx;
  PolyUOp *sink; /* identity key (CSE-deduped pointer) */
  uint32_t hash;
  PolyCompileMode mode;
  PolySchedule *sched;
} RealizeSchedCacheEntry;

typedef struct {
  PolyCtx *ctx;
  PolyUOp *sink; /* identity key */
  uint32_t hash;
  PolyCompileMode mode;
  PolyDeviceId device;
  int8_t optimize; /* codegen optimization level */
  int8_t devectorize; /* devectorize level (-1, 0, 1) */
  int8_t tc_opt; /* POLY_TC_OPT env (TC strictness level) */
  int8_t use_tc; /* POLY_USE_TC env (0=off, 1=full, 2=shape-only) */
  PolyCompiledPlan *plan;
} RealizePlanCacheEntry;

static RealizeSchedCacheEntry r_sched_cache[REALIZE_SCHED_CACHE_CAP];
static int r_sched_n = 0;

static RealizePlanCacheEntry r_plan_cache[REALIZE_PLAN_CACHE_CAP];
static int r_plan_n = 0;

static PolySchedule *r_sched_get(PolyCtx *ctx, PolyUOp *sink, uint32_t h, PolyCompileMode m) {
  for (int i = 0; i < r_sched_n; i++)
    if (r_sched_cache[i].ctx == ctx && r_sched_cache[i].sink == sink &&
        r_sched_cache[i].hash == h && r_sched_cache[i].mode == m)
      return r_sched_cache[i].sched;
  return NULL;
}

static void r_sched_put(
    PolyCtx *ctx,
    PolyUOp *sink,
    uint32_t h,
    PolyCompileMode m,
    PolySchedule *s
) {
  if (r_sched_n < REALIZE_SCHED_CACHE_CAP)
    r_sched_cache[r_sched_n++] = (RealizeSchedCacheEntry){ctx, sink, h, m, s};
}

static PolyCompiledPlan *r_plan_get(
    PolyCtx *ctx,
    PolyUOp *sink,
    uint32_t h,
    PolyCompileMode m,
    PolyDeviceId d,
    int8_t opt,
    int8_t devec,
    int8_t tc_opt,
    int8_t use_tc
) {
  for (int i = 0; i < r_plan_n; i++)
    if (r_plan_cache[i].ctx == ctx && r_plan_cache[i].sink == sink && r_plan_cache[i].hash == h &&
        r_plan_cache[i].mode == m && r_plan_cache[i].device == d &&
        r_plan_cache[i].optimize == opt && r_plan_cache[i].devectorize == devec &&
        r_plan_cache[i].tc_opt == tc_opt && r_plan_cache[i].use_tc == use_tc)
      return r_plan_cache[i].plan;
  return NULL;
}

static void r_plan_put(
    PolyCtx *ctx,
    PolyUOp *sink,
    uint32_t h,
    PolyCompileMode m,
    PolyDeviceId d,
    int8_t opt,
    int8_t devec,
    int8_t tc_opt,
    int8_t use_tc,
    PolyCompiledPlan *p
) {
  if (r_plan_n < REALIZE_PLAN_CACHE_CAP)
    r_plan_cache[r_plan_n++] =
        (RealizePlanCacheEntry){ctx, sink, h, m, d, opt, devec, tc_opt, use_tc, p};
}

/* Read current codegen optimization settings from env vars.
 * Returns (optimize, devectorize) as int8_t pair for cache key. */
static int8_t r_env_optimize(void) {
  const char *v = getenv("POLY_OPTIMIZE");
  return (int8_t)(v && v[0] != '\0' && v[0] != '0');
}
static int8_t r_env_devectorize(void) {
  const char *v = getenv("POLY_DEVECTORIZE");
  return v && v[0] != '\0' ? (int8_t)atoi(v) : 0;
}
static int8_t r_env_tc_opt(void) {
  const char *v = getenv("POLY_TC_OPT");
  return v && v[0] != '\0' ? (int8_t)atoi(v) : 0;
}
static int8_t r_env_use_tc(void) {
  const char *v = getenv("POLY_USE_TC");
  return v && v[0] != '\0' ? (int8_t)atoi(v) : 1; /* default: TC enabled */
}

/* Purge realize caches for a destroyed context.
 * Called from poly_frontend_ctx_cleanup (defined per-build below). */
static void realize_cache_purge(PolyCtx *ctx) {
  for (int i = r_plan_n - 1; i >= 0; i--) {
    if (r_plan_cache[i].ctx == ctx) {
      poly_compiled_plan_free(r_plan_cache[i].plan);
      r_plan_cache[i] = r_plan_cache[--r_plan_n];
    }
  }
  for (int i = r_sched_n - 1; i >= 0; i--) {
    if (r_sched_cache[i].ctx == ctx) {
      poly_schedule_free(r_sched_cache[i].sched);
      r_sched_cache[i] = r_sched_cache[--r_sched_n];
    }
  }
}

#ifdef __EMSCRIPTEN__
/* WASM builds: the native poly_frontend_ctx_cleanup is inside #ifndef __EMSCRIPTEN__.
 * Provide one for WASM that purges the realize caches. */
void poly_frontend_ctx_cleanup(PolyCtx *ctx) {
  if (!ctx) return;
  realize_cache_purge(ctx);
}
#endif

/* ══════════════════════════════════════════════════════════════════════ */
/*  poly_realize: available in ALL builds (native + WASM)                */
/* ══════════════════════════════════════════════════════════════════════ */

static PolyDeviceId infer_device(PolyBufferBinding *bindings, int n) {
  /* Check if any binding is on a non-CPU device */
  for (int i = 0; i < n; i++)
    if (bindings[i].handle.domain != POLY_DEVICE_CPU &&
        bindings[i].handle.domain != POLY_DEVICE_AUTO)
      return bindings[i].handle.domain;
  /* POLY_DEVICE=cpu|cuda|hip|x64|interp — unified backend selector.
   * Strict: if set but unavailable, warn (don't silently fall back to CPU). */
  const char *dev_env = getenv("POLY_DEVICE");
  if (dev_env && dev_env[0]) {
    if (strcmp(dev_env, "cpu") == 0) return POLY_DEVICE_CPU;
    if (strcmp(dev_env, "interp") == 0) return POLY_DEVICE_INTERP;
#ifdef POLY_HAS_CUDA
    if (strcmp(dev_env, "cuda") == 0) {
      if (poly_cuda_available()) return POLY_DEVICE_CUDA;
      fprintf(stderr, "polygrad: POLY_DEVICE=cuda but CUDA not available\n");
      return POLY_DEVICE_CPU;
    }
#endif
#ifdef POLY_HAS_HIP
    if (strcmp(dev_env, "hip") == 0) {
      if (poly_hip_available()) return POLY_DEVICE_HIP;
      fprintf(stderr, "polygrad: POLY_DEVICE=hip but HIP not available\n");
      return POLY_DEVICE_CPU;
    }
#endif
#ifdef POLY_HAS_X64
    if (strcmp(dev_env, "x64") == 0) return POLY_DEVICE_X64_JIT;
#endif
    if (strcmp(dev_env, "cpu") != 0)
      fprintf(stderr, "polygrad: unknown or unsupported POLY_DEVICE=%s, using CPU\n", dev_env);
    return POLY_DEVICE_CPU;
  }

  /* Legacy env vars (backward compat) */
#ifdef __EMSCRIPTEN__
  return POLY_DEVICE_WASM_JIT;
#else
#ifdef POLY_HAS_HIP
  {
    const char *hip_env = getenv("POLY_HIP");
    if (hip_env && hip_env[0] == '1' && poly_hip_available()) return POLY_DEVICE_HIP;
  }
#endif
#ifdef POLY_HAS_X64
  {
    const char *x64_env = getenv("POLY_X64");
    if (x64_env && x64_env[0] == '1') return POLY_DEVICE_X64_JIT;
  }
#endif
  return POLY_DEVICE_CPU;
#endif
}

static void **build_slot_data_from_bindings(
    PolyCtx *ctx,
    const PolySchedule *sched,
    PolyBufferBinding *bindings,
    int n_bindings
) {
  void **slot_data = calloc((size_t)sched->n_buf_slots, sizeof(void *));
  if (!slot_data) return NULL;

  (void)ctx;
  for (int s = 0; s < sched->n_buf_slots; s++) {
    if (sched->buf_slots[s].is_intermediate) continue;
    PolyUOp *slot_uop = sched->buf_slots[s].buf_uop;
    for (int j = 0; j < n_bindings; j++) {
      if (bindings[j].buffer == slot_uop) {
        slot_data[s] = bindings[j].handle.ptr;
        break;
      }
    }
    /* Phase E: const-registry autobind path removed. Unbound external
     * slots now stay NULL and the caller-side error reporting picks them
     * up — they used to be silently filled from g_const_bindings. */
  }
  return slot_data;
}

PolyCompiledPlan *poly_get_plan(PolyCtx *ctx, PolyUOp *tensor_sink, PolyDeviceId device) {
  if (!tensor_sink || tensor_sink->op != POLY_OP_SINK) return NULL;

  uint32_t hash = poly_structural_hash(tensor_sink) ^ (POLY_SCHED_CACHE_VERSION * 2654435761u);

  PolySchedule *sched = r_sched_get(ctx, tensor_sink, hash, POLY_MODE_CALL);
  if (!sched) {
    sched = poly_schedule_for(ctx, tensor_sink, POLY_MODE_CALL);
    if (!sched) return NULL;
    r_sched_put(ctx, tensor_sink, hash, POLY_MODE_CALL, sched);
  }

  int8_t opt = r_env_optimize(), devec = r_env_devectorize();
  int8_t tc_opt = r_env_tc_opt(), use_tc = r_env_use_tc();
  PolyCompiledPlan *plan =
      r_plan_get(ctx, tensor_sink, hash, POLY_MODE_CALL, device, opt, devec, tc_opt, use_tc);
  if (!plan) {
    plan = poly_compile_schedule(ctx, sched, device);
    if (!plan) return NULL;
    r_plan_put(ctx, tensor_sink, hash, POLY_MODE_CALL, device, opt, devec, tc_opt, use_tc, plan);
  }

  return plan;
}

/* Check if device needs device-memory but all bindings are host-domain.
 * Returns true for GPU backends (CUDA, HIP) when bindings are CPU/AUTO. */
static bool needs_device_migration(PolyDeviceId device, PolyBufferBinding *bindings, int n) {
  if (device != POLY_DEVICE_CUDA && device != POLY_DEVICE_HIP) return false;
  for (int i = 0; i < n; i++)
    if (bindings[i].handle.domain == device) return false; /* already on device */
  return true;
}

/* Shared device migration helpers */

/* Allocate device memory for user bindings + const-registry buffers.
 * Returns 0 on success, -1 on error. On success, *out_dev and *out_total
 * are set; caller must call unmigrate_from_device to clean up. */
static int migrate_to_device(
    PolyCtx *ctx,
    PolyUOp *tensor_sink,
    PolyBufferBinding *bindings,
    int n_bindings,
    PolyDeviceId device,
    PolyBufferBinding **out_dev,
    int *out_total
) {
  const PolyBackendDesc *be = poly_backend_get(device);
  if (!be) return -1;
  const PolyAllocator *alloc = be->get_allocator();

  PolyBufferBinding *dev = malloc((size_t)n_bindings * sizeof(PolyBufferBinding));
  if (!dev) return -1;

  /* Alloc + upload user bindings */
  for (int i = 0; i < n_bindings; i++) {
    PolyUOp *buf = bindings[i].buffer;
    size_t nbytes = (size_t)buf->arg.i * poly_dtype_itemsize(poly_dtype_scalar(buf->dtype));
    if (nbytes == 0) nbytes = sizeof(float);
    void *dptr = alloc->alloc(nbytes, alloc->dev_ctx);
    if (!dptr) {
      for (int j = 0; j < i; j++)
        alloc->free(dev[j].handle.ptr, alloc->dev_ctx);
      free(dev);
      return -1;
    }
    if (bindings[i].handle.ptr)
      alloc->copy_in(dptr, bindings[i].handle.ptr, nbytes, alloc->dev_ctx);
    dev[i].buffer = buf;
    dev[i].handle = (PolyBufferHandle){dptr, nbytes, device, true};
  }

  /* Phase E: the const-registry buffer migration block (formerly the
   * migrated_consts[64] workaround) was removed entirely. poly_arange /
   * poly_eye / poly_full / poly_tril / poly_triu / poly_rand are pure
   * UOps now and never produce host-backed const buffers, so there is
   * nothing to migrate to device. tensor_sink is unused after this
   * deletion (kept in the signature for ABI stability). */
  (void)tensor_sink;

  *out_dev = dev;
  *out_total = n_bindings;
  return 0;
}

/* Readback user buffers to host, free all device memory, free dev array. */
static void unmigrate_from_device(
    PolyBufferBinding *host_bindings,
    int n_user,
    PolyBufferBinding *dev,
    int n_total,
    PolyDeviceId device,
    bool readback
) {
  const PolyBackendDesc *be = poly_backend_get(device);
  if (!be) {
    free(dev);
    return;
  }
  const PolyAllocator *alloc = be->get_allocator();

  if (readback) {
    for (int i = 0; i < n_user; i++) {
      if (!host_bindings[i].handle.ptr) continue;
      alloc->copy_out(
          host_bindings[i].handle.ptr, dev[i].handle.ptr, dev[i].handle.nbytes, alloc->dev_ctx
      );
    }
  }
  for (int i = 0; i < n_total; i++)
    alloc->free(dev[i].handle.ptr, alloc->dev_ctx);
  free(dev);
}

int poly_realize(PolyCtx *ctx, PolyUOp *tensor_sink, PolyBufferBinding *bindings, int n_bindings) {
  if (!tensor_sink || tensor_sink->op != POLY_OP_SINK) {
    fprintf(stderr, "polygrad: realize: expected SINK\n");
    return -1;
  }

  PolyDeviceId device = infer_device(bindings, n_bindings);

  /* Transparent host-to-device migration: when a GPU backend is selected
   * (via env var like POLY_HIP=1) but bindings are host pointers, we
   * auto-migrate: alloc device mem, copy in, execute, copy out, free.
   * This lets POLY_HIP=1 make test route all 500+ tests through HIP. */
  if (needs_device_migration(device, bindings, n_bindings)) {
    PolyBufferBinding *dev = NULL;
    int total = 0;
    if (migrate_to_device(ctx, tensor_sink, bindings, n_bindings, device, &dev, &total) != 0)
      return -1;
    int ret = poly_realize(ctx, tensor_sink, dev, total);
    unmigrate_from_device(bindings, n_bindings, dev, total, device, ret == 0);
    return ret;
  }

  uint32_t hash = poly_structural_hash(tensor_sink) ^ (POLY_SCHED_CACHE_VERSION * 2654435761u);

  /* Schedule cache: per-context, keyed by sink pointer (CSE identity) */
  PolySchedule *sched = r_sched_get(ctx, tensor_sink, hash, POLY_MODE_CALL);
  if (!sched) {
    sched = poly_schedule_for(ctx, tensor_sink, POLY_MODE_CALL);
    if (!sched) return -1;
    r_sched_put(ctx, tensor_sink, hash, POLY_MODE_CALL, sched);
  }

  /* Compiled plan cache: per-context + per-device + per-optimization + per-TC-mode */
  int8_t opt = r_env_optimize(), devec = r_env_devectorize();
  int8_t tc_opt = r_env_tc_opt(), use_tc = r_env_use_tc();
  PolyCompiledPlan *plan =
      r_plan_get(ctx, tensor_sink, hash, POLY_MODE_CALL, device, opt, devec, tc_opt, use_tc);
  if (!plan) {
    plan = poly_compile_schedule(ctx, sched, device);
    if (!plan) return -1;
    r_plan_put(ctx, tensor_sink, hash, POLY_MODE_CALL, device, opt, devec, tc_opt, use_tc, plan);
  }

  /* Build slot_data from bindings and execute */
  void **slot_data = build_slot_data_from_bindings(ctx, sched, bindings, n_bindings);
  if (!slot_data) return -1;

  int ret = poly_compiled_plan_run(plan, slot_data, sched->n_buf_slots, NULL, 0);
  free(slot_data);
  return ret;
}

int poly_realize_ex(
    PolyCtx *ctx,
    PolyUOp *tensor_sink,
    PolyBufferBinding *bindings,
    int n_bindings,
    PolyVarBinding *var_bindings,
    int n_var_bindings
) {
  if (!tensor_sink || tensor_sink->op != POLY_OP_SINK) {
    fprintf(stderr, "polygrad: realize_ex: expected SINK\n");
    return -1;
  }

  PolyDeviceId device = infer_device(bindings, n_bindings);

  /* Transparent device migration (shared with poly_realize) */
  if (needs_device_migration(device, bindings, n_bindings)) {
    PolyBufferBinding *dev = NULL;
    int total = 0;
    if (migrate_to_device(ctx, tensor_sink, bindings, n_bindings, device, &dev, &total) != 0)
      return -1;
    int ret = poly_realize_ex(ctx, tensor_sink, dev, total, var_bindings, n_var_bindings);
    unmigrate_from_device(bindings, n_bindings, dev, total, device, ret == 0);
    return ret;
  }

  uint32_t hash = poly_structural_hash(tensor_sink) ^ (POLY_SCHED_CACHE_VERSION * 2654435761u);

  PolySchedule *sched = r_sched_get(ctx, tensor_sink, hash, POLY_MODE_CALL);
  if (!sched) {
    sched = poly_schedule_for(ctx, tensor_sink, POLY_MODE_CALL);
    if (!sched) return -1;
    r_sched_put(ctx, tensor_sink, hash, POLY_MODE_CALL, sched);
  }

  int8_t opt = r_env_optimize(), devec = r_env_devectorize();
  int8_t tc_opt = r_env_tc_opt(), use_tc = r_env_use_tc();
  PolyCompiledPlan *plan =
      r_plan_get(ctx, tensor_sink, hash, POLY_MODE_CALL, device, opt, devec, tc_opt, use_tc);
  if (!plan) {
    plan = poly_compile_schedule(ctx, sched, device);
    if (!plan) return -1;
    r_plan_put(ctx, tensor_sink, hash, POLY_MODE_CALL, device, opt, devec, tc_opt, use_tc, plan);
  }

  void **slot_data = build_slot_data_from_bindings(ctx, sched, bindings, n_bindings);
  if (!slot_data) return -1;

  int ret =
      poly_compiled_plan_run(plan, slot_data, sched->n_buf_slots, var_bindings, n_var_bindings);
  free(slot_data);
  return ret;
}

int poly_realize_flat(PolyCtx *ctx, PolyUOp *tensor_sink, PolyUOp **buffers, void **datas, int n) {
  return poly_realize_flat_device(ctx, tensor_sink, buffers, datas, n, POLY_DEVICE_AUTO);
}

int poly_realize_flat_device(
    PolyCtx *ctx,
    PolyUOp *tensor_sink,
    PolyUOp **buffers,
    void **datas,
    int n,
    PolyDeviceId device
) {
  /* datas[] are host pointers -- only host-addressable devices are valid.
   * For device-memory backends (CUDA/HIP), use poly_realize with proper
   * PolyBufferHandle bindings that carry device pointers. */
  PolyDeviceId dom = (device == POLY_DEVICE_AUTO) ? POLY_DEVICE_CPU : device;
  if (dom != POLY_DEVICE_AUTO && !poly_device_is_host_addressable(dom)) {
    fprintf(
        stderr,
        "poly_realize_flat_device: device %d is not host-addressable, "
        "use poly_realize with device-memory bindings\n",
        dom
    );
    return -1;
  }
  PolyBufferBinding *bindings = calloc((size_t)(n > 0 ? n : 1), sizeof(PolyBufferBinding));
  if (!bindings) return -1;
  for (int i = 0; i < n; i++) {
    bindings[i].buffer = buffers[i];
    bindings[i].handle = (PolyBufferHandle){datas[i], 0, dom, false};
  }
  int ret = poly_realize(ctx, tensor_sink, bindings, n);
  free(bindings);
  return ret;
}

#ifndef __EMSCRIPTEN__

/* Stateful realize builder */

static PolyBufferBinding g_realize_bindings[POLY_MAX_REALIZE_BUFS];
static int g_realize_n = 0;

void poly_realize_begin(PolyCtx *ctx) {
  (void)ctx;
  g_realize_n = 0;
}

void poly_realize_bind(PolyCtx *ctx, PolyUOp *buffer, void *data) {
  (void)ctx;
  if (g_realize_n >= POLY_MAX_REALIZE_BUFS) {
    fprintf(stderr, "polygrad: realize_bind: too many bindings (max %d)\n", POLY_MAX_REALIZE_BUFS);
    return;
  }
  g_realize_bindings[g_realize_n].buffer = buffer;
  g_realize_bindings[g_realize_n].handle = (PolyBufferHandle){data, 0, POLY_DEVICE_CPU, false};
  g_realize_n++;
}

int poly_realize_exec(PolyCtx *ctx, PolyUOp *tensor_sink) {
  int ret = poly_realize(ctx, tensor_sink, g_realize_bindings, g_realize_n);
  g_realize_n = 0;
  return ret;
}

/* CUDA realize (DEPRECATED stubs) * These are kept for Python/JS frontend backward compatibility.
 * New code should use poly_realize() with CUDA-domain PolyBufferHandle bindings.
 */

#ifdef POLY_HAS_CUDA

static int cuda_realize_counter = 0;

/* CUDA compiled program cache */
typedef struct {
  uint32_t hash;
  PolyCudaProgram *prog;
  int grid_x; /* grid blocks in x */
  int block_size;
} CudaCacheEntry;

static CudaCacheEntry cuda_prog_cache[PROG_CACHE_CAP];
static int cuda_prog_cache_n = 0;

/* GPU buffer cache: keeps GPU allocations alive between realize() calls.
 * Key: (host_ptr, bytes) → CUdeviceptr.
 * Eliminates repeated alloc + H2D overhead for unchanged inputs. */
#define GPU_BUF_CACHE_CAP 2048
typedef struct {
  void *host_ptr;
  size_t bytes;
  unsigned long long gpu_ptr;
} GpuBufCacheEntry;

static GpuBufCacheEntry gpu_buf_cache[GPU_BUF_CACHE_CAP];
static int gpu_buf_cache_n = 0;

static GpuBufCacheEntry *gpu_buf_lookup(void *host, size_t bytes) {
  for (int i = 0; i < gpu_buf_cache_n; i++)
    if (gpu_buf_cache[i].host_ptr == host && gpu_buf_cache[i].bytes == bytes)
      return &gpu_buf_cache[i];
  return NULL;
}

static void gpu_buf_insert(void *host, size_t bytes, unsigned long long gpu) {
  if (gpu_buf_cache_n < GPU_BUF_CACHE_CAP) {
    gpu_buf_cache[gpu_buf_cache_n].host_ptr = host;
    gpu_buf_cache[gpu_buf_cache_n].bytes = bytes;
    gpu_buf_cache[gpu_buf_cache_n].gpu_ptr = gpu;
    gpu_buf_cache_n++;
  }
}

static CudaCacheEntry *cuda_cache_get(uint32_t h) {
  for (int i = 0; i < cuda_prog_cache_n; i++)
    if (cuda_prog_cache[i].hash == h) return &cuda_prog_cache[i];
  return NULL;
}

static void cuda_cache_put(uint32_t h, PolyCudaProgram *prog, int gx, int bs) {
  if (cuda_prog_cache_n < PROG_CACHE_CAP) {
    cuda_prog_cache[cuda_prog_cache_n].hash = h;
    cuda_prog_cache[cuda_prog_cache_n].prog = prog;
    cuda_prog_cache[cuda_prog_cache_n].grid_x = gx;
    cuda_prog_cache[cuda_prog_cache_n].block_size = bs;
    cuda_prog_cache_n++;
  }
}

void poly_cuda_prog_cache_flush(void) {
  for (int i = 0; i < cuda_prog_cache_n; i++)
    poly_cuda_program_destroy(cuda_prog_cache[i].prog);
  cuda_prog_cache_n = 0;
}

int poly_realize_cuda(
    PolyCtx *ctx,
    PolyUOp *tensor_sink,
    PolyBufferBinding *bindings,
    int n_bindings
) {
  if (!tensor_sink || tensor_sink->op != POLY_OP_SINK) {
    fprintf(stderr, "polygrad: realize_cuda: expected SINK\n");
    return -1;
  }

  if (!poly_cuda_available()) {
    fprintf(stderr, "polygrad: realize_cuda: CUDA not available\n");
    return -1;
  }

  /* Structural hash for CUDA program cache */
  uint32_t cache_hash =
      poly_structural_hash(tensor_sink) ^ (POLY_SCHED_CACHE_VERSION * 2654435761u);

  /* 1. Schedule */
  PolyScheduleResult sr = poly_schedule_v2(ctx, tensor_sink);
  if (sr.n_kernels < 1) {
    poly_schedule_result_free(&sr);
    return -1;
  }

  int n_int = sr.n_intermediates;

  /* GPU memory for intermediate buffers */
  unsigned long long *d_intermediates = NULL;
  if (n_int > 0) {
    d_intermediates = calloc(n_int, sizeof(unsigned long long));
    for (int b = 0; b < n_int; b++) {
      int itemsize = (sr.intermediate_itemsizes && sr.intermediate_itemsizes[b] > 0)
                         ? sr.intermediate_itemsizes[b]
                         : (int)sizeof(float);
      d_intermediates[b] = poly_cuda_alloc(sr.intermediate_sizes[b] * itemsize);
    }
  }

  /* Track GPU allocations for user-bound buffers. Map BUFFER UOp → device ptr.
   * We use a flat array: up to POLY_MAX_REALIZE_BUFS entries. */
  PolyUOp *bound_bufs[POLY_MAX_REALIZE_BUFS];
  unsigned long long bound_dptrs[POLY_MAX_REALIZE_BUFS];
  int n_bound = 0;

  for (int i = 0; i < n_bindings && n_bound < POLY_MAX_REALIZE_BUFS; i++) {
    PolyUOp *buf = bindings[i].buffer;

    /* Determine buffer size from the shape (arg is int = number of elements) */
    int64_t n_elems = buf->arg.i;
    size_t bytes = (size_t)n_elems * poly_dtype_itemsize(poly_dtype_scalar(buf->dtype));

    /* Check GPU buffer cache — reuse allocation, always refresh data.
     * Host pointers can be reused by malloc after free, so cached GPU data
     * may be stale even when the host pointer matches. */
    unsigned long long dptr = 0;
    GpuBufCacheEntry *cached_buf = gpu_buf_lookup(bindings[i].handle.ptr, bytes);
    if (cached_buf) {
      dptr = cached_buf->gpu_ptr;
    } else {
      dptr = poly_cuda_alloc(bytes);
      if (!dptr) {
        fprintf(stderr, "polygrad: realize_cuda: GPU alloc failed for buffer %d\n", i);
        if (d_intermediates) {
          for (int b = 0; b < n_int; b++)
            poly_cuda_free(d_intermediates[b]);
          free(d_intermediates);
        }
        poly_schedule_result_free(&sr);
        return -1;
      }
      /* Cache the allocation */
      gpu_buf_insert(bindings[i].handle.ptr, bytes, dptr);
    }
    /* Always copy host → device (host data may have changed since last cache) */
    poly_cuda_copy_htod(dptr, bindings[i].handle.ptr, bytes);

    bound_bufs[n_bound] = buf;
    bound_dptrs[n_bound] = dptr;
    n_bound++;
  }

  /* Build intermediate buffer lookup */
  PolyMap *inter_set = NULL;
  if (n_int > 0 && sr.intermediate_buf_uops) {
    inter_set = poly_map_new(n_int < 4 ? 4 : (size_t)n_int);
    for (int b = 0; b < n_int; b++) {
      PolyUOp *ib = sr.intermediate_buf_uops[b];
      poly_map_set(inter_set, poly_ptr_hash(ib), ib, (PolyUOp *)(intptr_t)(b + 1), poly_ptr_eq);
    }
  }

  int ret = 0;
  for (int step = 0; step < sr.n_kernels && ret == 0; step++) {
    int k = sr.exec_order ? sr.exec_order[step] : step;
    if (!sr.kernel_n_params || !sr.param_to_buf || !sr.param_to_buf[k]) {
      fprintf(stderr, "polygrad: realize_cuda: missing PARAM mapping for kernel %d\n", k);
      ret = -1;
      break;
    }
    int n_params = sr.kernel_n_params[k];

    /* Build CUDA args: array of pointers to CUdeviceptr values */
    unsigned long long *dptrs = calloc(n_params, sizeof(unsigned long long));
    void **args = calloc(n_params, sizeof(void *));

    for (int i = 0; i < n_params; i++) {
      PolyUOp *buf = sr.param_to_buf[k][i];
      bool found = false;

      if (buf->op == POLY_OP_BUFFER) {
        for (int j = 0; j < n_bound; j++) {
          if (bound_bufs[j] == buf) {
            dptrs[i] = bound_dptrs[j];
            found = true;
            break;
          }
        }
        if (!found && inter_set) {
          PolyUOp *v = poly_map_get(inter_set, poly_ptr_hash(buf), buf, poly_ptr_eq);
          if (v) {
            dptrs[i] = d_intermediates[(int)((intptr_t)v - 1)];
            found = true;
          }
        }
      } else {
        assert(
            buf->op != POLY_OP_BUFFERIZE &&
            "unexpected BUFFERIZE in param_to_buf (old split path removed)"
        );
      }

      if (!found) {
        fprintf(stderr, "polygrad: realize_cuda: no GPU buffer for param %d in kernel %d\n", i, k);
        ret = -1;
        break;
      }
      args[i] = &dptrs[i];
    }

    if (ret == 0) {
      uint32_t kern_hash = cache_hash + (uint32_t)k;
      CudaCacheEntry *cached = cuda_cache_get(kern_hash);

      if (cached) {
        /* Cache hit — just launch with stored grid/block dims */
        ret = poly_cuda_launch(
            cached->prog, args, n_params, cached->grid_x, 1, 1, cached->block_size, 1, 1
        );
        if (ret == 0) ret = poly_cuda_sync();
      } else {
        /* Cache miss — full pipeline */
        int n_lin;
        PolyUOp **lin = poly_linearize_cuda(ctx, sr.kernels[k], &n_lin);
        if (!lin) {
          ret = -1;
          free(dptrs);
          free(args);
          break;
        }

        /* Extract grid/block size from SPECIAL ops.
         * gidx* → global parallelism (grid dimension)
         * lidx* → local parallelism (block dimension) */
        int grid_size = 0;
        int local_size = 0;
        for (int j = 0; j < n_lin; j++) {
          if (lin[j]->op == POLY_OP_SPECIAL && lin[j]->n_src > 0 &&
              lin[j]->src[0]->op == POLY_OP_CONST) {
            const char *sname = lin[j]->arg.str;
            if (sname && sname[0] == 'l') {
              local_size = (int)lin[j]->src[0]->arg.i;
            } else {
              grid_size = (int)lin[j]->src[0]->arg.i;
            }
          }
        }

        char fn_name[32];
        snprintf(fn_name, sizeof(fn_name), "k%d", cuda_realize_counter++);

        int block_size = local_size > 0 ? local_size : 256;
        char *src = poly_render_cuda(lin, n_lin, fn_name, block_size);
        free(lin);

        if (!src) {
          ret = -1;
          free(dptrs);
          free(args);
          break;
        }

        if (getenv("POLY_DUMP_CUDA"))
          fprintf(stderr, "=== CUDA SOURCE (%s) ===\n%s\n=== END ===\n", fn_name, src);

        PolyCudaProgram *prog = poly_compile_cuda(src, fn_name);
        if (!prog) {
          fprintf(stderr, "=== FAILED CUDA SOURCE (%s) ===\n%s\n=== END ===\n", fn_name, src);
          free(src);
          ret = -1;
          free(dptrs);
          free(args);
          break;
        }
        free(src);

        int gx;
        if (grid_size > 0 && local_size > 0) {
          /* Both gidx and lidx: grid = gidx_dim, block = lidx_dim */
          gx = grid_size;
        } else if (local_size > 0) {
          /* Only lidx (pure reduce): single block */
          gx = 1;
        } else if (grid_size > 0) {
          /* Only gidx (elementwise): current behavior */
          gx = (grid_size + block_size - 1) / block_size;
        } else {
          gx = 1;
        }
        ret = poly_cuda_launch(prog, args, n_params, gx, 1, 1, block_size, 1, 1);
        if (ret == 0) ret = poly_cuda_sync();

        /* Cache the compiled program (don't destroy it) */
        cuda_cache_put(kern_hash, prog, gx, block_size);
      }
    }

    free(dptrs);
    free(args);
  }

  /* GPU buffers stay resident — no automatic D2H copy.
   * Use poly_cuda_copyback() to explicitly read results.
   *
   * Don't free user-bound GPU memory — stays in cache for reuse.
   * Only free intermediates (not cached). */
  if (inter_set) poly_map_destroy(inter_set);
  if (d_intermediates) {
    for (int b = 0; b < n_int; b++)
      poly_cuda_free(d_intermediates[b]);
    free(d_intermediates);
  }
  poly_schedule_result_free(&sr);
  return ret;
}

void poly_cuda_flush_buffers(void) {
  for (int i = 0; i < gpu_buf_cache_n; i++)
    poly_cuda_free(gpu_buf_cache[i].gpu_ptr);
  gpu_buf_cache_n = 0;
}

int poly_cuda_copyback(PolyBufferBinding *bindings, int n_bindings) {
  for (int i = 0; i < n_bindings; i++) {
    int64_t n_elems = bindings[i].buffer->arg.i;
    size_t bytes =
        (size_t)n_elems * poly_dtype_itemsize(poly_dtype_scalar(bindings[i].buffer->dtype));
    GpuBufCacheEntry *cached = gpu_buf_lookup(bindings[i].handle.ptr, bytes);
    if (cached) {
      poly_cuda_copy_dtoh(bindings[i].handle.ptr, cached->gpu_ptr, bytes);
    }
  }
  return 0;
}

#endif /* POLY_HAS_CUDA */

#endif /* !__EMSCRIPTEN__ */

/* WASM kernel rendering */

static PolyUOp *g_kernel_bufs[POLY_MAX_REALIZE_BUFS];
static int g_kernel_n_bufs = 0;

struct PolyWasmStepPlan {
  int n_kernels;
  uint8_t **kernel_bytes;
  int *kernel_lens;
  int *kernel_n_params;
  int **kernel_param_buf_idxs;
  int n_total_buffers;
  int n_bindable_buffers;
  int *exec_order;
  int64_t *buffer_sizes; /* element count per buffer (bindable + intermediate) */
  int *buffer_itemsizes; /* bytes per element per buffer (e.g. 4 for f32, 8 for f64) */
};

void poly_wasm_stepplan_destroy(PolyWasmStepPlan *p);

uint8_t *poly_render_kernel_wasm(
    PolyCtx *ctx,
    PolyUOp *tensor_sink,
    int *wasm_len,
    int *n_bufs_out
) {
  if (!tensor_sink || tensor_sink->op != POLY_OP_SINK) {
    fprintf(stderr, "polygrad: render_kernel_wasm: expected SINK\n");
    *wasm_len = 0;
    *n_bufs_out = 0;
    return NULL;
  }

  /* Extract computation UOp (cache key): SINK → STORE → value */
  PolyUOp *comp = NULL;
  if (tensor_sink->n_src > 0 && tensor_sink->src[0]->op == POLY_OP_STORE &&
      tensor_sink->src[0]->n_src >= 2) {
    comp = tensor_sink->src[0]->src[1];
  }

  /* Check kernel cache (structural: matches even with different BUFFER UOps) */
  if (comp) {
    uint32_t h = poly_structural_hash(comp);
    PolyCachedKernel *cached =
        poly_map_get(poly_ctx_kernel_cache(ctx), h, comp, poly_structural_eq);
    if (cached) {
      /* Cache hit — return copy of cached bytes, collect NEW buffer ordering */
      uint8_t *copy = malloc(cached->len);
      if (copy) memcpy(copy, cached->bytes, cached->len);
      g_kernel_n_bufs =
          poly_collect_ordered_buffers(ctx, tensor_sink, g_kernel_bufs, POLY_MAX_REALIZE_BUFS);
      *wasm_len = cached->len;
      *n_bufs_out = g_kernel_n_bufs;
      return copy;
    }
  }

  /* Cache miss — full pipeline */

  /* 1. Reconstruct buffer ordering */
  g_kernel_n_bufs =
      poly_collect_ordered_buffers(ctx, tensor_sink, g_kernel_bufs, POLY_MAX_REALIZE_BUFS);

  /* 2. Schedule */
  PolyUOp *kernel = poly_schedule(ctx, tensor_sink);
  if (!kernel) {
    *wasm_len = 0;
    *n_bufs_out = 0;
    return NULL;
  }

  /* 3. Linearize */
  int n_lin;
  PolyUOp **lin = poly_linearize_env(ctx, kernel, &n_lin);
  if (!lin) {
    *wasm_len = 0;
    *n_bufs_out = 0;
    return NULL;
  }

  /* 4. Render to WASM binary */
  int size;
  uint8_t *wasm = poly_render_wasm(lin, n_lin, &size, false /* scalar */);
  free(lin);

  /* Store in kernel cache */
  if (comp && wasm && size > 0) {
    PolyCachedKernel *ck = malloc(sizeof(PolyCachedKernel));
    if (ck) {
      ck->bytes = malloc(size);
      if (ck->bytes) {
        memcpy(ck->bytes, wasm, size);
        ck->len = size;
        ck->n_bufs = g_kernel_n_bufs;
        memcpy(ck->bufs, g_kernel_bufs, g_kernel_n_bufs * sizeof(PolyUOp *));
        poly_map_set(
            poly_ctx_kernel_cache(ctx), poly_structural_hash(comp), comp, ck, poly_structural_eq
        );
      } else {
        free(ck);
      }
    }
  }

  *wasm_len = size;
  *n_bufs_out = g_kernel_n_bufs;
  return wasm;
}

PolyUOp *poly_kernel_buf(PolyCtx *ctx, int index) {
  (void)ctx;
  if (index < 0 || index >= g_kernel_n_bufs) return NULL;
  return g_kernel_bufs[index];
}

PolyWasmStepPlan *poly_render_step_wasm_plan(PolyCtx *ctx, PolyUOp *tensor_sink) {
  if (!ctx || !tensor_sink || tensor_sink->op != POLY_OP_SINK) {
    fprintf(stderr, "polygrad: wasm_stepplan: expected SINK\n");
    return NULL;
  }

  PolyScheduleResult sr = poly_schedule_v2(ctx, tensor_sink);
  if (sr.n_kernels <= 0 || !sr.kernels) {
    poly_schedule_result_free(&sr);
    return NULL;
  }

  PolyWasmStepPlan *p = calloc(1, sizeof(*p));
  if (!p) {
    poly_schedule_result_free(&sr);
    return NULL;
  }
  p->n_kernels = sr.n_kernels;
  p->kernel_bytes = calloc((size_t)p->n_kernels, sizeof(uint8_t *));
  p->kernel_lens = calloc((size_t)p->n_kernels, sizeof(int));
  p->kernel_n_params = calloc((size_t)p->n_kernels, sizeof(int));
  p->kernel_param_buf_idxs = calloc((size_t)p->n_kernels, sizeof(int *));
  p->exec_order = calloc((size_t)p->n_kernels, sizeof(int));
  if (!p->kernel_bytes || !p->kernel_lens || !p->kernel_n_params || !p->kernel_param_buf_idxs ||
      !p->exec_order) {
    poly_wasm_stepplan_destroy(p);
    poly_schedule_result_free(&sr);
    return NULL;
  }

  PolyUOp *ext_bufs[POLY_MAX_REALIZE_BUFS];
  int n_ext = poly_collect_ordered_buffers(ctx, tensor_sink, ext_bufs, POLY_MAX_REALIZE_BUFS);
  p->n_bindable_buffers = n_ext;
  p->n_total_buffers = n_ext + sr.n_intermediates;

  /* Populate global buffer list so poly_kernel_buf() works with step plan */
  g_kernel_n_bufs = n_ext;
  if (n_ext > 0) memcpy(g_kernel_bufs, ext_bufs, (size_t)n_ext * sizeof(PolyUOp *));

  /* Store per-buffer element counts (bindable from UOp arg, intermediate from schedule) */
  p->buffer_sizes = calloc((size_t)p->n_total_buffers, sizeof(int64_t));
  if (p->buffer_sizes) {
    for (int i = 0; i < n_ext; i++)
      p->buffer_sizes[i] = ext_bufs[i]->arg.i;
    for (int i = 0; i < sr.n_intermediates; i++)
      p->buffer_sizes[n_ext + i] = sr.intermediate_sizes ? sr.intermediate_sizes[i] : 0;
  }

  /* Store per-buffer itemsizes (bytes per element, dtype-aware) */
  p->buffer_itemsizes = calloc((size_t)p->n_total_buffers, sizeof(int));
  if (p->buffer_itemsizes) {
    for (int i = 0; i < n_ext; i++)
      p->buffer_itemsizes[i] = poly_dtype_itemsize(poly_dtype_scalar(ext_bufs[i]->dtype));
    for (int i = 0; i < sr.n_intermediates; i++)
      p->buffer_itemsizes[n_ext + i] = sr.intermediate_itemsizes ? sr.intermediate_itemsizes[i] : 4;
  }

  for (int k = 0; k < p->n_kernels; k++) {
    int n_lin = 0;
    PolyUOp **lin = poly_linearize_env(ctx, sr.kernels[k], &n_lin);
    if (!lin) {
      poly_wasm_stepplan_destroy(p);
      poly_schedule_result_free(&sr);
      return NULL;
    }
    int len = 0;
    uint8_t *bytes = poly_render_wasm(lin, n_lin, &len, false);
    free(lin);
    if (!bytes || len <= 0) {
      free(bytes);
      poly_wasm_stepplan_destroy(p);
      poly_schedule_result_free(&sr);
      return NULL;
    }
    p->kernel_bytes[k] = bytes;
    p->kernel_lens[k] = len;
    int n_params = (sr.kernel_n_params ? sr.kernel_n_params[k] : 0);
    p->kernel_n_params[k] = n_params;
    if (n_params > 0) {
      p->kernel_param_buf_idxs[k] = malloc((size_t)n_params * sizeof(int));
      if (!p->kernel_param_buf_idxs[k]) {
        poly_wasm_stepplan_destroy(p);
        poly_schedule_result_free(&sr);
        return NULL;
      }
      for (int i = 0; i < n_params; i++)
        p->kernel_param_buf_idxs[k][i] = -1;
      for (int i = 0; i < n_params; i++) {
        PolyUOp *pb = (sr.param_to_buf && sr.param_to_buf[k]) ? sr.param_to_buf[k][i] : NULL;
        int idx = -1;
        if (pb) idx = poly_find_buf_position(pb, ext_bufs, n_ext);
        if (idx < 0 && pb && sr.intermediate_buf_uops && sr.n_intermediates > 0) {
          int ib = poly_find_buf_position(pb, sr.intermediate_buf_uops, sr.n_intermediates);
          if (ib >= 0) idx = n_ext + ib;
        }
        if (idx < 0 && (!sr.param_to_buf || !sr.param_to_buf[k]) && i < n_ext) idx = i;
        p->kernel_param_buf_idxs[k][i] = idx;
      }
    }
  }

  if (sr.exec_order)
    memcpy(p->exec_order, sr.exec_order, (size_t)p->n_kernels * sizeof(int));
  else
    for (int i = 0; i < p->n_kernels; i++)
      p->exec_order[i] = i;

  poly_schedule_result_free(&sr);
  return p;
}

int poly_wasm_stepplan_n_kernels(const PolyWasmStepPlan *p) {
  return p ? p->n_kernels : 0;
}

const uint8_t *poly_wasm_stepplan_kernel_bytes(const PolyWasmStepPlan *p, int k, int *len) {
  if (!p || k < 0 || k >= p->n_kernels) return NULL;
  if (len) *len = p->kernel_lens[k];
  return p->kernel_bytes[k];
}

int poly_wasm_stepplan_kernel_n_params(const PolyWasmStepPlan *p, int k) {
  if (!p || k < 0 || k >= p->n_kernels) return 0;
  return p->kernel_n_params[k];
}

int poly_wasm_stepplan_n_buffers(const PolyWasmStepPlan *p) {
  return p ? p->n_total_buffers : 0;
}

int poly_wasm_stepplan_n_bindable_buffers(const PolyWasmStepPlan *p) {
  return p ? p->n_bindable_buffers : 0;
}

int poly_wasm_stepplan_kernel_param_buf_index(const PolyWasmStepPlan *p, int k, int param_idx) {
  if (!p || k < 0 || k >= p->n_kernels) return -1;
  if (param_idx < 0 || param_idx >= p->kernel_n_params[k]) return -1;
  return p->kernel_param_buf_idxs && p->kernel_param_buf_idxs[k]
             ? p->kernel_param_buf_idxs[k][param_idx]
             : -1;
}

const int *poly_wasm_stepplan_exec_order(const PolyWasmStepPlan *p, int *n) {
  if (!p) return NULL;
  if (n) *n = p->n_kernels;
  return p->exec_order;
}

int64_t poly_wasm_stepplan_buf_size(const PolyWasmStepPlan *p, int buf_idx) {
  if (!p || buf_idx < 0 || buf_idx >= p->n_total_buffers) return 0;
  return p->buffer_sizes ? p->buffer_sizes[buf_idx] : 0;
}

int64_t poly_wasm_stepplan_buf_nbytes(const PolyWasmStepPlan *p, int buf_idx) {
  if (!p || buf_idx < 0 || buf_idx >= p->n_total_buffers) return 0;
  int64_t elems = p->buffer_sizes ? p->buffer_sizes[buf_idx] : 0;
  int itemsize =
      (p->buffer_itemsizes && buf_idx < p->n_total_buffers) ? p->buffer_itemsizes[buf_idx] : 4;
  return elems * itemsize;
}

int poly_wasm_stepplan_bindable_buf_index(const PolyWasmStepPlan *p, int bi) {
  if (!p || bi < 0 || bi >= p->n_bindable_buffers) return -1;
  return bi; /* identity today; explicit contract for future indirection */
}

int poly_abi_version(void) {
  return POLYGRAD_ABI_VERSION;
}

/* WebGPU step plan */

struct PolyWebGpuStepPlan {
  int n_kernels;
  char **kernel_wgsl; /* WGSL source per kernel (malloc'd strings) */
  int *kernel_wgsl_lens;
  int *kernel_n_params;
  int **kernel_param_buf_idxs;
  int *kernel_grid; /* [k*3+dim]: dispatch workgroup counts */
  int *kernel_local; /* [k*3+dim]: workgroup sizes */
  int n_total_buffers;
  int n_bindable_buffers;
  int *exec_order;
  int64_t *buffer_sizes; /* element count per buffer */
  int *buffer_itemsizes; /* bytes per element per buffer */
};

/* Extract grid/local dimensions from linearized UOps.
 * SPECIAL("gidxN") → grid[N] = bound, SPECIAL("lidxN") → local[N] = bound.
 * Grid dispatch = global_bound for gidx (it's workgroup count, not thread count). */
static void webgpu_extract_dims(PolyUOp **lin, int n_lin, int grid[3], int local[3]) {
  grid[0] = 1;
  grid[1] = 1;
  grid[2] = 1;
  local[0] = 1;
  local[1] = 1;
  local[2] = 1;

  for (int j = 0; j < n_lin; j++) {
    if (lin[j]->op != POLY_OP_SPECIAL) continue;
    if (!lin[j]->arg.str || lin[j]->n_src < 1) continue;
    if (lin[j]->src[0]->op != POLY_OP_CONST) continue;

    const char *sn = lin[j]->arg.str;
    int slen = (int)strlen(sn);
    int dim_idx = (slen > 0) ? sn[slen - 1] - '0' : 0;
    if (dim_idx < 0 || dim_idx > 2) dim_idx = 0;
    int bound = (int)lin[j]->src[0]->arg.i;

    if (sn[0] == 'l')
      local[dim_idx] = bound;
    else
      grid[dim_idx] = bound;
  }
}

PolyWebGpuStepPlan *poly_render_step_webgpu_plan(PolyCtx *ctx, PolyUOp *tensor_sink) {
  if (!ctx || !tensor_sink || tensor_sink->op != POLY_OP_SINK) {
    fprintf(stderr, "polygrad: webgpu_stepplan: expected SINK\n");
    return NULL;
  }

  PolyScheduleResult sr = poly_schedule_v2(ctx, tensor_sink);
  if (sr.n_kernels <= 0 || !sr.kernels) {
    poly_schedule_result_free(&sr);
    return NULL;
  }

  PolyWebGpuStepPlan *p = calloc(1, sizeof(*p));
  if (!p) {
    poly_schedule_result_free(&sr);
    return NULL;
  }

  p->n_kernels = sr.n_kernels;
  p->kernel_wgsl = calloc((size_t)p->n_kernels, sizeof(char *));
  p->kernel_wgsl_lens = calloc((size_t)p->n_kernels, sizeof(int));
  p->kernel_n_params = calloc((size_t)p->n_kernels, sizeof(int));
  p->kernel_param_buf_idxs = calloc((size_t)p->n_kernels, sizeof(int *));
  p->kernel_grid = calloc((size_t)p->n_kernels * 3, sizeof(int));
  p->kernel_local = calloc((size_t)p->n_kernels * 3, sizeof(int));
  p->exec_order = calloc((size_t)p->n_kernels, sizeof(int));
  if (!p->kernel_wgsl || !p->kernel_wgsl_lens || !p->kernel_n_params || !p->kernel_param_buf_idxs ||
      !p->kernel_grid || !p->kernel_local || !p->exec_order) {
    poly_webgpu_stepplan_destroy(p);
    poly_schedule_result_free(&sr);
    return NULL;
  }

  /* Collect external (user-visible) buffers */
  PolyUOp *ext_bufs[POLY_MAX_REALIZE_BUFS];
  int n_ext = poly_collect_ordered_buffers(ctx, tensor_sink, ext_bufs, POLY_MAX_REALIZE_BUFS);
  p->n_bindable_buffers = n_ext;
  p->n_total_buffers = n_ext + sr.n_intermediates;

  /* Populate global buffer list so poly_kernel_buf() works */
  g_kernel_n_bufs = n_ext;
  if (n_ext > 0) memcpy(g_kernel_bufs, ext_bufs, (size_t)n_ext * sizeof(PolyUOp *));

  /* Per-buffer element counts and itemsizes */
  p->buffer_sizes = calloc((size_t)p->n_total_buffers, sizeof(int64_t));
  p->buffer_itemsizes = calloc((size_t)p->n_total_buffers, sizeof(int));
  if (p->buffer_sizes) {
    for (int i = 0; i < n_ext; i++)
      p->buffer_sizes[i] = ext_bufs[i]->arg.i;
    for (int i = 0; i < sr.n_intermediates; i++)
      p->buffer_sizes[n_ext + i] = sr.intermediate_sizes ? sr.intermediate_sizes[i] : 0;
  }
  if (p->buffer_itemsizes) {
    for (int i = 0; i < n_ext; i++)
      p->buffer_itemsizes[i] = poly_dtype_itemsize(poly_dtype_scalar(ext_bufs[i]->dtype));
    for (int i = 0; i < sr.n_intermediates; i++)
      p->buffer_itemsizes[n_ext + i] = sr.intermediate_itemsizes ? sr.intermediate_itemsizes[i] : 4;
  }

  /* Per-kernel: linearize → render WGSL → extract grid/local */
  for (int k = 0; k < p->n_kernels; k++) {
    int n_lin = 0;
    PolyUOp **lin = poly_linearize_webgpu(ctx, sr.kernels[k], &n_lin);
    if (!lin) {
      poly_webgpu_stepplan_destroy(p);
      poly_schedule_result_free(&sr);
      return NULL;
    }

    /* Extract grid and local dims from SPECIAL ops */
    webgpu_extract_dims(lin, n_lin, &p->kernel_grid[k * 3], &p->kernel_local[k * 3]);

    /* Render WGSL source */
    char fn_name[64];
    snprintf(fn_name, sizeof(fn_name), "k%d", k);
    char *wgsl = poly_render_wgsl(lin, n_lin, fn_name);
    free(lin);
    if (!wgsl) {
      poly_webgpu_stepplan_destroy(p);
      poly_schedule_result_free(&sr);
      return NULL;
    }

    if (getenv("POLY_DUMP_KERNELS"))
      fprintf(stderr, "=== WGSL KERNEL %s ===\n%s\n=== END ===\n", fn_name, wgsl);

    p->kernel_wgsl[k] = wgsl;
    p->kernel_wgsl_lens[k] = (int)strlen(wgsl);

    /* Param-to-buffer index mapping (same logic as WASM step plan) */
    int n_params = (sr.kernel_n_params ? sr.kernel_n_params[k] : 0);
    p->kernel_n_params[k] = n_params;
    if (n_params > 0) {
      p->kernel_param_buf_idxs[k] = malloc((size_t)n_params * sizeof(int));
      if (!p->kernel_param_buf_idxs[k]) {
        poly_webgpu_stepplan_destroy(p);
        poly_schedule_result_free(&sr);
        return NULL;
      }
      for (int i = 0; i < n_params; i++)
        p->kernel_param_buf_idxs[k][i] = -1;
      for (int i = 0; i < n_params; i++) {
        PolyUOp *pb = (sr.param_to_buf && sr.param_to_buf[k]) ? sr.param_to_buf[k][i] : NULL;
        int idx = -1;
        if (pb) idx = poly_find_buf_position(pb, ext_bufs, n_ext);
        if (idx < 0 && pb && sr.intermediate_buf_uops && sr.n_intermediates > 0) {
          int ib = poly_find_buf_position(pb, sr.intermediate_buf_uops, sr.n_intermediates);
          if (ib >= 0) idx = n_ext + ib;
        }
        if (idx < 0 && (!sr.param_to_buf || !sr.param_to_buf[k]) && i < n_ext) idx = i;
        p->kernel_param_buf_idxs[k][i] = idx;
      }
    }
  }

  if (sr.exec_order)
    memcpy(p->exec_order, sr.exec_order, (size_t)p->n_kernels * sizeof(int));
  else
    for (int i = 0; i < p->n_kernels; i++)
      p->exec_order[i] = i;

  poly_schedule_result_free(&sr);
  return p;
}

/* WebGPU step plan accessors */

int poly_webgpu_stepplan_n_kernels(const PolyWebGpuStepPlan *p) {
  return p ? p->n_kernels : 0;
}

const char *poly_webgpu_stepplan_kernel_wgsl(const PolyWebGpuStepPlan *p, int k, int *len) {
  if (!p || k < 0 || k >= p->n_kernels) return NULL;
  if (len) *len = p->kernel_wgsl_lens[k];
  return p->kernel_wgsl[k];
}

int poly_webgpu_stepplan_kernel_n_params(const PolyWebGpuStepPlan *p, int k) {
  if (!p || k < 0 || k >= p->n_kernels) return 0;
  return p->kernel_n_params[k];
}

int poly_webgpu_stepplan_kernel_grid(const PolyWebGpuStepPlan *p, int k, int dim) {
  if (!p || k < 0 || k >= p->n_kernels || dim < 0 || dim > 2) return 1;
  return p->kernel_grid[k * 3 + dim];
}

int poly_webgpu_stepplan_kernel_local(const PolyWebGpuStepPlan *p, int k, int dim) {
  if (!p || k < 0 || k >= p->n_kernels || dim < 0 || dim > 2) return 1;
  return p->kernel_local[k * 3 + dim];
}

int poly_webgpu_stepplan_n_buffers(const PolyWebGpuStepPlan *p) {
  return p ? p->n_total_buffers : 0;
}

int poly_webgpu_stepplan_n_bindable_buffers(const PolyWebGpuStepPlan *p) {
  return p ? p->n_bindable_buffers : 0;
}

int poly_webgpu_stepplan_bindable_buf_index(const PolyWebGpuStepPlan *p, int bi) {
  if (!p || bi < 0 || bi >= p->n_bindable_buffers) return -1;
  return bi;
}

int poly_webgpu_stepplan_kernel_param_buf_index(const PolyWebGpuStepPlan *p, int k, int param_idx) {
  if (!p || k < 0 || k >= p->n_kernels) return -1;
  if (param_idx < 0 || param_idx >= p->kernel_n_params[k]) return -1;
  return p->kernel_param_buf_idxs && p->kernel_param_buf_idxs[k]
             ? p->kernel_param_buf_idxs[k][param_idx]
             : -1;
}

const int *poly_webgpu_stepplan_exec_order(const PolyWebGpuStepPlan *p, int *n) {
  if (!p) return NULL;
  if (n) *n = p->n_kernels;
  return p->exec_order;
}

int64_t poly_webgpu_stepplan_buf_size(const PolyWebGpuStepPlan *p, int buf_idx) {
  if (!p || buf_idx < 0 || buf_idx >= p->n_total_buffers) return 0;
  return p->buffer_sizes ? p->buffer_sizes[buf_idx] : 0;
}

int64_t poly_webgpu_stepplan_buf_nbytes(const PolyWebGpuStepPlan *p, int buf_idx) {
  if (!p || buf_idx < 0 || buf_idx >= p->n_total_buffers) return 0;
  int64_t elems = p->buffer_sizes ? p->buffer_sizes[buf_idx] : 0;
  int itemsize =
      (p->buffer_itemsizes && buf_idx < p->n_total_buffers) ? p->buffer_itemsizes[buf_idx] : 4;
  return elems * itemsize;
}

void poly_webgpu_stepplan_destroy(PolyWebGpuStepPlan *p) {
  if (!p) return;
  if (p->kernel_wgsl) {
    for (int i = 0; i < p->n_kernels; i++)
      free(p->kernel_wgsl[i]);
    free(p->kernel_wgsl);
  }
  if (p->kernel_param_buf_idxs) {
    for (int i = 0; i < p->n_kernels; i++)
      free(p->kernel_param_buf_idxs[i]);
    free(p->kernel_param_buf_idxs);
  }
  free(p->kernel_wgsl_lens);
  free(p->kernel_n_params);
  free(p->kernel_grid);
  free(p->kernel_local);
  free(p->exec_order);
  free(p->buffer_sizes);
  free(p->buffer_itemsizes);
  free(p);
}

void poly_wasm_stepplan_destroy(PolyWasmStepPlan *p) {
  if (!p) return;
  if (p->kernel_bytes) {
    for (int i = 0; i < p->n_kernels; i++)
      free(p->kernel_bytes[i]);
    free(p->kernel_bytes);
  }
  if (p->kernel_param_buf_idxs) {
    for (int i = 0; i < p->n_kernels; i++)
      free(p->kernel_param_buf_idxs[i]);
    free(p->kernel_param_buf_idxs);
  }
  free(p->kernel_lens);
  free(p->kernel_n_params);
  free(p->exec_order);
  free(p->buffer_sizes);
  free(p->buffer_itemsizes);
  free(p);
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
