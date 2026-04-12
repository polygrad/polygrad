/* realize.c -- tinygrad-parity realize: graph is source of truth.
 *
 * Mirrors tinygrad's Tensor.realize() architecture:
 *   1. Walk SINK to find BUFFER UOps
 *   2. Look up handles from ctx side table (like tinygrad's `buffers` WeakKeyDict)
 *   3. Strip BIND values for var_vals (like complete_create_schedule_with_vars)
 *   4. Hash post-strip sink for cache keying (like transform_to_call normalization)
 *   5. Schedule, compile, execute
 *
 * The old poly_realize(ctx, sink, bindings, n) in frontend.c passes bindings
 * externally. This function reads them from the graph — no external bindings.
 */

#include "realize.h"
#include "frontend.h"
#include "frontend_internal.h"
#include "exec_plan.h"
#include "polygrad.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Buffer handle side table on PolyCtx.
 *
 * Tinygrad equivalent: uop/ops.py:108
 *   buffers: WeakKeyDictionary[UOp, Buffer] = WeakKeyDictionary()
 *
 * Polygrad: per-ctx PolyMap from BUFFER UOp pointer to PolyBuffer.
 * Per-ctx (not global) because arena-allocated UOp pointers are only valid
 * within their owning context. Cleaned up in poly_ctx_destroy.
 *
 * Implementation: a flat array for simplicity (BUFFER count per-ctx is
 * typically <100). Upgrade to PolyMap if this becomes a bottleneck.
 */

#define HANDLE_TABLE_INIT_CAP 32

typedef struct {
  PolyUOp *buf;
  PolyBuffer handle;
} HandleEntry;

typedef struct {
  HandleEntry *entries;
  int n;
  int cap;
} HandleTable;

/* Get or create the handle table for a context.
 * Stored as an opaque pointer on PolyCtx.realize_data. */
static HandleTable *get_handle_table(PolyCtx *ctx) {
  /* realize_data is a void* on PolyCtx — we use it for the handle table.
   * Initialized lazily on first access. */
  if (!ctx->realize_data) {
    HandleTable *ht = calloc(1, sizeof(HandleTable));
    if (!ht) return NULL;
    ht->entries = calloc(HANDLE_TABLE_INIT_CAP, sizeof(HandleEntry));
    if (!ht->entries) {
      free(ht);
      return NULL;
    }
    ht->cap = HANDLE_TABLE_INIT_CAP;
    ctx->realize_data = ht;
  }
  return (HandleTable *)ctx->realize_data;
}

void poly_ctx_set_handle(PolyCtx *ctx, PolyUOp *buf, PolyBuffer handle) {
  if (!ctx || !buf) return;
  HandleTable *ht = get_handle_table(ctx);
  if (!ht) return;

  /* Update existing entry if present */
  for (int i = 0; i < ht->n; i++) {
    if (ht->entries[i].buf == buf) {
      ht->entries[i].handle = handle;
      return;
    }
  }

  /* Grow if needed */
  if (ht->n >= ht->cap) {
    int new_cap = ht->cap * 2;
    HandleEntry *new_entries = realloc(ht->entries, (size_t)new_cap * sizeof(HandleEntry));
    if (!new_entries) return;
    ht->entries = new_entries;
    ht->cap = new_cap;
  }

  ht->entries[ht->n++] = (HandleEntry){.buf = buf, .handle = handle};
}

PolyBuffer *poly_ctx_get_handle(PolyCtx *ctx, PolyUOp *buf) {
  if (!ctx || !buf || !ctx->realize_data) return NULL;
  HandleTable *ht = (HandleTable *)ctx->realize_data;
  for (int i = 0; i < ht->n; i++) {
    if (ht->entries[i].buf == buf) return &ht->entries[i].handle;
  }
  return NULL;
}

bool poly_ctx_remove_handle(PolyCtx *ctx, PolyUOp *buf) {
  if (!ctx || !buf || !ctx->realize_data) return false;
  HandleTable *ht = (HandleTable *)ctx->realize_data;
  for (int i = 0; i < ht->n; i++) {
    if (ht->entries[i].buf == buf) {
      ht->entries[i] = ht->entries[--ht->n]; /* swap-remove */
      return true;
    }
  }
  return false;
}

/* Called from poly_ctx_destroy to free handle table. */
void poly_realize_ctx_cleanup(PolyCtx *ctx) {
  if (!ctx || !ctx->realize_data) return;
  HandleTable *ht = (HandleTable *)ctx->realize_data;
  free(ht->entries);
  free(ht);
  ctx->realize_data = NULL;
}

/* Infer device from handles in the side table for the given buffer UOps. */
static PolyDeviceId infer_device_from_handles(HandleTable *ht) {
  for (int i = 0; i < ht->n; i++) {
    PolyDeviceId d = ht->entries[i].handle.domain;
    if (d != POLY_DEVICE_AUTO && d != POLY_DEVICE_CPU) return d;
  }
  /* Check POLY_DEVICE env var */
  const char *env = getenv("POLY_DEVICE");
  if (env) {
    if (strcmp(env, "interp") == 0) return POLY_DEVICE_INTERP;
    if (strcmp(env, "x64") == 0) return POLY_DEVICE_X64;
    if (strcmp(env, "cuda") == 0) return POLY_DEVICE_CUDA;
    if (strcmp(env, "hip") == 0) return POLY_DEVICE_HIP;
  }
#ifdef __EMSCRIPTEN__
  return POLY_DEVICE_WASM_JIT;
#else
  return POLY_DEVICE_CPU;
#endif
}

/* Schedule + plan cache for poly_realize_graph.
 * Uses post-strip hash, fixing the pre-strip cache keying bug in frontend.c.
 * Separate from frontend.c's caches to avoid interference during migration. */

#define REALIZE_GRAPH_SCHED_CAP 128
#define REALIZE_GRAPH_PLAN_CAP 256

typedef struct {
  PolyCtx *ctx;
  PolyUOp *stripped_sink;
  uint32_t hash;
  PolySchedule *sched;
} GraphSchedCacheEntry;

typedef struct {
  PolyCtx *ctx;
  PolyUOp *stripped_sink;
  uint32_t hash;
  PolyDeviceId device;
  int8_t optimize;
  int8_t devectorize;
  PolyCompiledPlan *plan;
} GraphPlanCacheEntry;

static GraphSchedCacheEntry g_sched_cache[REALIZE_GRAPH_SCHED_CAP];
static int g_sched_n = 0;

static GraphPlanCacheEntry g_plan_cache[REALIZE_GRAPH_PLAN_CAP];
static int g_plan_n = 0;

static PolySchedule *graph_sched_get(PolyCtx *ctx, PolyUOp *stripped, uint32_t h) {
  for (int i = 0; i < g_sched_n; i++)
    if (g_sched_cache[i].ctx == ctx && g_sched_cache[i].stripped_sink == stripped &&
        g_sched_cache[i].hash == h)
      return g_sched_cache[i].sched;
  return NULL;
}

static void graph_sched_put(PolyCtx *ctx, PolyUOp *stripped, uint32_t h, PolySchedule *s) {
  if (g_sched_n < REALIZE_GRAPH_SCHED_CAP)
    g_sched_cache[g_sched_n++] = (GraphSchedCacheEntry){ctx, stripped, h, s};
}

static int8_t env_optimize(void) {
  const char *e = getenv("POLY_OPTIMIZE");
  return e ? (int8_t)atoi(e) : 0;
}

static int8_t env_devectorize(void) {
  const char *e = getenv("POLY_DEVECTORIZE");
  return e ? (int8_t)atoi(e) : -1;
}

static PolyCompiledPlan *graph_plan_get(
    PolyCtx *ctx, PolyUOp *stripped, uint32_t h, PolyDeviceId d, int8_t opt, int8_t devec
) {
  for (int i = 0; i < g_plan_n; i++)
    if (g_plan_cache[i].ctx == ctx && g_plan_cache[i].stripped_sink == stripped &&
        g_plan_cache[i].hash == h && g_plan_cache[i].device == d &&
        g_plan_cache[i].optimize == opt && g_plan_cache[i].devectorize == devec)
      return g_plan_cache[i].plan;
  return NULL;
}

static void graph_plan_put(
    PolyCtx *ctx, PolyUOp *stripped, uint32_t h, PolyDeviceId d, int8_t opt, int8_t devec,
    PolyCompiledPlan *p
) {
  if (g_plan_n < REALIZE_GRAPH_PLAN_CAP)
    g_plan_cache[g_plan_n++] = (GraphPlanCacheEntry){ctx, stripped, h, d, opt, devec, p};
}

/* Purge realize_graph caches for a destroyed context. */
static void graph_cache_purge(PolyCtx *ctx) {
  for (int i = g_sched_n - 1; i >= 0; i--) {
    if (g_sched_cache[i].ctx == ctx) g_sched_cache[i] = g_sched_cache[--g_sched_n];
  }
  for (int i = g_plan_n - 1; i >= 0; i--) {
    if (g_plan_cache[i].ctx == ctx) g_plan_cache[i] = g_plan_cache[--g_plan_n];
  }
}

int poly_realize_graph(PolyCtx *ctx, PolyUOp *tensor_sink) {
  if (!ctx || !tensor_sink || tensor_sink->op != POLY_OP_SINK) {
    fprintf(stderr, "polygrad: realize_graph: expected SINK\n");
    return -1;
  }

  HandleTable *ht = get_handle_table(ctx);
  if (!ht) return -1;

  /* 1. Strip BINDs, extract var_vals (tinygrad: complete_create_schedule_with_vars)
   * Uses post-strip sink for cache keying — different BIND values share
   * the same compiled kernel. Fixes the pre-strip cache bug in frontend.c. */
  PolyVarBinding extracted_vars[16];
  int n_extracted = 0;
  PolyUOp *stripped = poly_strip_bind_values(
      ctx, tensor_sink, extracted_vars, &n_extracted, 16, NULL, 0
  );

  uint32_t hash = poly_structural_hash(stripped) ^ (0x9E3779B9u); /* post-strip hash */

  /* 2. Schedule (cached by post-strip identity) */
  PolySchedule *sched = graph_sched_get(ctx, stripped, hash);
  if (!sched) {
    sched = poly_schedule_for(ctx, tensor_sink, POLY_MODE_CALL);
    if (!sched) return -1;
    graph_sched_put(ctx, stripped, hash, sched);
  }

  /* 3. Compile (cached by device + optimization flags) */
  PolyDeviceId device = infer_device_from_handles(ht);
  int8_t opt = env_optimize(), devec = env_devectorize();
  PolyCompiledPlan *plan = graph_plan_get(ctx, stripped, hash, device, opt, devec);
  if (!plan) {
    plan = poly_compile_schedule(ctx, sched, device);
    if (!plan) return -1;
    graph_plan_put(ctx, stripped, hash, device, opt, devec, plan);
  }

  /* 4. Build slot_data from side table (tinygrad: ExecItem.run ensure_allocated)
   * Walk the schedule's buf_slots. For each non-intermediate slot, look up
   * the BUFFER UOp's handle from the side table. For intermediates, the
   * compiled plan allocates workspace internally. */
  void **slot_data = calloc((size_t)sched->n_buf_slots, sizeof(void *));
  if (!slot_data) return -1;

  for (int s = 0; s < sched->n_buf_slots; s++) {
    if (sched->buf_slots[s].is_intermediate) continue;
    PolyUOp *buf_uop = sched->buf_slots[s].buf_uop;
    PolyBuffer *h = poly_ctx_get_handle(ctx, buf_uop);
    if (!h) {
      fprintf(
          stderr,
          "polygrad: realize_graph: no handle for buffer slot %d (op=%s). "
          "Attach handles via poly_ctx_set_handle before realize.\n",
          s, poly_op_name(buf_uop->op)
      );
      free(slot_data);
      return -1;
    }
    slot_data[s] = h->ptr;
  }

  /* 5. Execute (pass extracted BIND var_vals) */
  int ret = poly_compiled_plan_run(plan, slot_data, sched->n_buf_slots, extracted_vars, n_extracted);
  free(slot_data);
  return ret;
}
