/* ctx.c -- PolyCtx lifecycle: create, destroy, accessors */

#include "ctx.h"
#include "device.h"
#include "engine/schedule.h"
#include "uop/ops.h"
#include "utils.h"
#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include <string.h>

/* Tinygrad UOps disappear through Python refcounts, so stable JIT replay has
 * no global tracing pass. C batches its nonmoving mark/sweep until unreclaimed
 * UOp storage has grown by this bounded budget. */
#include <time.h>

/* Defined in ops.c */
void poly_init_group_ops(void);

/* Optional cleanup hooks (defined in other files, linked weakly). */
void poly_frontend_ctx_cleanup(PolyCtx *ctx) __attribute__((weak));
void poly_tensor_ctx_cleanup(PolyCtx *ctx) __attribute__((weak));
void poly_engine_ctx_cleanup(PolyCtx *ctx) __attribute__((weak));

int poly_shape_cache_evict_unmarked(PolyCtx *ctx, PolyMap *live);
void poly_shape_cache_destroy_all(PolyCtx *ctx);
int poly_shape_cache_live_value_roots(
    PolyCtx *ctx,
    PolyMap *live,
    PolyUOp ***roots_out,
    int *count_out
);

static void free_buffer_entry(const void *key, void *value, void *userdata) {
  (void)key;
  PolyCtx *ctx = (PolyCtx *)userdata;
  /* Free current + src chain and its C handle metadata. */
  poly_buffer_free_chain(ctx, (PolyBuffer *)value);
}

static bool logical_policy_from_env(const char *value, PolyLogicalPolicy *policy) {
  if (!value || !policy) return false;
  if (strcmp(value, "0") == 0)
    *policy = POLY_LOGICAL_NEVER;
  else if (strcmp(value, "1") == 0)
    *policy = POLY_LOGICAL_ALWAYS;
  else if (strcmp(value, "2") == 0)
    *policy = POLY_LOGICAL_UNTIL_REALIZE;
  else
    return false;
  return true;
}

PolyCtx *poly_ctx_new(void) {
  poly_init_group_ops();
  if (poly_get_default_float() < 0 || poly_get_default_int() < 0) return NULL;
  const char *logical_env = getenv("POLY_LOGICAL");
  /* Polygrad logical-lifetime divergence: retain portable producers until
   * successful materialization, then keep only the exact resource spine. */
  PolyLogicalPolicy logical_policy = POLY_LOGICAL_UNTIL_REALIZE;
  if (logical_env && !logical_policy_from_env(logical_env, &logical_policy)) return NULL;
  PolyCtx *ctx = malloc(sizeof(PolyCtx));
  if (!ctx) return NULL;
  ctx->arena = poly_arena_new(0);
  ctx->scratch = poly_arena_new(0);
  ctx->cse = poly_map_new(256);
  ctx->uop_storage = poly_map_new(256);
  ctx->uop_storage_bytes = 0;
  ctx->uop_storage_high_water = 0;
  ctx->schedule_cache = poly_map_new(16);
  ctx->to_program_cache = poly_map_new(16);
  ctx->runtime_cache = poly_map_new(16);
  ctx->graph_cache = poly_map_new(8);
  ctx->runtime_artifact_entries = 0;
  ctx->runtime_artifact_live_bytes = 0;
  ctx->launch_count = 0;
  ctx->runtime_cache_hits = 0;
  ctx->runtime_cache_misses = 0;
  ctx->buffer_read_count = 0;
  ctx->buffer_read_bytes = 0;
  ctx->buffer_write_count = 0;
  ctx->buffer_write_bytes = 0;
  ctx->buffer_copy_count = 0;
  ctx->buffer_copy_bytes = 0;
  ctx->global_ops = 0;
  ctx->global_mem = 0;
  ctx->time_sum_s = 0.0;
  ctx->kernel_count = 0;
  ctx->mem_used = 0;
  memset(ctx->mem_used_per_device, 0, sizeof(ctx->mem_used_per_device));
  ctx->mem_used_by_device = poly_map_new(8);
  ctx->stats_suppression_depth = 0;
  ctx->shape_cache = poly_map_new(64);
  ctx->buffers = poly_map_new(64);
  ctx->retained_uops = poly_map_new(16);
  ctx->collection_dirty = false;
  ctx->ir_collection_dirty = false;
  ctx->ir_collection_baseline_bytes = 0;
  ctx->collecting = false;
  ctx->execution_depth = 0;
  ctx->rng_states = poly_map_new(8);
  ctx->rng_seed = (uint64_t)time(NULL);
  ctx->rng_device_count = 0;
  ctx->tensors = NULL;
  ctx->n_tensors = 0;
  ctx->tensors_cap = 0;
  ctx->next_tensor_order = 1;
  ctx->active_jit_capture = NULL;
  ctx->name_map = poly_map_new(16);
  if (!ctx->arena || !ctx->scratch || !ctx->cse || !ctx->uop_storage || !ctx->schedule_cache ||
      !ctx->to_program_cache || !ctx->runtime_cache || !ctx->graph_cache ||
      !ctx->mem_used_by_device || !ctx->shape_cache || !ctx->buffers || !ctx->retained_uops ||
      !ctx->rng_states || !ctx->name_map) {
    if (ctx->arena) poly_arena_destroy(ctx->arena);
    if (ctx->scratch) poly_arena_destroy(ctx->scratch);
    if (ctx->cse) poly_map_destroy(ctx->cse);
    if (ctx->uop_storage) poly_map_destroy(ctx->uop_storage);
    if (ctx->schedule_cache) poly_map_destroy(ctx->schedule_cache);
    if (ctx->to_program_cache) poly_map_destroy(ctx->to_program_cache);
    if (ctx->runtime_cache) poly_map_destroy(ctx->runtime_cache);
    if (ctx->graph_cache) poly_map_destroy(ctx->graph_cache);
    if (ctx->mem_used_by_device) poly_map_destroy(ctx->mem_used_by_device);
    if (ctx->shape_cache) poly_map_destroy(ctx->shape_cache);
    if (ctx->buffers) poly_map_destroy(ctx->buffers);
    if (ctx->retained_uops) poly_map_destroy(ctx->retained_uops);
    if (ctx->rng_states) poly_map_destroy(ctx->rng_states);
    if (ctx->name_map) poly_map_destroy(ctx->name_map);
    free(ctx);
    return NULL;
  }
  ctx->entries = NULL;
  ctx->n_entries = 0;
  ctx->entries_cap = 0;
  ctx->ep = NULL;
  ctx->n_ep = 0;
  ctx->ep_cap = 0;
  ctx->next_buf_tag = 1;
  ctx->next_unique_id = 0;
  ctx->preferred_device = POLY_DEVICE_AUTO;
  /* Polygrad logical-lifetime divergence: environment initializes one
   * context; later scoped changes are explicit and affect future Tensors. */
  ctx->logical_policy = logical_policy;
  const char *dev_env = getenv("POLY_DEVICE");
  if (dev_env && dev_env[0]) {
    PolyDevice env_device = poly_device_by_name(dev_env);
    if (env_device != POLY_DEVICE_AUTO && env_device != POLY_DEVICE_HOST)
      ctx->preferred_device = env_device;
  }
  ctx->frontend_buffer_release = NULL;
  return ctx;
}

void poly_ctx_destroy(PolyCtx *ctx) {
  if (!ctx) return;
  if (poly_frontend_ctx_cleanup) poly_frontend_ctx_cleanup(ctx);
  if (poly_tensor_ctx_cleanup) poly_tensor_ctx_cleanup(ctx);
  if (poly_engine_ctx_cleanup) poly_engine_ctx_cleanup(ctx);
  poly_map_destroy(ctx->schedule_cache);
  poly_map_destroy(ctx->to_program_cache);
  poly_map_destroy(ctx->runtime_cache);
  poly_map_destroy(ctx->graph_cache);
  poly_shape_cache_destroy_all(ctx);
  poly_map_destroy(ctx->shape_cache);
  /* Free owned buffer ptrs before destroying the map. */
  poly_map_foreach(ctx->buffers, free_buffer_entry, ctx);
  poly_map_destroy(ctx->buffers);
  poly_map_destroy(ctx->retained_uops);
  poly_map_destroy(ctx->rng_states);
  poly_map_destroy(ctx->mem_used_by_device);
  free(ctx->tensors);
  poly_map_destroy(ctx->name_map);
  poly_map_destroy(ctx->cse);
  poly_uop_storage_destroy_all(ctx);
  poly_map_destroy(ctx->uop_storage);
  free(ctx->entries);
  free(ctx->ep);
  poly_arena_destroy(ctx->scratch);
  poly_arena_destroy(ctx->arena);
  free(ctx);
}

int poly_uop_retain(PolyCtx *ctx, PolyUOp *uop) {
  if (!ctx || !uop || !poly_ctx_owns_ptr(ctx, uop)) return -1;
  uintptr_t count =
      (uintptr_t)poly_map_get(ctx->retained_uops, poly_ptr_hash(uop), uop, poly_ptr_eq);
  if (count == UINTPTR_MAX) return -1;
  poly_map_set(ctx->retained_uops, poly_ptr_hash(uop), uop, (void *)(count + 1), poly_ptr_eq);
  return 0;
}

void poly_uop_release(PolyCtx *ctx, PolyUOp *uop) {
  if (!ctx || !uop) return;
  uintptr_t count =
      (uintptr_t)poly_map_get(ctx->retained_uops, poly_ptr_hash(uop), uop, poly_ptr_eq);
  if (count == 0) return;
  if (count == 1) {
    poly_map_remove(ctx->retained_uops, poly_ptr_hash(uop), uop, poly_ptr_eq);
    /* Tinygrad 2026-08-22/a9069c177a9d UOp.__del__ removes the weak CSE row
     * only after the final strong reference dies (uop/ops.py:241-246). */
    ctx->collection_dirty = true;
    ctx->ir_collection_dirty = true;
  } else {
    poly_map_set(ctx->retained_uops, poly_ptr_hash(uop), uop, (void *)(count - 1), poly_ptr_eq);
  }
}

typedef struct {
  PolyUOp **items;
  PolyBuffer **buffers;
  int count;
  int capacity;
  bool failed;
} ResidencyRows;

static void collect_residency_row(const void *key, void *value, void *userdata) {
  ResidencyRows *rows = (ResidencyRows *)userdata;
  if (!rows || rows->failed || !key || !value) return;
  if (rows->count >= rows->capacity) {
    int capacity = rows->capacity ? rows->capacity * 2 : 32;
    PolyUOp **items = realloc(rows->items, (size_t)capacity * sizeof(*items));
    if (!items) {
      rows->failed = true;
      return;
    }
    rows->items = items;
    PolyBuffer **buffers = realloc(rows->buffers, (size_t)capacity * sizeof(*buffers));
    if (!buffers) {
      rows->failed = true;
      return;
    }
    rows->buffers = buffers;
    rows->capacity = capacity;
  }
  rows->items[rows->count] = (PolyUOp *)key;
  rows->buffers[rows->count] = (PolyBuffer *)value;
  rows->count++;
}

typedef struct {
  PolyCtx *ctx;
  PolyMap *marked;
  PolyMap *visited;
  PolyUOp **stack;
  int n_stack;
  int cap_stack;
  bool failed;
} ResidencyMarker;

static bool residency_mark_root(ResidencyMarker *marker, PolyUOp *root) {
  /* Tinygrad 2026-08-22 UOp weak ownership makes shared DAG reachability one
   * object-lifetime relation. This is the C mark traversal for that concept;
   * one visited set is shared by every declared root in this collection. */
  if (!marker || marker->failed) return false;
  if (!root) return true;
  if (marker->n_stack >= marker->cap_stack) {
    int capacity = marker->cap_stack ? marker->cap_stack * 2 : 256;
    PolyUOp **stack = realloc(marker->stack, (size_t)capacity * sizeof(*stack));
    if (!stack) return false;
    marker->stack = stack;
    marker->cap_stack = capacity;
  }
  marker->stack[marker->n_stack++] = root;
  while (marker->n_stack > 0) {
    PolyUOp *uop = marker->stack[--marker->n_stack];
    if (!uop || poly_map_get(marker->visited, poly_ptr_hash(uop), uop, poly_ptr_eq)) continue;
    poly_map_set(marker->visited, poly_ptr_hash(uop), uop, uop, poly_ptr_eq);
    if (poly_map_get(marker->ctx->buffers, poly_ptr_hash(uop), uop, poly_ptr_eq))
      poly_map_set(marker->marked, poly_ptr_hash(uop), uop, uop, poly_ptr_eq);
    if (uop->n_src > marker->cap_stack - marker->n_stack) {
      int capacity = marker->cap_stack;
      while (capacity - marker->n_stack < uop->n_src)
        capacity *= 2;
      PolyUOp **stack = realloc(marker->stack, (size_t)capacity * sizeof(*stack));
      if (!stack) return false;
      marker->stack = stack;
      marker->cap_stack = capacity;
    }
    /* Current Tinygrad get_call_outs_ins/_collect_bufs derives JIT residency
     * from CALL arguments. src[0] is executable code, not a buffer owner. */
    int first_src = uop->op == POLY_OP_CALL ? 1 : 0;
    for (int i = first_src; i < uop->n_src; i++)
      marker->stack[marker->n_stack++] = uop->src[i];
  }
  return true;
}

static void mark_retained_uop(const void *key, void *value, void *userdata) {
  (void)value;
  ResidencyMarker *marker = (ResidencyMarker *)userdata;
  if (!marker || marker->failed) return;
  marker->failed = !residency_mark_root(marker, (PolyUOp *)key);
}

typedef struct {
  PolyMap *live;
  PolyUOp **stack;
  int n_stack;
  int cap_stack;
  bool failed;
} IrMarker;

static bool ir_mark_root(IrMarker *marker, PolyUOp *root) {
  /* Python owns the complete source closure of each live Tinygrad UOp. This
   * non-moving C trace reproduces that relation without making CSE/cache rows
   * owners. CALL source zero is included because IR, unlike residency, owns
   * the executable graph too. */
  if (!marker || marker->failed) return false;
  if (!root) return true;
  if (marker->n_stack >= marker->cap_stack) {
    int capacity = marker->cap_stack ? marker->cap_stack * 2 : 256;
    PolyUOp **stack = realloc(marker->stack, (size_t)capacity * sizeof(*stack));
    if (!stack) return false;
    marker->stack = stack;
    marker->cap_stack = capacity;
  }
  marker->stack[marker->n_stack++] = root;
  while (marker->n_stack > 0) {
    PolyUOp *uop = marker->stack[--marker->n_stack];
    if (!uop || poly_map_get(marker->live, poly_ptr_hash(uop), uop, poly_ptr_eq)) continue;
    poly_map_set(marker->live, poly_ptr_hash(uop), uop, uop, poly_ptr_eq);
    if (uop->n_src > marker->cap_stack - marker->n_stack) {
      int capacity = marker->cap_stack;
      while (capacity - marker->n_stack < uop->n_src)
        capacity *= 2;
      PolyUOp **stack = realloc(marker->stack, (size_t)capacity * sizeof(*stack));
      if (!stack) return false;
      marker->stack = stack;
      marker->cap_stack = capacity;
    }
    for (int i = 0; i < uop->n_src; i++)
      marker->stack[marker->n_stack++] = uop->src[i];
    /* Tinygrad UOp args are Python owners. C copies ProgramInfo/KernelInfo
     * beside the UOp, so trace their symbolic UOp fields explicitly. */
    if (uop->arg.kind == POLY_ARG_PROGRAM_INFO && uop->arg.program_info) {
      const PolyProgramInfo *info = uop->arg.program_info;
      int n_meta = info->n_vars + 6;
      if (n_meta > marker->cap_stack - marker->n_stack) {
        int capacity = marker->cap_stack;
        while (capacity - marker->n_stack < n_meta)
          capacity *= 2;
        PolyUOp **stack = realloc(marker->stack, (size_t)capacity * sizeof(*stack));
        if (!stack) return false;
        marker->stack = stack;
        marker->cap_stack = capacity;
      }
      for (int i = 0; i < 3; i++) {
        marker->stack[marker->n_stack++] = info->global_exprs[i];
        marker->stack[marker->n_stack++] = info->local_exprs[i];
      }
      for (int i = 0; i < info->n_vars; i++)
        marker->stack[marker->n_stack++] = info->vars[i];
    } else if (uop->arg.kind == POLY_ARG_KERNEL_INFO && uop->arg.kernel_info &&
               uop->arg.kernel_info->estimates) {
      if (3 > marker->cap_stack - marker->n_stack) {
        int capacity = marker->cap_stack;
        while (capacity - marker->n_stack < 3)
          capacity *= 2;
        PolyUOp **stack = realloc(marker->stack, (size_t)capacity * sizeof(*stack));
        if (!stack) return false;
        marker->stack = stack;
        marker->cap_stack = capacity;
      }
      marker->stack[marker->n_stack++] = uop->arg.kernel_info->estimates->ops;
      marker->stack[marker->n_stack++] = uop->arg.kernel_info->estimates->lds;
      marker->stack[marker->n_stack++] = uop->arg.kernel_info->estimates->mem;
    }
  }
  return true;
}

static void mark_retained_ir(const void *key, void *value, void *userdata) {
  (void)value;
  IrMarker *marker = userdata;
  if (!marker || marker->failed) return;
  marker->failed = !ir_mark_root(marker, (PolyUOp *)key);
}

typedef struct {
  IrMarker *ir;
  PolyMap *visited;
  PolyBuffer **stack;
  int n_stack;
  int cap_stack;
  bool failed;
} BufferIrMarker;

static bool mark_buffer_ir_root(BufferIrMarker *marker, PolyBuffer *root) {
  if (!marker || marker->failed) return false;
  if (!root) return true;
  if (marker->n_stack >= marker->cap_stack) {
    int capacity = marker->cap_stack ? marker->cap_stack * 2 : 64;
    PolyBuffer **stack = realloc(marker->stack, (size_t)capacity * sizeof(*stack));
    if (!stack) return false;
    marker->stack = stack;
    marker->cap_stack = capacity;
  }
  marker->stack[marker->n_stack++] = root;
  while (marker->n_stack > 0) {
    PolyBuffer *buffer = marker->stack[--marker->n_stack];
    if (!buffer || poly_map_get(marker->visited, poly_ptr_hash(buffer), buffer, poly_ptr_eq))
      continue;
    poly_map_set(marker->visited, poly_ptr_hash(buffer), buffer, buffer, poly_ptr_eq);
    if (!ir_mark_root(marker->ir, buffer->device_uop) ||
        !ir_mark_root(marker->ir, buffer->memory_device_uop))
      return false;
    int n_owned = buffer->n_bufs + 2;
    if (n_owned > marker->cap_stack - marker->n_stack) {
      int capacity = marker->cap_stack;
      while (capacity - marker->n_stack < n_owned)
        capacity *= 2;
      PolyBuffer **stack = realloc(marker->stack, (size_t)capacity * sizeof(*stack));
      if (!stack) return false;
      marker->stack = stack;
      marker->cap_stack = capacity;
    }
    marker->stack[marker->n_stack++] = buffer->base;
    marker->stack[marker->n_stack++] = buffer->src;
    for (int i = 0; i < buffer->n_bufs; i++)
      marker->stack[marker->n_stack++] = buffer->bufs[i];
  }
  return true;
}

static void mark_buffer_row_ir(const void *key, void *value, void *userdata) {
  BufferIrMarker *marker = userdata;
  if (!marker || marker->failed) return;
  /* Tinygrad Buffer objects own their exact DEVICE identity. Polygrad keeps
   * equivalent runtime metadata outside BUFFER.src, so collection must close
   * that C-only ownership edge before evicting weak CSE rows. */
  marker->failed = !ir_mark_root(marker->ir, (PolyUOp *)key) ||
                   !mark_buffer_ir_root(marker, (PolyBuffer *)value);
}

static int mark_live_ir(PolyCtx *ctx, PolyUOp *transient_root, PolyMap *live) {
  IrMarker marker = {.live = live};
  for (int i = 0; !marker.failed && i < ctx->n_tensors; i++) {
    PolyTensor *tensor = ctx->tensors[i];
    if (!tensor || tensor->owner_refs == 0) continue;
    marker.failed =
        !ir_mark_root(&marker, tensor->uop_logical) || !ir_mark_root(&marker, tensor->uop_physical);
  }
  for (int i = 0; !marker.failed && i < ctx->n_entries; i++)
    if (ctx->entries[i]) marker.failed = !ir_mark_root(&marker, ctx->entries[i]->buffer);
  for (int i = 0; !marker.failed && i < ctx->n_ep; i++)
    marker.failed = !ir_mark_root(&marker, ctx->ep[i].sink);
  if (!marker.failed && transient_root) marker.failed = !ir_mark_root(&marker, transient_root);
  if (!marker.failed) poly_map_foreach(ctx->retained_uops, mark_retained_ir, &marker);
  BufferIrMarker buffers = {.ir = &marker, .visited = poly_map_new(64)};
  if (!marker.failed && !buffers.visited)
    marker.failed = true;
  else if (!marker.failed)
    poly_map_foreach(ctx->buffers, mark_buffer_row_ir, &buffers);
  marker.failed = marker.failed || buffers.failed;
  free(buffers.stack);
  poly_map_destroy(buffers.visited);
  free(marker.stack);
  return marker.failed ? -1 : 0;
}

static int poly_ctx_collect_with_root(PolyCtx *ctx, PolyUOp *transient_root, bool collect_ir) {
  if (!ctx || ctx->collecting) return -1;
  ctx->collecting = true;
  PolyMap *marked = poly_map_new(64);
  ResidencyMarker roots = {
      .ctx = ctx,
      .marked = marked,
      .visited = poly_map_new(256),
  };
  bool failed = marked == NULL || roots.visited == NULL;
  for (int i = 0; !failed && i < ctx->n_tensors; i++) {
    PolyTensor *tensor = ctx->tensors[i];
    if (tensor && tensor->owner_refs > 0)
      failed = !residency_mark_root(&roots, tensor->uop_physical);
  }
  for (int i = 0; !failed && i < ctx->n_entries; i++)
    if (ctx->entries[i]) failed = !residency_mark_root(&roots, ctx->entries[i]->buffer);
  for (int i = 0; !failed && i < ctx->n_ep; i++)
    failed = !residency_mark_root(&roots, ctx->ep[i].sink);
  if (!failed && transient_root) failed = !residency_mark_root(&roots, transient_root);
  roots.failed = failed;
  if (!failed) poly_map_foreach(ctx->retained_uops, mark_retained_uop, &roots);
  failed = failed || roots.failed;
  free(roots.stack);
  poly_map_destroy(roots.visited);

  ResidencyRows rows = {0};
  if (!failed) poly_map_foreach(ctx->buffers, collect_residency_row, &rows);
  failed = failed || rows.failed;
  if (!failed) {
    /* Existing view and multi-buffer rows point at their storage owner. Close
     * that runtime alias relation without creating buffer metadata. */
    PolyMap *row_by_buffer = poly_map_new((size_t)rows.count * 2 + 16);
    int *alias_stack = rows.count ? malloc((size_t)rows.count * sizeof(*alias_stack)) : NULL;
    if (!row_by_buffer || (rows.count && !alias_stack)) {
      failed = true;
    } else {
      int n_alias = 0;
      for (int i = 0; i < rows.count; i++) {
        poly_map_set(
            row_by_buffer, poly_ptr_hash(rows.buffers[i]), rows.buffers[i],
            (void *)(uintptr_t)(i + 1), poly_ptr_eq
        );
        if (poly_map_get(marked, poly_ptr_hash(rows.items[i]), rows.items[i], poly_ptr_eq))
          alias_stack[n_alias++] = i;
      }
      while (n_alias > 0) {
        PolyBuffer *buffer = rows.buffers[alias_stack[--n_alias]];
        int n_owned = buffer->n_bufs + 2;
        for (int i = 0; i < n_owned; i++) {
          PolyBuffer *owned = i == 0 ? buffer->base : i == 1 ? buffer->src : buffer->bufs[i - 2];
          uintptr_t row =
              owned
                  ? (uintptr_t)poly_map_get(row_by_buffer, poly_ptr_hash(owned), owned, poly_ptr_eq)
                  : 0;
          if (!row) continue;
          int j = (int)row - 1;
          if (poly_map_get(marked, poly_ptr_hash(rows.items[j]), rows.items[j], poly_ptr_eq))
            continue;
          poly_map_set(
              marked, poly_ptr_hash(rows.items[j]), rows.items[j], rows.items[j], poly_ptr_eq
          );
          alias_stack[n_alias++] = j;
        }
      }
    }
    free(alias_stack);
    poly_map_destroy(row_by_buffer);
  }
  if (!failed) {
    for (int i = 0; i < rows.count; i++)
      if (!poly_map_get(marked, poly_ptr_hash(rows.items[i]), rows.items[i], poly_ptr_eq))
        poly_buffer_remove(ctx, rows.items[i]);
    ctx->collection_dirty = false;
  }
  free(rows.items);
  free(rows.buffers);
  poly_map_destroy(marked);
  PolyMap *live_ir = NULL;
  if (!failed && collect_ir) {
    live_ir = poly_map_new(256);
    PolyUOp **shape_roots = NULL;
    int n_shape_roots = 0;
    if (!live_ir || mark_live_ir(ctx, transient_root, live_ir) != 0 ||
        poly_shape_cache_live_value_roots(ctx, live_ir, &shape_roots, &n_shape_roots) != 0)
      failed = true;
    IrMarker shape_marker = {.live = live_ir};
    for (int i = 0; !failed && i < n_shape_roots; i++)
      failed = !ir_mark_root(&shape_marker, shape_roots[i]);
    free(shape_marker.stack);
    free(shape_roots);
    if (!failed && (poly_shape_cache_evict_unmarked(ctx, live_ir) != 0 ||
                    poly_uop_cse_evict_unmarked(ctx, live_ir) != 0))
      failed = true;
    if (!failed) {
      ctx->ir_collection_dirty = false;
      ctx->ir_collection_baseline_bytes = ctx->uop_storage_bytes;
    }
  }
  poly_map_destroy(live_ir);
  ctx->collecting = false;
  return failed ? -1 : 0;
}

int poly_ctx_collect(PolyCtx *ctx) {
  return poly_ctx_collect_with_root(ctx, NULL, true);
}

static bool poly_ctx_ir_collection_due(const PolyCtx *ctx) {
  if (!ctx || !ctx->ir_collection_dirty) return false;
  if (ctx->uop_storage_bytes < ctx->ir_collection_baseline_bytes) return true;
  return ctx->uop_storage_bytes - ctx->ir_collection_baseline_bytes >=
         POLY_IR_COLLECTION_MIN_GROWTH;
}

int poly_ctx_collect_at_safe_point(PolyCtx *ctx) {
  if (!ctx) return -1;
  bool collect_ir = poly_ctx_ir_collection_due(ctx);
  if (!ctx->collection_dirty && !collect_ir) return 0;
  return poly_ctx_collect_with_root(ctx, NULL, collect_ir);
}

int poly_ctx_collect_before_allocation(PolyCtx *ctx, PolyUOp *transient_root) {
  if (!ctx || !ctx->collection_dirty) return ctx ? 0 : -1;
  /* poly_run_linear retains its complete LINEAR, so allocation-time
   * collection preserves every current CALL argument without a pre-run scan. */
  return poly_ctx_collect_with_root(ctx, transient_root, false);
}

bool poly_ctx_owns_ptr(PolyCtx *ctx, const void *p) {
  if (!ctx || !p) return false;
  if (poly_uop_storage_contains(ctx, p)) return true;
  uintptr_t addr = (uintptr_t)p;
  for (PolyArenaBlock *b = ctx->arena->head; b; b = b->next) {
    uintptr_t start = (uintptr_t)b->data;
    uintptr_t end = start + b->used;
    if (addr >= start && addr < end) return true;
  }
  return false;
}

void poly_ctx_set_preferred_device(PolyCtx *ctx, PolyDevice device) {
  if (!ctx) return;
  ctx->preferred_device = device;
}

PolyDevice poly_ctx_get_preferred_device(PolyCtx *ctx) {
  return ctx ? ctx->preferred_device : POLY_DEVICE_AUTO;
}

int poly_ctx_set_logical_policy(PolyCtx *ctx, PolyLogicalPolicy policy) {
  /* Polygrad logical/placement boundary: this default is copied only by
   * subsequently constructed Tensor handles. */
  if (!ctx || policy < POLY_LOGICAL_NEVER || policy > POLY_LOGICAL_UNTIL_REALIZE) return -1;
  ctx->logical_policy = policy;
  return 0;
}

PolyLogicalPolicy poly_ctx_get_logical_policy(const PolyCtx *ctx) {
  return ctx ? ctx->logical_policy : POLY_LOGICAL_UNTIL_REALIZE;
}

void poly_ctx_set_frontend_buffer_release(PolyCtx *ctx, PolyFrontendBufferReleaseFn fn) {
  if (!ctx) return;
  ctx->frontend_buffer_release = fn;
}

PolyArena *poly_ctx_arena(PolyCtx *ctx) {
  return ctx->arena;
}
PolyMap *poly_ctx_shape_cache(PolyCtx *ctx) {
  return ctx->shape_cache;
}

typedef struct {
  size_t current;
  size_t source;
} BufferByteStats;

static void accum_buffer_bytes(const void *key, void *value, void *userdata) {
  (void)key;
  BufferByteStats *stats = (BufferByteStats *)userdata;
  PolyBuffer *b = (PolyBuffer *)value;
  if (!stats || !b) return;
  if (b->owned) stats->current += b->nbytes;
  if (b->src && b->src->owned) stats->source += b->src->nbytes;
}

int poly_ctx_stats(PolyCtx *ctx, PolyCtxStats *out) {
  if (!ctx || !out) return -1;
  /* Tinygrad 2026-08-22/a9069c177a9d helpers.py:298-306 exposes plain
   * GlobalCounters fields. A counter read must not change graph lifetime. */
  memset(out, 0, sizeof(*out));
  out->arena_bytes = poly_arena_used(ctx->arena) + ctx->uop_storage_bytes;
  out->arena_high_water = poly_arena_high_water(ctx->arena) + ctx->uop_storage_high_water;
  out->scratch_bytes = poly_arena_used(ctx->scratch);
  out->scratch_high_water = poly_arena_high_water(ctx->scratch);
  out->cse_entries = poly_map_len(ctx->cse);
  out->to_program_cache_entries = poly_map_len(ctx->to_program_cache);
  out->runtime_cache_entries = poly_map_len(ctx->runtime_cache);
  out->runtime_artifact_entries = ctx->runtime_artifact_entries;
  out->shape_cache_entries = poly_map_len(ctx->shape_cache);
  out->buffer_entries = poly_map_len(ctx->buffers);
  BufferByteStats buf_stats = {0};
  poly_map_foreach(ctx->buffers, accum_buffer_bytes, &buf_stats);
  out->buffer_owned_current_bytes = buf_stats.current;
  out->buffer_owned_source_bytes = buf_stats.source;
  out->buffer_owned_bytes = buf_stats.current + buf_stats.source;
  out->tensor_records = (size_t)ctx->n_tensors;
  out->registry_entries = (size_t)ctx->n_entries;
  out->entrypoint_entries = (size_t)ctx->n_ep;
  out->compiled_artifact_bytes = poly_runtime_cache_artifact_bytes(ctx);
  out->launch_count = ctx->launch_count;
  out->runtime_cache_hits = ctx->runtime_cache_hits;
  out->runtime_cache_misses = ctx->runtime_cache_misses;
  out->buffer_read_count = ctx->buffer_read_count;
  out->buffer_read_bytes = ctx->buffer_read_bytes;
  out->buffer_write_count = ctx->buffer_write_count;
  out->buffer_write_bytes = ctx->buffer_write_bytes;
  out->buffer_copy_count = ctx->buffer_copy_count;
  out->buffer_copy_bytes = ctx->buffer_copy_bytes;
  out->global_ops = ctx->global_ops;
  out->global_mem = ctx->global_mem;
  out->time_sum_s = ctx->time_sum_s;
  out->kernel_count = ctx->kernel_count;
  out->mem_used = ctx->mem_used;
  return 0;
}

void poly_ctx_reset_counters(PolyCtx *ctx) {
  if (!ctx) return;
  ctx->global_ops = 0;
  ctx->global_mem = 0;
  ctx->time_sum_s = 0.0;
  ctx->kernel_count = 0;
}

uint64_t poly_ctx_mem_used_for_device(PolyCtx *ctx, PolyDevice device) {
  if (!ctx || device < POLY_DEVICE_AUTO || device > POLY_DEVICE_DISK) return 0;
  return ctx->mem_used_per_device[device];
}

typedef struct {
  PolyUOp *device_uop;
  uint64_t bytes;
} PolyDeviceMemoryEntry;

static PolyDeviceMemoryEntry *poly_ctx_memory_entry(
    PolyCtx *ctx,
    PolyUOp *device_uop,
    bool create
) {
  if (!ctx || !ctx->mem_used_by_device || !device_uop) return NULL;
  PolyDeviceMemoryEntry *entry =
      poly_map_get(ctx->mem_used_by_device, poly_ptr_hash(device_uop), device_uop, poly_ptr_eq);
  if (entry || !create) return entry;
  entry = poly_arena_alloc(ctx->arena, sizeof(*entry), _Alignof(PolyDeviceMemoryEntry));
  if (!entry) return NULL;
  *entry = (PolyDeviceMemoryEntry){.device_uop = device_uop, .bytes = 0};
  poly_map_set(ctx->mem_used_by_device, poly_ptr_hash(device_uop), device_uop, entry, poly_ptr_eq);
  return entry;
}

uint64_t poly_ctx_mem_used_for_device_uop(PolyCtx *ctx, PolyUOp *device_uop) {
  PolyDeviceMemoryEntry *entry = poly_ctx_memory_entry(ctx, device_uop, false);
  return entry ? entry->bytes : 0;
}

void poly_ctx_record_memory_alloc_exact(
    PolyCtx *ctx,
    PolyUOp *device_uop,
    PolyDevice backend,
    size_t nbytes
) {
  if (!ctx || nbytes == 0) return;
  uint64_t bytes = (uint64_t)nbytes;
  ctx->mem_used = UINT64_MAX - ctx->mem_used < bytes ? UINT64_MAX : ctx->mem_used + bytes;
  if (backend >= POLY_DEVICE_AUTO && backend <= POLY_DEVICE_DISK) {
    uint64_t *per_device = &ctx->mem_used_per_device[backend];
    *per_device = UINT64_MAX - *per_device < bytes ? UINT64_MAX : *per_device + bytes;
  }
  PolyDeviceMemoryEntry *entry = poly_ctx_memory_entry(ctx, device_uop, true);
  if (entry) entry->bytes = UINT64_MAX - entry->bytes < bytes ? UINT64_MAX : entry->bytes + bytes;
}

void poly_ctx_record_memory_free_exact(
    PolyCtx *ctx,
    PolyUOp *device_uop,
    PolyDevice backend,
    size_t nbytes
) {
  if (!ctx || nbytes == 0) return;
  uint64_t bytes = (uint64_t)nbytes;
  ctx->mem_used = ctx->mem_used >= bytes ? ctx->mem_used - bytes : 0;
  if (backend >= POLY_DEVICE_AUTO && backend <= POLY_DEVICE_DISK) {
    uint64_t *per_device = &ctx->mem_used_per_device[backend];
    *per_device = *per_device >= bytes ? *per_device - bytes : 0;
  }
  PolyDeviceMemoryEntry *entry = poly_ctx_memory_entry(ctx, device_uop, false);
  if (entry) entry->bytes = entry->bytes >= bytes ? entry->bytes - bytes : 0;
}

void poly_ctx_record_memory_alloc(PolyCtx *ctx, PolyDevice device, size_t nbytes) {
  poly_ctx_record_memory_alloc_exact(ctx, poly_device_uop(ctx, device), device, nbytes);
}

void poly_ctx_record_memory_free(PolyCtx *ctx, PolyDevice device, size_t nbytes) {
  poly_ctx_record_memory_free_exact(ctx, poly_device_uop(ctx, device), device, nbytes);
}

#ifdef __EMSCRIPTEN__
/* wasm_common.js reads this public struct manually. Keep the ABI facts
 * compile-checked instead of relying on an unverified offset table. */
_Static_assert(offsetof(PolyCtxStats, global_ops) == 104, "wasm PolyCtxStats.global_ops offset");
_Static_assert(offsetof(PolyCtxStats, global_mem) == 112, "wasm PolyCtxStats.global_mem offset");
_Static_assert(offsetof(PolyCtxStats, time_sum_s) == 120, "wasm PolyCtxStats.time_sum_s offset");
_Static_assert(
    offsetof(PolyCtxStats, kernel_count) == 128,
    "wasm PolyCtxStats.kernel_count offset"
);
_Static_assert(offsetof(PolyCtxStats, mem_used) == 136, "wasm PolyCtxStats.mem_used offset");
_Static_assert(sizeof(PolyCtxStats) == 144, "wasm PolyCtxStats size");
#endif

PolyScratchMark poly_ctx_scratch_mark(PolyCtx *ctx) {
  return poly_arena_mark(ctx ? ctx->scratch : NULL);
}

void poly_ctx_scratch_rewind(PolyCtx *ctx, PolyScratchMark mark) {
  if (!ctx) return;
  poly_arena_rewind(ctx->scratch, mark);
}

void *poly_ctx_scratch_alloc(PolyCtx *ctx, size_t size, size_t align) {
  if (!ctx || !ctx->scratch) return NULL;
  return poly_arena_alloc(ctx->scratch, size, align);
}

int64_t poly_ctx_next_unique_id(PolyCtx *ctx) {
  return ctx ? ctx->next_unique_id++ : 0;
}

void poly_ctx_reserve_unique_id(PolyCtx *ctx, int64_t id) {
  if (!ctx || id < 0) return;
  if (ctx->next_unique_id <= id) ctx->next_unique_id = id + 1;
}

void poly_ctx_reserve_buf_tag(PolyCtx *ctx, int32_t tag) {
  if (!ctx || tag <= 0) return;
  if (ctx->next_buf_tag <= tag) ctx->next_buf_tag = tag + 1;
}
