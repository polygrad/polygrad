/* ctx.c -- PolyCtx lifecycle: create, destroy, accessors */

#include "ctx.h"
#include "device.h"
#include "engine/schedule.h"
#include "utils.h"
#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include <string.h>

/* Defined in ops.c */
void poly_init_group_ops(void);

/* Optional cleanup hooks (defined in other files, linked weakly). */
void poly_frontend_ctx_cleanup(PolyCtx *ctx) __attribute__((weak));
void poly_tensor_ctx_cleanup(PolyCtx *ctx) __attribute__((weak));
void poly_schedule_ctx_cleanup(PolyCtx *ctx) __attribute__((weak));

static void free_buffer_entry(const void *key, void *value, void *userdata) {
  (void)key;
  PolyCtx *ctx = (PolyCtx *)userdata;
  /* Free current + src chain. PolyBuffer struct itself is arena-allocated. */
  poly_buffer_free_chain(ctx, (PolyBuffer *)value);
}

PolyCtx *poly_ctx_new(void) {
  poly_init_group_ops();
  PolyCtx *ctx = malloc(sizeof(PolyCtx));
  if (!ctx) return NULL;
  ctx->arena = poly_arena_new(0);
  ctx->scratch = poly_arena_new(0);
  ctx->cse = poly_map_new(256);
  ctx->schedule_cache = poly_map_new(16);
  ctx->to_program_cache = poly_map_new(16);
  ctx->runtime_cache = poly_map_new(16);
  ctx->runtime_artifact_entries = 0;
  ctx->runtime_artifact_live_bytes = 0;
  ctx->launch_count = 0;
  ctx->schedule_cache_hits = 0;
  ctx->schedule_cache_misses = 0;
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
  ctx->tensors = NULL;
  ctx->n_tensors = 0;
  ctx->tensors_cap = 0;
  ctx->next_tensor_order = 1;
  ctx->active_jit_capture = NULL;
  ctx->name_map = poly_map_new(16);
  if (!ctx->arena || !ctx->scratch || !ctx->cse || !ctx->schedule_cache || !ctx->to_program_cache ||
      !ctx->runtime_cache || !ctx->mem_used_by_device || !ctx->shape_cache || !ctx->buffers ||
      !ctx->name_map) {
    if (ctx->arena) poly_arena_destroy(ctx->arena);
    if (ctx->scratch) poly_arena_destroy(ctx->scratch);
    if (ctx->cse) poly_map_destroy(ctx->cse);
    if (ctx->schedule_cache) poly_map_destroy(ctx->schedule_cache);
    if (ctx->to_program_cache) poly_map_destroy(ctx->to_program_cache);
    if (ctx->runtime_cache) poly_map_destroy(ctx->runtime_cache);
    if (ctx->mem_used_by_device) poly_map_destroy(ctx->mem_used_by_device);
    if (ctx->shape_cache) poly_map_destroy(ctx->shape_cache);
    if (ctx->buffers) poly_map_destroy(ctx->buffers);
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
  if (poly_schedule_ctx_cleanup) poly_schedule_ctx_cleanup(ctx);
  poly_map_destroy(ctx->schedule_cache);
  poly_map_destroy(ctx->to_program_cache);
  poly_map_destroy(ctx->runtime_cache);
  poly_map_destroy(ctx->shape_cache);
  /* Free owned buffer ptrs before destroying the map. */
  poly_map_foreach(ctx->buffers, free_buffer_entry, ctx);
  poly_map_destroy(ctx->buffers);
  poly_map_destroy(ctx->mem_used_by_device);
  free(ctx->tensors);
  poly_map_destroy(ctx->name_map);
  poly_map_destroy(ctx->cse);
  free(ctx->entries);
  free(ctx->ep);
  poly_arena_destroy(ctx->scratch);
  poly_arena_destroy(ctx->arena);
  free(ctx);
}

bool poly_ctx_owns_ptr(PolyCtx *ctx, const void *p) {
  if (!ctx || !p) return false;
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

void poly_ctx_set_frontend_buffer_release(PolyCtx *ctx, PolyFrontendBufferReleaseFn fn) {
  if (!ctx) return;
  ctx->frontend_buffer_release = fn;
}

PolyArena *poly_ctx_arena(PolyCtx *ctx) { return ctx->arena; }
PolyMap *poly_ctx_shape_cache(PolyCtx *ctx) { return ctx->shape_cache; }

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
  memset(out, 0, sizeof(*out));
  out->arena_bytes = poly_arena_used(ctx->arena);
  out->arena_high_water = poly_arena_high_water(ctx->arena);
  out->scratch_bytes = poly_arena_used(ctx->scratch);
  out->scratch_high_water = poly_arena_high_water(ctx->scratch);
  out->cse_entries = poly_map_len(ctx->cse);
  out->schedule_cache_entries = poly_map_len(ctx->schedule_cache);
  out->to_program_cache_entries = poly_map_len(ctx->to_program_cache);
  out->runtime_cache_entries = poly_map_len(ctx->runtime_cache);
  out->program_cache_entries = out->runtime_cache_entries;
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
  out->schedule_cache_hits = ctx->schedule_cache_hits;
  out->schedule_cache_misses = ctx->schedule_cache_misses;
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
_Static_assert(offsetof(PolyCtxStats, global_ops) == 120, "wasm PolyCtxStats.global_ops offset");
_Static_assert(offsetof(PolyCtxStats, global_mem) == 128, "wasm PolyCtxStats.global_mem offset");
_Static_assert(offsetof(PolyCtxStats, time_sum_s) == 136, "wasm PolyCtxStats.time_sum_s offset");
_Static_assert(offsetof(PolyCtxStats, kernel_count) == 144, "wasm PolyCtxStats.kernel_count offset");
_Static_assert(offsetof(PolyCtxStats, mem_used) == 152, "wasm PolyCtxStats.mem_used offset");
_Static_assert(sizeof(PolyCtxStats) == 160, "wasm PolyCtxStats size");
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
