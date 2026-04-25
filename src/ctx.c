/* ctx.c -- PolyCtx lifecycle: create, destroy, accessors */

#include "ctx.h"
#include "device.h"
#include "utils.h"
#include <stdlib.h>
#include <stdio.h>

/* Defined in ops.c */
void poly_init_group_ops(void);

/* Optional cleanup hooks (defined in other files, linked weakly). */
void poly_frontend_ctx_cleanup(PolyCtx *ctx) __attribute__((weak));

static void free_cached_kernel(const void *key, void *value, void *userdata) {
  (void)key;
  (void)userdata;
  PolyCachedKernel *ck = value;
  free(ck->bytes);
  free(ck);
}

static void free_buffer_entry(const void *key, void *value, void *userdata) {
  (void)key;
  (void)userdata;
  /* Free current + src chain. PolyBuffer struct itself is arena-allocated. */
  poly_buffer_free_chain((PolyBuffer *)value);
}

PolyCtx *poly_ctx_new(void) {
  poly_init_group_ops();
  PolyCtx *ctx = malloc(sizeof(PolyCtx));
  if (!ctx) return NULL;
  ctx->arena = poly_arena_new(0);
  ctx->cse = poly_map_new(256);
  ctx->kernel_cache = poly_map_new(16);
  ctx->shape_cache = poly_map_new(64);
  ctx->buffers = poly_map_new(64);
  ctx->name_map = poly_map_new(16);
  if (!ctx->arena || !ctx->cse || !ctx->kernel_cache || !ctx->shape_cache || !ctx->buffers ||
      !ctx->name_map) {
    if (ctx->arena) poly_arena_destroy(ctx->arena);
    if (ctx->cse) poly_map_destroy(ctx->cse);
    if (ctx->kernel_cache) poly_map_destroy(ctx->kernel_cache);
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
  ctx->preferred_device = POLY_DEVICE_AUTO;
  return ctx;
}

void poly_ctx_destroy(PolyCtx *ctx) {
  if (!ctx) return;
  if (poly_frontend_ctx_cleanup) poly_frontend_ctx_cleanup(ctx);
  poly_map_foreach(ctx->kernel_cache, free_cached_kernel, NULL);
  poly_map_destroy(ctx->kernel_cache);
  poly_map_destroy(ctx->shape_cache);
  /* Free owned buffer ptrs before destroying the map. */
  poly_map_foreach(ctx->buffers, free_buffer_entry, NULL);
  poly_map_destroy(ctx->buffers);
  poly_map_destroy(ctx->name_map);
  poly_map_destroy(ctx->cse);
  free(ctx->entries);
  free(ctx->ep);
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

PolyMap *poly_ctx_kernel_cache(PolyCtx *ctx) { return ctx->kernel_cache; }
PolyArena *poly_ctx_arena(PolyCtx *ctx) { return ctx->arena; }
PolyMap *poly_ctx_shape_cache(PolyCtx *ctx) { return ctx->shape_cache; }
