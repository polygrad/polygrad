/*
 * arena.c — Arena allocator: bulk allocation, bulk free
 *
 * All UOp nodes, src arrays, and arg data are allocated from the arena.
 * The entire arena is freed at once when the context is destroyed.
 */

#include "polygrad.h"
#include "arena.h"
#include <stdlib.h>
#include <string.h>

#define ARENA_DEFAULT_CAP (64 * 1024) /* 64 KB blocks */

static PolyArenaBlock *block_new(size_t cap) {
  if (cap > SIZE_MAX - sizeof(PolyArenaBlock)) return NULL;
  PolyArenaBlock *b = malloc(sizeof(PolyArenaBlock) + cap);
  if (!b) return NULL;
  b->next = NULL;
  b->cap = cap;
  b->used = 0;
  return b;
}

PolyArena *poly_arena_new(size_t initial_cap) {
  if (initial_cap == 0) initial_cap = ARENA_DEFAULT_CAP;
  PolyArena *a = malloc(sizeof(PolyArena));
  if (!a) return NULL;
  a->head = block_new(initial_cap);
  if (!a->head) {
    free(a);
    return NULL;
  }
  a->total_used = 0;
  a->high_water = 0;
  return a;
}

void *poly_arena_alloc(PolyArena *a, size_t size, size_t align) {
  if (!a || !a->head || size > SIZE_MAX - a->total_used) return NULL;
  if (align == 0) align = 8;
  if ((align & (align - 1)) != 0) return NULL;
  PolyArenaBlock *b = a->head;

  /* Align the address, not the offset: the flexible payload starts after
   * three machine words, so it is not even 8-byte aligned on wasm32. */
  size_t padding = (-(uintptr_t)(b->data + b->used)) & (align - 1);

  if (padding > b->cap - b->used || size > b->cap - b->used - padding) {
    size_t limit = SIZE_MAX - sizeof(PolyArenaBlock);
    if (align - 1 > limit || size > limit - (align - 1)) return NULL;
    size_t needed = size + align - 1;
    size_t new_cap = b->cap <= limit / 2 ? b->cap * 2 : limit;
    if (new_cap < needed) new_cap = needed;
    PolyArenaBlock *nb = block_new(new_cap);
    if (!nb) return NULL;
    nb->next = b;
    a->head = nb;
    b = nb;
    padding = (-(uintptr_t)b->data) & (align - 1);
  }

  size_t offset = b->used + padding;
  void *ptr = b->data + offset;
  b->used = offset + size;
  a->total_used += size;
  if (a->total_used > a->high_water) a->high_water = a->total_used;
  return ptr;
}

void poly_arena_reset(PolyArena *a) {
  /* free all blocks except the first, then reset it */
  PolyArenaBlock *b = a->head;
  while (b->next) {
    PolyArenaBlock *prev = b->next;
    b->next = prev->next;
    free(prev);
  }
  b->used = 0;
  a->total_used = 0;
  a->high_water = 0;
}

void poly_arena_destroy(PolyArena *a) {
  PolyArenaBlock *b = a->head;
  while (b) {
    PolyArenaBlock *next = b->next;
    free(b);
    b = next;
  }
  free(a);
}

size_t poly_arena_used(PolyArena *a) {
  return a ? a->total_used : 0;
}

size_t poly_arena_high_water(PolyArena *a) {
  return a ? a->high_water : 0;
}

PolyArenaMark poly_arena_mark(PolyArena *a) {
  PolyArenaMark mark = {0};
  if (!a || !a->head) return mark;
  mark.head = a->head;
  mark.used = a->head->used;
  mark.total_used = a->total_used;
  return mark;
}

void poly_arena_rewind(PolyArena *a, PolyArenaMark mark) {
  if (!a || !mark.head) return;

  PolyArenaBlock *scan = a->head;
  while (scan && scan != mark.head)
    scan = scan->next;
  if (!scan) return;

  while (a->head && a->head != mark.head) {
    PolyArenaBlock *next = a->head->next;
    free(a->head);
    a->head = next;
  }

  if (a->head) {
    a->head->used = mark.used;
    a->total_used = mark.total_used;
  }
}
