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

#define ARENA_DEFAULT_CAP (64 * 1024)  /* 64 KB blocks */

static PolyArenaBlock *block_new(size_t cap) {
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
  if (!a->head) { free(a); return NULL; }
  a->total_used = 0;
  return a;
}

void *poly_arena_alloc(PolyArena *a, size_t size, size_t align) {
  if (align == 0) align = 8;
  PolyArenaBlock *b = a->head;

  /* align the current offset */
  size_t offset = (b->used + align - 1) & ~(align - 1);

  if (offset + size > b->cap) {
    /* need a new block — at least big enough for this allocation */
    size_t new_cap = b->cap * 2;
    if (new_cap < size + align) new_cap = size + align;
    PolyArenaBlock *nb = block_new(new_cap);
    if (!nb) return NULL;
    nb->next = b;
    a->head = nb;
    b = nb;
    offset = 0;
  }

  void *ptr = b->data + offset;
  b->used = offset + size;
  a->total_used += size;
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
  return a->total_used;
}
