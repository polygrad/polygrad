/*
 * arena.h — Arena allocator internals
 */
#ifndef POLY_ARENA_H
#define POLY_ARENA_H

#include <stddef.h>
#include <stdint.h>

typedef struct PolyArenaBlock {
  struct PolyArenaBlock *next;
  size_t cap;
  size_t used;
  uint8_t data[];
} PolyArenaBlock;

struct PolyArena {
  PolyArenaBlock *head;
  size_t total_used;
};

typedef struct {
  PolyArenaBlock *head;
  size_t used;
  size_t total_used;
} PolyArenaMark;

PolyArenaMark poly_arena_mark(PolyArena *a);
void poly_arena_rewind(PolyArena *a, PolyArenaMark mark);

#endif /* POLY_ARENA_H */
