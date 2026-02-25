/*
 * hashmap.c — Open-addressing hash map (Robin Hood hashing)
 *
 * Used for CSE deduplication: key is (op, dtype, src[], arg),
 * value is the existing UOp pointer.
 */

#include "polygrad.h"
#include <stdlib.h>
#include <string.h>

typedef struct {
  uint32_t hash;
  const void *key;
  void *value;
  bool occupied;
} PolyMapEntry;

struct PolyMap {
  PolyMapEntry *entries;
  size_t cap;
  size_t len;
};

static size_t probe_distance(PolyMap *m, uint32_t hash, size_t slot) {
  size_t ideal = hash & (m->cap - 1);
  return (slot + m->cap - ideal) & (m->cap - 1);
}

PolyMap *poly_map_new(size_t initial_cap) {
  if (initial_cap < 16) initial_cap = 16;
  /* round up to power of 2 */
  size_t cap = 1;
  while (cap < initial_cap) cap <<= 1;

  PolyMap *m = malloc(sizeof(PolyMap));
  if (!m) return NULL;
  m->entries = calloc(cap, sizeof(PolyMapEntry));
  if (!m->entries) { free(m); return NULL; }
  m->cap = cap;
  m->len = 0;
  return m;
}

void poly_map_destroy(PolyMap *m) {
  free(m->entries);
  free(m);
}

size_t poly_map_len(PolyMap *m) {
  return m->len;
}

static void map_grow(PolyMap *m);

void *poly_map_get(PolyMap *m, uint32_t hash, const void *key,
                   bool (*eq)(const void *a, const void *b))
{
  size_t slot = hash & (m->cap - 1);
  size_t dist = 0;
  for (;;) {
    PolyMapEntry *e = &m->entries[slot];
    if (!e->occupied) return NULL;
    if (probe_distance(m, e->hash, slot) < dist) return NULL;
    if (e->hash == hash && eq(e->key, key)) return e->value;
    slot = (slot + 1) & (m->cap - 1);
    dist++;
  }
}

void poly_map_set(PolyMap *m, uint32_t hash, const void *key, void *value,
                  bool (*eq)(const void *a, const void *b))
{
  if (m->len * 4 >= m->cap * 3) map_grow(m);

  size_t slot = hash & (m->cap - 1);
  PolyMapEntry incoming = { hash, key, value, true };
  size_t dist = 0;

  for (;;) {
    PolyMapEntry *e = &m->entries[slot];
    if (!e->occupied) {
      *e = incoming;
      m->len++;
      return;
    }
    /* update existing key */
    if (e->hash == hash && eq(e->key, key)) {
      e->value = value;
      return;
    }
    /* Robin Hood: swap if current entry is richer */
    size_t existing_dist = probe_distance(m, e->hash, slot);
    if (existing_dist < dist) {
      PolyMapEntry tmp = *e;
      *e = incoming;
      incoming = tmp;
      dist = existing_dist;
    }
    slot = (slot + 1) & (m->cap - 1);
    dist++;
  }
}

void poly_map_remove(PolyMap *m, uint32_t hash, const void *key,
                     bool (*eq)(const void *a, const void *b))
{
  size_t slot = hash & (m->cap - 1);
  size_t dist = 0;
  for (;;) {
    PolyMapEntry *e = &m->entries[slot];
    if (!e->occupied) return;
    if (probe_distance(m, e->hash, slot) < dist) return;
    if (e->hash == hash && eq(e->key, key)) {
      /* found: backward-shift delete */
      m->len--;
      for (;;) {
        size_t next = (slot + 1) & (m->cap - 1);
        PolyMapEntry *ne = &m->entries[next];
        if (!ne->occupied || probe_distance(m, ne->hash, next) == 0) {
          m->entries[slot].occupied = false;
          return;
        }
        m->entries[slot] = *ne;
        slot = next;
      }
    }
    slot = (slot + 1) & (m->cap - 1);
    dist++;
  }
}

void poly_map_clear(PolyMap *m) {
  memset(m->entries, 0, m->cap * sizeof(PolyMapEntry));
  m->len = 0;
}

void poly_map_foreach(PolyMap *m, PolyMapIterFn fn, void *userdata) {
  for (size_t i = 0; i < m->cap; i++) {
    if (m->entries[i].occupied)
      fn(m->entries[i].key, m->entries[i].value, userdata);
  }
}

static void map_grow(PolyMap *m) {
  size_t old_cap = m->cap;
  PolyMapEntry *old = m->entries;
  size_t new_cap = old_cap * 2;

  m->entries = calloc(new_cap, sizeof(PolyMapEntry));
  m->cap = new_cap;
  m->len = 0;

  for (size_t i = 0; i < old_cap; i++) {
    if (old[i].occupied) {
      /* re-insert with identity equality (key pointer comparison) */
      size_t slot = old[i].hash & (new_cap - 1);
      PolyMapEntry incoming = old[i];
      size_t dist = 0;
      for (;;) {
        PolyMapEntry *e = &m->entries[slot];
        if (!e->occupied) {
          *e = incoming;
          m->len++;
          break;
        }
        size_t existing_dist = (slot + new_cap - (e->hash & (new_cap - 1))) & (new_cap - 1);
        if (existing_dist < dist) {
          PolyMapEntry tmp = *e;
          *e = incoming;
          incoming = tmp;
          dist = existing_dist;
        }
        slot = (slot + 1) & (new_cap - 1);
        dist++;
      }
    }
  }
  free(old);
}
