/* utils.c -- generic C utilities, no polygrad dependencies */

#include "utils.h"

bool poly_ptr_eq(const void *a, const void *b) { return a == b; }

uint32_t poly_ptr_hash(const void *p) {
  uintptr_t x = (uintptr_t)p;
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33;
  return (uint32_t)x;
}
