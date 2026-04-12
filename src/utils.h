/* utils.h -- generic C utilities, no polygrad dependencies */

#ifndef POLY_UTILS_H
#define POLY_UTILS_H

#include <stdint.h>
#include <stdbool.h>

/* Pointer hashing (splitmix64 finalizer) */
bool poly_ptr_eq(const void *a, const void *b);
uint32_t poly_ptr_hash(const void *p);

#endif /* POLY_UTILS_H */
