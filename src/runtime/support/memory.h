/* Current Tinygrad runtime/support/memory.py TLSF allocator. */

#ifndef POLY_RUNTIME_SUPPORT_MEMORY_H
#define POLY_RUNTIME_SUPPORT_MEMORY_H

#include <stddef.h>

typedef struct PolyTLSFAllocator PolyTLSFAllocator;

PolyTLSFAllocator *poly_tlsf_allocator_new(size_t size, size_t block_size, int lv2_count);
void poly_tlsf_allocator_destroy(PolyTLSFAllocator *allocator);
size_t poly_tlsf_allocator_alloc(PolyTLSFAllocator *allocator, size_t size);
int poly_tlsf_allocator_free(PolyTLSFAllocator *allocator, size_t offset);

#endif
