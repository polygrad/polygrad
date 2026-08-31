/* Current Tinygrad runtime/support/memory.py TLSF allocator. */

#include "runtime/support/memory.h"

#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  size_t *starts;
  int count;
  int capacity;
} TLSFBucket;

typedef struct {
  size_t start;
  size_t size;
  size_t next;
  size_t previous;
  bool has_previous;
  bool free;
  bool alive;
} TLSFBlock;

struct PolyTLSFAllocator {
  size_t size;
  size_t block_size;
  int level2_bits;
  int level2_count;
  int level1_count;
  TLSFBucket *buckets;
  int *level1_entries;
  TLSFBlock *blocks;
  int block_count;
  int block_capacity;
};

static int bit_length_size(size_t value) {
  int bits = 0;
  while (value) {
    bits++;
    value >>= 1;
  }
  return bits;
}

static int bit_length_int(int value) {
  int bits = 0;
  unsigned int remaining = (unsigned int)value;
  while (remaining) {
    bits++;
    remaining >>= 1;
  }
  return bits;
}

static size_t power_of_two(int exponent) {
  if (exponent <= 0) return 1;
  int bits = (int)(sizeof(size_t) * CHAR_BIT);
  if (exponent >= bits) return (size_t)1 << (bits - 1);
  return (size_t)1 << exponent;
}

static int level1(PolyTLSFAllocator *allocator, size_t size) {
  (void)allocator;
  return bit_length_size(size);
}

static int level2(PolyTLSFAllocator *allocator, size_t size) {
  int bits = bit_length_size(size);
  if (bits <= 0) return 0;
  size_t base = power_of_two(bits - 1);
  size_t denominator = power_of_two(bits - allocator->level2_bits);
  int result = (int)((size - base) / denominator);
  if (result < 0) return 0;
  if (result >= allocator->level2_count) return allocator->level2_count - 1;
  return result;
}

static TLSFBucket *bucket_for(PolyTLSFAllocator *allocator, size_t size) {
  int l1 = level1(allocator, size), l2 = level2(allocator, size);
  if (l1 < 0 || l1 >= allocator->level1_count || l2 < 0 || l2 >= allocator->level2_count)
    return NULL;
  return &allocator->buckets[l1 * allocator->level2_count + l2];
}

static int bucket_append(TLSFBucket *bucket, size_t start) {
  if (bucket->count >= bucket->capacity) {
    int capacity = bucket->capacity ? bucket->capacity * 2 : 4;
    size_t *starts = realloc(bucket->starts, (size_t)capacity * sizeof(*starts));
    if (!starts) return -1;
    bucket->starts = starts;
    bucket->capacity = capacity;
  }
  bucket->starts[bucket->count++] = start;
  return 0;
}

static int bucket_remove(TLSFBucket *bucket, size_t start) {
  if (!bucket) return -1;
  for (int i = 0; i < bucket->count; i++) {
    if (bucket->starts[i] != start) continue;
    memmove(
        bucket->starts + i, bucket->starts + i + 1,
        (size_t)(bucket->count - i - 1) * sizeof(*bucket->starts)
    );
    bucket->count--;
    return 0;
  }
  return -1;
}

static int find_block(PolyTLSFAllocator *allocator, size_t start) {
  for (int i = 0; i < allocator->block_count; i++)
    if (allocator->blocks[i].alive && allocator->blocks[i].start == start) return i;
  return -1;
}

static int update_block(
    PolyTLSFAllocator *allocator,
    size_t start,
    size_t size,
    size_t previous,
    bool has_previous,
    bool free
) {
  int index = find_block(allocator, start);
  if (index < 0) {
    if (allocator->block_count >= allocator->block_capacity) {
      int capacity = allocator->block_capacity ? allocator->block_capacity * 2 : 16;
      TLSFBlock *blocks = realloc(allocator->blocks, (size_t)capacity * sizeof(*blocks));
      if (!blocks) return -1;
      allocator->blocks = blocks;
      allocator->block_capacity = capacity;
    }
    index = allocator->block_count++;
  }
  allocator->blocks[index] = (TLSFBlock){
      .start = start,
      .size = size,
      .next = start + size,
      .previous = previous,
      .has_previous = has_previous,
      .free = free,
      .alive = true,
  };
  return index;
}

static int insert_block(
    PolyTLSFAllocator *allocator,
    size_t start,
    size_t size,
    size_t previous,
    bool has_previous
) {
  int existing = find_block(allocator, start);
  if (!has_previous && existing >= 0) {
    has_previous = allocator->blocks[existing].has_previous;
    previous = allocator->blocks[existing].previous;
  }
  TLSFBucket *bucket = bucket_for(allocator, size);
  if (!bucket || bucket_append(bucket, start) != 0) return -1;
  int l1 = level1(allocator, size);
  allocator->level1_entries[l1]++;
  return update_block(allocator, start, size, previous, has_previous, true) >= 0 ? 0 : -1;
}

static int remove_block(
    PolyTLSFAllocator *allocator,
    size_t start,
    size_t size,
    size_t previous,
    bool has_previous
) {
  int existing = find_block(allocator, start);
  if (!has_previous && existing >= 0) {
    has_previous = allocator->blocks[existing].has_previous;
    previous = allocator->blocks[existing].previous;
  }
  TLSFBucket *bucket = bucket_for(allocator, size);
  if (!bucket || bucket_remove(bucket, start) != 0) return -1;
  allocator->level1_entries[level1(allocator, size)]--;
  return update_block(allocator, start, size, previous, has_previous, false) >= 0 ? 0 : -1;
}

static int split_block(PolyTLSFAllocator *allocator, size_t start, size_t size, size_t first_size) {
  if (!allocator || first_size == 0 || first_size >= size) return -1;
  int index = find_block(allocator, start);
  if (index < 0 || !allocator->blocks[index].free) return -1;
  size_t next = allocator->blocks[index].next;
  size_t previous = allocator->blocks[index].previous;
  bool has_previous = allocator->blocks[index].has_previous;
  if (remove_block(allocator, start, size, previous, has_previous) != 0 ||
      insert_block(allocator, start, first_size, previous, has_previous) != 0 ||
      insert_block(allocator, start + first_size, size - first_size, start, true) != 0)
    return -1;
  int next_index = find_block(allocator, next);
  if (next_index >= 0) {
    allocator->blocks[next_index].previous = start + first_size;
    allocator->blocks[next_index].has_previous = true;
  }
  return 0;
}

static int merge_right(PolyTLSFAllocator *allocator, size_t start) {
  int index = find_block(allocator, start);
  if (index < 0 || !allocator->blocks[index].free) return -1;
  size_t size = allocator->blocks[index].size;
  size_t previous = allocator->blocks[index].previous;
  bool has_previous = allocator->blocks[index].has_previous;
  size_t next = allocator->blocks[index].next;
  while (true) {
    int next_index = find_block(allocator, next);
    if (next_index < 0 || !allocator->blocks[next_index].free) break;
    size_t next_size = allocator->blocks[next_index].size;
    size_t following = allocator->blocks[next_index].next;
    if (remove_block(allocator, start, size, previous, has_previous) != 0 ||
        remove_block(allocator, next, next_size, start, true) != 0 ||
        insert_block(allocator, start, size + next_size, previous, has_previous) != 0)
      return -1;
    allocator->blocks[next_index].alive = false;
    size += next_size;
    next = following;
  }
  int next_index = find_block(allocator, next);
  if (next_index >= 0) {
    allocator->blocks[next_index].previous = start;
    allocator->blocks[next_index].has_previous = true;
  }
  return 0;
}

PolyTLSFAllocator *poly_tlsf_allocator_new(size_t size, size_t block_size, int lv2_count) {
  PolyTLSFAllocator *allocator = calloc(1, sizeof(*allocator));
  if (!allocator) return NULL;
  allocator->size = size;
  allocator->block_size = block_size ? block_size : 16;
  allocator->level2_bits = bit_length_int(lv2_count > 0 ? lv2_count : 16);
  allocator->level2_count = 1 << allocator->level2_bits;
  allocator->level1_count = bit_length_size(size) + 1;
  if (allocator->level1_count <= 0) allocator->level1_count = 1;
  allocator->buckets = calloc(
      (size_t)allocator->level1_count * (size_t)allocator->level2_count, sizeof(*allocator->buckets)
  );
  allocator->level1_entries =
      calloc((size_t)allocator->level1_count, sizeof(*allocator->level1_entries));
  if (!allocator->buckets || !allocator->level1_entries ||
      (size > 0 && insert_block(allocator, 0, size, 0, false) != 0)) {
    poly_tlsf_allocator_destroy(allocator);
    return NULL;
  }
  return allocator;
}

void poly_tlsf_allocator_destroy(PolyTLSFAllocator *allocator) {
  if (!allocator) return;
  if (allocator->buckets)
    for (int i = 0; i < allocator->level1_count * allocator->level2_count; i++)
      free(allocator->buckets[i].starts);
  free(allocator->buckets);
  free(allocator->level1_entries);
  free(allocator->blocks);
  free(allocator);
}

size_t poly_tlsf_allocator_alloc(PolyTLSFAllocator *allocator, size_t requested) {
  if (!allocator) return SIZE_MAX;
  if (requested < allocator->block_size) requested = allocator->block_size;
  size_t size = requested;
  int bits = bit_length_size(size);
  size_t bucket_size = power_of_two(bits - allocator->level2_bits);
  size = size % bucket_size ? size + bucket_size - size % bucket_size : size;
  int start_level1 = level1(allocator, size);
  int size_bits = bit_length_size(size);
  for (int l1 = start_level1; l1 < allocator->level1_count; l1++) {
    if (allocator->level1_entries[l1] == 0) continue;
    int l2_start = l1 == size_bits ? level2(allocator, size) : 0;
    for (int l2 = l2_start; l2 < allocator->level2_count; l2++) {
      TLSFBucket *bucket = &allocator->buckets[l1 * allocator->level2_count + l2];
      if (bucket->count <= 0) continue;
      size_t start = bucket->starts[0];
      int index = find_block(allocator, start);
      if (index < 0 || allocator->blocks[index].size < size) continue;
      size_t available = allocator->blocks[index].size;
      if (available > requested && split_block(allocator, start, available, requested) != 0)
        return SIZE_MAX;
      if (remove_block(allocator, start, requested, 0, false) != 0) return SIZE_MAX;
      return start;
    }
  }
  return SIZE_MAX;
}

int poly_tlsf_allocator_free(PolyTLSFAllocator *allocator, size_t offset) {
  if (!allocator) return -1;
  int index = find_block(allocator, offset);
  if (index < 0) return -1;
  size_t size = allocator->blocks[index].size;
  size_t previous = allocator->blocks[index].previous;
  bool has_previous = allocator->blocks[index].has_previous;
  if (insert_block(allocator, offset, size, previous, has_previous) != 0) return -1;
  index = find_block(allocator, offset);
  while (index >= 0 && allocator->blocks[index].has_previous) {
    int previous_index = find_block(allocator, allocator->blocks[index].previous);
    if (previous_index < 0 || !allocator->blocks[previous_index].free) break;
    offset = allocator->blocks[previous_index].start;
    index = previous_index;
  }
  return merge_right(allocator, offset);
}
