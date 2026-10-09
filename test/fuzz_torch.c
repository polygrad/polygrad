#include "../src/loaders/torch_decode.h"
#include <stdlib.h>
#include <string.h>

size_t LLVMFuzzerMutate(uint8_t *data, size_t size, size_t max_size);
static unsigned read32(const uint8_t *p) {
  return (unsigned)p[0] | (unsigned)p[1] << 8 | (unsigned)p[2] << 16 | (unsigned)p[3] << 24;
}
static unsigned read16(const uint8_t *p) {
  return (unsigned)p[0] | (unsigned)p[1] << 8;
}
static void write32(uint8_t *p, unsigned n) {
  for (int i = 0; i < 4; i++)
    p[i] = (uint8_t)(n >> (8 * i));
}

/* Ordinary mutations almost always stop at the ZIP checksum. Half the runs
 * deliberately repair it after mutating an index, exercising the data grammar
 * as well as the container parser. This is test-only, never a loader bypass. */
size_t LLVMFuzzerCustomMutator(uint8_t *data, size_t size, size_t max_size, unsigned seed) {
  if (!(seed & 1)) return LLVMFuzzerMutate(data, size, max_size);
  for (size_t i = 0; i + 46 <= size; i++) {
    if (read32(data + i) != 0x02014b50) continue;
    size_t nn = read16(data + i + 28), n = read32(data + i + 24), off = read32(data + i + 42);
    if (nn < 8 || nn > size - i - 46 || memcmp(data + i + 46 + nn - 8, "data.pkl", 8) ||
        off > size || size - off < 30 || read32(data + off) != 0x04034b50)
      continue;
    size_t begin = off + 30 + read16(data + off + 26) + read16(data + off + 28);
    if (begin > size || n > size - begin) continue;
    size_t used = LLVMFuzzerMutate(data + begin, n, n);
    memset(data + begin + used, 0, n - used);
    unsigned crc = ~0u;
    for (size_t j = 0; j < n; j++) {
      crc ^= data[begin + j];
      for (int b = 0; b < 8; b++)
        crc = (crc >> 1) ^ ((crc & 1) ? 0xedb88320u : 0);
    }
    write32(data + i + 16, ~crc);
    write32(data + off + 14, ~crc);
    return size;
  }
  return LLVMFuzzerMutate(data, size, max_size);
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  PolyDecodedTensor *tensors = NULL;
  int count = 0;
  if (!poly_torch_decode(data, size, &tensors, &count)) {
    for (int i = 0; i < count; i++)
      free(tensors[i].name);
    free(tensors);
  }
  return 0;
}
