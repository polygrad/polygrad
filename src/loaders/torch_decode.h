#ifndef POLY_TORCH_DECODE_H
#define POLY_TORCH_DECODE_H
#include "decoded.h"
#include <stddef.h>

/* Restricted ZIP state dictionaries, never executable pickle objects. Contiguous
 * payloads borrow bytes. Packed views live in their name allocation, so callers
 * free each name and the table, just like the safetensors HF decode path. */
#if defined(__GNUC__)
__attribute__((visibility("hidden")))
#endif
int poly_torch_decode(const uint8_t *data, size_t len, PolyDecodedTensor **out, int *count);
#endif
