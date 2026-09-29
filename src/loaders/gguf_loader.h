#ifndef POLY_GGUF_LOADER_H
#define POLY_GGUF_LOADER_H

#include "../model.h"

#ifdef __cplusplus
extern "C" {
#endif

PolyModel *poly_gguf_load(
    const uint8_t *data,
    int64_t len,
    int max_batch,
    int max_seq_len,
    int cache_capacity,
    int prefill_chunk_size,
    PolyDevice device
);
/* Borrows the caller context; decoded storage is fresh for each import.
 * cache_capacity=0 preserves uncached loading. A positive capacity requests
 * generation entrypoints; prefill_chunk_size=0 then defaults to one token.
 * Types without cached generation reject the request. */
PolyModel *poly_gguf_load_into(
    PolyCtx *ctx,
    const uint8_t *data,
    int64_t len,
    int max_batch,
    int max_seq_len,
    int cache_capacity,
    int prefill_chunk_size,
    PolyDevice device
);

#ifdef __cplusplus
}
#endif
#endif
