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
    PolyDevice device
);
/* Borrows the caller context; decoded storage is fresh for each import. */
PolyModel *poly_gguf_load_into(
    PolyCtx *ctx,
    const uint8_t *data,
    int64_t len,
    int max_batch,
    int max_seq_len,
    PolyDevice device
);

#ifdef __cplusplus
}
#endif
#endif
