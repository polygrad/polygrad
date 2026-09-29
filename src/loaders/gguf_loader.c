/*
 * gguf_loader.c -- GGUF model loader (thin auto-dispatch wrapper)
 *
 * Decodes GGUF binary, looks up model by general.architecture,
 * dispatches to model-owned import function. Zero model-specific logic.
 */

#define _POSIX_C_SOURCE 200809L
#include "gguf_decode.h"
#include "gguf_loader.h"
#include "../models/registry.h"
#include "import_error.h"
#include "../model.h"
#include <stdio.h>

static PolyModel *gguf_load(
    PolyCtx *ctx,
    const uint8_t *data,
    int64_t len,
    int max_batch,
    int max_seq_len,
    int cache_capacity,
    int prefill_chunk_size,
    PolyDevice device
) {
  poly_import_error_clear();
  if (cache_capacity < 0 || prefill_chunk_size < 0 || (!cache_capacity && prefill_chunk_size) ||
      (cache_capacity && (max_batch > 1 || prefill_chunk_size > cache_capacity))) {
    poly_import_error_set(
        POLY_IMPORT_ERR_INVALID_ARGUMENT,
        "invalid cache_capacity/prefill_chunk_size or cache batch (must be 1)"
    );
    return NULL;
  }

  PolyGgufDecoded *gguf = NULL;
  if (poly_gguf_decode(data, len, &gguf) != 0 || !gguf) return NULL;

  const PolyModelType *desc = model_type_find(gguf->arch);
  if (!desc || !desc->from_gguf_decoded) {
    poly_import_error_set(
        POLY_IMPORT_ERR_UNSUPPORTED_MODEL,
        desc ? "%s does not support GGUF import" : "unsupported GGUF architecture '%s'",
        desc         ? desc->name
        : gguf->arch ? gguf->arch
                     : ""
    );
    poly_gguf_decoded_free(gguf);
    return NULL;
  }

  PolyGenericImportOpts opts = {
      .ctx = ctx,
      .max_batch = max_batch,
      .max_seq_len = max_seq_len,
      .cache_capacity = cache_capacity,
      .prefill_chunk_size = cache_capacity ? (prefill_chunk_size ? prefill_chunk_size : 1) : 0,
      .device = device,
  };
  PolyModel *inst = desc->from_gguf_decoded(gguf, &opts);

  poly_gguf_decoded_free(gguf);
  return inst;
}

PolyModel *poly_gguf_load(
    const uint8_t *data,
    int64_t len,
    int max_batch,
    int max_seq_len,
    int cache_capacity,
    int prefill_chunk_size,
    PolyDevice device
) {
  return gguf_load(
      NULL, data, len, max_batch, max_seq_len, cache_capacity, prefill_chunk_size, device
  );
}

PolyModel *poly_gguf_load_into(
    PolyCtx *ctx,
    const uint8_t *data,
    int64_t len,
    int max_batch,
    int max_seq_len,
    int cache_capacity,
    int prefill_chunk_size,
    PolyDevice device
) {
  return ctx ? gguf_load(
                   ctx, data, len, max_batch, max_seq_len, cache_capacity, prefill_chunk_size,
                   device
               )
             : NULL;
}
